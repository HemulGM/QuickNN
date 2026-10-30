unit QuickNN.Models.ZImage;

{$C+} // enable assertions
{$H+} // longstrings
{$POINTERMATH on} // manipulate, inc, dec, cast pointers
{$T+} // typed pointer when @ is used
{$R+} // raise an error when trying to access arrays out of their bounds

interface

uses
  QuickNN.Common,
  QuickNN.SafeTensors,
  QuickNN.Models.Transformers,
  QuickNN.Models.Qwen3,
  QuickNN.Models.VAE;

type

  // TQNNZImage = TQNNCtx;
  { TQNNZImageHelper }

  TQNNZImage = record
    // tokenizer : TQNNTokenizer;
    Qwen3Encoder: TQwen3Encoder;
    VAE: TVAE;
    Transformer: TTransformerZI;

    (* Configuration *)
    MaxWidth: LongInt;
    MaxHeight: LongInt;
    DefaultSteps: LongInt;
    DefaultGuidance: QNNFloat;
    IsDistilled: Boolean; (* 1 = distilled (4-step), 0 = base (50-step CFG) *)
    TextDim: LongInt; (* Text embedding dimension (7680 for 4B, varies for 9B) *)
    IsNonCommercial: Boolean; (* 1 if model has non-commercial license (9B) *)
    NumHeads: LongInt; (* Transformer attention heads (24 for 4B, 32 for 9B) *)

    (* Z-Image specific config (from transformer/config.json) *)
    IsZimage: Boolean; (* 1 = Z-Image S3-DiT, 0 = Flux MMDiT *)
    ZiDim: LongInt; (* Hidden dim (3840) *)
    ZiNLayers: LongInt; (* Main transformer layers (30) *)
    ZiNRefiner: LongInt; (* Noise/context refiner layers (2) *)
    ZiCapFeatDim: LongInt; (* Caption feature dim (2560) *)
    ZiInChannels: LongInt; (* VAE latent channels (16) *)
    ZiPatchSize: LongInt; (* Spatial patch size (2) *)
    ZiRoPETheta: QNNFloat; (* RoPE theta (256.0) *)
    ZiAxesDims: array[0..2] of LongInt; (* RoPE axis dims [32, 48, 48] *)
    ZiLatentChannels: LongInt; (* Patchified latent channels (64 = 16*2*2) *)

    (* VAE config (read from vae/config.json) *)
    VAEZChannels: LongInt; (* Latent channels before patchify (32 Flux, 16 Z-Image) *)
    VAEScaling: QNNFloat; (* Scaling factor (0.3611 for Z-Image, 0 = use batch norm) *)
    VAEShift: QNNFloat; (* Shift factor (0.1159 for Z-Image, 0 = use batch norm) *)

    (* Model info *)
    ModelName, ModelVersion, ModelDir: string; (* For reloading text encoder if released *)

    (* Memory mode *)
    UseMmap: Boolean; (* Retain native checkpoint mappings. *)
    ResidentGPU: Boolean; (* Keep the complete generation chain on Vulkan. *)
    constructor Load(const AModelDir: string; const OnStatus: TPhaseCallBack = nil);
    procedure LoadVAE();
    function Generate(const Prompt: RawByteString; Params: TGenerateParams): TQNNImage;
    function GenerateFromEmbeddings(
        const TextEmb: TMemoryBlock;
        const TextSeq: LongInt;
        Params: TGenerateParams
    ): TQNNImage;
    function EncodeText(const Prompt: RawByteString; var OutSeqLen: LongInt): TMemoryBlock;
    function GenerateResident(
        const Prompt: RawByteString;
        const Embeddings: TMemoryBlock;
        const EmbeddingRows: LongInt;
        Params: TGenerateParams
    ): TQNNImage;
    procedure PrepareLegacy;
    procedure Free();
  end;

implementation

uses
  System.SysUtils,
  System.Math,
  System.JSON,
  System.Generics.Collections,
  System.IOUtils,
  QuickNN.Kernels,
  QuickNN.Sampling,
  QuickNN.Vulkan,
  QuickNN.Vulkan.Operations,
  QuickNN.Models.Resident
{$IFDEF MSWINDOWS}
  ,
  Winapi.Windows
{$ENDIF}
      ;

{ TQNNZImage }

constructor TQNNZImage.Load(const AModelDir: string; const OnStatus: TPhaseCallBack);
var
  JSON: TJSONValue;
  ModelArc: string;
  HiddenSize: LongInt;
  Arr: TJSONArray;
begin
  Self := Default(TQNNZImage);
  ResidentGPU := True;
  PhaseCallback := OnStatus;
  MaxWidth := QNN_VAE_MAX_DIM;
  MaxHeight := QNN_VAE_MAX_DIM;
  ModelVersion := '1.0';
  ModelDir := AModelDir;

  (* Autodetect model type from model_index.json.
   * Distilled model has "is_distilled": true, base model does not.
   * Z-Image has "_class_name": "ZImagePipeline". *)
  ModelArc := Format('%s/model_index.json', [ModelDir]);
  if Assigned(PhaseCallback) then
    PhaseCallback('Loading [' + ModelArc + ']', False);
  if FileExists(ModelArc) then
  begin
    JSON := TJSONValue.ParseJSONValue(TFile.ReadAllText(ModelArc));
    ModelArc := JSON.GetValue('_class_name', '');
    Assert(
        (Pos('ZImagePipeline', ModelArc) = 1) or (Pos('Z-Image', ModelArc) = 1),
        'ERROR : Model is not a Z-Image pipeline'
    );
    // is_zimage := (modelArc='ZImagePipeline') or (modelArc='Z-Image');
    IsDistilled := JSON.GetValue('is_distilled', False);
    JSON.Free;
  end
  else
  begin // assuming default FLUX.2-klein-4b
    IsDistilled := True;
  end;
  if Assigned(PhaseCallback) then
    PhaseCallback('Loading [' + ModelArc + ']', True);

  ModelArc := Format('%s/transformer/config.json', [ModelDir]);
  if Assigned(PhaseCallback) then
    PhaseCallback('Loading [' + ModelArc + ']', False);
  JSON := TJSONValue.ParseJSONValue(TFile.ReadAllText(ModelArc));
  if JSON.FindValue('cap_feat_dim') <> nil then
    IsZimage := True;

  if Assigned(PhaseCallback) then
    PhaseCallback('Loading [' + ModelArc + ']', True);

  (* dim (hidden size) *)
  ZiDim := JSON.GetValue<LongInt>('zi-dim', 3840);
  NumHeads := ZiDim div 128; (* head_dim = 128 *)

  (* n_layers *)
  ZiNLayers := JSON.GetValue<LongInt>('n_layers', 30);

  (* n_refiner_layers *)
  ZiNRefiner := JSON.GetValue<LongInt>('n_refiner_layers', 2);

  (* cap_feat_dim *)
  ZiCapFeatDim := JSON.GetValue<LongInt>('cap_feat_dim', 2560);

  (* in_channels *)
  ZiInChannels := JSON.GetValue<LongInt>('in_channels', 16);

  (* patch_size *)
  ZiPatchSize := JSON.GetValue<LongInt>('patch_size', 2);

  (* rope_theta *)
  ZiRoPETheta := JSON.GetValue<QNNFloat>('rope_theta', 256.0);

  (* axes_dims - parse JSON array [32, 48, 48] *)
  Arr := JSON.GetValue<TJSONArray>('axes_dims');

  ZiAxesDims[0] := Arr[0].AsType<LongInt>;
  ZiAxesDims[1] := Arr[1].AsType<LongInt>;
  ZiAxesDims[2] := Arr[2].AsType<LongInt>;
  // zi_axes_dims[0] := 32;
  // zi_axes_dims[1] := 48;
  // zi_axes_dims[2] := 48;
  JSON.Free;

  (* Derived values *)
  ZiLatentChannels := ZiInChannels * ZiPatchSize * ZiPatchSize;
  TextDim := ZiCapFeatDim; (* 2560 for Z-Image *)

  HiddenSize := ZiDim;
  IsNonCommercial := False; (* Z-Image is Apache 2.0 *)
  DefaultSteps := 9; (* 8 NFE = 9 scheduler steps *)
  DefaultGuidance := 0.0; (* No CFG for Z-Image-Turbo *)
  IsDistilled := True; (* Treat as distilled (no CFG) *)
  ModelArc := '6B'; (* Z-Image-Turbo is 6B *)
  if HiddenSize <> 3840 then
    if HiddenSize > 3840 then
      ModelArc := 'large'
    else
      ModelArc := 'small';
  ModelName := 'Z-Image-Turbo-' + ModelArc;

  // transformer files will load later while inferencing
  if not FileExists(Format('%s/transformer/config.json', [ModelDir])) then
    Assert(
        FileExists(Format('%s/transformer/diffusion_pytorch_model.safetensors', [ModelDir])),
        'ERROR : Transformer model not found (missing config.json and safetensors)'
    );
  RngSeed(GetTickCount64());
end;

procedure TQNNZImage.LoadVAE();
var
  ModelArc: string;
  JSON: TJSONValue;
  Sf: TSafeTensorFile;
begin
  ModelArc := Format('%s/vae/config.json', [ModelDir]);
  if Assigned(PhaseCallback) then
    PhaseCallback('Loading [' + ModelArc + ']', False);
  JSON := TJSONValue.ParseJSONValue(TFile.ReadAllText(ModelArc));
  VAEZChannels := JSON.GetValue<LongInt>('latent_channels', QNN_VAE_Z_CHANNELS);
  VAEScaling := JSON.GetValue('scaling_factor', 0.0);
  VAEShift := JSON.GetValue('shift_factor', 0.0);
  JSON.Free;
  if Assigned(PhaseCallback) then
    PhaseCallback('Loading [' + ModelArc + ']', True);

  ModelArc := Format('%s/vae/diffusion_pytorch_model.safetensors', [ModelDir]);
  if Assigned(PhaseCallback) then
    PhaseCallback('Loading [' + ModelArc + ']', False);
  Sf := TSafeTensorFile.Open(ModelArc);
  try
    VAE.Load([Sf], VAEZChannels, VAEScaling, VAEShift);
  finally
    Sf.Close();
  end;
  if Assigned(PhaseCallback) then
    PhaseCallback('Loading [' + ModelArc + ']', True);
end;

function TQNNZImage.Generate(const Prompt: RawByteString; Params: TGenerateParams): TQNNImage;
var
  TextSeq: LongInt;
  TextEmb: TMemoryBlock;
begin
  if ResidentGPU and QNNGetGPUStatus.Available then
    Exit(GenerateResident(Prompt, Default(TMemoryBlock), 0, Params));
  PrepareLegacy;
  Params.Guidance := 0; // always zero or no?
  (* Encode text (Z-Image mode: extraction mode 1, single layer) *)
  TextEmb := EncodeText(Prompt, TextSeq);

  (* Release text encoder to free memory before loading transformer *)
  Qwen3Encoder.Free;

  Result := GenerateFromEmbeddings(TextEmb, TextSeq, Params);
  TextEmb.Free

end;

function TQNNZImage.GenerateFromEmbeddings(
    const TextEmb: TMemoryBlock;
    const TextSeq: LongInt;
    Params: TGenerateParams
): TQNNImage;
var
  PreH, PreW, PostH, PostW, ImageSeqLen: LongInt;
  Z, Schedule, Denoised, Latent: TMemoryBlock;
begin
  if ResidentGPU and QNNGetGPUStatus.Available then
    Exit(GenerateResident('', TextEmb, TextSeq, Params));
  PrepareLegacy;

  (* Validate dimensions *)
  if Params.Width <= 0 then
    Params.Width := 1024; (* Z-Image default: 1024x1024 *)
  if Params.Height <= 0 then
    Params.Height := 1024;
  if Params.NumSteps <= 0 then
    Params.NumSteps := DefaultSteps;

  (* Ensure dimensions are divisible by 16 *)
  Params.Width := (Params.Width div 16) * 16;
  Params.Height := (Params.Height div 16) * 16;
  if Params.Width < 64 then
    Params.Width := 64;
  if Params.Height < 64 then
    Params.Height := 64;
  Assert(
      (Params.Width <= QNN_VAE_MAX_DIM) and (Params.Height <= QNN_VAE_MAX_DIM),
      'Image size must be maximum of 1792 X 1792'
  );

  (* Load Z-Image transformer on-demand (persistent across generations). *)
  if Assigned(PhaseCallback) then
    PhaseCallback('Loading Z-Image transformer', False);
  Transformer.Load(
      ModelDir,
      ZiDim,
      ZiDim div 128,
      ZiNLayers,
      ZiNRefiner,
      ZiCapFeatDim,
      ZiInChannels,
      ZiPatchSize,
      ZiRoPETheta,
      @ZiAxesDims[0],
      UseMmap
  );
  if Assigned(PhaseCallback) then
    PhaseCallback('Loading Z-Image transformer', True);

  (* Z-Image latent dimensions:
   * The transformer works at pre-patchification: [in_ch, H/8, W/8]
   * where in_ch=16, and patchification happens inside the transformer.
   * VAE decode expects post-patchification: [latent_ch, H/16, W/16]
   * where latent_ch = in_ch * ps * ps = 64. *)
  PreH := Params.Height div 8; (* H/8: pre-patchification spatial *)
  PreW := Params.Width div 8;
  PostH := PreH div ZiPatchSize; (* H/16: post-patchification spatial *)
  PostW := PreW div ZiPatchSize;
  ImageSeqLen := PostH * PostW;

  (* Initialize noise at pre-patchification dimensions: [in_channels, H/8, W/8] *)
  if Params.Seed < 0 then
    Params.Seed := Randseed;

  Z := InitNoise(1, Transformer.InChannels, PreH, PreW, Params.Seed);

  (* Get Z-Image schedule (default FlowMatch; linear/power if explicitly requested). *)
  if Params.Schedule = QNN_SCHEDULE_DEFAULT then
    Params.Schedule := QNN_SCHEDULE_FLOWMATCH;
  Schedule := SelectedSchedule(Params, ImageSeqLen);

  (* Sample using Z-Image Euler method.
   * The transformer takes [in_channels, pre_h, pre_w] and returns same shape. *)
  Denoised :=
      Transformer
          .SampleEuler(Z, 1, {in_channels,} PreH, PreW, ZiPatchSize, TextEmb, TextSeq, Schedule, Params.NumSteps);
  // denoised.printCompare(readTensor());
  Transformer.Free;
  Z.Free;
  Schedule.Free;
  // denoised.printCompare(readTensor());
  (* Patchify transformer output for VAE decode:
   * [1, in_channels, H/8, W/8] -> [1, latent_ch, H/16, W/16]
   * where latent_ch = in_channels * zi_patch_size * zi_patch_size = 64 *)
  // latent_ch := zi_in_channels * zi_patch_size * zi_patch_size;
  // latent := TMemoryBlock.Create([1, zi_in_channels, zi_patch_size ,zi_patch_size, post_h, post_w ], 'ZI_GENERATE_FROM_EMBEDDINGS_LATENT');
  // printCompare(latent_ch*post_h*post_w, denoised, readTensor());

  // transpose :
  // in : [batch, channels, outH, patch_size, outW, patch_size]
  // out: [batch, channels, patch_size ,patch_size, outH ,outW]
  // axis : 0, 1, 3, 5, 2, 4
  Denoised.Reshape([1, ZiInChannels, PostH, ZiPatchSize, PostW, ZiPatchSize]);
  QNNPermut(Latent, Denoised, [0, 1, 3, 5, 2, 4]);
  // QNNPatchify(latent, denoised, 1, zi_in_channels, pre_h, pre_w, zi_patch_size);
  // printCompare(latent_ch*post_h*post_w, latent, readTensor());

  Denoised.Free;
  (* Decode latent to image *)
  // printCompare(latent_ch* post_h* post_w, latent, readTensor());
  LoadVAE();

  if Assigned(PhaseCallback) then
    PhaseCallback('decoding image', False);
  Result := VAE.Decode(Latent, 1, PostH, PostW);
  if Assigned(PhaseCallback) then
    PhaseCallback('decoding image', True);
  VAE.Free;
  Latent.Free;
end;

function TQNNZImage.EncodeText(const Prompt: RawByteString; var OutSeqLen: LongInt): TMemoryBlock;
var
  NumRealTokens: LongInt;
begin
  if ModelDir <> '' then
  begin
    if Assigned(PhaseCallback) then
      PhaseCallback('Loading Qwen3 encoder', False);
    Qwen3Encoder.Load(ModelDir, UseMmap);
    OutSeqLen := QWEN3_MAX_SEQ_LEN;;
    if Assigned(PhaseCallback) then
      PhaseCallback('Loading Qwen3 encoder', True);
  end;
  Qwen3Encoder.Model.SetExtractionMode(IsZimage);
  if Assigned(PhaseCallback) then
    PhaseCallback('encoding text', False);
  Result := Qwen3Encoder.EncodeText(Prompt, NumRealTokens);
  if IsZimage then
    OutSeqLen := NumRealTokens
  else
    OutSeqLen := QWEN3_MAX_SEQ_LEN;
  if Assigned(PhaseCallback) then
    PhaseCallback('encoding text', True);
end;

procedure TQNNZImage.Free();
begin
  Qwen3Encoder.Free;
  Transformer.Free;
  VAE.Free;
end;

procedure TQNNZImage.PrepareLegacy;
begin
  if Qwen3Encoder.Model.ResidentOnly then
    Qwen3Encoder.Free;
  if Transformer.ResidentOnly then
    Transformer.Free;
  if VAE.DecConvInWeight.IsAssigned then
    VAE.Free;
end;

{$I QuickNN.Models.ZImage.GPU.inc}

end.
