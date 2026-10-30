unit QuickNN.Models.Flux;

{$C+} // enable assertions
{$H+} // longstrings
{$POINTERMATH on} // manipulate, inc, dec, cast pointers
{$T+} // typed pointer when @ is used
{$R+} // raise an error when trying to access arrays out of their bounds

interface

uses
  System.SysUtils, System.Math, QuickNN.Common, QuickNN.Models.Transformers,
  QuickNN.Models.Qwen3, QuickNN.Models.VAE, QuickNN.SafeTensors, Winapi.Windows;

type

  // TQNNFlux = type TQNNCtx;

  { TQNNFlux }

  TQNNFlux = record
    // tokenizer : TQNNTokenizer;
    Qwen3Encoder: TQwen3Encoder;
    VAE: TVAE;
    Transformer: TTransformerFlux;

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
    // is_zimage          : boolean ;   (* 1 = Z-Image S3-DiT, 0 = Flux MMDiT *)
    // zi_dim             : longint   ; (* Hidden dim (3840) *)
    // zi_n_layers        : longint   ; (* Main transformer layers (30) *)
    // zi_n_refiner       : longint   ; (* Noise/context refiner layers (2) *)
    // zi_cap_feat_dim    : longint   ; (* Caption feature dim (2560) *)
    // zi_in_channels     : longint   ; (* VAE latent channels (16) *)
    // zi_patch_size      : longint   ; (* Spatial patch size (2) *)
    // zi_rope_theta      : QNNFloat  ; (* RoPE theta (256.0) *)
    // zi_axes_dims       : array[0..2] of longint   ;   (* RoPE axis dims [32, 48, 48] *)
    // zi_latent_channels : longint   ;(* Patchified latent channels (64 = 16*2*2) *)

    (* VAE config (read from vae/config.json) *)
    VAEZChannels: LongInt; (* Latent channels before patchify (32 Flux, 16 Z-Image) *)
    VAEScaling: QNNFloat; (* Scaling factor (0.3611 for Z-Image, 0 = use batch norm) *)
    VAEShift: QNNFloat; (* Shift factor (0.1159 for Z-Image, 0 = use batch norm) *)

    (* Model info *)
    ModelName, ModelVersion, ModelDir: string; (* For reloading text encoder if released *)

    (* Memory mode *)
    UseMmap: Boolean; (* Retain native checkpoint mappings. *)
    ResidentGPU: Boolean; (* Keep the complete generation chain on Vulkan. *)

    // progress

    constructor Load(const AModelDir: string; const OnStatus: TPhaseCallBack = nil);
    procedure LoadVAE(const VaeModelPath: string); overload;
    procedure LoadVAE(); overload;
    function GenerateLatent(const TextEmb: TMemoryBlock; const TextSeq, Height, Width, NumSteps: LongInt; const Seed: Int64; const ProgressCallback: TProgressCallback): TMemoryBlock;
    function Generate(const Prompt: RawByteString; var Params: TGenerateParams; const ProgressCallback: TProgressCallback = nil
): TQNNImage; overload;
    function Generate(const Prompt, NegPrompt: RawByteString; var Params: TGenerateParams; const ProgressCallback: TProgressCallback = nil
): TQNNImage; overload;
    function Generate(const Prompt, NegPrompt: RawByteString; var Params: TGenerateParams; const Image: TQNNImage; const ProgressCallback: TProgressCallback = nil
): TQNNImage; overload;
    function Generate(const Prompt: RawByteString; var Params: TGenerateParams; const Image: TQNNImage; const ProgressCallback: TProgressCallback = nil
): TQNNImage; overload;
    function Generate(const Prompt, NegPrompt: RawByteString; var Params: TGenerateParams; const Images: TArray<TQNNImage>; const ProgressCallback: TProgressCallback = nil
): TQNNImage; overload;
    // function generate(const prompt: rawbytestring; const images:TArray<TQNNImage>;params: TGenerateParams): TQNNImage; overload;
    function EncodeText(const Prompt: RawByteString; var OutSeqLen: LongInt): TMemoryBlock;
    function GenerateResident(const Prompt, NegativePrompt: RawByteString; var Params: TGenerateParams; const Images: TArray<TQNNImage>; const Progress: TProgressCallback = nil
): TQNNImage;
    procedure PrepareLegacy;
    procedure Free();
  end;

implementation

uses
  System.JSON, System.IOUtils, QuickNN.Sampling, QuickNN.Kernels, QuickNN.Vulkan,
  QuickNN.Vulkan.Operations, QuickNN.Models.Resident;

{ TQNNFlux }

constructor TQNNFlux.Load(const AModelDir: string; const OnStatus: TPhaseCallBack);
var
  JSON: TJSONValue;
  ModelArc: string;
  HiddenSize: LongInt;
begin
  Self := Default(TQNNFlux);
  ResidentGPU := True;
  UseMmap := True;
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
    Assert(Pos('flux', LowerCase(ModelArc)) = 1, 'ERROR : Model is not a Flux pipeline');
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
  NumHeads := JSON.GetValue<LongInt>('num_attention_heads', 24); // default 4B
  TextDim := JSON.GetValue<LongInt>('joint_attention_dim', 7680); // default 4B: 3 * 2560
  JSON.Free;
  HiddenSize := NumHeads * 128;

  VAEScaling := 0;
  VAEShift := 0;
  if Assigned(PhaseCallback) then
    PhaseCallback('Loading [' + ModelArc + ']', True);

  IsNonCommercial := HiddenSize > 3072; // shouldn't it the opposite? maans it's commercial if above 3072 ?
  if IsNonCommercial then
    ModelName := '9B'
  else
    ModelName := '4B';
  if IsDistilled then
  begin
    DefaultSteps := 4;
    DefaultGuidance := 1.0;
    ModelName := 'FLUX.2-klein-' + ModelName
  end
  else
  begin
    DefaultSteps := 50;
    DefaultGuidance := 4.0;
    ModelName := 'FLUX.2-klein-base-' + ModelName;
  end;

  // transformer files will load later while inferencing
  if not FileExists(Format('%s/transformer/config.json', [ModelDir])) then
    Assert(
      FileExists(Format('%s/transformer/diffusion_pytorch_model.safetensors', [ModelDir])),
      'ERROR : Transformer model not found (missing config.json and safetensors)'
    );
  RngSeed(GetTickCount64());
end;

procedure TQNNFlux.LoadVAE(const VaeModelPath: string);
var
  ModelArc: string;
  JSON: TJSONValue;
  Sf: TSafeTensorFile;
begin
  ModelArc := Format('%s/vae/config.json', [VaeModelPath]);
  if Assigned(PhaseCallback) then
    PhaseCallback('Loading [' + ModelArc + ']', False);
  JSON := TJSONValue.ParseJSONValue(TFile.ReadAllText(ModelArc));
  VAEZChannels := JSON.GetValue<LongInt>('latent_channels', QNN_VAE_Z_CHANNELS);
  VAEScaling := JSON.GetValue('scaling_factor', 0.0);
  VAEShift := JSON.GetValue('shift_factor', 0.0);
  JSON.Free;
  if Assigned(PhaseCallback) then
    PhaseCallback('Loading [' + ModelArc + ']', True);

  ModelArc := Format('%s/vae/diffusion_pytorch_model.safetensors', [VaeModelPath]);
  if Assigned(PhaseCallback) then
    PhaseCallback('Loading [' + ModelArc + ']', False);
  Sf := TSafeTensorFile.Open(ModelArc);
  VAE.UseMMAP := UseMmap;
  try
    VAE.Load([Sf], VAEZChannels, VAEScaling, VAEShift);
  finally
    Sf.Close();
  end;
  if Assigned(PhaseCallback) then
    PhaseCallback('Loading [' + ModelArc + ']', True);
end;

procedure TQNNFlux.LoadVAE();
var
  ModelArc: string;
  JSON: TJSONValue;
  Sf: TSafeTensorFile;
begin
  ModelArc := Format('%s/vae/config.json', [ModelDir]);
  if Assigned(PhaseCallback) then
    PhaseCallback('Loading [' + ModelArc + ']', False);
  JSON := TJSONValue.ParseJSONValue(TFile.ReadAllText(ModelArc));
  VAEZChannels := JSON.GetValue('latent_channels', QNN_VAE_Z_CHANNELS);
  VAEScaling := JSON.GetValue('scaling_factor', 0.0);
  VAEShift := JSON.GetValue('shift_factor', 0.0);
  JSON.Free;
  if Assigned(PhaseCallback) then
    PhaseCallback('Loading [' + ModelArc + ']', True);

  ModelArc := Format('%s/vae/diffusion_pytorch_model.safetensors', [ModelDir]);
  if Assigned(PhaseCallback) then
    PhaseCallback('Loading [' + ModelArc + ']', False);
  Sf := TSafeTensorFile.Open(ModelArc);
  VAE.UseMMAP := UseMmap;
  try
    VAE.Load([Sf], VAEZChannels, VAEScaling, VAEShift);
  finally
    Sf.Close();
  end;
  if Assigned(PhaseCallback) then
    PhaseCallback('Loading [' + ModelArc + ']', True);
end;

function TQNNFlux.GenerateLatent(const TextEmb: TMemoryBlock; const TextSeq, Height, Width, NumSteps: LongInt; const Seed: Int64; const ProgressCallback: TProgressCallback): TMemoryBlock;
var
  LatentH, LatentW, Channels: LongInt;
  Z, Schedule: TMemoryBlock;
begin
  (* Compute latent dimensions *)
  LatentH := Height div 16;
  LatentW := Width div 16;
  Channels := QNN_LATENT_CHANNELS;

  (* Initialize noise *)
  Z := InitNoise(1, Channels, LatentH, LatentW, Seed);

  (* Get schedule (4 steps for klein distilled) *)
  Schedule := ScheduleLinear(NumSteps);

  (* Sample (FLUX.2-klein is guidance-distilled, no CFG needed) *)
  Result :=
    Transformer.SampleEuler(Z, 1, Channels, LatentH, LatentW, TextEmb, TextSeq, Schedule, NumSteps, ProgressCallback);

  Z.Free;
  Schedule.Free;
end;

function TQNNFlux.Generate(const Prompt: RawByteString; var Params: TGenerateParams; const ProgressCallback: TProgressCallback): TQNNImage;
begin
  Result := Generate(Prompt, '', Params, ProgressCallback)
end;

// type TSingleArray= array of single;
//    TFloatArray = array[0..MaxLongint div 32] of single;
//    PFloatArray = ^TFloatArray;

function TQNNFlux.Generate(const Prompt, NegPrompt: RawByteString; var Params: TGenerateParams; const ProgressCallback: TProgressCallback): TQNNImage;
var
  TextEmb, TextEmbUncond: TMemoryBlock;
  LatentH, LatentW, TextSeq, ImageSeqLen, TextSeqUncond: LongInt;
  Z, Schedule, Latent: TMemoryBlock;
begin
  if ResidentGPU and QNNGetGPUStatus.Available then
    Exit(GenerateResident(Prompt, NegPrompt, Params, nil, ProgressCallback));
  PrepareLegacy;
  if Params.Width <= 0 then
    Params.Width := QNN_DEFAULT_WIDTH;
  if Params.Height <= 0 then
    Params.Height := QNN_DEFAULT_HEIGHT;
  if Params.NumSteps <= 0 then
    Params.NumSteps := DefaultSteps;
  if Params.Guidance <= 0 then
    Params.Guidance := DefaultGuidance;
  Params.Width := Max(16 * Params.Width div 16, 64);
  Params.Height := Max(16 * Params.Height div 16, 64);
  Assert(
    (Params.Width <= QNN_VAE_MAX_DIM) and (Params.Height <= QNN_VAE_MAX_DIM),
    'ERROR : Image maximum dimensions exceeded (1792x1792)!'
  );
  TextEmb := EncodeText(Prompt, TextSeq);

  if Params.Guidance = 1.0 then
  begin
    if not IsDistilled then
      TextEmbUncond := EncodeText('', TextSeqUncond);
  end
  else
    TextEmbUncond := EncodeText(NegPrompt, TextSeqUncond);
  Qwen3Encoder.Free;

  LatentH := Params.Height div 16;
  LatentW := Params.Width div 16;
  ImageSeqLen := LatentH * LatentW;

  if Params.Seed <= 0 then
    Params.Seed := GetTickCount64();
  if Params.Schedule = QNN_SCHEDULE_DEFAULT then
    Params.Schedule := QNN_SCHEDULE_SIGMOID;

  Z := InitNoise(1, QNN_LATENT_CHANNELS, LatentH, LatentW, Params.Seed);
  Schedule := SelectedSchedule(Params, ImageSeqLen);

  if not Transformer.IsLoaded() then
  begin
    if Assigned(PhaseCallback) then
      PhaseCallback('Loading FLUX.2 transformer', False);
    Transformer := TTransformerFlux.Load(ModelDir, UseMmap);
    if Assigned(PhaseCallback) then
      PhaseCallback('Loading FLUX.2 transformer', True);
  end;
  if (Params.Guidance = 1.0) and IsDistilled then
    Latent :=
      Transformer.SampleEuler(
      Z,
      1,
      QNN_LATENT_CHANNELS,
      LatentH,
      LatentW,
      TextEmb,
      TextSeq,
      Schedule,
      Params.NumSteps,
      ProgressCallback
    )
  else
    Latent :=
      Transformer.SampleEuler(
      Z,
      1,
      QNN_LATENT_CHANNELS,
      LatentH,
      LatentW,
      TextEmb,
      TextSeq,
      TextEmbUncond,
      TextSeqUncond,
      Params.Guidance,
      Schedule,
      Params.NumSteps,
      ProgressCallback
    );

  Z.Free;
  Schedule.Free;
  TextEmb.Free;
  TextEmbUncond.Free;
  Transformer.Free;

  LoadVAE();
  Result := VAE.Decode(Latent, 1, LatentH, LatentW);
  VAE.Free;

  Latent.Free;
  // todo continue fro here with sampeling
end;

const
  ATTENTION_MAX_BYTES = Int64(4) shl 30;

function AttentionBytes(const NumHeads, OutH, OutW: LongInt; const RefDims: PLongint; const NumRefs, TxtSeq: LongInt): NativeUInt;
var
  TotalSeq: NativeUInt;
  I: LongInt;
begin
  TotalSeq := (OutH div 16) * (OutW div 16);
  for I := 0 to NumRefs - 1 do
    TotalSeq := TotalSeq + NativeUInt((RefDims[I * 2] div 16) * (RefDims[I * 2 + 1] div 16));
  TotalSeq := TotalSeq + NativeUInt(TxtSeq);
  Result := NativeUInt(NumHeads) * TotalSeq * TotalSeq * SizeOf(QNNFloat)
end;

function FitRefsForAttention(const NumHeads, OutH, OutW: LongInt; const RefDims: PLongint; const NumRefs, TxtSeq: LongInt): Boolean;
var
  Best, I, H, W: LongInt;
  Tok, BestTok: NativeUInt;
begin
  if AttentionBytes(NumHeads, OutH, OutW, RefDims, NumRefs, TxtSeq) <= ATTENTION_MAX_BYTES then
    Exit(False);
  Result := False;
  while True do
  begin
    Best := -1;
    BestTok := 0;
    for I := 0 to NumRefs - 1 do
    begin
      Tok := (RefDims[I * 2] div 16) * (RefDims[I * 2 + 1] div 16);
      if Tok > BestTok then
      begin
        BestTok := Tok;
        Best := I
      end
    end;
    if (Best < 0) or (BestTok <= 1) then
      Break;
    H := Trunc(RefDims[Best * 2] * 0.9) div 16 * 16;
    W := Trunc(RefDims[Best * 2 + 1] * 0.9) div 16 * 16;
    if H < 16 then
      H := 16;
    if W < 16 then
      W := 16;
    if (H = RefDims[Best * 2]) and (W = RefDims[Best * 2 + 1]) then
      Break;
    RefDims[Best * 2] := H;
    RefDims[Best * 2 + 1] := W;
    Result := True;
    if AttentionBytes(NumHeads, OutH, OutW, RefDims, NumRefs, TxtSeq) <= ATTENTION_MAX_BYTES then
      Break;
  end;
end;

function TQNNFlux.Generate(const Prompt, NegPrompt: RawByteString; var Params: TGenerateParams; const Image: TQNNImage; const ProgressCallback: TProgressCallback): TQNNImage;
var
  Resized, ImgToUse: TQNNImage;
  Scale: QNNFloat;
  TextEmb, TextEmbUncond, ImgTensor, ImgLatent, Schedule, Latent, Z: TMemoryBlock;
  RefW, RefH, TextSeq, TextSeqUncond, LatentH, LatentW, NumSteps, OutLatW, ImageSeqLen, OutLatH, TOffset: LongInt;
  RefDims: array[0..1] of LongInt;
begin
  if ResidentGPU and QNNGetGPUStatus.Available then
    Exit(GenerateResident(Prompt, NegPrompt, Params, [Image], ProgressCallback));
  PrepareLegacy;
  if (Params.Width <= 0) then
    Params.Width := Image.Width;
  if Params.Height <= 0 then
    Params.Height := Image.Height;
  if (Params.Width > QNN_VAE_MAX_DIM) or (Params.Height > QNN_VAE_MAX_DIM) then
  begin
    Scale :=
      Single(QNN_VAE_MAX_DIM) / (QuickNN.Common.Ifthen(Params.Width > Params.Height, Params.Width, Params.Height));
    Params.Width := Trunc(Params.Width * Scale);
    Params.Height := Trunc(Params.Height * Scale)
  end;
  Params.Width := (Params.Width div 16) * 16;
  Params.Height := (Params.Height div 16) * 16;
  RefW := Params.Width;
  RefH := Params.Height;
  RefDims[0] := Params.Height;
  RefDims[1] := Params.Width;

  ImgToUse := Image;
  if FitRefsForAttention(NumHeads, Params.Height, Params.Width, @RefDims[0], 1, QNN_MAX_SEQ_LEN) then
  begin
    RefH := RefDims[0];
    RefW := RefDims[1];
  end;

  if (Image.Width <> RefW) or (Image.Height <> RefH) then
  begin
    Resized := Image.Resize(RefW, RefH);
    ImgToUse := Resized
  end;
  if Params.NumSteps <= 0 then
    Params.NumSteps := DefaultSteps;
  if Params.Guidance <= 0 then
    Params.Guidance := DefaultGuidance;
  TextEmb := EncodeText(Prompt, TextSeq);
  // text_emb_uncond := nil;
  // text_seq_uncond := 0;
  if Params.Guidance = 1.0 then
  begin
    if not IsDistilled then
      TextEmbUncond := EncodeText('', TextSeqUncond);
  end
  else
    TextEmbUncond := EncodeText(NegPrompt, TextSeqUncond);
  Qwen3Encoder.Free;

  if not Transformer.IsLoaded() then
  begin
    if Assigned(PhaseCallback) then
      PhaseCallback('Loading FLUX.2 transformer', False);
    Transformer := TTransformerFlux.Load(ModelDir, UseMmap);
    if Assigned(PhaseCallback) then
      PhaseCallback('Loading FLUX.2 transformer', True);
  end;

  if Assigned(PhaseCallback) then
    PhaseCallback('encoding reference image', False);
  ImgTensor := ImgToUse.AsMemoryBlock();
  Resized.Free;

  LoadVAE();
  // if vae.isLoaded() then
  ImgLatent := VAE.Encode(ImgTensor, 1, RefH, RefW, LatentH, LatentW);
  // else begin
  //    latent_h := ref_h div 16;
  //    latent_w := ref_w div 16;
  //    img_latent := TMemoryBloack.create(LATENT_CHANNELS, latent_h, latent_w])
  // end;
  if UseMmap then
    VAE.Free;
  ImgTensor.Free();
  if Assigned(PhaseCallback) then
    PhaseCallback('encoding reference image', True);

  NumSteps := Params.NumSteps;
  OutLatH := Params.Height div 16;
  OutLatW := Params.Width div 16;
  ImageSeqLen := OutLatH * OutLatW;
  Schedule := SelectedSchedule(Params, ImageSeqLen);
  if Params.Seed <= 0 then
    Params.Seed := GetTickCount64;

  Z := InitNoise(1, QNN_LATENT_CHANNELS, OutLatH, OutLatW, Params.Seed);
  TOffset := 10;
  // assert(not img_latent.is_Nan());
  // img_latent.printStat;
  if (Params.Guidance = 1.0) and IsDistilled then
    Latent :=
      Transformer.SampleEuler(
      Z,
      1,
      QNN_LATENT_CHANNELS,
      OutLatH,
      OutLatW,
      ImgLatent,
      LatentH,
      LatentW,
      TOffset,
      TextEmb,
      TextSeq,
      Schedule,
      NumSteps,
      ProgressCallback
    )
  else
    Latent :=
      Transformer.SampleEuler(
      Z,
      1,
      QNN_LATENT_CHANNELS,
      OutLatH,
      OutLatW,
      ImgLatent,
      LatentH,
      LatentW,
      TOffset,
      TextEmb,
      TextSeq,
      TextEmbUncond,
      TextSeqUncond,
      Params.Guidance,
      Schedule,
      NumSteps,
      ProgressCallback
    );
  Z.Free;
  ImgLatent.Free;
  Schedule.Free;
  TextEmb.Free;
  TextEmbUncond.Free;

  if not VAE.IsLoaded() then
    LoadVAE();
  if Assigned(PhaseCallback) then
    PhaseCallback('decoding image', False);
  Result := VAE.Decode(Latent, 1, OutLatH, OutLatW);
  if Assigned(PhaseCallback) then
    PhaseCallback('decoding image', True);
  Latent.Free;
end;

function TQNNFlux.Generate(const Prompt: RawByteString; var Params: TGenerateParams; const Image: TQNNImage; const ProgressCallback: TProgressCallback): TQNNImage;
begin
  Result := Generate(Prompt, '', Params, Image, ProgressCallback)
end;

function TQNNFlux.Generate(const Prompt, NegPrompt: RawByteString; var Params: TGenerateParams; const Images: TArray<TQNNImage>; const ProgressCallback: TProgressCallback): TQNNImage;
var
  I, TextSeq, TextSeqUncond, Rh, Rw, RefW, RefH, LatH, LatW, LatentH, LatentW, ImageSeqLen: LongInt;
  Scale: Single;
  TextEmb, TextEmbUncond, Tensor, Schedule, Z, Latent: TMemoryBlock;
  RefPixelDims: TArray<LongInt>;
  RefLatents: TArray<TImageRef>;
  RefData: TArray<TMemoryBlock>;
  Ref, ImgToUse, ResizedImg: TQNNImage;
begin
  if ResidentGPU and QNNGetGPUStatus.Available then
    Exit(GenerateResident(Prompt, NegPrompt, Params, Images, ProgressCallback));
  PrepareLegacy;
  Assert(Assigned(Images), 'ERROR : no images loaded.');

  (* Single reference - use optimized path *)
  if Length(Images) = 1 then
    Exit(Generate(Prompt, NegPrompt, Params, Images[0]));

  (* Use first reference dimensions if not specified *)
  if Params.Width <= 0 then
    Params.Width := Images[0].Width;
  if Params.Height <= 0 then
    Params.Height := Images[0].Height;

  (* Clamp to VAE max dimensions *)
  if (Params.Width > QNN_VAE_MAX_DIM) or (Params.Height > QNN_VAE_MAX_DIM) then
  begin
    Scale := QNN_VAE_MAX_DIM / QuickNN.Common.Ifthen(Params.Width > Params.Height, Params.Width, Params.Height);
    Params.Width := Trunc(Params.Width * Scale);
    Params.Height := Trunc(Params.Height * Scale);
  end;

  Params.Width := (Params.Width div 16) * 16;
  Params.Height := (Params.Height div 16) * 16;

  (* Resolve steps and guidance *)
  if Params.NumSteps <= 0 then
    Params.NumSteps := DefaultSteps;
  if Params.Guidance <= 0 then
    Params.Guidance := DefaultGuidance;

  (* Encode text *)
  TextEmb := EncodeText(Prompt, TextSeq);

  if Params.Guidance = 1.0 then
  begin
    if not IsDistilled then
      TextEmbUncond := EncodeText('', TextSeqUncond);
  end
  else
    TextEmbUncond := EncodeText(NegPrompt, TextSeqUncond);

  Qwen3Encoder.Free;

  if not Transformer.IsLoaded() then
  begin
    if Assigned(PhaseCallback) then
      PhaseCallback('Loading FLUX.2 transformer', False);
    Transformer := TTransformerFlux.Load(ModelDir, UseMmap);
    if Assigned(PhaseCallback) then
      PhaseCallback('Loading FLUX.2 transformer', True);
  end;

  (* Build reference pixel dimensions, clamped and rounded to 16. *)
  SetLength(RefPixelDims, Length(Images) * 2);

  for I := 0 to High(Images) do
  begin
    Rh := (Images[I].Height div 16) * 16;
    Rw := (Images[I].Width div 16) * 16;
    if Rh > QNN_VAE_MAX_DIM then
      Rh := QNN_VAE_MAX_DIM;
    if Rw > QNN_VAE_MAX_DIM then
      Rw := QNN_VAE_MAX_DIM;
    if Rh < 16 then
      Rh := 16;
    if Rw < 16 then
      Rw := 16;
    RefPixelDims[I * 2] := Rh;
    RefPixelDims[I * 2 + 1] := Rw;
  end;

  (* Shrink references if attention would exceed 4 GB. *)
  FitRefsForAttention(NumHeads, Params.Height, Params.Width, Pointer(RefPixelDims), Length(Images), QNN_MAX_SEQ_LEN);

  (* Encode all reference images *)
  SetLength(RefLatents, Length(Images));
  SetLength(RefData, Length(Images));

  LoadVAE();
  for I := 0 to High(Images) do
  begin
    Ref := Images[I];
    ImgToUse := Ref;

    RefH := RefPixelDims[I * 2];
    RefW := RefPixelDims[I * 2 + 1];

    (* Resize only if dimensions differ from original *)
    if (Ref.Width <> RefW) or (Ref.Height <> RefH) then
    begin
      ResizedImg.Free;
      ResizedImg := Ref.Resize(RefW, RefH);
      ImgToUse := ResizedImg;
    end;

    (* Encode to latent at reference's own size *)
    if Assigned(PhaseCallback) then
      PhaseCallback('encoding image ' + IntToStr(I + 1) + '/' + IntToStr(Length(Images)), False);
    Tensor := ImgToUse.AsMemoryBlock();
    RefData[I] := VAE.Encode(Tensor, 1, ImgToUse.Height, ImgToUse.Width, LatH, LatW);
    Tensor.Free;
    if Assigned(PhaseCallback) then
      PhaseCallback('encoding image done', True);

    RefLatents[I].Latent := RefData[I];
    RefLatents[I].H := LatH;
    RefLatents[I].W := LatW;
    RefLatents[I].TOffset := 10 * (I + 1); (* 10, 20, 30, ... *)
  end;

  (* Free resized images (latents are now encoded) *)

  LatentH := Params.Height div 16;
  LatentW := Params.Width div 16;
  ImageSeqLen := LatentH * LatentW;

  Schedule := SelectedSchedule(&Params, ImageSeqLen);
  if Params.Seed <= 0 then
    Params.Seed := GetTickCount64;
  Z := InitNoise(1, QNN_LATENT_CHANNELS, LatentH, LatentW, Params.Seed);

  (* Sample with multi-reference conditioning *)
  if (Params.Guidance = 1.0) and IsDistilled then
  begin
    Latent :=
      Transformer.SampleEuler(
      Z,
      1,
      QNN_LATENT_CHANNELS,
      LatentH,
      LatentW,
      RefLatents,
      TextEmb,
      TextSeq,
      Schedule,
      Params.NumSteps,
      ProgressCallback
    );
  end
  else
  begin
    Latent :=
      Transformer.SampleEuler(
      Z,
      1,
      QNN_LATENT_CHANNELS,
      LatentH,
      LatentW,
      RefLatents,
      TextEmb,
      TextSeq,
      TextEmbUncond,
      TextSeqUncond,
      Params.Guidance,
      Schedule,
      Params.NumSteps,
      ProgressCallback
    );
  end;

  (* Cleanup *)
  Z.Free;
  for I := 0 to High(RefData) do
    RefData[I].Free();

  Schedule.Free;
  TextEmb.Free;
  TextEmbUncond.Free;

  (* Decode *)
  if Assigned(PhaseCallback) then
    PhaseCallback('decoding image', False);
  Result := VAE.Decode(Latent, 1, LatentH, LatentW);
  if Assigned(PhaseCallback) then
    PhaseCallback('decoding image', True);

  Latent.Free;
end;

// function TQNNFlux.generate(const prompt: rawbytestring; const images: TArray<TQNNImage>; params: TGenerateParams): TQNNImage;
// begin
//
// end;

function TQNNFlux.EncodeText(const Prompt: RawByteString; var OutSeqLen: LongInt): TMemoryBlock;
var
  NumRealTokens: LongInt;
begin
  if ModelDir <> '' then
  begin
    if Assigned(PhaseCallback) then
      PhaseCallback('Loading Qwen3 encoder', False);
    Qwen3Encoder.Load(ModelDir, UseMmap);
    OutSeqLen := QWEN3_MAX_SEQ_LEN;
    ;
    if Assigned(PhaseCallback) then
      PhaseCallback('Loading Qwen3 encoder', True);
  end;
  if Assigned(PhaseCallback) then
    PhaseCallback('encoding text', False);
  Result := Qwen3Encoder.EncodeText(Prompt, NumRealTokens);
  OutSeqLen := QWEN3_MAX_SEQ_LEN;
  if Assigned(PhaseCallback) then
    PhaseCallback('encoding text', True);
end;

procedure TQNNFlux.Free();
begin
  Qwen3Encoder.Free;
  Transformer.Free;
  VAE.Free;
end;

procedure TQNNFlux.PrepareLegacy;
begin
  if Qwen3Encoder.Model.ResidentOnly then
    Qwen3Encoder.Free;
  if Transformer.ResidentOnly then
    Transformer.Free;
  if VAE.DecConvInWeight.IsAssigned then
    VAE.Free;
end;

{$I QuickNN.Models.Flux.GPU.inc}

end.

