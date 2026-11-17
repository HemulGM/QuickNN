unit QuickNN.Models.Flux1;

{ Native resident FLUX.1-dev pipeline. Weight files remain memory mapped in
  RAM; the shared Vulkan cache bounds their resident VRAM footprint. }

interface

uses
  System.SysUtils, QuickNN.Common, QuickNN.LoRA, QuickNN.ModelFiles;

type
  TFlux1Status = reference to procedure(const Text: string);

  TFlux1Params = record
    Width, Height, Steps: Integer;
    Seed: Int64;
    Guidance: Single;
    Prompt: string;
    LoRAs: TArray<TQNNLoRASpec>;
    class function Defaults: TFlux1Params; static;
  end;

  TFlux1Pipeline = class
  private
    FDirectory, FCheckpoint: string;
    FStatus: TFlux1Status;
    procedure Notify(const Message: string);
  public
    Components: TQNNModelFiles;
    constructor Create(const Directory: string; const Status: TFlux1Status = nil;
      const Checkpoint: string = '');
    function Generate(const Params: TFlux1Params): TQNNImage;
  end;

function Flux1Schedule(const ImageTokens, Steps: Integer): TArray<Single>;

implementation

uses
  System.Math, System.IOUtils, System.JSON, System.Diagnostics,
  QuickNN.Vulkan, QuickNN.Vulkan.Operations, QuickNN.Sampling,
  QuickNN.SafeTensors, QuickNN.Models.VAE, QuickNN.Models.Resident,
  QuickNN.Models.Flux1.Tokenizers, QuickNN.Models.Flux1.Text,
  QuickNN.Models.Flux1.Transformer;

class function TFlux1Params.Defaults: TFlux1Params;
begin
  Result := Default(TFlux1Params);
  Result.Width := 512; Result.Height := 512; Result.Steps := 20;
  Result.Seed := 42; Result.Guidance := 3.5;
  Result.Prompt := 'A red apple on a wooden table, soft daylight, realistic still life, no people.';
end;

function Flux1Schedule(const ImageTokens, Steps: Integer): TArray<Single>;
begin
  if (ImageTokens < 1) or (Steps < 1) or (Steps > 1000) then
    raise EArgumentOutOfRangeException.Create('FLUX.1 sampling schedule');
  SetLength(Result, Steps + 1);
  // Black Forest Labs' FLUX.1-dev dynamic time shift, sigma=1.
  var Mu := 0.5 + (1.15 - 0.5) * (ImageTokens - 256) / (4096 - 256);
  var Shift := Exp(Mu);
  Result[0] := 1;
  for var I := 1 to Steps - 1 do
  begin
    var T := 1 - I / Steps;
    Result[I] := Shift / (Shift + 1 / T - 1);
  end;
  Result[Steps] := 0;
end;

constructor TFlux1Pipeline.Create(const Directory: string; const Status: TFlux1Status;
  const Checkpoint: string);
begin
  inherited Create;
  FDirectory := ExcludeTrailingPathDelimiter(Directory);
  FCheckpoint := Checkpoint;
  if FCheckpoint = '' then FCheckpoint := TPath.Combine(FDirectory, 'flux1-dev.safetensors');
  FStatus := Status;
end;

procedure TFlux1Pipeline.Notify(const Message: string);
begin
  if Assigned(FStatus) then FStatus(Message);
end;

function TFlux1Pipeline.Generate(const Params: TFlux1Params): TQNNImage;
var
  Ctx: TQNNGPUContext;
  Work: TGPUWorkspace;
  Encoder: TFlux1TextEncoder;
  ClipTokenizer: TFlux1ClipTokenizer;
  T5Tokenizer: TFlux1T5Tokenizer;
  Transformer: TFlux1Transformer;
  VAE: TVAE;
  Text, Pooled, Latent, Prediction, Pixels: TQNNGPUTensor;
  Initial: TMemoryBlock;
  EosPosition, H, W: Integer;
  Steps: TArray<Single>;
  VAEFiles: TSafeTensorFiles;
  Config: TJSONValue;
begin
{$IFNDEF CPUX64}
  raise ENotSupportedException.Create('FLUX.1-dev generation requires Win64');
{$ENDIF}
  if (Params.Width < 64) or (Params.Height < 64) or (Params.Width > 2048) or (Params.Height > 2048)
    or (Params.Width mod 16 <> 0) or (Params.Height mod 16 <> 0)
    or (Int64(Params.Width div 16) * (Params.Height div 16) + 512 > 6144)
    or (Params.Steps < 1) or (Params.Steps > 1000) or IsNan(Params.Guidance)
    or IsInfinite(Params.Guidance) or (Params.Guidance < 0) or (Params.Guidance > 20) then
    raise EArgumentException.Create('Invalid FLUX.1-dev dimensions, steps or guidance');
  if not TFile.Exists(FCheckpoint) then raise EFileNotFoundException.Create('FLUX.1-dev checkpoint missing: ' + FCheckpoint);
  Ctx := nil; Work := nil; Encoder := nil; ClipTokenizer := nil; T5Tokenizer := nil; Transformer := nil;
  VAE := Default(TVAE); Initial := Default(TMemoryBlock);
  H := Params.Height div 16; W := Params.Width div 16;
  Steps := Flux1Schedule(H * W, Params.Steps);
  try
    Transformer := TFlux1Transformer.Create(FCheckpoint, TPath.Combine(FDirectory, 'transformer/config.json'), Params.LoRAs);
    Ctx := TQNNGPUContext.Create;
    Work := TGPUWorkspace.Create(Ctx);
    Notify('FLUX.1-dev: GPU CLIP-L text encoder');
    ClipTokenizer := TFlux1ClipTokenizer.Create(ModelComponentDirectory(Components.Tokenizer, TPath.Combine(FDirectory, 'tokenizer')));
    var ClipIds := ClipTokenizer.Encode(Params.Prompt, EosPosition);
    Encoder := TFlux1TextEncoder.Create(ModelComponentDirectory(Components.TextEncoder, TPath.Combine(FDirectory, 'text_encoder')), False);
    Pooled := Work.Own(Encoder.EncodeGPU(Ctx, ClipIds, EosPosition));
    Ctx.Flush; FreeAndNil(Encoder); FreeAndNil(ClipTokenizer); Ctx.TrimActivations;
    Notify('FLUX.1-dev: GPU T5-XXL text encoder');
    T5Tokenizer := TFlux1T5Tokenizer.Create(ModelComponentFile(Components.Tokenizer2, TPath.Combine(FDirectory, 'tokenizer_2/tokenizer.json')));
    var T5Ids := T5Tokenizer.Encode(Params.Prompt);
    Encoder := TFlux1TextEncoder.Create(ModelComponentDirectory(Components.TextEncoder2, TPath.Combine(FDirectory, 'text_encoder_2')), True);
    Text := Work.Own(Encoder.EncodeGPU(Ctx, T5Ids));
    Ctx.Flush; FreeAndNil(Encoder); FreeAndNil(T5Tokenizer); Ctx.TrimActivations;
    Notify('FLUX.1-dev: native BF16 transformer, RAM mappings and bounded VRAM cache');

    Notify(Format('FLUX.1-dev: %d active LoRA projection adapters', [Transformer.AdapterCount]));
    Initial := InitNoise(1, 64, H, W, Params.Seed);
    Latent := Work.Own(Ctx.Upload(Initial)); Initial.Free;
    for var I := 0 to Params.Steps - 1 do
    begin
      if Assigned(StepCallback) then StepCallback(I, Params.Steps);
      var Timer := TStopwatch.StartNew;
      Prediction := Work.Own(Transformer.ForwardGPU(Ctx, Latent, Text, Pooled, H, W, Length(T5Ids), Steps[I], Params.Guidance));
      Work.Add(Latent, Prediction, Latent.Count, Steps[I + 1] - Steps[I]);
      Work.Release(Prediction); Ctx.CheckFinite(Latent); Ctx.Flush;
      Notify(Format('FLUX.1-dev GPU step %d/%d: %.2f s', [I + 1, Params.Steps, Timer.Elapsed.TotalSeconds]));
    end;
    Work.Release(Text); Work.Release(Pooled);
    FreeAndNil(Transformer); Ctx.TrimActivations;
    Notify('FLUX.1-dev: tiled GPU VAE decode');
    Config := TJSONObject.ParseJSONValue(TFile.ReadAllText(TPath.Combine(ModelComponentDirectory(Components.VAE, TPath.Combine(FDirectory, 'vae')), 'config.json'), TEncoding.UTF8));
    try
      if not (Config is TJSONObject) or (Config.GetValue<Integer>('latent_channels', 0) <> 16) then
        raise EArgumentException.Create('Expected FLUX.1 16-channel VAE');
      SetLength(VAEFiles, 1);
      VAEFiles[0] := TSafeTensorFile.Open(ModelComponentFile(Components.VAE, TPath.Combine(FDirectory, 'vae/diffusion_pytorch_model.safetensors')));
      VAE.UseMMAP := True;
      VAE.Load(VAEFiles, 16, Config.GetValue<Single>('scaling_factor'), Config.GetValue<Single>('shift_factor'));
    finally Config.Free end;
    Pixels := Work.Own(VAEDecodeGPU(Ctx, VAE, Latent, H, W));
    Result := GPUDownloadRGB(Ctx, Pixels, Params.Height, Params.Width);
    Notify('FLUX.1-dev complete: final image readback only');
  finally
    Encoder.Free; ClipTokenizer.Free; T5Tokenizer.Free;
    Transformer.Free;
    Work.Free; Ctx.Free; VAE.Free; Initial.Free;
    for var FileData in VAEFiles do FileData.Close;
  end;
end;

end.
