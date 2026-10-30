unit QuickNN.Models.Qwen21;

{ SPDX-License-Identifier: Apache-2.0
  Native Pascal implementation; architecture and tensor conventions follow
  Qwen-Image 2.1 / Diffusers and ComfyUI / comfy-kitchen (Apache-2.0).
  Reference authors: Qwen-Image Team, Hugging Face Team and Comfy Org contributors.
  https://github.com/huggingface/diffusers/tree/main/src/diffusers/models
  https://github.com/Comfy-Org/ComfyUI/tree/master/comfy/ldm/qwen_image21
  https://github.com/Comfy-Org/comfy-kitchen/blob/main/comfy_kitchen/tensor/int8_utils.py
  See LICENSE.QwenImage21.txt in the repository root. }
{$POINTERMATH ON}
interface

uses
  System.SysUtils, QuickNN.Common, QuickNN.Models.Qwen21.Math, QuickNN.LoRA,
  QuickNN.Models.Qwen21.Conditioning;

type
  TQwen21Params = record
    Width, Height, Steps: Integer;
    Seed: Int64;
    Guidance: Single;
    Prompt, NegativePrompt: string;
    LoRAs: TArray<TQNNLoRASpec>;
    ReferenceImages: TArray<string>;
    ReferenceSize: Integer;
    ResidentGPU: Boolean;
    class function Defaults: TQwen21Params; static;
  end;

  TQwen21Pipeline = class
  private
    type
      // One validated input set is shared by both execution backends.
      TGenerationInputs = record
        Suffix, TextFile, DiffusionFile, VAEFile: string;
        LatentHeight, LatentWidth: Integer;
        Sigmas: TArray<Single>;
        LoRAs: TArray<TQNNLoRASpec>;
        Images: TQwen21ImageConditions;
      end;
  private
    FModels, FAssets: string;
    FInt8: Boolean;
    FStatus: TQwen21Status;
    procedure Notify(const Message: string);
    procedure Prepare(const Params: TQwen21Params; out Inputs: TGenerationInputs);
    function GenerateLegacy(const Params: TQwen21Params; const Inputs: TGenerationInputs): TQNNImage;
    function GenerateResident(const Params: TQwen21Params; const Inputs: TGenerationInputs): TQNNImage;
  public
    constructor Create(const ModelRoot, AssetsRoot: string; const UseInt8: Boolean = True; const Status: TQwen21Status = nil
);
    function Generate(const Params: TQwen21Params): TQNNImage;
  end;

implementation

uses
  System.Math, System.IOUtils, System.Diagnostics, QuickNN.Quantization,
  QuickNN.Models.Qwen21.Tokenizer, QuickNN.Models.Qwen21.Text,
  QuickNN.Models.Qwen21.Transformer, QuickNN.Models.Qwen21.VAE, QuickNN.Sampling,
  QuickNN.Vulkan, QuickNN.Models.Qwen21.Vision;

class function TQwen21Params.Defaults: TQwen21Params;
begin
  Result := Default(TQwen21Params);
  Result.Width := 512;
  Result.Height := 512;
  Result.Steps := 40;
  Result.Seed := 42;
  Result.Guidance := 1;
  Result.ReferenceSize := 512;
  Result.ResidentGPU := True;
  Result.Prompt := 'A red apple on a wooden table, soft daylight, realistic photograph.';
end;

constructor TQwen21Pipeline.Create(const ModelRoot, AssetsRoot: string; const UseInt8: Boolean; const Status: TQwen21Status);
begin
  inherited Create;
  FModels := ExcludeTrailingPathDelimiter(ModelRoot);
  FAssets := ExcludeTrailingPathDelimiter(AssetsRoot);
  FInt8 := UseInt8;
  FStatus := Status
end;

procedure TQwen21Pipeline.Notify(const Message: string);
begin
  if Assigned(FStatus) then
    FStatus(Message);
end;

procedure TQwen21Pipeline.Prepare(const Params: TQwen21Params; out Inputs: TGenerationInputs);
var
  MinimumJointRows: Int64;

  procedure RequireFile(const FileName: string);
  begin
    if not TFile.Exists(FileName) then
      raise EFileNotFoundException.Create('Required model/asset missing: ' + FileName);
  end;

begin
  Inputs := Default(TGenerationInputs);
{$IFNDEF CPUX64}
  raise ENotSupportedException.Create('Qwen-Image 2.1 generation requires a Win64 build');
{$ENDIF}
  if (Params.Width < 64)
    or (Params.Height < 64)
    or (Params.Width > 1024)
    or (Params.Height > 1024)
    or (Params.Width mod 32 <> 0)
    or (Params.Height mod 32 <> 0) then
    raise EArgumentException.Create('Width/height must be multiples of 32 between 64 and 1024');
  if (Params.Steps < 1)
    or (Params.Steps > 1000)
    or IsNan(Params.Guidance)
    or IsInfinite(Params.Guidance)
    or (Params.Guidance < 1)
    or (Params.Guidance > 20) then
    raise EArgumentException.Create('Invalid steps or CFG');
  if (Length(Params.ReferenceImages) > 0)
    and ((Params.ReferenceSize < 64) or (Params.ReferenceSize > 1024) or (Params.ReferenceSize mod 32 <> 0)) then
    raise EArgumentException.Create('Reference size must be 64..1024 divisible by 32');

  if FInt8 then
    Inputs.Suffix := 'int8_convrot'
  else
    Inputs.Suffix := 'bf16';
  Inputs.TextFile := TPath.Combine(FModels, 'text_encoders/qwen3vl_8b_' + Inputs.Suffix + '.safetensors');
  Inputs.DiffusionFile := TPath.Combine(FModels, 'diffusion_models/qwen_image_2.1_' + Inputs.Suffix + '.safetensors');
  Inputs.VAEFile := TPath.Combine(FModels, 'vae/qwen_image_2.1_vae_bf16.safetensors');
  RequireFile(Inputs.TextFile);
  RequireFile(Inputs.DiffusionFile);
  RequireFile(Inputs.VAEFile);
  for var FileName in TArray<string>.Create('tokenizer.json', 'encoder_config.json', 'vae_config.json', 'scheduler_config.json') do
    RequireFile(TPath.Combine(FAssets, FileName));

  Inputs.LoRAs := Copy(Params.LoRAs);
  for var I := 0 to High(Inputs.LoRAs) do
  begin
    if not TFile.Exists(Inputs.LoRAs[I].FileName) and not TPath.IsPathRooted(Inputs.LoRAs[I].FileName) then
      Inputs.LoRAs[I].FileName := TPath.Combine(TPath.Combine(FModels, 'loras'), Inputs.LoRAs[I].FileName);
    RequireFile(Inputs.LoRAs[I].FileName);
  end;

  Inputs.LatentHeight := Params.Height div 16;
  Inputs.LatentWidth := Params.Width div 16;
  Inputs.Sigmas :=
    Q21Schedule(
    Params.Steps,
    Inputs.LatentHeight * Inputs.LatentWidth,
    TPath.Combine(FAssets, 'scheduler_config.json')
  );
  SetLength(Inputs.Images, Length(Params.ReferenceImages));
  MinimumJointRows := Int64(Inputs.LatentHeight) * Inputs.LatentWidth;
  for var I := 0 to High(Inputs.Images) do
  begin
    RequireFile(Params.ReferenceImages[I]);
    Inputs.Images[I].Pixels := Q21LoadReference(Params.ReferenceImages[I], Params.ReferenceSize);
    Inc(MinimumJointRows, Int64(Inputs.Images[I].Pixels.Width div 16) * (Inputs.Images[I].Pixels.Height div 16));
    if MinimumJointRows >= 6144 then
      raise EArgumentException.Create('Reference/target images exceed the 6144-token joint budget; reduce image/reference sizes');
  end;
end;

function TQwen21Pipeline.Generate(const Params: TQwen21Params): TQNNImage;
var
  Inputs: TGenerationInputs;
begin
  try
    Prepare(Params, Inputs);
    if Params.ResidentGPU then
      Result := GenerateResident(Params, Inputs)
    else
      Result := GenerateLegacy(Params, Inputs);
  finally
    // Includes partially loaded references when preparation or inference fails.
    for var I := 0 to High(Inputs.Images) do
      Inputs.Images[I].Free;
  end;
end;

function TQwen21Pipeline.GenerateLegacy(const Params: TQwen21Params; const Inputs: TGenerationInputs): TQNNImage;
var
  Images: TQwen21ImageConditions;
  Vision: TQwen21Vision;
  PositiveSlots, NegativeSlots: TArray<Integer>;
  Adapters: TQNNLoRASet;
  Specs: TArray<TQNNLoRASpec>;
  Tokenizer: TQwen21Tokenizer;
  Encoder: TQwen21TextEncoder;
  Transformer: TQwen21Transformer;
  VAE: TQwen21VAE;
  Text, Negative, Latents, Noise, NegNoise, Initial: TMemoryBlock;
  Sigmas: TArray<Single>;
  Lh, Lw: Integer;
  Timer: TStopwatch;
  UseCFG: Boolean;
  Delta: Single;
begin
  Images := Inputs.Images;
  Specs := Inputs.LoRAs;
  Sigmas := Inputs.Sigmas;
  Lh := Inputs.LatentHeight;
  Lw := Inputs.LatentWidth;
  UseCFG := Params.Guidance > 1;
  Adapters := nil;
  Transformer := nil;
  try
    Adapters := TQNNLoRASet.Create(Specs);
    Transformer := TQwen21Transformer.Create(Inputs.DiffusionFile);
    Transformer.AttachLoRAs(Adapters);
    if Length(Specs) > 0 then
      Notify(
        Format(
          'LoRA stack: %d files, %d modules, %d enabled',
          [Length(Specs), Adapters.ModuleCount, Adapters.EnabledModuleCount]
        )
      );
    if Length(Images) > 0 then
    begin
      Notify('Encoding reference images with the RGBA VAE');
      VAE := TQwen21VAE.Create(Inputs.VAEFile, TPath.Combine(FAssets, 'vae_config.json'), FStatus);
      try
        for var I := 0 to High(Images) do
        begin
          Notify(Format('Reference %d: %dx%d', [I + 1, Images[I].Pixels.Width, Images[I].Pixels.Height]));
          Images[I].Latents := VAE.Encode(Images[I].Pixels);
        end;
      finally
        VAE.Free
      end;
      Notify('Encoding reference images with Qwen3-VL vision/DeepStack');
      Vision := TQwen21Vision.Create(Inputs.TextFile, FStatus);
      try
        for var I := 0 to High(Images) do
          Vision.Encode(Images[I])
      finally
        Vision.Free
      end;
    end;
    Notify('Loading Qwen3-VL-8B ' + Inputs.Suffix);
    Tokenizer := TQwen21Tokenizer.Create(TPath.Combine(FAssets, 'tokenizer.json'));
    try
      Encoder := TQwen21TextEncoder.Create(Inputs.TextFile, TPath.Combine(FAssets, 'encoder_config.json'), FStatus);
      try
        Text := Encoder.EncodePrompt(Tokenizer, Params.Prompt, Images, PositiveSlots);
        if UseCFG then
          Negative := Encoder.EncodePrompt(Tokenizer, Params.NegativePrompt, Images, NegativeSlots);
      finally
        Encoder.Free
      end;
    finally
      Tokenizer.Free
    end;
    Notify('Loading Qwen-Image 2.1 ' + Inputs.Suffix);
    try
      Transformer.SetConditioning(Text, Text.Count div 4096, Images, PositiveSlots);
      Initial := InitNoise(1, 64, Lh, Lw, Params.Seed);
      Latents := TMemoryBlock.Create([Lh * Lw, 64], 'qwen21-sampled-latents');
      for var P := 0 to Lh * Lw - 1 do
        for var C := 0 to 63 do
          System.PSingle(Latents)[NativeInt(P) * 64 + C] := System.PSingle(Initial)[NativeInt(C) * Lh * Lw + P];
      Initial.Free;
      for var Step := 0 to Params.Steps - 1 do
      begin
        Timer := TStopwatch.StartNew;
        if UseCFG then
          Transformer.SetConditioning(Text, Text.Count div 4096, Images, PositiveSlots);
        Noise := Transformer.forward(Latents, Lh, Lw, Sigmas[Step]);
        if UseCFG then
        begin
          Transformer.SetConditioning(Negative, Negative.Count div 4096, Images, NegativeSlots);
          NegNoise := Transformer.forward(Latents, Lh, Lw, Sigmas[Step]);
          for var I: NativeInt := 0 to Noise.Count - 1 do
            System.PSingle(Noise)[I] :=
              System.PSingle(NegNoise)[I]
              + Params.Guidance * (System.PSingle(Noise)[I] - System.PSingle(NegNoise)[I]);
          NegNoise.Free;
        end;
        Delta := Sigmas[Step + 1] - Sigmas[Step];
        for var I: NativeInt := 0 to Latents.Count - 1 do
          System.PSingle(Latents)[I] := System.PSingle(Latents)[I] + Delta * System.PSingle(Noise)[I];
        Noise.Free;
        QNNCheckFinite(Latents, Latents.Count, 'Euler sampler');
        Notify(
          Format(
            'Step %d/%d, sigma %.6f -> %.6f, %.2f s',
            [Step + 1, Params.Steps, Sigmas[Step], Sigmas[Step + 1], Timer.Elapsed.TotalSeconds]
          )
        );
      end;
    finally
      FreeAndNil(Transformer)
    end;
    Text.Free;
    Negative.Free;
    Notify('Decoding RGBA with Qwen-Image 2.1 VAE');
    VAE := TQwen21VAE.Create(Inputs.VAEFile, TPath.Combine(FAssets, 'vae_config.json'), FStatus);
    try
      Result := VAE.Decode(Latents, Lh, Lw)
    finally
      VAE.Free
    end;
    Notify('Generation complete');
  finally
    Transformer.Free;
    Adapters.Free;
    Text.Free;
    Negative.Free;
    Latents.Free;
    Noise.Free;
    NegNoise.Free;
    Initial.Free
  end;
end;
{$I QuickNN.Models.Qwen21.Pipeline.GPU.inc}

end.

