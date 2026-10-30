function TQwen21Pipeline.GenerateResident(const Params: TQwen21Params; const Inputs: TGenerationInputs): TQNNImage;
var
  Ctx: TQNNGPUContext;
  Images: TQwen21ImageConditions;
  Resident: TQwen21GPUImageConditions;
  VAE: TQwen21VAE;
  Vision: TQwen21Vision;
  Tokenizer: TQwen21Tokenizer;
  Encoder: TQwen21TextEncoder;
  Transformer: TQwen21Transformer;
  Adapters: TQNNLoRASet;
  Positive, Negative: TQwen21GPUCondition;
  Work: TQwen21GPUWork;
  PositiveSlots, NegativeSlots: TArray<Integer>;
  Text, NegativeText, Latents, Noise, NegativeNoise, Pixels, InitialGPU: TQNNGPUTensor;
  Initial: TMemoryBlock;
  Lh, Lw, LatentCount: Integer;
  Timer: TStopwatch;
begin
  Ctx := nil;
  VAE := nil;
  Vision := nil;
  Tokenizer := nil;
  Encoder := nil;
  Transformer := nil;
  Adapters := nil;
  Text := nil;
  NegativeText := nil;
  NegativeNoise := nil;
  InitialGPU := nil;
  Positive := Default(TQwen21GPUCondition);
  Negative := Default(TQwen21GPUCondition);
  Work := Default(TQwen21GPUWork);
  Images := Inputs.Images;
  Lh := Inputs.LatentHeight;
  Lw := Inputs.LatentWidth;
  LatentCount := Lh * Lw * 64;
  try
    Adapters := TQNNLoRASet.Create(Inputs.LoRAs);
    Transformer := TQwen21Transformer.Create(Inputs.DiffusionFile);
    Transformer.AttachLoRAs(Adapters);
    if Length(Inputs.LoRAs) > 0 then
      Notify(
          Format(
              'LoRA stack: %d files, %d modules, %d enabled',
              [Length(Inputs.LoRAs), Adapters.ModuleCount, Adapters.EnabledModuleCount]
          )
      );
    SetLength(Resident, Length(Images));
    Ctx := TQNNGPUContext.Create;
    Notify('Resident Vulkan: activations stay in VRAM through encoding, denoising and VAE decoding');

    if Length(Images) > 0 then
    begin
      Notify('GPU RGBA VAE reference encoding');
      VAE := TQwen21VAE.Create(Inputs.VAEFile, TPath.Combine(FAssets, 'vae_config.json'), FStatus);
      for var I := 0 to High(Images) do
      begin
        Resident[I].Pixels :=
            Ctx.UploadRaw(@Images[I].Pixels.Data[0], Images[I].Pixels.Width * Images[I].Pixels.Height);
        Resident[I].Latents := VAE.EncodeGPU(Ctx, Resident[I].Pixels, Images[I].Pixels.Height, Images[I].Pixels.Width);
      end;
      FreeAndNil(VAE);
      Notify('GPU Qwen3-VL vision/DeepStack');
      Vision := TQwen21Vision.Create(Inputs.TextFile, FStatus);
      for var I := 0 to High(Images) do
        Vision.EncodeGPU(Ctx, Images[I], Resident[I]);
      FreeAndNil(Vision);
    end;

    Notify('GPU Qwen3-VL-8B ' + Inputs.Suffix);
    Tokenizer := TQwen21Tokenizer.Create(TPath.Combine(FAssets, 'tokenizer.json'));
    Encoder := TQwen21TextEncoder.Create(Inputs.TextFile, TPath.Combine(FAssets, 'encoder_config.json'), FStatus);
    Text := Encoder.EncodePromptGPU(Ctx, Tokenizer, Params.Prompt, Images, Resident, PositiveSlots);
    if Params.Guidance > 1 then
      NegativeText := Encoder.EncodePromptGPU(Ctx, Tokenizer, Params.NegativePrompt, Images, Resident, NegativeSlots);
    FreeAndNil(Encoder);
    FreeAndNil(Tokenizer);
    Positive := Transformer.ConditionGPU(Ctx, Text, Text.Count div 4096, Images, Resident, PositiveSlots, Lh, Lw);
    if Params.Guidance > 1 then
      Negative :=
          Transformer
              .ConditionGPU(Ctx, NegativeText, NegativeText.Count div 4096, Images, Resident, NegativeSlots, Lh, Lw);
    Ctx.Recycle(Text);
    Ctx.Recycle(NegativeText);

    Initial := InitNoise(1, 64, Lh, Lw, Params.Seed);
    InitialGPU := Ctx.Upload(Initial);
    Initial.Free;
    Latents := Ctx.NewTensor(LatentCount);
    Ctx.Kernel(GO_CHW_TO_ROWS, LatentCount, Latents, InitialGPU, nil, nil, [Lh * Lw, 64, 0]);
    Ctx.Recycle(InitialGPU);
    Noise := Ctx.NewTensor(LatentCount);
    if Params.Guidance > 1 then
      NegativeNoise := Ctx.NewTensor(LatentCount);
    Ctx.Flush;

    Notify('GPU Qwen-Image 2.1 ' + Inputs.Suffix);
    for var Step := 0 to Params.Steps - 1 do
    begin
      Timer := TStopwatch.StartNew;
      Transformer.ForwardGPU(Ctx, Noise, Latents, Positive, Lh, Lw, Inputs.Sigmas[Step], Work);
      if Params.Guidance > 1 then
      begin
        Transformer.ForwardGPU(Ctx, NegativeNoise, Latents, Negative, Lh, Lw, Inputs.Sigmas[Step], Work);
        Ctx.Kernel(GO_CFG, LatentCount, Noise, Noise, NegativeNoise, nil, [QNNGPUFloatBits(Params.Guidance)]);
      end;
      Q21GPUAdd(Ctx, Latents, Noise, LatentCount, Inputs.Sigmas[Step + 1] - Inputs.Sigmas[Step]);
      Ctx.CheckFinite(Latents);
      Ctx.Flush;
      Notify(
          Format(
              'GPU step %d/%d, sigma %.6f -> %.6f, %.2f s',
              [Step + 1, Params.Steps, Inputs.Sigmas[Step], Inputs.Sigmas[Step + 1], Timer.Elapsed.TotalSeconds]
          )
      );
    end;

    Work.Free(Ctx);
    Positive.Free(Ctx);
    Negative.Free(Ctx);
    FreeAndNil(Transformer);
    FreeAndNil(Adapters);
    for var I := 0 to High(Resident) do
      Resident[I].Free(Ctx);
    Notify('GPU RGBA VAE decode');
    VAE := TQwen21VAE.Create(Inputs.VAEFile, TPath.Combine(FAssets, 'vae_config.json'), FStatus);
    Pixels := VAE.DecodeGPU(Ctx, Latents, Lh, Lw);
    Result := TQNNImage.Create(Params.Width, Params.Height, 4);
    try
      Ctx.Download(Pixels, @Result.Data[0], Params.Width * Params.Height);
    except
      Result.Free;
      raise;
    end;
    FreeAndNil(VAE);
    Notify('Generation complete (resident Vulkan, final RGBA readback only)');
  finally
    Encoder.Free;
    Tokenizer.Free;
    Vision.Free;
    VAE.Free;
    Transformer.Free;
    Adapters.Free;
    Ctx.Free;
    Initial.Free;
  end;
end;
