function TQNNFlux.GenerateResident(
    const Prompt, NegativePrompt: RawByteString;
    var Params: TGenerateParams;
    const Images: TArray<TQNNImage>;
    const Progress: TProgressCallback
): TQNNImage;
var
  Ctx: TQNNGPUContext;
  Work: TGPUWorkspace;
  Positive, Negative, RefPixels, NoiseGPU, Latent, Pixels: TQNNGPUTensor;
  Refs: TGPUImageRefs;
  RefImage, Resized: TQNNImage;
  Noise, Schedule: TMemoryBlock;
  TextRows, IgnoredRows, H, W, RefH, RefW: Integer;
begin
  Ctx := nil;
  Work := nil;
  if (Params.Width <= 0) and (Length(Images) > 0) then
    Params.Width := Images[0].Width;
  if (Params.Height <= 0) and (Length(Images) > 0) then
    Params.Height := Images[0].Height;
  if Params.Width <= 0 then
    Params.Width := QNN_DEFAULT_WIDTH;
  if Params.Height <= 0 then
    Params.Height := QNN_DEFAULT_HEIGHT;
  Params.Width := Max(64, Params.Width div 16 * 16);
  Params.Height := Max(64, Params.Height div 16 * 16);
  if (Params.Width > QNN_VAE_MAX_DIM) or (Params.Height > QNN_VAE_MAX_DIM) then
    raise EArgumentException.Create('FLUX dimensions exceed VAE limit');
  if Params.NumSteps <= 0 then
    Params.NumSteps := DefaultSteps;
  if Params.Guidance <= 0 then
    Params.Guidance := DefaultGuidance;
  if Params.Seed <= 0 then
    Params.Seed := GetTickCount64;
  H := Params.Height div 16;
  W := Params.Width div 16;
  try
    // Legacy mmap sampling may release its block arrays. Reload native resident
    // storage when changing execution modes instead of reusing that workspace.
    if not Qwen3Encoder.Model.ResidentOnly then
      if Assigned(Qwen3Encoder.Model.Layers) then
        Qwen3Encoder.Free;
    if Transformer.IsLoaded and not Transformer.ResidentOnly then
      Transformer.Free;
    if not Assigned(Qwen3Encoder.Model.Layers) then
      Qwen3Encoder.Load(ModelDir, UseMmap, True);
    Qwen3Encoder.Model.SetExtractionMode(False);
    if not Transformer.IsLoaded then
      Transformer := TTransformerFlux.Load(ModelDir, UseMmap, True);
    if not VAE.DecConvInWeight.IsAssigned then
      LoadVAE;
    Noise := InitNoise(1, Transformer.LatentChannels, H, W, Params.Seed);
    Schedule := SelectedSchedule(Params, H * W);
    Ctx := TQNNGPUContext.Create;
    Work := TGPUWorkspace.Create(Ctx);
    if Assigned(PhaseCallback) then
      PhaseCallback('Resident FLUX: text, transformer, Euler and VAE on GPU', False);
    Positive := Work.Own(Qwen3EncodeGPU(Ctx, Qwen3Encoder, Prompt, IgnoredRows));
    TextRows := QWEN3_MAX_SEQ_LEN;
    Negative := nil;
    if not ((Params.Guidance = 1) and IsDistilled) then
      Negative := Work.Own(Qwen3EncodeGPU(Ctx, Qwen3Encoder, NegativePrompt, IgnoredRows));
    SetLength(Refs, Length(Images));
    for var I := 0 to High(Images) do
    begin
      RefImage := Images[I];
      if Length(Images) = 1 then
      begin
        RefH := Params.Height;
        RefW := Params.Width;
      end
      else
      begin
        RefH := EnsureRange(RefImage.Height div 16 * 16, 16, QNN_VAE_MAX_DIM);
        RefW := EnsureRange(RefImage.Width div 16 * 16, 16, QNN_VAE_MAX_DIM);
      end;
      if (RefH <> RefImage.Height) or (RefW <> RefImage.Width) then
      begin
        Resized := RefImage.Resize(RefW, RefH);
        RefImage := Resized;
      end;
      try
        RefPixels := Work.Own(UploadImageGPU(Ctx, RefImage));
        Refs[I].Latent := Work.Own(VAEEncodeGPU(Ctx, VAE, RefPixels, RefH, RefW));
        Refs[I].H := RefH div 16;
        Refs[I].W := RefW div 16;
        Refs[I].TOffset := 10 * (I + 1);
        Work.Release(RefPixels);
      finally
        Resized.Free;
      end;
    end;
    NoiseGPU := Work.Own(Ctx.Upload(Noise));
    Latent :=
        Work.Own(
            FluxSampleGPU(
                Ctx,
                Transformer,
                NoiseGPU,
                H,
                W,
                Refs,
                Positive,
                Negative,
                TextRows,
                TextRows,
                Params.Guidance,
                Schedule,
                Params.NumSteps,
                Progress
            )
        );
    Pixels := Work.Own(VAEDecodeGPU(Ctx, VAE, Latent, H, W));
    Result := GPUDownloadRGB(Ctx, Pixels, Params.Height, Params.Width);
    if Assigned(PhaseCallback) then
      PhaseCallback('Resident FLUX generation', True);
  finally
    Work.Free;
    Ctx.Free;
    Noise.Free;
    Schedule.Free;
    // Model members keep their native host mappings and immutable cache identities
    // for the next generation. TQNNFlux.Free releases them and the VRAM weights.
  end;
end;
