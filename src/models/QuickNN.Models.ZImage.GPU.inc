function TQNNZImage.GenerateResident(
    const Prompt: RawByteString;
    const Embeddings: TMemoryBlock;
    const EmbeddingRows: LongInt;
    Params: TGenerateParams
): TQNNImage;
var
  Ctx: TQNNGPUContext;
  Work: TGPUWorkspace;
  Text, NoiseGPU, Denoised, Latent, Pixels: TQNNGPUTensor;
  Noise, Schedule: TMemoryBlock;
  TextRows, H, W: Integer;
begin
  Ctx := nil;
  Work := nil;
  if Params.Width <= 0 then
    Params.Width := 1024;
  if Params.Height <= 0 then
    Params.Height := 1024;
  Params.Width := Max(64, Params.Width div 16 * 16);
  Params.Height := Max(64, Params.Height div 16 * 16);
  if (Params.Width > QNN_VAE_MAX_DIM) or (Params.Height > QNN_VAE_MAX_DIM) then
    raise EArgumentException.Create('Z-Image dimensions exceed VAE limit');
  if Params.NumSteps <= 0 then
    Params.NumSteps := DefaultSteps;
  if Params.Seed < 0 then
    Params.Seed := RandSeed;
  if Params.Schedule = QNN_SCHEDULE_DEFAULT then
    Params.Schedule := QNN_SCHEDULE_FLOWMATCH;
  H := Params.Height div 8;
  W := Params.Width div 8;
  try
    // Legacy mmap sampling may release its block arrays. Reload native resident
    // storage when changing execution modes instead of reusing that workspace.
    if not Qwen3Encoder.Model.ResidentOnly then
      if Assigned(Qwen3Encoder.Model.Layers) then
        Qwen3Encoder.Free;
    if Transformer.IsLoaded and not Transformer.ResidentOnly then
      Transformer.Free;
    if not Embeddings.IsAssigned and not Assigned(Qwen3Encoder.Model.Layers) then
      Qwen3Encoder.Load(ModelDir, UseMmap, True);
    if not Transformer.IsLoaded then
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
          UseMmap,
          True
      );
    if not VAE.DecConvInWeight.IsAssigned then
      LoadVAE;
    Noise := InitNoise(1, Transformer.InChannels, H, W, Params.Seed);
    Schedule := SelectedSchedule(Params, (H div ZiPatchSize) * (W div ZiPatchSize));
    Ctx := TQNNGPUContext.Create;
    Work := TGPUWorkspace.Create(Ctx);
    if Assigned(PhaseCallback) then
      PhaseCallback('Resident Z-Image: text, transformer, Euler and VAE on GPU', False);
    if Embeddings.IsAssigned then
    begin
      TextRows := EmbeddingRows;
      Text := Work.Own(Ctx.Upload(Embeddings));
    end
    else
    begin
      Qwen3Encoder.Model.SetExtractionMode(True);
      Text := Work.Own(Qwen3EncodeGPU(Ctx, Qwen3Encoder, Prompt, TextRows));
    end;
    NoiseGPU := Work.Own(Ctx.Upload(Noise));
    Denoised := Work.Own(ZImageSampleGPU(Ctx, Transformer, NoiseGPU, H, W, Text, TextRows, Schedule, Params.NumSteps));
    Latent := Work.Patch(Denoised, H, W, Transformer.InChannels, Transformer.PatchSize, 2);
    Pixels := Work.Own(VAEDecodeGPU(Ctx, VAE, Latent, H div ZiPatchSize, W div ZiPatchSize));
    Result := GPUDownloadRGB(Ctx, Pixels, Params.Height, Params.Width);
    if Assigned(PhaseCallback) then
      PhaseCallback('Resident Z-Image generation', True);
  finally
    Work.Free;
    Ctx.Free;
    Noise.Free;
    Schedule.Free;
  end;
end;
