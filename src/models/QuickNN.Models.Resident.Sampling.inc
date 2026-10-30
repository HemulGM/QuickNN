function FluxSampleGPU(
    const Ctx: TQNNGPUContext;
    var Model: TTransformerFlux;
    const Noise: TQNNGPUTensor;
    const Height, Width: Integer;
    const Refs: TGPUImageRefs;
    const Positive, Negative: TQNNGPUTensor;
    const PositiveRows, NegativeRows: Integer;
    const Guidance: Single;
    const Schedule: TMemoryBlock;
    const Steps: Integer;
    const Progress: TProgressCallback
): TQNNGPUTensor;
var
  Work: TGPUWorkspace;
  Prediction, Unconditional: TQNNGPUTensor;
  Times: System.PSingle;
  Snapshot: TMemoryBlock;
  Preview: TQNNImage;
  PixelGPU: TQNNGPUTensor;
  Count: Integer;
begin
  if (Steps < 1) or (Schedule.Count < Steps + 1) or (Schedule.DataType <> dtF32) then
    raise EArgumentException.Create('Invalid resident FLUX schedule');
  Work := TGPUWorkspace.Create(Ctx);
  try
    Count := Height * Width * Model.LatentChannels;
    Result := Work.Copy(Noise, Count);
    Times := Schedule;
    for var Step := 0 to Steps - 1 do
    begin
      if Assigned(StepCallback) then
        StepCallback(Step, Steps);
      Unconditional := nil;
      if Negative <> nil then
        Unconditional :=
            Work.Own(FluxForwardGPU(Ctx, Model, Result, Height, Width, Refs, Negative, NegativeRows, Times[Step]));
      Prediction :=
          Work.Own(FluxForwardGPU(Ctx, Model, Result, Height, Width, Refs, Positive, PositiveRows, Times[Step]));
      if Unconditional <> nil then
        Ctx.Kernel(GO_CFG, Count, Prediction, Prediction, Unconditional, nil, [QNNGPUFloatBits(Guidance)]);
      Work.Add(Result, Prediction, Count, Times[Step + 1] - Times[Step]);
      Work.Release(Prediction);
      Work.Release(Unconditional);
      Ctx.CheckFinite(Result);
      // CPU tensor/image callbacks explicitly request a readback. Scalar StepCallback does not.
      if Assigned(Progress) then
      begin
        Snapshot := TMemoryBlock.Create([1, Model.LatentChannels, Height, Width], 'flux-explicit-progress');
        try
          Ctx.Download(Result, System.PSingle(Snapshot), Count);
          Progress(Step, Steps, Snapshot);
        finally
          Snapshot.Free;
        end;
      end;
      if Assigned(StepImageCallback) and Assigned(VAEPtr) and (Step + 1 < Steps) then
      begin
        PixelGPU := Work.Own(VAEDecodeGPU(Ctx, PVAE(VAEPtr)^, Result, Height, Width));
        Preview := GPUDownloadRGB(Ctx, PixelGPU, Height * 16, Width * 16);
        try
          StepImageCallback(Step, Steps, Preview);
        finally
          Preview.Free;
          Work.Release(PixelGPU);
        end;
      end;
    end;
    Work.Keep(Result);
  finally
    Work.Free;
  end;
end;

function ZImageSampleGPU(
    const Ctx: TQNNGPUContext;
    var Model: TTransformerZI;
    const Noise: TQNNGPUTensor;
    const Height, Width: Integer;
    const Text: TQNNGPUTensor;
    const TextRows: Integer;
    const Schedule: TMemoryBlock;
    const Steps: Integer
): TQNNGPUTensor;
var
  Work: TGPUWorkspace;
  Prediction, PackedLatent, PixelGPU: TQNNGPUTensor;
  Times: System.PSingle;
  Preview: TQNNImage;
  Count: Integer;
begin
  if (Steps < 1) or (Schedule.Count < Steps + 1) or (Schedule.DataType <> dtF32) then
    raise EArgumentException.Create('Invalid resident Z-Image schedule');
  Work := TGPUWorkspace.Create(Ctx);
  try
    Count := Height * Width * Model.InChannels;
    Result := Work.Copy(Noise, Count);
    Times := Schedule;
    for var Step := 0 to Steps - 1 do
    begin
      if Assigned(StepCallback) then
        StepCallback(Step, Steps);
      Prediction := Work.Own(ZImageForwardGPU(Ctx, Model, Result, Height, Width, Text, TextRows, 1 - Times[Step]));
      Work.Add(Result, Prediction, Count, Times[Step] - Times[Step + 1]);
      Work.Release(Prediction);
      Ctx.CheckFinite(Result);
      if Assigned(StepImageCallback) and Assigned(VAEPtr) and (Step + 1 < Steps) then
      begin
        PackedLatent := Work.Patch(Result, Height, Width, Model.InChannels, Model.PatchSize, 2);
        PixelGPU :=
            Work.Own(
                VAEDecodeGPU(Ctx, PVAE(VAEPtr)^, PackedLatent, Height div Model.PatchSize, Width div Model.PatchSize)
            );
        Preview := GPUDownloadRGB(Ctx, PixelGPU, Height * 8, Width * 8);
        try
          StepImageCallback(Step, Steps, Preview);
        finally
          Preview.Free;
          Work.Release(PixelGPU);
          Work.Release(PackedLatent);
        end;
      end;
    end;
    Work.Keep(Result);
  finally
    Work.Free;
  end;
end;

function UploadImageGPU(const Ctx: TQNNGPUContext; const Image: TQNNImage): TQNNGPUTensor;
var
  Work: TGPUWorkspace;
  PackedPixels: TArray<UInt32>;
  Input: TQNNGPUTensor;
begin
  if not (Image.Channels in [3, 4])
      or (Image.Width < 1)
      or (Image.Height < 1)
      or (Length(Image.Data) < Int64(Image.Width) * Image.Height * Image.Channels) then
    raise EArgumentException.Create('Resident image input requires RGB/RGBA pixels');
  SetLength(PackedPixels, Image.Width * Image.Height);
  for var I := 0 to High(PackedPixels) do
    PackedPixels[I] :=
        UInt32(Image.Data[I * Image.Channels])
            or (UInt32(Image.Data[I * Image.Channels + 1]) shl 8)
            or (UInt32(Image.Data[I * Image.Channels + 2]) shl 16);
  Work := TGPUWorkspace.Create(Ctx);
  try
    Input := Work.Own(Ctx.UploadRaw(@PackedPixels[0], Length(PackedPixels)));
    Result := Work.New(Int64(Image.Width) * Image.Height * 3);
    Ctx.Kernel(GO_PIXEL_INPUT, Result.Count, Result, Input, nil, nil, [Image.Width * Image.Height]);
    Work.Keep(Result);
  finally
    Work.Free;
  end;
end;
