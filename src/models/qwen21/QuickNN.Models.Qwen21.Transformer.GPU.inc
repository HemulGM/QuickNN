procedure TQwen21GPUCondition.Free(const Ctx: TQNNGPUContext);
begin
  Ctx.Recycle(Text);
  Ctx.Recycle(Positions);
  Ctx.Recycle(Limits);
  for var I := 0 to High(References) do
    Ctx.Recycle(References[I]);
  References := nil;
  Layout := Default(TQwen21SequenceLayout);
end;
procedure TQwen21GPUWork.Ensure(const Ctx: TQNNGPUContext; const Rows: Integer);
  procedure Grow(var Tensor: TQNNGPUTensor; const Count: Integer);
  begin
    if (Tensor = nil) or (Tensor.Count < Count) then
    begin
      Ctx.Recycle(Tensor);
      Tensor := Ctx.NewTensor(Count)
    end
  end;
begin
  Grow(X, Rows * HIDDEN);
  Grow(N, Rows * HIDDEN);
  Grow(Q, Rows * HIDDEN);
  Grow(K, Rows * HIDDEN);
  Grow(V, Rows * HIDDEN);
  Grow(A, Rows * HIDDEN);
  Grow(Y, Rows * HIDDEN);
  Grow(G, Rows * 2 * MLP);
  Grow(Active, Rows * MLP);
  Grow(T, 512);
  Grow(T1, 2 * HIDDEN);
  Grow(TEmb, 2 * HIDDEN);
  Grow(Modulation, 8 * HIDDEN);
  Grow(OutScale, HIDDEN);
end;
procedure TQwen21GPUWork.Free(const Ctx: TQNNGPUContext);
begin
  Ctx.Recycle(X);
  Ctx.Recycle(N);
  Ctx.Recycle(Q);
  Ctx.Recycle(K);
  Ctx.Recycle(V);
  Ctx.Recycle(A);
  Ctx.Recycle(Y);
  Ctx.Recycle(G);
  Ctx.Recycle(Active);
  Ctx.Recycle(T);
  Ctx.Recycle(T1);
  Ctx.Recycle(TEmb);
  Ctx.Recycle(Modulation);
  Ctx.Recycle(OutScale);
end;

function TQwen21Transformer.ConditionGPU(
    const Ctx: TQNNGPUContext;
    const Text: TQNNGPUTensor;
    const Rows: Integer;
    const Images: TQwen21ImageConditions;
    const Resident: TQwen21GPUImageConditions;
    const Slots: TArray<Integer>;
    const Height, Width: Integer
): TQwen21GPUCondition;
var
  Norm, Projected: TQNNGPUTensor;
begin
  Result := Default(TQwen21GPUCondition);
  Norm := nil;
  Projected := nil;
  if (Text = nil) or (Text.Count <> Rows * HIDDEN) or (Length(Images) <> Length(Resident)) then
    raise EArgumentException.Create('Invalid resident conditioning');
  Result.Layout := Q21ReferenceLayout(Rows, Images, Slots, Height, Width);
  try
    Norm := Ctx.NewTensor(Rows * HIDDEN);
    Projected := Ctx.NewTensor(Rows * HIDDEN);
    Result.Text := Ctx.NewTensor(Rows * HIDDEN);
    Q21GPURMS(Ctx, Norm, Text, FTextNorm, Rows, HIDDEN, True);
    FTextIn.ForwardGPU(Ctx, Projected, Norm, Rows);
    Ctx.Kernel(GO_UNARY, Rows * HIDDEN, Projected, Projected, nil, nil, [1]);
    FTextOut.ForwardGPU(Ctx, Result.Text, Projected, Rows);
    Result.Positions := Ctx.UploadRaw(@Result.Layout.Positions[0], Length(Result.Layout.Positions));
    Result.Limits := Ctx.UploadRaw(@Result.Layout.KeyLimits[0], Length(Result.Layout.KeyLimits));
    SetLength(Result.References, Length(Images));
    for var I := 0 to High(Images) do
    begin
      var RefRows := (Images[I].Pixels.Width div 16) * (Images[I].Pixels.Height div 16);
      if (Resident[I].Latents = nil) or (Resident[I].Latents.Count <> RefRows * 64) then
        raise EArgumentException.Create('Invalid resident reference latents');
      Result.References[I] := Ctx.NewTensor(RefRows * HIDDEN);
      FImageIn.ForwardGPU(Ctx, Result.References[I], Resident[I].Latents, RefRows);
    end;
  finally
    Ctx.Recycle(Norm);
    Ctx.Recycle(Projected)
  end;
end;

procedure TQwen21Transformer.ForwardGPU(
    const Ctx: TQNNGPUContext;
    const Dst, Latents: TQNNGPUTensor;
    const Condition: TQwen21GPUCondition;
    const Height, Width: Integer;
    const Sigma: Single;
    var Work: TQwen21GPUWork
);
var
  Rows, Prefix, ImageRows: Integer;
  Block: ^TQwen21Block;
begin
  Rows := Condition.Layout.Rows;
  Prefix := Condition.Layout.PrefixRows;
  ImageRows := Height * Width;
  if (Latents = nil) or (Latents.Count <> ImageRows * 64) or (Dst = nil) or (Dst.Count <> ImageRows * 64) then
    raise EArgumentException.Create('Invalid resident latent grid');
  Work.Ensure(Ctx, Rows);
  for var Segment in Condition.Layout.Segments do
    if not Segment.IsImage then
      Ctx.Copy(
          Work.X,
          Condition.Text,
          Segment.Count * HIDDEN,
          Segment.JointStart * HIDDEN,
          Segment.SourceStart * HIDDEN
      )
    else if Segment.SourceIndex < 0 then
    begin
      FImageIn.ForwardGPU(Ctx, Work.Y, Latents, ImageRows);
      Ctx.Copy(Work.X, Work.Y, ImageRows * HIDDEN, Segment.JointStart * HIDDEN);
    end
    else
      Ctx.Copy(Work.X, Condition.References[Segment.SourceIndex], Segment.Count * HIDDEN, Segment.JointStart * HIDDEN);
  Ctx.Kernel(GO_TIME, 512, Work.T, nil, nil, nil, [QNNGPUFloatBits(Sigma)]);
  FTime1.ForwardGPU(Ctx, Work.T1, Work.T, 2);
  Ctx.Kernel(GO_UNARY, 2 * HIDDEN, Work.T1, Work.T1, nil, nil, [0]);
  FTime2.ForwardGPU(Ctx, Work.TEmb, Work.T1, 2);
  Ctx.Copy(Work.T1, Work.TEmb, 2 * HIDDEN);
  Ctx.Kernel(GO_UNARY, 2 * HIDDEN, Work.T1, Work.T1, nil, nil, [0]);
  FModulation.ForwardGPU(Ctx, Work.Modulation, Work.T1, 2);
  for var I := 0 to High(FBlocks) do
  begin
    Block := @FBlocks[I];
    Ctx.Kernel(GO_MOD_NORM, Rows * HIDDEN, Work.N, Work.X, Work.Modulation, nil, [Rows, HIDDEN, 1, Prefix, 0, 0]);
    Block.Q.ForwardGPU(Ctx, Work.Q, Work.N, Rows);
    Block.K.ForwardGPU(Ctx, Work.K, Work.N, Rows);
    Block.V.ForwardGPU(Ctx, Work.V, Work.N, Rows);
    Q21GPURMS(Ctx, Work.Q, Work.Q, Block.QNorm, Rows * HEADS, HEAD_DIM);
    Q21GPURMS(Ctx, Work.K, Work.K, Block.KNorm, Rows * HEADS, HEAD_DIM);
    Ctx.Kernel(GO_IMAGE_ROPE, Rows * HEADS * 64, Work.Q, Work.Q, Condition.Positions, nil, [HEADS]);
    Ctx.Kernel(GO_IMAGE_ROPE, Rows * HEADS * 64, Work.K, Work.K, Condition.Positions, nil, [HEADS]);
    Ctx.Attention(Work.A, Work.Q, Work.K, Work.V, Rows, HEADS, HEADS, HEAD_DIM, Prefix, Condition.Limits);
    Block.O.ForwardGPU(Ctx, Work.Y, Work.A, Rows);
    Ctx.Kernel(GO_GATE_ADD, Rows * HIDDEN, Work.X, Work.Y, Work.Modulation, nil, [HIDDEN, Prefix, HIDDEN]);
    Ctx.Kernel(
        GO_MOD_NORM,
        Rows * HIDDEN,
        Work.N,
        Work.X,
        Work.Modulation,
        nil,
        [Rows, HIDDEN, 1, Prefix, 2 * HIDDEN, 0]
    );
    Block.GateUp.ForwardGPU(Ctx, Work.G, Work.N, Rows);
    Ctx.Kernel(GO_SWIGLU, Rows * MLP, Work.Active, Work.G, nil, nil, [MLP, 1]);
    Block.Down.ForwardGPU(Ctx, Work.Y, Work.Active, Rows);
    Ctx.Kernel(GO_GATE_ADD, Rows * HIDDEN, Work.X, Work.Y, Work.Modulation, nil, [HIDDEN, Prefix, 3 * HIDDEN]);
    Ctx.CheckFinite(Work.X, Rows * HIDDEN);
  end;
  FNormOut.ForwardGPU(Ctx, Work.OutScale, Work.T1, 1);
  Ctx.Kernel(
      GO_MOD_NORM,
      ImageRows * HIDDEN,
      Work.N,
      Work.X,
      Work.OutScale,
      nil,
      [ImageRows, HIDDEN, 1, 0, 0, Prefix * HIDDEN]
  );
  FOut.ForwardGPU(Ctx, Dst, Work.N, ImageRows);
  Ctx.CheckFinite(Dst);
end;
