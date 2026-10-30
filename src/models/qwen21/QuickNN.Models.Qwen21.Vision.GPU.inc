function TQwen21VisionMerger.ForwardGPU(
    const Ctx: TQNNGPUContext;
    const Src: TQNNGPUTensor;
    const Rows: Integer
): TQNNGPUTensor;
var
  Normalized, Projected: TQNNGPUTensor;
begin
  Normalized := nil;
  Projected := nil;
  try
    Normalized := Ctx.NewTensor(Rows * HIDDEN);
    Projected := Ctx.NewTensor(Rows div 4 * 4608);
    if PostShuffle then
      Q21GPUAffine(Ctx, Normalized, Src, Norm, NormBias, Rows div 4, 4608)
    else
      Q21GPUAffine(Ctx, Normalized, Src, Norm, NormBias, Rows, HIDDEN);
    FC1.ForwardGPU(Ctx, Projected, Normalized, Rows div 4);
    Ctx.Kernel(GO_UNARY, Rows div 4 * 4608, Projected, Projected, nil, nil, [2]);
    Result := Ctx.NewTensor(Rows div 4 * 4096);
    FC2.ForwardGPU(Ctx, Result, Projected, Rows div 4);
    Ctx.CheckFinite(Result);
  finally
    Ctx.Recycle(Normalized);
    Ctx.Recycle(Projected)
  end;
end;

procedure TQwen21Vision.EncodeGPU(
    const Ctx: TQNNGPUContext;
    var Image: TQwen21ImageCondition;
    var Resident: TQwen21GPUImageCondition
);
var
  Patches, X, N, Qkv, Q, K, V, A, Y, Mlp: TQNNGPUTensor;
  Gh, Gw, Rows: Integer;
  Block: ^TQwen21VisionBlock;
begin
  Gh := Image.Pixels.Height div 16;
  Gw := Image.Pixels.Width div 16;
  Rows := Gh * Gw;
  Image.GridHeight := Gh;
  Image.GridWidth := Gw;
  if (Rows < 4) or (Rows > 4096) or (Gh mod 2 <> 0) or (Gw mod 2 <> 0) or (Resident.Pixels = nil) then
    raise EArgumentException.Create('Invalid resident vision image/grid');
  Patches := nil;
  X := nil;
  N := nil;
  Qkv := nil;
  Q := nil;
  K := nil;
  V := nil;
  A := nil;
  Y := nil;
  Mlp := nil;
  try
    Patches := Ctx.NewTensor(Rows * 1536);
    Ctx.Kernel(GO_VISION_PATCH, Rows * 1536, Patches, Resident.Pixels, nil, nil, [Gh, Gw]);
    X := Ctx.NewTensor(Rows * HIDDEN);
    FPatch.ForwardGPU(Ctx, X, Patches, Rows);
    Ctx.Recycle(Patches);
    Ctx.KernelWeights(GO_VISION_POSITION, Rows * HIDDEN, X, X, FPosition, Default(TMemoryBlock), [Gh, Gw]);
    N := Ctx.NewTensor(Rows * HIDDEN);
    Qkv := Ctx.NewTensor(Rows * 3 * HIDDEN);
    Q := Ctx.NewTensor(Rows * HIDDEN);
    K := Ctx.NewTensor(Rows * HIDDEN);
    V := Ctx.NewTensor(Rows * HIDDEN);
    A := Ctx.NewTensor(Rows * HIDDEN);
    Y := Ctx.NewTensor(Rows * HIDDEN);
    Mlp := Ctx.NewTensor(Rows * 4304);
    for var I := 0 to 26 do
    begin
      Block := @FBlocks[I];
      Q21GPUAffine(Ctx, N, X, Block.Norm1, Block.Norm1Bias, Rows, HIDDEN);
      Block.Qkv.ForwardGPU(Ctx, Qkv, N, Rows);
      Ctx.Kernel(GO_SLICE, Rows * HIDDEN, Q, Qkv, nil, nil, [HIDDEN, 3 * HIDDEN, 0, 0]);
      Ctx.Kernel(GO_SLICE, Rows * HIDDEN, K, Qkv, nil, nil, [HIDDEN, 3 * HIDDEN, HIDDEN, 0]);
      Ctx.Kernel(GO_SLICE, Rows * HIDDEN, V, Qkv, nil, nil, [HIDDEN, 3 * HIDDEN, 2 * HIDDEN, 0]);
      Ctx.Kernel(GO_VISION_ROPE, Rows * HEADS * (HEAD_DIM div 2), Q, Q, nil, nil, [Gh, Gw]);
      Ctx.Kernel(GO_VISION_ROPE, Rows * HEADS * (HEAD_DIM div 2), K, K, nil, nil, [Gh, Gw]);
      Ctx.Attention(A, Q, K, V, Rows, HEADS, HEADS, HEAD_DIM, 0);
      Block.Proj.ForwardGPU(Ctx, Y, A, Rows);
      Q21GPUAdd(Ctx, X, Y, Rows * HIDDEN);
      Q21GPUAffine(Ctx, N, X, Block.Norm2, Block.Norm2Bias, Rows, HIDDEN);
      Block.FC1.ForwardGPU(Ctx, Mlp, N, Rows);
      Ctx.Kernel(GO_UNARY, Rows * 4304, Mlp, Mlp, nil, nil, [1]);
      Block.FC2.ForwardGPU(Ctx, Y, Mlp, Rows);
      Q21GPUAdd(Ctx, X, Y, Rows * HIDDEN);
      Ctx.CheckFinite(X);
      case I of
        8: Resident.DeepStack[0] := FDeep[0].ForwardGPU(Ctx, X, Rows);
        16: Resident.DeepStack[1] := FDeep[1].ForwardGPU(Ctx, X, Rows);
        24: Resident.DeepStack[2] := FDeep[2].ForwardGPU(Ctx, X, Rows)
      end;
      if Assigned(FStatus) and ((I + 1) mod 9 = 0) then
        FStatus(Format('GPU vision encoder %d/27', [I + 1]));
    end;
    Resident.Visual := FMerger.ForwardGPU(Ctx, X, Rows);
  finally
    Ctx.Recycle(Patches);
    Ctx.Recycle(X);
    Ctx.Recycle(N);
    Ctx.Recycle(Qkv);
    Ctx.Recycle(Q);
    Ctx.Recycle(K);
    Ctx.Recycle(V);
    Ctx.Recycle(A);
    Ctx.Recycle(Y);
    Ctx.Recycle(Mlp)
  end;
end;
