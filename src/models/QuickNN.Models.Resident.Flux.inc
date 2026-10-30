procedure FluxRoPEGPU(
    const Ctx: TQNNGPUContext;
    const Tensor: TQNNGPUTensor;
    const Heads, HeadDim, Height, Width, TimeOffset, RowOffset: Integer;
    const Text: Boolean;
    const Theta: Single
);
begin
  Ctx.Kernel(
      GO_FLUX_ROPE,
      Height * Width * Heads * HeadDim div 2,
      Tensor,
      Tensor,
      nil,
      nil,
      [Heads, HeadDim, Width, Height, TimeOffset, Ord(Text), QNNGPUFloatBits(Theta), RowOffset]
  );
end;

procedure FluxImageRoPEGPU(
    const Ctx: TQNNGPUContext;
    const Tensor: TQNNGPUTensor;
    const Model: TTransformerFlux;
    const Height, Width, Offset: Integer;
    const Refs: TGPUImageRefs
);
var
  Row: Integer;
begin
  FluxRoPEGPU(Ctx, Tensor, Model.NumHeads, Model.HeadDim, Height, Width, 0, Offset, False, Model.RoPETheta);
  Row := Offset + Height * Width;
  for var Ref in Refs do
  begin
    FluxRoPEGPU(Ctx, Tensor, Model.NumHeads, Model.HeadDim, Ref.H, Ref.W, Ref.TOffset, Row, False, Model.RoPETheta);
    Inc(Row, Ref.H * Ref.W);
  end;
end;

procedure FluxDoubleGPU(
    const Ctx: TQNNGPUContext;
    const Model: TTransformerFlux;
    const Block: TDoubleBlock;
    const Image, Text, ImageMod, TextMod: TQNNGPUTensor;
    const ImageRows, TextRows, Height, Width: Integer;
    const Refs: TGPUImageRefs
);
var
  Work: TGPUWorkspace;
  IQ, IK, IV, TQ, TK, TV, Q, K, V, A, N, Proj, ImageA, TextA: TQNNGPUTensor;
  Hidden, Rows: Integer;
begin
  Work := TGPUWorkspace.Create(Ctx);
  try
    Hidden := Model.HiddenSize;
    Rows := ImageRows + TextRows;
    N := Work.Norm(Image, Default(TMemoryBlock), Default(TMemoryBlock), ImageRows, Hidden, True);
    N := Work.Modulate(N, ImageMod, ImageRows, Hidden, 0, Hidden, 0);
    IQ := Work.Linear(N, GPUWeight(Block.ImgQWeight, Block.ImgQWeightBF16), ImageRows, Hidden, Hidden);
    IK := Work.Linear(N, GPUWeight(Block.ImgKWeight, Block.ImgKWeightBF16), ImageRows, Hidden, Hidden);
    IV := Work.Linear(N, GPUWeight(Block.ImgVWeight, Block.ImgVWeightBF16), ImageRows, Hidden, Hidden);
    IQ := Work.Norm(IQ, Block.ImgNormQWeight, Default(TMemoryBlock), ImageRows * Model.NumHeads, Model.HeadDim);
    IK := Work.Norm(IK, Block.ImgNormKWeight, Default(TMemoryBlock), ImageRows * Model.NumHeads, Model.HeadDim);
    FluxImageRoPEGPU(Ctx, IQ, Model, Height, Width, 0, Refs);
    FluxImageRoPEGPU(Ctx, IK, Model, Height, Width, 0, Refs);
    N := Work.Norm(Text, Default(TMemoryBlock), Default(TMemoryBlock), TextRows, Hidden, True);
    N := Work.Modulate(N, TextMod, TextRows, Hidden, 0, Hidden, 0);
    TQ := Work.Linear(N, GPUWeight(Block.TxtQWeight, Block.TxtQWeightBF16), TextRows, Hidden, Hidden);
    TK := Work.Linear(N, GPUWeight(Block.TxtKWeight, Block.TxtKWeightBF16), TextRows, Hidden, Hidden);
    TV := Work.Linear(N, GPUWeight(Block.TxtVWeight, Block.TxtVWeightBF16), TextRows, Hidden, Hidden);
    TQ := Work.Norm(TQ, Block.TxtNormQWeight, Default(TMemoryBlock), TextRows * Model.NumHeads, Model.HeadDim);
    TK := Work.Norm(TK, Block.TxtNormKWeight, Default(TMemoryBlock), TextRows * Model.NumHeads, Model.HeadDim);
    FluxRoPEGPU(Ctx, TQ, Model.NumHeads, Model.HeadDim, 1, TextRows, 0, 0, True, Model.RoPETheta);
    FluxRoPEGPU(Ctx, TK, Model.NumHeads, Model.HeadDim, 1, TextRows, 0, 0, True, Model.RoPETheta);
    Q := Work.Concat(TQ, IQ, TextRows * Hidden, ImageRows * Hidden);
    K := Work.Concat(TK, IK, TextRows * Hidden, ImageRows * Hidden);
    V := Work.Concat(TV, IV, TextRows * Hidden, ImageRows * Hidden);
    A := Work.New(Int64(Rows) * Hidden);
    Ctx.Attention(A, Q, K, V, Rows, Model.NumHeads, Model.NumHeads, Model.HeadDim, 0);
    TextA := Work.Copy(A, TextRows * Hidden);
    ImageA := Work.Copy(A, ImageRows * Hidden, TextRows * Hidden);
    Proj := Work.Linear(ImageA, GPUWeight(Block.ImgProjWeight, Block.ImgProjWeightBF16), ImageRows, Hidden, Hidden);
    Work.Modulate(Proj, ImageMod, ImageRows, Hidden, 1, 2 * Hidden, 0, Image);
    Proj := Work.Linear(TextA, GPUWeight(Block.TxtProjWeight, Block.TxtProjWeightBF16), TextRows, Hidden, Hidden);
    Work.Modulate(Proj, TextMod, TextRows, Hidden, 1, 2 * Hidden, 0, Text);
    N := Work.Norm(Image, Default(TMemoryBlock), Default(TMemoryBlock), ImageRows, Hidden, True);
    N := Work.Modulate(N, ImageMod, ImageRows, Hidden, 0, 4 * Hidden, 3 * Hidden);
    Proj :=
        Work.SwiGLU(
            N,
            GPUWeight(Block.ImgMlpGateWeight, Block.ImgMlpGateWeightBF16),
            GPUWeight(Block.ImgMlpUpWeight, Block.ImgMlpUpWeightBF16),
            GPUWeight(Block.ImgMlpDownWeight, Block.ImgMlpDownWeightBF16),
            ImageRows,
            Hidden,
            Model.MlpHidden
        );
    Work.Modulate(Proj, ImageMod, ImageRows, Hidden, 1, 5 * Hidden, 0, Image);
    N := Work.Norm(Text, Default(TMemoryBlock), Default(TMemoryBlock), TextRows, Hidden, True);
    N := Work.Modulate(N, TextMod, TextRows, Hidden, 0, 4 * Hidden, 3 * Hidden);
    Proj :=
        Work.SwiGLU(
            N,
            GPUWeight(Block.TxtMlpGateWeight, Block.TxtMlpGateWeightBF16),
            GPUWeight(Block.TxtMlpUpWeight, Block.TxtMlpUpWeightBF16),
            GPUWeight(Block.TxtMlpDownWeight, Block.TxtMlpDownWeightBF16),
            TextRows,
            Hidden,
            Model.MlpHidden
        );
    Work.Modulate(Proj, TextMod, TextRows, Hidden, 1, 5 * Hidden, 0, Text);
  finally
    Work.Free;
  end;
end;

procedure FluxSingleGPU(
    const Ctx: TQNNGPUContext;
    const Model: TTransformerFlux;
    const Block: TSingleBlock;
    const X, Modulation: TQNNGPUTensor;
    const Rows, TextRows, Height, Width: Integer;
    const Refs: TGPUImageRefs
);
var
  Work: TGPUWorkspace;
  N, Fused, Q, K, V, Gate, Up, A, Joined, Proj: TQNNGPUTensor;
  Hidden, Mlp, FusedDim: Integer;
begin
  Work := TGPUWorkspace.Create(Ctx);
  try
    Hidden := Model.HiddenSize;
    Mlp := Model.MlpHidden;
    FusedDim := Hidden * 3 + Mlp * 2;
    N := Work.Norm(X, Default(TMemoryBlock), Default(TMemoryBlock), Rows, Hidden, True);
    N := Work.Modulate(N, Modulation, Rows, Hidden, 0, Hidden, 0);
    Fused := Work.Linear(N, GPUWeight(Block.QKVMlpWeight, Block.QKVMlpWeightBF16), Rows, Hidden, FusedDim);
    Q := Work.Slice(Fused, Rows, Hidden, FusedDim, 0);
    K := Work.Slice(Fused, Rows, Hidden, FusedDim, Hidden);
    V := Work.Slice(Fused, Rows, Hidden, FusedDim, 2 * Hidden);
    Gate := Work.Slice(Fused, Rows, Mlp, FusedDim, 3 * Hidden);
    Up := Work.Slice(Fused, Rows, Mlp, FusedDim, 3 * Hidden + Mlp);
    Q := Work.Norm(Q, Block.NormQWeight, Default(TMemoryBlock), Rows * Model.NumHeads, Model.HeadDim);
    K := Work.Norm(K, Block.NormKWeight, Default(TMemoryBlock), Rows * Model.NumHeads, Model.HeadDim);
    FluxRoPEGPU(Ctx, Q, Model.NumHeads, Model.HeadDim, 1, TextRows, 0, 0, True, Model.RoPETheta);
    FluxRoPEGPU(Ctx, K, Model.NumHeads, Model.HeadDim, 1, TextRows, 0, 0, True, Model.RoPETheta);
    FluxImageRoPEGPU(Ctx, Q, Model, Height, Width, TextRows, Refs);
    FluxImageRoPEGPU(Ctx, K, Model, Height, Width, TextRows, Refs);
    A := Work.New(Int64(Rows) * Hidden);
    Ctx.Attention(A, Q, K, V, Rows, Model.NumHeads, Model.NumHeads, Model.HeadDim, 0);
    Ctx.Kernel(GO_SWIGLU, Rows * Mlp, Gate, Gate, Up, nil, [Mlp, 0]);
    Joined := Work.Join(A, Gate, Rows, Hidden, Mlp);
    Proj := Work.Linear(Joined, GPUWeight(Block.ProjMlpWeight, Block.ProjMlpWeightBF16), Rows, Hidden + Mlp, Hidden);
    Work.Modulate(Proj, Modulation, Rows, Hidden, 1, Hidden * 2, 0, X);
  finally
    Work.Free;
  end;
end;

function FluxForwardGPU(
    const Ctx: TQNNGPUContext;
    var Model: TTransformerFlux;
    const Latent: TQNNGPUTensor;
    const Height, Width: Integer;
    const Refs: TGPUImageRefs;
    const Text: TQNNGPUTensor;
    const TextRows: Integer;
    const Time: Single
): TQNNGPUTensor;
var
  Work: TGPUWorkspace;
  Input, Part, ImageX, TextX, Joined, TEmb, Silu, ImageMod, TextMod, SingleMod, FinalMod, N, OutRows: TQNNGPUTensor;
  ImageRows, TotalImageRows, Rows, Offset, Hidden: Integer;
begin
  Hidden := Model.HiddenSize;
  ImageRows := Height * Width;
  TotalImageRows := ImageRows;
  for var Ref in Refs do
    Inc(TotalImageRows, Ref.H * Ref.W);
  Rows := TotalImageRows + TextRows;
  if (Rows > 65535) or (TextRows < 1) or (Model.HeadDim mod 8 <> 0) or (Hidden <> Model.NumHeads * Model.HeadDim) then
    raise EArgumentException.Create('Invalid resident FLUX geometry');
  Work := TGPUWorkspace.Create(Ctx);
  try
    Input := Work.New(Int64(TotalImageRows) * Model.LatentChannels);
    Part := Work.CHWToRows(Latent, ImageRows, Model.LatentChannels);
    Ctx.Copy(Input, Part, ImageRows * Model.LatentChannels);
    Work.Release(Part);
    Offset := ImageRows;
    for var Ref in Refs do
    begin
      Part := Work.CHWToRows(Ref.Latent, Ref.H * Ref.W, Model.LatentChannels);
      Ctx.Copy(Input, Part, Ref.H * Ref.W * Model.LatentChannels, Offset * Model.LatentChannels);
      Work.Release(Part);
      Inc(Offset, Ref.H * Ref.W);
    end;
    ImageX :=
        Work.Linear(
            Input,
            GPUWeight(Model.ImgInWeight, Model.ImgInWeightBF16),
            TotalImageRows,
            Model.LatentChannels,
            Hidden
        );
    TextX := Work.Linear(Text, GPUWeight(Model.TxtInWeight, Model.TxtInWeightBF16), TextRows, Model.TextDim, Hidden);
    TEmb := Work.Sinusoidal(Time * 1000, Model.TimeEmbed.SincosDim);
    TEmb := Work.Linear(TEmb, Model.TimeEmbed.Fc1Weight, 1, Model.TimeEmbed.SincosDim, Hidden);
    TEmb := Work.Unary(TEmb, Hidden);
    TEmb := Work.Linear(TEmb, Model.TimeEmbed.Fc2Weight, 1, Hidden, Hidden);
    Silu := Work.Unary(TEmb, Hidden);
    ImageMod :=
        Work.Linear(Silu, GPUWeight(Model.AdalnDoubleImgWeight, Model.AdalnDoubleImgWeightBF16), 1, Hidden, Hidden * 6);
    TextMod :=
        Work.Linear(Silu, GPUWeight(Model.AdalnDoubleTxtWeight, Model.AdalnDoubleTxtWeightBF16), 1, Hidden, Hidden * 6);
    SingleMod :=
        Work.Linear(Silu, GPUWeight(Model.AdalnSingleWeight, Model.AdalnSingleWeightBF16), 1, Hidden, Hidden * 3);
    for var I := 0 to Model.NumDoubleLayers - 1 do
    begin
      if not (Model.DoubleBlocks[I].ImgQWeight.IsAssigned or Model.DoubleBlocks[I].ImgQWeightBF16.IsAssigned) then
        Model.DoubleBlocks[I].Load(Model.SfFiles, I, Hidden, Model.MlpHidden, Model.UseBF16, Model.UseMmap, True);
      FluxDoubleGPU(
          Ctx,
          Model,
          Model.DoubleBlocks[I],
          ImageX,
          TextX,
          ImageMod,
          TextMod,
          TotalImageRows,
          TextRows,
          Height,
          Width,
          Refs
      );
      if Assigned(SubstepCallback) then
        SubstepCallback(SUBSTEP_DOUBLE_BLOCK, I, Model.NumDoubleLayers);
    end;
    Joined := Work.Concat(TextX, ImageX, TextRows * Hidden, TotalImageRows * Hidden);
    Work.Release(ImageX);
    Work.Release(TextX);
    for var I := 0 to Model.NumSingleLayers - 1 do
    begin
      if not (Model.SingleBlocks[I].QKVMlpWeight.IsAssigned or Model.SingleBlocks[I].QKVMlpWeightBF16.IsAssigned) then
        Model.SingleBlocks[I].Load(Model.SfFiles, I, Hidden, Model.MlpHidden, Model.UseBF16, Model.UseMmap, True);
      FluxSingleGPU(Ctx, Model, Model.SingleBlocks[I], Joined, SingleMod, Rows, TextRows, Height, Width, Refs);
      if Assigned(SubstepCallback) then
        SubstepCallback(SUBSTEP_SINGLE_BLOCK, I, Model.NumSingleLayers);
    end;
    ImageX := Work.Copy(Joined, ImageRows * Hidden, TextRows * Hidden);
    FinalMod := Work.Linear(Silu, Model.FinalNormWeight, 1, Hidden, Hidden * 2);
    N := Work.Norm(ImageX, Default(TMemoryBlock), Default(TMemoryBlock), ImageRows, Hidden, True);
    N := Work.Modulate(N, FinalMod, ImageRows, Hidden, 0, 0, Hidden);
    OutRows :=
        Work.Linear(
            N,
            GPUWeight(Model.FinalProjWeight, Model.FinalProjWeightBF16),
            ImageRows,
            Hidden,
            Model.LatentChannels
        );
    Result := Work.RowsToCHW(OutRows, ImageRows, Model.LatentChannels);
    Ctx.CheckFinite(Result);
    Work.Keep(Result);
    if Assigned(SubstepCallback) then
      SubstepCallback(SUBSTEP_FINAL_LAYER, 0, 1);
  finally
    Work.Free;
  end;
end;
