procedure ZImageBlockGPU(
    const Ctx: TQNNGPUContext;
    const Model: TTransformerZI;
    const Block: TBlockZI;
    const X, TimeEmbedding, Positions: TQNNGPUTensor;
    const Rows: Integer
);
var
  Work: TGPUWorkspace;
  Modulation, N, Q, K, V, A, Projected, Ffn: TQNNGPUTensor;
  Hidden: Integer;
begin
  Work := TGPUWorkspace.Create(Ctx);
  try
    Hidden := Model.Dim;
    Modulation := nil;
    if Block.AdalnWeight.IsAssigned then
      Modulation := Work.Linear(TimeEmbedding, Block.AdalnWeight, 1, Model.AdalnDim, Hidden * 4, Block.AdalnBias);
    N := Work.Norm(X, Block.AttnNorm1, Default(TMemoryBlock), Rows, Hidden);
    if Modulation <> nil then
      N := Work.Modulate(N, Modulation, Rows, Hidden, 2, 0);
    Q := Work.Linear(N, Block.AttnQWeight, Rows, Hidden, Hidden);
    K := Work.Linear(N, Block.AttnKWeight, Rows, Hidden, Hidden);
    V := Work.Linear(N, Block.AttnVWeight, Rows, Hidden, Hidden);
    Q := Work.Norm(Q, Block.AttnNormQ, Default(TMemoryBlock), Rows * Model.NHeads, Model.HeadDim);
    K := Work.Norm(K, Block.AttnNormK, Default(TMemoryBlock), Rows * Model.NHeads, Model.HeadDim);
    Ctx.Kernel(
        GO_ZI_ROPE,
        Rows * Hidden div 2,
        Q,
        Q,
        Positions,
        nil,
        [
            Model.HeadDim,
            Model.NHeads,
            Model.AxesDims[0],
            Model.AxesDims[1],
            Model.AxesDims[2],
            QNNGPUFloatBits(Model.RoPETheta)
        ]
    );
    Ctx.Kernel(
        GO_ZI_ROPE,
        Rows * Hidden div 2,
        K,
        K,
        Positions,
        nil,
        [
            Model.HeadDim,
            Model.NHeads,
            Model.AxesDims[0],
            Model.AxesDims[1],
            Model.AxesDims[2],
            QNNGPUFloatBits(Model.RoPETheta)
        ]
    );
    A := Work.New(Int64(Rows) * Hidden);
    Ctx.Attention(A, Q, K, V, Rows, Model.NHeads, Model.NHeads, Model.HeadDim, 0);
    Projected := Work.Linear(A, Block.AttnOutWeight, Rows, Hidden, Hidden);
    N := Work.Norm(Projected, Block.AttnNorm2, Default(TMemoryBlock), Rows, Hidden);
    if Modulation <> nil then
      Work.Modulate(N, Modulation, Rows, Hidden, 3, Hidden, 0, X)
    else
      Work.Add(X, N, Rows * Hidden);
    N := Work.Norm(X, Block.FfnNorm1, Default(TMemoryBlock), Rows, Hidden);
    if Modulation <> nil then
      N := Work.Modulate(N, Modulation, Rows, Hidden, 2, Hidden * 2);
    Ffn := Work.SwiGLU(N, Block.FfnW1, Block.FfnW3, Block.FfnW2, Rows, Hidden, Model.FfnDim);
    N := Work.Norm(Ffn, Block.FfnNorm2, Default(TMemoryBlock), Rows, Hidden);
    if Modulation <> nil then
      Work.Modulate(N, Modulation, Rows, Hidden, 3, Hidden * 3, 0, X)
    else
      Work.Add(X, N, Rows * Hidden);
  finally
    Work.Free;
  end;
end;

function ZImageForwardGPU(
    const Ctx: TQNNGPUContext;
    var Model: TTransformerZI;
    const Latent: TQNNGPUTensor;
    const Height, Width: Integer;
    const Text: TQNNGPUTensor;
    const TextRows: Integer;
    const Time: Single
): TQNNGPUTensor;
var
  Work: TGPUWorkspace;
  Patch, ImageX, TextX, TimeEmbedding, ImgPos, CapPos, JointPos, X, ImageOut, N, Scale, Final: TQNNGPUTensor;
  ImageRows, ImagePadded, TextPadded, Rows, Feature, Hidden: Integer;
begin
  Feature := Model.PatchSize * Model.PatchSize * Model.InChannels;
  Hidden := Model.Dim;
  ImageRows := (Height div Model.PatchSize) * (Width div Model.PatchSize);
  ImagePadded := (ImageRows + 31) div 32 * 32;
  TextPadded := (TextRows + 31) div 32 * 32;
  Rows := ImagePadded + TextPadded;
  if (TextRows < 1)
      or (ImageRows < 1)
      or (Rows > 65535)
      or (Model.AxesDims[0] + Model.AxesDims[1] + Model.AxesDims[2] <> Model.HeadDim) then
    raise EArgumentException.Create('Invalid resident Z-Image geometry');
  Work := TGPUWorkspace.Create(Ctx);
  try
    TimeEmbedding := Work.Sinusoidal(Time * 1000, 256);
    TimeEmbedding := Work.Linear(TimeEmbedding, Model.TEmbMlp0Weight, 1, 256, Model.TEmbMidSize, Model.TEmbMlp0Bias);
    TimeEmbedding := Work.Unary(TimeEmbedding, Model.TEmbMidSize);
    TimeEmbedding :=
        Work.Linear(TimeEmbedding, Model.TEmbMlp2Weight, 1, Model.TEmbMidSize, Model.AdalnDim, Model.TEmbMlp2Bias);
    Patch := Work.Patch(Latent, Height, Width, Model.InChannels, Model.PatchSize, 0);
    ImageX := Work.Linear(Patch, Model.XEmbWeight, ImageRows, Feature, Hidden, Model.XEmbBias);
    ImageX := Work.PadRows(ImageX, Model.XPadToken, ImageRows, ImagePadded, Hidden);
    N := Work.Norm(Text, Model.CapEmbNorm, Default(TMemoryBlock), TextRows, Model.CapFeatDim);
    TextX := Work.Linear(N, Model.CapEmbLinearW, TextRows, Model.CapFeatDim, Hidden, Model.CapEmbLinearB);
    TextX := Work.PadRows(TextX, Model.CapPadToken, TextRows, TextPadded, Hidden);
    ImgPos := Work.New(Int64(ImagePadded) * 3);
    CapPos := Work.New(Int64(TextPadded) * 3);
    JointPos := Work.New(Int64(Rows) * 3);
    Ctx.Kernel(
        GO_ZI_POSITION,
        ImagePadded * 3,
        ImgPos,
        nil,
        nil,
        nil,
        [ImageRows, ImagePadded, TextPadded, Width div Model.PatchSize, 0]
    );
    Ctx.Kernel(
        GO_ZI_POSITION,
        TextPadded * 3,
        CapPos,
        nil,
        nil,
        nil,
        [ImageRows, ImagePadded, TextPadded, Width div Model.PatchSize, 1]
    );
    Ctx.Kernel(
        GO_ZI_POSITION,
        Rows * 3,
        JointPos,
        nil,
        nil,
        nil,
        [ImageRows, ImagePadded, TextPadded, Width div Model.PatchSize, 2]
    );
    for var I := 0 to Model.NRefiner - 1 do
    begin
      ZImageBlockGPU(Ctx, Model, Model.NoiseRefiner[I], ImageX, TimeEmbedding, ImgPos, ImagePadded);
      if Assigned(SubstepCallback) then
        SubstepCallback(SUBSTEP_DOUBLE_BLOCK, I, Model.NRefiner * 2);
    end;
    for var I := 0 to Model.NRefiner - 1 do
    begin
      ZImageBlockGPU(Ctx, Model, Model.ContextRefiner[I], TextX, nil, CapPos, TextPadded);
      if Assigned(SubstepCallback) then
        SubstepCallback(SUBSTEP_DOUBLE_BLOCK, Model.NRefiner + I, Model.NRefiner * 2);
    end;
    X := Work.Concat(ImageX, TextX, ImagePadded * Hidden, TextPadded * Hidden);
    Work.Release(ImageX);
    Work.Release(TextX);
    for var I := 0 to Model.NLayers - 1 do
    begin
      ZImageBlockGPU(Ctx, Model, Model.Layers[I], X, TimeEmbedding, JointPos, Rows);
      if Assigned(SubstepCallback) then
        SubstepCallback(SUBSTEP_SINGLE_BLOCK, I, Model.NLayers);
    end;
    ImageOut := Work.Copy(X, ImageRows * Hidden);
    Scale := Work.Unary(TimeEmbedding, Model.AdalnDim);
    Scale := Work.Linear(Scale, Model.FinalLayer.AdalnWeight, 1, Model.AdalnDim, Hidden, Model.FinalLayer.AdalnBias);
    N := Work.Norm(ImageOut, Default(TMemoryBlock), Default(TMemoryBlock), ImageRows, Hidden, True);
    N := Work.Modulate(N, Scale, ImageRows, Hidden, 2, 0);
    Final := Work.Linear(N, Model.FinalLayer.LinearWeight, ImageRows, Hidden, Feature, Model.FinalLayer.LinearBias);
    Result := Work.Patch(Final, Height, Width, Model.InChannels, Model.PatchSize, 1);
    Ctx.CheckFinite(Result);
    Work.Keep(Result);
    if Assigned(SubstepCallback) then
      SubstepCallback(SUBSTEP_FINAL_LAYER, 0, 1);
  finally
    Work.Free;
  end;
end;
