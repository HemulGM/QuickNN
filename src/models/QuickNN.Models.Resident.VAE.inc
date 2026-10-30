function VAEResGPU(
    const Ctx: TQNNGPUContext;
    const Block: TVAEResBlock;
    const Src: TQNNGPUTensor;
    const Height, Width, Groups: Integer
): TQNNGPUTensor;
var
  Work: TGPUWorkspace;
  N, Conv, Skip: TQNNGPUTensor;
begin
  Work := TGPUWorkspace.Create(Ctx);
  try
    N := Work.GroupNorm(Src, Block.Norm1Weight, Block.Norm1Bias, Block.InChannels, Height * Width, Groups);
    Ctx.Kernel(GO_UNARY, N.Count, N, N, nil, nil, [0]);
    Conv := Work.Conv(N, Block.Conv1Weight, Block.Conv1Bias, Height, Width, Block.InChannels, Block.OutChannels, 3);
    Work.Release(N);
    N := Work.GroupNorm(Conv, Block.Norm2Weight, Block.Norm2Bias, Block.OutChannels, Height * Width, Groups);
    Work.Release(Conv);
    Ctx.Kernel(GO_UNARY, N.Count, N, N, nil, nil, [0]);
    Result := Work.Conv(N, Block.Conv2Weight, Block.Conv2Bias, Height, Width, Block.OutChannels, Block.OutChannels, 3);
    Work.Release(N);
    if Block.InChannels = Block.OutChannels then
      Work.Add(Result, Src, Height * Width * Block.OutChannels)
    else
    begin
      Skip :=
          Work.Conv(Src, Block.SkipWeight, Block.SkipBias, Height, Width, Block.InChannels, Block.OutChannels, 1, 1, 0);
      Work.Add(Result, Skip, Height * Width * Block.OutChannels);
    end;
    Work.Keep(Result);
  finally
    Work.Free;
  end;
end;

function VAEAttentionGPU(
    const Ctx: TQNNGPUContext;
    const Block: TVAEAttnBlock;
    const Src: TQNNGPUTensor;
    const Height, Width, Groups: Integer
): TQNNGPUTensor;
var
  Work: TGPUWorkspace;
  N, Q, K, V, A, Chw: TQNNGPUTensor;
  Ch, Spatial: Integer;
begin
  Work := TGPUWorkspace.Create(Ctx);
  try
    Ch := Block.Channels;
    Spatial := Height * Width;
    N := Work.GroupNorm(Src, Block.NormWeight, Block.NormBias, Ch, Spatial, Groups);
    Q := Work.Conv(N, Block.QWeight, Block.QBias, Height, Width, Ch, Ch, 1, 1, 0);
    K := Work.Conv(N, Block.KWeight, Block.KBias, Height, Width, Ch, Ch, 1, 1, 0);
    V := Work.Conv(N, Block.VWeight, Block.VBias, Height, Width, Ch, Ch, 1, 1, 0);
    Q := Work.CHWToRows(Q, Spatial, Ch);
    K := Work.CHWToRows(K, Spatial, Ch);
    V := Work.CHWToRows(V, Spatial, Ch);
    A := Work.New(Int64(Spatial) * Ch);
    Ctx.Attention(A, Q, K, V, Spatial, 1, 1, Ch, 0);
    Chw := Work.RowsToCHW(A, Spatial, Ch);
    Result := Work.Conv(Chw, Block.OutWeight, Block.OutBias, Height, Width, Ch, Ch, 1, 1, 0);
    Work.Add(Result, Src, Spatial * Ch);
    Work.Keep(Result);
  finally
    Work.Free;
  end;
end;

function VAEDecodeGPU(
    const Ctx: TQNNGPUContext;
    const Model: TVAE;
    const Latent: TQNNGPUTensor;
    const Height, Width: Integer
): TQNNGPUTensor;
var
  Work: TGPUWorkspace;
  X, Next, N: TQNNGPUTensor;
  H, W, Ch, BlockIndex, UpIndex, Progress, Total: Integer;

  procedure Replace(const Tensor: TQNNGPUTensor);
  begin
    Work.Release(X);
    X := Tensor;
  end;

  procedure Notify;
  begin
    if Assigned(VAEProgressCallback) then
      VAEProgressCallback(Progress, Total);
    Inc(Progress);
  end;

begin
  Work := TGPUWorkspace.Create(Ctx);
  try
    X := Work.Copy(Latent, Height * Width * Model.LatentChannels);
    if Model.ScalingFactor <> 0 then
      Ctx.Kernel(
          GO_SCALE,
          X.Count,
          X,
          X,
          nil,
          nil,
          [QNNGPUFloatBits(1 / Model.ScalingFactor), QNNGPUFloatBits(Model.ShiftFactor)]
      )
    else
      Ctx.KernelWeights(
          GO_LATENT_BN,
          X.Count,
          X,
          X,
          Model.BnMean,
          Model.BnVar,
          [Height * Width, 1, QNNGPUFloatBits(Model.Eps)]
      );
    H := Height * 2;
    W := Width * 2;
    Replace(Work.Patch(X, H, W, Model.ZChannels, 2, 3));
    if Model.PostQuantConvWeight.IsAssigned then
      Replace(
          Work.Conv(
              X,
              Model.PostQuantConvWeight,
              Model.PostQuantConvBias,
              H,
              W,
              Model.ZChannels,
              Model.ZChannels,
              1,
              1,
              0
          )
      );
    Ch := Model.BaseChannels * Model.ChMult[3];
    Replace(Work.Conv(X, Model.DecConvInWeight, Model.DecConvInBias, H, W, Model.ZChannels, Ch, 3));
    Progress := 0;
    Total := 3 + 4 * (Model.NumResBlocks + 1);
    Next := Work.Own(VAEResGPU(Ctx, Model.DecMidBlock1, X, H, W, Model.NumGroups));
    Replace(Next);
    Notify;
    Replace(Work.Own(VAEAttentionGPU(Ctx, Model.DecMidAttn, X, H, W, Model.NumGroups)));
    Notify;
    Replace(Work.Own(VAEResGPU(Ctx, Model.DecMidBlock2, X, H, W, Model.NumGroups)));
    Notify;
    BlockIndex := 0;
    UpIndex := 0;
    for var Level := 3 downto 0 do
    begin
      Ch := Model.BaseChannels * Model.ChMult[Level];
      for var R := 0 to Model.NumResBlocks do
      begin
        Replace(Work.Own(VAEResGPU(Ctx, Model.DecUpBlocks[BlockIndex], X, H, W, Model.NumGroups)));
        Inc(BlockIndex);
        Notify;
      end;
      if Level > 0 then
      begin
        Next := Work.New(Int64(H) * W * Ch * 4);
        Ctx.Kernel(GO_NEAREST, Next.Count, Next, X, nil, nil, [H, W]);
        Replace(Next);
        H := H * 2;
        W := W * 2;
        Replace(
            Work.Conv(X, Model.DecUpsample[UpIndex].ConvWeight, Model.DecUpsample[UpIndex].ConvBias, H, W, Ch, Ch, 3)
        );
        Inc(UpIndex);
      end;
    end;
    N := Work.GroupNorm(X, Model.DecNormOutWeight, Model.DecNormOutBias, Model.BaseChannels, H * W, Model.NumGroups);
    N := Work.Unary(N, H * W * Model.BaseChannels);
    Result := Work.Conv(N, Model.DecConvOutWeight, Model.DecConvOutBias, H, W, Model.BaseChannels, 3, 3);
    Ctx.CheckFinite(Result);
    Work.Keep(Result);
  finally
    Work.Free;
  end;
end;

function VAEEncodeGPU(
    const Ctx: TQNNGPUContext;
    const Model: TVAE;
    const Pixels: TQNNGPUTensor;
    const Height, Width: Integer
): TQNNGPUTensor;
var
  Work: TGPUWorkspace;
  X, N: TQNNGPUTensor;
  H, W, Ch, BlockIndex, DownIndex, Progress, Total: Integer;

  procedure Replace(const Tensor: TQNNGPUTensor);
  begin
    Work.Release(X);
    X := Tensor;
  end;

  procedure Notify;
  begin
    if Assigned(VAEProgressCallback) then
      VAEProgressCallback(Progress, Total);
    Inc(Progress);
  end;

begin
  if (Height mod 16 <> 0) or (Width mod 16 <> 0) then
    raise EArgumentException.Create('Resident VAE input dimensions must be divisible by 16');
  Work := TGPUWorkspace.Create(Ctx);
  try
    H := Height;
    W := Width;
    X := Work.Conv(Pixels, Model.EncConvInWeight, Model.EncConvInBias, H, W, 3, Model.BaseChannels, 3);
    BlockIndex := 0;
    DownIndex := 0;
    Progress := 0;
    Total := 4 * Model.NumResBlocks + 3;
    for var Level := 0 to 3 do
    begin
      Ch := Model.BaseChannels * Model.ChMult[Level];
      for var R := 0 to Model.NumResBlocks - 1 do
      begin
        Replace(Work.Own(VAEResGPU(Ctx, Model.EncDownBlocks[BlockIndex], X, H, W, Model.NumGroups)));
        Inc(BlockIndex);
        Notify;
      end;
      if Level < 3 then
      begin
        Replace(
            Work.Conv(
                X,
                Model.EncDownsample[DownIndex].ConvWeight,
                Model.EncDownsample[DownIndex].ConvBias,
                H,
                W,
                Ch,
                Ch,
                3,
                2,
                0,
                True
            )
        );
        H := H div 2;
        W := W div 2;
        Inc(DownIndex);
      end;
    end;
    Replace(Work.Own(VAEResGPU(Ctx, Model.EncMidBlock1, X, H, W, Model.NumGroups)));
    Notify;
    Replace(Work.Own(VAEAttentionGPU(Ctx, Model.EncMidAttn, X, H, W, Model.NumGroups)));
    Notify;
    Replace(Work.Own(VAEResGPU(Ctx, Model.EncMidBlock2, X, H, W, Model.NumGroups)));
    Notify;
    Ch := Model.BaseChannels * Model.ChMult[3];
    N := Work.GroupNorm(X, Model.EncNormOutWeight, Model.EncNormOutBias, Ch, H * W, Model.NumGroups);
    N := Work.Unary(N, H * W * Ch);
    Replace(Work.Conv(N, Model.EncConvOutWeight, Model.EncConvOutBias, H, W, Ch, Model.ZChannels * 2, 3));
    if Model.QuantConvWeight.IsAssigned then
      Replace(
          Work.Conv(
              X,
              Model.QuantConvWeight,
              Model.QuantConvBias,
              H,
              W,
              Model.ZChannels * 2,
              Model.ZChannels * 2,
              1,
              1,
              0
          )
      );
    N := Work.Copy(X, H * W * Model.ZChannels);
    Result := Work.Patch(N, H, W, Model.ZChannels, 2, 2);
    if Model.ScalingFactor <> 0 then
      Ctx.Kernel(
          GO_SCALE,
          Result.Count,
          Result,
          Result,
          nil,
          nil,
          [QNNGPUFloatBits(Model.ScalingFactor), QNNGPUFloatBits(-Model.ShiftFactor * Model.ScalingFactor)]
      )
    else
      Ctx.KernelWeights(
          GO_LATENT_BN,
          Result.Count,
          Result,
          Result,
          Model.BnMean,
          Model.BnVar,
          [(H div 2) * (W div 2), 0, QNNGPUFloatBits(Model.Eps)]
      );
    Ctx.CheckFinite(Result);
    Work.Keep(Result);
  finally
    Work.Free;
  end;
end;
