function TQwen21Conv.ForwardGPU(
    const Ctx: TQNNGPUContext;
    const Src: TQNNGPUTensor;
    const Height, Width, Stride, Padding: Integer;
    const PadEnd: Boolean
): TQNNGPUTensor;
const
  TILE = 1024;
var
  Col, Projected: TQNNGPUTensor;
  Oh, Ow, Pad, Spatial, Inner, Start, Count: Integer;
begin
  Col := nil;
  Projected := nil;
  if Padding < 0 then
    Pad := Kernel div 2
  else
    Pad := Padding;
  Oh := (Height + 2 * Pad + Ord(PadEnd) - Kernel) div Stride + 1;
  Ow := (Width + 2 * Pad + Ord(PadEnd) - Kernel) div Stride + 1;
  Spatial := Oh * Ow;
  Inner := InputDim * Kernel * Kernel;
  if (Height < 1) or (Width < 1) or (Oh < 1) or (Ow < 1) or (Src.Count < InputDim * Height * Width) then
    raise EArgumentException.Create('Invalid resident convolution geometry');
  Result := Ctx.NewTensor(OutputDim * Spatial);
  try
    Col := Ctx.NewTensor(Min(TILE, Spatial) * Inner);
    Projected := Ctx.NewTensor(Min(TILE, Spatial) * OutputDim);
    Start := 0;
    while Start < Spatial do
    begin
      Count := Min(TILE, Spatial - Start);
      Ctx.Kernel(
          GO_IM2COL,
          Count * Inner,
          Col,
          Src,
          nil,
          nil,
          [Height, Width, InputDim, Kernel, Start, Ow, Stride, Pad]
      );
      Ctx.Linear(Projected, Col, Weight, Default(TMemoryBlock), Default(TMemoryBlock), Count, Inner, OutputDim);
      Ctx.KernelWeights(
          GO_CONV_OUTPUT,
          Count * OutputDim,
          Result,
          Projected,
          Bias,
          Default(TMemoryBlock),
          [OutputDim, Spatial, Start, Ord(Bias.IsAssigned)]
      );
      Inc(Start, Count);
    end;
  finally
    Ctx.Recycle(Col);
    Ctx.Recycle(Projected)
  end;
end;

function TQwen21VAERes.ForwardGPU(
    const Ctx: TQNNGPUContext;
    const Src: TQNNGPUTensor;
    const Height, Width: Integer
): TQNNGPUTensor;
var
  N, C1, N2, Skip: TQNNGPUTensor;
  Spatial: Integer;
begin
  N := nil;
  C1 := nil;
  N2 := nil;
  Skip := nil;
  Spatial := Height * Width;
  try
    N := Ctx.NewTensor(Conv1.InputDim * Spatial);
    Q21GPURMS(Ctx, N, Src, Gamma1, Spatial, Conv1.InputDim, False, True);
    Ctx.Kernel(GO_UNARY, Conv1.InputDim * Spatial, N, N, nil, nil, [0]);
    C1 := Conv1.ForwardGPU(Ctx, N, Height, Width);
    Ctx.Recycle(N);
    N2 := Ctx.NewTensor(Conv2.InputDim * Spatial);
    Q21GPURMS(Ctx, N2, C1, Gamma2, Spatial, Conv2.InputDim, False, True);
    Ctx.Kernel(GO_UNARY, Conv2.InputDim * Spatial, N2, N2, nil, nil, [0]);
    Ctx.Recycle(C1);
    Result := Conv2.ForwardGPU(Ctx, N2, Height, Width);
    Ctx.Recycle(N2);
    if Shortcut.Weight.IsAssigned then
    begin
      Skip := Shortcut.ForwardGPU(Ctx, Src, Height, Width);
      Q21GPUAdd(Ctx, Result, Skip, Conv2.OutputDim * Spatial)
    end
    else
      Q21GPUAdd(Ctx, Result, Src, Conv2.OutputDim * Spatial);
  finally
    Ctx.Recycle(N);
    Ctx.Recycle(C1);
    Ctx.Recycle(N2);
    Ctx.Recycle(Skip)
  end;
end;

function TQwen21VAEUp.ForwardGPU(
    const Ctx: TQNNGPUContext;
    const Src: TQNNGPUTensor;
    var Height, Width: Integer
): TQNNGPUTensor;
var
  X, Next, Resized: TQNNGPUTensor;
  InChannels, OutChannels, Spatial: Integer;
begin
  X := nil;
  Next := nil;
  Resized := nil;
  InChannels := Blocks[0].Conv1.InputDim;
  OutChannels := Blocks[2].Conv2.OutputDim;
  Spatial := Height * Width;
  try
    X := Blocks[0].ForwardGPU(Ctx, Src, Height, Width);
    for var J := 1 to 2 do
    begin
      Next := Blocks[J].ForwardGPU(Ctx, X, Height, Width);
      Ctx.Recycle(X);
      X := Next;
      Next := nil
    end;
    if not Upsample.Weight.IsAssigned then
    begin
      Result := X;
      X := nil;
      Exit
    end;
    Resized := Ctx.NewTensor(OutChannels * Spatial * 4);
    Ctx.Kernel(GO_NEAREST, OutChannels * Spatial * 4, Resized, X, nil, nil, [Height, Width]);
    Result := Upsample.ForwardGPU(Ctx, Resized, Height * 2, Width * 2);
    Ctx.Kernel(
        GO_UP_RESIDUAL,
        OutChannels * Spatial * 4,
        Result,
        Src,
        nil,
        nil,
        [Height, Width, InChannels, OutChannels, TemporalFactor]
    );
    Height := Height * 2;
    Width := Width * 2;
  finally
    Ctx.Recycle(X);
    Ctx.Recycle(Next);
    Ctx.Recycle(Resized)
  end;
end;

function TQwen21VAEUp.ForwardTiledGPU(const Ctx: TQNNGPUContext;
  const Src: TQNNGPUTensor; var Height, Width: Integer; const TileSize: Integer): TQNNGPUTensor;
const
  // Six 3x3 convolutions, followed by nearest 2x + 3x3: radius <= 7.
  // Channel RMS is local to a pixel; attention has already run globally.
  HALO = 8;
var
  Tile, Decoded: TQNNGPUTensor;
  InCh, OutCh, Scale, X, Y, Left, Top, TW, TH, CW, CH, DW, DH: Integer;
begin
  if (TileSize = 0) or ((Height <= TileSize) and (Width <= TileSize)) then
    Exit(ForwardGPU(Ctx, Src, Height, Width));
  if TileSize < 16 then
    raise EArgumentOutOfRangeException.Create('VAE tile core must be at least 16 pixels');
  InCh := Blocks[0].Conv1.InputDim;
  OutCh := Blocks[2].Conv2.OutputDim;
  Scale := 1 + Ord(Upsample.Weight.IsAssigned);
  Result := Ctx.NewTensor(OutCh * Height * Width * Scale * Scale);
  Tile := nil;
  Decoded := nil;
  try
    Y := 0;
    while Y < Height do
    begin
      CH := Min(TileSize, Height - Y);
      Top := Max(0, Y - HALO);
      TH := Min(Height, Y + CH + HALO) - Top;
      X := 0;
      while X < Width do
      begin
        CW := Min(TileSize, Width - X);
        Left := Max(0, X - HALO);
        TW := Min(Width, X + CW + HALO) - Left;
        Tile := Ctx.NewTensor(InCh * TH * TW);
        Ctx.CopyCHWRect(Tile, Src, InCh, Width, Height, TW, TH, Left, Top, 0, 0, TW, TH);
        DW := TW;
        DH := TH;
        Decoded := ForwardGPU(Ctx, Tile, DH, DW);
        Ctx.CopyCHWRect(Result, Decoded, OutCh, DW, DH, Width * Scale, Height * Scale,
          (X - Left) * Scale, (Y - Top) * Scale, X * Scale, Y * Scale, CW * Scale, CH * Scale);
        Ctx.Recycle(Decoded);
        Ctx.Recycle(Tile);
        Inc(X, CW);
      end;
      Inc(Y, CH);
    end;
    Height := Height * Scale;
    Width := Width * Scale;
  except
    Ctx.Recycle(Decoded);
    Ctx.Recycle(Tile);
    Ctx.Recycle(Result);
    raise;
  end;
end;

function TQwen21VAEDown.ForwardGPU(
    const Ctx: TQNNGPUContext;
    const Src: TQNNGPUTensor;
    var Height, Width: Integer
): TQNNGPUTensor;
var
  X, Next: TQNNGPUTensor;
  OldH, OldW, InChannels, OutChannels: Integer;
begin
  X := nil;
  Next := nil;
  OldH := Height;
  OldW := Width;
  InChannels := Blocks[0].Conv1.InputDim;
  OutChannels := Blocks[1].Conv2.OutputDim;
  try
    X := Blocks[0].ForwardGPU(Ctx, Src, Height, Width);
    Next := Blocks[1].ForwardGPU(Ctx, X, Height, Width);
    Ctx.Recycle(X);
    X := Next;
    Next := nil;
    if Downsample.Weight.IsAssigned then
    begin
      Next := Downsample.ForwardGPU(Ctx, X, Height, Width, 2, 0, True);
      Ctx.Recycle(X);
      X := Next;
      Next := nil;
      Height := Height div 2;
      Width := Width div 2;
    end;
    Ctx.Kernel(
        GO_DOWN_RESIDUAL,
        OutChannels * Height * Width,
        X,
        Src,
        nil,
        nil,
        [OldH, OldW, InChannels, OutChannels, SpatialFactor, TemporalFactor]
    );
    Result := X;
    X := nil;
  finally
    Ctx.Recycle(X);
    Ctx.Recycle(Next)
  end;
end;

function TQwen21VAE.AttentionGPU(
    const Ctx: TQNNGPUContext;
    const Src: TQNNGPUTensor;
    const Height, Width: Integer;
    const Encoder: Boolean
): TQNNGPUTensor;
var
  Norm, Qkv, Q, K, V, A, Chw: TQNNGPUTensor;
  QkvConv, ProjConv: ^TQwen21Conv;
  Gamma: TMemoryBlock;
  Spatial, Channels: Integer;
begin
  Norm := nil;
  Qkv := nil;
  Q := nil;
  K := nil;
  V := nil;
  A := nil;
  Chw := nil;
  if Encoder then
  begin
    QkvConv := @FEncQKV;
    ProjConv := @FEncProj;
    Gamma := FEncAttentionGamma
  end
  else
  begin
    QkvConv := @FQKV;
    ProjConv := @FProj;
    Gamma := FAttentionGamma
  end;
  Spatial := Height * Width;
  Channels := QkvConv.InputDim;
  try
    Norm := Ctx.NewTensor(Spatial * Channels);
    Q21GPURMS(Ctx, Norm, Src, Gamma, Spatial, Channels, False, True);
    Qkv := QkvConv.ForwardGPU(Ctx, Norm, Height, Width);
    Ctx.Recycle(Norm);
    Q := Ctx.NewTensor(Spatial * Channels);
    K := Ctx.NewTensor(Spatial * Channels);
    V := Ctx.NewTensor(Spatial * Channels);
    A := Ctx.NewTensor(Spatial * Channels);
    Ctx.Kernel(GO_CHW_TO_ROWS, Spatial * Channels, Q, Qkv, nil, nil, [Spatial, Channels, 0]);
    Ctx.Kernel(GO_CHW_TO_ROWS, Spatial * Channels, K, Qkv, nil, nil, [Spatial, Channels, Channels]);
    Ctx.Kernel(GO_CHW_TO_ROWS, Spatial * Channels, V, Qkv, nil, nil, [Spatial, Channels, 2 * Channels]);
    Ctx.Recycle(Qkv);
    Ctx.Attention(A, Q, K, V, Spatial, 1, 1, Channels, 0);
    Chw := Ctx.NewTensor(Spatial * Channels);
    Ctx.Kernel(GO_ROWS_TO_CHW, Spatial * Channels, Chw, A, nil, nil, [Spatial, Channels]);
    Result := ProjConv.ForwardGPU(Ctx, Chw, Height, Width);
    Q21GPUAdd(Ctx, Result, Src, Spatial * Channels);
  finally
    Ctx.Recycle(Norm);
    Ctx.Recycle(Qkv);
    Ctx.Recycle(Q);
    Ctx.Recycle(K);
    Ctx.Recycle(V);
    Ctx.Recycle(A);
    Ctx.Recycle(Chw)
  end;
end;

function TQwen21VAE.EncodeGPU(
    const Ctx: TQNNGPUContext;
    const Pixels: TQNNGPUTensor;
    const ImageHeight, ImageWidth: Integer
): TQNNGPUTensor;
var
  X, Next, N: TQNNGPUTensor;
  Height, Width: Integer;
  Mean, Std: TMemoryBlock;
begin
  X := nil;
  Next := nil;
  N := nil;
  LoadEncoder;
  Height := ImageHeight;
  Width := ImageWidth;
  try
    Mean := TMemoryBlock.Create([64], 'GPU-latent-mean');
    Std := TMemoryBlock.Create([64], 'GPU-latent-std');
    Move(FMean[0], System.PSingle(Mean)^, 64 * 4);
    Move(FStd[0], System.PSingle(Std)^, 64 * 4);
    Mean.MarkImmutable;
    Std.MarkImmutable;
    X := Ctx.NewTensor(4 * Height * Width);
    Ctx.Kernel(GO_PIXEL_INPUT, 4 * Height * Width, X, Pixels, nil, nil, [Height * Width]);
    Next := FEncIn.ForwardGPU(Ctx, X, Height, Width);
    Ctx.Recycle(X);
    X := Next;
    Next := nil;
    for var I := 0 to 4 do
    begin
      Next := FDowns[I].ForwardGPU(Ctx, X, Height, Width);
      Ctx.Recycle(X);
      X := Next;
      Next := nil;
      Ctx.CheckFinite(X)
    end;
    Next := FEncMid1.ForwardGPU(Ctx, X, Height, Width);
    Ctx.Recycle(X);
    X := Next;
    Next := nil;
    Next := AttentionGPU(Ctx, X, Height, Width, True);
    Ctx.Recycle(X);
    X := Next;
    Next := nil;
    Next := FEncMid2.ForwardGPU(Ctx, X, Height, Width);
    Ctx.Recycle(X);
    X := Next;
    Next := nil;
    N := Ctx.NewTensor(FEncOut.InputDim * Height * Width);
    Q21GPURMS(Ctx, N, X, FEncNorm, Height * Width, FEncOut.InputDim, False, True);
    Ctx.Kernel(GO_UNARY, FEncOut.InputDim * Height * Width, N, N, nil, nil, [0]);
    Ctx.Recycle(X);
    X := FEncOut.ForwardGPU(Ctx, N, Height, Width);
    Next := FQuant.ForwardGPU(Ctx, X, Height, Width);
    Result := Ctx.NewTensor(Height * Width * 64);
    Ctx.KernelWeights(GO_LATENT_NORMALIZE, Height * Width * 64, Result, Next, Mean, Std, [Height * Width]);
    Ctx.CheckFinite(Result);
  finally
    Ctx.Recycle(X);
    Ctx.Recycle(Next);
    Ctx.Recycle(N);
    Mean.Free;
    Std.Free
  end;
end;

function TQwen21VAE.DecodeGPU(
    const Ctx: TQNNGPUContext;
    const Latents: TQNNGPUTensor;
    const Height, Width: Integer;
    const TileSize: Integer
): TQNNGPUTensor;
var
  X, Next, N: TQNNGPUTensor;
  H, W, Core, TileH, TileW, X0, Y0, Left, Top, TH, TW: Integer;
  Tile, TileN, TileOut: TQNNGPUTensor;
  Mean, Std: TMemoryBlock;
begin
  if (TileSize <> 0) and (TileSize < 16) then
    raise EArgumentOutOfRangeException.Create('VAE tile core must be at least 16 pixels');
  Ctx.TrimActivations;
  Core := TileSize;
  H := Height;
  W := Width;
  X := nil;
  Next := nil;
  N := nil;
  Tile := nil; TileN := nil; TileOut := nil;
  try
    Mean := TMemoryBlock.Create([64], 'GPU-latent-mean');
    Std := TMemoryBlock.Create([64], 'GPU-latent-std');
    Move(FMean[0], System.PSingle(Mean)^, 64 * 4);
    Move(FStd[0], System.PSingle(Std)^, 64 * 4);
    Mean.MarkImmutable;
    Std.MarkImmutable;
    X := Ctx.NewTensor(64 * H * W);
    Ctx.KernelWeights(GO_LATENT_CHW, 64 * H * W, X, Latents, Mean, Std, [H * W]);
    Next := FPost.ForwardGPU(Ctx, X, H, W);
    Ctx.Recycle(X);
    X := Next;
    Next := nil;
    Next := FIn.ForwardGPU(Ctx, X, H, W);
    Ctx.Recycle(X);
    X := Next;
    Next := nil;
    Next := FMid1.ForwardGPU(Ctx, X, H, W);
    Ctx.Recycle(X);
    X := Next;
    Next := nil;
    Next := AttentionGPU(Ctx, X, H, W);
    Ctx.Recycle(X);
    X := Next;
    Next := nil;
    Next := FMid2.ForwardGPU(Ctx, X, H, W);
    Ctx.Recycle(X);
    X := Next;
    Next := nil;
    for var I := 0 to 4 do
    begin
      // A stage only borrows X. Previous stages' cached full-size maps
      // must not inflate the working set of its small overlapping tiles.
      Ctx.TrimActivations;
      // Bound the per-tile working set when other GPU clients consume VRAM.
      if Core > 32 then
        while (Core > 32) and (UInt64(Core + 16) * UInt64(Core + 16)
          * UInt64(FUps[I].Blocks[0].Conv1.InputDim) * 32 > Ctx.ActivationBudget div 2) do
          Core := Max(32, Core div 2);
      Next := FUps[I].ForwardTiledGPU(Ctx, X, H, W, Core);
      Ctx.Recycle(X);
      X := Next;
      Next := nil;
      Ctx.CheckFinite(X);
      if Assigned(FStatus) then
        FStatus(Format('GPU VAE block %d/5: %dx%d', [I + 1, W, H]));
    end;
    Ctx.TrimActivations;
    if (Core = 0) or ((H <= Core) and (W <= Core)) then
    begin
      N := Ctx.NewTensor(FOut.InputDim * H * W);
      Q21GPURMS(Ctx, N, X, FOutGamma, H * W, FOut.InputDim, False, True);
      Ctx.Kernel(GO_UNARY, N.Count, N, N, nil, nil, [0]);
      Ctx.Recycle(X);
      X := FOut.ForwardGPU(Ctx, N, H, W);
      Ctx.Recycle(N);
    end
    else
    begin
      // The final channel norm + convolution needs only a one-pixel halo.
      Next := Ctx.NewTensor(FOut.OutputDim * H * W);
      Y0 := 0;
      while Y0 < H do
      begin
        TileH := Min(Core, H - Y0);
        Top := Max(0, Y0 - 1);
        TH := Min(H, Y0 + TileH + 1) - Top;
        X0 := 0;
        while X0 < W do
        begin
          TileW := Min(Core, W - X0);
          Left := Max(0, X0 - 1);
          TW := Min(W, X0 + TileW + 1) - Left;
          Tile := Ctx.NewTensor(FOut.InputDim * TH * TW);
          TileN := Ctx.NewTensor(Tile.Count);
          Ctx.CopyCHWRect(Tile, X, FOut.InputDim, W, H, TW, TH, Left, Top, 0, 0, TW, TH);
          Q21GPURMS(Ctx, TileN, Tile, FOutGamma, TH * TW, FOut.InputDim, False, True);
          Ctx.Kernel(GO_UNARY, TileN.Count, TileN, TileN, nil, nil, [0]);
          TileOut := FOut.ForwardGPU(Ctx, TileN, TH, TW);
          Ctx.CopyCHWRect(Next, TileOut, FOut.OutputDim, TW, TH, W, H,
            X0 - Left, Y0 - Top, X0, Y0, TileW, TileH);
          Ctx.Recycle(TileOut); Ctx.Recycle(TileN); Ctx.Recycle(Tile);
          Inc(X0, TileW);
        end;
        Inc(Y0, TileH);
      end;
      Ctx.Recycle(X);
      X := Next;
      Next := nil;
    end;
    Ctx.CheckFinite(X);
    Result := Ctx.NewTensor(H * W);
    Ctx.Kernel(GO_PIXEL_OUTPUT, H * W, Result, X, nil, nil, [H * W]);
  finally
    Ctx.Recycle(X);
    Ctx.Recycle(Next);
    Ctx.Recycle(N);
    Ctx.Recycle(Tile); Ctx.Recycle(TileN); Ctx.Recycle(TileOut);
    Mean.Free;
    Std.Free
  end;
end;
