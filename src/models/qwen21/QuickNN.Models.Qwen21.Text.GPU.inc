function TQwen21TextEncoder.EncodePromptGPU(
    const Ctx: TQNNGPUContext;
    const Tokenizer: TQwen21Tokenizer;
    const Prompt: string;
    const Images: TQwen21ImageConditions;
    const Resident: TQwen21GPUImageConditions;
    out ImageSlots: TArray<Integer>
): TQNNGPUTensor;
var
  Ids, Positions: TArray<Integer>;
  Spans: TQwen21TextImageSpans;
  Drop, Rows, OutRows, Start, Stop, Slot: Integer;
  X, N, Q, K, V, A, Y, G, U, PositionGPU, IdsGPU: TQNNGPUTensor;
  Layer: ^TQwen21TextLayer;
  Keep: TArray<Boolean>;
begin
  X := nil;
  N := nil;
  Q := nil;
  K := nil;
  V := nil;
  A := nil;
  Y := nil;
  G := nil;
  U := nil;
  PositionGPU := nil;
  IdsGPU := nil;
  Q21BuildImagePrompt(Tokenizer, Prompt, Images, Ids, Positions, Spans, Drop);
  Rows := Length(Ids);
  if (Rows <= Drop)
      or ((Length(Images) = 0) and (Rows > 2048))
      or (Rows > 6144)
      or (Length(Images) <> Length(Resident)) then
    raise EArgumentException.Create('Invalid resident text/reference sequence');
  for var Id in Ids do
    if (Id < 0) or (Id >= FEmbedding.OutputDim) then
      raise EArgumentException.Create('Token outside embedding vocabulary');
  if Length(Positions) = 0 then
  begin
    SetLength(Positions, 3 * Rows);
    for var Axis := 0 to 2 do
      for var Row := 0 to Rows - 1 do
        Positions[Axis * Rows + Row] := Row
  end;
  try
    X := Ctx.NewTensor(Rows * FHidden);
    N := Ctx.NewTensor(Rows * FHidden);
    Q := Ctx.NewTensor(Rows * FHeads * FHeadDim);
    K := Ctx.NewTensor(Rows * FKVHeads * FHeadDim);
    V := Ctx.NewTensor(Rows * FKVHeads * FHeadDim);
    A := Ctx.NewTensor(Rows * FHeads * FHeadDim);
    Y := Ctx.NewTensor(Rows * FHidden);
    G := Ctx.NewTensor(Rows * FIntermediate);
    U := Ctx.NewTensor(Rows * FIntermediate);
    PositionGPU := Ctx.UploadRaw(@Positions[0], Length(Positions));
    IdsGPU := Ctx.UploadRaw(@Ids[0], Rows);
    Ctx.Gather(
        X,
        IdsGPU,
        FEmbedding.Weight,
        FEmbedding.Scales,
        Rows,
        FHidden,
        FEmbedding.OutputDim,
        FEmbedding.RotationGroup
    );
    for var Span in Spans do
    begin
      if (Span.TokenStart < Drop)
          or (Span.TokenStart + Span.TokenCount > Rows)
          or (Resident[Span.ImageIndex].Visual = nil)
          or (Resident[Span.ImageIndex].Visual.Count <> Span.TokenCount * FHidden) then
        raise EArgumentException.Create('Invalid resident visual span');
      Ctx.Copy(X, Resident[Span.ImageIndex].Visual, Span.TokenCount * FHidden, Span.TokenStart * FHidden);
    end;
    for var I := 0 to High(FLayers) do
    begin
      Layer := @FLayers[I];
      Q21GPURMS(Ctx, N, X, Layer.InputNorm, Rows, FHidden);
      Layer.Q.ForwardGPU(Ctx, Q, N, Rows);
      Layer.K.ForwardGPU(Ctx, K, N, Rows);
      Layer.V.ForwardGPU(Ctx, V, N, Rows);
      Q21GPURMS(Ctx, Q, Q, Layer.QNorm, Rows * FHeads, FHeadDim);
      Q21GPURMS(Ctx, K, K, Layer.KNorm, Rows * FKVHeads, FHeadDim);
      Ctx.Kernel(
          GO_TEXT_ROPE,
          Rows * FHeads * (FHeadDim div 2),
          Q,
          Q,
          PositionGPU,
          nil,
          [FHeads, FHeadDim, Rows, QNNGPUFloatBits(FTheta)]
      );
      Ctx.Kernel(
          GO_TEXT_ROPE,
          Rows * FKVHeads * (FHeadDim div 2),
          K,
          K,
          PositionGPU,
          nil,
          [FKVHeads, FHeadDim, Rows, QNNGPUFloatBits(FTheta)]
      );
      Ctx.Attention(A, Q, K, V, Rows, FHeads, FKVHeads, FHeadDim, Rows);
      Layer.O.ForwardGPU(Ctx, Y, A, Rows);
      Q21GPUAdd(Ctx, X, Y, Rows * FHidden);
      Q21GPURMS(Ctx, N, X, Layer.PostNorm, Rows, FHidden);
      Layer.Gate.ForwardGPU(Ctx, G, N, Rows);
      Layer.Up.ForwardGPU(Ctx, U, N, Rows);
      Ctx.Kernel(GO_SWIGLU, Rows * FIntermediate, G, G, U, nil, [FIntermediate, 0]);
      Layer.Down.ForwardGPU(Ctx, Y, G, Rows);
      Q21GPUAdd(Ctx, X, Y, Rows * FHidden);
      if I < 3 then
        for var Span in Spans do
        begin
          if (Resident[Span.ImageIndex].DeepStack[I] = nil)
              or (Resident[Span.ImageIndex].DeepStack[I].Count <> Span.TokenCount * FHidden) then
            raise EArgumentException.Create('Invalid resident DeepStack span');
          Ctx.Kernel(
              GO_ADD,
              Span.TokenCount * FHidden,
              X,
              Resident[Span.ImageIndex].DeepStack[I],
              nil,
              nil,
              [FHidden, FHidden, QNNGPUFloatBits(1), Span.TokenStart * FHidden, 0]
          );
        end;
      Ctx.CheckFinite(X, Rows * FHidden);
      if Assigned(FStatus) then
        FStatus(Format('GPU text encoder %d/%d', [I + 1, Length(FLayers)]));
    end;
    SetLength(Keep, Rows);
    for var Row := Drop to Rows - 1 do
      Keep[Row] := True;
    for var Span in Spans do
      for var Row := Span.TokenStart to Span.TokenStart + Span.TokenCount - 1 do
        Keep[Row] := False;
    OutRows := 0;
    for var Selected in Keep do
      if Selected then
        Inc(OutRows);
    if OutRows < 1 then
      raise EArgumentException.Create('Empty resident text conditioning');
    SetLength(ImageSlots, Length(Images));
    for var Span in Spans do
    begin
      Slot := 0;
      for var Row := Drop to Span.TokenStart - 1 do
        if Keep[Row] then
          Inc(Slot);
      ImageSlots[Span.ImageIndex] := Slot
    end;
    Result := Ctx.NewTensor(OutRows * FHidden);
    Start := Drop;
    OutRows := 0;
    while Start < Rows do
    begin
      if not Keep[Start] then
      begin
        Inc(Start);
        Continue
      end;
      Stop := Start + 1;
      while (Stop < Rows) and Keep[Stop] do
        Inc(Stop);
      Ctx.Copy(Result, X, (Stop - Start) * FHidden, OutRows * FHidden, Start * FHidden);
      Inc(OutRows, Stop - Start);
      Start := Stop;
    end;
    Ctx.CheckFinite(Result);
  finally
    Ctx.Recycle(X);
    Ctx.Recycle(N);
    Ctx.Recycle(Q);
    Ctx.Recycle(K);
    Ctx.Recycle(V);
    Ctx.Recycle(A);
    Ctx.Recycle(Y);
    Ctx.Recycle(G);
    Ctx.Recycle(U);
    Ctx.Recycle(PositionGPU);
    Ctx.Recycle(IdsGPU)
  end;
end;
