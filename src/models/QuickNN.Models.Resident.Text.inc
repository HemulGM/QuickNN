function Qwen3ForwardGPU(
    const Ctx: TQNNGPUContext;
    var Model: TQwen3Model;
    const Ids, Mask: TArray<LongInt>;
    const Rows: Integer
): TQNNGPUTensor;
var
  Work, LayerWork: TGPUWorkspace;
  X, N, Q, K, V, A, Y, Ffn, TokenGPU, PositionGPU, LimitGPU: TQNNGPUTensor;
  Positions, Limits: TArray<LongInt>;
  Layer: ^TQwen3Layer;
  LastLayer, ValidRows, Slots, Slot, OutDim, QDim, KvDim: Integer;
begin
  if (Rows < 1)
      or (Rows > QWEN3_MAX_SEQ_LEN)
      or (Length(Ids) < Rows)
      or ((Length(Mask) <> 0) and (Length(Mask) < Rows)) then
    raise EArgumentException.Create('Invalid resident Qwen3 input');
  ValidRows := Rows;
  if Length(Mask) <> 0 then
  begin
    ValidRows := 0;
    while (ValidRows < Rows) and (Mask[ValidRows] <> 0) do
      Inc(ValidRows);
    for var I := ValidRows to Rows - 1 do
      if Mask[I] <> 0 then
        raise EArgumentException.Create('Resident Qwen3 expects prefix padding');
  end;
  if ValidRows = 0 then
    raise EArgumentException.Create('Empty resident Qwen3 attention');
  for var Id in Ids do
    if (Id < 0) or (Id >= Model.VocabSize) then
      raise EArgumentException.Create('Qwen3 token outside embedding vocabulary');
  if Model.ExtractionMode = 1 then
  begin
    LastLayer := Length(Model.Layers) - 2;
    Slots := 1;
  end
  else
  begin
    LastLayer := QWEN3_OUTPUT_LAYER_3;
    Slots := 3;
  end;
  if (LastLayer < 0) or (LastLayer >= Length(Model.Layers)) then
    raise EArgumentException.Create('Qwen3 extraction layer is missing');
  Work := TGPUWorkspace.Create(Ctx);
  try
    OutDim := Slots * Model.HiddenSize;
    QDim := Model.NumHeads * Model.HeadDim;
    KvDim := Model.NumKvHeads * Model.HeadDim;
    Result := Work.New(Int64(Rows) * OutDim);
    TokenGPU := Work.Own(Ctx.UploadRaw(@Ids[0], Rows));
    SetLength(Positions, Rows * 3);
    SetLength(Limits, Rows);
    for var I := 0 to Rows - 1 do
    begin
      Limits[I] := Min(I + 1, ValidRows);
      for var Axis := 0 to 2 do
        Positions[Axis * Rows + I] := I;
    end;
    PositionGPU := Work.Own(Ctx.UploadRaw(@Positions[0], Length(Positions)));
    LimitGPU := Work.Own(Ctx.UploadRaw(@Limits[0], Rows));
    X := Work.New(Int64(Rows) * Model.HiddenSize);
    Ctx.Gather(X, TokenGPU, Model.EmbedTokens, Default(TMemoryBlock), Rows, Model.HiddenSize, Model.VocabSize, 0);
    for var I := 0 to LastLayer do
    begin
      Layer := @Model.Layers[I];
      Layer.Model := @Model;
      if not Layer.InputLayernormWeight.IsAssigned then
        Layer.Load(I, Model.UseMmap);
      LayerWork := TGPUWorkspace.Create(Ctx);
      try
        N := LayerWork.Norm(X, Layer.InputLayernormWeight, Default(TMemoryBlock), Rows, Model.HiddenSize);
        Q := LayerWork.Linear(N, Layer.Attn.QProjWeight, Rows, Model.HiddenSize, QDim);
        K := LayerWork.Linear(N, Layer.Attn.KProjWeight, Rows, Model.HiddenSize, KvDim);
        V := LayerWork.Linear(N, Layer.Attn.VProjWeight, Rows, Model.HiddenSize, KvDim);
        Q := LayerWork.Norm(Q, Layer.Attn.QNormWeight, Default(TMemoryBlock), Rows * Model.NumHeads, Model.HeadDim);
        K := LayerWork.Norm(K, Layer.Attn.KNormWeight, Default(TMemoryBlock), Rows * Model.NumKvHeads, Model.HeadDim);
        Ctx.Kernel(
            GO_TEXT_ROPE,
            Rows * QDim div 2,
            Q,
            Q,
            PositionGPU,
            nil,
            [Model.NumHeads, Model.HeadDim, Rows, QNNGPUFloatBits(Model.RoPETheta)]
        );
        Ctx.Kernel(
            GO_TEXT_ROPE,
            Rows * KvDim div 2,
            K,
            K,
            PositionGPU,
            nil,
            [Model.NumKvHeads, Model.HeadDim, Rows, QNNGPUFloatBits(Model.RoPETheta)]
        );
        A := LayerWork.New(Int64(Rows) * QDim);
        Ctx.Attention(A, Q, K, V, Rows, Model.NumHeads, Model.NumKvHeads, Model.HeadDim, Rows, LimitGPU);
        Y := LayerWork.Linear(A, Layer.Attn.OProjWeight, Rows, QDim, Model.HiddenSize);
        LayerWork.Add(X, Y, Rows * Model.HiddenSize);
        N := LayerWork.Norm(X, Layer.PostAttentionLayernormWeight, Default(TMemoryBlock), Rows, Model.HiddenSize);
        Ffn :=
            LayerWork.SwiGLU(
                N,
                Layer.Mlp.GateProjWeight,
                Layer.Mlp.UpProjWeight,
                Layer.Mlp.DownProjWeight,
                Rows,
                Model.HiddenSize,
                Model.IntermediateSize
            );
        LayerWork.Add(X, Ffn, Rows * Model.HiddenSize);
        Slot := -1;
        if Slots = 1 then
        begin
          if I = LastLayer then
            Slot := 0;
        end
        else
          case I of
            QWEN3_OUTPUT_LAYER_1: Slot := 0;
            QWEN3_OUTPUT_LAYER_2: Slot := 1;
            QWEN3_OUTPUT_LAYER_3: Slot := 2;
          end;
        if Slot >= 0 then
          Ctx.Kernel(
              GO_SCATTER,
              Rows * Model.HiddenSize,
              Result,
              X,
              nil,
              nil,
              [Model.HiddenSize, OutDim, Slot * Model.HiddenSize, 0]
          );
        if Assigned(TextProgressCallback) then
          TextProgressCallback(I, LastLayer + 1);
      finally
        LayerWork.Free;
      end;
    end;
    Ctx.CheckFinite(Result);
    Work.Keep(Result);
  finally
    Work.Free;
  end;
end;

function Qwen3EncodeGPU(
    const Ctx: TQNNGPUContext;
    var Encoder: TQwen3Encoder;
    const Prompt: RawByteString;
    out RealRows: LongInt
): TQNNGPUTensor;
var
  Tokens, Mask, Padded: TArray<LongInt>;
begin
  Tokens := Encoder.Tokenizer.TokenizeChat(Prompt, Encoder.Model.ExtractionMode = 1, QWEN3_MAX_SEQ_LEN);
  RealRows := Length(Tokens);
  SetLength(Mask, QWEN3_MAX_SEQ_LEN);
  Padded := PadTokens(Tokens, QWEN3_MAX_SEQ_LEN, Mask);
  Result := Qwen3ForwardGPU(Ctx, Encoder.Model, Padded, Mask, QWEN3_MAX_SEQ_LEN);
end;
