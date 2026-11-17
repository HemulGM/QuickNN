unit QuickNN.Models.Gemma2;

{$POINTERMATH ON}

interface

uses
  System.SysUtils, System.JSON, QuickNN.Common, QuickNN.SafeTensors,
  QuickNN.Vulkan, QuickNN.Vulkan.Operations, QuickNN.Models.Qwen3,
  QuickNN.Models.LLM.Types;

type
  TGemma2 = class
  private type
    TLayer = record
      InputNorm, AttentionNorm, FeedForwardNorm, OutputNorm: TMemoryBlock;
      Q, K, V, O, Gate, Up, Down: TMemoryBlock;
      Cache: TQNNGPUTensor;
    end;
    PLayer = ^TLayer;
  private
    FCtx: TQNNGPUContext;
    FModel: PQWEN3Model;
    FFiles: TSafeTensorFiles;
    FLayers: TArray<TLayer>;
    FEmbedding, FNorm: TMemoryBlock;
    FCapacity, FWindow: Integer;
    FEpsilon, FAttentionCap, FLogitCap, FScale: Single;
    FCacheBytes, FWeightBytes: UInt64;
    FCancelCheck: TFunc<Boolean>;
    function Weight(const Name: string; const Rows, Columns: Integer): TMemoryBlock;
    function Project(const Work: TGPUWorkspace; const Source: TQNNGPUTensor;
      const Matrix: TMemoryBlock; const Rows, Inputs, Outputs: Integer): TQNNGPUTensor;
  public
    constructor Create(const Context: TQNNGPUContext; var Model: TQwen3Model;
      const Config: TJSONValue; const Capacity: Integer);
    destructor Destroy; override;
    procedure Forward(const Ids: TArray<LongInt>; const GPUIds, Logits: TQNNGPUTensor; const Position: Integer);
    procedure Status(var Value: TLLMMemoryStatus);
    function StateIdentity: NativeUInt;
    property CacheBytes: UInt64 read FCacheBytes;
    property CancelCheck: TFunc<Boolean> read FCancelCheck write FCancelCheck;
  end;

implementation

uses
  System.Math, QuickNN.GGUF.Quantization;

function TGemma2.Weight(const Name: string; const Rows, Columns: Integer): TMemoryBlock;
begin
  var Tensor := FFiles.GetTensor(Name);
  if Tensor = nil then raise EArgumentException.Create('Missing Gemma 2 tensor: ' + Name);
  if Columns = 0 then
  begin
    if (Tensor.Ndim <> 1) or (Tensor.Count <> Rows) or (Tensor.Encoding <> weNative) then
      raise EArgumentException.Create('Invalid Gemma 2 normalization: ' + Name);
    // GGUF exports already include Gemma's +1 gamma offset. Do not add it twice.
    Result := Tensor.AsSingle(True);
  end
  else
  begin
    if (Tensor.Ndim <> 2) or (Tensor.Shape[0] <> Rows) or (Tensor.Shape[1] <> Columns) then
      raise EArgumentException.Create('Invalid Gemma 2 matrix: ' + Name);
    Result := FFiles.GetLinearWeight(Name, True);
  end;
  if Result.Encoding = weNative then Inc(FWeightBytes, UInt64(Result.Count) * UInt64(DATATYPE_BITS[Result.DataType] div 8))
  else Inc(FWeightBytes, GGUFQuantBytes(Result.Encoding, Result.Count));
end;

constructor TGemma2.Create(const Context: TQNNGPUContext; var Model: TQwen3Model;
  const Config: TJSONValue; const Capacity: Integer);
begin
  inherited Create;
  FCtx := Context; FModel := @Model; FCapacity := Capacity;
  Model.HiddenSize := Config.GetValue<Integer>('hidden_size');
  Model.IntermediateSize := Config.GetValue<Integer>('intermediate_size');
  Model.NumHeads := Config.GetValue<Integer>('num_attention_heads');
  Model.NumKvHeads := Config.GetValue<Integer>('num_key_value_heads');
  Model.HeadDim := Config.GetValue<Integer>('head_dim');
  Model.VocabSize := Config.GetValue<Integer>('vocab_size');
  Model.RoPETheta := Config.GetValue<Single>('rope_theta', 10000);
  var Layers := Config.GetValue<Integer>('num_hidden_layers');
  FEpsilon := Config.GetValue<Single>('rms_norm_eps', 1E-6);
  FWindow := Config.GetValue<Integer>('sliding_window', 4096);
  FAttentionCap := Config.GetValue<Single>('attn_logit_softcapping', 50);
  FLogitCap := Config.GetValue<Single>('final_logit_softcapping', 30);
  var QueryScale := Config.GetValue<Single>('query_pre_attn_scalar', Model.HeadDim);
  if (Layers < 1) or (Layers > 256) or (Model.HiddenSize < 1) or (Model.IntermediateSize < 1)
    or (Model.NumHeads < 1) or (Model.NumKvHeads < 1) or (Model.NumHeads mod Model.NumKvHeads <> 0)
    or (Model.HeadDim < 2) or (Model.HeadDim > 1152) or Odd(Model.HeadDim) or (Model.VocabSize < 1)
    or (Capacity < 2) or (Capacity > Config.GetValue<Integer>('max_position_embeddings'))
    or (FWindow < 1) or not (FEpsilon > 0) or not (FAttentionCap > 0)
    or not (FLogitCap > 0) or not (QueryScale > 0) then
    raise EArgumentException.Create('Invalid Gemma 2 geometry/configuration');
  FScale := 1 / Sqrt(QueryScale);
  FCacheBytes := UInt64(Layers) * 2 * UInt64(Capacity) * UInt64(Model.NumKvHeads) * UInt64(Model.HeadDim) * 4;
  if FCacheBytes > UInt64(1536) * 1024 * 1024 then
    raise EArgumentOutOfRangeException.Create('Gemma 2 KV cache exceeds 1.5 GiB; reduce context');
  Model.SfFiles := TSafeTensorFiles.Create(TSafeTensorFile.Open(Config.GetValue<string>('quicknn_gguf_file')));
  FFiles := Model.SfFiles;
  FEmbedding := Weight('token_embd.weight', Model.VocabSize, Model.HiddenSize);
  FNorm := Weight('output_norm.weight', Model.HiddenSize, 0);
  SetLength(FLayers, Layers);
  for var I := 0 to Layers - 1 do
  begin
    var Layer: PLayer := @FLayers[I]; var Prefix := 'blk.' + I.ToString + '.';
    Layer.InputNorm := Weight(Prefix + 'attn_norm.weight', Model.HiddenSize, 0);
    Layer.AttentionNorm := Weight(Prefix + 'post_attention_norm.weight', Model.HiddenSize, 0);
    Layer.FeedForwardNorm := Weight(Prefix + 'ffn_norm.weight', Model.HiddenSize, 0);
    Layer.OutputNorm := Weight(Prefix + 'post_ffw_norm.weight', Model.HiddenSize, 0);
    Layer.Q := Weight(Prefix + 'attn_q.weight', Model.NumHeads * Model.HeadDim, Model.HiddenSize);
    Layer.K := Weight(Prefix + 'attn_k.weight', Model.NumKvHeads * Model.HeadDim, Model.HiddenSize);
    Layer.V := Weight(Prefix + 'attn_v.weight', Model.NumKvHeads * Model.HeadDim, Model.HiddenSize);
    Layer.O := Weight(Prefix + 'attn_output.weight', Model.HiddenSize, Model.NumHeads * Model.HeadDim);
    Layer.Gate := Weight(Prefix + 'ffn_gate.weight', Model.IntermediateSize, Model.HiddenSize);
    Layer.Up := Weight(Prefix + 'ffn_up.weight', Model.IntermediateSize, Model.HiddenSize);
    Layer.Down := Weight(Prefix + 'ffn_down.weight', Model.HiddenSize, Model.IntermediateSize);
  end;
  if FWeightBytes > QNNGetGPUStatus.WeightCacheLimit then
    raise EArgumentOutOfRangeException.Create('Gemma 2 packed weights exceed the GPU weight-cache budget');
  for var I := 0 to High(FLayers) do
  begin
    if Assigned(FCancelCheck) and FCancelCheck() then raise ELLMCancelled.Create('Gemma 2 loading cancelled');
    FLayers[I].Cache := FCtx.NewTensor(2 * Capacity * Model.NumKvHeads * Model.HeadDim);
  end;
end;

destructor TGemma2.Destroy;
begin
  for var I := 0 to High(FLayers) do
  begin
    var Layer: PLayer := @FLayers[I];
    Layer.InputNorm.Free; Layer.AttentionNorm.Free; Layer.FeedForwardNorm.Free; Layer.OutputNorm.Free;
    Layer.Q.Free; Layer.K.Free; Layer.V.Free; Layer.O.Free;
    Layer.Gate.Free; Layer.Up.Free; Layer.Down.Free;
  end;
  FEmbedding.Free; FNorm.Free;
  inherited;
end;

function TGemma2.Project(const Work: TGPUWorkspace; const Source: TQNNGPUTensor;
  const Matrix: TMemoryBlock; const Rows, Inputs, Outputs: Integer): TQNNGPUTensor;
begin
  Result := Work.New(Int64(Rows) * Outputs);
  if (Rows = 1) and (Matrix.Encoding = weNative) then FCtx.VectorLinear(Result, Source, Matrix, Inputs, Outputs)
  else FCtx.Linear(Result, Source, Matrix, Default(TMemoryBlock), Default(TMemoryBlock), Rows, Inputs, Outputs);
end;

procedure TGemma2.Forward(const Ids: TArray<LongInt>; const GPUIds, Logits: TQNNGPUTensor; const Position: Integer);
begin
  var Rows := Length(Ids); var Hidden := FModel.HiddenSize;
  var QDim := FModel.NumHeads * FModel.HeadDim; var KVDim := FModel.NumKvHeads * FModel.HeadDim;
  var Work := TGPUWorkspace.Create(FCtx);
  try
    var Tokens := GPUIds;
    if Tokens = nil then Tokens := Work.Own(FCtx.UploadRaw(@Ids[0], Rows));
    var X := Work.New(Int64(Rows) * Hidden);
    FCtx.Gather(X, Tokens, FEmbedding, Default(TMemoryBlock), Rows, Hidden, FModel.VocabSize, 0);
    FCtx.Kernel(GO_SCALE, Rows * Hidden, X, X, nil, nil, [QNNGPUFloatBits(Sqrt(Hidden))]);
    for var I := 0 to High(FLayers) do
    begin
      if Assigned(FCancelCheck) and FCancelCheck() then raise ELLMCancelled.Create('Gemma 2 generation cancelled');
      var Layer: PLayer := @FLayers[I]; var LayerWork := TGPUWorkspace.Create(FCtx);
      try
        var N := LayerWork.Norm(X, Layer.InputNorm, Default(TMemoryBlock), Rows, Hidden, False, FEpsilon);
        var Q := Project(LayerWork, N, Layer.Q, Rows, Hidden, QDim);
        var K := Project(LayerWork, N, Layer.K, Rows, Hidden, KVDim);
        var V := Project(LayerWork, N, Layer.V, Rows, Hidden, KVDim);
        FCtx.Kernel(GO_LLM_ROPE, Rows * QDim div 2, Q, Q, nil, nil,
          [FModel.NumHeads, FModel.HeadDim, Position, QNNGPUFloatBits(FModel.RoPETheta)]);
        FCtx.Kernel(GO_LLM_ROPE, Rows * KVDim div 2, K, K, nil, nil,
          [FModel.NumKvHeads, FModel.HeadDim, Position, QNNGPUFloatBits(FModel.RoPETheta)]);
        FCtx.Kernel(GO_LLM_CACHE_WRITE, Rows * KVDim, Layer.Cache, K, V, nil,
          [Rows, FModel.NumKvHeads, FModel.HeadDim, FCapacity, Position]);
        var A := LayerWork.New(Int64(Rows) * QDim);
        var Window := 0; if not Odd(I) then Window := FWindow;
        FCtx.CachedAttention(A, Q, Layer.Cache, Rows, FCapacity, FModel.NumHeads,
          FModel.NumKvHeads, FModel.HeadDim, Position, Window, FAttentionCap, FScale);
        var Y := Project(LayerWork, A, Layer.O, Rows, QDim, Hidden);
        Y := LayerWork.Norm(Y, Layer.AttentionNorm, Default(TMemoryBlock), Rows, Hidden, False, FEpsilon);
        LayerWork.Add(X, Y, Rows * Hidden);
        N := LayerWork.Norm(X, Layer.FeedForwardNorm, Default(TMemoryBlock), Rows, Hidden, False, FEpsilon);
        var G := Project(LayerWork, N, Layer.Gate, Rows, Hidden, FModel.IntermediateSize);
        var U := Project(LayerWork, N, Layer.Up, Rows, Hidden, FModel.IntermediateSize);
        FCtx.Kernel(GO_GEGLU, Rows * FModel.IntermediateSize, G, G, U, nil, [FModel.IntermediateSize, 0]);
        Y := Project(LayerWork, G, Layer.Down, Rows, FModel.IntermediateSize, Hidden);
        Y := LayerWork.Norm(Y, Layer.OutputNorm, Default(TMemoryBlock), Rows, Hidden, False, FEpsilon);
        LayerWork.Add(X, Y, Rows * Hidden);
      finally LayerWork.Free end;
    end;
    var Last := Work.Copy(X, Hidden, (Rows - 1) * Hidden);
    var N := Work.Norm(Last, FNorm, Default(TMemoryBlock), 1, Hidden, False, FEpsilon);
    FCtx.Linear(Logits, N, FEmbedding, Default(TMemoryBlock), Default(TMemoryBlock), 1, Hidden, FModel.VocabSize);
    FCtx.Kernel(GO_SOFTCAP, FModel.VocabSize, Logits, Logits, nil, nil, [QNNGPUFloatBits(FLogitCap)]);
    FCtx.CheckFinite(Logits);
  finally Work.Free end;
end;

procedure TGemma2.Status(var Value: TLLMMemoryStatus);
begin
  Value.IsGGUF := True; Value.DenseQuantization := wqNative;
  Value.FixedGPUWeightBytes := FWeightBytes;
  Value.ResidentLayers := Length(FLayers); Value.StreamedLayers := 0; Value.CPULayers := 0;
  Value.QuantizedMappedWeightBytes := FWeightBytes; Value.ActivationBytes := FCtx.AllocatedBytes;
end;

function TGemma2.StateIdentity: NativeUInt;
begin
  Result := 0;
  if Length(FLayers) > 0 then Result := NativeUInt(FLayers[0].Cache);
end;

end.
