unit QuickNN.Models.QwenHybrid;

{$POINTERMATH ON}

interface

uses
  System.SysUtils, System.JSON, System.Generics.Collections, QuickNN.Common,
  QuickNN.SafeTensors, QuickNN.Vulkan, QuickNN.Vulkan.Operations,
  QuickNN.Models.Qwen3, QuickNN.Models.LLM.Types,
  QuickNN.Models.LLM.Quantization, QuickNN.Models.LLM.Memory;

type
  TQwenHybrid = class
  private
    type
      TLayer = record
        Linear, resident: Boolean;
        InputNorm, PostNorm, Gate, Up, Down: TMemoryBlock;
        Q, K, V, O, QNorm, KNorm: TMemoryBlock;
        QKV, Z, A, B, ALog, DtBias, Conv, GatedNorm: TMemoryBlock;
        State, ConvState, ConvGPU, NormGPU: TQNNGPUTensor;
      end;

      PLayer = ^TLayer;
  private
    FCtx: TQNNGPUContext;
    FModel: PQWEN3Model;
    FCache: TLLMInt4Cache;
    FHost: TLLMHostWeightCache;
    FSource: TSafeTensorFiles;
    FLayers: TArray<TLayer>;
    FEmbedding, FOutput, FNorm: TMemoryBlock;
    FCapacity, FKHeads, FVHeads, FKDim, FVDim, FConvSize, FRotaryDim: Integer;
    FEpsilon: Single;
    FQuantization: TLLMWeightQuantization;
    FCacheBytes, FRecurrentBytes, FFixedBytes, FMaxLayerBytes, FRAMLimit: UInt64;
    FResidentLayers: Integer;
    FStateValid: Boolean;
    FImageStarts: TArray<Integer>;
    FImageFeatures: TArray<TQNNGPUTensor>;
    FVisualPositions: TArray<LongInt>;
    FVisualNext: Integer;
    FVisionSignature: string;
    FCancelCheck: TFunc<Boolean>;
    function Load(const Name: string; const Rows: Integer = 0; const Columns: Integer = 0): TMemoryBlock;
    function Matrices(const Layer: TLayer): TArray<TMemoryBlock>;
    procedure PreparePlacement;
    procedure Preload(const Layer: TLayer);
    function Project(const Work: TGPUWorkspace; const Source: TQNNGPUTensor; const Weight: TMemoryBlock; const Rows, InputDim, OutputDim: Integer): TQNNGPUTensor;
    function Norm(const Work: TGPUWorkspace; const Source: TQNNGPUTensor; const Weight: TMemoryBlock; const Rows, Dim: Integer): TQNNGPUTensor;
    function LinearAttention(const Work: TGPUWorkspace; var Layer: TLayer; const Input: TQNNGPUTensor; const Rows: Integer): TQNNGPUTensor;
    function FullAttention(const Work: TGPUWorkspace; var Layer: TLayer; const Input: TQNNGPUTensor; const Rows, Position: Integer): TQNNGPUTensor;
  public
    constructor Create(const Context: TQNNGPUContext; var Model: TQwen3Model; const Directory: string; const Config: TJSONValue; const Capacity: Integer; const Memory: TLLMMemoryOptions);
    destructor Destroy; override;
    procedure SetVision(const Tokens: TArray<LongInt>; const Images: TArray<TLLMImage>; const Signature: string);
    procedure Reset;
    procedure forward(const Ids: TArray<LongInt>; const GPUIds, Logits: TQNNGPUTensor; const Position: Integer);
    procedure Status(var Value: TLLMMemoryStatus);
    function StateIdentity: NativeUInt;
    property CancelCheck: TFunc<Boolean> read FCancelCheck write FCancelCheck;
    property StateValid: Boolean read FStateValid;
    property VisionSignature: string read FVisionSignature;
    property CacheBytes: UInt64 read FCacheBytes;
  end;

implementation

uses
  System.Math, System.IOUtils, QuickNN.Quantization.Int4;

function WeightBytes(const Weight: TMemoryBlock): UInt64;
begin
  Result := UInt64(Weight.Count) * UInt64(DATATYPE_BITS[Weight.DataType] div 8);
end;

function TQwenHybrid.Load(const Name: string; const Rows, Columns: Integer): TMemoryBlock;
var
  Tensor: PSafeTensor;
begin
  Tensor := FSource.GetTensor(Name);
  if (Tensor = nil) or not (Tensor.DType in [StF32, StBF16]) then
    raise ESafeTensorError.Create('Missing native Qwen hybrid weight: ' + Name);
  if (Rows > 0) and ((Tensor.Ndim <> 2) or (Tensor.Shape[0] <> Rows) or (Tensor.Shape[1] <> Columns)) then
    raise ESafeTensorError.Create('Invalid Qwen hybrid matrix: ' + Name);
  if (FCache <> nil) and (Rows > 0) and not Name.EndsWith('.in_proj_a.weight')
    and not Name.EndsWith('.in_proj_b.weight') then
    Result := FCache.Weight(Name, Rows, Columns)
  else
    Result := FSource.GetLinearWeight(Name, True);
end;

constructor TQwenHybrid.Create(const Context: TQNNGPUContext; var Model: TQwen3Model; const Directory: string; const Config: TJSONValue; const Capacity: Integer; const Memory: TLLMMemoryOptions);
var
  Text, Rope, Kinds: TJSONValue;
  Prefix, CacheDirectory: string;
  QDim, KeyDim, ValueDim, MixedDim, Layers: Integer;
begin
  inherited Create;
  FCtx := Context;
  FModel := @Model;
  Text := Config.FindValue('text_config');
  if Text = nil then
    raise EArgumentException.Create('Missing Qwen hybrid text_config');
  Model.HiddenSize := Text.GetValue<Integer>('hidden_size');
  Model.IntermediateSize := Text.GetValue<Integer>('intermediate_size');
  Model.NumHeads := Text.GetValue<Integer>('num_attention_heads');
  Model.NumKvHeads := Text.GetValue<Integer>('num_key_value_heads');
  Model.HeadDim := Text.GetValue<Integer>('head_dim');
  Model.VocabSize := Text.GetValue<Integer>('vocab_size');
  Layers := Text.GetValue<Integer>('num_hidden_layers');
  FKHeads := Text.GetValue<Integer>('linear_num_key_heads');
  FVHeads := Text.GetValue<Integer>('linear_num_value_heads');
  FKDim := Text.GetValue<Integer>('linear_key_head_dim');
  FVDim := Text.GetValue<Integer>('linear_value_head_dim');
  FConvSize := Text.GetValue<Integer>('linear_conv_kernel_dim');
  FEpsilon := Text.GetValue<Single>('rms_norm_eps', 1E-6);
  FCapacity := Capacity;
  if (Layers < 1) or (Layers > 256) or (Model.HiddenSize < 1) or (Model.HiddenSize > 32768)
    or (Model.IntermediateSize < 1) or (Model.IntermediateSize > 131072)
    or (Model.VocabSize < 1) or (Model.NumHeads < 1) or (Model.NumHeads > 256) or (Model.NumKvHeads < 1) or (Model.NumKvHeads > 256)
    or (Model.NumHeads mod Model.NumKvHeads <> 0) or (Model.HeadDim < 2) or (Model.HeadDim > 1152)
    or (FKHeads < 1) or (FKHeads > 256) or (FVHeads < 1) or (FVHeads > 256) or (FVHeads mod FKHeads <> 0)
    or (FKDim < 1) or (FKDim > 256) or (FVDim < 1) or (FVDim > 256)
    or (FConvSize < 1) or (FConvSize > 16) or not (FEpsilon > 0) or IsInfinite(FEpsilon)
    or (Capacity < 2) or (Capacity > Text.GetValue<Integer>('max_position_embeddings', 262144)) then
    raise EArgumentException.Create('Invalid Qwen hybrid geometry or context');
  if Text.GetValue<Boolean>('attention_bias', False) or (Text.GetValue<string>('hidden_act', 'silu') <> 'silu') then
    raise ENotSupportedException.Create('Unsupported Qwen hybrid activation/bias');
  Rope := Text.FindValue('rope_parameters');
  if (Rope = nil) or (Rope.GetValue<string>('rope_type', '') <> 'default') then
    raise ENotSupportedException.Create('Unsupported Qwen hybrid RoPE');
  Model.RoPETheta := Rope.GetValue<Single>('rope_theta', 10000000);
  FRotaryDim := Round(Model.HeadDim * Rope.GetValue<Single>('partial_rotary_factor', 0.25));
  if (FRotaryDim < 2) or (FRotaryDim > Model.HeadDim) or (FRotaryDim mod 2 <> 0)
    or not (Model.RoPETheta > 0) then
    raise EArgumentException.Create('Invalid Qwen hybrid rotary geometry');
  Kinds := Text.FindValue('layer_types');
  if not (Kinds is TJSONArray) or (TJSONArray(Kinds).Count <> Layers) then
    raise EArgumentException.Create('Invalid Qwen hybrid layer_types');
  SetLength(Model.Layers, Layers); // Shared model metadata only; hybrid weights are owned here.
  SetLength(FLayers, Layers);
  for var I := 0 to Layers - 1 do
  begin
    var Kind := TJSONArray(Kinds).Items[I].Value;
    if (Kind <> 'linear_attention') and (Kind <> 'full_attention') then
      raise ENotSupportedException.Create('Unknown Qwen hybrid attention layer');
    FLayers[I].Linear := Kind = 'linear_attention';
    if FLayers[I].Linear then
    begin
      Inc(FRecurrentBytes, UInt64(FVHeads) * UInt64(FKDim) * UInt64(FVDim) * 4);
      Inc(FRecurrentBytes, UInt64(2 * FKHeads * FKDim + FVHeads * FVDim) * UInt64(FConvSize) * 4);
    end
    else
      Inc(FCacheBytes, UInt64(2) * UInt64(Capacity) * UInt64(Model.NumKvHeads) * UInt64(Model.HeadDim) * 4);
  end;
  Inc(FCacheBytes, FRecurrentBytes);
  if FCacheBytes > UInt64(1536) * 1024 * 1024 then
    raise EArgumentOutOfRangeException.Create('Hybrid KV/recurrent state exceeds 1.5 GiB; reduce context');
  Model.SfFiles := LoadSafeTensorShards(Directory, 'model.safetensors.index.json', 'model.safetensors');
  FSource := Model.SfFiles;
  FQuantization := Memory.DenseQuantization;
  if FQuantization = wqInt4 then
  begin
    CacheDirectory := Memory.ExpertCacheDirectory;
    if CacheDirectory = '' then
      CacheDirectory := TPath.Combine(Directory, '.quicknn-cache');
    FCache := TLLMInt4Cache.Create(FSource, CacheDirectory, Memory.OnExpertCacheProgress, iqDense);
  end;
  FEmbedding := Load('model.language_model.embed_tokens.weight', Model.VocabSize, Model.HiddenSize);
  FOutput := Load('lm_head.weight', Model.VocabSize, Model.HiddenSize);
  FNorm := Load('model.language_model.norm.weight');
  if FNorm.Count <> Model.HiddenSize then
    raise ESafeTensorError.Create('Invalid final Qwen hybrid norm width');
  QDim := Model.NumHeads * Model.HeadDim;
  KeyDim := FKHeads * FKDim;
  ValueDim := FVHeads * FVDim;
  MixedDim := 2 * KeyDim + ValueDim;
  for var I := 0 to High(FLayers) do
  begin
    Prefix := 'model.language_model.layers.' + I.ToString;
    var Layer: PLayer := @FLayers[I];
    Layer.InputNorm := Load(Prefix + '.input_layernorm.weight');
    Layer.PostNorm := Load(Prefix + '.post_attention_layernorm.weight');
    Layer.Gate := Load(Prefix + '.mlp.gate_proj.weight', Model.IntermediateSize, Model.HiddenSize);
    Layer.Up := Load(Prefix + '.mlp.up_proj.weight', Model.IntermediateSize, Model.HiddenSize);
    Layer.Down := Load(Prefix + '.mlp.down_proj.weight', Model.HiddenSize, Model.IntermediateSize);
    if Layer.Linear then
    begin
      Prefix := Prefix + '.linear_attn';
      Layer.QKV := Load(Prefix + '.in_proj_qkv.weight', MixedDim, Model.HiddenSize);
      Layer.Z := Load(Prefix + '.in_proj_z.weight', ValueDim, Model.HiddenSize);
      Layer.A := Load(Prefix + '.in_proj_a.weight', FVHeads, Model.HiddenSize);
      Layer.B := Load(Prefix + '.in_proj_b.weight', FVHeads, Model.HiddenSize);
      Layer.O := Load(Prefix + '.out_proj.weight', Model.HiddenSize, ValueDim);
      Layer.ALog := Load(Prefix + '.A_log');
      Layer.DtBias := Load(Prefix + '.dt_bias');
      Layer.Conv := Load(Prefix + '.conv1d.weight');
      Layer.GatedNorm := Load(Prefix + '.norm.weight');
      if (Layer.Conv.Count <> MixedDim * FConvSize) or (Layer.GatedNorm.Count <> FVDim)
        or (Layer.ALog.Count <> FVHeads) or (Layer.DtBias.Count <> FVHeads) then
        raise ESafeTensorError.Create('Invalid Gated DeltaNet parameter shape: ' + Prefix);
      Layer.State := FCtx.NewTensor(FVHeads * FKDim * FVDim);
      Layer.ConvState := FCtx.NewTensor(MixedDim * FConvSize);
      var ConvF32 := FSource.GetTensorDataMemBlock(Prefix + '.conv1d.weight', True);
      var NormF32 := FSource.GetTensorDataMemBlock(Prefix + '.norm.weight', True);
      try
        Layer.ConvGPU := FCtx.Upload(ConvF32);
        Layer.NormGPU := FCtx.Upload(NormF32);
      finally
        ConvF32.Free;
        NormF32.Free;
      end;
    end
    else
    begin
      Prefix := Prefix + '.self_attn';
      Layer.Q := Load(Prefix + '.q_proj.weight', 2 * QDim, Model.HiddenSize);
      Layer.K := Load(Prefix + '.k_proj.weight', Model.NumKvHeads * Model.HeadDim, Model.HiddenSize);
      Layer.V := Load(Prefix + '.v_proj.weight', Model.NumKvHeads * Model.HeadDim, Model.HiddenSize);
      Layer.O := Load(Prefix + '.o_proj.weight', Model.HiddenSize, QDim);
      Layer.QNorm := Load(Prefix + '.q_norm.weight');
      Layer.KNorm := Load(Prefix + '.k_norm.weight');
      if (Layer.QNorm.Count <> Model.HeadDim) or (Layer.KNorm.Count <> Model.HeadDim) then
        raise ESafeTensorError.Create('Invalid Qwen hybrid Q/K normalization width');
      Layer.State := FCtx.NewTensor(2 * Capacity * Model.NumKvHeads * Model.HeadDim);
    end;
    if (Layer.InputNorm.Count <> Model.HiddenSize) or (Layer.PostNorm.Count <> Model.HiddenSize) then
      raise ESafeTensorError.Create('Invalid Qwen hybrid normalization width');
  end;
  FRAMLimit := Memory.RAMCacheLimitBytes;
  FHost := TLLMHostWeightCache.Create(FRAMLimit);
  PreparePlacement;
  Reset;
end;

destructor TQwenHybrid.Destroy;

  procedure Release(var Weight: TMemoryBlock);
  begin
    Weight.Free;
  end;

begin
  FHost.Free;
  for var I := 0 to High(FLayers) do
  begin
    var Layer: PLayer := @FLayers[I];
    Release(Layer.InputNorm);
    Release(Layer.PostNorm);
    Release(Layer.Gate);
    Release(Layer.Up);
    Release(Layer.Down);
    Release(Layer.Q);
    Release(Layer.K);
    Release(Layer.V);
    Release(Layer.O);
    Release(Layer.QNorm);
    Release(Layer.KNorm);
    Release(Layer.QKV);
    Release(Layer.Z);
    Release(Layer.A);
    Release(Layer.B);
    Release(Layer.ALog);
    Release(Layer.DtBias);
    Release(Layer.Conv);
    Release(Layer.GatedNorm);
  end;
  Release(FEmbedding);
  Release(FOutput);
  Release(FNorm);
  FCache.Free;
  inherited;
end;

function TQwenHybrid.Matrices(const Layer: TLayer): TArray<TMemoryBlock>;
begin
  if Layer.Linear then
    Result := [Layer.Gate, Layer.Up, Layer.Down, Layer.QKV, Layer.Z, Layer.O]
  else
    Result := [Layer.Gate, Layer.Up, Layer.Down, Layer.Q, Layer.K, Layer.V, Layer.O];
end;

procedure TQwenHybrid.PreparePlacement;
var
  Limit, Bytes: UInt64;

  procedure Retain(const Weight: TMemoryBlock);
  begin
    if not Weight.IsAssigned then
      Exit;
    FCtx.RetainWeight(Weight, 1, Weight.Count);
    Inc(FFixedBytes, WeightBytes(Weight));
  end;

begin
  Limit := QNNGetGPUStatus.WeightCacheLimit;
  for var Layer in FLayers do
  begin
    Bytes := 0;
    for var Weight in Matrices(Layer) do
      Inc(Bytes, WeightBytes(Weight));
    FMaxLayerBytes := Max(FMaxLayerBytes, Bytes);
  end;
  Bytes := WeightBytes(FEmbedding) + WeightBytes(FOutput) + WeightBytes(FNorm);
  for var Layer in FLayers do
  begin
    Inc(Bytes, WeightBytes(Layer.InputNorm) + WeightBytes(Layer.PostNorm));
    if Layer.Linear then
      Inc(Bytes, WeightBytes(Layer.A) + WeightBytes(Layer.B) + WeightBytes(Layer.ALog) + WeightBytes(Layer.DtBias))
    else
      Inc(Bytes, WeightBytes(Layer.QNorm) + WeightBytes(Layer.KNorm));
  end;
  if Bytes + FMaxLayerBytes > Limit then
    raise ERangeError.Create('GPU weight budget cannot hold hybrid fixed weights and one streamed layer');
  Retain(FEmbedding);
  Retain(FOutput);
  Retain(FNorm);
  for var Layer in FLayers do
  begin
    Retain(Layer.InputNorm);
    Retain(Layer.PostNorm);
    if Layer.Linear then
    begin
      Retain(Layer.A);
      Retain(Layer.B);
      Retain(Layer.ALog);
      Retain(Layer.DtBias);
    end
    else
    begin
      Retain(Layer.QNorm);
      Retain(Layer.KNorm);
    end;
  end;
  for var I := 0 to High(FLayers) do
  begin
    Bytes := 0;
    for var Weight in Matrices(FLayers[I]) do
      Inc(Bytes, WeightBytes(Weight));
    if FFixedBytes + Bytes + FMaxLayerBytes > Limit then
      Break;
    for var Weight in Matrices(FLayers[I]) do
      Retain(Weight);
    FLayers[I].resident := True;
    Inc(FResidentLayers);
  end;
end;

procedure TQwenHybrid.Preload(const Layer: TLayer);
var
  Weights: TArray<TMemoryBlock>;
begin
  if Layer.resident then
    Exit;
  Weights := Matrices(Layer);
  // Get returns borrowed RAM copies. One whole layer fits in the host budget,
  // so collecting its sources cannot evict an earlier source from this batch.
  if FRAMLimit >= FMaxLayerBytes then
    for var I := 0 to High(Weights) do
      if not FCtx.IsWeightResident(Weights[I], 1, Weights[I].Count) then
        Weights[I] := FHost.Get(Weights[I]);
  FCtx.PreloadWeights(Weights);
end;

function TQwenHybrid.Project(const Work: TGPUWorkspace; const Source: TQNNGPUTensor; const Weight: TMemoryBlock; const Rows, InputDim, OutputDim: Integer): TQNNGPUTensor;
begin
  Result := Work.New(Int64(Rows) * OutputDim);
  if Weight.DataType = dtU8 then
    FCtx.Int4Linear(Result, Source, Weight, Rows, InputDim, OutputDim)
  else if Rows = 1 then
    FCtx.VectorLinear(Result, Source, Weight, InputDim, OutputDim)
  else
    FCtx.Linear(Result, Source, Weight, Default(TMemoryBlock), Default(TMemoryBlock), Rows, InputDim, OutputDim);
end;

function TQwenHybrid.Norm(const Work: TGPUWorkspace; const Source: TQNNGPUTensor; const Weight: TMemoryBlock; const Rows, Dim: Integer): TQNNGPUTensor;
begin
  Result := Work.New(Int64(Rows) * Dim);
  FCtx.KernelWeights(GO_NORM, Result.Count, Result, Source, Weight, Default(TMemoryBlock),
    [Rows, Dim, 12, 0, 0, 0, 0, QNNGPUFloatBits(FEpsilon)]); // zero-centered (1 + weight)
end;

function TQwenHybrid.LinearAttention(const Work: TGPUWorkspace; var Layer: TLayer; const Input: TQNNGPUTensor; const Rows: Integer): TQNNGPUTensor;
var
  Mixed, Z, A, B, AB, Gates, Convolved, Output, Gated: TQNNGPUTensor;
  MixedDim, ValueDim: Integer;
begin
  ValueDim := FVHeads * FVDim;
  MixedDim := 2 * FKHeads * FKDim + ValueDim;
  Mixed := Project(Work, Input, Layer.QKV, Rows, FModel.HiddenSize, MixedDim);
  Z := Project(Work, Input, Layer.Z, Rows, FModel.HiddenSize, ValueDim);
  A := Project(Work, Input, Layer.A, Rows, FModel.HiddenSize, FVHeads);
  B := Project(Work, Input, Layer.B, Rows, FModel.HiddenSize, FVHeads);
  AB := Work.Join(B, A, Rows, FVHeads, FVHeads);
  Gates := Work.New(Rows * 2 * FVHeads);
  FCtx.KernelWeights(GO_HYBRID_DELTA_GATES, Rows * FVHeads, Gates, AB, Layer.ALog, Layer.DtBias, [FVHeads]);
  Convolved := Work.New(Int64(Rows) * MixedDim);
  FCtx.Kernel(GO_HYBRID_CONV_SCAN, MixedDim, Convolved, Mixed, Layer.ConvGPU, Layer.ConvState,
    [MixedDim, FConvSize, Rows]);
  FCtx.Kernel(GO_HYBRID_QK_L2, Rows * 2 * FKHeads * FKDim, Convolved, Convolved, nil, nil,
    [Rows, FKHeads, FKDim, MixedDim]);
  Output := Work.New(Int64(Rows) * ValueDim);
  FCtx.Kernel(GO_HYBRID_DELTA_SCAN, Output.Count, Output, Convolved, Gates, Layer.State,
    [FVHeads, FKHeads, FKDim, FVDim, Rows]);
  Gated := Work.New(Output.Count);
  FCtx.Kernel(GO_HYBRID_GATED_NORM, Output.Count, Gated, Output, Layer.NormGPU, Z,
    [FVDim, QNNGPUFloatBits(FEpsilon)]);
  Result := Project(Work, Gated, Layer.O, Rows, ValueDim, FModel.HiddenSize);
end;

function TQwenHybrid.FullAttention(const Work: TGPUWorkspace; var Layer: TLayer; const Input: TQNNGPUTensor; const Rows, Position: Integer): TQNNGPUTensor;
var
  Raw, Q, Gate, K, V, Output, Positions: TQNNGPUTensor;
  HostPositions: TArray<LongInt>;
  QDim, KVDim: Integer;
begin
  QDim := FModel.NumHeads * FModel.HeadDim;
  KVDim := FModel.NumKvHeads * FModel.HeadDim;
  Raw := Project(Work, Input, Layer.Q, Rows, FModel.HiddenSize, 2 * QDim);
  Q := Work.Slice(Raw, Rows * FModel.NumHeads, FModel.HeadDim, 2 * FModel.HeadDim, 0);
  Gate := Work.Slice(Raw, Rows * FModel.NumHeads, FModel.HeadDim, 2 * FModel.HeadDim, FModel.HeadDim);
  K := Project(Work, Input, Layer.K, Rows, FModel.HiddenSize, KVDim);
  V := Project(Work, Input, Layer.V, Rows, FModel.HiddenSize, KVDim);
  Q := Norm(Work, Q, Layer.QNorm, Rows * FModel.NumHeads, FModel.HeadDim);
  K := Norm(Work, K, Layer.KNorm, Rows * FModel.NumKvHeads, FModel.HeadDim);
  Positions := nil;
  if Length(FVisualPositions) > 0 then
  begin
    SetLength(HostPositions, Rows * 3);
    for var R := 0 to Rows - 1 do
      for var Axis := 0 to 2 do
        if (Position + R) * 3 + Axis < Length(FVisualPositions) then
          HostPositions[R * 3 + Axis] := FVisualPositions[(Position + R) * 3 + Axis]
        else
          HostPositions[R * 3 + Axis] := FVisualNext + Position + R - Length(FVisualPositions) div 3;
    Positions := Work.Own(FCtx.UploadRaw(@HostPositions[0], Length(HostPositions)));
  end;
  FCtx.Kernel(GO_HYBRID_ROPE, Rows * FModel.NumHeads * FRotaryDim div 2, Q, Q, Positions, nil,
    [FModel.NumHeads, FRotaryDim, FModel.HeadDim, Position, QNNGPUFloatBits(FModel.RoPETheta), Ord(Positions <> nil)]);
  FCtx.Kernel(GO_HYBRID_ROPE, Rows * FModel.NumKvHeads * FRotaryDim div 2, K, K, Positions, nil,
    [FModel.NumKvHeads, FRotaryDim, FModel.HeadDim, Position, QNNGPUFloatBits(FModel.RoPETheta), Ord(Positions <> nil)]);
  FCtx.Kernel(GO_LLM_CACHE_WRITE, Rows * KVDim, Layer.State, K, V, nil,
    [Rows, FModel.NumKvHeads, FModel.HeadDim, FCapacity, Position]);
  Output := Work.New(Int64(Rows) * QDim);
  FCtx.CachedAttention(Output, Q, Layer.State, Rows, FCapacity, FModel.NumHeads, FModel.NumKvHeads, FModel.HeadDim, Position);
  FCtx.Kernel(GO_HYBRID_ATTN_GATE, Output.Count, Output, Output, Gate, nil, []);
  Result := Project(Work, Output, Layer.O, Rows, QDim, FModel.HiddenSize);
end;

procedure TQwenHybrid.forward(const Ids: TArray<LongInt>; const GPUIds, Logits: TQNNGPUTensor; const Position: Integer);
var
  Work, LayerWork: TGPUWorkspace;
  Tokens, X, N, Y, G, U, Last, Output: TQNNGPUTensor;
  Rows: Integer;
begin
  Rows := Length(Ids);
  FStateValid := False;
  Work := TGPUWorkspace.Create(FCtx);
  try
    Tokens := GPUIds;
    if Tokens = nil then
      Tokens := Work.Own(FCtx.UploadRaw(@Ids[0], Rows));
    X := Work.New(Int64(Rows) * FModel.HiddenSize);
    if FEmbedding.DataType = dtU8 then
      FCtx.KernelWeights(GO_HYBRID_INT4_GATHER, X.Count, X, Tokens, FEmbedding, Default(TMemoryBlock),
        [FModel.HiddenSize, Int4RowBytes(FModel.HiddenSize) div 4, FModel.VocabSize])
    else
      FCtx.Gather(X, Tokens, FEmbedding, Default(TMemoryBlock), Rows, FModel.HiddenSize, FModel.VocabSize, 0);
    for var J := 0 to High(FImageStarts) do
    begin
      var First := Max(Position, FImageStarts[J]);
      var LastRow := Min(Position + Rows, FImageStarts[J] + FImageFeatures[J].Count div FModel.HiddenSize);
      if First < LastRow then
        FCtx.Copy(X, FImageFeatures[J], (LastRow - First) * FModel.HiddenSize,
          (First - Position) * FModel.HiddenSize, (First - FImageStarts[J]) * FModel.HiddenSize);
    end;
    for var I := 0 to High(FLayers) do
    begin
      if Assigned(FCancelCheck) and FCancelCheck() then
        raise ELLMCancelled.Create('Hybrid generation cancelled');
      Preload(FLayers[I]);
      LayerWork := TGPUWorkspace.Create(FCtx);
      try
        var Layer: PLayer := @FLayers[I];
        N := Norm(LayerWork, X, Layer.InputNorm, Rows, FModel.HiddenSize);
        if Layer.Linear then
          Y := LinearAttention(LayerWork, Layer^, N, Rows)
        else
          Y := FullAttention(LayerWork, Layer^, N, Rows, Position);
        LayerWork.Add(X, Y, X.Count);
        N := Norm(LayerWork, X, Layer.PostNorm, Rows, FModel.HiddenSize);
        G := Project(LayerWork, N, Layer.Gate, Rows, FModel.HiddenSize, FModel.IntermediateSize);
        U := Project(LayerWork, N, Layer.Up, Rows, FModel.HiddenSize, FModel.IntermediateSize);
        FCtx.Kernel(GO_SWIGLU, G.Count, G, G, U, nil, [FModel.IntermediateSize, 0]);
        Y := Project(LayerWork, G, Layer.Down, Rows, FModel.IntermediateSize, FModel.HiddenSize);
        LayerWork.Add(X, Y, X.Count);
      finally
        LayerWork.Free;
      end;
    end;
    Last := Work.Copy(X, FModel.HiddenSize, (Rows - 1) * FModel.HiddenSize);
    N := Norm(Work, Last, FNorm, 1, FModel.HiddenSize);
    Output := Project(Work, N, FOutput, 1, FModel.HiddenSize, FModel.VocabSize);
    FCtx.Copy(Logits, Output, FModel.VocabSize);
    FCtx.CheckFinite(Logits);
    FStateValid := True;
  finally
    Work.Free;
  end;
end;

procedure TQwenHybrid.SetVision(const Tokens: TArray<LongInt>; const Images: TArray<TLLMImage>; const Signature: string);
var
  Row, ImageIndex, NextPosition, Base, Width, Height: Integer;
begin
  if FVisionSignature = Signature then
    Exit;
  Reset;
  FStateValid := False; // Token IDs alone do not identify different images.
  for var I := 0 to High(FImageFeatures) do
    FCtx.Recycle(FImageFeatures[I]);
  FImageFeatures := nil;
  FImageStarts := nil;
  FVisualPositions := nil;
  FVisionSignature := '';
  if Length(Images) = 0 then
    Exit;
  if FRotaryDim <> 64 then
    raise ENotSupportedException.Create('Unsupported multimodal rotary configuration');
  SetLength(FImageFeatures, Length(Images));
  SetLength(FImageStarts, Length(Images));
  SetLength(FVisualPositions, Length(Tokens) * 3);
  Row := 0;
  ImageIndex := 0;
  NextPosition := 0;
  while Row < Length(Tokens) do
  begin
    if Tokens[Row] = 248056 then
    begin
      if ImageIndex >= Length(Images) then
        raise EArgumentException.Create('Unexpected visual placeholder');
      var Image := Images[ImageIndex];
      Width := Image.GridWidth div 2;
      Height := Image.GridHeight div 2;
      if (Width < 1) or (Height < 1) or (Length(Image.Features) <> Width * Height * FModel.HiddenSize) then
        raise EArgumentException.Create('Missing or incompatible visual features');
      FImageStarts[ImageIndex] := Row;
      FImageFeatures[ImageIndex] := FCtx.UploadRaw(@Image.Features[0], Length(Image.Features));
      Base := NextPosition;
      for var Y := 0 to Height - 1 do
        for var X := 0 to Width - 1 do
        begin
          if (Row >= Length(Tokens)) or (Tokens[Row] <> 248056) then
            raise EArgumentException.Create('Visual span mismatch');
          FVisualPositions[Row * 3] := Base;
          FVisualPositions[Row * 3 + 1] := Base + Y;
          FVisualPositions[Row * 3 + 2] := Base + X;
          Inc(Row);
        end;
      NextPosition := Base + Max(Width, Height);
      Inc(ImageIndex);
    end
    else
    begin
      for var Axis := 0 to 2 do
        FVisualPositions[Row * 3 + Axis] := NextPosition;
      Inc(NextPosition);
      Inc(Row);
    end;
  end;
  if ImageIndex <> Length(Images) then
    raise EArgumentException.Create('Unused visual features');
  FVisualNext := NextPosition;
  FVisionSignature := Signature;
end;

procedure TQwenHybrid.Reset;
begin
  FStateValid := True;
  for var Layer in FLayers do
  begin
    FCtx.Kernel(GO_FILL, Layer.State.Count, Layer.State, nil, nil, nil, [0]);
    if Layer.Linear then
      FCtx.Kernel(GO_FILL, Layer.ConvState.Count, Layer.ConvState, nil, nil, nil, [0]);
  end;
end;

procedure TQwenHybrid.Status(var Value: TLLMMemoryStatus);
begin
  Value.IsHybrid := True;
  Value.DenseQuantization := FQuantization;
  Value.ResidentLayers := FResidentLayers;
  Value.StreamedLayers := Length(FLayers) - FResidentLayers;
  Value.FixedGPUWeightBytes := FFixedBytes;
  Value.RecurrentBytes := FRecurrentBytes;
  Value.ActivationBytes := FCtx.AllocatedBytes;
  if FCache <> nil then
  begin
    Value.QuantizedMappedWeightBytes := FCache.MappedBytes;
    Value.ExpertCacheDirectory := FCache.Directory;
  end;
  FHost.Status(Value);
end;

function TQwenHybrid.StateIdentity: NativeUInt;
begin
  Result := 0;
  if Length(FLayers) > 0 then
    Result := NativeUInt(FLayers[0].State);
end;

end.

