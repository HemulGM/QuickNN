unit QuickNN.Models.QwenFlashNext;

{$POINTERMATH ON}

interface

uses
  System.SysUtils, System.JSON, System.Generics.Collections, QuickNN.Common,
  QuickNN.SafeTensors, QuickNN.Vulkan, QuickNN.Vulkan.Operations,
  QuickNN.Models.Qwen3, QuickNN.Models.LLM.Types, QuickNN.Models.LLM.MoE;

type
  // Native qwen4exp: HC residual streams, Gated DeltaNet, QSA and sparse PLE.
  // Only selected expert slices are uploaded. The PLE table stays memory mapped.
  TQwenFlashNext = class
  private
    type
      TMixer = record
        Gamma: TQNNGPUTensor;
        Down, Up, Inject: TMemoryBlock;
      end;
      TLayer = record
        Linear: Boolean;
        AttentionMixer, FeedForwardMixer: TMixer;
        Q, K, V, O, QNorm, KNorm, Z, A, B, Decay, Bias: TMemoryBlock;
        SharedGate, SharedUp, SharedDown, SharedRouter: TMemoryBlock;
        IndexQ, IndexK, IndexQNorm, IndexKNorm: TMemoryBlock;
        State, ConvState, Conv, GatedNorm, IndexRaw, IndexPools: TQNNGPUTensor;
        PleKey, PleValue: TMemoryBlock;
        PleKeyNorm, PleQueryNorm, PleConvNorm, PleConv, PleState: TQNNGPUTensor;
      end;
      PLayer = ^TLayer;
  private
    FCtx: TQNNGPUContext;
    FModel: PQwen3Model;
    FFiles: TSafeTensorFiles;
    FWeights: TList<TMemoryBlock>;
    FMoE: TLLMMoE;
    FLayers: TArray<TLayer>;
    FEmbedding, FOutput: TMemoryBlock;
    FHead: TMixer;
    FPleTable: PSafeTensor;
    FHistory: TArray<LongInt>;
    FMultipliers: TArray<UInt64>;
    FOffsets, FVocabulary: TArray<Integer>;
    FCapacity, FHC, FLowRank, FKHeads, FVHeads, FKDim, FVDim, FConvSize, FRotary: Integer;
    FIndexHeads, FIndexDim, FIndexTopK, FPoolSize: Integer;
    FPleLayer, FPleNgram, FPleHeads, FPleDim, FPleKernel, FPleEOS: Integer;
    FEpsilon: Single;
    FCacheBytes, FRecurrentBytes, FFixedBytes: UInt64;
    FValid: Boolean;
    FCancelCheck: TFunc<Boolean>;
    function Weight(const Name: string; const Rows, Columns: Integer): TMemoryBlock;
    function Parameter(const Name: string; const Count: Integer): TQNNGPUTensor;
    function Project(const Work: TGPUWorkspace; const Input: TQNNGPUTensor;
      const Matrix: TMemoryBlock; const Rows, Inputs, Outputs: Integer): TQNNGPUTensor;
    function GroupNorm(const Work: TGPUWorkspace; const Input, Gamma: TQNNGPUTensor; const Rows: Integer): TQNNGPUTensor;
    function Mix(const Work: TGPUWorkspace; const Input: TQNNGPUTensor; const Mixer: TMixer;
      const Rows: Integer; out Inject: TQNNGPUTensor): TQNNGPUTensor;
    function LinearAttention(const Work: TGPUWorkspace; var Layer: TLayer;
      const Input: TQNNGPUTensor; const Rows: Integer): TQNNGPUTensor;
    function FullAttention(const Work: TGPUWorkspace; var Layer: TLayer;
      const Input: TQNNGPUTensor; const Rows, Position: Integer): TQNNGPUTensor;
    function PleEmbedding(const Work: TGPUWorkspace; const Ids: TArray<LongInt>): TQNNGPUTensor;
    procedure ApplyPle(const Work: TGPUWorkspace; var Layer: TLayer; const Input, Embedding: TQNNGPUTensor; const Rows: Integer);
    procedure CheckCancelled;
  public
    constructor Create(const Context: TQNNGPUContext; var Model: TQwen3Model;
      const Config: TJSONValue; const Capacity: Integer; const Memory: TLLMMemoryOptions);
    destructor Destroy; override;
    procedure Forward(const Ids: TArray<LongInt>; const GPUIds, Logits: TQNNGPUTensor; const Position: Integer);
    procedure Reset;
    procedure Status(var Value: TLLMMemoryStatus);
    function StateIdentity: NativeUInt;
    property CacheBytes: UInt64 read FCacheBytes;
    property StateValid: Boolean read FValid;
    property CancelCheck: TFunc<Boolean> read FCancelCheck write FCancelCheck;
  end;

implementation

uses
  System.Math, QuickNN.GGUF.Quantization, QuickNN.Models.LLM.GGUF;

function TQwenFlashNext.Weight(const Name: string; const Rows, Columns: Integer): TMemoryBlock;
begin
  var Tensor := FFiles.GetTensor(Name);
  if (Tensor = nil) or not (Tensor.DType in [StF32,StBF16,StF16,StU8]) then
    raise EArgumentException.Create('Missing qwen4exp tensor: '+Name);
  if Columns = 0 then
  begin
    if Tensor.Count <> Rows then raise EArgumentException.Create('Invalid qwen4exp parameter: '+Name);
    Result := Tensor.AsSingle(True);
  end
  else
  begin
    if not (((Tensor.Ndim = 2) and (Tensor.Shape[0] = Rows) and (Tensor.Shape[1] = Columns))
      or ((Rows = 1) and (Tensor.Ndim = 1) and (Tensor.Count = Columns))) then
      raise EArgumentException.Create('Invalid qwen4exp matrix: '+Name);
    Result := FFiles.GetLinearWeight(Name,True);
    Result.Shape := [Rows,Columns];
  end;
  FWeights.Add(Result);
  if Result.Encoding <> weNative then Inc(FFixedBytes,GGUFQuantBytes(Result.Encoding,Result.Size))
  else Inc(FFixedBytes,UInt64(Result.Size)*UInt64(DATATYPE_BITS[Result.DataType] div 8));
end;

function TQwenFlashNext.Parameter(const Name: string; const Count: Integer): TQNNGPUTensor;
begin
  var Value := Weight(Name,Count,0);
  Result := FCtx.Upload(Value);
end;

constructor TQwenFlashNext.Create(const Context: TQNNGPUContext; var Model: TQwen3Model;
  const Config: TJSONValue; const Capacity: Integer; const Memory: TLLMMemoryOptions);
var
  Meta, Text: TJSONValue;
  function IntegerValue(const Key: string): Integer;
  begin Result := LLMGGUFMetadata(Meta,'qwen4exp.'+Key).AsType<Integer> end;
  procedure LoadMixer(var Mixer: TMixer; const Prefix: string; const FinalHead: Boolean = False);
  begin
    Mixer.Gamma := Parameter(Prefix+'norm.weight',FHC*Model.HiddenSize);
    Mixer.Down := Weight(Prefix+'down.weight',FLowRank,FHC*Model.HiddenSize);
    Mixer.Up := Weight(Prefix+'up.weight',FHC*Model.HiddenSize,FLowRank);
    if not FinalHead then Mixer.Inject := Weight(Prefix+'inject.weight',FHC,FHC*Model.HiddenSize);
  end;
  function State(const Count: Int64): TQNNGPUTensor;
  begin
    if Count > High(Integer) then raise EArgumentException.Create('qwen4exp state is too large');
    Inc(FCacheBytes,UInt64(Count)*4); Result := FCtx.NewTensor(Count);
  end;
begin
  inherited Create;
  FCtx := Context; FModel := @Model; FCapacity := Capacity;
  FWeights := TList<TMemoryBlock>.Create;
  Text := Config.FindValue('text_config'); Meta := Config.FindValue('architecture_metadata');
  if (Text = nil) or (Meta = nil) then raise EArgumentException.Create('Missing qwen4exp configuration');
  Model.HiddenSize := Text.GetValue<Integer>('hidden_size');
  Model.IntermediateSize := Text.GetValue<Integer>('intermediate_size');
  Model.NumHeads := Text.GetValue<Integer>('num_attention_heads');
  Model.NumKvHeads := Text.GetValue<Integer>('num_key_value_heads');
  Model.HeadDim := Text.GetValue<Integer>('head_dim');
  Model.VocabSize := Text.GetValue<Integer>('vocab_size');
  Model.RoPETheta := LLMGGUFMetadata(Meta,'qwen4exp.rope.freq_base').AsType<Single>;
  var Layers := Text.GetValue<Integer>('num_hidden_layers');
  FHC := IntegerValue('hyper_connection.count'); FLowRank := IntegerValue('hyper_connection.low_rank');
  FKHeads := IntegerValue('ssm.group_count'); FVHeads := IntegerValue('ssm.time_step_rank');
  FKDim := IntegerValue('ssm.state_size'); FVDim := Text.GetValue<Integer>('linear_value_head_dim');
  FConvSize := IntegerValue('ssm.conv_kernel'); FRotary := IntegerValue('rope.dimension_count');
  FIndexHeads := IntegerValue('attention.indexer.head_count'); FIndexDim := IntegerValue('attention.indexer.key_length');
  FIndexTopK := IntegerValue('attention.indexer.top_k');
  var Ratios := (LLMGGUFMetadata(Meta,'qwen4exp.attention.compress_ratios') as TJSONArray);
  if Ratios.Count <> Layers then raise EArgumentException.Create('Invalid qwen4exp QSA ratios');
  FPoolSize := 0;
  for var Ratio in Ratios do if Ratio.AsType<Integer> > 0 then
  begin
    if (FPoolSize <> 0) and (FPoolSize <> Ratio.AsType<Integer>) then raise ENotSupportedException.Create('Mixed QSA pool sizes');
    FPoolSize := Ratio.AsType<Integer>;
  end;
  FEpsilon := Text.GetValue<Single>('rms_norm_eps');
  if (Layers < 1) or (Layers > 256) or (Model.HiddenSize < 1) or (Model.HiddenSize > 32768)
    or (FHC < 2) or (FHC > 16) or (FLowRank < 1) or (FLowRank > 4096)
    or (Model.NumHeads < 1) or (Model.NumHeads > 256) or (Model.NumKvHeads < 1) or (Model.NumKvHeads > 256) or (Model.NumHeads mod Model.NumKvHeads <> 0)
    or (Model.HeadDim < 2) or (Model.HeadDim > 1152) or (FRotary < 2) or (FRotary > Model.HeadDim) or Odd(FRotary)
    or (FKHeads < 1) or (FVHeads < 1) or (FVHeads > 256) or (FVHeads mod FKHeads <> 0)
    or (FKDim < 1) or (FKDim > 256) or (FVDim < 1) or (FVDim > 256)
    or (FConvSize < 1) or (FConvSize > 16) or (FPoolSize < 2) or (FPoolSize > 64)
    or (FIndexDim < FRotary) or (FIndexDim > 1152) or (FIndexHeads < 1) or (FIndexHeads > 256)
    or (FIndexTopK < FPoolSize) or (FIndexTopK mod FPoolSize <> 0)
    or not (FEpsilon > 0) or not (Model.RoPETheta > 0)
    or (Capacity < 2) or (Capacity > Text.GetValue<Integer>('max_position_embeddings')) then
    raise EArgumentException.Create('Invalid qwen4exp geometry or context');
  var PleLayers := (LLMGGUFMetadata(Meta,'qwen4exp.ple.layers') as TJSONArray);
  if PleLayers.Count <> 1 then raise ENotSupportedException.Create('qwen4exp needs exactly one PLE module');
  FPleLayer := PleLayers[0].AsType<Integer>;
  FPleNgram := IntegerValue('ple.ngram_size'); FPleHeads := IntegerValue('ple.heads_per_ngram');
  FPleDim := IntegerValue('embedding_length_per_layer_input'); FPleKernel := IntegerValue('ple.conv_kernel');
  FPleEOS := IntegerValue('ple.eos_token_id');
  if (FPleLayer < 0) or (FPleLayer >= Layers) or (FPleNgram < 2) or (FPleNgram > 16)
    or (FPleHeads < 1) or ((FPleNgram-1)*FPleHeads > 256) or (FPleDim < 1) or (FPleDim > Model.HiddenSize)
    or (FPleDim*(FPleNgram-1)*FPleHeads <> Model.HiddenSize) or (FPleKernel < 1) or (FPleKernel > 16) then
    raise EArgumentException.Create('Invalid qwen4exp PLE geometry');
  var Multipliers := (LLMGGUFMetadata(Meta,'qwen4exp.ple.layer_multipliers') as TJSONArray);
  var Offsets := (LLMGGUFMetadata(Meta,'qwen4exp.ple.head_offsets') as TJSONArray);
  var Vocabulary := (LLMGGUFMetadata(Meta,'qwen4exp.ple.head_vocab_sizes') as TJSONArray);
  if (Multipliers.Count <> FPleNgram) or (Offsets.Count <> (FPleNgram-1)*FPleHeads) or (Vocabulary.Count <> Offsets.Count) then
    raise EArgumentException.Create('Invalid PLE hash arrays');
  SetLength(FMultipliers,Multipliers.Count); SetLength(FOffsets,Offsets.Count); SetLength(FVocabulary,Offsets.Count);
  for var I := 0 to High(FMultipliers) do FMultipliers[I] := Multipliers[I].AsType<UInt64>;
  for var I := 0 to High(FOffsets) do
  begin FOffsets[I] := Offsets[I].AsType<Integer>; FVocabulary[I] := Vocabulary[I].AsType<Integer> end;
  var KindsCheck := Text.GetValue<TJSONArray>('layer_types');
  var ExpectedBytes: UInt64 := UInt64(FHC)*UInt64(Model.HiddenSize)*UInt64((FPleKernel-1)*FPleNgram+1)*4;
  for var I := 0 to Layers-1 do
    if KindsCheck[I].Value='linear_attention' then
      Inc(ExpectedBytes,(UInt64(FVHeads)*UInt64(FKDim)*UInt64(FVDim)
        +UInt64(2*FKHeads*FKDim+FVHeads*FVDim)*UInt64(FConvSize))*4)
    else Inc(ExpectedBytes,(UInt64(2)*UInt64(Capacity)*UInt64(Model.NumKvHeads)*UInt64(Model.HeadDim)
      +UInt64(128)*UInt64(Capacity)+UInt64(Capacity)*UInt64(FIndexDim)
      +UInt64((Capacity+FPoolSize-1) div FPoolSize)*UInt64(FIndexDim))*4);
  if ExpectedBytes>UInt64(1536)*1024*1024 then
    raise EArgumentException.Create('qwen4exp KV/recurrent/indexer state exceeds 1.5 GiB; reduce context');
  Model.SfFiles := LLMOpenGGUF(Config.GetValue<string>('quicknn_gguf_file')); FFiles := Model.SfFiles;
  SetLength(Model.Layers,Layers); SetLength(FLayers,Layers);
  FEmbedding := Weight('token_embd.weight',Model.VocabSize,Model.HiddenSize);
  FOutput := Weight('output.weight',Model.VocabSize,Model.HiddenSize);
  LoadMixer(FHead,'output_hc_',True);
  FPleTable := FFiles.GetTensor('per_layer_token_embd.weight');
  if (FPleTable = nil) or (FPleTable.Ndim <> 2) or (FPleTable.Shape[1] <> FPleDim) or (FPleTable.Encoding = weNative) then
    raise EArgumentException.Create('Missing packed PLE table in the GGUF shards');
  for var I := 0 to High(FOffsets) do
    if (FOffsets[I] < 0) or (FVocabulary[I] < 1) or (Int64(FOffsets[I])+FVocabulary[I] > FPleTable.Shape[0]) then
      raise EArgumentException.Create('PLE hash row range exceeds the mapped table');
  var MixedDim := 2*FKHeads*FKDim+FVHeads*FVDim;
  var KVDim := Model.NumKvHeads*Model.HeadDim;
  var QDim := Model.NumHeads*Model.HeadDim;
  var SharedDim := IntegerValue('expert_shared_feed_forward_length');
  var Kinds := Text.GetValue<TJSONArray>('layer_types');
  for var I := 0 to Layers-1 do
  begin
    var Layer: PLayer := @FLayers[I]; var Prefix := Format('blk.%d.',[I]);
    LoadMixer(Layer.AttentionMixer,Prefix+'hc_attn_'); LoadMixer(Layer.FeedForwardMixer,Prefix+'hc_ffn_');
    Layer.Linear := Kinds[I].Value = 'linear_attention';
    if Layer.Linear then
    begin
      Layer.Q := Weight(Prefix+'attn_qkv.weight',MixedDim,Model.HiddenSize);
      Layer.Z := Weight(Prefix+'attn_gate.weight',FVHeads*FVDim,Model.HiddenSize);
      Layer.A := Weight(Prefix+'ssm_alpha.weight',FVHeads,Model.HiddenSize);
      Layer.B := Weight(Prefix+'ssm_beta.weight',FVHeads,Model.HiddenSize);
      Layer.O := Weight(Prefix+'ssm_out.weight',Model.HiddenSize,FVHeads*FVDim);
      Layer.Decay := Weight(Prefix+'ssm_a',FVHeads,0); Layer.Bias := Weight(Prefix+'ssm_dt.bias',FVHeads,0);
      Layer.Conv := Parameter(Prefix+'ssm_conv1d.weight',MixedDim*FConvSize);
      Layer.GatedNorm := Parameter(Prefix+'ssm_norm.weight',FVDim);
      Layer.State := State(Int64(FVHeads)*FKDim*FVDim); Layer.ConvState := State(Int64(MixedDim)*FConvSize);
      Inc(FRecurrentBytes,UInt64(Layer.State.Count+Layer.ConvState.Count)*4);
    end
    else
    begin
      if Ratios[I].AsType<Integer> <> FPoolSize then raise EArgumentException.Create('Missing QSA ratio on a full-attention layer');
      Layer.Q := Weight(Prefix+'attn_q.weight',2*QDim,Model.HiddenSize);
      Layer.K := Weight(Prefix+'attn_k.weight',KVDim,Model.HiddenSize);
      Layer.V := Weight(Prefix+'attn_v.weight',KVDim,Model.HiddenSize);
      Layer.O := Weight(Prefix+'attn_output.weight',Model.HiddenSize,QDim);
      Layer.QNorm := Weight(Prefix+'attn_q_norm.weight',Model.HeadDim,0);
      Layer.KNorm := Weight(Prefix+'attn_k_norm.weight',Model.HeadDim,0);
      Layer.IndexQ := Weight(Prefix+'indexer.q_proj.weight',FIndexHeads*FIndexDim,Model.HiddenSize);
      Layer.IndexK := Weight(Prefix+'indexer.k_proj.weight',FIndexDim,Model.HiddenSize);
      Layer.IndexQNorm := Weight(Prefix+'indexer.q_norm.weight',FIndexDim,0);
      Layer.IndexKNorm := Weight(Prefix+'indexer.k_norm.weight',FIndexDim,0);
      Layer.State := State(Int64(2)*Capacity*KVDim+Int64(128)*Capacity);
      Layer.IndexRaw := State(Int64(Capacity)*FIndexDim);
      Layer.IndexPools := State(Int64((Capacity+FPoolSize-1) div FPoolSize)*FIndexDim);
    end;
    Layer.SharedGate := Weight(Prefix+'ffn_gate_shexp.weight',SharedDim,Model.HiddenSize);
    Layer.SharedUp := Weight(Prefix+'ffn_up_shexp.weight',SharedDim,Model.HiddenSize);
    Layer.SharedDown := Weight(Prefix+'ffn_down_shexp.weight',Model.HiddenSize,SharedDim);
    Layer.SharedRouter := Weight(Prefix+'ffn_gate_inp_shexp.weight',1,Model.HiddenSize);
    if I = FPleLayer then
    begin
      Layer.PleKey := Weight(Prefix+'ple_key.weight',FHC*Model.HiddenSize,Model.HiddenSize);
      Layer.PleValue := Weight(Prefix+'ple_value.weight',Model.HiddenSize,Model.HiddenSize);
      Layer.PleKeyNorm := Parameter(Prefix+'ple_norm_key.weight',FHC*Model.HiddenSize);
      Layer.PleQueryNorm := Parameter(Prefix+'ple_norm_query.weight',FHC*Model.HiddenSize);
      Layer.PleConvNorm := Parameter(Prefix+'ple_norm_conv.weight',FHC*Model.HiddenSize);
      Layer.PleConv := Parameter(Prefix+'ple_conv1d.weight',FHC*Model.HiddenSize*FPleKernel);
      Layer.PleState := State(Int64(FHC)*Model.HiddenSize*((FPleKernel-1)*FPleNgram+1));
      Inc(FRecurrentBytes,UInt64(Layer.PleState.Count)*4);
    end;
  end;
  if FCacheBytes > UInt64(1536)*1024*1024 then raise EArgumentException.Create('qwen4exp KV/recurrent/indexer state exceeds 1.5 GiB; reduce context');
  FMoE := TLLMMoE.Create(FFiles,Config,Layers,Model.HiddenSize,Memory.RAMCacheLimitBytes,Memory.ExpertBatchBytes,eqNative);
  Inc(FFixedBytes,FMoE.RouterBytes);
  if FFixedBytes+FMoE.ExpertBytes*UInt64(Config.GetValue<Integer>('num_experts_per_tok')) > QNNGetGPUStatus.WeightCacheLimit then
    raise EArgumentException.Create('Increase GPU weight-cache budget: fixed qwen4exp weights plus one routed expert batch do not fit');
  for var Matrix in FWeights do
    if Length(Matrix.Shape) = 2 then FCtx.RetainWeight(Matrix,Matrix.Shape[0],Matrix.Shape[1])
    else FCtx.RetainWeight(Matrix,1,Matrix.Count);
  FMoE.RetainRouters(FCtx);
  Reset;
end;

destructor TQwenFlashNext.Destroy;
begin
  FMoE.Free;
  if FWeights <> nil then for var Matrix in FWeights do Matrix.Free;
  FWeights.Free;
  inherited;
end;

procedure TQwenFlashNext.CheckCancelled;
begin
  if Assigned(FCancelCheck) and FCancelCheck() then raise ELLMCancelled.Create('qwen4exp generation cancelled');
end;

function TQwenFlashNext.Project(const Work: TGPUWorkspace; const Input: TQNNGPUTensor;
  const Matrix: TMemoryBlock; const Rows, Inputs, Outputs: Integer): TQNNGPUTensor;
begin
  Result := Work.New(Int64(Rows)*Outputs);
  if (Rows=1) and (Matrix.Encoding=weNative) then FCtx.VectorLinear(Result,Input,Matrix,Inputs,Outputs)
  else FCtx.Linear(Result,Input,Matrix,Default(TMemoryBlock),Default(TMemoryBlock),Rows,Inputs,Outputs);
end;

function TQwenFlashNext.GroupNorm(const Work: TGPUWorkspace; const Input, Gamma: TQNNGPUTensor; const Rows: Integer): TQNNGPUTensor;
begin
  Result := Work.New(Int64(Rows)*FHC*FModel.HiddenSize);
  FCtx.Kernel(GO_FLASH_GROUP_NORM,Result.Count,Result,Input,Gamma,nil,[FModel.HiddenSize,FHC,QNNGPUFloatBits(FEpsilon)]);
end;

function TQwenFlashNext.Mix(const Work: TGPUWorkspace; const Input: TQNNGPUTensor;
  const Mixer: TMixer; const Rows: Integer; out Inject: TQNNGPUTensor): TQNNGPUTensor;
begin
  var N := GroupNorm(Work,Input,Mixer.Gamma,Rows);
  var Low := Project(Work,N,Mixer.Down,Rows,FHC*FModel.HiddenSize,FLowRank);
  FCtx.Kernel(GO_SCALE,Low.Count,Low,Low,nil,nil,[QNNGPUFloatBits(1/FHC)]);
  var Activated := Work.Unary(Low,Low.Count);
  var Gate := Project(Work,Activated,Mixer.Up,Rows,FLowRank,FHC*FModel.HiddenSize);
  Result := Work.New(Int64(Rows)*FModel.HiddenSize);
  FCtx.Kernel(GO_FLASH_HC_MIX,Result.Count,Result,N,Gate,nil,[FModel.HiddenSize,FHC]);
  Inject := nil;
  if Mixer.Inject.IsAssigned then
  begin
    Inject := Project(Work,N,Mixer.Inject,Rows,FHC*FModel.HiddenSize,FHC);
    FCtx.Kernel(GO_SCALE,Inject.Count,Inject,Inject,nil,nil,[QNNGPUFloatBits(1/FHC)]);
  end;
end;

function TQwenFlashNext.LinearAttention(const Work: TGPUWorkspace; var Layer: TLayer;
  const Input: TQNNGPUTensor; const Rows: Integer): TQNNGPUTensor;
begin
  var ValueDim := FVHeads*FVDim; var MixedDim := 2*FKHeads*FKDim+ValueDim;
  var Mixed := Project(Work,Input,Layer.Q,Rows,FModel.HiddenSize,MixedDim);
  var Z := Project(Work,Input,Layer.Z,Rows,FModel.HiddenSize,ValueDim);
  var A := Project(Work,Input,Layer.A,Rows,FModel.HiddenSize,FVHeads);
  var B := Project(Work,Input,Layer.B,Rows,FModel.HiddenSize,FVHeads);
  var AB := Work.Join(B,A,Rows,FVHeads,FVHeads); var Gates := Work.New(Rows*2*FVHeads);
  FCtx.KernelWeights(GO_HYBRID_DELTA_GATES,Rows*FVHeads,Gates,AB,Layer.Decay,Layer.Bias,[FVHeads,1]);
  var Conv := Work.New(Int64(Rows)*MixedDim);
  FCtx.Kernel(GO_HYBRID_CONV_SCAN,MixedDim,Conv,Mixed,Layer.Conv,Layer.ConvState,[MixedDim,FConvSize,Rows]);
  FCtx.Kernel(GO_HYBRID_QK_L2,Rows*2*FKHeads*FKDim,Conv,Conv,nil,nil,[Rows,FKHeads,FKDim,MixedDim]);
  var Output := Work.New(Int64(Rows)*ValueDim);
  FCtx.Kernel(GO_HYBRID_DELTA_SCAN,Output.Count,Output,Conv,Gates,Layer.State,[FVHeads,FKHeads,FKDim,FVDim,Rows,1]);
  var Gated := Work.New(Output.Count);
  FCtx.Kernel(GO_HYBRID_GATED_NORM,Output.Count,Gated,Output,Layer.GatedNorm,Z,[FVDim,QNNGPUFloatBits(FEpsilon),1]);
  Result := Project(Work,Gated,Layer.O,Rows,ValueDim,FModel.HiddenSize);
end;

function TQwenFlashNext.FullAttention(const Work: TGPUWorkspace; var Layer: TLayer;
  const Input: TQNNGPUTensor; const Rows, Position: Integer): TQNNGPUTensor;
begin
  var QDim := FModel.NumHeads*FModel.HeadDim; var KVDim := FModel.NumKvHeads*FModel.HeadDim;
  var Raw := Project(Work,Input,Layer.Q,Rows,FModel.HiddenSize,2*QDim);
  var Q := Work.Slice(Raw,Rows*FModel.NumHeads,FModel.HeadDim,2*FModel.HeadDim,0);
  var Gate := Work.Slice(Raw,Rows*FModel.NumHeads,FModel.HeadDim,2*FModel.HeadDim,FModel.HeadDim);
  var K := Project(Work,Input,Layer.K,Rows,FModel.HiddenSize,KVDim);
  var V := Project(Work,Input,Layer.V,Rows,FModel.HiddenSize,KVDim);
  Q := Work.Norm(Q,Layer.QNorm,Default(TMemoryBlock),Rows*FModel.NumHeads,FModel.HeadDim,False,FEpsilon);
  K := Work.Norm(K,Layer.KNorm,Default(TMemoryBlock),Rows*FModel.NumKvHeads,FModel.HeadDim,False,FEpsilon);
  FCtx.Kernel(GO_HYBRID_ROPE,Rows*FModel.NumHeads*FRotary div 2,Q,Q,nil,nil,[FModel.NumHeads,FRotary,FModel.HeadDim,Position,QNNGPUFloatBits(FModel.RoPETheta)]);
  FCtx.Kernel(GO_HYBRID_ROPE,Rows*FModel.NumKvHeads*FRotary div 2,K,K,nil,nil,[FModel.NumKvHeads,FRotary,FModel.HeadDim,Position,QNNGPUFloatBits(FModel.RoPETheta)]);
  FCtx.Kernel(GO_LLM_CACHE_WRITE,Rows*KVDim,Layer.State,K,V,nil,[Rows,FModel.NumKvHeads,FModel.HeadDim,FCapacity,Position]);
  var IndexK := Project(Work,Input,Layer.IndexK,Rows,FModel.HiddenSize,FIndexDim);
  FCtx.Copy(Layer.IndexRaw,IndexK,IndexK.Count,Position*FIndexDim);
  var FirstPool := Position div FPoolSize; var NewPools := (Position+Rows) div FPoolSize-FirstPool;
  if NewPools > 0 then
  begin
    var Pooled := Work.New(NewPools*FIndexDim);
    FCtx.Kernel(GO_FLASH_QSA_POOL,Pooled.Count,Pooled,Layer.IndexRaw,nil,nil,[FIndexDim,FPoolSize,FirstPool]);
    Pooled := Work.Norm(Pooled,Layer.IndexKNorm,Default(TMemoryBlock),NewPools,FIndexDim,False,FEpsilon);
    FCtx.Kernel(GO_HYBRID_ROPE,NewPools*FRotary div 2,Pooled,Pooled,nil,nil,[1,FRotary,FIndexDim,FirstPool*FPoolSize,QNNGPUFloatBits(FModel.RoPETheta),0,FPoolSize]);
    FCtx.Copy(Layer.IndexPools,Pooled,Pooled.Count,FirstPool*FIndexDim);
  end;
  var IndexQ := Project(Work,Input,Layer.IndexQ,Rows,FModel.HiddenSize,FIndexHeads*FIndexDim);
  IndexQ := Work.Norm(IndexQ,Layer.IndexQNorm,Default(TMemoryBlock),Rows*FIndexHeads,FIndexDim,False,FEpsilon);
  FCtx.Kernel(GO_HYBRID_ROPE,Rows*FIndexHeads*FRotary div 2,IndexQ,IndexQ,nil,nil,[FIndexHeads,FRotary,FIndexDim,Position,QNNGPUFloatBits(FModel.RoPETheta)]);
  var PoolCapacity := (FCapacity+FPoolSize-1) div FPoolSize;
  var Scores := Work.New(Int64(Rows)*PoolCapacity);
  FCtx.Kernel(GO_FLASH_QSA_SCORE,Scores.Count,Scores,IndexQ,Layer.IndexPools,nil,[PoolCapacity,FIndexHeads,FIndexDim,Position,FPoolSize]);
  var Mask := Work.New(Int64(Rows)*FCapacity);
  FCtx.Kernel(GO_FLASH_QSA_MASK,Scores.Count,Mask,Scores,nil,nil,[PoolCapacity,FCapacity,Position,FPoolSize,FIndexTopK div FPoolSize]);
  FCtx.Copy(Layer.State,Mask,Mask.Count,2*FCapacity*KVDim);
  var Output := Work.New(Int64(Rows)*QDim);
  FCtx.CachedAttention(Output,Q,Layer.State,Rows,FCapacity,FModel.NumHeads,FModel.NumKvHeads,FModel.HeadDim,Position,0,0,0,True);
  FCtx.Kernel(GO_HYBRID_ATTN_GATE,Output.Count,Output,Output,Gate,nil,[]);
  Result := Project(Work,Output,Layer.O,Rows,QDim,FModel.HiddenSize);
end;

function TQwenFlashNext.PleEmbedding(const Work: TGPUWorkspace; const Ids: TArray<LongInt>): TQNNGPUTensor;
begin
  var Values: TArray<Single>; SetLength(Values,Length(Ids)*FModel.HiddenSize);
  var Context: TArray<UInt64>; SetLength(Context,FPleNgram);
  var RowBytes := GGUFQuantBytes(FPleTable.Encoding,FPleDim);
  for var I := 0 to High(Ids) do
  begin
    Context[0] := Ids[I]; var Cut := False;
    for var J := 1 to FPleNgram-1 do
    begin
      var Index := Length(FHistory)+I-J; var Token := FPleEOS;
      if Index >= Length(FHistory) then Token := Ids[Index-Length(FHistory)]
      else if Index >= 0 then Token := FHistory[Index];
      Cut := Cut or (Token=FPleEOS); if Cut then Token := FPleEOS;
      Context[J] := UInt64(Token);
    end;
    for var N := 2 to FPleNgram do
    begin
      var Mixed: UInt64 := Context[0]*FMultipliers[0];
      for var J := 1 to N-1 do Mixed := Mixed xor (Context[J]*FMultipliers[J]);
      for var H := 0 to FPleHeads-1 do
      begin
        var Head := (N-2)*FPleHeads+H;
        var Row := Mixed mod UInt64(FVocabulary[Head])+UInt64(FOffsets[Head]);
        GGUFDequantize(PByte(FPleTable.Data)+NativeInt(Row*RowBytes),FPleTable.Encoding,FPleDim,
          @Values[I*FModel.HiddenSize+Head*FPleDim]);
      end;
    end;
  end;
  Result := Work.Own(FCtx.UploadRaw(@Values[0],Length(Values)));
end;

procedure TQwenFlashNext.ApplyPle(const Work: TGPUWorkspace; var Layer: TLayer;
  const Input, Embedding: TQNNGPUTensor; const Rows: Integer);
begin
  var Key := Project(Work,Embedding,Layer.PleKey,Rows,FModel.HiddenSize,FHC*FModel.HiddenSize);
  var Value := Project(Work,Embedding,Layer.PleValue,Rows,FModel.HiddenSize,FModel.HiddenSize);
  Key := GroupNorm(Work,Key,Layer.PleKeyNorm,Rows);
  var Query := GroupNorm(Work,Input,Layer.PleQueryNorm,Rows);
  var Gated := Work.New(Input.Count);
  FCtx.Kernel(GO_FLASH_PLE_GATE,Gated.Count,Gated,Key,Query,Value,[FModel.HiddenSize,FHC]);
  var N := GroupNorm(Work,Gated,Layer.PleConvNorm,Rows); var Conv := Work.New(Input.Count);
  FCtx.Kernel(GO_FLASH_PLE_CONV,FHC*FModel.HiddenSize,Conv,N,Layer.PleConv,Layer.PleState,
    [FHC*FModel.HiddenSize,FPleKernel,FPleNgram,Rows]);
  Work.Add(Input,Gated,Input.Count); Work.Add(Input,Conv,Input.Count);
end;

procedure TQwenFlashNext.Forward(const Ids: TArray<LongInt>; const GPUIds, Logits: TQNNGPUTensor; const Position: Integer);
begin
  if not FValid or (Position <> Length(FHistory)) or (Length(Ids)<1) or (Length(Ids)>128)
    or (Position+Length(Ids)>FCapacity) then raise EArgumentException.Create('Invalid qwen4exp recurrent prefix');
  FValid := False; CheckCancelled; FMoE.CancelCheck := FCancelCheck;
  var Rows := Length(Ids); var Work := TGPUWorkspace.Create(FCtx);
  try
    var Tokens := GPUIds; if Tokens=nil then Tokens := Work.Own(FCtx.UploadRaw(@Ids[0],Rows));
    var Embedding := Work.New(Int64(Rows)*FModel.HiddenSize);
    FCtx.Gather(Embedding,Tokens,FEmbedding,Default(TMemoryBlock),Rows,FModel.HiddenSize,FModel.VocabSize,0);
    var X := Work.New(Int64(Rows)*FHC*FModel.HiddenSize);
    FCtx.Kernel(GO_FLASH_HC_INIT,X.Count,X,Embedding,nil,nil,[FModel.HiddenSize,FHC]);
    var Ple := PleEmbedding(Work,Ids);
    for var I := 0 to High(FLayers) do
    begin
      CheckCancelled; var Layer: PLayer := @FLayers[I]; var LayerWork := TGPUWorkspace.Create(FCtx);
      try
        if I=FPleLayer then ApplyPle(LayerWork,Layer^,X,Ple,Rows);
        var Inject: TQNNGPUTensor;
        var N := Mix(LayerWork,X,Layer.AttentionMixer,Rows,Inject);
        var Output: TQNNGPUTensor;
        if Layer.Linear then Output := LinearAttention(LayerWork,Layer^,N,Rows)
        else Output := FullAttention(LayerWork,Layer^,N,Rows,Position);
        FCtx.Kernel(GO_FLASH_HC_COMBINE,X.Count,X,Output,Inject,nil,[FModel.HiddenSize,FHC]);
        N := Mix(LayerWork,X,Layer.FeedForwardMixer,Rows,Inject);
        Output := FMoE.Forward(FCtx,LayerWork,N,I,Rows);
        var SharedDim := Layer.SharedGate.Shape[0];
        var Gate := Project(LayerWork,N,Layer.SharedGate,Rows,FModel.HiddenSize,SharedDim);
        var Up := Project(LayerWork,N,Layer.SharedUp,Rows,FModel.HiddenSize,SharedDim);
        FCtx.Kernel(GO_SWIGLU,Gate.Count,Gate,Gate,Up,nil,[SharedDim,0]);
        var Shared := Project(LayerWork,Gate,Layer.SharedDown,Rows,SharedDim,FModel.HiddenSize);
        var Router := Project(LayerWork,N,Layer.SharedRouter,Rows,FModel.HiddenSize,1);
        FCtx.Kernel(GO_FLASH_SHARED_GATE,Shared.Count,Shared,Shared,Router,nil,[FModel.HiddenSize]);
        LayerWork.Add(Output,Shared,Output.Count);
        FCtx.Kernel(GO_FLASH_HC_COMBINE,X.Count,X,Output,Inject,nil,[FModel.HiddenSize,FHC]);
      finally LayerWork.Free end;
    end;
    var Last := Work.Copy(X,FHC*FModel.HiddenSize,(Rows-1)*FHC*FModel.HiddenSize); var Ignore: TQNNGPUTensor;
    var N := Mix(Work,Last,FHead,1,Ignore);
    var Output := Project(Work,N,FOutput,1,FModel.HiddenSize,FModel.VocabSize);
    FCtx.Copy(Logits,Output,FModel.VocabSize);
    FHistory := FHistory+Ids; FValid := True;
  finally Work.Free end;
end;

procedure TQwenFlashNext.Reset;
begin
  for var Layer in FLayers do
  begin
    if Layer.Linear then
    begin
      FCtx.Kernel(GO_FILL,Layer.State.Count,Layer.State,nil,nil,nil,[0]);
      FCtx.Kernel(GO_FILL,Layer.ConvState.Count,Layer.ConvState,nil,nil,nil,[0]);
    end;
    if Layer.PleState<>nil then FCtx.Kernel(GO_FILL,Layer.PleState.Count,Layer.PleState,nil,nil,nil,[0]);
  end;
  FHistory := nil; FValid := True;
end;

function TQwenFlashNext.StateIdentity: NativeUInt;
begin
  Result := 0; if Length(FLayers)>0 then Result := NativeUInt(FLayers[0].State);
end;

procedure TQwenFlashNext.Status(var Value: TLLMMemoryStatus);
begin
  FMoE.Status(Value); Value.IsHybrid := True; Value.IsGGUF := True;
  Value.DenseQuantization := wqNative; Value.ExpertQuantization := eqNative;
  Value.FixedGPUWeightBytes := FFixedBytes; Value.RecurrentBytes := FRecurrentBytes;
  Value.ActivationBytes := FCacheBytes;
  // Attention/shared/HC weights are resident in every layer; expert slices stream.
  Value.ResidentLayers := Length(FLayers); Value.StreamedLayers := 0; Value.CPULayers := 0;
end;

end.
