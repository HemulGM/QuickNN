unit QuickNN.Models.LLM;

{$POINTERMATH ON}

interface

uses
  System.SysUtils, QuickNN.Common, QuickNN.Vulkan, QuickNN.Models.Qwen3,
  QuickNN.Models.LLM.Types, QuickNN.Models.LLM.Text, QuickNN.Models.LLM.MoE,
  QuickNN.Models.QwenHybrid;

type
  // A model/session is bound to its creating thread, like the resident context.
  // Token IDs are host control data. Weights, hidden states, logits, KV and
  // sampling stay on the device. MoE additionally reads small router control data.
  TQNNLLM = class
  private
    FModel: TQwen3Model;
    FTokenizer: TQNNTokenizer;
    FOutputWeight: TMemoryBlock;
    FCtx: TQNNGPUContext;
    FCaches: TArray<TQNNGPUTensor>;
    FLogits, FSeen, FSeenInitial, FSelected: TQNNGPUTensor;
    FHistory: TArray<LongInt>;
    FStopIds: TArray<Integer>;
    FModelKey, FPromptSignature: string;
    FCheckpoint: TLLMContinuationState;
    FDecoder: TLLMUTF8Stream;
    FRandomState: UInt32;
    FPendingToken: Boolean;
    FLastToken, FAutomaticContinuations: Integer;
    FCapacity, FPosition, FValidVocabulary: Integer;
    FKVCacheBytes: UInt64;
    FEpsilon: Single;
    FGenerating, FPlainTemplate, FRestoreWeightLimit: Boolean;
    FMoE: TLLMMoE;
    FHybrid: TQwenHybrid;
    FPreviousWeightLimit, FFixedWeightBytes: UInt64;
    function GetMemoryStatus: TLLMMemoryStatus;
    procedure PrepareMoEWeights;
    procedure SetCancelCallback(const OnToken: TLLMTokenCallback);
    procedure ForwardRows(const Ids: TArray<LongInt>; const GPUIds: TQNNGPUTensor = nil);
    function Sample(const Options: TLLMGenerationOptions; var RandomState: UInt32): Integer;
    function IsStop(const Token: Integer): Boolean;
    procedure ValidateOptions(const Options: TLLMGenerationOptions);
    function GetKVCacheIdentity: NativeUInt;
    function GetContinuation: TLLMContinuationState;
    function ChatSignature(const Messages: TLLMMessages; const EnableThinking: Boolean): string;
    function SameContinuation(const State: TLLMContinuationState): Boolean;
    function Prefill(const Tokens: TArray<LongInt>; const OnToken: TLLMTokenCallback; var Response: TLLMGenerationResult): Boolean;
    procedure Decode(const Options: TLLMGenerationOptions; const OnToken: TLLMTokenCallback; const StartedAt: Int64; var Response: TLLMGenerationResult);
  public
    constructor Create(const ModelDirectory, TokenizerDirectory: string; const MaxContext: Integer = 2048); overload;
    constructor Create(const ModelDirectory, TokenizerDirectory: string; const MaxContext: Integer; const Memory: TLLMMemoryOptions); overload;
    destructor Destroy; override;
    procedure ResetContext;
    function Generate(const Messages: TLLMMessages; const Options: TLLMGenerationOptions; const OnToken: TLLMTokenCallback = nil
): TLLMGenerationResult;
    function ContinueGeneration(const State: TLLMContinuationState; const Options: TLLMGenerationOptions; const OnToken: TLLMTokenCallback = nil
): TLLMGenerationResult;
    // Host token planning only; these methods do not alter resident GPU state.
    function PrepareChat(const Messages: TLLMMessages; const EnableThinking: Boolean): TArray<LongInt>;
    function EncodeChat(const Messages: TLLMMessages; const EnableThinking: Boolean): TArray<LongInt>;
    function RebaseContinuation(const State: TLLMContinuationState; const Messages: TLLMMessages; const EnableThinking: Boolean): TLLMContinuationState;
    property Continuation: TLLMContinuationState read GetContinuation;
    property AutomaticContinuations: Integer read FAutomaticContinuations;
    // Allocation identity for diagnostics; no GPU memory readback.
    property KVCacheIdentity: NativeUInt read GetKVCacheIdentity;
    // Explicit diagnostic/scoring API: appends tokens and returns CPU logits.
    // Generate never uses this readback path.
    function EvaluateTokens(const Ids: TArray<LongInt>): TArray<Single>;
    property ContextLength: Integer read FPosition;
    property ContextCapacity: Integer read FCapacity;
    property MemoryStatus: TLLMMemoryStatus read GetMemoryStatus;
    property KVCacheBytes: UInt64 read FKVCacheBytes;
  end;

implementation

uses
  System.JSON, System.Classes, System.Hash, System.Diagnostics, System.IOUtils,
  System.Math, QuickNN.Vulkan.Operations, QuickNN.SafeTensors;

const
  PREFILL_ROWS = 128;
  MAX_KV_BYTES: UInt64 = UInt64(1536) * 1024 * 1024;

constructor TQNNLLM.Create(const ModelDirectory, TokenizerDirectory: string; const MaxContext: Integer);
begin
  Create(ModelDirectory, TokenizerDirectory, MaxContext, TLLMMemoryOptions.Defaults);
end;

constructor TQNNLLM.Create(const ModelDirectory, TokenizerDirectory: string; const MaxContext: Integer; const Memory: TLLMMemoryOptions);
var
  Config, Generation, HybridConfig: TJSONValue;
  Value, StopValue: TJSONValue;
  MaxPositions: Integer;
  ModelType, CacheDirectory: string;
  Tied: Boolean;
  Weight: PSafeTensor;
  SeenInitial: TArray<UInt32>;

  procedure AddStop(const Item: TJSONValue);
  var
    Id: Integer;
  begin
    if Item = nil then
      Exit;
    if Item is TJSONArray then
    begin
      for var Token in TJSONArray(Item) do
        AddStop(Token);
      Exit;
    end;
    Id := Item.AsType<Integer>;
    if (Id < 0) or (Id >= FModel.VocabSize) then
      raise EArgumentException.Create('LLM stop token is outside vocabulary');
    if not IsStop(Id) then
      FStopIds := FStopIds + [Id];
  end;

begin
  inherited Create;
  if SizeOf(Pointer) <> 8 then
    raise ENotSupportedException.Create('Resident LLM generation requires Win64');
  FModelKey := ExpandFileName(ModelDirectory) + #1 + ExpandFileName(TokenizerDirectory);
  FCtx := TQNNGPUContext.Create;
  if Memory.GPUWeightLimitBytes > 0 then
  begin
    FPreviousWeightLimit := QNNGetGPUStatus.WeightCacheLimit;
    QNNSetGPUWeightCacheLimit(Memory.GPUWeightLimitBytes);
    FRestoreWeightLimit := True;
  end;
  Config := TJSONValue.ParseJSONValue(TFile.ReadAllText(TPath.Combine(ModelDirectory, 'config.json')));
  if Config = nil then
    raise EArgumentException.Create('Invalid LLM config.json');
  try
    ModelType := Config.GetValue<string>('model_type', '');
    if ModelType = 'qwen3_5' then
    begin
      FCapacity := MaxContext;
      HybridConfig := Config.FindValue('text_config');
      if HybridConfig = nil then
        raise EArgumentException.Create('Missing hybrid text_config');
      if (Config.FindValue('quantization_config') <> nil)
        and not (Config.FindValue('quantization_config') is TJSONNull) then
        raise ENotSupportedException.Create('Use native BF16/F32 hybrid sources for QuickNN INT4 conversion');
      FHybrid := TQwenHybrid.Create(FCtx, FModel, ModelDirectory, Config, MaxContext, Memory);
      FKVCacheBytes := FHybrid.CacheBytes;
      FEpsilon := HybridConfig.GetValue<Single>('rms_norm_eps', 1E-6);
      AddStop(HybridConfig.FindValue('eos_token_id'));
    end
    else
    begin
      if (ModelType <> 'qwen3') and (ModelType <> 'qwen3_moe') then
        raise ENotSupportedException.Create('Only Qwen3 and Qwen3 MoE causal language models are supported');
      if (Memory.ExpertQuantization = eqInt4) and (ModelType <> 'qwen3_moe') then
        raise ENotSupportedException.Create('INT4 expert storage requires a MoE model');
      FPlainTemplate := ModelType = 'qwen3_moe';
      if Config.GetValue<Boolean>('attention_bias', False)
        or Config.GetValue<Boolean>('use_sliding_window', False)
        or (Config.GetValue<string>('hidden_act', 'silu') <> 'silu') then
        raise ENotSupportedException.Create('Unsupported Qwen3 attention/activation configuration');
      for var Name in TArray<string>.Create('rope_scaling', 'quantization_config') do
      begin
        Value := Config.FindValue(Name);
        if (Value <> nil) and not (Value is TJSONNull) then
          raise ENotSupportedException.Create('Unsupported LLM configuration: ' + Name);
      end;
      Value := Config.FindValue('layer_types');
      if Value is TJSONArray then
        for var Kind in TJSONArray(Value) do
          if Kind.Value <> 'full_attention' then
            raise ENotSupportedException.Create('Only full Qwen3 attention layers are supported');
      MaxPositions := Config.GetValue<Integer>('max_position_embeddings', 32768);
      if (MaxContext < 2) or (MaxContext > MaxPositions) then
        raise EArgumentOutOfRangeException.Create('LLM context exceeds model position limit');
      FCapacity := MaxContext;
      FEpsilon := Config.GetValue<Single>('rms_norm_eps', 1E-6);
      if IsNan(FEpsilon) or IsInfinite(FEpsilon) or (FEpsilon <= 0) then
        raise EArgumentException.Create('Invalid LLM RMS epsilon');
      Tied := Config.GetValue<Boolean>('tie_word_embeddings', False);
      FModel.Load(ModelDirectory, True, True);
      if (Length(FModel.Layers) < 1)
        or (FModel.HiddenSize < 1)
        or (FModel.IntermediateSize < 1)
        or (FModel.VocabSize < 1)
        or (FModel.NumHeads < 1)
        or (FModel.NumKvHeads < 1)
        or (FModel.NumHeads mod FModel.NumKvHeads <> 0)
        or (FModel.HeadDim < 2)
        or (FModel.HeadDim mod 2 <> 0)
        or (FModel.HeadDim > 1152) then
        raise EArgumentException.Create('Invalid Qwen3 model geometry');
      FKVCacheBytes :=
        UInt64(Length(FModel.Layers)) * 2 * UInt64(FCapacity) * UInt64(FModel.NumKvHeads) * UInt64(FModel.HeadDim) * 4;
      if FKVCacheBytes > MAX_KV_BYTES then
        raise EArgumentOutOfRangeException.Create('KV cache exceeds resident budget; reduce MaxContext');
      Weight := FModel.SfFiles.GetTensor('lm_head.weight');
      if Weight <> nil then
        FOutputWeight := FModel.SfFiles.GetLinearWeight('lm_head.weight', True)
      else if Tied then
        FOutputWeight := FModel.EmbedTokens
      else
        raise EArgumentException.Create('Missing untied lm_head.weight');
      if (FOutputWeight.Count <> Int64(FModel.VocabSize) * FModel.HiddenSize)
        or not (FOutputWeight.DataType in [dtF32, dtBF16]) then
        raise EArgumentException.Create('Invalid LLM output projection');
      CacheDirectory := Memory.ExpertCacheDirectory;
      if CacheDirectory = '' then
        CacheDirectory := TPath.Combine(ModelDirectory, '.quicknn-cache');
      if ModelType = 'qwen3_moe' then
        FMoE :=
          TLLMMoE.Create(
          FModel.SfFiles,
          Config,
          Length(FModel.Layers),
          FModel.HiddenSize,
          Memory.RAMCacheLimitBytes,
          Memory.ExpertBatchBytes,
          Memory.ExpertQuantization,
          CacheDirectory,
          Memory.OnExpertCacheProgress
        );
      StopValue := Config.FindValue('eos_token_id');
      AddStop(StopValue);
    end;
    if TFile.Exists(TPath.Combine(ModelDirectory, 'generation_config.json')) then
    begin
      Generation :=
        TJSONValue.ParseJSONValue(TFile.ReadAllText(TPath.Combine(ModelDirectory, 'generation_config.json')));
      try
        if Generation = nil then
          raise EArgumentException.Create('Invalid generation_config.json');
        AddStop(Generation.FindValue('eos_token_id'));
      finally
        Generation.Free;
      end;
    end;
  finally
    Config.Free;
  end;
  if Length(FStopIds) = 0 then
    raise EArgumentException.Create('Missing LLM stop tokens');
  if TFile.Exists(TPath.Combine(TokenizerDirectory, 'tokenizer_config.json')) then
  begin
    Config := TJSONValue.ParseJSONValue(TFile.ReadAllText(TPath.Combine(TokenizerDirectory, 'tokenizer_config.json')));
    try
      if Config = nil then
        raise EArgumentException.Create('Invalid tokenizer_config.json');
      var Template := Config.GetValue<string>('chat_template', '');
      if Template <> '' then
        FPlainTemplate := Pos('<think>', Template) = 0;
    finally
      Config.Free;
    end;
  end;
  FTokenizer := TQNNTokenizer.Load(TPath.Combine(TokenizerDirectory, 'tokenizer.json'));
  if FMoE <> nil then
    PrepareMoEWeights;
  if FHybrid = nil then
  begin
    SetLength(FCaches, Length(FModel.Layers));
    for var I := 0 to High(FCaches) do
      FCaches[I] := FCtx.NewTensor(2 * FCapacity * FModel.NumKvHeads * FModel.HeadDim);
  end;
  FLogits := FCtx.NewTensor(FModel.VocabSize);
  FSeen := FCtx.NewTensor(FModel.VocabSize);
  FSelected := FCtx.NewTensor(1);
  SetLength(SeenInitial, FModel.VocabSize);
  for var Id := 0 to High(SeenInitial) do
    if (Id < Length(FTokenizer.Vocabs)) and (FTokenizer.Vocabs[Id] <> '') then
    begin
      SeenInitial[Id] := 0;
      Inc(FValidVocabulary);
    end
    else
      SeenInitial[Id] := High(UInt32);
  if FValidVocabulary = 0 then
    raise EArgumentException.Create('Empty LLM vocabulary');
  FSeenInitial := FCtx.UploadRaw(@SeenInitial[0], Length(SeenInitial));
  ResetContext;
end;

destructor TQNNLLM.Destroy;
begin
  FCtx.Free;
  FMoE.Free;
  FHybrid.Free;
  FOutputWeight.Free;
  if Length(FModel.SfFiles) <> 0 then
    FModel.Free;
  FTokenizer.Free;
  if FRestoreWeightLimit then
    QNNSetGPUWeightCacheLimit(FPreviousWeightLimit);
  inherited;
end;

function TQNNLLM.GetMemoryStatus: TLLMMemoryStatus;
var
  GPUStatus: TQNNGPUStatus;
begin
  Result := Default(TLLMMemoryStatus);
  GPUStatus := QNNGetGPUStatus;
  Result.GPUWeightBytes := GPUStatus.WeightCacheBytes;
  Result.GPUWeightLimitBytes := GPUStatus.WeightCacheLimit;
  Result.FixedGPUWeightBytes := FFixedWeightBytes;
  for var FileInfo in FModel.SfFiles do
    Inc(Result.MappedWeightBytes, UInt64(FileInfo.DataSize));
  if FMoE <> nil then
    FMoE.Status(Result);
  if FHybrid <> nil then
    FHybrid.Status(Result);
end;

procedure TQNNLLM.PrepareMoEWeights;
var
  Retain: Boolean;

  procedure Weight(const Value: TMemoryBlock; const Rows, Columns: Integer);
  begin
    if (Value.Size <> Int64(Rows) * Columns) or not (Value.DataType in [dtF32, dtBF16]) then
      raise EArgumentException.Create('Invalid fixed MoE weight: ' + Value.Name);
    if Retain then
      FCtx.RetainWeight(Value, Rows, Columns)
    else
      Inc(FFixedWeightBytes, UInt64(Value.Size) * UInt64(DATATYPE_BITS[Value.DataType] div 8));
  end;

  procedure Visit;
  var
    Layer: PQWEN3Layer;
  begin
    Weight(FModel.EmbedTokens, FModel.VocabSize, FModel.HiddenSize);
    if FOutputWeight.WeightIdentity <> FModel.EmbedTokens.WeightIdentity then
      Weight(FOutputWeight, FModel.VocabSize, FModel.HiddenSize);
    Weight(FModel.NormWeight, 1, FModel.HiddenSize);
    for var I := 0 to High(FModel.Layers) do
    begin
      Layer := @FModel.Layers[I];
      Layer.Model := @FModel;
      if not Layer.InputLayernormWeight.IsAssigned then
        Layer.Load(I, True, FMoE.IsSparse(I));
      Weight(Layer.InputLayernormWeight, 1, FModel.HiddenSize);
      Weight(Layer.PostAttentionLayernormWeight, 1, FModel.HiddenSize);
      Weight(Layer.Attn.QProjWeight, FModel.NumHeads * FModel.HeadDim, FModel.HiddenSize);
      Weight(Layer.Attn.KProjWeight, FModel.NumKvHeads * FModel.HeadDim, FModel.HiddenSize);
      Weight(Layer.Attn.VProjWeight, FModel.NumKvHeads * FModel.HeadDim, FModel.HiddenSize);
      Weight(Layer.Attn.OProjWeight, FModel.HiddenSize, FModel.NumHeads * FModel.HeadDim);
      Weight(Layer.Attn.QNormWeight, 1, FModel.HeadDim);
      Weight(Layer.Attn.KNormWeight, 1, FModel.HeadDim);
      if not FMoE.IsSparse(I) then
      begin
        Weight(Layer.Mlp.GateProjWeight, FModel.IntermediateSize, FModel.HiddenSize);
        Weight(Layer.Mlp.UpProjWeight, FModel.IntermediateSize, FModel.HiddenSize);
        Weight(Layer.Mlp.DownProjWeight, FModel.HiddenSize, FModel.IntermediateSize);
      end;
    end;
  end;

begin
  Retain := False;
  Visit;
  Inc(FFixedWeightBytes, FMoE.RouterBytes);
  if FFixedWeightBytes + FMoE.ExpertBytes > QNNGetGPUStatus.WeightCacheLimit then
    raise ERangeError.Create('GPU weight budget cannot hold fixed MoE weights and one expert');
  Retain := True;
  Visit;
  FMoE.RetainRouters(FCtx);
end;

procedure TQNNLLM.SetCancelCallback(const OnToken: TLLMTokenCallback);
var
  Probe: TFunc<Boolean>;
begin
  Probe := nil;
  if Assigned(OnToken) then
    Probe :=
      function: Boolean
      var
        Cancel: Boolean;
      begin
        Cancel := False;
        OnToken('', Cancel);
        Result := Cancel;
      end;
  if FMoE <> nil then
    FMoE.CancelCheck := Probe;
  if FHybrid <> nil then
    FHybrid.CancelCheck := Probe;
end;

procedure TQNNLLM.ResetContext;
begin
  if FGenerating then
    raise EInvalidOp.Create('Cannot reset an active LLM generation');
  FCtx.Copy(FSeen, FSeenInitial, FModel.VocabSize);
  if FHybrid <> nil then
    FHybrid.Reset;
  FPosition := 0;
  FHistory := nil;
  FCheckpoint := Default(TLLMContinuationState);
  FPendingToken := False;
  FDecoder.Reset;
end;

function TQNNLLM.IsStop(const Token: Integer): Boolean;
begin
  for var Id in FStopIds do
    if Id = Token then
      Exit(True);
  Result := False;
end;

procedure TQNNLLM.ForwardRows(const Ids: TArray<LongInt>; const GPUIds: TQNNGPUTensor);
var
  Work, LayerWork: TGPUWorkspace;
  X, TokenIds, N, Q, K, V, A, Y, G, U, Last: TQNNGPUTensor;
  Layer: ^TQwen3Layer;
  Rows, QDim, KvDim: Integer;

  function Project(const Src: TQNNGPUTensor; const Weight: TMemoryBlock; const InputDim, OutputDim: Integer): TQNNGPUTensor;
  begin
    if Rows = 1 then
    begin
      Result := LayerWork.New(OutputDim);
      FCtx.VectorLinear(Result, Src, Weight, InputDim, OutputDim);
    end
    else
      Result := LayerWork.Linear(Src, Weight, Rows, InputDim, OutputDim);
  end;

begin
  Rows := Length(Ids);
  if (Rows < 1) or (Int64(FPosition) + Rows > FCapacity) then
    raise EArgumentOutOfRangeException.Create('LLM KV context capacity exceeded');
  for var Id in Ids do
    if (Id < 0) or (Id >= FModel.VocabSize) then
      raise EArgumentOutOfRangeException.Create('LLM token outside vocabulary');
  if FHybrid <> nil then
  begin
    FHybrid.forward(Ids, GPUIds, FLogits, FPosition);
    SetLength(FHistory, FPosition + Rows);
    Move(Ids[0], FHistory[FPosition], Rows * SizeOf(LongInt));
    Inc(FPosition, Rows);
    Exit;
  end;
  Work := TGPUWorkspace.Create(FCtx);
  try
    if GPUIds <> nil then
      TokenIds := GPUIds
    else
      TokenIds := Work.Own(FCtx.UploadRaw(@Ids[0], Rows));
    X := Work.New(Int64(Rows) * FModel.HiddenSize);
    FCtx.Gather(X, TokenIds, FModel.EmbedTokens, Default(TMemoryBlock), Rows, FModel.HiddenSize, FModel.VocabSize, 0);
    QDim := FModel.NumHeads * FModel.HeadDim;
    KvDim := FModel.NumKvHeads * FModel.HeadDim;
    for var I := 0 to High(FModel.Layers) do
    begin
      Layer := @FModel.Layers[I];
      Layer.Model := @FModel;
      if not Layer.InputLayernormWeight.IsAssigned then
        Layer.Load(I, True, (FMoE <> nil) and FMoE.IsSparse(I));
      LayerWork := TGPUWorkspace.Create(FCtx);
      try
        N :=
          LayerWork
          .Norm(X, Layer.InputLayernormWeight, Default(TMemoryBlock), Rows, FModel.HiddenSize, False, FEpsilon);
        Q := Project(N, Layer.Attn.QProjWeight, FModel.HiddenSize, QDim);
        K := Project(N, Layer.Attn.KProjWeight, FModel.HiddenSize, KvDim);
        V := Project(N, Layer.Attn.VProjWeight, FModel.HiddenSize, KvDim);
        Q :=
          LayerWork.Norm(
          Q,
          Layer.Attn.QNormWeight,
          Default(TMemoryBlock),
          Rows * FModel.NumHeads,
          FModel.HeadDim,
          False,
          FEpsilon
        );
        K :=
          LayerWork.Norm(
          K,
          Layer.Attn.KNormWeight,
          Default(TMemoryBlock),
          Rows * FModel.NumKvHeads,
          FModel.HeadDim,
          False,
          FEpsilon
        );
        FCtx.Kernel(
          GO_LLM_ROPE,
          Rows * QDim div 2,
          Q,
          Q,
          nil,
          nil,
          [FModel.NumHeads, FModel.HeadDim, FPosition, QNNGPUFloatBits(FModel.RoPETheta)]
        );
        FCtx.Kernel(
          GO_LLM_ROPE,
          Rows * KvDim div 2,
          K,
          K,
          nil,
          nil,
          [FModel.NumKvHeads, FModel.HeadDim, FPosition, QNNGPUFloatBits(FModel.RoPETheta)]
        );
        FCtx.Kernel(
          GO_LLM_CACHE_WRITE,
          Rows * KvDim,
          FCaches[I],
          K,
          V,
          nil,
          [Rows, FModel.NumKvHeads, FModel.HeadDim, FCapacity, FPosition]
        );
        A := LayerWork.New(Int64(Rows) * QDim);
        FCtx.CachedAttention(
          A,
          Q,
          FCaches[I],
          Rows,
          FCapacity,
          FModel.NumHeads,
          FModel.NumKvHeads,
          FModel.HeadDim,
          FPosition
        );
        Y := Project(A, Layer.Attn.OProjWeight, QDim, FModel.HiddenSize);
        LayerWork.Add(X, Y, Rows * FModel.HiddenSize);
        N :=
          LayerWork.Norm(
          X,
          Layer.PostAttentionLayernormWeight,
          Default(TMemoryBlock),
          Rows,
          FModel.HiddenSize,
          False,
          FEpsilon
        );
        if (FMoE <> nil) and FMoE.IsSparse(I) then
          Y := FMoE.forward(FCtx, LayerWork, N, I, Rows)
        else
        begin
          G := Project(N, Layer.Mlp.GateProjWeight, FModel.HiddenSize, FModel.IntermediateSize);
          U := Project(N, Layer.Mlp.UpProjWeight, FModel.HiddenSize, FModel.IntermediateSize);
          FCtx.Kernel(GO_SWIGLU, Rows * FModel.IntermediateSize, G, G, U, nil, [FModel.IntermediateSize, 0]);
          Y := Project(G, Layer.Mlp.DownProjWeight, FModel.IntermediateSize, FModel.HiddenSize);
        end;
        LayerWork.Add(X, Y, Rows * FModel.HiddenSize);
      finally
        LayerWork.Free;
      end;
    end;
    Last := Work.Copy(X, FModel.HiddenSize, (Rows - 1) * FModel.HiddenSize);
    N := Work.Norm(Last, FModel.NormWeight, Default(TMemoryBlock), 1, FModel.HiddenSize, False, FEpsilon);
    FCtx.VectorLinear(FLogits, N, FOutputWeight, FModel.HiddenSize, FModel.VocabSize);
    FCtx.CheckFinite(FLogits);
    SetLength(FHistory, FPosition + Rows);
    Move(Ids[0], FHistory[FPosition], Rows * SizeOf(LongInt));
    Inc(FPosition, Rows);
  finally
    Work.Free;
  end;
end;

function TQNNLLM.Sample(const Options: TLLMGenerationOptions; var RandomState: UInt32): Integer;
var
  Uniform: Single;
begin
  RandomState := RandomState xor (RandomState shl 13);
  RandomState := RandomState xor (RandomState shr 17);
  RandomState := RandomState xor (RandomState shl 5);
  Uniform := Min(0.99999994, (Double(RandomState) + 0.5) / 4294967296.0);
  FCtx.SampleToken(
    FSelected,
    FLogits,
    FSeen,
    FModel.VocabSize,
    Min(Options.TopK, FValidVocabulary),
    Options.Temperature,
    Options.TopP,
    Options.RepetitionPenalty,
    Uniform
  );
  FCtx.Download(FSelected, @Result, 1);
  if (Result < 0) or (Result >= FModel.VocabSize) then
    raise EInvalidOp.Create('Invalid GPU sampled token');
  FCtx.Kernel(GO_LLM_SEEN, 1, FSeen, FSelected, nil, nil, [FModel.VocabSize]);
end;

procedure TQNNLLM.ValidateOptions(const Options: TLLMGenerationOptions);
begin
  if FPlainTemplate and Options.EnableThinking then
    raise ENotSupportedException.Create('This chat template does not support thinking mode');
  if (Options.MaxNewTokens < 1)
    or (Options.HardMaxNewTokens < 0)
    or (Options.AutoContinue and (Options.HardMaxNewTokens < Options.MaxNewTokens))
    or (Options.TopK < 1)
    or (Options.TopK > 64)
    or IsNan(Options.Temperature)
    or IsInfinite(Options.Temperature)
    or (Options.Temperature < 0)
    or IsNan(Options.TopP)
    or IsInfinite(Options.TopP)
    or (Options.TopP <= 0)
    or (Options.TopP > 1)
    or IsNan(Options.RepetitionPenalty)
    or IsInfinite(Options.RepetitionPenalty)
    or (Options.RepetitionPenalty <= 0) then
    raise EArgumentException.Create('Invalid LLM generation options (TopK must be 1..64)');
end;

function TQNNLLM.GetKVCacheIdentity: NativeUInt;
begin
  if FHybrid <> nil then
    Result := FHybrid.StateIdentity
  else if Length(FCaches) = 0 then
    Result := 0
  else
    Result := NativeUInt(FCaches[0]);
end;

function TQNNLLM.GetContinuation: TLLMContinuationState;
begin
  Result := FCheckpoint;
  // Published records own their IDs; a caller cannot alter the resident checkpoint.
  Result.Tokens := Copy(FCheckpoint.Tokens);
end;

function TQNNLLM.SameContinuation(const State: TLLMContinuationState): Boolean;
begin
  Result :=
    FCheckpoint.Available
    and (State.ModelKey = FCheckpoint.ModelKey)
    and (State.VisionSignature = FCheckpoint.VisionSignature)
    and (State.Text = FCheckpoint.Text)
    and (State.RandomState = FCheckpoint.RandomState)
    and (State.PendingUTF8 = FCheckpoint.PendingUTF8)
    and (State.PromptTokens = FCheckpoint.PromptTokens)
    and (State.GeneratedTokens = FCheckpoint.GeneratedTokens)
    and (Length(State.Tokens) = Length(FCheckpoint.Tokens));
  if (FHybrid <> nil) and not FHybrid.StateValid then
    Result := False;
  if Result then
    Result := CompareMem(@State.Tokens[0], @FCheckpoint.Tokens[0], Length(State.Tokens) * SizeOf(LongInt));
end;

function TQNNLLM.Prefill(const Tokens: TArray<LongInt>; const OnToken: TLLMTokenCallback; var Response: TLLMGenerationResult): Boolean;
var
  Common, Offset, Count: Integer;
  Work: TGPUWorkspace;
  Ids: TQNNGPUTensor;
  Cancel: Boolean;
begin
  Cancel := False;
  if Assigned(OnToken) then
    OnToken('', Cancel);
  if Cancel then
    Exit(False);
  Common := 0;
  while (Common < Length(Tokens)) and (Common < Length(FHistory)) and (Tokens[Common] = FHistory[Common]) do
    Inc(Common);
  // Recompute the last row if logits may belong to a later token.
  if Common = Length(Tokens) then
    Dec(Common);
  // Recurrent state describes the complete processed prefix. It cannot be
  // rewound by truncating the token/KV indices; rebuild on a changed prefix.
  if (FHybrid <> nil) and ((Common <> FPosition) or not FHybrid.StateValid) then
  begin
    Common := 0;
    FHybrid.Reset;
  end;
  Response.ReusedTokens := Common;
  FPosition := Common;
  SetLength(FHistory, Common);
  FCtx.Copy(FSeen, FSeenInitial, FModel.VocabSize);
  Work := TGPUWorkspace.Create(FCtx);
  try
    Ids := Work.Own(FCtx.UploadRaw(@Tokens[0], Length(Tokens)));
    FCtx.Kernel(GO_LLM_SEEN, Length(Tokens), FSeen, Ids, nil, nil, [FModel.VocabSize]);
  finally
    Work.Free
  end;
  Offset := Common;
  while Offset < Length(Tokens) do
  begin
    if Assigned(OnToken) then
      OnToken('', Cancel);
    if Cancel then
      Exit(False);
    Count := Min(PREFILL_ROWS, Length(Tokens) - Offset);
    try
      ForwardRows(Copy(Tokens, Offset, Count))
    except
      on ELLMMoECancelled do
        Exit(False)
    end;
    Inc(Response.PrefillTokens, Count);
    Inc(Offset, Count);
  end;
  Result := True;
end;

procedure TQNNLLM.Decode(const Options: TLLMGenerationOptions; const OnToken: TLLMTokenCallback; const StartedAt: Int64; var Response: TLLMGenerationResult);
var
  Budget, Available, Limit, InitialGenerated, Token: Integer;
  FirstAt, LastAt, DecodeStartedAt: Int64;
  PriorTotal, PriorDecode: Double;
  Cancel: Boolean;

  procedure Emit(const Value: string);
  begin
    if Value = '' then
      Exit;
    Response.Text := Response.Text + Value;
    if Assigned(OnToken) then
      OnToken(Value, Cancel);
  end;

begin
  FAutomaticContinuations := 0;
  PriorTotal := Response.TotalSeconds;
  PriorDecode := Response.DecodeSeconds;
  InitialGenerated := Response.GeneratedTokens;
  FirstAt := 0;
  LastAt := 0;
  Cancel := False;
  if (FHybrid <> nil) and Options.EnableThinking and (InitialGenerated = 0) then
    Emit('<think>'#10);
  DecodeStartedAt := TStopwatch.GetTimeStamp;
  Budget := Options.MaxNewTokens;
  if Options.AutoContinue then
    Budget := Options.HardMaxNewTokens;
  Available := FCapacity - FPosition - Ord(FPendingToken);
  Limit := Min(Budget, Available);
  Response.FinishReason := lfMaxTokens;
  for var Index := 0 to Limit - 1 do
  begin
    if Assigned(OnToken) then
      OnToken('', Cancel);
    if Cancel then
      Break;
    // Feed the previously emitted token once, using its existing device ID.
    // Soft boundaries neither submit another prompt nor rebuild sampling/KV state.
    if FPendingToken then
    begin
      try
        ForwardRows([FLastToken], FSelected)
      except
        on ELLMMoECancelled do
        begin
          Cancel := True;
          Break
        end
      end;
      FPendingToken := False;
    end;
    Token := Sample(Options, FRandomState);
    if IsStop(Token) then
    begin
      Response.FinishReason := lfEOS;
      Break
    end;
    if Options.AutoContinue and (Index > 0) and (Index mod Options.MaxNewTokens = 0) then
      Inc(FAutomaticContinuations);
    LastAt := TStopwatch.GetTimeStamp;
    if FirstAt = 0 then
      FirstAt := LastAt;
    Inc(Response.GeneratedTokens);
    FLastToken := Token;
    FPendingToken := True;
    Emit(FDecoder.Push(LLMTokenBytes(FTokenizer, Token)));
    if Cancel then
      Break;
  end;
  if Cancel then
    Response.FinishReason := lfCancelled
  else if Response.FinishReason <> lfEOS then
  begin
    if Available <= Budget then
      Response.FinishReason := lfContextFull
    else if Options.AutoContinue then
      Response.FinishReason := lfHardLimit;
  end;
  // An incomplete UTF-8 character must survive a limit/cancellation boundary.
  if Response.FinishReason = lfEOS then
    Emit(FDecoder.Finish);
  if Cancel then
    Response.FinishReason := lfCancelled;
  FCtx.Flush;
  Response.AutomaticContinuations := FAutomaticContinuations;
  Response.TotalSeconds := PriorTotal + (TStopwatch.GetTimeStamp - StartedAt) / TStopwatch.Frequency;
  if (Response.TimeToFirstTokenSeconds = 0) and (FirstAt <> 0) then
    Response.TimeToFirstTokenSeconds := (FirstAt - StartedAt) / TStopwatch.Frequency;
  if (InitialGenerated > 0) and (Response.GeneratedTokens > InitialGenerated) then
    Response.DecodeSeconds := PriorDecode + (LastAt - DecodeStartedAt) / TStopwatch.Frequency
  else if Response.GeneratedTokens - InitialGenerated > 1 then
    Response.DecodeSeconds := PriorDecode + (LastAt - FirstAt) / TStopwatch.Frequency;
  FCheckpoint := Default(TLLMContinuationState);
  if (Response.FinishReason <> lfEOS) and (Response.GeneratedTokens > 0) then
  begin
    FCheckpoint.ModelKey := FModelKey;
    FCheckpoint.PromptSignature := FPromptSignature;
    if FHybrid <> nil then
      FCheckpoint.VisionSignature := FHybrid.VisionSignature;
    FCheckpoint.Text := Response.Text;
    FCheckpoint.Tokens := Copy(FHistory);
    if FPendingToken then
      FCheckpoint.Tokens := FCheckpoint.Tokens + [FLastToken];
    FCheckpoint.PromptTokens := Response.PromptTokens;
    FCheckpoint.GeneratedTokens := Response.GeneratedTokens;
    FCheckpoint.RandomState := FRandomState;
    FCheckpoint.PendingUTF8 := FDecoder.PendingBytes;
    FCheckpoint.DecodeSeconds := Response.DecodeSeconds;
    FCheckpoint.TotalSeconds := Response.TotalSeconds;
    FCheckpoint.TimeToFirstTokenSeconds := Response.TimeToFirstTokenSeconds;
  end;
end;

function TQNNLLM.ChatSignature(const Messages: TLLMMessages; const EnableThinking: Boolean): string;
var
  Builder: TStringBuilder;
begin
  Builder := TStringBuilder.Create;
  try
    Builder.Append(Ord(EnableThinking)).Append('/').Append(Ord(FPlainTemplate)).Append(':');
    for var Message in Messages do
    begin
      Builder.Append(Ord(Message.Role)).Append(':').Append(Length(Message.Content)).Append(':').Append(Message.Content);
      for var Image in Message.Images do
        Builder.Append('/image:').Append(Image.Identity).Append('/').Append(Image.EncoderKey);
    end;
    Result := THashSHA2.GetHashString(Builder.ToString);
  finally
    Builder.Free
  end;
end;

function TQNNLLM.EncodeChat(const Messages: TLLMMessages; const EnableThinking: Boolean): TArray<LongInt>;
begin
  Result := LLMChatTokens(FTokenizer, Messages, EnableThinking, FPlainTemplate, FHybrid <> nil);
end;

function TQNNLLM.PrepareChat(const Messages: TLLMMessages; const EnableThinking: Boolean): TArray<LongInt>;
var
  Images: TArray<TLLMImage>;
  Signature: string;
begin
  Result := EncodeChat(Messages, EnableThinking);
  for var Message in Messages do
    for var Image in Message.Images do
    begin
      Images := Images + [Image];
      Signature := Signature + Image.Identity + '/' + Image.EncoderKey + ';';
    end;
  if Length(Images) > 0 then
  begin
    var Last := High(Result);
    while (Last >= 0) and (Result[Last] <> 248056) do
      Dec(Last);
    for var I := 0 to Last do
      Signature := Signature + '/' + Result[I].ToString;
  end;
  if FHybrid <> nil then
    FHybrid.SetVision(Result, Images, Signature);
end;

function TQNNLLM.RebaseContinuation(const State: TLLMContinuationState; const Messages: TLLMMessages; const EnableThinking: Boolean): TLLMContinuationState;
var
  Prompt: TArray<LongInt>;
begin
  if not State.Available or (State.ModelKey <> FModelKey) then
    raise EArgumentException.Create('Cannot rebase a continuation from another model');
  Prompt := PrepareChat(Messages, EnableThinking);
  var Signature := ChatSignature(Messages, EnableThinking);
  if (State.PromptSignature <> '') and (State.PromptSignature = Signature) then
  begin
    Result := State;
    Result.Tokens := Copy(State.Tokens);
    Exit;
  end;
  Result := State;
  Result.PromptSignature := Signature;
  Result.VisionSignature := '';
  if FHybrid <> nil then
    Result.VisionSignature := FHybrid.VisionSignature;
  Result.PromptTokens := Length(Prompt);
  // Preserve every generated ID, including the final token not yet forwarded.
  Result.Tokens := Prompt + Copy(State.Tokens, State.PromptTokens, State.GeneratedTokens);
end;

function TQNNLLM.Generate(const Messages: TLLMMessages; const Options: TLLMGenerationOptions; const OnToken: TLLMTokenCallback): TLLMGenerationResult;
var
  Prompt: TArray<LongInt>;
  StartedAt: Int64;
begin
  StartedAt := TStopwatch.GetTimeStamp;
  if FGenerating then
    raise EInvalidOp.Create('LLM generation is already active');
  ValidateOptions(Options);
  Prompt := PrepareChat(Messages, Options.EnableThinking);
  if (Length(Prompt) < 1) or (Length(Prompt) >= FCapacity) then
    raise EArgumentOutOfRangeException.Create('Chat prompt fills context; reset or shorten the history');
  for var Id in Prompt do
    if (Id < 0) or (Id >= FModel.VocabSize) then
      raise EArgumentException.Create('Tokenizer does not match the LLM vocabulary');
  Result := Default(TLLMGenerationResult);
  Result.PromptTokens := Length(Prompt);
  Result.FinishReason := lfCancelled;
  FPromptSignature := ChatSignature(Messages, Options.EnableThinking);
  FCheckpoint := Default(TLLMContinuationState);
  FPendingToken := False;
  FDecoder.Reset;
  FRandomState := UInt32(Options.Seed and $FFFFFFFF) xor UInt32(Options.Seed shr 32);
  if FRandomState = 0 then
    FRandomState := $9E3779B9;
  FAutomaticContinuations := 0;
  FGenerating := True;
  try
    SetCancelCallback(OnToken);
    try
      if Prefill(Prompt, OnToken, Result) then
        Decode(Options, OnToken, StartedAt, Result)
      else
        Result.TotalSeconds := (TStopwatch.GetTimeStamp - StartedAt) / TStopwatch.Frequency;
    except
      if FHybrid <> nil then
        FHybrid.Reset;
      FPosition := 0;
      FHistory := nil;
      FCheckpoint := Default(TLLMContinuationState);
      FPendingToken := False;
      raise;
    end;
  finally
    SetCancelCallback(nil);
    FGenerating := False;
  end;
end;

function TQNNLLM.ContinueGeneration(const State: TLLMContinuationState; const Options: TLLMGenerationOptions; const OnToken: TLLMTokenCallback): TLLMGenerationResult;
var
  StartedAt: Int64;
begin
  StartedAt := TStopwatch.GetTimeStamp;
  if FGenerating then
    raise EInvalidOp.Create('LLM generation is already active');
  ValidateOptions(Options);
  if not State.Available or (State.ModelKey <> FModelKey) or (Length(State.PendingUTF8) > 3) then
    raise EArgumentException.Create('Continuation does not match this model/tokenizer');
  if (State.VisionSignature <> '') and ((FHybrid = nil) or (State.VisionSignature <> FHybrid.VisionSignature)) then
    raise EArgumentException.Create('Restore image attachments with RebaseContinuation before replaying this visual checkpoint');
  if Length(State.Tokens) >= FCapacity then
    raise EArgumentOutOfRangeException.Create('Continuation fills context; increase the chat context');
  for var Id in State.Tokens do
    if (Id < 0) or (Id >= FModel.VocabSize) then
      raise EArgumentException.Create('Invalid continuation token');
  Result := Default(TLLMGenerationResult);
  Result.Text := State.Text;
  Result.PromptTokens := State.PromptTokens;
  Result.GeneratedTokens := State.GeneratedTokens;
  Result.DecodeSeconds := State.DecodeSeconds;
  Result.TotalSeconds := State.TotalSeconds;
  Result.TimeToFirstTokenSeconds := State.TimeToFirstTokenSeconds;
  Result.FinishReason := lfCancelled;
  Result.ResumedFromCache := SameContinuation(State);
  FAutomaticContinuations := 0;
  FGenerating := True;
  try
    SetCancelCallback(OnToken);
    try
      if Result.ResumedFromCache then
        Result.ReusedTokens := FPosition
      else
      begin
        FCheckpoint := Default(TLLMContinuationState);
        FPendingToken := False;
        if not Prefill(State.Tokens, OnToken, Result) then
          Exit;
      end;
      FPromptSignature := State.PromptSignature;
      FDecoder.PendingBytes := State.PendingUTF8;
      FRandomState := State.RandomState;
      Decode(Options, OnToken, StartedAt, Result);
    except
      if FHybrid <> nil then
        FHybrid.Reset;
      FPosition := 0;
      FHistory := nil;
      FCheckpoint := Default(TLLMContinuationState);
      FPendingToken := False;
      raise;
    end;
  finally
    SetCancelCallback(nil);
    FGenerating := False;
  end;
end;

function TQNNLLM.EvaluateTokens(const Ids: TArray<LongInt>): TArray<Single>;
var
  Offset, Count: Integer;
begin
  if FGenerating then
    raise EInvalidOp.Create('Cannot evaluate during generation');
  FCheckpoint := Default(TLLMContinuationState);
  FPendingToken := False;
  Offset := 0;
  if Length(Ids) = 0 then
    raise EArgumentException.Create('Empty LLM evaluation');
  if Int64(FPosition) + Length(Ids) > FCapacity then
    raise EArgumentOutOfRangeException.Create('LLM evaluation exceeds context');
  for var Id in Ids do
    if (Id < 0) or (Id >= FModel.VocabSize) then
      raise EArgumentOutOfRangeException.Create('LLM token outside vocabulary');
  while Offset < Length(Ids) do
  begin
    Count := Min(PREFILL_ROWS, Length(Ids) - Offset);
    ForwardRows(Copy(Ids, Offset, Count));
    Inc(Offset, Count);
  end;
  SetLength(Result, FModel.VocabSize);
  FCtx.Download(FLogits, @Result[0], FModel.VocabSize);
end;

end.

