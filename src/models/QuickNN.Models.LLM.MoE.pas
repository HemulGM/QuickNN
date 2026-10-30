unit QuickNN.Models.LLM.MoE;

{$POINTERMATH ON}

interface

uses
  System.SysUtils,
  System.JSON,
  System.Generics.Collections,
  QuickNN.Common,
  QuickNN.SafeTensors,
  QuickNN.Vulkan,
  QuickNN.Vulkan.Operations,
  QuickNN.Models.LLM.Types,
  QuickNN.Models.LLM.Memory,
  QuickNN.Models.LLM.Quantization;

type
  ELLMMoECancelled = class(EAbort);
  TLLMMoE = class
  private
    type
      TExpert = record
        Gate, Up, Down: TMemoryBlock;
      end;
      TLayer = record
        Router: TMemoryBlock;
        Experts: TArray<TExpert>;
      end;
  private
    FLayers: TArray<TLayer>;
    FCache: TLLMHostWeightCache;
    FQuantized: TMoEInt4Cache;
    FNative: TDictionary<IInterface, TMemoryBlock>;
    FHidden, FIntermediate, FExperts, FTopK: Integer;
    FNormalize: Boolean;
    FBatchBytes: UInt64;
    FCancelCheck: TFunc<Boolean>;
    FRouterReads, FRouterBytes, FDispatchBytes, FMaxExpertBytes: UInt64;
    function WeightFor(
        const Ctx: TQNNGPUContext;
        const Weight: TMemoryBlock;
        const InputDim, OutputDim: Integer
    ): TMemoryBlock;
  public
    constructor Create(
        const Files: TSafeTensorFiles;
        const Config: TJSONValue;
        const Layers, Hidden: Integer;
        const RAMLimit: UInt64;
        const BatchBytes: UInt64 = 128 * 1024 * 1024;
        const Quantization: TLLMExpertQuantization = eqNative;
        const CacheDirectory: string = '';
        const Progress: TProc<Integer, Integer> = nil
    );
    destructor Destroy; override;
    function IsSparse(const Layer: Integer): Boolean;
    procedure RetainRouters(const Ctx: TQNNGPUContext);
    function RouterBytes: UInt64;
    function ExpertBytes: UInt64;
    function Forward(
        const Ctx: TQNNGPUContext;
        const Work: TGPUWorkspace;
        const Src: TQNNGPUTensor;
        const Layer, Rows: Integer
    ): TQNNGPUTensor;
    procedure Status(var Value: TLLMMemoryStatus);
    property CancelCheck: TFunc<Boolean> read FCancelCheck write FCancelCheck;
  end;

implementation

uses
  System.Math,
  QuickNN.Quantization.Int4;

constructor TLLMMoE.Create(
    const Files: TSafeTensorFiles;
    const Config: TJSONValue;
    const Layers, Hidden: Integer;
    const RAMLimit, BatchBytes: UInt64;
    const Quantization: TLLMExpertQuantization;
    const CacheDirectory: string;
    const Progress: TProc<Integer, Integer>
);
var
  Step: Integer;
  OnlyLayers: TArray<Boolean>;
  List: TJSONValue;

  function Matrix(const Name: string; const R, C: Integer; const Expert: Boolean = False): TMemoryBlock;
  var
    Tensor: PSafeTensor;
  begin
    Tensor := Files.GetTensor(Name);
    if (Tensor = nil)
        or (Tensor.Ndim <> 2)
        or (Tensor.Shape[0] <> R)
        or (Tensor.Shape[1] <> C)
        or not (Tensor.DType in [StBF16, StF32]) then
      raise EArgumentException.Create('Missing or invalid native MoE matrix: ' + Name);
    if Expert and (FQuantized <> nil) then
      Result := FQuantized.Weight(Name, R, C)
    else
      Result := Files.GetLinearWeight(Name, True);
  end;

begin
  inherited Create;
  FHidden := Hidden;
  FBatchBytes := Min(BatchBytes, UInt64(128) * 1024 * 1024);
  FExperts := Config.GetValue<Integer>('num_experts', 0);
  FTopK := Config.GetValue<Integer>('num_experts_per_tok', 0);
  FIntermediate := Config.GetValue<Integer>('moe_intermediate_size', 0);
  Step := Config.GetValue<Integer>('decoder_sparse_step', 1);
  FNormalize := Config.GetValue<Boolean>('norm_topk_prob', False);
  if (FExperts < 1)
      or (FExperts > 256)
      or (FTopK < 1)
      or (FTopK > Min(FExperts, 64))
      or (FHidden < 1)
      or (FIntermediate < 1)
      or (Step < 1) then
    raise EArgumentException.Create('Invalid Qwen3 MoE geometry (experts <=256, top-k <=64)');
  SetLength(OnlyLayers, Layers);
  List := Config.FindValue('mlp_only_layers');
  if (List <> nil) and not (List is TJSONArray) then
    raise EArgumentException.Create('Invalid mlp_only_layers');
  if List is TJSONArray then
    for var Item in TJSONArray(List) do
    begin
      var Index := Item.AsType<Integer>;
      if (Index < 0) or (Index >= Layers) then
        raise EArgumentOutOfRangeException.Create('Invalid dense layer index');
      OnlyLayers[Index] := True;
    end;
  SetLength(FLayers, Layers);
  if Quantization = eqInt4 then
    FQuantized := TMoEInt4Cache.Create(Files, CacheDirectory, Progress);
  FCache := TLLMHostWeightCache.Create(RAMLimit);
  FNative := TDictionary<IInterface, TMemoryBlock>.Create;
  for var L := 0 to Layers - 1 do
    if not OnlyLayers[L] and ((L + 1) mod Step = 0) then
    begin
      var Prefix := Format('model.layers.%d.mlp.', [L]);
      FLayers[L].Router := Matrix(Prefix + 'gate.weight', FExperts, FHidden);
      SetLength(FLayers[L].Experts, FExperts);
      for var E := 0 to FExperts - 1 do
      begin
        var Name := Prefix + 'experts.' + E.ToString + '.';
        FLayers[L].Experts[E].Gate := Matrix(Name + 'gate_proj.weight', FIntermediate, FHidden, True);
        FLayers[L].Experts[E].Up := Matrix(Name + 'up_proj.weight', FIntermediate, FHidden, True);
        FLayers[L].Experts[E].Down := Matrix(Name + 'down_proj.weight', FHidden, FIntermediate, True);
        FNative.Add(FLayers[L].Experts[E].Gate.WeightIdentity, FLayers[L].Experts[E].Gate);
        FNative.Add(FLayers[L].Experts[E].Up.WeightIdentity, FLayers[L].Experts[E].Up);
        FNative.Add(FLayers[L].Experts[E].Down.WeightIdentity, FLayers[L].Experts[E].Down);
        var ExpertSize :=
            UInt64(FLayers[L].Experts[E].Gate.Size) * UInt64(DATATYPE_BITS[FLayers[L].Experts[E].Gate.DataType] div 8)
                + UInt64(FLayers[L].Experts[E].Up.Size) * UInt64(DATATYPE_BITS[FLayers[L].Experts[E].Up.DataType] div 8)
                + UInt64(FLayers[L].Experts[E].Down.Size)
                    * UInt64(DATATYPE_BITS[FLayers[L].Experts[E].Down.DataType] div 8);
        FMaxExpertBytes := Max(FMaxExpertBytes, ExpertSize);
      end;
    end;
end;

destructor TLLMMoE.Destroy;
begin
  FCache.Free;
  FNative.Free;
  for var L := 0 to High(FLayers) do
  begin
    FLayers[L].Router.Free;
    for var E := 0 to High(FLayers[L].Experts) do
    begin
      FLayers[L].Experts[E].Gate.Free;
      FLayers[L].Experts[E].Up.Free;
      FLayers[L].Experts[E].Down.Free;
    end;
  end;
  FQuantized.Free;
  inherited;
end;

function TLLMMoE.IsSparse(const Layer: Integer): Boolean;
begin
  Result := Length(FLayers[Layer].Experts) <> 0;
end;

procedure TLLMMoE.RetainRouters(const Ctx: TQNNGPUContext);
begin
  for var L := 0 to High(FLayers) do
    if IsSparse(L) then
      Ctx.RetainWeight(FLayers[L].Router, FExperts, FHidden);
  Ctx.OnWeightEvicted :=
      procedure(Identity: IInterface)
      var
        Native: TMemoryBlock;
      begin
        if FNative.TryGetValue(Identity, Native) then
          FCache.Store(Native);
      end;
end;

function TLLMMoE.RouterBytes: UInt64;
begin
  Result := 0;
  for var L := 0 to High(FLayers) do
    if IsSparse(L) then
      Inc(Result, UInt64(FLayers[L].Router.Size) * UInt64(DATATYPE_BITS[FLayers[L].Router.DataType] div 8));
end;

function TLLMMoE.ExpertBytes: UInt64;
begin
  Result := FMaxExpertBytes;
end;

function TLLMMoE.WeightFor(
    const Ctx: TQNNGPUContext;
    const Weight: TMemoryBlock;
    const InputDim, OutputDim: Integer
): TMemoryBlock;
begin
  var Width := InputDim;
  if Weight.DataType = dtU8 then
    Width := Int4RowBytes(InputDim);
  if Ctx.IsWeightResident(Weight, OutputDim, Width) then
    Result := Weight
  else if not FCache.TryTake(Weight, Result) then
    Result := Weight;
end;

function TLLMMoE.Forward(
    const Ctx: TQNNGPUContext;
    const Work: TGPUWorkspace;
    const Src: TQNNGPUTensor;
    const Layer, Rows: Integer
): TQNNGPUTensor;
var
  Logits, Routes, Indices, Gathered, G, U, Y: TQNNGPUTensor;
  HostRoutes, Order: TArray<UInt32>;
  Counts, Offsets, Next: TArray<Integer>;
  ExpertWork: TGPUWorkspace;
  ExpertRows, RouteIndex: Integer;
  Group: TArray<Integer>;
  Cursor, GroupCount: Integer;
  GroupBytes, Budget, Bytes: UInt64;
  Filled: Boolean;

  function Project(const Input: TQNNGPUTensor; const Weight: TMemoryBlock; const InDim, OutDim: Integer): TQNNGPUTensor;
  var
    Native: TMemoryBlock;
  begin
    Native := WeightFor(Ctx, Weight, InDim, OutDim);
    try
      if Native.DataType = dtU8 then
      begin
        Result := ExpertWork.New(ExpertRows * OutDim);
        Ctx.Int4Linear(Result, Input, Native, ExpertRows, InDim, OutDim);
      end
      else if ExpertRows = 1 then
      begin
        Result := ExpertWork.New(OutDim);
        Ctx.VectorLinear(Result, Input, Native, InDim, OutDim);
      end
      else
        Result := ExpertWork.Linear(Input, Native, ExpertRows, InDim, OutDim);
    finally
      Native.Free;
    end;
  end;

  function PaddedBytes(const Weight: TMemoryBlock): UInt64;
  begin
    Result := (UInt64(Weight.Size) * UInt64(DATATYPE_BITS[Weight.DataType] div 8) + 3) and not UInt64(3);
  end;

  procedure ExecuteExpert(const E: Integer; const Output: TQNNGPUTensor);
  begin
    if Assigned(FCancelCheck) and FCancelCheck() then
      raise ELLMMoECancelled.Create('MoE generation cancelled');
    ExpertRows := Counts[E];
    if ExpertRows = 0 then
      Exit;
    if not Filled then
    begin
      Ctx.Kernel(GO_FILL, Output.Count, Output, nil, nil, nil, [0]);
      Filled := True;
    end;
    ExpertWork := TGPUWorkspace.Create(Ctx);
    try
      RouteIndex := 0;
      if Rows = 1 then
      begin
        Gathered := Src;
        while HostRoutes[RouteIndex] <> UInt32(E) do
          Inc(RouteIndex);
      end
      else
      begin
        Gathered := ExpertWork.New(ExpertRows * FHidden);
        Ctx.Kernel(GO_MOE_GATHER, Gathered.Count, Gathered, Src, Indices, nil, [FHidden, FTopK, Offsets[E]]);
      end;
      G := Project(Gathered, FLayers[Layer].Experts[E].Gate, FHidden, FIntermediate);
      U := Project(Gathered, FLayers[Layer].Experts[E].Up, FHidden, FIntermediate);
      Ctx.Kernel(GO_SWIGLU, ExpertRows * FIntermediate, G, G, U, nil, [FIntermediate, 0]);
      Y := Project(G, FLayers[Layer].Experts[E].Down, FIntermediate, FHidden);
      Ctx.Kernel(
          GO_MOE_SCATTER,
          Y.Count,
          Output,
          Y,
          Indices,
          Routes,
          [FHidden, FTopK, Offsets[E], RouteIndex, Ord(Rows = 1), Rows * FTopK]
      );
    finally
      ExpertWork.Free;
    end;
  end;

  procedure PreloadGroup;
  var
    Native: TArray<TMemoryBlock>;
  begin
    if Assigned(FCancelCheck) and FCancelCheck() then
      raise ELLMMoECancelled.Create('MoE generation cancelled');
    SetLength(Native, GroupCount * 3);
    try
      for var I := 0 to GroupCount - 1 do
      begin
        var E := Group[I];
        Native[3 * I] := WeightFor(Ctx, FLayers[Layer].Experts[E].Gate, FHidden, FIntermediate);
        Native[3 * I + 1] := WeightFor(Ctx, FLayers[Layer].Experts[E].Up, FHidden, FIntermediate);
        Native[3 * I + 2] := WeightFor(Ctx, FLayers[Layer].Experts[E].Down, FIntermediate, FHidden);
      end;
      Ctx.PreloadWeights(Native);
    finally
      for var I := 0 to High(Native) do
        Native[I].Free;
    end;
  end;

begin
  if not IsSparse(Layer) or (Rows < 1) or (Rows > 128) then
    raise EArgumentException.Create('Invalid MoE forward geometry');
  Logits := Work.New(Rows * FExperts);
  if Rows = 1 then
    Ctx.VectorLinear(Logits, Src, FLayers[Layer].Router, FHidden, FExperts)
  else
    Ctx.Linear(
        Logits,
        Src,
        FLayers[Layer].Router,
        Default(TMemoryBlock),
        Default(TMemoryBlock),
        Rows,
        FHidden,
        FExperts
    );
  Ctx.CheckFinite(Logits);
  Routes := Work.New(Rows * FTopK * 2);
  Ctx.Kernel(GO_MOE_ROUTE, Routes.Count, Routes, Logits, nil, nil, [Rows, FExperts, FTopK, Ord(FNormalize)]);
  SetLength(HostRoutes, Rows * FTopK);
  Ctx.Download(Routes, @HostRoutes[0], Length(HostRoutes));
  Inc(FRouterReads);
  Inc(FRouterBytes, UInt64(Length(HostRoutes)) * 4 + 4);
  SetLength(Counts, FExperts);
  SetLength(Offsets, FExperts);
  for var R := 0 to Rows * FTopK - 1 do
  begin
    var E := HostRoutes[R];
    if E >= UInt32(FExperts) then
      raise EInvalidOp.Create('Invalid GPU expert ID');
    Inc(Counts[E]);
  end;
  for var E := 1 to FExperts - 1 do
    Offsets[E] := Offsets[E - 1] + Counts[E - 1];
  Indices := nil;
  if Rows > 1 then
  begin
    Next := Copy(Offsets);
    SetLength(Order, Rows * FTopK);
    for var R := 0 to Rows * FTopK - 1 do
    begin
      var E := HostRoutes[R];
      Order[Next[E]] := R;
      Inc(Next[E]);
    end;
    Indices := Work.Own(Ctx.UploadRaw(@Order[0], Length(Order)));
    Inc(FDispatchBytes, UInt64(Length(Order)) * 4);
  end;
  Result := Work.New(Rows * FHidden);
  Filled := False;
  Cursor := 0;
  Budget := Min(FBatchBytes, Ctx.WeightBatchCapacity);
  SetLength(Group, FExperts);
  while Cursor < FExperts do
  begin
    GroupCount := 0;
    GroupBytes := 0;
    while Cursor < FExperts do
    begin
      if Counts[Cursor] = 0 then
      begin
        Inc(Cursor);
        Continue
      end;
      Bytes :=
          PaddedBytes(FLayers[Layer].Experts[Cursor].Gate)
              + PaddedBytes(FLayers[Layer].Experts[Cursor].Up)
              + PaddedBytes(FLayers[Layer].Experts[Cursor].Down);
      if Bytes > Budget then
      begin
        if GroupCount <> 0 then
          Break;
        ExecuteExpert(Cursor, Result);
        Inc(Cursor);
        Continue;
      end;
      if Bytes > Budget - GroupBytes then
        Break;
      Group[GroupCount] := Cursor;
      Inc(GroupCount);
      Inc(GroupBytes, Bytes);
      Inc(Cursor);
    end;
    if GroupCount <> 0 then
    begin
      PreloadGroup;
      for var I := 0 to GroupCount - 1 do
        ExecuteExpert(Group[I], Result);
    end;
  end;
end;

procedure TLLMMoE.Status(var Value: TLLMMemoryStatus);
begin
  Value.IsMoE := True;
  if FQuantized <> nil then
  begin
    Value.ExpertQuantization := eqInt4;
    Value.QuantizedMappedWeightBytes := FQuantized.MappedBytes;
    Value.ExpertCacheDirectory := FQuantized.Directory;
  end;
  Value.ExpertsPerLayer := FExperts;
  Value.ExpertsPerToken := FTopK;
  Value.RouterReadbacks := FRouterReads;
  Value.RouterReadbackBytes := FRouterBytes;
  Value.DispatchUploadBytes := FDispatchBytes;
  FCache.Status(Value);
end;

end.
