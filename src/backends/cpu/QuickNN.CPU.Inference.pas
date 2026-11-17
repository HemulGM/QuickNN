unit QuickNN.CPU.Inference;

{$POINTERMATH ON}

interface

uses
  System.SysUtils, System.Classes, System.SyncObjs, System.Generics.Collections, System.Generics.Defaults, QuickNN.Common;

const
  QNN_CPU_WEIGHT_ALIGNMENT = 64;

type
  TInferencePool = class;
  TCPUInferenceWorker = class(TThread)
  private
    FOwner: TInferencePool;
    FIndex: Integer;
    FStart, FDone: TEvent;
  protected
    procedure Execute; override;
  public
    constructor Create(const Owner: TInferencePool; const Index: Integer);
    destructor Destroy; override;
  end;

  // Owns persistent CPU workers. Jobs never call GPU APIs or touch model state.
  TInferencePool = class
  private
    FWorkers: TArray<TCPUInferenceWorker>;
    FErrors: TArray<string>;
    FJob: TProc<Integer>;
    FThreadCount: Integer;
    procedure EnsureWorkers;
  public
    constructor Create(const Threads: Integer);
    destructor Destroy; override;
    procedure Run(const Job: TProc<Integer>);
    property ThreadCount: Integer read FThreadCount;
  end;

  TCPUProjection = class
  private type
    TPreparedMatrix = class
      Identity: IInterface;
      Data: TMemoryBlock;
      InputDim, OutputDim: Integer;
      RowBytes: NativeInt;
      Address: PByte;
      destructor Destroy; override;
    end;
  private
    FPrepared: TObjectDictionary<Pointer,TPreparedMatrix>;
    FPreparedBytes, FPreparedHits, FPreparedMisses: UInt64;
    FPool: TInferencePool;
  public
    constructor Create(const Threads: Integer);
    destructor Destroy; override;
    function Prepare(const Weights: TArray<TMemoryBlock>; const Budget: UInt64; const AllowPartial: Boolean = False): Boolean;
    property PreparedBytes: UInt64 read FPreparedBytes;
    property PreparedHits: UInt64 read FPreparedHits;
    property PreparedMisses: UInt64 read FPreparedMisses;
    procedure Linear(const Weight: TMemoryBlock; const Src, Dst: System.PSingle;
      const InputDim, OutputDim: Integer);
  end;

function InferenceDot(const Count: Integer; const A, B: System.PSingle): Single;
procedure InferenceMemory(const Count: Integer; const State, Memory: System.PSingle; const Decay, Key: Single);
procedure InferenceUpdate(const Count: Integer; const State, Delta, Output: System.PSingle; const Decay, Key, Query: Single);

implementation

uses
  System.Math, QuickNN.GGUF.Quantization, QuickNN.Quantization.Int4;

{$I QuickNN.CPU.Inference.AVX2.inc}

constructor TCPUInferenceWorker.Create(const Owner: TInferencePool; const Index: Integer);
begin
  // RTL starts Execute after the constructor has initialized both events.
  inherited Create(False);
  FreeOnTerminate := False;
  FOwner := Owner; FIndex := Index;
  FStart := TEvent.Create(nil, False, False, '');
  FDone := TEvent.Create(nil, False, False, '');
end;

destructor TCPUInferenceWorker.Destroy;
begin
  inherited;
  FStart.Free; FDone.Free;
end;

procedure TCPUInferenceWorker.Execute;
begin
  while not Terminated do
  begin
    FStart.WaitFor(INFINITE);
    if Terminated then Exit;
    try
      FOwner.FJob(FIndex);
    except
      on E: Exception do FOwner.FErrors[FIndex] := E.ClassName + ': ' + E.Message;
    end;
    FDone.SetEvent;
  end;
end;

constructor TInferencePool.Create(const Threads: Integer);
begin
  inherited Create;
  if (Threads < 0) or (Threads > 64) then raise EArgumentOutOfRangeException.Create('Invalid CPU inference thread count');
  FThreadCount := Threads;
  if FThreadCount = 0 then FThreadCount := Min(8, TThread.ProcessorCount);
  FThreadCount := Max(1, FThreadCount);
end;

procedure TInferencePool.EnsureWorkers;
begin
  if Length(FErrors) <> 0 then Exit;
  SetLength(FErrors, FThreadCount);
  SetLength(FWorkers, FThreadCount-1);
  try
    for var I := 0 to High(FWorkers) do FWorkers[I] := TCPUInferenceWorker.Create(Self, I+1);
  except
    for var Worker in FWorkers do if Worker<>nil then
    begin Worker.Terminate; Worker.FStart.SetEvent end;
    for var Worker in FWorkers do if Worker<>nil then
    begin Worker.WaitFor; Worker.Free end;
    FWorkers := nil; FErrors := nil;
    raise;
  end;
end;

destructor TInferencePool.Destroy;
begin
  for var Worker in FWorkers do if Worker <> nil then
  begin Worker.Terminate; Worker.FStart.SetEvent end;
  for var Worker in FWorkers do if Worker <> nil then
  begin Worker.WaitFor; Worker.Free end;
  inherited;
end;

procedure TInferencePool.Run(const Job: TProc<Integer>);
begin
  if Assigned(FJob) then raise EInvalidOp.Create('CPU inference pool cannot run recursively');
  EnsureWorkers; FJob := Job;
  try
    for var I := 0 to High(FErrors) do FErrors[I] := '';
    for var Worker in FWorkers do Worker.FStart.SetEvent;
    try Job(0)
    finally
      for var Worker in FWorkers do
        if Worker.FDone.WaitFor(INFINITE) <> wrSignaled then raise EInvalidOp.Create('CPU inference worker wait failed');
    end;
    for var Error in FErrors do if Error <> '' then raise EInvalidOp.Create(Error);
  finally FJob := nil end;
end;

function InferenceDot(const Count: Integer; const A, B: System.PSingle): Single;
begin
{$IF defined(CPUX64) and defined(USE_AVX2)}
  Result := VectorDot(Count, A, B);
{$ELSE}
  Result := 0;
  for var I := 0 to Count-1 do Result := Result + A[I]*B[I];
{$ENDIF}
end;

procedure InferenceMemory(const Count: Integer; const State, Memory: System.PSingle; const Decay, Key: Single);
begin
{$IF defined(CPUX64) and defined(USE_AVX2)}
  var Factors: array[0..1] of Single; Factors[0] := Decay; Factors[1] := Key;
  MemoryAccumulate(Count, State, Memory, @Factors[0]);
{$ELSE}
  for var I := 0 to Count-1 do Memory[I] := Memory[I] + (State[I]*Decay)*Key;
{$ENDIF}
end;

procedure InferenceUpdate(const Count: Integer; const State, Delta, Output: System.PSingle; const Decay, Key, Query: Single);
begin
{$IF defined(CPUX64) and defined(USE_AVX2)}
  var Factors: array[0..2] of Single; Factors[0] := Decay; Factors[1] := Key; Factors[2] := Query;
  StateUpdate(Count, State, Delta, Output, @Factors[0]);
{$ELSE}
  for var I := 0 to Count-1 do
  begin State[I] := State[I]*Decay + Delta[I]*Key; Output[I] := Output[I] + State[I]*Query end;
{$ENDIF}
end;

constructor TCPUProjection.Create(const Threads: Integer);
begin
  inherited Create;
  FPool := TInferencePool.Create(Threads);
  FPrepared := TObjectDictionary<Pointer,TPreparedMatrix>.Create([doOwnsValues]);
end;

destructor TCPUProjection.Destroy;
begin
  FPool.Free;
  FPrepared.Free;
  inherited;
end;

destructor TCPUProjection.TPreparedMatrix.Destroy;
begin
  Data.Free;
  inherited;
end;

function PreparedMatrixBytes(const Weight: TMemoryBlock): UInt64;
begin
  Result := UInt64(Weight.Shape[0]*Weight.Shape[1] div GGUFBlockElements(Weight.Encoding))*UInt64(GGUFCPUBlockBytes(Weight.Encoding))+QNN_CPU_WEIGHT_ALIGNMENT-1;
end;

function PreparedMatrixPriority(const Weight: TMemoryBlock): Double;
begin
  // Prefer formats with expensive codebook/scale/bit decoding per cached byte.
  case Weight.Encoding of
    weGGUFIQ3_S: Result := 8;
    weGGUFQ5_K: Result := 4;
    weGGUFQ4_K, weGGUFIQ4_NL: Result := 3;
    weGGUFIQ4_XS: Result := 2;
    weGGUFQ3_K: Result := 1;
    weGGUFQ6_K: Result := 0.5;
  else Result := 0.25 end;
  Result := Result*GGUFBlockElements(Weight.Encoding)/GGUFCPUBlockBytes(Weight.Encoding);
end;

function TCPUProjection.Prepare(const Weights: TArray<TMemoryBlock>; const Budget: UInt64; const AllowPartial: Boolean): Boolean;
var
  Selected: TDictionary<Pointer,TMemoryBlock>;
  Total: UInt64;


begin
  Result := False;
  if FPrepared.Count<>0 then raise EInvalidOp.Create('CPU weights are already prepared');
  Selected := TDictionary<Pointer,TMemoryBlock>.Create;
  try
    Total := 0;
    for var Weight in Weights do
    begin
      if Weight.Encoding=weNative then Continue;
      if not Weight.IsAssigned or (Weight.WeightIdentity=nil) or (Weight.Offset<>0)
        or (Length(Weight.Shape)<>2) or (Weight.Shape[0]<1) or (Weight.Shape[1]<1)
        or (Weight.Shape[0]>High(Integer)) or (Weight.Shape[1]>High(Integer)) then Exit;
      var Key := Pointer(Weight.WeightIdentity);
      if Selected.ContainsKey(Key) then Continue;
      GGUFWeightBytes(Weight,Weight.Shape[0],Weight.Shape[1]);
      var Bytes := PreparedMatrixBytes(Weight);
      if (Bytes>UInt64(High(NativeInt))) or (Bytes>High(UInt64)-Total) then Exit;
      Inc(Total,Bytes); Selected.Add(Key,Weight);
    end;
    if Total>Budget then
    begin
      if not AllowPartial then Exit;
      var Candidates := Selected.ToArray;
      TArray.Sort<TPair<Pointer,TMemoryBlock>>(Candidates,
        TComparer<TPair<Pointer,TMemoryBlock>>.Construct(
          function(const Left, Right: TPair<Pointer,TMemoryBlock>): Integer
          begin
            var L := PreparedMatrixPriority(Left.Value); var R := PreparedMatrixPriority(Right.Value);
            if L>R then Exit(-1); if L<R then Exit(1);
            var LB := PreparedMatrixBytes(Left.Value); var RB := PreparedMatrixBytes(Right.Value);
            if LB<RB then Exit(-1); if LB>RB then Exit(1);
            Result := CompareStr(Left.Value.Name,Right.Value.Name);
          end));
      Selected.Clear; Total := 0;
      for var Candidate in Candidates do
      begin
        var Bytes := PreparedMatrixBytes(Candidate.Value);
        if Bytes>Budget-Total then Continue;
        Selected.Add(Candidate.Key,Candidate.Value); Inc(Total,Bytes);
      end;
    end;
    if Selected.Count=0 then Exit;
    try
      for var Pair in Selected do
      begin
        var Weight := Pair.Value;
        var Matrix := TPreparedMatrix.Create;
        try
          Matrix.Identity := Weight.WeightIdentity;
          Matrix.InputDim := Weight.Shape[1]; Matrix.OutputDim := Weight.Shape[0];
          Matrix.RowBytes := (NativeInt(Matrix.InputDim) div GGUFBlockElements(Weight.Encoding))*GGUFCPUBlockBytes(Weight.Encoding);
          Matrix.Data := TMemoryBlock.Create([Int64(Matrix.RowBytes)*Matrix.OutputDim+QNN_CPU_WEIGHT_ALIGNMENT-1],'cpu-prepared',dtU8);
          Matrix.Address := PByte((NativeUInt(PByte(Matrix.Data))+QNN_CPU_WEIGHT_ALIGNMENT-1) and not NativeUInt(QNN_CPU_WEIGHT_ALIGNMENT-1));
          var Source := PByte(GGUFWeightPointer(Weight));
          var Destination := Matrix.Address;
          var SourceRowBytes := NativeInt(GGUFQuantBytes(Weight.Encoding,Matrix.InputDim));
          var Threads := FPool.ThreadCount;
          FPool.Run(procedure(Index: Integer)
            begin
              for var R := (Int64(Matrix.OutputDim)*Index) div Threads to (Int64(Matrix.OutputDim)*(Index+1)) div Threads-1 do
                GGUFPrepareCPU(Source+R*SourceRowBytes,Weight.Encoding,Matrix.InputDim,Destination+R*Matrix.RowBytes);
            end);
          FPrepared.Add(Pair.Key,Matrix);
        except
          Matrix.Free;
          raise;
        end;
      end;
      FPreparedBytes := Total;
      Result := True;
    except
      FPrepared.Clear;
      FPreparedBytes := 0;
      raise;
    end;
  finally
    Selected.Free;
  end;
end;

procedure TCPUProjection.Linear(const Weight: TMemoryBlock; const Src, Dst: System.PSingle;
  const InputDim, OutputDim: Integer);
var
  PackedData: PByte;
  RowBytes: NativeInt;
  Threads: Integer;
  Job: TProc<Integer>;
  Prepared: TPreparedMatrix;
  InputSums: TArray<Single>;
  AlignedSrc: System.PSingle;
{$IF defined(CPUX64) and defined(USE_AVX2)}
  AlignedInput: TArray<Single>;
{$ENDIF}
  SumPointer: System.PSingle;
begin
  if (Src = nil) or (Dst = nil) or (InputDim < 1) or (OutputDim < 1)
    or not Weight.IsAssigned or (Length(Weight.Shape) <> 2) or (Weight.Shape[0] <> OutputDim) then
    raise EArgumentException.Create('Invalid CPU linear projection');
  if Weight.Encoding <> weNative then
  begin
    if Weight.Shape[1] <> InputDim then raise EArgumentException.Create('Invalid CPU GGUF projection shape');
    GGUFWeightBytes(Weight, OutputDim, InputDim);
    RowBytes := NativeInt(GGUFQuantBytes(Weight.Encoding, InputDim)); PackedData := GGUFWeightPointer(Weight);
  end
  else if Weight.DataType = dtU8 then
  begin
    RowBytes := Int4RowBytes(InputDim); PackedData := PByte(Weight);
    if (Weight.Shape[1] <> RowBytes) or (Int64(Weight.Count)<Int64(OutputDim)*RowBytes) then raise EArgumentException.Create('Invalid CPU INT4 projection shape');
  end
  else
  begin
    if (Weight.Shape[1] <> InputDim) or (Int64(Weight.Count)<Int64(OutputDim)*InputDim) or not (Weight.DataType in [dtF32,dtBF16]) then
      raise EArgumentException.Create('Invalid CPU native projection shape');
    RowBytes := NativeInt(InputDim) * (DATATYPE_BITS[Weight.DataType] div 8);
    if Weight.DataType = dtBF16 then PackedData := PByte(PBF16(Weight)) else PackedData := PByte(System.PSingle(Weight));
  end;
  Prepared := nil;
  if FPrepared.TryGetValue(Pointer(Weight.WeightIdentity),Prepared) then
  begin
    if (InputDim<>Prepared.InputDim) or (OutputDim<>Prepared.OutputDim) then
      raise EArgumentException.Create('Prepared CPU projection geometry mismatch');
    Inc(FPreparedHits);
    PackedData := Prepared.Address; RowBytes := Prepared.RowBytes;
  end
  else if (FPrepared.Count<>0) and (Weight.Encoding<>weNative) then
    Inc(FPreparedMisses);
  AlignedSrc := Src;
{$IF defined(CPUX64) and defined(USE_AVX2)}
  if (NativeUInt(Src) and (QNN_CPU_WEIGHT_ALIGNMENT-1))<>0 then
  begin
    SetLength(AlignedInput,InputDim+QNN_CPU_WEIGHT_ALIGNMENT div SizeOf(Single));
    AlignedSrc := System.PSingle((NativeUInt(@AlignedInput[0])+QNN_CPU_WEIGHT_ALIGNMENT-1) and not NativeUInt(QNN_CPU_WEIGHT_ALIGNMENT-1));
    Move(Src^,AlignedSrc^,NativeInt(InputDim)*SizeOf(Single));
  end;
{$ENDIF}
  SumPointer := nil;
  if (Prepared<>nil) and (Weight.Encoding in [weGGUFQ4_K,weGGUFQ5_K]) then
  begin
    SetLength(InputSums,InputDim div 4+QNN_CPU_WEIGHT_ALIGNMENT div SizeOf(Single));
    SumPointer := System.PSingle((NativeUInt(@InputSums[0])+QNN_CPU_WEIGHT_ALIGNMENT-1) and not NativeUInt(QNN_CPU_WEIGHT_ALIGNMENT-1));
    GGUFCPUInputSums(AlignedSrc,SumPointer,InputDim);
  end;
  Threads := 1;
  if Int64(InputDim)*OutputDim >= 1048576 then Threads := FPool.ThreadCount;
  Job := procedure(Index: Integer)
    var Scratch: TArray<Single>; Row: System.PSingle;
    begin
{$IF defined(CPUX64) and defined(USE_AVX2)}
      if (Prepared=nil) and (Weight.Encoding in [weGGUFIQ3_S,weGGUFQ2_K,weGGUFIQ2_XXS,weGGUFIQ2_S,weGGUFQ2_0,weGGUFIQ3_XXS]) then
{$ELSE}
      if (Prepared=nil) and ((Weight.Encoding <> weNative) or (Weight.DataType <> dtF32)) then
{$ENDIF}
        SetLength(Scratch, ((InputDim+31) div 32)*32);
      for var R := (Int64(OutputDim)*Index) div Threads to (Int64(OutputDim)*(Index+1)) div Threads-1 do
      begin
        if Prepared<>nil then
        begin
          Dst[R] := GGUFDotCPU(PackedData+R*RowBytes,Weight.Encoding,AlignedSrc,InputDim,SumPointer);
          Continue;
        end;
        if (Weight.Encoding<>weNative) and GGUFTryDot(PackedData+R*RowBytes, Weight.Encoding, AlignedSrc, InputDim, Dst[R]) then Continue;
{$IF defined(CPUX64) and defined(USE_AVX2)}
        if Weight.Encoding=weNative then
        begin
          if Weight.DataType=dtBF16 then
          begin Dst[R] := BF16VectorDot(InputDim,AlignedSrc,PBF16(PackedData+R*RowBytes)); Continue end;
          if Weight.DataType=dtU8 then
          begin Dst[R] := Int4VectorDot(InputDim,AlignedSrc,PackedData+R*RowBytes); Continue end;
        end;
{$ENDIF}
        Row := System.PSingle(PackedData + R*RowBytes);
        if Weight.Encoding <> weNative then
        begin GGUFDequantize(PackedData+R*RowBytes, Weight.Encoding, InputDim, @Scratch[0]); Row := @Scratch[0] end
        else if Weight.DataType = dtBF16 then
        begin BF16ToSingle(InputDim, PBF16(PackedData+R*RowBytes), @Scratch[0]); Row := @Scratch[0] end
        else if Weight.DataType = dtU8 then
        begin
          for var Group := 0 to (InputDim+31) div 32-1 do
{$IF defined(CPUX64) and defined(USE_AVX2)}
            ExpandInt4(PackedData+R*RowBytes+Group*20, @Scratch[Group*32]);
{$ELSE}
          begin
            var Base := PackedData+R*RowBytes+Group*20;
            for var J := 0 to 31 do Scratch[Group*32+J] := (Integer((Base[4+J div 2] shr ((J and 1)*4)) and 15)-8)*System.PSingle(Base)^;
          end;
{$ENDIF}
          Row := @Scratch[0];
        end;
        Dst[R] := InferenceDot(InputDim, AlignedSrc, Row);
      end;
    end;
  if Threads = 1 then Job(0) else FPool.Run(Job);
end;

end.
