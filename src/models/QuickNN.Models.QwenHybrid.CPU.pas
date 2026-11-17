unit QuickNN.Models.QwenHybrid.CPU;

{$POINTERMATH ON}

interface

uses
  System.SysUtils, QuickNN.Common, QuickNN.CPU.Inference;

type
  TQwenCPUConfig = record
    Hidden, Intermediate, Heads, KVHeads, HeadDim, Capacity: Integer;
    KHeads, VHeads, KDim, VDim, ConvSize, RotaryDim: Integer;
    Epsilon, Theta: Single;
    GGUF: Boolean;
  end;

  // Matrix storage is borrowed from the model or its bounded host cache.
  TQwenCPUWeights = record
    Linear: Boolean;
    InputNorm, PostNorm, Gate, Up, Down: TMemoryBlock;
    Q, K, V, O, QNorm, KNorm: TMemoryBlock;
    QKV, Z, A, B, ALog, DtBias, Conv, GatedNorm: TMemoryBlock;
  end;

  TQwenCPULayer = class
  private
    FConfig: TQwenCPUConfig;
    FWeights: TQwenCPUWeights;
    FInputNorm, FPostNorm, FQNorm, FKNorm: TArray<Single>;
    FConv, FGatedNorm, FALog, FDtBias: TArray<Single>;
    procedure Delta(const Input: TArray<Single>; const Projection: TCPUProjection; out Output: TArray<Single>);
    procedure Attention(const Input: TArray<Single>; const Position: Integer;
      const Positions: TArray<LongInt>; const Projection: TCPUProjection; out Output: TArray<Single>);
    procedure Normalize(const Input, Gamma: TArray<Single>; const Dim: Integer; out Output: TArray<Single>);
  public
    State, ConvState: TArray<Single>;
    CPUCurrent, GPUCurrent: Boolean;
    constructor Create(const Config: TQwenCPUConfig; const Weights: TQwenCPUWeights);
    procedure Reset;
    procedure Step(var Hidden: TArray<Single>; const Position: Integer;
      const Positions: TArray<LongInt>; const Projection: TCPUProjection);
  end;

procedure QwenCPUApplyRoPE(var X: TArray<Single>; const Heads, Dim, RotaryDim, Position: Integer; const Theta: Single; const Positions: TArray<LongInt>);

implementation

uses System.Math;

procedure QwenCPUApplyRoPE(var X: TArray<Single>; const Heads, Dim, RotaryDim, Position: Integer; const Theta: Single; const Positions: TArray<LongInt>);
begin
  for var H := 0 to Heads-1 do
    for var J := 0 to RotaryDim div 2-1 do
    begin
      var Pos := Position;
      if Length(Positions)=3 then
      begin
        var Axis := J mod 3;
        if ((Axis=1) and (J>=33)) or ((Axis=2) and (J>=30)) then Axis := 0;
        Pos := Positions[Axis];
      end;
      var Angle := Pos*Power(Theta, -2*J/RotaryDim);
      var First := H*Dim+J; var Second := First+RotaryDim div 2;
      var A := X[First]; var B := X[Second];
      X[First] := A*Cos(Angle)-B*Sin(Angle); X[Second] := A*Sin(Angle)+B*Cos(Angle);
    end;
end;

function Vector(const Weight: TMemoryBlock): TArray<Single>;
begin
  SetLength(Result, Weight.Count);
  if Weight.DataType = dtBF16 then BF16ToSingle(Weight.Count, PBF16(Weight), @Result[0])
  else if Weight.DataType = dtF32 then Move(System.PSingle(Weight)^, Result[0], Weight.Count*4)
  else raise EArgumentException.Create('CPU hybrid vector must be FP32 or BF16');
end;

function Sigmoid(const X: Single): Single;
begin
  if X >= 0 then Result := 1/(1+Exp(-X))
  else begin var E := Exp(X); Result := E/(1+E) end;
end;

constructor TQwenCPULayer.Create(const Config: TQwenCPUConfig; const Weights: TQwenCPUWeights);
begin
  inherited Create; FConfig := Config; FWeights := Weights;
  FInputNorm := Vector(Weights.InputNorm); FPostNorm := Vector(Weights.PostNorm);
  if Weights.Linear then
  begin
    FConv := Vector(Weights.Conv); FGatedNorm := Vector(Weights.GatedNorm);
    FALog := Vector(Weights.ALog); FDtBias := Vector(Weights.DtBias);
    SetLength(State, Config.VHeads*Config.KDim*Config.VDim);
    SetLength(ConvState, (2*Config.KHeads*Config.KDim+Config.VHeads*Config.VDim)*Config.ConvSize);
  end
  else
  begin
    FQNorm := Vector(Weights.QNorm); FKNorm := Vector(Weights.KNorm);
    SetLength(State, 2*Config.Capacity*Config.KVHeads*Config.HeadDim);
  end;
  Reset;
end;

procedure TQwenCPULayer.Reset;
begin
  if Length(State)>0 then FillChar(State[0], Length(State)*4, 0);
  if Length(ConvState)>0 then FillChar(ConvState[0], Length(ConvState)*4, 0);
  CPUCurrent := True; GPUCurrent := True;
end;

procedure TQwenCPULayer.Normalize(const Input, Gamma: TArray<Single>; const Dim: Integer; out Output: TArray<Single>);
begin
  SetLength(Output, Length(Input));
  for var H := 0 to Length(Input) div Dim-1 do
  begin
    var Sum: Double := 0;
    for var J := 0 to Dim-1 do Sum := Sum + Sqr(Input[H*Dim+J]);
    var Inv: Single := 1/Sqrt(Sum/Dim+FConfig.Epsilon);
    for var J := 0 to Dim-1 do Output[H*Dim+J] := Input[H*Dim+J]*Inv*(Gamma[J]+Ord(not FConfig.GGUF));
  end;
end;

procedure TQwenCPULayer.Delta(const Input: TArray<Single>; const Projection: TCPUProjection; out Output: TArray<Single>);
var
  Mixed, Z, A, B, Values, Memory, DeltaValue: TArray<Single>;
begin
  var C := FConfig;
  var KeyDim := C.KHeads*C.KDim; var ValueDim := C.VHeads*C.VDim;
  SetLength(Mixed, 2*KeyDim+ValueDim); SetLength(Z, ValueDim);
  SetLength(A, C.VHeads); SetLength(B, C.VHeads);
  Projection.Linear(FWeights.QKV, @Input[0], @Mixed[0], C.Hidden, Length(Mixed));
  Projection.Linear(FWeights.Z, @Input[0], @Z[0], C.Hidden, ValueDim);
  Projection.Linear(FWeights.A, @Input[0], @A[0], C.Hidden, C.VHeads);
  Projection.Linear(FWeights.B, @Input[0], @B[0], C.Hidden, C.VHeads);
  for var Ch := 0 to High(Mixed) do
  begin
    var Base := Ch*C.ConvSize;
    for var J := 0 to C.ConvSize-2 do ConvState[Base+J] := ConvState[Base+J+1];
    ConvState[Base+C.ConvSize-1] := Mixed[Ch];
    var V: Single := 0;
    for var J := 0 to C.ConvSize-1 do V := V+ConvState[Base+J]*FConv[Base+J];
    Mixed[Ch] := V*Sigmoid(V);
  end;
  for var H := 0 to 2*C.KHeads-1 do
  begin
    var Sum: Double := 0;
    for var J := 0 to C.KDim-1 do Sum := Sum+Sqr(Mixed[H*C.KDim+J]);
    var Inv: Single := 1/Sqrt(Sum+1E-6);
    for var J := 0 to C.KDim-1 do Mixed[H*C.KDim+J] := Mixed[H*C.KDim+J]*Inv;
  end;
  SetLength(Values, ValueDim); SetLength(Memory, C.VDim); SetLength(DeltaValue, C.VDim);
  for var H := 0 to C.VHeads-1 do
  begin
    FillChar(Memory[0], C.VDim*4, 0);
    var X := A[H]+FDtBias[H];
    var Softplus := Max(X,0)+Ln(1+Exp(-Abs(X)));
    var Coefficient: Single := FALog[H];
    if not C.GGUF then Coefficient := -Exp(Coefficient);
    var Decay: Single := Exp(Coefficient*Softplus);
    var Beta := Sigmoid(B[H]);
    var KH := H div (C.VHeads div C.KHeads);
    if C.GGUF then KH := H mod C.KHeads;
    for var J := 0 to C.KDim-1 do
      InferenceMemory(C.VDim, @State[(H*C.KDim+J)*C.VDim], @Memory[0], Decay, Mixed[KeyDim+KH*C.KDim+J]);
    for var J := 0 to C.VDim-1 do DeltaValue[J] := (Mixed[2*KeyDim+H*C.VDim+J]-Memory[J])*Beta;
    for var J := 0 to C.KDim-1 do
      InferenceUpdate(C.VDim, @State[(H*C.KDim+J)*C.VDim], @DeltaValue[0], @Values[H*C.VDim],
        Decay, Mixed[KeyDim+KH*C.KDim+J], Mixed[KH*C.KDim+J]);
    var Sum: Double := 0;
    for var J := 0 to C.VDim-1 do
    begin Values[H*C.VDim+J] := Values[H*C.VDim+J]/Sqrt(C.KDim); Sum := Sum+Sqr(Values[H*C.VDim+J]) end;
    var Inv: Single := 1/Sqrt(Sum/C.VDim+C.Epsilon);
    for var J := 0 to C.VDim-1 do Values[H*C.VDim+J] := Values[H*C.VDim+J]*Inv*FGatedNorm[J]*Z[H*C.VDim+J]*Sigmoid(Z[H*C.VDim+J]);
  end;
  SetLength(Output, C.Hidden);
  Projection.Linear(FWeights.O, @Values[0], @Output[0], ValueDim, C.Hidden);
end;

procedure TQwenCPULayer.Attention(const Input: TArray<Single>; const Position: Integer;
  const Positions: TArray<LongInt>; const Projection: TCPUProjection; out Output: TArray<Single>);
var
  Raw, Q, K, V, Normalized, Values, Scores: TArray<Single>;


begin
  var C := FConfig; var QDim := C.Heads*C.HeadDim; var KVDim := C.KVHeads*C.HeadDim;
  SetLength(Raw, 2*QDim); SetLength(Q, QDim); SetLength(K, KVDim); SetLength(V, KVDim);
  Projection.Linear(FWeights.Q, @Input[0], @Raw[0], C.Hidden, Length(Raw));
  Projection.Linear(FWeights.K, @Input[0], @K[0], C.Hidden, KVDim);
  Projection.Linear(FWeights.V, @Input[0], @V[0], C.Hidden, KVDim);
  for var H := 0 to C.Heads-1 do Move(Raw[H*2*C.HeadDim], Q[H*C.HeadDim], C.HeadDim*4);
  Normalize(Q, FQNorm, C.HeadDim, Normalized); Q := Normalized;
  Normalize(K, FKNorm, C.HeadDim, Normalized); K := Normalized;
  QwenCPUApplyRoPE(Q, C.Heads, C.HeadDim, C.RotaryDim, Position, C.Theta, Positions);
  QwenCPUApplyRoPE(K, C.KVHeads, C.HeadDim, C.RotaryDim, Position, C.Theta, Positions);
  // Matches Vulkan: transposed keys followed by head-major values.
  for var H := 0 to C.KVHeads-1 do
    for var J := 0 to C.HeadDim-1 do
    begin
      State[(H*C.HeadDim+J)*C.Capacity+Position] := K[H*C.HeadDim+J];
      State[C.Capacity*KVDim+(H*C.Capacity+Position)*C.HeadDim+J] := V[H*C.HeadDim+J];
    end;
  SetLength(Values, QDim); SetLength(Scores, Position+1);
  for var H := 0 to C.Heads-1 do
  begin
    var KH := H div (C.Heads div C.KVHeads); var Maximum: Single := -MaxSingle;
    FillChar(Scores[0], Length(Scores)*4, 0);
    // K is transposed: vectorize across positions to read contiguous cache lines.
    for var J := 0 to C.HeadDim-1 do
      InferenceMemory(Position+1, @State[(KH*C.HeadDim+J)*C.Capacity], @Scores[0], 1, Q[H*C.HeadDim+J]);
    for var T := 0 to Position do
    begin Scores[T] := Scores[T]/Sqrt(C.HeadDim); Maximum := Max(Maximum, Scores[T]) end;
    var Denom: Double := 0;
    for var T := 0 to Position do begin Scores[T] := Exp(Scores[T]-Maximum); Denom := Denom+Scores[T] end;
    for var T := 0 to Position do
    begin
      var Scale: Single := Scores[T]/Denom;
      var Base := C.Capacity*KVDim+(KH*C.Capacity+T)*C.HeadDim;
      InferenceMemory(C.HeadDim, @State[Base], @Values[H*C.HeadDim], 1, Scale);
    end;
    for var J := 0 to C.HeadDim-1 do Values[H*C.HeadDim+J] := Values[H*C.HeadDim+J]*Sigmoid(Raw[H*2*C.HeadDim+C.HeadDim+J]);
  end;
  SetLength(Output, C.Hidden); Projection.Linear(FWeights.O, @Values[0], @Output[0], QDim, C.Hidden);
end;

procedure TQwenCPULayer.Step(var Hidden: TArray<Single>; const Position: Integer;
  const Positions: TArray<LongInt>; const Projection: TCPUProjection);
var
  Input, Output, Gate, Up: TArray<Single>;
begin
  if not CPUCurrent or (Position<0) or (Position>=FConfig.Capacity) then
    raise EInvalidOp.Create('CPU hybrid state is not synchronized');
  // Mark dirty before modifying recurrent state, including a failed projection.
  GPUCurrent := False;
  Normalize(Hidden, FInputNorm, FConfig.Hidden, Input);
  if FWeights.Linear then Delta(Input, Projection, Output)
  else Attention(Input, Position, Positions, Projection, Output);
  for var J := 0 to High(Hidden) do Hidden[J] := Hidden[J]+Output[J];
  Normalize(Hidden, FPostNorm, FConfig.Hidden, Input);
  SetLength(Gate, FConfig.Intermediate); SetLength(Up, FConfig.Intermediate);
  Projection.Linear(FWeights.Gate, @Input[0], @Gate[0], FConfig.Hidden, FConfig.Intermediate);
  Projection.Linear(FWeights.Up, @Input[0], @Up[0], FConfig.Hidden, FConfig.Intermediate);
  for var J := 0 to High(Gate) do
  begin
    var G := Gate[J];
    if G <= -20 then G := 0 else if G<20 then G := G*Sigmoid(G);
    Gate[J] := G*Up[J];
  end;
  Projection.Linear(FWeights.Down, @Gate[0], @Output[0], FConfig.Intermediate, FConfig.Hidden);
  for var J := 0 to High(Hidden) do Hidden[J] := Hidden[J]+Output[J];
end;

end.
