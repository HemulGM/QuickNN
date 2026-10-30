unit QuickNN.Quantization.Int4;

{$POINTERMATH ON}

interface

uses
  QuickNN.Common;

const
  QNN_INT4_GROUP_SIZE = 32;
  QNN_INT4_BLOCK_BYTES = 20;

function Int4RowBytes(const Columns: Integer): Integer;
// Row-major groups: FP32 scale, then 32 offset-binary signed nibbles.
// The scale is fitted once by least squares after the initial max-abs pass.
function QuantizeInt4(const Weight: TMemoryBlock): TMemoryBlock;

implementation

uses
  System.SysUtils,
  System.Math,
  System.Threading;

function Int4RowBytes(const Columns: Integer): Integer;
var
  Bytes: Int64;
begin
  Bytes := ((Int64(Columns) + QNN_INT4_GROUP_SIZE - 1) div QNN_INT4_GROUP_SIZE) * QNN_INT4_BLOCK_BYTES;
  if (Columns < 1) or (Bytes > High(Integer)) then
    raise EArgumentOutOfRangeException.Create('INT4 matrix width');
  Result := Bytes;
end;

procedure PackInt4Row(
    const SourceF32: System.PSingle;
    const SourceBF16: PBF16;
    const Destination: PByte;
    const Row, Columns, RowBytes: Integer
);
var
  Values: array[0..QNN_INT4_GROUP_SIZE - 1] of Single;
  Maximum, Scale: Single;
  Numerator: Double;
  Denominator, Q, Count, Base: Integer;
  Bits, WordValue: UInt32;
  Block: PByte;
begin
  for var Group := 0 to (Columns - 1) div QNN_INT4_GROUP_SIZE do
  begin
    Base := Group * QNN_INT4_GROUP_SIZE;
    Count := Min(QNN_INT4_GROUP_SIZE, Columns - Base);
    Maximum := 0;
    for var J := 0 to Count - 1 do
    begin
      if SourceBF16 <> nil then
      begin
        Bits := UInt32(SourceBF16[NativeInt(Row) * Columns + Base + J]) shl 16;
        Move(Bits, Values[J], 4);
      end
      else
        Values[J] := SourceF32[NativeInt(Row) * Columns + Base + J];
      if IsNan(Values[J]) or IsInfinite(Values[J]) then
        raise EArgumentException.Create('Cannot quantize non-finite MoE weights');
      Maximum := Max(Maximum, Abs(Values[J]));
    end;
    Scale := Maximum / 7;
    if (Maximum > 0) and (Scale = 0) then
      Scale := Maximum;
    if Scale > 0 then
    begin
      Numerator := 0;
      Denominator := 0;
      for var J := 0 to Count - 1 do
      begin
        Q := EnsureRange(Integer(Round(Values[J] / Double(Scale))), -7, 7);
        Numerator := Numerator + Values[J] * Double(Q);
        Inc(Denominator, Q * Q);
      end;
      if Denominator > 0 then
        Scale := Numerator / Denominator;
    end;
    Block := Destination + NativeInt(Row) * RowBytes + Group * QNN_INT4_BLOCK_BYTES;
    Move(Scale, Block^, 4);
    for var WordIndex := 0 to 3 do
    begin
      WordValue := 0;
      for var Nibble := 0 to 7 do
      begin
        var J := WordIndex * 8 + Nibble;
        Q := 0;
        if (J < Count) and (Scale > 0) then
          Q := EnsureRange(Integer(Round(Values[J] / Double(Scale))), -7, 7);
        WordValue := WordValue or (UInt32(Q + 8) shl (Nibble * 4));
      end;
      Move(WordValue, (Block + 4 + WordIndex * 4)^, 4);
    end;
  end;
end;

function QuantizeInt4(const Weight: TMemoryBlock): TMemoryBlock;
var
  Rows, Columns, RowBytes: Integer;
  Destination: PByte;
  SourceF32: System.PSingle;
  SourceBF16: PBF16;

begin
  Result := Default(TMemoryBlock);
  if not (Weight.DataType in [dtF32, dtBF16])
      or not Weight.IsAssigned
      or (Weight.Offset <> 0)
      or (Length(Weight.Shape) <> 2)
      or (Weight.Shape[0] < 1)
      or (Weight.Shape[0] > High(Integer))
      or (Weight.Shape[1] < 1)
      or (Weight.Shape[1] > High(Integer))
      or (Weight.Size <> Weight.Shape[0] * Weight.Shape[1]) then
    raise EArgumentException.Create('INT4 conversion requires a complete native matrix');
  Rows := Weight.Shape[0];
  Columns := Weight.Shape[1];
  RowBytes := Int4RowBytes(Columns);
  if UInt64(Rows) * UInt64(RowBytes) > UInt64(High(NativeInt)) then
    raise ERangeError.Create('INT4 matrix exceeds address space');
  Result := TMemoryBlock.Create([Rows, RowBytes], Weight.Name, dtU8);
  try
    Destination := PByte(Result);
    SourceF32 := nil;
    SourceBF16 := nil;
    if Weight.DataType = dtBF16 then
      SourceBF16 := PBF16(Weight)
    else
      SourceF32 := System.PSingle(Weight);
    if (Rows >= 64) and (Weight.Size >= 32768) then
      TParallel.For(
          0,
          Rows - 1,
          procedure(Row: Integer) begin PackInt4Row(SourceF32, SourceBF16, Destination, Row, Columns, RowBytes); end
      )
    else
      for var Row := 0 to Rows - 1 do
        PackInt4Row(SourceF32, SourceBF16, Destination, Row, Columns, RowBytes);
    Result.MarkImmutable;
  except
    Result.Free;
    raise;
  end;
end;

end.
