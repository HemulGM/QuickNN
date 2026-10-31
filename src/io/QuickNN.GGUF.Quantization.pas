unit QuickNN.GGUF.Quantization;
{$POINTERMATH ON}

{ Native GGML block layouts and scalar reference decoding.
  Format reference (GGML authors, MIT):
  https://github.com/ggml-org/ggml/blob/master/src/ggml-common.h
  https://github.com/ggml-org/ggml/blob/master/src/ggml-quants.c
  Packed weights are decoded in the Vulkan GEMM shared tile during inference. }

interface
uses
  System.SysUtils, QuickNN.Common;

function GGUFEncoding(const StorageType: UInt32): TQNNWeightEncoding;
function GGUFBlockElements(const Encoding: TQNNWeightEncoding): Integer;
function GGUFBlockBytes(const Encoding: TQNNWeightEncoding): Integer;
function GGUFQuantBytes(const Encoding: TQNNWeightEncoding; const Elements: Int64): UInt64;
function GGUFWeightBytes(const Weight: TMemoryBlock; const Rows, Columns: Integer): UInt64;
function GGUFWeightPointer(const Weight: TMemoryBlock): Pointer;
function GGUFValue(const Data: Pointer; const Encoding: TQNNWeightEncoding; const Index: NativeInt): Single;
procedure GGUFDequantize(const Data: Pointer; const Encoding: TQNNWeightEncoding; const Count: NativeInt; const Output: System.PSingle);

implementation

function GGUFEncoding(const StorageType: UInt32): TQNNWeightEncoding;
begin
  case StorageType of
    8: Result := weGGUFQ8_0;
    12: Result := weGGUFQ4_K;
    14: Result := weGGUFQ6_K;
  else Result := weNative end;
end;

function GGUFBlockElements(const Encoding: TQNNWeightEncoding): Integer;
begin
  case Encoding of
    weGGUFQ8_0: Result := 32;
    weGGUFQ4_K, weGGUFQ6_K: Result := 256;
  else raise EArgumentException.Create('Expected a packed GGUF encoding') end;
end;

function GGUFBlockBytes(const Encoding: TQNNWeightEncoding): Integer;
begin
  case Encoding of
    weGGUFQ8_0: Result := 34;
    weGGUFQ4_K: Result := 144;
    weGGUFQ6_K: Result := 210;
  else raise EArgumentException.Create('Expected a packed GGUF encoding') end;
end;

function GGUFQuantBytes(const Encoding: TQNNWeightEncoding; const Elements: Int64): UInt64;
begin
  var Block := GGUFBlockElements(Encoding);
  if (Elements < 0) or (Elements mod Block <> 0) then
    raise EArgumentException.Create('GGUF element count is not a complete block');
  if UInt64(Elements div Block) > UInt64(High(Int64)) div UInt64(GGUFBlockBytes(Encoding)) then
    raise EArgumentException.Create('GGUF compressed byte size overflow');
  Result := UInt64(Elements div Block) * UInt64(GGUFBlockBytes(Encoding));
end;

function GGUFWeightBytes(const Weight: TMemoryBlock; const Rows, Columns: Integer): UInt64;
begin
  if (Weight.Encoding = weNative) or (Weight.DataType <> dtU8) or (Rows < 1) or (Columns < 1)
    or (Columns mod GGUFBlockElements(Weight.Encoding) <> 0)
    or (Weight.Offset < 0) or (Weight.Offset mod GGUFBlockElements(Weight.Encoding) <> 0)
    or (Int64(Rows) * Columns > Int64(Weight.Size) - Weight.Offset) then
    raise EArgumentException.Create('Invalid packed GGUF matrix span');
  Result := GGUFQuantBytes(Weight.Encoding, Int64(Rows) * Columns);
end;

function GGUFWeightPointer(const Weight: TMemoryBlock): Pointer;
begin
  if (Weight.DataPtr = nil) or (Weight.Offset < 0)
    or (Weight.Offset mod GGUFBlockElements(Weight.Encoding) <> 0) then
    raise EArgumentException.Create('Invalid packed GGUF pointer or slice');
  Result := PByte(Weight.DataPtr) + NativeInt(GGUFQuantBytes(Weight.Encoding, Weight.Offset));
end;

function Half(const Data: PByte): Single;
begin
  var Bits: FP16;
  Move(Data^, Bits, SizeOf(Bits));
  FP16ToSingle(1, @Bits, @Result);
end;

function SignedByte(const Value: Byte): Integer;
begin
  Result := Value;
  if Result >= 128 then Dec(Result, 256);
end;

function GGUFValue(const Data: Pointer; const Encoding: TQNNWeightEncoding; const Index: NativeInt): Single;
var
  Base: PByte;
  I, Group, Scale, Minimum, Q, LowIndex, HighIndex, Lane: Integer;
begin
  if (Data = nil) or (Index < 0) then raise EArgumentException.Create('Invalid GGUF value span');
  var Block := GGUFBlockElements(Encoding);
  Base := PByte(Data) + (Index div Block) * GGUFBlockBytes(Encoding);
  I := Index mod Block;
  case Encoding of
    weGGUFQ8_0:
      Result := Half(Base) * SignedByte(Base[2 + I]);
    weGGUFQ4_K:
      begin
        Group := I div 32;
        if Group < 4 then
        begin
          Scale := Base[4 + Group] and 63;
          Minimum := Base[8 + Group] and 63;
        end
        else
        begin
          Scale := (Base[8 + Group] and 15) or ((Base[Group] shr 6) shl 4);
          Minimum := (Base[8 + Group] shr 4) or ((Base[4 + Group] shr 6) shl 4);
        end;
        Q := (Base[16 + (I div 64) * 32 + I mod 32] shr ((Group and 1) * 4)) and 15;
        Result := (Half(Base) * Scale) * Q - Half(Base + 2) * Minimum;
      end;
    weGGUFQ6_K:
      begin
        var Part := I div 128;
        var Quarter := (I mod 128) div 32;
        Lane := I mod 32;
        LowIndex := Part * 64 + (Quarter and 1) * 32 + Lane;
        HighIndex := 128 + Part * 32 + Lane;
        Q := ((Base[LowIndex] shr ((Quarter div 2) * 4)) and 15)
          or (((Base[HighIndex] shr (Quarter * 2)) and 3) shl 4);
        Scale := SignedByte(Base[192 + I div 16]);
        Result := (Half(Base + 208) * Scale) * (Q - 32);
      end;
  else raise EArgumentException.Create('Expected packed GGUF data') end;
end;

procedure GGUFDequantize(const Data: Pointer; const Encoding: TQNNWeightEncoding; const Count: NativeInt; const Output: System.PSingle);
begin
  if (Count < 0) or ((Count > 0) and (Output = nil)) then raise EArgumentException.Create('Invalid GGUF output span');
  for var I: NativeInt := 0 to Count - 1 do Output[I] := GGUFValue(Data, Encoding, I);
end;

end.
