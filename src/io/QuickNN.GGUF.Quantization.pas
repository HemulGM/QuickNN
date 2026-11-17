unit QuickNN.GGUF.Quantization;
{$POINTERMATH ON}

{ Native GGML block layouts and scalar reference decoding.
  Format reference (GGML authors, MIT):
  https://github.com/ggml-org/ggml/blob/master/src/ggml-common.h
  https://github.com/ggml-org/ggml/blob/master/src/ggml-quants.c
  QuantBits weights are decoded in the Vulkan GEMM shared tile during inference. }

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
// CPU-only packing: FP32 scales with the original integer quants/codebooks.
function GGUFCPUBlockBytes(const Encoding: TQNNWeightEncoding): Integer;
procedure GGUFPrepareCPU(const Data: Pointer; const Encoding: TQNNWeightEncoding; const Count: NativeInt; const Output: Pointer);
procedure GGUFCPUInputSums(const Input, Output: System.PSingle; const Count: Integer);
function GGUFDotCPU(const Data: Pointer; const Encoding: TQNNWeightEncoding; const Input: System.PSingle; const Count: Integer; const InputSums: System.PSingle = nil): Single;
function GGUFTryDot(const Data: Pointer; const Encoding: TQNNWeightEncoding; const Input: System.PSingle; const Count: Integer; out Value: Single): Boolean;
procedure GGUFDecodeBlock(const Data: Pointer; const Encoding: TQNNWeightEncoding; const Output: System.PSingle);
procedure GGUFDequantize(const Data: Pointer; const Encoding: TQNNWeightEncoding; const Count: NativeInt; const Output: System.PSingle);

implementation

{$I QuickNN.GGUF.Tables.inc}
const
  IQ4_NL_VALUES: array[0..15] of Integer = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113);

type
  TGGUFDecodeParams = record
    Scale, Bias: Single;
    LowShift, HighShift, LowMask, HighMask, HighLeft, Center, Count: Integer;
  end;
  PGGUFDecodeParams = ^TGGUFDecodeParams;
var
  SignedIQ3Grid: array[0..15,0..511] of UInt32;
{$I QuickNN.GGUF.Decode.AVX2.inc}
  {$I QuickNN.GGUF.Dot.AVX2.inc}
{$I QuickNN.GGUF.Prepared.AVX2.inc}

function GGUFEncoding(const StorageType: UInt32): TQNNWeightEncoding;
begin
  case StorageType of
    16: Result := weGGUFIQ2_XXS;
    18: Result := weGGUFIQ3_XXS;
    22: Result := weGGUFIQ2_S;
    42: Result := weGGUFQ2_0;
    8: Result := weGGUFQ8_0;
    10: Result := weGGUFQ2_K;
    12: Result := weGGUFQ4_K;
    14: Result := weGGUFQ6_K;
    11: Result := weGGUFQ3_K;
    13: Result := weGGUFQ5_K;
    20: Result := weGGUFIQ4_NL;
    21: Result := weGGUFIQ3_S;
    23: Result := weGGUFIQ4_XS;
  else Result := weNative end;
end;

function GGUFBlockElements(const Encoding: TQNNWeightEncoding): Integer;
begin
  case Encoding of
    weGGUFQ2_0: Result := 64;
    weGGUFIQ2_XXS, weGGUFIQ2_S, weGGUFIQ3_XXS: Result := 256;
    weGGUFQ8_0, weGGUFIQ4_NL: Result := 32;
    weGGUFQ4_K, weGGUFQ6_K, weGGUFQ3_K, weGGUFQ5_K, weGGUFIQ3_S, weGGUFIQ4_XS, weGGUFQ2_K: Result := 256;
  else raise EArgumentException.Create('Expected a packed GGUF encoding') end;
end;

function GGUFBlockBytes(const Encoding: TQNNWeightEncoding): Integer;
begin
  case Encoding of
    weGGUFIQ3_XXS: Result := 98;
    weGGUFIQ2_XXS: Result := 66;
    weGGUFIQ2_S: Result := 82;
    weGGUFQ2_0: Result := 18;
    weGGUFQ8_0: Result := 34;
    weGGUFQ2_K: Result := 84;
    weGGUFQ4_K: Result := 144;
    weGGUFQ6_K: Result := 210;
    weGGUFQ3_K, weGGUFIQ3_S: Result := 110;
    weGGUFQ5_K: Result := 176;
    weGGUFIQ4_NL: Result := 18;
    weGGUFIQ4_XS: Result := 136;
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
  if not Weight.IsAssigned or (Weight.DataType <> dtU8) or (Weight.Offset < 0)
    or (Weight.Offset mod GGUFBlockElements(Weight.Encoding) <> 0) then
    raise EArgumentException.Create('Invalid packed GGUF pointer or slice');
  Result := PByte(Weight) - Weight.Offset + NativeInt(GGUFQuantBytes(Weight.Encoding, Weight.Offset));
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
    weGGUFQ2_0:
      Result := Half(Base) * (Integer((Base[2 + I div 4] shr ((I mod 4)*2)) and 3) - 1);
    weGGUFIQ3_XXS:
      begin
        Group := I div 32;
        var Bits: UInt32; Move(Base[66+Group*4],Bits,4);
        var Signs := IQ2_SIGNS[(Bits shr (7*((I mod 32) div 8))) and 127];
        var Grid := IQ3_XXS_GRID[Base[2+I div 4]];
        Q := (Grid shr ((I mod 4)*8)) and 255;
        if Signs and (1 shl (I mod 8)) <> 0 then Q := -Q;
        Result := (Half(Base)*(0.5+(Bits shr 28))*0.5)*Q;
      end;
    weGGUFIQ2_XXS, weGGUFIQ2_S:
      begin
        Group := I div 32; Lane := (I mod 32) div 8;
        var Grid: UInt64; var Signs, QuantBits: UInt32; var Delta: Single;
        if Encoding = weGGUFIQ2_XXS then
        begin
          Move(Base[6 + Group*8], QuantBits, 4);
          Grid := IQ2_XXS_GRID[Base[2 + Group*8 + Lane]];
          Signs := IQ2_SIGNS[(QuantBits shr (7*Lane)) and 127];
          Delta := Half(Base) * (0.5 + (QuantBits shr 28)) * 0.25;
        end
        else
        begin
          Q := Integer(Base[2 + Group*4 + Lane]) or (((Base[66 + Group] shr (Lane*2)) and 3) shl 8);
          Grid := IQ2_S_GRID[Q]; Signs := Base[34 + I div 8];
          Q := (Base[74 + Group] shr (((I mod 32) div 16)*4)) and 15;
          Delta := Half(Base) * (0.5 + Q) * 0.25;
        end;
        Q := (Grid shr ((I mod 8)*8)) and 255;
        if Signs and (UInt32(1) shl (I mod 8)) <> 0 then Q := -Q;
        Result := Delta * Q;
      end;
    weGGUFQ8_0:
      Result := Half(Base) * SignedByte(Base[2 + I]);
    weGGUFQ2_K:
      begin
        Group := I div 16;
        Q := (Base[16 + (I div 128) * 32 + I mod 32] shr (((I mod 128) div 32) * 2)) and 3;
        Result := (Half(Base + 80) * (Base[Group] and 15)) * Q
          - Half(Base + 82) * (Base[Group] shr 4);
      end;
    weGGUFQ4_K, weGGUFQ5_K:
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
        var QuantOffset := 16;
        if Encoding = weGGUFQ5_K then QuantOffset := 48;
        Q := (Base[QuantOffset + (I div 64) * 32 + I mod 32] shr ((Group and 1) * 4)) and 15;
        if Encoding = weGGUFQ5_K then Q := Q or (((Base[16 + I mod 32] shr Group) and 1) shl 4);
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
    weGGUFQ3_K:
      begin
        Group := I div 16;
        Scale := ((Base[96 + Group mod 8] shr ((Group div 8) * 4)) and 15)
          or (((Base[104 + Group mod 4] shr ((Group div 4) * 2)) and 3) shl 4);
        Q := (Base[32 + (I div 128) * 32 + I mod 32] shr (((I mod 128) div 32) * 2)) and 3;
        if (Base[I mod 32] and (1 shl (I div 32))) = 0 then Dec(Q, 4);
        Result := (Half(Base + 108) * (Scale - 32)) * Q;
      end;
    weGGUFIQ4_NL:
      begin
        Q := (Base[2 + I mod 16] shr ((I div 16) * 4)) and 15;
        Result := Half(Base) * IQ4_NL_VALUES[Q];
      end;
    weGGUFIQ4_XS:
      begin
        Group := I div 32;
        Scale := ((Base[4 + Group div 2] shr ((Group mod 2) * 4)) and 15)
          or ((((Integer(Base[2]) or (Integer(Base[3]) shl 8)) shr (Group * 2)) and 3) shl 4);
        Q := (Base[8 + Group * 16 + I mod 16] shr (((I mod 32) div 16) * 4)) and 15;
        Result := (Half(Base) * (Scale - 32)) * IQ4_NL_VALUES[Q];
      end;
    weGGUFIQ3_S:
      begin
        Group := I div 32;
        var GridIndex := Integer(Base[2 + I div 4])
          or (((Base[66 + Group] shr ((I mod 32) div 4)) and 1) shl 8);
        Q := (IQ3_S_GRID[GridIndex] shr ((I mod 4) * 8)) and 255;
        if (Base[74 + I div 8] and (1 shl (I mod 8))) <> 0 then Q := -Q;
        Scale := 1 + 2 * ((Base[106 + Group div 2] shr ((Group mod 2) * 4)) and 15);
        Result := (Half(Base) * Scale) * Q;
      end;
  else raise EArgumentException.Create('Expected packed GGUF data') end;
end;

function GGUFTryDot(const Data: Pointer; const Encoding: TQNNWeightEncoding; const Input: System.PSingle; const Count: Integer; out Value: Single): Boolean;
begin
  Result := False;
{$IF defined(CPUX64) and defined(USE_AVX2)}
  if (Count<32) or (Count mod 32<>0) or (Data=nil) or (Input=nil) then Exit;
  if not (Encoding in [weGGUFQ8_0,weGGUFIQ4_NL]) and (Count mod 256<>0) then Exit;
  Result := True;
  case Encoding of
    weGGUFQ8_0: Value := PackedDotQ8(Data, Input, Count div 32);
    weGGUFIQ4_NL: Value := PackedDotIQ4NL(Data, Input, Count div 32);
    weGGUFQ6_K: Value := PackedDotQ6K(Data, Input, Count div 256);
    weGGUFQ3_K: Value := PackedDotQ3K(Data, Input, Count div 256);
    weGGUFQ4_K: Value := PackedDotQ4K(Data, Input, Count div 256);
    weGGUFQ5_K: Value := PackedDotQ5K(Data, Input, Count div 256);
    weGGUFIQ4_XS: Value := PackedDotIQ4XS(Data, Input, Count div 256);
  else Result := False end;
{$ENDIF}
end;

procedure GGUFDecodeBlock(const Data: Pointer; const Encoding: TQNNWeightEncoding; const Output: System.PSingle);
{$IF defined(CPUX64) and defined(USE_AVX2)}
var
  Base, Low, High: PByte;
  P: TGGUFDecodeParams;
  D, DM: Single;
  Scale, Minimum, Offset, Part, Quarter, Lane, Grid, Signs: Integer;
  Quants: array[0..255] of ShortInt;
{$ENDIF}
begin
  if (Data = nil) or (Output = nil) then raise EArgumentException.Create('Invalid GGUF block destination');
  if Encoding in [weGGUFIQ2_XXS, weGGUFIQ2_S, weGGUFQ2_0, weGGUFIQ3_XXS] then
  begin
    for var I := 0 to GGUFBlockElements(Encoding)-1 do Output[I] := GGUFValue(Data, Encoding, I);
    Exit;
  end;
{$IF defined(CPUX64) and defined(USE_AVX2)}
  Base := Data; P := Default(TGGUFDecodeParams); P.Count := 32;
  case Encoding of
    weGGUFQ8_0: DecodeSigned32(Base + 2, Output, Half(Base));
    weGGUFQ2_K:
      for var I := 0 to 255 do Output[I] := GGUFValue(Data, Encoding, I);
    weGGUFQ4_K, weGGUFQ5_K:
      begin
        D := Half(Base); DM := Half(Base + 2); Offset := 16;
        if Encoding = weGGUFQ5_K then Offset := 48;
        P.LowMask := 15; P.HighMask := 1; P.HighLeft := 4;
        for var G := 0 to 7 do
        begin
          if G < 4 then begin Scale := Base[4+G] and 63; Minimum := Base[8+G] and 63 end
          else begin Scale := (Base[8+G] and 15) or ((Base[G] shr 6) shl 4);
            Minimum := (Base[8+G] shr 4) or ((Base[4+G] shr 6) shl 4) end;
          P.Scale := D * Scale; P.Bias := -(DM * Minimum);
          P.LowShift := (G and 1) * 4; P.HighShift := G;
          Low := Base + Offset + (G div 2) * 32;
          if Encoding = weGGUFQ4_K then DecodeLow4(Low, Output + G*32, @P)
          else DecodeBits(Low, Base+16, Output+G*32, @P);
        end;
      end;
    weGGUFQ6_K, weGGUFQ3_K:
      begin
        P.Count := 16;
        if Encoding = weGGUFQ6_K then begin D := Half(Base+208); P.LowMask := 15; P.HighMask := 3; P.HighLeft := 4; P.Center := 32 end
        else begin D := Half(Base+108); P.LowMask := 3; P.HighMask := 1; P.HighLeft := 2; P.Center := 4 end;
        for var G := 0 to 15 do
        begin
          Part := G div 8; Quarter := (G mod 8) div 2; Lane := (G and 1) * 16;
          if Encoding = weGGUFQ6_K then
          begin
            Scale := SignedByte(Base[192+G]);
            Low := Base + Part*64 + (Quarter and 1)*32 + Lane;
            High := Base + 128 + Part*32 + Lane;
            P.LowShift := (Quarter div 2)*4; P.HighShift := Quarter*2;
          end
          else
          begin
            Scale := (((Base[96+G mod 8] shr ((G div 8)*4)) and 15)
              or (((Base[104+G mod 4] shr ((G div 4)*2)) and 3) shl 4)) - 32;
            Low := Base + 32 + Part*32 + Lane; High := Base + Lane;
            P.LowShift := Quarter*2; P.HighShift := G div 2;
          end;
          P.Scale := D * Scale; DecodeBits(Low, High, Output+G*16, @P);
        end;
      end;
    weGGUFIQ4_NL: DecodeIQ4(Base+2, Output, Half(Base));
    weGGUFIQ4_XS:
      begin
        D := Half(Base);
        for var G := 0 to 7 do
        begin
          Scale := (((Base[4+G div 2] shr ((G and 1)*4)) and 15)
            or ((((Integer(Base[2]) or (Integer(Base[3]) shl 8)) shr (G*2)) and 3) shl 4)) - 32;
          DecodeIQ4(Base+8+G*16, Output+G*32, D*Scale);
        end;
      end;
    weGGUFIQ3_S:
      begin
        for var I := 0 to 63 do
        begin
          Grid := Base[2+I] or (((Base[66+I div 8] shr (I mod 8)) and 1) shl 8);
          Signs := (Base[74+I div 2] shr ((I and 1)*4)) and 15;
          PUInt32(@Quants[I*4])^ := SignedIQ3Grid[Signs,Grid];
        end;
        D := Half(Base);
        for var G := 0 to 7 do
        begin
          Scale := 1 + 2*((Base[106+G div 2] shr ((G and 1)*4)) and 15);
          DecodeSigned32(PByte(@Quants[G*32]), Output+G*32, D*Scale);
        end;
      end;
  else raise EArgumentException.Create('Expected packed GGUF block') end;
{$ELSE}
  for var I := 0 to GGUFBlockElements(Encoding)-1 do Output[I] := GGUFValue(Data, Encoding, I);
{$ENDIF}
end;

procedure GGUFDequantize(const Data: Pointer; const Encoding: TQNNWeightEncoding; const Count: NativeInt; const Output: System.PSingle);
var
  Elements, Bytes, Offset: NativeInt;
begin
  if (Count < 0) or ((Count > 0) and ((Output = nil) or (Data = nil))) then
    raise EArgumentException.Create('Invalid GGUF output span');
  Elements := GGUFBlockElements(Encoding); Bytes := GGUFBlockBytes(Encoding); Offset := 0;
  while Count-Offset >= Elements do
  begin
    GGUFDecodeBlock(PByte(Data)+(Offset div Elements)*Bytes, Encoding, Output+Offset);
    Inc(Offset, Elements);
  end;
  for var I := Offset to Count-1 do Output[I] := GGUFValue(Data, Encoding, I);
end;

function GGUFCPUBlockBytes(const Encoding: TQNNWeightEncoding): Integer;
begin
  case Encoding of
    weGGUFIQ2_XXS, weGGUFIQ2_S, weGGUFQ2_0, weGGUFIQ3_XXS: Result := GGUFBlockElements(Encoding)*4;
    weGGUFQ8_0: Result := 36;
    weGGUFQ2_K: Result := 192;
    weGGUFQ4_K: Result := 192;
    weGGUFQ5_K: Result := 224;
    weGGUFQ6_K: Result := 256;
    weGGUFQ3_K: Result := 160;
    weGGUFIQ4_NL: Result := 20;
    weGGUFIQ4_XS: Result := 160;
    weGGUFIQ3_S: Result := 288;
  else raise EArgumentException.Create('Expected CPU GGUF encoding') end;
end;

procedure GGUFPrepareCPU(const Data: Pointer; const Encoding: TQNNWeightEncoding; const Count: NativeInt; const Output: Pointer);
var
  Base, Dest: PByte;
  Scale, Minimum: Integer;
  D, DM: Single;
begin
  var Elements := GGUFBlockElements(Encoding);
  if (Count<0) or (Count mod Elements<>0) or ((Count>0) and ((Data=nil) or (Output=nil))) then
    raise EArgumentException.Create('Invalid CPU GGUF packing span');
  if Encoding in [weGGUFIQ2_XXS, weGGUFIQ2_S, weGGUFQ2_0, weGGUFIQ3_XXS] then
  begin GGUFDequantize(Data, Encoding, Count, Output); Exit end;
  Base := Data; Dest := Output;
  var PackedBytes := GGUFBlockBytes(Encoding); var PreparedBytes := GGUFCPUBlockBytes(Encoding);
  for var Block := 0 to Count div Elements-1 do
  begin
    case Encoding of
      weGGUFQ2_K:
        begin
          D := Half(Base + 80); DM := Half(Base + 82);
          for var G := 0 to 15 do
          begin
            System.PSingle(Dest)[G] := D * (Base[G] and 15);
            System.PSingle(Dest)[16 + G] := DM * (Base[G] shr 4);
          end;
          Move(Base[16], Dest[128], 64);
        end;
      weGGUFQ8_0, weGGUFIQ4_NL:
        begin
          System.PSingle(Dest)^ := Half(Base);
          Move(Base[2], Dest[4], PackedBytes-2);
        end;
      weGGUFQ4_K, weGGUFQ5_K:
        begin
          D := Half(Base); DM := Half(Base+2);
          for var G := 0 to 7 do
          begin
            if G<4 then begin Scale := Base[4+G] and 63; Minimum := Base[8+G] and 63 end
            else begin Scale := (Base[8+G] and 15) or ((Base[G] shr 6) shl 4);
              Minimum := (Base[8+G] shr 4) or ((Base[4+G] shr 6) shl 4) end;
            System.PSingle(Dest)[G] := D*Scale; System.PSingle(Dest)[8+G] := DM*Minimum;
          end;
          Move(Base[16], Dest[64], PackedBytes-16);
        end;
      weGGUFQ6_K:
        begin
          D := Half(Base+208);
          for var G := 0 to 15 do System.PSingle(Dest)[G] := D*SignedByte(Base[192+G]);
          Move(Base[0], Dest[64], 192);
        end;
      weGGUFQ3_K:
        begin
          D := Half(Base+108);
          for var G := 0 to 15 do
          begin
            Scale := (((Base[96+G mod 8] shr ((G div 8)*4)) and 15)
              or (((Base[104+G mod 4] shr ((G div 4)*2)) and 3) shl 4))-32;
            System.PSingle(Dest)[G] := D*Scale;
          end;
          Move(Base[0], Dest[64], 96);
        end;
      weGGUFIQ4_XS:
        begin
          D := Half(Base);
          for var G := 0 to 7 do
          begin
            Scale := (((Base[4+G div 2] shr ((G and 1)*4)) and 15)
              or ((((Integer(Base[2]) or (Integer(Base[3]) shl 8)) shr (G*2)) and 3) shl 4))-32;
            System.PSingle(Dest)[G] := D*Scale;
          end;
          Move(Base[8], Dest[32], 128);
        end;
      weGGUFIQ3_S:
        begin
          D := Half(Base);
          for var G := 0 to 7 do
            System.PSingle(Dest)[G] := D*(1+2*((Base[106+G div 2] shr ((G and 1)*4)) and 15));
          for var I := 0 to 63 do
          begin
            var Grid := Base[2+I] or (((Base[66+I div 8] shr (I mod 8)) and 1) shl 8);
            var Signs := (Base[74+I div 2] shr ((I and 1)*4)) and 15;
            PUInt32(Dest+32+I*4)^ := SignedIQ3Grid[Signs,Grid];
          end;
        end;
    end;
    Inc(Base,PackedBytes); Inc(Dest,PreparedBytes);
  end;
end;

{$IF not (defined(CPUX64) and defined(USE_AVX2))}
function CPUValue(const Base: PByte; const Encoding: TQNNWeightEncoding; const I: Integer): Single;
var Q, G, Lane, Quarter, Part: Integer;
begin
  G := I div 32;
  case Encoding of
    weGGUFQ8_0: Exit(System.PSingle(Base)^*SignedByte(Base[4+I]));
    weGGUFIQ4_NL: Exit(System.PSingle(Base)^*IQ4_NL_VALUES[(Base[4+I mod 16] shr ((I div 16)*4)) and 15]);
    weGGUFQ4_K, weGGUFQ5_K:
      begin
        var Offset := 64;
        if Encoding=weGGUFQ5_K then Offset := 96;
        Q := (Base[Offset+(I div 64)*32+I mod 32] shr ((G and 1)*4)) and 15;
        if Encoding=weGGUFQ5_K then Q := Q or (((Base[64+I mod 32] shr G) and 1) shl 4);
        Exit(System.PSingle(Base)[G]*Q-System.PSingle(Base)[8+G]);
      end;
    weGGUFQ6_K:
      begin
        G := I div 16; Part := I div 128; Quarter := (I mod 128) div 32; Lane := I mod 32;
        Q := ((Base[64+Part*64+(Quarter and 1)*32+Lane] shr ((Quarter div 2)*4)) and 15)
          or (((Base[192+Part*32+Lane] shr (Quarter*2)) and 3) shl 4);
        Exit(System.PSingle(Base)[G]*(Q-32));
      end;
    weGGUFQ3_K:
      begin
        G := I div 16; Quarter := (I mod 128) div 32; Lane := I mod 32;
        Q := ((Base[96+(I div 128)*32+Lane] shr (Quarter*2)) and 3)
          or (((Base[64+Lane] shr (I div 32)) and 1) shl 2);
        Exit(System.PSingle(Base)[G]*(Q-4));
      end;
    weGGUFIQ4_XS:
      begin
        Q := (Base[32+G*16+I mod 16] shr (((I mod 32) div 16)*4)) and 15;
        Exit(System.PSingle(Base)[G]*IQ4_NL_VALUES[Q]);
      end;
    weGGUFIQ3_S: Exit(System.PSingle(Base)[G]*SignedByte(Base[32+I]));
  end;
  raise EArgumentException.Create('Expected CPU GGUF encoding');
end;
{$ENDIF}

procedure GGUFCPUInputSums(const Input, Output: System.PSingle; const Count: Integer);
begin
  if (Count<32) or (Count mod 32<>0) or (Input=nil) or (Output=nil) then
    raise EArgumentException.Create('Invalid CPU GGUF input sum span');
{$IF defined(CPUX64) and defined(USE_AVX2)}
  PreparedInputSums(Input,Output,Count div 32);
{$ELSE}
  for var G := 0 to Count div 32-1 do
    for var J := 0 to 7 do Output[G*8+J] := (Input[G*32+J]+Input[G*32+J+8])+(Input[G*32+J+16]+Input[G*32+J+24]);
{$ENDIF}
end;

function GGUFDotCPU(const Data: Pointer; const Encoding: TQNNWeightEncoding; const Input: System.PSingle; const Count: Integer; const InputSums: System.PSingle): Single;
{$IF defined(CPUX64) and defined(USE_AVX2)}
var Sums: TArray<Single>; SumPointer: System.PSingle;
{$ENDIF}
begin
  var Elements := GGUFBlockElements(Encoding);
  if (Count<1) or (Count mod Elements<>0) or (Data=nil) or (Input=nil) then
    raise EArgumentException.Create('Invalid prepared CPU GGUF dot span');
  if Encoding in [weGGUFIQ2_XXS, weGGUFIQ2_S, weGGUFQ2_0, weGGUFIQ3_XXS] then
  begin
    var Sum: Double := 0;
    for var I := 0 to Count-1 do Sum := Sum + Double(System.PSingle(Data)[I])*Input[I];
    Exit(Sum);
  end;
  if Encoding = weGGUFQ2_K then
  begin
    var Sum: Double := 0;
    for var I := 0 to Count - 1 do
    begin
      var Base := PByte(Data) + (I div 256) * 192;
      var J := I mod 256; var G := J div 16;
      var Q := (Base[128 + (J div 128) * 32 + J mod 32] shr (((J mod 128) div 32) * 2)) and 3;
      Sum := Sum + Double(Input[I]) * (System.PSingle(Base)[G] * Q - System.PSingle(Base)[16 + G]);
    end;
    Exit(Sum);
  end;
{$IF defined(CPUX64) and defined(USE_AVX2)}
  SumPointer := InputSums;
  if (Encoding in [weGGUFQ4_K,weGGUFQ5_K]) and (SumPointer=nil) then
  begin
    SetLength(Sums,Count div 4); SumPointer := @Sums[0];
    GGUFCPUInputSums(Input,SumPointer,Count);
  end;
  case Encoding of
    weGGUFQ8_0: Result := PreparedDotQ8(Data,Input,Count div 32);
    weGGUFQ4_K: Result := PreparedDotQ4K(Data,Input,Count div 256,SumPointer);
    weGGUFQ5_K: Result := PreparedDotQ5K(Data,Input,Count div 256,SumPointer);
    weGGUFQ6_K: Result := PreparedDotQ6K(Data,Input,Count div 256);
    weGGUFQ3_K: Result := PreparedDotQ3K(Data,Input,Count div 256);
    weGGUFIQ4_NL: Result := PreparedDotIQ4NL(Data,Input,Count div 32);
    weGGUFIQ4_XS: Result := PreparedDotIQ4XS(Data,Input,Count div 256);
    weGGUFIQ3_S: Result := PreparedDotIQ3S(Data,Input,Count div 256);
  else raise EArgumentException.Create('Expected CPU GGUF encoding') end;
{$ELSE}
  Result := 0; var Bytes := GGUFCPUBlockBytes(Encoding);
  for var I := 0 to Count-1 do Result := Result+Input[I]*CPUValue(PByte(Data)+(I div Elements)*Bytes,Encoding,I mod Elements);
{$ENDIF}
end;

procedure InitializeSignedGrid;
begin
  for var Signs := 0 to 15 do
    for var Grid := 0 to 511 do
    begin
      var Value: UInt32 := 0;
      for var I := 0 to 3 do
      begin
        var Q := Integer((IQ3_S_GRID[Grid] shr (I*8)) and 255);
        if Signs and (1 shl I) <> 0 then Q := -Q;
        Value := Value or (UInt32(Q and 255) shl (I*8));
      end;
      SignedIQ3Grid[Signs,Grid] := Value;
    end;
end;

initialization
  InitializeSignedGrid;
end.
