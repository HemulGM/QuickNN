unit QuickNN.GGUF;

{ Native GGUF v2/v3 reader. Layout follows the GGML GGUF specification:
  https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
  No GGML runtime or external binary is required. }

interface

uses
  System.SysUtils, System.Classes;

const
  GGUF_MAGIC = $46554747;

type
  EGGUFError = class(Exception);

  TGGUFTensor = record
    Name: string;
    Shape: TArray<Int64>; // Native row-major order; reversed from GGML ne[].
    StorageType: UInt32;
    Offset, ByteSize: Int64;
  end;

  TGGUFHeader = record
    Version, Alignment: UInt32;
    Architecture: string;
    DataOffset: Int64;
    Tensors: TArray<TGGUFTensor>;
    class function Read(const Stream: TStream): TGGUFHeader; static;
    class function ReadFile(const FileName: string): TGGUFHeader; static;
  end;

implementation

uses
  System.Generics.Collections, System.Generics.Defaults, QuickNN.Common, QuickNN.GGUF.Quantization;

const
  MAX_HEADER_BYTES = 100000000;
  MAX_ITEMS = 1000000;
  MAX_STRING_BYTES = 16 * 1024 * 1024;

type
  TGGUFReader = class
  private
    FStream: TStream;
    FStart: Int64;
    procedure Require(const Size: Int64);
    function U32: UInt32;
    function U64: UInt64;
    function ReadString: string;
    procedure SkipValue(const Kind: UInt32; const Depth: Integer = 0);
  public
    constructor Create(const Stream: TStream);
    function Read: TGGUFHeader;
  end;

constructor TGGUFReader.Create(const Stream: TStream);
begin
  inherited Create;
  FStream := Stream;
  FStart := Stream.Position;
end;

procedure TGGUFReader.Require(const Size: Int64);
begin
  if (Size < 0) or (FStream.Position < FStart)
    or (Size > FStream.Size - FStream.Position)
    or (Size > MAX_HEADER_BYTES - (FStream.Position - FStart)) then
    raise EGGUFError.Create('Truncated or oversized GGUF header');
end;

function TGGUFReader.U32: UInt32;
begin
  Require(4);
  FStream.ReadBuffer(Result, 4);
end;

function TGGUFReader.U64: UInt64;
begin
  Require(8);
  FStream.ReadBuffer(Result, 8);
end;

function TGGUFReader.ReadString: string;
begin
  var Size := U64;
  if Size > MAX_STRING_BYTES then
    raise EGGUFError.Create('Oversized GGUF string');
  Require(Size);
  var Bytes: TBytes;
  SetLength(Bytes, Size);
  if Size > 0 then FStream.ReadBuffer(Bytes[0], Size);
  try
    Result := TEncoding.UTF8.GetString(Bytes);
  except
    on E: EEncodingError do raise EGGUFError.Create('Invalid UTF-8 in GGUF header');
  end;
  var Encoded := TEncoding.UTF8.GetBytes(Result);
  if (Length(Encoded) <> Length(Bytes)) or ((Size > 0) and not CompareMem(@Bytes[0], @Encoded[0], Size)) then
    raise EGGUFError.Create('Invalid UTF-8 in GGUF header');
end;

procedure TGGUFReader.SkipValue(const Kind: UInt32; const Depth: Integer);
var
  Size: Int64;
begin
  Size := 0;
  if Depth > 16 then raise EGGUFError.Create('Excessive GGUF metadata nesting');
  case Kind of
    0, 1, 7: Size := 1;
    2, 3: Size := 2;
    4, 5, 6: Size := 4;
    10, 11, 12: Size := 8;
    8: begin ReadString; Exit end;
    9: begin
      var ElementType := U32;
      var Count := U64;
      if Count > MAX_ITEMS then raise EGGUFError.Create('Oversized GGUF metadata array');
      if Count > 0 then
        for var I: UInt64 := 0 to Count - 1 do SkipValue(ElementType, Depth + 1)
      else if ElementType > 12 then
        raise EGGUFError.Create('Unknown GGUF array element type');
      Exit;
    end;
  else
    raise EGGUFError.CreateFmt('Unknown GGUF metadata type %d', [Kind]);
  end;
  Require(Size);
  FStream.Position := FStream.Position + Size;
end;

function TGGUFReader.Read: TGGUFHeader;
begin
  Result := Default(TGGUFHeader);
  if U32 <> GGUF_MAGIC then raise EGGUFError.Create('Invalid GGUF magic (only little-endian GGUF is supported)');
  Result.Version := U32;
  if not (Result.Version in [2, 3]) then raise EGGUFError.CreateFmt('Unsupported GGUF version %d', [Result.Version]);
  var TensorCount := U64;
  var MetadataCount := U64;
  if (TensorCount > MAX_ITEMS) or (MetadataCount > MAX_ITEMS) then
    raise EGGUFError.Create('Too many GGUF tensors or metadata entries');
  Result.Alignment := 32;
  var Names := TDictionary<string, Boolean>.Create;
  try
    if MetadataCount > 0 then
      for var I: UInt64 := 0 to MetadataCount - 1 do
      begin
        var Key := ReadString;
        if (Key = '') or Names.ContainsKey(Key) then raise EGGUFError.Create('Empty or duplicate GGUF metadata key');
        Names.Add(Key, True);
        var Kind := U32;
        if Key = 'general.architecture' then
        begin
          if Kind <> 8 then raise EGGUFError.Create('GGUF architecture must be a string');
          Result.Architecture := ReadString;
        end
        else if Key = 'general.alignment' then
        begin
          if Kind <> 4 then raise EGGUFError.Create('GGUF alignment must be UINT32');
          Result.Alignment := U32;
          if (Result.Alignment = 0) or (Result.Alignment > 65536)
            or (Result.Alignment and (Result.Alignment - 1) <> 0) then
            raise EGGUFError.Create('Invalid GGUF alignment');
        end
        else SkipValue(Kind);
      end;
    Names.Clear;
    SetLength(Result.Tensors, TensorCount);
    for var I := 0 to High(Result.Tensors) do
    begin
      var Tensor: TGGUFTensor;
      Tensor := Default(TGGUFTensor);
      Tensor.Name := ReadString;
      if (Tensor.Name = '') or Names.ContainsKey(Tensor.Name) then raise EGGUFError.Create('Empty or duplicate GGUF tensor name');
      Names.Add(Tensor.Name, True);
      var Dims := U32;
      if (Dims < 1) or (Dims > 4) then raise EGGUFError.Create('Invalid GGUF tensor rank');
      SetLength(Tensor.Shape, Dims);
      var Elements: Int64 := 1;
      for var J := 0 to Integer(Dims) - 1 do
      begin
        var Dimension := U64;
        if (Dimension = 0) or (Dimension > UInt64(High(Int64)))
          or (Elements > High(Int64) div Int64(Dimension)) then
          raise EGGUFError.Create('Invalid or overflowing GGUF tensor shape');
        Tensor.Shape[Integer(Dims) - 1 - J] := Dimension;
        Elements := Elements * Int64(Dimension);
      end;
      Tensor.StorageType := U32;
      var ElementBytes: Integer;
      var Encoding := GGUFEncoding(Tensor.StorageType);
      case Tensor.StorageType of
        0: ElementBytes := 4; // GGML_TYPE_F32
        1, 30: ElementBytes := 2; // F16 / BF16
        8, 12, 14: ElementBytes := 0; // Q8_0, Q4_K, Q6_K (Q4_K_M mixture)
      else
        raise EGGUFError.CreateFmt('GGUF tensor "%s": storage type %d is not supported; use BF16/F16/F32, Q8_0 or Q4_K_M weights', [Tensor.Name, Tensor.StorageType]);
      end;
      if Encoding <> weNative then
      begin
        if Tensor.Shape[High(Tensor.Shape)] mod GGUFBlockElements(Encoding) <> 0 then
          raise EGGUFError.Create('GGUF row is not a complete quantization block: ' + Tensor.Name);
        if UInt64(Elements div GGUFBlockElements(Encoding)) > UInt64(High(Int64)) div UInt64(GGUFBlockBytes(Encoding)) then
          raise EGGUFError.Create('GGUF compressed byte size overflow');
        Tensor.ByteSize := GGUFQuantBytes(Encoding, Elements);
      end
      else
      begin
        if Elements > High(Int64) div ElementBytes then raise EGGUFError.Create('GGUF tensor byte size overflow');
        Tensor.ByteSize := Elements * ElementBytes;
      end;
      var Offset := U64;
      if (Offset > UInt64(High(Int64))) or (Offset mod Result.Alignment <> 0) then
        raise EGGUFError.Create('Invalid GGUF tensor offset');
      Tensor.Offset := Offset;
      Result.Tensors[I] := Tensor;
    end;
  finally Names.Free end;
  var Position := FStream.Position - FStart;
  Result.DataOffset := FStart + ((Position + Result.Alignment - 1) div Result.Alignment) * Result.Alignment;
  if Result.DataOffset > FStream.Size then raise EGGUFError.Create('Truncated GGUF alignment padding');
  // Padding between tensors is permitted; overlaps and truncated payloads are not.
  var Sorted := Copy(Result.Tensors);
  TArray.Sort<TGGUFTensor>(Sorted, TComparer<TGGUFTensor>.Construct(
    function(const A, B: TGGUFTensor): Integer
    begin
      if A.Offset < B.Offset then Exit(-1);
      if A.Offset > B.Offset then Exit(1);
      Result := 0;
    end));
  var EndOffset: Int64 := 0;
  var Payload := FStream.Size - Result.DataOffset;
  for var Tensor in Sorted do
  begin
    if (Tensor.Offset < EndOffset) or (Tensor.Offset > Payload) or (Tensor.ByteSize > Payload - Tensor.Offset) then
      raise EGGUFError.Create('Overlapping or truncated GGUF tensor: ' + Tensor.Name);
    EndOffset := Tensor.Offset + Tensor.ByteSize;
  end;
end;

class function TGGUFHeader.Read(const Stream: TStream): TGGUFHeader;
begin
  var Reader := TGGUFReader.Create(Stream);
  try Result := Reader.Read finally Reader.Free end;
end;

class function TGGUFHeader.ReadFile(const FileName: string): TGGUFHeader;
begin
  var Stream := TFileStream.Create(FileName, fmOpenRead or fmShareDenyWrite);
  try Result := Read(Stream) finally Stream.Free end;
end;

end.
