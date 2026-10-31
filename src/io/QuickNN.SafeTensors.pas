unit QuickNN.SafeTensors;

{$C+} // enable assertions
{$H+}
{$POINTERMATH on}
{$T+}

{$DEFINE USE_CALLOC}

interface

uses
  System.SysUtils,
  System.Classes,
  System.JSON,
  System.Generics.Collections,
  System.Generics.Defaults
{$IF defined(MSWINDOWS)}
  ,
  Winapi.Windows
{$ELSE}
  ,
  Posix.SysMman
{$ENDIF}
  ,
  QuickNN.Common;

const
  SAFETENSORS_MAX_TENSORS = 512;

type
  // https://github.com/huggingface/safetensors/blob/main/safetensors/src/tensor.rs

  TSafeTensorType = (
      StUNKNOWN = -1,
      /// Boolan type
      StBOOL,
      /// MXF4 (E2M1) <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>_
      StF4,
      /// MXF6 <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>_
      StF6E2M3,
      /// MXF6 <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>_
      StF6E3M2,
      /// Unsigned byte
      StU8,
      /// Signed byte
      StI8,
      /// FP8 <https://arxiv.org/pdf/2209.05433.pdf>_
      StF8E5M2,
      /// FP8 <https://arxiv.org/pdf/2209.05433.pdf>_
      StF8E4M3,
      /// F8_E8M0 <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>_
      StF8E8M0,
      /// Signed integer (16-bit)
      StI16,
      /// Unsigned integer (16-bit)
      StU16,
      /// Half-precision floating point
      StF16,
      /// Brain floating point
      StBF16,
      /// Signed integer (32-bit)
      StI32,
      /// Unsigned integer (32-bit)
      StU32,
      /// Floating point (32-bit)
      StF32,
      /// Complex (32-bit parts)
      StC64,
      /// Floating point (64-bit)
      StF64,
      /// Signed integer (64-bit)
      StI64,
      /// Unsigned integer (64-bit)
      StU64,
      /// INT4
      StI4
  );

const
  DTYPE_NAMES: array[Low(TSafeTensorType)..High(TSafeTensorType)] of AnsiString = (
      /// Boolan type
      'UNKNOWN',
      'BOOL',
      'F4', // e1m2
      'F6_E2M3',
      'F6_E3M2',
      'U8',
      'I8',
      'F8_E5M2',
      'F8_E4M3',
      'F8_E8M0',
      'I16',
      'U16',
      'F16',
      'BF16',
      'I32',
      'U32',
      'F32',
      'C64',
      'F64',
      'I64',
      'U64',
      'I4'
  );

  SAFETENSORTYPE_TO_DATATYPE: array[Low(TSafeTensorType)..High(TSafeTensorType)] of TQNNDataType = (
      /// Boolan type
      dtUndef, // UNKNOWEN
      dtBoolean,
      dtF4E2m1,
      dtUndef, // 'F6_E2M3',
      dtUndef, // 'F6_E3M2',
      dtU8, // 'U8',
      dtS8, // 'I8',
      dtF8E5m2, // 'F8_E5M2',
      dtF8E4m3, // 'F8_E4M3',
      dtUndef, // 'F8_E8M0',
      dtS16, // 'I16',
      dtU16, // 'U16',
      dtF16, // 'F16',
      dtBF16, // 'BF16',
      dtS32, // 'I32',
      dtUndef, // 'U32',
      dtF32, // 'F32',
      dtUndef, // 'C64',
      dtF64, // 'F64',
      dtUndef, // 'I64',
      dtUndef, // 'U64',
      dtS4 // I4
  );

type

  { TSafeTensorTypeHelper }

  TSafeTensorTypeHelper = record helper for TSafeTensorType
    function Bits: Byte;
    class function FromString(const Str: string): TSafeTensorType; static;
    function ToString(): string;
    function ToDataType(): TQNNDataType;
  end;

  PSafeTensor = ^TSafeTensor;

  { TSafeTensor }

  TSafeTensor = record
  public
    WeightIdentity: IInterface;
    Encoding: TQNNWeightEncoding;
    Name: string;
    Ndim: LongInt;
    Shape: TArray<Int64>; // array {[0..7]} of int64;
    DataOffset: Int64;
    DataSize: Int64;
    function IsBF16(): Boolean;
    function AsSingle(const UseMmap: Boolean = False): TMemoryBlock;
    function AsSinglePtr(): System.PSingle;
    function AsMappedPtr(): Pointer;
    function AsBF16(const UseMmap: Boolean = False): TMemoryBlock;
    function AsBF16Ptr: PBF16;
    function AsFP16(const UseMmap: Boolean = False): TMemoryBlock;
    function Count(): Int64;
    procedure Print();
  case DType: TSafeTensorType of
    StF16: (DataF16: PFP16);
    StBF16: (DataBF16: PBF16);
    StF32: (DataF32: System.PSingle);
    StI4: (DataI4: PInt4);
    StI8: (DataI8: PShortInt);
    StI16: (DataI16: PSmallInt);
    StI32: (DataI32: PLongint);
    StI64: (DataI64: PInt64);
    StBOOL: (DataBOOL: PBoolean);
    StUNKNOWN: (Data: Pointer);
  end;

  { TSafeTensorFile }

  ESafeTensorError = class(Exception);

  TSafeTensorFile = record
  private
    TensorIndex: TArray<Integer>;
    MappingOwner: IInterface;
    procedure OpenGGUF(const Stream: TFileStream);
  public
    Path, Architecture: string;
    IsGGUF: Boolean;
    Data: Pointer; (* mmapped file data *)
    FileSize, HeaderSize, DataSize, Offset: Int64;
    HeaderJson: UTF8String;
    Tensors: array of TSafeTensor;
    SfFile: file;
    JSON: TJSONObject;
    function TensorCount: LongInt;
    constructor Open(const Apath: string);
    procedure Close();
    function Find(const Name: string): PSafeTensor;
    function GetData(const SafeTensor: PSafeTensor; const UseMMAP: Boolean = True): TMemoryBlock; overload;
    function GetData(const TensorName: string; const UseMMAP: Boolean = True): TMemoryBlock; overload;
    procedure Print();
  end;

  TSafeTensorFiles = TArray<TSafeTensorFile>;

  { TSafetensoFilesHelper }

  TSafeTensorFilesHelper =
      record helper for
          TSafeTensorFiles
          // getData will allocate memory only if useMMap is false, result is always in F32
    function GetTensorDataMemBlock(const Name: string; const UseMMAP: Boolean): TMemoryBlock; overload;
    // function getTensorDataMemBlock(const name: string): TMemoryBlock; overload;
    function GetTensorDataMemBlockBF16(const Name: string; const UseMMAP: Boolean): TMemoryBlock; overload;

    // Linear weights retain native BF16, avoiding repeated host expansion.
    function GetLinearWeight(const Name: string; const UseMMAP: Boolean): TMemoryBlock;
    function GetTensor(const Name: string): PSafeTensor;

  end;

// Returned records share ownership of the mapping; close releases only this record.
function LoadSafeTensorShards(const Directory, IndexFile, SingleFile: string): TSafeTensorFiles;

const
  IS_LOAD_VERBOSE: Boolean = False;

implementation
uses
  System.IOUtils, QuickNN.GGUF, QuickNN.GGUF.Quantization;

type
  TSafeTensorResource = class(TInterfacedObject)
    Data: Pointer;
    Size: Int64;
    JSON: TJSONObject;
    destructor Destroy; override;
  end;

function MapFile(const Stream: TFileStream): Pointer;
{$IF defined(MSWINDOWS)}
var
  Mapping: THandle;
  ErrorCode: Cardinal;
begin
  Mapping := CreateFileMapping(Stream.Handle, nil, PAGE_READONLY, 0, 0, nil);
  if Mapping = 0 then
    RaiseLastOSError;
  try
    Result := MapViewOfFile(Mapping, FILE_MAP_READ, 0, 0, 0);
    if Result = nil then
    begin
      ErrorCode := GetLastError;
      RaiseLastOSError(ErrorCode);
    end;
  finally
    CloseHandle(Mapping);
  end;
end;
{$ELSE}
begin

  Result := Mmap(nil, Stream.Size, PROT_READ, MAP_PRIVATE, Stream.Handle, 0);

  if Result = MAP_FAILED then
    raise ESafeTensorError.Create('Unable to map safetensors file');
end;
{$ENDIF}

procedure UnmapFile(var Addr: Pointer; const Size: Int64);
begin
  if Addr = nil then
    Exit;
{$IF defined(MSWINDOWS)}
  if not UnmapViewOfFile(Addr) then
    RaiseLastOSError;
{$ELSE}
  if Munmap(Addr, Size) <> 0 then
    raise ESafeTensorError.Create('Unable to unmap safetensors file');
{$ENDIF}
  Addr := nil;
end;

destructor TSafeTensorResource.Destroy;
begin
  try
    UnmapFile(Data, Size);
  finally
    JSON.Free;
    inherited;
  end;
end;

{ TSafeTensorTypeHelper }

function TSafeTensorTypeHelper.Bits: Byte;
begin
  case Self of
    StBOOL: Result := 8;
    StF4, StI4: Result := 4;
    StF6E2M3, StF6E3M2: Result := 6;
    StU8, StI8, StF8E5M2, StF8E4M3, StF8E8M0: Result := 8;
    StI16, StU16, StF16, StBF16: Result := 16;
    StI32, StU32, StF32: Result := 32;
    StC64, StF64, StI64, StU64: Result := 64
  else
    Result := 0;
  end;
end;

class function TSafeTensorTypeHelper.FromString(const Str: string): TSafeTensorType;
var
  I: TSafeTensorType;
begin
  for I := Low(TSafeTensorType) to High(TSafeTensorType) do
    if string(DTYPE_NAMES[I]) = Str then
      Exit(I);
  Result := StUNKNOWN;
end;

function TSafeTensorTypeHelper.ToString(): string;
begin
  Result := string(DTYPE_NAMES[Self])
end;

function TSafeTensorTypeHelper.ToDataType(): TQNNDataType;
begin
  Result := SAFETENSORTYPE_TO_DATATYPE[Self]
end;

{ TSafeTensor }

function TSafeTensor.IsBF16(): Boolean;
begin
  Result := DType = StBF16;
end;
{$IFDEF USE_CALLOC}
function TSafeTensor.AsSingle(const UseMmap: Boolean): TMemoryBlock;
{$ELSE}
function TSafeTensor.AsSingle: TArray<Single>;
{$ENDIF}
begin
  if Encoding <> weNative then
  begin
    Result := TMemoryBlock.Create(Shape, Name, dtF32);
    try GGUFDequantize(Data, Encoding, Count, System.PSingle(Result))
    except Result.Free; raise end;
    Result.WeightIdentity := WeightIdentity;
    Exit;
  end;
  if not (DType in [StBF16, StF16, StF32]) then
    raise ESafeTensorError.Create('Unsupported conversion to F32: ' + Name);
  if UseMmap and (DType = StF32) then
  begin
    Result.AssignPtr(DataF32, Shape);
    Result.Name := Name;
    Result.WeightIdentity := WeightIdentity;
    Exit;
  end
  else
    Result := TMemoryBlock.Create(Shape, Name, dtF32);

  case DType of
    StF32:
    begin
      if Count > 0 then
        Move(Data^, System.PSingle(Result)^, SizeOf(Single) * Count)
    end;

    StF16:
    begin
      FP16ToSingle(Count, Data, System.PSingle(Result));
    end;

    StBF16:
    begin
      BF16ToSingle(Count, Data, System.PSingle(Result));
    end;
  end;
  Result.WeightIdentity := WeightIdentity;
end;

function TSafeTensor.AsSinglePtr(): System.PSingle;
begin
  if DType <> StF32 then
    raise ESafeTensorError.Create('Tensor is not F32: ' + Name);
  Result := Data
end;

function TSafeTensor.AsMappedPtr(): Pointer;
begin
  Result := Data
end;

function TSafeTensor.AsBF16(const UseMmap: Boolean): TMemoryBlock;
begin
  Assert(DType = StBF16, 'ERROR : Tensor type is not BF16!');
  if UseMmap then
  begin
    Result.DataType := dtBF16;
    Result.DataPtr := DataBF16;
    Result.Shape := Shape;
    Result.Size := Count();
    Result.Name := Name
  end
  else
    Result := TMemoryBlock.Create(Shape, Name, dtBF16, DataBF16);
  Result.WeightIdentity := WeightIdentity;
end;

function TSafeTensor.AsBF16Ptr: PBF16;
begin
  Assert(DType = StBF16, 'ERROR : Tensor type is not BF16!');
  Result := Data
end;

function TSafeTensor.AsFP16(const UseMmap: Boolean): TMemoryBlock;
begin
  Assert(DType = StF16, 'ERROR : Tensor type is not F16!');
  if UseMmap then
  begin
    Result.DataType := dtF16;
    Result.DataPtr := DataF16;
    Result.Shape := Shape;
    Result.Size := Count();
    Result.Name := Name
  end
  else
    Result := TMemoryBlock.Create(Shape, Name, dtF16, DataF16);
  Result.WeightIdentity := WeightIdentity;
end;

function TSafeTensor.Count(): Int64;
begin
  Result := Product(Shape)
end;

procedure TSafeTensor.Print();
var
  SShape: string;
  I: Integer;
begin
  SShape := '[';
  for I := 0 to High(Shape) do
  begin
    if I > 0 then
      SShape := SShape + ',';
    SShape := SShape + IntToStr(Shape[I]);
  end;
  WriteLn(
      Name,
      ': dtype=',
      Ord(DType),
      #9' shape=',
      SShape,
      ']'#9' offset=',
      Format('%.0n', [Currency(DataOffset)]),
      #9' size=',
      Format('%.0n', [Currency(DataSize)])
  );
end;

{ TSafeTensorFile }

function TSafeTensorFile.TensorCount: LongInt;
begin
  Result := Length(Tensors)
end;

function JSONInt64(const Value: TJSONValue): Int64;
begin
  if not (Value is TJSONNumber) then
    raise ESafeTensorError.Create('Expected an integer in safetensors header');
  if not TryStrToInt64(Value.Value, Result) then
    raise ESafeTensorError.Create('Invalid integer in safetensors header');
end;

procedure TSafeTensorFile.OpenGGUF(const Stream: TFileStream);
begin
  Stream.Position := 0;
  var Header := TGGUFHeader.Read(Stream);
  IsGGUF := True;
  Architecture := Header.Architecture;
  Offset := Header.DataOffset;
  HeaderSize := Offset;
  DataSize := FileSize - Offset;
  SetLength(Tensors, Length(Header.Tensors));
  SetLength(TensorIndex, Length(Tensors));
  for var I := 0 to High(Tensors) do
  begin
    var Source := Header.Tensors[I];
    Tensors[I].Name := Source.Name;
    Tensors[I].Shape := Source.Shape;
    Tensors[I].Ndim := Length(Source.Shape);
    Tensors[I].WeightIdentity := TInterfacedObject.Create;
    Tensors[I].Encoding := GGUFEncoding(Source.StorageType);
    case Source.StorageType of
      0: Tensors[I].DType := StF32;
      1: Tensors[I].DType := StF16;
      30: Tensors[I].DType := StBF16;
      8, 12, 14: Tensors[I].DType := StU8;
    end;
    Tensors[I].DataOffset := Offset + Source.Offset;
    Tensors[I].DataSize := Source.ByteSize;
    TensorIndex[I] := I;
  end;
  var Sorted := Tensors;
  TArray.Sort<Integer>(TensorIndex, TComparer<Integer>.Construct(
    function(const A, B: Integer): Integer
    begin Result := CompareStr(Sorted[A].Name, Sorted[B].Name) end));
  var Resource := TSafeTensorResource.Create;
  MappingOwner := Resource;
  Resource.Data := MapFile(Stream);
  Resource.Size := FileSize;
  Data := Resource.Data;
  for var I := 0 to High(Tensors) do
    Tensors[I].Data := PByte(Data) + NativeInt(Tensors[I].DataOffset);
end;

constructor TSafeTensorFile.Open(const Apath: string);
const
  MAX_HEADER_SIZE = 100000000;
var
  Stream: TFileStream;
  HeaderLength: UInt64;
  Parsed: TJSONValue;
  Item: TJSONPair;
  Descriptor: TJSONObject;
  Offsets, Dimensions: TJSONArray;
  DtypeValue: TJSONValue;
  I, J, N, BitSize: Integer;
  First, Last, Elements, TotalBits, ExpectedEnd: Int64;
  Metadata: TJSONObject;
  SortedTensors: TArray<TSafeTensor>;
  HeaderText: string;
  Resource: TSafeTensorResource;
begin
  Self := Default(TSafeTensorFile);
  Path := Apath;
  Resource := TSafeTensorResource.Create;
  MappingOwner := Resource;
  try
    Stream := TFileStream.Create(Apath, FmOpenRead or FmShareDenyWrite);
    try
      FileSize := Stream.Size;
      if (FileSize < 8) or (FileSize > High(NativeInt)) then
        raise ESafeTensorError.Create('Invalid file size or insufficient address space: ' + Apath);
      Stream.ReadBuffer(HeaderLength, SizeOf(HeaderLength));
      if UInt32(HeaderLength and $FFFFFFFF) = GGUF_MAGIC then
      begin
        OpenGGUF(Stream);
        Exit;
      end;
      if (HeaderLength < 2) or (HeaderLength > MAX_HEADER_SIZE) or (HeaderLength > UInt64(FileSize - 8)) then
        raise ESafeTensorError.Create('Invalid safetensors header length: ' + Apath);
      HeaderSize := HeaderLength;
      SetLength(HeaderJson, HeaderSize);
      Stream.ReadBuffer(HeaderJson[1], HeaderSize);
      // Parse UTF-8 bytes directly, without the Windows ANSI code page.
      HeaderText := TEncoding.UTF8.GetString(BytesOf(HeaderJson));
      if UTF8Encode(HeaderText) <> HeaderJson then
        raise ESafeTensorError.Create('Safetensors header is not valid UTF-8');
      Parsed := TJSONValue.ParseJSONValue(HeaderText);
      if not (Parsed is TJSONObject) then
      begin
        Parsed.Free;
        raise ESafeTensorError.Create('Safetensors header must be a JSON object: ' + Apath);
      end;
      JSON := TJSONObject(Parsed);
      Resource.JSON := JSON;
      Offset := 8 + HeaderSize;
      DataSize := FileSize - Offset;
      SetLength(Tensors, JSON.Count);
      N := 0;
      for Item in JSON do
      begin
        if Item.JsonString.Value = '__metadata__' then
        begin
          if not (Item.JsonValue is TJSONObject) then
            raise ESafeTensorError.Create('Invalid safetensors metadata');
          Metadata := TJSONObject(Item.JsonValue);
          for J := 0 to Metadata.Count - 1 do
            if not (Metadata.Pairs[J].JsonValue is TJSONString) then
              raise ESafeTensorError.Create('Safetensors metadata values must be strings');
          Continue;
        end;
        if not (Item.JsonValue is TJSONObject) then
          raise ESafeTensorError.Create('Invalid tensor descriptor: ' + Item.JsonString.Value);
        Descriptor := TJSONObject(Item.JsonValue);
        if not (Descriptor.GetValue('data_offsets') is TJSONArray)
            or not (Descriptor.GetValue('shape') is TJSONArray) then
          raise ESafeTensorError.Create('Invalid tensor offsets or shape');
        Offsets := TJSONArray(Descriptor.GetValue('data_offsets'));
        Dimensions := TJSONArray(Descriptor.GetValue('shape'));
        DtypeValue := Descriptor.GetValue('dtype');
        if (Offsets = nil) or (Offsets.Count <> 2) or (Dimensions = nil) or not (DtypeValue is TJSONString) then
          raise ESafeTensorError.Create('Missing tensor descriptor fields: ' + Item.JsonString.Value);
        Tensors[N].WeightIdentity := TInterfacedObject.Create;
        Tensors[N].Name := Item.JsonString.Value;
        Tensors[N].DType := TSafeTensorType.FromString(DtypeValue.Value);
        BitSize := Tensors[N].DType.Bits;
        if BitSize = 0 then
          raise ESafeTensorError.Create('Unknown safetensors dtype: ' + DtypeValue.Value);
        First := JSONInt64(Offsets.Items[0]);
        Last := JSONInt64(Offsets.Items[1]);
        if (First < 0) or (Last < First) or (Last > DataSize) then
          raise ESafeTensorError.Create('Tensor offsets are outside the data buffer: ' + Tensors[N].Name);
        Elements := 1;
        SetLength(Tensors[N].Shape, Dimensions.Count);
        for J := 0 to Dimensions.Count - 1 do
        begin
          Tensors[N].Shape[J] := JSONInt64(Dimensions.Items[J]);
          if Tensors[N].Shape[J] < 0 then
            raise ESafeTensorError.Create('Negative tensor dimension: ' + Tensors[N].Name);
          if (Tensors[N].Shape[J] <> 0) and (Elements > High(Int64) div Tensors[N].Shape[J]) then
            raise ESafeTensorError.Create('Tensor shape overflow: ' + Tensors[N].Name);
          Elements := Elements * Tensors[N].Shape[J];
        end;
        if Elements > High(Int64) div BitSize then
          raise ESafeTensorError.Create('Tensor byte size overflow: ' + Tensors[N].Name);
        TotalBits := Elements * BitSize;
        if (TotalBits mod 8 <> 0) or (TotalBits div 8 <> Last - First) then
          raise ESafeTensorError.Create('Tensor size does not match shape and dtype: ' + Tensors[N].Name);
        Tensors[N].Ndim := Dimensions.Count;
        Tensors[N].DataOffset := Offset + First;
        Tensors[N].DataSize := Last - First;
        Inc(N);
      end;
      SetLength(Tensors, N);
      SetLength(TensorIndex, N);
      for I := 0 to N - 1 do
        TensorIndex[I] := I;
      SortedTensors := Tensors;
      // Validate the complete buffer even if descriptors are not in offset order.
      TArray.Sort<Integer>(
          TensorIndex,
          TComparer<Integer>.Construct(
              function(const Left, Right: Integer): Integer
              begin
                if SortedTensors[Left].DataOffset < SortedTensors[Right].DataOffset then
                  Exit(-1);
                if SortedTensors[Left].DataOffset > SortedTensors[Right].DataOffset then
                  Exit(1);
                if SortedTensors[Left].DataSize < SortedTensors[Right].DataSize then
                  Exit(-1);
                if SortedTensors[Left].DataSize > SortedTensors[Right].DataSize then
                  Exit(1);
                Result := 0;
              end
          )
      );
      ExpectedEnd := Offset;
      for I in TensorIndex do
      begin
        if Tensors[I].DataOffset <> ExpectedEnd then
          raise ESafeTensorError.Create('Overlapping tensors or a gap in the data buffer');
        Inc(ExpectedEnd, Tensors[I].DataSize);
      end;
      if ExpectedEnd <> FileSize then
        raise ESafeTensorError.Create('Safetensors data buffer is incomplete or has trailing bytes');
      // Keep descriptor order and use a separate sorted index for O(log n) lookup.
      TArray.Sort<Integer>(
          TensorIndex,
          TComparer<Integer>.Construct(
              function(const Left, Right: Integer): Integer
              begin
                Result := CompareStr(SortedTensors[Left].Name, SortedTensors[Right].Name);
              end
          )
      );
      for I := 1 to N - 1 do
        if Tensors[TensorIndex[I - 1]].Name = Tensors[TensorIndex[I]].Name then
          raise ESafeTensorError.Create('Duplicate tensor name');
      Data := MapFile(Stream);
      Resource.Data := Data;
      Resource.Size := FileSize;
      for I := 0 to N - 1 do
        Tensors[I].Data := PByte(Data) + NativeInt(Tensors[I].DataOffset);
    finally
      Stream.Free;
    end;
  except
    on E: EGGUFError do
    begin Close; raise ESafeTensorError.Create(E.Message) end;
    on E: Exception do
    begin Close; raise end;
  end;
end;

procedure TSafeTensorFile.Close();
begin
  Self := Default(TSafeTensorFile);
end;

function TSafeTensorFile.Find(const Name: string): PSafeTensor;
var
  Lo, Hi, Mid, Cmp: Integer;
begin
  Lo := 0;
  Hi := High(TensorIndex);
  while Lo <= Hi do
  begin
    Mid := Lo + (Hi - Lo) div 2;
    Cmp := CompareStr(Tensors[TensorIndex[Mid]].Name, Name);
    if Cmp = 0 then
      Exit(@Tensors[TensorIndex[Mid]]);
    if Cmp < 0 then
      Lo := Mid + 1
    else
      Hi := Mid - 1;
  end;
  Result := nil;
end;

function TSafeTensorFile.GetData(const SafeTensor: PSafeTensor; const UseMMAP: Boolean): TMemoryBlock;
var
  Typ: TQNNDataType;
begin
  Result := Default(TMemoryBlock);
  if SafeTensor = nil then
    Exit;
  if Data = nil then
    raise ESafeTensorError.Create('Safetensors file is closed');
  if SafeTensor.Encoding <> weNative then
  begin
    if not UseMMAP then Exit(SafeTensor.AsSingle(False));
    Result.DataType := dtU8;
    Result.Encoding := SafeTensor.Encoding;
    Result.DataPtr := SafeTensor.Data;
    Result.Shape := SafeTensor.Shape;
    Result.Size := SafeTensor.Count;
    Result.Name := SafeTensor.Name;
    Result.WeightIdentity := SafeTensor.WeightIdentity;
    Exit;
  end;
  Typ := SafeTensor.DType.ToDataType;
  if (Typ = dtUndef) or not (DATATYPE_BITS[Typ] in [4, 8, 16, 32]) then
    raise ESafeTensorError.Create('Unsupported tensor storage type: ' + SafeTensor.DType.ToString);
  if UseMMAP then
  begin
    Result.Name := SafeTensor.Name;
    Result.DataType := Typ;
    Result.Shape := SafeTensor.Shape;
    Result.Size := SafeTensor.Count;
    Result.DataPtr := SafeTensor.Data;
  end
  else
    Result := TMemoryBlock.Create(SafeTensor.Shape, SafeTensor.Name, Typ, SafeTensor.Data);
  Result.WeightIdentity := SafeTensor.WeightIdentity;
end;

function TSafeTensorFile.GetData(const TensorName: string; const UseMMAP: Boolean): TMemoryBlock;
var
  Sf: PSafeTensor;
begin
  Result := Default(TMemoryBlock);
  Sf := Find(TensorName);
  if Assigned(Sf) then
    Exit(GetData(Sf, UseMMAP));
end;

procedure TSafeTensorFile.Print;
var
  I: Integer;
begin
  WriteLn('Safetensors file: ', Path);
  WriteLn('File size: ', Format('%.0n', [Currency(FileSize)]), ' bytes');
  WriteLn('Header size: ', Format('%.0n', [Currency(HeaderSize)]), ' bytes');
  WriteLn('Number of tensors: ', TensorCount, SLineBreak, '---------------------------', SLineBreak, SLineBreak);

  for I := 0 to High(Tensors) do
    Tensors[I].Print
end;

function LoadSafeTensorShards(const Directory, IndexFile, SingleFile: string): TSafeTensorFiles;
var
  Index: TJSONValue;
  Weights: TJSONObject;
  Names: TDictionary<string, Boolean>;
  Files: TList<string>;
  Pair: TJSONPair;
  FileName: string;
  I: Integer;
begin
  Result := nil;
  if not FileExists(TPath.Combine(Directory, IndexFile)) then
  begin
    SetLength(Result, 1);
    Result[0] := TSafeTensorFile.Open(TPath.Combine(Directory, SingleFile));
    Exit;
  end;
  Names := TDictionary<string, Boolean>.Create;
  Files := TList<string>.Create;
  try
    Index := TJSONValue.ParseJSONValue(TFile.ReadAllText(TPath.Combine(Directory, IndexFile)));
    try
      if not (Index is TJSONObject) then
        raise ESafeTensorError.Create('Invalid safetensors shard index');
      if not (TJSONObject(Index).GetValue('weight_map') is TJSONObject) then
        raise ESafeTensorError.Create('Missing weight_map in safetensors shard index');
      Weights := TJSONObject(TJSONObject(Index).GetValue('weight_map'));
      if Weights.Count = 0 then
        raise ESafeTensorError.Create('Empty safetensors shard index');
      for Pair in Weights do
      begin
        if not (Pair.JsonValue is TJSONString) or (Pair.JsonValue.Value = '') then
          raise ESafeTensorError.Create('Invalid shard filename');
        FileName := Pair.JsonValue.Value;
        if not Names.ContainsKey(FileName) then
        begin
          Names.Add(FileName, True);
          Files.Add(FileName);
        end;
      end;
    finally
      Index.Free;
    end;
    SetLength(Result, Files.Count);
    try
      for I := 0 to Files.Count - 1 do
        Result[I] := TSafeTensorFile.Open(TPath.Combine(Directory, Files[I]));
    except
      for I := 0 to High(Result) do
        Result[I].Close;
      Result := nil;
      raise;
    end;
  finally
    Files.Free;
    Names.Free;
  end;
end;

{ TSafeTensorFilesHelper }

// result will be F32
function TSafeTensorFilesHelper.GetTensorDataMemBlock(const Name: string; const UseMMAP: Boolean): TMemoryBlock;
var
  Tensor: PSafeTensor;
begin
  Tensor := GetTensor(Name);
  if Tensor = nil then
    raise ESafeTensorError.Create('Cannot find tensor "' + Name + '"');
  Result := Tensor.AsSingle(UseMMAP);
  if IS_LOAD_VERBOSE then
    Result.PrintStat;
end;

// result will be BF16
function TSafeTensorFilesHelper.GetTensorDataMemBlockBF16(const Name: string; const UseMMAP: Boolean): TMemoryBlock;
var
  Tensor: PSafeTensor;
begin
  Result := Default(TMemoryBlock);
  Tensor := GetTensor(Name);
  if Tensor = nil then
    Exit;
  case Tensor.DType of
    StBF16: Result := Tensor.AsBF16(UseMMAP);
    StF32:
    begin
      // A different output dtype always needs storage, even when mmap is allowed.
      Result := TMemoryBlock.Create(Tensor.Shape, Tensor.Name, dtBF16);
      SingleToBF16(Tensor.Count, Tensor.DataF32, Result);
      Result.WeightIdentity := Tensor.WeightIdentity;
    end;
  else
    raise ESafeTensorError.Create('Unsupported conversion to BF16: ' + Name);
  end;
end;

function TSafeTensorFilesHelper.GetLinearWeight(const Name: string; const UseMMAP: Boolean): TMemoryBlock;
var
  Tensor: PSafeTensor;
begin
  Tensor := GetTensor(Name);
  if Tensor = nil then
    raise ESafeTensorError.Create('Cannot find linear weight "' + Name + '"');
  if Tensor.Encoding <> weNative then
  begin
    if not UseMMAP then Exit(Tensor.AsSingle(False));
    Result := Default(TMemoryBlock);
    Result.DataType := dtU8;
    Result.Encoding := Tensor.Encoding;
    Result.Shape := Tensor.Shape;
    Result.Size := Tensor.Count;
    Result.DataPtr := Tensor.Data;
    Result.WeightIdentity := Tensor.WeightIdentity;
    Result.Name := Tensor.Name;
    Exit;
  end;
  if Tensor.DType = StBF16 then
    Result := Tensor.AsBF16(UseMMAP)
  else
    Result := Tensor.AsSingle(UseMMAP);
end;

function TSafeTensorFilesHelper.GetTensor(const Name: string): PSafeTensor;
var
  I: Integer;
begin
  for I := 0 to High(Self) do
  begin
    Result := TSafeTensorFile(Self[I]).Find(Name);
    if Assigned(Result) then
      Exit(Result);
  end;
  Result := nil;
  // if isConsole then writeln(ErrOutput, 'WARNING : Tensor [', name, '] not found');
end;

// procedure hello(const a:integer);begin writeln('hello', a); end;
// var sf : TSafeTensorFile;
//  ss: TArray<single>;

initialization

  // sf:= TSafeTensorFile.open('C:\development\flux2.c\FLUX.2-klein-base-4B\flux-2-klein-base-4b.safetensors');
  // ss := sf.tensors[20].asSingle;
  // sf.print ;
  // readln
end.
