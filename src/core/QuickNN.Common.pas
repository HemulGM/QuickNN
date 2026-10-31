unit QuickNN.Common;

{$TYPEINFO ON} // include typeinfo
{$C+} // enable assertions
{$H+}
{$POINTERMATH on}
{$T+}
{$Z4} // enums ar aligned to 4 bytes to match C API

{$DEFINE _USE_CALLOC}

interface
uses
  System.TypInfo,
  System.Generics.Collections;

const
  NUL = #0;
  SPACE = ' ';
  CARRIAGE_RETURN = #13;
  LINE_FEED = #10;
  TAB = #9;
  WHITE_SPACES = [SPACE, CARRIAGE_RETURN, LINE_FEED, TAB]; // a set of [space, line feed, carriage return, tab ]
// {$if not declared(FP16)}
type
  PPSingle = ^System.PSingle;
  BF16 = type Word;
  PBF16 = ^BF16;

  FP16 = type Word;
  PFP16 = ^FP16;
// {$endif}

{$IF not defined(USE_MULTITHREADING)}
type
  // delphi compiler
  TThreadProcNested = reference to procedure(Idx: IntPtr; Ptr: Pointer);
  TGroupProcNested =
      reference to procedure(const RangeStart, RangeEnd: IntPtr; const Params: Pointer); // must be [register]?

  TThreadProc = procedure(Idx: IntPtr; Ptr: Pointer);
  TGroupProc = procedure(const RangeStart, RangeEnd: IntPtr; const Params: Pointer);
{$ENDIF}

const
  EPSILON = 1e-6;

type
  TBlasInt = LongInt;
  TCBLASLayout = (CblasRowMajor = 101, CblasColMajor = 102);
  TCBLASOrder = TCBLASLayout;
  TCBLASTranspose = (CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113, CblasConjNoTrans = 114);

  TSingleArray = TArray<Single>;
  PFloatArray = ^TFloatArray;
  TFloatArray = array[0..MaxLongint div 32] of Single;
  TInterpolation = (INearest, ILinear, ICubic, ILanczos);

  TInt4 = -8..7;
  TInt4Array = packed array[0..MaxInt - 1] of TInt4;
  PInt4Array = ^TInt4Array;
  PInt4 = ^TInt4;

  TQNNDeviceType = (dvCPU, dvCUDA, dvVulkan, dvOpenCL, dvROCm);

  TQNNDevice = record
    DeviceType: TQNNDeviceType;
    Mem: Pointer;
  end;

  TQNNDataType = ( // matching oneDNN
      dtUndef = 0,
      dtF16 = 1,
      dtBF16 = 2,
      dtF32 = 3,
      dtS32 = 4,
      dtS8 = 5,
      dtU8 = 6,
      dtF64 = 7,
      dtBoolean = 8,
      dtF8E5m2 = 9,
      dtF8E4m3 = 10,
      dtS4 = 11,
      dtU4 = 12,
      dtE8m0 = 13,
      dtF4E2m1 = 14,
      dtF4E3m0 = 15,
      dtS16 = 16,
      dtU16 // ,
          // dtData_type_max = $7fff
  );

const
  DT_I4 = dtS4;
  DT_I8 = dtS8;
  DT_I16 = dtS16;
  DT_I32 = dtS32;

type
  PPQNNFloat = ^PQNNFloat;
  PQNNFloat = System.PSingle; // ^QNNFloat;
  QNNFloat = Single;

const
  DATATYPE_BITS: array[Low(TQNNDataType)..High(TQNNDataType)] of Byte =
      (0, 16, 16, 32, 32, 8, 8, 64, 8, 8, 8, 4, 4, 8, 4, 4, 16, 16);
  QNN_DEFAULT_DATATYPE = dtF32;

type

  TQNNSchedule = (
      QNN_SCHEDULE_DEFAULT = 0,
      QNN_SCHEDULE_LINEAR = 1,
      QNN_SCHEDULE_POWER = 2,
      QNN_SCHEDULE_SIGMOID = 3, (* Flux shifted sigmoid *)
      QNN_SCHEDULE_FLOWMATCH = 4 (* Z-Image FlowMatch Euler *)
  );

  TGenerateParams = record
    ModelName: string;
    Width, Height, NumSteps: LongInt;
    Seed: Int64;
    Guidance: Single;
    Schedule: TQNNSchedule;
    PowerAlpha: Single;
  end;
  // todo link TMemoryBlock to IMemoryArena for allocations

  // Packed GGUF storage is separate from the logical element type/shape.
  TQNNWeightEncoding = (weNative, weGGUFQ8_0, weGGUFQ4_K, weGGUFQ6_K);

  { TMemoryBlock }
  PMemoryBlock = ^TMemoryBlock;
  TMemoryBlock = record
    DataType: TQNNDataType;
    Encoding: TQNNWeightEncoding;
    Shape: TArray<Int64>;
    Offset: NativeInt;
    DataPtr: Pointer;
    Size: NativeInt;
{$IFDEF USE_CALLOC}
    Data32: PLongWord;
    Data16: PSmallInt;
    Data8: PByte;
    Data4: PInt4;
{$ELSE}
    Data32: TArray<LongWord>;
    Data16: TArray<Word>;
    Data8: TArray<Byte>;
    Data4: TArray<TInt4>;
{$ENDIF}
    // Identity is a promise that these model weights are immutable. It does
    // not own host storage, and is never inferred from a reusable CPU address.
    WeightIdentity: IInterface;
    Device: TQNNDevice;
    Name: string;
    const
      ERRSTR_CAST_ARRAY = 'MemoryBlock with non zero offset cannot be casted to an Array';
      ERRSTR_CAST_TYPE = 'ERROR : Data is not of ';
  private
    constructor Create(
        const ASize: NativeInt;
        const AName: string;
        const DType: TQNNDataType = QNN_DEFAULT_DATATYPE;
        const Src: Pointer = nil
    ); overload; // do not use
  public
    constructor Create(
        const AShape: TArray<Int64>;
        const AName: string;
        const DType: TQNNDataType = QNN_DEFAULT_DATATYPE;
        const AData: Pointer = nil
    ); overload;
    procedure Resize(const ASize: NativeInt; const ADataType: TQNNDataType); overload;
    procedure Resize(const AShape: TArray<Int64>; const ADataType: TQNNDataType); overload;
    procedure Free();
    procedure MarkImmutable;
    function Count: NativeInt;

    function Channels(): LongInt;
    function Height(): LongInt;
    function Width(): LongInt;
    procedure Reshape(const NewShape: TArray<Int64>);

    function IsAssigned(): Boolean;
    function IsAllocated(): Boolean;
    function IsNaN(): Boolean;
    procedure PrintStat;
    procedure AssignPtr(const Ptr: System.PSingle; const AShape: TArray<Int64>); overload;
    procedure AssignPtr(const Ptr: PBF16; const AShape: TArray<Int64>); overload;
    procedure AssignPtr(const Ptr: PFP16; const AShape: TArray<Int64>); overload;
    procedure AssignPtr(const Ptr: PLongint; const AShape: TArray<Int64>); overload;
    procedure AssignPtr(const Ptr: PSmallInt; const AShape: TArray<Int64>); overload;
    procedure AssignPtr(const Ptr: PShortInt; const AShape: TArray<Int64>); overload;
    procedure AssignPtr(const Ptr: PInt4; const AShape: TArray<Int64>); overload;

    // class operator implicit(const val:Pointer ):TMemoryBlock;

{$IFNDEF USE_CALLOC}
    class operator Implicit(const Val: TArray<LongInt>): TMemoryBlock;
    class operator Implicit(const Val: TArray<Single>): TMemoryBlock;
    class operator Implicit(const Val: TArray<BF16>): TMemoryBlock;
    class operator Implicit(const Val: TArray<FP16>): TMemoryBlock;
    class operator Implicit(const Val: TArray<SmallInt>): TMemoryBlock;
    class operator Implicit(const Val: TArray<ShortInt>): TMemoryBlock;
    class operator Implicit(const Val: TArray<TInt4>): TMemoryBlock;

    class operator Implicit(const Val: TMemoryBlock): TArray<LongInt>;
    class operator Implicit(const Val: TMemoryBlock): TArray<Single>;
    class operator Implicit(const Val: TMemoryBlock): TArray<BF16>;
    class operator Implicit(const Val: TMemoryBlock): TArray<FP16>;
    class operator Implicit(const Val: TMemoryBlock): TArray<SmallInt>;
    class operator Implicit(const Val: TMemoryBlock): TArray<ShortInt>;
    class operator Implicit(const Val: TMemoryBlock): TArray<TInt4>;
{$ENDIF}
    (*
      class operator implicit(const val:PLongint ):TMemoryBlock;
      class operator implicit(const val:PSingle  ):TMemoryBlock;
      class operator implicit(const val:PBF16    ):TMemoryBlock;
      class operator implicit(const val:PFP16    ):TMemoryBlock;
      class operator implicit(const val:Psmallint):TMemoryBlock;
      class operator implicit(const val:PShortint):TMemoryBlock;
      class operator implicit(const val:PINT4    ):TMemoryBlock;
*)

    class operator Implicit(const Val: TMemoryBlock): PLongint;
    class operator Implicit(const Val: TMemoryBlock): System.PSingle;
    class operator Implicit(const Val: TMemoryBlock): PBF16;
    class operator Implicit(const Val: TMemoryBlock): PFP16;
    class operator Implicit(const Val: TMemoryBlock): PSmallInt;
    class operator Implicit(const Val: TMemoryBlock): PShortInt;
    class operator Implicit(const Val: TMemoryBlock): PByte;
    class operator Implicit(const Val: TMemoryBlock): PInt4;
    class operator Implicit(const Val: TMemoryBlock): Boolean;
    class operator Add(const Src: TMemoryBlock; const AOffset: LongInt): TMemoryBlock;
    class operator Add(const Src: TMemoryBlock; const AOffset: Int64): TMemoryBlock;
    procedure PrintCompare(const Src: TMemoryBlock; const IsSumSqrDiff: Boolean = False);

    function TypeName(): string;
    class operator Initialize(out Val: TMemoryBlock);
  end;
  // old and new are memory sizes in bytes
  TOnMemoryUpdate = procedure(const Status: string; const Old, New: IntPtr; const Mem: TMemoryBlock);

  TQNNPixelOrder = (poHWC, poCHW);

  PQNNImage = ^TQNNImage;

  { TQNNImage }

  TQNNImage = record
    type
      TPngChunk = record
        Tag: array[0..3] of AnsiChar;
        Data: TArray<Byte>;
        Crc: LongWord;
      end;
    var
      Width: LongInt;
      Height: LongInt;
      Channels: LongInt; // RGB
      Data: TArray<Byte>; (* Row-major, channel-interleaved *)
    class function LoadFromFile(
        const FileName: string;
        ResizeWidth: LongInt = 0;
        ResizeHeight: LongInt = 0
    ): TQNNImage; static;
    class procedure AddPngMeta(const FileName: string; const Keyword, Meta: RawByteString); static;
    class procedure ComputePngCRCTable(); static;
    class function CalcPngCRC(
        const Buf: PByte;
        const Len: NativeInt;
        const InitCRC: LongWord = $ffffffff
    ): LongWord; static;
    function Resize(const W, H: LongInt): TQNNImage;
    constructor Create(const AWidth, AHeight: LongInt; const AChannels: LongInt = 3; const AData: PQNNFloat = nil);
    function AsMemoryBlock(
        const ADataType: TQNNDataType = QNN_DEFAULT_DATATYPE;
        const APixelOrder: TQNNPixelOrder = poCHW
    ): TMemoryBlock;
    procedure SaveToFile(const FileName: string; const TagKey: string = ''; const TagDesc: string = '');
    procedure Free();
  end;

  { TImageRef }
  PImageRef = ^TImageRef;

  TImageRef = record
    W, H, TOffset: LongInt;
    Latent: TMemoryBlock;
  end;

  { IMemoryArena }

  IMemoryArena = interface
    function Allocate(const ASize: NativeUInt): Pointer;
    procedure Resize(const ASize: NativeUInt);
    function ArenaSize: NativeUInt;
  end;

  { TMemoryArena }
  {$UNDEF DEBUG}
  TMemoryArena = class(TInterfacedObject, IMemoryArena)
  private
    FData: Pointer;
    FSize: NativeUInt; // in bytes
    FMemPos: NativeUInt;
  public
    constructor Create(const APool: NativeUInt);
    function ArenaSize: NativeUInt;
    procedure Resize(const ASize: NativeUInt);
    function Allocate(const ASize: NativeUInt): Pointer;
    destructor Destroy; override;
  end;

  TVocabs = TDictionary<RawByteString, LongInt>;

  TCHMult = array[0..3] of LongInt;

const
  LF = #10;
  CR = #13;
  SP = ' ';
  WHITE_SPACE = [TAB, LF, CR, SP];
  PUNCTUATIONS = ['!'..'/', ':'..'@', '['..'`', '{'..'~'];
  ALPHABETS = ['A'..'Z', 'a'..'z'];
  NUMERICS = ['0'..'9'];
  WORD_SEPERATORS = WHITE_SPACE + PUNCTUATIONS;

  QNN_LATENT_CHANNELS = 128; (* Flux: 32*2*2, Z-Image: 16*2*2=64 *)

  (* VAE architecture *)
  QNN_VAE_Z_CHANNELS = 32; (* Flux default; Z-Image uses 16 *)
  QNN_VAE_BASE_CH = 128;
  QNN_VAE_CH_MULT_0 = 1;
  QNN_VAE_CH_MULT_1 = 2;
  QNN_VAE_CH_MULT_2 = 4;
  QNN_VAE_CH_MULT_3 = 4;
  QNN_VAE_CH_MULT: TCHMult = (QNN_VAE_CH_MULT_0, QNN_VAE_CH_MULT_1, QNN_VAE_CH_MULT_2, QNN_VAE_CH_MULT_3);
  QNN_VAE_NUM_RES = 2;
  QNN_VAE_GROUPS = 32;
  QNN_VAE_MAX_DIM = 1792; (* Max image dimension for VAE *)

  (* Tokenizer *)
  QNN_MAX_SEQ_LEN = 512;
  QNN_VOCAB_HASH_SIZE = 150001;

  (* Sampling *)
  QNN_MAX_STEPS = 256;

// const
(* ========================================================================
   * Generation Parameters
   * ======================================================================== *)

(* Schedule type: 0 = model default (sigmoid for Flux, flowmatch for Z-Image) *)

// QNN_SCHEDULE_DEFAULT   = 0;
// QNN_SCHEDULE_LINEAR    = 1;
// QNN_SCHEDULE_POWER     = 2;
// QNN_SCHEDULE_SIGMOID   = 3;  (* Flux shifted sigmoid *)
// QNN_SCHEDULE_FLOWMATCH = 4;  (* Z-Image FlowMatch Euler *)

type
  TQNNParams = record
    Width: LongInt; (* Output width (default: 256) *)
    Height: LongInt; (* Output height (default: 256) *)
    NumSteps: LongInt; (* Inference steps (default: 4 distilled, 50 base) *)
    Seed: Int64; (* Random seed (-1 for random) *)
    Guidance: QNNFloat; (* CFG guidance scale (0 = auto from model type) *)
    Schedule: LongInt; (* Schedule type (SCHEDULE_*)
    PowerAlpha: QNNFloat; (* Exponent for power schedule (default: 2.0) *)
  end;

const
  (* Default parameters *)
  QNN_DEFAULT_WIDTH = 256;
  QNN_DEFAULT_HEIGHT = 256;
  QNN_PARAMS_DEFAULT: TQNNParams = (
      Width: QNN_DEFAULT_WIDTH;
      Height: QNN_DEFAULT_HEIGHT;
      NumSteps: 0;
      Seed: -1;
      Guidance: 0.0;
      Schedule: 0; // QNN_SCHEDULE_DEFAULT;
      PowerAlpha: 2.0
  );

function UTF8CharSize(const Str: AnsiChar): NativeInt;
function UTF8Length(const Str: RawByteString): NativeInt;

procedure BF16ToSingle(const N: NativeInt; const Src: PBF16; Dst: System.PSingle);
procedure SingleToBF16(const N: NativeInt; const Src: System.PSingle; Dst: PBF16);

procedure FP16ToSingle(const N: NativeInt; const Src: PFP16; Dst: System.PSingle);
procedure SingleToFP16(const N: NativeInt; const Src: System.PSingle; Dst: PFP16);

function Ifthen(const Cond: Boolean; const IfTrue, IfFalse: LongInt): LongInt; overload; inline;
function Ifthen(const Cond: Boolean; const IfTrue, IfFalse: Int64): Int64; overload; inline;
function Ifthen(const Cond: Boolean; const IfTrue, IfFalse: Single): Single; overload; inline;
function Ifthen(const Cond: Boolean; const IfTrue, IfFalse: Double): Double; overload; inline;
function Ifthen(const Cond: Boolean; const IfTrue, IfFalse: string): string; overload; inline;
function LCase(const C: AnsiChar): AnsiChar;
function UCase(const C: AnsiChar): AnsiChar;
function Product(const AR: TArray<Int64>): Int64;

function ReadInt(var F: file): LongInt;
function ReadSingles(var F: file; const Count: LongInt): TMemoryBlock;

{$IF defined(MSWINDOWS)}
function Calloc(Count, Size: UIntPtr): Pointer; cdecl; external 'msvcrt.dll' name 'calloc';
function Malloc(Size: UIntPtr): Pointer; cdecl; external 'msvcrt.dll' name 'malloc';
function Realloc(Mem: Pointer; Size: UIntPtr): Pointer; cdecl; external 'msvcrt.dll' name 'realloc';
procedure Free(Mem: Pointer); cdecl; external 'msvcrt.dll' name 'free';
procedure Printf(const Fmt: AnsiString); winapi; varargs; external 'msvcrt.dll' name 'printf';
{$ELSE}
function Calloc(Count, Size: UIntPtr): Pointer; cdecl; external name 'calloc';
function Malloc(Size: UIntPtr): Pointer; winapi; external name 'malloc';
function Realloc(Mem: Pointer; Size: UIntPtr): Pointer; winapi; external name 'realloc';
procedure Free(Mem: Pointer); cdecl; external name 'free';
function Mmap(
    Addr: Pointer;
    Len: UIntPtr;
    Prot, Flag, Fd: LongInt;
    Offset: IntPtr
): Pointer; winapi; external name 'mmap';
function Munmap(Addr: Pointer; Len: UIntPtr): LongInt; winapi; external name 'munmap';
procedure Printf(const Fmt: AnsiString); winapi; varargs; external name 'printf';

const
  PROT_NONE = $00;
  PROT_READ = $01;
  PROT_WRITE = $02;
  PROT_EXEC = $04;

  MAP_SHARED = 0001;
  MAP_PRIVATE = 0002;
  MAP_COPY = MAP_PRIVATE;

  MAP_FIXED = $0010;
  MAP_RENAME = $0020;

  MAP_NOCACHE = $0400;
  MAP_FAILED = Pointer(-1);

  MAP_FILE = $0000;
  MAP_ANON = $1000;
  MAP_ANONYMOUS = MAP_ANON;
{$ENDIF}

type
  TSubstepType = (
      SUBSTEP_DOUBLE_BLOCK, // Double-stream block completed
      SUBSTEP_SINGLE_BLOCK, // Single-stream block completed
      SUBSTEP_FINAL_LAYER // Final layer completed
  );

  (*
   * Substep callback - called during transformer forward pass.
   * type: which operation completed
   * index: 0-based index of this substep within its type
   * total: total count for this substep type
   *)
  TSubstepCallback = procedure(const &type: TSubstepType; const Index, Total: LongInt);

  (*
   * Step callback - called at sampling step boundaries.
   * step: current step (1-based), or 0 to indicate sampling is starting
   * total: total number of steps
   *)
  TStepCallback = procedure(const Step, Total: LongInt);
  TProgressCallback = procedure(const Step, Total: LongInt; const OutTensor: TMemoryBlock);

  (*
   * Phase callback - called at major phase boundaries.
   * phase: descriptive name ("encoding text", "decoding image", etc.)
   * done: 0 when starting, 1 when finished
   *)
  TPhaseCallBack = procedure(const Phase: string; const Done: Boolean);

  (*
   * Step image callback - called after each denoising step with decoded image.
   * step: current step (1-based)
   * total: total number of steps
   * img: decoded image at this step (caller must NOT free)
   *
   * To use: set both step_image_callback and step_image_vae before
   * calling the sampling function. The callback is only invoked when both are set.
   *)
  TStepImageCallback = procedure(const Step, Total: LongInt; const Img: TQNNImage (* todo maybe its an array? *));

  (*
   * Text encoder progress callback - called once per Qwen3 layer.
   * layer: current layer (0-based)
   * total: total number of layers (36)
   *)
  TTextProgressCallback = procedure(const Layer, Total: LongInt);

  (*
   * VAE progress callback - called once per resblock/attention block.
   * block: current block (0-based)
   * total: total number of blocks (11 for encoder, 15 for decoder)
   *)
  TVAEProgressCallback = procedure(const Block, Total: LongInt);

var
  SubstepCallback: TSubstepCallback;
  StepCallback: TStepCallback;
  PhaseCallback: TPhaseCallBack;
  StepImageCallback: TStepImageCallback;
  VAEPtr: Pointer;
  TextProgressCallback: TTextProgressCallback;
  VAEProgressCallback: TVAEProgressCallback;
  OnMemoryUpdate: TOnMemoryUpdate;

function IndexOf(const Needle: string; const Haystack: TArray<string>): NativeInt; overload;

function GetIndex(const Idx, Strides: TArray<Int64>): Int64;
function GetStrides(const Shape: TArray<Int64>): TArray<Int64>;

function GetFileSize64(var F: file): Int64;

implementation
uses
  System.SysUtils,
  System.Math
{$IFDEF MSWINDOWS}
  ,
  Winapi.Windows
{$ENDIF}
  ,
  System.UITypes,
  FMX.Types,
  FMX.Graphics,
  QuickNN.CPU;

var
  PNG_CRC_TABLE: array[0..255] of LongWord;

function GetFileSize64(
    var F: file
): Int64; // a work around Delphi FileSize function returning incorrect file size on large files
var
  Fl: LongWord;
begin
{$IF defined(MSWINDOWS)}
  Result := 0;
  Fl := GetFileSize(TFileRec(F).Handle, @Result);
  Result := Fl or (Result shl 32)
{$ELSE}
  Result := FileSize(F); // fallback to default
{$ENDIF}
end;

function GetIndex(const Idx, Strides: TArray<Int64>): Int64;
var
  I: NativeInt;
begin
  Assert(Length(Strides) = Length(Idx), '[getIndex]: idx and Tensor shape must be identical.');
  Result := 0;
  for I := 0 to High(Strides) do
    Inc(Result, Idx[I] * Strides[I]);
end;

function GetStrides(const Shape: TArray<Int64>): TArray<Int64>;
var
  N, I: Int64;
begin
  N := Length(Shape);
  SetLength(Result, N);
  if N = 0 then
    Exit;

  Result[N - 1] := 1;
  for I := N - 2 downto 0 do
    Result[I] := Result[I + 1] * Shape[I + 1];
end;

function LCase(const C: AnsiChar): AnsiChar;
begin
  Result := C;
  if Result in ['A'..'Z'] then
    Inc(Result, $20)

end;

function UCase(const C: AnsiChar): AnsiChar;
begin
  Result := C;
  if Result in ['a'..'z'] then
    Dec(Result, $20)
end;

function Product(const AR: TArray<Int64>): Int64;
var
  I: Integer;
begin
  // if not assigned(ar) then exit(0);
  Result := 1;
  for I := 0 to High(AR) do
    Result := Result * AR[I]
end;

function ReadInt(var F: file): LongInt;
begin
  BlockRead(F, Result, SizeOf(LongInt))
end;

function ReadSingles(var F: file; const Count: LongInt): TMemoryBlock;
begin
  Result := TMemoryBlock.Create(Count, 'readSingles ' + TGUID.NewGuid.ToString());
  BlockRead(F, System.PSingle(Result)^, Count * SizeOf(Single))
end;

function IndexOf(const Needle: string; const Haystack: TArray<string>): NativeInt;
var
  I: NativeInt;
begin
  for I := 0 to High(Haystack) do
    if Needle = Haystack[I] then
      Exit(I);
  Result := -1;
end;

{ TQNNImage }

function TQNNImage.Resize(const W, H: LongInt): TQNNImage;
var
  Y, X, X0, X1, Y0, Y1, C: LongInt;
  ScaleX, ScaleY, V, V00, V01, V10, V11, Wx, Wy, SrcX, SrcY: Single;
begin
  Result := TQNNImage.Create(W, H, Channels);

  ScaleX := Width / W;
  ScaleY := Height / H;

  for Y := 0 to H - 1 do
  begin
    for X := 0 to W - 1 do
    begin
      SrcX := (X + 0.5) * ScaleX - 0.5;
      SrcY := (Y + 0.5) * ScaleY - 0.5;

      X0 := Floor(SrcX);
      Y0 := Floor(SrcY);
      X1 := X0 + 1;
      Y1 := Y0 + 1;

      Wx := SrcX - X0;
      Wy := SrcY - Y0;

      if X0 < 0 then
        X0 := 0
      else if X0 >= Width then
        X0 := Width - 1;
      if X1 < 0 then
        X1 := 0
      else if X1 >= Width then
        X1 := Width - 1;
      if Y0 < 0 then
        Y0 := 0
      else if Y0 >= Height then
        Y0 := Height - 1;
      if Y1 < 0 then
        Y1 := 0
      else if Y1 >= Height then
        Y1 := Height - 1;

      for C := 0 to Channels - 1 do
      begin
        V00 := Data[(Y0 * Width + X0) * Channels + C];
        V01 := Data[(Y0 * Width + X1) * Channels + C];
        V10 := Data[(Y1 * Width + X0) * Channels + C];
        V11 := Data[(Y1 * Width + X1) * Channels + C];

        V := V00 * (1 - Wx) * (1 - Wy) + V01 * Wx * (1 - Wy) + V10 * (1 - Wx) * Wy + V11 * Wx * Wy;

        Result.Data[(Y * W + X) * Channels + C] := Round(V);
      end
    end
  end;
end;

constructor TQNNImage.Create(const AWidth, AHeight: LongInt; const AChannels: LongInt; const AData: PQNNFloat);
var
  Y, X, Ch: LongInt;
  Val: QNNFloat;
begin
  // data := nil;
  Width := AWidth;
  Height := AHeight;
  Channels := AChannels;
  SetLength(Data, Width * Height * Channels);
  if Assigned(AData) then
    for Y := 0 to Height - 1 do
      for X := 0 to Width - 1 do
        for Ch := 0 to Channels - 1 do
        begin
          Val := AData[(Ch * Height + Y) * Width + X];
          Val := (Val + 1.0) * 0.5;
          Val := Val * 255.0;
          if Val < 0 then
            Val := 0;
          if Val > 255 then
            Val := 255;
          Data[(Y * Width + X) * 3 + Ch] := Round(Val)
        end;
end;

function TQNNImage.AsMemoryBlock(const ADataType: TQNNDataType; const APixelOrder: TQNNPixelOrder): TMemoryBlock;
var
  C, Y, X, Idx: LongInt;
  SinglePtr: System.PSingle;
begin
  Assert(ADataType = dtF32, '[TQNNImage.asMemoryBlock]: ERROR DataType not implemented.');
  case APixelOrder of
    poHWC:
    begin
      Result := TMemoryBlock.Create([Height, Width, Channels], '', ADataType);
      case ADataType of
        dtF32:
        begin
          SinglePtr := Result;
          for Y := 0 to Height - 1 do
            for X := 0 to Width - 1 do
              for C := 0 to Channels - 1 do
              begin
                Idx := (Y * Width + X) * Channels + C;
                SinglePtr[Idx] := Data[Idx] * 2.0 / 255.0 - 1.0; // result range must be between -1.0 and +1.0
              end;
        end;

      end;
    end;
    poCHW:
    begin
      Result := TMemoryBlock.Create([Channels, Height, Width], IntToStr(Random($fffffff)), ADataType);
      case ADataType of
        dtF32:
        begin
          SinglePtr := Result;
          for C := 0 to Channels - 1 do
            for Y := 0 to Height - 1 do
              for X := 0 to Width - 1 do
              begin
                Idx := (Y * Width + X) * Channels + C;
                SinglePtr[(C * Height + Y) * Width + X] :=
                    Data[Idx] * 2.0 / 255.0 - 1.0; // result range must be between -1.0 and +1.0
              end;
        end;
      end
    end
  else
    Assert(False, '[TQNNImage.asMemoryBlock]: ERROR PixelOrder not implemented.');
  end;

end;

procedure TQNNImage.SaveToFile(const FileName: string; const TagKey: string; const TagDesc: string);
var
  Y, X: LongInt;

  Bmp: TBitmap;
  BmpData: TBitmapData;

  D: PByte;
  Ext: string;
begin

  Bmp := TBitmap.Create();
  Bmp.SetSize(Width, Height);
  Bmp.Map(TMapAccess.Write, BmpData);
  for Y := 0 to Height - 1 do
  begin
    D := BmpData.GetScanline(Y);
    for X := 0 to Width - 1 do
    begin
{$IFDEF MSWINDOWS}
      D[0] := Data[(Y * Width + X) * 3 + 2];
      D[1] := Data[(Y * Width + X) * 3 + 1];
      D[2] := Data[(Y * Width + X) * 3];
      D[3] := $FF;
{$ELSE}
      // todo check posix color order on delphi
      D[0] := $FF;
      D[1] := Data[(Y * Width + X) * 3];
      D[2] := Data[(Y * Width + X) * 3 + 1];
      D[3] := Data[(Y * Width + X) * 3 + 2];
{$ENDIF}
      Inc(D, 4);
    end;
  end;
  Bmp.Unmap(BmpData);
  Bmp.SaveToFile(FileName);
  Bmp.Free;

  Ext := LowerCase(ExtractFileExt(FileName));
  if (Ext = '.png') and (TagKey <> '') then
  begin
    AddPngMeta(FileName, UTF8String(TagKey), UTF8String(TagDesc));
  end;
end;

class function TQNNImage.LoadFromFile(const FileName: string; ResizeWidth: LongInt; ResizeHeight: LongInt): TQNNImage;

var
  Img: TBitmap;
  BmpData: TBitmapData;
  Y, X, LocalW, LocalH, ImSize: LongInt;
  D: PByte;
  P: TAlphaColor;
begin
  Result := Default(TQNNImage);
  Img := TBitmap.Create;
  try
    Img.LoadFromFile(FileName);
    if ResizeWidth <= 0 then
      ResizeWidth := Img.Width;
    if ResizeHeight <= 0 then
      ResizeHeight := Img.Height;
    LocalW := Img.Width;
    LocalH := Img.Height;
    Img.Map(TMapAccess.Read, BmpData);
    ImSize := ResizeHeight * ResizeWidth;
    SetLength(Result.Data, 3 * ImSize);
    for Y := 0 to ResizeHeight - 1 do
    begin
      for X := 0 to ResizeWidth - 1 do
      begin
        P := BmpData.GetPixel(Round(LocalW * X / ResizeWidth), Round(LocalH * Y / ResizeHeight)); // nearst neighor
        D := @Result.Data[3 * (Y * ResizeWidth + X)];
        D[0] := (P shr 16) and $FF;
        D[1] := (P shr 8) and $FF;
        D[2] := P and $FF;

      end;
    end;
    Result.Width := ResizeWidth;
    Result.Height := ResizeHeight;
    Result.Channels := 3;
  finally
    if Assigned(BmpData.Data) then
      Img.Unmap(BmpData);
    Img.Free
  end;
end;

function SwapEndian(const X: LongWord): LongWord;
begin
  Result := (X shl 24) or ((X shl 8) and $00ff0000) or ((X shr 8) and $0000ff00) or (X shr 24)
end;

class function TQNNImage.CalcPngCRC(const Buf: PByte; const Len: NativeInt; const InitCRC: LongWord): LongWord;
var
  I: NativeInt;
begin
  Result := InitCRC;
  for I := 0 to Len - 1 do
    Result := PNG_CRC_TABLE[(Result xor Buf[I]) and $ff] xor (Result shr 8);
  Result := Result xor $ffffffff; // isn't this the same as not()?
end;

class procedure TQNNImage.AddPngMeta(const FileName: string; const Keyword, Meta: RawByteString);
const
  PNG_SIGNATURE: UInt64 = $0A1A0A0D474E5089;
var
  F: file;
  Content: RawByteString;
  FSize: NativeInt;
  PngSig: UInt64;
  ChunkSize, R, I: LongWord;
  PngChunk: ^TPngChunk;
  PngChunks: TArray<TPngChunk>;
  Len, Crc: LongWord;
  StrData: AnsiString;
  Tag: array[0..3] of AnsiChar;
begin
  if (Keyword = '') or (Meta = '') then
    Exit;
  try
    AssignFile(F, FileName);
    Reset(F, 1);
    GetFileSize64(F);
    FSize := GetFileSize64(F);
    SetLength(Content, FSize);
    BlockRead(F, PngSig, SizeOf(PngSig));
    if PngSig = PNG_SIGNATURE then
    begin
      while not EOF(F) do
      begin
        BlockRead(F, ChunkSize, SizeOf(ChunkSize), R);
        ChunkSize := SwapEndian(ChunkSize);
        BlockRead(F, Tag, SizeOf(Tag));
        // if (FilePos(f)=fSize) or (tag='IEND') then break;
        // fpos := FilePos(f);
        // Seek(f, fpos + chunksize); // move to the next Chunk (data + CRC value)
        SetLength(PngChunks, Length(PngChunks) + 1);
        PngChunk := @PngChunks[High(PngChunks)];
        Move(Tag, PngChunk.Tag, SizeOf(Tag));
        SetLength(PngChunk.Data, ChunkSize);
        if Assigned(PngChunk.Data) then
          BlockRead(F, PngChunk.Data[0], ChunkSize);
        BlockRead(F, PngChunk.Crc, SizeOf(PngChunk.Crc));
        PngChunk.Crc := SwapEndian(PngChunk.Crc);
      end;
      Insert(Default(TPngChunk), PngChunks, 1);
      PngChunk := @PngChunks[1];
      PngChunk.Tag := 'tEXt';
      SetLength(PngChunk.Data, Length(Keyword) + Length(Meta) + 1);
      Move(Keyword[1], PngChunk.Data[0], Length(Keyword));
      Move(Meta[1], PngChunk.Data[Length(Keyword) + 1], Length(Meta));
      StrData := PngChunk.Tag + Keyword + #0 + Meta;
      PngChunk.Crc := CalcPngCRC(Pointer(StrData), Length(StrData));
      ReWrite(F, 1);
      BlockWrite(F, PNG_SIGNATURE, SizeOf(PNG_SIGNATURE));
      for I := 0 to High(PngChunks) do
      begin
        Len := SwapEndian(LongWord(Length(PngChunks[I].Data)));
        BlockWrite(F, Len, SizeOf(LongWord));
        BlockWrite(F, PngChunks[I].Tag, SizeOf(Tag));
        if Assigned(PngChunks[I].Data) then
        begin
          BlockWrite(F, PngChunks[I].Data[0], Length(PngChunks[I].Data));
        end;
        Crc := SwapEndian(PngChunks[I].Crc);
        BlockWrite(F, Crc, SizeOf(PngChunks[I].Crc));
      end;
    end;
  finally
    CloseFile(F)
  end;
end;

class procedure TQNNImage.ComputePngCRCTable();
var
  N, K: LongInt;
  C: LongWord;
begin
  for N := 0 to High(PNG_CRC_TABLE) do
  begin
    C := N;
    for K := 0 to 7 do
      if Boolean(C and 1) then
        C := $edb88320 xor (C shr 1)
      else
        C := C shr 1;
    PNG_CRC_TABLE[N] := C;
  end;
end;
procedure TQNNImage.Free();
begin
  if Assigned(Data) then
    SetLength(Data, 0);
  Self := Default(TQNNImage)
end;

{ TMemoryBlock }

constructor TMemoryBlock.Create(
    const ASize: NativeInt;
    const AName: string;
    const DType: TQNNDataType;
    const Src: Pointer
);

begin
  Self := Default(TMemoryBlock);
  DataType := DType;
  Name := AName;
  if Name = '' then
  begin

    Name := TGUID.NewGuid().ToString();
  end;
  if (ASize < 0) or (DATATYPE_BITS[DType] = 0) or (ASize > High(NativeInt) div DATATYPE_BITS[DType]) then
    raise ERangeError.Create('Invalid or overflowing memory block size');
  Size := ASize;
  if ASize = 0 then
    Exit;
  case DATATYPE_BITS[DType] of
    // 1 : setlength(Data1, aSize);
    4:
    begin
      Size := ASize;
{$IFDEF USE_CALLOC}
      Data4 := nil;
      Data4 := Malloc((ASize * DATATYPE_BITS[DType] + 4) div 8);
{$ELSE}
      SetLength(Data4, ASize);
{$ENDIF}
      if Assigned(OnMemoryUpdate) then
        OnMemoryUpdate('new', 0, (ASize * DATATYPE_BITS[DType] + 4) div 8, Self);
      Assert((ASize <> 0) and Assigned(Data4), 'ERROR : TMemoryBlock.Create, not enough memory!');
      if Assigned(Src) then
        Move(Src^, Pointer(Data4)^, (ASize + 1) div 2)
    end;

    8:
    begin
      Size := ASize;
{$IFDEF USE_CALLOC}
      Data8 := nil;
      Data8 := Malloc((ASize * DATATYPE_BITS[DType]) div 8);
{$ELSE}
      SetLength(Data8, ASize);
{$ENDIF}
      if Assigned(OnMemoryUpdate) then
        OnMemoryUpdate('new', 0, (ASize * DATATYPE_BITS[DType]) div 8, Self);
      Assert((ASize <> 0) and Assigned(Data8), 'ERROR : TMemoryBlock.Create, not enough memory!');
      if Assigned(Src) then
        Move(Src^, Pointer(Data8)^, (ASize * DATATYPE_BITS[DType]) div 8)
    end;

    16:
    begin
      Size := ASize;
{$IFDEF USE_CALLOC}
      Data16 := nil;
      Data16 := Malloc((ASize * DATATYPE_BITS[DType]) div 8);
{$ELSE}
      SetLength(Data16, ASize);
{$ENDIF}
      if Assigned(OnMemoryUpdate) then
        OnMemoryUpdate('new', 0, (ASize * DATATYPE_BITS[DType]) div 8, Self);
      Assert((ASize <> 0) and Assigned(Data16), 'ERROR : TMemoryBlock.Create, not enough memory!');
      if Assigned(Src) then
        Move(Src^, Pointer(Data16)^, (ASize * DATATYPE_BITS[DType]) div 8)
    end;

    32:
    begin
      Size := ASize;
{$IFDEF USE_CALLOC}
      Data32 := nil;
      Data32 := Malloc((ASize * DATATYPE_BITS[DType]) div 8);
      // Data32 := calloc(aSize, sizeOf(longint));
{$ELSE}
      SetLength(Data32, ASize);
{$ENDIF}
      if Assigned(OnMemoryUpdate) then
        OnMemoryUpdate('new', 0, (ASize * DATATYPE_BITS[DType]) div 8, Self);
      Assert((ASize <> 0) and Assigned(Data32), 'ERROR : TMemoryBlock.Create, not enough memory!');
      if Assigned(Src) then
        Move(Src^, Pointer(Data32)^, (ASize * DATATYPE_BITS[DType]) div 8)
    end;
  else
    Assert(False, 'ERROR : TMemoryBlock.create, Unsupported data type ')
  end;

end;

constructor TMemoryBlock.Create(
    const AShape: TArray<Int64>;
    const AName: string;
    const DType: TQNNDataType;
    const AData: Pointer
);
begin
  Self := Create(Product(AShape), AName, DType, AData);
  Shape := AShape
end;

procedure TMemoryBlock.Resize(const ASize: NativeInt; const ADataType: TQNNDataType);
var
  Status: string;
begin
  WeightIdentity := nil;
  Assert(not Assigned(DataPtr), 'TMemoryBlock.resize : cannot resize a pointer assigned memory!');
  Assert((ADataType <> dtUndef), 'TMemoryBlock.resize : Invalide DataType!');
  if DataType <> ADataType then
  begin
    Self.Free;
    Status := 'new';
  end;
  if IsAllocated() then
    Status := 'resize'
  else
    Status := 'new';
  if Name = '' then
    Name := TGUID.NewGuid.ToString();

  DataType := ADataType;
  case DATATYPE_BITS[DataType] of
    // 1 : setlength(Data1, aSize);
    4:
    begin
{$IFDEF USE_CALLOC}
      Data4 := Realloc(Data4, (ASize * DATATYPE_BITS[DataType]) div 8);
{$ELSE}
      SetLength(Data4, ASize);
{$ENDIF}
      if Assigned(OnMemoryUpdate) then
        OnMemoryUpdate(
            Status,
            (Size * DATATYPE_BITS[DataType] + 4) div 8,
            (ASize * DATATYPE_BITS[DataType] + 4) div 8,
            Self
        );
      Size := ASize;
      Assert(Assigned(Data4), 'ERROR : TMemoryBlock.Create, not enough memory!');
    end;

    8:
    begin
{$IFDEF USE_CALLOC}
      Data8 := Realloc(Data8, (ASize * DATATYPE_BITS[DataType]) div 8);
{$ELSE}
      SetLength(Data8, ASize);
{$ENDIF}
      if Assigned(OnMemoryUpdate) then
        OnMemoryUpdate(Status, Size * DATATYPE_BITS[DataType] div 8, (ASize * DATATYPE_BITS[DataType]) div 8, Self);
      Size := ASize;
      Assert(Assigned(Data8), 'ERROR : TMemoryBlock.Create, not enough memory!');
    end;

    16:
    begin
{$IFDEF USE_CALLOC}
      Data16 := Realloc(Data16, (ASize * DATATYPE_BITS[DataType]) div 8);
{$ELSE}
      SetLength(Data16, ASize);
{$ENDIF}
      if Assigned(OnMemoryUpdate) then
        OnMemoryUpdate(Status, Size * DATATYPE_BITS[DataType] div 8, (ASize * DATATYPE_BITS[DataType]) div 8, Self);
      Size := ASize;
      Assert(Assigned(Data16), 'ERROR : TMemoryBlock.Create, not enough memory!');
    end;

    32:
    begin
{$IFDEF USE_CALLOC}
      Data32 := Realloc(Data32, (ASize * DATATYPE_BITS[DataType]) div 8);
{$ELSE}
      SetLength(Data32, ASize);
{$ENDIF}
      if Assigned(OnMemoryUpdate) then
        OnMemoryUpdate(Status, Size * DATATYPE_BITS[DataType] div 8, (ASize * DATATYPE_BITS[DataType]) div 8, Self);
      Size := ASize;
      Assert(Assigned(Data32), 'ERROR : TMemoryBlock.Create, not enough memory!');
    end;
  else
    Assert(False, 'ERROR : TMemoryBlock.create, Unsupported data type ')
  end;
end;

procedure TMemoryBlock.Resize(const AShape: TArray<Int64>; const ADataType: TQNNDataType);
begin
  Resize(Product(AShape), ADataType);
  Shape := AShape;
end;

procedure TMemoryBlock.Free();
{$IF defined(USE_CALLOC)}
var
  M: Pointer;
{$ENDIF}
begin
  if Offset > 0 then
    Exit;
  if IsAllocated() then
  begin

    if Assigned(OnMemoryUpdate) and IsAllocated() then
      if DATATYPE_BITS[DataType] = 4 then
        OnMemoryUpdate('free', (Size * DATATYPE_BITS[DataType] + 4) div 8, 0, Self)
      else
        OnMemoryUpdate('free', (Size * DATATYPE_BITS[DataType]) div 8, 0, Self);
{$IFDEF USE_CALLOC}
    case DATATYPE_BITS[DataType] of
      4:
        if Assigned(Data4) and (Size > 0) then
        begin
          M := Data4;
          Data4 := nil;
          QuickNN.Common.Free(M)
        end;
      8:
        if Assigned(Data8) and (Size > 0) then
        begin
          M := Data8;
          Data8 := nil;
          QuickNN.Common.Free(M)
        end;
      16:
        if Assigned(Data16) and (Size > 0) then
        begin
          M := Data16;
          Data16 := nil;
          QuickNN.Common.Free(M)
        end;
      32:
        if Assigned(Data32) and (Size > 0) then
        begin
          M := Data32;
          Data32 := nil;
          QuickNN.Common.Free(M)
        end;
    end;
{$ELSE}
    //  case DATATYPE_BITS[DataType] of
    //    4 :setLength(Data4 , 0);
    //    8 :setLength(Data8 , 0);
    //    16:setLength(Data16, 0);
    //    32:setLength(Data32, 0);
    //  end;
    if Length(Data32) > 0 then
      SetLength(Data32, 0);
    if Length(Data16) > 0 then
      SetLength(Data16, 0);
    if Length(Data8) > 0 then
      SetLength(Data8, 0);
    if Length(Data4) > 0 then
      SetLength(Data4, 0);
{$ENDIF}
  end;
  Self.DataType := dtUndef;
  Self.Encoding := weNative;
  Self.Offset := 0;
  Self.Size := 0;
  Self.Shape := nil;
  Self.DataPtr := nil;
  Self.Name := '';
  WeightIdentity := nil;
  // fillchar(self, sizeof(self), #0);
end;

procedure TMemoryBlock.MarkImmutable;
begin
  if WeightIdentity = nil then
    WeightIdentity := TInterfacedObject.Create;
end;

function TMemoryBlock.Count: NativeInt;
begin
  Result := Size;
end;

function TMemoryBlock.Channels(): LongInt;
begin
  Assert(Length(Shape) > 2, 'ERROR : Memory has no channels dimension');
  Result := Shape[High(Shape) - 2]
end;

function TMemoryBlock.Height(): LongInt;
begin
  Assert(Length(Shape) > 1, 'ERROR : Memory has no height dimension');
  Result := Shape[High(Shape) - 1]
end;

function TMemoryBlock.Width(): LongInt;
begin
  Result := Shape[High(Shape)]
end;

procedure TMemoryBlock.Reshape(const NewShape: TArray<Int64>);
begin
  Assert(Product(NewShape) <= Size, 'ERROR [reshape] : new Tensor size shape not match the current Tensor size');
  Shape := NewShape
end;

function TMemoryBlock.IsAssigned(): Boolean;
begin
  Result := False;
  case DATATYPE_BITS[DataType] of
{$IFDEF USE_CALLOC}
    4: Result := Assigned(Data4);
    8: Result := Assigned(Data8);
    16: Result := Assigned(Data16);
    32: Result := Assigned(Data32)
{$ELSE}
    4: Result := Length(Data4) > 0;
    8: Result := Length(Data8) > 0;
    16: Result := Length(Data16) > 0;
    32: Result := Length(Data32) > 0
{$ENDIF}

  end;
  if not Result then
    Result := Assigned(DataPtr)
end;

function TMemoryBlock.IsAllocated(): Boolean;
begin
{$IFDEF USE_CALLOC}
  Result := Assigned(Data4) or Assigned(Data8) or Assigned(Data16) or Assigned(Data32);
{$ELSE}
  Result := (Length(Data4) > 0) or (Length(Data8) > 0) or (Length(Data16) > 0) or (Length(Data32) > 0);
{$ENDIF}
  if Result then
    Assert(
        not Assigned(DataPtr),
        'ERROR [TMemoryBlock.isAllocated()] : Overlapping assigned memory and allocated memory were found! while one should exist!'
    );
end;
procedure TMemoryBlock.PrintStat;
begin
  case DataType of
    dtF32: QuickNN.CPU.PrintStat(Self);
  else
    Assert(False, 'ERROR : Datatype ' + GetEnumName(TypeInfo(TQNNDataType), Ord(DataType)) + ' not implemented!')
  end;
end;

function TMemoryBlock.IsNaN(): Boolean;
var
  I: IntPtr;
  F: System.PSingle;

  B: PBF16;
  S: LongWord;
begin
  Result := False;
  if not (DataType in [dtF32, dtF64]) then
    Exit();
  case DataType of
    dtF32:
    begin
      F := Self;
      for I := 0 to Count - 1 do
        if System.Math.IsNan(F[I]) then
          Exit(True);
    end;
    dtBF16:
    begin
      B := Self;
      for I := 0 to Count - 1 do
      begin
        S := B[I] shl 16;
        if System.Math.IsNan(System.PSingle(@S)^) then
          Exit(True);
      end;
    end;
  else
    Assert(False, 'is_nan : not implemented for this data type')

  end;

end;

procedure TMemoryBlock.AssignPtr(const Ptr: System.PSingle; const AShape: TArray<Int64>);
begin
  Assert(not IsAllocated(), 'TMemoryBlock.assignPtr : Cannot assign a pointer to an already allocatd memory block!');
  WeightIdentity := nil;
  Encoding := weNative;
  Shape := AShape;
  Size := Product(AShape);
  DataType := dtF32;
  DataPtr := Ptr;
end;

procedure TMemoryBlock.AssignPtr(const Ptr: PBF16; const AShape: TArray<Int64>);
begin
  Assert(not IsAllocated(), 'TMemoryBlock.assignPtr : Cannot assign a pointer to an already allocatd memory block!');
  WeightIdentity := nil;
  Encoding := weNative;
  Shape := AShape;
  Size := Product(AShape);
  DataType := dtBF16;
  DataPtr := Ptr;
end;

procedure TMemoryBlock.AssignPtr(const Ptr: PFP16; const AShape: TArray<Int64>);
begin
  Assert(not IsAllocated(), 'TMemoryBlock.assignPtr : Cannot assign a pointer to an already allocatd memory block!');
  WeightIdentity := nil;
  Encoding := weNative;
  Shape := AShape;
  Size := Product(AShape);
  DataType := dtF16;
  DataPtr := Ptr;
end;

procedure TMemoryBlock.AssignPtr(const Ptr: PLongint; const AShape: TArray<Int64>);
begin
  Assert(not IsAllocated(), 'TMemoryBlock.assignPtr : Cannot assign a pointer to an already allocatd memory block!');
  WeightIdentity := nil;
  Encoding := weNative;
  Shape := AShape;
  Size := Product(AShape);
  DataType := dtS32;
  DataPtr := Ptr;
end;

procedure TMemoryBlock.AssignPtr(const Ptr: PSmallInt; const AShape: TArray<Int64>);
begin
  Assert(not IsAllocated(), 'TMemoryBlock.assignPtr : Cannot assign a pointer to an already allocatd memory block!');
  WeightIdentity := nil;
  Encoding := weNative;
  Shape := AShape;
  Size := Product(AShape);
  DataType := dtS16;
  DataPtr := Ptr;
end;

procedure TMemoryBlock.AssignPtr(const Ptr: PShortInt; const AShape: TArray<Int64>);
begin
  Assert(not IsAllocated(), 'TMemoryBlock.assignPtr : Cannot assign a pointer to an already allocatd memory block!');
  WeightIdentity := nil;
  Encoding := weNative;
  Shape := AShape;
  Size := Product(AShape);
  DataType := dtS8;
  DataPtr := Ptr;
end;

procedure TMemoryBlock.AssignPtr(const Ptr: PInt4; const AShape: TArray<Int64>);
begin
  Assert(not IsAllocated(), 'TMemoryBlock.assignPtr : Cannot assign a pointer to an already allocatd memory block!');
  WeightIdentity := nil;
  Encoding := weNative;
  Shape := AShape;
  Size := Product(AShape);
  DataType := dtS4;
  DataPtr := Ptr;
end;

// class operator TMemoryBlock.implicit(const val:Pointer ):TMemoryBlock;
// begin
//  assert(false, 'ERROR : Untyped tensors not allowed.')
// end;

{$IFNDEF USE_CALLOC}
class operator TMemoryBlock.Implicit(const Val: TArray<LongInt>): TMemoryBlock;
begin
  Result.DataType := dtS32;
  Result.Data32 := TArray<LongWord>(Val);
  Result.Size := Length(Val);
  // if assigned(onMemoryUpdate) then
  //  onMemoryUpdate('new', 0, result.Size*DATATYPE_BITS[result.DataType] div 8, result);
end;

class operator TMemoryBlock.Implicit(const Val: TArray<Single>): TMemoryBlock;
begin
  Result.DataType := dtF32;
  Result.Data32 := TArray<LongWord>(Val);
  Result.Size := Length(Val);
  // if assigned(onMemoryUpdate) then
  //  onMemoryUpdate('new', 0, result.Size*DATATYPE_BITS[result.DataType] div 8, result);
end;

class operator TMemoryBlock.Implicit(const Val: TArray<BF16>): TMemoryBlock;
begin
  Result.DataType := dtBF16;
  Result.Data16 := TArray<Word>(Val);
  Result.Size := Length(Val);
  // if assigned(onMemoryUpdate) then
  //  onMemoryUpdate('new', 0, result.Size*DATATYPE_BITS[result.DataType] div 8, result);
end;

class operator TMemoryBlock.Implicit(const Val: TArray<FP16>): TMemoryBlock;
begin
  Result.DataType := dtF16;
  Result.Data16 := TArray<Word>(Val);
  Result.Size := Length(Val);
  // if assigned(onMemoryUpdate) then
  //  onMemoryUpdate('new', 0, result.Size*DATATYPE_BITS[result.DataType] div 8, result);
end;

class operator TMemoryBlock.Implicit(const Val: TArray<SmallInt>): TMemoryBlock;
begin
  Result.DataType := dtS16;
  Result.Data16 := TArray<Word>(Val);
  Result.Size := Length(Val);
  // if assigned(onMemoryUpdate) then
  //  onMemoryUpdate('new', 0, result.Size*DATATYPE_BITS[result.DataType] div 8, result);
end;

class operator TMemoryBlock.Implicit(const Val: TArray<ShortInt>): TMemoryBlock;
begin
  Result.DataType := dtS8;
  Result.Data8 := TArray<Byte>(Val);
  Result.Size := Length(Val);
  // if assigned(onMemoryUpdate) then
  //  onMemoryUpdate('new', 0, result.Size*DATATYPE_BITS[result.DataType] div 8, result);
end;

class operator TMemoryBlock.Implicit(const Val: TArray<TInt4>): TMemoryBlock;
begin
  Result.DataType := dtS4;
  Result.Data4 := Val;
  Result.Size := Length(Val);
  // if assigned(onMemoryUpdate) then
  //  onMemoryUpdate('new', 0, (result.Size*DATATYPE_BITS[result.DataType]+4) div 8, result);
end;

class operator TMemoryBlock.Implicit(const Val: TMemoryBlock): TArray<LongInt>;
begin
  Assert(Val.DataType = dtS32, ERRSTR_CAST_TYPE + 'INT32');
  Assert(Val.Offset = 0, ERRSTR_CAST_ARRAY);
  Result := TArray<LongInt>(Val.Data32);
end;

class operator TMemoryBlock.Implicit(const Val: TMemoryBlock): TArray<Single>;
begin
  Assert(Val.DataType = dtF32, ERRSTR_CAST_TYPE + 'FP32');
  Assert(Val.Offset = 0, ERRSTR_CAST_ARRAY);
  Result := TArray<Single>(Val.Data32);
end;

class operator TMemoryBlock.Implicit(const Val: TMemoryBlock): TArray<BF16>;
begin
  Assert(Val.DataType = dtBF16, ERRSTR_CAST_TYPE + 'BF16');
  Assert(Val.Offset = 0, ERRSTR_CAST_ARRAY);
  Result := TArray<BF16>(Val.Data16);
end;

class operator TMemoryBlock.Implicit(const Val: TMemoryBlock): TArray<FP16>;
begin
  Assert(Val.DataType = dtF16, ERRSTR_CAST_TYPE + 'FP16');
  Assert(Val.Offset = 0, ERRSTR_CAST_ARRAY);
  Result := TArray<FP16>(Val.Data16);
end;

class operator TMemoryBlock.Implicit(const Val: TMemoryBlock): TArray<SmallInt>;
begin
  Assert(Val.DataType = dtS16, ERRSTR_CAST_TYPE + 'INT16');
  Assert(Val.Offset = 0, ERRSTR_CAST_ARRAY);
  Result := TArray<SmallInt>(Val.Data16);
end;

class operator TMemoryBlock.Implicit(const Val: TMemoryBlock): TArray<ShortInt>;
begin
  Assert(Val.DataType = dtS8, ERRSTR_CAST_TYPE + 'INT8');
  Assert(Val.Offset = 0, ERRSTR_CAST_ARRAY);
  Result := TArray<ShortInt>(Val.Data8);
end;

class operator TMemoryBlock.Implicit(const Val: TMemoryBlock): TArray<TInt4>;
begin
  Assert(Val.DataType = dtS4, ERRSTR_CAST_TYPE + 'INT4');
  Assert(Val.Offset = 0, ERRSTR_CAST_ARRAY);
  Result := TArray<TInt4>(Val.Data4);
end;

{$ENDIF}

class operator TMemoryBlock.Implicit(const Val: TMemoryBlock): Boolean;
begin
  Result := Val.IsAssigned();
end;

(*
class operator TMemoryBlock.implicit(const val:PLongint ):TMemoryBlock;
begin
  result.DataType := dts32;
  result.DataPtr := val;
end;

class operator TMemoryBlock.implicit(const val:PSingle  ):TMemoryBlock;
begin
  result.DataType := dtf32;
  result.DataPtr := val;
end;

class operator TMemoryBlock.implicit(const val:PBF16    ):TMemoryBlock;
begin
  result.DataType := dtBf16;
  result.DataPtr := val;
end;

class operator TMemoryBlock.implicit(const val:PFP16    ):TMemoryBlock;
begin
  result.DataType := dtF16;
  result.DataPtr := val;
end;

class operator TMemoryBlock.implicit(const val:Psmallint):TMemoryBlock;
begin
  result.DataType := dtS16;
  result.DataPtr := val;
end;

class operator TMemoryBlock.implicit(const val:PShortint):TMemoryBlock;
begin
  result.DataType := dtS8;
  result.DataPtr := val;
end;

class operator TMemoryBlock.implicit(const val:PINT4    ):TMemoryBlock;
begin
  result.DataType := dtS4;
  result.DataPtr := val;
end;
*)

class operator TMemoryBlock.Implicit(const Val: TMemoryBlock): PLongint;
begin
  Assert(Val.DataType = dtS32, 'ERROR : Data is not of ' + 'INT32');
  if Assigned(Val.Data32) then
    Result := Pointer(Val.Data32)
  else
    Result := Val.DataPtr;
  Inc(Result, Val.Offset)
end;

class operator TMemoryBlock.Implicit(const Val: TMemoryBlock): System.PSingle;
begin
  Assert(Val.DataType = dtF32, 'ERROR : Data is not of ' + 'FP32');
  if Assigned(Val.Data32) then
    Result := Pointer(Val.Data32)
  else
    Result := Val.DataPtr;
  Inc(Result, Val.Offset)
end;

class operator TMemoryBlock.Implicit(const Val: TMemoryBlock): PBF16;
begin
  Assert(Val.DataType = dtBF16, 'ERROR : Data is not of ' + 'BF16');
  if Assigned(Val.Data16) then
    Result := Pointer(Val.Data16)
  else
    Result := Val.DataPtr;
  Inc(Result, Val.Offset)
end;

class operator TMemoryBlock.Implicit(const Val: TMemoryBlock): PFP16;
begin
  Assert(Val.DataType = dtF16, 'ERROR : Data is not of ' + 'FP16');
  if Assigned(Val.Data16) then
    Result := Pointer(Val.Data16)
  else
    Result := Val.DataPtr;
  Inc(Result, Val.Offset)
end;

class operator TMemoryBlock.Implicit(const Val: TMemoryBlock): PSmallInt;
begin
  Assert(Val.DataType = dtS16, 'ERROR : Data is not of ' + 'INT16');
  if Assigned(Val.Data16) then
    Result := Pointer(Val.Data16)
  else
    Result := Val.DataPtr;
  Inc(Result, Val.Offset)
end;

class operator TMemoryBlock.Implicit(const Val: TMemoryBlock): PShortInt;
begin
  Assert(Val.DataType = dtS8, 'ERROR : Data is not of ' + 's8');
  if Assigned(Val.Data8) then
    Result := Pointer(Val.Data8)
  else
    Result := Val.DataPtr;
  Inc(Result, Val.Offset)
end;

class operator TMemoryBlock.Implicit(const Val: TMemoryBlock): PByte;
begin
  Assert(Val.DataType = dtU8, 'MemoryBlock is not U8');
  if Assigned(Val.Data8) then
    Result := Pointer(Val.Data8)
  else
    Result := Val.DataPtr;
  Inc(Result, Val.Offset);
end;

class operator TMemoryBlock.Implicit(const Val: TMemoryBlock): PInt4;
begin
  Assert(Val.DataType = dtS4, 'ERROR : Data is not of ' + 'INT4');
  if Assigned(Val.Data4) then
    Result := Pointer(Val.Data4)
  else
    Result := Val.DataPtr;
  Inc(Result, Val.Offset) // todo Implicit to PInt4 : check validity of int4 pointer offseting
end;

class operator TMemoryBlock.Add(const Src: TMemoryBlock; const AOffset: LongInt): TMemoryBlock;
begin
  Assert(InRange(Src.Offset + AOffset, 0, Src.Size), '[ERROR] offset is beyond the Tensor boundries.');
  Result := Src;
  Result.Offset := Src.Offset + AOffset;
end;

class operator TMemoryBlock.Add(const Src: TMemoryBlock; const AOffset: Int64): TMemoryBlock;
begin
  Assert(InRange(Src.Offset + AOffset, 0, Src.Size), '[ERROR] offset is beyond the Tensor boundries.');
  Result := Src;
  Result.Offset := Src.Offset + AOffset;
end;

procedure TMemoryBlock.PrintCompare(const Src: TMemoryBlock; const IsSumSqrDiff: Boolean = False);
var
  Md, Src1, Src2: Single;
begin
  Assert(
      (Count = Src.Count) and (DataType = Src.DataType),
      'Tensor sizes do not match! ' + IntToStr(Count()) + '<>' + IntToStr(Src.Count())
  );
  PrintStat;
  Src.PrintStat;
  case DataType of
    dtF32:
    begin
      if IsSumSqrDiff then
      begin
        Md := TQNNSingleOps.QNNSqrDistance(Count, Self, Src);
        WriteLn('SqrDistance :', Md: 1: 5);
      end
      else
      begin
        Md := TQNNSingleOps.QNNMaxAbsDiff2(Count, Self, Src, Src1, Src2);
        // if md<>0 then begin
        // end;
        WriteLn('MaxAbsDiff :', Md: 1: 5, ' max src1 :', Src1: 1: 6, ' max src2:', Src2: 1: 6);
      end;
    end
  else
    Assert(False, 'printCompare : datatype not implemented!')
  end;
end;

function TMemoryBlock.TypeName: string;
begin
  Result := GetEnumName(TypeInfo(TQNNDataType), Ord(DataType));
end;

class operator TMemoryBlock.Initialize(out Val: TMemoryBlock);
begin
  FillChar(Val, SizeOf(TMemoryBlock), #0);
end;

function UTF8CharSize(const Str: AnsiChar): NativeInt;
var
  C: Byte absolute Str;
begin
  if (C and $80) = 0 then
    Exit(1);
  if (C and $E0) = $C0 then
    Exit(2);
  if (C and $F0) = $E0 then
    Exit(3);
  if (C and $F8) = $F0 then
    Exit(4);
  Result := 1 (* if invalid *)
end;

function UTF8Length(const Str: RawByteString): NativeInt;
var
  S: PAnsiChar absolute Str;
  I: NativeInt;
begin
  Result := 0;
  I := 0;
  while I < Length(Str) do
  begin
    Inc(I, UTF8CharSize(S[I]));
    Inc(Result)
  end;
end;

type
  PConversionBits = ^ConversionBits;
  ConversionBits = record
  case Boolean of
    False: (I32: LongWord);
    True: (F32: Single);
  end;

procedure BF16ToSingle(const N: NativeInt; const Src: PBF16; Dst: System.PSingle);
var
  I: NativeInt;
  Cdst: PConversionBits absolute Dst;
begin
{$IF defined(CPUX64) and defined(USE_AVX2)}
  I := 0;
  while I + 8 <= N do
  begin
    BF16ToSingleX8AVX2(Src + I, Dst + I);
    Inc(I, 8)
  end;
  for I := I to N - 1 do
    Cdst[I].I32 := Src[I] shl 16;
{$ELSE}
  for I := 0 to N - 1 do
    Cdst[I].I32 := Src[I] shl 16;
{$ENDIF}
end;

procedure SingleToBF16(const N: NativeInt; const Src: System.PSingle; Dst: PBF16);
var
  I: NativeInt;
  Csrc: PConversionBits absolute Src;
begin
{$IF defined(CPUX64) and defined(USE_AVX2)}
  I := 0;
  while I + 8 <= N do
  begin
    SingleToBF16X8AVX2(Src + I, Dst + I);
    Inc(I, 8)
  end;
  for I := I to N - 1 do
    Dst[I] := Csrc[I].I32 shr 16;
{$ELSE}
  for I := 0 to N - 1 do
    Dst[I] := Csrc[I].I32 shr 16;
{$ENDIF}
end;

procedure FP16ToSingle(const N: NativeInt; const Src: PFP16; Dst: System.PSingle);
const
  MANTISSA_TABLE: array[0..2047] of LongWord = (
      $00000000,
      $33800000,
      $34000000,
      $34400000,
      $34800000,
      $34A00000,
      $34C00000,
      $34E00000,
      $35000000,
      $35100000,
      $35200000,
      $35300000,
      $35400000,
      $35500000,
      $35600000,
      $35700000,
      $35800000,
      $35880000,
      $35900000,
      $35980000,
      $35A00000,
      $35A80000,
      $35B00000,
      $35B80000,
      $35C00000,
      $35C80000,
      $35D00000,
      $35D80000,
      $35E00000,
      $35E80000,
      $35F00000,
      $35F80000,
      $36000000,
      $36040000,
      $36080000,
      $360C0000,
      $36100000,
      $36140000,
      $36180000,
      $361C0000,
      $36200000,
      $36240000,
      $36280000,
      $362C0000,
      $36300000,
      $36340000,
      $36380000,
      $363C0000,
      $36400000,
      $36440000,
      $36480000,
      $364C0000,
      $36500000,
      $36540000,
      $36580000,
      $365C0000,
      $36600000,
      $36640000,
      $36680000,
      $366C0000,
      $36700000,
      $36740000,
      $36780000,
      $367C0000,
      $36800000,
      $36820000,
      $36840000,
      $36860000,
      $36880000,
      $368A0000,
      $368C0000,
      $368E0000,
      $36900000,
      $36920000,
      $36940000,
      $36960000,
      $36980000,
      $369A0000,
      $369C0000,
      $369E0000,
      $36A00000,
      $36A20000,
      $36A40000,
      $36A60000,
      $36A80000,
      $36AA0000,
      $36AC0000,
      $36AE0000,
      $36B00000,
      $36B20000,
      $36B40000,
      $36B60000,
      $36B80000,
      $36BA0000,
      $36BC0000,
      $36BE0000,
      $36C00000,
      $36C20000,
      $36C40000,
      $36C60000,
      $36C80000,
      $36CA0000,
      $36CC0000,
      $36CE0000,
      $36D00000,
      $36D20000,
      $36D40000,
      $36D60000,
      $36D80000,
      $36DA0000,
      $36DC0000,
      $36DE0000,
      $36E00000,
      $36E20000,
      $36E40000,
      $36E60000,
      $36E80000,
      $36EA0000,
      $36EC0000,
      $36EE0000,
      $36F00000,
      $36F20000,
      $36F40000,
      $36F60000,
      $36F80000,
      $36FA0000,
      $36FC0000,
      $36FE0000,
      $37000000,
      $37010000,
      $37020000,
      $37030000,
      $37040000,
      $37050000,
      $37060000,
      $37070000,
      $37080000,
      $37090000,
      $370A0000,
      $370B0000,
      $370C0000,
      $370D0000,
      $370E0000,
      $370F0000,
      $37100000,
      $37110000,
      $37120000,
      $37130000,
      $37140000,
      $37150000,
      $37160000,
      $37170000,
      $37180000,
      $37190000,
      $371A0000,
      $371B0000,
      $371C0000,
      $371D0000,
      $371E0000,
      $371F0000,
      $37200000,
      $37210000,
      $37220000,
      $37230000,
      $37240000,
      $37250000,
      $37260000,
      $37270000,
      $37280000,
      $37290000,
      $372A0000,
      $372B0000,
      $372C0000,
      $372D0000,
      $372E0000,
      $372F0000,
      $37300000,
      $37310000,
      $37320000,
      $37330000,
      $37340000,
      $37350000,
      $37360000,
      $37370000,
      $37380000,
      $37390000,
      $373A0000,
      $373B0000,
      $373C0000,
      $373D0000,
      $373E0000,
      $373F0000,
      $37400000,
      $37410000,
      $37420000,
      $37430000,
      $37440000,
      $37450000,
      $37460000,
      $37470000,
      $37480000,
      $37490000,
      $374A0000,
      $374B0000,
      $374C0000,
      $374D0000,
      $374E0000,
      $374F0000,
      $37500000,
      $37510000,
      $37520000,
      $37530000,
      $37540000,
      $37550000,
      $37560000,
      $37570000,
      $37580000,
      $37590000,
      $375A0000,
      $375B0000,
      $375C0000,
      $375D0000,
      $375E0000,
      $375F0000,
      $37600000,
      $37610000,
      $37620000,
      $37630000,
      $37640000,
      $37650000,
      $37660000,
      $37670000,
      $37680000,
      $37690000,
      $376A0000,
      $376B0000,
      $376C0000,
      $376D0000,
      $376E0000,
      $376F0000,
      $37700000,
      $37710000,
      $37720000,
      $37730000,
      $37740000,
      $37750000,
      $37760000,
      $37770000,
      $37780000,
      $37790000,
      $377A0000,
      $377B0000,
      $377C0000,
      $377D0000,
      $377E0000,
      $377F0000,
      $37800000,
      $37808000,
      $37810000,
      $37818000,
      $37820000,
      $37828000,
      $37830000,
      $37838000,
      $37840000,
      $37848000,
      $37850000,
      $37858000,
      $37860000,
      $37868000,
      $37870000,
      $37878000,
      $37880000,
      $37888000,
      $37890000,
      $37898000,
      $378A0000,
      $378A8000,
      $378B0000,
      $378B8000,
      $378C0000,
      $378C8000,
      $378D0000,
      $378D8000,
      $378E0000,
      $378E8000,
      $378F0000,
      $378F8000,
      $37900000,
      $37908000,
      $37910000,
      $37918000,
      $37920000,
      $37928000,
      $37930000,
      $37938000,
      $37940000,
      $37948000,
      $37950000,
      $37958000,
      $37960000,
      $37968000,
      $37970000,
      $37978000,
      $37980000,
      $37988000,
      $37990000,
      $37998000,
      $379A0000,
      $379A8000,
      $379B0000,
      $379B8000,
      $379C0000,
      $379C8000,
      $379D0000,
      $379D8000,
      $379E0000,
      $379E8000,
      $379F0000,
      $379F8000,
      $37A00000,
      $37A08000,
      $37A10000,
      $37A18000,
      $37A20000,
      $37A28000,
      $37A30000,
      $37A38000,
      $37A40000,
      $37A48000,
      $37A50000,
      $37A58000,
      $37A60000,
      $37A68000,
      $37A70000,
      $37A78000,
      $37A80000,
      $37A88000,
      $37A90000,
      $37A98000,
      $37AA0000,
      $37AA8000,
      $37AB0000,
      $37AB8000,
      $37AC0000,
      $37AC8000,
      $37AD0000,
      $37AD8000,
      $37AE0000,
      $37AE8000,
      $37AF0000,
      $37AF8000,
      $37B00000,
      $37B08000,
      $37B10000,
      $37B18000,
      $37B20000,
      $37B28000,
      $37B30000,
      $37B38000,
      $37B40000,
      $37B48000,
      $37B50000,
      $37B58000,
      $37B60000,
      $37B68000,
      $37B70000,
      $37B78000,
      $37B80000,
      $37B88000,
      $37B90000,
      $37B98000,
      $37BA0000,
      $37BA8000,
      $37BB0000,
      $37BB8000,
      $37BC0000,
      $37BC8000,
      $37BD0000,
      $37BD8000,
      $37BE0000,
      $37BE8000,
      $37BF0000,
      $37BF8000,
      $37C00000,
      $37C08000,
      $37C10000,
      $37C18000,
      $37C20000,
      $37C28000,
      $37C30000,
      $37C38000,
      $37C40000,
      $37C48000,
      $37C50000,
      $37C58000,
      $37C60000,
      $37C68000,
      $37C70000,
      $37C78000,
      $37C80000,
      $37C88000,
      $37C90000,
      $37C98000,
      $37CA0000,
      $37CA8000,
      $37CB0000,
      $37CB8000,
      $37CC0000,
      $37CC8000,
      $37CD0000,
      $37CD8000,
      $37CE0000,
      $37CE8000,
      $37CF0000,
      $37CF8000,
      $37D00000,
      $37D08000,
      $37D10000,
      $37D18000,
      $37D20000,
      $37D28000,
      $37D30000,
      $37D38000,
      $37D40000,
      $37D48000,
      $37D50000,
      $37D58000,
      $37D60000,
      $37D68000,
      $37D70000,
      $37D78000,
      $37D80000,
      $37D88000,
      $37D90000,
      $37D98000,
      $37DA0000,
      $37DA8000,
      $37DB0000,
      $37DB8000,
      $37DC0000,
      $37DC8000,
      $37DD0000,
      $37DD8000,
      $37DE0000,
      $37DE8000,
      $37DF0000,
      $37DF8000,
      $37E00000,
      $37E08000,
      $37E10000,
      $37E18000,
      $37E20000,
      $37E28000,
      $37E30000,
      $37E38000,
      $37E40000,
      $37E48000,
      $37E50000,
      $37E58000,
      $37E60000,
      $37E68000,
      $37E70000,
      $37E78000,
      $37E80000,
      $37E88000,
      $37E90000,
      $37E98000,
      $37EA0000,
      $37EA8000,
      $37EB0000,
      $37EB8000,
      $37EC0000,
      $37EC8000,
      $37ED0000,
      $37ED8000,
      $37EE0000,
      $37EE8000,
      $37EF0000,
      $37EF8000,
      $37F00000,
      $37F08000,
      $37F10000,
      $37F18000,
      $37F20000,
      $37F28000,
      $37F30000,
      $37F38000,
      $37F40000,
      $37F48000,
      $37F50000,
      $37F58000,
      $37F60000,
      $37F68000,
      $37F70000,
      $37F78000,
      $37F80000,
      $37F88000,
      $37F90000,
      $37F98000,
      $37FA0000,
      $37FA8000,
      $37FB0000,
      $37FB8000,
      $37FC0000,
      $37FC8000,
      $37FD0000,
      $37FD8000,
      $37FE0000,
      $37FE8000,
      $37FF0000,
      $37FF8000,
      $38000000,
      $38004000,
      $38008000,
      $3800C000,
      $38010000,
      $38014000,
      $38018000,
      $3801C000,
      $38020000,
      $38024000,
      $38028000,
      $3802C000,
      $38030000,
      $38034000,
      $38038000,
      $3803C000,
      $38040000,
      $38044000,
      $38048000,
      $3804C000,
      $38050000,
      $38054000,
      $38058000,
      $3805C000,
      $38060000,
      $38064000,
      $38068000,
      $3806C000,
      $38070000,
      $38074000,
      $38078000,
      $3807C000,
      $38080000,
      $38084000,
      $38088000,
      $3808C000,
      $38090000,
      $38094000,
      $38098000,
      $3809C000,
      $380A0000,
      $380A4000,
      $380A8000,
      $380AC000,
      $380B0000,
      $380B4000,
      $380B8000,
      $380BC000,
      $380C0000,
      $380C4000,
      $380C8000,
      $380CC000,
      $380D0000,
      $380D4000,
      $380D8000,
      $380DC000,
      $380E0000,
      $380E4000,
      $380E8000,
      $380EC000,
      $380F0000,
      $380F4000,
      $380F8000,
      $380FC000,
      $38100000,
      $38104000,
      $38108000,
      $3810C000,
      $38110000,
      $38114000,
      $38118000,
      $3811C000,
      $38120000,
      $38124000,
      $38128000,
      $3812C000,
      $38130000,
      $38134000,
      $38138000,
      $3813C000,
      $38140000,
      $38144000,
      $38148000,
      $3814C000,
      $38150000,
      $38154000,
      $38158000,
      $3815C000,
      $38160000,
      $38164000,
      $38168000,
      $3816C000,
      $38170000,
      $38174000,
      $38178000,
      $3817C000,
      $38180000,
      $38184000,
      $38188000,
      $3818C000,
      $38190000,
      $38194000,
      $38198000,
      $3819C000,
      $381A0000,
      $381A4000,
      $381A8000,
      $381AC000,
      $381B0000,
      $381B4000,
      $381B8000,
      $381BC000,
      $381C0000,
      $381C4000,
      $381C8000,
      $381CC000,
      $381D0000,
      $381D4000,
      $381D8000,
      $381DC000,
      $381E0000,
      $381E4000,
      $381E8000,
      $381EC000,
      $381F0000,
      $381F4000,
      $381F8000,
      $381FC000,
      $38200000,
      $38204000,
      $38208000,
      $3820C000,
      $38210000,
      $38214000,
      $38218000,
      $3821C000,
      $38220000,
      $38224000,
      $38228000,
      $3822C000,
      $38230000,
      $38234000,
      $38238000,
      $3823C000,
      $38240000,
      $38244000,
      $38248000,
      $3824C000,
      $38250000,
      $38254000,
      $38258000,
      $3825C000,
      $38260000,
      $38264000,
      $38268000,
      $3826C000,
      $38270000,
      $38274000,
      $38278000,
      $3827C000,
      $38280000,
      $38284000,
      $38288000,
      $3828C000,
      $38290000,
      $38294000,
      $38298000,
      $3829C000,
      $382A0000,
      $382A4000,
      $382A8000,
      $382AC000,
      $382B0000,
      $382B4000,
      $382B8000,
      $382BC000,
      $382C0000,
      $382C4000,
      $382C8000,
      $382CC000,
      $382D0000,
      $382D4000,
      $382D8000,
      $382DC000,
      $382E0000,
      $382E4000,
      $382E8000,
      $382EC000,
      $382F0000,
      $382F4000,
      $382F8000,
      $382FC000,
      $38300000,
      $38304000,
      $38308000,
      $3830C000,
      $38310000,
      $38314000,
      $38318000,
      $3831C000,
      $38320000,
      $38324000,
      $38328000,
      $3832C000,
      $38330000,
      $38334000,
      $38338000,
      $3833C000,
      $38340000,
      $38344000,
      $38348000,
      $3834C000,
      $38350000,
      $38354000,
      $38358000,
      $3835C000,
      $38360000,
      $38364000,
      $38368000,
      $3836C000,
      $38370000,
      $38374000,
      $38378000,
      $3837C000,
      $38380000,
      $38384000,
      $38388000,
      $3838C000,
      $38390000,
      $38394000,
      $38398000,
      $3839C000,
      $383A0000,
      $383A4000,
      $383A8000,
      $383AC000,
      $383B0000,
      $383B4000,
      $383B8000,
      $383BC000,
      $383C0000,
      $383C4000,
      $383C8000,
      $383CC000,
      $383D0000,
      $383D4000,
      $383D8000,
      $383DC000,
      $383E0000,
      $383E4000,
      $383E8000,
      $383EC000,
      $383F0000,
      $383F4000,
      $383F8000,
      $383FC000,
      $38400000,
      $38404000,
      $38408000,
      $3840C000,
      $38410000,
      $38414000,
      $38418000,
      $3841C000,
      $38420000,
      $38424000,
      $38428000,
      $3842C000,
      $38430000,
      $38434000,
      $38438000,
      $3843C000,
      $38440000,
      $38444000,
      $38448000,
      $3844C000,
      $38450000,
      $38454000,
      $38458000,
      $3845C000,
      $38460000,
      $38464000,
      $38468000,
      $3846C000,
      $38470000,
      $38474000,
      $38478000,
      $3847C000,
      $38480000,
      $38484000,
      $38488000,
      $3848C000,
      $38490000,
      $38494000,
      $38498000,
      $3849C000,
      $384A0000,
      $384A4000,
      $384A8000,
      $384AC000,
      $384B0000,
      $384B4000,
      $384B8000,
      $384BC000,
      $384C0000,
      $384C4000,
      $384C8000,
      $384CC000,
      $384D0000,
      $384D4000,
      $384D8000,
      $384DC000,
      $384E0000,
      $384E4000,
      $384E8000,
      $384EC000,
      $384F0000,
      $384F4000,
      $384F8000,
      $384FC000,
      $38500000,
      $38504000,
      $38508000,
      $3850C000,
      $38510000,
      $38514000,
      $38518000,
      $3851C000,
      $38520000,
      $38524000,
      $38528000,
      $3852C000,
      $38530000,
      $38534000,
      $38538000,
      $3853C000,
      $38540000,
      $38544000,
      $38548000,
      $3854C000,
      $38550000,
      $38554000,
      $38558000,
      $3855C000,
      $38560000,
      $38564000,
      $38568000,
      $3856C000,
      $38570000,
      $38574000,
      $38578000,
      $3857C000,
      $38580000,
      $38584000,
      $38588000,
      $3858C000,
      $38590000,
      $38594000,
      $38598000,
      $3859C000,
      $385A0000,
      $385A4000,
      $385A8000,
      $385AC000,
      $385B0000,
      $385B4000,
      $385B8000,
      $385BC000,
      $385C0000,
      $385C4000,
      $385C8000,
      $385CC000,
      $385D0000,
      $385D4000,
      $385D8000,
      $385DC000,
      $385E0000,
      $385E4000,
      $385E8000,
      $385EC000,
      $385F0000,
      $385F4000,
      $385F8000,
      $385FC000,
      $38600000,
      $38604000,
      $38608000,
      $3860C000,
      $38610000,
      $38614000,
      $38618000,
      $3861C000,
      $38620000,
      $38624000,
      $38628000,
      $3862C000,
      $38630000,
      $38634000,
      $38638000,
      $3863C000,
      $38640000,
      $38644000,
      $38648000,
      $3864C000,
      $38650000,
      $38654000,
      $38658000,
      $3865C000,
      $38660000,
      $38664000,
      $38668000,
      $3866C000,
      $38670000,
      $38674000,
      $38678000,
      $3867C000,
      $38680000,
      $38684000,
      $38688000,
      $3868C000,
      $38690000,
      $38694000,
      $38698000,
      $3869C000,
      $386A0000,
      $386A4000,
      $386A8000,
      $386AC000,
      $386B0000,
      $386B4000,
      $386B8000,
      $386BC000,
      $386C0000,
      $386C4000,
      $386C8000,
      $386CC000,
      $386D0000,
      $386D4000,
      $386D8000,
      $386DC000,
      $386E0000,
      $386E4000,
      $386E8000,
      $386EC000,
      $386F0000,
      $386F4000,
      $386F8000,
      $386FC000,
      $38700000,
      $38704000,
      $38708000,
      $3870C000,
      $38710000,
      $38714000,
      $38718000,
      $3871C000,
      $38720000,
      $38724000,
      $38728000,
      $3872C000,
      $38730000,
      $38734000,
      $38738000,
      $3873C000,
      $38740000,
      $38744000,
      $38748000,
      $3874C000,
      $38750000,
      $38754000,
      $38758000,
      $3875C000,
      $38760000,
      $38764000,
      $38768000,
      $3876C000,
      $38770000,
      $38774000,
      $38778000,
      $3877C000,
      $38780000,
      $38784000,
      $38788000,
      $3878C000,
      $38790000,
      $38794000,
      $38798000,
      $3879C000,
      $387A0000,
      $387A4000,
      $387A8000,
      $387AC000,
      $387B0000,
      $387B4000,
      $387B8000,
      $387BC000,
      $387C0000,
      $387C4000,
      $387C8000,
      $387CC000,
      $387D0000,
      $387D4000,
      $387D8000,
      $387DC000,
      $387E0000,
      $387E4000,
      $387E8000,
      $387EC000,
      $387F0000,
      $387F4000,
      $387F8000,
      $387FC000,
      $38000000,
      $38002000,
      $38004000,
      $38006000,
      $38008000,
      $3800A000,
      $3800C000,
      $3800E000,
      $38010000,
      $38012000,
      $38014000,
      $38016000,
      $38018000,
      $3801A000,
      $3801C000,
      $3801E000,
      $38020000,
      $38022000,
      $38024000,
      $38026000,
      $38028000,
      $3802A000,
      $3802C000,
      $3802E000,
      $38030000,
      $38032000,
      $38034000,
      $38036000,
      $38038000,
      $3803A000,
      $3803C000,
      $3803E000,
      $38040000,
      $38042000,
      $38044000,
      $38046000,
      $38048000,
      $3804A000,
      $3804C000,
      $3804E000,
      $38050000,
      $38052000,
      $38054000,
      $38056000,
      $38058000,
      $3805A000,
      $3805C000,
      $3805E000,
      $38060000,
      $38062000,
      $38064000,
      $38066000,
      $38068000,
      $3806A000,
      $3806C000,
      $3806E000,
      $38070000,
      $38072000,
      $38074000,
      $38076000,
      $38078000,
      $3807A000,
      $3807C000,
      $3807E000,
      $38080000,
      $38082000,
      $38084000,
      $38086000,
      $38088000,
      $3808A000,
      $3808C000,
      $3808E000,
      $38090000,
      $38092000,
      $38094000,
      $38096000,
      $38098000,
      $3809A000,
      $3809C000,
      $3809E000,
      $380A0000,
      $380A2000,
      $380A4000,
      $380A6000,
      $380A8000,
      $380AA000,
      $380AC000,
      $380AE000,
      $380B0000,
      $380B2000,
      $380B4000,
      $380B6000,
      $380B8000,
      $380BA000,
      $380BC000,
      $380BE000,
      $380C0000,
      $380C2000,
      $380C4000,
      $380C6000,
      $380C8000,
      $380CA000,
      $380CC000,
      $380CE000,
      $380D0000,
      $380D2000,
      $380D4000,
      $380D6000,
      $380D8000,
      $380DA000,
      $380DC000,
      $380DE000,
      $380E0000,
      $380E2000,
      $380E4000,
      $380E6000,
      $380E8000,
      $380EA000,
      $380EC000,
      $380EE000,
      $380F0000,
      $380F2000,
      $380F4000,
      $380F6000,
      $380F8000,
      $380FA000,
      $380FC000,
      $380FE000,
      $38100000,
      $38102000,
      $38104000,
      $38106000,
      $38108000,
      $3810A000,
      $3810C000,
      $3810E000,
      $38110000,
      $38112000,
      $38114000,
      $38116000,
      $38118000,
      $3811A000,
      $3811C000,
      $3811E000,
      $38120000,
      $38122000,
      $38124000,
      $38126000,
      $38128000,
      $3812A000,
      $3812C000,
      $3812E000,
      $38130000,
      $38132000,
      $38134000,
      $38136000,
      $38138000,
      $3813A000,
      $3813C000,
      $3813E000,
      $38140000,
      $38142000,
      $38144000,
      $38146000,
      $38148000,
      $3814A000,
      $3814C000,
      $3814E000,
      $38150000,
      $38152000,
      $38154000,
      $38156000,
      $38158000,
      $3815A000,
      $3815C000,
      $3815E000,
      $38160000,
      $38162000,
      $38164000,
      $38166000,
      $38168000,
      $3816A000,
      $3816C000,
      $3816E000,
      $38170000,
      $38172000,
      $38174000,
      $38176000,
      $38178000,
      $3817A000,
      $3817C000,
      $3817E000,
      $38180000,
      $38182000,
      $38184000,
      $38186000,
      $38188000,
      $3818A000,
      $3818C000,
      $3818E000,
      $38190000,
      $38192000,
      $38194000,
      $38196000,
      $38198000,
      $3819A000,
      $3819C000,
      $3819E000,
      $381A0000,
      $381A2000,
      $381A4000,
      $381A6000,
      $381A8000,
      $381AA000,
      $381AC000,
      $381AE000,
      $381B0000,
      $381B2000,
      $381B4000,
      $381B6000,
      $381B8000,
      $381BA000,
      $381BC000,
      $381BE000,
      $381C0000,
      $381C2000,
      $381C4000,
      $381C6000,
      $381C8000,
      $381CA000,
      $381CC000,
      $381CE000,
      $381D0000,
      $381D2000,
      $381D4000,
      $381D6000,
      $381D8000,
      $381DA000,
      $381DC000,
      $381DE000,
      $381E0000,
      $381E2000,
      $381E4000,
      $381E6000,
      $381E8000,
      $381EA000,
      $381EC000,
      $381EE000,
      $381F0000,
      $381F2000,
      $381F4000,
      $381F6000,
      $381F8000,
      $381FA000,
      $381FC000,
      $381FE000,
      $38200000,
      $38202000,
      $38204000,
      $38206000,
      $38208000,
      $3820A000,
      $3820C000,
      $3820E000,
      $38210000,
      $38212000,
      $38214000,
      $38216000,
      $38218000,
      $3821A000,
      $3821C000,
      $3821E000,
      $38220000,
      $38222000,
      $38224000,
      $38226000,
      $38228000,
      $3822A000,
      $3822C000,
      $3822E000,
      $38230000,
      $38232000,
      $38234000,
      $38236000,
      $38238000,
      $3823A000,
      $3823C000,
      $3823E000,
      $38240000,
      $38242000,
      $38244000,
      $38246000,
      $38248000,
      $3824A000,
      $3824C000,
      $3824E000,
      $38250000,
      $38252000,
      $38254000,
      $38256000,
      $38258000,
      $3825A000,
      $3825C000,
      $3825E000,
      $38260000,
      $38262000,
      $38264000,
      $38266000,
      $38268000,
      $3826A000,
      $3826C000,
      $3826E000,
      $38270000,
      $38272000,
      $38274000,
      $38276000,
      $38278000,
      $3827A000,
      $3827C000,
      $3827E000,
      $38280000,
      $38282000,
      $38284000,
      $38286000,
      $38288000,
      $3828A000,
      $3828C000,
      $3828E000,
      $38290000,
      $38292000,
      $38294000,
      $38296000,
      $38298000,
      $3829A000,
      $3829C000,
      $3829E000,
      $382A0000,
      $382A2000,
      $382A4000,
      $382A6000,
      $382A8000,
      $382AA000,
      $382AC000,
      $382AE000,
      $382B0000,
      $382B2000,
      $382B4000,
      $382B6000,
      $382B8000,
      $382BA000,
      $382BC000,
      $382BE000,
      $382C0000,
      $382C2000,
      $382C4000,
      $382C6000,
      $382C8000,
      $382CA000,
      $382CC000,
      $382CE000,
      $382D0000,
      $382D2000,
      $382D4000,
      $382D6000,
      $382D8000,
      $382DA000,
      $382DC000,
      $382DE000,
      $382E0000,
      $382E2000,
      $382E4000,
      $382E6000,
      $382E8000,
      $382EA000,
      $382EC000,
      $382EE000,
      $382F0000,
      $382F2000,
      $382F4000,
      $382F6000,
      $382F8000,
      $382FA000,
      $382FC000,
      $382FE000,
      $38300000,
      $38302000,
      $38304000,
      $38306000,
      $38308000,
      $3830A000,
      $3830C000,
      $3830E000,
      $38310000,
      $38312000,
      $38314000,
      $38316000,
      $38318000,
      $3831A000,
      $3831C000,
      $3831E000,
      $38320000,
      $38322000,
      $38324000,
      $38326000,
      $38328000,
      $3832A000,
      $3832C000,
      $3832E000,
      $38330000,
      $38332000,
      $38334000,
      $38336000,
      $38338000,
      $3833A000,
      $3833C000,
      $3833E000,
      $38340000,
      $38342000,
      $38344000,
      $38346000,
      $38348000,
      $3834A000,
      $3834C000,
      $3834E000,
      $38350000,
      $38352000,
      $38354000,
      $38356000,
      $38358000,
      $3835A000,
      $3835C000,
      $3835E000,
      $38360000,
      $38362000,
      $38364000,
      $38366000,
      $38368000,
      $3836A000,
      $3836C000,
      $3836E000,
      $38370000,
      $38372000,
      $38374000,
      $38376000,
      $38378000,
      $3837A000,
      $3837C000,
      $3837E000,
      $38380000,
      $38382000,
      $38384000,
      $38386000,
      $38388000,
      $3838A000,
      $3838C000,
      $3838E000,
      $38390000,
      $38392000,
      $38394000,
      $38396000,
      $38398000,
      $3839A000,
      $3839C000,
      $3839E000,
      $383A0000,
      $383A2000,
      $383A4000,
      $383A6000,
      $383A8000,
      $383AA000,
      $383AC000,
      $383AE000,
      $383B0000,
      $383B2000,
      $383B4000,
      $383B6000,
      $383B8000,
      $383BA000,
      $383BC000,
      $383BE000,
      $383C0000,
      $383C2000,
      $383C4000,
      $383C6000,
      $383C8000,
      $383CA000,
      $383CC000,
      $383CE000,
      $383D0000,
      $383D2000,
      $383D4000,
      $383D6000,
      $383D8000,
      $383DA000,
      $383DC000,
      $383DE000,
      $383E0000,
      $383E2000,
      $383E4000,
      $383E6000,
      $383E8000,
      $383EA000,
      $383EC000,
      $383EE000,
      $383F0000,
      $383F2000,
      $383F4000,
      $383F6000,
      $383F8000,
      $383FA000,
      $383FC000,
      $383FE000,
      $38400000,
      $38402000,
      $38404000,
      $38406000,
      $38408000,
      $3840A000,
      $3840C000,
      $3840E000,
      $38410000,
      $38412000,
      $38414000,
      $38416000,
      $38418000,
      $3841A000,
      $3841C000,
      $3841E000,
      $38420000,
      $38422000,
      $38424000,
      $38426000,
      $38428000,
      $3842A000,
      $3842C000,
      $3842E000,
      $38430000,
      $38432000,
      $38434000,
      $38436000,
      $38438000,
      $3843A000,
      $3843C000,
      $3843E000,
      $38440000,
      $38442000,
      $38444000,
      $38446000,
      $38448000,
      $3844A000,
      $3844C000,
      $3844E000,
      $38450000,
      $38452000,
      $38454000,
      $38456000,
      $38458000,
      $3845A000,
      $3845C000,
      $3845E000,
      $38460000,
      $38462000,
      $38464000,
      $38466000,
      $38468000,
      $3846A000,
      $3846C000,
      $3846E000,
      $38470000,
      $38472000,
      $38474000,
      $38476000,
      $38478000,
      $3847A000,
      $3847C000,
      $3847E000,
      $38480000,
      $38482000,
      $38484000,
      $38486000,
      $38488000,
      $3848A000,
      $3848C000,
      $3848E000,
      $38490000,
      $38492000,
      $38494000,
      $38496000,
      $38498000,
      $3849A000,
      $3849C000,
      $3849E000,
      $384A0000,
      $384A2000,
      $384A4000,
      $384A6000,
      $384A8000,
      $384AA000,
      $384AC000,
      $384AE000,
      $384B0000,
      $384B2000,
      $384B4000,
      $384B6000,
      $384B8000,
      $384BA000,
      $384BC000,
      $384BE000,
      $384C0000,
      $384C2000,
      $384C4000,
      $384C6000,
      $384C8000,
      $384CA000,
      $384CC000,
      $384CE000,
      $384D0000,
      $384D2000,
      $384D4000,
      $384D6000,
      $384D8000,
      $384DA000,
      $384DC000,
      $384DE000,
      $384E0000,
      $384E2000,
      $384E4000,
      $384E6000,
      $384E8000,
      $384EA000,
      $384EC000,
      $384EE000,
      $384F0000,
      $384F2000,
      $384F4000,
      $384F6000,
      $384F8000,
      $384FA000,
      $384FC000,
      $384FE000,
      $38500000,
      $38502000,
      $38504000,
      $38506000,
      $38508000,
      $3850A000,
      $3850C000,
      $3850E000,
      $38510000,
      $38512000,
      $38514000,
      $38516000,
      $38518000,
      $3851A000,
      $3851C000,
      $3851E000,
      $38520000,
      $38522000,
      $38524000,
      $38526000,
      $38528000,
      $3852A000,
      $3852C000,
      $3852E000,
      $38530000,
      $38532000,
      $38534000,
      $38536000,
      $38538000,
      $3853A000,
      $3853C000,
      $3853E000,
      $38540000,
      $38542000,
      $38544000,
      $38546000,
      $38548000,
      $3854A000,
      $3854C000,
      $3854E000,
      $38550000,
      $38552000,
      $38554000,
      $38556000,
      $38558000,
      $3855A000,
      $3855C000,
      $3855E000,
      $38560000,
      $38562000,
      $38564000,
      $38566000,
      $38568000,
      $3856A000,
      $3856C000,
      $3856E000,
      $38570000,
      $38572000,
      $38574000,
      $38576000,
      $38578000,
      $3857A000,
      $3857C000,
      $3857E000,
      $38580000,
      $38582000,
      $38584000,
      $38586000,
      $38588000,
      $3858A000,
      $3858C000,
      $3858E000,
      $38590000,
      $38592000,
      $38594000,
      $38596000,
      $38598000,
      $3859A000,
      $3859C000,
      $3859E000,
      $385A0000,
      $385A2000,
      $385A4000,
      $385A6000,
      $385A8000,
      $385AA000,
      $385AC000,
      $385AE000,
      $385B0000,
      $385B2000,
      $385B4000,
      $385B6000,
      $385B8000,
      $385BA000,
      $385BC000,
      $385BE000,
      $385C0000,
      $385C2000,
      $385C4000,
      $385C6000,
      $385C8000,
      $385CA000,
      $385CC000,
      $385CE000,
      $385D0000,
      $385D2000,
      $385D4000,
      $385D6000,
      $385D8000,
      $385DA000,
      $385DC000,
      $385DE000,
      $385E0000,
      $385E2000,
      $385E4000,
      $385E6000,
      $385E8000,
      $385EA000,
      $385EC000,
      $385EE000,
      $385F0000,
      $385F2000,
      $385F4000,
      $385F6000,
      $385F8000,
      $385FA000,
      $385FC000,
      $385FE000,
      $38600000,
      $38602000,
      $38604000,
      $38606000,
      $38608000,
      $3860A000,
      $3860C000,
      $3860E000,
      $38610000,
      $38612000,
      $38614000,
      $38616000,
      $38618000,
      $3861A000,
      $3861C000,
      $3861E000,
      $38620000,
      $38622000,
      $38624000,
      $38626000,
      $38628000,
      $3862A000,
      $3862C000,
      $3862E000,
      $38630000,
      $38632000,
      $38634000,
      $38636000,
      $38638000,
      $3863A000,
      $3863C000,
      $3863E000,
      $38640000,
      $38642000,
      $38644000,
      $38646000,
      $38648000,
      $3864A000,
      $3864C000,
      $3864E000,
      $38650000,
      $38652000,
      $38654000,
      $38656000,
      $38658000,
      $3865A000,
      $3865C000,
      $3865E000,
      $38660000,
      $38662000,
      $38664000,
      $38666000,
      $38668000,
      $3866A000,
      $3866C000,
      $3866E000,
      $38670000,
      $38672000,
      $38674000,
      $38676000,
      $38678000,
      $3867A000,
      $3867C000,
      $3867E000,
      $38680000,
      $38682000,
      $38684000,
      $38686000,
      $38688000,
      $3868A000,
      $3868C000,
      $3868E000,
      $38690000,
      $38692000,
      $38694000,
      $38696000,
      $38698000,
      $3869A000,
      $3869C000,
      $3869E000,
      $386A0000,
      $386A2000,
      $386A4000,
      $386A6000,
      $386A8000,
      $386AA000,
      $386AC000,
      $386AE000,
      $386B0000,
      $386B2000,
      $386B4000,
      $386B6000,
      $386B8000,
      $386BA000,
      $386BC000,
      $386BE000,
      $386C0000,
      $386C2000,
      $386C4000,
      $386C6000,
      $386C8000,
      $386CA000,
      $386CC000,
      $386CE000,
      $386D0000,
      $386D2000,
      $386D4000,
      $386D6000,
      $386D8000,
      $386DA000,
      $386DC000,
      $386DE000,
      $386E0000,
      $386E2000,
      $386E4000,
      $386E6000,
      $386E8000,
      $386EA000,
      $386EC000,
      $386EE000,
      $386F0000,
      $386F2000,
      $386F4000,
      $386F6000,
      $386F8000,
      $386FA000,
      $386FC000,
      $386FE000,
      $38700000,
      $38702000,
      $38704000,
      $38706000,
      $38708000,
      $3870A000,
      $3870C000,
      $3870E000,
      $38710000,
      $38712000,
      $38714000,
      $38716000,
      $38718000,
      $3871A000,
      $3871C000,
      $3871E000,
      $38720000,
      $38722000,
      $38724000,
      $38726000,
      $38728000,
      $3872A000,
      $3872C000,
      $3872E000,
      $38730000,
      $38732000,
      $38734000,
      $38736000,
      $38738000,
      $3873A000,
      $3873C000,
      $3873E000,
      $38740000,
      $38742000,
      $38744000,
      $38746000,
      $38748000,
      $3874A000,
      $3874C000,
      $3874E000,
      $38750000,
      $38752000,
      $38754000,
      $38756000,
      $38758000,
      $3875A000,
      $3875C000,
      $3875E000,
      $38760000,
      $38762000,
      $38764000,
      $38766000,
      $38768000,
      $3876A000,
      $3876C000,
      $3876E000,
      $38770000,
      $38772000,
      $38774000,
      $38776000,
      $38778000,
      $3877A000,
      $3877C000,
      $3877E000,
      $38780000,
      $38782000,
      $38784000,
      $38786000,
      $38788000,
      $3878A000,
      $3878C000,
      $3878E000,
      $38790000,
      $38792000,
      $38794000,
      $38796000,
      $38798000,
      $3879A000,
      $3879C000,
      $3879E000,
      $387A0000,
      $387A2000,
      $387A4000,
      $387A6000,
      $387A8000,
      $387AA000,
      $387AC000,
      $387AE000,
      $387B0000,
      $387B2000,
      $387B4000,
      $387B6000,
      $387B8000,
      $387BA000,
      $387BC000,
      $387BE000,
      $387C0000,
      $387C2000,
      $387C4000,
      $387C6000,
      $387C8000,
      $387CA000,
      $387CC000,
      $387CE000,
      $387D0000,
      $387D2000,
      $387D4000,
      $387D6000,
      $387D8000,
      $387DA000,
      $387DC000,
      $387DE000,
      $387E0000,
      $387E2000,
      $387E4000,
      $387E6000,
      $387E8000,
      $387EA000,
      $387EC000,
      $387EE000,
      $387F0000,
      $387F2000,
      $387F4000,
      $387F6000,
      $387F8000,
      $387FA000,
      $387FC000,
      $387FE000
  );
const
  EXPONENT_TABLE: array[0..63] of LongWord = (
      $00000000,
      $00800000,
      $01000000,
      $01800000,
      $02000000,
      $02800000,
      $03000000,
      $03800000,
      $04000000,
      $04800000,
      $05000000,
      $05800000,
      $06000000,
      $06800000,
      $07000000,
      $07800000,
      $08000000,
      $08800000,
      $09000000,
      $09800000,
      $0A000000,
      $0A800000,
      $0B000000,
      $0B800000,
      $0C000000,
      $0C800000,
      $0D000000,
      $0D800000,
      $0E000000,
      $0E800000,
      $0F000000,
      $47800000,
      $80000000,
      $80800000,
      $81000000,
      $81800000,
      $82000000,
      $82800000,
      $83000000,
      $83800000,
      $84000000,
      $84800000,
      $85000000,
      $85800000,
      $86000000,
      $86800000,
      $87000000,
      $87800000,
      $88000000,
      $88800000,
      $89000000,
      $89800000,
      $8A000000,
      $8A800000,
      $8B000000,
      $8B800000,
      $8C000000,
      $8C800000,
      $8D000000,
      $8D800000,
      $8E000000,
      $8E800000,
      $8F000000,
      $C7800000
  );
const
  OFFSET_TABLE: array[0..63] of Word = (
      0,
      $400,
      $400,
      $400,
      $400,
      $400,
      $400,
      $400,
      $400,
      $400,
      $400,
      $400,
      $400,
      $400,
      $400,
      $400,
      $400,
      $400,
      $400,
      $400,
      $400,
      $400,
      $400,
      $400,
      $400,
      $400,
      $400,
      $400,
      $400,
      $400,
      $400,
      $400,
      0,
      $400,
      $400,
      $400,
      $400,
      $400,
      $400,
      $400,
      $400,
      $400,
      $400,
      $400,
      $400,
      $400,
      $400,
      $400,
      $400,
      $400,
      $400,
      $400,
      $400,
      $400,
      $400,
      $400,
      $400,
      $400,
      $400,
      $400,
      $400,
      $400,
      $400,
      $400
  );
var
  Bits: ConversionBits;
  I: NativeInt;
  Value: Word;
begin
  for I := 0 to N - 1 do
  begin
    Value := Word(Src[I]);
    Bits.I32 := MANTISSA_TABLE[OFFSET_TABLE[Value shr 10] + (Value and $3FF)] + EXPONENT_TABLE[Value shr 10];
    Dst[I] := Bits.F32;
  end;
end;

procedure SingleToFP16(const N: NativeInt; const Src: System.PSingle; Dst: PFP16);
const
  BASE_TABLE: array[0..511] of Word = (
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0000,
      $0001,
      $0002,
      $0004,
      $0008,
      $0010,
      $0020,
      $0040,
      $0080,
      $0100,
      $0200,
      $0400,
      $0800,
      $0C00,
      $1000,
      $1400,
      $1800,
      $1C00,
      $2000,
      $2400,
      $2800,
      $2C00,
      $3000,
      $3400,
      $3800,
      $3C00,
      $4000,
      $4400,
      $4800,
      $4C00,
      $5000,
      $5400,
      $5800,
      $5C00,
      $6000,
      $6400,
      $6800,
      $6C00,
      $7000,
      $7400,
      $7800,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $7C00,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8000,
      $8001,
      $8002,
      $8004,
      $8008,
      $8010,
      $8020,
      $8040,
      $8080,
      $8100,
      $8200,
      $8400,
      $8800,
      $8C00,
      $9000,
      $9400,
      $9800,
      $9C00,
      $A000,
      $A400,
      $A800,
      $AC00,
      $B000,
      $B400,
      $B800,
      $BC00,
      $C000,
      $C400,
      $C800,
      $CC00,
      $D000,
      $D400,
      $D800,
      $DC00,
      $E000,
      $E400,
      $E800,
      $EC00,
      $F000,
      $F400,
      $F800,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00,
      $FC00
  );
const
  SHIFT_TABLE: array[0..511] of Byte = (
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      23,
      22,
      21,
      20,
      19,
      18,
      17,
      16,
      15,
      14,
      13,
      13,
      13,
      13,
      13,
      13,
      13,
      13,
      13,
      13,
      13,
      13,
      13,
      13,
      13,
      13,
      13,
      13,
      13,
      13,
      13,
      13,
      13,
      13,
      13,
      13,
      13,
      13,
      13,
      13,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      13,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      23,
      22,
      21,
      20,
      19,
      18,
      17,
      16,
      15,
      14,
      13,
      13,
      13,
      13,
      13,
      13,
      13,
      13,
      13,
      13,
      13,
      13,
      13,
      13,
      13,
      13,
      13,
      13,
      13,
      13,
      13,
      13,
      13,
      13,
      13,
      13,
      13,
      13,
      13,
      13,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      24,
      13
  );
var
  Bits: ConversionBits;
  I: NativeInt;
begin
  for I := 0 to N - 1 do
  begin
    Bits.F32 := Src[I];
    PWord(Dst)[I] := BASE_TABLE[Bits.I32 shr 23] + Word((Bits.I32 and $7FFFFF) shr SHIFT_TABLE[Bits.I32 shr 23]);
  end
end;

function Ifthen(const Cond: Boolean; const IfTrue, IfFalse: LongInt): LongInt;
begin
  if Cond then
    Result := IfTrue
  else
    Result := IfFalse
end;

function Ifthen(const Cond: Boolean; const IfTrue, IfFalse: Int64): Int64;
begin
  if Cond then
    Result := IfTrue
  else
    Result := IfFalse
end;

function Ifthen(const Cond: Boolean; const IfTrue, IfFalse: Single): Single;
begin
  if Cond then
    Result := IfTrue
  else
    Result := IfFalse
end;

function Ifthen(const Cond: Boolean; const IfTrue, IfFalse: Double): Double;
begin
  if Cond then
    Result := IfTrue
  else
    Result := IfFalse
end;

function Ifthen(const Cond: Boolean; const IfTrue, IfFalse: string): string;
begin
  if Cond then
    Result := IfTrue
  else
    Result := IfFalse
end;

{ TMemoryArena }

constructor TMemoryArena.Create(const APool: NativeUInt);
begin
  FData := AllocMem(APool);
  Assert(Assigned(FData), 'ERROR Creating Memory Arena' + SLineBreak + 'Not enough memory!');
{$IFDEF DEBUG}
  WriteLn('Create : RefCount ', RefCount);
  ReadLn;
{$ENDIF}
  FMemPos := 0;
end;

function TMemoryArena.ArenaSize: NativeUInt;
begin
  Result := FSize
end;

procedure TMemoryArena.Resize(const ASize: NativeUInt);
begin
  ReallocMem(FData, ASize);
  Assert(Assigned(FData), 'ERROR Resizing Memory Arena' + SLineBreak + 'Not enough memory!');
  FSize := ASize
end;

function TMemoryArena.Allocate(const ASize: NativeUInt): Pointer;
begin
  Assert(FMemPos + ASize <= ArenaSize, 'ERROR not enough memory size in the memory arena');
  Inc(FMemPos, ASize);
  Result := PByte(FData) + FMemPos
end;

destructor TMemoryArena.Destroy;
begin
  FreeMem(FData);
{$IFDEF DEBUG}
  WriteLn('Destroy RefCount ', RefCount);
  ReadLn;
{$ENDIF}
  inherited Destroy;
end;

initialization
  TQNNImage.ComputePngCRCTable;

end.
