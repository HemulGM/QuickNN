unit QuickNN.Vulkan.Operations;

interface

uses
  System.SysUtils,
  System.Generics.Collections,
  QuickNN.Common,
  QuickNN.Vulkan;

type
  // A lexical workspace returns temporary activations to the context's VRAM pool.
  // It never downloads data; inputs are borrowed and returned results are detached.
  TGPUWorkspace = class
  private
    FItems: TList<TQNNGPUTensor>;
  public
    Ctx: TQNNGPUContext;
    constructor Create(const Context: TQNNGPUContext);
    destructor Destroy; override;
    function Own(const Tensor: TQNNGPUTensor): TQNNGPUTensor;
    function New(const Count: Int64): TQNNGPUTensor;
    function Keep(const Tensor: TQNNGPUTensor): TQNNGPUTensor;
    procedure Release(var Tensor: TQNNGPUTensor);
    function Copy(const Src: TQNNGPUTensor; const Count: Integer; const Offset: Integer = 0): TQNNGPUTensor;
    function Linear(
        const Src: TQNNGPUTensor;
        const Weight: TMemoryBlock;
        const Rows, InputDim, OutputDim: Integer;
        const Bias: TMemoryBlock
    ): TQNNGPUTensor; overload;
    function Linear(
        const Src: TQNNGPUTensor;
        const Weight: TMemoryBlock;
        const Rows, InputDim, OutputDim: Integer
    ): TQNNGPUTensor; overload;
    function Norm(
        const Src: TQNNGPUTensor;
        const Weight, Bias: TMemoryBlock;
        const Rows, Dim: Integer;
        const Center: Boolean = False;
        const Epsilon: Single = 1E-6
    ): TQNNGPUTensor;
    function Unary(const Src: TQNNGPUTensor; const Count: Integer; const Kind: Integer = 0): TQNNGPUTensor;
    function Slice(const Src: TQNNGPUTensor; const Rows, Width, Stride, Offset: Integer): TQNNGPUTensor;
    function Join(const A, B: TQNNGPUTensor; const Rows, WidthA, WidthB: Integer): TQNNGPUTensor;
    function Concat(const A, B: TQNNGPUTensor; const CountA, CountB: Integer): TQNNGPUTensor;
    function CHWToRows(const Src: TQNNGPUTensor; const Spatial, Channels: Integer): TQNNGPUTensor;
    function RowsToCHW(const Src: TQNNGPUTensor; const Spatial, Channels: Integer): TQNNGPUTensor;
    function Modulate(
        const Src, Modulation: TQNNGPUTensor;
        const Rows, Dim, Kind, ScaleOffset: Integer;
        const ShiftOffset: Integer = 0;
        const Dst: TQNNGPUTensor = nil
    ): TQNNGPUTensor;
    procedure Add(const Dst, Src: TQNNGPUTensor; const Count: Integer; const Scale: Single = 1);
    function SwiGLU(
        const Src: TQNNGPUTensor;
        const Gate, Up, Down: TMemoryBlock;
        const Rows, Hidden, Intermediate: Integer
    ): TQNNGPUTensor;
    function Conv(
        const Src: TQNNGPUTensor;
        const Weight, Bias: TMemoryBlock;
        const H, W, InCh, OutCh, Kernel: Integer;
        const Stride: Integer = 1;
        const Padding: Integer = 1;
        const PadEnd: Boolean = False
    ): TQNNGPUTensor;
    function GroupNorm(
        const Src: TQNNGPUTensor;
        const Weight, Bias: TMemoryBlock;
        const Channels, Spatial, Groups: Integer;
        const Epsilon: Single = 1E-6
    ): TQNNGPUTensor;
    function Patch(const Src: TQNNGPUTensor; const H, W, Channels, PatchSize, Mode: Integer): TQNNGPUTensor;
    function PadRows(
        const Src: TQNNGPUTensor;
        const Token: TMemoryBlock;
        const Rows, PaddedRows, Dim: Integer;
        const RepeatLast: Boolean = False
    ): TQNNGPUTensor;
    function Sinusoidal(const Time: Single; const Dim: Integer): TQNNGPUTensor;
  end;

function GPUWeight(const F32, BF16: TMemoryBlock): TMemoryBlock;
function GPUDownloadRGB(const Ctx: TQNNGPUContext; const CHW: TQNNGPUTensor; const Height, Width: Integer): TQNNImage;

implementation

uses
  System.Math;

constructor TGPUWorkspace.Create(const Context: TQNNGPUContext);
begin
  inherited Create;
  Ctx := Context;
  FItems := TList<TQNNGPUTensor>.Create;
end;

destructor TGPUWorkspace.Destroy;
var
  Tensor: TQNNGPUTensor;
begin
  for var I := 0 to FItems.Count - 1 do
  begin
    Tensor := FItems[I];
    Ctx.Recycle(Tensor);
  end;
  FItems.Free;
  inherited;
end;

function TGPUWorkspace.Own(const Tensor: TQNNGPUTensor): TQNNGPUTensor;
begin
  if Tensor = nil then
    raise EArgumentNilException.Create('GPU workspace tensor');
  FItems.Add(Tensor);
  Result := Tensor;
end;

function TGPUWorkspace.New(const Count: Int64): TQNNGPUTensor;
begin
  if (Count < 1) or (Count > MaxInt) then
    raise EArgumentOutOfRangeException.Create('GPU activation size');
  Result := Own(Ctx.NewTensor(Count));
end;

function TGPUWorkspace.Keep(const Tensor: TQNNGPUTensor): TQNNGPUTensor;
begin
  if FItems.Remove(Tensor) < 0 then
    raise EArgumentException.Create('Only owned activations can escape a workspace');
  Result := Tensor;
end;

procedure TGPUWorkspace.Release(var Tensor: TQNNGPUTensor);
begin
  if Tensor = nil then
    Exit;
  Keep(Tensor);
  Ctx.Recycle(Tensor);
end;

function TGPUWorkspace.Copy(const Src: TQNNGPUTensor; const Count, Offset: Integer): TQNNGPUTensor;
begin
  Result := New(Count);
  Ctx.Copy(Result, Src, Count, 0, Offset);
end;

function TGPUWorkspace.Linear(
    const Src: TQNNGPUTensor;
    const Weight: TMemoryBlock;
    const Rows, InputDim, OutputDim: Integer;
    const Bias: TMemoryBlock
): TQNNGPUTensor;
begin
  Result := New(Int64(Rows) * OutputDim);
  Ctx.Linear(Result, Src, Weight, Default(TMemoryBlock), Bias, Rows, InputDim, OutputDim);
end;

function TGPUWorkspace.Linear(
    const Src: TQNNGPUTensor;
    const Weight: TMemoryBlock;
    const Rows, InputDim, OutputDim: Integer
): TQNNGPUTensor;
begin
  Result := Linear(Src, Weight, Rows, InputDim, OutputDim, Default(TMemoryBlock));
end;

function TGPUWorkspace.Norm(
    const Src: TQNNGPUTensor;
    const Weight, Bias: TMemoryBlock;
    const Rows, Dim: Integer;
    const Center: Boolean;
    const Epsilon: Single
): TQNNGPUTensor;
var
  Flags: Integer;
begin
  Result := New(Int64(Rows) * Dim);
  if (Src.Count < Result.Count)
      or (Weight.IsAssigned and (Weight.Count < Dim))
      or (Bias.IsAssigned and (Bias.Count < Dim))
      or (Epsilon <= 0) then
    raise EArgumentException.Create('Invalid resident normalization');
  Flags := Ord(Center) or (Ord(not Bias.IsAssigned) * 8);
  Ctx.KernelWeights(
      GO_NORM,
      Result.Count,
      Result,
      Src,
      Weight,
      Bias,
      [Rows, Dim, Flags, 0, 0, 0, Ord(not Weight.IsAssigned), QNNGPUFloatBits(Epsilon)]
  );
end;

function TGPUWorkspace.Unary(const Src: TQNNGPUTensor; const Count, Kind: Integer): TQNNGPUTensor;
begin
  Result := New(Count);
  Ctx.Kernel(GO_UNARY, Count, Result, Src, nil, nil, [Kind]);
end;

function TGPUWorkspace.Slice(const Src: TQNNGPUTensor; const Rows, Width, Stride, Offset: Integer): TQNNGPUTensor;
begin
  if (Width < 1) or (Stride < Width) or (Offset < 0) or (Int64(Rows - 1) * Stride + Offset + Width > Src.Count) then
    raise EArgumentException.Create('Invalid resident strided slice');
  Result := New(Int64(Rows) * Width);
  Ctx.Kernel(GO_SLICE, Result.Count, Result, Src, nil, nil, [Width, Stride, Offset, 0]);
end;

function TGPUWorkspace.Join(const A, B: TQNNGPUTensor; const Rows, WidthA, WidthB: Integer): TQNNGPUTensor;
begin
  Result := New(Int64(Rows) * (WidthA + WidthB));
  Ctx.Kernel(GO_SCATTER, Rows * WidthA, Result, A, nil, nil, [WidthA, WidthA + WidthB, 0, 0]);
  Ctx.Kernel(GO_SCATTER, Rows * WidthB, Result, B, nil, nil, [WidthB, WidthA + WidthB, WidthA, 0]);
end;

function TGPUWorkspace.Concat(const A, B: TQNNGPUTensor; const CountA, CountB: Integer): TQNNGPUTensor;
begin
  Result := New(Int64(CountA) + CountB);
  Ctx.Copy(Result, A, CountA);
  Ctx.Copy(Result, B, CountB, CountA);
end;

function TGPUWorkspace.CHWToRows(const Src: TQNNGPUTensor; const Spatial, Channels: Integer): TQNNGPUTensor;
begin
  Result := New(Int64(Spatial) * Channels);
  Ctx.Kernel(GO_CHW_TO_ROWS, Result.Count, Result, Src, nil, nil, [Spatial, Channels, 0]);
end;

function TGPUWorkspace.RowsToCHW(const Src: TQNNGPUTensor; const Spatial, Channels: Integer): TQNNGPUTensor;
begin
  Result := New(Int64(Spatial) * Channels);
  Ctx.Kernel(GO_ROWS_TO_CHW, Result.Count, Result, Src, nil, nil, [Spatial, Channels]);
end;

function TGPUWorkspace.Modulate(
    const Src, Modulation: TQNNGPUTensor;
    const Rows, Dim, Kind, ScaleOffset, ShiftOffset: Integer;
    const Dst: TQNNGPUTensor
): TQNNGPUTensor;
begin
  Result := Dst;
  if Result = nil then
    Result := New(Int64(Rows) * Dim);
  Ctx.Kernel(GO_AFFINE, Rows * Dim, Result, Src, Modulation, nil, [Dim, Kind, ScaleOffset, ShiftOffset, 0, 0]);
end;

procedure TGPUWorkspace.Add(const Dst, Src: TQNNGPUTensor; const Count: Integer; const Scale: Single);
begin
  Ctx.Kernel(GO_ADD, Count, Dst, Src, nil, nil, [1, 1, QNNGPUFloatBits(Scale), 0, 0]);
end;

function TGPUWorkspace.SwiGLU(
    const Src: TQNNGPUTensor;
    const Gate, Up, Down: TMemoryBlock;
    const Rows, Hidden, Intermediate: Integer
): TQNNGPUTensor;
var
  Work: TGPUWorkspace;
  G, U: TQNNGPUTensor;
begin
  Work := TGPUWorkspace.Create(Ctx);
  try
    G := Work.Linear(Src, Gate, Rows, Hidden, Intermediate);
    U := Work.Linear(Src, Up, Rows, Hidden, Intermediate);
    Ctx.Kernel(GO_SWIGLU, Rows * Intermediate, G, G, U, nil, [Intermediate, 0]);
    Result := Linear(G, Down, Rows, Intermediate, Hidden);
  finally
    Work.Free;
  end;
end;

function TGPUWorkspace.Conv(
    const Src: TQNNGPUTensor;
    const Weight, Bias: TMemoryBlock;
    const H, W, InCh, OutCh, Kernel, Stride, Padding: Integer;
    const PadEnd: Boolean
): TQNNGPUTensor;
const
  TILE = 1024;
var
  Work: TGPUWorkspace;
  Input, Col, Projected: TQNNGPUTensor;
  IH, IW, OH, OW, Spatial, Inner, Start, Count: Integer;
begin
  Work := TGPUWorkspace.Create(Ctx);
  try
    Input := Src;
    IH := H;
    IW := W;
    if PadEnd then
    begin
      Input := Work.New(Int64(InCh) * (H + 1) * (W + 1));
      Ctx.Kernel(GO_PAD_END, Input.Count, Input, Src, nil, nil, [H, W]);
      Inc(IH);
      Inc(IW);
    end;
    OH := (IH + 2 * Padding - Kernel) div Stride + 1;
    OW := (IW + 2 * Padding - Kernel) div Stride + 1;
    Spatial := OH * OW;
    Inner := InCh * Kernel * Kernel;
    Result := New(Int64(OutCh) * Spatial);
    Col := Work.New(Int64(Min(TILE, Spatial)) * Inner);
    Projected := Work.New(Int64(Min(TILE, Spatial)) * OutCh);
    Start := 0;
    while Start < Spatial do
    begin
      Count := Min(TILE, Spatial - Start);
      Ctx.Kernel(GO_IM2COL, Count * Inner, Col, Input, nil, nil, [IH, IW, InCh, Kernel, Start, OW, Stride, Padding]);
      Ctx.Linear(Projected, Col, Weight, Default(TMemoryBlock), Default(TMemoryBlock), Count, Inner, OutCh);
      Ctx.KernelWeights(
          GO_CONV_OUTPUT,
          Count * OutCh,
          Result,
          Projected,
          Bias,
          Default(TMemoryBlock),
          [OutCh, Spatial, Start, Ord(Bias.IsAssigned)]
      );
      Inc(Start, Count);
    end;
  finally
    Work.Free;
  end;
end;

function TGPUWorkspace.GroupNorm(
    const Src: TQNNGPUTensor;
    const Weight, Bias: TMemoryBlock;
    const Channels, Spatial, Groups: Integer;
    const Epsilon: Single
): TQNNGPUTensor;
begin
  if (Groups < 1) or (Channels mod Groups <> 0) then
    raise EArgumentException.Create('Invalid resident GroupNorm');
  Result := New(Int64(Channels) * Spatial);
  Ctx.KernelWeights(
      GO_GROUP_NORM,
      Result.Count,
      Result,
      Src,
      Weight,
      Bias,
      [Groups, Channels, Spatial, QNNGPUFloatBits(Epsilon)]
  );
end;

function TGPUWorkspace.Patch(const Src: TQNNGPUTensor; const H, W, Channels, PatchSize, Mode: Integer): TQNNGPUTensor;
begin
  if (PatchSize < 1) or (H mod PatchSize <> 0) or (W mod PatchSize <> 0) then
    raise EArgumentException.Create('Invalid resident patch dimensions');
  Result := New(Int64(H) * W * Channels);
  Ctx.Kernel(GO_PATCH, Result.Count, Result, Src, nil, nil, [H, W, Channels, PatchSize, Mode]);
end;

function TGPUWorkspace.PadRows(
    const Src: TQNNGPUTensor;
    const Token: TMemoryBlock;
    const Rows, PaddedRows, Dim: Integer;
    const RepeatLast: Boolean
): TQNNGPUTensor;
begin
  Result := New(Int64(PaddedRows) * Dim);
  Ctx.KernelWeights(
      GO_PAD_WEIGHT,
      Result.Count,
      Result,
      Src,
      Token,
      Default(TMemoryBlock),
      [Dim, Rows, 0, Ord(RepeatLast)]
  );
end;

function TGPUWorkspace.Sinusoidal(const Time: Single; const Dim: Integer): TQNNGPUTensor;
begin
  Result := New(Dim);
  Ctx.Kernel(GO_SINUSOIDAL, Dim, Result, nil, nil, nil, [QNNGPUFloatBits(Time)]);
end;

function GPUWeight(const F32, BF16: TMemoryBlock): TMemoryBlock;
begin
  if BF16.IsAssigned then
    Result := BF16
  else
    Result := F32;
end;

function GPUDownloadRGB(const Ctx: TQNNGPUContext; const CHW: TQNNGPUTensor; const Height, Width: Integer): TQNNImage;
var
  Work: TGPUWorkspace;
  PackedPixels: TQNNGPUTensor;
  Pixels: TArray<UInt32>;
begin
  Work := TGPUWorkspace.Create(Ctx);
  try
    PackedPixels := Work.New(Int64(Height) * Width);
    Ctx.CheckFinite(CHW, Height * Width * 3);
    Ctx.Kernel(GO_RGB_OUTPUT, Height * Width, PackedPixels, CHW, nil, nil, [Height * Width]);
    SetLength(Pixels, Height * Width);
    Ctx.Download(PackedPixels, @Pixels[0], Length(Pixels));
    Result := TQNNImage.Create(Width, Height, 3);
    // Final pixel packing for the public RGB image/PNG interface, after one readback.
    for var I := 0 to High(Pixels) do
    begin
      Result.Data[I * 3] := Pixels[I] and $FF;
      Result.Data[I * 3 + 1] := (Pixels[I] shr 8) and $FF;
      Result.Data[I * 3 + 2] := (Pixels[I] shr 16) and $FF;
    end;
  finally
    Work.Free;
  end;
end;

end.
