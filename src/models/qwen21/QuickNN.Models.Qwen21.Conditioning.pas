unit QuickNN.Models.Qwen21.Conditioning;
{ SPDX-License-Identifier: Apache-2.0
  Native image preparation and joint-sequence layout. Multimodal conventions
  follow Qwen / Hugging Face / ComfyUI (Apache-2.0).
  See LICENSE.QwenImage21.txt in the repository root. }
{$POINTERMATH ON}
interface
uses
  QuickNN.Common,
  QuickNN.Models.Qwen21.Tokenizer,
  QuickNN.Vulkan;
type
  TQwen21GPUImageCondition = record
    Pixels, Latents, Visual: TQNNGPUTensor;
    DeepStack: array[0..2] of TQNNGPUTensor;
    procedure Free(const Ctx: TQNNGPUContext);
  end;
  TQwen21GPUImageConditions = TArray<TQwen21GPUImageCondition>;
  TQwen21ImageCondition = record
    Pixels: TQNNImage;
    Latents, Visual: TMemoryBlock;
    DeepStack: array[0..2] of TMemoryBlock;
    GridHeight, GridWidth: Integer;
    procedure Free;
    function VisualTokens: Integer;
  end;
  TQwen21ImageConditions = TArray<TQwen21ImageCondition>;
  TQwen21TextImageSpan = record
    ImageIndex, TokenStart, TokenCount: Integer;
  end;
  TQwen21TextImageSpans = TArray<TQwen21TextImageSpan>;
type
  TQwen21SequenceSegment = record
    IsImage: Boolean;
    SourceIndex, SourceStart, JointStart, Count: Integer;
  end;
  TQwen21SequenceLayout = record
    Segments: TArray<TQwen21SequenceSegment>;
    Positions: TArray<Single>;
    KeyLimits: TArray<Integer>;
    Rows, PrefixRows: Integer;
  end;
function Q21ReferenceLayout(
    const TextRows: Integer;
    const Images: TQwen21ImageConditions;
    const Slots: TArray<Integer>;
    const TargetHeight, TargetWidth: Integer
): TQwen21SequenceLayout;
procedure Q21BuildImagePrompt(
    const Tokenizer: TQwen21Tokenizer;
    const Prompt: string;
    const Images: TQwen21ImageConditions;
    out Ids, PositionIds: TArray<Integer>;
    out Spans: TQwen21TextImageSpans;
    out DropTokens: Integer
);
function Q21LoadReference(const FileName: string; const MaxSide: Integer): TQNNImage;
function Q21ResizeRGBA(const Image: TQNNImage; const Width, Height: Integer): TQNNImage;
implementation
uses
  System.SysUtils,
  System.Math,
  System.Generics.Collections,
  Winapi.Windows,
  Winapi.ActiveX,
  Winapi.Wincodec,
  System.Win.ComObj;

function Q21ReferenceLayout(
    const TextRows: Integer;
    const Images: TQwen21ImageConditions;
    const Slots: TArray<Integer>;
    const TargetHeight, TargetWidth: Integer
): TQwen21SequenceLayout;
var
  Total: Int64;
  Start, Stop, Joint, Position, H, W, SegmentIndex: Integer;
  Segment: TQwen21SequenceSegment;
  procedure AddSegment(const Image: Boolean; const Source, SourceStart, Count: Integer);
  begin
    if Count = 0 then
      Exit;
    Segment.IsImage := Image;
    Segment.SourceIndex := Source;
    Segment.SourceStart := SourceStart;
    Segment.JointStart := Joint;
    Segment.Count := Count;
    Result.Segments[SegmentIndex] := Segment;
    Inc(SegmentIndex);
  end;
begin
  Result := Default(TQwen21SequenceLayout);
  if (TextRows < 1) or (TargetHeight < 1) or (TargetWidth < 1) or (Length(Slots) <> Length(Images)) then
    raise EArgumentException.Create('Invalid reference layout');
  Total := Int64(TextRows) + Int64(TargetHeight) * TargetWidth;
  Start := 0;
  for var I := 0 to High(Images) do
  begin
    if (Slots[I] < Start)
        or (Slots[I] > TextRows)
        or (Images[I].Pixels.Width mod 16 <> 0)
        or (Images[I].Pixels.Height mod 16 <> 0)
        or (Images[I].Pixels.Width < 16)
        or (Images[I].Pixels.Height < 16) then
      raise EArgumentException.Create('Invalid reference image slot/dimensions');
    Start := Slots[I];
    Inc(Total, Int64(Images[I].Pixels.Width div 16) * (Images[I].Pixels.Height div 16));
  end;
  if Total > 6144 then
    raise EArgumentException.Create('Joint sequence exceeds 6144 tokens; reduce image/reference sizes');
  Result.Rows := Total;
  Result.PrefixRows := Total - Int64(TargetHeight) * TargetWidth;
  SetLength(Result.Positions, Total * 3);
  SetLength(Result.KeyLimits, Total);
  SetLength(Result.Segments, 2 * (Length(Images) + 1));
  Joint := 0;
  Position := 0;
  Start := 0;
  SegmentIndex := 0;
  for var I := 0 to Length(Images) do
  begin
    if I < Length(Images) then
      Stop := Slots[I]
    else
      Stop := TextRows;
    AddSegment(False, -1, Start, Stop - Start);
    for var J := Start to Stop - 1 do
    begin
      for var Axis := 0 to 2 do
        Result.Positions[Joint * 3 + Axis] := Position;
      Result.KeyLimits[Joint] := Joint + 1;
      Inc(Joint);
      Inc(Position);
    end;
    if I < Length(Images) then
    begin
      H := Images[I].Pixels.Height div 16;
      W := Images[I].Pixels.Width div 16
    end
    else
    begin
      H := TargetHeight;
      W := TargetWidth
    end;
    if I < Length(Images) then
      AddSegment(True, I, 0, H * W)
    else
      AddSegment(True, -1, 0, H * W);
    var BlockEnd := Joint + H * W;
    for var Y := 0 to H - 1 do
      for var X := 0 to W - 1 do
      begin
        Result.Positions[Joint * 3] := Position;
        Result.Positions[Joint * 3 + 1] := Y - (H - H div 2) + 0.5 * (H mod 2 - TargetHeight mod 2);
        Result.Positions[Joint * 3 + 2] := X - (W - W div 2) + 0.5 * (W mod 2 - TargetWidth mod 2);
        Result.KeyLimits[Joint] := BlockEnd;
        Inc(Joint);
      end;
    Inc(Position, Max(H, W));
    Start := Stop;
  end;
  SetLength(Result.Segments, SegmentIndex);
end;

procedure Q21BuildImagePrompt(
    const Tokenizer: TQwen21Tokenizer;
    const Prompt: string;
    const Images: TQwen21ImageConditions;
    out Ids, PositionIds: TArray<Integer>;
    out Spans: TQwen21TextImageSpans;
    out DropTokens: Integer
);
var
  References: string;
  BaseIds: TArray<Integer>;
  Tokens, Pt, Ph, Pw: TList<Integer>;
  ImageIndex, Position, H, W: Integer;
  Span: TQwen21TextImageSpan;
begin
  if Length(Images) = 0 then
  begin
    Ids := Tokenizer.Chat(Prompt, DropTokens);
    PositionIds := nil;
    Spans := nil;
    Exit
  end;
  References := '';
  for var I := 0 to High(Images) do
  begin
    if I > 0 then
      References := References + ' ';
    References := References + Format('<image%d><|vision_start|><|image_pad|><|vision_end|>', [I + 1]);
  end;
  BaseIds := Tokenizer.Chat(References + Prompt, DropTokens);
  Tokens := TList<Integer>.Create;
  Pt := TList<Integer>.Create;
  Ph := TList<Integer>.Create;
  Pw := TList<Integer>.Create;
  try
    ImageIndex := 0;
    Position := 0;
    SetLength(Spans, Length(Images));
    for var Token in BaseIds do
    begin
      if Token = 151655 then
      begin
        if ImageIndex >= Length(Images) then
          raise EArgumentException.Create('More image placeholders than references');
        H := Images[ImageIndex].GridHeight div 2;
        W := Images[ImageIndex].GridWidth div 2;
        if (H < 1)
            or (W < 1)
            or (Images[ImageIndex].GridHeight mod 2 <> 0)
            or (Images[ImageIndex].GridWidth mod 2 <> 0) then
          raise EArgumentException.Create('Invalid visual merge grid');
        Span.ImageIndex := ImageIndex;
        Span.TokenStart := Tokens.Count;
        Span.TokenCount := H * W;
        Spans[ImageIndex] := Span;
        for var Y := 0 to H - 1 do
          for var X := 0 to W - 1 do
          begin
            Tokens.Add(151655);
            Pt.Add(Position);
            Ph.Add(Position + Y);
            Pw.Add(Position + X);
          end;
        Inc(Position, Max(H, W));
        Inc(ImageIndex);
      end
      else
      begin
        Tokens.Add(Token);
        Pt.Add(Position);
        Ph.Add(Position);
        Pw.Add(Position);
        Inc(Position)
      end;
    end;
    if ImageIndex <> Length(Images) then
      raise EArgumentException.Create('Missing visual placeholders');
    Ids := Tokens.ToArray;
    SetLength(PositionIds, 3 * Tokens.Count);
    for var I := 0 to Tokens.Count - 1 do
    begin
      PositionIds[I] := Pt[I];
      PositionIds[Tokens.Count + I] := Ph[I];
      PositionIds[2 * Tokens.Count + I] := Pw[I]
    end;
  finally
    Tokens.Free;
    Pt.Free;
    Ph.Free;
    Pw.Free
  end;
end;

procedure TQwen21GPUImageCondition.Free(const Ctx: TQNNGPUContext);
begin
  Ctx.Recycle(Pixels);
  Ctx.Recycle(Latents);
  Ctx.Recycle(Visual);
  for var I := 0 to 2 do
    Ctx.Recycle(DeepStack[I])
end;

procedure TQwen21ImageCondition.Free;
begin
  Pixels.Free;
  Latents.Free;
  Visual.Free;
  for var I := 0 to 2 do
    DeepStack[I].Free;
  GridHeight := 0;
  GridWidth := 0
end;
function TQwen21ImageCondition.VisualTokens: Integer;
begin
  Result := GridHeight * GridWidth div 4
end;

function Q21ResizeRGBA(const Image: TQNNImage; const Width, Height: Integer): TQNNImage;
var
  Sx, Sy, Fx, Fy, Value: Double;
  X0, X1, Y0, Y1: Integer;
begin
  if (Image.Channels <> 4) or (Image.Width <= 0) or (Image.Height <= 0) or (Width <= 0) or (Height <= 0) then
    raise EArgumentException.Create('Invalid RGBA image/resize dimensions');
  Result := TQNNImage.Create(Width, Height, 4);
  for var Y := 0 to Height - 1 do
  begin
    Sy := EnsureRange((Y + 0.5) * Image.Height / Height - 0.5, Double(0), Double(Image.Height - 1));
    Y0 := Floor(Sy);
    Y1 := Min(Y0 + 1, Image.Height - 1);
    Fy := Sy - Y0;
    for var X := 0 to Width - 1 do
    begin
      Sx := EnsureRange((X + 0.5) * Image.Width / Width - 0.5, Double(0), Double(Image.Width - 1));
      X0 := Floor(Sx);
      X1 := Min(X0 + 1, Image.Width - 1);
      Fx := Sx - X0;
      for var C := 0 to 3 do
      begin
        Value :=
            (1 - Fy)
                    * ((1 - Fx) * Image.Data[(Y0 * Image.Width + X0) * 4 + C]
                        + Fx * Image.Data[(Y0 * Image.Width + X1) * 4 + C])
                + Fy
                    * ((1 - Fx) * Image.Data[(Y1 * Image.Width + X0) * 4 + C]
                        + Fx * Image.Data[(Y1 * Image.Width + X1) * 4 + C]);
        Result.Data[(Y * Width + X) * 4 + C] := EnsureRange(Round(Value), 0, 255);
      end;
    end;
  end;
end;

function Q21LoadReference(const FileName: string; const MaxSide: Integer): TQNNImage;
var
  Factory: IWICImagingFactory;
  Decoder: IWICBitmapDecoder;
  Frame: IWICBitmapFrameDecode;
  Converter: IWICFormatConverter;
  Hr: HRESULT;
  Uninit: Boolean;
  W, H: UINT;
  Image: TQNNImage;
  Scale: Double;
  OutW, OutH: Integer;
begin
  if (MaxSide < 64) or (MaxSide > 1024) or (MaxSide mod 32 <> 0) then
    raise EArgumentException.Create('Reference size must be 64..1024 divisible by 32');
  Hr := CoInitializeEx(nil, COINIT_MULTITHREADED);
  Uninit := Succeeded(Hr);
  if Failed(Hr) and (Hr <> RPC_E_CHANGED_MODE) then
    OleCheck(Hr);
  try
    OleCheck(CoCreateInstance(CLSID_WICImagingFactory, nil, CLSCTX_INPROC_SERVER, IWICImagingFactory, Factory));
    OleCheck(
        Factory.CreateDecoderFromFilename(
            PWideChar(FileName),
            GUID_VendorMicrosoft,
            GENERIC_READ,
            WICDecodeMetadataCacheOnLoad,
            Decoder
        )
    );
    OleCheck(Decoder.GetFrame(0, Frame));
    OleCheck(Frame.GetSize(W, H));
    if (W = 0) or (H = 0) or (UInt64(W) * H > 64 * 1024 * 1024) or (Max(W, H) / Min(W, H) > 200) then
      raise EArgumentException.Create('Reference image dimensions are unsupported');
    OleCheck(Factory.CreateFormatConverter(Converter));
    // Explicit straight RGBA preserves alpha; no external image library.
    OleCheck(
        Converter.Initialize(
            Frame,
            GUID_WICPixelFormat32bppRGBA,
            WICBitmapDitherTypeNone,
            nil,
            0,
            WICBitmapPaletteTypeCustom
        )
    );
    Image := TQNNImage.Create(W, H, 4);
    OleCheck(Converter.CopyPixels(nil, W * 4, Length(Image.Data), @Image.Data[0]));
    Scale := Min(1, Double(MaxSide) / Max(W, H));
    OutW := Max(32, Round(W * Scale / 32) * 32);
    OutH := Max(32, Round(H * Scale / 32) * 32);
    if (OutW = Integer(W)) and (OutH = Integer(H)) then
    begin
      Result := Image;
      Image := Default(TQNNImage)
    end
    else
      Result := Q21ResizeRGBA(Image, OutW, OutH);
  finally
    Image.Free;
    Converter := nil;
    Frame := nil;
    Decoder := nil;
    Factory := nil;
    if Uninit then
      CoUninitialize;
  end;
end;
end.
