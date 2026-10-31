unit LLMUI.Attachments;

interface

uses
  System.SysUtils, FMX.Graphics, LLMUI.Images;

const
  MAX_MESSAGE_IMAGES = 4;
  MAX_IMAGE_PIXELS = 64 * 1024 * 1024;

function IsImageFile(const FileName: string): Boolean;

function ImportImageFile(const ChatId, SourceFile: string): TChatImage;

function ImportImageBitmap(const ChatId, DisplayName: string; const Bitmap: FMX.Graphics.TBitmap): TChatImage;

function ClipboardImageFiles: TArray<string>;

procedure CopyImageToClipboard(const Bitmap: FMX.Graphics.TBitmap; const ClipboardOwner: NativeUInt);

function ReadClipboardImage(out Bitmap: FMX.Graphics.TBitmap): Boolean;

implementation

uses
  System.Classes, System.IOUtils, System.Hash, System.Math, System.Types,
  System.UIConsts, System.UITypes, Winapi.Windows, Winapi.ShellAPI,
  Winapi.ActiveX, Winapi.Wincodec, System.Win.ComObj, QuickNN.Common,
  QuickNN.Models.Qwen21.VAE;

const
  MAX_CLIPBOARD_BYTES = 300 * 1024 * 1024;

function IsImageFile(const FileName: string): Boolean;
begin
  var Ext := LowerCase(ExtractFileExt(FileName));
  Result := (Ext = '.png') or (Ext = '.jpg') or (Ext = '.jpeg') or (Ext = '.bmp')
    or (Ext = '.gif') or (Ext = '.tif') or (Ext = '.tiff') or (Ext = '.webp');
end;

function SaveImported(const ChatId, DisplayName: string; const Pixels: TQNNImage): TChatImage;
begin
  if (Pixels.Width < 1) or (Pixels.Height < 1) or (UInt64(Pixels.Width) * UInt64(Pixels.Height) > MAX_IMAGE_PIXELS) then
    raise EArgumentException.Create('Изображение должно содержать не более 64 мегапикселей.');
  Result := Default(TChatImage);
  Result.IsUserImage := True;
  Result.OriginalName := ExtractFileName(DisplayName);
  Result.ModelName := 'Ваше изображение';
  Result.Settings := TChatImageSettings.Defaults;
  Result.Settings.Width := Pixels.Width;
  Result.Settings.Height := Pixels.Height;
  Result.RelativePath := 'images/' + ChatId + '/' + TGUID.NewGuid.ToString + '.png';
  var Destination := Result.FileName;
  ForceDirectories(ExtractFileDir(Destination));
  var Temporary := Destination + '.tmp';
  try
    Q21SavePNG(Pixels, Temporary);
    var Source := TFileStream.Create(Temporary, fmOpenRead or fmShareDenyWrite);
    try
      Result.ContentHash := THashSHA2.GetHashString(Source)
    finally
      Source.Free
    end;
    TFile.Move(Temporary, Destination);
  finally
    if TFile.Exists(Temporary) then
      TFile.Delete(Temporary);
  end;
end;

function ImportImageFile(const ChatId, SourceFile: string): TChatImage;
var
  Factory: IWICImagingFactory;
  Decoder: IWICBitmapDecoder;
  Frame: IWICBitmapFrameDecode;
  Metadata: IWICMetadataQueryReader;
  Rotated: IWICBitmapFlipRotator;
  Source: IWICBitmapSource;
  Orientation: TPropVariant;
  Transform: DWORD;
  Converter: IWICFormatConverter;
  Pixels: TQNNImage;
  Width, Height: UINT;
  Hr: HRESULT;
begin
  if not IsImageFile(SourceFile) then
    raise EArgumentException.Create('Неподдерживаемый формат изображения: ' + ExtractFileName(SourceFile));
  if not TFile.Exists(SourceFile) then
    raise EFileNotFoundException.Create('Изображение не найдено: ' + SourceFile);
  Pixels := Default(TQNNImage);
  Hr := CoInitializeEx(nil, COINIT_APARTMENTTHREADED);
  if Failed(Hr) and (Hr <> RPC_E_CHANGED_MODE) then
    OleCheck(Hr);
  try
    OleCheck(CoCreateInstance(CLSID_WICImagingFactory, nil, CLSCTX_INPROC_SERVER, IWICImagingFactory, Factory));
    OleCheck(Factory.CreateDecoderFromFilename(PChar(SourceFile), GUID_VendorMicrosoft, GENERIC_READ,
        WICDecodeMetadataCacheOnLoad, Decoder));
    OleCheck(Decoder.GetFrame(0, Frame));
    OleCheck(Frame.GetSize(Width, Height));
    if (Width = 0) or (Height = 0) or (UInt64(Width) * Height > MAX_IMAGE_PIXELS)
      or (Max(Width, Height) / Min(Width, Height) > 200) then
      raise EArgumentException.Create('Недопустимый размер изображения: максимум 64 мегапикселя.');
    Source := Frame;
    Orientation := Default(TPropVariant);
    if Succeeded(Frame.GetMetadataQueryReader(Metadata)) then
    try
      var Query := Metadata.GetMetadataByName('/app1/ifd/{ushort=274}', Orientation);
      if Failed(Query) then
        Query := Metadata.GetMetadataByName('/ifd/{ushort=274}', Orientation);
      if Succeeded(Query) and (Orientation.vt = VT_UI2) then
      begin
        Transform := WICBitmapTransformRotate0;
        case Orientation.uiVal of
          2:
            Transform := WICBitmapTransformFlipHorizontal;
          3:
            Transform := WICBitmapTransformRotate180;
          4:
            Transform := WICBitmapTransformFlipVertical;
          5:
            Transform := WICBitmapTransformRotate90 or WICBitmapTransformFlipHorizontal;
          6:
            Transform := WICBitmapTransformRotate90;
          7:
            Transform := WICBitmapTransformRotate270 or WICBitmapTransformFlipHorizontal;
          8:
            Transform := WICBitmapTransformRotate270;
        end;
        if Transform <> WICBitmapTransformRotate0 then
        begin
          OleCheck(Factory.CreateBitmapFlipRotator(Rotated));
          OleCheck(Rotated.Initialize(Frame, Transform));
          Source := Rotated;
          OleCheck(Source.GetSize(Width, Height));
        end;
      end;
    finally
      PropVariantClear(Orientation)
    end;
    OleCheck(Factory.CreateFormatConverter(Converter));
    OleCheck(Converter.Initialize(Source, GUID_WICPixelFormat32bppRGBA, WICBitmapDitherTypeNone, nil, 0, WICBitmapPaletteTypeCustom));
    Pixels := TQNNImage.Create(Width, Height, 4);
    OleCheck(Converter.CopyPixels(nil, Width * 4, Length(Pixels.Data), @Pixels.Data[0]));
    Result := SaveImported(ChatId, SourceFile, Pixels);
  finally
    Pixels.Free;
    Converter := nil;
    Source := nil;
    Rotated := nil;
    Metadata := nil;
    Frame := nil;
    Decoder := nil;
    Factory := nil;
    if Succeeded(Hr) then
      CoUninitialize;
  end;
end;

function ImportImageBitmap(const ChatId, DisplayName: string; const Bitmap: FMX.Graphics.TBitmap): TChatImage;
var
  Map: TBitmapData;
  Pixels: TQNNImage;
begin
  if (Bitmap = nil) or (Bitmap.Width < 1) or (Bitmap.Height < 1)
    or (UInt64(Bitmap.Width) * UInt64(Bitmap.Height) > MAX_IMAGE_PIXELS) then
    raise EArgumentException.Create('Недопустимый размер изображения из буфера обмена.');
  Pixels := TQNNImage.Create(Bitmap.Width, Bitmap.Height, 4);
  try
    if not Bitmap.Map(TMapAccess.Read, Map) then
      raise EInvalidOp.Create('Не удалось прочитать изображение из буфера.');
    try
      for var Y := 0 to Bitmap.Height - 1 do
        for var X := 0 to Bitmap.Width - 1 do
        begin
          var Color := UnpremultiplyAlpha(Map.GetPixel(X, Y));
          var Offset := (Y * Bitmap.Width + X) * 4;
          Pixels.Data[Offset] := Byte(Color shr 16);
          Pixels.Data[Offset + 1] := Byte(Color shr 8);
          Pixels.Data[Offset + 2] := Byte(Color);
          Pixels.Data[Offset + 3] := Byte(Color shr 24);
        end;
    finally
      Bitmap.Unmap(Map)
    end;
    Result := SaveImported(ChatId, DisplayName, Pixels);
  finally
    Pixels.Free
  end;
end;

function ClipboardImageFiles: TArray<string>;
begin
  Result := nil;
  if not IsClipboardFormatAvailable(CF_HDROP) then
    Exit;
  if not OpenClipboard(0) then
    raise EInvalidOp.Create('Буфер обмена занят. Повторите вставку.');
  try
    var Handle := GetClipboardData(CF_HDROP);
    var Count := DragQueryFile(Handle, $FFFFFFFF, nil, 0);
    for var I := 0 to Integer(Count) - 1 do
    begin
      var NameLength := DragQueryFile(Handle, I, nil, 0);
      var Buffer: TArray<Char>;
      SetLength(Buffer, NameLength + 1);
      DragQueryFile(Handle, I, @Buffer[0], Length(Buffer));
      var FileName := string(PChar(@Buffer[0]));
      if IsImageFile(FileName) then
        Result := Result + [FileName];
      if Length(Result) > MAX_MESSAGE_IMAGES then
        Break;
    end;
  finally
    CloseClipboard
  end;
end;

procedure CopyImageToClipboard(const Bitmap: FMX.Graphics.TBitmap; const ClipboardOwner: NativeUInt);
var
  Png, Dib, DibAlpha: HGLOBAL;
  PngFormat: UINT;
  Stream: TMemoryStream;
  Info: TBitmapInfoHeader;
  InfoAlpha: TBitmapV5Header;
  Map: TBitmapData;
  Pixels, AlphaPixels: Pointer;
  PixelBytes: NativeUInt;
  Opened: Boolean;

  function AllocateBlock(const Size: NativeUInt): HGLOBAL;
  begin
    Result := GlobalAlloc(GMEM_MOVEABLE or GMEM_ZEROINIT, Size);
    if Result = 0 then
      RaiseLastOSError;
  end;

  function LockBlock(const Handle: HGLOBAL): Pointer;
  begin
    Result := GlobalLock(Handle);
    if Result = nil then
      RaiseLastOSError;
  end;

  procedure Publish(const Format: UINT; var Handle: HGLOBAL);
  begin
    if SetClipboardData(Format, Handle) = 0 then
      RaiseLastOSError;
    Handle := 0; // Windows now owns this allocation, independently of the viewer.
  end;

begin
  if (Bitmap = nil) or Bitmap.IsEmpty then
    Exit;
  if ClipboardOwner = 0 then
    raise EArgumentException.Create('Недоступно окно буфера обмена.');
  if UInt64(Bitmap.Width) * UInt64(Bitmap.Height) > MAX_IMAGE_PIXELS then
    raise EArgumentException.Create('Слишком большое изображение для буфера обмена.');
  Png := 0;
  Dib := 0;
  DibAlpha := 0;
  Stream := TMemoryStream.Create;
  try
    // Prepare every representation before replacing the user's clipboard.
    Bitmap.SaveToStream(Stream);
    if (Stream.Size = 0) or (Stream.Size > MAX_CLIPBOARD_BYTES) then
      raise EArgumentException.Create('Недопустимый размер изображения для буфера обмена.');
    PngFormat := RegisterClipboardFormat('PNG');
    if PngFormat = 0 then
      RaiseLastOSError;
    Png := AllocateBlock(Stream.Size);
    var PngData := LockBlock(Png);
    try
      Move(Stream.Memory^, PngData^, Stream.Size)
    finally
      GlobalUnlock(Png)
    end;

    PixelBytes := NativeUInt(Bitmap.Width) * NativeUInt(Bitmap.Height) * 4;
    Info := Default(TBitmapInfoHeader);
    Info.biSize := SizeOf(Info);
    Info.biWidth := Bitmap.Width;
    Info.biHeight := -Bitmap.Height;
    Info.biPlanes := 1;
    Info.biBitCount := 32;
    Info.biSizeImage := PixelBytes;
    Info.biCompression := BI_RGB;
    InfoAlpha := Default(TBitmapV5Header);
    InfoAlpha.bV5Size := SizeOf(InfoAlpha);
    InfoAlpha.bV5Width := Bitmap.Width;
    InfoAlpha.bV5Height := -Bitmap.Height;
    InfoAlpha.bV5Planes := 1;
    InfoAlpha.bV5BitCount := 32;
    InfoAlpha.bV5SizeImage := PixelBytes;
    InfoAlpha.bV5Compression := BI_BITFIELDS;
    InfoAlpha.bV5RedMask := $00FF0000;
    InfoAlpha.bV5GreenMask := $0000FF00;
    InfoAlpha.bV5BlueMask := $000000FF;
    InfoAlpha.bV5AlphaMask := $FF000000;
    InfoAlpha.bV5CSType := $73524742; // LCS_sRGB
    InfoAlpha.bV5Intent := 4; // LCS_GM_IMAGES
    Dib := AllocateBlock(SizeOf(Info) + PixelBytes);
    DibAlpha := AllocateBlock(SizeOf(InfoAlpha) + PixelBytes);
    Pixels := LockBlock(Dib);
    try
      AlphaPixels := LockBlock(DibAlpha);
      try
        Move(Info, Pixels^, SizeOf(Info));
        Move(InfoAlpha, AlphaPixels^, SizeOf(InfoAlpha));
        if not Bitmap.Map(TMapAccess.Read, Map) then
          raise EInvalidOp.Create('Не удалось прочитать изображение.');
        try
          for var Y := 0 to Bitmap.Height - 1 do
            for var X := 0 to Bitmap.Width - 1 do
            begin
              var Color := TAlphaColorRec(Map.GetPixel(X, Y));
              var Offset := (NativeUInt(Y) * NativeUInt(Bitmap.Width) + NativeUInt(X)) * 4;
              PCardinal(NativeUInt(AlphaPixels) + SizeOf(InfoAlpha) + Offset)^ := UnpremultiplyAlpha(Color.Color);
              // Legacy DIB consumers have no reliable alpha; use a white backdrop.
              Color.R := Min(255, Integer(Color.R) + 255 - Color.A);
              Color.G := Min(255, Integer(Color.G) + 255 - Color.A);
              Color.B := Min(255, Integer(Color.B) + 255 - Color.A);
              Color.A := 255;
              PCardinal(NativeUInt(Pixels) + SizeOf(Info) + Offset)^ := Color.Color;
            end;
        finally
          Bitmap.Unmap(Map)
        end;
      finally
        GlobalUnlock(DibAlpha)
      end;
    finally
      GlobalUnlock(Dib)
    end;

    for var Attempt := 1 to 50 do
    begin
      Opened := OpenClipboard(HWND(ClipboardOwner));
      if Opened then
        Break;
      Sleep(10);
    end;
    if not Opened then
      raise EInvalidOp.Create('Буфер обмена занят. Повторите копирование.');
    try
      if not EmptyClipboard then
        RaiseLastOSError;
      Publish(PngFormat, Png);
      Publish(CF_DIBV5, DibAlpha);
      Publish(CF_DIB, Dib);
    finally
      CloseClipboard
    end;
  finally
    if Png <> 0 then
      GlobalFree(Png);
    if DibAlpha <> 0 then
      GlobalFree(DibAlpha);
    if Dib <> 0 then
      GlobalFree(Dib);
    Stream.Free;
  end;
end;

function ReadClipboardImage(out Bitmap: FMX.Graphics.TBitmap): Boolean;
var
  Handle: THandle;
  Ptr: Pointer;
  Stream: TMemoryStream;
  Size: NativeUInt;
  Header: TBitmapFileHeader;
  Info: TBitmapInfoHeader;
  Format: UINT;
begin
  Bitmap := nil;
  Result := False;
  Format := RegisterClipboardFormat('PNG');
  if not IsClipboardFormatAvailable(Format) then
    if IsClipboardFormatAvailable(CF_DIBV5) then
      Format := CF_DIBV5
    else if IsClipboardFormatAvailable(CF_DIB) then
      Format := CF_DIB
    else
      Exit;
  if not OpenClipboard(0) then
    raise EInvalidOp.Create('Буфер обмена занят. Повторите вставку.');
  Stream := TMemoryStream.Create;
  try
    Handle := GetClipboardData(Format);
    Size := GlobalSize(Handle);
    if (Size = 0) or (Size > MAX_CLIPBOARD_BYTES) then
      raise EArgumentException.Create('Недопустимый размер данных буфера обмена.');
    Ptr := GlobalLock(Handle);
    if Ptr = nil then
      RaiseLastOSError;
    try
      if (Format = CF_DIB) or (Format = CF_DIBV5) then
      begin
        if Size < SizeOf(Info) then
          raise EArgumentException.Create('Неполное изображение в буфере обмена.');
        Move(Ptr^, Info, SizeOf(Info));
        if (Info.biSize < SizeOf(Info)) or (Info.biSize > Size) or (Info.biWidth < 1)
          or (Info.biHeight = 0) or (UInt64(Info.biWidth) * UInt64(Abs(Int64(Info.biHeight))) > MAX_IMAGE_PIXELS) then
          raise EArgumentException.Create('Недопустимое изображение в буфере обмена.');
        Header := Default(TBitmapFileHeader);
        Header.bfType := $4D42;
        Header.bfSize := Size + SizeOf(Header);
        Header.bfOffBits := SizeOf(Header) + Info.biSize;
        if (Info.biCompression = BI_BITFIELDS) and (Info.biSize = 40) then
          Inc(Header.bfOffBits, 12);
        if Info.biClrUsed > 0 then
          Inc(Header.bfOffBits, Info.biClrUsed * 4)
        else if Info.biBitCount <= 8 then
          Inc(Header.bfOffBits, (1 shl Info.biBitCount) * 4);
        if Header.bfOffBits > Header.bfSize then
          raise EArgumentException.Create('Неполное изображение в буфере обмена.');
        Stream.WriteBuffer(Header, SizeOf(Header));
      end;
      Stream.WriteBuffer(Ptr^, Size);
    finally
      GlobalUnlock(Handle)
    end;
    Stream.Position := 0;
    Bitmap := FMX.Graphics.TBitmap.Create;
    try
      Bitmap.LoadFromStream(Stream);
      if (Bitmap.Width < 1) or (Bitmap.Height < 1) or (UInt64(Bitmap.Width) * UInt64(Bitmap.Height) > MAX_IMAGE_PIXELS) then
        raise EArgumentException.Create('Недопустимый размер изображения из буфера обмена.');
      Result := True;
    except
      FreeAndNil(Bitmap);
      raise
    end;
  finally
    Stream.Free;
    CloseClipboard
  end;
end;

end.

