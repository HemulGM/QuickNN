unit QuickNN.Console.Sixel;

{$POINTERMATH on}

interface

uses
  System.Classes,
  System.SysUtils,
  System.Math,
  System.Generics.Collections;

type
  TPixelOrder = (poHWC, poCHW);

function EncodeSixel(
    const Buf: Pointer;
    const Width, Height: NativeInt;
    const Dither: Boolean = False;
    const MonoChrome: Boolean = False;
    const PixelOrder: TPixelOrder = poHWC
): RawByteString;

procedure PrintSixel(
    const Buf: Pointer;
    const Width, Height: NativeInt;
    const Dither: Boolean = False;
    const MonoChrome: Boolean = False;
    const PixelOrder: TPixelOrder = poHWC
);

implementation

type

  { TSixel }

  TSixel = packed record
    R, G, B: Byte;
    constructor Create(const AR, AG, AB: Byte); overload;
    constructor Create(const L: Byte); overload;
    class operator Add(const A, B: TSixel): TSixel;
    class operator Subtract(const A, B: TSixel): TSixel;
    class operator Multiply(const A, B: TSixel): TSixel;
    class operator IntDivide(const A, B: TSixel): TSixel;
    class operator Multiply(const A: TSixel; const B: Byte): TSixel;
    class operator IntDivide(const A: TSixel; const B: Byte): TSixel;
    class operator Multiply(const A: TSixel; const B: Single): TSixel;
    class operator Implicit(const A: Byte): TSixel;

  end;

  TSixelS = packed record
  R, G, B: Single
  end;

  PSixel = ^TSixel;

  TColorMap = TDictionary<TSixel, LongInt>;

function Comp(const A, B: LongWord): NativeInt; winapi;
begin
  Result := A - B
end;

function SxlIsEqual(const A, B: TSixel): Boolean;
begin
  Result := (A.R = B.R) and (A.G = B.G) and (A.B = B.B)
end;

function IntToAnsiStr(const Int: Integer): AnsiString; inline;
begin
  Result := AnsiString(System.SysUtils.IntToStr(Int))
end;

function EncodeSixel(
    const Buf: Pointer;
    const Width, Height: NativeInt;
    const Dither: Boolean;
    const MonoChrome: Boolean;
    const PixelOrder: TPixelOrder
): RawByteString;
const
  PIXEL = 1;
  PIXEL_COUNT = 6 div PIXEL;
  COLOR_SPACE: array[0..2] of Byte = (8, 8, 4);
var
  P: PSixel;
  Bytes: PByte;
  //  bitPix  : array of byte;
  Pos: NativeInt;
  ColorPal: TColorMap;
  Plt: AnsiString;
  SxlMin, SxlMax: TSixel;
  SxlScaler: TSixel;

  function GetSixel(const Y, X: NativeInt): TSixel;
  begin
    Result := P[Y * Width + X]
  end;

  function GetMono(const Y, X: NativeInt): TSixel;
  begin
    Result := 100 * Bytes[Y * Width + X] div 255
  end;

  function GetSixelHWC(const Data: PSixel; const Y, X: NativeInt; const AHeight, AWidth: NativeInt): TSixel;
  begin
    Result := Data[Y * AWidth + X]
  end;

  function GetSixelCHW(const Data: PByte; const Y, X: NativeInt; const AHeight, AWidth: NativeInt): TSixel;
  begin
    Result.R := Data[Y * AWidth + X];
    Result.G := Data[AHeight * AWidth + Y * Width + X];
    Result.B := Data[2 * AHeight * AWidth + Y * Width + X];
  end;

  procedure EncodeNum(const Ctl: AnsiChar; const Num: LongInt);
  var
    Sz: NativeInt;
  begin
    if Length(Plt) < Pos + 6 then
      SetLength(Plt, 2 * (Pos + 5));
    Plt[Pos] := Ctl;
    Inc(Pos);
    if Num > 99 then
      Sz := 3
    else if Num > 9 then
      Sz := 2
    else
      Sz := 1;
    Move(IntToAnsiStr(Num)[1], Plt[Pos], Sz);
    Inc(Pos, Sz);
  end;

  procedure EncodeChar(const Ctl: AnsiChar; const Num: LongInt; const Str: AnsiChar);
  var
    Sz: NativeInt;
  begin
    if Length(Plt) < Pos + 7 then
      SetLength(Plt, 2 * (Pos + 6));
    Plt[Pos] := Ctl;
    Inc(Pos);
    if Num > 99 then
      Sz := 3
    else if Num > 9 then
      Sz := 2
    else
      Sz := 1;
    Move(IntToAnsiStr(Num)[1], Plt[Pos], Sz);
    Inc(Pos, Sz);
    Plt[Pos] := Str;
    Inc(Pos);
  end;

  procedure EncodeStr(const Str: AnsiString);
  var
    Sz: NativeInt;
  begin
    Sz := Length(Str);
    if Length(Plt) < Pos + Sz then
      SetLength(Plt, 2 * (Pos + Sz));
    Move(Str[1], Plt[Pos], Sz);
    Inc(Pos, Sz)
  end;

  procedure EncodeRepeat(const Num: LongInt; const Str: AnsiChar); inline;
  var
    I: LongInt;
  begin
    for I := 0 to Num div 255 - 1 do
      EncodeChar('!', 255, Str);
    if Num mod 255 > 0 then
      EncodeChar('!', Num mod 255, Str)
  end;

  function Clr100(const Col: TSixel): TSixel; inline;
  begin
    Result.R := Trunc(100 * Col.R / 255);
    Result.G := Trunc(100 * Col.G / 255);
    Result.B := Trunc(100 * Col.B / 255);
  end;

  function ClrSpc(const Col: TSixel): TSixel;
  begin
    // result.r := round(100*(col.r - sxlMin.r)/255) div sxlScaler.r;
    // result.g := round(100*(col.g - sxlMin.g)/255) div sxlScaler.g;
    // result.b := round(100*(col.b - sxlMin.b)/255) div sxlScaler.b;
    // result.r := sxlScaler.r*(round(100*col.r/255) div sxlScaler.r);
    // result.g := sxlScaler.g*(round(100*col.g/255) div sxlScaler.g);
    // result.b := sxlScaler.b*(round(100*col.b/255) div sxlScaler.b);
    Result := SxlScaler * (Col div SxlScaler);
  end;

  procedure ReCalcPalette(var Mi, Ma: TSixel; var Scaler: TSixel);
  var
    I, R, G, B: LongInt;
    Sx: TSixel;
  begin
    // clrCount := 0;
    ColorPal.Clear;
    if MonoChrome then
      Mi.R := Bytes[0]
    else
      case PixelOrder of
        poHWC:
        begin
          Mi := P[0];
          Ma := P[0];
        end;
        poCHW:
        begin
          Mi := GetSixelCHW(Bytes, 0, 0, Height, Width);
          Ma := Mi;
        end;
      end;
    // mi.r := p[0].r;
    // mi.g := p[0].g;
    // mi.b := p[0].b;
    // ma.r := p[0].r;
    // ma.g := p[0].g;
    // ma.b := p[0].b;
    if MonoChrome then
      for I := 1 to Height * Width - 1 do
      begin
        Sx.R := Bytes[I];
        Mi.R := System.Math.Min(Sx.R, Mi.R);
        Ma.R := System.Math.Max(Sx.R, Ma.R);
      end
    else
      for I := 1 to Height * Width - 1 do
      begin
        case PixelOrder of
          poHWC: Sx := P[I];
          poCHW: Sx := GetSixelCHW(Bytes, 0, I, 1, Height * Width);
        end;
        Mi.R := System.Math.Min(Sx.R, Mi.R);
        Mi.G := System.Math.Min(Sx.G, Mi.G);
        Mi.B := System.Math.Min(Sx.B, Mi.B);
        Ma.R := System.Math.Max(Sx.R, Ma.R);
        Ma.G := System.Math.Max(Sx.G, Ma.G);
        Ma.B := System.Math.Max(Sx.B, Ma.B);
      end;
    // s.r := round(100 *(ma.r-mi.r) / 255) div 5;     // scaler
    // s.g := round(100 *(ma.g-mi.g) / 255) div 5;
    // s.b := round(100 *(ma.b-mi.b) / 255) div 5;
    if MonoChrome then
      for I := 0 to 100 do
      begin
        EncodeNum('#', I);
        EncodeNum(';', 2);
        EncodeNum(';', I);
        EncodeNum(';', I);
        EncodeNum(';', I);
        ColorPal.Add(TSixel.Create(I), I);
      end
    else
    begin
      Scaler.R := 100 div (COLOR_SPACE[0] - 1);
      Scaler.G := 100 div (COLOR_SPACE[1] - 1);
      Scaler.B := 100 div (COLOR_SPACE[2] - 1);
      // i:=16;
      I := 0;
      for R := 0 to COLOR_SPACE[0] - 1 do
        for G := 0 to COLOR_SPACE[1] - 1 do
          for B := 0 to COLOR_SPACE[2] - 1 do
          begin
            Sx.R := Scaler.R * R;
            Sx.G := Scaler.G * G;
            Sx.B := Scaler.B * B;
            EncodeNum('#', I);
            EncodeNum(';', 2);
            // encodeNum(';', mi.r + s.r*r);
            // encodeNum(';', mi.g + s.g*g);
            // encodeNum(';', mi.b + s.b*b); // encodeStr(#13#10);
            EncodeNum(';', Sx.R);
            EncodeNum(';', Sx.G);
            EncodeNum(';', Sx.B); // encodeStr(#13#10);
            ColorPal.Add(Sx, I);
            Inc(I)
          end;
    end
  end;

const // enable SIXEL + START SIXEL
  START_SIXEL = #$1B']80h' + #$1B'Pq';
  END_SIXEL = #$1B'\'; // ST
var
  CharCount: LongWord;
  BitPix: Byte;
  Sxl, LastSxl, QErr: TSixel;
  P1, P2, P3, P4: PSixel;
  Dith: TArray<TSixel>;
  // clrCount,
  Reg: LongInt;
  I, Idx, X, Y: NativeInt;

label
  Done;
begin

  if (Buf = nil) or (Width < 1) or (Height < 1) or (Int64(Width) * Height > High(Integer)) then
    raise EArgumentException.Create('Invalid Sixel image buffer/dimensions');
  ColorPal := TColorMap.Create(Width * Height);
  SetLength(Plt, Width * Height);
  SetLength(Dith, Width * Height);
  Pos := 1;
  // move(buf^, dith[0], width*height*sizeOf(p[0]));
  P := Pointer(Buf);
  Bytes := Pointer(Buf);
  ReCalcPalette(SxlMin, SxlMax, SxlScaler);

  if MonoChrome then
  else
    case PixelOrder of
      poHWC:
        for I := 0 to High(Dith) do
          Dith[I] := Clr100(P[I]);
      poCHW:
        for I := 0 to Height * Width - 1 do
          Dith[I] := Clr100(GetSixelCHW(Buf, 0, I, 1, Height * Width));
    end;

  P := Pointer(Dith);

  if Dither then
    for Y := 0 to Height - 2 do
      for X := 1 to Width - 2 do
      begin
        Idx := Y * Width + X;
        LastSxl := Dith[Idx];
        Sxl := ClrSpc(LastSxl);
        Dith[Idx] := Sxl;
        QErr := LastSxl - Sxl;
        P1 := @P[Idx + 1];
        P2 := @P[Idx + Width - 1];
        P3 := @P[Idx + Width];
        P4 := @P[Idx + Width + 1];

        P1^ := P1^+QErr * (7 / 16);
        P2^ := P2^+QErr * (3 / 16);
        P3^ := P3^+QErr * (5 / 16);
        P4^ := P4^+QErr * (1 / 16);
      end;

  //  setLength(bitPix, width);
  // write(copy(plt,1, pos-1));

  CharCount := 0;
  BitPix := 63;

  // t:=GetTickCount64;
  for Y := 0 to System.Math.Ceil(Height / PIXEL_COUNT) do
  begin
    for I := 0 to PIXEL_COUNT - 1 do
    begin
      if Y * PIXEL_COUNT + I >= Height then
        Break;
      // lastC := $ffffff;
      LastSxl.R := $ff;
      for X := 0 to Width - 1 do
      begin
        // idx := getSixel(y*pixCount + i, x);
        // c := col2Index(p[idx]);
        // sxl := col100(getSixel(y*pixCount + i, x));
        // sxl := col2Six(sxl);
        if MonoChrome then
          Sxl := GetMono(Y * PIXEL_COUNT + I, X)
        else
          Sxl := ClrSpc(GetSixel(Y * PIXEL_COUNT + I, X));
        BitPix := 63 + ((1 shl PIXEL) - 1) shl (I * PIXEL);
        if SxlIsEqual(Sxl, LastSxl) then
        begin
          // if c=lastC then begin
          Inc(CharCount)
        end
        else
        begin
          if CharCount > 0 then
          begin
            EncodeRepeat(CharCount * PIXEL, AnsiChar(BitPix));
            // encodeChar('!', charCount*pixel, ansichar(bitPix));
          end;
          // colorPal.TryGetValue(c, reg);
          if not ColorPal.TryGetValue(Sxl, Reg) then
            raise Exception.Create('Cannot locate color in the palette!');
          EncodeChar('#', Reg, AnsiChar(BitPix));
          CharCount := 0;
          if PIXEL > 1 then
            FillChar(Plt[Pos], PIXEL - 1, BitPix);
          Inc(Pos, PIXEL - 1);
        end;
        LastSxl := Sxl;
        // lastC := c
      end;
      if CharCount > 0 then
      begin
        EncodeRepeat(CharCount * PIXEL, AnsiChar(BitPix));
        // encodeChar('!', charCount*pixel, ansichar(bitPix));
        CharCount := 0;
      end;
      Plt[Pos] := '$';
      Inc(Pos); // overlay line
    end;
    Plt[Pos] := '-';
    Inc(Pos) // draw in new line
  end;
  Done:
  FreeAndNil(ColorPal);
  // writeln((GetTickCount64-t)/1000:1:3,'ms');

  (*
     P1          Pixel Aspect Ratio
                 (Vertical:Horizontal)

     Omitted     2:1
     0 or 1      5:1
     2           3:1
     3 or 4      2:1
     5 or 6      2:1
     7,8, or 9   1:1

    P2 selects how the terminal draws the background color.   You can 3$
    use one of three values.

         P2          Meaning

         0 or 2      Pixel positions  specified as  0 are  set to the
                     current background color.

         1           Pixel positions specified as  0 remain  at their
                     current color.

    P3 is  the horizontal  grid size  parameter.  The horizontal grid
    size is the horizontal distance  between  two  pixel  dots.   The
    VT300 ignores  this parameter because the horizontal grid size is
    fixed at 0.0195 cm (0.0075 in).
*)

  SetLength(Plt, Pos - 1);
  Result := START_SIXEL + '"1;1;' + IntToAnsiStr(Width) + ';' + IntToAnsiStr(Height) + Plt + END_SIXEL;
  // write(plt)
end;

procedure PrintSixel(
    const Buf: Pointer;
    const Width, Height: NativeInt;
    const Dither, MonoChrome: Boolean;
    const PixelOrder: TPixelOrder
);
begin
  Write(EncodeSixel(Buf, Width, Height, Dither, MonoChrome, PixelOrder));
end;

{ TSixel }

constructor TSixel.Create(const AR, AG, AB: Byte);
begin
  R := AR;
  G := AG;
  B := AB
end;

constructor TSixel.Create(const L: Byte);
begin
  R := L;
  G := L;
  B := L;
end;

class operator TSixel.Add(const A, B: TSixel): TSixel;
begin
  Result.R := A.R + B.R;
  Result.G := A.G + B.G;
  Result.B := A.B + B.B;
end;

class operator TSixel.Subtract(const A, B: TSixel): TSixel;
begin
  Result.R := A.R - B.R;
  Result.G := A.G - B.G;
  Result.B := A.B - B.B;
end;

class operator TSixel.Multiply(const A, B: TSixel): TSixel;
begin
  Result.R := A.R * B.R;
  Result.G := A.G * B.G;
  Result.B := A.B * B.B;
end;

class operator TSixel.IntDivide(const A, B: TSixel): TSixel;
begin
  Result.R := A.R div B.R;
  Result.G := A.G div B.G;
  Result.B := A.B div B.B;
end;

class operator TSixel.Multiply(const A: TSixel; const B: Byte): TSixel;
begin
  Result.R := A.R * B;
  Result.G := A.G * B;
  Result.B := A.B * B;
end;

class operator TSixel.IntDivide(const A: TSixel; const B: Byte): TSixel;
begin
  Result.R := A.R div B;
  Result.G := A.G div B;
  Result.B := A.B div B;
end;

class operator TSixel.Multiply(const A: TSixel; const B: Single): TSixel;
begin
  Result.R := Trunc(A.R * B);
  Result.G := Trunc(A.G * B);
  Result.B := Trunc(A.B * B);
end;

class operator TSixel.Implicit(const A: Byte): TSixel;
begin
  Result.R := A;
  Result.G := A;
  Result.B := A
end;

end.
