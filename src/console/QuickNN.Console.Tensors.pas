unit QuickNN.Console.Tensors;

{$H+}
{$POINTERMATH ON}

interface

uses
  QuickNN.Common;

type
  TTensorPrintStyle = (
    psValues,
    psGray5 {5 ascii shades},
    psGray24 {24 half char shades},
    psGray {256 half char shades},
    psColor8 {256 colors},
    psColor24 {~16M colors},
    psSIXELGray,
    psSIXEL,
    psSIXELDithered
    );

  // Import this unit explicitly to opt into tensor/image console rendering.
  TMemoryBlockConsoleHelper = record helper for TMemoryBlock
    function Print(const ConsolePixel: TTensorPrintStyle = psGray; Tile: Integer = 1; MinVal: Double = 0; MaxVal: Double = 0
    ): TArray<NativeInt>;
  end;

  TQNNImageConsoleHelper = record helper for TQNNImage
    procedure PrintSixel;
  end;

implementation

uses
  System.SysUtils, System.Math, QuickNN.CPU, QuickNN.Console.Ansi,
  QuickNN.Console.Sixel;

const
  PRINT_DIGITS = 5;

function ToStr(const F: Single): string; overload;
var
  Buffer: ShortString;
begin
  Str(F: 1: 4, Buffer);
  Result := string(Buffer)
end;

function ToStr(const F: Double): string; overload;
var
  Buffer: ShortString;
begin
  Str(F: 1: 4, Buffer);
  Result := string(Buffer)
end;

function ToStr(const F: Integer): string; overload;
var
  Buffer: ShortString;
begin
  Str(F, Buffer);
  Result := string(Buffer)
end;

function ToStr(const F: Int64): string; overload;
var
  Buffer: ShortString;
begin
  Str(F, Buffer);
  Result := string(Buffer)
end;

function ToStr(const F: ShortInt): string; overload;
var
  Buffer: ShortString;
begin
  Str(F, Buffer);
  Result := string(Buffer)
end;

function ToStr(const F: SmallInt): string; overload;
var
  Buffer: ShortString;
begin
  Str(F, Buffer);
  Result := string(Buffer)
end;

procedure PrintTensor(const Data: System.PSingle; const Shape: TArray<Int64>; const Offset: Int64 = 0; const Depth: Int64 = 0; Strides: TArray<Int64> = nil
    );
var
  I: Int64;
  Indent: string;
begin

  Assert(Depth < Length(Shape), 'ERROR Print : tensor depth is out of boundries.');
  Indent := string.Create(' ', Depth);

  if not Assigned(Strides) then
    Strides := GetStrides(Shape);
  if Depth = High(Shape) then
  begin
    for I := Offset to Offset + Shape[Depth] - 1 do
    begin
      if I = Offset then
      begin
        Write(Indent, '[', Data[I]: 1: PRINT_DIGITS)
      end
      else
        Write(', ', Data[I]: 1: PRINT_DIGITS)
    end;
    WriteLn(']');
  end
  else
  begin
    WriteLn(Indent, '[');
    for I := 0 to Shape[Depth] - 1 do
    begin
      PrintTensor(Data, Shape, Offset + I * Strides[Depth], Depth + 1, Strides);
    end;
    WriteLn(Indent, ']');
  end;
end;

function TMemoryBlockConsoleHelper.Print(const ConsolePixel: TTensorPrintStyle; Tile: Integer; MinVal: Double; MaxVal: Double): TArray<NativeInt>;
const
  CSI = #$1B'[';
  UP = #$1B'[A';
  DW = #$1B'[B';
  FW = #$1B'[C';
  BW = #$1B'[D';
  SC = #$1B'[s';
  RC = #$1B'[u';
  ER = #$1B'[0K';
  CPOS = #$1B'[6n';
var
  AMin, AMax: QNNFloat;
  I, J, K, T, LocalW, LocalH, LocalC, LocalArea, Index, OutArgMin, OutArgMax, Ow, Oh, LocalSize: Integer;
  L: LongWord;
  Range, R, G, B, R2, G2, B2: Double;
  S: WideString;
  Pixels: RawByteString;
  IsHalfChar, IsColor, IsSIXEL: Boolean;
  LocalData: PQNNFloat;
const
  SHADE: array[0..4] of string = (' ', '░', '▒', '▓', '█');
  // delphi will complain if no string type defined, MacOS will type rubbish if string type is defined!! :(
{$IF defined(MSWINDOWS) or defined(POSIX)}
  HALF_CHAR: WideChar = #$2580;
{$ELSE}
  HALF_CHAR: AnsiString = '▀'; // this will fail to compile on delphi - android
{$ENDIF}
begin
  Assert(DataType = dtF32, 'print : Tensor visualization is not impelemted for this type!');

  if not IsAssigned() then
    Exit;
  LocalSize := Count();
  S := TypeName() + ' Tensor (';
  for I := 0 to High(Shape) do
    if I = 0 then
      S := S + ToStr(Shape[I])
    else
    begin
      S := S + ' X ';
      S := S + ToStr(Shape[I]);
    end;
  S := S + ')';
  LocalData := Self;
  Write(S, CSI, Length(S), 'D', DW);
  // writeln(S);
  Ow := Length(S);
  Oh := 1;
  IsHalfChar := not (ConsolePixel in [psValues, psGray5, psSIXELGray, psSIXEL, psSIXELDithered]);
  IsColor := ConsolePixel in [psColor8, psColor24, psSIXEL, psSIXELDithered];
  IsSIXEL := ConsolePixel in [psSIXEL, psSIXELGray, psSIXELDithered];
  if ConsolePixel <> psValues then
  begin
    if MinVal = MaxVal then
    begin
      TQNNSingleOps.QNNMinMax(Count, PQNNFloat(Self), AMin, AMax, @OutArgMin, @OutArgMax);
      MinVal := AMin;
      MaxVal := AMax;
      S := '[min : ' + ToStr(AMin) + '@' + ToStr(OutArgMin) + ', max : ' + ToStr(AMax) + '@' + ToStr(OutArgMax) + ']';
      Write(S, CSI, Length(S), 'D', DW);
      // writeln(S);
      Ow := System.Math.Max(Ow, Length(S));
      Inc(Oh);
    end;

    if System.Math.IsNan(MinVal) or System.Math.IsNan(MaxVal) then
    begin
      Result := [Ow, Oh];
      Exit();
    end;
    LocalW := Shape[High(Shape)];
    if Length(Shape) > 1 then
    begin
      LocalH := Shape[High(Shape) - 1];
      LocalArea := LocalH * LocalW;
    end
    else
    begin
      LocalH := 1;
      LocalArea := LocalW;
    end;
    LocalC := (1 + 2 * Ord(IsColor));
    Range := MaxVal - MinVal;
    Result := [Ow, Oh];
    if (Range < EPSILON) or (Tile < 1) then
      Exit;
    S := '';
    Pixels := '';
    L := 0;
    for I := 0 to LocalSize div (LocalC * LocalArea * Tile) - 1 do
    begin
      for J := 0 to Ceil(LocalH / (1 + Ord(IsHalfChar))) - 1 do
      begin
        for T := 0 to Tile - 1 do
        begin
          for K := 0 to LocalW - 1 do
          begin
            Index :=
              I * LocalC * Tile * LocalArea + T * LocalC * LocalH * LocalW + J * (1 + Ord(IsHalfChar)) * LocalW + K;
            if Index < LocalSize then
            begin
              R := LocalData[Index];
              case ConsolePixel of
                psGray5:
                  S := S + SHADE[Round(4 * (R - MinVal) / Range)];
                psGray24:
                  begin
                    Inc(Index, LocalW);
                    if Index < LocalSize then
                      R2 := LocalData[Index]
                    else
                      R2 := MinVal;
                    R := 232 + 23 * (R - MinVal) / Range;
                    R2 := 232 + 23 * (R2 - MinVal) / Range;
                    S := S + #$1B'[38;5;' + ToStr(Round(R)) + 'm' + #$1B'[48;5;' + ToStr(Round(R2)) + 'm' + HALF_CHAR;
                  end;
                psGray:
                  begin
                    Inc(Index, LocalW);
                    if Index < LocalSize then
                      R2 := LocalData[Index]
                    else
                      R2 := MinVal;
                    R := $ff * (R - MinVal) / Range;
                    R2 := $ff * (R2 - MinVal) / Range;
                    S :=
                      S
                      + #$1B'[38;2;'
                      + ToStr(Round(R))
                      + ';'
                      + ToStr(Round(R))
                      + ';'
                      + ToStr(Round(R))
                      + 'm'
                      + #$1B'[48;2;'
                      + ToStr(Round(R2))
                      + ';'
                      + ToStr(Round(R2))
                      + ';'
                      + ToStr(Round(R2))
                      + 'm'
                      + HALF_CHAR;
                  end;
                psColor8:
                  begin
                    G := LocalData[Index + LocalArea];
                    B := LocalData[Index + LocalArea * 2];
                  // next line
                    Inc(Index, LocalW);
                    if Index < LocalSize then
                    begin
                      R2 := LocalData[Index];
                      G2 := LocalData[Index + LocalArea];
                      B2 := LocalData[Index + LocalArea * 2]
                    end
                    else
                    begin
                      R2 := MinVal;
                      G2 := MinVal;
                      B2 := MinVal
                    end;

                    R := 5 * (R - MinVal) / Range;
                    G := 5 * (G - MinVal) / Range;
                    B := 5 * (B - MinVal) / Range;

                    R2 := 5 * (R2 - MinVal) / Range;
                    G2 := 5 * (G2 - MinVal) / Range;
                    B2 := 5 * (B2 - MinVal) / Range;

                    S :=
                      S
                      + #$1B'[38;5;'
                      + ToStr(16 + Round(B) + 6 * Round(G) + 36 * Round(R))
                      + 'm'
                      + #$1B'[48;5;'
                      + ToStr(16 + Round(B2) + 6 * Round(G2) + 36 * Round(R2))
                      + 'm'
                      + HALF_CHAR;
                  end;
                psColor24:
                  begin
                    G := LocalData[Index + LocalArea];
                    B := LocalData[Index + LocalArea * 2];
                  // nex line
                    Inc(Index, LocalW);
                    if Index < LocalSize then
                    begin
                      R2 := LocalData[Index];
                      G2 := LocalData[Index + LocalArea];
                      B2 := LocalData[Index + LocalArea * 2];
                    end
                    else
                    begin
                      R2 := MinVal;
                      G2 := MinVal;
                      B2 := MinVal
                    end;

                    R := $ff * (R - MinVal) / Range;
                    G := $ff * (G - MinVal) / Range;
                    B := $ff * (B - MinVal) / Range;

                    R2 := $ff * (R2 - MinVal) / Range;
                    G2 := $ff * (G2 - MinVal) / Range;
                    B2 := $ff * (B2 - MinVal) / Range;

                    S :=
                      S
                      + #$1B'[38;2;'
                      + ToStr(Round(R))
                      + ';'
                      + ToStr(Round(G))
                      + ';'
                      + ToStr(Round(B))
                      + 'm'
                      + #$1B'[48;2;'
                      + ToStr(Round(R2))
                      + ';'
                      + ToStr(Round(G2))
                      + ';'
                      + ToStr(Round(B2))
                      + 'm'
                      + HALF_CHAR;
                  end;
                psSIXEL, psSIXELDithered:
                  begin
                    G := LocalData[Index + LocalArea];
                    B := LocalData[Index + LocalArea * 2];

                    R := $ff * (R - MinVal) / Range;
                    G := $ff * (G - MinVal) / Range;
                    B := $ff * (B - MinVal) / Range;
                    Pixels := Pixels + AnsiChar(Round(R)) + AnsiChar(Round(G)) + AnsiChar(Round(B));
                  end;
                psSIXELGray:
                  begin
                    R := $ff * (R - MinVal) / Range;
                    Pixels := Pixels + AnsiChar(Round(R));
                  end;
              end;
            end;
          end;
          // if consolePixel<>psGray5 then
          //  S := S + #$1B'[0m '
        end;
        if not IsSIXEL then
        begin
          Write(S, CSI, LocalW * Tile, 'D', DW);
          S := '';
        end;
        // writeln(S);
        Ow := System.Math.Max(Ow, LocalW * Tile);
        Inc(Oh);
        Inc(L);
      end;
      if IsSIXEL then
      begin
        PrintSixel(@Pixels[1], LocalW * Tile, LocalH, ConsolePixel = psSIXELDithered, ConsolePixel = psSIXELGray);
        Pixels := ''
      end
    end;
    if ConsolePixel <> psGray5 then
      S := RESET_ALL_MODES; // #$1B'[0m';
    if not IsSIXEL then
      Write(S);
    // writeln(S);
    Result := [Ow, Oh];
    Exit;
  end;
  PrintTensor(LocalData, Shape);
  Result := [Ow, Oh];
end;

procedure TQNNImageConsoleHelper.PrintSixel;
begin
  QuickNN.Console.Sixel.PrintSixel(Data, Width, Height, True);
end;

end.

