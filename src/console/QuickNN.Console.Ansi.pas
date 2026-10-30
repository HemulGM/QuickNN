(*

  ************************************
  ANSI Escape sequence Terminl Helper
  ************************************

  Copyright (C) 2024 <Haitham Shatti> <haitham.shatti at gmail dot com>

  This library is free software; you can redistribute it and/or modify it under the terms of the GNU Library General Public License as published by the Free Software Foundation; either version 2 of
  the License, or (at your option) any later version with the following modification:

  As a special exception, the copyright holders of this library give you permission to link this library with independent modules to produce an executable, regardless of the license terms of these
  independent modules,and to copy and distribute the resulting executable under terms of your choice, provided that you also meet, for each linked independent module, the terms and conditions of the
  license of that module. An independent module is a module which is not derived from or based on this library. If you modify this library, you may extend this exception to your version of the
  library, but you are not obligated to do so. If you do not wish to do so, delete this exception statement from your version.

  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
  Library General Public License for more details.

  You should have received a copy of the GNU Library General Public License along with this library; if not, write to the Free Software Foundation, Inc., 51 Franklin Street - Fifth Floor, Boston, MA
  02110-1335, USA.
*)

unit QuickNN.Console.Ansi;

interface

type
  TCursorDirection = (cmUp, cmDown, cmBackward, cmForward);

const
  SET_OVERLINE = #$1B'[53m';
  SET_BOLD = #$1B'[1m';
  SET_FAINT = #$1B'[2m';
  SET_ITALIC = #$1B'[3m';
  SET_UNDERLINE = #$1B'[4m';
  SET_SLOW_BLINK = #$1B'[5m';
  SET_FAST_BLINK = #$1B'[6m';
  SET_INVERSE = #$1B'[7m';
  SET_STRIKE = #$1B'[9m';

  CLEAR_OVERLINE = #$1B'[55m';
  CLEAR_BOLD = #$1B'[21m';
  CLEAR_FAINT = #$1B'[22m';
  CLEAR_ITALIC = #$1B'[23m';
  CLEAR_UNDERLINE = #$1B'[24m';
  CLEAR_BLINK = #$1B'[25m';
  CLEAR_INVERSE = #$1B'[27m';
  CLEAR_STRIKE = #$1B'[29m';
  CLEAR_FONT = #$1B'[10m';

  RESET_ALL_MODES = #$1B'[0m';
  RESET_COLOR = #$1B'[39m';
  RESET_BACK_COLOR = #$1B'[49m';

  SET_CLEAR_LINE_END = #$1B'[0K';
  SET_CLEAR_LINE_START = #$1B'[1K';
  SET_CLEAR_LINE = #$1B'[2K';

  COLOR_BLACK = 0;
  COLOR_MAROON = 1;
  COLOR_GREEN = 2;
  COLOR_OLIVE = 3;
  COLOR_NAVY = 4;
  COLOR_MAGENTA = 5;
  COLOR_TEAL = 6;
  COLOR_SILVER = 7;
  COLOR_GRAY = 8;
  COLOR_RED = 9;
  COLOR_LIME = 10;
  COLOR_YELLOW = 11;
  COLOR_BLUE = 12;
  COLOR_PURPLE = 13;
  COLOR_AQUA = 14;
  COLOR_WHITE = 15;

function CursorMove(const Direction: TCursorDirection; const Count: Integer = 1): RawByteString; overload;
function CursorMove(const RelativeCol, RelativeRow: Integer): RawByteString; overload;
procedure CursorShow(const Visible: Boolean);
procedure SaveScreen();
procedure RestoreScreen();

function SetColor4(const Color: Integer): RawByteString;
function SetBackColor4(const Color: Integer): RawByteString;
function SetColor5(const R, G, B: Byte): RawByteString;
function SetBackColor5(const R, G, B: Byte): RawByteString;
function SetColor(const R, G, B: Byte): RawByteString;
function SetBackColor(const R, G, B: Byte): RawByteString;
function SetGray(const Brightness: Byte): RawByteString;
function SetBackGray(const Brightness: Byte): RawByteString;
function SetCursorPos(const X, Y: Integer): RawByteString;
function SetCursorScrollDown(const Count: Integer = 1): RawByteString;
function SetCursorScrollUp(const Count: Integer = 1): RawByteString;

procedure ResetModes();
function SetFont(Font: Byte): RawByteString;

procedure CursorUp(const Count: Integer = 1);
procedure CursorDown(const Count: Integer = 1);
procedure CursorForward(const Count: Integer = 1);
procedure CursorBack(const Count: Integer = 1);
procedure CursorAbsPos(const X: Integer = 1; const Y: Integer = 1);
procedure CursorScrollUp(const Count: Integer = 1);
procedure CursorScrollDown(const Count: Integer = 1);
procedure CursorClearDown();
procedure CursorClearUp();
procedure CursorClearScreen();
procedure CursorHome();
procedure SaveCursorPos();
procedure RestoreCursorPos();
procedure SetUTF8Console();

implementation

{$IFDEF MSWINDOWS}
uses
  Winapi.Windows;

procedure EnableVirtualTerminal;
var
  ConsoleHandle: THandle;
  ConsoleMode: Dword;
begin
  if not IsConsole then
    Exit;
  ConsoleHandle := GetStdHandle(STD_OUTPUT_HANDLE);
  if GetConsoleMode(ConsoleHandle, ConsoleMode) then
    SetConsoleMode(ConsoleHandle, ConsoleMode or ENABLE_VIRTUAL_TERMINAL_PROCESSING or ENABLE_PROCESSED_OUTPUT);
end;
{$ENDIF}

function IntToStr(const I: Integer): RawByteString;
begin
  Str(I, Result)
end;

function CursorMove(const Direction: TCursorDirection; const Count: Integer): RawByteString;
begin
  case Direction of
    cmUp: Result := #$1B'[' + IntToStr(Count) + 'A';
    cmDown: Result := #$1B'[' + IntToStr(Count) + 'B';
    cmForward: Result := #$1B'[' + IntToStr(Count) + 'C';
    cmBackward: Result := #$1B'[' + IntToStr(Count) + 'D';
  end;
end;

function CursorMove(const RelativeCol, RelativeRow: Integer): RawByteString;
var
  Cm: TCursorDirection;
begin
  Result := '';
  if (RelativeCol <> 0) then
  begin
    if RelativeCol < 0 then
      Cm := cmBackward
    else
      Cm := cmForward;
    Result := CursorMove(Cm, Abs(RelativeCol));
  end;

  if (RelativeRow <> 0) then
  begin
    if RelativeRow < 0 then
      Cm := cmUp
    else
      Cm := cmDown;
    Result := Result + CursorMove(Cm, Abs(RelativeRow));
  end;
end;

procedure CursorShow(const Visible: Boolean);
begin
  if Visible then
    Write(#$1B'[?25h')
  else
    Write(#$1B'[?25l')
end;

procedure SaveScreen();
begin
  Write(#$1B'[?47h');
end;

procedure RestoreScreen();
begin
  Write(#$1B'[?47l');
end;

function SetColor4(const Color: Integer): RawByteString;
begin
  if Color div 8 > 0 then
    Result := #$1B'[9' + IntToStr(Color mod 8) + 'm'
  else
    Result := #$1B'[3' + IntToStr(Color) + 'm';
end;

function SetBackColor4(const Color: Integer): RawByteString;
begin
  if Color div 8 > 0 then
    Result := #$1B'[10' + IntToStr(Color mod 8) + 'm'
  else
    Result := #$1B'[4' + IntToStr(Color) + 'm';
end;

function SetColor5(const R, G, B: Byte): RawByteString;
var
  C: Byte;
begin
  C := 16 + Round(B) + 6 * Round(G) + 36 * Round(R);
  Result := #$1B'[38;5;' + IntToStr(C) + 'm';
end;

function SetBackColor5(const R, G, B: Byte): RawByteString;
var
  C: Byte;
begin
  C := 16 + Round(B) + 6 * Round(G) + 36 * Round(R);
  Result := #$1B'[48;5;' + IntToStr(C) + 'm';
end;

function SetColor(const R, G, B: Byte): RawByteString;
begin
  Result := #$1B'[38;2;' + IntToStr(R) + ';' + IntToStr(G) + ';' + IntToStr(B) + 'm';
end;

function SetGray(const Brightness: Byte): RawByteString;
begin
  case Brightness of
    0: Result := #$1B'[30m';
    1..24: Result := #$1B'[38;5;' + IntToStr(Brightness + 231) + 'm';
    25: Result := #$1B'[97m';
  end;
end;

function SetBackGray(const Brightness: Byte): RawByteString;
begin
  case Brightness of
    0: Result := #$1B'[40m';
    1..24: Result := #$1B'[48;5;' + IntToStr(Brightness + 231) + 'm';
    25: Result := #$1B'[107m';
  end;
end;

function SetCursorPos(const X, Y: Integer): RawByteString;
begin
  Result := #$1B'[' + IntToStr(Y) + ';' + IntToStr(X) + 'H'
end;

function SetCursorScrollDown(const Count: Integer): RawByteString;
begin
  Result := #$1B'[' + IntToStr(Count) + 'T'
end;

function SetCursorScrollUp(const Count: Integer): RawByteString;
begin
  Result := #$1B'[' + IntToStr(Count) + 'S'
end;

function SetBackColor(const R, G, B: Byte): RawByteString;
begin
  Result := #$1B'[48;2;' + IntToStr(R) + ';' + IntToStr(G) + ';' + IntToStr(B) + 'm';
end;

procedure ResetModes;
begin
  Write(RESET_ALL_MODES);
end;

function SetFont(Font: Byte): RawByteString;
begin
  Result := #$1B'[1' + IntToStr(Font) + 'm';
end;

procedure CursorUp(const Count: Integer);
begin
  Write(#$1B'[', Count, 'A')
end;

procedure CursorDown(const Count: Integer);
begin
  Write(#$1B'[', Count, 'B')
end;

procedure CursorForward(const Count: Integer);
begin
  Write(#$1B'[', Count, 'C')
end;

procedure CursorBack(const Count: Integer);
begin
  Write(#$1B'[', Count, 'D')
end;

procedure CursorAbsPos(const X: Integer; const Y: Integer);
begin
  Write(#$1B'[', Y, ';', X, 'H')
end;

procedure CursorScrollUp(const Count: Integer);
begin
  Write(#$1B'[', Count, 'S')
end;

procedure CursorScrollDown(const Count: Integer);
begin
  Write(#$1B'[', Count, 'T')
end;

procedure CursorClearDown();
begin
  Write(#$1B'[0J')
end;

procedure CursorClearUp();
begin
  Write(#$1B'[1J')
end;

procedure CursorClearScreen();
begin
  Write(#$1B'[2J')
end;

procedure CursorHome();
begin
  Write(#$1B'[1H')
end;

procedure SaveCursorPos();
begin
  Write(#$1B'7')
end;

procedure RestoreCursorPos();
begin
  Write(#$1B'8')
end;

procedure SetUTF8Console();
begin
  WriteLn(#$1B'%G'); // set console UTF8;
end;

{$IFDEF MSWINDOWS}
initialization
  EnableVirtualTerminal;
{$ENDIF}

end.
