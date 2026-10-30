unit QuickNN.LLMChat.Console;

interface

uses
  Winapi.Windows;

type
  TChatConsole = class
  private
    FInput, FOutput: THandle;
    FInputConsole, FOutputConsole, FEndOfInput, FFirstLine: Boolean;
    FPending: RawByteString;
    function DecodeLine(const Bytes: RawByteString): string;
  public
    constructor Create;
    procedure Write(const Text: string);
    procedure WriteLine(const Text: string = '');
    function ReadLine(out Text: string): Boolean;
  end;

procedure InstallChatInterruptHandler;

procedure RemoveChatInterruptHandler;

procedure ClearChatInterrupt;

function ChatInterrupted: Boolean;

implementation

uses
  System.SysUtils,
  QuickNN.Models.LLM.Text;

var
  InterruptRequested: LongInt;

function ConsoleControl(const Event: DWORD): BOOL; stdcall;
begin
  Result := (Event = CTRL_C_EVENT) or (Event = CTRL_BREAK_EVENT);
  if Result then
    InterlockedExchange(InterruptRequested, 1);
end;

procedure InstallChatInterruptHandler;
begin
  if not SetConsoleCtrlHandler(@ConsoleControl, True) then
    RaiseLastOSError;
end;

procedure RemoveChatInterruptHandler;
begin
  SetConsoleCtrlHandler(@ConsoleControl, False);
end;

procedure ClearChatInterrupt;
begin
  InterlockedExchange(InterruptRequested, 0);
end;

function ChatInterrupted: Boolean;
begin
  Result := InterlockedCompareExchange(InterruptRequested, 0, 0) <> 0;
end;

constructor TChatConsole.Create;
var
  Mode: DWORD;
begin
  inherited;
  FInput := GetStdHandle(STD_INPUT_HANDLE);
  FOutput := GetStdHandle(STD_OUTPUT_HANDLE);
  FInputConsole := GetConsoleMode(FInput, Mode);
  FOutputConsole := GetConsoleMode(FOutput, Mode);
  FFirstLine := True;
end;

procedure TChatConsole.Write(const Text: string);
var
  Written, Remaining: DWORD;
  Position: NativeInt;
  Bytes: UTF8String;
begin
  if Text = '' then
    Exit;
  Position := 0;
  if FOutputConsole then
  begin
    while Position < Length(Text) do
    begin
      Remaining := Length(Text) - Position;
      if not WriteConsoleW(FOutput, PChar(Text) + Position, Remaining, Written, nil) then
        RaiseLastOSError;
      if Written = 0 then
        raise EInOutError.Create('Console output was closed');
      Inc(Position, Written);
    end;
  end
  else
  begin
    Bytes := UTF8String(Text);
    while Position < Length(Bytes) do
    begin
      Remaining := Length(Bytes) - Position;
      if not WriteFile(FOutput, (PAnsiChar(Bytes) + Position)^, Remaining, Written, nil) then
        RaiseLastOSError;
      if Written = 0 then
        raise EInOutError.Create('Console output was closed');
      Inc(Position, Written);
    end;
  end;
end;

procedure TChatConsole.WriteLine(const Text: string);
begin
  Write(Text + #13#10);
end;

function TChatConsole.DecodeLine(const Bytes: RawByteString): string;
var
  Stream: TLLMUTF8Stream;
  Input: RawByteString;
begin
  Input := Bytes;
  if FFirstLine then
  begin
    FFirstLine := False;
    if Copy(Input, 1, 3) = RawByteString(AnsiChar($EF) + AnsiChar($BB) + AnsiChar($BF)) then
      Delete(Input, 1, 3);
  end;
  Stream.Reset;
  Result := Stream.Push(Input) + Stream.Finish;
end;

function TChatConsole.ReadLine(out Text: string): Boolean;
var
  WideBuffer: array[0..1023] of Char;
  Buffer: array[0..4095] of AnsiChar;
  ReadCount, Error: DWORD;
  LineEnd, OldLength: Integer;
  Piece, Bytes: RawByteString;
  WidePiece: string;
begin
  Text := '';
  if FInputConsole then
  begin
    repeat
      if not ReadConsoleW(FInput, @WideBuffer[0], Length(WideBuffer), ReadCount, nil) then
      begin
        if ChatInterrupted then
        begin
          Text := '';
          WriteLine;
          Exit(True);
        end;
        RaiseLastOSError;
      end;
      if ReadCount = 0 then
        Exit(Text <> '');
      SetString(WidePiece, PChar(@WideBuffer[0]), ReadCount);
      if (Text = '') and (WidePiece <> '') and (WidePiece[1] = #26) then
        Exit(False);
      Text := Text + WidePiece;
      if (Pos(#10, Text) > 0) or (Pos(#13, Text) > 0) then
      begin
        while (Text <> '') and CharInSet(Text[Length(Text)], [#10, #13]) do
          Delete(Text, Length(Text), 1);
        Exit(True);
      end;
    until False;
  end;
  repeat
    LineEnd := Pos(AnsiChar(#10), FPending);
    if LineEnd > 0 then
    begin
      Bytes := Copy(FPending, 1, LineEnd - 1);
      Delete(FPending, 1, LineEnd);
      if (Bytes <> '') and (Bytes[Length(Bytes)] = #13) then
        Delete(Bytes, Length(Bytes), 1);
      Text := DecodeLine(Bytes);
      Exit(True);
    end;
    if FEndOfInput then
    begin
      Result := FPending <> '';
      if Result then
        Text := DecodeLine(FPending);
      FPending := '';
      Exit;
    end;
    if not ReadFile(FInput, Buffer, SizeOf(Buffer), ReadCount, nil) then
    begin
      Error := GetLastError;
      if (Error = ERROR_BROKEN_PIPE) or (Error = ERROR_HANDLE_EOF) then
        ReadCount := 0
      else
        RaiseLastOSError(Error);
    end;
    if ReadCount = 0 then
      FEndOfInput := True
    else
    begin
      OldLength := Length(FPending);
      SetLength(Piece, ReadCount);
      Move(Buffer, Piece[1], ReadCount);
      SetLength(FPending, OldLength + Integer(ReadCount));
      Move(Piece[1], FPending[OldLength + 1], ReadCount);
    end;
  until False;
end;

end.
