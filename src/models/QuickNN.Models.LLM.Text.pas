unit QuickNN.Models.LLM.Text;

interface

uses
  QuickNN.Models.Qwen3, QuickNN.Models.LLM.Types;

type
  // Token boundaries can bisect a UTF-8 code point. Keep its bytes until complete.
  TLLMUTF8Stream = record
  private
    FPending: RawByteString;
  public
    function Push(const Bytes: RawByteString): string;
    function Finish: string;
    procedure Reset;
    property PendingBytes: RawByteString read FPending write FPending;
  end;

function LLMEncodePlain(var Tokenizer: TQNNTokenizer; const Text: string): TArray<LongInt>;

function LLMChatTokens(var Tokenizer: TQNNTokenizer; const Messages: TLLMMessages; const EnableThinking: Boolean; const PlainTemplate: Boolean = False
): TArray<LongInt>;

function LLMTokenBytes(var Tokenizer: TQNNTokenizer; const Token: Integer): RawByteString;

implementation

uses
  System.SysUtils, System.RegularExpressions, System.Generics.Collections;

const
  // Qwen3 tokenizer.json Split.Regex; UCP gives \s the Unicode semantics of the source tokenizer.
  QWEN3_SPLIT_PATTERN =
 '(*UCP)(?i:''s|''t|''re|''ve|''m|''ll|''d)|[^\r\n\p{L}\p{N}]?\p{L}+'
    + '|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+';

var
  ByteToCharacter: array[Byte] of Char;
  CharacterToByte: array[0..511] of SmallInt;

procedure InitializeByteMap;
var
  NextCode, Code: Integer;
begin
  for var I := Low(CharacterToByte) to High(CharacterToByte) do
    CharacterToByte[I] := -1;
  NextCode := 256;
  for var B := 0 to 255 do
  begin
    if ((B >= 33) and (B <= 126)) or ((B >= 161) and (B <= 172)) or (B >= 174) then
      Code := B
    else
    begin
      Code := NextCode;
      Inc(NextCode);
    end;
    ByteToCharacter[B] := Char(Code);
    CharacterToByte[Code] := B;
  end;
end;

function LLMEncodePlain(var Tokenizer: TQNNTokenizer; const Text: string): TArray<LongInt>;
var
  Regex: TRegEx;
  Tokens: TList<LongInt>;
  Bytes: UTF8String;
  Encoded: string;
  Id, Covered: Integer;
begin
  if not Assigned(Tokenizer.VocabsDic) then
    raise EInvalidOp.Create('Load the Qwen3 tokenizer before encoding text');
  Tokens := TList<LongInt>.Create;
  try
    Regex := TRegEx.Create(QWEN3_SPLIT_PATTERN);
    Covered := 1;
    for var Match in Regex.Matches(Text) do
    begin
      if Match.Index <> Covered then
        raise EArgumentException.Create('Qwen3 pre-tokenizer did not cover the input text');
      Inc(Covered, Match.Length);
      Bytes := UTF8String(Match.Value);
      SetLength(Encoded, Length(Bytes));
      for var I := 1 to Length(Bytes) do
        Encoded[I] := ByteToCharacter[Byte(Bytes[I])];
      for var Piece in Tokenizer.BPEEncodeWord(UTF8String(Encoded)) do
      begin
        Id := Tokenizer.GetId(Piece);
        if Id < 0 then
          raise EArgumentException.Create('The tokenizer has no vocabulary entry for a byte-level BPE piece');
        Tokens.Add(Id);
      end;
    end;
    if Covered <> Length(Text) + 1 then
      raise EArgumentException.Create('Qwen3 pre-tokenizer did not cover the input text');
    Result := Tokens.ToArray;
  finally
    Tokens.Free;
  end;
end;

function LLMChatTokens(var Tokenizer: TQNNTokenizer; const Messages: TLLMMessages; const EnableThinking: Boolean; const PlainTemplate: Boolean): TArray<LongInt>;
const
  ROLE_NAMES: array[TLLMRole] of string = ('system', 'user', 'assistant');
var
  Tokens: TList<LongInt>;
  LastUser, ClosingThink: Integer;
  Content: string;

  procedure Special(const Text: RawByteString);
  var
    Id: Integer;
  begin
    Id := Tokenizer.GetId(Text);
    if Id < 0 then
      raise EArgumentException.Create('Missing Qwen3 chat token: ' + UTF8ToString(Text));
    Tokens.Add(Id);
  end;

  procedure Plain(const Text: string);
  begin
    Tokens.AddRange(LLMEncodePlain(Tokenizer, Text));
  end;

begin
  if Length(Messages) = 0 then
    raise EArgumentException.Create('A chat needs at least one message');
  Tokens := TList<LongInt>.Create;
  try
    LastUser := -1;
    for var I := 0 to High(Messages) do
      if Messages[I].Role = lrUser then
        LastUser := I;
    for var MessageIndex := 0 to High(Messages) do
    begin
      var Message := Messages[MessageIndex];
      if (Ord(Message.Role) < Ord(Low(TLLMRole))) or (Ord(Message.Role) > Ord(High(TLLMRole))) then
        raise EArgumentException.Create('Unknown chat message role');
      Special('<|im_start|>');
      Content := Message.Content;
      // The official Qwen3 template omits old assistant reasoning before a new user query.
      if not PlainTemplate and (Message.Role = lrAssistant) and (MessageIndex < LastUser) then
      begin
        if Pos('</think>', Content) > 0 then
        begin
          ClosingThink := Pos('</think>', Content);
          Delete(Content, 1, ClosingThink + Length('</think>') - 1);
          while (Content <> '') and (Content[1] = #10) do
            Delete(Content, 1, 1);
        end;
      end;
      // Ordinary role/content text is tokenized together so leading newlines keep BPE semantics.
      // Special control IDs are inserted only by this framing code.
      Plain(ROLE_NAMES[Message.Role] + #10 + Content);
      Special('<|im_end|>');
      Plain(#10);
    end;
    Special('<|im_start|>');
    Plain('assistant' + #10);
    if not PlainTemplate and not EnableThinking then
    begin
      Special('<think>');
      Plain(#10#10);
      Special('</think>');
      Plain(#10#10);
    end;
    Result := Tokens.ToArray;
  finally
    Tokens.Free;
  end;
end;

function LLMTokenBytes(var Tokenizer: TQNNTokenizer; const Token: Integer): RawByteString;
var
  Encoded: string;
  Code: Integer;
begin
  if (Token < 0) or (Token >= Length(Tokenizer.Vocabs)) or (Tokenizer.Vocabs[Token] = '') then
    raise EArgumentOutOfRangeException.Create('Qwen3 token is outside the vocabulary');
  Encoded := UTF8ToString(Tokenizer.GetToken(Token));
  SetLength(Result, Length(Encoded));
  for var I := 1 to Length(Encoded) do
  begin
    Code := Ord(Encoded[I]);
    if (Code > High(CharacterToByte)) or (CharacterToByte[Code] < 0) then
      raise EArgumentException.Create('Qwen3 vocabulary contains an invalid byte-level character');
    Result[I] := AnsiChar(CharacterToByte[Code]);
  end;
end;

function TLLMUTF8Stream.Push(const Bytes: RawByteString): string;
var
  Input, Complete: RawByteString;
  I, Need, B, Next: Integer;
  Valid: Boolean;
begin
  Input := FPending + Bytes;
  FPending := '';
  Complete := '';
  I := 1;
  while I <= Length(Input) do
  begin
    B := Byte(Input[I]);
    if B < $80 then
      Need := 1
    else if (B >= $C2) and (B <= $DF) then
      Need := 2
    else if (B >= $E0) and (B <= $EF) then
      Need := 3
    else if (B >= $F0) and (B <= $F4) then
      Need := 4
    else
      Need := 0;
    Valid := Need <> 0;
    if Valid then
      for var J := 1 to Need - 1 do
      begin
        if I + J > Length(Input) then
          Break;
        Next := Byte(Input[I + J]);
        if (Next < $80) or (Next > $BF) then
          Valid := False;
      end;
    if Valid and (Need >= 3) and (I < Length(Input)) then
    begin
      Next := Byte(Input[I + 1]);
      if ((B = $E0) and (Next < $A0))
        or ((B = $ED) and (Next > $9F))
        or ((B = $F0) and (Next < $90))
        or ((B = $F4) and (Next > $8F)) then
        Valid := False;
    end;
    if Valid and (Length(Input) - I + 1 < Need) then
    begin
      FPending := Copy(Input, I, MaxInt);
      Break;
    end;
    if Valid then
    begin
      Complete := Complete + Copy(Input, I, Need);
      Inc(I, Need);
    end
    else
    begin
      Complete := Complete + RawByteString(UTF8Encode(string(#$FFFD)));
      Inc(I);
    end;
  end;
  Result := UTF8ToString(Complete);
end;

function TLLMUTF8Stream.Finish: string;
begin
  if FPending <> '' then
    Result := #$FFFD
  else
    Result := '';
  FPending := '';
end;

procedure TLLMUTF8Stream.Reset;
begin
  FPending := '';
end;

initialization
  InitializeByteMap;

end.

