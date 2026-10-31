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

function LLMEncodePlain(var Tokenizer: TQNNTokenizer; const Text: string; const Hybrid: Boolean = False): TArray<LongInt>;

function LLMChatTokens(var Tokenizer: TQNNTokenizer; const Messages: TLLMMessages; const EnableThinking: Boolean; const PlainTemplate: Boolean = False; const Hybrid: Boolean = False
): TArray<LongInt>;

function LLMTokenBytes(var Tokenizer: TQNNTokenizer; const Token: Integer): RawByteString;

implementation

uses
  System.SysUtils, System.RegularExpressions, System.Generics.Collections,
  Winapi.Windows;

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

function LLMEncodePlain(var Tokenizer: TQNNTokenizer; const Text: string; const Hybrid: Boolean): TArray<LongInt>;
var
  Regex: TRegEx;
  Tokens: TList<LongInt>;
  Bytes: UTF8String;
  Encoded: string;
  Id, Covered: Integer;
  Input, Pattern: string;
begin
  if not Assigned(Tokenizer.VocabsDic) then
    raise EInvalidOp.Create('Load the Qwen3 tokenizer before encoding text');
  Tokens := TList<LongInt>.Create;
  try
    Input := Text;
    Pattern := QWEN3_SPLIT_PATTERN;
    if Hybrid then
    begin
      var Needed := NormalizeString(NormalizationC, PChar(Input), Length(Input), nil, 0);
      if Needed < 0 then
        raise EArgumentException.Create('Cannot normalize Qwen hybrid input');
      var Normalized: string;
      SetLength(Normalized, Needed);
      if Needed > 0 then
      begin
        Needed := NormalizeString(NormalizationC, PChar(Input), Length(Input), PChar(Normalized), Needed);
        if Needed < 0 then
          raise EArgumentException.Create('Cannot normalize Qwen hybrid input');
        SetLength(Normalized, Needed);
      end;
      Input := Normalized;
      Pattern := StringReplace(Pattern, '\p{L}+', '[\p{L}\p{M}]+', []);
      Pattern := StringReplace(Pattern, '[^\s\p{L}\p{N}]', '[^\s\p{L}\p{M}\p{N}]', []);
    end;
    Regex := TRegEx.Create(Pattern);
    Covered := 1;
    for var Match in Regex.Matches(Input) do
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
    if Covered <> Length(Input) + 1 then
      raise EArgumentException.Create('Qwen3 pre-tokenizer did not cover the input text');
    Result := Tokens.ToArray;
  finally
    Tokens.Free;
  end;
end;

function LLMChatTokens(var Tokenizer: TQNNTokenizer; const Messages: TLLMMessages; const EnableThinking: Boolean; const PlainTemplate: Boolean; const Hybrid: Boolean): TArray<LongInt>;
const
  ROLE_NAMES: array[TLLMRole] of string = ('system', 'user', 'assistant');
var
  Tokens: TList<LongInt>;
  LastUser, ClosingThink: Integer;
  Content, Reasoning: string;

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
    Tokens.AddRange(LLMEncodePlain(Tokenizer, Text, Hybrid));
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
    if Hybrid and (LastUser < 0) then
      raise EArgumentException.Create('A Qwen hybrid chat needs a user query');
    if Hybrid and EnableThinking then
    begin
      Special('<|im_start|>');
      Plain('system'#10 + 'Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, '
        + 'consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.');
      if (Messages[0].Role = lrSystem) and (Trim(Messages[0].Content) <> '') then
        Plain(#10#10 + Trim(Messages[0].Content));
      Special('<|im_end|>');
      Plain(#10);
    end;
    for var MessageIndex := 0 to High(Messages) do
    begin
      var Message := Messages[MessageIndex];
      if (Length(Message.Images) > 0) and (Message.Role <> lrUser) then
        raise EArgumentException.Create('Image inputs require a user observation message');
      if Hybrid and (Message.Role = lrSystem) then
      begin
        if MessageIndex > 0 then
          raise EArgumentException.Create('The hybrid system message must be first');
        if EnableThinking or (Trim(Message.Content) = '') then
          Continue;
      end;
      if (Ord(Message.Role) < Ord(Low(TLLMRole))) or (Ord(Message.Role) > Ord(High(TLLMRole))) then
        raise EArgumentException.Create('Unknown chat message role');
      Special('<|im_start|>');
      Content := Message.Content;
      // The official Qwen3 template omits old assistant reasoning before a new user query.
      if not Hybrid and not PlainTemplate and (Message.Role = lrAssistant) and (MessageIndex < LastUser) then
      begin
        if Pos('</think>', Content) > 0 then
        begin
          ClosingThink := Pos('</think>', Content);
          Delete(Content, 1, ClosingThink + Length('</think>') - 1);
          while (Content <> '') and (Content[1] = #10) do
            Delete(Content, 1, 1);
        end;
      end;
      if Hybrid then
        Content := Trim(Content);
      if Hybrid and (Message.Role = lrAssistant) then
      begin
        Reasoning := '';
        if Content.StartsWith('<think>') then
        begin
          ClosingThink := Pos('</think>', Content);
          if ClosingThink > 0 then
          begin
            Reasoning := Trim(Copy(Content, 8, ClosingThink - 8));
            Content := Trim(Copy(Content, ClosingThink + 8, MaxInt));
          end
          else
          begin
            Reasoning := Trim(Copy(Content, 8, MaxInt));
            Content := '';
          end;
        end;
        Plain('assistant'#10);
        Special('<think>');
        Plain(#10 + Reasoning + #10);
        Special('</think>');
        Plain(#10#10 + Content);
      end
      else
      begin
        Plain(ROLE_NAMES[Message.Role] + #10);
        for var Image in Message.Images do
        begin
          if not Hybrid then
            raise ENotSupportedException.Create('This LLM has no vision encoder; choose a vision analysis model');
          if (Message.Role <> lrUser) or (Image.TokenCount < 1) then
            raise EArgumentException.Create('Images require a user message and a prepared visual grid');
          Special('<|vision_start|>');
          for var J := 1 to Image.TokenCount do
            Special('<|image_pad|>');
          Special('<|vision_end|>');
        end;
        Plain(Content);
      end;
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
    if Hybrid and EnableThinking then
    begin
      Special('<think>');
      Plain(#10);
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

