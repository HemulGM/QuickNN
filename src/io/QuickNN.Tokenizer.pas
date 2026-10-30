unit QuickNN.Tokenizer;

{$C+} // enable assertions
{$H+}
{$POINTERMATH on}

interface

uses
  System.SysUtils, QuickNN.Common;

type
  TQNNBPEMerge = record
    Left, Right, &Result, Priority: LongInt;
  end;

  { TQNNTokenizer }

  TQNNTokenizer = record
  private
    VocabsDic: TVocabs;
  public
    (* Vocabulary *)
    Vocabs: TArray<RawByteString>;

    (* BPE merges *)
    Merges: TArray<TQNNBPEMerge>;

    (* Special token IDs *)
    PadId, UnkId, BosId, EosId: LongInt;

    (* Configuration *)
    MaxLength: LongInt;
    AddBos, AddEos: Boolean;
    constructor Load(const FileName: string);
    function BPEEncodeWord(const Str: RawByteString): TArray<LongInt>;
    function Tokenize(const AText: RawByteString; MaxLen: LongInt = 0): TArray<LongInt>;
    function DeTokenize(const Tokens: TArray<LongInt>): RawByteString;
    procedure Free();
  end;

implementation

function GetListOfWords(const Str: RawByteString): TArray<RawByteString>;
var
  I, Start: NativeInt;
begin
  I := 1;
  Start := 1;
  Result := nil;
  while I <= Length(Str) do
  begin
    case Str[I] of
      TAB, LF, CR, SP, // skip white spaces
        '!'..'/', ':'..'@', '['..'`', '{'..'~': // insert punctuation
        begin
          Inc(I);
          Insert(Copy(Str, Start, I - Start), Result, Length(Result));
          Start := I;
        end;

      // TAB, LF, CR, SP : // skip white spaces
      //  begin
      //    inc(i);
      //    inc(start)
      //  end
    else
      begin
        while (I <= Length(Str)) and not (Str[I] in WORD_SEPERATORS) do // a word found, iterate to get number of chars
          Inc(I, UTF8CharSize(Str[I]));
        Insert(Copy(Str, Start, I - Start), Result, Length(Result));
        Start := I;
      end;
    end;
  end;
end;

{ TQNNTokenizer }

constructor TQNNTokenizer.Load(const FileName: string);
const
  TOK_MAGIC = 'FTOK';
var
  F: file;
  S: string;
  Config: array[0..7] of LongWord;
  Merge: array[0..2] of LongWord;
  I: Integer;
  Len: Word;
begin
  if not Assigned(VocabsDic) then
    VocabsDic := TVocabs.Create;
  Assert(FileExists(FileName), 'ERROR : File not found ' + FileName);
  AssignFile(F, FileName);
  Reset(F, 1);
  try
    SetLength(S, Length(TOK_MAGIC));
    BlockRead(F, S[1], Length(TOK_MAGIC));
    Assert(S = TOK_MAGIC, 'ERROR : Invalid magic header');
    BlockRead(F, Config, SizeOf(Config));
    PadId := Config[2];
    UnkId := Config[3];
    BosId := Config[4];
    EosId := Config[5];
    MaxLength := Config[6];
    AddBos := Boolean(Config[7] and 1);
    AddEos := Boolean((Config[7] shr 1) and 1);
    VocabsDic.Clear;
    SetLength(Vocabs, Config[0]);
    SetLength(Merges, Config[1]);
    for I := 0 to High(Vocabs) do
    begin
      BlockRead(F, Len, SizeOf(Len));
      SetLength(Vocabs[I], Len);
      BlockRead(F, Vocabs[I][1], Len);
      VocabsDic.AddOrSetValue(Vocabs[I], I);
    end;

    for I := 0 to High(Merges) do
    begin
      BlockRead(F, Merge, SizeOf(Merge));
      Merges[I].Left := Merge[0];
      Merges[I].Right := Merge[1];
      Merges[I].&Result := Merge[2];
      Merges[I].Priority := I;
    end;
  finally
    CloseFile(F);
  end
end;

function TQNNTokenizer.BPEEncodeWord(const Str: RawByteString): TArray<LongInt>;
var
  I, M, N, Id, Len, BestIdx, BestPriority: LongInt;
  S: RawByteString;
  Changed: Boolean;
begin
  Assert(Assigned(VocabsDic), 'ERROR : no vocabs loaded, use <load> function to load vocab 1st.');
  SetLength(Result, Length(Str));
  I := 1;
  N := 0;
  while I <= Length(Str) do
  begin
    Len := UTF8CharSize(Str[I]);
    S := Copy(Str, I, Len);
    if VocabsDic.TryGetValue(S, Id) then
    begin
      Result[N] := Id;
      Inc(N);
    end
    else
    begin
      for M := 1 to Len do
      begin
        if not VocabsDic.TryGetValue(UTF8String(Format('<0x%.2X>', [Ord(S[M])])), Id) then
          Id := UnkId;
        Result[N] := Id;
        Inc(N);
      end;
    end;
    Inc(I, Len);
  end;

  Changed := True;
  while Changed and (N > 1) do
  begin
    Changed := False;
    BestIdx := -1;
    BestPriority := Length(Merges) + 1;
    for I := 0 to N - 2 do
      for M := 0 to High(Merges) do
        if (Merges[M].Left = Result[I]) and (Merges[M].Right = Result[I + 1]) then
          if Merges[M].Priority < BestPriority then
          begin
            BestPriority := Merges[M].Priority;
            BestIdx := I
          end;
    if BestIdx >= 0 then
      for M := 0 to High(Merges) do
        if (Merges[M].Left = Result[BestIdx]) and (Merges[M].Right = Result[BestIdx + 1]) then
        begin
          Result[BestIdx] := Merges[M].&Result;
          // shift result left
          for I := BestIdx + 1 to N - 2 do
            Result[I] := Result[I + 1];
          Dec(N);
          Changed := True;
          Break
        end;
  end;
  SetLength(Result, N)

end;

function TQNNTokenizer.Tokenize(const AText: RawByteString; MaxLen: LongInt): TArray<LongInt>;
var
  Words: TArray<RawByteString>;
  WordTokens: TArray<LongInt>;
  Total, W, I: LongInt;

  procedure AppendToken(const Token: LongInt);
  begin
    Insert(Token, Result, Length(Result));
    Inc(Total);
  end;

begin
  Assert(Assigned(VocabsDic), 'ERROR : no vocabs loaded, use <load> function to load vocab 1st.');
  Result := nil;
  if MaxLen = 0 then
    MaxLen := MaxLength;

  Words := GetListOfWords(AText);
  Total := 0;

  if AddBos and (BosId >= 0) then
    AppendToken(BosId);

  W := 0;
  while (W < Length(Words)) and (Length(Result) < MaxLen - Ifthen(AddEos, 1, 0)) do
  begin
    WordTokens := BPEEncodeWord(Words[W]);
    I := 0;
    while (I < Length(WordTokens)) and (Length(Result) < MaxLen - Ifthen(AddEos, 1, 0)) do
    begin
      AppendToken(WordTokens[I]);
      Inc(I)
    end;

    Inc(W)
  end;

  if AddEos and (EosId >= 0) and (Length(Result) < MaxLen) then
    AppendToken(EosId);
end;

function TQNNTokenizer.DeTokenize(const Tokens: TArray<LongInt>): RawByteString;
var
  I, J, Id, TotalLen: Integer;
begin
  Assert(Assigned(VocabsDic), 'ERROR : no vocabs loaded, use <load> function to load vocab 1st.');
  Result := '';
  if not Assigned(Tokens) then
    Exit;
  TotalLen := 0;
  for I := 0 to High(Tokens) do
  begin
    Id := Tokens[I];
    if (Id >= 0) and (Id < Length(Vocabs)) then
      Inc(TotalLen, Length(Vocabs[Id]))
  end;
  SetLength(Result, TotalLen);

  J := 1;

  for I := 0 to High(Tokens) do
  begin
    Id := Tokens[I];
    if (Id >= 0) and (Id < Length(Vocabs)) then
    begin
      if (Id = BosId) or (Id = EosId) or (Id = PadId) then
        Continue;
      Move(Vocabs[Id][1], Result[J], Length(Vocabs[Id]));
      Inc(J, Length(Vocabs[Id]))
    end;
  end;

  SetLength(Result, J - 1)
end;

procedure TQNNTokenizer.Free();
begin
  FreeAndNil(VocabsDic);
  Vocabs := nil;
  Merges := nil
end;

function CreateSimpleTokenizer(): TQNNTokenizer;
var
  I: LongInt;
  Buf: RawByteString;
begin

  Result.PadId := 256;
  Result.UnkId := 257;
  Result.BosId := 258;
  Result.EosId := 259;
  Result.MaxLength := QNN_MAX_SEQ_LEN;
  Result.AddBos := True;
  Result.AddEos := True;

  SetLength(Result.Vocabs, 256 + 4);
  Result.VocabsDic := TVocabs.Create;
  (* Create vocabulary *)
  for I := 0 to 255 do
  begin

    if (I >= 32) and (I < 127) then
      Buf := AnsiChar(I)
    else
      Buf := UTF8String(Format('<0x%.2X>', [I]));

    Result.Vocabs[I] := Buf;
    Result.VocabsDic.TryAdd(Result.Vocabs[I], I);
  end;
  Result.Vocabs[256] := '<pad>';
  Result.Vocabs[257] := '<unk>';
  Result.Vocabs[258] := '<bos>';
  Result.Vocabs[259] := '<eos>';

  Result.VocabsDic.TryAdd('<pad>', 256);
  Result.VocabsDic.TryAdd('<unk>', 257);
  Result.VocabsDic.TryAdd('<bos>', 258);
  Result.VocabsDic.TryAdd('<eos>', 259);
end;

// const
//  TXT_ANSI = 'the quick brown fox jumps over the lazy dog '; // ansi
//  TXT_UTF8  ='أبجد هوز حطي كلمن';                            // utf-8
// var t: TQNNTokenizer;
//    tokens : TArray<longint>;
//    s : RawByteString;
initialization

  // t := createSimpleTokenizer();
  //
  // tokens := t.tokenize(TXT_ANSI+TXT_UTF8);
  // s := t.deTokenize(tokens);
  // t.free;
  // writeln(s);
  // readln



finalization

end.

