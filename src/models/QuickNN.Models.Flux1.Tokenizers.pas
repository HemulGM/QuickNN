unit QuickNN.Models.Flux1.Tokenizers;

{ Native CLIP byte BPE and T5 Unigram tokenization. The precompiled Unicode
  map is read using the documented SentencePiece/Darts format; no tokenizer
  DLL is linked. References: openai/CLIP, huggingface/tokenizers and
  google/sentencepiece (their respective MIT/Apache-2.0/BSD licenses). }
{$POINTERMATH ON}

interface

uses
  System.SysUtils, System.Generics.Collections;

type
  TFlux1ClipTokenizer = class
  private
    FVocab, FRanks: TDictionary<string, Integer>;
    FBytes: array[Byte] of Char;
    function WordTokens(const Word: string): TArray<Integer>;
  public
    constructor Create(const Directory: string);
    destructor Destroy; override;
    function Encode(const Text: string; out EosPosition: Integer): TArray<Integer>;
  end;

  TFlux1T5Tokenizer = class
  private
    FIds, FSpecial: TDictionary<string, Integer>;
    FScores: TArray<Double>;
    FMap: TBytes;
    FTrieBytes, FMaxLength: Integer;
    FUnknownScore: Double;
    function Normalize(const Text: string): string;
    function WordTokens(const Word: string): TArray<Integer>;
  public
    constructor Create(const JsonFile: string);
    destructor Destroy; override;
    function Encode(const Text: string; const MaxLength: Integer = 512): TArray<Integer>;
  end;

implementation

uses
  System.Math, System.JSON, System.IOUtils, System.RegularExpressions,
  System.NetEncoding, Winapi.Windows;

constructor TFlux1ClipTokenizer.Create(const Directory: string);
var
  JSON: TJSONValue;
  NextCode: Integer;
begin
  inherited Create;
  FVocab := TDictionary<string, Integer>.Create;
  FRanks := TDictionary<string, Integer>.Create;
  JSON := TJSONObject.ParseJSONValue(TFile.ReadAllText(TPath.Combine(Directory, 'vocab.json'), TEncoding.UTF8));
  try
    if not (JSON is TJSONObject) then raise EArgumentException.Create('Invalid CLIP vocabulary');
    for var Pair in TJSONObject(JSON) do FVocab.Add(Pair.JsonString.Value, Pair.JsonValue.GetValue<Integer>);
  finally JSON.Free end;
  NextCode := 256;
  for var B := 0 to 255 do
    if ((B >= 33) and (B <= 126)) or ((B >= 161) and (B <= 172)) or (B >= 174) then
      FBytes[B] := Char(B)
    else begin FBytes[B] := Char(NextCode); Inc(NextCode) end;
  var Rank := 0;
  for var Line in TFile.ReadAllLines(TPath.Combine(Directory, 'merges.txt'), TEncoding.UTF8) do
    if (Line <> '') and not Line.StartsWith('#') then
    begin FRanks.AddOrSetValue(Line.Trim, Rank); Inc(Rank) end;
end;

destructor TFlux1ClipTokenizer.Destroy;
begin
  FVocab.Free; FRanks.Free; inherited;
end;

function TFlux1ClipTokenizer.WordTokens(const Word: string): TArray<Integer>;
var
  Parts: TList<string>;
  Best, Position, Rank: Integer;
begin
  Parts := TList<string>.Create;
  try
    var Bytes := TEncoding.UTF8.GetBytes(Word);
    for var B in Bytes do Parts.Add(FBytes[B]);
    if Parts.Count = 0 then Exit(nil);
    Parts[Parts.Count - 1] := Parts.Last + '</w>';
    while Parts.Count > 1 do
    begin
      Best := MaxInt; Position := -1;
      for var I := 0 to Parts.Count - 2 do
        if FRanks.TryGetValue(Parts[I] + ' ' + Parts[I + 1], Rank) and (Rank < Best) then
        begin Best := Rank; Position := I end;
      if Position < 0 then Break;
      // Merge all occurrences of the selected pair, as CLIP BPE specifies.
      var Left := Parts[Position]; var Right := Parts[Position + 1];
      var I := 0;
      while I < Parts.Count - 1 do
        if (Parts[I] = Left) and (Parts[I + 1] = Right) then
        begin Parts[I] := Left + Right; Parts.Delete(I + 1); Inc(I) end
        else Inc(I);
    end;
    SetLength(Result, Parts.Count);
    for var I := 0 to Parts.Count - 1 do
      if not FVocab.TryGetValue(Parts[I], Result[I]) then Result[I] := FVocab['<|endoftext|>'];
  finally Parts.Free end;
end;

function TFlux1ClipTokenizer.Encode(const Text: string; out EosPosition: Integer): TArray<Integer>;
const
  PATTERN = '(*UCP)<\|startoftext\|>|<\|endoftext\|>|''s|''t|''re|''ve|''m|''ll|''d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+';
var
  Tokens: TList<Integer>;
begin
  // The RTL HTML decoder uses NUL-terminated input. Preserve embedded NULs.
  var Parts := Text.Split([#0]);
  for var I := 0 to High(Parts) do
    Parts[I] := TNetEncoding.HTML.Decode(TNetEncoding.HTML.Decode(Parts[I]));
  var Clean := string.Join(#0, Parts);
  if Clean <> '' then
  begin
    var Count := NormalizeString(NormalizationC, PChar(Clean), Length(Clean), nil, 0);
    if Count <= 0 then RaiseLastOSError;
    var Normalized: string;
    SetLength(Normalized, Count);
    Count := NormalizeString(NormalizationC, PChar(Clean), Length(Clean), PChar(Normalized), Count);
    if Count <= 0 then RaiseLastOSError;
    SetLength(Normalized, Count); Clean := Normalized;
  end;
  Clean := TRegEx.Replace(Clean, '(*UCP)\s+', ' ');
  Clean := TRegEx.Replace(Clean, '(*UCP)^\s+|\s+$', '').ToLowerInvariant;
  Tokens := TList<Integer>.Create;
  try
    Tokens.Add(FVocab['<|startoftext|>']);
    for var Match in TRegEx.Matches(Clean, PATTERN) do
      if FVocab.ContainsKey(Match.Value) and Match.Value.StartsWith('<|') then
        Tokens.Add(FVocab[Match.Value])
      else
        for var Token in WordTokens(Match.Value) do Tokens.Add(Token);
    EosPosition := Min(76, Tokens.Count);
    SetLength(Result, 77);
    for var I := 0 to 76 do
      if I < EosPosition then Result[I] := Tokens[I]
      else Result[I] := FVocab['<|endoftext|>'];
    for var I := 0 to EosPosition - 1 do
      if Result[I] = FVocab['<|endoftext|>'] then begin EosPosition := I; Break end;
  finally Tokens.Free end;
end;

constructor TFlux1T5Tokenizer.Create(const JsonFile: string);
var
  JSON, Model, Normalizer: TJSONValue;
  Vocab: TJSONArray;
begin
  inherited Create;
  FIds := TDictionary<string, Integer>.Create;
  FSpecial := TDictionary<string, Integer>.Create;
  JSON := TJSONObject.ParseJSONValue(TFile.ReadAllText(JsonFile, TEncoding.UTF8));
  try
    if not (JSON is TJSONObject) then raise EArgumentException.Create('Invalid T5 tokenizer JSON');
    Model := JSON.FindValue('model');
    if (Model = nil) or (Model.GetValue<string>('type', '') <> 'Unigram')
      or Model.GetValue<Boolean>('byte_fallback', False) then
      raise EArgumentException.Create('Expected T5 Unigram tokenizer without byte fallback');
    Vocab := Model.GetValue<TJSONArray>('vocab');
    SetLength(FScores, Vocab.Count); FUnknownScore := 0;
    for var I := 0 to Vocab.Count - 1 do
    begin
      var Pair := TJSONArray(Vocab.Items[I]);
      var Piece := Pair.Items[0].Value;
      FScores[I] := Pair.Items[1].GetValue<Double>;
      FIds.Add(Piece, I);
      FMaxLength := Max(FMaxLength, Length(Piece));
      FUnknownScore := Min(FUnknownScore, FScores[I]);
    end;
    FUnknownScore := FUnknownScore - 10;
    var Added := JSON.GetValue<TJSONArray>('added_tokens');
    if Added <> nil then
      for var Item in Added do
        FSpecial.AddOrSetValue(Item.GetValue<string>('content'), Item.GetValue<Integer>('id'));
    Normalizer := JSON.FindValue('normalizer');
    var Normalizers := Normalizer.GetValue<TJSONArray>('normalizers');
    if Normalizers = nil then raise EArgumentException.Create('Missing T5 normalization sequence');
    for var Item in Normalizers do
      if Item.GetValue<string>('type', '') = 'Precompiled' then
        FMap := TNetEncoding.Base64.DecodeStringToBytes(Item.GetValue<string>('precompiled_charsmap'));
    if Length(FMap) < 8 then raise EArgumentException.Create('Missing T5 precompiled Unicode map');
    Move(FMap[0], FTrieBytes, 4);
    if (FTrieBytes < 4) or (FTrieBytes mod 4 <> 0) or (FTrieBytes > Length(FMap) - 4) then
      raise EArgumentException.Create('Invalid T5 Unicode trie');
  finally JSON.Free end;
end;

destructor TFlux1T5Tokenizer.Destroy;
begin
  FIds.Free; FSpecial.Free; inherited;
end;

function TFlux1T5Tokenizer.Normalize(const Text: string): string;
var
  Bytes, Output: UTF8String;
  Position, Consumed, BestLength, MapOffset, Index, EndIndex: Integer;
  Node, UnitValue: UInt32;

  function Trie(const NodeIndex: UInt32): UInt32;
  begin
    if NodeIndex >= UInt32(FTrieBytes div 4) then Exit($FFFFFFFF);
    Move(FMap[4 + NodeIndex * 4], Result, 4);
  end;

  function Offset(const Value: UInt32): UInt32;
  begin
    Result := (Value shr 10) shl ((Value and 512) shr 6);
  end;

begin
  Bytes := UTF8String(Text); Output := ''; Position := 1;
  while Position <= Length(Bytes) do
  begin
    Node := Offset(Trie(0)); BestLength := 0; MapOffset := -1;
    Index := Position;
    while Index <= Length(Bytes) do
    begin
      Node := Node xor Byte(Bytes[Index]);
      UnitValue := Trie(Node);
      if (UnitValue and $800000FF) <> UInt32(Byte(Bytes[Index])) then Break;
      Node := Node xor Offset(UnitValue);
      if (UnitValue and 256) <> 0 then
      begin
        BestLength := Index - Position + 1;
        MapOffset := Integer(Trie(Node) and $7FFFFFFF);
      end;
      Inc(Index);
    end;
    if MapOffset >= 0 then
    begin
      Index := 4 + FTrieBytes + MapOffset;
      if (Index < 4 + FTrieBytes) or (Index >= Length(FMap)) then
        raise EArgumentException.Create('T5 Unicode mapping offset is invalid');
      EndIndex := Index;
      while (EndIndex < Length(FMap)) and (FMap[EndIndex] <> 0) do Inc(EndIndex);
      if EndIndex = Length(FMap) then raise EArgumentException.Create('Unterminated T5 Unicode mapping');
      var Piece: UTF8String;
      SetLength(Piece, EndIndex - Index);
      if Piece <> '' then Move(FMap[Index], Piece[1], Length(Piece));
      Output := Output + Piece;
      Inc(Position, BestLength);
    end
    else
    begin
      Consumed := 1;
      if Byte(Bytes[Position]) >= $F0 then Consumed := 4
      else if Byte(Bytes[Position]) >= $E0 then Consumed := 3
      else if Byte(Bytes[Position]) >= $C0 then Consumed := 2;
      Output := Output + Copy(Bytes, Position, Consumed); Inc(Position, Consumed);
    end;
  end;
  Result := string(Output).TrimRight;
  Result := TRegEx.Replace(Result, ' {2,}', #$2581);
  Result := Result.Replace(' ', #$2581);
  if (Result <> '') and not Result.StartsWith(#$2581) then Result := #$2581 + Result;
end;

function TFlux1T5Tokenizer.WordTokens(const Word: string): TArray<Integer>;
var
  Scores: TArray<Double>;
  Back, Ids: TArray<Integer>;
  Tokens: TList<Integer>;
  Token, N, ScalarLength: Integer;
begin
  N := Length(Word);
  SetLength(Scores, N + 1); SetLength(Back, N + 1); SetLength(Ids, N + 1);
  for var I := 1 to N do Scores[I] := NegInfinity;
  for var I := 0 to N - 1 do
    if not IsInfinite(Scores[I]) then
    begin
      ScalarLength := 1;
      if (Ord(Word[I + 1]) >= $D800) and (Ord(Word[I + 1]) <= $DBFF) and (I + 1 < N) then ScalarLength := 2;
      var HasScalar := False;
      for var L := 1 to Min(FMaxLength, N - I) do
        if FIds.TryGetValue(Word.Substring(I, L), Token) then
        begin
          if L = ScalarLength then HasScalar := True;
          if Scores[I] + FScores[Token] > Scores[I + L] then
          begin Scores[I + L] := Scores[I] + FScores[Token]; Back[I + L] := I; Ids[I + L] := Token end;
        end;
      if not HasScalar and (Scores[I] + FUnknownScore > Scores[I + ScalarLength]) then
      begin Scores[I + ScalarLength] := Scores[I] + FUnknownScore; Back[I + ScalarLength] := I; Ids[I + ScalarLength] := 2 end;
    end;
  Tokens := TList<Integer>.Create;
  try
    while N > 0 do
    begin
      if (Ids[N] <> 2) or (Tokens.Count = 0) or (Tokens.Last <> 2) then Tokens.Add(Ids[N]);
      if Back[N] >= N then raise EInvalidOp.Create('Invalid T5 tokenizer path');
      N := Back[N];
    end;
    Result := Tokens.ToArray;
    for var I := 0 to Length(Result) div 2 - 1 do
    begin
      Token := Result[I]; Result[I] := Result[High(Result) - I]; Result[High(Result) - I] := Token;
    end;
  finally Tokens.Free end;
end;

function TFlux1T5Tokenizer.Encode(const Text: string; const MaxLength: Integer): TArray<Integer>;
var
  Tokens: TList<Integer>;
begin
  if (MaxLength < 2) or (MaxLength > 1024) or (Length(Text) > 65536) then
    raise EArgumentOutOfRangeException.Create('T5 prompt/token limit');
  Tokens := TList<Integer>.Create;
  try
    // Special tokens are isolated before normalizing ordinary text.
    var Start := 1; var Position := 1;
    while Position <= Length(Text) do
    begin
      var Match := '';
      var Id := -1;
      if Text[Position] = '<' then
        for var Pair in FSpecial do
          if (Length(Pair.Key) > Length(Match)) and (Copy(Text, Position, Length(Pair.Key)) = Pair.Key) then
          begin Match := Pair.Key; Id := Pair.Value end;
      if Id >= 0 then
      begin
        var Segment := Normalize(Copy(Text, Start, Position - Start));
        for var Word in Segment.Split([#$2581]) do
          if Word <> '' then for var Token in WordTokens(#$2581 + Word) do Tokens.Add(Token);
        Tokens.Add(Id); Inc(Position, Length(Match)); Start := Position;
      end
      else Inc(Position);
    end;
    var Segment := Normalize(Copy(Text, Start, MaxInt));
    for var Word in Segment.Split([#$2581]) do
      if Word <> '' then for var Token in WordTokens(#$2581 + Word) do Tokens.Add(Token);
    // The checkpoint uses T5TokenizerFast: its postprocessor always appends EOS.
    SetLength(Result, MaxLength);
    var Count := Min(MaxLength - 1, Tokens.Count);
    for var I := 0 to Count - 1 do Result[I] := Tokens[I];
    Result[Count] := 1;
  finally Tokens.Free end;
end;

end.
