unit QuickNN.Models.Qwen21.Tokenizer;

{ SPDX-License-Identifier: Apache-2.0
  Native Pascal implementation; architecture and tensor conventions follow
  Qwen-Image 2.1 / Diffusers and ComfyUI / comfy-kitchen (Apache-2.0).
  Reference authors: Qwen-Image Team, Hugging Face Team and Comfy Org contributors.
  https://github.com/huggingface/diffusers/tree/main/src/diffusers/models
  https://github.com/Comfy-Org/ComfyUI/tree/master/comfy/ldm/qwen_image21
  https://github.com/Comfy-Org/comfy-kitchen/blob/main/comfy_kitchen/tensor/int8_utils.py
  See LICENSE.QwenImage21.txt in the repository root. }

{$POINTERMATH ON}

interface

uses
  System.SysUtils, System.Generics.Collections, System.RegularExpressions;

type
  TQwen21Tokenizer = class
  private
    FVocab, FMerges, FSpecial: TDictionary<string, Integer>;
    FByteChars: array[0..255] of Char;
    FPattern: TRegEx;
    procedure EncodePlain(const Text: string; const Output: TList<Integer>);
    procedure EncodePiece(const Text: string; const Output: TList<Integer>);
  public
    constructor Create(const FileName: string);
    destructor Destroy; override;
    function Encode(const Text: string): TArray<Integer>;
    function Chat(const Prompt: string; out DropTokens: Integer): TArray<Integer>;
  end;

implementation

uses
  System.JSON, System.IOUtils;

constructor TQwen21Tokenizer.Create(const FileName: string);
var
  Root: TJSONValue;
  Vocab: TJSONObject;
  Merges, Added: TJSONArray;
  Included: array[0..255] of Boolean;
  N, Id: Integer;
  A, B, Pattern: string;
begin
  inherited Create;
  FVocab := TDictionary<string, Integer>.Create;
  FMerges := TDictionary<string, Integer>.Create;
  FSpecial := TDictionary<string, Integer>.Create;
  FillChar(Included, SizeOf(Included), 0);
  for var I := 33 to 126 do
    Included[I] := True;
  for var I := 161 to 172 do
    Included[I] := True;
  for var I := 174 to 255 do
    Included[I] := True;
  N := 256;
  for var I := 0 to 255 do
    if Included[I] then
      FByteChars[I] := Char(I)
    else
    begin
      FByteChars[I] := Char(N);
      Inc(N)
    end;
  Root := TJSONObject.ParseJSONValue(TFile.ReadAllText(FileName, TEncoding.UTF8));
  if not (Root is TJSONObject) then
  begin
    Root.Free;
    raise EArgumentException.Create('Invalid tokenizer JSON')
  end;
  try
    Vocab := Root.FindValue('model.vocab') as TJSONObject;
    Merges := Root.FindValue('model.merges') as TJSONArray;
    Added := Root.FindValue('added_tokens') as TJSONArray;
    if (Vocab = nil) or (Merges = nil) or (Added = nil) then
      raise EArgumentException.Create('Incomplete tokenizer');
    for var Pair in Vocab do
      FVocab.Add(Pair.JsonString.Value, Pair.JsonValue.AsType<Integer>);
    for var I := 0 to Merges.Count - 1 do
    begin
      if Merges.Items[I] is TJSONArray then
      begin
        A := TJSONArray(Merges.Items[I]).Items[0].Value;
        B := TJSONArray(Merges.Items[I]).Items[1].Value;
      end
      else
      begin
        A := Merges.Items[I].Value;
        Id := Pos(' ', A);
        if Id = 0 then
          raise EArgumentException.Create('Invalid BPE merge');
        B := Copy(A, Id + 1);
        A := Copy(A, 1, Id - 1);
      end;
      FMerges.Add(A + #0 + B, I);
    end;
    for var Value in Added do
    begin
      A := Value.GetValue<string>('content');
      Id := Value.GetValue<Integer>('id');
      FSpecial.AddOrSetValue(A, Id);
      FVocab.AddOrSetValue(A, Id);
    end;
    // Match Qwen3-VL's Split/ByteLevel pre-tokenizer. Keep Unicode properties
    // and each digit separate; conversion to UTF-8 bytes happens after splitting.
    Pattern :=
      '(?i:''s|''t|''re|''ve|''m|''ll|''d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+';
    FPattern := TRegEx.Create(Pattern);
  finally
    Root.Free
  end;
end;

destructor TQwen21Tokenizer.Destroy;
begin
  FSpecial.Free;
  FMerges.Free;
  FVocab.Free;
  inherited;
end;

procedure TQwen21Tokenizer.EncodePiece(const Text: string; const Output: TList<Integer>);
var
  Bytes: TBytes;
  Pieces: TArray<string>;
  Rank, Best, BestAt, Count, Id: Integer;
begin
  Bytes := TEncoding.UTF8.GetBytes(Text);
  Count := Length(Bytes);
  SetLength(Pieces, Count);
  for var I := 0 to Count - 1 do
    Pieces[I] := FByteChars[Bytes[I]];
  while Count > 1 do
  begin
    Best := MaxInt;
    BestAt := -1;
    for var I := 0 to Count - 2 do
      if FMerges.TryGetValue(Pieces[I] + #0 + Pieces[I + 1], Rank) and (Rank < Best) then
      begin
        Best := Rank;
        BestAt := I
      end;
    if BestAt < 0 then
      Break;
    Pieces[BestAt] := Pieces[BestAt] + Pieces[BestAt + 1];
    for var I := BestAt + 1 to Count - 2 do
      Pieces[I] := Pieces[I + 1];
    Dec(Count);
  end;
  for var I := 0 to Count - 1 do
  begin
    if not FVocab.TryGetValue(Pieces[I], Id) then
      raise EArgumentException.Create('BPE token absent from vocabulary');
    Output.Add(Id);
  end;
end;

procedure TQwen21Tokenizer.EncodePlain(const Text: string; const Output: TList<Integer>);
var
  Covered: Integer;
begin
  Covered := 0;
  for var M in FPattern.Matches(Text) do
  begin
    if M.Index <> Covered + 1 then
      raise EArgumentException.Create('Tokenizer regex did not cover input');
    EncodePiece(M.Value, Output);
    Inc(Covered, M.Length);
  end;
  if Covered <> Length(Text) then
    raise EArgumentException.Create('Tokenizer left unconsumed text');
end;

function TQwen21Tokenizer.Encode(const Text: string): TArray<Integer>;
var
  Output: TList<Integer>;
  Posn, StartPos, BestAt, BestLen, BestId, P: Integer;
begin
  Output := TList<Integer>.Create;
  try
    Posn := 1;
    while Posn <= Length(Text) do
    begin
      BestAt := Length(Text) + 1;
      BestLen := 0;
      BestId := 0;
      for var Pair in FSpecial do
      begin
        P := Pos(Pair.Key, Text, Posn);
        if (P > 0) and ((P < BestAt) or ((P = BestAt) and (Length(Pair.Key) > BestLen))) then
        begin
          BestAt := P;
          BestLen := Length(Pair.Key);
          BestId := Pair.Value;
        end;
      end;
      StartPos := Posn;
      if BestAt > StartPos then
        EncodePlain(Copy(Text, StartPos, BestAt - StartPos), Output);
      if BestLen = 0 then
        Break;
      Output.Add(BestId);
      Posn := BestAt + BestLen;
    end;
    Result := Output.ToArray;
  finally
    Output.Free
  end;
end;

function TQwen21Tokenizer.Chat(const Prompt: string; out DropTokens: Integer): TArray<Integer>;
const
  SYSTEM_TURN = '<|im_start|>system'#10'Comprehend and analyze the provided prompt.<|im_end|>'#10;
begin
  DropTokens := Length(Encode(SYSTEM_TURN));
  Result := Encode(SYSTEM_TURN + '<|im_start|>user'#10 + Prompt + '<|im_end|>'#10'<|im_start|>assistant'#10);
end;

end.

