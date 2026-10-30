unit QuickNN.Models.Qwen3;

{$C+} // enable assertions
{$H+}
{$POINTERMATH on}

interface

uses
  System.SysUtils,
  QuickNN.Common,
  QuickNN.SafeTensors,
  System.JSON,
  System.Generics.Collections;

(* ========================================================================
 * Architecture Constants
 * ======================================================================== *)

const
  (* Fixed constants (same across model sizes) *)
  QWEN3_VOCAB_SIZE = 151936;
  QWEN3_MAX_SEQ_LEN = 512;
  QWEN3_RMS_NORM_EPS = 1e-6;
  QWEN3_ROPE_THETA = 1000000.0;

  (* Output layers to extract (0-indexed)
 * Python uses hidden_states[9,18,27] which are outputs AFTER layers 8,17,26
 * since hidden_states[0] is embedding and hidden_states[i] is output after layer i-1 *)
  QWEN3_OUTPUT_LAYER_1 = 8;
  QWEN3_OUTPUT_LAYER_2 = 17;
  QWEN3_OUTPUT_LAYER_3 = 26;

type
  TQNNBPEMerge = record
    Left: RawByteString;
    Right: RawByteString;
    Rank: LongInt; (* Lower rank = higher priority (merge first) *)
  end;

  { TQNNTokenizer }
  PQNNTokenizer = ^TQNNTokenizer;
  TQNNTokenizer = record
    (* Vocabulary: id -> token string *)
    Vocabs: TArray<RawByteString>;

    (* Hash table: token string -> id *)
    VocabsDic: TVocabs;

    (* BPE merges *)
    Merges: TArray<TQNNBPEMerge>;

    (* Merge rank lookup: "left right" -> rank *)
    MergeRanks: TVocabs;
    constructor Load(const JsonFilename: string);
    function GetToken(const Id: LongInt): RawByteString;
    function GetId(const Token: RawByteString): LongInt;
    function GetMergeRank(const Left, Right: RawByteString): LongInt;
    function BPEEncodeWord(const Str: RawByteString): TArray<RawByteString>;
    function Tokenize(const Str: RawByteString; MaxLen: LongInt = 0): TArray<LongInt>;
    function TokenizeChat(
        const Prompt: RawByteString;
        const SkipThinkTags: Boolean = False;
        MaxLen: LongInt = 0
    ): TArray<LongInt>;
    function DeTokenize(const Tokens: TArray<LongInt>): RawByteString;
    procedure Free;

    class operator Initialize(out Val: TQNNTokenizer);
  end;

  PQWEN3Model = ^TQwen3Model;
  PQWEN3Layer = ^TQwen3Layer;

  { TQWEN3Attention }

  TQWEN3Attention = record
    Model: PQWEN3Model;
    Layer: PQWEN3Layer;
    QProjWeight: TMemoryBlock; // [num_heads * head_dim, hidden] = [4096, 2560]
    KProjWeight: TMemoryBlock; // [num_kv_heads * head_dim, hidden] = [1024, 2560]
    VProjWeight: TMemoryBlock; // [num_kv_heads * head_dim, hidden] = [1024, 2560]
    OProjWeight: TMemoryBlock; // [hidden, num_heads * head_dim] = [2560, 4096]
    QNormWeight: TMemoryBlock; // [head_dim] = [128]
    KNormWeight: TMemoryBlock; // [head_dim] = [128]

    // incase of GPU
    QProjWeightBF16: TMemoryBlock;
    KProjWeightBF16: TMemoryBlock;
    VProjWeightBF16: TMemoryBlock;
    OProjWeightBF16: TMemoryBlock;
    QNormWeightBF16: TMemoryBlock;
    KNormWeightBF16: TMemoryBlock;
    procedure Forward(const SeqLen: LongInt; const AttentionMask: TArray<LongInt>);

  end;

  { TQWEN3MLP }

  TQwen3MLP = record
    Model: PQWEN3Model;
    Layer: PQWEN3Layer;
    GateProjWeight: TMemoryBlock; // [intermediate, hidden] = [9728, 2560]
    UpProjWeight: TMemoryBlock; // [intermediate, hidden] = [9728, 2560]
    DownProjWeight: TMemoryBlock; // [hidden, intermediate] = [2560, 9728]

    // GPU
    GateProjWeightBF16: TMemoryBlock;
    UpProjWeightBF16: TMemoryBlock;
    DownProjWeightBF16: TMemoryBlock;
    procedure Forward(const SeqLen: LongInt);
  end;

  { TQWEN3Layer }

  TQwen3Layer = record
    Model: PQWEN3Model;
    InputLayernormWeight: TMemoryBlock; // [hidden]
    PostAttentionLayernormWeight: TMemoryBlock; // [hidden]

    Attn: TQWEN3Attention;
    Mlp: TQwen3MLP;
    // BF16 layer norm weights (for GPU path) - unused currently, kept for future
    InputLayernormWeightBF16: TMemoryBlock;
    PostAttentionLayernormWeightBF16: TMemoryBlock;
    procedure Forward(const SeqLen: LongInt; const AttentionMask: TArray<LongInt>);
    procedure Load(const LayerIdx: LongInt; const AUseMMAP: Boolean; const AttentionOnly: Boolean = False);
    procedure Free();
  end;

  { TQWEN3Model }

  TQwen3Model = record
    (* Architecture (from config.json) *)
    HiddenSize, IntermediateSize, NumHeads, NumKvHeads, HeadDim, VocabSize, TextDim, ExtractionMode:
        LongInt; (* 0 = Flux (layers 8,17,26 concat), 1 = Z-Image (layer -2) *)
    RoPETheta: QNNFloat;

    (* Embedding layer *)
    EmbedTokens: TMemoryBlock; (* [vocab_size, hidden] *)

    (* Transformer layers *)
    Layers: TArray<TQwen3Layer>;

    (* Final layer norm *)
    NormWeight: TMemoryBlock; (* [hidden] *)

    (* RoPE precomputed *)
    RoPECos: TMemoryBlock; (* [max_seq_len, head_dim/2] *)
    RoPESin: TMemoryBlock; (* [max_seq_len, head_dim/2] *)

    (* Working memory *)
    HiddenState: TMemoryBlock; (* [seq_len, hidden] *)
    Residual: TMemoryBlock; (* [seq_len, hidden] *)
    QBuf: TMemoryBlock; (* [seq_len, num_heads * head_dim] *)
    KBuf: TMemoryBlock; (* [seq_len, num_kv_heads * head_dim] *)
    VBuf: TMemoryBlock; (* [seq_len, num_kv_heads * head_dim] *)
    AttnScores: TMemoryBlock; (* [num_heads, seq_len, seq_len] *)
    AttnOut: TMemoryBlock; (* [seq_len, num_heads * head_dim] *)
    MlpGate: TMemoryBlock; (* [seq_len, intermediate] *)
    MlpUp: TMemoryBlock; (* [seq_len, intermediate] *)
    MlpOut: TMemoryBlock; (* [seq_len, hidden] *)
    NormBuf: TMemoryBlock; (* [seq_len, hidden] *)

    (* Output layers storage (for extracting layers 9, 18, 27) *)
    LayerOutputs: array[0..2] of TMemoryBlock; (* [seq_len, hidden] each *)

    (* Pre-allocated attention work buffers (avoid per-call allocation) *)
    AttnQHead: TMemoryBlock; (* [seq_len, head_dim] *)
    AttnVHead: TMemoryBlock; (* [seq_len, head_dim] *)
    AttnOutHead: TMemoryBlock; (* [seq_len, head_dim] *)

    (* Mmap mode: keep safetensors files open, load layer weights on-demand *)
    UseMmap, UseBF16, ResidentOnly: Boolean;
    SfFiles: TSafeTensorFiles;
    procedure Load(const ModelDir: string; const AUseMMAP: Boolean; const Resident: Boolean = False);

    function Parse(const ModelDir: string): Boolean;
    procedure SetDefaults();
    procedure LoadSafeTensors(const ModelDir: string);
    procedure ReInitialize();
    procedure Free();
    procedure SetExtractionMode(const Mode: Boolean);
    function Forward(
        const InputIds: TArray<LongInt>;
        const AttentionMask: TArray<LongInt>;
        SeqLen: LongInt
    ): TMemoryBlock;
  (* BF16 GPU acceleration *)
  end;

  { TQWEN3Encoder }

  TQwen3Encoder = record
    Tokenizer: TQNNTokenizer;
    Model: TQwen3Model;
    procedure Load(const ModelDir: string; const AUseMMAP: Boolean; const Resident: Boolean = False);
    function EncodeText(const Prompt: RawByteString; var OutSeqLen: LongInt): TMemoryBlock;
    procedure Free;
  end;

function PadTokens(
    const Tokens: TArray<LongInt>;
    const MaxLen: LongInt;
    const AttentionMask: TArray<LongInt>
): TArray<LongInt>;

implementation
uses
  QuickNN.Kernels,
  QuickNN.Vulkan,
  System.IOUtils;

const
  // QWEN3_MAX_SEQ_LEN = 512;
  QWEN3_PAD_ID = 151643; (* <|endoftext|> *)
  QWEN3_IM_START_ID = 151644; (* <|im_start|> *)
  QWEN3_IM_END_ID = 151645; (* <|im_end|> *)
  QWEN3_THINK_START_ID = 151667; (* <think> *)
  QWEN3_THINK_END_ID = 151668; (* </think> *)

var
  ByteToUnicode: array[0..255] of LongInt;
  UnicodeToByte: array[0..511] of LongInt;

var
  ByteEncoderInitializd: Boolean = False;

procedure QNNHeadRSMNorm(const Dst, Src, Weight: TMemoryBlock; const SeqLen, NumHeads, HeadDim: LongInt);
var
  I, Off: LongInt;
begin
  for I := 0 to SeqLen - 1 do
  begin
    Off := I * NumHeads * HeadDim;
    QNNRMSNormRows(Dst + Off, Src + Off, Weight, NumHeads, HeadDim);
  end;
end;

procedure QWEN3ApplyRoPE(
    const Q, K, CosCache, SinCache: TMemoryBlock;
    const SeqLen, NumQHeads, NumKHeads, HeadDim: LongInt
);
begin
  QNNApplyRoPE3(Q, CosCache, SinCache, SeqLen, NumQHeads, HeadDim);
  QNNApplyRoPE3(K, CosCache, SinCache, SeqLen, NumKHeads, HeadDim);
end;

procedure InitByteEncoder();
var
  I, Offset: LongInt;
begin
  if ByteEncoderInitializd then
    Exit();
  FillChar(ByteToUnicode, SizeOf(ByteToUnicode), 0);
  FillChar(UnicodeToByte, SizeOf(UnicodeToByte), 0);
  for I := 33 to 126 do
  begin
    ByteToUnicode[I] := I;
    UnicodeToByte[I] := I
  end;
  for I := 161 to 172 do
  begin
    ByteToUnicode[I] := I;
    UnicodeToByte[I] := I
  end;
  for I := 174 to 255 do
  begin
    ByteToUnicode[I] := I;
    UnicodeToByte[I] := I
  end;
  Offset := 256;
  for I := 0 to 255 do
    if (ByteToUnicode[I] = 0) and (I <> 33) then
    begin
      ByteToUnicode[I] := Offset;
      UnicodeToByte[Offset] := I;
      Inc(Offset)
    end;
  ByteToUnicode[0] := 256;
  UnicodeToByte[256] := 0;
  ByteEncoderInitializd := True
end;

function EncodeByteToUTF8(const B: Byte; var Str: RawByteString): LongInt;
var
  Cp: LongInt;
begin
  InitByteEncoder();
  Cp := ByteToUnicode[B];
  if Cp < 128 then
  begin
    SetLength(Str, Length(Str) + 1);
    Str[Length(Str)] := AnsiChar(Cp);
    Exit(1)
  end
  else if Cp < 2048 then
  begin
    SetLength(Str, Length(Str) + 2);
    Str[Length(Str) - 1] := AnsiChar(($C0 or (Cp shr 6)));
    Str[Length(Str)] := AnsiChar(($80 or (Cp and $3F)));
    Exit(2)
  end;
  //
  SetLength(Str, Length(Str) + 1);
  Str[Length(Str)] := '?';
  Result := 1
end;

function PadTokens(
    const Tokens: TArray<LongInt>;
    const MaxLen: LongInt;
    const AttentionMask: TArray<LongInt>
): TArray<LongInt>;
begin
  SetLength(Result, MaxLen);

  Move(Tokens[0], Result[0], Length(Tokens) * SizeOf(Tokens[0]));
  FillDWord(Result[Length(Tokens)], Length(Result) - Length(Tokens), QWEN3_PAD_ID);

  if Assigned(AttentionMask) then
  begin
    FillDWord(AttentionMask[0], Length(Tokens), 1);
    FillDWord(AttentionMask[Length(Tokens)], Length(Result) - Length(Tokens), 0);
  end;

  // for i:=0 to high(result) do begin
  //  if i<length(tokens) then begin
  //    result[i] := tokens[i];
  //    if assigned(attentionMask) then
  //        attentionMask[i]:=1;
  //  end else begin
  //    result[i] := QWEN3_PAD_ID;
  //    if assigned(attentionMask) then
  //        attentionMask[i]:=0;
  //  end;
  // end;
end;

{ TQNNTokenizer }

constructor TQNNTokenizer.Load(const JsonFilename: string);
var
  JSON: TJSONValue;
  Vocab: TJSONObject;
  Merge, AddedTokens: TJSONArray;
  I, Id, Separator: LongInt;
  Parts: TJSONArray;
  MergeText: string;
  Content: RawByteString;
begin
  if not Assigned(VocabsDic) then
    VocabsDic := TVocabs.Create;
  if not Assigned(MergeRanks) then
    MergeRanks := TVocabs.Create;
  VocabsDic.Clear;
  MergeRanks.Clear;
  JSON := nil;
  try
    try
      JSON := TJSONValue.ParseJSONValue(TFile.ReadAllText(JsonFilename));
      if (JSON = nil)
          or not (JSON.FindValue('model.vocab') is TJSONObject)
          or not (JSON.FindValue('model.merges') is TJSONArray)
          or not (JSON.FindValue('added_tokens') is TJSONArray) then
        raise EArgumentException.Create('Invalid Qwen tokenizer structure: ' + JsonFilename);
      Vocab := JSON.FindValue('model.vocab') as TJSONObject;
      Merge := JSON.FindValue('model.merges') as TJSONArray;
      AddedTokens := JSON.FindValue('added_tokens') as TJSONArray;
      SetLength(Vocabs, Vocab.Count + AddedTokens.Count); // make space for added tokens
      SetLength(Merges, Merge.Count); // make space for added merges
      for I := 0 to Vocab.Count - 1 do
      begin
        Id := Vocab.Pairs[I].JsonValue.AsType<LongInt>;
        Vocabs[Id] := UTF8String(Vocab.Pairs[I].JsonString.Value);
        VocabsDic.TryAdd(Vocabs[Id], Id);
      end;

      for I := 0 to High(Merges) do
      begin
        if Merge[I] is TJSONArray then
        begin
          Parts := TJSONArray(Merge[I]);
          if (Parts.Count <> 2) or not (Parts[0] is TJSONString) or not (Parts[1] is TJSONString) then
            raise EArgumentException.CreateFmt('Invalid BPE merge pair at %d', [I]);
          Merges[I].Left := UTF8String(Parts[0].Value);
          Merges[I].Right := UTF8String(Parts[1].Value);
        end
        else if Merge[I] is TJSONString then
        begin
          MergeText := Merge[I].Value;
          Separator := Pos(' ', MergeText);
          if (Separator <= 1) or (Separator >= Length(MergeText)) then
            raise EArgumentException.CreateFmt('Invalid BPE merge string at %d', [I]);
          Merges[I].Left := UTF8String(Copy(MergeText, 1, Separator - 1));
          Merges[I].Right := UTF8String(Copy(MergeText, Separator + 1, MaxInt));
        end
        else
          raise EArgumentException.CreateFmt('Unsupported BPE merge representation at %d', [I]);
        Merges[I].Rank := I;
        if (Merges[I].Left <> '') and (Merges[I].Right <> '') then
        begin
          MergeRanks.TryAdd(Merges[I].Left + ' ' + Merges[I].Right, I);
        end;
      end;

      for I := 0 to AddedTokens.Count - 1 do
      begin
        Id := (AddedTokens[I] as TJSONObject).GetValue<LongInt>('id');
        Content := UTF8String((AddedTokens[I] as TJSONObject).GetValue<string>('content'));
        if (Content = '') or (Id < 0) or (Id >= Length(Vocabs)) then
          Continue; // skip IDs that has no allocated position
        Vocabs[Id] := Content;
        VocabsDic.TryAdd(Vocabs[Id], Id);
      end;
    except
      Self.Free;
      raise;
    end;
  finally
    JSON.Free;
  end;
end;

function TQNNTokenizer.GetToken(const Id: LongInt): RawByteString;
begin
  Result := '';
  if Assigned(Vocabs) and (Id >= 0) and (Id < Length(Vocabs)) then
    Result := Vocabs[Id]
end;

function TQNNTokenizer.GetId(const Token: RawByteString): LongInt;
begin
  if not Assigned(VocabsDic) or (Token = '') then
    Exit(-1);
  if not VocabsDic.TryGetValue(Token, Result) then
    Result := -1;
end;

function TQNNTokenizer.GetMergeRank(const Left, Right: RawByteString): LongInt;
begin
  Assert(Assigned(MergeRanks), 'ERROR : tokenizer is empty, load tokenizer file 1st!');
  if not MergeRanks.TryGetValue(Left + ' ' + Right, Result) then
    Result := -1;
end;

function TQNNTokenizer.BPEEncodeWord(const Str: RawByteString): TArray<RawByteString>;
  procedure AppendToken(const A: RawByteString);
  begin
    Insert(A, Result, Length(Result));
  end;

var
  I, Len, BestIdx, BestRank, Rank: LongInt;
  S: RawByteString;
  Changed: Boolean;

begin
  Assert(Assigned(VocabsDic), 'ERROR : no vocabs loaded, use <load> function to load vocab 1st.');
  Result := nil;
  I := 1;
  // n := 0;
  BestIdx := 0;

  while I <= Length(Str) do
  begin
    Len := UTF8CharSize(Str[I]);
    // s := copy(str, i, len);
    AppendToken(Copy(Str, I, Len));
    Inc(I, Len);
  end;

  Changed := True;
  while (Changed) do
  begin
    Changed := False;

    (* Find best merge (lowest rank) *)
    BestRank := Length(Merges) + 1;

    for I := 0 to High(Result) - 1 do
    begin
      Rank := GetMergeRank(Result[I], Result[I + 1]);
      if (Rank >= 0) and (Rank < BestRank) then
      begin
        BestRank := Rank;
        BestIdx := I;
        Changed := True
      end
    end;
    if not Changed then
      Break;
    (* Apply best merge *)
    (* Merge left and right at best_idx*)
    S := Result[BestIdx] + Result[BestIdx + 1];
    Result[BestIdx] := S;
    Delete(Result, BestIdx + 1, 1)
  end;

end;

function Pretokenize(const Text: RawByteString): TArray<RawByteString>;
var
  Count, Len: LongInt;
  P, Start: PAnsiChar;
  Lower: AnsiChar;
  Chunk: RawByteString;
begin
  Result := nil;
  Chunk := '';
  Count := 0;
  P := PAnsiChar(Text);
  while P^ <> NUL do
  begin
    Start := P;
    if (P[0] = '''') and (P[1] <> NUL) then
    begin
      Lower := LCase(P[1]);
      if Lower in ['s', 't', 'm', 'd'] then
        P := P + 2
      else if (Lower in ['r', 'v', 'l']) and (P[2] <> NUL) and (LCase(P[2]) in ['e', 'l']) then
        P := P + 3
      else
        Inc(P)
    end
    else if (P[0] in ALPHABETS) or (PByte(P)^ >= 128) then
      while (P[0] <> NUL) and ((P[0] in ALPHABETS) or (PByte(P)^ >= 128)) do
      begin
        if Byte(P[0]) >= 128 then
        begin
          if (Byte(P[0]) and $E0) = $C0 then
            Inc(P, 2)
          else if (Byte(P[0]) and $F0) = $E0 then
            Inc(P, 3)
          else if (Byte(P[0]) and $F8) = $F0 then
            Inc(P, 4)
          else
            Inc(P)
        end
        else
          Inc(P)
      end
    else if P[0] in NUMERICS then
      while P[0] in NUMERICS do
        Inc(P)
    else if (P[0] = ' ') and (P[1] <> NUL) and ((P[1] in ALPHABETS) or (Byte(P[1]) >= 128)) then
    begin
      Inc(P);
      while (P[0] <> NUL) and ((P[0] in ALPHABETS) or (Byte(P[0]) >= 128)) do
      begin
        if Byte(P[0]) >= 128 then
        begin
          if (PByte(P)^ and $E0) = $C0 then
            Inc(P, 2)
          else if (Byte(P[0]) and $F0) = $E0 then
            Inc(P, 3)
          else if (Byte(P[0]) and $F8) = $F0 then
            Inc(P, 4)
          else
            Inc(P)
        end
        else
          Inc(P)
      end
    end
    else if (P[0] = ' ') and (P[1] in NUMERICS) then
    begin
      Inc(P);
      while P[0] in NUMERICS do
        Inc(P)
    end
    else if P[0] in WHITE_SPACES then
      while P[0] in WHITE_SPACES do
        Inc(P)
    else
      Inc(P);
    if P > Start then
    begin
      Len := P - Start;
      SetLength(Chunk, Len);
      Move(Start[0], Chunk[1], Len);
      SetLength(Result, Length(Result) + 1);
      Result[Count] := Chunk;
      Inc(Count)
    end
  end
end;

function TextToUTF8(const Str: RawByteString): RawByteString;
var
  I: LongInt;
begin
  InitByteEncoder();
  Result := '';
  for I := 1 to Length(Str) do
  begin
    EncodeByteToUTF8(Byte(Str[I]), Result);
  end;
end;

function TQNNTokenizer.Tokenize(const Str: RawByteString; MaxLen: LongInt): TArray<LongInt>;
var
  Chunks: TArray<RawByteString>;
  Total, C: LongInt;
  ByteText: RawByteString;
  BpeTokens: TArray<RawByteString>;
  I: LongInt;
  Id: LongInt;
  Found: Boolean;
begin
  if (MaxLen <= 0) then
    MaxLen := QWEN3_MAX_SEQ_LEN;
  Chunks := Pretokenize(Str);
  Total := 0;
  C := 0;
  Result := nil;
  while (C < Length(Chunks)) and (Total < MaxLen) do
  begin
    ByteText := TextToUTF8(Chunks[C]);
    if ByteText <> '' then
    begin
      BpeTokens := BPEEncodeWord(ByteText);
      I := 0;
      while (I < Length(BpeTokens)) and (Total < MaxLen) do
      begin
        Found := VocabsDic.TryGetValue(BpeTokens[I], Id);
        if Found and (Id > 0) then
        begin
          SetLength(Result, Length(Result) + 1);
          Result[Total] := Id;
          Inc(Total)
        end;
        Inc(I)
      end;
    end;
    Inc(C)
  end
end;

(* Apply the Qwen3 chat template and tokenize the result. Template:
 * <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
 * For Flux, also appends <think>\n\n</think>\n\n to match the training
 * template that triggers direct generation. For Z-Image the think tags
 * are skipped (controlled by skipThinkTags). *)

function TQNNTokenizer.TokenizeChat(
    const Prompt: RawByteString;
    const SkipThinkTags: Boolean;
    MaxLen: LongInt
): TArray<LongInt>;
var
  Tokens: TArray<LongInt>;
  Total, I: LongInt;
begin
  if (MaxLen <= 0) then
    MaxLen := QWEN3_MAX_SEQ_LEN;
  Result := [QWEN3_IM_START_ID];
  Total := 1;

  Tokens := Tokenize('user'#10, MaxLen - Total);
  for I := 0 to High(Tokens) do
  begin
    if Total >= MaxLen then
      Break;
    Inc(Total);
    SetLength(Result, Total);
    Result[Total - 1] := Tokens[I];
  end;

  Tokens := Tokenize(Prompt, MaxLen - Total);
  for I := 0 to High(Tokens) do
  begin
    if Total >= MaxLen then
      Break;
    Inc(Total);
    SetLength(Result, Total);
    Result[Total - 1] := Tokens[I];
  end;
  if Total < MaxLen then
  begin
    Inc(Total);
    SetLength(Result, Total);
    Result[Total - 1] := QWEN3_IM_END_ID;
  end;

  Tokens := Tokenize(#10, MaxLen - Total);
  for I := 0 to High(Tokens) do
  begin
    if Total >= MaxLen then
      Break;
    Inc(Total);
    SetLength(Result, Total);
    Result[Total - 1] := Tokens[I];
  end;

  if Total < MaxLen then
  begin
    Inc(Total);
    SetLength(Result, Total);
    Result[Total - 1] := QWEN3_IM_START_ID;
  end;

  Tokens := Tokenize('assistant'#10, MaxLen - Total);
  for I := 0 to High(Tokens) do
  begin
    if Total >= MaxLen then
      Break;
    Inc(Total);
    SetLength(Result, Total);
    Result[Total - 1] := Tokens[I];
  end;

  if not SkipThinkTags then
  begin
    if Total < MaxLen then
    begin
      Inc(Total);
      SetLength(Result, Total);
      Result[Total - 1] := QWEN3_THINK_START_ID;
    end;

    Tokens := Tokenize(#10#10, MaxLen - Total);
    for I := 0 to High(Tokens) do
    begin
      if Total >= MaxLen then
        Break;
      Inc(Total);
      SetLength(Result, Total);
      Result[Total - 1] := Tokens[I];
    end;

    if Total < MaxLen then
    begin
      Inc(Total);
      SetLength(Result, Total);
      Result[Total - 1] := QWEN3_THINK_END_ID;
    end;

    Tokens := Tokenize(#10#10, MaxLen - Total);
    for I := 0 to High(Tokens) do
    begin
      if Total >= MaxLen then
        Break;
      Inc(Total);
      SetLength(Result, Total);
      Result[Total - 1] := Tokens[I];
    end;
  end;

end;

function TQNNTokenizer.DeTokenize(const Tokens: TArray<LongInt>): RawByteString;
var
  I, J, Id, TotalLen: LongInt;
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

  J := 0;

  for I := 0 to High(Tokens) do
  begin
    Id := Tokens[I];
    if (Id >= 0) and (Id < Length(Vocabs)) then
    begin
      Move(Vocabs[Id][1], Result[J + 1], Length(Vocabs[Id]));
      Inc(J, Length(Vocabs[Id]))
    end;
  end;
  //
  //  setLength(result, j)
end;

procedure TQNNTokenizer.Free;
begin
  if Assigned(VocabsDic) then
  begin
    VocabsDic.Free;
    VocabsDic := nil
  end;
  if Assigned(Merges) then
    SetLength(Merges, 0);
  if Assigned(Vocabs) then
    SetLength(Vocabs, 0);
  if Assigned(MergeRanks) then
  begin
    MergeRanks.Free;
    MergeRanks := nil
  end;
end;

class operator TQNNTokenizer.Initialize(out Val: TQNNTokenizer);
begin
  FillChar(Val, SizeOf(TQNNTokenizer), #0)
end;

{ TQWEN3Attention }

procedure TQWEN3Attention.Forward(const SeqLen: LongInt; const AttentionMask: TArray<LongInt>);
var
  Scale: QNNFloat;
  KvDim, QDim, HeadsPerKv, KvH, H: LongInt;
  QStrided, KStrided, VStrided, OutStrided: TMemoryBlock;
  Scores: TMemoryBlock;
begin
  Assert(Assigned(Model) and Assigned(Layer), 'ERROR : failed attention forward, no model loaded!');
  KvDim := Model.NumKvHeads * Model.HeadDim;
  QDim := Model.NumHeads * Model.HeadDim;
  Scale := 1.0 / Sqrt(Model.HeadDim);

  QNNLinearNoBias(Model.QBuf, Model.NormBuf, Layer.Attn.QProjWeight, SeqLen, Model.HiddenSize, QDim);
  QNNLinearNoBias(Model.KBuf, Model.NormBuf, Layer.Attn.KProjWeight, SeqLen, Model.HiddenSize, KvDim);
  QNNLinearNoBias(Model.VBuf, Model.NormBuf, Layer.Attn.VProjWeight, SeqLen, Model.HiddenSize, KvDim);

  (* Q/K RMS normalization (per-head) *)
  QNNHeadRSMNorm(Model.QBuf, Model.QBuf, Layer.Attn.QNormWeight, SeqLen, Model.NumHeads, Model.HeadDim);
  QNNHeadRSMNorm(Model.KBuf, Model.KBuf, Layer.Attn.KNormWeight, SeqLen, Model.NumKvHeads, Model.HeadDim);

  (* Apply RoPE *)
  QNNApplyRoPE3(Model.QBuf, Model.RoPECos, Model.RoPESin, SeqLen, Model.NumHeads, Model.HeadDim);
  QNNApplyRoPE3(Model.KBuf, Model.RoPECos, Model.RoPESin, SeqLen, Model.NumKvHeads, Model.HeadDim);

  HeadsPerKv := Model.NumHeads div Model.NumKvHeads;

  for H := 0 to Model.NumHeads - 1 do
  begin
    KvH := H div HeadsPerKv; (* Which KV head to use *)
    Scores := Model.AttnScores + H * SeqLen * SeqLen;

    (* Q accessed directly with strided lda (avoids copy)
     * Q[s,d] = q_buf[s * q_dim + h * head_dim + d] *)
    QStrided := Model.QBuf + H * Model.HeadDim;

    (* K accessed directly with strided lda + CblasTrans (avoids transpose)
     * K[s,d] = k_buf[s * kv_dim + kv_h * head_dim + d] *)
    KStrided := Model.KBuf + KvH * Model.HeadDim;

    CblasGemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasTrans,
        SeqLen,
        SeqLen,
        Model.HeadDim,
        Scale,
        QStrided,
        QDim,
        KStrided,
        KvDim,
        0.0,
        Scores,
        SeqLen
    );

    (* Apply causal mask and attention mask, then softmax *)
    // for i := 0 to seq_len-1 do begin          // todo [QWEN3Attention.Forward] optimize for GPU?
    //    for j := 0 to seq_len-1 do begin
    //        if (j > i) or assigned(attention_mask) and (attention_mask[j] = 0) then begin
    //            scores_ptr[i*seq_len + j] := Single.NegativeInfinity;//-1e9;
    //        end;
    //    end;
    //    QNNSoftmax(scores_ptr + i*seq_len, seq_len);
    // end;
    QNNMatTriangularFill(SeqLen, Scores, QNNFloat.NegativeInfinity, Pointer(AttentionMask));
    QNNSoftmaxRows(Scores, SeqLen, SeqLen);
    (* V can be accessed directly with strided lda (avoids copy)
     * V[s,d] = v_buf[s * kv_dim + kv_h * head_dim + d] *)
    VStrided := Model.VBuf + KvH * Model.HeadDim;

    (* Output can be written directly with strided ldc (avoids copy)
     * out[s,d] = attn_out[s * q_dim + h * head_dim + d] *)
    OutStrided := Model.AttnOut + H * Model.HeadDim;

    (* out = scores @ V using strided BLAS (avoids V copy and output copy)
     * scores: [seq_len, seq_len], V: [seq_len, head_dim] with ldb=kv_dim *)
    CblasGemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        SeqLen,
        Model.HeadDim,
        SeqLen,
        1.0,
        Scores,
        SeqLen,
        VStrided,
        KvDim,
        0.0,
        OutStrided,
        QDim
    );
  end;

  // model.attn_out.printCompare(readTensor);
  QNNLinearNoBias(Model.HiddenState, Model.AttnOut, Layer.Attn.OProjWeight, SeqLen, QDim, Model.HiddenSize);

end;

{ TQWEN3MLP }

procedure TQwen3MLP.Forward(const SeqLen: LongInt);
begin
  Assert(Assigned(Model) and Assigned(Layer), 'ERROR : failed MLP forward, no model loaded!');
  if SeqLen <= 0 then
    Exit;
  if TryVulkanSwiGLU(
      Model.MlpOut,
      Model.NormBuf,
      Layer.Mlp.GateProjWeight,
      Layer.Mlp.UpProjWeight,
      Layer.Mlp.DownProjWeight,
      SeqLen,
      Model.HiddenSize,
      Model.IntermediateSize) then
    Exit;
  if Model.MlpGate.Count < Int64(SeqLen) * Model.IntermediateSize then
    Model.MlpGate.Resize([SeqLen, Model.IntermediateSize], dtF32);
  if Model.MlpUp.Count < Int64(SeqLen) * Model.IntermediateSize then
    Model.MlpUp.Resize([SeqLen, Model.IntermediateSize], dtF32);
  (* Gate and Up projections *)
  QNNLinearNoBias(
      Model.MlpGate,
      Model.NormBuf,
      Layer.Mlp.GateProjWeight,
      SeqLen,
      Model.HiddenSize,
      Model.IntermediateSize
  );
  QNNLinearNoBias(Model.MlpUp, Model.NormBuf, Layer.Mlp.UpProjWeight, SeqLen, Model.HiddenSize, Model.IntermediateSize);

  (* SwiGLU: silu(gate) * up - fused for better performance *)
  QNNSiluMul(Model.MlpGate, Model.MlpUp, SeqLen * Model.IntermediateSize);

  (* Down projection *)
  QNNLinearNoBias(
      Model.MlpOut,
      Model.MlpGate,
      Layer.Mlp.DownProjWeight,
      SeqLen,
      Model.IntermediateSize,
      Model.HiddenSize
  );
end;

{ TQWEN3Layer }

procedure TQwen3Layer.Forward(const SeqLen: LongInt; const AttentionMask: TArray<LongInt>);
begin

  (* Save residual *)
  QNNCopy(Model.Residual, Model.HiddenState, SeqLen * Model.HiddenSize);

  (* Pre-attention LayerNorm *)
  QNNRMSNormRows(Model.NormBuf, Model.HiddenState, InputLayernormWeight, SeqLen, Model.HiddenSize);

  (* Self-attention *)
  Attn.Forward(SeqLen, AttentionMask);

  (* Residual connection *)
  QNNAddInplace(Model.HiddenState, Model.Residual, SeqLen * Model.HiddenSize);
  // for i := 0 to seq_len * model.hidden_size do
  //    model.hidden_state[i] += model.residual[i];

  (* Save residual *)
  QNNCopy(Model.Residual, Model.HiddenState, SeqLen * Model.HiddenSize);

  (* Pre-MLP LayerNorm *)
  QNNRMSNormRows(Model.NormBuf, Model.HiddenState, PostAttentionLayernormWeight, SeqLen, Model.HiddenSize);

  (* MLP *)
  Mlp.Forward(SeqLen);

  (* Residual connection *)
  QNNAdd(Model.HiddenState, Model.Residual, Model.MlpOut, SeqLen * Model.HiddenSize);

  // for i := 0 to seq_len * hidden_size-1 do
  //    model.hidden_state[i] = model.residual[i] + model.mlp_out[i];
end;

procedure TQwen3Layer.Load(const LayerIdx: LongInt; const AUseMMAP: Boolean; const AttentionOnly: Boolean);
  function Projection(const Name: string; const Mapped: Boolean): TMemoryBlock;
  begin
    if Model.ResidentOnly then
      Result := Model.SfFiles.GetLinearWeight(Name, Mapped)
    else
      Result := Model.SfFiles.GetTensorDataMemBlock(Name, Mapped);
  end;
begin
  Assert(Assigned(Model), 'ERROR : failed to load QWEN3 Layer, no model assigned yet');
  // Input layernorm
  InputLayernormWeight :=
      Model.SfFiles.GetTensorDataMemBlock(Format('model.layers.%d.input_layernorm.weight', [LayerIdx]), AUseMMAP);
  // Post-attention layernorm                                              layer->post_attention_layernorm_weight := sf_files.getTensorDataMemBlock(snprintf(name, sizeof(name), "model.layers.%d.post_attention_layernorm.weight", layer_idx);
  PostAttentionLayernormWeight :=
      Model
          .SfFiles
          .GetTensorDataMemBlock(Format('model.layers.%d.post_attention_layernorm.weight', [LayerIdx]), AUseMMAP);

  // Attention weights
  Attn.Model := Model;
  Attn.Layer := @Self;
  Attn.QProjWeight := Projection(Format('model.layers.%d.self_attn.q_proj.weight', [LayerIdx]), AUseMMAP);
  Attn.KProjWeight := Projection(Format('model.layers.%d.self_attn.k_proj.weight', [LayerIdx]), AUseMMAP);
  Attn.VProjWeight := Projection(Format('model.layers.%d.self_attn.v_proj.weight', [LayerIdx]), AUseMMAP);
  Attn.OProjWeight := Projection(Format('model.layers.%d.self_attn.o_proj.weight', [LayerIdx]), AUseMMAP);

  // Q/K norm
  Attn.QNormWeight :=
      Model.SfFiles.GetTensorDataMemBlock(Format('model.layers.%d.self_attn.q_norm.weight', [LayerIdx]), AUseMMAP);
  Attn.KNormWeight :=
      Model.SfFiles.GetTensorDataMemBlock(Format('model.layers.%d.self_attn.k_norm.weight', [LayerIdx]), AUseMMAP);

  if AttentionOnly then
    Exit;
  // MLP weights
  Mlp.Model := Model;
  Mlp.Layer := @Self;
  Mlp.GateProjWeight :=
      Model.SfFiles.GetLinearWeight(Format('model.layers.%d.mlp.gate_proj.weight', [LayerIdx]), AUseMMAP);
  Mlp.UpProjWeight := Model.SfFiles.GetLinearWeight(Format('model.layers.%d.mlp.up_proj.weight', [LayerIdx]), AUseMMAP);
  Mlp.DownProjWeight :=
      Model.SfFiles.GetLinearWeight(Format('model.layers.%d.mlp.down_proj.weight', [LayerIdx]), AUseMMAP);
end;

procedure TQwen3Layer.Free();
begin
  InputLayernormWeight.Free;
  (* Post-attention layernorm *)
  PostAttentionLayernormWeight.Free;

  (* Attention weights *)
  Attn.QProjWeight.Free;
  Attn.KProjWeight.Free;
  Attn.VProjWeight.Free;
  Attn.OProjWeight.Free;

  (* Q/K norm *)
  Attn.QNormWeight.Free;
  Attn.KNormWeight.Free;

  (* MLP weights *)
  Mlp.GateProjWeight.Free;
  Mlp.UpProjWeight.Free;
  Mlp.DownProjWeight.Free;
end;

{ TQWEN3Model }

procedure TQwen3Model.Load(const ModelDir: string; const AUseMMAP: Boolean; const Resident: Boolean);
var
  HalfDim, MaxSeq, I: LongInt;
begin
  UseMmap := AUseMMAP;
  ResidentOnly := Resident;
  Parse(ModelDir);

  LoadSafeTensors(ModelDir);

  if ResidentOnly then
    EmbedTokens := SfFiles.GetLinearWeight('model.embed_tokens.weight', AUseMMAP)
  else
    EmbedTokens := SfFiles.GetTensorDataMemBlock('model.embed_tokens.weight', AUseMMAP);
  // Resident extraction loads only the layers it actually executes, in either
  // mmap or owned-copy mode. Retain those loaded layers for subsequent calls.
  if not AUseMMAP and not ResidentOnly then
    for I := 0 to High(Layers) do
      Layers[I].Load(I, AUseMMAP);
  NormWeight := SfFiles.GetTensorDataMemBlock('model.norm.weight', AUseMMAP);

  if ResidentOnly then
    Exit;
  HalfDim := HeadDim div 2;
  MaxSeq := QWEN3_MAX_SEQ_LEN;

  RoPECos := TMemoryBlock.Create([MaxSeq, HalfDim], 'QWEN3_LOAD_ROPE_COS');
  RoPESin := TMemoryBlock.Create([MaxSeq, HalfDim], 'QWEN3_LOAD_ROPE_SIN');
  QNNComputeRoPE2(RoPECos, RoPESin, MaxSeq, HeadDim, RoPETheta);
  ReInitialize();

end;

function TQwen3Model.Parse(const ModelDir: string): Boolean;
var
  JSON: TJSONValue;
  NumLayers, I: LongInt;
  Fn: string;
begin
  Fn := ModelDir + '/config.json';
  Result := FileExists(Fn);
  if Result then
  begin
    JSON := TJSONValue.ParseJSONValue(TFile.ReadAllText(Fn));
    try

      HiddenSize := JSON.GetValue<LongInt>('hidden_size');
      NumHeads := JSON.GetValue<LongInt>('num_attention_heads');
      IntermediateSize := JSON.GetValue<LongInt>('intermediate_size', 9728);
      NumKvHeads := JSON.GetValue<LongInt>('num_key_value_heads', 8);
      HeadDim := JSON.GetValue<LongInt>('head_dim', 128);
      VocabSize := JSON.GetValue<LongInt>('vocab_size', QWEN3_VOCAB_SIZE);
      NumLayers := JSON.GetValue<LongInt>('num_hidden_layers', 36);
      RoPETheta := JSON.GetValue<QNNFloat>('rope_theta', QWEN3_ROPE_THETA);
      SetLength(Layers, NumLayers);
      for I := 0 to High(Layers) do
        Layers[I].Model := @Self;
      TextDim := HiddenSize * 3; // for flux *3(default) , for zi-image *1
    finally
      JSON.Free;
    end;
  end
  else
    SetDefaults();
end;

// Defaults for qwen3-4b case
procedure TQwen3Model.SetDefaults();
begin
  HiddenSize := 2560;
  NumHeads := 32;
  IntermediateSize := 9728;
  NumKvHeads := 8;
  HeadDim := 128;
  VocabSize := QWEN3_VOCAB_SIZE;
  // num_layers        := 36;
  RoPETheta := QWEN3_ROPE_THETA;
  SetLength(Layers, 36);
  for var I := 0 to High(Layers) do
    Layers[I].Model := @Self;
  TextDim := HiddenSize * 3;
end;

function IndexOf(const Arr: TArray<string>; const Str: string): LongInt;
var
  I: LongInt;
begin
  for I := 0 to High(Arr) do
    if Arr[I] = Str then
      Exit(I);
  Result := -1
end;

procedure TQwen3Model.LoadSafeTensors(const ModelDir: string);
begin
  if FileExists(TPath.Combine(ModelDir, 'model.safetensors.index.json'))
      or FileExists(TPath.Combine(ModelDir, 'model.safetensors')) then
    SfFiles := LoadSafeTensorShards(ModelDir, 'model.safetensors.index.json', 'model.safetensors')
  else
  begin
    // Preserve support for the original two-shard layout without an index.
    SetLength(SfFiles, 2);
    try
      SfFiles[0] := TSafeTensorFile.Open(TPath.Combine(ModelDir, 'model-00001-of-00002.safetensors'));
      SfFiles[1] := TSafeTensorFile.Open(TPath.Combine(ModelDir, 'model-00002-of-00002.safetensors'));
    except
      SfFiles[0].Close;
      SfFiles[1].Close;
      SfFiles := nil;
      raise;
    end;
  end;
end;

procedure TQwen3Model.ReInitialize();
var
  I, SeqLen: LongInt;
begin
  ExtractionMode := 0;
  SeqLen := QWEN3_MAX_SEQ_LEN;
  HiddenState := TMemoryBlock.Create([SeqLen, HiddenSize], 'QWEN3_INIT_hidden_state');
  Residual := TMemoryBlock.Create([SeqLen, HiddenSize], 'QWEN3_INIT_residual');
  QBuf := TMemoryBlock.Create([SeqLen, NumHeads, HeadDim], 'QWEN3_INIT_q_buf');
  KBuf := TMemoryBlock.Create([SeqLen, NumKvHeads, HeadDim], 'QWEN3_INIT_k_buf');
  VBuf := TMemoryBlock.Create([SeqLen, NumKvHeads, HeadDim], 'QWEN3_INIT_v_buf');
  AttnScores := TMemoryBlock.Create([NumHeads, SeqLen, SeqLen], 'QWEN3_INIT_attn_scores');
  AttnOut := TMemoryBlock.Create([SeqLen, NumHeads, HeadDim], 'QWEN3_INIT_attn_out');
  MlpOut := TMemoryBlock.Create([SeqLen, HiddenSize], 'QWEN3_INIT_mlp_out');
  NormBuf := TMemoryBlock.Create([SeqLen, HiddenSize], 'QWEN3_INIT_norm_buf');

  AttnQHead := TMemoryBlock.Create([SeqLen, HeadDim], 'QWEN3_INIT_attn_q_head');
  AttnVHead := TMemoryBlock.Create([SeqLen, HeadDim], 'QWEN3_INIT_attn_v_head');
  AttnOutHead := TMemoryBlock.Create([SeqLen, HeadDim], 'QWEN3_INIT_attn_out_head');

  for I := 0 to High(LayerOutputs) do
    LayerOutputs[I] := TMemoryBlock.Create([SeqLen, HiddenSize], 'QWEN3_LAYER_OUT_' + IntToStr(I));

end;

procedure TQwen3Model.Free();
var
  I: LongInt;
begin
  QNNGPUClearWeightCache;

  EmbedTokens.Free();
  NormWeight.Free();
  RoPECos.Free();
  RoPESin.Free();

  HiddenState.Free();
  Residual.Free();
  QBuf.Free();
  KBuf.Free();
  VBuf.Free();
  AttnScores.Free();
  AttnOut.Free();
  MlpGate.Free();
  MlpUp.Free();
  MlpOut.Free();
  NormBuf.Free();
  AttnQHead.Free();
  AttnVHead.Free();
  AttnOutHead.Free();

  for I := 0 to High(Layers) do
    Layers[I].Free();
  Layers := nil;
  for I := 0 to High(LayerOutputs) do
    LayerOutputs[I].Free();
  for I := 0 to High(SfFiles) do
    SfFiles[I].Close();
  SfFiles := nil;
  Self := Default(TQwen3Model)
end;

procedure TQwen3Model.SetExtractionMode(const Mode: Boolean);
begin
  ExtractionMode := Ord(Mode);
  if Mode then
    TextDim := HiddenSize
  else
    TextDim := HiddenSize * 3;
end;

(* ========================================================================
 * Forward Pass
 * ======================================================================== *)

(* Main Qwen3 forward pass. Runs embedding lookup then processes through
 * transformer layers. Flux mode: saves hidden states at layers 8, 17, 26
 * and concatenates them -> [seq, 3*hidden]. Z-Image mode: saves only
 * hidden_states[-2] (layer 34) -> [seq, hidden]. Stops early at the last
 * needed extraction layer to skip ~9 unnecessary layers of compute*)
function TQwen3Model.Forward(
    const InputIds: TArray<LongInt>;
    const AttentionMask: TArray<LongInt>;
    SeqLen: LongInt
): TMemoryBlock;
var
  ZImage: Boolean;
  LastLayer, S, TokenId, LayerIdx: LongInt;
begin

  ZImage := Boolean(ExtractionMode);
  if ZImage then
    LastLayer := Length(Layers) - 2
  else
    LastLayer := QWEN3_OUTPUT_LAYER_3;
  for S := 0 to SeqLen - 1 do
  begin
    TokenId := InputIds[S];
    if (TokenId >= 0) and (TokenId < VocabSize) then
      QNNCopy(HiddenState + S * HiddenSize, EmbedTokens + TokenId * HiddenSize, HiddenSize)
    else
      QNNFill(HiddenState + S * HiddenSize, 0, HiddenSize)
  end;

  for LayerIdx := 0 to LastLayer do
  begin
    if UseMmap then
      Layers[LayerIdx].Load(LayerIdx, UseMmap);
    Layers[LayerIdx].Forward(SeqLen, AttentionMask);
    if UseMmap then
      Layers[LayerIdx].Free;
    if ZImage then
    begin
      if LayerIdx = LastLayer then
        QNNCopy(LayerOutputs[0], HiddenState, SeqLen * HiddenSize)
    end
    else
    begin
      if LayerIdx = QWEN3_OUTPUT_LAYER_1 then
        QNNCopy(LayerOutputs[0], HiddenState, SeqLen * HiddenSize)
      else if LayerIdx = QWEN3_OUTPUT_LAYER_2 then
        QNNCopy(LayerOutputs[1], HiddenState, SeqLen * HiddenSize)
      else if LayerIdx = QWEN3_OUTPUT_LAYER_3 then
        QNNCopy(LayerOutputs[2], HiddenState, SeqLen * HiddenSize)
    end;
    if Assigned(TextProgressCallback) then
      TextProgressCallback(LayerIdx, LastLayer + 1)
  end;
  Result := TMemoryBlock.Create([SeqLen, TextDim], 'QWEN3_RESULT');

  if ZImage then
    QNNCopy(Result, LayerOutputs[0], SeqLen * HiddenSize)
  else
    for S := 0 to SeqLen - 1 do
    begin
      QNNCopy(Result + S * TextDim, LayerOutputs[0] + S * HiddenSize, HiddenSize);
      QNNCopy(Result + S * TextDim + HiddenSize, LayerOutputs[1] + S * HiddenSize, HiddenSize);
      QNNCopy(Result + S * TextDim + 2 * HiddenSize, LayerOutputs[2] + S * HiddenSize, HiddenSize)
    end;
end;

{ TQWEN3Encoder }

procedure TQwen3Encoder.Load(const ModelDir: string; const AUseMMAP: Boolean; const Resident: Boolean);
begin
  Tokenizer := TQNNTokenizer.Load(ModelDir + '/tokenizer/tokenizer.json');
  Model.Load(ModelDir + '/text_encoder', AUseMMAP, Resident);
end;

function TQwen3Encoder.EncodeText(const Prompt: RawByteString; var OutSeqLen: LongInt): TMemoryBlock;
var
  SkipThinkTags: Boolean;
  Tokens, AttentionMask, PaddedTokens: TArray<LongInt>;
begin
  AttentionMask := nil;;
  if not Assigned(Tokenizer.VocabsDic) or not Assigned(Model.Layers) then
    Exit(Default(TMemoryBlock));
  SkipThinkTags := Model.ExtractionMode = 1;
  Tokens := Tokenizer.TokenizeChat(Prompt, SkipThinkTags, QWEN3_MAX_SEQ_LEN);
  OutSeqLen := Length(Tokens);
  SetLength(AttentionMask, QWEN3_MAX_SEQ_LEN);
  PaddedTokens := PadTokens(Tokens, QWEN3_MAX_SEQ_LEN, AttentionMask);
  Result := Model.Forward(PaddedTokens, AttentionMask, QWEN3_MAX_SEQ_LEN);
end;

procedure TQwen3Encoder.Free;
begin
  Tokenizer.Free;
  Model.Free;
end;

// const
//  TXT_ANSI = 'the quick brown fox jumps over the lazy dog. '; // ansi
//  TXT_UTF8  ='أبجد هوز حطي كلمن';                            // utf-8
// var t: TQNNTokenizer;
// m : TQWEN3Model;
//    tokens : TArray<longint>;
// pretokens : TArray<string>;
//    s : string;
// f : uint64;
initialization

  // t := createSimpleTokenizer();
  // f := GetTickCount64;
  // t := TQNNTokenizer.load('c:\development\flux2.c\flux-klein-model\tokenizer\tokenizer.json');
  // m.loadSafeTensors('c:\development\flux2.c\flux-klein-model\text_encoder');
  // readln;

  // writeln('loading took ', (GetTickCount64-f)/1000:1:3, ' seconds');
  // pretokens := pretokenize(TXT_ANSI+TXT_UTF8);
  // tokens := t.tokenize(TXT_ANSI+TXT_UTF8);
  // s := t.deTokenize(tokens);
  // t.free;
  // writeln(s);
  // readln
  // readln

end.
