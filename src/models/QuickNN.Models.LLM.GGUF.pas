unit QuickNN.Models.LLM.GGUF;

interface

uses
  System.SysUtils, System.JSON, QuickNN.GGUF, QuickNN.Models.Qwen3, QuickNN.SafeTensors;

// Accept a GGUF file or a directory with one language checkpoint. Native
// safetensors directories retain their existing behavior and take precedence.
function LLMGGUFCheckpoint(const Location: string): string;
function LLMGGUFConfig(const FileName: string; const Vocabulary: TJSONObject = nil): TJSONObject;
function LLMGGUFTokenizer(const Metadata: TJSONObject): TQNNTokenizer;
function LLMGGUFWeightName(const NativeName: string): string;
function LLMGGUFShards(const FileName: string): TArray<string>;
function LLMOpenGGUF(const FileName: string): TSafeTensorFiles;
function LLMGGUFMetadata(const Metadata: TJSONValue; const Key: string): TJSONValue;

implementation

uses
  System.Classes, System.IOUtils, System.Math, System.Generics.Collections, System.RegularExpressions;

// GGUF metadata keys contain literal dots; generic GetValue<T> treats dots as JSON paths.
function LLMGGUFMetadata(const Metadata: TJSONValue; const Key: string): TJSONValue;
begin
  Result := nil;
  if Metadata is TJSONObject then Result := TJSONObject(Metadata).GetValue(Key);
  if Result = nil then raise EGGUFError.Create('Missing GGUF metadata: '+Key);
end;

function MetadataInteger(const Metadata: TJSONObject; const Key: string; const Default: Int64): Int64;
begin
  var Value := Metadata.GetValue(Key); Result := Default;
  if Value <> nil then Result := Value.AsType<Int64>;
end;

function FirstShard(const FileName: string): string;
begin
  Result := TPath.GetFullPath(TRegEx.Replace(FileName, '-[0-9]{5}-of-([0-9]{5})\.gguf$', '-00001-of-$1.gguf', [roIgnoreCase]));
end;

function LLMGGUFCheckpoint(const Location: string): string;
begin
  Result := '';
  if TFile.Exists(Location) and SameText(TPath.GetExtension(Location), '.gguf') then
    Exit(FirstShard(Location));
  if not TDirectory.Exists(Location) then Exit;
  if TFile.Exists(TPath.Combine(Location, 'model.safetensors'))
    or TFile.Exists(TPath.Combine(Location, 'model.safetensors.index.json')) then Exit;
  for var FileName in TDirectory.GetFiles(Location, '*.gguf') do
  begin
    var Name := TPath.GetFileName(FileName).ToLower;
    if Name.StartsWith('mmproj') or Name.StartsWith('mtp-') or Name.StartsWith('imatrix') then Continue;
    var Candidate := FirstShard(FileName);
    if (Result <> '') and not SameFileName(Result, Candidate) then
      raise EArgumentException.Create('Several GGUF language checkpoints found; select a file or use a directory with one model.');
    Result := Candidate;
  end;
end;

function LLMGGUFShards(const FileName: string): TArray<string>;
begin
  var First := FirstShard(FileName);
  var Meta := TJSONObject.Create;
  var Seen := TDictionary<string, Boolean>.Create;
  try
    var Header := TGGUFHeader.ReadFile(First, Meta, False);
    var Count := Integer(MetadataInteger(Meta,'split.count',1));
    if (Count < 1) or (Count > 256) or (MetadataInteger(Meta,'split.no',0) <> 0) then
      raise EGGUFError.Create('Invalid first GGUF shard metadata');
    SetLength(Result, Count); Result[0] := First;
    var Pattern := TRegEx.Match(First, '^(.*)-00001-of-([0-9]{5})\.gguf$', [roIgnoreCase]);
    if (Count > 1) and (not Pattern.Success or (StrToInt(Pattern.Groups[2].Value) <> Count)) then
      raise EGGUFError.Create('GGUF split filename does not match split.count');
    var Expected := MetadataInteger(Meta,'split.tensors.count',Length(Header.Tensors));
    var Total: Int64 := 0;
    for var I := 0 to Count-1 do
    begin
      if I > 0 then
      begin
        Result[I] := Pattern.Groups[1].Value + Format('-%5.5d-of-%5.5d.gguf', [I+1, Count]);
        if not TFile.Exists(Result[I]) then raise EGGUFError.Create('Missing GGUF shard: ' + Result[I]);
        Meta.Free; Meta := TJSONObject.Create;
        Header := TGGUFHeader.ReadFile(Result[I], Meta, False);
        if (MetadataInteger(Meta,'split.no',-1) <> I) or (MetadataInteger(Meta,'split.count',0) <> Count)
          or (MetadataInteger(Meta,'split.tensors.count',-1) <> Expected) then
          raise EGGUFError.Create('Inconsistent GGUF shard metadata: ' + Result[I]);
      end;
      for var Tensor in Header.Tensors do
      begin
        if Seen.ContainsKey(Tensor.Name) then raise EGGUFError.Create('Duplicate tensor across GGUF shards: ' + Tensor.Name);
        Seen.Add(Tensor.Name, True);
      end;
      Inc(Total, Length(Header.Tensors));
    end;
    if Total <> Expected then raise EGGUFError.Create('GGUF split tensor count mismatch');
  finally Seen.Free; Meta.Free end;
end;

function LLMOpenGGUF(const FileName: string): TSafeTensorFiles;
begin
  var Names := LLMGGUFShards(FileName);
  SetLength(Result, Length(Names));
  try
    for var I := 0 to High(Names) do Result[I] := TSafeTensorFile.Open(Names[I]);
  except
    for var I := 0 to High(Result) do Result[I].Close;
    raise;
  end;
end;

function FlashGGUFConfig(const Header: TGGUFHeader; const Metadata: TJSONObject; const FileName: string): TJSONObject;
const PREFIX = 'qwen4exp.';
begin
  if (LLMGGUFMetadata(Metadata,'tokenizer.ggml.model').Value <> 'gpt2')
    or (LLMGGUFMetadata(Metadata,'tokenizer.ggml.pre').Value <> 'qwen35') then
    raise ENotSupportedException.Create('Unsupported qwen4exp tokenizer');
  LLMGGUFShards(FileName); // Validate the complete set before offering it in the catalog.
  Result := TJSONObject.Create;
  try
    Result.AddPair('model_type', 'qwen4exp'); Result.AddPair('quicknn_gguf_file', FirstShard(FileName));
    Result.AddPair('quicknn_gguf_name', LLMGGUFMetadata(Metadata,'general.name').Value);
    var Text := TJSONObject.Create; Result.AddPair('text_config', Text);
    var Meta := TJSONObject.Create; Result.AddPair('architecture_metadata', Meta);
    for var Pair in Metadata do
      if Pair.JsonString.Value.StartsWith(PREFIX) then Meta.AddPair(Pair.JsonString.Value, Pair.JsonValue.Clone as TJSONValue);
    var Keys: TArray<string> := ['hidden_size','num_hidden_layers','num_attention_heads','num_key_value_heads','head_dim',
      'max_position_embeddings','linear_num_key_heads','linear_num_value_heads','linear_key_head_dim','linear_conv_kernel_dim'];
    var Values: TArray<string> := ['embedding_length','block_count','attention.head_count','attention.head_count_kv','attention.key_length',
      'context_length','ssm.group_count','ssm.time_step_rank','ssm.state_size','ssm.conv_kernel'];
    for var I := 0 to High(Keys) do
    begin
      var Value := Metadata.GetValue(PREFIX+Values[I]);
      if not (Value is TJSONNumber) then raise EGGUFError.Create('Missing qwen4exp metadata: '+Values[I]);
      Text.AddPair(Keys[I], Value.Clone as TJSONValue);
    end;
    var Layers := Text.GetValue<Integer>('num_hidden_layers');
    var Interval := LLMGGUFMetadata(Metadata,PREFIX+'full_attention_interval').AsType<Integer>;
    if (Layers < 1) or (Layers > 256) or (Interval < 1) then raise EGGUFError.Create('Invalid qwen4exp layer geometry');
    var Kinds := TJSONArray.Create; Text.AddPair('layer_types', Kinds);
    for var I := 0 to Layers-1 do
      if (I+1) mod Interval = 0 then Kinds.Add('full_attention') else Kinds.Add('linear_attention');
    Text.AddPair('vocab_size', TJSONNumber.Create(Header.VocabularySize));
    Text.AddPair('intermediate_size', TJSONNumber.Create(LLMGGUFMetadata(Metadata,PREFIX+'expert_feed_forward_length').AsType<Integer>));
    Text.AddPair('linear_value_head_dim', TJSONNumber.Create(LLMGGUFMetadata(Metadata,PREFIX+'ssm.inner_size').AsType<Integer> div Max(1,Text.GetValue<Integer>('linear_num_value_heads'))));
    Text.AddPair('rms_norm_eps', TJSONNumber.Create(LLMGGUFMetadata(Metadata,PREFIX+'attention.layer_norm_rms_epsilon').AsType<Double>));
    Text.AddPair('eos_token_id', Metadata.GetValue('tokenizer.ggml.eos_token_id').Clone as TJSONValue);
    Result.AddPair('num_experts', TJSONNumber.Create(LLMGGUFMetadata(Metadata,PREFIX+'expert_count').AsType<Integer>));
    Result.AddPair('num_experts_per_tok', TJSONNumber.Create(LLMGGUFMetadata(Metadata,PREFIX+'expert_used_count').AsType<Integer>));
    Result.AddPair('moe_intermediate_size', TJSONNumber.Create(LLMGGUFMetadata(Metadata,PREFIX+'expert_feed_forward_length').AsType<Integer>));
    Result.AddPair('norm_topk_prob', TJSONBool.Create(True));
  except Result.Free; raise end;
end;

function GemmaGGUFConfig(const Header: TGGUFHeader; const Metadata: TJSONObject; const FileName: string): TJSONObject;
  procedure CopyInteger(const Target, Source: string);
  begin
    var Value := Metadata.GetValue(Source);
    if Value = nil then raise EGGUFError.Create('Missing Gemma GGUF metadata: ' + Source);
    Result.AddPair(Target, TJSONNumber.Create(Value.AsType<Integer>));
  end;
  procedure Number(const Target, Source: string; const Default: Double);
  begin
    var Value := Metadata.GetValue(Source); var Actual := Default;
    if Value <> nil then Actual := Value.AsType<Double>;
    Result.AddPair(Target, TJSONNumber.Create(Actual));
  end;
begin
  var Tokenizer := Metadata.GetValue('tokenizer.ggml.model');
  if (Tokenizer = nil) or (Tokenizer.Value <> 'llama') then
    raise ENotSupportedException.Create('Gemma 2 requires GGUF SentencePiece BPE');
  var SplitCount := Metadata.GetValue('split.count');
  if (SplitCount <> nil) and (SplitCount.AsType<Integer> > 1) then
    raise ENotSupportedException.Create('Use a single-file Gemma 2 GGUF checkpoint');
  var RopeScaling := Metadata.GetValue('gemma2.rope.scaling.type');
  if (RopeScaling <> nil) and (RopeScaling.Value <> 'none') then
    raise ENotSupportedException.Create('Gemma 2 RoPE scaling is not implemented');
  Result := TJSONObject.Create;
  try
    Result.AddPair('model_type', 'gemma2'); Result.AddPair('quicknn_gguf_file', FileName);
    Result.AddPair('vocab_size', TJSONNumber.Create(Header.VocabularySize));
    CopyInteger('hidden_size', 'gemma2.embedding_length');
    CopyInteger('intermediate_size', 'gemma2.feed_forward_length');
    CopyInteger('num_hidden_layers', 'gemma2.block_count');
    CopyInteger('num_attention_heads', 'gemma2.attention.head_count');
    CopyInteger('num_key_value_heads', 'gemma2.attention.head_count_kv');
    CopyInteger('head_dim', 'gemma2.attention.key_length');
    CopyInteger('max_position_embeddings', 'gemma2.context_length');
    CopyInteger('eos_token_id', 'tokenizer.ggml.eos_token_id');
    var KeyDim := Result.GetValue<Integer>('head_dim');
    var ValueDim := Metadata.GetValue('gemma2.attention.value_length');
    if (ValueDim <> nil) and (ValueDim.AsType<Integer> <> KeyDim) then
      raise ENotSupportedException.Create('Gemma 2 key/value widths must match');
    Number('rms_norm_eps', 'gemma2.attention.layer_norm_rms_epsilon', 1E-6);
    Number('rope_theta', 'gemma2.rope.freq_base', 10000);
    if Metadata.GetValue('gemma2.attention.sliding_window') <> nil then
      CopyInteger('sliding_window', 'gemma2.attention.sliding_window')
    else Result.AddPair('sliding_window', TJSONNumber.Create(4096));
    Number('attn_logit_softcapping', 'gemma2.attn_logit_softcapping', 50);
    Number('final_logit_softcapping', 'gemma2.final_logit_softcapping', 30);
    var QueryScale: Single := KeyDim;
    if Result.GetValue<Integer>('num_hidden_layers') = 46 then
      QueryScale := Result.GetValue<Integer>('hidden_size') div Result.GetValue<Integer>('num_attention_heads');
    Result.AddPair('query_pre_attn_scalar', TJSONNumber.Create(QueryScale));
  except Result.Free; raise end;
end;

function LLMGGUFConfig(const FileName: string; const Vocabulary: TJSONObject): TJSONObject;
var
  Metadata, Text, Rope: TJSONObject;
  Header: TGGUFHeader;

  function Item(const Name: string): TJSONValue;
  begin
    Result := Metadata.GetValue(Name);
    if Result = nil then raise EGGUFError.Create('Missing GGUF metadata: ' + Name);
  end;
  function IntegerValue(const Name: string): Integer;
  begin
    if not (Item(Name) is TJSONNumber) or not TryStrToInt(Item(Name).Value, Result) then
      raise EGGUFError.Create('Invalid GGUF integer: ' + Name);
  end;
  procedure CopyInteger(const Target, Source: string);
  begin Text.AddPair(Target, TJSONNumber.Create(IntegerValue(Source))) end;
  procedure CopyNumber(const Target, Source: string; const Destination: TJSONObject);
  begin
    if not (Item(Source) is TJSONNumber) then raise EGGUFError.Create('Invalid GGUF number: ' + Source);
    Destination.AddPair(Target, Item(Source).Clone as TJSONValue);
  end;
begin
  Metadata := Vocabulary;
  if Metadata = nil then Metadata := TJSONObject.Create;
  Result := nil;
  try
    try
      Header := TGGUFHeader.ReadFile(FileName, Metadata, Vocabulary <> nil);
      if Header.Architecture = 'qwen4exp' then
      begin Result := FlashGGUFConfig(Header, Metadata, FileName); Exit end;
      if Header.Architecture = 'gemma2' then
      begin Result := GemmaGGUFConfig(Header, Metadata, FileName); Exit end;
      if Header.Architecture <> 'qwen35' then
        raise ENotSupportedException.Create('The GGUF LLM loader supports dense qwen35 / Qwen3.8 and Gemma 2.');
      if (Metadata.GetValue('split.count') <> nil) and (IntegerValue('split.count') > 1) then
        raise ENotSupportedException.Create('Split GGUF LLM checkpoints are not supported; use a single GGUF file.');
      if (Item('tokenizer.ggml.model').Value <> 'gpt2') or (Item('tokenizer.ggml.pre').Value <> 'qwen35') then
        raise ENotSupportedException.Create('Unsupported Qwen GGUF pre-tokenizer');
      var Blocks := IntegerValue('qwen35.block_count');
      var MTP := 0;
      if Metadata.GetValue('qwen35.nextn_predict_layers') <> nil then MTP := IntegerValue('qwen35.nextn_predict_layers');
      var Layers := Blocks - MTP;
      if (MTP < 0) or (Layers < 1) or (Layers > 256) or (Header.VocabularySize < 1) then
        raise EGGUFError.Create('Invalid GGUF layer/vocabulary count');
      Result := TJSONObject.Create;
      Result.AddPair('model_type', 'qwen3_5');
      Result.AddPair('quicknn_gguf_file', FileName);
      Result.AddPair('quicknn_gguf_name', Item('general.name').Value);
      Text := TJSONObject.Create;
      Result.AddPair('text_config', Text);
      CopyInteger('hidden_size', 'qwen35.embedding_length');
      CopyInteger('intermediate_size', 'qwen35.feed_forward_length');
      CopyInteger('num_attention_heads', 'qwen35.attention.head_count');
      CopyInteger('num_key_value_heads', 'qwen35.attention.head_count_kv');
      CopyInteger('head_dim', 'qwen35.attention.key_length');
      if IntegerValue('qwen35.attention.key_length') <> IntegerValue('qwen35.attention.value_length') then
        raise ENotSupportedException.Create('Different Qwen GGUF key/value dimensions are not supported');
      Text.AddPair('vocab_size', TJSONNumber.Create(Header.VocabularySize));
      Text.AddPair('num_hidden_layers', TJSONNumber.Create(Layers));
      CopyInteger('max_position_embeddings', 'qwen35.context_length');
      CopyInteger('eos_token_id', 'tokenizer.ggml.eos_token_id');
      CopyInteger('linear_num_key_heads', 'qwen35.ssm.group_count');
      CopyInteger('linear_num_value_heads', 'qwen35.ssm.time_step_rank');
      CopyInteger('linear_key_head_dim', 'qwen35.ssm.state_size');
      var ValueHeads := IntegerValue('qwen35.ssm.time_step_rank');
      var Inner := IntegerValue('qwen35.ssm.inner_size');
      if (ValueHeads < 1) or (Inner < 1) or (Inner mod ValueHeads <> 0) then
        raise EGGUFError.Create('Invalid GGUF SSM inner dimensions');
      Text.AddPair('linear_value_head_dim', TJSONNumber.Create(Inner div ValueHeads));
      CopyInteger('linear_conv_kernel_dim', 'qwen35.ssm.conv_kernel');
      CopyNumber('rms_norm_eps', 'qwen35.attention.layer_norm_rms_epsilon', Text);
      var Interval := IntegerValue('qwen35.full_attention_interval');
      if (Interval < 1) or (Interval > Layers) then raise EGGUFError.Create('Invalid GGUF attention interval');
      var Kinds := TJSONArray.Create;
      Text.AddPair('layer_types', Kinds);
      var Recurrent := Metadata.GetValue('qwen35.attention.recurrent_layers');
      if (Recurrent <> nil) and (not (Recurrent is TJSONArray) or (TJSONArray(Recurrent).Count < Layers)) then
        raise EGGUFError.Create('Invalid GGUF recurrent layer flags');
      for var I := 0 to Layers - 1 do
      begin
        var Linear := (I + 1) mod Interval <> 0;
        if Recurrent <> nil then Linear := TJSONArray(Recurrent)[I].AsType<Boolean>;
        if Linear then Kinds.Add('linear_attention') else Kinds.Add('full_attention');
      end;
      Rope := TJSONObject.Create;
      Text.AddPair('rope_parameters', Rope);
      Rope.AddPair('rope_type', 'default');
      CopyNumber('rope_theta', 'qwen35.rope.freq_base', Rope);
      var HeadDim := IntegerValue('qwen35.attention.key_length');
      if HeadDim < 1 then raise EGGUFError.Create('Invalid GGUF head dimension');
      Rope.AddPair('partial_rotary_factor', TJSONNumber.Create(IntegerValue('qwen35.rope.dimension_count') / HeadDim));
    except
      Result.Free;
      raise;
    end;
  finally
    if Vocabulary = nil then Metadata.Free;
  end;
end;

function LLMGGUFTokenizer(const Metadata: TJSONObject): TQNNTokenizer;
begin
  var Tokens := Metadata.GetValue('tokenizer.ggml.tokens');
  var Merges := Metadata.GetValue('tokenizer.ggml.merges');
  var Types := Metadata.GetValue('tokenizer.ggml.token_type');
  var Model := Metadata.GetValue('tokenizer.ggml.model');
  if (Model <> nil) and (Model.Value = 'llama') then
  begin
    var Scores := Metadata.GetValue('tokenizer.ggml.scores');
    if not (Tokens is TJSONArray) or not (Scores is TJSONArray) or not (Types is TJSONArray)
      or (TJSONArray(Tokens).Count <> TJSONArray(Scores).Count) or (TJSONArray(Tokens).Count <> TJSONArray(Types).Count) then
      raise EGGUFError.Create('Missing GGUF SentencePiece vocabulary/scores/types');
    var Pieces: TArray<string>; var Values: TArray<Single>; var Kinds: TArray<Integer>;
    SetLength(Pieces, TJSONArray(Tokens).Count); SetLength(Values, Length(Pieces)); SetLength(Kinds, Length(Pieces));
    for var I := 0 to High(Pieces) do
    begin Pieces[I] := TJSONArray(Tokens)[I].Value; Values[I] := TJSONArray(Scores)[I].AsType<Single>;
      Kinds[I] := TJSONArray(Types)[I].AsType<Integer> end;
    var AddPrefix := Metadata.GetValue('tokenizer.ggml.add_space_prefix');
    Exit(TQNNTokenizer.LoadSentencePiece(Pieces, Values, Kinds, (AddPrefix <> nil) and AddPrefix.AsType<Boolean>));
  end;
  if not (Tokens is TJSONArray) or not (Merges is TJSONArray) or not (Types is TJSONArray) then
    raise EGGUFError.Create('Missing GGUF BPE vocabulary, merges or token types');
  var TokenValues: TArray<string>;
  var MergeValues: TArray<string>;
  var TypeValues: TArray<Integer>;
  SetLength(TokenValues, TJSONArray(Tokens).Count);
  SetLength(MergeValues, TJSONArray(Merges).Count);
  SetLength(TypeValues, TJSONArray(Types).Count);
  for var I := 0 to High(TokenValues) do TokenValues[I] := TJSONArray(Tokens)[I].Value;
  for var I := 0 to High(MergeValues) do MergeValues[I] := TJSONArray(Merges)[I].Value;
  for var I := 0 to High(TypeValues) do TypeValues[I] := TJSONArray(Types)[I].AsType<Integer>;
  Result := TQNNTokenizer.LoadBPE(TokenValues, MergeValues, TypeValues);
end;

function LLMGGUFWeightName(const NativeName: string): string;
const
  ROOT = 'model.language_model.';
begin
  if NativeName = ROOT + 'embed_tokens.weight' then Exit('token_embd.weight');
  if NativeName = ROOT + 'norm.weight' then Exit('output_norm.weight');
  if NativeName = 'lm_head.weight' then Exit('output.weight');
  var Prefix := ROOT + 'layers.';
  if not NativeName.StartsWith(Prefix) then raise EGGUFError.Create('Unknown GGUF weight alias: ' + NativeName);
  var Rest := NativeName.Substring(Length(Prefix));
  var Separator := Pos('.', Rest);
  if Separator < 2 then raise EGGUFError.Create('Invalid GGUF layer alias');
  Prefix := 'blk.' + Copy(Rest, 1, Separator);
  Rest := Rest.Substring(Separator);
  if Rest = 'input_layernorm.weight' then Result := 'attn_norm.weight'
  else if Rest = 'post_attention_layernorm.weight' then Result := 'post_attention_norm.weight'
  else if Rest = 'mlp.gate_proj.weight' then Result := 'ffn_gate.weight'
  else if Rest = 'mlp.up_proj.weight' then Result := 'ffn_up.weight'
  else if Rest = 'mlp.down_proj.weight' then Result := 'ffn_down.weight'
  else if Rest = 'linear_attn.in_proj_qkv.weight' then Result := 'attn_qkv.weight'
  else if Rest = 'linear_attn.in_proj_z.weight' then Result := 'attn_gate.weight'
  else if Rest = 'linear_attn.in_proj_a.weight' then Result := 'ssm_alpha.weight'
  else if Rest = 'linear_attn.in_proj_b.weight' then Result := 'ssm_beta.weight'
  else if Rest = 'linear_attn.out_proj.weight' then Result := 'ssm_out.weight'
  else if Rest = 'linear_attn.A_log' then Result := 'ssm_a'
  else if Rest = 'linear_attn.dt_bias' then Result := 'ssm_dt.bias'
  else if Rest = 'linear_attn.conv1d.weight' then Result := 'ssm_conv1d.weight'
  else if Rest = 'linear_attn.norm.weight' then Result := 'ssm_norm.weight'
  else if Rest = 'self_attn.q_proj.weight' then Result := 'attn_q.weight'
  else if Rest = 'self_attn.k_proj.weight' then Result := 'attn_k.weight'
  else if Rest = 'self_attn.v_proj.weight' then Result := 'attn_v.weight'
  else if Rest = 'self_attn.o_proj.weight' then Result := 'attn_output.weight'
  else if Rest = 'self_attn.q_norm.weight' then Result := 'attn_q_norm.weight'
  else if Rest = 'self_attn.k_norm.weight' then Result := 'attn_k_norm.weight'
  else raise EGGUFError.Create('Unknown GGUF weight alias: ' + NativeName);
  Result := Prefix + Result;
end;

end.
