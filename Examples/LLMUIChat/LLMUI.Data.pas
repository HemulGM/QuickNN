unit LLMUI.Data;

interface

uses
  System.SysUtils, System.Generics.Collections, System.JSON,
  QuickNN.Models.LLM.Types, LLMUI.Context.Types;

type
  TMessageState = (msComplete, msStreaming, msStopped, msFailed, msLimited, msContextFull);

  TChatPreferences = record
    ModelId, SystemPrompt: string;
    Context: Integer;
    Generation: TLLMGenerationOptions;
    Compaction: TChatCompactionOptions;
    GPUCacheGiB, RAMCacheGiB: Integer;
    Int4Experts: Boolean;
    StreamOutput: Boolean;
    class function Defaults: TChatPreferences; static;
    function ToJSON: TJSONObject;
    class function FromJSON(const Value: TJSONValue): TChatPreferences; static;
  end;

  TModelEntry = class
  public
    Id, Name, Directory, TokenizerDirectory: string;
    ModelType, Validation, ResolvedTokenizer: string;
    Available, SupportsThinking: Boolean;
    Layers, KVHeads, HeadDim, MaxContext: Integer;
    constructor Create;
    procedure Inspect;
    function ToJSON: TJSONObject;
    function KVBytes(const Context: Integer): UInt64;
    class function FromJSON(const Value: TJSONValue): TModelEntry; static;
  end;

  TChatMessage = class
  public
    Id, Text, ErrorText, ModelName, ModelId: string;
    Continuation: TLLMContinuationState;
    Role: TLLMRole;
    State: TMessageState;
    Created: TDateTime;
    Tokens: Integer;
    DecodeRate, FirstTokenSeconds: Double;
    constructor Create;
    function ToJSON: TJSONObject;
    class function FromJSON(const Value: TJSONValue): TChatMessage; static;
  end;

  TChatSession = class
  public
    Id, Title, Draft: string;
    ScrollPosition: Double;
    FollowLatest: Boolean;
    Created, Updated: TDateTime;
    Preferences: TChatPreferences;
    ContextMemory: TChatContextMemory;
    Messages: TObjectList<TChatMessage>;
    constructor Create;
    destructor Destroy; override;
    function ToJSON: TJSONObject;
    function ModelMessages: TLLMMessages;
    function ContextHistory(const ExcludeMessageId: string = ''): TChatContextHistory;
    function Preview: string;
    class function FromJSON(const Value: TJSONValue): TChatSession; static;
  end;

  TChatStore = class
  private
    FPath: string;
    FRecoveredBackup: Boolean;
    procedure ReadDocument(const Text: string);
    procedure ImportLegacyStore;
  public
    Models: TObjectList<TModelEntry>;
    Chats: TObjectList<TChatSession>;
    ActiveChatId: string;
    RecoveryNotice: string;
    constructor Create(const FileName: string);
    destructor Destroy; override;
    procedure Load;
    procedure Save;
    procedure SeedLocalModels;
    function FindModel(const Id: string): TModelEntry;
    function FindChat(const Id: string): TChatSession;
    function NewChat: TChatSession;
    procedure RemoveChat(const Id: string);
    procedure RemoveModel(const Id: string);
    property Path: string read FPath;
  end;

function NewId: string;

function ShortText(const Text: string; const Limit: Integer = 80): string;

function DefaultStorePath: string;

implementation

uses
  System.Classes, System.IOUtils, System.Math, Winapi.Windows;

function NewId: string;
begin
  Result := TGUID.NewGuid.ToString;
end;

function ShortText(const Text: string; const Limit: Integer): string;
begin
  Result := Trim(Text.Replace(#13, ' ').Replace(#10, ' ').Replace('`', ''));
  if Length(Result) > Limit then
    Result := Copy(Result, 1, Limit - 1) + '…';
end;

function DefaultStorePath: string;
begin
  Result := TPath.Combine(GetEnvironmentVariable('LOCALAPPDATA'), 'QuickNN\LLMUIChat\state.json');
end;

class function TChatPreferences.Defaults: TChatPreferences;
begin
  Result := Default(TChatPreferences);
  Result.Context := 2048;
  Result.Generation := TLLMGenerationOptions.Defaults;
  Result.Compaction := TChatCompactionOptions.Defaults;
  Result.Generation.Temperature := 0.6;
  Result.Generation.MaxNewTokens := 512;
  Result.Generation.HardMaxNewTokens := 4096;
  Result.Generation.AutoContinue := True;
  Result.GPUCacheGiB := 12;
  Result.RAMCacheGiB := 4;
  Result.Int4Experts := True;
  Result.StreamOutput := True;
  Result.SystemPrompt :=
    'Ты — полезный и внимательный ассистент. Отвечай ясно, по существу и на языке собеседника.';
end;

function TChatPreferences.ToJSON: TJSONObject;
begin
  Result := TJSONObject.Create;
  Result.AddPair('model_id', ModelId);
  Result.AddPair('compaction', Compaction.ToJSON);
  Result.AddPair('system_prompt', SystemPrompt);
  Result.AddPair('context', TJSONNumber.Create(Context));
  Result.AddPair('temperature', TJSONNumber.Create(Generation.Temperature));
  Result.AddPair('top_p', TJSONNumber.Create(Generation.TopP));
  Result.AddPair('top_k', TJSONNumber.Create(Generation.TopK));
  Result.AddPair('max_tokens', TJSONNumber.Create(Generation.MaxNewTokens));
  Result.AddPair('hard_max_tokens', TJSONNumber.Create(Generation.HardMaxNewTokens));
  Result.AddPair('auto_continue', TJSONBool.Create(Generation.AutoContinue));
  Result.AddPair('repetition_penalty', TJSONNumber.Create(Generation.RepetitionPenalty));
  Result.AddPair('seed', TJSONNumber.Create(Int64(Generation.Seed)));
  Result.AddPair('thinking', TJSONBool.Create(Generation.EnableThinking));
  Result.AddPair('gpu_cache_gib', TJSONNumber.Create(GPUCacheGiB));
  Result.AddPair('ram_cache_gib', TJSONNumber.Create(RAMCacheGiB));
  Result.AddPair('int4_experts', TJSONBool.Create(Int4Experts));
  Result.AddPair('stream', TJSONBool.Create(StreamOutput));
end;

class function TChatPreferences.FromJSON(const Value: TJSONValue): TChatPreferences;
begin
  Result := Defaults;
  if not (Value is TJSONObject) then
    Exit;
  Result.ModelId := Value.GetValue<string>('model_id', '');
  Result.Compaction := TChatCompactionOptions.FromJSON(Value.FindValue('compaction'));
  Result.SystemPrompt := Value.GetValue<string>('system_prompt', Result.SystemPrompt);
  Result.Context := EnsureRange(Value.GetValue<Integer>('context', 2048), 512, 8192);
  Result.Generation.Temperature := EnsureRange(Value.GetValue<Double>('temperature', 0.6), 0.0, 2.0);
  Result.Generation.TopP := EnsureRange(Value.GetValue<Double>('top_p', 0.95), 0.01, 1.0);
  Result.Generation.TopK := EnsureRange(Value.GetValue<Integer>('top_k', 20), 1, 64);
  Result.Generation.MaxNewTokens := EnsureRange(Value.GetValue<Integer>('max_tokens', 512), 1, 8192);
  Result.Generation.HardMaxNewTokens :=
    EnsureRange(Value.GetValue<Integer>('hard_max_tokens', 4096), Result.Generation.MaxNewTokens, 8192);
  Result.Generation.AutoContinue := Value.GetValue<Boolean>('auto_continue', True);
  Result.Generation.RepetitionPenalty := EnsureRange(Value.GetValue<Double>('repetition_penalty', 1), 0.1, 2.0);
  Result.Generation.Seed := UInt64(Max(Int64(0), Value.GetValue<Int64>('seed', 42)));
  Result.Generation.EnableThinking := Value.GetValue<Boolean>('thinking', False);
  Result.GPUCacheGiB := EnsureRange(Value.GetValue<Integer>('gpu_cache_gib', 12), 1, 12);
  Result.RAMCacheGiB := EnsureRange(Value.GetValue<Integer>('ram_cache_gib', 4), 0, 16);
  Result.Int4Experts := Value.GetValue<Boolean>('int4_experts', True);
  Result.StreamOutput := Value.GetValue<Boolean>('stream', True);
end;

constructor TModelEntry.Create;
begin
  inherited;
  Id := NewId;
end;

procedure TModelEntry.Inspect;
var
  Config, Tokenizer: TJSONValue;
  Candidate: string;
begin
  Available := False;
  Validation := '';
  SupportsThinking := False;
  ModelType := '';
  ResolvedTokenizer := '';
  Config := nil;
  Tokenizer := nil;
  try
    if not TDirectory.Exists(Directory) then
      raise EArgumentException.Create('Каталог модели не найден.');
    Candidate := TPath.Combine(Directory, 'config.json');
    if not TFile.Exists(Candidate) then
      raise EArgumentException.Create('В каталоге нет config.json.');
    Config := TJSONValue.ParseJSONValue(TFile.ReadAllText(Candidate, TEncoding.UTF8));
    if not (Config is TJSONObject) then
      raise EArgumentException.Create('Некорректный config.json.');
    ModelType := Config.GetValue<string>('model_type', '');
    if (ModelType <> 'qwen3') and (ModelType <> 'qwen3_moe') then
      raise EArgumentException.Create('Этот движок поддерживает Qwen3 и Qwen3 MoE в safetensors.');
    if (Config.FindValue('quantization_config') <> nil)
      and not (Config.FindValue('quantization_config') is TJSONNull) then
      raise EArgumentException.Create('Нужен исходный BF16/F32 checkpoint. INT4 экспертов создаётся самим QuickNN.');
    if not TFile.Exists(TPath.Combine(Directory, 'model.safetensors'))
      and not TFile.Exists(TPath.Combine(Directory, 'model.safetensors.index.json')) then
      raise EArgumentException.Create('Не найдены model.safetensors или индекс весов.');
    ResolvedTokenizer := TokenizerDirectory;
    if ResolvedTokenizer = '' then
    begin
      ResolvedTokenizer := Directory;
      if not TFile.Exists(TPath.Combine(ResolvedTokenizer, 'tokenizer.json')) then
        ResolvedTokenizer := TPath.Combine(ExtractFileDir(Directory), 'tokenizer');
    end;
    if not TFile.Exists(TPath.Combine(ResolvedTokenizer, 'tokenizer.json')) then
      raise EArgumentException.Create('Не найден tokenizer.json. Укажите каталог токенизатора.');
    Layers := Config.GetValue<Integer>('num_hidden_layers', 0);
    KVHeads := Config.GetValue<Integer>('num_key_value_heads', 0);
    HeadDim := Config.GetValue<Integer>('head_dim', 0);
    if HeadDim = 0 then
      HeadDim :=
        Config.GetValue<Integer>('hidden_size', 0) div Max(1, Config.GetValue<Integer>('num_attention_heads', 1));
    if (Layers < 1) or (KVHeads < 1) or (HeadDim < 1) then
      raise EArgumentException.Create('Неверная геометрия attention.');
    MaxContext :=
      Min(
      Config.GetValue<Integer>('max_position_embeddings', 32768),
      Integer((UInt64(1536) * 1024 * 1024) div (UInt64(Layers) * 2 * UInt64(KVHeads) * UInt64(HeadDim) * 4))
    );
    SupportsThinking := ModelType = 'qwen3';
    Candidate := TPath.Combine(ResolvedTokenizer, 'tokenizer_config.json');
    if TFile.Exists(Candidate) then
    begin
      Tokenizer := TJSONValue.ParseJSONValue(TFile.ReadAllText(Candidate, TEncoding.UTF8));
      if Tokenizer <> nil then
        SupportsThinking := Tokenizer.GetValue<string>('chat_template', '').Contains('<think>');
    end;
    Available := True;
    if ModelType = 'qwen3_moe' then
      Validation := 'Qwen3 MoE'
    else
      Validation := 'Qwen3';
    Validation := Validation + Format(' · %d слоёв · до %d токенов', [Layers, MaxContext]);
  except
    on E: Exception do
      Validation := E.Message;
  end;
  Config.Free;
  Tokenizer.Free;
end;

function TModelEntry.KVBytes(const Context: Integer): UInt64;
begin
  Result :=
    UInt64(Max(0, Layers)) * 2 * UInt64(Max(0, Context)) * UInt64(Max(0, KVHeads)) * UInt64(Max(0, HeadDim)) * 4;
end;

function TModelEntry.ToJSON: TJSONObject;
begin
  Result := TJSONObject.Create;
  Result.AddPair('id', Id);
  Result.AddPair('name', Name);
  Result.AddPair('directory', Directory);
  Result.AddPair('tokenizer_directory', TokenizerDirectory);
end;

class function TModelEntry.FromJSON(const Value: TJSONValue): TModelEntry;
begin
  Result := TModelEntry.Create;
  try
    Result.Id := Value.GetValue<string>('id', Result.Id);
    Result.Name := Value.GetValue<string>('name', 'Модель');
    Result.Directory := Value.GetValue<string>('directory', '');
    Result.TokenizerDirectory := Value.GetValue<string>('tokenizer_directory', '');
    Result.Inspect;
  except
    Result.Free;
    raise;
  end;
end;

function ContinuationToJSON(const State: TLLMContinuationState): TJSONObject;
var
  Tokens: TJSONArray;
  Pending: string;
begin
  Result := TJSONObject.Create;
  Result.AddPair('version', TJSONNumber.Create(1));
  Result.AddPair('model_key', State.ModelKey);
  Result.AddPair('prompt_signature', State.PromptSignature);
  Result.AddPair('prompt_tokens', TJSONNumber.Create(State.PromptTokens));
  Result.AddPair('generated_tokens', TJSONNumber.Create(State.GeneratedTokens));
  Result.AddPair('random_state', TJSONNumber.Create(Int64(State.RandomState)));
  Result.AddPair('decode_seconds', TJSONNumber.Create(State.DecodeSeconds));
  Result.AddPair('total_seconds', TJSONNumber.Create(State.TotalSeconds));
  Result.AddPair('first_token_seconds', TJSONNumber.Create(State.TimeToFirstTokenSeconds));
  Pending := '';
  for var B in State.PendingUTF8 do
    Pending := Pending + IntToHex(Byte(B), 2);
  Result.AddPair('pending_utf8', Pending);
  Tokens := TJSONArray.Create;
  Result.AddPair('tokens', Tokens);
  for var Id in State.Tokens do
    Tokens.Add(Id);
end;

function ContinuationFromJSON(const Value: TJSONValue; const Text: string): TLLMContinuationState;
var
  Tokens: TJSONValue;
  Pending: string;
  Code: Integer;
  Random: Int64;
begin
  Result := Default(TLLMContinuationState);
  if not (Value is TJSONObject) then
    Exit;
  try
    if Value.GetValue<Integer>('version', 0) <> 1 then
      Exit;
    Result.ModelKey := Value.GetValue<string>('model_key', '');
    Result.PromptSignature := Value.GetValue<string>('prompt_signature', '');
    Result.Text := Text;
    Result.PromptTokens := Value.GetValue<Integer>('prompt_tokens', 0);
    Result.GeneratedTokens := Value.GetValue<Integer>('generated_tokens', 0);
    Random := Value.GetValue<Int64>('random_state', 0);
    if (Random < 0) or (Random > High(UInt32)) then
      Exit(Default(TLLMContinuationState));
    Result.RandomState := UInt32(Random);
    Result.DecodeSeconds := Max(0.0, Value.GetValue<Double>('decode_seconds', 0));
    Result.TotalSeconds := Max(0.0, Value.GetValue<Double>('total_seconds', 0));
    Result.TimeToFirstTokenSeconds := Max(0.0, Value.GetValue<Double>('first_token_seconds', 0));
    Pending := Value.GetValue<string>('pending_utf8', '');
    if (Length(Pending) > 6) or Odd(Length(Pending)) then
      Exit(Default(TLLMContinuationState));
    for var I := 1 to Length(Pending) div 2 do
    begin
      if not TryStrToInt('$' + Copy(Pending, I * 2 - 1, 2), Code) then
        Exit(Default(TLLMContinuationState));
      Result.PendingUTF8 := Result.PendingUTF8 + AnsiChar(Code);
    end;
    Tokens := Value.FindValue('tokens');
    if not (Tokens is TJSONArray) or (TJSONArray(Tokens).Count > 8192) then
      Exit(Default(TLLMContinuationState));
    SetLength(Result.Tokens, TJSONArray(Tokens).Count);
    for var I := 0 to High(Result.Tokens) do
      Result.Tokens[I] := TJSONArray(Tokens).Items[I].AsType<Integer>;
    if not Result.Available then
      Result := Default(TLLMContinuationState);
  except
    // A damaged optional checkpoint must not discard otherwise readable chat history.
    Result := Default(TLLMContinuationState);
  end;
end;

constructor TChatMessage.Create;
begin
  inherited;
  Id := NewId;
  Created := Now;
end;

function TChatMessage.ToJSON: TJSONObject;
begin
  Result := TJSONObject.Create;
  Result.AddPair('id', Id);
  Result.AddPair('text', Text);
  Result.AddPair('error', ErrorText);
  Result.AddPair('model_name', ModelName);
  Result.AddPair('model_id', ModelId);
  if Continuation.Available and (Continuation.Text = Text) then
    Result.AddPair('continuation', ContinuationToJSON(Continuation));
  Result.AddPair('role', TJSONNumber.Create(Ord(Role)));
  Result.AddPair('state', TJSONNumber.Create(Ord(State)));
  Result.AddPair('created', TJSONNumber.Create(Created));
  Result.AddPair('tokens', TJSONNumber.Create(Tokens));
  Result.AddPair('decode_rate', TJSONNumber.Create(DecodeRate));
  Result.AddPair('first_token_seconds', TJSONNumber.Create(FirstTokenSeconds));
end;

class function TChatMessage.FromJSON(const Value: TJSONValue): TChatMessage;
begin
  Result := TChatMessage.Create;
  try
    Result.Id := Value.GetValue<string>('id', Result.Id);
    Result.Text := Value.GetValue<string>('text', '');
    Result.ErrorText := Value.GetValue<string>('error', '');
    Result.ModelName := Value.GetValue<string>('model_name', '');
    Result.ModelId := Value.GetValue<string>('model_id', '');
    Result.Continuation := ContinuationFromJSON(Value.FindValue('continuation'), Result.Text);
    Result.Role := TLLMRole(EnsureRange(Value.GetValue<Integer>('role', 1), 1, 2));
    Result.State := TMessageState(EnsureRange(Value.GetValue<Integer>('state', 0), 0, Ord(High(TMessageState))));
    if Result.State = msStreaming then
      Result.State := msStopped;
    Result.Created := Value.GetValue<Double>('created', Now);
    Result.Tokens := Value.GetValue<Integer>('tokens', 0);
    Result.DecodeRate := Value.GetValue<Double>('decode_rate', 0);
    Result.FirstTokenSeconds := Value.GetValue<Double>('first_token_seconds', 0);
  except
    Result.Free;
    raise;
  end;
end;

constructor TChatSession.Create;
begin
  inherited;
  Id := NewId;
  Title := 'Новый чат';
  FollowLatest := True;
  Created := Now;
  Updated := Created;
  Preferences := TChatPreferences.Defaults;
  Messages := TObjectList<TChatMessage>.Create(True);
end;

destructor TChatSession.Destroy;
begin
  Messages.Free;
  inherited;
end;

function TChatSession.Preview: string;
begin
  Result := 'Пока нет сообщений';
  if Messages.Count > 0 then
    Result := ShortText(Messages.Last.Text, 64);
  if Result = '' then
    Result := 'Ответ прерван';
end;

function TChatSession.ModelMessages: TLLMMessages;
var
  List: TList<TLLMMessage>;
begin
  List := TList<TLLMMessage>.Create;
  try
    if Trim(Preferences.SystemPrompt) <> '' then
      List.Add(TLLMMessage.Create(lrSystem, Preferences.SystemPrompt));
    for var Message in Messages do
      if (Message.Text <> '')
        and ((Message.Role = lrUser) or (Message.State in [msComplete, msStopped, msLimited, msContextFull])) then
        List.Add(TLLMMessage.Create(Message.Role, Message.Text));
    Result := List.ToArray;
  finally
    List.Free;
  end;
end;

function TChatSession.ContextHistory(const ExcludeMessageId: string): TChatContextHistory;
var
  Items: TList<TChatContextMessage>;
  Item: TChatContextMessage;
begin
  Items := TList<TChatContextMessage>.Create;
  try
    for var Message in Messages do
      if (Message.Id <> ExcludeMessageId)
        and (Message.Text <> '')
        and ((Message.Role = lrUser) or (Message.State in [msComplete, msStopped, msLimited, msContextFull])) then
      begin
        Item.Id := Message.Id;
        Item.Message := TLLMMessage.Create(Message.Role, Message.Text);
        Items.Add(Item);
      end;
    Result := Items.ToArray;
  finally
    Items.Free
  end;
end;

function TChatSession.ToJSON: TJSONObject;
var
  Items: TJSONArray;
begin
  Result := TJSONObject.Create;
  Result.AddPair('id', Id);
  Result.AddPair('title', Title);
  Result.AddPair('draft', Draft);
  Result.AddPair('scroll_position', TJSONNumber.Create(ScrollPosition));
  Result.AddPair('follow_latest', TJSONBool.Create(FollowLatest));
  Result.AddPair('created', TJSONNumber.Create(Created));
  Result.AddPair('updated', TJSONNumber.Create(Updated));
  Result.AddPair('preferences', Preferences.ToJSON);
  if ContextMemory.Available then
    Result.AddPair('context_memory', ContextMemory.ToJSON);
  Items := TJSONArray.Create;
  Result.AddPair('messages', Items);
  for var Message in Messages do
    Items.AddElement(Message.ToJSON);
end;

class function TChatSession.FromJSON(const Value: TJSONValue): TChatSession;
var
  Items: TJSONValue;
begin
  Result := TChatSession.Create;
  try
    Result.Id := Value.GetValue<string>('id', Result.Id);
    Result.Title := Value.GetValue<string>('title', 'Новый чат');
    Result.Draft := Value.GetValue<string>('draft', '');
    Result.ScrollPosition := Value.GetValue<Double>('scroll_position', 0);
    if IsNan(Result.ScrollPosition) or IsInfinite(Result.ScrollPosition) then
      Result.ScrollPosition := 0
    else
      Result.ScrollPosition := Max(0, Result.ScrollPosition);
    Result.FollowLatest := Value.GetValue<Boolean>('follow_latest', True);
    Result.Created := Value.GetValue<Double>('created', Now);
    Result.Updated := Value.GetValue<Double>('updated', Result.Created);
    Result.Preferences := TChatPreferences.FromJSON(Value.FindValue('preferences'));
    Result.ContextMemory := TChatContextMemory.FromJSON(Value.FindValue('context_memory'));
    Items := Value.FindValue('messages');
    if Items is TJSONArray then
      for var Item in TJSONArray(Items) do
        Result.Messages.Add(TChatMessage.FromJSON(Item));
  except
    Result.Free;
    raise;
  end;
end;

constructor TChatStore.Create(const FileName: string);
begin
  inherited Create;
  FPath := FileName;
  Models := TObjectList<TModelEntry>.Create(True);
  Chats := TObjectList<TChatSession>.Create(True);
end;

destructor TChatStore.Destroy;
begin
  Chats.Free;
  Models.Free;
  inherited;
end;

procedure TChatStore.ReadDocument(const Text: string);
var
  Root, Items: TJSONValue;
begin
  Root := TJSONValue.ParseJSONValue(Text);
  try
    if not (Root is TJSONObject) or (Root.GetValue<Integer>('version', 0) <> 1) then
      raise EArgumentException.Create('Неизвестный или повреждённый формат истории чатов.');
    Models.Clear;
    Chats.Clear;
    Items := Root.FindValue('models');
    if Items is TJSONArray then
      for var Item in TJSONArray(Items) do
        Models.Add(TModelEntry.FromJSON(Item));
    Items := Root.FindValue('chats');
    if Items is TJSONArray then
      for var Item in TJSONArray(Items) do
        Chats.Add(TChatSession.FromJSON(Item));
    ActiveChatId := Root.GetValue<string>('active_chat_id', '');
  finally
    Root.Free;
  end;
end;

procedure TChatStore.ImportLegacyStore;
var
  LegacyPath: string;
begin
  // Import only the default store, once. Custom stores and existing QuickNN
  // history always take precedence; the old files remain available as backups.
  if not SameText(ExpandFileName(FPath), ExpandFileName(DefaultStorePath))
    or TFile.Exists(FPath) or TFile.Exists(FPath + '.bak') then
    Exit;
  LegacyPath := TPath.Combine(GetEnvironmentVariable('LOCALAPPDATA'), 'Envision\LLMUIChat\state.json');
  if not TFile.Exists(LegacyPath) and not TFile.Exists(LegacyPath + '.bak') then
    Exit;
  TDirectory.CreateDirectory(ExtractFileDir(FPath));
  if TFile.Exists(LegacyPath + '.bak') then
    TFile.Copy(LegacyPath + '.bak', FPath + '.bak', False);
  if TFile.Exists(LegacyPath) then
    TFile.Copy(LegacyPath, FPath, False);
end;

procedure TChatStore.Load;
begin
  ImportLegacyStore;
  if not TFile.Exists(FPath) then
  begin
    if TFile.Exists(FPath + '.bak') then
    begin
      ReadDocument(TFile.ReadAllText(FPath + '.bak', TEncoding.UTF8));
      FRecoveredBackup := True;
      RecoveryNotice := 'История восстановлена из резервной копии.';
    end
    else
      SeedLocalModels;
    Exit;
  end;
  try
    ReadDocument(TFile.ReadAllText(FPath, TEncoding.UTF8));
  except
    if not TFile.Exists(FPath + '.bak') then
      raise;
    ReadDocument(TFile.ReadAllText(FPath + '.bak', TEncoding.UTF8));
    FRecoveredBackup := True;
    RecoveryNotice := 'История восстановлена из резервной копии.';
  end;
end;

procedure TChatStore.Save;
var
  Root: TJSONObject;
  Items: TJSONArray;
  Stream: TFileStream;
  Bytes: TBytes;
  Temporary: string;
begin
  TDirectory.CreateDirectory(ExtractFileDir(FPath));
  Root := TJSONObject.Create;
  try
    Root.AddPair('version', TJSONNumber.Create(1));
    Root.AddPair('active_chat_id', ActiveChatId);
    Items := TJSONArray.Create;
    Root.AddPair('models', Items);
    for var Model in Models do
      Items.AddElement(Model.ToJSON);
    Items := TJSONArray.Create;
    Root.AddPair('chats', Items);
    for var Chat in Chats do
      Items.AddElement(Chat.ToJSON);
    Bytes := TEncoding.UTF8.GetBytes(Root.ToJSON);
  finally
    Root.Free;
  end;
  Temporary := FPath + '.tmp';
  Stream := TFileStream.Create(Temporary, fmCreate);
  try
    if Length(Bytes) > 0 then
      Stream.WriteBuffer(Bytes[0], Length(Bytes));
    if not FlushFileBuffers(Stream.Handle) then
      RaiseLastOSError;
  finally
    Stream.Free;
  end;
  if TFile.Exists(FPath) and not FRecoveredBackup then
    TFile.Copy(FPath, FPath + '.bak', True);
  if not MoveFileEx(PChar(Temporary), PChar(FPath), MOVEFILE_REPLACE_EXISTING or MOVEFILE_WRITE_THROUGH) then
    RaiseLastOSError;
  FRecoveredBackup := False;
end;

procedure TChatStore.SeedLocalModels;

  procedure Add(const Name, Directory, Tokenizer: string);
  var
    Entry: TModelEntry;
  begin
    if not TDirectory.Exists(Directory) then
      Exit;
    Entry := TModelEntry.Create;
    Entry.Name := Name;
    Entry.Directory := Directory;
    Entry.TokenizerDirectory := Tokenizer;
    Entry.Inspect;
    if Entry.Available then
      Models.Add(Entry)
    else
      Entry.Free;
  end;

begin
  Add('Qwen3-Coder 30B · MoE', 'H:\Models\Qwen3-Coder-30B-A3B-Instruct', '');
  Add('Qwen3 1.7B', 'H:\Models\Qwen3-1.7B', '');
  Add('Qwen3 4B', 'H:\Models\FLUX.2-klein-4B\text_encoder', 'H:\Models\FLUX.2-klein-4B\tokenizer');
end;

function TChatStore.FindModel(const Id: string): TModelEntry;
begin
  for var Model in Models do
    if Model.Id = Id then
      Exit(Model);
  Result := nil;
end;

function TChatStore.FindChat(const Id: string): TChatSession;
begin
  for var Chat in Chats do
    if Chat.Id = Id then
      Exit(Chat);
  Result := nil;
end;

function TChatStore.NewChat: TChatSession;
begin
  Result := TChatSession.Create;
  if Models.Count > 0 then
    Result.Preferences.ModelId := Models[0].Id;
  Chats.Insert(0, Result);
  ActiveChatId := Result.Id;
end;

procedure TChatStore.RemoveChat(const Id: string);
var
  Chat: TChatSession;
begin
  Chat := FindChat(Id);
  if Chat <> nil then
    Chats.Remove(Chat);
  if ActiveChatId = Id then
  begin
    ActiveChatId := '';
    if Chats.Count > 0 then
      ActiveChatId := Chats[0].Id;
  end;
end;

procedure TChatStore.RemoveModel(const Id: string);
var
  Model: TModelEntry;
begin
  for var Chat in Chats do
    if Chat.Preferences.ModelId = Id then
      Chat.Preferences.ModelId := '';
  Model := FindModel(Id);
  if Model <> nil then
    Models.Remove(Model);
end;

end.

