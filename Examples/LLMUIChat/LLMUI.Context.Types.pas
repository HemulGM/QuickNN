unit LLMUI.Context.Types;

interface

uses
  System.JSON, QuickNN.Models.LLM.Types;

type
  TChatCompactionOptions = record
    Enabled: Boolean;
    ThresholdPercent, KeepRecentTurns, SummaryTokens, ReplyReserveTokens: Integer;
    class function Defaults: TChatCompactionOptions; static;
    function ToJSON: TJSONObject;
    class function FromJSON(const Value: TJSONValue): TChatCompactionOptions; static;
  end;

  TChatContextMemory = record
    Text, ThroughMessageId, SourceHash: string;
    Compactions, CoveredMessages, BeforeTokens, AfterTokens: Integer;
    function Available: Boolean;
    function ToJSON: TJSONObject;
    class function FromJSON(const Value: TJSONValue): TChatContextMemory; static;
  end;

  TChatContextMessage = record
    Id: string;
    Message: TLLMMessage;
  end;

  TChatContextHistory = TArray<TChatContextMessage>;

implementation

uses
  System.SysUtils, System.Math;

class function TChatCompactionOptions.Defaults: TChatCompactionOptions;
begin
  Result := Default(TChatCompactionOptions);
  Result.Enabled := True;
  Result.ThresholdPercent := 80;
  Result.KeepRecentTurns := 3;
  Result.SummaryTokens := 256;
  Result.ReplyReserveTokens := 256;
end;

function TChatCompactionOptions.ToJSON: TJSONObject;
begin
  Result := TJSONObject.Create;
  Result.AddPair('enabled', TJSONBool.Create(Enabled));
  Result.AddPair('threshold_percent', TJSONNumber.Create(ThresholdPercent));
  Result.AddPair('keep_recent_turns', TJSONNumber.Create(KeepRecentTurns));
  Result.AddPair('summary_tokens', TJSONNumber.Create(SummaryTokens));
  Result.AddPair('reply_reserve_tokens', TJSONNumber.Create(ReplyReserveTokens));
end;

class function TChatCompactionOptions.FromJSON(const Value: TJSONValue): TChatCompactionOptions;
begin
  Result := Defaults;
  if not (Value is TJSONObject) then
    Exit;
  Result.Enabled := Value.GetValue<Boolean>('enabled', Result.Enabled);
  Result.ThresholdPercent := EnsureRange(Value.GetValue<Integer>('threshold_percent', Result.ThresholdPercent), 50, 95);
  Result.KeepRecentTurns := EnsureRange(Value.GetValue<Integer>('keep_recent_turns', Result.KeepRecentTurns), 1, 16);
  Result.SummaryTokens := EnsureRange(Value.GetValue<Integer>('summary_tokens', Result.SummaryTokens), 32, 1024);
  Result.ReplyReserveTokens :=
    EnsureRange(Value.GetValue<Integer>('reply_reserve_tokens', Result.ReplyReserveTokens), 16, 2048);
end;

function TChatContextMemory.Available: Boolean;
begin
  Result := (Text.Trim <> '') and (ThroughMessageId <> '') and (Length(SourceHash) = 64);
end;

function TChatContextMemory.ToJSON: TJSONObject;
begin
  Result := TJSONObject.Create;
  Result.AddPair('version', TJSONNumber.Create(1));
  Result.AddPair('text', Text);
  Result.AddPair('through_message_id', ThroughMessageId);
  Result.AddPair('source_hash', SourceHash);
  Result.AddPair('compactions', TJSONNumber.Create(Compactions));
  Result.AddPair('covered_messages', TJSONNumber.Create(CoveredMessages));
  Result.AddPair('before_tokens', TJSONNumber.Create(BeforeTokens));
  Result.AddPair('after_tokens', TJSONNumber.Create(AfterTokens));
end;

class function TChatContextMemory.FromJSON(const Value: TJSONValue): TChatContextMemory;
begin
  Result := Default(TChatContextMemory);
  if not (Value is TJSONObject) then
    Exit;
  try
    if Value.GetValue<Integer>('version', 0) <> 1 then
      Exit;
    Result.Text := Value.GetValue<string>('text', '');
    Result.ThroughMessageId := Value.GetValue<string>('through_message_id', '');
    Result.SourceHash := Value.GetValue<string>('source_hash', '');
    Result.Compactions := Max(0, Value.GetValue<Integer>('compactions', 0));
    Result.CoveredMessages := Max(0, Value.GetValue<Integer>('covered_messages', 0));
    Result.BeforeTokens := Max(0, Value.GetValue<Integer>('before_tokens', 0));
    Result.AfterTokens := Max(0, Value.GetValue<Integer>('after_tokens', 0));
    if not Result.Available or (Length(Result.Text) > 32768) then
      Result := Default(TChatContextMemory);
  except
    Result := Default(TChatContextMemory)
  end;
end;

end.

