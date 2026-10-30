unit QuickNN.Models.LLM.Types;

interface

uses
  System.SysUtils;

type
  TLLMRole = (lrSystem, lrUser, lrAssistant);
  TLLMMessage = record
    Role: TLLMRole;
    Content: string;
    class function Create(const Role: TLLMRole; const Content: string): TLLMMessage; static;
  end;
  TLLMMessages = TArray<TLLMMessage>;
  TLLMGenerationOptions = record
    // MaxNewTokens is a soft boundary when AutoContinue is enabled.
    // Each explicit Generate/ContinueGeneration call gets its own hard budget.
    MaxNewTokens, HardMaxNewTokens, TopK: Integer;
    AutoContinue: Boolean;
    Temperature, TopP, RepetitionPenalty: Single;
    Seed: UInt64;
    EnableThinking: Boolean;
    class function Defaults: TLLMGenerationOptions; static;
  end;
  TLLMExpertQuantization = (eqNative, eqInt4);
  TLLMMemoryOptions = record
    // Zero GPU limit preserves the caller's current global weight budget.
    GPUWeightLimitBytes, RAMCacheLimitBytes: UInt64;
    // Zero selects scalar uploads; otherwise capped at 128 MiB and available VRAM.
    ExpertBatchBytes: UInt64;
    ExpertQuantization: TLLMExpertQuantization;
    ExpertCacheDirectory: string;
    OnExpertCacheProgress: TProc<Integer, Integer>;
    class function Defaults: TLLMMemoryOptions; static;
  end;
  TLLMMemoryStatus = record
    IsMoE: Boolean;
    ExpertQuantization: TLLMExpertQuantization;
    QuantizedMappedWeightBytes: UInt64;
    ExpertCacheDirectory: string;
    ExpertsPerLayer, ExpertsPerToken: Integer;
    GPUWeightBytes, GPUWeightLimitBytes, FixedGPUWeightBytes: UInt64;
    RAMCacheBytes, RAMCacheLimitBytes, RAMCacheHits, RAMCacheMisses, RAMCacheEvictions: UInt64;
    MappedWeightBytes, RouterReadbacks, RouterReadbackBytes, DispatchUploadBytes: UInt64;
  end;
  TLLMFinishReason = (lfEOS, lfMaxTokens, lfCancelled, lfContextFull, lfHardLimit);
  // Token IDs and sampler/UTF-8 state only. GPU weights and KV never enter this record.
  // Immutable once published; it can rebuild a session after unloading the model.
  TLLMContinuationState = record
    ModelKey, Text, PromptSignature: string;
    Tokens: TArray<LongInt>;
    PromptTokens, GeneratedTokens: Integer;
    RandomState: UInt32;
    PendingUTF8: RawByteString;
    DecodeSeconds, TotalSeconds, TimeToFirstTokenSeconds: Double;
    function Available: Boolean;
  end;
  TLLMGenerationResult = record
    Text: string;
    PromptTokens, GeneratedTokens, ReusedTokens, PrefillTokens, AutomaticContinuations: Integer;
    ResumedFromCache: Boolean;
    FinishReason: TLLMFinishReason;
    TimeToFirstTokenSeconds, DecodeSeconds, TotalSeconds: Double;
    function DecodeTokensPerSecond: Double;
  end;
  TLLMTokenCallback = reference to procedure(const Text: string; var Cancel: Boolean);

implementation

function TLLMContinuationState.Available: Boolean;
begin
  Result :=
      (ModelKey <> '')
          and (PromptTokens > 0)
          and (GeneratedTokens > 0)
          and (Int64(PromptTokens) + GeneratedTokens = Length(Tokens));
end;

class function TLLMMemoryOptions.Defaults: TLLMMemoryOptions;
begin
  Result := Default(TLLMMemoryOptions);
  Result.RAMCacheLimitBytes := UInt64(4) * 1024 * 1024 * 1024;
  Result.ExpertBatchBytes := UInt64(128) * 1024 * 1024;
end;

function TLLMGenerationResult.DecodeTokensPerSecond: Double;
begin
  if (GeneratedTokens > 1) and (DecodeSeconds > 0) then
    Result := (GeneratedTokens - 1) / DecodeSeconds
  else
    Result := 0;
end;

class function TLLMMessage.Create(const Role: TLLMRole; const Content: string): TLLMMessage;
begin
  Result.Role := Role;
  Result.Content := Content;
end;

class function TLLMGenerationOptions.Defaults: TLLMGenerationOptions;
begin
  Result := Default(TLLMGenerationOptions);
  Result.MaxNewTokens := 256;
  Result.HardMaxNewTokens := 4096;
  Result.TopK := 20;
  Result.Temperature := 0.6;
  Result.TopP := 0.95;
  Result.RepetitionPenalty := 1;
  Result.Seed := 42;
  Result.EnableThinking := False;
end;

end.
