unit QuickNN.LLMChat.Settings;

interface

uses
  QuickNN.Models.LLM.Types;

type
  TChatModel = (cmQwen34B, cmQwen317B, cmQwen3Coder30BA3B);

  TChatSettings = record
    Model: TChatModel;
    ModelDirectory, TokenizerDirectory, SystemPrompt: string;
    MaxContext: Integer;
    Generation: TLLMGenerationOptions;
    Memory: TLLMMemoryOptions;
  end;

function ChatSettings: TChatSettings;

procedure ResolveChatDirectories(var Settings: TChatSettings);

implementation

uses
  System.SysUtils,
  System.IOUtils;

function ChatSettings: TChatSettings;
begin
  Result := Default(TChatSettings);
  // Выберите модель здесь и запустите QuickNNChat (F9). Параметры запуска не нужны.
  Result.Model := cmQwen3Coder30BA3B;
  // Пустые пути используют выбранный пресет. Можно указать свои абсолютные каталоги.
  Result.ModelDirectory := '';
  Result.TokenizerDirectory := '';
  Result.SystemPrompt := 'You are a helpful assistant. Answer clearly and concisely.';
  Result.MaxContext := 2048;
  Result.Memory := TLLMMemoryOptions.Defaults;
  if Result.Model = cmQwen3Coder30BA3B then
  begin
    // Запрос 12 GiB; backend сохраняет предел 75% device-local heap (здесь около 11.70 GiB).
    Result.Memory.GPUWeightLimitBytes := UInt64(12) * 1024 * 1024 * 1024;
    Result.Memory.RAMCacheLimitBytes := UInt64(4) * 1024 * 1024 * 1024;
    // eqNative сохраняет исходные BF16/F32 веса экспертов.
    Result.Memory.ExpertQuantization := eqInt4;
  end;
  Result.Generation := TLLMGenerationOptions.Defaults;
  Result.Generation.MaxNewTokens := 256;
  Result.Generation.EnableThinking := False;
  // Temperature=0 выбирает самый вероятный токен; при sampling TopK допустим 1..64.
end;

procedure ResolveChatDirectories(var Settings: TChatSettings);
begin
  if Settings.ModelDirectory = '' then
    case Settings.Model of
      cmQwen34B: Settings.ModelDirectory := 'H:\Models\FLUX.2-klein-4B\text_encoder';
      cmQwen317B: Settings.ModelDirectory := 'H:\Models\Qwen3-1.7B';
      cmQwen3Coder30BA3B: Settings.ModelDirectory := 'H:\Models\Qwen3-Coder-30B-A3B-Instruct';
    end;
  if (Settings.TokenizerDirectory = '') and TFile.Exists(TPath.Combine(Settings.ModelDirectory, 'tokenizer.json')) then
    Settings.TokenizerDirectory := Settings.ModelDirectory;
  if Settings.TokenizerDirectory = '' then
    case Settings.Model of
      cmQwen34B: Settings.TokenizerDirectory := TPath.Combine(ExtractFileDir(Settings.ModelDirectory), 'tokenizer');
      cmQwen317B, cmQwen3Coder30BA3B: Settings.TokenizerDirectory := Settings.ModelDirectory;
    end;
end;

end.
