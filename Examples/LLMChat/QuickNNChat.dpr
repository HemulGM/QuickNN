program QuickNNChat;

{$APPTYPE CONSOLE}

uses
  System.SysUtils,
  System.Diagnostics,
  QuickNN.Models.LLM,
  QuickNN.Models.LLM.Types,
  QuickNN.LLMChat.Settings in 'QuickNN.LLMChat.Settings.pas',
  QuickNN.LLMChat.Console in 'QuickNN.LLMChat.Console.pas';

var
  Console: TChatConsole;

procedure AddMessage(var Messages: TLLMMessages; const Role: TLLMRole; const Text: string);
begin
  SetLength(Messages, Length(Messages) + 1);
  Messages[High(Messages)] := TLLMMessage.Create(Role, Text);
end;

procedure Help;
begin
  Console.WriteLine(
    '/new — новый диалог, /memory — память, /help — помощь, /exit — выход.'
  );
  Console.WriteLine(
    'Ctrl+C останавливает текущий ответ. История очищается только командой /new.'
  );
end;

procedure Run;
var
  Settings: TChatSettings;
  Model: TQNNLLM;
  Messages: TLLMMessages;
  Line: string;
  Response: TLLMGenerationResult;
  Timer: TStopwatch;

  procedure NewConversation;
  begin
    Messages := nil;
    if Settings.SystemPrompt <> '' then
      AddMessage(Messages, lrSystem, Settings.SystemPrompt);
    Model.ResetContext;
  end;

  procedure ShowMemory;
  var
    Usage: TLLMMemoryStatus;
  begin
    Usage := Model.MemoryStatus;
    Console.WriteLine(
      Format(
        'Веса GPU: %.2f / %.2f GiB; из них закреплено: %.2f GiB; RAM-кеш: %.2f / %.2f GiB.',
        [
          Usage.GPUWeightBytes / 1073741824.0,
          Usage.GPUWeightLimitBytes / 1073741824.0,
          Usage.FixedGPUWeightBytes / 1073741824.0,
          Usage.RAMCacheBytes / 1073741824.0,
          Usage.RAMCacheLimitBytes / 1073741824.0
        ]
      )
    );
    if Usage.IsMoE then
      if Usage.ExpertQuantization = eqInt4 then
      begin
        Console.WriteLine(
          Format(
            'Эксперты: INT4; упакованные веса: %.2f GiB.',
            [Usage.QuantizedMappedWeightBytes / 1073741824.0]
          )
        );
        Console.WriteLine('INT4-кеш: ' + Usage.ExpertCacheDirectory);
      end
      else
        Console.WriteLine('Эксперты: исходные BF16/F32.');
  end;

begin
  if SizeOf(Pointer) <> 8 then
    raise ENotSupportedException.Create('Чат LLM требует Win64. Выберите Win64 в конфигурации проекта.');
  Settings := ChatSettings;
  ResolveChatDirectories(Settings);
  if not Assigned(Settings.Memory.OnExpertCacheProgress) then
    Settings.Memory.OnExpertCacheProgress :=
      procedure(Done, Total: Integer)
      begin
        if ChatInterrupted then
          raise EAbort.Create('Подготовка INT4 остановлена.');
        Console.WriteLine(Format('Подготовка INT4-кеша: %d/%d частей.', [Done, Total]));
      end;
  Console.WriteLine('QuickNN / Qwen3 LLM');
  Console.WriteLine('Model: ' + Settings.ModelDirectory);
  Console.WriteLine('Tokenizer: ' + Settings.TokenizerDirectory);
  Console.WriteLine('Загрузка модели и выделение KV-кеша в VRAM…');
  Model := TQNNLLM.Create(Settings.ModelDirectory, Settings.TokenizerDirectory, Settings.MaxContext, Settings.Memory);
  try
    NewConversation;
    Console.WriteLine(
      Format(
        'KV-кеш: %.1f MiB; контекст: %d токенов.',
        [Model.KVCacheBytes / 1048576.0, Settings.MaxContext]
      )
    );
    if Model.MemoryStatus.IsMoE then
      Console.WriteLine(
        Format(
          'MoE: %d/%d экспертов; GPU-кеш: %.1f GiB; RAM-кеш: %.1f GiB.',
          [
            Model.MemoryStatus.ExpertsPerToken,
            Model.MemoryStatus.ExpertsPerLayer,
            Model.MemoryStatus.GPUWeightLimitBytes / 1073741824.0,
            Model.MemoryStatus.RAMCacheLimitBytes / 1073741824.0
          ]
        )
      );
    Help;
    repeat
      ClearChatInterrupt;
      Console.Write('Вы> ');
      if not Console.ReadLine(Line) then
        Break;
      if Trim(Line) = '' then
        Continue;
      if SameText(Trim(Line), '/exit') then
        Break;
      if SameText(Trim(Line), '/help') then
      begin
        Help;
        Continue
      end;
      if SameText(Trim(Line), '/memory') then
      begin
        ShowMemory;
        Continue;
      end;
      if SameText(Trim(Line), '/new') then
      begin
        NewConversation;
        Console.WriteLine('Начат новый диалог.');
        Continue;
      end;
      AddMessage(Messages, lrUser, Line);
      Console.Write('Qwen> ');
      Timer := TStopwatch.StartNew;
      try
        Response :=
          Model.Generate(
          Messages,
          Settings.Generation,
          procedure(const Text: string; var Cancel: Boolean)
          begin
            Console.Write(Text);
            Cancel := ChatInterrupted;
          end);
      except
        on E: Exception do
        begin
          Console.WriteLine;
          Console.WriteLine(E.ClassName + ': ' + E.Message);
          SetLength(Messages, Length(Messages) - 1);
          Model.ResetContext;
          Continue;
        end;
      end;
      Console.WriteLine;
      AddMessage(Messages, lrAssistant, Response.Text);
      Console.WriteLine(
        Format(
          '[%d токенов; %.2f с; decode: %.1f ток/с; первый токен: %.3f с; KV: %d; переиспользовано: %d]',
          [
            Response.GeneratedTokens,
            Timer.Elapsed.TotalSeconds,
            Response.DecodeTokensPerSecond,
            Response.TimeToFirstTokenSeconds,
            Model.ContextLength,
            Response.ReusedTokens
          ]
        )
      );
      case Response.FinishReason of
        lfCancelled:
          Console.WriteLine('[Ответ остановлен.]');
        lfContextFull:
          Console.WriteLine(
            '[Контекст заполнен. Команда /new начинает новый диалог.]'
          );
        lfMaxTokens:
          Console.WriteLine('[Достигнут лимит длины ответа из ChatSettings.]');
      end;
    until False;
  finally
    Model.Free;
  end;
end;

begin
  Console := TChatConsole.Create;
  try
    try
      InstallChatInterruptHandler;
      try
        Run
      finally
        RemoveChatInterruptHandler
      end;
    except
      on E: Exception do
      begin
        Console.WriteLine(E.ClassName + ': ' + E.Message);
        ExitCode := 1;
      end;
    end;
  finally
    Console.Free;
  end;
end.

