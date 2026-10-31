unit LLMUI.Engine;

interface

uses
  System.Classes, System.SysUtils, System.SyncObjs, QuickNN.Models.LLM,
  QuickNN.Models.LLM.Types, LLMUI.Data, LLMUI.Context.Types, LLMUI.Images;

type
  TInferenceRequest = record
    JobId, ChatId, MessageId: string;
    ModelName, Directory, TokenizerDirectory: string;
    IsMoE, ContinueResponse, GenerateImage: Boolean;
    Image: TChatImage;
    Continuation: TLLMContinuationState;
    Preferences: TChatPreferences;
    Messages: TLLMMessages;
    History: TChatContextHistory;
    ContextMemory: TChatContextMemory;
  end;

  TInferenceSnapshot = record
    JobId, ChatId, MessageId, Phase, Delta, ErrorText, LoadedModel: string;
    Busy, Finished, MemoryUpdated, ImageReady: Boolean;
    Image: TChatImage;
    ImageDescriptions: TArray<TImageDescription>;
    ContextMemory: TChatContextMemory;
    Result: TLLMGenerationResult;
    Continuation: TLLMContinuationState;
    KVCacheIdentity: NativeUInt;
    GPUWeightBytes, GPUWeightLimit, RAMBytes, RAMLimit, KVBytes: UInt64;
  end;

  // The same worker creates, uses and frees the resident Vulkan context.
  // The UI polls immutable snapshots; no queued closures can outlive the form.
  TInferenceWorker = class(TThread)
  private
    FLock: TCriticalSection;
    FWake: TEvent;
    FPending, FUnload: Boolean;
    FCancel: Integer;
    FRequest: TInferenceRequest;
    FSnapshot: TInferenceSnapshot;
    FModel: TQNNLLM;
    FSessionKey, FSessionChat: string;
    procedure SetPhase(const Text: string);
    procedure UpdateMemory;
    procedure RunRequest(const OriginalRequest: TInferenceRequest);
    procedure RunConversation(const Request: TInferenceRequest; var Response: TLLMGenerationResult; var Checkpoint: TLLMContinuationState);
    procedure ReleaseModel;
  protected
    procedure Execute; override;
  public
    constructor Create;
    destructor Destroy; override;
    function Submit(const Request: TInferenceRequest): Boolean;
    procedure Cancel;
    procedure Unload;
    function Poll: TInferenceSnapshot;
    function IsBusy: Boolean;
    procedure Shutdown;
  end;

implementation

uses
  QuickNN.Vulkan, System.Math, LLMUI.Context, QuickNN.Models.LLM.Vision;

constructor TInferenceWorker.Create;
begin
  // RTL starts Execute in AfterConstruction, after every field is ready.
  inherited Create(False);
  FreeOnTerminate := False;
  FLock := TCriticalSection.Create;
  FWake := TEvent.Create(nil, False, False, '');
  FSnapshot.Phase := 'Готово';
end;

destructor TInferenceWorker.Destroy;
begin
  Shutdown;
  FWake.Free;
  FLock.Free;
  inherited;
end;

procedure TInferenceWorker.Shutdown;
begin
  if FWake = nil then
    Exit;
  Cancel;
  Terminate;
  FWake.SetEvent;
  WaitFor;
  FLock.Acquire;
  try
    if FPending or FSnapshot.Busy then
    begin
      FSnapshot.Result.FinishReason := lfCancelled;
      FSnapshot.Phase := 'Ответ остановлен';
      FSnapshot.Busy := False;
      FSnapshot.Finished := True;
    end;
    FPending := False;
    FRequest := Default(TInferenceRequest);
  finally
    FLock.Release;
  end;
end;

function TInferenceWorker.Submit(const Request: TInferenceRequest): Boolean;
begin
  FLock.Acquire;
  try
    Result := not Terminated and not FSnapshot.Busy and not FSnapshot.Finished and not FPending;
    if not Result then
      Exit;
    TInterlocked.Exchange(FCancel, 0);
    FRequest := Request;
    FRequest.Messages := Copy(Request.Messages);
    FRequest.History := Copy(Request.History);
    FPending := True;
    FUnload := False;
    FSnapshot := Default(TInferenceSnapshot);
    FSnapshot.JobId := Request.JobId;
    FSnapshot.ChatId := Request.ChatId;
    FSnapshot.MessageId := Request.MessageId;
    FSnapshot.ContextMemory := Request.ContextMemory;
    FSnapshot.Phase := 'Подготовка…';
    FSnapshot.Busy := True;
  finally
    FLock.Release;
  end;
  FWake.SetEvent;
end;

procedure TInferenceWorker.Cancel;
begin
  TInterlocked.Exchange(FCancel, 1);
end;

procedure TInferenceWorker.Unload;
begin
  FLock.Acquire;
  try
    if FSnapshot.Busy or FPending then
      Exit;
    FUnload := True;
  finally
    FLock.Release;
  end;
  FWake.SetEvent;
end;

function TInferenceWorker.IsBusy: Boolean;
begin
  FLock.Acquire;
  try
    Result := FSnapshot.Busy or FSnapshot.Finished or FPending;
  finally
    FLock.Release;
  end;
end;

function TInferenceWorker.Poll: TInferenceSnapshot;
begin
  FLock.Acquire;
  try
    Result := FSnapshot;
    FSnapshot.Delta := '';
    FSnapshot.Finished := False;
    FSnapshot.MemoryUpdated := False;
    FSnapshot.ImageReady := False;
    FSnapshot.ImageDescriptions := nil;
  finally
    FLock.Release;
  end;
end;

procedure TInferenceWorker.SetPhase(const Text: string);
begin
  FLock.Acquire;
  try
    FSnapshot.Phase := Text;
  finally
    FLock.Release;
  end;
end;

procedure TInferenceWorker.UpdateMemory;
var
  Usage: TLLMMemoryStatus;
begin
  if FModel = nil then
    Exit;
  Usage := FModel.MemoryStatus;
  FLock.Acquire;
  try
    FSnapshot.GPUWeightBytes := Usage.GPUWeightBytes;
    FSnapshot.GPUWeightLimit := Usage.GPUWeightLimitBytes;
    FSnapshot.RAMBytes := Usage.RAMCacheBytes;
    FSnapshot.RAMLimit := Usage.RAMCacheLimitBytes;
    FSnapshot.KVBytes := FModel.KVCacheBytes;
    FSnapshot.KVCacheIdentity := FModel.KVCacheIdentity;
  finally
    FLock.Release;
  end;
end;

procedure TInferenceWorker.ReleaseModel;
begin
  FreeAndNil(FModel);
  FSessionKey := '';
  FSessionChat := '';
  QNNConfigureGPU(gpuAuto);
  FLock.Acquire;
  try
    FSnapshot.LoadedModel := '';
    FSnapshot.GPUWeightBytes := 0;
    FSnapshot.GPUWeightLimit := 0;
    FSnapshot.KVBytes := 0;
    FSnapshot.KVCacheIdentity := 0;
    FSnapshot.RAMBytes := 0;
    FSnapshot.RAMLimit := 0;
  finally
    FLock.Release;
  end;
end;

procedure TInferenceWorker.RunConversation(const Request: TInferenceRequest; var Response: TLLMGenerationResult; var Checkpoint: TLLMContinuationState);
var
  Processor: TChatContextProcessor;
  State: TLLMContinuationState;
  Input: TLLMMessages;
  Options: TLLMGenerationOptions;
  OnToken: TLLMTokenCallback;
  Continuing, Force: Boolean;
  InitialGenerated, Budget, Remaining, Used: Integer;
  StartedAt, FirstTextAt: UInt64;
  PriorTotal: Double;

  procedure PublishMemory;
  begin
    if (Processor = nil) or not Processor.Changed then
      Exit;
    FLock.Acquire;
    try
      FSnapshot.ContextMemory := Processor.Memory;
      FSnapshot.MemoryUpdated := True
    finally
      FLock.Release
    end;
  end;

begin
  Processor := nil;
  StartedAt := TThread.GetTickCount64;
  FirstTextAt := 0;
  PriorTotal := Response.TotalSeconds;
  State := Request.Continuation;
  Continuing := Request.ContinueResponse;
  Force := False;
  InitialGenerated := Response.GeneratedTokens;
  Budget := Request.Preferences.Generation.MaxNewTokens;
  if Request.Preferences.Generation.AutoContinue then
    Budget := Request.Preferences.Generation.HardMaxNewTokens;
  OnToken :=
    procedure(const Text: string; var Cancel: Boolean)
    begin
      Cancel := TInterlocked.CompareExchange(FCancel, 0, 0) <> 0;
      if Text = '' then
        Exit;
      if FirstTextAt = 0 then
        FirstTextAt := TThread.GetTickCount64;
      FLock.Acquire;
      try
        FSnapshot.Delta := FSnapshot.Delta + Text;
        if FModel.AutomaticContinuations > 0 then
          FSnapshot.Phase :=
            Format('Автопродолжение · блок %d…', [FModel.AutomaticContinuations + 1])
        else
          FSnapshot.Phase := 'Генерация ответа…';
      finally
        FLock.Release
      end;
      UpdateMemory;
    end;
  try
    if Length(Request.History) > 0 then
      Processor :=
        TChatContextProcessor.Create(
        FModel,
        Request.Preferences.Compaction,
        Request.History,
        Request.Preferences.SystemPrompt,
        Request.ContextMemory,
        procedure(const Phase: string)
        begin
          SetPhase(Phase)
        end,
        function: Boolean
        begin
          Result := TInterlocked.CompareExchange(FCancel, 0, 0) <> 0
        end);
    PublishMemory;
    while True do
    begin
      if TInterlocked.CompareExchange(FCancel, 0, 0) <> 0 then
        raise EAbort.Create('Отменено');
      Remaining := Budget - (Response.GeneratedTokens - InitialGenerated);
      if Remaining <= 0 then
        Break;
      if Processor <> nil then
      begin
        if Continuing then
        begin
          State := FModel.RebaseContinuation(State, Processor.Messages, Request.Preferences.Generation.EnableThinking);
          Checkpoint := State;
        end;
        Processor.Prepare(State, Continuing, Request.Preferences.Generation.EnableThinking, Remaining, Force);
        PublishMemory;
        Input := Processor.Messages;
        if Continuing then
          Checkpoint := State;
      end
      else
        Input := Request.Messages;
      if Continuing then
        Used := Length(State.Tokens)
      else
        Used := Length(FModel.EncodeChat(Input, Request.Preferences.Generation.EnableThinking));
      if Used >= FModel.ContextCapacity then
      begin
        Response.FinishReason := lfContextFull;
        SetPhase(
          'Защищённые сообщения заполняют контекст. Увеличьте контекст или уменьшите сохраняемую историю.'
        );
        Exit;
      end;
      Options := Request.Preferences.Generation;
      Options.HardMaxNewTokens := Remaining;
      Options.MaxNewTokens := Min(Options.MaxNewTokens, Remaining);
      if not Options.AutoContinue then
        Options.MaxNewTokens := Remaining;
      if Continuing then
      begin
        SetPhase('Продолжение ответа…');
        Response := FModel.ContinueGeneration(State, Options, OnToken);
      end
      else
      begin
        SetPhase('Обработка запроса…');
        Response := FModel.Generate(Input, Options, OnToken);
      end;
      var Latest := FModel.Continuation;
      if Latest.Available or (Response.FinishReason = lfEOS) then
        Checkpoint := Latest
      else if Continuing and (Response.Text = State.Text) then
        Checkpoint := State
      else
        Checkpoint := Default(TLLMContinuationState);
      if (Response.FinishReason <> lfContextFull)
        or (Processor = nil)
        or not Checkpoint.Available
        or not Request.Preferences.Compaction.Enabled
        or (Response.GeneratedTokens - InitialGenerated >= Budget) then
        Break;
      // Continue within the original reply budget. Compaction cannot renew its hard cap.
      State := Checkpoint;
      Continuing := True;
      Force := True;
    end;
  finally
    Processor.Free;
    Response.TotalSeconds := PriorTotal + (TThread.GetTickCount64 - StartedAt) / 1000.0;
    if (InitialGenerated = 0) and (FirstTextAt <> 0) then
      Response.TimeToFirstTokenSeconds := (FirstTextAt - StartedAt) / 1000.0;
    if Checkpoint.Available then
    begin
      Checkpoint.TotalSeconds := Response.TotalSeconds;
      Checkpoint.TimeToFirstTokenSeconds := Response.TimeToFirstTokenSeconds;
    end;
  end;
end;

procedure TInferenceWorker.RunRequest(const OriginalRequest: TInferenceRequest);
var
  Request: TInferenceRequest;
  Memory: TLLMMemoryOptions;
  Response: TLLMGenerationResult;
  ErrorText, Key: string;
  Checkpoint: TLLMContinuationState;

  procedure PrepareMessage(var Message: TLLMMessage; const GeneratedImage: Boolean = False);
  begin
    if Length(Message.Images) = 0 then
      Exit;
    Message.Images := Copy(Message.Images);
    if LLMVisionAvailable(Request.Directory) then
      for var I := 0 to High(Message.Images) do
        LLMPrepareImage(Request.Directory, Message.Images[I],
          procedure
          begin
            ReleaseModel
          end,
          procedure(const Text: string)
          begin
            SetPhase(Text)
          end,
          function: Boolean
          begin
            Result := TInterlocked.CompareExchange(FCancel, 0, 0) <> 0
          end)
    else
    begin
      for var Image in Message.Images do
      begin
        if Image.Description <> '' then
          Message.Content := Message.Content + #10 + '[Описание изображения: ' + Image.Description + ']'
        else if GeneratedImage then
          Message.Content := Message.Content + #10 + '[Изображение сохранено. Его содержимое ещё не анализировалось.]'
        else
          raise ENotSupportedException.Create('У изображения пока нет описания. Выберите vision-модель.');
      end;
      Message.Images := nil;
    end;
  end;

begin
  Request := OriginalRequest;
  Request.History := Copy(OriginalRequest.History);
  Request.Messages := Copy(OriginalRequest.Messages);
  Response := Default(TLLMGenerationResult);
  Response.FinishReason := lfCancelled;
  if Request.ContinueResponse then
  begin
    Response.Text := Request.Continuation.Text;
    Response.PromptTokens := Request.Continuation.PromptTokens;
    Response.GeneratedTokens := Request.Continuation.GeneratedTokens;
    Response.DecodeSeconds := Request.Continuation.DecodeSeconds;
    Response.TotalSeconds := Request.Continuation.TotalSeconds;
    Response.TimeToFirstTokenSeconds := Request.Continuation.TimeToFirstTokenSeconds;
  end;
  ErrorText := '';
  Checkpoint := Request.Continuation;
  try
    if TInterlocked.CompareExchange(FCancel, 0, 0) <> 0 then
      raise EAbort.Create('Отменено');
    if Request.GenerateImage then
    begin
      ReleaseModel;
      SetPhase('Загрузка модели изображений…');
      GenerateChatImage(Request.Image,
        procedure(const Text: string)
        begin
          SetPhase(Text)
        end,
        function: Boolean
        begin
          Result := TInterlocked.CompareExchange(FCancel, 0, 0) <> 0
        end);
      FLock.Acquire;
      try
        FSnapshot.Image := Request.Image;
        FSnapshot.ImageReady := True
      finally
        FLock.Release
      end;
      Response.FinishReason := lfEOS;
      SetPhase('Изображение готово');
    end
    else
    begin
      var Boundary := ContextMemoryBoundary(Request.History, Request.Preferences.SystemPrompt, Request.ContextMemory);
      if Boundary >= ContextProtectedStart(Request.History, Request.Preferences.Compaction.KeepRecentTurns) then
        Boundary := -1;
      for var I := Boundary + 1 to High(Request.History) do
        PrepareMessage(Request.History[I].Message, Request.History[I].Id.EndsWith('/image'));
      // ContextHistory is authoritative when present; avoid encoding attachments twice.
      if Length(Request.History) = 0 then
        for var I := 0 to High(Request.Messages) do
          PrepareMessage(Request.Messages[I]);
      Key :=
        Request.Directory
        + #1
        + Request.TokenizerDirectory
        + #1
        + Request.Preferences.Context.ToString
        + '/'
        + Request.Preferences.GPUCacheGiB.ToString
        + '/'
        + Request.Preferences.RAMCacheGiB.ToString
        + '/'
        + BoolToStr(Request.Preferences.Int4Experts, True);
      if (FModel = nil) or (FSessionKey <> Key) then
      begin
        ReleaseModel;
        QNNConfigureGPU(gpuForce);
        SetPhase('Загрузка модели ' + Request.ModelName + '…');
        Memory := TLLMMemoryOptions.Defaults;
        Memory.GPUWeightLimitBytes := UInt64(Request.Preferences.GPUCacheGiB) * 1024 * 1024 * 1024;
        Memory.RAMCacheLimitBytes := UInt64(Request.Preferences.RAMCacheGiB) * 1024 * 1024 * 1024;
        if Request.Preferences.Int4Experts then
          Memory.DenseQuantization := wqInt4
        else
          Memory.DenseQuantization := wqNative;
        if Request.IsMoE and Request.Preferences.Int4Experts then
          Memory.ExpertQuantization := eqInt4;
        Memory.OnExpertCacheProgress :=
          procedure(Done, Total: Integer)
          begin
            if TInterlocked.CompareExchange(FCancel, 0, 0) <> 0 then
              raise EAbort.Create('Отменено');
            SetPhase(Format('Подготовка INT4-кеша: %d/%d частей', [Done, Total]));
          end;
        FModel := TQNNLLM.Create(Request.Directory, Request.TokenizerDirectory, Request.Preferences.Context, Memory);
        FSessionKey := Key;
      end;
      if FSessionChat <> Request.ChatId then
      begin
        FModel.ResetContext;
        FSessionChat := Request.ChatId;
      end;
      FLock.Acquire;
      try
        FSnapshot.LoadedModel := Request.ModelName;
      finally
        FLock.Release;
      end;
      UpdateMemory;
      if TInterlocked.CompareExchange(FCancel, 0, 0) <> 0 then
        raise EAbort.Create('Отменено');
      RunConversation(Request, Response, Checkpoint);
      UpdateMemory;
      if Response.FinishReason = lfCancelled then
        SetPhase('Ответ остановлен')
      else if Response.FinishReason = lfContextFull then
        SetPhase(
          'Контекст заполнен. Увеличьте контекст, включите автосжатие или уменьшите число защищённых запросов.'
        )
      else if Response.FinishReason = lfHardLimit then
        SetPhase('Достигнут жёсткий предел. Можно продолжить ответ.')
      else if Response.FinishReason = lfMaxTokens then
        SetPhase('Достигнут мягкий лимит. Можно продолжить ответ.')
      else
        SetPhase('Готово');
    end;
  except
    on E: Exception do
    begin
      if (E is EAbort) or (TInterlocked.CompareExchange(FCancel, 0, 0) <> 0) then
      begin
        Response.FinishReason := lfCancelled;
        SetPhase('Ответ остановлен');
      end
      else if E is EChatContextError then
      begin
        Response.FinishReason := lfContextFull;
        SetPhase(E.Message);
      end
      else
      begin
        ErrorText := E.Message;
        SetPhase('Не удалось выполнить запрос');
        try
          ReleaseModel
        except
          on CleanupError: Exception do
            ErrorText := ErrorText + sLineBreak + CleanupError.Message;
        end;
      end;
    end;
  end;
  if not Checkpoint.Available and Request.ContinueResponse and (Response.FinishReason <> lfEOS) then
    Checkpoint := Request.Continuation;
  FLock.Acquire;
  try
    if (Response.Text <> '') and (Response.FinishReason <> lfCancelled) then
    begin
      var Description := Response.Text;
      if Description.StartsWith('<think>') then
      begin
        var Closing := Pos('</think>', Description);
        if Closing > 0 then
          Description := Copy(Description, Closing + 8, MaxInt).Trim
        else
          Description := '';
      end;
      if Description <> '' then
        for var Item in Request.History do
          if (Item.Message.Role = lrUser) and (Item.Id = Request.History[High(Request.History)].Id) then
            for var Image in Item.Message.Images do
              if (Image.Description = '') and (Image.Identity <> '') then
              begin
                var Observation := Default(TImageDescription);
                Observation.MessageId := Item.Id;
                Observation.ContentHash := Image.Identity;
                Observation.Text := 'Вопрос об изображении: ' + Item.Message.Content + #10 + 'Ответ vision-модели: ' + Description;
                FSnapshot.ImageDescriptions := FSnapshot.ImageDescriptions + [Observation];
              end;
    end;
    FSnapshot.Continuation := Checkpoint;
    FSnapshot.Result := Response;
    FSnapshot.ErrorText := ErrorText;
    FSnapshot.Busy := False;
    FSnapshot.Finished := True;
  finally
    FLock.Release;
  end;
end;

procedure TInferenceWorker.Execute;
var
  Request: TInferenceRequest;
  HasRequest, ReleaseRequested: Boolean;
begin
  try
    while not Terminated do
    begin
      FWake.WaitFor(High(Cardinal));
      if Terminated then
        Break;
      FLock.Acquire;
      try
        HasRequest := FPending;
        ReleaseRequested := FUnload;
        Request := FRequest;
        FRequest := Default(TInferenceRequest);
        FPending := False;
        FUnload := False;
      finally
        FLock.Release;
      end;
      if HasRequest then
      begin
        RunRequest(Request);
        Request := Default(TInferenceRequest)
      end
      else if ReleaseRequested then
      begin
        ReleaseModel;
        SetPhase('Модель выгружена. VRAM освобождена.');
      end;
    end;
  finally
    ReleaseModel;
  end;
end;

end.

