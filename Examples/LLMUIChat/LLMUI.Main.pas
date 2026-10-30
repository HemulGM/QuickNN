unit LLMUI.Main;

interface

uses
  System.SysUtils, System.Types, System.UITypes, System.Classes,
  System.Generics.Collections, FMX.Types, FMX.Controls, FMX.Forms, FMX.Graphics,
  FMX.Dialogs, FMX.Filter.Effects, FMX.Layouts, FMX.StdCtrls, FMX.Objects,
  FMX.TabControl, FMX.Edit, FMX.EditBox, FMX.NumberBox, FMX.ListBox,
  FMX.Memo.Types, FMX.ScrollBox, FMX.Memo, FMX.Controls.Presentation, LLMUI.Data,
  LLMUI.Engine, LLMUI.Message, LLMUI.Entry;

type
  TFormMain = class(TForm)
    RectangleCanvas: TRectangle;
    LayoutRoot: TLayout;
    LayoutNavigation: TLayout;
    LayoutBrand: TLayout;
    RectangleBrand: TRectangle;
    PathBrandIcon: TPath;
    LabelBrand: TLabel;
    LabelLocalBadge: TLabel;
    RectangleNavigationDivider: TRectangle;
    TabControlMain: TTabControl;
    TabItemChats: TTabItem;
    LayoutChatPage: TLayout;
    LayoutThreads: TLayout;
    ButtonNewChat: TButton;
    PathNewChatIcon: TPath;
    EditSearch: TEdit;
    LabelThreads: TLabel;
    VertScrollBoxThreads: TVertScrollBox;
    LayoutThreadItems: TLayout;
    LayoutThreadsFooter: TLayout;
    ButtonManageModels: TButton;
    PathManageModelsIcon: TPath;
    LabelSidebarNote: TLabel;
    LayoutChatSettings: TLayout;
    LayoutChatSettingsHeading: TLayout;
    LayoutPreferenceTitleRow: TLayout;
    ButtonHidePreferences: TButton;
    PathHidePreferencesIcon: TPath;
    LabelChatSettings: TLabel;
    VertScrollBoxPreferences: TVertScrollBox;
    LayoutPreferenceFields: TLayout;
    LayoutModelChoice: TLayout;
    LabelModelChoice: TLabel;
    ComboBoxModel: TComboBox;
    LabelModelHint: TLabel;
    LayoutTemperature: TLayout;
    LayoutTemperatureHeading: TLayout;
    PanelTemperatureValue: TPanel;
    LabelTemperatureValue: TLabel;
    LabelTemperature: TLabel;
    TrackBarTemperature: TTrackBar;
    LayoutTemperatureScale: TLayout;
    LabelTemperatureCreative: TLabel;
    LabelTemperaturePrecise: TLabel;
    LayoutContext: TLayout;
    LabelContext: TLabel;
    ComboBoxContext: TComboBox;
    LabelKVEstimate: TLabel;
    LabelContextMemory: TLabel;
    LayoutCompaction: TLayout;
    LabelCompactionTitle: TLabel;
    LayoutAutoCompact: TLayout;
    LabelAutoCompact: TLabel;
    SwitchAutoCompact: TSwitch;
    LabelCompactionState: TLabel;
    LayoutCompactThreshold: TLayout;
    LabelCompactThreshold: TLabel;
    LabelCompactThresholdHint: TLabel;
    NumberBoxCompactThreshold: TNumberBox;
    LayoutKeepRecentTurns: TLayout;
    LabelKeepRecentTurns: TLabel;
    LabelKeepRecentTurnsHint: TLabel;
    NumberBoxKeepRecentTurns: TNumberBox;
    LayoutSummaryTokens: TLayout;
    LabelSummaryTokens: TLabel;
    LabelSummaryTokensHint: TLabel;
    NumberBoxSummaryTokens: TNumberBox;
    LayoutReplyReserve: TLayout;
    LabelReplyReserve: TLabel;
    LabelReplyReserveHint: TLabel;
    NumberBoxReplyReserve: TNumberBox;
    LayoutAnswerLength: TLayout;
    LabelAnswerLength: TLabel;
    NumberBoxMaxTokens: TNumberBox;
    LayoutHardLimit: TLayout;
    LabelHardLimit: TLabel;
    LabelHardLimitHint: TLabel;
    NumberBoxHardMaxTokens: TNumberBox;
    LayoutAutoContinue: TLayout;
    LabelAutoContinue: TLabel;
    SwitchAutoContinue: TSwitch;
    LabelAnswerLengthHint: TLabel;
    LayoutSystemPrompt: TLayout;
    LabelSystemPrompt: TLabel;
    MemoSystemPrompt: TMemo;
    LayoutStreamingSwitch: TLayout;
    LabelStreamingTitle: TLabel;
    SwitchStreaming: TSwitch;
    LabelStreaming: TLabel;
    LabelAdvanced: TLabel;
    LayoutTopP: TLayout;
    LabelTopP: TLabel;
    NumberBoxTopP: TNumberBox;
    LayoutTopK: TLayout;
    LabelTopK: TLabel;
    NumberBoxTopK: TNumberBox;
    LayoutRepetition: TLayout;
    LabelRepetition: TLabel;
    NumberBoxRepetition: TNumberBox;
    LayoutSeed: TLayout;
    LabelSeed: TLabel;
    NumberBoxSeed: TNumberBox;
    CheckBoxThinking: TCheckBox;
    LayoutPrecision: TLayout;
    LabelPrecision: TLabel;
    ComboBoxPrecision: TComboBox;
    LayoutGPUCache: TLayout;
    LabelGPUCache: TLabel;
    NumberBoxGPUCache: TNumberBox;
    LayoutRAMCache: TLayout;
    LabelRAMCache: TLabel;
    NumberBoxRAMCache: TNumberBox;
    LayoutConversation: TLayout;
    LayoutChatHeading: TLayout;
    ButtonTogglePreferences: TButton;
    PathTogglePreferencesIcon: TPath;
    ButtonDeleteChat: TButton;
    PathDeleteChatIcon: TPath;
    ButtonExportChat: TButton;
    PathExportChatIcon: TPath;
    LayoutChatTitle: TLayout;
    EditChatTitle: TEdit;
    LabelSelectedModel: TLabel;
    LayoutComposerDock: TLayout;
    LayoutLatestOverlay: TLayout;
    LayoutLatestButtonDock: TLayout;
    ButtonLatestMessage: TButton;
    PathLatestMessageIcon: TPath;
    PanelComposer: TPanel;
    LayoutComposer: TLayout;
    LayoutComposerFooter: TLayout;
    LabelSendHint: TLabel;
    ButtonSend: TButton;
    PathSendIcon: TPath;
    PathStopIcon: TPath;
    MemoComposer: TMemo;
    LayoutStatus: TLayout;
    ButtonUnloadModel: TButton;
    PathUnloadModelIcon: TPath;
    LabelMemory: TLabel;
    LabelStatus: TLabel;
    VertScrollBoxMessages: TVertScrollBox;
    LayoutMessageItems: TLayout;
    LayoutWelcome: TLayout;
    LayoutWelcomeContent: TLayout;
    LayoutWelcomeSymbol: TLayout;
    RectangleWelcomeSymbol: TRectangle;
    PathWelcomeIcon: TPath;
    LabelWelcomeTitle: TLabel;
    LabelWelcomeHint: TLabel;
    LayoutSuggestions: TLayout;
    ButtonSuggestionCode: TButton;
    ButtonSuggestionIdea: TButton;
    ButtonWelcomeModels: TButton;
    RectangleThreadsDivider: TRectangle;
    RectanglePreferencesDivider: TRectangle;
    TabItemSettings: TTabItem;
    LayoutSettingsPage: TLayout;
    LayoutSettingsHeader: TLayout;
    ButtonBack: TButton;
    PathBackIcon: TPath;
    LabelSettingsTitle: TLabel;
    LabelSettingsHint: TLabel;
    LayoutModelList: TLayout;
    ButtonNewModel: TButton;
    PathNewModelIcon: TPath;
    VertScrollBoxModels: TVertScrollBox;
    LayoutModelItems: TLayout;
    PanelModelEditor: TPanel;
    VertScrollBoxModelEditor: TVertScrollBox;
    LayoutModelEditorFields: TLayout;
    LabelModelEditorTitle: TLabel;
    LayoutModelName: TLayout;
    LabelModelName: TLabel;
    EditModelName: TEdit;
    LayoutModelDirectory: TLayout;
    LabelModelDirectory: TLabel;
    LayoutModelDirectoryInput: TLayout;
    ButtonBrowseModel: TButton;
    PathBrowseModelIcon: TPath;
    EditModelDirectory: TEdit;
    LayoutTokenizerDirectory: TLayout;
    LabelTokenizerDirectory: TLabel;
    LayoutTokenizerDirectoryInput: TLayout;
    ButtonBrowseTokenizer: TButton;
    PathBrowseTokenizerIcon: TPath;
    EditTokenizerDirectory: TEdit;
    LabelTokenizerHint: TLabel;
    LabelModelValidation: TLabel;
    LayoutModelActions: TLayout;
    ButtonSaveModel: TButton;
    ButtonRemoveModel: TButton;
    LabelStorageHeading: TLabel;
    LabelStorageHint: TLabel;
    LabelStoragePath: TLabel;
    ButtonOpenStorage: TButton;
    PathOpenStorageIcon: TPath;
    TimerUI: TTimer;
    SaveDialogExport: TSaveDialog;
    StyleBookWinUI3: TStyleBook;
    procedure FormCreate(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
    procedure FormResize(Sender: TObject);
    procedure FormCloseQuery(Sender: TObject; var CanClose: Boolean);
    procedure FormKeyDown(Sender: TObject; var Key: Word; var KeyChar: WideChar; Shift: TShiftState);
    procedure ButtonBackClick(Sender: TObject);
    procedure ButtonManageModelsClick(Sender: TObject);
    procedure ButtonNewChatClick(Sender: TObject);
    procedure EditSearchChangeTracking(Sender: TObject);
    procedure ButtonTogglePreferencesClick(Sender: TObject);
    procedure EditChatTitleExit(Sender: TObject);
    procedure PreferencesChanged(Sender: TObject);
    procedure MemoComposerChangeTracking(Sender: TObject);
    procedure MemoComposerKeyDown(Sender: TObject; var Key: Word; var KeyChar: WideChar; Shift: TShiftState);
    procedure ButtonSendClick(Sender: TObject);
    procedure ButtonDeleteChatClick(Sender: TObject);
    procedure ButtonExportChatClick(Sender: TObject);
    procedure ButtonUnloadModelClick(Sender: TObject);
    procedure ButtonSuggestionClick(Sender: TObject);
    procedure ButtonNewModelClick(Sender: TObject);
    procedure ButtonSaveModelClick(Sender: TObject);
    procedure ButtonRemoveModelClick(Sender: TObject);
    procedure ButtonBrowseModelClick(Sender: TObject);
    procedure ButtonBrowseTokenizerClick(Sender: TObject);
    procedure ButtonOpenStorageClick(Sender: TObject);
    procedure TimerUITimer(Sender: TObject);
    procedure MessagesResize(Sender: TObject);
    procedure ButtonLatestMessageClick(Sender: TObject);
    procedure MessagesViewportPositionChange(Sender: TObject; const OldViewportPosition, NewViewportPosition: TPointF; const ContentSizeChanged: Boolean);
    procedure LayoutComposerDockResize(Sender: TObject);
    procedure LayoutMessageItemsResize(Sender: TObject);
  private
    FStore: TChatStore;
    FEngine: TInferenceWorker;
    FFrames: TDictionary<string, TFrameMessage>;
    FModelIds: TList<string>;
    FReady, FUpdating, FClosing, FDirty, FRendering: Boolean;
    FFollowLatest: Boolean;
    FLastMessageScrollLimit: Single;
    FUpdatingViewport: Integer;
    FSaveAt: UInt64;
    FEditingModelId, FJobId, FJobChatId, FJobMessageId: string;
    FLastPhase: string;
    function ActiveChat: TChatSession;
    procedure ApplyWindowColors;
    procedure MarkDirty;
    procedure SaveState;
    procedure RenderThreads;
    procedure RenderModels;
    procedure RenderMessages(const ScrollToEnd: Boolean = True);
    procedure SelectChat(const Id: string);
    procedure ThreadSelected(Sender: TObject);
    procedure ModelSelected(Sender: TObject);
    procedure LoadPreferences;
    procedure RefreshModelsCombo;
    procedure UpdateAvailability;
    procedure UpdateKVLabel;
    procedure UpdateContextMemory;
    procedure SelectModelEditor(const Id: string);
    procedure Send(const Retry: Boolean; const ContinueAnswer: Boolean = False);
    function CanContinueMessage(const Chat: TChatSession; const Message: TChatMessage): Boolean;
    procedure ContinueResponse(Sender: TObject);
    procedure RetryResponse(Sender: TObject);
    function MessageScrollLimit: Single;
    function SnapMessageThumbPosition(const Position: TPointF): TPointF;
    procedure UpdateLatestButton;
    procedure RememberMessageViewport;
    procedure RestoreMessageViewport;
    procedure ScrollBottom;
    procedure ReflowMessages;
    procedure SyncMessageHeights;
    procedure ShowStatus(const Text: string; const IsError: Boolean = False);
    procedure SetActivePage(const Settings: Boolean);
  public
    procedure MouseWheel(Shift: TShiftState; WheelDelta: Integer; var Handled: Boolean); override;
  protected
    procedure CreateHandle; override;
    procedure DoShow; override;
  end;

var
  FormMain: TFormMain;

implementation

uses
  System.Math, System.IOUtils, System.Diagnostics, Winapi.Windows, Winapi.Dwmapi,
  Winapi.ShellAPI, FMX.Platform, FMX.Platform.Win, FMX.DialogService.Async,
  LLMUI.Controls, QuickNN.Models.LLM.Types, LLMUI.Context.Types, LLMUI.Context;

{$R *.fmx}

type
  TMessageTrackAccess = class(TCustomTrack)
  public
    function ThumbRectAt(const Value: Single): TRectF;
  end;

function TMessageTrackAccess.ThumbRectAt(const Value: Single): TRectF;
begin
  Result := GetThumbRect(Value);
end;

function TFormMain.ActiveChat: TChatSession;
begin
  Result := nil;
  if FStore <> nil then
    Result := FStore.FindChat(FStore.ActiveChatId);
end;

procedure TFormMain.ApplyWindowColors;
var
  Window: HWND;
  Enabled: LongBool;
  Color: COLORREF;
begin
  if Handle = nil then
    Exit;
  Window := WindowHandleToPlatform(Handle).Wnd;
  if not IsWindow(Window) then
    Exit;
  Enabled := True;
  DwmSetWindowAttribute(Window, 20, @Enabled, SizeOf(Enabled));
  Color := RGB(15, 20, 22);
  DwmSetWindowAttribute(Window, 35, @Color, SizeOf(Color));
  Color := RGB(229, 237, 236);
  DwmSetWindowAttribute(Window, 36, @Color, SizeOf(Color));
  Color := RGB(34, 52, 51);
  DwmSetWindowAttribute(Window, 34, @Color, SizeOf(Color));
end;

procedure TFormMain.MouseWheel(Shift: TShiftState; WheelDelta: Integer; var Handled: Boolean);
var
  MouseService: IFMXMouseService;
  Target: IControl;
begin
  if Handled then
    Exit;
  Target := Captured;
  if (Target = nil) and TPlatformServices.Current.SupportsPlatformService(IFMXMouseService, MouseService) then
    Target := ObjectAtPoint(MouseService.GetMousePos);
  // Intercept before FMX dispatches to styled edits/tracks: their own handlers
  // increment values even when an OnMouseWheel handler has set Handled.
  if (Target <> nil) and ScrollAncestorByWheel(Target.GetObject, WheelDelta) then
  begin
    Handled := True;
    Exit;
  end;
  inherited;
end;

procedure TFormMain.CreateHandle;
begin
  inherited;
  ApplyWindowColors;
end;

procedure TFormMain.DoShow;
begin
  inherited;
  ApplyWindowColors;
end;

procedure TFormMain.FormCreate(Sender: TObject);
begin
  FFrames := TDictionary<string, TFrameMessage>.Create;
  FModelIds := TList<string>.Create;
  FStore := TChatStore.Create(DefaultStorePath);
  try
    FStore.Load;
  except
    on E: Exception do
    begin
      ShowMessage(
        'Не удалось загрузить историю. Файл не будет перезаписан.'
        + sLineBreak
        + FStore.Path
        + sLineBreak
        + E.Message
      );
      Application.Terminate;
      Exit;
    end;
  end;
  for var Combo in [ComboBoxModel, ComboBoxContext, ComboBoxPrecision] do
    ConfigureComboBox(Combo);
  FEngine := TInferenceWorker.Create;
  FFollowLatest := True;
  FReady := True;
  LabelStoragePath.Text := FStore.Path;
  if FStore.FindChat(FStore.ActiveChatId) = nil then
  begin
    if FStore.Chats.Count = 0 then
      FStore.NewChat
    else
      FStore.ActiveChatId := FStore.Chats[0].Id;
  end;
  RenderModels;
  if FStore.Models.Count > 0 then
    SelectModelEditor(FStore.Models[0].Id)
  else
    SelectModelEditor('');
  SelectChat(FStore.ActiveChatId);
  SetActivePage(False);
  FormResize(Self);
  TimerUI.Enabled := True;
  MarkDirty;
  if FStore.RecoveryNotice <> '' then
    ShowStatus(FStore.RecoveryNotice);
end;

procedure TFormMain.FormDestroy(Sender: TObject);
begin
  TimerUI.Enabled := False;
  if FReady then
    SaveState;
  FReady := False;
  FEngine.Free;
  FFrames.Free;
  FModelIds.Free;
  FStore.Free;
end;

procedure TFormMain.FormCloseQuery(Sender: TObject; var CanClose: Boolean);
begin
  if FReady then
    EditChatTitleExit(Self);
  CanClose := not FReady or (FJobId = '');
  if not CanClose then
  begin
    FClosing := True;
    FEngine.Cancel;
    ShowStatus('Завершение генерации…');
  end;
end;

procedure TFormMain.FormResize(Sender: TObject);
begin
  if not FReady then
    Exit;
  if ClientWidth < 1280 then
  begin
    LayoutThreads.Width := 240;
    LayoutChatSettings.Width := 300;
  end
  else
  begin
    LayoutThreads.Width := 300;
    LayoutChatSettings.Width := 326;
  end;
  LayoutComposerDockResize(Self);
  ReflowMessages;
end;

procedure TFormMain.SetActivePage(const Settings: Boolean);
begin
  if Settings then
  begin
    TabControlMain.ActiveTab := TabItemSettings;
    RenderModels;
  end
  else
  begin
    TabControlMain.ActiveTab := TabItemChats;
  end;
  UpdateLatestButton;
end;

procedure TFormMain.ButtonBackClick(Sender: TObject);
begin
  SetActivePage(False);
end;

procedure TFormMain.ButtonManageModelsClick(Sender: TObject);
begin
  SetActivePage(True);
end;

procedure TFormMain.ShowStatus(const Text: string; const IsError: Boolean);
begin
  LabelStatus.Text := Text.Replace(#13, ' ').Replace(#10, ' ');
  LabelStatus.Hint := Text;
  LabelStatus.ShowHint := True;
  if IsError then
    LabelStatus.TextSettings.FontColor := $FFFFB6A8
  else
    LabelStatus.TextSettings.FontColor := $FF8DA79E;
end;

procedure TFormMain.MarkDirty;
begin
  if not FReady then
    Exit;
  if not FDirty then
    FSaveAt := GetTickCount64 + 750;
  FDirty := True;
end;

procedure TFormMain.SaveState;
begin
  if not FReady or not FDirty then
    Exit;
  try
    FStore.Save;
    FDirty := False;
  except
    on E: Exception do
    begin
      ShowStatus('Не удалось сохранить историю: ' + E.Message, True);
      FSaveAt := GetTickCount64 + 5000;
    end;
  end;
end;

procedure TFormMain.RenderThreads;
var
  Frame: TFrameEntry;
  Query: string;
begin
  if not FReady then
    Exit;
  LayoutThreadItems.BeginUpdate;
  try
    ClearControls(LayoutThreadItems);
    Query := LowerCase(Trim(EditSearch.Text));
    for var Chat in FStore.Chats do
    begin
      if (Query <> '') and not LowerCase(Chat.Title + ' ' + Chat.Preview).Contains(Query) then
        Continue;
      Frame := TFrameEntry.Create(Self);
      Frame.Name := 'FrameChatEntry' + Chat.Id.Replace('{', '').Replace('}', '').Replace('-', '');
      Frame.Parent := LayoutThreadItems;
      Frame.Align := TAlignLayout.Top;
      Frame.Margins.Bottom := 6;
      Frame.SetEntry(
        Chat.Id,
        Chat.Title,
        Chat.Preview,
        FormatDateTime('hh:nn', Chat.Updated),
        Chat.Id = FStore.ActiveChatId
      );
      Frame.OnSelect := ThreadSelected;
    end;
  finally
    LayoutThreadItems.EndUpdate;
  end;
  LayoutThreadItems.Height := 0;
  if LayoutThreadItems.ChildrenCount > 0 then
    for var Child in LayoutThreadItems.Children do
      if Child is TControl then
        LayoutThreadItems.Height := LayoutThreadItems.Height + TControl(Child).Height + 6;
end;

procedure TFormMain.ThreadSelected(Sender: TObject);
begin
  SelectChat(TFrameEntry(Sender).EntryId);
end;

procedure TFormMain.SelectChat(const Id: string);
var
  Chat: TChatSession;
begin
  Chat := FStore.FindChat(Id);
  if Chat = nil then
    Exit;
  // Resizing and clearing the old controls must not overwrite the new chat's view.
  Inc(FUpdatingViewport);
  try
    FStore.ActiveChatId := Id;
    FFollowLatest := Chat.FollowLatest;
    FUpdating := True;
    try
      EditChatTitle.Text := Chat.Title;
      MemoComposer.Text := Chat.Draft;
    finally
      FUpdating := False;
    end;
    LoadPreferences;
    RenderThreads;
    RenderMessages(False);
    ReflowMessages;
    UpdateAvailability;
  finally
    Dec(FUpdatingViewport);
  end;
  MarkDirty;
end;

procedure TFormMain.ButtonNewChatClick(Sender: TObject);
var
  Chat, Previous: TChatSession;
begin
  Previous := ActiveChat;
  Chat := FStore.NewChat;
  if Previous <> nil then
    Chat.Preferences := Previous.Preferences;
  SelectChat(Chat.Id);
  SetActivePage(False);
  MemoComposer.SetFocus;
end;

procedure TFormMain.EditSearchChangeTracking(Sender: TObject);
begin
  RenderThreads;
end;

procedure TFormMain.RefreshModelsCombo;
var
  Current: string;
begin
  if ActiveChat = nil then
    Exit;
  Current := ActiveChat.Preferences.ModelId;
  FUpdating := True;
  try
    ComboBoxModel.Items.Clear;
    FModelIds.Clear;
    for var Model in FStore.Models do
    begin
      ComboBoxModel.Items.Add(Model.Name);
      FModelIds.Add(Model.Id);
    end;
    ConfigureComboBox(ComboBoxModel);
    ComboBoxModel.ItemIndex := FModelIds.IndexOf(Current);
  finally
    FUpdating := False
  end;
end;

procedure TFormMain.LoadPreferences;
var
  Chat: TChatSession;
begin
  Chat := ActiveChat;
  if Chat = nil then
    Exit;
  RefreshModelsCombo;
  FUpdating := True;
  try
    ComboBoxContext.ItemIndex := ComboBoxContext.Items.IndexOf(Chat.Preferences.Context.ToString);
    TrackBarTemperature.Value := Chat.Preferences.Generation.Temperature;
    NumberBoxMaxTokens.Value := Chat.Preferences.Generation.MaxNewTokens;
    NumberBoxHardMaxTokens.Value := Chat.Preferences.Generation.HardMaxNewTokens;
    SwitchAutoContinue.IsChecked := Chat.Preferences.Generation.AutoContinue;
    SwitchAutoCompact.IsChecked := Chat.Preferences.Compaction.Enabled;
    NumberBoxCompactThreshold.Value := Chat.Preferences.Compaction.ThresholdPercent;
    NumberBoxKeepRecentTurns.Value := Chat.Preferences.Compaction.KeepRecentTurns;
    NumberBoxSummaryTokens.Value := Chat.Preferences.Compaction.SummaryTokens;
    NumberBoxReplyReserve.Value := Chat.Preferences.Compaction.ReplyReserveTokens;
    MemoSystemPrompt.Text := Chat.Preferences.SystemPrompt;
    NumberBoxTopP.Value := Chat.Preferences.Generation.TopP;
    NumberBoxTopK.Value := Chat.Preferences.Generation.TopK;
    NumberBoxRepetition.Value := Chat.Preferences.Generation.RepetitionPenalty;
    NumberBoxSeed.Value := Chat.Preferences.Generation.Seed;
    CheckBoxThinking.IsChecked := Chat.Preferences.Generation.EnableThinking;
    SwitchStreaming.IsChecked := Chat.Preferences.StreamOutput;
    NumberBoxGPUCache.Value := Chat.Preferences.GPUCacheGiB;
    NumberBoxRAMCache.Value := Chat.Preferences.RAMCacheGiB;
    if Chat.Preferences.Int4Experts then
      ComboBoxPrecision.ItemIndex := 0
    else
      ComboBoxPrecision.ItemIndex := 1;
  finally
    FUpdating := False
  end;
  UpdateKVLabel;
end;

procedure TFormMain.PreferencesChanged(Sender: TObject);
var
  Chat: TChatSession;
  Model: TModelEntry;
begin
  if not FReady or FUpdating then
    Exit;
  Chat := ActiveChat;
  if Chat = nil then
    Exit;
  if (ComboBoxModel.ItemIndex >= 0) and (ComboBoxModel.ItemIndex < FModelIds.Count) then
    Chat.Preferences.ModelId := FModelIds[ComboBoxModel.ItemIndex];
  if ComboBoxContext.ItemIndex >= 0 then
    Chat.Preferences.Context := StrToIntDef(ComboBoxContext.Selected.Text, 2048);
  Chat.Preferences.Generation.Temperature := TrackBarTemperature.Value;
  Chat.Preferences.Generation.MaxNewTokens := Round(NumberBoxMaxTokens.Value);
  Chat.Preferences.Generation.HardMaxNewTokens :=
    Max(Chat.Preferences.Generation.MaxNewTokens, Round(NumberBoxHardMaxTokens.Value));
  Chat.Preferences.Generation.AutoContinue := SwitchAutoContinue.IsChecked;
  Chat.Preferences.Compaction.Enabled := SwitchAutoCompact.IsChecked;
  Chat.Preferences.Compaction.ThresholdPercent := Round(NumberBoxCompactThreshold.Value);
  Chat.Preferences.Compaction.KeepRecentTurns := Round(NumberBoxKeepRecentTurns.Value);
  Chat.Preferences.Compaction.SummaryTokens := Round(NumberBoxSummaryTokens.Value);
  Chat.Preferences.Compaction.ReplyReserveTokens := Round(NumberBoxReplyReserve.Value);
  FUpdating := True;
  try
    NumberBoxHardMaxTokens.Value := Chat.Preferences.Generation.HardMaxNewTokens
  finally
    FUpdating := False
  end;
  Chat.Preferences.Generation.TopP := NumberBoxTopP.Value;
  Chat.Preferences.Generation.TopK := Round(NumberBoxTopK.Value);
  Chat.Preferences.Generation.RepetitionPenalty := NumberBoxRepetition.Value;
  Chat.Preferences.Generation.Seed := UInt64(Round(NumberBoxSeed.Value));
  Chat.Preferences.SystemPrompt := MemoSystemPrompt.Text;
  Chat.Preferences.StreamOutput := SwitchStreaming.IsChecked;
  Chat.Preferences.GPUCacheGiB := Round(NumberBoxGPUCache.Value);
  Chat.Preferences.RAMCacheGiB := Round(NumberBoxRAMCache.Value);
  Chat.Preferences.Int4Experts := ComboBoxPrecision.ItemIndex = 0;
  Model := FStore.FindModel(Chat.Preferences.ModelId);
  if (Model <> nil) and Model.SupportsThinking then
    Chat.Preferences.Generation.EnableThinking := CheckBoxThinking.IsChecked
  else
    Chat.Preferences.Generation.EnableThinking := False;
  UpdateKVLabel;
  UpdateAvailability;
  MarkDirty;
end;

procedure TFormMain.UpdateContextMemory;
var
  Chat: TChatSession;
begin
  Chat := ActiveChat;
  if Chat = nil then
    Exit;
  if Chat.ContextMemory.Available then
  begin
    LabelContextMemory.Text :=
      Format(
      'Память: %d сообщений · сжатий %d',
      [Chat.ContextMemory.CoveredMessages, Chat.ContextMemory.Compactions]
    );
    LabelCompactionState.Text :=
      Format(
      '%d → %d токенов после сжатия. Полная переписка сохранена.',
      [Chat.ContextMemory.BeforeTokens, Chat.ContextMemory.AfterTokens]
    );
  end
  else
  begin
    LabelContextMemory.Text := 'Память: полная история';
    LabelCompactionState.Text :=
      'История ещё не сжималась. Полная переписка сохраняется на диске.';
  end;
end;

procedure TFormMain.UpdateKVLabel;
var
  Chat: TChatSession;
  Model: TModelEntry;
begin
  Chat := ActiveChat;
  if Chat = nil then
    Exit;
  UpdateContextMemory;
  LabelTemperatureValue.Text := Format('%.1f', [Chat.Preferences.Generation.Temperature]);
  Model := FStore.FindModel(Chat.Preferences.ModelId);
  if Model = nil then
  begin
    LabelKVEstimate.Text := 'Выберите доступную модель';
    LabelSelectedModel.Text := 'Выберите модель справа';
    CheckBoxThinking.Enabled := False;
    ComboBoxPrecision.Enabled := False;
  end
  else
  begin
    LabelKVEstimate.Text :=
      Format(
      'KV-кеш: %.0f MiB · максимум %d токенов',
      [Model.KVBytes(Chat.Preferences.Context) / 1048576.0, Model.MaxContext]
    );
    LabelSelectedModel.Text := Model.Name;
    if not Model.Available then
      LabelSelectedModel.Text := Model.Name + ' · недоступна';
    FUpdating := True;
    try
      CheckBoxThinking.Enabled := Model.SupportsThinking;
      CheckBoxThinking.IsChecked := Chat.Preferences.Generation.EnableThinking and Model.SupportsThinking;
      ComboBoxPrecision.Enabled := Model.ModelType = 'qwen3_moe';
    finally
      FUpdating := False
    end;
  end;
end;

procedure TFormMain.UpdateAvailability;
var
  Model: TModelEntry;
  Chat: TChatSession;
begin
  if not FReady then
    Exit;
  Chat := ActiveChat;
  Model := nil;
  if Chat <> nil then
    Model := FStore.FindModel(Chat.Preferences.ModelId);
  if FJobId <> '' then
  begin
    ButtonSend.Enabled := True;
    PathStopIcon.Visible := True;
    PathSendIcon.Visible := False;
    ButtonSend.Hint := 'Остановить генерацию';
  end
  else
  begin
    ButtonSend.Enabled := (Chat <> nil) and (Model <> nil) and Model.Available and (Trim(MemoComposer.Text) <> '');
    PathStopIcon.Visible := False;
    PathSendIcon.Visible := True;
    ButtonSend.Hint := 'Отправить · Enter';
{$IFNDEF WIN64}
    ButtonSend.Enabled := False;
    ButtonSend.Hint := 'Для генерации запустите версию Win64.';
{$ENDIF}
  end;
  VertScrollBoxPreferences.Enabled := (FJobId = '') or (FJobChatId <> FStore.ActiveChatId);
  ButtonUnloadModel.Enabled := FJobId = '';
  NumberBoxCompactThreshold.Enabled := SwitchAutoCompact.IsChecked;
  NumberBoxKeepRecentTurns.Enabled := SwitchAutoCompact.IsChecked;
  NumberBoxSummaryTokens.Enabled := SwitchAutoCompact.IsChecked;
  NumberBoxReplyReserve.Enabled := SwitchAutoCompact.IsChecked;
  EditChatTitle.Enabled := Chat <> nil;
  ButtonDeleteChat.Enabled := Chat <> nil;
  ButtonExportChat.Enabled := (Chat <> nil) and (Chat.Messages.Count > 0);
  ButtonWelcomeModels.Visible := FStore.Models.Count = 0;
  if Chat <> nil then
    for var Message in Chat.Messages do
    begin
      var Frame: TFrameMessage;
      if FFrames.TryGetValue(Message.Id, Frame) then
        Frame.UpdateActions(
          (FJobId = '') and (Message = Chat.Messages.Last) and (Message.Role = lrAssistant),
          CanContinueMessage(Chat, Message)
        );
    end;
end;

procedure TFormMain.ButtonTogglePreferencesClick(Sender: TObject);
begin
  LayoutChatSettings.Visible := not LayoutChatSettings.Visible;
  RectanglePreferencesDivider.Visible := LayoutChatSettings.Visible;

  ReflowMessages;
end;

procedure TFormMain.EditChatTitleExit(Sender: TObject);
begin
  if not FReady or FUpdating or (ActiveChat = nil) then
    Exit;
  ActiveChat.Title := Trim(EditChatTitle.Text);
  if ActiveChat.Title = '' then
    ActiveChat.Title := 'Новый чат';
  EditChatTitle.Text := ActiveChat.Title;
  RenderThreads;
  MarkDirty;
end;

procedure TFormMain.MemoComposerChangeTracking(Sender: TObject);
begin
  if not FReady or FUpdating then
    Exit;
  if ActiveChat <> nil then
    ActiveChat.Draft := MemoComposer.Text;
  MarkDirty;
  UpdateAvailability;
end;

procedure TFormMain.FormKeyDown(Sender: TObject; var Key: Word; var KeyChar: WideChar; Shift: TShiftState);
begin
  if (ssCtrl in Shift) and ((Key = Ord('N')) or (KeyChar = 'n') or (KeyChar = 'N')) then
  begin
    ButtonNewChatClick(Self);
    Key := 0;
    KeyChar := #0;
  end
  else if (Key = vkEscape) and (FJobId <> '') then
  begin
    FEngine.Cancel;
    ShowStatus('Остановка…');
    Key := 0;
  end;
end;

procedure TFormMain.MemoComposerKeyDown(Sender: TObject; var Key: Word; var KeyChar: WideChar; Shift: TShiftState);
begin
  if (Key = vkReturn) and not (ssCtrl in Shift) then
  begin
    Key := 0;
    KeyChar := #0;
    Send(False);
  end;
end;

procedure TFormMain.Send(const Retry: Boolean; const ContinueAnswer: Boolean);
var
  Chat: TChatSession;
  Model: TModelEntry;
  Message: TChatMessage;
  Request: TInferenceRequest;
  LastUser: Integer;
begin
  if not FReady or (FJobId <> '') then
    Exit;
  if SizeOf(Pointer) <> 8 then
  begin
    ShowStatus('Для генерации запустите версию Win64.', True);
    Exit
  end;
  Chat := ActiveChat;
  if Chat = nil then
    Exit;
  Model := FStore.FindModel(Chat.Preferences.ModelId);
  if (Model = nil) or not Model.Available then
  begin
    ShowStatus(
      'Выберите доступную модель. Каталог находится на вкладке «Модели».',
      True
    );
    Exit;
  end;
  if Chat.Preferences.Context > Model.MaxContext then
  begin
    ShowStatus(
      Format(
        'Для этой модели выберите контекст не больше %d токенов.',
        [Model.MaxContext]
      ),
      True
    );
    Exit;
  end;
  if ContinueAnswer then
  begin
    if (Chat.Messages.Count = 0) or not CanContinueMessage(Chat, Chat.Messages.Last) then
      Exit;
  end
  else if Retry then
  begin
    LastUser := -1;
    for var I := Chat.Messages.Count - 1 downto 0 do
      if Chat.Messages[I].Role = lrUser then
      begin
        LastUser := I;
        Break
      end;
    if LastUser < 0 then
      Exit;
    while Chat.Messages.Count > LastUser + 1 do
      Chat.Messages.Delete(Chat.Messages.Count - 1);
  end
  else
  begin
    if Trim(MemoComposer.Text) = '' then
      Exit;
    // Only the last reply can be resumed. Avoid storing a full prompt per old reply.
    for var Previous in Chat.Messages do
      Previous.Continuation := Default(TLLMContinuationState);
    Message := TChatMessage.Create;
    Message.Role := lrUser;
    Message.Text := MemoComposer.Text.Trim;
    Chat.Messages.Add(Message);
    if Chat.Title = 'Новый чат' then
    begin
      Chat.Title := ShortText(Message.Text, 44);
      EditChatTitle.Text := Chat.Title;
    end;
    Chat.Draft := '';
    FUpdating := True;
    try
      MemoComposer.Lines.Clear
    finally
      FUpdating := False
    end;
  end;
  Request := Default(TInferenceRequest);
  Request.JobId := NewId;
  Request.ChatId := Chat.Id;
  Request.ModelName := Model.Name;
  Request.Directory := Model.Directory;
  Request.TokenizerDirectory := Model.ResolvedTokenizer;
  Request.IsMoE := Model.ModelType = 'qwen3_moe';
  Request.Preferences := Chat.Preferences;
  Request.ContinueResponse := ContinueAnswer;
  Request.ContextMemory := Chat.ContextMemory;
  if ContinueAnswer then
  begin
    Message := Chat.Messages.Last;
    Request.Continuation := Message.Continuation;
    Request.History := Chat.ContextHistory(Message.Id);
    // A partially saved new suffix must not be paired with an old token checkpoint.
    Message.Continuation := Default(TLLMContinuationState);
  end
  else
  begin
    Request.Messages := Chat.ModelMessages;
    Request.History := Chat.ContextHistory;
    Message := TChatMessage.Create;
    Message.Role := lrAssistant;
    Message.ModelName := Model.Name;
    Message.ModelId := Model.Id;
    Chat.Messages.Add(Message);
  end;
  Message.ErrorText := '';
  Message.State := msStreaming;
  Request.MessageId := Message.Id;
  Chat.Updated := Now;
  FStore.Chats.Move(FStore.Chats.IndexOf(Chat), 0);
  if not FEngine.Submit(Request) then
  begin
    Message.State := msFailed;
    Message.ErrorText := 'Движок занят. Повторите запрос.';
  end
  else
  begin
    FJobId := Request.JobId;
    FJobChatId := Chat.Id;
    FJobMessageId := Message.Id;
    FLastPhase := '';
    ShowStatus('Подготовка…');
  end;
  RenderThreads;
  if ContinueAnswer then
  begin
    var Frame: TFrameMessage;
    if FFrames.TryGetValue(Message.Id, Frame) then
      Frame.SetMessage(Message, False);
    ReflowMessages;
  end
  else
    RenderMessages;
  UpdateAvailability;
  MarkDirty;
end;

function TFormMain.CanContinueMessage(const Chat: TChatSession; const Message: TChatMessage): Boolean;
{$IFDEF WIN64}
var
  Model: TModelEntry;
{$ENDIF}
begin
  Result := False;
{$IFDEF WIN64}
  if (FJobId <> '') or (Chat = nil) or (Message = nil) then
    Exit;
  if (Chat.Messages.Count = 0)
    or (Chat.Messages.Last <> Message)
    or (Message.Role <> lrAssistant)
    or not (Message.State in [msLimited, msStopped, msContextFull, msFailed])
    or not Message.Continuation.Available
    or (Message.Text <> Message.Continuation.Text)
    or (Message.ModelId <> Chat.Preferences.ModelId) then
    Exit;
  if (Length(Message.Continuation.Tokens) >= Chat.Preferences.Context)
    and not ContextCanCompact(
      Chat.ContextHistory(Message.Id),
      Chat.Preferences.SystemPrompt,
      Chat.Preferences.Compaction,
      Chat.ContextMemory) then
    Exit;
  Model := FStore.FindModel(Message.ModelId);
  Result := (Model <> nil) and Model.Available;
  if Result then
    Result :=
      Message.Continuation.ModelKey = ExpandFileName(Model.Directory) + #1 + ExpandFileName(Model.ResolvedTokenizer);
{$ENDIF}
end;

procedure TFormMain.ContinueResponse(Sender: TObject);
begin
  Send(False, True);
end;

procedure TFormMain.ButtonSendClick(Sender: TObject);
begin
  if FJobId <> '' then
  begin
    FEngine.Cancel;
    ShowStatus('Остановка…');
  end
  else
    Send(False);
end;

procedure TFormMain.RetryResponse(Sender: TObject);
begin
  Send(True);
end;

function TFormMain.MessageScrollLimit: Single;
begin
  Result := Max(0, VertScrollBoxMessages.ContentBounds.Height - VertScrollBoxMessages.ViewportRect.Height);
end;

function TFormMain.SnapMessageThumbPosition(const Position: TPointF): TPointF;
var
  Thumb: TThumb;
  Parent: TFmxObject;
  Track: TMessageTrackAccess;
  CurrentRect, FirstRect, LastRect: TRectF;
begin
  Result := Position;
  if (Captured = nil) or not (Captured.GetObject is TThumb) then
    Exit;
  Thumb := TThumb(Captured.GetObject);
  if not Thumb.IsPressed or (Thumb.Track = nil) or (Thumb.Track.Orientation <> TOrientation.Vertical) then
    Exit;
  Parent := Thumb.Parent;
  while (Parent <> nil) and (Parent <> VertScrollBoxMessages) do
    Parent := Parent.Parent;
  if Parent = nil then
    Exit;
  Track := TMessageTrackAccess(Thumb.Track);
  CurrentRect := Track.ThumbRectAt(Track.Value);
  FirstRect := Track.ThumbRectAt(Track.Min);
  LastRect := Track.ThumbRectAt(Track.Max - Track.ViewportSize);
  // FMX rounds thumb bounds to pixels, but converts the pointer position back
  // through an unrounded range. Equal visible endpoints must reach exact edges.
  if SameValue(FirstRect.Top, LastRect.Top, 0.1) then
    Exit;
  if SameValue(CurrentRect.Top, FirstRect.Top, 0.1) then
    Result.Y := 0
  else if SameValue(CurrentRect.Bottom, LastRect.Bottom, 0.1) then
    Result.Y := MessageScrollLimit;
end;

procedure TFormMain.UpdateLatestButton;
begin
  if not FReady then
    Exit;
  ButtonLatestMessage.Visible :=
    (TabControlMain.ActiveTab = TabItemChats)
    and VertScrollBoxMessages.Visible
    and (VertScrollBoxMessages.ViewportPosition.Y < MessageScrollLimit - 2);
end;

procedure TFormMain.RememberMessageViewport;
var
  Chat: TChatSession;
  Position: Double;
begin
  if not FReady or FRendering or (FUpdatingViewport > 0) then
    Exit;
  Chat := ActiveChat;
  if Chat = nil then
    Exit;
  Position := Max(0, VertScrollBoxMessages.ViewportPosition.Y);
  if (Chat.FollowLatest = FFollowLatest) and SameValue(Chat.ScrollPosition, Position, 0.1) then
    Exit;
  Chat.FollowLatest := FFollowLatest;
  Chat.ScrollPosition := Position;
  MarkDirty;
end;

procedure TFormMain.RestoreMessageViewport;
var
  Target: TPointF;
  Chat: TChatSession;
begin
  Inc(FUpdatingViewport);
  try
    Chat := ActiveChat;
    if FFollowLatest then
      Target := PointF(0, MessageScrollLimit)
    else if Chat <> nil then
      // Keep the saved destination while asynchronous Markdown layout grows.
      // Clamping this frame must not discard the chat's original reading position.
      Target := PointF(0, Single(EnsureRange(Chat.ScrollPosition, 0, Double(MessageScrollLimit))))
    else
      Target := PointF(0, 0);
    if not SameValue(VertScrollBoxMessages.ViewportPosition.Y, Target.Y, 0.1) then
      VertScrollBoxMessages.ViewportPosition := Target;
  finally
    Dec(FUpdatingViewport);
  end;
  UpdateLatestButton;
end;

procedure TFormMain.ScrollBottom;
begin
  FFollowLatest := True;
  RestoreMessageViewport;
  RememberMessageViewport;
end;

procedure TFormMain.ButtonLatestMessageClick(Sender: TObject);
begin
  ScrollBottom;
end;

procedure TFormMain.MessagesViewportPositionChange(Sender: TObject; const OldViewportPosition, NewViewportPosition: TPointF; const ContentSizeChanged: Boolean);
begin
  if not FReady then
    Exit;
  var Limit := MessageScrollLimit;
  var ExtentUnchanged := SameValue(Limit, FLastMessageScrollLimit, 0.1);
  FLastMessageScrollLimit := Limit;
  if FRendering or (FUpdatingViewport > 0) then
    Exit;
  var Position := SnapMessageThumbPosition(NewViewportPosition);
  if not SameValue(Position.Y, NewViewportPosition.Y, 0.1) then
  begin
    Inc(FUpdatingViewport);
    try
      VertScrollBoxMessages.ViewportPosition := Position;
    finally
      Dec(FUpdatingViewport);
    end;
  end;
  // FMX also flags ordinary scrolling as a content-position change.
  // Only a changed extent or our own layout transaction may clamp the view.
  if not ContentSizeChanged or ExtentUnchanged then
  begin
    if Position.Y < OldViewportPosition.Y - 0.1 then
      FFollowLatest := False
    else if Position.Y >= MessageScrollLimit - 2 then
      FFollowLatest := True;
    RememberMessageViewport;
  end;
  UpdateLatestButton;
end;

procedure TFormMain.ReflowMessages;
var
  Total: Single;
begin
  if not FReady or FRendering then
    Exit;
  Inc(FUpdatingViewport);
  try
    Total := 0;
    LayoutMessageItems.BeginUpdate;
    try
      if LayoutMessageItems.ChildrenCount > 0 then
        for var Child in LayoutMessageItems.Children do
          if Child is TFrameMessage then
          begin
            TFrameMessage(Child).Width := LayoutMessageItems.Width;
            TFrameMessage(Child).SyncContentHeight;
            Total := Total + TFrameMessage(Child).Height + TFrameMessage(Child).Margins.Bottom;
          end;
      LayoutMessageItems.Height := Total;
    finally
      LayoutMessageItems.EndUpdate;
    end;
    VertScrollBoxMessages.RealignContent;
    RestoreMessageViewport;
  finally
    Dec(FUpdatingViewport);
  end;
end;

procedure TFormMain.SyncMessageHeights;
var
  Changed: Boolean;
begin
  if not FReady or FRendering then
    Exit;
  Inc(FUpdatingViewport);
  try
    Changed := False;
    for var Frame in FFrames.Values do
      if Frame.SyncContentHeight then
        Changed := True;
    if Changed then
      ReflowMessages;
    RestoreMessageViewport;
  finally
    Dec(FUpdatingViewport);
  end;
end;

procedure TFormMain.LayoutComposerDockResize(Sender: TObject);
var
  AvailableWidth: Single;
begin
  if (PanelComposer = nil) or (LayoutComposerDock = nil) then
    Exit;
  AvailableWidth := LayoutComposerDock.Width;
  if (VertScrollBoxMessages <> nil)
    and VertScrollBoxMessages.Visible
    and (LayoutMessageItems <> nil)
    and (LayoutMessageItems.Width > 0) then
    AvailableWidth := Min(AvailableWidth, LayoutMessageItems.Width);
  // Match the message column's center, excluding the outer scrollbar lane.
  PanelComposer.Margins.Right := Max(0, LayoutComposerDock.Width - AvailableWidth);
  PanelComposer.Width := Min(AvailableWidth, CHAT_CONTENT_MAX_WIDTH);
  if LayoutLatestButtonDock <> nil then
  begin
    LayoutLatestButtonDock.Margins.Right := PanelComposer.Margins.Right;
    LayoutLatestButtonDock.Margins.Bottom := LayoutComposerDock.Height + LayoutStatus.Height + 12;
  end;
end;

procedure TFormMain.LayoutMessageItemsResize(Sender: TObject);
begin
  LayoutComposerDockResize(Sender);
end;

procedure TFormMain.MessagesResize(Sender: TObject);
begin
  ReflowMessages;
end;

procedure TFormMain.RenderMessages(const ScrollToEnd: Boolean);
var
  Chat: TChatSession;
  Frame: TFrameMessage;
  Total: Single;
begin
  if not FReady or FRendering then
    Exit;
  FRendering := True;
  try
    FFrames.Clear;
    LayoutMessageItems.BeginUpdate;
    try
      ClearControls(LayoutMessageItems);
      Chat := ActiveChat;
      LayoutWelcome.Visible := (Chat = nil) or (Chat.Messages.Count = 0);
      VertScrollBoxMessages.Visible := not LayoutWelcome.Visible;
      Total := 0;
      if Chat <> nil then
        for var Message in Chat.Messages do
        begin
          Frame := TFrameMessage.Create(Self);
          Frame.Name := 'FrameMessage' + Message.Id.Replace('{', '').Replace('}', '').Replace('-', '');
          Frame.Parent := LayoutMessageItems;
          Frame.Align := TAlignLayout.Top;
          Frame.Width := LayoutMessageItems.Width;
          Frame.Margins.Bottom := 14;
          Frame.SetMessage(
            Message,
            (FJobId = '') and (Message = Chat.Messages.Last) and (Message.Role = lrAssistant),
            CanContinueMessage(Chat, Message)
          );
          Frame.OnRetry := RetryResponse;
          Frame.OnContinue := ContinueResponse;
          FFrames.Add(Message.Id, Frame);
          Total := Total + Frame.Height + 14;
        end;
      LayoutMessageItems.Height := Total;
    finally
      LayoutMessageItems.EndUpdate
    end;
  finally
    FRendering := False
  end;
  if ScrollToEnd then
    ScrollBottom
  else
    UpdateLatestButton;
end;

procedure TFormMain.ButtonDeleteChatClick(Sender: TObject);
var
  Id: string;
begin
  if ActiveChat = nil then
    Exit;
  Id := ActiveChat.Id;
  TDialogServiceAsync.MessageDialog(
    'Удалить этот чат вместе с историей?',
    TMsgDlgType.mtConfirmation,
    [TMsgDlgBtn.mbYes, TMsgDlgBtn.mbNo],
    TMsgDlgBtn.mbNo,
    0,
    procedure(const Result: System.UITypes.TModalResult)
    begin
      if (Result <> mrYes) or not FReady then
        Exit;
      if FJobChatId = Id then
        FEngine.Cancel;
      FStore.RemoveChat(Id);
      if FStore.Chats.Count = 0 then
        FStore.NewChat;
      SelectChat(FStore.ActiveChatId);
      MarkDirty;
    end);
end;

procedure TFormMain.ButtonExportChatClick(Sender: TObject);
var
  Chat: TChatSession;
  Text: TStringBuilder;
begin
  Chat := ActiveChat;
  if Chat = nil then
    Exit;
  SaveDialogExport.FileName := Chat.Title;
  for var C in System.IOUtils.TPath.GetInvalidFileNameChars do
    SaveDialogExport.FileName := string(SaveDialogExport.FileName).Replace(C, '_');
  if not SaveDialogExport.Execute then
    Exit;
  Text := TStringBuilder.Create;
  try
    Text.AppendLine('# ' + Chat.Title).AppendLine;
    for var Message in Chat.Messages do
    begin
      if Message.Role = lrUser then
        Text.AppendLine('## Вы')
      else
        Text.AppendLine('## ' + Message.ModelName);
      Text.AppendLine.AppendLine(Message.Text).AppendLine;
      if Message.ErrorText <> '' then
        Text.AppendLine('> ' + Message.ErrorText).AppendLine;
    end;
    TFile.WriteAllText(SaveDialogExport.FileName, Text.ToString, TEncoding.UTF8);
    ShowStatus('Чат экспортирован');
  finally
    Text.Free
  end;
end;

procedure TFormMain.ButtonUnloadModelClick(Sender: TObject);
begin
  if FJobId <> '' then
    Exit;
  FEngine.Unload;
  ShowStatus('Выгрузка модели…');
end;

procedure TFormMain.ButtonSuggestionClick(Sender: TObject);
begin
  if Sender = ButtonSuggestionCode then
    MemoComposer.Text :=
      'Помоги написать понятную функцию Python с примером использования.'
  else
    MemoComposer.Text :=
      'Помоги превратить идею проекта в понятный план. Сначала задай три уточняющих вопроса.';
  MemoComposer.SetFocus;
end;

procedure TFormMain.TimerUITimer(Sender: TObject);
var
  Snapshot: TInferenceSnapshot;
  Chat: TChatSession;
  Message: TChatMessage;
  Frame: TFrameMessage;
begin
  if not FReady then
    Exit;
  Inc(FUpdatingViewport);
  try
    Snapshot := FEngine.Poll;
    if Snapshot.Phase <> FLastPhase then
    begin
      FLastPhase := Snapshot.Phase;
      if not FClosing then
        ShowStatus(Snapshot.Phase, Snapshot.ErrorText <> '');
    end;
    LabelMemory.Visible := Snapshot.GPUWeightLimit > 0;
    ButtonUnloadModel.Visible := Snapshot.LoadedModel <> '';
    if Snapshot.GPUWeightLimit > 0 then
      LabelMemory.Text :=
        Format(
        'GPU %.1f/%.1f GiB · KV %.0f MiB',
        [
          Snapshot.GPUWeightBytes / 1073741824.0,
          Snapshot.GPUWeightLimit / 1073741824.0,
          Snapshot.KVBytes / 1048576.0
        ]
      )
    else
      LabelMemory.Text := '';
    if (Snapshot.JobId = FJobId) and (FJobId <> '') then
    begin
      Chat := FStore.FindChat(FJobChatId);
      Message := nil;
      if Chat <> nil then
        for var Item in Chat.Messages do
          if Item.Id = FJobMessageId then
          begin
            Message := Item;
            Break
          end;
      if Message <> nil then
      begin
        if Snapshot.MemoryUpdated then
        begin
          Chat.ContextMemory := Snapshot.ContextMemory;
          if Chat.Id = FStore.ActiveChatId then
            UpdateContextMemory;
          MarkDirty;
        end;
        if Snapshot.Delta <> '' then
        begin
          Message.Text := Message.Text + Snapshot.Delta;
          if (Chat.Id = FStore.ActiveChatId)
            and Chat.Preferences.StreamOutput
            and FFrames.TryGetValue(Message.Id, Frame) then
          begin
            Frame.SetMessage(Message, False);
            ReflowMessages;
          end;
          MarkDirty;
        end;
        if Snapshot.Finished then
        begin
          if Snapshot.ErrorText <> '' then
          begin
            Message.State := msFailed;
            Message.ErrorText := Snapshot.ErrorText;
          end
          else
          begin
            if Snapshot.Result.Text <> '' then
              Message.Text := Snapshot.Result.Text;
            if Snapshot.Result.FinishReason = lfCancelled then
              Message.State := msStopped
            else if Snapshot.Result.FinishReason in [lfMaxTokens, lfHardLimit] then
              Message.State := msLimited
            else if Snapshot.Result.FinishReason = lfContextFull then
              Message.State := msContextFull
            else
              Message.State := msComplete;
          end;
          if Snapshot.Continuation.Available and (Snapshot.Continuation.Text = Message.Text) then
            Message.Continuation := Snapshot.Continuation
          else
            Message.Continuation := Default(TLLMContinuationState);
          Message.Tokens := Snapshot.Result.GeneratedTokens;
          Message.DecodeRate := Snapshot.Result.DecodeTokensPerSecond;
          Message.FirstTokenSeconds := Snapshot.Result.TimeToFirstTokenSeconds;
          Chat.Updated := Now;
        end;
      end;
      if Snapshot.Finished then
      begin
        FJobId := '';
        FJobChatId := '';
        FJobMessageId := '';
        RenderThreads;
        // Update just the visible last reply. A finished background chat must not
        // destroy another chat's Markdown selection or code-copy feedback.
        var VisibleChat := ActiveChat;
        if (VisibleChat <> nil)
          and (VisibleChat.Messages.Count > 0)
          and FFrames.TryGetValue(VisibleChat.Messages.Last.Id, Frame) then
          Frame.SetMessage(
            VisibleChat.Messages.Last,
            VisibleChat.Messages.Last.Role = lrAssistant,
            CanContinueMessage(VisibleChat, VisibleChat.Messages.Last)
          );
        ReflowMessages;
        UpdateAvailability;
        MarkDirty;
        if Snapshot.ErrorText <> '' then
          ShowStatus(Snapshot.ErrorText, True);
        if FClosing then
        begin
          Close;
          Exit
        end;
      end;
    end;
    SyncMessageHeights;
    if FDirty and (GetTickCount64 >= FSaveAt) then
      SaveState;
  finally
    RestoreMessageViewport;
    Dec(FUpdatingViewport);
  end;
end;

procedure TFormMain.RenderModels;
var
  Frame: TFrameEntry;
  Badge: string;
begin
  if not FReady then
    Exit;
  LayoutModelItems.BeginUpdate;
  try
    ClearControls(LayoutModelItems);
    for var Model in FStore.Models do
    begin
      Frame := TFrameEntry.Create(Self);
      Frame.Name := 'FrameModelEntry' + Model.Id.Replace('{', '').Replace('}', '').Replace('-', '');
      Frame.Parent := LayoutModelItems;
      Frame.Align := TAlignLayout.Top;
      Frame.Margins.Bottom := 8;
      Badge := '';
      if not Model.Available then
        Badge := '!'
      else if Model.ModelType = 'qwen3_moe' then
        Badge := 'MoE';
      Frame.SetEntry(Model.Id, Model.Name, Model.Directory, Badge, Model.Id = FEditingModelId);
      Frame.OnSelect := ModelSelected;
    end;
  finally
    LayoutModelItems.EndUpdate
  end;
  LayoutModelItems.Height := FStore.Models.Count * 86;
end;

procedure TFormMain.ModelSelected(Sender: TObject);
begin
  SelectModelEditor(TFrameEntry(Sender).EntryId);
end;

procedure TFormMain.SelectModelEditor(const Id: string);
var
  Model: TModelEntry;
begin
  FEditingModelId := Id;
  Model := FStore.FindModel(Id);
  if Model = nil then
  begin
    LabelModelEditorTitle.Text := 'Добавить модель';
    EditModelName.Text := '';
    EditModelDirectory.Text := '';
    EditTokenizerDirectory.Text := '';
    LabelModelValidation.Text := 'Поддерживаются Qwen3 и Qwen3 MoE в формате safetensors.';
  end
  else
  begin
    Model.Inspect;
    LabelModelEditorTitle.Text := Model.Name;
    EditModelName.Text := Model.Name;
    EditModelDirectory.Text := Model.Directory;
    EditTokenizerDirectory.Text := Model.TokenizerDirectory;
    LabelModelValidation.Text := Model.Validation;
  end;
  ButtonRemoveModel.Enabled := Model <> nil;
  RenderModels;
end;

procedure TFormMain.ButtonNewModelClick(Sender: TObject);
begin
  SelectModelEditor('');
  EditModelName.SetFocus;
end;

procedure TFormMain.ButtonBrowseModelClick(Sender: TObject);
var
  Directory: string;
begin
  Directory := EditModelDirectory.Text;
  if SelectDirectory('Каталог с config.json и весами модели', '', Directory) then
    EditModelDirectory.Text := Directory;
end;

procedure TFormMain.ButtonBrowseTokenizerClick(Sender: TObject);
var
  Directory: string;
begin
  Directory := EditTokenizerDirectory.Text;
  if SelectDirectory('Каталог с tokenizer.json', '', Directory) then
    EditTokenizerDirectory.Text := Directory;
end;

procedure TFormMain.ButtonSaveModelClick(Sender: TObject);
var
  Edited, Candidate: TModelEntry;
begin
  if Trim(EditModelName.Text) = '' then
  begin
    LabelModelValidation.Text := 'Укажите название модели.';
    Exit
  end;
  if Trim(EditModelDirectory.Text) = '' then
  begin
    LabelModelValidation.Text := 'Укажите каталог модели.';
    Exit
  end;
  for var Model in FStore.Models do
    if (Model.Id <> FEditingModelId) and SameText(Model.Name, Trim(EditModelName.Text)) then
    begin
      LabelModelValidation.Text := 'Модель с таким названием уже есть.';
      Exit
    end;
  Candidate := TModelEntry.Create;
  try
    try
      Candidate.Name := Trim(EditModelName.Text);
      Candidate.Directory :=
        ExcludeTrailingPathDelimiter(System.IOUtils.TPath.GetFullPath(Trim(EditModelDirectory.Text).Trim(['"'])));
      Candidate.TokenizerDirectory := Trim(EditTokenizerDirectory.Text).Trim(['"']);
      if Candidate.TokenizerDirectory <> '' then
        Candidate.TokenizerDirectory := System.IOUtils.TPath.GetFullPath(Candidate.TokenizerDirectory);
      Candidate.Inspect;
      if not Candidate.Available then
      begin
        LabelModelValidation.Text := Candidate.Validation;
        Exit;
      end;
      Edited := FStore.FindModel(FEditingModelId);
      if Edited = nil then
      begin
        Edited := Candidate;
        Candidate := nil;
        FStore.Models.Add(Edited);
      end
      else
      begin
        Edited.Name := Candidate.Name;
        Edited.Directory := Candidate.Directory;
        Edited.TokenizerDirectory := Candidate.TokenizerDirectory;
        Edited.Inspect;
      end;
      FEditingModelId := Edited.Id;
      if (ActiveChat <> nil) and (ActiveChat.Preferences.ModelId = '') then
        ActiveChat.Preferences.ModelId := Edited.Id;
      SelectModelEditor(Edited.Id);
      LoadPreferences;
      UpdateAvailability;
      MarkDirty;
      LabelModelValidation.Text := Edited.Validation + sLineBreak + 'Модель сохранена.';
    except
      on E: Exception do
        LabelModelValidation.Text := E.Message;
    end;
  finally
    Candidate.Free
  end;
end;

procedure TFormMain.ButtonRemoveModelClick(Sender: TObject);
var
  Id: string;
begin
  Id := FEditingModelId;
  if FStore.FindModel(Id) = nil then
    Exit;
  TDialogServiceAsync.MessageDialog(
    'Убрать модель из каталога? Файлы модели и история чатов сохранятся.',
    TMsgDlgType.mtConfirmation,
    [TMsgDlgBtn.mbYes, TMsgDlgBtn.mbNo],
    TMsgDlgBtn.mbNo,
    0,
    procedure(const Result: System.UITypes.TModalResult)
    begin
      if (Result <> mrYes) or not FReady then
        Exit;
      FStore.RemoveModel(Id);
      SelectModelEditor('');
      LoadPreferences;
      UpdateAvailability;
      MarkDirty;
    end);
end;

procedure TFormMain.ButtonOpenStorageClick(Sender: TObject);
begin
  TDirectory.CreateDirectory(ExtractFileDir(FStore.Path));
  ShellExecute(0, 'open', PChar(ExtractFileDir(FStore.Path)), nil, nil, SW_SHOWNORMAL);
end;

end.

