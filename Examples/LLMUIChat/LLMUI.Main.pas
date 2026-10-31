unit LLMUI.Main;

interface

uses
  System.SysUtils, System.Types, System.UITypes, System.Classes,
  System.Generics.Collections, FMX.Types, FMX.Controls, FMX.Forms, FMX.Graphics,
  FMX.Dialogs, FMX.Filter.Effects, FMX.Layouts, FMX.StdCtrls, FMX.Objects,
  FMX.TabControl, FMX.Edit, FMX.EditBox, FMX.NumberBox, FMX.ListBox,
  FMX.Memo.Types, FMX.TextLayout, FMX.ScrollBox, FMX.Memo,
  FMX.Controls.Presentation, LLMUI.Data, LLMUI.Engine, LLMUI.Images,
  LLMUI.Attachments, LLMUI.Attachment, LLMUI.Message, LLMUI.Entry,
  LLMUI.VirtualMessages, LLMUI.ImageViewer;

type
  TFormMain = class(TForm)
    FrameImageViewer: TFrameImageViewer;
    LayoutModelTokenizerSlot: TLayout;
    LayoutModelCheckpointSlot: TLayout;
    LayoutModelImageOptionsSlot: TLayout;
    LayoutModelCheckpoint: TLayout;
    LabelModelCheckpoint: TLabel;
    LayoutModelCheckpointInput: TLayout;
    EditModelCheckpoint: TEdit;
    ButtonBrowseCheckpoint: TButton;
    PathBrowseCheckpointIcon: TPath;
    OpenDialogModelCheckpoint: TOpenDialog;
    ComboBoxCatalogKind: TComboBox;
    LayoutImageModelDefaults: TLayout;
    LayoutImageModelKind: TLayout;
    LabelImageModelKind: TLabel;
    ComboBoxImageModelKind: TComboBox;
    CheckBoxImageModelInt8: TCheckBox;
    LayoutImageModelDimensions: TLayout;
    LayoutImageModelSampling: TLayout;
    LayoutImageModelWidth: TLayout;
    LabelImageModelWidth: TLabel;
    NumberBoxImageModelWidth: TNumberBox;
    LayoutImageModelHeight: TLayout;
    LabelImageModelHeight: TLabel;
    NumberBoxImageModelHeight: TNumberBox;
    LayoutImageModelSteps: TLayout;
    LabelImageModelSteps: TLabel;
    NumberBoxImageModelSteps: TNumberBox;
    LayoutImageModelSeed: TLayout;
    LabelImageModelSeed: TLabel;
    NumberBoxImageModelSeed: TNumberBox;
    LayoutImageModelGuidance: TLayout;
    LabelImageModelGuidance: TLabel;
    NumberBoxImageModelGuidance: TNumberBox;
    HorzScrollBoxAttachments: THorzScrollBox;
    LayoutDraftImages: TLayout;
    ButtonAttach: TButton;
    PathAttachIcon: TPath;
    OpenDialogImages: TOpenDialog;
    RectangleImageDrop: TRectangle;
    LabelImageDrop: TLabel;
    ComboBoxImageModel: TComboBox;
    LabelImageModel: TLabel;
    LayoutImageModel: TLayout;
    EditImageDirectory: TEdit;
    LabelImageDirectory: TLabel;
    LayoutImageDirectory: TLayout;
    ButtonBrowseImageModel: TButton;
    ComboBoxImageAnalysis: TComboBox;
    LabelImageAnalysis: TLabel;
    LayoutImageAnalysis: TLayout;
    NumberBoxImageWidth: TNumberBox;
    LabelImageWidth: TLabel;
    LayoutImageWidth: TLayout;
    NumberBoxImageHeight: TNumberBox;
    LabelImageHeight: TLabel;
    LayoutImageHeight: TLayout;
    NumberBoxImageSteps: TNumberBox;
    LabelImageSteps: TLabel;
    LayoutImageSteps: TLayout;
    NumberBoxImageSeed: TNumberBox;
    LabelImageSeed: TLabel;
    LayoutImageSeed: TLayout;
    NumberBoxImageGuidance: TNumberBox;
    LabelImageGuidance: TLabel;
    LayoutImageGuidance: TLayout;
    LayoutImagePreferences: TLayout;
    LabelImageWorkflow: TLabel;
    ComboBoxComposerMode: TComboBox;
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
    LayoutComposerActions: TLayout;
    LayoutComposerMode: TLayout;
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
    LayoutMessagesBefore: TLayout;
    LayoutVisibleMessages: TLayout;
    LayoutMessagesAfter: TLayout;
    TimerMessages: TTimer;
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
    SaveDialogExport: TSaveDialog;
    TimerUI: TTimer;
    procedure ButtonAttachClick(Sender: TObject);
    procedure ButtonBrowseImageModelClick(Sender: TObject);
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
    procedure ComposerModeChanged(Sender: TObject);
    procedure MemoComposerResize(Sender: TObject);
    procedure MemoComposerChangeTracking(Sender: TObject);
    procedure MemoComposerKeyDown(Sender: TObject; var Key: Word; var KeyChar: WideChar; Shift: TShiftState);
    procedure ButtonSendClick(Sender: TObject);
    procedure ButtonDeleteChatClick(Sender: TObject);
    procedure ButtonExportChatClick(Sender: TObject);
    procedure ButtonUnloadModelClick(Sender: TObject);
    procedure ButtonSuggestionClick(Sender: TObject);
    procedure ButtonNewModelClick(Sender: TObject);
    procedure ButtonBrowseCheckpointClick(Sender: TObject);
    procedure ImageCheckpointChanged(Sender: TObject);
    procedure CatalogKindChanged(Sender: TObject);
    procedure ImageModelKindChanged(Sender: TObject);
    procedure ButtonSaveModelClick(Sender: TObject);
    procedure ButtonRemoveModelClick(Sender: TObject);
    procedure ButtonBrowseModelClick(Sender: TObject);
    procedure ButtonBrowseTokenizerClick(Sender: TObject);
    procedure ButtonOpenStorageClick(Sender: TObject);
    procedure TimerUITimer(Sender: TObject);
    procedure TimerMessagesTimer(Sender: TObject);
    procedure MessagesResize(Sender: TObject);
    procedure ButtonLatestMessageClick(Sender: TObject);
    procedure MessagesViewportPositionChange(Sender: TObject; const OldViewportPosition, NewViewportPosition: TPointF; const ContentSizeChanged: Boolean);
    procedure LayoutComposerDockResize(Sender: TObject);
    procedure LayoutMessageItemsResize(Sender: TObject);
  private
    FStore: TChatStore;
    FEngine: TInferenceWorker;
    FFrames: TDictionary<string, TFrameMessage>;
    FVirtualStates: TObjectDictionary<string, TChatVirtualMessages>;
    FVirtualMessages: TChatVirtualMessages;
    FFirstVisibleMessage, FLastVisibleMessage: Integer;
    FVisionModelIds: TList<string>;
    FImageModelIds: TList<string>;
    FModelIds: TList<string>;
    FReady, FUpdating, FClosing, FDirty, FRendering: Boolean;
    FFollowLatest: Boolean;
    FLastMessageScrollLimit: Single;
    FUpdatingViewport: Integer;
    FSaveAt: UInt64;
    FReflowingMessages, FRestoreMessagePosition: Boolean;
    FMessageViewportHeight: Single;
    FMessageResizeAt: UInt64;
    FLastMessageWidth, FRequestedMessageWidth: Single;
    FEditingModelId, FJobId, FJobChatId, FJobMessageId: string;
    FLastPhase: string;
    FComposerTextLayout: TTextLayout;
    FUpdatingComposerHeight: Boolean;
    procedure UpdateComposerHeight;
    procedure RenderDraftImages;
    procedure RemoveDraftImage(Sender: TObject);
    procedure AddImportedImage(const Image: TChatImage);
    function AcceptImageDrop(const Data: TDragObject; const Point: TPointF): Boolean;
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
    procedure SaveImageModel;
    procedure UpdateImageEditor;
    procedure Send(const Retry: Boolean; const ContinueAnswer: Boolean = False);
    function CanContinueMessage(const Chat: TChatSession; const Message: TChatMessage): Boolean;
    procedure ContinueResponse(Sender: TObject);
    procedure RetryResponse(Sender: TObject);
    function MessageScrollLimit: Single;
    function SnapMessageThumbPosition(const Position: TPointF): TPointF;
    procedure UpdateLatestButton;
    procedure RememberMessageViewport;
    function TargetMessagePosition: Double;
    procedure SaveMessageAnchor(const Position: Double);
    procedure ReleaseMessageFrames;
    procedure RealizeMessageRange(const First, Last: Integer);
    procedure MessageHeightChanged(Sender: TObject);
    procedure MessageWindow(const Position, ViewHeight, Buffer: Double; out First, Last: Integer);
    procedure QueueMessageLayout;
    procedure RestoreMessageViewport;
    procedure ScrollBottom;
    procedure RequestMessageResize;
    procedure ReflowMessages;
    procedure SyncMessageHeights;
    procedure ShowStatus(const Text: string; const IsError: Boolean = False);
    procedure SetActivePage(const Settings: Boolean);
  public
    procedure AttachImageFiles(const Files: TArray<string>);
    function PasteImageAttachments: Boolean;
    procedure DragEnter(const Data: TDragObject; const Point: TPointF); override;
    procedure DragOver(const Data: TDragObject; const Point: TPointF; var Operation: TDragOperation); override;
    procedure DragDrop(const Data: TDragObject; const Point: TPointF); override;
    procedure DragLeave; override;
    procedure MouseWheel(Shift: TShiftState; WheelDelta: Integer; var Handled: Boolean); override;
  protected
    procedure CreateHandle; override;
    procedure DoShow; override;
  end;

var
  FormMain: TFormMain;

implementation

uses
  System.Math, System.IOUtils, System.Diagnostics, Winapi.Windows,
  Winapi.ShellAPI, FMX.Platform, FMX.Platform.Win, LLMUI.Theme, LLMUI.Dialogs,
  LLMUI.Controls, QuickNN.Models.LLM.Types, LLMUI.Context.Types, LLMUI.Context;

{$R *.fmx}

const
  COMPOSER_MIN_MEMO_HEIGHT = 50;
  COMPOSER_MAX_MEMO_HEIGHT = 280;
  COMPOSER_MAX_WINDOW_FRACTION = 0.4;

type
  // Lock only this container's alignment. TControl.BeginUpdate recursively
  // visits every message/style child and is too costly for the layout timer.
  TMessageLayoutAccess = class(TLayout)
  public
    property AlignDisabled: Boolean read FDisableAlign write FDisableAlign;
  end;

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
begin
  ApplyChatWindowColors(Self);
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
  // Stop following on the input gesture, before the first animation frame:
  // an incoming token must not pin an upward wheel gesture back to the tail.
  if FFollowLatest and (WheelDelta > 0) and (Target <> nil) then
  begin
    var Parent := Target.GetObject;
    while (Parent <> nil) and (Parent <> VertScrollBoxMessages) do
      Parent := Parent.Parent;
    if Parent <> nil then
    begin
      FFollowLatest := False;
      RememberMessageViewport;
    end;
  end;
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
  QueueMessageLayout;
end;

procedure TFormMain.FormCreate(Sender: TObject);
begin
  StyleBook := ChatStyleBook;
  FFrames := TDictionary<string, TFrameMessage>.Create;
  FVirtualStates := TObjectDictionary<string, TChatVirtualMessages>.Create([doOwnsValues]);
  FFirstVisibleMessage := -1;
  FLastVisibleMessage := -1;
  FModelIds := TList<string>.Create;
  FVisionModelIds := TList<string>.Create;
  FImageModelIds := TList<string>.Create;
  FStore := TChatStore.Create(DefaultStorePath);
  try
    FStore.Load;
  except
    on E: Exception do
    begin
      TChatDialogs.Message(Self, 'Не удалось загрузить историю',
        'Файл не будет перезаписан.'
        + sLineBreak
        + FStore.Path
        + sLineBreak
        + E.Message, True
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
  TimerMessages.Enabled := False;
  if FReady then
    SaveState;
  FReady := False;
  FEngine.Free;
  FFrames.Free;
  FVirtualStates.Free;
  FImageModelIds.Free;
  FVisionModelIds.Free;
  FModelIds.Free;
  FStore.Free;
  FComposerTextLayout.Free;
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
  RequestMessageResize;
  QueueMessageLayout;
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
    StopScrollMotion(VertScrollBoxMessages);
    FStore.ActiveChatId := Id;
    FFollowLatest := Chat.FollowLatest;
    FMessageResizeAt := 0;
    FUpdating := True;
    try
      EditChatTitle.Text := Chat.Title;
      MemoComposer.Text := Chat.Draft;
    finally
      FUpdating := False;
    end;
    LoadPreferences;
    RenderDraftImages;
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
    ComboBoxImageAnalysis.Items.Clear;
    FVisionModelIds.Clear;
    FImageModelIds.Clear;
    ComboBoxImageModel.Items.Clear;
    for var ImageModel in FStore.ImageModels do
    begin
      ComboBoxImageModel.Items.Add(ImageModel.Name);
      FImageModelIds.Add(ImageModel.Id);
    end;
    FModelIds.Clear;
    for var Model in FStore.Models do
    begin
      ComboBoxModel.Items.Add(Model.Name);
      if Model.Available and Model.SupportsVision then
      begin
        ComboBoxImageAnalysis.Items.Add(Model.Name);
        FVisionModelIds.Add(Model.Id)
      end;
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
    ComboBoxImageModel.ItemIndex := FImageModelIds.IndexOf(Chat.Preferences.ImageGeneration.ModelId);
    var ImageModel := FStore.FindImageModel(Chat.Preferences.ImageGeneration.ModelId);
    EditImageDirectory.Text := '';
    if ImageModel <> nil then
      EditImageDirectory.Text := ImageModel.Settings.Directory;
    NumberBoxImageWidth.VertIncrement := 16;
    NumberBoxImageHeight.VertIncrement := 16;
    if (ImageModel <> nil) and (ImageModel.Settings.Kind = ipQwenImage) then
    begin
      NumberBoxImageWidth.VertIncrement := 32;
      NumberBoxImageHeight.VertIncrement := 32;
    end;
    NumberBoxImageWidth.Value := Chat.Preferences.ImageGeneration.Width;
    NumberBoxImageHeight.Value := Chat.Preferences.ImageGeneration.Height;
    NumberBoxImageSteps.Value := Chat.Preferences.ImageGeneration.Steps;
    NumberBoxImageSeed.Value := Chat.Preferences.ImageGeneration.Seed;
    NumberBoxImageGuidance.Value := Chat.Preferences.ImageGeneration.Guidance;
    ComboBoxImageAnalysis.ItemIndex := FVisionModelIds.IndexOf(Chat.Preferences.ImageGeneration.AnalysisModelId);
    if (ComboBoxImageAnalysis.ItemIndex < 0) and (FVisionModelIds.Count > 0) then
      ComboBoxImageAnalysis.ItemIndex := 0;
    ConfigureComboBox(ComboBoxImageAnalysis);
    ConfigureComboBox(ComboBoxImageModel);
    ConfigureComboBox(ComboBoxComposerMode);
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
  if Sender = ComboBoxImageModel then
  begin
    if (ComboBoxImageModel.ItemIndex >= 0) and (ComboBoxImageModel.ItemIndex < FImageModelIds.Count) then
    begin
      var ImageModel := FStore.FindImageModel(FImageModelIds[ComboBoxImageModel.ItemIndex]);
      if ImageModel <> nil then
      begin
        var AnalysisId := Chat.Preferences.ImageGeneration.AnalysisModelId;
        Chat.Preferences.ImageGeneration := ImageModel.Resolve(ImageModel.Settings);
        Chat.Preferences.ImageGeneration.AnalysisModelId := AnalysisId;
        LoadPreferences;
      end;
    end
    else
      Chat.Preferences.ImageGeneration.ModelId := '';
  end;
  Chat.Preferences.ImageGeneration.Width := Round(NumberBoxImageWidth.Value) div 16 * 16;
  Chat.Preferences.ImageGeneration.Height := Round(NumberBoxImageHeight.Value) div 16 * 16;
  Chat.Preferences.ImageGeneration.Steps := Round(NumberBoxImageSteps.Value);
  Chat.Preferences.ImageGeneration.Seed := Round(NumberBoxImageSeed.Value);
  Chat.Preferences.ImageGeneration.Guidance := NumberBoxImageGuidance.Value;
  if (ComboBoxImageAnalysis.ItemIndex >= 0) and (ComboBoxImageAnalysis.ItemIndex < FVisionModelIds.Count) then
    Chat.Preferences.ImageGeneration.AnalysisModelId := FVisionModelIds[ComboBoxImageAnalysis.ItemIndex];
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
      ComboBoxPrecision.Enabled := (Model.ModelType = 'qwen3_moe') or (Model.ModelType = 'qwen3_5');
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
    var ModelReady := (Model <> nil) and Model.Available;
    if (Chat <> nil) and (ComboBoxComposerMode.ItemIndex = 1) and (Length(Chat.DraftImages) = 0) then
    begin
      var ImageModel := FStore.FindImageModel(Chat.Preferences.ImageGeneration.ModelId);
      ModelReady := (ImageModel <> nil) and ImageModel.Available;
    end;
    ButtonSend.Enabled := (Chat <> nil) and ModelReady and ((Trim(MemoComposer.Text) <> '') or (Length(Chat.DraftImages) > 0));
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
  ButtonAttach.Enabled := Chat <> nil;
  EditChatTitle.Enabled := Chat <> nil;
  ButtonDeleteChat.Enabled := Chat <> nil;
  ButtonExportChat.Enabled := (Chat <> nil) and (Chat.Messages.Count > 0);
  ButtonWelcomeModels.Visible := FStore.Models.Count = 0;
  if (Chat <> nil) and (FVirtualMessages <> nil) then
    for var Pair in FFrames do
    begin
      var Index := FVirtualMessages.IndexOf(Pair.Key);
      if Index < 0 then
        Continue;
      var Message := FVirtualMessages.MessageAt(Index);
      Pair.Value.UpdateActions(
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

procedure TFormMain.ComposerModeChanged(Sender: TObject);
begin
  if FReady and not FUpdating then
    UpdateAvailability;
end;

procedure TFormMain.MemoComposerResize(Sender: TObject);
begin
  // FMX may still be aligning the memo's presentation. Fit after this layout pass.
  QueueMessageLayout;
end;

procedure TFormMain.MemoComposerChangeTracking(Sender: TObject);
begin
  if not FReady or FUpdating then
    Exit;
  if ActiveChat <> nil then
    ActiveChat.Draft := MemoComposer.Text;
  UpdateComposerHeight;
  MarkDirty;
  UpdateAvailability;
end;

procedure TFormMain.FormKeyDown(Sender: TObject; var Key: Word; var KeyChar: WideChar; Shift: TShiftState);
begin
  if (TabControlMain.ActiveTab = TabItemChats) and ((Focused = nil) or ButtonAttach.IsFocused)
    and (Key = Ord('V')) and (ssCtrl in Shift) and PasteImageAttachments then
  begin
    Key := 0;
    KeyChar := #0;
    Exit
  end;
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
  if (((Key = Ord('V')) and (ssCtrl in Shift)) or ((Key = vkInsert) and (ssShift in Shift)))
    and PasteImageAttachments then
  begin
    Key := 0;
    KeyChar := #0;
    Exit
  end;
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
  ImageModel: TImageModelEntry;
  ImageRequest: Boolean;
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
  if ContinueAnswer and (Chat.Messages.Count > 0) and ((Length(Chat.Messages.Last.Images) > 0)
    or (Chat.Messages.Last.Continuation.VisionSignature <> '')) then
    Model := FStore.FindModel(Chat.Messages.Last.ModelId);
  var HasUpload := not ContinueAnswer and (Length(Chat.DraftImages) > 0);
  if Retry then
    for var I := Chat.Messages.Count - 1 downto 0 do
      if Chat.Messages[I].Role = lrUser then
      begin
        HasUpload := Length(Chat.Messages[I].Images) > 0;
        Break
      end;
  ImageRequest := (ComboBoxComposerMode.ItemIndex = 1) and not ContinueAnswer and not HasUpload;
  if Retry and not HasUpload and (Chat.Messages.Count > 0) then
  begin
    var Last := Chat.Messages.Last;
    if (Last.Role = lrAssistant) and (Last.Text = '') and (Length(Last.Images) > 0)
      and not Last.Images[0].IsUserImage then
      ImageRequest := True;
  end;
  if not ImageRequest then
  begin
    if (Model = nil) or not Model.Available then
    begin
      ShowStatus(
        'Выберите доступную модель в каталоге настроек.',
        True
      );
      Exit;
    end;
  end;
  ImageModel := nil;
  if ImageRequest then
  begin
    ImageModel := FStore.FindImageModel(Chat.Preferences.ImageGeneration.ModelId);
    if ImageModel <> nil then
      ImageModel.Inspect;
    if (ImageModel = nil) or not ImageModel.Available then
    begin
      var Reason := 'Добавьте модель в каталоге настроек и выберите её в настройках чата.';
      if ImageModel <> nil then
        Reason := ImageModel.Validation;
      ShowStatus(Reason, True);
      Exit;
    end;
    try
      ValidateImageSettings(ImageModel.Resolve(Chat.Preferences.ImageGeneration));
    except
      on E: Exception do
      begin
        ShowStatus(E.Message, True);
        Exit
      end;
    end;
  end;
  if HasUpload and not ContinueAnswer then
  begin
    var AnalysisIndex := ComboBoxImageAnalysis.ItemIndex;
    if (AnalysisIndex < 0) or (AnalysisIndex >= FVisionModelIds.Count) then
    begin
      ShowStatus('Выберите модель анализа изображений в настройках чата.', True);
      Exit
    end;
    Model := FStore.FindModel(FVisionModelIds[AnalysisIndex]);
    if (Model = nil) or not Model.Available or (Model.ModelType <> 'qwen3_5') then
    begin
      ShowStatus('Для анализа требуется доступная Qwen hybrid vision-модель.', True);
      Exit
    end;
  end;
  if not ImageRequest and (Chat.Preferences.Context > Model.MaxContext) then
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
  if HasUpload and (Chat.Preferences.ModelId <> Model.Id) then
  begin
    Chat.Preferences.ModelId := Model.Id;
    FUpdating := True;
    try
      ComboBoxModel.ItemIndex := FModelIds.IndexOf(Model.Id)
    finally
      FUpdating := False
    end;
    UpdateKVLabel;
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
    if (Trim(MemoComposer.Text) = '') and (Length(Chat.DraftImages) = 0) then
      Exit;
    // Only the last reply can be resumed. Avoid storing a full prompt per old reply.
    for var Previous in Chat.Messages do
      Previous.Continuation := Default(TLLMContinuationState);
    Message := TChatMessage.Create;
    Message.Role := lrUser;
    Message.Text := MemoComposer.Text.Trim;
    Message.Images := Copy(Chat.DraftImages);
    if Message.Text = '' then
      Message.Text := 'Кратко опиши содержимое прикреплённых изображений.';
    Chat.DraftImages := nil;
    RenderDraftImages;
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
  if not ImageRequest then
  begin
    Request.ModelName := Model.Name;
    Request.Directory := Model.Directory;
    Request.TokenizerDirectory := Model.ResolvedTokenizer;
    Request.IsMoE := Model.ModelType = 'qwen3_moe';
  end;
  Request.Preferences := Chat.Preferences;
  Request.ContinueResponse := ContinueAnswer;
  Request.ContextMemory := Chat.ContextMemory;
  Request.GenerateImage := ImageRequest;
  if Request.GenerateImage then
  begin
    Request.Image := Default(TChatImage);
    Request.Image.Settings := ImageModel.Resolve(Chat.Preferences.ImageGeneration);
    Request.Image.Prompt := Chat.Messages.Last.Text;
    Request.Image.ModelName := ImageModel.Name;
    Request.Image.RelativePath := 'images/' + Chat.Id + '/' + NewId + '.png';
  end;
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
    if ImageRequest then
      Message.ModelName := ImageModel.Name
    else
    begin
      Message.ModelName := Model.Name;
      Message.ModelId := Model.Id;
    end;
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
    if Request.GenerateImage then
      ComboBoxComposerMode.ItemIndex := 0;
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
    or ((Length(Message.Images) = 0) and (Message.Continuation.VisionSignature = '')
    and (Message.ModelId <> Chat.Preferences.ModelId)) then
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

procedure TFormMain.SaveMessageAnchor(const Position: Double);
begin
  var Chat := ActiveChat;
  if (Chat = nil) or (FVirtualMessages = nil) or (FVirtualMessages.Count = 0) then
    Exit;
  var Index := FVirtualMessages.FindRow(Position);
  var Id := FVirtualMessages.Metrics(Index).Id;
  var Offset := Max(0, Position - FVirtualMessages.Top(Index));
  if (Chat.ScrollMessageId <> Id) or not SameValue(Chat.ScrollMessageOffset, Offset, 0.1) then
  begin
    Chat.ScrollMessageId := Id;
    Chat.ScrollMessageOffset := Offset;
    MarkDirty;
  end;
end;

function TFormMain.TargetMessagePosition: Double;
begin
  Result := 0;
  var Chat := ActiveChat;
  if (Chat = nil) or (FVirtualMessages = nil) then
    Exit;
  if FFollowLatest then
    Result := FVirtualMessages.TotalHeight - VertScrollBoxMessages.ViewportRect.Height
  else
  begin
    var Index := FVirtualMessages.IndexOf(Chat.ScrollMessageId);
    if Index < 0 then
      Result := Chat.ScrollPosition
    else
      Result := FVirtualMessages.Top(Index)
        + Min(Chat.ScrollMessageOffset, Max(0, FVirtualMessages.Metrics(Index).Height - 1));
  end;
  Result := EnsureRange(Result, 0, Max(0, FVirtualMessages.TotalHeight - VertScrollBoxMessages.ViewportRect.Height));
end;

procedure TFormMain.RememberMessageViewport;
begin
  if not FReady or FRendering or (FUpdatingViewport > 0) then
    Exit;
  var Chat := ActiveChat;
  if Chat = nil then
    Exit;
  var Position := Double(Max(0, VertScrollBoxMessages.ViewportPosition.Y));
  if (Chat.FollowLatest = FFollowLatest) and SameValue(Chat.ScrollPosition, Position, 0.1) then
    Exit;
  Chat.FollowLatest := FFollowLatest;
  Chat.ScrollPosition := Position;
  SaveMessageAnchor(Position);
  MarkDirty;
end;

procedure TFormMain.RestoreMessageViewport;
begin
  Inc(FUpdatingViewport);
  try
    var Target := Single(EnsureRange(TargetMessagePosition, 0, Double(MessageScrollLimit)));
    if not SameValue(VertScrollBoxMessages.ViewportPosition.Y, Target, 0.1) then
      MoveScrollViewport(VertScrollBoxMessages, PointF(0, Target));
  finally
    Dec(FUpdatingViewport);
  end;
  UpdateLatestButton;
end;

procedure TFormMain.QueueMessageLayout;
begin
  if FReady and not FRendering then
    TimerMessages.Enabled := True;
end;

procedure TFormMain.TimerMessagesTimer(Sender: TObject);
begin
  TimerMessages.Enabled := False;
  UpdateComposerHeight;
  ReflowMessages;
end;

procedure TFormMain.ScrollBottom;
begin
  StopScrollMotion(VertScrollBoxMessages);
  FFollowLatest := True;
  ReflowMessages;
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
  QueueMessageLayout;
end;

procedure TFormMain.RequestMessageResize;
begin
  if not FReady or FRendering or FReflowingMessages then
    Exit;
  var RequestedWidth := Min(LayoutMessageItems.Width, CHAT_CONTENT_MAX_WIDTH);
  if not SameValue(FRequestedMessageWidth, RequestedWidth, 0.1) then
  begin
    FRequestedMessageWidth := RequestedWidth;
    if not SameValue(FLastMessageWidth, RequestedWidth, 0.1) then
      FMessageResizeAt := GetTickCount64 + 100
    else
      FMessageResizeAt := 0;
  end;
end;

procedure TFormMain.ReleaseMessageFrames;
begin
  for var Pair in FFrames do
  begin
    if FVirtualMessages <> nil then
    begin
      var Index := FVirtualMessages.IndexOf(Pair.Key);
      if Index >= 0 then
        FVirtualMessages.Measure(Index, Pair.Value.LayoutRow.Width,
          Pair.Value.Height + MESSAGE_GAP, Pair.Value.ExpanderThinking.IsExpanded);
    end;
    Pair.Value.OnHeightChanged := nil;
    Pair.Value.OnRetry := nil;
    Pair.Value.OnContinue := nil;
    RetireControl(Pair.Value);
  end;
  FFrames.Clear;
  FFirstVisibleMessage := -1;
  FLastVisibleMessage := -1;
end;

procedure TFormMain.RealizeMessageRange(const First, Last: Integer);
begin
  for var Id in FFrames.Keys.ToArray do
  begin
    var Index := FVirtualMessages.IndexOf(Id);
    if (Index < First) or (Index > Last) then
    begin
      var Frame := FFrames[Id];
      if Index >= 0 then
        FVirtualMessages.Measure(Index, Frame.LayoutRow.Width, Frame.Height + MESSAGE_GAP,
          Frame.ExpanderThinking.IsExpanded);
      FFrames.Remove(Id);
      Frame.OnHeightChanged := nil;
      Frame.OnRetry := nil;
      Frame.OnContinue := nil;
      RetireControl(Frame);
    end;
  end;
  var Chat := ActiveChat;
  for var I := First to Last do
  begin
    var Message := FVirtualMessages.MessageAt(I);
    var Metric := FVirtualMessages.Metrics(I);
    var Frame: TFrameMessage;
    if not FFrames.TryGetValue(Message.Id, Frame) then
    begin
      Frame := TFrameMessage.Create(Self);
      Frame.Name := 'FrameMessage' + Message.Id.Replace('{', '').Replace('}', '').Replace('-', '');
      Frame.Parent := LayoutVisibleMessages;
      Frame.Align := TAlignLayout.Top;
      Frame.Width := LayoutMessageItems.Width;
      Frame.Margins.Bottom := MESSAGE_GAP;
      Frame.SetMessage(Message,
        (FJobId = '') and (Message = Chat.Messages.Last) and (Message.Role = lrAssistant),
        CanContinueMessage(Chat, Message));
      Frame.ExpanderThinking.IsExpanded := Metric.ThinkingExpanded;
      Frame.OnHeightChanged := MessageHeightChanged;
      Frame.OnRetry := RetryResponse;
      Frame.OnContinue := ContinueResponse;
      FFrames.Add(Message.Id, Frame);
    end;
    // Align=Top owns placement. Seed its sorting coordinate when the realized
    // window changes, so new rows are inserted before or after retained rows.
    Frame.Position.Y := FVirtualMessages.Top(I) - FVirtualMessages.Top(First);
    if Frame.Index <> I - First then
      Frame.Index := I - First;
  end;
  FFirstVisibleMessage := First;
  FLastVisibleMessage := Last;
end;

procedure TFormMain.MessageHeightChanged(Sender: TObject);
begin
  // Correct the outer viewport in the same event as an expander/Markdown
  // height change, before FMX can paint the temporarily shifted children.
  ReflowMessages;
end;

procedure TFormMain.MessageWindow(const Position, ViewHeight, Buffer: Double; out First, Last: Integer);
begin
  // Refill an overscan window only near its inner edge. Recycling rows at
  // every boundary does expensive Markdown/style work on animation frames.
  First := FFirstVisibleMessage;
  Last := FLastVisibleMessage;
  if (First >= 0) and (Last >= First) and (Last < FVirtualMessages.Count)
    and ((First = 0) or (Position >= FVirtualMessages.Top(First) + Buffer * 0.25))
    and ((Last = FVirtualMessages.Count - 1)
    or (Position + ViewHeight <= FVirtualMessages.Top(Last + 1) - Buffer * 0.25)) then
    Exit;
  First := FVirtualMessages.FindRow(Max(0, Position - Buffer));
  Last := FVirtualMessages.FindRow(Position + ViewHeight + Buffer);
end;

procedure TFormMain.ReflowMessages;
var
  RootLayout, RowLayout: TMessageLayoutAccess;
  RootWasLocked, RowsWereLocked, AllowResize, GeometryChanged: Boolean;
  First, Last, NewFirst, NewLast: Integer;
  Position, Buffer, ViewHeight: Double;
begin
  if not FReady or FRendering or FReflowingMessages or (FVirtualMessages = nil) then
    Exit;
  FReflowingMessages := True;
  Inc(FUpdatingViewport);
  RootLayout := TMessageLayoutAccess(LayoutMessageItems);
  RowLayout := TMessageLayoutAccess(LayoutVisibleMessages);
  RootWasLocked := RootLayout.AlignDisabled;
  RowsWereLocked := RowLayout.AlignDisabled;
  RootLayout.AlignDisabled := True;
  RowLayout.AlignDisabled := True;
  GeometryChanged := not SameValue(FMessageViewportHeight,
    VertScrollBoxMessages.ViewportRect.Height, 0.1);
  FMessageViewportHeight := VertScrollBoxMessages.ViewportRect.Height;
  try
    // Native animation can advance before its next position notification.
    // The persisted reading anchor is for restoring a chat, not for rewinding
    // live movement to the previous notification on every UI timer tick.
    if not FRestoreMessagePosition and not FFollowLatest then
      SaveMessageAnchor(VertScrollBoxMessages.ViewportPosition.Y);
    AllowResize := GetTickCount64 >= FMessageResizeAt;
    var ContentWidth := Min(LayoutMessageItems.Width, CHAT_CONTENT_MAX_WIDTH);
    if AllowResize then
    begin
      if not SameValue(FVirtualMessages.Width, ContentWidth, 0.1) then
      begin
        FVirtualMessages.Bind(ActiveChat, ContentWidth);
        GeometryChanged := True;
      end;
      FLastMessageWidth := ContentWidth;
      FMessageResizeAt := 0;
    end;
    ViewHeight := Max(1, VertScrollBoxMessages.ViewportRect.Height);
    Buffer := Max(400, ViewHeight);
    Position := TargetMessagePosition;
    MessageWindow(Position, ViewHeight, Buffer, First, Last);
    if First >= 0 then
      for var Pass := 1 to 3 do
      begin
        if (First <> FFirstVisibleMessage) or (Last <> FLastVisibleMessage) then
        begin
          RealizeMessageRange(First, Last);
          GeometryChanged := True;
        end;
        for var Pair in FFrames do
        begin
          var Index := FVirtualMessages.IndexOf(Pair.Key);
          var Frame := Pair.Value;
          if Frame.SyncContentHeight(AllowResize) then
            GeometryChanged := True;
          if FVirtualMessages.Measure(Index, Frame.LayoutRow.Width, Frame.Height + MESSAGE_GAP,
            Frame.ExpanderThinking.IsExpanded) then
            GeometryChanged := True;
        end;
        Position := TargetMessagePosition;
        MessageWindow(Position, ViewHeight, Buffer, NewFirst, NewLast);
        if (NewFirst = First) and (NewLast = Last) then
          Break;
        First := NewFirst;
        Last := NewLast;
        if Pass = 3 then
          QueueMessageLayout;
      end;
    if FVirtualMessages.Count = 0 then
    begin
      if FFrames.Count > 0 then
        ReleaseMessageFrames;
      LayoutMessagesBefore.Height := 0;
      LayoutMessagesAfter.Height := 0;
    end
    else
    begin
      LayoutMessagesBefore.Height := FVirtualMessages.Top(FFirstVisibleMessage);
      LayoutMessagesAfter.Height := FVirtualMessages.TotalHeight
        - FVirtualMessages.Top(FLastVisibleMessage + 1);
    end;
    GeometryChanged := GeometryChanged
      or not SameValue(LayoutMessageItems.Height, FVirtualMessages.TotalHeight, 0.1);
    LayoutMessageItems.Height := FVirtualMessages.TotalHeight;
  finally
    RowLayout.AlignDisabled := RowsWereLocked;
    RootLayout.AlignDisabled := RootWasLocked;
    try
      // Pure movement inside a measured window needs no content invalidation
      // or viewport assignment: FMX owns the animation between layout changes.
      if GeometryChanged then
      begin
        if not RootWasLocked then
          RootLayout.Realign;
        if not RowsWereLocked then
          RowLayout.Realign;
        VertScrollBoxMessages.RealignContent;
      end;
      if GeometryChanged or FRestoreMessagePosition or FFollowLatest then
        RestoreMessageViewport;
      var Chat := ActiveChat;
      var GeometryReady := Visible and SameValue(MessageScrollLimit,
        Max(0, FVirtualMessages.TotalHeight - VertScrollBoxMessages.ViewportRect.Height), 1);
      if GeometryReady then
        FRestoreMessagePosition := False;
      if GeometryReady and (Chat <> nil) and not FFollowLatest and (FVirtualMessages.Count > 0) then
        SaveMessageAnchor(VertScrollBoxMessages.ViewportPosition.Y);
      if GeometryReady and (Chat <> nil) and not SameValue(Chat.ScrollPosition, VertScrollBoxMessages.ViewportPosition.Y, 0.1) then
      begin
        Chat.ScrollPosition := VertScrollBoxMessages.ViewportPosition.Y;
        MarkDirty;
      end;
    finally
      Dec(FUpdatingViewport);
      FReflowingMessages := False;
    end;
  end;
end;

procedure TFormMain.SyncMessageHeights;
begin
  // One guarded pass handles pending widths and completed asynchronous layouts.
  ReflowMessages;
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
  UpdateComposerHeight;
end;

procedure TFormMain.LayoutMessageItemsResize(Sender: TObject);
begin
  LayoutComposerDockResize(Sender);
  RequestMessageResize;
end;

procedure TFormMain.MessagesResize(Sender: TObject);
begin
  RequestMessageResize;
end;

procedure TFormMain.RenderMessages(const ScrollToEnd: Boolean);
begin
  if not FReady or FRendering then
    Exit;
  FRendering := True;
  FRestoreMessagePosition := True;
  try
    ReleaseMessageFrames;
    var Chat := ActiveChat;
    LayoutWelcome.Visible := (Chat = nil) or (Chat.Messages.Count = 0);
    VertScrollBoxMessages.Visible := not LayoutWelcome.Visible;
    FVirtualMessages := nil;
    if Chat <> nil then
    begin
      if not FVirtualStates.TryGetValue(Chat.Id, FVirtualMessages) then
      begin
        FVirtualMessages := TChatVirtualMessages.Create;
        FVirtualStates.Add(Chat.Id, FVirtualMessages);
      end;
      FVirtualMessages.Bind(Chat, Min(LayoutMessageItems.Width, CHAT_CONTENT_MAX_WIDTH));
      if ScrollToEnd then
      begin
        FFollowLatest := True;
        Chat.FollowLatest := True;
      end;
    end;
  finally
    FRendering := False;
  end;
  ReflowMessages;
end;

procedure TFormMain.ButtonDeleteChatClick(Sender: TObject);
var
  Id: string;
begin
  if ActiveChat = nil then
    Exit;
  Id := ActiveChat.Id;
  if not TChatDialogs.Confirm(Self, 'Удалить чат?', 'Чат и его история будут удалены.', 'Удалить') or not FReady then
    Exit;
  if FJobChatId = Id then
    FEngine.Cancel;
  FStore.RemoveChat(Id);
  if FStore.Chats.Count = 0 then
    FStore.NewChat;
  SelectChat(FStore.ActiveChatId);
  FVirtualStates.Remove(Id);
  MarkDirty;
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
        for var Observation in Snapshot.ImageDescriptions do
          for var Source in Chat.Messages do
            if Source.Id = Observation.MessageId then
              for var I := 0 to High(Source.Images) do
                if Source.Images[I].ContentHash = Observation.ContentHash then
                begin
                  Source.Images[I].Description := Observation.Text;
                  MarkDirty
                end;
        if Snapshot.ImageReady then
        begin
          Message.Images := [Snapshot.Image];
          if FFrames.TryGetValue(Message.Id, Frame) then
            Frame.SetMessage(Message, False);
          if (Chat.Id = FStore.ActiveChatId) and (FVirtualMessages <> nil) then
            FVirtualMessages.MessageChanged(Message.Id, FFrames.ContainsKey(Message.Id));
          FStore.Save; // Persist the completed PNG independently of any later LLM request.
        end;
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
          end;
          if (Chat.Id = FStore.ActiveChatId) and (FVirtualMessages <> nil) then
            FVirtualMessages.MessageChanged(Message.Id, FFrames.ContainsKey(Message.Id));
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
          if (Chat.Id = FStore.ActiveChatId) and (FVirtualMessages <> nil) then
            FVirtualMessages.MessageChanged(Message.Id, FFrames.ContainsKey(Message.Id));
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
    Dec(FUpdatingViewport);
  end;
end;

procedure TFormMain.RenderModels;

  procedure AddEntry(const Id, Name, Directory, Badge: string);
  begin
    var Frame := TFrameEntry.Create(Self);
    Frame.Name := 'FrameModelEntry' + Id.Replace('{', '').Replace('}', '').Replace('-', '');
    Frame.Parent := LayoutModelItems;
    Frame.Align := TAlignLayout.Top;
    Frame.Margins.Bottom := 8;
    Frame.SetEntry(Id, Name, Directory, Badge, Id = FEditingModelId);
    Frame.OnSelect := ModelSelected;
  end;

begin
  if not FReady then
    Exit;
  LayoutModelItems.BeginUpdate;
  try
    ClearControls(LayoutModelItems);
    if ComboBoxCatalogKind.ItemIndex = 1 then
    begin
      for var Model in FStore.ImageModels do
      begin
        var Badge := '';
        if not Model.Available then
          Badge := '!';
        AddEntry(Model.Id, Model.Name, Model.Settings.Directory, Badge);
      end;
      LayoutModelItems.Height := FStore.ImageModels.Count * 86;
    end
    else
    begin
      for var Model in FStore.Models do
      begin
        var Badge := '';
        if not Model.Available then
          Badge := '!'
        else if Model.ModelType = 'qwen3_moe' then
          Badge := 'MoE';
        AddEntry(Model.Id, Model.Name, Model.Directory, Badge);
      end;
      LayoutModelItems.Height := FStore.Models.Count * 86;
    end;
  finally
    LayoutModelItems.EndUpdate
  end;
end;

procedure TFormMain.CatalogKindChanged(Sender: TObject);
begin
  if not FReady or FUpdating then
    Exit;
  var Id := '';
  if ComboBoxCatalogKind.ItemIndex = 1 then
  begin
    if FStore.ImageModels.Count > 0 then
      Id := FStore.ImageModels[0].Id;
  end
  else if FStore.Models.Count > 0 then
    Id := FStore.Models[0].Id;
  SelectModelEditor(Id);
end;

procedure TFormMain.UpdateImageEditor;
begin
  var IsImage := ComboBoxCatalogKind.ItemIndex = 1;
  var IsQwen := IsImage and (ComboBoxImageModelKind.ItemIndex = Ord(ipQwenImage));
  LayoutModelEditorFields.BeginUpdate;
  try
    LayoutModelTokenizerSlot.Height := 0;
    if not IsImage or IsQwen then
      LayoutModelTokenizerSlot.Height := 124;
    LayoutModelCheckpointSlot.Height := 0;
    if IsQwen then
      LayoutModelCheckpointSlot.Height := 94;
    LayoutModelImageOptionsSlot.Height := 0;
    if IsImage then
      LayoutModelImageOptionsSlot.Height := 252;
    if IsQwen then
      LayoutModelImageOptionsSlot.Height := 292;
    LayoutImageModelDefaults.Visible := IsImage;
    LayoutTokenizerDirectory.Visible := not IsImage or IsQwen;
    LayoutModelCheckpoint.Visible := IsQwen;
    CheckBoxImageModelInt8.Text := 'INT8 ConvRot (выключено: BF16)';
    if IsQwen and (EditModelCheckpoint.Text.Trim <> '') then
      CheckBoxImageModelInt8.Text := 'INT8 ConvRot для text encoder (выключено: BF16)';
    CheckBoxImageModelInt8.Visible := IsQwen;
    LabelModelDirectory.Text := 'Каталог модели';
    LabelTokenizerDirectory.Text := 'Каталог токенизатора';
    LabelTokenizerHint.Text := 'Укажите отдельно, если tokenizer.json хранится в другой папке.';
    if IsImage then
    begin
      LabelModelDirectory.Text := 'Каталог модели (transformer, text_encoder, vae, tokenizer)';
      LabelTokenizerDirectory.Text := 'Конфигурации и токенизатор Qwen-Image';
      LabelTokenizerHint.Text := 'tokenizer.json, encoder_config.json, vae_config.json, scheduler_config.json';
      if IsQwen then
      begin
        LabelModelDirectory.Text := 'Каталог весов (diffusion_models, text_encoders, vae)';
        if EditModelCheckpoint.Text.Trim <> '' then
          LabelModelDirectory.Text := 'Каталог text encoder и VAE (text_encoders, vae)';
      end;
    end;
    LayoutModelEditorFields.Height := 776;
    if IsImage then
      LayoutModelEditorFields.Height := 1076;
    if IsQwen then
      LayoutModelEditorFields.Height := 1170;
    NumberBoxImageModelWidth.VertIncrement := 16;
    NumberBoxImageModelHeight.VertIncrement := 16;
    if IsQwen then
    begin
      NumberBoxImageModelWidth.VertIncrement := 32;
      NumberBoxImageModelHeight.VertIncrement := 32;
    end;
  finally
    LayoutModelEditorFields.EndUpdate
  end;
end;

procedure TFormMain.ImageModelKindChanged(Sender: TObject);
begin
  if not FReady or FUpdating then
    Exit;
  case ComboBoxImageModelKind.ItemIndex of
    0:
      begin
        NumberBoxImageModelSteps.Value := 4;
        NumberBoxImageModelGuidance.Value := 1
      end;
    1:
      begin
        NumberBoxImageModelSteps.Value := 20;
        NumberBoxImageModelGuidance.Value := 4;
        if EditTokenizerDirectory.Text.Trim = '' then
          EditTokenizerDirectory.Text := FindQwenImageAssets;
      end;
    2:
      begin
        NumberBoxImageModelSteps.Value := 9;
        NumberBoxImageModelGuidance.Value := 0
      end;
  end;
  UpdateImageEditor;
end;

procedure TFormMain.ModelSelected(Sender: TObject);
begin
  SelectModelEditor(TFrameEntry(Sender).EntryId);
end;

procedure TFormMain.SelectModelEditor(const Id: string);
begin
  FEditingModelId := Id;
  FUpdating := True;
  try
    LabelModelEditorTitle.Text := 'Добавить модель';
    EditModelName.Text := '';
    EditModelDirectory.Text := '';
    EditTokenizerDirectory.Text := '';
    EditModelCheckpoint.Text := '';
    ButtonRemoveModel.Enabled := False;
    if ComboBoxCatalogKind.ItemIndex = 1 then
    begin
      var Settings := TChatImageSettings.Defaults;
      var Model := FStore.FindImageModel(Id);
      LabelModelValidation.Text := 'Выберите поддерживаемую архитектуру и каталог весов. Параметры по умолчанию можно изменить в каждом чате.';
      if Model <> nil then
      begin
        Model.Inspect;
        Settings := Model.Settings;
        LabelModelEditorTitle.Text := Model.Name;
        EditModelName.Text := Model.Name;
        LabelModelValidation.Text := Model.Validation;
        ButtonRemoveModel.Enabled := True;
      end;
      EditModelDirectory.Text := Settings.Directory;
      EditTokenizerDirectory.Text := Settings.AssetsDirectory;
      EditModelCheckpoint.Text := Settings.DiffusionCheckpoint;
      ComboBoxImageModelKind.ItemIndex := Ord(Settings.Kind);
      CheckBoxImageModelInt8.IsChecked := Settings.UseInt8;
      NumberBoxImageModelWidth.Value := Settings.Width;
      NumberBoxImageModelHeight.Value := Settings.Height;
      NumberBoxImageModelSteps.Value := Settings.Steps;
      NumberBoxImageModelSeed.Value := Settings.Seed;
      NumberBoxImageModelGuidance.Value := Settings.Guidance;
    end
    else
    begin
      var Model := FStore.FindModel(Id);
      LabelModelValidation.Text := 'Поддерживаются Qwen3, Qwen3 MoE и Qwen hybrid в формате safetensors.';
      if Model <> nil then
      begin
        Model.Inspect;
        LabelModelEditorTitle.Text := Model.Name;
        EditModelName.Text := Model.Name;
        EditModelDirectory.Text := Model.Directory;
        EditTokenizerDirectory.Text := Model.TokenizerDirectory;
        LabelModelValidation.Text := Model.Validation;
        ButtonRemoveModel.Enabled := True;
      end;
    end;
    UpdateImageEditor;
    ConfigureComboBox(ComboBoxCatalogKind);
    ConfigureComboBox(ComboBoxImageModelKind);
  finally
    FUpdating := False
  end;
  RenderModels;
end;

procedure TFormMain.ButtonNewModelClick(Sender: TObject);
begin
  SelectModelEditor('');
  EditModelName.SetFocus;
end;

procedure TFormMain.UpdateComposerHeight;
var
  Insets: TRectF;
  TextWidth, ImageHeight, ChromeHeight, MaxMemoHeight, DesiredHeight: Single;
  Content: TFmxObject;
  Value: string;
begin
  if not FReady or FUpdatingComposerHeight or (MemoComposer = nil)
    or (MemoComposer.Width <= 0) then
    Exit;
  FUpdatingComposerHeight := True;
  try
    // Measure the full draft, including offscreen wrapped lines and a trailing
    // empty line. Memo.ContentBounds can still contain estimates for those lines.
    Insets := TRectF.Create(0, 6, 0, 6);
    Content := MemoComposer.FindStyleResource('content');
    if Content is TControl then
      Insets := TControl(Content).Margins.Rect;
    TextWidth := Max(1, MemoComposer.Width - MemoComposer.Padding.Left - MemoComposer.Padding.Right
      - Insets.Left - Insets.Right - 2);
    Value := MemoComposer.Text + ' ';
    if FComposerTextLayout = nil then
      FComposerTextLayout := TTextLayoutManager.DefaultTextLayout.Create;
    if (FComposerTextLayout.Text <> Value)
      or not SameValue(FComposerTextLayout.MaxSize.X, TextWidth, 0.1)
      or (FComposerTextLayout.Font.Family <> MemoComposer.TextSettings.Font.Family)
      or not SameValue(FComposerTextLayout.Font.Size, MemoComposer.TextSettings.Font.Size, 0.1)
      or (FComposerTextLayout.Font.Style <> MemoComposer.TextSettings.Font.Style) then
    begin
      FComposerTextLayout.BeginUpdate;
      try
        FComposerTextLayout.Font.Assign(MemoComposer.TextSettings.Font);
        FComposerTextLayout.WordWrap := True;
        FComposerTextLayout.HorizontalAlign := MemoComposer.TextSettings.HorzAlign;
        FComposerTextLayout.VerticalAlign := TTextAlign.Leading;
        FComposerTextLayout.Trimming := TTextTrimming.None;
        FComposerTextLayout.MaxSize := PointF(TextWidth, 1000000);
        FComposerTextLayout.Text := Value;
      finally
        FComposerTextLayout.EndUpdate;
      end;
    end;
    ImageHeight := 0;
    if HorzScrollBoxAttachments.Visible then
      ImageHeight := HorzScrollBoxAttachments.Height + HorzScrollBoxAttachments.Margins.Top
        + HorzScrollBoxAttachments.Margins.Bottom;
    ChromeHeight := LayoutComposer.Padding.Top + LayoutComposer.Padding.Bottom
      + LayoutComposerFooter.Height + LayoutComposerFooter.Margins.Top + LayoutComposerFooter.Margins.Bottom;
    MaxMemoHeight := Max(COMPOSER_MIN_MEMO_HEIGHT, Min(COMPOSER_MAX_MEMO_HEIGHT,
        Floor(ClientHeight * COMPOSER_MAX_WINDOW_FRACTION - ChromeHeight - ImageHeight)));
    DesiredHeight := EnsureRange(Ceil(FComposerTextLayout.TextHeight + Insets.Top + Insets.Bottom
        + MemoComposer.Padding.Top + MemoComposer.Padding.Bottom + 2),
      COMPOSER_MIN_MEMO_HEIGHT, MaxMemoHeight) + ChromeHeight + ImageHeight;
    if not SameValue(LayoutComposerDock.Height, DesiredHeight, 0.1) then
    begin
      // Resizing the viewport must not turn following back on for a reader.
      RememberMessageViewport;
      Inc(FUpdatingViewport);
      try
        LayoutComposerDock.Height := DesiredHeight;
      finally
        Dec(FUpdatingViewport);
      end;
      QueueMessageLayout;
    end;
  finally
    FUpdatingComposerHeight := False;
  end;
end;

procedure TFormMain.RenderDraftImages;
begin
  if LayoutDraftImages.ChildrenCount > 0 then
    for var Child in LayoutDraftImages.Children do
      if Child is TFrameAttachment then
        TFrameAttachment(Child).OnRemove := nil;
  ClearControls(LayoutDraftImages);
  var Chat := ActiveChat;
  var Count := 0;
  if Chat <> nil then
    Count := Length(Chat.DraftImages);
  HorzScrollBoxAttachments.Visible := Count > 0;
  LayoutDraftImages.Width := Count * 154;
  if Chat <> nil then
    for var I := 0 to High(Chat.DraftImages) do
    begin
      var Frame := TFrameAttachment.Create(Self);
      Frame.Name := 'FrameDraftImage' + I.ToString;
      Frame.Parent := LayoutDraftImages;
      Frame.Align := TAlignLayout.Left;
      Frame.Margins.Right := 6;
      Frame.SetImages(Chat.DraftImages, I);
      Frame.OnRemove := RemoveDraftImage;
    end;
  LayoutComposerDockResize(Self);
  QueueMessageLayout;
end;

procedure TFormMain.RemoveDraftImage(Sender: TObject);
begin
  var Chat := ActiveChat;
  if (Chat = nil) or not (Sender is TFrameAttachment) then
    Exit;
  var Hash := TFrameAttachment(Sender).ContentHash;
  var Remaining: TArray<TChatImage>;
  for var Image in Chat.DraftImages do
    if Image.ContentHash <> Hash then
      Remaining := Remaining + [Image];
  Chat.DraftImages := Remaining;
  RenderDraftImages;
  UpdateAvailability;
  MarkDirty;
end;

procedure TFormMain.AddImportedImage(const Image: TChatImage);
begin
  var Chat := ActiveChat;
  if Chat = nil then
    Exit;
  for var Existing in Chat.DraftImages do
    if Existing.ContentHash = Image.ContentHash then
    begin
      if not SameFileName(Existing.FileName, Image.FileName) then
        TFile.Delete(Image.FileName);
      Exit;
    end;
  Chat.DraftImages := Chat.DraftImages + [Image];
  ComboBoxComposerMode.ItemIndex := 0;
end;

procedure TFormMain.AttachImageFiles(const Files: TArray<string>);
begin
  if not FReady or FClosing or (ActiveChat = nil) then
    Exit;
  try
    for var FileName in Files do
    begin
      if not IsImageFile(FileName) then
        Continue;
      if Length(ActiveChat.DraftImages) >= MAX_MESSAGE_IMAGES then
        raise EArgumentException.Create('К одному сообщению можно прикрепить до четырёх изображений.');
      AddImportedImage(ImportImageFile(ActiveChat.Id, FileName));
    end;
    ShowStatus('Изображения прикреплены. Можно добавить вопрос и отправить сообщение.');
  except
    on E: Exception do
      ShowStatus(E.Message, True)
  end;
  RenderDraftImages;
  UpdateAvailability;
  MarkDirty;
end;

procedure TFormMain.ButtonAttachClick(Sender: TObject);
begin
  if OpenDialogImages.Execute then
    AttachImageFiles(OpenDialogImages.Files.ToStringArray);
end;

function TFormMain.PasteImageAttachments: Boolean;
var
  Bitmap: FMX.Graphics.TBitmap;
begin
  Result := False;
  Bitmap := nil;
  if not FReady or FClosing or (ActiveChat = nil) then
    Exit;
  try
    var Files := ClipboardImageFiles;
    if Length(Files) > 0 then
    begin
      AttachImageFiles(Files);
      Exit(True)
    end;
    if not ReadClipboardImage(Bitmap) then
      Exit;
    Result := True;
    if Length(ActiveChat.DraftImages) >= MAX_MESSAGE_IMAGES then
      raise EArgumentException.Create('К одному сообщению можно прикрепить до четырёх изображений.');
    AddImportedImage(ImportImageBitmap(ActiveChat.Id, 'Из буфера обмена.png', Bitmap));
    RenderDraftImages;
    UpdateAvailability;
    MarkDirty;
    ShowStatus('Изображение из буфера прикреплено.');
  except
    on E: Exception do
    begin
      Result := True;
      ShowStatus(E.Message, True)
    end
  end;
  Bitmap.Free;
end;

function TFormMain.AcceptImageDrop(const Data: TDragObject; const Point: TPointF): Boolean;
begin
  Result := False;
  if FrameImageViewer.Visible or not FReady or FClosing or (ActiveChat = nil) or (TabControlMain.ActiveTab <> TabItemChats)
    or not LayoutConversation.AbsoluteRect.Contains(ScreenToClient(Point)) then
    Exit;
  for var FileName in Data.Files do
    if IsImageFile(FileName) then
      Exit(True);
end;

procedure TFormMain.DragEnter(const Data: TDragObject; const Point: TPointF);
begin
  if AcceptImageDrop(Data, Point) then
    RectangleImageDrop.Visible := True
  else
    inherited;
end;

procedure TFormMain.DragOver(const Data: TDragObject; const Point: TPointF; var Operation: TDragOperation);
begin
  RectangleImageDrop.Visible := AcceptImageDrop(Data, Point);
  if RectangleImageDrop.Visible then
    Operation := TDragOperation.Copy
  else
    inherited;
end;

procedure TFormMain.DragDrop(const Data: TDragObject; const Point: TPointF);
begin
  RectangleImageDrop.Visible := False;
  if AcceptImageDrop(Data, Point) then
    AttachImageFiles(Data.Files)
  else
    inherited;
end;

procedure TFormMain.DragLeave;
begin
  RectangleImageDrop.Visible := False;
  inherited;
end;

procedure TFormMain.ButtonBrowseImageModelClick(Sender: TObject);
begin
  SetActivePage(True);
  ComboBoxCatalogKind.ItemIndex := 1;
  CatalogKindChanged(ComboBoxCatalogKind);
end;

procedure TFormMain.ImageCheckpointChanged(Sender: TObject);
begin
  if FReady and not FUpdating then
    UpdateImageEditor;
end;

procedure TFormMain.ButtonBrowseCheckpointClick(Sender: TObject);
begin
  OpenDialogModelCheckpoint.FileName := EditModelCheckpoint.Text;
  if OpenDialogModelCheckpoint.Execute then
    EditModelCheckpoint.Text := OpenDialogModelCheckpoint.FileName;
end;

procedure TFormMain.ButtonBrowseModelClick(Sender: TObject);
var
  Directory: string;
begin
  Directory := EditModelDirectory.Text;
  if SelectDirectory('Каталог весов модели', '', Directory) then
    EditModelDirectory.Text := Directory;
end;

procedure TFormMain.ButtonBrowseTokenizerClick(Sender: TObject);
var
  Directory: string;
begin
  Directory := EditTokenizerDirectory.Text;
  if SelectDirectory('Каталог токенизатора и конфигураций', '', Directory) then
    EditTokenizerDirectory.Text := Directory;
end;

procedure TFormMain.SaveImageModel;
begin
  if EditModelName.Text.Trim = '' then
  begin
    LabelModelValidation.Text := 'Укажите название модели.';
    Exit
  end;
  for var Model in FStore.ImageModels do
    if (Model.Id <> FEditingModelId) and SameText(Model.Name, EditModelName.Text.Trim) then
    begin
      LabelModelValidation.Text := 'Модель с таким названием уже есть.';
      Exit
    end;
  var Candidate := TImageModelEntry.Create;
  try
    try
      Candidate.Name := EditModelName.Text.Trim;
      if EditModelDirectory.Text.Trim.Trim(['"']) = '' then
      begin
        LabelModelValidation.Text := 'Укажите каталог модели.';
        Exit
      end;
      Candidate.Settings.Directory := ExcludeTrailingPathDelimiter(TPath.GetFullPath(EditModelDirectory.Text.Trim.Trim(['"'])));
      Candidate.Settings.Kind := TImagePipelineKind(Max(0, ComboBoxImageModelKind.ItemIndex));
      Candidate.Settings.UseInt8 := CheckBoxImageModelInt8.IsChecked;
      if (Candidate.Settings.Kind = ipQwenImage) and (EditModelCheckpoint.Text.Trim.Trim(['"']) <> '') then
        Candidate.Settings.DiffusionCheckpoint := TPath.GetFullPath(EditModelCheckpoint.Text.Trim.Trim(['"']));
      if EditTokenizerDirectory.Text.Trim.Trim(['"']) <> '' then
        Candidate.Settings.AssetsDirectory := TPath.GetFullPath(EditTokenizerDirectory.Text.Trim.Trim(['"']));
      Candidate.Settings.Width := Round(NumberBoxImageModelWidth.Value);
      Candidate.Settings.Height := Round(NumberBoxImageModelHeight.Value);
      Candidate.Settings.Steps := Round(NumberBoxImageModelSteps.Value);
      Candidate.Settings.Seed := Round(NumberBoxImageModelSeed.Value);
      Candidate.Settings.Guidance := NumberBoxImageModelGuidance.Value;
      Candidate.Inspect;
      if not Candidate.Available then
      begin
        LabelModelValidation.Text := Candidate.Validation;
        Exit
      end;
      var Edited := FStore.FindImageModel(FEditingModelId);
      if Edited = nil then
      begin
        Edited := Candidate;
        Candidate := nil;
        FStore.ImageModels.Add(Edited);
      end
      else
      begin
        Edited.Name := Candidate.Name;
        Edited.Settings := Candidate.Settings;
        Edited.Inspect;
      end;
      if (ActiveChat <> nil) and (ActiveChat.Preferences.ImageGeneration.ModelId = '') then
      begin
        var AnalysisId := ActiveChat.Preferences.ImageGeneration.AnalysisModelId;
        ActiveChat.Preferences.ImageGeneration := Edited.Resolve(Edited.Settings);
        ActiveChat.Preferences.ImageGeneration.AnalysisModelId := AnalysisId;
      end;
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

procedure TFormMain.ButtonSaveModelClick(Sender: TObject);
var
  Edited, Candidate: TModelEntry;
begin
  if ComboBoxCatalogKind.ItemIndex = 1 then
  begin
    SaveImageModel;
    Exit;
  end;
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
  IsImage: Boolean;
begin
  Id := FEditingModelId;
  IsImage := ComboBoxCatalogKind.ItemIndex = 1;
  if (IsImage and (FStore.FindImageModel(Id) = nil))
    or (not IsImage and (FStore.FindModel(Id) = nil)) then
    Exit;
  if not TChatDialogs.Confirm(Self, 'Убрать модель?',
    'Модель будет удалена из каталога. Файлы модели и история чатов сохранятся.', 'Убрать') or not FReady then
    Exit;
  if IsImage then
    FStore.RemoveImageModel(Id)
  else
    FStore.RemoveModel(Id);
  SelectModelEditor('');
  LoadPreferences;
  UpdateAvailability;
  MarkDirty;
end;

procedure TFormMain.ButtonOpenStorageClick(Sender: TObject);
begin
  TDirectory.CreateDirectory(ExtractFileDir(FStore.Path));
  ShellExecute(0, 'open', PChar(ExtractFileDir(FStore.Path)), nil, nil, SW_SHOWNORMAL);
end;

end.

