unit LLMUI.Main;

interface

uses
  System.SysUtils, System.Types, System.UITypes, System.Classes,
  System.Generics.Collections, FMX.Types, FMX.Controls, FMX.Forms, FMX.Graphics,
  FMX.Dialogs, FMX.Menus, FMX.Filter.Effects, FMX.Layouts, FMX.StdCtrls,
  FMX.Objects, FMX.TabControl, FMX.Edit, FMX.EditBox, FMX.NumberBox, FMX.ListBox,
  FMX.Memo.Types, FMX.TextLayout, FMX.ScrollBox, FMX.Memo,
  FMX.Controls.Presentation, LLMUI.Data, LLMUI.Engine, LLMUI.Images,
  LLMUI.Attachments, LLMUI.Attachment, LLMUI.Message, LLMUI.Entry,
  LLMUI.VirtualMessages, LLMUI.ImageViewer, LLMUI.ModelDiscovery,
  LLMUI.RoleSection;

type
  TFormMain = class(TForm, IChatImageReferenceHost)
    PopupMenuImageReference: TPopupMenu;
    MenuItemUseImageReference: TMenuItem;
    MenuItemCopyAttachment: TMenuItem;
    FrameImageViewer: TFrameImageViewer;
    LayoutVisionContext: TLayout;
    LabelVisionContext: TLabel;
    NumberBoxVisionContext: TNumberBox;
    LayoutVisionMaxTokens: TLayout;
    LabelVisionMaxTokens: TLabel;
    NumberBoxVisionMaxTokens: TNumberBox;
    LayoutVisionTemperature: TLayout;
    LabelVisionTemperature: TLabel;
    NumberBoxVisionTemperature: TNumberBox;
    LayoutVisionInstruction: TLayout;
    LabelVisionInstruction: TLabel;
    MemoVisionInstruction: TMemo;
    ButtonBrowseModelFolder: TButton;
    FrameRoleSectionConversation: TFrameRoleSection;
    FrameRoleSectionVision: TFrameRoleSection;
    LabelVisionRoleHint: TLabel;
    FrameRoleSectionImages: TFrameRoleSection;
    ComboBoxImageAdapter: TComboBox;
    LabelWizardStep: TLabel;
    LayoutWizardHeader: TLayout;
    LayoutWizardFooter: TLayout;
    LayoutWizardNavigation: TLayout;
    LayoutWizardSource: TLayout;
    LabelWizardSourceHint: TLabel;
    LabelModelCapabilities: TLabel;
    LayoutWizardFiles: TLayout;
    LayoutModelRoot: TLayout;
    LabelModelRoot: TLabel;
    LayoutModelRootInput: TLayout;
    ButtonBrowseModelRoot: TButton;
    EditModelRoot: TEdit;
    LayoutModelTextEncoder: TLayout;
    LabelModelTextEncoder: TLabel;
    LayoutModelTextEncoderInput: TLayout;
    ButtonBrowseModelTextEncoder: TButton;
    EditModelTextEncoder: TEdit;
    LayoutModelTextEncoder2: TLayout;
    LabelModelTextEncoder2: TLabel;
    LayoutModelTextEncoder2Input: TLayout;
    ButtonBrowseModelTextEncoder2: TButton;
    EditModelTextEncoder2: TEdit;
    LayoutModelTokenizer: TLayout;
    LabelModelTokenizer: TLabel;
    LayoutModelTokenizerInput: TLayout;
    ButtonBrowseModelTokenizer: TButton;
    EditModelTokenizer: TEdit;
    LayoutModelTokenizer2: TLayout;
    LabelModelTokenizer2: TLabel;
    LayoutModelTokenizer2Input: TLayout;
    ButtonBrowseModelTokenizer2: TButton;
    EditModelTokenizer2: TEdit;
    LayoutModelVAE: TLayout;
    LabelModelVAE: TLabel;
    LayoutModelVAEInput: TLayout;
    ButtonBrowseModelVAE: TButton;
    EditModelVAE: TEdit;
    LayoutModelAdapters: TLayout;
    LabelModelAdapters: TLabel;
    ListBoxModelAdapters: TListBox;
    LayoutModelAdapterActions: TLayout;
    ButtonAddModelAdapter: TButton;
    ButtonRemoveModelAdapter: TButton;
    LayoutWizardDefaults: TLayout;
    LabelWizardSummary: TLabel;
    ButtonWizardNext: TButton;
    ButtonWizardBack: TButton;
    OpenDialogModelSource: TOpenDialog;
    OpenDialogModelComponent: TOpenDialog;
    OpenDialogModelAdapters: TOpenDialog;
    TimerModelInspect: TTimer;
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
    LayoutImageModelDefaults: TLayout;
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
    LayoutImageLoRA: TLayout;
    LabelImageLoRA: TLabel;
    EditImageLoRAFile: TEdit;
    LayoutImageLoRAActions: TLayout;
    ButtonBrowseImageLoRA: TButton;
    ButtonClearImageLoRA: TButton;
    LayoutImageLoRAStrength: TLayout;
    LabelImageLoRAStrength: TLabel;
    NumberBoxImageLoRAStrength: TNumberBox;
    OpenDialogImageLoRA: TOpenDialog;
    LayoutImagePreferences: TLayout;
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
    SpeedButtonNegativePrompt: TSpeedButton;
    PathSendIcon: TPath;
    PathStopIcon: TPath;
    MemoComposer: TMemo;
    LayoutStatus: TLayout;
    ButtonUnloadModel: TButton;
    PathUnloadModelIcon: TPath;
    LabelMemory: TLabel;
    LabelStatus: TLabel;
    LayoutStatusContent: TLayout;
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
    procedure BrowseModelFolder(Sender: TObject);
    procedure ModelSourceChanged(Sender: TObject);
    procedure ModelBundleChanged(Sender: TObject);
    procedure ModelInspectTimer(Sender: TObject);
    procedure WizardNextClick(Sender: TObject);
    procedure WizardBackClick(Sender: TObject);
    procedure WizardFilesChanged(Sender: TObject);
    procedure BrowseModelComponent(Sender: TObject);
    procedure AddModelAdapter(Sender: TObject);
    procedure RemoveModelAdapter(Sender: TObject);
    procedure ImageAdapterChanged(Sender: TObject);
    procedure RoleExpandedChanged(Sender: TObject);
    procedure ButtonAttachClick(Sender: TObject);
    procedure ButtonBrowseImageModelClick(Sender: TObject);
    procedure ImageReferenceMenuPopup(Sender: TObject);
    procedure UseImageReferenceClick(Sender: TObject);
    procedure CopyAttachmentClick(Sender: TObject);
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
    procedure NegativePromptClick(Sender: TObject);
    procedure MemoComposerResize(Sender: TObject);
    procedure MemoComposerChangeTracking(Sender: TObject);
    procedure MemoComposerKeyDown(Sender: TObject; var Key: Word; var KeyChar: WideChar; Shift: TShiftState);
    procedure ButtonSendClick(Sender: TObject);
    procedure ButtonDeleteChatClick(Sender: TObject);
    procedure ButtonExportChatClick(Sender: TObject);
    procedure ButtonUnloadModelClick(Sender: TObject);
    procedure ButtonSuggestionClick(Sender: TObject);
    procedure ButtonNewModelClick(Sender: TObject);
    procedure ButtonBrowseImageLoRAClick(Sender: TObject);
    procedure ButtonClearImageLoRAClick(Sender: TObject);
    procedure ButtonBrowseCheckpointClick(Sender: TObject);
    procedure ImageCheckpointChanged(Sender: TObject);
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
    FEditingNegativePrompt: Boolean;
    FReferenceDragPending, FReferenceDragActive: Boolean;
    FReferenceDragChatId, FReferenceDragPath: string;
    FReferenceDragTicket: TObject;
    FReferenceDragScrollAt: UInt64;
    FReferenceMenuImage: TChatImage;
    FReferenceMenuChatId: string;
    FReferenceMenuPosition: TPointF;
    FReferenceMenuPending: Boolean;
    FDetectedModel: TDetectedModel;
    FWizardStep: Integer;
    FAdapterFiles: TList<string>;
    FModelInspectPending: Boolean;
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
    procedure BeginDraftImageDrag(Sender: TObject);
    procedure StartDraftImageDrag;
    function ReferenceDropIndex(const Data: TDragObject; const Point: TPointF): Integer;
    procedure HighlightReferenceDrop(const Index: Integer);
    procedure MoveDraftImage(const TargetIndex: Integer);
    procedure SetNegativePromptEditing(const Value: Boolean);
    procedure UpdateNegativePromptState;
    function ComposerPromptText: string;
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
    procedure AnalyzeModelSource(const UseBundle: Boolean = False);
    procedure SetWizardStep(const Value: Integer);
    procedure RecalculateSettings;
    function WizardImageSettings: TChatImageSettings;
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
    procedure UseImageAsReference(const Image: TChatImage);
    procedure ShowImageReferenceMenu(const Image: TChatImage; const ScreenPosition: TPointF);
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
  System.Math, System.Rtti, QuickNN.ModelFiles, System.IOUtils,
  System.Diagnostics, Winapi.Windows, Winapi.ShellAPI, FMX.Platform,
  FMX.Platform.Win, LLMUI.Theme, LLMUI.Dialogs, LLMUI.Controls,
  QuickNN.Models.LLM.Types, LLMUI.Context.Types, LLMUI.Context;

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
  FAdapterFiles := TList<string>.Create;
  FrameRoleSectionConversation.OnExpandedChanged := RoleExpandedChanged;
  FrameRoleSectionVision.OnExpandedChanged := RoleExpandedChanged;
  FrameRoleSectionImages.OnExpandedChanged := RoleExpandedChanged;
  ListBoxModelAdapters.DefaultItemStyles.ItemStyle := 'chatcomboitem';
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
  FReferenceDragPending := False;
  FReferenceMenuPending := False;
  if MenuItemUseImageReference.View <> nil then
    MenuItemUseImageReference.View.Loop := False;
  FReferenceMenuImage := Default(TChatImage);
  TimerUI.Enabled := False;
  TimerMessages.Enabled := False;
  TimerModelInspect.Enabled := False;
  if FReady then
    SaveState;
  FReady := False;
  FEngine.Free;
  FFrames.Free;
  FVirtualStates.Free;
  FImageModelIds.Free;
  FVisionModelIds.Free;
  FModelIds.Free;
  FAdapterFiles.Free;
  FStore.Free;
  FComposerTextLayout.Free;
end;

procedure TFormMain.FormCloseQuery(Sender: TObject; var CanClose: Boolean);
begin
  // OLE is pumping the form's timer; keep it alive until the drag unwinds.
  if FReferenceDragActive then
  begin
    CanClose := False;
    Exit
  end;
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
      FEditingNegativePrompt := False;
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
      if not ImageModel.Available then
        Continue;
      ComboBoxImageModel.Items.Add(ImageModel.Name);
      FImageModelIds.Add(ImageModel.Id);
    end;
    FModelIds.Clear;
    for var Model in FStore.Models do
    begin
      if not Model.Available then
        Continue;
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
    NumberBoxVisionContext.Value := Chat.Preferences.VisionContext;
    NumberBoxVisionMaxTokens.Value := Chat.Preferences.VisionMaxTokens;
    NumberBoxVisionTemperature.Value := Chat.Preferences.VisionTemperature;
    MemoVisionInstruction.Text := Chat.Preferences.VisionInstruction;
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
    NumberBoxImageWidth.Max := 1024;
    NumberBoxImageHeight.Max := 1024;
    NumberBoxImageWidth.VertIncrement := 16;
    NumberBoxImageHeight.VertIncrement := 16;
    if (ImageModel <> nil) and (ImageModel.Settings.Kind = ipQwenImage) then
    begin
      NumberBoxImageWidth.Max := 2048;
      NumberBoxImageHeight.Max := 2048;
      NumberBoxImageWidth.VertIncrement := 8;
      NumberBoxImageHeight.VertIncrement := 8;
    end;
    if (ImageModel <> nil) and (ImageModel.Settings.Kind = ipFlux1) then
    begin
      NumberBoxImageWidth.Max := 2048;
      NumberBoxImageHeight.Max := 2048
    end;
    LayoutImageLoRA.Visible := (ImageModel <> nil) and ImageSupportsLoRA(ImageModel.Settings);
    EditImageLoRAFile.Text := Chat.Preferences.ImageGeneration.LoRAFile;
    EditImageLoRAFile.Hint := EditImageLoRAFile.Text;
    NumberBoxImageLoRAStrength.Value := Chat.Preferences.ImageGeneration.LoRAStrength;
    ComboBoxImageAdapter.Items.Clear;
    FAdapterFiles.Clear;
    ComboBoxImageAdapter.Items.Add('Без адаптера');
    FAdapterFiles.Add('');
    if ImageModel <> nil then
      for var FileName in ImageModel.Settings.Adapters + FindModelAdapters(ImageModel.Settings.Directory) do
        if FAdapterFiles.IndexOf(FileName) < 0 then
        begin
          ComboBoxImageAdapter.Items.Add(ChangeFileExt(ExtractFileName(FileName), ''));
          FAdapterFiles.Add(FileName)
        end;
    var AdapterIndex := FAdapterFiles.IndexOf(EditImageLoRAFile.Text);
    if (AdapterIndex < 0) and (EditImageLoRAFile.Text <> '') then
    begin
      ComboBoxImageAdapter.Items.Add(ChangeFileExt(ExtractFileName(EditImageLoRAFile.Text), ''));
      FAdapterFiles.Add(EditImageLoRAFile.Text);
      AdapterIndex := FAdapterFiles.Count - 1;
    end;
    ComboBoxImageAdapter.ItemIndex := Max(0, AdapterIndex);
    ConfigureComboBox(ComboBoxImageAdapter);
    FrameRoleSectionConversation.Expanded := Chat.Preferences.ChatSettingsExpanded;
    FrameRoleSectionVision.Expanded := Chat.Preferences.VisionSettingsExpanded;
    FrameRoleSectionImages.Expanded := Chat.Preferences.ImageSettingsExpanded;
    NumberBoxImageWidth.Value := Chat.Preferences.ImageGeneration.Width;
    NumberBoxImageHeight.Value := Chat.Preferences.ImageGeneration.Height;
    NumberBoxImageSteps.Value := Chat.Preferences.ImageGeneration.Steps;
    NumberBoxImageSeed.Value := Chat.Preferences.ImageGeneration.Seed;
    NumberBoxImageGuidance.Value := Chat.Preferences.ImageGeneration.Guidance;
    ComboBoxImageAnalysis.ItemIndex := FVisionModelIds.IndexOf(Chat.Preferences.ImageGeneration.AnalysisModelId);
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
  RecalculateSettings;
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
  if (Sender = ComboBoxImageModel) and (Chat.Preferences.ImageGeneration.NegativePrompt <> '') then
  begin
    var Target: TImageModelEntry := nil;
    if (ComboBoxImageModel.ItemIndex >= 0) and (ComboBoxImageModel.ItemIndex < FImageModelIds.Count) then
      Target := FStore.FindImageModel(FImageModelIds[ComboBoxImageModel.ItemIndex]);
    if (Target = nil) or not Target.SupportsNegativePrompt then
    begin
      var Name := 'Не выбрана модель';
      if Target <> nil then
        Name := Target.Name;
      var Accepted := TChatDialogs.Confirm(Self, 'Сбросить негативный промпт?',
        'Модель «' + Name + '» не поддерживает негативный промпт. При смене модели он будет сброшен. Основной промпт и вложения сохранятся.',
        'Сбросить и сменить');
      if not FReady or (ActiveChat <> Chat) then
        Exit;
      if not Accepted then
      begin
        FUpdating := True;
        try
          ComboBoxImageModel.ItemIndex := FImageModelIds.IndexOf(Chat.Preferences.ImageGeneration.ModelId)
        finally
          FUpdating := False
        end;
        Exit;
      end;
      SetNegativePromptEditing(False);
      Chat.Preferences.ImageGeneration.NegativePrompt := '';
    end;
  end;
  if (ComboBoxModel.ItemIndex >= 0) and (ComboBoxModel.ItemIndex < FModelIds.Count) then
    Chat.Preferences.ModelId := FModelIds[ComboBoxModel.ItemIndex];
  if ComboBoxContext.ItemIndex >= 0 then
    Chat.Preferences.Context := StrToIntDef(ComboBoxContext.Selected.Text, 2048);
  Chat.Preferences.Generation.Temperature := TrackBarTemperature.Value;
  Chat.Preferences.VisionContext := Round(NumberBoxVisionContext.Value);
  Chat.Preferences.VisionMaxTokens := Round(NumberBoxVisionMaxTokens.Value);
  Chat.Preferences.VisionTemperature := NumberBoxVisionTemperature.Value;
  Chat.Preferences.VisionInstruction := MemoVisionInstruction.Text;
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
        var NegativePrompt := Chat.Preferences.ImageGeneration.NegativePrompt;
        Chat.Preferences.ImageGeneration := ImageModel.Resolve(ImageModel.Settings);
        Chat.Preferences.ImageGeneration.AnalysisModelId := AnalysisId;
        Chat.Preferences.ImageGeneration.NegativePrompt := NegativePrompt;
        LoadPreferences;
      end;
    end
    else
      Chat.Preferences.ImageGeneration.ModelId := '';
  end;
  var ImageStep := 16;
  if Chat.Preferences.ImageGeneration.Kind = ipQwenImage then
    ImageStep := 8;
  Chat.Preferences.ImageGeneration.Width := Round(NumberBoxImageWidth.Value) div ImageStep * ImageStep;
  Chat.Preferences.ImageGeneration.Height := Round(NumberBoxImageHeight.Value) div ImageStep * ImageStep;
  Chat.Preferences.ImageGeneration.LoRAFile := EditImageLoRAFile.Text;
  Chat.Preferences.ImageGeneration.LoRAStrength := NumberBoxImageLoRAStrength.Value;
  Chat.Preferences.ImageGeneration.Steps := Round(NumberBoxImageSteps.Value);
  Chat.Preferences.ImageGeneration.Seed := Round(NumberBoxImageSeed.Value);
  Chat.Preferences.ImageGeneration.Guidance := NumberBoxImageGuidance.Value;
  if (ComboBoxImageAnalysis.ItemIndex >= 0) and (ComboBoxImageAnalysis.ItemIndex < FVisionModelIds.Count) then
    Chat.Preferences.ImageGeneration.AnalysisModelId := FVisionModelIds[ComboBoxImageAnalysis.ItemIndex]
  else if Sender = ComboBoxImageAnalysis then
    Chat.Preferences.ImageGeneration.AnalysisModelId := '';
  Model := FStore.FindModel(Chat.Preferences.ModelId);
  if (Sender = ComboBoxPrecision) and ((Model = nil) or not Model.IsGGUF) then
    Chat.Preferences.Int4Experts := ComboBoxPrecision.ItemIndex = 0;
  if (Model <> nil) and Model.SupportsThinking then
    Chat.Preferences.Generation.EnableThinking := CheckBoxThinking.IsChecked
  else
    Chat.Preferences.Generation.EnableThinking := False;
  UpdateKVLabel;
  UpdateAvailability;
  RecalculateSettings;
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
      ComboBoxPrecision.Enabled := not Model.IsGGUF and ((Model.ModelType = 'qwen3_moe') or (Model.ModelType = 'qwen3_5'));
      ComboBoxPrecision.Items[0] := 'INT4 · меньше памяти';
      ComboBoxPrecision.Items[1] := 'Исходная точность';
      if Model.IsGGUF then
      begin
        ComboBoxPrecision.Items[1] := 'GGUF · квантование файла';
        ComboBoxPrecision.ItemIndex := 1;
      end
      else if Chat.Preferences.Int4Experts then
        ComboBoxPrecision.ItemIndex := 0
      else
        ComboBoxPrecision.ItemIndex := 1;
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
  UpdateNegativePromptState;
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
    if (Chat <> nil) and (ComboBoxComposerMode.ItemIndex = 1) then
    begin
      var ImageModel := FStore.FindImageModel(Chat.Preferences.ImageGeneration.ModelId);
      ModelReady := (ImageModel <> nil) and ImageModel.Available;
      if ModelReady and (Length(Chat.DraftImages) > 0) and not ImageSupportsReferences(ImageModel.Settings.Kind) then
      begin
        var VisionModel := FStore.FindModel(Chat.Preferences.ImageGeneration.AnalysisModelId);
        ModelReady := (VisionModel <> nil) and VisionModel.Available and VisionModel.SupportsVision;
      end;
    end;
    if (Chat <> nil) and (ComboBoxComposerMode.ItemIndex <> 1) and (Length(Chat.DraftImages) > 0) then
    begin
      var VisionModel := FStore.FindModel(Chat.Preferences.ImageGeneration.AnalysisModelId);
      ModelReady := (VisionModel <> nil) and VisionModel.Available and VisionModel.SupportsVision;
    end;
    ButtonSend.Enabled := (Chat <> nil) and ModelReady and ((Trim(ComposerPromptText) <> '') or (Length(Chat.DraftImages) > 0));
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

function TFormMain.ComposerPromptText: string;
begin
  if FEditingNegativePrompt and (ActiveChat <> nil) then
    Result := ActiveChat.Draft
  else
    Result := MemoComposer.Text;
end;

procedure TFormMain.SetNegativePromptEditing(const Value: Boolean);
begin
  if (FEditingNegativePrompt = Value) or (ActiveChat = nil) then
    Exit;
  var Chat := ActiveChat;
  if FEditingNegativePrompt then
    Chat.Preferences.ImageGeneration.NegativePrompt := MemoComposer.Text
  else
    Chat.Draft := MemoComposer.Text;
  FEditingNegativePrompt := Value;
  var Updating := FUpdating;
  FUpdating := True;
  try
    if Value then
      MemoComposer.Text := Chat.Preferences.ImageGeneration.NegativePrompt
    else
      MemoComposer.Text := Chat.Draft;
  finally
    FUpdating := Updating
  end;
  UpdateComposerHeight;
  MarkDirty;
end;

procedure TFormMain.UpdateNegativePromptState;
begin
  var Chat := ActiveChat;
  var Model: TImageModelEntry := nil;
  if Chat <> nil then
    Model := FStore.FindImageModel(Chat.Preferences.ImageGeneration.ModelId);
  var CanEdit := (ComboBoxComposerMode.ItemIndex = 1) and (Model <> nil)
    and Model.Available and Model.SupportsNegativePrompt;
  if not CanEdit then
    SetNegativePromptEditing(False);
  SpeedButtonNegativePrompt.Enabled := CanEdit;
  SpeedButtonNegativePrompt.IsPressed := FEditingNegativePrompt;
  var Style := 'chatsecondarybutton';
  if FEditingNegativePrompt then
    Style := 'chatprimarybutton';
  if SpeedButtonNegativePrompt.StyleLookup <> Style then
    SpeedButtonNegativePrompt.StyleLookup := Style;
  if FEditingNegativePrompt then
  begin
    SpeedButtonNegativePrompt.Hint := 'Вернуться к основному промпту';
    MemoComposer.TextPrompt := 'Опишите, чего не должно быть в изображении…';
    LabelSendHint.Text := 'Негативный промпт · Shift+Enter — новая строка';
  end
  else
  begin
    SpeedButtonNegativePrompt.Hint := 'Редактировать негативный промпт';
    if not CanEdit then
      SpeedButtonNegativePrompt.Hint := 'Негативный промпт доступен в режиме изображения для совместимой модели';
    MemoComposer.TextPrompt := 'Напишите сообщение…';
    LabelSendHint.Text := 'Enter — отправить · Shift+Enter — новая строка';
  end;
  if CanEdit and (Chat.Preferences.ImageGeneration.Guidance <= 1) then
    SpeedButtonNegativePrompt.Hint := SpeedButtonNegativePrompt.Hint + '. Для применения нужен guidance больше 1';
end;

procedure TFormMain.NegativePromptClick(Sender: TObject);
begin
  if not FReady or FUpdating or not SpeedButtonNegativePrompt.Enabled then
    Exit;
  SetNegativePromptEditing(not FEditingNegativePrompt);
  UpdateAvailability;
  if MemoComposer.CanFocus then
    MemoComposer.SetFocus;
end;

procedure TFormMain.ComposerModeChanged(Sender: TObject);
begin
  if FReady and not FUpdating then
  begin
    if ComboBoxComposerMode.ItemIndex <> 1 then
      SetNegativePromptEditing(False);
    UpdateAvailability;
  end;
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
    if FEditingNegativePrompt then
      ActiveChat.Preferences.ImageGeneration.NegativePrompt := MemoComposer.Text
    else
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
  if (Key = vkReturn) and not (ssShift in Shift) then
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
  ImageRequest, VisionRequest: Boolean;
  EffectiveContext: Integer;
  OriginalImageRequest: TChatImage;
  AnalysisModel: TModelEntry;
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
  if FEditingNegativePrompt then
    Chat.Preferences.ImageGeneration.NegativePrompt := MemoComposer.Text;
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
  OriginalImageRequest := Default(TChatImage);
  ImageRequest := (ComboBoxComposerMode.ItemIndex = 1) and not ContinueAnswer;
  if Retry and (Chat.Messages.Count > 0) then
  begin
    var Last := Chat.Messages.Last;
    if Last.Role = lrAssistant then
    begin
      OriginalImageRequest := Last.ImageRequest;
      if (OriginalImageRequest.RelativePath = '') and (Length(Last.Images) > 0) and not Last.Images[0].IsUserImage then
        OriginalImageRequest := Last.Images[0];
      ImageRequest := OriginalImageRequest.RelativePath <> '';
    end;
  end;
  VisionRequest := HasUpload and not ImageRequest;
  if ContinueAnswer and (Chat.Messages.Count > 0) then
    VisionRequest := Chat.Messages.Last.Continuation.VisionSignature <> '';
  if not ImageRequest and not VisionRequest then
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
  AnalysisModel := nil;
  if ImageRequest then
  begin
    ImageModel := FStore.FindImageModel(Chat.Preferences.ImageGeneration.ModelId);
    if OriginalImageRequest.RelativePath <> '' then
      ImageModel := FStore.FindImageModel(OriginalImageRequest.Settings.ModelId);
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
    if HasUpload and not ImageSupportsReferences(ImageModel.Settings.Kind)
      and (OriginalImageRequest.ReferenceDescription.Trim = '') then
    begin
      var AnalysisId := Chat.Preferences.ImageGeneration.AnalysisModelId;
      if OriginalImageRequest.ReferenceAnalysis.ModelId <> '' then
        AnalysisId := OriginalImageRequest.ReferenceAnalysis.ModelId;
      AnalysisModel := FStore.FindModel(AnalysisId);
      if (AnalysisModel = nil) or not AnalysisModel.Available or not AnalysisModel.SupportsVision then
      begin
        ShowStatus('Для референсов этой модели выберите доступную модель в разделе «Анализ изображений».', True);
        Exit;
      end;
      var AnalysisContext := Chat.Preferences.VisionContext;
      if OriginalImageRequest.ReferenceAnalysis.ModelId <> '' then
        AnalysisContext := OriginalImageRequest.ReferenceAnalysis.Context;
      if AnalysisContext > AnalysisModel.MaxContext then
      begin
        ShowStatus(Format('Для анализа этой моделью выберите контекст не больше %d токенов.', [AnalysisModel.MaxContext]), True);
        Exit;
      end;
    end;
    try
      if OriginalImageRequest.RelativePath <> '' then
        ValidateImageSettings(OriginalImageRequest.Settings)
      else
        ValidateImageSettings(ImageModel.Resolve(Chat.Preferences.ImageGeneration));
    except
      on E: Exception do
      begin
        ShowStatus(E.Message, True);
        Exit
      end;
    end;
  end;
  if VisionRequest and not ContinueAnswer then
  begin
    var AnalysisIndex := ComboBoxImageAnalysis.ItemIndex;
    if (AnalysisIndex < 0) or (AnalysisIndex >= FVisionModelIds.Count) then
    begin
      ShowStatus('Выберите модель анализа изображений в настройках чата.', True);
      Exit
    end;
    Model := FStore.FindModel(FVisionModelIds[AnalysisIndex]);
    if (Model = nil) or not Model.Available or not Model.SupportsVision then
    begin
      ShowStatus('Для анализа требуется доступная Qwen hybrid vision-модель.', True);
      Exit
    end;
  end;
  if not ImageRequest and ((Model = nil) or not Model.Available) then
  begin
    ShowStatus('Выбранная модель недоступна.', True);
    Exit
  end;
  EffectiveContext := Chat.Preferences.Context;
  if VisionRequest then
    EffectiveContext := Chat.Preferences.VisionContext;
  if not ImageRequest and (EffectiveContext > Model.MaxContext) then
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
    if (Trim(ComposerPromptText) = '') and (Length(Chat.DraftImages) = 0) then
      Exit;
    // Only the last reply can be resumed. Avoid storing a full prompt per old reply.
    for var Previous in Chat.Messages do
      Previous.Continuation := Default(TLLMContinuationState);
    Message := TChatMessage.Create;
    Message.Role := lrUser;
    Message.Text := ComposerPromptText.Trim;
    Message.Images := Copy(Chat.DraftImages);
    if Message.Text = '' then
      if ImageRequest then
        Message.Text := 'Создай изображение на основе прикреплённых референсов.'
      else
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
    FEditingNegativePrompt := False;
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
  if VisionRequest then
  begin
    Request.Preferences.Context := Chat.Preferences.VisionContext;
    Request.Preferences.Generation := TLLMGenerationOptions.Defaults;
    Request.Preferences.Generation.EnableThinking := False;
    Request.Preferences.Generation.AutoContinue := False;
    Request.Preferences.Generation.MaxNewTokens := Chat.Preferences.VisionMaxTokens;
    Request.Preferences.Generation.HardMaxNewTokens := Chat.Preferences.VisionMaxTokens;
    Request.Preferences.Generation.Temperature := Chat.Preferences.VisionTemperature;
    if Chat.Preferences.VisionInstruction <> '' then
      Request.Preferences.SystemPrompt := Request.Preferences.SystemPrompt + sLineBreak + Chat.Preferences.VisionInstruction;
  end;
  Request.ContinueResponse := ContinueAnswer;
  Request.ContextMemory := Chat.ContextMemory;
  Request.GenerateImage := ImageRequest;
  if Request.GenerateImage then
  begin
    Request.Image := Default(TChatImage);
    Request.Image.Settings := ImageModel.Resolve(Chat.Preferences.ImageGeneration);
    if OriginalImageRequest.RelativePath <> '' then
    begin
      Request.Image.Settings := OriginalImageRequest.Settings;
      Request.Image.ReferencePaths := Copy(OriginalImageRequest.ReferencePaths);
      Request.Image.ReferenceAnalysis := OriginalImageRequest.ReferenceAnalysis;
      Request.Image.ReferenceDescription := OriginalImageRequest.ReferenceDescription;
    end
    else
      for var Reference in Chat.Messages.Last.Images do
        Request.Image.ReferencePaths := Request.Image.ReferencePaths + [Reference.RelativePath];
    if (AnalysisModel <> nil) and (Request.Image.ReferenceAnalysis.ModelId = '') then
    begin
      Request.Image.ReferenceAnalysis.ModelId := AnalysisModel.Id;
      Request.Image.ReferenceAnalysis.ModelName := AnalysisModel.Name;
      Request.Image.ReferenceAnalysis.Directory := AnalysisModel.Directory;
      Request.Image.ReferenceAnalysis.TokenizerDirectory := AnalysisModel.ResolvedTokenizer;
      Request.Image.ReferenceAnalysis.Context := Chat.Preferences.VisionContext;
      Request.Image.ReferenceAnalysis.MaxTokens := Chat.Preferences.VisionMaxTokens;
      Request.Image.ReferenceAnalysis.Temperature := Chat.Preferences.VisionTemperature;
      Request.Image.ReferenceAnalysis.Instruction := Chat.Preferences.VisionInstruction;
    end;
    Request.Image.Prompt := Chat.Messages.Last.Text;
    Request.Image.ModelName := ImageModel.Name;
    Request.Image.RelativePath := 'images/' + Chat.Id + '/' + NewId + '.png';
    for var I := Chat.Messages.Count - 1 downto 0 do
      if Chat.Messages[I].Role = lrUser then
      begin
        Chat.Messages[I].NegativePrompt := Request.Image.Settings.NegativePrompt;
        Break;
      end;
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
    begin
      Message.ModelName := ImageModel.Name;
      Message.ImageRequest := Request.Image;
    end
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
  var EffectiveContext := Chat.Preferences.Context;
  var SystemPrompt := Chat.Preferences.SystemPrompt;
  if Message.Continuation.VisionSignature <> '' then
  begin
    EffectiveContext := Chat.Preferences.VisionContext;
    if Chat.Preferences.VisionInstruction <> '' then
      SystemPrompt := SystemPrompt + sLineBreak + Chat.Preferences.VisionInstruction;
  end;
  if (Length(Message.Continuation.Tokens) >= EffectiveContext)
    and not ContextCanCompact(
      Chat.ContextHistory(Message.Id),
      SystemPrompt,
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
          Pair.Value.Height + MESSAGE_GAP, Pair.Value.ExpanderThinking.IsExpanded, Pair.Value.NegativePromptExpanded);
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
          Frame.ExpanderThinking.IsExpanded, Frame.NegativePromptExpanded);
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
      Frame.NegativePromptExpanded := Metric.NegativeExpanded;
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
            Frame.ExpanderThinking.IsExpanded, Frame.NegativePromptExpanded) then
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
  if LayoutStatusContent <> nil then
  begin
    LayoutStatusContent.Margins.Right := PanelComposer.Margins.Right;
    LayoutStatusContent.Width := PanelComposer.Width;
  end;
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
  SetNegativePromptEditing(False);
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
        if Snapshot.ImageRequestUpdated or Snapshot.ImageReady then
        begin
          Message.ImageRequest := Snapshot.Image;
          FStore.Save; // Keep a completed reference analysis even if generation is interrupted.
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
  if FReferenceDragPending then
    StartDraftImageDrag;
  if FReferenceMenuPending then
  begin
    FReferenceMenuPending := False;
    if FReady and not FClosing and not TChatImageViewer.IsOpen(Self)
      and (ActiveChat <> nil) and (ActiveChat.Id = FReferenceMenuChatId) then
    begin
      if Captured <> nil then
        SetCaptured(nil);
      PopupMenuImageReference.Popup(FReferenceMenuPosition.X, FReferenceMenuPosition.Y);
    end;
  end;
end;

procedure TFormMain.ModelSelected(Sender: TObject);
begin
  SelectModelEditor(TFrameEntry(Sender).EntryId);
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
      begin
        TFrameAttachment(Child).OnRemove := nil;
        TFrameAttachment(Child).OnBeginReferenceDrag := nil;
      end;
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
      Frame.Align := TAlignLayout.Left;
      // Left alignment sorts siblings by their initial X; seed it in reference order.
      Frame.Position.X := I * 154;
      Frame.Parent := LayoutDraftImages;
      Frame.Margins.Right := 6;
      Frame.SetImages(Chat.DraftImages, I);
      Frame.OnRemove := RemoveDraftImage;
      Frame.OnBeginReferenceDrag := BeginDraftImageDrag;
      Frame.RectangleDropTarget.BringToFront;
    end;
  LayoutComposerDockResize(Self);
  QueueMessageLayout;
end;

procedure TFormMain.BeginDraftImageDrag(Sender: TObject);
begin
  if not FReady or FClosing or FReferenceDragActive or not (Sender is TFrameAttachment)
    or (ActiveChat = nil) or (TFrameAttachment(Sender).Parent <> LayoutDraftImages) then
    Exit;
  var Index := TFrameAttachment(Sender).ImageIndex;
  if (Index < 0) or (Index >= Length(ActiveChat.DraftImages)) then
    Exit;
  FReferenceDragChatId := ActiveChat.Id;
  FReferenceDragPath := ActiveChat.DraftImages[Index].RelativePath;
  FReferenceDragPending := True;
end;

procedure TFormMain.StartDraftImageDrag;
var
  Service: IFMXDragDropService;
begin
  FReferenceDragPending := False;
  if not FReady or FClosing or (ActiveChat = nil) or (ActiveChat.Id <> FReferenceDragChatId)
    or FrameImageViewer.Visible or (GetAsyncKeyState(VK_LBUTTON) >= 0)
    or not TPlatformServices.Current.SupportsPlatformService(IFMXDragDropService, Service) then
    Exit;
  var Source := -1;
  for var I := 0 to High(ActiveChat.DraftImages) do
    if ActiveChat.DraftImages[I].RelativePath = FReferenceDragPath then
    begin
      Source := I;
      Break
    end;
  if Source < 0 then
    Exit;
  try
    var Bitmap := FMX.Graphics.TBitmap.Create;
    try
      var FileName := ActiveChat.DraftImages[Source].FileName;
      if TFile.Exists(FileName) then
        Bitmap.LoadThumbnailFromFile(FileName, 148, 104, False);
      if Bitmap.IsEmpty then
        Bitmap.SetSize(148, 104);
      var Ticket := TObject.Create;
      try
        var Data := Default(TDragObject);
        Data.Source := Self; // Never pass a transient thumbnail into OLE.
        // An opaque object offers only the private FMX drag format, no text or
        // file data that could be inserted into a memo or another application.
        Data.Data := TValue.From<TObject>(Ticket);
        if Captured <> nil then
          SetCaptured(nil);
        FReferenceDragTicket := Ticket;
        FReferenceDragActive := True;
        FReferenceDragScrollAt := 0;
        try
          Service.BeginDragDrop(Self, Data, Bitmap)
        finally
          FReferenceDragActive := False;
          FReferenceDragTicket := nil;
          FReferenceDragPath := '';
          FReferenceDragChatId := '';
          HighlightReferenceDrop(-1);
        end;
      finally
        Ticket.Free
      end;
    finally
      Bitmap.Free
    end;
  except
    on E: Exception do
      ShowStatus('Не удалось переместить референс: ' + E.Message, True);
  end;
end;

function TFormMain.ReferenceDropIndex(const Data: TDragObject; const Point: TPointF): Integer;
begin
  Result := -1;
  if not FReferenceDragActive or (Data.Source <> Self) or not Data.Data.IsObject
    or (FReferenceDragTicket = nil) or (Data.Data.AsObject <> FReferenceDragTicket)
    or not FReady or FClosing or (ActiveChat = nil) or (ActiveChat.Id <> FReferenceDragChatId)
    or FrameImageViewer.Visible or (TabControlMain.ActiveTab <> TabItemChats)
    or not HorzScrollBoxAttachments.AbsoluteRect.Contains(ScreenToClient(Point)) then
    Exit;
  var local := LayoutDraftImages.ScreenToLocal(Point);
  var Distance := MaxSingle;
  for var Child in LayoutDraftImages.Children do
    if Child is TFrameAttachment then
    begin
      var Frame := TFrameAttachment(Child);
      var Delta := Abs(local.X - Frame.Position.X - Frame.Width / 2);
      if Delta < Distance then
      begin
        Distance := Delta;
        Result := Frame.ImageIndex
      end;
    end;
end;

procedure TFormMain.HighlightReferenceDrop(const Index: Integer);
begin
  for var Child in LayoutDraftImages.Children do
    if Child is TFrameAttachment then
      TFrameAttachment(Child).RectangleDropTarget.Visible := TFrameAttachment(Child).ImageIndex = Index;
end;

procedure TFormMain.MoveDraftImage(const TargetIndex: Integer);
begin
  var Chat := ActiveChat;
  if (Chat = nil) or (Chat.Id <> FReferenceDragChatId) or (TargetIndex < 0)
    or (TargetIndex >= Length(Chat.DraftImages)) then
    Exit;
  var Source := -1;
  for var I := 0 to High(Chat.DraftImages) do
    if Chat.DraftImages[I].RelativePath = FReferenceDragPath then
    begin
      Source := I;
      Break
    end;
  if (Source < 0) or (Source = TargetIndex) then
    Exit;
  var Image := Chat.DraftImages[Source];
  if Source < TargetIndex then
    for var I := Source to TargetIndex - 1 do
      Chat.DraftImages[I] := Chat.DraftImages[I + 1]
  else
    for var I := Source downto TargetIndex + 1 do
      Chat.DraftImages[I] := Chat.DraftImages[I - 1];
  Chat.DraftImages[TargetIndex] := Image;
  var Position := HorzScrollBoxAttachments.ViewportPosition;
  RenderDraftImages;
  HorzScrollBoxAttachments.ViewportPosition := Position;
  MarkDirty;
  ShowStatus(Format('Изображение перемещено: <image%d>.', [TargetIndex + 1]));
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

procedure TFormMain.ShowImageReferenceMenu(const Image: TChatImage; const ScreenPosition: TPointF);
begin
  if not FReady or FClosing or (ActiveChat = nil) then
    Exit;
  FReferenceMenuImage := Image;
  FReferenceMenuChatId := ActiveChat.Id;
  FReferenceMenuPosition := ScreenPosition;
  FReferenceMenuPending := True;
end;

procedure TFormMain.ImageReferenceMenuPopup(Sender: TObject);
begin
  MenuItemUseImageReference.Enabled := False;
  MenuItemCopyAttachment.Enabled := False;
  if not FReady or FClosing or (ActiveChat = nil) or (ActiveChat.Id <> FReferenceMenuChatId) then
    Exit;
  try
    MenuItemUseImageReference.Enabled := TFile.Exists(FReferenceMenuImage.FileName)
  except
    on E: EArgumentException do
      MenuItemUseImageReference.Enabled := False
  end;
  MenuItemCopyAttachment.Enabled := MenuItemUseImageReference.Enabled;
end;

procedure TFormMain.CopyAttachmentClick(Sender: TObject);
begin
  var Image := FReferenceMenuImage;
  FReferenceMenuImage := Default(TChatImage);
  FReferenceMenuChatId := '';
  if MenuItemCopyAttachment.View <> nil then
    MenuItemCopyAttachment.View.Loop := False;
  if not FReady or FClosing then
    Exit;
  try
    var Bitmap := FMX.Graphics.TBitmap.Create;
    try
      Bitmap.LoadFromFile(Image.FileName);
      CopyImageToClipboard(Bitmap, WindowHandleToPlatform(Handle).Wnd);
    finally
      Bitmap.Free
    end;
    ShowStatus('Изображение скопировано в буфер обмена.');
  except
    on E: Exception do
      ShowStatus('Не удалось скопировать изображение: ' + E.Message, True);
  end;
end;

procedure TFormMain.UseImageReferenceClick(Sender: TObject);
begin
  var Image := FReferenceMenuImage;
  var ChatId := FReferenceMenuChatId;
  FReferenceMenuImage := Default(TChatImage);
  FReferenceMenuChatId := '';
  if MenuItemUseImageReference.View <> nil then
    MenuItemUseImageReference.View.Loop := False;
  if not FReady or FClosing or (ActiveChat = nil) or (ActiveChat.Id <> ChatId) then
    Exit;
  UseImageAsReference(Image);
end;

procedure TFormMain.UseImageAsReference(const Image: TChatImage);
begin
  if not FReady or FClosing or (ActiveChat = nil) then
    Exit;
  try
    var Reference := Image.AsReference;
    var Chat := ActiveChat;
    var Index := -1;
    for var I := 0 to High(Chat.DraftImages) do
      if (Chat.DraftImages[I].ContentHash = Reference.ContentHash)
        or SameFileName(Chat.DraftImages[I].FileName, Reference.FileName) then
      begin
        Index := I;
        Break
      end;
    var Added := Index < 0;
    if Added then
    begin
      if Length(Chat.DraftImages) >= MAX_MESSAGE_IMAGES then
        raise EArgumentException.Create('К одному сообщению можно прикрепить до четырёх изображений.');
      Index := Length(Chat.DraftImages);
      Chat.DraftImages := Chat.DraftImages + [Reference];
    end;
    // Reuse the archived file. Import deduplication must never delete a file
    // owned by an existing history message, even when its bytes match a draft.
    SetActivePage(False);
    SetNegativePromptEditing(False);
    ComboBoxComposerMode.ItemIndex := 1;
    if Added then
      RenderDraftImages;
    UpdateAvailability;
    MarkDirty;
    if MemoComposer.CanFocus then
      MemoComposer.SetFocus;
    if Added then
      ShowStatus(Format('Референс <image%d> добавлен. Дополните запрос и отправьте.', [Index + 1]))
    else
      ShowStatus(Format('Это изображение уже прикреплено как <image%d>.', [Index + 1]));
  except
    on E: Exception do
      ShowStatus('Не удалось добавить референс: ' + E.Message, True);
  end;
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
  if ReferenceDropIndex(Data, Point) >= 0 then
  begin
    HighlightReferenceDrop(ReferenceDropIndex(Data, Point));
    Exit;
  end;
  if FReferenceDragActive and (Data.Source = Self) then
    Exit;
  if AcceptImageDrop(Data, Point) then
    RectangleImageDrop.Visible := True
  else
    inherited;
end;

procedure TFormMain.DragOver(const Data: TDragObject; const Point: TPointF; var Operation: TDragOperation);
begin
  var Target := ReferenceDropIndex(Data, Point);
  HighlightReferenceDrop(Target);
  if Target >= 0 then
  begin
    Operation := TDragOperation.Move;
    // Keep clipped references reachable while holding the pointer at an edge.
    if GetTickCount64 >= FReferenceDragScrollAt then
    begin
      FReferenceDragScrollAt := GetTickCount64 + 30;
      var local := HorzScrollBoxAttachments.ScreenToLocal(Point);
      var Step: Single := 0;
      if local.X < 28 then
        Step := -10
      else if local.X > HorzScrollBoxAttachments.Width - 28 then
        Step := 10;
      if Step <> 0 then
        HorzScrollBoxAttachments.ViewportPosition := PointF(Max(0,
            HorzScrollBoxAttachments.ViewportPosition.X + Step), 0);
    end;
    Exit;
  end;
  if FReferenceDragActive and (Data.Source = Self) then
  begin
    Operation := TDragOperation.None;
    Exit
  end;
  RectangleImageDrop.Visible := AcceptImageDrop(Data, Point);
  if RectangleImageDrop.Visible then
    Operation := TDragOperation.Copy
  else
    inherited;
end;

procedure TFormMain.DragDrop(const Data: TDragObject; const Point: TPointF);
begin
  RectangleImageDrop.Visible := False;
  var Target := ReferenceDropIndex(Data, Point);
  HighlightReferenceDrop(-1);
  if Target >= 0 then
  begin
    MoveDraftImage(Target);
    FReferenceDragTicket := nil; // A native drop can only be applied once.
    Exit;
  end;
  if FReferenceDragActive and (Data.Source = Self) then
    Exit;
  if AcceptImageDrop(Data, Point) then
    AttachImageFiles(Data.Files)
  else
    inherited;
end;

procedure TFormMain.DragLeave;
begin
  HighlightReferenceDrop(-1);
  RectangleImageDrop.Visible := False;
  inherited;
end;

procedure TFormMain.ImageCheckpointChanged(Sender: TObject);
begin
  if FReady and not FUpdating then
    UpdateImageEditor;
end;

procedure TFormMain.ButtonBrowseImageLoRAClick(Sender: TObject);
begin
  OpenDialogImageLoRA.FileName := EditImageLoRAFile.Text;
  if OpenDialogImageLoRA.Execute then
  begin
    EditImageLoRAFile.Text := OpenDialogImageLoRA.FileName;
    EditImageLoRAFile.Hint := EditImageLoRAFile.Text;
    PreferencesChanged(ButtonBrowseImageLoRA);
  end;
end;

procedure TFormMain.ButtonClearImageLoRAClick(Sender: TObject);
begin
  EditImageLoRAFile.Text := '';
  PreferencesChanged(ButtonClearImageLoRA);
end;

procedure TFormMain.ButtonBrowseCheckpointClick(Sender: TObject);
begin
  OpenDialogModelCheckpoint.FileName := EditModelCheckpoint.Text;
  if OpenDialogModelCheckpoint.Execute then
    EditModelCheckpoint.Text := OpenDialogModelCheckpoint.FileName;
end;

procedure TFormMain.ButtonOpenStorageClick(Sender: TObject);
begin
  TDirectory.CreateDirectory(ExtractFileDir(FStore.Path));
  ShellExecute(0, 'open', PChar(ExtractFileDir(FStore.Path)), nil, nil, SW_SHOWNORMAL);
end;

procedure TFormMain.RenderModels;

  procedure AddEntry(const Id, Name, Location, Capabilities: string);
  begin
    var Frame := TFrameEntry.Create(Self);
    Frame.Name := 'FrameModelEntry' + Id.Replace('{', '').Replace('}', '').Replace('-', '');
    Frame.Parent := LayoutModelItems;
    Frame.Align := TAlignLayout.Top;
    Frame.Margins.Bottom := 8;
    Frame.SetEntry(Id, Name, Capabilities + ' · ' + Location, '', Id = FEditingModelId);
    Frame.OnSelect := ModelSelected;
  end;

begin
  if not FReady then
    Exit;
  LayoutModelItems.BeginUpdate;
  try
    ClearControls(LayoutModelItems);
    for var Model in FStore.Models do
    begin
      var Capabilities := 'Переписка';
      if Model.SupportsVision then
        Capabilities := Capabilities + ' · Vision';
      if not Model.Available then
        Capabilities := 'Недоступна';
      AddEntry(Model.Id, Model.Name, Model.Directory, Capabilities);
    end;
    for var Model in FStore.ImageModels do
    begin
      var Capabilities := 'Изображения';
      if not Model.Available then
        Capabilities := 'Недоступна';
      AddEntry(Model.Id, Model.Name, Model.Settings.Directory, Capabilities);
    end;
    LayoutModelItems.Height := (FStore.Models.Count + FStore.ImageModels.Count) * 86;
  finally
    LayoutModelItems.EndUpdate
  end;
end;

procedure TFormMain.SetWizardStep(const Value: Integer);
begin
  FWizardStep := EnsureRange(Value, 0, 2);
  LayoutWizardSource.Visible := FWizardStep = 0;
  LayoutWizardFiles.Visible := FWizardStep = 1;
  LayoutWizardDefaults.Visible := FWizardStep = 2;
  ButtonWizardBack.Visible := FWizardStep > 0;
  ButtonWizardNext.Visible := FWizardStep < 2;
  ButtonSaveModel.Visible := FWizardStep = 2;
  case FWizardStep of
    0:
      LabelWizardStep.Text := '1 / 3 · Основной файл';
    1:
      LabelWizardStep.Text := '2 / 3 · Файлы и адаптеры';
    2:
      LabelWizardStep.Text := '3 / 3 · Название и настройки';
  end;
  UpdateImageEditor;
  VertScrollBoxModelEditor.ViewportPosition := PointF(0, 0);
end;

procedure TFormMain.UpdateImageEditor;
begin
  LayoutModelEditorFields.BeginUpdate;
  try
    var IsImage := FDetectedModel.Kind = dkImage;
    var IsQwen := IsImage and (FDetectedModel.Image.Kind = ipQwenImage);
    var IsFlux1 := IsImage and (FDetectedModel.Image.Kind = ipFlux1);
    LayoutImageModelDefaults.Visible := IsImage;
    LayoutModelImageOptionsSlot.Visible := IsImage;
    LayoutModelTokenizerSlot.Visible := ((FDetectedModel.Kind = dkLanguage)
      and not SameText(ExtractFileExt(FDetectedModel.SourceFile), '.gguf')) or IsQwen;
    LayoutTokenizerDirectory.Visible := LayoutModelTokenizerSlot.Visible;
    LayoutModelCheckpointSlot.Visible := False;
    LayoutModelRoot.Visible := IsImage;
    LayoutModelTextEncoder.Visible := IsImage;
    LayoutModelTextEncoder2.Visible := IsFlux1;
    LayoutModelTokenizer.Visible := IsImage and not IsQwen;
    LayoutModelTokenizer2.Visible := IsFlux1;
    LayoutModelVAE.Visible := IsImage;
    LayoutModelAdapters.Visible := mcLoRA in FDetectedModel.Capabilities;
    CheckBoxImageModelInt8.Visible := IsQwen and (EditModelTextEncoder.Text = '');
    CheckBoxImageModelInt8.Enabled := IsQwen and (EditModelTextEncoder.Text = '');
    LabelModelTextEncoder.Text := 'Текстовый энкодер (каталог с весами)';
    LabelModelTextEncoder2.Text := 'T5-XXL (каталог с весами)';
    LabelModelVAE.Text := 'VAE (файл или каталог с config.json)';
    if IsFlux1 then
      LabelModelTextEncoder.Text := 'CLIP-L (каталог с весами)';
    if IsQwen then
    begin
      LabelModelTextEncoder.Text := 'Текстовый энкодер (файл safetensors)';
      LabelModelVAE.Text := 'VAE (файл safetensors)';
    end;
    LabelTokenizerDirectory.Text := 'Каталог токенизатора';
    LabelTokenizerHint.Text := 'Для GGUF словарь встроен; отдельный токенизатор не требуется.';
    if IsQwen then
    begin
      LabelTokenizerDirectory.Text := 'Конфигурации и токенизатор Qwen-Image';
      LabelTokenizerHint.Text := 'tokenizer.json, encoder_config.json, vae_config.json, scheduler_config.json';
    end;
    LabelModelCapabilities.Text := FDetectedModel.Architecture + sLineBreak + FDetectedModel.CapabilityText;
    if FDetectedModel.Kind in [dkUnknown, dkAdapter, dkComponent] then
      LabelModelCapabilities.Text := FDetectedModel.Notice;
    LabelWizardSummary.Text := FDetectedModel.Architecture + sLineBreak + FDetectedModel.CapabilityText
      + sLineBreak + 'Модель будет доступна только для совместимых ролей. Настройки можно менять отдельно в каждом чате.';
    var MaxSize := 1024;
    var SizeStep := 16;
    if IsFlux1 or IsQwen then
      MaxSize := 2048;
    if IsQwen then
      SizeStep := 8;
    NumberBoxImageModelWidth.Max := MaxSize;
    NumberBoxImageModelHeight.Max := MaxSize;
    NumberBoxImageModelWidth.VertIncrement := SizeStep;
    NumberBoxImageModelHeight.VertIncrement := SizeStep;
    FitVerticalLayout(LayoutImageModelDefaults);
    FitVerticalLayout(LayoutModelImageOptionsSlot);
    FitVerticalLayout(LayoutWizardSource);
    FitVerticalLayout(LayoutWizardFiles);
    FitVerticalLayout(LayoutWizardDefaults);
    FitVerticalLayout(LayoutModelEditorFields);
    FitVerticalLayout(LayoutWizardFooter);
    ButtonWizardNext.Enabled := not FModelInspectPending and (FDetectedModel.Kind in [dkLanguage, dkImage]);
    ButtonRemoveModel.Visible := FEditingModelId <> '';
  finally
    LayoutModelEditorFields.EndUpdate
  end;
end;

function TFormMain.WizardImageSettings: TChatImageSettings;
begin
  Result := FDetectedModel.Image;
  Result.SourceFile := FDetectedModel.SourceFile;
  Result.Directory := EditModelRoot.Text.Trim.Trim(['"']);
  Result.AssetsDirectory := EditTokenizerDirectory.Text.Trim.Trim(['"']);
  Result.Components.TextEncoder := EditModelTextEncoder.Text.Trim.Trim(['"']);
  Result.Components.TextEncoder2 := EditModelTextEncoder2.Text.Trim.Trim(['"']);
  Result.Components.Tokenizer := EditModelTokenizer.Text.Trim.Trim(['"']);
  Result.Components.Tokenizer2 := EditModelTokenizer2.Text.Trim.Trim(['"']);
  Result.Components.VAE := EditModelVAE.Text.Trim.Trim(['"']);
  Result.Adapters := ListBoxModelAdapters.Items.ToStringArray;
  Result.UseInt8 := CheckBoxImageModelInt8.IsChecked;
  Result.Width := Round(NumberBoxImageModelWidth.Value);
  Result.Height := Round(NumberBoxImageModelHeight.Value);
  Result.Steps := Round(NumberBoxImageModelSteps.Value);
  Result.Seed := Round(NumberBoxImageModelSeed.Value);
  Result.Guidance := NumberBoxImageModelGuidance.Value;
end;

procedure TFormMain.AnalyzeModelSource(const UseBundle: Boolean);
begin
  TimerModelInspect.Enabled := False;
  FModelInspectPending := False;
  var Root := '';
  if UseBundle then
    Root := EditModelRoot.Text.Trim;
  FDetectedModel := DetectLocalModel(EditModelDirectory.Text, Root);
  FUpdating := True;
  try
    if EditModelName.Text.Trim = '' then
      EditModelName.Text := FDetectedModel.SuggestedName;
    EditModelRoot.Text := FDetectedModel.Image.Directory;
    EditModelCheckpoint.Text := FDetectedModel.Image.DiffusionCheckpoint;
    EditTokenizerDirectory.Text := FDetectedModel.TokenizerDirectory;
    if FDetectedModel.Kind = dkImage then
      EditTokenizerDirectory.Text := FDetectedModel.Image.AssetsDirectory;
    EditModelTextEncoder.Text := FDetectedModel.Image.Components.TextEncoder;
    EditModelTextEncoder2.Text := FDetectedModel.Image.Components.TextEncoder2;
    EditModelTokenizer.Text := FDetectedModel.Image.Components.Tokenizer;
    EditModelTokenizer2.Text := FDetectedModel.Image.Components.Tokenizer2;
    EditModelVAE.Text := FDetectedModel.Image.Components.VAE;
    ListBoxModelAdapters.Items.Clear;
    for var FileName in FDetectedModel.Image.Adapters do
      ListBoxModelAdapters.Items.Add(FileName);
    CheckBoxImageModelInt8.IsChecked := FDetectedModel.Image.UseInt8;
    UpdateImageEditor;
    NumberBoxImageModelWidth.Value := FDetectedModel.Image.Width;
    NumberBoxImageModelHeight.Value := FDetectedModel.Image.Height;
    NumberBoxImageModelSteps.Value := FDetectedModel.Image.Steps;
    NumberBoxImageModelSeed.Value := FDetectedModel.Image.Seed;
    NumberBoxImageModelGuidance.Value := FDetectedModel.Image.Guidance;
    LabelModelValidation.Text := FDetectedModel.Notice;
    if FDetectedModel.Kind in [dkLanguage, dkImage] then
    begin
      var Error := FDetectedModel.Validate;
      if Error = '' then
        LabelModelValidation.Text := 'Комплект найден. Проверьте файлы и параметры на следующих шагах.'
      else
        LabelModelValidation.Text := 'Нужно дополнить комплект: ' + Error;
    end;
  finally
    FUpdating := False
  end;
end;

procedure TFormMain.ModelSourceChanged(Sender: TObject);
begin
  if not FReady or FUpdating then
    Exit;
  FModelInspectPending := True;
  TimerModelInspect.Enabled := False;
  TimerModelInspect.Enabled := True;
  ButtonWizardNext.Enabled := False;
end;

procedure TFormMain.ModelInspectTimer(Sender: TObject);
begin
  TimerModelInspect.Enabled := False;
  if FReady and not FUpdating then
    AnalyzeModelSource;
end;

procedure TFormMain.ModelBundleChanged(Sender: TObject);
begin
  if FReady and not FUpdating then
    AnalyzeModelSource(True);
end;

procedure TFormMain.WizardFilesChanged(Sender: TObject);
begin
  if not FReady or FUpdating then
    Exit;
  if FDetectedModel.Kind = dkImage then
    FDetectedModel.Image := WizardImageSettings
  else
    FDetectedModel.TokenizerDirectory := EditTokenizerDirectory.Text.Trim;
  LabelModelValidation.Text := FDetectedModel.Validate;
  if LabelModelValidation.Text = '' then
    LabelModelValidation.Text := 'Комплект готов к сохранению.';
  UpdateImageEditor;
end;

procedure TFormMain.WizardNextClick(Sender: TObject);
begin
  if FModelInspectPending then
    AnalyzeModelSource;
  if not (FDetectedModel.Kind in [dkLanguage, dkImage]) then
  begin
    LabelModelValidation.Text := FDetectedModel.Notice;
    Exit
  end;
  if FWizardStep = 1 then
  begin
    WizardFilesChanged(Sender);
    var Error := FDetectedModel.Validate;
    if Error <> '' then
    begin
      LabelModelValidation.Text := Error;
      Exit
    end;
  end;
  SetWizardStep(FWizardStep + 1);
end;

procedure TFormMain.WizardBackClick(Sender: TObject);
begin
  SetWizardStep(FWizardStep - 1);
end;

procedure TFormMain.BrowseModelComponent(Sender: TObject);
var
  Target: TEdit;
begin
  var Tag := TComponent(Sender).Tag;
  case Tag of
    0:
      Target := EditModelRoot;
    1:
      Target := EditModelTextEncoder;
    2:
      Target := EditModelTextEncoder2;
    3:
      Target := EditModelTokenizer;
    4:
      Target := EditModelTokenizer2;
  else
    Target := EditModelVAE
  end;
  // Component directories carry their own config and shard index. Qwen-Image
  // has separate single-file encoder/VAE checkpoints.
  if (Tag = 0) or ((FDetectedModel.Image.Kind <> ipQwenImage) and (Tag in [1, 2, 5]))
    or ((FDetectedModel.Kind = dkImage) and (FDetectedModel.Image.Kind = ipFlux1) and (Tag = 3)) then
  begin
    var Directory := ModelComponentDirectory(Target.Text, '');
    if SelectDirectory('Каталог компонента модели', '', Directory) then
      Target.Text := Directory;
  end
  else
  begin
    OpenDialogModelComponent.FileName := Target.Text;
    if OpenDialogModelComponent.Execute then
      Target.Text := OpenDialogModelComponent.FileName;
  end;
  if Tag = 0 then
    ModelBundleChanged(Target)
  else
    WizardFilesChanged(Target);
end;

procedure TFormMain.AddModelAdapter(Sender: TObject);
begin
  if not (mcLoRA in FDetectedModel.Capabilities) then
    Exit;
  if not OpenDialogModelAdapters.Execute then
    Exit;
  if ListBoxModelAdapters.Count + OpenDialogModelAdapters.Files.Count > 128 then
  begin
    LabelModelValidation.Text := 'Допускается до 128 адаптеров на модель.';
    Exit
  end;
  try
    for var FileName in OpenDialogModelAdapters.Files do
    begin
      var Error := AdapterValidation(FileName, FDetectedModel.Image.Kind);
      if Error <> '' then
      begin
        LabelModelValidation.Text := Error;
        Exit
      end;
      if ListBoxModelAdapters.Items.IndexOf(FileName) < 0 then
        ListBoxModelAdapters.Items.Add(FileName);
    end;
    WizardFilesChanged(Sender);
  except
    on E: Exception do
      LabelModelValidation.Text := E.Message
  end;
end;

procedure TFormMain.RemoveModelAdapter(Sender: TObject);
begin
  if ListBoxModelAdapters.ItemIndex >= 0 then
    ListBoxModelAdapters.Items.Delete(ListBoxModelAdapters.ItemIndex);
  WizardFilesChanged(Sender);
end;

procedure TFormMain.SelectModelEditor(const Id: string);
begin
  FEditingModelId := Id;
  FUpdating := True;
  try
    EditModelName.Text := '';
    EditModelDirectory.Text := '';
    EditModelRoot.Text := '';
    LabelModelEditorTitle.Text := 'Добавить модель';
    ButtonRemoveModel.Enabled := False;
    var Model := FStore.FindModel(Id);
    var ImageModel := FStore.FindImageModel(Id);
    if Model <> nil then
    begin
      EditModelName.Text := Model.Name;
      EditModelDirectory.Text := Model.SourceFile;
      if EditModelDirectory.Text = '' then
        EditModelDirectory.Text := Model.Directory;
    end
    else if ImageModel <> nil then
    begin
      EditModelName.Text := ImageModel.Name;
      EditModelDirectory.Text := ImageModel.Settings.SourceFile;
      if EditModelDirectory.Text = '' then
        EditModelDirectory.Text := ImageModel.Settings.DiffusionCheckpoint;
      if EditModelDirectory.Text = '' then
        EditModelDirectory.Text := ImageModel.Settings.Directory;
    end;
  finally
    FUpdating := False
  end;
  AnalyzeModelSource;
  FUpdating := True;
  try
    var Model := FStore.FindModel(Id);
    if Model <> nil then
    begin
      EditTokenizerDirectory.Text := Model.TokenizerDirectory;
      LabelModelEditorTitle.Text := Model.Name;
      ButtonRemoveModel.Enabled := True;
    end;
    var ImageModel := FStore.FindImageModel(Id);
    if ImageModel <> nil then
    begin
      FDetectedModel.Kind := dkImage;
      FDetectedModel.Image := ImageModel.Settings;
      LabelModelEditorTitle.Text := ImageModel.Name;
      ButtonRemoveModel.Enabled := True;
      EditModelRoot.Text := ImageModel.Settings.Directory;
      EditTokenizerDirectory.Text := ImageModel.Settings.AssetsDirectory;
      if ImageModel.Settings.Components.TextEncoder <> '' then
        EditModelTextEncoder.Text := ImageModel.Settings.Components.TextEncoder;
      if ImageModel.Settings.Components.TextEncoder2 <> '' then
        EditModelTextEncoder2.Text := ImageModel.Settings.Components.TextEncoder2;
      if ImageModel.Settings.Components.Tokenizer <> '' then
        EditModelTokenizer.Text := ImageModel.Settings.Components.Tokenizer;
      if ImageModel.Settings.Components.Tokenizer2 <> '' then
        EditModelTokenizer2.Text := ImageModel.Settings.Components.Tokenizer2;
      if ImageModel.Settings.Components.VAE <> '' then
        EditModelVAE.Text := ImageModel.Settings.Components.VAE;
      ListBoxModelAdapters.Items.Clear;
      for var FileName in ImageModel.Settings.Adapters do
        ListBoxModelAdapters.Items.Add(FileName);
      NumberBoxImageModelWidth.Value := ImageModel.Settings.Width;
      NumberBoxImageModelHeight.Value := ImageModel.Settings.Height;
      NumberBoxImageModelSteps.Value := ImageModel.Settings.Steps;
      NumberBoxImageModelSeed.Value := ImageModel.Settings.Seed;
      NumberBoxImageModelGuidance.Value := ImageModel.Settings.Guidance;
      CheckBoxImageModelInt8.IsChecked := ImageModel.Settings.UseInt8;
    end;
  finally
    FUpdating := False
  end;
  SetWizardStep(0);
  RenderModels;
end;

procedure TFormMain.ButtonNewModelClick(Sender: TObject);
begin
  SelectModelEditor('');
  EditModelDirectory.SetFocus;
end;

procedure TFormMain.ButtonBrowseImageModelClick(Sender: TObject);
begin
  SetActivePage(True);
  if ActiveChat <> nil then
    SelectModelEditor(ActiveChat.Preferences.ImageGeneration.ModelId);
end;

procedure TFormMain.ButtonBrowseModelClick(Sender: TObject);
begin
  OpenDialogModelSource.FileName := EditModelDirectory.Text;
  if OpenDialogModelSource.Execute then
  begin
    EditModelDirectory.Text := OpenDialogModelSource.FileName;
    AnalyzeModelSource
  end;
end;

procedure TFormMain.ButtonBrowseTokenizerClick(Sender: TObject);
begin
  var Directory := EditTokenizerDirectory.Text;
  if SelectDirectory('Каталог токенизатора и конфигураций', '', Directory) then
  begin
    EditTokenizerDirectory.Text := Directory;
    WizardFilesChanged(Sender)
  end;
end;

procedure TFormMain.SaveImageModel;
begin
  if FStore.FindModel(FEditingModelId) <> nil then
  begin
    LabelModelValidation.Text := 'Добавьте модель с другой архитектурой отдельной записью.';
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
    Candidate.Name := EditModelName.Text.Trim;
    Candidate.Settings := WizardImageSettings;
    for var FileName in Candidate.Settings.Adapters do
    begin
      var Error := AdapterValidation(FileName, Candidate.Settings.Kind);
      if Error <> '' then
      begin
        LabelModelValidation.Text := Error;
        Exit
      end;
    end;
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
      FStore.ImageModels.Add(Edited)
    end
    else
    begin
      Edited.Name := Candidate.Name;
      Edited.Settings := Candidate.Settings;
      Edited.Inspect
    end;
    if (ActiveChat <> nil) and (ActiveChat.Preferences.ImageGeneration.ModelId = '') then
      ActiveChat.Preferences.ImageGeneration := Edited.Resolve(Edited.Settings);
    FEditingModelId := Edited.Id;
    LoadPreferences;
    UpdateAvailability;
    MarkDirty;
    RenderModels;
    LabelModelValidation.Text := Edited.Validation + sLineBreak + 'Модель сохранена.';
  finally
    Candidate.Free
  end;
end;

procedure TFormMain.ButtonSaveModelClick(Sender: TObject);
begin
  try
    if FModelInspectPending then
      AnalyzeModelSource;
    if EditModelName.Text.Trim = '' then
      EditModelName.Text := FDetectedModel.SuggestedName;
    if EditModelName.Text.Trim = '' then
    begin
      LabelModelValidation.Text := 'Укажите название модели.';
      Exit
    end;
    if FDetectedModel.Kind = dkImage then
    begin
      SaveImageModel;
      Exit
    end;
    if FDetectedModel.Kind <> dkLanguage then
    begin
      LabelModelValidation.Text := FDetectedModel.Notice;
      Exit
    end;
    if FStore.FindImageModel(FEditingModelId) <> nil then
    begin
      LabelModelValidation.Text := 'Добавьте модель с другой архитектурой отдельной записью.';
      Exit
    end;
    for var Model in FStore.Models do
      if (Model.Id <> FEditingModelId) and SameText(Model.Name, EditModelName.Text.Trim) then
      begin
        LabelModelValidation.Text := 'Модель с таким названием уже есть.';
        Exit
      end;
    var Candidate := TModelEntry.Create;
    try
      Candidate.Name := EditModelName.Text.Trim;
      Candidate.Directory := FDetectedModel.Directory;
      Candidate.SourceFile := FDetectedModel.SourceFile;
      Candidate.TokenizerDirectory := EditTokenizerDirectory.Text.Trim;
      Candidate.Inspect;
      if not Candidate.Available then
      begin
        LabelModelValidation.Text := Candidate.Validation;
        Exit
      end;
      var Edited := FStore.FindModel(FEditingModelId);
      if Edited = nil then
      begin
        Edited := Candidate;
        Candidate := nil;
        FStore.Models.Add(Edited)
      end
      else
      begin
        Edited.Name := Candidate.Name;
        Edited.Directory := Candidate.Directory;
        Edited.SourceFile := Candidate.SourceFile;
        Edited.TokenizerDirectory := Candidate.TokenizerDirectory;
        Edited.Inspect;
      end;
      FEditingModelId := Edited.Id;
      if (ActiveChat <> nil) and (ActiveChat.Preferences.ModelId = '') then
        ActiveChat.Preferences.ModelId := Edited.Id;
      LoadPreferences;
      UpdateAvailability;
      MarkDirty;
      RenderModels;
      LabelModelValidation.Text := Edited.Validation + sLineBreak + 'Модель сохранена.';
    finally
      Candidate.Free
    end;
  except
    on E: Exception do
      LabelModelValidation.Text := E.Message
  end;
end;

procedure TFormMain.ButtonRemoveModelClick(Sender: TObject);
begin
  var Id := FEditingModelId;
  var IsImage := FStore.FindImageModel(Id) <> nil;
  if not IsImage and (FStore.FindModel(Id) = nil) then
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

procedure TFormMain.RecalculateSettings;
begin
  FitVerticalLayout(LayoutImagePreferences);
  var Model := FStore.FindModel(ActiveChat.Preferences.ModelId);
  FrameRoleSectionConversation.LabelSummary.Text := 'Нет доступных моделей';
  if Model <> nil then
    FrameRoleSectionConversation.LabelSummary.Text := Model.Name;
  Model := FStore.FindModel(ActiveChat.Preferences.ImageGeneration.AnalysisModelId);
  FrameRoleSectionVision.LabelSummary.Text := 'Нет доступных vision-моделей';
  if Model <> nil then
    FrameRoleSectionVision.LabelSummary.Text := Model.Name;
  var ImageModel := FStore.FindImageModel(ActiveChat.Preferences.ImageGeneration.ModelId);
  FrameRoleSectionImages.LabelSummary.Text := 'Нет доступных генераторов';
  if ImageModel <> nil then
    FrameRoleSectionImages.LabelSummary.Text := ImageModel.Name;
  FrameRoleSectionConversation.UpdateSize;
  FrameRoleSectionVision.UpdateSize;
  FrameRoleSectionImages.UpdateSize;
  FitVerticalLayout(LayoutPreferenceFields);
end;

procedure TFormMain.RoleExpandedChanged(Sender: TObject);
begin
  if FUpdating or not FReady or (ActiveChat = nil) then
    Exit;
  ActiveChat.Preferences.ChatSettingsExpanded := FrameRoleSectionConversation.Expanded;
  ActiveChat.Preferences.VisionSettingsExpanded := FrameRoleSectionVision.Expanded;
  ActiveChat.Preferences.ImageSettingsExpanded := FrameRoleSectionImages.Expanded;
  RecalculateSettings;
  MarkDirty;
end;

procedure TFormMain.ImageAdapterChanged(Sender: TObject);
begin
  if FUpdating or not FReady then
    Exit;
  if (ComboBoxImageAdapter.ItemIndex < 0) or (ComboBoxImageAdapter.ItemIndex >= FAdapterFiles.Count) then
    Exit;
  EditImageLoRAFile.Text := FAdapterFiles[ComboBoxImageAdapter.ItemIndex];
  EditImageLoRAFile.Hint := EditImageLoRAFile.Text;
  PreferencesChanged(Sender);
end;

procedure TFormMain.BrowseModelFolder(Sender: TObject);
begin
  var Directory := ExtractFileDir(EditModelDirectory.Text);
  if SelectDirectory('Каталог модели', '', Directory) then
  begin
    EditModelDirectory.Text := Directory;
    AnalyzeModelSource
  end;
end;

end.

