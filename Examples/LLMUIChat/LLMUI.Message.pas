unit LLMUI.Message;

interface

uses
  System.SysUtils, System.Classes, System.Types, System.UITypes, FMX.Types,
  FMX.Controls, FMX.Forms, FMX.Layouts, FMX.StdCtrls, FMX.Objects, FMX.Graphics,
  FMX.Controls.Presentation, FMX.TextLayout, Markdown4D.Fmx.Viewer, LLMUI.Data,
  LLMUI.Images, LLMUI.Attachment;

type
  TFrameMessage = class(TFrame)
    LayoutContent: TLayout;
    LayoutBubble: TLayout;
    LayoutAttachments: TLayout;
    FlowLayoutAttachments: TFlowLayout;
    LayoutRow: TLayout;
    LayoutAvatar: TLayout;
    CircleAvatar: TCircle;
    PathUserIcon: TPath;
    PanelBubble: TPanel;
    LayoutMessage: TLayout;
    LabelAuthor: TLabel;
    ExpanderThinking: TExpander;
    LayoutThinkingBody: TLayout;
    LayoutBody: TLayout;
    LayoutFooter: TLayout;
    LayoutNegativePrompt: TLayout;
    RectangleNegativePrompt: TRectangle;
    LayoutNegativeHeader: TLayout;
    ButtonToggleNegativePrompt: TButton;
    LayoutNegativeCaption: TLayout;
    LayoutNegativeChevron: TLayout;
    PathNegativeChevron: TPath;
    LabelNegativeCaption: TLabel;
    ButtonCopyNegativePrompt: TButton;
    PathNegativeCopy: TPath;
    LayoutNegativeBody: TLayout;
    LabelNegativePrompt: TLabel;
    LabelMetadata: TLabel;
    ButtonCopy: TButton;
    PathCopyIcon: TPath;
    ButtonContinue: TButton;
    ButtonRetry: TButton;
    PathRetryIcon: TPath;
    procedure ThinkingExpandedChanged(Sender: TObject);
    procedure FrameResize(Sender: TObject);
    procedure ButtonCopyClick(Sender: TObject);
    procedure ToggleNegativePromptClick(Sender: TObject);
    procedure CopyNegativePromptClick(Sender: TObject);
    procedure ButtonRetryClick(Sender: TObject);
    procedure ButtonContinueClick(Sender: TObject);
  private
    FImages: TArray<TChatImage>;
    FAttachments: TArray<TFrameAttachment>;
    FText, FMessageId: string;
    FUser, FRendering, FStreaming: Boolean;
    FThinkingText, FNegativePrompt: string;
    FNegativeExpanded: Boolean;
    FNegativeTextLayout: TTextLayout;
    FLastWidth: Single;
    FResizePending, FHeightDirty: Boolean;
    FMeasuredAnswerLayout, FMeasuredThinkingLayout: Integer;
    FMeasuredThinkingHeight: Single;
    FMarkdownViewer, FThinkingViewer: TMarkdownViewer;
    FOnRetry, FOnContinue, FOnHeightChanged: TNotifyEvent;
    procedure SetNegativePromptExpanded(const Value: Boolean);
    procedure UpdateNegativePromptHeight;
    procedure ShowAttachments;
    procedure RenderBody;
    procedure ApplyContentWidth;
    procedure UpdateMarkdown(const Viewer: TMarkdownViewer; const Value: string);
    procedure UpdateThinking;
    procedure ResolveImage(const Sender: TObject; const Url: string; const Bitmap: FMX.Graphics.TBitmap; var Handled: Boolean);
    procedure OpenLink(const Sender: TObject; const Url: string);
    function BodyHeight: Single;
    function ApplyContentHeight: Boolean;
  public
    constructor Create(AOwner: TComponent); override;
    destructor Destroy; override;
    procedure SetMessage(const Message: TChatMessage; const CanRetry: Boolean; const CanContinue: Boolean = False);
    procedure UpdateActions(const CanRetry, CanContinue: Boolean);
    function SyncContentHeight(const AllowResize: Boolean = True): Boolean;
    property MessageId: string read FMessageId;
    property NegativePromptExpanded: Boolean read FNegativeExpanded write SetNegativePromptExpanded;
    property MarkdownViewer: TMarkdownViewer read FMarkdownViewer;
    property ThinkingViewer: TMarkdownViewer read FThinkingViewer;
    property OnHeightChanged: TNotifyEvent read FOnHeightChanged write FOnHeightChanged;
    property OnRetry: TNotifyEvent read FOnRetry write FOnRetry;
    property OnContinue: TNotifyEvent read FOnContinue write FOnContinue;
  end;

implementation

uses
  System.Math, Winapi.Windows, Winapi.ShellAPI,
  QuickNN.Models.LLM.Types, LLMUI.Controls, LLMUI.Markdown;

{$R *.fmx}

constructor TFrameMessage.Create(AOwner: TComponent);
begin
  inherited;
  FMarkdownViewer := CreateChatMarkdownViewer(Self, LayoutBody);
  FMarkdownViewer.OnResolveImage := ResolveImage;
  FMarkdownViewer.OnLinkClick := OpenLink;
end;

destructor TFrameMessage.Destroy;
begin
  FNegativeTextLayout.Free;
  inherited;
end;

procedure TFrameMessage.UpdateNegativePromptHeight;
begin
  LayoutNegativePrompt.Visible := FUser and (FNegativePrompt.Trim <> '');
  LayoutNegativeBody.Visible := LayoutNegativePrompt.Visible and FNegativeExpanded;
  var TextHeight: Single := 0;
  if LayoutNegativeBody.Visible then
  begin
    // Plain native text keeps angle brackets and prompt punctuation literal.
    // Allocate and measure only when expanded, never for collapsed virtual rows.
    if FNegativeTextLayout = nil then FNegativeTextLayout := TTextLayoutManager.DefaultTextLayout.Create;
    var TextWidth := Max(1, LayoutNegativePrompt.Width - LayoutNegativeBody.Padding.Left
      - LayoutNegativeBody.Padding.Right - 2);
    if (FNegativeTextLayout.Text <> FNegativePrompt)
      or not SameValue(FNegativeTextLayout.MaxSize.X, TextWidth, 0.1) then
    begin
      FNegativeTextLayout.BeginUpdate;
      try
        FNegativeTextLayout.Font.Assign(LabelNegativePrompt.TextSettings.Font);
        FNegativeTextLayout.WordWrap := True;
        FNegativeTextLayout.HorizontalAlign := TTextAlign.Leading;
        FNegativeTextLayout.VerticalAlign := TTextAlign.Leading;
        FNegativeTextLayout.Trimming := TTextTrimming.None;
        FNegativeTextLayout.MaxSize := PointF(TextWidth, 1000000);
        FNegativeTextLayout.Text := FNegativePrompt;
      finally FNegativeTextLayout.EndUpdate end;
    end;
    TextHeight := Max(20, Ceil(FNegativeTextLayout.TextHeight + 2));
  end;
  LayoutNegativeBody.Height := TextHeight + LayoutNegativeBody.Padding.Top + LayoutNegativeBody.Padding.Bottom;
  LayoutNegativePrompt.Height := LayoutNegativeHeader.Height;
  if LayoutNegativeBody.Visible then
    LayoutNegativePrompt.Height := LayoutNegativePrompt.Height + LayoutNegativeBody.Height;
  if FNegativeExpanded then
  begin
    PathNegativeChevron.RotationAngle := 270;
    ButtonToggleNegativePrompt.Hint := 'Свернуть негативный промпт';
  end
  else
  begin
    PathNegativeChevron.RotationAngle := 180;
    ButtonToggleNegativePrompt.Hint := 'Показать использованный негативный промпт';
  end;
end;

procedure TFrameMessage.SetNegativePromptExpanded(const Value: Boolean);
begin
  var Expanded := Value and FUser and (FNegativePrompt.Trim <> '');
  if FNegativeExpanded = Expanded then Exit;
  FNegativeExpanded := Expanded;
  FHeightDirty := True;
  SyncContentHeight;
end;

procedure TFrameMessage.ToggleNegativePromptClick(Sender: TObject);
begin
  NegativePromptExpanded := not NegativePromptExpanded;
end;

procedure TFrameMessage.CopyNegativePromptClick(Sender: TObject);
begin
  if FNegativePrompt = '' then Exit;
  CopyText(FNegativePrompt);
  ButtonCopyNegativePrompt.Hint := 'Скопировано';
end;

procedure TFrameMessage.ResolveImage(const Sender: TObject; const Url: string; const Bitmap: FMX.Graphics.TBitmap; var Handled: Boolean);
begin
  // Model text does not implicitly authorize opening local files or fetching remote images.
  Handled := True;
end;

procedure TFrameMessage.OpenLink(const Sender: TObject; const Url: string);
begin
  if Url.StartsWith('https://', True) or Url.StartsWith('http://', True) then
    ShellExecute(0, 'open', PChar(Url), nil, nil, SW_SHOWNORMAL);
end;

function TFrameMessage.BodyHeight: Single;
begin
  if FMarkdownViewer.Visible then Result := Max(24, FMarkdownViewer.ContentHeight + 2)
  else Result := 0;
end;

function TFrameMessage.ApplyContentHeight: Boolean;
var
  DesiredBody, DesiredFrame: Single;
begin
  FHeightDirty := False;
  FMeasuredAnswerLayout := FMarkdownViewer.LayoutCount;
  FMeasuredThinkingLayout := 0;
  if FThinkingViewer <> nil then
    FMeasuredThinkingLayout := FThinkingViewer.LayoutCount;
  if ExpanderThinking.Visible then
    if (FThinkingViewer <> nil) and ExpanderThinking.IsExpanded then
      ExpanderThinking.ContentHeight :=
        FThinkingViewer.ContentHeight + LayoutThinkingBody.Padding.Top + LayoutThinkingBody.Padding.Bottom + 2;
  if LayoutAttachments.Visible then
  begin
    for var Attachment in FAttachments do
      Attachment.Width := Min(MESSAGE_ATTACHMENT_WIDTH, Max(1, FlowLayoutAttachments.Width));
    LayoutAttachments.Height := MessageAttachmentHeight(Length(FAttachments), FlowLayoutAttachments.Width);
  end;
  UpdateNegativePromptHeight;
  DesiredBody := BodyHeight;
  DesiredFrame :=
    DesiredBody
    + Ord(LabelAuthor.Visible) * LabelAuthor.Height
    + LayoutFooter.Height
    + LayoutMessage.Padding.Top
    + LayoutMessage.Padding.Bottom;
  if LayoutAttachments.Visible then
    DesiredFrame := DesiredFrame + LayoutAttachments.Height + LayoutAttachments.Margins.Bottom;
  if LayoutNegativePrompt.Visible then
    DesiredFrame := DesiredFrame + LayoutNegativePrompt.Height + LayoutNegativePrompt.Margins.Top;
  if ExpanderThinking.Visible then
    DesiredFrame := DesiredFrame + ExpanderThinking.Height + ExpanderThinking.Margins.Bottom;
  Result := not SameValue(LayoutBody.Height, DesiredBody) or not SameValue(Height, DesiredFrame);
  LayoutBody.Height := DesiredBody;
  Height := DesiredFrame;
  FMeasuredThinkingHeight := ExpanderThinking.Height;
  TChatMarkdownViewer(FMarkdownViewer).FitContentHeight;
  if FThinkingViewer <> nil then
    TChatMarkdownViewer(FThinkingViewer).FitContentHeight;
end;

procedure TFrameMessage.ApplyContentWidth;
begin
  FResizePending := False;
  FLastWidth := Min(Width, CHAT_CONTENT_MAX_WIDTH);
  LayoutRow.Width := FLastWidth;
  if FUser then
    LayoutRow.Padding.Left := Max(32, LayoutRow.Width * 0.13);
  FHeightDirty := True;
end;

function TFrameMessage.SyncContentHeight(const AllowResize: Boolean): Boolean;
var
  WidthChanged, LayoutChanged: Boolean;
begin
  Result := False;
  if FRendering or (FMarkdownViewer = nil) then
    Exit;
  WidthChanged := FResizePending and AllowResize;
  LayoutChanged := FMeasuredAnswerLayout <> FMarkdownViewer.LayoutCount;
  if FThinkingViewer <> nil then
    LayoutChanged := LayoutChanged or (FMeasuredThinkingLayout <> FThinkingViewer.LayoutCount);
  if not WidthChanged and not LayoutChanged and not FHeightDirty
    and SameValue(FMeasuredThinkingHeight, ExpanderThinking.Height, 0.1) then
    Exit;
  FRendering := True;
  try
    if WidthChanged then
      ApplyContentWidth;
    Result := ApplyContentHeight;
  finally
    FRendering := False;
  end;
  if Result and Assigned(FOnHeightChanged) then
    FOnHeightChanged(Self);
end;

procedure TFrameMessage.UpdateMarkdown(const Viewer: TMarkdownViewer; const Value: string);
var
  Markdown, Previous: string;
begin
  Markdown := Value;
  if Markdown = '' then
    Markdown := '…';
  Previous := Viewer.Text;
  if Markdown = Previous then
    Exit;
  if (Previous <> '') and (Previous <> '…') and Markdown.StartsWith(Previous) then
    Viewer.AppendMarkdown(Copy(Markdown, Length(Previous) + 1, MaxInt))
  else
    Viewer.Text := Markdown;
end;

procedure TFrameMessage.UpdateThinking;
begin
  if FThinkingViewer = nil then
  begin
    FThinkingViewer := CreateChatMarkdownViewer(Self, LayoutThinkingBody, 'MarkdownViewerThinking');
    FThinkingViewer.Theme.BackgroundColor := $00121D19;
    FThinkingViewer.Theme.TextColor := $FFA8BBB3;
    FThinkingViewer.OnResolveImage := ResolveImage;
    FThinkingViewer.OnLinkClick := OpenLink;
  end;
  UpdateMarkdown(FThinkingViewer, FThinkingText);
end;

procedure TFrameMessage.ThinkingExpandedChanged(Sender: TObject);
begin
  if FRendering or (FMarkdownViewer = nil) then
    Exit;
  if ExpanderThinking.IsExpanded then
    UpdateThinking;
  FHeightDirty := True;
  SyncContentHeight;
end;

procedure TFrameMessage.RenderBody;
var
  Parts: TChatMarkdownParts;
  HeightChanged: Boolean;
begin
  if FRendering or (FMarkdownViewer = nil) then
    Exit;
  FRendering := True;
  try
    LayoutRow.Width := Min(Width, CHAT_CONTENT_MAX_WIDTH);
    if FUser then
    begin
      LayoutRow.Padding.Left := Max(32, LayoutRow.Width * 0.13);
      LayoutRow.Padding.Right := 0;
      LayoutMessage.Padding.Rect := TRectF.Create(18, 14, 18, 12);
      LayoutAvatar.Align := TAlignLayout.Right;
      LayoutAvatar.Margins.Rect := TRectF.Create(10, 0, 0, 0);
    end
    else
    begin
      LayoutRow.Padding.Left := 0;
      LayoutRow.Padding.Right := 0;
      LayoutMessage.Padding.Rect := TRectF.Create(0, 8, 0, 8);
      LayoutAvatar.Align := TAlignLayout.Left;
      LayoutAvatar.Margins.Rect := TRectF.Create(0, 0, 10, 0);
    end;
    FLastWidth := LayoutRow.Width;
    FResizePending := False;
    if FUser then
    begin
      FMarkdownViewer.Theme.BackgroundColor := $00000000;
      LayoutAttachments.Padding.Right := 46;
      FlowLayoutAttachments.Justify := TFlowJustify.Right;
      FlowLayoutAttachments.JustifyLastLine := TFlowJustify.Right;
    end
    else
    begin
      FMarkdownViewer.Theme.BackgroundColor := $FF0F1416;
      LayoutAttachments.Padding.Right := 0;
      FlowLayoutAttachments.Justify := TFlowJustify.Left;
      FlowLayoutAttachments.JustifyLastLine := TFlowJustify.Left;
    end;
    ExpanderThinking.Visible := False;
    Realign;
    if FUser then
    begin
      FThinkingText := '';
      FMarkdownViewer.Visible := FText <> '';
      if FMarkdownViewer.Visible then UpdateMarkdown(FMarkdownViewer, FText);
    end
    else
    begin
      Parts := SplitThinkingMarkdown(FText, FStreaming);
      FThinkingText := Parts.Thinking;
      ExpanderThinking.Visible := Parts.HasThinking;
      if Parts.ThinkingOpen and FStreaming then
        ExpanderThinking.Text := 'Размышляет…'
      else
        ExpanderThinking.Text := 'Размышления';
      FMarkdownViewer.Visible := (Parts.Answer <> '') or
        (not Parts.HasThinking and ((Length(FImages) = 0) or FStreaming));
      if FMarkdownViewer.Visible then
        UpdateMarkdown(FMarkdownViewer, Parts.Answer);
      if Parts.HasThinking and ExpanderThinking.IsExpanded then
        UpdateThinking;
    end;
    HeightChanged := ApplyContentHeight;
  finally
    FRendering := False
  end;
  if HeightChanged and Assigned(FOnHeightChanged) then
    FOnHeightChanged(Self);
end;

procedure TFrameMessage.SetMessage(const Message: TChatMessage; const CanRetry: Boolean; const CanContinue: Boolean);
begin
  if FMessageId <> Message.Id then
  begin
    ExpanderThinking.IsExpanded := False;
    FNegativeExpanded := False;
  end;
  FMessageId := Message.Id;
  if FNegativePrompt <> Message.NegativePrompt then
    ButtonCopyNegativePrompt.Hint := 'Копировать негативный промпт';
  FNegativePrompt := Message.NegativePrompt;
  if FNegativePrompt.Trim = '' then FNegativeExpanded := False;
  LabelNegativePrompt.Text := FNegativePrompt;
  ButtonCopyNegativePrompt.Enabled := FNegativePrompt <> '';
  FImages := Copy(Message.Images);
  ShowAttachments;
  FStreaming := Message.State = msStreaming;
  FText := Message.Text;
  FUser := Message.Role = lrUser;
  LabelAuthor.Visible := not FUser;
  if FUser then
  begin
    LabelAuthor.Text := 'Вы';
    CircleAvatar.Fill.Color := $FF233E38;
  end
  else
  begin
    LabelAuthor.Text := Message.ModelName;
    if LabelAuthor.Text = '' then
      LabelAuthor.Text := 'Ассистент';
    CircleAvatar.Fill.Color := $FF164638;
  end;
  PathUserIcon.Visible := FUser;
  LayoutAvatar.Visible := FUser;
  PanelBubble.Visible := FUser;
  UpdateActions(CanRetry, CanContinue);
  if Message.State = msStreaming then
    LabelMetadata.Text := 'Генерация…'
  else if Message.State = msFailed then
  begin
    LabelMetadata.Text := 'Не удалось завершить ответ';
    if FText = '' then
      FText := Message.ErrorText;
  end
  else if Message.State = msLimited then
    LabelMetadata.Text := 'Достигнут лимит · ' + Message.Tokens.ToString + ' токенов'
  else if Message.State = msContextFull then
    LabelMetadata.Text := 'Контекст заполнен'
  else if Message.State = msStopped then
    LabelMetadata.Text := 'Остановлено'
  else if not FUser and (Message.Tokens > 0) then
    LabelMetadata.Text :=
      Format(
      '%d токенов · %.1f ток/с · %.2f с до ответа',
      [Message.Tokens, Message.DecodeRate, Message.FirstTokenSeconds]
    )
  else
    LabelMetadata.Text := FormatDateTime('hh:nn', Message.Created);
  ButtonCopy.Enabled := FText <> '';
  RenderBody;
end;

procedure TFrameMessage.FrameResize(Sender: TObject);
begin
  if (LayoutBody <> nil) and not (csLoading in ComponentState) then
    // The form batches width changes; resizing a frame never reparses its text.
    FResizePending := not SameValue(FLastWidth, Min(Width, CHAT_CONTENT_MAX_WIDTH), 0.1);
end;

procedure TFrameMessage.ShowAttachments;
begin
  while Length(FAttachments) > Length(FImages) do
  begin
    var Attachment := FAttachments[High(FAttachments)];
    SetLength(FAttachments, Length(FAttachments) - 1);
    Attachment.OnRemove := nil;
    RetireControl(Attachment);
  end;
  while Length(FAttachments) < Length(FImages) do
  begin
    var Attachment := TFrameAttachment.Create(Self);
    Attachment.Name := 'FrameAttachmentImage' + Length(FAttachments).ToString;
    Attachment.Parent := FlowLayoutAttachments;
    Attachment.Height := MESSAGE_ATTACHMENT_HEIGHT;
    FAttachments := FAttachments + [Attachment];
  end;
  for var I := 0 to High(FAttachments) do FAttachments[I].SetHistoryImages(FImages, I);
  LayoutAttachments.Visible := Length(FAttachments) > 0;
end;

procedure TFrameMessage.ButtonCopyClick(Sender: TObject);
begin
  CopyText(FText);
  ButtonCopy.Hint := 'Скопировано';
end;

procedure TFrameMessage.UpdateActions(const CanRetry, CanContinue: Boolean);
begin
  ButtonRetry.Visible := CanRetry;
  ButtonContinue.Visible := CanContinue;
end;

procedure TFrameMessage.ButtonContinueClick(Sender: TObject);
begin
  if Assigned(FOnContinue) then
    FOnContinue(Self);
end;

procedure TFrameMessage.ButtonRetryClick(Sender: TObject);
begin
  if Assigned(FOnRetry) then
    FOnRetry(Self);
end;

end.

