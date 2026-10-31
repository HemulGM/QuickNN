unit LLMUI.Message;

interface

uses
  System.SysUtils, System.Classes, System.Types, System.UITypes, FMX.Types,
  FMX.Controls, FMX.Forms, FMX.Layouts, FMX.StdCtrls, FMX.Objects, FMX.Graphics,
  FMX.Controls.Presentation, FMX.Dialogs, Markdown4D.Fmx.Viewer, LLMUI.Data,
  LLMUI.Images;

type
  TFrameMessage = class(TFrame)
    SaveDialogImage: TSaveDialog;
    ButtonNextImage: TButton;
    LayoutImageDetails: TLayout;
    ButtonOpenImage: TButton;
    ButtonSaveImage: TButton;
    LayoutImageActions: TLayout;
    LabelImageDetails: TLabel;
    ImageGenerated: TImage;
    LayoutImageCard: TLayout;
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
    TextUserMessage: TText;
    LayoutFooter: TLayout;
    LabelMetadata: TLabel;
    ButtonCopy: TButton;
    PathCopyIcon: TPath;
    ButtonContinue: TButton;
    ButtonRetry: TButton;
    PathRetryIcon: TPath;
    procedure ButtonNextImageClick(Sender: TObject);
    procedure ButtonOpenImageClick(Sender: TObject);
    procedure ButtonSaveImageClick(Sender: TObject);
    procedure ThinkingExpandedChanged(Sender: TObject);
    procedure FrameResize(Sender: TObject);
    procedure ButtonCopyClick(Sender: TObject);
    procedure ButtonRetryClick(Sender: TObject);
    procedure ButtonContinueClick(Sender: TObject);
  private
    FImages: TArray<TChatImage>;
    FImageIndex: Integer;
    FText, FMessageId, FImageFileName: string;
    FUser, FRendering, FStreaming: Boolean;
    FThinkingText: string;
    FLastWidth, FUserHeightWidth, FUserHeight: Single;
    FUserHeightText: string;
    FResizePending, FHeightDirty: Boolean;
    FMeasuredAnswerLayout, FMeasuredThinkingLayout: Integer;
    FMeasuredThinkingHeight: Single;
    FMarkdownViewer, FThinkingViewer: TMarkdownViewer;
    FOnRetry, FOnContinue, FOnHeightChanged: TNotifyEvent;
    procedure ShowAttachment;
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
    procedure SetMessage(const Message: TChatMessage; const CanRetry: Boolean; const CanContinue: Boolean = False);
    procedure UpdateActions(const CanRetry, CanContinue: Boolean);
    function SyncContentHeight(const AllowResize: Boolean = True): Boolean;
    property MessageId: string read FMessageId;
    property MarkdownViewer: TMarkdownViewer read FMarkdownViewer;
    property ThinkingViewer: TMarkdownViewer read FThinkingViewer;
    property OnHeightChanged: TNotifyEvent read FOnHeightChanged write FOnHeightChanged;
    property OnRetry: TNotifyEvent read FOnRetry write FOnRetry;
    property OnContinue: TNotifyEvent read FOnContinue write FOnContinue;
  end;

implementation

uses
  System.Math, System.IOUtils, FMX.TextLayout, Winapi.Windows, Winapi.ShellAPI,
  QuickNN.Models.LLM.Types, LLMUI.Controls, LLMUI.Markdown, LLMUI.ImageViewer;

{$R *.fmx}

constructor TFrameMessage.Create(AOwner: TComponent);
begin
  inherited;
  FMarkdownViewer := CreateChatMarkdownViewer(Self, LayoutBody);
  FMarkdownViewer.OnResolveImage := ResolveImage;
  FMarkdownViewer.OnLinkClick := OpenLink;
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
var
  TextLayout: TTextLayout;
begin
  if not FUser then
  begin
    if FMarkdownViewer.Visible then
      Exit(Max(24, FMarkdownViewer.ContentHeight + 2))
    else
      Exit(0);
  end;
  if (FUserHeightText = FText) and SameValue(FUserHeightWidth, TextUserMessage.Width, 0.1) then
    Exit(FUserHeight);
  TextLayout := TTextLayoutManager.DefaultTextLayout.Create;
  try
    TextLayout.BeginUpdate;
    TextLayout.Font.Family := 'Segoe UI';
    TextLayout.Font.Size := 16;
    TextLayout.MaxSize := PointF(Max(80, TextUserMessage.Width), 100000);
    TextLayout.WordWrap := True;
    TextLayout.Text := FText;
    TextLayout.EndUpdate;
    Result := Max(24, Ceil(TextLayout.TextHeight) + 4);
    FUserHeightText := FText;
    FUserHeightWidth := TextUserMessage.Width;
    FUserHeight := Result;
  finally
    TextLayout.Free
  end;
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
  if LayoutImageCard.Visible then
    LayoutImageCard.Height := 60 + Min(380, Max(120, LayoutMessage.Width * ImageGenerated.Bitmap.Height / Max(1, ImageGenerated.Bitmap.Width)));
  DesiredBody := BodyHeight;
  DesiredFrame :=
    DesiredBody
    + Ord(LabelAuthor.Visible) * LabelAuthor.Height
    + LayoutFooter.Height
    + LayoutMessage.Padding.Top
    + LayoutMessage.Padding.Bottom;
  if LayoutImageCard.Visible then
    DesiredFrame := DesiredFrame + LayoutImageCard.Height + LayoutImageCard.Margins.Bottom;
  if ExpanderThinking.Visible then
    DesiredFrame := DesiredFrame + ExpanderThinking.Height + ExpanderThinking.Margins.Bottom;
  Result := not SameValue(LayoutBody.Height, DesiredBody) or not SameValue(Height, DesiredFrame);
  LayoutBody.Height := DesiredBody;
  Height := DesiredFrame;
  FMeasuredThinkingHeight := ExpanderThinking.Height;
  if not FUser then
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
  LayoutChanged := not FUser and (FMeasuredAnswerLayout <> FMarkdownViewer.LayoutCount);
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
    TextUserMessage.Visible := FUser;
    FMarkdownViewer.Visible := not FUser;
    ExpanderThinking.Visible := False;
    Realign;
    if FUser then
      TextUserMessage.Text := FText
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
    FImageIndex := 0
  end;
  FMessageId := Message.Id;
  FImages := Copy(Message.Images);
  if FImageIndex >= Length(FImages) then
    FImageIndex := 0;
  ShowAttachment;
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

procedure TFrameMessage.ShowAttachment;
begin
  var NewImage := '';
  if Length(FImages) > 0 then
    NewImage := FImages[FImageIndex].FileName;
  if FImageFileName <> NewImage then
  begin
    ImageGenerated.Bitmap.Clear(0);
    FImageFileName := NewImage;
    if TFile.Exists(FImageFileName) then
    try
      ImageGenerated.Bitmap.LoadThumbnailFromFile(FImageFileName, 1024, 1024, False)
    except
      ImageGenerated.Bitmap.Clear(0)
    end;
  end;
  LayoutImageCard.Visible := Length(FImages) > 0;
  ButtonNextImage.Visible := Length(FImages) > 1;
  ButtonNextImage.Text := Format('%d/%d · далее', [FImageIndex + 1, Length(FImages)]);
  if LayoutImageCard.Visible then
  begin
    var Image := FImages[FImageIndex];
    if Image.IsUserImage then
      LabelImageDetails.Text := Format('%s · %d×%d', [Image.OriginalName, Image.Settings.Width, Image.Settings.Height])
    else
      LabelImageDetails.Text := Format('%s · %d×%d · seed %d', [Image.ModelName, Image.Settings.Width, Image.Settings.Height, Image.Settings.Seed]);
    if not TFile.Exists(FImageFileName) then
      LabelImageDetails.Text := 'Файл изображения недоступен · ' + LabelImageDetails.Text;
  end;
end;

procedure TFrameMessage.ButtonNextImageClick(Sender: TObject);
begin
  if Length(FImages) < 2 then
    Exit;
  FImageIndex := (FImageIndex + 1) mod Length(FImages);
  ShowAttachment;
  RenderBody;
end;

procedure TFrameMessage.ButtonOpenImageClick(Sender: TObject);
begin
  var DialogOwner: TComponent := Application;
  if Root <> nil then
    DialogOwner := Root.GetObject;
  TChatImageViewer.Open(DialogOwner, FImages, FImageIndex);
end;

procedure TFrameMessage.ButtonSaveImageClick(Sender: TObject);
begin
  // A modal picker pumps UI updates: virtualization may recycle this frame.
  var Source := FImageFileName;
  if not TFile.Exists(Source) then
    Exit;
  var DialogOwner: TComponent := Application;
  if Root <> nil then
    DialogOwner := Root.GetObject;
  var Dialog := TSaveDialog.Create(DialogOwner);
  try
    // Copy settings before Execute: the frame can be retired during its modal loop.
    Dialog.DefaultExt := SaveDialogImage.DefaultExt;
    Dialog.Filter := SaveDialogImage.Filter;
    Dialog.Options := SaveDialogImage.Options;
    Dialog.Title := 'Сохранить изображение';
    Dialog.FileName := ExtractFileName(Source);
    if Dialog.Execute and not SameFileName(Dialog.FileName, Source) then
      TFile.Copy(Source, Dialog.FileName, True);
  finally
    Dialog.Free;
  end;
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

