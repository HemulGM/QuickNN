unit LLMUI.Message;

interface

uses
  System.SysUtils, System.Classes, System.Types, System.UITypes, FMX.Types,
  FMX.Controls, FMX.Forms, FMX.Layouts, FMX.StdCtrls, FMX.Objects, FMX.Graphics,
  FMX.Controls.Presentation, Markdown4D.Fmx.Viewer, LLMUI.Data;

type
  TFrameMessage = class(TFrame)
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
    procedure ThinkingExpandedChanged(Sender: TObject);
    procedure FrameResize(Sender: TObject);
    procedure ButtonCopyClick(Sender: TObject);
    procedure ButtonRetryClick(Sender: TObject);
    procedure ButtonContinueClick(Sender: TObject);
  private
    FText, FMessageId: string;
    FUser, FRendering, FStreaming: Boolean;
    FThinkingText: string;
    FLastWidth: Single;
    FMarkdownViewer, FThinkingViewer: TMarkdownViewer;
    FOnRetry, FOnContinue: TNotifyEvent;
    procedure RenderBody;
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
    function SyncContentHeight: Boolean;
    property MessageId: string read FMessageId;
    property MarkdownViewer: TMarkdownViewer read FMarkdownViewer;
    property ThinkingViewer: TMarkdownViewer read FThinkingViewer;
    property OnRetry: TNotifyEvent read FOnRetry write FOnRetry;
    property OnContinue: TNotifyEvent read FOnContinue write FOnContinue;
  end;

implementation

uses
  System.Math, FMX.TextLayout, Winapi.Windows, Winapi.ShellAPI,
  QuickNN.Models.LLM.Types, LLMUI.Controls, LLMUI.Markdown;

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
  finally
    TextLayout.Free
  end;
end;

function TFrameMessage.ApplyContentHeight: Boolean;
var
  DesiredBody, DesiredFrame: Single;
begin
  if ExpanderThinking.Visible then
    if (FThinkingViewer <> nil) and ExpanderThinking.IsExpanded then
      ExpanderThinking.ContentHeight :=
        FThinkingViewer.ContentHeight + LayoutThinkingBody.Padding.Top + LayoutThinkingBody.Padding.Bottom + 2;
  DesiredBody := BodyHeight;
  DesiredFrame :=
    DesiredBody
    + Ord(LabelAuthor.Visible) * LabelAuthor.Height
    + LayoutFooter.Height
    + LayoutMessage.Padding.Top
    + LayoutMessage.Padding.Bottom;
  if ExpanderThinking.Visible then
    DesiredFrame := DesiredFrame + ExpanderThinking.Height + ExpanderThinking.Margins.Bottom;
  Result := not SameValue(LayoutBody.Height, DesiredBody) or not SameValue(Height, DesiredFrame);
  LayoutBody.Height := DesiredBody;
  Height := DesiredFrame;
  if not FUser then
    TChatMarkdownViewer(FMarkdownViewer).FitContentHeight;
  if FThinkingViewer <> nil then
    TChatMarkdownViewer(FThinkingViewer).FitContentHeight;
end;

function TFrameMessage.SyncContentHeight: Boolean;
begin
  Result := False;
  if FUser or FRendering or (FMarkdownViewer = nil) then
    Exit;
  FRendering := True;
  try
    Result := ApplyContentHeight
  finally
    FRendering := False
  end;
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
  SyncContentHeight;
end;

procedure TFrameMessage.RenderBody;
var
  Parts: TChatMarkdownParts;
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
    FLastWidth := Width;
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
      FMarkdownViewer.Visible := (Parts.Answer <> '') or not Parts.HasThinking;
      if FMarkdownViewer.Visible then
        UpdateMarkdown(FMarkdownViewer, Parts.Answer);
      if Parts.HasThinking and ExpanderThinking.IsExpanded then
        UpdateThinking;
    end;
    ApplyContentHeight;
  finally
    FRendering := False
  end;
end;

procedure TFrameMessage.SetMessage(const Message: TChatMessage; const CanRetry: Boolean; const CanContinue: Boolean);
begin
  if FMessageId <> Message.Id then
    ExpanderThinking.IsExpanded := False;
  FMessageId := Message.Id;
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
  if (LayoutBody <> nil) and not (csLoading in ComponentState) and (Abs(FLastWidth - Width) > 1) then
    RenderBody;
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

