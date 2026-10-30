unit LLMUI.Markdown;

interface

uses
  System.Classes, FMX.Controls, Markdown4D.Fmx.Viewer;

type
  // The outer chat owns scrolling. Spare viewport height is clipped by the
  // message body, so asynchronous Markdown layout has no temporary scroll range.
  TChatMarkdownViewer = class(TMarkdownViewer)
  private
    const
      CONTENT_HEIGHT_RESERVE = 128;
    var
      FAdjustingBounds: Boolean;
    procedure InnerScrollChanged(Sender: TObject);
  protected
    procedure Paint; override;
  public
    constructor Create(AOwner: TComponent); override;
    procedure FitContentHeight;
  end;

  TChatMarkdownParts = record
    Answer, Thinking: string;
    HasThinking, ThinkingOpen: Boolean;
  end;

function SplitThinkingMarkdown(const Source: string; const Streaming: Boolean): TChatMarkdownParts;

function CreateChatMarkdownViewer(const Owner: TComponent; const Parent: TControl; const ViewerName: string = 'MarkdownViewerResponse'
        ): TMarkdownViewer;

implementation

uses
  System.SysUtils, System.Math, FMX.Types, Markdown4D.Theme,
  Markdown4D.Layout.Interfaces, Markdown4D.Highlighter.Interfaces,
  Markdown4D.Highlighter.LineScanner;

type
  TChatCodeScanner = class(TSyntaxLineScanner)
  private
    const
      BLOCK_COMMENT = 1;
      TRIPLE_SINGLE = 2;
      TRIPLE_DOUBLE = 3;
      TEMPLATE_STRING = 4;
    procedure ReadString(const Quote: Char; const Triple: Boolean; const Continuing: Boolean);
    procedure ReadBlockComment;
  protected
    function Python: Boolean; virtual;
    procedure Resume; override;
    procedure ScanNext; override;
    function IsKeyword(const Identifier: string): Boolean; override;
  end;

  TPythonCodeScanner = class(TChatCodeScanner)
  protected
    function Python: Boolean; override;
  end;

  TChatCodeHighlighter = class(TSyntaxHighlighter)
  protected
    function ScannerClass: TSyntaxLineScannerClass; override;
  end;

  TPythonCodeHighlighter = class(TSyntaxHighlighter)
  protected
    function ScannerClass: TSyntaxLineScannerClass; override;
  end;

constructor TChatMarkdownViewer.Create(AOwner: TComponent);
begin
  inherited;
  OnScroll := InnerScrollChanged;
end;

procedure TChatMarkdownViewer.FitContentHeight;
var
  DesiredHeight: Single;
begin
  if FAdjustingBounds then
    Exit;
  FAdjustingBounds := True;
  try
    // The first height change may finish a queued width reflow.
    for var Pass := 1 to 2 do
    begin
      DesiredHeight := Max(24, ContentHeight) + CONTENT_HEIGHT_RESERVE;
      if not SameValue(Height, DesiredHeight) then
        Height := DesiredHeight;
    end;
    if ScrollOffset <> 0 then
      ScrollOffset := 0;
  finally
    FAdjustingBounds := False
  end;
end;

procedure TChatMarkdownViewer.InnerScrollChanged(Sender: TObject);
begin
  // Native streaming auto-follow runs before the form's next layout timer.
  FitContentHeight;
end;

procedure TChatMarkdownViewer.Paint;
begin
  // Width/zoom reflow can also finish asynchronously without an OnScroll event.
  // Only the clipped child viewport grows here; its parent is laid out by the form.
  FitContentHeight;
  inherited;
end;

function TChatCodeScanner.Python: Boolean;
begin
  Result := False;
end;

function TPythonCodeScanner.Python: Boolean;
begin
  Result := True;
end;

function TChatCodeHighlighter.ScannerClass: TSyntaxLineScannerClass;
begin
  Result := TChatCodeScanner;
end;

function TPythonCodeHighlighter.ScannerClass: TSyntaxLineScannerClass;
begin
  Result := TPythonCodeScanner;
end;

procedure TChatCodeScanner.ReadString(const Quote: Char; const Triple: Boolean; const Continuing: Boolean);
var
  Start, Delimiter: Integer;
  Closed: Boolean;
begin
  Start := FPosition;
  if Triple then
    Delimiter := 3
  else
    Delimiter := 1;
  if not Continuing then
    Inc(FPosition, Delimiter);
  Closed := False;
  while FPosition <= Length(FLine) do
  begin
    if CharAt(FPosition) = '\' then
      Inc(FPosition, 2)
    else if (CharAt(FPosition) = Quote)
      and (not Triple or ((CharAt(FPosition + 1) = Quote) and (CharAt(FPosition + 2) = Quote))) then
    begin
      Inc(FPosition, Delimiter);
      Closed := True;
      Break;
    end
    else
      Inc(FPosition);
  end;
  if Closed then
    FState := CleanState
  else if Triple then
  begin
    if Quote = #39 then
      FState := TRIPLE_SINGLE
    else
      FState := TRIPLE_DOUBLE;
  end
  else if Quote = '`' then
    FState := TEMPLATE_STRING;
  FBuilder.Add(TSyntaxTokenKind.StringLiteral, Start, Min(FPosition, Length(FLine) + 1) - Start);
end;

procedure TChatCodeScanner.ReadBlockComment;
var
  Start, Closing: Integer;
begin
  Start := FPosition;
  Closing := Pos('*/', FLine, FPosition);
  if Closing = 0 then
  begin
    TakeRestOfLine(TSyntaxTokenKind.Comment);
    FState := BLOCK_COMMENT;
  end
  else
  begin
    FPosition := Closing + 2;
    FBuilder.Add(TSyntaxTokenKind.Comment, Start, FPosition - Start);
    FState := CleanState;
  end;
end;

procedure TChatCodeScanner.Resume;
begin
  case FState of
    BLOCK_COMMENT:
      ReadBlockComment;
    TRIPLE_SINGLE:
      ReadString(#39, True, True);
    TRIPLE_DOUBLE:
      ReadString('"', True, True);
    TEMPLATE_STRING:
      ReadString('`', False, True);
  end;
end;

procedure TChatCodeScanner.ScanNext;
var
  Current: Char;
begin
  Current := CharAt(FPosition);
  if Python and (Current = '#') then
    TakeRestOfLine(TSyntaxTokenKind.Comment)
  else if not Python and (Current = '/') and (CharAt(FPosition + 1) = '/') then
    TakeRestOfLine(TSyntaxTokenKind.Comment)
  else if not Python and (Current = '/') and (CharAt(FPosition + 1) = '*') then
  begin
    FBuilder.Add(TSyntaxTokenKind.Comment, FPosition, 2);
    Inc(FPosition, 2);
    ReadBlockComment;
  end
  else if not Python and (Current = '#') then
    TakeRestOfLine(TSyntaxTokenKind.Directive)
  else if CharInSet(Current, [#39, '"', '`']) then
    ReadString(
      Current,
      Python and (Current <> '`') and (CharAt(FPosition + 1) = Current) and (CharAt(FPosition + 2) = Current),
      False
    )
  else if CharInSet(Current, Digits) then
    ScanNumber
  else if CharInSet(Current, IdentifierStartCharacters) then
    ScanIdentifier
  else
    AddPlainCharacter;
end;

function TChatCodeScanner.IsKeyword(const Identifier: string): Boolean;
const
  PYTHON_KEYWORDS =
      ' False None True and as assert async await break case class continue def del elif else except '
    + 'finally for from global if import in is lambda match nonlocal not or pass raise return try while with yield ';
  C_STYLE_KEYWORDS =
    ' abstract as async await auto bool boolean break byte case catch char class const constexpr '
    + 'continue debugger default defer delete do double dynamic else enum explicit export extends extern false final '
    + 'finally float fn for foreach from func function go goto if implements import in inline instanceof int interface '
    + 'internal is keyof let long match module mut namespace native new noexcept null nullptr object operator out '
    + 'override package private protected public readonly record ref require return sbyte sealed short signed sizeof '
    + 'static string struct super switch synchronized template this throw throws trait true try type typedef typeof '
    + 'uint ulong union unsigned use using var virtual void volatile when where while yield ';
begin
  if Python then
    Result := PYTHON_KEYWORDS.Contains(' ' + Identifier + ' ')
  else
    Result := C_STYLE_KEYWORDS.Contains(' ' + Identifier + ' ');
end;

procedure RegisterChatHighlighters;
var
  Existing, PythonHighlighter, CStyleHighlighter: IMarkdownSyntaxHighlighter;

  procedure RegisterMissing(const Language: string; const Highlighter: IMarkdownSyntaxHighlighter);
  var
    Found: IMarkdownSyntaxHighlighter;
  begin
    if not THighlighterRegistry.TryGet(Language, Found) then
      THighlighterRegistry.Register(Language, Highlighter);
  end;

  procedure Alias(const Source: string; const Names: array of string);
  begin
    if THighlighterRegistry.TryGet(Source, Existing) then
      for var Name in Names do
        RegisterMissing(Name, Existing);
  end;

begin
  Alias('pascal', ['delphi', 'pas', 'objectpascal']);
  Alias('xml', ['html', 'htm', 'xhtml', 'svg']);
  Alias('diff', ['patch', 'udiff']);
  PythonHighlighter := TPythonCodeHighlighter.Create;
  for var Name in ['python', 'py', 'python3'] do
    RegisterMissing(Name, PythonHighlighter);
  CStyleHighlighter := TChatCodeHighlighter.Create;
  for var Name in ['javascript', 'js', 'typescript', 'ts', 'jsx', 'tsx', 'c', 'cpp', 'c++', 'h', 'hpp', 'csharp', 'cs', 'java', 'kotlin', 'swift', 'go', 'rust', 'rs'] do
    RegisterMissing(Name, CStyleHighlighter);
end;

function SplitThinkingMarkdown(const Source: string; const Streaming: Boolean): TChatMarkdownParts;
const
  OPEN_TAG = '<think>';
  CLOSE_TAG = '</think>';
var
  Answer, Thinking: TStringBuilder;
  Position, LineEnd, Mark, Run, FenceLength, InlineTicks, Depth: Integer;
  Fence: Char;
  OpaqueLine, IsFence: Boolean;

  procedure Append(const Value: string);
  begin
    if Depth > 0 then
      Thinking.Append(Value)
    else
      Answer.Append(Value);
  end;

  function TrimLineBreaks(const Value: string): string;
  var
    First, Last: Integer;
  begin
    First := 1;
    Last := Length(Value);
    while (First <= Last) and CharInSet(Value[First], [#10, #13]) do
      Inc(First);
    while (Last >= First) and CharInSet(Value[Last], [#10, #13]) do
      Dec(Last);
    Result := Copy(Value, First, Last - First + 1);
  end;

begin
  Result := Default(TChatMarkdownParts);
  Answer := TStringBuilder.Create;
  Thinking := TStringBuilder.Create;
  try
    Position := 1;
    Fence := #0;
    FenceLength := 0;
    InlineTicks := 0;
    Depth := 0;
    while Position <= Length(Source) do
    begin
      LineEnd := Pos(#10, Source, Position);
      if LineEnd = 0 then
        LineEnd := Length(Source) + 1;
      Mark := Position;
      while (Mark < LineEnd) and (Source[Mark] = ' ') and (Mark - Position < 3) do
        Inc(Mark);
      Run := 0;
      if (Mark < LineEnd) and CharInSet(Source[Mark], ['`', '~']) then
        while (Mark + Run < LineEnd) and (Source[Mark + Run] = Source[Mark]) do
          Inc(Run);
      IsFence := (InlineTicks = 0) and (Run >= 3);
      OpaqueLine :=
        (Fence <> #0) or ((InlineTicks = 0) and ((Copy(Source, Position, 4) = '    ') or (Source[Position] = #9)));
      if OpaqueLine then
      begin
        if IsFence
          and (Source[Mark] = Fence)
          and (Run >= FenceLength)
          and (Trim(Copy(Source, Mark + Run, LineEnd - Mark - Run)) = '') then
          Fence := #0;
      end
      else if IsFence then
      begin
        Fence := Source[Mark];
        FenceLength := Run;
        OpaqueLine := True;
      end;
      if OpaqueLine then
      begin
        Append(Copy(Source, Position, Min(LineEnd, Length(Source)) - Position + 1));
        Position := LineEnd + 1;
        Continue;
      end;
      while Position <= Min(LineEnd, Length(Source)) do
      begin
        if Source[Position] = '`' then
        begin
          Run := 1;
          while (Position + Run <= Length(Source)) and (Source[Position + Run] = '`') do
            Inc(Run);
          if InlineTicks = 0 then
            InlineTicks := Run
          else if InlineTicks = Run then
            InlineTicks := 0;
          Append(Copy(Source, Position, Run));
          Inc(Position, Run);
          Continue;
        end;
        if (InlineTicks = 0) and (Source[Position] = '<') then
        begin
          if SameText(Copy(Source, Position, Length(OPEN_TAG)), OPEN_TAG) then
          begin
            if (Depth = 0) and (Thinking.Length > 0) then
              Thinking.Append(#10#10);
            Inc(Depth);
            Result.HasThinking := True;
            Inc(Position, Length(OPEN_TAG));
            Continue;
          end;
          if (Depth > 0) and SameText(Copy(Source, Position, Length(CLOSE_TAG)), CLOSE_TAG) then
          begin
            Dec(Depth);
            Inc(Position, Length(CLOSE_TAG));
            Continue
          end;
          if Streaming
            and (((Length(Source) - Position + 1 < Length(OPEN_TAG))
            and OPEN_TAG.StartsWith(Copy(Source, Position, MaxInt), True))
            or ((Depth > 0)
            and (Length(Source) - Position + 1 < Length(CLOSE_TAG))
            and CLOSE_TAG.StartsWith(Copy(Source, Position, MaxInt), True))) then
          begin
            Position := Length(Source) + 1;
            Break
          end;
        end;
        Append(Source[Position]);
        Inc(Position);
      end;
    end;
    Result.Answer := Answer.ToString;
    Result.Thinking := TrimLineBreaks(Thinking.ToString);
    if Result.HasThinking then
      Result.Answer := TrimLineBreaks(Result.Answer);
    Result.ThinkingOpen := Depth > 0;
  finally
    Thinking.Free;
    Answer.Free
  end;
end;

function CreateChatMarkdownViewer(const Owner: TComponent; const Parent: TControl; const ViewerName: string): TMarkdownViewer;
const
  HEADING_SIZES: array[1..6] of Single = (26, 23, 20, 18, 17, 16);
begin
  Result := TChatMarkdownViewer.Create(Owner);
  try
    Result.Name := ViewerName;
    Result.Stored := False;
    Result.Parent := Parent;
    Result.Align := TAlignLayout.Top;
    Result.AutoScroll := False;
    Result.Images.AllowRemote := False;
    Result.Images.RestrictToDocumentFolder := True;
    Result.ThemePreset := TMarkdownThemePreset.Dark;
    Result.Theme.BackgroundColor := $FF0F1416;
    Result.Theme.TextColor := $FFE5EDEC;
    Result.Theme.LinkColor := $FF55D0A8;
    Result.Theme.CodeTextColor := $FFE5EDEC;
    Result.Theme.CodeBackgroundColor := $FF171E20;
    Result.Theme.CodeSpanBackgroundColor := $FF26332F;
    Result.Theme.TableHeaderBackgroundColor := $FF22312D;
    Result.Theme.TableBorderColor := $FF3B4B47;
    Result.Theme.BlockQuoteBarColor := $FF55D0A8;
    Result.Theme.BlockQuoteTextColor := $FFA8BBB3;
    Result.Theme.ThematicBreakColor := $FF3B4B47;
    Result.Theme.ContentPadding := 0;
    Result.Theme.BlockSpacing := 12;
    Result.Theme.CodePadding := 12;
    Result.Theme.TableCellPadding := 10;
    Result.Theme.CodeFont := TMarkdownFontStyle.Create('Consolas', 13);
    for var Level := 1 to 6 do
    begin
      Result.Theme.HeadingFonts[Level] := TMarkdownFontStyle.Create('Segoe UI', HEADING_SIZES[Level], True);
      Result.Theme.HeadingSpacingAbove[Level] := 16;
      Result.Theme.HeadingSpacingBelow[Level] := 8;
    end;
    RegisterChatHighlighters;
    TChatMarkdownViewer(Result).FitContentHeight;
  except
    Result.Free;
    raise;
  end;
end;

end.

