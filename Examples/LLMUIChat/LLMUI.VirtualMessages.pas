unit LLMUI.VirtualMessages;

interface

uses
  System.SysUtils, System.Generics.Collections, LLMUI.Data;

const
  MESSAGE_GAP = 14;

type
  TMessageMetrics = class
  public
    Id, Source: string;
    State: TMessageState;
    Height: Double;
    Width: Single;
    Measured, ThinkingExpanded: Boolean;
  end;

  // Lightweight metadata only: no frames, Markdown documents or GPU objects.
  TChatVirtualMessages = class
  private
    FChat: TChatSession;
    FWidth: Single;
    FItems: TArray<TMessageMetrics>;
    FTree: TArray<Double>;
    FCache: TObjectDictionary<string, TMessageMetrics>;
    FIndices: TDictionary<string, Integer>;
    function GetCount: Integer;
    function EstimateHeight(const Message: TChatMessage; const Expanded: Boolean): Double;
    procedure AddHeight(const Index: Integer; const Delta: Double);
    procedure SetHeight(const Index: Integer; const Height: Double);
  public
    constructor Create;
    destructor Destroy; override;
    procedure Bind(const Chat: TChatSession; const Width: Single);
    procedure MessageChanged(const Id: string; const KeepHeight: Boolean);
    function Measure(const Index: Integer; const Width, Height: Single; const Expanded: Boolean): Boolean;
    function Top(const Index: Integer): Double;
    function TotalHeight: Double;
    function FindRow(const Position: Double): Integer;
    function IndexOf(const Id: string): Integer;
    function Metrics(const Index: Integer): TMessageMetrics;
    function MessageAt(const Index: Integer): TChatMessage;
    property Count: Integer read GetCount;
    property Width: Single read FWidth;
  end;

implementation

uses
  System.Math, QuickNN.Models.LLM.Types, LLMUI.Markdown;

constructor TChatVirtualMessages.Create;
begin
  inherited;
  FCache := TObjectDictionary<string, TMessageMetrics>.Create([doOwnsValues]);
  FIndices := TDictionary<string, Integer>.Create;
end;

destructor TChatVirtualMessages.Destroy;
begin
  FIndices.Free;
  FCache.Free;
  inherited;
end;

function TChatVirtualMessages.GetCount: Integer;
begin
  Result := Length(FItems);
end;

function TChatVirtualMessages.EstimateHeight(const Message: TChatMessage; const Expanded: Boolean): Double;
var
  Source: string;
  Available: Single;
  Columns, Column, Lines: Integer;
  Parts: TChatMarkdownParts;
begin
  Source := Message.Text;
  if (Source = '') and (Message.State = msFailed) then
    Source := Message.ErrorText;
  Available := Max(80, FWidth);
  Result := 66 + MESSAGE_GAP;
  if Message.Role = lrUser then
  begin
    Available := Max(80, Available * 0.87 - 82);
    Result := 54 + MESSAGE_GAP;
  end
  else
  begin
    Parts := SplitThinkingMarkdown(Source, Message.State = msStreaming);
    Source := Parts.Answer;
    if Parts.HasThinking then
    begin
      Result := Result + 42;
      if Expanded then
        Source := Source + #10 + Parts.Thinking;
    end;
  end;
  Columns := Max(12, Trunc(Available / 7.5));
  Lines := 1;
  Column := 0;
  for var C in Source do
    if C = #10 then
    begin
      Inc(Lines);
      Column := 0;
    end
    else if C <> #13 then
    begin
      Inc(Column);
      if Column > Columns then
      begin
        Inc(Lines);
        Column := 1;
      end;
    end;
  Result := Result + Max(24, Lines * 23);
  if Length(Message.Images) > 0 then
    Result := Result + 390;
end;

procedure TChatVirtualMessages.Bind(const Chat: TChatSession; const Width: Single);
var
  Item: TMessageMetrics;
begin
  FChat := Chat;
  FWidth := Width;
  FIndices.Clear;
  if Chat = nil then
    SetLength(FItems, 0)
  else
    SetLength(FItems, Chat.Messages.Count);
  for var I := 0 to High(FItems) do
  begin
    var Message := Chat.Messages[I];
    if not FCache.TryGetValue(Message.Id, Item) then
    begin
      Item := TMessageMetrics.Create;
      Item.Id := Message.Id;
      FCache.Add(Item.Id, Item);
    end;
    if not Item.Measured or not SameValue(Item.Width, Width, 0.1)
      or (Item.Source <> Message.Text) or (Item.State <> Message.State) then
    begin
      Item.Source := Message.Text;
      Item.State := Message.State;
      Item.Height := EstimateHeight(Message, Item.ThinkingExpanded);
      Item.Width := Width;
      Item.Measured := False;
    end;
    FItems[I] := Item;
    FIndices.Add(Item.Id, I);
  end;
  for var Id in FCache.Keys.ToArray do
    if not FIndices.ContainsKey(Id) then
      FCache.Remove(Id);
  SetLength(FTree, Count + 1);
  for var I := 0 to High(FTree) do
    FTree[I] := 0;
  // Linear construction of a Fenwick tree. Height corrections and lookups
  // afterwards take logarithmic time, regardless of total chat length.
  for var I := 1 to Count do
  begin
    FTree[I] := FTree[I] + FItems[I - 1].Height;
    var Parent := I + (I and  - I);
    if Parent <= Count then
      FTree[Parent] := FTree[Parent] + FTree[I];
  end;
end;

procedure TChatVirtualMessages.AddHeight(const Index: Integer; const Delta: Double);
begin
  var I := Index + 1;
  while I <= Count do
  begin
    FTree[I] := FTree[I] + Delta;
    I := I + (I and  - I);
  end;
end;

procedure TChatVirtualMessages.SetHeight(const Index: Integer; const Height: Double);
begin
  var Delta := Height - FItems[Index].Height;
  FItems[Index].Height := Height;
  AddHeight(Index, Delta);
end;

procedure TChatVirtualMessages.MessageChanged(const Id: string; const KeepHeight: Boolean);
begin
  var I := IndexOf(Id);
  if I < 0 then
    Exit;
  var Item := FItems[I];
  var Message := MessageAt(I);
  Item.Source := Message.Text;
  Item.State := Message.State;
  Item.Measured := False;
  if not KeepHeight then
    SetHeight(I, EstimateHeight(Message, Item.ThinkingExpanded));
end;

function TChatVirtualMessages.Measure(const Index: Integer; const Width, Height: Single; const Expanded: Boolean): Boolean;
begin
  var Item := FItems[Index];
  Result := not SameValue(Item.Height, Height, 0.1);
  if Result then
    SetHeight(Index, Max(1, Height));
  Item.Width := Width;
  Item.Measured := True;
  Item.ThinkingExpanded := Expanded;
end;

function TChatVirtualMessages.Top(const Index: Integer): Double;
begin
  Result := 0;
  var I := EnsureRange(Index, 0, Count);
  while I > 0 do
  begin
    Result := Result + FTree[I];
    I := I - (I and  - I);
  end;
end;

function TChatVirtualMessages.TotalHeight: Double;
begin
  Result := Top(Count);
end;

function TChatVirtualMessages.FindRow(const Position: Double): Integer;
var
  Prefix: Double;
  Bit, Index, Next: Integer;
begin
  if Count = 0 then
    Exit(-1);
  Bit := 1;
  while Bit <= Count div 2 do
    Bit := Bit * 2;
  Prefix := 0;
  Index := 0;
  while Bit > 0 do
  begin
    Next := Index + Bit;
    if (Next <= Count) and (Prefix + FTree[Next] <= Position) then
    begin
      Prefix := Prefix + FTree[Next];
      Index := Next;
    end;
    Bit := Bit div 2;
  end;
  Result := EnsureRange(Index, 0, Count - 1);
end;

function TChatVirtualMessages.IndexOf(const Id: string): Integer;
begin
  if not FIndices.TryGetValue(Id, Result) then
    Result := -1;
end;

function TChatVirtualMessages.Metrics(const Index: Integer): TMessageMetrics;
begin
  Result := FItems[Index];
end;

function TChatVirtualMessages.MessageAt(const Index: Integer): TChatMessage;
begin
  Result := FChat.Messages[Index];
end;

end.

