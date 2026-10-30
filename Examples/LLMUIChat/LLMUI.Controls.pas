unit LLMUI.Controls;

interface

uses
  FMX.Types, FMX.Controls, FMX.ListBox;

const
  CHAT_CONTENT_MAX_WIDTH = 800;

procedure ConfigureComboBox(const Combo: TComboBox);

procedure RetireControl(const Control: TControl);

procedure ClearControls(const Parent: TFmxObject);

procedure CopyText(const Text: string);

function ScrollAncestorByWheel(const Origin: TFmxObject; const WheelDelta: Integer): Boolean;

implementation

uses
  System.SysUtils, System.Types, System.UITypes, System.Math, FMX.Layouts,
  System.TypInfo, System.Rtti, FMX.Platform;

procedure ConfigureComboBox(const Combo: TComboBox);
begin
  Combo.ListBox.DefaultItemStyles.ItemStyle := 'chatcomboitem';
  Combo.ItemHeight := 40;
  for var I := 0 to Combo.Count - 1 do
  begin
    Combo.ListItems[I].StyleLookup := 'chatcomboitem';
    Combo.ListItems[I].StyledSettings := [];
    Combo.ListItems[I].TextSettings.HorzAlign := TTextAlign.Leading;
    Combo.ListItems[I].TextSettings.VertAlign := TTextAlign.Center;
    Combo.ListItems[I].TextSettings.Font.Size := 14;
    Combo.ListItems[I].TextSettings.FontColor := $FFE5EDEC;
    Combo.ListItems[I].TextSettings.Trimming := TTextTrimming.Character;
    Combo.ListItems[I].TextSettings.WordWrap := False;
  end;
end;

procedure Disconnect(const Obj: TFmxObject);
var
  Properties: PPropList;
  Count: Integer;
  Empty: TMethod;
begin
  Empty.Code := nil;
  Empty.Data := nil;
  Count := GetPropList(Obj.ClassInfo, [tkMethod], nil);
  GetMem(Properties, Count * SizeOf(Pointer));
  try
    GetPropList(Obj.ClassInfo, [tkMethod], Properties);
    for var I := 0 to Count - 1 do
      SetMethodProp(Obj, Properties[I], Empty);
  finally
    FreeMem(Properties);
  end;
  Obj.TagObject := nil;
  Obj.Name := '';
  for var I := 0 to Obj.ChildrenCount - 1 do
    Disconnect(Obj.Children[I]);
end;

procedure RetireControl(const Control: TControl);
var
  Root: IRoot;
  Captured: TFmxObject;
begin
  if Control = nil then
    Exit;
  Disconnect(Control);
  Root := Control.Root;
  if (Root <> nil) and (Root.Captured <> nil) then
  begin
    Captured := Root.Captured.GetObject;
    while Captured <> nil do
    begin
      if Captured = Control then
      begin
        Root.SetCaptured(nil);
        Break;
      end;
      Captured := Captured.Parent;
    end;
  end;
  Control.Parent := nil;
  Control.Release;
end;

procedure ClearControls(const Parent: TFmxObject);
begin
  while Parent.ChildrenCount > 0 do
    if Parent.Children[Parent.ChildrenCount - 1] is TControl then
      RetireControl(TControl(Parent.Children[Parent.ChildrenCount - 1]))
    else
      raise EInvalidOp.Create('Expected a visual container');
end;

function ScrollAncestorByWheel(const Origin: TFmxObject; const WheelDelta: Integer): Boolean;
var
  Parent: TFmxObject;
  Scroll: TVertScrollBox;
  Position: TPointF;
  Limit: Single;
begin
  Parent := Origin;
  while (Parent <> nil) and not (Parent is TVertScrollBox) do
    Parent := Parent.Parent;
  Result := Parent <> nil;
  if not Result then
    Exit;
  Scroll := TVertScrollBox(Parent);
  Position := Scroll.ViewportPosition;
  Limit := Max(0, Scroll.ContentBounds.Height - Scroll.ViewportRect.Height);
  Position.Y := EnsureRange(Position.Y - WheelDelta * 0.4, 0, Limit);
  Scroll.ViewportPosition := Position;
  // Consume at the boundary too: the wheel must never reach a value editor.
end;

procedure CopyText(const Text: string);
var
  Clipboard: IFMXClipboardService;
begin
  if TPlatformServices.Current.SupportsPlatformService(IFMXClipboardService, Clipboard) then
    Clipboard.SetClipboard(TValue.From<string>(Text));
end;

end.

