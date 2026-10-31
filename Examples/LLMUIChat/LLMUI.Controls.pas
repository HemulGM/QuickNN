unit LLMUI.Controls;

interface

uses
  System.Types, FMX.Types, FMX.Controls, FMX.ListBox, FMX.Layouts;

const
  CHAT_CONTENT_MAX_WIDTH = 800;

procedure ConfigureComboBox(const Combo: TComboBox);

procedure PrepareControlRelease(const Control: TControl);

procedure RetireControl(const Control: TControl);

procedure ClearControls(const Parent: TFmxObject);

procedure CopyText(const Text: string);

function ScrollAncestorByWheel(const Origin: TFmxObject; const WheelDelta: Integer): Boolean;

// Layout corrections move the animation target together with its origin.
procedure MoveScrollViewport(const Scroll: TVertScrollBox; const Position: TPointF);

procedure StopScrollMotion(const Scroll: TVertScrollBox);

implementation

uses
  System.SysUtils, System.UITypes, System.Math, System.Math.Vectors,
  FMX.InertialMovement, System.TypInfo, System.Rtti, FMX.Platform;

type
  TScrollControlAccess = class(TVertScrollBox);

  TScrollMotionAccess = class(TAniCalculations)
  public
    property MouseTarget;
    property Shown;
  end;

procedure MoveScrollViewport(const Scroll: TVertScrollBox; const Position: TPointF);
var
  Motion: TScrollMotionAccess;
  Pending: TAniCalculations.TTarget;
  Delta: TPointD;
  HalfPixel: Single;
begin
  Motion := TScrollMotionAccess(Scroll.AniCalculations);
  Delta := TPointD.Create(Position - Scroll.ViewportPosition);
  // Keep subpixel progress. Rounding every layout tick can cancel or stall a
  // slow animation even when the visible viewport has not actually moved.
  HalfPixel := 0.5;
  if Scroll.Scene <> nil then
    HalfPixel := HalfPixel / Max(1, Scroll.Scene.GetSceneScale);
  if (Abs(Delta.X) < HalfPixel) and (Abs(Delta.Y) < HalfPixel) then
    Exit;
  Pending := Motion.MouseTarget;
  Motion.ViewportPosition := Motion.ViewportPosition + Delta;
  if Pending.TargetType = TAniCalculations.TTargetType.Other then
  begin
    Pending.Point := Pending.Point + Delta;
    Motion.MouseTarget := Pending;
  end;
  TScrollControlAccess(Scroll).Realign;
end;

procedure StopScrollMotion(const Scroll: TVertScrollBox);
var
  Motion: TScrollMotionAccess;
  Pending: TAniCalculations.TTarget;
  Animated: Boolean;
begin
  Motion := TScrollMotionAccess(Scroll.AniCalculations);
  Pending := Default(TAniCalculations.TTarget);
  Pending.TargetType := TAniCalculations.TTargetType.Achieved;
  Motion.MouseTarget := Pending;
  // Clear velocity as well as the pending wheel destination before a chat
  // switch or an explicit jump. Do not carry momentum into another chat.
  Animated := Motion.Animation;
  Motion.Animation := False;
  Motion.Animation := Animated;
end;

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

procedure PrepareControlRelease(const Control: TControl);
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
end;

procedure RetireControl(const Control: TControl);
begin
  if Control = nil then
    Exit;
  PrepareControlRelease(Control);
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
  Motion: TScrollMotionAccess;
begin
  Parent := Origin;
  while (Parent <> nil) and not (Parent is TVertScrollBox) do
    Parent := Parent.Parent;
  Result := Parent <> nil;
  if not Result then
    Exit;
  Scroll := TVertScrollBox(Parent);
  if not Scroll.Enabled then
    Exit;
  Motion := TScrollMotionAccess(Scroll.AniCalculations);
  Motion.Shown := True;
  Motion.MouseWheel(0, -WheelDelta * 0.4);
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

