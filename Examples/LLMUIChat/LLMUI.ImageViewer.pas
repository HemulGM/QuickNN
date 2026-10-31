unit LLMUI.ImageViewer;

interface

uses
  System.Classes, System.SysUtils, System.Generics.Collections, System.Types,
  System.UITypes, FMX.Types, FMX.Controls, FMX.Forms, FMX.Layouts, FMX.Objects,
  FMX.StdCtrls, FMX.Graphics, FMX.Controls.Presentation, FMX.Dialogs, FMX.Menus,
  LLMUI.Images;

type
  TFrameImageViewer = class(TFrame)
    RectangleShade: TRectangle;
    LayoutViewer: TLayout;
    RectangleToolbar: TRectangle;
    RectangleFooter: TRectangle;
    ButtonClose: TButton;
    PathCloseIcon: TPath;
    LayoutToolbar: TLayout;
    LayoutNavigation: TLayout;
    ButtonPrevious: TButton;
    PathPreviousIcon: TPath;
    ButtonNext: TButton;
    PathNextIcon: TPath;
    LayoutFileActions: TLayout;
    ButtonCopy: TButton;
    PathCopyIcon: TPath;
    PopupMenuImage: TPopupMenu;
    MenuItemCopyImage: TMenuItem;
    ButtonSave: TButton;
    LayoutTools: TLayout;
    LayoutViewport: TLayout;
    PaintBoxImage: TPaintBox;
    LabelError: TLabel;
    LayoutFooter: TLayout;
    LabelFileName: TLabel;
    LabelDetails: TLabel;
    SaveDialogImage: TSaveDialog;
    LayoutZoom: TLayout;
    ButtonZoomOut: TButton;
    Path2: TPath;
    LabelZoom: TLabel;
    ButtonZoomIn: TButton;
    Path1: TPath;
    ButtonActualSize: TButton;
    ButtonFit: TButton;
    procedure CloseClick(Sender: TObject);
    procedure BackgroundClick(Sender: TObject);
    procedure ImageClick(Sender: TObject);
    procedure PaintImage(Sender: TObject; Canvas: TCanvas);
    procedure ViewportResize(Sender: TObject);
    procedure ImageMouseDown(Sender: TObject; Button: TMouseButton; Shift: TShiftState; X, Y: Single);
    procedure ImageMouseMove(Sender: TObject; Shift: TShiftState; X, Y: Single);
    procedure ImageMouseUp(Sender: TObject; Button: TMouseButton; Shift: TShiftState; X, Y: Single);
    procedure ImageMouseWheel(Sender: TObject; Shift: TShiftState; WheelDelta: Integer; var Handled: Boolean);
    procedure ImageDblClick(Sender: TObject);
    procedure ButtonPreviousClick(Sender: TObject);
    procedure ButtonNextClick(Sender: TObject);
    procedure ButtonZoomOutClick(Sender: TObject);
    procedure ButtonZoomInClick(Sender: TObject);
    procedure ButtonActualSizeClick(Sender: TObject);
    procedure ButtonFitClick(Sender: TObject);
    procedure CopyImageClick(Sender: TObject);
    procedure ImageMenuPopup(Sender: TObject);
    procedure ButtonSaveClick(Sender: TObject);
    procedure FormKeyDown(Sender: TObject; var Key: Word; var KeyChar: WideChar; Shift: TShiftState);
  private
    FHost: TForm;
    FPreviousFocus: TControl;
    FBlockedControls: TList<TControl>;
    FPreviousKeyDown: TKeyEvent;
    FPreviousCloseQuery: TCloseQueryEvent;
    FPreviousDeactivate: TNotifyEvent;
    FSuppressBackgroundClick: Boolean;
    FBitmap: TBitmap;
    FImages: TArray<TChatImage>;
    FIndex: Integer;
    FFileName: string;
    FZoom: Single;
    FOffset, FLastMouse: TPointF;
    FFitToWindow, FDragging: Boolean;
    procedure AttachHost;
    procedure DetachHost;
    procedure HostCloseQuery(Sender: TObject; var CanClose: Boolean);
    procedure HostDeactivate(Sender: TObject);
    function PixelScale: Single;
    function FitZoom: Single;
    procedure LoadImage;
    procedure UpdateView;
    procedure ClampOffset;
    procedure StopDragging;
  protected
    procedure Notification(AComponent: TComponent; Operation: TOperation); override;
  public
    constructor Create(AOwner: TComponent); override;
    destructor Destroy; override;
    procedure ShowImages(const Images: TArray<TChatImage>; const Index: Integer);
    procedure Close;
    procedure SetImages(const Images: TArray<TChatImage>; const Index: Integer);
    procedure ZoomAt(const Value: Single; const Pivot: TPointF);
    function ImageBounds: TRectF;
    property ImageIndex: Integer read FIndex;
    property Zoom: Single read FZoom;
  end;

  TChatImageViewer = class sealed
  public
    class function Find(const Host: TForm): TFrameImageViewer; static;
    class function IsOpen(const Host: TForm): Boolean; static;
    class procedure Open(const Owner: TComponent; const Images: TArray<TChatImage>; const Index: Integer = 0); static;
  end;

implementation

uses
  System.Math, System.IOUtils, FMX.Platform, FMX.Platform.Win, LLMUI.Attachments;

{$R *.fmx}

const
  VIEWPORT_MARGIN = 60;
  MIN_ZOOM = 0.30;
  MAX_ZOOM = 16;
  ZOOM_STEP = 1.25;

constructor TFrameImageViewer.Create(AOwner: TComponent);
begin
  inherited;
  FBlockedControls := TList<TControl>.Create;
  FBitmap := TBitmap.Create;
  FZoom := 1;
  FFitToWindow := True;
end;

destructor TFrameImageViewer.Destroy;
begin
  StopDragging;
  DetachHost;
  FBitmap.Free;
  FBlockedControls.Free;
  inherited;
end;

function TFrameImageViewer.PixelScale: Single;
begin
  Result := 1;
  if PaintBoxImage.Scene <> nil then
    Result := Max(1, PaintBoxImage.Scene.GetSceneScale);
end;

function TFrameImageViewer.FitZoom: Single;
begin
  Result := 1 / PixelScale;
  if (FBitmap <> nil) and not FBitmap.IsEmpty then
    Result := Min(MAX_ZOOM / PixelScale, Min(
        Max(1, PaintBoxImage.Width - 2 * VIEWPORT_MARGIN) / FBitmap.Width,
        Max(1, PaintBoxImage.Height - 2 * VIEWPORT_MARGIN) / FBitmap.Height));
end;

function TFrameImageViewer.ImageBounds: TRectF;
begin
  Result := TRectF.Empty;
  if (FBitmap = nil) or FBitmap.IsEmpty then
    Exit;
  var Size := PointF(FBitmap.Width * FZoom, FBitmap.Height * FZoom);
  Result := TRectF.Create((PaintBoxImage.Width - Size.X) / 2 + FOffset.X,
    (PaintBoxImage.Height - Size.Y) / 2 + FOffset.Y, 0, 0);
  Result.Width := Size.X;
  Result.Height := Size.Y;
end;

procedure TFrameImageViewer.ClampOffset;
begin
  if (FBitmap = nil) or FBitmap.IsEmpty then
    Exit;
  var ExtentX := Max(0, (FBitmap.Width * FZoom - PaintBoxImage.Width) / 2);
  var ExtentY := Max(0, (FBitmap.Height * FZoom - PaintBoxImage.Height) / 2);
  FOffset.X := EnsureRange(FOffset.X, -ExtentX, ExtentX);
  FOffset.Y := EnsureRange(FOffset.Y, -ExtentY, ExtentY);
end;

procedure TFrameImageViewer.UpdateView;
begin
  if FBitmap = nil then
    Exit;
  ButtonPrevious.Enabled := FIndex > 0;
  ButtonNext.Enabled := FIndex < High(FImages);
  LayoutNavigation.Visible := ButtonPrevious.Enabled or ButtonNext.Enabled;
  var Loaded := not FBitmap.IsEmpty;
  ButtonSave.Enabled := Loaded;
  ButtonCopy.Enabled := Loaded;
  MenuItemCopyImage.Enabled := Loaded;
  var MinimumScale: Single := Min(MIN_ZOOM / PixelScale, FitZoom);
  var MaximumScale: Single := MAX_ZOOM / PixelScale;
  ButtonZoomIn.Enabled := Loaded and (FZoom < MaximumScale);
  ButtonZoomOut.Enabled := Loaded and (FZoom > MinimumScale);
  ButtonActualSize.Enabled := Loaded;
  ButtonFit.Enabled := Loaded;
  LabelZoom.Text := Format('%.0f%%', [FZoom * PixelScale * 100]);
  LabelDetails.Text := Format('%d из %d', [FIndex + 1, Length(FImages)]);
  if Loaded then
    LabelDetails.Text := Format('%d × %d · ', [FBitmap.Width, FBitmap.Height]) + LabelDetails.Text;
  PaintBoxImage.Repaint;
end;

procedure TFrameImageViewer.SetImages(const Images: TArray<TChatImage>; const Index: Integer);
begin
  if Length(Images) = 0 then
    raise EArgumentException.Create('Нет изображений для просмотра.');
  // The originating message frame may be recycled while the modal window is open.
  FImages := Copy(Images);
  FIndex := EnsureRange(Index, 0, High(FImages));
  LoadImage;
end;

procedure TFrameImageViewer.LoadImage;
begin
  StopDragging;
  FBitmap.SetSize(0, 0);
  FFileName := '';
  FOffset := PointF(0, 0);
  FFitToWindow := True;
  LabelError.Text := '';
  LabelError.Visible := False;
  var Image := FImages[FIndex];
  var Name := Image.OriginalName;
  if Name = '' then
    Name := Image.ModelName;
  try
    FFileName := Image.FileName;
    if Name = '' then
      Name := ExtractFileName(FFileName);
    if not TFile.Exists(FFileName) then
      raise EFileNotFoundException.Create('Файл изображения недоступен.');
    FBitmap.LoadFromFile(FFileName);
    if FBitmap.IsEmpty then
      raise EReadError.Create('Файл не содержит доступного изображения.');
  except
    on E: Exception do
    begin
      FBitmap.SetSize(0, 0);
      LabelError.Text := 'Не удалось открыть изображение.' + sLineBreak + E.Message;
      LabelError.Visible := True;
    end;
  end;
  LabelFileName.Text := Name;
  LabelFileName.Hint := FFileName;
  FZoom := FitZoom;
  UpdateView;
end;

procedure TFrameImageViewer.AttachHost;
begin
  if FHost <> nil then
    Exit;
  if (Root = nil) or not (Root.GetObject is TForm) then
    raise EInvalidOp.Create('Просмотр должен быть размещён в основном окне.');
  FHost := TForm(Root.GetObject);
  FHost.FreeNotification(Self);
  if (FHost.Focused <> nil) and (FHost.Focused.GetObject is TControl) then
  begin
    FPreviousFocus := TControl(FHost.Focused.GetObject);
    FPreviousFocus.FreeNotification(Self);
  end;
  if Root.Captured <> nil then
    Root.SetCaptured(nil);
  // Block the background's input, not the host window: the overlay shares it.
  for var I := 0 to Parent.ChildrenCount - 1 do
    if (Parent.Children[I] is TControl) and (Parent.Children[I] <> Self) then
    begin
      var Control := TControl(Parent.Children[I]);
      if Control.Enabled then
      begin
        FBlockedControls.Add(Control);
        Control.FreeNotification(Self);
        Control.Enabled := False;
      end;
    end;
  FPreviousKeyDown := FHost.OnKeyDown;
  FPreviousCloseQuery := FHost.OnCloseQuery;
  FPreviousDeactivate := FHost.OnDeactivate;
  FHost.OnKeyDown := FormKeyDown;
  FHost.OnCloseQuery := HostCloseQuery;
  FHost.OnDeactivate := HostDeactivate;
end;

procedure TFrameImageViewer.DetachHost;
begin
  if FHost = nil then
    Exit;
  var Host := FHost;
  FHost := nil;
  StopDragging;
  if not (csDestroying in Host.ComponentState) then
  begin
    if TMethod(Host.OnKeyDown).Data = Self then
      Host.OnKeyDown := FPreviousKeyDown;
    if TMethod(Host.OnCloseQuery).Data = Self then
      Host.OnCloseQuery := FPreviousCloseQuery;
    if TMethod(Host.OnDeactivate).Data = Self then
      Host.OnDeactivate := FPreviousDeactivate;
    for var Control in FBlockedControls do
      if not (csDestroying in Control.ComponentState) then
      begin
        Control.Enabled := True;
        Control.RemoveFreeNotification(Self);
      end;
    if (FPreviousFocus <> nil) and (FPreviousFocus.Root <> nil)
      and (FPreviousFocus.Root.GetObject = Host) and FPreviousFocus.Visible and FPreviousFocus.Enabled then
      FPreviousFocus.SetFocus;
  end;
  FBlockedControls.Clear;
  if FPreviousFocus <> nil then
    FPreviousFocus.RemoveFreeNotification(Self);
  FPreviousFocus := nil;
  Host.RemoveFreeNotification(Self);
end;

procedure TFrameImageViewer.Notification(AComponent: TComponent; Operation: TOperation);
begin
  inherited;
  if Operation <> opRemove then
    Exit;
  if AComponent = FHost then
    FHost := nil;
  if AComponent = FPreviousFocus then
    FPreviousFocus := nil;
  if (FBlockedControls <> nil) and (AComponent is TControl) then
    FBlockedControls.Remove(TControl(AComponent));
end;

procedure TFrameImageViewer.HostCloseQuery(Sender: TObject; var CanClose: Boolean);
begin
  var Previous := FPreviousCloseQuery;
  Close;
  if Assigned(Previous) then
    Previous(Sender, CanClose);
end;

procedure TFrameImageViewer.HostDeactivate(Sender: TObject);
begin
  StopDragging;
  if Assigned(FPreviousDeactivate) then
    FPreviousDeactivate(Sender);
end;

procedure TFrameImageViewer.ShowImages(const Images: TArray<TChatImage>; const Index: Integer);
begin
  AttachHost;
  try
    SetImages(Images, Index);
    Visible := True;
    BringToFront;
    Realign;
    ViewportResize(PaintBoxImage);
    ButtonClose.SetFocus;
  except
    Visible := False;
    DetachHost;
    raise;
  end;
end;

procedure TFrameImageViewer.Close;
begin
  StopDragging;
  if MenuItemCopyImage.View <> nil then
    MenuItemCopyImage.View.Loop := False;
  Visible := False;
  DetachHost;
  FBitmap.SetSize(0, 0);
  FImages := nil;
  FFileName := '';
  FIndex := 0;
  UpdateView;
end;

procedure TFrameImageViewer.CloseClick(Sender: TObject);
begin
  Close;
end;

procedure TFrameImageViewer.BackgroundClick(Sender: TObject);
begin
  Close;
end;

procedure TFrameImageViewer.ImageClick(Sender: TObject);
var
  Mouse: IFMXMouseService;
begin
  if FSuppressBackgroundClick then
  begin
    FSuppressBackgroundClick := False;
    Exit;
  end;
  if TPlatformServices.Current.SupportsPlatformService(IFMXMouseService, Mouse)
    and not ImageBounds.Contains(PaintBoxImage.ScreenToLocal(Mouse.GetMousePos)) then
    Close;
end;

procedure TFrameImageViewer.ViewportResize(Sender: TObject);
begin
  if (FBitmap = nil) or FBitmap.IsEmpty then
    Exit;
  if FFitToWindow then
  begin
    FZoom := FitZoom;
    FOffset := PointF(0, 0);
  end;
  ClampOffset;
  UpdateView;
end;

procedure TFrameImageViewer.PaintImage(Sender: TObject; Canvas: TCanvas);
begin
  if (FBitmap = nil) or FBitmap.IsEmpty then
    Exit;
  var State := Canvas.SaveState;
  try
    Canvas.IntersectClipRect(PaintBoxImage.LocalRect);
    var Bounds := ImageBounds;
    Canvas.DrawBitmap(FBitmap, TRectF.Create(0, 0, FBitmap.Width, FBitmap.Height), Bounds, 1, False);
    Canvas.Stroke.Kind := TBrushKind.Solid;
    Canvas.Stroke.Color := $FF304342;
    Canvas.Stroke.Thickness := 1;
    Canvas.DrawRect(Bounds, 0, 0, [], 1);
  finally
    Canvas.RestoreState(State)
  end;
end;

procedure TFrameImageViewer.ZoomAt(const Value: Single; const Pivot: TPointF);
begin
  if (FBitmap = nil) or FBitmap.IsEmpty then
    Exit;
  var OldBounds := ImageBounds;
  var Position := (Pivot - OldBounds.TopLeft) / FZoom;
  FZoom := EnsureRange(Value, Min(MIN_ZOOM / PixelScale, FitZoom), MAX_ZOOM / PixelScale);
  FFitToWindow := False;
  FOffset := Pivot - Position * FZoom
    - PointF((PaintBoxImage.Width - FBitmap.Width * FZoom) / 2,
    (PaintBoxImage.Height - FBitmap.Height * FZoom) / 2);
  ClampOffset;
  UpdateView;
end;

procedure TFrameImageViewer.ImageMouseDown(Sender: TObject; Button: TMouseButton; Shift: TShiftState; X, Y: Single);
begin
  if (Button <> TMouseButton.mbLeft) or FBitmap.IsEmpty then
    Exit;
  FSuppressBackgroundClick := False;
  var Bounds := ImageBounds;
  if not Bounds.Contains(PointF(X, Y)) then
    Exit;
  if (Bounds.Width <= PaintBoxImage.Width) and (Bounds.Height <= PaintBoxImage.Height) then
    Exit;
  FLastMouse := PointF(X, Y);
  FDragging := True;
  Root.SetCaptured(PaintBoxImage);
end;

procedure TFrameImageViewer.ImageMouseMove(Sender: TObject; Shift: TShiftState; X, Y: Single);
begin
  if not FDragging then
    Exit;
  if not (ssLeft in Shift) or (Root = nil) or (Root.Captured = nil) or (Root.Captured.GetObject <> PaintBoxImage) then
  begin
    StopDragging;
    Exit;
  end;
  FSuppressBackgroundClick := True;
  FOffset := FOffset + PointF(X, Y) - FLastMouse;
  FLastMouse := PointF(X, Y);
  ClampOffset;
  PaintBoxImage.Repaint;
end;

procedure TFrameImageViewer.StopDragging;
begin
  FDragging := False;
  if (Root <> nil) and (Root.Captured <> nil)
    and (Root.Captured.GetObject = PaintBoxImage) then
    Root.SetCaptured(nil);
end;

procedure TFrameImageViewer.ImageMouseUp(Sender: TObject; Button: TMouseButton; Shift: TShiftState; X, Y: Single);
begin
  StopDragging;
end;

procedure TFrameImageViewer.ImageMouseWheel(Sender: TObject; Shift: TShiftState; WheelDelta: Integer; var Handled: Boolean);
var
  Mouse: IFMXMouseService;
begin
  var Pivot := PointF(PaintBoxImage.Width / 2, PaintBoxImage.Height / 2);
  if TPlatformServices.Current.SupportsPlatformService(IFMXMouseService, Mouse) then
    Pivot := PaintBoxImage.ScreenToLocal(Mouse.GetMousePos);
  ZoomAt(FZoom * Power(ZOOM_STEP, WheelDelta / 120), Pivot);
  Handled := True;
end;

procedure TFrameImageViewer.ImageDblClick(Sender: TObject);
begin
  if FFitToWindow then
    ButtonActualSizeClick(ButtonActualSize)
  else
    ButtonFitClick(ButtonFit);
end;

procedure TFrameImageViewer.ButtonPreviousClick(Sender: TObject);
begin
  if FIndex <= 0 then
    Exit;
  Dec(FIndex);
  LoadImage;
end;

procedure TFrameImageViewer.ButtonNextClick(Sender: TObject);
begin
  if FIndex >= High(FImages) then
    Exit;
  Inc(FIndex);
  LoadImage;
end;

procedure TFrameImageViewer.ButtonZoomOutClick(Sender: TObject);
begin
  ZoomAt(FZoom / ZOOM_STEP, PointF(PaintBoxImage.Width / 2, PaintBoxImage.Height / 2));
end;

procedure TFrameImageViewer.ButtonZoomInClick(Sender: TObject);
begin
  ZoomAt(FZoom * ZOOM_STEP, PointF(PaintBoxImage.Width / 2, PaintBoxImage.Height / 2));
end;

procedure TFrameImageViewer.ButtonActualSizeClick(Sender: TObject);
begin
  FOffset := PointF(0, 0);
  ZoomAt(1 / PixelScale, PointF(PaintBoxImage.Width / 2, PaintBoxImage.Height / 2));
end;

procedure TFrameImageViewer.ButtonFitClick(Sender: TObject);
begin
  FFitToWindow := True;
  FOffset := PointF(0, 0);
  ViewportResize(PaintBoxImage);
end;

procedure TFrameImageViewer.CopyImageClick(Sender: TObject);
begin
  if FBitmap.IsEmpty then
    Exit;
  CopyImageToClipboard(FBitmap, WindowHandleToPlatform(FHost.Handle).Wnd);
  LabelDetails.Text := 'Изображение скопировано в буфер обмена';
end;

procedure TFrameImageViewer.ImageMenuPopup(Sender: TObject);
begin
  StopDragging;
  UpdateView;
end;

procedure TFrameImageViewer.ButtonSaveClick(Sender: TObject);
begin
  if FBitmap.IsEmpty then
    Exit;
  SaveDialogImage.FileName := ExtractFileName(FFileName);
  if SaveDialogImage.Execute and not SameFileName(SaveDialogImage.FileName, FFileName) then
    TFile.Copy(FFileName, SaveDialogImage.FileName, True);
end;

procedure TFrameImageViewer.FormKeyDown(Sender: TObject; var Key: Word; var KeyChar: WideChar; Shift: TShiftState);
begin
  case Key of
    vkEscape:
      Close;
    vkLeft:
      ButtonPreviousClick(ButtonPrevious);
    vkRight:
      ButtonNextClick(ButtonNext);
    vkAdd:
      ButtonZoomInClick(ButtonZoomIn);
    vkSubtract:
      ButtonZoomOutClick(ButtonZoomOut);
    Ord('0'):
      ButtonFitClick(ButtonFit);
    Ord('1'):
      ButtonActualSizeClick(ButtonActualSize);
  else
    if (Key = Ord('C')) and (ssCtrl in Shift) then
      CopyImageClick(ButtonCopy)
    else if (Key = Ord('S')) and (ssCtrl in Shift) then
      ButtonSaveClick(ButtonSave)
    else if KeyChar = '+' then
      ButtonZoomInClick(ButtonZoomIn)
    else if KeyChar = '-' then
      ButtonZoomOutClick(ButtonZoomOut)
    else
      Exit;
  end;
  Key := 0;
  KeyChar := #0;
end;

class function TChatImageViewer.Find(const Host: TForm): TFrameImageViewer;

  function Visit(const ObjectRoot: TFmxObject): TFrameImageViewer;
  begin
    if ObjectRoot is TFrameImageViewer then
      Exit(TFrameImageViewer(ObjectRoot));
    for var I := 0 to ObjectRoot.ChildrenCount - 1 do
    begin
      Result := Visit(ObjectRoot.Children[I]);
      if Result <> nil then
        Exit;
    end;
    Result := nil;
  end;

begin
  Result := Visit(Host);
end;

class function TChatImageViewer.IsOpen(const Host: TForm): Boolean;
begin
  var Viewer := Find(Host);
  Result := (Viewer <> nil) and Viewer.Visible;
end;

class procedure TChatImageViewer.Open(const Owner: TComponent; const Images: TArray<TChatImage>; const Index: Integer);
begin
  if Length(Images) = 0 then
    Exit;
  if not (Owner is TForm) then
    raise EArgumentException.Create('Недоступно окно просмотра.');
  var Host := TForm(Owner);
  var Viewer := Find(Host);
  if Viewer = nil then
  begin
    Viewer := TFrameImageViewer.Create(Host);
    Viewer.Parent := Host;
    Viewer.Align := TAlignLayout.Contents;
  end;
  Viewer.ShowImages(Images, Index);
end;

end.

