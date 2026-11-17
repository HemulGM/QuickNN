unit LLMUI.Attachment;

interface

uses
  System.Classes, System.SysUtils, System.Types, System.UITypes, FMX.Types,
  FMX.Controls, FMX.Forms, FMX.Layouts, FMX.Objects, FMX.StdCtrls,
  FMX.Controls.Presentation, LLMUI.Images;

type
  TFrameAttachment = class(TFrame)
    PanelAttachment: TPanel;
    ImagePreview: TImage;
    LayoutCaption: TLayout;
    LabelFileName: TLabel;
    ButtonRemove: TButton;
    LayoutAttachmentOverlay: TLayout;
    RectangleCaptionShade: TRectangle;
    PathRemove: TPath;
    RectangleReferenceBadge: TRectangle;
    LayoutReferenceBadge: TLayout;
    LabelReference: TLabel;
    RectangleDropTarget: TRectangle;
    procedure ImagePreviewClick(Sender: TObject);
    procedure ImagePointerDown(Sender: TObject; Button: TMouseButton; Shift: TShiftState; X, Y: Single);
    procedure ImagePointerMove(Sender: TObject; Shift: TShiftState; X, Y: Single);
    procedure ImageContextMenu(Sender: TObject; Button: TMouseButton; Shift: TShiftState; X, Y: Single);
    procedure ButtonRemoveClick(Sender: TObject);
  private
    FHash, FFileName: string;
    FImageMissing: Boolean;
    FImages: TArray<TChatImage>;
    FImageIndex: Integer;
    FOnBeginReferenceDrag: TNotifyEvent;
    FOnRemove: TNotifyEvent;
    FDragOrigin: TPointF;
    FDragTracking, FSuppressClick: Boolean;
  published
    property OnBeginReferenceDrag: TNotifyEvent read FOnBeginReferenceDrag write FOnBeginReferenceDrag;
  public
    procedure SetImage(const Image: TChatImage);
    procedure SetImages(const Images: TArray<TChatImage>; const Index: Integer);
    procedure SetHistoryImages(const Images: TArray<TChatImage>; const Index: Integer);
    property ImageIndex: Integer read FImageIndex;
    property ContentHash: string read FHash;
    property OnRemove: TNotifyEvent read FOnRemove write FOnRemove;
  end;

implementation

uses
  System.IOUtils, LLMUI.ImageViewer;

{$R *.fmx}

procedure TFrameAttachment.SetImage(const Image: TChatImage);
begin
  FImages := [Image];
  FImageIndex := 0;
  FDragTracking := False;
  FSuppressClick := False;
  LabelReference.Text := '<image1>';
  RectangleReferenceBadge.Visible := True;
  FHash := Image.ContentHash;
  var FileName := Image.FileName;
  if FFileName <> FileName then
  begin
    FFileName := FileName;
    FImageMissing := not TFile.Exists(FileName);
    ImagePreview.Bitmap.Clear(0);
    if not FImageMissing then
    try
      ImagePreview.Bitmap.LoadThumbnailFromFile(FileName, 256, 256, False)
    except
      FImageMissing := True;
      ImagePreview.Bitmap.Clear(0)
    end;
  end;
  var Caption := Image.OriginalName;
  if Caption = '' then
    Caption := Image.ModelName;
  if Caption = '' then
    Caption := 'Изображение';
  var Details := Format('%s · %d×%d', [Caption, Image.Settings.Width, Image.Settings.Height]);
  if not Image.IsUserImage then
    Details := Details + ' · seed ' + Image.Settings.Seed.ToString;
  if FImageMissing then
  begin
    Caption := 'Файл недоступен';
    Details := Caption + ' · ' + Details;
  end;
  LabelFileName.Text := Caption;
  LabelFileName.Hint := Details;
  PanelAttachment.Hint := Details;
  ImagePreview.Hint := Details;
end;

procedure TFrameAttachment.SetImages(const Images: TArray<TChatImage>; const Index: Integer);
begin
  if (Index < 0) or (Index >= Length(Images)) then
    raise EArgumentOutOfRangeException.Create('Недопустимый индекс изображения.');
  SetImage(Images[Index]);
  FImages := Copy(Images);
  FImageIndex := Index;
  LabelReference.Text := Format('<image%d>', [Index + 1]);
end;

procedure TFrameAttachment.SetHistoryImages(const Images: TArray<TChatImage>; const Index: Integer);
begin
  SetImages(Images, Index);
  ButtonRemove.Visible := False;
  RectangleReferenceBadge.Visible := False;
  OnBeginReferenceDrag := nil;
  FOnRemove := nil;
end;

procedure TFrameAttachment.ImagePointerDown(Sender: TObject; Button: TMouseButton; Shift: TShiftState; X, Y: Single);
begin
  FSuppressClick := False;
  FDragTracking := (Button = TMouseButton.mbLeft) and Assigned(OnBeginReferenceDrag);
  if FDragTracking and (Sender is TControl) then
  begin
    FDragOrigin := TControl(Sender).LocalToScreen(PointF(X, Y));
    if Root <> nil then
      Root.SetCaptured(TControl(Sender));
  end;
end;

procedure TFrameAttachment.ImagePointerMove(Sender: TObject; Shift: TShiftState; X, Y: Single);
begin
  if not FDragTracking or not (ssLeft in Shift) or not (Sender is TControl) then
    Exit;
  var Position := TControl(Sender).LocalToScreen(PointF(X, Y));
  if Position.Distance(FDragOrigin) < 6 then
    Exit;
  FDragTracking := False;
  FSuppressClick := True;
  // Only queue the operation here: native drag pumps messages and can retire
  // this frame. The durable form starts it after this pointer handler returns.
  if Assigned(OnBeginReferenceDrag) then
    OnBeginReferenceDrag(Self);
end;

procedure TFrameAttachment.ImageContextMenu(Sender: TObject; Button: TMouseButton; Shift: TShiftState; X, Y: Single);
var
  Host: IChatImageReferenceHost;
begin
  FDragTracking := False;
  if (Button <> TMouseButton.mbRight) or (Root = nil) or not (Sender is TControl)
    or (FImageIndex < 0) or (FImageIndex >= Length(FImages))
    or not Supports(Root.GetObject, IChatImageReferenceHost, Host) then
    Exit;
  // The host queues the popup after the pointer event. No nested menu loop can retain
  // this thumbnail while virtualization retires its originating message frame.
  Host.ShowImageReferenceMenu(FImages[FImageIndex], TControl(Sender).LocalToScreen(PointF(X, Y)));
end;

procedure TFrameAttachment.ImagePreviewClick(Sender: TObject);
begin
  if FSuppressClick then
    Exit;
  var DialogOwner: TComponent := Application;
  if Root <> nil then
    DialogOwner := Root.GetObject;
  TChatImageViewer.Open(DialogOwner, FImages, FImageIndex);
end;

procedure TFrameAttachment.ButtonRemoveClick(Sender: TObject);
begin
  if Assigned(FOnRemove) then
    FOnRemove(Self);
end;

end.

