unit LLMUI.Attachment;

interface

uses
  System.Classes, System.SysUtils, FMX.Types, FMX.Controls, FMX.Forms,
  FMX.Layouts, FMX.Objects, FMX.StdCtrls, FMX.Controls.Presentation,
  LLMUI.Images;

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
    procedure ImagePreviewClick(Sender: TObject);
    procedure ButtonRemoveClick(Sender: TObject);
  private
    FHash: string;
    FImages: TArray<TChatImage>;
    FImageIndex: Integer;
    FOnRemove: TNotifyEvent;
  public
    procedure SetImage(const Image: TChatImage);
    procedure SetImages(const Images: TArray<TChatImage>; const Index: Integer);
    property ContentHash: string read FHash;
    property OnRemove: TNotifyEvent read FOnRemove write FOnRemove;
  end;

implementation

uses
  LLMUI.ImageViewer;

{$R *.fmx}

procedure TFrameAttachment.SetImage(const Image: TChatImage);
begin
  FImages := [Image];
  FImageIndex := 0;
  FHash := Image.ContentHash;
  LabelFileName.Text := Image.OriginalName;
  LabelFileName.Hint := Image.OriginalName;
  try
    ImagePreview.Bitmap.LoadThumbnailFromFile(Image.FileName, 256, 256, False)
  except
    ImagePreview.Bitmap.Clear(0)
  end;
end;

procedure TFrameAttachment.SetImages(const Images: TArray<TChatImage>; const Index: Integer);
begin
  if (Index < 0) or (Index >= Length(Images)) then
    raise EArgumentOutOfRangeException.Create('Недопустимый индекс изображения.');
  SetImage(Images[Index]);
  FImages := Copy(Images);
  FImageIndex := Index;
end;

procedure TFrameAttachment.ImagePreviewClick(Sender: TObject);
begin
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

