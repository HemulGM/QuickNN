unit LLMUI.Entry;

interface

uses
  System.Classes, System.SysUtils, FMX.Types, FMX.Controls, FMX.Forms,
  FMX.Objects, FMX.Layouts, FMX.StdCtrls, FMX.Controls.Presentation;

type
  TFrameEntry = class(TFrame)
    ButtonSelect: TButton;
    LayoutEntry: TLayout;
    LayoutEntryAvatar: TLayout;
    CircleEntryAvatar: TCircle;
    PathEntryIcon: TPath;
    LayoutEntryText: TLayout;
    LayoutEntryHeadline: TLayout;
    LabelBadge: TLabel;
    LabelEntryTitle: TLabel;
    LabelEntrySubtitle: TLabel;
    RectangleSelectedMarker: TRectangle;
    procedure ButtonSelectClick(Sender: TObject);
  private
    FOnSelect: TNotifyEvent;
  public
    EntryId: string;
    procedure SetEntry(const Id, Title, Subtitle, Badge: string; const Selected: Boolean);
    property OnSelect: TNotifyEvent read FOnSelect write FOnSelect;
  end;

implementation

{$R *.fmx}

procedure TFrameEntry.SetEntry(const Id, Title, Subtitle, Badge: string; const Selected: Boolean);
begin
  EntryId := Id;
  RectangleSelectedMarker.Visible := Selected;
  LabelEntryTitle.Text := Title;
  LabelEntrySubtitle.Text := Subtitle;
  LabelBadge.Text := Badge;
  if Selected then
    ButtonSelect.StyleLookup := 'chatrowselected'
  else
    ButtonSelect.StyleLookup := 'chatrowbutton';
end;

procedure TFrameEntry.ButtonSelectClick(Sender: TObject);
begin
  if Assigned(FOnSelect) then
    FOnSelect(Self);
end;

end.

