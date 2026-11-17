unit LLMUI.RoleSection;

interface

uses
  System.Classes, System.SysUtils, FMX.Types, FMX.Controls, FMX.Layouts,
  FMX.Forms, FMX.Objects, FMX.StdCtrls, FMX.Controls.Presentation;

type
  TFrameRoleSection = class(TFrame)
    LayoutHeader: TLayout;
    ButtonToggle: TButton;
    LayoutCaption: TLayout;
    LabelTitle: TLabel;
    LabelSummary: TLabel;
    PathChevron: TPath;
    LayoutChevron: TLayout;
    LayoutBody: TLayout;
    RectangleDivider: TRectangle;
    procedure ButtonToggleClick(Sender: TObject);
  private
    FExpanded, FMeasuring: Boolean;
    FOnExpandedChanged: TNotifyEvent;
    procedure SetExpanded(const Value: Boolean);
  public
    constructor Create(AOwner: TComponent); override;
    procedure UpdateSize;
    property Expanded: Boolean read FExpanded write SetExpanded;
    property OnExpandedChanged: TNotifyEvent read FOnExpandedChanged write FOnExpandedChanged;
  end;

implementation

uses
  LLMUI.Controls;

{$R *.fmx}

constructor TFrameRoleSection.Create(AOwner: TComponent);
begin
  inherited;
  FExpanded := True;
end;

procedure TFrameRoleSection.UpdateSize;
begin
  if FMeasuring then
    Exit;
  FMeasuring := True;
  try
    FitVerticalLayout(LayoutBody);
    LayoutBody.Visible := FExpanded;
    Height := LayoutHeader.Height + RectangleDivider.Height;
    if FExpanded then
      Height := Height + LayoutBody.Height;
    if FExpanded then
      PathChevron.RotationAngle := 270
    else
      PathChevron.RotationAngle := 180;
    if FExpanded then
      ButtonToggle.Hint := 'Свернуть раздел'
    else
      ButtonToggle.Hint := 'Развернуть раздел';
  finally
    FMeasuring := False
  end;
end;

procedure TFrameRoleSection.SetExpanded(const Value: Boolean);
begin
  if FExpanded = Value then
    Exit;
  FExpanded := Value;
  UpdateSize;
end;

procedure TFrameRoleSection.ButtonToggleClick(Sender: TObject);
begin
  Expanded := not Expanded;
  if Assigned(FOnExpandedChanged) then
    FOnExpandedChanged(Self);
end;

end.

