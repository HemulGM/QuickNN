unit LLMUI.Dialogs;

interface

uses
  System.Classes, System.SysUtils, System.UITypes, FMX.Types, FMX.Controls,
  FMX.Forms, FMX.Layouts, FMX.Objects, FMX.StdCtrls, FMX.Edit,
  FMX.Controls.Presentation;

type
  TChatDialogResultProc = reference to procedure(const Result: TModalResult);

  TChatWindow = class(TForm)
  private
    FReleasePending: Boolean;
  protected
    procedure CreateHandle; override;
    procedure DoShow; override;
  public
    constructor Create(AOwner: TComponent); override;
    procedure ReleaseDialog;
  end;

  TFormChatDialog = class(TChatWindow)
    RectangleCanvas: TRectangle;
    LayoutContent: TLayout;
    LayoutHeading: TLayout;
    RectangleIcon: TRectangle;
    PathDialogIcon: TPath;
    LabelTitle: TLabel;
    VertScrollBoxMessage: TVertScrollBox;
    LabelMessage: TLabel;
    LayoutInput: TLayout;
    EditInput: TEdit;
    LayoutActions: TLayout;
    ButtonAccept: TButton;
    ButtonCancel: TButton;
    procedure ActionClick(Sender: TObject);
    procedure FormKeyDown(Sender: TObject; var Key: Word; var KeyChar: WideChar; Shift: TShiftState);
  private
    FAsync: Boolean;
    FResultProc: TChatDialogResultProc;
    FCompleted: Boolean;
  protected
    procedure DoShow; override;
    procedure DoClose(var CloseAction: TCloseAction); override;
  public
    procedure Configure(const Title, Message, AcceptText: string; const Confirmation: Boolean);
    procedure Open(const ResultProc: TChatDialogResultProc);
  end;

  TChatDialogs = class sealed
  public
    class function Confirm(const Owner: TComponent; const Title, Message, AcceptText: string): Boolean; static;
    class procedure ConfirmAsync(const Owner: TComponent; const Title, Message, AcceptText: string; const ResultProc: TChatDialogResultProc); static;
    class procedure Message(const Owner: TComponent; const Title, Text: string; const Modal: Boolean = True); static;
    class function Input(const Owner: TComponent; const Title, Text: string; var Value: string): Boolean; static;
    class procedure InstallExceptionHandler; static;
  end;

implementation

uses
  System.Math, LLMUI.Theme, LLMUI.Controls;

{$R *.fmx}

type
  TChatExceptionHandler = class(TComponent)
  private
    FReporting: Boolean;
    procedure Handle(Sender: TObject; E: Exception);
  end;

constructor TChatWindow.Create(AOwner: TComponent);
begin
  inherited;
  StyleBook := ChatStyleBook;
end;

procedure TChatWindow.CreateHandle;
begin
  inherited;
  ApplyChatWindowColors(Self);
end;

procedure TChatWindow.DoShow;
begin
  inherited;
  ApplyChatWindowColors(Self);
end;

procedure TChatWindow.ReleaseDialog;
begin
  if FReleasePending then
    Exit;
  FReleasePending := True;
  // Forms outlive the click that closes them. Clear capture and callbacks first.
  if Captured <> nil then
    SetCaptured(nil);
  for var I := 0 to ChildrenCount - 1 do
    if Children[I] is TControl then
      PrepareControlRelease(TControl(Children[I]));
  OnKeyDown := nil;
  // The release pool owns the pending destruction, not the closing owner form.
  if Owner <> nil then
    Owner.RemoveComponent(Self);
  // FMX queues form destruction after the current event, including native handles.
  ReleaseForm;
end;

procedure TFormChatDialog.DoShow;
begin
  Realign;
  ClientHeight := Min(520, Max(224, Ceil(LabelMessage.Height) + 172 + Ord(LayoutInput.Visible) * 60));
  inherited;
end;

procedure TFormChatDialog.Configure(const Title, Message, AcceptText: string; const Confirmation: Boolean);
begin
  Caption := Title + ' — QuickNN';
  LabelTitle.Text := Title;
  LabelMessage.Text := Message;
  ButtonAccept.Text := AcceptText;
  ButtonAccept.ModalResult := mrOk;
  ButtonCancel.Visible := Confirmation;
  ButtonAccept.Default := not Confirmation;
  ButtonAccept.Cancel := not Confirmation;
  ButtonCancel.Default := Confirmation;
  ButtonCancel.Cancel := Confirmation;
  if Confirmation then
    ButtonAccept.ModalResult := mrYes;
  if Confirmation then
    ActiveControl := ButtonCancel
  else
    ActiveControl := ButtonAccept;
end;

procedure TFormChatDialog.Open(const ResultProc: TChatDialogResultProc);
begin
  FAsync := True;
  FResultProc := ResultProc;
  Show;
end;

procedure TFormChatDialog.ActionClick(Sender: TObject);
begin
  if FCompleted then
    Exit;
  ModalResult := TButton(Sender).ModalResult;
  if FAsync then
    Close;
end;

procedure TFormChatDialog.FormKeyDown(Sender: TObject; var Key: Word; var KeyChar: WideChar; Shift: TShiftState);
begin
  if Key = vkEscape then
  begin
    Key := 0;
    ModalResult := mrCancel;
    if FAsync then
      Close;
  end;
end;

procedure TFormChatDialog.DoClose(var CloseAction: TCloseAction);
begin
  inherited;
  if CloseAction = TCloseAction.caNone then
    Exit;
  if ModalResult = mrNone then
    ModalResult := mrCancel;
  if FAsync then
  begin
    CloseAction := TCloseAction.caHide;
    if FCompleted then
      Exit;
    FCompleted := True;
    var Callback := FResultProc;
    FResultProc := nil;
    var Answer := ModalResult;
    Hide;
    ReleaseDialog;
    if Assigned(Callback) then
      Callback(Answer);
  end;
end;

class function TChatDialogs.Confirm(const Owner: TComponent; const Title, Message, AcceptText: string): Boolean;
begin
  var Dialog := TFormChatDialog.Create(Owner);
  try
    Dialog.Configure(Title, Message, AcceptText, True);
    Result := Dialog.ShowModal = mrYes;
  finally
    Dialog.ReleaseDialog
  end;
end;

class procedure TChatDialogs.ConfirmAsync(const Owner: TComponent; const Title, Message, AcceptText: string; const ResultProc: TChatDialogResultProc);
begin
  var Dialog := TFormChatDialog.Create(Owner);
  try
    Dialog.Configure(Title, Message, AcceptText, True);
    Dialog.Open(ResultProc);
  except
    Dialog.ReleaseDialog;
    raise;
  end;
end;

class procedure TChatDialogs.Message(const Owner: TComponent; const Title, Text: string; const Modal: Boolean);
begin
  var Dialog := TFormChatDialog.Create(Owner);
  if Modal then
  begin
    try
      Dialog.Configure(Title, Text, 'Понятно', False);
      Dialog.ShowModal;
    finally
      Dialog.ReleaseDialog
    end;
  end
  else
  begin
    try
      Dialog.Configure(Title, Text, 'Понятно', False);
      Dialog.Open(nil);
    except
      Dialog.ReleaseDialog;
      raise;
    end;
  end;
end;

class function TChatDialogs.Input(const Owner: TComponent; const Title, Text: string; var Value: string): Boolean;
begin
  var Dialog := TFormChatDialog.Create(Owner);
  try
    Dialog.Configure(Title, Text, 'Создать', True);
    Dialog.LayoutInput.Visible := True;
    Dialog.EditInput.Text := Value;
    Dialog.ButtonAccept.ModalResult := mrOk;
    Dialog.ButtonAccept.Default := True;
    Dialog.ButtonCancel.Default := False;
    Dialog.ActiveControl := Dialog.EditInput;
    Result := Dialog.ShowModal = mrOk;
    if Result then
      Value := Dialog.EditInput.Text.Trim;
  finally
    Dialog.ReleaseDialog
  end;
end;

procedure TChatExceptionHandler.Handle(Sender: TObject; E: Exception);
begin
  if (E is EAbort) or FReporting then
    Exit;
  FReporting := True;
  try
    TChatDialogs.Message(Application, 'Не удалось выполнить действие', E.Message);
  finally
    FReporting := False
  end;
end;

class procedure TChatDialogs.InstallExceptionHandler;
begin
  var Handler := TChatExceptionHandler.Create(Application);
  Application.OnException := Handler.Handle;
end;

end.

