unit LLMUI.Theme;

interface

uses
  System.Classes, FMX.Types, FMX.Controls, FMX.Forms;

type
  TDataModuleChatTheme = class(TDataModule)
    StyleBookChat: TStyleBook;
  end;

function ChatStyleBook: TStyleBook;

procedure ApplyChatWindowColors(const Form: TCommonCustomForm);

implementation

{%CLASSGROUP 'FMX.Controls.TControl'}

uses
  Winapi.Windows, Winapi.Dwmapi, FMX.Platform.Win;

{$R *.dfm}

var
  Theme: TDataModuleChatTheme;

function ChatStyleBook: TStyleBook;
begin
  if Theme = nil then
    Theme := TDataModuleChatTheme.Create(Application);
  Result := Theme.StyleBookChat;
end;

procedure ApplyChatWindowColors(const Form: TCommonCustomForm);
var
  Window: HWND;
  Enabled: LongBool;
  Color: COLORREF;
begin
  if Form.Handle = nil then
    Exit;
  Window := WindowHandleToPlatform(Form.Handle).Wnd;
  if not IsWindow(Window) then
    Exit;
  Enabled := True;
  DwmSetWindowAttribute(Window, 20, @Enabled, SizeOf(Enabled));
  Color := RGB(15, 20, 22);
  DwmSetWindowAttribute(Window, 35, @Color, SizeOf(Color));
  Color := RGB(229, 237, 236);
  DwmSetWindowAttribute(Window, 36, @Color, SizeOf(Color));
  Color := RGB(34, 52, 51);
  DwmSetWindowAttribute(Window, 34, @Color, SizeOf(Color));
end;

end.

