program LLMUIChat;

uses
  System.StartUpCopy,
  System.SysUtils,
  System.Hash,
  Winapi.Windows,
  FMX.Dialogs,
  FMX.Forms,
  LLMUI.Main in 'LLMUI.Main.pas' {FormMain},
  LLMUI.Data in 'LLMUI.Data.pas',
  LLMUI.Context.Types in 'LLMUI.Context.Types.pas',
  LLMUI.Context in 'LLMUI.Context.pas',
  LLMUI.Engine in 'LLMUI.Engine.pas',
  LLMUI.Controls in 'LLMUI.Controls.pas',
  LLMUI.Markdown in 'LLMUI.Markdown.pas',
  LLMUI.Message in 'LLMUI.Message.pas' {FrameMessage: TFrame},
  LLMUI.Entry in 'LLMUI.Entry.pas' {FrameEntry: TFrame};

{$R *.res}

var
  InstanceMutex: THandle;

begin
  // Concurrent instances must not overwrite the same chat history.
  InstanceMutex :=
      CreateMutex(
          nil,
          True,
          PChar('Local\QuickNN.LLMUIChat.' + THashSHA2.GetHashString(LowerCase(ExpandFileName(DefaultStorePath))))
      );
  if InstanceMutex = 0 then
    RaiseLastOSError;
  if GetLastError = ERROR_ALREADY_EXISTS then
  begin
    CloseHandle(InstanceMutex);
    ShowMessage(
        'QuickNN Chat уже запущен. Переключитесь в открытое окно приложения.'
    );
    Exit;
  end;
  try
    Application.Initialize;
    Application.CreateForm(TFormMain, FormMain);
    Application.Run;
  finally
    ReleaseMutex(InstanceMutex);
    CloseHandle(InstanceMutex);
  end;
end.
