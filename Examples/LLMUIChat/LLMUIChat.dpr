program LLMUIChat;

uses
  System.StartUpCopy,
  System.SysUtils,
  FMX.Forms,
  LLMUI.Theme in 'LLMUI.Theme.pas' {DataModuleChatTheme: TDataModule},
  LLMUI.Dialogs in 'LLMUI.Dialogs.pas' {FormChatDialog},
  LLMUI.ImageViewer in 'LLMUI.ImageViewer.pas' {FrameImageViewer: TFrame},
  LLMUI.Main in 'LLMUI.Main.pas' {FormMain},
  LLMUI.Data in 'LLMUI.Data.pas',
  LLMUI.Images in 'LLMUI.Images.pas',
  LLMUI.Attachments in 'LLMUI.Attachments.pas',
  LLMUI.Attachment in 'LLMUI.Attachment.pas' {FrameAttachment: TFrame},
  LLMUI.VirtualMessages in 'LLMUI.VirtualMessages.pas',
  LLMUI.Context.Types in 'LLMUI.Context.Types.pas',
  LLMUI.Context in 'LLMUI.Context.pas',
  LLMUI.Engine in 'LLMUI.Engine.pas',
  LLMUI.Controls in 'LLMUI.Controls.pas',
  LLMUI.Markdown in 'LLMUI.Markdown.pas',
  LLMUI.Message in 'LLMUI.Message.pas' {FrameMessage: TFrame},
  LLMUI.Entry in 'LLMUI.Entry.pas' {FrameEntry: TFrame};

{$R *.res}

begin
  Application.Initialize;
  TChatDialogs.InstallExceptionHandler;
  Application.CreateForm(TFormMain, FormMain);
  Application.Run;
end.

