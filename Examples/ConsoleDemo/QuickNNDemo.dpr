program QuickNNDemo;

{$APPTYPE CONSOLE}

uses
  System.SysUtils,
  System.IOUtils,
  System.Diagnostics,
  QuickNN.Common,
  QuickNN.Vulkan,
  QuickNN.Models.Flux,
  QuickNN.Models.ZImage,
  QuickNN.Models.Qwen21,
  QuickNN.Models.Qwen21.VAE,
  QuickNN.Demo.Settings in 'QuickNN.Demo.Settings.pas';

function FindRepositoryRoot: string;
var
  Directory, Parent: string;
begin
  Directory := ExtractFileDir(ParamStr(0));
  repeat
    if TFile.Exists(TPath.Combine(Directory, 'QuickNN.groupproj')) then
      Exit(Directory);
    Parent := ExtractFileDir(Directory);
    if Parent = Directory then
      Break;
    Directory := Parent;
  until Directory = '';
  // Standalone exe: explicit model/assets/output paths can be set in DemoSettings.
  Result := ExtractFileDir(ParamStr(0));
end;

procedure ReportStatus(const Message: string);
begin
  WriteLn(Message);
  Flush(Output);
end;

procedure ReportPhase(const Status: string; const Done: Boolean);
begin
  if Done then
    ReportStatus(Status + ' done')
  else
    ReportStatus(Status);
end;

procedure ReportProgress(const Index, Total: LongInt);
begin
  WriteLn('  ', Index + 1, '/', Total);
  Flush(Output);
end;

function GenerateQwen(const Settings: TDemoSettings; const Directory: string): TQNNImage;
var
  Params: TQwen21Params;
  Pipeline: TQwen21Pipeline;
begin
  if SizeOf(Pointer) <> 8 then
    raise ENotSupportedException.Create('Qwen-Image generation requires the Win64 target.');
  QNNConfigureGPU(gpuForce);
  Params := TQwen21Params.Defaults;
  Params.Prompt := Settings.Prompt;
  Params.NegativePrompt := Settings.NegativePrompt;
  Params.Width := Settings.Width;
  Params.Height := Settings.Height;
  Params.Steps := Settings.Steps;
  Params.Seed := Settings.Seed;
  Params.Guidance := Settings.Guidance;
  Params.ReferenceImages := Settings.ReferenceImages;
  Params.ReferenceSize := Settings.ReferenceSize;
  Params.LoRAs := Settings.LoRAs;
  Params.ResidentGPU := Settings.ResidentGPU;
  Pipeline :=
    TQwen21Pipeline.Create(Directory, Settings.AssetsDirectory, Settings.Model = dmQwenImage21Int8, ReportStatus);
  try
    Result := Pipeline.Generate(Params);
  finally
    Pipeline.Free;
  end;
end;

function GenerateDiffusers(const Settings: TDemoSettings; const Directory: string): TQNNImage;
var
  Params: TGenerateParams;
  Flux: TQNNFlux;
  ZImage: TQNNZImage;
begin
  if (Length(Settings.ReferenceImages) <> 0) or (Length(Settings.LoRAs) <> 0) then
    raise ENotSupportedException.Create('This demo accepts reference images and LoRAs only for Qwen-Image 2.1.');
  if (Settings.Model = dmZImageTurbo) and (Settings.NegativePrompt <> '') then
    raise ENotSupportedException.Create('The Z-Image demo does not accept a negative prompt.');
  if not TFile.Exists(TPath.Combine(Directory, 'model_index.json')) then
    raise EFileNotFoundException.Create('Expected a Diffusers model directory: ' + Directory);
  QNNConfigureGPU(gpuAuto);
  Params := Default(TGenerateParams);
  Params.Width := Settings.Width;
  Params.Height := Settings.Height;
  Params.NumSteps := Settings.Steps;
  Params.Seed := Settings.Seed;
  Params.Guidance := Settings.Guidance;
  Params.PowerAlpha := 2;
  StepCallback := ReportProgress;
  TextProgressCallback := ReportProgress;
  VAEProgressCallback := ReportProgress;
  try
    if Settings.Model = dmZImageTurbo then
    begin
      ZImage := TQNNZImage.Load(Directory, ReportPhase);
      ZImage.ResidentGPU := Settings.ResidentGPU;
      try
        Result := ZImage.Generate(UTF8String(Settings.Prompt), Params);
      finally
        ZImage.Free;
      end;
    end
    else
    begin
      Flux := TQNNFlux.Load(Directory, ReportPhase);
      Flux.ResidentGPU := Settings.ResidentGPU;
      try
        Result := Flux.Generate(UTF8String(Settings.Prompt), UTF8String(Settings.NegativePrompt), Params);
      finally
        Flux.Free;
      end;
    end;
  finally
    PhaseCallback := nil;
    StepCallback := nil;
    TextProgressCallback := nil;
    VAEProgressCallback := nil;
  end;
end;

procedure Run;
var
  Settings: TDemoSettings;
  Root, Directory: string;
  Image: TQNNImage;
  Timer: TStopwatch;
  Status: TQNNGPUStatus;
begin
  Settings := DemoSettings;
  Root := FindRepositoryRoot;
  Directory := DefaultModelDirectory(Settings);
  if Settings.Steps = 0 then
    Settings.Steps := DefaultSteps(Settings.Model);
  if (Settings.Width <= 0)
    or (Settings.Height <= 0)
    or (Settings.Width mod 16 <> 0)
    or (Settings.Height mod 16 <> 0)
    or (Settings.Steps < 1) then
    raise EArgumentException.Create('Use positive dimensions divisible by 16 and a positive step count.');
  if Settings.AssetsDirectory = '' then
    Settings.AssetsDirectory := TPath.Combine(Root, 'assets/qwen-image-2.1');
  if Settings.OutputFile = '' then
    Settings.OutputFile :=
      TPath.Combine(
      Root,
      'output/'
      + ModelName(Settings.Model)
      + '-'
      + FormatDateTime('yyyymmdd-hhnnss-zzz', Now)
      + '-'
      + IntToStr(Settings.Seed)
      + '.png'
    );
  Settings.OutputFile := ExpandFileName(Settings.OutputFile);
  if not SameText(ExtractFileExt(Settings.OutputFile), '.png') then
    raise EArgumentException.Create('OutputFile must have a .png extension.');
  ForceDirectories(ExtractFileDir(Settings.OutputFile));
  ReportStatus('QuickNN / ' + ModelName(Settings.Model));
  ReportStatus('Models: ' + Directory);
  ReportStatus(
    Format(
      '%dx%d, %d steps, seed %d, CFG %.2f',
      [Settings.Width, Settings.Height, Settings.Steps, Settings.Seed, Settings.Guidance]
    )
  );
  ReportStatus('Prompt: ' + Settings.Prompt);
  Timer := TStopwatch.StartNew;
  case Settings.Model of
    dmQwenImage21Int8, dmQwenImage21BF16:
      Image := GenerateQwen(Settings, Directory);
    dmFlux2Klein4B, dmFlux2Klein9B, dmZImageTurbo:
      Image := GenerateDiffusers(Settings, Directory);
  else
    raise EArgumentOutOfRangeException.Create('Unknown demo model');
  end;
  try
    Q21SavePNG(Image, Settings.OutputFile);
  finally
    Image.Free;
  end;
  ReportStatus('Saved: ' + Settings.OutputFile);
  ReportStatus(Format('Completed in %.2f seconds', [Timer.Elapsed.TotalSeconds]));
  Status := QNNGetGPUStatus;
  if Status.Available then
  begin
    ReportStatus('GPU: ' + Status.DeviceName);
    ReportStatus(
      Format(
        'Activation uploads %d bytes, downloads %d bytes, readbacks %d',
        [Status.ActivationUploadBytes, Status.ActivationDownloadBytes, Status.Readbacks]
      )
    );
  end;
end;

begin
  try
    Run;
  except
    on E: Exception do
    begin
      WriteLn(ErrOutput, E.ClassName + ': ' + E.Message);
      ExitCode := 1;
    end;
  end;
end.

