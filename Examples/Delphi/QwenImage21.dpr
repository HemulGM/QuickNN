program QwenImage21;

{$APPTYPE CONSOLE}

uses
  Winapi.Windows,
  System.SysUtils,
  System.IOUtils,
  System.JSON,
  System.Diagnostics,
  QuickNN.Common,
  QuickNN.LoRA,
  QuickNN.Models.Qwen21,
  QuickNN.Models.Qwen21.VAE,
  QuickNN.Vulkan;

function FindAssets: string;
var
  Dir: string;
begin
  Dir := ExtractFileDir(ParamStr(0));
  for var I := 0 to 7 do
  begin
    Result := TPath.Combine(Dir, 'assets/qwen-image-2.1');
    if TFile.Exists(TPath.Combine(Result, 'tokenizer.json')) then
      Exit;
    Dir := ExtractFileDir(Dir);
  end;
  raise EFileNotFoundException.Create('Cannot locate assets/qwen-image-2.1; use --assets');
end;

procedure Run;
var
  LoraJSON: TJSONArray;
  LoraItem: TJSONObject;
  Params: TQwen21Params;
  Models, Assets, OutputPath, Arg, Precision: string;
  Index: Integer;
  Pipeline: TQwen21Pipeline;
  Image: TQNNImage;
  Timer: TStopwatch;
  JSON: TJSONObject;
  Status: TQNNGPUStatus;

  function NextValue: string;
  begin
    Inc(Index);
    if Index > ParamCount then
      raise EArgumentException.Create('Missing value for ' + Arg);
    Result := ParamStr(Index)
  end;

begin
  Params := TQwen21Params.Defaults;
  Models := 'H:\Models';
  Assets := '';
  OutputPath := '';
  Precision := 'int8';
  Index := 1;
  while Index <= ParamCount do
  begin
    Arg := ParamStr(Index);
    if Arg = '--help' then
    begin
      WriteLn('QwenImage21 [--models H:\Models] [--assets DIR] [--precision int8|bf16]');
      WriteLn('  [--prompt TEXT | --prompt-file UTF8.txt] [--negative TEXT] [--cfg 1]');
      WriteLn('  [--image FILE]... [--reference-size 512] (ordered image references)');
      WriteLn('  [--lora FILE --lora-scale 1]... (repeat for multiple adapters)');
      WriteLn('  [--legacy-gpu] (host-mediated comparison path; default is resident Vulkan)');
      WriteLn('  [--width 512] [--height 512] [--steps 40] [--seed 42] [--output image.png]');
      Exit;
    end
    else if Arg = '--models' then
      Models := NextValue
    else if Arg = '--assets' then
      Assets := NextValue
    else if Arg = '--precision' then
      Precision := NextValue
    else if Arg = '--prompt' then
      Params.Prompt := NextValue
    else if Arg = '--prompt-file' then
      Params.Prompt := TFile.ReadAllText(NextValue, TEncoding.UTF8)
    else if Arg = '--negative' then
      Params.NegativePrompt := NextValue
    else if Arg = '--cfg' then
      Params.Guidance := StrToFloat(NextValue, TFormatSettings.Invariant)
    else if Arg = '--image' then
    begin
      SetLength(Params.ReferenceImages, Length(Params.ReferenceImages) + 1);
      Params.ReferenceImages[High(Params.ReferenceImages)] := NextValue;
    end
    else if Arg = '--reference-size' then
      Params.ReferenceSize := StrToInt(NextValue)
    else if Arg = '--lora' then
    begin
      SetLength(Params.LoRAs, Length(Params.LoRAs) + 1);
      Params.LoRAs[High(Params.LoRAs)] := TQNNLoRASpec.Create(NextValue, 1);
    end
    else if Arg = '--lora-scale' then
    begin
      if Length(Params.LoRAs) = 0 then
        raise EArgumentException.Create('--lora-scale requires a preceding --lora');
      Params.LoRAs[High(Params.LoRAs)].Strength := StrToFloat(NextValue, TFormatSettings.Invariant);
    end
    else if Arg = '--legacy-gpu' then
      Params.ResidentGPU := False
    else if Arg = '--width' then
      Params.Width := StrToInt(NextValue)
    else if Arg = '--height' then
      Params.Height := StrToInt(NextValue)
    else if Arg = '--steps' then
      Params.Steps := StrToInt(NextValue)
    else if Arg = '--seed' then
      Params.Seed := StrToInt64(NextValue)
    else if Arg = '--output' then
      OutputPath := NextValue
    else
      raise EArgumentException.Create('Unknown option: ' + Arg);
    Inc(Index);
  end;
  if not ((Precision = 'int8') or (Precision = 'bf16')) then
    raise EArgumentException.Create('Precision must be int8 or bf16');
  if Assets = '' then
    Assets := FindAssets;
  if OutputPath = '' then
    OutputPath :=
      TPath.Combine(
      ExtractFileDir(ExtractFileDir(Assets)),
      'output/qwen-image21-' + Precision + '-' + IntToStr(Params.Seed) + '.png'
    );
  OutputPath := ExpandFileName(OutputPath);
  if not SameText(ExtractFileExt(OutputPath), '.png') then
    raise EArgumentException.Create('Output must be PNG');
  ForceDirectories(ExtractFileDir(OutputPath));
  WriteLn('Qwen-Image 2.1 / native Pascal + Vulkan / ' + Precision);
  WriteLn(
    Format(
      '%dx%d, %d steps, seed %d, CFG %.2f',
      [Params.Width, Params.Height, Params.Steps, Params.Seed, Params.Guidance]
    )
  );
  WriteLn('Prompt: ' + Params.Prompt);
  Flush(Output);
  QNNConfigureGPU(gpuForce);
  Timer := TStopwatch.StartNew;
  Pipeline :=
    TQwen21Pipeline.Create(
    Models,
    Assets,
    Precision = 'int8',
    procedure(const Msg: string)
    begin
      WriteLn(Msg);
      Flush(Output)
    end);
  try
    Image := Pipeline.Generate(Params);
    try
      Q21SavePNG(Image, OutputPath)
    finally
      Image.Free
    end;
    Status := QNNGetGPUStatus;
    JSON := TJSONObject.Create;
    try
      JSON.AddPair('model_root', ExpandFileName(Models));
      JSON.AddPair('assets_root', ExpandFileName(Assets));
      JSON.AddPair('model', 'Qwen-Image 2.1');
      JSON.AddPair('precision', Precision);
      JSON.AddPair('prompt', Params.Prompt);
      JSON.AddPair('negative_prompt', Params.NegativePrompt);
      JSON.AddPair('width', TJSONNumber.Create(Params.Width));
      JSON.AddPair('height', TJSONNumber.Create(Params.Height));
      JSON.AddPair('steps', TJSONNumber.Create(Params.Steps));
      JSON.AddPair('seed', TJSONNumber.Create(Params.Seed));
      JSON.AddPair('guidance', TJSONNumber.Create(Params.Guidance));
      JSON.AddPair('seconds', TJSONNumber.Create(Timer.Elapsed.TotalSeconds));
      JSON.AddPair('gpu', Status.DeviceName);
      JSON.AddPair('activation_precision', 'F32');
      JSON.AddPair('resident_gpu', TJSONBool.Create(Params.ResidentGPU));
      JSON.AddPair('gpu_activation_upload_bytes', TJSONNumber.Create(Int64(Status.ActivationUploadBytes)));
      JSON.AddPair('gpu_activation_download_bytes', TJSONNumber.Create(Int64(Status.ActivationDownloadBytes)));
      JSON.AddPair('gpu_readbacks', TJSONNumber.Create(Int64(Status.Readbacks)));
      JSON.AddPair('gpu_submissions', TJSONNumber.Create(Int64(Status.Submissions)));
      JSON.AddPair('gpu_convrot_calls', TJSONNumber.Create(Int64(Status.ConvRotCalls)));
      JSON.AddPair('gpu_tensor_calls', TJSONNumber.Create(Int64(Status.TensorCalls)));
      JSON.AddPair('gpu_resident_peak_bytes', TJSONNumber.Create(Int64(Status.ResidentPeakBytes)));
      var References := TJSONArray.Create;
      JSON.AddPair('reference_images', References);
      for var Path in Params.ReferenceImages do
        References.Add(ExpandFileName(Path));
      JSON.AddPair('reference_size', TJSONNumber.Create(Params.ReferenceSize));
      LoraJSON := TJSONArray.Create;
      JSON.AddPair('loras', LoraJSON);
      for var Spec in Params.LoRAs do
      begin
        var LoraPath := Spec.FileName;
        if not TFile.Exists(LoraPath) and not TPath.IsPathRooted(LoraPath) then
          LoraPath := TPath.Combine(TPath.Combine(Models, 'loras'), LoraPath);
        LoraItem := TJSONObject.Create;
        LoraItem.AddPair('file', ExpandFileName(LoraPath));
        LoraItem.AddPair('strength', TJSONNumber.Create(Spec.Strength));
        LoraJSON.AddElement(LoraItem);
      end;
      JSON.AddPair('gpu_attention_calls', TJSONNumber.Create(Int64(Status.AttentionCalls)));
      JSON.AddPair('gpu_gemm_calls', TJSONNumber.Create(Int64(Status.GemmCalls)));
      TFile.WriteAllText(ChangeFileExt(OutputPath, '.json'), JSON.Format(2), TEncoding.UTF8);
    finally
      JSON.Free
    end;
    WriteLn('Saved: ' + OutputPath);
    WriteLn(
      Format('Total %.2f s, GPU %s, GEMMs %d', [Timer.Elapsed.TotalSeconds, Status.DeviceName, Status.GemmCalls])
    );
  finally
    Pipeline.Free;
    QNNConfigureGPU(gpuOff)
  end;
end;

begin
  SetTextCodePage(Output, 65001);
  var PreviousConsoleCP := GetConsoleOutputCP;
  SetConsoleOutputCP(65001);
  try
    try
      Run
    except
      on E: Exception do
      begin
        WriteLn('FAILED: ' + E.ClassName + ': ' + E.Message);
        ExitCode := 1
      end
    end;
    Flush(Output);
  finally
    if PreviousConsoleCP <> 0 then
      SetConsoleOutputCP(PreviousConsoleCP)
  end;
end.

