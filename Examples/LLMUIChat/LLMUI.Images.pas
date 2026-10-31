unit LLMUI.Images;

interface

uses
  System.SysUtils, System.JSON, QuickNN.Models.LLM.Types;

type
  TImagePipelineKind = (ipFlux, ipQwenImage, ipZImage);

  TChatImageSettings = record
    Kind: TImagePipelineKind;
    ModelId, Directory, AssetsDirectory, DiffusionCheckpoint, AnalysisModelId: string;
    UseInt8: Boolean;
    Width, Height, Steps: Integer;
    Seed: Int64;
    Guidance: Single;
    class function Defaults: TChatImageSettings; static;
    function ToJSON: TJSONObject;
    class function FromJSON(const Value: TJSONValue): TChatImageSettings; static;
  end;

  TImageModelEntry = class
  public
    Id, Name, Validation: string;
    Available: Boolean;
    Settings: TChatImageSettings;
    constructor Create;
    procedure Inspect;
    function Resolve(const Preferences: TChatImageSettings): TChatImageSettings;
    function ToJSON: TJSONObject;
    class function FromJSON(const Value: TJSONValue): TImageModelEntry; static;
  end;

  TChatImage = record
    RelativePath, Prompt, ModelName, Description, ContentHash: string;
    IsUserImage: Boolean;
    OriginalName: string;
    Settings: TChatImageSettings;
    function FileName: string;
    function ToJSON: TJSONObject;
    class function FromJSON(const Value: TJSONValue): TChatImage; static;
    function LLMImage: TLLMImage;
  end;

  TImageDescription = record
    MessageId, ContentHash, Text: string;
  end;

  TImagePhase = reference to procedure(const Text: string);

function ImagePipelineName(const Kind: TImagePipelineKind): string;

function FindQwenImageAssets: string;

procedure ValidateImageSettings(const Settings: TChatImageSettings);

procedure GenerateChatImage(var Image: TChatImage; const OnPhase: TImagePhase; const Cancelled: TFunc<Boolean>);

implementation

uses
  System.IOUtils, System.Math, System.Classes, System.Hash, QuickNN.Common,
  QuickNN.Vulkan, QuickNN.GGUF, QuickNN.Models.Flux, QuickNN.Models.ZImage,
  QuickNN.Models.Qwen21, QuickNN.Models.Qwen21.VAE;

var
  ActivePhase: TImagePhase;

var
  ActiveCancel: TFunc<Boolean>;

procedure Progress(const Index, Total: LongInt);
begin
  if Assigned(ActiveCancel) and ActiveCancel() then
    raise EAbort.Create('Генерация остановлена');
  if Assigned(ActivePhase) then
    ActivePhase(Format('Генерация изображения: %d/%d', [Index + 1, Total]));
end;

procedure Phase(const Text: string; const Done: Boolean);
begin
  if not Done and Assigned(ActivePhase) then
    ActivePhase(Text);
end;

class function TChatImageSettings.Defaults: TChatImageSettings;
begin
  Result := Default(TChatImageSettings);
  Result.UseInt8 := True;
  Result.Width := 256;
  Result.Height := 256;
  Result.Steps := 4;
  Result.Seed := 42;
  Result.Guidance := 1;
end;

function TChatImageSettings.ToJSON: TJSONObject;
begin
  Result := TJSONObject.Create;
  Result.AddPair('kind', TJSONNumber.Create(Ord(Kind)));
  Result.AddPair('model_id', ModelId);
  Result.AddPair('directory', Directory);
  Result.AddPair('assets_directory', AssetsDirectory);
  Result.AddPair('diffusion_checkpoint', DiffusionCheckpoint);
  Result.AddPair('use_int8', UseInt8);
  Result.AddPair('analysis_model_id', AnalysisModelId);
  Result.AddPair('width', TJSONNumber.Create(Width));
  Result.AddPair('height', TJSONNumber.Create(Height));
  Result.AddPair('steps', TJSONNumber.Create(Steps));
  Result.AddPair('seed', TJSONNumber.Create(Seed));
  Result.AddPair('guidance', TJSONNumber.Create(Guidance));
end;

class function TChatImageSettings.FromJSON(const Value: TJSONValue): TChatImageSettings;
begin
  Result := Defaults;
  if not (Value is TJSONObject) then
    Exit;
  Result.Kind := TImagePipelineKind(EnsureRange(Value.GetValue<Integer>('kind', 0), 0, 2));
  Result.ModelId := Value.GetValue<string>('model_id', '');
  Result.Directory := Value.GetValue<string>('directory', '');
  Result.AssetsDirectory := Value.GetValue<string>('assets_directory', '');
  Result.DiffusionCheckpoint := Value.GetValue<string>('diffusion_checkpoint', '');
  Result.UseInt8 := Value.GetValue<Boolean>('use_int8', True);
  Result.AnalysisModelId := Value.GetValue<string>('analysis_model_id', '');
  Result.Width := EnsureRange(Value.GetValue<Integer>('width', 256), 64, 1024) div 16 * 16;
  Result.Height := EnsureRange(Value.GetValue<Integer>('height', 256), 64, 1024) div 16 * 16;
  Result.Steps := EnsureRange(Value.GetValue<Integer>('steps', 4), 1, 100);
  Result.Seed := Value.GetValue<Int64>('seed', 42);
  Result.Guidance := EnsureRange(Value.GetValue<Double>('guidance', 1), 0.0, 20.0);
end;

function ImagePipelineName(const Kind: TImagePipelineKind): string;
begin
  case Kind of
    ipFlux:
      Result := 'FLUX.2 Klein';
    ipQwenImage:
      Result := 'Qwen-Image 2.1';
    ipZImage:
      Result := 'Z-Image Turbo';
  end;
end;

function FindQwenImageAssets: string;
begin
  Result := ExtractFileDir(ParamStr(0));
  repeat
    var Candidate := TPath.Combine(Result, 'assets\qwen-image-2.1');
    if TFile.Exists(TPath.Combine(Candidate, 'tokenizer.json')) then
      Exit(Candidate);
    var Parent := ExtractFileDir(Result);
    if Parent = Result then
      Exit('');
    Result := Parent;
  until False;
end;

procedure ValidateImageSettings(const Settings: TChatImageSettings);
begin
  var Step := 16;
  if Settings.Kind = ipQwenImage then
    Step := 32;
  if (Settings.Width < 64) or (Settings.Height < 64)
    or (Settings.Width > 1024) or (Settings.Height > 1024)
    or (Settings.Width mod Step <> 0) or (Settings.Height mod Step <> 0) then
    raise EArgumentException.CreateFmt('Размеры должны быть от 64 до 1024 и кратны %d.', [Step]);
  if (Settings.Steps < 1) or (Settings.Steps > 100)
    or IsNan(Settings.Guidance) or IsInfinite(Settings.Guidance)
    or (Settings.Guidance < 0) or (Settings.Guidance > 20)
    or ((Settings.Kind = ipQwenImage) and (Settings.Guidance < 1)) then
    raise EArgumentException.Create('Проверьте число шагов (1–100) и guidance (0–20; Qwen: 1–20).');
end;

constructor TImageModelEntry.Create;
begin
  inherited;
  Id := TGUID.NewGuid.ToString;
  Settings := TChatImageSettings.Defaults;
end;

procedure TImageModelEntry.Inspect;

  procedure RequireFile(const FileName: string);
  begin
    if not TFile.Exists(FileName) then
      raise EFileNotFoundException.Create('Не найден файл: ' + FileName);
  end;

  procedure RequireWeights(const Directory, BaseName: string);
  begin
    var Index := TPath.Combine(Directory, BaseName + '.safetensors.index.json');
    if not TFile.Exists(Index) then
    begin
      RequireFile(TPath.Combine(Directory, BaseName + '.safetensors'));
      Exit;
    end;
    var JSON := TJSONValue.ParseJSONValue(TFile.ReadAllText(Index, TEncoding.UTF8));
    try
      if not (JSON is TJSONObject) or not (JSON.FindValue('weight_map') is TJSONObject) then
        raise EArgumentException.Create('Повреждён индекс весов: ' + Index);
      var Map := TJSONObject(JSON.FindValue('weight_map'));
      if Map.Count = 0 then
        raise EArgumentException.Create('Пустой индекс весов: ' + Index);
      for var Pair in Map do
        RequireFile(TPath.Combine(Directory, Pair.JsonValue.Value));
    finally
      JSON.Free;
    end;
  end;

begin
  Available := False;
  try
    if Settings.Directory.Trim = '' then
      raise EArgumentException.Create('Укажите каталог модели.');
    if not TDirectory.Exists(Settings.Directory) then
      raise EDirectoryNotFoundException.Create('Каталог модели не найден: ' + Settings.Directory);
    ValidateImageSettings(Settings);
    if Settings.Kind = ipQwenImage then
    begin
      var Suffix := 'bf16';
      if Settings.UseInt8 then
        Suffix := 'int8_convrot';
      RequireFile(TPath.Combine(Settings.Directory, 'text_encoders/qwen3vl_8b_' + Suffix + '.safetensors'));
      if Settings.DiffusionCheckpoint = '' then
        RequireFile(TPath.Combine(Settings.Directory, 'diffusion_models/qwen_image_2.1_' + Suffix + '.safetensors'))
      else
      begin
        RequireFile(Settings.DiffusionCheckpoint);
        if SameText(ExtractFileExt(Settings.DiffusionCheckpoint), '.gguf') then
        begin
          var Header := TGGUFHeader.ReadFile(Settings.DiffusionCheckpoint);
          if Header.Architecture <> 'qwen_image21' then
            raise EArgumentException.Create('Ожидается GGUF qwen_image21, обнаружена архитектура: ' + Header.Architecture);
          if Length(Header.Tensors) = 0 then
            raise EArgumentException.Create('GGUF не содержит тензоров.');
        end;
      end;
      RequireFile(TPath.Combine(Settings.Directory, 'vae/qwen_image_2.1_vae_bf16.safetensors'));
      if Settings.AssetsDirectory = '' then
        raise EArgumentException.Create('Укажите каталог конфигураций и токенизатора Qwen-Image.');
      for var FileName in TArray<string>.Create('tokenizer.json', 'encoder_config.json', 'vae_config.json', 'scheduler_config.json') do
        RequireFile(TPath.Combine(Settings.AssetsDirectory, FileName));
    end
    else
    begin
      var Index := TPath.Combine(Settings.Directory, 'model_index.json');
      RequireFile(Index);
      var JSON := TJSONValue.ParseJSONValue(TFile.ReadAllText(Index, TEncoding.UTF8));
      try
        if not (JSON is TJSONObject) then
          raise EArgumentException.Create('Повреждён model_index.json.');
        var Pipeline := JSON.GetValue<string>('_class_name', '');
        if ((Settings.Kind = ipFlux) and not Pipeline.StartsWith('Flux2', True))
          or ((Settings.Kind = ipZImage) and not Pipeline.StartsWith('ZImage', True)
          and not SameText(Pipeline, 'Z-Image')) then
          raise EArgumentException.Create('Выбранный тип не соответствует model_index.json: ' + Pipeline);
      finally
        JSON.Free
      end;
      for var Part in TArray<string>.Create('transformer', 'text_encoder', 'vae') do
        RequireFile(TPath.Combine(Settings.Directory, Part + '/config.json'));
      RequireFile(TPath.Combine(Settings.Directory, 'tokenizer/tokenizer.json'));
      RequireWeights(TPath.Combine(Settings.Directory, 'transformer'), 'diffusion_pytorch_model');
      RequireWeights(TPath.Combine(Settings.Directory, 'text_encoder'), 'model');
      RequireFile(TPath.Combine(Settings.Directory, 'vae/diffusion_pytorch_model.safetensors'));
    end;
    Available := True;
    Validation := ImagePipelineName(Settings.Kind) + ' · файлы доступны. Эти параметры используются при выборе модели в чате.';
  except
    on E: Exception do
      Validation := E.Message;
  end;
end;

function TImageModelEntry.Resolve(const Preferences: TChatImageSettings): TChatImageSettings;
begin
  Result := Preferences;
  Result.ModelId := Id;
  Result.Kind := Settings.Kind;
  Result.Directory := Settings.Directory;
  Result.AssetsDirectory := Settings.AssetsDirectory;
  Result.DiffusionCheckpoint := Settings.DiffusionCheckpoint;
  Result.UseInt8 := Settings.UseInt8;
end;

function TImageModelEntry.ToJSON: TJSONObject;
begin
  Result := TJSONObject.Create;
  Result.AddPair('id', Id);
  Result.AddPair('name', Name);
  Result.AddPair('settings', Settings.ToJSON);
end;

class function TImageModelEntry.FromJSON(const Value: TJSONValue): TImageModelEntry;
begin
  Result := TImageModelEntry.Create;
  try
    Result.Id := Value.GetValue<string>('id', Result.Id);
    Result.Name := Value.GetValue<string>('name', 'Image model');
    Result.Settings := TChatImageSettings.FromJSON(Value.FindValue('settings'));
    Result.Inspect;
  except
    Result.Free;
    raise;
  end;
end;

function TChatImage.FileName: string;
begin
  var Root := TPath.GetFullPath(TPath.Combine(GetEnvironmentVariable('LOCALAPPDATA'), 'QuickNN\LLMUIChat'));
  Result := TPath.GetFullPath(TPath.Combine(Root, RelativePath));
  if (RelativePath = '') or TPath.IsPathRooted(RelativePath)
    or not Result.StartsWith(IncludeTrailingPathDelimiter(Root), True) then
    raise EArgumentException.Create('Invalid chat image path');
end;

function TChatImage.ToJSON: TJSONObject;
begin
  Result := TJSONObject.Create;
  Result.AddPair('user_image', IsUserImage);
  Result.AddPair('original_name', OriginalName);
  Result.AddPair('sha256', ContentHash);
  Result.AddPair('file', RelativePath);
  Result.AddPair('prompt', Prompt);
  Result.AddPair('model', ModelName);
  Result.AddPair('description', Description);
  Result.AddPair('settings', Settings.ToJSON);
end;

class function TChatImage.FromJSON(const Value: TJSONValue): TChatImage;
begin
  Result.IsUserImage := Value.GetValue<Boolean>('user_image', False);
  Result.OriginalName := Value.GetValue<string>('original_name', '');
  Result.ContentHash := Value.GetValue<string>('sha256', '');
  Result.RelativePath := Value.GetValue<string>('file', '');
  Result.Prompt := Value.GetValue<string>('prompt', '');
  Result.ModelName := Value.GetValue<string>('model', '');
  Result.Description := Value.GetValue<string>('description', '');
  Result.Settings := TChatImageSettings.FromJSON(Value.FindValue('settings'));
  Result.FileName; // Reject paths outside the application image archive.
end;

function TChatImage.LLMImage: TLLMImage;
begin
  Result := Default(TLLMImage);
  Result.FileName := FileName;
  Result.Identity := ContentHash;
  Result.Description := Description;
end;

procedure GenerateChatImage(var Image: TChatImage; const OnPhase: TImagePhase; const Cancelled: TFunc<Boolean>);
var
  Flux: TQNNFlux;
  ZImage: TQNNZImage;
  Qwen: TQwen21Pipeline;
  Params: TGenerateParams;
  QParams: TQwen21Params;
  Output: TQNNImage;
  PreviousLimit: UInt64;
  Destination, Temporary, Assets: string;
begin
  ValidateImageSettings(Image.Settings);
  if not TDirectory.Exists(Image.Settings.Directory) then
    raise EDirectoryNotFoundException.Create('Image model directory not found');
  Destination := Image.FileName;
  ForceDirectories(ExtractFileDir(Destination));
  Temporary := Destination + '.tmp';
  ActivePhase := OnPhase;
  ActiveCancel := Cancelled;
  Output := Default(TQNNImage);
  PreviousLimit := QNNGetGPUStatus.WeightCacheLimit;
  try
    QNNConfigureGPU(gpuForce);
    QNNSetGPUWeightCacheLimit(UInt64(8) * 1024 * 1024 * 1024);
    StepCallback := Progress;
    TextProgressCallback := Progress;
    VAEProgressCallback := Progress;
    Params := Default(TGenerateParams);
    Params.Width := Image.Settings.Width;
    Params.Height := Image.Settings.Height;
    Params.NumSteps := Image.Settings.Steps;
    Params.Seed := Image.Settings.Seed;
    Params.Guidance := Image.Settings.Guidance;
    Params.PowerAlpha := 2;
    if Image.Settings.Kind = ipFlux then
    begin
      Flux := TQNNFlux.Load(Image.Settings.Directory, Phase);
      try
        Flux.ResidentGPU := True;
        Output := Flux.Generate(UTF8String(Image.Prompt), Params)
      finally
        Flux.Free
      end;
    end
    else if Image.Settings.Kind = ipZImage then
    begin
      ZImage := TQNNZImage.Load(Image.Settings.Directory, Phase);
      try
        ZImage.ResidentGPU := True;
        Output := ZImage.Generate(UTF8String(Image.Prompt), Params)
      finally
        ZImage.Free
      end;
    end
    else
    begin
      Assets := Image.Settings.AssetsDirectory;
      if Assets = '' then
        Assets := FindQwenImageAssets; // Backward-compatible archived settings.
      QParams := TQwen21Params.Defaults;
      QParams.Prompt := Image.Prompt;
      QParams.Width := Params.Width;
      QParams.Height := Params.Height;
      QParams.Steps := Params.NumSteps;
      QParams.Seed := Params.Seed;
      QParams.Guidance := Params.Guidance;
      QParams.ResidentGPU := True;
      Qwen := TQwen21Pipeline.Create(Image.Settings.Directory, Assets, Image.Settings.UseInt8,
        procedure(const Text: string)
        begin
          if Assigned(Cancelled) and Cancelled() then
            raise EAbort.Create('Генерация остановлена');
          if Assigned(OnPhase) then
            OnPhase(Text);
        end, Image.Settings.DiffusionCheckpoint);
      try
        Output := Qwen.Generate(QParams)
      finally
        Qwen.Free
      end;
    end;
    if Assigned(Cancelled) and Cancelled() then
      raise EAbort.Create('Генерация остановлена');
    Q21SavePNG(Output, Temporary);
    TFile.Move(Temporary, Destination);
    var Source := TFileStream.Create(Destination, fmOpenRead or fmShareDenyWrite);
    try
      Image.ContentHash := THashSHA2.GetHashString(Source)
    finally
      Source.Free
    end;
  finally
    Output.Free;
    StepCallback := nil;
    TextProgressCallback := nil;
    VAEProgressCallback := nil;
    PhaseCallback := nil;
    ActivePhase := nil;
    ActiveCancel := nil;
    QNNGPUClearWeightCache;
    QNNSetGPUWeightCacheLimit(PreviousLimit);
    if TFile.Exists(Temporary) then
      TFile.Delete(Temporary);
  end;
end;

end.

