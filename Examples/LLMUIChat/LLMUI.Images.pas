unit LLMUI.Images;

interface

uses
  System.SysUtils, System.JSON, QuickNN.Models.LLM.Types, QuickNN.ModelFiles;

type
  TImagePipelineKind = (ipFlux, ipQwenImage, ipZImage, ipFlux1);

  TChatImageSettings = record
    Kind: TImagePipelineKind;
    ModelId, Directory, AssetsDirectory, DiffusionCheckpoint, AnalysisModelId: string;
    LoRAFile, NegativePrompt: string;
    SourceFile: string;
    Components: TQNNModelFiles;
    Adapters: TArray<string>;
    LoRAStrength: Single;
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
    Available, SupportsNegativePrompt: Boolean;
    Settings: TChatImageSettings;
    constructor Create;
    procedure Inspect;
    function Resolve(const Preferences: TChatImageSettings): TChatImageSettings;
    function ToJSON: TJSONObject;
    class function FromJSON(const Value: TJSONValue): TImageModelEntry; static;
  end;

  TReferenceAnalysisSettings = record
    ModelId, ModelName, Directory, TokenizerDirectory, Instruction: string;
    Context, MaxTokens: Integer;
    Temperature: Single;
    function ToJSON: TJSONObject;
    class function FromJSON(const Value: TJSONValue): TReferenceAnalysisSettings; static;
  end;

  TChatImage = record
    RelativePath, Prompt, ModelName, Description, ContentHash: string;
    IsUserImage: Boolean;
    OriginalName: string;
    ReferencePaths: TArray<string>;
    ReferenceAnalysis: TReferenceAnalysisSettings;
    ReferenceDescription: string;
    Settings: TChatImageSettings;
    function FileName: string;
    function ToJSON: TJSONObject;
    class function FromJSON(const Value: TJSONValue): TChatImage; static;
    function LLMImage: TLLMImage;
    function AsReference: TChatImage;
    function ReferenceFiles: TArray<string>;
    function NeedsReferenceAnalysis: Boolean;
    function GenerationPrompt: string;
  end;

  TImageDescription = record
    MessageId, ContentHash, Text: string;
  end;

  TImagePhase = reference to procedure(const Text: string);

function ImagePipelineName(const Kind: TImagePipelineKind): string;

function ImageSupportsLoRA(const Settings: TChatImageSettings): Boolean;

function ImageSupportsNegativePrompt(const Settings: TChatImageSettings): Boolean;

function ImageSupportsReferences(const Kind: TImagePipelineKind): Boolean;

function FindQwenImageAssets(const ModelDirectory: string): string;

procedure ValidateImageSettings(const Settings: TChatImageSettings);

procedure GenerateChatImage(var Image: TChatImage; const OnPhase: TImagePhase; const Cancelled: TFunc<Boolean>);

implementation

uses
  System.IOUtils, System.Math, System.Classes, System.Hash,
  System.Generics.Collections, QuickNN.Common, QuickNN.Vulkan, QuickNN.GGUF,
  QuickNN.Models.Flux, QuickNN.Models.ZImage, QuickNN.Models.Qwen21,
  QuickNN.Models.Qwen21.VAE, QuickNN.Models.Qwen21.Conditioning,
  QuickNN.Models.Flux1, QuickNN.LoRA;

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
  Result.LoRAStrength := 1;
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
  Result.AddPair('source_file', SourceFile);
  Result.AddPair('components', Components.ToJSON);
  var AdapterFiles := TJSONArray.Create;
  for var FileName in Adapters do
    AdapterFiles.Add(FileName);
  Result.AddPair('adapters', AdapterFiles);
  Result.AddPair('model_id', ModelId);
  Result.AddPair('directory', Directory);
  Result.AddPair('assets_directory', AssetsDirectory);
  Result.AddPair('diffusion_checkpoint', DiffusionCheckpoint);
  Result.AddPair('use_int8', UseInt8);
  Result.AddPair('lora_file', LoRAFile);
  Result.AddPair('lora_strength', TJSONNumber.Create(LoRAStrength));
  Result.AddPair('analysis_model_id', AnalysisModelId);
  Result.AddPair('width', TJSONNumber.Create(Width));
  Result.AddPair('height', TJSONNumber.Create(Height));
  Result.AddPair('steps', TJSONNumber.Create(Steps));
  Result.AddPair('seed', TJSONNumber.Create(Seed));
  Result.AddPair('guidance', TJSONNumber.Create(Guidance));
  Result.AddPair('negative_prompt', NegativePrompt);
end;

class function TChatImageSettings.FromJSON(const Value: TJSONValue): TChatImageSettings;
begin
  Result := Defaults;
  if not (Value is TJSONObject) then
    Exit;
  Result.Kind := TImagePipelineKind(EnsureRange(Value.GetValue<Integer>('kind', 0), 0, Ord(High(TImagePipelineKind))));
  Result.ModelId := Value.GetValue<string>('model_id', '');
  Result.SourceFile := Value.GetValue<string>('source_file', '');
  Result.Components := TQNNModelFiles.FromJSON(Value.FindValue('components'));
  var AdapterFiles := Value.FindValue('adapters');
  if AdapterFiles is TJSONArray then
  begin
    SetLength(Result.Adapters, Min(128, TJSONArray(AdapterFiles).Count));
    for var I := 0 to High(Result.Adapters) do
      Result.Adapters[I] := TJSONArray(AdapterFiles).Items[I].Value;
  end;
  Result.Directory := Value.GetValue<string>('directory', '');
  Result.AssetsDirectory := Value.GetValue<string>('assets_directory', '');
  if Result.Kind = ipQwenImage then
    Result.AssetsDirectory := Qwen21AssetsDirectory(Result.Directory, Result.AssetsDirectory);
  Result.DiffusionCheckpoint := Value.GetValue<string>('diffusion_checkpoint', '');
  Result.UseInt8 := Value.GetValue<Boolean>('use_int8', True);
  Result.LoRAFile := Value.GetValue<string>('lora_file', '');
  Result.LoRAStrength := EnsureRange(Value.GetValue<Single>('lora_strength', 1), -4.0, 4.0);
  Result.AnalysisModelId := Value.GetValue<string>('analysis_model_id', '');
  var SizeStep := 16;
  var MaxSize := 1024;
  if Result.Kind = ipFlux1 then
    MaxSize := 2048;
  if Result.Kind = ipQwenImage then
  begin
    SizeStep := 8;
    MaxSize := 2048;
  end;
  Result.Width := EnsureRange(Value.GetValue<Integer>('width', 256), 64, MaxSize) div SizeStep * SizeStep;
  Result.Height := EnsureRange(Value.GetValue<Integer>('height', 256), 64, MaxSize) div SizeStep * SizeStep;
  Result.Steps := EnsureRange(Value.GetValue<Integer>('steps', 4), 1, 100);
  Result.Seed := Value.GetValue<Int64>('seed', 42);
  Result.Guidance := EnsureRange(Value.GetValue<Double>('guidance', 1), 0.0, 20.0);
  Result.NegativePrompt := Value.GetValue<string>('negative_prompt', '');
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
    ipFlux1:
      Result := 'FLUX.1-dev';
  end;
end;

function ImageSupportsLoRA(const Settings: TChatImageSettings): Boolean;
begin
  Result := Settings.Kind in [ipFlux1, ipQwenImage];
  if Settings.Kind <> ipFlux then Exit;
  try
    var FileName := TPath.Combine(Settings.Directory, 'transformer/config.json');
    if not TFile.Exists(FileName) or (TFile.GetSize(FileName) > 1024 * 1024) then Exit;
    var Config := TJSONValue.ParseJSONValue(TFile.ReadAllText(FileName, TEncoding.UTF8));
    try
      Result := (Config is TJSONObject)
        and (Config.GetValue<Integer>('num_attention_heads', 0) = 24)
        and (Config.GetValue<Integer>('attention_head_dim', 0) = 128)
        and (Config.GetValue<Integer>('num_layers', 0) = 5)
        and (Config.GetValue<Integer>('num_single_layers', 0) = 20)
        and (Config.GetValue<Integer>('joint_attention_dim', 0) = 7680);
    finally Config.Free end;
  except on E: Exception do Result := False end;
end;

function ImageSupportsNegativePrompt(const Settings: TChatImageSettings): Boolean;
begin
  Result := Settings.Kind = ipQwenImage;
  if Settings.Kind <> ipFlux then
    Exit;
  // Match TQNNFlux.Load: without a model index, Klein defaults to distilled.
  var FileName := TPath.Combine(Settings.Directory, 'model_index.json');
  if not TFile.Exists(FileName) then
    Exit;
  try
    if TFile.GetSize(FileName) > 1024 * 1024 then
      Exit;
    var JSON := TJSONValue.ParseJSONValue(TFile.ReadAllText(FileName, TEncoding.UTF8));
    try
      Result := (JSON is TJSONObject) and not JSON.GetValue<Boolean>('is_distilled', False);
    finally
      JSON.Free
    end;
  except
    on E: Exception do
      Result := False;
  end;
end;

function ImageSupportsReferences(const Kind: TImagePipelineKind): Boolean;
begin
  Result := Kind in [ipFlux, ipQwenImage];
end;

function FindQwenImageAssets(const ModelDirectory: string): string;
begin
  Result := Qwen21AssetsDirectory(ModelDirectory);
end;

procedure ValidateImageSettings(const Settings: TChatImageSettings);
begin
  var Step := 16;
  var MaxSize := 1024;
  if Settings.Kind = ipFlux1 then
  begin
    MaxSize := 2048;
    if Int64(Settings.Width div 16) * (Settings.Height div 16) + 512 > 6144 then
      raise EArgumentException.Create('Размер изображения превышает бюджет FLUX.1-dev.');
  end;
  if Settings.LoRAFile <> '' then
  begin
    if not ImageSupportsLoRA(Settings) then
      raise EArgumentException.Create('LoRA доступна для FLUX.1-dev, FLUX.2 Klein 4B и Qwen-Image 2.1.');
    if not TFile.Exists(Settings.LoRAFile) then
      raise EFileNotFoundException.Create('Файл LoRA не найден: ' + Settings.LoRAFile);
  end;
  if IsNan(Settings.LoRAStrength) or IsInfinite(Settings.LoRAStrength)
    or (Abs(Settings.LoRAStrength) > 4) then
    raise EArgumentException.Create('Сила LoRA должна быть от -4 до 4.');
  if Settings.Kind = ipQwenImage then
  begin
    Step := 8;
    MaxSize := 2048;
    if Int64((Settings.Width + 31) div 32) * ((Settings.Height + 31) div 32) * 4 >= 6144 then
      raise EArgumentException.Create('Размер изображения превышает бюджет модели в 6144 токена.');
  end;
  if (Settings.Width < 64) or (Settings.Height < 64)
    or (Settings.Width > MaxSize) or (Settings.Height > MaxSize)
    or (Settings.Width mod Step <> 0) or (Settings.Height mod Step <> 0) then
    raise EArgumentException.CreateFmt('Размеры должны быть от 64 до %d и кратны %d.', [MaxSize, Step]);
  if (Settings.Steps < 1) or (Settings.Steps > 100)
    or IsNan(Settings.Guidance) or IsInfinite(Settings.Guidance)
    or (Settings.Guidance < 0) or (Settings.Guidance > 20)
    or ((Settings.Kind = ipQwenImage) and (Settings.Guidance < 1)) then
    raise EArgumentException.Create('Проверьте число шагов (1–100) и guidance (0–20; Qwen: 1–20).');
  if Settings.NegativePrompt.Trim <> '' then
  begin
    if not ImageSupportsNegativePrompt(Settings) then
      raise EArgumentException.Create('Эта модель не поддерживает негативный промпт. Выберите совместимую модель или сбросьте негативный промпт.');
    if Settings.Guidance <= 1 then
      raise EArgumentException.Create('Для негативного промпта установите guidance больше 1.');
  end;
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
  SupportsNegativePrompt := ImageSupportsNegativePrompt(Settings);
  try
    if Settings.Directory.Trim = '' then
      raise EArgumentException.Create('Укажите каталог модели.');
    if not TDirectory.Exists(Settings.Directory) then
      raise EDirectoryNotFoundException.Create('Каталог модели не найден: ' + Settings.Directory);
    if Settings.Kind = ipQwenImage then
      Settings.AssetsDirectory := Qwen21AssetsDirectory(Settings.Directory, Settings.AssetsDirectory);
    ValidateImageSettings(Settings);
    if Settings.Kind = ipFlux1 then
    begin
      var Checkpoint := Settings.DiffusionCheckpoint;
      if Checkpoint = '' then
        Checkpoint := TPath.Combine(Settings.Directory, 'flux1-dev.safetensors');
      RequireFile(Checkpoint);
      if not SameText(ExtractFileExt(Checkpoint), '.safetensors') then
        raise EArgumentException.Create('FLUX.1-dev ожидает базовые веса safetensors.');
      var Header := ReadModelTensorHeader(Checkpoint);
      try
        if (Header.GetValue('img_in.weight') = nil) or (Header.GetValue('double_blocks.0.img_attn.qkv.weight') = nil)
          or (Header.GetValue('guidance_in.in_layer.weight') = nil) then
          raise EArgumentException.Create('Ожидается базовая FLUX.1-dev; файл LoRA не является самостоятельной моделью.');
      finally
        Header.Free
      end;
      for var Part in TArray<string>.Create('transformer', 'text_encoder', 'text_encoder_2', 'vae') do
      begin
        var ComponentRoot := TPath.Combine(Settings.Directory, Part);
        if Part = 'text_encoder' then
          ComponentRoot := ModelComponentDirectory(Settings.Components.TextEncoder, ComponentRoot);
        if Part = 'text_encoder_2' then
          ComponentRoot := ModelComponentDirectory(Settings.Components.TextEncoder2, ComponentRoot);
        if Part = 'vae' then
          ComponentRoot := ModelComponentDirectory(Settings.Components.VAE, ComponentRoot);
        RequireFile(TPath.Combine(ComponentRoot, 'config.json'));
      end;
      RequireWeights(ModelComponentDirectory(Settings.Components.TextEncoder, TPath.Combine(Settings.Directory, 'text_encoder')), 'model');
      RequireWeights(ModelComponentDirectory(Settings.Components.TextEncoder2, TPath.Combine(Settings.Directory, 'text_encoder_2')), 'model');
      RequireFile(ModelComponentFile(Settings.Components.Tokenizer, TPath.Combine(Settings.Directory, 'tokenizer/vocab.json')));
      RequireFile(TPath.Combine(ModelComponentDirectory(Settings.Components.Tokenizer, TPath.Combine(Settings.Directory, 'tokenizer')), 'merges.txt'));
      RequireFile(ModelComponentFile(Settings.Components.Tokenizer2, TPath.Combine(Settings.Directory, 'tokenizer_2/tokenizer.json')));
      RequireFile(ModelComponentFile(Settings.Components.VAE, TPath.Combine(Settings.Directory, 'vae/diffusion_pytorch_model.safetensors')));
    end
    else if Settings.Kind = ipQwenImage then
    begin
      var Suffix := 'bf16';
      if Settings.UseInt8 then
        Suffix := 'int8_convrot';
      RequireFile(ModelComponentFile(Settings.Components.TextEncoder, TPath.Combine(Settings.Directory, 'text_encoders/qwen3vl_8b_' + Suffix + '.safetensors')));
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
      RequireFile(ModelComponentFile(Settings.Components.VAE, TPath.Combine(Settings.Directory, 'vae/qwen_image_2.1_vae_bf16.safetensors')));
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
      begin
        var ComponentRoot := TPath.Combine(Settings.Directory, Part);
        if Part = 'text_encoder' then
          ComponentRoot := ModelComponentDirectory(Settings.Components.TextEncoder, ComponentRoot);
        if Part = 'text_encoder_2' then
          ComponentRoot := ModelComponentDirectory(Settings.Components.TextEncoder2, ComponentRoot);
        if Part = 'vae' then
          ComponentRoot := ModelComponentDirectory(Settings.Components.VAE, ComponentRoot);
        RequireFile(TPath.Combine(ComponentRoot, 'config.json'));
      end;
      RequireFile(ModelComponentFile(Settings.Components.Tokenizer, TPath.Combine(Settings.Directory, 'tokenizer/tokenizer.json')));
      RequireWeights(TPath.Combine(Settings.Directory, 'transformer'), 'diffusion_pytorch_model');
      RequireWeights(ModelComponentDirectory(Settings.Components.TextEncoder, TPath.Combine(Settings.Directory, 'text_encoder')), 'model');
      RequireFile(ModelComponentFile(Settings.Components.VAE, TPath.Combine(Settings.Directory, 'vae/diffusion_pytorch_model.safetensors')));
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
  Result.Components := Settings.Components;
  Result.SourceFile := Settings.SourceFile;
  Result.Adapters := Copy(Settings.Adapters);
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

function TReferenceAnalysisSettings.ToJSON: TJSONObject;
begin
  Result := TJSONObject.Create;
  Result.AddPair('model_id', ModelId);
  Result.AddPair('model_name', ModelName);
  Result.AddPair('directory', Directory);
  Result.AddPair('tokenizer_directory', TokenizerDirectory);
  Result.AddPair('instruction', Instruction);
  Result.AddPair('context', TJSONNumber.Create(Context));
  Result.AddPair('max_tokens', TJSONNumber.Create(MaxTokens));
  Result.AddPair('temperature', TJSONNumber.Create(Temperature));
end;

class function TReferenceAnalysisSettings.FromJSON(const Value: TJSONValue): TReferenceAnalysisSettings;
begin
  Result := Default(TReferenceAnalysisSettings);
  if not (Value is TJSONObject) then
    Exit;
  Result.ModelId := Value.GetValue<string>('model_id', '');
  Result.ModelName := Value.GetValue<string>('model_name', '');
  Result.Directory := Value.GetValue<string>('directory', '');
  Result.TokenizerDirectory := Value.GetValue<string>('tokenizer_directory', '');
  Result.Instruction := Value.GetValue<string>('instruction', '');
  Result.Context := EnsureRange(Value.GetValue<Integer>('context', 2048), 128, 65536);
  Result.MaxTokens := EnsureRange(Value.GetValue<Integer>('max_tokens', 512), 1, 32768);
  Result.Temperature := EnsureRange(Value.GetValue<Single>('temperature', 0), 0.0, 2.0);
end;

function TChatImage.NeedsReferenceAnalysis: Boolean;
begin
  Result := (Length(ReferencePaths) > 0) and not ImageSupportsReferences(Settings.Kind)
    and (ReferenceDescription.Trim = '');
end;

function TChatImage.GenerationPrompt: string;
begin
  Result := Prompt;
  if (Length(ReferencePaths) > 0) and not ImageSupportsReferences(Settings.Kind)
    and (ReferenceDescription.Trim <> '') then
    Result := 'Reference image descriptions:' + #10 + ReferenceDescription.Trim
      + #10#10 + 'User generation request:' + #10 + Prompt;
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
  var References := TJSONArray.Create;
  for var Path in ReferencePaths do
    References.Add(Path);
  Result.AddPair('reference_files', References);
  Result.AddPair('reference_analysis', ReferenceAnalysis.ToJSON);
  Result.AddPair('reference_description', ReferenceDescription);
end;

class function TChatImage.FromJSON(const Value: TJSONValue): TChatImage;
begin
  Result := Default(TChatImage);
  Result.IsUserImage := Value.GetValue<Boolean>('user_image', False);
  Result.OriginalName := Value.GetValue<string>('original_name', '');
  Result.ContentHash := Value.GetValue<string>('sha256', '');
  Result.RelativePath := Value.GetValue<string>('file', '');
  Result.Prompt := Value.GetValue<string>('prompt', '');
  Result.ModelName := Value.GetValue<string>('model', '');
  Result.Description := Value.GetValue<string>('description', '');
  Result.Settings := TChatImageSettings.FromJSON(Value.FindValue('settings'));
  Result.ReferenceAnalysis := TReferenceAnalysisSettings.FromJSON(Value.FindValue('reference_analysis'));
  Result.ReferenceDescription := Value.GetValue<string>('reference_description', '');
  var References := Value.FindValue('reference_files');
  if References is TJSONArray then
  begin
    if TJSONArray(References).Count > 4 then
      raise EArgumentException.Create('Too many reference images');
    for var Path in TJSONArray(References) do
      Result.ReferencePaths := Result.ReferencePaths + [Path.Value];
  end;
  Result.ReferenceFiles;
  Result.FileName; // Reject paths outside the application image archive.
end;

function TChatImage.ReferenceFiles: TArray<string>;
begin
  if Length(ReferencePaths) > 4 then
    raise EArgumentException.Create('Допускается до четырёх референсов.');
  SetLength(Result, Length(ReferencePaths));
  for var I := 0 to High(ReferencePaths) do
  begin
    var Reference := Default(TChatImage);
    Reference.RelativePath := ReferencePaths[I];
    Result[I] := Reference.FileName;
  end;
end;

function TChatImage.AsReference: TChatImage;
begin
  Result := Default(TChatImage);
  var Source := FileName; // Validates that the file belongs to the chat archive.
  var Stream := TFileStream.Create(Source, fmOpenRead or fmShareDenyWrite);
  try
    Result.ContentHash := THashSHA2.GetHashString(Stream)
  finally
    Stream.Free
  end;
  Result.RelativePath := RelativePath;
  Result.IsUserImage := True;
  Result.OriginalName := OriginalName;
  if (Result.OriginalName = '') and (ModelName <> '') then
    Result.OriginalName := ModelName + '.png';
  if Result.OriginalName = '' then
    Result.OriginalName := ExtractFileName(Source);
  Result.ModelName := 'Из истории чата';
  Result.Description := Description;
  Result.Settings := TChatImageSettings.Defaults;
  Result.Settings.Width := Settings.Width;
  Result.Settings.Height := Settings.Height;
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
  Flux1: TFlux1Pipeline;
  Params: TGenerateParams;
  QParams: TQwen21Params;
  Output: TQNNImage;
  References: TArray<TQNNImage>;
  ReferenceFiles: TArray<string>;
  PreviousLimit: UInt64;
  Destination, Temporary, Assets: string;
begin
  ValidateImageSettings(Image.Settings);
  ReferenceFiles := Image.ReferenceFiles;
  if Image.NeedsReferenceAnalysis then
    raise ENotSupportedException.Create('Сначала выполните анализ референсов выбранной vision-моделью.');
  for var FileName in ReferenceFiles do
    if not TFile.Exists(FileName) then
      raise EFileNotFoundException.Create('Референс не найден: ' + FileName);
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
    var Prompt := Image.GenerationPrompt;
    Params := Default(TGenerateParams);
    Params.Width := Image.Settings.Width;
    Params.Height := Image.Settings.Height;
    Params.NumSteps := Image.Settings.Steps;
    Params.Seed := Image.Settings.Seed;
    Params.Guidance := Image.Settings.Guidance;
    Params.PowerAlpha := 2;
    if Image.Settings.Kind = ipFlux1 then
    begin
      var FParams := TFlux1Params.Defaults;
      FParams.Width := Params.Width;
      FParams.Height := Params.Height;
      FParams.Steps := Params.NumSteps;
      FParams.Seed := Params.Seed;
      FParams.Guidance := Params.Guidance;
      FParams.Prompt := Prompt;
      if Image.Settings.LoRAFile <> '' then
        FParams.LoRAs := [TQNNLoRASpec.Create(Image.Settings.LoRAFile, Image.Settings.LoRAStrength)];
      Flux1 := TFlux1Pipeline.Create(Image.Settings.Directory,
        procedure(const Text: string)
        begin
          if Assigned(Cancelled) and Cancelled() then
            raise EAbort.Create('Генерация остановлена');
          if Assigned(OnPhase) then
            OnPhase(Text);
        end, Image.Settings.DiffusionCheckpoint);
      Flux1.Components := Image.Settings.Components;
      try
        Output := Flux1.Generate(FParams)
      finally
        Flux1.Free
      end;
    end
    else if Image.Settings.Kind = ipFlux then
    begin
      Flux := TQNNFlux.Load(Image.Settings.Directory, Phase);
      Flux.Components := Image.Settings.Components;
      try
        Flux.ResidentGPU := True;
        if Image.Settings.LoRAFile <> '' then
          Flux.LoRAs := [TQNNLoRASpec.Create(Image.Settings.LoRAFile, Image.Settings.LoRAStrength)];
        SetLength(References, Length(ReferenceFiles));
        for var I := 0 to High(References) do
          References[I] := Q21LoadReference(ReferenceFiles[I], 512);
        if Length(References) = 0 then
          Output := Flux.Generate(UTF8String(Prompt), UTF8String(Image.Settings.NegativePrompt), Params)
        else
          Output := Flux.Generate(UTF8String(Prompt), UTF8String(Image.Settings.NegativePrompt), Params, References)
      finally
        Flux.Free
      end;
    end
    else if Image.Settings.Kind = ipZImage then
    begin
      ZImage := TQNNZImage.Load(Image.Settings.Directory, Phase);
      ZImage.Components := Image.Settings.Components;
      try
        ZImage.ResidentGPU := True;
        Output := ZImage.Generate(UTF8String(Prompt), Params)
      finally
        ZImage.Free
      end;
    end
    else
    begin
      Assets := Image.Settings.AssetsDirectory;
      if Assets = '' then
        Assets := FindQwenImageAssets(Image.Settings.Directory);
      QParams := TQwen21Params.Defaults;
      QParams.Prompt := Prompt;
      QParams.NegativePrompt := Image.Settings.NegativePrompt;
      QParams.Width := Params.Width;
      QParams.Height := Params.Height;
      QParams.Steps := Params.NumSteps;
      QParams.Seed := Params.Seed;
      QParams.Guidance := Params.Guidance;
      QParams.ResidentGPU := True;
      QParams.ReferenceImages := ReferenceFiles;
      if Image.Settings.LoRAFile <> '' then
        QParams.LoRAs := [TQNNLoRASpec.Create(Image.Settings.LoRAFile, Image.Settings.LoRAStrength)];
      Qwen := TQwen21Pipeline.Create(Image.Settings.Directory, Assets, Image.Settings.UseInt8,
        procedure(const Text: string)
        begin
          if Assigned(Cancelled) and Cancelled() then
            raise EAbort.Create('Генерация остановлена');
          if Assigned(OnPhase) then
            OnPhase(Text);
        end, Image.Settings.DiffusionCheckpoint);
      Qwen.Components := Image.Settings.Components;
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
    for var Reference in References do
      Reference.Free;
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

