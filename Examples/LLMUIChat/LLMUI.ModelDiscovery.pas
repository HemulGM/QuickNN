unit LLMUI.ModelDiscovery;

interface

uses
  System.SysUtils, System.JSON, QuickNN.ModelFiles, LLMUI.Images;

type
  TModelCapability = (mcText, mcVision, mcImages, mcThinking, mcLoRA, mcReferenceImages, mcNegativePrompt);
  TModelCapabilities = set of TModelCapability;
  TDetectedModelKind = (dkUnknown, dkLanguage, dkImage, dkAdapter, dkComponent);

  TDetectedModel = record
    Kind: TDetectedModelKind;
    SourceFile, Directory, SuggestedName, Architecture, TokenizerDirectory: string;
    Capabilities: TModelCapabilities;
    Image: TChatImageSettings;
    Notice: string;
    function CapabilityText: string;
    function Validate: string;
  end;

function DetectLocalModel(const Source: string; const BundleDirectory: string = ''): TDetectedModel;
function FindModelAdapters(const Directory: string): TArray<string>;
function AdapterValidation(const FileName: string; const Kind: TImagePipelineKind): string;

implementation

uses
  System.Classes, System.IOUtils, System.Generics.Collections, System.Math,
  QuickNN.GGUF, QuickNN.Models.LLM.GGUF, LLMUI.Data;

function ReadJSON(const FileName: string): TJSONValue;
begin
  Result := nil;
  if TFile.Exists(FileName) then
  begin
    if TFile.GetSize(FileName) > 16 * 1024 * 1024 then
      raise EArgumentException.Create('Слишком большой файл конфигурации: ' + FileName);
    Result := TJSONValue.ParseJSONValue(TFile.ReadAllText(FileName, TEncoding.UTF8));
    if not (Result is TJSONObject) then
    begin Result.Free; raise EArgumentException.Create('Повреждён JSON: ' + FileName) end;
  end;
end;


function FirstFile(const Directory: string; const Names: array of string): string;
begin
  Result := '';
  for var Name in Names do
  begin
    var Candidate := TPath.Combine(Directory, Name);
    if TFile.Exists(Candidate) then Exit(Candidate);
  end;
end;

function BundleRoot(const Source: string): string;
begin
  Result := Source;
  if TFile.Exists(Result) then Result := ExtractFileDir(Result);
  for var I := 0 to 3 do
  begin
    if TFile.Exists(TPath.Combine(Result, 'model_index.json'))
      or TFile.Exists(TPath.Combine(Result, 'encoder_config.json')) then Exit;
    var Config := ReadJSON(TPath.Combine(Result, 'config.json'));
    try
      if (Config <> nil) and (Config.FindValue('model_type') <> nil) then Exit;
    finally Config.Free end;
    var Parent := ExtractFileDir(Result);
    if Parent = Result then Break;
    Result := Parent;
  end;
  Result := Source;
  if TFile.Exists(Result) then Result := ExtractFileDir(Result);
end;

function PipelineAt(const Directory: string): string;
begin
  Result := '';
  var JSON := ReadJSON(TPath.Combine(Directory, 'model_index.json'));
  try
    if JSON <> nil then Result := JSON.GetValue<string>('_class_name', '');
  finally JSON.Free end;
end;

function NearbyBundle(const SourceDirectory, Pipeline: string): string;
begin
  Result := '';
  var Parent := ExtractFileDir(ExcludeTrailingPathDelimiter(SourceDirectory));
  var Count := 0;
  for var Candidate in TDirectory.GetDirectories(Parent) do
  begin
    Inc(Count); if Count > 128 then Break;
    try
      if not SameText(PipelineAt(Candidate), Pipeline) then Continue;
    except on E: Exception do Continue end;
    if Result <> '' then Exit(''); // Ambiguous: let the user choose explicitly.
    Result := Candidate;
  end;
end;

function IsBundleWeight(const FileName, Directory, SingleName, IndexName: string): Boolean;
begin
  Result := SameText(TPath.GetFullPath(FileName), TPath.GetFullPath(TPath.Combine(Directory, SingleName)));
  if Result then Exit;
  var JSON := ReadJSON(TPath.Combine(Directory, IndexName));
  try
    if (JSON = nil) or not (JSON.FindValue('weight_map') is TJSONObject) then Exit;
    for var Pair in TJSONObject(JSON.FindValue('weight_map')) do
      if SameText(TPath.GetFullPath(FileName), TPath.GetFullPath(TPath.Combine(Directory, Pair.JsonValue.Value))) then Exit(True);
  finally JSON.Free end;
end;

function TDetectedModel.CapabilityText: string;
begin
  Result := '';
  if mcText in Capabilities then Result := 'Переписка';
  if mcVision in Capabilities then Result := Result + ' · Анализ изображений';
  if mcImages in Capabilities then Result := 'Генерация изображений';
  if mcThinking in Capabilities then Result := Result + ' · Размышления';
  if mcReferenceImages in Capabilities then Result := Result + ' · Референсные изображения';
  if mcLoRA in Capabilities then Result := Result + ' · LoRA';
  if mcNegativePrompt in Capabilities then Result := Result + ' · Негативный промпт';
  if Result = '' then Result := 'Нет поддерживаемых ролей';
end;

function TDetectedModel.Validate: string;
begin
  if Kind = dkLanguage then
  begin
    var Entry := TModelEntry.Create;
    try
      Entry.Directory := Directory; Entry.TokenizerDirectory := TokenizerDirectory;
      Entry.Inspect;
      if Entry.Available then Exit('');
      Exit(Entry.Validation);
    finally Entry.Free end;
  end;
  if Kind = dkImage then
  begin
    var Entry := TImageModelEntry.Create;
    try
      Entry.Settings := Image; Entry.Inspect;
      if Entry.Available then Exit('');
      Exit(Entry.Validation);
    finally Entry.Free end;
  end;
  Result := Notice;
  if Result = '' then Result := 'Архитектура не поддерживается QuickNN.';
end;

function AdapterValidation(const FileName: string; const Kind: TImagePipelineKind): string;
begin
  Result := '';
  if not (Kind in [ipFlux, ipFlux1, ipQwenImage]) then Exit('LoRA для этой архитектуры пока не поддерживается.');
  var JSON := ReadModelTensorHeader(FileName);
  try
    var Count := 0;
    for var Pair in JSON do
      if Pair.JsonString.Value.EndsWith('.lora_down.weight') or Pair.JsonString.Value.EndsWith('.lora_A.weight') then
      begin
        var Target := Pair.JsonString.Value;
        var Up := Target.Replace('.lora_down.weight', '.lora_up.weight').Replace('.lora_A.weight', '.lora_B.weight');
        if JSON.GetValue(Up) = nil then Exit('У адаптера нет парной матрицы: ' + Up);
        var DownShape := Pair.JsonValue.FindValue('shape');
        var UpShape := JSON.GetValue(Up).FindValue('shape');
        if not (DownShape is TJSONArray) or not (UpShape is TJSONArray)
          or (TJSONArray(DownShape).Count <> 2) or (TJSONArray(UpShape).Count <> 2)
          or (TJSONArray(DownShape).Items[0].GetValue<Integer> <> TJSONArray(UpShape).Items[1].GetValue<Integer>) then
          Exit('Неверные размеры парных матриц LoRA.');
        if (Kind = ipFlux) and not (Target.StartsWith('transformer.') or Target.StartsWith('transformer_blocks.')
          or Target.StartsWith('single_transformer_blocks.') or Target.StartsWith('x_embedder.')
          or Target.StartsWith('context_embedder.') or Target.StartsWith('double_stream_modulation_')
          or Target.StartsWith('single_stream_modulation.') or Target.StartsWith('time_guidance_embed.')
          or Target.StartsWith('norm_out.') or Target.StartsWith('proj_out.')) then
          Exit('Для FLUX.2 Klein выберите адаптер в формате Diffusers с совместимыми проекциями.');
        if (Kind = ipFlux1) and (Target.Contains('transformer_blocks') or Target.Contains('lora_te')) then
          Exit('Этот формат LoRA не поддерживается FLUX.1-dev.');
        if (Kind = ipQwenImage) and (Target.Contains('double_blocks') or Target.Contains('single_blocks')) then
          Exit('Адаптер FLUX не подходит для Qwen-Image.');
        Inc(Count);
      end;
    if Count = 0 then Result := 'Файл не содержит поддерживаемых матриц LoRA.';
  finally JSON.Free end;
end;

function FindModelAdapters(const Directory: string): TArray<string>;
begin
  var Files := TList<string>.Create;
  try
    for var Subdir in TArray<string>.Create('loras', 'adapters') do
    begin
      var Location := TPath.Combine(Directory, Subdir);
      if not TDirectory.Exists(Location) then Continue;
      for var FileName in TDirectory.GetFiles(Location, '*.safetensors', TSearchOption.soAllDirectories) do
      begin
        if Files.Count >= 128 then Break;
        Files.Add(FileName);
      end;
    end;
    Result := Files.ToArray;
  finally Files.Free end;
end;

function DetectLocalModel(const Source: string; const BundleDirectory: string): TDetectedModel;
var
  Header, Metadata: TJSONObject;
  Config: TJSONValue;
  Pipeline, Architecture, FileName, Root: string;
begin
  Result := Default(TDetectedModel); Result.Image := TChatImageSettings.Defaults;
  Header := nil; Metadata := nil; Config := nil;
  try
  try
    var Location := Source.Trim.Trim(['"']);
    if not TDirectory.Exists(Location) and not TFile.Exists(Location) then
      raise EArgumentException.Create('Выберите существующий файл модели или её каталог.');
    Location := TPath.GetFullPath(Location); Root := BundleRoot(Location);
    if TFile.Exists(Location) then FileName := Location
    else
    begin
      FileName := FirstFile(Root, ['flux1-dev.safetensors', 'model.safetensors.index.json', 'model.safetensors', 'model_index.json', 'config.json']);
      if FileName = '' then
      begin
        var Files := TDirectory.GetFiles(Location, '*.gguf');
        if Length(Files) = 1 then FileName := Files[0]
        else if Length(Files) > 1 then FileName := LLMGGUFCheckpoint(Location);
      end;
    end;
    Result.SourceFile := FileName;
    if FileName = '' then raise EArgumentException.Create('Основной файл модели не найден.');
    if not SameText(ExtractFileExt(FileName), '.gguf') and not SameText(ExtractFileExt(FileName), '.safetensors')
      and not SameText(ExtractFileExt(FileName), '.json') then
      raise EArgumentException.Create('Выберите GGUF, safetensors или JSON-индекс модели.');
    if SameText(ExtractFileExt(FileName), '.gguf') then
    begin
      FileName := LLMGGUFCheckpoint(FileName); Result.SourceFile := FileName;
      Metadata := TJSONObject.Create;
      var GGUF := TGGUFHeader.ReadFile(FileName, Metadata, False);
      Architecture := GGUF.Architecture;
      var NameValue := Metadata.GetValue('general.name');
      if NameValue is TJSONString then Result.SuggestedName := NameValue.Value;
    end
    else if SameText(ExtractFileExt(FileName), '.safetensors') then
    begin
      Header := ReadModelTensorHeader(FileName);
      var Meta := Header.FindValue('__metadata__');
      if Meta is TJSONObject then
      begin
        var Title := TJSONObject(Meta).GetValue('modelspec.title');
        if Title is TJSONString then Result.SuggestedName := Title.Value;
      end;
      for var Pair in Header do
        if Pair.JsonString.Value.EndsWith('.lora_down.weight') or Pair.JsonString.Value.EndsWith('.lora_A.weight') then
        begin
          Result.Kind := dkAdapter;
          Result.Notice := 'Это LoRA-адаптер. Выберите базовую модель, затем добавьте этот файл на шаге «Файлы и адаптеры».';
          Exit;
        end;
      if Header.GetValue('double_blocks.0.img_attn.qkv.weight') <> nil then
      begin
        if Header.GetValue('guidance_in.in_layer.weight') <> nil then Architecture := 'flux1-dev'
        else Architecture := 'flux1-schnell';
      end;
      var ParentName := ExtractFileName(ExtractFileDir(FileName)).ToLowerInvariant;
      if (ParentName = 'vae') or (ParentName = 'text_encoder') or (ParentName = 'text_encoder_2')
        or (ParentName = 'text_encoders') or (Header.GetValue('decoder.conv_in.weight') <> nil) then
      begin
        Result.Kind := dkComponent;
        Result.Notice := 'Выбран вспомогательный компонент. Основной файл — веса трансформера, GGUF или индекс языковой модели.';
        Exit;
      end;
    end;
    if BundleDirectory <> '' then Root := TPath.GetFullPath(BundleDirectory);
    Pipeline := PipelineAt(Root);
    if Architecture = 'qwen_image21' then
    begin
      if not SameText(Pipeline, 'QwenImage21Pipeline') and (BundleDirectory = '') then
      begin
        var Nearby := NearbyBundle(Root, 'QwenImage21Pipeline');
        if Nearby <> '' then Root := Nearby;
      end;
      Pipeline := 'QwenImage21Pipeline';
    end;
    Config := ReadJSON(TPath.Combine(Root, 'config.json'));
    if (Architecture = '') and (Config <> nil) then Architecture := Config.GetValue<string>('model_type', '');
    if (Pipeline = 'FluxPipeline') and (Architecture = '') then Architecture := 'flux1-dev';
    Result.Directory := Root;
    Result.Image.Directory := Root;
    if (Architecture = 'flux1-dev') or SameText(Pipeline, 'QwenImage21Pipeline')
      or Pipeline.StartsWith('Flux2Klein', True) or Pipeline.StartsWith('ZImage', True) then
    begin
      Result.Kind := dkImage; Result.Capabilities := [mcImages];
      if SameText(Pipeline, 'QwenImage21Pipeline') then
      begin
        Result.Architecture := 'Qwen-Image 2.1'; Result.Image.Kind := ipQwenImage;
        Result.Image.Steps := 20; Result.Image.Guidance := 4;
        Result.Capabilities := Result.Capabilities + [mcLoRA];
        if Header <> nil then
          Result.Image.UseInt8 := Header.ToJSON.Contains('weight_scale');
        if SameText(ExtractFileExt(FileName), '.gguf') or SameText(ExtractFileExt(FileName), '.safetensors') then
          Result.Image.DiffusionCheckpoint := FileName
        else
        begin
          var Suffix := 'bf16';
          if TFile.Exists(TPath.Combine(Root, 'diffusion_models/qwen_image_2.1_int8_convrot.safetensors')) then
          begin Suffix := 'int8_convrot'; Result.Image.UseInt8 := True end;
          Result.Image.DiffusionCheckpoint := FirstFile(Root, ['diffusion_models/qwen_image_2.1_' + Suffix + '.safetensors']);
        end;
        Result.Image.AssetsDirectory := Root;
        if not TFile.Exists(TPath.Combine(Root, 'encoder_config.json')) then Result.Image.AssetsDirectory := FindQwenImageAssets(Root);
        var Suffix := 'bf16'; if Result.Image.UseInt8 then Suffix := 'int8_convrot';
        Result.Image.Components.TextEncoder := TPath.Combine(Root, 'text_encoders/qwen3vl_8b_' + Suffix + '.safetensors');
        Result.Image.Components.VAE := TPath.Combine(Root, 'vae/qwen_image_2.1_vae_bf16.safetensors');
      end
      else if Architecture = 'flux1-dev' then
      begin
        Result.Architecture := 'FLUX.1-dev'; Result.Image.Kind := ipFlux1;
        Result.Image.Steps := 20; Result.Image.Guidance := 3.5;
        Result.Capabilities := Result.Capabilities + [mcLoRA];
        Result.Image.Width := 512; Result.Image.Height := 512;
        if SameText(ExtractFileExt(FileName), '.safetensors') then Result.Image.DiffusionCheckpoint := FileName;
        Result.Image.Components.TextEncoder := TPath.Combine(Root, 'text_encoder');
        Result.Image.Components.TextEncoder2 := TPath.Combine(Root, 'text_encoder_2');
        Result.Image.Components.Tokenizer := TPath.Combine(Root, 'tokenizer');
        Result.Image.Components.Tokenizer2 := TPath.Combine(Root, 'tokenizer_2');
        Result.Image.Components.VAE := TPath.Combine(Root, 'vae');
      end
      else
      begin
        if SameText(ExtractFileExt(FileName), '.safetensors')
          and not IsBundleWeight(FileName, TPath.Combine(Root, 'transformer'),
            'diffusion_pytorch_model.safetensors', 'diffusion_pytorch_model.safetensors.index.json') then
          raise EArgumentException.Create('Выбранный файл не входит в индекс трансформера этого комплекта.');
        Result.Image.Components.TextEncoder := TPath.Combine(Root, 'text_encoder');
        Result.Image.Components.Tokenizer := TPath.Combine(Root, 'tokenizer');
        Result.Image.Components.VAE := TPath.Combine(Root, 'vae');
        if Pipeline.StartsWith('Flux2', True) then
        begin
            Result.Architecture := 'FLUX.2 Klein'; Result.Image.Kind := ipFlux;
          var Index := ReadJSON(TPath.Combine(Root, 'model_index.json'));
          try
            if (Index <> nil) and not Index.GetValue<Boolean>('is_distilled', False) then
            begin Result.Image.Steps := 50; Result.Image.Guidance := 4 end;
          finally Index.Free end;

        end
        else
        begin Result.Architecture := 'Z-Image Turbo'; Result.Image.Kind := ipZImage; Result.Image.Steps := 9; Result.Image.Guidance := 0 end;
      end;
      if ImageSupportsReferences(Result.Image.Kind) then Include(Result.Capabilities, mcReferenceImages);
      if ImageSupportsLoRA(Result.Image) then Include(Result.Capabilities, mcLoRA);
      if ImageSupportsNegativePrompt(Result.Image) then Include(Result.Capabilities, mcNegativePrompt);
      Result.Image.Adapters := FindModelAdapters(Root);
    end
    else if (Architecture = 'qwen3') or (Architecture = 'qwen3_moe') or (Architecture = 'qwen3_5') or (Architecture = 'qwen35') or (Architecture = 'gemma2') or (Architecture = 'qwen4exp') then
    begin
      if SameText(ExtractFileExt(FileName), '.safetensors')
        and not IsBundleWeight(FileName, Root, 'model.safetensors', 'model.safetensors.index.json') then
        raise EArgumentException.Create('Выбранный файл не входит в индекс этой языковой модели.');
      Result.Kind := dkLanguage; Result.Architecture := 'Qwen3';
      if Architecture = 'qwen3_moe' then Result.Architecture := 'Qwen3 MoE';
      if (Architecture = 'qwen3_5') or (Architecture = 'qwen35') then Result.Architecture := 'Qwen hybrid · Gated DeltaNet';
      if Architecture = 'qwen4exp' then Result.Architecture := 'Qwen3.8 Flash Next · qwen4exp';
      if Architecture = 'gemma2' then Result.Architecture := 'Gemma 2';
      Result.Capabilities := [mcText];
      if Architecture <> 'qwen3_moe' then Include(Result.Capabilities, mcThinking);
      if SameText(ExtractFileExt(FileName), '.gguf') then Result.Directory := FileName
      else
      begin
        Result.TokenizerDirectory := Root;
        if not TFile.Exists(TPath.Combine(Root, 'tokenizer.json')) then
          Result.TokenizerDirectory := TPath.Combine(ExtractFileDir(Root), 'tokenizer');
      end;
      var Entry := TModelEntry.Create;
      try
        Entry.Directory := Result.Directory; Entry.TokenizerDirectory := Result.TokenizerDirectory; Entry.Inspect;
        if Entry.SupportsVision then Include(Result.Capabilities, mcVision);
        if Entry.SupportsThinking then Include(Result.Capabilities, mcThinking) else Exclude(Result.Capabilities, mcThinking);
        if not Entry.Available then Result.Notice := Entry.Validation;
      finally Entry.Free end;
    end
    else
    begin
      Result.Architecture := Architecture;
      Result.Notice := 'Архитектура не поддерживается QuickNN: ' + Architecture;
    end;
    if Result.SuggestedName = '' then
    begin
      var Name := ChangeFileExt(ExtractFileName(FileName), '');
      if (ExtractFileExt(FileName) = '.json') or Name.StartsWith('model', True)
        or Name.StartsWith('diffusion_pytorch_model', True) then Name := ExtractFileName(Root);
      Result.SuggestedName := Name;
    end;
    if Result.SuggestedName = '' then Result.SuggestedName := Result.Architecture;
  except
    on E: Exception do
    begin Result.Kind := dkUnknown; Result.Capabilities := []; Result.Notice := E.Message end;
  end;
  finally Config.Free; Header.Free; Metadata.Free end;
end;

end.
