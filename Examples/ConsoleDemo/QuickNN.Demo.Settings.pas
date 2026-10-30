unit QuickNN.Demo.Settings;

interface

uses
  QuickNN.LoRA;

type
  TDemoModel = (dmQwenImage21Int8, dmQwenImage21BF16, dmFlux2Klein4B, dmFlux2Klein9B, dmZImageTurbo);

  TDemoSettings = record
    Model: TDemoModel;
    ModelsRoot, ModelDirectory, AssetsDirectory, OutputFile: string;
    Prompt, NegativePrompt: string;
    Width, Height, Steps: Integer;
    Seed: Int64;
    Guidance: Single;
    ReferenceImages: TArray<string>;
    LoRAs: TArray<TQNNLoRASpec>;
    ReferenceSize: Integer;
    ResidentGPU: Boolean;
  end;

function DemoSettings: TDemoSettings;

function ModelName(const Model: TDemoModel): string;

function DefaultSteps(const Model: TDemoModel): Integer;

function DefaultModelDirectory(const Settings: TDemoSettings): string;

implementation

uses
  System.SysUtils,
  System.IOUtils;

function DemoSettings: TDemoSettings;
begin
  Result := Default(TDemoSettings);

  // Выбор модели: измените только эту строку и запустите проект (F9).
  Result.Model := dmQwenImage21Int8;

  Result.ResidentGPU := True; // False = CPU/host-mediated comparison path.

  Result.ModelsRoot := 'H:\Models';
  // Пусто = каталог выбранного пресета внутри ModelsRoot.
  Result.ModelDirectory := '';
  // Пусто = автоматически найти assets/qwen-image-2.1 относительно exe.
  Result.AssetsDirectory := '';
  // Пусто = output/<модель>-<дата>-<seed>.png в корне QuickNN.
  Result.OutputFile := '';

  Result.Prompt := 'A red apple on a wooden table, soft daylight, realistic photograph.';
  Result.NegativePrompt := '';
  Result.Width := 256;
  Result.Height := 256;
  Result.Steps := 0; // 0 = пресет: Qwen 20, FLUX Klein 4, Z-Image Turbo 9.
  Result.Seed := 42;
  Result.Guidance := 1.0;

  // Необязательно, только для Qwen-Image 2.1:
  Result.ReferenceImages := [];
  Result.LoRAs := [];
  // Пример: [TQNNLoRASpec.Create('adapter.safetensors', 0.5)].
  Result.ReferenceSize := 512;
end;

function ModelName(const Model: TDemoModel): string;
const
  NAMES: array[TDemoModel] of string =
      ('Qwen-Image-2.1-INT8', 'Qwen-Image-2.1-BF16', 'FLUX.2-klein-4B', 'FLUX.2-klein-9B', 'Z-Image-Turbo');
begin
  Result := NAMES[Model];
end;

function DefaultSteps(const Model: TDemoModel): Integer;
begin
  case Model of
    dmQwenImage21Int8, dmQwenImage21BF16: Result := 20;
    dmFlux2Klein4B, dmFlux2Klein9B: Result := 4;
    dmZImageTurbo: Result := 9;
  else
    raise EArgumentOutOfRangeException.Create('Unknown demo model');
  end;
end;

function DefaultModelDirectory(const Settings: TDemoSettings): string;
begin
  if Settings.ModelDirectory <> '' then
    Exit(ExpandFileName(Settings.ModelDirectory));
  case Settings.Model of
    dmQwenImage21Int8, dmQwenImage21BF16: Result := Settings.ModelsRoot;
    dmFlux2Klein4B, dmFlux2Klein9B, dmZImageTurbo:
      Result := TPath.Combine(Settings.ModelsRoot, ModelName(Settings.Model));
  else
    raise EArgumentOutOfRangeException.Create('Unknown demo model');
  end;
  Result := ExpandFileName(Result);
end;

end.
