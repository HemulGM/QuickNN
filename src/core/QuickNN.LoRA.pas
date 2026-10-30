unit QuickNN.LoRA;

{ Native additive LoRA execution. Base weights stay immutable, including INT8
  ConvRot storage. Formula: y += strength * alpha/rank * (x A^T) B^T.
  Target/input/output dimensions are validated before any adapters are attached. }
{$POINTERMATH ON}

interface

uses
  System.SysUtils, System.Generics.Collections, QuickNN.SafeTensors,
  QuickNN.Common, QuickNN.Quantization, QuickNN.Vulkan;

type
  TQNNLoRASpec = record
    FileName: string;
    Strength: Single;
    class function Create(const FileName: string; const Strength: Single = 1): TQNNLoRASpec; static;
  end;

  TQNNLoRAResolver = reference to function(const Target: string; out OutputOffset: Integer): PQNNLinearWeight;

  IQNNLoRAModule = interface(IQNNLinearAdapter)
    function Target: string;
    function Description: string;
    function Enabled: Boolean;
    procedure Validate(const Weight: TQNNLinearWeight; const OutputOffset: Integer);
    procedure Bind(var Weight: TQNNLinearWeight; const OutputOffset: Integer);
  end;

  TQNNLoRASet = class
  private
    FFiles: TDictionary<string, TSafeTensorFile>;
    FModules: TList<IQNNLoRAModule>;
    FSpecs: TArray<TQNNLoRASpec>;
  public
    constructor Create(const Specs: TArray<TQNNLoRASpec>);
    destructor Destroy; override;
    procedure Attach(const Resolver: TQNNLoRAResolver);
    function ModuleCount: Integer;
    function EnabledModuleCount: Integer;
  end;

implementation

uses
  System.Math, System.IOUtils, QuickNN.Kernels;

type
  TQNNLoRAModule = class(TInterfacedObject, IQNNLinearAdapter, IQNNLoRAModule, IQNNGPULinearAdapter)
  private
    FFile: TSafeTensorFile;
    FDown, FUp: TMemoryBlock;
    FTarget, FDescription: string;
    FScale: Single;
    FInput, FOutput, FRank, FOffset: Integer;
    FBound: Boolean;
  public
    procedure ApplyGPU(const Context: TQNNGPUContext; const Dst, Src: TQNNGPUTensor; const Rows, InputDim, OutputDim: Integer);
    constructor Create(const FileData: TSafeTensorFile; const Target, DownName, UpName, AlphaName: string; const Strength: Single);
    destructor Destroy; override;
    procedure Apply(const Dst, Src: TMemoryBlock; const Rows, InputDim, OutputDim: Integer);
    function Target: string;
    function Description: string;
    function Enabled: Boolean;
    procedure Validate(const Weight: TQNNLinearWeight; const OutputOffset: Integer);
    procedure Bind(var Weight: TQNNLinearWeight; const OutputOffset: Integer);
  end;

class function TQNNLoRASpec.Create(const FileName: string; const Strength: Single): TQNNLoRASpec;
begin
  Result.FileName := FileName;
  Result.Strength := Strength
end;

constructor TQNNLoRAModule.Create(const FileData: TSafeTensorFile; const Target, DownName, UpName, AlphaName: string; const Strength: Single);
var
  DownTensor, UpTensor, AlphaTensor: PSafeTensor;
  AlphaMemory: TMemoryBlock;
  Scale, Alpha: Double;
begin
  inherited Create;
  FFile := FileData;
  FTarget := Target;
  FDescription := ExtractFileName(FFile.Path) + ': ' + Target;
  DownTensor := FFile.Find(DownName);
  UpTensor := FFile.Find(UpName);
  if (DownTensor = nil) or (UpTensor = nil) then
    raise ESafeTensorError.Create('Incomplete LoRA A/B pair: ' + FDescription);
  if (DownTensor.Ndim <> 2)
    or (UpTensor.Ndim <> 2)
    or not (DownTensor.DType in [StBF16, StF16, StF32])
    or not (UpTensor.DType in [StBF16, StF16, StF32]) then
    raise ESafeTensorError.Create('LoRA requires rank-two BF16/F16/F32 matrices: ' + FDescription);
  if (DownTensor.Shape[0] <= 0)
    or (DownTensor.Shape[0] > 4096)
    or (DownTensor.Shape[1] <= 0)
    or (DownTensor.Shape[1] > MaxInt)
    or (UpTensor.Shape[0] <= 0)
    or (UpTensor.Shape[0] > MaxInt)
    or (UpTensor.Shape[1] <> DownTensor.Shape[0]) then
    raise ESafeTensorError.Create('Invalid LoRA rank or matrix shape: ' + FDescription);
  FRank := DownTensor.Shape[0];
  FInput := DownTensor.Shape[1];
  FOutput := UpTensor.Shape[0];
  Alpha := FRank;
  AlphaTensor := FFile.Find(AlphaName);
  if AlphaTensor <> nil then
  begin
    if AlphaTensor.Count <> 1 then
      raise ESafeTensorError.Create('LoRA alpha must be scalar: ' + FDescription);
    try
      AlphaMemory := AlphaTensor.AsSingle(False);
      Alpha := System.PSingle(AlphaMemory)^
    finally
      AlphaMemory.Free
    end;
  end;
  Scale := Double(Strength) * Alpha / FRank;
  if IsNan(Alpha) or IsInfinite(Alpha) or IsNan(Scale) or IsInfinite(Scale) or (Abs(Scale) > MaxSingle) then
    raise ESafeTensorError.Create('Non-finite LoRA scale/alpha: ' + FDescription);
  FScale := Scale;
  if DownTensor.DType = StBF16 then
    FDown := DownTensor.AsBF16(True)
  else
    FDown := DownTensor.AsSingle(True);
  if UpTensor.DType = StBF16 then
    FUp := UpTensor.AsBF16(True)
  else
    FUp := UpTensor.AsSingle(True);
end;

destructor TQNNLoRAModule.Destroy;
begin
  FDown.Free;
  FUp.Free;
  FFile.Close;
  inherited
end;

function TQNNLoRAModule.Target: string;
begin
  Result := FTarget
end;

function TQNNLoRAModule.Description: string;
begin
  Result := FDescription
end;

function TQNNLoRAModule.Enabled: Boolean;
begin
  Result := FScale <> 0
end;

procedure TQNNLoRAModule.Validate(const Weight: TQNNLinearWeight; const OutputOffset: Integer);
begin
  if (Weight.InputDim <> FInput) or (OutputOffset < 0) or (Int64(OutputOffset) + FOutput > Weight.OutputDim) then
    raise ESafeTensorError.Create('LoRA/base dimensions do not match: ' + FDescription);
  // A full target must cover all rows. Partial rows are only explicitly mapped
  // aliases (gate_layer/proj into the fused gate_up projection).
  if (OutputOffset = 0) and (FOutput <> Weight.OutputDim) and not FTarget.EndsWith('.img_mlp.gate_layer') then
    raise ESafeTensorError.Create('LoRA output rows do not cover the target: ' + FDescription);
end;

procedure TQNNLoRAModule.Bind(var Weight: TQNNLinearWeight; const OutputOffset: Integer);
var
  Adapter: IQNNLinearAdapter;
begin
  Validate(Weight, OutputOffset);
  if FScale = 0 then
    Exit;
  if FBound and (FOffset <> OutputOffset) then
    raise EInvalidOp.Create('LoRA module already bound with a different row offset');
  FOffset := OutputOffset;
  FBound := True;
  Adapter := Self;
  Weight.AddAdapter(Adapter);
end;

procedure TQNNLoRAModule.Apply(const Dst, Src: TMemoryBlock; const Rows, InputDim, OutputDim: Integer);
const
  TILE_ROWS = 512;
var
  Hidden, Delta: TMemoryBlock;
  Start, Count: Integer;
  Source, Destination: TMemoryBlock;
begin
  if FScale = 0 then
    Exit;
  if not FBound or (InputDim <> FInput) or (Int64(FOffset) + FOutput > OutputDim) then
    raise EArgumentException.Create('Invalid LoRA binding: ' + FDescription);
  try
    Hidden := TMemoryBlock.Create([Min(Rows, TILE_ROWS), FRank], 'lora-rank-activation');
    Delta := TMemoryBlock.Create([Min(Rows, TILE_ROWS), FOutput], 'lora-output-delta');
    Start := 0;
    while Start < Rows do
    begin
      Count := Min(TILE_ROWS, Rows - Start);
      Source := Src + NativeInt(Start) * InputDim;
      QNNLinearNoBias(Hidden, Source, FDown, Count, FInput, FRank);
      QNNLinearNoBias(Delta, Hidden, FUp, Count, FRank, FOutput);
      for var Row := 0 to Count - 1 do
      begin
        Destination := Dst + NativeInt(Start + Row) * OutputDim + FOffset;
        for var J := 0 to FOutput - 1 do
          System.PSingle(Destination)[J] :=
            System.PSingle(Destination)[J] + FScale * System.PSingle(Delta)[NativeInt(Row) * FOutput + J];
      end;
      Inc(Start, Count);
    end;
  finally
    Hidden.Free;
    Delta.Free
  end;
end;

procedure TQNNLoRAModule.ApplyGPU(const Context: TQNNGPUContext; const Dst, Src: TQNNGPUTensor; const Rows, InputDim, OutputDim: Integer);
var
  Hidden, Delta: TQNNGPUTensor;
begin
  if FScale = 0 then
    Exit;
  if not FBound or (InputDim <> FInput) or (Int64(FOffset) + FOutput > OutputDim) then
    raise EArgumentException.Create('Invalid resident LoRA binding');
  Hidden := Context.Scratch(2, Rows * FRank);
  Delta := Context.Scratch(3, Rows * FOutput);
  Context.Linear(Hidden, Src, FDown, Default(TMemoryBlock), Default(TMemoryBlock), Rows, FInput, FRank);
  Context.Linear(Delta, Hidden, FUp, Default(TMemoryBlock), Default(TMemoryBlock), Rows, FRank, FOutput);
  Context
    .Kernel(GO_ADD, Rows * FOutput, Dst, Delta, nil, nil, [FOutput, OutputDim, QNNGPUFloatBits(FScale), FOffset, 0]);
end;

function CanonicalTarget(const Name: string): string;
const
  PREFIXES: array[0..2] of string = ('diffusion_model.', 'base_model.model.', 'transformer.');
begin
  Result := Name;
  for var Prefix in PREFIXES do
    if Result.StartsWith(Prefix) then
    begin
      Result := Result.Substring(Length(Prefix));
      Break
    end;
end;

constructor TQNNLoRASet.Create(const Specs: TArray<TQNNLoRASpec>);
var
  FileData: TSafeTensorFile;
  FileName, Name, Target, DownSuffix, UpSuffix, DownName, UpName, AlphaName: string;
  Seen: TDictionary<string, Boolean>;
  Module: IQNNLoRAModule;
  Found: Integer;
  Recognized: Boolean;
begin
  inherited Create;
  FFiles := TDictionary<string, TSafeTensorFile>.Create;
  FModules := TList<IQNNLoRAModule>.Create;
  FSpecs := Copy(Specs);
  for var Spec in Specs do
  begin
    if IsNan(Spec.Strength) or IsInfinite(Spec.Strength) then
      raise EArgumentException.Create('LoRA strength must be finite');
    FileName := ExpandFileName(Spec.FileName);
    if not FFiles.TryGetValue(LowerCase(FileName), FileData) then
    begin
      FileData := TSafeTensorFile.Open(FileName);
      FFiles.Add(LowerCase(FileName), FileData);
    end;
    Seen := TDictionary<string, Boolean>.Create;
    try
      Found := 0;
      for var Tensor in FileData.Tensors do
      begin
        Name := Tensor.Name;
        DownSuffix := '';
        UpSuffix := '';
        if Name.EndsWith('.lora_A.weight') then
        begin
          DownSuffix := '.lora_A.weight';
          UpSuffix := '.lora_B.weight'
        end
        else if Name.EndsWith('.lora_down.weight') then
        begin
          DownSuffix := '.lora_down.weight';
          UpSuffix := '.lora_up.weight'
        end;
        if DownSuffix = '' then
          Continue;
        Target := Name.Substring(0, Length(Name) - Length(DownSuffix));
        DownName := Name;
        UpName := Target + UpSuffix;
        AlphaName := Target + '.alpha';
        if FileData.Find(AlphaName) = nil then
          AlphaName := Target + '.lora_alpha';
        Module := TQNNLoRAModule.Create(FileData, CanonicalTarget(Target), DownName, UpName, AlphaName, Spec.Strength);
        FModules.Add(Module);
        Inc(Found);
        Seen.AddOrSetValue(DownName, True);
        Seen.AddOrSetValue(UpName, True);
        if FileData.Find(AlphaName) <> nil then
          Seen.AddOrSetValue(AlphaName, True);
      end;
      if Found = 0 then
        raise ESafeTensorError.Create('No supported LoRA A/B modules: ' + FileName);
      for var Tensor in FileData.Tensors do
      begin
        Recognized := Seen.ContainsKey(Tensor.Name);
        if not Recognized then
          raise ESafeTensorError.Create('Unsupported or unpaired LoRA tensor: ' + Tensor.Name + ' in ' + FileName);
      end;
    finally
      Seen.Free
    end;
  end;
end;

destructor TQNNLoRASet.Destroy;
begin
  FModules.Free;
  FFiles.Free;
  inherited
end;

procedure TQNNLoRASet.Attach(const Resolver: TQNNLoRAResolver);
var
  Weight: PQNNLinearWeight;
  Offset: Integer;
begin
  if not Assigned(Resolver) then
    raise EArgumentException.Create('LoRA target resolver missing');
  // First validate every module. Unsupported targets cannot leave a partially
  // attached stack behind, and no adapter may silently disappear.
  for var Module in FModules do
  begin
    Weight := Resolver(Module.Target, Offset);
    if Weight = nil then
      raise ESafeTensorError.Create('Unsupported LoRA target: ' + Module.Description);
    Module.Validate(Weight^, Offset);
  end;
  for var Module in FModules do
  begin
    Weight := Resolver(Module.Target, Offset);
    Module.Bind(Weight^, Offset)
  end;
end;

function TQNNLoRASet.ModuleCount: Integer;
begin
  Result := FModules.Count
end;

function TQNNLoRASet.EnabledModuleCount: Integer;
begin
  Result := 0;
  for var Module in FModules do
    if Module.Enabled then
      Inc(Result);
end;

end.

