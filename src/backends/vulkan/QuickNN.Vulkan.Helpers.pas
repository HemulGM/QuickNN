unit QuickNN.Vulkan.Helpers;

{$C+} // or {$ASSERTIONS ON} override to compile with assertion routine for SAFE_CALL
{$POINTERMATH on}
{$Z4}
{$T+}
interface

uses
  System.SysUtils, System.Generics.Collections, System.Math, System.TypInfo,
  QuickNN.Vulkan.API;

const
  MAX_COMMAND_BUFFER_COUNT = 1;

type
  PHalf = ^Half;

  Half = type SmallInt;
  { TWorkloadSizes }

  TWorkloadSizes = record
    X, Y, Z: LongWord;
    class operator Implicit(const A: TArray<LongWord>): TWorkloadSizes;
  end;

  { TVulkanDevice }

  TVulkanDevice = record
  public
    PhysicalDevice: VkPhysicalDevice;
    MemoryProperties: VkPhysicalDeviceMemoryProperties;
    DeviceProperties: VkPhysicalDeviceProperties;
    DeviceFeatures: VkPhysicalDeviceFeatures;
    QueueIndex: LongInt;
    QueuePropCount: LongWord;
    QueueProperties: TArray<VkQueueFamilyProperties>;
    function GetMemoryIndex(const MemType: LongWord; const MemPropertyFlags: VkMemoryPropertyFlags; const Preferred: VkMemoryPropertyFlags = 0
    ): LongWord;
  end;

  TVulkanPrecision = (
      // floating
  VpF32,
    VpBf16,
    VpF16,
    VpF8E5m2,
    VpF8E4m3,
    VpF4E2m1,
    VpF4E3m0,
    VpF64,
      // signed integers
    VpS32,
    VpS16,
    VpS8,
    VpS4,
      // unsigned integers
    VpU32,
    VpU16,
    VpU8,
    VpU4,
    VpBoolean
          // vpE8m0,
  );
  { TVulkanMemory }

  TVulkanMemory = record
    Buffer: VkBuffer;
    Memory: VkDeviceMemory;
    Size: VkDeviceSize;
    Precision: TVulkanPrecision;
    IsStaging: Boolean;
    Scaler: TArray<TVulkanMemory>;
    BlockSize: Int64;
    class operator Initialize(out Mem: TVulkanMemory);
  end;

{$IF not defined(Int4)}
  TInt4 = -8..7;
{$ENDIF}

  TVulkanArgType = (AtPointer, AtInt4, AtInt8, AtInt16, AtInt32, AtInt64, AtBF16, AtHalf, AtSingle, AtDouble);

  { TVulkanArg }

  TVulkanArg = record
    class operator Implicit(const Src: Pointer): TVulkanArg;
    class operator Implicit(const Src: ShortInt): TVulkanArg;
    class operator Implicit(const Src: SmallInt): TVulkanArg;
    class operator Implicit(const Src: LongInt): TVulkanArg;
    class operator Implicit(const Src: Int64): TVulkanArg;
    class operator Implicit(const Src: Half): TVulkanArg;
    class operator Implicit(const Src: Single): TVulkanArg;
    class operator Implicit(const Src: Double): TVulkanArg;
    class operator Implicit(const Src: TVulkanMemory): TVulkanArg;

    class operator Implicit(const Src: TVulkanArg): Pointer;
    class operator Implicit(const Src: TVulkanArg): ShortInt;
    class operator Implicit(const Src: TVulkanArg): SmallInt;
    class operator Implicit(const Src: TVulkanArg): LongInt;
    class operator Implicit(const Src: TVulkanArg): Int64;
    class operator Implicit(const Src: TVulkanArg): Half;
    class operator Implicit(const Src: TVulkanArg): Single;
    class operator Implicit(const Src: TVulkanArg): Double;
    class operator Initialize(out Src: TVulkanArg);
  public
    // argType : TVulkanArgType;
    // value   : uint64;
    Size: UInt64;
    case ArgType: TVulkanArgType of
      AtPointer:
        (VPointer: Pointer);
      AtInt4:
        (VInt4: TInt4);
      AtInt8:
        (VInt8: ShortInt);
      AtInt16:
        (VInt16: SmallInt);
      AtInt32:
        (VInt32: LongInt);
      AtInt64:
        (VInt64: Int64);
      AtHalf:
        (VHalf: Half);
      AtSingle:
        (VSingle: Single);
      AtDouble:
        (VDouble: Double);
  end;

  { TVulkanPipeline }

  TVulkanPipeline = record
    ShaderId: LongWord;
    Shader: VkShaderModule;
    Device: VkDevice;
    Pipeline: VkPipeline;
    PipelineLayout: VkPipelineLayout;
    DescPool: VkDescriptorPool;
    DescSet: VkDescriptorSet;
    DescSets: TArray<VkDescriptorSet>;
    DescSetLayout: VkDescriptorSetLayout;
    ConstantRange: VkPushConstantRange;
    BufferCount: Integer;
    procedure Free();
  end;

  TVulkanCommand = record
    Buffers: array[0..MAX_COMMAND_BUFFER_COUNT - 1] of VkCommandBuffer;
    Fences: array[0..MAX_COMMAND_BUFFER_COUNT - 1] of VkFence;
  end;

  TVulkanArgs = TArray<TVulkanArg>;

const
  VULKAN_ARGS_MAX_BUFFERS_COUNT = $10;
  VULKAN_ARGS_MAX_CONSTANTS_COUNT = $10;

type
  TKernelHandle = type LongInt;
  { TVulkanCompute }

  TVulkanCompute = class
    type
      // TVulkanOperations = TDictionary<TNNOperation, TVulkanPipeline>;
      TVulkanConstants = array[0..VULKAN_ARGS_MAX_CONSTANTS_COUNT - 1] of LongInt;
    // TSPIVTDictionary = TDictionary<rawbytestring, TKernelHandle>;
    const
      VK_BUFFER_USAGE_STORAGE = ( LongWord(VK_BUFFER_USAGE_TRANSFER_DST_BIT) or LongWord(VK_BUFFER_USAGE_TRANSFER_SRC_BIT))
        or LongWord(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
      VK_BUFFER_USAGE_STAGING = ( LongWord(VK_BUFFER_USAGE_TRANSFER_DST_BIT) or LongWord(VK_BUFFER_USAGE_TRANSFER_SRC_BIT));
      VK_MEMORY_PROPERTY_STORAGE = LongWord(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
      VK_MEMORY_PROPERTY_STAGING = LongWord(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) or LongWord(VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  private
    class function LoadFile(const FileName: TFileName): RawByteString; static;
    // queueFlags             : TArray<VkQueueFlags>;
    class procedure CreateInstance(const AppName: AnsiString); static;
    class procedure DestroyInstance(); static;
    class function DeviceCount(): LongWord; static;
    class function QueryPhysicalDevices(const Flags: VkQueueFlags = VkQueueFlags(VK_QUEUE_COMPUTE_BIT)
        ): TArray<TVulkanDevice>; static;

  protected
    VulkanDevice: TVulkanDevice;
    Device: VkDevice;
    Queue: VkQueue;
    CmdPool: VkCommandPool;
    Cmd: TVulkanCommand;

  // shaders               : TArray<VkShaderModule>;

  // Ops                   : TVulkanOperations;

  protected
    VulkanPipelines: TArray<TVulkanPipeline>;
    CommandBufferStarted: Boolean;
    procedure MemoryBarrier(const SrcStage, DstStage: VkPipelineStageFlags; const SrcAccess, DstAccess: VkAccessFlags);
    function CreateShader(const Code: Pointer; const ByteSize: NativeInt): VkShaderModule;
  public
    class var
      Instance: VkInstance;
      VulkanDevices: TArray<TVulkanDevice>;
    constructor Create(const DeviceIndex: LongInt = 0; const QueuePriority: Single = 1.0);
    destructor Destroy; override;
    // function CompileSPIRV(const sourcefile:TFileName):RawByteString;
    function CreateBuffer(const ByteSize: VkDeviceSize; const Usage: VkBufferUsageFlags = VK_BUFFER_USAGE_STORAGE
        ): VkBuffer;
    function AllocMem(const Buffer: VkBuffer; const MemProperty: VkMemoryPropertyFlags = VK_MEMORY_PROPERTY_STORAGE; const Preferred: VkMemoryPropertyFlags = 0
        ): VkDeviceMemory;
    function CreateStorageMemory(const Size: UInt64): TVulkanMemory;
    function CreateStagingMemory(const Size: UInt64; const HostCached: Boolean = False): TVulkanMemory;
    procedure FreeMemory(const Mem: TVulkanMemory);
    function MapMemory(const Mem: TVulkanMemory; const AOffset: LongWord = 0): Pointer;
    procedure UnMapMemory(const Mem: TVulkanMemory);
    procedure PushToDevice(const AData: Pointer; const Mem: TVulkanMemory; N: NativeInt = -1);
    procedure PullFromDevice(const AData: Pointer; const Mem: TVulkanMemory; N: NativeInt = -1);
    procedure CopyBuffer(const Src, Dst: TVulkanMemory; const N: VkDeviceSize = VK_WHOLE_SIZE; const SrcOffset: VkDeviceSize = 0; const DstOffset: VkDeviceSize = 0
        );
    function LoadShader(const FileNameSpirV: string): VkShaderModule;
    procedure PreparePipeline(const Op: TKernelHandle; const Shader: VkShaderModule; const LocalThreads: TWorkloadSizes; const BufferCount: Integer = VULKAN_ARGS_MAX_BUFFERS_COUNT; const SetCount: Integer = 1
        ); overload;
    procedure PreparePipeline(const Op: TKernelHandle; const Shader: VkShaderModule); overload;
    procedure DispatchPipeline(const Op: TKernelHandle; const Args: TVulkanArgs; const X: LongWord; const Y: LongWord = 1; const Z: LongWord = 1; const SetIndex: Integer = 0
        );
    procedure BeginCommadBuffer;
    procedure EndCommandBuffer;
    procedure Finish;
  end;

type
  Nfloat = Single;

  Pnfloat = ^Nfloat;

procedure CheckVulkanResult(const Res: VkResult); overload; // inline;

function CeilDiv(const M, N: LongWord): LongWord; inline;

function ArraySum(const N: LongWord; const A: Pnfloat): Single; overload;

function ArraySum(const A: TArray<Nfloat>): Single; overload;

function ArrayMax(const N: LongWord; const A: Pnfloat): Nfloat; overload;

function ArrayMax(const A: TArray<Nfloat>): Nfloat; overload;

function ArrayMin(const N: LongWord; const A: Pnfloat): Nfloat; overload;

function ArrayMin(const A: TArray<Nfloat>): Nfloat; overload;

procedure ArrayMinMax(const N: LongWord; const A: Pnfloat; const LocalMin, LocalMax: Pnfloat); overload;

procedure ArrayMinMax(const A: TArray<Nfloat>; var LocalMin, LocalMax: Nfloat); overload;

function ArraySumSqrDiff(const N: LongWord; const A: Pnfloat; const B: Single): Single; overload;

function ArraySumSqrDiff(const A: TArray<Nfloat>; const B: Single): Single; overload;

function ArrayVariance(const N: LongWord; const Mean: Single; const A: Pnfloat): Single; overload;

function ArrayVariance(const N: LongWord; const A: Pnfloat): Single; overload;

function ArrayVariace(const A: TArray<Nfloat>): Single; overload;

procedure ArrayRand(const N: LongWord; const Start, Finish: Nfloat; const A: Pnfloat); overload;

function ArrayRand(const N: LongWord; const Start: Single = -1; const Finish: Single = 1): TArray<Nfloat>; overload;

procedure ArrayLinSpace(const N: LongWord; const Start, Finish: Nfloat; const A: Pnfloat); overload;

function ArrayLinSpace(const N: LongWord; const Start: Single = -1; const Finish: Single = 1): TArray<Nfloat>; overload;

procedure ArrayAdd(const N: LongWord; const A, B, C: Pnfloat); overload;

procedure ArrayAdd(const A, B, C: TArray<Nfloat>); overload;

procedure ArrayAxpy(const N: LongWord; const A: Nfloat; const X, Y: Pnfloat);

procedure ArrayPrint(const N: LongWord; const A: Pnfloat; const Decimals: Word = 2); overload;

procedure ArrayPrint(const A: TArray<Nfloat>; const Decimals: Word = 2); overload;

procedure ArrayStat(const N: LongWord; const A: Pnfloat; const Decimals: Word = 2); overload;

procedure ArrayStat(const A: TArray<Nfloat>; const Decimals: Word = 2); overload;

function ArraySumSqrDiff(const N: LongWord; const A, B: Pnfloat): Single; overload;

function ArraySumSqrDiff(const A, B: TArray<Nfloat>): Single; overload;

procedure ArrayPrintDiff(const N: LongWord; const A, B: Pnfloat; const Eps: Single = 0.000001); overload;

procedure ArrayPrintDiff(const A, B: TArray<Nfloat>; const Eps: Single = 0.000001); overload;

procedure ArrayFma(const N: LongWord; const A, B: Nfloat; const X, Y: Pnfloat); overload;

procedure ArrayGemmNN(const M, N, K: LongWord; const Alpha: Single; const A: TArray<Nfloat>; const Lda: LongWord; const B: TArray<Nfloat>; const Ldb: LongWord; const Beta: Single; const C: TArray<Nfloat>; const Ldc: LongWord); overload;

procedure ArrayGemmTN(const M, N, K: LongWord; const Alpha: Single; const A: TArray<Nfloat>; const Lda: LongWord; const B: TArray<Nfloat>; const Ldb: LongWord; const Beta: Single; const C: TArray<Nfloat>; const Ldc: LongWord); overload;

procedure ArrayGemmNT(const M, N, K: LongWord; const Alpha: Single; const A: TArray<Nfloat>; const Lda: LongWord; const B: TArray<Nfloat>; const Ldb: LongWord; const Beta: Single; const C: TArray<Nfloat>; const Ldc: LongWord); overload;

implementation
{$IFDEF MSWINDOWS}

uses
  Winapi.Windows;
{$ENDIF}

procedure CheckVulkanResult(const Res: VkResult);
begin
  if Res <> VK_SUCCESS then
    raise Exception.CreateFmt('Vulkan error %d', [LongInt(Res)])
end;

function VK_MAKE_API_VERSION(const Variant, Major, Minor, Patch: LongWord): LongWord; inline;
begin
  Result := ((Variant shl 29) or (Major shl 22) or (Minor shl 12) or Patch)
end;

function VK_MAKE_VERSION(const Major, Minor, Patch: LongWord): LongWord; inline;
begin
  Result := ((Major shl 22) or (Minor shl 12) or Patch)
end;

{ TWorkloadSizes }

class operator TWorkloadSizes.Implicit(const A: TArray<LongWord>): TWorkloadSizes;
begin
  Move(Pointer(A)^, Result, Min(Length(A), 3) * SizeOf(LongInt))
end;

{ TVulkanDevice }

function TVulkanDevice.GetMemoryIndex(const MemType: LongWord; const MemPropertyFlags: VkMemoryPropertyFlags; const Preferred: VkMemoryPropertyFlags): LongWord;
var
  I: LongWord;
begin
  if Preferred <> 0 then
    for I := 0 to MemoryProperties.MemoryTypeCount - 1 do
      if ((MemType and (LongWord(1) shl I)) <> 0)
        and ((MemoryProperties.MemoryTypes[I].PropertyFlags and (MemPropertyFlags or Preferred))
        = (MemPropertyFlags or Preferred)) then
        Exit(I);
  for I := 0 to MemoryProperties.MemoryTypeCount - 1 do
    if (MemType and (1 shl I) > 0)
      and ((MemoryProperties.MemoryTypes[I].PropertyFlags and MemPropertyFlags) = MemPropertyFlags) then
      Exit(I);
  raise Exception.Create('No compatible Vulkan memory type');
end;

{ TVulkanMemory }

class operator TVulkanMemory.Initialize(out Mem: TVulkanMemory);
begin
  FillChar(Mem, SizeOf(TVulkanMemory), #0)
end;

{ TVulkanArg }

class operator TVulkanArg.Implicit(const Src: Pointer): TVulkanArg;
begin
  Result.ArgType := AtPointer;
  Result.VPointer := Src;
end;

class operator TVulkanArg.Implicit(const Src: ShortInt): TVulkanArg;
begin
  Result.ArgType := AtInt8;
  Result.VInt8 := Src;
end;

class operator TVulkanArg.Implicit(const Src: SmallInt): TVulkanArg;
begin
  Result.ArgType := AtInt16;
  Result.VInt16 := Src;
end;

class operator TVulkanArg.Implicit(const Src: LongInt): TVulkanArg;
begin
  Result.ArgType := AtInt32;
  Result.VInt32 := Src;
end;

class operator TVulkanArg.Implicit(const Src: Int64): TVulkanArg;
begin
  Result.ArgType := AtInt64;
  Result.VInt64 := Src;
end;

class operator TVulkanArg.Implicit(const Src: Half): TVulkanArg;
begin
  Result.ArgType := AtHalf;
  Result.VHalf := Src;
end;

class operator TVulkanArg.Implicit(const Src: Single): TVulkanArg;
begin
  Result.ArgType := AtSingle;
  Result.VSingle := Src;
end;

class operator TVulkanArg.Implicit(const Src: Double): TVulkanArg;
begin
  Result.ArgType := AtDouble;
  Result.VDouble := Src;
end;

class operator TVulkanArg.Implicit(const Src: TVulkanMemory): TVulkanArg;
begin
  Result.Size := Src.Size;
  Result.VPointer := Src.Buffer;
end;

class operator TVulkanArg.Implicit(const Src: TVulkanArg): Pointer;
begin
  Result := Src.VPointer;
end;

class operator TVulkanArg.Implicit(const Src: TVulkanArg): ShortInt;
begin
  Result := Src.VInt8;
end;

class operator TVulkanArg.Implicit(const Src: TVulkanArg): SmallInt;
begin
  Result := Src.VInt16;
end;

class operator TVulkanArg.Implicit(const Src: TVulkanArg): LongInt;
begin
  Result := Src.VInt32;
end;

class operator TVulkanArg.Implicit(const Src: TVulkanArg): Int64;
begin
  Result := Src.VInt64;
end;

class operator TVulkanArg.Implicit(const Src: TVulkanArg): Half;
begin
  Result := Src.VHalf;
end;

class operator TVulkanArg.Implicit(const Src: TVulkanArg): Single;
begin
  Result := Src.VSingle;
end;

class operator TVulkanArg.Implicit(const Src: TVulkanArg): Double;
begin
  Result := Src.VDouble;
end;

class operator TVulkanArg.Initialize(out Src: TVulkanArg);
begin
  FillChar(Src, SizeOf(TVulkanArg), 0)
end;

{ TVulkanPipeline }

procedure TVulkanPipeline.Free();
begin
  if Device <> nil then
  begin
    if Pipeline <> nil then
      VkDestroyPipeline(Device, Pipeline, nil);
    if DescPool <> nil then
      VkDestroyDescriptorPool(Device, DescPool, nil);
    if PipelineLayout <> nil then
      VkDestroyPipelineLayout(Device, PipelineLayout, nil);
    if DescSetLayout <> nil then
      VkDestroyDescriptorSetLayout(Device, DescSetLayout, nil);
    if Shader <> nil then
      VkDestroyShaderModule(Device, Shader, nil);
  end;
  Self := Default(TVulkanPipeline);
end;

{ TVulkanCompute }

class function TVulkanCompute.LoadFile(const FileName: TFileName): RawByteString;
var
  F: file;
begin
  Assert(FileExists(FileName), 'File not found!');
  try
    AssignFile(F, FileName);
    Reset(F, 1);
    SetLength(Result, FileSize(F));
    if Length(Result) > 0 then
      BlockRead(F, Result[1], Length(Result));
  finally
    Close(F)
  end;
end;

class procedure TVulkanCompute.CreateInstance(const AppName: AnsiString);
var
  Info: VkInstanceCreateInfo;
  Application: VkApplicationInfo;
begin
  Instance := nil;
  Application := Default(VkApplicationInfo);
  Application.SType := VK_STRUCTURE_TYPE_APPLICATION_INFO;
  Application.PApplicationName := PAnsiChar(AppName);
  Application.ApplicationVersion := VK_MAKE_VERSION(1, 0, 0);
  Application.ApiVersion := VK_API_VERSION_1_0;
  Info := Default(VkInstanceCreateInfo);
  Info.SType := VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  Info.PApplicationInfo := @Application;
  CheckVulkanResult(VkCreateInstance(@Info, nil, @Instance));
end;

class procedure TVulkanCompute.DestroyInstance;
begin
  if Instance <> nil then
    VkDestroyInstance(Instance, nil);
  Instance := nil;
end;

class function TVulkanCompute.DeviceCount(): LongWord;
begin
  Result := 0;
  CheckVulkanResult(VkEnumeratePhysicalDevices(Instance, @Result, nil))
end;

class function TVulkanCompute.QueryPhysicalDevices(const Flags: VkQueueFlags): TArray<TVulkanDevice>;
var
  Pd: TArray<VkPhysicalDevice>;
  R: TArray<TVulkanDevice>;
  DevCount: LongWord;
  I, J: Integer;
begin
  DevCount := DeviceCount();
  Result := nil;
  SetLength(R, DevCount);
  SetLength(Pd, DevCount);
  CheckVulkanResult(VkEnumeratePhysicalDevices(Instance, @DevCount, Pointer(Pd)));
  for I := 0 to High(R) do
  begin
    R[I].QueueIndex := -1;
    ;
    R[I].PhysicalDevice := Pd[I];
    VkGetPhysicalDeviceProperties(Pd[I], @R[I].DeviceProperties);
    VkGetPhysicalDeviceFeatures(Pd[I], @R[I].DeviceFeatures);
    VkGetPhysicalDeviceMemoryProperties(Pd[I], @R[I].MemoryProperties);
    VkGetPhysicalDeviceQueueFamilyProperties(Pd[I], @R[I].QueuePropCount, nil);
    SetLength(R[I].QueueProperties, R[I].QueuePropCount);
    VkGetPhysicalDeviceQueueFamilyProperties(Pd[I], @R[I].QueuePropCount, Pointer(R[I].QueueProperties));
    for J := 0 to R[I].QueuePropCount - 1 do
      if 0 <> (R[I].QueueProperties[J].QueueFlags and Flags) then
      begin
        R[I].QueueIndex := J;
        Insert(R[I], Result, Length(Result));
        Break
      end;
  end;
end;

constructor TVulkanCompute.Create(const DeviceIndex: LongInt; const QueuePriority: Single);
var
  QueueInfo: VkDeviceQueueCreateInfo;
  DeviceInfo: VkDeviceCreateInfo;
  PoolInfo: VkCommandPoolCreateInfo;
  AllocInfo: VkCommandBufferAllocateInfo;
  FenceInfo: VkFenceCreateInfo;
  I: Integer;
begin
  inherited Create;
  if Instance = nil then
  begin
    CreateInstance(AnsiString(ExtractFileName(ParamStr(0))));
    try
      VulkanDevices := QueryPhysicalDevices();
    except
      DestroyInstance;
      raise;
    end;
  end;
  if Length(VulkanDevices) = 0 then
    raise Exception.Create('No Vulkan compute device');
  I := DeviceIndex;
  if I < 0 then
  begin
    I := 0;
    for var Candidate := 0 to High(VulkanDevices) do
      if VulkanDevices[Candidate].DeviceProperties.DeviceType = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU then
      begin
        I := Candidate;
        Break;
      end;
  end;
  if I >= Length(VulkanDevices) then
    raise ERangeError.Create('Invalid Vulkan device index');
  VulkanDevice := VulkanDevices[I];
  QueueInfo := Default(VkDeviceQueueCreateInfo);
  QueueInfo.SType := VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  QueueInfo.QueueFamilyIndex := VulkanDevice.QueueIndex;
  QueueInfo.QueueCount := 1;
  QueueInfo.PQueuePriorities := @QueuePriority;

  DeviceInfo := Default(VkDeviceCreateInfo);
  DeviceInfo.SType := VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  DeviceInfo.QueueCreateInfoCount := 1;
  DeviceInfo.PQueueCreateInfos := @QueueInfo;
  CheckVulkanResult(VkCreateDevice(VulkanDevice.PhysicalDevice, @DeviceInfo, nil, @Device));
  VkGetDeviceQueue(Device, VulkanDevice.QueueIndex, 0, @Queue);

  PoolInfo := Default(VkCommandPoolCreateInfo);
  PoolInfo.SType := VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  PoolInfo.QueueFamilyIndex := VulkanDevice.QueueIndex;
  PoolInfo.Flags := LongWord(VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
  CheckVulkanResult(VkCreateCommandPool(Device, @PoolInfo, nil, @CmdPool));

  AllocInfo := Default(VkCommandBufferAllocateInfo);
  AllocInfo.SType := VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  AllocInfo.CommandPool := CmdPool;
  AllocInfo.Level := VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  AllocInfo.CommandBufferCount := MAX_COMMAND_BUFFER_COUNT;
  CheckVulkanResult(VkAllocateCommandBuffers(Device, @AllocInfo, @Cmd.Buffers[0]));

  FenceInfo := Default(VkFenceCreateInfo);
  FenceInfo.SType := VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  for I := 0 to High(Cmd.Fences) do
  begin
    Cmd.Fences[I] := nil;
    CheckVulkanResult(VkCreateFence(Device, @FenceInfo, nil, @Cmd.Fences[I]));
  end;

  // Ops := TVulkanOperations.Create;

  // CompileSPIRV('Z:\Development\vulkan_gemm\vec_add.comp');
end;

destructor TVulkanCompute.Destroy;
var
  I: Integer;
begin
  if Device <> nil then
  begin
    VkDeviceWaitIdle(Device);
    for I := 0 to High(VulkanPipelines) do
      VulkanPipelines[I].Free;
    for I := 0 to High(Cmd.Fences) do
      if Cmd.Fences[I] <> nil then
        VkDestroyFence(Device, Cmd.Fences[I], nil);
    if CmdPool <> nil then
      VkDestroyCommandPool(Device, CmdPool, nil);
    VkDestroyDevice(Device, nil);
  end;
  inherited;
end;

{
function TVulkanCompute.CompileSPIRV(const sourcefile: TFileName): RawByteString;
var
  shader, prog : GLuint;
  source : ansistring;
  sourcePtr : PAnsiChar;
  sourceLen : GlInt;
begin
  shader := glCreateShader(GL_COMPUTE_SHADER);
  source := loadFile(sourcefile);
  sourcePtr := PAnsiChar(source);
  sourceLen := length(source);

  glShaderSource(shader, 1, @sourcePtr, @sourceLen);
  glCompileShader(shader);
  glGetShaderiv(shader, GL_COMPILE_STATUS, @sourceLen);

  //prog := glCreateProgram();
  //glAttachShader(prog, shader);
  //glLinkProgram(prog);
  //glGetProgramiv(prog, GL_PROGRAM_BINARY_LENGTH, @sourceLen);


end;
}

function TVulkanCompute.CreateBuffer(const ByteSize: VkDeviceSize; const Usage: VkBufferUsageFlags): VkBuffer;
var
  CreateInfo: VkBufferCreateInfo;
begin
  CreateInfo := Default(VkBufferCreateInfo);
  CreateInfo.SType := VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  CreateInfo.Size := ByteSize;
  CreateInfo.Usage := Usage;
  CreateInfo.SharingMode := VK_SHARING_MODE_EXCLUSIVE;
  CheckVulkanResult(VkCreateBuffer(Device, @CreateInfo, nil, @Result))
end;

function TVulkanCompute.AllocMem(const Buffer: VkBuffer; const MemProperty: VkMemoryPropertyFlags; const Preferred: VkMemoryPropertyFlags): VkDeviceMemory;
var
  MemReq: VkMemoryRequirements;
  MemInfo: VkMemoryAllocateInfo;
begin
  VkGetBufferMemoryRequirements(Device, Buffer, @MemReq);
  MemInfo := Default(VkMemoryAllocateInfo);
  MemInfo.SType := VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  MemInfo.AllocationSize := MemReq.Size;
  MemInfo.MemoryTypeIndex := VulkanDevice.GetMemoryIndex(MemReq.MemoryTypeBits, MemProperty, Preferred);
  try
    CheckVulkanResult(VkAllocateMemory(Device, @MemInfo, nil, @Result));
  except
    begin
      raise
    end;
  end;
  try
    CheckVulkanResult(VkBindBufferMemory(Device, Buffer, Result, 0));
  except
    VkFreeMemory(Device, Result, nil);
    raise;
  end;
end;

function TVulkanCompute.CreateStorageMemory(const Size: UInt64): TVulkanMemory;
begin
  Result.Buffer := CreateBuffer(Size);
  try
    Result.Memory := AllocMem(Result.Buffer);
  except
    VkDestroyBuffer(Device, Result.Buffer, nil);
    raise;
  end;
  Result.Size := Size;
  Result.IsStaging := False
end;

function TVulkanCompute.CreateStagingMemory(const Size: UInt64; const HostCached: Boolean): TVulkanMemory;
begin
  Result.Buffer := CreateBuffer(Size, VK_BUFFER_USAGE_STAGING);
  try
    Result.Memory :=
      AllocMem(
      Result.Buffer,
      VK_MEMORY_PROPERTY_STAGING,
      LongWord(VK_MEMORY_PROPERTY_HOST_CACHED_BIT) * LongWord(Ord(HostCached))
    );
  except
    VkDestroyBuffer(Device, Result.Buffer, nil);
    raise;
  end;
  Result.Size := Size;
  Result.IsStaging := True
end;

procedure TVulkanCompute.FreeMemory(const Mem: TVulkanMemory);
begin
  if Mem.Buffer <> nil then
    VkDestroyBuffer(Device, Mem.Buffer, nil);
  if Mem.Memory <> nil then
    VkFreeMemory(Device, Mem.Memory, nil);
end;

function TVulkanCompute.MapMemory(const Mem: TVulkanMemory; const AOffset: LongWord): Pointer;
begin
  CheckVulkanResult(VkMapMemory(Device, Mem.Memory, AOffset, VK_WHOLE_SIZE, 0, @Result));
end;

procedure TVulkanCompute.UnMapMemory(const Mem: TVulkanMemory);
begin
  VkUnMapMemory(Device, Mem.Memory);
end;

procedure TVulkanCompute.PushToDevice(const AData: Pointer; const Mem: TVulkanMemory; N: NativeInt);
var
  D: Pointer;
  // cmdInfo  : VkCommandBufferBeginInfo;
  StagingMem: TVulkanMemory;
begin
  Assert(Assigned(AData));
  if N < 0 then
    N := Mem.Size;
  Assert(
    (not CommandBufferStarted) or (N < $10000),
    'ERROR [PushtoDevice] : buffer is too big to push to device during command record state!, retry after calling EndCommandBuffer()'
  );

  if CommandBufferStarted then
  begin
    VkCmdUpdateBuffer(Cmd.Buffers[0], Mem.Buffer, 0, N, AData);
    // cmdInfo:= default(VkCommandBufferBeginInfo);
    // cmdInfo.sType:= VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    // SAFE_CALL(vkBeginCommandBuffer(cmdBuffer, ))
  end
  else if Mem.IsStaging then
  begin
    D := MapMemory(Mem);
    Move(AData^, D^, N);
    UnMapMemory(Mem);
  end
  else
  begin
    StagingMem := CreateStagingMemory(N);
    D := MapMemory(StagingMem);
    Move(AData^, D^, N);
    UnMapMemory(StagingMem);
    BeginCommadBuffer;
    CopyBuffer(StagingMem, Mem);
    EndCommandBuffer;
    Finish;
    FreeMemory(StagingMem);
  end;
end;

procedure TVulkanCompute.PullFromDevice(const AData: Pointer; const Mem: TVulkanMemory; N: NativeInt);
var
  D: Pointer;
  StagingMem: TVulkanMemory;
  CmdStarted: Boolean;
begin
  Assert(Assigned(AData));
  if N < 0 then
    N := Mem.Size;
  if Mem.IsStaging then
  begin
    D := MapMemory(Mem);
    Move(D^, AData^, N);
    UnMapMemory(Mem);
  end
  else
  begin
    StagingMem := CreateStagingMemory(N, True);
    CmdStarted := CommandBufferStarted;
    if not CmdStarted then
      BeginCommadBuffer;
    CopyBuffer(Mem, StagingMem, N);
    EndCommandBuffer;
    Finish;
    D := MapMemory(StagingMem);
    Move(D^, AData^, N);
    UnMapMemory(StagingMem);
    FreeMemory(StagingMem);
    if CmdStarted then
      BeginCommadBuffer;
  end;
end;

procedure TVulkanCompute.CopyBuffer(const Src, Dst: TVulkanMemory; const N: VkDeviceSize; const SrcOffset, DstOffset: VkDeviceSize);
var
  Copy: VkBufferCopy;
begin
  if not CommandBufferStarted then
    raise Exception.Create('Vulkan command buffer is not recording');
  if (SrcOffset > Src.Size) or (DstOffset > Dst.Size) then
    raise ERangeError.Create('Vulkan copy offset outside buffer');
  Copy.SrcOffset := SrcOffset;
  Copy.DstOffset := DstOffset;
  if N = VK_WHOLE_SIZE then
    Copy.Size := Src.Size - SrcOffset
  else
    Copy.Size := N;
  if (Copy.Size = 0) or (Copy.Size > Src.Size - SrcOffset) or (Copy.Size > Dst.Size - DstOffset) then
    raise ERangeError.Create('Vulkan copy size outside buffer');
  VkCmdCopyBuffer(Cmd.Buffers[0], Src.Buffer, Dst.Buffer, 1, @Copy);
end;

procedure TVulkanCompute.MemoryBarrier(const SrcStage, DstStage: VkPipelineStageFlags; const SrcAccess, DstAccess: VkAccessFlags);
var
  Barrier: VkMemoryBarrier;
begin
  Barrier := Default(VkMemoryBarrier);
  Barrier.SType := VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  Barrier.SrcAccessMask := SrcAccess;
  Barrier.DstAccessMask := DstAccess;
  VkCmdPipelineBarrier(Cmd.Buffers[0], SrcStage, DstStage, 0, 1, @Barrier, 0, nil, 0, nil);
end;

function TVulkanCompute.CreateShader(const Code: Pointer; const ByteSize: NativeInt): VkShaderModule;
var
  Info: VkShaderModuleCreateInfo;
begin
  Info := Default(VkShaderModuleCreateInfo);
  Info.SType := VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  Info.CodeSize := ByteSize;
  Info.PCode := Code;
  Result := nil;
  CheckVulkanResult(VkCreateShaderModule(Device, @Info, nil, @Result));
end;

function TVulkanCompute.LoadShader(const FileNameSpirV: string): VkShaderModule;
var
  SPIRV: RawByteString;
  MInfo: VkShaderModuleCreateInfo;
begin
  // assert((index<0) or (index>=length(shaders)) or not assigned(shaders[index]), 'A shader at index ['+IntToStr(index)+'] is already loaded!');
  SPIRV := LoadFile(FileNameSpirV);
  MInfo := Default(VkShaderModuleCreateInfo);
  MInfo.SType := VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  MInfo.CodeSize := Length(SPIRV);
  MInfo.PCode := Puint32_t(SPIRV);
  CheckVulkanResult(VkCreateShaderModule(Device, @MInfo, nil, @Result));
end;

procedure TVulkanCompute.PreparePipeline(const Op: TKernelHandle; const Shader: VkShaderModule; const LocalThreads: TWorkloadSizes; const BufferCount, SetCount: Integer);
type
  TVkDescriptorSetLayoutBindings = array[0..VULKAN_ARGS_MAX_BUFFERS_COUNT - 1] of VkDescriptorSetLayoutBinding;
var
  // pl : TVulkanPipeline;
  I // , j, k
: LongInt;
  DescLayoutBinds: TVkDescriptorSetLayoutBindings;
  PCreateInfo: VkPipelineLayoutCreateInfo;
  DescCreateInfo: VkDescriptorSetLayoutCreateInfo;
  PStageInfo: VkPipelineShaderStageCreateInfo;
  PComputeInfo: VkComputePipelineCreateInfo;
  DescPoolSize: VkDescriptorPoolSize;
  DescPoolInfo: VkDescriptorPoolCreateInfo;
  AllocInfo: VkDescriptorSetAllocateInfo;
begin
  // j := 0; k := 0;
  // if not Ops.TryGetValue(Op, pl) then begin
  DescLayoutBinds := Default(TVkDescriptorSetLayoutBindings);
  // for i:=0 to high(args) do begin
  for I := 0 to BufferCount - 1 do
  begin
    // if args[i].argType = atPointer then begin
    DescLayoutBinds[I].Binding := I;
    DescLayoutBinds[I].DescriptorType := VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    DescLayoutBinds[I].DescriptorCount := 1;
    DescLayoutBinds[I].StageFlags := LongWord(VK_SHADER_STAGE_COMPUTE_BIT);
    // inc(j)
    // end;
  end;

  if Op > High(VulkanPipelines) then
    SetLength(VulkanPipelines, Op + 1)
  else
    VulkanPipelines[Op].Free();
  DescCreateInfo := Default(VkDescriptorSetLayoutCreateInfo);
  DescCreateInfo.SType := VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  DescCreateInfo.BindingCount := BufferCount;
  DescCreateInfo.PBindings := @DescLayoutBinds[0];

  // pl := default(TVulkanPipeline);
  // pushSize is set to TVulkanConstants.OP size (=4) plus the number of constants in
  // args multiplied by 4 aligning to the rest of TVulkanConstants
  VulkanPipelines[Op].Device := Device;
  VulkanPipelines[Op].Shader := Shader;
  VulkanPipelines[Op].BufferCount := BufferCount;
  CheckVulkanResult(VkCreateDescriptorSetLayout(Device, @DescCreateInfo, nil, @VulkanPipelines[Op].DescSetLayout));

  VulkanPipelines[Op].ConstantRange.StageFlags := LongWord(VK_SHADER_STAGE_COMPUTE_BIT);
  VulkanPipelines[Op].ConstantRange.Size := VULKAN_ARGS_MAX_CONSTANTS_COUNT * 4;
  PCreateInfo := Default(VkPipelineLayoutCreateInfo);
  PCreateInfo.SType := VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  PCreateInfo.SetLayoutCount := 1;
  PCreateInfo.PSetLayouts := @VulkanPipelines[Op].DescSetLayout;
  PCreateInfo.PushConstantRangeCount := 1;
  PCreateInfo.PPushConstantRanges := @VulkanPipelines[Op].ConstantRange;

  CheckVulkanResult(VkCreatePipelineLayout(Device, @PCreateInfo, nil, @VulkanPipelines[Op].PipelineLayout));

  PStageInfo := Default(VkPipelineShaderStageCreateInfo);
  PStageInfo.SType := VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  PStageInfo.Stage := VK_SHADER_STAGE_COMPUTE_BIT;
  // todo embed .spv files with in the executable resources
  PStageInfo.Module := Shader; // loadShader(SPIRVFile);//shaders[ord(Op)];
  PStageInfo.PName := 'main';
  {
    for i:=low(pSpecialMap) to high(pSpecialMap) do begin
      pSpecialMap[i].constantID := i;
      pSpecialMap[i].offset := i*sizeof(longword);
      pSpecialMap[i].size := sizeof(longword);
    end;
    //pSpecialInfo := default(VkSpecializationInfo);
    pSpecialInfo.mapEntryCount:= length(pSpecialMap);
    pSpecialInfo.pMapEntries := @pSpecialMap;
    pSpecialInfo.dataSize := sizeOf(localThreads);
    pSpecialInfo.pData := @localThreads;

    pStageInfo.pSpecializationInfo:= @pSpecialInfo;
}
  VulkanPipelines[Op].ShaderId := Op;
  PComputeInfo := Default(VkComputePipelineCreateInfo);
  PComputeInfo.SType := VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  PComputeInfo.Stage := PStageInfo;
  PComputeInfo.Layout := VulkanPipelines[Op].PipelineLayout;
  CheckVulkanResult(VkCreateComputePipelines(Device, nil, 1, @PComputeInfo, nil, @VulkanPipelines[Op].Pipeline));

  DescPoolSize.&type := VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  DescPoolSize.DescriptorCount := BufferCount * SetCount;

  DescPoolInfo := Default(VkDescriptorPoolCreateInfo);
  DescPoolInfo.SType := VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  DescPoolInfo.MaxSets := SetCount;
  DescPoolInfo.PoolSizeCount := 1;
  DescPoolInfo.PPoolSizes := @DescPoolSize;
  CheckVulkanResult(VkCreateDescriptorPool(Device, @DescPoolInfo, nil, @VulkanPipelines[Op].DescPool));

  AllocInfo := Default(VkDescriptorSetAllocateInfo);
  AllocInfo.SType := VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  AllocInfo.DescriptorPool := VulkanPipelines[Op].DescPool;
  AllocInfo.DescriptorSetCount := 1;
  AllocInfo.PSetLayouts := @VulkanPipelines[Op].DescSetLayout;
  SetLength(VulkanPipelines[Op].DescSets, SetCount);
  for I := 0 to SetCount - 1 do
    CheckVulkanResult(VkAllocateDescriptorSets(Device, @AllocInfo, @VulkanPipelines[Op].DescSets[I]));
  VulkanPipelines[Op].DescSet := VulkanPipelines[Op].DescSets[0];
end;

procedure TVulkanCompute.PreparePipeline(const Op: TKernelHandle; const Shader: VkShaderModule);
const
  DEFAULT_WORKLOAD: TWorkloadSizes = (X: $100; Y: 1; Z: 1);
begin
  PreparePipeline(Op, Shader, DEFAULT_WORKLOAD);
end;

procedure TVulkanCompute.DispatchPipeline(const Op: TKernelHandle; const Args: TVulkanArgs; const X: LongWord; const Y: LongWord; const Z: LongWord; const SetIndex: Integer);
type
  TVukanDescWrites = array[0..VULKAN_ARGS_MAX_BUFFERS_COUNT - 1] of VkWriteDescriptorSet;

  TVulkanDescBufferInfo = array[0..VULKAN_ARGS_MAX_BUFFERS_COUNT - 1] of VkDescriptorBufferInfo;
var
  BufInfo: TVulkanDescBufferInfo;
  Writes: TVukanDescWrites;
  I, J, K: LongInt;
  DescriptorSet: VkDescriptorSet;
  ConstArgs: TVulkanConstants;
begin
  // if not Ops.TryGetValue(Op, pl) then exit;
  if not CommandBufferStarted then
    raise Exception.Create('Vulkan dispatch is not recording');
  if (SetIndex < 0) or (SetIndex >= Length(VulkanPipelines[Op].DescSets)) then
    raise ERangeError.Create('Invalid Vulkan descriptor set slot');
  DescriptorSet := VulkanPipelines[Op].DescSets[SetIndex];
  J := 0;
  K := 0;
  Writes := Default(TVukanDescWrites);
  for I := 0 to High(Args) do
  begin
    case Args[I].ArgType of
      AtPointer:
        begin
          if J >= VulkanPipelines[Op].BufferCount then
            raise ERangeError.Create('Too many Vulkan buffers');
          BufInfo[J].Buffer := Args[I].VPointer;
          BufInfo[J].Offset := 0;
          BufInfo[J].Range := VK_WHOLE_SIZE;

          Writes[J].SType := VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
          Writes[J].DstSet := DescriptorSet;
          Writes[J].DstBinding := J;
          Writes[J].DescriptorCount := 1;
          Writes[J].DescriptorType := VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
          Writes[J].PBufferInfo := @BufInfo[J];
          Inc(J)
        end
    else
      begin
        if K >= VULKAN_ARGS_MAX_CONSTANTS_COUNT then
          raise ERangeError.Create('Too many Vulkan constants');
        ConstArgs[K] := Args[I].VInt32;
        Inc(K)
      end;
    end;
  end;
  VkCmdBindPipeline(Cmd.Buffers[0], VK_PIPELINE_BIND_POINT_COMPUTE, VulkanPipelines[Op].Pipeline);
  VkUpdateDescriptorSets(Device, J, @Writes[0], 0, nil);
  VkCmdBindDescriptorSets(
    Cmd.Buffers[0],
    VK_PIPELINE_BIND_POINT_COMPUTE,
    VulkanPipelines[Op].PipelineLayout,
    0,
    1,
    @DescriptorSet,
    0,
    nil
  );
  VkCmdPushConstants(
    Cmd.Buffers[0],
    VulkanPipelines[Op].PipelineLayout,
    LongWord(VK_SHADER_STAGE_COMPUTE_BIT),
    0,
    K * SizeOf(LongInt),
    @ConstArgs[0]
  );
  VkCmdDispatch(Cmd.Buffers[0], X, Y, Z);
end;

procedure TVulkanCompute.BeginCommadBuffer;
var
  Info: VkCommandBufferBeginInfo;
begin
  if CommandBufferStarted then
    Exit;
  CheckVulkanResult(VkResetCommandBuffer(Cmd.Buffers[0], 0));
  Info := Default(VkCommandBufferBeginInfo);
  Info.SType := VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  CheckVulkanResult(VkBeginCommandBuffer(Cmd.Buffers[0], @Info));
  CommandBufferStarted := True
end;

procedure TVulkanCompute.EndCommandBuffer;
begin
  if not CommandBufferStarted then
    Exit;
  CheckVulkanResult(VkEndCommandBuffer(Cmd.Buffers[0]));
  CommandBufferStarted := False
end;

procedure TVulkanCompute.Finish;
var
  SubmitInfo: VkSubmitInfo;
begin
  SubmitInfo := Default(VkSubmitInfo);
  SubmitInfo.SType := VK_STRUCTURE_TYPE_SUBMIT_INFO;
  SubmitInfo.CommandBufferCount := 1;
  SubmitInfo.PCommandBuffers := @Cmd.Buffers[0];
  CheckVulkanResult(VkResetFences(Device, 1, @Cmd.Fences[0]));
  CheckVulkanResult(VkQueueSubmit(Queue, 1, @SubmitInfo, Cmd.Fences[0]));
  CheckVulkanResult(VkWaitForFences(Device, 1, @Cmd.Fences[0], VK_TRUE, UInt64.MaxValue));

  // SAFE_CALL(vkQueueSubmit(queue, 1, @submitInfo, nil));
  // SAFE_CALL(vkQueueWaitIdle(queue));
  // SAFE_CALL(vkDeviceWaitIdle(device));
end;

function CeilDiv(const M, N: LongWord): LongWord; inline;
begin
  Result := (M + N - 1) div N
end;

// function max(const a,b:NativeInt):longword;inline; overload;
// begin
//  if a>=b then exit(a) else exit(b)
// end;

(*
function max(const a,b:nfloat):nfloat;inline; overload;
begin
  if a>=b then exit(a) else exit(b)
end;

function max(const a,b:NativeInt):nativeInt;inline; overload;
begin
  if a>=b then exit(a) else exit(b)
end;

//function min(const a,b:longword):longword;inline; overload;
//begin
//  if a<=b then exit(a) else exit(b)
//end;

function min(const a,b:nfloat):nfloat;inline; overload;
begin
  if a<=b then exit(a) else exit(b)
end;

function min(const a,b:nativeInt):nativeInt;inline; overload;
begin
  if a<=b then exit(a) else exit(b)
end;
*)

// function array_sum(const N:longword; const a:Pnfloat):nfloat; overload;
// var
//  i: longword;
// begin
//  if not assigned(a) or (N=0) then exit(0);
//  result := a[0];
//  for i:= 1 to N-1 do
//    result := result + a[i]
// end;

function ArraySum(const N: LongWord; const A: Pnfloat): Single; overload;
var
  I: LongWord;
begin
  if not Assigned(A) or (N = 0) then
    Exit(0);
  Result := Single(A[0]);
  for I := 1 to N - 1 do
    Result := Result + Single(A[I])
end;

// function array_sum(const a:TArray<nfloat>):nfloat; overload;
// begin
//  result := array_sum(length(a), pointer(a))
// end;

function ArraySum(const A: TArray<Nfloat>): Single; overload;
begin
  Result := ArraySum(Length(A), Pointer(A))
end;

function ArrayMax(const N: LongWord; const A: Pnfloat): Nfloat; overload;
var
  I: LongWord;
begin
  if not Assigned(A) or (N = 0) then
    Exit(0);
  Result := A[0];
  for I := 1 to N - 1 do
    Result := System.Math.Max(Result, A[I])
end;

function ArrayMax(const A: TArray<Nfloat>): Nfloat; overload;
begin
  Result := ArrayMax(Length(A), Pointer(A))
end;

function ArrayMin(const N: LongWord; const A: Pnfloat): Nfloat; overload;
var
  I: LongWord;
begin
  if not Assigned(A) or (N = 0) then
    Exit(0);
  Result := A[0];
  for I := 1 to N - 1 do
    Result := System.Math.Min(Result, A[I])
end;

function ArrayMin(const A: TArray<Nfloat>): Nfloat; overload;
begin
  Result := ArrayMax(Length(A), Pointer(A))
end;

procedure ArrayMinMax(const N: LongWord; const A: Pnfloat; const LocalMin, LocalMax: Pnfloat); overload;
var
  I: LongWord;
begin
  Assert(Assigned(LocalMin) and Assigned(LocalMax));
  LocalMin^ := A[0];
  LocalMax^ := A[0];
  for I := 1 to N - 1 do
  begin
    LocalMin^ := System.Math.Min(LocalMin^, A[I]);
    LocalMax^ := System.Math.Max(LocalMax^, A[I])
  end;
end;

procedure ArrayMinMax(const A: TArray<Nfloat>; var LocalMin, LocalMax: Nfloat); overload;
begin
  ArrayMinMax(Length(A), Pointer(A), @LocalMin, @LocalMax)
end;

// function array_sum_sqr_diff(const N:longword; const a:Pnfloat; const b:nfloat):nfloat;overload;
// var i:longword;
//  diff , r:single;
// begin
//  if not assigned(a) or (N=0) then exit(0);
//  diff := a[0] - b;
//  r := sqr(diff);
//  for i:=1 to N-1 do begin
//    diff := a[i] - b;
//    r := r + sqr(diff);
//  end;
//  result := r
// end;

function ArraySumSqrDiff(const N: LongWord; const A: Pnfloat; const B: Single): Single; overload;
var
  I: LongWord;
  Diff: Single;
begin
  if not Assigned(A) or (N = 0) then
    Exit(0);
  Diff := Single(A[0]) - B;
  Result := Sqr(Diff);
  for I := 1 to N - 1 do
  begin
    Diff := Single(A[I]) - B;
    Result := Result + Sqr(Diff);
  end;
end;

// function array_sum_sqr_diff(const a:TArray<nfloat>; const b:nfloat):nfloat;overload;
// begin
//  result := array_sum_sqr_diff(length(a), pointer(a), b);
// end;

function ArraySumSqrDiff(const A: TArray<Nfloat>; const B: Single): Single; overload;
begin
  Result := ArraySumSqrDiff(Length(A), Pointer(A), B);
end;

// function array_variance(const N:longword; const mean:nfloat; const a:Pnfloat):nfloat; overload;
// var
//  i:longword;
// begin
//  if not assigned(a) or (N=0) or (N=1) then exit(0);
//  result := array_sum_sqr_diff(N, a, mean)/(N-1);
// end;

function ArrayVariance(const N: LongWord; const Mean: Single; const A: Pnfloat): Single; overload;
begin
  if not Assigned(A) or (N = 0) or (N = 1) then
    Exit(0);
  Result := ArraySumSqrDiff(N, A, Mean) / (N - 1);
end;

// function array_variance(const N:longword; const a:Pnfloat):nfloat; overload;
// var mean : nfloat;
// begin
//  if not assigned(a) or (N=0) or (N=1) then exit(0);
//  mean := array_sum(N, a)/ N;
//  result := array_variance(N, mean, a)
// end;

function ArrayVariance(const N: LongWord; const A: Pnfloat): Single; overload;
var
  Mean: Single;
begin
  if not Assigned(A) or (N = 0) or (N = 1) then
    Exit(0);
  Mean := ArraySum(N, A) / N;
  Result := ArrayVariance(N, Mean, A)
end;

// function array_variace(const a:TArray<nfloat>):nfloat; overload;
// begin
//  result := array_variance(length(a), pointer(a))
// end;

function ArrayVariace(const A: TArray<Nfloat>): Single; overload;
begin
  Result := ArrayVariance(Length(A), Pointer(A))
end;

procedure ArrayRand(const N: LongWord; const Start, Finish: Nfloat; const A: Pnfloat); overload;
var
  I: LongWord;
  Scal: Nfloat;
begin
  Scal := Finish - Start;
  for I := 0 to N - 1 do
    A[I] := Random() * Scal + Start;
end;

function ArrayRand(const N: LongWord; const Start: Single = -1; const Finish: Single = 1): TArray<Nfloat>; overload;
begin
  SetLength(Result, N);
  ArrayRand(N, Start, Finish, Pointer(Result));
end;

procedure ArrayLinSpace(const N: LongWord; const Start, Finish: Nfloat; const A: Pnfloat); overload;
var
  I: LongWord;
  Scal: Nfloat;
begin
  Scal := Finish - Start;
  for I := 0 to N - 1 do
    A[I] := Start + I * Scal / N;
end;

function ArrayLinSpace(const N: LongWord; const Start: Single = -1; const Finish: Single = 1): TArray<Nfloat>; overload;
begin
  SetLength(Result, N);
  ArrayLinSpace(N, Start, Finish, Pointer(Result))
end;

procedure ArrayAdd(const N: LongWord; const A, B, C: Pnfloat); overload;
var
  I: LongWord;
begin
  Assert(Assigned(A) and Assigned(B) and Assigned(C));
  for I := 0 to N - 1 do
    C[I] := A[I] + B[I]
end;

procedure ArrayAdd(const A, B, C: TArray<Nfloat>); overload;
begin
  Assert(Assigned(A) and Assigned(B) and Assigned(C));
  ArrayAdd(Min(Min(Length(A), Length(B)), Length(C)), Pointer(A), Pointer(B), Pointer(C))
end;

procedure ArrayAxpy(const N: LongWord; const A: Nfloat; const X, Y: Pnfloat);
var
  I: LongWord;
begin
  Assert(Assigned(X) and Assigned(Y));
  for I := 0 to N - 1 do
    Y[I] := A * X[I] + Y[I]
end;

procedure ArrayPrint(const N: LongWord; const A: Pnfloat; const Decimals: Word = 2); overload;
var
  S: string;
  Sv: ShortString;
  // i,
  W: Integer;
  R, C, Rows, Cols: Integer;
begin
  if not Assigned(A) or (N = 0) then
    Exit;
  Rows := Round(Sqrt(N));
  Cols := N div LongWord(Rows);
  W := Length(IntToStr(Round(ArrayMax(N, A))));
  // str(single(a[0]):w:decimals, sv);
  // s := intToStr(0)+' : '+sv+slineBreak;
  S := '';
  for R := 0 to Rows - 1 do
  begin
    for C := 0 to Cols - 1 do
    begin
      Str(Single(A[R * Cols + C]): W + Decimals + 1: Decimals, Sv);
      if C > 0 then
        S := S + ',' + string(Sv)
      else
        S := S + string(Sv);
    end;
    WriteLn(S);
    S := ''
  end;
  // writeln(s);
end;

procedure ArrayPrint(const A: TArray<Nfloat>; const Decimals: Word = 2); overload;
begin
  ArrayPrint(Length(A), Pointer(A), Decimals)
end;

procedure ArrayStat(const N: LongWord; const A: Pnfloat; const Decimals: Word = 2); overload;
var
  I: LongWord;
  Mih, Mah: Nfloat;
  Mi, Ma, Mean, Variance: Single;
begin
  ArrayMinMax(N, A, @Mih, @Mah);
  Mi := Mih;
  Ma := Mah;
  Mean := ArraySum(N, A) / N;
  Variance := ArrayVariance(N, Mean, A);
  I := Max(Length(IntToStr(Round(Mi))), Length(IntToStr(Round(Ma))));
  I := Max(I, Max(Length(IntToStr(Round(Mean))), Length(IntToStr(Round(Variance)))));
  WriteLn(
    '[',
    N,
    '] mean :',
    Mean:
    I:
    Decimals,
    ', stdDev :',
    Sqrt(Variance):
    I:
    Decimals,
    ', min/max :',
    Mi:
    I:
    Decimals,
    '/',
    Ma: I: Decimals
  );
end;

procedure ArrayStat(const A: TArray<Nfloat>; const Decimals: Word = 2); overload;
begin
  ArrayStat(Length(A), Pointer(A), Decimals)
end;

function ArraySumSqrDiff(const N: LongWord; const A, B: Pnfloat): Single; overload;
var
  I: LongWord;
begin
  Assert(Assigned(A) and Assigned(B));
  Result := 0;
  for I := 0 to N - 1 do
    Result := Result + Sqr(Single(A[I] - B[I]));
end;

function ArraySumSqrDiff(const A, B: TArray<Nfloat>): Single; overload;
begin
  Result := ArraySumSqrDiff(Length(A), Pointer(A), Pointer(B))
end;

// function abs(const x:half):half;inline;
// begin
//  result := abs(single(x));
// end;

procedure ArrayPrintDiff(const N: LongWord; const A, B: Pnfloat; const Eps: Single = 0.000001); overload;
const
  EPSILON = 0.0001;
var
  I: LongWord;
  Diff, Err, ADiff, Aa: Single;
begin
  for I := 0 to N - 1 do
  begin
    Diff := A[I] - B[I];
    ADiff := Abs(Diff);
    Aa := Abs(A[I]);
    if (Aa < EPSILON) and (ADiff < EPSILON) then
      Continue;
    if Aa > 0 then
      Err := ADiff / Aa
    else
      Err := ADiff;
    if Abs(Err) > Eps then
      WriteLn('[', I, ']: ', Single(A[I]): 1: 6, ' / ', Single(B[I]): 1: 6);
  end;
end;

procedure ArrayPrintDiff(const A, B: TArray<Nfloat>; const Eps: Single = 0.000001); overload;
begin
  ArrayPrintDiff(Min(Length(A), Length(B)), Pointer(A), Pointer(B), Eps);
end;

procedure ArrayFma(const N: LongWord; const A, B: Nfloat; const X, Y: Pnfloat); overload;
var
  I: LongInt;
begin
  Assert(Assigned(X) and Assigned(Y));
  for I := 0 to N - 1 do
    Y[I] := A * X[I] + B;
end;

procedure ArrayFma(const A, B: Nfloat; const X, Y: TArray<Nfloat>); overload;
begin
  ArrayFma(Length(X), A, B, Pointer(X), Pointer(Y))
end;

{$IFDEF CPUX64}

procedure SaxpyAVX2(const N: LongWord; const A: Single; const X, Y: System.PSingle); assembler;
asm

//.NOSTACK

        Mov     Eax, N
        AND     Eax, 31
        Vbroadcastss Ymm1, A
        SHR     N, 5
        Jz      @Skip1

@While1:
        Vmulps  Ymm0, Ymm1, Yword[X]
        Vaddps  Ymm0, Ymm0, Yword[Y]
        Vmulps  Ymm2, Ymm1, Yword[X + 32]
        Vaddps  Ymm2, Ymm2, Yword[Y + 32]
        Vmulps  Ymm3, Ymm1, Yword[X + 32 * 2]
        Vaddps  Ymm3, Ymm3, Yword[Y + 32 * 2]
        Vmulps  Ymm4, Ymm1, Yword[X + 32 * 3]
        Vaddps  Ymm4, Ymm4, Yword[Y + 32 * 3]
        Vmovups Yword[Y], Ymm0
        Vmovups Yword[Y + 32], Ymm2
        Vmovups Yword[Y + 32 * 2], Ymm3
        Vmovups Yword[Y + 32 * 3], Ymm4
        Add     X, 4 * 8 * SizeOf(Single)
        Add     Y, 4 * 8 * SizeOf(Single)
        Dec     N
        Jnz     @While1

@Skip1:
        Mov     N, Eax
        SHR     N, 3
        Jz      @Skip2

@While2:
        Vmulps  Ymm0, Ymm1, Yword[X]
        Vaddps  Ymm0, Ymm0, Yword[Y]
        Vmovups Yword[Y], Ymm0
        Add     X, 8 * SizeOf(Single)
        Add     Y, 8 * SizeOf(Single)
        Dec     N
        Jnz     @While2

@Skip2:
        AND     Eax, 7
        Jz      @Done

@While3:
        Vmulss  Xmm0, Xmm1, Dword[X]
        Vaddss  Xmm0, Xmm0, Dword[Y]
        Movss   Dword[Y], Xmm0
        Add     X, SizeOf(Single)
        Add     Y, SizeOf(Single)
        Dec     Eax
        Jnz     @While3

@Done:
end;

procedure HaxpyAVX2(const N: LongWord; const A: Single; const X, Y: PHalf); assembler;
const
  RND = 0;
var
  AR: packed array[0..7] of Word;
asm

//.NOSTACK

        Mov     Eax, N
        AND     Eax, 31
        Vbroadcastss Ymm1, A
  //vcvtsh2ss    ymm1,  a
        SHR     N, 5
        Jz      @Skip1

@While1:
// intels F16C instructions, using OP cods instead
  //vcvtph2ps    ymm0, [x]
  //vcvtph2ps    ymm5, [y]
        Db      $C4, $C2, $7D, $13, $00       // vcvtph2ps ymm0, [x]
        Db      $C4, $C2, $7D, $13, $29       // vcvtph2ps ymm5, [y]
        Vmulps  Ymm0, Ymm1, Ymm0
        Vaddps  Ymm0, Ymm0, Ymm5
  //vcvtph2ps    ymm2, [x+16]
  //vcvtph2ps    ymm5, [y+16]
        Db      $C4, $C2, $7D, $13, $50, $10  // vcvtph2ps ymm2, [x+$10]
        Db      $C4, $C2, $7D, $13, $69, $10  // vcvtph2ps ymm5, [y+$10]
        Vmulps  Ymm2, Ymm1, Ymm2
        Vaddps  Ymm2, Ymm2, Ymm5
  //vcvtph2ps    ymm3, [x+2*16]
  //vcvtph2ps    ymm5, [y+2*16]
        Db      $C4, $C2, $7D, $13, $58, $20  // vcvtph2ps ymm3, [x+$20]
        Db      $C4, $C2, $7D, $13, $69, $20  // vcvtph2ps ymm5, [y+$20]
        Vmulps  Ymm3, Ymm1, Ymm3
        Vaddps  Ymm3, Ymm3, Ymm5
  //vcvtph2ps    ymm4, [x+3*16]
  //vcvtph2ps    ymm5, [y+3*16]
        Db      $C4, $C2, $7D, $13, $60, $30  // vcvtph2ps ymm4, [x+$30]
        Db      $C4, $C2, $7D, $13, $69, $30  // vcvtph2ps ymm5, [y+$30]
        Vmulps  Ymm4, Ymm1, Ymm4
        Vaddps  Ymm4, Ymm4, Ymm5
  //vcvtps2ph    [Y]     , ymm0, RND
  //vcvtps2ph    [Y+16]  , ymm2, RND
  //vcvtps2ph    [Y+16*2], ymm3, RND
  //vcvtps2ph    [Y+16*3], ymm4, RND
        Db      $C4, $C3, $7D, $1D, $01, RND      //  vcvtps2ph [y]    , ymm0, RND
        Db      $C4, $C3, $7D, $1D, $51, $10, RND //  vcvtps2ph [y+$10], ymm2, RND
        Db      $C4, $C3, $7D, $1D, $59, $20, RND //  vcvtps2ph [y+$20], ymm3, RND
        Db      $C4, $C3, $7D, $1D, $61, $30, RND //  vcvtps2ph [y+$30], ymm4, RND
        Add     X, 4 * 8 * SizeOf(Half)
        Add     Y, 4 * 8 * SizeOf(Half)
        Dec     N
        Jnz     @While1

@Skip1:
        Mov     N, Eax
        SHR     N, 3
        Jz      @Skip2

@While2:
  //vcvtph2ps    ymm0, [x]
  //vcvtph2ps    ymm5, [y]
        Db      $C4, $C2, $7D, $13, $00           // vcvtph2ps ymm0, [x]
        Db      $C4, $C2, $7D, $13, $29           // vcvtph2ps ymm5, [y]
        Vmulps  Ymm0, Ymm1, Ymm0
        Vaddps  Ymm0, Ymm0, Ymm5
  //vcvtps2ph    [Y], ymm0, RND
        Db      $C4, $C3, $7D, $1D, $01, RND      //  vcvtps2ph [y]    , ymm0, RND
        Add     X, 8 * SizeOf(Half)
        Add     Y, 8 * SizeOf(Half)
        Dec     N
        Jnz     @While2

@Skip2:
        AND     Eax, 7
        Jz      @Done
  //vcvtph2ps    ymm0, [x]
  //vcvtph2ps    ymm5, [y]
        Db      $C4, $C2, $7D, $13, $00           // vcvtph2ps ymm0, [x]
        Db      $C4, $C2, $7D, $13, $29           // vcvtph2ps ymm5, [y]
        Vmulps  Ymm0, Ymm1, Ymm0
        Vaddps  Ymm0, Ymm0, Ymm5
  //vcvtps2ph    ar  , ymm0, RND       // storing to the stack
        Db      $C4, $E3, $7D, $1D, $45, $F0, RND //  vcvtps2ph [rbp-$10], ymm0, RND

@While3:
        Dec     Eax
        Mov     Cx, Word[AR + Rax * 2]
        Mov     Word[Y + Rax * 2], Cx
        Jnz     @While3

@Done:
end;

{$ENDIF}

type
  TBlasInt = LongInt;

  TCBLASLayout = (CblasRowMajor = 101, CblasColMajor = 102);
// {$else}
// const
//  CblasRowMajor = CBLAS_Layout.CblasRowMajor;
//  CblasColMajor = CBLAS_Layout.CblasColMajor;
// {$endif}

// {$if not declared(CBLAS_ORDER)}

type
  TCBLASOrder = TCBLASLayout;
// {$endif}
// {$if not declared(CBLAS_TRANSPOSE)}

type
  TCBLASTranspose = (CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113, CblasConjNoTrans = 114);

{$IF not defined(cblas_sgemm)}
procedure CblasSgemm(Order: TCBLASOrder; TransA: TCBLASTranspose; TransB: TCBLASTranspose; M: TBlasInt; N: TBlasInt; K: TBlasInt; Alpha: Single; A: System.PSingle; Lda: TBlasInt; B: System.PSingle; Ldb: TBlasInt; Beta: Single; C: System.PSingle; Ldc: TBlasInt); winapi; external 'libopenblas.dll' name 'cblas_sgemm';
{$ENDIF}

{$T+}

procedure ArrayGemmNN(const M, N, K: LongWord; const Alpha: Single; const A: TArray<Nfloat>; const Lda: LongWord; const B: TArray<Nfloat>; const Ldb: LongWord; const Beta: Single; const C: TArray<Nfloat>; const Ldc: LongWord); overload;
begin
  CblasSgemm(
    CblasRowMajor,
    CblasNoTrans,
    CblasNoTrans,
    M,
    N,
    K,
    Alpha,
    Pnfloat(A),
    Lda,
    Pnfloat(B),
    Ldb,
    Beta,
    Pnfloat(C),
    Ldc
  );
end;

procedure ArrayGemmNT(const M, N, K: LongWord; const Alpha: Single; const A: TArray<Nfloat>; const Lda: LongWord; const B: TArray<Nfloat>; const Ldb: LongWord; const Beta: Single; const C: TArray<Nfloat>; const Ldc: LongWord); overload;
begin
  CblasSgemm(
    CblasRowMajor,
    CblasNoTrans,
    CblasTrans,
    M,
    N,
    K,
    Alpha,
    Pnfloat(A),
    Lda,
    Pnfloat(B),
    Ldb,
    Beta,
    Pnfloat(C),
    Ldc
  )
end;

procedure ArrayGemmTN(const M, N, K: LongWord; const Alpha: Single; const A: TArray<Nfloat>; const Lda: LongWord; const B: TArray<Nfloat>; const Ldb: LongWord; const Beta: Single; const C: TArray<Nfloat>; const Ldc: LongWord); overload;
begin
  CblasSgemm(
    CblasRowMajor,
    CblasTrans,
    CblasNoTrans,
    M,
    N,
    K,
    Alpha,
    Pnfloat(A),
    Lda,
    Pnfloat(B),
    Ldb,
    Beta,
    Pnfloat(C),
    Ldc
  )
end;
{$T-}

initialization
  // Initialize only when the optional backend is requested. CPU startup never
  // loads a driver or changes the application's floating point exception mask.



finalization
  if TVulkanCompute.Instance <> nil then
    TVulkanCompute.DestroyInstance();

end.

