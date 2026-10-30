unit QuickNN.Vulkan;

{$POINTERMATH ON}

interface

uses
  System.SysUtils, System.Math, QuickNN.Common, QuickNN.Quantization.Int4
{$IF defined(MSWINDOWS) and defined(CPUX64)}
    , QuickNN.Vulkan.Helpers, System.Generics.Collections
{$ENDIF}
;

type
  TQNNGPUMode = (gpuOff, gpuAuto, gpuForce);

  TQNNGPUStatus = record
    Available: Boolean;
    DeviceName, LastError: string;
    GemmCalls, BF16Calls, AllocatedBytes, AttentionCalls: UInt64;
    TensorCalls, ConvRotCalls, Int4Calls, ResidentPeakBytes: UInt64;
    WeightCacheBytes, WeightCacheLimit, WeightCacheHits, WeightCacheMisses: UInt64;
    WeightUploads, WeightUploadBytes, WeightEvictions: UInt64;
    PipelineCalls, Submissions, Readbacks: UInt64;
    ActivationUploadBytes, ActivationDownloadBytes: UInt64;
  end;

// Auto routes sufficiently large products to Vulkan. Force also routes small
// products (useful for parity tests); unavailable/unsupported work stays on CPU.
procedure QNNConfigureGPU(const Mode: TQNNGPUMode; const DeviceIndex: Integer = -1);

function QNNGetGPUStatus: TQNNGPUStatus;

function TryVulkanGemm(const Order: TCBLASOrder; const TransA, TransB: TCBLASTranspose; const M, N, K: Int64; const Alpha: Single; const A: System.PSingle; const Lda: Int64; const B: System.PSingle; const Ldb: Int64; const Beta: Single; const C: System.PSingle; const Ldc: Int64): Boolean;

function TryVulkanLinearBF16(const Dst, Src: System.PSingle; const Weight: PBF16; const Seq, SrcDim, DstDim: Integer): Boolean;

// Only explicitly immutable weights are retained. Call invalidate after changing
// their contents; cache keys never depend on a potentially reused CPU address.
procedure QNNSetGPUWeightCacheLimit(const Bytes: UInt64);

procedure QNNGPUClearWeightCache;

procedure QNNInvalidateGPUWeight(const Weight: TMemoryBlock);

function TryVulkanLinear(const Dst, Src: System.PSingle; const Weight: TMemoryBlock; const Seq, InputDim, OutputDim: Integer): Boolean;

function TryVulkanAttention(const Dst, Q, K, V: TMemoryBlock; const Rows, Heads, KvHeads, HeadDim, CausalRows: Integer; const KeyLimits: TArray<Integer> = nil
    ): Boolean;

function TryVulkanInt8Linear(const Dst, Src: System.PSingle; const Weight, Scales: TMemoryBlock; const Seq, InputDim, OutputDim: Integer): Boolean;

function TryVulkanSwiGLU(const Dst, Src, Gate, Up, Down: TMemoryBlock; const Seq, Hidden, Mlp: Integer): Boolean;

type
  TQNNGPUContext = class;
{$IF defined(MSWINDOWS) and defined(CPUX64)}

  TQNNVulkan = class;

  TGPUWeight = class
    Owner: TQNNVulkan;
    Identity, ScaleIdentity: IInterface;
    Memory: TVulkanMemory;
    LastUse: UInt64;
    Pins: Integer;
    destructor Destroy; override;
  end;

  TQNNVulkan = class(TVulkanCompute)
  private
    FInputA, FInputB, FOutput: TVulkanMemory;
    FAttentionAvailable: Boolean;
    FResident: TQNNGPUContext;
    FUploadA, FUploadB, FReadback: TVulkanMemory;
    FGraphGate, FGraphUp, FWeightUpload: TVulkanMemory;
    FWeights: TDictionary<string, TGPUWeight>;
    FWeightBytes, FWeightLimit, FPhysicalWeightLimit, FWeightClock: UInt64;
    FCounters: TQNNGPUStatus;
    function WorkingBytes: UInt64;
    procedure GrowWorking(var Buffer: TVulkanMemory; const Size: UInt64; const Staging, Cached: Boolean);
    procedure ReleaseWorking;
    procedure PrepareNetwork(const Seq, InputDim, OutputDim, MlpDim: Integer);
    function EvictOne: Boolean; overload;
    function EvictOne(const ReuseBytes: UInt64; out Reused: TVulkanMemory): Boolean; overload;
    function AcquireInt8Weight(const Weight, Scales: TMemoryBlock; const Rows, Cols: Integer): TGPUWeight;
    function WeightKey(const Weight: TMemoryBlock; const Rows, Cols: Integer): string;
    function AcquireWeight(const Weight: TMemoryBlock; const Rows, Cols: Integer; const AllowPackedInt4: Boolean = False
    ): TGPUWeight;
    procedure RecordLinear(const A, B, C: TVulkanMemory; const Seq, InputDim, OutputDim: Integer; const BF16: Boolean; const Slot: Integer);
    procedure BeginNetwork(const Src: System.PSingle; const Seq, Dim: Integer);
    procedure EndNetwork(const Dst: System.PSingle; const Seq, Dim: Integer);
    procedure EnsureBuffers(const ABytes, BBytes, CBytes: UInt64);
    function MatrixBytes(const Rows, Cols, Stride, ElementSize: Int64): UInt64;
    procedure Upload(const Source: Pointer; const Staging: TVulkanMemory; const Size, PaddedSize: UInt64);
  public
    constructor Create(const DeviceIndex: Integer = -1);
    destructor Destroy; override;
    function DeviceName: string;
    function AllocatedBytes: UInt64;
    procedure SetWeightLimit(const Limit: UInt64);
    procedure ClearWeights;
    procedure InvalidateWeight(const Identity: IInterface);
    procedure CopyCounters(var Status: TQNNGPUStatus);
    procedure Attention(const Dst, Q, K, V: TMemoryBlock; const Rows, Heads, KvHeads, HeadDim, CausalRows: Integer; const KeyLimits: TArray<Integer>);
    procedure CachedInt8Linear(const Dst, Src: System.PSingle; const Weight, Scales: TMemoryBlock; const Seq, InputDim, OutputDim: Integer);
    procedure CachedLinear(const Dst, Src: System.PSingle; const Weight: TMemoryBlock; const Seq, InputDim, OutputDim: Integer);
    procedure Swiglu(const Dst, Src: System.PSingle; const Gate, Up, Down: TMemoryBlock; const Seq, Hidden, Mlp: Integer);
    procedure Sgemm(const TransA, TransB: Boolean; const M, N, K: Integer; const Alpha: Single; const A: System.PSingle; const Lda: Integer; const B: Pointer; const Ldb: Integer; const Beta: Single; const C: System.PSingle; const Ldc: Integer; const BIsBF16: Boolean = False
    );
    // Explicit resident-buffer path for later layer chaining. Synchronous: each
    // dispatch owns its descriptor update and completes before this method returns.
    procedure Gemm(const TransA, TransB: Boolean; const M, N, K: Integer; const Alpha: Single; const A: TVulkanMemory; const Lda: Integer; const B: TVulkanMemory; const Ldb: Integer; const Beta: Single; const C: TVulkanMemory; const Ldc, OffsetA, OffsetB, OffsetC: Integer; const BIsBF16: Boolean = False
    );
  end;
{$ENDIF}

{$I QuickNN.Vulkan.Context.API.inc}

implementation

uses
  System.SyncObjs
{$IF defined(MSWINDOWS) and defined(CPUX64)}
    , QuickNN.Vulkan.API, Winapi.Windows
{$ENDIF}
;

const
  RESIDENT_DISPATCH_SLOTS = 1024;
  AUTO_MIN_PRODUCTS = 64 * 1024 * 1024;
  WEIGHT_UPLOAD_CHUNK: UInt64 = 16 * 1024 * 1024;
  // Device buffers plus coherent staging: cap transient allocations at 1 GiB.
  MAX_WORKING_BYTES: UInt64 = 1024 * 1024 * 1024;

var
  GPULock: TCriticalSection;
  GPUMode: TQNNGPUMode = gpuAuto;
  GPUDeviceIndex: Integer = -1;
  GPUWeightLimit: UInt64 = UInt64(8) * 1024 * 1024 * 1024;
  GPUAttempted: Boolean;
  GPUStatus: TQNNGPUStatus;
{$IF defined(MSWINDOWS) and defined(CPUX64)}
  GPU: TQNNVulkan;

{$I generated/QuickNN.Vulkan.Gemm.SPIRV.inc}
{$I generated/QuickNN.Vulkan.Swiglu.SPIRV.inc}
{$I generated/QuickNN.Vulkan.Attention.SPIRV.inc}
{$I generated/QuickNN.Vulkan.Tensor.SPIRV.inc}
{$I generated/QuickNN.Vulkan.Moe.SPIRV.inc}

destructor TGPUWeight.Destroy;
begin
  if Owner <> nil then
    Owner.FreeMemory(Memory);
  inherited;
end;

constructor TQNNVulkan.Create(const DeviceIndex: Integer);
var
  Workload: TWorkloadSizes;
  PreviousMask: TFPUExceptionMask;
  I: Integer;
  Heap: UInt64;
begin
  PreviousMask := GetExceptionMask;
  SetExceptionMask([ExInvalidOp, ExDenormalized, ExZeroDivide, ExOverflow, ExUnderflow, ExPrecision]);
  try
    inherited Create(DeviceIndex);
    if (VulkanDevice.DeviceProperties.Limits.MaxComputeWorkGroupInvocations < 256)
      or (VulkanDevice.DeviceProperties.Limits.MaxComputeWorkGroupSize[0] < 16)
      or (VulkanDevice.DeviceProperties.Limits.MaxComputeWorkGroupSize[1] < 16)
      or (VulkanDevice.DeviceProperties.Limits.MaxComputeSharedMemorySize < 16384) then
      raise Exception.Create('Vulkan device does not support the GEMM tile');
    FWeights := TDictionary<string, TGPUWeight>.Create;
    for I := 0 to VulkanDevice.MemoryProperties.MemoryHeapCount - 1 do
      if (VulkanDevice.MemoryProperties.MemoryHeaps[I].Flags and LongWord(VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)) <> 0 then
      begin
        Heap := VulkanDevice.MemoryProperties.MemoryHeaps[I].Size;
        FPhysicalWeightLimit := Max(FPhysicalWeightLimit, Heap div 4 * 3);
      end;
    SetWeightLimit(GPUWeightLimit);
    Workload.X := 16;
    Workload.Y := 16;
    Workload.Z := 1;
    PreparePipeline(
      TKernelHandle(0),
      CreateShader(@QNN_GEMM_SPIRV[0], SizeOf(QNN_GEMM_SPIRV)),
      Workload,
      3,
      RESIDENT_DISPATCH_SLOTS
    );
    Workload.X := 128;
    Workload.Y := 1;
    PreparePipeline(TKernelHandle(1), CreateShader(@QNN_SWIGLU_SPIRV[0], SizeOf(QNN_SWIGLU_SPIRV)), Workload, 3);
    FAttentionAvailable := VulkanDevice.DeviceProperties.Limits.MaxComputeSharedMemorySize >= 32768;
    if FAttentionAvailable then
      PreparePipeline(
        TKernelHandle(2),
        CreateShader(@QNN_ATTENTION_SPIRV[0], SizeOf(QNN_ATTENTION_SPIRV)),
        Workload,
        3,
        RESIDENT_DISPATCH_SLOTS
      );
    Workload.X := 256;
    PreparePipeline(
      TKernelHandle(3),
      CreateShader(@QNN_TENSOR_SPIRV[0], SizeOf(QNN_TENSOR_SPIRV)),
      Workload,
      4,
      RESIDENT_DISPATCH_SLOTS
    );
    PreparePipeline(
      TKernelHandle(4),
      CreateShader(@QNN_MOE_SPIRV[0], SizeOf(QNN_MOE_SPIRV)),
      Workload,
      4,
      RESIDENT_DISPATCH_SLOTS
    );
  finally
    SetExceptionMask(PreviousMask);
  end;
end;

destructor TQNNVulkan.Destroy;
begin
  if Device <> nil then
    VkDeviceWaitIdle(Device);
  if FWeights <> nil then
    ClearWeights;
  FWeights.Free;
  ReleaseWorking;
  inherited;
end;

function TQNNVulkan.DeviceName: string;
begin
  Result := string(PAnsiChar(@VulkanDevice.DeviceProperties.DeviceName[0]));
end;

function TQNNVulkan.AllocatedBytes: UInt64;
begin
  Result := WorkingBytes + FWeightBytes;
  if FResident <> nil then
    Inc(Result, FResident.AllocatedBytes);
end;

function TQNNVulkan.MatrixBytes(const Rows, Cols, Stride, ElementSize: Int64): UInt64;
var
  Elements: Int64;
begin
  if (Rows <= 0) or (Cols <= 0) or (Stride < Cols) then
    raise ERangeError.Create('Invalid Vulkan matrix dimensions/stride');
  Elements := (Rows - 1) * Stride + Cols;
  if Elements > High(Integer) div ElementSize then
    raise ERangeError.Create('Vulkan matrix exceeds supported indexing range');
  Result := Elements * ElementSize;
  if Result > VulkanDevice.DeviceProperties.Limits.MaxStorageBufferRange then
    raise ERangeError.Create('Vulkan matrix exceeds maxStorageBufferRange');
end;

procedure TQNNVulkan.EnsureBuffers(const ABytes, BBytes, CBytes: UInt64);
var
  Proposed: UInt64;
begin
  Proposed :=
    2 * (Max(ABytes, UInt64(FInputA.Size)) + Max(BBytes, UInt64(FInputB.Size)) + Max(CBytes, UInt64(FOutput.Size)))
    + FGraphGate.Size
    + FGraphUp.Size
    + FWeightUpload.Size;
  if Proposed > MAX_WORKING_BYTES then
    ReleaseWorking;
  GrowWorking(FInputA, ABytes, False, False);
  GrowWorking(FUploadA, ABytes, True, False);
  GrowWorking(FInputB, BBytes, False, False);
  GrowWorking(FUploadB, BBytes, True, False);
  GrowWorking(FOutput, CBytes, False, False);
  GrowWorking(FReadback, CBytes, True, True);
end;

procedure TQNNVulkan.Upload(const Source: Pointer; const Staging: TVulkanMemory; const Size, PaddedSize: UInt64);
var
  Target: Pointer;
begin
  Target := MapMemory(Staging);
  try
    Move(Source^, Target^, Size);
    if PaddedSize > Size then
      FillChar((PByte(Target) + Size)^, PaddedSize - Size, 0);
  finally
    UnMapMemory(Staging);
  end;
end;

procedure TQNNVulkan.Sgemm(const TransA, TransB: Boolean; const M, N, K: Integer; const Alpha: Single; const A: System.PSingle; const Lda: Integer; const B: Pointer; const Ldb: Integer; const Beta: Single; const C: System.PSingle; const Ldc: Integer; const BIsBF16: Boolean);
var
  ABytes, BBytes, BPadded, CBytes: UInt64;
  Row, Flags: Integer;
  Mapped: System.PSingle;
  PreviousMask: TFPUExceptionMask;
begin
  if (M <= 0) or (N <= 0) or (K <= 0) or (M > High(Integer) - 63) or (N > High(Integer) - 63) then
    raise ERangeError.Create('Invalid Vulkan GEMM dimensions');
  if (A = nil) or (B = nil) or (C = nil) or (Ldc < N) then
    raise EArgumentException.Create('Invalid Vulkan GEMM pointer or output stride');
  if TransA then
    ABytes := MatrixBytes(K, M, Lda, 4)
  else
    ABytes := MatrixBytes(M, K, Lda, 4);
  if TransB then
    BBytes := MatrixBytes(N, K, Ldb, 4 - 2 * Ord(BIsBF16))
  else
    BBytes := MatrixBytes(K, N, Ldb, 4 - 2 * Ord(BIsBF16));
  BPadded := (BBytes + 3) and not UInt64(3);
  CBytes := MatrixBytes(M, N, N, 4);
  MatrixBytes(M, N, Ldc, 4); // validate host output indexing too
  if (UInt64(N + 63) div 64 > VulkanDevice.DeviceProperties.Limits.MaxComputeWorkGroupCount[0])
    or (UInt64(M + 63) div 64 > VulkanDevice.DeviceProperties.Limits.MaxComputeWorkGroupCount[1]) then
    raise ERangeError.Create('Vulkan GEMM workgroup count exceeds device limit');
  EnsureBuffers(ABytes, BPadded, CBytes);
  PreviousMask := GetExceptionMask;
  SetExceptionMask([ExInvalidOp, ExDenormalized, ExZeroDivide, ExOverflow, ExUnderflow, ExPrecision]);
  try
    Upload(A, FUploadA, ABytes, ABytes);
    Upload(B, FUploadB, BBytes, BPadded);
    if Beta <> 0 then
    begin
      Mapped := MapMemory(FReadback);
      try
        for Row := 0 to M - 1 do
          Move(C[Int64(Row) * Ldc], Mapped[Int64(Row) * N], NativeInt(N) * 4);
      finally
        UnMapMemory(FReadback)
      end;
    end;
    Flags := Ord(TransA) or (Ord(TransB) shl 1) or (Ord(BIsBF16) shl 2);
    BeginCommadBuffer;
    CopyBuffer(FUploadA, FInputA, ABytes);
    CopyBuffer(FUploadB, FInputB, BPadded);
    if Beta <> 0 then
      CopyBuffer(FReadback, FOutput, CBytes);
    MemoryBarrier(
      LongWord(VK_PIPELINE_STAGE_TRANSFER_BIT),
      LongWord(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT),
      LongWord(VK_ACCESS_TRANSFER_WRITE_BIT),
      LongWord(VK_ACCESS_SHADER_READ_BIT) or LongWord(VK_ACCESS_SHADER_WRITE_BIT)
    );
    DispatchPipeline(
      TKernelHandle(0),
      [M, N, K, Alpha, FInputA.Buffer, 0, Lda, FInputB.Buffer, 0, Ldb, Beta, FOutput.Buffer, 0, N, Flags],
      (N + 63) div 64,
      (M + 63) div 64
    );
    MemoryBarrier(
      LongWord(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT),
      LongWord(VK_PIPELINE_STAGE_TRANSFER_BIT),
      LongWord(VK_ACCESS_SHADER_WRITE_BIT),
      LongWord(VK_ACCESS_TRANSFER_READ_BIT)
    );
    CopyBuffer(FOutput, FReadback, CBytes);
    MemoryBarrier(
      LongWord(VK_PIPELINE_STAGE_TRANSFER_BIT),
      LongWord(VK_PIPELINE_STAGE_HOST_BIT),
      LongWord(VK_ACCESS_TRANSFER_WRITE_BIT),
      LongWord(VK_ACCESS_HOST_READ_BIT)
    );
    EndCommandBuffer;
    Finish;
    Inc(FCounters.Submissions);
    Inc(FCounters.ActivationUploadBytes, ABytes + BBytes);
    if Beta <> 0 then
      Inc(FCounters.ActivationUploadBytes, CBytes);
    Inc(FCounters.Readbacks);
    Inc(FCounters.ActivationDownloadBytes, CBytes);
    // C padding is never touched, and beta=0 never uploads/reads old C.
    Mapped := MapMemory(FReadback);
    try
      for Row := 0 to M - 1 do
        Move(Mapped[Int64(Row) * N], C[Int64(Row) * Ldc], NativeInt(N) * 4);
    finally
      UnMapMemory(FReadback)
    end;
  finally
    SetExceptionMask(PreviousMask)
  end;
end;

procedure TQNNVulkan.Gemm(const TransA, TransB: Boolean; const M, N, K: Integer; const Alpha: Single; const A: TVulkanMemory; const Lda: Integer; const B: TVulkanMemory; const Ldb: Integer; const Beta: Single; const C: TVulkanMemory; const Ldc, OffsetA, OffsetB, OffsetC: Integer; const BIsBF16: Boolean);
var
  ABytes, BBytes, CBytes: UInt64;
  Flags: Integer;
begin
  if (M <= 0) or (N <= 0) or (K <= 0) or (M > High(Integer) - 63) or (N > High(Integer) - 63) then
    raise ERangeError.Create('Invalid resident GEMM dimensions');
  if (OffsetA < 0) or (OffsetB < 0) or (OffsetC < 0) then
    raise ERangeError.Create('Negative Vulkan offset');
  if TransA then
    ABytes := MatrixBytes(K, M, Lda, 4)
  else
    ABytes := MatrixBytes(M, K, Lda, 4);
  if TransB then
    BBytes := MatrixBytes(N, K, Ldb, 4 - 2 * Ord(BIsBF16))
  else
    BBytes := MatrixBytes(K, N, Ldb, 4 - 2 * Ord(BIsBF16));
  CBytes := MatrixBytes(M, N, Ldc, 4);
  if (UInt64(N + 63) div 64 > VulkanDevice.DeviceProperties.Limits.MaxComputeWorkGroupCount[0])
    or (UInt64(M + 63) div 64 > VulkanDevice.DeviceProperties.Limits.MaxComputeWorkGroupCount[1]) then
    raise ERangeError.Create('Resident GEMM workgroup count exceeds device limit');
  if (ABytes + UInt64(OffsetA) * 4 > A.Size)
    or (((BBytes + UInt64(OffsetB) * UInt64(4 - 2 * Ord(BIsBF16)) + 3) and not UInt64(3)) > B.Size)
    or (CBytes + UInt64(OffsetC) * 4 > C.Size) then
    raise ERangeError.Create('Vulkan GEMM buffer too small');
  if (A.Buffer = nil)
    or (B.Buffer = nil)
    or (C.Buffer = nil)
    or (A.Size > VulkanDevice.DeviceProperties.Limits.MaxStorageBufferRange)
    or (B.Size > VulkanDevice.DeviceProperties.Limits.MaxStorageBufferRange)
    or (C.Size > VulkanDevice.DeviceProperties.Limits.MaxStorageBufferRange) then
    raise ERangeError.Create('Invalid resident buffer or storage range');
  if (C.Buffer = A.Buffer) or (C.Buffer = B.Buffer) then
    raise EArgumentException.Create('Resident GEMM output must not alias its inputs');
  if CommandBufferStarted then
    raise Exception.Create('Resident GEMM requires an idle command buffer');
  Flags := Ord(TransA) or (Ord(TransB) shl 1) or (Ord(BIsBF16) shl 2);
  BeginCommadBuffer;
  MemoryBarrier(
    LongWord(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT),
    LongWord(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT),
    LongWord(VK_ACCESS_MEMORY_WRITE_BIT),
    LongWord(VK_ACCESS_SHADER_READ_BIT) or LongWord(VK_ACCESS_SHADER_WRITE_BIT)
  );
  DispatchPipeline(
    TKernelHandle(0),
    [M, N, K, Alpha, A.Buffer, OffsetA, Lda, B.Buffer, OffsetB, Ldb, Beta, C.Buffer, OffsetC, Ldc, Flags],
    (N + 63) div 64,
    (M + 63) div 64
  );
  MemoryBarrier(
    LongWord(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT),
    LongWord(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT),
    LongWord(VK_ACCESS_SHADER_WRITE_BIT),
    LongWord(VK_ACCESS_MEMORY_READ_BIT)
  );
  EndCommandBuffer;
  Finish;
end;

function TQNNVulkan.WorkingBytes: UInt64;
begin
  Result :=
    FInputA.Size
    + FInputB.Size
    + FOutput.Size
    + FUploadA.Size
    + FUploadB.Size
    + FReadback.Size
    + FGraphGate.Size
    + FGraphUp.Size
    + FWeightUpload.Size;
end;

procedure TQNNVulkan.GrowWorking(var Buffer: TVulkanMemory; const Size: UInt64; const Staging, Cached: Boolean);
begin
  if Buffer.Size >= Size then
    Exit;
  if WorkingBytes - Buffer.Size + Size > MAX_WORKING_BYTES then
    raise ERangeError.Create('Vulkan working memory budget exceeded');
  FreeMemory(Buffer);
  Buffer := Default(TVulkanMemory);
  if Staging then
    Buffer := CreateStagingMemory(Size, Cached)
  else
    Buffer := CreateStorageMemory(Size);
end;

procedure TQNNVulkan.ReleaseWorking;

  procedure Release(var Buffer: TVulkanMemory);
  begin
    FreeMemory(Buffer);
    Buffer := Default(TVulkanMemory)
  end;

begin
  Release(FInputA);
  Release(FInputB);
  Release(FOutput);
  Release(FUploadA);
  Release(FUploadB);
  Release(FReadback);
  Release(FGraphGate);
  Release(FGraphUp);
  Release(FWeightUpload);
end;

procedure TQNNVulkan.PrepareNetwork(const Seq, InputDim, OutputDim, MlpDim: Integer);
var
  ABytes, CBytes, GBytes, Proposed: UInt64;
begin
  if (Seq <= 0) or (Seq > High(Integer) - 63) then
    raise ERangeError.Create('Invalid network row count');
  ABytes := MatrixBytes(Seq, InputDim, InputDim, 4);
  CBytes := MatrixBytes(Seq, OutputDim, OutputDim, 4);
  GBytes := 0;
  if MlpDim > 0 then
    GBytes := MatrixBytes(Seq, MlpDim, MlpDim, 4);
  if (UInt64(Seq + 63) div 64 > VulkanDevice.DeviceProperties.Limits.MaxComputeWorkGroupCount[1])
    or (UInt64(OutputDim + 63) div 64 > VulkanDevice.DeviceProperties.Limits.MaxComputeWorkGroupCount[0])
    or ((MlpDim > 0)
    and (UInt64(MlpDim + 63) div 64 > VulkanDevice.DeviceProperties.Limits.MaxComputeWorkGroupCount[0])) then
    raise ERangeError.Create('Network exceeds Vulkan dispatch limits');
  Proposed :=
    2 * Max(ABytes, UInt64(FInputA.Size))
    + 2 * Max(CBytes, UInt64(FOutput.Size))
    + 2 * Max(GBytes, UInt64(FGraphGate.Size))
    + WEIGHT_UPLOAD_CHUNK;
  if Proposed > MAX_WORKING_BYTES then
    ReleaseWorking;
  // Raw GEMM's B buffers are not used when weights are resident.
  FreeMemory(FInputB);
  FInputB := Default(TVulkanMemory);
  FreeMemory(FUploadB);
  FUploadB := Default(TVulkanMemory);
  GrowWorking(FInputA, ABytes, False, False);
  GrowWorking(FUploadA, ABytes, True, False);
  GrowWorking(FOutput, CBytes, False, False);
  GrowWorking(FReadback, CBytes, True, True);
  if GBytes > 0 then
  begin
    GrowWorking(FGraphGate, GBytes, False, False);
    GrowWorking(FGraphUp, GBytes, False, False);
  end;
end;

function TQNNVulkan.EvictOne: Boolean;
var
  Unused: TVulkanMemory;
begin
  Result := EvictOne(0, Unused);
end;

function TQNNVulkan.EvictOne(const ReuseBytes: UInt64; out Reused: TVulkanMemory): Boolean;
var
  Pair: TPair<string, TGPUWeight>;
  Oldest: TGPUWeight;
  Key: string;
begin
  Reused := Default(TVulkanMemory);
  Oldest := nil;
  for Pair in FWeights do
    if (Pair.Value.Pins = 0) and ((Oldest = nil) or (Pair.Value.LastUse < Oldest.LastUse)) then
    begin
      Oldest := Pair.Value;
      Key := Pair.Key;
    end;
  Result := Oldest <> nil;
  if not Result then
    Exit;
  if (FResident <> nil) and Assigned(FResident.FOnWeightEvicted) then
    FResident.FOnWeightEvicted(Oldest.Identity);
  Dec(FWeightBytes, Oldest.Memory.Size);
  FWeights.Remove(Key);
  if (ReuseBytes <> 0) and (Oldest.Memory.Size = ReuseBytes) then
  begin
    Reused := Oldest.Memory;
    Oldest.Memory := Default(TVulkanMemory);
  end;
  Oldest.Free;
  Inc(FCounters.WeightEvictions);
end;

procedure TQNNVulkan.SetWeightLimit(const Limit: UInt64);
var
  ActualLimit, PinnedBytes: UInt64;
begin
  if FResident <> nil then
    FResident.Flush;
  ActualLimit := Min(Limit, FPhysicalWeightLimit);
  PinnedBytes := 0;
  for var Entry in FWeights.Values do
    if Entry.Pins <> 0 then
      Inc(PinnedBytes, Entry.Memory.Size);
  if ActualLimit < PinnedBytes then
    raise ERangeError.Create('GPU budget cannot evict retained weights');
  FWeightLimit := ActualLimit;
  while FWeightBytes > FWeightLimit do
    if not EvictOne then
      raise ERangeError.Create('Resident weights are pinned');
end;

procedure TQNNVulkan.ClearWeights;
var
  Pair: TPair<string, TGPUWeight>;
begin
  for Pair in FWeights do
    if Pair.Value.Pins <> 0 then
      raise EInvalidOp.Create('Cannot clear retained GPU weights while their context is active');
  for Pair in FWeights do
    Pair.Value.Free;
  FWeights.Clear;
  FWeightBytes := 0;
end;

procedure TQNNVulkan.InvalidateWeight(const Identity: IInterface);
var
  Keys: TList<string>;
  Pair: TPair<string, TGPUWeight>;
  Key: string;
  Entry: TGPUWeight;
begin
  if Identity = nil then
    Exit;
  Keys := TList<string>.Create;
  try
    for Pair in FWeights do
      if (Pair.Value.Identity = Identity) or (Pair.Value.ScaleIdentity = Identity) then
        Keys.Add(Pair.Key);
    for Key in Keys do
    begin
      Entry := FWeights[Key];
      if Entry.Pins <> 0 then
        raise Exception.Create('Cannot invalidate an active GPU weight');
      Dec(FWeightBytes, Entry.Memory.Size);
      FWeights.Remove(Key);
      Entry.Free;
    end;
  finally
    Keys.Free
  end;
end;

function TQNNVulkan.WeightKey(const Weight: TMemoryBlock; const Rows, Cols: Integer): string;
var
  ByteCount: UInt64;
begin
  if (Weight.WeightIdentity = nil)
    or not (Weight.DataType in [dtF32, dtBF16, dtU8])
    or not Weight.IsAssigned
    or (Int64(Rows) * Cols > Int64(Weight.Size) - Weight.Offset) then
    raise EArgumentException.Create('Weight must have immutable identity and sufficient native or packed storage');
  ByteCount := MatrixBytes(Rows, Cols, Cols, DATATYPE_BITS[Weight.DataType] div 8);
  Result :=
    IntToHex(NativeUInt(Pointer(Weight.WeightIdentity)), SizeOf(Pointer) * 2)
    + '/'
    + IntToStr(Ord(Weight.DataType))
    + '/'
    + IntToStr(Weight.Offset)
    + '/'
    + UIntToStr(ByteCount);
end;

function TQNNVulkan.AcquireWeight(const Weight: TMemoryBlock; const Rows, Cols: Integer; const AllowPackedInt4: Boolean): TGPUWeight;
var
  Key: string;
  ByteCount, Padded, Offset, Chunk, Actual: UInt64;
  Source: Pointer;
begin
  if (Weight.WeightIdentity = nil)
    or (not (Weight.DataType in [dtF32, dtBF16]) and not (AllowPackedInt4 and (Weight.DataType = dtU8)))
    or not Weight.IsAssigned
    or (Int64(Rows) * Cols > Int64(Weight.Size) - Weight.Offset) then
    raise EArgumentException.Create('Weight must have immutable identity and sufficient native or packed storage');
  ByteCount := MatrixBytes(Rows, Cols, Cols, DATATYPE_BITS[Weight.DataType] div 8);
  Padded := (ByteCount + 3) and not UInt64(3);
  Key := WeightKey(Weight, Rows, Cols);
  Inc(FWeightClock);
  if FWeights.TryGetValue(Key, Result) then
  begin
    Inc(Result.Pins);
    Result.LastUse := FWeightClock;
    Inc(FCounters.WeightCacheHits);
    Exit;
  end;
  Inc(FCounters.WeightCacheMisses);
  if FResident <> nil then
    FResident.Flush;
  if Padded > FWeightLimit then
    raise ERangeError.Create('Weight exceeds VRAM cache limit');
  while FWeightBytes + Padded > FWeightLimit do
    if not EvictOne then
      raise ERangeError.Create('VRAM cache cannot hold the active chain');
  GrowWorking(FWeightUpload, Min(Padded, WEIGHT_UPLOAD_CHUNK), True, False);
  Result := TGPUWeight.Create;
  Result.Owner := Self;
  Result.Identity := Weight.WeightIdentity; // Keeps identity unique, not host storage alive.
  Result.LastUse := FWeightClock;
  Result.Pins := 1;
  if Weight.DataType = dtU8 then
    Source := PByte(Weight)
  else if Weight.DataType = dtBF16 then
    Source := PBF16(Weight)
  else
    Source := System.PSingle(Weight);
  try
    Result.Memory := CreateStorageMemory(Padded);
    Offset := 0;
    while Offset < Padded do
    begin
      Chunk := Min(UInt64(FWeightUpload.Size), Padded - Offset);
      Actual := Min(Chunk, ByteCount - Min(Offset, ByteCount));
      Upload(PByte(Source) + Offset, FWeightUpload, Actual, Chunk);
      BeginCommadBuffer;
      CopyBuffer(FWeightUpload, Result.Memory, Chunk, 0, Offset);
      MemoryBarrier(
        LongWord(VK_PIPELINE_STAGE_TRANSFER_BIT),
        LongWord(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT),
        LongWord(VK_ACCESS_TRANSFER_WRITE_BIT),
        LongWord(VK_ACCESS_SHADER_READ_BIT)
      );
      EndCommandBuffer;
      Finish;
      Inc(FCounters.Submissions);
      Inc(Offset, Chunk);
    end;
    FWeights.Add(Key, Result);
    Inc(FWeightBytes, Padded);
    Inc(FCounters.WeightUploads);
    Inc(FCounters.WeightUploadBytes, ByteCount);
  except
    Result.Free;
    raise
  end;
end;

procedure TQNNVulkan.Attention(const Dst, Q, K, V: TMemoryBlock; const Rows, Heads, KvHeads, HeadDim, CausalRows: Integer; const KeyLimits: TArray<Integer>);
var
  PackedKV: TMemoryBlock;
  IndexK, IndexV, BaseV, SourceIndex: NativeInt;
  ABytes, BBytes: UInt64;
begin
  if not FAttentionAvailable then
    raise EArgumentException.Create('GPU needs 32 KiB shared memory for attention');
  ABytes := MatrixBytes(Rows, Heads * HeadDim, Heads * HeadDim, 4);
  BBytes :=
    MatrixBytes(
    Rows,
    KvHeads * HeadDim * 2 + Ord(Length(KeyLimits) > 0),
    KvHeads * HeadDim * 2 + Ord(Length(KeyLimits) > 0),
    4
  );
  EnsureBuffers(ABytes, BBytes, ABytes);
  PackedKV :=
    TMemoryBlock.Create([NativeInt(2) * KvHeads * HeadDim * Rows + Length(KeyLimits)], 'attention-kv-packedKV');
  try
    BaseV := NativeInt(Rows) * KvHeads * HeadDim;
    for var H := 0 to KvHeads - 1 do
      for var D := 0 to HeadDim - 1 do
        for var Row := 0 to Rows - 1 do
        begin
          SourceIndex := (NativeInt(Row) * KvHeads + H) * HeadDim + D;
          IndexK := (NativeInt(H) * HeadDim + D) * Rows + Row;
          IndexV := BaseV + (NativeInt(H) * Rows + Row) * HeadDim + D;
          System.PSingle(PackedKV)[IndexK] := System.PSingle(K)[SourceIndex];
          System.PSingle(PackedKV)[IndexV] := System.PSingle(V)[SourceIndex];
        end;
    for var Row := 0 to High(KeyLimits) do
      PUInt32(System.PSingle(PackedKV))[BaseV * 2 + Row] := KeyLimits[Row];
    Upload(System.PSingle(Q), FUploadA, ABytes, ABytes);
    Upload(System.PSingle(PackedKV), FUploadB, BBytes, BBytes);
    BeginCommadBuffer;
    CopyBuffer(FUploadA, FInputA, ABytes);
    CopyBuffer(FUploadB, FInputB, BBytes);
    MemoryBarrier(
      LongWord(VK_PIPELINE_STAGE_TRANSFER_BIT),
      LongWord(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT),
      LongWord(VK_ACCESS_TRANSFER_WRITE_BIT),
      LongWord(VK_ACCESS_SHADER_READ_BIT)
    );
    DispatchPipeline(
      TKernelHandle(2),
      [
        Rows,
        Heads,
        KvHeads,
        HeadDim,
        CausalRows,
        Ord(Length(KeyLimits) > 0),
        FInputA.Buffer,
        FInputB.Buffer,
        FOutput.Buffer
      ],
      Rows,
      Heads,
      1
    );
    Inc(FCounters.ActivationUploadBytes, ABytes + BBytes);
    EndNetwork(Dst, Rows, Heads * HeadDim);
  finally
    PackedKV.Free
  end;
end;

function TQNNVulkan.AcquireInt8Weight(const Weight, Scales: TMemoryBlock; const Rows, Cols: Integer): TGPUWeight;
var
  Key: string;
  RawBytes, ScaleStart, ScaleBytes, Total, Off, Chunk, Actual: UInt64;
  Source: Pointer;

  procedure Send(const Ptr: Pointer; const Bytes, DestOffset: UInt64);
  begin
    Off := 0;
    while Off < Bytes do
    begin
      Chunk := Min(UInt64(FWeightUpload.Size), Bytes - Off);
      Actual := Min(Chunk, Bytes - Off);
      Upload(PByte(Ptr) + Off, FWeightUpload, Actual, (Chunk + 3) and not UInt64(3));
      BeginCommadBuffer;
      CopyBuffer(FWeightUpload, Result.Memory, (Chunk + 3) and not UInt64(3), 0, DestOffset + Off);
      MemoryBarrier(
        LongWord(VK_PIPELINE_STAGE_TRANSFER_BIT),
        LongWord(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT),
        LongWord(VK_ACCESS_TRANSFER_WRITE_BIT),
        LongWord(VK_ACCESS_SHADER_READ_BIT)
      );
      EndCommandBuffer;
      Finish;
      Inc(FCounters.Submissions);
      Inc(Off, Chunk);
    end;
  end;

begin
  if (Weight.WeightIdentity = nil)
    or (Scales.WeightIdentity = nil)
    or (Weight.DataType <> DT_I8)
    or (Scales.DataType <> dtF32)
    or (Int64(Rows) * Cols > Int64(Weight.Size) - Weight.Offset)
    or (Int64(Rows) > Int64(Scales.Size) - Scales.Offset)
    or not Weight.IsAssigned
    or not Scales.IsAssigned then
    raise EArgumentException.Create('Invalid immutable INT8 matrix or per-row F32 scales');
  RawBytes := MatrixBytes(Rows, Cols, Cols, 1);
  ScaleStart := (RawBytes + 3) and not UInt64(3);
  ScaleBytes := UInt64(Rows) * 4;
  Total := ScaleStart + ScaleBytes;
  if Total > VulkanDevice.DeviceProperties.Limits.MaxStorageBufferRange then
    raise ERangeError.Create('Packed INT8 matrix and scales exceed maxStorageBufferRange');
  Key :=
    'i8/'
    + IntToHex(NativeUInt(Pointer(Weight.WeightIdentity)), SizeOf(Pointer) * 2)
    + '/'
    + IntToHex(NativeUInt(Pointer(Scales.WeightIdentity)), SizeOf(Pointer) * 2)
    + '/'
    + IntToStr(Weight.Offset)
    + '/'
    + IntToStr(Scales.Offset)
    + '/'
    + IntToStr(Rows)
    + '/'
    + IntToStr(Cols);
  Inc(FWeightClock);
  if FWeights.TryGetValue(Key, Result) then
  begin
    Inc(Result.Pins);
    Result.LastUse := FWeightClock;
    Inc(FCounters.WeightCacheHits);
    Exit;
  end;
  Inc(FCounters.WeightCacheMisses);
  if FResident <> nil then
    FResident.Flush;
  if Total > FWeightLimit then
    raise ERangeError.Create('INT8 weight exceeds VRAM cache limit');
  while FWeightBytes + Total > FWeightLimit do
    if not EvictOne then
      raise ERangeError.Create('VRAM cannot hold active INT8 weights');
  GrowWorking(FWeightUpload, Min(Total, WEIGHT_UPLOAD_CHUNK), True, False);
  Result := TGPUWeight.Create;
  Result.Owner := Self;
  Result.Identity := Weight.WeightIdentity;
  Result.ScaleIdentity := Scales.WeightIdentity;
  Result.LastUse := FWeightClock;
  Result.Pins := 1;
  try
    Result.Memory := CreateStorageMemory(Total);
    Source := PShortInt(Weight);
    Send(Source, RawBytes, 0);
    Source := System.PSingle(Scales);
    Send(Source, ScaleBytes, ScaleStart);
    FWeights.Add(Key, Result);
    Inc(FWeightBytes, Total);
    Inc(FCounters.WeightUploads);
    Inc(FCounters.WeightUploadBytes, RawBytes + ScaleBytes);
  except
    Result.Free;
    raise
  end;
end;

procedure TQNNVulkan.CachedInt8Linear(const Dst, Src: System.PSingle; const Weight, Scales: TMemoryBlock; const Seq, InputDim, OutputDim: Integer);
var
  Entry: TGPUWeight;
begin
  PrepareNetwork(Seq, InputDim, OutputDim, 0);
  Entry := AcquireInt8Weight(Weight, Scales, OutputDim, InputDim);
  try
    BeginNetwork(Src, Seq, InputDim);
    DispatchPipeline(
      TKernelHandle(0),
      [
        Seq,
        OutputDim,
        InputDim,
        Single(1),
        FInputA.Buffer,
        0,
        InputDim,
        Entry.Memory.Buffer,
        0,
        InputDim,
        Single(0),
        FOutput.Buffer,
        0,
        OutputDim,
        10
      ],
      (OutputDim + 63) div 64,
      (Seq + 63) div 64,
      1,
      0
    );
    EndNetwork(Dst, Seq, OutputDim);
  finally
    Dec(Entry.Pins)
  end;
end;

procedure TQNNVulkan.RecordLinear(const A, B, C: TVulkanMemory; const Seq, InputDim, OutputDim: Integer; const BF16: Boolean; const Slot: Integer);
begin
  DispatchPipeline(
    TKernelHandle(0),
    [
      Seq,
      OutputDim,
      InputDim,
      Single(1),
      A.Buffer,
      0,
      InputDim,
      B.Buffer,
      0,
      InputDim,
      Single(0),
      C.Buffer,
      0,
      OutputDim,
      2 or (Ord(BF16) shl 2)
    ],
    (OutputDim + 63) div 64,
    (Seq + 63) div 64,
    1,
    Slot
  );
end;

procedure TQNNVulkan.BeginNetwork(const Src: System.PSingle; const Seq, Dim: Integer);
var
  Bytes: UInt64;
begin
  Bytes := UInt64(Seq) * UInt64(Dim) * 4;
  Upload(Src, FUploadA, Bytes, Bytes);
  BeginCommadBuffer;
  CopyBuffer(FUploadA, FInputA, Bytes);
  MemoryBarrier(
    LongWord(VK_PIPELINE_STAGE_TRANSFER_BIT),
    LongWord(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT),
    LongWord(VK_ACCESS_TRANSFER_WRITE_BIT),
    LongWord(VK_ACCESS_SHADER_READ_BIT)
  );
  Inc(FCounters.ActivationUploadBytes, Bytes);
end;

procedure TQNNVulkan.EndNetwork(const Dst: System.PSingle; const Seq, Dim: Integer);
var
  Bytes: UInt64;
  Mapped: Pointer;
begin
  Bytes := UInt64(Seq) * UInt64(Dim) * 4;
  MemoryBarrier(
    LongWord(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT),
    LongWord(VK_PIPELINE_STAGE_TRANSFER_BIT),
    LongWord(VK_ACCESS_SHADER_WRITE_BIT),
    LongWord(VK_ACCESS_TRANSFER_READ_BIT)
  );
  CopyBuffer(FOutput, FReadback, Bytes);
  MemoryBarrier(
    LongWord(VK_PIPELINE_STAGE_TRANSFER_BIT),
    LongWord(VK_PIPELINE_STAGE_HOST_BIT),
    LongWord(VK_ACCESS_TRANSFER_WRITE_BIT),
    LongWord(VK_ACCESS_HOST_READ_BIT)
  );
  EndCommandBuffer;
  Finish;
  Inc(FCounters.Submissions);
  Mapped := MapMemory(FReadback);
  try
    Move(Mapped^, Dst^, Bytes)
  finally
    UnMapMemory(FReadback)
  end;
  Inc(FCounters.Readbacks);
  Inc(FCounters.ActivationDownloadBytes, Bytes);
end;

procedure TQNNVulkan.CachedLinear(const Dst, Src: System.PSingle; const Weight: TMemoryBlock; const Seq, InputDim, OutputDim: Integer);
var
  Entry: TGPUWeight;
begin
  PrepareNetwork(Seq, InputDim, OutputDim, 0);
  Entry := AcquireWeight(Weight, OutputDim, InputDim);
  try
    BeginNetwork(Src, Seq, InputDim);
    RecordLinear(FInputA, Entry.Memory, FOutput, Seq, InputDim, OutputDim, Weight.DataType = dtBF16, 0);
    EndNetwork(Dst, Seq, OutputDim);
  finally
    Dec(Entry.Pins)
  end;
end;

procedure TQNNVulkan.Swiglu(const Dst, Src: System.PSingle; const Gate, Up, Down: TMemoryBlock; const Seq, Hidden, Mlp: Integer);
var
  GateEntry, UpEntry, DownEntry: TGPUWeight;
  Count, Groups, GroupsX, GroupsY: UInt64;
begin
  GateEntry := nil;
  UpEntry := nil;
  DownEntry := nil;
  PrepareNetwork(Seq, Hidden, Hidden, Mlp);
  Count := UInt64(Seq) * UInt64(Mlp);
  Groups := (Count + 127) div 128;
  GroupsX := Min(Groups, UInt64(VulkanDevice.DeviceProperties.Limits.MaxComputeWorkGroupCount[0]));
  GroupsY := (Groups + GroupsX - 1) div GroupsX;
  if GroupsY > VulkanDevice.DeviceProperties.Limits.MaxComputeWorkGroupCount[1] then
    raise ERangeError.Create('SwiGLU activation exceeds dispatch limits');
  try
    GateEntry := AcquireWeight(Gate, Mlp, Hidden);
    UpEntry := AcquireWeight(Up, Mlp, Hidden);
    DownEntry := AcquireWeight(Down, Hidden, Mlp);
    BeginNetwork(Src, Seq, Hidden);
    RecordLinear(FInputA, GateEntry.Memory, FGraphGate, Seq, Hidden, Mlp, Gate.DataType = dtBF16, 0);
    RecordLinear(FInputA, UpEntry.Memory, FGraphUp, Seq, Hidden, Mlp, Up.DataType = dtBF16, 1);
    MemoryBarrier(
      LongWord(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT),
      LongWord(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT),
      LongWord(VK_ACCESS_SHADER_WRITE_BIT),
      LongWord(VK_ACCESS_SHADER_READ_BIT) or LongWord(VK_ACCESS_SHADER_WRITE_BIT)
    );
    DispatchPipeline(
      TKernelHandle(1),
      [FGraphGate.Buffer, FGraphUp.Buffer, FGraphGate.Buffer, Integer(Count)],
      GroupsX,
      GroupsY
    );
    MemoryBarrier(
      LongWord(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT),
      LongWord(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT),
      LongWord(VK_ACCESS_SHADER_WRITE_BIT),
      LongWord(VK_ACCESS_SHADER_READ_BIT)
    );
    RecordLinear(FGraphGate, DownEntry.Memory, FOutput, Seq, Mlp, Hidden, Down.DataType = dtBF16, 2);
    EndNetwork(Dst, Seq, Hidden);
    Inc(FCounters.PipelineCalls);
  finally
    if DownEntry <> nil then
      Dec(DownEntry.Pins);
    if UpEntry <> nil then
      Dec(UpEntry.Pins);
    if GateEntry <> nil then
      Dec(GateEntry.Pins);
  end;
end;

procedure TQNNVulkan.CopyCounters(var Status: TQNNGPUStatus);
begin
  Status.WeightCacheBytes := FWeightBytes;
  Status.WeightCacheLimit := FWeightLimit;
  Status.WeightCacheHits := FCounters.WeightCacheHits;
  Status.WeightCacheMisses := FCounters.WeightCacheMisses;
  Status.WeightUploads := FCounters.WeightUploads;
  Status.WeightUploadBytes := FCounters.WeightUploadBytes;
  Status.WeightEvictions := FCounters.WeightEvictions;
  Status.Submissions := FCounters.Submissions;
  Status.Readbacks := FCounters.Readbacks;
  Status.ActivationUploadBytes := FCounters.ActivationUploadBytes;
  Status.ActivationDownloadBytes := FCounters.ActivationDownloadBytes;
  Status.PipelineCalls := FCounters.PipelineCalls;
end;

{$ENDIF}

function EnsureGPU: Boolean;
begin
  Result := False;
  if GPUMode = gpuOff then
    Exit;
{$IF defined(MSWINDOWS) and defined(CPUX64)}
  if not GPUAttempted then
  begin
    GPUAttempted := True;
    try
      GPU := TQNNVulkan.Create(GPUDeviceIndex);
      GPUStatus.Available := True;
      GPUStatus.DeviceName := GPU.DeviceName;
    except
      on E: Exception do
        GPUStatus.LastError := E.ClassName + ': ' + E.Message;
    end;
  end;
  Result := GPU <> nil;
{$ELSE}
  GPUStatus.LastError := 'Vulkan backend currently requires Delphi on Windows x64';
{$ENDIF}
end;

procedure QNNConfigureGPU(const Mode: TQNNGPUMode; const DeviceIndex: Integer);
begin
  GPULock.Acquire;
  try
{$IF defined(MSWINDOWS) and defined(CPUX64)}
    if (GPU <> nil) and (GPU.FResident <> nil) then
      raise EInvalidOp.Create('Cannot reconfigure an active resident context');
    FreeAndNil(GPU);
{$ENDIF}
    GPUStatus := Default(TQNNGPUStatus);
    GPUAttempted := False;
    GPUDeviceIndex := DeviceIndex;
    GPUMode := Mode;
  finally
    GPULock.Release
  end;
end;

function QNNGetGPUStatus: TQNNGPUStatus;
begin
  GPULock.Acquire;
  try
    EnsureGPU;
{$IF defined(MSWINDOWS) and defined(CPUX64)}
    GPUStatus.AllocatedBytes := 0;
    GPUStatus.WeightCacheBytes := 0;
    if GPU <> nil then
    begin
      GPUStatus.AllocatedBytes := GPU.AllocatedBytes;
      GPU.CopyCounters(GPUStatus);
    end;
{$ENDIF}
    Result := GPUStatus;
  finally
    GPULock.Release
  end;
end;

procedure QNNSetGPUWeightCacheLimit(const Bytes: UInt64);
begin
  GPULock.Acquire;
  try
{$IF defined(MSWINDOWS) and defined(CPUX64)}
    if GPU <> nil then
      GPU.SetWeightLimit(Bytes);
{$ENDIF}
    GPUWeightLimit := Bytes;
  finally
    GPULock.Release
  end;
end;

procedure QNNGPUClearWeightCache;
begin
  GPULock.Acquire;
  try
{$IF defined(MSWINDOWS) and defined(CPUX64)}
    if GPU <> nil then
    begin
      if GPU.FResident <> nil then
        GPU.FResident.Flush;
      GPU.ClearWeights
    end;
{$ENDIF}
  finally
    GPULock.Release
  end;
end;

procedure QNNInvalidateGPUWeight(const Weight: TMemoryBlock);
begin
  GPULock.Acquire;
  try
{$IF defined(MSWINDOWS) and defined(CPUX64)}
    if GPU <> nil then
      GPU.InvalidateWeight(Weight.WeightIdentity);
{$ENDIF}
  finally
    GPULock.Release
  end;
end;

function UseNetworkGPU(const Seq, InputDim, OutputDim: Integer; const Products: Double): Boolean;
begin
  Result :=
    (GPUMode <> gpuOff)
    and (Seq > 0)
    and (InputDim > 0)
    and (OutputDim > 0)
    and (InputDim <= High(Integer) - 63)
    and (OutputDim <= High(Integer) - 63);
  if (GPUMode = gpuAuto) and ((Seq < 8) or (Products < AUTO_MIN_PRODUCTS)) then
    Result := False;
end;

function TryVulkanLinear(const Dst, Src: System.PSingle; const Weight: TMemoryBlock; const Seq, InputDim, OutputDim: Integer): Boolean;
{$IF defined(MSWINDOWS) and defined(CPUX64)}
var
  PreviousMask: TFPUExceptionMask;
{$ENDIF}
begin
  Result := False;
  if (Weight.WeightIdentity = nil)
    or (Dst = nil)
    or (Src = nil)
    or not UseNetworkGPU(Seq, InputDim, OutputDim, Double(Seq) * InputDim * OutputDim) then
    Exit;
  GPULock.Acquire;
  try
    if not EnsureGPU then
      Exit;
{$IF defined(MSWINDOWS) and defined(CPUX64)}
    PreviousMask := GetExceptionMask;
    SetExceptionMask([ExInvalidOp, ExDenormalized, ExZeroDivide, ExOverflow, ExUnderflow, ExPrecision]);
    try
      try
        GPU.CachedLinear(Dst, Src, Weight, Seq, InputDim, OutputDim);
        Inc(GPUStatus.GemmCalls);
        if Weight.DataType = dtBF16 then
          Inc(GPUStatus.BF16Calls);
        Result := True;
      except
        on E: Exception do
        begin
          GPUStatus.LastError := E.ClassName + ': ' + E.Message;
          if not (E is ERangeError) and not (E is EArgumentException) then
          begin
            FreeAndNil(GPU);
            GPUStatus.Available := False;
          end;
        end;
      end;
    finally
      SetExceptionMask(PreviousMask)
    end;
{$ENDIF}
  finally
    GPULock.Release
  end;
end;

function TryVulkanAttention(const Dst, Q, K, V: TMemoryBlock; const Rows, Heads, KvHeads, HeadDim, CausalRows: Integer; const KeyLimits: TArray<Integer>): Boolean;
{$IF defined(MSWINDOWS) and defined(CPUX64)}
var
  PreviousMask: TFPUExceptionMask;
{$ENDIF}
begin
  Result := False;
  if (GPUMode = gpuOff)
    or (Rows <= 0)
    or (Rows > 6144)
    or (Heads <= 0)
    or (Heads > 128)
    or (KvHeads <= 0)
    or (Heads mod KvHeads <> 0)
    or (HeadDim <= 0)
    or (HeadDim > 128)
    or (CausalRows < 0)
    or (CausalRows > Rows)
    or (Dst.DataType <> dtF32)
    or (Q.DataType <> dtF32)
    or (K.DataType <> dtF32)
    or (V.DataType <> dtF32)
    or (Int64(Rows) * Heads * HeadDim > Dst.Size - Dst.Offset)
    or (Int64(Rows) * Heads * HeadDim > Q.Size - Q.Offset)
    or (Int64(Rows) * KvHeads * HeadDim > K.Size - K.Offset)
    or (Int64(Rows) * KvHeads * HeadDim > V.Size - V.Offset)
    or not Dst.IsAssigned
    or not Q.IsAssigned
    or not K.IsAssigned
    or not V.IsAssigned then
    Exit;
  if (Length(KeyLimits) <> 0) and (Length(KeyLimits) <> Rows) then
    Exit;
  for var Limit in KeyLimits do
    if (Limit < 1) or (Limit > Rows) then
      Exit;
  GPULock.Acquire;
  try
    if not EnsureGPU then
      Exit;
{$IF defined(MSWINDOWS) and defined(CPUX64)}
    PreviousMask := GetExceptionMask;
    SetExceptionMask([ExInvalidOp, ExDenormalized, ExZeroDivide, ExOverflow, ExUnderflow, ExPrecision]);
    try
      try
        GPU.Attention(Dst, Q, K, V, Rows, Heads, KvHeads, HeadDim, CausalRows, KeyLimits);
        Inc(GPUStatus.AttentionCalls);
        GPUStatus.LastError := '';
        Result := True;
      except
        on E: Exception do
          GPUStatus.LastError := E.ClassName + ': ' + E.Message
      end;
    finally
      SetExceptionMask(PreviousMask)
    end;
{$ENDIF}
  finally
    GPULock.Release
  end;
end;

function TryVulkanInt8Linear(const Dst, Src: System.PSingle; const Weight, Scales: TMemoryBlock; const Seq, InputDim, OutputDim: Integer): Boolean;
{$IF defined(MSWINDOWS) and defined(CPUX64)}
var
  PreviousMask: TFPUExceptionMask;
{$ENDIF}
begin
  Result := False;
  if (Dst = nil) or (Src = nil) or not UseNetworkGPU(Seq, InputDim, OutputDim, Double(Seq) * InputDim * OutputDim) then
    Exit;
  GPULock.Acquire;
  try
    if not EnsureGPU then
      Exit;
{$IF defined(MSWINDOWS) and defined(CPUX64)}
    PreviousMask := GetExceptionMask;
    SetExceptionMask([ExInvalidOp, ExDenormalized, ExZeroDivide, ExOverflow, ExUnderflow, ExPrecision]);
    try
      try
        GPU.CachedInt8Linear(Dst, Src, Weight, Scales, Seq, InputDim, OutputDim);
        Inc(GPUStatus.GemmCalls);
        GPUStatus.LastError := '';
        Result := True;
      except
        on E: Exception do
          GPUStatus.LastError := E.ClassName + ': ' + E.Message
      end;
    finally
      SetExceptionMask(PreviousMask)
    end;
{$ENDIF}
  finally
    GPULock.Release
  end;
end;

function TryVulkanSwiGLU(const Dst, Src, Gate, Up, Down: TMemoryBlock; const Seq, Hidden, Mlp: Integer): Boolean;
{$IF defined(MSWINDOWS) and defined(CPUX64)}
var
  PreviousMask: TFPUExceptionMask;
{$ENDIF}
begin
  Result := False;
  if (Dst.DataType <> dtF32)
    or (Src.DataType <> dtF32)
    or (Gate.WeightIdentity = nil)
    or (Up.WeightIdentity = nil)
    or (Down.WeightIdentity = nil)
    or not UseNetworkGPU(Seq, Hidden, Mlp, Double(Seq) * Hidden * Mlp * 3)
    or (Int64(Seq) * Hidden > Src.Size - Src.Offset)
    or (Int64(Seq) * Hidden > Dst.Size - Dst.Offset)
    or not Src.IsAssigned
    or not Dst.IsAssigned then
    Exit;
  GPULock.Acquire;
  try
    if not EnsureGPU then
      Exit;
{$IF defined(MSWINDOWS) and defined(CPUX64)}
    PreviousMask := GetExceptionMask;
    SetExceptionMask([ExInvalidOp, ExDenormalized, ExZeroDivide, ExOverflow, ExUnderflow, ExPrecision]);
    try
      try
        GPU.Swiglu(Dst, Src, Gate, Up, Down, Seq, Hidden, Mlp);
        Inc(GPUStatus.GemmCalls, 3);
        Inc(GPUStatus.BF16Calls, Ord(Gate.DataType = dtBF16) + Ord(Up.DataType = dtBF16) + Ord(Down.DataType = dtBF16));
        Result := True;
      except
        on E: Exception do
        begin
          GPUStatus.LastError := E.ClassName + ': ' + E.Message;
          if not (E is ERangeError) and not (E is EArgumentException) then
          begin
            FreeAndNil(GPU);
            GPUStatus.Available := False;
          end;
        end;
      end;
    finally
      SetExceptionMask(PreviousMask)
    end;
{$ENDIF}
  finally
    GPULock.Release
  end;
end;

function TryProduct(const TransA, TransB: Boolean; const M, N, K: Int64; const Alpha: Single; const A: System.PSingle; const Lda: Int64; const B: Pointer; const Ldb: Int64; const Beta: Single; const C: System.PSingle; const Ldc: Int64; const BIsBF16: Boolean): Boolean;
begin
  Result := False;
  if (GPUMode = gpuOff)
    or (M <= 0)
    or (N <= 0)
    or (K <= 0)
    or (Alpha = 0)
    or (M > High(Integer) - 63)
    or (N > High(Integer) - 63)
    or (K > High(Integer))
    or (Lda > High(Integer))
    or (Ldb > High(Integer))
    or (Ldc > High(Integer)) then
    Exit;
  if (GPUMode = gpuAuto) and ((M < 8) or (Double(M) * N * K < AUTO_MIN_PRODUCTS)) then
    Exit;
  GPULock.Acquire;
  try
    if not EnsureGPU then
      Exit;
{$IF defined(MSWINDOWS) and defined(CPUX64)}
    try
      GPU.Sgemm(TransA, TransB, M, N, K, Alpha, A, Lda, B, Ldb, Beta, C, Ldc, BIsBF16);
      Inc(GPUStatus.GemmCalls);
      if BIsBF16 then
        Inc(GPUStatus.BF16Calls);
      Result := True;
    except
      on E: Exception do
      begin
        GPUStatus.LastError := E.ClassName + ': ' + E.Message;
        // Bad/oversized work may use CPU without disabling valid future work.
        if not (E is ERangeError) and not (E is EArgumentException) then
        begin
          FreeAndNil(GPU);
          GPUStatus.Available := False;
        end;
      end;
    end;
{$ENDIF}
  finally
    GPULock.Release
  end;
end;

function TryVulkanGemm(const Order: TCBLASOrder; const TransA, TransB: TCBLASTranspose; const M, N, K: Int64; const Alpha: Single; const A: System.PSingle; const Lda: Int64; const B: System.PSingle; const Ldb: Int64; const Beta: Single; const C: System.PSingle; const Ldc: Int64): Boolean;
begin
  // For real-valued matrices conjugate transpose is just transpose.
  if Order = CblasColMajor then
    Result :=
      TryProduct(
      TransB in [CblasTrans, CblasConjTrans],
      TransA in [CblasTrans, CblasConjTrans],
      N,
      M,
      K,
      Alpha,
      B,
      Ldb,
      A,
      Lda,
      Beta,
      C,
      Ldc,
      False
    )
  else
    Result :=
      TryProduct(
      TransA in [CblasTrans, CblasConjTrans],
      TransB in [CblasTrans, CblasConjTrans],
      M,
      N,
      K,
      Alpha,
      A,
      Lda,
      B,
      Ldb,
      Beta,
      C,
      Ldc,
      False
    );
end;

function TryVulkanLinearBF16(const Dst, Src: System.PSingle; const Weight: PBF16; const Seq, SrcDim, DstDim: Integer): Boolean;
begin
  Result := TryProduct(False, True, Seq, DstDim, SrcDim, 1, Src, SrcDim, Weight, SrcDim, 0, Dst, DstDim, True);
end;

{$I QuickNN.Vulkan.Context.Implementation.inc}

initialization
  GPULock := TCriticalSection.Create;
  if SameText(GetEnvironmentVariable('QUICKNN_GPU'), 'off') then
    GPUMode := gpuOff;
  if SameText(GetEnvironmentVariable('QUICKNN_GPU'), 'force') then
    GPUMode := gpuForce;

finalization
{$IF defined(MSWINDOWS) and defined(CPUX64)}
  GPU.Free;
{$ENDIF}
  GPULock.Free;

end.

