// Resident activation arena. One context owns the GPU lock on its creating
// thread. Recycle only after recording the last use; buffers stay alive until
// commands finish. No activation readback occurs in Flush or weight eviction.
TQNNGPUTensor = class
private
  FOwner: TQNNGPUContext;
  FLive: Boolean;
{$IF defined(MSWINDOWS) and defined(CPUX64)}
  FMemory: TVulkanMemory;
{$ENDIF}
public
  Count: Integer;
end;
TQNNGPUContext = class
private
{$IF defined(MSWINDOWS) and defined(CPUX64)}
  FBuffers: TArray<TQNNGPUTensor>;
{$ENDIF}
  FError: TQNNGPUTensor;
  FScratch: array[0..3] of TQNNGPUTensor;
  FBytes: UInt64;
  FOnWeightEvicted: TProc<IInterface>;
{$IF defined(MSWINDOWS) and defined(CPUX64)}
  FThread: Cardinal;
  FLocked: Boolean;
  FPending: Integer;
  FSlots: array[0..4] of Integer;
{$ENDIF}
{$IF defined(MSWINDOWS) and defined(CPUX64)}
  FDevice: TQNNVulkan;
  FPins: TArray<TGPUWeight>;
  FRetained: TArray<TGPUWeight>;
  FUpload, FRead, FBatchUpload: TVulkanMemory;
  FMask: TFPUExceptionMask;
  procedure Start;
  procedure Barrier;
  procedure Pin(const Entry: TGPUWeight);
  procedure RecordKernel(
      const Op, N: Integer;
      const Dst, A, B, C: TQNNGPUTensor;
      const Args: array of Integer;
      const Wb, Wc: TGPUWeight;
      const Bf16b, Bf16c: Boolean
  );
{$ENDIF}
  procedure Check(const Tensor: TQNNGPUTensor; const Minimum: Int64 = 1);
public
  constructor Create;
  destructor Destroy; override;
  function NewTensor(const Count: Integer): TQNNGPUTensor;
  procedure Recycle(var Tensor: TQNNGPUTensor);
  function Scratch(const Slot, Count: Integer): TQNNGPUTensor;
  procedure Flush;
  // Retained static weights cannot be evicted by streamed MoE experts.
  procedure RetainWeight(const Weight: TMemoryBlock; const Rows, Columns: Integer);
  // All sources are consumed synchronously; weights remain pinned until Flush.
  procedure PreloadWeights(const Weights: array of TMemoryBlock);
  function WeightBatchCapacity: UInt64;
  function IsWeightResident(const Weight: TMemoryBlock; const Rows, Columns: Integer): Boolean;
  function Upload(const Data: TMemoryBlock): TQNNGPUTensor;
  function UploadRaw(const Data: Pointer; const Count: Integer): TQNNGPUTensor;
  procedure Download(const Tensor: TQNNGPUTensor; const Dst: Pointer; const Count: Integer = -1);
  procedure Copy(
      const Dst, Src: TQNNGPUTensor;
      const Count: Integer;
      const DstOffset: Integer = 0;
      const SrcOffset: Integer = 0
  );
  procedure Kernel(const Op, N: Integer; const Dst, A, B, C: TQNNGPUTensor; const Args: array of Integer);
  procedure KernelWeights(
      const Op, N: Integer;
      const Dst, A: TQNNGPUTensor;
      const B, C: TMemoryBlock;
      const Args: array of Integer
  );
  procedure Linear(
      const Dst, Src: TQNNGPUTensor;
      const Weight, Scales, Bias: TMemoryBlock;
      const Rows, InputDim, OutputDim: Integer;
      const Rotation: Integer = 0
  );
  procedure Attention(
      const Dst, Q, K, V: TQNNGPUTensor;
      const Rows, Heads, KvHeads, Dim, Causal: Integer;
      const Limits: TQNNGPUTensor = nil
  );
  // A one-token projection avoids padding decoding to a 64-row GEMM tile.
  procedure VectorLinear(const Dst, Src: TQNNGPUTensor; const Weight: TMemoryBlock; const InputDim, OutputDim: Integer);
  // Exact tiled top-k and nucleus sampling; Dst receives a device token ID.
  // Packed INT4 remains compressed in VRAM, including multi-row prefill.
  procedure Int4Linear(
      const Dst, Src: TQNNGPUTensor;
      const Weight: TMemoryBlock;
      const Rows, InputDim, OutputDim: Integer
  );
  procedure SampleToken(
      const Dst, Logits, Seen: TQNNGPUTensor;
      const Vocab, TopK: Integer;
      const Temperature, TopP, Penalty, Uniform: Single
  );
  procedure CachedAttention(
      const Dst, Q, Cache: TQNNGPUTensor;
      const QueryRows, Capacity, Heads, KvHeads, Dim, Position: Integer
  );
  procedure Gather(
      const Dst, Ids: TQNNGPUTensor;
      const Weight, Scales: TMemoryBlock;
      const Rows, Dim, Vocab, Rotation: Integer
  );
  procedure CheckFinite(const Tensor: TQNNGPUTensor; const Count: Integer = -1);
  function AllocatedBytes: UInt64;
  property OnWeightEvicted: TProc<IInterface> read FOnWeightEvicted write FOnWeightEvicted;
end;
function QNNGPUFloatBits(const Value: Single): Integer;
const
  GO_FILL = 0;
  GO_CONV_ROT = 1;
  GO_NORM = 2;
  GO_UNARY = 3;
  GO_ADD = 4;
  GO_MOD_NORM = 5;
  GO_GATE_ADD = 6;
  GO_TEXT_ROPE = 7;
  GO_IMAGE_ROPE = 8;
  GO_SWIGLU = 9;
  GO_PACK_KV = 10;
  GO_SLICE = 11;
  GO_TIME = 12;
  GO_FINITE = 13;
  GO_CFG = 14;
  GO_GATHER = 15;
  GO_VISION_PATCH = 16;
  GO_VISION_POSITION = 17;
  GO_VISION_ROPE = 18;
  GO_PIXEL_INPUT = 19;
  GO_IM2COL = 20;
  GO_CONV_OUTPUT = 21;
  GO_LATENT_CHW = 22;
  GO_LATENT_NORMALIZE = 23;
  GO_NEAREST = 24;
  GO_UP_RESIDUAL = 25;
  GO_DOWN_RESIDUAL = 26;
  GO_PIXEL_OUTPUT = 27;
  GO_CHW_TO_ROWS = 28;
  GO_ROWS_TO_CHW = 29;

  GO_GROUP_NORM = 30;
  GO_AFFINE = 31;
  GO_ROTARY_CACHE = 32;
  GO_SINUSOIDAL = 33;
  GO_PATCH = 34;
  GO_LATENT_BN = 35;
  GO_PAD_END = 36;
  GO_RGB_OUTPUT = 37;
  GO_PAD_WEIGHT = 38;
  GO_SCATTER = 39;
  GO_ZI_ROPE = 40;
  GO_ZI_POSITION = 41;
  GO_FLUX_ROPE = 42;
  GO_SCALE = 43;

  GO_LLM_CACHE_WRITE = 44;
  GO_LLM_ROPE = 45;
  GO_LLM_SAMPLE = 46;
  GO_LLM_MATVEC = 47;
  GO_LLM_SEEN = 48;
  GO_LLM_MATVEC_BF16 = 49;
  GO_LLM_TOPK_TILE = 50;
  GO_LLM_TOPK_MERGE = 51;
  GO_MOE_ROUTE = 52;
  GO_MOE_GATHER = 53;
  GO_MOE_SCATTER = 54;
  GO_MOE_INT4_MATVEC = 55;
  GO_MOE_INT4_GEMM = 56;
