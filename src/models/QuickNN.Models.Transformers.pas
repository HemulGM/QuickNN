unit QuickNN.Models.Transformers;

{$C+} // enable assertions
{$H+}
{$POINTERMATH on}
{$T+}

{.$define USE_MULTITHREADING}

interface

uses
  System.SysUtils, System.Math, QuickNN.SafeTensors, QuickNN.Common,
  QuickNN.Models.VAE, QuickNN.Models.Qwen3, System.JSON
{$IFDEF USE_MULTITHREADING}
    , QuickNN.Threading
{$ENDIF}
;

const
  FLUX2_LATENT_RGB_PROJ: TArray<TArray<Single>> =
    [
      [0.000736, -0.008385, -0.019710],
      [-0.001352, -0.016392, 0.020693],
      [-0.006376, 0.002428, 0.036736],
      [0.039384, 0.074167, 0.119789],
      [0.007464, -0.005705, -0.004734],
      [-0.004086, 0.005287, -0.000409],
      [-0.032835, 0.050802, -0.028120],
      [-0.003158, -0.000835, 0.000406],
      [-0.112840, -0.084337, -0.023083],
      [0.001462, -0.006656, 0.000549],
      [-0.009980, -0.007480, 0.009702],
      [0.032540, 0.000214, -0.061388],
      [0.011023, 0.000694, 0.007143],
      [-0.001468, -0.006723, -0.001678],
      [-0.005921, -0.010320, -0.003907],
      [-0.028434, 0.027584, 0.018457],
      [0.014349, 0.011523, 0.000441],
      [0.009874, 0.003081, 0.001507],
      [0.002218, 0.005712, 0.001563],
      [0.053010, -0.019844, 0.008683],
      [-0.002507, 0.005384, 0.000938],
      [-0.002177, -0.011366, 0.003559],
      [-0.000261, 0.015121, -0.003240],
      [-0.003944, -0.002083, 0.005043],
      [-0.009138, 0.011336, 0.003781],
      [0.011429, 0.003985, -0.003855],
      [0.010518, -0.005586, 0.010131],
      [0.007883, 0.002912, -0.001473],
      [-0.003318, -0.003160, 0.003684],
      [-0.034560, -0.008740, 0.012996],
      [0.000166, 0.001079, -0.012153],
      [0.017772, 0.000937, -0.011953]
    ];
  FLUX2_LATENT_RGB_BIAS: TArray<Single> = [-0.028738, -0.098463, -0.107619];

type
  PSingles = ^TSingles;

  TSingles = TMemoryBlock;

  TBF16s = TMemoryBlock;

  // AdaLN-Zero modulation parameters (shared across blocks)
  TModeWeight = record
    ModWeight: TSingles; (* [hidden * 6] for double, [hidden * 3] for single *)
  end;

  (* Double-stream block (MM-DiT style) *)
  PDoubleBlock = ^TDoubleBlock;

  { TDoubleBlock }

  TDoubleBlock = record
    (* Image stream - separate Q, K, V weights (f32 and bf16) *)
    // [hidden, hidden]
    ImgQWeight: TSingles;
    ImgKWeight: TSingles;
    ImgVWeight: TSingles;
    ImgQWeightBF16: TBF16s;
    ImgKWeightBF16: TBF16s;
    ImgVWeightBF16: TBF16s;

    // [head_dim] - QK norm on Q and K
    ImgNormQWeight: TSingles;
    ImgNormKWeight: TSingles;

    // [hidden, hidden]
    ImgProjWeight: TSingles;
    ImgProjWeightBF16: TBF16s;

    // [mlp_hidden, hidden]
    ImgMlpGateWeight: TSingles;
    ImgMlpGateWeightBF16: TBF16s;
    ImgMlpUpWeight: TSingles;
    ImgMlpUpWeightBF16: TBF16s;

    // [hidden, mlp_hidden]
    ImgMlpDownWeight: TSingles;
    ImgMlpDownWeightBF16: TBF16s;

    // Text stream - separate Q, K, V weights
    // [hidden, hidden]
    TxtQWeight: TSingles;
    TxtKWeight: TSingles;
    TxtVWeight: TSingles;
    TxtQWeightBF16: TBF16s;
    TxtKWeightBF16: TBF16s;
    TxtVWeightBF16: TBF16s;

    // [head_dim] - QK norm on Q and K
    TxtNormQWeight: TSingles;
    TxtNormKWeight: TSingles;

    // [hidden, hidden]
    TxtProjWeight: TSingles;
    TxtProjWeightBF16: TBF16s;

    // [mlp_hidden, hidden]
    TxtMlpGateWeight: TSingles;
    TxtMlpGateWeightBF16: TBF16s;
    TxtMlpUpWeight: TSingles;
    TxtMlpUpWeightBF16: TBF16s;

    // [hidden, mlp_hidden]
    TxtMlpDownWeight: TSingles;
    TxtMlpDownWeightBF16: TBF16s;
    Ismmap: Boolean;

    procedure Load(const SafeTensorFiles: TSafeTensorFiles; const Idx, H, Mlp: LongInt; const UseBF16: Boolean = True; const AUseMMAP: Boolean = True; const resident: Boolean = False
    );
    procedure Free;
  end;

  // Single-stream block (Parallel DiT style, fused)
  PSingleBlock = ^TSingleBlock;

  { TSingleBlock }
  TSingleBlock = record
    // Fused QKV + FFN input projection
    // [hidden*3 + mlp_hidden*2, hidden]
    QKVMlpWeight: TSingles;
    QKVMlpWeightBF16: TBF16s;

    // QK normalization
    // [head_dim]
    NormQWeight: TSingles;
    NormKWeight: TSingles;

    // Fused attention out + FFN down projection
    // [hidden, hidden + mlp_hidden]
    ProjMlpWeight: TSingles;
    ProjMlpWeightBF16: TBF16s;
    procedure Load(const SafeTensorFiles: TSafeTensorFiles; const Idx, H, Mlp: LongInt; const UseBF16: Boolean = True; const AUseMMAP: Boolean = True; const resident: Boolean = False
    );
    procedure Free;
  end;

  (* Timestep embedding MLP
   * FLUX.2-klein uses 256-dim sinusoidal embedding (128 frequencies)
   * linear_1: [hidden, 256] - projects sinusoidal to hidden
   * linear_2: [hidden, hidden] - another linear layer *)

  { TTimeEmbed }
  PTimeEmbed = ^TTimeEmbed;

  TTimeEmbed = record
    Fc1Weight: TSingles; // [hidden, 256]
    Fc2Weight: TSingles; // [hidden, hidden]
    SincosDim: LongInt; // 256 for FLUX.2-klein
    procedure forward(const Dst, SinCos: TMemoryBlock; const Hidden: LongInt; const OutSilu: TMemoryBlock);
    procedure Free();
  end;

const
  MAX_TF_SHARDS = 4;

type

  // Full transformer context

  { TTransformerFlux }
  PTransformerFlux = ^TTransformerFlux;

  TTransformerFlux = record
    const
      CONFIG_FILE = '/transformer/config.json';
  public
    HiddenSize: LongInt; // 3072
    NumHeads: LongInt; // 24
    HeadDim: LongInt; // 128
    MlpHidden: LongInt; // hidden * 3 = 9216
    NumDoubleLayers: LongInt; // 5
    NumSingleLayers: LongInt; // 20
    TextDim: LongInt; // 7680
    LatentChannels: LongInt; // 128
    RoPETheta: Single; // 2000
    RoPEDim: LongInt; // 128
    UseBF16: Boolean; // Use bf16 weights (1) or f32 (0)

    // Input projections
    ImgInWeight: TSingles; // [hidden, latent_channels]
    TxtInWeight: TSingles; // [hidden, text_dim]
    ImgInWeightBF16: TBF16s; // [hidden, latent_channels]
    TxtInWeightBF16: TBF16s; // [hidden, text_dim]

    TimeEmbed: TTimeEmbed;
    TimeFreq: TSingles; // [hidden/2]
    AdalnDoubleImgWeight: TSingles; // [hidden * 6, hidden] for double block img stream
    AdalnDoubleImgWeightBF16: TBF16s;
    AdalnDoubleTxtWeight: TSingles; // [hidden * 6, hidden] for double block txt stream
    AdalnDoubleTxtWeightBF16: TBF16s;
    AdalnSingleWeight: TSingles; // [hidden * 3, hidden] for single block
    AdalnSingleWeightBF16: TBF16s;

    // Transformer block
    DoubleBlocks: TArray<TDoubleBlock>;
    SingleBlocks: TArray<TSingleBlock>;

    FinalNormWeight: TSingles; // [hidden]
    FinalProjWeight: TSingles; // [latent_channels, hidden]
    FinalProjWeightBF16: TBF16s;

    RoPEFreqs: TSingles; // [max_seq, head_dim/2, 2]  - legacy 1D
    RoPECos: TSingles; // [max_seq, axis_dim] - 2D cos frequencies
    RoPESin: TSingles; // [max_seq, axis_dim] - 2D sin frequencies

    MaxSeqLen: LongInt;
    AxisDim: LongInt; // 32 for FLUX (128 head_dim / 4 axes)

    ImgHidden: TSingles; // [max_img_seq, hidden]
    TxtHidden: TSingles; // [max_txt_seq, hidden]

    Q: TSingles; // [max_seq, hidden]
    K: TSingles;
    V: TSingles;

    AttnOut: TSingles; // [max_seq, hidden]
    MlpBuffer: TSingles; // [max_seq, mlp_hidden]

    Work1: TSingles;
    Work2: TSingles;
    WorkSize: NativeInt;

    // Pre-allocated attention workspaces to avoid malloc in hot pat
    AttnQT: TSingles; //  [max_seq, hidden] transposed Q
    AttnKT: TSingles; //  [max_seq, hidden] transposed K
    AttnVT: TSingles; //  [max_seq, hidden] transposed V
    AttnOutT: TSingles; //  [max_seq, hidden] transposed output
    AttnScores: TSingles; //  [num_heads, seq, seq] attention scores (BLAS/Metal only)
    AttnScoresAlloc: Int64; //  Currently allocated size
    WorkSeqAlloc: LongInt; //  Currently allocated sequence length for work buffers
    AttnCatK: TSingles; //  [max_seq, hidden] concatenated K
    AttnCatV: TSingles; //  [max_seq, hidden] concatenated V

    // Single-block work buffers (pre-allocated to avoid malloc in hot path)
    SingleQ: TSingles; // [max_seq, hidden]
    SingleK: TSingles; // [max_seq, hidden]
    SingleV: TSingles; // [max_seq, hidden]
    SingleMlpGate: TSingles; // [max_seq, mlp_hidden]
    SingleMlpUp: TSingles; // [max_seq, mlp_hidden]
    SingleAttnOut: TSingles; // [max_seq, hidden]
    SingleConcat: TSingles; // [max_seq, hidden + mlp_hidden]

    // FFN work buffers (shared by double and single blocks)
    FfnGate: TSingles; // [max_seq, mlp_hidden]
    FfnUp: TSingles; // [max_seq, mlp_hidden]

    // Double-block work buffers
    TEmbSilu: TSingles; // [hidden]
    DoubleModImg: TSingles; // [hidden * 6]
    DoubleModTxt: TSingles; // [hidden * 6]
    DoubleImgAttnOut: TSingles; // [max_seq, hidden]
    DoubleTxtAttnOut: TSingles; // [max_seq, hidden]

    // Cached 2D RoPE embeddings (to avoid malloc/compute each step)
    CachedImgRoPECos: TSingles; // [img_seq * axis_dim * 4]
    CachedImgRoPESin: TSingles;
    CachedImgH: LongInt;
    CachedImgW: LongInt;
    CachedRefRoPECos: TSingles; // [ref_seq * axis_dim * 4] for img2img
    CachedRefRoPESin: TSingles;
    CachedRefH: LongInt;
    CachedRefW: LongInt;
    CachedRefTOffset: LongInt;

    CachedTxtRoPECos: TSingles; // [txt_seq * head_dim] */
    CachedTxtRoPESin: TSingles;
    CachedTxtSeq: LongInt;

    // Combined RoPE cache for img2img (target + reference)
    CachedCombinedRoPECos: TSingles; // [(img_seq + ref_seq) * axis_dim * 4]
    CachedCombinedRoPESin: TSingles;
    CachedCombinedImgH: LongInt;
    CachedCombinedImgW: LongInt;
    CachedCombinedRefH: LongInt;
    CachedCombinedRefW: LongInt;
    CachedCombinedTOffset: LongInt;

    UseMmap, ResidentOnly: Boolean;
    SfFiles: TSafeTensorFiles;
    NumSfFiles: LongInt;
    Eps: Single;
    class function Parse(const ModelDir: string): TTransformerFlux; static;
    constructor Load(const ModelDir: string; const AUseMMAP: Boolean = True; const resident: Boolean = False);
    function IsLoaded(): Boolean;
    procedure ReInitialize(const TotalSeq: LongInt);
    procedure Free();
    procedure FreeMMapCache();
    procedure GetCachedRefRoPE(const PatchH, PatchW, TOffset: LongInt; var CosOut: TSingles; var SinOut: TSingles);
    procedure GetCachedImgRoPE(const PatchH, PatchW: LongInt; var CosOut: TSingles; var SinOut: TSingles);
    procedure GetCachedTxtRoPE(const TxtSeq: LongInt; var CosOut: TSingles; var SinOut: TSingles);
    procedure GetCachedCombinedRoPE(const ImgH, ImgW, RefH, RefW, TOffset: LongInt; var CosOut: TSingles; var SinOut: TSingles);
    procedure MultiHeadForward(const Dst, Q, K, V: TMemoryBlock; const Seq {, heads, head_dim}: LongInt);
    procedure JointAttentionForward(const ImgOut, TxtOut, ImgQ, ImgK, ImgV, TxtQ, TxtK, TxtV: TMemoryBlock; const ImgSeq, TxtSeq {, heads, head_dim}: LongInt);
    procedure FfnSwigluForward(const Dst, X: TMemoryBlock; const GateWeight, UpWeight, DownWeight: TSingles; const GateWeightBF16, UpWeightBF16, DownWeightBF16: TBF16s; const Seq, Hidden, MlpHidden: LongInt);

    procedure DoubleBlockForward(const ImgHidden, TxtHidden: TMemoryBlock; const BlockIdx: LongInt; const ImgMod, TxtMod, ImgRoPECos, ImgRoPESin, TxtRoPECos, TxtRoPESin: TMemoryBlock; const ImgSeq, TxtSeq: LongInt);

    procedure SingleBlockForward(const Hidden: TMemoryBlock; const BlockIdx: LongInt; const TEmb, AdalnWeight, ImgRoPECos, ImgRoPESin, TxtRoPECos, TxtRoPESin: TMemoryBlock; const Seq, ImgOffset: LongInt);

    // for text to image
    function forward(const ImgLatent: TMemoryBlock; const ImgH, ImgW: LongInt; const TxtEmb: TMemoryBlock; const TxtSeq: LongInt; Timestep: QNNFloat): TMemoryBlock; overload;
    // for reference image with text to image?
    function forward(const ImgLatent: TMemoryBlock; const ImgH, ImgW: LongInt; const RefLatent: TMemoryBlock; const RefH, RefW, TOffset: LongInt; const TxtEmb: TMemoryBlock; const TxtSeq: LongInt; Timestep: QNNFloat): TMemoryBlock; overload;
    // for multi-reference images with text to image
    function forward(const ImgLatent: TMemoryBlock; const ImgH, ImgW: LongInt; const Refs: TArray<TImageRef>; const TxtEmb: TMemoryBlock; const TxtSeq: LongInt; const Timestep: Single): TMemoryBlock; overload;

    // simplest for distilled
    function SampleEuler(const Z: TMemoryBlock; const Batch, Channels, H, W: LongInt; const TextEmb: TMemoryBlock; const TextSeq: LongInt; const Schedule: TMemoryBlock; const NumSteps: LongInt; const ProgressCallback: TProgressCallback): TMemoryBlock; overload;

    // simplist distilled with one latent image
    function SampleEuler(const Z: TMemoryBlock; const Batch, Channels, H, W: LongInt; const RefLatent: TMemoryBlock; const RefH, RefW, TOffset: LongInt; const TextEmb: TMemoryBlock; const TextSeq: LongInt; const Schedule: TMemoryBlock; const NumSteps: LongInt; const ProgressCallback: TProgressCallback): TMemoryBlock; overload;

    // non distilled with uncoditioning and guidance
    function SampleEuler(const Z: TMemoryBlock; const Batch, Channels, H, W: LongInt; const TextEmbCond: TMemoryBlock; const TextSeqCond: LongInt; const TextEmbUncond: TMemoryBlock; const TextSeqUncond: LongInt; const GuidanceScale: QNNFloat; const Schedule: TMemoryBlock; const NumSteps: LongInt; const ProgressCallback: TProgressCallback): TMemoryBlock; overload;

    // non distiled with unconditioning, guidance and one latent image
    function SampleEuler(const Z: TMemoryBlock; const Batch, Channels, H, W: LongInt; const RefLatent: TMemoryBlock; const RefH, RefW, TOffset: LongInt; const TextEmbCond: TMemoryBlock; const TextSeqCond: LongInt; const TextEmbUncond: TMemoryBlock; const TextSeqUncond: LongInt; const GuidanceScale: QNNFloat; const Schedule: TMemoryBlock; const NumSteps: LongInt; const ProgressCallback: TProgressCallback): TMemoryBlock; overload;

    // simplest disilled with multi images
    function SampleEuler(const Z: TMemoryBlock; const Batch, Channels, H, W: LongInt; const Refs: TArray<TImageRef>; const TextEmb: TMemoryBlock; const TextSeq: LongInt; const Schedule: TMemoryBlock; const NumSteps: LongInt; const ProgressCallback: TProgressCallback): TMemoryBlock; overload;

    // non disilled with unconditioning, gudance and multi images
    function SampleEuler(const Z: TMemoryBlock; const Batch, Channels, H, W: LongInt; const Refs: TArray<TImageRef>; const TextEmbCond: TMemoryBlock; const TextSeqCond: LongInt; const TextEmbUncond: TMemoryBlock; const TextSeqUncond: LongInt; const GuidanceScale: QNNFloat; const Schedule: PQNNFloat; NumSteps: LongInt; const ProgressCallback: TProgressCallback): TMemoryBlock; overload;
  end;

const
  ZI_SEQ_MULTI_OF = 32; (* Pad sequences to multiples of 32 *)
  ZI_NORM_EPS = 1e-5; (* RMSNorm epsilon *)
  ZI_BF16_SDPA_SEQ = 1024; (* Prefer bf16 SDPA at large sequence lengths *)
  ZI_MAX_SHARDS = 32;

type
  PBlockZI = ^TBlockZI;

  { TBlockZi }

  TBlockZI = record
    (* Attention *)
    AttnQWeight: TSingles; (* [dim, dim] *)
    AttnKWeight: TSingles; (* [dim, dim] *)
    AttnVWeight: TSingles; (* [dim, dim] *)
    AttnOutWeight: TSingles; (* [dim, dim] *)
    AttnNormQ: TSingles; (* [n_heads, head_dim] for QK norm *)
    AttnNormK: TSingles; (* [n_heads, head_dim] *)
    AttnNorm1: TSingles; (* [dim] RMSNorm before attention *)
    AttnNorm2: TSingles; (* [dim] RMSNorm after attention *)

    (* FFN (SwiGLU) *)
    FfnW1: TSingles; (* [ffn_dim, dim] gate projection *)
    FfnW2: TSingles; (* [dim, ffn_dim] down projection *)
    FfnW3: TSingles; (* [ffn_dim, dim] up projection *)
    FfnNorm1: TSingles; (* [dim] RMSNorm before FFN *)
    FfnNorm2: TSingles; (* [dim] RMSNorm after FFN *)

    (* AdaLN modulation (NULL for context_refiner blocks) *)
    AdalnWeight: TSingles; (* [4*dim, adaln_dim] *)
    AdalnBias: TSingles; (* [4*dim] *)
    procedure Load(const Files: TSafeTensorFiles; const Prefix: string; const HasModulation: Boolean; const AUseMMAP: Boolean = True; const resident: Boolean = False
    );
    procedure Free();
  end;

  (* Final layer weights *)
  PFinalZI = ^TFinalZI;

  TFinalZI = record
    AdalnWeight: TSingles; (* [dim, adaln_dim] *)
    AdalnBias: TSingles; (* [dim] *)
    NormWeight: TSingles; (* NULL (no affine) or [dim] *)
    LinearWeight: TSingles; (* [out_ch, dim] *)
    LinearBias: TSingles; (* [out_ch] *)
  end;

  (* Z-Image transformer context *)

  { TTransformerZi }

  TTransformerZI = record
    (* Architecture config *)
    Dim: LongInt; (* 3840 *)
    NHeads: LongInt; (* 30 *)
    HeadDim: LongInt; (* 128 *)
    NLayers: LongInt; (* 30 *)
    NRefiner: LongInt; (* 2 *)
    FfnDim: LongInt; (* 8*dim/3 = 10240 *)
    InChannels: LongInt; (* 16 *)
    PatchSize: LongInt; (* 2 *)
    AdalnDim: LongInt; (* min(dim, 256) = 256 *)
    RoPETheta: Single; (* 256.0 *)
    AxesDims: array[0..2] of LongInt; (* [32, 48, 48] *)
    AxesLens: array[0..2] of LongInt; (* [1024, 512, 512] *)

    (* Embedders *)
    TEmbMlp0Weight: TSingles; (* [mid_size, 256] *)
    TEmbMlp0Bias: TSingles; (* [mid_size] *)
    TEmbMlp2Weight: TSingles; (* [adaln_dim, mid_size] *)
    TEmbMlp2Bias: TSingles; (* [adaln_dim] *)
    TEmbMidSize: LongInt; (* intermediate timestep MLP size *)

    CapEmbNorm: TSingles; (* [cap_feat_dim] RMSNorm weight *)
    CapEmbLinearW: TSingles; (* [dim, cap_feat_dim] *)
    CapEmbLinearB: TSingles; (* [dim] *)
    CapFeatDim: LongInt; (* 2560 *)

    XEmbWeight: TSingles; (* [dim, patch_feat] where patch_feat = ps*ps*in_ch *)
    XEmbBias: TSingles; (* [dim] *)

    XPadToken: TSingles; (* [dim] *)
    CapPadToken: TSingles; (* [dim] *)

    (* Transformer blocks *)
    NoiseRefiner: TArray<TBlockZI>; (* [n_refiner] *)
    ContextRefiner: TArray<TBlockZI>; (* [n_refiner] *)
    Layers: TArray<TBlockZI>; (* [n_layers] *)

    (* Final layer *)
    FinalLayer: TFinalZI;

    (* CPU mmap mode: keep shard files open and use direct f32 pointers. *)
    MmapF32Weights, ResidentOnly: Boolean;
    SfFiles: TSafeTensorFiles;
    NumSfFiles: LongInt;

    (* Precomputed RoPE frequencies (complex pairs) *)
    RoPECos: array[0..2] of TSingles; (* [axes_lens[i], axes_dims[i]/2] *)
    RoPESin: array[0..2] of TSingles; (* [axes_lens[i], axes_dims[i]/2] *)

    (* Working memory *)
    WorkX: TSingles; (* Main token buffer *)
    WorkTmp: TSingles; (* Temporary buffer *)
    WorkQKV: TSingles; (* Q, K, V buffers *)
    WorkAttn: TSingles; (* Attention scores *)
    WorkFfn: TSingles; (* FFN intermediate *)
    WorkAlloc: NativeInt; (* Total allocated *)
    MaxSeq: LongInt; (* Max sequence length allocated for *)
    procedure AttentionForward(const Dst, Src: TMemoryBlock; const Block: TBlockZI; const PosIds: PLongint; const Mask: PLongint; const Seq: LongInt);
    procedure FfnForward(const Dst, Src: TMemoryBlock; const Block: TBlockZI; const Seq: LongInt);
    procedure BlockForward(const Dst: TMemoryBlock; const Block: TBlockZI; const PosIds: PLongint; const Mask: PLongint; const TEmb: TMemoryBlock; const Seq: LongInt);
    procedure PreComputeRope();
    procedure FinalComputeScale(const Scale, TEmb: TMemoryBlock);
    procedure FinalForward(const Dst, Src, TEmb: TMemoryBlock; const Seq: LongInt);
    function forward(const Latent: TMemoryBlock; const LatentH, LatentW: LongInt; const Timestep: QNNFloat; const CapFeats: TMemoryBlock; const CapSeqLen: LongInt): TMemoryBlock;
    procedure ApplyRope(const Dst: PQNNFloat; const PosIds: PLongint; const Seq, NHeads: LongInt);
    procedure TimeStepEmbed(const Dst: TMemoryBlock; const T: QNNFloat);
    function SampleEuler(const Z: TMemoryBlock; const Batch {, channels}, H, W, PatchSize: LongInt; const CapFeats: TMemoryBlock; const CapSeq: LongInt; const Schedule: PQNNFloat; const NumSteps: LongInt; const ProgressCallback: TStepCallback = nil
    ): TMemoryBlock;
    procedure Load(const ModelDir: string; const Adim, AnHeads, AnLayers, AnRefiner, AcapFeatDim, AinChannels, ApatchSize: LongInt; const AropeTheta: QNNFloat; const AAxesDims: PLongint; const AUseMMAP: Boolean = True; const resident: Boolean = False
    );
    function IsLoaded(): Boolean;
    procedure Free();
  end;

procedure PatchifyZi(const Dst: PQNNFloat; const Src: PQNNFloat; const InCh, Height, Width, PatchSize: LongInt);

procedure UnPatchifyZi(const Dst: PQNNFloat; const Src: PQNNFloat; const InCh, Height, Width, PatchSize: LongInt);

function LoadShards(const ModelDir: string): TSafeTensorFiles;

implementation

uses
  QuickNN.Kernels, QuickNN.Vulkan, System.IOUtils;

function LoadShards(const ModelDir: string): TSafeTensorFiles;
begin
  Result :=
    LoadSafeTensorShards(
    TPath.Combine(ModelDir, 'transformer'),
    'diffusion_pytorch_model.safetensors.index.json',
    'diffusion_pytorch_model.safetensors'
  );
end;

{ TDoubleBlock }

procedure TDoubleBlock.Load(const SafeTensorFiles: TSafeTensorFiles; const Idx, H, Mlp: LongInt; const UseBF16: Boolean; const AUseMMAP: Boolean; const resident: Boolean);
const
  TRANSFORMER_BLOCK: array[0..15] of string = (
    'transformer_blocks.%d.attn.norm_q.weight',
    'transformer_blocks.%d.attn.norm_k.weight',
    'transformer_blocks.%d.attn.to_q.weight',
    'transformer_blocks.%d.attn.to_k.weight',
    'transformer_blocks.%d.attn.to_v.weight',
    'transformer_blocks.%d.attn.to_out.0.weight',
    'transformer_blocks.%d.ff.linear_in.weight',
    'transformer_blocks.%d.ff.linear_out.weight',
    'transformer_blocks.%d.attn.norm_added_q.weight',
    'transformer_blocks.%d.attn.norm_added_k.weight',
    'transformer_blocks.%d.attn.add_q_proj.weight',
    'transformer_blocks.%d.attn.add_k_proj.weight',
    'transformer_blocks.%d.attn.add_v_proj.weight',
    'transformer_blocks.%d.attn.to_add_out.weight',
    'transformer_blocks.%d.ff_context.linear_in.weight',
    'transformer_blocks.%d.ff_context.linear_out.weight'
  );

  function Projection(const Name: string; const Mapped: Boolean): TMemoryBlock;
  begin
    if resident then
      Result := SafeTensorFiles.GetLinearWeight(Name, Mapped)
    else
      Result := SafeTensorFiles.GetTensorDataMemBlock(Name, Mapped);
  end;

begin
  // as in forward order 1st do the norm q and k weights;

  ImgNormQWeight := SafeTensorFiles.GetTensorDataMemBlock(Format(TRANSFORMER_BLOCK[0], [Idx]), AUseMMAP);
  ImgNormKWeight := SafeTensorFiles.GetTensorDataMemBlock(Format(TRANSFORMER_BLOCK[1], [Idx]), AUseMMAP);
  Ismmap := AUseMMAP;
  if UseBF16 then
  begin
    (* Image Q, K, V projections - skip f32 if bf16 available *)
    ImgQWeightBF16 := SafeTensorFiles.GetTensorDataMemBlockBF16(Format(TRANSFORMER_BLOCK[2], [Idx]), AUseMMAP);
    ImgKWeightBF16 := SafeTensorFiles.GetTensorDataMemBlockBF16(Format(TRANSFORMER_BLOCK[3], [Idx]), AUseMMAP);
    ImgVWeightBF16 := SafeTensorFiles.GetTensorDataMemBlockBF16(Format(TRANSFORMER_BLOCK[4], [Idx]), AUseMMAP);
    ImgProjWeightBF16 := SafeTensorFiles.GetTensorDataMemBlockBF16(Format(TRANSFORMER_BLOCK[5], [Idx]), AUseMMAP);
    (* Image FFN - linear_in contains gate and up fused - skip f32 if bf16 available *)
    ImgMlpGateWeightBF16 := SafeTensorFiles.GetTensorDataMemBlockBF16(Format(TRANSFORMER_BLOCK[6], [Idx]), AUseMMAP);
    Assert(
      AUseMMAP or (ImgMlpGateWeightBF16.Count >= Mlp * H * 2),
      'ERROR : incorrect size of ' + Format(TRANSFORMER_BLOCK[6], [Idx])
    );
    ImgMlpUpWeightBF16 := ImgMlpGateWeightBF16 + Mlp * H;
    ImgMlpDownWeightBF16 := SafeTensorFiles.GetTensorDataMemBlockBF16(Format(TRANSFORMER_BLOCK[7], [Idx]), AUseMMAP);
    // during forward attention norm added q and v should be here,
    // this is just loading so will be loaded after this block
    // txt_norm_q_weight := safeTensorFiles.getTensorDataMemBlock(format(TRANSFORMER_BLOCK[8], [idx]), use_mmap);
    // txt_norm_k_weight := safeTensorFiles.getTensorDataMemBlock(format(TRANSFORMER_BLOCK[9], [idx]), use_mmap);

    TxtQWeightBF16 := SafeTensorFiles.GetTensorDataMemBlockBF16(Format(TRANSFORMER_BLOCK[10], [Idx]), AUseMMAP);
    TxtKWeightBF16 := SafeTensorFiles.GetTensorDataMemBlockBF16(Format(TRANSFORMER_BLOCK[11], [Idx]), AUseMMAP);
    TxtVWeightBF16 := SafeTensorFiles.GetTensorDataMemBlockBF16(Format(TRANSFORMER_BLOCK[12], [Idx]), AUseMMAP);
    (* Text Q, K, V projections - skip f32 if bf16 available *)
    TxtProjWeightBF16 := SafeTensorFiles.GetTensorDataMemBlockBF16(Format(TRANSFORMER_BLOCK[13], [Idx]), AUseMMAP);
    TxtMlpGateWeightBF16 := SafeTensorFiles.GetTensorDataMemBlockBF16(Format(TRANSFORMER_BLOCK[14], [Idx]), AUseMMAP);
    Assert(
      AUseMMAP or (TxtMlpGateWeightBF16.Count >= Mlp * H * 2),
      'ERROR : incorrect size of ' + Format(TRANSFORMER_BLOCK[14], [Idx])
    );
    TxtMlpUpWeightBF16 := TxtMlpGateWeightBF16 + Mlp * H;
    TxtMlpDownWeightBF16 := SafeTensorFiles.GetTensorDataMemBlockBF16(Format(TRANSFORMER_BLOCK[15], [Idx]), AUseMMAP)
  end
  else
  begin
    (* Image Q, K, V projections - skip f32 if bf16 available *)
    ImgQWeight := Projection(Format(TRANSFORMER_BLOCK[2], [Idx]), AUseMMAP);
    ImgKWeight := Projection(Format(TRANSFORMER_BLOCK[3], [Idx]), AUseMMAP);
    ImgVWeight := Projection(Format(TRANSFORMER_BLOCK[4], [Idx]), AUseMMAP);
    ImgProjWeight := Projection(Format(TRANSFORMER_BLOCK[5], [Idx]), AUseMMAP);
    (* Image FFN - linear_in contains gate and up fused - skip f32 if bf16 available *)
    ImgMlpGateWeight := Projection(Format(TRANSFORMER_BLOCK[6], [Idx]), AUseMMAP);
    Assert(
      AUseMMAP or (ImgMlpGateWeight.Count >= Mlp * H * 2),
      'ERROR : incorrect size of ' + Format(TRANSFORMER_BLOCK[6], [Idx])
    );
    ImgMlpUpWeight := ImgMlpGateWeight + Mlp * H;
    ImgMlpDownWeight := Projection(Format(TRANSFORMER_BLOCK[7], [Idx]), AUseMMAP);
    // during forward attention norm added q and v should be here,
    // this is just loading so will be loaded after this block
    // txt_norm_q_weight := safeTensorFiles.getTensorDataMemBlock(format(TRANSFORMER_BLOCK[8], [idx]), use_mmap);
    // txt_norm_k_weight := safeTensorFiles.getTensorDataMemBlock(format(TRANSFORMER_BLOCK[9], [idx]), use_mmap);

    TxtQWeight := Projection(Format(TRANSFORMER_BLOCK[10], [Idx]), AUseMMAP);
    TxtKWeight := Projection(Format(TRANSFORMER_BLOCK[11], [Idx]), AUseMMAP);
    TxtVWeight := Projection(Format(TRANSFORMER_BLOCK[12], [Idx]), AUseMMAP);
    (* Text Q, K, V projections - skip f32 if bf16 available *)
    TxtProjWeight := Projection(Format(TRANSFORMER_BLOCK[13], [Idx]), AUseMMAP);
    TxtMlpGateWeight := Projection(Format(TRANSFORMER_BLOCK[14], [Idx]), AUseMMAP);
    Assert(
      AUseMMAP or (TxtMlpGateWeight.Count >= Mlp * H * 2),
      'ERROR : incorrect size of ' + Format(TRANSFORMER_BLOCK[14], [Idx])
    );
    TxtMlpUpWeight := TxtMlpGateWeight + Mlp * H;
    TxtMlpDownWeight := Projection(Format(TRANSFORMER_BLOCK[15], [Idx]), AUseMMAP)
  end;

  (* Text stream - QK norm weights (always f32) *)
  TxtNormQWeight :=
    SafeTensorFiles.GetTensorDataMemBlock(Format(TRANSFORMER_BLOCK[8], [Idx]), False {use_mmap}); // always allocate
  TxtNormKWeight :=
    SafeTensorFiles.GetTensorDataMemBlock(Format(TRANSFORMER_BLOCK[9], [Idx]), False {use_mmap}); // always allocate
end;

procedure TSingleBlock.Load(const SafeTensorFiles: TSafeTensorFiles; const Idx, H, Mlp: LongInt; const UseBF16: Boolean; const AUseMMAP: Boolean; const resident: Boolean);
const
  TRANSFORMER_BLOCK: array[0..3] of string = (
    'single_transformer_blocks.%d.attn.norm_q.weight',
    'single_transformer_blocks.%d.attn.norm_k.weight',
    'single_transformer_blocks.%d.attn.to_qkv_mlp_proj.weight',
    'single_transformer_blocks.%d.attn.to_out.weight'
  );

  function Projection(const Name: string; const Mapped: Boolean): TMemoryBlock;
  begin
    if resident then
      Result := SafeTensorFiles.GetLinearWeight(Name, Mapped)
    else
      Result := SafeTensorFiles.GetTensorDataMemBlock(Name, Mapped);
  end;

begin

  (* QK norm weights (always f32, small) *)
  NormQWeight := SafeTensorFiles.GetTensorDataMemBlock(Format(TRANSFORMER_BLOCK[0], [Idx]), AUseMMAP);
  NormKWeight := SafeTensorFiles.GetTensorDataMemBlock(Format(TRANSFORMER_BLOCK[1], [Idx]), AUseMMAP);

  if UseBF16 then
  begin
    (* Fused QKV+MLP input projection - skip f32 if bf16 available *)
    QKVMlpWeightBF16 := SafeTensorFiles.GetTensorDataMemBlockBF16(Format(TRANSFORMER_BLOCK[2], [Idx]), AUseMMAP);
    (* Fused attn out + MLP down projection - skip f32 if bf16 available *)
    ProjMlpWeightBF16 := SafeTensorFiles.GetTensorDataMemBlockBF16(Format(TRANSFORMER_BLOCK[3], [Idx]), AUseMMAP);
  end
  else
  begin
    QKVMlpWeight := Projection(Format(TRANSFORMER_BLOCK[2], [Idx]), AUseMMAP);
    ProjMlpWeight := Projection(Format(TRANSFORMER_BLOCK[3], [Idx]), AUseMMAP);
  end;
end;

procedure TDoubleBlock.Free;
begin
  ImgNormQWeight.Free;
  ImgNormKWeight.Free;
  ImgQWeight.Free;
  ImgKWeight.Free;
  ImgVWeight.Free;
  ImgProjWeight.Free;
  ImgMlpGateWeight.Free;
  ImgMlpUpWeight.Free;
  ImgMlpDownWeight.Free;
  //
  ImgQWeightBF16.Free;
  ImgKWeightBF16.Free;
  ImgVWeightBF16.Free;
  ImgProjWeightBF16.Free;
  ImgMlpGateWeightBF16.Free;
  ImgMlpUpWeightBF16.Free;
  ImgMlpDownWeightBF16.Free;

  TxtNormQWeight.Free;
  TxtNormKWeight.Free;
  TxtQWeight.Free;
  TxtKWeight.Free;
  TxtVWeight.Free;
  TxtProjWeight.Free;
  TxtMlpGateWeight.Free;
  TxtMlpUpWeight.Free;
  TxtMlpDownWeight.Free;

  TxtQWeightBF16.Free;
  TxtKWeightBF16.Free;
  TxtVWeightBF16.Free;
  TxtProjWeightBF16.Free;
  TxtMlpGateWeightBF16.Free;
  TxtMlpUpWeightBF16.Free;
  TxtMlpDownWeightBF16.Free;
  Self := Default(TDoubleBlock);
end;

procedure TSingleBlock.Free;
begin
  NormQWeight.Free;
  NormKWeight.Free;
  QKVMlpWeight.Free;
  ProjMlpWeight.Free;
  // (* bf16 pointers are direct mmap pointers - just clear, don't free *)
  //
  QKVMlpWeightBF16.Free;
  ProjMlpWeightBF16.Free;

  Self := Default(TSingleBlock);
end;

{ TTimeEmbed }

procedure TTimeEmbed.forward(const Dst, SinCos: TMemoryBlock; const Hidden: LongInt; const OutSilu: TMemoryBlock);
begin
  // if length(workspace)< 1*hidden then
  //  setLength(workspace, hidden);

  QNNLinearNoBias(OutSilu, SinCos, Fc1Weight, 1, SincosDim, Hidden);
  QNNSiluInplace(OutSilu, Hidden);
  QNNLinearNoBias(Dst, OutSilu, Fc2Weight, 1, Hidden, Hidden);
end;

procedure TTimeEmbed.Free();
begin
  Fc1Weight.Free; // [hidden, 256]
  Fc2Weight.Free; // [hidden, hidden]
  Self := Default(TTimeEmbed)
end;

{ TTransformerFlux }

class function TTransformerFlux.Parse(const ModelDir: string): TTransformerFlux;
var
  JSON: TJSONValue;
  HeadsCount, LocalHeadDim, LayerCount, SingleCount, JointAttDim, InChannels: LongInt;
  MlpRatio, LocalRopeTheta: Single;
// eps : single;
begin
  Result := Default(TTransformerFlux);
  JSON := TJSONValue.ParseJSONValue(TFile.ReadAllText(ModelDir + CONFIG_FILE));
  HeadsCount := JSON.GetValue<LongInt>('num_attention_heads');
  LocalHeadDim := JSON.GetValue<LongInt>('attention_head_dim');
  LayerCount := JSON.GetValue<LongInt>('num_layers');
  SingleCount := JSON.GetValue<LongInt>('num_single_layers');
  JointAttDim := JSON.GetValue<LongInt>('joint_attention_dim');
  InChannels := JSON.GetValue<LongInt>('in_channels');
  MlpRatio := JSON.GetValue<Single>('mlp_ratio');
  LocalRopeTheta := JSON.GetValue<Single>('rope_theta');
  Result.Eps := JSON.GetValue<Single>('eps');
  JSON.Free;

  Result.NumHeads := HeadsCount;
  Result.HeadDim := LocalHeadDim;
  Result.HiddenSize := HeadsCount * LocalHeadDim;

  if MlpRatio > 0 then
    Result.MlpHidden := Trunc(Result.HiddenSize * MlpRatio)
  else
    Result.MlpHidden := Result.HiddenSize * 3;

  Result.NumDoubleLayers := QuickNN.Common.Ifthen(LayerCount > 0, LayerCount, 5);
  Result.NumSingleLayers := QuickNN.Common.Ifthen(SingleCount > 0, SingleCount, 20);
  Result.TextDim := QuickNN.Common.Ifthen(JointAttDim > 0, JointAttDim, 7680);
  Result.LatentChannels := QuickNN.Common.Ifthen(InChannels > 0, InChannels, 128);

  if LocalRopeTheta > 0 then
    Result.RoPETheta := LocalRopeTheta
  else
    Result.RoPETheta := 2000.0;

  Result.RoPEDim := LocalHeadDim;
  Result.AxisDim := LocalHeadDim div 4; // 4 RoPE axes

  Result.SfFiles := LoadShards(ModelDir);
end;

constructor TTransformerFlux.Load(const ModelDir: string; const AUseMMAP: Boolean; const resident: Boolean);
var
  Cfgfile: string;

  function Projection(const Name: string; const Mapped: Boolean): TMemoryBlock;
  begin
    if resident then
      Result := SfFiles.GetLinearWeight(Name, Mapped)
    else
      Result := SfFiles.GetTensorDataMemBlock(Name, Mapped);
  end;

begin
  Self := Default(TTransformerFlux);
  Cfgfile := ModelDir;
  if FileExists(Cfgfile + CONFIG_FILE) then
    Self := Parse(Cfgfile)
  else
  begin
    HiddenSize := 3072;
    NumHeads := 24;
    HeadDim := 128;
    MlpHidden := 9216;
    NumDoubleLayers := 5;
    NumSingleLayers := 20;
    TextDim := 7680;
    LatentChannels := 128;
    RoPETheta := 2000.0;
    RoPEDim := 128;
    AxisDim := 32;
  end;
  MaxSeqLen := 52000;
  UseMmap := AUseMMAP;
  ResidentOnly := resident;
  if Length(SfFiles) = 0 then
    SfFiles := LoadShards(ModelDir);

  ImgInWeight := Projection('x_embedder.weight', UseMmap);
  TxtInWeight := Projection('context_embedder.weight', UseMmap);
  if UseBF16 then
  begin
    ImgInWeightBF16 := SfFiles.GetTensorDataMemBlockBF16('x_embedder.weight', UseMmap);
    TxtInWeightBF16 := SfFiles.GetTensorDataMemBlockBF16('context_embedder.weight', UseMmap)
  end;
  TimeEmbed.SincosDim := 256;
  TimeEmbed.Fc1Weight := Projection('time_guidance_embed.timestep_embedder.linear_1.weight', UseMmap);
  TimeEmbed.Fc2Weight := Projection('time_guidance_embed.timestep_embedder.linear_2.weight', UseMmap);
  AdalnDoubleImgWeight := Projection('double_stream_modulation_img.linear.weight', UseMmap);
  AdalnDoubleTxtWeight := Projection('double_stream_modulation_txt.linear.weight', UseMmap);
  AdalnSingleWeight := Projection('single_stream_modulation.linear.weight', UseMmap);
  if UseBF16 then
  begin
    AdalnDoubleImgWeightBF16 :=
      SfFiles.GetTensorDataMemBlockBF16('double_stream_modulation_img.linear.weight', UseMmap);
    AdalnDoubleTxtWeightBF16 :=
      SfFiles.GetTensorDataMemBlockBF16('double_stream_modulation_txt.linear.weight', UseMmap);
    AdalnSingleWeightBF16 := SfFiles.GetTensorDataMemBlockBF16('single_stream_modulation.linear.weight', UseMmap)
  end;
  SetLength(DoubleBlocks, NumDoubleLayers);
  SetLength(SingleBlocks, NumSingleLayers);
  FinalNormWeight := Projection('norm_out.linear.weight', UseMmap);
  FinalProjWeight := Projection('proj_out.weight', UseMmap);
  if UseBF16 then
    FinalProjWeightBF16 := SfFiles.GetTensorDataMemBlockBF16('proj_out.weight', UseMmap);
  if resident then
    Exit;
  RoPEFreqs := TSingles.Create([MaxSeqLen, HeadDim], 'FLUX_LOAD_ROPE_FREQS');
  if RoPEFreqs.IsAssigned() then
    QNNComputeRoPE(RoPEFreqs, MaxSeqLen, HeadDim, RoPETheta);

  (* Small constant-size buffers - always allocate *)
  TEmbSilu := TSingles.Create([HiddenSize], 'FLUX_LOAD_t_emb_silu');
  DoubleModImg := TSingles.Create([HiddenSize, 6], 'FLUX_LOAD_double_mod_img');
  DoubleModTxt := TSingles.Create([HiddenSize, 6], 'FLUX_LOAD_double_mod_txt');
end;

function TTransformerFlux.IsLoaded(): Boolean;
begin
  Result := Assigned(SfFiles)
end;

procedure TTransformerFlux.ReInitialize(const TotalSeq: LongInt);
var
  FusedDim, Hidden, Mlp: LongInt;
  AttnScoresNeed: TArray<Int64>;
begin

  if IsUsingBlas then
  begin
    AttnScoresNeed := [NumHeads, TotalSeq, TotalSeq];

    if Product(AttnScoresNeed) > AttnScoresAlloc then
    begin
      if AttnScores.IsAssigned() then
        AttnScores.Resize(AttnScoresNeed, QNN_DEFAULT_DATATYPE)
      else
        AttnScores := TMemoryBlock.Create(AttnScoresNeed, 'FLUX_INIT_ATTN_SCORES');
      AttnScoresAlloc := Product(AttnScoresNeed);
    end;
  end;

  if TotalSeq <= WorkSeqAlloc then
    Exit;

  Hidden := HiddenSize;
  Mlp := MlpHidden;

  ImgHidden.Free;
  TxtHidden.Free;
  Work1.Free;
  Work2.Free;
  AttnQT.Free;
  AttnKT.Free;
  AttnVT.Free;
  AttnOutT.Free;
  AttnCatK.Free;
  AttnCatV.Free;
  SingleQ.Free;
  SingleK.Free;
  SingleV.Free;
  SingleMlpGate.Free;
  SingleMlpUp.Free;
  SingleAttnOut.Free;
  SingleConcat.Free;
  FfnGate.Free;
  FfnUp.Free;
  DoubleImgAttnOut.Free;
  DoubleTxtAttnOut.Free;

  ImgHidden := TMemoryBlock.Create([TotalSeq, Hidden], 'FLUX_INIT_IMG_HIDDEN');
  TxtHidden := TMemoryBlock.Create([TotalSeq, Hidden], 'FLUX_INIT_TXT_HIDDEN');
  (* work2 needs to hold fused QKV+MLP output: seq * (hidden*3 + mlp*2) + mod_params (hidden*3)
  * fused_dim = hidden*3 + mlp*2 = 3072*3 + 9216*2 = 27648 *)
  FusedDim := Hidden * 3 + Mlp * 2;

  WorkSize := TotalSeq * FusedDim + Hidden * 3;

  Work1 := TMemoryBlock.Create([TotalSeq, Hidden], 'FLUX_INIT_work1          ');
  Work2 := TMemoryBlock.Create([WorkSize], 'FLUX_INIT_work2          ');
  AttnQT := TMemoryBlock.Create([TotalSeq, Hidden], 'FLUX_INIT_attn_q_t       ');
  AttnKT := TMemoryBlock.Create([TotalSeq, Hidden], 'FLUX_INIT_attn_k_t       ');
  AttnVT := TMemoryBlock.Create([TotalSeq, Hidden], 'FLUX_INIT_attn_v_t       ');
  AttnOutT := TMemoryBlock.Create([TotalSeq, Hidden], 'FLUX_INIT_attn_out_t     ');
  AttnCatK := TMemoryBlock.Create([TotalSeq, Hidden], 'FLUX_INIT_attn_cat_k     ');
  AttnCatV := TMemoryBlock.Create([TotalSeq, Hidden], 'FLUX_INIT_attn_cat_v     ');
  SingleQ := TMemoryBlock.Create([TotalSeq, Hidden], 'FLUX_INIT_single_q       ');
  SingleK := TMemoryBlock.Create([TotalSeq, Hidden], 'FLUX_INIT_single_k       ');
  SingleV := TMemoryBlock.Create([TotalSeq, Hidden], 'FLUX_INIT_single_v       ');
  SingleMlpGate := TMemoryBlock.Create([TotalSeq, Mlp], 'FLUX_INIT_single_mlp_gate');
  SingleMlpUp := TMemoryBlock.Create([TotalSeq, Mlp], 'FLUX_INIT_single_mlp_up  ');
  SingleAttnOut := TMemoryBlock.Create([TotalSeq, Hidden], 'FLUX_INIT_single_attn_out');
  SingleConcat := TMemoryBlock.Create([TotalSeq, (Hidden + Mlp)], 'FLUX_INIT_single_concat');
  DoubleImgAttnOut := TMemoryBlock.Create([TotalSeq, Hidden], 'FLUX_INIT_double_img_attn_out');
  DoubleTxtAttnOut := TMemoryBlock.Create([TotalSeq, Hidden], 'FLUX_INIT_double_txt_attn_out');
  WorkSeqAlloc := TotalSeq;
end;

procedure TTransformerFlux.Free();
var
  I: LongInt;
begin
  QNNGPUClearWeightCache;
  ImgInWeight.Free;
  TxtInWeight.Free;
  ImgInWeightBF16.Free;
  TxtInWeightBF16.Free;
  TimeEmbed.Fc1Weight.Free;
  TimeEmbed.Fc2Weight.Free;
  for I := 0 to High(DoubleBlocks) do
    DoubleBlocks[I].Free;
  for I := 0 to High(SingleBlocks) do
    SingleBlocks[I].Free;

  SetLength(DoubleBlocks, 0);
  SetLength(SingleBlocks, 0);

  FinalNormWeight.Free;
  FinalProjWeight.Free;
  FinalProjWeightBF16.Free;
  RoPEFreqs.Free;
  ImgHidden.Free;
  TxtHidden.Free;
  Work1.Free;
  Work2.Free;
  AdalnDoubleImgWeight.Free;
  AdalnDoubleTxtWeight.Free;
  AdalnSingleWeight.Free;
  AdalnDoubleImgWeightBF16.Free;
  AdalnDoubleTxtWeightBF16.Free;
  AdalnSingleWeightBF16.Free;
  AttnQT.Free;
  AttnKT.Free;
  AttnVT.Free;
  AttnOutT.Free;
  AttnScores.Free;
  AttnCatK.Free;
  AttnCatV.Free;
  SingleQ.Free;
  SingleK.Free;
  SingleV.Free;
  SingleMlpGate.Free;
  SingleMlpUp.Free;
  SingleAttnOut.Free;
  SingleConcat.Free;
  FfnGate.Free;
  FfnUp.Free;
  TEmbSilu.Free;
  DoubleModImg.Free;
  DoubleModTxt.Free;
  DoubleImgAttnOut.Free;
  DoubleTxtAttnOut.Free;
  CachedImgRoPECos.Free;
  CachedImgRoPESin.Free;
  CachedRefRoPECos.Free;
  CachedRefRoPESin.Free;
  CachedTxtRoPECos.Free;
  CachedTxtRoPESin.Free;
  CachedCombinedRoPECos.Free;
  CachedCombinedRoPESin.Free;
  for I := 0 to High(SfFiles) do
    SfFiles[I].Close();
  SfFiles := nil;
  Self := Default(TTransformerFlux)
end;

procedure TTransformerFlux.FreeMMapCache();
var
  I: LongInt;
begin
  if UseMmap then
  begin
    for I := 0 to High(DoubleBlocks) do
      DoubleBlocks[I].Free;
    for I := 0 to High(SingleBlocks) do
      SingleBlocks[I].Free;
    SetLength(DoubleBlocks, 0);
    SetLength(SingleBlocks, 0);
  end;
end;

procedure TTransformerFlux.GetCachedRefRoPE(const PatchH, PatchW, TOffset: LongInt; var CosOut: TSingles; var SinOut: TSingles);
var
  Seq, LocalAxisDim: LongInt;
// size: NativeInt;
begin
  Seq := PatchH * PatchW;
  LocalAxisDim := AxisDim;
  if (CachedRefH = PatchH)
    and (CachedRefW = PatchW)
    and (CachedRefTOffset = TOffset)
    and CachedRefRoPECos.IsAssigned()
    and CachedRefRoPESin.IsAssigned() then
  begin
    CosOut := CachedRefRoPECos;
    SinOut := CachedRefRoPESin;
    Exit()
  end;
  if CachedRefRoPECos.IsAssigned() then
    CachedRefRoPECos.Free();
  if CachedRefRoPESin.IsAssigned() then
    CachedRefRoPESin.Free();
  // size := seq * axisDim * 4 ;
  CachedRefRoPECos := TMemoryBlock.Create([Seq, LocalAxisDim * 4], 'FLUX_GET_CACHED_ROPE_COS');
  CachedRefRoPESin := TMemoryBlock.Create([Seq, LocalAxisDim * 4], 'FLUX_GET_CACHED_ROPE_SIN');
  CachedRefH := PatchH;
  CachedRefW := PatchW;
  CachedRefTOffset := TOffset;
  QNNComputeRoPE2DOffset(CachedRefRoPECos, CachedRefRoPESin, PatchH, PatchW, LocalAxisDim, RoPETheta, TOffset);
  CosOut := CachedRefRoPECos;
  SinOut := CachedRefRoPESin
end;

procedure TTransformerFlux.GetCachedImgRoPE(const PatchH, PatchW: LongInt; var CosOut: TSingles; var SinOut: TSingles);
var
  Seq, LocalAxisDim: LongInt;
// size: NativeInt;
begin
  Seq := PatchH * PatchW;
  LocalAxisDim := AxisDim;
  if (CachedImgH = PatchH) and (CachedImgW = PatchW) and (Boolean(CachedImgRoPECos) and Boolean(CachedImgRoPESin)) then
  begin
    CosOut := CachedImgRoPECos;
    SinOut := CachedImgRoPESin;
    Exit()
  end;
  // if assigned(cached_img_rope_cos) then
  //    free();
  // if assigned(cached_img_rope_sin) then
  //    free();
  // size := seq * axisDim * 4 ;
  if CachedImgRoPECos.IsAssigned() then
    CachedImgRoPECos.Free;
  if CachedImgRoPESin.IsAssigned() then
    CachedImgRoPESin.Free;

  CachedImgRoPECos := TMemoryBlock.Create([Seq, LocalAxisDim, 4], 'FLUX_GET_CACHED_IMG_ROPE_COS');
  CachedImgRoPESin := TMemoryBlock.Create([Seq, LocalAxisDim, 4], 'FLUX_GET_CACHED_IMG_ROPE_SIN');
  CachedImgH := PatchH;
  CachedImgW := PatchW;
  QNNComputeRoPE2D(CachedImgRoPECos, CachedImgRoPESin, PatchH, PatchW, LocalAxisDim, RoPETheta);

  CosOut := CachedImgRoPECos;
  SinOut := CachedImgRoPESin
end;

procedure TTransformerFlux.GetCachedTxtRoPE(const TxtSeq: LongInt; var CosOut: TSingles; var SinOut: TSingles);
var
  LocalHeadDim: LongInt;
// size: NativeInt;
begin
  LocalHeadDim := HeadDim;
  // size := txt_seq * headDim;

  if (CachedTxtSeq = TxtSeq) and (Boolean(CachedTxtRoPECos) and Boolean(CachedTxtRoPESin)) then
  begin
    CosOut := CachedTxtRoPECos;
    SinOut := CachedTxtRoPESin;
    Exit()
  end;
  if CachedTxtRoPECos.IsAssigned() then
    CachedTxtRoPECos.Free();
  if CachedTxtRoPESin.IsAssigned() then
    CachedTxtRoPESin.Free();
  CachedTxtRoPECos := TMemoryBlock.Create([TxtSeq, LocalHeadDim], 'FLUX_GET_CACHED_TXT_ROPE_COS');
  CachedTxtRoPESin := TMemoryBlock.Create([TxtSeq, LocalHeadDim], 'FLUX_GET_CACHED_TXT_ROPE_SIN');
  CachedTxtSeq := TxtSeq;
  QNNComputeRoPEText(CachedTxtRoPECos, CachedTxtRoPESin, TxtSeq, AxisDim, RoPETheta);
  CosOut := CachedTxtRoPECos;
  SinOut := CachedTxtRoPESin
end;

procedure TTransformerFlux.GetCachedCombinedRoPE(const ImgH, ImgW, RefH, RefW, TOffset: LongInt; var CosOut: TSingles; var SinOut: TSingles);
var
  ImgSeq, RefSeq, CombinedSeq, LocalAxisDim: LongInt;
  {combined_size,}
  ImgSize, RefSize: NativeInt;
  ImgRoPECos, ImgRoPESin, RefRoPECos, RefRoPESin: TSingles;
begin
  ImgSeq := ImgH * ImgW;
  RefSeq := RefH * RefW;
  CombinedSeq := ImgSeq + RefSeq;
  LocalAxisDim := AxisDim;
  // combined_size := combined_seq * axisDim * 4;
  ImgSize := ImgSeq * LocalAxisDim * 4;
  RefSize := RefSeq * LocalAxisDim * 4;
  if (CachedCombinedImgH = ImgH)
    and (CachedCombinedImgW = ImgW)
    and (CachedCombinedRefH = RefH)
    and (CachedCombinedRefW = RefW)
    and (CachedCombinedTOffset = TOffset)
    and CachedCombinedRoPECos.IsAssigned()
    and CachedCombinedRoPESin.IsAssigned() then
  begin
    CosOut := CachedCombinedRoPECos;
    SinOut := CachedCombinedRoPESin;
    Exit()
  end;
  if CachedCombinedRoPECos.IsAssigned() then
    CachedCombinedRoPECos.Free();
  if CachedCombinedRoPESin.IsAssigned() then
    CachedCombinedRoPESin.Free();
  CachedCombinedRoPECos := TMemoryBlock.Create([CombinedSeq, LocalAxisDim, 4], 'FLUX_GET_CACHED_COMBINED_ROPE_COS');
  CachedCombinedRoPESin := TMemoryBlock.Create([CombinedSeq, LocalAxisDim, 4], 'FLUX_GET_CACHED_COMBINED_ROPE_SIN');
  CachedCombinedImgH := ImgH;
  CachedCombinedImgW := ImgW;
  CachedCombinedRefH := RefH;
  CachedCombinedRefW := RefW;
  CachedCombinedTOffset := TOffset;

  GetCachedImgRoPE(ImgH, ImgW, ImgRoPECos, ImgRoPESin);

  GetCachedRefRoPE(RefH, RefW, TOffset, RefRoPECos, RefRoPESin);

  QNNCopy(CachedCombinedRoPECos, ImgRoPECos, ImgSize);
  QNNCopy(CachedCombinedRoPECos + ImgSize, RefRoPECos, RefSize);
  QNNCopy(CachedCombinedRoPESin, ImgRoPESin, ImgSize);
  QNNCopy(CachedCombinedRoPESin + ImgSize, RefRoPESin, RefSize);

  CosOut := CachedCombinedRoPECos;
  SinOut := CachedCombinedRoPESin
end;

procedure TTransformerFlux.MultiHeadForward(const Dst, Q, K, V: TMemoryBlock; const Seq: LongInt);
var
  Hidden, LocalHeadDim: LongInt;
  Scale: QNNFloat;
  Scores: TMemoryBlock;
  Worker: TGroupProcNested;
begin
  Worker :=
    procedure(const Start, Finish: IntPtr; const Data: Pointer)

    var
      I: LongInt;
      Qh, KH, Vh, Oh, Sh: TMemoryBlock;
    begin
      for I := Start to Finish do
      begin
        Qh := Q + I * LocalHeadDim;
        KH := K + I * LocalHeadDim;
        Vh := V + I * LocalHeadDim;
        Oh := Dst + I * LocalHeadDim;
        Sh := Scores + I * Seq * Seq;

        CblasGemm(
          CblasRowMajor,
          CblasNoTrans,
          CblasTrans,
          Seq,
          Seq,
          LocalHeadDim,
          Scale,
          Qh {Q      + i*headDim},
          Hidden,
          KH {K      + i*headDim},
          Hidden,
          0.0,
          Sh {scores + i*seq*seq},
          Seq
        );
        QNNSoftmaxRows(Sh {scores + i*seq*seq}, Seq, Seq);

        CblasGemm(
          CblasRowMajor,
          CblasNoTrans,
          CblasNoTrans,
          Seq,
          LocalHeadDim,
          Seq,
          1.0,
          Sh {scores + i*seq*seq},
          Seq,
          Vh {V      + i*headDim},
          Hidden,
          0.0,
          Oh {dst    + i*headDim},
          Hidden
        );
      end;
    end;

  LocalHeadDim := HeadDim;
  Hidden := NumHeads * HeadDim; // but we use self.hidden_size
  Scale := 1 / Sqrt(HeadDim);
  Scores := AttnScores; // Capture the score buffer explicitly for the anonymous procedure.
  if not IsUsingBlas then
    QNNFlashAttention(Dst, Q, K, V, NumHeads, Seq, Seq, HeadDim, Scale {1/sqrt(head_dim)})
  else
{$IFDEF _USE_MULTITHREADING} // CAUTION : do not multithread
    Mp.&for(Worker, 0, NumHeads - 1);
{$ELSE}
  Worker(0, NumHeads - 1, nil);
{$ENDIF}
end;

procedure TTransformerFlux.JointAttentionForward(const ImgOut, TxtOut, ImgQ, ImgK, ImgV, TxtQ, TxtK, TxtV: TMemoryBlock; const ImgSeq, TxtSeq: LongInt);
var
  TotalSeq, Hidden, LocalHeadDim: LongInt;
  Scale: QNNFloat;
  CatK, CatV, Scores: TMemoryBlock;
  Worker: TGroupProcNested;
begin
  Worker :=
    procedure(const Start, Finish: IntPtr; const Data: Pointer)

    var
      I: LongInt;
      ImgQh, ImgSh, ImgOh, TxtQh, TxtSh, TxtOh, KH, Vh: TMemoryBlock;
    begin
      for I := Start to Finish do
      begin
        ImgQh := ImgQ + I * LocalHeadDim;
        TxtQh := TxtQ + I * LocalHeadDim;
        KH := CatK + I * LocalHeadDim;
        Vh := CatV + I * LocalHeadDim;
        ImgOh := ImgOut + I * LocalHeadDim;
        TxtOh := TxtOut + I * LocalHeadDim;
        ImgSh := Scores + I * TotalSeq * TotalSeq;
        TxtSh := ImgSh + ImgSeq * TotalSeq; // txt_sh is an offset from img_sh

        CblasGemm(
          CblasRowMajor,
          CblasNoTrans,
          CblasTrans,
          ImgSeq,
          TotalSeq,
          LocalHeadDim,
          Scale,
          ImgQh,
          Hidden,
          KH,
          Hidden,
          0.0,
          ImgSh,
          TotalSeq
        );
        QNNSoftmaxRows(ImgSh, ImgSeq, TotalSeq);
        CblasGemm(
          CblasRowMajor,
          CblasNoTrans,
          CblasNoTrans,
          ImgSeq,
          LocalHeadDim,
          TotalSeq,
          1.0,
          ImgSh,
          TotalSeq,
          Vh,
          Hidden,
          0.0,
          ImgOh,
          Hidden
        );

        CblasGemm(
          CblasRowMajor,
          CblasNoTrans,
          CblasTrans,
          TxtSeq,
          TotalSeq,
          LocalHeadDim,
          Scale,
          TxtQh,
          Hidden,
          KH,
          Hidden,
          0.0,
          TxtSh,
          TotalSeq
        );
        QNNSoftmaxRows(TxtSh, TxtSeq, TotalSeq);
        CblasGemm(
          CblasRowMajor,
          CblasNoTrans,
          CblasNoTrans,
          TxtSeq,
          LocalHeadDim,
          TotalSeq,
          1.0,
          TxtSh,
          TotalSeq,
          Vh,
          Hidden,
          0.0,
          TxtOh,
          Hidden
        );
      end;
    end;

  TotalSeq := ImgSeq + TxtSeq;
  LocalHeadDim := HeadDim;
  Hidden := NumHeads * HeadDim;
  Scale := 1 / Sqrt(HeadDim);

  (* Use pre-allocated buffers for K/V concatenation *)
  CatK := AttnCatK;
  CatV := AttnCatV;
  Scores := AttnScores; // because delphi does not capture Self in lambda procedures

  // Concatenate K, V from both streams in [seq, heads, head_dim] format
  // IMPORTANT: Python (official Flux2) concatenates as [TEXT, IMAGE]
  QNNCopy(AttnCatK, TxtK, TxtSeq * Hidden);
  QNNCopy(AttnCatV, TxtV, TxtSeq * Hidden);
  QNNCopy(AttnCatK + TxtSeq * Hidden, ImgK, ImgSeq * Hidden);
  QNNCopy(AttnCatV + TxtSeq * Hidden, ImgV, ImgSeq * Hidden);

  if not IsUsingBlas then
  begin
    QNNFlashAttention(ImgOut, ImgQ, AttnCatK, AttnCatV, NumHeads, ImgSeq, TotalSeq, HeadDim, Scale);
    QNNFlashAttention(TxtOut, TxtQ, AttnCatK, AttnCatV, NumHeads, TxtSeq, TotalSeq, HeadDim, Scale);
  end
  else
  begin
{$IFDEF _USE_MULTITHREADING} // CAUTION : do not multithread
    Mp.&for(Worker, 0, NumHeads - 1);
{$ELSE}
    Worker(0, NumHeads - 1, nil)
{$ENDIF}
  end
end;

procedure TTransformerFlux.FfnSwigluForward(const Dst, X: TMemoryBlock; const GateWeight, UpWeight, DownWeight: TSingles; const GateWeightBF16, UpWeightBF16, DownWeightBF16: TBF16s; const Seq, Hidden, MlpHidden: LongInt);
var
  Gate, Up, Down: TMemoryBlock;
begin
  if Seq <= 0 then
    Exit;
  Gate := GateWeight;
  if GateWeightBF16.IsAssigned then
    Gate := GateWeightBF16;
  Up := UpWeight;
  if UpWeightBF16.IsAssigned then
    Up := UpWeightBF16;
  Down := DownWeight;
  if DownWeightBF16.IsAssigned then
    Down := DownWeightBF16;
  if TryVulkanSwiGLU(Dst, X, Gate, Up, Down, Seq, Hidden, MlpHidden) then
    Exit;
  // Host intermediates are needed only for the fallback path.
  if FfnGate.Count < Int64(Seq) * MlpHidden then
    FfnGate.Resize([Seq, MlpHidden], dtF32);
  if FfnUp.Count < Int64(Seq) * MlpHidden then
    FfnUp.Resize([Seq, MlpHidden], dtF32);
  QNNLinearBF16OrF32(FfnGate, X, GateWeight, GateWeightBF16, Seq, Hidden, MlpHidden);
  QNNLinearBF16OrF32(FfnUp, X, UpWeight, UpWeightBF16, Seq, Hidden, MlpHidden);
  QNNSiluMul(FfnGate, FfnUp, Seq * MlpHidden);
  QNNLinearBF16OrF32(Dst, FfnGate, DownWeight, DownWeightBF16, Seq, MlpHidden, Hidden);
end;

(* ========================================================================
  Double-Stream Block (MM-DiT)
  ========================================================================

  One MM-DiT (Multi-Modal DiT) double block. Processes image and text as
  separate streams with their own Q/K/V projections, but performs joint
  attention (K and V are concatenated from both streams so each modality
  attends to the other). Each stream gets its own AdaLN modulation with
  6 parameters: shift1, scale1, gate1 (pre-attention), shift2, scale2,
  gate2 (pre-FFN). The gating mechanism controls information flow from
  the attention and FFN residual paths. *)

procedure TTransformerFlux.DoubleBlockForward(const ImgHidden, TxtHidden: TMemoryBlock; const BlockIdx: LongInt; const ImgMod, TxtMod, ImgRoPECos, ImgRoPESin, TxtRoPECos, TxtRoPESin: TMemoryBlock; const ImgSeq, TxtSeq: LongInt);
const
  AXISDIM = 32;
var
  ImgShift1, ImgScale1, ImgGate1, ImgShift2, ImgScale2, ImgGate2, TxtShift1, TxtScale1, TxtGate1, TxtShift2, TxtScale2, TxtGate2, ImgNorm, ImgQ, ImgK, ImgV, TxtNorm, TxtQ, TxtK, TxtV, ImgProj, TxtProj: TMemoryBlock;
  Block: PDoubleBlock;
begin

  (* Extract pre-computed modulation parameters *)
  ImgShift1 := ImgMod;
  ImgScale1 := ImgMod + HiddenSize;
  ImgGate1 := ImgMod + HiddenSize * 2;
  ImgShift2 := ImgMod + HiddenSize * 3;
  ImgScale2 := ImgMod + HiddenSize * 4;
  ImgGate2 := ImgMod + HiddenSize * 5;

  TxtShift1 := TxtMod;
  TxtScale1 := TxtMod + HiddenSize;
  TxtGate1 := TxtMod + HiddenSize * 2;
  TxtShift2 := TxtMod + HiddenSize * 3;
  TxtScale2 := TxtMod + HiddenSize * 4;
  TxtGate2 := TxtMod + HiddenSize * 5;

  ImgNorm := Work1;
  TxtNorm := ImgNorm + ImgSeq * HiddenSize;

  ImgQ := Work2;
  ImgK := ImgQ + ImgSeq * HiddenSize;
  ImgV := ImgK + ImgSeq * HiddenSize;

  TxtQ := ImgV + ImgSeq * HiddenSize;
  TxtK := TxtQ + TxtSeq * HiddenSize;
  TxtV := TxtK + TxtSeq * HiddenSize;

  Block := @DoubleBlocks[BlockIdx];

  QNNAdaLN(ImgNorm, ImgHidden, ImgShift1, ImgScale1, ImgSeq, HiddenSize);

  QNNLinearBF16OrF32(ImgQ, ImgNorm, Block.ImgQWeight, Block.ImgQWeightBF16, ImgSeq, HiddenSize, HiddenSize);
  QNNLinearBF16OrF32(ImgK, ImgNorm, Block.ImgKWeight, Block.ImgKWeightBF16, ImgSeq, HiddenSize, HiddenSize);
  QNNLinearBF16OrF32(ImgV, ImgNorm, Block.ImgVWeight, Block.ImgVWeightBF16, ImgSeq, HiddenSize, HiddenSize);
  QNNQKRMSNorm(ImgQ, ImgK, Block.ImgNormQWeight, Block.ImgNormKWeight, ImgSeq, NumHeads, HeadDim);

  QNNApplyRoPE2D(ImgQ, ImgRoPECos, ImgRoPESin, ImgSeq, NumHeads, HeadDim, AXISDIM);
  QNNApplyRoPE2D(ImgK, ImgRoPECos, ImgRoPESin, ImgSeq, NumHeads, HeadDim, AXISDIM);
  // printStat(txt_hidden, img_seq*hidden_size);
  QNNAdaLN(TxtNorm, TxtHidden, TxtShift1, TxtScale1, TxtSeq, HiddenSize);
  // printStat(txt_norm, img_seq*hidden_size);
  QNNLinearBF16OrF32(TxtQ, TxtNorm, Block.TxtQWeight, Block.TxtQWeightBF16, TxtSeq, HiddenSize, HiddenSize);
  QNNLinearBF16OrF32(TxtK, TxtNorm, Block.TxtKWeight, Block.TxtKWeightBF16, TxtSeq, HiddenSize, HiddenSize);
  QNNLinearBF16OrF32(TxtV, TxtNorm, Block.TxtVWeight, Block.TxtVWeightBF16, TxtSeq, HiddenSize, HiddenSize);
  QNNQKRMSNorm(TxtQ, TxtK, Block.TxtNormQWeight, Block.TxtNormKWeight, TxtSeq, NumHeads, HeadDim);

  // printStat(txt_q, txt_seq * hidden_size);
  // printStat(txt_k, txt_seq * hidden_size);
  // printStat(txt_v, txt_seq * hidden_size);

  QNNApplyRoPE2D(TxtQ, TxtRoPECos, TxtRoPESin, TxtSeq, NumHeads, HeadDim, AXISDIM);
  QNNApplyRoPE2D(TxtK, TxtRoPECos, TxtRoPESin, TxtSeq, NumHeads, HeadDim, AXISDIM);

  // printStat(txt_q, txt_seq * hidden_size);
  // printStat(txt_k, txt_seq * hidden_size);
  // printStat(txt_v, txt_seq * hidden_size);

  JointAttentionForward(
    DoubleImgAttnOut,
    DoubleTxtAttnOut,
    ImgQ,
    ImgK,
    ImgV,
    TxtQ,
    TxtK,
    TxtV,
    ImgSeq,
    TxtSeq {, num_heads, head_dim}
  );

  ImgProj := Work1;
  TxtProj := ImgProj + ImgSeq * HiddenSize;

  QNNLinearBF16OrF32(
    ImgProj,
    DoubleImgAttnOut,
    Block.ImgProjWeight,
    Block.ImgProjWeightBF16,
    ImgSeq,
    HiddenSize,
    HiddenSize
  );
  QNNLinearBF16OrF32(
    TxtProj,
    DoubleTxtAttnOut,
    Block.TxtProjWeight,
    Block.TxtProjWeightBF16,
    TxtSeq,
    HiddenSize,
    HiddenSize
  );

  QNNGatedAdd(ImgHidden, ImgGate1, ImgProj, ImgSeq, HiddenSize);
  QNNGatedAdd(TxtHidden, TxtGate1, TxtProj, TxtSeq, HiddenSize);

  QNNAdaLN(ImgNorm, ImgHidden, ImgShift2, ImgScale2, ImgSeq, HiddenSize);
  FfnSwigluForward(
    ImgProj,
    ImgNorm,
    Block.ImgMlpGateWeight,
    Block.ImgMlpUpWeight,
    Block.ImgMlpDownWeight,
    Block.ImgMlpGateWeightBF16,
    Block.ImgMlpUpWeightBF16,
    Block.ImgMlpDownWeightBF16,
    ImgSeq,
    HiddenSize,
    MlpHidden
  );
  QNNGatedAdd(ImgHidden, ImgGate2, ImgProj, ImgSeq, HiddenSize);

  QNNAdaLN(TxtNorm, TxtHidden, TxtShift2, TxtScale2, TxtSeq, HiddenSize);
  FfnSwigluForward(
    TxtProj,
    TxtNorm,
    Block.TxtMlpGateWeight,
    Block.TxtMlpUpWeight,
    Block.TxtMlpDownWeight,
    Block.TxtMlpGateWeightBF16,
    Block.TxtMlpUpWeightBF16,
    Block.TxtMlpDownWeightBF16,
    TxtSeq,
    HiddenSize,
    MlpHidden
  );
  QNNGatedAdd(TxtHidden, TxtGate2, TxtProj, TxtSeq, HiddenSize);
end;

(* ========================================================================
   Single-stream DiT block operating on concatenated [text, image] tokens.
   ========================================================================

   Unlike double blocks, text and image share the same self-attention and are
   processed as one sequence. Uses a fused projection that outputs [Q, K, V,
   gate, up] in one matmul for efficiency. Text and image portions get
   different RoPE (text: axis 3, image: axes 1,2). After attention, the SwiGLU
   MLP output is concatenated with the attention output before the output
   projection -- this parallel attention+MLP design saves a serial step.*)

procedure TTransformerFlux.SingleBlockForward(const Hidden: TMemoryBlock; const BlockIdx: LongInt; const TEmb, AdalnWeight, ImgRoPECos, ImgRoPESin, TxtRoPECos, TxtRoPESin: TMemoryBlock; const Seq, ImgOffset: LongInt);
const
  AXIS_DIM = 32;
var
  FusedDim, ModSize, S, ImgSeq, TxtSeq: LongInt;
  ModParams, Shift, Scale, Gate, Norm, FusedOut, Row, ImgQ, ImgK, ProjOut: TMemoryBlock;
  Block: PSingleBlock;
begin

  FusedDim := HiddenSize * 3 + MlpHidden * 2;
  ImgSeq := Seq - ImgOffset;
  ModSize := HiddenSize * 3;
  QNNSilu(TEmbSilu, TEmb, HiddenSize);
  // for i := 0 to hidden_size -1 do
  //    begin
  //        x := t_emb[i];
  //        t_emb_silu[i] := x / (1.0+expf(-x))
  //    end;
  ModParams := Work2 + Seq * FusedDim;

  QNNLinearNoBias(ModParams, TEmbSilu, AdalnWeight, 1, HiddenSize, ModSize);
  Shift := ModParams;
  Scale := ModParams + HiddenSize;
  Gate := ModParams + HiddenSize * 2;
  Norm := Work1;
  QNNAdaLN(Norm, Hidden, Shift, Scale, Seq, HiddenSize);
  FusedOut := Work2;
  Block := @SingleBlocks[BlockIdx];
  QNNLinearBF16OrF32(FusedOut, Norm, Block.QKVMlpWeight, Block.QKVMlpWeightBF16, Seq, HiddenSize, FusedDim);

  for S := 0 to Seq - 1 do
  begin
    Row := FusedOut + S * FusedDim;
    QNNCopy(SingleQ + S * HiddenSize, Row, HiddenSize);
    QNNCopy(SingleK + S * HiddenSize, Row + HiddenSize, HiddenSize);
    QNNCopy(SingleV + S * HiddenSize, Row + HiddenSize * 2, HiddenSize);
    QNNCopy(SingleMlpGate + S * MlpHidden, Row + HiddenSize * 3, MlpHidden);
    QNNCopy(SingleMlpUp + S * MlpHidden, Row + HiddenSize * 3 + MlpHidden, MlpHidden)
  end;
  QNNQKRMSNorm(SingleQ, SingleK, Block.NormQWeight, Block.NormKWeight, Seq, NumHeads, HeadDim);
  TxtSeq := ImgOffset;
  QNNApplyRoPE2D(SingleQ, TxtRoPECos, TxtRoPESin, TxtSeq, NumHeads, HeadDim, AXIS_DIM);
  QNNApplyRoPE2D(SingleK, TxtRoPECos, TxtRoPESin, TxtSeq, NumHeads, HeadDim, AXIS_DIM);
  ImgQ := SingleQ + ImgOffset * HiddenSize;
  ImgK := SingleK + ImgOffset * HiddenSize;
  QNNApplyRoPE2D(ImgQ, ImgRoPECos, ImgRoPESin, ImgSeq, NumHeads, HeadDim, AXIS_DIM);
  QNNApplyRoPE2D(ImgK, ImgRoPECos, ImgRoPESin, ImgSeq, NumHeads, HeadDim, AXIS_DIM);
  MultiHeadForward(SingleAttnOut, SingleQ, SingleK, SingleV, Seq);
  QNNSiluMul(SingleMlpGate, SingleMlpUp, Seq * MlpHidden);
  for S := 0 to Seq - 1 do
  begin
    QNNCopy(SingleConcat + S * (HiddenSize + MlpHidden), SingleAttnOut + S * HiddenSize, HiddenSize);
    QNNCopy(SingleConcat + S * (HiddenSize + MlpHidden) + HiddenSize, SingleMlpGate + S * MlpHidden, MlpHidden)
  end;
  ProjOut := Work1;
  QNNLinearBF16OrF32(
    ProjOut,
    SingleConcat,
    Block.ProjMlpWeight,
    Block.ProjMlpWeightBF16,
    Seq,
    HiddenSize + MlpHidden,
    HiddenSize
  );
  QNNGatedAdd(Hidden, Gate, ProjOut, Seq, HiddenSize)
end;

function TTransformerFlux.forward(const ImgLatent: TMemoryBlock; const ImgH, ImgW: LongInt; const TxtEmb: TMemoryBlock; const TxtSeq: LongInt; Timestep: QNNFloat): TMemoryBlock;
var
  ImgSeq, TotalSeq, DoubleModSize, I: LongInt;
  // x: single;
  TEmb, ImgTransposed, ConcatHidden, ImgRoPECos, TxtRoPECos, ImgRoPESin, TxtRoPESin, OutputNlc: TSingles;
  FinalMod, FinalScale, FinalShift, FinalNorm: TMemoryBlock;
  TSincos: TMemoryBlock; // array[0..255] of QNNFloat;
begin
  ImgSeq := ImgH * ImgW;
  TotalSeq := ImgSeq + TxtSeq;
  ReInitialize(TotalSeq);

  TEmb := TMemoryBlock.Create([HiddenSize], 'FLUX_FW_TIME_EMB');
  TSincos := QNNTimeStepEmbedding(Timestep * 1000.0, TimeEmbed.SincosDim, 10000.0);

  TimeEmbed.forward(TEmb, TSincos, HiddenSize, TEmbSilu);
  TSincos.Free;

  GetCachedImgRoPE(ImgH, ImgW, ImgRoPECos, ImgRoPESin);

  GetCachedTxtRoPE(TxtSeq, TxtRoPECos, TxtRoPESin);

  ImgTransposed := TMemoryBlock.Create([ImgSeq, LatentChannels], 'FLUX_FW_IMG_TRANSPOSED');
  // for pos := 0 to img_seq -1 do
  //    for c := 0 to latent_channels -1 do
  //        img_transposed[pos*latent_channels + c] := img_latent[c*img_seq + pos];
  QNNMatTranspose(ImgTransposed, ImgLatent, LatentChannels, ImgSeq);

  QNNLinearBF16OrF32(ImgHidden, ImgTransposed, ImgInWeight, ImgInWeightBF16, ImgSeq, LatentChannels, HiddenSize);
  ImgTransposed.Free;

  QNNLinearBF16OrF32(TxtHidden, TxtEmb, TxtInWeight, TxtInWeightBF16, TxtSeq, TextDim, HiddenSize);
  // printStat(txt_emb, txt_seq*hidden_size);
  // printStat(txt_hidden, txt_seq*hidden_size);

  DoubleModSize := HiddenSize * 6;

  // for j := 0 to hidden_size -1 do
  //    begin
  //        x := t_emb[j];
  //        t_emb_silu[j] := x / (1.0+expf(-x))
  //    end;
  QNNSilu(TEmbSilu, TEmb, HiddenSize);
  QNNLinearNoBias(DoubleModImg, TEmbSilu, AdalnDoubleImgWeight, 1, HiddenSize, DoubleModSize);
  QNNLinearNoBias(DoubleModTxt, TEmbSilu, AdalnDoubleTxtWeight, 1, HiddenSize, DoubleModSize);

  for I := 0 to NumDoubleLayers {high(double_blocks)} - 1 do
  begin
    if UseMmap and not (Boolean(DoubleBlocks[I].ImgQWeight) or Boolean(DoubleBlocks[I].ImgQWeightBF16)) then
      DoubleBlocks[I].Load(SfFiles, I, HiddenSize, MlpHidden, UseBF16);
    // todo TTransformerFlux.forward : {} nest to a param means it's redundant, it is a record member var, remove/refactor
    DoubleBlockForward(
      ImgHidden,
      TxtHidden {},
      I,
      DoubleModImg {},
      DoubleModTxt {},
      ImgRoPECos,
      ImgRoPESin,
      TxtRoPECos,
      TxtRoPESin,
      ImgSeq,
      TxtSeq
    );

    if UseMmap then
      DoubleBlocks[I].Free;
    // img_hidden.printStat;
    // txt_hidden.printStat;
    if Assigned(SubstepCallback) then
      SubstepCallback(SUBSTEP_DOUBLE_BLOCK, I, NumDoubleLayers);
  end;
  ConcatHidden := TMemoryBlock.Create([TotalSeq, HiddenSize], 'FLUX_FW_CONCAT_HIDDEN');
  QNNCopy(ConcatHidden, TxtHidden, TxtSeq * HiddenSize);
  QNNCopy(ConcatHidden + TxtSeq * HiddenSize, ImgHidden, ImgSeq * HiddenSize);
  for I := 0 to NumSingleLayers - 1 {high(single_blocks)} do
  begin
    if UseMmap and not (Boolean(SingleBlocks[I].QKVMlpWeight) or Boolean(SingleBlocks[I].QKVMlpWeightBF16)) then
      SingleBlocks[I].Load(SfFiles, I, HiddenSize, MlpHidden, UseBF16);
    SingleBlockForward(
      ConcatHidden,
      I,
      TEmb,
      AdalnSingleWeight,
      ImgRoPECos,
      ImgRoPESin,
      TxtRoPECos,
      TxtRoPESin,
      TotalSeq,
      TxtSeq
    );
    if UseMmap then
      SingleBlocks[I].Free;
    //    free_single_block_weights(@single_blocks[i]);
    if Assigned(SubstepCallback) then
      SubstepCallback(SUBSTEP_SINGLE_BLOCK, I, NumSingleLayers);
  end;

  QNNCopy(ImgHidden, ConcatHidden + TxtSeq * HiddenSize, ImgSeq * HiddenSize);
  ConcatHidden.Free;
  QNNSilu(TEmbSilu, TEmb, HiddenSize);
  // for i := 0 to hidden_size -1 do
  //    begin
  //        x := t_emb[i];
  //        t_emb_silu[i] := x / (1.0+expf(-x))
  //    end;
  FinalMod := DoubleModImg;
  QNNLinearNoBias(FinalMod, TEmbSilu, FinalNormWeight, 1, HiddenSize, HiddenSize * 2);
  FinalScale := FinalMod;
  FinalShift := FinalMod + HiddenSize;
  FinalNorm := Work1;
  QNNAdaLN(FinalNorm, ImgHidden, FinalShift, FinalScale, ImgSeq, HiddenSize, -6);
  OutputNlc := TMemoryBlock.Create([ImgSeq, LatentChannels], 'FLUX_FW_OUT_NLC');
  QNNLinearBF16OrF32(OutputNlc, FinalNorm, FinalProjWeight, FinalProjWeightBF16, ImgSeq, HiddenSize, LatentChannels);
  Result := TMemoryBlock.Create([ImgSeq, LatentChannels], 'FLUX_FW_RESULT');
  // for pos := 0 to img_seq -1 do
  //    for c := 0 to latent_channels -1 do
  //        result[c * img_seq+pos] := output_nlc[pos * latent_channels+c];
  QNNMatTranspose(Result, OutputNlc, ImgSeq, LatentChannels);
  OutputNlc.Free;
  TEmb.Free;

  if Assigned(SubstepCallback) then
    SubstepCallback(SUBSTEP_FINAL_LAYER, 0, 1);
end;

function TTransformerFlux.forward(const ImgLatent: TMemoryBlock; const ImgH, ImgW: LongInt; const RefLatent: TMemoryBlock; const RefH, RefW, TOffset: LongInt; const TxtEmb: TMemoryBlock; const TxtSeq: LongInt; Timestep: QNNFloat): TMemoryBlock;
var
  ImgSeq, RefSeq, CombinedImgSeq, TotalSeq, I, DoubleModSize: LongInt;
  TSincos: TMemoryBlock; // array[0..255] of QNNFloat;

  // output_nlc_ptr, result_ptr, combined_transposed_ptr, img_latent_ptr, ref_latent_ptr: PQNNFloat;

  FinalMod, FinalScale, FinalShift, FinalNorm, TEmb, CombinedTransposed, CombinedHidden, TxtRoPECos, TxtRoPESin, CombinedRoPECos, CombinedRoPESin, ConcatHidden, OutputNlc: TMemoryBlock;
begin
  if not RefLatent.IsAssigned() then
    Exit(forward(ImgLatent, ImgH, ImgW, TxtEmb, TxtSeq, Timestep));
  ImgSeq := ImgH * ImgW;
  RefSeq := RefH * RefW;
  CombinedImgSeq := ImgSeq + RefSeq;
  TotalSeq := CombinedImgSeq + TxtSeq;
  ReInitialize(TotalSeq);
  TEmb := TMemoryBlock.Create([HiddenSize], 'FLUX_FW_TIME_EMBED');
  TSincos := QNNTimeStepEmbedding(Timestep * 1000.0, TimeEmbed.SincosDim, 10000.0);
  TimeEmbed.forward(TEmb, TSincos, HiddenSize, TEmbSilu);
  TSincos.Free;
  GetCachedCombinedRoPE(ImgH, ImgW, RefH, RefW, TOffset, CombinedRoPECos, CombinedRoPESin);

  GetCachedTxtRoPE(TxtSeq, TxtRoPECos, TxtRoPESin);
  CombinedTransposed := TMemoryBlock.Create([CombinedImgSeq, LatentChannels], 'FLUX_FW_COMBINED_TRANSPOSED');
  // combined_transposed_ptr := combined_transposed;
  // img_latent_ptr := img_latent;
  // ref_latent_ptr := ref_latent;
  // for pos := 0 to img_seq -1 do
  //    for c := 0 to latent_channels -1 do
  //        combined_transposed_ptr[pos*latent_channels + c] := img_latent_ptr[c*img_seq + pos];

  // for pos := 0 to ref_seq -1 do
  //    for c := 0 to latent_channels -1 do
  //        combined_transposed_ptr[(img_seq + pos)*latent_channels + c] := ref_latent_ptr[c*ref_seq + pos];

  QNNMatTranspose(CombinedTransposed, ImgLatent, LatentChannels, ImgSeq);
  QNNMatTranspose(CombinedTransposed + ImgSeq * LatentChannels, RefLatent, LatentChannels, RefSeq);

  CombinedHidden := TMemoryBlock.Create([CombinedImgSeq, HiddenSize], 'FLUX_FW_COMBINED_HIDDEN');
  QNNLinearBF16OrF32(
    CombinedHidden,
    CombinedTransposed,
    ImgInWeight,
    ImgInWeightBF16,
    CombinedImgSeq,
    LatentChannels,
    HiddenSize
  );
  CombinedTransposed.Free;
  QNNLinearBF16OrF32(TxtHidden, TxtEmb, TxtInWeight, TxtInWeightBF16, TxtSeq, TextDim, HiddenSize);
  DoubleModSize := HiddenSize * 6;

  QNNSilu(TEmbSilu, TEmb, HiddenSize);
  // for j := 0 to hidden_size -1 do
  //    begin
  //        x := t_emb[j];
  //        t_emb_silu[j] := x / (1.0+expf(-x))
  //    end;
  QNNLinearNoBias(DoubleModImg, TEmbSilu, AdalnDoubleImgWeight, 1, HiddenSize, DoubleModSize);
  QNNLinearNoBias(DoubleModTxt, TEmbSilu, AdalnDoubleTxtWeight, 1, HiddenSize, DoubleModSize);
  for I := 0 to NumDoubleLayers - 1 do
  begin
    if UseMmap then
      DoubleBlocks[I].Load(SfFiles, I, HiddenSize, MlpHidden, UseBF16);
    DoubleBlockForward(
      CombinedHidden,
      TxtHidden,
      I,
      DoubleModImg,
      DoubleModTxt,
      CombinedRoPECos,
      CombinedRoPESin,
      TxtRoPECos,
      TxtRoPESin,
      CombinedImgSeq,
      TxtSeq
    );
    if UseMmap then
      DoubleBlocks[I].Free;
    if Assigned(SubstepCallback) then
      SubstepCallback(SUBSTEP_DOUBLE_BLOCK, I, NumDoubleLayers)
  end;
  ConcatHidden := TMemoryBlock.Create([TotalSeq, HiddenSize], 'FLUX_FW_CONACT_HIDDEN');
  QNNCopy(ConcatHidden, TxtHidden, TxtSeq * HiddenSize);
  QNNCopy(ConcatHidden + TxtSeq * HiddenSize, CombinedHidden, CombinedImgSeq * HiddenSize);
  for I := 0 to NumSingleLayers - 1 do
  begin
    if UseMmap then
      SingleBlocks[I].Load(SfFiles, I, HiddenSize, MlpHidden, UseBF16);
    SingleBlockForward(
      ConcatHidden,
      I,
      TEmb,
      AdalnSingleWeight,
      CombinedRoPECos,
      CombinedRoPESin,
      TxtRoPECos,
      TxtRoPESin,
      TotalSeq,
      TxtSeq
    );
    if UseMmap then
      SingleBlocks[I].Free;
    if Assigned(SubstepCallback) then
      SubstepCallback(SUBSTEP_SINGLE_BLOCK, I, NumSingleLayers)
  end;
  ImgHidden := TMemoryBlock.Create([ImgSeq, HiddenSize], 'FLUX_FW_IMG_HIDDEN');
  QNNCopy(ImgHidden, ConcatHidden + TxtSeq * HiddenSize, ImgSeq * HiddenSize);
  ConcatHidden.Free;
  QNNSilu(TEmbSilu, TEmb, HiddenSize);
  // for i := 0 to hidden_size -1 do
  //    begin
  //        x := t_emb[i];
  //        t_emb_silu[i] := x div (1.0+expf(-x))
  //    end;
  TEmb.Free;
  FinalMod := DoubleModImg;
  QNNLinearNoBias(FinalMod, TEmbSilu, FinalNormWeight, 1, HiddenSize, HiddenSize * 2);
  FinalScale := FinalMod;
  FinalShift := FinalMod + HiddenSize;
  FinalNorm := Work1;
  QNNAdaLN(FinalNorm, ImgHidden, FinalShift, FinalScale, ImgSeq, HiddenSize, -6);
  ImgHidden.Free;
  OutputNlc := TMemoryBlock.Create([ImgSeq, LatentChannels], 'FLUX_FW_OUTPUT_NLC');
  QNNLinearBF16OrF32(OutputNlc, FinalNorm, FinalProjWeight, FinalProjWeightBF16, ImgSeq, HiddenSize, LatentChannels);
  Result := TMemoryBlock.Create([ImgSeq, LatentChannels], 'FLUX_FW_RESULT');
  // output_nlc_ptr := output_nlc;
  // result_ptr := result;
  // for pos := 0 to img_seq -1 do
  //    for c := 0 to latent_channels -1 do
  //        result_ptr[c * img_seq+pos] := output_nlc_ptr[pos * latent_channels+c];
  QNNMatTranspose(Result, OutputNlc, ImgSeq, LatentChannels);
  OutputNlc.Free;
  if Assigned(SubstepCallback) then
    SubstepCallback(SUBSTEP_FINAL_LAYER, 0, 1);
end;

function TTransformerFlux.forward(const ImgLatent: TMemoryBlock; const ImgH, ImgW: LongInt; const Refs: TArray<TImageRef>; const TxtEmb: TMemoryBlock; const TxtSeq: LongInt; const Timestep: Single): TMemoryBlock;
var
  ImgSeq, AxisDim, TotalRefSeq, R, CombinedImgSeq, TotalSeq, RoPEOffset, TransposeOffset,
  {ref_seq, trans_offset,}
I, DoubleModSize: LongInt;
  TEmb, CombinedRoPECos, CombinedRoPESin, TxtRoPECos, TxtRoPESin, OutputNlc, CombinedTransposed, CombinedHidden, ConcatHidden, ImgHidden, TSincos, FinalMod, FinalScale, FinalShift, FinalNorm: TMemoryBlock;
// result_ptr, output_nlc_ptr, combined_transposed_ptr, ref_latent_ptr, img_latent_ptr : PQNNFloat;
// t_sincos : array[0..255] of QNNFloat;
begin
  if not Assigned(Refs) then
    Exit(forward(ImgLatent, ImgH, ImgW, TxtEmb, TxtSeq, Timestep));
  if (Length(Refs) = 1) then
    Exit(
      forward(ImgLatent, ImgH, ImgW, Refs[0].Latent, Refs[0].H, Refs[0].W, Refs[0].TOffset, TxtEmb, TxtSeq, Timestep)
    );

  ImgSeq := ImgH * ImgW;
  AxisDim := 32;
  TotalRefSeq := 0;
  for R := 0 to High(Refs) do
    Inc(TotalRefSeq, Refs[R].H * Refs[R].W);
  CombinedImgSeq := ImgSeq + TotalRefSeq;
  TotalSeq := CombinedImgSeq + TxtSeq;
  ReInitialize(TotalSeq);
  TEmb := TMemoryBlock.Create([HiddenSize], 'FLUX_FW_TIME_EMB');

  TSincos := QNNTimeStepEmbedding(Timestep * 1000.0, TimeEmbed.SincosDim, 10000.0);
  TimeEmbed.forward(TEmb, TSincos, HiddenSize, TEmbSilu);
  TSincos.Free;
  CombinedRoPECos := TMemoryBlock.Create([CombinedImgSeq, AxisDim, 4], 'FLUX_FW_COMBINED_ROPE_COS');
  CombinedRoPESin := TMemoryBlock.Create([CombinedImgSeq, AxisDim, 4], 'FLUX_FW_COMBINED_ROPE_SIN');
  QNNComputeRoPE2D(CombinedRoPECos, CombinedRoPESin, ImgH, ImgW, AxisDim, RoPETheta);
  RoPEOffset := ImgSeq * AxisDim * 4;
  for R := 0 to High(Refs) do
  begin
    QNNComputeRoPE2DOffset(
      CombinedRoPECos + RoPEOffset,
      CombinedRoPESin + RoPEOffset,
      Refs[R].H,
      Refs[R].W,
      AxisDim,
      RoPETheta,
      Refs[R].TOffset
    );
    Inc(RoPEOffset, Refs[R].H * Refs[R].W * AxisDim * 4)
  end;

  GetCachedTxtRoPE(TxtSeq, TxtRoPECos, TxtRoPESin);
  CombinedTransposed := TMemoryBlock.Create([CombinedImgSeq, LatentChannels], 'FLUX_FW_COMBINED_TRANSPOSED');
  // combined_transposed_ptr := combined_transposed;
  // img_latent_ptr := img_latent;
  // for pos := 0 to img_seq -1 do
  //    for c := 0 to latent_channels -1 do
  //        combined_transposed_ptr[pos * latent_channels+c] := img_latent_ptr[c * img_seq+pos];
  QNNMatTranspose(CombinedTransposed, ImgLatent, LatentChannels, ImgSeq);
  TransposeOffset := ImgSeq;
  for R := 0 to High(Refs) do
  begin
    // ref_latent_ptr := refs[r].latent;
    // ref_seq := refs[r].h * refs[r].w;
    // for pos := 0 to ref_seq -1 do
    //    for c := 0 to latent_channels -1 do
    //        combined_transposed_ptr[(trans_offset + pos)*latent_channels + c] := ref_latent_ptr[c*ref_seq + pos];
    QNNMatTranspose(
      CombinedTransposed + TransposeOffset * LatentChannels,
      Refs[R].Latent,
      LatentChannels,
      Refs[R].H * Refs[R].W
    );
    Inc(TransposeOffset, Refs[R].H * Refs[R].W);
  end;
  CombinedHidden := TMemoryBlock.Create([CombinedImgSeq, HiddenSize], 'FLUX_FW_COMBINED_HIDDEN');
  QNNLinearBF16OrF32(
    CombinedHidden,
    CombinedTransposed,
    ImgInWeight,
    ImgInWeightBF16,
    CombinedImgSeq,
    LatentChannels,
    HiddenSize
  );
  CombinedTransposed.Free;
  QNNLinearBF16OrF32(TxtHidden, TxtEmb, TxtInWeight, TxtInWeightBF16, TxtSeq, TextDim, HiddenSize);
  DoubleModSize := HiddenSize * 6;
  QNNSilu(TEmbSilu, TEmb, HiddenSize);
  // for j := 0 to hidden_size -1 do
  //    begin
  //        x := t_emb[j];
  //        tf.t_emb_silu[j] := x / (1.0+expf(-x))
  //    end;
  QNNLinearNoBias(DoubleModImg, TEmbSilu, AdalnDoubleImgWeight, 1, HiddenSize, DoubleModSize);
  QNNLinearNoBias(DoubleModTxt, TEmbSilu, AdalnDoubleTxtWeight, 1, HiddenSize, DoubleModSize);
  for I := 0 to NumDoubleLayers - 1 do
  begin
    if UseMmap then
      DoubleBlocks[I].Load(SfFiles, I, HiddenSize, MlpHidden, UseBF16);
    DoubleBlockForward(
      CombinedHidden,
      TxtHidden,
      I,
      DoubleModImg,
      DoubleModTxt,
      CombinedRoPECos,
      CombinedRoPESin,
      TxtRoPECos,
      TxtRoPESin,
      CombinedImgSeq,
      TxtSeq
    );
    if UseMmap then
      DoubleBlocks[I].Free();
    if Assigned(SubstepCallback) then
      SubstepCallback(SUBSTEP_DOUBLE_BLOCK, I, NumDoubleLayers)
  end;
  ConcatHidden := TMemoryBlock.Create([TotalSeq, HiddenSize], 'FLUX_FW_CONCAT)HIDDEN');
  QNNCopy(ConcatHidden, TxtHidden, TxtSeq * HiddenSize);
  QNNCopy(ConcatHidden + TxtSeq * HiddenSize, CombinedHidden, CombinedImgSeq * HiddenSize);
  CombinedHidden.Free;
  for I := 0 to NumSingleLayers - 1 do
  begin
    if UseMmap then
      SingleBlocks[I].Load(SfFiles, I, HiddenSize, MlpHidden, UseBF16);
    SingleBlockForward(
      ConcatHidden,
      I,
      TEmb,
      AdalnSingleWeight,
      CombinedRoPECos,
      CombinedRoPESin,
      TxtRoPECos,
      TxtRoPESin,
      TotalSeq,
      TxtSeq
    );
    if UseMmap then
      SingleBlocks[I].Free();
    if Assigned(SubstepCallback) then
      SubstepCallback(SUBSTEP_SINGLE_BLOCK, I, NumSingleLayers)
  end;
  CombinedRoPECos.Free;
  CombinedRoPESin.Free;
  ImgHidden := TMemoryBlock.Create([ImgSeq, HiddenSize], 'FLUX_FW_IMG_HIDDEN');
  QNNCopy(ImgHidden, ConcatHidden + TxtSeq * HiddenSize, ImgSeq * HiddenSize);
  ConcatHidden.Free;
  QNNSilu(TEmbSilu, TEmb, HiddenSize);
  // for i := 0 to hidden_size -1 do
  //    begin
  //        x := t_emb[i];
  //        tf.t_emb_silu[i] := x / (1.0+expf(-x))
  //    end;
  TEmb.Free();
  FinalMod := DoubleModImg;
  QNNLinearNoBias(FinalMod, TEmbSilu, FinalNormWeight, 1, HiddenSize, HiddenSize * 2);
  FinalScale := FinalMod;
  FinalShift := FinalMod + HiddenSize;
  FinalNorm := Work1;
  QNNAdaLN(FinalNorm, ImgHidden, FinalShift, FinalScale, ImgSeq, HiddenSize);
  ImgHidden.Free;
  OutputNlc := TMemoryBlock.Create([ImgSeq, LatentChannels], 'FLUX_FW_OUTPUT_NLC');
  QNNLinearBF16OrF32(OutputNlc, FinalNorm, FinalProjWeight, FinalProjWeightBF16, ImgSeq, HiddenSize, LatentChannels);
  Result := TMemoryBlock.Create([ImgSeq, LatentChannels], 'FLUX_FW_RESULT');
  // result_ptr := result;
  // output_nlc_ptr := output_nlc;
  // for pos := 0 to img_seq -1 do
  //    for c := 0 to latent_channels -1 do
  //        result_ptr[c * img_seq+pos] := output_nlc_ptr[pos * latent_channels+c];
  QNNMatTranspose(Result, OutputNlc, ImgSeq, LatentChannels);
  OutputNlc.Free;
  if Assigned(SubstepCallback) then
    SubstepCallback(SUBSTEP_FINAL_LAYER, 0, 1);
end;

function TTransformerFlux.SampleEuler(const Z: TMemoryBlock; const Batch, Channels, H, W: LongInt; const TextEmb: TMemoryBlock; const TextSeq: LongInt; const Schedule: TMemoryBlock; const NumSteps: LongInt; const ProgressCallback: TProgressCallback): TMemoryBlock;
var
  LatentSize, Step: LongInt;
  VCond: TMemoryBlock;
  TCurr, TNext, Dt: QNNFloat;
  Img: TQNNImage;
  SchedulePtr: PQNNFloat;
begin
  LatentSize := Batch * Channels * H * W;
  Result := TMemoryBlock.Create([Batch, Channels, H, W], 'FLUX_SAMPLE_EULER_RESULT');

  QNNCopy(Result, Z, LatentSize);
  SchedulePtr := Schedule;
  for Step := 0 to NumSteps - 1 do
  begin
    TCurr := SchedulePtr[Step];
    TNext := SchedulePtr[Step + 1];
    Dt := TNext - TCurr;
    if Assigned(StepCallback) then
      StepCallback(Step, NumSteps);
    VCond := forward(Result, H, W, TextEmb, TextSeq, TCurr);
    QNNFusedScaleAdd(Result, VCond, Result, Dt, LatentSize);
    VCond.Free();
    if Assigned(ProgressCallback) then
      ProgressCallback(Step, NumSteps, Result);
    if Assigned(StepImageCallback) and Assigned(VAEPtr) and (Step + 1 < NumSteps) then
    begin
      Img := PVAE(VAEPtr).Decode(Result, 1, H, W);
      if Assigned(Img.Data) then
      begin
        StepImageCallback(Step, NumSteps, Img);
        Img.Free
      end
    end
  end;
  FreeMMapCache();
end;

function TTransformerFlux.SampleEuler(const Z: TMemoryBlock; const Batch, Channels, H, W: LongInt; const RefLatent: TMemoryBlock; const RefH, RefW, TOffset: LongInt; const TextEmb: TMemoryBlock; const TextSeq: LongInt; const Schedule: TMemoryBlock; const NumSteps: LongInt; const ProgressCallback: TProgressCallback): TMemoryBlock;
var
  LatentSize, Step: LongInt;
  TCurr, TNext, Dt: QNNFloat;
  V: TMemoryBlock;
  Img: TQNNImage;
  SchedulePtr: PQNNFloat;
begin
  // assert(not ref_latent.is_Nan());
  // ref_latent.printStat;
  LatentSize := Batch * Channels * H * W;

  (* Working buffer *)
  Result := TMemoryBlock.Create([Batch, Channels, H, W], 'FLUX_SAMPLE_EULER_RESULT');
  QNNCopy(Result, Z, LatentSize);
  SchedulePtr := Schedule;
  for Step := 0 to NumSteps - 1 do
  begin
    TCurr := SchedulePtr[Step];
    TNext := SchedulePtr[Step + 1];
    Dt := TNext - TCurr;

    (* Notify step start *)
    if Assigned(StepCallback) then
      StepCallback(Step, NumSteps);
    (* Predict velocity with reference image conditioning *)
    V := forward(Result, H, W, RefLatent, RefH, RefW, TOffset, TextEmb, TextSeq, TCurr);
    (* Euler step: z_next = result + dt * v *)
    QNNFusedScaleAdd(Result, V, Result, Dt, LatentSize);

    V.Free;

    if Assigned(ProgressCallback) then
      ProgressCallback(Step, NumSteps, Result);

    (* Step image callback - decode and display intermediate result *)
    if Assigned(StepImageCallback) and Assigned(VAEPtr) and (Step + 1 < NumSteps) then
    begin
      Img := PVAE(VAEPtr).Decode(Result, 1, H, W);
      StepImageCallback(Step, NumSteps, Img);
      Img.Free;
    end
  end;

  FreeMMapCache();
end;

function TTransformerFlux.SampleEuler(const Z: TMemoryBlock; const Batch, Channels, H, W: LongInt; const TextEmbCond: TMemoryBlock; const TextSeqCond: LongInt; const TextEmbUncond: TMemoryBlock; const TextSeqUncond: LongInt; const GuidanceScale: QNNFloat; const Schedule: TMemoryBlock; const NumSteps: LongInt; const ProgressCallback: TProgressCallback): TMemoryBlock;
var
  LatentSize, Step, I: LongInt;
  TCurr, TNext, Dt, V: QNNFloat;
  VUncond, VCond: TMemoryBlock;
  VUncondPtr, VCondPtr, ResultPtr, SchedulePtr: PQNNFloat;
  Img: TQNNImage;
begin
  LatentSize := Batch * Channels * H * W;
  Result := TMemoryBlock.Create([Batch, Channels, H, W], 'FLUX_SAMPLE_EULER_RESULT');
  ResultPtr := Result;
  QNNCopy(Result, Z, LatentSize);
  SchedulePtr := Schedule;
  for Step := 0 to NumSteps - 1 do
  begin
    TCurr := SchedulePtr[Step];
    TNext := SchedulePtr[Step + 1];
    Dt := TNext - TCurr;
    if Assigned(StepCallback) then
      StepCallback(Step, NumSteps);
    VUncond := forward(Result, H, W, TextEmbUncond, TextSeqUncond, TCurr);
    VUncondPtr := VUncond;
    VCond := forward(Result, H, W, TextEmbCond, TextSeqCond, TCurr);
    VCondPtr := VCond;
    // todo sampleEuuler refactor as fused axpy
    for I := 0 to LatentSize - 1 do
    begin
      V := VUncondPtr[I] + GuidanceScale * (VCondPtr[I] - VUncondPtr[I]);
      ResultPtr[I] := ResultPtr[I] + (Dt * V)
    end;
    VUncond.Free;
    VCond.Free;
    if Assigned(ProgressCallback) then
      ProgressCallback(Step, NumSteps, Result);
    if Assigned(StepImageCallback) and Assigned(VAEPtr) and (Step + 1 < NumSteps) then
    begin
      Img := PVAE(VAEPtr).Decode(Result, 1, H, W);
      if Assigned(Img.Data) then
      begin
        StepImageCallback(Step, NumSteps, Img);
        Img.Free
      end
    end
  end;
  FreeMMapCache();
end;

function TTransformerFlux.SampleEuler(const Z: TMemoryBlock; const Batch, Channels, H, W: LongInt; const RefLatent: TMemoryBlock; const RefH, RefW, TOffset: LongInt; const TextEmbCond: TMemoryBlock; const TextSeqCond: LongInt; const TextEmbUncond: TMemoryBlock; const TextSeqUncond: LongInt; const GuidanceScale: QNNFloat; const Schedule: TMemoryBlock; const NumSteps: LongInt; const ProgressCallback: TProgressCallback): TMemoryBlock;
var
  LatentSize, Step, I: LongInt;
  TCurr, TNext, Dt, V: QNNFloat;
  VUncond, VCond: TMemoryBlock;
  VCondPtr, VUncondPtr, ResultPtr, SchedulePtr: PQNNFloat;
  Img: TQNNImage;
begin
  LatentSize := Batch * Channels * H * W;
  Result := TMemoryBlock.Create([Batch, Channels, H, W], 'FLUX_SAMPLE_EULER_RESULT');
  ResultPtr := Result;
  QNNCopy(Result, Z, LatentSize);
  SchedulePtr := Schedule;
  for Step := 0 to NumSteps - 1 do
  begin
    TCurr := SchedulePtr[Step];
    TNext := SchedulePtr[Step + 1];
    Dt := TNext - TCurr;
    if Assigned(StepCallback) then
      StepCallback(Step, NumSteps);
    VUncond := forward(Result, H, W, RefLatent, RefH, RefW, TOffset, TextEmbUncond, TextSeqUncond, TCurr);
    VCond := forward(Result, H, W, RefLatent, RefH, RefW, TOffset, TextEmbCond, TextSeqCond, TCurr);
    VCondPtr := VCond;
    VUncondPtr := VUncond;

    for I := 0 to LatentSize - 1 do
    begin
      V := VUncondPtr[I] + GuidanceScale * (VCondPtr[I] - VUncondPtr[I]);
      ResultPtr[I] := ResultPtr[I] + (Dt * V)
    end;
    VUncond.Free;
    VCond.Free;
    if Assigned(ProgressCallback) then
      ProgressCallback(Step, NumSteps, Result);
    if Assigned(StepImageCallback) and Assigned(VAEPtr) and (Step + 1 < NumSteps) then
    begin
      Img := PVAE(VAEPtr).Decode(Result, 1, H, W);
      if Assigned(Img.Data) then
      begin
        StepImageCallback(Step, NumSteps, Img);
        Img.Free
      end
    end
  end;
  FreeMMapCache()
end;

function TTransformerFlux.SampleEuler(const Z: TMemoryBlock; const Batch, Channels, H, W: LongInt; const Refs: TArray<TImageRef>; const TextEmb: TMemoryBlock; const TextSeq: LongInt; const Schedule: TMemoryBlock; const NumSteps: LongInt; const ProgressCallback: TProgressCallback): TMemoryBlock;
var
  Step, LatentSize: LongInt;
  TCurr, TNext, Dt: QNNFloat;
  V: TMemoryBlock;
  Img: TQNNImage;
  SchedulePtr: PQNNFloat;
begin
  LatentSize := Batch * Channels * H * W;

  Result := TMemoryBlock.Create([Batch, Channels, H, W], 'FLUX_SAMPLE_EULER_RESULT');
  QNNCopy(Result, Z, LatentSize);
  SchedulePtr := Schedule;
  for Step := 0 to NumSteps - 1 do
  begin
    TCurr := SchedulePtr[Step];
    TNext := SchedulePtr[Step + 1];
    Dt := TNext - TCurr;

    if Assigned(StepCallback) then
      StepCallback(Step, NumSteps);

    (* Predict velocity with multiple reference images *)
    V := forward(Result, H, W, Refs, TextEmb, TextSeq, TCurr);

    (* Euler step *)
    QNNFusedScaleAdd(Result, V, Result, Dt, LatentSize);
    V.Free;

    if Assigned(ProgressCallback) then
      ProgressCallback(Step, NumSteps, Result);

    if Assigned(StepImageCallback) and Assigned(VAEPtr) and (Step + 1 < NumSteps) then
    begin
      Img := PVAE(VAEPtr).Decode(Result, 1, H, W);
      StepImageCallback(Step, NumSteps, Img);
      Img.Free;
    end;
  end;

  FreeMMapCache();
end;

function TTransformerFlux.SampleEuler(const Z: TMemoryBlock; const Batch, Channels, H, W: LongInt; const Refs: TArray<TImageRef>; const TextEmbCond: TMemoryBlock; const TextSeqCond: LongInt; const TextEmbUncond: TMemoryBlock; const TextSeqUncond: LongInt; const GuidanceScale: QNNFloat; const Schedule: PQNNFloat; NumSteps: LongInt; const ProgressCallback: TProgressCallback): TMemoryBlock;
var
  Prediction, Unconditional: TMemoryBlock;
  Img: TQNNImage;
  ResultPtr, PredictionPtr, UnconditionalPtr: System.PSingle;
  Count: Integer;
  Dt: Single;
begin
  if (Schedule = nil) or (NumSteps < 1) then
    raise EArgumentException.Create('Invalid FLUX reference sampling schedule');
  Count := Batch * Channels * H * W;
  Result := TMemoryBlock.Create([Batch, Channels, H, W], 'FLUX_REFERENCE_CFG_RESULT');
  try
    QNNCopy(Result, Z, Count);
    ResultPtr := Result;
    for var Step := 0 to NumSteps - 1 do
    begin
      if Assigned(StepCallback) then
        StepCallback(Step, NumSteps);
      Unconditional := forward(Result, H, W, Refs, TextEmbUncond, TextSeqUncond, Schedule[Step]);
      try
        Prediction := forward(Result, H, W, Refs, TextEmbCond, TextSeqCond, Schedule[Step]);
        try
          PredictionPtr := Prediction;
          UnconditionalPtr := Unconditional;
          Dt := Schedule[Step + 1] - Schedule[Step];
          for var I := 0 to Count - 1 do
            ResultPtr[I] :=
              ResultPtr[I] + Dt * (UnconditionalPtr[I] + GuidanceScale * (PredictionPtr[I] - UnconditionalPtr[I]));
        finally
          Prediction.Free;
        end;
      finally
        Unconditional.Free;
      end;
      if Assigned(ProgressCallback) then
        ProgressCallback(Step, NumSteps, Result);
      if Assigned(StepImageCallback) and Assigned(VAEPtr) and (Step + 1 < NumSteps) then
      begin
        Img := PVAE(VAEPtr)^.Decode(Result, Batch, H, W);
        try
          StepImageCallback(Step, NumSteps, Img);
        finally
          Img.Free;
        end;
      end;
    end;
    FreeMMapCache;
  except
    Result.Free;
    raise;
  end;
end;

{ TBlockZi }

procedure TBlockZI.Load(const Files: TSafeTensorFiles; const Prefix: string; const HasModulation: Boolean; const AUseMMAP: Boolean; const resident: Boolean);

  function Projection(const Name: string; const Mapped: Boolean): TMemoryBlock;
  begin
    if resident then
      Result := Files.GetLinearWeight(Name, Mapped)
    else
      Result := Files.GetTensorDataMemBlock(Name, Mapped);
  end;

begin

  (* Attention weights *)
  AttnQWeight := Projection(Prefix + '.attention.to_q.weight', AUseMMAP);
  AttnKWeight := Projection(Prefix + '.attention.to_k.weight', AUseMMAP);
  AttnVWeight := Projection(Prefix + '.attention.to_v.weight', AUseMMAP);
  AttnOutWeight := Projection(Prefix + '.attention.to_out.0.weight', AUseMMAP);

  (* QK norm *)
  AttnNormQ := Files.GetTensorDataMemBlock(Prefix + '.attention.norm_q.weight', AUseMMAP);
  AttnNormK := Files.GetTensorDataMemBlock(Prefix + '.attention.norm_k.weight', AUseMMAP);

  (* Pre/post attention norms *)
  AttnNorm1 := Files.GetTensorDataMemBlock(Prefix + '.attention_norm1.weight', AUseMMAP);
  AttnNorm2 := Files.GetTensorDataMemBlock(Prefix + '.attention_norm2.weight', AUseMMAP);

  (* FFN weights *)
  FfnW1 := Files.GetLinearWeight(Prefix + '.feed_forward.w1.weight', AUseMMAP);
  FfnW2 := Files.GetLinearWeight(Prefix + '.feed_forward.w2.weight', AUseMMAP);
  FfnW3 := Files.GetLinearWeight(Prefix + '.feed_forward.w3.weight', AUseMMAP);

  (* FFN norms *)
  FfnNorm1 := Files.GetTensorDataMemBlock(Prefix + '.ffn_norm1.weight', AUseMMAP);
  FfnNorm2 := Files.GetTensorDataMemBlock(Prefix + '.ffn_norm2.weight', AUseMMAP);

  (* AdaLN modulation (only for modulated blocks) *)
  if HasModulation then
  begin
    AdalnWeight := Projection(Prefix + '.adaLN_modulation.0.weight', AUseMMAP);
    AdalnBias := Files.GetTensorDataMemBlock(Prefix + '.adaLN_modulation.0.bias', AUseMMAP);
  end
  else
  begin
    //      adaln_weight := nil;
    //      adaln_bias := nil;
  end
end;

procedure TBlockZI.Free();
begin
  (* Attention weights *)
  AttnQWeight.Free;
  AttnKWeight.Free;
  AttnVWeight.Free;
  AttnOutWeight.Free;
  AttnNormQ.Free;
  AttnNormK.Free;
  AttnNorm1.Free;
  AttnNorm2.Free;
  FfnW1.Free;
  FfnW2.Free;
  FfnW3.Free;
  FfnNorm1.Free;
  FfnNorm2.Free;
  if AdalnWeight.IsAssigned() then
    AdalnWeight.Free;
  if AdalnBias.IsAssigned() then
    AdalnBias.Free;
end;

{ TTransformerZi }

(* Converts scalar timestep to a 256-dim vector using log-spaced frequencies,
 * the same idea as the original Transformer positional encoding but here it
 * encodes the denoising step. The caller scales the input by 1000 before
 * calling (t * 1000.0f), mapping the [0,1] sigma range to [0,1000]. *)
// compute rope ?
procedure SinusoidalEmbeddingZi(const Dst: PQNNFloat; const T: QNNFloat; const Dim: LongInt);
var
  Half, I: LongInt;
  Freq, Angle: QNNFloat;
const
  LOG_MAX_PERIOD: QNNFloat = 9.2103403719761827360719658187375; // ln(10000.0);
begin
  Half := Dim div 2;
  // log_max_period = ln(10000.0);
  for I := 0 to Half - 1 do
  begin
    Freq := Exp(-LOG_MAX_PERIOD * I / Half);
    Angle := T * Freq;
    Dst[I] := Cos(Angle);
    Dst[I + Half] := Sin(Angle);
  end
end;

procedure TTransformerZI.AttentionForward(const Dst, Src: TMemoryBlock; const Block: TBlockZI; const PosIds: PLongint; const Mask: PLongint; const Seq: LongInt);
var
  Q, K, V, AttnOut
      {, qi, kj, oi, vj}: TMemoryBlock;
  Scores: TMemoryBlock;
  Scale: QNNFloat;
begin

  Q := WorkQKV;
  K := Q + Seq * Dim;
  V := K + Seq * Dim;

  (* Q, K, V projections *)
  QNNMatMulNT(Q, Src, Block.AttnQWeight, Seq, Dim, Dim);
  QNNMatMulNT(K, Src, Block.AttnKWeight, Seq, Dim, Dim);
  QNNMatMulNT(V, Src, Block.AttnVWeight, Seq, Dim, Dim);

  (* QK normalization *)
  QNNRMSNormSeq(Q, Block.AttnNormQ, Seq, NHeads, HeadDim);
  QNNRMSNormSeq(K, Block.AttnNormK, Seq, NHeads, HeadDim);

  (* Apply RoPE *)
  ApplyRope(Q, PosIds, Seq, NHeads);
  ApplyRope(K, PosIds, Seq, NHeads);

  (* Scaled dot-product attention per head *)
  Scale := 1.0 / Sqrt(HeadDim);
  AttnOut := WorkFfn;

  Scores := WorkAttn;
  for var H := 0 to NHeads - 1 do
  begin

    (* Compute Q @ K^T for this head *)
    // for i := 0 to seq-1 do begin
    //    qi := Q + i * dim + h * head_dim;
    //    for j := 0 to seq-1 do begin
    //        kj := K + j * dim + h * head_dim;
    //        dot := 0;
    //        for d := 0 to head_dim-1 do
    //            dot := dot + qi[d]*kj[d];
    //        scores[i*seq + j] := dot*scale;
    //    end
    // end;
    CblasGemm(
      CblasRowMajor,
      CblasNoTrans,
      CblasTrans,
      Seq,
      Seq,
      HeadDim,
      Scale,
      Q + H * HeadDim,
      Dim,
      K + H * HeadDim,
      Dim,
      0,
      Scores,
      Seq
    );

    (* Apply mask: set padding positions to -inf *)
    if Assigned(Mask) then
      QNNMaskFill(Scores, Mask, Single.NegativeInfinity, Seq, Seq);
    // for i := 0 to seq-1 do
    //    for j := 0 to seq-1 do
    //        if mask[j]=0 then
    //            scores[i*seq + j] := -1e9;

    (* Softmax *)
    QNNSoftmaxRows(Scores, Seq, Seq);

    (* Scores @ V *)
    // for i := 0 to seq-1 do begin
    //    oi := attn_out + i*dim + h*head_dim;
    //    QNNFill(oi, 0, head_dim);
    //    for j  := 0 to seq-1 do begin
    //        s  := scores[i * seq + j];
    //        vj := V + j * dim + h * head_dim;
    //        for d := 0 to head_dim-1 do
    //            oi[d] := oi[d] + s*vj[d];
    //    end;
    // end;
    CblasGemm(
      CblasRowMajor,
      CblasNoTrans,
      CblasNoTrans,
      Seq,
      HeadDim,
      Seq,
      1.0,
      Scores,
      Seq,
      V + H * HeadDim,
      Dim,
      0,
      AttnOut + H * HeadDim,
      Dim
    )
  end;
  QNNMatMulNT(Dst, AttnOut, Block.AttnOutWeight, Seq, Dim, Dim);
end;

procedure TTransformerZI.FfnForward(const Dst, Src: TMemoryBlock; const Block: TBlockZI; const Seq: LongInt);
var
  Gate, Up: TMemoryBlock;
  N: LongInt;
begin
  if Seq <= 0 then
    Exit;
  if TryVulkanSwiGLU(Dst, Src, Block.FfnW1, Block.FfnW3, Block.FfnW2, Seq, Dim, FfnDim) then
    Exit;
  N := Seq * FfnDim;
  Gate := WorkFfn;
  Up := Gate + N;

  (* W1 (gate) and W3 (up) projections *)
  QNNMatMulNT(Gate, Src, Block.FfnW1, Seq, Dim, FfnDim);
  QNNMatMulNT(Up, Src, Block.FfnW3, Seq, Dim, FfnDim);

  (* SiLU(gate) * up *)
  QNNSiluInplace(Gate, N);
  // for i := 0 to n-1 do gate[i] := gate[i]*up[i];
  QNNMulInplace(Gate, Up, N);

  (* W2 (down) projection *)
  QNNMatMulNT(Dst, Gate, Block.FfnW2, Seq, FfnDim, Dim);
end;

procedure TTransformerZI.BlockForward(const Dst: TMemoryBlock; const Block: TBlockZI; const PosIds: PLongint; const Mask: PLongint; const TEmb: TMemoryBlock; const Seq: LongInt);
var
  LocalMod: array[0..4 * {dim} 3840 - 1] of QNNFloat;
  ModMem: TMemoryBlock;
  N, S: LongInt;
  ScaleMsa, GateMsa, ScaleMlp, GateMlp: TMemoryBlock;
  NormOut, AttnOut, FfnOut, Scaled: TMemoryBlock;
begin
  N := Seq * Dim;
  AttnOut := WorkTmp;
  NormOut := AttnOut + N;
  Scaled := AttnOut + 2 * N;
  FfnOut := AttnOut + 3 * N;
  if not WorkTmp.IsAssigned() or (MaxSeq < Seq) then
    Exit();
  ModMem.AssignPtr(PQNNFloat(@LocalMod[0]), [Length(LocalMod)]);
  if Block.AdalnWeight.IsAssigned() then
  begin
    Assert(TEmb.IsAssigned(), 'ERROR : blockForward t_emb not assigned!');
    QNNMatMulNT(ModMem, TEmb, Block.AdalnWeight, 1, AdalnDim, 4 * Dim);
    // for i := 0 to 4 * dim -1 do
    //    _mod[i] := _mod[i] + block.adaln_bias[i];
    QNNAddInplace(ModMem, Block.AdalnBias, 4 * Dim);
    ScaleMsa := ModMem;
    GateMsa := ModMem + Dim;
    ScaleMlp := ModMem + 2 * Dim;
    GateMlp := ModMem + 3 * Dim;
    // for i := 0 to dim -1 do
    //    begin
    //        scale_msa[i] := 1.0+scale_msa[i];
    //        gate_msa[i] := tanh(gate_msa[i]);
    //        scale_mlp[i] := 1.0+scale_mlp[i];
    //        gate_mlp[i] := tanh(gate_mlp[i])
    //    end;
    QNNBiasInplace(ScaleMsa, 1.0, Dim);
    QNNTanh(GateMsa, Dim);
    QNNBiasInplace(ScaleMlp, 1.0, Dim);
    QNNTanh(GateMlp, Dim);
    QNNRMSNormRows(NormOut, Dst, Block.AttnNorm1, Seq, Dim);

    for S := 0 to Seq - 1 do
      // for i := 0 to dim -1 do
      //    scaled[s * dim+i] := norm_out[s * dim+i] * scale_msa[i];
      QNNMul(Scaled + S * Dim, NormOut + S * Dim, ScaleMsa, Dim);

    AttentionForward(AttnOut, Scaled, Block, PosIds, Mask, Seq);
    QNNRMSNormRows(NormOut, AttnOut, Block.AttnNorm2, Seq, Dim);
    for S := 0 to Seq - 1 do
      // for i := 0 to dim -1 do
      //    dst[s * dim+i] := dst[s * dim+i] + (gate_msa[i] * norm_out[s * dim+i]);
      QNNFusedMulAdd(Dst + S * Dim, GateMsa, NormOut + S * Dim, Dst + S * Dim, Dim);
    QNNRMSNormRows(NormOut, Dst, Block.FfnNorm1, Seq, Dim);
    for S := 0 to Seq - 1 do
      QNNMul(Scaled + S * Dim, NormOut + S * Dim, ScaleMlp, Dim);
    // for i := 0 to dim -1 do
    //    scaled[s * dim+i] := norm_out[s * dim+i] * scale_mlp[i];
    FfnForward(FfnOut, Scaled, Block, Seq);
    QNNRMSNormRows(NormOut, FfnOut, Block.FfnNorm2, Seq, Dim);
    for S := 0 to Seq - 1 do
      QNNFusedMulAdd(Dst + S * Dim, GateMlp, NormOut + S * Dim, Dst + S * Dim, Dim);
    // for i := 0 to dim -1 do
    //    dst[s * dim+i] := dst[s * dim+i] + (gate_mlp[i] * norm_out[s * dim+i])
  end
  else
  begin
    QNNRMSNormRows(NormOut, Dst, Block.AttnNorm1, Seq, Dim);
    AttentionForward(AttnOut, NormOut, Block, PosIds, Mask, Seq);
    QNNRMSNormRows(NormOut, AttnOut, Block.AttnNorm2, Seq, Dim);
    // for i := 0 to N -1 do
    //    dst[i] := dst[i] + norm_out[i];
    QNNAddInplace(Dst, NormOut, N);
    QNNRMSNormRows(NormOut, Dst, Block.FfnNorm1, Seq, Dim);
    FfnForward(FfnOut, NormOut, Block, Seq);
    QNNRMSNormRows(NormOut, FfnOut, Block.FfnNorm2, Seq, Dim);
    // for i := 0 to N -1 do
    //    dst[i] := dst[i] + norm_out[i]
    QNNAddInplace(Dst, NormOut, N);
  end
end;

procedure TTransformerZI.PreComputeRope;
var
  Ax, D, HalfD, Pos, MaxPos, I: LongInt;
  Freq, Angle: QNNFloat;
  LocalRopeCos, LocalRopeSin: PQNNFloat;
begin
  for Ax := 0 to 3 - 1 do
  begin
    D := AxesDims[Ax];
    HalfD := D div 2;
    MaxPos := AxesLens[Ax];

    RoPECos[Ax] := TMemoryBlock.Create([MaxPos, HalfD], 'ZI_PRECOMPUTE_ROPE_COS_' + IntToStr(Ax));
    RoPESin[Ax] := TMemoryBlock.Create([MaxPos, HalfD], 'ZI_PRECOMPUTE_ROPE_SIN_' + IntToStr(Ax));
    LocalRopeCos := RoPECos[Ax];
    LocalRopeSin := RoPESin[Ax];
    for Pos := 0 to MaxPos - 1 do
    begin
      for I := 0 to HalfD - 1 do
      begin
        Freq := 1.0 / System.Math.Power(RoPETheta, (2 * I) / D);
        Angle := Pos * Freq;
        LocalRopeCos[Pos * HalfD + I] := Cos(Angle);
        LocalRopeSin[Pos * HalfD + I] := Sin(Angle);
      end;
    end
  end
end;

procedure TTransformerZI.FinalComputeScale(const Scale, TEmb: TMemoryBlock);
var
  SiluEmb: array[0..255] of QNNFloat;
  SiluEmbMem: TMemoryBlock;
begin
  SiluEmbMem.AssignPtr(PQNNFloat(@SiluEmb[0]), [Length(SiluEmb)]);
  QNNCopy(SiluEmbMem, TEmb, AdalnDim);
  QNNSiluInplace(SiluEmbMem, AdalnDim);

  QNNMatMulNT(Scale, SiluEmbMem, FinalLayer.AdalnWeight, 1, AdalnDim, Dim);
  // for (int i = 0; i < dim; i++)
  //    scale[i] = 1.0 + scale[i] + adaln_bias[i];
  QNNFusedBiasAdd(Scale, FinalLayer.AdalnBias, Scale, 1.0, Dim);
end;

procedure TTransformerZI.FinalForward(const Dst, Src, TEmb: TMemoryBlock; const Seq: LongInt);
var
  OutDim, S: LongInt;
  Scale, Normed: TMemoryBlock;
// normed_ptr : PQNNFloat;
// mean, variance, inv_std : QNNFloat;
begin
  OutDim := PatchSize * PatchSize * InChannels;

  Scale := TMemoryBlock.Create([Dim], 'ZI_FINAL_SCALE');
  FinalComputeScale(Scale, TEmb);

  (* LayerNorm (no affine) -> scale *)
  Normed := TMemoryBlock.Create([Seq, Dim], 'ZI_FINAL_NORMED');
  for S := 0 to Seq - 1 do
  begin
    // xr := src + s * dim;
    // nr := normed + s * dim;

    (* Compute mean and variance *)
    // mean := QNNMean(dim, xr);
    // variance := QNNVariance(dim, xr, mean);
    // inv_std = 1.0 / sqrt(variance + 1e-6); (* Final LayerNorm uses 1e-6 *)
    // QNNFusedBiasMulScale(nr, xr, scale, -mean, inv_std, dim);
    QNNNorm(Dim, Normed + S * Dim, Src + S * Dim, Scale)
  end;

  (* Linear projection: dim -> out_dim *)
  QNNMatMulNT(Dst, Normed, FinalLayer.LinearWeight, Seq, Dim, OutDim);
  for S := 0 to Seq - 1 do
    // for i = 0 to out_dim-1 do
    //    dst[s*out_dim + i] := dst[s*out_dim + i] + linear_bias[i];
    QNNAddInplace(Dst + S * OutDim, FinalLayer.LinearBias, OutDim);
  Scale.Free;
  Normed.Free;
end;

(* ========================================================================
 * Patchify / Unpatchify
 * ======================================================================== *)

(* Converts latent [in_ch, H, W] to patch sequence [n_patches, ps*ps*in_ch].
 * Gathers each ps x ps spatial block into a flat vector, ordering as
 * (ph, pw, channel). This is the inverse of unpatchify and creates the
 * token sequence the transformer operates on. *)
procedure PatchifyZi(const Dst: PQNNFloat; const Src: PQNNFloat; const InCh, Height, Width, PatchSize: LongInt);
var
  HTokens, WTokens,
  // patch_feat, patch_idx, di, sx, sy
H, W, C, Ph, Pw, InIdx, OutIdx: LongInt;
begin
  HTokens := Height div PatchSize;
  WTokens := Width div PatchSize;
  // patch_feat := patch_size * patch_size * in_ch;

  for H := 0 to HTokens - 1 do
  begin
    for W := 0 to WTokens - 1 do
    begin

      // di        := h*W_tokens*patch_size*patch_size*in_ch + w*patch_size*patch_size*in_ch;

      (* Gather patch: iterate (ph, pw, c) *)
      for Ph := 0 to PatchSize - 1 do
        for Pw := 0 to PatchSize - 1 do
          for C := 0 to InCh - 1 do
          begin
            InIdx :=
              C * HTokens * PatchSize * WTokens * PatchSize
              + H * PatchSize * WTokens * PatchSize
              + Ph * WTokens * PatchSize
              + W * PatchSize
              + Pw;
            //  [in_ch, H_Tokens, patch_size, W_tokens, patch_size]
            OutIdx :=
              H * WTokens * PatchSize * PatchSize * InCh
              + W * PatchSize * PatchSize * InCh
              + Ph * PatchSize * InCh
              + Pw * InCh
              + C;
            //  [H_Tokens, W_tokens, patch_size, patch_size, in_ch]
            // transpose -> [1, 3, 2, 4, 0]
            Dst[OutIdx] := Src[InIdx];
          end
    end
  end
end;

(* Unpatchify: [n_patches, patch_feat_dim] -> [in_ch, H, W] *)
procedure UnPatchifyZi(const Dst: PQNNFloat; const Src: PQNNFloat; const InCh, Height, Width, PatchSize: LongInt);
var
  // patch_idx, patch_feat, si, sy, sx
  InIdx, OutIdx, HTokens, WTokens, H, W, Ph, Pw, C: LongInt;
// d : PQNNFloat;
begin
  HTokens := Height div PatchSize;
  WTokens := Width div PatchSize;
  // patch_feat := patch_size * patch_size * in_ch;

  for H := 0 to HTokens - 1 do
    for W := 0 to WTokens - 1 do
    begin
      // patch_idx := h*W_tokens + w;
      // si := 0;
      for Ph := 0 to PatchSize - 1 do
        for Pw := 0 to PatchSize - 1 do
          for C := 0 to InCh - 1 do
          begin
            InIdx :=
              H * WTokens * PatchSize * PatchSize * InCh
              + W * PatchSize * PatchSize * InCh
              + Ph * PatchSize * InCh
              + Pw * InCh
              + C;
            // [H_tokens, W_tokens, patch_size, patch_size, in_ch]
            OutIdx :=
              C * HTokens * PatchSize * WTokens * PatchSize
              + H * PatchSize * WTokens * PatchSize
              + Ph * WTokens * PatchSize
              + W * PatchSize
              + Pw;
            // [in_ch, H_tokens, patch_size, W_tokens, patch_size]
            // transpose -> [4, 0, 2, 1, 3]
            Dst[OutIdx] := Src[InIdx];
            // inc(si)
          end
    end
end;

function TTransformerZI.forward(const Latent: TMemoryBlock; const LatentH, LatentW: LongInt; const Timestep: QNNFloat; const CapFeats: TMemoryBlock; const CapSeqLen: LongInt): TMemoryBlock;
var
  PatchFeat, HTokens, WTokens, ImgSeq, RefinerTotal, ImgPad, CapPad, ImgPadded, CapPadded, UnifiedSeq, Needed, I, S, H, W, OutCh: LongInt;
  TEmb: array[0..255] of QNNFloat;
  ImgPatches, ImgEmb, CapEmb, CapNormed, CapPaddedFeats, ImgOut, FinalOut, TEmbMem: TMemoryBlock;
  CapPos, ImgPos, UnifiedPos: TArray<LongInt>;
  Unified: TMemoryBlock;
begin
  TEmbMem.AssignPtr(PQNNFloat(@TEmb[0]), [Length(TEmb)]);
  PatchFeat := PatchSize * PatchSize * InChannels; (* 64 *)

  HTokens := LatentH div PatchSize;
  WTokens := LatentW div PatchSize;
  ImgSeq := HTokens * WTokens;
  RefinerTotal := NRefiner * 2;

  (* Pad sequences to multiples of ZI_SEQ_MULTI_OF *)
  ImgPad := (ZI_SEQ_MULTI_OF - (ImgSeq mod ZI_SEQ_MULTI_OF)) mod ZI_SEQ_MULTI_OF;
  CapPad := (ZI_SEQ_MULTI_OF - (CapSeqLen mod ZI_SEQ_MULTI_OF)) mod ZI_SEQ_MULTI_OF;
  ImgPadded := ImgSeq + ImgPad;
  CapPadded := CapSeqLen + CapPad;
  UnifiedSeq := ImgPadded + CapPadded;

  (* Ensure working memory is sufficient *)
  Needed :=
    UnifiedSeq * Dim * 4
    + UnifiedSeq * Dim * 3
    + (* QKV *) UnifiedSeq * UnifiedSeq
    + (* attention scores *) UnifiedSeq * FfnDim * 2;
  if Needed > WorkAlloc then
  begin
    WorkX.Free;
    WorkTmp.Free;
    WorkQKV.Free;
    WorkAttn.Free;
    WorkFfn.Free;

    WorkX := TMemoryBlock.Create([UnifiedSeq, Dim], 'ZI_FW_work_x');
    WorkTmp := TMemoryBlock.Create([UnifiedSeq, Dim, 4], 'ZI_FW_work_tmp');
    WorkQKV := TMemoryBlock.Create([UnifiedSeq, Dim, 3], 'ZI_FW_work_qkv');
    WorkAttn := TMemoryBlock.Create([UnifiedSeq, UnifiedSeq], 'ZI_FW_work_attn');
    WorkFfn := TMemoryBlock.Create([UnifiedSeq, FfnDim, 2], 'ZI_FW_work_ffn');
    WorkAlloc := Needed;
    MaxSeq := UnifiedSeq;
  end;

  (* 1. Timestep embedding *)

  TimeStepEmbed(TEmbMem, Timestep);

  (* 2. Patchify image -> [img_seq, patch_feat] *)
  // transpose tensor :
  //   in shape : [in_ch, latent_h div patch_size, patch_size, latent_w div patch_size, patch_size]
  //   out shape : [latent_h div patch_size, latent_w div patch_size, patch_size, patch_size, in_channels]

  ImgPatches := TMemoryBlock.Create([ImgPadded, PatchFeat], 'ZI_FW_IMG_PATCHES');
  // patchifyZi(img_patches, latent, in_channels, latent_h, latent_w, patch_size);
  Latent.Reshape([InChannels, LatentH div PatchSize, PatchSize, LatentW div PatchSize, PatchSize]);
  QNNPermut(ImgPatches, Latent, [1, 3, 2, 4, 0]);
  // img_patches_ptr := img_patches;

  (* Pad image patches (repeat last token) *)
  // todo ZI.Forward  use QNNBroadcast instead of copy loop
  for I := ImgSeq to ImgPadded - 1 do
    QNNCopy(ImgPatches + I * PatchFeat, ImgPatches + (ImgSeq - 1) * PatchFeat, PatchFeat);

  (* Embed image: [img_padded, patch_feat] -> [img_padded, dim] *)
  ImgEmb := TMemoryBlock.Create([ImgPadded, Dim], 'ZI_FW_IMG_EMB');
  // img_emb_ptr := img_emb;
  QNNMatMulNT(ImgEmb, ImgPatches, XEmbWeight, ImgPadded, PatchFeat, Dim);
  for S := 0 to ImgPadded - 1 do
    QNNAddInplace(ImgEmb + S * Dim, XEmbBias, Dim);
  // for i := 0 to dim-1 do
  //    img_emb[s * dim + i] := img_emb[s * dim + i] + x_emb_bias[i];
  ImgPatches.Free;

  (* Apply pad token to image padding positions *)
  // todo ZI.Forward  use QNNBroadcast instead of copy loop
  for S := ImgSeq to ImgPadded - 1 do
    QNNCopy(ImgEmb + S * Dim, XPadToken, Dim);

  (* 3. Caption embedding: RMSNorm -> Linear *)
  CapEmb := TMemoryBlock.Create([CapPadded, Dim], 'ZI_FW_CAP_EMB');
  // cap_emb_ptr := cap_emb;
  CapNormed := TMemoryBlock.Create([CapPadded, CapFeatDim], 'ZI_FW_CAP_NORMED');

  (* Pad caption features (repeat last token) *)
  CapPaddedFeats := TMemoryBlock.Create([CapPadded, CapFeatDim], 'ZI_FW_CAP_PADDED_FEATS');
  // cap_padded_feats_ptr := cap_padded_feats;
  QNNCopy(CapPaddedFeats, CapFeats, CapSeqLen * CapFeatDim);
  (* Apply pad token to image padding positions *)
  // todo ZI.Forward  use QNNBroadcast instead of copy loop
  for S := CapSeqLen to CapPadded - 1 do
    QNNCopy(CapPaddedFeats + S * CapFeatDim, CapFeats + (CapSeqLen - 1) * CapFeatDim, CapFeatDim);

  QNNRMSNormRows(CapNormed, CapPaddedFeats, CapEmbNorm, CapPadded, CapFeatDim);
  QNNMatMulNT(CapEmb, CapNormed, CapEmbLinearW, CapPadded, CapFeatDim, Dim);

  CapPaddedFeats.Free;
  CapNormed.Free;

  for S := 0 to CapSeqLen - 1 do // cap_padded-1 do  // todo should be to cap_seq_len-1, see the next fot to know why
    // for (int i = 0; i < dim; i++)
    //    cap_emb[s * dim + i] += tf->cap_emb_linear_b[i];
    QNNAddInplace(CapEmb + S * Dim, CapEmbLinearB, Dim);

  (* Apply pad token to caption padding positions *)
  for S := CapSeqLen to CapPadded - 1 do
    QNNCopy(CapEmb + S * Dim, CapPadToken, Dim);

  (* 4. Build position IDs *)

  (* Image position IDs: (T=cap_padded+1, H=h_idx, W=w_idx)
     * All image tokens share the same T position (one frame). *)
  SetLength(ImgPos, ImgPadded * 3); // (int *)calloc(img_padded * 3, sizeof(int));

  for H := 0 to HTokens - 1 do
  begin
    for W := 0 to WTokens - 1 do
    begin
      I := H * WTokens + W;
      ImgPos[I * 3 + 0] := CapPadded + 1; (* T (same for all) *)
      ImgPos[I * 3 + 1] := H; (* H *)
      ImgPos[I * 3 + 2] := W; (* W *)
    end
  end;
  (* Padding tokens get (0, 0, 0) *)

  (* Caption position IDs: (T=1+seq_idx, H=0, W=0) *)
  SetLength(CapPos, CapPadded * 3);
  for S := 0 to CapPadded - 1 do
  begin
    CapPos[S * 3 + 0] := 1 + S; (* T *)
    CapPos[S * 3 + 1] := 0; (* H *)
    CapPos[S * 3 + 2] := 0; (* W *)
  end;

  (* 5. Noise refiner: image-only self-attention with modulation *)
  for I := 0 to NRefiner - 1 do
  begin
    BlockForward(ImgEmb, NoiseRefiner[I], Pointer(ImgPos), nil, TEmbMem, ImgPadded);
    if Assigned(SubstepCallback) then
      SubstepCallback(SUBSTEP_DOUBLE_BLOCK, I, RefinerTotal);
  end;

  (* 6. Context refiner: caption-only self-attention without modulation *)
  for I := 0 to NRefiner - 1 do
  begin
    BlockForward(CapEmb, ContextRefiner[I], Pointer(CapPos), nil, Default(TMemoryBlock), CapPadded);
    if Assigned(SubstepCallback) then
      SubstepCallback(SUBSTEP_DOUBLE_BLOCK, NRefiner + I, RefinerTotal);
  end;

  (* 7. Build unified sequence: [image_tokens, caption_tokens] *)
  Unified := WorkX;
  QNNCopy(Unified, ImgEmb, ImgPadded * Dim);
  QNNCopy(Unified + ImgPadded * Dim, CapEmb, CapPadded * Dim);
  ImgEmb.Free;
  CapEmb.Free;

  (* Unified position IDs *)
  SetLength(UnifiedPos, UnifiedSeq * 3);
  Move(ImgPos[0], UnifiedPos[0], ImgPadded * 3 * SizeOf(LongInt));
  Move(CapPos[0], UnifiedPos[ImgPadded * 3], CapPadded * 3 * SizeOf(LongInt));

  (* 8. Main transformer layers *)
  for I := 0 to NLayers - 1 do
  begin
    BlockForward(Unified, Layers[I], Pointer(UnifiedPos), nil, TEmbMem, UnifiedSeq);
    if Assigned(SubstepCallback) then
      SubstepCallback(SUBSTEP_SINGLE_BLOCK, I, NLayers);
  end;

  SetLength(UnifiedPos, 0);

  (* 9. Final layer: extract image tokens only, then project *)
  ImgOut := TMemoryBlock.Create([ImgSeq, Dim], 'ZI_FW_IMG_OUT');
  QNNCopy(ImgOut, Unified, ImgSeq * Dim);

  OutCh := PatchSize * PatchSize * InChannels; (* 64 *)
  FinalOut := TMemoryBlock.Create([ImgSeq * OutCh], 'ZI_FW_FINALE_OUT');
  FinalForward(FinalOut, ImgOut, TEmbMem, ImgSeq);
  ImgOut.Free();
  if Assigned(SubstepCallback) then
    SubstepCallback(SUBSTEP_FINAL_LAYER, 0, 1);
  (* 10. Unpatchify: [n_patches, 64] -> [16, latent_h, latent_w] *)
  // result := TMemoryBlock.Create([in_channels, latent_h, latent_w], 'ZI_FW_RESULT');
  // transpose:
  // in shape   :  //[H_tokens, W_tokens, patch_size, patch_size, in_ch]
  // out shape  :  //[in_ch, H_tokens, patch_size, W_tokens, patch_size]
  FinalOut.Reshape([LatentH div PatchSize, LatentW div PatchSize, PatchSize, PatchSize, InChannels]);
  Result := QNNPermut(FinalOut, [4, 0, 2, 1, 3]);
  // transpose -> [4, 0, 2, 1, 3]
  //    UnpatchifyZi(result, final_out, in_channels, latent_h, latent_w, patch_size);

  FinalOut.Free();
end;

procedure TTransformerZI.ApplyRope(const Dst: PQNNFloat; const PosIds: PLongint; const Seq, NHeads: LongInt);
var
  Offset, D, HalfD, S, Pos, H, I, Ax: LongInt;
  CosTab, SinTab, Head, LocalRopeCos, LocalRopeSin: PQNNFloat;
  X0, X1, C, Sn: QNNFloat;
begin
  Offset := 0;

  for Ax := 0 to 3 - 1 do
  begin
    D := AxesDims[Ax];
    HalfD := D div 2;
    LocalRopeCos := RoPECos[Ax];
    LocalRopeSin := RoPESin[Ax];

    for S := 0 to Seq - 1 do
    begin
      Pos := PosIds[S * 3 + Ax];
      if (Pos < 0) or (Pos >= AxesLens[Ax]) then
        Continue;

      CosTab := LocalRopeCos + Pos * HalfD;
      SinTab := LocalRopeSin + Pos * HalfD;

      for H := 0 to NHeads - 1 do
      begin
        Head := Dst + (S * NHeads + H) * HeadDim + Offset;
        for I := 0 to HalfD - 1 do
        begin
          X0 := Head[2 * I];
          X1 := Head[2 * I + 1];
          C := CosTab[I];
          Sn := SinTab[I];
          Head[2 * I] := X0 * C - X1 * Sn;
          Head[2 * I + 1] := X1 * C + X0 * Sn;
        end
      end
    end;
    Inc(Offset, D);
  end
end;

(* Converts scalar timestep to a 256-dim vector using log-spaced frequencies,
 * the same idea as the original Transformer positional encoding but here it
 * encodes the denoising step. The caller scales the input by 1000 before
 * calling (t * 1000.0f), mapping the [0,1] sigma range to [0,1000]. *)
procedure TTransformerZI.TimeStepEmbed(const Dst: TMemoryBlock; const T: QNNFloat);
var
  SinEmb: array[0..255] of QNNFloat;
  Hidden, SinEmbMem: TMemoryBlock;
begin
  SinEmbMem.AssignPtr(@SinEmb[0], [Length(SinEmb)]);
  SinusoidalEmbeddingZi(PQNNFloat(@SinEmb[0]), T * 1000.0, 256);

  (* MLP: Linear(256 -> t_emb_mid_size) + SiLU + Linear(t_emb_mid_size -> adaln_dim) *)
  Hidden := TMemoryBlock.Create([TEmbMidSize], 'ZI_TIMESTEP_EMBED');

  (* Linear 0 *)
  QNNMatMulNT(Hidden, SinEmbMem, TEmbMlp0Weight, 1, 256, TEmbMidSize);
  QNNAddInplace(Hidden, TEmbMlp0Bias, TEmbMidSize);
  // for i := 0 to t_emb_mid_size-1 do
  //    hidden[i] := hidden[i] + t_emb_mlp0_bias[i];

  (* SiLU *)
  QNNSiluInplace(Hidden, TEmbMidSize);

  (* Linear 2 *)
  QNNMatMulNT(Dst, Hidden, TEmbMlp2Weight, 1, TEmbMidSize, AdalnDim);
  QNNAddInplace(Dst, TEmbMlp2Bias, AdalnDim);
  // for i := 0 to adaln_dim-1 do
  //    dst[i] := dst[i] + t_emb_mlp2_bias[i];

  Hidden.Free();
end;

function TTransformerZI.SampleEuler(const Z: TMemoryBlock; const Batch, H, W, PatchSize: LongInt; const CapFeats: TMemoryBlock; const CapSeq: LongInt; const Schedule: PQNNFloat; const NumSteps: LongInt; const ProgressCallback: TStepCallback): TMemoryBlock;
var
  LatentSize, StepH, StepW, StepCh {, step_latent_size}, Step, DecodeH, DecodeW: LongInt;
  Sigma, SigmaNext, Dt, Timestep: QNNFloat;
  StepLatent, ModelOut: TMemoryBlock;
  Img: TQNNImage;
begin
  LatentSize := Batch * InChannels {channels} * H * W;
  Result := TMemoryBlock.Create([Batch, InChannels {channels}, H, W], 'ZI_SAMPLE_EULER_RESULT');
  QNNCopy(Result, Z, LatentSize);
  //    step_latent := nil;
  StepH := H;
  StepW := W;
  if Assigned(StepImageCallback) and Assigned(VAEPtr) and (NumSteps > 1) and (PatchSize > 1) then
  begin
    if (H mod PatchSize = 0) and (W mod PatchSize = 0) then
    begin
      StepCh := InChannels {channels} * PatchSize * PatchSize;
      StepH := H div PatchSize;
      StepW := W div PatchSize;
      // step_latent_size := batch * step_ch * step_h * step_w;
      StepLatent := TMemoryBlock.Create([Batch, StepCh, StepH, StepW], 'ZI_SAMPEL_EULER_STEP_LATENT')
    end
  end;
  for Step := 0 to NumSteps - 1 do
  begin
    Sigma := Schedule[Step];
    SigmaNext := Schedule[Step + 1];
    Dt := SigmaNext - Sigma;
    Timestep := 1.0 - Sigma;
    if Assigned(StepCallback) then
      StepCallback(Step, NumSteps);
    ModelOut := forward(Result, H, W, Timestep, CapFeats, CapSeq);

    // for i := 0 to latent_size -1 do
    //    result[i] := result[i] + (dt * (-model_out[i]));
    QNNFusedScaleAdd(Result, ModelOut, Result, -Dt, LatentSize);
    ModelOut.Free;
    if Assigned(ProgressCallback) then
      ProgressCallback(Step + 1, NumSteps);
    if Assigned(StepImageCallback) and Assigned(VAEPtr) and (Step + 1 < NumSteps) then
    begin
      DecodeH := H;
      DecodeW := W;
      if PatchSize > 1 then
      begin
        if not StepLatent.IsAssigned() then
          Continue;
        // QNNPatchify(step_latent, result, batch, in_channels{channels}, h, w, patch_size);
        Result.Reshape([Batch, InChannels, H div PatchSize, PatchSize, W div PatchSize, PatchSize]);
        QNNPermut(StepLatent, Result, [0, 1, 3, 5, 2, 4]);
        DecodeH := StepH;
        DecodeW := StepW
      end;
      Img := PVAE(VAEPtr).Decode(StepLatent, 1, DecodeH, DecodeW);
      StepImageCallback(Step, NumSteps, Img);
      Img.Free
    end
  end;
  StepLatent.Free;
end;

procedure TTransformerZI.Load(const ModelDir: string; const Adim, AnHeads, AnLayers, AnRefiner, AcapFeatDim, AinChannels, ApatchSize: LongInt; const AropeTheta: QNNFloat; const AAxesDims: PLongint; const AUseMMAP: Boolean; const resident: Boolean);
var
  I: LongInt;
  Sf: PSafeTensor;

  function Projection(const Name: string; const Mapped: Boolean): TMemoryBlock;
  begin
    if resident then
      Result := SfFiles.GetLinearWeight(Name, Mapped)
    else
      Result := SfFiles.GetTensorDataMemBlock(Name, Mapped);
  end;

begin
  ResidentOnly := resident;
  (* Set config *)
  Dim := Adim;
  NHeads := AnHeads;
  HeadDim := Dim div NHeads;
  NLayers := AnLayers;
  NRefiner := AnRefiner;
  FfnDim := (8 * Dim div 3 + 255) div 256 * 256; (* Round up to 256 *)
  InChannels := AinChannels;
  PatchSize := ApatchSize;
  if Dim < 256 then
    AdalnDim := Dim
  else
    AdalnDim := 256;
  RoPETheta := AropeTheta;
  CapFeatDim := AcapFeatDim;

  for I := 0 to 3 - 1 do
  begin
    AxesDims[I] := AAxesDims[I];
    AxesLens[I] := 1024; (* Default max positions *)
  end;

  (* Open safetensors files *)

  SfFiles := LoadShards(ModelDir);

  NumSfFiles := Length(SfFiles);

  (* Determine FFN dimension from weights *)
  Sf := SfFiles.GetTensor('layers.0.feed_forward.w1.weight');
  if Assigned(Sf) then
    FfnDim := Sf.Shape[0];

  (* Determine t_embedder mid_size from weights *)
  TEmbMidSize := 1024; (* Default *)
  Sf := SfFiles.GetTensor('t_embedder.mlp.0.weight');
  if Assigned(Sf) then
    TEmbMidSize := Sf.Shape[0];

  (* BLAS/CPU fast-load mode: keep mmap files open and use direct f32 pointers. *)
  MmapF32Weights := AUseMMAP;

  (* Load timestep embedder *)
  TEmbMlp0Weight := Projection('t_embedder.mlp.0.weight', AUseMMAP);
  TEmbMlp0Bias := SfFiles.GetTensorDataMemBlock('t_embedder.mlp.0.bias', AUseMMAP);
  TEmbMlp2Weight := Projection('t_embedder.mlp.2.weight', AUseMMAP);
  TEmbMlp2Bias := SfFiles.GetTensorDataMemBlock('t_embedder.mlp.2.bias', AUseMMAP);

  (* Load caption embedder: RMSNorm + Linear *)
  CapEmbNorm := SfFiles.GetTensorDataMemBlock('cap_embedder.0.weight', AUseMMAP);
  CapEmbLinearW := Projection('cap_embedder.1.weight', AUseMMAP);
  CapEmbLinearB := SfFiles.GetTensorDataMemBlock('cap_embedder.1.bias', AUseMMAP);

  (* Load image embedder *)
  XEmbWeight := Projection(Format('all_x_embedder.%d-1.weight', [PatchSize]), AUseMMAP);
  XEmbBias := SfFiles.GetTensorDataMemBlock(Format('all_x_embedder.%d-1.bias', [PatchSize]), False);

  (* Pad tokens *)
  XPadToken := SfFiles.GetTensorDataMemBlock('x_pad_token', AUseMMAP);
  CapPadToken := SfFiles.GetTensorDataMemBlock('cap_pad_token', AUseMMAP);

  (* Load noise refiner blocks *)
  SetLength(NoiseRefiner, NRefiner);

  for I := 0 to High(NoiseRefiner) do
    NoiseRefiner[I].Load(SfFiles, Format('noise_refiner.%d', [I]), True, AUseMMAP, resident);

  (* Load context refiner blocks (no modulation) *)
  SetLength(ContextRefiner, NRefiner);

  for I := 0 to NRefiner - 1 do
    ContextRefiner[I].Load(SfFiles, Format('context_refiner.%d', [I]), False, AUseMMAP, resident);

  (* Load main transformer blocks *)
  SetLength(Layers, NLayers);

  for I := 0 to NLayers - 1 do
    Layers[I].Load(SfFiles, Format('layers.%d', [I]), True, AUseMMAP, resident);

  (* Load final layer *)
  FinalLayer.AdalnWeight := Projection(Format('all_final_layer.%d-1.adaLN_modulation.1.weight', [PatchSize]), AUseMMAP);
  FinalLayer.AdalnBias :=
    SfFiles.GetTensorDataMemBlock(Format('all_final_layer.%d-1.adaLN_modulation.1.bias', [PatchSize]), AUseMMAP);
  // final_layer.norm_weight := sf_files.getTensorDataMemBlock(format('all_final_layer.%d-1.norm_final.weight', [patch_size]), useMMAP);
  FinalLayer.LinearWeight := Projection(Format('all_final_layer.%d-1.linear.weight', [PatchSize]), AUseMMAP);
  FinalLayer.LinearBias :=
    SfFiles.GetTensorDataMemBlock(Format('all_final_layer.%d-1.linear.bias', [PatchSize]), AUseMMAP);

  (* Precompute RoPE tables *)
  if not resident then
    PreComputeRope();

  (* Allocate initial working memory (will be resized as needed) *)
  WorkAlloc := 0;
  // work_x := nil;
  // work_tmp := nil;
  // work_qkv := nil;
  // work_attn := nil;
  // work_ffn := nil;
  MaxSeq := 0;
end;

function TTransformerZI.IsLoaded(): Boolean;
begin
  Result := Assigned(SfFiles)
end;

procedure TTransformerZI.Free;
var
  I: LongInt;
begin
  QNNGPUClearWeightCache;
  // if not mmap_f32_weights then begin
  TEmbMlp0Weight.Free;
  TEmbMlp0Bias.Free;
  TEmbMlp2Weight.Free;
  TEmbMlp2Bias.Free;
  CapEmbNorm.Free;
  CapEmbLinearW.Free;
  CapEmbLinearB.Free;
  XEmbWeight.Free;
  XEmbBias.Free;
  XPadToken.Free;
  CapPadToken.Free;
  // end;

  if Assigned(NoiseRefiner) then
  begin
    for I := 0 to High(NoiseRefiner) do
      NoiseRefiner[I].Free();
    SetLength(NoiseRefiner, 0);
  end;
  if Assigned(ContextRefiner) then
  begin
    for I := 0 to High(ContextRefiner) do
      ContextRefiner[I].Free;
    SetLength(ContextRefiner, 0);
  end;
  if Assigned(Layers) then
  begin
    for I := 0 to High(Layers) do
      Layers[I].Free;
    SetLength(Layers, 0);
  end;

  FinalLayer.AdalnWeight.Free;
  FinalLayer.AdalnBias.Free;
  // final_layer.norm_weight.free;
  FinalLayer.LinearWeight.Free;
  FinalLayer.LinearBias.Free;

  for I := 0 to High(SfFiles) do
    SfFiles[I].Close();
  NumSfFiles := 0;

  for I := 0 to 3 - 1 do
  begin
    RoPECos[I].Free;
    RoPESin[I].Free;
  end;

  WorkX.Free;
  WorkTmp.Free;
  WorkQKV.Free;
  WorkAttn.Free;
  WorkFfn.Free;
end;

end.

