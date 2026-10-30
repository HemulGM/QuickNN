unit QuickNN.CPU;

{$IFDEF CPUX64}
// {$EXCESSPRECISION OFF}
{$ENDIF}
{$C+} // enable assertions
{$H+} // longstrings
{$M+} // include typeinfo
{$POINTERMATH on} // manipulate, inc, dec, cast pointers
{$T+} // typed pointer when @ is used
{$R+} // raise an error when trying to access arrays out of their bounds
{$Z4} // aligh enums to 4 bytes (compatibility with external routines)

// {$define USE_MULTITHREADING}

interface
uses
  System.SysUtils,
  System.Classes,
  System.Math,
  System.TypInfo
{$IF defined(MSWINDOWS)}
  ,
  Winapi.Windows
{$ELSEIF defined(DARWIN)}
{$ELSEIF defined(UNIX)}
  ,
  Posix.Dlfcn
{$ENDIF}
{$IFDEF USE_MULTITHREADING}
  ,
  QuickNN.Threading
{$ENDIF}
  ,
  QuickNN.Common;

// {$if not declared(CBLAS_LAYOUT)}
// type
//  CBLAS_Layout = (CblasRowMajor = 101, CblasColMajor = 102);
// {$else}
// const
//  CblasRowMajor = CBLAS_Layout.CblasRowMajor;
//  CblasColMajor = CBLAS_Layout.CblasColMajor;
// {$endif}

// {$if not declared(CBLAS_ORDER)}
// type
//  CBLAS_ORDER = CBLAS_Layout;
// {$endif}
// {$if not declared(CBLAS_TRANSPOSE)}
// type
//  CBLAS_TRANSPOSE = (CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans =
//    113, CblasConjNoTrans = 114);
// {$else}
// const
// CblasNoTrans = CBLAS_TRANSPOSE.CblasNoTrans;
// CblasTrans = CBLAS_TRANSPOSE.CblasTrans;
// CblasConjTrans = CBLAS_TRANSPOSE.CblasConjTrans;
// CblasConjNoTrans = CBLAS_TRANSPOSE.CblasConjNoTrans ;
// {$endif}

(* Global callback pointers - set by caller before inference *)
// Add and Mul terms used for vector vector element wise operations
// Scale and Bias terms used for vector Scalar element wise operations
type

  { TQNNSingleOPS }

  TQNNSingleOps = class
  private
    class function CubicInterpolation(const B, C, T: Single): Single; overload; static;
    class function LinearInterpolation(const A, B, T: Single): Single; static;
    class function Qasum(N: Int64; X: System.PSingle; IncX: Int64): Single; winapi; static;
    class procedure Qaxpy(N: Int64; Alpha: Single; X: System.PSingle; Y: System.PSingle); static;
    class procedure QaxpyStrided(
        N: Int64;
        Alpha: Single;
        X: System.PSingle;
        IncX: Int64;
        Y: System.PSingle;
        Incy: Int64
    ); winapi; static;
    class function Qdot(N: Int64; X: System.PSingle; Y: System.PSingle): Single; static;
    class function QdotStrided(
        N: Int64;
        X: System.PSingle;
        IncX: Int64;
        Y: System.PSingle;
        Incy: Int64
    ): Single; winapi; static;
    class procedure Qgemm(
        Order: TCBLASOrder;
        TransA: TCBLASTranspose;
        TransB: TCBLASTranspose;
        M: Int64;
        N: Int64;
        K: Int64;
        Alpha: Single;
        A: System.PSingle;
        Lda: Int64;
        B: System.PSingle;
        Ldb: Int64;
        Beta: Single;
        C: System.PSingle;
        Ldc: Int64
    ); winapi; static;
    class procedure Qscale(N: Int64; Alpha: Single; X: System.PSingle; IncX: Int64); winapi; static;
  public
    class var
      CblasSgemm:
          procedure(
              Order: TCBLASOrder;
              TransA: TCBLASTranspose;
              TransB: TCBLASTranspose;
              M: Int64;
              N: Int64;
              K: Int64;
              Alpha: Single;
              A: System.PSingle;
              Lda: Int64;
              B: System.PSingle;
              Ldb: Int64;
              Beta: Single;
              C: System.PSingle;
              Ldc: Int64
          ); winapi;
      CblasSaxpy:
          procedure(N: Int64; Alpha: Single; X: System.PSingle; IncX: Int64; Y: System.PSingle; Incy: Int64); winapi;
      CblasSdot: function(N: Int64; X: System.PSingle; IncX: Int64; Y: System.PSingle; Incy: Int64): Single; winapi;
      CblasSasum: function(N: Int64; X: System.PSingle; IncX: Int64): Single; winapi;
      CblasSscal: procedure(N: Int64; Alpha: Single; X: System.PSingle; IncX: Int64 = 1); winapi;
      CblasSbgemm:
          procedure(
              Order: TCBLASOrder;
              TransA: TCBLASTranspose;
              TransB: TCBLASTranspose;
              M: Int64;
              N: Int64;
              K: Int64;
              Alpha: Single;
              A: PBF16;
              Lda: Int64;
              B: PBF16;
              Ldb: Int64;
              Beta: Single;
              C: System.PSingle;
              Ldc: Int64
          ); winapi;
      OpenblasSetNumThreads: procedure(NumThreads: LongInt); winapi;
      OpenblasGetNumThreads: function(): LongInt; winapi;
      OpenblasGetNumProcs: function(): LongInt; winapi;
      OpenblasGetConfig: function(): PAnsiChar;
      OpenblasGetCorename: function(): PAnsiChar;

      IsUsingBlas: Boolean;
      Workspace: TArray<Single>;
  public
    class procedure CblasGemm(
        Order: TCBLASOrder;
        TransA: TCBLASTranspose;
        TransB: TCBLASTranspose;
        M: Int64;
        N: Int64;
        K: Int64;
        Alpha: Single;
        A: System.PSingle;
        Lda: Int64;
        B: System.PSingle;
        Ldb: Int64;
        Beta: Single;
        C: System.PSingle;
        Ldc: Int64
    );
    class procedure CblasCPUGemm(
        Order: TCBLASOrder;
        TransA: TCBLASTranspose;
        TransB: TCBLASTranspose;
        M: Int64;
        N: Int64;
        K: Int64;
        Alpha: Single;
        A: System.PSingle;
        Lda: Int64;
        B: System.PSingle;
        Ldb: Int64;
        Beta: Single;
        C: System.PSingle;
        Ldc: Int64
    );
    class procedure CblasAxpy(N: Int64; Alpha: Single; X: System.PSingle; IncX: Int64; Y: System.PSingle; Incy: Int64);
    class procedure CblasScal(N: Int64; Alpha: Single; X: System.PSingle; IncX: Int64 = 1);
    class function CblasDot(N: Int64; X: System.PSingle; IncX: Int64; Y: System.PSingle; Incy: Int64): Single;
    class function CblasAsum(N: Int64; X: System.PSingle; IncX: Int64): Single;

    class procedure PrintCompare(
        const N: LongInt;
        const Src1, Src2: System.PSingle;
        const IsSumSqrDiff: Boolean = False
    );

    class procedure PrintStat(const Src: System.PSingle; const N: LongInt);
    class procedure QNNAdd(const Dst, A, B: System.PSingle; const N: Integer); overload;
    class procedure QNNAddInplace(const Dst, A: System.PSingle; const N: Integer); overload;
    class procedure QNNMul(const Dst, A, B: System.PSingle; const N: Integer); overload;
    class procedure QNNMulInplace(const Dst, A: System.PSingle; const N: Integer); overload;
    class procedure QNNScale(const Dst, Src: System.PSingle; const AScale: Single; const N: LongInt); overload;
    class procedure QNNScaleInplace(const Dst: System.PSingle; const AScale: Single; const N: LongInt); overload;
    class procedure QNNBias(const Dst, Src: System.PSingle; const ABias: Single; const N: LongInt); overload;
    class procedure QNNBiasInplace(const Dst: System.PSingle; const ABias: Single; const N: LongInt); overload;

    class procedure QNNFusedMulAdd(const Dst, Src, SrcM, SrcA: System.PSingle; const N: LongInt); overload;
    class procedure QNNFusedAddMul(const Dst, Src, SrcA, SrcM: System.PSingle; const N: LongInt); overload;
    class procedure QNNFusedScaleBias(
        const Dst, Src: System.PSingle;
        const AScale, ABias: Single;
        const N: LongInt
    ); overload;
    class procedure QNNFusedBiasScale(
        const Dst, Src: System.PSingle;
        const ABias, AScale: Single;
        const N: LongInt
    ); overload;
    class procedure QNNFusedMulScale(
        const Dst, Src1, Src2: System.PSingle;
        const AScale: Single;
        const N: LongInt
    ); overload;
    class procedure QNNFusedBiasMulScale(
        const Dst, Src, SrcM: System.PSingle;
        const ABias, AScale: Single;
        const N: LongInt
    ); overload;
    class procedure QNNFusedScaleAdd(
        const Dst, Src, SrcAdd: System.PSingle;
        const AScale: Single;
        const N: LongInt
    ); overload;
    class procedure QNNFusedBiasAdd(
        const Dst, Src, SrcA: System.PSingle;
        const ABias: Single;
        const N: LongInt
    ); overload;
    class function QNNDot(const A, B: System.PSingle; const N: LongInt): Single;

    class procedure QNNAccAdd(const Dst, A, B: System.PSingle; const N: Integer); overload;
    class procedure QNNAccMul(const Dst, A, B: System.PSingle; const N: Integer); overload;

    class procedure QNNMatMulNN(const C, A, B: TMemoryBlock; const M, K, N: Integer); overload;
    class procedure QNNMatMulNT(const C, A, B: TMemoryBlock; const M, K, N: Integer); overload;
    class procedure QNNMatMulTN(const C, A, B: TMemoryBlock; const M, K, N: Integer); overload;

    class procedure QNNMatTranspose(const Dst, Src: System.PSingle; const SrcRows, SrcCols: LongInt);
    class procedure QNNLinear(const Dst, X, W, B: System.PSingle; const SeqLen, InDIM, OutDIM: Integer);
    class procedure QNNLinearNoBias(const Dst, X, W: System.PSingle; const SeqLen, InDIM, OutDIM: Integer);
    class procedure QNNLinearNoBiasBF16(
        const Dst, X: System.PSingle;
        const W: PBF16;
        const SeqLen, InDIM, OutDIM: Integer
    );
    class procedure QNNIm2Col(
        const AChannels,
        AHeight,
        AWidth,
        KernelHeight,
        KernelWidth,
        PadHeight,
        PadWidth,
        StrideY,
        StrideX,
        DilationY,
        DilationX: Integer;
        const Im: System.PSingle;
        const Col: System.PSingle;
        const ImOffset: Integer = 0;
        const ColOffset: Integer = 0;
        const MultiThread: Boolean = False
    );

    class procedure QNNIm2ColStridedBatched(
        const AChannels,
        AHeight,
        AWidth,
        KernelHeight,
        KernelWidth,
        PadHeight,
        PadWidth,
        StrideY,
        StrideX,
        DilationY,
        DilationX: Integer;
        const Im: System.PSingle;
        const ImStride, ImOffset: Integer;
        const Col: System.PSingle;
        const ColStride, ColOffset: Integer;
        const Batch: Integer
    );

    class procedure QNNConv2d(
        const Dst, Src, Weights, Bias: System.PSingle;
        const InCh, OutCh, H, W, KH, KW, Stride, Padding: LongInt;
        const Batch: LongInt = 1
    );

    class procedure QNNNorm(
        const N: LongInt;
        const Dst, Src: System.PSingle;
        const Weights: System.PSingle = nil;
        const Stride: LongInt = 1
    ); overload;

    class procedure QNNRMSNorm(
        const N: LongInt;
        const Dst, Src: System.PSingle;
        const Weights: System.PSingle = nil;
        const Stride: LongInt = 1
    ); overload;
    class procedure QNNRMSNormRows(const Dst, Src, Weight: System.PSingle; const Rows, Dim: Integer); overload;
    class procedure QNNTanh(const Dst: System.PSingle; const N: LongInt);

    class procedure QNNQKRMSNorm(
        const Q, K, QWeights, KWeights: System.PSingle;
        const Seq, Heads, HeadDim: LongInt
    ); overload;
    class procedure QNNRMSNormSeq(const Dst, NormWeights: System.PSingle; const Seq, Heads, HeadDim: LongInt); overload;

    class procedure QNNGroupNorm(
        const Dst, Src, Gamma, Beta: System.PSingle;
        const Batch, Channels, H, W, NumGroups: Integer
    );
    class procedure QNNBatchNorm(
        const Dst, Src, RunningMean, RunningVar, Gamma, Beta: System.PSingle;
        const Batch, Channels, H, W: Integer
    );

    class procedure QNNMinMax(
        const N: Integer;
        const Src: System.PSingle;
        out OutMin, OutMax: Single;
        const ArgMin: PInteger;
        const ArgMax: PInteger = nil;
        const Stride: Integer = 1
    );
    class function QNNMax(
        const N: Integer;
        const Src: System.PSingle;
        const Arg: PInteger = nil;
        const Stride: Integer = 1
    ): Single;
    class function QNNMin(
        const N: Integer;
        const Src: System.PSingle;
        const Arg: PInteger = nil;
        const Stride: Integer = 1
    ): Single;
    class function QNNMaxAbs(const N: Integer; const Src: System.PSingle; const Stride: Integer = 1): Single;
    class function QNNMinAbs(const N: Integer; const Src: System.PSingle; const Stride: Integer = 1): Single;
    class function QNNArgMax(const N: Integer; const Src: System.PSingle; const Stride: Integer = 1): Integer;
    class function QNNArgMin(const N: Integer; const Src: System.PSingle; const Stride: Integer = 1): Integer;
    class function QNNSum(const N: Integer; const Src: System.PSingle; const Stride: Integer = 1): Single;
    class function QNNSumSqr(const N: Integer; const Src: System.PSingle; const Stride: Integer = 1): Single;
    class function QNNSumSqrDiff(
        const N: Integer;
        const Src: System.PSingle;
        const AMean: Single;
        const Stride: Integer = 1
    ): Single; overload;
    class function QNNSqrDistance(
        const N: Integer;
        const Src1, Src2: System.PSingle;
        const Stride: Integer = 1
    ): Single; overload;
    class function QNNMaxAbsDiff2(
        const N: Integer;
        const Src1, Src2: System.PSingle;
        out OutSrc1, OutSrc2: Single;
        const Stride: Integer = 1
    ): Single; overload;
    class function QNNMaxAbsDiff(
        const N: Integer;
        const Src1, Src2: System.PSingle;
        const Stride: Integer = 1
    ): Single; overload;
    class function QNNSumAbsDiff(
        const N: Integer;
        const Src1, Src2: System.PSingle;
        const Stride: Integer = 1
    ): Single; overload;
    class function QNNSumAbsDiffScalar(
        const N: Integer;
        const Src: System.PSingle;
        const AMean: Single;
        const Stride: Integer = 1
    ): Single; overload;
    class function QNNMean(const N: Integer; const Src: System.PSingle; const Stride: Integer = 1): Single;
    class function QNNVariance(
        const N: Integer;
        const Src: System.PSingle;
        const AMean: Single;
        const Stride: Integer = 1;
        const IsPopulation: Boolean = True
    ): Single;
    class procedure QNNSigmoidInplace(const X: System.PSingle; const N: Integer); overload;
    class procedure QNNSigmoid(const Dst, Src: System.PSingle; const N: Integer); overload;
    class procedure QNNSiluInplace(const X: System.PSingle; const N: Integer); overload;
    class procedure QNNSilu(const Dst, Src: System.PSingle; const N: Integer); overload;
    class procedure QNNSiluMul(const Gate, Up: System.PSingle; const N: Integer);
    class procedure QNNSoftmax(const X: System.PSingle; const N: Integer); overload;
    class procedure QNNSoftmaxRows(const X: System.PSingle; const Rows, Cols: Integer); overload;
    class procedure QNNAttention(
        const Dst, Q, K, V: System.PSingle;
        const Batch, Heads, SeqQ, SeqK, HeadDim: LongInt;
        const Scale: Single
    );
    class procedure QNNFlashAttentionHead(
        const Dst, Q, K, V: System.PSingle;
        const SeqQ, SeqK, HeadDim: LongInt;
        const Scale: Single
    );
    class procedure QNNFlashAttentionHeadTiled(
        const Dst, Q, K, V: System.PSingle;
        const SeqQ, SeqK, HeadDim: LongInt;
        const Scale: Single;
        const TileScores: System.PSingle;
        const QTileSize, KTileSize: LongInt
    );
    class procedure QNNFlashAttention(
        const Dst, Q, K, V: System.PSingle;
        const Heads, SeqQ, SeqK, HeadDim: LongInt;
        const Scale: Single
    );
    class procedure QNNUpSampleNearest(
        const Dst, Src: System.PSingle;
        const Batch, Channels, H, W, ScaleH, ScaleW: LongInt
    );
    class procedure QNNUpSample(
        const Dst, Src: System.PSingle;
        const Batch, Channels, H, W: LongInt;
        const ScaleH, ScaleW: Single;
        const Interpolation: TInterpolation = INearest
    );
    class procedure QNNPatchify(const Dst, Src: System.PSingle; const Batch, Channels, H, W, PatchSize: LongInt);
    class procedure QNNUnpatchify(const Dst, Src: System.PSingle; const Batch, Channels, H, W, PatchSize: LongInt);
    class procedure QNNPermut(
        const Dst, Src: System.PSingle;
        const AxisPerm: TArray<Byte>;
        const InShape: TArray<Int64>;
        var OutShape: TArray<Int64>
    );
    class procedure QNNCopy(const Dst, Src: System.PSingle; const N: Integer); overload;
    class procedure QNNCopyStrided(
        const Dst: System.PSingle;
        const DstStride: Integer;
        const Src: System.PSingle;
        const SrcStride: Integer;
        const N: Integer
    ); overload;
    class procedure QNNFill(const Dst: System.PSingle; const Val: Single; const N: Integer; const Stride: Integer = 1);
    class procedure QNNBroadcast(const Dst: Pointer; const Src; const SrcSize, N: Integer; const Stride: Integer = 1);
    class procedure QNNGatedAdd(const Dst, Gate, Proj: System.PSingle; const Seq, Hidden: Integer);
    class procedure QNNComputeRoPE(
        const Dst: System.PSingle;
        const MaxSeq, Dim: LongInt;
        const Theta: Single
    ); overload;
    class procedure QNNComputeRoPE(
        const CosOut, SinOut: System.PSingle;
        const MaxSeqLen, HeadDim: LongInt;
        const Theta: Single
    ); overload;
    class procedure QNNComputeRoPE2D(
        const CosDst, SinDst: System.PSingle;
        const PatchH, PatchW, Dim: LongInt;
        const Theta: Single
    );
    class procedure QNNComputeRoPE2DOffset(
        const CosDst, SinDst: System.PSingle;
        const PatchH, PatchW, Dim: LongInt;
        const Theta: Single;
        const OffsetT: LongInt
    );
    class procedure QNNComputeRoPEText(
        const CosDst, SinDst: System.PSingle;
        const TxtSeq, AxisDim: LongInt;
        const Theta: Single
    );
    class procedure QNNApplyRoPE(const Dst, Freqs: System.PSingle; const Batch, Seq, Heads, HeadDim: LongInt); overload;
    class procedure QNNApplyRoPE2(const Dst, CosIn, SinIn: System.PSingle; const NumHeads, HeadDim: LongInt); overload;
    class procedure QNNApplyRoPE3(
        const Dst, CosIn, SinIn: System.PSingle;
        const SeqLen, NumHeads, HeadDim: LongInt
    ); overload;
    class procedure QNNApplyRoPEQK(
        const Q, K, CosCache, SinCache: System.PSingle;
        const SeqLen, NumQHeads, NumKvHeads, HeadDim: LongInt
    ); overload;

    class procedure QNNApplyRoPE2D(
        const Dst, CosFreq, SinFreq: System.PSingle;
        const Seq, Heads, HeadDim, Dim: LongInt
    );
    // Positional Embedding
    class function QNNTimeStepEmbedding(const T: Single; const Dim: Integer; const MaxPeriod: Single): TMemoryBlock;
    class procedure QNNMatTriangularFill(
        const Dim: Integer;
        const Dst: System.PSingle;
        const Val: Single;
        const Mask: PLongint = nil
    );
    class procedure QNNMaskFill(
        const Dst: System.PSingle;
        const Mask: PLongint;
        const Val: Single;
        const Rows, Cols: LongInt
    );
    // Adaptive Layer Normalization
    class procedure QNNAdaLN(
        const Dst, Src, Bias, Scale: System.PSingle;
        const Seq, Hidden: LongInt;
        const Eps: Single = EPSILON
    );

    class procedure QNNLinearBF16OrF32(
        const Dst, Src: System.PSingle;
        const Weight: TMemoryBlock;
        const WeightBF16: TMemoryBlock;
        const Seq, SrcDim, DstDim: Integer
    ); inline;
  end;

{$IF defined(CPUX64) and defined(USE_AVX2)}
procedure SingleToBF16X8AVX2(const Src: System.PSingle; const Dst: PBF16);
procedure BF16ToSingleX8AVX2(const Src: PBF16; const Dst: System.PSingle);
{$ENDIF}

procedure PrintStat(const Src: TMemoryBlock); overload;
// procedure writeTensor(const buf:TMemoryBlock);
// function readTensor():TMemoryBlock;
// function readArray():TArray<integer>;
procedure CompareArray(const A, B: TArray<Integer>); overload;
procedure CompareArray(const A, B: PInteger; const N: LongInt); overload;
procedure CompareArray(const A, B: System.PSingle; const N: LongInt); overload;
procedure PrintArray(const Src: System.PSingle; const N: LongInt); overload;
procedure PrintArray(const Src: PLongint; const N: LongInt); overload;
procedure PrintArray(const Src: PInt64; const N: LongInt); overload;
procedure RandArray(const N: LongInt; const Dst: System.PSingle; const AMin: Single = -5.0; AMax: Single = 5.0);
{$IF defined(MACOS) or defined(DARWIN) or defined(USE_STATIC_LINK)}

procedure CblasSaxpy(
    N: Int64;
    Alpha: Single;
    X: System.PSingle;
    IncX: Int64;
    Y: System.PSingle;
    Incy: Int64
); winapi; external name 'cblas_saxpy';
function CblasSdot(
    N: Int64;
    X: System.PSingle;
    IncX: Int64;
    Y: System.PSingle;
    Incy: Int64
): Single; winapi; external name 'cblas_sdot';
function CblasSasum(N: Int64; X: System.PSingle; IncX: Int64): Single; winapi; external name 'cblas_sasum';
procedure CblasSscal(N: Int64; Alpha: Single; X: System.PSingle; IncX: Int64); winapi; external name 'cblas_sscal';
procedure CblasSgemm(
    Order: TCBLASOrder;
    TransA: TCBLASTranspose;
    TransB: TCBLASTranspose;
    M: Int64;
    N: Int64;
    K: Int64;
    Alpha: Single;
    A: System.PSingle;
    Lda: Int64;
    B: System.PSingle;
    Ldb: Int64;
    Beta: Single;
    C: System.PSingle;
    Ldc: Int64
); winapi; external name 'cblas_sgemm';

procedure CblasDaxpy(
    N: Int64;
    Alpha: Double;
    X: System.PDouble;
    IncX: Int64;
    Y: System.PDouble;
    Incy: Int64
); winapi; external name 'cblas_daxpy';
function CblasDdot(
    N: Int64;
    X: System.PDouble;
    IncX: Int64;
    Y: System.PDouble;
    Incy: Int64
): Double; winapi; external name 'cblas_ddot';
function CblasDasum(N: Int64; X: System.PDouble; IncX: Int64): Double; winapi; external name 'cblas_dasum';
procedure CblasDscal(N: Int64; Alpha: Double; X: System.PDouble; IncX: Int64); winapi; external name 'cblas_dscal';
procedure CblasDgemm(
    Order: TCBLASOrder;
    TransA: TCBLASTranspose;
    TransB: TCBLASTranspose;
    M: Int64;
    N: Int64;
    K: Int64;
    Alpha: Double;
    A: System.PDouble;
    Lda: Int64;
    B: System.PDouble;
    Ldb: Int64;
    Beta: Double;
    C: System.PDouble;
    Ldc: Int64
); winapi; external name 'cblas_dgemm';
{$ENDIF}

procedure AltPermut(const Dst, Src: System.PSingle; const InShape: TArray<Int64>; const AxisPerm: TArray<Byte>);

var
  HBLASLIB: HMODULE;

implementation
uses
  QuickNN.Vulkan;

// var vk : TQNNVulkan;

function Sqr(const X: Single): Single; inline;
begin
  Exit(X * X)
end;

// {$macro on}
// {$define fast_exp:=exp}

function FastExp(X: Single): Single; inline;
var
  N, R, P: Single;
  V: record
  case Boolean of
    False: (I: LongInt);
    True: (F: Single)
  end;
const
  MAX_EXP = Single(8.872283E+001);
  MIN_EXP = Single(-8.733654E+001);

begin
  // result := exp(x)
  if (X < -87.33) then
    Exit(0.0);
  if (X > 88.72) then
    Exit(1e38);
  // if (x < MIN_EXP) then  x := MIN_EXP;
  // if (x > MAX_EXP)  then  x := MAX_EXP;
  N := Floor(X * 1.4426950408889634 + 0.5);
  R := X - N * 0.6931471805599453;
  P := 1.0 + R * (1.0 + R * (0.5 + R * (0.16666667 + R * (0.04166667 + R * 0.00833333))));
  V.F := P;
  Inc(V.I, Trunc(N) shl 23);
  Result := V.F;
end;

{$i QuickNN.CPU.AVX2.inc}

class procedure TQNNSingleOps.CblasGemm(
    Order: TCBLASOrder;
    TransA, TransB: TCBLASTranspose;
    M, N, K: Int64;
    Alpha: Single;
    A: System.PSingle;
    Lda: Int64;
    B: System.PSingle;
    Ldb: Int64;
    Beta: Single;
    C: System.PSingle;
    Ldc: Int64
);
var
  Row, Col: Int64;
  Value: Single;
begin
  if (M < 0) or (N < 0) or (K < 0) then
    raise ERangeError.Create('Negative GEMM dimensions');
  if (M = 0) or (N = 0) then
    Exit;
  if TransA = CblasConjTrans then
    TransA := CblasTrans;
  if TransA = CblasConjNoTrans then
    TransA := CblasNoTrans;
  if TransB = CblasConjTrans then
    TransB := CblasTrans;
  if TransB = CblasConjNoTrans then
    TransB := CblasNoTrans;
  if Order = CblasColMajor then
  begin
    CblasGemm(CblasRowMajor, TransB, TransA, N, M, K, Alpha, B, Ldb, A, Lda, Beta, C, Ldc);
    Exit;
  end;
  if (C = nil) or (Ldc < N) then
    raise EArgumentException.Create('Invalid GEMM output');
  if (K = 0) or (Alpha = 0) then
  begin
    // BLAS requires these cases to ignore A/B, and beta=0 to ignore old C.
    if Beta = 1 then
      Exit;
    for Row := 0 to M - 1 do
      if Beta = 0 then
        FillChar(C[Row * Ldc], N * SizeOf(Single), 0)
      else
        for Col := 0 to N - 1 do
          C[Row * Ldc + Col] := Beta * C[Row * Ldc + Col];
    Exit;
  end;
  if TryVulkanGemm(Order, TransA, TransB, M, N, K, Alpha, A, Lda, B, Ldb, Beta, C, Ldc) then
    Exit;
  if (TransA = CblasTrans) and (TransB = CblasTrans) and not IsUsingBlas then
  begin
    // The original Pascal fallback supports NN/NT/TN only.
    for Row := 0 to M - 1 do
      for Col := 0 to N - 1 do
      begin
        Value := Alpha * QdotStrided(K, A + Row, Lda, B + Col * Ldb, 1);
        if Beta = 0 then
          C[Row * Ldc + Col] := Value
        else
          C[Row * Ldc + Col] := Value + Beta * C[Row * Ldc + Col];
      end;
    Exit;
  end;
  CblasSgemm(Order, TransA, TransB, M, N, K, Alpha, A, Lda, B, Ldb, Beta, C, Ldc);
end;

class procedure TQNNSingleOps.CblasCPUGemm(
    Order: TCBLASOrder;
    TransA, TransB: TCBLASTranspose;
    M, N, K: Int64;
    Alpha: Single;
    A: System.PSingle;
    Lda: Int64;
    B: System.PSingle;
    Ldb: Int64;
    Beta: Single;
    C: System.PSingle;
    Ldc: Int64
);
var
  Row, Col: Int64;
  Value: Single;
begin
  if (M < 0) or (N < 0) or (K < 0) then
    raise ERangeError.Create('Negative GEMM dimensions');
  if (M = 0) or (N = 0) then
    Exit;
  if TransA = CblasConjTrans then
    TransA := CblasTrans;
  if TransA = CblasConjNoTrans then
    TransA := CblasNoTrans;
  if TransB = CblasConjTrans then
    TransB := CblasTrans;
  if TransB = CblasConjNoTrans then
    TransB := CblasNoTrans;
  if Order = CblasColMajor then
  begin
    CblasCPUGemm(CblasRowMajor, TransB, TransA, N, M, K, Alpha, B, Ldb, A, Lda, Beta, C, Ldc);
    Exit;
  end;
  if (C = nil) or (Ldc < N) then
    raise EArgumentException.Create('Invalid GEMM output');
  if (K = 0) or (Alpha = 0) then
  begin
    // BLAS requires these cases to ignore A/B, and beta=0 to ignore old C.
    if Beta = 1 then
      Exit;
    for Row := 0 to M - 1 do
      if Beta = 0 then
        FillChar(C[Row * Ldc], N * SizeOf(Single), 0)
      else
        for Col := 0 to N - 1 do
          C[Row * Ldc + Col] := Beta * C[Row * Ldc + Col];
    Exit;
  end;
  if (TransA = CblasTrans) and (TransB = CblasTrans) and not IsUsingBlas then
  begin
    // The original Pascal fallback supports NN/NT/TN only.
    for Row := 0 to M - 1 do
      for Col := 0 to N - 1 do
      begin
        Value := Alpha * QdotStrided(K, A + Row, Lda, B + Col * Ldb, 1);
        if Beta = 0 then
          C[Row * Ldc + Col] := Value
        else
          C[Row * Ldc + Col] := Value + Beta * C[Row * Ldc + Col];
      end;
    Exit;
  end;
  CblasSgemm(Order, TransA, TransB, M, N, K, Alpha, A, Lda, B, Ldb, Beta, C, Ldc);
end;

class procedure TQNNSingleOps.CblasAxpy(
    N: Int64;
    Alpha: Single;
    X: System.PSingle;
    IncX: Int64;
    Y: System.PSingle;
    Incy: Int64
);
begin
  CblasSaxpy(N, Alpha, X, IncX, Y, Incy)
end;

class procedure TQNNSingleOps.CblasScal(N: Int64; Alpha: Single; X: System.PSingle; IncX: Int64);
begin
  CblasSscal(N, Alpha, X, IncX)
end;

class function TQNNSingleOps.CblasDot(N: Int64; X: System.PSingle; IncX: Int64; Y: System.PSingle; Incy: Int64): Single;
begin
  Result := CblasSdot(N, X, IncX, Y, Incy)
end;

class function TQNNSingleOps.CblasAsum(N: Int64; X: System.PSingle; IncX: Int64): Single;
begin
  Result := CblasSasum(N, X, IncX)
end;

class procedure TQNNSingleOps.QNNMinMax(
    const N: Integer;
    const Src: System.PSingle;
    out OutMin, OutMax: Single;
    const ArgMin: PInteger;
    const ArgMax: PInteger;
    const Stride: Integer
);
var
  I: Integer;
begin
  Assert(Assigned(Src), 'ERROR QNNMax: [src] is empty!');
  OutMin := Src[0];
  OutMax := Src[0];
  if Assigned(ArgMax) then
    ArgMax^ := 0;
  if Assigned(ArgMin) then
    ArgMin^ := 0;
  if Stride = 1 then
  begin
    for I := 1 to N - 1 do
    begin
      if Src[I] > OutMax then
      begin
        OutMax := Src[I];
        if Assigned(ArgMax) then
          ArgMax^ := I
      end;
      if Src[I] < OutMin then
      begin
        OutMin := Src[I];
        if Assigned(ArgMin) then
          ArgMin^ := I
      end;
    end
  end
  else
    for I := 1 to N - 1 do
    begin
      if Src[I * Stride] > OutMax then
      begin
        OutMax := Src[I * Stride];
        if Assigned(ArgMax) then
          ArgMax^ := I * Stride
      end;
      if Src[I * Stride] < OutMin then
      begin
        OutMin := Src[I * Stride];
        if Assigned(ArgMin) then
          ArgMin^ := I * Stride
      end
    end
end;

class function TQNNSingleOps.QNNMax(
    const N: Integer;
    const Src: System.PSingle;
    const Arg: PInteger;
    const Stride: Integer
): Single;
var
  I: Integer;
begin
  Assert(Assigned(Src), 'ERROR QNNMax: [src] is empty!');
  Result := Src[0];
  if Assigned(Arg) then
    Arg^ := 0;
  if Stride = 1 then
  begin
{$IF defined(CPUX64) and defined(USE_AVX2)}
    if not Assigned(Arg) then
      Exit(SMaxAVX2(N, Src));
{$ENDIF}
    for I := 1 to N - 1 do
      if Src[I] > Result then
      begin
        Result := Src[I];
        if Assigned(Arg) then
          Arg^ := I
      end
  end
  else
    for I := 1 to N - 1 do
      if Src[I * Stride] > Result then
      begin
        Result := Src[I * Stride];
        if Assigned(Arg) then
          Arg^ := I * Stride
      end

end;

class function TQNNSingleOps.QNNMin(
    const N: Integer;
    const Src: System.PSingle;
    const Arg: PInteger;
    const Stride: Integer
): Single;
var
  I: Integer;
begin
  Assert(Assigned(Src), 'ERROR QNNMin: [src] is empty!');
  Result := Src[0];
  if Assigned(Arg) then
    Arg^ := 0;
  if Stride = 1 then
  begin
{$IF defined(CPUX64) and defined(USE_AVX2)}
    if not Assigned(Arg) then
      Exit(SMinAVX2(N, Src));
{$ENDIF}
    for I := 1 to N - 1 do
      if Src[I] < Result then
      begin
        Result := Src[I];
        if Assigned(Arg) then
          Arg^ := I
      end;
  end
  else
    for I := 1 to N - 1 do
      if Src[I * Stride] < Result then
      begin
        Result := Src[I * Stride];
        if Assigned(Arg) then
          Arg^ := I * Stride
      end
end;

class function TQNNSingleOps.QNNMaxAbs(const N: Integer; const Src: System.PSingle; const Stride: Integer): Single;
var
  I: Integer;
  AB: Single;
begin
  Assert(Assigned(Src), 'ERROR QNNMin: [src] is empty!');
  Result := Abs(Src[0]);
  if Stride = 1 then
  begin
    for I := 1 to N - 1 do
    begin
      AB := Abs(Src[I]);
      if AB > Result then
        Result := AB;
    end;
  end
  else
  begin
    for I := 1 to N - 1 do
    begin
      AB := Abs(Src[I * Stride]);
      if AB > Result then
        Result := AB;
    end;
  end;
end;

class function TQNNSingleOps.QNNMinAbs(const N: Integer; const Src: System.PSingle; const Stride: Integer): Single;
var
  I: Integer;
  AB: Single;
begin
  Assert(Assigned(Src), 'ERROR QNNMin: [src] is empty!');
  Result := Abs(Src[0]);
  if Stride = 1 then
  begin
    for I := 1 to N - 1 do
    begin
      AB := Abs(Src[I]);
      if AB < Result then
        Result := AB;
    end;
  end
  else
  begin
    for I := 1 to N - 1 do
    begin
      AB := Abs(Src[I * Stride]);
      if AB < Result then
        Result := AB;
    end;
  end;
end;

class function TQNNSingleOps.QNNArgMax(const N: Integer; const Src: System.PSingle; const Stride: Integer): Integer;
var
  I: Integer;
  Ma: Single;
begin
  Assert(Assigned(Src), 'ERROR QNNArgMax: [src] is empty!');
  Ma := Src[0];
  Result := 0;
  if Stride = 1 then
  begin
    for I := 1 to N - 1 do
      if Src[I] > Ma then
      begin
        Ma := Src[I];
        Result := I
      end;
  end
  else
    for I := 1 to N - 1 do
      if Src[I * Stride] > Ma then
      begin
        Ma := Src[I * Stride];
        Result := I
      end;

end;

class function TQNNSingleOps.QNNArgMin(const N: Integer; const Src: System.PSingle; const Stride: Integer): Integer;
var
  I: Integer;
  Mi: Single;
begin
  Assert(Assigned(Src), 'ERROR QNNArgMin: [src] is empty!');
  Mi := Src[0];
  Result := 0;
  if Stride = 1 then
  begin
    for I := 1 to N - 1 do
      if Src[I] < Mi then
      begin
        Mi := Src[I];
        Result := I
      end;
  end
  else
    for I := 1 to N - 1 do
      if Src[I * Stride] < Mi then
      begin
        Mi := Src[I * Stride];
        Result := I
      end;
end;

class function TQNNSingleOps.QNNSum(const N: Integer; const Src: System.PSingle; const Stride: Integer): Single;
var
  I: Integer;
begin
  // todo simdify mean
  Result := 0;
  if Stride = 1 then
    for I := 0 to N - 1 do
      Result := Result + Src[I]
  else
    for I := 0 to N - 1 do
      Result := Result + Src[I * Stride];
end;

class function TQNNSingleOps.QNNSumSqr(const N: Integer; const Src: System.PSingle; const Stride: Integer): Single;
var
  I: Integer;
begin
  // todo simdify SumSqr
  Result := 0;
  if Stride = 1 then
{$IF defined(CPUX64) and defined(USE_AVX2)}
    Exit(SSumSqrAVX2(N, Src))
{$ELSE}
    for I := 0 to N - 1 do
      Result := Result + Sqr(Src[I])
{$ENDIF}
  else
    for I := 0 to N - 1 do
      Result := Result + Sqr(Src[I * Stride]);
end;

class function TQNNSingleOps.QNNSumSqrDiff(
    const N: Integer;
    const Src: System.PSingle;
    const AMean: Single;
    const Stride: Integer
): Single;
var
  I: Integer;
begin
  // todo simdify mean
  Result := 0;
  if Stride = 1 then
{$IF defined(CPUX64) and defined(USE_AVX2)}
    Exit(SSumSqrDiffAVX2(N, Src, @AMean))
{$ELSE}
    for I := 0 to N - 1 do
      Result := Result + Sqr(Src[I] - AMean)
{$ENDIF}
  else
    for I := 0 to N - 1 do
      Result := Result + Sqr(Src[I * Stride] - AMean);
end;

class function TQNNSingleOps.QNNSqrDistance(
    const N: Integer;
    const Src1, Src2: System.PSingle;
    const Stride: Integer
): Single;
var
  I: Integer;
begin
  // todo simdify mean
  Result := 0;
  if Stride = 1 then
    for I := 0 to N - 1 do
      Result := Result + Sqr(Src1[I] - Src2[I])
  else
    for I := 0 to N - 1 do
      Result := Result + Sqr(Src1[I * Stride] - Src2[I * Stride]);
end;

class function TQNNSingleOps.QNNMaxAbsDiff2(
    const N: Integer;
    const Src1, Src2: System.PSingle;
    out OutSrc1, OutSrc2: Single;
    const Stride: Integer
): Single;
var
  I: Integer;
  Ab2: Double;
begin
  // todo simdify mean
  OutSrc1 := 0;
  OutSrc2 := 0;

  Result := 0;
  if Stride = 1 then
    for I := 0 to N - 1 do
    begin
      Ab2 := Abs(Src1[I] - Src2[I]);
      // if src1[i] <> 0 then
      //  ab := abs(ab2/src1[i])
      // else
      //  ab:=ab2;
      // assert((ab<0.01) or (ab2< 0.01), 'at pos '+ intToStr(i));
      if Ab2 > Result then
      begin
        OutSrc1 := Src1[I];
        OutSrc2 := Src2[I];
        Result := Ab2;
      end
    end
  else
    for I := 0 to N - 1 do
      Result := System.Math.Max(Result, Abs(Src1[I * Stride] - Src2[I * Stride]));
end;

class function TQNNSingleOps.QNNMaxAbsDiff(
    const N: Integer;
    const Src1, Src2: System.PSingle;
    const Stride: Integer
): Single;
var
  I: Integer;
  Ab2: Double;
begin
  // todo simdify mean
  Assert(Assigned(Src1) and Assigned(Src2), 'ERROR : src1 and src2must be assigned');
  Result := 0;
  if Stride = 1 then
    for I := 0 to N - 1 do
    begin
      Ab2 := Abs(Src1[I] - Src2[I]);
      //  ab := abs(ab2/src1[i])
      // else
      //  ab:=ab2;
      // assert((ab<0.01) or (ab2< 0.01), 'at pos '+ intToStr(i));
      if Ab2 > Result then
      begin
        Result := Ab2;
      end
    end
  else
    for I := 0 to N - 1 do
      Result := System.Math.Max(Result, Abs(Src1[I * Stride] - Src2[I * Stride]));
end;

class function TQNNSingleOps.QNNSumAbsDiff(
    const N: Integer;
    const Src1, Src2: System.PSingle;
    const Stride: Integer
): Single;
var
  I: Integer;
begin
  // todo simdify mean
  Result := 0;
  if Stride = 1 then
    for I := 0 to N - 1 do
      Result := Result + Abs(Src1[I] - Src2[I])
  else
    for I := 0 to N - 1 do
      Result := Result + Abs(Src1[I * Stride] - Src2[I * Stride]);
end;

class function TQNNSingleOps.QNNSumAbsDiffScalar(
    const N: Integer;
    const Src: System.PSingle;
    const AMean: Single;
    const Stride: Integer
): Single;
var
  I: Integer;
begin
  // todo simdify mean
  Result := 0;
  if Stride = 1 then
    for I := 0 to N - 1 do
      Result := Result + Abs(Src[I] - AMean)
  else
    for I := 0 to N - 1 do
      Result := Result + Abs(Src[I * Stride] - AMean);
end;

class function TQNNSingleOps.QNNMean(const N: Integer; const Src: System.PSingle; const Stride: Integer): Single;
begin
  // todo simdify mean
  if N = 0 then
    Exit(0);
  Result := QNNSum(N, Src, Stride);
  Result := Result / N
end;

class function TQNNSingleOps.QNNVariance(
    const N: Integer;
    const Src: System.PSingle;
    const AMean: Single;
    const Stride: Integer;
    const IsPopulation: Boolean
): Single;
begin
  if ((N = 1) and not IsPopulation) or (N = 0) then
    Exit(0);
  Result := QNNSumSqrDiff(N, Src, AMean, Stride);
  if IsPopulation then
    Result := Result / N
  else
    Result := Result / (N - 1)
end;

class procedure TQNNSingleOps.QNNAdd(const Dst, A, B: System.PSingle; const N: Integer);
var
  I: Integer;
begin
  // todo simdify add
  for I := 0 to N - 1 do
    Dst[I] := A[I] + B[I]
end;

class procedure TQNNSingleOps.QNNAddInplace(const Dst, A: System.PSingle; const N: Integer);
var
  I: Integer;
begin
  // todo SIMDIFY ADD_inplace
  for I := 0 to N - 1 do
    Dst[I] := Dst[I] + A[I]
end;

class procedure TQNNSingleOps.QNNMul(const Dst, A, B: System.PSingle; const N: Integer);
var
  I: Integer;
begin
  // todo simdify add
  for I := 0 to N - 1 do
    Dst[I] := A[I] * B[I]
end;

class procedure TQNNSingleOps.QNNMulInplace(const Dst, A: System.PSingle; const N: Integer);
var
  I: Integer;
begin
  // todo SIMDIFY ADD_inplace
  for I := 0 to N - 1 do
    Dst[I] := A[I] * Dst[I]
end;

class procedure TQNNSingleOps.QNNScale(const Dst, Src: System.PSingle; const AScale: Single; const N: LongInt);
var
  I: Integer;
begin
  // todo SIMDIFY QNNScale
  if Dst = Src then
    CblasScal(N, AScale, Dst, 1)
  else
    for I := 0 to N - 1 do
      Dst[I] := Src[I] * AScale
end;

class procedure TQNNSingleOps.QNNScaleInplace(const Dst: System.PSingle; const AScale: Single; const N: LongInt);
begin
  // todo SIMDIFY QNNScale
  CblasScal(N, AScale, Dst, 1);
  // for i:=0 to N-1 do
  //    dst[i] := dst[i]*aScale
end;

class procedure TQNNSingleOps.QNNBias(const Dst, Src: System.PSingle; const ABias: Single; const N: LongInt);
var
  I: Integer;
begin
  // todo SIMDIFY QNNBias
  for I := 0 to N - 1 do
    Dst[I] := Src[I] + ABias;
end;

class procedure TQNNSingleOps.QNNBiasInplace(const Dst: System.PSingle; const ABias: Single; const N: LongInt);
var
  I: Integer;
begin
  // todo SIMDIFY QNNBias
  for I := 0 to N - 1 do
    Dst[I] := Dst[I] + ABias;
end;

class procedure TQNNSingleOps.QNNFusedMulAdd(const Dst, Src, SrcM, SrcA: System.PSingle; const N: LongInt);
var
  I: Integer;
begin
  // todo SIMDIFY QNNFusedMulAdd
  for I := 0 to N - 1 do
    Dst[I] := Src[I] * SrcM[I] + SrcA[I]
end;

class procedure TQNNSingleOps.QNNFusedScaleBias(
    const Dst, Src: System.PSingle;
    const AScale, ABias: Single;
    const N: LongInt
);
{$IF not (defined(CPUX64) and defined(USE_AVX2))}
var
  I: Integer;
{$ENDIF}
begin
  // todo SIMDIFY QNNFusedScaleBias (Scalars)
{$IF defined(CPUX64) and defined(USE_AVX2)}
  SFusedScaleBiasAVX2(N, Dst, Src, AScale, ABias);
{$ELSE}
  for I := 0 to N - 1 do
    Dst[I] := Src[I] * AScale + ABias
{$ENDIF}

end;

class procedure TQNNSingleOps.QNNFusedAddMul(const Dst, Src, SrcA, SrcM: System.PSingle; const N: LongInt);
var
  I: Integer;
begin
  // todo SIMDIFY QNNFusedAddMul
  for I := 0 to N - 1 do
    Dst[I] := (Src[I] + SrcA[I]) * SrcM[I]
end;

class procedure TQNNSingleOps.QNNFusedBiasScale(
    const Dst, Src: System.PSingle;
    const ABias, AScale: Single;
    const N: LongInt
);
var
  I: Integer;
begin
  // todo SIMDIFY QNNFusedScaleBias (Scalars)
  case (2 * LongInt(ABias = 0.0)) or LongInt(AScale = 1.0) of
    1: QNNBias(Dst, Src, ABias, N);
    2: QNNScale(Dst, Src, AScale, N);
    3: QNNCopy(Dst, Src, N);
  else
    for I := 0 to N - 1 do
      Dst[I] := (Src[I] + ABias) * AScale
  end;
end;

class procedure TQNNSingleOps.QNNFusedMulScale(
    const Dst, Src1, Src2: System.PSingle;
    const AScale: Single;
    const N: LongInt
);
{$IF not (defined(CPUX64) and defined(USE_AVX2))}
var
  I: Integer;
{$ENDIF}
begin
  // todo priority SIMDIFY QNNFusedMulScale (Scalars)
  if AScale = 1.0 then
    QNNMul(Dst, Src1, Src2, N)
  else
{$IF defined(CPUX64) and defined(USE_AVX2)}
    SFusedMulScaleAVX2(N, Dst, Src1, Src2, AScale);
{$ELSE}
    for I := 0 to N - 1 do
      Dst[I] := Src1[I] * Src2[I] * AScale
{$ENDIF}
end;

class procedure TQNNSingleOps.QNNFusedBiasMulScale(
    const Dst, Src, SrcM: System.PSingle;
    const ABias, AScale: Single;
    const N: LongInt
);
var
  I: Integer;
begin
  // todo SIMDIFY QNNFusedBiasMulScale (Scalars)
  for I := 0 to N - 1 do
    Dst[I] := (Src[I] + ABias) * SrcM[I] * AScale
end;

class procedure TQNNSingleOps.QNNFusedScaleAdd(
    const Dst, Src, SrcAdd: System.PSingle;
    const AScale: Single;
    const N: LongInt
);
var
  I: LongInt;
begin
  if SrcAdd = Dst then
    CblasAxpy(N, AScale, Src, 1, Dst, 1)
  else
    for I := 0 to N - 1 do
      Dst[I] := AScale * Src[I] + SrcAdd[I]
end;

class procedure TQNNSingleOps.QNNFusedBiasAdd(
    const Dst, Src, SrcA: System.PSingle;
    const ABias: Single;
    const N: LongInt
);
var
  I: Integer;
begin
  // todo SIMDIFY QNNFusedBiasAdd (Scalars)
  for I := 0 to N - 1 do
    Dst[I] := ABias + Src[I] + SrcA[I]
end;

class function TQNNSingleOps.QNNDot(const A, B: System.PSingle; const N: LongInt): Single;
begin
  Result := CblasDot(N, A, 1, B, 1);
end;

class procedure TQNNSingleOps.QNNGatedAdd(const Dst, Gate, Proj: System.PSingle; const Seq, Hidden: Integer);
var
  I: Integer;
begin
  // todo gated add [priority SIMDIFY]
  for I := 0 to Seq - 1 do
    QNNAccMul(Dst + I * Hidden, Gate, Proj + I * Hidden, Hidden);
end;

// RoPE (Rotary Position Embeddings)
class procedure TQNNSingleOps.QNNComputeRoPE(
    const Dst: System.PSingle;
    const MaxSeq, Dim: LongInt;
    const Theta: Single
);
var
  Pos, I, HalfDim, D: LongInt;
  Freq, Angle: Single;
begin
  HalfDim := Dim div 2;
  for Pos := 0 to MaxSeq - 1 do
  begin
    D := Pos * HalfDim * 2;
    for I := 0 to HalfDim - 1 do
    begin
      Freq := 1.0 / Power(Theta, (2 * I) / Dim);
      Angle := Pos * Freq;
      Inc(D, 2);
      Dst[D] := Cos(Angle);
      Dst[D + 1] := Sin(Angle);
    end;
  end;
end;

class procedure TQNNSingleOps.QNNComputeRoPE(
    const CosOut, SinOut: System.PSingle;
    const MaxSeqLen, HeadDim: LongInt;
    const Theta: Single
);
var
  HalfDim, Pos, I: LongInt;
  Freq, Angle: Single;
begin
  HalfDim := HeadDim div 2;

  for Pos := 0 to MaxSeqLen - 1 do
  begin
    for I := 0 to HalfDim - 1 do
    begin
      Freq := 1.0 / Power(Theta, (2 * I) / HeadDim);
      Angle := Pos * Freq;
      CosOut[Pos * HalfDim + I] := Cos(Angle);
      SinOut[Pos * HalfDim + I] := Sin(Angle);
    end
  end;
end;

class procedure TQNNSingleOps.QNNComputeRoPE2D(
    const CosDst, SinDst: System.PSingle;
    const PatchH, PatchW, Dim: LongInt;
    const Theta: Single
);
var
  D, Hy, Wx, Pos, HalfDim: LongInt;
  CosP, SinP: System.PSingle;
  AngleH, AngleW, CosH, CosW, SinH, SinW: Single;
  Freqs: array[0..15] of Single;
begin
  HalfDim := Dim div 2; (* 16 dims per half-axis *)
  // int seq = patch_h * patch_w;
  // (void)seq;

  (* Precompute base frequencies on stack (axis_dim is always 32, half_axis=16) *)
  for D := 0 to HalfDim - 1 do
    Freqs[D] := 1 / Power(Theta, (2 * D) / Dim);

  for Hy := 0 to PatchH - 1 do
  begin
    for Wx := 0 to PatchW - 1 do
    begin
      Pos := Hy * PatchW + Wx;
      CosP := CosDst + Pos * Dim * 4; (* 4 axes * 32 dims each = 128 *)
      SinP := SinDst + Pos * Dim * 4;

      (* Axis 0 (dims 0-31): T position = 0, so cos=1, sin=0 *)
      for D := 0 to Dim - 1 do
      begin
        CosP[D] := 1.0;
        SinP[D] := 0.0;
      end;

      (* Axis 1 (dims 32-63): H position (y/height)
       * Python RoPE stacks [cos, -sin, sin, cos] as 2x2 matrix per freq.
       * For apply_rope: out = [[cos, -sin], [sin, cos]] @ [x0, x1]
       * We store cos/sin per pair and apply_rope_2d handles the rotation.
       *)
      for D := 0 to HalfDim - 1 do
      begin
        AngleH := Hy * Freqs[D];
        CosH := Cos(AngleH);
        SinH := Sin(AngleH);
        (* Each frequency contributes to a pair of dimensions *)
        Pos := Dim + D * 2;
        CosP[Pos] := CosH;
        CosP[Pos + 1] := CosH;
        SinP[Pos] := SinH;
        SinP[Pos + 1] := SinH;
      end;

      (* Axis 2 (dims 64-95): W position (x/width) *)
      for D := 0 to HalfDim - 1 do
      begin
        AngleW := Wx * Freqs[D];
        CosW := Cos(AngleW);
        SinW := Sin(AngleW);
        Pos := Dim * 2 + D * 2;
        CosP[Pos] := CosW;
        CosP[Pos + 1] := CosW;
        SinP[Pos] := SinW;
        SinP[Pos + 1] := SinW;
      end;

      (* Axis 3 (dims 96-127): L position = 0, so cos=1, sin=0 *)
      for D := 0 to Dim - 1 do
      begin
        Pos := Dim * 3 + D;
        CosP[Pos] := 1.0;
        SinP[Pos] := 0.0;
      end
    end
  end

end;

class procedure TQNNSingleOps.QNNComputeRoPE2DOffset(
    const CosDst, SinDst: System.PSingle;
    const PatchH, PatchW, Dim: LongInt;
    const Theta: Single;
    const OffsetT: LongInt
);
var
  D, Hy, Wx, Pos, HalfDim: LongInt;
  CosP, SinP: System.PSingle;
  AngleH, AngleW, AngleT, CosH, CosW, CosT, SinH, SinW, SinT: Single;
  Freqs: array[0..15] of Single;
begin
  HalfDim := Dim div 2; (* 16 dims per half-axis *)
  // int seq = patch_h * patch_w;
  // (void)seq;

  (* Precompute base frequencies on stack (axis_dim is always 32, half_axis=16) *)
  for D := 0 to HalfDim - 1 do
    Freqs[D] := 1 / Power(Theta, (2 * D) / Dim);

  for Hy := 0 to PatchH - 1 do
  begin
    for Wx := 0 to PatchW - 1 do
    begin
      Pos := Hy * PatchW + Wx;
      CosP := CosDst + Pos * Dim * 4; (* 4 axes * 32 dims each = 128 *)
      SinP := SinDst + Pos * Dim * 4;

      (* Axis 0 (dims 0-31): T position = t_offset (non-zero for refs) *)
      for D := 0 to HalfDim - 1 do
      begin
        AngleT := OffsetT * Freqs[D];
        CosT := Cos(AngleT);
        SinT := Sin(AngleT);
        CosP[D * 2] := CosT;
        CosP[D * 2 + 1] := CosT;
        SinP[D * 2] := SinT;
        SinP[D * 2 + 1] := SinT;
      end;

      for D := 0 to HalfDim - 1 do
      begin
        AngleH := Hy * Freqs[D];
        CosH := Cos(AngleH);
        SinH := Sin(AngleH);
        (* Each frequency contributes to a pair of dimensions *)
        CosP[Dim + D * 2] := CosH;
        CosP[Dim + D * 2 + 1] := CosH;
        SinP[Dim + D * 2] := SinH;
        SinP[Dim + D * 2 + 1] := SinH;
      end;

      (* Axis 2 (dims 64-95): W position (x/width) *)
      for D := 0 to HalfDim - 1 do
      begin
        AngleW := Wx * Freqs[D];
        CosW := Cos(AngleW);
        SinW := Sin(AngleW);
        CosP[Dim * 2 + D * 2] := CosW;
        CosP[Dim * 2 + D * 2 + 1] := CosW;
        SinP[Dim * 2 + D * 2] := SinW;
        SinP[Dim * 2 + D * 2 + 1] := SinW;
      end;

      (* Axis 3 (dims 96-127): L position = 0, so cos=1, sin=0 *)
      for D := 0 to Dim - 1 do
      begin
        CosP[Dim * 3 + D] := 1.0;
        SinP[Dim * 3 + D] := 0.0;
      end
    end
  end
end;

class procedure TQNNSingleOps.QNNComputeRoPEText(
    const CosDst, SinDst: System.PSingle;
    const TxtSeq, AxisDim: LongInt;
    const Theta: Single
);
var
  HalfAxis, HeadDim, D, S, Pos: LongInt;
  CosP: System.PSingle;
  SinP: System.PSingle;
  Angle, CosL, SinL: Single;
  Freqs: array[0..15] of Single;
begin
  HalfAxis := AxisDim div 2;
  HeadDim := AxisDim * 4;
  for D := 0 to HalfAxis - 1 do
    Freqs[D] := 1.0 / Power(Theta, (2 * D) / AxisDim);
  for S := 0 to TxtSeq - 1 do
  begin
    CosP := CosDst + S * HeadDim;
    SinP := SinDst + S * HeadDim;
    for D := 0 to AxisDim * 3 - 1 do
    begin
      CosP[D] := 1.0;
      SinP[D] := 0.0
    end;
    for D := 0 to HalfAxis - 1 do
    begin
      Angle := S * Freqs[D];
      CosL := Cos(Angle);
      SinL := Sin(Angle);
      Pos := AxisDim * 3 + D * 2;
      CosP[Pos] := CosL;
      CosP[Pos + 1] := CosL;
      SinP[Pos] := SinL;
      SinP[Pos + 1] := SinL
    end
  end
end;

class procedure TQNNSingleOps.Qscale(N: Int64; Alpha: Single; X: System.PSingle; IncX: Int64); winapi;
var
  I: LongInt;
begin
  // todo qscale Simdify
  if IncX = 1 then
{$IF defined(CPUX64) and defined(USE_AVX2)}
    SScaleInplaceAVX2(N, X, Alpha)
{$ELSE}
    for I := 0 to N - 1 do
      X[I] := X[I] * Alpha
{$ENDIF}
  else
    for I := 0 to N - 1 do
      X[I * IncX] := X[I * IncX] * Alpha
end;

class procedure TQNNSingleOps.Qaxpy(N: Int64; Alpha: Single; X: System.PSingle; Y: System.PSingle);
{$IF not (defined(CPUX64) and defined(USE_AVX2))}
var
  I: LongInt;
{$ENDIF}
begin
  // todo qaxpy Simdify
{$IF defined(CPUX64) and defined(USE_AVX2)}
  SaxpyAVX2(N, Alpha, X, Y)
{$ELSE}
  for I := 0 to N - 1 do
    Y[I] := Alpha * X[I] + Y[I]
{$ENDIF}
end;

class procedure TQNNSingleOps.QaxpyStrided(
    N: Int64;
    Alpha: Single;
    X: System.PSingle;
    IncX: Int64;
    Y: System.PSingle;
    Incy: Int64
); winapi;
var
  I: LongInt;
begin
  // todo qaxpy Simdify
  if (IncX = 1) and (Incy = 1) then
{$IF defined(CPUX64) and defined(USE_AVX2)}
    SaxpyAVX2(N, Alpha, X, Y)
{$ELSE}
    for I := 0 to N - 1 do
      Y[I] := Alpha * X[I] + Y[I]
{$ENDIF}
  else
    for I := 0 to N - 1 do
      Y[I * Incy] := Alpha * X[I * IncX] + Y[I * Incy]
end;

class function TQNNSingleOps.Qdot(N: Int64; X: System.PSingle; Y: System.PSingle): Single;
{$IF not (defined(CPUX64) and defined(USE_AVX2))}
var
  I: LongInt;
{$ENDIF}
begin
  // todo qdot Simdify

{$IF defined(CPUX64) and defined(USE_AVX2)}
  Result := SdotAVX2(N, X, Y)
{$ELSE}
  Result := 0;
  for I := 0 to N - 1 do
    Result := Result + X[I] * Y[I]
{$ENDIF}
end;

class function TQNNSingleOps.QdotStrided(
    N: Int64;
    X: System.PSingle;
    IncX: Int64;
    Y: System.PSingle;
    Incy: Int64
): Single; winapi;
var
  I: LongInt;
begin
  // todo qdot Simdify
  if (IncX = 1) and (Incy = 1) then
  begin
{$IF defined(CPUX64) and defined(USE_AVX2)}
    Result := SdotAVX2(N, X, Y)
{$ELSE}
    Result := 0;
    for I := 0 to N - 1 do
      Result := Result + X[I] * Y[I]
{$ENDIF}
  end
  else
  begin
    Result := 0;
    for I := 0 to N - 1 do
      Result := Result + X[I * IncX] * Y[I * Incy]
  end;
end;

function CeilDiv(const A, B: Int64): LongWord;
begin
  Result := (A + B - 1) div B
end;

const
  // according to my CPU this is the sweet spot
  TILE_M = 8;
  TILE_N = 8;
  TILE_K = 8;

procedure GemmScaleRow(const Count: Int64; const Beta: Single; const Data: System.PSingle);
begin
  if Beta = 0 then
    FillChar(Data^, Count * SizeOf(Single), 0)
  else
    TQNNSingleOps.Qscale(Count, Beta, Data, 1);
end;

procedure GemmNN(
    const M, N, K: Int64;
    const Alpha: Single;
    const A: System.PSingle;
    const Lda: Int64;
    const B: System.PSingle;
    const Ldb: Int64;
    const Beta: Single;
    const C: System.PSingle;
    const Ldc: Int64
);

var
  Tile: TThreadProcNested;
begin
  Tile :=
      procedure(TmStart: IntPtr; Data: Pointer)

      var
        TmFinish, TkStart, TkFinish, Tm, Tk: Int64;
      begin

        // tm_start := 0;
        // while tm_start < CEIL_DIV(M, TILE_M)*TILE_M do begin
        TmFinish := TmStart + TILE_M;
        if TmFinish > M then
          TmFinish := M;

        for Tm := TmStart to TmFinish - 1 do
          GemmScaleRow(N, Beta, C + Tm * Ldc);

        TkStart := 0;
        while TkStart < K {CEIL_DIV(K, TILE_K)*TILE_K} do
        begin
          TkFinish := TkStart + TILE_K;
          if TkFinish > K then
            TkFinish := K;

          for Tm := TmStart to TmFinish - 1 do
          begin
            for Tk := TkStart to TkFinish - 1 do
            begin
              TQNNSingleOps.Qaxpy(N, Alpha * A[Tm * Lda + Tk], B + Tk * Ldb, C + Tm * Ldc);
            end;
          end;

          Inc(TkStart, TILE_K)
        end;
        // inc(tm_start, TILE_M)
        // end;
      end;

{$IF not defined(USE_MULTITHREADING)}
  var I: Int64;
{$ENDIF}

{$IFDEF USE_MULTITHREADING}
  Mp.&for(Tile, 0, CeilDiv(M, TILE_M) * TILE_M, nil, TILE_M);
{$ELSE}
  I := 0;
  while I < CeilDiv(M, TILE_M) * TILE_M - 1 do
  begin
    Tile(I, nil);
    Inc(I, TILE_M)
  end
{$ENDIF}
end;

procedure GemmNT(
    const M, N, K: Int64;
    const Alpha: Single;
    const A: System.PSingle;
    const Lda: Int64;
    const B: System.PSingle;
    const Ldb: Int64;
    const Beta: Single;
    const C: System.PSingle;
    const Ldc: Int64
);

var
  Tile: TThreadProcNested;
begin
  Tile :=
      procedure(TmStart: IntPtr; Data: Pointer)

      var
        TmFinish,
        TNStart,
        TNFinish,
        Tm,
        Tn
            // , tk_start, tk_finish, k_span, n_span
            : Int64;
        APart: Single;
        CC: System.PSingle;
      begin

        // tm_start := 0;
        // while tm_start < CEIL_DIV(M, TILE_M)*TILE_M do begin
        TmFinish := TmStart + TILE_M;
        if TmFinish > M then
          TmFinish := M;

        for Tm := TmStart to TmFinish - 1 do
          GemmScaleRow(N, Beta, C + Tm * Ldc);

        TNStart := 0;
        while TNStart < N {CEIL_DIV(N, TILE_N)*TILE_N} do
        begin
          TNFinish := TNStart + TILE_N;
          if TNFinish > N then
            TNFinish := N;
          // n_span := tn_finish - tn_start;
          // tk_start := 0;
          (* K tiling is slower,  perhaps we need to select TILE values
       based on the matricies dimensions and the CPU L1/L2 sizes   *)
          // while tk_start < K do begin
          //  tk_finish := tk_start + TILE_K;
          //  if tk_finish>K then tk_finish := K;
          //  k_span := tk_finish - tk_start;
          //  for tm:=tm_start to tm_finish-1 do begin
          //    CC := C + tm*ldc;
          //    for tn:= tn_start to tn_finish-1 do begin
          //      CC[tn] := CC[tn] + ALPHA*TQNNSingleOPS.qdot(k_span, A + tm*lda + tk_start, B + tn*ldb + tk_start);
          //    end;
          //  end;
          //  inc(tk_start, TILE_K)
          // end;

          // for tm := tm_start to tm_finish-1 do begin
          //  CC := C + tm*ldc;
          //  for tn := tn_start to tn_finish-1 do begin
          //    a_part := ALPHA*TQNNSingleOps.qdot(K, A + tm*lda, B + tn*ldb);
          //    CC[tn] := CC[tn] + a_part;
          //  end;
          // end;

          for Tm := TmStart to TmFinish - 1 do
          begin
            CC := C + Tm * Ldc;
            Tn := TNStart;
{$IF defined(CPUX64) and defined(USE_AVX2)}
            while Tn + 4 <= TNFinish do
            begin
              SDotX4AVX2(A + Tm * Lda, B + Tn * Ldb, CC + Tn, K, Ldb, Alpha);
              Inc(Tn, 4)
            end;
{$ENDIF}
            while Tn < TNFinish do
            begin
              APart := Alpha * TQNNSingleOps.Qdot(K, A + Tm * Lda, B + Tn * Ldb);
              CC[Tn] := CC[Tn] + APart;
              Inc(Tn)
            end;
          end;
          Inc(TNStart, TILE_N)
        end;
        //  inc(tm_start, TILE_M)
        // end;
      end;
{$IF not defined(USE_MULTITHREADING)}
  var I: Int64;
{$ENDIF}

  // TQNNSingleOPS.cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, alpha, A, lda, B, ldb, BETA, C, ldc);
  // exit;
{$IFDEF USE_MULTITHREADING}
  Mp.&for(Tile, 0, CeilDiv(M, TILE_M) * TILE_M, nil, TILE_M);
{$ELSE}
  I := 0;
  while I < CeilDiv(M, TILE_M) * TILE_M - 1 do
  begin
    Tile(I, nil);
    Inc(I, TILE_M)
  end
{$ENDIF}
end;

procedure GemmTN(
    const M, N, K: Int64;
    const Alpha: Single;
    const A: System.PSingle;
    const Lda: Int64;
    const B: System.PSingle;
    const Ldb: Int64;
    const Beta: Single;
    const C: System.PSingle;
    const Ldc: Int64
);

var
  Tile: TThreadProcNested;
begin
  Tile :=
      procedure(TmStart: IntPtr; Data: Pointer)

      var
        TmFinish, TkStart, TkFinish, Tm, Tk: Int64;
        Aa: System.PSingle;
      begin
        // tm_start := 0;
        // while tm_start < CEIL_DIV(M, TILE_M)*TILE_M do begin
        TmFinish := TmStart + TILE_M;
        if TmFinish > M then
          TmFinish := M;

        for Tm := TmStart to TmFinish - 1 do
          GemmScaleRow(N, Beta, C + Tm * Ldc);

        TkStart := 0;
        while TkStart < K {CEIL_DIV(K, TILE_K)*TILE_K} do
        begin
          TkFinish := TkStart + TILE_K;
          if TkFinish > K then
            TkFinish := K;

          for Tk := TkStart to TkFinish - 1 do
          begin
            Aa := A + Tk * Lda;
            for Tm := TmStart to TmFinish - 1 do
            begin
              TQNNSingleOps.Qaxpy(N, Alpha * Aa[Tm], B + Tk * Ldb, C + Tm * Ldc);
            end;
          end;

          Inc(TkStart, TILE_K)
        end;
        // inc(tm_start, TILE_M)
        // end;
      end;

{$IF not defined(USE_MULTITHREADING)}
  var I: Int64;
{$ENDIF}

{$IFDEF USE_MULTITHREADING}
  Mp.&for(Tile, 0, CeilDiv(M, TILE_M) * TILE_M, nil, TILE_M);
{$ELSE}
  I := 0;
  while I < CeilDiv(M, TILE_M) * TILE_M - 1 do
  begin
    Tile(I, nil);
    Inc(I, TILE_M)
  end
{$ENDIF}
end;

class procedure TQNNSingleOps.Qgemm(
    Order: TCBLASOrder;
    TransA: TCBLASTranspose;
    TransB: TCBLASTranspose;
    M: Int64;
    N: Int64;
    K: Int64;
    Alpha: Single;
    A: System.PSingle;
    Lda: Int64;
    B: System.PSingle;
    Ldb: Int64;
    Beta: Single;
    C: System.PSingle;
    Ldc: Int64
); winapi;

begin
  // todo qgemm Simdify
  Assert(
      (Order = CblasRowMajor) and not ((TransA = CblasTrans) and (TransB = CblasTrans)),
      'ERROR : Operation is not supported using the provided arguments!'
  );

  if (TransA = CblasNoTrans) and (TransB = CblasNoTrans) then
    GemmNN(M, N, K, Alpha, A, Lda, B, Ldb, Beta, C, Ldc)
  else if (TransA = CblasNoTrans) and (TransB = CblasTrans) then
    GemmNT(M, N, K, Alpha, A, Lda, B, Ldb, Beta, C, Ldc)
  else if (TransA = CblasTrans) and (TransB = CblasNoTrans) then
    GemmTN(M, N, K, Alpha, A, Lda, B, Ldb, Beta, C, Ldc)
  else
    Exit;

end;

function Min(const A, B: LongInt): LongInt; inline;
begin
  if A > B then
    Exit(B)
  else
    Exit(A)
end;

class function TQNNSingleOps.Qasum(N: Int64; X: System.PSingle; IncX: Int64): Single; winapi;
var
  I: LongInt;
begin
  Result := 0;
  if IncX = 1 then
    for I := 0 to N - 1 do
      Result := Result + Abs(X[I])
  else
    for I := 0 to N - 1 do
      Result := Result + Abs(X[I * IncX])
end;

class procedure TQNNSingleOps.QNNAccAdd(const Dst, A, B: System.PSingle; const N: Integer);
var
  I: Integer;
begin
  // todo simdify accadd
  for I := 0 to N - 1 do
    Dst[I] := Dst[I] + A[I] + B[I]
end;

class procedure TQNNSingleOps.QNNAccMul(const Dst, A, B: System.PSingle; const N: Integer);
var
  I: Integer;
begin
  // todo simdify accmul
  for I := 0 to N - 1 do
    Dst[I] := Dst[I] + A[I] * B[I]
end;

class procedure TQNNSingleOps.QNNMatMulNN(const C, A, B: TMemoryBlock; const M, K, N: Integer);
begin
  Assert((A.Count >= M * K) or (Assigned(A.DataPtr) and (A.Count = 0)), 'ERROR : QNNMatMul out of bound [A]');
  Assert((B.Count >= K * N) or (Assigned(B.DataPtr) and (B.Count = 0)), 'ERROR : QNNMatMul out of bound [B]');
  Assert((C.Count >= M * N) or (Assigned(C.DataPtr) and (C.Count = 0)), 'ERROR : QNNMatMul out of bound [C]');
  CblasGemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1, A, K, B, N, 0, C, N);
end;

class procedure TQNNSingleOps.QNNMatMulNT(const C, A, B: TMemoryBlock; const M, K, N: Integer);
begin
  Assert((A.Count >= M * K) or (Assigned(A.DataPtr) and (A.Count = 0)), 'ERROR : QNNMatMul out of bound [A]');
  Assert((B.Count >= N * K) or (Assigned(B.DataPtr) and (B.Count = 0)), 'ERROR : QNNMatMul out of bound [B]');
  Assert((C.Count >= M * N) or (Assigned(C.DataPtr) and (C.Count = 0)), 'ERROR : QNNMatMul out of bound [C]');
  CblasGemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1, A, K, B, K, 0, C, N);
end;

class procedure TQNNSingleOps.QNNMatMulTN(const C, A, B: TMemoryBlock; const M, K, N: Integer);
begin
  Assert((A.Count >= K * M) or (Assigned(A.DataPtr) and (A.Count = 0)), 'ERROR : QNNMatMul out of bound [A]');
  Assert((B.Count >= N * K) or (Assigned(B.DataPtr) and (B.Count = 0)), 'ERROR : QNNMatMul out of bound [B]');
  Assert((C.Count >= M * N) or (Assigned(C.DataPtr) and (C.Count = 0)), 'ERROR : QNNMatMul out of bound [C]');
  CblasGemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, N, K, 1, A, K, B, K, 0, C, N);
end;

class procedure TQNNSingleOps.QNNMatTranspose(const Dst, Src: System.PSingle; const SrcRows, SrcCols: LongInt);
var
  C, R: LongInt;
begin
  for C := 0 to SrcCols - 1 do
    for R := 0 to SrcRows - 1 do
      Dst[C * SrcRows + R] := Src[R * SrcCols + C]
end;

class procedure TQNNSingleOps.QNNLinear(const Dst, X, W, B: System.PSingle; const SeqLen, InDIM, OutDIM: Integer);
var
  I: Integer;
begin
  CblasGemm(CblasRowMajor, CblasNoTrans, CblasTrans, SeqLen, OutDIM, InDIM, 1.0, X, InDIM, W, InDIM, 0.0, Dst, OutDIM);

  (* Add bias if present *)
  if Assigned(B) then
    for I := 0 to SeqLen - 1 do
      QNNAddInplace(Dst + I * OutDIM, B, OutDIM)
end;

class procedure TQNNSingleOps.QNNLinearNoBias(const Dst, X, W: System.PSingle; const SeqLen, InDIM, OutDIM: Integer);
begin
  CblasGemm(CblasRowMajor, CblasNoTrans, CblasTrans, SeqLen, OutDIM, InDIM, 1.0, X, InDIM, W, InDIM, 0.0, Dst, OutDIM);
end;

class procedure TQNNSingleOps.QNNLinearNoBiasBF16(
    const Dst, X: System.PSingle;
    const W: PBF16;
    const SeqLen, InDIM, OutDIM: Integer
);
begin
  if TryVulkanLinearBF16(Dst, X, W, SeqLen, InDIM, OutDIM) then
    Exit;
  // if assigned(cblas_sbgemm) then
  //  cblas_sbgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
  //            seqLen, outDIM, inDIM,
  //            1.0,
  //            x, inDIM,
  //            W, inDIM,
  //            0.0,
  //            dst, outDIM)
  // else
  begin
    if Length(Workspace) < InDIM * OutDIM then
      SetLength(Workspace, InDIM * OutDIM);
    BF16ToSingle(InDIM * OutDIM, W, Pointer(Workspace));
    QNNLinearNoBias(Dst, X, Pointer(Workspace), SeqLen, InDIM, OutDIM);
  end
end;

class procedure TQNNSingleOps.QNNIm2Col(
    const AChannels,
    AHeight,
    AWidth,
    KernelHeight,
    KernelWidth,
    PadHeight,
    PadWidth,
    StrideY,
    StrideX,
    DilationY,
    DilationX: Integer;
    const Im: System.PSingle;
    const Col: System.PSingle;
    const ImOffset: Integer;
    const ColOffset: Integer;
    const MultiThread: Boolean
);
var
  Channel, OutputH, OutputW, ChannelSize, OutChannelSize, KernelSize: Integer;

  I2cExt: TThreadProcNested;
begin
  // {$ifdef USE_TELEMETRY} if benchmark then metrics.ops.start(opIm2colExt);{$endif}
  I2cExt :=
      procedure(Idx: IntPtr; Ptr: Pointer)

      var
        KernelRow, KernelCol, OutputCol, OutputRow, InputRow, InputCol, I: NativeInt;
        DIm, DCol: System.PSingle;
      begin
        DIm := Im + ImOffset + ChannelSize * Idx;
        DCol := Col + ColOffset + KernelSize * OutChannelSize * Idx;
        for KernelRow := 0 to KernelHeight - 1 do
          for KernelCol := 0 to KernelWidth - 1 do
          begin
            // fillChar(d_col^, output_w*output_h*sizeOf(single), #0);
            InputRow := -PadHeight + KernelRow * DilationY;
            for OutputRow := 0 to OutputH - 1 do
            begin
              if (NativeUInt(InputRow) < NativeUInt(AHeight)) then
              begin
                InputCol := -PadWidth + KernelCol * DilationX;
                if (StrideX = 1) then
                begin
                  // fillchar(d_col[0], output_w*SizeOf(d_col^), #0);
                  Move(
                      DIm[InputRow * AWidth + Ord(InputCol >= 0) * InputCol],
                      DCol[Ord(InputCol < 0) * Abs(InputCol)],
                      (OutputW - Abs(InputCol)) * SizeOf(DCol^)
                  );
                end
                else
                  for OutputCol := 0 to OutputW - 1 do
                  begin
                    I := OutputCol * StrideX + InputCol;
                    DCol[OutputCol] := Ord(NativeUInt(I) < NativeUInt(AWidth)) * DIm[InputRow * AWidth + I];
                  end; // or use the following (UnOptimized)
              end
              else
                // fillchar(d_col^, output_w*SizeOf(d_col^), #0)
                    ;
              Inc(DCol, OutputW);
              Inc(InputRow, StrideY);
            end;
          end;
      end;

  OutputW := (AWidth + 2 * PadWidth - (DilationX * (KernelWidth - 1) + 1)) div StrideX + 1;
  OutputH := (AHeight + 2 * PadHeight - (DilationY * (KernelHeight - 1) + 1)) div StrideY + 1;
  ChannelSize := AHeight * AWidth;
  OutChannelSize := OutputW * OutputH;
  KernelSize := KernelWidth * KernelHeight;
{$IFDEF USE_MULTITHREADING}
  if MultiThread then
    Mp.&for(I2cExt, 0, AChannels {, @p})
  else
    for Channel := 0 to AChannels - 1 do
      I2cExt(Channel, {@p} nil);
{$ELSE}
  for Channel := 0 to AChannels - 1 do
    I2cExt(Channel, {@p} nil);
{$ENDIF}
  // {$ifdef USE_TELEMETRY} if benchmark then metrics.ops.finish(opIm2colExt);{$endif}
end;

class procedure TQNNSingleOps.QNNIm2ColStridedBatched(
    const AChannels,
    AHeight,
    AWidth,
    KernelHeight,
    KernelWidth,
    PadHeight,
    PadWidth,
    StrideY,
    StrideX,
    DilationY,
    DilationX: Integer;
    const Im: System.PSingle;
    const ImStride, ImOffset: Integer;
    const Col: System.PSingle;
    const ColStride, ColOffset: Integer;
    const Batch: Integer
);
var
  B: NativeInt;
  Mt: Boolean;

  I2c: TThreadProcNested;
begin
  I2c :=
      procedure(Idx: IntPtr; Ptr: Pointer)
      begin
        QNNIm2Col(
            AChannels,
            AHeight,
            AWidth,
            KernelHeight,
            KernelWidth,
            PadHeight,
            PadWidth,
            StrideY,
            StrideX,
            DilationY,
            DilationX,
            Im + Idx * ImStride,
            Col + Idx * ColStride,
            ImOffset,
            ColOffset,
            Mt
        );
      end;

  Mt := Batch = 1;
{$IFDEF USE_MULTITHREADING}
  if not Mt then
    Mp.&for(I2c, 0, Batch, nil)
  else
{$ENDIF}
    for B := 0 to Batch - 1 do
      QNNIm2Col(
          AChannels,
          AHeight,
          AWidth,
          KernelHeight,
          KernelWidth,
          PadHeight,
          PadWidth,
          StrideY,
          StrideX,
          DilationY,
          DilationX,
          Im + B * ImStride,
          Col + B * ColStride,
          ImOffset,
          ColOffset,
          Mt
      );
end;

// {$define USE_IM2Col}
class procedure TQNNSingleOps.QNNConv2d(
    const Dst, Src, Weights, Bias: System.PSingle;
    const InCh, OutCh, H, W, KH, KW, Stride, Padding: LongInt;
    const Batch: LongInt
);
const
  MAX_COL_SIZE = 256 * 1024 * 1024;
var
  OutW, OutH, K, B: Integer;

{$IF defined(USE_NAIVE)}
  procedure Kernel(Idx: IntPtr; Data: Pointer);
  var
    Oc, Oh, Ow, Ic, KernelRow, KernelCol, Ih, Iw, InIdx, WIdx, OutIdx: LongInt;
    Sum: Single;
    SS, WW, DD: System.PSingle;
  begin
    SS := Src;
    WW := Weights;
    DD := Dst;
    Oc := Idx div OutCh;
    Oh := Idx mod OutH;
    for Ow := 0 to OutW - 1 do
    begin
      if Assigned(Bias) then
        Sum := Bias[Oc]
      else
        Sum := 0.0;
      for Ic := 0 to InCh - 1 do
        for KernelRow := 0 to KH - 1 do
          for KernelCol := 0 to KW - 1 do
          begin
            Ih := Oh * Stride - Padding + KernelRow;
            Iw := Ow * Stride - Padding + KernelCol;

            if (Ih >= 0) and (Ih < H) and (Iw >= 0) and (Iw < W) then
            begin
              InIdx := B * InCh * H * W + Ic * H * W + Ih * W + Iw;
              WIdx := Oc * InCh * KH * KW + Ic * KH * KW + KernelRow * KW + KernelCol;
              Sum := Sum + SS[InIdx] * WW[WIdx];
            end;
          end;

      OutIdx := B * OutCh * OutH * OutW + Oc * OutH * OutW + Oh * OutW + Ow;
      DD[OutIdx] := Sum;
    end;
  end;

var
  I: LongInt;
{$ELSE}
  ColSize, {max_col_size, } TileColSize, RowSize: IntPtr;
  Col, Col2: TMemoryBlock;
  ColPtr, InB, OutB, OutTile: System.PSingle;
  TileRows,
  TileStart,
  TileEnd,
  TileH,
  TilePixels,
  ColRow,
  Ic,
  KernelRow,
  KernelCol,
  Oh,
  Ow,
  Ih,
  Iw, {col_idx, }
  Oc,
  OutSpatial,
  InputCol: LongInt;
  BVal: Single;
{$ENDIF}
begin
  Assert(Assigned(Dst) and Assigned(Src) and Assigned(Weights), 'ERROR : dst, src and weights must be assigned');

{$IF defined(USE_NAIVE)}
  OutH := (H + 2 * Padding - KH) div Stride + 1;
  OutW := (W + 2 * Padding - KW) div Stride + 1;
  // Naive implementation with threadings (fallback)
  for B := 0 to Batch - 1 do
  {$IFDEF USE_MULTITHREADING}
    Mp.&for(Kernel, 0, OutCh * OutH);
  {$ELSE}
    for I := 0 to OutCh * OutH - 1 do
      Kernel(I, nil);
  {$ENDIF}
  // for oc := 0 to out_ch-1 do
  //    for oh := 0 to outH-1 do
  // for ow := 0 to outW-1 do begin
  //    if assigned(bias) then sum := bias[oc] else sum := 0.0;
  //    for ic := 0 to in_ch-1 do
  //        for kernel_row := 0 to kH-1 do
  //            for kernel_col := 0 to KW-1 do begin
  //                ih := oh * stride - padding + kernel_row;
  //                iw := ow * stride - padding + kernel_col;
  //
  //                if (ih >= 0) and (ih < H) and (iw >= 0) and (iw < W) then begin
  //                    in_idx := b * in_ch * H * W + ic * H * W + ih * W + iw;
  //                    w_idx  := oc * in_ch * kH * kW + ic * kH * kW + kernel_row * kW + kernel_col;
  //                    sum := sum + src[in_idx] * weights[w_idx];
  //                end;
  //            end;
  //
  //    out_idx := b * out_ch * outH * outW + oc * outH * outW + oh * outW + ow;
  //    dst[out_idx] := sum;
  // end;
  Exit;
{$ELSE}
  OutH := (H + 2 * Padding - KH) div Stride + 1;
  OutW := (W + 2 * Padding - KW) div Stride + 1;
  OutSpatial := OutH * OutW;
  ColSize := InCh * KH * KW * OutH * OutW;

  TileRows := OutH;
  {$IF not defined(USE_IM2COL)}
  if ColSize > MAX_COL_SIZE then
  begin
    RowSize := InCh * KH * KW * OutW;
    TileRows := MAX_COL_SIZE div RowSize;
    if TileRows < 1 then
      TileRows := 1
  end;
  TileColSize := InCh * KH * KW * TileRows * OutW;
  {$ELSE}
  {$ENDIF}
  if KH * KW <> 1 then
    Col := TMemoryBlock.Create([InCh, KH, KW, TileRows, OutW], 'IM2COL ' + TGUID.NewGuid.ToString());
  // col2 := TMemoryBlock.Create([in_ch, kH, kW, tile_rows, outW]);

  for B := 0 to Batch - 1 do
  begin
    InB := Src + B * InCh * H * W;
    OutB := Dst + B * OutCh * OutH * OutW;
    TileStart := 0;
    while TileStart < OutH do
    begin
      TileEnd := TileStart + TileRows;
      if TileEnd > OutH then
        TileEnd := OutH;
      TileH := TileEnd - TileStart;
      TilePixels := TileH * OutW;
      ColRow := 0;
      if KH * KW = 1 then // if pointwise conv
        Col.AssignPtr(InB, [InCh, KH, KW, TileRows, OutW])
      else
  {$IF defined(USE_IM2COL)}
        QNNIm2Col(InCh, H, W, KH, KW, Padding, Padding, Stride, Stride, 1, 1, InB, Col);
  {$ELSE}
        for Ic := 0 to InCh - 1 do
          for KernelRow := 0 to KH - 1 do
            for KernelCol := 0 to KW - 1 do
            begin
              for Oh := TileStart to TileEnd - 1 do
              begin
                Ih := Oh * Stride + KernelRow - Padding;
                // col_idx := col_row * tile_pixels+(oh-tile_start) * outW;
                ColPtr := Col + (ColRow * TilePixels + (Oh - TileStart) * OutW);
                InputCol := KernelCol - Padding;
                if (LongWord(Ih) < LongWord(H)) then
                  if Stride = 1 then
                  begin
                    FillChar(ColPtr[0], OutW * SizeOf(Single), #0);
                    Move(
                        InB[Ic * H * W + Ih * W + Ord(InputCol >= 0) * InputCol],
                        ColPtr[Ord(InputCol < 0) * Abs(InputCol)],
                        (OutW - Abs(InputCol)) * SizeOf(Single)
                    )
                        // for ow := 0 to outW -1 do begin
                        //  iw := ow -padding + kernel_col;
                        //  if (longword(iw) < longword(W)) then
                        //    col_ptr[ow] := in_b[ic * H * W+ih * W+iw]
                        //  //else
                        //  //  col_ptr[ow] := 0.0
                        // end
                  end
                  else
                    for Ow := 0 to OutW - 1 do
                    begin
                      Iw := Ow * Stride - Padding + KernelCol;
                      if (LongWord(Iw) < LongWord(W)) then
                        ColPtr[Ow] := InB[Ic * H * W + Ih * W + Iw]
                      else
                        ColPtr[Ow] := 0.0
                    end
                else
                  FillChar(ColPtr[0], OutW * SizeOf(Single), #0);
              end;
              Inc(ColRow)
            end;
  {$ENDIF}
      // QNNIm2Col(in_ch, H, W, KH, KW, padding, padding, stride, stride, 1, 1, in_b, col2);
      // col.printCompare(col2);
      K := InCh * KH * KW;
      OutTile := OutB + TileStart * OutW;
      CblasGemm(
          CblasRowMajor,
          CblasNoTrans,
          CblasNoTrans,
          OutCh,
          TilePixels,
          K,
          1.0,
          Weights,
          K,
          Col,
          TilePixels,
          0.0,
          OutTile,
          OutH * OutW
      );
      Inc(TileStart, TileRows)
    end;
    if Assigned(Bias) then
      for Oc := 0 to OutCh - 1 do
      begin
        BVal := Bias[Oc];
        QNNBiasInplace(OutB + Oc * OutSpatial, BVal, OutSpatial);
        // out_ch_ptr := out_b + oc*outH*outW;
        // for i := 0 to outSpatial -1 do
        //    out_ch_ptr[i] := out_ch_ptr[i] + b_val
      end
  end;
  Col.Free;
  Col2.Free;
{$ENDIF}
end;

class procedure TQNNSingleOps.QNNNorm(
    const N: LongInt;
    const Dst, Src: System.PSingle;
    const Weights: System.PSingle;
    const Stride: LongInt
);
var
  Mean, InvStdDev: Single;
  I: LongInt;
begin
  Mean := QNNMean(N, Src, Stride);
  InvStdDev := 1 / Sqrt(QNNVariance(N, Src, Mean, Stride) + EPSILON);
  if Assigned(Weights) then
    if Stride = 1 then
      QNNFusedBiasMulScale(Dst, Src, Weights, -Mean, InvStdDev, N)
          // for i:=0 to N-1 do
          //  dst[i] := (src[i]-mean)*invStdDev*weights[i]
    else
      for I := 0 to N - 1 do
        Dst[I * Stride] := (Src[I * Stride] - Mean) * InvStdDev * Weights[I * Stride]
  else if Stride = 1 then
    QNNFusedBiasScale(Dst, Src, -Mean, InvStdDev, N)
        // for i:=0 to N-1 do
        //  dst[i] := (src[i]-mean)*invStdDev
  else
    for I := 0 to N - 1 do
      Dst[I * Stride] := (Src[I * Stride] - Mean) * InvStdDev
end;

class procedure TQNNSingleOps.QNNRMSNorm(
    const N: LongInt;
    const Dst, Src: System.PSingle;
    const Weights: System.PSingle;
    const Stride: LongInt
);
var
  InvRMS: Single;
  I: LongInt;
begin
  // sum_sq := 0.0;
  // for i := 0 to N-1 do
  //  sum_sq := sum_sq + src[i]*src[i];
  // rms := sqrt(sum_sq / N + EPSILON);
  // invRMS := 1.0 / rms;
  // for i := 0 to N-1 do
  //    dst[i] := src[i]*invRMS*weights[i];
  InvRMS := 1.0 / Sqrt(QNNSumSqr(N, Src, Stride) / N + EPSILON);
  if Assigned(Weights) then
    if Stride = 1 then
      QNNFusedMulScale(Dst, Src, Weights, InvRMS, N)
    else
      for I := 0 to N - 1 do
        Dst[I * Stride] := Src[I * Stride] * Weights[I] * InvRMS
  else if Stride = 1 then
    QNNScale(Dst, Src, InvRMS, N)
  else
    for I := 0 to N - 1 do
      Dst[I * Stride] := Src[I * Stride] * InvRMS

end;

class procedure TQNNSingleOps.QNNRMSNormRows(const Dst, Src, Weight: System.PSingle; const Rows, Dim: Integer);
var
  R
      // , i
      : Integer;
// sq_sum, rms, rms_inv:Single;
// x_row, out_row : PSingle;
begin
  // todo RMS_NORM Simdify
  for R := 0 to Rows - 1 do
    QNNRMSNorm(Dim, Dst + R * Dim, Src + R * Dim, Weight, 1);

end;

class procedure TQNNSingleOps.QNNTanh(const Dst: System.PSingle; const N: LongInt);
var
  I: Integer;
begin
  for I := 0 to N - 1 do
    Dst[I] := Tanh(Dst[I])
end;

class procedure TQNNSingleOps.QNNQKRMSNorm(
    const Q, K, QWeights, KWeights: System.PSingle;
    const Seq, Heads, HeadDim: LongInt
);
var
  Qh, KH: System.PSingle;

  I, J: LongInt;
begin
  for I := 0 to Seq - 1 do
  begin
    for J := 0 to Heads - 1 do
    begin
      (* Q normalization *)
      Qh := Q + I * Heads * HeadDim + J * HeadDim;

      // float sum_sq = 0.0f;
      // for (int d = 0; d < head_dim; d++) begin
      //    sum_sq += qh[d] * qh[d];
      // end

      // rmsInv := 1.0 / sqrtf(sum_sq / head_dim + eps);
      // for (int d = 0; d < head_dim; d++) begin
      //    qh[d] = qh[d] * rms_inv * q_weight[d];
      // end
      QNNRMSNorm(HeadDim, Qh, Qh, QWeights);

      (* K normalization *)
      KH := K + I * Heads * HeadDim + J * HeadDim;
      // sum_sq = 0.0f;
      // for (int d = 0; d < head_dim; d++) begin
      //    sum_sq += kh[d] * kh[d];
      // end
      // rms_inv = 1.0f / sqrtf(sum_sq / head_dim + eps);
      // for (int d = 0; d < head_dim; d++) begin
      //    kh[d] = kh[d] * rms_inv * k_weight[d];
      // end
      QNNRMSNorm(HeadDim, KH, KH, KWeights);

    end
  end

end;

class procedure TQNNSingleOps.QNNRMSNormSeq(const Dst, NormWeights: System.PSingle; const Seq, Heads, HeadDim: LongInt);
var
  S: LongInt;
  Head: System.PSingle;
begin
  for S := 0 to Seq - 1 do
  begin
    Head := Dst + S * Heads * HeadDim;
    QNNRMSNormRows(Head, Head, NormWeights, Heads, HeadDim);
  end;
end;

class procedure TQNNSingleOps.QNNGroupNorm(
    const Dst, Src, Gamma, Beta: System.PSingle;
    const Batch, Channels, H, W, NumGroups: Integer
);
var
  ChannelsPerGroup, Spatial, B, G, CStart, C: Integer;
  Mean, Variance, StdInv: Single;
  Src1, Dst1: System.PSingle;
begin
  ChannelsPerGroup := Channels div NumGroups;
  Spatial := H * W;
  // todo group norm simdify
  for B := 0 to Batch - 1 do
  begin
    for G := 0 to NumGroups - 1 do
    begin
      CStart := G * ChannelsPerGroup;

      // mean := 0.0;
      // count := 0;
      // for c := c_start to c_end-1 do begin
      //    for i := 0 to spatial-1 do begin
      //        idx := b*channels*spatial + c*spatial + i;
      //        mean := mean + src[idx];
      //        inc(count);
      //    end
      // end;
      // mean := mean/count;

      Src1 := Src + B * Channels * Spatial + CStart * Spatial;
      Dst1 := Dst + B * Channels * Spatial + CStart * Spatial;
      Mean := QNNMean(Spatial * ChannelsPerGroup, Src1, 1);

      // variance := 0.0;
      // for c := c_start to c_end-1 do begin
      //    for i := 0 to spatial-1 do begin
      //        idx := b*channels*spatial + c*spatial + i;
      //        diff := src[idx] - mean;
      //        variance := variance + diff*diff;
      //    end
      // end;
      // variance := variance / count;
      Variance := QNNVariance(Spatial * ChannelsPerGroup, Src1, Mean);

      StdInv := 1.0 / Sqrt(Variance + EPSILON);

      for C := 0 to ChannelsPerGroup - 1 do
        QNNFusedScaleBias(
            Dst1 + C * Spatial,
            Src1 + C * Spatial,
            Gamma[C + CStart] * StdInv,
            Beta[C + CStart] - Mean * Gamma[C + CStart] * StdInv,
            Spatial
        );
      // for c := c_start to c_end-1 do begin
      //  for i := 0 to spatial-1 do begin
      //    idx := b*channels*spatial + c * spatial + i;
      //    dst[idx] := src[idx]*gamma[c]*std_inv - mean*gamma[c]*std_inv + beta[c];
      //  end
      // end
    end
  end
end;

class procedure TQNNSingleOps.QNNBatchNorm(
    const Dst, Src, RunningMean, RunningVar, Gamma, Beta: System.PSingle;
    const Batch, Channels, H, W: Integer
);
var
  Spatial, C, N: Integer;
  Mean, Variance, StdInv, G, BVal: Single;
  Src1, Dst1: System.PSingle;
begin
  // todo BatchNorm simdify
  Spatial := H * W;
  for C := 0 to Channels - 1 do
  begin
    Mean := RunningMean[C];
    Variance := RunningVar[C];
    StdInv := 1.0 / Sqrt(Variance + EPSILON);
    if Assigned(Gamma) then
      G := Gamma[C]
    else
      G := 1.0;
    if Assigned(Beta) then
      BVal := Beta[C]
    else
      BVal := 0.0;

    for N := 0 to Batch - 1 do
    begin
      Src1 := Src + N * Channels * Spatial + C * Spatial;
      Dst1 := Dst + N * Channels * Spatial + C * Spatial;
      QNNFusedScaleBias(Dst1, Src1, G * StdInv, BVal - Mean * G * StdInv, Spatial);
    end

    // for n := 0 to batch-1 do begin
    //    for i := 0 to spatial-1 do begin
    //        idx := n * channels * spatial + c * spatial + i;
    //        dst[idx] := g * (src[idx] - mean) * std_inv + b_val;
    //    end
    // end

  end
end;

class procedure TQNNSingleOps.QNNSigmoidInplace(const X: System.PSingle; const N: Integer);
{$IF not (defined(CPUX64) and defined(USE_AVX2))}
var
  I: Integer;
{$ENDIF}
begin
  // todo sigmoid inplace simdify
{$IF defined(CPUX64) and defined(USE_AVX2)}
  SSiLUAVX2(N, X, nil, X, nil);
{$ELSE}
  for I := 0 to N - 1 do
    X[I] := Single(1.0) / (Single(1.0) + FastExp(-X[I]));
{$ENDIF}
end;

class procedure TQNNSingleOps.QNNSigmoid(const Dst, Src: System.PSingle; const N: Integer);
{$IF not (defined(CPUX64) and defined(USE_AVX2))}
var
  I: Integer;
{$ENDIF}
begin
  // todo sigmoid simdify
{$IF defined(CPUX64) and defined(USE_AVX2)}
  SSiLUAVX2(N, Src, nil, Dst, nil);
{$ELSE}
  for I := 0 to N - 1 do
    Dst[I] := Single(1.0) / (Single(1.0) + FastExp(-Src[I]));
{$ENDIF}
end;

class procedure TQNNSingleOps.QNNSiluInplace(const X: System.PSingle; const N: Integer);
{$IF not (defined(CPUX64) and defined(USE_AVX2))}
var
  I: Integer;

  Val: Single;
{$ENDIF}
begin
  // todo silu inplace simdify
{$IF defined(CPUX64) and defined(USE_AVX2)}
  SSiLUAVX2(N, X, nil, nil, X);
{$ELSE}
  for I := 0 to N - 1 do
  begin
    Val := X[I];
    X[I] := Val / (Single(1.0) + FastExp(-Val));
  end
{$ENDIF}
end;

class procedure TQNNSingleOps.QNNSilu(const Dst, Src: System.PSingle; const N: Integer);
{$IF not (defined(CPUX64) and defined(USE_AVX2))}
var
  I: Integer;

  Val: Single;
{$ENDIF}
begin
  // todo silu priority simdify
{$IF defined(CPUX64) and defined(USE_AVX2)}
  SSiLUAVX2(N, Src, nil, nil, Dst);
{$ELSE}
  for I := 0 to N - 1 do
  begin
    Val := Src[I];
    Dst[I] := Val / (Single(1.0) + FastExp(-Val));
  end
{$ENDIF}
end;

class procedure TQNNSingleOps.QNNSiluMul(const Gate, Up: System.PSingle; const N: Integer);
{$IF not (defined(CPUX64) and defined(USE_AVX2))}
var
  I: Integer;

  Val: Single;
{$ENDIF}
begin

  // todo siluMul priority simdify
{$IF defined(CPUX64) and defined(USE_AVX2)}
  SSiLUAVX2(N, Gate, Up, nil, Gate);
{$ELSE}
  for I := 0 to N - 1 do
  begin
    Val := Gate[I];
    Gate[I] := (Val / (Single(1.0) + FastExp(-Val))) * Up[I];
  end
{$ENDIF}
end;

class procedure TQNNSingleOps.QNNSoftmax(const X: System.PSingle; const N: Integer);
{$IF not (defined(CPUX64) and defined(USE_AVX2))}
var
  I: LongInt;

  MaxVal, Sum, InvSum: Single;
{$ENDIF}
begin
  // setlength(Q, N);
  // move(X^, Q[0], N*sizeof(single));
  // sSoftMax_avx2(N, pointer(Q));
{$IF defined(CPUX64) and defined(USE_AVX2)}
  SSoftmaxAVX2(N, X);
{$ELSE}
  MaxVal := QNNMax(N, X);
  Sum := 0.0;
  for I := 0 to N - 1 do
  begin
    X[I] := FastExp(X[I] - MaxVal);
    // x[i] := exp(x[i]-max_val);
    Sum := Sum + X[I]
  end;
  InvSum := Single(1.0) / Sum;
  QNNScaleInplace(X, InvSum, N);
{$ENDIF}
  // printCompare(N, x, pointer(Q));
end;

// procedure softmax(const N: longint ;const a:TArray<single>; const offset:longint);
// var
//  i: longint;
//  max_val, sum, inv_sum: Single;
// begin
//  // todo softmax simdify
//  //max_val := x[0];
//  //for i := 1 to N -1 do
//  //    if x[i] > max_val then
//  //        max_val := x[i];
//  max_val := TQNNSingleOps.QNNMax(N, PSingle(a)+offset);
//  sum := 0.0;
//  for i := 0 to N-1 do begin
//      //x[i] := fast_exp(x[i]-max_val);
//      a[i+offset] := exp(a[i+offset]-max_val);
//      sum := sum + a[i+offset]
//  end;
//  inv_sum := single(1.0) / sum;
//  for i := 0 to N-1 do
//      a[i+offset] := a[i+offset] * inv_sum
// end;

class procedure TQNNSingleOps.QNNSoftmaxRows(const X: System.PSingle; const Rows, Cols: Integer);
var
{$IF not defined(USE_MULTITHREADING)}
  I: LongInt;
{$ENDIF}
  // c : longint;
  // row: PSingle;
  // max_val, sum, inv_sum: Single;
  /// /a : TArray<single>;

  Worker: TThreadProcNested;
begin
  Worker := procedure(Idx: IntPtr; Data: Pointer) begin QNNSoftmax(X + Idx * Cols, Cols); end;

  // Pointer(a) := X;
  // todo softmax simdify
{$IFDEF USE_MULTITHREADING}
  Mp.&for(Worker, 0, Rows);
{$ELSE}
  for I := 0 to Rows - 1 do
    // softmax(cols, TArray<Single>(x), i*cols);
    QNNSoftmax(X + I * Cols, Cols);
{$ENDIF}
  // begin
  //  row := x+ i*cols;
  //  max_val := row[0];
  //  for c := 1 to cols -1 do
  //      if row[c] > max_val then
  //          max_val := row[c];
  //  sum := 0.0;
  //  for c := 0 to cols -1 do
  //      begin
  //          row[c] := fast_exp(row[c]-max_val);
  //          sum := sum + row[c]
  //      end;
  //  inv_sum := 1.0 / sum;
  //  for c := 0 to cols -1 do
  //      row[c] := row[c] * inv_sum
  // end
end;

(* ========================================================================
 * Attention Operations
 * ======================================================================== *)

(* SDPA (Scaled dot-product attention): softmax(Q @ K^T / sqrt(d)) @ V.
 * This is the naive implementation that materializes the full seq_q x seq_k
 * attention matrix. Used only for small sequences; the transformer's main
 * attention path uses flash_attention() or the GPU kernel instead. *)
class procedure TQNNSingleOps.QNNAttention(
    const Dst, Q, K, V: System.PSingle;
    const Batch, Heads, SeqQ, SeqK, HeadDim: LongInt;
    const Scale: Single
);
var
  Scores, Qq, Kk, Vv, O: System.PSingle;
  B, H: LongInt;
begin
  // todo QNNAttention SIMDIFY
  if Length(Workspace) < SeqQ * SeqK then
    SetLength(Workspace, SeqQ * SeqK);
  Scores := Pointer(Workspace);
  for B := 0 to Batch - 1 do
    for H := 0 to Heads - 1 do
    begin
      Qq := Q + (B * Heads + H) * SeqQ * HeadDim;
      Kk := K + (B * Heads + H) * SeqK * HeadDim;
      Vv := V + (B * Heads + H) * SeqK * HeadDim;
      O := Dst + (B * Heads + H) * SeqQ * HeadDim;

      CblasGemm(
          CblasRowMajor,
          CblasNoTrans,
          CblasTrans,
          SeqQ,
          SeqK,
          HeadDim,
          Scale,
          Qq,
          HeadDim,
          Kk,
          HeadDim,
          0,
          Scores,
          SeqK
      );
      // if scale<>1 then QNNScale(scores, scores, scale, seq_q*seq_k);

      QNNSoftmaxRows(Scores, SeqQ, SeqK);

      CblasGemm(
          CblasRowMajor,
          CblasNoTrans,
          CblasNoTrans,
          SeqQ,
          HeadDim,
          SeqK,
          1,
          Scores,
          SeqK,
          Vv,
          HeadDim,
          0,
          O,
          HeadDim
      );

    end;
end;

(* ========================================================================
 * Flash Attention - Memory-Efficient Tiled Attention
 *
 * Uses online softmax algorithm to compute attention without materializing
 * the full [seq_q, seq_k] attention matrix. Reduces memory from O(n�) to O(n).
 *
 * Algorithm (for each query position):
 * 1. Initialize: max_score = -inf, sum = 0, output = 0
 * 2. For each key/value block:
 *    - Compute local scores = Q @ K^T * scale
 *    - Update running max and sum with correction factors
 *    - Accumulate weighted values into output
 * 3. Normalize: output /= sum
 *
 * Reference: "FlashAttention: Fast and Memory-Efficient Exact Attention"
 * ======================================================================== *)

(*
 * Flash attention for a single head.
 * Q: [seq_q, head_dim], K: [seq_k, head_dim], V: [seq_k, head_dim]
 * out: [seq_q, head_dim]
 * Uses O(head_dim) working memory per query instead of O(seq_k).
 *)
class procedure TQNNSingleOps.QNNFlashAttentionHead(
    const Dst, Q, K, V: System.PSingle;
    const SeqQ, SeqK, HeadDim: LongInt;
    const Scale: Single
);
var
  I, J: LongInt;
  QRow, ORow, KRow, VRow: System.PSingle;
  MaxScore, SumExp, Score, Correction, Weight, InvSum: Single;
begin
  // todo FlashAttentionHead simdify
  for I := 0 to SeqQ - 1 do
  begin
    QRow := Q + I * HeadDim;
    ORow := Dst + I * HeadDim;
    MaxScore := -1;
    SumExp := 0.0;
    QNNFill(ORow, 0, HeadDim);
    // for d := 0 to head_dim -1 do
    //    o_row[d] := 0.0;
    for J := 0 to SeqK - 1 do
    begin
      KRow := K + J * HeadDim;
      VRow := V + J * HeadDim;
      // score := 0.0;
      // for d := 0 to head_dim -1 do
      //    score := score + (q_row[d] * k_row[d]);
      // score := score * scale;
      Score := Scale * QNNDot(QRow, KRow, HeadDim);
      if Score > MaxScore then
      begin
        Correction := FastExp(MaxScore - Score);
        // correction := exp(max_score-score);
        SumExp := SumExp * Correction + 1.0;
        QNNFusedScaleAdd(ORow, ORow, VRow, Correction, HeadDim);
        // for d := 0 to head_dim -1 do
        //    o_row[d] := o_row[d]*correction + v_row[d];
        MaxScore := Score
      end
      else
      begin
        Weight := FastExp(Score - MaxScore);
        // weight := exp(score-max_score);
        SumExp := SumExp + Weight;
        CblasAxpy(HeadDim, Weight, VRow, 1, ORow, 1);
        // for d := 0 to head_dim -1 do
        //    o_row[d] := o_row[d] + (weight * v_row[d])
      end
    end;
    InvSum := Single(1.0) / SumExp;
    CblasScal(HeadDim, InvSum, ORow, 1);
    // for d := 0 to head_dim -1 do
    //    o_row[d] := o_row[d] * inv_sum
  end
end;

(*
 * Flash attention with BLAS-optimized tiling.
 * Processes queries in tiles for better cache utilization.
 * Uses BLAS for tile-level matrix operations when available.
 *
 * Q: [seq_q, head_dim], K: [seq_k, head_dim], V: [seq_k, head_dim]
 * out: [seq_q, head_dim]
 * tile_scores: scratch buffer of size [q_tile_size, k_tile_size]
 *)
class procedure TQNNSingleOps.QNNFlashAttentionHeadTiled(
    const Dst, Q, K, V: System.PSingle;
    const SeqQ, SeqK, HeadDim: LongInt;
    const Scale: Single;
    const TileScores: System.PSingle;
    const QTileSize, KTileSize: LongInt
);
var
  MaxScores, SumExps, QTile, KTile, VTile, ScoreRow, ORow, OutTile, VRow: System.PSingle;
  I, KStart, KEnd, KLen, QStart, QEnd, QLen, Qi, Ki: LongInt;
  TileMax, OldMax, NewMax, Correction, Weight: Single;
begin
  // todo FlashAttentionHeadTiled simdify
  // cache frendly flash attention head
  if Length(Workspace) < 2 * SeqQ then
    SetLength(Workspace, 2 * SeqQ);
  MaxScores := Pointer(Workspace);
  SumExps := @Workspace[SeqQ];

  // for i := 0 to seq_q -1 do
  //    begin
  //        max_scores[i] := -1;
  //        sum_exps[i] := 0.0
  //    end;

  QNNFill(MaxScores, -1, SeqQ);
  QNNFill(SumExps, 0.0, SeqQ);

  // fillchar(dst^, seq_q*head_dim*sizeof(Single), 0);
  QNNFill(Dst, 0, SeqQ * HeadDim);
  KStart := 0;
  while KStart < SeqK do
  begin
    if (KStart + KTileSize < SeqK) then
      KEnd := KStart + KTileSize
    else
      KEnd := SeqK;
    KLen := KEnd - KStart;
    QStart := 0;
    while QStart < SeqQ do
    begin
      if (QStart + QTileSize < SeqQ) then
        QEnd := QStart + QTileSize
      else
        QEnd := SeqQ;
      QLen := QEnd - QStart;
      QTile := Q + QStart * HeadDim;
      KTile := K + KStart * HeadDim;
      VTile := V + KStart * HeadDim;
      OutTile := Dst + QStart * HeadDim;
      CblasGemm(
          CblasRowMajor,
          CblasNoTrans,
          CblasTrans,
          QLen,
          KLen,
          HeadDim,
          Scale,
          QTile,
          HeadDim,
          KTile,
          HeadDim,
          0.0,
          TileScores,
          KTileSize
      );
      for Qi := 0 to QLen - 1 do
      begin
        I := QStart + Qi;
        ScoreRow := TileScores + Qi * KTileSize;
        ORow := OutTile + Qi * HeadDim;
        TileMax := QNNMax(KLen, ScoreRow);
        // tile_max := score_row[0];
        // for ki := 1 to k_len -1 do
        //    if score_row[ki] > tile_max then
        //        tile_max := score_row[ki];
        OldMax := MaxScores[I];
        if (TileMax > OldMax) then
          NewMax := TileMax
        else
          NewMax := OldMax;
        if OldMax > -1 then
        begin
          Correction := FastExp(OldMax - NewMax);
          // correction := exp(old_max-new_max);
          SumExps[I] := SumExps[I] * Correction;
          CblasScal(HeadDim, Correction, ORow, 1);
          // for d := 0 to head_dim -1 do
          //    o_row[d] := o_row[d] * correction
        end;
        for Ki := 0 to KLen - 1 do
        begin
          Weight := FastExp(ScoreRow[Ki] - NewMax);
          // weight := exp(score_row[ki]-new_max);
          SumExps[I] := SumExps[I] + Weight;
          VRow := VTile + Ki * HeadDim;
          CblasAxpy(HeadDim, Weight, VRow, 1, ORow, 1);
          // for d := 0 to head_dim -1 do
          //    o_row[d] := o_row[d] + (weight * v_row[d])
        end;
        MaxScores[I] := NewMax
      end;
      Inc(QStart, QTileSize)
    end;
    KStart := KStart + KTileSize
  end;
  for I := 0 to SeqQ - 1 do
  begin
    CblasScal(HeadDim, Single(1.0) / SumExps[I], Dst + I * HeadDim, 1);
    // inv_sum := 1.0 / sum_exps[i];
    // o_row := dst+i * head_dim;
    // for d := 0 to head_dim -1 do
    //    o_row[d] := o_row[d] * inv_sum
  end;
end;

(*
 * Flash attention for multi-head attention.
 * Works on [seq, heads*head_dim] layout (same as transformer tensors).
 *
 * Q: [seq_q, heads * head_dim]
 * K: [seq_k, heads * head_dim]
 * V: [seq_k, heads * head_dim]
 * out: [seq_q, heads * head_dim]
 *
 * Memory usage: O(seq_q + tile_size�) instead of O(seq_q * seq_k)
 *)
class procedure TQNNSingleOps.QNNFlashAttention(
    const Dst, Q, K, V: System.PSingle;
    const Heads, SeqQ, SeqK, HeadDim: LongInt;
    const Scale: Single
);
const
  Q_TILE_SIZE = 32;
  K_TILE_SIZE = 64;
var
  H, I, J, Hidden, Off: LongInt;
  QHead, KHead, VHead, OutHead, QContig, KContig, VContig, OutContig: System.PSingle;
  TileScores: array[0..Q_TILE_SIZE * K_TILE_SIZE - 1] of Single;

// scores : PSingle;
begin
  // tile_scores := PSingle(malloc(q_tile_size * k_tile_size * sizeof(float)));
  Hidden := Heads * HeadDim;
  if Length(Workspace) < 2 * HeadDim * (SeqK + SeqQ) then
    SetLength(Workspace, 2 * HeadDim * (SeqK + SeqQ));

  (*
  scores := pointer(workspace);
  for i := 0 to heads-1 do begin
      //qh := Q      + i*head_dim;
      //kh := K      + i*head_dim;
      //vh := V      + i*head_dim;
      //oh := dst    + i*head_dim;
      //sh := scores + i*seq_q*seq_k;

      cblas_gemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                  seq_q, seq_k, head_dim,
                  scale,
                  Q      + i*head_dim, hidden,
                  K      + i*head_dim, hidden,
                  0.0,
                  scores + i*seq_q*seq_k, seq_k);
      QNNSoftmax(scores + i*seq_q*seq_k, seq_q, seq_k);

      cblas_gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                  seq_k, head_dim, seq_q,
                  1.0,
                  scores + i*seq_q*seq_k, seq_k,
                  V      + i*head_dim, hidden,
                  0.0,
                  dst    + i*head_dim, hidden);
  end;
  exit;
*)

  Off := 0;
  QContig := @Workspace[Off];
  Inc(Off, SeqQ * HeadDim);
  KContig := @Workspace[Off];
  Inc(Off, SeqK * HeadDim);
  VContig := @Workspace[Off];
  Inc(Off, SeqK * HeadDim);
  OutContig := @Workspace[Off];

  for H := 0 to Heads - 1 do
  begin
    QHead := Q + H * HeadDim;
    KHead := K + H * HeadDim;
    VHead := V + H * HeadDim;
    OutHead := Dst + H * HeadDim;
    if (SeqQ <= 64) and (SeqK <= 128) then
    begin
      for I := 0 to SeqQ - 1 do
        QNNCopy(QContig + I * HeadDim, QHead + I * Hidden, HeadDim);
      // for d := 0 to head_dim -1 do
      //    Q_contig[i * head_dim+d] := Q_head[i * hidden+d];
      for J := 0 to SeqK - 1 do
      begin
        QNNCopy(KContig + J * HeadDim, KHead + J * Hidden {heads*head_dim}, HeadDim);
        QNNCopy(VContig + J * HeadDim, VHead + J * Hidden {heads*head_dim}, HeadDim);
        // for d := 0 to head_dim -1 do
        //    begin
        //        K_contig[j * head_dim+d] := K_head[j * hidden+d];
        //        V_contig[j * head_dim+d] := V_head[j * hidden+d]
        //    end;
      end;
      QNNFlashAttentionHead(OutContig, QContig, KContig, VContig, SeqQ, SeqK, HeadDim, Scale);
      for I := 0 to SeqQ - 1 do
        QNNCopy(OutHead + I * Hidden, OutContig + I * HeadDim, HeadDim);
      // for d := 0 to head_dim -1 do
      //    out_head[i * hidden+d] := out_contig[i * head_dim+d];
    end
    else
    begin
      for I := 0 to SeqQ - 1 do
        QNNCopy(QContig + I * HeadDim, QHead + I * Hidden, HeadDim);
      // for d := 0 to head_dim -1 do
      //    Q_contig[i * head_dim+d] := Q_head[i * hidden+d];
      for J := 0 to SeqK - 1 do
      begin
        QNNCopy(KContig + J * HeadDim, KHead + J * Hidden {heads*head_dim}, HeadDim);
        QNNCopy(VContig + J * HeadDim, VHead + J * Hidden {heads*head_dim}, HeadDim);
        // for d := 0 to head_dim -1 do
        //    begin
        //        K_contig[j * head_dim+d] := K_head[j * hidden+d];
        //        V_contig[j * head_dim+d] := V_head[j * hidden+d]
        //    end;
      end;
      QNNFlashAttentionHeadTiled(
          OutContig,
          QContig,
          KContig,
          VContig,
          SeqQ,
          SeqK,
          HeadDim,
          Scale,
          @TileScores[0],
          Q_TILE_SIZE,
          K_TILE_SIZE
      );

      for I := 0 to SeqQ - 1 do
        QNNCopy(OutHead + I * Hidden, OutContig + I * HeadDim, HeadDim);
      // for d := 0 to head_dim -1 do
      //    out_head[i * hidden+d] := out_contig[i * head_dim+d];
    end
  end;
end;

(* Apply precomputed RoPE (Rotary Position Embedding) in-place using the
 * split-half convention: dim d pairs with dim d+half for rotation. This is
 * the Flux convention (4-axis, split-half); Z-Image uses consecutive pairs
 * via a separate kernel. RoPE lets the transformer learn relative position
 * from the dot-product structure of Q and K. *)
class procedure TQNNSingleOps.QNNApplyRoPE(const Dst, Freqs: System.PSingle; const Batch, Seq, Heads, HeadDim: LongInt);
var
  Head: System.PSingle;
  HalfDim, B, S, H, I: LongInt;
  CosVal, SinVal: Single;
  X0, X1: Single;
begin
  // todo QNNApply RoPE SIMDIFY
  HalfDim := HeadDim div 2;
  for B := 0 to Batch - 1 do
    for S := 0 to Seq - 1 do
      for H := 0 to Heads - 1 do
      begin
        Head := Dst + ((B * Seq + S) * Heads + H) * HeadDim;
        for I := 0 to HalfDim - 1 do
        begin
          CosVal := Freqs[S * HalfDim * 2 + I * 2];
          SinVal := Freqs[S * HalfDim * 2 + I * 2 + 1];
          X0 := Head[I];
          X1 := Head[I + HalfDim];
          Head[I] := X0 * CosVal - X1 * SinVal;
          Head[I + HalfDim] := X0 * SinVal + X1 * CosVal
        end
      end
end;

class procedure TQNNSingleOps.QNNApplyRoPE2(const Dst, CosIn, SinIn: System.PSingle; const NumHeads, HeadDim: LongInt);
var
  HalfDim, H, I: LongInt;
  Head: System.PSingle;
  X0, X1: Single;
begin
  HalfDim := HeadDim div 2; // todo Priority SIMD
  for H := 0 to NumHeads - 1 do
  begin
    Head := Dst + H * HeadDim;
    for I := 0 to HalfDim - 1 do
    begin
      X0 := Head[I];
      X1 := Head[I + HalfDim];
      Head[I] := X0 * CosIn[I] - X1 * SinIn[I];
      Head[I + HalfDim] := X0 * SinIn[I] + X1 * CosIn[I];
    end;
  end;
end;

class procedure TQNNSingleOps.QNNApplyRoPE3(
    const Dst, CosIn, SinIn: System.PSingle;
    const SeqLen, NumHeads, HeadDim: LongInt
);
var
  HalfDim, S: LongInt;
begin
  HalfDim := HeadDim div 2; // todo Priority SIMD
  for S := 0 to SeqLen - 1 do
    QNNApplyRoPE2(Dst + S * NumHeads * HeadDim, CosIn + S * HalfDim, SinIn + S * HalfDim, NumHeads, HeadDim)
end;

class procedure TQNNSingleOps.QNNApplyRoPEQK(
    const Q, K, CosCache, SinCache: System.PSingle;
    const SeqLen, NumQHeads, NumKvHeads, HeadDim: LongInt
);
var
  HalfDim, S, I, H: LongInt;
  X0, X1, CosVal, SinVal: Double;
  QHead, KHead, CosRow, SinRow: System.PSingle;
begin
  HalfDim := HeadDim div 2;

  (* Apply RoPE to Q *)
  for S := 0 to SeqLen - 1 do
  begin
    CosRow := CosCache + S * HalfDim;
    SinRow := SinCache + S * HalfDim;

    for H := 0 to NumQHeads - 1 do
    begin
      QHead := Q + S * NumQHeads * HeadDim + H * HeadDim;

      for I := 0 to HalfDim - 1 do
      begin
        X0 := QHead[I];
        X1 := QHead[I + HalfDim];
        CosVal := CosRow[I];
        SinVal := SinRow[I];
        QHead[I] := X0 * CosVal - X1 * SinVal;
        QHead[I + HalfDim] := X0 * SinVal + X1 * CosVal;
      end;
    end;
  end;

  (* Apply RoPE to K *)
  for S := 0 to SeqLen - 1 do
  begin
    CosRow := CosCache + S * HalfDim;
    SinRow := SinCache + S * HalfDim;

    for H := 0 to NumKvHeads - 1 do
    begin
      KHead := K + S * NumKvHeads * HeadDim + H * HeadDim;

      for I := 0 to HalfDim - 1 do
      begin
        X0 := KHead[I];
        X1 := KHead[I + HalfDim];
        CosVal := CosRow[I];
        SinVal := SinRow[I];

        KHead[I] := X0 * CosVal - X1 * SinVal;
        KHead[I + HalfDim] := X0 * SinVal + X1 * CosVal;
      end;
    end;
  end;
end;

class procedure TQNNSingleOps.QNNApplyRoPE2D(
    const Dst, CosFreq, SinFreq: System.PSingle;
    const Seq, Heads, HeadDim, Dim: LongInt
);
var
  S, D, H: LongInt;
  CosS, SinS, Vec: System.PSingle;
  CosVal, SinVal, X0, X1: Single;
begin
  // axis_dim;  (* head_dim = 128 = 4 * axis_dim (axis_dim = 32) *)
  for S := 0 to Seq - 1 do
  begin // todo priority SIMD
    CosS := CosFreq + S * HeadDim; (* [128] *)
    SinS := SinFreq + S * HeadDim;

    for H := 0 to Heads - 1 do
    begin
      Vec := Dst + (S * Heads + H) * HeadDim;

      (* Apply rotation to all 128 dims in pairs (0,1), (2,3), ... (126,127) *)
      D := 0;
      while D < HeadDim do
      begin
        CosVal := CosS[D]; (* cos[d] == cos[d+1] due to repeat_interleave *)
        SinVal := SinS[D];
        X0 := Vec[D];
        X1 := Vec[D + 1];
        (* Complex rotation: (x0 + i*x1) * (cos + i*sin) *)
        Vec[D] := X0 * CosVal - X1 * SinVal;
        Vec[D + 1] := X1 * CosVal + X0 * SinVal;
        Inc(D, 2)
      end
    end
  end
end;

class function TQNNSingleOps.QNNTimeStepEmbedding(
    const T: Single;
    const Dim: Integer;
    const MaxPeriod: Single
): TMemoryBlock;
var
  HalfDim, I: LongInt;
  LogMax, Freq, Angle: Single;
  DstPtr: System.PSingle;
begin
  HalfDim := Dim div 2;
  LogMax := Ln(MaxPeriod);
  Result := TMemoryBlock.Create([2, HalfDim], 'QNNTimeStepEmbedding' + TGUID.NewGuid.ToString());
  DstPtr := Result;
  for I := 0 to HalfDim - 1 do
  begin
    (* freq = exp(-log(max_period) * i / half_dim) *)
    Freq := FastExp(-LogMax * (I / HalfDim));
    // freq := exp(-log_max * (i / half_dim));
    Angle := T * Freq;
    DstPtr[I] := Cos(Angle); (* cos part first (flip_sin_to_cos=True) *)
    DstPtr[I + HalfDim] := Sin(Angle); (* sin part second *)
  end
end;

class procedure TQNNSingleOps.QNNMatTriangularFill(
    const Dim: Integer;
    const Dst: System.PSingle;
    const Val: Single;
    const Mask: PLongint
);
var
  R, C: LongInt;

begin
  // assert((dim>0) and (dim*dim<=dst.count) and (length(dst.shape)>1) and (shape[high(shape)], shape[high(shape)-1]) and dst.isAssigned(), 'ERROR : Matrix must be square!');
  // h := shape[high(shape)-1];
  // w := shape[high(shape)];
  Assert(Assigned(Dst), 'ERROR : [dst] must be assigned!.');
  for R := 0 to Dim - 1 do
    for C := 0 to Dim - 1 do
      if (C > R) or (Assigned(Mask) and not Boolean(Mask[C])) then
        Dst[R * Dim + C] := Val

end;

class procedure TQNNSingleOps.QNNMaskFill(
    const Dst: System.PSingle;
    const Mask: PLongint;
    const Val: Single;
    const Rows, Cols: LongInt
);
var
  R, C: LongInt;
begin
  Assert(Assigned(Dst) and Assigned(Mask), 'ERROR : dst and mask must be assigned.');
  for R := 0 to Rows - 1 do
    for C := 0 to Cols - 1 do
      if Boolean(Mask[C]) then
        Dst[R * Cols + C] := Val
end;

// Adaptive Layer Normalization
class procedure TQNNSingleOps.QNNAdaLN(
    const Dst, Src, Bias, Scale: System.PSingle;
    const Seq, Hidden: LongInt;
    const Eps: Single
);
var
  I, J: LongInt;
  InRow, OutRow: System.PSingle;
  Mean, Variance, StdInv, Norm: Single;
begin
  for I := 0 to Seq - 1 do
  begin // todo priority SIMD
    InRow := Src + I * Hidden;
    OutRow := Dst + I * Hidden;
    Mean := QNNMean(Hidden, InRow);
    Variance := QNNVariance(Hidden, InRow, Mean);
    StdInv := 1.0 / Sqrt(Variance + EPSILON);
    for J := 0 to Hidden - 1 do
    begin
      Norm := (InRow[J] - Mean) * StdInv;
      OutRow[J] := (1 + Scale[J]) * Norm + Bias[J]
    end;
  end;

end;

class procedure TQNNSingleOps.QNNLinearBF16OrF32(
    const Dst, Src: System.PSingle;
    const Weight: TMemoryBlock;
    const WeightBF16: TMemoryBlock;
    const Seq, SrcDim, DstDim: Integer
);
begin
  Assert(
      Weight.IsAssigned() or WeightBF16.IsAssigned(),
      'ERROR QNNLINEAR_BF16_OR_F32 : no weight or weight_16 assigned!'
  );
  if WeightBF16.IsAssigned() then
    QNNLinearNoBiasBF16(Dst, Src, WeightBF16, Seq, SrcDim, DstDim)
  else
    QNNLinearNoBias(Dst, Src, Weight, Seq, SrcDim, DstDim);
end;

class procedure TQNNSingleOps.QNNUpSampleNearest(
    const Dst, Src: System.PSingle;
    const Batch, Channels, H, W, ScaleH, ScaleW: LongInt
);
var
  OutH, OutW, B, C, Oh, Ow, Ih, Iw, InIdx, OutIdx: LongInt;
begin
  OutH := H * ScaleH;
  OutW := W * ScaleW;
  for B := 0 to Batch - 1 do
    for C := 0 to Channels - 1 do
      for Oh := 0 to OutH - 1 do
        for Ow := 0 to OutW - 1 do
        begin
          Ih := Oh div ScaleH;
          Iw := Ow div ScaleW;
          InIdx := B * Channels * H * W + C * H * W + Ih * W + Iw;
          OutIdx := B * Channels * OutH * OutW + C * OutH * OutW + Oh * OutW + Ow;
          Dst[OutIdx] := Src[InIdx]
        end
end;

class function TQNNSingleOps.LinearInterpolation(const A, B, T: Single): Single;
begin
  Result := A + T * (B - A)
end;

class function TQNNSingleOps.CubicInterpolation(const B, C, T: Single): Single;
begin
  // result := b + t*(0.5*c - 0.5*a + t*(a -2.5*b + 2*c -0.5*d + t*(1.5*b - 0.5*a - 1.5*c + 0.5*d))) ;

  // result := b + 0.5 * t*(c - a + t*(2.0*a - 5.0*b + 4.0*c - d + t*(3.0*(b - c) + d - a)))
  Result := B + 0.5 * T * (C + T * (-5.0 * B + 4.0 * C + T * (3.0 * (B - C))))

end;

class procedure TQNNSingleOps.QNNUpSample(
    const Dst, Src: System.PSingle;
    const Batch, Channels, H, W: LongInt;
    const ScaleH, ScaleW: Single;
    const Interpolation: TInterpolation
);
var
  OutH, OutW, B, C, Oh, Ow, Ih, Iw, InIdx, OutIdx: LongInt;
  Fx, Fy, P1, P2: Single;
begin
  // [QNNUpSample] simdify
  Assert((ScaleW >= 1) and (ScaleH >= 1), 'ERROR QNNUpSample: Scales must be larger or equal than 1.0');
  Assert(Interpolation in [INearest, ILinear, ICubic], 'ERROR QNNUpSample: Selected interpolation is not implemented!');
  OutH := Ceil(H * ScaleH);
  OutW := Ceil(W * ScaleW);
  if Interpolation = INearest then
  begin
    for B := 0 to Batch - 1 do
      for C := 0 to Channels - 1 do
        for Oh := 0 to OutH - Ceil(ScaleH) do
        begin
          Ih := Trunc(Oh / ScaleH);
          for Ow := 0 to OutW - Ceil(ScaleW) do
          begin
            Iw := Trunc(Ow / ScaleW);
            OutIdx := B * Channels * OutH * OutW + C * OutH * OutW + Oh * OutW + Ow;
            InIdx := B * Channels * H * W + C * H * W + Ih * W + Iw;
            Dst[OutIdx] := Src[InIdx];
          end;
        end;
    Exit
  end;
  if Interpolation = ILinear then
  begin
    for B := 0 to Batch - 1 do
      for C := 0 to Channels - 1 do
        for Oh := 0 to OutH - Ceil(ScaleH) - 1 do
        begin
          Fy := Oh / ScaleH;
          Ih := Trunc(Fy);
          Fy := Frac(Fy);
          for Ow := 0 to OutW - Ceil(ScaleW) - 1 do
          begin
            Fx := Ow / ScaleW;
            Iw := Trunc(Fx);
            Fx := Frac(Fx);
            OutIdx := B * Channels * OutH * OutW + C * OutH * OutW + Oh * OutW + Ow;
            InIdx := B * Channels * H * W + C * H * W + Ih * W + Iw;
            P1 := LinearInterpolation(Src[InIdx], Src[InIdx + 1], Fx);
            P2 := LinearInterpolation(Src[InIdx + W], Src[InIdx + W + 1], Fx);
            Dst[OutIdx] := LinearInterpolation(P1, P2, Fy);
          end;
        end;
    Exit
  end;
  if Interpolation = ICubic then
  begin
    for B := 0 to Batch - 1 do
      for C := 0 to Channels - 1 do
        for Oh := 0 to OutH - Ceil(ScaleH) - 1 do
        begin
          Fy := Oh / ScaleH;
          Ih := Trunc(Fy);
          Fy := Frac(Fy);
          for Ow := 0 to OutW - Ceil(ScaleW) - 1 do
          begin
            Fx := Ow / ScaleW;
            Iw := Trunc(Fx);
            Fx := Frac(Fx);
            OutIdx := B * Channels * OutH * OutW + C * OutH * OutW + Oh * OutW + Ow;
            InIdx := B * Channels * H * W + C * H * W + Ih * W + Iw;
            P1 := CubicInterpolation(Src[InIdx], Src[InIdx + 1], Fx);
            P2 := CubicInterpolation(Src[InIdx + W], Src[InIdx + W + 1], Fx);
            Dst[OutIdx] := CubicInterpolation(P1, P2, Fy);
          end;
          // fy := oh/scale_h;
          // ih := trunc(fy);
          // fy := frac(fy);
          // if ih<1 then begin
          //  ih:=1 ; fy := -1+fy
          // end else
          // if ih> H-2 then begin
          //  ih := H-2 ; fy := 1
          // end else
          // if ih> H-3 then begin
          //  ih := H-3 ; fy := 1+fy
          // end;
          // for ow := 0 to outW - ceil(scale_w)-1 do
          //  begin
          //    fx := ow/scale_w;
          //    iw := trunc(fx);
          //    fx := frac(fx);
          //    out_idx := b * channels * outH * outW + c * outH * outW+oh * outW+ow;
          //    if iw<1 then begin
          //      iw:=1 ; fx := -1+fx
          //    end else
          //    if iw> W-2 then begin
          //      iw := W-2 ; fx := 1
          //    end else
          //    if iw> W-3 then begin
          //      iw := W-3 ; fx := 1+fx
          //    end;
          //    in_idx := b * channels * H * W+c * H * W+ih * W+iw;
          //    p1 := cubicInterpolation(src[in_idx-W-1], src[in_idx-W], src[in_idx-W+1], src[in_idx-W+2], fx);
          //    p2 := cubicInterpolation(src[in_idx-1], src[in_idx], src[in_idx+1], src[in_idx+2], fx);
          //    p3 := cubicInterpolation(src[in_idx+W-1], src[in_idx+W], src[in_idx+W+1], src[in_idx+W+2], fx);
          //    p4 := cubicInterpolation(src[in_idx+2*W-1], src[in_idx+2*W], src[in_idx+2*W+1], src[in_idx+2*W+2], fx);
          //    dst[out_idx] := cubicInterpolation(p1, p2, p3, p4, fy);
          //  end;
        end;
    Exit
  end;

end;

(* Convert spatial latent to patch tokens for the diffusion transformer.
 * Groups each ps x ps spatial block into a single token vector:
 * [batch, channels, H, W] -> [batch, channels*ps*ps, H/ps, W/ps].
 * The transformer operates on these patch tokens, not individual spatial
 * positions, reducing sequence length by ps*ps (4x for ps=2). *)
class procedure TQNNSingleOps.QNNPatchify(
    const Dst, Src: System.PSingle;
    const Batch, Channels, H, W, PatchSize: LongInt
);
var
  OutH, OutW, {out_ch, } B, C, Ph, Pw, Pi, Pj, {ih, iw, } InIdx, {out_c, } OutIdx: LongInt;
begin
  OutH := H div PatchSize;
  OutW := W div PatchSize;
  for B := 0 to Batch - 1 do
    for C := 0 to Channels - 1 do
      for Pi := 0 to PatchSize - 1 do
        for Pj := 0 to PatchSize - 1 do
          for Ph := 0 to OutH - 1 do
            for Pw := 0 to OutW - 1 do
            begin
              // ih := ph * patch_size+pi;
              // iw := pw * patch_size+pj;
              InIdx :=
                  B * Channels * OutH * PatchSize * OutW * PatchSize
                      + C * OutH * PatchSize * OutW * PatchSize
                      + Ph * PatchSize * OutW * PatchSize
                      + Pi * OutW * PatchSize
                      + Pw * PatchSize
                      + Pj;
              OutIdx :=
                  B * Channels * PatchSize * PatchSize * OutH * OutW
                      + C * PatchSize * PatchSize * OutH * OutW
                      + Pi * PatchSize * OutH * OutW
                      + Pj * OutH * OutW
                      + Ph * OutW
                      + Pw;
              // [batch, channels, outH, patch_size, outW, patch_size]
              // [batch, channels, patch_size ,patch_size, outH ,outW]
              // transpose : 0, 1, 3, 5, 2, 4
              Dst[OutIdx] := Src[InIdx]
            end;
  // [b*channels*H*W + c*H*W + ph*patch_size*W + pw*patch_size + pi*W + pj]
end;

class procedure TQNNSingleOps.QNNUnpatchify(
    const Dst, Src: System.PSingle;
    const Batch, Channels, H, W, PatchSize: LongInt
);
var {in_ch, outH, outW, }
  B, C, Ph, Pw, Pi, Pj, {in_c, } InIdx, {oh, ow, } OutIdx: LongInt;
begin
  // in_ch := channels * patch_size * patch_size;
  // outH := H * patch_size;
  // outW := W * patch_size;
  for B := 0 to Batch - 1 do
    for C := 0 to Channels - 1 do
      for Ph := 0 to H - 1 do
        for Pw := 0 to W - 1 do
          for Pi := 0 to PatchSize - 1 do
            for Pj := 0 to PatchSize - 1 do
            begin
              InIdx :=
                  B * Channels * PatchSize * PatchSize * H * W
                      + C * PatchSize * PatchSize * H * W
                      + Pi * PatchSize * H * W
                      + Pj * H * W
                      + Ph * W
                      + Pw;
              // [batch, channels, patch_size, patch_size ,H, W]
              OutIdx :=
                  B * Channels * H * PatchSize * W * PatchSize
                      + C * H * PatchSize * W * PatchSize
                      + Ph * PatchSize * W * PatchSize
                      + Pi * W * PatchSize
                      + Pw * PatchSize
                      + Pj;
              // [batch, channels, H, patch_size, W, patch_size]
              // transpose [0, 1, 4, 2, 5, 3]
              Dst[OutIdx] := Src[InIdx]
            end
end;
(*
class procedure TQNNSingleOPS.QNNPermut(
  const dst, src: PSingle;
  const axisPerm: TArray<byte>;
  const inShape: TArray<int64>; outShape: TArray<int64>;
  inIdx: TArray<int64>; outIdx: TArray<int64>; const lvl: int64;
  inStrides:TArray<int64> = nil; outStrides:TArray<int64> = nil);
var
  i: int64;
  outLinear, inLinear: int64;
begin
  { Iterate all values for this output axis }

  if not assigned(inStrides)  then inStrides := getStrides(inShape);
  if not assigned(outStrides) then outStrides := getStrides(outShape);
  if not assigned(outShape) then setLength(outShape, length(inShape));
  for i:=0 to high(inShape) do
    outShape[i]:= inShape[axisPerm[i]];
  if not assigned(inIdx)  then setLength(inIdx, length(inShape));
  if not assigned(outIdx) then setLength(outIdx, length(outShape));

  if lvl<high(outshape) then begin
    for i := 0 to OutShape[lvl]-1 do
      begin
        OutIdx[lvl] := i;
        InIdx[AxisPerm[lvl]] := i;
        QNNPermut(dst, src,  axisPerm, inShape, OutShape, InIdx, outIdx, lvl+1, inStrides, outStrides);
      end
  end
    else begin
      for i := 0 to OutShape[lvl]-1 do
        begin
          OutIdx[lvl] := i;
          InIdx[AxisPerm[lvl]] := i;
          inLinear := getIndex(InIdx, InStrides);
          outLinear := getIndex(OutIdx, OutStrides);
          dst[outLinear] := src[inLinear];
        end;

      //inLinear  := StridesToLastIndex(InIdx , InStrides);
      //outLinear := StridesToLastIndex(OutIdx, OutStrides);
      //if lvl>1 then lastStride := inStrides[lvl-1] else lastStride := 1;
      //for i := 0 to OutShape[lvl]-1 do
      //  begin
      //    dst[outLinear+i] := src[inLinear + i*lastStride];
      //  end;
    end;
end;


procedure QNNPermutRec(
  const dst, src: PSingle;
  const axisPerm: TArray<byte>;
  const inShape: TArray<int64>; const outShape: TArray<int64>;
  inIdx: TArray<int64>; outIdx: TArray<int64>; const lvl: int64;
  const inStrides: TArray<int64>; const outStrides: TArray<int64>;
  const inBase, outBase: int64);
var
  i, inLinear, outLinear: int64;
begin
  if lvl < high(outShape) then begin
    for i := 0 to outShape[lvl]-1 do begin
      outIdx[lvl] := i;
      inIdx[axisPerm[lvl]] := i;
      QNNPermutRec(dst, src, axisPerm, inShape, outShape,
                   inIdx, outIdx, lvl+1, inStrides, outStrides,
                   inBase  + i * inStrides[axisPerm[lvl]],
                   outBase + i * outStrides[lvl]);
    end;
  end
  else begin
    inLinear  := inBase;
    outLinear := outBase;
    for i := 0 to outShape[lvl]-1 do begin
      dst[outLinear] := src[inLinear];
      Inc(outLinear, outStrides[lvl]);
      Inc(inLinear,  inStrides[axisPerm[lvl]]);
    end;
  end;
end;


class procedure TQNNSingleOPS.QNNPermut(
  const dst, src: PSingle;
  const axisPerm: TArray<byte>;
  const inShape: TArray<int64>; outShape: TArray<int64>;
  inIdx: TArray<int64> = nil; outIdx: TArray<int64> = nil; const lvl: int64 = 0;
  inStrides: TArray<int64> = nil; outStrides: TArray<int64> = nil);
var
  i: int64;
begin
  if not assigned(inStrides)  then inStrides := getStrides(inShape);
  if not assigned(outShape) then outShape := Copy(inShape);
  for i := 0 to high(inShape) do
    outShape[i] := inShape[axisPerm[i]];
  if not assigned(outStrides) then outStrides := getStrides(outShape);
  if not assigned(inIdx)  then setLength(inIdx, length(inShape));
  if not assigned(outIdx) then setLength(outIdx, length(outShape));

  QNNPermutRec(dst, src, axisPerm, inShape, outShape,
               inIdx, outIdx, 0, inStrides, outStrides, 0, 0);
end;
*)

class procedure TQNNSingleOps.QNNPermut(
    const Dst, Src: System.PSingle;
    const AxisPerm: TArray<Byte>;
    const InShape: TArray<Int64>;
    var OutShape: TArray<Int64>
);
var
  I, Lvl, {ndims, } Total: Int64;
  OutLinear, InLinear: Int64;
  OShape, InIdx, OutIdx, InStrides, OutStrides: TArray<Int64>;

begin
  InStrides := GetStrides(InShape);
  OShape := Copy(InShape);
  for I := 0 to High(InShape) do
    OShape[I] := InShape[AxisPerm[I]];
  OutStrides := GetStrides(OShape);
  SetLength(InIdx, Length(InShape)); // setLength will initiate the array to zero
  SetLength(OutIdx, Length(OShape)); // setLength will initiate the array to zero

  // ndims := length(oShape);
  Total := Product(OShape);
  // for i := 0 to ndims-1 do total := total * oShape[i];

  // for i := 0 to ndims-1 do begin inIdx[i] := 0; outIdx[i] := 0; end;
  OutLinear := 0;
  InLinear := 0;

  for I := 0 to Total - 1 do
  begin
    Dst[OutLinear] := Src[InLinear];

    Lvl := High(OShape);
    while (Lvl >= 0) and (OutIdx[Lvl] + 1 >= OShape[Lvl]) do
    begin
      Dec(OutLinear, (OShape[Lvl] - 1) * OutStrides[Lvl]);
      Dec(InLinear, (OShape[Lvl] - 1) * InStrides[AxisPerm[Lvl]]);
      OutIdx[Lvl] := 0;
      Dec(Lvl);
    end;
    if Lvl >= 0 then
    begin
      Inc(OutIdx[Lvl]);
      // inIdx[axisPerm[lvl]] := outIdx[lvl];
      Inc(OutLinear, OutStrides[Lvl]);
      Inc(InLinear, InStrides[AxisPerm[Lvl]]);
    end;
  end;
  if Assigned(OutShape) then
    OutShape := OShape
end;

// ============================================================================
// QNNReduce DRAFT (commented, not active).
//
// numpy-style tensor reduction over one or more axes.
//   - outShape = inShape with reduceAxes removed; remaining dims keep original
//     order (numpy behaviour).
//   - For each output element it folds the whole reduced volume via the binary
//     combiner reduceCb:  acc = src[first]; acc = reduceCb(acc, src[next]) ...
//   - divideByCount=True => mean: divide the fold result by the total number of
//     reduced elements (product of the reduced axis sizes). Only meaningful when
//     reduceCb is a sum-combiner; caller's responsibility otherwise.
//   - dst must be pre-allocated to product(outShape) elements (scalar output => 1).
//   - The reduced shape is returned in the `out outShape` parameter.
//
// Validation:
//   - each axis in reduceAxes must satisfy  0 <= axis < length(inShape)
//   - no duplicate axes
//   - empty reduceAxes => identity (copies src to dst); reducing ALL axes yields
//     a single scalar element.
//
// Callback type (declare once near the class):
//   TReduceFunc = function(const a, b: Single): Single;
// ============================================================================
{
class procedure TQNNSingleOPS.QNNReduce(
  const dst, src: PSingle;
  const inShape: TArray<int64>;
  const reduceAxes: TArray<int64>;
  const reduceCb: TReduceFunc;
  const divideByCount: boolean;
  out outShape: TArray<int64>);
var
  ndims, outNdim, na, i, j, k, d, oLin, rem, v, base, extra, t, reduceCount, st: int64;
  inStrides, outStrides, keepAxes, rIdx: TArray<int64>;
  toReduce: TArray<boolean>;
  acc: Single;
begin
  ndims := length(inShape);
  na    := length(reduceAxes);

  // validate & mark reduced axes
  setLength(toReduce, ndims);
  for i := 0 to ndims-1 do toReduce[i] := False;
  for j := 0 to na-1 do begin
    d := reduceAxes[j];
    if (d < 0) or (d >= ndims) then
      raise Exception.Create(Format('QNNReduce: axis %d out of range [0..%d]', [d, ndims-1]));
    if toReduce[d] then
      raise Exception.Create(Format('QNNReduce: duplicate axis %d', [d]));
    toReduce[d] := True;
  end;

  // build keepAxes (original order) and outShape
  outNdim := 0;
  for i := 0 to ndims-1 do if not toReduce[i] then Inc(outNdim);
  setLength(keepAxes, outNdim);
  setLength(outShape, outNdim);
  j := 0;
  for i := 0 to ndims-1 do
    if not toReduce[i] then begin
      keepAxes[j] := i;
      outShape[j] := inShape[i];
      Inc(j);
    end;

  inStrides  := getStrides(inShape);
  outStrides := getStrides(outShape);

  // total number of elements removed by the reduction
  reduceCount := 1;
  for j := 0 to na-1 do reduceCount := reduceCount * inShape[reduceAxes[j]];

  setLength(rIdx, na);
  for oLin := 0 to product(outShape)-1 do begin
    // decode output linear index into kept-dim indices -> input base offset
    rem  := oLin;
    base := 0;
    for j := 0 to outNdim-1 do begin
      v := rem div outStrides[j];
      rem := rem mod outStrides[j];
      Inc(base, v * inStrides[keepAxes[j]]);
    end;

    // fold: start from first reduced combination, then combine the rest
    acc := src[base];
    if na = 1 then begin
      st := inStrides[reduceAxes[0]];
      for t := 1 to reduceCount-1 do
        acc := reduceCb(acc, src[base + t*st]);
    end
    else begin
      for j := 0 to na-1 do rIdx[j] := 0;
      for t := 1 to reduceCount-1 do begin
        // advance odometer over the reduced axes
        j := na-1;
        while (j >= 0) and (rIdx[j] + 1 >= inShape[reduceAxes[j]]) do begin
          rIdx[j] := 0;
          Dec(j);
        end;
        Inc(rIdx[j]);
        extra := 0;
        for k := 0 to na-1 do
          Inc(extra, rIdx[k] * inStrides[reduceAxes[k]]);
        acc := reduceCb(acc, src[base + extra]);
      end;
    end;

    if divideByCount then acc := acc / reduceCount;
    dst[oLin] := acc;
  end;
end;
}

// var
//  i, newLvl, dst_stride: int64;
//  src_start, dst_start:PSingle;
//  newStrides, oldStrides: Tarray<int64>;
// begin
{  if not assigned(newShape) then begin
    setLength(newShape, length(oldShape));
    setLength(indicies, length(oldShape));
    setLength(newIndicies, length(oldShape));
    for i := 0 to High(axis) do
      newShape[axis[i]] := oldShape[i];
  end;
  newStrides := getStrides(newShape);
  oldStrides := getStrides(oldShape);
  if lvl < high(oldShape) then
    for i := 0 to oldShape[lvl] - 1 do begin
      indicies[lvl] := i;
      newIndicies[axis[lvl]] := i;
      QNNPermute(dst, src, axis, oldShape, newShape, Indicies, newIndicies, lvl + 1)
    end
  else begin// lvl == high(FShape)
    newLvl := axis[lvl];

    indicies[lvl] := 0;
    src_start := src + getIndex(indicies, oldStrides);

    newIndicies[newLvl] := 0;
    i := getIndex(newIndicies, newStrides);
    dst_start := dst + i;

    newIndicies[newLvl] := oldShape[lvl];
    dst_stride := (getIndex(newIndicies, newStrides) - i) div oldShape[lvl];

    for i := 0 to oldShape[lvl] - 1 do begin
      //indicies[lvl] := i;
      //newIndicies[newLvl] := i;
      //dst.Data[dst.getIndex(newIndicies)] := src.Data[src.getIndex(indicies)];
     dst_start[i*dst_stride] := src_start[i];
    end;
  end;
}
// end;

class procedure TQNNSingleOps.QNNCopy(const Dst, Src: System.PSingle; const N: Integer);
begin
  Move(Src^, Dst^, N * SizeOf(Single))
end;

class procedure TQNNSingleOps.QNNCopyStrided(
    const Dst: System.PSingle;
    const DstStride: Integer;
    const Src: System.PSingle;
    const SrcStride: Integer;
    const N: Integer
);
var
  I: Integer;
begin
  if DstStride * SrcStride = 1 then
    Move(Src^, Dst^, N * SizeOf(Single))
  else
    for I := 0 to N - 1 do
      Dst[I * DstStride] := Src[I * SrcStride]
end;

class procedure TQNNSingleOps.QNNFill(
    const Dst: System.PSingle;
    const Val: Single;
    const N: Integer;
    const Stride: Integer
);
var
  I: Integer;
begin
  if (Val = 0) and (Stride = 1) then
    FillChar(Dst^, N * SizeOf(Single), 0)
  else
    for I := 0 to N - 1 do
      Dst[I * Stride] := Val
end;

class procedure TQNNSingleOps.QNNBroadcast(
    const Dst: Pointer;
    const Src;
    const SrcSize, N: Integer;
    const Stride: Integer
);
var
  I: Integer;
begin
  for I := 0 to N - 1 do
    Move(Src, (PByte(Dst) + I * Stride * SrcSize)^, SrcSize)
end;

const
{$IF defined(MSWINDOWS)}
  BLASLIB = 'openblas.dll';
  BLASLIB64 = 'openblas_64.dll';
{$ELSEIF defined(LINUX)}
  BLASLIB = 'openblas.so';
  BLASLIB64 = 'openblas64.so';
{$ELSEIF defined(MACOS) or defined(DARWIN)}
  BLASLIB = 'libopenblas.dylib';
  BLASLIB64 = 'libopenblas_64.dylib';
{$ENDIF}

(*
procedure s2b(const a:TArray<single>; const result : TArray<byte>);
var
  i: Integer;
  v : single;
begin
  assert(length(a)=length(result));
  //result := nil;
  //setLength(result, length(a));
  for i:= 0 to high(a) do begin
    if a[i]>1 then v:=1 else
    if a[i]<0 then v:=0 else v := a[i];
    result[i] := round(v*255)
  end;
end;

procedure b2s(const a:TArray<byte>; const result : TArray<single>);
var
  i: Integer;
begin
  assert(length(a)=length(result));
  //result := nil;
  //setLength(result, length(a));
  for i:= 0 to high(a) do
    result[i] := a[i]/255
end;

const scale = 40;
  WW = 11;
  HH = 10;
var
  im1, om1:  TArray<byte>;
  im2, om2 : TArray<single>;
  i : longint;
*)

class procedure TQNNSingleOps.PrintStat(const Src: System.PSingle; const N: LongInt);
const
  DECIMALS = 3;
var
  Mi, Ma, Ami, Ama, Mean, StdDev: Single;
  Argma, Argmi: LongInt;
begin
  Mi := QNNMin(N, Src, @Argmi);
  Ma := QNNMax(N, Src, @Argma);
  Ami := QNNMinAbs(N, Src);
  Ama := QNNMaxAbs(N, Src);
  Mean := QNNMean(N, Src);
  StdDev := Sqrt(QNNVariance(N, Src, Mean));
  WriteLn(
      '[',
      N,
      ']',
      'Min :',
      Mi:
          1:
          DECIMALS,
          '@',
          Argmi,
          ', Max:',
          Ma:
          1:
          DECIMALS,
          '@',
          Argma,
          ', minAbs:',
          Ami:
          1:
          DECIMALS,
          ', maxAbs:',
          Ama:
          1:
          DECIMALS,
          ', mean:',
          Mean:
          1:
          DECIMALS,
          ', stddev:',
          StdDev: 1: DECIMALS
  );
end;

procedure PrintStat(const Src: TMemoryBlock);
begin
  case Src.DataType of
    dtF32: TQNNSingleOps.PrintStat(Src, Src.Count);
  else
    Assert(False, 'printStat : datatype not implemented!')
  end;
end;

class procedure TQNNSingleOps.PrintCompare(
    const N: LongInt;
    const Src1, Src2: System.PSingle;
    const IsSumSqrDiff: Boolean = False
);
var
  Md, Out1, Out2: Single;
  IsDiff: Boolean;
const
  L_EPSILON = 3e-3;
begin
  Assert(Assigned(Src1) and Assigned(Src2), 'ERROR : src1 and src2 must be assigned');
  WriteLn('');
  PrintStat(Src1, N);
  PrintStat(Src2, N);
  if IsSumSqrDiff then
  begin
    Md := QNNSqrDistance(N, Src1, Src2);
    IsDiff := Md > L_EPSILON;
    WriteLn('SqrDistance :', Md: 1: 5);
    if IsDiff then
      ReadLn;
  end
  else
  begin
    Md := QNNMaxAbsDiff2(N, Src1, Src2, Out1, Out2);
    IsDiff := Md > L_EPSILON;
    WriteLn('MaxAbsDiff :', Md: 1: 5, ' max src1 :', Out1: 1: 6, ' max src2:', Out2: 1: 6);
    if IsDiff then
      ReadLn;
  end;
end;

{
procedure writeTensor(const buf:TMemoryBlock);
var f: file; rt:integer;
begin
  assignFile(f, 'c:\development\tensor.1');
  Rewrite(f, sizeof(single));
  BlockWrite(f, PSingle(buf)^, buf.count(), rt);
  closefile(f);
  assert(rt = buf.Count(), 'Unmatch blocksize write');
end;

function readTensor(): TMemoryBlock;
var f:File; rd:integer;
begin
  assignFile(f, 'c:\development\tensor');
  reset(f, sizeof(single));
  result := TMemoryBlock.Create(fileSize(f), 'readTensor ' + TGUID.NewGuid.ToString() );
  BlockRead(f, PSingle(result)^, result.count, rd);
  CloseFile(f);
  assert(rd = result.Count(), 'Unmatch blcoksize read');
end;

function readArray: TArray<integer>;
var f:File; rd:integer;
begin
  assignFile(f, 'c:\development\tensor');
  reset(f, sizeof(integer));
  setLength(result, fileSize(f));
  BlockRead(f, PInteger(result)^, length(result), rd);
  CloseFile(f);
  assert(rd = length(result), 'Unmatch blcoksize read');
end;
}

procedure CompareArray(const A, B: TArray<Integer>);
var
  I: LongInt;
begin
  Assert(High(A) = High(B), 'unmatched array length');
  for I := 0 to High(A) - 1 do
    Assert(A[I] = B[I], ' Unmatched element at ' + IntToStr(I))
end;

procedure CompareArray(const A, B: PInteger; const N: LongInt);
var
  I: LongInt;
begin
  for I := 0 to N - 1 do
    Assert(A[I] = B[I], ' Unmatched element at ' + IntToStr(I))
end;

procedure CompareArray(const A, B: System.PSingle; const N: LongInt);
var
  I: LongInt;
begin
  for I := 0 to N - 1 do
    Assert(A[I] = B[I], ' Unmatched element at ' + IntToStr(I))
end;

// procedure __imp_prn(const str: pchar);
// begin
//  printf(str)
// end;

procedure PrintArray(const Src: System.PSingle; const N: LongInt);
var
  I: LongInt;
begin
  Write('[');
  if N > 0 then
  begin
    Write(Src[0]: 1: 4);
    for I := 1 to N - 1 do
      Write(', ', Src[I]: 1: 4)

  end;
  WriteLn(']')
end;

procedure PrintArray(const Src: PLongint; const N: LongInt);
var
  I: LongInt;
begin
  Write('[');
  if N > 0 then
  begin
    Write(Src[0]);
    for I := 1 to N - 1 do
      Write(', ', Src[I])

  end;
  WriteLn(']')
end;

procedure PrintArray(const Src: PInt64; const N: LongInt);
var
  I: LongInt;
begin
  Write('[');
  if N > 0 then
  begin
    Write(Src[0]);
    for I := 1 to N - 1 do
      Write(', ', Src[I])

  end;
  WriteLn(']')
end;

procedure RandArray(const N: LongInt; const Dst: System.PSingle; const AMin: Single = -5.0; AMax: Single = 5.0);
var
  I: LongInt;
begin
  for I := 0 to N - 1 do
    Dst[I] := AMin + Random() * (AMax - AMin);
end;

procedure WriteTransposeRec(
    Dim: Int64;
    const OutShape: TArray<Int64>;
    const OutStrides: TArray<Int64>; { not strictly needed if we compute out linear index directly }
    const InStrides: TArray<Int64>;
    const AxisPerm: TArray<Byte>; { out axis k gets in axis AxisPerm[k] }
    const InShape: TArray<Int64>;
    var OutIdx: TArray<Int64>;
    var InIdx: TArray<Int64>;
    const Src, Dst: System.PSingle
);
var
  I: Int64;
  OutLinear, InLinear: Int64;
begin
  { Iterate all values for this output axis }
  if Dim < High(OutShape) then
  begin
    for I := 0 to OutShape[Dim] - 1 do
    begin
      OutIdx[Dim] := I;
      InIdx[AxisPerm[Dim]] := I;
      WriteTransposeRec(Dim + 1, OutShape, OutStrides, InStrides, AxisPerm, InShape, OutIdx, InIdx, Src, Dst);
    end
  end
  else
  begin
    for I := 0 to OutShape[Dim] - 1 do
    begin
      OutIdx[Dim] := I;
      InIdx[AxisPerm[Dim]] := I;
      InLinear := GetIndex(InIdx, InStrides);
      OutLinear := GetIndex(OutIdx, OutStrides);
      Dst[OutLinear] := Src[InLinear];
    end;

    // inLinear  := StridesToLastIndex(InIdx , InStrides);
    // outLinear := StridesToLastIndex(OutIdx, OutStrides);
    // if dim>1 then lastStride := inStrides[dim-1] else lastStride := 1;
    // for i := 0 to OutShape[dim]-1 do
    //  begin
    //    dst[outLinear+i] := src[inLinear + i*lastStride];
    //  end;
  end;
end;

procedure AltPermut(const Dst, Src: System.PSingle; const InShape: TArray<Int64>; const AxisPerm: TArray<Byte>);
var
  Rank, K: Int64;
  OutShape, InStrides, OutStrides, OutIdx, InIdx: TArray<Int64>;
begin
  Rank := Length(AxisPerm);

  { Build output shape: out shape on axis k is in shape on axis AxisPerm[k] }
  SetLength(OutShape, Rank);
  for K := 0 to Rank - 1 do
    OutShape[K] := InShape[AxisPerm[K]];

  { Strides for linearization }
  InStrides := GetStrides(InShape);
  OutStrides := GetStrides(OutShape);

  { Allocate index arrays }
  SetLength(OutIdx, Rank);
  SetLength(InIdx, Rank);

  { Start recursion to enumerate all output index tuples }
  WriteTransposeRec(0, OutShape, OutStrides, InStrides, AxisPerm, InShape, OutIdx, InIdx, Src, Dst);

end;

initialization
  TQNNSingleOps.CblasSaxpy := TQNNSingleOps.QaxpyStrided;
  TQNNSingleOps.CblasSdot := TQNNSingleOps.QdotStrided;
  TQNNSingleOps.CblasSscal := TQNNSingleOps.Qscale;
  TQNNSingleOps.CblasSgemm := TQNNSingleOps.Qgemm;
  TQNNSingleOps.CblasSasum := TQNNSingleOps.Qasum;

  HBLASLIB := 0;

{$IF defined(MACOS) or defined(darwin) or defined(USE_STATIC_LINK)}
  TQNNSingleOps.CblasSgemm := CblasSgemm;
  TQNNSingleOps.CblasSaxpy := CblasSaxpy;
  TQNNSingleOps.CblasSdot := CblasSdot;
  TQNNSingleOps.CblasSasum := CblasSasum;
  TQNNSingleOps.CblasSscal := CblasSscal;
  TQNNSingleOps.IsUsingBlas := True;
{$ELSE}
  {$IFNDEF QNN_NATIVE_ONLY}
  HBLASLIB := LoadLibrary(Blaslib);
  if HBLASLIB = 0 then
    HBLASLIB := LoadLibrary('lib' + Blaslib);
  if HBLASLIB = 0 then
    HBLASLIB := LoadLibrary('lib' + Blaslib + '.0');

  if HBLASLIB = 0 then
    HBLASLIB := LoadLibrary(Blaslib64);
  if HBLASLIB = 0 then
    HBLASLIB := LoadLibrary('lib' + Blaslib64);
  if HBLASLIB = 0 then
    HBLASLIB := LoadLibrary('lib' + Blaslib64 + '.0');

  // Keep the Pascal fallbacks if the detected library is incomplete.
  if (HBLASLIB <> 0)
      and ((GetProcAddress(HBLASLIB, 'cblas_sgemm') = nil)
          or (GetProcAddress(HBLASLIB, 'cblas_saxpy') = nil)
          or (GetProcAddress(HBLASLIB, 'cblas_sdot') = nil)
          or (GetProcAddress(HBLASLIB, 'cblas_sasum') = nil)
          or (GetProcAddress(HBLASLIB, 'cblas_sscal') = nil)) then
  begin
    FreeLibrary(HBLASLIB);
    HBLASLIB := 0;
  end;
  if (HBLASLIB > 0) then
  begin
    TQNNSingleOps.CblasSgemm := GetProcAddress(HBLASLIB, 'cblas_sgemm');
    TQNNSingleOps.CblasSaxpy := GetProcAddress(HBLASLIB, 'cblas_saxpy');
    TQNNSingleOps.CblasSdot := GetProcAddress(HBLASLIB, 'cblas_sdot');
    TQNNSingleOps.CblasSasum := GetProcAddress(HBLASLIB, 'cblas_sasum');
    TQNNSingleOps.CblasSscal := GetProcAddress(HBLASLIB, 'cblas_sscal');
    TQNNSingleOps.CblasSbgemm := GetProcAddress(HBLASLIB, 'cblas_sbgemm');
    TQNNSingleOps.OpenblasSetNumThreads := GetProcAddress(HBLASLIB, 'openblas_set_num_threads');
    TQNNSingleOps.OpenblasGetNumThreads := GetProcAddress(HBLASLIB, 'openblas_get_num_threads');
    TQNNSingleOps.OpenblasGetNumProcs := GetProcAddress(HBLASLIB, 'openblas_get_num_procs');
    TQNNSingleOps.OpenblasGetConfig := GetProcAddress(HBLASLIB, 'openblas_get_config');
    TQNNSingleOps.OpenblasGetCorename := GetProcAddress(HBLASLIB, 'openblas_get_corename');
  end;
  TQNNSingleOps.IsUsingBlas := HBLASLIB <> 0;
  if IsConsole and (HBLASLIB <> 0) and Assigned(TQNNSingleOps.OpenblasGetConfig) then
    WriteLn('Using [', AnsiString(TQNNSingleOps.OpenblasGetConfig()), ']');
  {$ENDIF}
{$ENDIF}
  Randomize;

finalization

  if HBLASLIB > 0 then
  begin
    FreeLibrary(HBLASLIB);
    TQNNSingleOps.CblasSgemm := nil;
    TQNNSingleOps.CblasSaxpy := nil;
    TQNNSingleOps.CblasSdot := nil;
    TQNNSingleOps.CblasSasum := nil;
    TQNNSingleOps.CblasSscal := nil;
    TQNNSingleOps.CblasSbgemm := nil;
    TQNNSingleOps.OpenblasGetConfig := nil;
    TQNNSingleOps.OpenblasGetCorename := nil;
  end

end.
