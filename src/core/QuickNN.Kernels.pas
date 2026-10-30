unit QuickNN.Kernels;

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
  System.SysUtils, System.Classes, System.Math, System.TypInfo
{$IF defined(MSWINDOWS)}
    , Winapi.Windows
{$ELSEIF defined(DARWIN)}
{$ENDIF}
{$IFDEF USE_MULTITHREADING}
    , QuickNN.Threading
{$ENDIF}
    , QuickNN.Common
{$IFDEF USE_PROFILING}
    , QuickNN.Metrics
{$ENDIF}
;

var
  HBLASLIB: HMODULE;
  Workspace: TArray<QNNFloat>;

{$IF defined(USE_PROFILING)}
  Metrics: TTensorMetrics;
{$ENDIF}

procedure CblasGemm(Order: TCBLASOrder; TransA: TCBLASTranspose; TransB: TCBLASTranspose; M: TBlasInt; N: TBlasInt; K: TBlasInt; Alpha: Single; A: TMemoryBlock; Lda: TBlasInt; B: TMemoryBlock; Ldb: TBlasInt; Beta: Single; C: TMemoryBlock; Ldc: TBlasInt); winapi;

procedure CblasAxpy(N: TBlasInt; Alpha: Single; X: TMemoryBlock; IncX: TBlasInt; Y: TMemoryBlock; Incy: TBlasInt); winapi;

function CblasDot(N: TBlasInt; X: TMemoryBlock; IncX: TBlasInt; Y: TMemoryBlock; Incy: TBlasInt): Single; winapi;

function CblasAsum(N: TBlasInt; X: TMemoryBlock; IncX: TBlasInt): QNNFloat; winapi;

procedure CblasScal(N: TBlasInt; Alpha: QNNFloat; X: TMemoryBlock; IncX: TBlasInt = 1); winapi;

(* Global callback pointers - set by caller before inference *)
// Add and Mul terms used for vector vector element wise operations
// Scale and Bias terms used for vector Scalar element wise operations

procedure QNNAdd(const Dst, A, B: TMemoryBlock; const N: Integer);

procedure QNNAddInplace(const Dst, A: TMemoryBlock; const N: Integer);

procedure QNNMul(const Dst, A, B: TMemoryBlock; const N: Integer);

procedure QNNMulInplace(const Dst, A: TMemoryBlock; const N: Integer);

procedure QNNScale(const Dst, Src: TMemoryBlock; const AScale: QNNFloat; const N: LongInt);

procedure QNNScaleInplace(const Dst: TMemoryBlock; const AScale: QNNFloat; const N: LongInt);

procedure QNNBias(const Dst, Src: TMemoryBlock; const ABias: QNNFloat; const N: LongInt);

procedure QNNBiasInplace(const Dst: TMemoryBlock; const ABias: QNNFloat; const N: LongInt);

procedure QNNFusedMulAdd(const Dst, Src, SrcM, SrcA: TMemoryBlock; const N: LongInt);

procedure QNNFusedAddMul(const Dst, Src, SrcA, SrcM: TMemoryBlock; const N: LongInt);

procedure QNNFusedScaleBias(const Dst, Src: TMemoryBlock; const AScale, ABias: QNNFloat; const N: LongInt);

procedure QNNFusedBiasScale(const Dst, Src: TMemoryBlock; const ABias, AScale: QNNFloat; const N: LongInt);

procedure QNNFusedMulScale(const Dst, Src, SrcM: TMemoryBlock; const AScale: QNNFloat; const N: LongInt);

procedure QNNFusedBiasMulScale(const Dst, Src, SrcM: TMemoryBlock; const ABias, AScale: QNNFloat; const N: LongInt);

procedure QNNFusedScaleAdd(const Dst, Src, SrcAdd: TMemoryBlock; const AScale: QNNFloat; const N: LongInt);

procedure QNNFusedBiasAdd(const Dst, Src, SrcA: TMemoryBlock; const ABias: QNNFloat; const N: LongInt);

function QNNDot(const A, B: TMemoryBlock; const N: LongInt): QNNFloat;

procedure QNNAccAdd(const Dst, A, B: TMemoryBlock; const N: Integer);

procedure QNNAccMul(const Dst, A, B: TMemoryBlock; const N: Integer);

procedure QNNMatMulNN(const C, A, B: TMemoryBlock; const M, K, N: Integer);

procedure QNNMatMulNT(const C, A, B: TMemoryBlock; const M, K, N: Integer);

procedure QNNMatMulTN(const C, A, B: TMemoryBlock; const M, K, N: Integer);

procedure QNNMatTranspose(const Dst, Src: TMemoryBlock; const SrcRows, SrcCols: LongInt);

procedure QNNLinear(const Dst, X, W, B: TMemoryBlock; const SeqLen, InDIM, OutDIM: Integer);

procedure QNNLinearNoBias(const Dst, X, W: TMemoryBlock; const SeqLen, InDIM, OutDIM: Integer);

procedure QNNLinearNoBiasBF16(const Dst, X: TMemoryBlock; const W: PBF16; const SeqLen, InDIM, OutDIM: Integer);

procedure QNNIm2Col(const AChannels, AHeight, AWidth, KernelHeight, KernelWidth, PadHeight, PadWidth, StrideY, StrideX, DilationY, DilationX: Integer; const Im: TMemoryBlock; const Col: TMemoryBlock; const ImOffset: Integer = 0; const ColOffset: Integer = 0; const MultiThread: Boolean = False
);

procedure QNNIm2ColStridedBatched(const AChannels, AHeight, AWidth, KernelHeight, KernelWidth, PadHeight, PadWidth, StrideY, StrideX, DilationY, DilationX: Integer; const Im: TMemoryBlock; const ImStride, ImOffset: Integer; const Col: TMemoryBlock; const ColStride, ColOffset: Integer; const Batch: Integer);

function QNNConv2DShape(const Batch, Filters, SrcH, SrcW, KH, KW, Padding, Stride: Int64): TArray<Int64>;

procedure QNNConv2d(var Dst: TMemoryBlock; const Src, Weights, Bias: TMemoryBlock; const InCh, OutCh, H, W, KH, KW, Stride, Padding: LongInt; const Batch: LongInt = 1
);

procedure QNNNorm(const N: LongInt; const Dst, Src: TMemoryBlock; const Weights: TMemoryBlock; const Stride: LongInt = 1
);

procedure QNNRMSNorm(const N: LongInt; const Dst, Src: TMemoryBlock; const Weights: TMemoryBlock; const Stride: LongInt = 1
);

procedure QNNRMSNormRows(const Dst, Src, Weight: TMemoryBlock; const Rows, Dim: Integer);

procedure QNNTanh(const Dst: TMemoryBlock; const N: LongInt);

procedure QNNQKRMSNorm(const Q, K, QWeights, KWeights: TMemoryBlock; const Seq, Heads, HeadDim: LongInt);

procedure QNNRMSNormSeq(const Dst, NormWeights: TMemoryBlock; const Seq, Heads, HeadDim: LongInt);

procedure QNNGroupNorm(const Dst, Src, Gamma, Beta: TMemoryBlock; const Batch, Channels, H, W, NumGroups: Integer);

procedure QNNBatchNormGammaBeta(const Dst, Src, RunningMean, RunningVar, Gamma, Beta: TMemoryBlock; const Batch, Channels, H, W: Integer);

procedure QNNBatchNorm(const Dst, Src, RunningMean, RunningVar: TMemoryBlock; const Batch, Channels, H, W: Integer);

procedure QNNMinMax(const N: Integer; const Src: TMemoryBlock; out OutMin, OutMax: QNNFloat; const ArgMin: PInteger; const ArgMax: PInteger = nil; const Stride: Integer = 1
);

function QNNMax(const N: Integer; const Src: TMemoryBlock; const Arg: PInteger = nil; const Stride: Integer = 1
): QNNFloat;

function QNNMin(const N: Integer; const Src: TMemoryBlock; const Arg: PInteger = nil; const Stride: Integer = 1
): QNNFloat;

function QNNMaxAbs(const N: Integer; const Src: TMemoryBlock; const Stride: Integer = 1): QNNFloat;

function QNNMinAbs(const N: Integer; const Src: TMemoryBlock; const Stride: Integer = 1): QNNFloat;

function QNNArgMax(const N: Integer; const Src: TMemoryBlock; const Stride: Integer = 1): Integer;

function QNNArgMin(const N: Integer; const Src: TMemoryBlock; const Stride: Integer = 1): Integer;

function QNNSum(const N: Integer; const Src: TMemoryBlock; const Stride: Integer = 1): QNNFloat;

function QNNSumSqr(const N: Integer; const Src: TMemoryBlock; const Stride: Integer = 1): QNNFloat;

function QNNSumSqrDiff(const N: Integer; const Src: TMemoryBlock; const AMean: QNNFloat; const Stride: Integer = 1
): QNNFloat;

function QNNSqrDistance(const N: Integer; const Src1, Src2: TMemoryBlock; const Stride: Integer = 1): QNNFloat;

function QNNMaxAbsDiff2(const N: Integer; const Src1, Src2: TMemoryBlock; out OutSrc1, OutSrc2: QNNFloat; const Stride: Integer = 1
): QNNFloat;

function QNNMaxAbsDiff(const N: Integer; const Src1, Src2: TMemoryBlock; const Stride: Integer = 1): QNNFloat;

function QNNSumAbsDiff(const N: Integer; const Src1, Src2: TMemoryBlock; const Stride: Integer = 1): QNNFloat;

function QNNSumAbsDiffScalar(const N: Integer; const Src: TMemoryBlock; const AMean: QNNFloat; const Stride: Integer = 1
): QNNFloat;

function QNNMean(const N: Integer; const Src: TMemoryBlock; const Stride: Integer = 1): QNNFloat;

function QNNVariance(const N: Integer; const Src: TMemoryBlock; const AMean: QNNFloat; const Stride: Integer = 1; const IsPopulation: Boolean = True
): QNNFloat;

procedure QNNSigmoidInplace(const X: TMemoryBlock; const N: Integer);

procedure QNNSigmoid(const Dst, Src: TMemoryBlock; const N: Integer);

procedure QNNSiluInplace(const X: TMemoryBlock; const N: Integer);

procedure QNNSilu(const Dst, Src: TMemoryBlock; const N: Integer);

procedure QNNSiluMul(const Gate, Up: TMemoryBlock; const N: Integer);

procedure QNNSoftmax(const X: TMemoryBlock; const N: Integer);

procedure QNNSoftmaxRows(const X: TMemoryBlock; const Rows, Cols: Integer);

procedure QNNAttention(const Dst, Q, K, V: TMemoryBlock; const Batch, Heads, SeqQ, SeqK, HeadDim: LongInt; const Scale: QNNFloat);

procedure QNNFlashAttentionHead(const Dst, Q, K, V: TMemoryBlock; const SeqQ, SeqK, HeadDim: LongInt; const Scale: QNNFloat);

procedure QNNFlashAttentionHeadTiled(const Dst, Q, K, V: TMemoryBlock; const SeqQ, SeqK, HeadDim: LongInt; const Scale: QNNFloat; const TileScores: TMemoryBlock; const QTileSize, KTileSize: LongInt);

procedure QNNFlashAttention(const Dst, Q, K, V: TMemoryBlock; const SeqQ, SeqK, Heads, HeadDim: LongInt; const Scale: QNNFloat);

procedure QNNUpSampleNearest(const Dst, Src: TMemoryBlock; const Batch, Channels, H, W, ScaleH, ScaleW: LongInt);

procedure QNNUpSample(const Dst, Src: TMemoryBlock; const Batch, Channels, H, W: LongInt; const ScaleH, ScaleW: QNNFloat; const Interpolation: TInterpolation = INearest
);

procedure QNNPatchify(const Dst, Src: TMemoryBlock; const Batch, Channels, H, W, PatchSize: LongInt);

procedure QNNUnpatchify(const Dst, Src: TMemoryBlock; const Batch, Channels, H, W, PatchSize: LongInt);

procedure QNNPermut(var Dst: TMemoryBlock; const Src: TMemoryBlock; const Axis: TArray<Byte>); overload;

function QNNPermut(const Src: TMemoryBlock; const Axis: TArray<Byte>): TMemoryBlock; overload;

procedure QNNCopy(const Dst, Src: TMemoryBlock; const N: Integer);

procedure QNNCopyStrided(const Dst: TMemoryBlock; const DstStride: Integer; const Src: TMemoryBlock; const SrcStride: Integer; const N: Integer);

procedure QNNFill(const Dst: TMemoryBlock; const Val: QNNFloat; const N: Integer; const Stride: Integer = 1);

procedure QNNBroadcast(const Dst: Pointer; const Src; const SrcSize, N: Integer; const Stride: Integer = 1);

procedure QNNGatedAdd(const Dst, Gate, Proj: TMemoryBlock; const Seq, Hidden: Integer);

procedure QNNComputeRoPE(const Dst: TMemoryBlock; const MaxSeq, Dim: LongInt; const Theta: QNNFloat);

procedure QNNComputeRoPE2(const CosOut, SinOut: TMemoryBlock; const MaxSeqLen, HeadDim: LongInt; const Theta: QNNFloat);

procedure QNNComputeRoPE2D(const CosOut, SinDst: TMemoryBlock; const PatchH, PatchW, Dim: LongInt; const Theta: QNNFloat);

procedure QNNComputeRoPE2DOffset(const CosOut, SinOut: TMemoryBlock; const PatchH, PatchW, Dim: LongInt; const Theta: QNNFloat; const OffsetT: LongInt);

procedure QNNComputeRoPEText(const CosOut, SinOut: TMemoryBlock; const TxtSeq, AxisDim: LongInt; const Theta: QNNFloat);

procedure QNNApplyRoPE(const Dst, Freqs: TMemoryBlock; const Batch, Seq, Heads, HeadDim: LongInt);

procedure QNNApplyRoPE2(const Dst, CosIn, SinIn: TMemoryBlock; const NumHeads, HeadDim: LongInt);

procedure QNNApplyRoPE3(const Dst, CosIn, SinIn: TMemoryBlock; const SeqLen, NumHeads, HeadDim: LongInt);

procedure QNNApplyRoPEQK(const Q, K, CosCache, SinCache: TMemoryBlock; const SeqLen, NumQHeads, NumKvHeads, HeadDim: LongInt);

procedure QNNApplyRoPE2D(const Dst, CosFreq, SinFreq: TMemoryBlock; const Seq, Heads, HeadDim, Dim: LongInt);
// Positional Embedding

function QNNTimeStepEmbedding(const T: QNNFloat; const Dim: Integer; const MaxPeriod: QNNFloat): TMemoryBlock;

procedure QNNMatTriangularFill(const Dim: Integer; const Dst: TMemoryBlock; const Val: QNNFloat; const Mask: PLongint = nil
);

procedure QNNMaskFill(const Dst: TMemoryBlock; const Mask: PLongint; const Val: QNNFloat; const Rows, Cols: LongInt);

// Adaptive Layer Normalization
procedure QNNAdaLN(const Dst, Src, Bias, Scale: TMemoryBlock; const Seq, Hidden: LongInt; const Eps: QNNFloat = EPSILON
);

procedure QNNLinearBF16OrF32(const Dst, Src: System.PSingle; const Weight: TMemoryBlock; const WeightBF16: TMemoryBlock; const Seq, SrcDim, DstDim: Integer);

{$IF not declared(FillWord)}
  {$DEFINE FILLWORD}
procedure FillWord(var Dst; const Count: IntPtr; const Val: Word);
{$ENDIF}
{$IF not declared(FillDWord)}
  {$DEFINE FILLDWORD}

procedure FillDWord(var Dst; const Count: IntPtr; const Val: LongWord);
{$ENDIF}
{$IF not declared(FillQWord)}
  {$DEFINE FILLQWORD}

procedure FillQWord(var Dst; const Count: IntPtr; const Val: UInt64);
{$ENDIF}

function IsUsingBlas(): Boolean;
(*
procedure printStat(const src:PQNNFloat; const N:longint);  overload;
procedure printStat(const src:TMemoryBlock);                overload;
procedure printCompare(const N:longint; const src1, src2:PQNNFloat; const isSumSqrDiff:boolean =false);
procedure writeTensor(const buf:TMemoryBlock);
function readTensor():TMemoryBlock;
function readArray():TArray<integer>;
procedure compareArray(const a, b:TArray<Integer>);             overload;
procedure compareArray(const a, b: PInteger; const N:longint);  overload;
procedure compareArray(const a, b: PSingle; const N:longint);  overload;

procedure qgemm(Order:CBLAS_ORDER; TransA:CBLAS_TRANSPOSE; TransB:CBLAS_TRANSPOSE; M:blasint; N:blasint; K:blasint; alpha:single; A:PQNNFloat; lda:blasint; B:PQNNFloat; ldb:blasint; beta:single; C:PQNNFloat; ldc:blasint); WINAPI;
*)

implementation

uses
  QuickNN.CPU, QuickNN.Vulkan;

const
  OP_NOT_IMPL = 'Operation is not implemented for this datatype!';

{$IF defined(FILLWORD)}
procedure FillWord(var Dst; const Count: IntPtr; const Val: Word);
var
  I: IntPtr;
  D: PWord;
begin
  D := @Dst;
  for I := 0 to Count - 1 do
    D[I] := Val
end;
{$ENDIF}

{$IF defined(FILLDWORD)}

procedure FillDWord(var Dst; const Count: IntPtr; const Val: LongWord);
var
  I: IntPtr;
  D: PLongWord;
begin
  D := @Dst;
  for I := 0 to Count - 1 do
    D[I] := Val
end;
{$ENDIF}

{$IF defined(FILLQWORD)}

procedure FillQWord(var Dst; const Count: IntPtr; const Val: UInt64);
var
  I: IntPtr;
  D: PUInt64;
begin
  D := @Dst;
  for I := 0 to Count - 1 do
    D[I] := Val
end;
{$ENDIF}

function IsUsingBlas(): Boolean;
begin
  Result := TQNNSingleOps.IsUsingBlas
end;

procedure CblasGemm(Order: TCBLASOrder; TransA: TCBLASTranspose; TransB: TCBLASTranspose; M: TBlasInt; N: TBlasInt; K: TBlasInt; Alpha: Single; A: TMemoryBlock; Lda: TBlasInt; B: TMemoryBlock; Ldb: TBlasInt; Beta: Single; C: TMemoryBlock; Ldc: TBlasInt); winapi;
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opGemm);
{$ENDIF}
  case C.DataType of
    dtF32:
      TQNNSingleOps.CblasGemm(Order, TransA, TransB, M, N, K, Alpha, A, Lda, B, Ldb, Beta, C, Ldc);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opGemm);
{$ENDIF}
end;

procedure CblasAxpy(N: TBlasInt; Alpha: Single; X: TMemoryBlock; IncX: TBlasInt; Y: TMemoryBlock; Incy: TBlasInt); winapi;
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opAxpy);
{$ENDIF}
  case X.DataType of
    dtF32:
      TQNNSingleOps.CblasAxpy(N, Alpha, X, IncX, Y, Incy);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opAxpy);
{$ENDIF}
end;

function CblasDot(N: TBlasInt; X: TMemoryBlock; IncX: TBlasInt; Y: TMemoryBlock; Incy: TBlasInt): Single; winapi;
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opDot);
{$ENDIF}
  case X.DataType of
    dtF32:
      Result := TQNNSingleOps.CblasDot(N, X, IncX, Y, Incy);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opDot);
{$ENDIF}
end;

function CblasAsum(N: TBlasInt; X: TMemoryBlock; IncX: TBlasInt): QNNFloat; winapi;
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opAbsSum);
{$ENDIF}
  case X.DataType of
    dtF32:
      Result := TQNNSingleOps.CblasAsum(N, X, IncX);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opAbsSum);
{$ENDIF}
end;

procedure CblasScal(N: TBlasInt; Alpha: QNNFloat; X: TMemoryBlock; IncX: TBlasInt); winapi;
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opMulvs);
{$ENDIF}
  case X.DataType of
    dtF32:
      TQNNSingleOps.CblasScal(N, Alpha, X, IncX);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opMulvs);
{$ENDIF}
end;

procedure QNNAdd(const Dst, A, B: TMemoryBlock; const N: Integer);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opAddvv);
{$ENDIF}
  Assert(A.Offset + N <= A.Size, 'ERROR [QNNAdd] : N is out of source tesnor bounds');
  Assert(B.Offset + N <= B.Size, 'ERROR [QNNAdd] : N is out of destination tesnor bounds');
  Assert(Dst.Offset + N <= Dst.Size, 'ERROR [QNNAdd] : N is out of destination tesnor bounds');
  case Dst.DataType of
    dtF32:
      TQNNSingleOps.QNNAdd(Dst, A, B, N);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opAddvv);
{$ENDIF}
end;

procedure QNNAddInplace(const Dst, A: TMemoryBlock; const N: Integer);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opAddvv);
{$ENDIF}
  Assert(A.Offset + N <= A.Size, 'ERROR [QNNAddInplace] : N is out of source tesnor bounds');
  Assert(Dst.Offset + N <= Dst.Size, 'ERROR [QNNAddInplace] : N is out of destination tesnor bounds');
  case Dst.DataType of
    dtF32:
      TQNNSingleOps.QNNAddInplace(Dst, A, N);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opAddvv);
{$ENDIF}
end;

procedure QNNMul(const Dst, A, B: TMemoryBlock; const N: Integer);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opMulvv);
{$ENDIF}
  Assert(A.Offset + N <= A.Size, 'ERROR [QNNMul] : N is out of source tesnor bounds');
  Assert(B.Offset + N <= B.Size, 'ERROR [QNNMul] : N is out of destination tesnor bounds');
  Assert(Dst.Offset + N <= Dst.Size, 'ERROR [QNNMul] : N is out of destination tesnor bounds');
  case Dst.DataType of
    dtF32:
      TQNNSingleOps.QNNMul(Dst, A, B, N);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opMulvv);
{$ENDIF}
end;

procedure QNNMulInplace(const Dst, A: TMemoryBlock; const N: Integer);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opMulvv);
{$ENDIF}
  Assert(A.Offset + N <= A.Size, 'ERROR [QNNMulInplace] : N is out of source tesnor bounds');
  Assert(Dst.Offset + N <= Dst.Size, 'ERROR [QNNMulInplace] : N is out of destination tesnor bounds');
  case Dst.DataType of
    dtF32:
      TQNNSingleOps.QNNMulInplace(Dst, A, N);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opMulvv);
{$ENDIF}
end;

procedure QNNScale(const Dst, Src: TMemoryBlock; const AScale: QNNFloat; const N: LongInt);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opMulvs);
{$ENDIF}
  Assert(Src.Offset + N <= Src.Size, 'ERROR [QNNScale] : N is out of source tesnor bounds');
  Assert(Dst.Offset + N <= Dst.Size, 'ERROR [QNNScale] : N is out of destination tesnor bounds');
  case Dst.DataType of
    dtF32:
      TQNNSingleOps.QNNScale(Dst, Src, AScale, N);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opMulvs);
{$ENDIF}
end;

procedure QNNScaleInplace(const Dst: TMemoryBlock; const AScale: QNNFloat; const N: LongInt);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opMulvs);
{$ENDIF}
  Assert(Dst.Offset + N <= Dst.Size, 'ERROR [QNNScaleInplace] : N is out of destination tesnor bounds');
  case Dst.DataType of
    dtF32:
      TQNNSingleOps.QNNScaleInplace(Dst, AScale, N);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opMulvs);
{$ENDIF}
end;

procedure QNNBias(const Dst, Src: TMemoryBlock; const ABias: QNNFloat; const N: LongInt);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opAddvs);
{$ENDIF}
  Assert(Src.Offset + N <= Src.Size, 'ERROR [QNNBias] : N is out of source tesnor bounds');
  Assert(Dst.Offset + N <= Dst.Size, 'ERROR [QNNBias] : N is out of destination tesnor bounds');
  case Dst.DataType of
    dtF32:
      TQNNSingleOps.QNNBias(Dst, Src, ABias, N)
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opAddvs);
{$ENDIF}
end;

procedure QNNBiasInplace(const Dst: TMemoryBlock; const ABias: QNNFloat; const N: LongInt);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opAddvs);
{$ENDIF}
  Assert(Dst.Offset + N <= Dst.Size, 'ERROR [QNNBiasInplace] : N is out of destination tesnor bounds');
  case Dst.DataType of
    dtF32:
      TQNNSingleOps.QNNBiasInplace(Dst, ABias, N);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opAddvs);
{$ENDIF}
end;

procedure QNNFusedMulAdd(const Dst, Src, SrcM, SrcA: TMemoryBlock; const N: LongInt);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opFmavv);
{$ENDIF}
  Assert(Src.Offset + N <= Src.Size, 'ERROR [QNNFusedMulAdd] : N is out of source tesnor bounds');
  Assert(Dst.Offset + N <= Dst.Size, 'ERROR [QNNFusedMulAdd] : N is out of destination tesnor bounds');
  Assert(SrcA.Offset + N <= SrcA.Size, 'ERROR [QNNFusedMulAdd] : N is out of sourceA tesnor bounds');
  Assert(SrcM.Offset + N <= SrcM.Size, 'ERROR [QNNFusedMulAdd] : N is out of sourceM tesnor bounds');
  case Dst.DataType of
    dtF32:
      TQNNSingleOps.QNNFusedMulAdd(Dst, Src, SrcM, SrcA, N);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opFmavv);
{$ENDIF}
end;

procedure QNNFusedAddMul(const Dst, Src, SrcA, SrcM: TMemoryBlock; const N: LongInt);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opFmavv);
{$ENDIF}
  Assert(Src.Offset + N <= Src.Size, 'ERROR [QNNFusedAddMul] : N is out of source tesnor bounds');
  Assert(Dst.Offset + N <= Dst.Size, 'ERROR [QNNFusedAddMul] : N is out of destination tesnor bounds');
  Assert(SrcA.Offset + N <= SrcA.Size, 'ERROR [QNNFusedAddMul] : N is out of sourceA tesnor bounds');
  Assert(SrcM.Offset + N <= SrcM.Size, 'ERROR [QNNFusedAddMul] : N is out of sourceM tesnor bounds');
  case Dst.DataType of
    dtF32:
      TQNNSingleOps.QNNFusedAddMul(Dst, Src, SrcA, SrcM, N);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opFmavv);
{$ENDIF}
end;

procedure QNNFusedScaleBias(const Dst, Src: TMemoryBlock; const AScale, ABias: QNNFloat; const N: LongInt);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opFmavss);
{$ENDIF}
  Assert(Src.Offset + N <= Src.Size, 'ERROR [QNNFusedScaleBias] : N is out of source tesnor bounds');
  Assert(Dst.Offset + N <= Dst.Size, 'ERROR [QNNFusedScaleBias] : N is out of destination tesnor bounds');
  case Dst.DataType of
    dtF32:
      TQNNSingleOps.QNNFusedScaleBias(Dst, Src, AScale, ABias, N);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opFmavss);
{$ENDIF}
end;

procedure QNNFusedBiasScale(const Dst, Src: TMemoryBlock; const ABias, AScale: QNNFloat; const N: LongInt);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opFmavss);
{$ENDIF}
  Assert(Src.Offset + N <= Src.Size, 'ERROR [QNNFusedBiasScale] : N is out of source tesnor bounds');
  Assert(Dst.Offset + N <= Dst.Size, 'ERROR [QNNFusedBiasScale] : N is out of destination tesnor bounds');
  case Dst.DataType of
    dtF32:
      TQNNSingleOps.QNNFusedBiasScale(Dst, Src, ABias, AScale, N);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opFmavss);
{$ENDIF}
end;

procedure QNNFusedMulScale(const Dst, Src, SrcM: TMemoryBlock; const AScale: QNNFloat; const N: LongInt);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opFmavvs);
{$ENDIF}
  Assert(Src.Offset + N <= Src.Size, 'ERROR [QNNFusedMulScale] : N is out of source tesnor bounds');
  Assert(SrcM.Offset + N <= SrcM.Size, 'ERROR [QNNFusedMulScale] : N is out of sourceM tesnor bounds');
  Assert(Dst.Offset + N <= Dst.Size, 'ERROR [QNNFusedMulScale] : N is out of destination tesnor bounds');
  case Dst.DataType of
    dtF32:
      TQNNSingleOps.QNNFusedMulScale(Dst, Src, SrcM, AScale, N);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opFmavvs);
{$ENDIF}
end;

procedure QNNFusedBiasMulScale(const Dst, Src, SrcM: TMemoryBlock; const ABias, AScale: QNNFloat; const N: LongInt);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opFmavss);
{$ENDIF}
  Assert(Src.Offset + N <= Src.Size, 'ERROR [QNNFusedBiasMulScale] : N is out of source tesnor bounds');
  Assert(SrcM.Offset + N <= SrcM.Size, 'ERROR [QNNFusedBiasMulScale] : N is out of sourceM tesnor bounds');
  Assert(Dst.Offset + N <= Dst.Size, 'ERROR [QNNFusedBiasMulScale] : N is out of destination tesnor bounds');
  case Dst.DataType of
    dtF32:
      TQNNSingleOps.QNNFusedBiasMulScale(Dst, Src, SrcM, ABias, AScale, N);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opFmavss);
{$ENDIF}
end;

procedure QNNFusedScaleAdd(const Dst, Src, SrcAdd: TMemoryBlock; const AScale: QNNFloat; const N: LongInt);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opFmavvs);
{$ENDIF}
  Assert(Src.Offset + N <= Src.Size, 'ERROR [QNNFusedScaleAdd] : N is out of source tesnor bounds');
  Assert(SrcAdd.Offset + N <= SrcAdd.Size, 'ERROR [QNNFusedScaleAdd] : N is out of dstAdd tesnor bounds');
  Assert(Dst.Offset + N <= Dst.Size, 'ERROR [QNNFusedScaleAdd] : N is out of destination tesnor bounds');

  case Dst.DataType of
    dtF32:
      TQNNSingleOps.QNNFusedScaleAdd(Dst, Src, SrcAdd, AScale, N);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opFmavvs);
{$ENDIF}
end;

procedure QNNFusedBiasAdd(const Dst, Src, SrcA: TMemoryBlock; const ABias: QNNFloat; const N: LongInt);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opFmavvs);
{$ENDIF}
  Assert(Src.Offset + N <= Src.Size, 'ERROR [QNNFusedBiasAdd] : N is out of source tesnor bounds');
  Assert(SrcA.Offset + N <= SrcA.Size, 'ERROR [QNNFusedBiasAdd] : N is out of sourceAdd tesnor bounds');
  Assert(Dst.Offset + N <= Dst.Size, 'ERROR [QNNFusedBiasAdd] : N is out of destination tesnor bounds');
  case Dst.DataType of
    dtF32:
      TQNNSingleOps.QNNFusedBiasAdd(Dst, Src, SrcA, ABias, N);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opFmavvs);
{$ENDIF}
end;

function QNNDot(const A, B: TMemoryBlock; const N: LongInt): QNNFloat;
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opDot);
{$ENDIF}
  Assert(A.Offset + N <= A.Size, 'ERROR [QNNDot] : N is out of source tesnor bounds');
  Assert(B.Offset + N <= B.Size, 'ERROR [QNNDot] : N is out of destination tesnor bounds');
  case A.DataType of
    dtF32:
      Result := TQNNSingleOps.QNNDot(A, B, N);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opDot);
{$ENDIF}
end;

procedure QNNAccAdd(const Dst, A, B: TMemoryBlock; const N: Integer);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opFmavv);
{$ENDIF}
  Assert(A.Offset + N <= A.Size, 'ERROR [QNNAccAdd] : N is out of source tesnor bounds');
  Assert(B.Offset + N <= B.Size, 'ERROR [QNNAccAdd] : N is out of destination tesnor bounds');
  Assert(Dst.Offset + N <= Dst.Size, 'ERROR [QNNAccAdd] : N is out of destination tesnor bounds');
  case Dst.DataType of
    dtF32:
      TQNNSingleOps.QNNAccAdd(Dst, A, B, N);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opFmavv);
{$ENDIF}
end;

procedure QNNAccMul(const Dst, A, B: TMemoryBlock; const N: Integer);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opFmavv);
{$ENDIF}
  Assert(A.Offset + N <= A.Size, 'ERROR [QNNAccMul] : N is out of source tesnor bounds');
  Assert(B.Offset + N <= B.Size, 'ERROR [QNNAccMul] : N is out of destination tesnor bounds');
  Assert(Dst.Offset + N <= Dst.Size, 'ERROR [QNNAccMul] : N is out of destination tesnor bounds');
  case Dst.DataType of
    dtF32:
      TQNNSingleOps.QNNAccMul(Dst, A, B, N);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opFmavv);
{$ENDIF}
end;

procedure QNNMatMulNN(const C, A, B: TMemoryBlock; const M, K, N: Integer);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opGemm);
{$ENDIF}
  case C.DataType of
    dtF32:
      TQNNSingleOps.QNNMatMulNN(C, A, B, M, K, N);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opGemm);
{$ENDIF}
end;

procedure QNNMatMulNT(const C, A, B: TMemoryBlock; const M, K, N: Integer);
begin
  if (C.DataType = dtF32) and (A.DataType = dtF32) and TryVulkanLinear(C, A, B, M, K, N) then
    Exit;
{$IF defined(USE_PROFILING)}
  Metrics.Start(opGemm);
{$ENDIF}
  case C.DataType of
    dtF32:
      if B.DataType = dtBF16 then
        TQNNSingleOps.QNNLinearNoBiasBF16(C, A, PBF16(B), M, K, N)
      else
        TQNNSingleOps.QNNMatMulNT(C, A, B, M, K, N);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opGemm);
{$ENDIF}
end;

procedure QNNMatMulTN(const C, A, B: TMemoryBlock; const M, K, N: Integer);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opGemm);
{$ENDIF}
  case C.DataType of
    dtF32:
      TQNNSingleOps.QNNMatMulTN(C, A, B, M, K, N);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opGemm);
{$ENDIF}
end;

procedure QNNMatTranspose(const Dst, Src: TMemoryBlock; const SrcRows, SrcCols: LongInt);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opOther);
{$ENDIF}
  Assert(Src.Offset + SrcRows * SrcCols <= Src.Size, 'QNNTranspose : src dimensions out of bounds.');
  Assert(Dst.Offset + SrcRows * SrcCols <= Dst.Size, 'QNNTranspose : dst dimensions out of bounds.');
  case Dst.DataType of
    dtF32:
      TQNNSingleOps.QNNMatTranspose(Dst, Src, SrcRows, SrcCols);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opOther);
{$ENDIF}
end;

procedure QNNLinear(const Dst, X, W, B: TMemoryBlock; const SeqLen, InDIM, OutDIM: Integer);
var
  Row: Integer;
begin
  if (Dst.DataType = dtF32) and (X.DataType = dtF32) and TryVulkanLinear(Dst, X, W, SeqLen, InDIM, OutDIM) then
  begin
    if B.IsAssigned then
      for Row := 0 to SeqLen - 1 do
        TQNNSingleOps.QNNAddInplace(System.PSingle(Dst) + Int64(Row) * OutDIM, B, OutDIM);
    Exit;
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Start(opGemm);
{$ENDIF}
  case Dst.DataType of
    dtF32:
      if W.DataType = dtBF16 then
      begin
        TQNNSingleOps.QNNLinearNoBiasBF16(Dst, X, PBF16(W), SeqLen, InDIM, OutDIM);
        if B.IsAssigned then
          for Row := 0 to SeqLen - 1 do
            TQNNSingleOps.QNNAddInplace(System.PSingle(Dst) + Int64(Row) * OutDIM, B, OutDIM);
      end
      else
        TQNNSingleOps.QNNLinear(Dst, X, W, B, SeqLen, InDIM, OutDIM);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opGemm);
{$ENDIF}
end;

procedure QNNLinearNoBias(const Dst, X, W: TMemoryBlock; const SeqLen, InDIM, OutDIM: Integer);
begin
  if (Dst.DataType = dtF32) and (X.DataType = dtF32) and TryVulkanLinear(Dst, X, W, SeqLen, InDIM, OutDIM) then
    Exit;
{$IF defined(USE_PROFILING)}
  Metrics.Start(opGemm);
{$ENDIF}
  case Dst.DataType of
    dtF32:
      if W.DataType = dtBF16 then
        TQNNSingleOps.QNNLinearNoBiasBF16(Dst, X, PBF16(W), SeqLen, InDIM, OutDIM)
      else
        TQNNSingleOps.QNNLinearNoBias(Dst, X, W, SeqLen, InDIM, OutDIM);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opGemm);
{$ENDIF}
end;

procedure QNNLinearNoBiasBF16(const Dst, X: TMemoryBlock; const W: PBF16; const SeqLen, InDIM, OutDIM: Integer);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opGemm);
{$ENDIF}
  case Dst.DataType of
    dtF32:
      TQNNSingleOps.QNNLinearNoBiasBF16(Dst, X, W, SeqLen, InDIM, OutDIM);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opGemm);
{$ENDIF}
end;

procedure QNNIm2Col(const AChannels, AHeight, AWidth, KernelHeight, KernelWidth, PadHeight, PadWidth, StrideY, StrideX, DilationY, DilationX: Integer; const Im: TMemoryBlock; const Col: TMemoryBlock; const ImOffset: Integer; const ColOffset: Integer; const MultiThread: Boolean);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opIm2col);
{$ENDIF}

  case Col.DataType of
    dtF32:
      TQNNSingleOps.QNNIm2Col(
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
        Im,
        Col,
        ImOffset,
        ColOffset,
        MultiThread
      );
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opIm2col);
{$ENDIF}
end;

procedure QNNIm2ColStridedBatched(const AChannels, AHeight, AWidth, KernelHeight, KernelWidth, PadHeight, PadWidth, StrideY, StrideX, DilationY, DilationX: Integer; const Im: TMemoryBlock; const ImStride, ImOffset: Integer; const Col: TMemoryBlock; const ColStride, ColOffset: Integer; const Batch: Integer);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opIm2ColExt);
{$ENDIF}

  case Col.DataType of
    dtF32:
      TQNNSingleOps.QNNIm2ColStridedBatched(
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
        Im,
        ImStride,
        ImOffset,
        Col,
        ColStride,
        ColOffset,
        Batch
      );
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opIm2ColExt);
{$ENDIF}
end;

// calculate Conv2D target shape dimension from an input dimension
function QNNConv2DShape(const Batch, Filters, SrcH, SrcW, KH, KW, Padding, Stride: Int64): TArray<Int64>;
begin
  Result := [Batch, Filters, (SrcH + 2 * Padding - KH) div Stride + 1, (SrcW + 2 * Padding - KW) div Stride + 1];
end;

const
  SZ: Int64 = 0;

procedure QNNConv2d(var Dst: TMemoryBlock; const Src, Weights, Bias: TMemoryBlock; const InCh, OutCh, H, W, KH, KW, Stride, Padding: LongInt; const Batch: LongInt);
var
  Dims: TArray<Int64>;
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opConv2D);
{$ENDIF}
  Dims := QNNConv2DShape(Batch, OutCh, H, W, KH, KW, Padding, Stride);
  Assert(Dst.Offset + Product(Dims) <= Dst.Size, 'ERROR [Conv2D] : Incorrect destination size');
  Dst.Reshape(Dims);
  case Dst.DataType of
    dtF32:
      TQNNSingleOps.QNNConv2d(Dst, Src, Weights, Bias, InCh, OutCh, H, W, KH, KW, Stride, Padding, Batch);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opConv2D);
{$ENDIF}
end;

procedure QNNNorm(const N: LongInt; const Dst, Src: TMemoryBlock; const Weights: TMemoryBlock; const Stride: LongInt);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opNormalize);
{$ENDIF}
  Assert(Src.Offset + N <= Src.Size, 'ERROR [QNNNorm] : N is out of source tesnor bounds');
  Assert(Weights.Offset + N <= Weights.Size, 'ERROR [QNNNorm] : N is out of weights tesnor bounds');
  Assert(Dst.Offset + N <= Dst.Size, 'ERROR [QNNNorm] : N is out of destination tesnor bounds');
  case Dst.DataType of
    dtF32:
      TQNNSingleOps.QNNNorm(N, Dst, Src, Weights, Stride);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opNormalize);
{$ENDIF}
end;

procedure QNNRMSNorm(const N: LongInt; const Dst, Src: TMemoryBlock; const Weights: TMemoryBlock; const Stride: LongInt);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opRMSNorm);
{$ENDIF}
  Assert(Src.Offset + N <= Src.Size, 'ERROR [QNNRMSNorm] : N is out of source tesnor bounds');
  Assert(Weights.Offset + N <= Weights.Size, 'ERROR [QNNRMSNorm] : N is out of weights tesnor bounds');
  Assert(Dst.Offset + N <= Dst.Size, 'ERROR [QNNRMSNorm] : N is out of destination tesnor bounds');
  case Dst.DataType of
    dtF32:
      TQNNSingleOps.QNNRMSNorm(N, Dst, Src, Weights, Stride);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opRMSNorm);
{$ENDIF}
end;

procedure QNNRMSNormRows(const Dst, Src, Weight: TMemoryBlock; const Rows, Dim: Integer);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opRMSNorm);
{$ENDIF}
  Assert(Src.Offset + Dim * Rows <= Src.Size, 'ERROR [QNNRMSNormRows] : out of source tesnor bounds');
  Assert(Weight.Offset + Dim <= Weight.Size, 'ERROR [QNNRMSNormRows] : dim is out of weight tesnor bounds');
  Assert(Dst.Offset + Dim * Rows <= Dst.Size, 'ERROR [QNNRMSNormRows] : out of destination tesnor bounds');
  case Dst.DataType of
    dtF32:
      TQNNSingleOps.QNNRMSNormRows(Dst, Src, Weight, Rows, Dim);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opRMSNorm);
{$ENDIF}
end;

procedure QNNTanh(const Dst: TMemoryBlock; const N: LongInt);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opTanh);
{$ENDIF}
  Assert(Dst.Offset + N <= Dst.Size, 'ERROR [QNNTanh] : N is out of destination tesnor bounds');
  case Dst.DataType of
    dtF32:
      TQNNSingleOps.QNNTanh(Dst, N);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opTanh);
{$ENDIF}
end;

procedure QNNQKRMSNorm(const Q, K, QWeights, KWeights: TMemoryBlock; const Seq, Heads, HeadDim: LongInt);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opRMSNorm);
{$ENDIF}
  Assert(Q.Offset + Seq * Heads * HeadDim <= Q.Size, 'ERROR [QNNQKRMSNorm] : out of Q tesnor bounds');
  Assert(QWeights.Offset + HeadDim <= QWeights.Size, 'ERROR [QNNQKRMSNorm] : headDim is out of Qweights tesnor bounds');
  Assert(K.Offset + Seq * Heads * HeadDim <= K.Size, 'ERROR [QNNQKRMSNorm] : out of K tesnor bounds');
  Assert(KWeights.Offset + HeadDim <= KWeights.Size, 'ERROR [QNNQKRMSNorm] : headDim is out of Kweights tesnor bounds');
  case Q.DataType of
    dtF32:
      TQNNSingleOps.QNNQKRMSNorm(Q, K, QWeights, KWeights, Seq, Heads, HeadDim);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opRMSNorm);
{$ENDIF}
end;

procedure QNNRMSNormSeq(const Dst, NormWeights: TMemoryBlock; const Seq, Heads, HeadDim: LongInt);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opRMSNorm);
{$ENDIF}
  Assert(
    NormWeights.Offset + HeadDim <= NormWeights.Size,
    'ERROR [QNNRMSNormSeq] : headDim is out of weights tesnor bounds'
  );
  Assert(Dst.Offset + Seq * Heads * HeadDim <= Dst.Size, 'ERROR [QNNRMSNormSeq] : out of destination tesnor bounds');
  case Dst.DataType of
    dtF32:
      TQNNSingleOps.QNNRMSNormSeq(Dst, NormWeights, Seq, Heads, HeadDim);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opRMSNorm);
{$ENDIF}
end;

procedure QNNGroupNorm(const Dst, Src, Gamma, Beta: TMemoryBlock; const Batch, Channels, H, W, NumGroups: Integer);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opGroupNorm);
{$ENDIF}
  Assert(Gamma.Offset + Channels <= Gamma.Size, 'ERROR [QNNGroupNorm] : out of gamma tesnor bounds');
  Assert(Beta.Offset + Channels <= Beta.Size, 'ERROR [QNNGroupNorm] : out of beta tesnor bounds');
  Assert(Src.Offset + Batch * Channels * H * W <= Src.Size, 'ERROR [QNNGroupNorm] : out of source tesnor bounds');
  Assert(Dst.Offset + Batch * Channels * H * W <= Dst.Size, 'ERROR [QNNGroupNorm] : out of destination tesnor bounds');

  case Dst.DataType of
    dtF32:
      TQNNSingleOps.QNNGroupNorm(Dst, Src, Gamma, Beta, Batch, Channels, H, W, NumGroups)
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opGroupNorm);
{$ENDIF}
end;

procedure QNNBatchNormGammaBeta(const Dst, Src, RunningMean, RunningVar, Gamma, Beta: TMemoryBlock; const Batch, Channels, H, W: Integer);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opBatchNorm);
{$ENDIF}
  Assert(
    RunningMean.Offset + Channels <= RunningMean.Size,
    'ERROR [QNNBatchNormGammaBeta] : out of running_mean tesnor bounds'
  );
  Assert(
    RunningVar.Offset + Channels <= RunningVar.Size,
    'ERROR [QNNBatchNormGammaBeta] : out of running_var tesnor bounds'
  );
  Assert(Gamma.Offset + Channels <= Gamma.Size, 'ERROR [QNNBatchNormGammaBeta] : out of gamma tesnor bounds');
  Assert(Beta.Offset + Channels <= Beta.Size, 'ERROR [QNNBatchNormGammaBeta] : out of beta tesnor bounds');
  Assert(
    Src.Offset + Batch * Channels * H * W <= Src.Size,
    'ERROR [QNNBatchNormGammaBeta] : out of source tesnor bounds'
  );
  Assert(
    Dst.Offset + Batch * Channels * H * W <= Dst.Size,
    'ERROR [QNNBatchNormGammaBeta] : out of destination tesnor bounds'
  );
  case Dst.DataType of
    dtF32:
      TQNNSingleOps.QNNBatchNorm(Dst, Src, RunningMean, RunningVar, Gamma, Beta, Batch, Channels, H, W);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opBatchNorm);
{$ENDIF}
end;

procedure QNNBatchNorm(const Dst, Src, RunningMean, RunningVar: TMemoryBlock; const Batch, Channels, H, W: Integer);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opBatchNorm);
{$ENDIF}
  Assert(RunningMean.Offset + Channels <= RunningMean.Size, 'ERROR [QNNBatchNorm] : out of running_mean tesnor bounds');
  Assert(RunningVar.Offset + Channels <= RunningVar.Size, 'ERROR [QNNBatchNorm] : out of running_var tesnor bounds');
  Assert(Src.Offset + Batch * Channels * H * W <= Src.Size, 'ERROR [QNNBatchNorm] : out of source tesnor bounds');
  Assert(Dst.Offset + Batch * Channels * H * W <= Dst.Size, 'ERROR [QNNBatchNorm] : out of destination tesnor bounds');
  case Dst.DataType of
    dtF32:
      TQNNSingleOps.QNNBatchNorm(Dst, Src, RunningMean, RunningVar, nil, nil, Batch, Channels, H, W);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opBatchNorm);
{$ENDIF}
end;

procedure QNNMinMax(const N: Integer; const Src: TMemoryBlock; out OutMin, OutMax: QNNFloat; const ArgMin: PInteger; const ArgMax: PInteger; const Stride: Integer);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opMinAndMax);
{$ENDIF}

  case Src.DataType of
    dtF32:
      TQNNSingleOps.QNNMinMax(N, Src, OutMin, OutMax, ArgMin, ArgMax, Stride);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opMinAndMax);
{$ENDIF}
end;

function QNNMax(const N: Integer; const Src: TMemoryBlock; const Arg: PInteger; const Stride: Integer): QNNFloat;
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opMax);
{$ENDIF}

  case Src.DataType of
    dtF32:
      Result := TQNNSingleOps.QNNMax(N, Src, Arg, Stride);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opMax);
{$ENDIF}
end;

function QNNMin(const N: Integer; const Src: TMemoryBlock; const Arg: PInteger; const Stride: Integer): QNNFloat;
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opMin);
{$ENDIF}

  case Src.DataType of
    dtF32:
      Result := TQNNSingleOps.QNNMin(N, Src, Arg, Stride);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opMin);
{$ENDIF}
end;

function QNNMaxAbs(const N: Integer; const Src: TMemoryBlock; const Stride: Integer): QNNFloat;
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opMaxAbs);
{$ENDIF}

  case Src.DataType of
    dtF32:
      Result := TQNNSingleOps.QNNMaxAbs(N, Src, Stride);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opMaxAbs);
{$ENDIF}
end;

function QNNMinAbs(const N: Integer; const Src: TMemoryBlock; const Stride: Integer): QNNFloat;
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opMinAbs);
{$ENDIF}

  case Src.DataType of
    dtF32:
      Result := TQNNSingleOps.QNNMinAbs(N, Src, Stride);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opMinAbs);
{$ENDIF}
end;

function QNNArgMax(const N: Integer; const Src: TMemoryBlock; const Stride: Integer): Integer;
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opArgMax);
{$ENDIF}

  case Src.DataType of
    dtF32:
      Result := TQNNSingleOps.QNNArgMax(N, Src, Stride);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opArgMax);
{$ENDIF}
end;

function QNNArgMin(const N: Integer; const Src: TMemoryBlock; const Stride: Integer): Integer;
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opArgMin);
{$ENDIF}

  case Src.DataType of
    dtF32:
      Result := TQNNSingleOps.QNNArgMin(N, Src, Stride);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opArgMin);
{$ENDIF}
end;

function QNNSum(const N: Integer; const Src: TMemoryBlock; const Stride: Integer): QNNFloat;
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opSum);
{$ENDIF}

  case Src.DataType of
    dtF32:
      Result := TQNNSingleOps.QNNSum(N, Src, Stride);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opSum);
{$ENDIF}
end;

function QNNSumSqr(const N: Integer; const Src: TMemoryBlock; const Stride: Integer): QNNFloat;
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opSumSqr);
{$ENDIF}

  case Src.DataType of
    dtF32:
      Result := TQNNSingleOps.QNNSumSqr(N, Src, Stride);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opSumSqr);
{$ENDIF}
end;

function QNNSumSqrDiff(const N: Integer; const Src: TMemoryBlock; const AMean: QNNFloat; const Stride: Integer): QNNFloat;
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opSumSqrDiffScaler);
{$ENDIF}

  case Src.DataType of
    dtF32:
      Result := TQNNSingleOps.QNNSumSqrDiff(N, Src, AMean, Stride);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opSumSqrDiffScaler);
{$ENDIF}
end;

function QNNSqrDistance(const N: Integer; const Src1, Src2: TMemoryBlock; const Stride: Integer): QNNFloat;
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opDistance);
{$ENDIF}

  case Src1.DataType of
    dtF32:
      Result := TQNNSingleOps.QNNSqrDistance(N, Src1, Src2, Stride);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opDistance);
{$ENDIF}
end;

function QNNMaxAbsDiff2(const N: Integer; const Src1, Src2: TMemoryBlock; out OutSrc1, OutSrc2: QNNFloat; const Stride: Integer): QNNFloat;
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opMaxAbsDiff);
{$ENDIF}

  case Src1.DataType of
    dtF32:
      Result := TQNNSingleOps.QNNMaxAbsDiff2(N, Src1, Src2, OutSrc1, OutSrc2, Stride);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opMaxAbsDiff);
{$ENDIF}
end;

function QNNMaxAbsDiff(const N: Integer; const Src1, Src2: TMemoryBlock; const Stride: Integer): QNNFloat;
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opMaxAbsDiff);
{$ENDIF}

  case Src1.DataType of
    dtF32:
      Result := TQNNSingleOps.QNNMaxAbsDiff(N, Src1, Src2, Stride);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opMaxAbsDiff);
{$ENDIF}
end;

function QNNSumAbsDiff(const N: Integer; const Src1, Src2: TMemoryBlock; const Stride: Integer): QNNFloat;
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opSumAbsDiff);
{$ENDIF}

  case Src1.DataType of
    dtF32:
      Result := TQNNSingleOps.QNNSumAbsDiff(N, Src1, Src2, Stride);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opSumAbsDiff);
{$ENDIF}
end;

function QNNSumAbsDiffScalar(const N: Integer; const Src: TMemoryBlock; const AMean: QNNFloat; const Stride: Integer): QNNFloat;
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opAbsDiffScaler);
{$ENDIF}

  case Src.DataType of
    dtF32:
      Result := TQNNSingleOps.QNNSumAbsDiffScalar(N, Src, AMean, Stride);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opAbsDiffScaler);
{$ENDIF}
end;

function QNNMean(const N: Integer; const Src: TMemoryBlock; const Stride: Integer): QNNFloat;
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opMeans);
{$ENDIF}

  case Src.DataType of
    dtF32:
      Result := TQNNSingleOps.QNNMean(N, Src, Stride);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opMeans);
{$ENDIF}
end;

function QNNVariance(const N: Integer; const Src: TMemoryBlock; const AMean: QNNFloat; const Stride: Integer; const IsPopulation: Boolean): QNNFloat;
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opVariances);
{$ENDIF}

  case Src.DataType of
    dtF32:
      Result := TQNNSingleOps.QNNVariance(N, Src, AMean, Stride, IsPopulation);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opVariances);
{$ENDIF}
end;

procedure QNNSigmoidInplace(const X: TMemoryBlock; const N: Integer);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opSigmoidInplace);
{$ENDIF}

  case X.DataType of
    dtF32:
      TQNNSingleOps.QNNSigmoidInplace(X, N);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opSigmoidInplace);
{$ENDIF}
end;

procedure QNNSigmoid(const Dst, Src: TMemoryBlock; const N: Integer);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opSigmoid);
{$ENDIF}

  case Dst.DataType of
    dtF32:
      TQNNSingleOps.QNNSigmoid(Dst, Src, N);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opSigmoid);
{$ENDIF}
end;

procedure QNNSiluInplace(const X: TMemoryBlock; const N: Integer);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opSiLUInplace);
{$ENDIF}

  case X.DataType of
    dtF32:
      TQNNSingleOps.QNNSiluInplace(X, N);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opSiLUInplace);
{$ENDIF}
end;

procedure QNNSilu(const Dst, Src: TMemoryBlock; const N: Integer);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opSiLU);
{$ENDIF}

  case Dst.DataType of
    dtF32:
      TQNNSingleOps.QNNSilu(Dst, Src, N);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opSiLU);
{$ENDIF}
end;

procedure QNNSiluMul(const Gate, Up: TMemoryBlock; const N: Integer);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opSiLUMul);
{$ENDIF}

  case Gate.DataType of
    dtF32:
      TQNNSingleOps.QNNSiluMul(Gate, Up, N);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opSiLUMul);
{$ENDIF}
end;

procedure QNNSoftmax(const X: TMemoryBlock; const N: Integer);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opSoftmax);
{$ENDIF}

  case X.DataType of
    dtF32:
      TQNNSingleOps.QNNSoftmax(X, N);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opSoftmax);
{$ENDIF}
end;

procedure QNNSoftmaxRows(const X: TMemoryBlock; const Rows, Cols: Integer);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opSoftmax);
{$ENDIF}

  case X.DataType of
    dtF32:
      // if x.offset=0 then
      TQNNSingleOps.QNNSoftmaxRows(X, Rows, Cols)
          // else begin
          //  assert(x.size - x.offset>= rows*cols, 'QNNSoftmax : out of bounds');
          //  for i:=0 to Rows-1 do
          //    TQNNSingleOPs.QNNSoftmax(x + i*cols, cols);
          // end
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opSoftmax);
{$ENDIF}
end;

procedure QNNAttention(const Dst, Q, K, V: TMemoryBlock; const Batch, Heads, SeqQ, SeqK, HeadDim: LongInt; const Scale: QNNFloat);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opAttention);
{$ENDIF}

  case Dst.DataType of
    dtF32:
      TQNNSingleOps.QNNAttention(Dst, Q, K, V, Batch, Heads, SeqQ, SeqK, HeadDim, Scale);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opAttention);
{$ENDIF}
end;

procedure QNNFlashAttentionHead(const Dst, Q, K, V: TMemoryBlock; const SeqQ, SeqK, HeadDim: LongInt; const Scale: QNNFloat);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opAttentionHead);
{$ENDIF}

  case Dst.DataType of
    dtF32:
      TQNNSingleOps.QNNFlashAttentionHead(Dst, Q, K, V, SeqQ, SeqK, HeadDim, Scale);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opAttentionHead);
{$ENDIF}
end;

procedure QNNFlashAttentionHeadTiled(const Dst, Q, K, V: TMemoryBlock; const SeqQ, SeqK, HeadDim: LongInt; const Scale: QNNFloat; const TileScores: TMemoryBlock; const QTileSize, KTileSize: LongInt);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opAttentionHeadTiled);
{$ENDIF}

  case Dst.DataType of
    dtF32:
      TQNNSingleOps
        .QNNFlashAttentionHeadTiled(Dst, Q, K, V, SeqQ, SeqK, HeadDim, Scale, TileScores, QTileSize, KTileSize);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opAttentionHeadTiled);
{$ENDIF}
end;

procedure QNNFlashAttention(const Dst, Q, K, V: TMemoryBlock; const SeqQ, SeqK, Heads, HeadDim: LongInt; const Scale: QNNFloat);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opFlashAttention);
{$ENDIF}

  case Dst.DataType of
    dtF32:
      TQNNSingleOps.QNNFlashAttention(Dst, Q, K, V, SeqQ, SeqK, Heads, HeadDim, Scale);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opFlashAttention);
{$ENDIF}
end;

procedure QNNUpSampleNearest(const Dst, Src: TMemoryBlock; const Batch, Channels, H, W, ScaleH, ScaleW: LongInt);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opUpSampleNearest);
{$ENDIF}

  case Dst.DataType of
    dtF32:
      TQNNSingleOps.QNNUpSampleNearest(Dst, Src, Batch, Channels, H, W, ScaleH, ScaleW);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opUpSampleNearest);
{$ENDIF}
end;

procedure QNNUpSample(const Dst, Src: TMemoryBlock; const Batch, Channels, H, W: LongInt; const ScaleH, ScaleW: QNNFloat; const Interpolation: TInterpolation);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opUpSample);
{$ENDIF}

  case Dst.DataType of
    dtF32:
      TQNNSingleOps.QNNUpSample(Dst, Src, Batch, Channels, H, W, ScaleH, ScaleW, Interpolation);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opUpSample);
{$ENDIF}
end;

procedure QNNPatchify(const Dst, Src: TMemoryBlock; const Batch, Channels, H, W, PatchSize: LongInt);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opPatchify);
{$ENDIF}
  Assert(Dst.Size = Src.Size, 'ERROR [QNNPatchify] : Source and Destination sizes mismatch.');
  case Dst.DataType of
    dtF32:
      TQNNSingleOps.QNNPatchify(Dst, Src, Batch, Channels, H, W, PatchSize);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opPatchify);
{$ENDIF}
end;

procedure QNNUnpatchify(const Dst, Src: TMemoryBlock; const Batch, Channels, H, W, PatchSize: LongInt);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opUnPatchify);
{$ENDIF}
  Assert(Dst.Size = Src.Size, 'ERROR [QNNUnPatchify] : Source and Destination sizes mismatch.');
  case Dst.DataType of
    dtF32:
      TQNNSingleOps.QNNUnpatchify(Dst, Src, Batch, Channels, H, W, PatchSize);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opUnPatchify);
{$ENDIF}
end;

procedure QNNPermut(var Dst: TMemoryBlock; const Src: TMemoryBlock; const Axis: TArray<Byte>);
var
  I: Int64;
  Dims: set of Byte;
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opTranspose);
{$ENDIF}
  Dims := [];
  Assert(
    Length(Axis) = Length(Src.Shape),
    'ERROR [QNNPermute] : Axis and Source dimensions dimensions must be the same.'
  );
  if not Dst.IsAssigned() then
  begin
    Dst := TMemoryBlock.Create(Copy(Src.Shape), '');
  end
  else
  begin
    Assert(
      Dst.Size + Dst.Offset >= Src.Offset + Product(Src.Shape),
      'ERROR [QNNPermute] : Destination or Source shapes out of bounds.'
    );
  end;

  for I := 0 to High(Axis) do
  begin
    Assert(
      not (Axis[I] in Dims) and (Axis[I] < Length(Axis)),
      'ERROR [QNNPermute] : axis cannot be out of dimensions boundries or duplicated!'
    );
    Include(Dims, Axis[I]);
  end;

  case Src.DataType of
    dtF32:
      // AltPermut(dst, src, src.shape, axis);
      TQNNSingleOps.QNNPermut(Dst, Src, Axis, Src.Shape, Dst.Shape);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opTranspose);
{$ENDIF}
end;

function QNNPermut(const Src: TMemoryBlock; const Axis: TArray<Byte>): TMemoryBlock;
begin
  QNNPermut(Result, Src, Axis)
end;

procedure QNNCopy(const Dst, Src: TMemoryBlock; const N: Integer);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opCopy);
{$ENDIF}
  Assert(Src.Offset + N <= Src.Size, 'ERROR [QNNCopy] : N is out of source tesnor bounds');
  Assert(Dst.Offset + N <= Dst.Size, 'ERROR [QNNCopy] : N is out of destination tesnor bounds');
  case Dst.DataType of
    dtF32:
      TQNNSingleOps.QNNCopy(Dst, Src, N);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opCopy);
{$ENDIF}
end;

procedure QNNCopyStrided(const Dst: TMemoryBlock; const DstStride: Integer; const Src: TMemoryBlock; const SrcStride: Integer; const N: Integer);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opCopyStrided);
{$ENDIF}
  Assert(Src.Offset + N * SrcStride <= Src.Size, 'ERROR [QNNCopyStrided] : N is out of source tesnor bounds');
  Assert(Dst.Offset + N * DstStride <= Dst.Size, 'ERROR [QNNCopyStrided] : N is out of destination tesnor bounds');
  case Dst.DataType of
    dtF32:
      TQNNSingleOps.QNNCopyStrided(Dst, DstStride, Src, SrcStride, N);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opCopyStrided);
{$ENDIF}
end;

procedure QNNFill(const Dst: TMemoryBlock; const Val: QNNFloat; const N: Integer; const Stride: Integer);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opFill);
{$ENDIF}
  Assert(Dst.Offset + N * Stride <= Dst.Size, 'ERROR [QNNFill] : N is out of destination tesnor bounds');
  case Dst.DataType of
    dtF32:
      TQNNSingleOps.QNNFill(Dst, Val, N, Stride);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opFill);
{$ENDIF}
end;

procedure QNNBroadcast(const Dst: Pointer; const Src; const SrcSize, N: Integer; const Stride: Integer);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opOther);
{$ENDIF}

  raise ENotSupportedException.Create(OP_NOT_IMPL);
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opOther);
{$ENDIF}
end;

procedure QNNGatedAdd(const Dst, Gate, Proj: TMemoryBlock; const Seq, Hidden: Integer);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opFmavv);
{$ENDIF}

  case Dst.DataType of
    dtF32:
      TQNNSingleOps.QNNGatedAdd(Dst, Gate, Proj, Seq, Hidden);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opFmavv);
{$ENDIF}
end;

procedure QNNComputeRoPE(const Dst: TMemoryBlock; const MaxSeq, Dim: LongInt; const Theta: QNNFloat);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opROPE);
{$ENDIF}

  case Dst.DataType of
    dtF32:
      TQNNSingleOps.QNNComputeRoPE(Dst, MaxSeq, Dim, Theta);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opROPE);
{$ENDIF}
end;

procedure QNNComputeRoPE2(const CosOut, SinOut: TMemoryBlock; const MaxSeqLen, HeadDim: LongInt; const Theta: QNNFloat);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opROPE2);
{$ENDIF}

  case CosOut.DataType of
    dtF32:
      TQNNSingleOps.QNNComputeRoPE(CosOut, SinOut, MaxSeqLen, HeadDim, Theta);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opROPE2);
{$ENDIF}
end;

procedure QNNComputeRoPE2D(const CosOut, SinDst: TMemoryBlock; const PatchH, PatchW, Dim: LongInt; const Theta: QNNFloat);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opROPE2D);
{$ENDIF}

  case CosOut.DataType of
    dtF32:
      TQNNSingleOps.QNNComputeRoPE2D(CosOut, SinDst, PatchH, PatchW, Dim, Theta);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opROPE2D);
{$ENDIF}
end;

procedure QNNComputeRoPE2DOffset(const CosOut, SinOut: TMemoryBlock; const PatchH, PatchW, Dim: LongInt; const Theta: QNNFloat; const OffsetT: LongInt);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opROPE2DOffset);
{$ENDIF}

  case CosOut.DataType of
    dtF32:
      TQNNSingleOps.QNNComputeRoPE2DOffset(CosOut, SinOut, PatchH, PatchW, Dim, Theta, OffsetT);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opROPE2DOffset);
{$ENDIF}
end;

procedure QNNComputeRoPEText(const CosOut, SinOut: TMemoryBlock; const TxtSeq, AxisDim: LongInt; const Theta: QNNFloat);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opROPEText);
{$ENDIF}

  case CosOut.DataType of
    dtF32:
      TQNNSingleOps.QNNComputeRoPEText(CosOut, SinOut, TxtSeq, AxisDim, Theta);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opROPEText);
{$ENDIF}
end;

procedure QNNApplyRoPE(const Dst, Freqs: TMemoryBlock; const Batch, Seq, Heads, HeadDim: LongInt);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opApplyROPE);
{$ENDIF}

  case Dst.DataType of
    dtF32:
      TQNNSingleOps.QNNApplyRoPE(Dst, Freqs, Batch, Seq, Heads, HeadDim);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opApplyROPE);
{$ENDIF}
end;

procedure QNNApplyRoPE2(const Dst, CosIn, SinIn: TMemoryBlock; const NumHeads, HeadDim: LongInt);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opApplyROPE2);
{$ENDIF}

  case Dst.DataType of
    dtF32:
      TQNNSingleOps.QNNApplyRoPE2(Dst, CosIn, SinIn, NumHeads, HeadDim);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opApplyROPE2);
{$ENDIF}
end;

procedure QNNApplyRoPE3(const Dst, CosIn, SinIn: TMemoryBlock; const SeqLen, NumHeads, HeadDim: LongInt);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opApplyROPE3);
{$ENDIF}

  case Dst.DataType of
    dtF32:
      TQNNSingleOps.QNNApplyRoPE3(Dst, CosIn, SinIn, SeqLen, NumHeads, HeadDim);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opApplyROPE3);
{$ENDIF}
end;

procedure QNNApplyRoPEQK(const Q, K, CosCache, SinCache: TMemoryBlock; const SeqLen, NumQHeads, NumKvHeads, HeadDim: LongInt);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opApplyROPEQK);
{$ENDIF}

  case Q.DataType of
    dtF32:
      TQNNSingleOps.QNNApplyRoPEQK(Q, K, CosCache, SinCache, SeqLen, NumQHeads, NumKvHeads, HeadDim);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opApplyROPEQK);
{$ENDIF}
end;

procedure QNNApplyRoPE2D(const Dst, CosFreq, SinFreq: TMemoryBlock; const Seq, Heads, HeadDim, Dim: LongInt);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opApplyROPE2D);
{$ENDIF}

  case Dst.DataType of
    dtF32:
      TQNNSingleOps.QNNApplyRoPE2D(Dst, CosFreq, SinFreq, Seq, Heads, HeadDim, Dim);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opApplyROPE2D);
{$ENDIF}
end;

function QNNTimeStepEmbedding(const T: QNNFloat; const Dim: Integer; const MaxPeriod: QNNFloat): TMemoryBlock;
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opTimeStepEmbedding);
{$ENDIF}
  Result := TQNNSingleOps.QNNTimeStepEmbedding(T, Dim, MaxPeriod);
  // OP_IMPL_FAIL()
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opTimeStepEmbedding);
{$ENDIF}
end;

procedure QNNMatTriangularFill(const Dim: Integer; const Dst: TMemoryBlock; const Val: QNNFloat; const Mask: PLongint);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opTriangularFill);
{$ENDIF}

  case Dst.DataType of
    dtF32:
      TQNNSingleOps.QNNMatTriangularFill(Dim, Dst, Val, Mask);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opTriangularFill);
{$ENDIF}
end;

procedure QNNMaskFill(const Dst: TMemoryBlock; const Mask: PLongint; const Val: QNNFloat; const Rows, Cols: LongInt);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opMaskFill);
{$ENDIF}

  case Dst.DataType of
    dtF32:
      TQNNSingleOps.QNNMaskFill(Dst, Mask, Val, Rows, Cols);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opMaskFill);
{$ENDIF}
end;

procedure QNNAdaLN(const Dst, Src, Bias, Scale: TMemoryBlock; const Seq, Hidden: LongInt; const Eps: QNNFloat);
begin
{$IF defined(USE_PROFILING)}
  Metrics.Start(opAdaLN);
{$ENDIF}

  case Dst.DataType of
    dtF32:
      TQNNSingleOps.QNNAdaLN(Dst, Src, Bias, Scale, Seq, Hidden, Eps);
  else
    raise ENotSupportedException.Create(OP_NOT_IMPL)
  end;
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opAdaLN);
{$ENDIF}
end;

procedure QNNLinearBF16OrF32(const Dst, Src: System.PSingle; const Weight: TMemoryBlock; const WeightBF16: TMemoryBlock; const Seq, SrcDim, DstDim: Integer);
begin
  if WeightBF16.IsAssigned then
  begin
    if TryVulkanLinear(Dst, Src, WeightBF16, Seq, SrcDim, DstDim) then
      Exit;
  end
  else if TryVulkanLinear(Dst, Src, Weight, Seq, SrcDim, DstDim) then
    Exit;
{$IF defined(USE_PROFILING)}
  Metrics.Start(opGemm);
{$ENDIF}
  TQNNSingleOps.QNNLinearBF16OrF32(Dst, Src, Weight, WeightBF16, Seq, SrcDim, DstDim);
  // OP_IMPL_FAIL()
{$IF defined(USE_PROFILING)}
  Metrics.Finish(opGemm);
{$ENDIF}
end;

end.

