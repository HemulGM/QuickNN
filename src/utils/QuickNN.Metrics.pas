unit QuickNN.Metrics;

interface

type

{$IF not declared(SizeInt)}
  SizeInt = IntPtr;
{$ENDIF}
  TMeasureOps = (
      opAllocCPU,
      opIncFill,
      opFill,
      opCopy,
      opCopyStrided,
      opMaxNorm,
      opStdDevNorm,
      opRMSNorm,
      opBatchAddvs,
      opAddvs,
      opBatchMulvs,
      opGroupNorm,
      opBatchNorm,
      opMulvs,
      opBatchSubvs,
      opSubvs,
      opBatchDivvs,
      opDivvs,
      opPow,
      opAxpy,
      opMulvv,
      opAddvv,
      opSubvv,
      opDivvv,
      opROPE,
      opROPE2,
      opROPE2D,
      opROPEText,
      opROPEOffset,
      opROPE2DOffset,
      opApplyROPE,
      opApplyROPE2,
      opApplyROPE3,
      opApplyROPEQK,
      opApplyROPE2D,
      opFmavv,
      opDot,
      opBatchFmavss,
      opFmavss,
      opFmavvs,
      opGemm,
      opIm2col,
      opCol2im,
      opTimeStepEmbedding,
      opTriangularFill,
      opMaskFill,
      opAdaLN,
      opConv2D,
      opConcat,
      opAddConcat,
      opHostToDevice,
      opDeviceToHost,
      opMemAllocate,
      opMemRelease,
      opGPUAllocate,
      opArgMax,
      opArgMin,
      opConvert,
      opQuantize,
      opDequantize,
      opAttention,
      opAttentionHead,
      opAttentionHeadTiled,
      opFlashAttention,
      opGPURelease,
      opIm2ColExt,
      opCol2ImExt,
      opMeans,
      opVariances,
      opSoftmax,
      opSoftmaxInplace,
      opSiLU,
      opSiLUInplace,
      opSiLUMul,
      opCrossEntropySoftmax,
      opMeansVars,
      opNormalize,
      opMeansVarsDelta,
      opNormalizeDelta,
      opAddDots,
      opForwardBias,
      opBackwardBias,
      opForwardScale,
      opForwardScaleAdd,
      opFwMaxpool,
      opBwMaxpool,
      opReduce,
      opFwDropout,
      opBwDropout,
      opL2,
      opClip,
      opInvSqrt,
      opMin,
      opMax,
      opSumAbsDiff,
      opSum,
      opSumSqrDiff,
      opSumSqrDiffScaler,
      opDistance,
      opSumSqr,
      opMaxAbs,
      opMinAbs,
      opMinAndMax,
      opInit,
      opDispose,
      opActivate,
      opDerive,
      opSigmoid,
      opSigmoidInplace,
      opPatchify,
      opUnPatchify,
      opTranspose,
      opUpSample,
      opUpSampleNearest,
      opUpSampleCubic,
      opMaxAbsDiff,
      opAbsDiff,
      opAbsDiffScaler,
      opAbsSum,
      opOther,
      opTanh
  );

  { TTensorOps }
  PTensorMetrics = ^TTensorMetrics;

  { TTensorMetrics }

  TTensorMetrics = record
  private
    M: array[0..999] of Int64;
    Stack: LongInt;
    function GetItem(I: TMeasureOps): Int64;
  public
    Elapsed: array[Low(TMeasureOps)..High(TMeasureOps)] of Int64;
    Counts: array[Low(TMeasureOps)..High(TMeasureOps)] of Int64;
    // stackArray : TArray<TMeasureOps>;
    procedure Start(const A: TMeasureOps);
    procedure Finish(const A: TMeasureOps);
    procedure Reset();
    procedure Print();
    function Total(): Int64;
    property Item[I: TMeasureOps]: Int64 read GetItem; default;
    class operator Initialize(out Dst: TTensorMetrics);
  end;

implementation
uses
  QuickNN.Timing,
  System.SysUtils,
  System.TypInfo;

{ TTensorMetric }

function TTensorMetrics.GetItem(I: TMeasureOps): Int64;
begin
  Result := Elapsed[I]
end;

procedure TTensorMetrics.Start(const A: TMeasureOps);
begin
  M[Stack] := Clock;
  Inc(Counts[A]);
  Inc(Stack);
  // insert(a, stackArray, length(stackArray))
end;

procedure TTensorMetrics.Finish(const A: TMeasureOps);
begin
  Dec(Stack);
  Dec(Counts[A]);
  Elapsed[A] := Elapsed[A] + Clock() - M[Stack];
  // delete(stackArray, high(stackArray), 1)
end;

procedure TTensorMetrics.Reset();
begin
  FillChar(Self, SizeOf(Self), #0)
end;

procedure TTensorMetrics.Print();
var
  T: Int64;
  I: TMeasureOps;
  S: string;
begin
  T := 0;
  for I := Low(TMeasureOps) to High(TMeasureOps) do
    if Elapsed[I] <> 0 then
    begin
      Inc(T, Elapsed[I]);
      S := GetEnumName(TypeInfo(TMeasureOps), Ord(I));
      WriteLn(Format('%-20s', [Copy(S, 3)]), Elapsed[I] / CLOCKS_PER_MS: 10: 3, 'ms')
    end;
  WriteLn('--------------------------------');
  WriteLn(Format('%-20s', ['total ']): 20, T / CLOCKS_PER_MS: 10: 3, 'ms')

end;

function TTensorMetrics.Total(): Int64;
var
  I: TMeasureOps;
begin
  Result := 0;
  for I := Low(TMeasureOps) to High(TMeasureOps) do
    Inc(Result, Elapsed[I])
end;

class operator TTensorMetrics.Initialize(out Dst: TTensorMetrics);
begin
  FillChar(Dst, SizeOf(Dst), #0)
end;

end.
