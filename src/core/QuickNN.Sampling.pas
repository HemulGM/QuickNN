unit QuickNN.Sampling;

{$C+} // enable assertions
{$H+} // longstrings
{$POINTERMATH on} // manipulate, inc, dec, cast pointers
{$T+} // typed pointer when @ is used
{$R+} // raise an error when trying to access arrays out of their bounds
interface

uses
  System.SysUtils, System.Math, QuickNN.Common;

function ScheduleLinear(const NumSteps: LongInt): TMemoryBlock;

function SchedulePower(const NumSteps: LongInt; const Alpha: QNNFloat): TMemoryBlock;

function ScheduleSigmoid(const NumSteps: LongInt; const Shift: QNNFloat): TMemoryBlock;

function ScheduleResolution(const NumSteps, Height, Width: LongInt): TMemoryBlock;

function ComputeEmpiricalMu(const ImageSeqLen, NumSteps: LongInt): Single;

function GeneralizedTimeSnrShift(const T, Mu, Sigma: QNNFloat): QNNFloat;

function ScheduleFlux(const NumSteps, ImageSeqLen: LongInt): TMemoryBlock;

function ScheduleZimage(const NumSteps, ImageSeqLen: LongInt): TMemoryBlock;

function InitNoise(const Batch, Channels, H, W: LongInt; const Seed: Int64): TMemoryBlock;

procedure RngSeed(Seed: UInt64);

function SelectedSchedule(const Params: TGenerateParams; const ImageSeqLen: LongInt): TMemoryBlock;

implementation

function ScheduleLinear(const NumSteps: LongInt): TMemoryBlock;
var
  I: LongInt;
  ResultPtr: PQNNFloat;
begin
  Result := TMemoryBlock.Create([NumSteps + 1], 'SCHEDULER_LINEAR');
  ResultPtr := Result;
  for I := 0 to NumSteps do
    ResultPtr[I] := 1.0 - I / NumSteps;
end;

(*
 * Power schedule: denser steps at the start (high noise), sparser at the end.
 * schedule[i] = 1 - (i/n)^alpha
 * alpha=1.0 is linear, alpha=2.0 is quadratic, etc.
 *)

function SchedulePower(const NumSteps: LongInt; const Alpha: QNNFloat): TMemoryBlock;
var
  I: LongInt;
  ResultPtr: PQNNFloat;
begin
  Result := TMemoryBlock.Create([NumSteps + 1], 'SCHEDULER_POWER');
  ResultPtr := Result;
  for I := 0 to NumSteps do
    ResultPtr[I] := 1.0 - Power(I / NumSteps, Alpha);
end;

(*
 * Shifted sigmoid schedule (better for flow matching)
 * shift controls where the inflection point is
 *)
function ScheduleSigmoid(const NumSteps: LongInt; const Shift: QNNFloat): TMemoryBlock;
var
  I: LongInt;
  T, X: QNNFloat;
  ResultPtr: PQNNFloat;
begin
  Result := TMemoryBlock.Create([NumSteps + 1], 'SCHEDULER_SIGMOID');
  ResultPtr := Result;
  for I := 0 to NumSteps do
  begin
    T := I / NumSteps;
    // Shifted sigmoid: more steps at the end
    X := (T - 0.5) * 10.0 + Shift;
    ResultPtr[I] := 1.0 - 1.0 / (1.0 + Exp(-X));
  end;
  // Ensure endpoints
  ResultPtr[0] := 1.0;
  ResultPtr[NumSteps] := 0.0;
end;

(*
 * Resolution-dependent schedule (as used in FLUX.2)
 * Higher resolutions use more steps at the start
*)
function ScheduleResolution(const NumSteps, Height, Width: LongInt): TMemoryBlock;
var
  I, Pixels: LongInt;
  ResultPtr: PQNNFloat;
  Shift, T: QNNFloat;
begin
  Result := TMemoryBlock.Create([NumSteps + 1], 'SCHEDULER_RESOLUTION');
  ResultPtr := Result;
  Pixels := Height * Width;
  Shift := 0.0;
  if Pixels >= 1024 * 1024 then
    Shift := 1.0
  else if Pixels >= 512 * 512 then
    Shift := 0.5;
  for I := 0 to NumSteps do
  begin
    T := Power(I / NumSteps, 1.0 + Shift * 0.5);
    ResultPtr[I] := 1.0 - T
  end
end;

(*
 * FLUX.2 official schedule with empirical mu calculation
 * Matches Python's get_schedule() function from official flux2 code
 *
 * Compute the empirical shift parameter mu for the resolution-dependent
 * noise schedule. The constants a1, b1, a2, b2 are fitted from the Flux
 * training distribution and control how the SNR schedule adapts to different
 * image resolutions. Higher resolution images need more denoising steps
 * at high noise levels. Interpolates between two linear fits based on
 * step count, with a cutoff at 4300 tokens. *)

function ComputeEmpiricalMu(const ImageSeqLen, NumSteps: LongInt): Single;
var
  A1, A2, B1, B2, M200, M10, A, B: QNNFloat;
begin
  A1 := 8.73809524e-05;
  B1 := 1.89833333;
  A2 := 0.00016927;
  B2 := 0.45666666;
  if ImageSeqLen > 4300 then
    Exit(A2 * ImageSeqLen + B2);
  M200 := A2 * ImageSeqLen + B2;
  M10 := A1 * ImageSeqLen + B1;
  A := (M200 - M10) / 190.0;
  B := M200 - 200.0 * A;
  Exit(A * NumSteps + B)
end;

(* Apply exponential SNR (Signal-to-Noise Ratio) shift to a timestep.
 * Maps t in [0,1] through t/(t + (1-t)*exp(-mu)), shifting the schedule
 * toward more time spent at higher noise levels. The boundary guards
 * (t<=0, t>=1) prevent division by zero. *)
function GeneralizedTimeSnrShift(const T, Mu, Sigma: QNNFloat): QNNFloat;
begin
  if (T <= 0.0) then
    Exit(0.0);
  if T >= 1.0 then
    Exit(1.0);
  Exit(Exp(Mu) / (Exp(Mu) + Power(1.0 / T - 1.0, Sigma)))
end;

function ScheduleFlux(const NumSteps, ImageSeqLen: LongInt): TMemoryBlock;
var
  ResultPtr: PQNNFloat;
  T, Mu: QNNFloat;
  I: LongInt;
begin
  Result := TMemoryBlock.Create([NumSteps + 1], 'SCHEDULER_FLUX');
  ResultPtr := Result;
  Mu := ComputeEmpiricalMu(ImageSeqLen, NumSteps);
  for I := 0 to NumSteps do
  begin
    T := 1.0 - I / NumSteps;
    ResultPtr[I] := GeneralizedTimeSnrShift(T, Mu, 1.0)
  end
end;

(*
 * Z-Image schedule: FlowMatchEulerDiscreteScheduler with static shift.
 *
 * Matches diffusers FlowMatchEulerDiscreteScheduler.set_timesteps()
 * for use_dynamic_shifting=False, invert_sigmas=False.
 *
 * Diffusers behavior (with num_train_timesteps=1000, shift=3.0):
 * 1) Build training sigmas from [1.0 .. 1/1000], then shift once.
 * 2) For inference, linspace from sigma_max to sigma_min (already shifted).
 * 3) Shift again.
 * 4) Append terminal sigma=0.
 *
 * Returns array of num_steps+1 sigma values.
 * The sampling loop uses these as sigmas directly with Euler:
 *   dt = sigma_next - sigma (negative for denoising)
 *   The transformer timestep = (1 - sigma), range [0, 1]
 *)

function ScheduleZimage(const NumSteps, ImageSeqLen: LongInt): TMemoryBlock;
var
  I: LongInt;
  ResultPtr: PQNNFloat;
  Shift, SigmaMax, SigmaMin, SigmaTrainMin, U, Raw: QNNFloat;
begin
  Result := TMemoryBlock.Create([NumSteps + 1], 'SCHEDULER_ZIMAGE');
  ResultPtr := Result;
  Shift := 3.0;
  SigmaMax := 1.0;
  SigmaTrainMin := 1.0 / 1000.0; (* num_train_timesteps=1000 *)
  SigmaMin := Shift * SigmaTrainMin / (1.0 + (Shift - 1.0) * SigmaTrainMin);

  (* Diffusers set_timesteps(): linspace(sigma_max, sigma_min, num_steps),
      * then apply static shift one more time. *)
  for I := 0 to NumSteps - 1 do
  begin
    if NumSteps > 1 then
      U := I / (NumSteps - 1)
    else
      U := 0.0;
    Raw := SigmaMax + (SigmaMin - SigmaMax) * U;
    ResultPtr[I] := Shift * Raw / (1.0 + (Shift - 1.0) * Raw);
  end;
  (* Diffusers appends terminal sigma=0 (invert_sigmas=False). *)
  ResultPtr[NumSteps] := 0.0;
end;

var
  RngState: array[0..3] of UInt64 = (UInt64($853c49e6748fea9b), UInt64($da3e39cb94b95bdb), UInt64($647c4677a2884327), UInt64($c6e7918d2e2969f5));

function Rotl(const X: UInt64; const K: LongInt): UInt64; inline;
begin
  Result := (X shl K) or (X shr (64 - K))
end;

{$R-} // disable range check
{$Q-} // disable arithmetic overflow check
function Xoshiro256ss(): UInt64;
var
  T: UInt64;
begin
  Result := Rotl(RngState[1] * 5, 7) * 9;
  T := RngState[1] shl 17;
  RngState[2] := RngState[2] xor RngState[0];
  RngState[3] := RngState[3] xor RngState[1];
  RngState[1] := RngState[1] xor RngState[2];
  RngState[0] := RngState[0] xor RngState[3];
  RngState[2] := RngState[2] xor T;
  RngState[3] := Rotl(RngState[3], 45);
end;

procedure RngSeed(Seed: UInt64);
var
  I: LongInt;
  Z: UInt64;
begin
  (* SplitMix64 to initialize state from seed *)
  for I := 0 to 3 do
  begin
    Seed := Seed + UInt64($9e3779b97f4a7c15);
    Z := Seed;
    Z := (Z xor (Z shr 30)) * UInt64($bf58476d1ce4e5b9);
    Z := (Z xor (Z shr 27)) * UInt64($94d049bb133111eb);
    RngState[I] := Z xor (Z shr 31);
  end;
end;
{$Q+}
{$R+}

function Rand: Single;
begin
  Result := (Xoshiro256ss() shr 11) * (1.0 / 9007199254740992.0);
end;

function RandomNormal: Single;
var
  U1, U2: QNNFloat;
begin
  (* Box-Muller transform *)
  U1 := Rand();
  U2 := Rand();
  (* Avoid log(0) *)
  while U1 = 0.0 do
    U1 := Rand();
  Result := Sqrt(-2.0 * Ln(U1)) * Cos(2.0 * 3.14159265358979323846 * U2);
end;

procedure Randn(const Dst: PQNNFloat; const N: LongInt);
var
  I: LongInt;
begin
  for I := 0 to N - 1 do
    Dst[I] := RandomNormal();
end;

function InitNoise(const Batch, Channels, H, W: LongInt; const Seed: Int64): TMemoryBlock;
const
  NOISE_MAX_LATENT_DIM = 1792 div 16; (* 1792/16 = 112 *)
var
  TargetSize, Sy, Sx, SrcIdx, DstIdx, B, Ty, Tx, MaxH, MaxW, MaxSize, C: LongInt;
  MaxNoise: TMemoryBlock;
  ResultPtr, MaxNoisePtr: PQNNFloat;
begin
  TargetSize := Batch * Channels * H * W;
  Result := TMemoryBlock.Create([Batch, Channels, H, W], 'INIT_NOISE_RESULT');
  ResultPtr := Result;

  if Seed >= 0 then
    RngSeed(Seed);

  (* If target is max size or larger, just generate directly *)
  if (H >= NOISE_MAX_LATENT_DIM) and (W >= NOISE_MAX_LATENT_DIM) then
  begin
    Randn(Result, TargetSize);
    Exit;
  end;

  (* Generate result at max latent size *)
  MaxH := NOISE_MAX_LATENT_DIM;
  MaxW := NOISE_MAX_LATENT_DIM;
  MaxSize := Batch * Channels * MaxH * MaxW;
  MaxNoise := TMemoryBlock.Create([MaxSize], 'INIT_NOISE_MAX_NOISE');
  MaxNoisePtr := MaxNoise;
  Randn(MaxNoisePtr, MaxSize);

  (* Subsample to target size using nearest-neighbor *)
  for B := 0 to Batch - 1 do
    for C := 0 to Channels - 1 do
      for Ty := 0 to H - 1 do
        for Tx := 0 to W - 1 do
        begin
          (* Map target position to source position *)
          Sy := Ty * MaxH div H;
          Sx := Tx * MaxW div W;

          SrcIdx := ((B * Channels + C) * MaxH + Sy) * MaxW + Sx;
          DstIdx := ((B * Channels + C) * H + Ty) * W + Tx;
          ResultPtr[DstIdx] := MaxNoisePtr[SrcIdx];
        end;
  MaxNoise.Free;
end;

function SelectedSchedule(const Params: TGenerateParams; const ImageSeqLen: LongInt): TMemoryBlock;
begin
  case Params.Schedule of
    QNN_SCHEDULE_LINEAR:
      Exit(ScheduleLinear(Params.NumSteps));
    QNN_SCHEDULE_POWER:
      Exit(SchedulePower(Params.NumSteps, Params.PowerAlpha));
    QNN_SCHEDULE_FLOWMATCH:
      Exit(ScheduleZimage(Params.NumSteps, ImageSeqLen));
    QNN_SCHEDULE_SIGMOID:
      Exit(ScheduleFlux(Params.NumSteps, ImageSeqLen));
  else
    Exit(ScheduleFlux(Params.NumSteps, ImageSeqLen));
  end;
end;

initialization

end.

