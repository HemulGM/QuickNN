unit QuickNN.Models.Qwen21.Math;

{ SPDX-License-Identifier: Apache-2.0
  Native Pascal implementation; architecture and tensor conventions follow
  Qwen-Image 2.1 / Diffusers and ComfyUI / comfy-kitchen (Apache-2.0).
  Reference authors: Qwen-Image Team, Hugging Face Team and Comfy Org contributors.
  https://github.com/huggingface/diffusers/tree/main/src/diffusers/models
  https://github.com/Comfy-Org/ComfyUI/tree/master/comfy/ldm/qwen_image21
  https://github.com/Comfy-Org/comfy-kitchen/blob/main/comfy_kitchen/tensor/int8_utils.py
  See LICENSE.QwenImage21.txt in the repository root. }
{$POINTERMATH ON}
interface

uses
  QuickNN.Common, QuickNN.Vulkan;

type
  TQwen21Status = reference to procedure(const Message: string);

procedure Q21GPURMS(const Ctx: TQNNGPUContext; const Dst, Src: TQNNGPUTensor; const Gamma: TMemoryBlock; const Rows, Dim: Integer; const ZeroCentered: Boolean = False; const Chw: Boolean = False
);

procedure Q21GPUAffine(const Ctx: TQNNGPUContext; const Dst, Src: TQNNGPUTensor; const Gamma, Beta: TMemoryBlock; const Rows, Dim: Integer);

procedure Q21GPUAdd(const Ctx: TQNNGPUContext; const Dst, Src: TQNNGPUTensor; const Count: Integer; const Scale: Single = 1
);

procedure Q21RMS(const Dst, Src, Gamma: TMemoryBlock; const Rows, Dim: Integer; const ZeroCentered: Boolean = False);

procedure Q21HeadNorm(const Data, Gamma: TMemoryBlock; const Rows, Heads, Dim: Integer);

procedure Q21TextRoPE(const Data: TMemoryBlock; const Rows, Heads, Dim: Integer; const Theta: Double; const PositionIds: TArray<Integer> = nil
);

procedure Q21ImageRoPE(const Data: TMemoryBlock; const Rows, Heads, TextRows, Height, Width: Integer; const PositionIds: TArray<Single> = nil
);

procedure Q21Attention(const Dst, Q, K, V: TMemoryBlock; const Rows, Heads, KvHeads, HeadDim, CausalRows: Integer; const KeyLimits: TArray<Integer> = nil
);

procedure Q21ModNorm(const Dst, Src, Scales: TMemoryBlock; const Rows, Dim, PrefixRows, ScaleOffset: Integer);

procedure Q21GateAdd(const Dst, Src, Modulation: TMemoryBlock; const Rows, Dim, PrefixRows, GateOffset: Integer);

procedure Q21Silu(const Data: TMemoryBlock; const Count: Integer);

procedure Q21Gelu(const Data: TMemoryBlock; const Count: Integer);

function Q21Schedule(const Steps, ImageTokens: Integer; const ConfigPath: string): TArray<Single>;

implementation

uses
  System.SysUtils, System.Math, System.JSON, System.IOUtils,
  QuickNN.Quantization, QuickNN.CPU;

procedure Q21GPURMS(const Ctx: TQNNGPUContext; const Dst, Src: TQNNGPUTensor; const Gamma: TMemoryBlock; const Rows, Dim: Integer; const ZeroCentered, Chw: Boolean);
begin
  Ctx.KernelWeights(
    GO_NORM,
    Rows * Dim,
    Dst,
    Src,
    Gamma,
    Default(TMemoryBlock),
    [Rows, Dim, Ord(Chw) * 2 + Ord(ZeroCentered) * 4, 0, 0, 0, Ord(not Gamma.IsAssigned)]
  )
end;

procedure Q21GPUAffine(const Ctx: TQNNGPUContext; const Dst, Src: TQNNGPUTensor; const Gamma, Beta: TMemoryBlock; const Rows, Dim: Integer);
begin
  Ctx.KernelWeights(GO_NORM, Rows * Dim, Dst, Src, Gamma, Beta, [Rows, Dim, 1, 0, 0, 0, 0])
end;

procedure Q21GPUAdd(const Ctx: TQNNGPUContext; const Dst, Src: TQNNGPUTensor; const Count: Integer; const Scale: Single);
begin
  Ctx.Kernel(GO_ADD, Count, Dst, Src, nil, nil, [Count, Count, QNNGPUFloatBits(Scale), 0, 0])
end;

procedure Q21RMS(const Dst, Src, Gamma: TMemoryBlock; const Rows, Dim: Integer; const ZeroCentered: Boolean);
var
  Sum: Double;
  Inv, Weight: Single;
  Offset: NativeInt;
begin
  for var R := 0 to Rows - 1 do
  begin
    Offset := NativeInt(R) * Dim;
    Sum := 0;
    for var J := 0 to Dim - 1 do
      Sum := Sum + Sqr(Double(System.PSingle(Src)[Offset + J]));
    Inv := 1 / Sqrt(Sum / Dim + 1E-6);
    for var J := 0 to Dim - 1 do
    begin
      Weight := QNNWeightValue(Gamma, J);
      if ZeroCentered then
        Weight := Weight + 1;
      System.PSingle(Dst)[Offset + J] := System.PSingle(Src)[Offset + J] * Inv * Weight;
    end;
  end;
end;

procedure Q21HeadNorm(const Data, Gamma: TMemoryBlock; const Rows, Heads, Dim: Integer);
begin
  Q21RMS(Data, Data, Gamma, Rows * Heads, Dim)
end;

procedure Q21TextRoPE(const Data: TMemoryBlock; const Rows, Heads, Dim: Integer; const Theta: Double; const PositionIds: TArray<Integer>);
var
  Angle, Cs, Sn: Double;
  A, B: Single;
  Base: NativeInt;
  Axis, Position: Integer;
begin
  if (Length(PositionIds) <> 0) and ((Length(PositionIds) <> 3 * Rows) or (Dim <> 128)) then
    raise EArgumentException.Create('Invalid interleaved mRoPE positions');
  for var R := 0 to Rows - 1 do
    for var J := 0 to Dim div 2 - 1 do
    begin
      Position := R;
      if Length(PositionIds) > 0 then
      begin
        Axis := 0;
        if J < 60 then
          Axis := J mod 3;
        Position := PositionIds[Axis * Rows + R];
      end;
      Angle := Position / Power(Theta, 2 * J / Dim);
      Cs := Cos(Angle);
      Sn := Sin(Angle);
      for var H := 0 to Heads - 1 do
      begin
        Base := (NativeInt(R) * Heads + H) * Dim + J;
        A := System.PSingle(Data)[Base];
        B := System.PSingle(Data)[Base + Dim div 2];
        System.PSingle(Data)[Base] := A * Cs - B * Sn;
        System.PSingle(Data)[Base + Dim div 2] := A * Sn + B * Cs;
      end;
    end;
end;

procedure Q21ImageRoPE(const Data: TMemoryBlock; const Rows, Heads, TextRows, Height, Width: Integer; const PositionIds: TArray<Single>);
const
  DIMS: array[0..2] of Integer = (16, 56, 56);
var
  Position, Angle, Cs, Sn: Double;
  A, B: Single;
  Base: NativeInt;
  Start: Integer;
begin
  if Length(PositionIds) > 0 then
  begin
    if Length(PositionIds) <> Rows * 3 then
      raise EArgumentException.Create('Invalid joint RoPE positions')
  end
  else if Rows <> TextRows + Height * Width then
    raise EArgumentException.Create('Invalid image RoPE grid');
  for var R := 0 to Rows - 1 do
  begin
    Start := 0;
    for var Axis := 0 to 2 do
    begin
      if Length(PositionIds) > 0 then
        Position := PositionIds[R * 3 + Axis]
      else if R < TextRows then
        Position := R
      else
        case Axis of
          0:
            Position := TextRows;
          1:
            Position := (R - TextRows) div Width - (Height - Height div 2);
        else
          Position := (R - TextRows) mod Width - (Width - Width div 2);
        end;
      for var J := 0 to DIMS[Axis] div 2 - 1 do
      begin
        Angle := Position / Power(10000, 2 * J / DIMS[Axis]);
        Cs := Cos(Angle);
        Sn := Sin(Angle);
        for var H := 0 to Heads - 1 do
        begin
          Base := (NativeInt(R) * Heads + H) * 128 + Start + 2 * J;
          A := System.PSingle(Data)[Base];
          B := System.PSingle(Data)[Base + 1];
          System.PSingle(Data)[Base] := A * Cs - B * Sn;
          System.PSingle(Data)[Base + 1] := A * Sn + B * Cs;
        end;
      end;
      Inc(Start, DIMS[Axis]);
    end;
  end;
end;

procedure Q21Attention(const Dst, Q, K, V: TMemoryBlock; const Rows, Heads, KvHeads, HeadDim, CausalRows: Integer; const KeyLimits: TArray<Integer>);
var
  Scores: TMemoryBlock;
  Qp, Kp, Vp, Op, Sp: System.PSingle;
  Maximum, Sum: Double;
  KeyHead, KeyEnd: Integer;
begin
  if (Length(KeyLimits) <> 0) and (Length(KeyLimits) <> Rows) then
    raise EArgumentException.Create('Invalid attention mask rows');
  for var Limit in KeyLimits do
    if (Limit < 1) or (Limit > Rows) then
      raise EArgumentException.Create('Invalid attention key limit');
  if TryVulkanAttention(Dst, Q, K, V, Rows, Heads, KvHeads, HeadDim, CausalRows, KeyLimits) then
    Exit;
  Scores := TMemoryBlock.Create([Rows, Rows], 'qwen21-attention-scores');
  try
    Sp := Scores;
    for var H := 0 to Heads - 1 do
    begin
      KeyHead := H div (Heads div KvHeads);
      Qp := System.PSingle(Q) + H * HeadDim;
      Kp := System.PSingle(K) + KeyHead * HeadDim;
      Vp := System.PSingle(V) + KeyHead * HeadDim;
      Op := System.PSingle(Dst) + H * HeadDim;
      // Small per-head products stay on the native CPU BLAS path: forcing
      // strided views through Vulkan would upload unused neighbouring heads.
      TQNNSingleOps.CblasCPUGemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasTrans,
        Rows,
        Rows,
        HeadDim,
        1 / Sqrt(HeadDim),
        Pointer(Qp),
        Heads * HeadDim,
        Pointer(Kp),
        KvHeads * HeadDim,
        0,
        Pointer(Sp),
        Rows
      );
      for var R := 0 to Rows - 1 do
      begin
        if Length(KeyLimits) > 0 then
          KeyEnd := KeyLimits[R]
        else if R < CausalRows then
          KeyEnd := R + 1
        else
          KeyEnd := Rows;
        Maximum := NegInfinity;
        for var J := 0 to Rows - 1 do
        begin
          if J >= KeyEnd then
            Sp[NativeInt(R) * Rows + J] := NegInfinity;
          Maximum := Max(Maximum, Sp[NativeInt(R) * Rows + J]);
        end;
        Sum := 0;
        for var J := 0 to Rows - 1 do
        begin
          if J >= KeyEnd then
            Sp[NativeInt(R) * Rows + J] := 0
          else
            Sp[NativeInt(R) * Rows + J] := Exp(Sp[NativeInt(R) * Rows + J] - Maximum);
          Sum := Sum + Sp[NativeInt(R) * Rows + J];
        end;
        for var J := 0 to Rows - 1 do
          Sp[NativeInt(R) * Rows + J] := Sp[NativeInt(R) * Rows + J] / Sum;
      end;
      TQNNSingleOps.CblasCPUGemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        Rows,
        HeadDim,
        Rows,
        1,
        Pointer(Sp),
        Rows,
        Pointer(Vp),
        KvHeads * HeadDim,
        0,
        Pointer(Op),
        Heads * HeadDim
      );
    end;
  finally
    Scores.Free
  end;
end;

procedure Q21ModNorm(const Dst, Src, Scales: TMemoryBlock; const Rows, Dim, PrefixRows, ScaleOffset: Integer);
var
  Sum, Mean, Variance: Double;
  Inv, Scale: Single;
  Off, ModOff: NativeInt;
begin
  for var R := 0 to Rows - 1 do
  begin
    Off := NativeInt(R) * Dim;
    Sum := 0;
    for var J := 0 to Dim - 1 do
      Sum := Sum + System.PSingle(Src)[Off + J];
    Mean := Sum / Dim;
    Variance := 0;
    for var J := 0 to Dim - 1 do
      Variance := Variance + Sqr(System.PSingle(Src)[Off + J] - Mean);
    Inv := 1 / Sqrt(Variance / Dim + 1E-6);
    ModOff := ScaleOffset;
    if R < PrefixRows then
      Inc(ModOff, 4 * Dim);
    for var J := 0 to Dim - 1 do
    begin
      Scale := System.PSingle(Scales)[ModOff + J];
      System.PSingle(Dst)[Off + J] := (System.PSingle(Src)[Off + J] - Mean) * Inv * (1 + Scale);
    end;
  end;
end;

procedure Q21GateAdd(const Dst, Src, Modulation: TMemoryBlock; const Rows, Dim, PrefixRows, GateOffset: Integer);
var
  Off, ModOff: NativeInt;
  Gates: TArray<Single>;
begin
  // Gates are shared across tokens. Evaluate tanh twice per channel, rather
  // than once per channel for every text/image token in every block.
  SetLength(Gates, 2 * Dim);
  for var J := 0 to Dim - 1 do
  begin
    Gates[J] := Tanh(System.PSingle(Modulation)[GateOffset + J]);
    Gates[Dim + J] := Tanh(System.PSingle(Modulation)[GateOffset + 4 * Dim + J]);
  end;
  for var Row := 0 to Rows - 1 do
  begin
    Off := NativeInt(Row) * Dim;
    if Row < PrefixRows then
      ModOff := Dim
    else
      ModOff := 0;
    for var J := 0 to Dim - 1 do
      System.PSingle(Dst)[Off + J] := System.PSingle(Dst)[Off + J] + Gates[ModOff + J] * System.PSingle(Src)[Off + J];
  end;
end;

var
  SiluTable: array[0..10240] of Single;

procedure Q21Silu(const Data: TMemoryBlock; const Count: Integer);
var
  X, Fraction: Single;
  Position: Double;
  Index: Integer;
begin
  // Linear interpolation at 1/256 spacing on [-20,20]. The interpolation
  // error is below 1e-6 before F32 rounding; tails contribute below 4.2e-8.
  // This avoids a transcendental call per activation without the much coarser
  // approximation used by the legacy AVX2 exp/reciprocal implementation.
  for var I := 0 to Count - 1 do
  begin
    X := System.PSingle(Data)[I];
    if X >= 20 then
      System.PSingle(Data)[I] := X
    else if X <= -20 then
      System.PSingle(Data)[I] := 0
    else
    begin
      Position := Double(X) * 256 + 5120;
      Index := Trunc(Position);
      Fraction := Position - Index;
      System.PSingle(Data)[I] := SiluTable[Index] + Fraction * (SiluTable[Index + 1] - SiluTable[Index]);
    end;
  end;
end;

procedure Q21Gelu(const Data: TMemoryBlock; const Count: Integer);
var
  X: Single;
begin
  for var I := 0 to Count - 1 do
  begin
    X := System.PSingle(Data)[I];
    System.PSingle(Data)[I] := 0.5 * X * (1 + Tanh(Sqrt(2 / Pi) * (X + 0.044715 * X * X * X)));
  end;
end;

function Q21Schedule(const Steps, ImageTokens: Integer; const ConfigPath: string): TArray<Single>;
var
  JSON: TJSONValue;
  BaseTokens, MaxTokens: Integer;
  BaseShift, MaxShift, Mu, T, Last, Stretch, Terminal: Double;
begin
  if (Steps < 1) or (Steps > 1000) then
    raise EArgumentException.Create('Steps must be between 1 and 1000');
  JSON := TJSONObject.ParseJSONValue(TFile.ReadAllText(ConfigPath));
  if JSON = nil then
    raise EArgumentException.Create('Invalid scheduler config');
  try
    BaseTokens := JSON.GetValue<Integer>('base_image_seq_len', 256);
    MaxTokens := JSON.GetValue<Integer>('max_image_seq_len', 8192);
    BaseShift := JSON.GetValue<Double>('base_shift', 0.5);
    MaxShift := JSON.GetValue<Double>('max_shift', 0.9);
    Terminal := JSON.GetValue<Double>('shift_terminal', 0.02);
    if (MaxTokens <= BaseTokens) or (Terminal < 0) or (Terminal >= 1) then
      raise EArgumentException.Create('Invalid scheduler parameters');
    Mu := BaseShift + (ImageTokens - BaseTokens) * (MaxShift - BaseShift) / (MaxTokens - BaseTokens);
    SetLength(Result, Steps + 1);
    for var I := 0 to Steps - 1 do
    begin
      T := 1 - I / Steps;
      Result[I] := Exp(Mu) / (Exp(Mu) + (1 / T - 1));
    end;
    Last := Result[Steps - 1];
    Stretch := (1 - Last) / (1 - Terminal);
    if Stretch > 0 then
      for var I := 0 to Steps - 1 do
        Result[I] := 1 - (1 - Result[I]) / Stretch;
    Result[Steps] := 0;
  finally
    JSON.Free
  end;
end;

initialization
  for var I := 0 to High(SiluTable) do
  begin
    var X: Double := (I - 5120) / 256;
    SiluTable[I] := X / (1 + Exp(-X));
  end;

end.

