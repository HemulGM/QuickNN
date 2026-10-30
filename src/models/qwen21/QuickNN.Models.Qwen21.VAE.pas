unit QuickNN.Models.Qwen21.VAE;

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
  System.SysUtils,
  QuickNN.Common,
  QuickNN.SafeTensors,
  QuickNN.Models.Qwen21.Math,
  QuickNN.Vulkan;

type
  TQwen21Conv = record
    Weight, Bias: TMemoryBlock;
    InputDim, OutputDim, Kernel: Integer;
    procedure Load(const Files: TSafeTensorFiles; const Prefix: string);
    function ForwardGPU(
        const Ctx: TQNNGPUContext;
        const Src: TQNNGPUTensor;
        const Height, Width: Integer;
        const Stride: Integer = 1;
        const Padding: Integer = -1;
        const PadEnd: Boolean = False
    ): TQNNGPUTensor;
    function Forward(
        const Src: TMemoryBlock;
        const Height, Width: Integer;
        const Stride: Integer = 1;
        const Padding: Integer = -1;
        const PadEnd: Boolean = False
    ): TMemoryBlock;
    procedure Free;
  end;

  TQwen21VAERes = record
    Gamma1, Gamma2: TMemoryBlock;
    Conv1, Conv2, Shortcut: TQwen21Conv;
    procedure Load(const Files: TSafeTensorFiles; const Prefix: string);
    function ForwardGPU(
        const Ctx: TQNNGPUContext;
        const Src: TQNNGPUTensor;
        const Height, Width: Integer
    ): TQNNGPUTensor;
    function Forward(const Src: TMemoryBlock; const Height, Width: Integer): TMemoryBlock;
    procedure Free;
  end;

  TQwen21VAEUp = record
    Blocks: array[0..2] of TQwen21VAERes;
    Upsample: TQwen21Conv;
    TemporalFactor: Integer;
    procedure Load(const Files: TSafeTensorFiles; const Index: Integer);
    function ForwardGPU(const Ctx: TQNNGPUContext; const Src: TQNNGPUTensor; var Height, Width: Integer): TQNNGPUTensor;
    function Forward(const Src: TMemoryBlock; var Height, Width: Integer): TMemoryBlock;
    procedure Free;
  end;

  TQwen21VAEDown = record
    Blocks: array[0..1] of TQwen21VAERes;
    Downsample: TQwen21Conv;
    TemporalFactor, SpatialFactor: Integer;
    procedure Load(const Files: TSafeTensorFiles; const Index: Integer);
    function ForwardGPU(const Ctx: TQNNGPUContext; const Src: TQNNGPUTensor; var Height, Width: Integer): TQNNGPUTensor;
    function Forward(const Src: TMemoryBlock; var Height, Width: Integer): TMemoryBlock;
    procedure Free;
  end;

  TQwen21VAE = class
  private
    FFiles: TSafeTensorFiles;
    FPost, FIn, FOut, FQKV, FProj: TQwen21Conv;
    FMid1, FMid2: TQwen21VAERes;
    FUps: array[0..4] of TQwen21VAEUp;
    FAttentionGamma, FOutGamma: TMemoryBlock;
    FMean, FStd: TArray<Single>;
    FStatus: TQwen21Status;
    FEncoderLoaded: Boolean;
    FEncIn, FEncOut, FQuant, FEncQKV, FEncProj: TQwen21Conv;
    FEncNorm, FEncAttentionGamma: TMemoryBlock;
    FEncMid1, FEncMid2: TQwen21VAERes;
    FDowns: array[0..4] of TQwen21VAEDown;
    procedure LoadEncoder;
    function AttentionGPU(
        const Ctx: TQNNGPUContext;
        const Src: TQNNGPUTensor;
        const Height, Width: Integer;
        const Encoder: Boolean = False
    ): TQNNGPUTensor;
    function Attention(
        const Src: TMemoryBlock;
        const Height, Width: Integer;
        const Encoder: Boolean = False
    ): TMemoryBlock;
  public
    constructor Create(const Checkpoint, ConfigPath: string; const Status: TQwen21Status = nil);
    destructor Destroy; override;
    function EncodeGPU(
        const Ctx: TQNNGPUContext;
        const Pixels: TQNNGPUTensor;
        const ImageHeight, ImageWidth: Integer
    ): TQNNGPUTensor;
    function DecodeGPU(
        const Ctx: TQNNGPUContext;
        const Latents: TQNNGPUTensor;
        const Height, Width: Integer
    ): TQNNGPUTensor;
    function Encode(const Image: TQNNImage): TMemoryBlock;
    function Decode(const Latents: TMemoryBlock; const Height, Width: Integer): TQNNImage;
  end;

procedure Q21ChannelRMS(const Dst, Src, Gamma: TMemoryBlock; const Channels, Spatial: Integer);

procedure Q21SavePNG(const Image: TQNNImage; const FileName: string);

implementation

uses
  System.Math,
  System.JSON,
  System.Generics.Collections,
  System.IOUtils,
  System.Classes,
  System.ZLib,
  QuickNN.Quantization,
  QuickNN.Kernels;

procedure Q21ChannelRMS(const Dst, Src, Gamma: TMemoryBlock; const Channels, Spatial: Integer);
var
  Sum, Inv: Double;
begin
  for var P := 0 to Spatial - 1 do
  begin
    Sum := 0;
    for var C := 0 to Channels - 1 do
      Sum := Sum + Sqr(Double(System.PSingle(Src)[NativeInt(C) * Spatial + P]));
    Inv := Sqrt(Channels) / Max(Sqrt(Sum), 1E-12);
    for var C := 0 to Channels - 1 do
      System.PSingle(Dst)[NativeInt(C) * Spatial + P] :=
          System.PSingle(Src)[NativeInt(C) * Spatial + P] * Inv * System.PSingle(Gamma)[C];
  end;
end;

procedure TQwen21Conv.Load(const Files: TSafeTensorFiles; const Prefix: string);
var
  T, B: PSafeTensor;
begin
  T := Files.GetTensor(Prefix + '.weight');
  if (T = nil) or not (T.Ndim in [4, 5]) then
    raise ESafeTensorError.Create('Missing image convolution: ' + Prefix);
  if (T.Ndim = 5) and (T.Shape[2] <> 1) then
    raise ESafeTensorError.Create('Expected single-frame convolution: ' + Prefix);
  InputDim := T.Shape[1];
  OutputDim := T.Shape[0];
  Kernel := T.Shape[T.Ndim - 1];
  if (InputDim <= 0) or (OutputDim <= 0) or not (Kernel in [1, 3]) or (T.Shape[T.Ndim - 2] <> Kernel) then
    raise ESafeTensorError.Create('Invalid convolution shape: ' + Prefix);
  Weight := Files.GetLinearWeight(Prefix + '.weight', True);
  B := Files.GetTensor(Prefix + '.bias');
  if B <> nil then
  begin
    if B.Count <> OutputDim then
      raise ESafeTensorError.Create('Invalid convolution bias: ' + Prefix);
    Bias := B.AsSingle(True);
  end;
end;

function TQwen21Conv.Forward(
    const Src: TMemoryBlock;
    const Height, Width: Integer;
    const Stride, Padding: Integer;
    const PadEnd: Boolean
): TMemoryBlock;
const
  TILE_PIXELS = 1024;
var
  Col, Projected: TMemoryBlock;
  SrcSpatial, Spatial, OutHeight, OutWidth, Pad, Inner, Start, Count, Iy, Ix, Ky, Kx, Ch, Py, Px: Integer;
  Index: NativeInt;
  Value: Single;
begin
  if Stride < 1 then
    raise EArgumentException.Create('Invalid convolution stride');
  if Padding < 0 then
    Pad := Kernel div 2
  else
    Pad := Padding;
  SrcSpatial := Height * Width;
  OutHeight := (Height + 2 * Pad + Ord(PadEnd) - Kernel) div Stride + 1;
  OutWidth := (Width + 2 * Pad + Ord(PadEnd) - Kernel) div Stride + 1;
  Spatial := OutHeight * OutWidth;
  Inner := InputDim * Kernel * Kernel;
  if (Src.Count < Int64(InputDim) * SrcSpatial) or (Spatial <= 0) then
    raise EArgumentException.Create('Invalid convolution input');
  Result := TMemoryBlock.Create([OutputDim, OutHeight, OutWidth], 'qwen21-vae-conv');
  try
    try
      Col := TMemoryBlock.Create([Min(TILE_PIXELS, Spatial), Inner], 'qwen21-vae-im2col');
      Projected := TMemoryBlock.Create([Min(TILE_PIXELS, Spatial), OutputDim], 'qwen21-vae-projected');
      Start := 0;
      while Start < Spatial do
      begin
        Count := Min(TILE_PIXELS, Spatial - Start);
        for var P := 0 to Count - 1 do
        begin
          Py := (Start + P) div OutWidth;
          Px := (Start + P) mod OutWidth;
          Index := NativeInt(P) * Inner;
          for Ch := 0 to InputDim - 1 do
            for Ky := 0 to Kernel - 1 do
              for Kx := 0 to Kernel - 1 do
              begin
                Iy := Py * Stride + Ky - Pad;
                Ix := Px * Stride + Kx - Pad;
                if (Iy >= 0) and (Iy < Height) and (Ix >= 0) and (Ix < Width) then
                  Value := System.PSingle(Src)[NativeInt(Ch) * SrcSpatial + Iy * Width + Ix]
                else
                  Value := 0;
                System.PSingle(Col)[Index] := Value;
                Inc(Index);
              end;
        end;
        QNNLinearNoBias(Projected, Col, Weight, Count, Inner, OutputDim);
        for var P := 0 to Count - 1 do
          for Ch := 0 to OutputDim - 1 do
          begin
            Value := System.PSingle(Projected)[NativeInt(P) * OutputDim + Ch];
            if Bias.IsAssigned then
              Value := Value + System.PSingle(Bias)[Ch];
            System.PSingle(Result)[NativeInt(Ch) * Spatial + Start + P] := Value;
          end;
        Inc(Start, Count);
      end;
    except
      Result.Free;
      raise
    end;
  finally
    Col.Free;
    Projected.Free
  end;
end;

procedure TQwen21Conv.Free;
begin
  Weight.Free;
  Bias.Free;
  InputDim := 0;
  OutputDim := 0;
  Kernel := 0
end;

procedure TQwen21VAERes.Load(const Files: TSafeTensorFiles; const Prefix: string);
begin
  Gamma1 := QNNRequiredTensor(Files, Prefix + '.residual.0.gamma').AsSingle(True);
  Gamma2 := QNNRequiredTensor(Files, Prefix + '.residual.3.gamma').AsSingle(True);
  Conv1.Load(Files, Prefix + '.residual.2');
  Conv2.Load(Files, Prefix + '.residual.6');
  if Conv1.InputDim <> Conv1.OutputDim then
    Shortcut.Load(Files, Prefix + '.shortcut');
  if (Gamma1.Count <> Conv1.InputDim)
      or (Gamma2.Count <> Conv1.OutputDim)
      or (Conv2.InputDim <> Conv1.OutputDim)
      or (Conv2.OutputDim <> Conv1.OutputDim) then
    raise ESafeTensorError.Create('Inconsistent VAE residual shapes: ' + Prefix);
end;

function TQwen21VAERes.Forward(const Src: TMemoryBlock; const Height, Width: Integer): TMemoryBlock;
var
  N, C1, N2, Skip: TMemoryBlock;
  Spatial: Integer;
begin
  Spatial := Height * Width;
  try
    N := TMemoryBlock.Create([Conv1.InputDim, Height, Width], 'qwen21-vae-norm1');
    Q21ChannelRMS(N, Src, Gamma1, Conv1.InputDim, Spatial);
    Q21Silu(N, Conv1.InputDim * Spatial);
    C1 := Conv1.Forward(N, Height, Width);
    N.Free;
    N2 := TMemoryBlock.Create([Conv1.OutputDim, Height, Width], 'qwen21-vae-norm2');
    Q21ChannelRMS(N2, C1, Gamma2, Conv1.OutputDim, Spatial);
    Q21Silu(N2, Conv1.OutputDim * Spatial);
    C1.Free;
    Result := Conv2.Forward(N2, Height, Width);
    if Shortcut.Weight.IsAssigned then
    begin
      Skip := Shortcut.Forward(Src, Height, Width);
      QNNAddInplace(Result, Skip, Conv1.OutputDim * Spatial);
    end
    else
      QNNAddInplace(Result, Src, Conv1.OutputDim * Spatial);
  finally
    N.Free;
    C1.Free;
    N2.Free;
    Skip.Free
  end;
end;

procedure TQwen21VAERes.Free;
begin
  Gamma1.Free;
  Gamma2.Free;
  Conv1.Free;
  Conv2.Free;
  Shortcut.Free
end;

procedure TQwen21VAEUp.Load(const Files: TSafeTensorFiles; const Index: Integer);
var
  P: string;
begin
  P := Format('decoder.upsamples.%d.upsamples.', [Index]);
  for var J := 0 to 2 do
    Blocks[J].Load(Files, P + IntToStr(J));
  if Index < 4 then
    Upsample.Load(Files, P + '3.resample.1');
  if Index < 3 then
    TemporalFactor := 2
  else
    TemporalFactor := 1;
  if Upsample.Weight.IsAssigned
      and ((Upsample.InputDim <> Blocks[2].Conv2.OutputDim)
          or (Upsample.OutputDim <> Blocks[2].Conv2.OutputDim)
          or (Upsample.OutputDim * TemporalFactor * 4 mod Blocks[0].Conv1.InputDim <> 0)) then
    raise ESafeTensorError.Create('Inconsistent VAE upsample shapes: ' + P);
end;

function TQwen21VAEUp.Forward(const Src: TMemoryBlock; var Height, Width: Integer): TMemoryBlock;
var
  X, Next, Resized: TMemoryBlock;
  InChannels, OutChannels, Repeats, InputChannel, Spatial, OutSpatial: Integer;
begin
  Spatial := Height * Width;
  InChannels := Blocks[0].Conv1.InputDim;
  OutChannels := Blocks[2].Conv2.OutputDim;
  try
    X := TMemoryBlock.Create([InChannels, Height, Width], 'qwen21-up-input');
    QNNCopy(X, Src, InChannels * Spatial);
    for var J := 0 to 2 do
    begin
      Next := Blocks[J].Forward(X, Height, Width);
      X.Free;
      X := Next;
      Next := Default(TMemoryBlock)
    end;
    if not Upsample.Weight.IsAssigned then
    begin
      Result := X;
      X := Default(TMemoryBlock);
      Exit
    end;
    OutSpatial := 4 * Spatial;
    Resized := TMemoryBlock.Create([OutChannels, Height * 2, Width * 2], 'qwen21-up-nearest');
    for var C := 0 to OutChannels - 1 do
      for var Y := 0 to Height * 2 - 1 do
        for var Xx := 0 to Width * 2 - 1 do
          System.PSingle(Resized)[NativeInt(C) * OutSpatial + Y * (Width * 2) + Xx] :=
              System.PSingle(X)[NativeInt(C) * Spatial + (Y div 2) * Width + Xx div 2];
    Result := Upsample.Forward(Resized, Height * 2, Width * 2);
    X.Free;
    Resized.Free;
    // First-frame specialization of DupUp3D(repeat_interleave -> reshape ->
    // pixel shuffle -> drop factor_t-1 frames). This includes channel shuffling.
    Repeats := OutChannels * TemporalFactor * 4 div InChannels;
    for var C := 0 to OutChannels - 1 do
      for var Y := 0 to Height * 2 - 1 do
        for var Xx := 0 to Width * 2 - 1 do
        begin
          InputChannel := (C * TemporalFactor * 4 + (TemporalFactor - 1) * 4 + (Y mod 2) * 2 + Xx mod 2) div Repeats;
          System.PSingle(Result)[NativeInt(C) * OutSpatial + Y * (Width * 2) + Xx] :=
              System.PSingle(Result)[NativeInt(C) * OutSpatial + Y * (Width * 2) + Xx]
                  + System.PSingle(Src)[NativeInt(InputChannel) * Spatial + (Y div 2) * Width + Xx div 2];
        end;
    Height := Height * 2;
    Width := Width * 2;
  finally
    X.Free;
    Next.Free;
    Resized.Free
  end;
end;

procedure TQwen21VAEUp.Free;
begin
  for var I := 0 to 2 do
    Blocks[I].Free;
  Upsample.Free
end;

procedure TQwen21VAEDown.Load(const Files: TSafeTensorFiles; const Index: Integer);
var
  Prefix: string;
begin
  Prefix := Format('encoder.downsamples.%d.downsamples.', [Index]);
  for var I := 0 to 1 do
    Blocks[I].Load(Files, Prefix + IntToStr(I));
  SpatialFactor := 1;
  TemporalFactor := 1;
  if Index < 4 then
  begin
    SpatialFactor := 2;
    Downsample.Load(Files, Prefix + '2.resample.1')
  end;
  if (Index > 0) and (Index < 4) then
    TemporalFactor := 2;
end;

function TQwen21VAEDown.Forward(const Src: TMemoryBlock; var Height, Width: Integer): TMemoryBlock;
var
  X, Next: TMemoryBlock;
  InChannels, OutChannels, GroupSize, OldW, SrcSpatial, DstSpatial, Flat, C, Time, Dy, Dx: Integer;
  Sum: Double;
begin
  OldW := Width;
  SrcSpatial := Height * Width;
  InChannels := Blocks[0].Conv1.InputDim;
  OutChannels := Blocks[1].Conv2.OutputDim;
  GroupSize := InChannels * TemporalFactor * SpatialFactor * SpatialFactor div OutChannels;
  if (GroupSize < 1) or (InChannels * TemporalFactor * SpatialFactor * SpatialFactor mod OutChannels <> 0) then
    raise ESafeTensorError.Create('Invalid residual downsample channel grouping');
  try
    X := TMemoryBlock.Create([InChannels, Height, Width], 'vae-down-input');
    QNNCopy(X, Src, X.Count);
    for var I := 0 to 1 do
    begin
      Next := Blocks[I].Forward(X, Height, Width);
      X.Free;
      X := Next;
      Next := Default(TMemoryBlock)
    end;
    if Downsample.Weight.IsAssigned then
    begin
      Next := Downsample.Forward(X, Height, Width, 2, 0, True);
      X.Free;
      X := Next;
      Next := Default(TMemoryBlock);
      Height := Height div 2;
      Width := Width div 2;
    end;
    DstSpatial := Height * Width;
    // First-frame AvgDown3D: temporal positions preceding the single image
    // are zero-padded, then spatial/temporal/channel groups are averaged.
    for var OutChannel := 0 to OutChannels - 1 do
      for var Y := 0 to Height - 1 do
        for var Xx := 0 to Width - 1 do
        begin
          Sum := 0;
          for var J := 0 to GroupSize - 1 do
          begin
            Flat := OutChannel * GroupSize + J;
            C := Flat div (TemporalFactor * SpatialFactor * SpatialFactor);
            Time := Flat div (SpatialFactor * SpatialFactor) mod TemporalFactor;
            Dy := Flat div SpatialFactor mod SpatialFactor;
            Dx := Flat mod SpatialFactor;
            if Time = TemporalFactor - 1 then
              Sum :=
                  Sum
                      + System.PSingle(Src)[
                          NativeInt(C) * SrcSpatial + (Y * SpatialFactor + Dy) * OldW + Xx * SpatialFactor + Dx];
          end;
          System.PSingle(X)[NativeInt(OutChannel) * DstSpatial + Y * Width + Xx] :=
              System.PSingle(X)[NativeInt(OutChannel) * DstSpatial + Y * Width + Xx] + Sum / GroupSize;
        end;
    Result := X;
    X := Default(TMemoryBlock);
  finally
    X.Free;
    Next.Free
  end;
end;

procedure TQwen21VAEDown.Free;
begin
  for var I := 0 to 1 do
    Blocks[I].Free;
  Downsample.Free
end;

procedure TQwen21VAE.LoadEncoder;
begin
  if FEncoderLoaded then
    Exit;
  FEncIn.Load(FFiles, 'encoder.conv1');
  FEncOut.Load(FFiles, 'encoder.head.2');
  FQuant.Load(FFiles, 'conv1');
  FEncNorm := QNNRequiredTensor(FFiles, 'encoder.head.0.gamma').AsSingle(True);
  FEncMid1.Load(FFiles, 'encoder.middle.0');
  FEncMid2.Load(FFiles, 'encoder.middle.2');
  FEncAttentionGamma := QNNRequiredTensor(FFiles, 'encoder.middle.1.norm.gamma').AsSingle(True);
  FEncQKV.Load(FFiles, 'encoder.middle.1.to_qkv');
  FEncProj.Load(FFiles, 'encoder.middle.1.proj');
  for var I := 0 to 4 do
    FDowns[I].Load(FFiles, I);
  if (FEncIn.InputDim <> 4)
      or (FEncIn.OutputDim <> 96)
      or (FEncOut.OutputDim <> 128)
      or (FQuant.InputDim <> 128)
      or (FQuant.OutputDim <> 128) then
    raise ESafeTensorError.Create('Unexpected RGBA VAE encoder');
  FEncoderLoaded := True;
end;

constructor TQwen21VAE.Create(const Checkpoint, ConfigPath: string; const Status: TQwen21Status);
var
  JSON: TJSONValue;
  Means, Stds: TJSONArray;
begin
  inherited Create;
  FStatus := Status;
  JSON := TJSONObject.ParseJSONValue(TFile.ReadAllText(ConfigPath));
  if JSON = nil then
    raise EArgumentException.Create('Invalid Qwen-Image 2.1 VAE config');
  try
    if (JSON.GetValue<Integer>('z_dim') <> 64)
        or (JSON.GetValue<Integer>('decoder_base_dim') <> 144)
        or (JSON.GetValue<Integer>('out_channels') <> 4) then
      raise EArgumentException.Create('Expected RGBA Qwen-Image 2.1 VAE');
    Means := JSON.FindValue('latents_mean') as TJSONArray;
    Stds := JSON.FindValue('latents_std') as TJSONArray;
    if (Means = nil) or (Stds = nil) or (Means.Count <> 64) or (Stds.Count <> 64) then
      raise EArgumentException.Create('Missing VAE latent statistics');
    SetLength(FMean, 64);
    SetLength(FStd, 64);
    for var I := 0 to 63 do
    begin
      FMean[I] := Means.Items[I].AsType<Single>;
      FStd[I] := Stds.Items[I].AsType<Single>
    end;
  finally
    JSON.Free
  end;
  SetLength(FFiles, 1);
  FFiles[0] := TSafeTensorFile.Open(Checkpoint);
  FPost.Load(FFiles, 'conv2');
  FIn.Load(FFiles, 'decoder.conv1');
  FOut.Load(FFiles, 'decoder.head.2');
  FMid1.Load(FFiles, 'decoder.middle.0');
  FMid2.Load(FFiles, 'decoder.middle.2');
  FAttentionGamma := QNNRequiredTensor(FFiles, 'decoder.middle.1.norm.gamma').AsSingle(True);
  FQKV.Load(FFiles, 'decoder.middle.1.to_qkv');
  FProj.Load(FFiles, 'decoder.middle.1.proj');
  FOutGamma := QNNRequiredTensor(FFiles, 'decoder.head.0.gamma').AsSingle(True);
  for var I := 0 to 4 do
    FUps[I].Load(FFiles, I);
end;

destructor TQwen21VAE.Destroy;
begin
  QNNGPUClearWeightCache;
  FPost.Free;
  FIn.Free;
  FOut.Free;
  FQKV.Free;
  FProj.Free;
  FMid1.Free;
  FMid2.Free;
  FAttentionGamma.Free;
  FOutGamma.Free;
  FEncIn.Free;
  FEncOut.Free;
  FQuant.Free;
  FEncQKV.Free;
  FEncProj.Free;
  FEncNorm.Free;
  FEncAttentionGamma.Free;
  FEncMid1.Free;
  FEncMid2.Free;
  for var I := 0 to 4 do
    FDowns[I].Free;
  for var I := 0 to 4 do
    FUps[I].Free;
  for var I := 0 to High(FFiles) do
    FFiles[I].Close;
  inherited;
end;

function TQwen21VAE.Attention(
    const Src: TMemoryBlock;
    const Height, Width: Integer;
    const Encoder: Boolean
): TMemoryBlock;
var
  Norm, Qkv, Q, K, V, A, Chw: TMemoryBlock;
  Spatial, Channels: Integer;
  QkvConv, ProjConv: ^TQwen21Conv;
  Gamma: TMemoryBlock;
begin
  if Encoder then
  begin
    QkvConv := @FEncQKV;
    ProjConv := @FEncProj;
    Gamma := FEncAttentionGamma
  end
  else
  begin
    QkvConv := @FQKV;
    ProjConv := @FProj;
    Gamma := FAttentionGamma
  end;
  Spatial := Height * Width;
  Channels := QkvConv.InputDim;
  try
    Norm := TMemoryBlock.Create([Channels, Height, Width], 'qwen21-vae-attn-norm');
    Q21ChannelRMS(Norm, Src, Gamma, Channels, Spatial);
    Qkv := QkvConv.Forward(Norm, Height, Width);
    Norm.Free;
    Q := TMemoryBlock.Create([Spatial, Channels], 'qwen21-vae-q');
    K := TMemoryBlock.Create([Spatial, Channels], 'qwen21-vae-k');
    V := TMemoryBlock.Create([Spatial, Channels], 'qwen21-vae-v');
    A := TMemoryBlock.Create([Spatial, Channels], 'qwen21-vae-a');
    for var P := 0 to Spatial - 1 do
      for var C := 0 to Channels - 1 do
      begin
        System.PSingle(Q)[NativeInt(P) * Channels + C] := System.PSingle(Qkv)[NativeInt(C) * Spatial + P];
        System.PSingle(K)[NativeInt(P) * Channels + C] := System.PSingle(Qkv)[NativeInt(C + Channels) * Spatial + P];
        System.PSingle(V)[NativeInt(P) * Channels + C] :=
            System.PSingle(Qkv)[NativeInt(C + 2 * Channels) * Spatial + P];
      end;
    Qkv.Free;
    Q21Attention(A, Q, K, V, Spatial, 1, 1, Channels, 0);
    Chw := TMemoryBlock.Create([Channels, Height, Width], 'qwen21-vae-attn-chw');
    for var P := 0 to Spatial - 1 do
      for var C := 0 to Channels - 1 do
        System.PSingle(Chw)[NativeInt(C) * Spatial + P] := System.PSingle(A)[NativeInt(P) * Channels + C];
    Result := ProjConv.Forward(Chw, Height, Width);
    QNNAddInplace(Result, Src, Channels * Spatial);
  finally
    Norm.Free;
    Qkv.Free;
    Q.Free;
    K.Free;
    V.Free;
    A.Free;
    Chw.Free
  end;
end;

function TQwen21VAE.Encode(const Image: TQNNImage): TMemoryBlock;
var
  X, Next, N: TMemoryBlock;
  Height, Width: Integer;
  Value: Single;
begin
  if not (Image.Channels in [3, 4])
      or (Image.Width < 16)
      or (Image.Height < 16)
      or (Image.Width mod 16 <> 0)
      or (Image.Height mod 16 <> 0) then
    raise EArgumentException.Create('Invalid VAE reference image');
  LoadEncoder;
  Height := Image.Height;
  Width := Image.Width;
  try
    X := TMemoryBlock.Create([4, Height, Width], 'vae-encoder-input');
    for var P := 0 to Height * Width - 1 do
      for var C := 0 to 3 do
      begin
        if C < Image.Channels then
          Value := Image.Data[NativeInt(P) * Image.Channels + C] / 127.5 - 1
        else
          Value := 1;
        System.PSingle(X)[NativeInt(C) * Height * Width + P] := Value;
      end;
    Next := FEncIn.Forward(X, Height, Width);
    X.Free;
    X := Next;
    Next := Default(TMemoryBlock);
    for var I := 0 to 4 do
    begin
      Next := FDowns[I].Forward(X, Height, Width);
      X.Free;
      X := Next;
      Next := Default(TMemoryBlock);
      QNNCheckFinite(X, X.Count, 'VAE encoder down block ' + IntToStr(I));
      if Assigned(FStatus) then
        FStatus(Format('VAE encoder %d/5: %dx%d', [I + 1, Width, Height]));
    end;
    Next := FEncMid1.Forward(X, Height, Width);
    X.Free;
    X := Next;
    Next := Default(TMemoryBlock);
    Next := Attention(X, Height, Width, True);
    X.Free;
    X := Next;
    Next := Default(TMemoryBlock);
    Next := FEncMid2.Forward(X, Height, Width);
    X.Free;
    X := Next;
    Next := Default(TMemoryBlock);
    N := TMemoryBlock.Create([FEncOut.InputDim, Height, Width], 'vae-encoder-final-norm');
    Q21ChannelRMS(N, X, FEncNorm, FEncOut.InputDim, Height * Width);
    Q21Silu(N, N.Count);
    X.Free;
    X := FEncOut.Forward(N, Height, Width);
    Next := FQuant.Forward(X, Height, Width);
    QNNCheckFinite(Next, Next.Count, 'VAE encoded distribution');
    Result := TMemoryBlock.Create([Height * Width, 64], 'vae-normalized-reference');
    for var P := 0 to Height * Width - 1 do
      for var C := 0 to 63 do
        System.PSingle(Result)[NativeInt(P) * 64 + C] :=
            (System.PSingle(Next)[NativeInt(C) * Height * Width + P] - FMean[C]) / FStd[C];
  finally
    X.Free;
    Next.Free;
    N.Free
  end;
end;

function TQwen21VAE.Decode(const Latents: TMemoryBlock; const Height, Width: Integer): TQNNImage;
var
  X, Next, N: TMemoryBlock;
  H, W, Spatial: Integer;
  Pixel: Single;
begin
  H := Height;
  W := Width;
  Spatial := H * W;
  if (Latents.DataType <> dtF32) or (Latents.Count <> Int64(Spatial) * 64) then
    raise EArgumentException.Create('Invalid VAE latents');
  try
    X := TMemoryBlock.Create([64, H, W], 'qwen21-vae-latents');
    for var P := 0 to Spatial - 1 do
      for var C := 0 to 63 do
        System.PSingle(X)[NativeInt(C) * Spatial + P] :=
            System.PSingle(Latents)[NativeInt(P) * 64 + C] * FStd[C] + FMean[C];
    Next := FPost.Forward(X, H, W);
    X.Free;
    X := Next;
    Next := Default(TMemoryBlock);
    Next := FIn.Forward(X, H, W);
    X.Free;
    X := Next;
    Next := Default(TMemoryBlock);
    Next := FMid1.Forward(X, H, W);
    X.Free;
    X := Next;
    Next := Default(TMemoryBlock);
    Next := Attention(X, H, W);
    X.Free;
    X := Next;
    Next := Default(TMemoryBlock);
    Next := FMid2.Forward(X, H, W);
    X.Free;
    X := Next;
    Next := Default(TMemoryBlock);
    for var I := 0 to 4 do
    begin
      Next := FUps[I].Forward(X, H, W);
      X.Free;
      X := Next;
      Next := Default(TMemoryBlock);
      QNNCheckFinite(X, X.Count, 'VAE up block ' + IntToStr(I));
      if Assigned(FStatus) then
        FStatus(Format('VAE block %d/5: %dx%d', [I + 1, W, H]));
    end;
    N := TMemoryBlock.Create([FOut.InputDim, H, W], 'qwen21-vae-final-norm');
    Q21ChannelRMS(N, X, FOutGamma, FOut.InputDim, H * W);
    Q21Silu(N, N.Count);
    X.Free;
    X := FOut.Forward(N, H, W);
    QNNCheckFinite(X, X.Count, 'Decoded RGBA image');
    Result := TQNNImage.Create(W, H, 4);
    for var P := 0 to H * W - 1 do
      for var C := 0 to 3 do
      begin
        Pixel := EnsureRange(System.PSingle(X)[NativeInt(C) * H * W + P], Single(-1), Single(1));
        Result.Data[NativeInt(P) * 4 + C] := Round((Pixel + 1) * 127.5);
      end;
  finally
    X.Free;
    Next.Free;
    N.Free
  end;
end;

procedure Q21SavePNG(const Image: TQNNImage; const FileName: string);
const
  SIGNATURE: array[0..7] of Byte = ($89, $50, $4E, $47, $0D, $0A, $1A, $0A);
var
  Stream: TFileStream;
  Raw, Compressed: TMemoryStream;
  Zip: TZCompressionStream;
  Row: TBytes;
  Data, Header: TBytes;
  Crc: UInt32;

  procedure WriteBE(const Value: UInt32);
  var
    Bytes: array[0..3] of Byte;
  begin
    Bytes[0] := (Value shr 24) and 255;
    Bytes[1] := (Value shr 16) and 255;
    Bytes[2] := (Value shr 8) and 255;
    Bytes[3] := Value and 255;
    Stream.WriteBuffer(Bytes, 4)
  end;

  procedure UpdateCRC(const Ptr: PByte; const Count: NativeInt);
  begin
    for var I: NativeInt := 0 to Count - 1 do
    begin
      Crc := Crc xor Ptr[I];
      for var B := 0 to 7 do
        if Crc and 1 <> 0 then
          Crc := (Crc shr 1) xor $EDB88320
        else
          Crc := Crc shr 1;
    end;
  end;

  procedure Chunk(const Name: AnsiString; const Bytes: TBytes);
  begin
    WriteBE(Length(Bytes));
    Stream.WriteBuffer(PAnsiChar(Name)^, 4);
    if Length(Bytes) > 0 then
      Stream.WriteBuffer(Bytes[0], Length(Bytes));
    Crc := $FFFFFFFF;
    UpdateCRC(PByte(PAnsiChar(Name)), 4);
    if Length(Bytes) > 0 then
      UpdateCRC(@Bytes[0], Length(Bytes));
    WriteBE(not Crc);
  end;

begin
  if not (Image.Channels in [3, 4])
      or (Image.Width < 1)
      or (Image.Height < 1)
      or (Length(Image.Data) <> Int64(Image.Width) * Image.Height * Image.Channels) then
    raise EArgumentException.Create('Expected RGB or RGBA image');
  Raw := TMemoryStream.Create;
  Compressed := TMemoryStream.Create;
  try
    SetLength(Row, NativeInt(Image.Width) * Image.Channels + 1);
    Row[0] := 0;
    for var Y := 0 to Image.Height - 1 do
    begin
      Move(Image.Data[NativeInt(Y) * Image.Width * Image.Channels], Row[1], Image.Width * Image.Channels);
      Raw.WriteBuffer(Row[0], Length(Row));
    end;
    Zip := TZCompressionStream.Create(Compressed);
    try
      Raw.Position := 0;
      Zip.CopyFrom(Raw, Raw.Size)
    finally
      Zip.Free
    end;
    SetLength(Data, Compressed.Size);
    Move(Compressed.Memory^, Data[0], Length(Data));
    SetLength(Header, 13);
    FillChar(Header[0], 13, 0);
    Header[0] := (UInt32(Image.Width) shr 24) and 255;
    Header[1] := (UInt32(Image.Width) shr 16) and 255;
    Header[2] := (UInt32(Image.Width) shr 8) and 255;
    Header[3] := Image.Width and 255;
    Header[4] := (UInt32(Image.Height) shr 24) and 255;
    Header[5] := (UInt32(Image.Height) shr 16) and 255;
    Header[6] := (UInt32(Image.Height) shr 8) and 255;
    Header[7] := Image.Height and 255;
    Header[8] := 8;
    if Image.Channels = 4 then
      Header[9] := 6
    else
      Header[9] := 2;
    Stream := TFileStream.Create(FileName, FmCreate);
    try
      Stream.WriteBuffer(SIGNATURE, 8);
      Chunk('IHDR', Header);
      Chunk('IDAT', Data);
      Chunk('IEND', nil)
    finally
      Stream.Free
    end;
  finally
    Raw.Free;
    Compressed.Free
  end;
end;
{$I QuickNN.Models.Qwen21.VAE.GPU.inc}

end.
