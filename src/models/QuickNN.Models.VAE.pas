unit QuickNN.Models.VAE;

{$C+} // enable assertions
{$H+} // longstrings
{$POINTERMATH on} // manipulate, inc, dec, cast pointers
{$T+} // typed pointer when @ is used
{$R+} // raise an error when trying to access arrays out of their bounds

interface

uses
  System.SysUtils,
  QuickNN.SafeTensors,
  QuickNN.Common;

type

  { TVAEResBlock }

  TVAEResBlock = record
    Norm1Weight,
    Norm1Bias, (* [channels] *)
    Conv1Weight,
    Conv1Bias, (* [out_ch, in_ch, 3, 3] *)
    Norm2Weight,
    Norm2Bias, (* [channels] *)
    Conv2Weight,
    Conv2Bias, (* [out_ch, out_ch, 3, 3] *)
    SkipWeight,
    SkipBias: TMemoryBlock; (* [out_ch, in_ch, 1, 1] if in_ch != out_ch *)
    InChannels, OutChannels: LongInt;
    procedure Forward(var Dst: TMemoryBlock; const Src, Work: TMemoryBlock; const Batch, H, W, NumGroups: LongInt);
    // procedure load(var f: file);                                     overload;
    procedure Load(
        const Sf: TSafeTensorFiles;
        const APrefix: string;
        const InCh, OutCh: LongInt;
        const UseMMAP: Boolean = False
    ); overload;
    procedure Free();
  end;

  { TVAEAttnBlock }

  TVAEAttnBlock = record
    NormWeight,
    NormBias, (* [channels] *)
    QWeight,
    QBias, (* [channels, channels, 1, 1] *)
    KWeight,
    KBias, (* [channels, channels, 1, 1] *)
    VWeight,
    VBias, (* [channels, channels, 1, 1] *)
    OutWeight,
    OutBias: TMemoryBlock; (* [channels, channels, 1, 1] *)
    Channels: LongInt;
    procedure Forward(
        var Dst: TMemoryBlock;
        const Src: TMemoryBlock;
        var Work: TMemoryBlock;
        const Batch, H, W, NumGroups: LongInt
    );
    // procedure load(var f:file);                                    overload;
    procedure Load(
        const Sf: TSafeTensorFiles;
        const APrefix: string;
        const AChannels: LongInt;
        const UseMMAP: Boolean = False
    ); overload;
    procedure Free();
  end;

  { TVAEDownSample }

  TVAEDownSample = record
    ConvWeight, ConvBias: TMemoryBlock; (* [channels, channels, 3, 3] *)
    Channels: LongInt;
    procedure Free();
  end;

  TVAEUpSample = type TVAEDownSample;

  (* VAE context *)

  { TVAE }
  PVAE = ^TVAE;
  TVAE = record
  private
    MappedFiles: TSafeTensorFiles;
  public
    (* Configuration *)
    ZChannels, (* 32 (Flux) or 16 (Z-Image) *)
    LatentChannels, (* z_channels * 4: 128 (Flux) or 64 (Z-Image) *)
    BaseChannels, (* 128 *)
    NumResBlocks, (* 2 *)
    NumGroups: LongInt; (* 32 *)
    ChMult: TCHMult; (* {1, 2, 4, 4} *)
    Eps, ScalingFactor, ShiftFactor: Single; (* 0 = use batch norm (Flux), else Z-Image shift *)

    (* Encoder weights *)
    EncConvInWeight, EncConvInBias: TMemoryBlock; (* [128, 3, 3, 3] *)

    (* Encoder down blocks: 4 levels, each with num_res_blocks + optional downsample *)
    EncDownBlocks: TArray<TVAEResBlock>; (* 4 * 2 = 8 resblocks *)
    EncDownsample: TArray<TVAEDownSample>; (* 3 downsamples (not at last level) *)

    (* Encoder mid block *)
    EncMidBlock1: TVAEResBlock;
    EncMidAttn: TVAEAttnBlock;
    EncMidBlock2: TVAEResBlock;

    (* Encoder output *)
    EncNormOutWeight, EncNormOutBias, (*[512] *) EncConvOutWeight, EncConvOutBias: TMemoryBlock; (* [64, 512, 3, 3] *)

    (* Decoder weights *)
    DecConvInWeight, DecConvInBias: TMemoryBlock; (* [512, z_channels, 3, 3] *)

    (* Decoder mid block *)
    DecMidBlock1: TVAEResBlock;
    DecMidAttn: TVAEAttnBlock;
    DecMidBlock2: TVAEResBlock;

    (* Decoder up blocks: 4 levels, each with num_res_blocks+1 + optional upsample *)
    DecUpBlocks: TArray<TVAEResBlock>; (* 4 * 3 = 12 resblocks *)
    DecUpsample: TArray<TVAEUpSample>; (* 3 upsamples *)

    (* Decoder output *)
    DecNormOutWeight, DecNormOutBias, (* [128] *) DecConvOutWeight, DecConvOutBias: TMemoryBlock; (* [3, 128, 3, 3] *)

    (* Normalization stats for latent space (Flux only, NULL for Z-Image) *)
    BnMean: TMemoryBlock; (* [latent_channels] *)
    BnVar: TMemoryBlock; (* [latent_channels] *)

    (* Post-quantization conv (1x1) - Flux only, NULL for Z-Image *)
    QuantConvWeight, (* [z_ch*2, z_ch*2, 1, 1] - encoder *)
    QuantConvBias, (* [z_ch*2] *)
    PostQuantConvWeight, (* [z_ch, z_ch, 1, 1] - decoder *)
    PostQuantConvBias: TMemoryBlock; (* [z_ch] *)

    (* Working memory (allocated for max image size) *)
    MaxH, MaxW: LongInt;
    Work1, Work2, Work3: TMemoryBlock; // PSingle;
    WorkSize: IntPtr;
    UseMMAP: Boolean;
    function Encode(const Img: TMemoryBlock; const Batch, H, W: LongInt; var OutH, OutW: LongInt): TMemoryBlock;
    function Decode(const Latent: TMemoryBlock; const Batch, LatentH, LatentW: LongInt): TQNNImage; overload;
    procedure Decode(const Dst, Latent: TMemoryBlock; const Batch, LatentH, LatentW: LongInt); overload;
    class function Preview(
        const Latents: TMemoryBlock;
        const LatentRGBProj: TArray<TArray<Single>>;
        const LatentHeight, LatentWidth, Dims, Frames: LongInt;
        const LatentRGBBias: TArray<Single>;
        const PatchSize: LongInt
    ): TQNNImage; static;
    // procedure load(var f: file);                                    overload;
    procedure Load(
        const Sf: TSafeTensorFiles;
        const AZChannels: LongInt;
        const AScalingFactor, AShiftFactor: QNNFloat
    ); overload;
    function IsLoaded(): Boolean;
    class procedure PadRightBottom(const Dst, Src: TMemoryBlock; Batch, Channels, H, W: LongInt); static;
    procedure Free;
  end;

implementation
uses
  QuickNN.Kernels,
  QuickNN.Vulkan;

function ReadInt(var F: file): LongInt;
begin
  BlockRead(F, Result, SizeOf(LongInt))
end;

function ReadSingles(var F: file; const Count: LongInt): TMemoryBlock;
var
  S: string;
begin
  S := TFileRec(F).Name;
  Result := TMemoryBlock.Create([Count], S + '[' + IntToStr(FilePos(F) div 128) + ']');
  BlockRead(F, System.PSingle(Result)^, Count * SizeOf(Single))
end;

{ TVAEResBlock }

procedure TVAEResBlock.Forward(
    var Dst: TMemoryBlock;
    const Src, Work: TMemoryBlock;
    const Batch, H, W, NumGroups: LongInt
);
var
  Spatial: LongInt;
  Conv1Out: TMemoryBlock;
begin
  Spatial := H * W;
  if InChannels = OutChannels then
    QNNCopy(Dst, Src, Batch * InChannels * Spatial)
  else
    QNNConv2d(Dst, Src, SkipWeight, SkipBias, InChannels, OutChannels, H, W, 1, 1, 1, 0, Batch);
  // printCompare(batch*in_channels*spatial, dst, readTensor());
  QNNGroupNorm(Work, Src, Norm1Weight, Norm1Bias, Batch, InChannels, H, W, NumGroups);
  // printCompare(batch*in_channels*spatial, work, readTensor());
  QNNSiluInplace(Work, Batch * InChannels * Spatial);
  // printCompare(batch*in_channels*spatial, work, readTensor());
  Conv1Out := Work + Batch * InChannels * Spatial;
  QNNConv2d(Conv1Out, Work, Conv1Weight, Conv1Bias, InChannels, OutChannels, H, W, 3, 3, 1, 1, Batch);
  QNNGroupNorm(Work, Conv1Out, Norm2Weight, Norm2Bias, Batch, OutChannels, H, W, NumGroups);
  QNNSiluInplace(Work, Batch * OutChannels * Spatial);
  QNNConv2d(Conv1Out, Work, Conv2Weight, Conv2Bias, OutChannels, OutChannels, H, W, 3, 3, 1, 1, Batch);
  QNNAddInplace(Dst, Conv1Out, Batch * OutChannels * Spatial);
  // printCompare(batch*out_channels*spatial, dst, readTensor());
end;

// procedure TVAEResBlock.load(var f: file);
// begin
//  in_channels  := readInt(f);
//  out_channels := readInt(f);
//
//  norm1_weight := readSingles(f, in_channels);
//  norm1_bias   := readSingles(f, in_channels);
//  conv1_weight := readSingles(f, out_channels * in_channels * 3 * 3);
//  conv1_bias   := readSingles(f, out_channels);
//  norm2_weight := readSingles(f, out_channels);
//  norm2_bias   := readSingles(f, out_channels);
//  conv2_weight := readSingles(f, out_channels * out_channels * 3 * 3);
//  conv2_bias   := readSingles(f, out_channels);
//
//  if (in_channels <> out_channels) then begin
//      skip_weight := readSingles(f, out_channels * in_channels);
//      skip_bias   := readSingles(f, out_channels);
//  end else begin
//      //skip_weight := nil;
//      //skip_bias := nil;
//  end;
//
// end;

procedure TVAEResBlock.Load(
    const Sf: TSafeTensorFiles;
    const APrefix: string;
    const InCh, OutCh: LongInt;
    const UseMMAP: Boolean
);
begin
  InChannels := InCh;
  OutChannels := OutCh;
  Norm1Weight := Sf.GetTensorDataMemBlock(Format('%s.norm1.weight', [APrefix]), UseMMAP);
  Norm1Bias := Sf.GetTensorDataMemBlock(Format('%s.norm1.bias', [APrefix]), UseMMAP);
  Conv1Weight := Sf.GetTensorDataMemBlock(Format('%s.conv1.weight', [APrefix]), UseMMAP);
  Conv1Bias := Sf.GetTensorDataMemBlock(Format('%s.conv1.bias', [APrefix]), UseMMAP);
  Norm2Weight := Sf.GetTensorDataMemBlock(Format('%s.norm2.weight', [APrefix]), UseMMAP);
  Norm2Bias := Sf.GetTensorDataMemBlock(Format('%s.norm2.bias', [APrefix]), UseMMAP);
  Conv2Weight := Sf.GetTensorDataMemBlock(Format('%s.conv2.weight', [APrefix]), UseMMAP);
  Conv2Bias := Sf.GetTensorDataMemBlock(Format('%s.conv2.bias', [APrefix]), UseMMAP);
  if InCh <> OutCh then
  begin
    SkipWeight := Sf.GetTensorDataMemBlock(Format('%s.conv_shortcut.weight', [APrefix]), UseMMAP);
    SkipBias := Sf.GetTensorDataMemBlock(Format('%s.conv_shortcut.bias', [APrefix]), UseMMAP)
  end
  else
  begin
    // skip_weight := nil;
    // skip_bias   := nil
  end;

end;

procedure TVAEResBlock.Free();
begin
  Norm1Weight.Free;
  Norm1Bias.Free; (* [channels] *)
  Conv1Weight.Free;
  Conv1Bias.Free; (* [out_ch, in_ch, 3, 3] *)
  Norm2Weight.Free;
  Norm2Bias.Free; (* [channels] *)
  Conv2Weight.Free;
  Conv2Bias.Free; (* [out_ch, out_ch, 3, 3] *)
  SkipWeight.Free;
  SkipBias.Free;
end;

{ TVAEAttnBlock }

procedure TVAEAttnBlock.Forward(
    var Dst: TMemoryBlock;
    const Src: TMemoryBlock;
    var Work: TMemoryBlock;
    const Batch, H, W, NumGroups: LongInt
);
var
  Ch, Spatial, B: LongInt;
  Q, K, V, Qb, Kb, Vb, Ob, AttnOut {, q_ptr, k_ptr, v_ptr, o_ptr}: TMemoryBlock;
  Scale: QNNFloat;
  QT, KT, VT, OT, Scores: TMemoryBlock;
begin
  Ch := Channels;
  Spatial := H * W;
  QNNGroupNorm(Work, Src, NormWeight, NormBias, Batch, Ch, H, W, NumGroups);
  Q := Work + Batch * Ch * Spatial;
  K := Q + Batch * Ch * Spatial;
  V := K + Batch * Ch * Spatial;
  QNNConv2d(Q, Work, QWeight, QBias, Ch, Ch, H, W, 1, 1, 1, 0, Batch);
  QNNConv2d(K, Work, KWeight, KBias, Ch, Ch, H, W, 1, 1, 1, 0, Batch);
  QNNConv2d(V, Work, VWeight, VBias, Ch, Ch, H, W, 1, 1, 1, 0, Batch);
  Scale := 1.0 / Sqrt(Ch);
  AttnOut := V + Batch * Ch * Spatial;
  QT := TMemoryBlock.Create([Spatial, Ch], 'VAE_FW_Q ' + TGUID.NewGuid.ToString()); // q_ptr := q_t;
  KT := TMemoryBlock.Create([Spatial, Ch], 'VAE_FW_K ' + TGUID.NewGuid.ToString()); // k_ptr := k_t;
  VT := TMemoryBlock.Create([Spatial, Ch], 'VAE_FW_V ' + TGUID.NewGuid.ToString()); // v_ptr := v_t;
  OT := TMemoryBlock.Create([Spatial, Ch], 'VAE_FW_O ' + TGUID.NewGuid.ToString()); // o_ptr := o_t;
  Scores := TMemoryBlock.Create([Spatial, Spatial], 'VAE_FQ_SCORES ' + TGUID.NewGuid.ToString());
  for B := 0 to Batch - 1 do
  begin
    Qb := Q + B * Ch * Spatial;
    Kb := K + B * Ch * Spatial;
    Vb := V + B * Ch * Spatial;
    Ob := AttnOut + B * Ch * Spatial;
    QNNMatTranspose(QT, Qb, Ch, Spatial);
    QNNScaleInplace(QT, Scale, Ch * Spatial);
    QNNMatTranspose(KT, Kb, Ch, Spatial);
    QNNMatTranspose(VT, Vb, Ch, Spatial);

    // for c := 0 to ch -1 do
    //    for i := 0 to spatial -1 do
    //        begin
    //            q_ptr[i * ch+c] := qb[c * spatial+i] * scale;
    //            k_ptr[i * ch+c] := kb[c * spatial+i];
    //            v_ptr[i * ch+c] := vb[c * spatial+i]
    //        end;
    QNNMatMulNT(Scores, QT, KT, Spatial, Ch, Spatial);
    QNNSoftmaxRows(Scores, Spatial, Spatial);
    QNNMatMulNN(OT, Scores, VT, Spatial, Spatial, Ch);
    QNNMatTranspose(Ob, OT, Spatial, Ch);
    // for c := 0 to ch -1 do
    //    for i := 0 to spatial -1 do
    //        ob[c * spatial+i] := o_ptr[i * ch+c]
  end;
  QT.Free;
  KT.Free;
  VT.Free;
  OT.Free;
  Scores.Free;
  QNNConv2d(Work, AttnOut, OutWeight, OutBias, Ch, Ch, H, W, 1, 1, 1, 0, Batch);
  QNNAdd(Dst, Src, Work, Batch * Ch * Spatial);
end;

// procedure TVAEAttnBlock.load(var f: file);
// begin
//  channels    := readInt(f);
//  norm_weight := readSingles(f, channels);
//  norm_bias   := readSingles(f, channels);
//  q_weight    := readSingles(f, channels * channels);
//  q_bias      := readSingles(f, channels);
//  k_weight    := readSingles(f, channels * channels);
//  k_bias      := readSingles(f, channels);
//  v_weight    := readSingles(f, channels * channels);
//  v_bias      := readSingles(f, channels);
//  out_weight  := readSingles(f, channels * channels);
//  out_bias    := readSingles(f, channels);
// end;

procedure TVAEAttnBlock.Load(
    const Sf: TSafeTensorFiles;
    const APrefix: string;
    const AChannels: LongInt;
    const UseMMAP: Boolean
);
begin
  Channels := AChannels;

  NormWeight := Sf.GetTensorDataMemBlock(Format('%s.group_norm.weight', [APrefix]), UseMMAP);
  NormBias := Sf.GetTensorDataMemBlock(Format('%s.group_norm.bias', [APrefix]), UseMMAP);

  QWeight := Sf.GetTensorDataMemBlock(Format('%s.to_q.weight', [APrefix]), UseMMAP);
  QBias := Sf.GetTensorDataMemBlock(Format('%s.to_q.bias', [APrefix]), UseMMAP);

  KWeight := Sf.GetTensorDataMemBlock(Format('%s.to_k.weight', [APrefix]), UseMMAP);
  KBias := Sf.GetTensorDataMemBlock(Format('%s.to_k.bias', [APrefix]), UseMMAP);

  VWeight := Sf.GetTensorDataMemBlock(Format('%s.to_v.weight', [APrefix]), UseMMAP);
  VBias := Sf.GetTensorDataMemBlock(Format('%s.to_v.bias', [APrefix]), UseMMAP);

  OutWeight := Sf.GetTensorDataMemBlock(Format('%s.to_out.0.weight', [APrefix]), UseMMAP);
  OutBias := Sf.GetTensorDataMemBlock(Format('%s.to_out.0.bias', [APrefix]), UseMMAP);
end;

procedure TVAEAttnBlock.Free();
begin
  NormWeight.Free;
  NormBias.Free; (* [channels] *)
  QWeight.Free;
  QBias.Free; (* [channels, channels, 1, 1] *)
  KWeight.Free;
  KBias.Free; (* [channels, channels, 1, 1] *)
  VWeight.Free;
  VBias.Free; (* [channels, channels, 1, 1] *)
  OutWeight.Free;
  OutBias.Free;
end;

{ TVAEDownSample }

procedure TVAEDownSample.Free();
begin
  ConvWeight.Free();
  ConvBias.Free;
end;

{ TVAE }

function TVAE.IsLoaded(): Boolean;
begin
  Result := EncConvInWeight.IsAssigned() and DecConvInWeight.IsAssigned();
end;

function TVAE.Encode(const Img: TMemoryBlock; const Batch, H, W: LongInt; var OutH, OutW: LongInt): TMemoryBlock;
// const ch_mult : array[0..3] of longint = (1, 2, 4, 4);
var
  CurH,
  CurW,
  BlockIdx,
  DownIdx,
  Progress,
  TotalBlocks,
  Level,
  ChOut,
  R,
  PaddedH,
  PaddedW,
  NewH,
  NewW,
  MidCh,
  ZCh,
  LatentH,
  LatentW,
  ZSpatial,
  B,
  PatchH,
  PatchW,
  LatCh,
  N: LongInt;
  X, Work, Padded: TMemoryBlock;
  Mean: TMemoryBlock;
begin
  if Assigned(PhaseCallback) then
    PhaseCallback('VAE Encode', False);

  if not Work1.IsAllocated() then
    Work1 := TMemoryBlock.Create([Batch, 4, BaseChannels, H, W], 'VAE_WORK1');
  if not Work2.IsAllocated() then
    Work2 := TMemoryBlock.Create([Batch, 4, BaseChannels, H, W], 'VAE_WORK2');
  if not Work3.IsAllocated() then
    Work3 := TMemoryBlock.Create([Batch, 4, BaseChannels, H, W], 'VAE_WORK3');

  X := Work1;
  Work := Work2;
  CurH := H;
  CurW := W;
  QNNConv2d(X, Img, EncConvInWeight, EncConvInBias, 3, BaseChannels, H, W, 3, 3, 1, 1, Batch);
  BlockIdx := 0;
  DownIdx := 0;
  Progress := 0;
  TotalBlocks := 4 * NumResBlocks + 3;
  for Level := 0 to 4 - 1 do
  begin
    ChOut := BaseChannels * ChMult[Level];
    for R := 0 to NumResBlocks - 1 do
    begin
      EncDownBlocks[BlockIdx].Forward(Work, X, Work3, Batch, CurH, CurW, NumGroups);
      Inc(BlockIdx);
      QNNCopy(X, Work, Batch * ChOut * CurH * CurW);
      if Assigned(VAEProgressCallback) then
      begin
        VAEProgressCallback(Progress, TotalBlocks);
        Inc(Progress);
      end
    end;
    if Level < 3 then
    begin
      Padded := Work3;
      PaddedH := CurH + 1;
      PaddedW := CurW + 1;
      NewH := (PaddedH - 3) div 2 + 1;
      NewW := (PaddedW - 3) div 2 + 1;
      PadRightBottom(Padded, X, Batch, ChOut, CurH, CurW);
      QNNConv2d(
          Work,
          Padded,
          EncDownsample[DownIdx].ConvWeight,
          EncDownsample[DownIdx].ConvBias,
          ChOut,
          ChOut,
          PaddedH,
          PaddedW,
          3,
          3,
          2,
          0,
          Batch
      );
      Inc(DownIdx);
      CurH := NewH;
      CurW := NewW;
      QNNCopy(X, Work, Batch * ChOut * CurH * CurW);
    end
  end;
  MidCh := BaseChannels * ChMult[3];
  EncMidBlock1.Forward(Work, X, Work3, Batch, CurH, CurW, NumGroups);
  if Assigned(VAEProgressCallback) then
  begin
    VAEProgressCallback(Progress, TotalBlocks);
    Inc(Progress)
  end;
  EncMidAttn.Forward(X, Work, Work3, Batch, CurH, CurW, NumGroups); // < 0 then
  // exit(nil);
  if Assigned(VAEProgressCallback) then
  begin
    VAEProgressCallback(Progress, TotalBlocks);
    Inc(Progress)
  end;
  EncMidBlock2.Forward(Work, X, Work3, Batch, CurH, CurW, NumGroups);
  QNNCopy(X, Work, Batch * MidCh * CurH * CurW);
  if Assigned(VAEProgressCallback) then
  begin
    VAEProgressCallback(Progress, TotalBlocks);
  end;
  QNNGroupNorm(Work, X, EncNormOutWeight, EncNormOutBias, Batch, MidCh, CurH, CurW, NumGroups);
  QNNSiluInplace(Work, Batch * MidCh * CurH * CurW);
  ZCh := ZChannels * 2;
  QNNConv2d(X, Work, EncConvOutWeight, EncConvOutBias, MidCh, ZCh, CurH, CurW, 3, 3, 1, 1, Batch);
  if Boolean(QuantConvWeight) then
  begin
    QNNConv2d(Work, X, QuantConvWeight, QuantConvBias, ZCh, ZCh, CurH, CurW, 1, 1, 1, 0, Batch);
    QNNCopy(X, Work, Batch * ZCh * CurH * CurW)
  end;
  LatentH := CurH;
  LatentW := CurW;
  ZSpatial := LatentH * LatentW;
  Mean :=
      TMemoryBlock.Create(
          [Batch, ZChannels, LatentH div 2, 2, LatentW div 2, 2],
          'VAE_ENCODE_MEAN ' + TGUID.NewGuid.ToString()
      );
  for B := 0 to Batch - 1 do
    QNNCopy(Mean + B * ZChannels * ZSpatial, X + B * ZCh * ZSpatial, ZChannels * ZSpatial);
  PatchH := LatentH div 2;
  PatchW := LatentW div 2;
  LatCh := LatentChannels;
  // result := TMemoryBlock.Create([batch, lat_ch, patch_h, patch_w], 'VAE_ENCODE_RESULT '+ TGUID.NewGuid.ToString());
  // QNNPatchify(result, mean, batch, z_channels, latent_h, latent_w, 2);
  Mean.Reshape([Batch, ZChannels, LatentH div 2, 2, LatentW div 2, 2]);
  Result := QNNPermut(Mean, [0, 1, 3, 5, 2, 4]);
  Mean.Free;
  if ScalingFactor <> 0.0 then
  begin
    N := Batch * LatCh * PatchH * PatchW;
    QNNFusedBiasScale(Result, Result, -ShiftFactor, ScalingFactor, N);
    // for i := 0 to n -1 do
    //    latent[i] := (latent[i]-shift_factor) * scaling_factor
  end
  else
  begin
    QNNBatchNorm(Work, Result, BnMean, BnVar, Batch, LatCh, PatchH, PatchW);
    QNNCopy(Result, Work, Batch * LatCh * PatchH * PatchW)
  end;
  OutH := PatchH;
  OutW := PatchW;
  if Assigned(PhaseCallback) then
    PhaseCallback('VAE Encode', True);
end;

function TVAE.Decode(const Latent: TMemoryBlock; const Batch, LatentH, LatentW: LongInt): TQNNImage;
// const ch_mult : array[0..3] of integer = (1, 2, 4, 4);
var
  X, Work: TMemoryBlock;
  MeanPtr, VarPtr: PQNNFloat;
  LatCh,
  ZSpatial,
  B,
  C,
  UnpatchH,
  UnpatchW,
  CurW,
  CurH,
  MidCh,
  Progress,
  TotalBlocks,
  BlockIdx,
  UpIdx,
  Level,
  ChOut,
  R,
  NewH,
  NewW,
  OutCh,
  H,
  W: LongInt;
  Mean, Std: Single;

begin
  if Assigned(PhaseCallback) then
    PhaseCallback('VAE Decode', False);
  // if not work1.isAssigned() then work1:=TMemoryBlock.Create( , 'VAE_WORK1');
  // if not work2.isAssigned() then work1:=TMemoryBlock.Create( , 'VAE_WORK1');
  if not Work1.IsAllocated() then
    Work1 := TMemoryBlock.Create([Batch, 4, BaseChannels, LatentH * 16, LatentW * 16], 'VAE_WORK1');
  if not Work2.IsAllocated() then
    Work2 := TMemoryBlock.Create([Batch, 4, BaseChannels, LatentH * 16, LatentW * 16], 'VAE_WORK2');
  if not Work3.IsAllocated() then
    Work3 := TMemoryBlock.Create([Batch, 4, BaseChannels, LatentH * 16, LatentW * 16], 'VAE_WORK3');

  X := Work1;
  Work := Work2;
  LatCh := LatentChannels;
  ZSpatial := LatentH * LatentW;
  X.Shape := [Batch, LatCh, LatentH, LatentW];
  // if x.size < product(x.shape) then x.reSize(x.shape, latent.DataType);
  QNNCopy(X, Latent, Batch * LatCh * ZSpatial);
  // printCompare( batch * lat_ch * z_spatial, x, readTensor());
  if ScalingFactor <> 0.0 then
    QNNFusedScaleBias(X, X, 1 / ScalingFactor, ShiftFactor, Batch * LatCh * ZSpatial)
        // begin
        //    n := batch * lat_ch * z_spatial;
        //    for i := 0 to n -1 do
        //        x[i] := x[i]/scaling_factor + shift_factor
        // end
  else
  begin
    // QNNBatchNorm(x, x, bn_mean, bn_var, nil, nil, batch, lat_ch, latent_h, latent_w);
    MeanPtr := BnMean;
    VarPtr := BnVar;

    for C := 0 to LatCh - 1 do
    begin
      Mean := MeanPtr[C];
      Std := Sqrt(VarPtr[C] + Eps);
      for B := 0 to Batch - 1 do
        QNNFusedScaleBias(X + (B * LatCh + C) * ZSpatial, X + (B * LatCh + C) * ZSpatial, Std, Mean, ZSpatial);
      // for i := 0 to z_spatial -1 do
      //    begin
      //        idx := b * lat_ch * z_spatial+c * z_spatial+i;
      //        x[idx] := x[idx] * std + mean
      //    end
    end;
  end;
  // printCompare(batch * lat_ch * z_spatial, x, readTensor());
  UnpatchH := LatentH * 2;
  UnpatchW := LatentW * 2;
  // QNNUnpatchify(work, x, batch, z_channels, latent_h, latent_w, 2);
  X.Shape := [Batch, ZChannels, 2, 2, LatentH, LatentW];
  // if work.size<x.size then
  //    work.resize(x.shape, x.DataType)
  // else
  Work.Shape := X.Shape;
  QNNPermut(Work, X, [0, 1, 4, 2, 5, 3]);
  QNNCopy(X, Work, Batch * ZChannels * UnpatchH * UnpatchW);
  // printCompare(batch * z_channels * unpatch_h * unpatch_w, x, readTensor());
  CurH := UnpatchH;
  CurW := UnpatchW;

  if Boolean(PostQuantConvWeight) then
  begin
    QNNConv2d(Work, X, PostQuantConvWeight, PostQuantConvBias, ZChannels, ZChannels, CurH, CurW, 1, 1, 1, 0, Batch);
    // if x.size<work.size then x.reSize(work.shape, work.DataType);
    QNNCopy(X, Work, Batch * ZChannels * CurH * CurW)
  end;
  MidCh := BaseChannels * ChMult[3];
  QNNConv2d(Work, X, DecConvInWeight, DecConvInBias, ZChannels, MidCh, CurH, CurW, 3, 3, 1, 1, Batch);
  // if x.size<work.size then x.reSize(work.shape, work.DataType);
  QNNCopy(X, Work, Batch * MidCh * CurH * CurW);
  // printCompare(batch*mid_ch*cur_h*cur_w, x, readTensor());

  Progress := 0;
  TotalBlocks := 3 + 4 * (NumResBlocks + 1);
  DecMidBlock1.Forward(Work, X, Work3, Batch, CurH, CurW, NumGroups);

  if Assigned(VAEProgressCallback) then
  begin
    VAEProgressCallback(Progress, TotalBlocks);
    Inc(Progress)
  end;
  DecMidAttn.Forward(X, Work, Work3, Batch, CurH, CurW, NumGroups); // < 0 then

  // exit(nil);
  if Assigned(VAEProgressCallback) then
  begin
    VAEProgressCallback(Progress, TotalBlocks);
    Inc(Progress)
  end;
  DecMidBlock2.Forward(Work, X, Work3, Batch, CurH, CurW, NumGroups);
  // if x.size<work.size then x.reSize(work.shape, work.DataType);
  QNNCopy(X, Work, Batch * MidCh * CurH * CurW);
  if Assigned(VAEProgressCallback) then
  begin
    VAEProgressCallback(Progress, TotalBlocks);
    Inc(Progress)
  end;
  BlockIdx := 0;
  UpIdx := 0;
  Level := 3;
  while Level >= 0 do
  begin
    ChOut := BaseChannels * ChMult[Level];
    for R := 0 to NumResBlocks + 1 - 1 do
    begin
      DecUpBlocks[BlockIdx].Forward(Work, X, Work3, Batch, CurH, CurW, NumGroups);
      Inc(BlockIdx);
      // if x.size<work.size then x.reSize(work.shape, work.dataType);
      QNNCopy(X, Work, Batch * ChOut * CurH * CurW);
      if Assigned(VAEProgressCallback) then
      begin
        VAEProgressCallback(Progress, TotalBlocks);
        Inc(Progress)
      end;
    end;
    if Level > 0 then
    begin
      NewH := CurH * 2;
      NewW := CurW * 2;
      Work.Shape := [Batch, ChOut, NewH, NewW];
      // if work.size<product(work.shape) then
      //  work.resize(work.shape, work.dataType);
      QNNUpSampleNearest(Work, X, Batch, ChOut, CurH, CurW, 2, 2);
      QNNConv2d(
          X,
          Work,
          DecUpsample[UpIdx].ConvWeight,
          DecUpsample[UpIdx].ConvBias,
          ChOut,
          ChOut,
          NewH,
          NewW,
          3,
          3,
          1,
          1,
          Batch
      );
      Inc(UpIdx);
      CurH := NewH;
      CurW := NewW
    end;
    Dec(Level)
  end;
  OutCh := BaseChannels;

  Work.Shape := [Batch, OutCh, CurH, CurW];
  // if work.size<product(work.shape) then
  //  work.reSize(work.shape, work.DataType);

  QNNGroupNorm(Work, X, DecNormOutWeight, DecNormOutBias, Batch, OutCh, CurH, CurW, NumGroups);
  QNNSiluInplace(Work, Batch * OutCh * CurH * CurW);
  QNNConv2d(X, Work, DecConvOutWeight, DecConvOutBias, OutCh, 3, CurH, CurW, 3, 3, 1, 1, Batch);
  H := CurH;
  W := CurW;

  Result := TQNNImage.Create(W, H, 3, X);
  // if not assigned(result.data) then
  //    exit(nil);
  // for y := 0 to H -1 do
  //  for c := 0 to W -1 do
  //    for ch := 0 to 3 -1 do begin
  //      val := x[ch * H * W+y * W+c];
  //      val := (val+1.0) * 0.5;
  //      val := val * 255.0;
  //      if val < 0 then
  //          val := 0;
  //      if val > 255 then
  //          val := 255;
  //      result.data[(y*W + c)*3 + ch] := round(val)
  //    end;
  if Assigned(PhaseCallback) then
    PhaseCallback('VAE Decode', True);

end;

procedure TVAE.Decode(const Dst, Latent: TMemoryBlock; const Batch, LatentH, LatentW: LongInt);
const
  CH_MULT: array[0..3] of Integer = (1, 2, 4, 4);
var
  X, Work: TMemoryBlock;

  LatCh,
  ZSpatial,
  UnpatchH,
  UnpatchW,
  CurW,
  CurH,
  MidCh,
  Progress,
  TotalBlocks,
  BlockIdx,
  UpIdx,
  Level,
  ChOut,
  R,
  NewH,
  NewW,
  OutCh: LongInt;

begin
  if Assigned(PhaseCallback) then
    PhaseCallback('VAE Decode', False);
  X := Work1;
  Work := Work2;
  LatCh := LatentChannels;
  ZSpatial := LatentH * LatentW;
  QNNCopy(X, Latent, Batch * LatCh * ZSpatial);
  if ScalingFactor <> 0.0 then
    QNNFusedScaleBias(X, X, 1 / ScalingFactor, ShiftFactor, Batch * LatCh * ZSpatial)
        // begin
        //    n := batch * lat_ch * z_spatial;
        //    for i := 0 to n -1 do
        //        x[i] := x[i]/scaling_factor + shift_factor
        // end
  else
  begin
    QNNBatchNorm(X, X, BnMean, BnVar, Batch, LatCh, LatentH, LatentW);
    // x is [batch, latent_channels, latent_h, latent_w]
    // mean_ptr := bn_mean;
    // var_ptr := bn_var;
    // for c := 0 to lat_ch -1 do begin
    //    mean := mean_ptr[c];
    //    std := sqrt(var_ptr[c]+eps);
    //    for b := 0 to batch -1 do
    //        QNNFusedScaleBias(x+(b*lat_ch + c)*z_spatial, x+(b*lat_ch + c)*z_spatial, std, mean, z_spatial);// x is [batch, latent_channels, latent_h, latent_w]
    //        //for i := 0 to z_spatial -1 do
    //        //    begin
    //        //        idx := b * lat_ch * z_spatial+c * z_spatial+i;
    //        //        x[idx] := x[idx] * std + mean
    //        //    end
    // end;
  end;
  UnpatchH := LatentH * 2;
  UnpatchW := LatentW * 2;
  // QNNUnpatchify(work, x, batch, z_channels, latent_h, latent_w, 2);
  X.Shape := [Batch, ZChannels, 2, 2, LatentH, LatentW];
  Work.Shape := X.Shape;
  QNNPermut(Work, X, [0, 1, 4, 2, 5, 3]);
  QNNCopy(X, Work, Batch * ZChannels * UnpatchH * UnpatchW);
  // x and work is [batch, z_channels, unpatch_h, unpatch_w]
  CurH := UnpatchH;
  CurW := UnpatchW;
  if Boolean(PostQuantConvWeight) then
  begin
    QNNConv2d(Work, X, PostQuantConvWeight, PostQuantConvBias, ZChannels, ZChannels, CurH, CurW, 1, 1, 1, 0, Batch);
    QNNCopy(X, Work, Batch * ZChannels * CurH * CurW)
        // x and work is [batch, z_channel, latent_h*2, latent_w*2]
  end;
  MidCh := BaseChannels * CH_MULT[3];
  QNNConv2d(Work, X, DecConvInWeight, DecConvInBias, ZChannels, MidCh, CurH, CurW, 3, 3, 1, 1, Batch);
  QNNCopy(X, Work, Batch * MidCh * CurH * CurW);
  Progress := 0;
  TotalBlocks := 3 + 4 * (NumResBlocks + 1);
  DecMidBlock1.Forward(Work, X, Work3, Batch, CurH, CurW, NumGroups);
  if Assigned(VAEProgressCallback) then
  begin
    VAEProgressCallback(Progress, TotalBlocks);
    Inc(Progress)
  end;
  DecMidAttn.Forward(X, Work, Work3, Batch, CurH, CurW, NumGroups); // < 0 then
  // exit(nil);
  if Assigned(VAEProgressCallback) then
  begin
    VAEProgressCallback(Progress, TotalBlocks);
    Inc(Progress)
  end;
  DecMidBlock2.Forward(Work, X, Work3, Batch, CurH, CurW, NumGroups);
  QNNCopy(X, Work, Batch * MidCh * CurH * CurW);
  if Assigned(VAEProgressCallback) then
  begin
    VAEProgressCallback(Progress, TotalBlocks);
    Inc(Progress)
  end;
  BlockIdx := 0;
  UpIdx := 0;
  Level := 3;
  while Level >= 0 do
  begin
    ChOut := BaseChannels * CH_MULT[Level];
    for R := 0 to NumResBlocks + 1 - 1 do
    begin
      DecUpBlocks[BlockIdx].Forward(Work, X, Work3, Batch, CurH, CurW, NumGroups);
      Inc(BlockIdx);
      QNNCopy(X, Work, Batch * ChOut * CurH * CurW);
      if Assigned(VAEProgressCallback) then
      begin
        VAEProgressCallback(Progress, TotalBlocks);
        Inc(Progress)
      end;
    end;
    if Level > 0 then
    begin
      NewH := CurH * 2;
      NewW := CurW * 2;
      QNNUpSampleNearest(Work, X, Batch, ChOut, CurH, CurW, 2, 2);
      QNNConv2d(
          X,
          Work,
          DecUpsample[UpIdx].ConvWeight,
          DecUpsample[UpIdx].ConvBias,
          ChOut,
          ChOut,
          NewH,
          NewW,
          3,
          3,
          1,
          1,
          Batch
      );
      Inc(UpIdx);
      CurH := NewH;
      CurW := NewW
    end;
    Dec(Level)
  end;
  OutCh := BaseChannels;
  QNNGroupNorm(Work, X, DecNormOutWeight, DecNormOutBias, Batch, OutCh, CurH, CurW, NumGroups);
  QNNSiluInplace(Work, Batch * OutCh * CurH * CurW);
  QNNConv2d(X, Work, DecConvOutWeight, DecConvOutBias, OutCh, 3, CurH, CurW, 3, 3, 1, 1, Batch);
  QNNCopy(Dst, X, 4 * CurW * CurH);
  if Assigned(PhaseCallback) then
    PhaseCallback('VAE Decode', True);
end;

class function TVAE.Preview(
    const Latents: TMemoryBlock;
    const LatentRGBProj: TArray<TArray<Single>>;
    const LatentHeight, LatentWidth, Dims, Frames: LongInt;
    const LatentRGBBias: TArray<Single>;
    const PatchSize: LongInt
): TQNNImage;
var
  RGBWidth, RGBHeight, UnpatchedDim, K, RGBY, RGBX, LatentY, LatentX, ChannelOffset, LatentChannel, PixelId, D: LongInt;
  R, G, B, Value: Single;
  LatentsPtr: PQNNFloat;
begin

  RGBWidth := LatentWidth * PatchSize;
  RGBHeight := LatentHeight * PatchSize;

  LatentsPtr := Latents;
  UnpatchedDim := Dims div (PatchSize * PatchSize);
  Result := TQNNImage.Create(RGBWidth, RGBHeight);
  for K := 0 to Frames - 1 do
  begin
    for RGBX := 0 to RGBWidth - 1 do
    begin
      for RGBY := 0 to RGBHeight - 1 do
      begin
        LatentX := RGBX div PatchSize;
        LatentY := RGBY div PatchSize;

        ChannelOffset := 0;
        if PatchSize > 1 then
          ChannelOffset := ((RGBY mod PatchSize) * PatchSize + (RGBX mod PatchSize));

        // should be incremented by 1 for each pixel
        PixelId := K * RGBWidth * RGBHeight + RGBY * RGBWidth + RGBX;

        R := 0;
        G := 0;
        B := 0;
        if Assigned(LatentRGBProj) then
        begin
          for D := 0 to UnpatchedDim - 1 do
          begin
            LatentChannel := D * PatchSize * PatchSize + ChannelOffset;
            Value := LatentsPtr[LatentX + LatentWidth * (LatentY + LatentHeight * LatentChannel)];
            R := R + Value * LatentRGBProj[D][0];
            G := G + Value * LatentRGBProj[D][1];
            B := B + Value * LatentRGBProj[D][2];
          end
        end
        else
        begin
          // interpret first 3 channels as RGB
          R := LatentsPtr[LatentX + LatentWidth * (LatentY + LatentHeight * 0)];
          G := LatentsPtr[LatentX + LatentWidth * (LatentY + LatentHeight * 1)];
          B := LatentsPtr[LatentX + LatentWidth * (LatentY + LatentHeight * 2)];
        end;
        if Assigned(LatentRGBBias) then
        begin
          // bias
          R := R + LatentRGBBias[0];
          G := G + LatentRGBBias[1];
          B := B + LatentRGBBias[2];
        end;
        // change range to from -1..1 to 0..1
        R := R * 0.5 + 0.5;
        G := G * 0.5 + 0.5;
        B := B * 0.5 + 0.5;

        // clamp rgb values to [0,1] range
        if R > 1.0 then
          R := 1.0
        else if R < 0.0 then
          R := 0.;
        if G > 1.0 then
          G := 1.0
        else if G < 0.0 then
          G := 0.;
        if B > 1.0 then
          B := 1.0
        else if B < 0.0 then
          B := 0.;

        Result.Data[PixelId * 3 + 0] := Trunc(R * 255);
        Result.Data[PixelId * 3 + 1] := Trunc(G * 255);
        Result.Data[PixelId * 3 + 2] := Trunc(B * 255);
      end
    end
  end
end;

// procedure TVAE.load(var f: file);
// var i, ch, mid_ch:longint;
// begin
//    z_channels      := readInt(f);
//    latent_channels := z_channels * 4;  (* 2x2 patchify *)
//    base_channels   := readInt(f);
//    num_res_blocks  := readInt(f);
//    num_groups      := readInt(f);
//    max_h           := readInt(f);
//    max_w           := readInt(f);
//    scaling_factor  := 0.0;
//    shift_factor    := 0.0;
//
//    ch_mult    := QNN_VAE_CH_MULT;
//    //ch_mult[0] := QNN_VAE_CH_MULT_0;
//    //ch_mult[1] := QNN_VAE_CH_MULT_1;
//    //ch_mult[2] := QNN_VAE_CH_MULT_2;
//    //ch_mult[3] := QNN_VAE_CH_MULT_3;
//
//    eps := EPSILON;  (* batch_norm_eps from config *)
//
//    (* Read encoder conv_in *)
//    enc_conv_in_weight := readSingles(f, base_channels*3*3*3);
//    enc_conv_in_bias   := readSingles(f, base_channels);
//
//    (* Read encoder down blocks *)
//    setLength(enc_down_blocks, 4*num_res_blocks);
//    for i := 0 to high(enc_down_blocks) do
//        enc_down_blocks[i].load(f);
//
//
//    (* Read encoder downsamples *)
//    setLength(enc_downsample, 3);;
//    for i:=0 to high(enc_downsample) do begin
//        ch := base_channels*ch_mult[i];
//        enc_downsample[i].channels := ch;
//        enc_downsample[i].conv_weight := readSingles(f, ch *ch*3*3);
//        enc_downsample[i].conv_bias   := readSingles(f, ch);
//    end;
//
//    (* Read encoder mid block *)
//    enc_mid_block1.load(f);
//    enc_mid_attn.load(f);
//    enc_mid_block2.load(f);
//
//    (* Read encoder output *)
//    mid_ch := base_channels*ch_mult[3];
//    enc_norm_out_weight := readSingles(f, mid_ch);
//    enc_norm_out_bias   := readSingles(f, mid_ch);
//    enc_conv_out_weight := readSingles(f, z_channels*2*mid_ch*3*3);
//    enc_conv_out_bias   := readSingles(f, z_channels*2);
//
//    (* Read decoder conv_in *)
//    dec_conv_in_weight := readSingles(f, mid_ch*z_channels*3*3);
//    dec_conv_in_bias   := readSingles(f, mid_ch);
//
//    (* Read decoder mid block *)
//    dec_mid_block1.load(f);
//    dec_mid_attn.load(f);
//    dec_mid_block2.load(f);
//
//    (* Read decoder up blocks *)
//    setLength(dec_up_blocks, 4 * (num_res_blocks + 1));
//    for i := 0 to high(dec_up_blocks) do
//        dec_up_blocks[i].load(f);
//
//    (* Read decoder upsamples *)
//    setLength(dec_upsample,3);
//    for i := 0 to high(dec_upsample) do begin
//        ch := base_channels * ch_mult[3 - i];
//        dec_upsample[i].channels := ch;
//        dec_upsample[i].conv_weight := readSingles(f, ch * ch * 3 * 3);
//        dec_upsample[i].conv_bias := readSingles(f, ch);
//    end;
//
//    (* Read decoder output *)
//    dec_norm_out_weight := readSingles(f, base_channels);
//    dec_norm_out_bias   := readSingles(f, base_channels);
//    dec_conv_out_weight := readSingles(f, 3*base_channels * 3 * 3);
//    dec_conv_out_bias   := readSingles(f, 3);
//
//    (* Read batch norm stats *)
//    bn_mean := readSingles(f, latent_channels);
//    bn_var  := readSingles(f, latent_channels);
//
//    (* Allocate working memory *)
//    work_size := 4*mid_ch*max_h*max_w;
//    work1     := TMemoryBlock.create(work_size);
//    work2     := TMemoryBlock.create(work_size);
//    work3     := TMemoryBlock.create(work_size);
//
// end;

procedure TVAE.Load(
    const Sf: TSafeTensorFiles;
    const AZChannels: LongInt;
    const AScalingFactor, AShiftFactor: QNNFloat
);
var
  NumDownBlocks, BlockIdx, Level, Ch, PrevCh, R, InCh, I, MidCh, NumUpBlocks, UpIdx, Lc, MaxSpatial, MaxChannels:
      LongInt;
  BnMeanT, BnVarT: PSafeTensor;
begin

  // Retain independent record copies while F32 weights borrow mmap pointers.
  if UseMMAP then
    MappedFiles := Copy(Sf);
  ZChannels := AZChannels;
  LatentChannels := ZChannels * 4;
  BaseChannels := 128;
  NumResBlocks := 2;
  NumGroups := 32;
  MaxH := QNN_VAE_MAX_DIM;
  MaxW := QNN_VAE_MAX_DIM;
  ScalingFactor := AScalingFactor;
  ShiftFactor := AShiftFactor;
  if (ScalingFactor <> 0.0) then
    Eps := 1e-6
  else
    Eps := 1e-4;
  ChMult := QNN_VAE_CH_MULT;
  EncConvInWeight := Sf.GetTensorDataMemBlock('encoder.conv_in.weight', UseMMAP);
  EncConvInBias := Sf.GetTensorDataMemBlock('encoder.conv_in.bias', UseMMAP);
  NumDownBlocks := 4 * NumResBlocks;
  SetLength(EncDownBlocks, NumDownBlocks);
  BlockIdx := 0;
  for Level := 0 to 4 - 1 do
  begin
    Ch := BaseChannels * ChMult[Level];
    if (Level = 0) then
      PrevCh := BaseChannels
    else
      PrevCh := BaseChannels * ChMult[Level - 1];
    for R := 0 to NumResBlocks - 1 do
    begin
      if ((R = 0) and (Level > 0)) then
        InCh := PrevCh
      else
        InCh := Ch;
      EncDownBlocks[BlockIdx].Load(Sf, Format('encoder.down_blocks.%d.resnets.%d', [Level, R]), InCh, Ch, UseMMAP);
      Inc(BlockIdx)
    end
  end;
  SetLength(EncDownsample, 3);
  for I := 0 to High(EncDownsample) do
  begin
    Ch := BaseChannels * ChMult[I];
    EncDownsample[I].Channels := Ch;
    EncDownsample[I].ConvWeight :=
        Sf.GetTensorDataMemBlock(Format('encoder.down_blocks.%d.downsamplers.0.conv.weight', [I]), UseMMAP);
    EncDownsample[I].ConvBias :=
        Sf.GetTensorDataMemBlock(Format('encoder.down_blocks.%d.downsamplers.0.conv.bias', [I]), UseMMAP)
  end;
  MidCh := BaseChannels * ChMult[3];
  EncMidBlock1.Load(Sf, 'encoder.mid_block.resnets.0', MidCh, MidCh, UseMMAP);
  EncMidAttn.Load(Sf, 'encoder.mid_block.attentions.0', MidCh, UseMMAP);
  EncMidBlock2.Load(Sf, 'encoder.mid_block.resnets.1', MidCh, MidCh, UseMMAP);
  EncNormOutWeight := Sf.GetTensorDataMemBlock('encoder.conv_norm_out.weight', UseMMAP);
  EncNormOutBias := Sf.GetTensorDataMemBlock('encoder.conv_norm_out.bias', UseMMAP);
  EncConvOutWeight := Sf.GetTensorDataMemBlock('encoder.conv_out.weight', UseMMAP);
  EncConvOutBias := Sf.GetTensorDataMemBlock('encoder.conv_out.bias', UseMMAP);
  if Assigned(Sf.GetTensor('quant_conv.weight')) then
  begin
    QuantConvWeight := Sf.GetTensorDataMemBlock('quant_conv.weight', UseMMAP);
    QuantConvBias := Sf.GetTensorDataMemBlock('quant_conv.bias', UseMMAP)
  end;
  DecConvInWeight := Sf.GetTensorDataMemBlock('decoder.conv_in.weight', UseMMAP);
  DecConvInBias := Sf.GetTensorDataMemBlock('decoder.conv_in.bias', UseMMAP);
  DecMidBlock1.Load(Sf, 'decoder.mid_block.resnets.0', MidCh, MidCh, UseMMAP);
  DecMidAttn.Load(Sf, 'decoder.mid_block.attentions.0', MidCh, UseMMAP);
  DecMidBlock2.Load(Sf, 'decoder.mid_block.resnets.1', MidCh, MidCh, UseMMAP);
  NumUpBlocks := 4 * (NumResBlocks + 1);
  SetLength(DecUpBlocks, NumUpBlocks);
  BlockIdx := 0;
  Level := 3;
  while Level >= 0 do
  begin
    Ch := BaseChannels * ChMult[Level];
    if (Level = 3) then
      PrevCh := MidCh
    else
      PrevCh := BaseChannels * ChMult[Level + 1];
    for R := 0 to NumResBlocks + 1 - 1 do
    begin
      if (R = 0) then
        InCh := PrevCh
      else
        InCh := Ch;
      UpIdx := 3 - Level;
      DecUpBlocks[BlockIdx].Load(Sf, Format('decoder.up_blocks.%d.resnets.%d', [UpIdx, R]), InCh, Ch, UseMMAP);
      Inc(BlockIdx);
    end;
    Dec(Level)
  end;
  SetLength(DecUpsample, 3);
  for I := 0 to High(DecUpsample) do
  begin
    Ch := BaseChannels * ChMult[3 - I];
    DecUpsample[I].Channels := Ch;
    DecUpsample[I].ConvWeight :=
        Sf.GetTensorDataMemBlock(Format('decoder.up_blocks.%d.upsamplers.0.conv.weight', [I]), UseMMAP);
    DecUpsample[I].ConvBias :=
        Sf.GetTensorDataMemBlock(Format('decoder.up_blocks.%d.upsamplers.0.conv.bias', [I]), UseMMAP)
  end;
  DecNormOutWeight := Sf.GetTensorDataMemBlock('decoder.conv_norm_out.weight', UseMMAP);
  DecNormOutBias := Sf.GetTensorDataMemBlock('decoder.conv_norm_out.bias', UseMMAP);
  DecConvOutWeight := Sf.GetTensorDataMemBlock('decoder.conv_out.weight', UseMMAP);
  DecConvOutBias := Sf.GetTensorDataMemBlock('decoder.conv_out.bias', UseMMAP);
  BnMeanT := Sf.GetTensor('bn.running_mean');
  if Assigned(BnMeanT) then
  begin
    BnMean := BnMeanT.AsSingle(UseMMAP);
    BnVarT := Sf.GetTensor('bn.running_var');
    if Assigned(BnVarT) then
      BnVar := BnVarT.AsSingle(UseMMAP)
    else
      BnVar := Default(TMemoryBlock) // nil
  end;
  if Assigned(Sf.GetTensor('post_quant_conv.weight')) then
  begin
    PostQuantConvWeight := Sf.GetTensorDataMemBlock('post_quant_conv.weight', UseMMAP);
    PostQuantConvBias := Sf.GetTensorDataMemBlock('post_quant_conv.bias', UseMMAP)
  end;
  if not Boolean(BnMean) and (ScalingFactor = 0.0) then
  begin
    Lc := LatentChannels;
    BnMean := TMemoryBlock.Create([Lc], 'VAE_LOAD_BN_MEAN');
    BnVar := TMemoryBlock.Create([Lc], 'VAE_LOAD_BN_VAR');
    QNNFill(BnVar, 1.0, Lc);
    BnMean.MarkImmutable;
    BnVar.MarkImmutable;
  end;
  MaxSpatial := MaxH * MaxW;
  MaxChannels := BaseChannels;
  WorkSize := 4 * MaxChannels * MaxSpatial;
  //
  // work1 := TMemoryBlock.Create([work_size], 'VAE_WORK1');//calloc(work_size, sizeof(single));//;
  // work2 := TMemoryBlock.Create([work_size], 'VAE_WORK2');//calloc(work_size, sizeof(single));//;
  // work3 := TMemoryBlock.Create([work_size], 'VAE_WORK3');//calloc(work_size, sizeof(single));//
end;

class procedure TVAE.PadRightBottom(const Dst, Src: TMemoryBlock; Batch, Channels, H, W: LongInt);
var
  Hp, Wp, B, C: LongInt;
  InPlane, OutPlane: IntPtr;
  S, D: TMemoryBlock;
  Y: LongInt;
begin
  Hp := H + 1;
  Wp := W + 1;
  InPlane := H * W;
  OutPlane := Hp * Wp;
  QNNFill(Dst, 0, Batch * Channels * OutPlane);
  for B := 0 to Batch - 1 do
    for C := 0 to Channels - 1 do
    begin
      S := Src + (B * Channels + C) * InPlane;
      D := Dst + (B * Channels + C) * OutPlane;
      for Y := 0 to H - 1 do
        QNNCopy(D + Y * Wp, S + Y * W, W)
    end
end;

procedure TVAE.Free;
var
  I: LongInt;
begin
  QNNGPUClearWeightCache;

  EncConvInWeight.Free;
  EncConvInBias.Free;
  for I := 0 to High(EncDownBlocks) do
    EncDownBlocks[I].Free;
  EncDownBlocks := nil;
  for I := 0 to High(EncDownsample) do
    EncDownsample[I].Free;
  EncDownsample := nil;
  EncMidBlock1.Free;
  EncMidAttn.Free;
  EncMidBlock2.Free;
  EncNormOutWeight.Free;
  EncNormOutBias.Free;
  EncConvOutWeight.Free;
  EncConvOutBias.Free;
  QuantConvWeight.Free;
  QuantConvBias.Free;
  DecConvInWeight.Free;
  DecConvInBias.Free;
  DecMidBlock1.Free;
  DecMidAttn.Free;
  DecMidBlock2.Free;
  for I := 0 to High(DecUpBlocks) do
    DecUpBlocks[I].Free;
  DecUpBlocks := nil;
  for I := 0 to High(DecUpsample) do
  begin
    DecUpsample[I].ConvWeight.Free;
    DecUpsample[I].ConvBias.Free;
  end;
  DecUpsample := nil;
  DecNormOutWeight.Free;
  DecNormOutBias.Free;
  DecConvOutWeight.Free;
  DecConvOutBias.Free;
  BnMean.Free;
  BnVar.Free;
  PostQuantConvWeight.Free;
  PostQuantConvBias.Free;
  BnMean.Free;
  BnVar.Free;
  Work1.Free;
  Work2.Free;
  Work3.Free;
  for I := 0 to High(MappedFiles) do
    MappedFiles[I].Close;
  Self := Default(TVAE)
end;
end.
