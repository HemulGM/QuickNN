unit QuickNN.Models.Qwen21.Transformer;

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
  System.SysUtils, QuickNN.Common, QuickNN.Quantization, QuickNN.SafeTensors,
  QuickNN.Vulkan, QuickNN.Models.Qwen21.Math, QuickNN.LoRA,
  QuickNN.Models.Qwen21.Conditioning;

type
  TQwen21GPUCondition = record
    Text, Positions, Limits: TQNNGPUTensor;
    References: TArray<TQNNGPUTensor>;
    Layout: TQwen21SequenceLayout;
    procedure Free(const Ctx: TQNNGPUContext);
  end;

  TQwen21GPUWork = record
    X, N, Q, K, V, A, Y, G, Active, T, T1, TEmb, Modulation, OutScale: TQNNGPUTensor;
    procedure Ensure(const Ctx: TQNNGPUContext; const Rows: Integer);
    procedure Free(const Ctx: TQNNGPUContext);
  end;

type
  TQwen21Block = record
    QNorm, KNorm: TMemoryBlock;
    Q, K, V, O, GateUp, Gate, Up, Down: TQNNLinearWeight;
    procedure Load(const Files: TSafeTensorFiles; const Index: Integer);
    procedure Free;
  end;

  TQwen21Transformer = class
  private
    FFiles: TSafeTensorFiles;
    FBlocks: TArray<TQwen21Block>;
    FTextNorm, FText: TMemoryBlock;
    FReferences: TQwen21ImageConditions;
    FReferenceSlots: TArray<Integer>;
    FReferenceProjected: TArray<TMemoryBlock>;
    FTextIn, FTextOut, FImageIn, FTime1, FTime2, FModulation, FNormOut, FOut: TQNNLinearWeight;
    FStatus: TQwen21Status;
    function ResolveLinear(const Target: string; out OutputOffset: Integer): PQNNLinearWeight;
  public
    constructor Create(const Checkpoint: string; const Status: TQwen21Status = nil);
    destructor Destroy; override;
    function ConditionGPU(const Ctx: TQNNGPUContext; const Text: TQNNGPUTensor; const Rows: Integer; const Images: TQwen21ImageConditions; const resident: TQwen21GPUImageConditions; const Slots: TArray<Integer>; const Height, Width: Integer): TQwen21GPUCondition;
    procedure ForwardGPU(const Ctx: TQNNGPUContext; const Dst, Latents: TQNNGPUTensor; const Condition: TQwen21GPUCondition; const Height, Width: Integer; const Sigma: Single; var Work: TQwen21GPUWork);
    procedure AttachLoRAs(const Adapters: TQNNLoRASet);
    procedure SetText(const Text: TMemoryBlock; const Rows: Integer);
    procedure SetConditioning(const Text: TMemoryBlock; const Rows: Integer; const Images: TQwen21ImageConditions; const Slots: TArray<Integer>);
    function forward(const Latents: TMemoryBlock; const Height, Width: Integer; const Sigma: Single): TMemoryBlock;
  end;

implementation

uses
  System.Math, QuickNN.Kernels;

const
  HIDDEN = 4096;
  MLP = 12288;
  HEADS = 32;
  HEAD_DIM = 128;

procedure TQwen21Block.Load(const Files: TSafeTensorFiles; const Index: Integer);
var
  P: string;
begin
  P := Format('transformer_blocks.%d.', [Index]);
  QNorm := QNNRequiredTensor(Files, P + 'attn.norm_q.weight').AsSingle(True);
  KNorm := QNNRequiredTensor(Files, P + 'attn.norm_k.weight').AsSingle(True);
  Q.Load(Files, P + 'attn.to_q');
  K.Load(Files, P + 'attn.to_k');
  V.Load(Files, P + 'attn.to_v');
  O.Load(Files, P + 'attn.to_out.0');
  if Files.GetTensor(P + 'img_mlp.gate_up.weight') <> nil then
    GateUp.Load(Files, P + 'img_mlp.gate_up')
  else
  begin
    Gate.Load(Files, P + 'img_mlp.gate_layer');
    Up.Load(Files, P + 'img_mlp.proj');
    if (Gate.InputDim <> HIDDEN) or (Gate.OutputDim <> MLP)
      or (Up.InputDim <> HIDDEN) or (Up.OutputDim <> MLP) then
      raise EArgumentException.Create('Invalid Qwen-Image 2.1 split MLP dimensions');
  end;
  Down.Load(Files, P + 'img_mlp.out');
  if (Q.InputDim <> HIDDEN)
    or (Q.OutputDim <> HIDDEN)
    or (K.InputDim <> HIDDEN)
    or (K.OutputDim <> HIDDEN)
    or (V.InputDim <> HIDDEN)
    or (V.OutputDim <> HIDDEN)
    or (O.InputDim <> HIDDEN)
    or (O.OutputDim <> HIDDEN)
    or (GateUp.Weight.IsAssigned and ((GateUp.InputDim <> HIDDEN) or (GateUp.OutputDim <> 2 * MLP)))
    or (Down.InputDim <> MLP)
    or (Down.OutputDim <> HIDDEN) then
    raise EArgumentException.Create('Invalid Qwen-Image 2.1 block dimensions');
end;

procedure TQwen21Block.Free;
begin
  QNorm.Free;
  KNorm.Free;
  Q.Free;
  K.Free;
  V.Free;
  O.Free;
  GateUp.Free;
  Down.Free
end;

constructor TQwen21Transformer.Create(const Checkpoint: string; const Status: TQwen21Status);
begin
  inherited Create;
  FStatus := Status;
  SetLength(FFiles, 1);
  FFiles[0] := TSafeTensorFile.Open(Checkpoint);
  if FFiles[0].IsGGUF and (FFiles[0].Architecture <> 'qwen_image21') then
    raise EArgumentException.Create('Expected GGUF architecture qwen_image21; found ' + FFiles[0].Architecture);
  FTextNorm := QNNRequiredTensor(FFiles, 'txt_in.text_norm.weight').AsSingle(True);
  FTextIn.Load(FFiles, 'txt_in.in_layer');
  FTextOut.Load(FFiles, 'txt_in.out_layer');
  FImageIn.Load(FFiles, 'img_in');
  FTime1.Load(FFiles, 'time_text_embed.timestep_embedder.linear_1');
  FTime2.Load(FFiles, 'time_text_embed.timestep_embedder.linear_2');
  FModulation.Load(FFiles, 'modulation.1');
  FNormOut.Load(FFiles, 'norm_out.linear');
  FOut.Load(FFiles, 'proj_out');
  if (FTextNorm.Count <> HIDDEN)
    or (FImageIn.InputDim <> 64)
    or (FImageIn.OutputDim <> HIDDEN)
    or (FOut.InputDim <> HIDDEN)
    or (FOut.OutputDim <> 64)
    or (FTime1.InputDim <> 256)
    or (FTime1.OutputDim <> HIDDEN)
    or (FModulation.OutputDim <> 4 * HIDDEN) then
    raise EArgumentException.Create('Expected Qwen-Image 2.1 diffusion architecture');
  SetLength(FBlocks, 32);
  for var I := 0 to High(FBlocks) do
    FBlocks[I].Load(FFiles, I);
end;

destructor TQwen21Transformer.Destroy;
begin
  QNNGPUClearWeightCache;
  for var I := 0 to High(FBlocks) do
    FBlocks[I].Free;
  for var I := 0 to High(FReferenceProjected) do
    FReferenceProjected[I].Free;
  FTextNorm.Free;
  FText.Free;
  FTextIn.Free;
  FTextOut.Free;
  FImageIn.Free;
  FTime1.Free;
  FTime2.Free;
  FModulation.Free;
  FNormOut.Free;
  FOut.Free;
  for var I := 0 to High(FFiles) do
    FFiles[I].Close;
  inherited;
end;

function TQwen21Transformer.ResolveLinear(const Target: string; out OutputOffset: Integer): PQNNLinearWeight;
var
  Parts: TArray<string>;
  Index: Integer;
  Name: string;
begin
  Result := nil;
  OutputOffset := 0;
  if Target = 'txt_in.in_layer' then
    Exit(@FTextIn);
  if Target = 'txt_in.out_layer' then
    Exit(@FTextOut);
  if Target = 'img_in' then
    Exit(@FImageIn);
  if Target = 'time_text_embed.timestep_embedder.linear_1' then
    Exit(@FTime1);
  if Target = 'time_text_embed.timestep_embedder.linear_2' then
    Exit(@FTime2);
  if Target = 'modulation.1' then
    Exit(@FModulation);
  if Target = 'norm_out.linear' then
    Exit(@FNormOut);
  if Target = 'proj_out' then
    Exit(@FOut);
  Parts := Target.Split(['.']);
  if (Length(Parts) < 4)
    or (Parts[0] <> 'transformer_blocks')
    or not TryStrToInt(Parts[1], Index)
    or (Index < 0)
    or (Index >= Length(FBlocks)) then
    Exit;
  Name := string.Join('.', Parts, 2, Length(Parts) - 2);
  if Name = 'attn.to_q' then
    Result := @FBlocks[Index].Q
  else if Name = 'attn.to_k' then
    Result := @FBlocks[Index].K
  else if Name = 'attn.to_v' then
    Result := @FBlocks[Index].V
  else if Name = 'attn.to_out.0' then
    Result := @FBlocks[Index].O
  else if (Name = 'img_mlp.gate_up') and FBlocks[Index].GateUp.Weight.IsAssigned then
    Result := @FBlocks[Index].GateUp
  else if Name = 'img_mlp.gate_layer' then
  begin
    if FBlocks[Index].Gate.Weight.IsAssigned then Result := @FBlocks[Index].Gate
    else Result := @FBlocks[Index].GateUp;
  end
  else if Name = 'img_mlp.proj' then
  begin
    if FBlocks[Index].Up.Weight.IsAssigned then Result := @FBlocks[Index].Up
    else
    begin
      Result := @FBlocks[Index].GateUp;
      OutputOffset := MLP;
    end;
  end
  else if Name = 'img_mlp.out' then
    Result := @FBlocks[Index].Down;
end;

procedure TQwen21Transformer.AttachLoRAs(const Adapters: TQNNLoRASet);
begin
  if Adapters = nil then
    Exit;
  Adapters.Attach(
    function(const Target: string; out OutputOffset: Integer): PQNNLinearWeight
    begin
      Result := ResolveLinear(Target, OutputOffset)
    end);
end;

procedure TQwen21Transformer.SetText(const Text: TMemoryBlock; const Rows: Integer);
var
  Norm, Projected: TMemoryBlock;
begin
  if (Rows < 1) or (Rows > 6144) or (Text.Count < Int64(Rows) * HIDDEN) then
    raise EArgumentException.Create('Invalid text conditioning');
  FText.Free;
  for var I := 0 to High(FReferenceProjected) do
    FReferenceProjected[I].Free;
  FReferenceProjected := nil;
  FReferences := nil;
  FReferenceSlots := nil;
  try
    Norm := TMemoryBlock.Create([Rows, HIDDEN], 'qwen21-text-norm');
    Projected := TMemoryBlock.Create([Rows, HIDDEN], 'qwen21-text-projected');
    FText := TMemoryBlock.Create([Rows, HIDDEN], 'qwen21-text');
    Q21RMS(Norm, Text, FTextNorm, Rows, HIDDEN, True);
    FTextIn.forward(Projected, Norm, Rows);
    Q21Gelu(Projected, Rows * HIDDEN);
    FTextOut.forward(FText, Projected, Rows);
    QNNCheckFinite(FText, Rows * HIDDEN, 'Projected text');
  finally
    Norm.Free;
    Projected.Free
  end;
end;

procedure TQwen21Transformer.SetConditioning(const Text: TMemoryBlock; const Rows: Integer; const Images: TQwen21ImageConditions; const Slots: TArray<Integer>);
var
  Previous, RefRows: Integer;
begin
  if Length(Images) <> Length(Slots) then
    raise EArgumentException.Create('Reference slot count mismatch');
  Previous := 0;
  for var I := 0 to High(Images) do
  begin
    if (Slots[I] < Previous) or (Slots[I] > Rows) then
      raise EArgumentException.Create('Invalid reference slot order');
    Previous := Slots[I];
    RefRows := (Images[I].Pixels.Height div 16) * (Images[I].Pixels.Width div 16);
    if (RefRows < 1) or (Images[I].Latents.Count <> Int64(RefRows) * 64) then
      raise EArgumentException.Create('Invalid reference latents');
  end;
  SetText(Text, Rows);
  FReferences := Images;
  FReferenceSlots := Copy(Slots);
  SetLength(FReferenceProjected, Length(Images));
  for var I := 0 to High(Images) do
  begin
    RefRows := Images[I].Latents.Count div 64;
    FReferenceProjected[I] := TMemoryBlock.Create([RefRows, HIDDEN], 'qwen21-reference-projection');
    FImageIn.forward(FReferenceProjected[I], Images[I].Latents, RefRows);
  end;
end;

function TQwen21Transformer.forward(const Latents: TMemoryBlock; const Height, Width: Integer; const Sigma: Single): TMemoryBlock;
var
  Layout: TQwen21SequenceLayout;
  TextRows, ImageRows, Rows, PrefixRows: Integer;
  X, N, Q, K, V, A, Y, G, Active, T, T1, TEmb, Modulation, OutScale: TMemoryBlock;
  GateView, UpView: TMemoryBlock;
  Block: ^TQwen21Block;
  Angle: Double;
  Offset: NativeInt;
  Used: Boolean;
begin
  TextRows := FText.Count div HIDDEN;
  ImageRows := Height * Width;
  Layout := Q21ReferenceLayout(TextRows, FReferences, FReferenceSlots, Height, Width);
  Rows := Layout.Rows;
  PrefixRows := Layout.PrefixRows;
  if (Height <= 0) or (Width <= 0) or (TextRows <= 0) or (Latents.Count <> Int64(ImageRows) * 64) then
    raise EArgumentException.Create('Invalid Qwen-Image 2.1 latent grid');
  try
    X := TMemoryBlock.Create([Rows, HIDDEN], 'qwen21-joint');
    N := TMemoryBlock.Create([Rows, HIDDEN], 'qwen21-norm');
    Q := TMemoryBlock.Create([Rows, HIDDEN], 'qwen21-q');
    K := TMemoryBlock.Create([Rows, HIDDEN], 'qwen21-k');
    V := TMemoryBlock.Create([Rows, HIDDEN], 'qwen21-v');
    A := TMemoryBlock.Create([Rows, HIDDEN], 'qwen21-attention');
    Y := TMemoryBlock.Create([Rows, HIDDEN], 'qwen21-output');
    T := TMemoryBlock.Create([2, 256], 'qwen21-time');
    T1 := TMemoryBlock.Create([2, HIDDEN], 'qwen21-time1');
    TEmb := TMemoryBlock.Create([2, HIDDEN], 'qwen21-temb');
    Modulation := TMemoryBlock.Create([2, 4 * HIDDEN], 'qwen21-modulation');
    OutScale := TMemoryBlock.Create([2, 4 * HIDDEN], 'qwen21-final-scale');
    for var Segment in Layout.Segments do
      if not Segment.IsImage then
        Move(
          System.PSingle(FText)[NativeInt(Segment.SourceStart) * HIDDEN],
          System.PSingle(X)[NativeInt(Segment.JointStart) * HIDDEN],
          NativeInt(Segment.Count) * HIDDEN * 4
        )
      else if Segment.SourceIndex < 0 then
        FImageIn.forward(X + NativeInt(Segment.JointStart) * HIDDEN, Latents, ImageRows)
      else
        Move(
          System.PSingle(FReferenceProjected[Segment.SourceIndex])^,
          System.PSingle(X)[NativeInt(Segment.JointStart) * HIDDEN],
          NativeInt(Segment.Count) * HIDDEN * 4
        );
    for var R := 0 to 1 do
      for var J := 0 to 127 do
      begin
        if R = 0 then
          Angle := Double(Sigma) * 1000 * Exp(-Ln(10000) * J / 128)
        else
          Angle := 0;
        System.PSingle(T)[R * 256 + J] := Cos(Angle);
        System.PSingle(T)[R * 256 + J + 128] := Sin(Angle);
      end;
    FTime1.forward(T1, T, 2);
    Q21Silu(T1, 2 * HIDDEN);
    FTime2.forward(TEmb, T1, 2);
    QNNCopy(T1, TEmb, 2 * HIDDEN);
    Q21Silu(T1, 2 * HIDDEN);
    FModulation.forward(Modulation, T1, 2);
    for var I := 0 to High(FBlocks) do
    begin
      Block := @FBlocks[I];
      Q21ModNorm(N, X, Modulation, Rows, HIDDEN, PrefixRows, 0);
      Block.Q.forward(Q, N, Rows);
      Block.K.forward(K, N, Rows);
      Block.V.forward(V, N, Rows);
      Q21HeadNorm(Q, Block.QNorm, Rows, HEADS, HEAD_DIM);
      Q21HeadNorm(K, Block.KNorm, Rows, HEADS, HEAD_DIM);
      Q21ImageRoPE(Q, Rows, HEADS, TextRows, Height, Width, Layout.Positions);
      Q21ImageRoPE(K, Rows, HEADS, TextRows, Height, Width, Layout.Positions);
      Q21Attention(A, Q, K, V, Rows, HEADS, HEADS, HEAD_DIM, PrefixRows, Layout.KeyLimits);
      Block.O.forward(Y, A, Rows);
      Q21GateAdd(X, Y, Modulation, Rows, HIDDEN, PrefixRows, HIDDEN);
      Q21ModNorm(N, X, Modulation, Rows, HIDDEN, PrefixRows, 2 * HIDDEN);
      Used := False;
      if not Block.GateUp.HasAdapters
        and not Block.Gate.HasAdapters
        and not Block.Up.HasAdapters
        and not Block.Down.HasAdapters
        and (Block.GateUp.Weight.DataType <> DT_I8)
        and (Block.Down.Weight.DataType <> DT_I8) then
      begin
        if Block.Gate.Weight.IsAssigned then
        begin
          GateView := Block.Gate.Weight;
          UpView := Block.Up.Weight;
        end
        else
        begin
          GateView := Block.GateUp.Weight;
          UpView := Block.GateUp.Weight + Int64(MLP) * HIDDEN;
        end;
        Used := TryVulkanSwiGLU(Y, N, GateView, UpView, Block.Down.Weight, Rows, HIDDEN, MLP);
        GateView := Default(TMemoryBlock);
        UpView := Default(TMemoryBlock);
      end;
      if not Used then
      begin
        if not G.IsAssigned then
        begin
          G := TMemoryBlock.Create([Rows, 2 * MLP], 'qwen21-gate-up');
          Active := TMemoryBlock.Create([Rows, MLP], 'qwen21-swiglu');
        end;
        if Block.Gate.Weight.IsAssigned then
        begin
          Block.Gate.forward(G, N, Rows);
          Block.Up.forward(Active, N, Rows);
          Q21Silu(G, Rows * MLP);
          for var J := 0 to Rows * MLP - 1 do
            System.PSingle(Active)[J] := System.PSingle(G)[J] * System.PSingle(Active)[J];
        end
        else
        begin
          Block.GateUp.forward(G, N, Rows);
          for var R := 0 to Rows - 1 do
          begin
            Offset := NativeInt(R) * 2 * MLP;
            Q21Silu(G + Offset, MLP);
            for var J := 0 to MLP - 1 do
              System.PSingle(Active)[NativeInt(R) * MLP + J] :=
                System.PSingle(G)[Offset + J] * System.PSingle(G)[Offset + MLP + J];
          end;
        end;
        Block.Down.forward(Y, Active, Rows);
      end;
      Q21GateAdd(X, Y, Modulation, Rows, HIDDEN, PrefixRows, 3 * HIDDEN);
      QNNCheckFinite(X, Rows * HIDDEN, 'Diffusion block ' + IntToStr(I));
      if Assigned(FStatus) and ((I + 1) mod 8 = 0) then
        FStatus(Format('Diffusion block %d/32', [I + 1]));
    end;
    // Final scale is from the sampled timestep, never the text's zero row.
    FNormOut.forward(OutScale, T1, 1);
    Q21ModNorm(N, X + NativeInt(PrefixRows) * HIDDEN, OutScale, ImageRows, HIDDEN, 0, 0);
    Result := TMemoryBlock.Create([ImageRows, 64], 'qwen21-noise-prediction');
    FOut.forward(Result, N, ImageRows);
    QNNCheckFinite(Result, ImageRows * 64, 'Noise prediction');
  finally
    X.Free;
    N.Free;
    Q.Free;
    K.Free;
    V.Free;
    A.Free;
    Y.Free;
    G.Free;
    Active.Free;
    T.Free;
    T1.Free;
    TEmb.Free;
    Modulation.Free;
    OutScale.Free
  end;
end;
{$I QuickNN.Models.Qwen21.Transformer.GPU.inc}

end.

