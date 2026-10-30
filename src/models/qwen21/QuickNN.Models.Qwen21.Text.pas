unit QuickNN.Models.Qwen21.Text;

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
  QuickNN.Vulkan, QuickNN.Models.Qwen21.Math, QuickNN.Models.Qwen21.Conditioning,
  QuickNN.Models.Qwen21.Tokenizer;

type
  TQwen21TextLayer = record
    InputNorm, PostNorm, QNorm, KNorm: TMemoryBlock;
    Q, K, V, O, Gate, Up, Down: TQNNLinearWeight;
    procedure Load(const Files: TSafeTensorFiles; const Index: Integer);
    procedure Free;
  end;

  TQwen21TextEncoder = class
  private
    FFiles: TSafeTensorFiles;
    FEmbedding: TQNNLinearWeight;
    FLayers: TArray<TQwen21TextLayer>;
    FStatus: TQwen21Status;
    FHidden, FIntermediate, FHeads, FKVHeads, FHeadDim: Integer;
    FTheta: Double;
    function EncodeTokens(const Ids: TArray<Integer>; const DropTokens: Integer; const Images: TQwen21ImageConditions; const Spans: TQwen21TextImageSpans; const PositionIds: TArray<Integer>; out ImageSlots: TArray<Integer>): TMemoryBlock;
  public
    constructor Create(const Checkpoint, ConfigPath: string; const Status: TQwen21Status = nil);
    destructor Destroy; override;
    function EncodePromptGPU(const Ctx: TQNNGPUContext; const Tokenizer: TQwen21Tokenizer; const Prompt: string; const Images: TQwen21ImageConditions; const resident: TQwen21GPUImageConditions; out ImageSlots: TArray<Integer>): TQNNGPUTensor;
    function Encode(const Ids: TArray<Integer>; const DropTokens: Integer): TMemoryBlock;
    function EncodePrompt(const Tokenizer: TQwen21Tokenizer; const Prompt: string; const Images: TQwen21ImageConditions; out ImageSlots: TArray<Integer>): TMemoryBlock;
  end;

implementation

uses
  System.JSON, System.IOUtils, QuickNN.Kernels;

procedure TQwen21TextLayer.Load(const Files: TSafeTensorFiles; const Index: Integer);
var
  P: string;
begin
  P := Format('model.layers.%d.', [Index]);
  InputNorm := QNNRequiredTensor(Files, P + 'input_layernorm.weight').AsSingle(True);
  PostNorm := QNNRequiredTensor(Files, P + 'post_attention_layernorm.weight').AsSingle(True);
  QNorm := QNNRequiredTensor(Files, P + 'self_attn.q_norm.weight').AsSingle(True);
  KNorm := QNNRequiredTensor(Files, P + 'self_attn.k_norm.weight').AsSingle(True);
  Q.Load(Files, P + 'self_attn.q_proj');
  K.Load(Files, P + 'self_attn.k_proj');
  V.Load(Files, P + 'self_attn.v_proj');
  O.Load(Files, P + 'self_attn.o_proj');
  Gate.Load(Files, P + 'mlp.gate_proj');
  Up.Load(Files, P + 'mlp.up_proj');
  Down.Load(Files, P + 'mlp.down_proj');
  if (InputNorm.Count <> 4096) or (PostNorm.Count <> 4096) or (QNorm.Count <> 128) or (KNorm.Count <> 128) then
    raise ESafeTensorError.Create('Invalid Qwen3-VL norm shape: ' + P);
end;

procedure TQwen21TextLayer.Free;
begin
  InputNorm.Free;
  PostNorm.Free;
  QNorm.Free;
  KNorm.Free;
  Q.Free;
  K.Free;
  V.Free;
  O.Free;
  Gate.Free;
  Up.Free;
  Down.Free;
end;

constructor TQwen21TextEncoder.Create(const Checkpoint, ConfigPath: string; const Status: TQwen21Status);
var
  Root, JSON: TJSONValue;
  Layers: Integer;
begin
  inherited Create;
  FStatus := Status;
  Root := TJSONObject.ParseJSONValue(TFile.ReadAllText(ConfigPath));
  if Root = nil then
    raise EArgumentException.Create('Invalid Qwen3-VL configuration');
  try
    JSON := Root.FindValue('text_config');
    if JSON = nil then
      raise EArgumentException.Create('Qwen3-VL text_config missing');
    FHidden := JSON.GetValue<Integer>('hidden_size');
    FIntermediate := JSON.GetValue<Integer>('intermediate_size');
    FHeads := JSON.GetValue<Integer>('num_attention_heads');
    FKVHeads := JSON.GetValue<Integer>('num_key_value_heads');
    FHeadDim := JSON.GetValue<Integer>('head_dim');
    Layers := JSON.GetValue<Integer>('num_hidden_layers');
    FTheta := JSON.GetValue<Double>('rope_theta');
    if (FHidden <> 4096)
      or (FIntermediate <> 12288)
      or (FHeads <> 32)
      or (FKVHeads <> 8)
      or (FHeadDim <> 128)
      or (Layers <> 36) then
      raise EArgumentException.Create('Expected Qwen3-VL-8B text architecture');
  finally
    Root.Free
  end;
  SetLength(FFiles, 1);
  FFiles[0] := TSafeTensorFile.Open(Checkpoint);
  FEmbedding.Load(FFiles, 'model.embed_tokens');
  if (FEmbedding.InputDim <> FHidden) or (FEmbedding.OutputDim <> 151936) then
    raise EArgumentException.Create('Wrong embedding shape');
  SetLength(FLayers, Layers);
  for var I := 0 to High(FLayers) do
  begin
    FLayers[I].Load(FFiles, I);
    if (FLayers[I].Q.InputDim <> FHidden)
      or (FLayers[I].Q.OutputDim <> FHeads * FHeadDim)
      or (FLayers[I].K.OutputDim <> FKVHeads * FHeadDim)
      or (FLayers[I].V.OutputDim <> FKVHeads * FHeadDim)
      or (FLayers[I].O.OutputDim <> FHidden)
      or (FLayers[I].Gate.OutputDim <> FIntermediate)
      or (FLayers[I].Up.OutputDim <> FIntermediate)
      or (FLayers[I].Down.OutputDim <> FHidden) then
      raise EArgumentException.Create('Wrong Qwen3-VL layer shape');
  end;
end;

destructor TQwen21TextEncoder.Destroy;
begin
  QNNGPUClearWeightCache;
  for var I := 0 to High(FLayers) do
    FLayers[I].Free;
  FEmbedding.Free;
  for var I := 0 to High(FFiles) do
    FFiles[I].Close;
  inherited;
end;

function TQwen21TextEncoder.EncodeTokens(const Ids: TArray<Integer>; const DropTokens: Integer; const Images: TQwen21ImageConditions; const Spans: TQwen21TextImageSpans; const PositionIds: TArray<Integer>; out ImageSlots: TArray<Integer>): TMemoryBlock;
var
  X, N, Q, K, V, A, Y, G, U: TMemoryBlock;
  Rows, OutRows, Slot: Integer;
  Layer: ^TQwen21TextLayer;
  Keep: TArray<Boolean>;
begin
  Rows := Length(Ids);
  if (Rows <= DropTokens) or (DropTokens < 0) or ((Rows > 2048) and (Length(Images) = 0)) or (Rows > 6144) then
    raise EArgumentException.Create('Invalid text sequence length');
  try
    X := TMemoryBlock.Create([Rows, FHidden], 'qwen3vl-hidden');
    N := TMemoryBlock.Create([Rows, FHidden], 'qwen3vl-norm');
    Q := TMemoryBlock.Create([Rows, FHeads * FHeadDim], 'qwen3vl-q');
    K := TMemoryBlock.Create([Rows, FKVHeads * FHeadDim], 'qwen3vl-k');
    V := TMemoryBlock.Create([Rows, FKVHeads * FHeadDim], 'qwen3vl-v');
    A := TMemoryBlock.Create([Rows, FHeads * FHeadDim], 'qwen3vl-attention');
    Y := TMemoryBlock.Create([Rows, FHidden], 'qwen3vl-output');
    FEmbedding.Gather(X, Ids);
    for var Span in Spans do
    begin
      if (Span.ImageIndex < 0)
        or (Span.ImageIndex >= Length(Images))
        or (Span.TokenStart < DropTokens)
        or (Span.TokenStart + Span.TokenCount > Rows)
        or (Images[Span.ImageIndex].Visual.Count <> Int64(Span.TokenCount) * FHidden) then
        raise EArgumentException.Create('Invalid visual token span');
      Move(
        System.PSingle(Images[Span.ImageIndex].Visual)^,
        System.PSingle(X)[NativeInt(Span.TokenStart) * FHidden],
        NativeInt(Span.TokenCount) * FHidden * 4
      );
    end;
    for var I := 0 to High(FLayers) do
    begin
      Layer := @FLayers[I];
      Q21RMS(N, X, Layer.InputNorm, Rows, FHidden);
      Layer.Q.forward(Q, N, Rows);
      Layer.K.forward(K, N, Rows);
      Layer.V.forward(V, N, Rows);
      Q21HeadNorm(Q, Layer.QNorm, Rows, FHeads, FHeadDim);
      Q21HeadNorm(K, Layer.KNorm, Rows, FKVHeads, FHeadDim);
      Q21TextRoPE(Q, Rows, FHeads, FHeadDim, FTheta, PositionIds);
      Q21TextRoPE(K, Rows, FKVHeads, FHeadDim, FTheta, PositionIds);
      Q21Attention(A, Q, K, V, Rows, FHeads, FKVHeads, FHeadDim, Rows);
      Layer.O.forward(Y, A, Rows);
      QNNAddInplace(X, Y, Rows * FHidden);
      Q21RMS(N, X, Layer.PostNorm, Rows, FHidden);
      if (Layer.Gate.Weight.DataType = DT_I8)
        or (Layer.Up.Weight.DataType = DT_I8)
        or (Layer.Down.Weight.DataType = DT_I8)
        or not TryVulkanSwiGLU(
        Y,
        N,
        Layer.Gate.Weight,
        Layer.Up.Weight,
        Layer.Down.Weight,
        Rows,
        FHidden,
        FIntermediate) then
      begin
        if not G.IsAssigned then
        begin
          G := TMemoryBlock.Create([Rows, FIntermediate], 'qwen3vl-gate');
          U := TMemoryBlock.Create([Rows, FIntermediate], 'qwen3vl-up');
        end;
        Layer.Gate.forward(G, N, Rows);
        Layer.Up.forward(U, N, Rows);
        Q21Silu(G, Rows * FIntermediate);
        for var J := 0 to Rows * FIntermediate - 1 do
          System.PSingle(G)[J] := System.PSingle(G)[J] * System.PSingle(U)[J];
        Layer.Down.forward(Y, G, Rows);
      end;
      QNNAddInplace(X, Y, Rows * FHidden);
      if I < 3 then
        for var Span in Spans do
        begin
          if Images[Span.ImageIndex].DeepStack[I].Count <> Int64(Span.TokenCount) * FHidden then
            raise EArgumentException.Create('Invalid DeepStack span');
          QNNAddInplace(
            X + NativeInt(Span.TokenStart) * FHidden,
            Images[Span.ImageIndex].DeepStack[I],
            Span.TokenCount * FHidden
          );
        end;
      QNNCheckFinite(X, Rows * FHidden, 'Qwen3-VL layer ' + IntToStr(I));
      if Assigned(FStatus) then
        FStatus(Format('Text encoder %d/%d', [I + 1, Length(FLayers)]));
    end;
    // Qwen-Image 2.1 consumes the final decoder block BEFORE final RMSNorm,
    // then removes only the system turn. User/assistant role tokens remain.
    SetLength(Keep, Rows);
    for var Row := DropTokens to Rows - 1 do
      Keep[Row] := True;
    for var Span in Spans do
      for var Row := Span.TokenStart to Span.TokenStart + Span.TokenCount - 1 do
        Keep[Row] := False;
    OutRows := 0;
    for var Selected in Keep do
      if Selected then
        Inc(OutRows);
    if OutRows < 1 then
      raise EArgumentException.Create('Empty image/text conditioning');
    SetLength(ImageSlots, Length(Images));
    for var Span in Spans do
    begin
      Slot := 0;
      for var Row := DropTokens to Span.TokenStart - 1 do
        if Keep[Row] then
          Inc(Slot);
      ImageSlots[Span.ImageIndex] := Slot;
    end;
    Result := TMemoryBlock.Create([OutRows, FHidden], 'qwen-image21-text-conditioning');
    OutRows := 0;
    for var Row := DropTokens to Rows - 1 do
      if Keep[Row] then
      begin
        Move(
          System.PSingle(X)[NativeInt(Row) * FHidden],
          System.PSingle(Result)[NativeInt(OutRows) * FHidden],
          FHidden * 4
        );
        Inc(OutRows);
      end;
  finally
    X.Free;
    N.Free;
    Q.Free;
    K.Free;
    V.Free;
    A.Free;
    Y.Free;
    G.Free;
    U.Free
  end;
end;

function TQwen21TextEncoder.Encode(const Ids: TArray<Integer>; const DropTokens: Integer): TMemoryBlock;
var
  Slots: TArray<Integer>;
begin
  Result := EncodeTokens(Ids, DropTokens, nil, nil, nil, Slots)
end;

function TQwen21TextEncoder.EncodePrompt(const Tokenizer: TQwen21Tokenizer; const Prompt: string; const Images: TQwen21ImageConditions; out ImageSlots: TArray<Integer>): TMemoryBlock;
var
  Ids, PositionIds: TArray<Integer>;
  Spans: TQwen21TextImageSpans;
  Drop: Integer;
begin
  Q21BuildImagePrompt(Tokenizer, Prompt, Images, Ids, PositionIds, Spans, Drop);
  Result := EncodeTokens(Ids, Drop, Images, Spans, PositionIds, ImageSlots);
end;
{$I QuickNN.Models.Qwen21.Text.GPU.inc}

end.

