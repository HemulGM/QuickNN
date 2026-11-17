unit QuickNN.Models.Flux1.Text;

{ SPDX-License-Identifier: Apache-2.0
  Native CLIP/T5 encoder execution, following Hugging Face Transformers'
  Apache-2.0 modeling_clip.py and modeling_t5.py. No external runtime. }
{$POINTERMATH ON}

interface

uses
  System.SysUtils, System.Generics.Collections, QuickNN.Common, QuickNN.SafeTensors,
  QuickNN.Quantization, QuickNN.Vulkan;

type
  TFlux1TextEncoder = class
  private
    FFiles: TSafeTensorFiles;
    FLinear: TDictionary<string, PQNNLinearWeight>;
    FNorms: TDictionary<string, TMemoryBlock>;
    FEmbedding, FPosition: TMemoryBlock;
    FHidden, FHeads, FHeadDim, FMlp, FLayers, FVocab, FBuckets, FDistance: Integer;
    FEpsilon: Single;
    FT5: Boolean;
    function Weight(const Name: string): TMemoryBlock;
    procedure AddLinear(const Name: string; const Input, Output: Integer);
  public
    constructor Create(const Directory: string; const IsT5: Boolean);
    destructor Destroy; override;
    function EncodeGPU(const Ctx: TQNNGPUContext; const Ids: TArray<Integer>;
      const PooledIndex: Integer = -1): TQNNGPUTensor;
    property HiddenSize: Integer read FHidden;
  end;

implementation

uses
  System.JSON, System.IOUtils, QuickNN.Vulkan.Operations;

procedure TFlux1TextEncoder.AddLinear(const Name: string; const Input, Output: Integer);
var
  Layer: PQNNLinearWeight;
begin
  New(Layer);
  try
    Layer.Load(FFiles, Name);
    if (Layer.InputDim <> Input) or (Layer.OutputDim <> Output) then
      raise ESafeTensorError.Create('FLUX.1 text projection shape: ' + Name);
    FLinear.Add(Name, Layer);
  except Layer.Free; Dispose(Layer); raise end;
end;

function TFlux1TextEncoder.Weight(const Name: string): TMemoryBlock;
begin
  if not FNorms.TryGetValue(Name, Result) then
  begin
    var Tensor := QNNRequiredTensor(FFiles, Name);
    Result := Tensor.AsSingle(True);
    FNorms.Add(Name, Result);
  end;
end;

constructor TFlux1TextEncoder.Create(const Directory: string; const IsT5: Boolean);
var
  Config: TJSONValue;
  Prefix: string;
begin
  inherited Create;
  FT5 := IsT5;
  FLinear := TDictionary<string, PQNNLinearWeight>.Create;
  FNorms := TDictionary<string, TMemoryBlock>.Create;
  Config := TJSONObject.ParseJSONValue(TFile.ReadAllText(TPath.Combine(Directory, 'config.json'), TEncoding.UTF8));
  try
    if not (Config is TJSONObject) then raise EArgumentException.Create('Invalid FLUX.1 text configuration');
    if FT5 then
    begin
      if (Config.GetValue<string>('model_type', '') <> 't5')
        or (Config.GetValue<string>('feed_forward_proj', '') <> 'gated-gelu') then
        raise EArgumentException.Create('Expected T5 v1.1 gated-GELU encoder');
      FHidden := Config.GetValue<Integer>('d_model');
      FMlp := Config.GetValue<Integer>('d_ff');
      FHeads := Config.GetValue<Integer>('num_heads');
      FHeadDim := Config.GetValue<Integer>('d_kv');
      FLayers := Config.GetValue<Integer>('num_layers');
      FEpsilon := Config.GetValue<Single>('layer_norm_epsilon', 1E-6);
      FBuckets := Config.GetValue<Integer>('relative_attention_num_buckets', 32);
      FDistance := Config.GetValue<Integer>('relative_attention_max_distance', 128);
    end
    else
    begin
      FHidden := Config.GetValue<Integer>('hidden_size');
      FMlp := Config.GetValue<Integer>('intermediate_size');
      FHeads := Config.GetValue<Integer>('num_attention_heads');
      if FHeads < 1 then raise EArgumentException.Create('Invalid CLIP head count');
      FHeadDim := FHidden div FHeads;
      FLayers := Config.GetValue<Integer>('num_hidden_layers');
      FEpsilon := Config.GetValue<Single>('layer_norm_eps', 1E-5);
      if Config.GetValue<string>('hidden_act', 'quick_gelu') <> 'quick_gelu' then
        raise EArgumentException.Create('Expected CLIP QuickGELU activation');
    end;
    FVocab := Config.GetValue<Integer>('vocab_size');
    if (FHidden < 1) or (FHidden > 8192) or (FMlp < 1) or (FMlp > 65536)
      or (FLayers < 1) or (FLayers > 96) or (FHeads < 1) or (FHeads > 128)
      or (FHeadDim < 1) or (FHeads * FHeadDim <> FHidden) or (FEpsilon <= 0)
      or (FT5 and ((FBuckets < 4) or (FBuckets mod 4 <> 0) or (FDistance <= FBuckets div 4))) then
      raise EArgumentException.Create('Invalid FLUX.1 text encoder dimensions');
  finally Config.Free end;
  FFiles := LoadSafeTensorShards(Directory, 'model.safetensors.index.json', 'model.safetensors');
  if FT5 then
  begin
    FEmbedding := FFiles.GetLinearWeight('shared.weight', True);
    if not FEmbedding.IsAssigned then FEmbedding := FFiles.GetLinearWeight('encoder.embed_tokens.weight', True);
    for var I := 0 to FLayers - 1 do
    begin
      Prefix := Format('encoder.block.%d.', [I]);
      for var Name in TArray<string>.Create('q','k','v','o') do
        AddLinear(Prefix + 'layer.0.SelfAttention.' + Name, FHidden, FHidden);
      AddLinear(Prefix + 'layer.1.DenseReluDense.wi_0', FHidden, FMlp);
      AddLinear(Prefix + 'layer.1.DenseReluDense.wi_1', FHidden, FMlp);
      AddLinear(Prefix + 'layer.1.DenseReluDense.wo', FMlp, FHidden);
      Weight(Prefix + 'layer.0.layer_norm.weight');
      Weight(Prefix + 'layer.1.layer_norm.weight');
    end;
    FPosition := Weight('encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight');
    if FPosition.Count <> FBuckets * FHeads then raise EArgumentException.Create('T5 position bias dimensions');
    Weight('encoder.final_layer_norm.weight');
  end
  else
  begin
    FEmbedding := FFiles.GetLinearWeight('text_model.embeddings.token_embedding.weight', True);
    FPosition := FFiles.GetLinearWeight('text_model.embeddings.position_embedding.weight', True);
    if FPosition.Count < FHidden * 77 then raise EArgumentException.Create('CLIP position embedding dimensions');
    for var I := 0 to FLayers - 1 do
    begin
      Prefix := Format('text_model.encoder.layers.%d.', [I]);
      for var Name in TArray<string>.Create('q_proj','k_proj','v_proj','out_proj') do
        AddLinear(Prefix + 'self_attn.' + Name, FHidden, FHidden);
      AddLinear(Prefix + 'mlp.fc1', FHidden, FMlp);
      AddLinear(Prefix + 'mlp.fc2', FMlp, FHidden);
      for var Norm in TArray<string>.Create('layer_norm1','layer_norm2') do
      begin Weight(Prefix + Norm + '.weight'); Weight(Prefix + Norm + '.bias') end;
    end;
    Weight('text_model.final_layer_norm.weight'); Weight('text_model.final_layer_norm.bias');
  end;
  if FEmbedding.Count <> Int64(FVocab) * FHidden then raise EArgumentException.Create('FLUX.1 text embedding dimensions');
end;

destructor TFlux1TextEncoder.Destroy;
begin
  QNNGPUClearWeightCache;
  if FLinear <> nil then for var Pair in FLinear do
  begin Pair.Value.Free; Dispose(Pair.Value) end;
  FLinear.Free;
  if FNorms <> nil then for var Pair in FNorms do Pair.Value.Free;
  FNorms.Free; FEmbedding.Free; FPosition.Free;
  for var FileData in FFiles do FileData.Close;
  inherited;
end;

function TFlux1TextEncoder.EncodeGPU(const Ctx: TQNNGPUContext; const Ids: TArray<Integer>;
  const PooledIndex: Integer): TQNNGPUTensor;
var
  Work, Local: TGPUWorkspace;
  X, IDsGPU, PositionGPU, BiasGPU, N, Q, K, V, A, Projected, Gate, Up: TQNNGPUTensor;
  Rows: Integer;

  function Linear(const Src: TQNNGPUTensor; const Name: string): TQNNGPUTensor;
  begin
    var Layer := FLinear[Name];
    Result := Local.New(Int64(Rows) * Layer.OutputDim);
    Layer.ForwardGPU(Ctx, Result, Src, Rows);
  end;

  function Norm(const Src: TQNNGPUTensor; const Name: string): TQNNGPUTensor;
  begin
    var Bias := Default(TMemoryBlock);
    if not FT5 then Bias := Weight(Name + '.bias');
    Result := Local.Norm(Src, Weight(Name + '.weight'), Bias, Rows, FHidden, not FT5, FEpsilon);
  end;

begin
  Rows := Length(Ids);
  if (Rows < 1) or (Rows > 1024) or (not FT5 and (Rows > 77))
    or (PooledIndex < -1) or (PooledIndex >= Rows) then
    raise EArgumentException.Create('FLUX.1 text token count/pooling index');
  for var Id in Ids do if (Id < 0) or (Id >= FVocab) then raise EArgumentException.Create('Invalid FLUX.1 token ID');
  Work := TGPUWorkspace.Create(Ctx);
  BiasGPU := nil;
  try
    IDsGPU := Work.Own(Ctx.UploadRaw(@Ids[0], Rows));
    X := Work.New(Int64(Rows) * FHidden);
    Ctx.Gather(X, IDsGPU, FEmbedding, Default(TMemoryBlock), Rows, FHidden, FVocab, 0);
    Work.Release(IDsGPU);
    if FT5 then
    begin
      BiasGPU := Work.New(Int64(FHeads) * Rows * Rows);
      Ctx.KernelWeights(GO_T5_POSITION_BIAS, BiasGPU.Count, BiasGPU, nil,
        FPosition, Default(TMemoryBlock), [Rows, FHeads, FBuckets, FDistance]);
    end
    else
    begin
      var Positions: TArray<Integer>;
      SetLength(Positions, Rows);
      for var I := 0 to Rows - 1 do Positions[I] := I;
      IDsGPU := Work.Own(Ctx.UploadRaw(@Positions[0], Rows));
      PositionGPU := Work.New(Int64(Rows) * FHidden);
      Ctx.Gather(PositionGPU, IDsGPU, FPosition, Default(TMemoryBlock), Rows, FHidden, 77, 0);
      Work.Add(X, PositionGPU, X.Count); Work.Release(PositionGPU); Work.Release(IDsGPU);
    end;
    for var I := 0 to FLayers - 1 do
    begin
      Local := TGPUWorkspace.Create(Ctx);
      try
        var Prefix := Format('text_model.encoder.layers.%d.', [I]);
        var Attn := Prefix + 'self_attn.';
        var Norm1 := Prefix + 'layer_norm1';
        var Norm2 := Prefix + 'layer_norm2';
        if FT5 then
        begin
          Prefix := Format('encoder.block.%d.', [I]);
          Attn := Prefix + 'layer.0.SelfAttention.';
          Norm1 := Prefix + 'layer.0.layer_norm'; Norm2 := Prefix + 'layer.1.layer_norm';
        end;
        N := Norm(X, Norm1);
        if FT5 then begin Q := Linear(N, Attn + 'q'); K := Linear(N, Attn + 'k'); V := Linear(N, Attn + 'v') end
        else begin Q := Linear(N, Attn + 'q_proj'); K := Linear(N, Attn + 'k_proj'); V := Linear(N, Attn + 'v_proj') end;
        Local.Release(N);
        A := Local.New(Int64(Rows) * FHidden);
        var Causal := Rows;
        if FT5 then Causal := 0;
        Ctx.Attention(A, Q, K, V, Rows, FHeads, FHeads, FHeadDim, Causal, nil, BiasGPU);
        Local.Release(Q); Local.Release(K); Local.Release(V);
        if FT5 then Projected := Linear(A, Attn + 'o') else Projected := Linear(A, Attn + 'out_proj');
        Work.Add(X, Projected, X.Count); Local.Release(A); Local.Release(Projected);
        N := Norm(X, Norm2);
        if FT5 then
        begin
          Gate := Linear(N, Prefix + 'layer.1.DenseReluDense.wi_0');
          Up := Linear(N, Prefix + 'layer.1.DenseReluDense.wi_1');
          Ctx.Kernel(GO_UNARY, Gate.Count, Gate, Gate, nil, nil, [1]);
          Ctx.Kernel(GO_AFFINE, Gate.Count, Gate, Gate, Up, nil, [Gate.Count, 4, 0, 0, 0, 0]);
          Projected := Linear(Gate, Prefix + 'layer.1.DenseReluDense.wo');
        end
        else
        begin
          Gate := Linear(N, Prefix + 'mlp.fc1');
          Ctx.Kernel(GO_UNARY, Gate.Count, Gate, Gate, nil, nil, [3]);
          Projected := Linear(Gate, Prefix + 'mlp.fc2');
        end;
        Work.Add(X, Projected, X.Count);
      finally Local.Free end;
      if Assigned(TextProgressCallback) then TextProgressCallback(I + 1, FLayers);
    end;
    Local := TGPUWorkspace.Create(Ctx);
    try
      if FT5 then N := Norm(X, 'encoder.final_layer_norm') else N := Norm(X, 'text_model.final_layer_norm');
      if PooledIndex >= 0 then Result := Work.Copy(N, FHidden, PooledIndex * FHidden)
      else Result := Work.Copy(N, N.Count);
      Ctx.CheckFinite(Result); Work.Keep(Result);
    finally Local.Free end;
  finally Work.Free end;
end;

end.
