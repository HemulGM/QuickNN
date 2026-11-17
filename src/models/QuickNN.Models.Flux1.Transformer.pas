unit QuickNN.Models.Flux1.Transformer;

{ SPDX-License-Identifier: Apache-2.0
  FLUX.1 architecture follows Black Forest Labs' Apache-2.0 source:
  https://github.com/black-forest-labs/flux/tree/main/src/flux
  Model weights have their own license. No external inference runtime is used. }
{$POINTERMATH ON}

interface

uses
  System.SysUtils, System.Generics.Collections, QuickNN.Common,
  QuickNN.SafeTensors, QuickNN.Quantization, QuickNN.LoRA, QuickNN.Vulkan;

type
  TFlux1Transformer = class
  private
    FFiles: TSafeTensorFiles;
    FLinear: TDictionary<string, PQNNLinearWeight>;
    FTargets: TDictionary<string, PQNNLinearWeight>;
    FVectors: TDictionary<string, TMemoryBlock>;
    FAdapters: TQNNLoRASet;
    FHidden, FHeads, FHeadDim, FMlp, FDoubleCount, FSingleCount: Integer;
    FInput, FTextDim, FPoolDim, FAxis0, FAxis1: Integer;
    FTheta: Single;
    procedure LoadLinear(const Name: string; const Input, Output: Integer);
    procedure LoadVector(const Name: string);
    function Resolve(const Target: string; out Offset: Integer): PQNNLinearWeight;
  public
    constructor Create(const Checkpoint, ConfigPath: string; const Adapters: TArray<TQNNLoRASpec>);
    destructor Destroy; override;
    function ForwardGPU(const Ctx: TQNNGPUContext; const Latent, Text, Pooled: TQNNGPUTensor;
      const Height, Width, TextRows: Integer; const Time, Guidance: Single): TQNNGPUTensor;
    function AdapterCount: Integer;
    property HiddenSize: Integer read FHidden;
  end;

implementation

uses
  System.Math, System.JSON, System.IOUtils, QuickNN.Vulkan.Operations;

procedure TFlux1Transformer.LoadLinear(const Name: string; const Input, Output: Integer);
var
  Layer: PQNNLinearWeight;
begin
  New(Layer);
  try
    Layer.Load(FFiles, Name);
    if (Layer.InputDim <> Input) or (Layer.OutputDim <> Output) then
      raise ESafeTensorError.Create('FLUX.1 matrix shape mismatch: ' + Name);
    FLinear.Add(Name, Layer);
  except
    Layer.Free; Dispose(Layer); raise;
  end;
  FTargets.Add(Name, Layer);
  FTargets.Add('lora_unet_' + Name.Replace('.', '_'), Layer);
end;

procedure TFlux1Transformer.LoadVector(const Name: string);
var
  Tensor: PSafeTensor;
begin
  Tensor := QNNRequiredTensor(FFiles, Name);
  if Tensor.Count <> FHeadDim then
    raise ESafeTensorError.Create('FLUX.1 QK norm shape mismatch: ' + Name);
  FVectors.Add(Name, Tensor.AsSingle(True));
end;

function TFlux1Transformer.Resolve(const Target: string; out Offset: Integer): PQNNLinearWeight;
begin
  Offset := 0;
  if not FTargets.TryGetValue(Target, Result) then Result := nil;
end;

constructor TFlux1Transformer.Create(const Checkpoint, ConfigPath: string;
  const Adapters: TArray<TQNNLoRASpec>);
var
  Config: TJSONValue;
  Axes: TJSONArray;
  Prefix: string;
begin
  inherited Create;
  FLinear := TDictionary<string, PQNNLinearWeight>.Create;
  FTargets := TDictionary<string, PQNNLinearWeight>.Create;
  FVectors := TDictionary<string, TMemoryBlock>.Create;
  Config := TJSONObject.ParseJSONValue(TFile.ReadAllText(ConfigPath, TEncoding.UTF8));
  try
    if not (Config is TJSONObject) or (Config.GetValue<string>('_class_name', '') <> 'FluxTransformer2DModel')
      or not Config.GetValue<Boolean>('guidance_embeds', False) then
      raise EArgumentException.Create('Expected FLUX.1-dev transformer configuration');
    FHeads := Config.GetValue<Integer>('num_attention_heads', 24);
    FHeadDim := Config.GetValue<Integer>('attention_head_dim', 128);
    FHidden := FHeads * FHeadDim;
    FMlp := FHidden * 4;
    FDoubleCount := Config.GetValue<Integer>('num_layers', 19);
    FSingleCount := Config.GetValue<Integer>('num_single_layers', 38);
    FInput := Config.GetValue<Integer>('in_channels', 64);
    FTextDim := Config.GetValue<Integer>('joint_attention_dim', 4096);
    FPoolDim := Config.GetValue<Integer>('pooled_projection_dim', 768);
    FTheta := 10000;
    Axes := Config.FindValue('axes_dims_rope') as TJSONArray;
    var OwnAxes := Axes = nil;
    if OwnAxes then
    begin
      Axes := TJSONArray.Create;
      Axes.Add(16); Axes.Add(56); Axes.Add(56);
    end;
    try
    if Axes.Count <> 3 then raise EArgumentException.Create('FLUX.1 requires three RoPE axes');
    FAxis0 := Axes.Items[0].GetValue<Integer>;
    FAxis1 := Axes.Items[1].GetValue<Integer>;
    if (FHeads < 1) or (FHidden > 8192) or (FInput <> 64)
      or (FDoubleCount < 1) or (FDoubleCount > 64) or (FSingleCount < 1) or (FSingleCount > 128)
      or (FAxis0 < 2) or (FAxis1 < 2) or (FAxis0 mod 2 <> 0) or (FAxis1 mod 2 <> 0)
      or (Axes.Items[2].GetValue<Integer> < 2)
      or (FAxis0 + FAxis1 + Axes.Items[2].GetValue<Integer> <> FHeadDim) then
      raise EArgumentException.Create('Invalid FLUX.1 transformer dimensions');
    finally if OwnAxes then Axes.Free end;
  finally Config.Free end;
  SetLength(FFiles, 1);
  FFiles[0] := TSafeTensorFile.Open(Checkpoint);
  LoadLinear('img_in', FInput, FHidden);
  LoadLinear('txt_in', FTextDim, FHidden);
  for var Name in TArray<string>.Create('time_in', 'guidance_in', 'vector_in') do
  begin
    var Dim := 256;
    if Name = 'vector_in' then Dim := FPoolDim;
    LoadLinear(Name + '.in_layer', Dim, FHidden);
    LoadLinear(Name + '.out_layer', FHidden, FHidden);
  end;
  for var I := 0 to FDoubleCount - 1 do
    for var Stream in TArray<string>.Create('img', 'txt') do
    begin
      Prefix := Format('double_blocks.%d.%s', [I, Stream]);
      LoadLinear(Prefix + '_mod.lin', FHidden, FHidden * 6);
      LoadLinear(Prefix + '_attn.qkv', FHidden, FHidden * 3);
      LoadLinear(Prefix + '_attn.proj', FHidden, FHidden);
      LoadLinear(Prefix + '_mlp.0', FHidden, FMlp);
      LoadLinear(Prefix + '_mlp.2', FMlp, FHidden);
      LoadVector(Prefix + '_attn.norm.query_norm.scale');
      LoadVector(Prefix + '_attn.norm.key_norm.scale');
    end;
  for var I := 0 to FSingleCount - 1 do
  begin
    Prefix := Format('single_blocks.%d.', [I]);
    LoadLinear(Prefix + 'modulation.lin', FHidden, FHidden * 3);
    LoadLinear(Prefix + 'linear1', FHidden, FHidden * 3 + FMlp);
    LoadLinear(Prefix + 'linear2', FHidden + FMlp, FHidden);
    LoadVector(Prefix + 'norm.query_norm.scale');
    LoadVector(Prefix + 'norm.key_norm.scale');
  end;
  LoadLinear('final_layer.adaLN_modulation.1', FHidden, FHidden * 2);
  LoadLinear('final_layer.linear', FHidden, FInput);
  FAdapters := TQNNLoRASet.Create(Adapters);
  FAdapters.Attach(
    function(const Target: string; out Offset: Integer): PQNNLinearWeight
    begin Result := Resolve(Target, Offset) end);
end;

destructor TFlux1Transformer.Destroy;
begin
  QNNGPUClearWeightCache;
  if FLinear <> nil then
    for var Pair in FLinear do
    begin Pair.Value.Free; Dispose(Pair.Value) end;
  FLinear.Free; FTargets.Free;
  if FVectors <> nil then
    for var Pair in FVectors do Pair.Value.Free;
  FVectors.Free;
  FAdapters.Free;
  for var FileData in FFiles do FileData.Close;
  inherited;
end;

function TFlux1Transformer.AdapterCount: Integer;
begin
  Result := FAdapters.EnabledModuleCount;
end;

function TFlux1Transformer.ForwardGPU(const Ctx: TQNNGPUContext;
  const Latent, Text, Pooled: TQNNGPUTensor; const Height, Width, TextRows: Integer;
  const Time, Guidance: Single): TQNNGPUTensor;
{$IFNDEF CPUX64}
begin
  raise ENotSupportedException.Create('FLUX.1-dev GPU execution requires Win64');
end;
{$ELSE}
var
  Work: TGPUWorkspace;
  ImageX, TextX, Vec, Part, Joined, Silu: TQNNGPUTensor;
  ImageRows, Rows: Integer;

  function Linear(const Owner: TGPUWorkspace; const Src: TQNNGPUTensor;
    const Name: string; const Count: Integer): TQNNGPUTensor;
  begin
    var Layer := FLinear[Name];
    Result := Owner.New(Int64(Count) * Layer.OutputDim);
    Layer.ForwardGPU(Ctx, Result, Src, Count);
  end;

  function Embed(const Src: TQNNGPUTensor; const Name: string): TQNNGPUTensor;
  var
    Tmp: TQNNGPUTensor;
  begin
    Tmp := Linear(Work, Src, Name + '.in_layer', 1);
    Ctx.Kernel(GO_UNARY, Tmp.Count, Tmp, Tmp, nil, nil, [0]);
    Result := Linear(Work, Tmp, Name + '.out_layer', 1);
    Work.Release(Tmp);
  end;

  procedure Rope(const Tensor: TQNNGPUTensor);
  begin
    Ctx.Kernel(GO_FLUX1_ROPE, Rows * FHidden div 2, Tensor, Tensor, nil, nil,
      [FHeads, FHeadDim, TextRows, Width, QNNGPUFloatBits(FTheta), FAxis0, FAxis1]);
  end;

  procedure DoubleBlock(const Index: Integer);
  var
    Local: TGPUWorkspace;
    IQKV, TQKV, IM, TM, Q, K, V, A, N, OutValue: TQNNGPUTensor;
    Prefix: string;

    function StreamQKV(const X: TQNNGPUTensor; const IsImage: Boolean;
      out Modulation: TQNNGPUTensor): TQNNGPUTensor;
    begin
      var Count := TextRows;
      var Stream := 'txt';
      if IsImage then begin Count := ImageRows; Stream := 'img' end;
      Modulation := Linear(Local, Silu, Prefix + Stream + '_mod.lin', 1);
      N := Local.Norm(X, Default(TMemoryBlock), Default(TMemoryBlock), Count, FHidden, True);
      Ctx.Kernel(GO_AFFINE, N.Count, N, N, Modulation, nil, [FHidden, 0, FHidden, 0, 0, 0]);
      Result := Linear(Local, N, Prefix + Stream + '_attn.qkv', Count);
      Local.Release(N);
    end;

    function QKVPart(const Offset: Integer): TQNNGPUTensor;
    var
      T, I: TQNNGPUTensor;
    begin
      T := Local.Slice(TQKV, TextRows, FHidden, FHidden * 3, Offset);
      I := Local.Slice(IQKV, ImageRows, FHidden, FHidden * 3, Offset);
      Result := Local.Concat(T, I, TextRows * FHidden, ImageRows * FHidden);
      Local.Release(T); Local.Release(I);
    end;

    procedure NormQK(const Tensor: TQNNGPUTensor; const Kind: string);
    var
      T, I, NT, NI: TQNNGPUTensor;
    begin
      T := Local.Copy(Tensor, TextRows * FHidden);
      I := Local.Copy(Tensor, ImageRows * FHidden, TextRows * FHidden);
      NT := Local.Norm(T, FVectors[Prefix + 'txt_attn.norm.' + Kind + '_norm.scale'], Default(TMemoryBlock), TextRows * FHeads, FHeadDim);
      NI := Local.Norm(I, FVectors[Prefix + 'img_attn.norm.' + Kind + '_norm.scale'], Default(TMemoryBlock), ImageRows * FHeads, FHeadDim);
      Ctx.Copy(Tensor, NT, NT.Count); Ctx.Copy(Tensor, NI, NI.Count, NT.Count);
      Local.Release(T); Local.Release(I); Local.Release(NT); Local.Release(NI);
    end;

    procedure FinishStream(const X, Modulation: TQNNGPUTensor; const IsImage: Boolean);
    begin
      var Count := TextRows;
      var Offset := 0;
      var Stream := 'txt';
      if IsImage then begin Count := ImageRows; Offset := TextRows * FHidden; Stream := 'img' end;
      N := Local.Copy(A, Count * FHidden, Offset);
      OutValue := Linear(Local, N, Prefix + Stream + '_attn.proj', Count);
      Local.Modulate(OutValue, Modulation, Count, FHidden, 1, FHidden * 2, 0, X);
      Local.Release(N); Local.Release(OutValue);
      N := Local.Norm(X, Default(TMemoryBlock), Default(TMemoryBlock), Count, FHidden, True);
      Ctx.Kernel(GO_AFFINE, N.Count, N, N, Modulation, nil, [FHidden, 0, FHidden * 4, FHidden * 3, 0, 0]);
      OutValue := Linear(Local, N, Prefix + Stream + '_mlp.0', Count);
      Local.Release(N);
      Ctx.Kernel(GO_UNARY, OutValue.Count, OutValue, OutValue, nil, nil, [1]);
      N := Linear(Local, OutValue, Prefix + Stream + '_mlp.2', Count);
      Local.Modulate(N, Modulation, Count, FHidden, 1, FHidden * 5, 0, X);
      Local.Release(N); Local.Release(OutValue);
    end;

  begin
    Local := TGPUWorkspace.Create(Ctx);
    try
      Prefix := Format('double_blocks.%d.', [Index]);
      IQKV := StreamQKV(ImageX, True, IM);
      TQKV := StreamQKV(TextX, False, TM);
      Q := QKVPart(0); K := QKVPart(FHidden); V := QKVPart(FHidden * 2);
      Local.Release(IQKV); Local.Release(TQKV);
      NormQK(Q, 'query'); NormQK(K, 'key'); Rope(Q); Rope(K);
      A := Local.New(Int64(Rows) * FHidden);
      Ctx.Attention(A, Q, K, V, Rows, FHeads, FHeads, FHeadDim, 0);
      Local.Release(Q); Local.Release(K); Local.Release(V);
      FinishStream(ImageX, IM, True); FinishStream(TextX, TM, False);
    finally Local.Free end;
  end;

  procedure SingleBlock(const Index: Integer);
  var
    Local: TGPUWorkspace;
    Modulation, N, Fused, Q, K, V, Mlp, A, Both, OutValue: TQNNGPUTensor;
    Prefix: string;
  begin
    Local := TGPUWorkspace.Create(Ctx);
    try
      Prefix := Format('single_blocks.%d.', [Index]);
      Modulation := Linear(Local, Silu, Prefix + 'modulation.lin', 1);
      N := Local.Norm(Joined, Default(TMemoryBlock), Default(TMemoryBlock), Rows, FHidden, True);
      Ctx.Kernel(GO_AFFINE, N.Count, N, N, Modulation, nil, [FHidden, 0, FHidden, 0, 0, 0]);
      Fused := Linear(Local, N, Prefix + 'linear1', Rows);
      Local.Release(N);
      Q := Local.Slice(Fused, Rows, FHidden, FHidden * 3 + FMlp, 0);
      K := Local.Slice(Fused, Rows, FHidden, FHidden * 3 + FMlp, FHidden);
      V := Local.Slice(Fused, Rows, FHidden, FHidden * 3 + FMlp, FHidden * 2);
      Mlp := Local.Slice(Fused, Rows, FMlp, FHidden * 3 + FMlp, FHidden * 3);
      Local.Release(Fused);
      N := Local.Norm(Q, FVectors[Prefix + 'norm.query_norm.scale'], Default(TMemoryBlock), Rows * FHeads, FHeadDim);
      Local.Release(Q); Q := N;
      N := Local.Norm(K, FVectors[Prefix + 'norm.key_norm.scale'], Default(TMemoryBlock), Rows * FHeads, FHeadDim);
      Local.Release(K); K := N;
      Rope(Q); Rope(K);
      A := Local.New(Int64(Rows) * FHidden);
      Ctx.Attention(A, Q, K, V, Rows, FHeads, FHeads, FHeadDim, 0);
      Ctx.Kernel(GO_UNARY, Mlp.Count, Mlp, Mlp, nil, nil, [1]);
      Both := Local.Join(A, Mlp, Rows, FHidden, FMlp);
      OutValue := Linear(Local, Both, Prefix + 'linear2', Rows);
      Local.Modulate(OutValue, Modulation, Rows, FHidden, 1, FHidden * 2, 0, Joined);
    finally Local.Free end;
  end;

begin
  ImageRows := Height * Width;
  Rows := ImageRows + TextRows;
  if (Height < 1) or (Width < 1) or (TextRows < 1) or (Rows > 6144)
    or (Int64(ImageRows) * FInput > Latent.Count) or (Int64(TextRows) * FTextDim > Text.Count)
    or (Pooled.Count < FPoolDim) or IsNan(Time) or IsNan(Guidance) then
    raise EArgumentException.Create('Invalid resident FLUX.1 inputs');
  Work := TGPUWorkspace.Create(Ctx);
  try
    Part := Work.CHWToRows(Latent, ImageRows, FInput);
    ImageX := Linear(Work, Part, 'img_in', ImageRows); Work.Release(Part);
    TextX := Linear(Work, Text, 'txt_in', TextRows);
    Part := Work.Sinusoidal(Time * 1000, 256); Vec := Embed(Part, 'time_in'); Work.Release(Part);
    Part := Work.Sinusoidal(Guidance * 1000, 256);
    var Guide := Embed(Part, 'guidance_in'); Work.Release(Part);
    Work.Add(Vec, Guide, FHidden); Work.Release(Guide);
    Part := Embed(Pooled, 'vector_in'); Work.Add(Vec, Part, FHidden); Work.Release(Part);
    Silu := Work.Unary(Vec, FHidden); Work.Release(Vec);
    for var I := 0 to FDoubleCount - 1 do
    begin
      DoubleBlock(I);
      if Assigned(SubstepCallback) then SubstepCallback(SUBSTEP_DOUBLE_BLOCK, I, FDoubleCount);
    end;
    Joined := Work.Concat(TextX, ImageX, TextRows * FHidden, ImageRows * FHidden);
    Work.Release(TextX); Work.Release(ImageX);
    for var I := 0 to FSingleCount - 1 do
    begin
      SingleBlock(I);
      if Assigned(SubstepCallback) then SubstepCallback(SUBSTEP_SINGLE_BLOCK, I, FSingleCount);
    end;
    ImageX := Work.Copy(Joined, ImageRows * FHidden, TextRows * FHidden); Work.Release(Joined);
    Vec := Linear(Work, Silu, 'final_layer.adaLN_modulation.1', 1);
    Part := Work.Norm(ImageX, Default(TMemoryBlock), Default(TMemoryBlock), ImageRows, FHidden, True);
    Ctx.Kernel(GO_AFFINE, Part.Count, Part, Part, Vec, nil, [FHidden, 0, FHidden, 0, 0, 0]);
    var OutRows := Linear(Work, Part, 'final_layer.linear', ImageRows);
    Result := Work.RowsToCHW(OutRows, ImageRows, FInput);
    Ctx.CheckFinite(Result); Work.Keep(Result);
  finally Work.Free end;
end;
{$ENDIF}

end.
