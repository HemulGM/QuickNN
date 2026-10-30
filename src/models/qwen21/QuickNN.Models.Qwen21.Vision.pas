unit QuickNN.Models.Qwen21.Vision;
{ Qwen3-VL-8B vision encoder and DeepStack. Tensor conventions follow
  Qwen / Hugging Face / ComfyUI reference implementations (Apache-2.0).
  See LICENSE.QwenImage21.txt. }
{$POINTERMATH ON}
interface
uses
  System.SysUtils,
  QuickNN.Common,
  QuickNN.Quantization,
  QuickNN.SafeTensors,
  QuickNN.Vulkan,
  QuickNN.Models.Qwen21.Math,
  QuickNN.Models.Qwen21.Conditioning;
type
  TQwen21VisionBlock = record
    Norm1, Norm1Bias, Norm2, Norm2Bias: TMemoryBlock;
    Qkv, Proj, FC1, FC2: TQNNLinearWeight;
    procedure Load(const Files: TSafeTensorFiles; const Index: Integer);
    procedure Free;
  end;
  TQwen21VisionMerger = record
    Norm, NormBias: TMemoryBlock;
    FC1, FC2: TQNNLinearWeight;
    PostShuffle: Boolean;
    procedure Load(const Files: TSafeTensorFiles; const Prefix: string; const Post: Boolean);
    function Forward(const Src: TMemoryBlock; const Rows: Integer): TMemoryBlock;
    function ForwardGPU(const Ctx: TQNNGPUContext; const Src: TQNNGPUTensor; const Rows: Integer): TQNNGPUTensor;
    procedure Free;
  end;
  TQwen21Vision = class
  private
    FFiles: TSafeTensorFiles;
    FPatch: TQNNLinearWeight;
    FPosition: TMemoryBlock;
    FBlocks: array[0..26] of TQwen21VisionBlock;
    FMerger: TQwen21VisionMerger;
    FDeep: array[0..2] of TQwen21VisionMerger;
    FStatus: TQwen21Status;
  public
    constructor Create(const Checkpoint: string; const Status: TQwen21Status = nil);
    destructor Destroy; override;
    procedure EncodeGPU(
        const Ctx: TQNNGPUContext;
        var Image: TQwen21ImageCondition;
        var Resident: TQwen21GPUImageCondition
    );
    procedure Encode(var Image: TQwen21ImageCondition);
  end;
procedure Q21AffineNorm(const Dst, Src, Gamma, Beta: TMemoryBlock; const Rows, Dim: Integer);
procedure Q21VisionRoPE(const Data: TMemoryBlock; const Height, Width: Integer);
implementation
uses
  System.Math,
  QuickNN.Kernels;
const
  HIDDEN = 1152;
  HEADS = 16;
  HEAD_DIM = 72;

procedure Q21AffineNorm(const Dst, Src, Gamma, Beta: TMemoryBlock; const Rows, Dim: Integer);
var
  Mean, Variance, Sum, Inv: Double;
  Offset: NativeInt;
begin
  if (Gamma.Count <> Dim) or (Beta.Count <> Dim) then
    raise EArgumentException.Create('LayerNorm parameter shape mismatch');
  for var Row := 0 to Rows - 1 do
  begin
    Offset := NativeInt(Row) * Dim;
    Sum := 0;
    for var J := 0 to Dim - 1 do
      Sum := Sum + System.PSingle(Src)[Offset + J];
    Mean := Sum / Dim;
    Variance := 0;
    for var J := 0 to Dim - 1 do
      Variance := Variance + Sqr(System.PSingle(Src)[Offset + J] - Mean);
    Inv := 1 / Sqrt(Variance / Dim + 1E-6);
    for var J := 0 to Dim - 1 do
      System.PSingle(Dst)[Offset + J] :=
          (System.PSingle(Src)[Offset + J] - Mean) * Inv * System.PSingle(Gamma)[J] + System.PSingle(Beta)[J];
  end;
end;

procedure Q21VisionRoPE(const Data: TMemoryBlock; const Height, Width: Integer);
var
  Row, Y, X, Axis, Index: Integer;
  Angle, Cs, Sn: Double;
  A, B: Single;
  Base: NativeInt;
begin
  if (Height mod 2 <> 0) or (Width mod 2 <> 0) then
    raise EArgumentException.Create('Vision grid must be even');
  Row := 0;
  for var By := 0 to Height div 2 - 1 do
    for var Bx := 0 to Width div 2 - 1 do
      for var Dy := 0 to 1 do
        for var Dx := 0 to 1 do
        begin
          Y := 2 * By + Dy;
          X := 2 * Bx + Dx;
          for var J := 0 to HEAD_DIM div 2 - 1 do
          begin
            if J < 18 then
            begin
              Axis := Y;
              Index := J
            end
            else
            begin
              Axis := X;
              Index := J - 18
            end;
            Angle := Axis / Power(10000, 2 * Index / 36);
            Cs := Cos(Angle);
            Sn := Sin(Angle);
            for var H := 0 to HEADS - 1 do
            begin
              Base := (NativeInt(Row) * HEADS + H) * HEAD_DIM + J;
              A := System.PSingle(Data)[Base];
              B := System.PSingle(Data)[Base + 36];
              System.PSingle(Data)[Base] := A * Cs - B * Sn;
              System.PSingle(Data)[Base + 36] := A * Sn + B * Cs;
            end;
          end;
          Inc(Row);
        end;
end;

procedure GeluExact(const Data: TMemoryBlock; const Count: Integer);
var
  X, A, T, ErfValue: Double;
begin
  for var I := 0 to Count - 1 do
  begin
    X := System.PSingle(Data)[I];
    A := Abs(X) / Sqrt(2);
    T := 1 / (1 + 0.3275911 * A);
    ErfValue :=
        1
            - (((((1.061405429 * T - 1.453152027) * T) + 1.421413741) * T - 0.284496736) * T + 0.254829592)
                * T
                * Exp(-A * A);
    if X < 0 then
      ErfValue := -ErfValue;
    System.PSingle(Data)[I] := 0.5 * X * (1 + ErfValue);
  end;
end;

procedure TQwen21VisionBlock.Load(const Files: TSafeTensorFiles; const Index: Integer);
var
  P: string;
begin
  P := Format('model.visual.blocks.%d.', [Index]);
  Norm1 := QNNRequiredTensor(Files, P + 'norm1.weight').AsSingle(True);
  Norm1Bias := QNNRequiredTensor(Files, P + 'norm1.bias').AsSingle(True);
  Norm2 := QNNRequiredTensor(Files, P + 'norm2.weight').AsSingle(True);
  Norm2Bias := QNNRequiredTensor(Files, P + 'norm2.bias').AsSingle(True);
  Qkv.Load(Files, P + 'attn.qkv');
  Proj.Load(Files, P + 'attn.proj');
  FC1.Load(Files, P + 'mlp.linear_fc1');
  FC2.Load(Files, P + 'mlp.linear_fc2');
  if (Qkv.InputDim <> HIDDEN)
      or (Qkv.OutputDim <> 3 * HIDDEN)
      or (Proj.InputDim <> HIDDEN)
      or (Proj.OutputDim <> HIDDEN)
      or (FC1.InputDim <> HIDDEN)
      or (FC1.OutputDim <> 4304)
      or (FC2.InputDim <> 4304)
      or (FC2.OutputDim <> HIDDEN) then
    raise ESafeTensorError.Create('Unexpected Qwen3-VL visual block shape: ' + P);
end;
procedure TQwen21VisionBlock.Free;
begin
  Norm1.Free;
  Norm1Bias.Free;
  Norm2.Free;
  Norm2Bias.Free;
  Qkv.Free;
  Proj.Free;
  FC1.Free;
  FC2.Free
end;

procedure TQwen21VisionMerger.Load(const Files: TSafeTensorFiles; const Prefix: string; const Post: Boolean);
begin
  PostShuffle := Post;
  Norm := QNNRequiredTensor(Files, Prefix + '.norm.weight').AsSingle(True);
  NormBias := QNNRequiredTensor(Files, Prefix + '.norm.bias').AsSingle(True);
  FC1.Load(Files, Prefix + '.linear_fc1');
  FC2.Load(Files, Prefix + '.linear_fc2');
  if (FC1.InputDim <> 4608) or (FC1.OutputDim <> 4608) or (FC2.InputDim <> 4608) or (FC2.OutputDim <> 4096) then
    raise ESafeTensorError.Create('Unexpected visual merger: ' + Prefix);
end;
function TQwen21VisionMerger.Forward(const Src: TMemoryBlock; const Rows: Integer): TMemoryBlock;
var
  NormData, Projected: TMemoryBlock;
begin
  if Rows mod 4 <> 0 then
    raise EArgumentException.Create('Incomplete spatial merge group');
  try
    NormData := TMemoryBlock.Create([Rows, HIDDEN], 'vision-merge-norm');
    Projected := TMemoryBlock.Create([Rows div 4, 4608], 'vision-merge-hidden');
    if PostShuffle then
      Q21AffineNorm(NormData, Src, Self.Norm, Self.NormBias, Rows div 4, 4608)
    else
      Q21AffineNorm(NormData, Src, Self.Norm, Self.NormBias, Rows, HIDDEN);
    FC1.Forward(Projected, NormData, Rows div 4);
    GeluExact(Projected, Projected.Count);
    Result := TMemoryBlock.Create([Rows div 4, 4096], 'vision-merged-features');
    FC2.Forward(Result, Projected, Rows div 4);
    QNNCheckFinite(Result, Result.Count, 'Vision merger');
  finally
    NormData.Free;
    Projected.Free
  end;
end;
procedure TQwen21VisionMerger.Free;
begin
  Norm.Free;
  NormBias.Free;
  FC1.Free;
  FC2.Free
end;

constructor TQwen21Vision.Create(const Checkpoint: string; const Status: TQwen21Status);
var
  Tensor: PSafeTensor;
begin
  inherited Create;
  FStatus := Status;
  SetLength(FFiles, 1);
  FFiles[0] := TSafeTensorFile.Open(Checkpoint);
  Tensor := QNNRequiredTensor(FFiles, 'model.visual.patch_embed.proj.weight');
  if (Tensor.Ndim <> 5)
      or (Tensor.Shape[0] <> HIDDEN)
      or (Tensor.Shape[1] <> 3)
      or (Tensor.Shape[2] <> 2)
      or (Tensor.Shape[3] <> 16)
      or (Tensor.Shape[4] <> 16) then
    raise ESafeTensorError.Create('Unexpected visual patch embedding');
  FPatch.InputDim := 1536;
  FPatch.OutputDim := HIDDEN;
  FPatch.Weight := FFiles.GetLinearWeight(Tensor.Name, True);
  FPatch.Bias := QNNRequiredTensor(FFiles, 'model.visual.patch_embed.proj.bias').AsSingle(True);
  FPosition := FFiles.GetLinearWeight('model.visual.pos_embed.weight', True);
  if FPosition.Count <> 2304 * HIDDEN then
    raise ESafeTensorError.Create('Unexpected learned vision position table');
  for var I := 0 to 26 do
    FBlocks[I].Load(FFiles, I);
  FMerger.Load(FFiles, 'model.visual.merger', False);
  for var I := 0 to 2 do
    FDeep[I].Load(FFiles, Format('model.visual.deepstack_merger_list.%d', [I]), True);
end;
destructor TQwen21Vision.Destroy;
begin
  QNNGPUClearWeightCache;
  FPatch.Free;
  FPosition.Free;
  for var I := 0 to 26 do
    FBlocks[I].Free;
  FMerger.Free;
  for var I := 0 to 2 do
    FDeep[I].Free;
  for var I := 0 to High(FFiles) do
    FFiles[I].Close;
  inherited;
end;

procedure TQwen21Vision.Encode(var Image: TQwen21ImageCondition);
var
  Patches, X, N, Qkv, Q, K, V, A, Y, Mlp: TMemoryBlock;
  Gh, Gw, Rows, Row, Py, Px, Index, Y0, Y1, X0, X1: Integer;
  Hy, Wx, Fy, Fx, Position: Double;
  Block: ^TQwen21VisionBlock;
begin
  if (Image.Pixels.Channels <> 4) or (Image.Pixels.Width mod 32 <> 0) or (Image.Pixels.Height mod 32 <> 0) then
    raise EArgumentException.Create('Vision input must be RGBA with dimensions divisible by 32');
  Gh := Image.Pixels.Height div 16;
  Gw := Image.Pixels.Width div 16;
  Rows := Gh * Gw;
  Image.GridHeight := Gh;
  Image.GridWidth := Gw;
  if (Rows < 4) or (Rows > 4096) then
    raise EArgumentException.Create('Vision image token budget exceeded');
  try
    Patches := TMemoryBlock.Create([Rows, 1536], 'vision-patches');
    Row := 0;
    for var By := 0 to Gh div 2 - 1 do
      for var Bx := 0 to Gw div 2 - 1 do
        for var Dy := 0 to 1 do
          for var Dx := 0 to 1 do
          begin
            Py := (By * 2 + Dy) * 16;
            Px := (Bx * 2 + Dx) * 16;
            Index := Row * 1536;
            for var C := 0 to 2 do
              for var T := 0 to 1 do
                for var Yy := 0 to 15 do
                  for var Xx := 0 to 15 do
                  begin
                    System.PSingle(Patches)[Index] :=
                        Image.Pixels.Data[((Py + Yy) * Image.Pixels.Width + Px + Xx) * 4 + C] / 127.5 - 1;
                    Inc(Index);
                  end;
            Inc(Row);
          end;
    X := TMemoryBlock.Create([Rows, HIDDEN], 'vision-hidden');
    FPatch.Forward(X, Patches, Rows);
    Patches.Free;
    Row := 0;
    for var By := 0 to Gh div 2 - 1 do
      for var Bx := 0 to Gw div 2 - 1 do
        for var Dy := 0 to 1 do
          for var Dx := 0 to 1 do
          begin
            Hy := (By * 2 + Dy) * 47 / (Gh - 1);
            Wx := (Bx * 2 + Dx) * 47 / (Gw - 1);
            Y0 := Floor(Hy);
            X0 := Floor(Wx);
            Y1 := Min(Y0 + 1, 47);
            X1 := Min(X0 + 1, 47);
            Fy := Hy - Y0;
            Fx := Wx - X0;
            for var J := 0 to HIDDEN - 1 do
            begin
              Position :=
                  (1 - Fy)
                          * ((1 - Fx) * QNNWeightValue(FPosition, (Y0 * 48 + X0) * HIDDEN + J)
                              + Fx * QNNWeightValue(FPosition, (Y0 * 48 + X1) * HIDDEN + J))
                      + Fy
                          * ((1 - Fx) * QNNWeightValue(FPosition, (Y1 * 48 + X0) * HIDDEN + J)
                              + Fx * QNNWeightValue(FPosition, (Y1 * 48 + X1) * HIDDEN + J));
              System.PSingle(X)[NativeInt(Row) * HIDDEN + J] :=
                  System.PSingle(X)[NativeInt(Row) * HIDDEN + J] + Position;
            end;
            Inc(Row);
          end;
    N := TMemoryBlock.Create([Rows, HIDDEN], 'vision-norm');
    Qkv := TMemoryBlock.Create([Rows, 3 * HIDDEN], 'vision-qkv');
    Q := TMemoryBlock.Create([Rows, HIDDEN], 'vision-q');
    K := TMemoryBlock.Create([Rows, HIDDEN], 'vision-k');
    V := TMemoryBlock.Create([Rows, HIDDEN], 'vision-v');
    A := TMemoryBlock.Create([Rows, HIDDEN], 'vision-attn');
    Y := TMemoryBlock.Create([Rows, HIDDEN], 'vision-out');
    Mlp := TMemoryBlock.Create([Rows, 4304], 'vision-mlp');
    for var I := 0 to 26 do
    begin
      Block := @FBlocks[I];
      Q21AffineNorm(N, X, Block.Norm1, Block.Norm1Bias, Rows, HIDDEN);
      Block.Qkv.Forward(Qkv, N, Rows);
      for var Token := 0 to Rows - 1 do
      begin
        Move(
            System.PSingle(Qkv)[NativeInt(Token) * 3 * HIDDEN],
            System.PSingle(Q)[NativeInt(Token) * HIDDEN],
            HIDDEN * 4
        );
        Move(
            System.PSingle(Qkv)[NativeInt(Token) * 3 * HIDDEN + HIDDEN],
            System.PSingle(K)[NativeInt(Token) * HIDDEN],
            HIDDEN * 4
        );
        Move(
            System.PSingle(Qkv)[NativeInt(Token) * 3 * HIDDEN + 2 * HIDDEN],
            System.PSingle(V)[NativeInt(Token) * HIDDEN],
            HIDDEN * 4
        );
      end;
      Q21VisionRoPE(Q, Gh, Gw);
      Q21VisionRoPE(K, Gh, Gw);
      Q21Attention(A, Q, K, V, Rows, HEADS, HEADS, HEAD_DIM, 0);
      Block.Proj.Forward(Y, A, Rows);
      QNNAddInplace(X, Y, Rows * HIDDEN);
      Q21AffineNorm(N, X, Block.Norm2, Block.Norm2Bias, Rows, HIDDEN);
      Block.FC1.Forward(Mlp, N, Rows);
      Q21Gelu(Mlp, Mlp.Count);
      Block.FC2.Forward(Y, Mlp, Rows);
      QNNAddInplace(X, Y, Rows * HIDDEN);
      QNNCheckFinite(X, X.Count, 'Vision block ' + IntToStr(I));
      case I of
        8: Image.DeepStack[0] := FDeep[0].Forward(X, Rows);
        16: Image.DeepStack[1] := FDeep[1].Forward(X, Rows);
        24: Image.DeepStack[2] := FDeep[2].Forward(X, Rows)
      end;
      if Assigned(FStatus) and ((I + 1) mod 9 = 0) then
        FStatus(Format('Vision encoder %d/27', [I + 1]));
    end;
    Image.Visual := FMerger.Forward(X, Rows);
  finally
    Patches.Free;
    X.Free;
    N.Free;
    Qkv.Free;
    Q.Free;
    K.Free;
    V.Free;
    A.Free;
    Y.Free;
    Mlp.Free
  end;
end;
{$I QuickNN.Models.Qwen21.Vision.GPU.inc}
end.
