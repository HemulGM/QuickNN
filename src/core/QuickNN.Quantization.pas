unit QuickNN.Quantization;

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
  System.SysUtils, QuickNN.Common, QuickNN.SafeTensors, QuickNN.Vulkan;

type
  IQNNLinearAdapter = interface
    procedure Apply(const Dst, Src: TMemoryBlock; const Rows, InputDim, OutputDim: Integer);
  end;

  IQNNGPULinearAdapter = interface
    ['{33991D61-2403-4C69-A08D-18F0AC329D1A}']
    procedure ApplyGPU(const Context: TQNNGPUContext; const Dst, Src: TQNNGPUTensor; const Rows, InputDim, OutputDim: Integer);
  end;

  PQNNLinearWeight = ^TQNNLinearWeight;
  // ComfyUI int8_tensorwise / ConvRot storage. Activations stay F32; weights
  // remain packed INT8 in host mappings and the Vulkan cache (W8A32 execution).

  TQNNLinearWeight = record
    Weight, Scales, Bias: TMemoryBlock;
    InputDim, OutputDim, RotationGroup: Integer;
    Adapters: TArray<IQNNLinearAdapter>;
    procedure AddAdapter(const Adapter: IQNNLinearAdapter);
    function HasAdapters: Boolean;
    procedure Load(const Files: TSafeTensorFiles; const Prefix: string);
    procedure forward(const Dst, Src: TMemoryBlock; const Rows: Integer);
    procedure ForwardGPU(const Context: TQNNGPUContext; const Dst, Src: TQNNGPUTensor; const Rows: Integer);
    procedure Gather(const Dst: TMemoryBlock; const Ids: TArray<Integer>);
    procedure Free;
  end;

function QNNRequiredTensor(const Files: TSafeTensorFiles; const Name: string): PSafeTensor;

procedure QNNRegularHadamard(const Data: System.PSingle; const Count, GroupSize: Integer);

function QNNWeightValue(const Data: TMemoryBlock; const Index: NativeInt): Single;

procedure QNNCheckFinite(const Data: TMemoryBlock; const Count: NativeInt; const Context: string);

implementation

uses
  System.Math, System.JSON, QuickNN.Kernels;

function QNNRequiredTensor(const Files: TSafeTensorFiles; const Name: string): PSafeTensor;
begin
  Result := Files.GetTensor(Name);
  if Result = nil then
    raise ESafeTensorError.Create('Cannot find tensor "' + Name + '"');
end;

function QNNWeightValue(const Data: TMemoryBlock; const Index: NativeInt): Single;
var
  Bits: UInt32;
begin
  case Data.DataType of
    dtBF16:
      begin
        Bits := UInt32(PBF16(Data)[Index]) shl 16;
        Move(Bits, Result, 4)
      end;
    dtF32:
      Result := System.PSingle(Data)[Index];
  else
    raise EArgumentException.Create('Expected a BF16/F32 weight');
  end;
end;

procedure QNNCheckFinite(const Data: TMemoryBlock; const Count: NativeInt; const Context: string);
begin
  for var I: NativeInt := 0 to Count - 1 do
    if IsNan(System.PSingle(Data)[I]) or IsInfinite(System.PSingle(Data)[I]) then
      raise EInvalidOp.Create(Context + ': non-finite value at ' + IntToStr(I));
end;

procedure QNNRegularHadamard(const Data: System.PSingle; const Count, GroupSize: Integer);
var
  Stride, BlockStart, J, Base, Group, N: Integer;
  A, B, C, D, Scale: Single;
begin
  if (GroupSize < 4) or (Count < 0) or (Count mod GroupSize <> 0) then
    raise EArgumentException.Create('Invalid ConvRot group size/count');
  N := GroupSize;
  while (N > 1) and (N mod 4 = 0) do
    N := N div 4;
  if N <> 1 then
    raise EArgumentException.Create('ConvRot requires a power of four');
  Scale := 1 / Sqrt(GroupSize);
  // H4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]].
  // Its normalized Kronecker powers are symmetric and orthogonal.
  for Group := 0 to Count div GroupSize - 1 do
  begin
    Base := Group * GroupSize;
    Stride := 1;
    while Stride < GroupSize do
    begin
      BlockStart := 0;
      while BlockStart < GroupSize do
      begin
        for J := 0 to Stride - 1 do
        begin
          A := Data[Base + BlockStart + J];
          B := Data[Base + BlockStart + J + Stride];
          C := Data[Base + BlockStart + J + 2 * Stride];
          D := Data[Base + BlockStart + J + 3 * Stride];
          Data[Base + BlockStart + J] := A + B + C - D;
          Data[Base + BlockStart + J + Stride] := A + B - C + D;
          Data[Base + BlockStart + J + 2 * Stride] := A - B + C + D;
          Data[Base + BlockStart + J + 3 * Stride] := -A + B + C + D;
        end;
        Inc(BlockStart, 4 * Stride);
      end;
      Stride := Stride * 4;
    end;
    for J := 0 to GroupSize - 1 do
      Data[Base + J] := Data[Base + J] * Scale;
  end;
end;

procedure TQNNLinearWeight.Load(const Files: TSafeTensorFiles; const Prefix: string);
var
  Tensor, Marker, ScaleTensor, BiasTensor: PSafeTensor;
  JSON: TJSONValue;
  Text: UTF8String;
  FormatName: string;
  Scale: Single;
  N: Integer;
begin
  Self.Free;
  Self := Default(TQNNLinearWeight);
  Tensor := Files.GetTensor(Prefix + '.weight');
  if (Tensor = nil)
    or (Tensor.Ndim <> 2)
    or (Tensor.Shape[0] <= 0)
    or (Tensor.Shape[1] <= 0)
    or (Tensor.Shape[0] > MaxInt)
    or (Tensor.Shape[1] > MaxInt) then
    raise ESafeTensorError.Create('Missing or invalid linear matrix: ' + Prefix);
  OutputDim := Tensor.Shape[0];
  InputDim := Tensor.Shape[1];
  if Tensor.DType = StI8 then
  begin
    Marker := Files.GetTensor(Prefix + '.comfy_quant');
    ScaleTensor := Files.GetTensor(Prefix + '.weight_scale');
    if (Marker = nil)
      or (Marker.DType <> StU8)
      or (Marker.DataSize > 16384)
      or (ScaleTensor = nil)
      or (ScaleTensor.DType <> StF32) then
      raise ESafeTensorError.Create('INT8 matrix requires quantization metadata and F32 scales: ' + Prefix);
    SetString(Text, PAnsiChar(Marker.AsMappedPtr), Marker.DataSize);
    JSON := TJSONValue.ParseJSONValue(string(Text));
    if not (JSON is TJSONObject) then
    begin
      JSON.Free;
      raise ESafeTensorError.Create('Invalid INT8 metadata: ' + Prefix)
    end;
    try
      FormatName := JSON.GetValue<string>('format', '');
      if FormatName <> 'int8_tensorwise' then
        raise ESafeTensorError.Create('Unsupported quantization format ' + FormatName + ': ' + Prefix);
      if JSON.GetValue<Boolean>('convrot', False) then
      begin
        RotationGroup := JSON.GetValue<Integer>('convrot_groupsize', 256);
        N := RotationGroup;
        if N < 4 then
          raise ESafeTensorError.Create('Invalid ConvRot group: ' + Prefix);
        while (N > 1) and (N mod 4 = 0) do
          N := N div 4;
        if (N <> 1) or (InputDim mod RotationGroup <> 0) then
          raise ESafeTensorError.Create('ConvRot group does not divide input: ' + Prefix);
      end;
    finally
      JSON.Free
    end;
    Weight.AssignPtr(PShortInt(Tensor.AsMappedPtr), Tensor.Shape);
    Weight.WeightIdentity := Tensor.WeightIdentity;
    if (ScaleTensor.Count <> 1) and (ScaleTensor.Count <> OutputDim) then
      raise ESafeTensorError.Create('Invalid scale count: ' + Prefix);
    if ScaleTensor.Count = 1 then
    begin
      Scales := TMemoryBlock.Create([OutputDim], Prefix + '.expanded-scales');
      Scale := System.PSingle(ScaleTensor.AsMappedPtr)^;
      for var I := 0 to OutputDim - 1 do
        System.PSingle(Scales)[I] := Scale;
      Scales.MarkImmutable;
    end
    else
      Scales := ScaleTensor.AsSingle(True);
    for var I := 0 to OutputDim - 1 do
    begin
      Scale := System.PSingle(Scales)[I];
      if IsNan(Scale) or IsInfinite(Scale) or (Scale < 0) then
        raise ESafeTensorError.Create('Invalid quantization scale: ' + Prefix);
    end;
  end
  else
    Weight := Files.GetLinearWeight(Prefix + '.weight', True);
  BiasTensor := Files.GetTensor(Prefix + '.bias');
  if BiasTensor <> nil then
  begin
    if BiasTensor.Count <> OutputDim then
      raise ESafeTensorError.Create('Invalid bias shape: ' + Prefix);
    Bias := BiasTensor.AsSingle(True);
  end;
end;

procedure TQNNLinearWeight.forward(const Dst, Src: TMemoryBlock; const Rows: Integer);
var
  Rotated, Original, Source: TMemoryBlock;
  Input: System.PSingle;
  Sum: Double;
  Start, BatchRows: Integer;
begin
  if (Rows <= 0)
    or (Src.DataType <> dtF32)
    or (Dst.DataType <> dtF32)
    or (Int64(Rows) * InputDim > Src.Size - Src.Offset)
    or (Int64(Rows) * OutputDim > Dst.Size - Dst.Offset) then
    raise EArgumentException.Create('Invalid linear input/output span');
  Source := Src;
  try
    if HasAdapters
      and (NativeUInt(System.PSingle(Src)) < NativeUInt(System.PSingle(Dst)) + UInt64(Rows) * UInt64(OutputDim) * 4)
      and (NativeUInt(System.PSingle(Dst))
      < NativeUInt(System.PSingle(Src)) + UInt64(Rows) * UInt64(InputDim) * 4) then
    begin
      Original := TMemoryBlock.Create([Rows, InputDim], 'lora-original-input');
      Move(System.PSingle(Src)^, System.PSingle(Original)^, NativeInt(Rows) * InputDim * 4);
      Source := Original;
    end;
    if Weight.DataType <> DT_I8 then
      QNNLinearNoBias(Dst, Source, Weight, Rows, InputDim, OutputDim)
    else
    begin
      try
        Input := System.PSingle(Source);
        if RotationGroup <> 0 then
        begin
          Rotated := TMemoryBlock.Create([Rows, InputDim], 'convrot-activation');
          Move(Input^, System.PSingle(Rotated)^, NativeInt(Rows) * InputDim * 4);
          QNNRegularHadamard(System.PSingle(Rotated), Rows * InputDim, RotationGroup);
          Input := System.PSingle(Rotated);
        end;
        Start := 0;
        while Start < Rows do
        begin
          // Keep IO well below the backend's 1 GiB transient budget even for
          // 1024px images and fused 24576-column gate/up projections.
          BatchRows := Min(512, Rows - Start);
          if not TryVulkanInt8Linear(
            System.PSingle(Dst) + NativeInt(Start) * OutputDim,
            Input + NativeInt(Start) * InputDim,
            Weight,
            Scales,
            BatchRows,
            InputDim,
            OutputDim) then
          begin
            for var R := Start to Start + BatchRows - 1 do
              for var O := 0 to OutputDim - 1 do
              begin
                Sum := 0;
                for var K := 0 to InputDim - 1 do
                  Sum :=
                    Sum + Double(Input[NativeInt(R) * InputDim + K]) * PShortInt(Weight)[NativeInt(O) * InputDim + K];
                System.PSingle(Dst)[NativeInt(R) * OutputDim + O] := Sum * System.PSingle(Scales)[O];
              end;
          end;
          Inc(Start, BatchRows);
        end;
      finally
        Rotated.Free
      end;
    end;
    if Bias.IsAssigned then
      for var R := 0 to Rows - 1 do
        for var O := 0 to OutputDim - 1 do
          System.PSingle(Dst)[NativeInt(R) * OutputDim + O] :=
            System.PSingle(Dst)[NativeInt(R) * OutputDim + O] + System.PSingle(Bias)[O];
    for var Adapter in Adapters do
      Adapter.Apply(Dst, Source, Rows, InputDim, OutputDim);
  finally
    Original.Free
  end;
end;

procedure TQNNLinearWeight.ForwardGPU(const Context: TQNNGPUContext; const Dst, Src: TQNNGPUTensor; const Rows: Integer);
var
  gpuAdapter: IQNNGPULinearAdapter;
begin
  for var Adapter in Adapters do
    if not Supports(Adapter, IQNNGPULinearAdapter, gpuAdapter) then
      raise ENotSupportedException.Create('Linear adapter has no resident GPU implementation');
  Context.Linear(Dst, Src, Weight, Scales, Bias, Rows, InputDim, OutputDim, RotationGroup);
  for var Adapter in Adapters do
  begin
    Supports(Adapter, IQNNGPULinearAdapter, gpuAdapter);
    gpuAdapter.ApplyGPU(Context, Dst, Src, Rows, InputDim, OutputDim);
  end;
end;

procedure TQNNLinearWeight.Gather(const Dst: TMemoryBlock; const Ids: TArray<Integer>);
var
  Row: System.PSingle;
  Scale: Single;
begin
  if (Dst.DataType <> dtF32) or (Int64(Length(Ids)) * InputDim > Dst.Size - Dst.Offset) then
    raise EArgumentException.Create('Invalid embedding destination');
  for var I := 0 to High(Ids) do
  begin
    if (Ids[I] < 0) or (Ids[I] >= OutputDim) then
      raise ERangeError.Create('Token outside embedding vocabulary');
    Row := System.PSingle(Dst) + NativeInt(I) * InputDim;
    if Weight.DataType = DT_I8 then
    begin
      Scale := System.PSingle(Scales)[Ids[I]];
      for var K := 0 to InputDim - 1 do
        Row[K] := PShortInt(Weight)[NativeInt(Ids[I]) * InputDim + K] * Scale;
      if RotationGroup <> 0 then
        QNNRegularHadamard(Row, InputDim, RotationGroup);
    end
    else
      for var K := 0 to InputDim - 1 do
        Row[K] := QNNWeightValue(Weight, NativeInt(Ids[I]) * InputDim + K);
  end;
end;

procedure TQNNLinearWeight.AddAdapter(const Adapter: IQNNLinearAdapter);
begin
  if Adapter = nil then
    raise EArgumentException.Create('Nil linear adapter');
  SetLength(Adapters, Length(Adapters) + 1);
  Adapters[High(Adapters)] := Adapter;
end;

function TQNNLinearWeight.HasAdapters: Boolean;
begin
  Result := Length(Adapters) <> 0
end;

procedure TQNNLinearWeight.Free;
begin
  Adapters := nil;
  Weight.Free;
  Scales.Free;
  Bias.Free;
  InputDim := 0;
  OutputDim := 0;
  RotationGroup := 0;
end;

end.

