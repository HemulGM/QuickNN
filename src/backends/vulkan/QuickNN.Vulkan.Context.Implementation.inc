function QNNGPUFloatBits(const Value: Single): Integer;
begin
  Move(Value, Result, 4)
end;

constructor TQNNGPUContext.Create;
begin
  inherited Create;
{$IF defined(MSWINDOWS) and defined(CPUX64)}
  GPULock.Acquire;
  FLocked := True;
  FMask := GetExceptionMask;
  SetExceptionMask([ExInvalidOp, ExDenormalized, ExZeroDivide, ExOverflow, ExUnderflow, ExPrecision]);
  if not EnsureGPU then
    raise ENotSupportedException.Create('Resident Vulkan unavailable: ' + GPUStatus.LastError);
  if GPU.FResident <> nil then
    raise EInvalidOp.Create('Only one resident context may own the Vulkan command buffer');
  FThread := GetCurrentThreadId;
  FActivePrefetch := -1; FQueuedPrefetch := -1; FWaitPrefetch := -1;
  FDevice := GPU;
  FDevice.FResident := Self;
  FError := NewTensor(1);
  Kernel(GO_FILL, 1, FError, nil, nil, nil, [0]);
{$ELSE}
  raise ENotSupportedException.Create('Resident Vulkan requires Delphi Windows x64');
{$ENDIF}
end;

destructor TQNNGPUContext.Destroy;
begin
{$IF defined(MSWINDOWS) and defined(CPUX64)}
  try
    if FDevice <> nil then
    begin
      try
        EndWeightPrefetch
      finally
        DestroyPrefetch;
        FOnWeightEvicted := nil;
        FDevice.FResident := nil;
        for var Entry in FRetained do
          Dec(Entry.Pins);
        FRetained := nil;
        for var Tensor in FBuffers do
        begin
          FDevice.FreeMemory(Tensor.FMemory);
          Tensor.Free
        end;
        FDevice.FreeMemory(FUpload);
        FDevice.FreeMemory(FRead);
        FDevice.FreeMemory(FBatchUpload);
      end;
    end;
  finally
    if FLocked then
    begin
      SetExceptionMask(FMask);
      GPULock.Release
    end;
  end;
{$ENDIF}
  inherited;
end;

procedure TQNNGPUContext.Check(const Tensor: TQNNGPUTensor; const Minimum: Int64);
begin
  if (Tensor = nil) or (Tensor.FOwner <> Self) or not Tensor.FLive or (Minimum < 0) or (Tensor.Count < Minimum) then
    raise EArgumentException.Create('Invalid resident tensor owner, lifetime or span');
{$IF defined(MSWINDOWS) and defined(CPUX64)}
  if GetCurrentThreadId <> FThread then
    raise EInvalidOp.Create('Resident context must stay on its creating thread');
{$ENDIF}
end;

function TQNNGPUContext.ActivationBudget: UInt64;
{$IF defined(MSWINDOWS) and defined(CPUX64)}
var
  Budget, Usage, Reserve: UInt64;
{$ENDIF}
begin
  Result := FBytes;
{$IF defined(MSWINDOWS) and defined(CPUX64)}
  Inc(Result, FDevice.LocalMemoryHeadroom(Budget, Usage, Reserve));
{$ENDIF}
end;

procedure TQNNGPUContext.TrimActivations;
{$IF defined(MSWINDOWS) and defined(CPUX64)}
var
  Index: Integer;
{$ENDIF}
begin
{$IF defined(MSWINDOWS) and defined(CPUX64)}
  Flush;
  Index := 0;
  while Index < Length(FBuffers) do
    if not FBuffers[Index].FLive then
    begin
      Dec(FBytes, FBuffers[Index].FMemory.Size);
      FDevice.FreeMemory(FBuffers[Index].FMemory);
      FBuffers[Index].Free;
      FBuffers[Index] := FBuffers[High(FBuffers)];
      SetLength(FBuffers, Length(FBuffers) - 1);
    end
    else
      Inc(Index);
{$ENDIF}
end;

function TQNNGPUContext.NewTensor(const Count: Integer): TQNNGPUTensor;
{$IF defined(MSWINDOWS) and defined(CPUX64)}
var
  Bytes, Budget, Usage, Reserve, Headroom: UInt64;
{$ENDIF}
begin
  Result := nil;
{$IF defined(MSWINDOWS) and defined(CPUX64)}
  if GetCurrentThreadId <> FThread then
    raise EInvalidOp.Create('Resident allocation must stay on its creating thread');
  if (Count < 1) or (UInt64(Count) * 4 > FDevice.VulkanDevice.DeviceProperties.Limits.MaxStorageBufferRange) then
    raise ERangeError.Create('Resident tensor exceeds device storage range');
  Bytes := UInt64(Count) * 4;
  for var Tensor in FBuffers do
    if not Tensor.FLive
        and (Tensor.FMemory.Size >= Bytes)
        and ((Result = nil) or (Tensor.FMemory.Size < Result.FMemory.Size)) then
      Result := Tensor;
  if Result = nil then
  begin
    Headroom := FDevice.LocalMemoryHeadroom(Budget, Usage, Reserve);
    if Bytes > Headroom then
    begin
      TrimActivations;
      Headroom := FDevice.LocalMemoryHeadroom(Budget, Usage, Reserve);
      // Only unpinned cached weights can yield space to live activations.
      while (Bytes > Headroom) and FDevice.EvictOne do
        Headroom := FDevice.LocalMemoryHeadroom(Budget, Usage, Reserve);
    end;
    if Bytes > Headroom then
      raise ERangeError.CreateFmt(
        'GPU activation needs %d MiB; available %d MiB after %d MiB VRAM reserve (budget %d MiB, used %d MiB)',
        [Bytes div 1048576, Headroom div 1048576, Reserve div 1048576, Budget div 1048576, Usage div 1048576]);
    Result := TQNNGPUTensor.Create;
    Result.FOwner := Self;
    try
      Result.FMemory := FDevice.CreateStorageMemory(Bytes)
    except
      Result.Free;
      raise
    end;
    Inc(FBytes, Bytes);
    FPeakBytes := Max(FPeakBytes, FBytes);
    SetLength(FBuffers, Length(FBuffers) + 1);
    FBuffers[High(FBuffers)] := Result;
    GPUStatus.ResidentPeakBytes := Max(GPUStatus.ResidentPeakBytes, FBytes);
  end;
  Result.Count := Count;
  Result.FLive := True;
{$ENDIF}
end;

procedure TQNNGPUContext.Recycle(var Tensor: TQNNGPUTensor);
begin
  if Tensor = nil then
    Exit;
  Check(Tensor);
  Tensor.FLive := False;
  Tensor := nil
end;

procedure TQNNGPUContext.CopyCHWRect(const Dst, Src: TQNNGPUTensor;
  const Channels, SrcWidth, SrcHeight, DstWidth, DstHeight,
    SrcX, SrcY, DstX, DstY, Width, Height: Integer);
begin
  if (Channels < 1) or (Width < 1) or (Height < 1)
    or (SrcX < 0) or (SrcY < 0) or (DstX < 0) or (DstY < 0)
    or (Int64(SrcX) + Width > SrcWidth) or (Int64(SrcY) + Height > SrcHeight)
    or (Int64(DstX) + Width > DstWidth) or (Int64(DstY) + Height > DstHeight)
    or (Int64(Channels) * Width * Height > MaxInt) or (Dst = Src) then
    raise EArgumentException.Create('Invalid GPU CHW rectangle');
  Check(Src, Int64(Channels) * SrcWidth * SrcHeight);
  Check(Dst, Int64(Channels) * DstWidth * DstHeight);
  Kernel(GO_CHW_RECT, Channels * Width * Height, Dst, Src, nil, nil,
    [SrcHeight * SrcWidth, SrcWidth, DstHeight * DstWidth, DstWidth,
      SrcY * SrcWidth + SrcX, DstY * DstWidth + DstX, Height * Width, Width]);
end;

function TQNNGPUContext.Scratch(const Slot, Count: Integer): TQNNGPUTensor;
begin
  if (Slot < 0) or (Slot > High(FScratch)) then
    raise EArgumentOutOfRangeException.Create('Resident scratch slot');
  if (FScratch[Slot] = nil) or (FScratch[Slot].Count < Count) then
  begin
    Recycle(FScratch[Slot]);
    FScratch[Slot] := NewTensor(Count)
  end;
  Result := FScratch[Slot];
end;

function TQNNGPUContext.AllocatedBytes: UInt64;
begin
  Result := FBytes;
{$IF defined(MSWINDOWS) and defined(CPUX64)}
  Inc(Result, FUpload.Size + FRead.Size + FBatchUpload.Size);
  for var Slot in FPrefetch do Inc(Result, Slot.Staging.Size);
{$ENDIF}
end;

{$IF defined(MSWINDOWS) and defined(CPUX64)}
procedure TQNNGPUContext.Start;
begin
  for var Slot in FSlots do
    if Slot >= RESIDENT_DISPATCH_SLOTS then
    begin
      Flush;
      Break;
    end;
  if FPending = 0 then
  begin
    FDevice.BeginCommadBuffer;
    if FComputeTimingPool <> nil then
    begin
      VkCmdResetQueryPool(FDevice.Cmd.Buffers[0], FComputeTimingPool, 0, 2);
      VkCmdWriteTimestamp(FDevice.Cmd.Buffers[0], VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, FComputeTimingPool, 0);
    end;
  end;
end;
procedure TQNNGPUContext.Barrier;
begin
  FDevice.MemoryBarrier(
      LongWord(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT) or LongWord(VK_PIPELINE_STAGE_TRANSFER_BIT),
      LongWord(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT) or LongWord(VK_PIPELINE_STAGE_TRANSFER_BIT),
      LongWord(VK_ACCESS_SHADER_WRITE_BIT) or LongWord(VK_ACCESS_TRANSFER_WRITE_BIT),
      LongWord(VK_ACCESS_SHADER_READ_BIT)
          or LongWord(VK_ACCESS_SHADER_WRITE_BIT)
          or LongWord(VK_ACCESS_TRANSFER_READ_BIT)
          or LongWord(VK_ACCESS_TRANSFER_WRITE_BIT)
  );
  Inc(FPending);
end;
procedure TQNNGPUContext.Pin(const Entry: TGPUWeight);
begin
  if Entry = nil then
    Exit;
  SetLength(FPins, Length(FPins) + 1);
  FPins[High(FPins)] := Entry
end;

procedure TQNNGPUContext.RecordKernel(
    const Op, N: Integer;
    const Dst, A, B, C: TQNNGPUTensor;
    const Args: array of Integer;
    const Wb, Wc: TGPUWeight;
    const Bf16b, Bf16c: Boolean
);
var
  Params: TVulkanArgs;
  Am, Bm, Cm: TVulkanMemory;
  Groups, Pipeline: Integer;
begin
  Check(Dst);
  if N < 1 then
    raise EArgumentException.Create('Empty resident dispatch');
  if Length(Args) > 8 then
    raise EArgumentException.Create('Too many tensor arguments');
  Am := FError.FMemory;
  Bm := Am;
  Cm := Am;
  if A <> nil then
  begin
    Check(A);
    Am := A.FMemory
  end;
  if B <> nil then
  begin
    Check(B);
    Bm := B.FMemory
  end;
  if C <> nil then
  begin
    Check(C);
    Cm := C.FMemory
  end;
  if Wb <> nil then
    Bm := Wb.Memory;
  if Wc <> nil then
    Cm := Wc.Memory;
  SetLength(Params, 20);
  Params[0] := Op;
  Params[1] := N;
  for var I := 2 to 15 do
    Params[I] := Integer(0);
  for var I := 0 to High(Args) do
    Params[6 + I] := Args[I];
  Params[14] := Ord(Bf16b);
  Params[15] := Ord(Bf16c);
  Params[16] := Am.Buffer;
  Params[17] := Bm.Buffer;
  Params[18] := Cm.Buffer;
  Params[19] := Dst.FMemory.Buffer;
  if Op = GO_CONV_ROT then
  begin
    if (Args[0] < 4) or (Args[0] > 1024) or (N mod Args[0] <> 0) then
      raise EArgumentException.Create('Unsupported resident ConvRot group');
    var Size := Args[0];
    while (Size > 1) and (Size mod 4 = 0) do
      Size := Size div 4;
    if Size <> 1 then
      raise EArgumentException.Create('ConvRot requires a power of four');
    Groups := N div Args[0];
  end
  else if Op = GO_LLM_TOPK_TILE then
    Groups := (Args[0] + 511) div 512
  else if Op = GO_GGUF_SWIGLU_VECTOR then
    Groups := (Args[1] + 3) div 4
  else if Op in [GO_GGUF_MATVEC, GO_LLM_SWIGLU_VECTOR, GO_INT4_SWIGLU_VECTOR] then
    Groups := (Args[1] + 7) div 8
  else if Op in [GO_MOE_INT4_MATVEC, GO_MOE_INT4_ACCUMULATE] then
    Groups := (Args[1] + 7) div 8
  else if Op = GO_MOE_INT4_GEMM then
    Groups := ((Args[1] + 31) div 32) * ((Args[2] + 7) div 8)
  else if Op = GO_HYBRID_QK_L2 then
    Groups := Args[0] * 2 * Args[1]
  else if Op = GO_HYBRID_DELTA_SCAN then
    Groups := Args[0]
  else if Op in [GO_FLASH_GROUP_NORM, GO_FLASH_PLE_GATE] then
    Groups := N div Args[0]
  else if Op = GO_HYBRID_GATED_NORM then
    Groups := N div Args[0]
  else if Op = GO_LLM_MATVEC_BF16 then
    Groups := (N + 7) div 8
  else if Op = GO_LLM_MATVEC then
    Groups := (N + 3) div 4
  else if Op in [GO_NORM, GO_MOD_NORM, GO_GROUP_NORM, GO_GROUP_NORM_STATS, GO_MOE_ROUTE] then
    Groups := Args[0]
  else
    Groups := (N + 255) div 256;
  Pipeline := 3;
  if Op in [GO_MOE_ROUTE, GO_MOE_GATHER, GO_MOE_SCATTER, GO_MOE_INT4_MATVEC, GO_MOE_INT4_GEMM, GO_INT4_SWIGLU_VECTOR, GO_MOE_INT4_ACCUMULATE] then
    Pipeline := 4;
  if ((Op >= GO_HYBRID_INT4_GATHER) and (Op <= GO_HYBRID_ROPE))
    or ((Op >= GO_FLASH_GROUP_NORM) and (Op <= GO_FLASH_SHARED_GATE)) then
    Pipeline := 5;
  if Op in [GO_GGUF_MATVEC, GO_GGUF_SWIGLU_VECTOR] then Pipeline := 6;
  var GroupsX := Min(Groups, Integer(FDevice.VulkanDevice.DeviceProperties.Limits.MaxComputeWorkGroupCount[0]));
  var GroupsY := (Groups + GroupsX - 1) div GroupsX;
  if (GroupsY > Integer(FDevice.VulkanDevice.DeviceProperties.Limits.MaxComputeWorkGroupCount[1]))
    or ((GroupsY > 1) and (Pipeline <> 3)) then
    raise ERangeError.Create('Tensor dispatch exceeds device workgroup limit');
  Params[2] := Groups;
  Start;
  FDevice.DispatchPipeline(TKernelHandle(Pipeline), Params, GroupsX, GroupsY, 1, FSlots[Pipeline]);
  Inc(FSlots[Pipeline]);
  Barrier;
  Inc(GPUStatus.TensorCalls);
  if Op = GO_CONV_ROT then
    Inc(GPUStatus.ConvRotCalls);
end;
{$ENDIF}

procedure TQNNGPUContext.RetainWeight(const Weight: TMemoryBlock; const Rows, Columns: Integer);
{$IF defined(MSWINDOWS) and defined(CPUX64)}
var
  Entry: TGPUWeight;
{$ENDIF}
begin
{$IF defined(MSWINDOWS) and defined(CPUX64)}
  Check(FError);
  Entry := FDevice.AcquireWeight(Weight, Rows, Columns, Weight.DataType = dtU8);
  for var Existing in FRetained do
    if Existing = Entry then
    begin
      Dec(Entry.Pins);
      Exit;
    end;
  try
    SetLength(FRetained, Length(FRetained) + 1);
    FRetained[High(FRetained)] := Entry;
    Inc(FRetainedBytes, Entry.Memory.Size);
  except
    Dec(Entry.Pins);
    raise;
  end;
{$ELSE}
  raise ENotSupportedException.Create('Resident weights require Win64');
{$ENDIF}
end;

function TQNNGPUContext.IsWeightResident(const Weight: TMemoryBlock; const Rows, Columns: Integer): Boolean;
begin
{$IF defined(MSWINDOWS) and defined(CPUX64)}
  Check(FError);
  Result := FDevice.FWeights.ContainsKey(FDevice.WeightKey(Weight, Rows, Columns));
{$ELSE}
  Result := False;
{$ENDIF}
end;

procedure TQNNGPUContext.Flush;
begin
{$IF defined(MSWINDOWS) and defined(CPUX64)}
  if FDevice = nil then
    Exit;
  if GetCurrentThreadId <> FThread then
    raise EInvalidOp.Create('Resident submission must stay on its creating thread');
  if FPending <> 0 then
  begin
    SubmitPending;
    FDevice.WaitForCompletion;
  end;
  // A synchronous preload can hold pins before any compute is recorded.
  for var Entry in FPins do
    if Entry <> nil then
      Dec(Entry.Pins);
  FPins := nil;
  FPending := 0;
  FillChar(FSlots, SizeOf(FSlots), 0);
{$ENDIF}
end;

function TQNNGPUContext.WeightBatchCapacity: UInt64;
begin
{$IF defined(MSWINDOWS) and defined(CPUX64)}
  Check(FError);
  Result := FDevice.FWeightLimit - Min(FDevice.FWeightLimit, FRetainedBytes);
{$ELSE}
  Result := 0;
{$ENDIF}
end;

procedure TQNNGPUContext.PreloadWeights(const Weights: array of TMemoryBlock; const StagingBytesLimit: UInt64);
{$IF defined(MSWINDOWS) and defined(CPUX64)}
const
  MAX_STAGING_BYTES: UInt64 = 512 * 1024 * 1024;
var
  Keys: TArray<string>;
  Sizes, PaddedSizes: TArray<UInt64>;
  Created: TArray<Boolean>;
  Unique: TDictionary<string, Byte>;
  Total, Capacity, StageBytes, StageLimit, Used, Offset, Chunk, Actual: UInt64;
  UploadedBytes, UploadedWeights: UInt64;
  Entry: TGPUWeight;
  Source, Mapped: Pointer;
  Recording: Boolean;
  Reused: TVulkanMemory;

  procedure PrefetchSources;
  type
    TPrefetchMemory =
        function(Process: THandle; Count: NativeUInt; Ranges: PWin32MemoryRangeEntry; Flags: Cardinal): LongBool;
        stdcall;
  var
    Prefetch: TPrefetchMemory;
    Ranges: TArray<TWin32MemoryRangeEntry>;
    Count: Integer;
    Bytes: UInt64;
  begin
    // Use the native pointer signature: some RTL versions declare the range
    // argument as a record by value, which cannot describe an array safely.
    Prefetch := TPrefetchMemory(GetProcAddress(GetModuleHandle('kernel32.dll'), 'PrefetchVirtualMemory'));
    if not Assigned(Prefetch) then
      Exit;
    SetLength(Ranges, Length(Weights));
    Count := 0;
    Bytes := 0;
    for var I := 0 to High(Weights) do
      if (FPins[I] = nil) and (Weights[I].DataPtr <> nil) then
      begin
        Ranges[Count].VirtualAddress := Weights[I].DataPtr;
        Ranges[Count].NumberOfBytes := Min(Sizes[I], MAX_STAGING_BYTES - Bytes);
        Inc(Bytes, Ranges[Count].NumberOfBytes);
        Inc(Count);
        if Bytes = MAX_STAGING_BYTES then
          Break;
      end;
    // Advisory only: low memory or an unavailable hint never changes results.
    if Count <> 0 then
      Prefetch(GetCurrentProcess, Count, @Ranges[0], 0);
  end;

  procedure SubmitCopies;
  begin
    if not Recording then
      Exit;
    if Mapped <> nil then
    begin
      FDevice.UnMapMemory(FBatchUpload);
      Mapped := nil;
    end;
    FDevice.MemoryBarrier(
        LongWord(VK_PIPELINE_STAGE_TRANSFER_BIT),
        LongWord(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT),
        LongWord(VK_ACCESS_TRANSFER_WRITE_BIT),
        LongWord(VK_ACCESS_SHADER_READ_BIT)
    );
    FDevice.EndCommandBuffer;
    Recording := False;
    FDevice.Finish;
    Inc(FDevice.FCounters.Submissions);
    Used := 0;
  end;
{$ENDIF}
begin
{$IF defined(MSWINDOWS) and defined(CPUX64)}
  Check(FError);
  if Length(Weights) = 0 then
    Exit;
  // Vulkan buffer copies require four-byte aligned offsets and lengths.
  StageLimit := Min(MAX_STAGING_BYTES, Max(UInt64(4), StagingBytesLimit)) and not UInt64(3);
  SetLength(Keys, Length(Weights));
  SetLength(Sizes, Length(Weights));
  SetLength(PaddedSizes, Length(Weights));
  SetLength(Created, Length(Weights));
  Capacity := WeightBatchCapacity;
  Total := 0;
  Unique := TDictionary<string, Byte>.Create;
  try
    for var I := 0 to High(Weights) do
    begin
      if (Length(Weights[I].Shape) <> 2)
          or (Weights[I].Offset <> 0)
          or (Weights[I].Shape[0] < 1)
          or (Weights[I].Shape[0] > High(Integer))
          or (Weights[I].Shape[1] < 1)
          or (Weights[I].Shape[1] > High(Integer))
          or (Weights[I].Size <> Weights[I].Shape[0] * Weights[I].Shape[1]) then
        raise EArgumentException.Create('Preload requires complete immutable native matrices');
      Keys[I] := FDevice.WeightKey(Weights[I], Weights[I].Shape[0], Weights[I].Shape[1]);
      if Weights[I].Encoding <> weNative then
        Sizes[I] := GGUFWeightBytes(Weights[I], Weights[I].Shape[0], Weights[I].Shape[1])
      else
        Sizes[I] := UInt64(Weights[I].Size) * UInt64(DATATYPE_BITS[Weights[I].DataType] div 8);
      PaddedSizes[I] := (Sizes[I] + 3) and not UInt64(3);
      if Unique.TryAdd(Keys[I], 0) then
      begin
        if PaddedSizes[I] > Capacity - Total then
          raise ERangeError.Create('Weight batch exceeds unretained GPU budget');
        Inc(Total, PaddedSizes[I]);
      end;
    end;
  finally
    Unique.Free
  end;
  Flush;
  SetLength(FPins, Length(Weights));
  Mapped := nil;
  Recording := False;
  Used := 0;
  UploadedBytes := 0;
  UploadedWeights := 0;
  Reused := Default(TVulkanMemory);
  try
    // Protect every existing batch hit before eviction can choose one of them.
    for var I := 0 to High(Weights) do
      if FDevice.FWeights.TryGetValue(Keys[I], Entry) then
      begin
        Inc(FDevice.FWeightClock);
        Entry.LastUse := FDevice.FWeightClock;
        Inc(Entry.Pins);
        FPins[I] := Entry;
        Inc(FDevice.FCounters.WeightCacheHits);
      end;
    PrefetchSources;
    for var I := 0 to High(Weights) do
    begin
      if FPins[I] <> nil then
        Continue;
      // A repeated key may have been inserted earlier in this same batch.
      if FDevice.FWeights.TryGetValue(Keys[I], Entry) then
      begin
        Inc(Entry.Pins);
        FPins[I] := Entry;
        Inc(FDevice.FCounters.WeightCacheHits);
        Continue;
      end;
      Inc(FDevice.FCounters.WeightCacheMisses);
      if FBatchUpload.Size = 0 then
        FBatchUpload := FDevice.CreateStagingMemory(Min(Total, StageLimit));
      StageBytes := Min(Total, StageLimit);
      if FBatchUpload.Size < StageBytes then
      begin
        FDevice.FreeMemory(FBatchUpload);
        FBatchUpload := Default(TVulkanMemory);
        FBatchUpload := FDevice.CreateStagingMemory(StageBytes);
      end;
      Reused := Default(TVulkanMemory);
      while FDevice.FWeightBytes + PaddedSizes[I] > FDevice.FWeightLimit do
      begin
        if Reused.Buffer = nil then
        begin
          if not FDevice.EvictOne(PaddedSizes[I], Reused) then
            raise ERangeError.Create('VRAM cannot hold the active weight batch');
        end
        else if not FDevice.EvictOne then
          raise ERangeError.Create('VRAM cannot hold the active weight batch');
      end;
      Entry := TGPUWeight.Create;
      Entry.Owner := FDevice;
      Entry.Identity := Weights[I].WeightIdentity;
      Inc(FDevice.FWeightClock);
      Entry.LastUse := FDevice.FWeightClock;
      Entry.Pins := 1;
      try
        if Reused.Buffer <> nil then
        begin
          Entry.Memory := Reused;
          Reused := Default(TVulkanMemory);
        end
        else
          Entry.Memory := FDevice.CreateStorageMemory(PaddedSizes[I]);
        FDevice.FWeights.Add(Keys[I], Entry);
      except
        Entry.Free;
        raise
      end;
      FPins[I] := Entry;
      Created[I] := True;
      Inc(FDevice.FWeightBytes, PaddedSizes[I]);
      if Weights[I].Encoding <> weNative then
        Source := GGUFWeightPointer(Weights[I])
      else if Weights[I].DataType = dtU8 then
        Source := PByte(Weights[I])
      else if Weights[I].DataType = dtBF16 then
        Source := PBF16(Weights[I])
      else
        Source := System.PSingle(Weights[I]);
      Offset := 0;
      while Offset < PaddedSizes[I] do
      begin
        if Used = FBatchUpload.Size then
          SubmitCopies;
        if not Recording then
        begin
          FDevice.BeginCommadBuffer;
          Recording := True;
          Mapped := FDevice.MapMemory(FBatchUpload);
        end;
        Chunk := Min(PaddedSizes[I] - Offset, FBatchUpload.Size - Used);
        Actual := Min(Chunk, Sizes[I] - Min(Offset, Sizes[I]));
        if Actual <> 0 then
          Move((PByte(Source) + Offset)^, (PByte(Mapped) + Used)^, Actual);
        if Chunk > Actual then
          FillChar((PByte(Mapped) + Used + Actual)^, Chunk - Actual, 0);
        FDevice.CopyBuffer(FBatchUpload, Entry.Memory, Chunk, Used, Offset);
        Inc(Used, Chunk);
        Inc(Offset, Chunk);
      end;
      Inc(UploadedWeights);
      Inc(UploadedBytes, Sizes[I]);
    end;
    SubmitCopies;
    Inc(FDevice.FCounters.WeightUploads, UploadedWeights);
    Inc(FDevice.FCounters.WeightUploadBytes, UploadedBytes);
  except
    // Finish recorded copies before releasing their destinations, then remove
    // all newly inserted entries so a failed batch can never look resident.
    try
      SubmitCopies
    finally
      FDevice.FreeMemory(Reused);
      for var EntryToRelease in FPins do
        if EntryToRelease <> nil then
          Dec(EntryToRelease.Pins);
      for var I := 0 to High(Created) do
        if Created[I] then
        begin
          Entry := FPins[I];
          FDevice.FWeights.Remove(Keys[I]);
          Dec(FDevice.FWeightBytes, Entry.Memory.Size);
          Entry.Free;
        end;
      FPins := nil;
    end;
    raise;
  end;
{$ELSE}
  raise ENotSupportedException.Create('Weight batches require Win64');
{$ENDIF}
end;

function TQNNGPUContext.UploadRaw(const Data: Pointer; const Count: Integer): TQNNGPUTensor;
begin
  Result := NewTensor(Count);
{$IF defined(MSWINDOWS) and defined(CPUX64)}
  if Data = nil then
    raise EArgumentNilException.Create('Resident upload');
  Flush;
  if FUpload.Size < UInt64(Count) * 4 then
  begin
    FDevice.FreeMemory(FUpload);
    FUpload := FDevice.CreateStagingMemory(UInt64(Count) * 4)
  end;
  FDevice.Upload(Data, FUpload, UInt64(Count) * 4, UInt64(Count) * 4);
  Start;
  FDevice.CopyBuffer(FUpload, Result.FMemory, UInt64(Count) * 4);
  Barrier;
  Inc(FDevice.FCounters.ActivationUploadBytes, UInt64(Count) * 4);
{$ENDIF}
end;
function TQNNGPUContext.Upload(const Data: TMemoryBlock): TQNNGPUTensor;
begin
  if Data.DataType <> dtF32 then
    raise EArgumentException.Create('Resident upload requires F32 storage');
  Result := UploadRaw(System.PSingle(Data), Data.Count);
end;

procedure TQNNGPUContext.Download(const Tensor: TQNNGPUTensor; const Dst: Pointer; const Count: Integer);
{$IF defined(MSWINDOWS) and defined(CPUX64)}
var
  N: Integer;
  Bytes: UInt64;
  Mapped: Pointer;
  Error: UInt32;
{$ENDIF}
begin
  Check(Tensor);
  if Dst = nil then
    raise EArgumentNilException.Create('Resident download');
{$IF defined(MSWINDOWS) and defined(CPUX64)}
  N := Count;
  if N = -1 then
    N := Tensor.Count;
  if N < 1 then
    raise EArgumentOutOfRangeException.Create('Empty resident download');
  Check(Tensor, N);
  Bytes := UInt64(N) * 4;
  if FRead.Size < Bytes + 4 then
  begin
    FDevice.FreeMemory(FRead);
    FRead := FDevice.CreateStagingMemory(Bytes + 4, True)
  end;
  // Append the tiny token/error readback to the recorded forward pass.
  // One fence now completes both compute and transfer, retaining weight pins.
  Start;
  FDevice.CopyBuffer(Tensor.FMemory, FRead, Bytes);
  FDevice.CopyBuffer(FError.FMemory, FRead, 4, 0, Bytes);
  FDevice.MemoryBarrier(
      LongWord(VK_PIPELINE_STAGE_TRANSFER_BIT),
      LongWord(VK_PIPELINE_STAGE_HOST_BIT),
      LongWord(VK_ACCESS_TRANSFER_WRITE_BIT),
      LongWord(VK_ACCESS_HOST_READ_BIT)
  );
  Inc(FPending);
  Flush;
  Mapped := FDevice.MapMemory(FRead);
  try
    Move((PByte(Mapped) + Bytes)^, Error, 4);
    if Error = 0 then
      Move(Mapped^, Dst^, Bytes)
  finally
    FDevice.UnMapMemory(FRead)
  end;
  Inc(FDevice.FCounters.Readbacks);
  Inc(FDevice.FCounters.ActivationDownloadBytes, Bytes + 4);
  if Error <> 0 then
    raise EInvalidOp.Create('Resident GPU computation produced non-finite values');
{$ENDIF}
end;

procedure TQNNGPUContext.Copy(const Dst, Src: TQNNGPUTensor; const Count, DstOffset, SrcOffset: Integer);
begin
  Check(Dst, Int64(DstOffset) + Count);
  Check(Src, Int64(SrcOffset) + Count);
  if (Count < 1) or (DstOffset < 0) or (SrcOffset < 0) or (Dst = Src) then
    raise EArgumentException.Create('Invalid/overlapping resident copy');
{$IF defined(MSWINDOWS) and defined(CPUX64)}
  Start;
  FDevice.CopyBuffer(Src.FMemory, Dst.FMemory, UInt64(Count) * 4, UInt64(SrcOffset) * 4, UInt64(DstOffset) * 4);
  Barrier;
{$ENDIF}
end;

procedure TQNNGPUContext.Kernel(const Op, N: Integer; const Dst, A, B, C: TQNNGPUTensor; const Args: array of Integer);
begin
{$IF defined(MSWINDOWS) and defined(CPUX64)}
  RecordKernel(Op, N, Dst, A, B, C, Args, nil, nil, False, False);
{$ENDIF}
end;

procedure TQNNGPUContext.KernelWeights(
    const Op, N: Integer;
    const Dst, A: TQNNGPUTensor;
    const B, C: TMemoryBlock;
    const Args: array of Integer
);
{$IF defined(MSWINDOWS) and defined(CPUX64)}
var
  Wb, Wc: TGPUWeight;
{$ENDIF}
begin
{$IF defined(MSWINDOWS) and defined(CPUX64)}
  Wb := nil;
  Wc := nil;
  try
    if B.IsAssigned then
      Wb := FDevice.AcquireWeight(B, 1, B.Count, B.DataType = dtU8);
    if C.IsAssigned then
      Wc := FDevice.AcquireWeight(C, 1, C.Count, C.DataType = dtU8);
    RecordKernel(Op, N, Dst, A, nil, nil, Args, Wb, Wc, B.DataType = dtBF16, C.DataType = dtBF16);
    Pin(Wb);
    Wb := nil;
    Pin(Wc);
    Wc := nil;
  finally
    if Wb <> nil then
      Dec(Wb.Pins);
    if Wc <> nil then
      Dec(Wc.Pins)
  end;
{$ENDIF}
end;

procedure TQNNGPUContext.Linear(
    const Dst, Src: TQNNGPUTensor;
    const Weight, Scales, Bias: TMemoryBlock;
    const Rows, InputDim, OutputDim, Rotation: Integer
);
{$IF defined(MSWINDOWS) and defined(CPUX64)}
var
  Entry: TGPUWeight;
  Input: TQNNGPUTensor;
  Flags, GroupsX: Integer;
{$ENDIF}
begin
  Check(Src, Int64(Rows) * InputDim);
  Check(Dst, Int64(Rows) * OutputDim);
  if (Rows < 1) or (InputDim < 1) or (OutputDim < 1) or (Src = Dst) then
    raise EArgumentException.Create('Invalid resident linear dimensions/alias');
{$IF defined(MSWINDOWS) and defined(CPUX64)}
  Entry := nil;
  try
    if Weight.DataType = DT_I8 then
      Entry := FDevice.AcquireInt8Weight(Weight, Scales, OutputDim, InputDim)
    else
      Entry := FDevice.AcquireWeight(Weight, OutputDim, InputDim);
    Input := Src;
    if Rotation <> 0 then
    begin
      Input := Scratch(0, Rows * InputDim);
      Kernel(GO_CONV_ROT, Rows * InputDim, Input, Src, nil, nil, [Rotation]);
    end;
    if (Rows = 1) and (Weight.Encoding <> weNative) then
    begin
      RecordKernel(GO_GGUF_MATVEC, OutputDim, Dst, Input, nil, nil,
        [InputDim, OutputDim, Ord(Weight.Encoding)], Entry, nil, False, False);
      Pin(Entry); Entry := nil;
      Inc(GPUStatus.GemmCalls); Inc(GPUStatus.GGUFCalls);
      if Bias.IsAssigned then KernelWeights(GO_CONV_OUTPUT, OutputDim, Dst, Dst, Bias,
        Default(TMemoryBlock), [OutputDim, 0, 0, 2]);
      Exit;
    end;
    Flags := 2 or (Ord(Weight.Encoding) shl 4);
    if Weight.DataType = DT_I8 then
      Flags := Flags or 8
    else if Weight.DataType = dtBF16 then
      Flags := Flags or 4;
    GroupsX := (OutputDim + 63) div 64;
    if (Flags and 256) <> 0 then GroupsX := (OutputDim + 3) div 4;
    Start;
    FDevice.DispatchPipeline(
        TKernelHandle(0),
        [
            Rows,
            OutputDim,
            InputDim,
            Single(1),
            Input.FMemory.Buffer,
            0,
            InputDim,
            Entry.Memory.Buffer,
            0,
            InputDim,
            Single(0),
            Dst.FMemory.Buffer,
            0,
            OutputDim,
            Flags
        ],
        GroupsX,
        (Rows + 63) div 64,
        1,
        FSlots[0]
    );
    Inc(FSlots[0]);
    Barrier;
    Pin(Entry);
    Entry := nil;
    Inc(GPUStatus.GemmCalls);
    if Weight.Encoding <> weNative then Inc(GPUStatus.GGUFCalls);
    if Weight.DataType = dtBF16 then
      Inc(GPUStatus.BF16Calls);
    if Bias.IsAssigned then
      KernelWeights(GO_CONV_OUTPUT, Rows * OutputDim, Dst, Dst, Bias, Default(TMemoryBlock), [OutputDim, 0, 0, 2]);
  finally
    if Entry <> nil then
      Dec(Entry.Pins)
  end;
{$ENDIF}
end;

procedure TQNNGPUContext.Attention(
    const Dst, Q, K, V: TQNNGPUTensor;
    const Rows, Heads, KvHeads, Dim, Causal: Integer;
    const Limits: TQNNGPUTensor;
    const T5Bias: TQNNGPUTensor
);
{$IF defined(MSWINDOWS) and defined(CPUX64)}
var
  PackedKV: TQNNGPUTensor;
  Base, Extra, Mode: Integer;
{$ENDIF}
begin
  if (Rows < 1)
      or (Rows > 65535)
      or (Heads < 1)
      or (KvHeads < 1)
      or (Heads mod KvHeads <> 0)
      or (Dim < 1)
      or (Dim > 1152) then
    raise EArgumentException.Create('Invalid resident attention geometry');
  Check(Q, Int64(Rows) * Heads * Dim);
  Check(K, Int64(Rows) * KvHeads * Dim);
  Check(V, Int64(Rows) * KvHeads * Dim);
  Check(Dst, Int64(Rows) * Heads * Dim);
{$IF defined(MSWINDOWS) and defined(CPUX64)}
  if not FDevice.FAttentionAvailable then
    raise ENotSupportedException.Create('Resident attention needs 32 KiB shared memory');
  Base := 2 * Rows * KvHeads * Dim;
  Extra := Rows * Ord(Limits <> nil);
  Mode := Ord(Limits <> nil);
  if T5Bias <> nil then
  begin
    if (Limits <> nil) or (Causal <> 0) or (Int64(Heads) * Rows * Rows + Base > MaxInt) then
      raise EArgumentException.Create('Invalid T5 relative attention bias');
    Extra := Heads * Rows * Rows;
    Check(T5Bias, Extra);
    Mode := 3;
  end;
  PackedKV := Scratch(1, Base + Extra);
  Kernel(GO_PACK_KV, Base, PackedKV, K, V, nil, [Rows, KvHeads, Dim]);
  if Limits <> nil then
    Copy(PackedKV, Limits, Rows, Base);
  if T5Bias <> nil then Copy(PackedKV, T5Bias, Extra, Base);
  Start;
  FDevice.DispatchPipeline(
      TKernelHandle(2),
      [
          Rows,
          Heads,
          KvHeads,
          Dim,
          Causal,
          Mode, 0, Single(0), Single(0),
          Q.FMemory.Buffer,
          PackedKV.FMemory.Buffer,
          Dst.FMemory.Buffer
      ],
      Rows,
      Heads,
      1,
      FSlots[2]
  );
  Inc(FSlots[2]);
  Barrier;
  Inc(GPUStatus.AttentionCalls);
{$ENDIF}
end;

procedure TQNNGPUContext.VectorLinear(
    const Dst, Src: TQNNGPUTensor;
    const Weight: TMemoryBlock;
    const InputDim, OutputDim: Integer
);
{$IF defined(MSWINDOWS) and defined(CPUX64)}
var
  Entry: TGPUWeight;
  Op: Integer;
{$ENDIF}
begin
  Check(Src, InputDim);
  Check(Dst, OutputDim);
  if (InputDim < 1) or (OutputDim < 1) or (Src = Dst) or not (Weight.DataType in [dtF32, dtBF16]) then
    raise EArgumentException.Create('Invalid resident vector projection');
{$IF defined(MSWINDOWS) and defined(CPUX64)}
  Op := GO_LLM_MATVEC;
  if (Weight.DataType = dtBF16) and (InputDim mod 4 = 0) then
    Op := GO_LLM_MATVEC_BF16;
  Entry := FDevice.AcquireWeight(Weight, OutputDim, InputDim);
  try
    RecordKernel(Op, OutputDim, Dst, Src, nil, nil, [InputDim, OutputDim], Entry, nil, Weight.DataType = dtBF16, False);
    Pin(Entry);
    Entry := nil;
    Inc(GPUStatus.GemmCalls);
    if Weight.DataType = dtBF16 then
      Inc(GPUStatus.BF16Calls);
  finally
    if Entry <> nil then
      Dec(Entry.Pins);
  end;
{$ENDIF}
end;

function TQNNGPUContext.TrySwiGLUVector(const Dst, Src: TQNNGPUTensor; const Gate, Up: TMemoryBlock;
  const InputDim, OutputDim: Integer): Boolean;
{$IF defined(MSWINDOWS) and defined(CPUX64)}
var
  GateEntry, UpEntry: TGPUWeight;
  Op, Width: Integer;
  PairBytes: UInt64;
{$ENDIF}
begin
  Result := False;
  Check(Src, InputDim); Check(Dst, OutputDim);
  if (InputDim < 1) or (OutputDim < 1) or (Dst = Src) then
    raise EArgumentException.Create('Invalid fused SwiGLU dimensions');
{$IF defined(MSWINDOWS) and defined(CPUX64)}
  Op := GO_LLM_SWIGLU_VECTOR; Width := InputDim;
  if (Gate.Encoding <> weNative) and (Up.Encoding <> weNative) then Op := GO_GGUF_SWIGLU_VECTOR
  else if (Gate.DataType = dtU8) and (Up.DataType = dtU8) and (Gate.Encoding = weNative) and (Up.Encoding = weNative) then
  begin Op := GO_INT4_SWIGLU_VECTOR; Width := Int4RowBytes(InputDim) end
  else if (Gate.Encoding <> weNative) or (Up.Encoding <> weNative)
    or not (Gate.DataType in [dtF32, dtBF16]) or not (Up.DataType in [dtF32, dtBF16]) then Exit;
  if (Length(Gate.Shape) <> 2) or (Length(Up.Shape) <> 2)
    or (Gate.Shape[0] <> OutputDim) or (Up.Shape[0] <> OutputDim)
    or (Gate.Shape[1] <> Width) or (Up.Shape[1] <> Width) then
    raise EArgumentException.Create('Invalid fused SwiGLU weight shape');
  if Op = GO_GGUF_SWIGLU_VECTOR then
    PairBytes := GGUFWeightBytes(Gate, OutputDim, Width) + GGUFWeightBytes(Up, OutputDim, Width)
  else PairBytes := UInt64(Gate.Count) * UInt64(DATATYPE_BITS[Gate.DataType] div 8)
    + UInt64(Up.Count) * UInt64(DATATYPE_BITS[Up.DataType] div 8);
  // Scalar streaming remains available when a deliberately tiny cache cannot
  // hold both projections simultaneously. Existing resident pairs need no space.
  if (PairBytes > WeightBatchCapacity) and not (IsWeightResident(Gate, OutputDim, Width)
    and IsWeightResident(Up, OutputDim, Width)) then Exit;
  GateEntry := nil; UpEntry := nil;
  try
    GateEntry := FDevice.AcquireWeight(Gate, OutputDim, Width, Gate.DataType = dtU8);
    UpEntry := FDevice.AcquireWeight(Up, OutputDim, Width, Up.DataType = dtU8);
    if Op = GO_GGUF_SWIGLU_VECTOR then
      RecordKernel(Op, OutputDim, Dst, Src, nil, nil, [InputDim, OutputDim, Ord(Gate.Encoding), Ord(Up.Encoding)], GateEntry, UpEntry, False, False)
    else if Op = GO_INT4_SWIGLU_VECTOR then
      RecordKernel(Op, OutputDim, Dst, Src, nil, nil, [InputDim, OutputDim, Width div 4], GateEntry, UpEntry, False, False)
    else RecordKernel(Op, OutputDim, Dst, Src, nil, nil, [InputDim, OutputDim], GateEntry, UpEntry, Gate.DataType = dtBF16, Up.DataType = dtBF16);
    Pin(GateEntry); GateEntry := nil; Pin(UpEntry); UpEntry := nil;
    Inc(GPUStatus.GemmCalls, 2);
    if Op = GO_GGUF_SWIGLU_VECTOR then Inc(GPUStatus.GGUFCalls, 2);
    if Op = GO_INT4_SWIGLU_VECTOR then Inc(GPUStatus.Int4Calls, 2);
    if Gate.DataType = dtBF16 then Inc(GPUStatus.BF16Calls);
    if Up.DataType = dtBF16 then Inc(GPUStatus.BF16Calls);
    Result := True;
  finally
    if GateEntry <> nil then Dec(GateEntry.Pins);
    if UpEntry <> nil then Dec(UpEntry.Pins);
  end;
{$ENDIF}
end;

procedure TQNNGPUContext.Int4LinearAccumulate(const Dst, Src, Probabilities: TQNNGPUTensor;
  const Weight: TMemoryBlock; const InputDim, OutputDim, ProbabilityIndex: Integer);
{$IF defined(MSWINDOWS) and defined(CPUX64)}
var Entry: TGPUWeight;
{$ENDIF}
begin
  Check(Dst, OutputDim); Check(Src, InputDim); Check(Probabilities, Int64(ProbabilityIndex) + 1);
  if (InputDim < 1) or (OutputDim < 1) or (ProbabilityIndex < 0) or (Dst = Src) or (Dst = Probabilities)
    or (Weight.DataType <> dtU8) or (Weight.Encoding <> weNative) or (Weight.Offset <> 0)
    or (Length(Weight.Shape) <> 2) or (Weight.Shape[0] <> OutputDim) or (Weight.Shape[1] <> Int4RowBytes(InputDim)) then
    raise EArgumentException.Create('Invalid accumulated INT4 vector projection');
{$IF defined(MSWINDOWS) and defined(CPUX64)}
  Entry := FDevice.AcquireWeight(Weight, OutputDim, Int4RowBytes(InputDim), True);
  try
    RecordKernel(GO_MOE_INT4_ACCUMULATE, OutputDim, Dst, Src, nil, Probabilities,
      [InputDim, OutputDim, Int4RowBytes(InputDim) div 4, ProbabilityIndex], Entry, nil, False, False);
    Pin(Entry); Entry := nil;
    Inc(GPUStatus.Int4Calls);
  finally if Entry <> nil then Dec(Entry.Pins) end;
{$ENDIF}
end;

procedure TQNNGPUContext.Int4Linear(
    const Dst, Src: TQNNGPUTensor;
    const Weight: TMemoryBlock;
    const Rows, InputDim, OutputDim: Integer
);
{$IF defined(MSWINDOWS) and defined(CPUX64)}
var
  Entry: TGPUWeight;
  Op: Integer;
{$ENDIF}
begin
  Check(Src, Int64(Rows) * InputDim);
  Check(Dst, Int64(Rows) * OutputDim);
  if (Rows < 1)
      or (Rows > 128)
      or (InputDim < 1)
      or (OutputDim < 1)
      or (Src = Dst)
      or (Weight.DataType <> dtU8)
      or (Weight.Offset <> 0)
      or (Length(Weight.Shape) <> 2)
      or (Weight.Shape[0] <> OutputDim)
      or (Weight.Shape[1] <> Int4RowBytes(InputDim))
      or (Int64(Rows) * OutputDim > High(Integer)) then
    raise EArgumentException.Create('Invalid packed INT4 projection');
{$IF defined(MSWINDOWS) and defined(CPUX64)}
  Entry := FDevice.AcquireWeight(Weight, OutputDim, Int4RowBytes(InputDim), True);
  try
    Op := GO_MOE_INT4_GEMM;
    if Rows = 1 then
      Op := GO_MOE_INT4_MATVEC;
    RecordKernel(
        Op,
        Rows * OutputDim,
        Dst,
        Src,
        nil,
        nil,
        [InputDim, OutputDim, Rows, Int4RowBytes(InputDim) div 4],
        Entry,
        nil,
        False,
        False
    );
    Pin(Entry);
    Entry := nil;
    Inc(GPUStatus.GemmCalls);
    Inc(GPUStatus.Int4Calls);
  finally
    if Entry <> nil then
      Dec(Entry.Pins);
  end;
{$ELSE}
  raise ENotSupportedException.Create('INT4 inference requires Win64 Vulkan');
{$ENDIF}
end;

procedure TQNNGPUContext.SampleToken(
    const Dst, Logits, Seen: TQNNGPUTensor;
    const Vocab, TopK: Integer;
    const Temperature, TopP, Penalty, Uniform: Single
);
var
  K, CandidatesCount: Integer;
  Tiles: TQNNGPUTensor;
  Args: array[0..5] of Integer;
begin
  Check(Dst, 1);
  Check(Logits, Vocab);
  Check(Seen, Vocab);
  if (Vocab < 1)
      or (TopK < 1)
      or (TopK > 64)
      or IsNan(Temperature)
      or IsInfinite(Temperature)
      or (Temperature < 0)
      or IsNan(TopP)
      or (TopP <= 0)
      or (TopP > 1)
      or IsNan(Penalty)
      or IsInfinite(Penalty)
      or (Penalty <= 0)
      or IsNan(Uniform)
      or (Uniform < 0)
      or (Uniform >= 1)
      or (Dst = Logits)
      or (Dst = Seen) then
    raise EArgumentException.Create('Invalid GPU token sampling arguments');
  K := Min(Vocab, TopK);
  if Temperature = 0 then
    K := 1;
  Args[0] := Vocab;
  Args[1] := K;
  Args[2] := QNNGPUFloatBits(Temperature);
  Args[3] := QNNGPUFloatBits(TopP);
  Args[4] := QNNGPUFloatBits(Penalty);
  Args[5] := QNNGPUFloatBits(Uniform);
  if Vocab <= 512 then
    Kernel(GO_LLM_SAMPLE, 1, Dst, Logits, Seen, nil, Args)
  else
  begin
    CandidatesCount := ((Vocab + 511) div 512) * K;
    Tiles := Scratch(2, 2 * CandidatesCount);
    Kernel(GO_LLM_TOPK_TILE, 2 * CandidatesCount, Tiles, Logits, Seen, nil, Args);
    Args[0] := CandidatesCount;
    Kernel(GO_LLM_TOPK_MERGE, 1, Dst, Tiles, nil, nil, Args);
  end;
end;

procedure TQNNGPUContext.CachedAttention(
    const Dst, Q, Cache: TQNNGPUTensor;
    const QueryRows, Capacity, Heads, KvHeads, Dim, Position: Integer;
    const Window: Integer; const Softcap, Scale: Single; const Masked: Boolean
);
begin
  if (QueryRows < 1)
      or (QueryRows > 65535)
      or (Capacity < 1)
      or (Position < 0)
      or (Int64(Position) + QueryRows > Capacity)
      or (Heads < 1)
      or (KvHeads < 1)
      or (Heads mod KvHeads <> 0)
      or (Dim < 1)
      or (Dim > 1152) or (Window < 0) or (Softcap < 0) or IsNan(Softcap) or IsInfinite(Softcap)
      or (Scale < 0) or IsNan(Scale) or IsInfinite(Scale) then
    raise EArgumentException.Create('Invalid resident cached attention');
  Check(Q, Int64(QueryRows) * Heads * Dim);
  Check(Dst, Int64(QueryRows) * Heads * Dim);
  Check(Cache, Int64(2) * Capacity * KvHeads * Dim + Int64(Ord(Masked))*QueryRows*Capacity);
{$IF defined(MSWINDOWS) and defined(CPUX64)}
  if not FDevice.FAttentionAvailable then
    raise ENotSupportedException.Create('Cached attention needs 32 KiB shared memory');
  Start;
  FDevice.DispatchPipeline(
      TKernelHandle(2),
      [Capacity, Heads, KvHeads, Dim, Position, 2 + 2*Ord(Masked), Window, Softcap, Scale, Q.FMemory.Buffer, Cache.FMemory.Buffer, Dst.FMemory.Buffer],
      QueryRows,
      Heads,
      1,
      FSlots[2]
  );
  Inc(FSlots[2]);
  Barrier;
  Inc(GPUStatus.AttentionCalls);
{$ENDIF}
end;

procedure TQNNGPUContext.Gather(
    const Dst, Ids: TQNNGPUTensor;
    const Weight, Scales: TMemoryBlock;
    const Rows, Dim, Vocab, Rotation: Integer
);
{$IF defined(MSWINDOWS) and defined(CPUX64)}
var
  Entry: TGPUWeight;
  Kind: Integer;
{$ENDIF}
begin
  Check(Dst, Int64(Rows) * Dim);
  Check(Ids, Rows);
{$IF defined(MSWINDOWS) and defined(CPUX64)}
  Entry := nil;
  try
    Kind := 0;
    if Weight.DataType = DT_I8 then
    begin
      Entry := FDevice.AcquireInt8Weight(Weight, Scales, Vocab, Dim);
      Kind := 2
    end
    else
    begin
      Entry := FDevice.AcquireWeight(Weight, Vocab, Dim);
      if Weight.DataType = dtBF16 then
        Kind := 1;
      if Weight.Encoding <> weNative then Kind := 16 + Ord(Weight.Encoding);
    end;
    RecordKernel(GO_GATHER, Rows * Dim, Dst, nil, nil, Ids, [Dim, Kind, Vocab], Entry, nil, False, False);
    Pin(Entry);
    Entry := nil;
    if Rotation <> 0 then
      Kernel(GO_CONV_ROT, Rows * Dim, Dst, Dst, nil, nil, [Rotation]);
  finally
    if Entry <> nil then
      Dec(Entry.Pins)
  end;
{$ENDIF}
end;

procedure TQNNGPUContext.CheckFinite(const Tensor: TQNNGPUTensor; const Count: Integer);
var
  N: Integer;
begin
  N := Count;
  if N = -1 then
    N := Tensor.Count;
  Check(Tensor, N);
  Kernel(GO_FINITE, N, FError, Tensor, nil, nil, [])
end;
