// A single host thread owns both queues and cache metadata. Compute descriptors
// and activation buffers are reused only after the compute fence. Each DMA slot
// owns a mapped coherent staging buffer, command buffer, fence and semaphore.
{$IF defined(MSWINDOWS) and defined(CPUX64)}
procedure TQNNGPUContext.SubmitPending;
var
  Ready: VkSemaphore;
begin
  if (FPending = 0) or FDevice.ComputeSubmitted then Exit;
  if FComputeTimingPool <> nil then
    VkCmdWriteTimestamp(FDevice.Cmd.Buffers[0], VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, FComputeTimingPool, 1);
  FDevice.EndCommandBuffer;
  Ready := nil;
  if FWaitPrefetch >= 0 then Ready := FPrefetch[FWaitPrefetch].Ready;
  FDevice.Submit(Ready);
  if FWaitPrefetch >= 0 then
  begin
    FPrefetch[FWaitPrefetch].SemaphorePending := False;
    FWaitPrefetch := -1;
  end;
  Inc(FDevice.FCounters.Submissions);
end;

procedure TQNNGPUContext.ReleasePrefetchSlot(const Index: Integer);
var
  Info: VkSemaphoreCreateInfo;
  Times: array[0..1] of UInt64;
  First, Last: UInt64;
begin
  if Index < 0 then Exit;
  var Slot: PPrefetchSlot := @FPrefetch[Index];
  if Slot.Submitted then
  begin
    CheckVulkanResult(VkWaitForFences(FDevice.Device, 1, @Slot.Fence, VK_TRUE, UInt64.MaxValue));
    Slot.Submitted := False;
  end;
  if Slot.HasComputeTimes and (Slot.TimingPool <> nil) then
  begin
    if VkGetQueryPoolResults(FDevice.Device, Slot.TimingPool, 0, 2, SizeOf(Times), @Times[0],
      SizeOf(UInt64), LongWord(VK_QUERY_RESULT_64_BIT)) = VK_SUCCESS then
    begin
      Inc(FDevice.FCounters.TimedAsyncWeightBatches);
      Inc(FDevice.FCounters.AsyncTransferNanoseconds,
        Round((Times[1] - Times[0]) * Double(FDevice.VulkanDevice.DeviceProperties.Limits.TimestampPeriod)));
      First := Max(Times[0], Slot.ComputeTimes[0]); Last := Min(Times[1], Slot.ComputeTimes[1]);
      if Last > First then
        Inc(FDevice.FCounters.AsyncOverlapNanoseconds,
          Round((Last - First) * Double(FDevice.VulkanDevice.DeviceProperties.Limits.TimestampPeriod)));
    end;
    Slot.HasComputeTimes := False;
  end;
  // Cancellation can leave a signaled semaphore without a compute consumer.
  // Its upload fence is complete, so replacement cannot race either queue.
  if Slot.SemaphorePending then
  begin
    VkDestroySemaphore(FDevice.Device, Slot.Ready, nil);
    Slot.Ready := nil;
    Info := Default(VkSemaphoreCreateInfo);
    Info.SType := VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    CheckVulkanResult(VkCreateSemaphore(FDevice.Device, @Info, nil, @Slot.Ready));
    Slot.SemaphorePending := False;
  end;
  for var Entry in Slot.Entries do
    if Entry <> nil then Dec(Entry.Pins);
  Slot.Entries := nil;
  Slot.Keys := nil;
end;

procedure TQNNGPUContext.DestroyPrefetch;
begin
  for var I := 0 to High(FPrefetch) do
  begin
    var Slot: PPrefetchSlot := @FPrefetch[I];
    if Slot.Mapped <> nil then FDevice.UnMapMemory(Slot.Staging);
    Slot.Mapped := nil;
    FDevice.FreeMemory(Slot.Staging);
    Slot.Staging := Default(TVulkanMemory);
    if Slot.Fence <> nil then VkDestroyFence(FDevice.Device, Slot.Fence, nil);
    if Slot.Ready <> nil then VkDestroySemaphore(FDevice.Device, Slot.Ready, nil);
    Slot.Fence := nil; Slot.Ready := nil;
    if Slot.TimingPool <> nil then VkDestroyQueryPool(FDevice.Device, Slot.TimingPool, nil);
    Slot.TimingPool := nil;
  end;
  if FPrefetchPool <> nil then VkDestroyCommandPool(FDevice.Device, FPrefetchPool, nil);
  FPrefetchPool := nil;
  if FComputeTimingPool <> nil then VkDestroyQueryPool(FDevice.Device, FComputeTimingPool, nil);
  FComputeTimingPool := nil;
  FPrefetchEnabled := False;
end;
{$ENDIF}

function TQNNGPUContext.EnableWeightPrefetch(const MaxLayerBytes: UInt64): Boolean;
{$IF defined(MSWINDOWS) and defined(CPUX64)}
var
  PoolInfo: VkCommandPoolCreateInfo;
  AllocInfo: VkCommandBufferAllocateInfo;
  FenceInfo: VkFenceCreateInfo;
  SemaphoreInfo: VkSemaphoreCreateInfo;
  QueryInfo: VkQueryPoolCreateInfo;
  Profile: Boolean;
{$ENDIF}
begin
  Result := False;
{$IF defined(MSWINDOWS) and defined(CPUX64)}
  Check(FError);
  if FPrefetchEnabled then Exit(MaxLayerBytes <= FPrefetchBytes);
  if (FDevice.TransferQueue = nil) or (MaxLayerBytes = 0)
    or (MaxLayerBytes > UInt64(512) * 1024 * 1024)
    or (((MaxLayerBytes + 3) and not UInt64(3)) > WeightBatchCapacity div 2) then Exit;
  Flush;
  FPrefetchBytes := (MaxLayerBytes + 3) and not UInt64(3);
  try
    PoolInfo := Default(VkCommandPoolCreateInfo);
    PoolInfo.SType := VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    PoolInfo.QueueFamilyIndex := FDevice.TransferQueueIndex;
    PoolInfo.Flags := LongWord(VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
    CheckVulkanResult(VkCreateCommandPool(FDevice.Device, @PoolInfo, nil, @FPrefetchPool));
    AllocInfo := Default(VkCommandBufferAllocateInfo);
    AllocInfo.SType := VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    AllocInfo.CommandPool := FPrefetchPool;
    AllocInfo.Level := VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    AllocInfo.CommandBufferCount := 1;
    FenceInfo := Default(VkFenceCreateInfo);
    FenceInfo.SType := VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    SemaphoreInfo := Default(VkSemaphoreCreateInfo);
    SemaphoreInfo.SType := VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    Profile := (GetEnvironmentVariable('QUICKNN_PROFILE_TRANSFERS') = '1')
      and (FDevice.VulkanDevice.QueueProperties[FDevice.VulkanDevice.QueueIndex].TimestampValidBits = 64)
      and (FDevice.VulkanDevice.QueueProperties[FDevice.TransferQueueIndex].TimestampValidBits = 64);
    QueryInfo := Default(VkQueryPoolCreateInfo);
    QueryInfo.SType := VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    QueryInfo.QueryType := VK_QUERY_TYPE_TIMESTAMP; QueryInfo.QueryCount := 2;
    if Profile then CheckVulkanResult(VkCreateQueryPool(FDevice.Device, @QueryInfo, nil, @FComputeTimingPool));
    for var I := 0 to High(FPrefetch) do
    begin
      var Slot: PPrefetchSlot := @FPrefetch[I];
      CheckVulkanResult(VkAllocateCommandBuffers(FDevice.Device, @AllocInfo, @Slot.CommandBuffer));
      CheckVulkanResult(VkCreateFence(FDevice.Device, @FenceInfo, nil, @Slot.Fence));
      CheckVulkanResult(VkCreateSemaphore(FDevice.Device, @SemaphoreInfo, nil, @Slot.Ready));
      if Profile then CheckVulkanResult(VkCreateQueryPool(FDevice.Device, @QueryInfo, nil, @Slot.TimingPool));
      Slot.Staging := FDevice.CreateStagingMemory(FPrefetchBytes);
      Slot.Mapped := FDevice.MapMemory(Slot.Staging);
    end;
    FPrefetchEnabled := True;
    Result := True;
  except
    // Resource pressure changes scheduling, never the model's numerical path.
    DestroyPrefetch;
  end;
{$ENDIF}
end;

procedure TQNNGPUContext.BeginWeightPrefetch;
begin
{$IF defined(MSWINDOWS) and defined(CPUX64)}
  Check(FError);
  if not FPrefetchEnabled then raise EInvalidOp.Create('Layer prefetch is disabled');
  // Submit before RAM-cache lookups and CPU copies. The command buffer stays
  // immutable until PrefetchWeights joins this fence, including on exceptions.
  SubmitPending;
{$ENDIF}
end;

procedure TQNNGPUContext.PrefetchWeights(const Weights: array of TMemoryBlock; const DeferJoin: Boolean);
{$IF defined(MSWINDOWS) and defined(CPUX64)}
var
  Keys: TArray<string>;
  Sizes, Padded: TArray<UInt64>;
  Created: TArray<Boolean>;
  Unique: TDictionary<string, Byte>;
  Entry: TGPUWeight;
  Reused: TVulkanMemory;
  Total, Used, Actual: UInt64;
  SlotIndex: Integer;
  BeginInfo: VkCommandBufferBeginInfo;
  SubmitInfo: VkSubmitInfo;
  CopyInfo: VkBufferCopy;
  Source: Pointer;
  Recording, Prepared: Boolean;
{$ENDIF}
begin
{$IF defined(MSWINDOWS) and defined(CPUX64)}
  Check(FError);
  if not FPrefetchEnabled or (FQueuedPrefetch >= 0) or (Length(Weights) = 0) then
    raise EInvalidOp.Create('Invalid layer prefetch state');
  SetLength(Keys, Length(Weights)); SetLength(Sizes, Length(Weights));
  SetLength(Padded, Length(Weights)); SetLength(Created, Length(Weights));
  Total := 0;
  Unique := TDictionary<string, Byte>.Create;
  try
    for var I := 0 to High(Weights) do
    begin
      if (Length(Weights[I].Shape) <> 2) or (Weights[I].Offset <> 0)
        or (Weights[I].Shape[0] < 1) or (Weights[I].Shape[0] > High(Integer))
        or (Weights[I].Shape[1] < 1) or (Weights[I].Shape[1] > High(Integer))
        or (Weights[I].Size <> Weights[I].Shape[0] * Weights[I].Shape[1]) then
        raise EArgumentException.Create('Layer prefetch requires complete immutable matrices');
      Keys[I] := FDevice.WeightKey(Weights[I], Weights[I].Shape[0], Weights[I].Shape[1]);
      if Weights[I].Encoding <> weNative then Sizes[I] := GGUFWeightBytes(Weights[I], Weights[I].Shape[0], Weights[I].Shape[1])
      else Sizes[I] := UInt64(Weights[I].Size) * UInt64(DATATYPE_BITS[Weights[I].DataType] div 8);
      Padded[I] := (Sizes[I] + 3) and not UInt64(3);
      if Unique.TryAdd(Keys[I], 0) then Inc(Total, Padded[I]);
    end;
  finally Unique.Free end;
  if (Total > FPrefetchBytes) or (Total > WeightBatchCapacity div 2) then
    raise ERangeError.Create('Layer prefetch exceeds the double-buffer budget');
  BeginWeightPrefetch;
  SlotIndex := FNextPrefetch;
  var Slot: PPrefetchSlot := @FPrefetch[SlotIndex];
  ReleasePrefetchSlot(SlotIndex);
  Prepared := Length(Slot.PreparedKeys) = Length(Keys);
  if Prepared then
    for var I := 0 to High(Keys) do
      if Slot.PreparedKeys[I] <> Keys[I] then Prepared := False;
  Slot.Keys := Keys; SetLength(Slot.Entries, Length(Weights));
  Reused := Default(TVulkanMemory); Used := 0; Recording := False;
  try
    // Protect all cache hits before eviction. This includes duplicate aliases.
    for var I := 0 to High(Weights) do
      if FDevice.FWeights.TryGetValue(Keys[I], Entry) then
      begin
        Inc(Entry.Pins); Slot.Entries[I] := Entry;
        Inc(FDevice.FCounters.WeightCacheHits);
      end;
    for var I := 0 to High(Weights) do
    begin
      if Slot.Entries[I] <> nil then Continue;
      if FDevice.FWeights.TryGetValue(Keys[I], Entry) then
      begin
        Inc(Entry.Pins); Slot.Entries[I] := Entry;
        Inc(FDevice.FCounters.WeightCacheHits); Continue;
      end;
      while FDevice.FWeightBytes + Padded[I] > FDevice.FWeightLimit do
      begin
        if Reused.Buffer = nil then
        begin
          if not FDevice.EvictOne(Padded[I], Reused) then
            raise ERangeError.Create('VRAM cannot hold both streamed layers');
        end
        else if not FDevice.EvictOne then raise ERangeError.Create('Pinned layer prefetch budget exhausted');
      end;
      Entry := TGPUWeight.Create;
      Entry.Owner := FDevice; Entry.Identity := Weights[I].WeightIdentity;
      Inc(FDevice.FWeightClock); Entry.LastUse := FDevice.FWeightClock; Entry.Pins := 1;
      try
        if Reused.Buffer <> nil then
        begin Entry.Memory := Reused; Reused := Default(TVulkanMemory) end
        else Entry.Memory := FDevice.CreateStorageMemory(Padded[I]);
        FDevice.FWeights.Add(Keys[I], Entry);
      except Entry.Free; raise end;
      Inc(FDevice.FWeightBytes, Padded[I]); Slot.Entries[I] := Entry; Created[I] := True;
      Inc(FDevice.FCounters.WeightCacheMisses);
      if Weights[I].Encoding <> weNative then Source := GGUFWeightPointer(Weights[I])
      else if Weights[I].DataType = dtU8 then Source := PByte(Weights[I])
      else if Weights[I].DataType = dtBF16 then Source := PBF16(Weights[I])
      else Source := System.PSingle(Weights[I]);
      Actual := Sizes[I];
      if not Prepared then
      begin
        Move(Source^, (PByte(Slot.Mapped) + Used)^, Actual);
        if Padded[I] > Actual then FillChar((PByte(Slot.Mapped) + Used + Actual)^, Padded[I] - Actual, 0);
      end;
      if not Recording then
      begin
        CheckVulkanResult(VkResetCommandBuffer(Slot.CommandBuffer, 0));
        BeginInfo := Default(VkCommandBufferBeginInfo);
        BeginInfo.SType := VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        CheckVulkanResult(VkBeginCommandBuffer(Slot.CommandBuffer, @BeginInfo));
        Recording := True;
        if Slot.TimingPool <> nil then
        begin
          VkCmdResetQueryPool(Slot.CommandBuffer, Slot.TimingPool, 0, 2);
          VkCmdWriteTimestamp(Slot.CommandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, Slot.TimingPool, 0);
        end;
      end;
      CopyInfo := Default(VkBufferCopy); CopyInfo.SrcOffset := Used; CopyInfo.Size := Padded[I];
      if Prepared then CopyInfo.SrcOffset := Slot.PreparedOffsets[I];
      VkCmdCopyBuffer(Slot.CommandBuffer, Slot.Staging.Buffer, Entry.Memory.Buffer, 1, @CopyInfo);
      Inc(Used, Padded[I]);
    end;
    if Recording then
    begin
      if Slot.TimingPool <> nil then
        VkCmdWriteTimestamp(Slot.CommandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, Slot.TimingPool, 1);
      CheckVulkanResult(VkEndCommandBuffer(Slot.CommandBuffer)); Recording := False;
      SubmitInfo := Default(VkSubmitInfo); SubmitInfo.SType := VK_STRUCTURE_TYPE_SUBMIT_INFO;
      SubmitInfo.CommandBufferCount := 1; SubmitInfo.PCommandBuffers := @Slot.CommandBuffer;
      SubmitInfo.SignalSemaphoreCount := 1; SubmitInfo.PSignalSemaphores := @Slot.Ready;
      CheckVulkanResult(VkResetFences(FDevice.Device, 1, @Slot.Fence));
      CheckVulkanResult(VkQueueSubmit(FDevice.TransferQueue, 1, @SubmitInfo, Slot.Fence));
      Slot.Submitted := True; Slot.SemaphorePending := True;
      Inc(FDevice.FCounters.Submissions); Inc(FDevice.FCounters.AsyncWeightBatches);
      Inc(FDevice.FCounters.AsyncWeightBytes, Used);
      if FDevice.ComputeSubmitted then Inc(FDevice.FCounters.OverlappedWeightBatches);
      for var I := 0 to High(Weights) do if Created[I] then
      begin Inc(FDevice.FCounters.WeightUploads); Inc(FDevice.FCounters.WeightUploadBytes, Sizes[I]) end;
    end;
    Slot.PreparedKeys := nil; Slot.PreparedOffsets := nil; Slot.PreparedIdentities := nil;
    FQueuedPrefetch := SlotIndex; FNextPrefetch := 1 - SlotIndex;
    if not DeferJoin then CompleteWeightPrefetch;
  except
    if Recording then VkEndCommandBuffer(Slot.CommandBuffer);
    FDevice.FreeMemory(Reused);
    Flush;
    ReleasePrefetchSlot(SlotIndex);
    // Until submission succeeds, newly inserted entries contain no valid data.
    for var I := 0 to High(Created) do if Created[I] then
      if FDevice.FWeights.TryGetValue(Keys[I], Entry) then
      begin
        FDevice.FWeights.Remove(Keys[I]); Dec(FDevice.FWeightBytes, Entry.Memory.Size); Entry.Free;
      end;
    raise;
  end;
{$ELSE}
  raise ENotSupportedException.Create('Asynchronous uploads require Win64');
{$ENDIF}
end;

procedure TQNNGPUContext.StageWeightPrefetch(const Weights: array of TMemoryBlock);
{$IF defined(MSWINDOWS) and defined(CPUX64)}
var
  Keys: TArray<string>;
  Offsets, Sizes, Padded: TArray<UInt64>;
  Identities: TArray<IInterface>;
  Unique: TDictionary<string, Integer>;
  Total: UInt64;
  Source: Pointer;
  Existing: Integer;
{$ENDIF}
begin
{$IF defined(MSWINDOWS) and defined(CPUX64)}
  Check(FError);
  if not FPrefetchEnabled or (FQueuedPrefetch < 0) then
    raise EInvalidOp.Create('Host staging requires an uploaded next layer');
  var Slot: PPrefetchSlot := @FPrefetch[FNextPrefetch];
  // Only the host buffer is recycled here. Its previous GPU weights can still
  // be in use by the compute queue until CompleteWeightPrefetch joins them.
  if Slot.Submitted then
  begin
    CheckVulkanResult(VkWaitForFences(FDevice.Device, 1, @Slot.Fence, VK_TRUE, UInt64.MaxValue));
    Slot.Submitted := False;
  end;
  Slot.PreparedKeys := nil; Slot.PreparedOffsets := nil; Slot.PreparedIdentities := nil;
  SetLength(Keys, Length(Weights)); SetLength(Offsets, Length(Weights));
  SetLength(Sizes, Length(Weights)); SetLength(Padded, Length(Weights));
  SetLength(Identities, Length(Weights)); Total := 0;
  Unique := TDictionary<string, Integer>.Create;
  try
    for var I := 0 to High(Weights) do
    begin
      if (Length(Weights[I].Shape) <> 2) or (Weights[I].Offset <> 0)
        or (Weights[I].Shape[0] < 1) or (Weights[I].Shape[0] > High(Integer))
        or (Weights[I].Shape[1] < 1) or (Weights[I].Shape[1] > High(Integer))
        or (Weights[I].Size <> Weights[I].Shape[0] * Weights[I].Shape[1]) then
        raise EArgumentException.Create('Host staging requires complete immutable matrices');
      Keys[I] := FDevice.WeightKey(Weights[I], Weights[I].Shape[0], Weights[I].Shape[1]);
      Identities[I] := Weights[I].WeightIdentity;
      if Weights[I].Encoding <> weNative then Sizes[I] := GGUFWeightBytes(Weights[I], Weights[I].Shape[0], Weights[I].Shape[1])
      else Sizes[I] := UInt64(Weights[I].Size) * UInt64(DATATYPE_BITS[Weights[I].DataType] div 8);
      Padded[I] := (Sizes[I] + 3) and not UInt64(3);
      if Unique.TryGetValue(Keys[I], Existing) then Offsets[I] := Offsets[Existing]
      else
      begin
        Unique.Add(Keys[I], I); Offsets[I] := Total; Inc(Total, Padded[I]);
      end;
    end;
    if Total > Slot.Staging.Size then raise ERangeError.Create('Host-staged layer exceeds its buffer');
    for var Pair in Unique do
    begin
      var I := Pair.Value;
      if Weights[I].Encoding <> weNative then Source := GGUFWeightPointer(Weights[I])
      else if Weights[I].DataType = dtU8 then Source := PByte(Weights[I])
      else if Weights[I].DataType = dtBF16 then Source := PBF16(Weights[I])
      else Source := System.PSingle(Weights[I]);
      Move(Source^, (PByte(Slot.Mapped) + Offsets[I])^, Sizes[I]);
      if Padded[I] > Sizes[I] then
        FillChar((PByte(Slot.Mapped) + Offsets[I] + Sizes[I])^, Padded[I] - Sizes[I], 0);
    end;
  finally Unique.Free end;
  Slot.PreparedKeys := Keys; Slot.PreparedOffsets := Offsets; Slot.PreparedIdentities := Identities;
{$ENDIF}
end;

procedure TQNNGPUContext.CompleteWeightPrefetch;
{$IF defined(MSWINDOWS) and defined(CPUX64)}
var
  HadCompute: Boolean;
{$ENDIF}
begin
{$IF defined(MSWINDOWS) and defined(CPUX64)}
  Check(FError);
  HadCompute := FDevice.ComputeSubmitted or (FPending <> 0);
  // CPU preparation of layer N+2 and DMA of N+1 have run alongside compute N.
  // Reuse compute descriptors/activations only after this fence completes.
  Flush;
  if HadCompute and (FComputeTimingPool <> nil) and (FQueuedPrefetch >= 0) then
  begin
    var Slot: PPrefetchSlot := @FPrefetch[FQueuedPrefetch];
    if Slot.SemaphorePending then
      Slot.HasComputeTimes := VkGetQueryPoolResults(FDevice.Device, FComputeTimingPool, 0, 2,
        SizeOf(Slot.ComputeTimes), @Slot.ComputeTimes[0], SizeOf(UInt64), LongWord(VK_QUERY_RESULT_64_BIT)) = VK_SUCCESS;
  end;
  ReleasePrefetchSlot(FActivePrefetch); FActivePrefetch := -1;
{$ENDIF}
end;

function TQNNGPUContext.ActivatePrefetchedWeights(const Weights: array of TMemoryBlock): Boolean;
begin
  Result := False;
{$IF defined(MSWINDOWS) and defined(CPUX64)}
  Check(FError);
  if FQueuedPrefetch < 0 then Exit;
  var Slot: PPrefetchSlot := @FPrefetch[FQueuedPrefetch];
  if Length(Weights) <> Length(Slot.Keys) then Exit;
  for var I := 0 to High(Weights) do
    if FDevice.WeightKey(Weights[I], Weights[I].Shape[0], Weights[I].Shape[1]) <> Slot.Keys[I] then Exit;
  if (FActivePrefetch >= 0) or (FWaitPrefetch >= 0) then
    raise EInvalidOp.Create('Previous prefetched layer is still active');
  FActivePrefetch := FQueuedPrefetch; FQueuedPrefetch := -1;
  if Slot.SemaphorePending then FWaitPrefetch := FActivePrefetch;
  Result := True;
{$ENDIF}
end;

procedure TQNNGPUContext.SealWeightPrefetch;
begin
{$IF defined(MSWINDOWS) and defined(CPUX64)}
  Check(FError);
  if FQueuedPrefetch >= 0 then raise EInvalidOp.Create('Unused prefetched layer at forward completion');
  if FActivePrefetch < 0 then Exit;
  if FPending = 0 then
  begin
    EndWeightPrefetch;
    Exit;
  end;
  // Transfer the last layer's pins to the normal compute batch, so logits and
  // sampling can share a submission without a special end-of-forward wait.
  var Slot: PPrefetchSlot := @FPrefetch[FActivePrefetch];
  for var Entry in Slot.Entries do Pin(Entry);
  Slot.Entries := nil; Slot.Keys := nil; FActivePrefetch := -1;
{$ENDIF}
end;

procedure TQNNGPUContext.EndWeightPrefetch;
begin
{$IF defined(MSWINDOWS) and defined(CPUX64)}
  Flush;
  for var I := 0 to High(FPrefetch) do
  begin
    ReleasePrefetchSlot(I);
    FPrefetch[I].PreparedKeys := nil; FPrefetch[I].PreparedOffsets := nil;
    FPrefetch[I].PreparedIdentities := nil;
  end;
  FActivePrefetch := -1; FQueuedPrefetch := -1; FWaitPrefetch := -1;
{$ENDIF}
end;
