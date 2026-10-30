unit QuickNN.Models.LLM.Memory;

interface

uses
  System.SysUtils, System.Generics.Collections, QuickNN.Common,
  QuickNN.Models.LLM.Types;

type
  // Owns bounded native BF16/F32 or packed INT4 byte copies. Get returns borrowed storage, valid
  // until the next cache mutation. TryTake transfers ownership to the caller;
  // GPU allocation may demote another weight while that upload source stays alive.
  TLLMHostWeightCache = class
  private
    type
      TEntry = class
        Key: string;
        Memory: TMemoryBlock;
        Bytes: UInt64;
        Previous, Next: TEntry;
        destructor Destroy; override;
      end;
  private
    FItems: TDictionary<string, TEntry>;
    FFirst, FLast: TEntry;
    FLimit, FBytes, FHits, FMisses, FEvictions: UInt64;
    procedure Unlink(const Entry: TEntry);
    procedure Touch(const Entry: TEntry);
    procedure Evict;
    function GetCore(const Weight: TMemoryBlock; const TrackLookup: Boolean): TMemoryBlock;
    function KeyFor(const Weight: TMemoryBlock): string;
  public
    constructor Create(const Limit: UInt64);
    destructor Destroy; override;
    function Get(const Weight: TMemoryBlock): TMemoryBlock;
    procedure Store(const Weight: TMemoryBlock);
    function TryTake(const Weight: TMemoryBlock; out Memory: TMemoryBlock): Boolean;
    procedure Clear;
    procedure Status(var Value: TLLMMemoryStatus);
  end;

implementation

destructor TLLMHostWeightCache.TEntry.Destroy;
begin
  Memory.Free;
  inherited;
end;

constructor TLLMHostWeightCache.Create(const Limit: UInt64);
begin
  inherited Create;
  FLimit := Limit;
  FItems := TDictionary<string, TEntry>.Create;
end;

destructor TLLMHostWeightCache.Destroy;
begin
  Clear;
  FItems.Free;
  inherited;
end;

procedure TLLMHostWeightCache.Unlink(const Entry: TEntry);
begin
  if Entry.Previous <> nil then
    Entry.Previous.Next := Entry.Next
  else
    FFirst := Entry.Next;
  if Entry.Next <> nil then
    Entry.Next.Previous := Entry.Previous
  else
    FLast := Entry.Previous;
  Entry.Previous := nil;
  Entry.Next := nil;
end;

procedure TLLMHostWeightCache.Touch(const Entry: TEntry);
begin
  Entry.Previous := FLast;
  if FLast <> nil then
    FLast.Next := Entry
  else
    FFirst := Entry;
  FLast := Entry;
end;

procedure TLLMHostWeightCache.Evict;
var
  Entry: TEntry;
begin
  Entry := FFirst;
  if Entry = nil then
    raise EInvalidOp.Create('Empty host weight cache');
  Unlink(Entry);
  FItems.Remove(Entry.Key);
  Dec(FBytes, Entry.Bytes);
  Inc(FEvictions);
  Entry.Free;
end;

function TLLMHostWeightCache.KeyFor(const Weight: TMemoryBlock): string;
var
  Bytes: UInt64;
begin
  Bytes := UInt64(Weight.Size) * UInt64(DATATYPE_BITS[Weight.DataType] div 8);
  Result :=
    IntToHex(NativeUInt(Pointer(Weight.WeightIdentity)), SizeOf(Pointer) * 2)
    + '/'
    + IntToStr(Ord(Weight.DataType))
    + '/'
    + UIntToStr(Bytes);
end;

function TLLMHostWeightCache.Get(const Weight: TMemoryBlock): TMemoryBlock;
begin
  Result := GetCore(Weight, True);
end;

procedure TLLMHostWeightCache.Store(const Weight: TMemoryBlock);
begin
  GetCore(Weight, False);
end;

function TLLMHostWeightCache.TryTake(const Weight: TMemoryBlock; out Memory: TMemoryBlock): Boolean;
var
  Entry: TEntry;
begin
  Memory := Default(TMemoryBlock);
  if (Weight.Offset <> 0)
    or not Weight.IsAssigned
    or (Weight.WeightIdentity = nil)
    or not (Weight.DataType in [dtBF16, dtF32, dtU8]) then
  begin
    Inc(FMisses);
    Exit(False);
  end;
  Result := FItems.TryGetValue(KeyFor(Weight), Entry);
  if not Result then
  begin
    Inc(FMisses);
    Exit;
  end;
  Inc(FHits);
  Unlink(Entry);
  FItems.Remove(Entry.Key);
  Dec(FBytes, Entry.Bytes);
  // Transfer ownership out of the cache before GPU allocation can demote
  // another weight. This prevents eviction from invalidating the upload source.
  Memory := Entry.Memory;
  Entry.Memory := Default(TMemoryBlock);
  Entry.Free;
end;

function TLLMHostWeightCache.GetCore(const Weight: TMemoryBlock; const TrackLookup: Boolean): TMemoryBlock;
var
  Key: string;
  Entry: TEntry;
  Bytes: UInt64;
  Source: Pointer;
begin
  if not Weight.IsAssigned or (Weight.WeightIdentity = nil) or not (Weight.DataType in [dtBF16, dtF32, dtU8]) then
    raise EArgumentException.Create('Host cache requires immutable native or packed weights');
  // Expert tensors are full matrices. Slices keep their original offset/cache key.
  if (FLimit = 0) or (Weight.Offset <> 0) then
    Exit(Weight);
  Bytes := UInt64(Weight.Size) * UInt64(DATATYPE_BITS[Weight.DataType] div 8);
  if Bytes > FLimit then
    Exit(Weight);
  Key := KeyFor(Weight);
  if FItems.TryGetValue(Key, Entry) then
  begin
    if TrackLookup then
      Inc(FHits);
    Unlink(Entry);
    Touch(Entry);
    Exit(Entry.Memory);
  end;
  if TrackLookup then
    Inc(FMisses);
  while FBytes + Bytes > FLimit do
    Evict;
  Entry := TEntry.Create;
  try
    if Weight.DataType = dtU8 then
      Source := PByte(Weight)
    else if Weight.DataType = dtBF16 then
      Source := PBF16(Weight)
    else
      Source := System.PSingle(Weight);
    Entry.Memory := TMemoryBlock.Create(Weight.Shape, Weight.Name, Weight.DataType, Source);
    Entry.Memory.WeightIdentity := Weight.WeightIdentity;
    Entry.Key := Key;
    Entry.Bytes := Bytes;
    FItems.Add(Key, Entry);
  except
    Entry.Free;
    raise;
  end;
  Touch(Entry);
  Inc(FBytes, Bytes);
  Result := Entry.Memory;
end;

procedure TLLMHostWeightCache.Clear;
begin
  if FItems <> nil then
  begin
    for var Entry in FItems.Values do
      Entry.Free;
    FItems.Clear;
  end;
  FFirst := nil;
  FLast := nil;
  FBytes := 0;
end;

procedure TLLMHostWeightCache.Status(var Value: TLLMMemoryStatus);
begin
  Value.RAMCacheBytes := FBytes;
  Value.RAMCacheLimitBytes := FLimit;
  Value.RAMCacheHits := FHits;
  Value.RAMCacheMisses := FMisses;
  Value.RAMCacheEvictions := FEvictions;
end;

end.

