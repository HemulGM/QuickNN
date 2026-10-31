unit QuickNN.Models.LLM.Quantization;

{$POINTERMATH ON}

interface

uses
  System.SysUtils, QuickNN.Common, QuickNN.SafeTensors;

type
  TLLMInt4Scope = (iqExperts, iqDense);

  TLLMInt4Cache = class
  private
    FFiles: TSafeTensorFiles;
    FDirectory, FFingerprint, FFormat, FIndexName: string;
    FScope: TLLMInt4Scope;
    procedure Build(const Sources: TSafeTensorFiles; const Progress: TProc<Integer, Integer>);
    procedure ValidateIndex;
  public
    constructor Create(const Sources: TSafeTensorFiles; const Directory: string; const Progress: TProc<Integer, Integer> = nil; const Scope: TLLMInt4Scope = iqExperts
    );
    destructor Destroy; override;
    function Weight(const Name: string; const Rows, Columns: Integer): TMemoryBlock;
    function MappedBytes: UInt64;
    property Directory: string read FDirectory;
    class function SourceFingerprint(const Sources: TSafeTensorFiles): string; static;
  end;

  // Kept as a source-compatible name for existing MoE callers.
  TMoEInt4Cache = TLLMInt4Cache;

implementation

uses
  System.Classes, System.IOUtils, System.JSON, System.Hash, System.Math,
  System.Generics.Collections, QuickNN.Quantization.Int4;

const
  CACHE_FORMAT = 'quicknn-moe-int4-g32-ls-v1';
  DENSE_CACHE_FORMAT = 'quicknn-dense-int4-g32-ls-v1';
  LEGACY_CACHE_FORMAT = 'envision-moe-int4-g32-ls-v1';
  CACHE_INDEX = 'experts.safetensors.index.json';
  MATRICES_PER_SHARD = 384;

function IsExpert(const Name: string): Boolean;
begin
  Result :=
    (Pos('.mlp.experts.', Name) <> 0)
    and (Name.EndsWith('.gate_proj.weight')
    or Name.EndsWith('.up_proj.weight')
    or Name.EndsWith('.down_proj.weight'));
end;

function FingerprintForFormat(const Sources: TSafeTensorFiles; const Format: string): string;
var
  Hash: THashSHA2;
begin
  Hash := THashSHA2.Create;
  Hash.Update(Format);
  for var FileInfo in Sources do
  begin
    Hash.Update(TPath.GetFullPath(FileInfo.Path));
    Hash.Update(IntToStr(FileInfo.FileSize));
    Hash.Update(FormatDateTime('yyyymmddhhnnsszzz', TFile.GetLastWriteTimeUtc(FileInfo.Path)));
    Hash.Update(string(FileInfo.HeaderJson));
  end;
  Result := Hash.HashAsString;
end;

class function TLLMInt4Cache.SourceFingerprint(const Sources: TSafeTensorFiles): string;
begin
  Result := FingerprintForFormat(Sources, CACHE_FORMAT);
end;

constructor TLLMInt4Cache.Create(const Sources: TSafeTensorFiles; const Directory: string; const Progress: TProc<Integer, Integer>; const Scope: TLLMInt4Scope);
var
  Lock: TFileStream;
  LegacyRoot, LegacyDirectory, LegacyFingerprint: string;
begin
  inherited Create;
  if Directory = '' then
    raise EArgumentException.Create('Missing INT4 cache directory');
  FScope := Scope;
  FIndexName := CACHE_INDEX;
  if FScope = iqDense then
    FIndexName := 'weights.safetensors.index.json';
  FFormat := CACHE_FORMAT;
  if FScope = iqDense then
    FFormat := DENSE_CACHE_FORMAT;
  FFingerprint := FingerprintForFormat(Sources, FFormat);
  FDirectory := TPath.Combine(TPath.GetFullPath(Directory), FFormat + '-' + FFingerprint);
  if (FScope = iqExperts) and not TFile.Exists(TPath.Combine(FDirectory, FIndexName)) then
  begin
    // Cache headers and fingerprints are versioned on disk. Reuse a verified
    // old cache without renaming directories or requantizing all expert weights.
    LegacyRoot := ExcludeTrailingPathDelimiter(TPath.GetFullPath(Directory));
    if SameText(ExtractFileName(LegacyRoot), '.quicknn-cache') then
      LegacyRoot := TPath.Combine(ExtractFileDir(LegacyRoot), '.envision-cache');
    LegacyFingerprint := FingerprintForFormat(Sources, LEGACY_CACHE_FORMAT);
    LegacyDirectory := TPath.Combine(LegacyRoot, LEGACY_CACHE_FORMAT + '-' + LegacyFingerprint);
    if TFile.Exists(TPath.Combine(LegacyDirectory, FIndexName)) then
    begin
      FDirectory := LegacyDirectory;
      FFingerprint := LegacyFingerprint;
      FFormat := LEGACY_CACHE_FORMAT;
    end;
  end;
  if not TFile.Exists(TPath.Combine(FDirectory, FIndexName)) then
  begin
    TDirectory.CreateDirectory(FDirectory);
    // Only a committed index makes a cache visible. Exclusive sharing also
    // prevents concurrent builders from publishing the same shard names.
    Lock := TFileStream.Create(TPath.Combine(FDirectory, 'build.lock'), fmCreate or fmShareExclusive);
    try
      if not TFile.Exists(TPath.Combine(FDirectory, FIndexName)) then
        Build(Sources, Progress);
    finally
      Lock.Free;
    end;
  end;
  ValidateIndex;
  FFiles := LoadSafeTensorShards(FDirectory, FIndexName, '');
  for var FileInfo in FFiles do
    if (FileInfo.JSON.GetValue<string>('__metadata__.format', '') <> FFormat)
      or (FileInfo.JSON.GetValue<string>('__metadata__.source_fingerprint', '') <> FFingerprint) then
      raise ESafeTensorError.Create('Invalid INT4 shard format');
end;

destructor TLLMInt4Cache.Destroy;
begin
  for var I := 0 to High(FFiles) do
    FFiles[I].Close;
  inherited;
end;

procedure TLLMInt4Cache.ValidateIndex;
var
  Index: TJSONValue;
  Map: TJSONObject;
begin
  Index := TJSONValue.ParseJSONValue(TFile.ReadAllText(TPath.Combine(FDirectory, FIndexName), TEncoding.UTF8));
  try
    if not (Index is TJSONObject)
      or (Index.GetValue<string>('metadata.format', '') <> FFormat)
      or (Index.GetValue<string>('metadata.source_fingerprint', '') <> FFingerprint) then
      raise ESafeTensorError.Create('Invalid or stale INT4 cache index');
    if not (Index.FindValue('weight_map') is TJSONObject) then
      raise ESafeTensorError.Create('Missing INT4 weight map');
    Map := TJSONObject(Index.FindValue('weight_map'));
    for var Pair in Map do
      if not (Pair.JsonValue is TJSONString)
        or (Pair.JsonValue.Value = '')
        or (ExtractFileName(Pair.JsonValue.Value) <> Pair.JsonValue.Value)
        or Pair.JsonValue.Value.Contains(':')
        or Pair.JsonValue.Value.Contains('/')
        or Pair.JsonValue.Value.Contains('\') then
        raise ESafeTensorError.Create('Invalid INT4 shard path');
  finally
    Index.Free;
  end;
end;

procedure TLLMInt4Cache.Build(const Sources: TSafeTensorFiles; const Progress: TProc<Integer, Integer>);
var
  Tensors: TList<PSafeTensor>;
  Header, Index, Map, Metadata, Item: TJSONObject;
  Stream: TFileStream;
  Source, Encoded: TMemoryBlock;
  HeaderText: UTF8String;
  HeaderBytes: UInt64;
  Offset, Bytes: Int64;
  Name, Temporary: string;
  Shards, Start, Count, PerShard: Integer;
  Id: TGUID;
begin
  Tensors := TList<PSafeTensor>.Create;
  Index := TJSONObject.Create;
  try
    for var F := 0 to High(Sources) do
      for var I := 0 to High(Sources[F].Tensors) do
        if ((FScope = iqExperts) and IsExpert(Sources[F].Tensors[I].Name))
          or ((FScope = iqDense) and (Sources[F].Tensors[I].Ndim = 2)
          and ((Sources[F].Tensors[I].Name = 'lm_head.weight')
          or Sources[F].Tensors[I].Name.StartsWith('model.language_model.'))
          and Sources[F].Tensors[I].Name.EndsWith('.weight')
          and not Sources[F].Tensors[I].Name.EndsWith('.in_proj_a.weight')
          and not Sources[F].Tensors[I].Name.EndsWith('.in_proj_b.weight')) then
        begin
          var Tensor: PSafeTensor := @Sources[F].Tensors[I];
          if (Tensor.Ndim <> 2)
            or not (Tensor.DType in [StF32, StBF16])
            or (Tensor.Shape[0] < 1)
            or (Tensor.Shape[0] > High(Integer))
            or (Tensor.Shape[1] < 1)
            or (Tensor.Shape[1] > High(Integer)) then
            raise EArgumentException.Create('Invalid native INT4 source: ' + Tensor.Name);
          Tensors.Add(Tensor);
        end;
    if Tensors.Count = 0 then
      raise EArgumentException.Create('No native weights selected for INT4 conversion');
    Metadata := TJSONObject.Create;
    Index.AddPair('metadata', Metadata);
    Metadata.AddPair('format', FFormat);
    Metadata.AddPair('source_fingerprint', FFingerprint);
    Map := TJSONObject.Create;
    Index.AddPair('weight_map', Map);
    PerShard := MATRICES_PER_SHARD;
    if FScope = iqDense then
      PerShard := 32;
    Shards := (Tensors.Count + PerShard - 1) div PerShard;
    if Assigned(Progress) then
      Progress(0, Shards);
    for var Shard := 0 to Shards - 1 do
    begin
      Start := Shard * PerShard;
      Count := Min(PerShard, Tensors.Count - Start);
      if FScope = iqDense then
        Name := Format('weights-%5.5d-of-%5.5d.safetensors', [Shard + 1, Shards])
      else
        Name := Format('experts-%5.5d-of-%5.5d.safetensors', [Shard + 1, Shards]);
      CreateGUID(Id);
      Temporary := TPath.Combine(FDirectory, GUIDToString(Id) + '.part');
      Header := TJSONObject.Create;
      try
        Metadata := TJSONObject.Create;
        Header.AddPair('__metadata__', Metadata);
        Metadata.AddPair('format', FFormat);
        Metadata.AddPair('source_fingerprint', FFingerprint);
        Offset := 0;
        for var I := Start to Start + Count - 1 do
        begin
          var Tensor := Tensors[I];
          Bytes := Tensor.Shape[0] * Int4RowBytes(Tensor.Shape[1]);
          Item := TJSONObject.Create;
          Header.AddPair(Tensor.Name, Item);
          Item.AddPair('dtype', 'U8');
          Item.AddPair(
            'shape',
            TJSONArray.Create(TJSONNumber.Create(Tensor.Shape[0]), TJSONNumber.Create(Int4RowBytes(Tensor.Shape[1])))
          );
          Item.AddPair(
            'data_offsets',
            TJSONArray.Create(TJSONNumber.Create(Offset), TJSONNumber.Create(Offset + Bytes))
          );
          Inc(Offset, Bytes);
          Map.AddPair(Tensor.Name, Name);
        end;
        HeaderText := UTF8String(Header.ToJSON);
        while Length(HeaderText) mod 8 <> 0 do
          HeaderText := HeaderText + ' ';
        HeaderBytes := Length(HeaderText);
        Stream := TFileStream.Create(Temporary, fmCreate);
        try
          Stream.WriteBuffer(HeaderBytes, 8);
          Stream.WriteBuffer(HeaderText[1], Length(HeaderText));
          for var I := Start to Start + Count - 1 do
          begin
            var Tensor := Tensors[I];
            Source := Default(TMemoryBlock);
            Source.Name := Tensor.Name;
            Source.Shape := Tensor.Shape;
            Source.Size := Tensor.Count;
            Source.DataType := Tensor.DType.ToDataType;
            Source.DataPtr := Tensor.Data;
            Encoded := QuantizeInt4(Source);
            try
              Stream.WriteBuffer(PByte(Encoded)^, Encoded.Size);
            finally
              Encoded.Free;
            end;
          end;
        finally
          Stream.Free;
        end;
        // A failed conversion cannot publish an index or a partial shard.
        if TFile.Exists(TPath.Combine(FDirectory, Name)) then
          TFile.Delete(TPath.Combine(FDirectory, Name));
        TFile.Move(Temporary, TPath.Combine(FDirectory, Name));
      finally
        Header.Free;
        if TFile.Exists(Temporary) then
          TFile.Delete(Temporary);
      end;
      if Assigned(Progress) then
        Progress(Shard + 1, Shards);
    end;
    if FingerprintForFormat(Sources, FFormat) <> FFingerprint then
      raise EInvalidOp.Create('Source weights changed during INT4 conversion');
    Temporary := TPath.Combine(FDirectory, FIndexName + '.part');
    TFile.WriteAllText(Temporary, Index.ToJSON, TEncoding.UTF8);
    TFile.Move(Temporary, TPath.Combine(FDirectory, FIndexName));
  finally
    Index.Free;
    Tensors.Free;
  end;
end;

function TLLMInt4Cache.Weight(const Name: string; const Rows, Columns: Integer): TMemoryBlock;
var
  Tensor: PSafeTensor;
begin
  Tensor := FFiles.GetTensor(Name);
  if (Tensor = nil)
    or (Tensor.DType <> StU8)
    or (Tensor.Ndim <> 2)
    or (Tensor.Shape[0] <> Rows)
    or (Tensor.Shape[1] <> Int4RowBytes(Columns)) then
    raise ESafeTensorError.Create('Missing or invalid packed INT4 weight: ' + Name);
  Result := Default(TMemoryBlock);
  Result.Name := Name;
  Result.DataType := dtU8;
  Result.Shape := Tensor.Shape;
  Result.Size := Tensor.Count;
  Result.DataPtr := Tensor.Data;
  Result.WeightIdentity := Tensor.WeightIdentity;
end;

function TLLMInt4Cache.MappedBytes: UInt64;
begin
  Result := 0;
  for var FileInfo in FFiles do
    Inc(Result, UInt64(FileInfo.DataSize));
end;

end.

