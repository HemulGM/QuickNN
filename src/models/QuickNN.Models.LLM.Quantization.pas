unit QuickNN.Models.LLM.Quantization;

{$POINTERMATH ON}

interface

uses
  System.SysUtils,
  QuickNN.Common,
  QuickNN.SafeTensors;

type
  TMoEInt4Cache = class
  private
    FFiles: TSafeTensorFiles;
    FDirectory, FFingerprint, FFormat: string;
    procedure Build(const Sources: TSafeTensorFiles; const Progress: TProc<Integer, Integer>);
    procedure ValidateIndex;
  public
    constructor Create(
        const Sources: TSafeTensorFiles;
        const Directory: string;
        const Progress: TProc<Integer, Integer> = nil
    );
    destructor Destroy; override;
    function Weight(const Name: string; const Rows, Columns: Integer): TMemoryBlock;
    function MappedBytes: UInt64;
    property Directory: string read FDirectory;
    class function SourceFingerprint(const Sources: TSafeTensorFiles): string; static;
  end;

implementation

uses
  System.Classes,
  System.IOUtils,
  System.JSON,
  System.Hash,
  System.Math,
  System.Generics.Collections,
  QuickNN.Quantization.Int4;

const
  CACHE_FORMAT = 'quicknn-moe-int4-g32-ls-v1';
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

class function TMoEInt4Cache.SourceFingerprint(const Sources: TSafeTensorFiles): string;
begin
  Result := FingerprintForFormat(Sources, CACHE_FORMAT);
end;

constructor TMoEInt4Cache.Create(
    const Sources: TSafeTensorFiles;
    const Directory: string;
    const Progress: TProc<Integer, Integer>
);
var
  Lock: TFileStream;
  LegacyRoot, LegacyDirectory, LegacyFingerprint: string;
begin
  inherited Create;
  if Directory = '' then
    raise EArgumentException.Create('Missing INT4 cache directory');
  FFingerprint := SourceFingerprint(Sources);
  FDirectory := TPath.Combine(TPath.GetFullPath(Directory), CACHE_FORMAT + '-' + FFingerprint);
  FFormat := CACHE_FORMAT;
  if not TFile.Exists(TPath.Combine(FDirectory, CACHE_INDEX)) then
  begin
    // Cache headers and fingerprints are versioned on disk. Reuse a verified
    // old cache without renaming directories or requantizing all expert weights.
    LegacyRoot := ExcludeTrailingPathDelimiter(TPath.GetFullPath(Directory));
    if SameText(ExtractFileName(LegacyRoot), '.quicknn-cache') then
      LegacyRoot := TPath.Combine(ExtractFileDir(LegacyRoot), '.envision-cache');
    LegacyFingerprint := FingerprintForFormat(Sources, LEGACY_CACHE_FORMAT);
    LegacyDirectory := TPath.Combine(LegacyRoot, LEGACY_CACHE_FORMAT + '-' + LegacyFingerprint);
    if TFile.Exists(TPath.Combine(LegacyDirectory, CACHE_INDEX)) then
    begin
      FDirectory := LegacyDirectory;
      FFingerprint := LegacyFingerprint;
      FFormat := LEGACY_CACHE_FORMAT;
    end;
  end;
  if not TFile.Exists(TPath.Combine(FDirectory, CACHE_INDEX)) then
  begin
    TDirectory.CreateDirectory(FDirectory);
    // Only a committed index makes a cache visible. Exclusive sharing also
    // prevents concurrent builders from publishing the same shard names.
    Lock := TFileStream.Create(TPath.Combine(FDirectory, 'build.lock'), fmCreate or fmShareExclusive);
    try
      if not TFile.Exists(TPath.Combine(FDirectory, CACHE_INDEX)) then
        Build(Sources, Progress);
    finally
      Lock.Free;
    end;
  end;
  ValidateIndex;
  FFiles := LoadSafeTensorShards(FDirectory, CACHE_INDEX, '');
  for var FileInfo in FFiles do
    if (FileInfo.JSON.GetValue<string>('__metadata__.format', '') <> FFormat)
        or (FileInfo.JSON.GetValue<string>('__metadata__.source_fingerprint', '') <> FFingerprint) then
      raise ESafeTensorError.Create('Invalid INT4 shard format');
end;

destructor TMoEInt4Cache.Destroy;
begin
  for var I := 0 to High(FFiles) do
    FFiles[I].Close;
  inherited;
end;

procedure TMoEInt4Cache.ValidateIndex;
var
  Index: TJSONValue;
  Map: TJSONObject;
begin
  Index := TJSONValue.ParseJSONValue(TFile.ReadAllText(TPath.Combine(FDirectory, CACHE_INDEX), TEncoding.UTF8));
  try
    if not (Index is TJSONObject)
        or (Index.GetValue<string>('metadata.format', '') <> FFormat)
        or (Index.GetValue<string>('metadata.source_fingerprint', '') <> FFingerprint) then
      raise ESafeTensorError.Create('Invalid or stale MoE INT4 cache index');
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

procedure TMoEInt4Cache.Build(const Sources: TSafeTensorFiles; const Progress: TProc<Integer, Integer>);
var
  Tensors: TList<PSafeTensor>;
  Header, Index, Map, Metadata, Item: TJSONObject;
  Stream: TFileStream;
  Source, Encoded: TMemoryBlock;
  HeaderText: UTF8String;
  HeaderBytes: UInt64;
  Offset, Bytes: Int64;
  Name, Temporary: string;
  Shards, Start, Count: Integer;
  Id: TGUID;
begin
  Tensors := TList<PSafeTensor>.Create;
  Index := TJSONObject.Create;
  try
    for var F := 0 to High(Sources) do
      for var I := 0 to High(Sources[F].Tensors) do
        if IsExpert(Sources[F].Tensors[I].Name) then
        begin
          var Tensor: PSafeTensor := @Sources[F].Tensors[I];
          if (Tensor.Ndim <> 2)
              or not (Tensor.DType in [StF32, StBF16])
              or (Tensor.Shape[0] < 1)
              or (Tensor.Shape[0] > High(Integer))
              or (Tensor.Shape[1] < 1)
              or (Tensor.Shape[1] > High(Integer)) then
            raise EArgumentException.Create('Invalid native MoE expert: ' + Tensor.Name);
          Tensors.Add(Tensor);
        end;
    if Tensors.Count = 0 then
      raise EArgumentException.Create('No native MoE experts to quantize');
    Metadata := TJSONObject.Create;
    Index.AddPair('metadata', Metadata);
    Metadata.AddPair('format', CACHE_FORMAT);
    Metadata.AddPair('source_fingerprint', FFingerprint);
    Map := TJSONObject.Create;
    Index.AddPair('weight_map', Map);
    Shards := (Tensors.Count + MATRICES_PER_SHARD - 1) div MATRICES_PER_SHARD;
    if Assigned(Progress) then
      Progress(0, Shards);
    for var Shard := 0 to Shards - 1 do
    begin
      Start := Shard * MATRICES_PER_SHARD;
      Count := Min(MATRICES_PER_SHARD, Tensors.Count - Start);
      Name := Format('experts-%5.5d-of-%5.5d.safetensors', [Shard + 1, Shards]);
      CreateGUID(Id);
      Temporary := TPath.Combine(FDirectory, GUIDToString(Id) + '.part');
      Header := TJSONObject.Create;
      try
        Metadata := TJSONObject.Create;
        Header.AddPair('__metadata__', Metadata);
        Metadata.AddPair('format', CACHE_FORMAT);
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
    if SourceFingerprint(Sources) <> FFingerprint then
      raise EInvalidOp.Create('Source weights changed during INT4 conversion');
    Temporary := TPath.Combine(FDirectory, CACHE_INDEX + '.part');
    TFile.WriteAllText(Temporary, Index.ToJSON, TEncoding.UTF8);
    TFile.Move(Temporary, TPath.Combine(FDirectory, CACHE_INDEX));
  finally
    Index.Free;
    Tensors.Free;
  end;
end;

function TMoEInt4Cache.Weight(const Name: string; const Rows, Columns: Integer): TMemoryBlock;
var
  Tensor: PSafeTensor;
begin
  Tensor := FFiles.GetTensor(Name);
  if (Tensor = nil)
      or (Tensor.DType <> StU8)
      or (Tensor.Ndim <> 2)
      or (Tensor.Shape[0] <> Rows)
      or (Tensor.Shape[1] <> Int4RowBytes(Columns)) then
    raise ESafeTensorError.Create('Missing or invalid packed INT4 expert: ' + Name);
  Result := Default(TMemoryBlock);
  Result.Name := Name;
  Result.DataType := dtU8;
  Result.Shape := Tensor.Shape;
  Result.Size := Tensor.Count;
  Result.DataPtr := Tensor.Data;
  Result.WeightIdentity := Tensor.WeightIdentity;
end;

function TMoEInt4Cache.MappedBytes: UInt64;
begin
  Result := 0;
  for var FileInfo in FFiles do
    Inc(Result, UInt64(FileInfo.DataSize));
end;

end.
