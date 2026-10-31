unit QuickNN.Models.LLM.Vision;

interface

uses
  System.SysUtils, QuickNN.Models.LLM.Types;

type
  TLLMVisionPhase = reference to procedure(const Text: string);

// A short-lived GPU encoder. Its context is released before loading the decoder.
procedure LLMPrepareImage(const ModelDirectory: string; var Image: TLLMImage; const BeforeGPU: TProc; const OnPhase: TLLMVisionPhase; const Cancelled: TFunc<Boolean>);

function LLMVisionAvailable(const ModelDirectory: string): Boolean;

implementation

uses
  Winapi.Windows, System.Classes, System.JSON, System.IOUtils, System.Hash,
  QuickNN.Common, QuickNN.Vulkan, QuickNN.Models.Qwen21.Vision,
  QuickNN.Models.Qwen21.Conditioning;

function LLMVisionAvailable(const ModelDirectory: string): Boolean;
var
  Config: TJSONValue;
begin
  Config := nil;
  try
    Config := TJSONValue.ParseJSONValue(TFile.ReadAllText(TPath.Combine(ModelDirectory, 'config.json')));
    Result := (Config <> nil) and (Config.GetValue<string>('model_type', '') = 'qwen3_5')
      and (Config.FindValue('vision_config') <> nil);
  except
    Result := False;
  end;
  Config.Free;
end;

procedure LLMPrepareImage(const ModelDirectory: string; var Image: TLLMImage; const BeforeGPU: TProc; const OnPhase: TLLMVisionPhase; const Cancelled: TFunc<Boolean>);
const
  CACHE_MAGIC: UInt64 = $31564E4E4B434955;
var
  Config: TJSONValue;
  Encoder: TQwen21Vision;
  Ctx: TQNNGPUContext;
  Input: TQwen21ImageCondition;
  resident: TQwen21GPUImageCondition;
  Stream: TFileStream;
  CacheFile, Fingerprint: string;
  Magic, PreviousLimit: UInt64;
  Count, Height, Width: Integer;

  procedure CheckCancelled;
  begin
    if Assigned(Cancelled) and Cancelled() then
      raise ELLMCancelled.Create('Vision cancelled');
  end;

begin
  CheckCancelled;
  if not LLMVisionAvailable(ModelDirectory) then
    raise ENotSupportedException.Create('Select a supported Qwen hybrid vision checkpoint');
  if not TFile.Exists(Image.FileName) then
    raise EFileNotFoundException.Create('Image not found: ' + Image.FileName);
  Stream := TFileStream.Create(Image.FileName, fmOpenRead or fmShareDenyWrite);
  try
    Image.Identity := THashSHA2.GetHashString(Stream)
  finally
    Stream.Free
  end;
  Fingerprint := ExpandFileName(ModelDirectory) + TFile.ReadAllText(TPath.Combine(ModelDirectory, 'config.json'));
  for var FileName in TDirectory.GetFiles(ModelDirectory, '*.safetensors') do
    Fingerprint := Fingerprint + ExtractFileName(FileName) + IntToStr(TFile.GetSize(FileName))
      + DateTimeToStr(TFile.GetLastWriteTimeUtc(FileName), TFormatSettings.Invariant);
  Image.EncoderKey := THashSHA2.GetHashString('quicknn-qwen35-vision-v3-rgb-white/512/' + Fingerprint);
  CacheFile := TPath.Combine(ExtractFileDir(Image.FileName), THashSHA2.GetHashString(Image.Identity + '/' + Image.EncoderKey) + '.vision');
  if TFile.Exists(CacheFile) then
  begin
    Stream := TFileStream.Create(CacheFile, fmOpenRead or fmShareDenyWrite);
    try
      if Stream.Size >= 20 then
      begin
        Stream.ReadBuffer(Magic, 8);
        Stream.ReadBuffer(Height, 4);
        Stream.ReadBuffer(Width, 4);
        Stream.ReadBuffer(Count, 4);
        if (Magic = CACHE_MAGIC) and (Height >= 2) and (Width >= 2) and (Height <= 32) and (Width <= 32)
          and (Count = Height div 2 * (Width div 2) * 5120) and (Stream.Size = 20 + Int64(Count) * 4 + 64) then
        begin
          var SourceHash: TBytes;
          SetLength(SourceHash, 64);
          Stream.ReadBuffer(SourceHash[0], 64);
          if TEncoding.ASCII.GetString(SourceHash) = Image.Identity then
          begin
            Image.GridHeight := Height;
            Image.GridWidth := Width;
            SetLength(Image.Features, Count);
            Stream.ReadBuffer(Image.Features[0], Count * 4);
            Exit;
          end;
        end;
      end;
    finally
      Stream.Free
    end;
  end;
  Config := TJSONValue.ParseJSONValue(TFile.ReadAllText(TPath.Combine(ModelDirectory, 'config.json')));
  try
    var Vision := Config.FindValue('vision_config');
    if (Vision.GetValue<Integer>('depth') <> 27) or (Vision.GetValue<Integer>('hidden_size') <> 1152)
      or (Vision.GetValue<Integer>('out_hidden_size') <> 5120) or (Vision.GetValue<Integer>('patch_size') <> 16)
      or (Vision.GetValue<Integer>('num_heads') <> 16) or (Vision.GetValue<Integer>('spatial_merge_size') <> 2)
      or (Vision.GetValue<Integer>('temporal_patch_size') <> 2) or (Vision.GetValue<Integer>('intermediate_size') <> 4304)
      or (Vision.GetValue<string>('hidden_act') <> 'gelu_pytorch_tanh') then
      raise ENotSupportedException.Create('Unsupported vision geometry');
  finally
    Config.Free
  end;
  if Assigned(BeforeGPU) then
    BeforeGPU();
  if Assigned(OnPhase) then
    OnPhase('Кодирование изображения на GPU…');
  Encoder := nil;
  Ctx := nil;
  Input := Default(TQwen21ImageCondition);
  resident := Default(TQwen21GPUImageCondition);
  PreviousLimit := QNNGetGPUStatus.WeightCacheLimit;
  try
    QNNConfigureGPU(gpuForce);
    QNNSetGPUWeightCacheLimit(UInt64(2) * 1024 * 1024 * 1024);
    Input.Pixels := Q21LoadReference(Image.FileName, 512);
    for var Pixel := 0 to Input.Pixels.Width * Input.Pixels.Height - 1 do
    begin
      var Alpha := Integer(Input.Pixels.Data[Pixel * 4 + 3]);
      if Alpha < 255 then
      begin
        for var Channel := 0 to 2 do
          Input.Pixels.Data[Pixel * 4 + Channel] := (Integer(Input.Pixels.Data[Pixel * 4 + Channel]) * Alpha + 255 * (255 - Alpha) + 127) div 255;
        Input.Pixels.Data[Pixel * 4 + 3] := 255;
      end;
    end;
    Ctx := TQNNGPUContext.Create;
    resident.Pixels := Ctx.UploadRaw(@Input.Pixels.Data[0], Input.Pixels.Width * Input.Pixels.Height);
    Encoder := TQwen21Vision.CreateHybrid(ModelDirectory,
      procedure(const Phase: string)
      begin
        if Assigned(Cancelled) and Cancelled() then
          raise ELLMCancelled.Create('Vision cancelled');
        if Assigned(OnPhase) then
          OnPhase(Phase);
      end, 5120);
    Encoder.EncodeGPU(Ctx, Input, resident);
    CheckCancelled;
    Image.GridHeight := Input.GridHeight;
    Image.GridWidth := Input.GridWidth;
    SetLength(Image.Features, resident.Visual.Count);
    Ctx.Download(resident.Visual, @Image.Features[0]);
  finally
    if Ctx <> nil then
      resident.Free(Ctx);
    Ctx.Free;
    Encoder.Free;
    Input.Free;
    QNNGPUClearWeightCache;
    QNNSetGPUWeightCacheLimit(PreviousLimit);
  end;
  CheckCancelled;
  var Temporary := CacheFile + '.tmp';
  try
    Stream := TFileStream.Create(Temporary, fmCreate);
    try
      Magic := CACHE_MAGIC;
      Count := Length(Image.Features);
      Stream.WriteBuffer(Magic, 8);
      Stream.WriteBuffer(Image.GridHeight, 4);
      Stream.WriteBuffer(Image.GridWidth, 4);
      Stream.WriteBuffer(Count, 4);
      var HashBytes := TEncoding.ASCII.GetBytes(Image.Identity);
      Stream.WriteBuffer(HashBytes[0], 64);
      Stream.WriteBuffer(Image.Features[0], Count * 4);
    finally
      Stream.Free
    end;
    if not MoveFileEx(PChar(Temporary), PChar(CacheFile), MOVEFILE_REPLACE_EXISTING or MOVEFILE_WRITE_THROUGH) then
      RaiseLastOSError;
  finally
    if TFile.Exists(Temporary) then
      TFile.Delete(Temporary);
  end;
end;

end.

