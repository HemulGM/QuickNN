unit QuickNN.ModelFiles;

interface

uses
  System.SysUtils, System.JSON;

type
  // Optional locations override only the corresponding component. Empty paths
  // retain the pipeline's standard directory layout.
  TQNNModelFiles = record
    TextEncoder, TextEncoder2, Tokenizer, Tokenizer2, VAE: string;
    function ToJSON: TJSONObject;
    class function FromJSON(const Value: TJSONValue): TQNNModelFiles; static;
  end;

function ReadModelTensorHeader(const FileName: string): TJSONObject;

function ModelComponentFile(const OverridePath, DefaultFile: string): string;
function ModelComponentDirectory(const OverridePath, DefaultDirectory: string): string;

implementation

uses
  System.IOUtils, System.Classes, System.Generics.Collections;

function ReadModelTensorHeader(const FileName: string): TJSONObject;
var
  Stream: TFileStream;
  Size: UInt64;
  Bytes: TBytes;
begin
  // Inspect metadata only, without mapping or reading tensor payloads or GPU use.
  Stream := TFileStream.Create(FileName, fmOpenRead or fmShareDenyNone);
  try
    if Stream.Size < 8 then raise EArgumentException.Create('Truncated safetensors header');
    Stream.ReadBuffer(Size, 8);
    if (Size < 2) or (Size > 64 * 1024 * 1024) or (Size > UInt64(Stream.Size - 8)) then
      raise EArgumentException.Create('Повреждён заголовок safetensors.');
    SetLength(Bytes, Size); Stream.ReadBuffer(Bytes[0], Length(Bytes));
    var JSON := TJSONValue.ParseJSONValue(TEncoding.UTF8.GetString(Bytes));
    if not (JSON is TJSONObject) then
    begin JSON.Free; raise EArgumentException.Create('Некорректный заголовок safetensors.') end;
    try
      for var Pair in TJSONObject(JSON) do
      begin
        if Pair.JsonString.Value = '__metadata__' then Continue;
        var Offsets := Pair.JsonValue.FindValue('data_offsets');
        var StartOffset, EndOffset: Int64;
        if not (Offsets is TJSONArray) or (TJSONArray(Offsets).Count <> 2)
          or not TryStrToInt64(TJSONArray(Offsets).Items[0].Value, StartOffset)
          or not TryStrToInt64(TJSONArray(Offsets).Items[1].Value, EndOffset)
          or (StartOffset < 0) or (EndOffset < StartOffset)
          or (UInt64(EndOffset) > UInt64(Stream.Size) - Size - 8) then
          raise EArgumentException.Create('Truncated or invalid tensor payload: ' + Pair.JsonString.Value);
      end;
    except JSON.Free; raise end;
    Result := TJSONObject(JSON);
  finally Stream.Free end;
end;


function ModelComponentFile(const OverridePath, DefaultFile: string): string;
begin
  Result := OverridePath;
  if Result = '' then Exit(DefaultFile);
  if TDirectory.Exists(Result) then Result := TPath.Combine(Result, ExtractFileName(DefaultFile.Replace('/', PathDelim)));
end;

function ModelComponentDirectory(const OverridePath, DefaultDirectory: string): string;
begin
  Result := OverridePath;
  if Result = '' then Exit(DefaultDirectory);
  if TFile.Exists(Result) then Result := ExtractFileDir(Result);
end;

function TQNNModelFiles.ToJSON: TJSONObject;
begin
  Result := TJSONObject.Create;
  Result.AddPair('text_encoder', TextEncoder);
  Result.AddPair('text_encoder_2', TextEncoder2);
  Result.AddPair('tokenizer', Tokenizer);
  Result.AddPair('tokenizer_2', Tokenizer2);
  Result.AddPair('vae', VAE);
end;

class function TQNNModelFiles.FromJSON(const Value: TJSONValue): TQNNModelFiles;
begin
  Result := Default(TQNNModelFiles);
  if not (Value is TJSONObject) then Exit;
  Result.TextEncoder := Value.GetValue<string>('text_encoder', '');
  Result.TextEncoder2 := Value.GetValue<string>('text_encoder_2', '');
  Result.Tokenizer := Value.GetValue<string>('tokenizer', '');
  Result.Tokenizer2 := Value.GetValue<string>('tokenizer_2', '');
  Result.VAE := Value.GetValue<string>('vae', '');
end;

end.
