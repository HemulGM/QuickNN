unit QuickNN.Models.Resident;

{$POINTERMATH ON}

interface

uses
  QuickNN.Common,
  QuickNN.Vulkan,
  QuickNN.Models.Qwen3,
  QuickNN.Models.Transformers,
  QuickNN.Models.VAE;

type
  TGPUImageRef = record
    Latent: TQNNGPUTensor;
    H, W, TOffset: Integer;
  end;
  TGPUImageRefs = TArray<TGPUImageRef>;

function Qwen3ForwardGPU(
    const Ctx: TQNNGPUContext;
    var Model: TQwen3Model;
    const Ids, Mask: TArray<LongInt>;
    const Rows: Integer
): TQNNGPUTensor;
function Qwen3EncodeGPU(
    const Ctx: TQNNGPUContext;
    var Encoder: TQwen3Encoder;
    const Prompt: RawByteString;
    out RealRows: LongInt
): TQNNGPUTensor;
function FluxForwardGPU(
    const Ctx: TQNNGPUContext;
    var Model: TTransformerFlux;
    const Latent: TQNNGPUTensor;
    const Height, Width: Integer;
    const Refs: TGPUImageRefs;
    const Text: TQNNGPUTensor;
    const TextRows: Integer;
    const Time: Single
): TQNNGPUTensor;
function ZImageForwardGPU(
    const Ctx: TQNNGPUContext;
    var Model: TTransformerZI;
    const Latent: TQNNGPUTensor;
    const Height, Width: Integer;
    const Text: TQNNGPUTensor;
    const TextRows: Integer;
    const Time: Single
): TQNNGPUTensor;
function VAEEncodeGPU(
    const Ctx: TQNNGPUContext;
    const Model: TVAE;
    const Pixels: TQNNGPUTensor;
    const Height, Width: Integer
): TQNNGPUTensor;
function VAEDecodeGPU(
    const Ctx: TQNNGPUContext;
    const Model: TVAE;
    const Latent: TQNNGPUTensor;
    const Height, Width: Integer
): TQNNGPUTensor;

function FluxSampleGPU(
    const Ctx: TQNNGPUContext;
    var Model: TTransformerFlux;
    const Noise: TQNNGPUTensor;
    const Height, Width: Integer;
    const Refs: TGPUImageRefs;
    const Positive, Negative: TQNNGPUTensor;
    const PositiveRows, NegativeRows: Integer;
    const Guidance: Single;
    const Schedule: TMemoryBlock;
    const Steps: Integer;
    const Progress: TProgressCallback = nil
): TQNNGPUTensor;
function ZImageSampleGPU(
    const Ctx: TQNNGPUContext;
    var Model: TTransformerZI;
    const Noise: TQNNGPUTensor;
    const Height, Width: Integer;
    const Text: TQNNGPUTensor;
    const TextRows: Integer;
    const Schedule: TMemoryBlock;
    const Steps: Integer
): TQNNGPUTensor;
function UploadImageGPU(const Ctx: TQNNGPUContext; const Image: TQNNImage): TQNNGPUTensor;

implementation

uses
  System.SysUtils,
  System.Math,
  QuickNN.Vulkan.Operations;

{$I QuickNN.Models.Resident.Text.inc}
{$I QuickNN.Models.Resident.Flux.inc}
{$I QuickNN.Models.Resident.ZImage.inc}
{$I QuickNN.Models.Resident.VAE.inc}
{$I QuickNN.Models.Resident.Sampling.inc}

end.
