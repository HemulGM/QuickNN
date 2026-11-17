# QuickNN

**Local image generation and language model inference in Delphi / Object Pascal.**

QuickNN provides tensor operations, safetensors loading, CPU kernels, and a Vulkan compute backend for running neural networks directly in Delphi applications. The repository includes image-generation pipelines, an autoregressive LLM engine, console examples, and a FireMonkey chat application with [image generation and visual context](docs/image-chat.md).

## Features

- **Native tensor and model code:** memory-mapped safetensors and supported GGUF checkpoints, sharded checkpoints, packed quantized weights, tokenization, sampling, and LoRA support.
- **Vulkan acceleration:** matrix multiplication, attention, normalization, convolutions, activations, and other operations used by the model pipelines. Compute shaders are embedded in the executable.
- **Resident GPU execution:** weights and intermediate tensors stay in VRAM across operation chains, with bounded caches, a driver-aware activation budget and [tiled VAE decoding](docs/gpu-vae-memory.md).
- **Image generation:** Qwen-Image 2.1, FLUX.1-dev, FLUX.2 Klein, and Z-Image Turbo pipelines, including an INT8 ConvRot path for Qwen-Image 2.1.
- **LLM generation:** Qwen3, Qwen3 MoE, Qwen3.8-27B, Qwen3.8 Flash Next and Gemma 2 text generation with Gated DeltaNet, streaming Unicode output, a GPU KV cache, cancellation, and configurable sampling.
- **Weight memory management:** batched expert/layer loading, RAM/VRAM weight caches, persistent INT4 caches, and partial dense-model offload built from original checkpoints.
- **Local chat UI:** user image attachments via file picker, drag/drop and clipboard, persistent vision context, virtualized message history, multiple chats, a model catalog, per-chat settings, Markdown for both user and assistant messages with syntax highlighting and code-copy buttons, attachment thumbnail galleries above messages, collapsible reasoning, response continuation, automatic history compression, and saved reading positions.

## Model support

| Family | Supported input and execution |
| --- | --- |
| **Qwen3** | Dense safetensors checkpoints, including Qwen3-1.7B and Qwen3-4B; GPU prefill/decode and a resident KV cache. |
| **Gemma 2** | Dense GGUF with an embedded SentencePiece BPE tokenizer, alternating local/global attention, attention/logit softcapping, GPU sampling and a resident KV cache. Tested with Dirty Shirley Writer v2 9B Q2_K (mixed Q2_K/Q3_K/Q6_K). [Guide](docs/llm.md#gemma-2-gguf). |
| **Qwen3 MoE** | `qwen3_moe`, including Qwen3-Coder-30B-A3B-Instruct; routed expert execution with RAM/VRAM caching and optional INT4 expert weights. |
| **Qwen3.8-27B** | Text backbones from `qwen3_5` safetensors or dense `qwen35` GGUF; Gated DeltaNet, persistent CPU/GPU recurrent and KV state, packed quantized projections and CPU execution for RAM-resident layers. [Guide](docs/qwen3.8.md). |
| **Qwen3.8 Flash Next** | Split `qwen4exp` GGUF with hyper-connections, Gated DeltaNet, QSA, PLE n-gram embeddings and 512 routed experts. GPU compute with bounded expert uploads and mmap for the PLE table; tested on a 16 GiB RTX 5080. [Guide](docs/qwen-flash-next.md). |
| **Qwen-Image 2.1** | BF16 and INT8 ConvRot safetensors, plus BF16/F16/F32, Q8_0 and Q4_K_M GGUF diffusion transformers; resident GPU generation, reference images, and LoRA adapters. |
| **FLUX.1-dev** | Native BFL BF16 safetensors transformer, CLIP-L byte BPE and T5-XXL Unigram encoders, distilled guidance, bounded GPU weight cache, GPU VAE and additive native/Kohya LoRA. [Guide](docs/flux1-dev.md). |
| **FLUX.2 Klein** | 4B and 9B presets using complete Diffusers-style model directories; resident text encoder, transformer, sampler, and VAE paths. Klein **4B** also supports Diffusers-layout LoRA on GPU with per-chat adapter selection and strength. [Guide](docs/image-chat.md#flux2-klein-4b-lora). |
| **Z-Image Turbo** | Complete Diffusers-style model directories and a resident GPU pipeline. |

Checkpoint compatibility depends on the model architecture, configuration, and tensor layout. Real-checkpoint generation has been exercised with the Qwen models, FLUX.1-dev (including LoRA), and FLUX.2 Klein 4B variants. Z-Image's resident path currently has synthetic parity coverage; see the model guides for validation details.

LLM input supports original BF16/F32 safetensors, dense `qwen35` / `gemma2` GGUF and split `qwen4exp` GGUF checkpoints. QuickNN can generate its own packed INT4 caches from safetensors; GGUF weights preserve their original mixed precision. GPTQ, AWQ and other GGUF LLM architectures are not supported. Qwen-Image 2.1 also accepts supported GGUF diffusion transformers, including Q8_0 and mixed Q4_K/Q6_K (Q4_K_M). Model weights are obtained separately. Tokenizer/configuration files are stored beside the Qwen-Image weights in the model directory.

GPU decode uses packed GGUF vector kernels and fused gate/up/SwiGLU projections for supported GGUF, native INT4 and BF16/F32 weights. Hybrid offload batches a complete layer when possible; weights stay compressed during transfers. See the [LLM performance measurements](docs/llm-performance.md) for the tested hardware, limits and reproduction steps. The hybrid backend runs RAM-resident layers on CPU during decode; [CPU offload measurements and settings](docs/llm-cpu-offload.md) describe the placement and state transitions. Its alternative GPU streaming mode supports [asynchronous layer prefetch](docs/llm-async-prefetch.md) with double buffering inside the configured GPU weight budget. CPU decode additionally supports [budgeted SIMD weight preparation and direct matvec kernels](docs/llm-cpu-matvec.md), including partial preparation for smaller RAM caches. A [comparison with llama.cpp CUDA](docs/llama-cuda-comparison.md) measures the same GGUF checkpoint and input tokens on both engines.

## Getting started

### Requirements

- **Windows and Delphi / RAD Studio 13.** The examples are Delphi projects; Free Pascal and Lazarus are not supported.
- **Win64 and a compatible Vulkan compute device/driver** for full GPU generation. Win32 builds support CPU/component checks and the chat interface; LLM generation is disabled in the Win32 UI.
- **Model checkpoints and enough RAM/VRAM** for the selected model, context, and cache budgets. MoE can stream expert weights from RAM. The Qwen3.8 dense hybrid backend executes its RAM-resident prefix on CPU during decode and keeps its GPU suffix resident.

Initialize the Markdown renderer submodule from the repository root:

```console
git submodule update --init --recursive
```

Markdown4D is compiled from source; installing its components into the IDE is unnecessary. Standard builds use embedded shaders.

### Build and run the chat application

Open [QuickNN.groupproj](QuickNN.groupproj) for the main examples, or open [LLMUIChat.dproj](Examples/LLMUIChat/LLMUIChat.dproj) directly. Select **Win64 / Release** and run the project.

In the application:

1. Open the unified model catalog and start **Добавить модель**. Select a primary GGUF/safetensors file, shard index or complete model directory. The wizard determines the supported architecture and roles from metadata/configuration/tensor names; it fills an empty display name automatically and preserves names you have entered.
2. Review the discovered tokenizer, text encoders, VAE and other component paths. You can replace their locations and register compatible LoRA adapters for image models. Missing components and unsupported primary files are reported before saving.
3. Create a chat and expand its **Переписка**, **Анализ изображений** or **Генерация изображений** section. Each picker lists only compatible available models. Image adapters and generation parameters are selected per chat; section expansion is saved per chat too.
4. Press **Enter** to send and **Shift+Enter** to insert a newline. Use the stop button to cancel generation. In image mode, the **Негатив** button switches to a separate negative prompt for compatible generators; it requires guidance above 1. Both prompt texts are saved per chat, and incompatible model changes ask before clearing the negative text.

Image mode saves a PNG and its execution settings without automatically asking an LLM to analyze it. In this mode, attached images go directly to reference-capable generators (FLUX.2 Klein and Qwen-Image 2.1), using their own encoders. Generators without direct reference support first use the vision model selected in the chat to describe attachments, then generate from that description and the original prompt. To request LLM analysis, attach an image in message mode or ask a separate question. Vision has its own model, context, response limit, temperature and extra instruction; image analysis does not change the model selected for ordinary conversation. Existing model IDs, history and chat settings are retained.

See the [model wizard and role guide](docs/model-catalog.md), [image chat guide](docs/image-chat.md), [FLUX.1-dev guide](docs/flux1-dev.md) and [GGUF comparison](docs/gguf-quantization.md).


Chats, drafts, settings, and reading positions are saved to `%LOCALAPPDATA%\QuickNN\LLMUIChat\state.json`, with a backup copy. Automatic context compression keeps the full original conversation on disk. Existing history from the previous application name is imported automatically.

See the [model and chat settings guide](docs/model-catalog.md) for continuation, context compression, and persistence details.

### Console examples

| Project | Purpose | Configuration |
| --- | --- | --- |
| [QuickNNDemo](Examples/ConsoleDemo/QuickNNDemo.dproj) | Generate an image with Qwen-Image 2.1, FLUX.1-dev, FLUX.2 Klein, or Z-Image Turbo. | Edit `DemoSettings` in [QuickNN.Demo.Settings.pas](Examples/ConsoleDemo/QuickNN.Demo.Settings.pas); no launch arguments. |
| [QuickNNChat](Examples/LLMChat/QuickNNChat.dproj) | Streaming Qwen3 / Qwen3 MoE console chat. | Edit `ChatSettings` in [QuickNN.LLMChat.Settings.pas](Examples/LLMChat/QuickNN.LLMChat.Settings.pas); no launch arguments. |
| [QwenImage21](Examples/Delphi/QwenImage21.dproj) | Dedicated Qwen-Image 2.1 command-line pipeline. | Run with `--help`. |
| [QuickNN](Examples/Delphi/QuickNN.dproj) | Lower-level diffusion console example. | Configure the model paths and generation settings in the program source. |

Some presets use `H:\Models` as a local example path. Set the paths to your own checkpoints before running. The image demo writes PNG files under `output` by default; building a project does not start generation.

## Source layout

| Directory | Contents |
| --- | --- |
| `src/core` | Tensors, kernels, quantization, LoRA, and sampling. |
| `src/backends/cpu` | Native CPU operations and SIMD kernels. |
| `src/backends/vulkan` | Vulkan bindings, GPU execution, shaders, and embedded SPIR-V. |
| `src/io` | Safetensors/GGUF loading, quantization formats and shared tokenization. |
| `src/models` | Diffusion and LLM implementations; Qwen-Image components are under `qwen21`. |
| `src/console` | Optional ANSI/Sixel output, separate from the inference core. |
| `Examples` | FMX chat and console applications. |
| `tests` | CPU/GPU, model, quantization, and chat regression checks. |
| `tools` | Build/cleanup scripts, checkpoint diagnostics, and benchmarks. |
| `Markdown4D` | Markdown renderer submodule used by the chat UI. |

## License

QuickNN is distributed under the [MIT License](LICENSE). Model weights, tokenizer assets, adapters, and third-party source dependencies retain their respective licenses.

### Qwen3.8 GGUF LLM checkpoints

The native LLM engine and LLMUIChat load dense `qwen35` GGUF checkpoints,
including Unsloth `Qwen3.8-27B-UD-Q4_K_XL`, without an inference DLL.
In the LLM catalog, enter the directory containing one language `.gguf` and leave
the tokenizer directory empty. Architecture, BPE vocabulary/merges and stop tokens
come from GGUF; external `config.json` and `tokenizer.json` are not required.
The catalog preserves the file's mixed precision and disables reconversion.

Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ3_S, IQ4_NL and IQ4_XS weights stay compressed
in mmap, the bounded RAM cache and VRAM. Prefill uses GPU kernels; single-token
decode executes a CPU prefix followed by a resident GPU suffix. Each side retains
its KV/DeltaNet state; state copies synchronize only when execution changes device.
Weights are not re-uploaded on each decoded token. MTP tensors are
excluded from the ordinary decoder. On the 16 GiB RTX 5080, the tested budgets
are 11 GiB for GPU weights and 8 GiB for the RAM cache. Budgets are configurable
per chat. `Examples/LLMChat/QuickNN.LLMChat.Settings.pas` also contains a GGUF
preset, selected by default for the console demo.

This GGUF holds text weights. The current GGUF LLM loader does not load `mmproj`
vision encoders; this catalog entry is correctly marked as text-only. The existing
safetensors vision path is unchanged.

Verification: `QuickNNTests.exe -gguf-only` validates all packed formats against
independent GGML vectors on CPU and GPU; `-gguf-llm-only` uses a standalone tiny
GGUF with reference logits and continuation checks; `-gguf-llm-real` tests the
model on H:. `LLMUIChatChecks.exe -gguf-language-chat` covers catalog creation,
persistence, precision settings and actual GPU chat generation.
