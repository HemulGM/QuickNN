# QuickNN

**Local image generation and language model inference in Delphi / Object Pascal.**

QuickNN provides tensor operations, safetensors loading, CPU kernels, and a Vulkan compute backend for running neural networks directly in Delphi applications. The repository includes image-generation pipelines, an autoregressive LLM engine, console examples, and a FireMonkey chat application.

## Features

- **Native tensor and model code:** memory-mapped safetensors, sharded checkpoints, BF16/F32 weights, tokenization, sampling, and LoRA support.
- **Vulkan acceleration:** matrix multiplication, attention, normalization, convolutions, activations, and other operations used by the model pipelines. Compute shaders are embedded in the executable.
- **Resident GPU execution:** weights and intermediate tensors stay in VRAM across operation chains, with bounded caches and reusable temporary buffers.
- **Image generation:** Qwen-Image 2.1, FLUX.2 Klein, and Z-Image Turbo pipelines, including an INT8 ConvRot path for Qwen-Image 2.1.
- **LLM generation:** Qwen3 and Qwen3 MoE, streaming Unicode output, a GPU KV cache, cancellation, and configurable sampling.
- **MoE memory management:** batched expert loading, RAM/VRAM weight caches, and a persistent INT4 expert cache built from the original checkpoint.
- **Local chat UI:** multiple chats, a model catalog, per-chat settings, Markdown with syntax highlighting and code-copy buttons, collapsible reasoning, response continuation, automatic history compression, and saved reading positions.

## Model support

| Family | Supported input and execution |
| --- | --- |
| **Qwen3** | Dense safetensors checkpoints, including Qwen3-1.7B and Qwen3-4B; GPU prefill/decode and a resident KV cache. |
| **Qwen3 MoE** | `qwen3_moe`, including Qwen3-Coder-30B-A3B-Instruct; routed expert execution with RAM/VRAM caching and optional INT4 expert weights. |
| **Qwen-Image 2.1** | BF16 and INT8 ConvRot checkpoints; resident GPU generation, reference images, and LoRA adapters. |
| **FLUX.2 Klein** | 4B and 9B presets using complete Diffusers-style model directories; resident text encoder, transformer, sampler, and VAE paths. |
| **Z-Image Turbo** | Complete Diffusers-style model directories and a resident GPU pipeline. |

Checkpoint compatibility depends on the model architecture, configuration, and tensor layout. Real-checkpoint generation has been exercised with the Qwen models and FLUX.2 Klein 4B variants. Z-Image's resident path currently has synthetic parity coverage; see the model guides for validation details.

LLM input uses original BF16/F32 safetensors checkpoints. QuickNN generates its own packed INT4 expert cache; GGUF, GPTQ, and AWQ checkpoints are not supported input formats. Model weights are obtained separately. The repository includes tokenizer/configuration assets for the Qwen-Image pipeline.

## Getting started

### Requirements

- **Windows and Delphi / RAD Studio 13.** The examples are Delphi projects; Free Pascal and Lazarus are not supported.
- **Win64 and a compatible Vulkan compute device/driver** for full GPU generation. Win32 builds support CPU/component checks and the chat interface; LLM generation is disabled in the Win32 UI.
- **Model checkpoints and enough RAM/VRAM** for the selected model, context, and cache budgets. MoE can keep part of its expert weights in RAM and stream them to the GPU.

Initialize the Markdown renderer submodule from the repository root:

```console
git submodule update --init --recursive
```

Markdown4D is compiled from source; installing its components into the IDE is unnecessary. Standard builds use embedded shaders.

### Build and run the chat application

Open [QuickNN.groupproj](QuickNN.groupproj) for the main examples, or open [LLMUIChat.dproj](Examples/LLMUIChat/LLMUIChat.dproj) directly. Select **Win64 / Release** and run the project.

In the application:

1. Open the model catalog and add a display name and checkpoint directory. A Qwen3 directory needs `config.json` and `model.safetensors` or `model.safetensors.index.json`. Supply a separate tokenizer directory when necessary.
2. Create a chat, select its model, and adjust the system prompt, context, generation limits, sampling, and memory budgets.
3. Press **Enter** to send and **Ctrl+Enter** to insert a newline. Use the stop button to cancel generation.

Chats, drafts, settings, and reading positions are saved to `%LOCALAPPDATA%\QuickNN\LLMUIChat\state.json`, with a backup copy. Automatic context compression keeps the full original conversation on disk. Existing history from the previous application name is imported automatically.

See the [chat application guide](Examples/LLMUIChat/README.md) for continuation, context compression, and persistence details.

### Console examples

| Project | Purpose | Configuration |
| --- | --- | --- |
| [QuickNNDemo](Examples/ConsoleDemo/QuickNNDemo.dproj) | Generate an image with Qwen-Image 2.1, FLUX.2 Klein, or Z-Image Turbo. | Edit `DemoSettings` in [QuickNN.Demo.Settings.pas](Examples/ConsoleDemo/QuickNN.Demo.Settings.pas); no launch arguments. |
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
| `src/io` | Safetensors loading and shared tokenization. |
| `src/models` | Diffusion and LLM implementations; Qwen-Image components are under `qwen21`. |
| `src/console` | Optional ANSI/Sixel output, separate from the inference core. |
| `Examples` | FMX chat and console applications. |
| `tests` | CPU/GPU, model, quantization, and chat regression checks. |
| `tools` | Build/cleanup scripts, checkpoint diagnostics, and benchmarks. |
| `Markdown4D` | Markdown renderer submodule used by the chat UI. |

## License

QuickNN is distributed under the [MIT License](LICENSE). Model weights, tokenizer assets, adapters, and third-party source dependencies retain their respective licenses.
