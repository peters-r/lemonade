# Strixite

The experimental `strixite` recipe runs a locally built `strix_server` subprocess on
Linux with AMD Strix Halo (`gfx1151`) and 128 GB of RAM. It serves chat and text
completions, including SSE streaming, reasoning, tool calls, and image input.
Strixite does not expose the Responses, embeddings, or reranking APIs.

Build the engine using [Strixite's setup instructions](https://github.com/shawnshekari/strixite#quick-start).
Lemonade does not download a Strixite executable or manage its ROCm runtime. Add
the directory containing `strix_server` to the server process's `PATH`, or set:

```sh
export LEMONADE_STRIXITE_SYSTEM_BIN=/absolute/path/to/strixite/build/strix/strix_server
```

Alternatively, set `strixite.system_bin` in Lemonade's `config.json` to the
executable or its directory. Restart `lemond` after changing its environment.

Download and run the registered model:

```sh
lemonade pull Qwen3.8-Flash-Next-Strixite
lemonade run Qwen3.8-Flash-Next-Strixite
```

The download selects the shipping weight layout, n-gram table, vision tower,
tokenizer, and configuration files from `wemoh/Qwen3.8-Flash-Next-strixw`
(approximately 115 GiB). An existing download can be registered with `POST /pull`
using a new `user.*` model name, `recipe: "strixite"`, `source: "local_path"`, and
an absolute `checkpoint` path pointing to its repository root. Keep the upstream
directory layout intact.

`--ctx-size` maps to Strixite's `--capacity`; when unset, Lemonade keeps the upstream
524,288-token default. Pass other settings through `--strixite-args` or `strixite.args`, for
example `--thinking off --total-slots 1 --low-memory adapt`. Lemonade reserves the
model paths, model ID, host, port, and capacity. `--config` files may supply tuning
settings; Lemonade's command-line values take precedence for managed settings.
Prompt-cache files default to `strixite/prompt-cache` under Lemonade's cache
directory, and `--prompt-cache-dir` can override that location.

The subprocess inherits Lemonade's stdout and stderr at every Lemonade log level.
Control Strixite's verbosity with its own `log-level` setting in the configuration
file or `--log-level` in `strixite_args`.

Lemonade allows Strixite up to 90 seconds to exit after SIGTERM so its RAM prompt
cache can flush to disk. Container deployments should allow at least 120 seconds
for the outer stop timeout, with the systemd service timeout longer still.

The integration uses Lemonade's existing cancellation-aware HTTP proxy and
subprocess lifecycle. `StrixiteBackendTest` runs in the CPU-only C++ CI suite with
a fake child server to check launch arguments, discovery, artifact selection,
non-streaming and streaming forwarding, and unloading.

On a Linux Strix Halo host with the upstream memory/kernel setup and sufficient
disk space, run the hardware smoke test after building Lemonade and Strixite:

```sh
python test/server_llm.py --wrapped-server strixite --backend system
```

Also send a chat request containing a base64 image data URL, verify Prometheus
metrics while the model is loaded, and interrupt a non-streaming request during
prefill to check that a subsequent request remains usable. Hardware CI requires
a Strix Halo runner provisioned with the local executable, ROCm runtime, and model
files; the CPU-only test does not validate inference quality or GPU execution.
