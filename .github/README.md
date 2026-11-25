# llama.cpp · Inkling fork

Faster **Inkling-2-Mini** on Apple Silicon. A fork of [Unsloth's llama.cpp](https://github.com/unslothai/llama.cpp) (what Unsloth Studio ships) with Metal kernels and fused ggml ops for Inkling 2's looped architecture.
Same GGUF, same quality, **~1.6× decode on an M1 Max**. A pipeline follows Unsloth's daily releases and installs the build into apps.

| M1 Max 64 GB, Inkling-2-Mini UD-Q4_K_XL | stock | this fork |
|---|---|---|
| `llama-bench` decode (tg64), on AC power | 23.2 tok/s | **36.5 tok/s** |
| `llama-server` with Unsloth Studio's flags, chat at effort high | 22.7–22.8 tok/s | **31.0–33.6 tok/s** |
| same, as Studio ships it (DSpark speculative decoding on) | 20.0–25.4 tok/s | — |
| KLD vs a CPU reference, 32 chunks, two texts (lower is better) | 0.00295 / 0.00211 | 0.00277 / 0.00229 |
| quiz, 10 math questions, effort high | 10/10 | 10/10 |

Speed rows: b12675 on AC power (b12676 merged cleanly; in Low Power Mode on battery both builds drop, to 15.9 and 20.1). KLD: b12676, prose text / this repo's docs; the difference is within noise on one text and borderline (+0.00018 ± 0.00009) on the other, top-token agreement 98.6% vs 98.7%.

## Install

Needs macOS on Apple Silicon, Xcode command-line tools, `cmake` (`brew install cmake`).

```bash
git clone https://github.com/GlebRadchenko/llama.cpp && cd llama.cpp
inkling/inkling.sh install studio                    # build, then 3 clicks in Unsloth Studio
MODEL=~/models/Inkling-2-Mini-UD-Q4_K_XL.gguf inkling/inkling.sh install server   # OpenAI API for any app
```

| target | what you get |
|---|---|
| `install studio` | static Metal binaries in `build-inkling/bin`. Studio takes a custom llama.cpp only from its own window: **Settings → llama.cpp folder → `build-inkling/bin`**, **speculative decoding → Off**, reload the model. Undo: clear the folder setting. |
| `install server` | `llama-server` as a login service (launchd) on `http://127.0.0.1:8080/v1`, reasoning effort high, 64k context, 4 slots. Any OpenAI-compatible client works with it: Open WebUI, Jan, Continue, Cursor, scripts. `PORT`, `CTX`, `EFFORT`, `ALIAS` override the defaults; `inkling/inkling.sh uninstall server` removes it. |
| `build` | just the binaries (`llama-server`, `llama-cli`, `llama-bench`, `llama-perplexity`), for anything that runs a llama.cpp binary |

Apps that embed their own engine (Ollama, LM Studio) cannot swap it: point them at the `server` endpoint instead.

## Follow upstream

Unsloth publishes a llama.cpp release almost daily: ggml-org's build `bNNNNN` plus a set of PRs, among them Inkling 2 support.

```bash
inkling/inkling.sh sync            # import the latest Unsloth release, merge it into this branch
inkling/inkling.sh check           # op tests (Metal vs CPU)
MODEL=... inkling/inkling.sh check # + KLD and decode speed against the stock build of the same release
inkling/inkling.sh install studio  # rebuild in place; Studio picks it up on the next model load
```

| branch | content |
|---|---|
| **`inkling`** (default) | the work: the changes below, merged with each Unsloth release |
| `unsloth` | one commit per Unsloth release, the exact source of the release tarball. Its parents are the previous import and ggml-org's `bNNNNN`, so every merge is a real 3-way merge and `git diff ggml-bNNNNN unsloth-<tag>` shows Unsloth's PR set |
| `master` | Unsloth's `master`, untouched |

Tags: `unsloth-<release>` for each import, `ggml-bNNNNN` for its ggml-org base. A merge conflict stops `sync`; fix it, commit, run `check`. Ship a merge only if `check` shows patched KLD not above stock and the speed holds.

## Why it was slow, and what changed

Decode was not memory-bound: ~3.6 GB of weights per token is ~10 ms at the M1 Max's bandwidth, but the GPU spent ~42 ms per token. Inkling 2 runs its middle layers in a loop (48 layer visits of many tiny ops), which made **~5.7k serial Metal dispatches per token**, each paying ~5–8 µs of launch and barrier latency. Every change below removes dispatches or serial barrier levels: **5682 → 2371 dispatches, 4378 → 1703 barriers, 41.9 → ~26 ms GPU time per token.**

| commit | change | decode A/B |
|---|---|---|
| [`5d505ab`](https://github.com/GlebRadchenko/llama.cpp/commit/5d505aba) | NVFP4 `mul_mv`: word loads, LUT-free E2M1 decode (two nibbles per half2), 4 rows per simdgroup | +6.7% |
| [`0c1570f`](https://github.com/GlebRadchenko/llama.cpp/commit/0c1570fe) | QK clip as `t·l2_norm(x, eps=t)`: 23 → 8 dispatches per layer | +8% |
| [`2d5998a`](https://github.com/GlebRadchenko/llama.cpp/commit/2d5998af) | load-time derived weights; conv residual + scale folded into the taps; Q/K/V loop controllers batched | +8.5% |
| [`1060000`](https://github.com/GlebRadchenko/llama.cpp/commit/10600002) | new `GGML_GLU_OP_SITU` (softcapped SwiGLU), CPU + Metal: 9 → 1 dispatch | +3.4% |
| [`97b9a3d`](https://github.com/GlebRadchenko/llama.cpp/commit/97b9a3dc) | one-token relative bias: gather, then project; per-query sink handed to flash attention | +10.6% |
| [`60fe440`](https://github.com/GlebRadchenko/llama.cpp/commit/60fe440a) | clip factors folded into the attention scale and the bias table | +0.5% |
| [`98957df`](https://github.com/GlebRadchenko/llama.cpp/commit/98957df3) | new `GGML_OP_DEPTH_MIX` (attention over depth), CPU + Metal, kernel specialized per shape | +10% |
| [`1cf8db6`](https://github.com/GlebRadchenko/llama.cpp/commit/1cf8db65) | Metal fusion CONCAT + CPY + SSM_CONV (short conv with its state store) | +4.8% |
| [`dd462a8`](https://github.com/GlebRadchenko/llama.cpp/commit/dd462a80) | MoE per-expert scales folded (SITU row scale; down scale into the router weights) | +1.8% |
| [`4c11e1e`](https://github.com/GlebRadchenko/llama.cpp/commit/4c11e1ed) | new `GGML_OP_LOOP_GATE` (loop controller update), elementwise | +2.6% |
| [`fa3b0b8`](https://github.com/GlebRadchenko/llama.cpp/commit/fa3b0b83) | test cases: `INK_EVAL=1 test-backend-ops test` (39 cases), `INK_PERF=1 ... perf` | — |

Full log with per-step measurements: [`inkling/SPEEDUP.md`](../inkling/SPEEDUP.md).

## How quality is checked

- **KLD against a CPU-only reference** (`llama-perplexity -dev none`, every op on the CPU), on the prefill path (ubatch 512) and the decode path (ubatch 1).
- Compare with the **stock build of the same release on the same text**, never with an old number. The looped model amplifies float rounding: with 4 chunks, an exact algebraic rewrite moves KLD by ±0.0003 (stock itself moved 0.00158 → 0.00128 between two releases). Decide with 32 chunks.
- b12675, 32 chunks prefill: stock 0.00295, fork 0.00277 (max 1.82 vs 0.57). 8 chunks decode: stock 0.00279, fork 0.00274.
- Every new op and kernel is tested against its CPU implementation (`test-backend-ops`, 39/39).
- 10/10 on a math quiz at effort low and high, and with 4 concurrent requests.

## Everything else we tried

| idea | result |
|---|---|
| Runtime knobs on stock: backend sampling, MoE de-dup, no command concurrency, no residency sets, poll/priority tuning, private/shared buffers | all within noise (20.2–23.3 vs 21.3–22.0 tok/s): the cost is in the graph, not the runtime |
| Studio's extra flags (`--kv-unified`, `--metrics`, `--ctx-checkpoints`, mlock, 64k context, 4 slots) | no cost: 33.2–33.7 tok/s with each, on the patched build |
| CPU expert offload, the RTX tip from Reddit (`-nopo 1 -ot "blk\.([0-5])\.ffn_.*_exps.*=CPU"`) | slower on unified memory: pp512 323 → 196, tg64 37.5 → 33.2 |
| MLX (mlx-vlm 0.8.1, the original NVFP4 weights) | slower: 18.6 tok/s, 18.7–19.2 with its DSpark drafter |
| DSpark speculative decoding | a loss with these changes: 23–29 vs 31–34 tok/s. The 2-token verify costs 43.5 ms vs 28.1 ms for one token (MoE reads experts per token), and prose drafts are often rejected. On stock it is about neutral |
| Q8_0 requant of the experts | no gain |
| LOOP_GATE with the RMS norm fused in | 1.5% slower (one threadgroup per row uses 1–3 GPU cores); kept the elementwise form, the fused one is behind `INK_GATE_NORM=1` |
| DEPTH_MIX split over more threadgroups | no gain, reverted |
| upstream's own LUT-free NVFP4 kernel (landed in b12675) | 35.0 vs 36.5 tok/s for ours; ours kept |
| conv fusion, first version | wrong (KLD 0.10–0.20): it skipped the concat output that the full-attention taps also read. Caught by KLD, fixed, bit-exact since |

## How good is the model

Inkling-2-Mini at effort high, measured on this machine (Studio API, one sample per task):

- **42 fresh tasks** (math, code with hidden tests, logic, instruction following in EN and RU, knowledge, ~10k-token long context, trick questions): Inkling 42/42, Claude Opus 4.6 42/42, Claude Opus 4.5 39/42. The set was too easy to separate the top two.
- **TSP heuristic** (pure Python, 2 s per instance, 100–1000 points, 0–100 score): Opus 4.6 ≈ 82, Inkling ≈ 61 on its valid samples. Inkling used an O(n²) matrix with exhaustive 2-opt, fine to 500 points but weak at 1000; one of its 3 samples hit the 32k-token cap mid-code.

## Ports

| release | merge |
|---|---|
| b12651-mix-c7059d6 | original work |
| b12675-mix-d14fe45 | conflicts in 3 commits: upstream landed its own NVFP4 decode (ours kept, faster), dropped a reserve-only block, added a shift for padded banded spans (now reads our clip-folded table), added an MoE amax rescale for other archs |
| b12676-mix-c32dcc8 | clean merge |

## Limits

- Tuned and verified for Metal and this model. The new ops have CPU code; CUDA and Vulkan would run them on the CPU, so this fork is not for NVIDIA boxes.
- Not measured: image and video input, speed beyond ~8k tokens of context.

## Upstream

Inkling 2 lives in [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp); Unsloth pins PRs from there. ggml-org's rules forbid agent-submitted PRs and want small changes their authors can defend line by line, and maintainers already have small PRs open on the same graph overhead (ggml-org#32948, #32947). The best candidate is the NVFP4 kernel (`5d505ab`): small, generic, no overlapping PR.

## License

MIT, same as llama.cpp.
