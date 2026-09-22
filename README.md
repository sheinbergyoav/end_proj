# end_proj
end project 

<img width="257" height="76" alt="image" src="https://github.com/user-attachments/assets/a7990566-7826-4646-95f6-38dabff5b424" />

# TurboQuant on Xtensa: A Hardware/Software Co-Designed LLM Inference Engine

Porting a quantized LLaMA-2 (`stories15M`) inference engine to an FPU-less **Cadence Tensilica Xtensa LX (tie_dev2)** embedded core, with a custom KV-cache compression scheme (**TurboQuant**) implemented as hand-designed **TIE (Tensilica Instruction Extension)** hardware instructions.

This repository contains the full toolchain: offline model export and constant generation, the custom instruction-set extensions, the C inference engine, and the Cadence Xplorer build/simulation flow used to produce the results below.

---

## Why this exists

Autoregressive decoding keeps every past Key/Value vector in a KV-cache that grows without bound as generation continues — on memory-constrained edge hardware, this cache (not compute) becomes the bottleneck. TurboQuant compresses each KV vector to **4 bits** (Haar rotation → Lloyd-Max scalar quantization → QJL residual sign projection) and evaluates the compressed representation directly in custom hardware, avoiding both the memory cost of FP32 caching and the runtime cost of decompressing on a core with no FPU.

## Results at a glance

| Metric | Baseline | This work | Change |
|---|---|---|---|
| Per-token KV-cache footprint | 2,304 B | 384 B | **8.0× smaller** |
| Total KV-cache (256 tokens) | 3.38 MB | 0.56 MB | **6.0× smaller** |
| Total execution cycles/token | 3.86 B | 1.96 B | **~49% fewer** |
| Pipeline interlock stalls | 17.66% | 2.88% | **40.4pp reduction** |

Full breakdown, methodology, and profiling tables are in [`docs/`](docs/).

## Repository structure

```
.
├── model/
│   ├── export.py                    # PyTorch/HF checkpoint -> Q8_0 quantized .bin (ak42 layout)
│   └── generate_tq_constants.py     # Derives Haar/Gaussian matrices + Lloyd-Max codebook -> .h + .tie
├── tie/
│   ├── tq_cent_table.tie            # Hardware centroid ROM (8 x 16-bit constants)
│   ├── tq_tie_step1.tie             # Sign scoring   (TQ_SGN_LOAD, TQ_SGNACC2, TQ_SACC_RD)
│   ├── tq_tie_step2.tie             # Index load     (TQ_IDX_LOAD)
│   ├── tq_tie_step3.tie             # Dual MAC decode(TQ_DECMAC2, TQ_ACC_RD)
│   ├── tq_tie_step4.tie             # Value accumulate(TQ_SCALEACC2, TQ_VACC_RD0/1)
│   ├── tq_tie_q15mac2.tie           # Dual 16-bit fixed-point MAC (Q12 projection pipeline)
│   └── tq_tie_sdot4.tie             # Single-cycle INT8x4 dot-product (quantized matmul)
├── src/
│   └── main.c                       # Inference engine (forward pass, RoPE, sampler, debug flags)
├── bin/                              # Quantized model binaries (.bin) — generated, not committed
├── docs/                             # Full technical write-up, figures, and results tables
└── README.md
```

## Hardware target

- **Core:** Cadence Xtensa LX8.0.5, custom configuration `tie_dev2` — 7-stage pipeline, **no hardware FPU**
- **Local memory:** 256 KB Tightly-Coupled Memory (TCM), split as DRAM0 (128 KB) + DRAM1 (128 KB), 1-cycle latency
- **System memory:** 512 MB over a 128-bit AXI4 bus (model weights, KV-cache, activations)
- **Custom datapath:** single-cycle TIE instructions for TurboQuant scoring/decoding and quantized-matmul dot products

## Build & run

```bash
# 1. Export and quantize the model
python model/export.py bin/stories15M_q80.bin --version 2 --checkpoint stories15M.pt

# 2. Generate TurboQuant hardware/software constants from the exported model
python model/generate_tq_constants.py \
    -m bin/stories15M_q80.bin \
    -o tq_constants_stories.h \
    -t tie/tq_cent_table.tie

# 3. Add the .tie files to the TIE compiler in Cadence Xplorer and synthesize the ISA

# 4. Cross-compile for tie_dev2
xt-xcc -O2 -DNDEBUG -xtensa-core=tie_dev2 -mlongcalls -o eprj2 src/main.c

# 5. Run on the instruction-set simulator with cycle profiling
xt-run \
  --xtensa-core=tie_dev2 \
  --xtensa-system=<path-to-tie_dev2-config> \
  --xtensa-params=<path-to-tie_dev2_tq_2.tdk> \
  --console \
  --tieprint=TIEprint.log \
  --summary \
  eprj2
```

Deterministic verification uses **greedy argmax sampling** (`temperature = 0.0`) so that hardware output can be diff-checked byte-for-byte against the desktop golden reference — see `sample()` in `src/main.c`.

## Key optimizations implemented

1. **TurboQuant KV-cache compression** — Haar rotation + Lloyd-Max quantization + QJL sign projection, evaluated in single-cycle custom TIE hardware (no runtime centroid memory loads; the codebook is synthesized into combinatorial logic).
2. **Factored value-path math** — dense projection matrices moved outside the per-token attention loop and applied once per head instead of once per token.
3. **RoPE look-up tables** — rotary embedding angles depend only on position, so trigonometric evaluation is precomputed once at startup instead of 8,640 soft-float calls/token.
4. **Division-free dequantization** — replaced per-iteration integer division (16–34 cycles, non-pipelined) with a sequentially incrementing counter, removing ~300,000 divisions/token.
5. **Q12 fixed-point headroom** — reworked accumulator scaling to guarantee no 32-bit overflow across the full projection dimension without widening registers.
6. **4-byte alignment enforcement** — eliminated unaligned-load exceptions from `int16_t` arrays accessed as packed 32-bit words by custom MAC instructions.

## Documentation

The full write-up — theoretical background, hardware/software co-design details, verification methodology, and complete experimental results — lives in [`docs/`](docs/) as the project report, including all figures referenced above.

## License

Sheinberg Yoav and Shor Guy
