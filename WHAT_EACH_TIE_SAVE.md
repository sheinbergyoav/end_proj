# What Each Original TIE Instruction Saves

Even without built-in Cadence library macros, your original TIE implementations accelerate execution by mapping multi-instruction software loops into single-cycle combinational hardware datapaths using raw Verilog operators (`*`, `+`), manual sign-extension wiring (`{{24{...}}, ...}`), and hardware multiplexing.

---

## 1. `sdot4.tie` (`SDOT4` / Matrix Projections)

- **The Software Bottleneck:** Calculating a 4-element dot product on a standard 32-bit RISC core requires loading 4 separate bytes (or loading a 32-bit word followed by three shifts and masks), sign-extending each 8-bit slice into a 32-bit register, executing 4 scalar multiplication instructions, and chaining 4 additions, along with loop branch overhead. This requires roughly 15 to 22 instructions and CPU cycles per 4 matrix elements.

- **The Original TIE Hardware Datapath:**
  - Receives two 32-bit packed registers containing four 8-bit values each.
  - Slices the byte lanes directly at zero gate delay and performs two's-complement sign-extension into 32-bit signed values using wire concatenation: `{{24{a_packed[msb]}}, a_packed[7:0]}`.
  - Executes four parallel combinational multiplications (`prod0 = a0 * b0; ...; prod3 = a3 * b3;`) and resolves the total running sum (`assign res = acc + prod0 + prod1 + prod2 + prod3;`) in one clock cycle.

- **What It Saves:** **Saves ~15–20 CPU cycles and instructions per 4 elements**. It also reduces memory load instructions by **75%**, loading four INT8 weights in a single 32-bit memory transaction instead of four individual byte accesses.

---

## 2. `tq_cent_table.tie` (Centroid Lookup)

- **The Software Bottleneck:** Reading codebook centroids from an array in memory requires address calculation, a pointer dereference, and an SRAM or data cache load (`L16SI` / `L32I`). This incurs a load-to-use interlock penalty (stalling the pipeline if the next instruction needs the data immediately) and consumes L1 data cache capacity.

- **The Original TIE Hardware Datapath:**
  - Implements the codebook as a dedicated combinational ROM decoder or hardwired table directly inside the execution datapath.
  - Indexes the centroid value directly via immediate hardware decoding.

- **What It Saves:** **Saves memory read latency and data cache capacity**. It eliminates load instructions and load-use pipeline bubbles by retrieving the 16-bit centroid in zero memory-bus cycles.

---

## 3. `tq_tie_step1.tie` (`TQ_SGNACC2` / Residual Sign Accumulation)

- **The Software Bottleneck:** TurboQuant residual reconstruction adjusts accumulated values by either adding or subtracting a scalar step size $$\Delta$$ based on 1-bit sign flags:

```c
if (sign0) acc -= step0; else acc += step0;
if (sign1) acc -= step1; else acc += step1;
```

  In software, data-dependent branches trigger branch mispredictions, flushing the CPU pipeline and incurring a 3- to 5-cycle penalty per mispredict.

- **The Original TIE Hardware Datapath:**
  - Evaluates sign bits combinational using ternary conditional hardware multiplexers (`sign ? (acc - step) : (acc + step)`) or two's-complement sign-inversion logic.
  - Evaluates both sign lanes concurrently in a single clock cycle without software branching.

- **What It Saves:** **Completely eliminates branch instructions and pipeline branch misprediction stalls**. It folds testing, conditional inversion, and dual accumulation into a single clock cycle.

---

## 4. `tq_tie_step2.tie` (`TQ_IDX_LOAD` / Index Unpacking & Address Calculation)

- **The Software Bottleneck:** To look up quantized weights, packed sub-byte indices (such as 4-bit nibbles stored inside a 32-bit word) must be extracted in software:

```c
uint32_t idx = (packed_word >> (slot * 4)) & 0x0F;
uint32_t addr = base_addr + (idx << 2);
```

  This creates a serialized chain of ALU operations (`EXTUI` / shift, `AND`, shift-left, `ADD`) taking 4 to 6 scalar instructions per index.

- **The Original TIE Hardware Datapath:**
  - Uses hardwired bit-slicing (`wire [3:0] idx0 = packed[3:0]; ...`) which costs **zero gates and zero propagation delay**.
  - Selects the target index with a combinational multiplexer and computes the effective byte address with raw adder logic (`assign target_addr = base_addr + (chosen_idx << 2);`) in one cycle.

- **What It Saves:** **Saves 4 to 6 scalar ALU instructions and reduces register pressure** by resolving sub-byte unpacking and address calculation in a single cycle.

---

## 5. `step3.tie` / `tq_tie_step3.tie` (`TQ_MAC2` / Dual 16-bit Centroid MAC)

- **The Software Bottleneck:** Calculating attention scores ($$Q \cdot K^T$$) over unpacked 16-bit centroid channels requires sequential multiplications, intermediate register writebacks, and subsequent accumulation instructions.

- **The Original TIE Hardware Datapath:**
  - Unpacks two 16-bit pairs from standard 32-bit registers and computes two multiplications and accumulations in parallel using combinational logic:

```tie
assign out_acc = in_acc + (cents[15:0] * acts[15:0]) + (cents[31:16] * acts[31:16]);
```

- **What It Saves:** **Cuts arithmetic instruction count by over 50%** during attention scoring by parallelizing two multiplications and accumulations into a single execution step.

---

## 6. `tq_tie_step4.tie` (`TQ_VEC_ADD2` / `TQ_VEC_ADD4` / Value Accumulation)

- **The Software Bottleneck:** Value vector aggregation ($$\sum \alpha_j V_j$$) requires element-wise vector additions across multi-dimensional hidden states. In standard 32-bit scalar execution, doing this channel by channel causes high register pressure, leading to stack spilling and refilling.

- **The Original TIE Hardware Datapath:**
  - Executes multiple 16-bit lane additions concurrently.
  - When paired with a custom wide register file (`regfile simd64`), it operates across 64-bit vector widths simultaneously (`assign v_out = {lane3, lane2, lane1, lane0};`).

- **What It Saves:** **Saves 50%–75% of vector addition instructions** and relieves general-purpose register pressure, preventing memory spill/fill overhead.

---

## How `sdot4` Relates to the TurboQuant Acceleration Pipeline

While `sdot4` operates on INT8 weight matrices rather than quantized KV-cache centroids, it is essential to the TurboQuant acceleration framework for two primary architectural reasons:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                             PRODUCER STAGE                                  │
│                                                                             │
│  Activation x ──┐                                                           │
│                 ├─► [ SDOT4 GEMV ] ──► Raw Key (K) & Value (V) Vectors      │
│  Weights W_k/v ─┘                                                           │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             CONSUMER STAGE                                  │
│                                                                             │
│  Raw K & V Vectors ──► [ TurboQuant Compression ]                           │
│                             │                                               │
│                             ▼                                               │
│                        KV-Cache:                                            │
│                        - 4-bit Centroid Indices                             │
│                        - 1-bit Residual Signs                               │
│                             │                                               │
│                             ▼                                               │
│  Attention Query Q ─┐       │                                               │
│                     ├─► [ TQ_IDX_LOAD ]                                     │
│                     │   [ TQ_LOOKUP_CENT ] ──► Accelerated Attention Output │
│                     │   [ TQ_SGNACC2 ]                                      │
│                     │   [ TQ_MAC2 / VEC_ADD ]                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1. The Producer–Consumer Relationship

TurboQuant does not quantize the static model weights; it quantizes the **dynamic Key ($$K$$) and Value ($$V$$) token states generated on-the-fly during autoregressive decoding**.

- **`sdot4` is the Producer:** For every incoming token, the current hidden state must be multiplied by the projection matrices ($$W_k$$ and $$W_v$$) to generate the raw Key and Value vectors. Because weights are stored as INT8, `sdot4` performs this dense linear transformation.
- **TurboQuant is the Consumer:** As soon as `sdot4` outputs the raw $$K$$ and $$V$$ vectors, TurboQuant compresses them into discrete centroid indices and residual signs to store them in the KV-cache.
- **The Attention Step:** When subsequent tokens attend to past context, the specialized `tq_*` instructions (`TQ_IDX_LOAD`, `TQ_LOOKUP_CENT`, `TQ_SGNACC2`, `TQ_MAC2`) dequantize and score those stored tokens against the Query vector.

Without `sdot4`, the pipeline would suffer a bottleneck in generating the uncompressed vectors before TurboQuant could compress or process them.

### 2. Amdahl's Law Across Full Transformer Inference

In a Transformer decoder like TinyStories, runtime is divided across two major compute categories:

1. **Dense Projections (GEMV):** Static matrix-vector multiplications across large linear layers ($$W_q, W_k, W_v, W_{out}$$, FFN Gate, FFN Up, FFN Down, and the Logits classifier). These account for **70% to 80% of total inference operations**.
2. **Attention Scoring & Context Generation:** Computing $$Q \cdot K^T$$ and multiplying by $$V$$ over previous tokens in the KV-cache.

If you only accelerated the TurboQuant KV-cache calculations, Amdahl's Law dictates that overall model speedup would remain strictly limited by the un-accelerated dense matrix multiplications.

By pairing `sdot4` (accelerating the static INT8 linear layers) with the `tq_*` instructions (accelerating the dynamic compressed KV-cache attention layers), the hardware accelerates **both compute-bound and memory-bound phases** of the inference loop, achieving balanced speedups across the entire network.
