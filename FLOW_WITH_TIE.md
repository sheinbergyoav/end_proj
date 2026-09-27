The inference loop of the TinyStories model follows the standard autoregressive Transformer decoder pipeline ($x \to \text{RMSNorm} \to \text{QKV Projections} \to \text{Attention with TurboQuant KV-Cache} \to \text{Out Projection} \to \text{FFN/MLP} \to \text{Logits}$).

The custom TIE operations accelerate compute-bound matrix multiplications (dense GEMV) and the quantized TurboQuant KV-cache attention datapath.

---

### End-to-End C Inference Flow & TIE Insertion Points

```text
Token ID
   │
   ▼
[1] Embedding Lookup (vocab_size -> dim)
   │
   ▼
┌─────────────────────────── Transformer Layer Loop (L Layers) ──────────────────────────┐
│                                                                                        │
│  [2] RMSNorm: x -> x_norm                                                              │
│                                                                                        │
│  [3] Attention Projections: W_q, W_k, W_v @ x_norm ───► Uses: SDOT4                   │
│                                                                                        │
│  [4] KV-Cache Compression (TurboQuant):                                                │
│      - Centroid Index Generation & Sign Extraction                                     │
│                                                                                        │
│  [5] Multi-Head Attention Evaluation:                                                  │
│      - Step 2: Address calculation for centroid entries ──► Uses: TQ_IDX_LOAD           │
│      - Centroid Dequantization & Lookup ───────────────► Uses: TQ_LOOKUP_CENT / TABLE │
│      - Step 1: Residual sign-bit path accumulation ─────► Uses: TQ_SGNACC2             │
│      - Step 3: Dual 16-bit centroid dot-product (Q·K) ──► Uses: TQ_MAC2                │
│      - Softmax: scores = exp(Q·K / sqrt(d)) / sum(...)                                 │
│      - Step 4: Weighted V reconstruction & accumulation ─► Uses: TQ_VEC_ADD2/ADD4      │
│                                                                                        │
│  [6] Attention Output Projection: W_o @ attn_out ──────► Uses: SDOT4                   │
│                                                                                        │
│  [7] Residual Stream Accumulation: x = x + attn_out                                    │
│                                                                                        │
│  [8] RMSNorm: x -> x_norm2                                                             │
│                                                                                        │
│  [9] FFN Projections (SwiGLU MLP):                                                     │
│      - Gate & Up Projections: W_gate, W_up @ x_norm2 ───► Uses: SDOT4                  │
│      - Activation: silu(gate) * up                                                     │
│      - Down Projection: W_down @ hidden ────────────────► Uses: SDOT4                  │
│                                                                                        │
│  [10] Residual Stream Accumulation: x = x + ffn_out                                    │
└────────────────────────────────────────────────────────────────────────────────────────┘
   │
   ▼
[11] Final RMSNorm & Classifier Head (W_logits @ x) ──────► Uses: SDOT4
   │
   ▼
[12] Sampler / Argmax -> Next Token

```

---

### Detailed Breakdown of Each TIE Operation

#### 1. `sdot4.tie` (`SDOT4` / Matrix Projections)

* **Stage in C Code:** Inside the linear projection routine (`matmul_int8(float* out, int8_t* W, int8_t* x, ...)`). It is invoked inside the inner loop of:
* Attention $Q, K, V$ linear projections ($W_q, W_k, W_v$).
* Attention output projection ($W_o$).
* FFN Gate ($W_{gate}$), Up ($W_{up}$), and Down ($W_{down}$) projections.
* Final vocabulary classifier projection ($W_{classifier}$).


* **C Execution Snippet:**
```c
for (int k = 0; k < dim; k += 4) {
    uint32_t w4 = *(uint32_t*)&W_row[k]; // 4 packed int8 weights
    uint32_t x4 = *(uint32_t*)&x_vec[k]; // 4 packed int8 activations
    acc = SDOT4(res, w4, x4, acc);
}

```


* **Inputs $\rightarrow$ Outputs:**
* `in AR a_packed` (32-bit): 4 signed 8-bit weight values (`a[7:0]`, `a[15:8]`, `a[23:16]`, `a[31:24]`).
* `in AR b_packed` (32-bit): 4 signed 8-bit activation values (`b[7:0]`, `b[15:8]`, `b[23:16]`, `b[31:24]`).
* `in AR acc` (32-bit): 32-bit signed accumulator running sum.
* `out AR res` (32-bit): Result of $acc + \sum_{i=0}^3 (a_i \cdot b_i)$.


* **Register File:** Standard 32-bit general-purpose `AR` registers.


* **Data Buffer (DB) Accessed:**
* Weight buffers stored in external DRAM / L1 Data Memory ($W_q, W_k, W_v, W_1, W_2, W_3$).
* Dynamic activation scratchpad vector $x$.


* **Instruction Encoding:** 24-bit Xtensa standard instruction format.



---

#### 2. `tq_tie_step2.tie` (`TQ_IDX_LOAD` / Index Extraction & Pointer Arithmetic)

* **Stage in C Code:** During Key/Value vector dequantization in the attention layer. Before looking up a quantized centroid, compressed 4-bit indices packed into 32-bit words must be extracted to generate effective table addresses.
* **C Execution Snippet:**
```c
// Extract index at slot i (0 to 3) and compute centroid byte address:
uint32_t packed_idx = kv_cache_indices[token_idx * num_chunks + c];
uint32_t cent_addr  = TQ_IDX_LOAD(centroid_base_addr, packed_idx, slot_sel);

```


* **Inputs $\rightarrow$ Outputs:**
* `in AR base_addr` (32-bit): Pointer to the base of the centroid table in memory.
* `in AR packed_indices` (32-bit): Word containing packed 4-bit indices.
* `in AR slot_sel` (32-bit): 2-bit selector (`slot_sel[1:0]`) choosing which nibble (index 0, 1, 2, or 3) to extract.
* `out AR target_addr` (32-bit): Effective address equal to $base\_addr + (\text{extracted\_idx} \ll 2)$.


* **Register File:** Standard 32-bit `AR` registers.


* **Data Buffer (DB) Accessed:** KV-Cache index buffer containing the compressed 4-bit centroid indices.
* **Instruction Encoding:** 24-bit Xtensa instruction format.



---

#### 3. `tq_cent_table.tie` (`table` / Centroid Lookup ROM)

* **Stage in C Code:** Used immediately after index isolation to load the 16-bit centroid coordinate without performing an off-core SRAM/DRAM memory transaction.
* **C Execution Snippet:**
```c
// Retrieve hardwired fixed-point centroid value directly from silicon:
int16_t cent_val = TQ_LOOKUP_CENT(centroid_index);

```


* **Inputs $\rightarrow$ Outputs:**
* `in tq_centroids cent_entry` (4-bit table operand index).


* `out AR centroid_val` (32-bit): 16-bit signed centroid constant sign-extended to 32 bits.




* **Register File:** `AR` register for output; input encoded directly into the instruction immediate field referencing the table space.


* **Data Buffer (DB) Accessed:** Internal silicon ROM/LUT decoder (zero-cycle memory bus overhead).


* **Instruction Encoding:** 24-bit Xtensa instruction format.



---

#### 4. `tq_tie_step1.tie` (`TQ_SGNACC2` / Residual Sign Accumulation)

* **Stage in C Code:** In the TurboQuant attention scoring/reconstruction loop. TurboQuant represents quantization residuals using 1-bit signs multiplied by a scalar step size $\Delta$:

$$v_i = \text{centroid}_i + \text{sign}_i \cdot \Delta$$


* **C Execution Snippet:**
```c
// Process two 16-bit residual channels in parallel without branching:
uint32_t step_pair = (step1 << 16) | step0;
uint32_t signs     = (sign_bit1 << 1) | sign_bit0;
acc = TQ_SGNACC2(acc, step_pair, signs);

```


* **Inputs $\rightarrow$ Outputs:**
* `in AR acc` (32-bit): Current reconstructed scalar channel accumulator.
* `in AR step_val` (32-bit): Two packed 16-bit quantization residual steps ($\Delta_0 = \text{step}[15:0]$, $\Delta_1 = \text{step}$).
* `in AR sgn_bits` (32-bit): Sign bitfield (`bit 0` determines sign for $\Delta_0$, `bit 1` for $\Delta_1$).
* `out AR res` (32-bit): Updated accumulator after conditional add/sub:

$$\text{res} = \text{acc} + ((-1)^{\text{bit}_0} \cdot \Delta_0) + ((-1)^{\text{bit}_1} \cdot \Delta_1)$$




* **Register File:** Standard 32-bit `AR` registers.


* **Data Buffer (DB) Accessed:** KV-Cache sign-bit bitmask buffer and scalar step size table.
* **Instruction Encoding:** 24-bit Xtensa instruction format.



---

#### 5. `step3.tie` / `tq_tie_step3.tie` (`TQ_MAC2` / Dual 16-bit Centroid MAC)

* **Stage in C Code:** Inside the $Q \cdot K^T$ matrix-vector attention scoring loop. Once centroids and residual steps are unpacked into 16-bit fixed-point format, they are multiplied by query vector entries and accumulated.
* **C Execution Snippet:**
```c
for (int d = 0; d < head_dim; d += 2) {
    uint32_t cents = *(uint32_t*)&k_reconstructed[d]; // 2x 16-bit fixed-point
    uint32_t acts  = *(uint32_t*)&q_vector[d];        // 2x 16-bit fixed-point
    attn_score = TQ_MAC2(attn_score, cents, acts);
}

```


* **Inputs $\rightarrow$ Outputs:**
* `in AR in_acc` (32-bit): Running 32-bit scalar product accumulator.
* `in AR centroids` (32-bit): Two 16-bit fixed-point centroid values (`cents[15:0]`, `cents[31:16]`).
* `in AR activations` (32-bit): Two 16-bit query vector elements (`acts[15:0]`, `acts[31:16]`).
* `out AR out_acc` (32-bit): Accumulated sum:

$$\text{out\_acc} = \text{in\_acc} + (\text{cents}_0 \cdot \text{acts}_0) + (\text{cents}_1 \cdot \text{acts}_1)$$




* **Register File:** Standard 32-bit `AR` registers.


* **Data Buffer (DB) Accessed:** Query vector scratchpad ($Q$) and dequantized Key vector buffer ($K$).
* **Instruction Encoding:** 24-bit Xtensa instruction format.



---

#### 6. `tq_tie_step4.tie` (`TQ_VEC_ADD2` / `TQ_VEC_ADD4` / SIMD Value Accumulation)

* **Stage in C Code:** In the Value aggregation phase ($\text{out} = \sum_j \text{softmax\_weight}_j \cdot V_j$) and the residual stream additions ($x = x + \text{attn\_out}$, $x = x + \text{ffn\_out}$).
* **Configuration A: Standard Register File Variant (`TQ_VEC_ADD2`)**
* **C Execution Snippet:**
```c
uint32_t v_acc = *(uint32_t*)&out_context[d];
uint32_t v_val = *(uint32_t*)&weighted_v[d];
*(uint32_t*)&out_context[d] = TQ_VEC_ADD2(v_acc, v_val);

```


* **Inputs $\rightarrow$ Outputs:**
* `in AR v_in1` (32-bit): Two 16-bit values ($v_{1\_hi}, v_{1\_lo}$).
* `in AR v_in2` (32-bit): Two 16-bit values ($v_{2\_hi}, v_{2\_lo}$).
* `out AR v_out` (32-bit): Parallel lane additions: $\{v_{1\_hi} + v_{2\_hi}, \; v_{1\_lo} + v_{2\_lo}\}$.


* **Register File:** Standard 32-bit `AR` registers.


* **Data Buffer (DB):** Context accumulator vector buffer and Value vector cache.


* **Configuration B: Custom Register File Variant (`TQ_VEC_ADD4` with `regfile simd64`)**
* **C Execution Snippet:**
```c
// Using custom SIMD 64-bit vector registers to avoid AR register pressure:
simd64 v_acc = READ_SIMD64(v_ptr);
simd64 v_val = READ_SIMD64(val_ptr);
v_acc = TQ_VEC_ADD4(v_acc, v_val);

```


* **Inputs $\rightarrow$ Outputs:**
* `in simd64 v1` (64-bit): 4 packed 16-bit vector lanes.


* `in simd64 v2` (64-bit): 4 packed 16-bit vector lanes.


* `out simd64 v_out` (64-bit): 4 packed 16-bit element-wise additions.




* **Register File:** Custom 64-bit wide, 16-entry `simd64` register file (`v0`–`v15`).


* **Data Buffer (DB):** Multi-head attention output vector context array.
* **Instruction Encoding:** 24-bit custom Xtensa instruction mapped to the `simd64` register file ports.





---

### Summary Matrix

| TIE File | Hardware Instruction | Inference Loop Phase | Inputs (Width & RegFile) | Output (Width & RegFile) | Primary Data Buffer (DB) |
| --- | --- | --- | --- | --- | --- |
| **`sdot4.tie`** | `SDOT4` | GEMV / Projections ($Q, K, V, O$, FFN, Head) | 3x 32-bit (`AR`)

 | 1x 32-bit (`AR`)

 | Weight tensors & Activation arrays |
| **`tq_tie_step2.tie`** | `TQ_IDX_LOAD` | KV-Cache Index Unpacking | 3x 32-bit (`AR`)

 | 1x 32-bit (`AR`)

 | Packed 4-bit index buffer |
| **`tq_cent_table.tie`** | `TQ_LOOKUP_CENT` | Centroid Retrieval | 1x 4-bit (`table`)

 | 1x 32-bit (`AR`)

 | Internal Silicon Table ROM

 |
| **`tq_tie_step1.tie`** | `TQ_SGNACC2` | Residual Sign Reconstruction | 3x 32-bit (`AR`)

 | 1x 32-bit (`AR`)

 | Sign-bit bitmask array & Step table |
| **`step3.tie`** | `TQ_MAC2` | Attention Scoring ($Q \cdot K^T$) | 3x 32-bit (`AR`)

 | 1x 32-bit (`AR`)

 | Dequantized Key buffer & Query vector |
| **`tq_tie_step4.tie`** | `TQ_VEC_ADD2` / `ADD4` | Value Aggregation ($\sum \alpha V$) | 2x 32-bit (`AR`) or 2x 64-bit (`simd64`)

 | 1x 32-bit (`AR`) or 1x 64-bit (`simd64`)

 | Output Context accumulator buffer |
