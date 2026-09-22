/*
 * main.c  quant
 *
 *  Created on: Aug 9, 2026
 *      Author: sheinby70
 */

/* Inference for Llama-2 Transformer model in pure C, int8 quantized forward pass. */
//FOR FULL DEBUG MODE : ENABLE/DISABLE
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <assert.h>

//choose model
#define TINY_STORIES
//#define TINY_LLAMA

#define USE_HEADER_FOR_TQ_CONSTANTS 1 // Get constants from header file instead of runtime .
#define FP_SUPPORTED 0
#define XTENSA_RUN 1

// ---------------------------------------------------------------------------
// TurboQuant TIE acceleration
//   TQ_TIE_LEVEL 0 = original pure C
//   TQ_TIE_LEVEL 3 = index term of tq_score via TQ_DECMAC2 + centroid ROM
//   TQ_TIE_LEVEL 4 = also accelerates Value path using TQ_SCALEACC2
//   TQ_TIE_SGN   1 = also use TQ_SGNACC2 for the sign term (needs step1.tie)
//   TQ_TRACE_ENABLE 1 = restore the original per-call printf tracing
// ---------------------------------------------------------------------------
#define TQ_TIE_LEVEL     4
#define TQ_TIE_SGN       1
#define TQ_TRACE_ENABLE  0

#if TQ_TRACE_ENABLE
#define TQ_TRACE(...) printf(__VA_ARGS__)
#else
#define TQ_TRACE(...) ((void)0)
#endif

#if (TQ_TIE_LEVEL >= 3) || TQ_TIE_SGN
#include <xtensa/tie/tq_tie2.h>
#endif

#ifdef TINY_STORIES

#define USE_HEADER_FILES 1            // Set to 0 to use fopen/fread
#define MODEL_PARAMS_H              "stories15M_q80.h"
#define MODEL_TOKENIZER_H           "tokenizer_data.h"
#define MODEL_CONSTANTS_H           "tq_constants_stories.h"
#define MODEL_PTR_bin               bin_stories15M_q80_bin
#define MODEL_SIZE_bin              bin_stories15M_q80_bin_len
#define TOKEINZER_PTR_bin           bin_tokenizer_bin

#define MODEL_PARAMS_BIN "bin//stories15M_q80.bin"
#define MODEL_TOKENIZER_BIN "bin//tokenizer.bin"

#define EXAMPALE_PROMPT_FOR_MODEL "Once upon a time"
#endif

#ifdef TINY_LLAMA
#define DEBUG_ALL_FLAGS
#define USE_HEADER_FILES 0            // Set to 0 to use fopen/fread
//#define MODEL_PARAMS_H
//#define MODEL_TOKENIZER_H
#define MODEL_CONSTANTS_H "tq_constants_tinyllama.h"
//#define MODEL_PTR_bin
//#define MODEL_SIZE_bin
//#define TOKEINZER_PTR_bin

#define MODEL_PARAMS_BIN "bin//tinyllama_q80.bin"
#define MODEL_TOKENIZER_BIN "bin//tokenizer_tinyllama.bin"

/*
#define EXAMPALE_PROMPT_FOR_MODEL "<|system|>\n" \
                         "You are a concise assistant. Answer in one sentence only.</s>\n" \
                         "<|user|>\n" \
                         "What is the speed of light</s>\n" \
                         "<|assistant|>\n"
*/
#define EXAMPALE_PROMPT_FOR_MODEL "hello"

#endif


#if USE_HEADER_FILES
    #include MODEL_PARAMS_H
    #include MODEL_TOKENIZER_H
#endif //USE_HEADER_FILES


#if USE_HEADER_FOR_TQ_CONSTANTS
#include MODEL_CONSTANTS_H
#endif //USE_HEADER_FOR_TQ_CONSTANTS


// ----------------------------------------------------------------------------
// Globals
int GS = 0; // group size global for quantization of the weights

// TurboQuant KV-cache quantization parameters (see TurboQuant library section below)
#define TQ_BITS 4                          // total bits/coord for cached K and V (PROD mode)
#define TQ_SEED 0x54517545ULL              // fixed seed: Pi/M must match between quant & dequant

// DEBUG FLAGS (Set to 1 to enable, 0 to disable)
#ifndef DEBUG_ALL_FLAGS
#define DEBUG_MALLOC_RUN_STATE      0
#define DEBUG_FREE_RUN_STATE        0
#define DEBUG_MEMORY_MAP_WEIGHTS    0
#define DEBUG_READ_CHECKPOINT       0
#define DEBUG_BUILD_TRANSFORMER     0
#define DEBUG_FREE_TRANSFORMER      0
#define DEBUG_RMSNORM               0
#define DEBUG_SOFTMAX               0
#define DEBUG_MATMUL                0
#define DEBUG_FORWARD               0
#define DEBUG_COMPARE_TOKENS        0
#define DEBUG_BUILD_TOKENIZER       0
#define DEBUG_FREE_TOKENIZER        0
#define DEBUG_DECODE                0
#define DEBUG_SAFE_PRINTF           0
#define DEBUG_STR_LOOKUP            0
#define DEBUG_ENCODE                0
#define DEBUG_SAMPLE_ARGMAX         0
#define DEBUG_SAMPLE_MULT           0
#define DEBUG_COMPARE               0
#define DEBUG_SAMPLE_TOPP           0
#define DEBUG_BUILD_SAMPLER         0
#define DEBUG_FREE_SAMPLER          0
#define DEBUG_RANDOM_U32            0
#define DEBUG_RANDOM_F32            0
#define DEBUG_SAMPLE                0
#define DEBUG_TIME_IN_MS            0
#define DEBUG_GENERATE              0
#define DEBUG_READ_STDIN            0
#define DEBUG_CHAT                  0
#define DEBUG_ERROR_USAGE           0
#define DEBUG_MAIN                  0

#else
#define DEBUG_MALLOC_RUN_STATE      1
#define DEBUG_FREE_RUN_STATE        1
#define DEBUG_MEMORY_MAP_WEIGHTS    1
#define DEBUG_READ_CHECKPOINT       1
#define DEBUG_BUILD_TRANSFORMER     1
#define DEBUG_FREE_TRANSFORMER      1
#define DEBUG_RMSNORM               1
#define DEBUG_SOFTMAX               1
#define DEBUG_MATMUL                1
#define DEBUG_FORWARD               1
#define DEBUG_COMPARE_TOKENS        0
#define DEBUG_BUILD_TOKENIZER       1
#define DEBUG_FREE_TOKENIZER        1
#define DEBUG_DECODE                1
#define DEBUG_SAFE_PRINTF           1
#define DEBUG_STR_LOOKUP            1
#define DEBUG_ENCODE                1
#define DEBUG_SAMPLE_ARGMAX         1
#define DEBUG_SAMPLE_MULT           1
#define DEBUG_COMPARE               1
#define DEBUG_SAMPLE_TOPP           1
#define DEBUG_BUILD_SAMPLER         1
#define DEBUG_FREE_SAMPLER          1
#define DEBUG_RANDOM_U32            1
#define DEBUG_RANDOM_F32            1
#define DEBUG_SAMPLE                1
#define DEBUG_TIME_IN_MS            1
#define DEBUG_GENERATE              1
#define DEBUG_READ_STDIN            1
#define DEBUG_CHAT                  1
#define DEBUG_ERROR_USAGE           1
#define DEBUG_MAIN                  1
#endif

// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// TurboQuant: KV-cache vector quantizer, adapted from turboquant.c
//   Zandieh, Daliri, Hadian, Mirrokni, "TurboQuant: Online Vector Quantization
//   with Near-optimal Distortion Rate", arXiv:2504.19874.
// Only the PROD variant (Algorithm 2, unbiased inner-product estimator) is
// used here, to quantize the per-head Key and Value vectors in the KV cache.
// Keys need PROD because attention scores are query.key inner products
// scored directly against the compressed code (tq_query_prepare/tq_score,
// no dequantize); values are reconstructed once per timestep with
// tq_dequant before the attention-weighted sum.
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---- RNG (xoshiro256**), independent of the Sampler's RNG ---- */
typedef struct { uint64_t s[4]; } tq_rng_t;

static inline uint64_t tq_rotl64(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static uint64_t tq_rng_next(tq_rng_t *r) {
    uint64_t res = tq_rotl64(r->s[1] * 5, 7) * 9;
    uint64_t t = r->s[1] << 17;
    r->s[2] ^= r->s[0];
    r->s[3] ^= r->s[1];
    r->s[1] ^= r->s[2];
    r->s[0] ^= r->s[3];
    r->s[2] ^= t;
    r->s[3] = tq_rotl64(r->s[3], 45);
    return res;
}

static void tq_rng_seed(tq_rng_t *r, uint64_t seed) {
    for (int i = 0; i < 4; i++) {
        seed += 0x9E3779B97F4A7C15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        r->s[i] = z ^ (z >> 31);
    }
    for (int i = 0; i < 16; i++) {
        (void)tq_rng_next(r);
    }
}

static inline double tq_rng_uniform(tq_rng_t *r) {
    return ((tq_rng_next(r) >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}

static double tq_rng_normal(tq_rng_t *r) {
    double u1 = tq_rng_uniform(r);
    double u2 = tq_rng_uniform(r);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/* ---- Lloyd-Max codebook against the exact Beta density of Lemma 1 ----
 * Grid/iteration counts are cut down from the turboquant.c reference
 * (400001/500) to keep static scratch and startup cost small for a
 * bare-metal target; still far finer than the <=128 centroids need. */
#define TQ_LLOYD_GRID   20001
#define TQ_LLOYD_ITERS  300
#define TQ_LLOYD_TOL    1e-13

static double tq_lloyd_max_beta(int d, int K, float *cent, float *thresh)
{
    static double x[TQ_LLOYD_GRID];
    static double w[TQ_LLOYD_GRID];
    const int N = TQ_LLOYD_GRID;
    const double dx = 2.0 / (N - 1);
    const double expo = 0.5 * (d - 3);

    double W = 0.0;
    for (int j = 0; j < N; j++) {
        x[j] = -1.0 + dx * j;
        double t = 1.0 - x[j] * x[j];
        w[j] = (t <= 0.0) ? 0.0 : exp(expo * log(t)) * dx;
        W += w[j];
    }
    for (int j = 0; j < N; j++) {
        w[j] /= W;
    }

    if (K <= 1) {
        if (K == 1) {
            cent[0] = 0.0f;
        }
        double c = 0.0;
        for (int j = 0; j < N; j++) {
            c += w[j] * x[j] * x[j];
        }
        return c;
    }

    double *c = (double *)malloc(sizeof(double) * K);
    double *sw = (double *)malloc(sizeof(double) * K);
    double *sx = (double *)malloc(sizeof(double) * K);

    {
        double acc = 0.0;
        int i = 0;
        for (int j = 0; j < N && i < K; j++) {
            acc += w[j];
            while (i < K && acc >= (i + 0.5) / K) {
                c[i++] = x[j];
            }
        }
        while (i < K) {
            c[i++] = x[N - 1];
        }
    }

    for (int it = 0; it < TQ_LLOYD_ITERS; it++) {
        for (int i = 0; i < K; i++) {
            sw[i] = 0.0;
            sx[i] = 0.0;
        }

        int b = 0;
        for (int j = 0; j < N; j++) {
            while (b < K - 1 && x[j] > 0.5 * (c[b] + c[b + 1])) {
                b++;
            }
            sw[b] += w[j];
            sx[b] += w[j] * x[j];
        }

        double delta = 0.0;
        for (int i = 0; i < K; i++) {
            if (sw[i] > 0.0) {
                double nc = sx[i] / sw[i];
                double dd = fabs(nc - c[i]);
                if (dd > delta) {
                    delta = dd;
                }
                c[i] = nc;
            }
        }
        if (delta < TQ_LLOYD_TOL) {
            break;
        }
    }

    double cost = 0.0;
    {
        int b = 0;
        for (int j = 0; j < N; j++) {
            while (b < K - 1 && x[j] > 0.5 * (c[b] + c[b + 1])) {
                b++;
            }
            double e = x[j] - c[b];
            cost += w[j] * e * e;
        }
    }

    for (int i = 0; i < K; i++) {
        cent[i] = (float)c[i];
    }
    for (int i = 0; i < K - 1; i++) {
        thresh[i] = (float)(0.5 * (c[i] + c[i + 1]));
    }

    free(c);
    free(sw);
    free(sx);
    return cost;
}

/* ---- context: Haar rotation Pi (isometry) + QJL Gaussian matrix M ----
 * M := S Pi^T, drawn directly i.i.d. N(0,1) -- see turboquant.c for why
 * that's equivalent to forming S separately. */
typedef struct {
    int d;
    int b;
    int mse_bits;
    int K;
    int prod;
    float *Pi;
    float *M;
    float *cent;
    float *thresh;
    double cost;
    float *y, *yq, *r;
    float prod_scale;  // Precomputed sqrtf(M_PI / 2.0f) / d
    // --- NEW: TIE Fixed-Point Buffers ---
    int16_t *q16;      // Q15 copy of qbuf (for TIE MACs)
    float    inv_sq;   // scale back from Q15
    int      tie_ready;
#if TQ_TIE_SGN
    int16_t *qm16;     // Q15 copy of qbuf + d
    float    inv_sqm;
#endif
} tq_ctx;

void tq_matvec_q15(const int16_t* mat_q15, const float* vec_in, float* vec_out, int d) {
    // 1. Convert input vector to Q15 once
    int16_t vec_q15[d];
    for (int i = 0; i < d; i++) {
        vec_q15[i] = (int16_t)lrintf(vec_in[i] * 32767.0f);
    }

    // Cast arrays to 32-bit pointers so we load 2x 16-bit values at once
    const uint32_t* v32 = (const uint32_t*)vec_q15;
    int pairs = d >> 1; // Number of dual-MACs per row

    // 2. Compute Matrix-Vector product
    for (int r = 0; r < d; r++) {
        const uint32_t* m32 = (const uint32_t*)(mat_q15 + (r * d));
        int32_t acc = 0;

        // THE HARDWARE LOOP: Processes 2 multiplications per cycle
        #pragma unroll 4
        for (int c = 0; c < pairs; c++) {
            // Note: Because acc is a standard 32-bit AR register, XCC returns the updated value
            Q15MAC2(acc, m32[c], v32[c]);
        }

        // 3. Convert accumulated Q15 back to float
        // We multiplied Q15 * Q15, so we divide by (32768 * 32767)
        vec_out[r] = (float)acc * (1.0f / (32768.0f * 32767.0f));
    }
}

/*
static void tq_matvec(const float *A, const float *x, float *y, int d) {
    TQ_TRACE("inside tq_matvec\n");
    for (int i = 0; i < d; i++) {
        const float *a = A + (size_t)i * d;
        double s = 0.0;
        for (int j = 0; j < d; j++) {
            s += (double)a[j] * x[j];
        }
        y[i] = (float)s;
    }
}
*/
void tq_matvec(const float* mat, const float* vec_in, float* vec_out, int d) {
    // 1. Transparently route to the pre-compiled integer arrays!
    // (Note: Arrays may still be named _Q15 in the header, but must be generated with Q12 scale)
    const int16_t* mat_q12 = (mat == TQ_PI_MATRIX) ? TQ_PI_Q15 : TQ_M_Q15;

    // 2. Convert input vector to Q12 once WITH STRICT CLAMPING
    int16_t vec_q12[d] __attribute__((aligned(4)));
    for (int i = 0; i < d; i++) {
        float val = vec_in[i] * 4095.0f; // Scale to Q12
        if (val > 4095.0f) val = 4095.0f;
        else if (val < -4096.0f) val = -4096.0f;
        vec_q12[i] = (int16_t)lrintf(val);
    }

    // Cast arrays to 32-bit pointers so we load 2x 16-bit values at once
    const uint32_t* v32 = (const uint32_t*)vec_q12;
    int pairs = d >> 1;

    // 3. Compute Matrix-Vector product
    for (int r = 0; r < d; r++) {
        const uint32_t* m32 = (const uint32_t*)(mat_q12 + (r * d));
        int32_t acc = 0;

        // THE HARDWARE LOOP: Processes 2 multiplications per cycle
        #pragma unroll 4
        for (int c = 0; c < pairs; c++) {
            // Note: If XCC compiler says "assigning from void",
            // remove "acc = " and just call Q15MAC2(acc, m32[c], v32[c]);
            Q15MAC2(acc, m32[c], v32[c]);
        }

        // Convert accumulated Q24 (Q12 * Q12) back to float
        vec_out[r] = (float)acc * (1.0f / (4096.0f * 4095.0f));
    }
}
/*
static void tq_matvec_T(const float *A, const float *x, float *y, int d) {
    TQ_TRACE("inside tq_matvec_T\n");
    for (int j = 0; j < d; j++) {
        y[j] = 0.0f;
    }
    for (int i = 0; i < d; i++) {
        const float *a = A + (size_t)i * d;
        float xi = x[i];
        if (xi == 0.0f) {
            continue;
        }
        for (int j = 0; j < d; j++) {
            y[j] += a[j] * xi;
        }
    }
}
*/
void tq_matvec_T(const float* mat, const float* vec_in, float* vec_out, int d) {
    // 1. Transparently route to the PRE-TRANSPOSED integer arrays!
    const int16_t* mat_q12_transposed = (mat == TQ_PI_MATRIX) ? TQ_PI_T_Q15 : TQ_M_T_Q15;

    // 2. Convert input vector to Q12 once WITH STRICT CLAMPING
    int16_t vec_q12[d] __attribute__((aligned(4)));
    for (int i = 0; i < d; i++) {
        float val = vec_in[i] * 4095.0f; // Scale to Q12
        if (val > 4095.0f) val = 4095.0f;
        else if (val < -4096.0f) val = -4096.0f;
        vec_q12[i] = (int16_t)lrintf(val);
    }

    const uint32_t* v32 = (const uint32_t*)vec_q12;
    int pairs = d >> 1;

    // 3. Compute Matrix-Vector product
    for (int r = 0; r < d; r++) {
        // Because it's pre-transposed, we just read rows normally!
        const uint32_t* m32 = (const uint32_t*)(mat_q12_transposed + (r * d));
        int32_t acc = 0;

        // THE HARDWARE LOOP: Processes 2 multiplications per cycle
        #pragma unroll 4
        for (int c = 0; c < pairs; c++) {
            Q15MAC2(acc, m32[c], v32[c]);
        }

        // Convert accumulated Q24 (Q12 * Q12) back to float
        vec_out[r] = (float)acc * (1.0f / (4096.0f * 4095.0f));
    }
}


static void tq_gen_haar(float *Pi, int d, tq_rng_t *rng) {
    TQ_TRACE("inside tq_gen_haar\n");
    for (size_t i = 0; i < (size_t)d * d; i++) {
        Pi[i] = (float)tq_rng_normal(rng);
    }
    for (int i = 0; i < d; i++) {
        float *vi = Pi + (size_t)i * d;
        for (int k = 0; k < i; k++) {
            const float *vk = Pi + (size_t)k * d;
            double dot = 0.0;
            for (int j = 0; j < d; j++) {
                dot += (double)vk[j] * vi[j];
            }
            for (int j = 0; j < d; j++) {
                vi[j] -= (float)(dot * vk[j]);
            }
        }
        double nrm = 0.0;
        for (int j = 0; j < d; j++) {
            nrm += (double)vi[j] * vi[j];
        }
        nrm = sqrt(nrm);
        for (int j = 0; j < d; j++) {
            vi[j] = (float)(vi[j] / nrm);
        }
    }
}

#if (TQ_TIE_LEVEL >= 3) || TQ_TIE_SGN
static int tq_init_tie(tq_ctx *c)
{
    if (c->mse_bits != 3) {
        fprintf(stderr, "TIE: built for 3-bit indices, ctx has mse_bits=%d\n", c->mse_bits);
        return -1;
    }
    if ((c->d & 7) != 0) {
        fprintf(stderr, "TIE: d must be a multiple of 8, got %d\n", c->d);
        return -1;
    }
    if (c->K != 8) {
        fprintf(stderr, "TIE: centroid ROM has 8 entries, ctx has K=%d\n", c->K);
        return -1;
    }
    c->q16 = (int16_t *)malloc(sizeof(int16_t) * c->d);
    if (!c->q16) {
        return -1;
    }
#if TQ_TIE_SGN
    c->qm16 = (int16_t *)malloc(sizeof(int16_t) * c->d);
    if (!c->qm16) {
        return -1;
    }
#endif

    printf("[TQ] centroid ROM check -- must match tq_cent_table.tie:\n");
    for (int k = 0; k < c->K; k++) {
        float v = c->cent[k] * 32768.0f;
        if (v >  32767.0f) {
            v =  32767.0f;
        }
        if (v < -32768.0f) {
            v = -32768.0f;
        }
        printf("    16'h%04X,   /* k=%d  %+.6f */\n",
               (unsigned)(uint16_t)(int16_t)lrintf(v), k, c->cent[k]);
    }
    c->tie_ready = 1;
    return 0;
}
#endif

#if USE_HEADER_FOR_TQ_CONSTANTS
static int tq_init(tq_ctx *c, int d, int b, int prod, uint64_t seed)
{
    TQ_TRACE("tq_init from constants(headers) \n");
    memset(c, 0, sizeof(*c));
    // Automatically checks against the header file we generated
    if (d != TQ_EXPECTED_D || b != TQ_EXPECTED_B) {
        fprintf(stderr, "Error: Header constants compiled for d=%d b=%d, but model requires d=%d b=%d\n",
                TQ_EXPECTED_D, TQ_EXPECTED_B, d, b);
        return -1;
    }

    c->d = d;
    c->b = b;
    c->prod = prod;
    c->prod_scale = sqrtf((float)(M_PI / 2.0)) / (float)d;
    c->mse_bits = prod ? b - 1 : b;
    c->K = 1 << c->mse_bits;

    // Point directly to the static arrays defined in tq_constants.h
    c->Pi     = (float*)TQ_PI_MATRIX;
    c->M      = (float*)TQ_M_MATRIX;
    c->cent   = (float*)TQ_CENTROIDS;
    c->thresh = (float*)TQ_THRESHOLDS;

    // Allocate only the temporary working buffers
    c->y      = (float *)malloc(sizeof(float) * d);
    c->yq     = (float *)malloc(sizeof(float) * d);
    c->r      = (float *)malloc(sizeof(float) * d);

    if (!c->y || !c->yq || !c->r) {
        return -1;
    }

    // --- NEW: Init TIE buffers ---
#if (TQ_TIE_LEVEL >= 3) || TQ_TIE_SGN
    if (tq_init_tie(c) != 0) {
        return -1;
    }
#endif

    // We no longer calculate c->cost here, as the matrices are precomputed.
    return 0;
}
#else
static int tq_init(tq_ctx *c, int d, int b, int prod, uint64_t seed)
{
    TQ_TRACE("tq_init RUNTIME (no constants, no headers) \n");
    memset(c, 0, sizeof(*c));
    if (d < 4 || b < 1 || b > 8) {
        return -1;
    }

    c->d = d;
    c->b = b;
    c->prod = prod;
    c->prod_scale = sqrtf((float)(M_PI / 2.0)) / (float)d;
    c->mse_bits = prod ? b - 1 : b;
    c->K = 1 << c->mse_bits;

    c->Pi     = (float *)malloc(sizeof(float) * (size_t)d * d);
    c->cent   = (float *)malloc(sizeof(float) * (c->K > 0 ? c->K : 1));
    c->thresh = (float *)malloc(sizeof(float) * (c->K > 1 ? c->K - 1 : 1));
    c->y      = (float *)malloc(sizeof(float) * d);
    c->yq     = (float *)malloc(sizeof(float) * d);
    c->r      = (float *)malloc(sizeof(float) * d);

    if (!c->Pi || !c->cent || !c->thresh || !c->y || !c->yq || !c->r) {
        return -1;
    }

    tq_rng_t rng;
    tq_rng_seed(&rng, seed);
    tq_gen_haar(c->Pi, d, &rng);

    if (prod) {
        c->M = (float *)malloc(sizeof(float) * (size_t)d * d);
        if (!c->M) {
            return -1;
        }
        for (size_t i = 0; i < (size_t)d * d; i++) {
            c->M[i] = (float)tq_rng_normal(&rng);
        }
    }

    c->cost = tq_lloyd_max_beta(d, c->K, c->cent, c->thresh);

#if (TQ_TIE_LEVEL >= 3) || TQ_TIE_SGN
    if (tq_init_tie(c) != 0) {
        return -1;
    }
#endif

    return 0;
}
#endif

static void tq_free(tq_ctx *c) {
#if USE_HEADER_FOR_TQ_CONSTANTS
    TQ_TRACE("inside tq_free with TQ cosntants\n");
    // Only free the temporary working buffers
    free(c->y);
    free(c->yq);
    free(c->r);
#else
    TQ_TRACE("inside tq_free without TQ cosntants\n");
    // Free everything that was dynamically allocated
    free(c->Pi);
    if (c->prod) {
        free(c->M);
    }
    free(c->cent);
    free(c->thresh);
    free(c->y);
    free(c->yq);
    free(c->r);
#endif
#if (TQ_TIE_LEVEL >= 3) || TQ_TIE_SGN
    if (c->q16) {
        free(c->q16);
    }
#if TQ_TIE_SGN
    if (c->qm16) {
        free(c->qm16);
    }
#endif
#endif
    memset(c, 0, sizeof(*c));
}

/* code layout:  MSE:  [norm][ idx bits ]
 *               PROD: [norm][gamma][ idx bits ][ sign bits ] */
static size_t tq_idx_bytes(const tq_ctx *c) {
    return ((size_t)c->d * c->mse_bits + 7) / 8;
}

static size_t tq_sgn_bytes(const tq_ctx *c) {
    return c->prod ? (((size_t)c->d + 7) / 8) : 0;
}

static size_t tq_code_size(const tq_ctx *c) {
    TQ_TRACE("inside tq_code_size\n");
    return sizeof(float) * (c->prod ? 2 : 1) + tq_idx_bytes(c) + tq_sgn_bytes(c);
}

static inline void tq_bits_put(uint8_t *buf, size_t pos, int nbits, uint32_t v) {
    TQ_TRACE("inside tq_bits_put\n");
    for (int i = 0; i < nbits; i++) {
        size_t p = pos + i;
        if ((v >> i) & 1u) {
            buf[p >> 3] |=  (uint8_t)(1u << (p & 7));
        } else {
            buf[p >> 3] &= (uint8_t)~(1u << (p & 7));
        }
    }
}

static inline uint32_t tq_bits_get(const uint8_t *buf, size_t pos, int nbits) {
    TQ_TRACE("inside tq_bits_get\n");
    uint32_t v = 0;
    for (int i = 0; i < nbits; i++) {
        size_t p = pos + i;
        v |= (uint32_t)((buf[p >> 3] >> (p & 7)) & 1u) << i;
    }
    return v;
}

static inline int tq_quant_scalar(const tq_ctx *c, float v) {
    TQ_TRACE("inside tq_quant_scalar\n");
    int k = 0;
    while (k < c->K - 1 && v > c->thresh[k]) {
        k++;
    }
    return k;
}

/* Algorithm 1/2: y = Pi x ; idx_j = argmin_k |y_j - c_k| ; (PROD only) then
 * r' = y - yq, gamma = ||r'||, qjl = sign(M r'). x need not be unit-norm. */
static void tq_quant(tq_ctx *c, const float *x, uint8_t *code)
{
    TQ_TRACE("inside tq_quant\n");
    const int d = c->d;
    memset(code, 0, tq_code_size(c));

    double n2 = 0.0;
    for (int j = 0; j < d; j++) {
        n2 += (double)x[j] * x[j];
    }
    float nrm = (float)sqrt(n2);

    if (nrm == 0.0f) {
        memcpy(code, &nrm, sizeof(float));
        return;
    }

    for (int j = 0; j < d; j++) {
        c->r[j] = (float)(x[j] / nrm);
    }
    tq_matvec(c->Pi, c->r, c->y, d);

    uint8_t *p = code;
    memcpy(p, &nrm, sizeof(float));
    p += sizeof(float);
    uint8_t *gamma_slot = NULL;
    if (c->prod) {
        gamma_slot = p;
        p += sizeof(float);
    }
    uint8_t *idx = p;
    uint8_t *sgn = p + tq_idx_bytes(c);

    for (int j = 0; j < d; j++) {
        int k = (c->K > 1) ? tq_quant_scalar(c, c->y[j]) : 0;
        c->yq[j] = (c->K >= 1) ? c->cent[k] : 0.0f;
        if (c->mse_bits > 0) {
            tq_bits_put(idx, (size_t)j * c->mse_bits, c->mse_bits, (uint32_t)k);
        }
    }

    if (!c->prod) {
        return;
    }

    double g2 = 0.0;
    for (int j = 0; j < d; j++) {
        c->r[j] = c->y[j] - c->yq[j];
        g2 += (double)c->r[j] * c->r[j];
    }
    float gamma = (float)sqrt(g2);
    memcpy(gamma_slot, &gamma, sizeof(float));

    for (int i = 0; i < d; i++) {
        const float *m = c->M + (size_t)i * d;
        double s = 0.0;
        for (int j = 0; j < d; j++) {
            s += (double)m[j] * c->r[j];
        }
        if (s >= 0.0) {
            sgn[i >> 3] |= (uint8_t)(1u << (i & 7));
        }
    }
}

/* MSE : x~ = ||x|| * Pi^T yq
 * PROD: x~ = ||x|| * Pi^T ( yq + sqrt(pi/2)/d * gamma * M^T qjl ) */
static void tq_dequant(tq_ctx *c, const uint8_t *code, float *xr)
{
    TQ_TRACE("inside tq_dequant\n");
    const int d = c->d;
    const uint8_t *p = code;
    float nrm;
    memcpy(&nrm, p, sizeof(float));
    p += sizeof(float);
    float gamma = 0.0f;
    if (c->prod) {
        memcpy(&gamma, p, sizeof(float));
        p += sizeof(float);
    }
    const uint8_t *idx = p;
    const uint8_t *sgn = p + tq_idx_bytes(c);

    if (nrm == 0.0f) {
        for (int j = 0; j < d; j++) {
            xr[j] = 0.0f;
        }
        return;
    }

    for (int j = 0; j < d; j++) {
        int k = (c->mse_bits > 0) ? (int)tq_bits_get(idx, (size_t)j * c->mse_bits, c->mse_bits) : 0;
        c->yq[j] = (c->K >= 1) ? c->cent[k] : 0.0f;
    }

    if (c->prod) {
        const double scale = sqrt(M_PI / 2.0) / d * gamma;
        for (int j = 0; j < d; j++) {
            c->r[j] = 0.0f;
        }
        for (int i = 0; i < d; i++) {
            float s = ((sgn[i >> 3] >> (i & 7)) & 1u) ? 1.0f : -1.0f;
            const float *m = c->M + (size_t)i * d;
            for (int j = 0; j < d; j++) {
                c->r[j] += m[j] * s;
            }
        }
        for (int j = 0; j < d; j++) {
            c->yq[j] += (float)(scale * c->r[j]);
        }
    }

    tq_matvec_T(c->Pi, c->yq, xr, d);
    for (int j = 0; j < d; j++) {
        xr[j] *= nrm;
    }
}

/* Fast scoring: rotate the query once (qbuf must hold 2*d floats), then
 * score each cached code in O(d) without dequantizing it. */
static void tq_query_prepare(tq_ctx *c, const float *q, float *qbuf)
{
    TQ_TRACE("inside tq_query_prepare\n");
    tq_matvec(c->Pi, q, qbuf, c->d);
    if (c->prod) {
        tq_matvec(c->M, qbuf, qbuf + c->d, c->d);
    }

    // --- NEW: Q15 Fixed-Point translation for TIE ---
#if TQ_TIE_LEVEL >= 3
    {
        const int d = c->d;
        float mx = 0.0f;
        for (int j = 0; j < d; j++) {
            float a = fabsf(qbuf[j]);
            if (a > mx) {
                mx = a;
            }
        }
        if (mx > 0.0f) {
            const float sc = 32767.0f / mx;
            c->inv_sq = mx / 32767.0f;
            for (int j = 0; j < d; j++) {
            	float val = qbuf[j] * sc;
            	if (val > 32767.0f) val = 32767.0f;
            	else if (val < -32768.0f) val = -32768.0f;
            	c->q16[j] = (int16_t)lrintf(val);
            }
        } else {
            c->inv_sq = 0.0f;
            memset(c->q16, 0, sizeof(int16_t) * d);
        }
    }
#endif
#if TQ_TIE_SGN
    if (c->prod) {
        const int d = c->d;
        const float *qm = qbuf + d;
        float mx = 0.0f;
        for (int i = 0; i < d; i++) {
            float a = fabsf(qm[i]);
            if (a > mx) {
                mx = a;
            }
        }
        if (mx > 0.0f) {
            const float sc = 32767.0f / mx;
            c->inv_sqm = mx / 32767.0f;
            for (int i = 0; i < d; i++) {
            	float val = qm[i] * sc;
            	if (val > 32767.0f) val = 32767.0f;
            	else if (val < -32768.0f) val = -32768.0f;
            	c->qm16[i] = (int16_t)lrintf(val);
            }
        } else {
            c->inv_sqm = 0.0f;
            memset(c->qm16, 0, sizeof(int16_t) * d);
        }
    }
#endif
}

static float tq_score(const tq_ctx *c, const uint8_t *code, const float *qbuf)
{
    TQ_TRACE("inside tq_score\n");
    const int d = c->d;
    const uint8_t *p = code;
    float nrm;
    memcpy(&nrm, p, sizeof(float));
    p += sizeof(float);
    float gamma = 0.0f;
    if (c->prod) {
        memcpy(&gamma, p, sizeof(float));
        p += sizeof(float);
    }
    const uint8_t *idx = p;
    const uint8_t *sgn = p + tq_idx_bytes(c);

    if (nrm == 0.0f) {
        return 0.0f;
    }

    double s = 0.0;

    // --- NEW: TIE Accelerated Scoring ---
#if TQ_TIE_LEVEL >= 3
    {
        const uint32_t *q32 = (const uint32_t *)c->q16;
        const int ngroups = d >> 3; // 8 coords per 3 bytes

        TQ_ACC_CLR();
        for (int g = 0; g < ngroups; g++) {
            const uint8_t *ip = idx + 3 * g;
            TQ_IDX_LOAD((uint32_t)ip[0] | ((uint32_t)ip[1] << 8) | ((uint32_t)ip[2] << 16));
            const uint32_t *qp = q32 + (g << 2);
            TQ_DECMAC2(qp[0]);
            TQ_DECMAC2(qp[1]);
            TQ_DECMAC2(qp[2]);
            TQ_DECMAC2(qp[3]);
        }
        s = (double)((float)(int32_t)TQ_ACC_RD() * c->inv_sq);
    }
#else
    // Fallback standard C
    if (c->mse_bits > 0) {
        for (int j = 0; j < d; j++) {
            int k = (int)tq_bits_get(idx, (size_t)j * c->mse_bits, c->mse_bits);
            s += (double)qbuf[j] * c->cent[k];
        }
    }
#endif

    // Sign term
    if (c->prod) {
#if TQ_TIE_SGN
        const uint32_t *qm32 = (const uint32_t *)c->qm16;
        TQ_SACC_CLR();
        for (int i = 0; i < d; i += 32) {
            int n = d - i;
            if (n > 32) {
                n = 32;
            }
            uint32_t w = 0;
            for (int bi = 0; bi < ((n + 7) >> 3); bi++) {
                w |= (uint32_t)sgn[(i >> 3) + bi] << (8 * bi);
            }
            TQ_SGN_LOAD(w);
            const uint32_t *qp = qm32 + (i >> 1);
            for (int pi = 0; pi < (n >> 1); pi++) {
                TQ_SGNACC2(qp[pi]);
            }
        }
        double t = (double)((float)(int32_t)TQ_SACC_RD() * c->inv_sqm);
#else
        const float *qm = qbuf + d;
        double t = 0.0;
        for (int i = 0; i < d; i++) {
            t += ((sgn[i >> 3] >> (i & 7)) & 1u) ? qm[i] : -qm[i];
        }
#endif
        s += (double)(c->prod_scale * gamma * (float)t);
    }

    return (float)(s * nrm);
}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Transformer model

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

typedef struct {
    int8_t* q;    // quantized values
    float* s; // scaling factors
} QuantizedTensor;

typedef struct {
    // token embedding table
    QuantizedTensor *q_tokens; // (vocab_size, dim)
    float* token_embedding_table; // same, but dequantized

    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    QuantizedTensor *wq; // (layer, dim, n_heads * head_size)
    QuantizedTensor *wk; // (layer, dim, n_kv_heads * head_size)
    QuantizedTensor *wv; // (layer, dim, n_kv_heads * head_size)
    QuantizedTensor *wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    QuantizedTensor *w1; // (layer, hidden_dim, dim)
    QuantizedTensor *w2; // (layer, dim, hidden_dim)
    QuantizedTensor *w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    QuantizedTensor *wcls;
} TransformerWeights;

typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    QuantizedTensor xq; // quantized x (dim,)
    QuantizedTensor hq; // quantized hb (hidden_dim,)
    float *q; // query (dim,)
    float *k; // key (dim,)
    float *v; // value (dim,)
    float *att; // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits
    // kv cache: packed TurboQuant codes, (layer, seq_len, n_kv_heads, kv_code_size bytes)
    uint8_t* key_cache;
    uint8_t* value_cache;
    // TurboQuant scratch
    float* v_dequant; // head_size, reconstructed value vector for the weighted sum
    float* tq_qbuf;   // 2*head_size, rotated query for tq_score
    // --- NEW: Factored Value Path Buffers ---
    float* tq_accy;   // head_size
    float* tq_accs;   // head_size
    float* tq_tmp;    // head_size
    // --- NEW: Precomputed RoPE Tables ---
    float* rope_cos;  // seq_len * (head_size / 2)
    float* rope_sin;  // seq_len * (head_size / 2)
} RunState;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    float* data; // malloc'd data pointer replacing memory map
    size_t file_size; // size of the checkpoint file in bytes
    tq_ctx kv_tq;          // TurboQuant context shared by keys and values
    size_t kv_code_size;   // bytes per cached head-vector (== tq_code_size(&kv_tq))
} Transformer;

void malloc_run_state(RunState* s, Config* p, size_t kv_code_size) {
#if DEBUG_MALLOC_RUN_STATE
    printf("[DEBUG] malloc_run_state(dim=%d, hidden_dim=%d, n_layers=%d, n_heads=%d, n_kv_heads=%d, vocab_size=%d, seq_len=%d)\n", p->dim, p->hidden_dim, p->n_layers, p->n_heads, p->n_kv_heads, p->vocab_size, p->seq_len);
#endif
    // we calloc instead of malloc to keep valgrind happy
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int head_size = p->dim / p->n_heads;
    size_t kv_slots = (size_t)p->n_layers * p->seq_len * p->n_kv_heads;
    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->xq = (QuantizedTensor) { .q = calloc(p->dim, sizeof(int8_t)), .s = calloc(p->dim, sizeof(float)) };
    s->hq = (QuantizedTensor) { .q = calloc(p->hidden_dim, sizeof(int8_t)), .s = calloc(p->hidden_dim, sizeof(float)) };
    s->q = calloc(p->dim, sizeof(float));
    s->k = calloc(kv_dim, sizeof(float));
    s->v = calloc(kv_dim, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    s->key_cache = calloc(kv_slots * kv_code_size, sizeof(uint8_t));
    s->value_cache = calloc(kv_slots * kv_code_size, sizeof(uint8_t));
    s->v_dequant = calloc(head_size, sizeof(float));
    s->tq_qbuf = calloc(2 * head_size, sizeof(float));

    // Factored Path allocations
    s->tq_accy = calloc(head_size, sizeof(float));
    s->tq_accs = calloc(head_size, sizeof(float));
    s->tq_tmp  = calloc(head_size, sizeof(float));

    // --- NEW: Precompute RoPE LUT efficiently ---
    int half_head = head_size / 2;
    s->rope_cos = malloc(p->seq_len * half_head * sizeof(float));
    s->rope_sin = malloc(p->seq_len * half_head * sizeof(float));
    float *freqs = malloc(half_head * sizeof(float));

    for (int j = 0; j < half_head; j++) {
        freqs[j] = 1.0f / powf(10000.0f, (j * 2) / (float)head_size);
    }
    for (int p_idx = 0; p_idx < p->seq_len; p_idx++) {
        for (int j = 0; j < half_head; j++) {
            float val = p_idx * freqs[j];
            s->rope_cos[p_idx * half_head + j] = cosf(val);
            s->rope_sin[p_idx * half_head + j] = sinf(val);
        }
    }
    free(freqs);

    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->k || !s->v || !s->att || !s->logits || !s->key_cache
     || !s->value_cache || !s->v_dequant || !s->tq_qbuf
     || !s->tq_accy || !s->tq_accs || !s->tq_tmp
     || !s->rope_cos || !s->rope_sin) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState* s) {
#if DEBUG_FREE_RUN_STATE
    printf("[DEBUG] free_run_state()\n");
#endif
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->xq.q);
    free(s->xq.s);
    free(s->hq.q);
    free(s->hq.s);
    free(s->q);
    free(s->k);
    free(s->v);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
    free(s->v_dequant);
    free(s->tq_qbuf);
    free(s->tq_accy);
    free(s->tq_accs);
    free(s->tq_tmp);
    // --- NEW: Free RoPE LUT ---
    free(s->rope_cos);
    free(s->rope_sin);
}

// ----------------------------------------------------------------------------
// Quantization functions
#if FP_SUPPORTED
void dequantize(QuantizedTensor *qx, float* x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = qx->q[i] * qx->s[i / GS];
    }
}
#else
//no floating point
void dequantize(QuantizedTensor *qx, float* x, int n) {
    int num_groups = n / GS;
    for (int group = 0; group < num_groups; group++) {
        // Fetch the scale factor once per group!
        float scale = qx->s[group];

        // Process the group without any division
        for (int i = 0; i < GS; i++) {
            x[group * GS + i] = qx->q[group * GS + i] * scale;
        }
    }
}
#endif

void quantize(QuantizedTensor *qx, float* x, int n) {
    int num_groups = n / GS;
    float Q_MAX = 127.0f;

    for (int group = 0; group < num_groups; group++) {

        // find the max absolute value in the current group
        float wmax = 0.0;
        for (int i = 0; i < GS; i++) {
            float val = fabs(x[group * GS + i]);
            if (val > wmax) {
                wmax = val;
            }
        }

        // calculate and write the scaling factor
        float scale = wmax / Q_MAX;
        qx->s[group] = scale;

        // calculate and write the quantized values
        for (int i = 0; i < GS; i++) {
            float quant_value = x[group * GS + i] / scale; // scale
            int8_t quantized = (int8_t) round(quant_value); // round and clamp
            qx->q[group * GS + i] = quantized;
        }
    }
}

/* initialize `n` x quantized tensor (with `size_each` elements), starting from memory pointed at *ptr */
QuantizedTensor *init_quantized_tensors(void **ptr, int n, int size_each) {
    void *p = *ptr;
    QuantizedTensor *res = malloc(n * sizeof(QuantizedTensor));
    for(int i=0; i<n; i++) {
        /* map quantized int8 values*/
        res[i].q = (int8_t*)p;
        p = (int8_t*)p + size_each;
        /* map scale factors */
        res[i].s = (float*)p;
        p = (float*)p + size_each / GS;
    }
    *ptr = p; // advance ptr to current position
    return res;
}

void memory_map_weights(TransformerWeights *w, Config* p, void* ptr, uint8_t shared_classifier) {
#if DEBUG_MEMORY_MAP_WEIGHTS
    printf("[DEBUG] memory_map_weights(shared_classifier=%d)\n", shared_classifier);
#endif
    int head_size = p->dim / p->n_heads;
    // first are the parameters that are kept in fp32 (the rmsnorm (1D) weights)
    float* fptr = (float*) ptr; // cast our pointer to float*
    w->rms_att_weight = fptr;
    fptr += p->n_layers * p->dim;
    w->rms_ffn_weight = fptr;
    fptr += p->n_layers * p->dim;
    w->rms_final_weight = fptr;
    fptr += p->dim;

    // now read all the quantized weights
    ptr = (void*)fptr; // now cast the pointer back to void*
    w->q_tokens = init_quantized_tensors(&ptr, 1, p->vocab_size * p->dim);
    // dequantize token embedding table
    //w->token_embedding_table = malloc(p->vocab_size * p->dim * sizeof(float));
    //dequantize(w->q_tokens, w->token_embedding_table, p->vocab_size * p->dim);
    w->token_embedding_table = NULL; // No longer allocated!

    w->wq = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_heads * head_size));
    w->wk = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * head_size));
    w->wv = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * head_size));
    w->wo = init_quantized_tensors(&ptr, p->n_layers, (p->n_heads * head_size) * p->dim);

    w->w1 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);
    w->w2 = init_quantized_tensors(&ptr, p->n_layers, p->hidden_dim * p->dim);
    w->w3 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);

    w->wcls = shared_classifier ? w->q_tokens : init_quantized_tensors(&ptr, 1, p->dim * p->vocab_size);
}

void read_checkpoint(const char* checkpoint, Config* config, TransformerWeights* weights,
                     float** data, size_t* file_size) {
#if DEBUG_READ_CHECKPOINT
    printf("[DEBUG] read_checkpoint(checkpoint='%s')\n", checkpoint);
#endif
#if USE_HEADER_FILES
    // 1. Point to the static header array
    unsigned char* ptr = MODEL_PTR_bin ; // MATCH NAME IN model_data.h
    *file_size = MODEL_SIZE_bin;    // MATCH NAME IN model_data.h
    *data = NULL; // Prevent free() from crashing later

    // 2. Read Magic Number
    uint32_t magic_number;
    memcpy(&magic_number, ptr, sizeof(uint32_t));
    ptr += sizeof(uint32_t);
    if (magic_number != 0x616b3432) {
        fprintf(stderr, "Bad magic number\n");
        exit(EXIT_FAILURE);
    }

    // 3. Read Version
    int version;
    memcpy(&version, ptr, sizeof(int));
    ptr += sizeof(int);
    if (version != 2) {
        fprintf(stderr, "Bad version\n");
        exit(EXIT_FAILURE);
    }

    // 4. Read Config
    memcpy(config, ptr, sizeof(Config));
    ptr += sizeof(Config);

    // 5. Read Quantization Flags
    uint8_t shared_classifier;
    memcpy(&shared_classifier, ptr, sizeof(uint8_t));
    ptr += sizeof(uint8_t);

    int group_size;
    memcpy(&group_size, ptr, sizeof(int));
    ptr += sizeof(int);
    GS = group_size;

    // 6. Map the rest of the weights
    int header_size = 256;
    void* weights_ptr = (void*)(MODEL_PTR_bin + header_size);
    memory_map_weights(weights, config, weights_ptr, shared_classifier);

#else

    FILE *file = fopen(checkpoint, "rb");
    if (!file) {
        fprintf(stderr, "Couldn't open file %s\n", checkpoint);
        exit(EXIT_FAILURE);
    }
    // read in magic number (uint32), has to be 0x616b3432, i.e. "ak42" in ASCII
    uint32_t magic_number;
    if (fread(&magic_number, sizeof(uint32_t), 1, file) != 1) {
        exit(EXIT_FAILURE);
    }
    if (magic_number != 0x616b3432) {
        fprintf(stderr, "Bad magic number\n");
        exit(EXIT_FAILURE);
    }
    // read in the version number (uint32), has to be 2
    int version;
    if (fread(&version, sizeof(int), 1, file) != 1) {
        exit(EXIT_FAILURE);
    }
    if (version != 2) {
        fprintf(stderr, "Bad version %d, need version 2\n", version);
        exit(EXIT_FAILURE);
    }
    int header_size = 256; // the header size for version 2 in bytes
    // read in the Config
    if (fread(config, sizeof(Config), 1, file) != 1) {
        exit(EXIT_FAILURE);
    }
    // read in flags
    uint8_t shared_classifier; // a byte to indicate if the classifier is shared
    if (fread(&shared_classifier, sizeof(uint8_t), 1, file) != 1) {
        exit(EXIT_FAILURE);
    }
    int group_size; // the group size used in quantization
    if (fread(&group_size, sizeof(int), 1, file) != 1) {
        exit(EXIT_FAILURE);
    }
    GS = group_size; // set as global, as it will be used in many places

    // figure out the file size
    fseek(file, 0, SEEK_END); // move file pointer to end of file
    *file_size = (size_t)ftell(file); // get the file size, in bytes
    fseek(file, 0, SEEK_SET); // rewind to start

    // allocate physical memory to hold the file
    *data = (float*)malloc(*file_size);
    if (!*data) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }

    // read the entire file into RAM
    if (fread(*data, 1, *file_size, file) != *file_size) {
        fprintf(stderr, "read failed!\n");
        exit(EXIT_FAILURE);
    }
    fclose(file);

    // point to the weights past the header
    void* weights_ptr = ((char*)*data) + header_size;
    memory_map_weights(weights, config, weights_ptr, shared_classifier);
#endif
}

void build_transformer(Transformer *t, const char* checkpoint_path) {
#if DEBUG_BUILD_TRANSFORMER
    printf("[DEBUG] build_transformer(checkpoint_path='%s')\n", checkpoint_path);
#endif
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->data, &t->file_size);
    // set up the TurboQuant KV-cache quantizer (one context shared by K and V,
    // sized to a single attention head)
    int head_size = t->config.dim / t->config.n_heads;
    if (tq_init(&t->kv_tq, head_size, TQ_BITS, /*prod=*/1, TQ_SEED) != 0) {
        fprintf(stderr, "tq_init failed (head_size=%d, bits=%d)\n", head_size, TQ_BITS);
        exit(EXIT_FAILURE);
    }
    t->kv_code_size = tq_code_size(&t->kv_tq);
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config, t->kv_code_size);
    int kv_dim = (t->config.dim * t->config.n_kv_heads) / t->config.n_heads;
    size_t unquantized_bytes = (size_t)t->config.n_layers * t->config.seq_len * kv_dim * sizeof(float) * 2;
    size_t tq_slots = (size_t)t->config.n_layers * t->config.seq_len * t->config.n_kv_heads;
    size_t turboquant_bytes = tq_slots * t->kv_code_size * 2;

    printf("\n[MEMORY SAVINGS ANALYSIS]\n");
    printf("----------------------------------------\n");
    printf("Unquantized FP32 KV Cache: %.2f MB\n", (float)unquantized_bytes / (1024.0f * 1024.0f));
    printf("TurboQuant KV Cache (%d bits): %.2f MB\n", TQ_BITS, (float)turboquant_bytes / (1024.0f * 1024.0f));
    printf("Memory Reduction Factor:   %.2fx\n", (float)unquantized_bytes / (float)turboquant_bytes);
    printf("----------------------------------------\n\n");
}

void free_transformer(Transformer* t) {
#if DEBUG_FREE_TRANSFORMER
    printf("[DEBUG] free_transformer()\n");
#endif
    // free QuantizedTensors
    free(t->weights.q_tokens);
    free(t->weights.token_embedding_table);
    free(t->weights.wq);
    free(t->weights.wk);
    free(t->weights.wv);
    free(t->weights.wo);
    free(t->weights.w1);
    free(t->weights.w2);
    free(t->weights.w3);
    if(t->weights.wcls != t->weights.q_tokens) {
        free(t->weights.wcls);
    }
    // free the malloc'd buffer holding the model
    if (t->data) {
        free(t->data);
    }
    // free the RunState buffers
    free_run_state(&t->state);
    // free the TurboQuant context
    tq_free(&t->kv_tq);
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

void rmsnorm(float* o, float* x, float* weight, int size) {
#if DEBUG_RMSNORM
    printf("[DEBUG] rmsnorm(size=%d)\n", size);
#endif
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void softmax(float* x, int size) {
#if DEBUG_SOFTMAX
    printf("[DEBUG] softmax(size=%d)\n", size);
#endif
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

    void matmul(float* xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d) {
#if DEBUG_MATMUL
    printf("[DEBUG] matmul(n=%d, d=%d)\n", n, d);
#endif
        int i;
        for (i = 0; i < d; i++) {
            float val = 0.0f;
            int in = i * n;

            // Calculate base group indices ONCE per row
            int w_group = in / GS;
            int x_group = 0;

            for (int j = 0; j <= n - GS; j += GS) {
                int32_t ival = 0;

                // Cast byte buffers to 32-bit word pointers (4 bytes per word)
                const uint32_t *xq32 = (const uint32_t *)(x->q + j);
                const uint32_t *wq32 = (const uint32_t *)(w->q + in + j);
                int words_per_group = GS >> 2; // e.g., 64 / 4 = 16 words

                #pragma unroll 4
                for (int k = 0; k < words_per_group; k++) {
                    // 1 instruction replaces 4 loads, 4 sign-extends, 4 mults, and 4 adds!
                    SDOT4(ival, xq32[k], wq32[k]);
                }

                // Use the incrementing counters instead of hardware division
                val += ((float) ival) * w->s[w_group] * x->s[x_group];
                w_group++;
                x_group++;
            }

            xout[i] = val;
        }
    }

float* forward(Transformer* transformer, int token, int pos) {
#if DEBUG_FORWARD
    printf("[DEBUG] forward(token=%d, pos=%d)\n", token, pos);
#endif

    // a few convenience variables
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;
    const float inv_sqrt_head_size = 1.0f / sqrtf((float)head_size);

    // copy the token embedding into x
    //memcpy(x, w->token_embedding_table + token*dim, dim * sizeof(float));
    //replaced to lazy embedding instead all
    int group_start = (token * dim) / GS;
    int offset = token * dim;
        for (int i = 0; i < dim; i++) {
            float scale = w->q_tokens->s[group_start + (i / GS)];
            x[i] = w->q_tokens->q[offset + i] * scale;
        }
    // forward all the layers
    for(int l = 0; l < p->n_layers; l++) {

        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        // qkv matmuls for this position
        quantize(&s->xq, s->xb, dim);
        matmul(s->q, &s->xq, w->wq + l, dim, dim);
        matmul(s->k, &s->xq, w->wk + l, dim, kv_dim);
        matmul(s->v, &s->xq, w->wv + l, dim, kv_dim);

        // RoPE relative positional encoding: instant lookups from precomputed LUT
        assert(pos < p->seq_len && "Position exceeds RoPE LUT size");
        int half_head = head_size >> 1;
        const float* cos_row = s->rope_cos + pos * half_head;
        const float* sin_row = s->rope_sin + pos * half_head;

        for (int i = 0; i < dim; i+=2) {
            int h_idx = (i % head_size) >> 1;
            float fcr = cos_row[h_idx];
            float fci = sin_row[h_idx];
            int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k; // the vector to rotate (query or key)
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }

        // save key,value at this time step (pos) to our kv cache, one TurboQuant
        // code per kv-head (each code covers a head_size-length sub-vector)
        size_t loff_codes = (size_t)l * p->seq_len * p->n_kv_heads; // kv cache layer offset, in code slots
        uint8_t* key_cache_row = s->key_cache + (loff_codes + (size_t)pos * p->n_kv_heads) * transformer->kv_code_size;
        uint8_t* value_cache_row = s->value_cache + (loff_codes + (size_t)pos * p->n_kv_heads) * transformer->kv_code_size;
        for (int kh = 0; kh < p->n_kv_heads; kh++) {
            tq_quant(&transformer->kv_tq, s->k + kh * head_size, key_cache_row + (size_t)kh * transformer->kv_code_size);
            tq_quant(&transformer->kv_tq, s->v + kh * head_size, value_cache_row + (size_t)kh * transformer->kv_code_size);
        }

        // multihead attention. iterate over all heads
        int h;
        // #pragma omp parallel for private(h)
        for (h = 0; h < p->n_heads; h++) {
            // get the query vector for this head
            float* q = s->q + h * head_size;
            // attention scores for this head
            float* att = s->att + h * p->seq_len;
            int kvh = h / kv_mul; // which kv-head this query head reads from

            // rotate the query once for this head; tq_score reuses it for every timestep
            tq_query_prepare(&transformer->kv_tq, q, s->tq_qbuf);

            // iterate over all time, including the current one
            for (int t = 0; t <= pos; t++) {
                // get the quantized key code for this head and at this timestep
                const uint8_t* kcode = s->key_cache + (loff_codes + (size_t)t * p->n_kv_heads + kvh) * transformer->kv_code_size;
                // score directly against the code, no dequantize (TurboQuant PROD)
                float score = tq_score(&transformer->kv_tq, kcode, s->tq_qbuf);
                score *= inv_sqrt_head_size;
                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att, pos + 1);

            // --- NEW: Factored Value Path ---
            memset(s->tq_accy, 0, head_size * sizeof(float));
            memset(s->tq_accs, 0, head_size * sizeof(float));

            tq_ctx *kvc = &transformer->kv_tq;
            const float c_scale = sqrtf(M_PI / 2.0f) / head_size;
            int pairs_count = head_size >> 1;

#if TQ_TIE_LEVEL >= 4
            // Declare TIE registers (no xt_ prefix based on your header)
            TQ_VACC acc_regs[32];
            // Clear the hardware accumulators ONCE before the loop
            for (int i = 0; i < pairs_count; i++) {
                acc_regs[i] = TQ_VACC_CLR();
            }
#endif

            for (int t = 0; t <= pos; t++) {
                float a = att[t];
                if (a == 0.0f) {
                    continue; // Skip empty attention weights
                }

                const uint8_t* vcode = s->value_cache + (loff_codes + (size_t)t * p->n_kv_heads + kvh) * transformer->kv_code_size;

                // Read norms and bounds manually
                float nrm;
                memcpy(&nrm, vcode, sizeof(float));
                if (nrm == 0.0f) {
                    continue;
                }

                float gamma = 0.0f;
                const uint8_t *idx_ptr = vcode + sizeof(float);
                if (kvc->prod) {
                    memcpy(&gamma, idx_ptr, sizeof(float));
                    idx_ptr += sizeof(float);
                }
                const uint8_t *sgn_ptr = idx_ptr + tq_idx_bytes(kvc);

                // --- TIE Step 4: Index Hoisting & TQ_SCALEACC2 Preparation ---
                // Unpack the indices into pairs ONCE per timestep
                uint8_t hoisted_pairs[32]; // Accommodates up to head_size=64
                for (int j = 0; j < pairs_count; j++) {
                    uint32_t idx0 = tq_bits_get(idx_ptr, (size_t)(j * 2) * kvc->mse_bits, kvc->mse_bits);
                    uint32_t idx1 = tq_bits_get(idx_ptr, (size_t)(j * 2 + 1) * kvc->mse_bits, kvc->mse_bits);
                    hoisted_pairs[j] = (uint8_t)((idx1 << 3) | idx0);
                }

                float w_norm = a * nrm;

#if TQ_TIE_LEVEL >= 4
                // Hardware accumulation via TQ_SCALEACC2
                int16_t w_q15 = (int16_t)lrintf(w_norm * 32767.0f);
                if (w_q15 != 0) {
                    for (int i = 0; i < pairs_count; i++) {
                        // TQ_SCALEACC2 updates the register in place
                        TQ_SCALEACC2(acc_regs[i], w_q15, hoisted_pairs[i]);
                    }
                }
#else
                // C fallback matching the hardware logic
                for (int i = 0; i < pairs_count; i++) {
                    int idx0 = hoisted_pairs[i] & 0x7;
                    int idx1 = (hoisted_pairs[i] >> 3) & 0x7;
                    s->tq_accy[i * 2]     += w_norm * kvc->cent[idx0];
                    s->tq_accy[i * 2 + 1] += w_norm * kvc->cent[idx1];
                }
#endif

                // Accumulate signs
                if (kvc->prod) {
                    float w_sgn = w_norm * c_scale * gamma;
                    for (int i = 0; i < head_size; i += 8) {
                        uint8_t b = sgn_ptr[i >> 3];

                        s->tq_accs[i+0] += (b & 1) ? w_sgn : -w_sgn;
                        b >>= 1;
                        s->tq_accs[i+1] += (b & 1) ? w_sgn : -w_sgn;
                        b >>= 1;
                        s->tq_accs[i+2] += (b & 1) ? w_sgn : -w_sgn;
                        b >>= 1;
                        s->tq_accs[i+3] += (b & 1) ? w_sgn : -w_sgn;
                        b >>= 1;
                        s->tq_accs[i+4] += (b & 1) ? w_sgn : -w_sgn;
                        b >>= 1;
                        s->tq_accs[i+5] += (b & 1) ? w_sgn : -w_sgn;
                        b >>= 1;
                        s->tq_accs[i+6] += (b & 1) ? w_sgn : -w_sgn;
                        b >>= 1;
                        s->tq_accs[i+7] += (b & 1) ? w_sgn : -w_sgn;
                    }
                }
            } // END OF TIMESTEP LOOP

#if TQ_TIE_LEVEL >= 4
            // Read the accumulated values back to C floats ONCE after the loop.
            //const float inv_scale = 1.0f / (32767.0f * 32768.0f);
            const float inv_scale = 1.0f / 32767.0f;
            for (int i = 0; i < pairs_count; i++) {
                s->tq_accy[i * 2]     = (float)((int32_t)TQ_VACC_RD0(acc_regs[i])) * inv_scale;
                s->tq_accy[i * 2 + 1] = (float)((int32_t)TQ_VACC_RD1(acc_regs[i])) * inv_scale;
            }
#endif

            // OUTSIDE THE LOOP: Apply the dense matrix multiplications exactly ONCE
            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));

            if (kvc->prod) {
                tq_matvec_T(kvc->M, s->tq_accs, s->tq_tmp, head_size);
                for (int j = 0; j < head_size; j++) {
                    s->tq_accy[j] += s->tq_tmp[j];
                }
            }
            tq_matvec_T(kvc->Pi, s->tq_accy, xb, head_size);
        }

        // final matmul to get the output of the attention
        quantize(&s->xq, s->xb, dim);
        matmul(s->xb2, &s->xq, w->wo + l, dim, dim);

        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        quantize(&s->xq, s->xb, dim);
        matmul(s->hb, &s->xq, w->w1 + l, dim, hidden_dim);
        matmul(s->hb2, &s->xq, w->w3 + l, dim, hidden_dim);

        // SwiGLU non-linearity
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val)));
            // elementwise multiply with w3(x)
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        // final matmul to get the output of the ffn
        quantize(&s->hq, s->hb, hidden_dim);
        matmul(s->xb, &s->hq, w->w2 + l, hidden_dim, dim);

        // residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    quantize(&s->xq, x, dim);
    matmul(s->logits, &s->xq, w->wcls, dim, p->vocab_size);

    return s->logits;
}
// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

int compare_tokens(const void *a, const void *b) {
#if DEBUG_COMPARE_TOKENS
    printf("[DEBUG] compare_tokens(str_a='%s', str_b='%s')\n", ((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
#endif
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size) {
#if DEBUG_BUILD_TOKENIZER
    printf("[DEBUG] build_tokenizer(tokenizer_path='%s', vocab_size=%d)\n", tokenizer_path, vocab_size);
#endif
    // i should have written the vocab_size into the tokenizer file... sigh
    t->vocab_size = vocab_size;
    // malloc space to hold the scores and the strings
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL; // initialized lazily
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
#if USE_HEADER_FILES
    unsigned char* ptr = TOKEINZER_PTR_bin; // MATCH NAME IN tokenizer_data.h
    // read in the file
    // Read max token length
    memcpy(&t->max_token_length, ptr, sizeof(int));
    ptr += sizeof(int);

    int len;
    for (int i = 0; i < vocab_size; i++) {
        // Read score
        memcpy(t->vocab_scores + i, ptr, sizeof(float));
        ptr += sizeof(float);

        // Read string length
        memcpy(&len, ptr, sizeof(int));
        ptr += sizeof(int);

        // Read string and append null terminator
        t->vocab[i] = (char *)malloc(len + 1);
        memcpy(t->vocab[i], ptr, len);
        ptr += len;
        t->vocab[i][len] = '\0';
    }
#else
    // read in the file
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) {
        fprintf(stderr, "couldn't load %s\n", tokenizer_path);
        exit(EXIT_FAILURE);
    }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) {
        fprintf(stderr, "failed read\n");
        exit(EXIT_FAILURE);
    }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) {
            fprintf(stderr, "failed read\n");
            exit(EXIT_FAILURE);
        }
        if (fread(&len, sizeof(int), 1, file) != 1) {
            fprintf(stderr, "failed read\n");
            exit(EXIT_FAILURE);
        }
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) {
            fprintf(stderr, "failed read\n");
            exit(EXIT_FAILURE);
        }
        t->vocab[i][len] = '\0'; // add the string terminating token
    }
    fclose(file);
#endif
}

void free_tokenizer(Tokenizer* t) {
#if DEBUG_FREE_TOKENIZER
    printf("[DEBUG] free_tokenizer()\n");
#endif
    for (int i = 0; i < t->vocab_size; i++) {
        free(t->vocab[i]);
    }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

char* decode(Tokenizer* t, int prev_token, int token) {
#if DEBUG_DECODE
    printf("[DEBUG] decode(prev_token=%d, token=%d)\n", prev_token, token);
#endif
    char *piece = t->vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') {
        piece++;
    }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(char *piece) {
#if DEBUG_SAFE_PRINTF
    printf("[DEBUG] safe_printf(piece='%s')\n", piece ? piece : "NULL");
#endif
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece == NULL) {
        return;
    }
    if (piece[0] == '\0') {
        return;
    }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // bad byte, don't print it
        }
    }
    printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
#if DEBUG_STR_LOOKUP
    printf("[DEBUG] str_lookup(str='%s', vocab_size=%d)\n", str, vocab_size);
#endif
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = { .str = str }; // acts as the key to search for
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
#if DEBUG_ENCODE
    printf("[DEBUG] encode(text='%s', bos=%d, eos=%d)\n", text ? text : "NULL", bos, eos);
#endif
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == NULL) {
        fprintf(stderr, "cannot encode NULL text\n");
        exit(EXIT_FAILURE);
    }

    if (t->sorted_vocab == NULL) {
        // lazily malloc and sort the vocabulary
        t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    char* str_buffer = malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    size_t str_len = 0;

    // start at 0 tokens
    *n_tokens = 0;

    // add optional BOS (=1) token, if desired
    if (bos) {
        tokens[(*n_tokens)++] = 1;
    }

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point <-> UTF-8 conversion
    // First code point    Last code point    Byte 1    Byte 2    Byte 3    Byte 4
    // U+0000    U+007F        0xxxxxxx
    // U+0080    U+07FF        110xxxxx    10xxxxxx
    // U+0800    U+FFFF        1110xxxx    10xxxxxx    10xxxxxx
    // U+10000    U+10FFFF    11110xxx    10xxxxxx    10xxxxxx    10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (char *c = text; *c != '\0'; c++) {

        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((*c & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; // token length decreased
    }

    // add optional EOS (=2) token, if desired
    if (eos) {
        tokens[(*n_tokens)++] = 2;
    }

    free(str_buffer);
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
    int vocab_size;
    ProbIndex* probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float* probabilities, int n) {
#if DEBUG_SAMPLE_ARGMAX
    printf("[DEBUG] sample_argmax(n=%d)\n", n);
#endif
    // return the index that has the highest probability
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
#if DEBUG_SAMPLE_MULT
    printf("[DEBUG] sample_mult(n=%d, coin=%f)\n", n, coin);
#endif
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

int compare(const void* a, const void* b) {
#if DEBUG_COMPARE
    printf("[DEBUG] compare(prob_a=%f, prob_b=%f)\n", ((ProbIndex*)a)->prob, ((ProbIndex*)b)->prob);
#endif
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
#if DEBUG_SAMPLE_TOPP
    printf("[DEBUG] sample_topp(n=%d, topp=%f, coin=%f)\n", n, topp, coin);
#endif
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    int n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of rounding errors
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
#if DEBUG_BUILD_SAMPLER
    printf("[DEBUG] build_sampler(vocab_size=%d, temperature=%f, topp=%f, rng_seed=%llu)\n", vocab_size, temperature, topp, rng_seed);
#endif
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    // buffer only used with nucleus sampling; may not need but it's ~small
    sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* sampler) {
#if DEBUG_FREE_SAMPLER
    printf("[DEBUG] free_sampler()\n");
#endif
    free(sampler->probindex);
}

unsigned int random_u32(unsigned long long *state) {
#if DEBUG_RANDOM_U32
    printf("[DEBUG] random_u32()\n");
#endif
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
#if DEBUG_RANDOM_F32
    printf("[DEBUG] random_f32()\n");
#endif
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler* sampler, float* logits) {
#if DEBUG_SAMPLE
    printf("[DEBUG] sample()\n");
#endif
    // sample the token given the logits and some hyperparameters
    int next;
    if (sampler->temperature == 0.0f) {
        // greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        // apply the temperature to the logits
        for (int q=0; q<sampler->vocab_size; q++) {
            logits[q] /= sampler->temperature;
        }
        // apply softmax to the logits to get the probabilities for next token
        softmax(logits, sampler->vocab_size);
        // flip a (float) coin (this is our source of entropy for sampling)
        float coin = random_f32(&sampler->rng_state);
        // we sample from this distribution to get the next token
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            // simply sample from the predicted probability distribution
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

// ----------------------------------------------------------------------------
// utilities: time

long time_in_ms() {
#if DEBUG_TIME_IN_MS
    printf("[DEBUG] time_in_ms()\n");
#endif
    // return time in milliseconds, for benchmarking the model speed
    return (long)(((double)clock() / CLOCKS_PER_SEC) * 1000);
}

// ----------------------------------------------------------------------------
// generation loop

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
#if DEBUG_GENERATE
    printf("[DEBUG] generate(prompt='%s', steps=%d)\n", prompt ? prompt : "NULL", steps);
#endif
    char *empty_prompt = "";
    if (prompt == NULL) {
        prompt = empty_prompt;
    }

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    // start the main loop
    long start = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;     // position in the sequence
    while (pos < steps) {

        // forward the transformer to get logits for the next token
        float* logits = forward(transformer, token, pos);

        // advance the state state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            next = sample(sampler, logits);
        }
        pos++;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (next == 1 || next == 2) {
            break;
        }

        // print the token as string, decode it with the Tokenizer object
        char* piece = decode(tokenizer, token, next);
        safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
        fflush(stdout);
        token = next;

        // init the timer here because the first iteration can be slower
        if (start == 0) {
            start = time_in_ms();
        }
    }
    printf("\n");

    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
    }

    free(prompt_tokens);
}

void read_stdin(const char* guide, char* buffer, size_t bufsize) {
#if DEBUG_READ_STDIN
    printf("[DEBUG] read_stdin(guide='%s', bufsize=%zu)\n", guide, bufsize);
#endif
    // read a line from stdin, up to but not including \n
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0'; // strip newline
        }
    }
}

// ----------------------------------------------------------------------------
// chat loop
// I manually inspected the tokens for a few chat conversations compared to
// python reference and that seemed ok, but this was not thoroughly tested and
// is not safely implemented, it's more a proof of concept atm.

void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
          char *cli_user_prompt, char *cli_system_prompt, int steps) {

#if DEBUG_CHAT
    printf("[DEBUG] chat(cli_user_prompt='%s', cli_system_prompt='%s', steps=%d)\n", cli_user_prompt ? cli_user_prompt : "NULL", cli_system_prompt ? cli_system_prompt : "NULL", steps);
#endif
    // buffers for reading the system prompt and user prompt from stdin
    // you'll notice they are soomewhat haphazardly and unsafely set atm
    char system_prompt[512];
    char user_prompt[512];
    char rendered_prompt[1152];
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc(1152 * sizeof(int));
    int user_idx;

    // start the main loop
    int8_t user_turn = 1; // user starts
    int next;        // will store the next token in the sequence
    int token;       // stores the current token to feed into the transformer
    int prev_token;
    int pos = 0;     // position in the sequence
    while (pos < steps) {

        // when it is the user's turn to contribute tokens to the dialog...
        if (user_turn) {
            // get the (optional) system prompt at position 0
            if (pos == 0) {
                // at position 0, the user can also contribute a system prompt
                if (cli_system_prompt == NULL) {
                    // system prompt was not passed in, attempt to get it from stdin
                    read_stdin("Enter system prompt (optional): ", system_prompt, sizeof(system_prompt));
                } else {
                    // system prompt was passed in, use it
                    strcpy(system_prompt, cli_system_prompt);
                }
            }
            // get the user prompt
            if (pos == 0 && cli_user_prompt != NULL) {
                // user prompt for position 0 was passed in, use it
                strcpy(user_prompt, cli_user_prompt);
            } else {
                // otherwise get user prompt from stdin
                read_stdin("User: ", user_prompt, sizeof(user_prompt));
            }
            // render user/system prompts into the Llama 2 Chat schema
            if (pos == 0 && system_prompt[0] != '\0') {
                char system_template[] = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]";
                sprintf(rendered_prompt, system_template, system_prompt, user_prompt);
            } else {
                char user_template[] = "[INST] %s [/INST]";
                sprintf(rendered_prompt, user_template, user_prompt);
            }
            // encode the rendered prompt into tokens
            encode(tokenizer, rendered_prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
            user_idx = 0; // reset the user index
            user_turn = 0;
            printf("Assistant: ");
        }

        // determine the token to pass into the transformer next
        if (user_idx < num_prompt_tokens) {
            // if we are still processing the input prompt, force the next prompt token
            token = prompt_tokens[user_idx++];
        } else {
            // otherwise use the next token sampled from previous turn
            token = next;
        }
        // EOS (=2) token ends the Assistant turn
        if (token == 2) {
            user_turn = 1;
        }

        // forward the transformer to get logits for the next token
        float* logits = forward(transformer, token, pos);
        next = sample(sampler, logits);
        pos++;

        if (user_idx >= num_prompt_tokens && next != 2) {
            // the Assistant is responding, so print its output
            char* piece = decode(tokenizer, token, next);
            safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
            fflush(stdout);
        }
        if (next == 2) {
            printf("\n");
        }
    }
    printf("\n");
    free(prompt_tokens);
}

// ----------------------------------------------------------------------------
// CLI

void error_usage() {
#if DEBUG_ERROR_USAGE
    printf("[DEBUG] error_usage()\n");
#endif
    fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
    fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
    fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
    // Cast to void to prevent "unused parameter" compiler warnings
    (void)argc;
    (void)argv;
    long t_start, t_end;
#if DEBUG_MAIN
    printf("[DEBUG] main(argc=%d)\n", argc);
#endif

    // Hardcode parameters
#if USE_HEADER_FILES
    char *checkpoint_path = NULL; // for now defined in the function itslef :/
    char *tokenizer_path  = NULL;
#else

    char *checkpoint_path = MODEL_PARAMS_BIN;
    char *tokenizer_path  = MODEL_TOKENIZER_BIN;

#endif
    char *prompt          = EXAMPALE_PROMPT_FOR_MODEL;
    float temperature = 0.0f;
    float topp        = 0.9f;
#if XTENSA_RUN
    int steps         = 10; //was 256
#else
    int steps         = 10; //was 256
#endif
    unsigned long long rng_seed = 0;
    char *mode        = "generate";
    char *system_prompt = NULL;

    // parameter validation/overrides
    if (rng_seed <= 0) {
        rng_seed = (unsigned int)time(NULL);
    }
    if (temperature < 0.0) {
        temperature = 0.0;
    }
    if (topp < 0.0 || 1.0 < topp) {
        topp = 0.9;
    }
    if (steps < 0) {
        steps = 0;
    }

    t_start = time_in_ms();
    // build the Transformer via the model .bin file
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    t_end = time_in_ms();
    printf("[PROFILE] Model loaded in: %ld ms\n", t_end - t_start);
    if (steps == 0 || steps > transformer.config.seq_len) {
        steps = transformer.config.seq_len; // override to ~max length
    }

    // build the Tokenizer via the tokenizer .bin file
    t_start = time_in_ms();
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);
    t_end = time_in_ms();
    printf("[PROFILE] Tokenizer loaded in: %ld ms\n", t_end - t_start);

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    // run!
    if (strcmp(mode, "generate") == 0) {
        generate(&transformer, &tokenizer, &sampler, prompt, steps);
    } else if (strcmp(mode, "chat") == 0) {
        chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, steps);
    } else {
        fprintf(stderr, "unknown mode: %s\n", mode);
        error_usage();
    }

    // memory and file handles cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}
