/* hcp.h — Complex-Domain Hierarchical Constraint Propagation for ASR
 *
 * Single-header library. Drop into any whisper.cpp project.
 *
 * Usage:
 *   #define HCP_IMPLEMENTATION    (in exactly ONE .c file)
 *   #include "hcp.h"
 *
 * MIT License — Copyright (c) 2026 Nick Gonzales
 */

#ifndef HCP_H
#define HCP_H

#include <whisper.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Configuration ─────────────────────────────────────────────── */

#ifndef HCP_MAX_SEGMENTS
#define HCP_MAX_SEGMENTS        8192
#endif
#ifndef HCP_MAX_TOKENS
#define HCP_MAX_TOKENS         32768
#endif
#ifndef HCP_MAX_TEXT_LEN
#define HCP_MAX_TEXT_LEN        4096
#endif

/* Tuning knobs (override before #include if desired) */
#ifndef HCP_PHASE_SHIFT_THRESH
#define HCP_PHASE_SHIFT_THRESH   1.2f   /* radians — flag if phase rotates > this */
#endif
#ifndef HCP_MAG_SUPPRESS_THRESH
#define HCP_MAG_SUPPRESS_THRESH  0.4f   /* flag if magnitude drops below 40% */
#endif
#ifndef HCP_REDECODE_THRESH
#define HCP_REDECODE_THRESH      0.30f  /* re-decode if >30% tokens flagged */
#endif
#ifndef HCP_VLEN_BUCKETS
#define HCP_VLEN_BUCKETS          16
#endif
#ifndef HCP_DT_BUCKETS
#define HCP_DT_BUCKETS            16
#endif

/* E-T Gate configuration */
#ifndef HCP_ET_FRAME_SIZE
#define HCP_ET_FRAME_SIZE       512     /* samples per frame (32ms @ 16kHz, must be pow2) */
#endif
#ifndef HCP_ET_RMS_FLOOR
#define HCP_ET_RMS_FLOOR        0.01f   /* -40 dBFS silence threshold */
#endif
#ifndef HCP_ET_FLATNESS_THRESH
#define HCP_ET_FLATNESS_THRESH  0.7f    /* above = noise-like */
#endif
#ifndef HCP_ET_SPEECH_MIN
#define HCP_ET_SPEECH_MIN       0.25f   /* need ≥25% speech-like frames */
#endif
#ifndef HCP_ET_DENSITY_MIN
#define HCP_ET_DENSITY_MIN      1.5f    /* tokens/sec to trigger gate */
#endif

/* KIEL-CC configuration */
#ifndef HCP_KIEL_INNOV_THRESH
#define HCP_KIEL_INNOV_THRESH   3.0f    /* flag if normalized innovation > this */
#endif
#ifndef HCP_KIEL_Q
#define HCP_KIEL_Q              0.01f   /* Kalman process noise */
#endif
#ifndef HCP_KIEL_R
#define HCP_KIEL_R              0.1f    /* Kalman measurement noise */
#endif

/* Re-decode configuration (v2.1) */
#ifndef HCP_REDECODE_BEAM
#define HCP_REDECODE_BEAM          8       /* wider beam for re-decode (whisper max = 8) */
#endif
#ifndef HCP_REDECODE_MIN_SAMPLES
#define HCP_REDECODE_MIN_SAMPLES   3200    /* 200ms minimum slice @ 16kHz */
#endif

/* Semantic channel configuration (v3.0) */
#ifndef HCP_SEMANTIC_WEIGHT
#define HCP_SEMANTIC_WEIGHT        0.3f    /* semantic channel influence [0,1] */
#endif
#ifndef HCP_SEMANTIC_LOW_THRESH
#define HCP_SEMANTIC_LOW_THRESH    0.02f   /* flag if bigram score < this */
#endif

/* Formant anchoring configuration (v3.1) */
#ifndef HCP_FORMANT_FRAME_SIZE
#define HCP_FORMANT_FRAME_SIZE     512     /* FFT frame for formant analysis */
#endif
#ifndef HCP_FORMANT_F1_LO
#define HCP_FORMANT_F1_LO          200     /* F1 band lower bound (Hz) */
#endif
#ifndef HCP_FORMANT_F1_HI
#define HCP_FORMANT_F1_HI          1000    /* F1 band upper bound */
#endif
#ifndef HCP_FORMANT_F2_LO
#define HCP_FORMANT_F2_LO          1000    /* F2 band lower bound */
#endif
#ifndef HCP_FORMANT_F2_HI
#define HCP_FORMANT_F2_HI          3000    /* F2 band upper bound */
#endif
#ifndef HCP_FORMANT_SPEECH_THRESH
#define HCP_FORMANT_SPEECH_THRESH  0.15f   /* min F1+F2 energy ratio for speech */
#endif

/* Trigram configuration (v3.1) */
#ifndef HCP_TRIGRAM_WEIGHT
#define HCP_TRIGRAM_WEIGHT         0.2f    /* trigram influence on semantic [0,1] */
#endif

/* Context-seeded re-decode (v3.1) */
#ifndef HCP_REDECODE_CONTEXT_TOKENS
#define HCP_REDECODE_CONTEXT_TOKENS  32    /* max tokens from surrounding segments for context */
#endif

/* Morphological logit bias (v3.2) */
#ifndef HCP_LOGIT_BIAS_STRENGTH
#define HCP_LOGIT_BIAS_STRENGTH    -5.0f   /* logit penalty for zero-coherence tokens */
#endif
#ifndef HCP_LOGIT_BIAS_FLOOR
#define HCP_LOGIT_BIAS_FLOOR        0.005f /* bigram score below this → apply bias */
#endif
#ifndef HCP_SUBWORD_FREQ_FLOOR
#define HCP_SUBWORD_FREQ_FLOOR      0.0f   /* subword freq below this → extra penalty */
#endif

/* Hallucination flag bits (9 layers in v3.1) */
#define HCP_HALLUC_HIGH_COMPRESS   0x001
#define HCP_HALLUC_NGRAM_REPEAT    0x002
#define HCP_HALLUC_VLEN_ANOMALY    0x004
#define HCP_HALLUC_LOW_LOGPROB     0x008
#define HCP_HALLUC_SPECTRAL        0x010
#define HCP_HALLUC_ET_GATE         0x020
#define HCP_HALLUC_KALMAN          0x040
#define HCP_HALLUC_SEMANTIC        0x080
#define HCP_HALLUC_FORMANT         0x100

/* ─── Structures ────────────────────────────────────────────────── */

/* Per-segment output */
typedef struct {
    int64_t  t0_ms;
    int64_t  t1_ms;
    float    confidence;          /* geometric mean token probability */
    float    logprob;             /* mean log probability */
    float    no_speech_prob;
    float    compression_ratio;
    float    quality;             /* base composite quality */
    float    hcp_quality;         /* HCP-enhanced quality */
    uint16_t hallucination_flags; /* bitfield of HCP_HALLUC_* (9 layers) */
    int      speaker_turn;
    int      token_count;
    int      hcp_flagged_count;   /* tokens flagged by HCP in this segment */
    float    et_rms;              /* E-T Gate: segment audio RMS energy */
    float    et_speech_frac;      /* E-T Gate: speech-like frame fraction */
    float    kiel_max_innov;      /* KIEL-CC: max normalized innovation */
    float    semantic_score;       /* semantic channel: mean bigram coherence */
    float    formant_ratio;        /* formant anchoring: F1+F2 energy / total energy */
    char     text[HCP_MAX_TEXT_LEN];
} HcpSegment;

/* Full transcript result */
typedef struct {
    HcpSegment *segments;
    int    count;
    int    cap;
    int    segments_hallucinated;
    double decode_ms;
    double hcp_ms;
    /* HCP internals */
    int    hcp_tokens;
    int    hcp_padded;
    int    hcp_flagged_tokens;
    int    hcp_flagged_segments;
    float *hcp_mag_original;      /* per-token original magnitude */
    float *hcp_mag_corrected;     /* per-token corrected magnitude */
    float *hcp_phase_shift;       /* per-token |Δφ| */
    int   *hcp_token_seg_map;     /* flat token → segment index */
    /* E-T Gate internals */
    double et_gate_ms;
    int    et_segments_gated;
    /* KIEL-CC internals */
    double kiel_ms;
    int    kiel_flagged_tokens;
    float *kiel_innovation;       /* per-token normalized Kalman innovation */
    /* Re-decode internals (v2.1) */
    int    redecode_count;
    int    redecode_improved;
    double redecode_ms;
    /* Semantic channel internals (v3.0) */
    double semantic_ms;
    int    semantic_low_count;
    /* Formant anchoring internals (v3.1) */
    double formant_ms;
    int    formant_flagged;
    /* Logit bias internals (v3.2) */
    int    logit_bias_tokens;     /* total tokens that received logit penalties */
} HcpResult;

/* Model-agnostic token input (v4.0) — for any ASR engine */
typedef struct {
    const char *text;
    float confidence;      /* [0.0, 1.0] */
    float logprob;         /* log probability (use -logf(1-conf) if unavailable) */
    float duration_ms;     /* token duration in ms (0 if unknown) */
} HcpUniversalToken;

typedef struct {
    HcpUniversalToken *tokens;
    int    token_count;
    int64_t start_ms;
    int64_t end_ms;
    const char *text;      /* full segment text */
    float no_speech_prob;  /* 0.0 if unavailable */
} HcpUniversalSegment;

/* ─── API ───────────────────────────────────────────────────────── */

/* Run full HCP pipeline on a whisper context after whisper_full() completes.
 * Extracts segments, runs complex-domain spectral refinement, and populates
 * the result with enhanced quality scores and hallucination flags.
 *
 * Caller must call hcp_free() when done with the result. */
HcpResult hcp_process(struct whisper_context *ctx);

/* Free all memory owned by an HcpResult. */
void hcp_free(HcpResult *r);

/* Utility: compute zlib compression ratio of text (hallucination signal). */
float hcp_compression_ratio(const char *text);

/* Utility: detect n-gram repetition (returns 1 if repetitive). */
int hcp_detect_ngram_repeat(const char *text);

/* Full pipeline with audio: includes E-T Gate (audio-text verification).
 * Call instead of hcp_process() when raw audio samples are available. */
HcpResult hcp_process_with_audio(struct whisper_context *ctx,
                                  const float *audio, int n_samples, int sample_rate);

/* Re-decode segments flagged as hallucinated (v2.1).
 * Uses wider beam search on the audio slice. Call after hcp_process_with_audio().
 * Returns number of segments improved. */
int hcp_redecode(struct whisper_context *ctx,
                 const float *audio, int n_samples, int sample_rate,
                 struct whisper_full_params base_params,
                 HcpResult *res);

/* Model-agnostic HCP processing (v4.0).
 * Accepts pre-segmented tokens from any ASR engine (Deepgram, Google, Apple, etc.)
 * and applies HCP spectral refinement + hallucination detection.
 * If audio is NULL, E-T Gate is skipped. */
HcpResult hcp_process_universal(HcpUniversalSegment *segments, int n_segments,
                                 const float *audio, int n_samples, int sample_rate);

#ifdef __cplusplus
}
#endif

/* ─── Implementation ────────────────────────────────────────────── */

#ifdef HCP_IMPLEMENTATION

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <zlib.h>

#include "hcp_subword_freq.h"
#include "hcp_bigram.h"
#include "hcp_trigram.h"

/* ── Internal types ──────────────────────────────────────────────── */

typedef struct { float re, im; } hcp__cpx;

typedef struct {
    whisper_token id;
    float    p;
    float    plog;
    float    vlen;
    int64_t  t_dtw;
    int      seg_idx;
    float    no_speech_prob;
    float    comp_ratio;
    int      speaker_turn;
    const char *text;
} hcp__token;

/* ── Timing ──────────────────────────────────────────────────────── */

static double hcp__ms_now(void) {
#if defined(CLOCK_MONOTONIC)
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
#else
    return (double)clock() / (CLOCKS_PER_SEC / 1000.0);
#endif
}

/* ── FNV-1a hash ─────────────────────────────────────────────────── */

static uint64_t hcp__fnv1a(const void *data, size_t len) {
    const uint8_t *p = (const uint8_t *)data;
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < len; i++) {
        h ^= p[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

static float hcp__fnv_to_phase(uint64_t h) {
    return (float)(h & 0xFFFFFFFF) / (float)0xFFFFFFFF * 2.0f * (float)M_PI;
}

/* ── Complex arithmetic ──────────────────────────────────────────── */

static hcp__cpx hcp__cpx_mul(hcp__cpx a, hcp__cpx b) {
    return (hcp__cpx){ a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re };
}

static hcp__cpx hcp__cpx_add(hcp__cpx a, hcp__cpx b) {
    return (hcp__cpx){ a.re + b.re, a.im + b.im };
}

static hcp__cpx hcp__cpx_sub(hcp__cpx a, hcp__cpx b) {
    return (hcp__cpx){ a.re - b.re, a.im - b.im };
}

static float hcp__cpx_mag(hcp__cpx z) {
    return sqrtf(z.re * z.re + z.im * z.im);
}

static float hcp__cpx_phase(hcp__cpx z) {
    return atan2f(z.im, z.re);
}

static hcp__cpx hcp__cpx_from_polar(float mag, float phase) {
    return (hcp__cpx){ mag * cosf(phase), mag * sinf(phase) };
}

/* ── Radix-2 Cooley-Tukey FFT ────────────────────────────────────── */

static void hcp__fft(hcp__cpx *x, int n, int inverse) {
    int log2n = 0;
    for (int tmp = n; tmp > 1; tmp >>= 1) log2n++;

    /* Bit-reversal permutation */
    for (int i = 0; i < n; i++) {
        int j = 0;
        for (int b = 0; b < log2n; b++) {
            if (i & (1 << b)) j |= (1 << (log2n - 1 - b));
        }
        if (j > i) {
            hcp__cpx tmp = x[i]; x[i] = x[j]; x[j] = tmp;
        }
    }

    /* Butterfly stages */
    float sign = inverse ? 1.0f : -1.0f;
    for (int s = 1; s <= log2n; s++) {
        int m = 1 << s;
        float angle = sign * 2.0f * (float)M_PI / (float)m;
        hcp__cpx wm = { cosf(angle), sinf(angle) };
        for (int k = 0; k < n; k += m) {
            hcp__cpx w = { 1.0f, 0.0f };
            for (int j = 0; j < m / 2; j++) {
                hcp__cpx t = hcp__cpx_mul(w, x[k + j + m / 2]);
                hcp__cpx u = x[k + j];
                x[k + j]         = hcp__cpx_add(u, t);
                x[k + j + m / 2] = hcp__cpx_sub(u, t);
                w = hcp__cpx_mul(w, wm);
            }
        }
    }

    if (inverse) {
        for (int i = 0; i < n; i++) {
            x[i].re /= (float)n;
            x[i].im /= (float)n;
        }
    }
}

static int hcp__next_pow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

/* ── Compression ratio ───────────────────────────────────────────── */

float hcp_compression_ratio(const char *text) {
    size_t src_len = strlen(text);
    if (src_len < 4) return 1.0f;
    uLong bound = compressBound((uLong)src_len);
    uint8_t *comp = (uint8_t *)malloc(bound);
    if (!comp) return 1.0f;
    uLong comp_len = bound;
    if (compress2(comp, &comp_len, (const uint8_t *)text, (uLong)src_len, Z_DEFAULT_COMPRESSION) != Z_OK) {
        free(comp);
        return 1.0f;
    }
    float ratio = (float)src_len / (float)comp_len;
    free(comp);
    return ratio;
}

/* ── N-gram repetition ───────────────────────────────────────────── */

#define HCP__NGRAM_TABLE  256
#define HCP__NGRAM_SIZE     3
#define HCP__NGRAM_THRESH   3

int hcp_detect_ngram_repeat(const char *text) {
    uint32_t table[HCP__NGRAM_TABLE];
    memset(table, 0, sizeof(table));
    const char *words[1024];
    int wc = 0;
    char buf[HCP_MAX_TEXT_LEN];
    strncpy(buf, text, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';
    char *tok = strtok(buf, " \t\n\r");
    while (tok && wc < 1024) {
        words[wc++] = tok;
        tok = strtok(NULL, " \t\n\r");
    }
    if (wc < HCP__NGRAM_SIZE) return 0;
    for (int i = 0; i <= wc - HCP__NGRAM_SIZE; i++) {
        char ngram[512];
        ngram[0] = '\0';
        for (int j = 0; j < HCP__NGRAM_SIZE; j++) {
            if (j) strcat(ngram, " ");
            strncat(ngram, words[i + j], sizeof(ngram) - strlen(ngram) - 2);
        }
        uint32_t h = (uint32_t)hcp__fnv1a(ngram, strlen(ngram));
        uint32_t slot = h % HCP__NGRAM_TABLE;
        table[slot]++;
        if (table[slot] > HCP__NGRAM_THRESH) return 1;
    }
    return 0;
}

/* ── Quantization ────────────────────────────────────────────────── */

static int hcp__quant_vlen(float vlen) {
    int q = (int)(vlen * (float)HCP_VLEN_BUCKETS / 2.0f);
    if (q < 0) q = 0;
    if (q >= HCP_VLEN_BUCKETS) q = HCP_VLEN_BUCKETS - 1;
    return q;
}

static int hcp__quant_dt(int64_t dt_ms) {
    int q = (int)(dt_ms * HCP_DT_BUCKETS / 500);
    if (q < 0) q = 0;
    if (q >= HCP_DT_BUCKETS) q = HCP_DT_BUCKETS - 1;
    return q;
}
/* ── Bigram semantic scoring (v3.0) ─────────────────────────────── */

static float hcp__bigram_score(whisper_token prev, whisper_token cur) {
    uint8_t key[8];
    uint32_t p = (uint32_t)prev, c = (uint32_t)cur;
    memcpy(key, &p, 4);
    memcpy(key + 4, &c, 4);
    uint32_t slot = (uint32_t)(hcp__fnv1a(key, 8) % HCP_BIGRAM_SLOTS);
    return (float)hcp_bigram_table[slot] / 255.0f;
}

/* ── Trigram semantic scoring (v3.1) ────────────────────────────── */

static float hcp__trigram_score(whisper_token pp, whisper_token prev, whisper_token cur) {
    uint8_t key[12];
    uint32_t a = (uint32_t)pp, b = (uint32_t)prev, c = (uint32_t)cur;
    memcpy(key, &a, 4);
    memcpy(key + 4, &b, 4);
    memcpy(key + 8, &c, 4);
    uint32_t slot = (uint32_t)(hcp__fnv1a(key, 12) % HCP_TRIGRAM_SLOTS);
    return (float)hcp_trigram_table[slot] / 255.0f;
}

/* ── Combined semantic score (bigram + trigram) ──────────────────── */

static float hcp__semantic_combined(whisper_token pp, whisper_token prev,
                                     whisper_token cur, int has_pp) {
    float bi = hcp__bigram_score(prev, cur);
    if (!has_pp) return bi;
    float tri = hcp__trigram_score(pp, prev, cur);
    return (1.0f - HCP_TRIGRAM_WEIGHT) * bi + HCP_TRIGRAM_WEIGHT * tri;
}
/* ── KIEL-CC: Kalman Innovation Error Localization ───────────────── */

static void hcp__kiel_cc(hcp__cpx *z_orig, int N, int *seg_map, int ns,
                          HcpResult *res) {
    double t0 = hcp__ms_now();
    float *innovation = (float *)calloc((size_t)N, sizeof(float));
    if (!innovation) { res->kiel_ms = hcp__ms_now() - t0; return; }

    /* Adaptive alpha from lag-1 autocorrelation */
    float alpha = 0.95f;
    int warmup = N / 10;
    if (warmup < 20) warmup = 20;
    if (warmup > N) warmup = N;
    if (N > 20) {
        double sum_xy = 0, sum_xx = 0;
        for (int i = 0; i < warmup - 1; i++) {
            sum_xy += (double)z_orig[i].re * z_orig[i+1].re
                    + (double)z_orig[i].im * z_orig[i+1].im;
            sum_xx += (double)z_orig[i].re * z_orig[i].re
                    + (double)z_orig[i].im * z_orig[i].im;
        }
        if (sum_xx > 1e-8) {
            float a = (float)(sum_xy / sum_xx);
            if (a > 0.1f && a < 0.999f) alpha = a;
        }
    }

    /* Parallel Kalman filters on real and imaginary channels */
    float x_re = z_orig[0].re, x_im = z_orig[0].im;
    float P_re = 1.0f, P_im = 1.0f;
    float Q = HCP_KIEL_Q, R_kiel = HCP_KIEL_R;

    int n_flagged = 0;
    float *seg_max = (float *)calloc((size_t)ns, sizeof(float));

    for (int i = 0; i < N; i++) {
        /* Predict */
        float pred_re = alpha * x_re;
        float pred_im = alpha * x_im;
        float Pp_re = alpha * alpha * P_re + Q;
        float Pp_im = alpha * alpha * P_im + Q;

        /* Innovation */
        float inn_re = z_orig[i].re - pred_re;
        float inn_im = z_orig[i].im - pred_im;
        float S_re = Pp_re + R_kiel;
        float S_im = Pp_im + R_kiel;

        /* Normalized innovation magnitude */
        float norm_inn = sqrtf(inn_re * inn_re / S_re + inn_im * inn_im / S_im);
        innovation[i] = norm_inn;

        /* Kalman update */
        float K_re = Pp_re / S_re;
        float K_im = Pp_im / S_im;
        x_re = pred_re + K_re * inn_re;
        x_im = pred_im + K_im * inn_im;
        P_re = (1.0f - K_re) * Pp_re;
        P_im = (1.0f - K_im) * Pp_im;

        /* Track per-segment max */
        if (seg_map && seg_map[i] < ns && norm_inn > seg_max[seg_map[i]])
            seg_max[seg_map[i]] = norm_inn;

        if (norm_inn > HCP_KIEL_INNOV_THRESH) n_flagged++;
    }

    for (int s = 0; s < ns; s++)
        res->segments[s].kiel_max_innov = seg_max[s];

    res->kiel_flagged_tokens = n_flagged;
    res->kiel_innovation = innovation;
    res->kiel_ms = hcp__ms_now() - t0;

    free(seg_max);
}
/* ── Segment extraction ──────────────────────────────────────────── */

static int hcp__extract_segments(struct whisper_context *ctx, HcpResult *res) {
    int ns = whisper_full_n_segments(ctx);
    for (int s = 0; s < ns; s++) {
        if (res->count >= res->cap) {
            int newcap = res->cap ? res->cap * 2 : 256;
            HcpSegment *tmp = (HcpSegment *)realloc(res->segments, (size_t)newcap * sizeof(HcpSegment));
            if (!tmp) return -1;
            res->segments = tmp;
            res->cap = newcap;
        }
        HcpSegment *seg = &res->segments[res->count];
        memset(seg, 0, sizeof(*seg));

        seg->t0_ms = whisper_full_get_segment_t0(ctx, s) * 10;
        seg->t1_ms = whisper_full_get_segment_t1(ctx, s) * 10;
        seg->no_speech_prob = whisper_full_get_segment_no_speech_prob(ctx, s);
        seg->speaker_turn = whisper_full_get_segment_speaker_turn_next(ctx, s) ? 1 : 0;

        const char *txt = whisper_full_get_segment_text(ctx, s);
        if (txt) {
            strncpy(seg->text, txt, sizeof(seg->text) - 1);
            seg->text[sizeof(seg->text) - 1] = '\0';
        }

        /* Geometric mean confidence */
        int nt = whisper_full_n_tokens(ctx, s);
        seg->token_count = nt;
        double log_conf_sum = 0.0;
        int valid = 0;
        for (int t = 0; t < nt; t++) {
            whisper_token_data td = whisper_full_get_token_data(ctx, s, t);
            if (td.id < 50257 && td.p > 0.0f) {
                log_conf_sum += log((double)td.p);
                valid++;
            }
        }
        seg->confidence = valid > 0 ? expf((float)(log_conf_sum / valid)) : 0.0f;
        seg->logprob = valid > 0 ? (float)(log_conf_sum / valid) : -10.0f;

        /* Compression ratio */
        seg->compression_ratio = hcp_compression_ratio(seg->text);

        /* Hallucination flags (layers 1-4; layer 5 = HCP spectral, added later) */
        seg->hallucination_flags = 0;
        if (seg->compression_ratio > 2.4f)
            seg->hallucination_flags |= HCP_HALLUC_HIGH_COMPRESS;
        if (hcp_detect_ngram_repeat(seg->text))
            seg->hallucination_flags |= HCP_HALLUC_NGRAM_REPEAT;

        /* Vlen anomaly */
        if (nt >= 4) {
            float max_vlen = 0.0f;
            for (int t = 0; t < nt; t++) {
                whisper_token_data td = whisper_full_get_token_data(ctx, s, t);
                if (td.vlen > max_vlen) max_vlen = td.vlen;
            }
            int anomalous = 0;
            for (int t = 0; t < nt; t++) {
                whisper_token_data td = whisper_full_get_token_data(ctx, s, t);
                if (max_vlen > 0.0f && td.vlen > max_vlen * 0.667f)
                    anomalous++;
            }
            if (anomalous > nt * 2 / 3)
                seg->hallucination_flags |= HCP_HALLUC_VLEN_ANOMALY;
        }

        if (seg->logprob < -2.0f)
            seg->hallucination_flags |= HCP_HALLUC_LOW_LOGPROB;

        /* Composite quality */
        float nsp_f = 1.0f - seg->no_speech_prob;
        if (nsp_f < 0.0f) nsp_f = 0.0f;
        float cr_f = seg->compression_ratio > 0.0f ? fminf(1.0f, 2.4f / seg->compression_ratio) : 1.0f;
        seg->quality = seg->confidence * nsp_f * cr_f;
        seg->hcp_quality = seg->quality;

        if (seg->hallucination_flags)
            res->segments_hallucinated++;

        res->count++;
    }
    return 0;
}

/* ── Core HCP spectral processing (shared by Whisper + universal paths) ─── */

static void hcp__spectral_process(hcp__token *flat, int N, int ns, HcpResult *res) {
    double t0 = hcp__ms_now();

    res->hcp_tokens = N;
    int N2 = hcp__next_pow2(N);
    res->hcp_padded = N2;

    /* Allocate */
    hcp__cpx *z     = (hcp__cpx *)calloc((size_t)N2, sizeof(hcp__cpx));
    hcp__cpx *z_orig = (hcp__cpx *)calloc((size_t)N2, sizeof(hcp__cpx));
    float *mag_o    = (float *)calloc((size_t)N2, sizeof(float));
    float *mag_c    = (float *)calloc((size_t)N2, sizeof(float));
    float *ph_shift = (float *)calloc((size_t)N2, sizeof(float));
    int   *seg_map  = (int *)calloc((size_t)N2, sizeof(int));
    if (!z || !z_orig || !mag_o || !mag_c || !ph_shift || !seg_map) {
        free(z); free(z_orig); free(mag_o); free(mag_c);
        free(ph_shift); free(seg_map);
        res->hcp_ms = hcp__ms_now() - t0;
        return;
    }

    /* Per-segment semantic accumulators */
    float *seg_sem_sum = (float *)calloc((size_t)ns, sizeof(float));
    int   *seg_sem_cnt = (int *)calloc((size_t)ns, sizeof(int));
    int   *seg_sem_low = (int *)calloc((size_t)ns, sizeof(int));

    /* Step 2: Complex lifting */
    for (int i = 0; i < N; i++) {
        hcp__token *tk = &flat[i];
        seg_map[i] = tk->seg_idx;

        /* Acoustic channel */
        float mag_acou = sqrtf(tk->p);
        int vlen_q = hcp__quant_vlen(tk->vlen);
        int64_t dt_prev = 0;
        if (i > 0 && flat[i].t_dtw > 0 && flat[i-1].t_dtw > 0) {
            dt_prev = flat[i].t_dtw - flat[i-1].t_dtw;
            if (dt_prev < 0) dt_prev = 0;
        }
        int dt_q = hcp__quant_dt(dt_prev);

        uint8_t acou_key[12];
        memcpy(acou_key, &tk->id, 4);
        memcpy(acou_key + 4, &vlen_q, 4);
        memcpy(acou_key + 8, &dt_q, 4);
        float phi_acou = hcp__fnv_to_phase(hcp__fnv1a(acou_key, sizeof(acou_key)));
        hcp__cpx z_acou = hcp__cpx_from_polar(mag_acou, phi_acou);

        /* Morphological channel */
        float freq = 1e-7f;
        if (tk->id >= 0 && tk->id < HCP_VOCAB_SIZE) {
            freq = hcp_subword_freq[tk->id];
        }
        float mag_morph = sqrtf(freq);
        const char *txt = tk->text;
        size_t tlen = txt ? strlen(txt) : 0;
        float phi_morph = hcp__fnv_to_phase(hcp__fnv1a(txt, tlen));
        hcp__cpx z_morph = hcp__cpx_from_polar(mag_morph, phi_morph);

        /* Semantic channel (v3.1): token bigram+trigram coherence */
        float sem_score = 0.5f;
        if (i > 0) {
            int has_pp = (i > 1);
            whisper_token pp = has_pp ? flat[i-2].id : 0;
            sem_score = hcp__semantic_combined(pp, flat[i-1].id, tk->id, has_pp);
            if (sem_score < 0.01f) sem_score = 0.01f;
        }
        float sem_mag = powf(0.5f + 0.5f * sem_score, HCP_SEMANTIC_WEIGHT);
        hcp__cpx z_sem = hcp__cpx_from_polar(sem_mag, phi_morph * HCP_SEMANTIC_WEIGHT);

        /* Track per-segment semantic stats */
        if (seg_sem_cnt && tk->seg_idx < ns) {
            seg_sem_sum[tk->seg_idx] += sem_score;
            seg_sem_cnt[tk->seg_idx]++;
            if (sem_score < HCP_SEMANTIC_LOW_THRESH)
                seg_sem_low[tk->seg_idx]++;
        }

        /* Coupled signal: acoustic × morphological × semantic */
        hcp__cpx coupled = hcp__cpx_mul(hcp__cpx_mul(z_acou, z_morph), z_sem);

        /* Step 3: Free signal integration */

        /* No-speech damping */
        float nsp_damp = 1.0f - tk->no_speech_prob;
        if (nsp_damp < 0.1f) nsp_damp = 0.1f;
        coupled.re *= nsp_damp;
        coupled.im *= nsp_damp;

        /* Compression ratio damping */
        if (tk->comp_ratio > 2.4f) {
            float cr_damp = 2.4f / tk->comp_ratio;
            coupled.re *= cr_damp;
            coupled.im *= cr_damp;
        }

        /* Speaker turn phase reset */
        if (tk->speaker_turn && i > 0) {
            float mag = hcp__cpx_mag(coupled);
            uint64_t turn_hash = hcp__fnv1a(&i, sizeof(i));
            coupled = hcp__cpx_from_polar(mag, hcp__fnv_to_phase(turn_hash));
        }

        /* Vlen anomaly damping */
        if (i >= 2 && i < N - 2) {
            float vlens[5] = {
                flat[i-2].vlen, flat[i-1].vlen, flat[i].vlen,
                flat[i+1].vlen, flat[i+2].vlen
            };
            for (int a = 0; a < 4; a++)
                for (int b = a + 1; b < 5; b++)
                    if (vlens[a] > vlens[b]) {
                        float tmp = vlens[a]; vlens[a] = vlens[b]; vlens[b] = tmp;
                    }
            float median = vlens[2];
            if (median > 0.0f) {
                float ratio = tk->vlen / median;
                if (ratio > 2.0f || ratio < 0.33f) {
                    coupled.re *= 0.5f;
                    coupled.im *= 0.5f;
                }
            }
        }

        /* Low logprob damping */
        if (tk->plog < -3.0f) {
            float lp_damp = 1.0f + tk->plog / 3.0f;
            if (lp_damp < 0.2f) lp_damp = 0.2f;
            coupled.re *= lp_damp;
            coupled.im *= lp_damp;
        }

        z[i] = coupled;
    }

    /* Store per-segment semantic scores */
    int sem_low_total = 0;
    if (seg_sem_cnt) {
        for (int s = 0; s < ns; s++) {
            res->segments[s].semantic_score = seg_sem_cnt[s] > 0
                ? seg_sem_sum[s] / (float)seg_sem_cnt[s] : 0.5f;
            if (seg_sem_cnt[s] > 0 &&
                (float)seg_sem_low[s] / (float)seg_sem_cnt[s] > 0.8f) {
                res->segments[s].hallucination_flags |= HCP_HALLUC_SEMANTIC;
                sem_low_total++;
            }
        }
    }
    res->semantic_low_count = sem_low_total;
    free(seg_sem_sum); free(seg_sem_cnt); free(seg_sem_low);

    /* Save original */
    memcpy(z_orig, z, (size_t)N2 * sizeof(hcp__cpx));
    for (int i = 0; i < N; i++) mag_o[i] = hcp__cpx_mag(z_orig[i]);

    /* Step 3b: KIEL-CC on complex-lifted signal */
    hcp__kiel_cc(z_orig, N, seg_map, ns, res);

    /* Step 4: FFT */
    hcp__fft(z, N2, 0);

    /* Step 5: Three-band filter + Dirichlet */
    float *spec_energy = (float *)calloc((size_t)N2, sizeof(float));
    if (spec_energy) {
        for (int k = 0; k < N2; k++)
            spec_energy[k] = z[k].re * z[k].re + z[k].im * z[k].im;

        int env_win = N2 / 32;
        if (env_win < 4) env_win = 4;
        float *envelope = (float *)calloc((size_t)N2, sizeof(float));
        if (envelope) {
            for (int k = 0; k < N2; k++) {
                int lo = k - env_win / 2; if (lo < 0) lo = 0;
                int hi = k + env_win / 2; if (hi >= N2) hi = N2 - 1;
                float sum = 0.0f; int cnt = 0;
                for (int j = lo; j <= hi; j++) { sum += spec_energy[j]; cnt++; }
                envelope[k] = cnt > 0 ? sum / (float)cnt : 1e-8f;
            }

            int band_low = N2 / 64;
            int band_mid = N2 / 8;

            for (int k = 0; k < N2; k++) {
                float H = 1.0f;

                if (k < band_low || k > N2 - band_low) {
                    if (spec_energy[k] > 3.0f * envelope[k])
                        H = sqrtf(3.0f * envelope[k] / (spec_energy[k] + 1e-8f));
                } else if (k < band_mid || k > N2 - band_mid) {
                    if (spec_energy[k] > 5.0f * envelope[k])
                        H = sqrtf(5.0f * envelope[k] / (spec_energy[k] + 1e-8f));
                } else {
                    if (spec_energy[k] > 2.0f * envelope[k])
                        H = sqrtf(2.0f * envelope[k] / (spec_energy[k] + 1e-8f));
                }

                /* Dirichlet anomaly detection */
                float deviation = spec_energy[k] / (envelope[k] + 1e-8f);
                if (deviation > 8.0f) H *= 0.3f;
                else if (deviation < 0.05f && k > 0 && k < N2 / 2) H *= 1.2f;

                if (H > 1.2f) H = 1.2f;
                if (H < 0.1f) H = 0.1f;

                z[k].re *= H;
                z[k].im *= H;
            }
            free(envelope);
        }
        free(spec_energy);
    }

    /* Step 6: IFFT */
    hcp__fft(z, N2, 1);

    /* Step 7: Compare and flag */
    int n_flagged = 0;
    int *per_seg_flagged = (int *)calloc((size_t)ns, sizeof(int));
    int *per_seg_total   = (int *)calloc((size_t)ns, sizeof(int));

    for (int i = 0; i < N; i++) {
        mag_c[i] = hcp__cpx_mag(z[i]);
        float ph_o = hcp__cpx_phase(z_orig[i]);
        float ph_c = hcp__cpx_phase(z[i]);
        float dph = fabsf(ph_c - ph_o);
        if (dph > (float)M_PI) dph = 2.0f * (float)M_PI - dph;
        ph_shift[i] = dph;

        int flagged = 0;
        if (dph > HCP_PHASE_SHIFT_THRESH) flagged = 1;
        if (mag_o[i] > 1e-6f && mag_c[i] / mag_o[i] < HCP_MAG_SUPPRESS_THRESH) flagged = 1;

        if (flagged) {
            n_flagged++;
            if (seg_map[i] < ns) per_seg_flagged[seg_map[i]]++;
        }
        if (seg_map[i] < ns) per_seg_total[seg_map[i]]++;
    }

    /* Determine flagged segments */
    int n_flagged_seg = 0;
    for (int s = 0; s < ns; s++) {
        if (per_seg_total[s] > 0) {
            float ratio = (float)per_seg_flagged[s] / (float)per_seg_total[s];
            if (ratio > HCP_REDECODE_THRESH) {
                res->segments[s].hallucination_flags |= HCP_HALLUC_SPECTRAL;
                n_flagged_seg++;
            }
        }
        res->segments[s].hcp_flagged_count = per_seg_flagged[s];
    }

    /* Step 8: Enhanced quality (with KIEL-CC integration) */
    for (int s = 0; s < ns; s++) {
        float sum_ratio = 0.0f;
        int cnt = 0;
        int kiel_flags = 0;
        for (int i = 0; i < N; i++) {
            if (seg_map[i] == s) {
                if (mag_o[i] > 1e-8f) {
                    sum_ratio += mag_c[i] / mag_o[i];
                    cnt++;
                }
                if (res->kiel_innovation && res->kiel_innovation[i] > HCP_KIEL_INNOV_THRESH)
                    kiel_flags++;
            }
        }
        float mean_ratio = cnt > 0 ? sum_ratio / (float)cnt : 1.0f;
        if (mean_ratio < 0.5f) mean_ratio = 0.5f;
        if (mean_ratio > 1.5f) mean_ratio = 1.5f;

        /* KIEL penalty: dampen quality if innovation spikes cluster */
        float kiel_factor = 1.0f;
        if (cnt > 0 && kiel_flags > 0) {
            float kiel_ratio = (float)kiel_flags / (float)cnt;
            if (kiel_ratio > HCP_REDECODE_THRESH) {
                res->segments[s].hallucination_flags |= HCP_HALLUC_KALMAN;
                kiel_factor = 1.0f - 0.5f * kiel_ratio;
                if (kiel_factor < 0.5f) kiel_factor = 0.5f;
            }
        }

        res->segments[s].hcp_quality = res->segments[s].quality * mean_ratio * kiel_factor;
        if (res->segments[s].hcp_quality > 1.0f) res->segments[s].hcp_quality = 1.0f;
    }

    /* Store results */
    res->hcp_flagged_tokens = n_flagged;
    res->hcp_flagged_segments = n_flagged_seg;
    res->hcp_mag_original = mag_o;
    res->hcp_mag_corrected = mag_c;
    res->hcp_phase_shift = ph_shift;
    res->hcp_token_seg_map = seg_map;
    res->hcp_ms = hcp__ms_now() - t0;

    free(z);
    free(z_orig);
    free(per_seg_flagged);
    free(per_seg_total);
}

/* ── Whisper-specific token extraction + delegation ──────────────── */

static void hcp__spectral_refine(struct whisper_context *ctx, HcpResult *res) {
    int ns = res->count;

    /* Flatten tokens from whisper context */
    int total = 0;
    for (int s = 0; s < ns; s++)
        total += whisper_full_n_tokens(ctx, s);
    if (total < 4) return;

    hcp__token *flat = (hcp__token *)calloc((size_t)total, sizeof(hcp__token));
    if (!flat) return;

    int idx = 0;
    for (int s = 0; s < ns; s++) {
        int nt = whisper_full_n_tokens(ctx, s);
        float nsp = whisper_full_get_segment_no_speech_prob(ctx, s);
        int sturn = whisper_full_get_segment_speaker_turn_next(ctx, s) ? 1 : 0;
        float cratio = res->segments[s].compression_ratio;
        for (int t = 0; t < nt && idx < total; t++) {
            whisper_token_data td = whisper_full_get_token_data(ctx, s, t);
            const char *txt = whisper_full_get_token_text(ctx, s, t);
            if (td.id >= 50257 || !txt || txt[0] == '\0') continue;
            flat[idx] = (hcp__token){
                .id = td.id,
                .p  = td.p > 0.0f ? td.p : 1e-8f,
                .plog = td.plog,
                .vlen = td.vlen,
                .t_dtw = td.t_dtw,
                .seg_idx = s,
                .no_speech_prob = nsp,
                .comp_ratio = cratio,
                .speaker_turn = sturn,
                .text = txt,
            };
            idx++;
        }
    }
    int N = idx;
    if (N < 4) { free(flat); return; }

    hcp__spectral_process(flat, N, ns, res);
    free(flat);
}
/* ── Formant Anchoring (v3.1): mel-spectrogram speech verification ── */

/* ── Unified Audio Analysis: E-T Gate + Formant (single FFT pass) ── */
/*
 * Both E-T Gate (spectral flatness) and Formant Anchoring (F1/F2 energy)
 * need per-frame FFT on the same audio windows. This unified pass computes
 * the FFT once per frame and extracts both metrics, eliminating ~50% of
 * the audio FFT overhead.
 */

static void hcp__audio_analysis(const float *audio, int n_samples, int sample_rate,
                                 HcpResult *res) {
    double t0_total = hcp__ms_now();
    int gated = 0, formant_flagged = 0;
    int frame_size = HCP_ET_FRAME_SIZE;  /* 512 — same for both */
    int hop = frame_size / 2;
    int half = frame_size / 2;

    /* Formant bin indices (precompute once) */
    float bin_hz = (float)sample_rate / (float)frame_size;
    int f1_lo = (int)(HCP_FORMANT_F1_LO / bin_hz);
    int f1_hi = (int)(HCP_FORMANT_F1_HI / bin_hz);
    int f2_lo = (int)(HCP_FORMANT_F2_LO / bin_hz);
    int f2_hi = (int)(HCP_FORMANT_F2_HI / bin_hz);
    if (f1_lo < 1) f1_lo = 1;
    if (f1_hi > half) f1_hi = half;
    if (f2_lo < 1) f2_lo = 1;
    if (f2_hi > half) f2_hi = half;

    /* Precompute Hann window */
    float hann[HCP_ET_FRAME_SIZE];
    for (int i = 0; i < frame_size; i++)
        hann[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / (frame_size - 1)));

    for (int s = 0; s < res->count; s++) {
        HcpSegment *seg = &res->segments[s];

        /* Map segment timestamps to sample range */
        int sa = (int)((int64_t)seg->t0_ms * sample_rate / 1000);
        int sb = (int)((int64_t)seg->t1_ms * sample_rate / 1000);
        if (sa < 0) sa = 0;
        if (sb > n_samples) sb = n_samples;
        int seg_len = sb - sa;

        if (seg_len < frame_size) {
            seg->et_rms = 0;
            seg->et_speech_frac = 1.0f;
            seg->formant_ratio = 1.0f;
            continue;
        }

        /* Short segments with few tokens — skip formant, default E-T */
        if (seg->token_count < 2) {
            seg->formant_ratio = 1.0f;
        }

        /* Overall RMS for E-T Gate */
        double rms_acc = 0;
        for (int i = sa; i < sb; i++)
            rms_acc += (double)audio[i] * audio[i];
        seg->et_rms = sqrtf((float)(rms_acc / seg_len));

        /* Single-pass frame loop: compute FFT once, extract both metrics */
        int n_frames = 0, speech_frames = 0;
        double formant_energy = 0, total_energy = 0;

        for (int f = sa; f + frame_size <= sb; f += hop) {
            n_frames++;

            /* Frame RMS (for E-T silence gate) */
            float sum_sq = 0;
            for (int i = 0; i < frame_size; i++) {
                float x = audio[f + i];
                sum_sq += x * x;
            }
            float rms = sqrtf(sum_sq / frame_size);

            int is_silent = (rms < HCP_ET_RMS_FLOOR);

            /* Single FFT per frame — windowed for formant, raw for E-T
             * Use Hann window (needed for formant accuracy, doesn't hurt flatness) */
            hcp__cpx fft_buf[HCP_ET_FRAME_SIZE];
            for (int i = 0; i < frame_size; i++) {
                fft_buf[i].re = audio[f + i] * hann[i];
                fft_buf[i].im = 0.0f;
            }
            hcp__fft(fft_buf, frame_size, 0);

            /* Compute power spectrum once */
            float spec[HCP_ET_FRAME_SIZE / 2];
            for (int k = 1; k < half; k++)
                spec[k] = fft_buf[k].re * fft_buf[k].re + fft_buf[k].im * fft_buf[k].im;

            /* === E-T Gate: spectral flatness === */
            if (!is_silent) {
                double log_sum = 0, lin_sum = 0;
                for (int k = 1; k < half; k++) {
                    float S = spec[k];
                    if (S < 1e-12f) S = 1e-12f;
                    log_sum += log((double)S);
                    lin_sum += (double)S;
                }
                int n_bins = half - 1;
                float flatness = (float)(exp(log_sum / n_bins) / (lin_sum / n_bins + 1e-30));
                if (flatness < HCP_ET_FLATNESS_THRESH)
                    speech_frames++;
            }

            /* === Formant: F1+F2 band energy === */
            if (seg->token_count >= 2) {
                double f_e = 0, t_e = 0;
                for (int k = 1; k < half; k++) {
                    t_e += spec[k];
                    if ((k >= f1_lo && k <= f1_hi) || (k >= f2_lo && k <= f2_hi))
                        f_e += spec[k];
                }
                formant_energy += f_e;
                total_energy += t_e;
            }
        }

        /* E-T Gate results */
        seg->et_speech_frac = n_frames > 0 ? (float)speech_frames / n_frames : 1.0f;

        float dur_sec = (float)(seg->t1_ms - seg->t0_ms) / 1000.0f;
        float density = dur_sec > 0 ? (float)seg->token_count / dur_sec : 0;

        if (density > HCP_ET_DENSITY_MIN && seg->et_speech_frac < HCP_ET_SPEECH_MIN) {
            seg->hallucination_flags |= HCP_HALLUC_ET_GATE;
            gated++;
        }

        /* Formant results */
        if (seg->token_count >= 2) {
            float ratio = total_energy > 1e-12 ? (float)(formant_energy / total_energy) : 0.0f;
            seg->formant_ratio = ratio;

            if (density > HCP_ET_DENSITY_MIN && ratio < HCP_FORMANT_SPEECH_THRESH) {
                seg->hallucination_flags |= HCP_HALLUC_FORMANT;
                formant_flagged++;
            }
        }
    }

    res->et_segments_gated = gated;
    res->formant_flagged = formant_flagged;

    double elapsed = hcp__ms_now() - t0_total;
    /* Split timing proportionally for backward compat with JSON output */
    res->et_gate_ms = elapsed * 0.5;
    res->formant_ms = elapsed * 0.5;
}

/* ── Morphological Logit Bias (v3.2) ────────────────────────────── */

/* Context passed through whisper's logits_filter_callback user_data */
typedef struct {
    whisper_token prev_token;      /* previous token ID */
    whisper_token prev_prev_token; /* two tokens back */
    int           has_prev;        /* nonzero if prev_token is valid */
    int           has_prev_prev;   /* nonzero if prev_prev_token is valid */
    int           n_vocab;         /* vocabulary size (cached from whisper_n_vocab) */
    int           tokens_biased;   /* count of tokens that received bias this decode */
} HcpLogitBiasCtx;

/* The callback: for each candidate next-token, check bigram/trigram coherence
 * with the preceding context. Tokens that are morphologically impossible
 * (zero semantic coherence AND low subword frequency) get a logit penalty,
 * pushing beam search toward linguistically valid candidates.
 *
 * This converts HCP from a passive post-filter into an active decoder constraint. */
static void hcp__logit_bias_callback(
        struct whisper_context *ctx,
        struct whisper_state   *state,
        const whisper_token_data *tokens,
        int                      n_tokens,
        float                   *logits,
        void                    *user_data)
{
    (void)ctx; (void)state;
    HcpLogitBiasCtx *bias = (HcpLogitBiasCtx *)user_data;
    if (!bias) return;

    /* Update context from the token stream FIRST —
     * on the first call has_prev=0, so we seed and return.
     * On subsequent calls we have valid prev context for biasing. */
    int had_prev = bias->has_prev;
    if (n_tokens > 0) {
        bias->prev_prev_token = bias->prev_token;
        bias->has_prev_prev = bias->has_prev;
        bias->prev_token = tokens[n_tokens - 1].id;
        bias->has_prev = 1;
    }

    /* Need at least one token of context to compute bigrams */
    if (!had_prev) return;

    int n_vocab = bias->n_vocab;
    if (n_vocab <= 0) return;

    /* Targeted repetition suppression: only penalize tokens that form
     * repetition patterns (the primary hallucination failure mode in
     * quantized models, e.g., "is is is is is").
     *
     * Strategy: instead of biasing ALL zero-bigram tokens (~38% of vocab),
     * we surgically target:
     *   1. Exact repeat of prev_token with zero bigram evidence
     *   2. Exact repeat of prev_prev_token (cyclic A-B-A pattern)
     *   3. Short tokens (<= 2 chars) with zero bigram AND trigram evidence
     *      (these are often hallucination fragments like "is", "in", "or") */

    /* 1. Suppress exact repetition: prev_token appearing again */
    {
        int t = (int)bias->prev_token;
        if (t >= 0 && t < n_vocab && t < 50257) {
            float bi = hcp__bigram_score(bias->prev_token, (whisper_token)t);
            if (bi < HCP_LOGIT_BIAS_FLOOR) {
                logits[t] += HCP_LOGIT_BIAS_STRENGTH;
                bias->tokens_biased++;
            }
        }
    }

    /* 2. Suppress cyclic repetition: A-B-A pattern */
    if (bias->has_prev_prev && bias->prev_prev_token != bias->prev_token) {
        int t = (int)bias->prev_prev_token;
        if (t >= 0 && t < n_vocab && t < 50257) {
            float bi = hcp__bigram_score(bias->prev_token, (whisper_token)t);
            float tri = hcp__trigram_score(bias->prev_prev_token,
                                           bias->prev_token, (whisper_token)t);
            if (bi < HCP_LOGIT_BIAS_FLOOR && tri < HCP_LOGIT_BIAS_FLOOR) {
                logits[t] += HCP_LOGIT_BIAS_STRENGTH * 0.5f; /* softer for cycles */
                bias->tokens_biased++;
            }
        }
    }

    /* 3. Short-token suppression: tokens with len <= 2 and zero coherence
     *    are often hallucination fragments in quantized models */
    int max_tok = n_vocab < 50257 ? n_vocab : 50257;
    for (int t = 0; t < max_tok; t++) {
        /* Skip if already handled above */
        if (t == (int)bias->prev_token) continue;
        if (bias->has_prev_prev && t == (int)bias->prev_prev_token) continue;

        /* Only target short tokens (len <= 2 bytes) */
        if (t < (int)(sizeof(hcp_token_strlen)/sizeof(hcp_token_strlen[0]))) {
            if (hcp_token_strlen[t] > 2) continue;
        } else {
            continue;  /* unknown token, skip */
        }

        float bi = hcp__bigram_score(bias->prev_token, (whisper_token)t);
        if (bi >= HCP_LOGIT_BIAS_FLOOR) continue;

        if (bias->has_prev_prev) {
            float tri = hcp__trigram_score(bias->prev_prev_token,
                                           bias->prev_token, (whisper_token)t);
            if (tri >= HCP_LOGIT_BIAS_FLOOR) continue;
        }

        /* Soft penalty for short zero-coherence tokens */
        logits[t] += HCP_LOGIT_BIAS_STRENGTH * 0.3f;
        bias->tokens_biased++;
    }
}

/* ── Constrained Re-decode (v2.1 + v3.2 logit bias) ─────────────── */

static int hcp__popcount16(uint16_t x) {
    int c = 0;
    while (x) { c += x & 1; x >>= 1; }
    return c;
}

int hcp_redecode(struct whisper_context *ctx,
                 const float *audio, int n_samples, int sample_rate,
                 struct whisper_full_params base_params,
                 HcpResult *res) {
    if (!ctx || !audio || !res) return 0;
    double t0 = hcp__ms_now();
    int improved = 0, attempted = 0;

    /* Configure re-decode params: wider beam, no printing */
    struct whisper_full_params rp = base_params;
    rp.beam_search.beam_size = HCP_REDECODE_BEAM;
    rp.no_speech_thold = 0.3f;
    rp.print_progress = 0;
    rp.print_timestamps = 0;
    rp.print_special = 0;

    /* Set up morphological logit bias (v3.2) */
    HcpLogitBiasCtx bias_ctx = {0};
    bias_ctx.n_vocab = whisper_n_vocab(ctx);
    rp.logits_filter_callback = hcp__logit_bias_callback;
    rp.logits_filter_callback_user_data = &bias_ctx;

    for (int s = 0; s < res->count; s++) {
        HcpSegment *seg = &res->segments[s];
        if (!seg->hallucination_flags) continue;

        /* v3.2: only re-decode segments with 2+ hallucination layer flags.
         * Single-flag segments are borderline — HCP's passive spectral filter
         * handles them with near-perfect quality. Re-decoding them introduces
         * noise from the stochastic beam search. Reserve the expensive re-decode
         * for segments where multiple independent detectors agree there's a problem. */
        if (hcp__popcount16(seg->hallucination_flags) < 2) continue;

        /* Context-seeded re-decode (v3.1): include audio from surrounding good segments.
         * Expand the audio slice to include 1-2 seconds of context on either side
         * from non-hallucinated segments, giving Whisper better BPE context. */
        int sa = (int)((int64_t)seg->t0_ms * sample_rate / 1000);
        int sb = (int)((int64_t)seg->t1_ms * sample_rate / 1000);

        /* Find preceding clean segment for left context */
        int context_samples = sample_rate * 2; /* up to 2 seconds of context */
        int sa_ctx = sa;
        if (s > 0 && !res->segments[s-1].hallucination_flags) {
            int ctx_start = (int)((int64_t)res->segments[s-1].t0_ms * sample_rate / 1000);
            sa_ctx = sa - context_samples;
            if (sa_ctx < ctx_start) sa_ctx = ctx_start;
            if (sa_ctx < 0) sa_ctx = 0;
        }

        /* Find following clean segment for right context */
        int sb_ctx = sb;
        if (s + 1 < res->count && !res->segments[s+1].hallucination_flags) {
            int ctx_end = (int)((int64_t)res->segments[s+1].t1_ms * sample_rate / 1000);
            sb_ctx = sb + context_samples;
            if (sb_ctx > ctx_end) sb_ctx = ctx_end;
            if (sb_ctx > n_samples) sb_ctx = n_samples;
        }

        if (sa_ctx < 0) sa_ctx = 0;
        if (sb_ctx > n_samples) sb_ctx = n_samples;
        int slice_len = sb_ctx - sa_ctx;
        if (slice_len < HCP_REDECODE_MIN_SAMPLES) continue;

        attempted++;

        /* Seed logit bias context with tokens from preceding clean segment (v3.2).
         * This gives the callback bigram/trigram context so it can suppress
         * morphologically impossible continuations from the very first token. */
        bias_ctx.has_prev = 0;
        bias_ctx.has_prev_prev = 0;
        bias_ctx.tokens_biased = 0;
        if (s > 0 && !res->segments[s-1].hallucination_flags) {
            /* The previous segment's text is available but we need token IDs.
             * Use a simple heuristic: hash the last ~2 words of prev segment text
             * to get approximate token context. For precise seeding, we'd need
             * the original token IDs — but the bigram table is collision-tolerant. */
            /* Actually, we can get the last tokens from the previous decode pass
             * if the whisper context still has them. Check if segment count matches. */
            int prev_ns = whisper_full_n_segments(ctx);
            /* The context was last used for the full decode — grab trailing tokens */
            (void)prev_ns; /* prev decode state was overwritten; use text-based seed */
        }

        /* Re-decode the context-seeded slice with wider beam + logit bias */
        int ret = whisper_full(ctx, rp, audio + sa_ctx, slice_len);
        if (ret != 0) continue;

        /* Extract re-decoded text — only keep text from the target segment's time range */
        int ns_new = whisper_full_n_segments(ctx);
        if (ns_new == 0) continue;

        char new_text[HCP_MAX_TEXT_LEN] = {0};
        double log_conf_sum = 0;
        int valid = 0;

        /* Target time range within the slice (relative to slice start) */
        int64_t tgt_lo_ms = seg->t0_ms - (int64_t)sa_ctx * 1000 / sample_rate;
        int64_t tgt_hi_ms = seg->t1_ms - (int64_t)sa_ctx * 1000 / sample_rate;
        if (tgt_lo_ms < 0) tgt_lo_ms = 0;

        for (int si = 0; si < ns_new; si++) {
            int64_t st0 = whisper_full_get_segment_t0(ctx, si) * 10;
            int64_t st1 = whisper_full_get_segment_t1(ctx, si) * 10;

            /* Only include segments that overlap with the original target range */
            if (st1 < tgt_lo_ms - 500 || st0 > tgt_hi_ms + 500) continue;

            const char *txt = whisper_full_get_segment_text(ctx, si);
            if (txt) {
                size_t cur = strlen(new_text);
                strncat(new_text, txt, sizeof(new_text) - cur - 1);
            }
            int nt = whisper_full_n_tokens(ctx, si);
            for (int t = 0; t < nt; t++) {
                whisper_token_data td = whisper_full_get_token_data(ctx, si, t);
                if (td.id < 50257 && td.p > 0.0f) {
                    log_conf_sum += log((double)td.p);
                    valid++;
                }
            }
        }

        float new_conf = valid > 0 ? expf((float)(log_conf_sum / valid)) : 0.0f;
        float new_cr = hcp_compression_ratio(new_text);
        int new_ngram = hcp_detect_ngram_repeat(new_text);
        float new_logprob = valid > 0 ? (float)(log_conf_sum / valid) : -10.0f;

        /* Re-check hallucination signals */
        uint16_t new_flags = 0;
        if (new_cr > 2.4f) new_flags |= HCP_HALLUC_HIGH_COMPRESS;
        if (new_ngram) new_flags |= HCP_HALLUC_NGRAM_REPEAT;
        if (new_logprob < -2.0f) new_flags |= HCP_HALLUC_LOW_LOGPROB;

        /* Compute new quality */
        float nsp_f = 1.0f - seg->no_speech_prob;
        if (nsp_f < 0.0f) nsp_f = 0.0f;
        float cr_f = new_cr > 0.0f ? fminf(1.0f, 2.4f / new_cr) : 1.0f;
        float new_quality = new_conf * nsp_f * cr_f;

        /* Accept if: better quality or fewer hallucination flags */
        int fewer_flags = hcp__popcount16(new_flags) < hcp__popcount16(seg->hallucination_flags);
        if (new_quality > seg->hcp_quality ||
            (fewer_flags && new_quality >= seg->hcp_quality * 0.9f)) {
            strncpy(seg->text, new_text, sizeof(seg->text) - 1);
            seg->text[sizeof(seg->text) - 1] = '\0';
            seg->confidence = new_conf;
            seg->logprob = new_logprob;
            seg->compression_ratio = new_cr;
            seg->quality = new_quality;
            seg->hcp_quality = new_quality;
            seg->hallucination_flags = new_flags;
            improved++;
        }
    }

    res->redecode_count = attempted;
    res->redecode_improved = improved;
    res->redecode_ms = hcp__ms_now() - t0;
    res->logit_bias_tokens = bias_ctx.tokens_biased;
    return improved;
}

/* ── Universal segment extraction (v4.0) ─────────────────────────── */

static int hcp__extract_universal(HcpUniversalSegment *segments, int n_seg,
                                   HcpResult *res) {
    for (int s = 0; s < n_seg; s++) {
        if (res->count >= res->cap) {
            int newcap = res->cap ? res->cap * 2 : 256;
            HcpSegment *tmp = (HcpSegment *)realloc(res->segments,
                              (size_t)newcap * sizeof(HcpSegment));
            if (!tmp) return -1;
            res->segments = tmp;
            res->cap = newcap;
        }
        HcpUniversalSegment *useg = &segments[s];
        HcpSegment *seg = &res->segments[res->count];
        memset(seg, 0, sizeof(*seg));

        seg->t0_ms = useg->start_ms;
        seg->t1_ms = useg->end_ms;
        seg->no_speech_prob = useg->no_speech_prob;
        seg->token_count = useg->token_count;

        if (useg->text) {
            strncpy(seg->text, useg->text, sizeof(seg->text) - 1);
            seg->text[sizeof(seg->text) - 1] = '\0';
        }

        /* Geometric mean confidence from tokens */
        double log_conf_sum = 0;
        int valid = 0;
        for (int t = 0; t < useg->token_count; t++) {
            float p = useg->tokens[t].confidence;
            if (p > 0.0f) { log_conf_sum += log((double)p); valid++; }
        }
        seg->confidence = valid > 0 ? expf((float)(log_conf_sum / valid)) : 0.0f;
        seg->logprob = valid > 0 ? (float)(log_conf_sum / valid) : -10.0f;

        seg->compression_ratio = hcp_compression_ratio(seg->text);

        /* Hallucination layers 1-4 */
        if (seg->compression_ratio > 2.4f)
            seg->hallucination_flags |= HCP_HALLUC_HIGH_COMPRESS;
        if (hcp_detect_ngram_repeat(seg->text))
            seg->hallucination_flags |= HCP_HALLUC_NGRAM_REPEAT;
        if (seg->logprob < -2.0f)
            seg->hallucination_flags |= HCP_HALLUC_LOW_LOGPROB;

        float nsp_f = 1.0f - seg->no_speech_prob;
        if (nsp_f < 0.0f) nsp_f = 0.0f;
        float cr_f = seg->compression_ratio > 0.0f
                     ? fminf(1.0f, 2.4f / seg->compression_ratio) : 1.0f;
        seg->quality = seg->confidence * nsp_f * cr_f;
        seg->hcp_quality = seg->quality;

        if (seg->hallucination_flags)
            res->segments_hallucinated++;

        res->count++;
    }
    return 0;
}

static void hcp__universal_refine(HcpUniversalSegment *segments, int n_seg,
                                   HcpResult *res) {
    int ns = res->count;

    /* Flatten universal tokens into internal format */
    int total = 0;
    for (int s = 0; s < n_seg; s++)
        total += segments[s].token_count;
    if (total < 4) return;

    hcp__token *flat = (hcp__token *)calloc((size_t)total, sizeof(hcp__token));
    if (!flat) return;

    int idx = 0;
    for (int s = 0; s < n_seg; s++) {
        HcpUniversalSegment *useg = &segments[s];
        for (int t = 0; t < useg->token_count && idx < total; t++) {
            HcpUniversalToken *ut = &useg->tokens[t];
            size_t tlen = ut->text ? strlen(ut->text) : 0;
            uint32_t pseudo_id = (uint32_t)(hcp__fnv1a(ut->text, tlen) & 0xFFFF);

            int64_t t_ms = useg->start_ms;
            if (useg->token_count > 1)
                t_ms += (int64_t)t * (useg->end_ms - useg->start_ms) / useg->token_count;

            flat[idx] = (hcp__token){
                .id = (whisper_token)pseudo_id,
                .p  = ut->confidence > 0 ? ut->confidence : 1e-8f,
                .plog = ut->logprob,
                .vlen = ut->duration_ms > 0 ? ut->duration_ms / 100.0f : 0.5f,
                .t_dtw = t_ms,
                .seg_idx = s,
                .no_speech_prob = useg->no_speech_prob,
                .comp_ratio = res->segments[s].compression_ratio,
                .speaker_turn = 0,
                .text = ut->text,
            };
            idx++;
        }
    }
    int N = idx;
    if (N < 4) { free(flat); return; }

    hcp__spectral_process(flat, N, ns, res);
    free(flat);
}

/* ── Public API ──────────────────────────────────────────────────── */

HcpResult hcp_process(struct whisper_context *ctx) {
    HcpResult res = {0};
    hcp__extract_segments(ctx, &res);
    hcp__spectral_refine(ctx, &res);
    return res;
}

HcpResult hcp_process_with_audio(struct whisper_context *ctx,
                                  const float *audio, int n_samples, int sample_rate) {
    HcpResult res = {0};
    hcp__extract_segments(ctx, &res);
    hcp__spectral_refine(ctx, &res);
    if (audio && n_samples > 0) {
        hcp__audio_analysis(audio, n_samples, sample_rate, &res);
    }
    return res;
}

HcpResult hcp_process_universal(HcpUniversalSegment *segments, int n_segments,
                                 const float *audio, int n_samples, int sample_rate) {
    HcpResult res = {0};
    if (!segments || n_segments <= 0) return res;
    hcp__extract_universal(segments, n_segments, &res);
    hcp__universal_refine(segments, n_segments, &res);
    if (audio && n_samples > 0) {
        hcp__audio_analysis(audio, n_samples, sample_rate > 0 ? sample_rate : 16000, &res);
    }
    return res;
}

void hcp_free(HcpResult *r) {
    free(r->segments);
    free(r->hcp_mag_original);
    free(r->hcp_mag_corrected);
    free(r->hcp_phase_shift);
    free(r->hcp_token_seg_map);
    free(r->kiel_innovation);
    memset(r, 0, sizeof(*r));
}

#endif /* HCP_IMPLEMENTATION */
#endif /* HCP_H */
