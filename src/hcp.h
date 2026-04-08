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

/* Hallucination flag bits */
#define HCP_HALLUC_HIGH_COMPRESS   0x01
#define HCP_HALLUC_NGRAM_REPEAT    0x02
#define HCP_HALLUC_VLEN_ANOMALY    0x04
#define HCP_HALLUC_LOW_LOGPROB     0x08
#define HCP_HALLUC_SPECTRAL        0x10
#define HCP_HALLUC_ET_GATE         0x20
#define HCP_HALLUC_KALMAN          0x40

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
    uint8_t  hallucination_flags; /* bitfield of HCP_HALLUC_* */
    int      speaker_turn;
    int      token_count;
    int      hcp_flagged_count;   /* tokens flagged by HCP in this segment */
    float    et_rms;              /* E-T Gate: segment audio RMS energy */
    float    et_speech_frac;      /* E-T Gate: speech-like frame fraction */
    float    kiel_max_innov;      /* KIEL-CC: max normalized innovation */
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
} HcpResult;

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

/* ── Core HCP spectral refinement ────────────────────────────────── */

static void hcp__spectral_refine(struct whisper_context *ctx, HcpResult *res) {
    double t0 = hcp__ms_now();
    int ns = res->count;

    /* Flatten tokens */
    int total = 0;
    for (int s = 0; s < ns; s++) {
        total += whisper_full_n_tokens(ctx, s);
    }
    if (total < 4) { res->hcp_ms = hcp__ms_now() - t0; return; }

    hcp__token *flat = (hcp__token *)calloc((size_t)total, sizeof(hcp__token));
    if (!flat) { res->hcp_ms = hcp__ms_now() - t0; return; }

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
    if (N < 4) { free(flat); res->hcp_ms = hcp__ms_now() - t0; return; }

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
        free(flat); free(z); free(z_orig); free(mag_o); free(mag_c);
        free(ph_shift); free(seg_map);
        res->hcp_ms = hcp__ms_now() - t0;
        return;
    }

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

        /* Coupled signal */
        hcp__cpx coupled = hcp__cpx_mul(z_acou, z_morph);

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

    free(flat);
    free(z);
    free(z_orig);
    free(per_seg_flagged);
    free(per_seg_total);
}
/* ── E-T Gate: Energy–Text Cross-Agreement ───────────────────── */

static void hcp__et_gate(const float *audio, int n_samples, int sample_rate,
                          HcpResult *res) {
    double t0 = hcp__ms_now();
    int gated = 0;
    int hop = HCP_ET_FRAME_SIZE / 2;

    for (int s = 0; s < res->count; s++) {
        HcpSegment *seg = &res->segments[s];

        /* Map segment timestamps to sample range */
        int sa = (int)((int64_t)seg->t0_ms * sample_rate / 1000);
        int sb = (int)((int64_t)seg->t1_ms * sample_rate / 1000);
        if (sa < 0) sa = 0;
        if (sb > n_samples) sb = n_samples;
        int seg_len = sb - sa;

        if (seg_len < HCP_ET_FRAME_SIZE) {
            seg->et_rms = 0;
            seg->et_speech_frac = 1.0f;
            continue;
        }

        /* Overall RMS */
        double rms_acc = 0;
        for (int i = sa; i < sb; i++)
            rms_acc += (double)audio[i] * audio[i];
        seg->et_rms = sqrtf((float)(rms_acc / seg_len));

        /* Frame-by-frame: RMS + spectral flatness */
        int n_frames = 0, speech_frames = 0;

        for (int f = sa; f + HCP_ET_FRAME_SIZE <= sb; f += hop) {
            n_frames++;

            /* Frame RMS */
            float sum_sq = 0;
            for (int i = 0; i < HCP_ET_FRAME_SIZE; i++) {
                float x = audio[f + i];
                sum_sq += x * x;
            }
            float rms = sqrtf(sum_sq / HCP_ET_FRAME_SIZE);

            if (rms < HCP_ET_RMS_FLOOR) continue; /* silence */

            /* Spectral flatness: geometric_mean(|X|²) / arithmetic_mean(|X|²) */
            hcp__cpx fft_buf[HCP_ET_FRAME_SIZE];
            for (int i = 0; i < HCP_ET_FRAME_SIZE; i++) {
                fft_buf[i].re = audio[f + i];
                fft_buf[i].im = 0.0f;
            }
            hcp__fft(fft_buf, HCP_ET_FRAME_SIZE, 0);

            int half = HCP_ET_FRAME_SIZE / 2;
            double log_sum = 0, lin_sum = 0;
            for (int k = 1; k < half; k++) {
                float S = fft_buf[k].re * fft_buf[k].re + fft_buf[k].im * fft_buf[k].im;
                if (S < 1e-12f) S = 1e-12f;
                log_sum += log((double)S);
                lin_sum += (double)S;
            }
            int n_bins = half - 1;
            float flatness = (float)(exp(log_sum / n_bins) / (lin_sum / n_bins + 1e-30));

            if (flatness < HCP_ET_FLATNESS_THRESH)
                speech_frames++;
        }

        seg->et_speech_frac = n_frames > 0 ? (float)speech_frames / n_frames : 1.0f;

        /* Gate: high word density but low speech fraction */
        float dur_sec = (float)(seg->t1_ms - seg->t0_ms) / 1000.0f;
        float density = dur_sec > 0 ? (float)seg->token_count / dur_sec : 0;

        if (density > HCP_ET_DENSITY_MIN && seg->et_speech_frac < HCP_ET_SPEECH_MIN) {
            seg->hallucination_flags |= HCP_HALLUC_ET_GATE;
            gated++;
        }
    }

    res->et_segments_gated = gated;
    res->et_gate_ms = hcp__ms_now() - t0;
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
    if (audio && n_samples > 0)
        hcp__et_gate(audio, n_samples, sample_rate, &res);
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
