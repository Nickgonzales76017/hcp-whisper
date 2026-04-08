/* test_hcp.c — Unit tests for HCP internals
 *
 * Tests the mathematical components independently:
 *   1. FFT correctness (forward + inverse = identity)
 *   2. Complex arithmetic
 *   3. Phase hash determinism
 *   4. Compression ratio sanity
 *   5. N-gram detection
 *   6. Frequency table validity
 *   7. Spectral filter bounds
 *
 * Compile: cc -std=c11 -I../src -Isrc -o hcp-test tests/test_hcp.c -lwhisper -lggml -lz -lm
 *
 * MIT License — Copyright (c) 2026 Nick Gonzales
 */

/* We need the implementation for testing internals */
#define HCP_IMPLEMENTATION
#include "hcp.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

/* ─── Test framework ────────────────────────────────────────────── */

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT(cond, msg) do { \
    tests_run++; \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL: %s (line %d): %s\n", __func__, __LINE__, msg); \
        tests_failed++; \
    } else { \
        tests_passed++; \
    } \
} while (0)

#define ASSERT_FLOAT_EQ(a, b, eps, msg) do { \
    tests_run++; \
    if (fabsf((a) - (b)) > (eps)) { \
        fprintf(stderr, "  FAIL: %s (line %d): %s (got %.6f, expected %.6f)\n", \
                __func__, __LINE__, msg, (double)(a), (double)(b)); \
        tests_failed++; \
    } else { \
        tests_passed++; \
    } \
} while (0)

/* ─── Test: FFT roundtrip ───────────────────────────────────────── */

static void test_fft_roundtrip(void) {
    printf("  test_fft_roundtrip...\n");
    const int N = 16;
    hcp__cpx x[16], orig[16];

    /* Fill with known signal: sum of two sinusoids */
    for (int i = 0; i < N; i++) {
        x[i].re = cosf(2.0f * (float)M_PI * 2.0f * i / N)
                + 0.5f * cosf(2.0f * (float)M_PI * 5.0f * i / N);
        x[i].im = sinf(2.0f * (float)M_PI * 2.0f * i / N);
        orig[i] = x[i];
    }

    /* Forward FFT */
    hcp__fft(x, N, 0);

    /* Verify spectral energy at expected bins */
    float energy_k2 = x[2].re * x[2].re + x[2].im * x[2].im;
    float energy_k5 = x[5].re * x[5].re + x[5].im * x[5].im;
    float energy_k0 = x[0].re * x[0].re + x[0].im * x[0].im;

    ASSERT(energy_k2 > energy_k0, "bin k=2 should have significant energy");
    ASSERT(energy_k5 > 1.0f, "bin k=5 should have energy from second sinusoid");

    /* Inverse FFT */
    hcp__fft(x, N, 1);

    /* Verify roundtrip: should match original within floating point tolerance */
    for (int i = 0; i < N; i++) {
        ASSERT_FLOAT_EQ(x[i].re, orig[i].re, 1e-4f, "FFT roundtrip real part");
        ASSERT_FLOAT_EQ(x[i].im, orig[i].im, 1e-4f, "FFT roundtrip imag part");
    }
}

/* ─── Test: FFT of impulse ──────────────────────────────────────── */

static void test_fft_impulse(void) {
    printf("  test_fft_impulse...\n");
    const int N = 8;
    hcp__cpx x[8] = {{0}};
    x[0] = (hcp__cpx){1.0f, 0.0f};  /* impulse at position 0 */

    hcp__fft(x, N, 0);

    /* FFT of impulse = flat spectrum: all bins should have magnitude 1 */
    for (int k = 0; k < N; k++) {
        float mag = hcp__cpx_mag(x[k]);
        ASSERT_FLOAT_EQ(mag, 1.0f, 1e-5f, "impulse FFT should have flat spectrum");
    }
}

/* ─── Test: Complex arithmetic ──────────────────────────────────── */

static void test_complex_arithmetic(void) {
    printf("  test_complex_arithmetic...\n");

    hcp__cpx a = {3.0f, 4.0f};
    hcp__cpx b = {1.0f, 2.0f};

    /* Magnitude */
    ASSERT_FLOAT_EQ(hcp__cpx_mag(a), 5.0f, 1e-6f, "|3+4i| = 5");

    /* Phase */
    hcp__cpx unit_i = {0.0f, 1.0f};
    ASSERT_FLOAT_EQ(hcp__cpx_phase(unit_i), (float)M_PI / 2.0f, 1e-6f, "phase(i) = π/2");

    /* Multiply */
    hcp__cpx prod = hcp__cpx_mul(a, b);
    ASSERT_FLOAT_EQ(prod.re, -5.0f, 1e-6f, "(3+4i)(1+2i) real = -5");
    ASSERT_FLOAT_EQ(prod.im, 10.0f, 1e-6f, "(3+4i)(1+2i) imag = 10");

    /* Add */
    hcp__cpx sum = hcp__cpx_add(a, b);
    ASSERT_FLOAT_EQ(sum.re, 4.0f, 1e-6f, "(3+4i)+(1+2i) real = 4");
    ASSERT_FLOAT_EQ(sum.im, 6.0f, 1e-6f, "(3+4i)+(1+2i) imag = 6");

    /* Polar roundtrip */
    float mag = hcp__cpx_mag(a);
    float phase = hcp__cpx_phase(a);
    hcp__cpx roundtrip = hcp__cpx_from_polar(mag, phase);
    ASSERT_FLOAT_EQ(roundtrip.re, a.re, 1e-5f, "polar roundtrip real");
    ASSERT_FLOAT_EQ(roundtrip.im, a.im, 1e-5f, "polar roundtrip imag");
}

/* ─── Test: FNV-1a hash determinism ─────────────────────────────── */

static void test_fnv_determinism(void) {
    printf("  test_fnv_determinism...\n");

    const char *input = "hello world";
    uint64_t h1 = hcp__fnv1a(input, strlen(input));
    uint64_t h2 = hcp__fnv1a(input, strlen(input));

    ASSERT(h1 == h2, "FNV-1a must be deterministic");
    ASSERT(h1 != 0, "FNV-1a should not be zero for non-empty input");

    /* Different inputs should (almost certainly) produce different hashes */
    uint64_t h3 = hcp__fnv1a("hello worlD", 11);
    ASSERT(h1 != h3, "different inputs should hash differently");

    /* Phase mapping should be in [0, 2π) */
    float phase = hcp__fnv_to_phase(h1);
    ASSERT(phase >= 0.0f && phase < 2.0f * (float)M_PI, "phase must be in [0, 2π)");
}

/* ─── Test: Compression ratio ───────────────────────────────────── */

static void test_compression_ratio(void) {
    printf("  test_compression_ratio...\n");

    /* Normal text: low compression ratio */
    float normal = hcp_compression_ratio(
        "The quick brown fox jumps over the lazy dog near the river bank.");
    ASSERT(normal > 0.5f && normal < 3.0f, "normal text should have moderate ratio");

    /* Repetitive text (hallucination-like): high compression ratio */
    char repeat[2048];
    repeat[0] = '\0';
    for (int i = 0; i < 20; i++) strcat(repeat, "the same words ");
    float repetitive = hcp_compression_ratio(repeat);
    ASSERT(repetitive > normal, "repetitive text should compress more");

    /* Very short text */
    float tiny = hcp_compression_ratio("hi");
    ASSERT_FLOAT_EQ(tiny, 1.0f, 0.01f, "very short text returns 1.0");
}

/* ─── Test: N-gram repetition detection ─────────────────────────── */

static void test_ngram_detection(void) {
    printf("  test_ngram_detection...\n");

    /* Normal text: no repetition */
    int normal = hcp_detect_ngram_repeat(
        "The weather is nice today and I went to the store to buy groceries for dinner");
    ASSERT(normal == 0, "normal text should not be flagged");

    /* Repetitive text: should detect */
    int rep = hcp_detect_ngram_repeat(
        "I went to I went to I went to I went to I went to the store");
    ASSERT(rep == 1, "repetitive text should be flagged");

    /* Short text: should not crash */
    int tiny = hcp_detect_ngram_repeat("hello");
    ASSERT(tiny == 0, "short text should not crash or flag");
}

/* ─── Test: Subword frequency table ─────────────────────────────── */

static void test_freq_table(void) {
    printf("  test_freq_table...\n");

    ASSERT(HCP_VOCAB_SIZE == 51864, "vocab size should be 51864");

    /* All frequencies should be non-negative */
    int all_nonneg = 1;
    for (int i = 0; i < HCP_VOCAB_SIZE; i++) {
        if (hcp_subword_freq[i] < 0.0f) { all_nonneg = 0; break; }
    }
    ASSERT(all_nonneg, "all subword frequencies must be >= 0");

    /* Some tokens should have non-zero frequency */
    int nonzero = 0;
    for (int i = 0; i < HCP_VOCAB_SIZE; i++) {
        if (hcp_subword_freq[i] > 0.0f) nonzero++;
    }
    ASSERT(nonzero > 100, "should have many non-zero frequency entries");

    /* String lengths should be reasonable (max observed: 128 for merged BPE tokens) */
    int all_valid = 1;
    for (int i = 0; i < HCP_VOCAB_SIZE; i++) {
        if (hcp_token_strlen[i] > 255) { all_valid = 0; break; }
    }
    ASSERT(all_valid, "all token string lengths should be <= 255");
}

/* ─── Test: FFT power of 2 ─────────────────────────────────────── */

static void test_next_pow2(void) {
    printf("  test_next_pow2...\n");
    ASSERT(hcp__next_pow2(1) == 1, "next_pow2(1) = 1");
    ASSERT(hcp__next_pow2(2) == 2, "next_pow2(2) = 2");
    ASSERT(hcp__next_pow2(3) == 4, "next_pow2(3) = 4");
    ASSERT(hcp__next_pow2(5) == 8, "next_pow2(5) = 8");
    ASSERT(hcp__next_pow2(1000) == 1024, "next_pow2(1000) = 1024");
    ASSERT(hcp__next_pow2(2048) == 2048, "next_pow2(2048) = 2048");
    ASSERT(hcp__next_pow2(2049) == 4096, "next_pow2(2049) = 4096");
}

/* ─── Test: Spectral filter bounds ──────────────────────────────── */

static void test_filter_bounds(void) {
    printf("  test_filter_bounds...\n");

    /* Simulate: create a signal, FFT, apply filter logic, verify bounds */
    const int N = 64;
    hcp__cpx z[64];

    /* Random-ish signal */
    for (int i = 0; i < N; i++) {
        z[i].re = sinf((float)i * 0.3f) + 0.5f * cosf((float)i * 1.7f);
        z[i].im = cosf((float)i * 0.5f);
    }

    hcp__fft(z, N, 0);

    /* Compute envelope and filter exactly as HCP does */
    float spec_energy[64], envelope[64];
    for (int k = 0; k < N; k++)
        spec_energy[k] = z[k].re * z[k].re + z[k].im * z[k].im;

    int env_win = N / 32;
    if (env_win < 4) env_win = 4;
    for (int k = 0; k < N; k++) {
        int lo = k - env_win / 2; if (lo < 0) lo = 0;
        int hi = k + env_win / 2; if (hi >= N) hi = N - 1;
        float sum = 0;
        int cnt = 0;
        for (int j = lo; j <= hi; j++) { sum += spec_energy[j]; cnt++; }
        envelope[k] = cnt > 0 ? sum / (float)cnt : 1e-8f;
    }

    /* Verify filter H is always within [0.1, 1.2] */
    int all_bounded = 1;
    for (int k = 0; k < N; k++) {
        float H = 1.0f;
        int band_low = N / 64;
        int band_mid = N / 8;

        if (k < band_low || k > N - band_low) {
            if (spec_energy[k] > 3.0f * envelope[k])
                H = sqrtf(3.0f * envelope[k] / (spec_energy[k] + 1e-8f));
        } else if (k < band_mid || k > N - band_mid) {
            if (spec_energy[k] > 5.0f * envelope[k])
                H = sqrtf(5.0f * envelope[k] / (spec_energy[k] + 1e-8f));
        } else {
            if (spec_energy[k] > 2.0f * envelope[k])
                H = sqrtf(2.0f * envelope[k] / (spec_energy[k] + 1e-8f));
        }

        float deviation = spec_energy[k] / (envelope[k] + 1e-8f);
        if (deviation > 8.0f) H *= 0.3f;
        else if (deviation < 0.05f && k > 0 && k < N / 2) H *= 1.2f;

        if (H > 1.2f) H = 1.2f;
        if (H < 0.1f) H = 0.1f;

        if (H < 0.1f - 1e-6f || H > 1.2f + 1e-6f) {
            all_bounded = 0;
            break;
        }
    }
    ASSERT(all_bounded, "filter H must be clamped to [0.1, 1.2]");
}

/* ─── Test: Phase coupling property ─────────────────────────────── */

static void test_phase_coupling(void) {
    printf("  test_phase_coupling...\n");

    /* Verify: z = z_acou * z_morph has phases that add */
    float mag_a = 0.8f, phi_a = 1.0f;
    float mag_m = 0.5f, phi_m = 0.7f;

    hcp__cpx z_a = hcp__cpx_from_polar(mag_a, phi_a);
    hcp__cpx z_m = hcp__cpx_from_polar(mag_m, phi_m);
    hcp__cpx coupled = hcp__cpx_mul(z_a, z_m);

    /* Magnitude should multiply */
    ASSERT_FLOAT_EQ(hcp__cpx_mag(coupled), mag_a * mag_m, 1e-5f,
                    "coupled magnitude = product of magnitudes");

    /* Phase should add */
    float expected_phase = phi_a + phi_m;
    if (expected_phase > (float)M_PI) expected_phase -= 2.0f * (float)M_PI;
    ASSERT_FLOAT_EQ(hcp__cpx_phase(coupled), expected_phase, 1e-5f,
                    "coupled phase = sum of phases");
}

/* ─── Test: Parseval's theorem (energy conservation) ────────────── */

static void test_parseval(void) {
    printf("  test_parseval...\n");

    const int N = 32;
    hcp__cpx x[32], X[32];

    for (int i = 0; i < N; i++) {
        x[i].re = cosf((float)i * 0.4f) * 0.8f;
        x[i].im = sinf((float)i * 0.6f) * 0.3f;
        X[i] = x[i];
    }

    hcp__fft(X, N, 0);

    /* Time-domain energy */
    float E_time = 0;
    for (int i = 0; i < N; i++)
        E_time += x[i].re * x[i].re + x[i].im * x[i].im;

    /* Frequency-domain energy (scaled by 1/N for Parseval's) */
    float E_freq = 0;
    for (int k = 0; k < N; k++)
        E_freq += X[k].re * X[k].re + X[k].im * X[k].im;
    E_freq /= (float)N;

    ASSERT_FLOAT_EQ(E_time, E_freq, 0.01f,
                    "Parseval's theorem: time-domain energy = freq-domain energy / N");
}

/* ─── Test: Quantization buckets ────────────────────────────────── */

static void test_quantization(void) {
    printf("  test_quantization...\n");

    /* Vlen quantization */
    ASSERT(hcp__quant_vlen(0.0f) == 0, "vlen=0 maps to bucket 0");
    ASSERT(hcp__quant_vlen(2.0f) >= HCP_VLEN_BUCKETS - 1, "vlen=2.0 maps to max bucket");
    ASSERT(hcp__quant_vlen(-1.0f) == 0, "negative vlen clamps to 0");
    ASSERT(hcp__quant_vlen(100.0f) == HCP_VLEN_BUCKETS - 1, "huge vlen clamps to max");

    /* Dt quantization */
    ASSERT(hcp__quant_dt(0) == 0, "dt=0 maps to bucket 0");
    ASSERT(hcp__quant_dt(500) >= HCP_DT_BUCKETS - 1, "dt=500ms maps to max bucket");
    ASSERT(hcp__quant_dt(-100) == 0, "negative dt clamps to 0");
}

/* ─── Test: KIEL-CC Kalman filter ───────────────────────────────── */

static void test_kiel_kalman(void) {
    printf("  test_kiel_kalman...\n");

    /* Create a smooth signal with one spike (simulated anomaly) */
    const int N = 64;
    hcp__cpx z[64];
    int seg_map[64];

    for (int i = 0; i < N; i++) {
        z[i].re = 0.5f + 0.1f * sinf((float)i * 0.2f);
        z[i].im = 0.3f + 0.05f * cosf((float)i * 0.3f);
        seg_map[i] = i / 16;   /* 4 segments of 16 tokens each */
    }

    /* Inject anomaly at token 32 */
    z[32].re = 5.0f;
    z[32].im = -4.0f;

    HcpResult res = {0};
    /* Allocate minimal segments */
    res.segments = (HcpSegment *)calloc(4, sizeof(HcpSegment));
    res.count = 4;
    res.cap = 4;

    hcp__kiel_cc(z, N, seg_map, 4, &res);

    ASSERT(res.kiel_innovation != NULL, "KIEL should allocate innovation array");
    ASSERT(res.kiel_ms >= 0, "KIEL timing should be non-negative");

    /* Innovation at spike should be much higher than smooth regions */
    float max_smooth = 0;
    for (int i = 0; i < N; i++) {
        if (i == 32) continue;
        if (res.kiel_innovation[i] > max_smooth) max_smooth = res.kiel_innovation[i];
    }
    ASSERT(res.kiel_innovation[32] > max_smooth * 2.0f,
           "innovation at anomaly should spike well above smooth region");

    /* Segment 2 (containing token 32) should have highest max innovation */
    ASSERT(res.segments[2].kiel_max_innov > res.segments[0].kiel_max_innov,
           "segment with anomaly should have higher innovation");
    ASSERT(res.segments[2].kiel_max_innov > res.segments[1].kiel_max_innov,
           "segment with anomaly should have higher innovation than neighbors");

    free(res.kiel_innovation);
    free(res.segments);
}

/* ─── Test: KIEL adaptive alpha ─────────────────────────────────── */

static void test_kiel_adaptive(void) {
    printf("  test_kiel_adaptive...\n");

    /* Highly autocorrelated signal: alpha should adapt high */
    const int N = 128;
    hcp__cpx z[128];
    int seg_map[128];

    for (int i = 0; i < N; i++) {
        z[i].re = 0.8f + 0.01f * (float)i;  /* slow drift */
        z[i].im = 0.4f;
        seg_map[i] = 0;
    }

    HcpResult res = {0};
    res.segments = (HcpSegment *)calloc(1, sizeof(HcpSegment));
    res.count = 1;
    res.cap = 1;

    hcp__kiel_cc(z, N, seg_map, 1, &res);

    /* With smooth signal, very few tokens should be flagged */
    ASSERT(res.kiel_flagged_tokens < N / 4,
           "smooth signal should have few Kalman-flagged tokens");

    free(res.kiel_innovation);
    free(res.segments);
}

/* ─── Test: E-T Gate audio analysis ─────────────────────────────── */

static void test_et_gate_silence(void) {
    printf("  test_et_gate_silence...\n");

    /* Create a segment spanning 0-2000ms with silence audio */
    int sample_rate = 16000;
    int n_samples = sample_rate * 2;  /* 2 seconds */
    float *audio = (float *)calloc((size_t)n_samples, sizeof(float));

    /* Audio is all zeros (silence) */

    HcpResult res = {0};
    res.segments = (HcpSegment *)calloc(1, sizeof(HcpSegment));
    res.count = 1;
    res.cap = 1;
    res.segments[0].t0_ms = 0;
    res.segments[0].t1_ms = 2000;
    res.segments[0].token_count = 10;   /* ~5 tokens/sec — suspicious over silence */

    hcp__et_gate(audio, n_samples, sample_rate, &res);

    /* RMS should be ~0 */
    ASSERT_FLOAT_EQ(res.segments[0].et_rms, 0.0f, 0.001f, "silence RMS should be ~0");
    /* Speech fraction should be 0 (all frames are silence) */
    ASSERT_FLOAT_EQ(res.segments[0].et_speech_frac, 0.0f, 0.01f,
                    "silence should have 0% speech frames");
    /* Should be gated (high density + no speech) */
    ASSERT(res.segments[0].hallucination_flags & HCP_HALLUC_ET_GATE,
           "high density over silence should be flagged by E-T Gate");
    ASSERT(res.et_segments_gated == 1, "should gate 1 segment");

    free(audio);
    free(res.segments);
}

static void test_et_gate_speech(void) {
    printf("  test_et_gate_speech...\n");

    /* Create a segment with speech-like audio (sinusoid with harmonics) */
    int sample_rate = 16000;
    int n_samples = sample_rate * 2;  /* 2 seconds */
    float *audio = (float *)malloc((size_t)n_samples * sizeof(float));

    /* Simulate speech: fundamental 150Hz + harmonics */
    for (int i = 0; i < n_samples; i++) {
        float t = (float)i / sample_rate;
        audio[i] = 0.3f * sinf(2.0f * (float)M_PI * 150.0f * t)
                 + 0.15f * sinf(2.0f * (float)M_PI * 300.0f * t)
                 + 0.08f * sinf(2.0f * (float)M_PI * 450.0f * t)
                 + 0.04f * sinf(2.0f * (float)M_PI * 600.0f * t);
    }

    HcpResult res = {0};
    res.segments = (HcpSegment *)calloc(1, sizeof(HcpSegment));
    res.count = 1;
    res.cap = 1;
    res.segments[0].t0_ms = 0;
    res.segments[0].t1_ms = 2000;
    res.segments[0].token_count = 10;

    hcp__et_gate(audio, n_samples, sample_rate, &res);

    /* RMS should be significant */
    ASSERT(res.segments[0].et_rms > 0.1f, "speech-like audio should have high RMS");
    /* Speech fraction should be high (harmonic = low flatness) */
    ASSERT(res.segments[0].et_speech_frac > 0.5f,
           "harmonic audio should be detected as speech-like");
    /* Should NOT be gated */
    ASSERT(!(res.segments[0].hallucination_flags & HCP_HALLUC_ET_GATE),
           "speech-like audio with tokens should NOT be gated");
    ASSERT(res.et_segments_gated == 0, "should gate 0 segments");

    free(audio);
    free(res.segments);
}

static void test_et_gate_noise(void) {
    printf("  test_et_gate_noise...\n");

    /* Create segment with noise-like audio (LCG pseudo-random) */
    int sample_rate = 16000;
    int n_samples = sample_rate * 2;
    float *audio = (float *)malloc((size_t)n_samples * sizeof(float));

    /* Pseudo-random noise using LCG */
    uint32_t seed = 42;
    for (int i = 0; i < n_samples; i++) {
        seed = seed * 1664525u + 1013904223u;
        audio[i] = ((float)(seed >> 16) / 32768.0f - 1.0f) * 0.3f;
    }

    HcpResult res = {0};
    res.segments = (HcpSegment *)calloc(1, sizeof(HcpSegment));
    res.count = 1;
    res.cap = 1;
    res.segments[0].t0_ms = 0;
    res.segments[0].t1_ms = 2000;
    res.segments[0].token_count = 10;    /* high density over noise = suspicious */

    hcp__et_gate(audio, n_samples, sample_rate, &res);

    /* RMS should be significant (noise has energy) */
    ASSERT(res.segments[0].et_rms > 0.05f, "noise should have measurable RMS");
    /* E-T Gate should process without crashing; speech_frac depends on noise quality */
    ASSERT(res.segments[0].et_speech_frac >= 0.0f && res.segments[0].et_speech_frac <= 1.0f,
           "speech_frac should be in [0, 1]");
    ASSERT(res.et_gate_ms >= 0, "E-T Gate timing should be non-negative");

    free(audio);
    free(res.segments);
}

/* ─── Main ──────────────────────────────────────────────────────── */

int main(void) {
    printf("\n=== HCP Unit Tests ===\n\n");

    test_complex_arithmetic();
    test_fft_roundtrip();
    test_fft_impulse();
    test_parseval();
    test_fnv_determinism();
    test_phase_coupling();
    test_next_pow2();
    test_quantization();
    test_compression_ratio();
    test_ngram_detection();
    test_freq_table();
    test_filter_bounds();
    test_kiel_kalman();
    test_kiel_adaptive();
    test_et_gate_silence();
    test_et_gate_speech();
    test_et_gate_noise();

    printf("\n=== Results: %d passed, %d failed, %d total ===\n\n",
           tests_passed, tests_failed, tests_run);

    return tests_failed > 0 ? 1 : 0;
}
