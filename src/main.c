/* hcp-whisper — Standalone transcription binary with HCP refinement
 *
 * Usage:
 *   hcp-whisper <audio.wav> [output-dir] [options]
 *
 * Options:
 *   --model <path>      Path to whisper GGML model (default: auto-detect)
 *   --language <lang>    Language code (default: "en")
 *   --beam-size <n>      Beam search width (default: 5)
 *   --threads <n>        CPU threads (default: 4)
 *   --no-hcp             Disable HCP refinement (baseline only)
 *   --no-gpu             Disable GPU acceleration
 *   --json               Output JSON with full HCP metrics
 *   --txt                Output plain text
 *   --srt                Output SRT subtitles
 *   --vtt                Output VTT subtitles
 *   --all                Output all formats (default)
 *
 * MIT License — Copyright (c) 2026 Nick Gonzales
 */

#define HCP_IMPLEMENTATION
#include "hcp.h"

#include <ggml-backend.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/* ─── WAV loader ────────────────────────────────────────────────── */

static float *load_wav_f32(const char *path, int *out_samples, int *out_rate) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "error: cannot open %s\n", path); return NULL; }

    /* Read RIFF header */
    char riff[4]; fread(riff, 1, 4, fp);
    if (memcmp(riff, "RIFF", 4) != 0) { fclose(fp); return NULL; }

    uint32_t file_size; fread(&file_size, 4, 1, fp);
    char wave[4]; fread(wave, 1, 4, fp);
    if (memcmp(wave, "WAVE", 4) != 0) { fclose(fp); return NULL; }

    uint16_t audio_format = 0, channels = 0, bits_per_sample = 0;
    uint32_t sample_rate = 0, data_size = 0;
    int found_fmt = 0, found_data = 0;

    while (!found_data) {
        char chunk_id[4];
        uint32_t chunk_size;
        if (fread(chunk_id, 1, 4, fp) != 4) break;
        if (fread(&chunk_size, 4, 1, fp) != 1) break;

        if (memcmp(chunk_id, "fmt ", 4) == 0) {
            fread(&audio_format, 2, 1, fp);
            fread(&channels, 2, 1, fp);
            fread(&sample_rate, 4, 1, fp);
            fseek(fp, 6, SEEK_CUR); /* byte rate + block align */
            fread(&bits_per_sample, 2, 1, fp);
            if (chunk_size > 16) fseek(fp, chunk_size - 16, SEEK_CUR);
            found_fmt = 1;
        } else if (memcmp(chunk_id, "data", 4) == 0) {
            data_size = chunk_size;
            found_data = 1;
        } else {
            fseek(fp, chunk_size, SEEK_CUR);
        }
    }

    if (!found_fmt || !found_data || audio_format != 1) {
        fprintf(stderr, "error: unsupported WAV format (need PCM)\n");
        fclose(fp);
        return NULL;
    }

    int n_samples = (int)(data_size / (bits_per_sample / 8) / channels);
    float *pcm = (float *)malloc((size_t)n_samples * sizeof(float));
    if (!pcm) { fclose(fp); return NULL; }

    if (bits_per_sample == 16) {
        int16_t *buf = (int16_t *)malloc(data_size);
        if (!buf) { free(pcm); fclose(fp); return NULL; }
        fread(buf, 1, data_size, fp);
        for (int i = 0; i < n_samples; i++) {
            /* If stereo, take first channel */
            pcm[i] = (float)buf[i * channels] / 32768.0f;
        }
        free(buf);
    } else {
        fprintf(stderr, "error: unsupported bit depth %d\n", bits_per_sample);
        free(pcm);
        fclose(fp);
        return NULL;
    }

    fclose(fp);
    *out_samples = n_samples;
    *out_rate = (int)sample_rate;
    return pcm;
}

/* Simple linear resampling */
static float *resample(const float *src, int src_len, int src_rate, int dst_rate, int *dst_len) {
    if (src_rate == dst_rate) {
        *dst_len = src_len;
        float *out = (float *)malloc((size_t)src_len * sizeof(float));
        if (out) memcpy(out, src, (size_t)src_len * sizeof(float));
        return out;
    }
    double ratio = (double)dst_rate / (double)src_rate;
    *dst_len = (int)((double)src_len * ratio);
    float *out = (float *)malloc((size_t)(*dst_len) * sizeof(float));
    if (!out) return NULL;
    for (int i = 0; i < *dst_len; i++) {
        double src_pos = (double)i / ratio;
        int idx = (int)src_pos;
        double frac = src_pos - idx;
        if (idx + 1 < src_len)
            out[i] = (float)((1.0 - frac) * src[idx] + frac * src[idx + 1]);
        else if (idx < src_len)
            out[i] = src[idx];
        else
            out[i] = 0.0f;
    }
    return out;
}

/* ─── Output writers ────────────────────────────────────────────── */

static void ensure_dir(const char *path) {
    char tmp[PATH_MAX];
    size_t len = strlen(path);
    if (len == 0 || len >= sizeof(tmp)) return;
    strcpy(tmp, path);
    for (size_t i = 1; i < len; i++) {
        if (tmp[i] == '/') {
            tmp[i] = '\0';
            mkdir(tmp, 0755);
            tmp[i] = '/';
        }
    }
    mkdir(tmp, 0755);
}

static void write_json(const char *path, HcpResult *res) {
    FILE *fp = fopen(path, "w");
    if (!fp) return;

    float q_base_sum = 0, q_hcp_sum = 0;
    for (int i = 0; i < res->count; i++) {
        q_base_sum += res->segments[i].quality;
        q_hcp_sum += res->segments[i].hcp_quality;
    }

    fprintf(fp, "{\n");
    fprintf(fp, "  \"algorithm\": \"Complex-Domain HCP Spectral Refinement\",\n");
    fprintf(fp, "  \"version\": \"2.0.0\",\n");
    fprintf(fp, "  \"license\": \"MIT\",\n");
    fprintf(fp, "  \"total_segments\": %d,\n", res->count);
    fprintf(fp, "  \"hallucinated_segments\": %d,\n", res->segments_hallucinated);
    fprintf(fp, "  \"hcp\": {\n");
    fprintf(fp, "    \"tokens\": %d,\n", res->hcp_tokens);
    fprintf(fp, "    \"padded_fft_size\": %d,\n", res->hcp_padded);
    fprintf(fp, "    \"flagged_tokens\": %d,\n", res->hcp_flagged_tokens);
    fprintf(fp, "    \"flagged_segments\": %d,\n", res->hcp_flagged_segments);
    fprintf(fp, "    \"elapsed_ms\": %.1f,\n", res->hcp_ms);
    fprintf(fp, "    \"quality_base_avg\": %.4f,\n", res->count > 0 ? q_base_sum / res->count : 0.0f);
    fprintf(fp, "    \"quality_hcp_avg\": %.4f,\n", res->count > 0 ? q_hcp_sum / res->count : 0.0f);
    fprintf(fp, "    \"quality_uplift_pct\": %.1f,\n",
            res->count > 0 && q_base_sum > 0 ? ((q_hcp_sum - q_base_sum) / q_base_sum * 100.0f) : 0.0f);
    fprintf(fp, "    \"kiel\": {\n");
    fprintf(fp, "      \"flagged_tokens\": %d,\n", res->kiel_flagged_tokens);
    fprintf(fp, "      \"elapsed_ms\": %.1f\n", res->kiel_ms);
    fprintf(fp, "    },\n");
    fprintf(fp, "    \"et_gate\": {\n");
    fprintf(fp, "      \"segments_gated\": %d,\n", res->et_segments_gated);
    fprintf(fp, "      \"elapsed_ms\": %.1f\n", res->et_gate_ms);
    fprintf(fp, "    }\n");
    fprintf(fp, "  },\n");
    fprintf(fp, "  \"segments\": [\n");

    for (int i = 0; i < res->count; i++) {
        HcpSegment *s = &res->segments[i];
        fprintf(fp, "    {\n");
        fprintf(fp, "      \"t0_ms\": %lld,\n", (long long)s->t0_ms);
        fprintf(fp, "      \"t1_ms\": %lld,\n", (long long)s->t1_ms);
        fprintf(fp, "      \"confidence\": %.4f,\n", s->confidence);
        fprintf(fp, "      \"quality\": %.4f,\n", s->quality);
        fprintf(fp, "      \"hcp_quality\": %.4f,\n", s->hcp_quality);
        fprintf(fp, "      \"hallucination_flags\": %d,\n", s->hallucination_flags);
        fprintf(fp, "      \"hcp_flagged_tokens\": %d,\n", s->hcp_flagged_count);
        fprintf(fp, "      \"et_rms\": %.4f,\n", s->et_rms);
        fprintf(fp, "      \"et_speech_frac\": %.2f,\n", s->et_speech_frac);
        fprintf(fp, "      \"kiel_max_innovation\": %.2f,\n", s->kiel_max_innov);
        fprintf(fp, "      \"token_count\": %d,\n", s->token_count);
        fprintf(fp, "      \"speaker_turn\": %s,\n", s->speaker_turn ? "true" : "false");
        /* JSON-safe text */
        fprintf(fp, "      \"text\": \"");
        for (const char *c = s->text; *c; c++) {
            switch (*c) {
                case '"':  fprintf(fp, "\\\""); break;
                case '\\': fprintf(fp, "\\\\"); break;
                case '\n': fprintf(fp, "\\n");  break;
                case '\r': fprintf(fp, "\\r");  break;
                case '\t': fprintf(fp, "\\t");  break;
                default:
                    if ((unsigned char)*c < 0x20) fprintf(fp, "\\u%04x", (unsigned char)*c);
                    else fputc(*c, fp);
            }
        }
        fprintf(fp, "\"\n");
        fprintf(fp, "    }%s\n", (i + 1 < res->count) ? "," : "");
    }
    fprintf(fp, "  ]\n}\n");
    fclose(fp);
}

static void write_txt(const char *path, HcpResult *res) {
    FILE *fp = fopen(path, "w");
    if (!fp) return;
    for (int i = 0; i < res->count; i++)
        fprintf(fp, "%s", res->segments[i].text);
    fprintf(fp, "\n");
    fclose(fp);
}

static void write_srt(const char *path, HcpResult *res) {
    FILE *fp = fopen(path, "w");
    if (!fp) return;
    for (int i = 0; i < res->count; i++) {
        HcpSegment *s = &res->segments[i];
        int64_t t0 = s->t0_ms, t1 = s->t1_ms;
        fprintf(fp, "%d\n", i + 1);
        fprintf(fp, "%02lld:%02lld:%02lld,%03lld --> %02lld:%02lld:%02lld,%03lld\n",
                t0/3600000, (t0/60000)%60, (t0/1000)%60, t0%1000,
                t1/3600000, (t1/60000)%60, (t1/1000)%60, t1%1000);
        fprintf(fp, "%s\n\n", s->text);
    }
    fclose(fp);
}

static void write_vtt(const char *path, HcpResult *res) {
    FILE *fp = fopen(path, "w");
    if (!fp) return;
    fprintf(fp, "WEBVTT\n\n");
    for (int i = 0; i < res->count; i++) {
        HcpSegment *s = &res->segments[i];
        int64_t t0 = s->t0_ms, t1 = s->t1_ms;
        fprintf(fp, "%02lld:%02lld:%02lld.%03lld --> %02lld:%02lld:%02lld.%03lld\n",
                t0/3600000, (t0/60000)%60, (t0/1000)%60, t0%1000,
                t1/3600000, (t1/60000)%60, (t1/1000)%60, t1%1000);
        fprintf(fp, "%s\n\n", s->text);
    }
    fclose(fp);
}

/* ─── Model auto-detection ──────────────────────────────────────── */

static const char *find_model(void) {
    static char path[PATH_MAX];
    const char *candidates[] = {
        NULL, /* will be filled with $HOME/.local/share/whisper/ggml-base.en-q5_0.bin */
        NULL, /* $HOME/.local/share/whisper/ggml-base.en.bin */
        "/opt/homebrew/share/whisper/ggml-base.en.bin",
        NULL
    };

    const char *home = getenv("HOME");
    char p0[PATH_MAX], p1[PATH_MAX];
    if (home) {
        snprintf(p0, sizeof(p0), "%s/.local/share/whisper/ggml-base.en-q5_0.bin", home);
        snprintf(p1, sizeof(p1), "%s/.local/share/whisper/ggml-base.en.bin", home);
        candidates[0] = p0;
        candidates[1] = p1;
    }

    for (int i = 0; candidates[i]; i++) {
        if (access(candidates[i], R_OK) == 0) {
            strncpy(path, candidates[i], sizeof(path) - 1);
            return path;
        }
    }
    return NULL;
}

/* ─── Usage ─────────────────────────────────────────────────────── */

static void usage(const char *prog) {
    fprintf(stderr,
        "hcp-whisper — Whisper transcription with Complex-Domain HCP refinement\n"
        "\n"
        "Usage: %s <audio.wav> [output-dir] [options]\n"
        "\n"
        "Options:\n"
        "  --model <path>     Path to whisper GGML model\n"
        "  --language <lang>  Language code (default: en)\n"
        "  --beam-size <n>    Beam search width (default: 5)\n"
        "  --threads <n>      CPU threads (default: 4)\n"
        "  --no-hcp           Disable HCP refinement\n"
        "  --no-gpu           Disable GPU acceleration\n"
        "  --json             Output JSON only\n"
        "  --txt              Output text only\n"
        "  --srt              Output SRT only\n"
        "  --vtt              Output VTT only\n"
        "  --all              Output all formats (default)\n"
        "  -h, --help         Show this help\n"
        "\n"
        "MIT License — https://github.com/Nickgonzales76017/hcp-whisper\n",
        prog);
}

/* ─── Main ──────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    if (argc < 2) { usage(argv[0]); return 1; }

    /* Parse arguments */
    const char *audio_path = NULL;
    const char *output_dir = NULL;
    const char *model_path = NULL;
    const char *language = "en";
    int beam_size = 5;
    int threads = 4;
    int use_hcp = 1;
    int use_gpu = 1;
    int fmt_json = 0, fmt_txt = 0, fmt_srt = 0, fmt_vtt = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv[0]); return 0;
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--language") == 0 && i + 1 < argc) {
            language = argv[++i];
        } else if (strcmp(argv[i], "--beam-size") == 0 && i + 1 < argc) {
            beam_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--no-hcp") == 0) {
            use_hcp = 0;
        } else if (strcmp(argv[i], "--no-gpu") == 0) {
            use_gpu = 0;
        } else if (strcmp(argv[i], "--json") == 0) { fmt_json = 1;
        } else if (strcmp(argv[i], "--txt") == 0) { fmt_txt = 1;
        } else if (strcmp(argv[i], "--srt") == 0) { fmt_srt = 1;
        } else if (strcmp(argv[i], "--vtt") == 0) { fmt_vtt = 1;
        } else if (strcmp(argv[i], "--all") == 0) {
            fmt_json = fmt_txt = fmt_srt = fmt_vtt = 1;
        } else if (!audio_path) {
            audio_path = argv[i];
        } else if (!output_dir) {
            output_dir = argv[i];
        }
    }

    if (!audio_path) { fprintf(stderr, "error: no audio file specified\n"); return 1; }
    if (!fmt_json && !fmt_txt && !fmt_srt && !fmt_vtt) {
        /* Default: all formats */
        fmt_json = fmt_txt = fmt_srt = fmt_vtt = 1;
    }
    if (!output_dir) output_dir = ".";

    /* Auto-detect model */
    if (!model_path) model_path = find_model();
    if (!model_path) {
        fprintf(stderr, "error: no whisper model found. Use --model <path>\n"
                        "  Download: whisper-cli --download-model base.en\n");
        return 1;
    }

    /* Load ggml backends (REQUIRED for whisper 1.8+) */
    ggml_backend_load_all();

    /* Load audio */
    fprintf(stderr, "[hcp-whisper] loading audio: %s\n", audio_path);
    int n_samples, sample_rate;
    float *pcm = load_wav_f32(audio_path, &n_samples, &sample_rate);
    if (!pcm) return 1;

    /* Resample to 16kHz if needed */
    float *audio = pcm;
    int audio_len = n_samples;
    if (sample_rate != 16000) {
        fprintf(stderr, "[hcp-whisper] resampling %d → 16000 Hz\n", sample_rate);
        audio = resample(pcm, n_samples, sample_rate, 16000, &audio_len);
        free(pcm);
        if (!audio) { fprintf(stderr, "error: resample failed\n"); return 1; }
    }

    fprintf(stderr, "[hcp-whisper] audio: %.1f seconds, %d samples\n",
            (double)audio_len / 16000.0, audio_len);

    /* Init whisper */
    fprintf(stderr, "[hcp-whisper] loading model: %s\n", model_path);
    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = use_gpu;
    cparams.flash_attn = use_gpu;
    cparams.dtw_token_timestamps = 1;
    cparams.dtw_aheads_preset = WHISPER_AHEADS_BASE_EN;

    struct whisper_context *ctx = whisper_init_from_file_with_params(model_path, cparams);
    if (!ctx) { fprintf(stderr, "error: failed to load model\n"); free(audio); return 1; }

    /* Decode */
    fprintf(stderr, "[hcp-whisper] decoding with beam_size=%d, threads=%d, hcp=%s\n",
            beam_size, threads, use_hcp ? "on" : "off");

    struct whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH);
    wparams.n_threads = threads;
    wparams.language = language;
    wparams.translate = 0;
    wparams.beam_search.beam_size = beam_size;
    wparams.token_timestamps = 1;
    wparams.tdrz_enable = 1;    /* TinyDiarize */
    wparams.no_timestamps = 0;
    wparams.print_progress = 1;
    wparams.print_timestamps = 0;
    wparams.print_special = 0;

    double t_start = hcp__ms_now();
    int ret = whisper_full(ctx, wparams, audio, audio_len);
    double decode_ms = hcp__ms_now() - t_start;

    if (ret != 0) {
        fprintf(stderr, "error: whisper_full() failed (%d)\n", ret);
        free(audio);
        whisper_free(ctx);
        return 1;
    }

    fprintf(stderr, "[hcp-whisper] decode: %.1f ms (%d segments)\n",
            decode_ms, whisper_full_n_segments(ctx));

    /* Run HCP (with audio for E-T Gate) */
    HcpResult result = hcp_process_with_audio(ctx, audio, audio_len, 16000);
    result.decode_ms = decode_ms;

    free(audio);

    if (use_hcp) {
        fprintf(stderr, "[hcp-whisper] HCP: %d tokens, %d flagged (%.1f%%), "
                "%d segments flagged, %.1f ms\n",
                result.hcp_tokens, result.hcp_flagged_tokens,
                result.hcp_tokens > 0 ? (float)result.hcp_flagged_tokens / result.hcp_tokens * 100 : 0,
                result.hcp_flagged_segments, result.hcp_ms);

        fprintf(stderr, "[hcp-whisper] KIEL-CC: %d tokens flagged, %.1f ms\n",
                result.kiel_flagged_tokens, result.kiel_ms);
        fprintf(stderr, "[hcp-whisper] E-T Gate: %d segments gated, %.1f ms\n",
                result.et_segments_gated, result.et_gate_ms);

        /* Print quality summary */
        float q_base = 0, q_hcp = 0;
        for (int i = 0; i < result.count; i++) {
            q_base += result.segments[i].quality;
            q_hcp += result.segments[i].hcp_quality;
        }
        if (result.count > 0) {
            q_base /= result.count;
            q_hcp /= result.count;
            fprintf(stderr, "[hcp-whisper] quality: %.4f (base) → %.4f (hcp) [+%.1f%%]\n",
                    q_base, q_hcp, (q_hcp - q_base) / q_base * 100);
        }
    }

    fprintf(stderr, "[hcp-whisper] hallucinated segments: %d / %d\n",
            result.segments_hallucinated, result.count);

    /* Write outputs */
    ensure_dir(output_dir);
    char path[PATH_MAX];

    if (fmt_json) {
        snprintf(path, sizeof(path), "%s/transcript.json", output_dir);
        write_json(path, &result);
        fprintf(stderr, "[hcp-whisper] wrote %s\n", path);
    }
    if (fmt_txt) {
        snprintf(path, sizeof(path), "%s/transcript.txt", output_dir);
        write_txt(path, &result);
        fprintf(stderr, "[hcp-whisper] wrote %s\n", path);
    }
    if (fmt_srt) {
        snprintf(path, sizeof(path), "%s/transcript.srt", output_dir);
        write_srt(path, &result);
        fprintf(stderr, "[hcp-whisper] wrote %s\n", path);
    }
    if (fmt_vtt) {
        snprintf(path, sizeof(path), "%s/transcript.vtt", output_dir);
        write_vtt(path, &result);
        fprintf(stderr, "[hcp-whisper] wrote %s\n", path);
    }

    hcp_free(&result);
    whisper_free(ctx);
    return 0;
}
