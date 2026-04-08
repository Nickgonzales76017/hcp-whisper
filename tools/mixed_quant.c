/* mixed_quant.c — Mixed-bit quantizer for whisper GGML models
 *
 * Reads a fp16 whisper model and applies per-tensor quantization:
 *   - Embeddings + positional: keep as-is (f32/f16)
 *   - Attention Q/K/V/Out weights: q5_0 (preserve decoder accuracy)
 *   - FFN mlp weights: q4_0 (aggressive compression)
 *   - LayerNorm, biases: keep as f32 (tiny, precision-critical)
 *
 * This produces a model smaller than uniform q4_0 while maintaining
 * attention quality that matters for decoder coherence.
 *
 * Build: cc -O2 -o mixed-quant tools/mixed_quant.c -L/opt/homebrew/lib -lggml -lm
 * Usage: ./mixed-quant input.bin output.bin [attn_quant] [ffn_quant]
 *        attn_quant: q4_0, q5_0, q5_1, q8_0 (default: q5_0)
 *        ffn_quant:  q4_0, q4_1, q5_0 (default: q4_0)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <ggml.h>

/* ── GGML type helpers ──────────────────────────────────────────── */

static enum ggml_type parse_quant_type(const char *s) {
    if (strcmp(s, "q4_0") == 0) return GGML_TYPE_Q4_0;
    if (strcmp(s, "q4_1") == 0) return GGML_TYPE_Q4_1;
    if (strcmp(s, "q5_0") == 0) return GGML_TYPE_Q5_0;
    if (strcmp(s, "q5_1") == 0) return GGML_TYPE_Q5_1;
    if (strcmp(s, "q8_0") == 0) return GGML_TYPE_Q8_0;
    fprintf(stderr, "Unknown quant type: %s\n", s);
    exit(1);
}

static const char *type_name(enum ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32:  return "f32";
        case GGML_TYPE_F16:  return "f16";
        case GGML_TYPE_Q4_0: return "q4_0";
        case GGML_TYPE_Q4_1: return "q4_1";
        case GGML_TYPE_Q5_0: return "q5_0";
        case GGML_TYPE_Q5_1: return "q5_1";
        case GGML_TYPE_Q8_0: return "q8_0";
        default: return "???";
    }
}

/* Determine target quantization type for a tensor based on its name */
static enum ggml_type pick_quant(const char *name, enum ggml_type src_type,
                                  enum ggml_type attn_q, enum ggml_type ffn_q) {
    /* Never quantize: positional embeddings, biases, layer norms */
    if (strstr(name, "positional_embedding")) return src_type;
    if (strstr(name, ".bias"))                return GGML_TYPE_F32;
    if (strstr(name, "_ln.weight"))           return GGML_TYPE_F32;
    if (strstr(name, "ln_post"))              return GGML_TYPE_F32;

    /* Only quantize f16 tensors (f32 are already small) */
    if (src_type != GGML_TYPE_F16) return src_type;

    /* Token embedding: use attention quant (it's critical for decoder) */
    if (strstr(name, "token_embedding"))      return attn_q;

    /* Attention weights: Q, K, V, Out — preserve precision */
    if (strstr(name, "attn.query.weight"))     return attn_q;
    if (strstr(name, "attn.key.weight"))       return attn_q;
    if (strstr(name, "attn.value.weight"))     return attn_q;
    if (strstr(name, "attn.out.weight"))       return attn_q;
    if (strstr(name, "cross_attn.query"))      return attn_q;
    if (strstr(name, "cross_attn.key"))        return attn_q;
    if (strstr(name, "cross_attn.value"))      return attn_q;
    if (strstr(name, "cross_attn.out"))        return attn_q;

    /* FFN/MLP weights: aggressive compression */
    if (strstr(name, "mlp.0.weight"))          return ffn_q;
    if (strstr(name, "mlp.2.weight"))          return ffn_q;

    /* Encoder conv weights: use FFN quant (compute-heavy, less precision-sensitive) */
    if (strstr(name, "conv1.weight"))          return ffn_q;
    if (strstr(name, "conv2.weight"))          return ffn_q;

    /* Default: use FFN quant for any remaining f16 tensor */
    return ffn_q;
}

/* ── Dequantize f16 to f32 ──────────────────────────────────────── */

static void dequant_f16(const void *src, float *dst, int64_t n) {
    const uint16_t *s = (const uint16_t *)src;
    for (int64_t i = 0; i < n; i++) {
        dst[i] = ggml_fp16_to_fp32(s[i]);
    }
}

/* ── Main ──────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s input.bin output.bin [attn_quant] [ffn_quant]\n", argv[0]);
        fprintf(stderr, "  attn_quant: q4_0|q5_0|q5_1|q8_0 (default: q5_0)\n");
        fprintf(stderr, "  ffn_quant:  q4_0|q4_1|q5_0 (default: q4_0)\n");
        return 1;
    }

    const char *input_path  = argv[1];
    const char *output_path = argv[2];
    enum ggml_type attn_q = argc > 3 ? parse_quant_type(argv[3]) : GGML_TYPE_Q5_0;
    enum ggml_type ffn_q  = argc > 4 ? parse_quant_type(argv[4]) : GGML_TYPE_Q4_0;

    printf("Mixed-bit quantizer: attn=%s, ffn=%s\n", type_name(attn_q), type_name(ffn_q));

    /* Read input model */
    FILE *fin = fopen(input_path, "rb");
    if (!fin) { perror("fopen input"); return 1; }

    FILE *fout = fopen(output_path, "wb");
    if (!fout) { perror("fopen output"); fclose(fin); return 1; }

    /* Read and write magic */
    uint32_t magic;
    fread(&magic, 4, 1, fin);
    fwrite(&magic, 4, 1, fout);

    if (magic != 0x67676d6c) {
        fprintf(stderr, "Not a GGML whisper model (magic: 0x%08x)\n", magic);
        fclose(fin); fclose(fout); return 1;
    }

    /* Read hyperparameters */
    struct {
        int32_t n_vocab, n_audio_ctx, n_audio_state, n_audio_head, n_audio_layer;
        int32_t n_text_ctx, n_text_state, n_text_head, n_text_layer;
        int32_t n_mels, ftype;
    } hparams;
    fread(&hparams, sizeof(hparams), 1, fin);

    /* Write hparams with updated ftype for mixed quantization.
     * ftype encodes the quantization: ftype = base + version * 1000
     * We'll use a custom code: ftype = attn_type + ffn_type * 100  */
    int32_t new_ftype = (int32_t)attn_q + (int32_t)ffn_q * 1000;
    hparams.ftype = new_ftype;
    fwrite(&hparams, sizeof(hparams), 1, fout);

    printf("Model: vocab=%d, audio_state=%d, text_state=%d\n",
           hparams.n_vocab, hparams.n_audio_state, hparams.n_text_state);
    printf("Audio layers=%d, text layers=%d\n",
           hparams.n_audio_layer, hparams.n_text_layer);

    /* Copy mel filters */
    int32_t n_mel_filters, n_mel_fft;
    fread(&n_mel_filters, 4, 1, fin);
    fread(&n_mel_fft, 4, 1, fin);
    fwrite(&n_mel_filters, 4, 1, fout);
    fwrite(&n_mel_fft, 4, 1, fout);

    size_t mel_size = (size_t)n_mel_filters * n_mel_fft * 4;
    void *mel_data = malloc(mel_size);
    fread(mel_data, 1, mel_size, fin);
    fwrite(mel_data, 1, mel_size, fout);
    free(mel_data);

    /* Copy vocab: first read n_vocab_base, then that many string entries */
    int32_t n_vocab_base;
    fread(&n_vocab_base, 4, 1, fin);
    fwrite(&n_vocab_base, 4, 1, fout);
    printf("Vocab: %d base tokens (+ %d added)\n", n_vocab_base,
           hparams.n_vocab - n_vocab_base);

    for (int i = 0; i < n_vocab_base; i++) {
        uint32_t len;
        fread(&len, 4, 1, fin);
        fwrite(&len, 4, 1, fout);
        char buf[1024];
        if (len > sizeof(buf)) {
            fprintf(stderr, "Vocab entry too long: %u\n", len);
            fclose(fin); fclose(fout); return 1;
        }
        fread(buf, 1, len, fin);
        fwrite(buf, 1, len, fout);
    }

    /* Process tensors */
    size_t total_in = 0, total_out = 0;
    int n_tensors = 0;
    int n_quantized = 0;

    /* Allocate buffers for quantization work */
    size_t max_buf = 512 * 51864 * sizeof(float); /* largest tensor: token_embedding */
    float *f32_buf = (float *)malloc(max_buf);
    void  *quant_buf = malloc(max_buf); /* quantized output (always smaller) */
    if (!f32_buf || !quant_buf) {
        fprintf(stderr, "Failed to allocate work buffers (%zu bytes)\n", max_buf);
        fclose(fin); fclose(fout); return 1;
    }

    while (1) {
        long loop_start = ftell(fin);
        int32_t n_dims;
        if (fread(&n_dims, 4, 1, fin) != 1) break;
        if (n_dims < 1 || n_dims > 4) {
            fprintf(stderr, "Invalid n_dims=%d at file offset %ld, stopping\n", n_dims, loop_start);
            break;
        }

        int32_t name_len, src_type_i;
        fread(&name_len, 4, 1, fin);
        fread(&src_type_i, 4, 1, fin);

        int32_t dims[4] = {1, 1, 1, 1};
        for (int d = 0; d < n_dims; d++) {
            fread(&dims[d], 4, 1, fin);
        }

        char name[256] = {0};
        fread(name, 1, name_len, fin);

        /* Data immediately follows the name (no alignment in whisper GGML format) */
        long data_start = ftell(fin);

        /* Calculate source size */
        int64_t n_elements = (int64_t)dims[0] * dims[1] * dims[2] * dims[3];
        enum ggml_type src_type = (enum ggml_type)src_type_i;
        size_t src_bytes;
        if (src_type == GGML_TYPE_F32) {
            src_bytes = n_elements * 4;
        } else if (src_type == GGML_TYPE_F16) {
            src_bytes = n_elements * 2;
        } else {
            /* Already quantized — should not happen for fp16 input */
            size_t blk = ggml_blck_size(src_type);
            size_t type_sz = ggml_type_size(src_type);
            src_bytes = (n_elements / blk) * type_sz;
        }

        /* Read source data */
        void *src_data = malloc(src_bytes);
        fread(src_data, 1, src_bytes, fin);
        total_in += src_bytes;

        /* Determine target type */
        enum ggml_type dst_type = pick_quant(name, src_type, attn_q, ffn_q);

        /* Quantize if needed */
        void *dst_data = src_data;
        size_t dst_bytes = src_bytes;
        int32_t dst_type_i = src_type_i;

        if (dst_type != src_type && src_type == GGML_TYPE_F16) {
            /* Dequantize f16 → f32 */
            dequant_f16(src_data, f32_buf, n_elements);

            /* Quantize f32 → target */
            int64_t nrows = (n_dims > 1) ? dims[1] : 1;
            if (n_dims > 2) nrows *= dims[2];
            int64_t n_per_row = dims[0];

            size_t qsize = ggml_quantize_chunk(dst_type, f32_buf, quant_buf,
                                                0, nrows, n_per_row, NULL);

            dst_data = quant_buf;
            dst_bytes = qsize;
            dst_type_i = (int32_t)dst_type;
            n_quantized++;
        } else if (dst_type == GGML_TYPE_F32 && src_type == GGML_TYPE_F16) {
            /* Promote f16 → f32 for precision-critical tensors */
            dequant_f16(src_data, f32_buf, n_elements);
            dst_data = f32_buf;
            dst_bytes = n_elements * 4;
            dst_type_i = (int32_t)GGML_TYPE_F32;
        }

        /* Write tensor header */
        fwrite(&n_dims, 4, 1, fout);
        fwrite(&name_len, 4, 1, fout);
        fwrite(&dst_type_i, 4, 1, fout);
        for (int d = 0; d < n_dims; d++) {
            fwrite(&dims[d], 4, 1, fout);
        }
        fwrite(name, 1, name_len, fout);

        /* Write tensor data immediately after name (no alignment) */

        /* Write tensor data */
        fwrite(dst_data, 1, dst_bytes, fout);
        total_out += dst_bytes;

        float src_mb = src_bytes / (1024.0f * 1024.0f);
        float dst_mb = dst_bytes / (1024.0f * 1024.0f);
        printf("  %50s - [%5d,%5d,%5d] %s -> %s  %6.2fMB -> %6.2fMB  (hdr@%ld data@%ld-%ld)\n",
               name, dims[0], dims[1], dims[2],
               type_name(src_type), type_name(dst_type),
               src_mb, dst_mb, loop_start, data_start, data_start + (long)src_bytes);
        fflush(stdout);

        free(src_data);
        n_tensors++;
    }

    fclose(fin);
    fclose(fout);

    ggml_quantize_free();
    free(f32_buf);
    free(quant_buf);

    printf("\n%d tensors processed, %d quantized\n", n_tensors, n_quantized);
    printf("Size: %.1f MB -> %.1f MB (%.1f%% reduction)\n",
           total_in / (1024.0 * 1024.0), total_out / (1024.0 * 1024.0),
           100.0 * (1.0 - (double)total_out / total_in));

    return 0;
}
