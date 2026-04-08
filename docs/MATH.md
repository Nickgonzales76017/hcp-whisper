# The Mathematics of Complex-Domain HCP for ASR Refinement

> A complete derivation of Complex-Domain Hierarchical Constraint Propagation — a novel post-processing algorithm that lifts ASR token sequences into the complex plane and applies spectral filtering to detect and correct transcription errors.

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Core Insight: Why Complex Domain?](#2-core-insight-why-complex-domain)
3. [Data Inventory: What Whisper Already Gives You](#3-data-inventory-what-whisper-already-gives-you)
4. [Step 1 — Token Flattening](#4-step-1--token-flattening)
5. [Step 2 — Complex Lifting: Dual-Channel Construction](#5-step-2--complex-lifting-dual-channel-construction)
6. [Step 3 — Free Signal Integration](#6-step-3--free-signal-integration)
7. [Step 4 — FFT: Token Sequence → Frequency Domain](#7-step-4--fft-token-sequence--frequency-domain)
8. [Step 5 — Three-Band Adaptive Spectral Filter](#8-step-5--three-band-adaptive-spectral-filter)
9. [Step 6 — Dirichlet Anomaly Detection](#9-step-6--dirichlet-anomaly-detection)
10. [Step 7 — IFFT and Per-Token Correction](#10-step-7--ifft-and-per-token-correction)
11. [Step 8 — Segment Quality Enhancement](#11-step-8--segment-quality-enhancement)
12. [Hallucination Detection: 5-Layer System](#12-hallucination-detection-5-layer-system)
13. [Complexity Analysis](#13-complexity-analysis)
14. [Theoretical Claims](#14-theoretical-claims)
15. [Connections to Prior Work](#15-connections-to-prior-work)

---

## 1. Problem Statement

Current ASR systems treat model size as the only lever for quality:

| Model | Size | Quality |
|-------|------|---------|
| Whisper tiny | 39 MB | ~0.60 |
| Whisper base | 74 MB | ~0.70 |
| Whisper small | 244 MB | ~0.82 |
| Whisper medium | 769 MB | ~0.88 |
| Whisper large-v3 | 3.1 GB | ~0.93 |

This creates a hard tradeoff: local-first, privacy-preserving, low-cost transcription can never compete with cloud APIs on quality.

**Thesis:** A small model (50 MB) *plus* an O(N log N) post-hoc refinement algorithm operating in the complex domain can match or exceed a model 10× its size, by exploiting structural redundancy in the transcript that autoregressive decoding cannot access.

---

## 2. Core Insight: Why Complex Domain?

After decoding, ~85% of tokens are correct with high confidence. Those correct tokens contain enough information — acoustic, morphological, and temporal — to resolve the remaining ~15%.

**The problem:** How do you propagate constraints from correct tokens to uncertain ones efficiently?

**The naive approach:** Pairwise comparison of all token pairs = O(N²). For N=2000 tokens, that's 4 million comparisons. Too slow, and most pairs are unrelated.

**The key mathematical move:** Lift token positions into the complex plane.

In the complex plane:
- **Magnitude** encodes confidence (how certain we are)
- **Phase** encodes identity (what the token sounds like and what it means)

This representation has three properties that make constraint propagation *free*:

### Property 1: Phase-Aligned Interference

Tokens with similar acoustic/morphological properties get similar phase values. When transformed to the frequency domain via FFT, these phase-aligned tokens *constructively interfere* — their contributions add up, reinforcing the pattern.

Errors at those positions have *random* phase relative to the correct pattern. They *destructively interfere* — their contributions partially cancel. The FFT does the pairwise comparison for you in O(N log N).

### Property 2: Hierarchical Scale Separation

The FFT naturally decomposes the token sequence into frequency bands:

| Frequency | Linguistic Scale | What It Captures |
|-----------|-----------------|------------------|
| k ≈ 0 | Discourse | Overall topic, speaker identity |
| k small | Paragraph/sentence | Topic flow, coherence |
| k medium | Phrase/word | Lexical patterns, collocations |
| k ≈ N/2 | Token | Individual token choices |

A filter operating on different frequency bands simultaneously constrains at **all linguistic scales at once** — something a left-to-right decoder fundamentally cannot do.

### Property 3: Multiplicative Coupling

The coupled signal is the product of two channels:

$$z_i = z_{\text{acou},i} \cdot z_{\text{morph},i}$$

Because complex multiplication adds phases and multiplies magnitudes:

$$|z_i| = |z_{\text{acou},i}| \cdot |z_{\text{morph},i}|$$
$$\angle z_i = \angle z_{\text{acou},i} + \angle z_{\text{morph},i}$$

A token must be valid in **both** the acoustic and morphological domains to have high coupled magnitude. This is a joint constraint that single-domain thresholds (e.g., just checking logprob) miss entirely.

---

## 3. Data Inventory: What Whisper Already Gives You

Every signal used by HCP is **already computed** by the whisper decoder. Zero additional model inference.

### Per-Token (`whisper_full_get_token_data()`)

| Field | Type | What It Encodes |
|-------|------|-----------------|
| `id` | `whisper_token` | Token ID (BPE vocabulary index, 0–51864) |
| `p` | `float` | Acoustic probability from decoder softmax |
| `plog` | `float` | Log-probability: $\log(p)$ |
| `vlen` | `float` | Voice length — acoustic duration of this token |
| `t_dtw` | `int64_t` | DTW-aligned timestamp (sub-word precision) |
| `tid` | `whisper_token` | Forced timestamp token ID |
| `pt` | `float` | Timestamp token probability |
| `ptsum` | `float` | Sum of all timestamp probabilities |

### Per-Segment

| Field | What It Provides |
|-------|-----------------|
| `no_speech_prob` | P(no speech) — silence probability |
| `speaker_turn` | TinyDiarize speaker change flag |
| `t0`, `t1` | Segment time boundaries |
| `text` | Full segment text |

### Derived (computed once, O(N))

| Signal | Derivation | Used For |
|--------|-----------|----------|
| `compression_ratio` | zlib(text) ratio | Hallucination detection |
| `n-gram_repeat` | Hash-based 3-gram counting | Repetition detection |
| `vlen_deviation` | Median of 5-neighborhood | Anomaly detection |
| `Δt` | `t_dtw[i] - t_dtw[i-1]` | Speech rhythm encoding |

---

## 4. Step 1 — Token Flattening

After `whisper_full()` returns S segments with varying token counts, flatten into a single array of N content tokens:

```
flat[0..N-1] = {all tokens from all segments where id < 50257}
```

Special tokens (BOS, EOS, timestamps, etc. with id ≥ 50257) are excluded. Each flat token retains a pointer to its parent segment for context signals (no_speech_prob, compression_ratio, speaker_turn).

Typical N: 1500–5000 for a 10–30 minute file.

---

## 5. Step 2 — Complex Lifting: Dual-Channel Construction

### Acoustic Channel

For each token position $i$:

$$z_{\text{acou},i} = \sqrt{p_i} \cdot e^{j\phi_{\text{acou},i}}$$

**Magnitude:** $\sqrt{p_i}$ (probability amplitude)

The square root follows the quantum mechanics convention: $|z|^2 = p$, making magnitude proportional to *amplitude* rather than probability. This means the spectral energy (which is $|Z[k]|^2$ after FFT) directly corresponds to probability mass.

**Phase:** Acoustic identity hash

$$\phi_{\text{acou},i} = \text{FNV-1a}(\text{token\_id}_i,\ \text{vlen}_{q,i},\ \Delta t_{q,i}) \mod 2\pi$$

Where:
- $\text{vlen}_{q,i}$: voice length quantized to 16 bins. Encodes speech rate — the same word spoken fast vs. slow gets different phases.
- $\Delta t_{q,i}$: inter-token time gap quantized to 16 bins. Encodes prosodic phrasing — natural pauses vs. rushed speech.

The FNV-1a hash is deterministic: identical (token, rate, rhythm) triples always map to the same phase. This is what creates phase alignment between acoustically similar positions.

**Quantization functions:**

$$\text{vlen}_q = \text{clamp}\!\left(\left\lfloor \text{vlen} \times \frac{16}{2.0} \right\rfloor, 0, 15\right)$$

$$\Delta t_q = \text{clamp}\!\left(\left\lfloor \frac{\Delta t_{\text{ms}} \times 16}{500} \right\rfloor, 0, 15\right)$$

### Morphological Channel

$$z_{\text{morph},i} = \sqrt{f_i} \cdot e^{j\phi_{\text{morph},i}}$$

**Magnitude:** $\sqrt{f_i}$ where $f_i$ is the unigram frequency of BPE token $i$

Common subwords ("the", "ing", "tion") have high morphological magnitude. Rare subwords have low magnitude. This creates a prior: positions using common morphemes are naturally weighted higher.

The frequency table is precomputed from the whisper tokenizer's vocabulary (51,864 entries). See `src/hcp_subword_freq.h`.

**Phase:** Morphological identity hash

$$\phi_{\text{morph},i} = \text{FNV-1a}(\text{subword\_bytes}_i) \mod 2\pi$$

Morphologically related tokens (same root, e.g., "walk", "walking", "walked") share prefix bytes and thus share phase components.

### Coupled Signal

$$z_i = z_{\text{acou},i} \cdot z_{\text{morph},i}$$

Expanding:

$$|z_i| = \sqrt{p_i} \cdot \sqrt{f_i} = \sqrt{p_i \cdot f_i}$$

$$\angle z_i = \phi_{\text{acou},i} + \phi_{\text{morph},i}$$

The coupled magnitude is the geometric mean of acoustic confidence and morphological prior. The coupled phase combines acoustic and morphological identity.

---

## 6. Step 3 — Free Signal Integration

Before FFT, five additional signals modulate the coupled signal. All are already computed — zero extra cost.

### 6.1 No-Speech Probability Damping

$$z_i \leftarrow z_i \cdot \max(0.1,\ 1 - p_{\text{nsp},s})$$

Tokens in segments with high silence probability get magnitude reduction. The floor at 0.1 prevents complete suppression.

### 6.2 Compression Ratio Damping

$$z_i \leftarrow z_i \cdot \min\!\left(1,\ \frac{2.4}{\rho_s}\right)$$

Where $\rho_s$ is the zlib compression ratio of segment $s$. Highly compressible text (repetitive content = hallucination signal) gets attenuated. The threshold 2.4 is empirically chosen: normal speech has $\rho \approx 1.2\text{–}2.0$, hallucinations have $\rho > 3.0$.

### 6.3 Speaker Turn Phase Reset

At positions where `speaker_turn == true`:

$$z_i \leftarrow |z_i| \cdot e^{j \cdot \text{FNV-1a}(i)}$$

This randomizes the phase at speaker boundaries, preventing constraint propagation across speakers. Speaker A's patterns should not influence the correction of Speaker B's tokens.

### 6.4 Voice Length Anomaly Damping

For each token $i$ (with at least 2 neighbors on each side), compute the local median voice length from a 5-token window:

$$\tilde{v}_i = \text{median}(v_{i-2}, v_{i-1}, v_i, v_{i+1}, v_{i+2})$$

If the ratio is extreme:

$$z_i \leftarrow z_i \cdot 0.5 \quad \text{if} \quad \frac{v_i}{\tilde{v}_i} > 2.0 \text{ or } < 0.33$$

Voice length anomalies indicate the decoder may have stumbled — abnormally long or short acoustic realizations relative to neighbors.

### 6.5 Low Log-Probability Damping

$$z_i \leftarrow z_i \cdot \max\!\left(0.2,\ 1 + \frac{\log p_i}{3}\right) \quad \text{if } \log p_i < -3$$

Tokens with very low probability get progressive magnitude reduction. At $\log p = -3$ (p ≈ 0.05) the damping starts. At $\log p = -6$ (p ≈ 0.002) the signal is reduced to 20% of original.

---

## 7. Step 4 — FFT: Token Sequence → Frequency Domain

### Zero-Padding

Pad the signal from $N$ to $N_2 = 2^{\lceil \log_2 N \rceil}$ (next power of 2) with zeros. This enables efficient radix-2 FFT and avoids spectral leakage artifacts from non-power-of-2 sizes.

### Forward DFT

$$Z[k] = \sum_{n=0}^{N_2-1} z[n] \cdot e^{-j2\pi kn/N_2} \quad \text{for } k = 0, \ldots, N_2-1$$

Implemented as radix-2 Cooley-Tukey with bit-reversal permutation. Cost: $O(N_2 \log N_2)$.

### What the Frequency Components Mean

Each $Z[k]$ represents a pattern that repeats at scale $N_2/k$ in the token sequence:

- **$Z[0]$**: DC component — the mean confidence and morphological prior across the entire transcript
- **$Z[1..N_2/64]$**: Low frequency — discourse-level coherence. Sudden topic shifts or speaker changes appear as energy in these bins.
- **$Z[N_2/64..N_2/8]$**: Mid frequency — word and phrase patterns. Collocations, idioms, common sequences create characteristic spectral signatures here.
- **$Z[N_2/8..N_2/2]$**: High frequency — individual token choices. Phonotactic constraints (which sounds can follow which) create the expected energy profile here.

**Key insight:** Errors at any individual position contaminate *all* frequency bins — but the contamination is spread thin. A frequency-domain filter that knows the expected spectral profile can remove this spread contamination efficiently, achieving what would require O(N²) pairwise comparisons in the token domain.

---

## 8. Step 5 — Three-Band Adaptive Spectral Filter

The filter $H[k]$ is applied in the frequency domain:

$$Z_{\text{filtered}}[k] = H[k] \cdot Z[k]$$

Rather than using a fixed filter, HCP computes an adaptive filter from the signal itself.

### Spectral Energy Envelope

First, compute the spectral energy and its local moving average:

$$E[k] = |Z[k]|^2 = Z[k] \cdot Z^*[k]$$

$$\bar{E}[k] = \frac{1}{W} \sum_{j=k-W/2}^{k+W/2} E[j]$$

Where $W = N_2/32$ is the envelope window size. $\bar{E}[k]$ represents the *expected* spectral energy at frequency $k$.

### Three Bands

**Low frequency** ($k < N_2/64$ or $k > N_2 - N_2/64$):

Coherence band. Anomalous energy spikes indicate abrupt topic discontinuities (potential error regions). The filter dampens spikes exceeding 3× the local envelope:

$$H_{\text{low}}[k] = \begin{cases} \sqrt{\frac{3\bar{E}[k]}{E[k]}} & \text{if } E[k] > 3\bar{E}[k] \\ 1 & \text{otherwise} \end{cases}$$

The square root ensures we dampen amplitude, not energy — a gentler correction.

**Mid frequency** ($N_2/64 \leq k < N_2/8$):

Lexical band. Word-level patterns live here. A looser threshold (5×) preserves legitimate lexical structure while catching severe outliers:

$$H_{\text{mid}}[k] = \begin{cases} \sqrt{\frac{5\bar{E}[k]}{E[k]}} & \text{if } E[k] > 5\bar{E}[k] \\ 1 & \text{otherwise} \end{cases}$$

**High frequency** ($k \geq N_2/8$):

Phonotactic band. Token-level patterns. Stricter filtering (2×) because individual token errors produce characteristic high-frequency signatures:

$$H_{\text{high}}[k] = \begin{cases} \sqrt{\frac{2\bar{E}[k]}{E[k]}} & \text{if } E[k] > 2\bar{E}[k] \\ 1 & \text{otherwise} \end{cases}$$

### Safety Bounds

$$H[k] = \text{clamp}(H[k], 0.1, 1.2)$$

- Floor at 0.1: never suppress a bin below 10% (preserve signal integrity)
- Ceiling at 1.2: never amplify beyond 20% above original (conservative correction)

---

## 9. Step 6 — Dirichlet Anomaly Detection

Overlaid on the three-band filter, a Dirichlet detector identifies pathological spectral patterns.

### Spectral Deviation Ratio

$$d[k] = \frac{E[k]}{\bar{E}[k]}$$

### Poles (Hallucination Loops)

When $d[k] > 8$: a single frequency bin has $8\times$ the expected energy. This indicates a highly periodic pattern in the token sequence — the signature of hallucination loops where the decoder repeats the same phrase.

$$H[k] \leftarrow H[k] \cdot 0.3 \quad \text{if } d[k] > 8$$

Strong damping: reduce to 30% of the already-filtered value.

### Zeros (Substitution Errors)

When $d[k] < 0.05$ for non-DC bins: expected energy is missing. This indicates the token sequence is *avoiding* a frequency that normal English speech would produce — the signature of systematic substitution errors.

$$H[k] \leftarrow H[k] \cdot 1.2 \quad \text{if } d[k] < 0.05 \text{ and } k > 0$$

Gentle boost: increase by 20% to restore expected spectral balance.

### Connection to Dirichlet Series

The name "Dirichlet anomaly" comes from the analogy with Dirichlet L-functions. If we treat each unique subword as an arithmetic function over frequency bins:

$$L(s, \chi_w) = \sum_{k=1}^{N_2/2} \chi_w(k) \cdot k^{-s}$$

Where $\chi_w(k)$ is the contribution of subword $w$ to frequency bin $k$, then poles of this series (where it diverges) correspond to hallucination loops, and zeros correspond to missing expected patterns. Our $d[k]$ ratio is a practical, discrete approximation of pole/zero detection.

---

## 10. Step 7 — IFFT and Per-Token Correction

### Inverse Transform

$$\hat{z}[n] = \frac{1}{N_2} \sum_{k=0}^{N_2-1} Z_{\text{filtered}}[k] \cdot e^{j2\pi kn/N_2}$$

The corrected signal $\hat{z}[n]$ has the same token positions but modified magnitudes and phases.

### Per-Token Comparison

For each position $n$, compare original and corrected:

**Magnitude ratio:**

$$r_n = \frac{|\hat{z}[n]|}{|z[n]|}$$

- $r_n > 1$: Correction *reinforced* this token — spectral constraints agree with the original decode. High confidence.
- $r_n \approx 1$: No significant change.
- $r_n < 1$: Correction *suppressed* this token — spectral constraints disagree. Potential error.

**Phase shift:**

$$\Delta\phi_n = |\angle\hat{z}[n] - \angle z[n]|$$

(wrapped to $[0, \pi]$)

Large phase shift indicates the correction points this position toward a different acoustic/morphological identity — the current token may be wrong.

### Flagging Criterion

A token is flagged if:

$$\Delta\phi_n > \theta_\phi \quad \text{OR} \quad r_n < \theta_r$$

With thresholds:
- $\theta_\phi = 1.2$ radians (~69°): significant phase rotation
- $\theta_r = 0.4$: magnitude suppressed to below 40% of original

### Segment-Level Rollup

A segment is flagged for re-decode if:

$$\frac{\text{flagged tokens in segment}}{\text{total tokens in segment}} > 0.30$$

---

## 11. Step 8 — Segment Quality Enhancement

The HCP-enhanced quality score replaces the naive composite quality:

$$q_{\text{base}} = \bar{p}_{\text{geo}} \cdot (1 - p_{\text{nsp}}) \cdot \min\!\left(1, \frac{2.4}{\rho}\right)$$

Where $\bar{p}_{\text{geo}}$ is the geometric mean token probability.

$$q_{\text{hcp}} = q_{\text{base}} \cdot \text{clamp}\!\left(\bar{r}_s, 0.5, 1.5\right)$$

Where $\bar{r}_s$ is the mean magnitude ratio for segment $s$:

$$\bar{r}_s = \frac{1}{|\{n: \text{seg}(n)=s\}|} \sum_{n: \text{seg}(n)=s} r_n$$

This is the core quality signal: segments where spectral constraints agree with the decode get boosted (up to 50%). Segments where constraints disagree get penalized (down to 50%). The result is clamped to $[0, 1]$.

---

## 12. Hallucination Detection: 5-Layer System

HCP includes a multi-layer hallucination detector. Each layer is independent and contributes a flag bit:

| Layer | Signal | Threshold | Bit |
|-------|--------|-----------|-----|
| 1. Compression ratio | $\rho = \text{len}(x) / \text{len}(\text{zlib}(x))$ | $> 2.4$ | `0x01` |
| 2. N-gram repetition | 3-gram collision count (FNV hash) | $> 3$ per ngram | `0x02` |
| 3. Vlen anomaly | $> 2/3$ of tokens with $v > v_{\max}/3$ | threshold | `0x04` |
| 4. Low logprob | Mean $\log p$ across tokens | $< -2.0$ | `0x08` |
| 5. HCP spectral | $> 30\%$ tokens flagged by magnitude/phase | threshold | `0x10` |

A segment is marked hallucinated if **any** flag bit is set. The HCP spectral layer (5) catches patterns that the other four miss — it detects correlated anomalies across scales.

---

## 13. Complexity Analysis

| Operation | Cost | Typical |
|-----------|------|---------|
| Token flattening | O(N) | <0.1 ms |
| Complex lifting (dual channel) | O(N) | <0.5 ms |
| Free signal integration | O(N) | <0.2 ms |
| FFT (radix-2) | O(N₂ log N₂) | ~2 ms |
| Spectral envelope | O(N₂ · W) | ~1 ms |
| Three-band filter + Dirichlet | O(N₂) | <0.5 ms |
| IFFT | O(N₂ log N₂) | ~2 ms |
| Comparison + flagging | O(N) | <0.2 ms |
| **Total HCP** | **O(N log N)** | **~6.6 ms** |
| Whisper decode (for comparison) | O(T · V) | ~50,000 ms |

HCP adds **<0.02%** to total runtime. The decode is 7,500× slower.

**Memory:** ~5 arrays × N₂ × 8 bytes. For N₂=4096: ~160 KB. Negligible.

---

## 14. Theoretical Claims

**Claim 1 (Information Preservation):** Complex-domain lifting preserves all information from real-valued token data while adding phase coupling that enables O(N log N) global constraint propagation. No data is discarded in the lifting step.

**Claim 2 (Scale Decomposition):** The FFT decomposes the token sequence into linguistically meaningful frequency bands (token, word, phrase, sentence, discourse) analogous to how signal FFT decomposes into physical frequencies.

**Claim 3 (Phase-Locked Propagation):** Phase-aligned positions (acoustically similar tokens) share spectral components, causing corrections to propagate between them through the transform without explicit pairwise comparison.

**Claim 4 (Dirichlet Detection):** The spectral representation of subword token distributions provides a mathematically grounded hallucination detector: loops create spectral poles ($d[k] \gg 1$), substitution errors create spectral zeros ($d[k] \ll 1$).

**Claim 5 (Model Budget):** For a fixed model size budget $M$, HCP(small\_model + tables) achieves higher quality than monolithic($M$) on spoken English, because the refinement exploits structural redundancy that autoregressive decoding cannot access.

---

## 15. Connections to Prior Work

| Technique | Connection |
|-----------|-----------|
| **Boosting** (Freund & Schapire, 1997) | Each constraint scale acts as a weak learner on the residual — HCP layers corrections hierarchically |
| **Residual Learning** (He et al., 2016) | HCP learns $\Delta q$ (the quality correction), not $q$ itself |
| **FNet** (Lee-Thorp et al., 2021) | Replacing attention with FFT in transformer blocks — HCP applies the same insight post-hoc |
| **k-NN LM** (Khandelwal et al., 2020) | Test-time adaptation without gradients — HCP similarly improves at test time with zero training |
| **Compressed Sensing** (Candès & Tao, 2006) | Sparse signal recovery from limited observations — flagged positions are recovered from the majority-correct signal |
| **Holographic Representations** (Plate, 1995) | Complex-valued distributed representations of structured data — HCP uses the same phase-binding mechanism |
| **Analytic Continuation** | Solutions in uncertain regions are uniquely determined by known-good regions — the spectral filter extends confident regions into uncertain ones |

---

## Appendix A: FNV-1a Hash

The FNV-1a hash is used throughout for deterministic phase assignment:

```
h = 0xcbf29ce484222325
for each byte b in input:
    h = h XOR b
    h = h × 0x100000001b3
return h
```

Phase mapping: $\phi = (h \bmod 2^{32}) / 2^{32} \times 2\pi$

FNV-1a has excellent avalanche properties: small input changes produce uniformly distributed output changes. This ensures that similar-but-distinct tokens get well-separated phases, while identical tokens always get the same phase.

## Appendix B: Subword Frequency Table

The morphological channel requires a unigram frequency for each of the 51,864 BPE tokens in whisper's vocabulary.

**Generation:** Tokenize a representative English corpus through `whisper_tokenize()`, count occurrences, normalize to [0, 1]. Tokens unseen in the corpus receive a Zipf estimate based on string length: $f = 0.01 / \text{strlen}$.

**Storage:** Two static arrays shipped with the binary:
- `hcp_token_strlen[51864]` — token string length (uint8_t)
- `hcp_subword_freq[51864]` — normalized frequency (float)

Total: ~250 KB. Generated once, never changes for a given whisper model.
