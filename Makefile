# hcp-whisper
CC      = cc
CFLAGS  = -O2 -Wall -Wextra -std=c11
TARGET  = hcp-whisper
SRC     = src/main.c

# Homebrew paths (macOS)
WHISPER_INC = $(shell brew --prefix 2>/dev/null)/include
WHISPER_LIB = $(shell brew --prefix 2>/dev/null)/lib

# Fallback
ifeq ($(WHISPER_INC),/include)
  WHISPER_INC = /usr/local/include
  WHISPER_LIB = /usr/local/lib
endif

INCLUDES = -I$(WHISPER_INC) -Isrc
LDFLAGS  = -L$(WHISPER_LIB) -lwhisper -lggml -lz -lm

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(SRC) src/hcp.h src/hcp_subword_freq.h src/hcp_bigram.h
	$(CC) $(CFLAGS) $(INCLUDES) -o $(TARGET) $(SRC) $(LDFLAGS)

# Run test on sample audio (requires WAV file)
test: $(TARGET)
	@echo "=== HCP-Whisper Test Suite ==="
	$(CC) $(CFLAGS) $(INCLUDES) -o hcp-test tests/test_hcp.c $(LDFLAGS)
	./hcp-test

# Quick smoke test with a real audio file
smoke: $(TARGET)
	@if [ -z "$(AUDIO)" ]; then echo "Usage: make smoke AUDIO=path/to/file.wav"; exit 1; fi
	./$(TARGET) $(AUDIO) results/smoke --all
	@echo "=== Results ==="
	@cat results/smoke/transcript.json | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Segments: {d[\"total_segments\"]}'); print(f'Quality: {d[\"hcp\"][\"quality_base_avg\"]:.4f} → {d[\"hcp\"][\"quality_hcp_avg\"]:.4f} (+{d[\"hcp\"][\"quality_uplift_pct\"]:.1f}%)'); print(f'HCP: {d[\"hcp\"][\"elapsed_ms\"]:.1f}ms, {d[\"hcp\"][\"flagged_tokens\"]}/{d[\"hcp\"][\"tokens\"]} flagged')"

clean:
	rm -f $(TARGET) hcp-test
	rm -rf results/
