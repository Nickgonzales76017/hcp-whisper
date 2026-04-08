import struct, os

path = os.path.expanduser("~/.local/share/whisper/ggml-base.en.bin")
with open(path, "rb") as f:
    magic = struct.unpack("<I", f.read(4))[0]
    hparams = struct.unpack("<11i", f.read(44))
    n_vocab = hparams[0]
    print(f"Magic: {hex(magic)}, n_vocab: {n_vocab}")
    print(f"Hparams: {hparams}")

    n_mel_f = struct.unpack("<i", f.read(4))[0]
    n_mel_fft = struct.unpack("<i", f.read(4))[0]
    print(f"Mel: {n_mel_f} x {n_mel_fft}")
    mel_bytes = n_mel_f * n_mel_fft * 4
    f.read(mel_bytes)
    print(f"Position after mel: {f.tell()}")

    for i in range(5):
        word_len = struct.unpack("<I", f.read(4))[0]
        if word_len > 1000:
            print(f"  vocab[{i}]: len={word_len} SUSPICIOUS")
            f.seek(-4, 1)
            raw = f.read(32)
            print(f"  Raw bytes: {raw.hex()}")
            break
        word = f.read(word_len)
        print(f"  vocab[{i}]: len={word_len}, word={repr(word[:30])}")
