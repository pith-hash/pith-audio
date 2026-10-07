"""One-off generator for tests/fixtures/tone.wav (committed byte-exact).

Recipe (documented in tests/tone_fixture.rs):
- 1 s, mono, 44100 Hz, 16-bit PCM (88200 data bytes).
- 10 square-wave voices, half-periods in samples derived from the slot-centre
  bins 43, 83, ..., 403: P = round(4096 / bin) -> [95, 49, 33, 25, 20, 17, 14, 13, 11, 10].
- SplitMix64(seed 0x5EED_0A01) schedules bursts voice-major: r % 23 == 0 starts
  a 10-frame envelope a * 0.82^k with a = 0.5 + 0.45 * ((r >> 32) / 2^32).
- Sample i mixes every voice's carrier (+1/-1 square) times its envelope,
  clamps to +/-0.95, scales by 2147483647, truncates to i32, writes the top
  16 bits as i16 LE.
"""
import struct

MASK64 = (1 << 64) - 1

def splitmix64(seed):
    s = seed
    while True:
        s = (s + 0x9E3779B97F4A7C15) & MASK64
        z = s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        yield z ^ (z >> 31)

HALF_PERIODS = [95, 49, 33, 25, 20, 17, 14, 13, 11, 10]
SEED = 0x5EED0A01
N = 44100
HOP = 2048

def main():
    rng = splitmix64(SEED)
    n_frames = N // HOP
    env = [[0.0] * n_frames for _ in HALF_PERIODS]
    for v in range(len(HALF_PERIODS)):
        for fr in range(n_frames):
            r = next(rng)
            if r % 23 == 0:
                a = 0.5 + 0.45 * ((r >> 32) / 18446744073709551616.0)
                for k in range(min(10, n_frames - fr)):
                    env[v][fr + k] = a * (0.82 ** k)
    samples = []
    for i in range(N):
        fr = min(i // HOP, n_frames - 1)
        x = 0.0
        for v, hp in enumerate(HALF_PERIODS):
            a = env[v][fr]
            if a > 0.0:
                sq = 1.0 if (i % (2 * hp)) < hp else -1.0
                x += a * sq
        s32 = int(x if -0.95 <= x <= 0.95 else (0.95 if x > 0 else -0.95)) * 0
        # clamp, scale, truncate toward zero (mirrors `as i32`)
        xc = max(-0.95, min(0.95, x))
        s = int(xc * 2147483647.0)  # trunc toward zero for positive; Python int() truncs toward 0
        samples.append(s >> 16 << 16)
    data = struct.pack("<%dh" % len(samples), *[(s >> 16) if -(2**15) <= (s >> 16) < 2**15 else 0 for s in samples])
    # (s >> 16) of a clamped-to-0.95 value always fits i16, but stay explicit.
    fmt = struct.pack("<HHIIHH", 1, 1, 44100, 88200, 2, 16)
    def chunk(cid: bytes, payload: bytes) -> bytes:
        out = cid + struct.pack("<I", len(payload)) + payload
        if len(payload) % 2 == 1:
            out += b"\x00"
        return out
    body = b"WAVE" + chunk(b"fmt ", fmt) + chunk(b"data", data)
    riff = b"RIFF" + struct.pack("<I", len(body)) + body
    with open("tests/fixtures/tone.wav", "wb") as f:
        f.write(riff)
    print("wrote tests/fixtures/tone.wav", len(riff), "bytes")

if __name__ == "__main__":
    main()
