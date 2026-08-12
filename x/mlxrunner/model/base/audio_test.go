package base

import (
	"bytes"
	"encoding/binary"
	"math"
	"strings"
	"testing"
)

func wavBytes(format uint16, channels, rate, bits int, pcm []byte) []byte {
	var b bytes.Buffer
	b.WriteString("RIFF")
	binary.Write(&b, binary.LittleEndian, uint32(36+len(pcm)))
	b.WriteString("WAVE")
	b.WriteString("fmt ")
	binary.Write(&b, binary.LittleEndian, uint32(16))
	binary.Write(&b, binary.LittleEndian, format)
	binary.Write(&b, binary.LittleEndian, uint16(channels))
	binary.Write(&b, binary.LittleEndian, uint32(rate))
	binary.Write(&b, binary.LittleEndian, uint32(rate*channels*bits/8))
	binary.Write(&b, binary.LittleEndian, uint16(channels*bits/8))
	binary.Write(&b, binary.LittleEndian, uint16(bits))
	b.WriteString("data")
	binary.Write(&b, binary.LittleEndian, uint32(len(pcm)))
	b.Write(pcm)
	return b.Bytes()
}

func requireSamples(t *testing.T, data []byte, wantRate int, want []float32, tol float32) {
	t.Helper()
	samples, rate, err := DecodeAudio(data)
	if err != nil {
		t.Fatal(err)
	}
	if rate != wantRate {
		t.Fatalf("rate %d, want %d", rate, wantRate)
	}
	if len(samples) != len(want) {
		t.Fatalf("%d samples, want %d", len(samples), len(want))
	}
	for i := range want {
		if diff := samples[i] - want[i]; diff > tol || diff < -tol {
			t.Fatalf("sample %d: %v, want %v", i, samples[i], want[i])
		}
	}
}

func TestDecodeWAVPCM16(t *testing.T) {
	var pcm bytes.Buffer
	for _, v := range []int16{0, 16384, -16384, 32767, -32768} {
		binary.Write(&pcm, binary.LittleEndian, v)
	}
	want := []float32{0, 0.5, -0.5, 32767.0 / 32768, -1}
	requireSamples(t, wavBytes(1, 1, 16000, 16, pcm.Bytes()), 16000, want, 0)
}

func TestDecodeWAVPCM8(t *testing.T) {
	pcm := []byte{128, 255, 0, 192}
	want := []float32{0, 127.0 / 128, -1, 0.5}
	requireSamples(t, wavBytes(1, 1, 8000, 8, pcm), 8000, want, 0)
}

func TestDecodeWAVPCM24(t *testing.T) {
	var pcm []byte
	for _, v := range []int32{0, 1 << 22, -(1 << 22)} {
		pcm = append(pcm, byte(v), byte(v>>8), byte(v>>16))
	}
	want := []float32{0, 0.5, -0.5}
	requireSamples(t, wavBytes(1, 1, 16000, 24, pcm), 16000, want, 0)
}

func TestDecodeWAVPCM32(t *testing.T) {
	var pcm bytes.Buffer
	for _, v := range []int32{0, 1 << 30, -(1 << 30)} {
		binary.Write(&pcm, binary.LittleEndian, v)
	}
	want := []float32{0, 0.5, -0.5}
	requireSamples(t, wavBytes(1, 1, 16000, 32, pcm.Bytes()), 16000, want, 0)
}

func TestDecodeWAVFloat32(t *testing.T) {
	var pcm bytes.Buffer
	want := []float32{0, 0.25, -1, 1}
	for _, v := range want {
		binary.Write(&pcm, binary.LittleEndian, v)
	}
	requireSamples(t, wavBytes(3, 1, 44100, 32, pcm.Bytes()), 44100, want, 0)
}

func TestDecodeWAVStereoDownmix(t *testing.T) {
	var pcm bytes.Buffer
	for _, v := range []float32{1, 0, -0.5, 0.5} {
		binary.Write(&pcm, binary.LittleEndian, v)
	}
	want := []float32{0.5, 0}
	requireSamples(t, wavBytes(3, 2, 16000, 32, pcm.Bytes()), 16000, want, 0)
}

func TestDecodeWAVExtensible(t *testing.T) {
	var pcm bytes.Buffer
	binary.Write(&pcm, binary.LittleEndian, int16(16384))

	var b bytes.Buffer
	b.WriteString("RIFF")
	binary.Write(&b, binary.LittleEndian, uint32(60+pcm.Len()))
	b.WriteString("WAVE")
	b.WriteString("fmt ")
	binary.Write(&b, binary.LittleEndian, uint32(40))
	binary.Write(&b, binary.LittleEndian, uint16(0xfffe))
	binary.Write(&b, binary.LittleEndian, uint16(1))
	binary.Write(&b, binary.LittleEndian, uint32(16000))
	binary.Write(&b, binary.LittleEndian, uint32(32000))
	binary.Write(&b, binary.LittleEndian, uint16(2))
	binary.Write(&b, binary.LittleEndian, uint16(16))
	binary.Write(&b, binary.LittleEndian, uint16(22)) // extension size
	binary.Write(&b, binary.LittleEndian, uint16(16)) // valid bits
	binary.Write(&b, binary.LittleEndian, uint32(0))  // channel mask
	binary.Write(&b, binary.LittleEndian, uint16(1))  // subformat: PCM
	b.Write(make([]byte, 14))                         // rest of subformat GUID
	b.WriteString("data")
	binary.Write(&b, binary.LittleEndian, uint32(pcm.Len()))
	b.Write(pcm.Bytes())

	requireSamples(t, b.Bytes(), 16000, []float32{0.5}, 0)
}

func TestDecodeAudioErrors(t *testing.T) {
	adpcm := wavBytes(2, 1, 16000, 4, make([]byte, 8))
	noData := wavBytes(1, 1, 16000, 16, nil)
	noData = noData[:len(noData)-8]
	zeroChannels := wavBytes(1, 0, 16000, 16, make([]byte, 4))
	// A tiny file whose declared 1 Hz rate makes it run for over 10 minutes.
	tooLong := wavBytes(1, 1, 1, 8, make([]byte, maxAudioSeconds+1))

	cases := []struct {
		name string
		data []byte
		want string
	}{
		{"garbage", []byte("not audio at all"), "unrecognized audio format"},
		{"empty", nil, "unrecognized audio format"},
		{"mp3 id3", []byte("ID3\x04\x00rest"), "unrecognized audio format"},
		{"mp3 sync", []byte{0xff, 0xfb, 0x90, 0x00}, "unrecognized audio format"},
		{"truncated riff", []byte("RIFF\x00\x00\x00\x00WAVE"), "no fmt chunk"},
		{"no data chunk", noData, "no data chunk"},
		{"adpcm", adpcm, "unsupported format 2"},
		{"zero channels", zeroChannels, "invalid fmt"},
		{"longer than the duration cap", tooLong, "audio longer"},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			_, _, err := DecodeAudio(tt.data)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error %v, want %q", err, tt.want)
			}
		})
	}
}

func TestResample(t *testing.T) {
	tone := func(n int, freq, rate float64) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = float32(math.Sin(2 * math.Pi * freq * float64(i) / rate))
		}
		return out
	}
	// The kernel spans ~18 output samples at these ratios; stay well clear
	// of the edge truncation when comparing against the analytic tone.
	const margin = 200

	same := tone(100, 1000, 16000)
	if got := Resample(same, 16000, 16000); &got[0] != &same[0] {
		t.Fatal("same-rate resample should return the input")
	}

	// A band-limited tone resampled ideally is the same tone sampled at
	// the new rate.
	down := Resample(tone(44100, 1000, 44100), 44100, 16000)
	if len(down) != 16000 {
		t.Fatalf("%d samples, want 16000", len(down))
	}
	for i := margin; i < len(down)-margin; i++ {
		want := math.Sin(2 * math.Pi * 1000 * float64(i) / 16000)
		if diff := math.Abs(float64(down[i]) - want); diff > 5e-3 {
			t.Fatalf("downsample at %d: %v, want %v", i, down[i], want)
		}
	}

	up := Resample(tone(8000, 1000, 8000), 8000, 16000)
	if len(up) != 16000 {
		t.Fatalf("%d samples, want 16000", len(up))
	}
	for i := margin; i < len(up)-margin; i++ {
		want := math.Sin(2 * math.Pi * 1000 * float64(i) / 16000)
		if diff := math.Abs(float64(up[i]) - want); diff > 5e-3 {
			t.Fatalf("upsample at %d: %v, want %v", i, up[i], want)
		}
	}

	// Content above the target Nyquist must be filtered out, not aliased
	// into the band.
	alias := Resample(tone(44100, 10000, 44100), 44100, 16000)
	for i := margin; i < len(alias)-margin; i++ {
		if math.Abs(float64(alias[i])) > 0.01 {
			t.Fatalf("aliased content at %d: %v", i, alias[i])
		}
	}

	// Per-sample weight normalization keeps DC exact.
	dc := make([]float32, 1000)
	for i := range dc {
		dc[i] = 0.5
	}
	for i, v := range Resample(dc, 44100, 16000) {
		if math.Abs(float64(v)-0.5) > 1e-6 {
			t.Fatalf("dc at %d: %v", i, v)
		}
	}
}
