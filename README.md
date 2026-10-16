# numkit/comm

[![License: 0BSD](https://img.shields.io/badge/License-0BSD-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B17)

The **Communications Toolbox** for **[NumKit](https://github.com/numkit/numkit)**.

High-performance digital and analog communications algorithms with MATLAB R2025b parity.

---

## Features

- **Digital Modulation**: PSK (`pskmod`, `pskdemod`), DPSK (`dpskmod`, `dpskdemod`), PAM (`pammod`, `pamdemod`), QAM (`qammod`, `qamdemod`, `genqammod`, `genqamdemod`, `mil188qammod`, `mil188qamdemod`), APSK (`apskmod`, `apskdemod`), FSK (`fskmod`, `fskdemod`), MSK (`mskmod`, `mskdemod`), OFDM (`ofdmmod`, `ofdmdemod`), constellation normalization (`modnorm`).
- **Analog Modulation**: AM (`ammod`, `amdemod`), FM (`fmmod`, `fmdemod`), PM (`pmmod`, `pmdemod`), SSB (`ssbmod`, `ssbdemod`).
- **Channel Modeling & RF Metrics**: AWGN (`awgn`, `wgn`), binary symmetric channel (`bsc`), Rayleigh & Rician fading (`rayleighchan`, `ricianchan`), Q-function & Marcum Q (`qfunc`, `qfuncinv`, `marcumq`), theoretical BER (`berawgn`, `berconfint`), SNR conversion (`convertSNR`, `noisebw`).
- **Pulse Shaping, Equalization & Link Utilities**: Raised cosine & Gaussian filters (`rcosdesign`, `gaussdesign`), pulse shaping (`rectpulse`, `intdump`), scrambling (`scrambler`, `descrambler`), error counting (`biterr`, `symerr`), companding (`compand`).
- **Source Coding & Quantization**: Huffman (`huffmandict`, `huffmanenco`, `huffmandeco`), arithmetic coding (`arithenco`, `arithdeco`), DPCM (`dpcmenco`, `dpcmdeco`, `dpcmopt`), scalar quantization (`quantiz`, `lloyds`), base conversions (`bit2int`, `int2bit`, `bi2de`, `de2bi`, `vec2mat`), random sources (`randsrc`, `randerr`).
- **Error-Control Coding**: Convolutional coding & Viterbi decoding (`poly2trellis`, `istrellis`, `convenc`, `vitdec`), linear & cyclic block codes (`hammgen`, `cyclpoly`, `cyclgen`, `gen2par`, `syndtable`, `encode`, `decode`).

---

## Building as a Dynamic Module

```bash
cmake -B build -S .
cmake --build build --config Release
```

Produces `numkit_comm.dll` (Windows) or `libnumkit_comm.so` (Linux/macOS).

---

## Usage in NumKit

Install via the NumKit package manager:

```bash
nk pkg install comm
```

Or load dynamically in a script:

```matlab
% Loaded automatically on first call to any comm function:
y = qammod([0 1 2 3], 4);
disp(qamdemod(y, 4));
```

---

## License

This software is released under the **[BSD Zero Clause License (0BSD)](LICENSE)**.
