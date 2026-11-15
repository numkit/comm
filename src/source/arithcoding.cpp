// toolboxes/comm/src/source/arithcoding.cpp
//
// Arithmetic coding encoder + decoder.
//
//   code = arithenco(seq, counts)
//   seq  = arithdeco(code, counts, len)
//
// Algorithm: classic adaptive-precision arithmetic coding from
// Sayood, "Introduction to Data Compression" (Morgan Kaufmann),
// the same reference MATLAB R2025b's arithenco.m / arithdeco.m cite.
//
// Word length N = ceil(log2(sum(counts))) + 2. The lower / upper
// bounds are kept in N+1-bit unsigned integers; E1/E2/E3 rescaling
// emits / shifts bits as the bounds tighten.

#include <keel/core.h>
#include <keel/memory.h>
#include <numkit/comm/source/arithcoding.hpp>

#include <numkit/value/argcheck.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace numkit::comm {

namespace {

bool isRow(const Value &v)
{
    return v.dims().rows() == 1 && v.dims().cols() >= 1;
}

void readVecU(const Value &v, std::vector<keel::u64> &out)
{
    const keel::usize n = v.numel();
    out.resize(n);
    for (keel::usize i = 0; i < n; ++i) {
        const double d = v.elemAsDouble(i);
        if (!(d > 0.0) || std::floor(d) != d)
            throw Error("arithenco: counts must be positive integers",
                        0, 0, "arithenco", "", "numkit:arithenco:Counts");
        out[i] = static_cast<keel::u64>(d);
    }
}

// Bit at position p (1-based from LSB), matching MATLAB's bitget(x, p).
inline keel::u64 bitget(keel::u64 x, int p)
{
    return (x >> (p - 1)) & 1ULL;
}

// Mask x to N low bits (clear bits N+1 and above).
inline keel::u64 maskN(keel::u64 x, int N)
{
    return x & ((N == 64) ? ~0ULL : ((1ULL << N) - 1ULL));
}

void arithenco_argcheck(const Value &counts, std::vector<keel::u64> &cnt)
{
    readVecU(counts, cnt);
    if (cnt.empty())
        throw Error("arithenco: counts must be non-empty",
                    0, 0, "arithenco", "", "numkit:arithenco:Counts");
}

void arithdeco_argcheck(const Value &counts, std::vector<keel::u64> &cnt)
{
    readVecU(counts, cnt);
}

} // namespace

Value arithenco(const Value &seq, const Value &counts,
                keel::Allocator *mr)
{
    std::vector<keel::u64> cnt;
    arithenco_argcheck(counts, cnt);
    const keel::usize K = cnt.size();

    // Cumulative counts: cum[0]=0, cum[i]=sum(cnt[0..i-1])
    std::vector<keel::u64> cum(K + 1, 0);
    for (keel::usize i = 0; i < K; ++i) cum[i + 1] = cum[i] + cnt[i];
    const keel::u64 total = cum[K];

    const int N = static_cast<int>(std::ceil(std::log2(static_cast<double>(total)))) + 2;
    const keel::u64 topMask = 1ULL << (N - 1);

    keel::u64 dec_low = 0;
    keel::u64 dec_up  = (1ULL << N) - 1;
    keel::u64 E3_count = 0;

    std::vector<keel::u8> bits;
    bits.reserve(seq.numel() * (static_cast<keel::usize>(std::ceil(std::log2(K)) + 2)) + N);

    const keel::usize Nseq = seq.numel();
    for (keel::usize k = 0; k < Nseq; ++k) {
        const double sd = seq.elemAsDouble(k);
        if (!(sd > 0.0) || std::floor(sd) != sd
            || static_cast<keel::usize>(sd) > K)
            throw Error("arithenco: symbol out of range [1, length(counts)]",
                        0, 0, "arithenco", "", "numkit:arithenco:Sym");
        const keel::usize s = static_cast<keel::usize>(sd);

        const keel::u64 span = dec_up - dec_low + 1;
        const keel::u64 low_new = dec_low + (span * cum[s - 1]) / total;
        dec_up  = dec_low + (span * cum[s]) / total - 1;
        dec_low = low_new;

        // E1/E2/E3 rescaling.
        for (;;) {
            if (bitget(dec_low, N) == bitget(dec_up, N)) {
                const keel::u64 b = bitget(dec_low, N);
                bits.push_back(static_cast<keel::u8>(b));
                dec_low = (dec_low << 1) + 0;
                dec_up  = (dec_up  << 1) + 1;
                // Flush queued E3 bits (complement of b).
                if (E3_count > 0) {
                    for (keel::u64 i = 0; i < E3_count; ++i)
                        bits.push_back(static_cast<keel::u8>(b ^ 1ULL));
                    E3_count = 0;
                }
                dec_low = maskN(dec_low, N);
                dec_up  = maskN(dec_up,  N);
            } else if (bitget(dec_low, N - 1) == 1
                    && bitget(dec_up,  N - 1) == 0) {
                dec_low = (dec_low << 1) + 0;
                dec_up  = (dec_up  << 1) + 1;
                dec_low = maskN(dec_low, N);
                dec_up  = maskN(dec_up,  N);
                dec_low ^= topMask;
                dec_up  ^= topMask;
                ++E3_count;
            } else {
                break;
            }
        }
    }

    // Termination: emit final bits of dec_low (with E3 handling).
    // bin_low MSB-first = N bits of dec_low.
    std::vector<keel::u8> bin_low(N);
    for (int i = 0; i < N; ++i)
        bin_low[i] = static_cast<keel::u8>((dec_low >> (N - 1 - i)) & 1ULL);
    if (E3_count == 0) {
        for (int i = 0; i < N; ++i) bits.push_back(bin_low[i]);
    } else {
        const keel::u8 b = bin_low[0];
        bits.push_back(b);
        for (keel::u64 i = 0; i < E3_count; ++i)
            bits.push_back(static_cast<keel::u8>(b ^ 1));
        for (int i = 1; i < N; ++i) bits.push_back(bin_low[i]);
    }

    // Output preserves seq orientation.
    const bool row = isRow(seq);
    const keel::usize L = bits.size();
    Value out = Value::matrix(row ? 1 : L, row ? L : 1,
                              ValueType::DOUBLE, mr);
    double *o = out.doubleDataMut();
    for (keel::usize i = 0; i < L; ++i) o[i] = static_cast<double>(bits[i]);
    return out;
}

Value arithdeco(const Value &code, const Value &counts, keel::usize len,
                keel::Allocator *mr)
{
    std::vector<keel::u64> cnt;
    arithdeco_argcheck(counts, cnt);
    const keel::usize K = cnt.size();
    std::vector<keel::u64> cum(K + 1, 0);
    for (keel::usize i = 0; i < K; ++i) cum[i + 1] = cum[i] + cnt[i];
    const keel::u64 total = cum[K];

    const int N = static_cast<int>(std::ceil(std::log2(static_cast<double>(total)))) + 2;
    const keel::u64 topMask = 1ULL << (N - 1);

    // Read code bits.
    const keel::usize Lcode = code.numel();
    auto readBit = [&](keel::usize i) -> keel::u64 {
        const double d = code.elemAsDouble(i);
        if (d != 0.0 && d != 1.0)
            throw Error("arithdeco: code bits must be 0 or 1",
                        0, 0, "arithdeco", "", "numkit:arithdeco:Bit");
        return static_cast<keel::u64>(d);
    };
    if (Lcode < static_cast<keel::usize>(N))
        throw Error("arithdeco: code shorter than required word length",
                    0, 0, "arithdeco", "", "numkit:arithdeco:Short");

    keel::u64 dec_low = 0;
    keel::u64 dec_up  = (1ULL << N) - 1;
    keel::u64 dec_tag = 0;
    for (int i = 0; i < N; ++i)
        dec_tag = (dec_tag << 1) | readBit(static_cast<keel::usize>(i));

    // Find bin index `ptr` (1..K) such that cum[ptr-1] <= value < cum[ptr].
    auto pick = [&](keel::u64 value) -> keel::usize {
        if (value >= cum[K]) return K;
        // Linear scan from end (matches MATLAB's order; K is small).
        for (keel::usize i = K; i >= 1; --i) {
            if (cum[i - 1] <= value && value < cum[i]) return i;
        }
        return 1;
    };

    std::vector<double> dseq(len);
    keel::usize k = static_cast<keel::usize>(N);   // next code-bit index (0-based)
    for (keel::usize i = 0; i < len; ++i) {
        const keel::u64 span = dec_up - dec_low + 1;
        const keel::u64 dec_tag_new = ((dec_tag - dec_low + 1) * total - 1) / span;
        const keel::usize ptr = pick(dec_tag_new);
        dseq[i] = static_cast<double>(ptr);

        const keel::u64 low_new = dec_low + (span * cum[ptr - 1]) / total;
        dec_up  = dec_low + (span * cum[ptr]) / total - 1;
        dec_low = low_new;

        for (;;) {
            const bool e1e2 = (bitget(dec_low, N) == bitget(dec_up, N));
            const bool e3   = (bitget(dec_low, N - 1) == 1
                            && bitget(dec_up,  N - 1) == 0);
            if (!e1e2 && !e3) break;
            if (k >= Lcode) break;
            const keel::u64 cb = readBit(k);
            ++k;
            if (e1e2) {
                dec_low = (dec_low << 1);
                dec_up  = (dec_up  << 1) + 1;
                dec_tag = (dec_tag << 1) + cb;
                dec_low = maskN(dec_low, N);
                dec_up  = maskN(dec_up,  N);
                dec_tag = maskN(dec_tag, N);
            } else { // e3
                dec_low = (dec_low << 1);
                dec_up  = (dec_up  << 1) + 1;
                dec_tag = (dec_tag << 1) + cb;
                dec_low = maskN(dec_low, N);
                dec_up  = maskN(dec_up,  N);
                dec_tag = maskN(dec_tag, N);
                dec_low ^= topMask;
                dec_up  ^= topMask;
                dec_tag ^= topMask;
            }
        }
    }

    // Output preserves code orientation.
    const bool row = isRow(code);
    Value out = Value::matrix(row ? 1 : len, row ? len : 1,
                              ValueType::DOUBLE, mr);
    if (len > 0) std::copy(dseq.begin(), dseq.end(), out.doubleDataMut());
    return out;
}

} // namespace numkit::comm
