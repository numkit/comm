// toolboxes/comm/src/modulation/fsk_ofdm.cpp

#include <keel/array.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/complex.h>
#include <numkit/comm/modulation/fsk_ofdm.hpp>

#include <numkit/signal/transforms/fft.hpp>

#include <numkit/value/argcheck.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace numkit::comm {

namespace {

inline int to_gray(int b) { return b ^ (b >> 1); }
inline int from_gray(int g) {
    int b = g;
    while (g >>= 1) b ^= g;
    return b;
}

inline int symbol_to_freq_idx(int s, int /*M*/, keel::StringRef order) {
    return (order == "bin") ? s : to_gray(s);
}
inline int freq_idx_to_symbol(int k, int /*M*/, keel::StringRef order) {
    return (order == "bin") ? k : from_gray(k);
}

void fskmod_argcheck(int M, double freq_sep, int nsamp, double fs)
{
    if (M < 2)
        throw Error("fskmod: M must be ≥ 2", 0, 0, "fskmod", "",
                    "numkit:fskmod:badM");
    if (freq_sep <= 0.0 || nsamp <= 0 || fs <= 0.0)
        throw Error("fskmod: freq_sep / nsamp / fs must be positive",
                    0, 0, "fskmod", "", "numkit:fskmod:badparam");
}

void fskdemod_argcheck(int M, double freq_sep, int nsamp, double fs)
{
    if (M < 2)
        throw Error("fskdemod: M must be ≥ 2", 0, 0, "fskdemod", "",
                    "numkit:fskdemod:badM");
    if (freq_sep <= 0.0 || nsamp <= 0 || fs <= 0.0)
        throw Error("fskdemod: freq_sep / nsamp / fs must be positive",
                    0, 0, "fskdemod", "", "numkit:fskdemod:badparam");
}

void ofdmmod_argcheck(const Value &in, int nfft, int cplen)
{
    if (nfft <= 0 || cplen < 0)
        throw Error("ofdmmod: bad nfft / cplen", 0, 0, "ofdmmod", "",
                    "numkit:ofdmmod:badparam");
    if ((int)in.dims().rows() != nfft)
        throw Error("ofdmmod: in must have nfft rows", 0, 0, "ofdmmod", "",
                    "numkit:ofdmmod:size");
}

void ofdmdemod_argcheck(const Value &in, int nfft, int cplen)
{
    if (nfft <= 0 || cplen < 0)
        throw Error("ofdmdemod: bad nfft / cplen", 0, 0, "ofdmdemod", "",
                    "numkit:ofdmdemod:badparam");
    const keel::usize out_per_sym = (keel::usize)(nfft + cplen);
    if (in.numel() % out_per_sym != 0)
        throw Error("ofdmdemod: input length must be a multiple of (nfft + cplen)",
                    0, 0, "ofdmdemod", "", "numkit:ofdmdemod:size");
}

} // anonymous

Value fskmod(const Value &x, int M, double freq_sep, int nsamp,
             double fs, keel::StringRef phase_continuity,
             keel::StringRef symbol_order,
             keel::Allocator *mr)
{
    fskmod_argcheck(M, freq_sep, nsamp, fs);

    const keel::usize Nsym = x.numel();
    const keel::usize Nout = Nsym * (keel::usize)nsamp;
    Value out = Value::matrix(Nout, 1, ValueType::COMPLEX, mr);
    if (Nout == 0) return out;
    keel::c64 *od = out.complexDataMut();

    const bool cont = (phase_continuity == "cont");

    double phase = 0.0;
    for (keel::usize s = 0; s < Nsym; ++s) {
        const int sym = (int)x.elemAsDouble(s);
        const int k = symbol_to_freq_idx(sym, M, symbol_order);
        // Tone frequency, centered around 0 (symmetric tones).
        const double f = (double(k) - 0.5 * (M - 1)) * freq_sep;
        const double dphi = 2.0 * M_PI * f / fs;
        for (int n = 0; n < nsamp; ++n) {
            if (cont) {
                od[s * nsamp + (keel::usize)n] = keel::c64(std::cos(phase), std::sin(phase));
                phase += dphi;
            } else {
                const double th = 2.0 * M_PI * f * double(n) / fs;
                od[s * nsamp + (keel::usize)n] = keel::c64(std::cos(th), std::sin(th));
            }
        }
    }
    return out;
}

Value fskdemod(const Value &y, int M, double freq_sep, int nsamp,
               double fs, keel::StringRef symbol_order,
               keel::Allocator *mr)
{
    fskdemod_argcheck(M, freq_sep, nsamp, fs);

    const keel::usize Nsamps = y.numel();
    const keel::usize Nsym = Nsamps / (keel::usize)nsamp;
    Value out = Value::matrix(Nsym, 1, ValueType::DOUBLE, mr);
    if (Nsym == 0) return out;
    double *od = out.doubleDataMut();

    // Pre-compute reference tone phasors.
    const keel::usize ns = static_cast<keel::usize>(nsamp);
    keel::Array<keel::c64> ref(static_cast<keel::usize>(M) * ns, mr);
    for (int k = 0; k < M; ++k) {
        const double f = (double(k) - 0.5 * (M - 1)) * freq_sep;
        const double dphi = 2.0 * M_PI * f / fs;
        for (int n = 0; n < nsamp; ++n) {
            const double th = dphi * double(n);
            ref[static_cast<keel::usize>(k) * ns + static_cast<keel::usize>(n)] = keel::c64(std::cos(th), std::sin(th));
        }
    }

    for (keel::usize s = 0; s < Nsym; ++s) {
        // Correlate the segment against each candidate tone (conjugate
        // multiply + sum, take magnitude).
        int bestK = 0;
        double bestE = -1.0;
        for (int k = 0; k < M; ++k) {
            keel::c64 acc(0.0, 0.0);
            for (int n = 0; n < nsamp; ++n) {
                keel::c64 yi;
                if (y.type() == ValueType::COMPLEX)
                    yi = y.complexData()[s * nsamp + (keel::usize)n];
                else
                    yi = keel::c64(y.elemAsDouble(s * nsamp + (keel::usize)n), 0.0);
                acc += yi * keel::conj(ref[static_cast<keel::usize>(k) * ns + static_cast<keel::usize>(n)]);
            }
            const double E = keel::norm(acc);
            if (E > bestE) { bestE = E; bestK = k; }
        }
        od[s] = double(freq_idx_to_symbol(bestK, M, symbol_order));
    }
    return out;
}

// ════════════════════════════════════════════════════════════════════
// ofdmmod / ofdmdemod
// ════════════════════════════════════════════════════════════════════

Value ofdmmod(const Value &in, int nfft, int cplen,
              keel::Allocator *mr)
{
    ofdmmod_argcheck(in, nfft, cplen);

    const keel::usize cols = in.dims().cols();
    const keel::usize Nsym = cols;
    const keel::usize out_per_sym = (keel::usize)(nfft + cplen);
    Value out = Value::matrix(out_per_sym * Nsym, 1, ValueType::COMPLEX, mr);
    if (Nsym == 0) return out;
    keel::c64 *od = out.complexDataMut();

    // For each column, take IFFT and prepend the cyclic prefix.
    for (keel::usize s = 0; s < Nsym; ++s) {
        // Extract column s into a 1×nfft Value, ifft along dim 1.
        Value col = Value::matrix((keel::usize)nfft, 1, ValueType::COMPLEX, mr);
        keel::c64 *cd = col.complexDataMut();
        for (int i = 0; i < nfft; ++i) {
            if (in.type() == ValueType::COMPLEX)
                cd[i] = in.complexData()[s * (keel::usize)nfft + (keel::usize)i];
            else
                cd[i] = keel::c64(in.elemAsDouble(s * (keel::usize)nfft + (keel::usize)i), 0.0);
        }
        Value tx = ::numkit::signal::ifft(col, -1, 1, mr);

        keel::c64 *txd = nullptr;
        keel::Array<keel::c64> tmp(mr);
        if (tx.type() == ValueType::COMPLEX) {
            txd = const_cast<keel::c64*>(tx.complexData());
        } else {
            tmp.resize((keel::usize)nfft);
            for (int i = 0; i < nfft; ++i) tmp[(keel::usize)i] = keel::c64(tx.elemAsDouble(i), 0.0);
            txd = tmp.data();
        }

        // CP = last cplen samples of the IFFT output.
        for (int i = 0; i < cplen; ++i)
            od[s * out_per_sym + (keel::usize)i] = txd[(keel::usize)nfft - (keel::usize)cplen + (keel::usize)i];
        // Body = full IFFT.
        for (int i = 0; i < nfft; ++i)
            od[s * out_per_sym + (keel::usize)cplen + (keel::usize)i] = txd[(keel::usize)i];
    }
    return out;
}

Value ofdmdemod(const Value &in, int nfft, int cplen, int symoffset,
                keel::Allocator *mr)
{
    ofdmdemod_argcheck(in, nfft, cplen);
    if (symoffset < 0)         symoffset = cplen;
    if (symoffset > cplen)     symoffset = cplen;

    const keel::usize out_per_sym = (keel::usize)(nfft + cplen);
    const keel::usize Nsamps = in.numel();
    const keel::usize Nsym = Nsamps / out_per_sym;

    Value out = Value::matrix((keel::usize)nfft, Nsym, ValueType::COMPLEX, mr);
    keel::c64 *od = out.complexDataMut();

    for (keel::usize s = 0; s < Nsym; ++s) {
        // Drop the CP region (size symoffset, default cplen).
        Value col = Value::matrix((keel::usize)nfft, 1, ValueType::COMPLEX, mr);
        keel::c64 *cd = col.complexDataMut();
        for (int i = 0; i < nfft; ++i) {
            const keel::usize src = s * out_per_sym + (keel::usize)symoffset + (keel::usize)i;
            if (in.type() == ValueType::COMPLEX) cd[i] = in.complexData()[src];
            else                                  cd[i] = keel::c64(in.elemAsDouble(src), 0.0);
        }
        Value rx = ::numkit::signal::fft(col, -1, 1, mr);
        // Pack into the output column.
        const keel::c64 *rxd = rx.complexData();
        for (int i = 0; i < nfft; ++i)
            od[s * (keel::usize)nfft + (keel::usize)i] = rxd[i];
    }
    return out;
}

} // namespace numkit::comm
