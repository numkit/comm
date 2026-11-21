// toolboxes/comm/src/modulation/analog.cpp
//
// Analog modulators: pmmod (phase modulation).
//
// Forms a new TU since the analog mod/demod family expands over time
// (ammod, fmmod, ssbmod planned). Each is a closed-form expression
// in cos / sin of a phase term; demods can pull in filtfilt or
// Hilbert and live alongside.

#include <keel/array.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/complex.h>
#include <numkit/comm/modulation/analog.hpp>

#include <numkit/signal/transforms/hilbert.hpp>
#include <numkit/signal/filter_design/filter_design.hpp>     // butter
#include <numkit/signal/digital_filtering/filter.hpp>        // filtfilt

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

void pmmod_argcheck(double fc, double fs, double phasedev)
{
    if (!(fs > 0.0))
        throw Error("pmmod: Fs must be positive",
                    0, 0, "pmmod", "", "numkit:pmmod:Fs");
    if (!(fc > 0.0))
        throw Error("pmmod: Fc must be positive",
                    0, 0, "pmmod", "", "numkit:pmmod:Fc");
    if (fs < 2.0 * fc)
        throw Error("pmmod: Fs must be >= 2*Fc",
                    0, 0, "pmmod", "", "numkit:pmmod:FsLessThan2Fc");
    if (!(phasedev > 0.0))
        throw Error("pmmod: phasedev must be positive",
                    0, 0, "pmmod", "", "numkit:pmmod:InvalidPhaseDev");
}

void ammod_argcheck(double fc, double fs)
{
    if (!(fs > 0.0))
        throw Error("ammod: Fs must be positive",
                    0, 0, "ammod", "", "numkit:ammod:Fs");
    if (!(fc > 0.0))
        throw Error("ammod: Fc must be positive",
                    0, 0, "ammod", "", "numkit:ammod:Fc");
    if (fs < 2.0 * fc)
        throw Error("ammod: Fs must be >= 2*Fc",
                    0, 0, "ammod", "", "numkit:ammod:FsLessThan2Fc");
}

void fmmod_argcheck(double fc, double fs, double freqdev)
{
    if (!(fs > 0.0))
        throw Error("fmmod: Fs must be positive",
                    0, 0, "fmmod", "", "numkit:fmmod:Fs");
    if (fc < 0.0)
        throw Error("fmmod: Fc must be non-negative",
                    0, 0, "fmmod", "", "numkit:fmmod:Fc");
    if (fs < 2.0 * fc)
        throw Error("fmmod: Fs must be >= 2*Fc",
                    0, 0, "fmmod", "", "numkit:fmmod:FsLessThan2Fc");
    if (!(freqdev > 0.0))
        throw Error("fmmod: freqdev must be positive",
                    0, 0, "fmmod", "", "numkit:fmmod:InvalidFreqdev");
}

void mskmod_argcheck(const Value &x, int nSamp)
{
    if (nSamp <= 0)
        throw Error("mskmod: nSamp must be a positive integer",
                    0, 0, "mskmod", "", "numkit:mskmod:nSamp");
    const keel::usize n = x.numel();
    for (keel::usize i = 0; i < n; ++i) {
        const double xi = x.elemAsDouble(i);
        if (xi != 0.0 && xi != 1.0)
            throw Error("mskmod: input must be binary (0 or 1)",
                        0, 0, "mskmod", "", "numkit:mskmod:NotBinary");
    }
}

void mskdemod_argcheck(const Value &y, int nSamp)
{
    if (nSamp <= 0)
        throw Error("mskdemod: nSamp must be a positive integer",
                    0, 0, "mskdemod", "", "numkit:mskdemod:nSamp");
    const auto &d = y.dims();
    const keel::usize H = (d.rows() == 1 && d.cols() >= 1) ? d.cols() : d.rows();
    if (H == 0 || H % keel::as<keel::usize>(nSamp) != 0)
        throw Error("mskdemod: signal length must be a positive multiple of nSamp",
                    0, 0, "mskdemod", "", "numkit:mskdemod:length");
}

void ssbmod_argcheck(double fc, double fs)
{
    if (!(fs > 0.0))
        throw Error("ssbmod: Fs must be positive",
                    0, 0, "ssbmod", "", "numkit:ssbmod:Fs");
    if (!(fc > 0.0))
        throw Error("ssbmod: Fc must be positive",
                    0, 0, "ssbmod", "", "numkit:ssbmod:Fc");
    if (fs <= 2.0 * fc)
        throw Error("ssbmod: Fs must be > 2*Fc",
                    0, 0, "ssbmod", "", "numkit:ssbmod:Fs2Fc");
}

void pmdemod_argcheck(double fc, double fs, double phasedev)
{
    if (!(fs > 0.0))
        throw Error("pmdemod: Fs must be positive", 0, 0, "pmdemod", "", "numkit:pmdemod:Fs");
    if (!(fc > 0.0))
        throw Error("pmdemod: Fc must be positive", 0, 0, "pmdemod", "", "numkit:pmdemod:Fc");
    if (fs < 2.0 * fc)
        throw Error("pmdemod: Fs must be >= 2*Fc", 0, 0, "pmdemod", "", "numkit:pmdemod:FsLessThan2Fc");
    if (!(phasedev > 0.0))
        throw Error("pmdemod: phasedev must be positive", 0, 0, "pmdemod", "", "numkit:pmdemod:InvalidPhaseDev");
}

void fmdemod_argcheck(double fc, double fs, double freqdev)
{
    if (!(fs > 0.0))
        throw Error("fmdemod: Fs must be positive", 0, 0, "fmdemod", "", "numkit:fmdemod:Fs");
    if (!(fc > 0.0))
        throw Error("fmdemod: Fc must be positive", 0, 0, "fmdemod", "", "numkit:fmdemod:Fc");
    if (fs < 2.0 * fc)
        throw Error("fmdemod: Fs must be >= 2*Fc", 0, 0, "fmdemod", "", "numkit:fmdemod:FsLessThan2Fc");
    if (!(freqdev > 0.0))
        throw Error("fmdemod: freqdev must be positive", 0, 0, "fmdemod", "", "numkit:fmdemod:InvalidFreqDev");
}

} // namespace

// ── pmmod ──────────────────────────────────────────────────────────
// Phase modulation — closed-form definition (Haykin, "Communication Systems"):
//   t = (0:1/Fs:(N-1)/Fs)'
//   y = cos(2*pi*Fc*t + phasedev*x + ini_phase)
//
// Vector-orientation contract: 1-D row preserved as row in output;
// columns processed independently.
Value pmmod(const Value &x, double fc, double fs, double phasedev,
            double ini_phase, keel::Allocator *mr)
{
    pmmod_argcheck(fc, fs, phasedev);

    const auto &d = x.dims();
    keel::usize H = d.rows();
    keel::usize W = d.cols();
    const bool was_row = (H == 1 && W >= 1);
    if (was_row) {
        // Reorient row -> column for processing.
        std::swap(H, W);
    }

    Value out = Value::matrix(H, W, ValueType::DOUBLE, mr);
    double *o = out.doubleDataMut();

    const double twoPiFc = 2.0 * M_PI * fc;
    const double inv_fs  = 1.0 / fs;

    // Process column-major: for each column index c, walk rows.
    for (keel::usize c = 0; c < W; ++c) {
        for (keel::usize r = 0; r < H; ++r) {
            const double t  = keel::as<double>(r) * inv_fs;
            const double xi = was_row
                                  ? x.elemAsDouble(r)
                                  : x.elemAsDouble(c * H + r);
            const double phase = twoPiFc * t + phasedev * xi + ini_phase;
            o[c * H + r] = std::cos(phase);
        }
    }

    if (was_row) {
        // Output should be 1xN row to mirror input.
        Value row = Value::matrix(1, H, ValueType::DOUBLE, mr);
        std::copy(o, o + H, row.doubleDataMut());
        return row;
    }
    return out;
}

// ── ammod ──────────────────────────────────────────────────────────
// Amplitude modulation — closed-form definition (Haykin, "Communication Systems"):
//   t = (0:1/Fs:(N-1)/Fs)'
//   y = (x + carr_amp) .* cos(2*pi*Fc*t + ini_phase)
//
// carr_amp == 0 -> DSB-SC (suppressed carrier)
// carr_amp != 0 -> DSB-TC (transmitted carrier)
//
// Vector-orientation contract identical to pmmod.
Value ammod(const Value &x, double fc, double fs, double ini_phase,
            double carr_amp, keel::Allocator *mr)
{
    ammod_argcheck(fc, fs);

    const auto &d = x.dims();
    keel::usize H = d.rows();
    keel::usize W = d.cols();
    const bool was_row = (H == 1 && W >= 1);
    if (was_row) {
        std::swap(H, W);
    }

    Value out = Value::matrix(H, W, ValueType::DOUBLE, mr);
    double *o = out.doubleDataMut();

    const double twoPiFc = 2.0 * M_PI * fc;
    const double inv_fs  = 1.0 / fs;

    for (keel::usize c = 0; c < W; ++c) {
        for (keel::usize r = 0; r < H; ++r) {
            const double t  = keel::as<double>(r) * inv_fs;
            const double xi = was_row
                                  ? x.elemAsDouble(r)
                                  : x.elemAsDouble(c * H + r);
            o[c * H + r] = (xi + carr_amp)
                         * std::cos(twoPiFc * t + ini_phase);
        }
    }

    if (was_row) {
        Value row = Value::matrix(1, H, ValueType::DOUBLE, mr);
        std::copy(o, o + H, row.doubleDataMut());
        return row;
    }
    return out;
}

// ── fmmod ──────────────────────────────────────────────────────────
// Frequency modulation — closed-form definition (Haykin, "Communication Systems"):
//   t      = (0:1/Fs:(N-1)/Fs)'
//   int_x  = cumsum(x) / Fs       (column-wise cumulative sum)
//   y      = cos(2*pi*Fc*t + 2*pi*freqdev*int_x + ini_phase)
//
// Vector-orientation contract identical to pmmod / ammod.
Value fmmod(const Value &x, double fc, double fs, double freqdev,
            double ini_phase, keel::Allocator *mr)
{
    fmmod_argcheck(fc, fs, freqdev);

    const auto &d = x.dims();
    keel::usize H = d.rows();
    keel::usize W = d.cols();
    const bool was_row = (H == 1 && W >= 1);
    if (was_row) {
        std::swap(H, W);
    }

    Value out = Value::matrix(H, W, ValueType::DOUBLE, mr);
    double *o = out.doubleDataMut();

    const double twoPiFc      = 2.0 * M_PI * fc;
    const double twoPiFreqDev = 2.0 * M_PI * freqdev;
    const double inv_fs       = 1.0 / fs;

    // Per column: walk samples, accumulating int_x = cumsum(x)/Fs.
    for (keel::usize c = 0; c < W; ++c) {
        double acc = 0.0;
        for (keel::usize r = 0; r < H; ++r) {
            const double xi = was_row
                                  ? x.elemAsDouble(r)
                                  : x.elemAsDouble(c * H + r);
            acc += xi * inv_fs;            // running cumsum / Fs
            const double t = keel::as<double>(r) * inv_fs;
            o[c * H + r] = std::cos(twoPiFc * t
                                  + twoPiFreqDev * acc
                                  + ini_phase);
        }
    }

    if (was_row) {
        Value row = Value::matrix(1, H, ValueType::DOUBLE, mr);
        std::copy(o, o + H, row.doubleDataMut());
        return row;
    }
    return out;
}

// ── mskmod (differential variant) ──────────────────────────────────
// Minimum-shift keying, differentially-encoded path — standard
// definition (Haykin, "Communication Systems"):
//
//   xPm    = 2*x - 1                       (bits {0,1} -> {-1,+1})
//   xCum   = cumsum([0; xPm])              (length N+1)
//   coarseTime = (0:N)'
//   fineTime   = (0 : 1/nSamp : N - 1/nSamp)'   (length N*nSamp)
//   phaseVec   = pi/2 * interp1(coarseTime, xCum, fineTime)
//   y          = exp(1i * (phaseVec + ini_phase))
//
// Linear interpolation: between adjacent integer samples j and j+1,
// xCum changes monotonically, and the cumulative phase ramps linearly
// so the exponential traces a half-arc of the unit circle per symbol.
//
// KNOWN GAP: non-differential MSK path is deferred -- it requires
// rectpulse + circshift on the I/Q rails (we have rectpulse but the
// arrangement is more involved; will get its own cycle).
Value mskmod(const Value &x, int nSamp, double ini_phase,
             keel::Allocator *mr)
{
    mskmod_argcheck(x, nSamp);

    const auto &d = x.dims();
    keel::usize H = d.rows();
    keel::usize W = d.cols();
    const bool was_row = (H == 1 && W >= 1);
    if (was_row) std::swap(H, W);

    // Build xCum per column.
    const keel::usize Nout = H * keel::as<keel::usize>(nSamp);
    Value out = Value::matrix(Nout, W, ValueType::COMPLEX, mr);
    keel::c64 *o = out.complexDataMut();

    keel::Array<double> xCum(H + 1, mr);
    for (keel::usize c = 0; c < W; ++c) {
        // Compute xCum: xCum[0] = 0; xCum[k+1] = xCum[k] + (2*x[k] - 1)
        xCum[0] = 0.0;
        for (keel::usize r = 0; r < H; ++r) {
            const double xi = was_row
                                  ? x.elemAsDouble(r)
                                  : x.elemAsDouble(c * H + r);
            // Validate {0, 1}.
            if (xi != 0.0 && xi != 1.0)
                throw Error("mskmod: input must be binary (0 or 1)",
                            0, 0, "mskmod", "", "numkit:mskmod:NotBinary");
            xCum[r + 1] = xCum[r] + (2.0 * xi - 1.0);
        }

        const double inv_nSamp = 1.0 / keel::as<double>(nSamp);
        for (keel::usize j = 0; j < Nout; ++j) {
            const double tFine = keel::as<double>(j) * inv_nSamp;
            const keel::usize base  = keel::as<keel::usize>(std::floor(tFine));
            const double frac  = tFine - keel::as<double>(base);
            // Guard tail: at tFine = N - 1/nSamp the base = N-1, so
            // base+1 = N is in range.
            const double interp = xCum[base]
                                + frac * (xCum[base + 1] - xCum[base]);
            const double phase  = M_PI * 0.5 * interp + ini_phase;
            o[c * Nout + j] = keel::c64(std::cos(phase), std::sin(phase));
        }
    }

    if (was_row) {
        // Output matches MATLAB: 1xN row -> 1x(N*nSamp) row.
        Value row = Value::matrix(1, Nout, ValueType::COMPLEX, mr);
        std::copy(o, o + Nout, row.complexDataMut());
        return row;
    }
    return out;
}

// ── mskdemod (differential variant) ────────────────────────────────
// Coherent inverse of mskmod. Each bit is the sign of the symbol's
// accumulated phase increment: bit_k = (Σ angle(y[n]·conj(y[n-1])) > 0)
// over the within-symbol sample pairs n = k·nSamp+1 … (k+1)·nSamp−1.
// Using phase *increments* makes the decision invariant to a constant
// phase rotation (and robust to noise), so ini_phase only feeds the
// returned final-phase state. Matches MATLAB R2025b mskdemod (default
// 'diff'); the non-differential path is deferred, as for mskmod.
Value mskdemod(const Value &y, int nSamp, double ini_phase,
               double *phase_out, keel::Allocator *mr)
{
    mskdemod_argcheck(y, nSamp);

    const auto &d = y.dims();
    keel::usize H = d.rows();
    keel::usize W = d.cols();
    const bool was_row = (H == 1 && W >= 1);
    if (was_row) std::swap(H, W);   // a row is one channel of length H

    const keel::usize N = H / keel::as<keel::usize>(nSamp);   // symbols per channel
    const bool cplx = y.isComplex();

    Value out = Value::matrix(N, W, ValueType::DOUBLE, mr);
    double *o = out.doubleDataMut();

    auto getC = [&](keel::usize col, keel::usize row) -> keel::c64 {
        const keel::usize idx = was_row ? row : (col * H + row);
        return cplx ? y.complexData()[idx] : keel::c64(y.elemAsDouble(idx), 0.0);
    };

    double lastPhase = ini_phase;
    for (keel::usize c = 0; c < W; ++c) {
        long long cumSum = 0;
        for (keel::usize k = 0; k < N; ++k) {
            const keel::usize start = k * keel::as<keel::usize>(nSamp);
            double acc = 0.0;
            for (keel::usize n = start + 1; n < start + keel::as<keel::usize>(nSamp); ++n) {
                const keel::c64 p = getC(c, n) * keel::conj(getC(c, n - 1));
                acc += std::atan2(p.imag(), p.real());
            }
            const int bit = (acc > 0.0) ? 1 : 0;
            o[c * N + k] = keel::as<double>(bit);
            cumSum += (2 * bit - 1);
        }
        double ph = std::fmod(ini_phase + (M_PI * 0.5) * (double)cumSum, 2.0 * M_PI);
        if (ph < 0.0) ph += 2.0 * M_PI;
        lastPhase = ph;
    }
    if (phase_out) *phase_out = lastPhase;

    if (was_row) {
        Value row = Value::matrix(1, N, ValueType::DOUBLE, mr);
        std::copy(o, o + N, row.doubleDataMut());
        return row;
    }
    return out;
}

// ── ssbmod ─────────────────────────────────────────────────────────
// Single-sideband modulation — closed-form definition (Haykin, "Communication Systems"):
//   t = (0:1/Fs:(N-1)/Fs)'
//   Lower sideband (default):
//     y = x.*cos(2π·Fc·t + ini_phase)
//       + imag(hilbert(x)).*sin(2π·Fc·t + ini_phase)
//   Upper sideband ('upper'):
//     y = x.*cos(2π·Fc·t + ini_phase)
//       - imag(hilbert(x)).*sin(2π·Fc·t + ini_phase)
//
// Vector-orientation contract identical to pmmod / ammod / fmmod.
Value ssbmod(const Value &x, double fc, double fs, double ini_phase,
             bool upper, keel::Allocator *mr)
{
    ssbmod_argcheck(fc, fs);

    const auto &d = x.dims();
    keel::usize H = d.rows();
    keel::usize W = d.cols();
    const bool was_row = (H == 1 && W >= 1);
    if (was_row) {
        std::swap(H, W);
    }

    Value out = Value::matrix(H, W, ValueType::DOUBLE, mr);
    double *o = out.doubleDataMut();

    const double twoPiFc = 2.0 * M_PI * fc;
    const double inv_fs  = 1.0 / fs;
    const double sign    = upper ? -1.0 : +1.0;

    // hilbert is column-wise per MATLAB. Extract each column into a
    // 1-D Value and call signal::hilbert on it.
    for (keel::usize c = 0; c < W; ++c) {
        // Build column vector for this c
        Value col_in = Value::matrix(H, 1, ValueType::DOUBLE, mr);
        double *cp = col_in.doubleDataMut();
        for (keel::usize r = 0; r < H; ++r) {
            cp[r] = was_row ? x.elemAsDouble(r)
                            : x.elemAsDouble(c * H + r);
        }
        Value analytic = numkit::signal::hilbert(col_in, mr);
        // analytic.complexData()[i].imag() is imag(hilbert(col_in))
        const auto *cdat = analytic.complexData();
        for (keel::usize r = 0; r < H; ++r) {
            const double t  = keel::as<double>(r) * inv_fs;
            const double xi = cp[r];
            const double ang = twoPiFc * t + ini_phase;
            o[c * H + r] = xi * std::cos(ang)
                         + sign * cdat[r].imag() * std::sin(ang);
        }
    }

    if (was_row) {
        Value row = Value::matrix(1, H, ValueType::DOUBLE, mr);
        std::copy(o, o + H, row.doubleDataMut());
        return row;
    }
    return out;
}

Value pmdemod(const Value &y, double fc, double fs, double phasedev,
              double ini_phase, keel::Allocator *mr)
{
    pmdemod_argcheck(fc, fs, phasedev);

    const auto &d = y.dims();
    keel::usize H = d.rows(), W = d.cols();
    const bool was_row = (H == 1 && W >= 1);
    if (was_row) std::swap(H, W);

    Value out = Value::matrix(H, W, ValueType::DOUBLE, mr);
    double *o = out.doubleDataMut();
    const double twoPiFc = 2.0 * M_PI * fc;
    const double inv_fs = 1.0 / fs;

    for (keel::usize c = 0; c < W; ++c) {
        Value col_in = Value::matrix(H, 1, ValueType::DOUBLE, mr);
        double *cp = col_in.doubleDataMut();
        for (keel::usize r = 0; r < H; ++r)
            cp[r] = was_row ? y.elemAsDouble(r) : y.elemAsDouble(c * H + r);
        Value analytic = numkit::signal::hilbert(col_in, mr);
        const auto *z = analytic.complexData();
        for (keel::usize r = 0; r < H; ++r) {
            const double ang = twoPiFc * (keel::as<double>(r) * inv_fs) + ini_phase;
            const double ca = std::cos(ang), sa = std::sin(ang);
            const double re = z[r].real(), im = z[r].imag();
            // angle(z·exp(-j·ang))
            o[c * H + r] = std::atan2(im * ca - re * sa, re * ca + im * sa) / phasedev;
        }
    }
    if (was_row) {
        Value row = Value::matrix(1, H, ValueType::DOUBLE, mr);
        std::copy(o, o + H, row.doubleDataMut());
        return row;
    }
    return out;
}

Value fmdemod(const Value &y, double fc, double fs, double freqdev,
              double ini_phase, keel::Allocator *mr)
{
    fmdemod_argcheck(fc, fs, freqdev);

    const auto &d = y.dims();
    keel::usize H = d.rows(), W = d.cols();
    const bool was_row = (H == 1 && W >= 1);
    if (was_row) std::swap(H, W);

    Value out = Value::matrix(H, W, ValueType::DOUBLE, mr);
    double *o = out.doubleDataMut();
    const double twoPiFc = 2.0 * M_PI * fc;
    const double inv_fs = 1.0 / fs;
    const double scale = fs / (2.0 * M_PI * freqdev);

    keel::Array<double> ph(H, mr);
    for (keel::usize c = 0; c < W; ++c) {
        Value col_in = Value::matrix(H, 1, ValueType::DOUBLE, mr);
        double *cp = col_in.doubleDataMut();
        for (keel::usize r = 0; r < H; ++r)
            cp[r] = was_row ? y.elemAsDouble(r) : y.elemAsDouble(c * H + r);
        Value analytic = numkit::signal::hilbert(col_in, mr);
        const auto *z = analytic.complexData();
        for (keel::usize r = 0; r < H; ++r) {
            const double ang = twoPiFc * (keel::as<double>(r) * inv_fs) + ini_phase;
            const double ca = std::cos(ang), sa = std::sin(ang);
            const double re = z[r].real(), im = z[r].imag();
            ph[r] = std::atan2(im * ca - re * sa, re * ca + im * sa);
        }
        // x = [0, diff(unwrap(ph))] · fs/(2π·freqdev)
        if (H > 0) o[c * H + 0] = 0.0;
        for (keel::usize r = 1; r < H; ++r) {
            double dlt = ph[r] - ph[r - 1];
            dlt -= 2.0 * M_PI * std::round(dlt / (2.0 * M_PI));   // wrap to [-π, π]
            o[c * H + r] = dlt * scale;
        }
    }
    if (was_row) {
        Value row = Value::matrix(1, H, ValueType::DOUBLE, mr);
        std::copy(o, o + H, row.doubleDataMut());
        return row;
    }
    return out;
}

namespace {
// Coherent (synchronous) detection shared by amdemod / ssbdemod:
//   2 · filtfilt(butter(5, fc·2/fs), y · cos(2π·fc·t + ini_phase)).
Value coherentDetect(const Value &y, double fc, double fs, double ini_phase,
                     keel::Allocator *mr)
{
    const auto &d = y.dims();
    const keel::usize N = y.numel();
    Value yc = Value::matrix(d.rows(), d.cols(), ValueType::DOUBLE, mr);
    double *yo = yc.doubleDataMut();
    const double twoPiFc = 2.0 * M_PI * fc;
    const double inv_fs = 1.0 / fs;
    for (keel::usize i = 0; i < N; ++i)   // flat sample order: sample i → t = i/fs
        yo[i] = y.elemAsDouble(i) * std::cos(twoPiFc * (keel::as<double>(i) * inv_fs) + ini_phase);

    auto [b, a] = numkit::signal::butter(5, fc * 2.0 / fs, "low", mr);
    Value z = numkit::signal::filtfilt(b, a, yc, mr);

    Value out = Value::matrix(d.rows(), d.cols(), ValueType::DOUBLE, mr);
    double *oo = out.doubleDataMut();
    const double *zd = z.doubleData();
    for (keel::usize i = 0; i < N; ++i) oo[i] = 2.0 * zd[i];
    return out;
}

void checkCarrier(double fc, double fs, const char *fn)
{
    if (!(fs > 0.0))
        throw Error(keel::format("{}: Fs must be positive", fn), 0, 0, fn, "", "numkit:demod:Fs");
    if (!(fc > 0.0))
        throw Error(keel::format("{}: Fc must be positive", fn), 0, 0, fn, "", "numkit:demod:Fc");
    if (fs < 2.0 * fc)
        throw Error(keel::format("{}: Fs must be >= 2*Fc", fn), 0, 0, fn, "", "numkit:demod:FsLessThan2Fc");
}
} // anonymous

Value amdemod(const Value &y, double fc, double fs, double ini_phase,
              double carr_amp, keel::Allocator *mr)
{
    checkCarrier(fc, fs, "amdemod");
    Value z = coherentDetect(y, fc, fs, ini_phase, mr);
    if (carr_amp != 0.0) {
        double *zd = z.doubleDataMut();
        const keel::usize N = z.numel();
        for (keel::usize i = 0; i < N; ++i) zd[i] -= carr_amp;
    }
    return z;
}

Value ssbdemod(const Value &y, double fc, double fs, double ini_phase,
               keel::Allocator *mr)
{
    checkCarrier(fc, fs, "ssbdemod");
    return coherentDetect(y, fc, fs, ini_phase, mr);
}

} // namespace numkit::comm
