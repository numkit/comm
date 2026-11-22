// toolboxes/comm/src/modulation/analog.cpp
//
// Analog & MSK modulators / demodulators (pmmod/pmdemod, ammod/amdemod,
// fmmod/fmdemod, ssbmod/ssbdemod, mskmod/mskdemod), delegating per-channel
// numerical kernels to keel::comm.

#include <keel/arena.h>
#include <keel/array.h>
#include <keel/comm.h>
#include <keel/complex.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <numkit/comm/modulation/analog.hpp>

#include <numkit/signal/digital_filtering/filter.hpp>
#include <numkit/signal/filter_design/filter_design.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include <utility>

namespace numkit::comm {

namespace {

void checkPositiveCarrier(keel::f64 fc, keel::f64 fs, const char *fn,
                          const char *id_fs, const char *id_fc, const char *id_nyq,
                          bool strict_nyq = false)
{
    if (!(fs > 0.0))
        throw Error(keel::format("{}: Fs must be positive", fn), 0, 0, fn, "", id_fs);
    if (!(fc > 0.0))
        throw Error(keel::format("{}: Fc must be positive", fn), 0, 0, fn, "", id_fc);
    if (strict_nyq ? (fs <= 2.0 * fc) : (fs < 2.0 * fc))
        throw Error(keel::format("{}: Fs must be {} 2*Fc", fn, strict_nyq ? ">" : ">="),
                    0, 0, fn, "", id_nyq);
}

template <typename Fn>
Value runRealCols(const Value &x, keel::Allocator *mr, Fn &&fn)
{
    const auto &d = x.dims();
    keel::usize H = d.rows();
    keel::usize W = d.cols();
    const bool was_row = (H == 1 && W >= 1);
    if (was_row) std::swap(H, W);

    Value out = was_row ? Value::matrix(1, H, ValueType::DOUBLE, mr)
                        : Value::matrix(H, W, ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> col(H, 0.0, scratch.allocator());
    for (keel::usize c = 0; c < W; ++c) {
        for (keel::usize r = 0; r < H; ++r)
            col[r] = was_row ? x.elemAsDouble(r) : x.elemAsDouble(c * H + r);
        fn(col.as_ref(), keel::MutArrayRef<keel::f64>(od + c * H, H), scratch.allocator());
    }
    return out;
}

Value coherentDetect(const Value &y, keel::f64 fc, keel::f64 fs, keel::f64 ini_phase,
                     keel::Allocator *mr)
{
    const auto &d = y.dims();
    const keel::usize N = y.numel();
    Value yc = Value::matrix(d.rows(), d.cols(), ValueType::DOUBLE, mr);
    keel::f64 *yo = yc.doubleDataMut();
    const keel::f64 twoPiFc = keel::kTau * fc;
    const keel::f64 inv_fs = 1.0 / fs;
    for (keel::usize i = 0; i < N; ++i)
        yo[i] = y.elemAsDouble(i) * keel::cos(twoPiFc * (keel::as<keel::f64>(i) * inv_fs) + ini_phase);

    auto [b, a] = numkit::signal::butter(5, fc * 2.0 / fs, "low", mr);
    Value z = numkit::signal::filtfilt(b, a, yc, mr);

    Value out = Value::matrix(d.rows(), d.cols(), ValueType::DOUBLE, mr);
    keel::f64 *oo = out.doubleDataMut();
    const keel::f64 *zd = z.doubleData();
    for (keel::usize i = 0; i < N; ++i) oo[i] = 2.0 * zd[i];
    return out;
}

} // namespace

Value pmmod(const Value &x, keel::f64 fc, keel::f64 fs, keel::f64 phasedev,
            keel::f64 ini_phase, keel::Allocator *mr)
{
    checkPositiveCarrier(fc, fs, "pmmod", "numkit:pmmod:Fs", "numkit:pmmod:Fc",
                         "numkit:pmmod:FsLessThan2Fc");
    if (!(phasedev > 0.0))
        throw Error("pmmod: phasedev must be positive",
                    0, 0, "pmmod", "", "numkit:pmmod:InvalidPhaseDev");

    return runRealCols(x, mr, [&](keel::ArrayRef<keel::f64> in, keel::MutArrayRef<keel::f64> out,
                                  keel::Allocator *) {
        (void)keel::comm::pm_mod(in, fc, fs, phasedev, ini_phase, out);
    });
}

Value ammod(const Value &x, keel::f64 fc, keel::f64 fs, keel::f64 ini_phase,
            keel::f64 carr_amp, keel::Allocator *mr)
{
    checkPositiveCarrier(fc, fs, "ammod", "numkit:ammod:Fs", "numkit:ammod:Fc",
                         "numkit:ammod:FsLessThan2Fc");

    return runRealCols(x, mr, [&](keel::ArrayRef<keel::f64> in, keel::MutArrayRef<keel::f64> out,
                                  keel::Allocator *) {
        (void)keel::comm::am_mod(in, fc, fs, ini_phase, carr_amp, out);
    });
}

Value fmmod(const Value &x, keel::f64 fc, keel::f64 fs, keel::f64 freqdev,
            keel::f64 ini_phase, keel::Allocator *mr)
{
    if (!(fs > 0.0))
        throw Error("fmmod: Fs must be positive", 0, 0, "fmmod", "", "numkit:fmmod:Fs");
    if (fc < 0.0)
        throw Error("fmmod: Fc must be non-negative", 0, 0, "fmmod", "", "numkit:fmmod:Fc");
    if (fs < 2.0 * fc)
        throw Error("fmmod: Fs must be >= 2*Fc", 0, 0, "fmmod", "", "numkit:fmmod:FsLessThan2Fc");
    if (!(freqdev > 0.0))
        throw Error("fmmod: freqdev must be positive",
                    0, 0, "fmmod", "", "numkit:fmmod:InvalidFreqdev");

    return runRealCols(x, mr, [&](keel::ArrayRef<keel::f64> in, keel::MutArrayRef<keel::f64> out,
                                  keel::Allocator *) {
        (void)keel::comm::fm_mod(in, fc, fs, freqdev, ini_phase, out);
    });
}

Value mskmod(const Value &x, keel::i32 nSamp, keel::f64 ini_phase, keel::Allocator *mr)
{
    if (nSamp <= 0)
        throw Error("mskmod: nSamp must be a positive integer",
                    0, 0, "mskmod", "", "numkit:mskmod:nSamp");
    for (keel::usize i = 0; i < x.numel(); ++i) {
        const keel::f64 xi = x.elemAsDouble(i);
        if (xi != 0.0 && xi != 1.0)
            throw Error("mskmod: input must be binary (0 or 1)",
                        0, 0, "mskmod", "", "numkit:mskmod:NotBinary");
    }

    const auto &d = x.dims();
    keel::usize H = d.rows(), W = d.cols();
    const bool was_row = (H == 1 && W >= 1);
    if (was_row) std::swap(H, W);

    const keel::usize Nout = H * keel::as<keel::usize>(nSamp);
    Value out = was_row ? Value::matrix(1, Nout, ValueType::COMPLEX, mr)
                        : Value::matrix(Nout, W, ValueType::COMPLEX, mr);
    keel::c64 *od = out.complexDataMut();

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::u8> bits(H, 0u, scratch.allocator());
    for (keel::usize c = 0; c < W; ++c) {
        for (keel::usize r = 0; r < H; ++r) {
            const keel::f64 xi = was_row ? x.elemAsDouble(r) : x.elemAsDouble(c * H + r);
            bits[r] = (xi != 0.0) ? 1u : 0u;
        }
        (void)keel::comm::msk_mod_diff(bits.as_ref(), nSamp, ini_phase,
                                       keel::MutArrayRef<keel::c64>(od + c * Nout, Nout),
                                       scratch.allocator());
    }
    return out;
}

Value mskdemod(const Value &y, keel::i32 nSamp, keel::f64 ini_phase,
               keel::f64 *phase_out, keel::Allocator *mr)
{
    if (nSamp <= 0)
        throw Error("mskdemod: nSamp must be a positive integer",
                    0, 0, "mskdemod", "", "numkit:mskdemod:nSamp");
    const auto &d = y.dims();
    keel::usize H = d.rows(), W = d.cols();
    const bool was_row = (H == 1 && W >= 1);
    if (was_row) std::swap(H, W);
    if (H == 0 || H % keel::as<keel::usize>(nSamp) != 0)
        throw Error("mskdemod: signal length must be a positive multiple of nSamp",
                    0, 0, "mskdemod", "", "numkit:mskdemod:length");

    const keel::usize N = H / keel::as<keel::usize>(nSamp);
    const bool cplx = y.isComplex();
    Value out = was_row ? Value::matrix(1, N, ValueType::DOUBLE, mr)
                        : Value::matrix(N, W, ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::c64> col(H, keel::c64(0.0, 0.0), scratch.allocator());
    keel::Array<keel::u8> bits(N, 0u, scratch.allocator());
    keel::f64 lastPhase = ini_phase;

    for (keel::usize c = 0; c < W; ++c) {
        for (keel::usize r = 0; r < H; ++r) {
            const keel::usize idx = was_row ? r : (c * H + r);
            col[r] = cplx ? y.complexData()[idx] : keel::c64(y.elemAsDouble(idx), 0.0);
        }
        (void)keel::comm::msk_demod_diff(col.as_ref(), nSamp, ini_phase, bits.as_mut_ref(),
                                         &lastPhase);
        for (keel::usize k = 0; k < N; ++k)
            od[c * N + k] = keel::as<keel::f64>(bits[k]);
    }
    if (phase_out) *phase_out = lastPhase;
    return out;
}

Value ssbmod(const Value &x, keel::f64 fc, keel::f64 fs, keel::f64 ini_phase,
             bool upper, keel::Allocator *mr)
{
    checkPositiveCarrier(fc, fs, "ssbmod", "numkit:ssbmod:Fs", "numkit:ssbmod:Fc",
                         "numkit:ssbmod:Fs2Fc", /*strict_nyq=*/true);

    return runRealCols(x, mr, [&](keel::ArrayRef<keel::f64> in, keel::MutArrayRef<keel::f64> out,
                                  keel::Allocator *alloc) {
        (void)keel::comm::ssb_mod(in, fc, fs, ini_phase, upper, out, alloc);
    });
}

Value pmdemod(const Value &y, keel::f64 fc, keel::f64 fs, keel::f64 phasedev,
              keel::f64 ini_phase, keel::Allocator *mr)
{
    checkPositiveCarrier(fc, fs, "pmdemod", "numkit:pmdemod:Fs", "numkit:pmdemod:Fc",
                         "numkit:pmdemod:FsLessThan2Fc");
    if (!(phasedev > 0.0))
        throw Error("pmdemod: phasedev must be positive",
                    0, 0, "pmdemod", "", "numkit:pmdemod:InvalidPhaseDev");

    return runRealCols(y, mr, [&](keel::ArrayRef<keel::f64> in, keel::MutArrayRef<keel::f64> out,
                                  keel::Allocator *alloc) {
        (void)keel::comm::pm_demod(in, fc, fs, phasedev, ini_phase, out, alloc);
    });
}

Value fmdemod(const Value &y, keel::f64 fc, keel::f64 fs, keel::f64 freqdev,
              keel::f64 ini_phase, keel::Allocator *mr)
{
    checkPositiveCarrier(fc, fs, "fmdemod", "numkit:fmdemod:Fs", "numkit:fmdemod:Fc",
                         "numkit:fmdemod:FsLessThan2Fc");
    if (!(freqdev > 0.0))
        throw Error("fmdemod: freqdev must be positive",
                    0, 0, "fmdemod", "", "numkit:fmdemod:InvalidFreqDev");

    return runRealCols(y, mr, [&](keel::ArrayRef<keel::f64> in, keel::MutArrayRef<keel::f64> out,
                                  keel::Allocator *alloc) {
        (void)keel::comm::fm_demod(in, fc, fs, freqdev, ini_phase, out, alloc);
    });
}

Value amdemod(const Value &y, keel::f64 fc, keel::f64 fs, keel::f64 ini_phase,
              keel::f64 carr_amp, keel::Allocator *mr)
{
    checkPositiveCarrier(fc, fs, "amdemod", "numkit:demod:Fs", "numkit:demod:Fc",
                         "numkit:demod:FsLessThan2Fc");
    Value z = coherentDetect(y, fc, fs, ini_phase, mr);
    if (carr_amp != 0.0) {
        keel::f64 *zd = z.doubleDataMut();
        const keel::usize N = z.numel();
        for (keel::usize i = 0; i < N; ++i) zd[i] -= carr_amp;
    }
    return z;
}

Value ssbdemod(const Value &y, keel::f64 fc, keel::f64 fs, keel::f64 ini_phase,
               keel::Allocator *mr)
{
    checkPositiveCarrier(fc, fs, "ssbdemod", "numkit:demod:Fs", "numkit:demod:Fc",
                         "numkit:demod:FsLessThan2Fc");
    return coherentDetect(y, fc, fs, ini_phase, mr);
}

} // namespace numkit::comm
