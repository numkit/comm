// toolboxes/comm/src/channel/channel.cpp

#include <keel/array.h>
#include <keel/comm.h>
#include <keel/complex.h>
#include <keel/core.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/specfun.h>
#include <keel/tuple.h>
#include <numkit/comm/channel/channel.hpp>

#include <numkit/ops/rng_context.hpp>

#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>


namespace numkit::comm {

namespace {

// Compute average power of a complex / real array.
keel::f64 avg_power(const Value &x) {
    const keel::usize N = x.numel();
    if (N == 0) return 0.0;
    keel::f64 s = 0.0;
    if (x.type() == ValueType::COMPLEX) {
        const keel::c64 *cd = x.complexData();
        for (keel::usize i = 0; i < N; ++i) s += keel::norm(cd[i]);
    } else {
        for (keel::usize i = 0; i < N; ++i) {
            const keel::f64 v = x.elemAsDouble(i);
            s += v * v;
        }
    }
    return s / keel::f64(N);
}

template <typename Op>
Value elementwise(keel::Allocator *mr, const Value &x, Op op) {
    const keel::usize N = x.numel();
    if (x.isScalar()) return Value::scalar(op(x.toScalar()), mr);
    const auto &d = x.dims();
    Value out;
    if (d.is3D()) out = Value::matrix3d(d.rows(), d.cols(), d.pages(), ValueType::DOUBLE, mr);
    else          out = Value::matrix(d.rows(), d.cols(), ValueType::DOUBLE, mr);
    if (N == 0) return out;
    keel::f64 *od = out.doubleDataMut();
    for (keel::usize i = 0; i < N; ++i) od[i] = op(x.elemAsDouble(i));
    return out;
}

void awgn_argcheck(const Value &x, keel::f64 snr_db, keel::f64 sigpower_db)
{
    (void)x;
    (void)snr_db;
    (void)sigpower_db;
}

void wgn_argcheck(keel::i32 m, keel::i32 n, keel::f64 p, keel::StringRef type)
{
    (void)m;
    (void)n;
    (void)p;
    (void)type;
}

void bsc_argcheck(const Value &x, keel::f64 p)
{
    (void)x;
    if (p < 0.0 || p > 1.0)
        throw Error("bsc: p must be in [0, 1]", 0, 0, "bsc", "",
                    "numkit:bsc:badp");
}

void qfunc_argcheck(const Value &x)
{
    (void)x;
}

void qfuncinv_argcheck(const Value &p)
{
    (void)p;
}

void marcumq_argcheck(const Value &a_v, const Value &b_v, keel::i32 m)
{
    (void)a_v;
    (void)b_v;
    (void)m;
}

void berawgn_argcheck(const Value &EbNo_dB, keel::StringRef mod, keel::i32 M)
{
    (void)EbNo_dB;
    (void)mod;
    (void)M;
}

void noisebw_argcheck(const Value &num, const Value &den, keel::i32 Nsamp, keel::f64 fs)
{
    (void)num;
    (void)den;
    (void)Nsamp;
    (void)fs;
}

void berconfint_argcheck(keel::f64 numErrs, keel::f64 numBits, keel::f64 level)
{
    if (!(numBits > 0.0))
        throw Error("berconfint: numBits must be positive",
                    0, 0, "berconfint", "", "numkit:berconfint:nbits");
    if (numErrs < 0.0 || numErrs > numBits)
        throw Error("berconfint: numErrs must satisfy 0 ≤ numErrs ≤ numBits",
                    0, 0, "berconfint", "", "numkit:berconfint:nerrs");
    if (!(level > 0.0 && level < 1.0))
        throw Error("berconfint: level must lie in (0, 1)",
                    0, 0, "berconfint", "", "numkit:berconfint:level");
}

void convertSNR_argcheck(const Value &snr_in, keel::StringRef in_type,
                         keel::StringRef out_type, keel::i32 bits_per_symbol)
{
    (void)snr_in;
    (void)in_type;
    (void)out_type;
    (void)bits_per_symbol;
}

} // anonymous

Value awgn(::numkit::ops::RngContext &rng, const Value &x, keel::f64 snr_db, keel::f64 sigpower_db,
           keel::Allocator *mr)
{
    awgn_argcheck(x, snr_db, sigpower_db);
    const bool is_complex = (x.type() == ValueType::COMPLEX);
    keel::f64 sig_pow = (sigpower_db < -1e9)  // -inf sentinel = "measured"
                     ? avg_power(x)
                     : keel::pow(10.0, sigpower_db / 10.0);
    if (sig_pow <= 0.0) sig_pow = 1.0;
    const keel::f64 snr_lin = keel::pow(10.0, snr_db / 10.0);
    const keel::f64 noise_pow = sig_pow / snr_lin;
    const keel::f64 sigma = keel::sqrt(is_complex ? noise_pow / 2.0 : noise_pow);

    const keel::usize N = x.numel();
    Value out;
    const auto &d = x.dims();
    const ValueType ty = is_complex ? ValueType::COMPLEX : ValueType::DOUBLE;
    if (x.isScalar()) out = Value::matrix(1, 1, ty, mr);
    else if (d.is3D())out = Value::matrix3d(d.rows(), d.cols(), d.pages(), ty, mr);
    else              out = Value::matrix(d.rows(), d.cols(), ty, mr);

    if (is_complex) {
        keel::c64 *od = out.complexDataMut();
        const keel::c64 *xd = x.complexData();
        for (keel::usize i = 0; i < N; ++i) {
            keel::f64 re = sigma * rng.nextNormal();
            keel::f64 im = sigma * rng.nextNormal();
            od[i] = xd[i] + keel::c64(re, im);
        }
    } else {
        keel::f64 *od = out.doubleDataMut();
        for (keel::usize i = 0; i < N; ++i)
            od[i] = x.elemAsDouble(i) + sigma * rng.nextNormal();
    }
    return out;
}

Value wgn(::numkit::ops::RngContext &rng, keel::i32 m, keel::i32 n, keel::f64 p, keel::StringRef type,
          bool complex_out, keel::Allocator *mr)
{
    wgn_argcheck(m, n, p, type);
    keel::f64 power_lin = 0.0;
    if (type == "linear") power_lin = p;
    else if (type == "dBm") power_lin = keel::pow(10.0, (p - 30.0) / 10.0);
    else /* dBW */         power_lin = keel::pow(10.0, p / 10.0);
    if (power_lin < 0.0) power_lin = 0.0;
    const keel::f64 sigma = keel::sqrt(complex_out ? power_lin / 2.0 : power_lin);

    if (complex_out) {
        Value out = Value::matrix(m, n, ValueType::COMPLEX, mr);
        keel::c64 *od = out.complexDataMut();
        const keel::usize N = (keel::usize)m * (keel::usize)n;
        for (keel::usize i = 0; i < N; ++i) {
            keel::f64 re = sigma * rng.nextNormal();
            keel::f64 im = sigma * rng.nextNormal();
            od[i] = keel::c64(re, im);
        }
        return out;
    }
    Value out = Value::matrix(m, n, ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();
    const keel::usize N = (keel::usize)m * (keel::usize)n;
    for (keel::usize i = 0; i < N; ++i) od[i] = sigma * rng.nextNormal();
    return out;
}

Value bsc(::numkit::ops::RngContext &rng, const Value &x, keel::f64 p, keel::Allocator *mr) {
    bsc_argcheck(x, p);

    const keel::usize N = x.numel();
    Value out;
    const auto &d = x.dims();
    if (x.isScalar()) out = Value::matrix(1, 1, ValueType::DOUBLE, mr);
    else if (d.is3D())out = Value::matrix3d(d.rows(), d.cols(), d.pages(), ValueType::DOUBLE, mr);
    else              out = Value::matrix(d.rows(), d.cols(), ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();
    for (keel::usize i = 0; i < N; ++i) {
        const keel::i32 b = (keel::i32)x.elemAsDouble(i) & 1;
        od[i] = keel::f64(rng.nextUniform() < p ? (b ^ 1) : b);
    }
    return out;
}

Value qfunc(const Value &x, keel::Allocator *mr) {
    qfunc_argcheck(x);
    return elementwise(mr, x, [](keel::f64 v) {
        return keel::comm::qfunc(v);
    });
}

Value qfuncinv(const Value &p, keel::Allocator *mr) {
    qfuncinv_argcheck(p);
    return elementwise(mr, p, [](keel::f64 v) {
        if (!(v > 0.0 && v < 1.0)) return keel::Limits<keel::f64>::quiet_nan();
        // Q(x) = 0.5 * erfc(x / sqrt(2)). Inverse: x = sqrt(2) * erfcinv(2v).
        return keel::sqrt(2.0) * keel::specfun::erfcinv(2.0 * v);
    });
}

Value marcumq(const Value &a_v, const Value &b_v, keel::i32 m,
              keel::Allocator *mr)
{
    marcumq_argcheck(a_v, b_v, m);
    if (m < 1) m = 1;
    const keel::usize Na = a_v.numel(), Nb = b_v.numel();
    const keel::usize N = keel::max(Na, Nb);
    Value out = Value::matrix((Na > Nb) ? Na : Nb, 1, ValueType::DOUBLE, mr);
    if (N == 0) return out;
    keel::f64 *od = out.doubleDataMut();
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 a = a_v.elemAsDouble(Na > 1 ? i : 0);
        const keel::f64 b = b_v.elemAsDouble(Nb > 1 ? i : 0);
        od[i] = keel::specfun::marcumq(a, b, m);
    }
    return out;
}

Value berawgn(const Value &EbNo_dB, keel::StringRef mod, keel::i32 M,
              keel::Allocator *mr)
{
    berawgn_argcheck(EbNo_dB, mod, M);
    keel::comm::AwgnMod awgn_mod = keel::comm::AwgnMod::kPsk;
    bool valid_mod = true;
    if (mod == "psk")       awgn_mod = keel::comm::AwgnMod::kPsk;
    else if (mod == "qam")  awgn_mod = keel::comm::AwgnMod::kQam;
    else if (mod == "pam")  awgn_mod = keel::comm::AwgnMod::kPam;
    else if (mod == "fsk")  awgn_mod = keel::comm::AwgnMod::kFsk;
    else if (mod == "dpsk") awgn_mod = keel::comm::AwgnMod::kDpsk;
    else                    valid_mod = false;

    return elementwise(mr, EbNo_dB, [=](keel::f64 dB) {
        if (!valid_mod) return keel::Limits<keel::f64>::quiet_nan();
        return keel::comm::ber_awgn(dB, awgn_mod, M);
    });
}

Value noisebw(const Value &num, const Value &den, keel::i32 Nsamp,
              keel::f64 fs, keel::Allocator *mr)
{
    noisebw_argcheck(num, den, Nsamp, fs);
    const keel::usize Mn = num.numel();
    const keel::usize Md = den.numel();
    keel::Array<keel::f64> bn(Mn, mr), an(Md, mr);
    for (keel::usize i = 0; i < Mn; ++i) bn[i] = num.elemAsDouble(i);
    for (keel::usize i = 0; i < Md; ++i) an[i] = den.elemAsDouble(i);
    return Value::scalar(keel::comm::noise_bw(bn.as_ref(), an.as_ref(), Nsamp, fs), mr);
}

keel::Tuple<Value, Value>
berconfint(keel::f64 numErrs, keel::f64 numBits, keel::f64 level,
           keel::Allocator *mr)
{
    berconfint_argcheck(numErrs, numBits, level);

    const keel::f64 k = numErrs;
    const keel::f64 n = numBits;
    const keel::f64 ber = k / n;
    const keel::f64 alpha = 1.0 - level;

    // Clopper-Pearson exact binomial CI.
    const keel::f64 lo = (k == 0.0) ? 0.0
                                    : keel::specfun::betaincinv(alpha / 2.0, k, n - k + 1.0);
    const keel::f64 hi = (k == n)   ? 1.0
                                    : keel::specfun::betaincinv(1.0 - alpha / 2.0, k + 1.0, n - k);

    Value berV = Value::scalar(ber, mr);
    Value ciV  = Value::matrix(1, 2, ValueType::DOUBLE, mr);
    keel::f64 *cd = ciV.doubleDataMut();
    cd[0] = lo;
    cd[1] = hi;
    return {keel::move(berV), keel::move(ciV)};
}

Value convertSNR(const Value &snr_in,
                 keel::StringRef in_type,
                 keel::StringRef out_type,
                 keel::i32 bits_per_symbol,
                 keel::Allocator *mr)
{
    convertSNR_argcheck(snr_in, in_type, out_type, bits_per_symbol);
    if (in_type == out_type) return snr_in;
    return elementwise(mr, snr_in, [&](keel::f64 dB) {
        // Convert via Es/No as the pivot.
        keel::f64 EsNo;
        if      (in_type == "ebno") EsNo = dB + 10.0 * keel::log10(keel::f64(bits_per_symbol));
        else if (in_type == "esno") EsNo = dB;
        else /*snr*/                 EsNo = dB;  // assume Es/No = SNR (1 sample/symbol)
        if      (out_type == "ebno") return EsNo - 10.0 * keel::log10(keel::f64(bits_per_symbol));
        else if (out_type == "esno") return EsNo;
        else                          return EsNo;
    });
}

} // namespace numkit::comm
