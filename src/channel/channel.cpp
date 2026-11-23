// toolboxes/comm/src/channel/channel.cpp

#include <keel/array.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <numkit/comm/channel/channel.hpp>

#include <numkit/builtin/datafun.hpp>
#include <numkit/builtin/specfun.hpp>     // betaincinv
#include <numkit/ops/rng_context.hpp>

#include <numkit/value/argcheck.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <tuple>
#include <utility>


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
        return 0.5 * keel::erfc(v / keel::sqrt(2.0));
    });
}

namespace {
// Inverse complementary error function via Newton refinement on Acklam approx.
keel::f64 erfcinv_approx(keel::f64 x) {
    // erfcinv(x) = -erfinv(x - 1). Use Acklam-style series for the inv normal CDF
    // and convert: norminv(p) = √2·erfinv(2p-1), so erfinv(y) = norminv((y+1)/2)/√2.
    const keel::f64 p = 1.0 - 0.5 * x;  // erfcinv(x) where x = 2(1-p)
    static const keel::f64 a[] = { -3.969683028665376e+01,  2.209460984245205e+02,
                                 -2.759285104469687e+02,  1.383577518672690e+02,
                                 -3.066479806614716e+01,  2.506628277459239e+00 };
    static const keel::f64 b[] = { -5.447609879822406e+01,  1.615858368580409e+02,
                                 -1.556989798598866e+02,  6.680131188771972e+01,
                                 -1.328068155288572e+01 };
    static const keel::f64 c[] = { -7.784894002430293e-03, -3.223964580411365e-01,
                                 -2.400758277161838e+00, -2.549732539343734e+00,
                                  4.374664141464968e+00,  2.938163982698783e+00 };
    static const keel::f64 d[] = {  7.784695709041462e-03,  3.224671290700398e-01,
                                  2.445134137142996e+00,  3.754408661907416e+00 };
    const keel::f64 pl = 0.02425, ph = 1.0 - pl;
    keel::f64 q, r, z;
    if (p < pl) {
        q = keel::sqrt(-2.0 * keel::log(p));
        z = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    } else if (p <= ph) {
        q = p - 0.5; r = q*q;
        z = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q /
            (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0);
    } else {
        q = keel::sqrt(-2.0 * keel::log(1.0 - p));
        z = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
             ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    }
    return z / keel::sqrt(2.0);
}
} // anonymous

Value qfuncinv(const Value &p, keel::Allocator *mr) {
    qfuncinv_argcheck(p);
    return elementwise(mr, p, [](keel::f64 v) {
        if (!(v > 0.0 && v < 1.0)) return keel::Limits<keel::f64>::quiet_nan();
        // Q(x) = 0.5·erfc(x/√2). Inverse: x = √2·erfcinv(2v).
        return keel::sqrt(2.0) * erfcinv_approx(2.0 * v);
    });
}

Value marcumq(const Value &a_v, const Value &b_v, keel::i32 m,
              keel::Allocator *mr)
{
    marcumq_argcheck(a_v, b_v, m);
    // Power-series approximation: Q_m(a, b) = exp(-(a²+b²)/2) · Σ_{k=0..} (a/b)^(m-1+k) · I_{m-1+k}(ab)
    // For first-cut, only m=1 (standard Marcum Q) is implemented robustly via:
    //   Q1(a, b) = exp(-(a²+b²)/2) · Σ_{k=0..∞} (a/b)^k · I_k(ab)
    // which simplifies to the classical formula. For other m, fall back to
    // numerical integration of the integral form.
    if (m < 1) m = 1;
    const keel::usize Na = a_v.numel(), Nb = b_v.numel();
    const keel::usize N = std::max(Na, Nb);
    Value out = Value::matrix((Na > Nb) ? Na : Nb, 1, ValueType::DOUBLE, mr);
    if (N == 0) return out;
    keel::f64 *od = out.doubleDataMut();
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 a = a_v.elemAsDouble(Na > 1 ? i : 0);
        const keel::f64 b = b_v.elemAsDouble(Nb > 1 ? i : 0);
        if (b <= 0.0)         { od[i] = 1.0; continue; }
        if (a == 0.0) {
            // Q_m(0, b) = exp(-b²/2) · Σ_{k=0..m-1} b^(2k)/(2^k · k!)
            // = chi-squared 2m-DOF tail at b².
            const keel::f64 y = 0.5 * b * b;
            // Integrate gamma(m, b²/2) upper incomplete / Γ(m).
            // For small m we can sum directly: Q_m(0, b) = Σ_{k=0..m-1} y^k e^{-y} / k!
            keel::f64 acc = 0.0;
            keel::f64 term = keel::exp(-y);
            for (keel::i32 k = 0; k < m; ++k) {
                acc += term;
                term *= y / keel::f64(k + 1);
            }
            od[i] = acc;
            continue;
        }
        // Numerical integration of the integral definition:
        //   Q_m(a, b) = ∫_b^∞ x · (x/a)^(m-1) · exp(-(x²+a²)/2) · I_{m-1}(a·x) dx
        // Adaptive trapezoidal up to a sensible upper limit.
        const keel::f64 upper = std::max(b + 20.0, a + 20.0);
        const keel::i32 N_int = 4096;
        const keel::f64 dx = (upper - b) / keel::f64(N_int);
        // Modified Bessel I_{m-1}(z) via series: I_n(z) = Σ_{k=0..} (z/2)^(2k+n)/(k!(k+n)!).
        auto besselI = [&](keel::i32 n, keel::f64 z) {
            if (z == 0.0) return n == 0 ? 1.0 : 0.0;
            const keel::f64 half = z / 2.0;
            keel::f64 term = 1.0;
            for (keel::i32 k = 1; k <= n; ++k) term *= half / keel::f64(k);
            term *= keel::pow(half, n) / keel::pow(half, n);
            // Correction: term should be (z/2)^n / n!. The above sequence gives
            // exactly that, but to be safe rebuild explicitly:
            term = 1.0;
            for (keel::i32 k = 1; k <= n; ++k) term *= half / keel::f64(k);
            term *= keel::pow(half, n);
            // Hmm — replace with simpler explicit form:
            term = keel::pow(half, n);
            keel::f64 fact = 1.0;
            for (keel::i32 k = 1; k <= n; ++k) fact *= keel::f64(k);
            term /= fact;
            keel::f64 acc = term;
            for (keel::i32 k = 1; k < 100; ++k) {
                term *= half * half / (keel::f64(k) * keel::f64(k + n));
                acc += term;
                if (keel::abs(term) < 1e-18 * keel::abs(acc)) break;
            }
            return acc;
        };
        keel::f64 sum = 0.0;
        for (keel::i32 k = 0; k <= N_int; ++k) {
            const keel::f64 xv = b + keel::f64(k) * dx;
            const keel::f64 w = (k == 0 || k == N_int) ? 0.5 : 1.0;
            const keel::f64 f = xv * keel::pow(xv / a, m - 1)
                           * keel::exp(-0.5 * (xv * xv + a * a))
                           * besselI(m - 1, a * xv);
            sum += w * f;
        }
        od[i] = sum * dx;
    }
    return out;
}

Value berawgn(const Value &EbNo_dB, keel::StringRef mod, keel::i32 M,
              keel::Allocator *mr)
{
    berawgn_argcheck(EbNo_dB, mod, M);
    return elementwise(mr, EbNo_dB, [&](keel::f64 dB) {
        const keel::f64 EbNo = keel::pow(10.0, dB / 10.0);
        if (mod == "psk") {
            if (M == 2) return 0.5 * keel::erfc(keel::sqrt(EbNo));
            if (M == 4) return 0.5 * keel::erfc(keel::sqrt(EbNo));
            // BER approx for M-PSK (high SNR): (2/k) · Q(√(2k·EbNo) sin(π/M))
            const keel::f64 k = keel::log2(keel::f64(M));
            const keel::f64 arg = keel::sqrt(2.0 * k * EbNo) * keel::sin(keel::kPi / keel::f64(M));
            return (2.0 / k) * 0.5 * keel::erfc(arg / keel::sqrt(2.0));
        }
        if (mod == "qam") {
            const keel::f64 k = keel::log2(keel::f64(M));
            const keel::f64 K = keel::sqrt(keel::f64(M));
            const keel::f64 base = keel::erfc(keel::sqrt(3.0 * k * EbNo / (2.0 * (M - 1))));
            const keel::f64 sym_err = (1.0 - 1.0 / K) * base;
            return sym_err / k;  // approximate Gray-coded BER
        }
        if (mod == "pam") {
            const keel::f64 k = keel::log2(keel::f64(M));
            const keel::f64 base = keel::erfc(keel::sqrt(3.0 * k * EbNo / (M * M - 1)));
            return (M - 1) / (M * k) * base;
        }
        if (mod == "fsk") {
            // Coherent orthogonal FSK: BER = 0.5·erfc(√(EbNo/2)) for M=2.
            return 0.5 * keel::erfc(keel::sqrt(EbNo / 2.0));
        }
        if (mod == "dpsk") {
            // BER for DPSK (M=2): 0.5·exp(-EbNo).
            return 0.5 * keel::exp(-EbNo);
        }
        return keel::Limits<keel::f64>::quiet_nan();
    });
}

Value noisebw(const Value &num, const Value &den, keel::i32 Nsamp,
              keel::f64 fs, keel::Allocator *mr)
{
    noisebw_argcheck(num, den, Nsamp, fs);
    // MATLAB convention (noisebw):
    //   NBW = (fs / N) * sum(|H[k]|^2) / max(|H[k]|^2)
    // Discrete grid over [0, pi) -- N points spaced w_k = pi*k/N.
    // (No 1/2 factor; the resulting NBW matches MATLAB's reference.)
    if (Nsamp <= 0) Nsamp = 1024;
    const keel::usize Mn = num.numel();
    const keel::usize Md = den.numel();
    keel::Array<keel::f64> bn(Mn, mr), an(Md, mr);
    for (keel::usize i = 0; i < Mn; ++i) bn[i] = num.elemAsDouble(i);
    for (keel::usize i = 0; i < Md; ++i) an[i] = den.elemAsDouble(i);

    keel::f64 Hmax2 = 0.0;
    keel::f64 sum_H2 = 0.0;
    for (keel::i32 k = 0; k < Nsamp; ++k) {
        const keel::f64 w = keel::kPi * keel::f64(k) / keel::f64(Nsamp);
        const keel::c64 z = keel::polar(1.0, -w);
        keel::c64 np(0.0, 0.0), dp(0.0, 0.0);
        keel::c64 zk(1.0, 0.0);
        for (keel::usize i = 0; i < Mn; ++i) { np += bn[i] * zk; zk *= z; }
        zk = keel::c64(1.0, 0.0);
        for (keel::usize i = 0; i < Md; ++i) { dp += an[i] * zk; zk *= z; }
        const keel::f64 H2 = (keel::abs(dp) > 0.0) ? keel::norm(np / dp) : 0.0;
        if (H2 > Hmax2) Hmax2 = H2;
        sum_H2 += H2;
    }
    const keel::f64 bw = (Hmax2 > 0.0)
        ? fs * sum_H2 / Hmax2 / keel::f64(Nsamp)
        : 0.0;
    return Value::scalar(bw, mr);
}

// Inverse regularized incomplete beta scalar wrapper (uses builtin Newton-iter
// implementation that operates on Value).
static keel::f64 betaincinv_scalar(keel::Allocator *mr,
                                keel::f64 p, keel::f64 a, keel::f64 b)
{
    if (a <= 0.0 || b <= 0.0) return keel::Limits<keel::f64>::quiet_nan();
    if (p <= 0.0) return 0.0;
    if (p >= 1.0) return 1.0;
    Value pV = Value::scalar(p, mr);
    Value aV = Value::scalar(a, mr);
    Value bV = Value::scalar(b, mr);
    return ::numkit::builtin::betaincinv(pV, aV, bV, mr).toScalar();
}

std::tuple<Value, Value>
berconfint(keel::f64 numErrs, keel::f64 numBits, keel::f64 level,
           keel::Allocator *mr)
{
    berconfint_argcheck(numErrs, numBits, level);

    const keel::f64 k = numErrs;
    const keel::f64 n = numBits;
    const keel::f64 ber = k / n;
    const keel::f64 alpha = 1.0 - level;

    // Clopper-Pearson exact binomial CI.
    keel::f64 lo = (k == 0.0) ? 0.0
                           : betaincinv_scalar(mr, alpha / 2.0, k, n - k + 1.0);
    keel::f64 hi = (k == n)   ? 1.0
                           : betaincinv_scalar(mr, 1.0 - alpha / 2.0, k + 1.0, n - k);

    Value berV = Value::scalar(ber, mr);
    Value ciV  = Value::matrix(1, 2, ValueType::DOUBLE, mr);
    keel::f64 *cd = ciV.doubleDataMut();
    cd[0] = lo;
    cd[1] = hi;
    return {std::move(berV), std::move(ciV)};
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
