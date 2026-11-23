// toolboxes/comm/src/channel/fading.cpp
//
// Frequency-flat Rayleigh / Rician fading channels. iid per-sample —
// no Doppler correlation, no taps; the simplest "flat fade" model
// for first-cut digital-comm sims composing with the cycle-17–20
// modulation primitives.

#include <keel/array.h>
#include <keel/math.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/complex.h>
#include <numkit/comm/channel/fading.hpp>

#include <numkit/builtin/datafun.hpp>
#include <numkit/ops/rng_context.hpp>

#include <numkit/value/argcheck.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>


namespace numkit::comm {

namespace {

// Generate N iid samples of complex Gaussian CN(0, 1), i.e.
//   real, imag ~ N(0, 1/2)  →  E[|h|²] = 1.
keel::Array<keel::c64> sampleCN01(::numkit::ops::RngContext &rng, keel::usize N, keel::Allocator *mr) {
    keel::Array<keel::c64> h(N, mr);
    if (N == 0) return h;
    const keel::f64 s = keel::sqrt(0.5);
    keel::Array<keel::f64> re(N, mr), im(N, mr);
    for (keel::usize i = 0; i < N; ++i) re[i] = s * rng.nextNormal();
    for (keel::usize i = 0; i < N; ++i) im[i] = s * rng.nextNormal();
    for (keel::usize i = 0; i < N; ++i) h[i] = keel::c64(re[i], im[i]);
    return h;
}

// Pull element i of `x` as a complex value (x may be real or complex).
keel::c64 asComplex(const Value &x, keel::usize i) {
    if (x.type() == ValueType::COMPLEX) return x.complexData()[i];
    return keel::c64(x.elemAsDouble(i), 0.0);
}

void rayleighchan_argcheck(const Value &x)
{
    (void)x;
}

void ricianchan_argcheck(const Value &x, keel::f64 K)
{
    (void)x;
    if (K < 0.0)
        throw Error("ricianchan: K-factor must be ≥ 0",
                    0, 0, "ricianchan", "", "numkit:ricianchan:K");
}

} // anonymous

Value rayleighchan(::numkit::ops::RngContext &rng, const Value &x, keel::Allocator *mr)
{
    rayleighchan_argcheck(x);
    const keel::usize N = x.numel();
    Value y = Value::matrix(x.dims().rows(), x.dims().cols(),
                            ValueType::COMPLEX, mr);
    if (N == 0) return y;
    auto h = sampleCN01(rng, N, mr);
    keel::c64 *yd = y.complexDataMut();
    for (keel::usize i = 0; i < N; ++i) yd[i] = h[i] * asComplex(x, i);
    return y;
}

Value ricianchan(::numkit::ops::RngContext &rng, const Value &x, keel::f64 K, keel::Allocator *mr)
{
    ricianchan_argcheck(x, K);
    const keel::usize N = x.numel();
    Value y = Value::matrix(x.dims().rows(), x.dims().cols(),
                            ValueType::COMPLEX, mr);
    if (N == 0) return y;
    auto g = sampleCN01(rng, N, mr);
    const keel::f64 losAmp = keel::sqrt(K / (K + 1.0));
    const keel::f64 sctAmp = keel::sqrt(1.0 / (K + 1.0));
    keel::c64 *yd = y.complexDataMut();
    for (keel::usize i = 0; i < N; ++i) {
        const keel::c64 h = keel::c64(losAmp, 0.0) + sctAmp * g[i];
        yd[i] = h * asComplex(x, i);
    }
    return y;
}

} // namespace numkit::comm
