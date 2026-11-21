// toolboxes/comm/src/modulation/psk.cpp
//
// M-PSK and M-DPSK modulation/demodulation, delegated to keel::comm.

#include <keel/arena.h>
#include <keel/array.h>
#include <keel/comm.h>
#include <keel/complex.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <numkit/comm/modulation/psk.hpp>

#include <numkit/value/argcheck.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

namespace numkit::comm {

namespace {

keel::comm::SymbolOrder parseOrder(keel::StringRef order) {
    return (order == "bin") ? keel::comm::SymbolOrder::kBinary
                            : keel::comm::SymbolOrder::kGray;
}

Value alloc_complex_like(keel::Allocator *mr, const Value &x) {
    const auto &d = x.dims();
    if (x.isScalar()) return Value::matrix(1, 1, ValueType::COMPLEX, mr);
    if (d.is3D())
        return Value::matrix3d(d.rows(), d.cols(), d.pages(), ValueType::COMPLEX, mr);
    return Value::matrix(d.rows(), d.cols(), ValueType::COMPLEX, mr);
}

Value alloc_double_like(keel::Allocator *mr, const Value &x) {
    const auto &d = x.dims();
    if (x.isScalar()) return Value::scalar(0.0, mr);
    if (d.is3D())
        return Value::matrix3d(d.rows(), d.cols(), d.pages(), ValueType::DOUBLE, mr);
    return Value::matrix(d.rows(), d.cols(), ValueType::DOUBLE, mr);
}

void pskmod_argcheck(int M) {
    if (M < 2)
        throw Error("pskmod: M must be ≥ 2", 0, 0, "pskmod", "", "numkit:pskmod:badM");
}

void pskdemod_argcheck(int M) {
    if (M < 2)
        throw Error("pskdemod: M must be ≥ 2", 0, 0, "pskdemod", "", "numkit:pskdemod:badM");
}

void dpskmod_argcheck(int M) {
    if (M < 2)
        throw Error("dpskmod: M must be ≥ 2", 0, 0, "dpskmod", "", "numkit:dpskmod:badM");
}

void dpskdemod_argcheck(int M) {
    if (M < 2)
        throw Error("dpskdemod: M must be ≥ 2", 0, 0, "dpskdemod", "", "numkit:dpskdemod:badM");
}

} // anonymous

Value pskmod(const Value &x, int M, keel::f64 ini_phase,
             keel::StringRef symbol_order, keel::Allocator *mr)
{
    pskmod_argcheck(M);
    Value out = alloc_complex_like(mr, x);
    const keel::usize N = x.numel();
    if (N == 0) return out;
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::i32> syms(N, scratch.allocator());
    for (keel::usize i = 0; i < N; ++i) syms[i] = keel::as<keel::i32>(x.elemAsDouble(i));
    (void)keel::comm::psk_mod(syms, keel::as<keel::i32>(M), ini_phase,
                              parseOrder(symbol_order), {out.complexDataMut(), N});
    return out;
}

Value pskdemod(const Value &y, int M, keel::f64 ini_phase,
               keel::StringRef symbol_order, keel::Allocator *mr)
{
    pskdemod_argcheck(M);
    Value out = alloc_double_like(mr, y);
    const keel::usize N = y.numel();
    if (N == 0) return out;
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::i32> syms(N, scratch.allocator());
    if (y.type() == ValueType::COMPLEX) {
        (void)keel::comm::psk_demod({y.complexData(), N}, keel::as<keel::i32>(M), ini_phase,
                                    parseOrder(symbol_order), syms);
    } else {
        keel::Array<keel::c64> samples(N, scratch.allocator());
        for (keel::usize i = 0; i < N; ++i) samples[i] = keel::c64(y.elemAsDouble(i), 0.0);
        (void)keel::comm::psk_demod(samples, keel::as<keel::i32>(M), ini_phase,
                                    parseOrder(symbol_order), syms);
    }
    keel::f64 *od = out.doubleDataMut();
    for (keel::usize i = 0; i < N; ++i) od[i] = keel::as<keel::f64>(syms[i]);
    return out;
}

Value dpskmod(const Value &x, int M, keel::f64 phase_rot,
              keel::StringRef symbol_order, keel::Allocator *mr)
{
    dpskmod_argcheck(M);
    Value out = alloc_complex_like(mr, x);
    const keel::usize N = x.numel();
    if (N == 0) return out;
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::i32> syms(N, scratch.allocator());
    for (keel::usize i = 0; i < N; ++i) syms[i] = keel::as<keel::i32>(x.elemAsDouble(i));
    (void)keel::comm::dpsk_mod(syms, keel::as<keel::i32>(M), phase_rot,
                               parseOrder(symbol_order), {out.complexDataMut(), N});
    return out;
}

Value dpskdemod(const Value &y, int M, keel::f64 phase_rot,
                keel::StringRef symbol_order, keel::Allocator *mr)
{
    dpskdemod_argcheck(M);
    Value out = alloc_double_like(mr, y);
    const keel::usize N = y.numel();
    if (N == 0) return out;
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::i32> syms(N, scratch.allocator());
    if (y.type() == ValueType::COMPLEX) {
        (void)keel::comm::dpsk_demod({y.complexData(), N}, keel::as<keel::i32>(M), phase_rot,
                                     parseOrder(symbol_order), syms);
    } else {
        keel::Array<keel::c64> samples(N, scratch.allocator());
        for (keel::usize i = 0; i < N; ++i) samples[i] = keel::c64(y.elemAsDouble(i), 0.0);
        (void)keel::comm::dpsk_demod(samples, keel::as<keel::i32>(M), phase_rot,
                                     parseOrder(symbol_order), syms);
    }
    keel::f64 *od = out.doubleDataMut();
    for (keel::usize i = 0; i < N; ++i) od[i] = keel::as<keel::f64>(syms[i]);
    return out;
}

} // namespace numkit::comm
