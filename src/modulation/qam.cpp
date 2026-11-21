// toolboxes/comm/src/modulation/qam.cpp
//
// M-PAM, rectangular M-QAM, and modnorm, delegated to keel::comm.

#include <keel/arena.h>
#include <keel/array.h>
#include <keel/comm.h>
#include <keel/complex.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <numkit/comm/modulation/qam.hpp>

#include <numkit/value/argcheck.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

namespace numkit::comm {

namespace {

keel::comm::SymbolOrder parseOrder(keel::StringRef order) {
    return (order == "bin") ? keel::comm::SymbolOrder::kBinary
                            : keel::comm::SymbolOrder::kGray;
}

Value alloc_double_like(keel::Allocator *mr, const Value &x) {
    const auto &d = x.dims();
    if (x.isScalar()) return Value::scalar(0.0, mr);
    if (d.is3D())
        return Value::matrix3d(d.rows(), d.cols(), d.pages(), ValueType::DOUBLE, mr);
    return Value::matrix(d.rows(), d.cols(), ValueType::DOUBLE, mr);
}

Value alloc_complex_like(keel::Allocator *mr, const Value &x) {
    const auto &d = x.dims();
    if (x.isScalar()) return Value::matrix(1, 1, ValueType::COMPLEX, mr);
    if (d.is3D())
        return Value::matrix3d(d.rows(), d.cols(), d.pages(), ValueType::COMPLEX, mr);
    return Value::matrix(d.rows(), d.cols(), ValueType::COMPLEX, mr);
}

void pammod_argcheck(int M) {
    if (M < 2)
        throw Error("pammod: M must be ≥ 2", 0, 0, "pammod", "", "numkit:pammod:badM");
}

void pamdemod_argcheck(int M) {
    if (M < 2)
        throw Error("pamdemod: M must be ≥ 2", 0, 0, "pamdemod", "", "numkit:pamdemod:badM");
}

void qammod_argcheck(int M) {
    if (M < 4)
        throw Error("qammod: M must be ≥ 4", 0, 0, "qammod", "", "numkit:qammod:badM");
}

void qamdemod_argcheck(int M) {
    if (M < 4)
        throw Error("qamdemod: M must be ≥ 4", 0, 0, "qamdemod", "", "numkit:qamdemod:badM");
}

void modnorm_argcheck(const Value &ref, keel::StringRef type, double target) {
    (void)ref;
    (void)type;
    (void)target;
}

} // anonymous

Value pammod(const Value &x, int M, double ini_phase,
             keel::StringRef symbol_order, keel::Allocator *mr)
{
    pammod_argcheck(M);
    Value out = alloc_complex_like(mr, x);
    const keel::usize N = x.numel();
    if (N == 0) return out;
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::i32> syms(N, scratch.allocator());
    for (keel::usize i = 0; i < N; ++i) syms[i] = static_cast<keel::i32>(x.elemAsDouble(i));
    (void)keel::comm::pam_mod(syms, static_cast<keel::i32>(M), ini_phase,
                              parseOrder(symbol_order), {out.complexDataMut(), N});
    return out;
}

Value pamdemod(const Value &y, int M, double ini_phase,
               keel::StringRef symbol_order, keel::Allocator *mr)
{
    pamdemod_argcheck(M);
    Value out = alloc_double_like(mr, y);
    const keel::usize N = y.numel();
    if (N == 0) return out;
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::i32> syms(N, scratch.allocator());
    if (y.type() == ValueType::COMPLEX) {
        (void)keel::comm::pam_demod({y.complexData(), N}, static_cast<keel::i32>(M), ini_phase,
                                    parseOrder(symbol_order), syms);
    } else {
        keel::Array<keel::c64> samples(N, scratch.allocator());
        for (keel::usize i = 0; i < N; ++i) samples[i] = keel::c64(y.elemAsDouble(i), 0.0);
        (void)keel::comm::pam_demod(samples, static_cast<keel::i32>(M), ini_phase,
                                    parseOrder(symbol_order), syms);
    }
    double *od = out.doubleDataMut();
    for (keel::usize i = 0; i < N; ++i) od[i] = static_cast<double>(syms[i]);
    return out;
}

Value qammod(const Value &x, int M, keel::StringRef symbol_order,
             bool unit_power, keel::Allocator *mr)
{
    qammod_argcheck(M);
    Value out = alloc_complex_like(mr, x);
    const keel::usize N = x.numel();
    if (N == 0) return out;
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::i32> syms(N, scratch.allocator());
    for (keel::usize i = 0; i < N; ++i) syms[i] = static_cast<keel::i32>(x.elemAsDouble(i));
    (void)keel::comm::qam_mod(syms, static_cast<keel::i32>(M), parseOrder(symbol_order),
                              unit_power, {out.complexDataMut(), N});
    return out;
}

Value qamdemod(const Value &y, int M, keel::StringRef symbol_order,
               bool unit_power, keel::Allocator *mr)
{
    qamdemod_argcheck(M);
    Value out = alloc_double_like(mr, y);
    const keel::usize N = y.numel();
    if (N == 0) return out;
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::i32> syms(N, scratch.allocator());
    if (y.type() == ValueType::COMPLEX) {
        (void)keel::comm::qam_demod({y.complexData(), N}, static_cast<keel::i32>(M),
                                    parseOrder(symbol_order), unit_power, syms);
    } else {
        keel::Array<keel::c64> samples(N, scratch.allocator());
        for (keel::usize i = 0; i < N; ++i) samples[i] = keel::c64(y.elemAsDouble(i), 0.0);
        (void)keel::comm::qam_demod(samples, static_cast<keel::i32>(M),
                                    parseOrder(symbol_order), unit_power, syms);
    }
    double *od = out.doubleDataMut();
    for (keel::usize i = 0; i < N; ++i) od[i] = static_cast<double>(syms[i]);
    return out;
}

Value modnorm(const Value &ref, keel::StringRef type, double target, keel::Allocator *mr)
{
    modnorm_argcheck(ref, type, target);
    const keel::usize N = ref.numel();
    if (N == 0) return Value::scalar(1.0, mr);
    const auto mode = (type == "avpow") ? keel::comm::NormMode::kAveragePower
                                        : keel::comm::NormMode::kPeakPower;
    if (ref.type() == ValueType::COMPLEX) {
        return Value::scalar(keel::comm::modnorm({ref.complexData(), N}, mode, target), mr);
    }
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::c64> c(N, scratch.allocator());
    for (keel::usize i = 0; i < N; ++i) c[i] = keel::c64(ref.elemAsDouble(i), 0.0);
    return Value::scalar(keel::comm::modnorm(c, mode, target), mr);
}

} // namespace numkit::comm
