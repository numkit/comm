// toolboxes/comm/src/modulation/generic_qam.cpp
//
// Generic constellation modulation/demodulation delegating to keel::comm.

#include <keel/arena.h>
#include <keel/array.h>
#include <keel/comm.h>
#include <keel/complex.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <numkit/comm/modulation/generic_qam.hpp>

#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

namespace numkit::comm {

namespace {

keel::Array<keel::c64> readComplex(const Value &v, keel::Allocator *alloc)
{
    const keel::usize n = v.numel();
    keel::Array<keel::c64> out(n, keel::c64(0.0, 0.0), alloc);
    if (v.isComplex()) {
        const keel::c64 *src = v.complexData();
        for (keel::usize i = 0; i < n; ++i) out[i] = src[i];
    } else {
        for (keel::usize i = 0; i < n; ++i) out[i] = keel::c64(v.elemAsDouble(i), 0.0);
    }
    return out;
}

} // namespace

Value genqammod(const Value &x, const Value &constellation, keel::Allocator *mr)
{
    const keel::usize M = constellation.numel();
    if (M == 0)
        throw Error("genqammod: constellation must be non-empty",
                    0, 0, "genqammod", "", "numkit:genqammod:EmptyConst");

    const keel::usize N = x.numel();
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::i32> syms(N, 0, scratch.allocator());
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 xi = x.elemAsDouble(i);
        if (xi < 0.0 || xi >= keel::as<keel::f64>(M))
            throw Error("genqammod: index out of range [0, M-1]",
                        0, 0, "genqammod", "", "numkit:genqammod:OutOfRange");
        syms[i] = keel::as<keel::i32>(xi);
    }

    const auto C = readComplex(constellation, scratch.allocator());
    const bool out_complex = constellation.isComplex();
    Value out = Value::matrix(x.dims().rows(), x.dims().cols(),
                              out_complex ? ValueType::COMPLEX : ValueType::DOUBLE, mr);
    if (out_complex) {
        (void)keel::comm::genqam_mod(syms.as_ref(), C.as_ref(),
                                     keel::MutArrayRef<keel::c64>(out.complexDataMut(), N));
    } else {
        keel::Array<keel::c64> tmp(N, keel::c64(0.0, 0.0), scratch.allocator());
        (void)keel::comm::genqam_mod(syms.as_ref(), C.as_ref(), tmp.as_mut_ref());
        keel::f64 *o = out.doubleDataMut();
        for (keel::usize i = 0; i < N; ++i) o[i] = tmp[i].real();
    }
    return out;
}

Value genqamdemod(const Value &y, const Value &constellation, keel::Allocator *mr)
{
    if (constellation.numel() == 0)
        throw Error("genqamdemod: constellation must be non-empty",
                    0, 0, "genqamdemod", "", "numkit:genqamdemod:EmptyConst");

    const keel::usize N = y.numel();
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const auto C = readComplex(constellation, scratch.allocator());
    const auto samples = readComplex(y, scratch.allocator());

    keel::Array<keel::i32> idx(N, 0, scratch.allocator());
    (void)keel::comm::genqam_demod(samples.as_ref(), C.as_ref(), idx.as_mut_ref());

    Value out = Value::matrix(y.dims().rows(), y.dims().cols(), ValueType::DOUBLE, mr);
    keel::f64 *o = out.doubleDataMut();
    for (keel::usize i = 0; i < N; ++i) o[i] = keel::as<keel::f64>(idx[i]);
    return out;
}

} // namespace numkit::comm
