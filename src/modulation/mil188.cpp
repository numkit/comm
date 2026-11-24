// toolboxes/comm/src/modulation/mil188.cpp
//
// MIL-STD-188-110B/C QAM constellation modulation/demodulation.
//
//   y = mil188qammod(x, M)
//   z = mil188qamdemod(y, M)
//
// Accepts M in {16, 32, 64, 256}, delegating the constellation tables and
// modulation/demodulation kernels to keel::comm.

#include <keel/array.h>
#include <keel/comm.h>
#include <keel/complex.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <numkit/comm/modulation/mil188.hpp>

#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

namespace numkit::comm {

namespace {

// <fn>_argcheck for the MIL-STD QAM pair: the M-set contract lives
// here so the interpreter and direct C++ calls answer with ONE
// identifier per function (bugs/opened/lang/reg-domain-logic.md).
void mil188MSetArgcheck(keel::i32 M, keel::StringRef fn)
{
    argcheck::expectedOneOf(keel::as<keel::f64>(M), {16, 32, 64, 256},
                            2, "M", fn);
}

} // namespace

Value mil188qammod(const Value &x, keel::i32 M, keel::Allocator *mr)
{
    mil188MSetArgcheck(M, "mil188qammod");

    const keel::usize Nx = x.numel();
    keel::Array<keel::i32> syms(Nx, mr);
    for (keel::usize i = 0; i < Nx; ++i) {
        const keel::f64 xi = x.elemAsDouble(i);
        if (xi < 0.0 || xi >= keel::as<keel::f64>(M))
            throw Error("mil188qammod: input out of range [0, M-1]",
                        0, 0, "mil188qammod", "",
                        "numkit:mil188qammod:OutOfRange");
        syms[i] = keel::as<keel::i32>(xi);
    }

    Value out = Value::matrix(x.dims().rows(), x.dims().cols(),
                              ValueType::COMPLEX, mr);
    (void)keel::comm::mil188_qam_mod(
        syms.as_ref(), M, keel::MutArrayRef<keel::c64>(out.complexDataMut(), Nx));
    return out;
}

Value mil188qamdemod(const Value &y, keel::i32 M, keel::Allocator *mr)
{
    mil188MSetArgcheck(M, "mil188qamdemod");

    const keel::usize Ny = y.numel();
    keel::Array<keel::i32> syms(Ny, mr);
    if (y.isComplex()) {
        (void)keel::comm::mil188_qam_demod(
            keel::ArrayRef<keel::c64>(y.complexData(), Ny), M, syms.as_mut_ref());
    } else {
        keel::Array<keel::c64> samples(Ny, mr);
        for (keel::usize i = 0; i < Ny; ++i) {
            samples[i] = keel::c64(y.elemAsDouble(i), 0.0);
        }
        (void)keel::comm::mil188_qam_demod(samples.as_ref(), M, syms.as_mut_ref());
    }

    Value out = Value::matrix(y.dims().rows(), y.dims().cols(),
                              ValueType::DOUBLE, mr);
    keel::f64 *o = out.doubleDataMut();
    for (keel::usize i = 0; i < Ny; ++i) {
        o[i] = keel::as<keel::f64>(syms[i]);
    }
    return out;
}

} // namespace numkit::comm
