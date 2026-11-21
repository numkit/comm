// toolboxes/comm/src/modulation/generic_qam.cpp
//
// Generic constellation modulation/demodulation:
//   y = genqammod(x, constellation)   -- index -> constellation point
//   x = genqamdemod(y, constellation) -- nearest-point demodulation
//
// Default 'integer' input mode only. Bit-input mode
// (`'InputType','bit'`) is documented as deferred -- requires
// bit-grouping by log2(M) which we'll add when called for.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/complex.h>
#include <numkit/comm/modulation/generic_qam.hpp>

#include <numkit/value/argcheck.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <limits>

namespace numkit::comm {

// Read constellation point k as Cd.
static keel::c64 constAt(const Value &c, keel::usize k) {
    if (c.isComplex())
        return c.complexData()[k];
    return keel::c64(c.elemAsDouble(k), 0.0);
}

namespace {

void genqammod_argcheck(const Value &x, const Value &constellation)
{
    const keel::usize M = constellation.numel();
    if (M == 0)
        throw Error("genqammod: constellation must be non-empty",
                    0, 0, "genqammod", "", "numkit:genqammod:EmptyConst");
    const keel::usize N = x.numel();
    for (keel::usize i = 0; i < N; ++i) {
        const double xi = x.elemAsDouble(i);
        if (xi < 0.0 || xi >= keel::as<double>(M))
            throw Error("genqammod: index out of range [0, M-1]",
                        0, 0, "genqammod", "",
                        "numkit:genqammod:OutOfRange");
    }
}

void genqamdemod_argcheck(const Value &constellation)
{
    if (constellation.numel() == 0)
        throw Error("genqamdemod: constellation must be non-empty",
                    0, 0, "genqamdemod", "", "numkit:genqamdemod:EmptyConst");
}

} // namespace

Value genqammod(const Value &x, const Value &constellation,
                keel::Allocator *mr)
{
    genqammod_argcheck(x, constellation);

    const auto &d = x.dims();
    const keel::usize H = d.rows();
    const keel::usize W = d.cols();
    const keel::usize N = H * W;

    // Output type: complex iff constellation is complex.
    const bool out_complex = constellation.isComplex();
    Value out = Value::matrix(H, W,
                              out_complex ? ValueType::COMPLEX
                                          : ValueType::DOUBLE,
                              mr);

    if (out_complex) {
        keel::c64 *o = out.complexDataMut();
        for (keel::usize i = 0; i < N; ++i) {
            const keel::usize k = keel::as<keel::usize>(x.elemAsDouble(i));
            o[i] = constellation.complexData()[k];
        }
    } else {
        double *o = out.doubleDataMut();
        for (keel::usize i = 0; i < N; ++i) {
            const keel::usize k = keel::as<keel::usize>(x.elemAsDouble(i));
            o[i] = constellation.elemAsDouble(k);
        }
    }
    return out;
}

Value genqamdemod(const Value &y, const Value &constellation,
                  keel::Allocator *mr)
{
    genqamdemod_argcheck(constellation);
    const keel::usize M = constellation.numel();

    const auto &d = y.dims();
    const keel::usize H = d.rows();
    const keel::usize W = d.cols();
    const keel::usize N = H * W;

    Value out = Value::matrix(H, W, ValueType::DOUBLE, mr);
    double *o = out.doubleDataMut();
    const bool y_complex = y.isComplex();

    for (keel::usize i = 0; i < N; ++i) {
        const keel::c64 yi = y_complex
                          ? y.complexData()[i]
                          : keel::c64(y.elemAsDouble(i), 0.0);
        keel::usize best_k = 0;
        double best_d2 = std::numeric_limits<double>::infinity();
        for (keel::usize k = 0; k < M; ++k) {
            const keel::c64 ck = constAt(constellation, k);
            const double dr = yi.real() - ck.real();
            const double di = yi.imag() - ck.imag();
            const double d2 = dr * dr + di * di;
            if (d2 < best_d2) {
                best_d2 = d2;
                best_k  = k;
            }
        }
        o[i] = keel::as<double>(best_k);
    }
    return out;
}

} // namespace numkit::comm
