// toolboxes/comm/src/eq/scrambler.cpp
//
// Multiplicative scrambler / descrambler. Both share the same
// polynomial and initial state; the descrambler is the algebraic
// inverse of the scrambler.
//
// Multiplicative scrambler (per-bit):
//   fb = XOR over i=1..n where g_i = 1 of state[i-1]
//   y[k] = x[k] XOR fb
//   shift state right by 1, store y[k] at state[0]
//
// Multiplicative descrambler:
//   fb = XOR over i=1..n where g_i = 1 of state[i-1]
//   x[k] = y[k] XOR fb
//   shift state right by 1, store y[k] at state[0]   (NOT x[k])
//
// Note both sides clock in the *channel* bit y[k]: this is what makes
// the descrambler self-synchronizing — losing or flipping a few
// channel bits eventually clears out of the register.

#include <keel/array.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <numkit/comm/eq/scrambler.hpp>

#include <numkit/value/argcheck.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>

namespace numkit::comm {

namespace {

keel::Array<int> readBitVec(const Value &v, const char * /*name*/, keel::Allocator *mr) {
    const keel::usize N = v.numel();
    keel::Array<int> out(N, mr);
    for (keel::usize i = 0; i < N; ++i) {
        const double x = v.elemAsDouble(i);
        const int b = (x != 0.0) ? 1 : 0;
        out[i] = b;
    }
    return out;
}

void validatePoly(keel::ArrayRef<int> poly, const char *name) {
    if (poly.size() < 2)
        throw Error(keel::format("{}: polynomial must have at least 2 coefficients (degree ≥ 1)", name),
                    0, 0, name, "", "numkit:scrambler:poly");
    if (poly[0] == 0)
        throw Error(keel::format("{}: polynomial constant term g_0 must be non-zero", name),
                    0, 0, name, "", "numkit:scrambler:poly");
}

Value packDoubles(keel::Allocator *mr,
                  keel::ArrayRef<int> bits)
{
    Value out = Value::matrix(bits.size(), 1, ValueType::DOUBLE, mr);
    if (!bits.empty()) {
        double *od = out.doubleDataMut();
        for (keel::usize i = 0; i < bits.size(); ++i) od[i] = double(bits[i]);
    }
    return out;
}

void scrambler_argcheck(keel::ArrayRef<int> polyVec,
                        keel::ArrayRef<int> state,
                        const char *name,
                        const char *stateErrId)
{
    validatePoly(polyVec, name);
    const keel::usize n = polyVec.size() - 1;
    if (state.size() != n)
        throw Error(keel::format("{}: initState length must equal polynomial order", name),
                    0, 0, name, "", stateErrId);
}

} // anonymous

Value scrambler(const Value &x, const Value &poly,
                const Value &initState,
                keel::Allocator *mr)
{
    auto xBits   = readBitVec(x, "scrambler", mr);
    auto polyVec = readBitVec(poly, "scrambler", mr);
    auto state   = readBitVec(initState, "scrambler", mr);
    scrambler_argcheck(polyVec, state, "scrambler", "numkit:scrambler:state");
    const keel::usize n = polyVec.size() - 1;

    keel::Array<int> y(xBits.size(), 0, mr);
    for (keel::usize k = 0; k < xBits.size(); ++k) {
        int fb = 0;
        for (keel::usize i = 1; i <= n; ++i)
            if (polyVec[i]) fb ^= state[i - 1];
        y[k] = xBits[k] ^ fb;
        // Shift register right by one position; new MSB = y[k].
        for (keel::usize i = n; i-- > 1;) state[i] = state[i - 1];
        state[0] = y[k];
    }
    return packDoubles(mr, y);
}

Value descrambler(const Value &y, const Value &poly,
                  const Value &initState,
                  keel::Allocator *mr)
{
    auto yBits   = readBitVec(y, "descrambler", mr);
    auto polyVec = readBitVec(poly, "descrambler", mr);
    auto state   = readBitVec(initState, "descrambler", mr);
    scrambler_argcheck(polyVec, state, "descrambler", "numkit:descrambler:state");
    const keel::usize n = polyVec.size() - 1;

    keel::Array<int> x(yBits.size(), 0, mr);
    for (keel::usize k = 0; k < yBits.size(); ++k) {
        int fb = 0;
        for (keel::usize i = 1; i <= n; ++i)
            if (polyVec[i]) fb ^= state[i - 1];
        x[k] = yBits[k] ^ fb;
        // Shift register; clock in the CHANNEL bit y[k] (not x[k]).
        for (keel::usize i = n; i-- > 1;) state[i] = state[i - 1];
        state[0] = yBits[k];
    }
    return packDoubles(mr, x);
}

} // namespace numkit::comm
