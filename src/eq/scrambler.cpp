// toolboxes/comm/src/eq/scrambler.cpp
//
// Multiplicative scrambler / descrambler, delegated to keel::comm.

#include <keel/arena.h>
#include <keel/array.h>
#include <keel/comm.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <numkit/comm/eq/scrambler.hpp>

#include <numkit/value/argcheck.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

namespace numkit::comm {

namespace {

keel::Array<keel::u8> readBitVec(const Value &v, keel::Allocator *alloc) {
    const keel::usize N = v.numel();
    keel::Array<keel::u8> out(N, alloc);
    for (keel::usize i = 0; i < N; ++i) {
        out[i] = (v.elemAsDouble(i) != 0.0) ? keel::as<keel::u8>(1) : keel::as<keel::u8>(0);
    }
    return out;
}

void validatePoly(keel::ArrayRef<keel::u8> poly, const char *name) {
    if (poly.size() < 2)
        throw Error(keel::format("{}: polynomial must have at least 2 coefficients (degree ≥ 1)", name),
                    0, 0, name, "", "numkit:scrambler:poly");
    if (poly[0] == 0)
        throw Error(keel::format("{}: polynomial constant term g_0 must be non-zero", name),
                    0, 0, name, "", "numkit:scrambler:poly");
}

Value packDoubles(keel::Allocator *mr, keel::ArrayRef<keel::u8> bits) {
    Value out = Value::matrix(bits.size(), 1, ValueType::DOUBLE, mr);
    if (!bits.empty()) {
        double *od = out.doubleDataMut();
        for (keel::usize i = 0; i < bits.size(); ++i) od[i] = keel::as<double>(bits[i]);
    }
    return out;
}

void scrambler_argcheck(keel::ArrayRef<keel::u8> polyVec,
                        keel::ArrayRef<keel::u8> state,
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
                const Value &initState, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xBits   = readBitVec(x, scratch.allocator());
    auto polyVec = readBitVec(poly, scratch.allocator());
    auto state   = readBitVec(initState, scratch.allocator());
    scrambler_argcheck(polyVec, state, "scrambler", "numkit:scrambler:state");

    keel::Array<keel::u8> y(xBits.size(), 0, scratch.allocator());
    (void)keel::comm::scramble(xBits, polyVec, state, y);
    return packDoubles(mr, y);
}

Value descrambler(const Value &y, const Value &poly,
                  const Value &initState, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto yBits   = readBitVec(y, scratch.allocator());
    auto polyVec = readBitVec(poly, scratch.allocator());
    auto state   = readBitVec(initState, scratch.allocator());
    scrambler_argcheck(polyVec, state, "descrambler", "numkit:descrambler:state");

    keel::Array<keel::u8> x(yBits.size(), 0, scratch.allocator());
    (void)keel::comm::descramble(yBits, polyVec, state, x);
    return packDoubles(mr, x);
}

} // namespace numkit::comm
