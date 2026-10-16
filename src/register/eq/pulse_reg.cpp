// toolboxes/comm/src/eq/pulse_reg.cpp
//
// Register half of the comm pulse-shaping builtins: the CallContext
// wrappers rcosdesign / gaussdesign / rectpulse / intdump that parse args
// and delegate to the engine-free compute in pulse.cpp. library.cpp
// forward-declares + registers these by name.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <numkit/comm/eq/pulse.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, Span, ctx.engine->resource()
#include <numkit/value/value.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>

#include <string>

namespace numkit::comm {
namespace detail {

void rcosdesign_reg(Span<const Value> args, size_t /*nargout*/,
                    Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 4, "rcosdesign");
    const double beta = args[0].toScalar();
    const int span    = argcheck::expectedPositiveInteger(
        argcheck::expectedReal(args[1], 2, "span", "rcosdesign"), 2, "span", "rcosdesign");
    const int sps     = argcheck::expectedPositiveInteger(
        argcheck::expectedReal(args[2], 3, "sps", "rcosdesign"), 3, "sps", "rcosdesign");
    std::string shape = "normal";
    if (args.size() >= 4 && !args[3].isEmpty()) {
        if (!args[3].isChar() && !args[3].isString())
            throw Error("rcosdesign: shape must be a string",
                        0, 0, "rcosdesign", "", "nk:rcosdesign:shape");
        shape = args[3].toString();
    }
    outs[0] = rcosdesign(beta, span, sps, shape, ctx.engine->resource());
}

void gaussdesign_reg(Span<const Value> args, size_t /*nargout*/,
                     Span<Value> outs, CallContext &ctx)
{
    // R2025b defaults: span 3, sps 2 — both optional.
    argcheck::checkNargin(args.size(), 1, 3, "gaussdesign");
    const double BT = args[0].toScalar();
    const int span = (args.size() >= 2 && !args[1].isEmpty())
        ? argcheck::expectedPositiveInteger(
              argcheck::expectedReal(args[1], 2, "span", "gaussdesign"),
              2, "span", "gaussdesign")
        : 3;
    const int sps = (args.size() >= 3 && !args[2].isEmpty())
        ? argcheck::expectedPositiveInteger(
              argcheck::expectedReal(args[2], 3, "sps", "gaussdesign"),
              3, "sps", "gaussdesign")
        : 2;
    outs[0] = gaussdesign(BT, span, sps, ctx.engine->resource());
}

void rectpulse_reg(Span<const Value> args, size_t /*nargout*/,
                   Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "rectpulse");
    const int n = argcheck::expectedPositiveInteger(
        argcheck::expectedReal(args[1], 2, "n", "rectpulse"), 2, "n", "rectpulse");
    outs[0] = rectpulse(args[0], n, ctx.engine->resource());
}

void intdump_reg(Span<const Value> args, size_t /*nargout*/,
                 Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "intdump");
    const int n = argcheck::expectedPositiveInteger(
        argcheck::expectedReal(args[1], 2, "n", "intdump"), 2, "n", "intdump");
    outs[0] = intdump(args[0], n, ctx.engine->resource());
}

} // namespace detail

} // namespace numkit::comm
