// toolboxes/comm/src/modulation/psk_reg.cpp
//
// Register half of the comm PSK builtins: the CallContext wrappers
// pskmod / pskdemod / dpskmod / dpskdemod that parse the optional
// ini_phase + symbol-order args and delegate to the engine-free compute in
// psk.cpp. library.cpp forward-declares + registers these by name.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <keel/core.h>
#include <keel/ref.h>
#include <numkit/comm/modulation/psk.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, Span, ctx.engine->resource()
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>

#include <string>

namespace numkit::comm {
namespace detail {

namespace {
std::string parse_order(keel::ArrayRef<Value> args, keel::usize i, const std::string &def) {
    if (i >= args.size()) return def;
    if (!(args[i].isChar() || args[i].isString())) return def;
    auto s = args[i].toString();
    if (s == "bin" || s == "gray") return std::string(s.as_ref());
    return def;
}
}

void pskmod_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 10, "pskmod");
    const int M = (int)args[1].toScalar();
    const double ini = (args.size() >= 3 && !args[2].isEmpty()
                        && !(args[2].isChar() || args[2].isString()))
                        ? args[2].toScalar() : 0.0;
    auto order = parse_order(args, args.size() >= 4 ? 3 : 2, "gray");
    outs[0] = pskmod(args[0], M, ini, order, ctx.engine->allocator());
}

void pskdemod_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "pskdemod");  // no max: name-value tail in the reference (no count wall to 12 args)
    const int M = (int)args[1].toScalar();
    const double ini = (args.size() >= 3 && !args[2].isEmpty()
                        && !(args[2].isChar() || args[2].isString()))
                        ? args[2].toScalar() : 0.0;
    auto order = parse_order(args, args.size() >= 4 ? 3 : 2, "gray");
    outs[0] = pskdemod(args[0], M, ini, order, ctx.engine->allocator());
}

void dpskmod_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "dpskmod");  // no max: name-value tail in the reference (no count wall to 12 args)
    const int M = (int)args[1].toScalar();
    const double rot = (args.size() >= 3 && !args[2].isEmpty()
                        && !(args[2].isChar() || args[2].isString()))
                        ? args[2].toScalar() : 0.0;
    auto order = parse_order(args, args.size() >= 4 ? 3 : 2, "gray");
    outs[0] = dpskmod(args[0], M, rot, order, ctx.engine->allocator());
}

void dpskdemod_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                   keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "dpskdemod");  // no max: name-value tail in the reference (no count wall to 12 args)
    const int M = (int)args[1].toScalar();
    const double rot = (args.size() >= 3 && !args[2].isEmpty()
                        && !(args[2].isChar() || args[2].isString()))
                        ? args[2].toScalar() : 0.0;
    auto order = parse_order(args, args.size() >= 4 ? 3 : 2, "gray");
    outs[0] = dpskdemod(args[0], M, rot, order, ctx.engine->allocator());
}

} // namespace detail
} // namespace numkit::comm
