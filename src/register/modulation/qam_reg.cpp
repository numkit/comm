// toolboxes/comm/src/modulation/qam_reg.cpp
//
// Register half of the comm PAM/QAM builtins: the CallContext wrappers
// pammod / pamdemod / qammod / qamdemod / modnorm that parse the symbol-
// order / UnitAveragePower options and delegate to the engine-free compute
// in qam.cpp. library.cpp forward-declares + registers these by name.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <keel/core.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <numkit/comm/modulation/qam.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, Span, ctx.engine->resource()
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>

namespace numkit::comm {
namespace detail {

namespace {
// MATLAB R2025b defaults differ by function: pammod/pamdemod default to
// 'bin' (binary symbol mapping); qammod/qamdemod default to 'gray'.
keel::String parse_order(keel::ArrayRef<Value> args, keel::usize start, const char *dflt) {
    for (keel::usize i = start; i < args.size(); ++i) {
        if (args[i].isChar() || args[i].isString()) {
            keel::String s = args[i].toString();
            if (s == "bin" || s == "gray") return s;
        }
    }
    return keel::String(dflt);
}

bool parse_unit_power(keel::ArrayRef<Value> args, keel::usize start) {
    for (keel::usize i = start; i + 1 < args.size(); ++i) {
        if ((args[i].isChar() || args[i].isString())
            && args[i].toString() == "UnitAveragePower")
            return args[i + 1].toScalar() != 0.0;
    }
    return false;
}
}

void pammod_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 4, "pammod");
    const int M = (int)args[1].toScalar();
    const keel::f64 ini = (args.size() >= 3 && !args[2].isEmpty()
                        && !(args[2].isChar() || args[2].isString()))
                        ? args[2].toScalar() : 0.0;
    auto order = parse_order(args, 2, "bin");
    outs[0] = pammod(args[0], M, ini, order, ctx.engine->allocator());
}

void pamdemod_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 4, "pamdemod");
    const int M = (int)args[1].toScalar();
    const keel::f64 ini = (args.size() >= 3 && !args[2].isEmpty()
                        && !(args[2].isChar() || args[2].isString()))
                        ? args[2].toScalar() : 0.0;
    auto order = parse_order(args, 2, "bin");
    outs[0] = pamdemod(args[0], M, ini, order, ctx.engine->allocator());
}

void qammod_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 11, "qammod");
    const int M = (int)args[1].toScalar();
    auto order = parse_order(args, 2, "gray");
    bool up = parse_unit_power(args, 2);
    outs[0] = qammod(args[0], M, order, up, ctx.engine->allocator());
}

void qamdemod_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "qamdemod");  // no max: name-value tail in the reference (no count wall to 12 args)
    const int M = (int)args[1].toScalar();
    auto order = parse_order(args, 2, "gray");
    bool up = parse_unit_power(args, 2);
    outs[0] = qamdemod(args[0], M, order, up, ctx.engine->allocator());
}

void modnorm_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 3, "modnorm");
    keel::String type("avpow");
    if (args[1].isChar() || args[1].isString()) type = args[1].toString();
    const keel::f64 target = args[2].toScalar();
    outs[0] = modnorm(args[0], type, target, ctx.engine->allocator());
}

} // namespace detail
} // namespace numkit::comm
