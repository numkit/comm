// toolboxes/comm/src/modulation/fsk_ofdm_reg.cpp
//
// Register half of the comm FSK/OFDM builtins: the CallContext wrappers
// fskmod / fskdemod / ofdmmod / ofdmdemod that parse the phase-continuity /
// symbol-order / symoffset options and delegate to the engine-free compute
// in fsk_ofdm.cpp. library.cpp forward-declares + registers these by name.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <numkit/comm/modulation/fsk_ofdm.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, Span, ctx.engine->resource()
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>

#include <string>

namespace numkit::comm {
namespace detail {

namespace {
std::string parse_str(Span<const Value> args, size_t start, const std::string &def) {
    for (size_t i = start; i < args.size(); ++i)
        if (args[i].isChar() || args[i].isString()) return args[i].toString();
    return def;
}
}

void fskmod_reg(Span<const Value> args, size_t /*nargout*/,
                Span<Value> outs, CallContext &ctx)
{
    // R2025b: fs optional, default 1 (4-arg form probed equal to fs=1).
    argcheck::checkNargin(args.size(), 4, 7, "fskmod");
    const int    M   = (int)args[1].toScalar();
    const double sep = args[2].toScalar();
    const int    n   = (int)args[3].toScalar();
    const double fs  = (args.size() >= 5 && !args[4].isEmpty())
                       ? args[4].toScalar() : 1.0;
    auto cont  = parse_str(args, 5, "cont");
    auto order = parse_str(args, 5, "gray");
    // If first string is recognised as 'gray'/'bin', that's the symbol order.
    for (size_t i = 5; i < args.size(); ++i) {
        if (args[i].isChar() || args[i].isString()) {
            auto s = args[i].toString();
            if (s == "gray" || s == "bin") order = s;
            else if (s == "cont" || s == "discont") cont = s;
        }
    }
    outs[0] = fskmod(args[0], M, sep, n, fs, cont, order, ctx.engine->resource());
}

void fskdemod_reg(Span<const Value> args, size_t /*nargout*/,
                  Span<Value> outs, CallContext &ctx)
{
    // R2025b: fs optional, default 1 (4-arg form probed equal to fs=1).
    argcheck::checkNargin(args.size(), 4, 6, "fskdemod");
    const int    M   = (int)args[1].toScalar();
    const double sep = args[2].toScalar();
    const int    n   = (int)args[3].toScalar();
    const double fs  = (args.size() >= 5 && !args[4].isEmpty())
                       ? args[4].toScalar() : 1.0;
    auto order = parse_str(args, 5, "gray");
    outs[0] = fskdemod(args[0], M, sep, n, fs, order, ctx.engine->resource());
}

void ofdmmod_reg(Span<const Value> args, size_t /*nargout*/,
                 Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 8, "ofdmmod");
    const int nfft  = (int)args[1].toScalar();
    const int cplen = (int)args[2].toScalar();
    outs[0] = ofdmmod(args[0], nfft, cplen, ctx.engine->resource());
}

void ofdmdemod_reg(Span<const Value> args, size_t /*nargout*/,
                   Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 8, "ofdmdemod");
    const int nfft  = (int)args[1].toScalar();
    const int cplen = (int)args[2].toScalar();
    const int sym   = (args.size() >= 4 && !args[3].isEmpty())
                      ? (int)args[3].toScalar() : cplen;
    outs[0] = ofdmdemod(args[0], nfft, cplen, sym, ctx.engine->resource());
}

} // namespace detail
} // namespace numkit::comm
