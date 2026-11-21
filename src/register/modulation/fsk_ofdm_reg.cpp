// toolboxes/comm/src/modulation/fsk_ofdm_reg.cpp
//
// Register half of the comm FSK/OFDM builtins: the CallContext wrappers
// fskmod / fskdemod / ofdmmod / ofdmdemod that parse the phase-continuity /
// symbol-order / symoffset options and delegate to the engine-free compute
// in fsk_ofdm.cpp. library.cpp forward-declares + registers these by name.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <keel/core.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <numkit/comm/modulation/fsk_ofdm.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, Span, ctx.engine->resource()
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>

namespace numkit::comm {
namespace detail {

namespace {
keel::String parse_str(keel::ArrayRef<Value> args, keel::usize start, keel::StringRef def) {
    for (keel::usize i = start; i < args.size(); ++i)
        if (args[i].isChar() || args[i].isString()) return args[i].toString();
    return keel::String(def);
}
}

void fskmod_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    // R2025b: fs optional, default 1 (4-arg form probed equal to fs=1).
    argcheck::checkNargin(args.size(), 4, 7, "fskmod");
    const keel::i32    M   = (keel::i32)args[1].toScalar();
    const keel::f64 sep = args[2].toScalar();
    const keel::i32    n   = (keel::i32)args[3].toScalar();
    const keel::f64 fs  = (args.size() >= 5 && !args[4].isEmpty())
                       ? args[4].toScalar() : 1.0;
    auto cont  = parse_str(args, 5, "cont");
    auto order = parse_str(args, 5, "gray");
    // If first string is recognised as 'gray'/'bin', that's the symbol order.
    for (keel::usize i = 5; i < args.size(); ++i) {
        if (args[i].isChar() || args[i].isString()) {
            const keel::String s = args[i].toString();
            if (s == "gray" || s == "bin") order.assign(s);
            else if (s == "cont" || s == "discont") cont.assign(s);
        }
    }
    outs[0] = fskmod(args[0], M, sep, n, fs, cont, order, ctx.engine->allocator());
}

void fskdemod_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    // R2025b: fs optional, default 1 (4-arg form probed equal to fs=1).
    argcheck::checkNargin(args.size(), 4, 6, "fskdemod");
    const keel::i32    M   = (keel::i32)args[1].toScalar();
    const keel::f64 sep = args[2].toScalar();
    const keel::i32    n   = (keel::i32)args[3].toScalar();
    const keel::f64 fs  = (args.size() >= 5 && !args[4].isEmpty())
                       ? args[4].toScalar() : 1.0;
    auto order = parse_str(args, 5, "gray");
    outs[0] = fskdemod(args[0], M, sep, n, fs, order, ctx.engine->allocator());
}

void ofdmmod_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 8, "ofdmmod");
    const keel::i32 nfft  = (keel::i32)args[1].toScalar();
    const keel::i32 cplen = (keel::i32)args[2].toScalar();
    outs[0] = ofdmmod(args[0], nfft, cplen, ctx.engine->allocator());
}

void ofdmdemod_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                   keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 8, "ofdmdemod");
    const keel::i32 nfft  = (keel::i32)args[1].toScalar();
    const keel::i32 cplen = (keel::i32)args[2].toScalar();
    const keel::i32 sym   = (args.size() >= 4 && !args[3].isEmpty())
                      ? (keel::i32)args[3].toScalar() : cplen;
    outs[0] = ofdmdemod(args[0], nfft, cplen, sym, ctx.engine->allocator());
}

} // namespace detail
} // namespace numkit::comm
