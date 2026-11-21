// toolboxes/comm/src/channel/channel_reg.cpp
//
// Register half of the comm channel/metric builtins: the CallContext
// wrappers awgn / wgn / bsc / qfunc / qfuncinv / marcumq / berawgn /
// noisebw / berconfint / convertSNR that parse args and delegate to the
// engine-free compute in channel.cpp. library.cpp forward-declares +
// registers these by name.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <keel/core.h>
#include <keel/format.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <numkit/comm/channel/channel.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, Span, ctx.engine->resource()
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>

#include <utility>

namespace numkit::comm {
namespace detail {

void awgn_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
              keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 5, "awgn");
    // Marshal-only: voiced extraction; the 'measured' sentinel semantics
    // and the compute-side behaviour live in channel.cpp's awgn.
    const keel::f64 snr = argcheck::expectedReal(args[1], 2, "snr", "awgn");
    keel::f64 sp = 0.0;  // default 0 dBW (1 W linear); "measured" uses -1e10 sentinel
    if (args.size() >= 3 && !args[2].isEmpty()) {
        if (args[2].isChar() || args[2].isString()) sp = -1e10;
    }
    if (args.size() >= 3 && !args[2].isEmpty()) {
        if (args[2].isChar() || args[2].isString()) {
            // "measured" or absent → keep sentinel.
        } else {
            sp = argcheck::expectedReal(args[2], 3, "sigpower", "awgn");
        }
    }
    if (args.size() >= 4 && (args[3].isChar() || args[3].isString())) {
        if (args.size() >= 5)
            throw Error("awgn: powertype must be the last input argument",
                        0, 0, "awgn", "", "numkit:awgn:powerTypeMustBeLast");
    }
    outs[0] = awgn(ctx.engine->rng(), args[0], snr, sp, ctx.engine->allocator());
}

void wgn_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
             keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 7, "wgn");
    const keel::i32 m = (keel::i32)args[0].toScalar();
    const keel::i32 n = (keel::i32)args[1].toScalar();
    const keel::f64 p = args[2].toScalar();
    keel::String type("dBW");
    bool complex_out = false;
    for (keel::usize i = 3; i < args.size(); ++i) {
        if (args[i].isChar() || args[i].isString()) {
            const keel::String s = args[i].toString();
            if      (s == "dBW" || s == "dBm" || s == "linear") type.assign(s);
            else if (s == "complex") complex_out = true;
            else if (s == "real")    complex_out = false;
        }
    }
    outs[0] = wgn(ctx.engine->rng(), m, n, p, type, complex_out, ctx.engine->allocator());
}

void bsc_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
             keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "bsc");
    outs[0] = bsc(ctx.engine->rng(), args[0], args[1].toScalar(), ctx.engine->allocator());
}

void qfunc_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
               keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "qfunc");
    outs[0] = qfunc(args[0], ctx.engine->allocator());
}

void qfuncinv_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "qfuncinv");
    outs[0] = qfuncinv(args[0], ctx.engine->allocator());
}

void marcumq_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "marcumq");
    const keel::i32 m = (args.size() >= 3 && !args[2].isEmpty())
                  ? (keel::i32)args[2].toScalar() : 1;
    outs[0] = marcumq(args[0], args[1], m, ctx.engine->allocator());
}

void berawgn_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, "berawgn");  // no max: option tail in the reference (no count wall to 12 args)
    keel::String mod = args[1].toString();
    const keel::i32 M = (keel::i32)args[2].toScalar();
    outs[0] = berawgn(args[0], mod, M, ctx.engine->allocator());
}

void noisebw_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 4, 4, "noisebw");
    const keel::i32    n  = (keel::i32)args[2].toScalar();
    const keel::f64 fs = args[3].toScalar();
    outs[0] = noisebw(args[0], args[1], n, fs, ctx.engine->allocator());
}

void berconfint_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                    keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "berconfint");
    const keel::f64 k     = args[0].toScalar();
    const keel::f64 n     = args[1].toScalar();
    const keel::f64 level = (args.size() >= 3 && !args[2].isEmpty())
                         ? args[2].toScalar() : 0.95;
    auto [ber, ci] = berconfint(k, n, level, ctx.engine->allocator());
    outs[0] = std::move(ber);
    if (nargout > 1) outs[1] = std::move(ci);
}

void convertSNR_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                    keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "convertSNR");  // no max: mode/name-value tail in the reference (no count wall to 12 args)
    auto toLower = [](keel::StringRef sv) {
        keel::String s(sv);
        for (char &c : s)
            if (c >= 'A' && c <= 'Z') c = keel::as<char>(c + 32);
        return s;
    };
    auto isValidMode = [](keel::StringRef m) {
        return m == "ebno" || m == "esno" || m == "snr";
    };

    if (!(args[1].isChar() || args[1].isString()))
        throw Error("convertSNR: input 2 (inputmode) must be a string ('ebno', 'esno', or 'snr')",
                    0, 0, "convertSNR", "", "numkit:convertSNR:badInputMode");
    const keel::String rawInType = args[1].toString();
    keel::String in_type = toLower(rawInType);
    if (!isValidMode(in_type))
        throw Error(keel::format("convertSNR: input 2 (inputmode) must be 'ebno', 'esno', or 'snr' (got '{}')", rawInType),
                    0, 0, "convertSNR", "", "numkit:convertSNR:badInputMode");

    keel::String out_type("snr");
    keel::usize nvStart = 2;
    if (args.size() >= 3 && (args[2].isChar() || args[2].isString())) {
        const keel::String rawCand = args[2].toString();
        keel::String cand = toLower(rawCand);
        if (isValidMode(cand)) {
            out_type = keel::move(cand);
            nvStart = 3;
        } else if (args.size() == 3) {
            throw Error(keel::format("convertSNR: input 3 (outputmode) must be 'ebno', 'esno', or 'snr' (got '{}')", rawCand),
                        0, 0, "convertSNR", "", "numkit:convertSNR:badOutputMode");
        }
    }

    if ((args.size() - nvStart) % 2 != 0)
        throw Error("convertSNR: Name-Value arguments must occur in pairs",
                    0, 0, "convertSNR", "", "numkit:convertSNR:unpairedNameValue");

    keel::i32 k = 1;
    for (keel::usize i = nvStart; i + 1 < args.size(); i += 2) {
        if (!(args[i].isChar() || args[i].isString()))
            throw Error("convertSNR: parameter name must be a string",
                        0, 0, "convertSNR", "", "numkit:convertSNR:badParamName");
        const keel::String rawKey = args[i].toString();
        const keel::String key = toLower(rawKey);
        if (key == "bitspersymbol") {
            k = keel::as<keel::i32>(argcheck::expectedPositiveInteger(
                argcheck::expectedReal(args[i + 1], keel::i32(i + 2), "BitsPerSymbol", "convertSNR"),
                keel::i32(i + 2), "BitsPerSymbol", "convertSNR"));
        } else if (key == "codingrate" || key == "samplespersymbol" ||
                   key == "bandwidth" || key == "symbolrate") {
            (void)argcheck::expectedReal(args[i + 1], keel::i32(i + 2), rawKey, "convertSNR");
        } else {
            throw Error(keel::format("convertSNR: unknown parameter '{}'", rawKey),
                        0, 0, "convertSNR", "", "numkit:convertSNR:badParamName");
        }
    }
    outs[0] = convertSNR(args[0], in_type, out_type, k, ctx.engine->allocator());
}

} // namespace detail
} // namespace numkit::comm
