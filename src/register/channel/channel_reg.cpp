// toolboxes/comm/src/channel/channel_reg.cpp
//
// Register half of the comm channel/metric builtins: the CallContext
// wrappers awgn / wgn / bsc / qfunc / qfuncinv / marcumq / berawgn /
// noisebw / berconfint / convertSNR that parse args and delegate to the
// engine-free compute in channel.cpp. library.cpp forward-declares +
// registers these by name.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <numkit/comm/channel/channel.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, Span, ctx.engine->resource()
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>

#include <string>
#include <utility>

namespace numkit::comm {
namespace detail {

void awgn_reg(Span<const Value> args, size_t /*nargout*/,
              Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 5, "awgn");
    // Marshal-only: voiced extraction; the 'measured' sentinel semantics
    // and the compute-side behaviour live in channel.cpp's awgn.
    const double snr = argcheck::expectedReal(args[1], 2, "snr", "awgn");
    double sp = -1e10;  // sentinel = "measured"
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
    outs[0] = awgn(ctx.engine->rng(), args[0], snr, sp, ctx.engine->resource());
}

void wgn_reg(Span<const Value> args, size_t /*nargout*/,
             Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 7, "wgn");
    const int m = (int)args[0].toScalar();
    const int n = (int)args[1].toScalar();
    const double p = args[2].toScalar();
    std::string type = "dBW";
    bool complex_out = false;
    for (size_t i = 3; i < args.size(); ++i) {
        if (args[i].isChar() || args[i].isString()) {
            const auto s = args[i].toString();
            if      (s == "dBW" || s == "dBm" || s == "linear") type = s;
            else if (s == "complex") complex_out = true;
            else if (s == "real")    complex_out = false;
        }
    }
    outs[0] = wgn(ctx.engine->rng(), m, n, p, type, complex_out, ctx.engine->resource());
}

void bsc_reg(Span<const Value> args, size_t /*nargout*/,
             Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "bsc");
    outs[0] = bsc(ctx.engine->rng(), args[0], args[1].toScalar(), ctx.engine->resource());
}

void qfunc_reg(Span<const Value> args, size_t /*nargout*/,
               Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "qfunc");
    outs[0] = qfunc(args[0], ctx.engine->resource());
}

void qfuncinv_reg(Span<const Value> args, size_t /*nargout*/,
                  Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "qfuncinv");
    outs[0] = qfuncinv(args[0], ctx.engine->resource());
}

void marcumq_reg(Span<const Value> args, size_t /*nargout*/,
                 Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "marcumq");
    const int m = (args.size() >= 3 && !args[2].isEmpty())
                  ? (int)args[2].toScalar() : 1;
    outs[0] = marcumq(args[0], args[1], m, ctx.engine->resource());
}

void berawgn_reg(Span<const Value> args, size_t /*nargout*/,
                 Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, "berawgn");  // no max: option tail in the reference (no count wall to 12 args)
    std::string mod = args[1].toString();
    const int M = (int)args[2].toScalar();
    outs[0] = berawgn(args[0], mod, M, ctx.engine->resource());
}

void noisebw_reg(Span<const Value> args, size_t /*nargout*/,
                 Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 4, 4, "noisebw");
    const int    n  = (int)args[2].toScalar();
    const double fs = args[3].toScalar();
    outs[0] = noisebw(args[0], args[1], n, fs, ctx.engine->resource());
}

void berconfint_reg(Span<const Value> args, size_t nargout,
                    Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "berconfint");
    const double k     = args[0].toScalar();
    const double n     = args[1].toScalar();
    const double level = (args.size() >= 3 && !args[2].isEmpty())
                         ? args[2].toScalar() : 0.95;
    auto [ber, ci] = berconfint(k, n, level, ctx.engine->resource());
    outs[0] = std::move(ber);
    if (nargout > 1) outs[1] = std::move(ci);
}

void convertSNR_reg(Span<const Value> args, size_t /*nargout*/,
                    Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "convertSNR");  // no max: mode/name-value tail in the reference (no count wall to 12 args)
    auto toLower = [](std::string s) {
        for (char &c : s)
            if (c >= 'A' && c <= 'Z') c = static_cast<char>(c + 32);
        return s;
    };
    auto isValidMode = [](const std::string &m) {
        return m == "ebno" || m == "esno" || m == "snr";
    };

    if (!(args[1].isChar() || args[1].isString()))
        throw Error("convertSNR: input 2 (inputmode) must be a string ('ebno', 'esno', or 'snr')",
                    0, 0, "convertSNR", "", "numkit:convertSNR:badInputMode");
    std::string in_type = toLower(args[1].toString());
    if (!isValidMode(in_type))
        throw Error("convertSNR: input 2 (inputmode) must be 'ebno', 'esno', or 'snr' (got '" +
                        args[1].toString() + "')",
                    0, 0, "convertSNR", "", "numkit:convertSNR:badInputMode");

    std::string out_type = "snr";
    size_t nvStart = 2;
    if (args.size() >= 3 && (args[2].isChar() || args[2].isString())) {
        std::string cand = toLower(args[2].toString());
        if (isValidMode(cand)) {
            out_type = cand;
            nvStart = 3;
        } else if (args.size() == 3) {
            throw Error("convertSNR: input 3 (outputmode) must be 'ebno', 'esno', or 'snr' (got '" +
                            args[2].toString() + "')",
                        0, 0, "convertSNR", "", "numkit:convertSNR:badOutputMode");
        }
    }

    if ((args.size() - nvStart) % 2 != 0)
        throw Error("convertSNR: Name-Value arguments must occur in pairs",
                    0, 0, "convertSNR", "", "numkit:convertSNR:unpairedNameValue");

    int k = 1;
    for (size_t i = nvStart; i + 1 < args.size(); i += 2) {
        if (!(args[i].isChar() || args[i].isString()))
            throw Error("convertSNR: parameter name must be a string",
                        0, 0, "convertSNR", "", "numkit:convertSNR:badParamName");
        const std::string key = toLower(args[i].toString());
        if (key == "bitspersymbol") {
            k = static_cast<int>(argcheck::expectedPositiveInteger(
                argcheck::expectedReal(args[i + 1], int(i + 2), "BitsPerSymbol", "convertSNR"),
                int(i + 2), "BitsPerSymbol", "convertSNR"));
        } else if (key == "codingrate" || key == "samplespersymbol" ||
                   key == "bandwidth" || key == "symbolrate") {
            (void)argcheck::expectedReal(args[i + 1], int(i + 2), args[i].toString(), "convertSNR");
        } else {
            throw Error("convertSNR: unknown parameter '" + args[i].toString() + "'",
                        0, 0, "convertSNR", "", "numkit:convertSNR:badParamName");
        }
    }
    outs[0] = convertSNR(args[0], in_type, out_type, k, ctx.engine->resource());
}

} // namespace detail
} // namespace numkit::comm
