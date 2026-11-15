// toolboxes/comm/src/modulation/generic_qam_reg.cpp
//
// Register half of the comm generic-constellation builtins: the CallContext
// wrappers genqammod / genqamdemod that delegate to the engine-free compute
// in generic_qam.cpp. library.cpp forward-declares + registers these by
// name.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <numkit/comm/modulation/generic_qam.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, Span, ctx.engine->resource()
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <string>

namespace numkit::comm {
namespace detail {

namespace {

std::string toLowerAscii(keel::StringRef sv) {
    std::string s(sv.data(), sv.size());
    for (char &ch : s)
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    return s;
}

Value scaleConstellationUnitPower(const Value &c, keel::Allocator *mr) {
    const keel::usize M = c.numel();
    if (M == 0) return c;
    double sumP = 0.0;
    if (c.isComplex()) {
        const auto *cd = c.complexData();
        for (keel::usize k = 0; k < M; ++k)
            sumP += cd[k].real() * cd[k].real() + cd[k].imag() * cd[k].imag();
    } else {
        for (keel::usize k = 0; k < M; ++k) {
            const double v = c.elemAsDouble(k);
            sumP += v * v;
        }
    }
    const double avgP = sumP / static_cast<double>(M);
    if (!(avgP > 0.0)) return c;
    const double scale = 1.0 / std::sqrt(avgP);
    if (c.isComplex()) {
        Value out = Value::matrix(c.dims().rows(), c.dims().cols(), ValueType::COMPLEX, mr);
        const auto *src = c.complexData();
        auto *dst = out.complexDataMut();
        for (keel::usize k = 0; k < M; ++k) dst[k] = src[k] * scale;
        return out;
    } else {
        Value out = Value::matrix(c.dims().rows(), c.dims().cols(), ValueType::DOUBLE, mr);
        auto *dst = out.doubleDataMut();
        for (keel::usize k = 0; k < M; ++k) dst[k] = c.elemAsDouble(k) * scale;
        return out;
    }
}

keel::usize checkBitsPerSymbol(keel::usize M, const char *fn) {
    if (M < 2 || (M & (M - 1)) != 0)
        throw Error(std::string(fn) + ": constellation length M must be a power of 2 (>= 2) in bit mode",
                    0, 0, fn, "", std::string("numkit:") + fn + ":NotPow2");
    keel::usize bps = 0;
    while ((keel::usize(1) << bps) < M) ++bps;
    return bps;
}

Value packBitsToSymbols(const Value &x, keel::usize bps, const char *fn, keel::Allocator *mr) {
    const keel::usize H = x.dims().rows();
    const keel::usize W = x.dims().cols();
    const bool rowVec = (H == 1 && W >= bps && (H % bps != 0));
    const keel::usize bitLen = rowVec ? W : H;
    if (bitLen == 0 || (bitLen % bps) != 0)
        throw Error(std::string(fn) + ": bit input length must be a positive multiple of log2(M)",
                    0, 0, fn, "", std::string("numkit:") + fn + ":BitLength");
    const keel::usize nSyms = bitLen / bps;
    const keel::usize nCols = rowVec ? 1 : W;
    Value syms = rowVec ? Value::matrix(1, nSyms, ValueType::DOUBLE, mr)
                        : Value::matrix(nSyms, nCols, ValueType::DOUBLE, mr);
    double *sd = syms.doubleDataMut();
    for (keel::usize c = 0; c < nCols; ++c) {
        for (keel::usize s = 0; s < nSyms; ++s) {
            keel::usize val = 0;
            for (keel::usize b = 0; b < bps; ++b) {
                const keel::usize idx = rowVec ? (s * bps + b) : (c * H + s * bps + b);
                const double bit = x.elemAsDouble(idx);
                if (bit != 0.0 && bit != 1.0)
                    throw Error(std::string(fn) + ": bit input must contain only 0 and 1",
                                0, 0, fn, "", std::string("numkit:") + fn + ":InvalidBit");
                val = (val << 1) | (bit != 0.0 ? 1u : 0u);
            }
            sd[rowVec ? s : (c * nSyms + s)] = static_cast<double>(val);
        }
    }
    return syms;
}

Value unpackSymbolsToBits(const Value &syms, keel::usize bps, keel::Allocator *mr) {
    const keel::usize H = syms.dims().rows();
    const keel::usize W = syms.dims().cols();
    const bool rowVec = (H == 1 && W > 1);
    const keel::usize nSyms = rowVec ? W : H;
    const keel::usize nCols = rowVec ? 1 : W;
    Value bits = rowVec ? Value::matrix(1, nSyms * bps, ValueType::DOUBLE, mr)
                        : Value::matrix(nSyms * bps, nCols, ValueType::DOUBLE, mr);
    double *bd = bits.doubleDataMut();
    for (keel::usize c = 0; c < nCols; ++c) {
        for (keel::usize s = 0; s < nSyms; ++s) {
            const keel::usize val = static_cast<keel::usize>(syms.elemAsDouble(rowVec ? s : (c * H + s)));
            for (keel::usize b = 0; b < bps; ++b) {
                const keel::usize bit = (val >> (bps - 1 - b)) & 1u;
                const keel::usize outIdx = rowVec ? (s * bps + b) : (c * (nSyms * bps) + s * bps + b);
                bd[outIdx] = static_cast<double>(bit);
            }
        }
    }
    return bits;
}

} // namespace

void genqammod_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                   keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "genqammod");
    auto *mr = ctx.engine->allocator();
    bool bitInput = false;
    bool unitAvgPow = false;
    for (keel::usize i = 2; i < args.size(); i += 2) {
        if (i + 1 >= args.size() || (!args[i].isChar() && !args[i].isString()))
            throw Error("genqammod: options must be Name-Value pairs",
                        0, 0, "genqammod", "", "numkit:genqammod:BadNV");
        const std::string key = toLowerAscii(args[i].toString());
        if (key == "inputtype") {
            const std::string v = toLowerAscii(args[i + 1].toString());
            if (v == "bit") bitInput = true;
            else if (v == "integer") bitInput = false;
            else throw Error("genqammod: InputType must be 'integer' or 'bit'",
                             0, 0, "genqammod", "", "numkit:genqammod:BadInputType");
        } else if (key == "unitaveragepower") {
            unitAvgPow = args[i + 1].toScalar() != 0.0;
        } else if (key == "plotconstellation") {
            // accept and ignore plotting flag
        } else {
            throw Error("genqammod: unknown option '" + std::string(args[i].toString().as_ref()) + "'",
                        0, 0, "genqammod", "", "numkit:genqammod:BadOption");
        }
    }
    Value c = unitAvgPow ? scaleConstellationUnitPower(args[1], mr) : args[1];
    Value x = args[0];
    if (bitInput) {
        const keel::usize bps = checkBitsPerSymbol(c.numel(), "genqammod");
        x = packBitsToSymbols(x, bps, "genqammod", mr);
    }
    outs[0] = genqammod(x, c, mr);
}

void genqamdemod_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                     keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "genqamdemod");
    auto *mr = ctx.engine->allocator();
    bool bitOutput = false;
    bool unitAvgPow = false;
    for (keel::usize i = 2; i < args.size(); i += 2) {
        if (i + 1 >= args.size() || (!args[i].isChar() && !args[i].isString()))
            throw Error("genqamdemod: options must be Name-Value pairs",
                        0, 0, "genqamdemod", "", "numkit:genqamdemod:BadNV");
        const std::string key = toLowerAscii(args[i].toString());
        if (key == "outputtype") {
            const std::string v = toLowerAscii(args[i + 1].toString());
            if (v == "bit") bitOutput = true;
            else if (v == "integer") bitOutput = false;
            else throw Error("genqamdemod: OutputType must be 'integer' or 'bit'",
                             0, 0, "genqamdemod", "", "numkit:genqamdemod:BadOutputType");
        } else if (key == "unitaveragepower") {
            unitAvgPow = args[i + 1].toScalar() != 0.0;
        } else if (key == "noisevariance" || key == "plotconstellation") {
            // accept
        } else {
            throw Error("genqamdemod: unknown option '" + std::string(args[i].toString().as_ref()) + "'",
                        0, 0, "genqamdemod", "", "numkit:genqamdemod:BadOption");
        }
    }
    Value c = unitAvgPow ? scaleConstellationUnitPower(args[1], mr) : args[1];
    Value syms = genqamdemod(args[0], c, mr);
    if (bitOutput) {
        const keel::usize bps = checkBitsPerSymbol(c.numel(), "genqamdemod");
        syms = unpackSymbolsToBits(syms, bps, mr);
    }
    outs[0] = std::move(syms);
}

} // namespace detail
} // namespace numkit::comm
