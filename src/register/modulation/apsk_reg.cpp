// toolboxes/comm/src/modulation/apsk_reg.cpp
//
// Register half of the comm APSK builtins: the CallContext wrappers
// apskmod / apskdemod that marshal the per-ring M / radii Values into
// scratch Span buffers and delegate to the engine-free compute in apsk.cpp.
// library.cpp forward-declares + registers these by name.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <numkit/comm/modulation/apsk.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, Span, ctx.engine->resource()
#include <numkit/value/scratch.hpp>
#include <numkit/value/span.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <string>
#include <vector>

namespace numkit::comm {
namespace detail {

namespace {

std::string toLowerApsk(std::string s) {
    for (char &ch : s)
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    return s;
}

// Build a ScratchVec<size_t> from a Value's elements via elemAsDouble.
ScratchVec<size_t> valueToScratchSizes(const Value &v, std::pmr::memory_resource *mr)
{
    ScratchVec<size_t> out(mr);
    const size_t n = v.numel();
    out.reserve(n);
    for (size_t i = 0; i < n; ++i)
        out.push_back(static_cast<size_t>(v.elemAsDouble(i)));
    return out;
}

ScratchVec<double> valueToScratchDoubles(const Value &v, std::pmr::memory_resource *mr)
{
    ScratchVec<double> out(mr);
    const size_t n = v.numel();
    out.reserve(n);
    for (size_t i = 0; i < n; ++i)
        out.push_back(v.elemAsDouble(i));
    return out;
}

// Build per-ring Gray symbol mapping of length sum(M).
Value buildGrayMapping(Span<const size_t> M, std::pmr::memory_resource *mr)
{
    size_t Ntot = 0;
    bool sameRingSize = !M.empty();
    for (size_t m : M) {
        Ntot += m;
        if (m != M[0]) sameRingSize = false;
    }
    Value mp = Value::matrix(1, Ntot, ValueType::DOUBLE, mr);
    double *md = mp.doubleDataMut();
    size_t cum = 0;
    const size_t R = M.size();
    const bool pow2Rings = (R >= 2) && ((R & (R - 1)) == 0) && sameRingSize && ((M[0] & (M[0] - 1)) == 0);
    for (size_t r = 0; r < R; ++r) {
        const size_t Mr = M[r];
        const size_t ringGray = pow2Rings ? (r ^ (r >> 1)) : r;
        const size_t ringBase = pow2Rings ? (ringGray * Mr) : cum;
        // Generate Gray permutation of 0..Mr-1
        size_t p2 = 1;
        while (p2 < Mr) p2 <<= 1;
        size_t written = 0;
        for (size_t k = 0; k < p2 && written < Mr; ++k) {
            const size_t g = k ^ (k >> 1);
            if (g < Mr) {
                md[cum + written] = static_cast<double>(ringBase + g);
                ++written;
            }
        }
        cum += Mr;
    }
    return mp;
}

size_t apskBitsPerSymbol(Span<const size_t> M, const char *fn)
{
    size_t Ntot = 0;
    for (size_t m : M) Ntot += m;
    if (Ntot < 2 || (Ntot & (Ntot - 1)) != 0)
        throw Error(std::string(fn) + ": sum(M) must be a power of 2 (>= 2) in bit mode",
                    0, 0, fn, "", std::string("numkit:") + fn + ":NotPow2");
    size_t bps = 0;
    while ((size_t(1) << bps) < Ntot) ++bps;
    return bps;
}

Value packApskBits(const Value &x, size_t bps, const char *fn, std::pmr::memory_resource *mr)
{
    const size_t H = x.dims().rows();
    const size_t W = x.dims().cols();
    const bool rowVec = (H == 1 && W >= bps && (H % bps != 0));
    const size_t bitLen = rowVec ? W : H;
    if (bitLen == 0 || (bitLen % bps) != 0)
        throw Error(std::string(fn) + ": bit input length must be a positive multiple of log2(sum(M))",
                    0, 0, fn, "", std::string("numkit:") + fn + ":BitLength");
    const size_t nSyms = bitLen / bps;
    const size_t nCols = rowVec ? 1 : W;
    Value syms = rowVec ? Value::matrix(1, nSyms, ValueType::DOUBLE, mr)
                        : Value::matrix(nSyms, nCols, ValueType::DOUBLE, mr);
    double *sd = syms.doubleDataMut();
    for (size_t c = 0; c < nCols; ++c) {
        for (size_t s = 0; s < nSyms; ++s) {
            size_t val = 0;
            for (size_t b = 0; b < bps; ++b) {
                const size_t idx = rowVec ? (s * bps + b) : (c * H + s * bps + b);
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

Value unpackApskBits(const Value &syms, size_t bps, std::pmr::memory_resource *mr)
{
    const size_t H = syms.dims().rows();
    const size_t W = syms.dims().cols();
    const bool rowVec = (H == 1 && W > 1);
    const size_t nSyms = rowVec ? W : H;
    const size_t nCols = rowVec ? 1 : W;
    Value bits = rowVec ? Value::matrix(1, nSyms * bps, ValueType::DOUBLE, mr)
                        : Value::matrix(nSyms * bps, nCols, ValueType::DOUBLE, mr);
    double *bd = bits.doubleDataMut();
    for (size_t c = 0; c < nCols; ++c) {
        for (size_t s = 0; s < nSyms; ++s) {
            const size_t val = static_cast<size_t>(syms.elemAsDouble(rowVec ? s : (c * H + s)));
            for (size_t b = 0; b < bps; ++b) {
                const size_t bit = (val >> (bps - 1 - b)) & 1u;
                const size_t outIdx = rowVec ? (s * bps + b) : (c * (nSyms * bps) + s * bps + b);
                bd[outIdx] = static_cast<double>(bit);
            }
        }
    }
    return bits;
}

void parseApskTail(Span<const Value> args, Span<const size_t> M, const char *fn, bool isMod,
                   Value &po, Value &mp, bool &bitMode, std::pmr::memory_resource *mr)
{
    po = Value::Empty;
    mp = Value::Empty;
    bitMode = false;
    size_t i = 3;
    if (i < args.size() && !args[i].isChar() && !args[i].isString()) {
        po = args[i];
        ++i;
    }
    if (i < args.size() && !args[i].isChar() && !args[i].isString()) {
        mp = args[i];
        ++i;
    }
    bool useGray = false;
    for (; i < args.size(); i += 2) {
        if (i + 1 >= args.size() || (!args[i].isChar() && !args[i].isString()))
            throw Error(std::string(fn) + ": options must be Name-Value pairs",
                        0, 0, fn, "", std::string("numkit:") + fn + ":BadNV");
        const std::string key = toLowerApsk(args[i].toString());
        const Value &val = args[i + 1];
        if (key == "symbolmapping") {
            if (val.isChar() || val.isString()) {
                const std::string vs = toLowerApsk(val.toString());
                if (vs == "gray") useGray = true;
                else if (vs == "custom" || vs == "bin" || vs == "binary") useGray = false;
                else throw Error(std::string(fn) + ": SymbolMapping must be 'gray', 'custom', or a numeric vector",
                                 0, 0, fn, "", std::string("numkit:") + fn + ":BadMapping");
            } else {
                mp = val;
                useGray = false;
            }
        } else if (key == "customsymbolmapping") {
            mp = val;
            useGray = false;
        } else if (isMod && key == "inputtype") {
            const std::string vs = toLowerApsk(val.toString());
            if (vs == "bit") bitMode = true;
            else if (vs == "integer") bitMode = false;
            else throw Error("apskmod: InputType must be 'integer' or 'bit'",
                             0, 0, "apskmod", "", "numkit:apskmod:BadInputType");
        } else if (!isMod && key == "outputtype") {
            const std::string vs = toLowerApsk(val.toString());
            if (vs == "bit") bitMode = true;
            else if (vs == "integer") bitMode = false;
            else throw Error("apskdemod: OutputType must be 'integer' or 'bit'",
                             0, 0, "apskdemod", "", "numkit:apskdemod:BadOutputType");
        } else if (key == "plotconstellation" || key == "noisevariance") {
            // accept
        } else {
            throw Error(std::string(fn) + ": unknown option '" + args[i].toString() + "'",
                        0, 0, fn, "", std::string("numkit:") + fn + ":BadOption");
        }
    }
    if (useGray && mp.isEmpty())
        mp = buildGrayMapping(M, mr);
}

} // namespace

void apskmod_reg(Span<const Value> args, size_t /*nargout*/,
                 Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, "apskmod");
    auto *mr = ctx.engine->resource();
    ScratchArena scratch(mr);
    auto Mv = valueToScratchSizes(args[1], &scratch);
    auto Rv = valueToScratchDoubles(args[2], &scratch);
    Span<const size_t> Mspan(Mv.data(), Mv.size());
    Value po, mp;
    bool bitMode = false;
    parseApskTail(args, Mspan, "apskmod", true, po, mp, bitMode, mr);
    Value x = args[0];
    if (bitMode) {
        const size_t bps = apskBitsPerSymbol(Mspan, "apskmod");
        x = packApskBits(x, bps, "apskmod", mr);
    }
    outs[0] = apskmod(x, Mspan, Span<const double>(Rv.data(), Rv.size()), po, mp, mr);
}

void apskdemod_reg(Span<const Value> args, size_t /*nargout*/,
                   Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, "apskdemod");
    auto *mr = ctx.engine->resource();
    ScratchArena scratch(mr);
    auto Mv = valueToScratchSizes(args[1], &scratch);
    auto Rv = valueToScratchDoubles(args[2], &scratch);
    Span<const size_t> Mspan(Mv.data(), Mv.size());
    Value po, mp;
    bool bitMode = false;
    parseApskTail(args, Mspan, "apskdemod", false, po, mp, bitMode, mr);
    Value syms = apskdemod(args[0], Mspan, Span<const double>(Rv.data(), Rv.size()), po, mp, mr);
    if (bitMode) {
        const size_t bps = apskBitsPerSymbol(Mspan, "apskdemod");
        syms = unpackApskBits(syms, bps, mr);
    }
    outs[0] = std::move(syms);
}

} // namespace detail
} // namespace numkit::comm
