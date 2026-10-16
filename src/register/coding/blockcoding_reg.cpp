// toolboxes/comm/src/coding/blockcoding_reg.cpp
//
// Register half of the comm block-coding builtins: the CallContext wrappers
// gen2par / hammgen / cyclpoly / cyclgen / encode / decode that parse args,
// destructure the multi-output result structs (HammgenResult /
// CyclgenResult / EncodeResult / DecodeResult) from the engine-free compute
// in blockcoding.cpp, and emit the documented outputs. library.cpp
// forward-declares + registers these by name.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <numkit/comm/coding/blockcoding.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, Span, ctx.engine->resource()
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>

#include <string>
#include <utility>

namespace numkit::comm {
namespace detail {

void gen2par_reg(Span<const Value> args, size_t /*nargout*/,
                 Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "gen2par");
    outs[0] = gen2par(args[0], ctx.engine->resource());
}

void syndtable_reg(Span<const Value> args, size_t /*nargout*/,
                   Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "syndtable");
    outs[0] = syndtable(args[0], ctx.engine->resource());
}

void hammgen_reg(Span<const Value> args, size_t nargout,
                 Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "hammgen");
    const long long m = static_cast<long long>(args[0].toScalar());
    const Value &prim = (args.size() >= 2) ? args[1] : Value::Empty;
    HammgenResult res = hammgen(m, prim, ctx.engine->resource());
    outs[0] = std::move(res.h);
    if (nargout >= 2) outs[1] = std::move(res.g);
    if (nargout >= 3) outs[2] = Value::scalar(static_cast<double>(res.n),
                                              ctx.engine->resource());
    if (nargout >= 4) outs[3] = Value::scalar(static_cast<double>(res.k),
                                              ctx.engine->resource());
}

void cyclpoly_reg(Span<const Value> args, size_t /*nargout*/,
                  Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "cyclpoly");
    const long long n = static_cast<long long>(args[0].toScalar());
    const long long k = static_cast<long long>(args[1].toScalar());
    const std::string opt =
        (args.size() >= 3 && (args[2].isChar() || args[2].isString()))
            ? args[2].toString() : std::string();
    outs[0] = cyclpoly(n, k, opt, ctx.engine->resource());
}

void cyclgen_reg(Span<const Value> args, size_t nargout,
                 Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "cyclgen");
    const long long n = static_cast<long long>(args[0].toScalar());
    const std::string opt =
        (args.size() >= 3 && (args[2].isChar() || args[2].isString()))
            ? args[2].toString() : std::string("system");
    CyclgenResult res = cyclgen(n, args[1], opt, ctx.engine->resource());
    outs[0] = std::move(res.h);
    if (nargout >= 2) outs[1] = std::move(res.g);
    if (nargout >= 3) outs[2] = Value::scalar(static_cast<double>(res.k),
                                              ctx.engine->resource());
}

void encode_reg(Span<const Value> args, size_t nargout,
                Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 5, "encode");
    const long long n = static_cast<long long>(args[1].toScalar());
    const long long k = static_cast<long long>(args[2].toScalar());
    const std::string method =
        (args.size() >= 4 && (args[3].isChar() || args[3].isString()))
            ? args[3].toString() : std::string("hamming/binary");
    const Value &opt = (args.size() >= 5) ? args[4] : Value::Empty;
    EncodeResult res = encode(args[0], n, k, method, opt, ctx.engine->resource());
    outs[0] = std::move(res.code);
    if (nargout >= 2) outs[1] = Value::scalar(static_cast<double>(res.added),
                                              ctx.engine->resource());
}

void decode_reg(Span<const Value> args, size_t nargout,
                Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 6, "decode");
    const long long n = static_cast<long long>(args[1].toScalar());
    const long long k = static_cast<long long>(args[2].toScalar());
    const std::string method =
        (args.size() >= 4 && (args[3].isChar() || args[3].isString()))
            ? args[3].toString() : std::string("hamming/binary");
    const Value &opt = (args.size() >= 5) ? args[4] : Value::Empty;
    DecodeResult res = decode(args[0], n, k, method, opt, ctx.engine->resource());
    outs[0] = std::move(res.msg);
    if (nargout >= 2) outs[1] = std::move(res.err);
}

} // namespace detail

} // namespace numkit::comm
