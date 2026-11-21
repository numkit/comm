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

#include <keel/core.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <numkit/comm/coding/blockcoding.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, Span, ctx.engine->resource()
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>

#include <utility>

namespace numkit::comm {
namespace detail {

void gen2par_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "gen2par");
    outs[0] = gen2par(args[0], ctx.engine->allocator());
}

void syndtable_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                   keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "syndtable");
    outs[0] = syndtable(args[0], ctx.engine->allocator());
}

void hammgen_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "hammgen");
    const keel::i64 m = keel::as<keel::i64>(args[0].toScalar());
    const Value &prim = (args.size() >= 2) ? args[1] : Value::Empty;
    HammgenResult res = hammgen(m, prim, ctx.engine->allocator());
    outs[0] = std::move(res.h);
    if (nargout >= 2) outs[1] = std::move(res.g);
    if (nargout >= 3) outs[2] = Value::scalar(keel::as<keel::f64>(res.n),
                                              ctx.engine->allocator());
    if (nargout >= 4) outs[3] = Value::scalar(keel::as<keel::f64>(res.k),
                                              ctx.engine->allocator());
}

void cyclpoly_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "cyclpoly");
    const keel::i64 n = keel::as<keel::i64>(args[0].toScalar());
    const keel::i64 k = keel::as<keel::i64>(args[1].toScalar());
    const keel::String opt =
        (args.size() >= 3 && (args[2].isChar() || args[2].isString()))
            ? args[2].toString() : keel::String();
    outs[0] = cyclpoly(n, k, opt, ctx.engine->allocator());
}

void cyclgen_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "cyclgen");
    const keel::i64 n = keel::as<keel::i64>(args[0].toScalar());
    const keel::String opt =
        (args.size() >= 3 && (args[2].isChar() || args[2].isString()))
            ? args[2].toString() : keel::String("system");
    CyclgenResult res = cyclgen(n, args[1], opt, ctx.engine->allocator());
    outs[0] = std::move(res.h);
    if (nargout >= 2) outs[1] = std::move(res.g);
    if (nargout >= 3) outs[2] = Value::scalar(keel::as<keel::f64>(res.k),
                                              ctx.engine->allocator());
}

void encode_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 5, "encode");
    const keel::i64 n = keel::as<keel::i64>(args[1].toScalar());
    const keel::i64 k = keel::as<keel::i64>(args[2].toScalar());
    const keel::String method =
        (args.size() >= 4 && (args[3].isChar() || args[3].isString()))
            ? args[3].toString() : keel::String("hamming/binary");
    const Value &opt = (args.size() >= 5) ? args[4] : Value::Empty;
    EncodeResult res = encode(args[0], n, k, method, opt, ctx.engine->allocator());
    outs[0] = std::move(res.code);
    if (nargout >= 2) outs[1] = Value::scalar(keel::as<keel::f64>(res.added),
                                              ctx.engine->allocator());
}

void decode_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 6, "decode");
    const keel::i64 n = keel::as<keel::i64>(args[1].toScalar());
    const keel::i64 k = keel::as<keel::i64>(args[2].toScalar());
    const keel::String method =
        (args.size() >= 4 && (args[3].isChar() || args[3].isString()))
            ? args[3].toString() : keel::String("hamming/binary");
    const Value &opt = (args.size() >= 5) ? args[4] : Value::Empty;
    DecodeResult res = decode(args[0], n, k, method, opt, ctx.engine->allocator());
    outs[0] = std::move(res.msg);
    if (nargout >= 2) outs[1] = std::move(res.err);
}

} // namespace detail

} // namespace numkit::comm
