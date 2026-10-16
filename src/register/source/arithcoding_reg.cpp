// toolboxes/comm/src/source/arithcoding_reg.cpp
//
// Register half of the comm arithmetic-coding builtins: the CallContext
// wrappers arithenco / arithdeco that delegate to the engine-free compute
// in arithcoding.cpp. library.cpp forward-declares + registers these by
// name.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <numkit/comm/source/arithcoding.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, Span, ctx.engine->resource()
#include <numkit/value/value.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>

namespace numkit::comm {
namespace detail {

void arithenco_reg(Span<const Value> args, size_t /*nargout*/,
                   Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "arithenco");
    outs[0] = arithenco(args[0], args[1], ctx.engine->resource());
}

void arithdeco_reg(Span<const Value> args, size_t /*nargout*/,
                   Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 3, "arithdeco");
    const size_t len = static_cast<size_t>(argcheck::expectedPositiveInteger(
        argcheck::expectedReal(args[2], 3, "len", "arithdeco"), 3, "len", "arithdeco"));
    outs[0] = arithdeco(args[0], args[1], len, ctx.engine->resource());
}

} // namespace detail

} // namespace numkit::comm
