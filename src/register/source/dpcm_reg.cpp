// toolboxes/comm/src/source/dpcm_reg.cpp
//
// Register half of the comm DPCM builtins: the CallContext wrappers
// dpcmenco / dpcmdeco that destructure the (out, quanterr) pair from the
// engine-free compute in dpcm.cpp. library.cpp forward-declares + registers
// these by name.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <numkit/comm/source/dpcm.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, Span, ctx.engine->resource()
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>

#include <utility>

namespace numkit::comm {
namespace detail {

void dpcmenco_reg(Span<const Value> args, size_t nargout,
                  Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 4, 4, "dpcmenco");
    auto *mr = ctx.engine->resource();
    auto [indx, quanterr] = dpcmenco(args[0], args[1], args[2], args[3], mr);
    outs[0] = std::move(indx);
    if (nargout > 1) outs[1] = std::move(quanterr);
}

void dpcmdeco_reg(Span<const Value> args, size_t nargout,
                  Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 3, "dpcmdeco");
    auto *mr = ctx.engine->resource();
    auto [sig, quanterr] = dpcmdeco(args[0], args[1], args[2], mr);
    outs[0] = std::move(sig);
    if (nargout > 1) outs[1] = std::move(quanterr);
}

} // namespace detail

} // namespace numkit::comm
