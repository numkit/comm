// toolboxes/comm/src/channel/fading_reg.cpp
//
// Register half of the comm fading-channel builtins: the CallContext
// wrappers rayleighchan / ricianchan that delegate to the engine-free
// compute in fading.cpp. library.cpp forward-declares + registers these by
// name.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <numkit/comm/channel/fading.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, Span, ctx.engine->resource()
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>

namespace numkit::comm {
namespace detail {

void rayleighchan_reg(Span<const Value> args, size_t /*nargout*/,
                      Span<Value> outs, CallContext &ctx)
{
    // Absent from the R2025b reference (System-object API there);
    // contract fixed to the implemented signature.
    argcheck::checkNargin(args.size(), 1, 1, "rayleighchan");
    outs[0] = rayleighchan(ctx.engine->rng(), args[0], ctx.engine->resource());
}

void ricianchan_reg(Span<const Value> args, size_t /*nargout*/,
                    Span<Value> outs, CallContext &ctx)
{
    // Absent from the R2025b reference (System-object API there);
    // contract fixed to the implemented signature.
    argcheck::checkNargin(args.size(), 2, 2, "ricianchan");
    outs[0] = ricianchan(ctx.engine->rng(), args[0], args[1].toScalar(),
                         ctx.engine->resource());
}

} // namespace detail

} // namespace numkit::comm
