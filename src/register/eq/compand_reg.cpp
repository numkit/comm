// toolboxes/comm/src/eq/compand_reg.cpp
//
// Register half of the comm `compand` builtin: the CallContext wrapper that
// reads the param / V / method args and delegates to the engine-free compute
// in compand.cpp. library.cpp forward-declares + registers it by name.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <numkit/comm/eq/compand.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, Span, ctx.engine->resource()
#include <numkit/value/value.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>

#include <string>

namespace numkit::comm {
namespace detail {

void compand_reg(Span<const Value> args, size_t /*nargout*/,
                 Span<Value> outs, CallContext &ctx)
{
    // R2025b: method optional, default 'mu/compressor' (probed equal).
    argcheck::checkNargin(args.size(), 3, 4, "compand");
    std::string method = "mu/compressor";
    if (args.size() >= 4 && !args[3].isEmpty()) {
        if (!args[3].isChar() && !args[3].isString())
            throw Error("compand: method must be a string",
                        0, 0, "compand", "", "numkit:compand:method");
        method = args[3].toString();
    }
    const double param = args[1].toScalar();
    const double V     = args[2].toScalar();
    outs[0] = compand(args[0], param, V, method, ctx.engine->resource());
}

} // namespace detail

} // namespace numkit::comm
