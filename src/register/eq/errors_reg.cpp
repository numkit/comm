// toolboxes/comm/src/eq/errors_reg.cpp
//
// Register half of the comm error-metric builtins: the CallContext wrappers
// biterr / symerr that destructure the (count, ratio) pair from the
// engine-free compute in errors.cpp. library.cpp forward-declares +
// registers these by name.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <numkit/comm/eq/errors.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, Span, ctx.engine->resource()
#include <numkit/value/value.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>

#include <utility>

namespace numkit::comm {
namespace detail {

// biterr(x, y[, k]): out[0] = total bit count, out[1] = ratio.
void biterr_reg(Span<const Value> args, size_t nargout,
                Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 4, "biterr");
    int k = 0;  // 0 -> auto width
    if (args.size() >= 3 && !args[2].isEmpty()) {
        k = argcheck::expectedPositiveInteger(
        argcheck::expectedReal(args[2], 3, "k", "biterr"), 3, "k", "biterr");
    }
    auto [number, ratio] = biterr(args[0], args[1], k, ctx.engine->resource());
    outs[0] = std::move(number);
    if (nargout > 1) outs[1] = std::move(ratio);
}

// symerr(x, y[, flg]): out[0] = symbol error count, out[1] = ratio.
void symerr_reg(Span<const Value> args, size_t nargout,
                Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "symerr");
    std::string flg = "overall";
    if (args.size() >= 3 && !args[2].isEmpty()) {
        if (!args[2].isChar() && !args[2].isString())
            throw Error("symerr: flg must be 'overall', 'column-wise', or 'row-wise'",
                        0, 0, "symerr", "", "numkit:symerr:flg");
        flg = args[2].toString();
    }
    auto [count, ratio] = symerr(args[0], args[1], flg, ctx.engine->resource());
    outs[0] = std::move(count);
    if (nargout > 1) outs[1] = std::move(ratio);
}

} // namespace detail

} // namespace numkit::comm
