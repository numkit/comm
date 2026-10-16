// toolboxes/comm/src/source/random_source_reg.cpp
//
// Register half of the comm random-source builtins: the CallContext
// wrappers randsrc / randerr that apply MATLAB defaults (alphabet [-1 1],
// 1 error per row), parse the optional reproducibility state, and delegate
// to the engine-free compute in random_source.cpp. library.cpp
// forward-declares + registers these by name.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <numkit/comm/source/random_source.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, Span, ctx.engine->resource()
#include <numkit/value/value.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>

namespace numkit::comm {
namespace detail {

void randsrc_reg(Span<const Value> args, size_t /*nargout*/,
                 Span<Value> outs, CallContext &ctx)
{
    // R2025b contract (0..4): randsrc() -> 1x1, randsrc(m) -> m x m.
    argcheck::checkNargin(args.size(), 0, 4, "randsrc");
    const size_t m = (args.size() >= 1) ? argcheck::arrayDim(args[0], "randsrc") : 1;
    const size_t n = (args.size() >= 2) ? argcheck::arrayDim(args[1], "randsrc") : m;
    auto *mr = ctx.engine->resource();

    // Default alphabet = [-1, 1] if not provided.
    Value default_alphabet;
    const Value *alphabet = nullptr;
    if (args.size() >= 3 && !args[2].isEmpty()) {
        alphabet = &args[2];
    } else {
        default_alphabet = Value::matrix(1, 2, ValueType::DOUBLE, mr);
        double *d = default_alphabet.doubleDataMut();
        d[0] = -1.0;
        d[1] =  1.0;
        alphabet = &default_alphabet;
    }

    bool have_state = false;
    uint32_t state = 0;
    if (args.size() >= 4 && !args[3].isEmpty()) {
        have_state = true;
        const double s = args[3].toScalar();
        if (s < 0.0)
            throw Error("randsrc: state must be a non-negative integer",
                        0, 0, "randsrc", "", "nk:randsrc:InvalidState");
        state = static_cast<uint32_t>(s);
    }

    outs[0] = randsrc(ctx.engine->rng(), m, n, *alphabet, have_state, state, mr);
}

void randerr_reg(Span<const Value> args, size_t /*nargout*/,
                 Span<Value> outs, CallContext &ctx)
{
    // R2025b contract (1..4): randerr(m) -> m x m; 0 args rejects.
    argcheck::checkNargin(args.size(), 1, 4, "randerr");
    const size_t m = argcheck::arrayDim(args[0], "randerr");
    const size_t n = (args.size() >= 2) ? argcheck::arrayDim(args[1], "randerr") : m;
    auto *mr = ctx.engine->resource();

    // Default errspec = 1 error per row.
    Value default_errspec;
    const Value *errspec = nullptr;
    if (args.size() >= 3 && !args[2].isEmpty()) {
        errspec = &args[2];
    } else {
        default_errspec = Value::scalar(1.0, mr);
        errspec = &default_errspec;
    }

    bool have_state = false;
    uint32_t state = 0;
    if (args.size() >= 4 && !args[3].isEmpty()) {
        have_state = true;
        const double s = args[3].toScalar();
        if (s < 0.0)
            throw Error("randerr: state must be a non-negative integer",
                        0, 0, "randerr", "", "nk:randerr:InvalidState");
        state = static_cast<uint32_t>(s);
    }

    outs[0] = randerr(ctx.engine->rng(), m, n, *errspec, have_state, state, mr);
}

} // namespace detail

} // namespace numkit::comm
