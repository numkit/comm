// toolboxes/comm/src/modulation/mil188_reg.cpp
//
// Register half of the comm MIL-STD-188 QAM builtins: the CallContext
// wrappers mil188qammod / mil188qamdemod that delegate to the engine-free
// compute in mil188.cpp. library.cpp forward-declares + registers these by
// name.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <keel/core.h>
#include <keel/ref.h>
#include <numkit/comm/modulation/mil188.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, Span, ctx.engine->resource()
#include <numkit/value/value.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>

namespace numkit::comm {
namespace detail {

void mil188qammod_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                      keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 10, "mil188qammod");
    // Marshal-only: the M-set domain check lives in the compute-side
    // mil188MSetArgcheck (one identifier for eval and direct C++ calls).
    const keel::i32 M = argcheck::expectedInteger(
        argcheck::expectedReal(args[1], 2, "M", "mil188qammod"),
        2, "M", "mil188qammod");
    outs[0] = mil188qammod(args[0], M, ctx.engine->allocator());
}

void mil188qamdemod_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                        keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "mil188qamdemod");  // no max: name-value tail in the reference (no count wall to 12 args)
    // Marshal-only: the M-set domain check lives in the compute-side
    // mil188MSetArgcheck (one identifier for eval and direct C++ calls).
    const keel::i32 M = argcheck::expectedInteger(
        argcheck::expectedReal(args[1], 2, "M", "mil188qamdemod"),
        2, "M", "mil188qamdemod");
    outs[0] = mil188qamdemod(args[0], M, ctx.engine->allocator());
}

} // namespace detail
} // namespace numkit::comm
