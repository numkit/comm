// toolboxes/comm/src/eq/compand.cpp
//
// μ-law and A-law companding, delegated to keel::comm.

#include <keel/arena.h>
#include <keel/array.h>
#include <keel/comm.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <numkit/comm/eq/compand.hpp>

#include <numkit/value/argcheck.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

namespace numkit::comm {

namespace {

void compand_argcheck(double param, double V, keel::StringRef method)
{
    if (param <= 0.0)
        throw Error("compand: param (μ or A) must be positive",
                    0, 0, "compand", "", "numkit:compand:param");
    if (V <= 0.0)
        throw Error("compand: V (peak magnitude) must be positive",
                    0, 0, "compand", "", "numkit:compand:V");

    const bool isMu = (method.size() >= 2
                       && (method[0] == 'm' || method[0] == 'M')
                       && (method[1] == 'u' || method[1] == 'U'));
    const bool isA  = (method.size() >= 1
                       && (method[0] == 'a' || method[0] == 'A'));
    if (!isMu && !isA)
        throw Error("compand: method must start with 'mu/' or 'A/'",
                    0, 0, "compand", "", "numkit:compand:method");
    const bool isCompress = method.contains("compressor");
    const bool isExpand = method.contains("expander");
    if (!isCompress && !isExpand)
        throw Error("compand: method must end with '/compressor' or '/expander'",
                    0, 0, "compand", "", "numkit:compand:method");
}

} // namespace

Value compand(const Value &x, double param, double V,
              keel::StringRef method, keel::Allocator *mr)
{
    compand_argcheck(param, V, method);

    const bool isMu = (method.size() >= 2
                       && (method[0] == 'm' || method[0] == 'M')
                       && (method[1] == 'u' || method[1] == 'U'));
    const bool isCompress = method.contains("compressor");

    const auto mode = isMu
        ? (isCompress ? keel::comm::CompandMode::kMuCompress : keel::comm::CompandMode::kMuExpand)
        : (isCompress ? keel::comm::CompandMode::kACompress  : keel::comm::CompandMode::kAExpand);

    Value out = Value::matrix(x.dims().rows(), x.dims().cols(), ValueType::DOUBLE, mr);
    const keel::usize N = x.numel();
    if (N == 0) return out;

    keel::MutArrayRef<double> outRef(out.doubleDataMut(), N);
    if (x.type() == ValueType::DOUBLE) {
        (void)keel::comm::compand({x.doubleData(), N}, param, V, mode, outRef);
    } else {
        keel::ArenaTemp scratch = keel::get_scratch(mr);
        keel::Array<double> in(N, scratch.allocator());
        for (keel::usize i = 0; i < N; ++i) in[i] = x.elemAsDouble(i);
        (void)keel::comm::compand(in, param, V, mode, outRef);
    }
    return out;
}

} // namespace numkit::comm
