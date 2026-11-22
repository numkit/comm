// toolboxes/comm/src/source/lloyds.cpp
//
// Lloyd-Max scalar quantizer designer delegating to keel::comm.

#include <keel/arena.h>
#include <keel/algo.h>
#include <keel/array.h>
#include <keel/comm.h>
#include <keel/core.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <numkit/comm/source/lloyds.hpp>

#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>


namespace numkit::comm {

namespace {

Value vecToRow(keel::Allocator *mr, keel::ArrayRef<keel::f64> v)
{
    Value out = Value::matrix(1, v.size(), ValueType::DOUBLE, mr);
    if (!v.empty()) keel::copy_n(out.doubleDataMut(), v.data(), v.size());
    return out;
}

} // namespace

std::tuple<Value, Value, keel::f64, keel::f64>
lloyds(const Value &training_set, const Value &ini_codebook, keel::f64 tol, keel::Allocator *mr)
{
    const keel::usize N = training_set.numel();
    if (N == 0)
        throw Error("lloyds: training set must be non-empty",
                    0, 0, "lloyds", "", "numkit:lloyds:EmptyTRAINING_SET");

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> training(N, 0.0, scratch.allocator());
    keel::f64 min_t = training_set.elemAsDouble(0);
    keel::f64 max_t = min_t;
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 x = training_set.elemAsDouble(i);
        training[i] = x;
        if (x < min_t) min_t = x;
        if (x > max_t) max_t = x;
    }
    if (max_t <= min_t)
        throw Error("lloyds: training set must span a positive range",
                    0, 0, "lloyds", "", "numkit:lloyds:InvalidTRAINING_SET");

    keel::Array<keel::f64> init_cb(scratch.allocator());
    keel::usize init_k = 0;
    if (ini_codebook.numel() == 0) {
        throw Error("lloyds: ini_codebook must have at least 1 entry",
                    0, 0, "lloyds", "", "numkit:lloyds:InvalidINI_CODEBOOK");
    } else if (ini_codebook.numel() == 1) {
        const keel::i64 K = keel::as<keel::i64>(keel::floor(ini_codebook.toScalar()));
        if (K < 1)
            throw Error("lloyds: ini_codebook count must be positive",
                        0, 0, "lloyds", "", "numkit:lloyds:NonPositiveINI_CODEBOOK");
        init_k = keel::as<keel::usize>(K);
    } else {
        init_cb.resize(ini_codebook.numel());
        for (keel::usize i = 0; i < ini_codebook.numel(); ++i)
            init_cb[i] = ini_codebook.elemAsDouble(i);
    }

    keel::comm::LloydsResult res;
    (void)keel::comm::lloyds(training.as_ref(), init_cb.as_ref(), init_k, tol, res,
                             scratch.allocator());
    return {vecToRow(mr, res.partition.as_ref()), vecToRow(mr, res.codebook.as_ref()),
            res.distortion, res.rel_distortion};
}

} // namespace numkit::comm
