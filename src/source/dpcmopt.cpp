// toolboxes/comm/src/source/dpcmopt.cpp
//
// DPCM parameter optimiser delegating Yule-Walker / Levinson-Durbin
// and Lloyd-Max residual quantizer design to keel::comm.

#include <keel/arena.h>
#include <keel/array.h>
#include <keel/comm.h>
#include <keel/core.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <numkit/comm/source/dpcmopt.hpp>

#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include <cstring>

namespace numkit::comm {

namespace {

Value packRow(keel::ArrayRef<keel::f64> v, keel::Allocator *mr)
{
    Value out = Value::matrix(1, v.size(), ValueType::DOUBLE, mr);
    if (!v.empty()) std::memcpy(out.doubleDataMut(), v.data(), v.size() * sizeof(keel::f64));
    return out;
}

} // namespace

DpcmOptResult dpcmopt(const Value &training_set, keel::i32 ord,
                      const Value &ini_codebook, keel::Allocator *mr)
{
    if (ord < 1)
        throw numkit::Error("The predictor order must be a positive integer.",
                            0, 0, "dpcmopt", "", "numkit:dpcmopt:InvalidOrd");

    const keel::usize N = training_set.numel();
    if (N < keel::as<keel::usize>(ord) + 3)
        throw numkit::Error("The size of the training set is not large enough for the given predictor order.",
                            0, 0, "dpcmopt", "", "numkit:dpcmopt:InvalidInput");

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> x(N, 0.0, scratch.allocator());
    for (keel::usize i = 0; i < N; ++i) x[i] = training_set.elemAsDouble(i);

    keel::Array<keel::f64> init_cb(scratch.allocator());
    keel::usize init_k = 0;
    if (!ini_codebook.isEmpty()) {
        if (ini_codebook.numel() == 1) {
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
    }

    keel::comm::DpcmOptResult res;
    const auto st = keel::comm::dpcm_opt(x.as_ref(), ord, init_cb.as_ref(), init_k, res,
                                         scratch.allocator());
    if (st != keel::comm::CommStatus::kOk)
        throw Error("lloyds: training set must span a positive range",
                    0, 0, "lloyds", "", "numkit:lloyds:InvalidTRAINING_SET");

    DpcmOptResult out;
    out.predictor = packRow(res.predictor.as_ref(), mr);
    if (!ini_codebook.isEmpty()) {
        out.partition = packRow(res.partition.as_ref(), mr);
        out.codebook = packRow(res.codebook.as_ref(), mr);
    }
    return out;
}

} // namespace numkit::comm
