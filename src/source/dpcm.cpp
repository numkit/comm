// toolboxes/comm/src/source/dpcm.cpp
//
// Differential Pulse Code Modulation encoder + decoder delegating to keel::comm.

#include <keel/arena.h>
#include <keel/array.h>
#include <keel/comm.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <numkit/comm/source/dpcm.hpp>

#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

namespace numkit::comm {

namespace {

bool isRow(const Value &v)
{
    return v.dims().rows() == 1 && v.dims().cols() >= 1;
}

keel::Array<keel::f64> readVec(const Value &v, keel::Allocator *alloc)
{
    const keel::usize n = v.numel();
    keel::Array<keel::f64> out(n, 0.0, alloc);
    for (keel::usize i = 0; i < n; ++i) out[i] = v.elemAsDouble(i);
    return out;
}

} // namespace

keel::Pair<Value, Value>
dpcmenco(const Value &sig, const Value &codebook,
         const Value &partition, const Value &predictor,
         keel::Allocator *mr)
{
    if (codebook.numel() != partition.numel() + 1)
        throw Error("dpcmenco: length(codebook) must equal length(partition) + 1",
                    0, 0, "dpcmenco", "", "numkit:dpcmenco:InvalidPartitionSize");
    if (predictor.numel() < 2)
        throw Error("dpcm: predictor must have at least 2 elements",
                    0, 0, "dpcm", "", "numkit:dpcm:InvalidPredictor");

    const keel::usize N = sig.numel();
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const auto sv = readVec(sig, scratch.allocator());
    const auto cv = readVec(codebook, scratch.allocator());
    const auto pv = readVec(partition, scratch.allocator());
    const auto pr = readVec(predictor, scratch.allocator());
    keel::Array<keel::usize> idx(N, 0u, scratch.allocator());

    const bool row = isRow(sig);
    Value indx_v = Value::matrix(row ? 1 : N, row ? N : 1, ValueType::DOUBLE, mr);
    Value quanterr_v = Value::matrix(row ? 1 : N, row ? N : 1, ValueType::DOUBLE, mr);
    (void)keel::comm::dpcm_encode(sv.as_ref(), cv.as_ref(), pv.as_ref(), pr.as_ref(),
                                  idx.as_mut_ref(),
                                  keel::MutArrayRef<keel::f64>(quanterr_v.doubleDataMut(), N),
                                  scratch.allocator());

    keel::f64 *oi = indx_v.doubleDataMut();
    for (keel::usize i = 0; i < N; ++i) oi[i] = keel::as<keel::f64>(idx[i]);
    return {keel::move(indx_v), keel::move(quanterr_v)};
}

keel::Pair<Value, Value>
dpcmdeco(const Value &indx, const Value &codebook,
         const Value &predictor, keel::Allocator *mr)
{
    if (predictor.numel() < 2)
        throw Error("dpcm: predictor must have at least 2 elements",
                    0, 0, "dpcm", "", "numkit:dpcm:InvalidPredictor");

    const keel::usize K = codebook.numel();
    const keel::usize N = indx.numel();
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::usize> iv(N, 0u, scratch.allocator());
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 v = indx.elemAsDouble(i);
        if (!(v >= 0.0) || v >= keel::as<keel::f64>(K))
            throw Error("dpcmdeco: index out of codebook range",
                        0, 0, "dpcmdeco", "", "numkit:dpcmdeco:OutOfRange");
        iv[i] = keel::as<keel::usize>(v);
    }

    const auto cv = readVec(codebook, scratch.allocator());
    const auto pr = readVec(predictor, scratch.allocator());

    const bool row = isRow(indx);
    Value sig_v = Value::matrix(row ? 1 : N, row ? N : 1, ValueType::DOUBLE, mr);
    Value quanterr_v = Value::matrix(row ? 1 : N, row ? N : 1, ValueType::DOUBLE, mr);
    (void)keel::comm::dpcm_decode(iv.as_ref(), cv.as_ref(), pr.as_ref(),
                                  keel::MutArrayRef<keel::f64>(sig_v.doubleDataMut(), N),
                                  keel::MutArrayRef<keel::f64>(quanterr_v.doubleDataMut(), N),
                                  scratch.allocator());
    return {keel::move(sig_v), keel::move(quanterr_v)};
}

} // namespace numkit::comm
