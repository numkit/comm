// toolboxes/comm/src/source/quantiz.cpp
//
// Scalar quantizer applier delegating to keel::comm.

#include <keel/arena.h>
#include <keel/array.h>
#include <keel/comm.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <numkit/comm/source/quantiz.hpp>

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

Value quantiz_indx(const Value &sig, const Value &partition, keel::Allocator *mr)
{
    const keel::usize N = sig.numel();
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const auto sv = readVec(sig, scratch.allocator());
    const auto pv = readVec(partition, scratch.allocator());
    keel::Array<keel::usize> idx(N, 0u, scratch.allocator());
    (void)keel::comm::quantiz_index(sv.as_ref(), pv.as_ref(), idx.as_mut_ref());

    const bool row = isRow(sig);
    Value indx = Value::matrix(row ? 1 : N, row ? N : 1, ValueType::DOUBLE, mr);
    keel::f64 *o = indx.doubleDataMut();
    for (keel::usize i = 0; i < N; ++i) o[i] = keel::as<keel::f64>(idx[i]);
    return indx;
}

QuantizResult
quantiz(const Value &sig, const Value &partition, const Value &codebook, keel::Allocator *mr)
{
    if (codebook.numel() != partition.numel() + 1)
        throw Error("quantiz: length(codebook) must equal length(partition) + 1",
                    0, 0, "quantiz", "", "numkit:quantiz:invalidCodebook");

    const keel::usize N = sig.numel();
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const auto sv = readVec(sig, scratch.allocator());
    const auto pv = readVec(partition, scratch.allocator());
    const auto cv = readVec(codebook, scratch.allocator());
    keel::Array<keel::usize> idx(N, 0u, scratch.allocator());

    const bool row = isRow(sig);
    Value indx = Value::matrix(row ? 1 : N, row ? N : 1, ValueType::DOUBLE, mr);
    Value quantv = Value::matrix(row ? 1 : N, row ? N : 1, ValueType::DOUBLE, mr);
    keel::f64 distor = 0.0;
    (void)keel::comm::quantiz(sv.as_ref(), pv.as_ref(), cv.as_ref(), idx.as_mut_ref(),
                              keel::MutArrayRef<keel::f64>(quantv.doubleDataMut(), N), distor);

    keel::f64 *oi = indx.doubleDataMut();
    for (keel::usize i = 0; i < N; ++i) oi[i] = keel::as<keel::f64>(idx[i]);
    return {keel::move(indx), keel::move(quantv), distor};
}

} // namespace numkit::comm
