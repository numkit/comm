// toolboxes/comm/src/source/quantiz.cpp
//
// Scalar quantizer applier (companion to dpcmenco's inline bin search).
//
//   indx                       = quantiz(sig, partition)
//   [indx, quantv]             = quantiz(sig, partition, codebook)
//   [indx, quantv, distor]     = quantiz(sig, partition, codebook)
//
// indx(i) = sum(partition < sig(i)) ∈ [0, K-1]   (K = numel(codebook))
// quantv  = codebook(indx + 1)
// distor  = mean((sig - quantv).^2)

#include <keel/core.h>
#include <keel/memory.h>
#include <numkit/comm/source/quantiz.hpp>

#include <numkit/value/argcheck.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <utility>

namespace numkit::comm {

namespace {

bool isRow(const Value &v)
{
    return v.dims().rows() == 1 && v.dims().cols() >= 1;
}

void quantiz_indx_argcheck(const Value &sig, const Value &partition)
{
    (void)sig;
    (void)partition;
}

void quantiz_argcheck(const Value &partition, const Value &codebook)
{
    if (codebook.numel() != partition.numel() + 1)
        throw Error("quantiz: length(codebook) must equal "
                    "length(partition) + 1",
                    0, 0, "quantiz", "", "numkit:quantiz:invalidCodebook");
}

} // namespace

Value quantiz_indx(const Value &sig, const Value &partition,
                   keel::Allocator *mr)
{
    quantiz_indx_argcheck(sig, partition);
    const keel::usize N = sig.numel();
    const keel::usize M = partition.numel();
    const bool row = isRow(sig);
    Value indx = Value::matrix(row ? 1 : N, row ? N : 1,
                               ValueType::DOUBLE, mr);
    keel::f64 *o = indx.doubleDataMut();
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 xi = sig.elemAsDouble(i);
        keel::usize cnt = 0;
        for (keel::usize k = 0; k < M; ++k) {
            if (partition.elemAsDouble(k) < xi) ++cnt;
            else break;   // partition is monotonic non-decreasing
        }
        o[i] = keel::as<keel::f64>(cnt);
    }
    return indx;
}

QuantizResult
quantiz(const Value &sig, const Value &partition, const Value &codebook,
        keel::Allocator *mr)
{
    quantiz_argcheck(partition, codebook);
    const keel::usize N = sig.numel();

    Value indx = quantiz_indx(sig, partition, mr);
    const keel::f64 *idx = indx.doubleData();

    Value quantv = Value::matrix(indx.dims().rows(), indx.dims().cols(),
                                 ValueType::DOUBLE, mr);
    keel::f64 *q = quantv.doubleDataMut();
    keel::f64 sse = 0.0;
    for (keel::usize i = 0; i < N; ++i) {
        const keel::usize k = keel::as<keel::usize>(idx[i]);
        const keel::f64 cv = codebook.elemAsDouble(k);
        q[i] = cv;
        const keel::f64 d = sig.elemAsDouble(i) - cv;
        sse += d * d;
    }
    return {std::move(indx), std::move(quantv),
            N > 0 ? sse / keel::as<keel::f64>(N) : 0.0};
}

} // namespace numkit::comm
