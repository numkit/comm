// toolboxes/comm/src/source/lloyds.cpp
//
// Lloyd-Max scalar quantizer designer.
//
//   [partition, codebook, distor, rel] = lloyds(training, ini_codebook [, tol])
//
// Algorithm: alternate centroid update (each codebook entry becomes the
// mean of its bin) with partition update (midpoint between consecutive
// codebook entries) until the relative distortion change drops below
// `tol` (default 1e-7) or the absolute distortion drops below
// eps*max(training).
//
// `ini_codebook` may be either an integer K (initialise K linspace-
// centred codebook entries across the training range) or a vector
// (used directly, sorted ascending).
//
// Plotting form (4-arg) deferred -- nk doesn't ship plot().

#include <keel/array.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/math.h>
#include <keel/ref.h>
#include <numkit/comm/source/lloyds.hpp>
#include <numkit/comm/source/quantiz.hpp>

#include <numkit/value/argcheck.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace numkit::comm {

namespace {

// Fill out_v with double values sourced from the training-set Value.
void readVector(const Value &v, keel::Array<keel::f64> &out)
{
    const keel::usize N = v.numel();
    out.resize(N);
    for (keel::usize i = 0; i < N; ++i) out[i] = v.elemAsDouble(i);
}

// Build a Value(1, n, DOUBLE) from a double array on `mr`.
Value vecToRow(keel::Allocator *mr, keel::ArrayRef<keel::f64> v)
{
    Value out = Value::matrix(1, v.size(), ValueType::DOUBLE, mr);
    if (!v.empty()) std::copy(v.begin(), v.end(), out.doubleDataMut());
    return out;
}

void lloyds_argcheck(const Value &training_set, const Value &ini_codebook,
                     keel::Array<keel::f64> &training,
                     keel::Array<keel::f64> &codebook,
                     keel::f64 &min_train, keel::f64 &max_train)
{
    readVector(training_set, training);
    if (training.empty())
        throw Error("lloyds: training set must be non-empty",
                    0, 0, "lloyds", "", "numkit:lloyds:EmptyTRAINING_SET");
    min_train = *std::min_element(training.begin(), training.end());
    max_train = *std::max_element(training.begin(), training.end());
    if (max_train <= min_train)
        throw Error("lloyds: training set must span a positive range",
                    0, 0, "lloyds", "", "numkit:lloyds:InvalidTRAINING_SET");

    if (ini_codebook.numel() == 1) {
        const keel::f64 Kd = ini_codebook.toScalar();
        const long K = keel::as<long>(std::floor(Kd));
        if (K < 1)
            throw Error("lloyds: ini_codebook count must be positive",
                        0, 0, "lloyds", "",
                        "numkit:lloyds:NonPositiveINI_CODEBOOK");
        const keel::f64 step = (max_train - min_train) / keel::as<keel::f64>(K);
        codebook.resize(K);
        for (long i = 0; i < K; ++i) {
            codebook[keel::as<keel::usize>(i)] =
                min_train + step * 0.5 + step * keel::as<keel::f64>(i);
        }
    } else {
        readVector(ini_codebook, codebook);
        std::sort(codebook.begin(), codebook.end());
    }
    if (codebook.empty())
        throw Error("lloyds: ini_codebook must have at least 1 entry",
                    0, 0, "lloyds", "", "numkit:lloyds:InvalidINI_CODEBOOK");
}

} // namespace

std::tuple<Value, Value, keel::f64, keel::f64>
lloyds(const Value &training_set, const Value &ini_codebook,
       keel::f64 tol, keel::Allocator *mr)
{
    keel::Array<keel::f64> training(mr);
    keel::Array<keel::f64> codebook(mr);
    keel::f64 min_train = 0.0;
    keel::f64 max_train = 0.0;
    lloyds_argcheck(training_set, ini_codebook, training, codebook,
                    min_train, max_train);
    const keel::usize K = codebook.size();

    if (tol <= 0.0) tol = 1e-7;

    // Helper: midpoint partition from codebook.
    auto midpoint = [&]() -> keel::Array<keel::f64> {
        keel::Array<keel::f64> p(K > 0 ? K - 1 : 0, mr);
        for (keel::usize i = 0; i + 1 < K; ++i) {
            p[i] = 0.5 * (codebook[i] + codebook[i + 1]);
        }
        std::sort(p.begin(), p.end());
        return p;
    };

    // Helper: classify training samples into bins given partition.
    // bin index = sum(partition < x).
    auto classify = [&](keel::ArrayRef<keel::f64> part,
                        keel::Array<keel::usize> &bins) {
        bins.resize(training.size());
        for (keel::usize i = 0; i < training.size(); ++i) {
            const keel::f64 x = training[i];
            keel::usize b = 0;
            while (b < part.size() && part[b] < x) ++b;
            bins[i] = b;
        }
    };

    // Helper: distortion = mean((x - codebook(bin))^2).
    auto distortion = [&](keel::ArrayRef<keel::usize> bins) {
        keel::f64 sse = 0.0;
        for (keel::usize i = 0; i < training.size(); ++i) {
            const keel::f64 d = training[i] - codebook[bins[i]];
            sse += d * d;
        }
        return sse / keel::as<keel::f64>(training.size());
    };

    keel::Array<keel::f64> partition = midpoint();
    keel::Array<keel::usize> bins(mr);
    classify(partition, bins);
    keel::f64 distor      = distortion(bins);
    keel::f64 last_distor = 0.0;

    const keel::f64 ter_cond2 = std::numeric_limits<keel::f64>::epsilon()
                             * keel::abs(max_train);
    keel::f64 rel_distor = (distor > ter_cond2)
                            ? keel::abs(distor - last_distor) / distor
                            : distor;

    // Iterate.
    while (rel_distor > tol && rel_distor > ter_cond2) {
        // Centroid update per bin.
        for (keel::usize i = 0; i < K; ++i) {
            keel::f64 sum = 0.0;
            keel::usize cnt = 0;
            for (keel::usize j = 0; j < training.size(); ++j) {
                if (bins[j] == i) { sum += training[j]; ++cnt; }
            }
            if (cnt > 0) {
                codebook[i] = sum / keel::as<keel::f64>(cnt);
            } else {
                // Empty bin: per MATLAB lloyds.m, fall back to a midpoint
                // between adjacent partition boundaries (or training extrema).
                if (i == 0) {
                    if (partition.empty()) {
                        codebook[i] = (max_train + min_train) * 0.5;
                    } else {
                        // Try mean of training_set <= partition[0]
                        keel::f64 s = 0.0; keel::usize c = 0;
                        for (keel::f64 x : training)
                            if (x <= partition[0]) { s += x; ++c; }
                        codebook[i] = c ? s / c
                                        : 0.5 * (partition[0] + min_train);
                    }
                } else if (i == K - 1) {
                    keel::f64 s = 0.0; keel::usize c = 0;
                    for (keel::f64 x : training)
                        if (x >= partition[i - 1]) { s += x; ++c; }
                    codebook[i] = c ? s / c
                                    : 0.5 * (max_train + partition[i - 1]);
                } else {
                    keel::f64 s = 0.0; keel::usize c = 0;
                    for (keel::f64 x : training)
                        if (x >= partition[i - 1] && x <= partition[i]) {
                            s += x; ++c;
                        }
                    codebook[i] = c ? s / c
                                    : 0.5 * (partition[i] + partition[i - 1]);
                }
            }
        }
        partition = midpoint();
        classify(partition, bins);
        last_distor = distor;
        distor      = distortion(bins);
        rel_distor  = (distor > ter_cond2)
                          ? keel::abs(distor - last_distor) / distor
                          : distor;
    }

    return {vecToRow(mr, partition), vecToRow(mr, codebook),
            distor, rel_distor};
}

} // namespace numkit::comm
