// toolboxes/comm/src/source/random_source.cpp
//
// Random data sources: randsrc.
// Future: randerr planned for the same TU (random binary error matrix).

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <numkit/comm/source/random_source.hpp>

#include <numkit/builtin/datafun.hpp>
#include <numkit/ops/matlab_mt19937.hpp>
#include <numkit/ops/rng_context.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <mutex>
#include <numeric>

namespace numkit::comm {

namespace {

// Gather alphabet + cumulative-probability arrays from the alphabet
// argument. MATLAB accepts:
//   row vector       -> alphabet, uniform probabilities
//   2-row matrix     -> [alphabet; probabilities]
// Returns the symbol count K via len_out.
void parseAlphabet(const Value &a,
                   keel::Allocator *mr,
                   keel::c64 **alpha_out, keel::f64 **prob_out, keel::usize *len_out,
                   bool *is_complex_out)
{
    const keel::usize R = a.dims().rows();
    const keel::usize C = a.dims().cols();
    const bool a_cplx = a.isComplex();

    if (R == 1 && C >= 1) {
        // Plain alphabet row -> uniform probs.
        const keel::usize K = C;
        auto *al = static_cast<keel::c64 *>(
            mr->alloc_bytes(K * sizeof(keel::c64), alignof(keel::c64)));
        auto *pr = static_cast<keel::f64 *>(
            mr->alloc_bytes(K * sizeof(keel::f64), alignof(keel::f64)));
        bool any_imag = false;
        for (keel::usize k = 0; k < K; ++k) {
            al[k] = a_cplx ? a.complexElem(k) : keel::c64(a.elemAsDouble(k), 0.0);
            if (al[k].imag() != 0.0) any_imag = true;
            pr[k] = keel::as<keel::f64>(k + 1) / keel::as<keel::f64>(K);
        }
        *alpha_out      = al;
        *prob_out       = pr;
        *len_out        = K;
        *is_complex_out = a_cplx && any_imag;
        return;
    }

    if (R == 2 && C >= 1) {
        const keel::usize K = C;
        auto *al = static_cast<keel::c64 *>(
            mr->alloc_bytes(K * sizeof(keel::c64), alignof(keel::c64)));
        auto *pr = static_cast<keel::f64 *>(
            mr->alloc_bytes(K * sizeof(keel::f64), alignof(keel::f64)));
        // Column-major: alphabet[k] = a(0, k) at index 0 + k*R = k*2;
        // prob[k] = a(1, k) at index 1 + k*2.
        keel::f64 sum = 0.0;
        bool any_imag = false;
        for (keel::usize k = 0; k < K; ++k) {
            const keel::c64 sym = a_cplx ? a.complexElem(k * R + 0)
                                       : keel::c64(a.elemAsDouble(k * R + 0), 0.0);
            const keel::c64 pkc = a_cplx ? a.complexElem(k * R + 1)
                                       : keel::c64(a.elemAsDouble(k * R + 1), 0.0);
            const keel::f64 pk = pkc.real();
            if (pkc.imag() != 0.0 || !(pk >= 0.0 && pk <= 1.0))
                throw Error("randsrc: probabilities must be in [0, 1]",
                            0, 0, "randsrc", "",
                            "numkit:randsrc:InvalidProbabilityVal");
            al[k] = sym;
            if (sym.imag() != 0.0) any_imag = true;
            sum  += pk;
            pr[k] = sum;          // cumulative
        }
        if (keel::abs(sum - 1.0) > std::sqrt(2.220446049250313e-16))
            throw Error("randsrc: probabilities must sum to 1",
                        0, 0, "randsrc", "",
                        "numkit:randsrc:InvalidProbabilitySum");
        *alpha_out      = al;
        *prob_out       = pr;
        *len_out        = K;
        *is_complex_out = a_cplx && any_imag;
        return;
    }

    if (R == 1 && C == 0)
        throw Error("randsrc: alphabet must be non-empty",
                    0, 0, "randsrc", "", "numkit:randsrc:EmptyAlphabet");

    throw Error("randsrc: alphabet must be a row vector or 2-row matrix",
                0, 0, "randsrc", "", "numkit:randsrc:InvalidAlphabet");
}

void randsrc_argcheck(const Value &a,
                      keel::Allocator *mr,
                      keel::c64 **alpha_out, keel::f64 **prob_out, keel::usize *len_out,
                      bool *is_complex_out)
{
    parseAlphabet(a, mr, alpha_out, prob_out, len_out, is_complex_out);
}

} // namespace

Value randsrc(::numkit::ops::RngContext &rng, keel::usize m, keel::usize n, const Value &alphabet,
              bool have_state, keel::u32 state,
              keel::Allocator *mr)
{
    using ::numkit::ops::MatlabMT19937;

    // Parse alphabet → cumulative probability table.
    keel::c64 *alpha     = nullptr;
    keel::f64  *prob      = nullptr;
    keel::usize   K         = 0;
    bool     is_cplx   = false;
    randsrc_argcheck(alphabet, mr, &alpha, &prob, &K, &is_cplx);

    const keel::usize N = m * n;
    if (is_cplx) {
        Value out = Value::matrix(m, n, ValueType::COMPLEX, mr);
        keel::c64 *o = out.complexDataMut();
        if (have_state) {
            MatlabMT19937 local;
            local.seed(state);
            for (keel::usize i = 0; i < N; ++i) {
                const keel::f64 r = local.genRes53();
                keel::usize idx = 0;
                while (idx < K - 1 && r >= prob[idx]) ++idx;
                o[i] = alpha[idx];
            }
        } else {
            auto &gen = rng;
            for (keel::usize i = 0; i < N; ++i) {
                const keel::f64 r = gen.genRes53();
                keel::usize idx = 0;
                while (idx < K - 1 && r >= prob[idx]) ++idx;
                o[i] = alpha[idx];
            }
        }
        return out;
    }

    Value out = Value::matrix(m, n, ValueType::DOUBLE, mr);
    keel::f64 *o = out.doubleDataMut();

    // Choose generator. With explicit state -> dedicated engine for
    // reproducibility (matches MATLAB's RandStream(...,'Seed',state)).
    if (have_state) {
        MatlabMT19937 local;
        local.seed(state);
        for (keel::usize i = 0; i < N; ++i) {
            const keel::f64 r = local.genRes53();
            // idx = 1 + sum(r >= prob(k)) → (1-based) → 0-based = sum(r >= prob).
            keel::usize idx = 0;
            while (idx < K - 1 && r >= prob[idx]) ++idx;
            o[i] = alpha[idx].real();
        }
    } else {
        auto &gen = rng;
        for (keel::usize i = 0; i < N; ++i) {
            const keel::f64 r = gen.genRes53();
            keel::usize idx = 0;
            while (idx < K - 1 && r >= prob[idx]) ++idx;
            o[i] = alpha[idx].real();
        }
    }

    return out;
}

// ── randerr ────────────────────────────────────────────────────────
// Random bit-error matrix. Per row i:
//   - draw the error count from the probability CDF;
//   - choose that many random column positions by sorting a vector of
//     random keys and marking the lowest-keyed columns (a standard
//     random-subset selection).
//
// `errspec` is one of:
//   - scalar k                            -> exactly k errors per row
//   - row vector [k1 k2 ...]              -> uniform pick from {k1, k2, ...}
//   - 2-row matrix [counts; probabilities] -> weighted pick
namespace {

void parseErrspec(const Value &e, keel::Allocator *mr,
                  keel::i32 **counts_out, keel::f64 **prob_out, keel::usize *len_out,
                  keel::usize n)
{
    const keel::usize R = e.dims().rows();
    const keel::usize C = e.dims().cols();

    if (R == 1 && C >= 1) {
        const keel::usize K = C;
        auto *cs = static_cast<keel::i32 *>(
            mr->alloc_bytes(K * sizeof(keel::i32), alignof(keel::i32)));
        auto *pr = static_cast<keel::f64 *>(
            mr->alloc_bytes(K * sizeof(keel::f64), alignof(keel::f64)));
        for (keel::usize k = 0; k < K; ++k) {
            const keel::f64 v = e.elemAsDouble(k);
            if (!(v >= 0.0) || std::floor(v) != v)
                throw Error("randerr: error counts must be non-negative integers",
                            0, 0, "randerr", "",
                            "numkit:randerr:InvalidErrorsElements");
            if (keel::as<keel::usize>(v) > n)
                throw Error("randerr: error count exceeds n",
                            0, 0, "randerr", "",
                            "numkit:randerr:InvalidErrorsForm");
            cs[k] = keel::as<keel::i32>(v);
            pr[k] = keel::as<keel::f64>(k + 1) / keel::as<keel::f64>(K);
        }
        *counts_out = cs;
        *prob_out   = pr;
        *len_out    = K;
        return;
    }

    if (R == 2 && C >= 1) {
        const keel::usize K = C;
        auto *cs = static_cast<keel::i32 *>(
            mr->alloc_bytes(K * sizeof(keel::i32), alignof(keel::i32)));
        auto *pr = static_cast<keel::f64 *>(
            mr->alloc_bytes(K * sizeof(keel::f64), alignof(keel::f64)));
        keel::f64 sum = 0.0;
        for (keel::usize k = 0; k < K; ++k) {
            const keel::f64 v = e.elemAsDouble(k * R + 0);
            const keel::f64 p = e.elemAsDouble(k * R + 1);
            if (!(v >= 0.0) || std::floor(v) != v)
                throw Error("randerr: error counts must be non-negative integers",
                            0, 0, "randerr", "",
                            "numkit:randerr:InvalidErrorsElements");
            if (keel::as<keel::usize>(v) > n)
                throw Error("randerr: error count exceeds n",
                            0, 0, "randerr", "",
                            "numkit:randerr:InvalidErrorsForm");
            if (!(p >= 0.0 && p <= 1.0))
                throw Error("randerr: probabilities must be in [0, 1]",
                            0, 0, "randerr", "",
                            "numkit:randerr:Invalid2ndRowErrorsVal");
            cs[k] = keel::as<keel::i32>(v);
            sum += p;
            pr[k] = sum;
        }
        if (keel::abs(sum - 1.0) > std::sqrt(2.220446049250313e-16))
            throw Error("randerr: probabilities must sum to 1",
                        0, 0, "randerr", "",
                        "numkit:randerr:InvalidProbabilitySum");
        *counts_out = cs;
        *prob_out   = pr;
        *len_out    = K;
        return;
    }

    throw Error("randerr: errors must be a row vector or 2-row matrix",
                0, 0, "randerr", "", "numkit:randerr:InvalidErrorsDims");
}

void randerr_argcheck(const Value &e, keel::Allocator *mr,
                      keel::i32 **counts_out, keel::f64 **prob_out, keel::usize *len_out,
                      keel::usize n)
{
    parseErrspec(e, mr, counts_out, prob_out, len_out, n);
}

template <typename Rng>
void fillOneRow(keel::f64 *o_col_major, keel::usize i, keel::usize m, keel::usize n,
                const keel::i32 *counts, const keel::f64 *prob, keel::usize K, Rng &rng,
                keel::Array<keel::f64> &rs, keel::Array<keel::usize> &p)
{
    // 1) Pick error count from CDF.
    const keel::f64 u = rng.genRes53();
    keel::usize k = 0;
    while (k < K - 1 && u >= prob[k]) ++k;
    const keel::i32 num = counts[k];

    // 2) n random values + sort permutation.
    for (keel::usize j = 0; j < n; ++j) rs[j] = rng.genRes53();
    std::iota(p.begin(), p.end(), keel::usize{0});
    std::sort(p.begin(), p.end(),
              [&](keel::usize a, keel::usize b) { return rs[a] < rs[b]; });

    // 3) out(i, j) = (p[j] < num)  (0-based, mirroring p<=num 1-based).
    for (keel::usize j = 0; j < n; ++j) {
        const bool one = (p[j] < keel::as<keel::usize>(num));
        o_col_major[i + j * m] = one ? 1.0 : 0.0;
    }
}

} // namespace

Value randerr(::numkit::ops::RngContext &rng, keel::usize m, keel::usize n, const Value &errspec,
              bool have_state, keel::u32 state,
              keel::Allocator *mr)
{
    using ::numkit::ops::MatlabMT19937;

    keel::i32    *counts = nullptr;
    keel::f64 *prob   = nullptr;
    keel::usize  K      = 0;
    randerr_argcheck(errspec, mr, &counts, &prob, &K, n);

    Value out = Value::matrix(m, n, ValueType::DOUBLE, mr);
    keel::f64 *o = out.doubleDataMut();
    std::fill(o, o + m * n, 0.0);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> rs(n, scratch.allocator());
    keel::Array<keel::usize> p(n, scratch.allocator());

    if (have_state) {
        MatlabMT19937 local;
        local.seed(state);
        for (keel::usize i = 0; i < m; ++i)
            fillOneRow(o, i, m, n, counts, prob, K, local, rs, p);
    } else {
        auto &gen = rng;
        for (keel::usize i = 0; i < m; ++i)
            fillOneRow(o, i, m, n, counts, prob, K, gen, rs, p);
    }

    return out;
}

} // namespace numkit::comm
