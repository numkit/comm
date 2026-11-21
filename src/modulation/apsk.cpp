// toolboxes/comm/src/modulation/apsk.cpp
//
// Amplitude-Phase Shift Keying (multi-ring constellation):
//   y = apskmod(x, M, radii [, phaseoffset [, mapping]])
//   z = apskdemod(y, M, radii [, phaseoffset [, mapping]])
//
// `M`     : per-ring symbol counts (vector); total constellation size N = sum(M).
// `radii` : per-ring radius (same length as M).
// `phaseoffset` : per-ring phase offset (defaults to pi ./ M).
// `mapping` : optional length-N permutation of indices [0..N-1].
//             Default identity (`0:N-1`). Gray defaults are deferred --
//             MATLAB's per-ring Gray for non-power-of-2 needs more probing.
//
// Constellation point at logical index `idx` (after mapping):
//   ring r is the smallest with idx < cum(r);    k = idx - cum(r-1)
//   point = radii[r] * exp(i * (phaseoffset[r] + 2π·k / M[r]))

#include <keel/array.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/complex.h>
#include <numkit/comm/modulation/apsk.hpp>

#include <numkit/value/argcheck.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <cmath>
#include <limits>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace numkit::comm {

namespace {

// Build the natural-order constellation: for each ring r, its M[r]
// points at radii[r] * exp(i * (phaseoffset[r] + 2π·k/M[r])).
keel::Array<keel::c64> buildConstellation(keel::ArrayRef<keel::usize> M,
                                   keel::ArrayRef<keel::f64> radii,
                                   keel::ArrayRef<keel::f64> phaseoffset,
                                   keel::Allocator *mr)
{
    const keel::usize R = M.size();
    keel::usize N = 0;
    for (keel::usize m : M) N += m;
    keel::Array<keel::c64> C(mr);
    C.reserve(N);
    for (keel::usize r = 0; r < R; ++r) {
        const keel::usize Mr = M[r];
        const keel::f64 radr = radii[r];
        const keel::f64 pho  = phaseoffset[r];
        for (keel::usize k = 0; k < Mr; ++k) {
            const keel::f64 theta = pho + 2.0 * M_PI * keel::as<keel::f64>(k) / keel::as<keel::f64>(Mr);
            C.emplace_back(radr * std::cos(theta), radr * std::sin(theta));
        }
    }
    return C;
}

void readVecInt(const Value &v, keel::Array<int> &out)
{
    const keel::usize n = v.numel();
    out.resize(n);
    for (keel::usize i = 0; i < n; ++i)
        out[i] = keel::as<int>(v.elemAsDouble(i));
}

void readVec(const Value &v, keel::Array<keel::f64> &out)
{
    const keel::usize n = v.numel();
    out.resize(n);
    for (keel::usize i = 0; i < n; ++i) out[i] = v.elemAsDouble(i);
}

// `mapping` is a length-N permutation: when the user wants symbol idx
// to map to constellation point `mapping(idx) + 1` (1-based MATLAB)
// or `mapping[idx]` (0-based). We invert it so that
// constellation_index = invMapping[user_index].
keel::Array<int> invertMapping(keel::ArrayRef<int> mapping, keel::Allocator *mr)
{
    const keel::usize N = mapping.size();
    keel::Array<int> inv(N, -1, mr);
    for (keel::usize i = 0; i < N; ++i) {
        if (mapping[i] < 0 || keel::as<keel::usize>(mapping[i]) >= N)
            throw Error("apsk: SymbolMapping out of range",
                        0, 0, "apsk", "", "numkit:apsk:Mapping");
        inv[keel::as<keel::usize>(mapping[i])] = keel::as<int>(i);
    }
    return inv;
}

void apsk_argcheck(keel::ArrayRef<keel::usize> M, keel::ArrayRef<keel::f64> radii,
                   const Value &phaseoffset_v, const Value &mapping_v,
                   const char *fn)
{
    if (M.empty() || M.size() != radii.size())
        throw Error(keel::format("{}: M and radii must be vectors of same length", fn),
                    0, 0, fn, "", keel::format("numkit:{}:DimMismatch", fn));
    if (!phaseoffset_v.isEmpty()
        && phaseoffset_v.numel() != 1
        && phaseoffset_v.numel() != M.size())
        throw Error(keel::format("{}: phaseoffset length must match M", fn),
                    0, 0, fn, "", keel::format("numkit:{}:PhaseLen", fn));
    keel::usize Ntot = 0;
    for (keel::usize m : M) Ntot += m;
    if (!mapping_v.isEmpty() && mapping_v.numel() != Ntot)
        throw Error(keel::format("{}: SymbolMapping length must equal sum(M)", fn),
                    0, 0, fn, "", keel::format("numkit:{}:MappingLen", fn));
}

} // namespace

Value apskmod(const Value &x, keel::ArrayRef<keel::usize> M, keel::ArrayRef<keel::f64> radii,
              const Value &phaseoffset_v, const Value &mapping_v,
              keel::Allocator *mr)
{
    apsk_argcheck(M, radii, phaseoffset_v, mapping_v, "apskmod");
    keel::Array<keel::f64> phaseoffset(M.size(), mr);
    if (!phaseoffset_v.isEmpty()) {
        readVec(phaseoffset_v, phaseoffset);
        if (phaseoffset.size() == 1) {
            // Scalar broadcast.
            const keel::f64 v = phaseoffset[0];
            phaseoffset.assign(M.size(), v);
        } else if (phaseoffset.size() != M.size()) {
            throw Error("apskmod: phaseoffset length must match M",
                        0, 0, "apskmod", "", "numkit:apskmod:PhaseLen");
        }
    } else {
        for (keel::usize r = 0; r < M.size(); ++r)
            phaseoffset[r] = M_PI / keel::as<keel::f64>(M[r]);
    }

    keel::usize Ntot = 0;
    for (keel::usize m : M) Ntot += m;

    keel::Array<keel::c64> C = buildConstellation(M, radii, phaseoffset, mr);

    // Mapping: identity by default; otherwise length-Ntot permutation.
    keel::Array<int> invmap(mr);
    if (!mapping_v.isEmpty()) {
        keel::Array<int> mapping(mr);
        readVecInt(mapping_v, mapping);
        if (mapping.size() != Ntot)
            throw Error("apskmod: SymbolMapping length must equal sum(M)",
                        0, 0, "apskmod", "", "numkit:apskmod:MappingLen");
        invmap = invertMapping(mapping, mr);
    } else {
        invmap.resize(Ntot);
        for (keel::usize i = 0; i < Ntot; ++i) invmap[i] = keel::as<int>(i);
    }

    Value out = Value::matrix(x.dims().rows(), x.dims().cols(),
                              ValueType::COMPLEX, mr);
    keel::c64 *o = out.complexDataMut();
    const keel::usize Nx = x.numel();
    for (keel::usize i = 0; i < Nx; ++i) {
        const long xi = keel::as<long>(x.elemAsDouble(i));
        if (xi < 0 || keel::as<keel::usize>(xi) >= Ntot)
            throw Error("apskmod: input index out of range [0, sum(M)-1]",
                        0, 0, "apskmod", "", "numkit:apskmod:OutOfRange");
        o[i] = C[keel::as<keel::usize>(invmap[keel::as<keel::usize>(xi)])];
    }
    return out;
}

Value apskdemod(const Value &y, keel::ArrayRef<keel::usize> M, keel::ArrayRef<keel::f64> radii,
                const Value &phaseoffset_v, const Value &mapping_v,
                keel::Allocator *mr)
{
    apsk_argcheck(M, radii, phaseoffset_v, mapping_v, "apskdemod");
    keel::Array<keel::f64> phaseoffset(M.size(), mr);
    if (!phaseoffset_v.isEmpty()) {
        readVec(phaseoffset_v, phaseoffset);
        if (phaseoffset.size() == 1) {
            const keel::f64 v = phaseoffset[0];
            phaseoffset.assign(M.size(), v);
        } else if (phaseoffset.size() != M.size()) {
            throw Error("apskdemod: phaseoffset length must match M",
                        0, 0, "apskdemod", "", "numkit:apskdemod:PhaseLen");
        }
    } else {
        for (keel::usize r = 0; r < M.size(); ++r)
            phaseoffset[r] = M_PI / keel::as<keel::f64>(M[r]);
    }

    keel::usize Ntot = 0;
    for (keel::usize m : M) Ntot += m;
    keel::Array<keel::c64> C = buildConstellation(M, radii, phaseoffset, mr);

    keel::Array<int> mapping(mr);
    if (!mapping_v.isEmpty()) {
        readVecInt(mapping_v, mapping);
        if (mapping.size() != Ntot)
            throw Error("apskdemod: SymbolMapping length must equal sum(M)",
                        0, 0, "apskdemod", "", "numkit:apskdemod:MappingLen");
    } else {
        mapping.resize(Ntot);
        for (keel::usize i = 0; i < Ntot; ++i) mapping[i] = keel::as<int>(i);
    }

    Value out = Value::matrix(y.dims().rows(), y.dims().cols(),
                              ValueType::DOUBLE, mr);
    keel::f64 *o = out.doubleDataMut();
    const keel::usize Ny = y.numel();
    const bool y_complex = y.isComplex();
    for (keel::usize i = 0; i < Ny; ++i) {
        const keel::c64 yi = y_complex
                          ? y.complexData()[i]
                          : keel::c64(y.elemAsDouble(i), 0.0);
        // Find nearest constellation point (by squared distance).
        keel::usize best_k = 0;
        keel::f64 best_d2 = std::numeric_limits<keel::f64>::infinity();
        for (keel::usize k = 0; k < C.size(); ++k) {
            const keel::f64 dr = yi.real() - C[k].real();
            const keel::f64 di = yi.imag() - C[k].imag();
            const keel::f64 d2 = dr * dr + di * di;
            if (d2 < best_d2) { best_d2 = d2; best_k = k; }
        }
        // Map constellation index back through user mapping.
        o[i] = keel::as<keel::f64>(mapping[best_k]);
    }
    return out;
}

} // namespace numkit::comm
