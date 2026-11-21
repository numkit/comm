// toolboxes/comm/src/modulation/apsk.cpp
//
// Multi-ring APSK modulation / demodulation delegating constellation
// construction and nearest-point slicing to keel::comm.

#include <keel/arena.h>
#include <keel/array.h>
#include <keel/comm.h>
#include <keel/complex.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <numkit/comm/modulation/apsk.hpp>

#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

namespace numkit::comm {

namespace {

void apsk_argcheck(keel::ArrayRef<keel::usize> M, keel::ArrayRef<keel::f64> radii,
                   const Value &phaseoffset_v, const Value &mapping_v, const char *fn)
{
    if (M.empty() || M.size() != radii.size())
        throw Error(keel::format("{}: M and radii must be vectors of same length", fn),
                    0, 0, fn, "", keel::format("numkit:{}:DimMismatch", fn));
    if (!phaseoffset_v.isEmpty() && phaseoffset_v.numel() != 1 && phaseoffset_v.numel() != M.size())
        throw Error(keel::format("{}: phaseoffset length must match M", fn),
                    0, 0, fn, "", keel::format("numkit:{}:PhaseLen", fn));
    keel::usize Ntot = 0;
    for (keel::usize m : M) Ntot += m;
    if (!mapping_v.isEmpty() && mapping_v.numel() != Ntot)
        throw Error(keel::format("{}: SymbolMapping length must equal sum(M)", fn),
                    0, 0, fn, "", keel::format("numkit:{}:MappingLen", fn));
}

keel::Array<keel::f64> readPhases(const Value &v, keel::Allocator *alloc)
{
    const keel::usize n = v.numel();
    keel::Array<keel::f64> out(n, 0.0, alloc);
    for (keel::usize i = 0; i < n; ++i) out[i] = v.elemAsDouble(i);
    return out;
}

keel::Array<keel::i32> readMapping(const Value &v, keel::usize Ntot, keel::Allocator *alloc)
{
    keel::Array<keel::i32> out(Ntot, 0, alloc);
    if (!v.isEmpty()) {
        for (keel::usize i = 0; i < Ntot; ++i) {
            const keel::i32 m = keel::as<keel::i32>(v.elemAsDouble(i));
            if (m < 0 || keel::as<keel::usize>(m) >= Ntot)
                throw Error("apsk: SymbolMapping out of range",
                            0, 0, "apsk", "", "numkit:apsk:Mapping");
            out[i] = m;
        }
    } else {
        for (keel::usize i = 0; i < Ntot; ++i) out[i] = keel::as<keel::i32>(i);
    }
    return out;
}

} // namespace

Value apskmod(const Value &x, keel::ArrayRef<keel::usize> M, keel::ArrayRef<keel::f64> radii,
              const Value &phaseoffset_v, const Value &mapping_v, keel::Allocator *mr)
{
    apsk_argcheck(M, radii, phaseoffset_v, mapping_v, "apskmod");
    keel::usize Ntot = 0;
    for (keel::usize m : M) Ntot += m;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const auto phases = readPhases(phaseoffset_v, scratch.allocator());
    keel::Array<keel::c64> C(scratch.allocator());
    (void)keel::comm::apsk_constellation(M, radii, phases.as_ref(), C, scratch.allocator());

    const auto mapping = readMapping(mapping_v, Ntot, scratch.allocator());
    keel::Array<keel::c64> mapped_C(Ntot, keel::c64(0.0, 0.0), scratch.allocator());
    for (keel::usize i = 0; i < Ntot; ++i) {
        mapped_C[keel::as<keel::usize>(mapping[i])] = C[i];
    }

    const keel::usize Nx = x.numel();
    keel::Array<keel::i32> syms(Nx, 0, scratch.allocator());
    for (keel::usize i = 0; i < Nx; ++i) {
        const keel::i64 xi = keel::as<keel::i64>(x.elemAsDouble(i));
        if (xi < 0 || keel::as<keel::usize>(xi) >= Ntot)
            throw Error("apskmod: input index out of range [0, sum(M)-1]",
                        0, 0, "apskmod", "", "numkit:apskmod:OutOfRange");
        syms[i] = keel::as<keel::i32>(xi);
    }

    Value out = Value::matrix(x.dims().rows(), x.dims().cols(), ValueType::COMPLEX, mr);
    (void)keel::comm::genqam_mod(syms.as_ref(), mapped_C.as_ref(),
                                 keel::MutArrayRef<keel::c64>(out.complexDataMut(), Nx));
    return out;
}

Value apskdemod(const Value &y, keel::ArrayRef<keel::usize> M, keel::ArrayRef<keel::f64> radii,
                const Value &phaseoffset_v, const Value &mapping_v, keel::Allocator *mr)
{
    apsk_argcheck(M, radii, phaseoffset_v, mapping_v, "apskdemod");
    keel::usize Ntot = 0;
    for (keel::usize m : M) Ntot += m;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const auto phases = readPhases(phaseoffset_v, scratch.allocator());
    keel::Array<keel::c64> C(scratch.allocator());
    (void)keel::comm::apsk_constellation(M, radii, phases.as_ref(), C, scratch.allocator());
    const auto mapping = readMapping(mapping_v, Ntot, scratch.allocator());

    const keel::usize Ny = y.numel();
    keel::Array<keel::c64> samples(Ny, keel::c64(0.0, 0.0), scratch.allocator());
    if (y.isComplex()) {
        const keel::c64 *yd = y.complexData();
        for (keel::usize i = 0; i < Ny; ++i) samples[i] = yd[i];
    } else {
        for (keel::usize i = 0; i < Ny; ++i) samples[i] = keel::c64(y.elemAsDouble(i), 0.0);
    }

    keel::Array<keel::i32> idx(Ny, 0, scratch.allocator());
    (void)keel::comm::genqam_demod(samples.as_ref(), C.as_ref(), idx.as_mut_ref());

    Value out = Value::matrix(y.dims().rows(), y.dims().cols(), ValueType::DOUBLE, mr);
    keel::f64 *o = out.doubleDataMut();
    for (keel::usize i = 0; i < Ny; ++i)
        o[i] = keel::as<keel::f64>(mapping[keel::as<keel::usize>(idx[i])]);
    return out;
}

} // namespace numkit::comm
