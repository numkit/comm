// toolboxes/comm/src/modulation/fsk_ofdm.cpp
//
// FSK and OFDM modulation / demodulation delegating pure kernels to keel::comm.

#include <keel/arena.h>
#include <keel/array.h>
#include <keel/comm.h>
#include <keel/complex.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <numkit/comm/modulation/fsk_ofdm.hpp>

#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

namespace numkit::comm {

namespace {

keel::comm::SymbolOrder parseOrder(keel::StringRef order)
{
    return (order == "bin") ? keel::comm::SymbolOrder::kBinary
                            : keel::comm::SymbolOrder::kGray;
}

void fsk_argcheck(keel::i32 M, keel::f64 freq_sep, keel::i32 nsamp, keel::f64 fs, const char *fn)
{
    if (M < 2)
        throw Error(keel::format("{}: M must be >= 2", fn), 0, 0, fn, "",
                    keel::format("numkit:{}:badM", fn));
    if (freq_sep <= 0.0 || nsamp <= 0 || fs <= 0.0)
        throw Error(keel::format("{}: freq_sep / nsamp / fs must be positive", fn),
                    0, 0, fn, "", keel::format("numkit:{}:badparam", fn));
}

} // anonymous

Value fskmod(const Value &x, keel::i32 M, keel::f64 freq_sep, keel::i32 nsamp,
             keel::f64 fs, keel::StringRef phase_continuity,
             keel::StringRef symbol_order, keel::Allocator *mr)
{
    fsk_argcheck(M, freq_sep, nsamp, fs, "fskmod");
    const keel::usize Nsym = x.numel();
    const keel::usize Nout = Nsym * keel::as<keel::usize>(nsamp);
    Value out = Value::matrix(Nout, 1, ValueType::COMPLEX, mr);
    if (Nout == 0) return out;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::i32> syms(Nsym, 0, scratch.allocator());
    for (keel::usize i = 0; i < Nsym; ++i) syms[i] = keel::as<keel::i32>(x.elemAsDouble(i));

    (void)keel::comm::fsk_mod(syms.as_ref(), M, freq_sep, nsamp, fs, phase_continuity == "cont",
                              parseOrder(symbol_order),
                              keel::MutArrayRef<keel::c64>(out.complexDataMut(), Nout));
    return out;
}

Value fskdemod(const Value &y, keel::i32 M, keel::f64 freq_sep, keel::i32 nsamp,
               keel::f64 fs, keel::StringRef symbol_order, keel::Allocator *mr)
{
    fsk_argcheck(M, freq_sep, nsamp, fs, "fskdemod");
    const keel::usize Nsamps = y.numel();
    const keel::usize Nsym = Nsamps / keel::as<keel::usize>(nsamp);
    Value out = Value::matrix(Nsym, 1, ValueType::DOUBLE, mr);
    if (Nsym == 0) return out;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::c64> samples(Nsamps, keel::c64(0.0, 0.0), scratch.allocator());
    if (y.type() == ValueType::COMPLEX) {
        const keel::c64 *yd = y.complexData();
        for (keel::usize i = 0; i < Nsamps; ++i) samples[i] = yd[i];
    } else {
        for (keel::usize i = 0; i < Nsamps; ++i) samples[i] = keel::c64(y.elemAsDouble(i), 0.0);
    }

    keel::Array<keel::i32> syms(Nsym, 0, scratch.allocator());
    (void)keel::comm::fsk_demod(samples.as_ref(), M, freq_sep, nsamp, fs, parseOrder(symbol_order),
                                syms.as_mut_ref(), scratch.allocator());
    keel::f64 *od = out.doubleDataMut();
    for (keel::usize i = 0; i < Nsym; ++i) od[i] = keel::as<keel::f64>(syms[i]);
    return out;
}

Value ofdmmod(const Value &in, keel::i32 nfft, keel::i32 cplen, keel::Allocator *mr)
{
    if (nfft <= 0 || cplen < 0)
        throw Error("ofdmmod: bad nfft / cplen", 0, 0, "ofdmmod", "", "numkit:ofdmmod:badparam");
    if (keel::as<keel::i32>(in.dims().rows()) != nfft)
        throw Error("ofdmmod: in must have nfft rows", 0, 0, "ofdmmod", "", "numkit:ofdmmod:size");

    const keel::usize nfft_u = keel::as<keel::usize>(nfft);
    const keel::usize cp_u = keel::as<keel::usize>(cplen);
    const keel::usize nsym = in.dims().cols();
    const keel::usize nout = (nfft_u + cp_u) * nsym;
    Value out = Value::matrix(nout, 1, ValueType::COMPLEX, mr);
    if (nsym == 0) return out;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const keel::usize nin = nfft_u * nsym;
    keel::Array<keel::c64> in_c(nin, keel::c64(0.0, 0.0), scratch.allocator());
    if (in.type() == ValueType::COMPLEX) {
        const keel::c64 *src = in.complexData();
        for (keel::usize i = 0; i < nin; ++i) in_c[i] = src[i];
    } else {
        for (keel::usize i = 0; i < nin; ++i) in_c[i] = keel::c64(in.elemAsDouble(i), 0.0);
    }

    (void)keel::comm::ofdm_mod(in_c.as_ref(), nfft_u, nsym, cp_u,
                               keel::MutArrayRef<keel::c64>(out.complexDataMut(), nout),
                               scratch.allocator());
    return out;
}

Value ofdmdemod(const Value &in, keel::i32 nfft, keel::i32 cplen, keel::i32 symoffset,
                keel::Allocator *mr)
{
    if (nfft <= 0 || cplen < 0)
        throw Error("ofdmdemod: bad nfft / cplen", 0, 0, "ofdmdemod", "", "numkit:ofdmdemod:badparam");
    const keel::usize nfft_u = keel::as<keel::usize>(nfft);
    const keel::usize cp_u = keel::as<keel::usize>(cplen);
    const keel::usize out_per_sym = nfft_u + cp_u;
    if (in.numel() % out_per_sym != 0)
        throw Error("ofdmdemod: input length must be a multiple of (nfft + cplen)",
                    0, 0, "ofdmdemod", "", "numkit:ofdmdemod:size");

    if (symoffset < 0 || symoffset > cplen) symoffset = cplen;
    const keel::usize nsamps = in.numel();
    const keel::usize nsym = nsamps / out_per_sym;

    Value out = Value::matrix(nfft_u, nsym, ValueType::COMPLEX, mr);
    if (nsym == 0) return out;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::c64> in_c(nsamps, keel::c64(0.0, 0.0), scratch.allocator());
    if (in.type() == ValueType::COMPLEX) {
        const keel::c64 *src = in.complexData();
        for (keel::usize i = 0; i < nsamps; ++i) in_c[i] = src[i];
    } else {
        for (keel::usize i = 0; i < nsamps; ++i) in_c[i] = keel::c64(in.elemAsDouble(i), 0.0);
    }

    (void)keel::comm::ofdm_demod(in_c.as_ref(), nfft_u, nsym, cp_u, keel::as<keel::usize>(symoffset),
                                 keel::MutArrayRef<keel::c64>(out.complexDataMut(), nfft_u * nsym),
                                 scratch.allocator());
    return out;
}

} // namespace numkit::comm
