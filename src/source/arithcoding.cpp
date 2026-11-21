// toolboxes/comm/src/source/arithcoding.cpp
//
// Arithmetic coding encoder + decoder delegating to keel::comm.

#include <keel/arena.h>
#include <keel/array.h>
#include <keel/comm.h>
#include <keel/core.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <numkit/comm/source/arithcoding.hpp>

#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

namespace numkit::comm {

namespace {

bool isRow(const Value &v)
{
    return v.dims().rows() == 1 && v.dims().cols() >= 1;
}

void readCounts(const Value &v, keel::Array<keel::u64> &out)
{
    const keel::usize n = v.numel();
    out.resize(n);
    for (keel::usize i = 0; i < n; ++i) {
        const keel::f64 d = v.elemAsDouble(i);
        if (!(d > 0.0) || keel::floor(d) != d)
            throw Error("arithenco: counts must be positive integers",
                        0, 0, "arithenco", "", "numkit:arithenco:Counts");
        out[i] = keel::as<keel::u64>(d);
    }
}

} // namespace

Value arithenco(const Value &seq, const Value &counts, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::u64> cnt(scratch.allocator());
    readCounts(counts, cnt);
    if (cnt.empty())
        throw Error("arithenco: counts must be non-empty",
                    0, 0, "arithenco", "", "numkit:arithenco:Counts");

    const keel::usize K = cnt.size();
    const keel::usize Nseq = seq.numel();
    keel::Array<keel::usize> svec(Nseq, 0u, scratch.allocator());
    for (keel::usize k = 0; k < Nseq; ++k) {
        const keel::f64 sd = seq.elemAsDouble(k);
        if (!(sd > 0.0) || keel::floor(sd) != sd || keel::as<keel::usize>(sd) > K)
            throw Error("arithenco: symbol out of range [1, length(counts)]",
                        0, 0, "arithenco", "", "numkit:arithenco:Sym");
        svec[k] = keel::as<keel::usize>(sd);
    }

    keel::Array<keel::u8> bits(scratch.allocator());
    (void)keel::comm::arith_encode(svec.as_ref(), cnt.as_ref(), bits, scratch.allocator());

    const bool row = isRow(seq);
    const keel::usize L = bits.size();
    Value out = Value::matrix(row ? 1 : L, row ? L : 1, ValueType::DOUBLE, mr);
    keel::f64 *o = out.doubleDataMut();
    for (keel::usize i = 0; i < L; ++i) o[i] = keel::as<keel::f64>(bits[i]);
    return out;
}

Value arithdeco(const Value &code, const Value &counts, keel::usize len, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::u64> cnt(scratch.allocator());
    readCounts(counts, cnt);

    keel::u64 total = 0ULL;
    for (keel::usize i = 0; i < cnt.size(); ++i) total += cnt[i];
    const keel::i32 wordLen = keel::as<keel::i32>(keel::ceil(keel::log2(keel::as<keel::f64>(total)))) + 2;

    const keel::usize Lcode = code.numel();
    if (Lcode < keel::as<keel::usize>(wordLen))
        throw Error("arithdeco: code shorter than required word length",
                    0, 0, "arithdeco", "", "numkit:arithdeco:Short");

    keel::Array<keel::u8> cbits(Lcode, 0u, scratch.allocator());
    for (keel::usize i = 0; i < Lcode; ++i) {
        const keel::f64 d = code.elemAsDouble(i);
        if (d != 0.0 && d != 1.0)
            throw Error("arithdeco: code bits must be 0 or 1",
                        0, 0, "arithdeco", "", "numkit:arithdeco:Bit");
        cbits[i] = keel::as<keel::u8>(d);
    }

    keel::Array<keel::usize> dseq(len, 0u, scratch.allocator());
    (void)keel::comm::arith_decode(cbits.as_ref(), cnt.as_ref(), len, dseq.as_mut_ref(),
                                   scratch.allocator());

    const bool row = isRow(code);
    Value out = Value::matrix(row ? 1 : len, row ? len : 1, ValueType::DOUBLE, mr);
    keel::f64 *o = out.doubleDataMut();
    for (keel::usize i = 0; i < len; ++i) o[i] = keel::as<keel::f64>(dseq[i]);
    return out;
}

} // namespace numkit::comm
