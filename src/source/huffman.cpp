// toolboxes/comm/src/source/huffman.cpp
//
// Huffman dictionary builder, encoder, and decoder delegating pure
// coding kernels to keel::comm.

#include <keel/arena.h>
#include <keel/array.h>
#include <keel/comm.h>
#include <keel/core.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <numkit/comm/source/huffman.hpp>

#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

namespace numkit::comm {

namespace {

void huffmandict_argcheck(const Value &symbols, const Value &probs)
{
    const keel::usize K = symbols.numel();
    if (K == 0)
        throw Error("huffmandict: symbols must be non-empty",
                    0, 0, "huffmandict", "", "numkit:huffmandict:Empty");
    if (probs.numel() != K)
        throw Error("huffmandict: probs must match length of symbols",
                    0, 0, "huffmandict", "", "numkit:huffmandict:LenMismatch");

    keel::f64 sum = 0.0;
    for (keel::usize k = 0; k < K; ++k) {
        const keel::f64 p = probs.elemAsDouble(k);
        if (!(p >= 0.0 && p <= 1.0))
            throw Error("huffmandict: probabilities must lie in [0, 1]",
                        0, 0, "huffmandict", "", "numkit:huffmandict:InvalidProb");
        sum += p;
    }
    if (keel::abs(sum - 1.0) > keel::sqrt(keel::Limits<keel::f64>::kEpsilon))
        throw Error("huffmandict: probabilities must sum to 1",
                    0, 0, "huffmandict", "", "numkit:huffmandict:InvalidProbSum");
}

struct FlatDict {
    keel::Array<keel::f64> syms;
    keel::Array<keel::u32> offsets;
    keel::Array<keel::u8> bits;
};

FlatDict readDict(const Value &dict, keel::Allocator *alloc)
{
    if (dict.dims().cols() != 2)
        throw Error("huffman codec: dict must be K-by-2 cell",
                    0, 0, "huffman", "", "numkit:huffman:DictShape");
    const keel::usize K = dict.dims().rows();
    FlatDict fd{
        keel::Array<keel::f64>(K, 0.0, alloc),
        keel::Array<keel::u32>(K + 1u, 0u, alloc),
        keel::Array<keel::u8>(alloc),
    };
    for (keel::usize k = 0; k < K; ++k) {
        fd.syms[k] = dict.cellAt(k).toScalar();
        const Value &codeV = dict.cellAt(k + K);
        const keel::usize L = codeV.numel();
        fd.offsets[k] = keel::as<keel::u32>(fd.bits.size());
        for (keel::usize b = 0; b < L; ++b) {
            const keel::f64 v = codeV.elemAsDouble(b);
            if (v != 0.0 && v != 1.0)
                throw Error("huffman codec: dict codes must contain 0/1",
                            0, 0, "huffman", "", "numkit:huffman:DictBits");
            fd.bits.push_back(keel::as<keel::u8>(v));
        }
    }
    fd.offsets[K] = keel::as<keel::u32>(fd.bits.size());
    return fd;
}

bool isRowOriented(const Value &v)
{
    return v.dims().rows() == 1 && v.dims().cols() >= 1;
}

} // namespace

std::pair<Value, keel::f64>
huffmandict(const Value &symbols, const Value &probs, keel::Allocator *mr)
{
    huffmandict_argcheck(symbols, probs);
    const keel::usize K = symbols.numel();

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> pv(K, 0.0, scratch.allocator());
    for (keel::usize k = 0; k < K; ++k) pv[k] = probs.elemAsDouble(k);

    keel::comm::HuffmanDict hd;
    (void)keel::comm::huffman_dict(pv.as_ref(), hd, scratch.allocator());

    Value dict = Value::cell(K, 2, mr);
    for (keel::usize k = 0; k < K; ++k) {
        dict.cellAt(k) = Value::scalar(symbols.elemAsDouble(k), mr);
        const keel::u32 beg = hd.offsets[k];
        const keel::u32 end = hd.offsets[k + 1u];
        const keel::usize L = keel::as<keel::usize>(end - beg);
        Value code = Value::matrix(1, L, ValueType::DOUBLE, mr);
        keel::f64 *cd = code.doubleDataMut();
        for (keel::usize i = 0; i < L; ++i)
            cd[i] = keel::as<keel::f64>(hd.bits[keel::as<keel::usize>(beg) + i]);
        dict.cellAt(K + k) = std::move(code);
    }
    return {std::move(dict), hd.avg_len};
}

Value huffmanenco(const Value &sig, const Value &dict, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const FlatDict fd = readDict(dict, scratch.allocator());

    const keel::usize N = sig.numel();
    keel::Array<keel::usize> sym_idx(N, 0u, scratch.allocator());
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 s = sig.elemAsDouble(i);
        bool found = false;
        for (keel::usize k = 0; k < fd.syms.size(); ++k) {
            if (fd.syms[k] == s) {
                sym_idx[i] = k;
                found = true;
                break;
            }
        }
        if (!found)
            throw Error("huffmanenco: symbol not in dict",
                        0, 0, "huffmanenco", "", "numkit:huffmanenco:UnknownSym");
    }

    keel::Array<keel::u8> encoded(scratch.allocator());
    (void)keel::comm::huffman_encode(sym_idx.as_ref(), fd.offsets.as_ref(), fd.bits.as_ref(),
                                     encoded, scratch.allocator());

    const bool row = isRowOriented(sig);
    const keel::usize M = encoded.size();
    Value out = Value::matrix(row ? 1 : M, row ? M : 1, ValueType::DOUBLE, mr);
    keel::f64 *o = out.doubleDataMut();
    for (keel::usize i = 0; i < M; ++i) o[i] = keel::as<keel::f64>(encoded[i]);
    return out;
}

Value huffmandeco(const Value &bits, const Value &dict, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const FlatDict fd = readDict(dict, scratch.allocator());

    const keel::usize N = bits.numel();
    keel::Array<keel::u8> bvec(N, 0u, scratch.allocator());
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 v = bits.elemAsDouble(i);
        if (v != 0.0 && v != 1.0)
            throw Error("huffmandeco: input bits must be 0 or 1",
                        0, 0, "huffmandeco", "", "numkit:huffmandeco:NonBit");
        bvec[i] = keel::as<keel::u8>(v);
    }

    keel::Array<keel::usize> syms(scratch.allocator());
    const auto st = keel::comm::huffman_decode(bvec.as_ref(), fd.offsets.as_ref(), fd.bits.as_ref(),
                                               syms, scratch.allocator());
    if (st != keel::comm::CommStatus::kOk)
        throw Error("huffmandeco: bit pattern does not match dict",
                    0, 0, "huffmandeco", "", "numkit:huffmandeco:NoMatch");

    const bool row = isRowOriented(bits);
    const keel::usize M = syms.size();
    Value out = Value::matrix(row ? 1 : M, row ? M : 1, ValueType::DOUBLE, mr);
    keel::f64 *o = out.doubleDataMut();
    for (keel::usize i = 0; i < M; ++i) o[i] = fd.syms[syms[i]];
    return out;
}

} // namespace numkit::comm
