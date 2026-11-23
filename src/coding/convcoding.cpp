// toolboxes/comm/src/coding/convcoding.cpp
//
// Convolutional coding (poly2trellis, convenc, vitdec, istrellis),
// delegating pure numerical kernels to keel::comm.

#include <keel/arena.h>
#include <keel/array.h>
#include <keel/comm.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <numkit/comm/coding/convcoding.hpp>

#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

namespace numkit::comm {

namespace {

struct TrellisBuffers {
    keel::usize numStates = 0;
    keel::usize numOutputSymbols = 0;
    keel::i32 n = 0;
    keel::Array<keel::u32> nextStates;
    keel::Array<keel::u32> outputs;
};

TrellisBuffers readTrellis(const Value &t, const char *who, keel::Allocator *alloc)
{
    if (!t.isStruct() || t.numel() != 1)
        throw Error(keel::format("{}: trellis must be a 1x1 struct", who),
                    0, 0, who, "", keel::format("numkit:{}:trellis", who));
    const auto &el = t.structArrayElem(0);
    auto field = [&](const char *f) -> const Value & {
        const Value *v = el.get(f);
        if (!v)
            throw Error(keel::format("{}: trellis is missing field '{}'", who, f),
                        0, 0, who, "", keel::format("numkit:{}:trellis", who));
        return *v;
    };

    TrellisBuffers tb;
    tb.numStates        = keel::as<keel::usize>(field("numStates").toScalar());
    tb.numOutputSymbols = keel::as<keel::usize>(field("numOutputSymbols").toScalar());
    tb.n                = keel::as<keel::i32>(keel::round(keel::log2(keel::as<keel::f64>(tb.numOutputSymbols))));
    const Value &nsV    = field("nextStates");
    const Value &ouV    = field("outputs");
    const keel::usize sz = tb.numStates * 2u;
    tb.nextStates = keel::Array<keel::u32>(sz, 0u, alloc);
    tb.outputs    = keel::Array<keel::u32>(sz, 0u, alloc);
    const keel::f64 *nsd = nsV.doubleData();
    const keel::f64 *oud = ouV.doubleData();
    for (keel::usize i = 0; i < sz; ++i) {
        tb.nextStates[i] = keel::as<keel::u32>(nsd[i]);
        tb.outputs[i]    = keel::as<keel::u32>(oud[i]);
    }
    return tb;
}

} // namespace

Value poly2trellis(const Value &constraintLength, const Value &codeGenerator, keel::Allocator *mr)
{
    if (constraintLength.numel() != 1)
        throw Error("poly2trellis: only a scalar ConstraintLength (rate 1/n) "
                    "is supported in this revision (rate k/n deferred)",
                    0, 0, "poly2trellis", "", "numkit:poly2trellis:rate");
    const keel::i64 K = keel::as<keel::i64>(constraintLength.toScalar());
    if (K < 1)
        throw Error("poly2trellis: ConstraintLength must be >= 1",
                    0, 0, "poly2trellis", "", "numkit:poly2trellis:K");

    const keel::usize n = codeGenerator.numel();
    if (n == 0)
        throw Error("poly2trellis: CodeGenerator must be non-empty",
                    0, 0, "poly2trellis", "", "numkit:poly2trellis:gen");

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::u64> gens(n, 0ULL, scratch.allocator());
    for (keel::usize i = 0; i < n; ++i)
        gens[i] = keel::as<keel::u64>(codeGenerator.elemAsDouble(i));

    keel::comm::Trellis tr;
    const auto st = keel::comm::poly2trellis(keel::as<keel::i32>(K), gens.as_ref(), tr,
                                             scratch.allocator());
    if (st != keel::comm::CommStatus::kOk)
        throw Error("poly2trellis: CodeGenerator entries must be octal numbers (digits 0-7)",
                    0, 0, "poly2trellis", "", "numkit:poly2trellis:octal");

    Value ns = Value::matrix(tr.num_states, tr.num_input_symbols, ValueType::DOUBLE, mr);
    Value ou = Value::matrix(tr.num_states, tr.num_input_symbols, ValueType::DOUBLE, mr);
    keel::f64 *nsd = ns.doubleDataMut();
    keel::f64 *oud = ou.doubleDataMut();
    for (keel::usize i = 0; i < tr.next_states.size(); ++i) {
        nsd[i] = keel::as<keel::f64>(tr.next_states[i]);
        oud[i] = keel::as<keel::f64>(tr.outputs[i]);
    }

    Value t = Value::structArray(1, 1, mr);
    auto &el = t.structArrayElem(0);
    el.put(keel::String::from("numInputSymbols"),  Value::scalar(keel::as<keel::f64>(tr.num_input_symbols),  mr));
    el.put(keel::String::from("numOutputSymbols"), Value::scalar(keel::as<keel::f64>(tr.num_output_symbols), mr));
    el.put(keel::String::from("numStates"),        Value::scalar(keel::as<keel::f64>(tr.num_states),         mr));
    el.put(keel::String::from("nextStates"), keel::move(ns));
    el.put(keel::String::from("outputs"),    keel::move(ou));
    return t;
}

Value convenc(const Value &msg, const Value &trellis, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const TrellisBuffers tb = readTrellis(trellis, "convenc", scratch.allocator());
    const keel::usize L = msg.numel();
    const keel::usize outLen = L * keel::as<keel::usize>(tb.n);
    const bool col = (msg.dims().cols() == 1 && msg.dims().rows() > 1);
    Value out = col ? Value::matrix(outLen, 1, ValueType::DOUBLE, mr)
                    : Value::matrix(1, outLen, ValueType::DOUBLE, mr);
    if (outLen == 0) return out;

    keel::Array<keel::u8> in_bits(L, 0u, scratch.allocator());
    for (keel::usize i = 0; i < L; ++i) in_bits[i] = (msg.elemAsDouble(i) != 0.0) ? 1u : 0u;
    keel::Array<keel::u8> out_bits(outLen, 0u, scratch.allocator());
    (void)keel::comm::convenc(in_bits.as_ref(), tb.nextStates.as_ref(), tb.outputs.as_ref(),
                              tb.numStates, tb.n, out_bits.as_mut_ref());

    keel::f64 *od = out.doubleDataMut();
    for (keel::usize i = 0; i < outLen; ++i) od[i] = keel::as<keel::f64>(out_bits[i]);
    return out;
}

Value vitdec(const Value &code, const Value &trellis, keel::i64 /*tblen*/,
             keel::StringRef opmode, keel::StringRef dectype, keel::Allocator *mr)
{
    auto lower = [](keel::StringRef sv) {
        keel::String s(sv);
        s.to_ascii_lower_in_place();
        return s;
    };
    const keel::String dt = lower(dectype);
    if (dt != "hard")
        throw Error("vitdec: only hard-decision decoding is supported in this "
                    "revision (soft/unquant deferred)",
                    0, 0, "vitdec", "", "numkit:vitdec:dectype");
    const keel::String mode = lower(opmode);
    if (mode != "trunc" && mode != "term")
        throw Error("vitdec: opmode must be 'trunc' or 'term' in this revision "
                    "('cont' deferred)",
                    0, 0, "vitdec", "", "numkit:vitdec:opmode");

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const TrellisBuffers tb = readTrellis(trellis, "vitdec", scratch.allocator());
    const keel::usize total = code.numel();
    if (tb.n <= 0 || (total % keel::as<keel::usize>(tb.n)) != 0)
        throw Error("vitdec: code length must be a multiple of log2(numOutputSymbols)",
                    0, 0, "vitdec", "", "numkit:vitdec:len");

    const keel::usize L = total / keel::as<keel::usize>(tb.n);
    const bool col = (code.dims().cols() == 1 && code.dims().rows() > 1);
    Value out = col ? Value::matrix(L, 1, ValueType::DOUBLE, mr)
                    : Value::matrix(1, L, ValueType::DOUBLE, mr);
    if (L == 0) return out;

    keel::Array<keel::u8> in_bits(total, 0u, scratch.allocator());
    for (keel::usize i = 0; i < total; ++i) in_bits[i] = (code.elemAsDouble(i) != 0.0) ? 1u : 0u;
    keel::Array<keel::u8> out_bits(L, 0u, scratch.allocator());
    (void)keel::comm::vitdec_hard(in_bits.as_ref(), tb.nextStates.as_ref(), tb.outputs.as_ref(),
                                  tb.numStates, tb.n, mode == "term", out_bits.as_mut_ref(),
                                  scratch.allocator());

    keel::f64 *od = out.doubleDataMut();
    for (keel::usize i = 0; i < L; ++i) od[i] = keel::as<keel::f64>(out_bits[i]);
    return out;
}

Value istrellis(const Value &S, keel::Allocator *mr)
{
    auto no = [&]() { return Value::logicalScalar(false, mr); };
    auto isPow2 = [](keel::f64 v) {
        if (v < 1.0 || v != keel::floor(v)) return false;
        const keel::u64 u = keel::as<keel::u64>(v);
        return (u & (u - 1ULL)) == 0ULL;
    };
    if (!S.isStruct() || S.numel() != 1) return no();
    const auto &el = S.structArrayElem(0);
    const Value *nIvPtr = el.get("numInputSymbols");
    const Value *nOvPtr = el.get("numOutputSymbols");
    const Value *nSvPtr = el.get("numStates");
    const Value *nsPtr  = el.get("nextStates");
    const Value *ouPtr  = el.get("outputs");
    if (!nIvPtr || !nOvPtr || !nSvPtr || !nsPtr || !ouPtr) return no();

    if (!nIvPtr->isScalar() || !nOvPtr->isScalar() || !nSvPtr->isScalar()) return no();
    const keel::f64 nI = nIvPtr->toScalar(), nO = nOvPtr->toScalar(), nS = nSvPtr->toScalar();
    if (!isPow2(nI) || !isPow2(nO) || !isPow2(nS)) return no();

    if (nsPtr->isStruct() || nsPtr->isCell() || ouPtr->isStruct() || ouPtr->isCell()) return no();
    const keel::usize Sn = keel::as<keel::usize>(nS);
    const keel::usize In = keel::as<keel::usize>(nI);
    if (nsPtr->dims().rows() != Sn || nsPtr->dims().cols() != In ||
        ouPtr->dims().rows() != Sn || ouPtr->dims().cols() != In)
        return no();

    const keel::usize N = Sn * In;
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 v = nsPtr->elemAsDouble(i);
        if (v < 0.0 || v >= nS || v != keel::floor(v)) return no();
        const keel::f64 w = ouPtr->elemAsDouble(i);
        if (w < 0.0 || w >= nO || w != keel::floor(w)) return no();
    }
    return Value::logicalScalar(true, mr);
}

} // namespace numkit::comm
