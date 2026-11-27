// modules/comm/src/register/comm_system_objects.cpp
//
// Communications Toolbox System objects and RF/visualization helpers:
//   comm.BarkerCode
//   comm.DBPSKModulator / comm.DBPSKDemodulator
//   comm.QPSKModulator  / comm.QPSKDemodulator
//   comm.RaisedCosineTransmitFilter / comm.RaisedCosineReceiveFilter
//   comm.AWGNChannel
//   comm.CRCGenerator / comm.CRCDetector
//   comm.OFDMModulator / comm.OFDMDemodulator
//   iqimbal, eyediagram

#include <keel/array.h>
#include <keel/complex.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <keel/number.h>
#include <keel/ref.h>
#include <keel/smart_ptr.h>
#include <keel/string.h>
#include <keel/utf8.h>

#include <numkit/ops/rng_context.hpp>
#include <numkit/sys/plugin.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/object.hpp>
#include <numkit/value/value.hpp>

namespace numkit::comm::detail {

namespace {

constexpr keel::f64 kPi = 3.14159265358979323846;

keel::Allocator *allocOf(const CallContext &ctx) noexcept
{
    return ctx.engine ? ctx.engine->allocator() : keel::default_allocator();
}

keel::i32 roundI32(keel::f64 v) noexcept
{
    return keel::as<keel::i32>(keel::round_i64(v));
}

bool callExt(CallContext &ctx, keel::StringRef name, keel::ArrayRef<Value> args,
             keel::usize nargout, keel::MutArrayRef<Value> outs)
{
    if (!ctx.engine) return false;
    const ExternalFunc *fn = ctx.engine->findExternal(name, ctx.env);
    if (!fn) return false;
    (*fn)(args, nargout, outs, ctx);
    return true;
}

void addCloneMethod(BuiltinClass &cls)
{
    keel::String cname = cls.name.clone();
    cls.methods.put(keel::String::from("clone"),
                    [cname = keel::move(cname)](Value &self, keel::ArrayRef<Value>, keel::usize,
                                                keel::MutArrayRef<Value> outs, CallContext &ctx) {
                        auto *mr = allocOf(ctx);
                        if (const ObjectState *os = self.objectStateConst())
                            outs[0] = Value::object(cname, os->deepCopy(mr), mr);
                    });
}

// ── 1. comm.BarkerCode ──────────────────────────────────────────────────
struct BarkerPayload final : public NativePayload {
    keel::i32 length = 7;
    keel::i32 samplesPerFrame = 7;
    keel::String outputDataType;
    keel::usize cursor = 0;

    BarkerPayload()
        : outputDataType(keel::String::from("double"))
    {}

    keel::Arc<NativePayload> clone(keel::Allocator *alloc = nullptr) const override
    {
        auto *a = alloc ? alloc : keel::default_allocator();
        auto p = keel::make_arc_in<BarkerPayload>(a);
        p->length = length;
        p->samplesPerFrame = samplesPerFrame;
        p->outputDataType = outputDataType.clone();
        p->cursor = cursor;
        return p;
    }
};

keel::ArrayRef<keel::i32> barkerSeqFor(keel::i32 len)
{
    static constexpr keel::i32 kB1[]  = {-1};
    static constexpr keel::i32 kB2[]  = {-1, 1};
    static constexpr keel::i32 kB3[]  = {-1, -1, 1};
    static constexpr keel::i32 kB4[]  = {-1, -1, 1, -1};
    static constexpr keel::i32 kB5[]  = {-1, -1, -1, 1, -1};
    static constexpr keel::i32 kB7[]  = {-1, -1, -1, 1, 1, -1, 1};
    static constexpr keel::i32 kB11[] = {-1, -1, -1, 1, 1, 1, -1, 1, 1, -1, 1};
    static constexpr keel::i32 kB13[] = {-1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1};
    switch (len) {
    case 1:  return {kB1, 1};
    case 2:  return {kB2, 2};
    case 3:  return {kB3, 3};
    case 4:  return {kB4, 4};
    case 5:  return {kB5, 5};
    case 7:  return {kB7, 7};
    case 11: return {kB11, 11};
    case 13: return {kB13, 13};
    default:
        throw Error("comm.BarkerCode: Length must be one of 1, 2, 3, 4, 5, 7, 11, or 13.",
                    0, 0, "comm.BarkerCode", "", "numkit:comm:InvalidBarkerLength");
    }
}

void barkerApplyNV(BarkerPayload &p, keel::ArrayRef<Value> args)
{
    for (keel::usize i = 0; i + 1 < args.size(); i += 2) {
        if (!args[i].isChar() && !args[i].isString()) continue;
        const keel::String k = args[i].toString();
        if (k.equals_ignore_ascii_case("Length")) {
            p.length = roundI32(args[i + 1].toScalar());
            (void)barkerSeqFor(p.length);
            p.cursor = 0;
        } else if (k.equals_ignore_ascii_case("SamplesPerFrame")) {
            p.samplesPerFrame = keel::max(1, roundI32(args[i + 1].toScalar()));
        } else if (k.equals_ignore_ascii_case("OutputDataType")) {
            p.outputDataType = args[i + 1].toString();
        }
    }
}

void barkerStepImpl(Value &self, keel::ArrayRef<Value>, keel::usize,
                    keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    auto *mr = allocOf(ctx);
    ObjectState *os = self.objectStateMut();
    auto &p = *static_cast<BarkerPayload *>(os->native.get());
    const auto seq = barkerSeqFor(p.length);
    const keel::usize n = keel::as<keel::usize>(keel::max(1, p.samplesPerFrame));
    Value y = Value::matrix(n, 1, ValueType::DOUBLE, mr);
    for (keel::usize i = 0; i < n; ++i) {
        y.doubleDataMut()[i] = keel::as<keel::f64>(seq[p.cursor % seq.size()]);
        p.cursor = (p.cursor + 1) % seq.size();
    }
    outs[0] = keel::move(y);
}

BuiltinClass makeBarkerClass()
{
    BuiltinClass cls;
    cls.name = keel::String::from("comm.BarkerCode");
    cls.isHandle = true;
    cls.construct = [](keel::ArrayRef<Value> args, CallContext &ctx) -> Value {
        auto *mr = allocOf(ctx);
        auto payload = keel::make_arc_in<BarkerPayload>(mr);
        barkerApplyNV(*payload, args);
        auto st = keel::make_arc_in<ObjectState>(mr, mr);
        st->native = keel::move(payload);
        return Value::object("comm.BarkerCode", keel::move(st), mr);
    };
    cls.subsref = &barkerStepImpl;
    cls.methods.put(keel::String::from("step"), &barkerStepImpl);
    cls.methods.put(keel::String::from("reset"),
                    [](Value &self, keel::ArrayRef<Value>, keel::usize,
                       keel::MutArrayRef<Value> outs, CallContext &) {
                        if (ObjectState *os = self.objectStateMut(); os && os->native)
                            static_cast<BarkerPayload *>(os->native.get())->cursor = 0;
                        if (!outs.empty()) outs[0] = Value();
                    });
    cls.methods.put(keel::String::from("release"),
                    [](Value &self, keel::ArrayRef<Value>, keel::usize,
                       keel::MutArrayRef<Value> outs, CallContext &) {
                        if (ObjectState *os = self.objectStateMut(); os && os->native)
                            static_cast<BarkerPayload *>(os->native.get())->cursor = 0;
                        if (!outs.empty()) outs[0] = Value();
                    });
    cls.propGet = [](const Value &self, keel::StringRef name, Value &out, CallContext &ctx) -> bool {
        auto *mr = allocOf(ctx);
        const auto &p = *static_cast<const BarkerPayload *>(self.objectStateConst()->native.get());
        if (name.equals_ignore_ascii_case("Length")) { out = Value::scalar(keel::as<keel::f64>(p.length), mr); return true; }
        if (name.equals_ignore_ascii_case("SamplesPerFrame")) { out = Value::scalar(keel::as<keel::f64>(p.samplesPerFrame), mr); return true; }
        if (name.equals_ignore_ascii_case("OutputDataType")) { out = Value::fromString(p.outputDataType, mr); return true; }
        return false;
    };
    cls.propSet = [](Value &self, keel::StringRef name, const Value &val, CallContext &) -> bool {
        auto &p = *static_cast<BarkerPayload *>(self.objectStateMut()->native.get());
        if (name.equals_ignore_ascii_case("Length")) { p.length = roundI32(val.toScalar()); (void)barkerSeqFor(p.length); p.cursor = 0; return true; }
        if (name.equals_ignore_ascii_case("SamplesPerFrame")) { p.samplesPerFrame = keel::max(1, roundI32(val.toScalar())); return true; }
        if (name.equals_ignore_ascii_case("OutputDataType")) { p.outputDataType = val.toString(); return true; }
        return false;
    };
    return cls;
}

void commBarkerCode_ctor_reg(keel::ArrayRef<Value> args, keel::usize, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    auto *mr = allocOf(ctx);
    auto payload = keel::make_arc_in<BarkerPayload>(mr);
    barkerApplyNV(*payload, args);
    auto st = keel::make_arc_in<ObjectState>(mr, mr);
    st->native = keel::move(payload);
    outs[0] = Value::object("comm.BarkerCode", keel::move(st), mr);
}

// ── 2. comm.DBPSKModulator & comm.DBPSKDemodulator ──────────────────────
struct DbpskPayload final : public NativePayload {
    keel::f64 phaseRotation = 0.0;
    bool bitIO = false;
    bool isDemod = false;
    keel::f64 prevPhase = 0.0;
    keel::c64 prevSym{1.0, 0.0};

    keel::Arc<NativePayload> clone(keel::Allocator *alloc = nullptr) const override
    {
        auto *a = alloc ? alloc : keel::default_allocator();
        auto p = keel::make_arc_in<DbpskPayload>(a);
        p->phaseRotation = phaseRotation;
        p->bitIO = bitIO;
        p->isDemod = isDemod;
        p->prevPhase = prevPhase;
        p->prevSym = prevSym;
        return p;
    }
};

void dbpskApplyNV(DbpskPayload &p, keel::ArrayRef<Value> args)
{
    keel::usize i = 0;
    if (args.size() == 1 && args[0].isScalar()) {
        p.phaseRotation = args[0].toScalar();
        return;
    }
    for (; i + 1 < args.size(); i += 2) {
        if (!args[i].isChar() && !args[i].isString()) continue;
        const keel::String k = args[i].toString();
        if (k.equals_ignore_ascii_case("PhaseRotation"))
            p.phaseRotation = args[i + 1].toScalar();
        else if (k.equals_ignore_ascii_case("BitInput") || k.equals_ignore_ascii_case("BitOutput"))
            p.bitIO = args[i + 1].toBool();
    }
}

void dbpskStepImpl(Value &self, keel::ArrayRef<Value> args, keel::usize,
                   keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "comm.DBPSK.step");
    auto *mr = allocOf(ctx);
    auto &p = *static_cast<DbpskPayload *>(self.objectStateMut()->native.get());
    const Value &x = args[0];
    const keel::usize nRows = x.dims().rows();
    const keel::usize nCols = x.dims().cols();
    if (!p.isDemod) {
        Value y = Value::matrix(nRows, nCols, ValueType::COMPLEX, mr);
        for (keel::usize c = 0; c < nCols; ++c) {
            keel::f64 ph = (c == 0) ? p.prevPhase : 0.0;
            for (keel::usize r = 0; r < nRows; ++r) {
                const keel::usize idx = r + c * nRows;
                const keel::i32 b = roundI32(x.elemAsDouble(idx)) & 1;
                ph += p.phaseRotation + keel::as<keel::f64>(b) * kPi;
                y.complexDataMut()[idx] = keel::c64{keel::cos(ph), keel::sin(ph)};
            }
            if (c == 0) p.prevPhase = ph;
        }
        outs[0] = keel::move(y);
    } else {
        Value y = Value::matrix(nRows, nCols, ValueType::DOUBLE, mr);
        for (keel::usize c = 0; c < nCols; ++c) {
            keel::c64 prev = (c == 0) ? p.prevSym : keel::c64{1.0, 0.0};
            for (keel::usize r = 0; r < nRows; ++r) {
                const keel::usize idx = r + c * nRows;
                const keel::c64 cur = x.isComplex() ? x.complexData()[idx] : keel::c64{x.elemAsDouble(idx), 0.0};
                // conj(prev) * cur rotated by -phaseRotation
                const keel::c64 prod{prev.re * cur.re + prev.im * cur.im,
                                     prev.re * cur.im - prev.im * cur.re};
                const keel::f64 ang = keel::atan2(prod.im, prod.re) - p.phaseRotation;
                const keel::f64 cang = keel::cos(ang);
                y.doubleDataMut()[idx] = (cang >= 0.0) ? 0.0 : 1.0;
                prev = cur;
            }
            if (c == 0) p.prevSym = prev;
        }
        outs[0] = keel::move(y);
    }
}

BuiltinClass makeDbpskClass(const char *clsName, bool isDemod)
{
    BuiltinClass cls;
    cls.name = keel::String::from(clsName);
    cls.isHandle = true;
    cls.construct = [clsName, isDemod](keel::ArrayRef<Value> args, CallContext &ctx) -> Value {
        auto *mr = allocOf(ctx);
        auto payload = keel::make_arc_in<DbpskPayload>(mr);
        payload->isDemod = isDemod;
        dbpskApplyNV(*payload, args);
        auto st = keel::make_arc_in<ObjectState>(mr, mr);
        st->native = keel::move(payload);
        return Value::object(clsName, keel::move(st), mr);
    };
    cls.subsref = &dbpskStepImpl;
    cls.methods.put(keel::String::from("step"), &dbpskStepImpl);
    auto resetFn = [](Value &self, keel::ArrayRef<Value>, keel::usize,
                      keel::MutArrayRef<Value> outs, CallContext &) {
        auto &p = *static_cast<DbpskPayload *>(self.objectStateMut()->native.get());
        p.prevPhase = 0.0;
        p.prevSym = keel::c64{1.0, 0.0};
        if (!outs.empty()) outs[0] = Value();
    };
    cls.methods.put(keel::String::from("reset"), resetFn);
    cls.methods.put(keel::String::from("release"), resetFn);
    cls.propGet = [](const Value &self, keel::StringRef name, Value &out, CallContext &ctx) -> bool {
        auto *mr = allocOf(ctx);
        const auto &p = *static_cast<const DbpskPayload *>(self.objectStateConst()->native.get());
        if (name.equals_ignore_ascii_case("PhaseRotation")) { out = Value::scalar(p.phaseRotation, mr); return true; }
        if (name.equals_ignore_ascii_case("BitInput") || name.equals_ignore_ascii_case("BitOutput")) { out = Value::logicalScalar(p.bitIO, mr); return true; }
        return false;
    };
    cls.propSet = [](Value &self, keel::StringRef name, const Value &val, CallContext &) -> bool {
        auto &p = *static_cast<DbpskPayload *>(self.objectStateMut()->native.get());
        if (name.equals_ignore_ascii_case("PhaseRotation")) { p.phaseRotation = val.toScalar(); return true; }
        if (name.equals_ignore_ascii_case("BitInput") || name.equals_ignore_ascii_case("BitOutput")) { p.bitIO = val.toBool(); return true; }
        return false;
    };
    return cls;
}

void commDbpskMod_ctor_reg(keel::ArrayRef<Value> args, keel::usize, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    auto *mr = allocOf(ctx);
    auto payload = keel::make_arc_in<DbpskPayload>(mr);
    payload->isDemod = false;
    dbpskApplyNV(*payload, args);
    auto st = keel::make_arc_in<ObjectState>(mr, mr);
    st->native = keel::move(payload);
    outs[0] = Value::object("comm.DBPSKModulator", keel::move(st), mr);
}

void commDbpskDemod_ctor_reg(keel::ArrayRef<Value> args, keel::usize, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    auto *mr = allocOf(ctx);
    auto payload = keel::make_arc_in<DbpskPayload>(mr);
    payload->isDemod = true;
    dbpskApplyNV(*payload, args);
    auto st = keel::make_arc_in<ObjectState>(mr, mr);
    st->native = keel::move(payload);
    outs[0] = Value::object("comm.DBPSKDemodulator", keel::move(st), mr);
}

// ── 3. comm.QPSKModulator & comm.QPSKDemodulator ────────────────────────
struct QpskPayload final : public NativePayload {
    keel::f64 phaseOffset = kPi / 4.0;
    bool bitIO = false;
    bool gray = true;
    bool isDemod = false;

    keel::Arc<NativePayload> clone(keel::Allocator *alloc = nullptr) const override
    {
        auto *a = alloc ? alloc : keel::default_allocator();
        auto p = keel::make_arc_in<QpskPayload>(a);
        p->phaseOffset = phaseOffset;
        p->bitIO = bitIO;
        p->gray = gray;
        p->isDemod = isDemod;
        return p;
    }
};

void qpskApplyNV(QpskPayload &p, keel::ArrayRef<Value> args)
{
    for (keel::usize i = 0; i + 1 < args.size(); i += 2) {
        if (!args[i].isChar() && !args[i].isString()) continue;
        const keel::String k = args[i].toString();
        if (k.equals_ignore_ascii_case("PhaseOffset"))
            p.phaseOffset = args[i + 1].toScalar();
        else if (k.equals_ignore_ascii_case("BitInput") || k.equals_ignore_ascii_case("BitOutput"))
            p.bitIO = args[i + 1].toBool();
        else if (k.equals_ignore_ascii_case("SymbolMapping"))
            p.gray = !args[i + 1].toString().equals_ignore_ascii_case("Binary");
    }
}

void qpskStepImpl(Value &self, keel::ArrayRef<Value> args, keel::usize,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "comm.QPSK.step");
    auto *mr = allocOf(ctx);
    const auto &p = *static_cast<const QpskPayload *>(self.objectStateMut()->native.get());
    const Value &x = args[0];
    const keel::usize nRows = x.dims().rows();
    const keel::usize nCols = x.dims().cols();
    // Gray mapping for M=4: sym 0->0, 1->1, 2->3, 3->2 (angles 0, pi/2, 3pi/2, pi + phaseOffset)
    static constexpr keel::i32 kGrayToIdx[4] = {0, 1, 3, 2};
    static constexpr keel::i32 kIdxToGray[4] = {0, 1, 3, 2};

    if (!p.isDemod) {
        if (p.bitIO) {
            if (nRows % 2 != 0)
                throw Error("comm.QPSKModulator: BitInput requires an even number of rows.",
                            0, 0, "step", "", "numkit:comm:QpskBitLength");
            const keel::usize outRows = nRows / 2;
            Value y = Value::matrix(outRows, nCols, ValueType::COMPLEX, mr);
            for (keel::usize c = 0; c < nCols; ++c) {
                for (keel::usize r = 0; r < outRows; ++r) {
                    const keel::i32 b0 = roundI32(x.elemAsDouble(2 * r + c * nRows)) & 1;
                    const keel::i32 b1 = roundI32(x.elemAsDouble(2 * r + 1 + c * nRows)) & 1;
                    const keel::i32 sym = (b0 << 1) | b1;
                    const keel::i32 idx = p.gray ? kGrayToIdx[sym] : sym;
                    const keel::f64 ang = p.phaseOffset + keel::as<keel::f64>(idx) * (kPi / 2.0);
                    y.complexDataMut()[r + c * outRows] = keel::c64{keel::cos(ang), keel::sin(ang)};
                }
            }
            outs[0] = keel::move(y);
        } else {
            Value y = Value::matrix(nRows, nCols, ValueType::COMPLEX, mr);
            for (keel::usize i = 0; i < x.numel(); ++i) {
                const keel::i32 sym = roundI32(x.elemAsDouble(i)) & 3;
                const keel::i32 idx = p.gray ? kGrayToIdx[sym] : sym;
                const keel::f64 ang = p.phaseOffset + keel::as<keel::f64>(idx) * (kPi / 2.0);
                y.complexDataMut()[i] = keel::c64{keel::cos(ang), keel::sin(ang)};
            }
            outs[0] = keel::move(y);
        }
    } else {
        auto decodeSym = [&](keel::c64 z) -> keel::i32 {
            keel::f64 ang = keel::atan2(z.im, z.re) - p.phaseOffset;
            while (ang < 0.0) ang += 2.0 * kPi;
            const keel::i32 q = roundI32(ang / (kPi / 2.0)) & 3;
            return p.gray ? kIdxToGray[q] : q;
        };
        if (p.bitIO) {
            const keel::usize outRows = nRows * 2;
            Value y = Value::matrix(outRows, nCols, ValueType::DOUBLE, mr);
            for (keel::usize c = 0; c < nCols; ++c) {
                for (keel::usize r = 0; r < nRows; ++r) {
                    const keel::usize i = r + c * nRows;
                    const keel::c64 z = x.isComplex() ? x.complexData()[i] : keel::c64{x.elemAsDouble(i), 0.0};
                    const keel::i32 sym = decodeSym(z);
                    y.doubleDataMut()[2 * r + c * outRows]     = keel::as<keel::f64>((sym >> 1) & 1);
                    y.doubleDataMut()[2 * r + 1 + c * outRows] = keel::as<keel::f64>(sym & 1);
                }
            }
            outs[0] = keel::move(y);
        } else {
            Value y = Value::matrix(nRows, nCols, ValueType::DOUBLE, mr);
            for (keel::usize i = 0; i < x.numel(); ++i) {
                const keel::c64 z = x.isComplex() ? x.complexData()[i] : keel::c64{x.elemAsDouble(i), 0.0};
                y.doubleDataMut()[i] = keel::as<keel::f64>(decodeSym(z));
            }
            outs[0] = keel::move(y);
        }
    }
}

BuiltinClass makeQpskClass(const char *clsName, bool isDemod)
{
    BuiltinClass cls;
    cls.name = keel::String::from(clsName);
    cls.isHandle = true;
    cls.construct = [clsName, isDemod](keel::ArrayRef<Value> args, CallContext &ctx) -> Value {
        auto *mr = allocOf(ctx);
        auto payload = keel::make_arc_in<QpskPayload>(mr);
        payload->isDemod = isDemod;
        qpskApplyNV(*payload, args);
        auto st = keel::make_arc_in<ObjectState>(mr, mr);
        st->native = keel::move(payload);
        return Value::object(clsName, keel::move(st), mr);
    };
    cls.subsref = &qpskStepImpl;
    cls.methods.put(keel::String::from("step"), &qpskStepImpl);
    auto noop = [](Value &, keel::ArrayRef<Value>, keel::usize,
                   keel::MutArrayRef<Value> outs, CallContext &) {
        if (!outs.empty()) outs[0] = Value();
    };
    cls.methods.put(keel::String::from("reset"), noop);
    cls.methods.put(keel::String::from("release"), noop);
    cls.propGet = [](const Value &self, keel::StringRef name, Value &out, CallContext &ctx) -> bool {
        auto *mr = allocOf(ctx);
        const auto &p = *static_cast<const QpskPayload *>(self.objectStateConst()->native.get());
        if (name.equals_ignore_ascii_case("PhaseOffset")) { out = Value::scalar(p.phaseOffset, mr); return true; }
        if (name.equals_ignore_ascii_case("BitInput") || name.equals_ignore_ascii_case("BitOutput")) { out = Value::logicalScalar(p.bitIO, mr); return true; }
        if (name.equals_ignore_ascii_case("SymbolMapping")) { out = Value::fromString(p.gray ? "Gray" : "Binary", mr); return true; }
        return false;
    };
    cls.propSet = [](Value &self, keel::StringRef name, const Value &val, CallContext &) -> bool {
        auto &p = *static_cast<QpskPayload *>(self.objectStateMut()->native.get());
        if (name.equals_ignore_ascii_case("PhaseOffset")) { p.phaseOffset = val.toScalar(); return true; }
        if (name.equals_ignore_ascii_case("BitInput") || name.equals_ignore_ascii_case("BitOutput")) { p.bitIO = val.toBool(); return true; }
        if (name.equals_ignore_ascii_case("SymbolMapping")) { p.gray = !val.toString().equals_ignore_ascii_case("Binary"); return true; }
        return false;
    };
    return cls;
}

void commQpskMod_ctor_reg(keel::ArrayRef<Value> args, keel::usize, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    auto *mr = allocOf(ctx);
    auto payload = keel::make_arc_in<QpskPayload>(mr);
    payload->isDemod = false;
    qpskApplyNV(*payload, args);
    auto st = keel::make_arc_in<ObjectState>(mr, mr);
    st->native = keel::move(payload);
    outs[0] = Value::object("comm.QPSKModulator", keel::move(st), mr);
}

void commQpskDemod_ctor_reg(keel::ArrayRef<Value> args, keel::usize, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    auto *mr = allocOf(ctx);
    auto payload = keel::make_arc_in<QpskPayload>(mr);
    payload->isDemod = true;
    qpskApplyNV(*payload, args);
    auto st = keel::make_arc_in<ObjectState>(mr, mr);
    st->native = keel::move(payload);
    outs[0] = Value::object("comm.QPSKDemodulator", keel::move(st), mr);
}

// ── 4. comm.RaisedCosineTransmitFilter & ReceiveFilter ──────────────────
struct RaisedCosinePayload final : public NativePayload {
    bool isRx = false;
    bool sqrtShape = true;
    keel::f64 rolloff = 0.25;
    keel::i32 span = 10;
    keel::i32 sps = 8;
    keel::i32 decimation = 1;
    keel::i32 decimationOffset = 0;
    keel::f64 gain = 1.0;
    keel::Array<keel::f64> taps;
    keel::Array<keel::Array<keel::c64>> delay;
    bool isComplex = false;

    explicit RaisedCosinePayload(keel::Allocator *mr = nullptr)
        : taps(mr), delay(mr)
    {}

    void invalidate()
    {
        taps.clear();
        delay.clear();
        isComplex = false;
    }

    void ensureTaps(keel::Allocator *mr)
    {
        if (!taps.empty()) return;
        const keel::i32 sp = keel::max(1, span);
        const keel::i32 os = keel::max(1, sps);
        const keel::usize N = keel::as<keel::usize>(sp * os + 1);
        taps.resize(N, 0.0);
        const keel::f64 beta = rolloff;
        const keel::f64 half = 0.5 * keel::as<keel::f64>(sp * os);
        const keel::f64 osF = keel::as<keel::f64>(os);
        for (keel::usize k = 0; k < N; ++k) {
            const keel::f64 t = (keel::as<keel::f64>(k) - half) / osF;
            if (sqrtShape) {
                if (keel::abs(t) < 1e-12) {
                    taps[k] = (1.0 - beta + 4.0 * beta / kPi);
                } else if (beta > 0.0 && keel::abs(keel::abs(4.0 * beta * t) - 1.0) < 1e-9) {
                    const keel::f64 a = (1.0 + 2.0 / kPi) * keel::sin(kPi / (4.0 * beta));
                    const keel::f64 b = (1.0 - 2.0 / kPi) * keel::cos(kPi / (4.0 * beta));
                    taps[k] = (beta / keel::sqrt(2.0)) * (a + b);
                } else {
                    const keel::f64 num = keel::sin(kPi * t * (1.0 - beta))
                                        + 4.0 * beta * t * keel::cos(kPi * t * (1.0 + beta));
                    const keel::f64 den = kPi * t * (1.0 - (4.0 * beta * t) * (4.0 * beta * t));
                    taps[k] = num / den;
                }
            } else {
                if (keel::abs(t) < 1e-12) {
                    taps[k] = 1.0;
                } else if (beta > 0.0 && keel::abs(keel::abs(2.0 * beta * t) - 1.0) < 1e-9) {
                    taps[k] = (kPi / 4.0) * (keel::sin(kPi / (2.0 * beta)) / (kPi / (2.0 * beta)));
                } else {
                    const keel::f64 sinc = keel::sin(kPi * t) / (kPi * t);
                    taps[k] = sinc * keel::cos(kPi * beta * t) / (1.0 - (2.0 * beta * t) * (2.0 * beta * t));
                }
            }
        }
        // Normalize to unit L2 norm (matching rcosdesign), then apply linear Gain.
        keel::f64 e = 0.0;
        for (keel::usize k = 0; k < N; ++k) e += taps[k] * taps[k];
        const keel::f64 scale = (e > 0.0) ? (gain / keel::sqrt(e)) : gain;
        for (keel::usize k = 0; k < N; ++k) taps[k] *= scale;
        (void)mr;
    }

    keel::Arc<NativePayload> clone(keel::Allocator *alloc = nullptr) const override
    {
        auto *a = alloc ? alloc : keel::default_allocator();
        auto p = keel::make_arc_in<RaisedCosinePayload>(a, a);
        p->isRx = isRx;
        p->sqrtShape = sqrtShape;
        p->rolloff = rolloff;
        p->span = span;
        p->sps = sps;
        p->decimation = decimation;
        p->decimationOffset = decimationOffset;
        p->gain = gain;
        p->taps = taps.clone(a);
        p->isComplex = isComplex;
        p->delay.reserve(delay.size());
        for (keel::usize i = 0; i < delay.size(); ++i)
            p->delay.push_back(delay[i].clone(a));
        return p;
    }
};

void rcosApplyNV(RaisedCosinePayload &p, keel::ArrayRef<Value> args)
{
    for (keel::usize i = 0; i + 1 < args.size(); i += 2) {
        if (!args[i].isChar() && !args[i].isString()) continue;
        const keel::String k = args[i].toString();
        if (k.equals_ignore_ascii_case("Shape")) {
            const keel::String v = args[i + 1].toString();
            p.sqrtShape = !v.equals_ignore_ascii_case("Normal");
        } else if (k.equals_ignore_ascii_case("RolloffFactor")) {
            p.rolloff = args[i + 1].toScalar();
        } else if (k.equals_ignore_ascii_case("FilterSpanInSymbols")) {
            p.span = keel::max(1, roundI32(args[i + 1].toScalar()));
        } else if (k.equals_ignore_ascii_case("OutputSamplesPerSymbol")
                   || k.equals_ignore_ascii_case("InputSamplesPerSymbol")) {
            p.sps = keel::max(1, roundI32(args[i + 1].toScalar()));
        } else if (k.equals_ignore_ascii_case("DecimationFactor")) {
            p.decimation = keel::max(1, roundI32(args[i + 1].toScalar()));
        } else if (k.equals_ignore_ascii_case("DecimationOffset")) {
            p.decimationOffset = keel::max(0, roundI32(args[i + 1].toScalar()));
        } else if (k.equals_ignore_ascii_case("Gain")) {
            p.gain = args[i + 1].toScalar();
        }
    }
    p.invalidate();
}

void rcosStepImpl(Value &self, keel::ArrayRef<Value> args, keel::usize,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "comm.RaisedCosineFilter.step");
    auto *mr = allocOf(ctx);
    auto &p = *static_cast<RaisedCosinePayload *>(self.objectStateMut()->native.get());
    p.ensureTaps(mr);

    const Value &x = args[0];
    const keel::usize nRows = x.dims().rows();
    const keel::usize nCols = x.dims().cols();
    const keel::usize nTaps = p.taps.size();
    const keel::usize histLen = (nTaps > 0) ? (nTaps - 1) : 0;
    const bool outComplex = x.isComplex() || p.isComplex;
    if (outComplex) p.isComplex = true;

    if (p.delay.size() != nCols) {
        p.delay.clear();
        p.delay.reserve(nCols);
        for (keel::usize ch = 0; ch < nCols; ++ch)
            p.delay.push_back(keel::Array<keel::c64>(histLen, keel::c64{0.0, 0.0}, mr));
    }

    if (!p.isRx) {
        const keel::usize L = keel::as<keel::usize>(keel::max(1, p.sps));
        const keel::usize outRows = nRows * L;
        Value y = Value::matrix(outRows, nCols, outComplex ? ValueType::COMPLEX : ValueType::DOUBLE, mr);
        for (keel::usize ch = 0; ch < nCols; ++ch) {
            auto &buf = p.delay[ch];
            for (keel::usize r = 0; r < nRows; ++r) {
                const keel::usize idx = r + ch * nRows;
                const keel::c64 xin = x.isComplex() ? x.complexData()[idx] : keel::c64{x.elemAsDouble(idx), 0.0};
                for (keel::usize sub = 0; sub < L; ++sub) {
                    const keel::c64 s = (sub == 0) ? xin : keel::c64{0.0, 0.0};
                    keel::c64 acc{p.taps[0] * s.re, p.taps[0] * s.im};
                    for (keel::usize k = 0; k < histLen; ++k) {
                        acc.re += p.taps[k + 1] * buf[k].re;
                        acc.im += p.taps[k + 1] * buf[k].im;
                    }
                    for (keel::usize k = histLen; k > 1; --k)
                        buf[k - 1] = buf[k - 2];
                    if (histLen > 0)
                        buf[0] = s;
                    const keel::usize oidx = (r * L + sub) + ch * outRows;
                    if (outComplex) y.complexDataMut()[oidx] = acc;
                    else y.doubleDataMut()[oidx] = acc.re;
                }
            }
        }
        outs[0] = keel::move(y);
    } else {
        const keel::usize D = keel::as<keel::usize>(keel::max(1, p.decimation));
        const keel::usize off = keel::as<keel::usize>(p.decimationOffset) % D;
        const keel::usize outRows = nRows / D;
        Value y = Value::matrix(outRows, nCols, outComplex ? ValueType::COMPLEX : ValueType::DOUBLE, mr);
        for (keel::usize ch = 0; ch < nCols; ++ch) {
            auto &buf = p.delay[ch];
            keel::usize outR = 0;
            for (keel::usize r = 0; r < nRows; ++r) {
                const keel::usize idx = r + ch * nRows;
                const keel::c64 s = x.isComplex() ? x.complexData()[idx] : keel::c64{x.elemAsDouble(idx), 0.0};
                keel::c64 acc{p.taps[0] * s.re, p.taps[0] * s.im};
                for (keel::usize k = 0; k < histLen; ++k) {
                    acc.re += p.taps[k + 1] * buf[k].re;
                    acc.im += p.taps[k + 1] * buf[k].im;
                }
                for (keel::usize k = histLen; k > 1; --k)
                    buf[k - 1] = buf[k - 2];
                if (histLen > 0)
                    buf[0] = s;
                if (r % D == off && outR < outRows) {
                    const keel::usize oidx = outR + ch * outRows;
                    if (outComplex) y.complexDataMut()[oidx] = acc;
                    else y.doubleDataMut()[oidx] = acc.re;
                    ++outR;
                }
            }
        }
        outs[0] = keel::move(y);
    }
}

BuiltinClass makeRaisedCosineClass(const char *clsName, bool isRx)
{
    BuiltinClass cls;
    cls.name = keel::String::from(clsName);
    cls.isHandle = true;
    cls.construct = [clsName, isRx](keel::ArrayRef<Value> args, CallContext &ctx) -> Value {
        auto *mr = allocOf(ctx);
        auto payload = keel::make_arc_in<RaisedCosinePayload>(mr, mr);
        payload->isRx = isRx;
        if (isRx) payload->decimation = 1;
        rcosApplyNV(*payload, args);
        auto st = keel::make_arc_in<ObjectState>(mr, mr);
        st->native = keel::move(payload);
        return Value::object(clsName, keel::move(st), mr);
    };
    cls.subsref = &rcosStepImpl;
    cls.methods.put(keel::String::from("step"), &rcosStepImpl);
    auto resetFn = [](Value &self, keel::ArrayRef<Value>, keel::usize,
                      keel::MutArrayRef<Value> outs, CallContext &) {
        static_cast<RaisedCosinePayload *>(self.objectStateMut()->native.get())->delay.clear();
        if (!outs.empty()) outs[0] = Value();
    };
    cls.methods.put(keel::String::from("reset"), resetFn);
    cls.methods.put(keel::String::from("release"), resetFn);
    cls.methods.put(keel::String::from("coeffs"),
                    [](Value &self, keel::ArrayRef<Value>, keel::usize,
                       keel::MutArrayRef<Value> outs, CallContext &ctx) {
                        auto *mr = allocOf(ctx);
                        auto &p = *static_cast<RaisedCosinePayload *>(self.objectStateMut()->native.get());
                        p.ensureTaps(mr);
                        Value num = Value::matrix(1, p.taps.size(), ValueType::DOUBLE, mr);
                        for (keel::usize k = 0; k < p.taps.size(); ++k)
                            num.doubleDataMut()[k] = p.taps[k];
                        Value s = Value::structure(mr);
                        s.setFieldAll("Numerator", keel::move(num));
                        outs[0] = keel::move(s);
                    });
    cls.propGet = [](const Value &self, keel::StringRef name, Value &out, CallContext &ctx) -> bool {
        auto *mr = allocOf(ctx);
        const auto &p = *static_cast<const RaisedCosinePayload *>(self.objectStateConst()->native.get());
        if (name.equals_ignore_ascii_case("Shape")) { out = Value::fromString(p.sqrtShape ? "Square root" : "Normal", mr); return true; }
        if (name.equals_ignore_ascii_case("RolloffFactor")) { out = Value::scalar(p.rolloff, mr); return true; }
        if (name.equals_ignore_ascii_case("FilterSpanInSymbols")) { out = Value::scalar(keel::as<keel::f64>(p.span), mr); return true; }
        if (name.equals_ignore_ascii_case("OutputSamplesPerSymbol") || name.equals_ignore_ascii_case("InputSamplesPerSymbol")) {
            out = Value::scalar(keel::as<keel::f64>(p.sps), mr); return true;
        }
        if (name.equals_ignore_ascii_case("DecimationFactor")) { out = Value::scalar(keel::as<keel::f64>(p.decimation), mr); return true; }
        if (name.equals_ignore_ascii_case("DecimationOffset")) { out = Value::scalar(keel::as<keel::f64>(p.decimationOffset), mr); return true; }
        if (name.equals_ignore_ascii_case("Gain")) { out = Value::scalar(p.gain, mr); return true; }
        return false;
    };
    return cls;
}

void commRcosTx_ctor_reg(keel::ArrayRef<Value> args, keel::usize, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    auto *mr = allocOf(ctx);
    auto payload = keel::make_arc_in<RaisedCosinePayload>(mr, mr);
    payload->isRx = false;
    rcosApplyNV(*payload, args);
    auto st = keel::make_arc_in<ObjectState>(mr, mr);
    st->native = keel::move(payload);
    outs[0] = Value::object("comm.RaisedCosineTransmitFilter", keel::move(st), mr);
}

void commRcosRx_ctor_reg(keel::ArrayRef<Value> args, keel::usize, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    auto *mr = allocOf(ctx);
    auto payload = keel::make_arc_in<RaisedCosinePayload>(mr, mr);
    payload->isRx = true;
    payload->decimation = 1;
    rcosApplyNV(*payload, args);
    auto st = keel::make_arc_in<ObjectState>(mr, mr);
    st->native = keel::move(payload);
    outs[0] = Value::object("comm.RaisedCosineReceiveFilter", keel::move(st), mr);
}

// ── 5. comm.AWGNChannel ─────────────────────────────────────────────────
struct AwgnChannelPayload final : public NativePayload {
    keel::String noiseMethod;
    keel::f64 ebNo = 10.0;
    keel::f64 esNo = 10.0;
    keel::f64 snr = 10.0;
    keel::f64 variance = 1.0;
    keel::f64 bitsPerSymbol = 1.0;
    keel::f64 signalPower = 1.0;
    keel::f64 samplesPerSymbol = 1.0;

    AwgnChannelPayload()
        : noiseMethod(keel::String::from("Signal to noise ratio (Eb/No)"))
    {}

    keel::Arc<NativePayload> clone(keel::Allocator *alloc = nullptr) const override
    {
        auto *a = alloc ? alloc : keel::default_allocator();
        auto p = keel::make_arc_in<AwgnChannelPayload>(a);
        p->noiseMethod = noiseMethod.clone();
        p->ebNo = ebNo;
        p->esNo = esNo;
        p->snr = snr;
        p->variance = variance;
        p->bitsPerSymbol = bitsPerSymbol;
        p->signalPower = signalPower;
        p->samplesPerSymbol = samplesPerSymbol;
        return p;
    }
};

void awgnChanApplyNV(AwgnChannelPayload &p, keel::ArrayRef<Value> args)
{
    for (keel::usize i = 0; i + 1 < args.size(); i += 2) {
        if (!args[i].isChar() && !args[i].isString()) continue;
        const keel::String k = args[i].toString();
        if (k.equals_ignore_ascii_case("NoiseMethod")) p.noiseMethod = args[i + 1].toString();
        else if (k.equals_ignore_ascii_case("EbNo")) p.ebNo = args[i + 1].toScalar();
        else if (k.equals_ignore_ascii_case("EsNo")) p.esNo = args[i + 1].toScalar();
        else if (k.equals_ignore_ascii_case("SNR")) p.snr = args[i + 1].toScalar();
        else if (k.equals_ignore_ascii_case("Variance")) p.variance = args[i + 1].toScalar();
        else if (k.equals_ignore_ascii_case("BitsPerSymbol")) p.bitsPerSymbol = args[i + 1].toScalar();
        else if (k.equals_ignore_ascii_case("SignalPower")) p.signalPower = args[i + 1].toScalar();
        else if (k.equals_ignore_ascii_case("SamplesPerSymbol")) p.samplesPerSymbol = args[i + 1].toScalar();
    }
}

void awgnChanStepImpl(Value &self, keel::ArrayRef<Value> args, keel::usize,
                      keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "comm.AWGNChannel.step");
    auto *mr = allocOf(ctx);
    const auto &p = *static_cast<const AwgnChannelPayload *>(self.objectStateMut()->native.get());
    const Value &x = args[0];
    keel::f64 noiseVar = p.variance;
    if (p.noiseMethod.equals_ignore_ascii_case("Signal to noise ratio (SNR)")) {
        const keel::f64 snrLin = keel::pow(10.0, p.snr / 10.0);
        noiseVar = p.signalPower / snrLin;
    } else if (p.noiseMethod.equals_ignore_ascii_case("Signal to noise ratio (Es/No)")) {
        const keel::f64 esLin = keel::pow(10.0, p.esNo / 10.0);
        noiseVar = (p.signalPower * p.samplesPerSymbol) / esLin;
    } else if (p.noiseMethod.equals_ignore_ascii_case("Signal to noise ratio (Eb/No)")) {
        const keel::f64 ebLin = keel::pow(10.0, p.ebNo / 10.0);
        noiseVar = (p.signalPower * p.samplesPerSymbol) / (p.bitsPerSymbol * ebLin);
    }
    auto &rng = ctx.engine->rng();
    const keel::usize nRows = x.dims().rows();
    const keel::usize nCols = x.dims().cols();
    if (x.isComplex()) {
        const keel::f64 sigma = keel::sqrt(0.5 * noiseVar);
        Value y = Value::matrix(nRows, nCols, ValueType::COMPLEX, mr);
        for (keel::usize i = 0; i < x.numel(); ++i) {
            const keel::c64 z = x.complexData()[i];
            y.complexDataMut()[i] = keel::c64{z.re + sigma * rng.nextNormal(), z.im + sigma * rng.nextNormal()};
        }
        outs[0] = keel::move(y);
    } else {
        const keel::f64 sigma = keel::sqrt(noiseVar);
        Value y = Value::matrix(nRows, nCols, ValueType::DOUBLE, mr);
        for (keel::usize i = 0; i < x.numel(); ++i)
            y.doubleDataMut()[i] = x.elemAsDouble(i) + sigma * rng.nextNormal();
        outs[0] = keel::move(y);
    }
}

BuiltinClass makeAwgnChannelClass()
{
    BuiltinClass cls;
    cls.name = keel::String::from("comm.AWGNChannel");
    cls.isHandle = true;
    cls.construct = [](keel::ArrayRef<Value> args, CallContext &ctx) -> Value {
        auto *mr = allocOf(ctx);
        auto payload = keel::make_arc_in<AwgnChannelPayload>(mr);
        awgnChanApplyNV(*payload, args);
        auto st = keel::make_arc_in<ObjectState>(mr, mr);
        st->native = keel::move(payload);
        return Value::object("comm.AWGNChannel", keel::move(st), mr);
    };
    cls.subsref = &awgnChanStepImpl;
    cls.methods.put(keel::String::from("step"), &awgnChanStepImpl);
    auto noop = [](Value &, keel::ArrayRef<Value>, keel::usize,
                   keel::MutArrayRef<Value> outs, CallContext &) {
        if (!outs.empty()) outs[0] = Value();
    };
    cls.methods.put(keel::String::from("reset"), noop);
    cls.methods.put(keel::String::from("release"), noop);
    return cls;
}

void commAwgnChannel_ctor_reg(keel::ArrayRef<Value> args, keel::usize, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    auto *mr = allocOf(ctx);
    auto payload = keel::make_arc_in<AwgnChannelPayload>(mr);
    awgnChanApplyNV(*payload, args);
    auto st = keel::make_arc_in<ObjectState>(mr, mr);
    st->native = keel::move(payload);
    outs[0] = Value::object("comm.AWGNChannel", keel::move(st), mr);
}

// ── 6. comm.CRCGenerator & comm.CRCDetector ─────────────────────────────
struct CrcPayload final : public NativePayload {
    bool isDetector = false;
    keel::Array<keel::u8> poly; // binary vector [g_d, g_{d-1}, ..., g_0] with poly[0]==1
    keel::Array<keel::u8> initCond;
    keel::Array<keel::u8> finalXor;
    keel::i32 checksumsPerFrame = 1;

    explicit CrcPayload(keel::Allocator *mr = nullptr)
        : poly(mr), initCond(mr), finalXor(mr)
    {
        // Default: z^16 + z^12 + z^5 + 1
        poly.resize(17, 0);
        poly[0] = 1; poly[4] = 1; poly[11] = 1; poly[16] = 1;
    }

    keel::Arc<NativePayload> clone(keel::Allocator *alloc = nullptr) const override
    {
        auto *a = alloc ? alloc : keel::default_allocator();
        auto p = keel::make_arc_in<CrcPayload>(a, a);
        p->isDetector = isDetector;
        p->poly = poly.clone(a);
        p->initCond = initCond.clone(a);
        p->finalXor = finalXor.clone(a);
        p->checksumsPerFrame = checksumsPerFrame;
        return p;
    }
};

keel::Array<keel::u8> parseCrcPoly(const Value &v, keel::Allocator *mr)
{
    keel::Array<keel::u8> poly(mr);
    if (v.isChar() || v.isString()) {
        const keel::String s = v.toString();
        keel::Array<keel::i32> exps(mr);
        keel::i32 maxExp = 0;
        keel::usize i = 0;
        while (i < s.size()) {
            while (i < s.size() && (keel::is_ascii_space(s[i]) || s[i] == '+')) ++i;
            if (i >= s.size()) break;
            if (s[i] == 'z' || s[i] == 'Z' || s[i] == 'x' || s[i] == 'X') {
                ++i;
                while (i < s.size() && keel::is_ascii_space(s[i])) ++i;
                keel::i32 e = 1;
                if (i < s.size() && s[i] == '^') {
                    ++i;
                    while (i < s.size() && keel::is_ascii_space(s[i])) ++i;
                    keel::usize start = i;
                    while (i < s.size() && keel::is_ascii_digit(s[i])) ++i;
                    if (auto parsed = keel::parse_i64(s.substr(start, i - start)); parsed.is_ok())
                        e = keel::as<keel::i32>(parsed.value());
                }
                exps.push_back(e);
                if (e > maxExp) maxExp = e;
            } else if (keel::is_ascii_digit(s[i])) {
                keel::usize start = i;
                while (i < s.size() && keel::is_ascii_digit(s[i])) ++i;
                if (auto parsed = keel::parse_i64(s.substr(start, i - start)); parsed.is_ok() && parsed.value() != 0)
                    exps.push_back(0);
            } else {
                ++i;
            }
        }
        poly.resize(keel::as<keel::usize>(maxExp + 1), 0);
        for (keel::usize k = 0; k < exps.size(); ++k) {
            const keel::usize idx = keel::as<keel::usize>(maxExp - exps[k]);
            poly[idx] ^= 1u;
        }
        return poly;
    }
    // Numeric vector: either binary vector [1 0 0 1] or descending exponents [3 0].
    bool allBinary = true;
    for (keel::usize k = 0; k < v.numel(); ++k) {
        const keel::f64 d = v.elemAsDouble(k);
        if (d != 0.0 && d != 1.0) { allBinary = false; break; }
    }
    if (allBinary && v.numel() >= 2 && v.elemAsDouble(0) == 1.0) {
        poly.reserve(v.numel());
        for (keel::usize k = 0; k < v.numel(); ++k)
            poly.push_back(keel::as<keel::u8>(roundI32(v.elemAsDouble(k)) & 1));
    } else {
        keel::i32 maxExp = 0;
        for (keel::usize k = 0; k < v.numel(); ++k) {
            const keel::i32 e = roundI32(v.elemAsDouble(k));
            if (e > maxExp) maxExp = e;
        }
        poly.resize(keel::as<keel::usize>(maxExp + 1), 0);
        for (keel::usize k = 0; k < v.numel(); ++k) {
            const keel::i32 e = roundI32(v.elemAsDouble(k));
            if (e >= 0) poly[keel::as<keel::usize>(maxExp - e)] ^= 1u;
        }
    }
    return poly;
}

void crcApplyArgs(CrcPayload &p, keel::ArrayRef<Value> args, keel::Allocator *mr)
{
    keel::usize i = 0;
    if (args.size() == 1 || (args.size() >= 1 && args.size() % 2 == 1)) {
        p.poly = parseCrcPoly(args[0], mr);
        i = 1;
    }
    for (; i + 1 < args.size(); i += 2) {
        if (!args[i].isChar() && !args[i].isString()) continue;
        const keel::String k = args[i].toString();
        if (k.equals_ignore_ascii_case("Polynomial")) {
            p.poly = parseCrcPoly(args[i + 1], mr);
        } else if (k.equals_ignore_ascii_case("ChecksumsPerFrame")) {
            p.checksumsPerFrame = keel::max(1, roundI32(args[i + 1].toScalar()));
        }
    }
}

keel::Array<keel::u8> computeCrcBits(keel::ArrayRef<keel::u8> msg,
                                     keel::ArrayRef<keel::u8> poly,
                                     keel::Allocator *mr)
{
    const keel::usize deg = (poly.size() > 1) ? (poly.size() - 1) : 0;
    keel::Array<keel::u8> reg(deg, 0, mr);
    if (deg == 0) return reg;
    // Standard shift-register CRC: feed msg followed by deg zero bits.
    const keel::usize total = msg.size() + deg;
    for (keel::usize t = 0; t < total; ++t) {
        const keel::u8 inBit = (t < msg.size()) ? (msg[t] & 1u) : 0u;
        const keel::u8 top = reg[0];
        for (keel::usize j = 0; j + 1 < deg; ++j)
            reg[j] = reg[j + 1];
        reg[deg - 1] = inBit;
        if (top) {
            for (keel::usize j = 0; j < deg; ++j)
                reg[j] ^= poly[j + 1];
        }
    }
    return reg;
}

void crcStepImpl(Value &self, keel::ArrayRef<Value> args, keel::usize nargout,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "comm.CRC.step");
    auto *mr = allocOf(ctx);
    const auto &p = *static_cast<const CrcPayload *>(self.objectStateMut()->native.get());
    const Value &x = args[0];
    const keel::usize deg = (p.poly.size() > 1) ? (p.poly.size() - 1) : 0;
    const keel::usize n = x.numel();
    const bool isLog = x.isLogical();

    keel::Array<keel::u8> bits(n, 0, mr);
    for (keel::usize i = 0; i < n; ++i)
        bits[i] = keel::as<keel::u8>(x.elemAsDouble(i) != 0.0 ? 1 : 0);

    if (!p.isDetector) {
        auto crc = computeCrcBits(bits.as_ref(), p.poly.as_ref(), mr);
        const keel::usize outLen = n + deg;
        Value y = Value::matrix(outLen, 1, isLog ? ValueType::LOGICAL : ValueType::DOUBLE, mr);
        for (keel::usize i = 0; i < n; ++i) {
            if (isLog) y.logicalDataMut()[i] = bits[i];
            else y.doubleDataMut()[i] = keel::as<keel::f64>(bits[i]);
        }
        for (keel::usize i = 0; i < deg; ++i) {
            if (isLog) y.logicalDataMut()[n + i] = crc[i];
            else y.doubleDataMut()[n + i] = keel::as<keel::f64>(crc[i]);
        }
        outs[0] = keel::move(y);
    } else {
        if (n < deg)
            throw Error("comm.CRCDetector: input length must be at least the CRC polynomial degree.",
                        0, 0, "step", "", "numkit:comm:CrcShortInput");
        const keel::usize msgLen = n - deg;
        auto expected = computeCrcBits(keel::ArrayRef<keel::u8>(bits.data(), msgLen), p.poly.as_ref(), mr);
        bool err = false;
        for (keel::usize i = 0; i < deg; ++i) {
            if (expected[i] != bits[msgLen + i]) { err = true; break; }
        }
        Value y = Value::matrix(msgLen, 1, isLog ? ValueType::LOGICAL : ValueType::DOUBLE, mr);
        for (keel::usize i = 0; i < msgLen; ++i) {
            if (isLog) y.logicalDataMut()[i] = bits[i];
            else y.doubleDataMut()[i] = keel::as<keel::f64>(bits[i]);
        }
        outs[0] = keel::move(y);
        if (nargout >= 2 && outs.size() >= 2)
            outs[1] = Value::logicalScalar(err, mr);
    }
}

BuiltinClass makeCrcClass(const char *clsName, bool isDet)
{
    BuiltinClass cls;
    cls.name = keel::String::from(clsName);
    cls.isHandle = true;
    cls.construct = [clsName, isDet](keel::ArrayRef<Value> args, CallContext &ctx) -> Value {
        auto *mr = allocOf(ctx);
        auto payload = keel::make_arc_in<CrcPayload>(mr, mr);
        payload->isDetector = isDet;
        crcApplyArgs(*payload, args, mr);
        auto st = keel::make_arc_in<ObjectState>(mr, mr);
        st->native = keel::move(payload);
        return Value::object(clsName, keel::move(st), mr);
    };
    cls.subsref = &crcStepImpl;
    cls.methods.put(keel::String::from("step"), &crcStepImpl);
    auto noop = [](Value &, keel::ArrayRef<Value>, keel::usize,
                   keel::MutArrayRef<Value> outs, CallContext &) {
        if (!outs.empty()) outs[0] = Value();
    };
    cls.methods.put(keel::String::from("reset"), noop);
    cls.methods.put(keel::String::from("release"), noop);
    return cls;
}

void commCrcGen_ctor_reg(keel::ArrayRef<Value> args, keel::usize, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    auto *mr = allocOf(ctx);
    auto payload = keel::make_arc_in<CrcPayload>(mr, mr);
    payload->isDetector = false;
    crcApplyArgs(*payload, args, mr);
    auto st = keel::make_arc_in<ObjectState>(mr, mr);
    st->native = keel::move(payload);
    outs[0] = Value::object("comm.CRCGenerator", keel::move(st), mr);
}

void commCrcDet_ctor_reg(keel::ArrayRef<Value> args, keel::usize, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    auto *mr = allocOf(ctx);
    auto payload = keel::make_arc_in<CrcPayload>(mr, mr);
    payload->isDetector = true;
    crcApplyArgs(*payload, args, mr);
    auto st = keel::make_arc_in<ObjectState>(mr, mr);
    st->native = keel::move(payload);
    outs[0] = Value::object("comm.CRCDetector", keel::move(st), mr);
}

// ── 7. comm.OFDMModulator & comm.OFDMDemodulator ────────────────────────
struct OfdmPayload final : public NativePayload {
    bool isDemod = false;
    keel::i32 fftLength = 64;
    keel::i32 guardLeft = 6;
    keel::i32 guardRight = 5;
    keel::i32 cpLength = 16;
    keel::i32 numSymbols = 1;
    bool insertDCNull = false;

    keel::Arc<NativePayload> clone(keel::Allocator *alloc = nullptr) const override
    {
        auto *a = alloc ? alloc : keel::default_allocator();
        auto p = keel::make_arc_in<OfdmPayload>(a);
        p->isDemod = isDemod;
        p->fftLength = fftLength;
        p->guardLeft = guardLeft;
        p->guardRight = guardRight;
        p->cpLength = cpLength;
        p->numSymbols = numSymbols;
        p->insertDCNull = insertDCNull;
        return p;
    }
};

void ofdmApplyArgs(OfdmPayload &p, keel::ArrayRef<Value> args)
{
    keel::usize i = 0;
    if (args.size() >= 1 && args[0].isObject()
        && args[0].objectClassNameRef() == "comm.OFDMModulator") {
        if (const ObjectState *os = args[0].objectStateConst(); os && os->native) {
            const auto &src = *static_cast<const OfdmPayload *>(os->native.get());
            p.fftLength = src.fftLength;
            p.guardLeft = src.guardLeft;
            p.guardRight = src.guardRight;
            p.cpLength = src.cpLength;
            p.numSymbols = src.numSymbols;
            p.insertDCNull = src.insertDCNull;
        }
        i = 1;
    }
    for (; i + 1 < args.size(); i += 2) {
        if (!args[i].isChar() && !args[i].isString()) continue;
        const keel::String k = args[i].toString();
        if (k.equals_ignore_ascii_case("FFTLength")) {
            p.fftLength = keel::max(2, roundI32(args[i + 1].toScalar()));
        } else if (k.equals_ignore_ascii_case("NumGuardBandCarriers")) {
            if (args[i + 1].numel() >= 2) {
                p.guardLeft = keel::max(0, roundI32(args[i + 1].elemAsDouble(0)));
                p.guardRight = keel::max(0, roundI32(args[i + 1].elemAsDouble(1)));
            }
        } else if (k.equals_ignore_ascii_case("CyclicPrefixLength")) {
            p.cpLength = keel::max(0, roundI32(args[i + 1].elemAsDouble(0)));
        } else if (k.equals_ignore_ascii_case("NumSymbols")) {
            p.numSymbols = keel::max(1, roundI32(args[i + 1].toScalar()));
        } else if (k.equals_ignore_ascii_case("InsertDCNull") || k.equals_ignore_ascii_case("RemoveDCCarrier")) {
            p.insertDCNull = args[i + 1].toBool();
        }
    }
}

void ofdmStepImpl(Value &self, keel::ArrayRef<Value> args, keel::usize nargout,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "comm.OFDM.step");
    auto *mr = allocOf(ctx);
    const auto &p = *static_cast<const OfdmPayload *>(self.objectStateMut()->native.get());
    Value callArgs[3] = {
        args[0],
        Value::scalar(keel::as<keel::f64>(p.fftLength), mr),
        Value::scalar(keel::as<keel::f64>(p.cpLength), mr)
    };
    const char *fnName = p.isDemod ? "ofdmdemod" : "ofdmmod";
    if (!callExt(ctx, fnName, keel::ArrayRef<Value>(callArgs, 3), nargout, outs))
        throw Error("comm.OFDM: underlying ofdmmod/ofdmdemod unavailable.", 0, 0, "step", "", "numkit:comm:OfdmError");
}

BuiltinClass makeOfdmClass(const char *clsName, bool isDemod)
{
    BuiltinClass cls;
    cls.name = keel::String::from(clsName);
    cls.isHandle = true;
    cls.construct = [clsName, isDemod](keel::ArrayRef<Value> args, CallContext &ctx) -> Value {
        auto *mr = allocOf(ctx);
        auto payload = keel::make_arc_in<OfdmPayload>(mr);
        payload->isDemod = isDemod;
        ofdmApplyArgs(*payload, args);
        auto st = keel::make_arc_in<ObjectState>(mr, mr);
        st->native = keel::move(payload);
        return Value::object(clsName, keel::move(st), mr);
    };
    cls.subsref = &ofdmStepImpl;
    cls.methods.put(keel::String::from("step"), &ofdmStepImpl);
    auto noop = [](Value &, keel::ArrayRef<Value>, keel::usize,
                   keel::MutArrayRef<Value> outs, CallContext &) {
        if (!outs.empty()) outs[0] = Value();
    };
    cls.methods.put(keel::String::from("reset"), noop);
    cls.methods.put(keel::String::from("release"), noop);
    cls.propGet = [](const Value &self, keel::StringRef name, Value &out, CallContext &ctx) -> bool {
        auto *mr = allocOf(ctx);
        const auto &p = *static_cast<const OfdmPayload *>(self.objectStateConst()->native.get());
        if (name.equals_ignore_ascii_case("FFTLength")) { out = Value::scalar(keel::as<keel::f64>(p.fftLength), mr); return true; }
        if (name.equals_ignore_ascii_case("CyclicPrefixLength")) { out = Value::scalar(keel::as<keel::f64>(p.cpLength), mr); return true; }
        if (name.equals_ignore_ascii_case("NumSymbols")) { out = Value::scalar(keel::as<keel::f64>(p.numSymbols), mr); return true; }
        if (name.equals_ignore_ascii_case("InsertDCNull") || name.equals_ignore_ascii_case("RemoveDCCarrier")) {
            out = Value::logicalScalar(p.insertDCNull, mr); return true;
        }
        if (name.equals_ignore_ascii_case("NumGuardBandCarriers")) {
            Value g = Value::matrix(2, 1, ValueType::DOUBLE, mr);
            g.doubleDataMut()[0] = keel::as<keel::f64>(p.guardLeft);
            g.doubleDataMut()[1] = keel::as<keel::f64>(p.guardRight);
            out = keel::move(g);
            return true;
        }
        return false;
    };
    return cls;
}

void commOfdmMod_ctor_reg(keel::ArrayRef<Value> args, keel::usize, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    auto *mr = allocOf(ctx);
    auto payload = keel::make_arc_in<OfdmPayload>(mr);
    payload->isDemod = false;
    ofdmApplyArgs(*payload, args);
    auto st = keel::make_arc_in<ObjectState>(mr, mr);
    st->native = keel::move(payload);
    outs[0] = Value::object("comm.OFDMModulator", keel::move(st), mr);
}

void commOfdmDemod_ctor_reg(keel::ArrayRef<Value> args, keel::usize, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    auto *mr = allocOf(ctx);
    auto payload = keel::make_arc_in<OfdmPayload>(mr);
    payload->isDemod = true;
    ofdmApplyArgs(*payload, args);
    auto st = keel::make_arc_in<ObjectState>(mr, mr);
    st->native = keel::move(payload);
    outs[0] = Value::object("comm.OFDMDemodulator", keel::move(st), mr);
}

// ── 8. iqimbal ──────────────────────────────────────────────────────────
void iqimbal_argcheck(keel::ArrayRef<Value> args)
{
    argcheck::checkNargin(args.size(), 2, 3, "iqimbal");
    if (!args[0].isNumeric())
        argcheck::detail::fail("iqimbal: input 1 (x) must be numeric.", "iqimbal", "numkit:iqimbal:invalidInput");
    (void)argcheck::expectedReal(args[1], 2, "ampImb", "iqimbal");
    if (args.size() >= 3)
        (void)argcheck::expectedReal(args[2], 3, "phImb", "iqimbal");
}

void iqimbal_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    iqimbal_argcheck(args);
    auto *mr = allocOf(ctx);
    const Value &x = args[0];
    const keel::f64 ampDb = args[1].toScalar();
    const keel::f64 phDeg = (args.size() >= 3) ? args[2].toScalar() : 0.0;
    const keel::f64 g = keel::pow(10.0, ampDb / 40.0);
    const keel::f64 invG = 1.0 / g;
    const keel::f64 halfPhi = (phDeg * kPi / 180.0) * 0.5;
    const keel::f64 cp = keel::cos(halfPhi);
    const keel::f64 sp = keel::sin(halfPhi);

    Value y = Value::matrix(x.dims().rows(), x.dims().cols(), ValueType::COMPLEX, mr);
    for (keel::usize i = 0; i < x.numel(); ++i) {
        const keel::f64 re = x.isComplex() ? x.complexData()[i].re : x.elemAsDouble(i);
        const keel::f64 im = x.isComplex() ? x.complexData()[i].im : 0.0;
        const keel::f64 yr =  g * re * cp - invG * im * sp;
        const keel::f64 yi = -g * re * sp + invG * im * cp;
        y.complexDataMut()[i] = keel::c64{yr, yi};
    }
    outs[0] = keel::move(y);
}

// ── 9. eyediagram ───────────────────────────────────────────────────────
void eyediagram_argcheck(keel::ArrayRef<Value> args)
{
    argcheck::checkNargin(args.size(), 2, 6, "eyediagram");
    if (!args[0].isNumeric() || args[0].isEmpty())
        argcheck::detail::fail("eyediagram: input 1 (x) must be a nonempty numeric array.", "eyediagram", "numkit:eyediagram:invalidInput");
    (void)argcheck::expectedPositiveInteger(argcheck::expectedReal(args[1], 2, "n", "eyediagram"), 2, "n", "eyediagram");
}

void eyediagram_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                    keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    eyediagram_argcheck(args);
    auto *mr = allocOf(ctx);
    const Value &x = args[0];
    const keel::usize n = keel::as<keel::usize>(keel::max(1, roundI32(args[1].toScalar())));
    const keel::f64 period = (args.size() >= 3 && args[2].isScalar()) ? args[2].toScalar() : 1.0;
    const keel::usize offset = (args.size() >= 4 && args[3].isScalar())
        ? keel::as<keel::usize>(keel::max(0, roundI32(args[3].toScalar()))) : 0;

    Value figOut[1];
    (void)callExt(ctx, "figure", {}, 1, keel::MutArrayRef<Value>(figOut, 1));

    const keel::usize total = x.numel();
    const keel::usize nTraces = (total > offset) ? ((total - offset) / n) : 0;
    if (nTraces > 0) {
        Value tVec = Value::matrix(n, 1, ValueType::DOUBLE, mr);
        const keel::f64 dt = period / keel::as<keel::f64>(n);
        for (keel::usize k = 0; k < n; ++k)
            tVec.doubleDataMut()[k] = -0.5 * period + keel::as<keel::f64>(k) * dt;

        Value yMat = Value::matrix(n, nTraces, ValueType::DOUBLE, mr);
        for (keel::usize tr = 0; tr < nTraces; ++tr) {
            for (keel::usize k = 0; k < n; ++k) {
                const keel::usize src = offset + tr * n + k;
                const keel::f64 val = x.isComplex() ? x.complexData()[src].re : x.elemAsDouble(src);
                yMat.doubleDataMut()[k + tr * n] = val;
            }
        }
        Value plotArgs[2] = {tVec, yMat};
        Value dummy[1];
        (void)callExt(ctx, "plot", keel::ArrayRef<Value>(plotArgs, 2), 0, keel::MutArrayRef<Value>(dummy, 1));
        Value titleArg[1] = {Value::fromString("Eye Diagram", mr)};
        (void)callExt(ctx, "title", keel::ArrayRef<Value>(titleArg, 1), 0, keel::MutArrayRef<Value>(dummy, 1));
        Value xlArg[1] = {Value::fromString("Time", mr)};
        (void)callExt(ctx, "xlabel", keel::ArrayRef<Value>(xlArg, 1), 0, keel::MutArrayRef<Value>(dummy, 1));
        Value ylArg[1] = {Value::fromString("Amplitude", mr)};
        (void)callExt(ctx, "ylabel", keel::ArrayRef<Value>(ylArg, 1), 0, keel::MutArrayRef<Value>(dummy, 1));
    }
    outs[0] = (nargout >= 1) ? figOut[0] : Value();
}

// ── 10. scatterplot ─────────────────────────────────────────────────────
void scatterplot_argcheck(keel::ArrayRef<Value> args)
{
    argcheck::checkNargin(args.size(), 1, 5, "scatterplot");
    if (!args[0].isNumeric())
        argcheck::detail::fail("scatterplot: input 1 (x) must be numeric.", "scatterplot", "numkit:scatterplot:invalidInput");
}

void scatterplot_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                     keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    scatterplot_argcheck(args);
    auto *mr = allocOf(ctx);
    const Value &x = args[0];
    const keel::usize n = (args.size() >= 2 && args[1].isScalar())
        ? keel::as<keel::usize>(keel::max(1, roundI32(args[1].toScalar()))) : 1;
    const keel::usize offset = (args.size() >= 3 && args[2].isScalar())
        ? keel::as<keel::usize>(keel::max(0, roundI32(args[2].toScalar()))) : 0;
    const keel::String plotStr = (args.size() >= 4 && (args[3].isChar() || args[3].isString()))
        ? args[3].toString() : keel::String::from("b.");

    Value figOut[1];
    if (args.size() >= 5 && args[4].isObject()) {
        figOut[0] = args[4];
        Value fargs[1] = {args[4]};
        Value dummy[1];
        (void)callExt(ctx, "figure", keel::ArrayRef<Value>(fargs, 1), 0, keel::MutArrayRef<Value>(dummy, 1));
    } else {
        (void)callExt(ctx, "figure", {}, 1, keel::MutArrayRef<Value>(figOut, 1));
    }

    const keel::usize total = x.numel();
    keel::usize count = 0;
    if (total > offset)
        count = (total - offset + n - 1) / n;
    Value re = Value::matrix(count, 1, ValueType::DOUBLE, mr);
    Value im = Value::matrix(count, 1, ValueType::DOUBLE, mr);
    for (keel::usize k = 0; k < count; ++k) {
        const keel::usize src = offset + k * n;
        re.doubleDataMut()[k] = x.isComplex() ? x.complexData()[src].re : x.elemAsDouble(src);
        im.doubleDataMut()[k] = x.isComplex() ? x.complexData()[src].im : 0.0;
    }
    Value plotArgs[3] = {re, im, Value::fromString(plotStr, mr)};
    Value dummy[1];
    (void)callExt(ctx, "plot", keel::ArrayRef<Value>(plotArgs, 3), 0, keel::MutArrayRef<Value>(dummy, 1));
    Value titleArg[1] = {Value::fromString("Scatter plot", mr)};
    (void)callExt(ctx, "title", keel::ArrayRef<Value>(titleArg, 1), 0, keel::MutArrayRef<Value>(dummy, 1));
    Value xlArg[1] = {Value::fromString("In-Phase", mr)};
    (void)callExt(ctx, "xlabel", keel::ArrayRef<Value>(xlArg, 1), 0, keel::MutArrayRef<Value>(dummy, 1));
    Value ylArg[1] = {Value::fromString("Quadrature", mr)};
    (void)callExt(ctx, "ylabel", keel::ArrayRef<Value>(ylArg, 1), 0, keel::MutArrayRef<Value>(dummy, 1));
    outs[0] = (nargout >= 1) ? figOut[0] : Value();
}

// ── 8. comm.ConstellationDiagram ────────────────────────────────────────
struct ConstellationPayload final : public NativePayload {
    Value refConstellation;
    keel::String symSource;
    keel::i32 symToDisplay = 1000;
    keel::String name;
    keel::String title;
    bool showGrid = true;
    keel::Array<keel::f64> position;
    keel::Array<keel::f64> xLimits;
    keel::Array<keel::f64> yLimits;

    explicit ConstellationPayload(keel::Allocator *mr = nullptr)
        : symSource(keel::String::from("Auto"))
        , name(keel::String::from("Constellation Diagram"))
        , title(keel::String::from(""))
        , position(mr)
        , xLimits(mr)
        , yLimits(mr)
    {
        position.push_back(100.0);
        position.push_back(100.0);
        position.push_back(460.0);
        position.push_back(460.0);
        xLimits.push_back(-1.5);
        xLimits.push_back(1.5);
        yLimits.push_back(-1.5);
        yLimits.push_back(1.5);
    }

    keel::Arc<NativePayload> clone(keel::Allocator *alloc = nullptr) const override
    {
        auto *a = alloc ? alloc : keel::default_allocator();
        auto p = keel::make_arc_in<ConstellationPayload>(a, a);
        p->refConstellation = refConstellation;
        p->symSource = symSource.clone();
        p->symToDisplay = symToDisplay;
        p->name = name.clone();
        p->title = title.clone();
        p->showGrid = showGrid;
        p->position = position.clone(a);
        p->xLimits = xLimits.clone(a);
        p->yLimits = yLimits.clone(a);
        return p;
    }
};

void constDiagApplyNV(ConstellationPayload &p, keel::ArrayRef<Value> args, keel::Allocator *mr)
{
    auto readVec = [&](const Value &v) {
        keel::Array<keel::f64> out(mr);
        out.reserve(v.numel());
        for (keel::usize k = 0; k < v.numel(); ++k) out.push_back(v.elemAsDouble(k));
        return out;
    };
    for (keel::usize i = 0; i + 1 < args.size(); i += 2) {
        if (!args[i].isChar() && !args[i].isString()) continue;
        const keel::String k = args[i].toString();
        if (k.equals_ignore_ascii_case("ReferenceConstellation")) p.refConstellation = args[i + 1];
        else if (k.equals_ignore_ascii_case("SymbolsToDisplaySource")) p.symSource = args[i + 1].toString();
        else if (k.equals_ignore_ascii_case("SymbolsToDisplay")) p.symToDisplay = keel::max(1, roundI32(args[i + 1].toScalar()));
        else if (k.equals_ignore_ascii_case("Name")) p.name = args[i + 1].toString();
        else if (k.equals_ignore_ascii_case("Title")) p.title = args[i + 1].toString();
        else if (k.equals_ignore_ascii_case("ShowGrid")) p.showGrid = args[i + 1].toBool();
        else if (k.equals_ignore_ascii_case("Position")) p.position = readVec(args[i + 1]);
        else if (k.equals_ignore_ascii_case("XLimits")) p.xLimits = readVec(args[i + 1]);
        else if (k.equals_ignore_ascii_case("YLimits")) p.yLimits = readVec(args[i + 1]);
    }
}

void constDiagStepImpl(Value & /*self*/, keel::ArrayRef<Value> args, keel::usize,
                       keel::MutArrayRef<Value> outs, CallContext &)
{
    argcheck::checkNargin(args.size(), 1, 8, "comm.ConstellationDiagram.step");
    if (!outs.empty()) outs[0] = Value();
}

BuiltinClass makeConstellationDiagramClass()
{
    BuiltinClass cls;
    cls.name = keel::String::from("comm.ConstellationDiagram");
    cls.isHandle = true;
    cls.construct = [](keel::ArrayRef<Value> args, CallContext &ctx) -> Value {
        auto *mr = allocOf(ctx);
        auto payload = keel::make_arc_in<ConstellationPayload>(mr, mr);
        constDiagApplyNV(*payload, args, mr);
        auto st = keel::make_arc_in<ObjectState>(mr, mr);
        st->native = keel::move(payload);
        return Value::object("comm.ConstellationDiagram", keel::move(st), mr);
    };
    cls.subsref = &constDiagStepImpl;
    cls.methods.put(keel::String::from("step"), &constDiagStepImpl);
    auto noop = [](Value &, keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value> outs, CallContext &) {
        if (!outs.empty()) outs[0] = Value();
    };
    cls.methods.put(keel::String::from("reset"), noop);
    cls.methods.put(keel::String::from("release"), noop);
    cls.methods.put(keel::String::from("show"), noop);
    cls.methods.put(keel::String::from("hide"), noop);
    cls.propGet = [](const Value &self, keel::StringRef name, Value &out, CallContext &ctx) -> bool {
        auto *mr = allocOf(ctx);
        const auto &p = *static_cast<const ConstellationPayload *>(self.objectStateConst()->native.get());
        auto makeRow = [&](const keel::Array<keel::f64> &a) {
            Value v = Value::matrix(1, a.size(), ValueType::DOUBLE, mr);
            for (keel::usize k = 0; k < a.size(); ++k) v.doubleDataMut()[k] = a[k];
            return v;
        };
        if (name.equals_ignore_ascii_case("ReferenceConstellation")) { out = p.refConstellation; return true; }
        if (name.equals_ignore_ascii_case("SymbolsToDisplaySource")) { out = Value::fromString(p.symSource, mr); return true; }
        if (name.equals_ignore_ascii_case("SymbolsToDisplay")) { out = Value::scalar(keel::as<keel::f64>(p.symToDisplay), mr); return true; }
        if (name.equals_ignore_ascii_case("Name")) { out = Value::fromString(p.name, mr); return true; }
        if (name.equals_ignore_ascii_case("Title")) { out = Value::fromString(p.title, mr); return true; }
        if (name.equals_ignore_ascii_case("ShowGrid")) { out = Value::logicalScalar(p.showGrid, mr); return true; }
        if (name.equals_ignore_ascii_case("Position")) { out = makeRow(p.position); return true; }
        if (name.equals_ignore_ascii_case("XLimits")) { out = makeRow(p.xLimits); return true; }
        if (name.equals_ignore_ascii_case("YLimits")) { out = makeRow(p.yLimits); return true; }
        return false;
    };
    cls.propSet = [](Value &self, keel::StringRef name, const Value &val, CallContext &ctx) -> bool {
        auto *mr = allocOf(ctx);
        auto &p = *static_cast<ConstellationPayload *>(self.objectStateMut()->native.get());
        auto readVec = [&](const Value &v) {
            keel::Array<keel::f64> out(mr);
            out.reserve(v.numel());
            for (keel::usize k = 0; k < v.numel(); ++k) out.push_back(v.elemAsDouble(k));
            return out;
        };
        if (name.equals_ignore_ascii_case("ReferenceConstellation")) { p.refConstellation = val; return true; }
        if (name.equals_ignore_ascii_case("SymbolsToDisplaySource")) { p.symSource = val.toString(); return true; }
        if (name.equals_ignore_ascii_case("SymbolsToDisplay")) { p.symToDisplay = keel::max(1, roundI32(val.toScalar())); return true; }
        if (name.equals_ignore_ascii_case("Name")) { p.name = val.toString(); return true; }
        if (name.equals_ignore_ascii_case("Title")) { p.title = val.toString(); return true; }
        if (name.equals_ignore_ascii_case("ShowGrid")) { p.showGrid = val.toBool(); return true; }
        if (name.equals_ignore_ascii_case("Position")) { p.position = readVec(val); return true; }
        if (name.equals_ignore_ascii_case("XLimits")) { p.xLimits = readVec(val); return true; }
        if (name.equals_ignore_ascii_case("YLimits")) { p.yLimits = readVec(val); return true; }
        return true;
    };
    return cls;
}

void commConstDiag_ctor_reg(keel::ArrayRef<Value> args, keel::usize, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    auto *mr = allocOf(ctx);
    auto payload = keel::make_arc_in<ConstellationPayload>(mr, mr);
    constDiagApplyNV(*payload, args, mr);
    auto st = keel::make_arc_in<ObjectState>(mr, mr);
    st->native = keel::move(payload);
    outs[0] = Value::object("comm.ConstellationDiagram", keel::move(st), mr);
}

// ── 9. comm.SymbolSynchronizer ──────────────────────────────────────────
struct SymbolSyncPayload final : public NativePayload {
    keel::String tedMethod;
    keel::i32 sps = 2;
    keel::f64 dampingFactor = 1.0;
    keel::f64 normLoopBw = 0.01;
    keel::f64 detectorGain = 2.7;

    // Loop states (Rice 2009 piecewise parabolic interpolator + PI loop filter + mod-1 NCO)
    keel::c64 interpBuf[3] = {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
    keel::c64 tedBuf[4] = {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
    keel::f64 loopFiltState = 0.0;
    keel::f64 loopPrevIn = 0.0;
    keel::f64 ncoCounter = 0.0;
    keel::f64 mu = 0.0;
    bool prevStrobe = false;
    keel::i32 strobeHistory = 0;

    SymbolSyncPayload()
        : tedMethod(keel::String::from("Zero-Crossing (decision-directed)"))
    {}

    void resetStates()
    {
        for (keel::usize i = 0; i < 3; ++i) interpBuf[i] = {0.0, 0.0};
        for (keel::usize i = 0; i < 4; ++i) tedBuf[i] = {0.0, 0.0};
        loopFiltState = 0.0;
        loopPrevIn = 0.0;
        ncoCounter = 0.0;
        mu = 0.0;
        prevStrobe = false;
        strobeHistory = 0;
    }

    keel::Arc<NativePayload> clone(keel::Allocator *alloc = nullptr) const override
    {
        auto *a = alloc ? alloc : keel::default_allocator();
        auto p = keel::make_arc_in<SymbolSyncPayload>(a);
        p->tedMethod = tedMethod.clone();
        p->sps = sps;
        p->dampingFactor = dampingFactor;
        p->normLoopBw = normLoopBw;
        p->detectorGain = detectorGain;
        for (keel::usize i = 0; i < 3; ++i) p->interpBuf[i] = interpBuf[i];
        for (keel::usize i = 0; i < 4; ++i) p->tedBuf[i] = tedBuf[i];
        p->loopFiltState = loopFiltState;
        p->loopPrevIn = loopPrevIn;
        p->ncoCounter = ncoCounter;
        p->mu = mu;
        p->prevStrobe = prevStrobe;
        p->strobeHistory = strobeHistory;
        return p;
    }
};

void symSyncApplyNV(SymbolSyncPayload &p, keel::ArrayRef<Value> args)
{
    for (keel::usize i = 0; i + 1 < args.size(); i += 2) {
        if (!args[i].isChar() && !args[i].isString()) continue;
        const keel::String k = args[i].toString();
        if (k.equals_ignore_ascii_case("TimingErrorDetector")) p.tedMethod = args[i + 1].toString();
        else if (k.equals_ignore_ascii_case("SamplesPerSymbol")) { p.sps = keel::max(2, roundI32(args[i + 1].toScalar())); p.resetStates(); }
        else if (k.equals_ignore_ascii_case("DampingFactor")) p.dampingFactor = args[i + 1].toScalar();
        else if (k.equals_ignore_ascii_case("NormalizedLoopBandwidth")) p.normLoopBw = args[i + 1].toScalar();
        else if (k.equals_ignore_ascii_case("DetectorGain")) p.detectorGain = args[i + 1].toScalar();
    }
}

void symSyncStepImpl(Value &self, keel::ArrayRef<Value> args, keel::usize nargout,
                     keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "comm.SymbolSynchronizer.step");
    auto *mr = allocOf(ctx);
    auto &p = *static_cast<SymbolSyncPayload *>(self.objectStateMut()->native.get());

    const Value &x = args[0];
    const keel::usize N = x.numel();
    const keel::f64 spsF = keel::as<keel::f64>(keel::max(2, p.sps));
    const keel::f64 zeta = (p.dampingFactor > 0.0) ? p.dampingFactor : 1.0;
    const keel::f64 bn = (p.normLoopBw > 0.0) ? p.normLoopBw : 0.01;
    const keel::f64 kp = (p.detectorGain != 0.0) ? p.detectorGain : 2.7;
    const keel::f64 theta = bn / (spsF * (zeta + 0.25 / zeta));
    const keel::f64 d = (1.0 + 2.0 * zeta * theta + theta * theta) * kp;
    const keel::f64 k1 = (4.0 * zeta * theta) / d;
    const keel::f64 k2 = (4.0 * theta * theta) / d;
    const bool isGardner = p.tedMethod.as_ref().find('G') != keel::kInvalidIndex
                        || p.tedMethod.as_ref().find('g') != keel::kInvalidIndex;

    keel::Array<keel::c64> syms(mr);
    syms.reserve(N / keel::as<keel::usize>(keel::max(1, p.sps)) + 4);
    keel::Array<keel::f64> tErrOut(mr);
    if (nargout >= 2) tErrOut.reserve(N);

    for (keel::usize i = 0; i < N; ++i) {
        const keel::c64 xin = x.isComplex()
            ? x.complexData()[i]
            : keel::c64{x.elemAsDouble(i), 0.0};

        // Piecewise parabolic Farrow interpolator (alpha = 0.5)
        const keel::c64 x0 = xin;
        const keel::c64 x1 = p.interpBuf[0];
        const keel::c64 x2 = p.interpBuf[1];
        const keel::c64 x3 = p.interpBuf[2];
        const keel::c64 v2{0.5 * (x0.re - x1.re - x2.re + x3.re),
                           0.5 * (x0.im - x1.im - x2.im + x3.im)};
        const keel::c64 v1{-0.5 * x0.re + 1.5 * x1.re - 0.5 * x2.re - 0.5 * x3.re,
                           -0.5 * x0.im + 1.5 * x1.im - 0.5 * x2.im - 0.5 * x3.im};
        const keel::c64 v0 = x2;
        const keel::c64 intOut{(v2.re * p.mu + v1.re) * p.mu + v0.re,
                               (v2.im * p.mu + v1.im) * p.mu + v0.im};

        p.interpBuf[2] = p.interpBuf[1];
        p.interpBuf[1] = p.interpBuf[0];
        p.interpBuf[0] = xin;

        keel::f64 e = 0.0;
        if (p.prevStrobe) {
            syms.push_back(intOut);
            const keel::c64 yPrev = p.tedBuf[1];
            const keel::c64 yMid = p.tedBuf[0];
            if (isGardner) {
                e = yMid.re * (yPrev.re - intOut.re) + yMid.im * (yPrev.im - intOut.im);
            } else {
                auto sgn = [](keel::f64 v) -> keel::f64 { return (v > 0.0) ? 1.0 : ((v < 0.0) ? -1.0 : 0.0); };
                e = yMid.re * (sgn(yPrev.re) - sgn(intOut.re))
                  + yMid.im * (sgn(yPrev.im) - sgn(intOut.im));
            }
        }
        if (nargout >= 2) tErrOut.push_back(p.mu);

        // Update TED shift register
        p.tedBuf[1] = p.tedBuf[0];
        p.tedBuf[0] = intOut;

        // PI loop filter
        const keel::f64 loopOut = p.loopFiltState + k1 * e;
        p.loopFiltState += k2 * e;

        // Modulo-1 NCO
        const keel::f64 w = (1.0 / spsF) + loopOut;
        const keel::f64 nextCounter = p.ncoCounter - w;
        if (nextCounter < 0.0) {
            p.prevStrobe = true;
            p.mu = (w > 1e-12) ? (p.ncoCounter / w) : 0.0;
            p.mu = keel::clamp(p.mu, 0.0, 1.0);
            p.ncoCounter = nextCounter + 1.0;
        } else {
            p.prevStrobe = false;
            p.ncoCounter = nextCounter;
        }
    }

    Value y = Value::matrix(syms.size(), 1, x.isComplex() ? ValueType::COMPLEX : ValueType::DOUBLE, mr);
    for (keel::usize i = 0; i < syms.size(); ++i) {
        if (x.isComplex()) y.complexDataMut()[i] = syms[i];
        else y.doubleDataMut()[i] = syms[i].re;
    }
    outs[0] = keel::move(y);
    if (nargout >= 2 && outs.size() > 1) {
        Value te = Value::matrix(tErrOut.size(), 1, ValueType::DOUBLE, mr);
        for (keel::usize i = 0; i < tErrOut.size(); ++i) te.doubleDataMut()[i] = tErrOut[i];
        outs[1] = keel::move(te);
    }
}

BuiltinClass makeSymbolSyncClass()
{
    BuiltinClass cls;
    cls.name = keel::String::from("comm.SymbolSynchronizer");
    cls.isHandle = true;
    cls.construct = [](keel::ArrayRef<Value> args, CallContext &ctx) -> Value {
        auto *mr = allocOf(ctx);
        auto payload = keel::make_arc_in<SymbolSyncPayload>(mr);
        symSyncApplyNV(*payload, args);
        auto st = keel::make_arc_in<ObjectState>(mr, mr);
        st->native = keel::move(payload);
        return Value::object("comm.SymbolSynchronizer", keel::move(st), mr);
    };
    cls.subsref = &symSyncStepImpl;
    cls.methods.put(keel::String::from("step"), &symSyncStepImpl);
    cls.methods.put(keel::String::from("reset"),
                    [](Value &self, keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value> outs, CallContext &) {
                        if (ObjectState *os = self.objectStateMut(); os && os->native)
                            static_cast<SymbolSyncPayload *>(os->native.get())->resetStates();
                        if (!outs.empty()) outs[0] = Value();
                    });
    cls.methods.put(keel::String::from("release"),
                    [](Value &self, keel::ArrayRef<Value>, keel::usize, keel::MutArrayRef<Value> outs, CallContext &) {
                        if (ObjectState *os = self.objectStateMut(); os && os->native)
                            static_cast<SymbolSyncPayload *>(os->native.get())->resetStates();
                        if (!outs.empty()) outs[0] = Value();
                    });
    cls.propGet = [](const Value &self, keel::StringRef name, Value &out, CallContext &ctx) -> bool {
        auto *mr = allocOf(ctx);
        const auto &p = *static_cast<const SymbolSyncPayload *>(self.objectStateConst()->native.get());
        if (name.equals_ignore_ascii_case("TimingErrorDetector")) { out = Value::fromString(p.tedMethod, mr); return true; }
        if (name.equals_ignore_ascii_case("SamplesPerSymbol")) { out = Value::scalar(keel::as<keel::f64>(p.sps), mr); return true; }
        if (name.equals_ignore_ascii_case("DampingFactor")) { out = Value::scalar(p.dampingFactor, mr); return true; }
        if (name.equals_ignore_ascii_case("NormalizedLoopBandwidth")) { out = Value::scalar(p.normLoopBw, mr); return true; }
        if (name.equals_ignore_ascii_case("DetectorGain")) { out = Value::scalar(p.detectorGain, mr); return true; }
        return false;
    };
    cls.propSet = [](Value &self, keel::StringRef name, const Value &val, CallContext &) -> bool {
        auto &p = *static_cast<SymbolSyncPayload *>(self.objectStateMut()->native.get());
        if (name.equals_ignore_ascii_case("TimingErrorDetector")) { p.tedMethod = val.toString(); return true; }
        if (name.equals_ignore_ascii_case("SamplesPerSymbol")) { p.sps = keel::max(2, roundI32(val.toScalar())); p.resetStates(); return true; }
        if (name.equals_ignore_ascii_case("DampingFactor")) { p.dampingFactor = val.toScalar(); return true; }
        if (name.equals_ignore_ascii_case("NormalizedLoopBandwidth")) { p.normLoopBw = val.toScalar(); return true; }
        if (name.equals_ignore_ascii_case("DetectorGain")) { p.detectorGain = val.toScalar(); return true; }
        return false;
    };
    return cls;
}

void commSymbolSync_ctor_reg(keel::ArrayRef<Value> args, keel::usize, keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    auto *mr = allocOf(ctx);
    auto payload = keel::make_arc_in<SymbolSyncPayload>(mr);
    symSyncApplyNV(*payload, args);
    auto st = keel::make_arc_in<ObjectState>(mr, mr);
    st->native = keel::move(payload);
    outs[0] = Value::object("comm.SymbolSynchronizer", keel::move(st), mr);
}

} // anonymous namespace

void installCommSystemObjects(PluginContext &engine)
{
    auto regCls = [&](BuiltinClass cls) {
        addCloneMethod(cls);
        engine.registerClass(keel::move(cls));
    };

    regCls(makeBarkerClass());
    engine.registerFunction("comm", "BarkerCode", &commBarkerCode_ctor_reg);

    regCls(makeDbpskClass("comm.DBPSKModulator", false));
    engine.registerFunction("comm", "DBPSKModulator", &commDbpskMod_ctor_reg);
    regCls(makeDbpskClass("comm.DBPSKDemodulator", true));
    engine.registerFunction("comm", "DBPSKDemodulator", &commDbpskDemod_ctor_reg);

    regCls(makeQpskClass("comm.QPSKModulator", false));
    engine.registerFunction("comm", "QPSKModulator", &commQpskMod_ctor_reg);
    regCls(makeQpskClass("comm.QPSKDemodulator", true));
    engine.registerFunction("comm", "QPSKDemodulator", &commQpskDemod_ctor_reg);

    regCls(makeRaisedCosineClass("comm.RaisedCosineTransmitFilter", false));
    engine.registerFunction("comm", "RaisedCosineTransmitFilter", &commRcosTx_ctor_reg);
    regCls(makeRaisedCosineClass("comm.RaisedCosineReceiveFilter", true));
    engine.registerFunction("comm", "RaisedCosineReceiveFilter", &commRcosRx_ctor_reg);

    regCls(makeAwgnChannelClass());
    engine.registerFunction("comm", "AWGNChannel", &commAwgnChannel_ctor_reg);

    regCls(makeCrcClass("comm.CRCGenerator", false));
    engine.registerFunction("comm", "CRCGenerator", &commCrcGen_ctor_reg);
    regCls(makeCrcClass("comm.CRCDetector", true));
    engine.registerFunction("comm", "CRCDetector", &commCrcDet_ctor_reg);

    regCls(makeOfdmClass("comm.OFDMModulator", false));
    engine.registerFunction("comm", "OFDMModulator", &commOfdmMod_ctor_reg);
    regCls(makeOfdmClass("comm.OFDMDemodulator", true));
    engine.registerFunction("comm", "OFDMDemodulator", &commOfdmDemod_ctor_reg);

    regCls(makeConstellationDiagramClass());
    engine.registerFunction("comm", "ConstellationDiagram", &commConstDiag_ctor_reg);
    regCls(makeSymbolSyncClass());
    engine.registerFunction("comm", "SymbolSynchronizer", &commSymbolSync_ctor_reg);

    engine.registerFunction("comm.rf", "iqimbal", &iqimbal_reg);
    engine.registerFunction("comm.vis", "eyediagram", &eyediagram_reg);
    engine.registerFunction("comm.vis", "scatterplot", &scatterplot_reg);
}

} // namespace numkit::comm::detail
