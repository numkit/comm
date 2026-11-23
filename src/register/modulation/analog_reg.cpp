// toolboxes/comm/src/modulation/analog_reg.cpp
//
// Register half of the comm analog-modulation builtins: the CallContext
// wrappers pmmod / ammod / fmmod / mskmod / ssbmod that parse the carrier /
// deviation / phase / sideband options and delegate to the engine-free
// compute in analog.cpp. library.cpp forward-declares + registers these by
// name.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <numkit/comm/modulation/analog.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, Span, ctx.engine->resource()
#include <numkit/value/value.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
namespace numkit::comm {
namespace detail {

void pmmod_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
               keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 4, 5, "pmmod");
    const keel::f64 fc = args[1].toScalar();
    const keel::f64 fs = args[2].toScalar();
    const keel::f64 phasedev = args[3].toScalar();
    keel::f64 ini_phase = 0.0;
    if (args.size() >= 5 && !args[4].isEmpty()) {
        ini_phase = args[4].toScalar();
    }
    outs[0] = pmmod(args[0], fc, fs, phasedev, ini_phase,
                    ctx.engine->allocator());
}

void ammod_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
               keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 5, "ammod");
    const keel::f64 fc = args[1].toScalar();
    const keel::f64 fs = args[2].toScalar();
    keel::f64 ini_phase = 0.0;
    keel::f64 carr_amp  = 0.0;
    if (args.size() >= 4 && !args[3].isEmpty())
        ini_phase = args[3].toScalar();
    if (args.size() >= 5 && !args[4].isEmpty())
        carr_amp = args[4].toScalar();
    outs[0] = ammod(args[0], fc, fs, ini_phase, carr_amp,
                    ctx.engine->allocator());
}

void fmmod_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
               keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 4, 5, "fmmod");
    const keel::f64 fc = args[1].toScalar();
    const keel::f64 fs = args[2].toScalar();
    const keel::f64 freqdev = args[3].toScalar();
    keel::f64 ini_phase = 0.0;
    if (args.size() >= 5 && !args[4].isEmpty())
        ini_phase = args[4].toScalar();
    outs[0] = fmmod(args[0], fc, fs, freqdev, ini_phase,
                    ctx.engine->allocator());
}

void pmdemod_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 4, 5, "pmdemod");
    keel::f64 ini_phase = 0.0;
    if (args.size() >= 5 && !args[4].isEmpty())
        ini_phase = args[4].toScalar();
    outs[0] = pmdemod(args[0], args[1].toScalar(), args[2].toScalar(),
                      args[3].toScalar(), ini_phase, ctx.engine->allocator());
}

void fmdemod_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 4, 5, "fmdemod");
    keel::f64 ini_phase = 0.0;
    if (args.size() >= 5 && !args[4].isEmpty())
        ini_phase = args[4].toScalar();
    outs[0] = fmdemod(args[0], args[1].toScalar(), args[2].toScalar(),
                      args[3].toScalar(), ini_phase, ctx.engine->allocator());
}

void amdemod_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 7, "amdemod");
    keel::f64 ini_phase = 0.0, carr_amp = 0.0;
    if (args.size() >= 4 && !args[3].isEmpty()) ini_phase = args[3].toScalar();
    if (args.size() >= 5 && !args[4].isEmpty()) carr_amp = args[4].toScalar();
    outs[0] = amdemod(args[0], args[1].toScalar(), args[2].toScalar(),
                      ini_phase, carr_amp, ctx.engine->allocator());
}

void ssbdemod_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 6, "ssbdemod");
    keel::f64 ini_phase = 0.0;
    if (args.size() >= 4 && !args[3].isEmpty()) ini_phase = args[3].toScalar();
    outs[0] = ssbdemod(args[0], args[1].toScalar(), args[2].toScalar(),
                       ini_phase, ctx.engine->allocator());
}

namespace {

Value nondiffEncodeBits(const Value &x, keel::Allocator *mr)
{
    const auto &d = x.dims();
    keel::usize H = d.rows();
    keel::usize W = d.cols();
    const bool was_row = (H == 1 && W >= 1);
    if (was_row) keel::swap(H, W);

    Value out = was_row ? Value::matrix(1, H, ValueType::DOUBLE, mr)
                        : Value::matrix(H, W, ValueType::DOUBLE, mr);
    keel::f64 *o = out.doubleDataMut();
    for (keel::usize c = 0; c < W; ++c) {
        keel::i32 prevA = 1; // initial I-rail state +1
        for (keel::usize r = 0; r < H; ++r) {
            const keel::usize idx = was_row ? r : (c * H + r);
            const keel::f64 xi = x.elemAsDouble(idx);
            if (xi != 0.0 && xi != 1.0)
                throw Error("mskmod: input must be binary (0 or 1)",
                            0, 0, "mskmod", "", "numkit:mskmod:NotBinary");
            const keel::i32 ak = (xi != 0.0) ? 1 : -1;
            const keel::i32 dk = ((r % 2) == 0) ? (prevA * ak) : (-prevA * ak);
            o[idx] = (dk > 0) ? 1.0 : 0.0;
            prevA = ak;
        }
    }
    return out;
}

Value nondiffDecodeBits(const Value &diffBits, keel::Allocator *mr)
{
    const auto &d = diffBits.dims();
    keel::usize H = d.rows();
    keel::usize W = d.cols();
    const bool was_row = (H == 1 && W >= 1);
    if (was_row) keel::swap(H, W);

    Value out = was_row ? Value::matrix(1, H, ValueType::DOUBLE, mr)
                        : Value::matrix(H, W, ValueType::DOUBLE, mr);
    keel::f64 *o = out.doubleDataMut();
    for (keel::usize c = 0; c < W; ++c) {
        keel::i32 prevA = 1;
        for (keel::usize r = 0; r < H; ++r) {
            const keel::usize idx = was_row ? r : (c * H + r);
            const keel::i32 dk = (diffBits.elemAsDouble(idx) != 0.0) ? 1 : -1;
            const keel::i32 ak = ((r % 2) == 0) ? (prevA * dk) : (-prevA * dk);
            o[idx] = (ak > 0) ? 1.0 : 0.0;
            prevA = ak;
        }
    }
    return out;
}

} // namespace

void mskmod_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 4, "mskmod");
    const keel::i32 nSamp = argcheck::expectedPositiveInteger(
        argcheck::expectedReal(args[1], 2, "nSamp", "mskmod"), 2, "nSamp", "mskmod");
    auto *mr = ctx.engine->allocator();
    bool nondiff = false;
    keel::f64 ini_phase = 0.0;
    keel::usize i = 2;
    if (i < args.size() && (args[i].isChar() || args[i].isString())) {
        const keel::String enc = args[i].toString();
        keel::String elo;
        for (char ch : enc) elo.push_back(keel::to_ascii_lower(ch));
        if (elo == "nondiff") nondiff = true;
        else if (elo == "diff") nondiff = false;
        else
            throw Error("mskmod: dataenc must be 'diff' or 'nondiff'",
                        0, 0, "mskmod", "", "numkit:mskmod:dataenc");
        ++i;
    }
    if (i < args.size() && !args[i].isEmpty())
        ini_phase = args[i].toScalar();

    Value xEff = nondiff ? nondiffEncodeBits(args[0], mr) : args[0];
    outs[0] = mskmod(xEff, nSamp, ini_phase, mr);
    if (nargout >= 2 && outs.size() >= 2) {
        keel::f64 phaseOut = 0.0;
        (void)mskdemod(outs[0], nSamp, ini_phase, &phaseOut, mr);
        outs[1] = Value::scalar(phaseOut, mr);
    }
}

void mskdemod_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 5, "mskdemod");
    const keel::i32 nSamp = argcheck::expectedPositiveInteger(
        argcheck::expectedReal(args[1], 2, "nSamp", "mskdemod"), 2, "nSamp", "mskdemod");
    auto *mr = ctx.engine->allocator();
    bool nondiff = false;
    keel::f64 ini_phase = 0.0;
    keel::usize i = 2;
    if (i < args.size() && (args[i].isChar() || args[i].isString())) {
        const keel::String enc = args[i].toString();
        keel::String elo;
        for (char ch : enc) elo.push_back(keel::to_ascii_lower(ch));
        if (elo == "nondiff") nondiff = true;
        else if (elo == "diff") nondiff = false;
        else
            throw Error("mskdemod: dataenc must be 'diff' or 'nondiff'",
                        0, 0, "mskdemod", "", "numkit:mskdemod:dataenc");
        ++i;
    }
    if (i < args.size() && !args[i].isEmpty())
        ini_phase = args[i].toScalar();

    keel::f64 phaseOut = 0.0;
    Value bits = mskdemod(args[0], nSamp, ini_phase,
                          nargout >= 2 ? &phaseOut : nullptr, mr);
    if (nondiff)
        bits = nondiffDecodeBits(bits, mr);
    outs[0] = keel::move(bits);
    if (outs.size() >= 2)
        outs[1] = Value::scalar(phaseOut, mr);
}

void ssbmod_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 5, "ssbmod");
    const keel::f64 fc = args[1].toScalar();
    const keel::f64 fs = args[2].toScalar();
    keel::f64 ini_phase = 0.0;
    if (args.size() >= 4 && !args[3].isEmpty())
        ini_phase = args[3].toScalar();
    bool upper = false;
    if (args.size() >= 5 && !args[4].isEmpty()) {
        if (!args[4].isChar() && !args[4].isString())
            throw Error("ssbmod: method must be a string ('upper')",
                        0, 0, "ssbmod", "", "numkit:ssbmod:InvStr");
        keel::String m = args[4].toString();
        // Match MATLAB behaviour: any string containing 'up' selects USB.
        m.to_ascii_lower_in_place();
        upper = m.contains("up");
        if (!upper && !m.empty())
            throw Error("ssbmod: method must be 'upper'",
                        0, 0, "ssbmod", "", "numkit:ssbmod:InvStr");
    }
    outs[0] = ssbmod(args[0], fc, fs, ini_phase, upper,
                     ctx.engine->allocator());
}

} // namespace detail
} // namespace numkit::comm
