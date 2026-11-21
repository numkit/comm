// toolboxes/comm/src/source/base_conversions_reg.cpp
//
// Register half of the comm base-conversion builtins: the CallContext
// wrappers (bit2int / int2bit / bi2de / de2bi / vec2mat) that parse args
// and delegate to the engine-free compute in base_conversions.cpp.
// The 'left-msb'/'right-msb' flag parser is register-side (only the legacy
// bi2de/de2bi string-option forms use it). library.cpp forward-declares +
// registers these by name.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <keel/core.h>
#include <keel/format.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <numkit/comm/source/base_conversions.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, Span, ctx.engine->resource()
#include <numkit/value/value.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>

namespace numkit::comm {

namespace {

// Detect 'left-msb' vs 'right-msb' option (MATLAB style for bi2de/de2bi).
// Returns true if MSB-first, false if LSB-first.  bi2de defaults to
// LSB-first ('right-msb'), de2bi same.
bool parseMsbFlag(const Value &v, const char *who)
{
    if (!v.isChar() && !v.isString())
        throw Error(keel::format("{}: flag must be 'left-msb' or 'right-msb'", who),
                    0, 0, who, "", keel::format("numkit:{}:BadFlag", who));
    const keel::String s = v.toString();
    if (s == "left-msb")  return true;
    if (s == "right-msb") return false;
    throw Error(keel::format("{}: unknown flag '{}'", who, s),
                0, 0, who, "", keel::format("numkit:{}:BadFlag", who));
}

} // namespace

namespace detail {

void bit2int_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 5, "bit2int");
    const int n = argcheck::expectedPositiveInteger(
        argcheck::expectedReal(args[1], 2, "n", "bit2int"), 2, "n", "bit2int");
    bool msb = true;
    if (args.size() >= 3) msb = (args[2].toScalar() != 0.0);
    outs[0] = bit2int(args[0], n, msb, ctx.engine->allocator());
}

void int2bit_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "int2bit");
    const int n = argcheck::expectedPositiveInteger(
        argcheck::expectedReal(args[1], 2, "n", "int2bit"), 2, "n", "int2bit");
    bool msb = true;
    if (args.size() >= 3) msb = (args[2].toScalar() != 0.0);
    outs[0] = int2bit(args[0], n, msb, ctx.engine->allocator());
}

void bi2de_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
               keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "bi2de");
    int base = 2;
    bool msb = false;  // default 'right-msb' = LSB-first
    for (keel::usize i = 1; i < args.size(); ++i) {
        if (args[i].isChar() || args[i].isString())
            msb = parseMsbFlag(args[i], "bi2de");
        else if (!args[i].isEmpty())  // empty [] base -> keep default 2
            base = argcheck::expectedInteger(
                argcheck::expectedReal(args[i], int(i) + 1, "base", "bi2de"),
                int(i) + 1, "base", "bi2de");
    }
    // Marshal-only: the base >= 2 domain check lives in the compute-side
    // bi2de_argcheck (crossed by direct C++ calls too).
    outs[0] = bi2de(args[0], base, msb, ctx.engine->allocator());
}

void de2bi_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
               keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 4, "de2bi");
    int n = -1;       // -1 = auto
    int base = 2;
    bool msb = false; // default 'right-msb'
    int numericFound = 0;
    for (keel::usize i = 1; i < args.size(); ++i) {
        if (args[i].isChar() || args[i].isString())
            msb = parseMsbFlag(args[i], "de2bi");
        else if (numericFound == 0) {
            // First positional numeric is n. An EMPTY [] means "auto width"
            // (MATLAB: de2bi(d, [], base)) — keep n = -1, don't toScalar([]).
            if (!args[i].isEmpty())
                n = argcheck::expectedPositiveInteger(
                    argcheck::expectedReal(args[i], int(i) + 1, "n", "de2bi"),
                    int(i) + 1, "n", "de2bi");
            ++numericFound;
        } else {
            if (!args[i].isEmpty())
                base = argcheck::expectedInteger(
                    argcheck::expectedReal(args[i], int(i) + 1, "base", "de2bi"),
                    int(i) + 1, "base", "de2bi");
            ++numericFound;
        }
    }
    outs[0] = de2bi(args[0], n, base, msb, ctx.engine->allocator());
}

void vec2mat_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "vec2mat");
    const int n = argcheck::expectedPositiveInteger(
        argcheck::expectedReal(args[1], 2, "n", "vec2mat"), 2, "n", "vec2mat");
    keel::f64 padval = 0.0;
    if (args.size() >= 3) padval = args[2].toScalar();
    auto [m, pad] = vec2mat(args[0], n, padval, ctx.engine->allocator());
    outs[0] = m;
    if (nargout >= 2 && outs.size() >= 2)
        outs[1] = Value::scalar(keel::as<keel::f64>(pad), ctx.engine->allocator());
}

} // namespace detail

} // namespace numkit::comm
