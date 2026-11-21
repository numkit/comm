// toolboxes/comm/src/eq/pulse.cpp
//
// rcosdesign — raised-cosine and root-raised-cosine FIR design.
// Closed-form coefficients:
//   RC:   p(t) = sinc(t/T) · cos(πβt/T) / (1 − (2βt/T)²)
//   RRC:  h(t) = (4β / π√T) · [cos((1+β)πt/T)
//                  + sin((1−β)πt/T) / (4βt/T)]
//                / (1 − (4βt/T)²)
// with the standard l'Hôpital limits at t = 0 and t = ±T/(4β).
// Output normalised to unit energy, matching MATLAB R2025b.

#include <keel/array.h>
#include <keel/comm.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/math.h>
#include <keel/ref.h>
#include <numkit/comm/eq/pulse.hpp>

#include <numkit/value/argcheck.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace numkit::comm {

namespace {

void rcosdesign_argcheck(keel::f64 beta, keel::i32 span, keel::i32 sps, keel::StringRef shape)
{
    if (!(beta >= 0.0 && beta <= 1.0))
        throw Error("rcosdesign: beta must be in [0, 1]",
                    0, 0, "rcosdesign", "", "numkit:rcosdesign:beta");
    if (span <= 0)
        throw Error("rcosdesign: span must be a positive integer",
                    0, 0, "rcosdesign", "", "numkit:rcosdesign:span");
    if (sps <= 0)
        throw Error("rcosdesign: sps must be a positive integer",
                    0, 0, "rcosdesign", "", "numkit:rcosdesign:sps");

    const bool rrc = (shape == "sqrt" || shape == "Sqrt" || shape == "SQRT");
    if (!rrc && shape != "normal" && shape != "Normal" &&
        shape != "NORMAL" && !shape.empty())
        throw Error("rcosdesign: shape must be 'normal' or 'sqrt'",
                    0, 0, "rcosdesign", "", "numkit:rcosdesign:shape");
}

void gaussdesign_argcheck(keel::f64 BT, keel::i32 span, keel::i32 sps)
{
    if (!(BT > 0.0))
        throw Error("gaussdesign: BT must be positive",
                    0, 0, "gaussdesign", "", "numkit:gaussdesign:BT");
    if (span <= 0)
        throw Error("gaussdesign: span must be a positive integer",
                    0, 0, "gaussdesign", "", "numkit:gaussdesign:span");
    if (sps <= 0)
        throw Error("gaussdesign: sps must be a positive integer",
                    0, 0, "gaussdesign", "", "numkit:gaussdesign:sps");
}

void rectpulse_argcheck(keel::i32 n)
{
    if (n <= 0)
        throw Error("rectpulse: n must be a positive integer",
                    0, 0, "rectpulse", "", "numkit:rectpulse:n");
}

void intdump_argcheck(const Value &x, keel::i32 n)
{
    if (n <= 0)
        throw Error("intdump: n must be a positive integer",
                    0, 0, "intdump", "", "numkit:intdump:n");
    const auto &d = x.dims();
    const keel::usize H = d.rows();
    const keel::usize W = d.cols();
    const bool is_row = (H == 1 && W >= 1);
    const keel::usize un = keel::as<keel::usize>(n);
    if (is_row) {
        if (W % un != 0)
            throw Error("intdump: row length must be a multiple of n",
                        0, 0, "intdump", "", "numkit:intdump:size");
    } else {
        if (H % un != 0)
            throw Error("intdump: column/row count must be a multiple of n",
                        0, 0, "intdump", "", "numkit:intdump:size");
    }
}

} // anonymous

Value rcosdesign(keel::f64 beta, keel::i32 span, keel::i32 sps,
                 keel::StringRef shape,
                 keel::Allocator *mr)
{
    rcosdesign_argcheck(beta, span, sps, shape);
    const bool rrc = (shape == "sqrt" || shape == "Sqrt" || shape == "SQRT");
    const keel::usize N = keel::as<keel::usize>(span) * keel::as<keel::usize>(sps) + 1;
    Value r = Value::matrix(1, N, ValueType::DOUBLE, mr);
    (void)keel::comm::rcos_design(beta, keel::as<keel::usize>(span),
                                  keel::as<keel::usize>(sps),
                                  rrc ? keel::comm::RcosShape::kSqrt : keel::comm::RcosShape::kNormal,
                                  {r.doubleDataMut(), N});
    return r;
}

Value gaussdesign(keel::f64 BT, keel::i32 span, keel::i32 sps,
                  keel::Allocator *mr)
{
    gaussdesign_argcheck(BT, span, sps);
    const keel::usize N = keel::as<keel::usize>(span) * keel::as<keel::usize>(sps) + 1;
    Value r = Value::matrix(1, N, ValueType::DOUBLE, mr);
    (void)keel::comm::gauss_design(BT, keel::as<keel::usize>(span),
                                   keel::as<keel::usize>(sps),
                                   {r.doubleDataMut(), N});
    return r;
}

// ── rectpulse ───────────────────────────────────────────────────────
// rectpulse(x, n) repeats each sample of x n times along the leading
// non-singleton dimension. Vector inputs preserve orientation; matrix
// inputs repeat each row n times (column count unchanged).
Value rectpulse(const Value &x, keel::i32 n, keel::Allocator *mr)
{
    rectpulse_argcheck(n);

    const auto &d = x.dims();
    const keel::usize H = d.rows();
    const keel::usize W = d.cols();
    const bool is_row = (H == 1 && W >= 1);
    const bool is_col = (W == 1 && H >= 1);

    Value out;
    if (is_row) {
        out = Value::matrix(1, W * keel::as<keel::usize>(n), ValueType::DOUBLE, mr);
        keel::f64 *od = out.doubleDataMut();
        for (keel::usize c = 0; c < W; ++c) {
            const keel::f64 v = x.elemAsDouble(c);
            for (keel::i32 k = 0; k < n; ++k)
                od[c * keel::as<keel::usize>(n) + keel::as<keel::usize>(k)] = v;
        }
    } else if (is_col) {
        out = Value::matrix(H * keel::as<keel::usize>(n), 1, ValueType::DOUBLE, mr);
        keel::f64 *od = out.doubleDataMut();
        for (keel::usize r = 0; r < H; ++r) {
            const keel::f64 v = x.elemAsDouble(r);
            for (keel::i32 k = 0; k < n; ++k)
                od[r * keel::as<keel::usize>(n) + keel::as<keel::usize>(k)] = v;
        }
    } else {
        const keel::usize Nh = H * keel::as<keel::usize>(n);
        out = Value::matrix(Nh, W, ValueType::DOUBLE, mr);
        keel::f64 *od = out.doubleDataMut();
        for (keel::usize c = 0; c < W; ++c) {
            for (keel::usize r = 0; r < H; ++r) {
                const keel::f64 v = x.elemAsDouble(c * H + r);
                for (keel::i32 k = 0; k < n; ++k)
                    od[c * Nh + r * keel::as<keel::usize>(n) + keel::as<keel::usize>(k)] = v;
            }
        }
    }
    return out;
}

// ── intdump ─────────────────────────────────────────────────────────
// Inverse of rectpulse: average each n consecutive samples along the
// leading non-singleton dimension. Length along that axis must be
// divisible by n.
Value intdump(const Value &x, keel::i32 n, keel::Allocator *mr)
{
    intdump_argcheck(x, n);

    const auto &d = x.dims();
    const keel::usize H = d.rows();
    const keel::usize W = d.cols();
    const bool is_row = (H == 1 && W >= 1);
    const bool is_col = (W == 1 && H >= 1);
    const keel::usize un = keel::as<keel::usize>(n);

    Value out;
    if (is_row) {
        if (W % un != 0)
            throw Error("intdump: row length must be a multiple of n",
                        0, 0, "intdump", "", "numkit:intdump:size");
        const keel::usize Wo = W / un;
        out = Value::matrix(1, Wo, ValueType::DOUBLE, mr);
        keel::f64 *od = out.doubleDataMut();
        for (keel::usize c = 0; c < Wo; ++c) {
            keel::f64 s = 0.0;
            for (keel::usize k = 0; k < un; ++k)
                s += x.elemAsDouble(c * un + k);
            od[c] = s / keel::as<keel::f64>(un);
        }
    } else if (is_col) {
        if (H % un != 0)
            throw Error("intdump: column length must be a multiple of n",
                        0, 0, "intdump", "", "numkit:intdump:size");
        const keel::usize Ho = H / un;
        out = Value::matrix(Ho, 1, ValueType::DOUBLE, mr);
        keel::f64 *od = out.doubleDataMut();
        for (keel::usize r = 0; r < Ho; ++r) {
            keel::f64 s = 0.0;
            for (keel::usize k = 0; k < un; ++k)
                s += x.elemAsDouble(r * un + k);
            od[r] = s / keel::as<keel::f64>(un);
        }
    } else {
        // Matrix: average n consecutive rows per column.
        if (H % un != 0)
            throw Error("intdump: row count must be a multiple of n",
                        0, 0, "intdump", "", "numkit:intdump:size");
        const keel::usize Ho = H / un;
        out = Value::matrix(Ho, W, ValueType::DOUBLE, mr);
        keel::f64 *od = out.doubleDataMut();
        for (keel::usize c = 0; c < W; ++c) {
            for (keel::usize r = 0; r < Ho; ++r) {
                keel::f64 s = 0.0;
                for (keel::usize k = 0; k < un; ++k)
                    s += x.elemAsDouble(c * H + r * un + k);
                od[c * Ho + r] = s / keel::as<keel::f64>(un);
            }
        }
    }
    return out;
}

} // namespace numkit::comm
