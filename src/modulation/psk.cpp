// toolboxes/comm/src/modulation/psk.cpp

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/complex.h>
#include <numkit/comm/modulation/psk.hpp>

#include <numkit/value/argcheck.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace numkit::comm {

namespace {

// Gray code encoding: g = b ⊕ (b >> 1).
inline int to_gray(int b) { return b ^ (b >> 1); }

// Gray → binary.
inline int from_gray(int g) {
    int b = g;
    while (g >>= 1) b ^= g;
    return b;
}

// Map an integer symbol s ∈ 0..M-1 to a phase index in 0..M-1 according
// to the chosen ordering. For "gray" the natural index k is replaced
// by gray⁻¹(k); for "bin" (binary), it's identity.
inline int symbol_to_phase_idx(int s, int M, const std::string &order) {
    if (order == "bin") return s;
    // Default: "gray". MATLAB convention: phase 2π·k/M corresponds to
    // symbol s where Gray(s) = k. Inverse: s = Gray⁻¹(k) → k = Gray(s).
    return to_gray(s) % M;
}

inline int phase_idx_to_symbol(int k, int M, const std::string &order) {
    if (order == "bin") return k;
    return from_gray(k) % M;
}

Value alloc_complex_like(keel::Allocator *mr, const Value &x) {
    const auto &d = x.dims();
    if (x.isScalar()) return Value::matrix(1, 1, ValueType::COMPLEX, mr);
    if (d.is3D())
        return Value::matrix3d(d.rows(), d.cols(), d.pages(), ValueType::COMPLEX, mr);
    return Value::matrix(d.rows(), d.cols(), ValueType::COMPLEX, mr);
}

Value alloc_double_like(keel::Allocator *mr, const Value &x) {
    const auto &d = x.dims();
    if (x.isScalar()) return Value::scalar(0.0, mr);
    if (d.is3D())
        return Value::matrix3d(d.rows(), d.cols(), d.pages(), ValueType::DOUBLE, mr);
    return Value::matrix(d.rows(), d.cols(), ValueType::DOUBLE, mr);
}

void pskmod_argcheck(int M) {
    if (M < 2)
        throw Error("pskmod: M must be ≥ 2", 0, 0, "pskmod", "",
                    "numkit:pskmod:badM");
}

void pskdemod_argcheck(int M) {
    if (M < 2)
        throw Error("pskdemod: M must be ≥ 2", 0, 0, "pskdemod", "",
                    "numkit:pskdemod:badM");
}

void dpskmod_argcheck(int M) {
    if (M < 2)
        throw Error("dpskmod: M must be ≥ 2", 0, 0, "dpskmod", "",
                    "numkit:dpskmod:badM");
}

void dpskdemod_argcheck(int M) {
    if (M < 2)
        throw Error("dpskdemod: M must be ≥ 2", 0, 0, "dpskdemod", "",
                    "numkit:dpskdemod:badM");
}

} // anonymous

Value pskmod(const Value &x, int M, double ini_phase,
             const std::string &symbol_order,
             keel::Allocator *mr)
{
    pskmod_argcheck(M);
    Value out = alloc_complex_like(mr, x);
    const keel::usize N = x.numel();
    if (N == 0) return out;
    keel::c64 *od = out.complexDataMut();
    const double dphi = 2.0 * M_PI / double(M);
    for (keel::usize i = 0; i < N; ++i) {
        const int s = (int)x.elemAsDouble(i);
        const int k = symbol_to_phase_idx(s, M, symbol_order);
        const double theta = dphi * k + ini_phase;
        od[i] = keel::c64(std::cos(theta), std::sin(theta));
    }
    return out;
}

Value pskdemod(const Value &y, int M, double ini_phase,
               const std::string &symbol_order,
               keel::Allocator *mr)
{
    pskdemod_argcheck(M);
    Value out = alloc_double_like(mr, y);
    const keel::usize N = y.numel();
    if (N == 0) return out;
    double *od = out.doubleDataMut();
    const double dphi = 2.0 * M_PI / double(M);
    for (keel::usize i = 0; i < N; ++i) {
        keel::c64 c;
        if (y.type() == ValueType::COMPLEX) c = y.complexData()[i];
        else c = keel::c64(y.elemAsDouble(i), 0.0);
        // Subtract initial phase, find nearest constellation point.
        double angle = keel::arg(c) - ini_phase;
        int k = (int)std::lround(angle / dphi);
        // Wrap to 0..M-1.
        k = ((k % M) + M) % M;
        od[i] = double(phase_idx_to_symbol(k, M, symbol_order));
    }
    return out;
}

Value dpskmod(const Value &x, int M, double phase_rot,
              const std::string &symbol_order,
              keel::Allocator *mr)
{
    dpskmod_argcheck(M);
    Value out = alloc_complex_like(mr, x);
    const keel::usize N = x.numel();
    if (N == 0) return out;
    keel::c64 *od = out.complexDataMut();
    const double dphi = 2.0 * M_PI / double(M);
    double phase = phase_rot;  // initial reference
    for (keel::usize i = 0; i < N; ++i) {
        const int s = (int)x.elemAsDouble(i);
        const int k = symbol_to_phase_idx(s, M, symbol_order);
        phase += dphi * k;
        od[i] = keel::c64(std::cos(phase), std::sin(phase));
    }
    return out;
}

Value dpskdemod(const Value &y, int M, double phase_rot,
                const std::string &symbol_order,
                keel::Allocator *mr)
{
    dpskdemod_argcheck(M);
    Value out = alloc_double_like(mr, y);
    const keel::usize N = y.numel();
    if (N == 0) return out;
    double *od = out.doubleDataMut();
    const double dphi = 2.0 * M_PI / double(M);
    double prev_phase = phase_rot;
    for (keel::usize i = 0; i < N; ++i) {
        keel::c64 c;
        if (y.type() == ValueType::COMPLEX) c = y.complexData()[i];
        else c = keel::c64(y.elemAsDouble(i), 0.0);
        const double cur = keel::arg(c);
        double diff = cur - prev_phase;
        int k = (int)std::lround(diff / dphi);
        k = ((k % M) + M) % M;
        od[i] = double(phase_idx_to_symbol(k, M, symbol_order));
        prev_phase = cur;
    }
    return out;
}

} // namespace numkit::comm
