// toolboxes/comm/src/coding/convcoding.cpp
//
// Convolutional coding (Error Correction Codes): poly2trellis.
//
// MATLAB R2025b semantics (verified via probe):
//   trellis = poly2trellis(K, [g1 g2 ... gn])  — rate 1/n feed-forward.
//   poly2trellis(3, [6 7]) ->
//     numInputSymbols=2, numOutputSymbols=4, numStates=4,
//     nextStates=[0 2;0 2;1 3;1 3], outputs=[0 3;1 2;3 0;2 1].

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <numkit/comm/coding/convcoding.hpp>

#include <numkit/value/argcheck.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace numkit::comm {

namespace {

// Interpret a non-negative integer whose decimal digits are an octal
// numeral (MATLAB CodeGenerator convention): 171 -> octal 171 -> 121.
keel::u64 octalToDecimal(keel::u64 oct, const char *who)
{
    keel::u64 result = 0, mult = 1;
    while (oct > 0) {
        const keel::u64 digit = oct % 10;
        if (digit > 7)
            throw Error(std::string(who) + ": CodeGenerator entries must be "
                        "octal numbers (digits 0-7)",
                        0, 0, who, "", std::string("numkit:") + who + ":octal");
        result += digit * mult;
        mult *= 8;
        oct /= 10;
    }
    return result;
}

inline int bitParity(keel::u64 x)
{
    int p = 0;
    while (x) { p ^= 1; x &= (x - 1); }
    return p;
}

// Read the (rate-1/n) parameters out of a trellis struct. The returned
// pointers alias the struct's matrix fields, which stay valid for the
// duration of the caller (the trellis Value outlives the call).
struct TrellisView {
    keel::usize numStates        = 0;
    keel::usize numOutputSymbols = 0;
    int         n                = 0;   // output bits per step
    const double *nextStates     = nullptr;  // numStates × 2, column-major
    const double *outputs        = nullptr;
};

TrellisView readTrellis(const Value &t, const char *who)
{
    if (!t.isStruct() || t.numel() != 1)
        throw Error(std::string(who) + ": trellis must be a 1x1 struct",
                    0, 0, who, "", std::string("numkit:") + who + ":trellis");
    const auto &el = t.structArrayElem(0);
    auto field = [&](const char *f) -> const Value & {
        const Value *v = el.get(f);
        if (!v)
            throw Error(std::string(who) + ": trellis is missing field '" + f + "'",
                        0, 0, who, "", std::string("numkit:") + who + ":trellis");
        return *v;
    };
    TrellisView tv;
    tv.numStates        = static_cast<keel::usize>(field("numStates").toScalar());
    tv.numOutputSymbols = static_cast<keel::usize>(field("numOutputSymbols").toScalar());
    tv.n = static_cast<int>(std::lround(std::log2(
        static_cast<double>(tv.numOutputSymbols))));
    tv.nextStates = field("nextStates").doubleData();
    tv.outputs    = field("outputs").doubleData();
    return tv;
}

} // namespace

Value poly2trellis(const Value &constraintLength, const Value &codeGenerator,
                   keel::Allocator *mr)
{
    if (constraintLength.numel() != 1)
        throw Error("poly2trellis: only a scalar ConstraintLength (rate 1/n) "
                    "is supported in this revision (rate k/n deferred)",
                    0, 0, "poly2trellis", "", "numkit:poly2trellis:rate");
    const long long K = static_cast<long long>(constraintLength.toScalar());
    if (K < 1)
        throw Error("poly2trellis: ConstraintLength must be >= 1",
                    0, 0, "poly2trellis", "", "numkit:poly2trellis:K");

    const keel::usize n = codeGenerator.numel();
    if (n == 0)
        throw Error("poly2trellis: CodeGenerator must be non-empty",
                    0, 0, "poly2trellis", "", "numkit:poly2trellis:gen");

    // Parse octal generators into K-bit binary masks.
    std::vector<keel::u64> g(n);
    for (keel::usize i = 0; i < n; ++i)
        g[i] = octalToDecimal(
            static_cast<keel::u64>(codeGenerator.elemAsDouble(i)),
            "poly2trellis");

    const keel::usize numStates = static_cast<keel::usize>(1) << (K - 1);
    const keel::usize numInput  = 2;                                   // k = 1
    const keel::usize numOutput = static_cast<keel::usize>(1) << n;

    Value ns = Value::matrix(numStates, numInput, ValueType::DOUBLE, mr);
    Value ou = Value::matrix(numStates, numInput, ValueType::DOUBLE, mr);
    double *nsd = ns.doubleDataMut();
    double *oud = ou.doubleDataMut();

    const keel::u64 stateMask = numStates - 1;  // 0 when numStates==1
    for (keel::usize s = 0; s < numStates; ++s) {
        for (keel::usize u = 0; u < numInput; ++u) {
            // K-bit register: newest input at the MSB, then the (K-1) state
            // bits below it.
            const keel::u64 reg =
                (static_cast<keel::u64>(u) << (K - 1)) | s;
            const keel::u64 nextState = (reg >> 1) & stateMask;
            keel::u64 out = 0;
            for (keel::usize i = 0; i < n; ++i)            // g1 = MSB
                out = (out << 1) | static_cast<keel::u64>(bitParity(reg & g[i]));
            // Column-major (numStates × 2): element (s, u) at u*numStates + s.
            nsd[u * numStates + s] = static_cast<double>(nextState);
            oud[u * numStates + s] = static_cast<double>(out);
        }
    }

    Value t = Value::structArray(1, 1, mr);
    auto &el = t.structArrayElem(0);
    el.put(keel::String::from("numInputSymbols"),  Value::scalar(static_cast<double>(numInput),  mr));
    el.put(keel::String::from("numOutputSymbols"), Value::scalar(static_cast<double>(numOutput), mr));
    el.put(keel::String::from("numStates"),        Value::scalar(static_cast<double>(numStates), mr));
    el.put(keel::String::from("nextStates"), std::move(ns));
    el.put(keel::String::from("outputs"),    std::move(ou));
    return t;
}

Value convenc(const Value &msg, const Value &trellis,
              keel::Allocator *mr)
{
    const TrellisView tv = readTrellis(trellis, "convenc");
    const keel::usize L = msg.numel();
    const int n = tv.n;
    const keel::usize outLen = L * static_cast<keel::usize>(n);
    // Output takes msg's orientation (column input -> column output).
    const bool col = (msg.dims().cols() == 1 && msg.dims().rows() > 1);
    Value out = col ? Value::matrix(outLen, 1, ValueType::DOUBLE, mr)
                    : Value::matrix(1, outLen, ValueType::DOUBLE, mr);
    if (outLen == 0) return out;

    double *od = out.doubleDataMut();
    keel::usize state = 0, oi = 0;
    for (keel::usize i = 0; i < L; ++i) {
        const int bit = (msg.elemAsDouble(i) != 0.0) ? 1 : 0;
        // Column-major (numStates × 2): element (state, bit).
        const keel::usize idx = static_cast<keel::usize>(bit) * tv.numStates + state;
        const keel::u64 outWord = static_cast<keel::u64>(tv.outputs[idx]);
        for (int j = n - 1; j >= 0; --j)                 // MSB (g1) first
            od[oi++] = static_cast<double>((outWord >> j) & 1u);
        state = static_cast<keel::usize>(tv.nextStates[idx]);
    }
    return out;
}

namespace {

struct VitdecPlan {
    TrellisView tv;
    int n;
    keel::usize L;
    keel::usize S;
    std::string mode;   // normalized opmode ('trunc' | 'term')
};

// vitdec_argcheck: decision-mode and code-length contract, crossed by
// every call path (the reg marshals the opmode/dectype defaults only).
VitdecPlan vitdec_argcheck(const Value &code, const Value &trellis,
                           const std::string &opmode,
                           const std::string &dectype)
{
    auto lower = [](std::string s) {
        for (char &c : s)
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return s;
    };
    const std::string dt = lower(dectype);
    if (dt != "hard")
        throw Error("vitdec: only hard-decision decoding is supported in this "
                    "revision (soft/unquant deferred)",
                    0, 0, "vitdec", "", "numkit:vitdec:dectype");
    const std::string mode = lower(opmode);
    if (mode != "trunc" && mode != "term")
        throw Error("vitdec: opmode must be 'trunc' or 'term' in this revision "
                    "('cont' deferred)",
                    0, 0, "vitdec", "", "numkit:vitdec:opmode");

    const TrellisView tv = readTrellis(trellis, "vitdec");
    const int n = tv.n;
    const keel::usize total = code.numel();
    if (n <= 0 || (total % static_cast<keel::usize>(n)) != 0)
        throw Error("vitdec: code length must be a multiple of "
                    "log2(numOutputSymbols)",
                    0, 0, "vitdec", "", "numkit:vitdec:len");
    return {tv, n, total / static_cast<keel::usize>(n), tv.numStates, mode};
}

} // namespace

Value vitdec(const Value &code, const Value &trellis, long long /*tblen*/,
             const std::string &opmode, const std::string &dectype,
             keel::Allocator *mr)
{
    const VitdecPlan p = vitdec_argcheck(code, trellis, opmode, dectype);
    const TrellisView &tv = p.tv;
    const int n = p.n;
    const keel::usize L = p.L;
    const keel::usize S = p.S;
    const std::string &mode = p.mode;

    const bool col = (code.dims().cols() == 1 && code.dims().rows() > 1);
    Value out = col ? Value::matrix(L, 1, ValueType::DOUBLE, mr)
                    : Value::matrix(1, L, ValueType::DOUBLE, mr);
    if (L == 0) return out;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::u64> rsym(L, scratch.allocator());
    for (keel::usize t = 0; t < L; ++t) {
        keel::u64 w = 0;
        for (int j = 0; j < n; ++j)
            w = (w << 1) | (code.elemAsDouble(t * n + j) != 0.0 ? 1u : 0u);
        rsym[t] = w;
    }

    const double INF = std::numeric_limits<double>::infinity();
    keel::Array<double> metric(S, scratch.allocator());
    keel::Array<double> newMetric(S, scratch.allocator());
    for (keel::usize s = 0; s < S; ++s) metric[s] = INF;
    metric[0] = 0.0;
    keel::Array<int>          prevState(L * S, scratch.allocator());
    keel::Array<keel::u8> decBit(L * S, scratch.allocator());

    auto hamming = [](keel::u64 a, keel::u64 b) {
        keel::u64 x = a ^ b;
        int c = 0;
        while (x) { ++c; x &= (x - 1); }
        return c;
    };

    // Add-compare-select forward recursion.
    for (keel::usize t = 0; t < L; ++t) {
        for (keel::usize s = 0; s < S; ++s) newMetric[s] = INF;
        for (keel::usize s = 0; s < S; ++s) {
            if (metric[s] == INF) continue;
            for (keel::usize u = 0; u < 2; ++u) {
                const keel::usize idx = u * S + s;               // (state s, input u)
                const keel::usize ns  = static_cast<keel::usize>(tv.nextStates[idx]);
                const keel::u64 ow = static_cast<keel::u64>(tv.outputs[idx]);
                const double cand = metric[s] + hamming(rsym[t], ow);
                if (cand < newMetric[ns]) {
                    newMetric[ns] = cand;
                    prevState[t * S + ns] = static_cast<int>(s);
                    decBit[t * S + ns]    = static_cast<keel::u8>(u);
                }
            }
        }
        keel::swap(metric, newMetric);
    }

    // Pick the traceback start state: state 0 for 'term', else min metric.
    keel::usize endState = 0;
    if (mode != "term") {
        double best = metric[0];
        for (keel::usize s = 1; s < S; ++s)
            if (metric[s] < best) { best = metric[s]; endState = s; }
    }

    // Traceback (full).
    double *od = out.doubleDataMut();
    keel::usize st = endState;
    for (keel::usize ti = L; ti-- > 0;) {
        od[ti] = static_cast<double>(decBit[ti * S + st]);
        const int p = prevState[ti * S + st];
        st = (p >= 0) ? static_cast<keel::usize>(p) : 0;
    }
    return out;
}

Value istrellis(const Value &S, keel::Allocator *mr)
{
    auto no = [&]() { return Value::logicalScalar(false, mr); };
    auto isPow2 = [](double v) {
        if (v < 1.0 || v != std::floor(v)) return false;
        const keel::u64 u = static_cast<keel::u64>(v);
        return (u & (u - 1)) == 0;
    };
    if (!S.isStruct() || S.numel() != 1) return no();
    const auto &el = S.structArrayElem(0);
    const Value *nIvPtr = el.get("numInputSymbols");
    const Value *nOvPtr = el.get("numOutputSymbols");
    const Value *nSvPtr = el.get("numStates");
    const Value *nsPtr  = el.get("nextStates");
    const Value *ouPtr  = el.get("outputs");
    if (!nIvPtr || !nOvPtr || !nSvPtr || !nsPtr || !ouPtr) return no();

    const Value &nIv = *nIvPtr;
    const Value &nOv = *nOvPtr;
    const Value &nSv = *nSvPtr;
    const Value &ns  = *nsPtr;
    const Value &ou  = *ouPtr;
    if (!nIv.isScalar() || !nOv.isScalar() || !nSv.isScalar()) return no();
    const double nI = nIv.toScalar(), nO = nOv.toScalar(), nS = nSv.toScalar();
    if (!isPow2(nI) || !isPow2(nO) || !isPow2(nS)) return no();

    if (ns.isStruct() || ns.isCell() || ou.isStruct() || ou.isCell()) return no();
    const keel::usize Sn = static_cast<keel::usize>(nS);
    const keel::usize In = static_cast<keel::usize>(nI);
    if (ns.dims().rows() != Sn || ns.dims().cols() != In ||
        ou.dims().rows() != Sn || ou.dims().cols() != In)
        return no();

    const keel::usize N = Sn * In;
    for (keel::usize i = 0; i < N; ++i) {
        const double v = ns.elemAsDouble(i);
        if (v < 0.0 || v >= nS || v != std::floor(v)) return no();
        const double w = ou.elemAsDouble(i);
        if (w < 0.0 || w >= nO || w != std::floor(w)) return no();
    }
    return Value::logicalScalar(true, mr);
}

} // namespace numkit::comm
