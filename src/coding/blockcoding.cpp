// toolboxes/comm/src/coding/blockcoding.cpp
//
// Block linear coding (Error Correction Codes): gen2par, hammgen,
// cyclpoly, cyclgen, encode, decode. All GF(2) (binary) arithmetic on
// plain double matrices.
//
// MATLAB R2025b semantics (verified via probe + toolbox-doc algorithm):
//   hammgen(m): H(:,i) = coeffs of x^i mod p(x), p = default primitive
//     polynomial of degree m, ascending power. First m columns = I_m, so
//     the code is systematic and g = gen2par(h).
//   gen2par(g): systematic generator<->parity converter (involution).
//   cyclpoly(n,k): first degree-(n-k) poly 1+...+x^(n-k) dividing x^n-1.
//   cyclgen(n,p): cyclic parity/generator matrices (system / nonsystem).
//   encode/decode: reshape into words, generator-multiply / syndrome-decode.

#include <keel/array.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <numkit/comm/coding/blockcoding.hpp>

#include <numkit/value/argcheck.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>

namespace numkit::comm {

namespace {

// ── GF(2) primitives ───────────────────────────────────────────────────

// Default primitive polynomial of degree m (m = 2..16) as an integer bit
// mask: bit j set ⇔ coefficient of x^j is 1. Matches MATLAB gfprimdf.
// Bit 0 (constant) and bit m (leading) are always set.
keel::u64 defaultPrimMask(long long m, const char *who)
{
    switch (m) {
    case 2:  return 0x7ULL;        // 1+x+x^2
    case 3:  return 0xBULL;        // 1+x+x^3
    case 4:  return 0x13ULL;       // 1+x+x^4
    case 5:  return 0x25ULL;       // 1+x^2+x^5
    case 6:  return 0x43ULL;       // 1+x+x^6
    case 7:  return 0x89ULL;       // 1+x^3+x^7
    case 8:  return 0x11DULL;      // 1+x^2+x^3+x^4+x^8
    case 9:  return 0x211ULL;      // 1+x^4+x^9
    case 10: return 0x409ULL;      // 1+x^3+x^10
    case 11: return 0x805ULL;      // 1+x^2+x^11
    case 12: return 0x1053ULL;     // 1+x+x^4+x^6+x^12
    case 13: return 0x201BULL;     // 1+x+x^3+x^4+x^13
    case 14: return 0x4443ULL;     // 1+x+x^6+x^10+x^14
    case 15: return 0x8003ULL;     // 1+x+x^15
    case 16: return 0x1100BULL;    // 1+x+x^3+x^12+x^16
    default:
        throw Error(keel::format("{}: M out of supported range [2,16]", who),
                    0, 0, who, "", keel::format("numkit:{}:range", who));
    }
}

// Read element (i,j) of a numeric Value (column-major) as 0/1.
inline int matBit(const Value &v, keel::usize i, keel::usize j)
{
    const keel::usize rows = v.dims().rows();
    return v.elemAsDouble(j * rows + i) != 0.0 ? 1 : 0;
}

// ── GF(2) polynomial arithmetic (coefficients ascending: index = power) ─

// Highest set power of a, or -1 for the zero polynomial.
inline int polyDeg(keel::ArrayRef<int> a)
{
    for (int i = static_cast<int>(a.size()) - 1; i >= 0; --i)
        if (a[i] & 1) return i;
    return -1;
}

// Remainder of a mod b over GF(2). b must be nonzero.
keel::Array<int> polyMod(keel::ArrayRef<int> aIn, keel::ArrayRef<int> b)
{
    keel::Array<int> a(aIn);
    const int db = polyDeg(b);
    for (auto &x : a) x &= 1;
    for (int da = polyDeg(a); da >= db && da >= 0; da = polyDeg(a))
        for (int j = 0; j <= db; ++j) a[j + (da - db)] ^= b[j];
    a.resize(db > 0 ? static_cast<keel::usize>(db) : 0);  // degree < db
    return a;
}

// Quotient + remainder of a / b over GF(2).
struct PolyQR { keel::Array<int> q, r; };
PolyQR polyDivMod(keel::ArrayRef<int> aIn, keel::ArrayRef<int> b)
{
    keel::Array<int> a(aIn);
    const int db = polyDeg(b);
    for (auto &x : a) x &= 1;
    const int da0 = polyDeg(a);
    keel::Array<int> q(da0 >= db ? static_cast<keel::usize>(da0 - db + 1) : 1, 0);
    for (int da = da0; da >= db && da >= 0; da = polyDeg(a)) {
        q[da - db] ^= 1;
        for (int j = 0; j <= db; ++j) a[j + (da - db)] ^= b[j];
    }
    a.resize(db > 0 ? static_cast<keel::usize>(db) : 1);
    return {keel::move(q), keel::move(a)};
}

// Read a numeric Value (row or column vector) as a 0/1 coefficient list.
keel::Array<int> readPolyRow(const Value &v)
{
    const keel::usize nel = v.numel();
    keel::Array<int> p(nel);
    for (keel::usize i = 0; i < nel; ++i) p[i] = v.elemAsDouble(i) != 0.0 ? 1 : 0;
    return p;
}

inline keel::String toLower(keel::StringRef sv)
{
    keel::String s(sv);
    for (auto &c : s) c = keel::to_ascii_lower(c);
    return s;
}

// Build a row vector Value from a 0/1 coefficient list.
Value polyToRow(keel::ArrayRef<int> t, keel::Allocator *mr)
{
    Value v = Value::matrix(1, t.size(), ValueType::DOUBLE, mr);
    double *d = v.doubleDataMut();
    for (keel::usize j = 0; j < t.size(); ++j) d[j] = t[j];
    return v;
}

// Stack `rows` cyclic right-shifts of `row` into a rows×n DOUBLE matrix
// (each successive row is the previous rotated one position right).
Value shiftMatrix(keel::ArrayRef<int> rowIn, long long rows, long long n,
                  keel::Allocator *mr)
{
    keel::Array<int> row(rowIn);
    Value out = Value::matrix(static_cast<keel::usize>(rows),
                              static_cast<keel::usize>(n), ValueType::DOUBLE, mr);
    double *d = out.doubleDataMut();
    for (long long r = 0; r < rows; ++r) {
        for (long long c = 0; c < n; ++c) d[c * rows + r] = row[c];
        const int last = row[n - 1];
        for (long long c = n - 1; c > 0; --c) row[c] = row[c - 1];
        row[0] = last;
    }
    return out;
}

// ── gen2par core (shared by gen2par + hammgen) ─────────────────────────

// Convert an r×c binary matrix (r < c) between [P|I_r]/[I_r|P] systematic
// form and its complement. Returns a fresh (c-r)×c DOUBLE Value.
Value gen2parImpl(const Value &mat, const char *who,
                  keel::Allocator *mr)
{
    const keel::usize r = mat.dims().rows();
    const keel::usize c = mat.dims().cols();
    if (r == 0 || c == 0 || r >= c)
        throw Error(keel::format("{}: input must be r×c with r < c", who),
                    0, 0, who, "", keel::format("numkit:{}:shape", who));
    const keel::usize pcols = c - r;  // parity columns / output rows

    // Branch A: last r columns form I_r  ⇒ mat = [P | I_r]
    bool lastIsI = true;
    for (keel::usize i = 0; i < r && lastIsI; ++i)
        for (keel::usize t = 0; t < r; ++t)
            if (matBit(mat, i, pcols + t) != (i == t ? 1 : 0)) { lastIsI = false; break; }

    // Branch B: first r columns form I_r ⇒ mat = [I_r | P]
    bool firstIsI = true;
    for (keel::usize i = 0; i < r && firstIsI; ++i)
        for (keel::usize t = 0; t < r; ++t)
            if (matBit(mat, i, t) != (i == t ? 1 : 0)) { firstIsI = false; break; }

    Value out = Value::matrix(pcols, c, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();
    auto put = [&](keel::usize i, keel::usize j, int b) { od[j * pcols + i] = b; };

    if (lastIsI) {
        // out = [I_(c-r) | Pᵀ], P = mat(:, 0..pcols-1)
        for (keel::usize i = 0; i < pcols; ++i) {
            for (keel::usize j = 0; j < pcols; ++j) put(i, j, i == j ? 1 : 0);
            for (keel::usize t = 0; t < r; ++t) put(i, pcols + t, matBit(mat, t, i));
        }
    } else if (firstIsI) {
        // out = [Pᵀ | I_(c-r)], P = mat(:, r..c-1)
        for (keel::usize i = 0; i < pcols; ++i) {
            for (keel::usize t = 0; t < r; ++t) put(i, t, matBit(mat, t, r + i));
            for (keel::usize j = 0; j < pcols; ++j) put(i, r + j, i == j ? 1 : 0);
        }
    } else {
        throw Error(keel::format("{}: neither the first nor the last r "
                    "columns form an identity matrix (input not systematic)", who),
                    0, 0, who, "", keel::format("numkit:{}:notsystematic", who));
    }
    return out;
}

} // namespace

// ── gen2par ─────────────────────────────────────────────────────────────

Value gen2par(const Value &mat, keel::Allocator *mr)
{
    return gen2parImpl(mat, "gen2par", mr);
}

// ── hammgen ─────────────────────────────────────────────────────────────

HammgenResult hammgen(long long m, const Value &primPoly,
                      keel::Allocator *mr)
{
    if (m < 2)
        throw Error("hammgen: M must be an integer >= 2",
                    0, 0, "hammgen", "", "numkit:hammgen:M");

    // Primitive polynomial as an integer mask (bit j = coeff of x^j).
    keel::u64 pmask;
    if (primPoly.isEmpty()) {
        pmask = defaultPrimMask(m, "hammgen");
    } else {
        const keel::usize len = primPoly.numel();
        if (static_cast<long long>(len) != m + 1)
            throw Error("hammgen: P must be a binary row of length M+1",
                        0, 0, "hammgen", "", "numkit:hammgen:P");
        pmask = 0;
        for (keel::usize j = 0; j < len; ++j)
            if (primPoly.elemAsDouble(j) != 0.0) pmask |= (1ULL << j);
        if ((pmask & 1ULL) == 0 || (pmask & (1ULL << m)) == 0)
            throw Error("hammgen: P must have nonzero constant and degree-M terms",
                        0, 0, "hammgen", "", "numkit:hammgen:P");
    }

    const long long n = (1LL << m) - 1;
    const long long k = n - m;

    // H(j,i) = bit j of (x^i mod p), i = 0..n-1, j = 0..m-1.
    Value h = Value::matrix(static_cast<keel::usize>(m),
                            static_cast<keel::usize>(n), ValueType::DOUBLE, mr);
    double *hd = h.doubleDataMut();
    const keel::u64 topBit = 1ULL << m;
    keel::u64 r = 1;  // x^0
    for (long long i = 0; i < n; ++i) {
        for (long long j = 0; j < m; ++j)
            hd[i * m + j] = (r >> j) & 1ULL;     // column-major, row j
        r <<= 1;                                  // multiply by x
        if (r & topBit) r ^= pmask;               // reduce mod p
    }

    HammgenResult res;
    res.h = std::move(h);
    res.g = gen2parImpl(res.h, "hammgen", mr);    // first m cols are I_m
    res.n = n;
    res.k = k;
    return res;
}

// ── cyclpoly ────────────────────────────────────────────────────────────

Value cyclpoly(long long n, long long k, keel::StringRef opt,
               keel::Allocator *mr)
{
    if (n < 1)
        throw Error("cyclpoly: N must be a positive integer",
                    0, 0, "cyclpoly", "", "numkit:cyclpoly:N");
    if (k < 1 || k > n)
        throw Error("cyclpoly: K must satisfy 1 <= K <= N",
                    0, 0, "cyclpoly", "", "numkit:cyclpoly:K");

    const long long m = n - k;
    if (m == 0) return polyToRow({1}, mr);
    if (m == 1) return polyToRow({1, 1}, mr);

    // x^n + 1 over GF(2).
    keel::Array<int> pp(static_cast<keel::usize>(n) + 1, 0);
    pp[0] = 1; pp[static_cast<keel::usize>(n)] = 1;
    const long long nn = (1LL << (m - 1)) - 1;

    // Candidate degree-m polynomial 1 + (m-1 inner bits of i) + x^m.
    auto candidate = [&](long long i, bool msbFirst) {
        keel::Array<int> t(static_cast<keel::usize>(m) + 1, 0);
        t[0] = 1; t[static_cast<keel::usize>(m)] = 1;
        for (long long p = 0; p < m - 1; ++p)
            t[static_cast<keel::usize>(1 + p)] =
                msbFirst ? static_cast<int>((i >> (m - 2 - p)) & 1)
                         : static_cast<int>((i >> p) & 1);
        return t;
    };
    auto divides = [&](keel::ArrayRef<int> t) {
        return polyDeg(polyMod(pp, t)) < 0;  // remainder is zero
    };
    auto terms = [](keel::ArrayRef<int> t) {
        int s = 0; for (int x : t) s += x; return s;
    };

    const keel::String o = toLower(opt);
    if (o.empty()) {                                   // first found
        for (long long i = 0; i <= nn; ++i) {
            auto t = candidate(i, true);
            if (divides(t)) return polyToRow(t, mr);
        }
    } else if (o.starts_with("mi")) {                  // fewest terms
        for (long long j = 2; j <= m + 1; ++j)
            for (long long i = 0; i <= nn; ++i) {
                auto t = candidate(i, true);
                if (terms(t) == j && divides(t)) return polyToRow(t, mr);
            }
    } else if (o.starts_with("ma")) {                  // most terms
        for (long long j = m + 1; j >= 2; --j)
            for (long long i = 0; i <= nn; ++i) {
                auto t = candidate(i, false);          // LSB-first, per MATLAB
                if (terms(t) == j && divides(t)) return polyToRow(t, mr);
            }
    } else {                                           // "all" / exhaustive
        keel::Array<keel::Array<int>> rows;
        for (long long i = 0; i <= nn; ++i) {
            auto t = candidate(i, true);
            if (divides(t)) rows.push_back(keel::move(t));
        }
        std::stable_sort(rows.begin(), rows.end(),
            [&](const keel::Array<int> &a, const keel::Array<int> &b) {
                return terms(a) < terms(b);
            });
        if (rows.empty()) return Value::matrix(0, 0, ValueType::DOUBLE, mr);
        Value out = Value::matrix(rows.size(), static_cast<keel::usize>(m) + 1,
                                  ValueType::DOUBLE, mr);
        double *d = out.doubleDataMut();
        const keel::usize R = rows.size();
        for (keel::usize r = 0; r < R; ++r)
            for (keel::usize c = 0; c < rows[r].size(); ++c)
                d[c * R + r] = rows[r][c];
        return out;
    }
    return Value::matrix(0, 0, ValueType::DOUBLE, mr);  // none found
}

// ── cyclgen ─────────────────────────────────────────────────────────────

CyclgenResult cyclgen(long long n, const Value &genpoly,
                      keel::StringRef opt, keel::Allocator *mr)
{
    keel::Array<int> p = readPolyRow(genpoly);
    const int m = polyDeg(p);
    if (m < 1)
        throw Error("cyclgen: generator polynomial must have degree >= 1",
                    0, 0, "cyclgen", "", "numkit:cyclgen:P");
    const long long k = n - m;
    if (k < 1)
        throw Error("cyclgen: N must exceed the generator polynomial degree",
                    0, 0, "cyclgen", "", "numkit:cyclgen:N");

    // p must divide x^n + 1.
    keel::Array<int> pp(static_cast<keel::usize>(n) + 1, 0);
    pp[0] = 1; pp[static_cast<keel::usize>(n)] = 1;
    if (polyDeg(polyMod(pp, p)) >= 0)
        throw Error("cyclgen: generator polynomial must divide x^N - 1",
                    0, 0, "cyclgen", "", "numkit:cyclgen:P2");

    CyclgenResult res;
    res.k = k;
    const bool nonsys = toLower(opt).contains("no");

    if (nonsys) {
        // Parity poly h(x) = (x^n + 1) / p(x), degree k.
        keel::Array<int> pc = polyDivMod(pp, p).q;
        pc.resize(static_cast<keel::usize>(k) + 1, 0);
        keel::Array<int> hrow(static_cast<keel::usize>(n), 0);
        for (long long j = 0; j <= k; ++j) hrow[j] = pc[k - j];   // fliplr(pc)
        res.h = shiftMatrix(hrow, n - k, n, mr);
        keel::Array<int> grow(static_cast<keel::usize>(n), 0);
        for (keel::usize j = 0; j < p.size(); ++j) grow[j] = p[j];
        res.g = shiftMatrix(grow, k, n, mr);
    } else {
        // Systematic: b(i,:) = x^(m+i) mod p (k×m); h=[I_m | b'], g=[b | I_k].
        keel::Array<keel::Array<int>> b(static_cast<keel::usize>(k));
        for (long long i = 0; i < k; ++i) {
            keel::Array<int> xpow(static_cast<keel::usize>(m + i) + 1, 0);
            xpow[static_cast<keel::usize>(m + i)] = 1;
            auto rem = polyMod(xpow, p);
            rem.resize(static_cast<keel::usize>(m), 0);
            b[static_cast<keel::usize>(i)] = keel::move(rem);
        }
        const keel::usize M = static_cast<keel::usize>(m);
        const keel::usize K = static_cast<keel::usize>(k);
        const keel::usize N = static_cast<keel::usize>(n);
        res.h = Value::matrix(M, N, ValueType::DOUBLE, mr);
        double *hd = res.h.doubleDataMut();
        for (keel::usize r = 0; r < M; ++r) {
            for (keel::usize c = 0; c < M; ++c) hd[c * M + r] = (r == c) ? 1 : 0;
            for (keel::usize t = 0; t < K; ++t) hd[(M + t) * M + r] = b[t][r];
        }
        res.g = Value::matrix(K, N, ValueType::DOUBLE, mr);
        double *gd = res.g.doubleDataMut();
        for (keel::usize r = 0; r < K; ++r) {
            for (keel::usize c = 0; c < M; ++c) gd[c * K + r] = b[r][c];
            for (keel::usize t = 0; t < K; ++t) gd[(M + t) * K + r] = (r == t) ? 1 : 0;
        }
    }
    return res;
}

// ── encode / decode shared helpers ──────────────────────────────────────

namespace {

// Convert a numeric Value matrix to a row-major 0/1 matrix.
keel::Array<keel::Array<int>> matToRows(const Value &v)
{
    const keel::usize R = v.dims().rows(), C = v.dims().cols();
    keel::Array<keel::Array<int>> rows;
    rows.reserve(R);
    for (keel::usize i = 0; i < R; ++i) {
        keel::Array<int> row(C);
        for (keel::usize j = 0; j < C; ++j) row[j] = matBit(v, i, j);
        rows.push_back(keel::move(row));
    }
    return rows;
}

// Split "hamming/decimal" into base="hamming" + decimal=true.
void parseMethod(keel::StringRef method, keel::String &base, bool &decimal,
                 const char *who)
{
    const keel::String m = toLower(method);
    decimal = m.contains("decimal");
    if (m.starts_with("hamming"))      base.assign("hamming");
    else if (m.starts_with("linear"))  base.assign("linear");
    else if (m.starts_with("cyclic"))  base.assign("cyclic");
    else throw Error(keel::format("{}: unknown coding method '{}'", who, method),
                     0, 0, who, "", keel::format("numkit:{}:method", who));
}

// Build the k×n generator (gen) and (n-k)×n parity-check (h) matrices for
// the requested method, as row-major 0/1 matrices.
void buildCode(keel::StringRef base, long long n, long long k,
               const Value &opt, const char *who,
               keel::Array<keel::Array<int>> &gen,
               keel::Array<keel::Array<int>> &h,
               keel::Allocator *mr)
{
    if (base == "hamming") {
        const long long m = n - k;
        if (((1LL << m) - 1) != n)
            throw Error(keel::format("{}: N and K do not form a Hamming "
                        "code (2^(N-K)-1 must equal N)", who),
                        0, 0, who, "", keel::format("numkit:{}:nklen", who));
        HammgenResult hr = hammgen(m, opt.isEmpty() ? Value::Empty : opt, mr);
        h = matToRows(hr.h);
        gen = matToRows(hr.g);
    } else if (base == "linear") {
        if (opt.isEmpty())
            throw Error(keel::format("{}: the linear method requires a "
                        "generator matrix", who),
                        0, 0, who, "", keel::format("numkit:{}:genmat", who));
        if (static_cast<long long>(opt.dims().rows()) != k ||
            static_cast<long long>(opt.dims().cols()) != n)
            throw Error(keel::format("{}: generator matrix must be K x N", who),
                        0, 0, who, "", keel::format("numkit:{}:gendims", who));
        gen = matToRows(opt);
        h = matToRows(gen2par(opt, mr));
    } else {  // cyclic
        Value gp = opt;
        if (opt.isEmpty()) {
            gp = cyclpoly(n, k, "", mr);
            if (gp.numel() == 0)
                throw Error(keel::format("{}: no valid cyclic generator "
                            "polynomial for (N,K)", who),
                            0, 0, who, "", keel::format("numkit:{}:genpoly", who));
        }
        CyclgenResult cr = cyclgen(n, gp, "system", mr);
        gen = matToRows(cr.g);
        h = matToRows(cr.h);
    }
}

// MSB-first syndrome integer of error pattern e against parity-check h
// (m×n): bit j (row j of h) carries weight 2^(m-1-j).
inline long long syndromeInt(const keel::Array<keel::Array<int>> &h,
                             keel::ArrayRef<int> e, long long m, long long n)
{
    long long s = 0;
    for (long long j = 0; j < m; ++j) {
        int b = 0;
        for (long long i = 0; i < n; ++i) b ^= (h[j][i] & e[i]);
        s = (s << 1) | (b & 1);
    }
    return s;
}

// Standard-array syndrome decoding table: row (syndrome integer) -> the
// minimum-weight coset leader (error pattern). Weight 0, then 1, 2, ...
// until every syndrome has a leader. (Tie-breaking for weight >= 2 leaders
// follows lexicographic position order, which suffices for the common
// single-error-correcting codes.)
keel::Array<keel::Array<int>>
buildSyndTable(const keel::Array<keel::Array<int>> &h, long long n, long long m)
{
    const long long T = 1LL << m;
    keel::Array<keel::Array<int>> table(static_cast<keel::usize>(T));
    keel::Array<char> filled(static_cast<keel::usize>(T), 0);
    long long remaining = T;

    keel::Array<int> e(static_cast<keel::usize>(n), 0);
    const long long s0 = syndromeInt(h, e, m, n);
    table[static_cast<keel::usize>(s0)] = keel::move(e);
    filled[static_cast<keel::usize>(s0)] = 1;
    --remaining;

    for (long long w = 1; w <= n && remaining > 0; ++w) {
        keel::Array<long long> idx(static_cast<keel::usize>(w));
        for (long long t = 0; t < w; ++t) idx[static_cast<keel::usize>(t)] = t;
        while (true) {
            keel::Array<int> ep(static_cast<keel::usize>(n), 0);
            for (long long t = 0; t < w; ++t) ep[static_cast<keel::usize>(idx[t])] = 1;
            const long long s = syndromeInt(h, ep, m, n);
            if (!filled[static_cast<keel::usize>(s)]) {
                table[static_cast<keel::usize>(s)] = keel::move(ep);
                filled[static_cast<keel::usize>(s)] = 1;
                if (--remaining == 0) break;
            }
            long long t = w - 1;
            while (t >= 0 && idx[static_cast<keel::usize>(t)] == n - w + t) --t;
            if (t < 0) break;
            ++idx[static_cast<keel::usize>(t)];
            for (long long u = t + 1; u < w; ++u)
                idx[static_cast<keel::usize>(u)] = idx[static_cast<keel::usize>(u - 1)] + 1;
        }
    }
    return table;
}

} // namespace

// ── encode ──────────────────────────────────────────────────────────────

EncodeResult encode(const Value &msg, long long n, long long k,
                    keel::StringRef method, const Value &opt,
                    keel::Allocator *mr)
{
    if (n < 1) throw Error("encode: N must be a positive integer",
                           0, 0, "encode", "", "numkit:encode:N");
    if (k < 1) throw Error("encode: K must be a positive integer",
                           0, 0, "encode", "", "numkit:encode:K");
    if (n <= k) throw Error("encode: N must exceed K",
                            0, 0, "encode", "", "numkit:encode:nlek");

    keel::String base; bool decimal;
    parseMethod(method, base, decimal, "encode");

    const keel::usize R = msg.dims().rows(), C = msg.dims().cols();
    const keel::usize nel = msg.numel();
    const bool isVector = (R == 1 || C == 1);
    const bool isRowVector = isVector && (R == 1);

    keel::Array<keel::Array<int>> words;
    long long added = 0;
    int typeFlag = 0;

    if (decimal) {
        typeFlag = 1;
        if (!isVector)
            throw Error("encode: message must be a vector for the decimal format",
                        0, 0, "encode", "", "numkit:encode:msgvec");
        const double maxv = std::ldexp(1.0, static_cast<int>(k)) - 1.0;
        for (keel::usize w = 0; w < nel; ++w) {
            const double x = msg.elemAsDouble(w);
            if (x < 0 || x > maxv || std::floor(x) != x)
                throw Error("encode: decimal message entries must be integers "
                            "in [0, 2^K-1]",
                            0, 0, "encode", "", "numkit:encode:msgint");
            keel::Array<int> word(static_cast<keel::usize>(k), 0);
            const unsigned long long xi = static_cast<unsigned long long>(x);
            for (long long p = 0; p < k; ++p)
                word[static_cast<keel::usize>(p)] = static_cast<int>((xi >> p) & 1ULL);
            words.push_back(keel::move(word));
        }
    } else {
        for (keel::usize i = 0; i < nel; ++i) {
            const double x = msg.elemAsDouble(i);
            if (x != 0.0 && x != 1.0)
                throw Error("encode: binary message must contain only 0s and 1s",
                            0, 0, "encode", "", "numkit:encode:bin");
        }
        if (isVector) {
            typeFlag = 2;
            added = static_cast<long long>(nel % static_cast<keel::usize>(k));
            if (added) added = k - added;
            const keel::usize numWords = (nel + static_cast<keel::usize>(added))
                                         / static_cast<keel::usize>(k);
            for (keel::usize w = 0; w < numWords; ++w) {
                keel::Array<int> word(static_cast<keel::usize>(k), 0);
                for (long long p = 0; p < k; ++p) {
                    const keel::usize idx = w * static_cast<keel::usize>(k) + p;
                    if (idx < nel) word[static_cast<keel::usize>(p)] =
                        msg.elemAsDouble(idx) != 0.0 ? 1 : 0;
                }
                words.push_back(keel::move(word));
            }
        } else {
            if (static_cast<long long>(C) != k)
                throw Error("encode: message matrix must have K columns",
                            0, 0, "encode", "", "numkit:encode:cols");
            typeFlag = 0;
            for (keel::usize w = 0; w < R; ++w) {
                keel::Array<int> word(static_cast<keel::usize>(k));
                for (long long p = 0; p < k; ++p)
                    word[static_cast<keel::usize>(p)] = matBit(msg, w, static_cast<keel::usize>(p));
                words.push_back(keel::move(word));
            }
        }
    }

    keel::Array<keel::Array<int>> gen, h;
    buildCode(base, n, k, opt, "encode", gen, h, mr);

    const keel::usize numWords = words.size();
    const keel::usize N = static_cast<keel::usize>(n);
    keel::Array<keel::Array<int>> code;
    code.reserve(numWords);
    for (keel::usize w = 0; w < numWords; ++w) {
        keel::Array<int> row(N, 0);
        for (long long c = 0; c < n; ++c) {
            int b = 0;
            for (long long p = 0; p < k; ++p)
                b ^= (words[w][static_cast<keel::usize>(p)] & gen[static_cast<keel::usize>(p)][static_cast<keel::usize>(c)]);
            row[static_cast<keel::usize>(c)] = b & 1;
        }
        code.push_back(keel::move(row));
    }

    EncodeResult res;
    res.added = added;
    if (typeFlag == 1) {
        res.code = isRowVector ? Value::matrix(1, numWords, ValueType::DOUBLE, mr)
                               : Value::matrix(numWords, 1, ValueType::DOUBLE, mr);
        double *d = res.code.doubleDataMut();
        for (keel::usize w = 0; w < numWords; ++w) {
            double v = 0;
            for (long long p = 0; p < n; ++p)
                if (code[w][static_cast<keel::usize>(p)]) v += std::ldexp(1.0, static_cast<int>(p));
            d[w] = v;
        }
    } else if (typeFlag == 2) {
        const keel::usize L = numWords * N;
        res.code = isRowVector ? Value::matrix(1, L, ValueType::DOUBLE, mr)
                               : Value::matrix(L, 1, ValueType::DOUBLE, mr);
        double *d = res.code.doubleDataMut();
        for (keel::usize w = 0; w < numWords; ++w)
            for (keel::usize p = 0; p < N; ++p) d[w * N + p] = code[w][p];
    } else {
        res.code = Value::matrix(numWords, N, ValueType::DOUBLE, mr);
        double *d = res.code.doubleDataMut();
        for (keel::usize w = 0; w < numWords; ++w)
            for (keel::usize c = 0; c < N; ++c) d[c * numWords + w] = code[w][c];
    }
    return res;
}

// ── decode ──────────────────────────────────────────────────────────────

DecodeResult decode(const Value &code, long long n, long long k,
                    keel::StringRef method, const Value &opt,
                    keel::Allocator *mr)
{
    if (n < 1) throw Error("decode: N must be a positive integer",
                           0, 0, "decode", "", "numkit:decode:N");
    if (k < 1) throw Error("decode: K must be a positive integer",
                           0, 0, "decode", "", "numkit:decode:K");
    if (n <= k) throw Error("decode: N must exceed K",
                            0, 0, "decode", "", "numkit:decode:nlek");

    keel::String base; bool decimal;
    parseMethod(method, base, decimal, "decode");

    const keel::usize R = code.dims().rows(), C = code.dims().cols();
    const keel::usize nel = code.numel();
    const bool isVector = (R == 1 || C == 1);
    const bool isRowVector = isVector && (R == 1);
    const keel::usize N = static_cast<keel::usize>(n);

    keel::Array<keel::Array<int>> wordsIn;
    int typeFlag = 0;

    if (decimal) {
        typeFlag = 1;
        if (!isVector)
            throw Error("decode: code must be a vector for the decimal format",
                        0, 0, "decode", "", "numkit:decode:codevec");
        const double maxv = std::ldexp(1.0, static_cast<int>(n)) - 1.0;
        for (keel::usize w = 0; w < nel; ++w) {
            const double x = code.elemAsDouble(w);
            if (x < 0 || x > maxv || std::floor(x) != x)
                throw Error("decode: decimal code entries must be integers in "
                            "[0, 2^N-1]",
                            0, 0, "decode", "", "numkit:decode:codeint");
            keel::Array<int> word(N, 0);
            const unsigned long long xi = static_cast<unsigned long long>(x);
            for (keel::usize p = 0; p < N; ++p) word[p] = static_cast<int>((xi >> p) & 1ULL);
            wordsIn.push_back(keel::move(word));
        }
    } else {
        for (keel::usize i = 0; i < nel; ++i) {
            const double x = code.elemAsDouble(i);
            if (x != 0.0 && x != 1.0)
                throw Error("decode: binary code must contain only 0s and 1s",
                            0, 0, "decode", "", "numkit:decode:bin");
        }
        if (isVector) {
            typeFlag = 2;
            long long pad = static_cast<long long>(nel % N);
            if (pad) pad = static_cast<long long>(N) - pad;
            const keel::usize numWords = (nel + static_cast<keel::usize>(pad)) / N;
            for (keel::usize w = 0; w < numWords; ++w) {
                keel::Array<int> word(N, 0);
                for (keel::usize p = 0; p < N; ++p) {
                    const keel::usize idx = w * N + p;
                    if (idx < nel) word[p] = code.elemAsDouble(idx) != 0.0 ? 1 : 0;
                }
                wordsIn.push_back(keel::move(word));
            }
        } else {
            if (static_cast<long long>(C) != n)
                throw Error("decode: code matrix must have N columns",
                            0, 0, "decode", "", "numkit:decode:cols");
            typeFlag = 0;
            for (keel::usize w = 0; w < R; ++w) {
                keel::Array<int> word(N);
                for (keel::usize p = 0; p < N; ++p) word[p] = matBit(code, w, p);
                wordsIn.push_back(keel::move(word));
            }
        }
    }

    keel::Array<keel::Array<int>> gen, h;
    buildCode(base, n, k, opt, "decode", gen, h, mr);
    const long long m = n - k;
    const keel::Array<keel::Array<int>> trt = buildSyndTable(h, n, m);

    // Systematic message columns: gen = [I_k | ...] or [... | I_k].
    bool firstK = true, lastK = true;
    for (long long i = 0; i < k && (firstK || lastK); ++i)
        for (long long j = 0; j < k; ++j) {
            const int want = (i == j) ? 1 : 0;
            if (gen[static_cast<keel::usize>(i)][static_cast<keel::usize>(j)] != want) firstK = false;
            if (gen[static_cast<keel::usize>(i)][static_cast<keel::usize>(n - k + j)] != want) lastK = false;
        }
    if (!firstK && !lastK)
        throw Error("decode: generator matrix is not in systematic form",
                    0, 0, "decode", "", "numkit:decode:gensys");
    const long long msgOff = firstK ? 0 : (n - k);

    const keel::usize numWords = wordsIn.size();
    const keel::usize K = static_cast<keel::usize>(k);
    keel::Array<keel::Array<int>> msgRows;
    msgRows.reserve(numWords);
    keel::Array<long long> errCnt(numWords, 0);
    for (keel::usize w = 0; w < numWords; ++w) {
        const long long s = syndromeInt(h, wordsIn[w], m, n);
        const keel::Array<int> &leader = trt[static_cast<keel::usize>(s)];
        keel::Array<int> msgRow(K, 0);
        long long cnt = 0;
        for (keel::usize p = 0; p < K; ++p)
            msgRow[p] = wordsIn[w][static_cast<keel::usize>(msgOff) + p] ^ leader[static_cast<keel::usize>(msgOff) + p];
        msgRows.push_back(keel::move(msgRow));
        for (keel::usize p = 0; p < N; ++p) cnt += leader[p];
        errCnt[w] = cnt;
    }

    DecodeResult res;
    // msg output
    if (typeFlag == 1) {
        res.msg = isRowVector ? Value::matrix(1, numWords, ValueType::DOUBLE, mr)
                              : Value::matrix(numWords, 1, ValueType::DOUBLE, mr);
        double *d = res.msg.doubleDataMut();
        for (keel::usize w = 0; w < numWords; ++w) {
            double v = 0;
            for (keel::usize p = 0; p < K; ++p)
                if (msgRows[w][p]) v += std::ldexp(1.0, static_cast<int>(p));
            d[w] = v;
        }
    } else if (typeFlag == 2) {
        const keel::usize L = numWords * K;
        res.msg = isRowVector ? Value::matrix(1, L, ValueType::DOUBLE, mr)
                              : Value::matrix(L, 1, ValueType::DOUBLE, mr);
        double *d = res.msg.doubleDataMut();
        for (keel::usize w = 0; w < numWords; ++w)
            for (keel::usize p = 0; p < K; ++p) d[w * K + p] = msgRows[w][p];
    } else {
        res.msg = Value::matrix(numWords, K, ValueType::DOUBLE, mr);
        double *d = res.msg.doubleDataMut();
        for (keel::usize w = 0; w < numWords; ++w)
            for (keel::usize c = 0; c < K; ++c) d[c * numWords + w] = msgRows[w][c];
    }
    // err output: one count per word (column).
    res.err = Value::matrix(numWords, 1, ValueType::DOUBLE, mr);
    double *ed = res.err.doubleDataMut();
    for (keel::usize w = 0; w < numWords; ++w) ed[w] = static_cast<double>(errCnt[w]);
    return res;
}

namespace {

struct SyndPlan {
    keel::usize nk;   // parity rows (n-k)
    keel::usize n;    // codeword length
};

// syndtable_argcheck: parity-check matrix contract, crossed by every
// call path (the reg is a marshal-only passthrough).
SyndPlan syndtable_argcheck(const Value &H)
{
    const keel::usize nk = H.dims().rows();
    const keel::usize n  = H.dims().cols();
    if (nk == 0 || n == 0)
        throw Error("syndtable: H must be a nonempty (n-k) x n parity-check matrix",
                    0, 0, "syndtable", "", "numkit:syndtable:empty");
    if (nk > 24)
        throw Error("syndtable: n-k too large (table would be 2^(n-k) rows)",
                    0, 0, "syndtable", "", "numkit:syndtable:size");
    return {nk, n};
}

// Next lexicographic combination of `w` indices drawn from [0, n), in place.
// Returns false when `c` was the last combination.
bool nextCombination(keel::Array<keel::usize> &c, keel::usize n) {
    const keel::usize w = c.size();
    if (w == 0) return false;
    keel::usize i = w - 1;
    while (true) {
        if (c[i] < n - w + i) {
            ++c[i];
            for (keel::usize j = i + 1; j < w; ++j) c[j] = c[j - 1] + 1;
            return true;
        }
        if (i == 0) return false;
        --i;
    }
}
} // anonymous

Value syndtable(const Value &H, keel::Allocator *mr)
{
    const SyndPlan plan = syndtable_argcheck(H);
    const keel::usize nk = plan.nk;
    const keel::usize n  = plan.n;

    const keel::usize M = static_cast<keel::usize>(1) << nk;

    // Syndrome of a single-bit error at position j = column j of H, packed
    // MSB-first (row 0 of H is the most significant bit, matching
    // bi2de(..., 'left-msb')).
    keel::Array<keel::u32> colSynd(n, 0, mr);
    for (keel::usize j = 0; j < n; ++j) {
        keel::u32 s = 0;
        for (keel::usize i = 0; i < nk; ++i)
            if (H.elemAsDouble(j * nk + i) != 0.0)
                s |= (keel::u32{1} << (nk - 1 - i));
        colSynd[j] = s;
    }

    Value out = Value::matrix(M, n, ValueType::DOUBLE, mr);
    double *d = out.doubleDataMut();              // column-major: (row, col) at col*M + row
    std::fill(d, d + M * n, 0.0);

    keel::Array<keel::u8> filled(M, 0, mr);
    filled[0] = 1;                                 // syndrome 0 → all-zero pattern
    keel::usize cnt = 1;

    keel::Array<keel::usize> comb(mr);
    for (keel::usize w = 1; w <= n && cnt < M; ++w) {
        comb.resize(w);
        for (keel::usize i = 0; i < w; ++i) comb[i] = i;
        do {
            keel::u32 s = 0;
            for (keel::usize p : comb) s ^= colSynd[p];
            if (!filled[s]) {
                filled[s] = 1;
                for (keel::usize p : comb) d[p * M + s] = 1.0;
                if (++cnt == M) break;
            }
        } while (nextCombination(comb, n));
    }
    return out;
}

} // namespace numkit::comm
