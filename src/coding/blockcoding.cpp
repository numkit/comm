// toolboxes/comm/src/coding/blockcoding.cpp
//
// Linear block coding (gen2par, hammgen, cyclpoly, cyclgen, encode, decode,
// syndtable), delegating GF(2) numerical kernels to keel::comm.

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
#include <numkit/comm/coding/blockcoding.hpp>

#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

namespace numkit::comm {

namespace {

inline keel::String toLower(keel::StringRef sv)
{
    keel::String s(sv);
    s.to_ascii_lower_in_place();
    return s;
}

keel::Array<keel::u8> readBitsColMajor(const Value &v, keel::Allocator *alloc)
{
    const keel::usize rows = v.dims().rows();
    const keel::usize cols = v.dims().cols();
    keel::Array<keel::u8> out(rows * cols, 0u, alloc);
    for (keel::usize j = 0; j < cols; ++j)
        for (keel::usize i = 0; i < rows; ++i)
            out[j * rows + i] = (v.elemAsDouble(j * rows + i) != 0.0) ? 1u : 0u;
    return out;
}

Value packBitsColMajor(keel::ArrayRef<keel::u8> bits, keel::usize rows, keel::usize cols,
                       keel::Allocator *mr)
{
    Value out = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    keel::f64 *d = out.doubleDataMut();
    const keel::usize n = rows * cols;
    for (keel::usize i = 0; i < n; ++i) d[i] = keel::as<keel::f64>(bits[i]);
    return out;
}

void parseMethod(keel::StringRef method, keel::String &base, bool &decimal, const char *who)
{
    const keel::String m = toLower(method);
    decimal = m.contains("decimal");
    if (m.starts_with("hamming"))      base.assign("hamming");
    else if (m.starts_with("linear"))  base.assign("linear");
    else if (m.starts_with("cyclic"))  base.assign("cyclic");
    else throw Error(keel::format("{}: unknown coding method '{}'", who, method),
                     0, 0, who, "", keel::format("numkit:{}:method", who));
}

void buildCodeMatrices(keel::StringRef base, keel::i64 n, keel::i64 k,
                       const Value &opt, const char *who,
                       keel::Array<keel::u8> &gen_col, keel::Array<keel::u8> &h_col,
                       keel::Allocator *alloc)
{
    if (base == "hamming") {
        const keel::i64 m = n - k;
        if (((1LL << m) - 1) != n)
            throw Error(keel::format("{}: N and K do not form a Hamming "
                        "code (2^(N-K)-1 must equal N)", who),
                        0, 0, who, "", keel::format("numkit:{}:nklen", who));
        HammgenResult hr = hammgen(m, opt.isEmpty() ? Value::Empty : opt, alloc);
        h_col = readBitsColMajor(hr.h, alloc);
        gen_col = readBitsColMajor(hr.g, alloc);
    } else if (base == "linear") {
        if (opt.isEmpty())
            throw Error(keel::format("{}: the linear method requires a generator matrix", who),
                        0, 0, who, "", keel::format("numkit:{}:genmat", who));
        if (keel::as<keel::i64>(opt.dims().rows()) != k ||
            keel::as<keel::i64>(opt.dims().cols()) != n)
            throw Error(keel::format("{}: generator matrix must be K x N", who),
                        0, 0, who, "", keel::format("numkit:{}:gendims", who));
        gen_col = readBitsColMajor(opt, alloc);
        const auto st = keel::comm::gen2par(gen_col.as_ref(), keel::as<keel::usize>(k),
                                            keel::as<keel::usize>(n), h_col, alloc);
        if (st == keel::comm::CommStatus::kNotSystematic)
            throw Error(keel::format("{}: neither the first nor the last r "
                        "columns form an identity matrix (input not systematic)", who),
                        0, 0, who, "", keel::format("numkit:{}:notsystematic", who));
        if (st != keel::comm::CommStatus::kOk)
            throw Error(keel::format("{}: input must be r×c with r < c", who),
                        0, 0, who, "", keel::format("numkit:{}:shape", who));
    } else { // cyclic
        Value gp = opt;
        if (opt.isEmpty()) {
            gp = cyclpoly(n, k, "", alloc);
            if (gp.numel() == 0)
                throw Error(keel::format("{}: no valid cyclic generator polynomial for (N,K)", who),
                            0, 0, who, "", keel::format("numkit:{}:genpoly", who));
        }
        CyclgenResult cr = cyclgen(n, gp, "system", alloc);
        gen_col = readBitsColMajor(cr.g, alloc);
        h_col = readBitsColMajor(cr.h, alloc);
    }
}

} // namespace

Value gen2par(const Value &mat, keel::Allocator *mr)
{
    const keel::usize r = mat.dims().rows();
    const keel::usize c = mat.dims().cols();
    if (r == 0 || c == 0 || r >= c)
        throw Error("gen2par: input must be r×c with r < c",
                    0, 0, "gen2par", "", "numkit:gen2par:shape");

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const auto bits = readBitsColMajor(mat, scratch.allocator());
    keel::Array<keel::u8> out_bits(scratch.allocator());
    const auto st = keel::comm::gen2par(bits.as_ref(), r, c, out_bits, scratch.allocator());
    if (st == keel::comm::CommStatus::kNotSystematic)
        throw Error("gen2par: neither the first nor the last r "
                    "columns form an identity matrix (input not systematic)",
                    0, 0, "gen2par", "", "numkit:gen2par:notsystematic");
    return packBitsColMajor(out_bits.as_ref(), c - r, c, mr);
}

HammgenResult hammgen(keel::i64 m, const Value &primPoly, keel::Allocator *mr)
{
    if (m < 2)
        throw Error("hammgen: M must be an integer >= 2",
                    0, 0, "hammgen", "", "numkit:hammgen:M");

    keel::u64 pmask = 0ULL;
    if (primPoly.isEmpty()) {
        pmask = keel::comm::default_prim_poly(keel::as<keel::i32>(m));
        if (pmask == 0ULL)
            throw Error("hammgen: M out of supported range [2,16]",
                        0, 0, "hammgen", "", "numkit:hammgen:range");
    } else {
        const keel::usize len = primPoly.numel();
        if (keel::as<keel::i64>(len) != m + 1)
            throw Error("hammgen: P must be a binary row of length M+1",
                        0, 0, "hammgen", "", "numkit:hammgen:P");
        for (keel::usize j = 0; j < len; ++j)
            if (primPoly.elemAsDouble(j) != 0.0) pmask |= (1ULL << j);
        if ((pmask & 1ULL) == 0ULL || (pmask & (1ULL << m)) == 0ULL)
            throw Error("hammgen: P must have nonzero constant and degree-M terms",
                        0, 0, "hammgen", "", "numkit:hammgen:P");
    }

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::comm::BlockCodeMatrices bm;
    (void)keel::comm::hammgen(keel::as<keel::i32>(m), pmask, bm, scratch.allocator());

    HammgenResult res;
    res.h = packBitsColMajor(bm.h.as_ref(), keel::as<keel::usize>(m), bm.n, mr);
    res.g = packBitsColMajor(bm.g.as_ref(), bm.k, bm.n, mr);
    res.n = keel::as<keel::i64>(bm.n);
    res.k = keel::as<keel::i64>(bm.k);
    return res;
}

Value cyclpoly(keel::i64 n, keel::i64 k, keel::StringRef opt, keel::Allocator *mr)
{
    if (n < 1)
        throw Error("cyclpoly: N must be a positive integer",
                    0, 0, "cyclpoly", "", "numkit:cyclpoly:N");
    if (k < 1 || k > n)
        throw Error("cyclpoly: K must satisfy 1 <= K <= N",
                    0, 0, "cyclpoly", "", "numkit:cyclpoly:K");

    const keel::String o = toLower(opt);
    keel::comm::CyclPolyMode mode = keel::comm::CyclPolyMode::kAll;
    if (o.empty()) mode = keel::comm::CyclPolyMode::kFirst;
    else if (o.starts_with("mi")) mode = keel::comm::CyclPolyMode::kMin;
    else if (o.starts_with("ma")) mode = keel::comm::CyclPolyMode::kMax;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::comm::CyclPolyResult res;
    (void)keel::comm::cyclpoly(keel::as<keel::i32>(n), keel::as<keel::i32>(k), mode, res,
                               scratch.allocator());
    if (res.num_polys == 0)
        return Value::matrix(0, 0, ValueType::DOUBLE, mr);
    return packBitsColMajor(res.rows_col_major.as_ref(), res.num_polys, res.poly_len, mr);
}

CyclgenResult cyclgen(keel::i64 n, const Value &genpoly, keel::StringRef opt, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const keel::usize nel = genpoly.numel();
    keel::Array<keel::u8> p(nel, 0u, scratch.allocator());
    keel::i32 deg = -1;
    for (keel::usize i = 0; i < nel; ++i) {
        if (genpoly.elemAsDouble(i) != 0.0) {
            p[i] = 1u;
            deg = keel::as<keel::i32>(i);
        }
    }
    if (deg < 1)
        throw Error("cyclgen: generator polynomial must have degree >= 1",
                    0, 0, "cyclgen", "", "numkit:cyclgen:P");
    if (n - deg < 1)
        throw Error("cyclgen: N must exceed the generator polynomial degree",
                    0, 0, "cyclgen", "", "numkit:cyclgen:N");

    const bool systematic = !toLower(opt).contains("no");
    keel::comm::BlockCodeMatrices bm;
    const auto st = keel::comm::cyclgen(keel::as<keel::i32>(n), p.as_ref(), systematic, bm,
                                        scratch.allocator());
    if (st != keel::comm::CommStatus::kOk)
        throw Error("cyclgen: generator polynomial must divide x^N - 1",
                    0, 0, "cyclgen", "", "numkit:cyclgen:P2");

    CyclgenResult res;
    res.h = packBitsColMajor(bm.h.as_ref(), bm.n - bm.k, bm.n, mr);
    res.g = packBitsColMajor(bm.g.as_ref(), bm.k, bm.n, mr);
    res.k = keel::as<keel::i64>(bm.k);
    return res;
}

EncodeResult encode(const Value &msg, keel::i64 n, keel::i64 k,
                    keel::StringRef method, const Value &opt, keel::Allocator *mr)
{
    if (n < 1) throw Error("encode: N must be a positive integer", 0, 0, "encode", "", "numkit:encode:N");
    if (k < 1) throw Error("encode: K must be a positive integer", 0, 0, "encode", "", "numkit:encode:K");
    if (n <= k) throw Error("encode: N must exceed K", 0, 0, "encode", "", "numkit:encode:nlek");

    keel::String base;
    bool decimal = false;
    parseMethod(method, base, decimal, "encode");

    const keel::usize R = msg.dims().rows(), C = msg.dims().cols();
    const keel::usize nel = msg.numel();
    const bool isVector = (R == 1 || C == 1);
    const bool isRowVector = isVector && (R == 1);
    const keel::usize K = keel::as<keel::usize>(k);
    const keel::usize N = keel::as<keel::usize>(n);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::u8> msg_words(scratch.allocator());
    keel::usize numWords = 0;
    keel::i64 added = 0;
    keel::i32 typeFlag = 0;

    if (decimal) {
        typeFlag = 1;
        if (!isVector)
            throw Error("encode: message must be a vector for the decimal format",
                        0, 0, "encode", "", "numkit:encode:msgvec");
        const keel::f64 maxv = keel::ldexp(1.0, keel::as<keel::i32>(k)) - 1.0;
        numWords = nel;
        msg_words.resize(numWords * K, 0u);
        for (keel::usize w = 0; w < nel; ++w) {
            const keel::f64 x = msg.elemAsDouble(w);
            if (x < 0.0 || x > maxv || keel::floor(x) != x)
                throw Error("encode: decimal message entries must be integers in [0, 2^K-1]",
                            0, 0, "encode", "", "numkit:encode:msgint");
            const keel::u64 xi = keel::as<keel::u64>(x);
            for (keel::usize p = 0; p < K; ++p)
                msg_words[w * K + p] = keel::as<keel::u8>((xi >> p) & 1ULL);
        }
    } else {
        for (keel::usize i = 0; i < nel; ++i) {
            const keel::f64 x = msg.elemAsDouble(i);
            if (x != 0.0 && x != 1.0)
                throw Error("encode: binary message must contain only 0s and 1s",
                            0, 0, "encode", "", "numkit:encode:bin");
        }
        if (isVector) {
            typeFlag = 2;
            added = keel::as<keel::i64>(nel % K);
            if (added) added = k - added;
            numWords = (nel + keel::as<keel::usize>(added)) / K;
            msg_words.resize(numWords * K, 0u);
            for (keel::usize i = 0; i < nel; ++i)
                msg_words[i] = (msg.elemAsDouble(i) != 0.0) ? 1u : 0u;
        } else {
            if (keel::as<keel::i64>(C) != k)
                throw Error("encode: message matrix must have K columns",
                            0, 0, "encode", "", "numkit:encode:cols");
            typeFlag = 0;
            numWords = R;
            msg_words.resize(numWords * K, 0u);
            for (keel::usize w = 0; w < R; ++w)
                for (keel::usize p = 0; p < K; ++p)
                    msg_words[w * K + p] = (msg.elemAsDouble(p * R + w) != 0.0) ? 1u : 0u;
        }
    }

    keel::Array<keel::u8> gen_col(scratch.allocator()), h_col(scratch.allocator());
    buildCodeMatrices(base, n, k, opt, "encode", gen_col, h_col, scratch.allocator());

    keel::Array<keel::u8> code_words(numWords * N, 0u, scratch.allocator());
    (void)keel::comm::block_encode(msg_words.as_ref(), numWords, gen_col.as_ref(), K, N,
                                   code_words.as_mut_ref());

    EncodeResult res;
    res.added = added;
    if (typeFlag == 1) {
        res.code = isRowVector ? Value::matrix(1, numWords, ValueType::DOUBLE, mr)
                               : Value::matrix(numWords, 1, ValueType::DOUBLE, mr);
        keel::f64 *d = res.code.doubleDataMut();
        for (keel::usize w = 0; w < numWords; ++w) {
            keel::f64 v = 0.0;
            for (keel::usize p = 0; p < N; ++p)
                if (code_words[w * N + p] != 0u) v += keel::ldexp(1.0, keel::as<keel::i32>(p));
            d[w] = v;
        }
    } else if (typeFlag == 2) {
        const keel::usize L = numWords * N;
        res.code = isRowVector ? Value::matrix(1, L, ValueType::DOUBLE, mr)
                               : Value::matrix(L, 1, ValueType::DOUBLE, mr);
        keel::f64 *d = res.code.doubleDataMut();
        for (keel::usize i = 0; i < L; ++i) d[i] = keel::as<keel::f64>(code_words[i]);
    } else {
        res.code = Value::matrix(numWords, N, ValueType::DOUBLE, mr);
        keel::f64 *d = res.code.doubleDataMut();
        for (keel::usize w = 0; w < numWords; ++w)
            for (keel::usize c = 0; c < N; ++c)
                d[c * numWords + w] = keel::as<keel::f64>(code_words[w * N + c]);
    }
    return res;
}

DecodeResult decode(const Value &code, keel::i64 n, keel::i64 k,
                    keel::StringRef method, const Value &opt, keel::Allocator *mr)
{
    if (n < 1) throw Error("decode: N must be a positive integer", 0, 0, "decode", "", "numkit:decode:N");
    if (k < 1) throw Error("decode: K must be a positive integer", 0, 0, "decode", "", "numkit:decode:K");
    if (n <= k) throw Error("decode: N must exceed K", 0, 0, "decode", "", "numkit:decode:nlek");

    keel::String base;
    bool decimal = false;
    parseMethod(method, base, decimal, "decode");

    const keel::usize R = code.dims().rows(), C = code.dims().cols();
    const keel::usize nel = code.numel();
    const bool isVector = (R == 1 || C == 1);
    const bool isRowVector = isVector && (R == 1);
    const keel::usize K = keel::as<keel::usize>(k);
    const keel::usize N = keel::as<keel::usize>(n);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::u8> code_words(scratch.allocator());
    keel::usize numWords = 0;
    keel::i32 typeFlag = 0;

    if (decimal) {
        typeFlag = 1;
        if (!isVector)
            throw Error("decode: code must be a vector for the decimal format",
                        0, 0, "decode", "", "numkit:decode:codevec");
        const keel::f64 maxv = keel::ldexp(1.0, keel::as<keel::i32>(n)) - 1.0;
        numWords = nel;
        code_words.resize(numWords * N, 0u);
        for (keel::usize w = 0; w < nel; ++w) {
            const keel::f64 x = code.elemAsDouble(w);
            if (x < 0.0 || x > maxv || keel::floor(x) != x)
                throw Error("decode: decimal code entries must be integers in [0, 2^N-1]",
                            0, 0, "decode", "", "numkit:decode:codeint");
            const keel::u64 xi = keel::as<keel::u64>(x);
            for (keel::usize p = 0; p < N; ++p)
                code_words[w * N + p] = keel::as<keel::u8>((xi >> p) & 1ULL);
        }
    } else {
        for (keel::usize i = 0; i < nel; ++i) {
            const keel::f64 x = code.elemAsDouble(i);
            if (x != 0.0 && x != 1.0)
                throw Error("decode: binary code must contain only 0s and 1s",
                            0, 0, "decode", "", "numkit:decode:bin");
        }
        if (isVector) {
            typeFlag = 2;
            keel::usize pad = nel % N;
            if (pad != 0) pad = N - pad;
            numWords = (nel + pad) / N;
            code_words.resize(numWords * N, 0u);
            for (keel::usize i = 0; i < nel; ++i)
                code_words[i] = (code.elemAsDouble(i) != 0.0) ? 1u : 0u;
        } else {
            if (keel::as<keel::i64>(C) != n)
                throw Error("decode: code matrix must have N columns",
                            0, 0, "decode", "", "numkit:decode:cols");
            typeFlag = 0;
            numWords = R;
            code_words.resize(numWords * N, 0u);
            for (keel::usize w = 0; w < R; ++w)
                for (keel::usize p = 0; p < N; ++p)
                    code_words[w * N + p] = (code.elemAsDouble(p * R + w) != 0.0) ? 1u : 0u;
        }
    }

    keel::Array<keel::u8> gen_col(scratch.allocator()), h_col(scratch.allocator());
    buildCodeMatrices(base, n, k, opt, "decode", gen_col, h_col, scratch.allocator());

    keel::Array<keel::u8> msg_words(numWords * K, 0u, scratch.allocator());
    keel::Array<keel::i32> err_cnt(numWords, 0, scratch.allocator());
    const auto st = keel::comm::block_decode(code_words.as_ref(), numWords, gen_col.as_ref(),
                                             h_col.as_ref(), K, N, msg_words.as_mut_ref(),
                                             err_cnt.as_mut_ref(), scratch.allocator());
    if (st == keel::comm::CommStatus::kNotSystematic)
        throw Error("decode: generator matrix is not in systematic form",
                    0, 0, "decode", "", "numkit:decode:gensys");

    DecodeResult res;
    if (typeFlag == 1) {
        res.msg = isRowVector ? Value::matrix(1, numWords, ValueType::DOUBLE, mr)
                              : Value::matrix(numWords, 1, ValueType::DOUBLE, mr);
        keel::f64 *d = res.msg.doubleDataMut();
        for (keel::usize w = 0; w < numWords; ++w) {
            keel::f64 v = 0.0;
            for (keel::usize p = 0; p < K; ++p)
                if (msg_words[w * K + p] != 0u) v += keel::ldexp(1.0, keel::as<keel::i32>(p));
            d[w] = v;
        }
    } else if (typeFlag == 2) {
        const keel::usize L = numWords * K;
        res.msg = isRowVector ? Value::matrix(1, L, ValueType::DOUBLE, mr)
                              : Value::matrix(L, 1, ValueType::DOUBLE, mr);
        keel::f64 *d = res.msg.doubleDataMut();
        for (keel::usize i = 0; i < L; ++i) d[i] = keel::as<keel::f64>(msg_words[i]);
    } else {
        res.msg = Value::matrix(numWords, K, ValueType::DOUBLE, mr);
        keel::f64 *d = res.msg.doubleDataMut();
        for (keel::usize w = 0; w < numWords; ++w)
            for (keel::usize c = 0; c < K; ++c)
                d[c * numWords + w] = keel::as<keel::f64>(msg_words[w * K + c]);
    }

    res.err = Value::matrix(numWords, 1, ValueType::DOUBLE, mr);
    keel::f64 *ed = res.err.doubleDataMut();
    for (keel::usize w = 0; w < numWords; ++w) ed[w] = keel::as<keel::f64>(err_cnt[w]);
    return res;
}

Value syndtable(const Value &H, keel::Allocator *mr)
{
    const keel::usize nk = H.dims().rows();
    const keel::usize n  = H.dims().cols();
    if (nk == 0 || n == 0)
        throw Error("syndtable: H must be a nonempty (n-k) x n parity-check matrix",
                    0, 0, "syndtable", "", "numkit:syndtable:empty");
    if (nk > 24)
        throw Error("syndtable: n-k too large (table would be 2^(n-k) rows)",
                    0, 0, "syndtable", "", "numkit:syndtable:size");

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const auto h_bits = readBitsColMajor(H, scratch.allocator());
    const keel::usize M = keel::usize{1} << nk;
    keel::Array<keel::u8> tbl(M * n, 0u, scratch.allocator());
    (void)keel::comm::syndtable(h_bits.as_ref(), nk, n, tbl.as_mut_ref(), scratch.allocator());
    return packBitsColMajor(tbl.as_ref(), M, n, mr);
}

} // namespace numkit::comm
