// toolboxes/comm/src/source/huffman.cpp
//
// huffmandict: build a Huffman dictionary from symbol probabilities.
//
// Returned as an N-by-2 cell array:
//   dict{k, 1} = symbol value (scalar double)
//   dict{k, 2} = bit code (1×L row vector of 0/1 doubles)
//
// Optional second output is the average code length
// avglen = Σ p_i · length(code_i).
//
// Algorithm: classic min-heap of (combined_prob, node_id) pairs.
// Two smallest pulled per step, fused as parent with bit assignment
// 0=first, 1=second; tree walked top-down to label leaves with
// their bit string. The MATLAB tie-breaking convention is matched
// by sorting with stable order (lexicographic by current symbol
// index), which reproduces MATLAB R2025b on the standard probe set.

#include <keel/array.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/math.h>
#include <keel/ref.h>
#include <numkit/comm/source/huffman.hpp>

#include <numkit/value/argcheck.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>

namespace numkit::comm {

namespace {

struct HeapNode {
    keel::f64 prob;
    keel::i32    id;        // Node id; leaves are 0..K-1, internals are K..2K-2.
    keel::i32    order;     // Insertion order to break ties stably.
    bool operator>(const HeapNode &b) const {
        if (prob != b.prob) return prob > b.prob;
        return order > b.order;
    }
};

struct TreeNode {
    keel::i32 parent  = -1;
    keel::i32 bit     = 0;  // 0 if this is the "first/lower" child, 1 otherwise.
};

void huffmandict_argcheck(const Value &symbols, const Value &probs)
{
    const keel::usize K = symbols.numel();
    if (K == 0)
        throw Error("huffmandict: symbols must be non-empty",
                    0, 0, "huffmandict", "", "numkit:huffmandict:Empty");
    if (probs.numel() != K)
        throw Error("huffmandict: probs must match length of symbols",
                    0, 0, "huffmandict", "", "numkit:huffmandict:LenMismatch");

    // Validate probs.
    keel::f64 sum = 0.0;
    for (keel::usize k = 0; k < K; ++k) {
        const keel::f64 p = probs.elemAsDouble(k);
        if (!(p >= 0.0 && p <= 1.0))
            throw Error("huffmandict: probabilities must lie in [0, 1]",
                        0, 0, "huffmandict", "",
                        "numkit:huffmandict:InvalidProb");
        sum += p;
    }
    if (keel::abs(sum - 1.0) > std::sqrt(2.220446049250313e-16))
        throw Error("huffmandict: probabilities must sum to 1",
                    0, 0, "huffmandict", "",
                    "numkit:huffmandict:InvalidProbSum");
}

} // namespace

std::pair<Value, keel::f64>
huffmandict(const Value &symbols, const Value &probs,
            keel::Allocator *mr)
{
    huffmandict_argcheck(symbols, probs);
    const keel::usize K = symbols.numel();

    // Edge case: a single symbol gets a single-bit "0" code.
    if (K == 1) {
        Value dict = Value::cell(1, 2, mr);
        dict.cellAt(0)               = Value::scalar(symbols.elemAsDouble(0), mr);
        Value code = Value::matrix(1, 1, ValueType::DOUBLE, mr);
        code.doubleDataMut()[0] = 0.0;
        dict.cellAt(1)               = std::move(code);
        return {std::move(dict), 1.0};
    }

    // Reserve tree storage: 2K-1 nodes (K leaves + K-1 internals).
    keel::Array<TreeNode> tree(2 * K - 1, mr);

    // Build initial leaves min-heap.
    keel::Array<HeapNode> heap(mr);
    heap.reserve(K);
    auto heapCmp = [](const HeapNode &a, const HeapNode &b) { return a > b; };
    keel::i32 order_seed = 0;
    for (keel::usize k = 0; k < K; ++k) {
        heap.push_back({probs.elemAsDouble(k), keel::as<keel::i32>(k), order_seed++});
        std::push_heap(heap.begin(), heap.end(), heapCmp);
    }

    keel::i32 next_id = keel::as<keel::i32>(K);
    while (heap.size() > 1) {
        std::pop_heap(heap.begin(), heap.end(), heapCmp);
        HeapNode a = heap.back(); heap.pop_back();
        std::pop_heap(heap.begin(), heap.end(), heapCmp);
        HeapNode b = heap.back(); heap.pop_back();
        // a is the smaller -> assign bit 0; b -> bit 1.
        tree[a.id].parent = next_id;  tree[a.id].bit = 0;
        tree[b.id].parent = next_id;  tree[b.id].bit = 1;
        heap.push_back({a.prob + b.prob, next_id, order_seed++});
        std::push_heap(heap.begin(), heap.end(), heapCmp);
        ++next_id;
    }
    // Heap root is the tree root; tree[root].parent stays -1.

    // Walk leaf -> root, collect bits, reverse. Build dict cell.
    Value dict = Value::cell(K, 2, mr);
    keel::f64 avglen = 0.0;
    keel::Array<keel::f64> bits(mr);
    for (keel::usize k = 0; k < K; ++k) {
        bits.clear();
        keel::i32 cur = keel::as<keel::i32>(k);
        while (tree[cur].parent != -1) {
            bits.push_back(keel::as<keel::f64>(tree[cur].bit));
            cur = tree[cur].parent;
        }
        std::reverse(bits.begin(), bits.end());
        const keel::usize L = bits.size();
        // MATLAB cell layout is column-major: column 0 = symbols, column 1 = codes.
        dict.cellAt(k)           = Value::scalar(symbols.elemAsDouble(k), mr);
        Value code = Value::matrix(1, L, ValueType::DOUBLE, mr);
        if (L > 0)
            std::copy(bits.begin(), bits.end(), code.doubleDataMut());
        dict.cellAt(K + k)       = std::move(code);
        avglen += probs.elemAsDouble(k) * keel::as<keel::f64>(L);
    }
    return {std::move(dict), avglen};
}

// ── huffmanenco / huffmandeco helpers ──────────────────────────────

namespace {

// Read dict: K-by-2 cell, column 0 = symbol scalars, column 1 = code rows.
// Output:
//   syms[k]      = symbol value
//   codes[k]     = vector<uint8_t> of 0/1
struct DictView {
    keel::Array<keel::f64>                 syms;
    keel::Array<keel::Array<keel::u8>>  codes;
    keel::usize K = 0;
};

DictView readDict(const Value &dict)
{
    if (dict.dims().cols() != 2)
        throw Error("huffman codec: dict must be K-by-2 cell",
                    0, 0, "huffman", "", "numkit:huffman:DictShape");
    const keel::usize K = dict.dims().rows();
    DictView dv;
    dv.K = K;
    dv.syms.resize(K);
    dv.codes.resize(K);
    for (keel::usize k = 0; k < K; ++k) {
        // Cell layout: column-major. Index k = (row=k, col=0); k+K = col 1.
        const Value &symV  = dict.cellAt(k);
        const Value &codeV = dict.cellAt(k + K);
        dv.syms[k] = symV.toScalar();
        const keel::usize L = codeV.numel();
        dv.codes[k].resize(L);
        for (keel::usize b = 0; b < L; ++b) {
            const keel::f64 v = codeV.elemAsDouble(b);
            if (v != 0.0 && v != 1.0)
                throw Error("huffman codec: dict codes must contain 0/1",
                            0, 0, "huffman", "", "numkit:huffman:DictBits");
            dv.codes[k][b] = keel::as<keel::u8>(v);
        }
    }
    return dv;
}

// Lookup symbol -> code index. Linear scan; K is small (typically <=256).
keel::i32 findSymbolIdx(const DictView &dv, keel::f64 sym)
{
    for (keel::usize k = 0; k < dv.K; ++k) {
        if (dv.syms[k] == sym) return keel::as<keel::i32>(k);
    }
    return -1;
}

// Decode-side: build a prefix tree.
//   left/right child indices (-1 = none); leaf_sym = >=0 symbol index.
struct DecodeTree {
    struct Node {
        keel::i32 left  = -1;
        keel::i32 right = -1;
        keel::i32 sym   = -1;   // index into dv.syms; -1 if internal node.
    };
    keel::Array<Node> nodes{ Node{} };  // start with root.

    void insert(keel::ArrayRef<keel::u8> code, keel::i32 sym_idx)
    {
        keel::i32 cur = 0;
        for (keel::u8 bit : code) {
            keel::i32 child = (bit == 0) ? nodes[cur].left
                                   : nodes[cur].right;
            if (child < 0) {
                // push_back may reallocate -> read by value above, then
                // write back via index after the push completes.
                nodes.push_back(Node{});
                child = keel::as<keel::i32>(nodes.size() - 1);
                if (bit == 0) nodes[cur].left  = child;
                else          nodes[cur].right = child;
            }
            cur = child;
        }
        nodes[cur].sym = sym_idx;
    }
};

bool isRowOriented(const Value &v)
{
    return v.dims().rows() == 1 && v.dims().cols() >= 1;
}

DictView huffmanenco_argcheck(const Value &dict)
{
    return readDict(dict);
}

DictView huffmandeco_argcheck(const Value &dict)
{
    return readDict(dict);
}

} // namespace

Value huffmanenco(const Value &sig, const Value &dict,
                  keel::Allocator *mr)
{
    DictView dv = huffmanenco_argcheck(dict);

    // Walk sig once to compute output bit count.
    const keel::usize N = sig.numel();
    keel::usize total_bits = 0;
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 s = sig.elemAsDouble(i);
        const keel::i32 k = findSymbolIdx(dv, s);
        if (k < 0)
            throw Error("huffmanenco: symbol not in dict",
                        0, 0, "huffmanenco", "",
                        "numkit:huffmanenco:UnknownSym");
        total_bits += dv.codes[k].size();
    }

    const bool row = isRowOriented(sig);
    Value out = Value::matrix(row ? 1 : total_bits,
                              row ? total_bits : 1,
                              ValueType::DOUBLE, mr);
    keel::f64 *o = out.doubleDataMut();

    keel::usize pos = 0;
    for (keel::usize i = 0; i < N; ++i) {
        const keel::i32 k = findSymbolIdx(dv, sig.elemAsDouble(i));
        for (keel::u8 bit : dv.codes[k])
            o[pos++] = keel::as<keel::f64>(bit);
    }
    return out;
}

Value huffmandeco(const Value &bits, const Value &dict,
                  keel::Allocator *mr)
{
    DictView dv = huffmandeco_argcheck(dict);

    // Build decode tree.
    DecodeTree tree;
    for (keel::usize k = 0; k < dv.K; ++k)
        tree.insert(dv.codes[k], keel::as<keel::i32>(k));

    // Walk bits, emit symbols.
    const keel::usize N = bits.numel();
    keel::Array<keel::f64> emitted(mr);
    keel::i32 cur = 0;
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 v = bits.elemAsDouble(i);
        if (v != 0.0 && v != 1.0)
            throw Error("huffmandeco: input bits must be 0 or 1",
                        0, 0, "huffmandeco", "",
                        "numkit:huffmandeco:NonBit");
        const keel::i32 next = (v == 0.0) ? tree.nodes[cur].left
                                    : tree.nodes[cur].right;
        if (next < 0)
            throw Error("huffmandeco: bit pattern does not match dict",
                        0, 0, "huffmandeco", "",
                        "numkit:huffmandeco:NoMatch");
        cur = next;
        if (tree.nodes[cur].sym >= 0) {
            emitted.push_back(dv.syms[tree.nodes[cur].sym]);
            cur = 0;
        }
    }
    if (cur != 0)
        throw Error("huffmandeco: trailing bits not a complete code",
                    0, 0, "huffmandeco", "",
                    "numkit:huffmandeco:Incomplete");

    const bool row = isRowOriented(bits);
    const keel::usize M = emitted.size();
    Value out = Value::matrix(row ? 1 : M, row ? M : 1,
                              ValueType::DOUBLE, mr);
    if (M > 0) std::copy(emitted.begin(), emitted.end(), out.doubleDataMut());
    return out;
}

} // namespace numkit::comm
