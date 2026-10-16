// toolboxes/comm/src/source/huffman_reg.cpp
//
// Register half of the comm Huffman builtins: the CallContext wrappers
// huffmandict / huffmanenco / huffmandeco that destructure the (dict,
// avglen) pair from huffmandict and delegate to the engine-free compute in
// huffman.cpp. library.cpp forward-declares + registers these by name.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <numkit/comm/source/huffman.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, Span, ctx.engine->resource()
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>

#include <utility>

namespace numkit::comm {
namespace detail {

void huffmandict_reg(Span<const Value> args, size_t nargout,
                     Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 4, "huffmandict");
    auto *mr = ctx.engine->resource();
    auto [dict, avglen] = huffmandict(args[0], args[1], mr);
    outs[0] = std::move(dict);
    if (nargout > 1)
        outs[1] = Value::scalar(avglen, mr);
}

void huffmanenco_reg(Span<const Value> args, size_t /*nargout*/,
                     Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "huffmanenco");
    outs[0] = huffmanenco(args[0], args[1], ctx.engine->resource());
}

void huffmandeco_reg(Span<const Value> args, size_t /*nargout*/,
                     Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "huffmandeco");
    outs[0] = huffmandeco(args[0], args[1], ctx.engine->resource());
}

} // namespace detail

} // namespace numkit::comm
