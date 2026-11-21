// toolboxes/comm/src/eq/errors.cpp
//
// biterr / symerr — bit and symbol error counts/rates, delegated to keel::comm.

#include <keel/arena.h>
#include <keel/array.h>
#include <keel/comm.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <numkit/comm/eq/errors.hpp>

#include <numkit/value/argcheck.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <utility>

namespace numkit::comm {

namespace {

void biterr_argcheck(const Value &x, const Value &y, int k)
{
    const keel::usize n = x.numel();
    if (y.numel() != n)
        throw Error("biterr: x and y must have the same number of elements",
                    0, 0, "biterr", "", "numkit:biterr:size");
    if (k < 0)
        throw Error("biterr: k must be a positive integer",
                    0, 0, "biterr", "", "numkit:biterr:k");
    const keel::f64 *xd = x.doubleData();
    const keel::f64 *yd = y.doubleData();
    for (keel::usize i = 0; i < n; ++i) {
        if (xd[i] < 0.0 || yd[i] < 0.0)
            throw Error("biterr: inputs must be non-negative integers",
                        0, 0, "biterr", "", "numkit:biterr:negative");
    }
}

keel::String symerr_argcheck(const Value &x, const Value &y, keel::StringRef flg)
{
    keel::String mode(flg);
    for (char &ch : mode) {
        if (ch >= 'A' && ch <= 'Z') ch = keel::as<char>(ch - 'A' + 'a');
    }
    if (mode.empty() || mode == "overall") {
        if (y.numel() != x.numel())
            throw Error("symerr: x and y must have the same number of elements",
                        0, 0, "symerr", "", "numkit:symerr:size");
        return keel::String("overall");
    }
    if (mode == "column-wise" || mode == "row-wise") {
        if (y.dims().rows() != x.dims().rows() || y.dims().cols() != x.dims().cols())
            throw Error(keel::format("symerr: x and y must have the same dimensions for '{}'", mode),
                        0, 0, "symerr", "", "numkit:symerr:size");
        return mode;
    }
    throw Error("symerr: flg must be 'overall', 'column-wise', or 'row-wise'",
                0, 0, "symerr", "", "numkit:symerr:flg");
}

} // anonymous namespace

std::pair<Value, Value> biterr(const Value &x, const Value &y, int k,
                               keel::Allocator *mr)
{
    biterr_argcheck(x, y, k);
    const keel::usize n = x.numel();
    const keel::f64 *xd = x.doubleData();
    const keel::f64 *yd = y.doubleData();

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::u64> a(n, scratch.allocator());
    keel::Array<keel::u64> b(n, scratch.allocator());
    for (keel::usize i = 0; i < n; ++i) {
        a[i] = keel::as<keel::u64>(xd[i]);
        b[i] = keel::as<keel::u64>(yd[i]);
    }
    const auto stats = keel::comm::bit_error_rate(a, b, keel::as<keel::i32>(k));
    return {Value::scalar(keel::as<keel::f64>(stats.count), mr),
            Value::scalar(stats.ratio, mr)};
}

std::pair<Value, Value> symerr(const Value &x, const Value &y,
                               keel::StringRef flg, keel::Allocator *mr)
{
    const keel::String mode = symerr_argcheck(x, y, flg);

    if (mode == "overall") {
        const keel::usize n = x.numel();
        const auto stats = keel::comm::sym_error_rate({x.doubleData(), n}, {y.doubleData(), n});
        return {Value::scalar(keel::as<keel::f64>(stats.count), mr),
                Value::scalar(stats.ratio, mr)};
    }

    if (mode == "column-wise") {
        const keel::usize r = x.dims().rows();
        const keel::usize c = x.dims().cols();
        const keel::f64 *xd = x.doubleData();
        const keel::f64 *yd = y.doubleData();
        Value counts = Value::matrix(1, c, ValueType::DOUBLE, mr);
        Value ratios = Value::matrix(1, c, ValueType::DOUBLE, mr);
        keel::f64 *cd = counts.doubleDataMut();
        keel::f64 *rd = ratios.doubleDataMut();
        for (keel::usize j = 0; j < c; ++j) {
            const auto stats = keel::comm::sym_error_rate({xd + j * r, r}, {yd + j * r, r});
            cd[j] = keel::as<keel::f64>(stats.count);
            rd[j] = stats.ratio;
        }
        return {std::move(counts), std::move(ratios)};
    }

    const keel::usize r = x.dims().rows();
    const keel::usize c = x.dims().cols();
    const keel::f64 *xd = x.doubleData();
    const keel::f64 *yd = y.doubleData();
    Value counts = Value::matrix(r, 1, ValueType::DOUBLE, mr);
    Value ratios = Value::matrix(r, 1, ValueType::DOUBLE, mr);
    keel::f64 *cd = counts.doubleDataMut();
    keel::f64 *rd = ratios.doubleDataMut();
    for (keel::usize i = 0; i < r; ++i) {
        keel::u64 cnt = 0;
        for (keel::usize j = 0; j < c; ++j) {
            if (xd[j * r + i] != yd[j * r + i]) ++cnt;
        }
        cd[i] = keel::as<keel::f64>(cnt);
        rd[i] = (c > 0) ? keel::as<keel::f64>(cnt) / keel::as<keel::f64>(c) : 0.0;
    }
    return {std::move(counts), std::move(ratios)};
}

std::pair<Value, Value> symerr(const Value &x, const Value &y, keel::Allocator *mr)
{
    return symerr(x, y, "overall", mr);
}

} // namespace numkit::comm
