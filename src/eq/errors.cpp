// toolboxes/comm/src/eq/errors.cpp
//
// biterr / symerr -- bit and symbol error counts/rates between two
// arrays. MATLAB Communications Toolbox metrics.

#include <numkit/comm/eq/errors.hpp>

#include <numkit/value/scratch.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>

namespace numkit::comm {

namespace {

// Count bit differences between two non-negative integers (Hamming
// distance over their binary representations). Treats both as uint64.
inline std::uint64_t hammingBits(std::uint64_t a, std::uint64_t b)
{
    std::uint64_t x = a ^ b;
    std::uint64_t count = 0;
    while (x) { count += (x & 1); x >>= 1; }
    return count;
}

} // anonymous namespace

// ── Public C++ API (see eq/errors.hpp) ────────────────────────────────

std::pair<Value, Value> biterr(const Value &x, const Value &y, int k,
                               std::pmr::memory_resource *mr)
{
    const std::size_t n = x.numel();
    if (y.numel() != n)
        throw Error("biterr: x and y must have the same number of elements",
                    0, 0, "biterr", "", "numkit:biterr:size");

    const double *xd = x.doubleData();
    const double *yd = y.doubleData();

    // Determine k: explicit (>= 1), else smallest covering bit width.
    int k_bits = k;
    if (k_bits < 1) {
        std::uint64_t mx = 0;
        for (std::size_t i = 0; i < n; ++i) {
            const double xv = xd[i], yv = yd[i];
            if (xv < 0.0 || yv < 0.0)
                throw Error("biterr: inputs must be non-negative integers",
                            0, 0, "biterr", "", "numkit:biterr:negative");
            const std::uint64_t xi = static_cast<std::uint64_t>(xv);
            const std::uint64_t yi = static_cast<std::uint64_t>(yv);
            mx = std::max(mx, std::max(xi, yi));
        }
        k_bits = 1;
        while ((static_cast<std::uint64_t>(1) << k_bits) <= mx) ++k_bits;
    }

    std::uint64_t total = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint64_t xi = static_cast<std::uint64_t>(xd[i]);
        const std::uint64_t yi = static_cast<std::uint64_t>(yd[i]);
        total += hammingBits(xi, yi);
    }
    const double total_bits =
        static_cast<double>(n) * static_cast<double>(k_bits);
    const double ratio = (total_bits > 0.0)
        ? static_cast<double>(total) / total_bits
        : 0.0;
    return {Value::scalar(static_cast<double>(total), mr),
            Value::scalar(ratio, mr)};
}

std::pair<Value, Value> symerr(const Value &x, const Value &y,
                               std::string_view flg,
                               std::pmr::memory_resource *mr)
{
    std::string mode(flg);
    for (char &ch : mode) {
        if (ch >= 'A' && ch <= 'Z') ch = static_cast<char>(ch - 'A' + 'a');
    }

    if (mode.empty() || mode == "overall") {
        const std::size_t n = x.numel();
        if (y.numel() != n)
            throw Error("symerr: x and y must have the same number of elements",
                        0, 0, "symerr", "", "numkit:symerr:size");

        const double *xd = x.doubleData();
        const double *yd = y.doubleData();
        std::uint64_t count = 0;
        for (std::size_t i = 0; i < n; ++i)
            if (xd[i] != yd[i]) ++count;

        const double ratio = (n > 0)
            ? static_cast<double>(count) / static_cast<double>(n)
            : 0.0;
        return {Value::scalar(static_cast<double>(count), mr),
                Value::scalar(ratio, mr)};
    }

    if (mode == "column-wise") {
        const std::size_t r = x.dims().rows();
        const std::size_t c = x.dims().cols();
        if (y.dims().rows() != r || y.dims().cols() != c)
            throw Error("symerr: x and y must have the same dimensions for 'column-wise'",
                        0, 0, "symerr", "", "numkit:symerr:size");
        const double *xd = x.doubleData();
        const double *yd = y.doubleData();
        Value counts = Value::matrix(1, c, ValueType::DOUBLE, mr);
        Value ratios = Value::matrix(1, c, ValueType::DOUBLE, mr);
        double *cd = counts.doubleDataMut();
        double *rd = ratios.doubleDataMut();
        for (std::size_t j = 0; j < c; ++j) {
            std::uint64_t cnt = 0;
            for (std::size_t i = 0; i < r; ++i) {
                if (xd[j * r + i] != yd[j * r + i]) ++cnt;
            }
            cd[j] = static_cast<double>(cnt);
            rd[j] = (r > 0) ? static_cast<double>(cnt) / static_cast<double>(r) : 0.0;
        }
        return {std::move(counts), std::move(ratios)};
    }

    if (mode == "row-wise") {
        const std::size_t r = x.dims().rows();
        const std::size_t c = x.dims().cols();
        if (y.dims().rows() != r || y.dims().cols() != c)
            throw Error("symerr: x and y must have the same dimensions for 'row-wise'",
                        0, 0, "symerr", "", "numkit:symerr:size");
        const double *xd = x.doubleData();
        const double *yd = y.doubleData();
        Value counts = Value::matrix(r, 1, ValueType::DOUBLE, mr);
        Value ratios = Value::matrix(r, 1, ValueType::DOUBLE, mr);
        double *cd = counts.doubleDataMut();
        double *rd = ratios.doubleDataMut();
        for (std::size_t i = 0; i < r; ++i) {
            std::uint64_t cnt = 0;
            for (std::size_t j = 0; j < c; ++j) {
                if (xd[j * r + i] != yd[j * r + i]) ++cnt;
            }
            cd[i] = static_cast<double>(cnt);
            rd[i] = (c > 0) ? static_cast<double>(cnt) / static_cast<double>(c) : 0.0;
        }
        return {std::move(counts), std::move(ratios)};
    }

    throw Error("symerr: flg must be 'overall', 'column-wise', or 'row-wise'",
                0, 0, "symerr", "", "numkit:symerr:flg");
}

std::pair<Value, Value> symerr(const Value &x, const Value &y,
                               std::pmr::memory_resource *mr)
{
    return symerr(x, y, "overall", mr);
}

} // namespace numkit::comm
