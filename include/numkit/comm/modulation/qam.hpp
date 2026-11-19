/// @file qam.hpp
/// @ingroup group_comm
// toolboxes/comm/include/numkit/comm/modulation/qam.hpp
//
// PAM / rectangular QAM modulators + demodulators, plus modnorm.

#pragma once

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <numkit/value/value.hpp>

namespace numkit::comm {

/// @addtogroup group_comm
/// @{


/// @brief M-ary PAM modulator
/// (`y = pammod(x, M, ini_phase, symbol_order)`).
///
/// Maps a symbol `s ∈ 0..M-1` to a real-valued amplitude in
/// `{-(M-1), …, -1, 1, …, (M-1)}`. Default: Gray ordering,
/// no power normalisation.
///
/// @param x             Integer symbols in `0..M-1`.
/// @param M             Modulation order.
/// @param ini_phase     Phase offset in radians (default 0).
/// @param symbol_order  `"gray"` (default) or `"bin"`.
/// @param mr            Memory resource (nullptr → process default).
/// @return              Real amplitudes, same shape as `x`.
/// @see pamdemod, qammod
Value pammod(const Value &x, int M, keel::f64 ini_phase,
             keel::StringRef symbol_order,
             keel::Allocator *mr = nullptr);

/// @brief M-ary PAM demodulator — nearest-amplitude decision
/// (`x = pamdemod(y, M, ini_phase, symbol_order)`).
///
/// @param y             Received samples.
/// @param M             Modulation order.
/// @param ini_phase     Phase offset used at modulation time.
/// @param symbol_order  `"gray"` (default) or `"bin"`.
/// @param mr            Memory resource (nullptr → process default).
/// @return              Integer symbol indices, same shape as `y`.
/// @see pammod
Value pamdemod(const Value &y, int M, keel::f64 ini_phase,
               keel::StringRef symbol_order,
               keel::Allocator *mr = nullptr);

/// @brief Rectangular M-ary QAM modulator
/// (`y = qammod(x, M, symbol_order, unit_power)`).
///
/// Requires `M = K²` (perfect square); for non-square `M` the
/// implementation falls back to a rectangular grid
/// `floor(√M) × ceil(M / floor(√M))`.
///
/// @param x             Integer symbols in `0..M-1`.
/// @param M             Modulation order.
/// @param symbol_order  `"gray"` (default) or `"bin"`.
/// @param unit_power    If true, scale the constellation to unit
///                      average power.
/// @param mr            Memory resource (nullptr → process default).
/// @return              Complex baseband samples, same shape as `x`.
/// @see qamdemod, pammod, genqammod
Value qammod(const Value &x, int M, keel::StringRef symbol_order,
             bool unit_power,
             keel::Allocator *mr = nullptr);

/// @brief Rectangular M-ary QAM demodulator
/// (`x = qamdemod(y, M, symbol_order, unit_power)`).
///
/// @param y             Received complex samples.
/// @param M             Modulation order.
/// @param symbol_order  `"gray"` (default) or `"bin"`.
/// @param unit_power    If true, the modulator used unit-power
///                      scaling.
/// @param mr            Memory resource (nullptr → process default).
/// @return              Integer symbol indices, same shape as `y`.
/// @see qammod
Value qamdemod(const Value &y, int M, keel::StringRef symbol_order,
               bool unit_power,
               keel::Allocator *mr = nullptr);

/// @brief Constellation normalisation factor
/// (`factor = modnorm(ref, type, target)`).
///
/// Returns the scalar `factor` such that `ref · factor` has the
/// requested average or peak power.
///
/// @param ref     Reference constellation.
/// @param type    `"avpow"` (average power) or `"peakpow"`
///                (peak power).
/// @param target  Desired power level.
/// @param mr      Memory resource (nullptr → process default).
/// @return        Real scalar scaling factor.
/// @throws Error  Unknown `type` or empty `ref`.
/// @see qammod, pammod
Value modnorm(const Value &ref, keel::StringRef type, keel::f64 target,
              keel::Allocator *mr = nullptr);


/// @}
} // namespace numkit::comm
