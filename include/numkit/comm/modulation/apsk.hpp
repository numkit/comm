/// @file apsk.hpp
/// @ingroup group_comm
// toolboxes/comm/include/numkit/comm/modulation/apsk.hpp
//
// Amplitude-Phase Shift Keying multi-ring constellation.

#pragma once

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <numkit/value/value.hpp>

namespace numkit::comm {

/// @addtogroup group_comm
/// @{


/// @brief APSK modulator
/// (`y = apskmod(x, M, radii, phaseoffset, mapping)`).
///
/// Indexes integer symbols `x` into a multi-ring PSK constellation
/// built from `M` symbols per ring at the corresponding `radii`.
/// `M` and `radii` must have matching length.
///
/// Default `phaseoffset` is `π / M_k` per ring. Default `mapping`
/// is the identity permutation `0..sum(M)-1`; pass an explicit
/// permutation to reorder. Per-ring Gray default is deferred.
///
/// @param x            Integer symbol indices in `0..sum(M)-1`.
/// @param M            Per-ring symbol count (length R).
/// @param radii        Per-ring radius (length R, same as `M`).
/// @param phaseoffset  Optional per-ring phase offset; `Value::Empty`
///                     uses the default `π / M_k`.
/// @param mapping      Optional symbol-mapping permutation;
///                     `Value::Empty` uses identity.
/// @param mr           Memory resource (nullptr → process default).
/// @return             Complex baseband samples, same shape as `x`.
/// @see apskdemod
Value apskmod(const Value &x, keel::ArrayRef<keel::usize> M,
              keel::ArrayRef<keel::f64> radii,
              const Value &phaseoffset = Value::Empty,
              const Value &mapping = Value::Empty,
              keel::Allocator *mr = nullptr);

/// @brief APSK demodulator
/// (`z = apskdemod(y, M, radii, phaseoffset, mapping)`).
///
/// Inverts @ref apskmod via nearest-constellation-point search.
///
/// @param y            Received complex samples.
/// @param M            Per-ring symbol count.
/// @param radii        Per-ring radius.
/// @param phaseoffset  Optional per-ring phase offset; `Value::Empty`
///                     uses the default `π / M_k`.
/// @param mapping      Optional symbol-mapping permutation;
///                     `Value::Empty` uses identity.
/// @param mr           Memory resource (nullptr → process default).
/// @return             Integer symbol indices, same shape as `y`.
/// @see apskmod
Value apskdemod(const Value &y, keel::ArrayRef<keel::usize> M,
                keel::ArrayRef<keel::f64> radii,
                const Value &phaseoffset = Value::Empty,
                const Value &mapping = Value::Empty,
                keel::Allocator *mr = nullptr);


/// @}
} // namespace numkit::comm
