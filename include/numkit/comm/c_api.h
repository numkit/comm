/* include/numkit/comm/c_api.h
 *
 * Self-contained C-ABI for the `numkit/comm` toolbox.
 *
 * Supports both:
 *   1. Pre-resolved O(1) dispatch (`numkit_comm_resolve` + `numkit_comm_invoke`)
 *      with zero string hashing on the hot path.
 *   2. Dynamic name/method dispatch (`numkit_comm_call`) for MATLAB overloads,
 *      Name-Value pairs, and toolbox object methods.
 */
#ifndef NUMKIT_COMM_C_API_H
#define NUMKIT_COMM_C_API_H

#include <numkit/value/c_api.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
#  if defined(NUMKIT_PLUGIN_BUILD)
#    define NUMKIT_COMM_API __declspec(dllexport)
#  elif defined(NUMKIT_COMM_STATIC)
#    define NUMKIT_COMM_API
#  else
#    define NUMKIT_COMM_API __declspec(dllimport)
#  endif
#else
#  if defined(__GNUC__) && __GNUC__ >= 4
#    define NUMKIT_COMM_API __attribute__((visibility("default")))
#  else
#    define NUMKIT_COMM_API
#  endif
#endif

/* Opaque isolated `comm` toolbox context */
typedef struct numkit_comm_ctx_s numkit_comm_ctx;

/* Create an isolated `comm` toolbox context (engine-free, zero interpreter overhead). */
NUMKIT_COMM_API numkit_comm_ctx *numkit_comm_ctx_create(numkit_error *err);

/* Destroy `comm` toolbox context and release its resources. */
NUMKIT_COMM_API void numkit_comm_ctx_destroy(numkit_comm_ctx *ctx);

/* Seed the `comm` context's random number stream (MATLAB `rng(seed)` semantics). */
NUMKIT_COMM_API void numkit_comm_seed_rng(numkit_comm_ctx *ctx, uint64_t seed);

/* Return 1 if function `name` is registered in `comm`, 0 otherwise. */
NUMKIT_COMM_API int numkit_comm_has_function(const numkit_comm_ctx *ctx, const char *name);

/* Register a host-language callback (`numkit_host_fn`) into this context. */
NUMKIT_COMM_API int numkit_comm_register_fn(numkit_comm_ctx *ctx,
                                            const char *name,
                                            numkit_host_fn fn,
                                            void *user_data,
                                            numkit_error *err);

/* Resolve function `name` once into an O(1) function pointer (`NULL` if not found). */
NUMKIT_COMM_API const numkit_fn *numkit_comm_resolve(const numkit_comm_ctx *ctx,
                                                     const char *name);

/* Invoke a pre-resolved function pointer directly with zero lookup overhead. */
NUMKIT_COMM_API int numkit_comm_invoke(numkit_comm_ctx *ctx,
                                       const numkit_fn *fn,
                                       const numkit_val *const *args,
                                       size_t nargs,
                                       size_t nargout,
                                       numkit_val **outs,
                                       numkit_error *err);

/* Dynamically invoke `comm` function or object method `name`. */
NUMKIT_COMM_API int numkit_comm_call(numkit_comm_ctx *ctx,
                                     const char *name,
                                     const numkit_val *const *args,
                                     size_t nargs,
                                     size_t nargout,
                                     numkit_val **outs,
                                     numkit_error *err);

/* Format a value or `comm` class object (e.g. `sym`, `tf`, `ss`) to UTF-8 text. */
NUMKIT_COMM_API size_t numkit_comm_format(const numkit_comm_ctx *ctx,
                                          const numkit_val *v,
                                          char *buf,
                                          size_t buf_cap);

#ifdef __cplusplus
}
#endif

#endif /* NUMKIT_COMM_C_API_H */
