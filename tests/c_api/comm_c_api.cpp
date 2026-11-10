#include <numkit/comm/c_api.h>
#include <gtest/gtest.h>

TEST(NumkitCommCApi, QpskModDemodRoundtrip)
{
    const int64_t base = numkit_debug_live_values();
    numkit_error err{};
    numkit_comm_ctx *ctx = numkit_comm_ctx_create(&err);
    ASSERT_NE(ctx, nullptr) << err.message;

    double syms[4] = {0.0, 1.0, 2.0, 3.0};
    size_t dims[2] = {1, 4};
    numkit_val *vSyms = numkit_val_view(syms, NUMKIT_DTYPE_DOUBLE, dims, 2, &err);
    numkit_val *vM = numkit_val_scalar(4.0);
    const numkit_val *modArgs[2] = {vSyms, vM};
    numkit_val *tx = nullptr;
    ASSERT_EQ(numkit_comm_call(ctx, "pskmod", modArgs, 2, 1, &tx, &err), 0) << err.message;

    const numkit_fn *demodFn = numkit_comm_resolve(ctx, "pskdemod");
    ASSERT_NE(demodFn, nullptr);
    const numkit_val *demodArgs[2] = {tx, vM};
    numkit_val *rx = nullptr;
    ASSERT_EQ(numkit_comm_invoke(ctx, demodFn, demodArgs, 2, 1, &rx, &err), 0) << err.message;
    const auto *rxPtr = static_cast<const double *>(numkit_val_raw_data(rx));
    for (int i = 0; i < 4; ++i) EXPECT_DOUBLE_EQ(rxPtr[i], syms[i]);

    numkit_val_release(rx);
    numkit_val_release(tx);
    numkit_val_release(vM);
    numkit_val_release(vSyms);
    numkit_comm_ctx_destroy(ctx);
    EXPECT_EQ(numkit_debug_live_values(), base);
}
