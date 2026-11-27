// toolboxes/comm/tests/comm_modulation_test.cpp
//
// Coverage for modulation/{psk,fsk_ofdm}.cpp (parity-spec only before this):
// pskmod/pskdemod, dpskmod/dpskdemod, fskmod/fskdemod, ofdmmod/ofdmdemod.
// Modulators are verified by the demod(mod(x)) == x round-trip (exact and
// independent of constellation/ordering conventions), matching psk_smoke /
// fsk_ofdm_smoke. Outputs are compared via sum(abs(a(:)-b(:))) so row/column
// orientation does not matter.

#include "dual_engine_fixture.hpp"

#include <cmath>

using namespace m_test;

class CommModulationTest : public DualEngineTest
{};

TEST_P(CommModulationTest, PskRoundTripUnitMagnitude)
{
    eval("data = [0 1 2 3 0 2 1 3]; s = pskmod(data, 4); out = pskdemod(s, 4); "
         "e = sum(abs(out(:) - data(:)));");
    EXPECT_DOUBLE_EQ(evalScalar("e"), 0.0);
    EXPECT_NEAR(evalScalar("max(abs(abs(s) - 1))"), 0.0, 1e-12);  // QPSK points on unit circle
}

TEST_P(CommModulationTest, PskBinarySymbolOrder)
{
    eval("data = [0 1 2 3 0 2 1 3]; s = pskmod(data, 4, 0, 'bin'); "
         "out = pskdemod(s, 4, 0, 'bin'); e = sum(abs(out(:) - data(:)));");
    EXPECT_DOUBLE_EQ(evalScalar("e"), 0.0);
}

TEST_P(CommModulationTest, DpskRoundTrip)
{
    eval("data = [0 1 2 3 0 2 1 3]; out = dpskdemod(dpskmod(data, 4), 4); "
         "e = sum(abs(out(:) - data(:)));");
    EXPECT_DOUBLE_EQ(evalScalar("e"), 0.0);
}

TEST_P(CommModulationTest, FskmodOutputLength)
{
    eval("y = fskmod([0 1 2 3 0 2 1 3], 4, 100, 10, 2000);");
    EXPECT_EQ(eval("y").numel(), 80u);  // 8 symbols x 10 samples/symbol
}

// fskmod -> fskdemod round-trip. Also guards bugs/comm/fsk_tw_divergence.md: a
// TreeWalker x(:) row->column bug used to break this on TW (fskdemod returns a
// column, the input is a row, so out(:) - data(:) needs both column-ified).
TEST_P(CommModulationTest, FskRoundTrip)
{
    eval("data = [0 1 2 3 0 2 1 3]; out = fskdemod(fskmod(data, 4, 100, 10, 2000), "
         "4, 100, 10, 2000); e = sum(abs(out(:) - data(:)));");
    EXPECT_DOUBLE_EQ(evalScalar("e"), 0.0);
}

TEST_P(CommModulationTest, OfdmRoundTrip)
{
    eval("in = reshape(1:16, 8, 2); y = ofdmmod(in, 8, 2); "
         "out = ofdmdemod(y, 8, 2); e = max(max(abs(out - in)));");
    EXPECT_EQ(eval("y").numel(), 20u);  // (nfft + cplen) x Nsym = 10 x 2
    EXPECT_NEAR(evalScalar("e"), 0.0, 1e-9);
}

TEST_P(CommModulationTest, CommSystemObjectsAndIqImbal)
{
    // comm.BarkerCode
    eval("bc = comm.BarkerCode('Length', 7, 'SamplesPerFrame', 13);"
         "s = step(bc);");
    EXPECT_DOUBLE_EQ(evalScalar("numel(s)"), 13.0);
    EXPECT_DOUBLE_EQ(evalScalar("s(1)"), -1.0);
    EXPECT_DOUBLE_EQ(evalScalar("s(4)"), 1.0);
    EXPECT_DOUBLE_EQ(evalScalar("s(8)"), -1.0);

    // comm.DBPSKModulator + Demodulator
    eval("dm = comm.DBPSKModulator();"
         "dd = comm.DBPSKDemodulator();"
         "bits = [0; 1; 0; 1; 1; 0];"
         "sy = dm(bits);"
         "rx = dd(sy);");
    EXPECT_NEAR(evalScalar("real(sy(1))"), 1.0, 1e-12);
    EXPECT_NEAR(evalScalar("real(sy(2))"), -1.0, 1e-12);
    EXPECT_NEAR(evalScalar("real(sy(4))"), 1.0, 1e-12);
    EXPECT_DOUBLE_EQ(evalScalar("sum(abs(rx - bits))"), 0.0);

    // comm.QPSKModulator + Demodulator (BitInput / BitOutput) + clone()
    eval("qm = comm.QPSKModulator('BitInput', true);"
         "qm2 = clone(qm);"
         "qd = comm.QPSKDemodulator('BitOutput', true);"
         "qb = [0; 0; 0; 1; 1; 0; 1; 1];"
         "qy = qm2(qb);"
         "qrx = qd(qy);");
    EXPECT_NEAR(evalScalar("real(qy(1))"), std::sqrt(0.5), 1e-12);
    EXPECT_NEAR(evalScalar("imag(qy(1))"), std::sqrt(0.5), 1e-12);
    EXPECT_DOUBLE_EQ(evalScalar("sum(abs(qrx - qb))"), 0.0);

    // comm.RaisedCosineTransmitFilter + ReceiveFilter
    eval("tx = comm.RaisedCosineTransmitFilter('OutputSamplesPerSymbol', 4, 'FilterSpanInSymbols', 8);"
         "c = coeffs(tx);"
         "ty = tx([1; 0]);");
    EXPECT_DOUBLE_EQ(evalScalar("numel(c.Numerator)"), 33.0);
    EXPECT_NEAR(evalScalar("norm(c.Numerator)"), 1.0, 1e-12);
    EXPECT_DOUBLE_EQ(evalScalar("numel(ty)"), 8.0);

    // comm.CRCGenerator + CRCDetector
    eval("x = logical([1 0 1 1 0 1 0 1 1 1 0 1]');"
         "cg = comm.CRCGenerator('z^3 + 1');"
         "cd = comm.CRCDetector('z^3 + 1');"
         "cw = cg(x);"
         "cwErr = cw; cwErr(1) = ~cwErr(1);"
         "[txOk, errOk] = cd(cw);"
         "[txBad, errBad] = cd(cwErr);");
    EXPECT_DOUBLE_EQ(evalScalar("numel(cw)"), 15.0);
    EXPECT_DOUBLE_EQ(evalScalar("double(cw(13))"), 1.0);
    EXPECT_DOUBLE_EQ(evalScalar("double(cw(14))"), 1.0);
    EXPECT_DOUBLE_EQ(evalScalar("double(cw(15))"), 0.0);
    EXPECT_DOUBLE_EQ(evalScalar("double(errOk)"), 0.0);
    EXPECT_DOUBLE_EQ(evalScalar("double(errBad)"), 1.0);
    EXPECT_DOUBLE_EQ(evalScalar("sum(abs(double(txOk) - double(x)))"), 0.0);

    // iqimbal exact reference parity
    eval("zi = iqimbal(1 + 1i, 2, 25);");
    EXPECT_NEAR(evalScalar("real(zi)"), 0.9025096608, 1e-4);
    EXPECT_NEAR(evalScalar("imag(zi)"), 0.6273026273, 1e-4);
}

INSTANTIATE_DUAL(CommModulationTest);
