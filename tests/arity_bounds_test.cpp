// toolboxes/comm/tests/arity_bounds_test.cpp
//
// Arity-contract guards for the comm domain (argcheck migration,
// 2026-09-17). Bounded contracts probed against R2025b get a max+1 call
// that must fail with nk:<fn>:maxnargin; checkNargin is the first reg
// statement, so the dummy values never reach compute code. The four
// reference-absent names (rayleighchan / ricianchan / scrambler /
// descrambler — System objects in R2025b) guard the nk-only contract.
// A second fixture pins the newly implemented R2025b short forms.

#include <numkit/bundle/standard_engine.hpp>
#include <numkit/core/engine.hpp>
#include <numkit/value/error.hpp>
#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace {

struct Case
{
    const char *fn;
    int maxArgs;
};

const std::vector<Case> kBounds = {
    {"amdemod", 7},
    {"ammod", 5},
    {"arithdeco", 3},
    {"arithenco", 2},
    {"awgn", 5},
    {"berconfint", 3},
    {"bi2de", 3},
    {"bit2int", 5},
    {"biterr", 4},
    {"bsc", 3},
    {"compand", 4},
    {"convenc", 4},
    {"cyclgen", 3},
    {"cyclpoly", 3},
    {"decode", 6},
    {"de2bi", 4},
    {"descrambler", 3},
    {"dpcmdeco", 3},
    {"dpcmenco", 4},
    {"dpcmopt", 3},
    {"encode", 5},
    {"fmmod", 5},
    {"fmdemod", 5},
    {"fskdemod", 6},
    {"fskmod", 7},
    {"gaussdesign", 3},
    {"gen2par", 1},
    {"hammgen", 2},
    {"huffmandeco", 2},
    {"huffmandict", 4},
    {"huffmanenco", 2},
    {"int2bit", 3},
    {"intdump", 2},
    {"istrellis", 1},
    {"lloyds", 4},
    {"marcumq", 3},
    {"mil188qammod", 10},
    {"modnorm", 3},
    {"mskdemod", 5},
    {"mskmod", 4},
    {"noisebw", 4},
    {"ofdmdemod", 8},
    {"ofdmmod", 8},
    {"pamdemod", 4},
    {"pammod", 4},
    {"pmmod", 5},
    {"pmdemod", 5},
    {"poly2trellis", 3},
    {"pskmod", 10},
    {"qammod", 11},
    {"qfunc", 1},
    {"qfuncinv", 1},
    {"quantiz", 3},
    {"randerr", 4},
    {"randsrc", 4},
    {"rayleighchan", 1},
    {"rcosdesign", 4},
    {"rectpulse", 2},
    {"ricianchan", 2},
    {"scrambler", 3},
    {"ssbdemod", 6},
    {"ssbmod", 5},
    {"symerr", 3},
    {"syndtable", 1},
    {"vec2mat", 3},
    {"vitdec", 11},
    {"wgn", 7},
};

std::string callExpr(const Case &c)
{
    std::string s = c.fn;
    s += '(';
    for (int i = 0; i <= c.maxArgs; ++i) {
        if (i)
            s += ", ";
        s += '1';
    }
    s += ')';
    return s;
}

} // namespace

class CommArityBoundsTest : public ::testing::TestWithParam<Case>
{
public:
    numkit::StandardEngine engine;
};

TEST_P(CommArityBoundsTest, RejectsMaxPlusOne)
{
    const Case &c = GetParam();
    try {
        engine.eval(callExpr(c));
        FAIL() << "expected maxnargin error: " << callExpr(c);
    } catch (const numkit::Error &e) {
        EXPECT_EQ(e.identifier(), std::string("numkit:") + c.fn + ":maxnargin")
            << callExpr(c) << "\n  got: " << e.what();
    }
}

INSTANTIATE_TEST_SUITE_P(CommArityBounds, CommArityBoundsTest,
                         ::testing::ValuesIn(kBounds));

// ─── R2025b short forms implemented with the migration ────────────────

class CommArityFormsTest : public ::testing::Test
{
public:
    numkit::StandardEngine engine;
    numkit::Value eval(const std::string &c) { return engine.eval(c); }
    double evalScalar(const std::string &c) { return eval(c).toScalar(); }
    void expectError(const std::string &code, const std::string &id)
    {
        try {
            engine.eval(code);
            FAIL() << "expected error " << id << " for: " << code;
        } catch (const numkit::Error &e) {
            EXPECT_EQ(e.identifier(), id) << code << "\n  got: " << e.what();
        }
    }
};

TEST_F(CommArityFormsTest, RandsrcZeroAndOneArg)
{
    EXPECT_EQ(eval("numel(randsrc())").toScalar(), 1.0);
    eval("r = randsrc(3);");
    eval("assert(size(r) == [3 3]);");
}

TEST_F(CommArityFormsTest, RanderrOneArg)
{
    eval("e = randerr(4);");
    eval("assert(size(e) == [4 4]);");
    eval("assert(all(sum(e, 2) == 1));");
}

TEST_F(CommArityFormsTest, GaussdesignDefaults)
{
    eval("g1 = gaussdesign(0.3); g2 = gaussdesign(0.3, 3); g3 = gaussdesign(0.3, 3, 2);");
    eval("assert(isequal(g1, g2) && isequal(g1, g3));");
    EXPECT_EQ(evalScalar("numel(g1)"), 7.0);
}

TEST_F(CommArityFormsTest, CompandDefaultMethod)
{
    eval("x = 1:5;");
    eval("a = compand(x, 255, 5); b = compand(x, 255, 5, 'mu/compressor');");
    eval("assert(isequal(a, b));");
}

TEST_F(CommArityFormsTest, FskFsOptional)
{
    eval("y1 = fskmod(0:3, 4, 0.1, 8); y2 = fskmod(0:3, 4, 0.1, 8, 1);");
    eval("assert(isequal(y1, y2));");
    eval("d = fskdemod(y1, 4, 0.1, 8);");
    // nk's fsk kernels are column-oriented; compare shape-agnostically.
    eval("assert(isequal(double(d(:)), (0:3)'));");
}

TEST_F(CommArityFormsTest, SymerrFlagDeferred)
{
    expectError("symerr([1 0; 0 1], [0 1; 1 0], 'invalid');",
                "numkit:symerr:flg");
}

TEST_F(CommArityFormsTest, ConvertSNRNeedsMode)
{
    // R2025b rejects the 1-arg form (mode required).
    expectError("convertSNR(10);", "numkit:convertSNR:nargin");
}

TEST_F(CommArityFormsTest, VitdecNeedsFiveArgs)
{
    // R2025b requires (code, trellis, tblen, opmode, dectype).
    expectError("t = poly2trellis(3, [7 5]); vitdec(zeros(1, 8), t, 5);",
                "numkit:vitdec:nargin");
    expectError("t = poly2trellis(3, [7 5]); vitdec(zeros(1, 8), t, 5, 'trunc');",
                "numkit:vitdec:nargin");
}

TEST_F(CommArityFormsTest, Mil188MSetMatchesReference)
{
    eval("y = mil188qammod(0:15, 256);");
    EXPECT_EQ(evalScalar("numel(y)"), 16.0);
    expectError("mil188qammod(0, 128);", "numkit:mil188qammod:MNotChoice");
}
