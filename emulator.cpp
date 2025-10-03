#include <cmath>
#include <cstdint>

constexpr int LAYERS = 6;
constexpr int HEADS = 5;
constexpr int MLP_SCALE = 4;
constexpr int EMBED_SIZE = 240;
constexpr int HEAD_SIZE = EMBED_SIZE / HEADS;
constexpr int VOCAB_SIZE = 1920;
constexpr int OUTPUT_SIZE = 8;

constexpr int FIXED_POINT_SIZE = 24;
constexpr int FIXED_POINT_MASK = (1 << FIXED_POINT_SIZE) - 1;
constexpr int MATMUL_FIXED_POINT = 18;
constexpr int MATMUL_EXTRA_PRECISION = 4;
constexpr int MATMUL_BIG_MASK = (1 << (FIXED_POINT_SIZE + MATMUL_EXTRA_PRECISION)) - 1;

constexpr int LAYERNORM_CONST = (1LL << 32) / EMBED_SIZE;
constexpr int LAYERNORM_CONST_2 = static_cast<int>((1LL << 27) / std::sqrt(EMBED_SIZE));  // 8663717
constexpr int ATT_CONST = static_cast<int>((1LL << 26) / std::sqrt(EMBED_SIZE));  // 4331858

constexpr int EPS = static_cast<int>(1e-5 * EMBED_SIZE * (1LL << (2 * MATMUL_FIXED_POINT)));
