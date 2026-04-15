#include <vector>
#include <cmath>

void cpu_reduction_ref(const std::vector<float>& xs, float& y) {
  y = 0.0f;
  for (const float& x : xs) {
    y += x;
  }
}
