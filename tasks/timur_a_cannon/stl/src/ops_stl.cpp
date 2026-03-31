#include "timur_a_cannon/stl/include/ops_stl.hpp"

#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

namespace timur_a_cannon {

TimurACannonMatrixMultiplicationSTL::TimurACannonMatrixMultiplicationSTL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool TimurACannonMatrixMultiplicationSTL::ValidationImpl() {
  const auto &input = GetInput();
  int b_size = std::get<0>(input);
  const auto &mat_a = std::get<1>(input);
  const auto &mat_b = std::get<2>(input);

  if (b_size <= 0 || mat_a.empty() || mat_b.empty()) {
    return false;
  }
  size_t n = mat_a.size();
  return mat_a[0].size() == n && mat_b.size() == n && (n % static_cast<size_t>(b_size) == 0);
}

bool TimurACannonMatrixMultiplicationSTL::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

void TimurACannonMatrixMultiplicationSTL::BlockMultiplyAccumulate(const std::vector<std::vector<double>> &a,
                                                                  const std::vector<std::vector<double>> &b,
                                                                  std::vector<std::vector<double>> &c, int b_size) {
  for (int i = 0; i < b_size; ++i) {
    for (int k = 0; k < b_size; ++k) {
      double temp = a[i][k];
      for (int j = 0; j < b_size; ++j) {
        c[i][j] += temp * b[k][j];
      }
    }
  }
}

void TimurACannonMatrixMultiplicationSTL::DistributeData(
    const std::vector<std::vector<double>> &src_a, const std::vector<std::vector<double>> &src_b,
    std::vector<std::vector<std::vector<std::vector<double>>>> &bl_a,
    std::vector<std::vector<std::vector<std::vector<double>>>> &bl_b, int b_size, int grid_sz) {
  for (int i = 0; i < grid_sz; ++i) {
    for (int j = 0; j < grid_sz; ++j) {
      int shift = (i + j) % grid_sz;
      for (int row = 0; row < b_size; ++row) {
        // Использование std::copy — это "STL-way" вместо ручных циклов
        std::copy(src_a[(i * b_size) + row].begin() + (shift * b_size),
                  src_a[(i * b_size) + row].begin() + (shift * b_size) + b_size, bl_a[i][j][row].begin());

        std::copy(src_b[(shift * b_size) + row].begin() + (j * b_size),
                  src_b[(shift * b_size) + row].begin() + (j * b_size) + b_size, bl_b[i][j][row].begin());
      }
    }
  }
}

void TimurACannonMatrixMultiplicationSTL::RotateBlocksA(
    std::vector<std::vector<std::vector<std::vector<double>>>> &blocks, int grid_sz) {
  for (int i = 0; i < grid_sz; ++i) {
    // Циклический сдвиг строки блоков влево на 1
    std::rotate(blocks[i].begin(), blocks[i].begin() + 1, blocks[i].end());
  }
}

void TimurACannonMatrixMultiplicationSTL::RotateBlocksB(
    std::vector<std::vector<std::vector<std::vector<double>>>> &blocks, int grid_sz) {
  for (int j = 0; j < grid_sz; ++j) {
    auto first_block = std::move(blocks[0][j]);
    for (int i = 0; i < grid_sz - 1; ++i) {
      blocks[i][j] = std::move(blocks[i + 1][j]);
    }
    blocks[grid_sz - 1][j] = std::move(first_block);
  }
}

bool TimurACannonMatrixMultiplicationSTL::RunImpl() {
  const auto &input = GetInput();
  int b_size = std::get<0>(input);
  int n = static_cast<int>(std::get<1>(input).size());
  int grid_sz = n / b_size;

  using Matrix = std::vector<std::vector<double>>;
  using BlockGrid = std::vector<std::vector<Matrix>>;

  BlockGrid bl_a(grid_sz, std::vector<Matrix>(grid_sz, Matrix(b_size, std::vector<double>(b_size))));
  BlockGrid bl_b(grid_sz, std::vector<Matrix>(grid_sz, Matrix(b_size, std::vector<double>(b_size))));
  BlockGrid bl_c(grid_sz, std::vector<Matrix>(grid_sz, Matrix(b_size, std::vector<double>(b_size, 0.0))));

  DistributeData(std::get<1>(input), std::get<2>(input), bl_a, bl_b, b_size, grid_sz);

  for (int step = 0; step < grid_sz; ++step) {
    for (int i = 0; i < grid_sz; ++i) {
      for (int j = 0; j < grid_sz; ++j) {
        BlockMultiplyAccumulate(bl_a[i][j], bl_b[i][j], bl_c[i][j], b_size);
      }
    }
    if (step < grid_sz - 1) {
      RotateBlocksA(bl_a, grid_sz);
      RotateBlocksB(bl_b, grid_sz);
    }
  }

  Matrix res_mat(n, std::vector<double>(n));
  for (int i = 0; i < grid_sz; ++i) {
    for (int j = 0; j < grid_sz; ++j) {
      for (int row = 0; row < b_size; ++row) {
        std::copy(bl_c[i][j][row].begin(), bl_c[i][j][row].end(), res_mat[(i * b_size) + row].begin() + (j * b_size));
      }
    }
  }

  GetOutput() = std::move(res_mat);
  return true;
}

bool TimurACannonMatrixMultiplicationSTL::PostProcessingImpl() {
  return true;
}

}  // namespace timur_a_cannon
