#include "posternak_a_crs_mul_complex_matrix/omp/include/ops_omp.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <unordered_map>
#include <vector>

#include "posternak_a_crs_mul_complex_matrix/common/include/common.hpp"

namespace posternak_a_crs_mul_complex_matrix {

PosternakACRSMulComplexMatrixOMP::PosternakACRSMulComplexMatrixOMP(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = CRSMatrix{};
}

bool PosternakACRSMulComplexMatrixOMP::ValidationImpl() {
  const auto &[a, b] = GetInput();
  if (!a.IsValid() || !b.IsValid() || a.cols != b.rows) {
    return false;
  }
  return true;
}

bool PosternakACRSMulComplexMatrixOMP::PreProcessingImpl() {
  const auto &[a, b] = GetInput();
  auto &res = GetOutput();
  
  res.rows = a.rows;
  res.cols = b.cols;
  return true;
}

bool PosternakACRSMulComplexMatrixOMP::RunImpl() {
  const auto &[a, b] = GetInput();
  auto &res = GetOutput();
  
  if (a.values.empty() || b.values.empty()) {
    res.values.clear();
    res.index_col.clear();
    res.index_row.assign(res.rows + 1, 0);
    return true;
  }
  
  std::vector<size_t> row_nnz(res.rows, 0);
  
// каждый поток определяет количество ненулевых элементов в своих строках  
#pragma omp parallel for default(none) shared(a, b, res, row_nnz) schedule(dynamic)
  for (int row = 0; row < res.rows; ++row) {
    std::unordered_map<int, std::complex<double>> row_sum;
    
    for (int idx_a = a.index_row[row]; idx_a < a.index_row[row + 1]; ++idx_a) {
      int col_a = a.index_col[idx_a];
      auto val_a = a.values[idx_a];
      
      for (int idx_b = b.index_row[col_a]; idx_b < b.index_row[col_a + 1]; ++idx_b) {
        int col_b = b.index_col[idx_b];
        auto val_b = b.values[idx_b];
        
        row_sum[col_b] += val_a * val_b;
      }
    }
    
    int nnz_local = 0;
    for (const auto &[col, val] : row_sum) {
      if (std::abs(val) > 1e-12) ++nnz_local;
    }
    row_nnz[row] = nnz_local;
  }

// структурируем результат, чтобы избежать конфликта потоков
#pragma omp single
  {
    for (int i = 1; i < res.rows; ++i) {
      row_nnz[i] += row_nnz[i - 1];
    }
    
    res.values.resize(row_nnz.back());
    res.index_col.resize(row_nnz.back());
    res.index_row.resize(res.rows + 1);
    
    for (int i = 0; i <= res.rows; ++i) {
      res.index_row[i] = (i == 0 ? 0 : row_nnz[i - 1]);
    }
  }

// записываем результат в итоговый массив параллельно
#pragma omp parallel for default(none) shared(a, b, res) schedule(dynamic)
  for (int row = 0; row < res.rows; ++row) {
    std::unordered_map<int, std::complex<double>> row_sum;
    
    for (int idx_a = a.index_row[row]; idx_a < a.index_row[row + 1]; ++idx_a) {
      int col_a = a.index_col[idx_a];
      auto val_a = a.values[idx_a];
      
      for (int idx_b = b.index_row[col_a]; idx_b < b.index_row[col_a + 1]; ++idx_b) {
        int col_b = b.index_col[idx_b];
        auto val_b = b.values[idx_b];
        
        row_sum[col_b] += val_a * val_b;
      }
    }
    
    std::vector<std::pair<int, std::complex<double>>> sorted(row_sum.begin(), row_sum.end());
    std::sort(sorted.begin(), sorted.end(), 
              [](const auto &p1, const auto &p2) { return p1.first < p2.first; });
    
    size_t start = res.index_row[row];
    size_t pos = start;
    
    for (const auto &[col_idx, value] : sorted) {
      if (std::abs(value) > 1e-12) {
        res.values[pos] = value;
        res.index_col[pos] = col_idx;
        ++pos;
      }
    }
  }
  
  return res.IsValid();
}

bool PosternakACRSMulComplexMatrixOMP::PostProcessingImpl() {
  return GetOutput().IsValid();
}

}  // namespace posternak_a_crs_mul_complex_matrix
