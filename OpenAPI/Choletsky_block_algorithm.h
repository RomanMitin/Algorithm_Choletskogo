#pragma once
#include"Matrix.h"
#include"Matrix_func.h"

Matrix Cholesky_decomposition_block(const Matrix& mat) noexcept;

Matrix Cholesky_decomposition_block_with_matrixblock_mult(const Matrix& mat, int64_t block_size = 128, int block_sz_n = 64, int block_sz_m = 64) noexcept;