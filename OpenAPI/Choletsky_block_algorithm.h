#pragma once
#include"Matrix.h"
#include"Matrix_func.h"

Matrix Cholesky_decomposition_block(const Matrix& mat);

Matrix Cholesky_decomposition_block_with_matrixblock_mult(const Matrix& mat);