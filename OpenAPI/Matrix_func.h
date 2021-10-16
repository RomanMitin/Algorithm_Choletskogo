#pragma once
#include"Matrix.h"

Matrix create_Lower_triangle_matrix(int64_t size);

Matrix sqr(const Matrix& mat);

Matrix create_positive_definite_matrix(size_t size);

Matrix Cholesky_decomposition(const Matrix& mat);

type error_rate(Matrix lower_triangle_exp, Matrix lower_triangle_calcul);