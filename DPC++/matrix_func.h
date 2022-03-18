#pragma once
#include<utility>
#include"..\\OpenAPI\Matrix.h"

Matrix Cholesky_decomposition(const Matrix& mat) noexcept;

Matrix create_Lower_triangle_matrix(int64_t N);

Matrix sqr(const Matrix&);

std::pair<type, double> error_rate(const Matrix& lower_triangle_exp, const Matrix& lower_triangle_calcul);