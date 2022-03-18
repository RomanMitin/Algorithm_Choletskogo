#pragma once
#include<algorithm>
#include"Matrix.h"

double randd();

//Matrix mklcholetsky_algorithm(Matrix& mat);

Matrix create_Lower_triangle_matrix(int64_t size);

Matrix sqr(const Matrix& mat);

Matrix create_positive_definite_matrix(size_t size);

//Matrix Cholesky_decomposition(const Matrix& mat) noexcept;

//Matrix Cholesky_decomposition_block(const Matrix& mat);

std::pair<type, double> error_rate(const Matrix& lower_triangle_exp, const Matrix& lower_triangle_calcul);