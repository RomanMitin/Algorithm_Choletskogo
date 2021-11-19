#pragma once
#include"Matrix.h"

void Get_matrix_form_file(Matrix& a, Matrix& l);

Matrix start_alg(Matrix& mat, int alg, double& time, int64_t a = 128, int b = 64, int c = 64);

int to_int(char* s);

void start_home(const bool, size_t);

