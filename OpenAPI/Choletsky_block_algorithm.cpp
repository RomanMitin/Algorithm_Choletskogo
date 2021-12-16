#include<cmath>
#include<algorithm>
#include<mkl.h>
#include"Choletsky_block_algorithm.h"

Matrix Cholesky_decomposition_block_with_matrixblock_mult(const Matrix& mat, int64_t block_size, int block_sz_n, int block_sz_m) noexcept
{
	Matrix result(mat);

	int64_t shift;
	for (shift = 0; shift < int64_t(result.sizer() - block_size); shift += block_size)
	{
		// Обычный алгоритм Холетского для блока L11
		result[shift][shift] = sqrt(result[shift][shift]);

		//#pragma omp parallel for
		#pragma omp simd
		for (int64_t i = 1; i < block_size; i++)
		{
			result[i + shift][shift] = result[i + shift][shift] / result[shift][shift];
		}

		for (int64_t i = 1; i < block_size; i++)
		{
			type sum1 = 0;
			#pragma omp simd reduction( + : sum1)
			for (int64_t j = 0; j < i; j++)
			{
				sum1 += result[i + shift][j + shift] * result[i + shift][j + shift];
			}

			result[i + shift][i + shift] = sqrt(result[i + shift][i + shift] - sum1);

			#pragma omp parallel for
			for (int64_t j = i + 1; j < block_size; j++)
			{
				type sum = 0;
				#pragma omp simd reduction( + : sum)
				for (int64_t p = 0; p < i; p++)
				{
					sum += result[i + shift][p + shift] * result[j + shift][p + shift];
				}
				result[j + shift][i + shift] = (result[j + shift][i + shift] - sum) / result[i + shift][i + shift];
			}
		}


		// Решение нижнетреугольной системы для нахождения блока L21
		for (int64_t i = 0; i < block_size; i++)
		{
			#pragma omp parallel for
			for (int64_t j = 0; j < result.sizer() - block_size - shift; j++)
			{
				type tmp = result[j + block_size + shift][i + shift];
				type sum = 0;
				#pragma omp simd reduction( + : sum)
				for (size_t k = 0; k < i; k++)
				{
					sum += result[i + shift][k + shift] * result[j + block_size + shift][k + shift];
				}
				result[j + block_size + shift][i + shift] = tmp - sum;

				result[j + block_size + shift][i + shift] /= result[i + shift][i + shift];
			}
		}

		int n1 = result.sizec() - block_size - shift;
		int m1 = block_size;
		int m2 = n1;
		// Нахождение редуцированной матрицы A22 с помощью вычитания из исходной матрицы A22 блока L21 "В квадрате"
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n1, m2, m1, 1.0, result[block_size + shift] + shift, result.sizec(), \
			result[block_size + shift] + shift, result.sizec(), -1.0, result[block_size + shift] + shift + block_size, result.sizec());
		/*
		#pragma omp parallel for
		for (int ib = 0; ib < n1; ib += block_sz_n)
			for (int kb = 0; kb < m1; kb += block_sz_m)
				for (int jb = 0; jb < m2; jb += block_sz_n)
					for (int i = ib; i < std::min(ib + block_sz_n, n1); ++i)
						for (int j = jb; j < std::min(jb + block_sz_n, m2); ++j)
						{
							type sum2 = 0;
							#pragma omp simd reduction( + : sum2) 
							for (int k = kb; k < std::min(kb + block_sz_m, m1); ++k)
							{
								sum2 += result[i + block_size + shift][k + shift] * result[j + block_size + shift][k + shift];
							}
							result[i + block_size + shift][j + block_size + shift] -= sum2;
						}*/
	}

	block_size = result.sizer() - shift;

	result[shift][shift] = sqrt(result[shift][shift]);

	//Обычный алгоритм Холетского для последнего блока 
	#pragma omp simd
	for (int64_t i = 1; i < block_size; i++)
	{
		result[i + shift][shift] = result[i + shift][shift] / result[shift][shift];
	}

	for (int64_t i = 1; i < block_size; i++)
	{
		type sum1 = 0;
		#pragma omp simd reduction( + : sum1)
		for (int64_t j = 0; j < i; j++)
		{
			sum1 += result[i + shift][j + shift] * result[i + shift][j + shift];
		}

		result[i + shift][i + shift] = sqrt(result[i + shift][i + shift] - sum1);

		//#pragma omp parallel for
		for (int64_t j = i + 1; j < block_size; j++)
		{
			type sum = 0;
			for (int64_t p = 0; p < i; p++)
			{
				sum += result[i + shift][p + shift] * result[j + shift][p + shift];
			}
			result[j + shift][i + shift] = (result[j + shift][i + shift] - sum) / result[i + shift][i + shift];
		}
	}

	#pragma omp parallel for
	for (int i = 0; i < result.sizec() - 1; i++)
	{
		memset(result[i] + i + 1, 0, sizeof(type)* (result.sizer() - i - 1));
	}

	return result;
}