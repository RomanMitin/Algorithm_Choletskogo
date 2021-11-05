#include<cmath>
#include<algorithm>
#include"Choletsky_block_algorithm.h"

Matrix Cholesky_decomposition_block(const Matrix &mat)
{
	if (mat.sizec() != mat.sizer())
		throw std::exception();


	int64_t block_size = 196;

	Matrix result(mat);

	int64_t shift;
	for (shift = 0; shift < int64_t(result.sizer() - block_size); shift += block_size)
	{
		// Обычный алгоритм Холетского для блока L11
		result[shift][shift] = sqrt(result[shift][shift]);

		//#pragma omp parallel for
		for (int64_t i = 1; i < block_size; i++)
		{
			result[i + shift][shift] = result[i + shift][shift] / result[shift][shift];
		}

		for (int64_t i = 1; i < block_size; i++)
		{
			type sum1 = 0;
			//#pragma omp SIMD reduction(+: sum1)
			for (int64_t j = 0; j < i; j++)
			{
				sum1 += result[i+ shift][j + shift] * result[i + shift][j + shift];
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

	
		// Решение нижнетреугольной системы для нахождения блока L21
		for (int64_t i = 0; i < block_size; i++)
		{
			#pragma omp parallel for
			for (int64_t j = 0; j < result.sizer() - block_size - shift; j++)
			{
				type tmp = result[j + block_size + shift][i + shift];
				result[j + block_size + shift][i + shift] = 0;
				for (size_t k = 0; k < i; k++)
				{
					result[j + block_size + shift][i + shift] += \
						result[i + shift][k + shift] * result[j + block_size + shift][k + shift];
				}

				result[j + block_size + shift][i + shift] =\
					tmp - result[j + block_size + shift][i + shift];

				result[j + block_size + shift][i + shift] /= result[i + shift][i + shift];
			}
		}


	
		// Нахождение редуцированной матрицы A22 с помощью вычитания из исходной матрицы A22 блока L21 "В квадрате"
		#pragma omp parallel for
		for (int64_t i = 0; i < result.sizer() - block_size - shift; i++)
		{
			for (int64_t j = 0; j < result.sizec() - block_size - shift; j++)
			{
				type elem = 0;
				for (int64_t k = 0; k < block_size; k++)
				{
					elem += result[i + block_size + shift][k + shift] * result[j + block_size + shift][k + shift];
				}
				result[i + block_size + shift][j + block_size + shift] -= elem;
			}
		} 
	}

	block_size = result.sizer() - shift;

	result[shift][shift] = sqrt(result[shift][shift]);

	//Обычный алгоритм Холетского для последнего блока 
	for (int64_t i = 1; i < block_size; i++)
	{
		result[i + shift][shift] = result[i + shift][shift] / result[shift][shift];
	}

	for (int64_t i = 1; i < block_size; i++)
	{
		type sum1 = 0;
		
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
		for (int j = i + 1; j < result.sizer(); j++)
		{
			result[i][j] = 0;
		}
	}
	return result;
}

Matrix Cholesky_decomposition_block2(const Matrix& mat)
{
	if (mat.sizec() != mat.sizer())
		throw std::exception();


	int64_t block_size = 196;

	Matrix result(mat);

	int64_t shift;
	for (shift = 0; shift < int64_t(result.sizer() - block_size); shift += block_size)
	{
		// Обычный алгоритм Холетского для блока L11
		result[shift][shift] = sqrt(result[shift][shift]);

		//#pragma omp parallel for
		for (int64_t i = 1; i < block_size; i++)
		{
			result[i + shift][shift] = result[i + shift][shift] / result[shift][shift];
		}

		for (int64_t i = 1; i < block_size; i++)
		{
			type sum1 = 0;
			//#pragma omp SIMD reduction(+: sum1)
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


		// Решение нижнетреугольной системы для нахождения блока L21
		for (int64_t i = 0; i < block_size; i++)
		{
			#pragma omp parallel for
			for (int64_t j = 0; j < result.sizer() - block_size - shift; j++)
			{
				type tmp = result[j + block_size + shift][i + shift];
				result[j + block_size + shift][i + shift] = 0;
				for (size_t k = 0; k < i; k++)
				{
					result[j + block_size + shift][i + shift] += \
						result[i + shift][k + shift] * result[j + block_size + shift][k + shift];
				}

				result[j + block_size + shift][i + shift] = \
					tmp - result[j + block_size + shift][i + shift];

				result[j + block_size + shift][i + shift] /= result[i + shift][i + shift];
			}
		}


		// Нахождение редуцированной матрицы A22 с помощью вычитания из исходной матрицы A22 блока L21 "В квадрате"
		int block_sz_n = 64, block_sz_m = 64;
		int n1 = result.sizec() - block_size - shift;
		int m1 = block_size;
		int n2 = m1, m2 = n1;
		#pragma omp parallel for
		for (int ib = 0; ib < n1; ib += block_sz_n)
			for (int kb = 0; kb < m1; kb += block_sz_m)
				for (int jb = 0; jb < m2; jb += block_sz_n)
					for (int i = ib; i < std::min(ib + block_sz_n, n1); ++i)
						for (int k = kb; k < std::min(kb + block_sz_m, m1); ++k)
							for (int j = jb; j < std::min(jb + block_sz_n, m2); ++j)
								result[i + block_size + shift][j + block_size + shift] -= \
								result[i + block_size + shift][k + shift] * result[j + block_size + shift][k + shift];
	}

	block_size = result.sizer() - shift;

	result[shift][shift] = sqrt(result[shift][shift]);

	//Обычный алгоритм Холетского для последнего блока 
	for (int64_t i = 1; i < block_size; i++)
	{
		result[i + shift][shift] = result[i + shift][shift] / result[shift][shift];
	}

	for (int64_t i = 1; i < block_size; i++)
	{
		type sum1 = 0;

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
		for (int j = i + 1; j < result.sizer(); j++)
		{
			result[i][j] = 0;
		}
	}

	return result;
}