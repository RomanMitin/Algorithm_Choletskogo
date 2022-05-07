#include<cmath>
#include<algorithm>
#include<mkl.h>
#include"Choletsky_block_algorithm.h"

#ifdef USE_OMP

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
		//triangular_solver_low(result, block_size, result.sizec() - block_size - shift, shift, block_size);
		#pragma omp parallel for
		for (int64_t j = 0; j < result.sizer() - block_size - shift; j++)
		{
			for (int64_t i = 0; i < block_size; i++)
			{
				type sum = 0;
				#pragma omp simd reduction( + : sum)
				for (size_t k = 0; k < i; k++)
				{
					sum += result[i + shift][k + shift] * result[j + block_size + shift][k + shift];
				}
				result[j + block_size + shift][i + shift] -= sum;

				result[j + block_size + shift][i + shift] /= result[i + shift][i + shift];
			}
		}

		int n1 = result.sizec() - block_size - shift;
		int m1 = block_size;
		int m2 = n1;
		// Нахождение редуцированной матрицы A22 с помощью вычитания из исходной матрицы A22 блока L21 "В квадрате"
	/*	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n1, m2, m1, -1.0f, result[block_size + shift] + shift, result.sizec(), \
			result[block_size + shift] + shift, result.sizec(), 1.0f, result[block_size + shift] + shift + block_size, result.sizec());*/
		
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
						}
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
#endif // USE_OMP
//void triangular_solver_low(Matrix& mat, size_t n, size_t m, int64_t shirt, int64_t block_size)
//{
//	// happy debug
//
//	constexpr size_t column_per_thread = 128;
//	constexpr size_t basic_subsystem_size = 128;
//
//	const size_t threads_block_count = (m + column_per_thread - 1) / column_per_thread;
//	const size_t subsystem_count = (n + basic_subsystem_size - 1) / basic_subsystem_size;
//
//	/*if (b != res) {
//		memcpy(res, b, n * m * sizeof(type));
//	}*/
//
//#pragma omp parallel for
//	for (size_t thread_group = 0; thread_group < threads_block_count; thread_group++)
//	{
//		const size_t column_block_end = min((thread_group + 1) * column_per_thread, m);
//		for (size_t subsystem_begin = 0; subsystem_begin < subsystem_count; subsystem_begin++)
//		{
//			const size_t row_block_end = min((subsystem_begin + 1) * basic_subsystem_size, n);
//			for (size_t i = subsystem_begin * basic_subsystem_size; i < row_block_end; i++)
//			{
//#pragma omp simd
//				for (size_t j = thread_group * column_per_thread; j < column_block_end; j++)
//				{
//					// res[i][j] = b[i][j] / a[i][i];
//					res[i * n + j] /= a[i * n + i];
//				}
//				for (size_t k = i + 1; k < row_block_end; k++)
//				{
//#pragma omp simd
//					for (size_t j = thread_group * column_per_thread; j < column_block_end; j++)
//					{
//						// res[k][j] -= res[i][j] * a[k][i]
//						res[k * n + j] -= res[i * n + j] * a[k * n + i];
//					}
//				}
//			}
//
//			for (size_t additional_block_id = subsystem_begin + 1; additional_block_id < subsystem_count; additional_block_id++)
//			{
//				size_t additional_block_end = min((additional_block_id + 1) * basic_subsystem_size, n);
//				for (size_t i = additional_block_id * basic_subsystem_size; i < additional_block_end; i++)
//				{
//					for (size_t k = subsystem_begin * basic_subsystem_size; k < row_block_end; k++)
//					{
//#pragma omp simd
//						for (size_t j = thread_group * column_per_thread; j < column_block_end; j++)
//						{
//							// res[i][j] -= a[i][k] * b[k][j];
//							res[i * n + j] -= a[i * n + k] * res[k * n + j];
//						}
//					}
//				}
//			}
//		}
//	}
//}