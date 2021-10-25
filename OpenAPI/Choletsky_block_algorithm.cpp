#include<cmath>
#include<algorithm>
#include"Choletsky_block_algorithm.h"

Matrix Cholesky_decomposition_block(Matrix &mat)
{
	if (mat.sizec() != mat.sizer())
		throw std::exception();

	/*Matrix A22(0, 0);*/

	int64_t block_size = 256;

	Matrix result(mat.sizer(), mat.sizec());

	/*if (mat.sizer() <= block_size)
	{
		return Cholesky_decomposition(mat);
	}*/


	//Matrix L11 = Cholesky_decomposition(mat.submatrix(0, block_size, 0, block_size));
	//result.insert_submatrix(L11, 0, 0);
	int64_t shift;
	for (shift = 0; shift < int64_t(mat.sizer() - block_size); shift += block_size)
	{
		result[shift][shift] = sqrt(mat[shift][shift]);

		//#pragma omp parallel for
		for (int64_t i = 1; i < block_size; i++)
		{
			result[i + shift][shift] = mat[i + shift][shift] / result[shift][shift];
		}

		for (int64_t i = 1; i < block_size; i++)
		{
			type sum1 = 0;
			//#pragma omp parallel for reduction(+: sum1)
			for (int64_t j = 0; j < i; j++)
			{
				sum1 += result[i+ shift][j + shift] * result[i + shift][j + shift];
			}

			result[i + shift][i + shift] = sqrt(mat[i + shift][i + shift] - sum1);

			//#pragma omp parallel for
			for (int64_t j = i + 1; j < block_size; j++)
			{
				type sum = 0;
				for (int64_t p = 0; p < i; p++)
				{
					sum += result[i + shift][p + shift] * result[j + shift][p + shift];
				}
				result[j + shift][i + shift] = (mat[j + shift][i + shift] - sum) / result[i + shift][i + shift];
			}
		}

		//Matrix L21 = findL21(L11, mat.submatrix(block_size, mat.sizer(), 0, block_size));
		//result.insert_submatrix(L21, block_size, 0);
		for (int64_t i = 0; i < block_size; i++)
		{
			//#pragma omp parallel for
			for (int64_t j = 0; j < mat.sizer() - block_size - shift; j++)
			{
				for (size_t k = 0; k < i; k++)
				{
					result[j + block_size + shift][i + shift] += \
						result[i + shift][k + shift] * result[j + block_size + shift][k + shift];
				}

				result[j + block_size + shift][i + shift] =\
					mat[j + block_size + shift][i + shift] - result[j + block_size + shift][i + shift];

				result[j + block_size + shift][i + shift] /= result[i + shift][i + shift];
			}
		}


		//A22 = mat.submatrix(block_size, mat.sizer(), block_size, mat.sizec()) - sqr(L21);
		for (int64_t i = 0; i < mat.sizer() - block_size - shift; i++)
		{
			for (int64_t j = 0; j < mat.sizec() - block_size - shift; j++)
			{
				type elem = 0;
				for (int64_t k = 0; k < block_size; k++)
				{
					elem += result[i + block_size + shift][k + shift] * result[j + block_size + shift][k + shift];
				}
				mat[i + block_size + shift][j + block_size + shift] -= elem;
			}
		}
	}

	block_size = mat.sizer() - shift;

	result[shift][shift] = sqrt(mat[shift][shift]);

	//#pragma omp parallel for
	for (int64_t i = 1; i < block_size; i++)
	{
		result[i + shift][shift] = mat[i + shift][shift] / result[shift][shift];
	}

	for (int64_t i = 1; i < block_size; i++)
	{
		type sum1 = 0;
		//#pragma omp parallel for reduction(+: sum1)
		for (int64_t j = 0; j < i; j++)
		{
			sum1 += result[i + shift][j + shift] * result[i + shift][j + shift];
		}

		result[i + shift][i + shift] = sqrt(mat[i + shift][i + shift] - sum1);

		//#pragma omp parallel for
		for (int64_t j = i + 1; j < block_size; j++)
		{
			type sum = 0;
			for (int64_t p = 0; p < i; p++)
			{
				sum += result[i + shift][p + shift] * result[j + shift][p + shift];
			}
			result[j + shift][i + shift] = (mat[j + shift][i + shift] - sum) / result[i + shift][i + shift];
		}
	}

	//result.insert_submatrix(Cholesky_decomposition_block(A22), block_size, block_size);

	return result;
}