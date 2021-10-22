#include<omp.h>
#include<cmath>
#include"Matrix_func.h"


double randd()
{
	return (rand() % 100 - 50) / 10.0;
}


Matrix create_Lower_triangle_matrix(int64_t size)
{
	Matrix result(size,size);

	for (int64_t i = 0; i < size; i++)
	{
#pragma omp parallel for
		for (int64_t j = 0; j < size; j++)
		{
			if (i >= j)
			{
				while (result[i][j] == 0)
				{
					result[i][j] = randd();
				}
				if (i == j)
					result[i][j] += 100;
			}
		}
	}

	return result;
}

Matrix sqr(const Matrix& mat)
{
	/*if (mat.sizec() != mat.sizer())
		throw std::exception();*/

	Matrix result(mat.sizer(),mat.sizer());

#pragma omp parallel for
	for (int64_t i = 0; i < result.sizer(); i++)
	{
		for (int64_t j = 0; j < result.sizec(); j++)
		{
			for (int64_t k = 0; k < mat.sizec(); k++)
			{
				result[i][j] += mat[i][k] * mat[j][k];
			}
		}
	}

	return result;
}

Matrix create_positive_definite_matrix(size_t size)
{
	return sqr(create_Lower_triangle_matrix(size));
}

Matrix Cholesky_decomposition(const Matrix& mat)
{
	if (mat.sizec() != mat.sizer())
		throw std::exception();

	Matrix l(mat.sizer(),mat.sizec());
	l[0][0] = sqrt(mat[0][0]);


#pragma omp parallel for
	for (int64_t i = 1; i < l.sizer(); i++)
	{
		l[i][0] = mat[i][0] / l[0][0];
	}

	for (int64_t i = 1; i < l.sizer(); i++)
	{
		type sum1 = 0;
#pragma omp parallel for reduction(+: sum1)
		for (int64_t j = 0; j < i; j++)
		{
			sum1 += l[i][j] * l[i][j];
		}

		l[i][i] = sqrt(mat[i][i] - sum1);

#pragma omp parallel for
		for (int64_t j = i + 1; j < l.sizer(); j++)
		{
			type sum = 0;
			for (int64_t p = 0; p < i; p++)
			{
				sum += l[i][p] * l[j][p];
			}
			l[j][i] = (mat[j][i] - sum) / l[i][i];
		}
	}
	
	return l;
}

template <typename L_M, typename M>
M calc_L(const L_M& a, const M& b, size_t a_size, size_t b_columns)
{
	M ans(a_size, b_columns);

#pragma omp parallel for
	for (int64_t i = 0; i < a_size; i++)
	{
		for (size_t j = 0; j < b_columns; j++)
		{
			for (size_t k = 0; k < i; k++)
			{
				ans[i][j] += a[i][k] * ans[k][j];
			}

			ans[i][j] = b[i][j] - ans[i][j];

			ans[i][j] /= a[i][i];
		}
	}


	return ans;
}

Matrix findL21(const Matrix& L11, Matrix A21)
{
	A21.transposition();
	//Matrix result(L11.sizec(), A21.sizec());
	return calc_L(L11, A21, L11.sizer(), A21.sizec()).transposition();

	//return result.transposition();
}


Matrix Cholesky_decomposition_block(const Matrix& mat)
{
	if (mat.sizec() != mat.sizer())
		throw std::exception();

	Matrix A22(0, 0);
	const static int64_t block_size = 200;
	Matrix result(mat.sizer(), mat.sizec());

	if (mat.sizer() <= block_size)
	{
		return Cholesky_decomposition(mat);
	}

	{
		Matrix L11 = Cholesky_decomposition(mat.submatrix(0, block_size, 0, block_size));
		result.insert_submatrix(L11, 0, 0);
		Matrix L21 = findL21(L11, mat.submatrix(block_size, mat.sizer(), 0, block_size));
		result.insert_submatrix(L21, block_size, 0);
		A22 = mat.submatrix(block_size, mat.sizer(), block_size, mat.sizec()) - sqr(L21);
	}

	result.insert_submatrix(Cholesky_decomposition_block(A22), block_size, block_size);
	return result;
}

std::pair<type,double> error_rate(const Matrix& lower_triangle_exp,const Matrix& lower_triangle_calcul)
{

	if (lower_triangle_calcul.sizer() != lower_triangle_exp.sizer())
		throw "Wrong size in error_rate";
	
	type errormax = 0.0;
	double relative_error = 0;
	//int im, jm;

	for (int64_t i = 0; i < lower_triangle_calcul.sizer(); i++)
	{
		for (int64_t j = 0; j < lower_triangle_calcul.sizer(); j++)
		{
			type error1 = abs(lower_triangle_calcul[i][j] - lower_triangle_exp[i][j]);
			type error2 = abs(lower_triangle_calcul[i][j] + lower_triangle_exp[i][j]);
			if (std::min(error1, error2) > errormax)
			{
				//im = i;
				//jm = j;
				errormax = std::min(error1, error2);
				relative_error = abs(lower_triangle_calcul[i][j] - lower_triangle_exp[i][j]) / lower_triangle_exp[i][j] * 100;
			}
		}
	}
	//std::cout << im << ' ' << jm<<'\n';
	//std::cout << lower_triangle_calcul[im][jm] << '\n';
	//std::cout << lower_triangle_exp[im][jm] << '\n';
	std::pair<type, double> result(errormax, abs(relative_error));
	return result;
}