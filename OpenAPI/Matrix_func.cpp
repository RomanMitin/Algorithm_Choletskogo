#include<omp.h>
#include"Matrix_func.h"


double randd()
{
	return (rand() % 100 - 50) / 10.0;
}


Matrix create_Lower_triangle_matrix(int64_t size)
{
	Matrix result(size);

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
			}
		}
	}

	return result;
}

Matrix sqr(const Matrix& mat)
{
	Matrix result(mat.size());

#pragma omp parallel for
	for (int64_t i = 0; i < result.size(); i++)
	{
		for (int64_t j = 0; j < result.size(); j++)
		{
			for (int64_t k = 0; k < result.size(); k++)
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
	Matrix l(mat.size());
	l[0][0] = sqrt(mat[0][0]);


#pragma omp parallel for
	for (int64_t i = 1; i < l.size(); i++)
	{
		l[i][0] = mat[i][0] / l[0][0];
	}

	for (int64_t i = 1; i < l.size(); i++)
	{
		type sum1 = 0;
#pragma omp parallel for reduction(+: sum1)
		for (int64_t j = 0; j < i; j++)
		{
			sum1 += l[i][j] * l[i][j];
		}

		l[i][i] = sqrt(mat[i][i] - sum1);


		for (int64_t j = i + 1; j < l.size(); j++)
		{
			type sum = 0;
#pragma omp parallel for reduction(+: sum)
			for (int64_t p = 0; p < i; p++)
			{
				sum += l[i][p] * l[j][p];
			}
			l[j][i] = (mat[j][i] - sum) / l[i][i];
		}
	}
	
	return l;
}


type error_rate(Matrix lower_triangle_exp, Matrix lower_triangle_calcul)
{
	if (lower_triangle_calcul.size() != lower_triangle_exp.size())
		throw "Wrong size in error_rate";

	type errormax = 0.0;
	int im, jm;

	for (int64_t i = 0; i < lower_triangle_calcul.size(); i++)
	{
		for (int64_t j = 0; j < lower_triangle_calcul.size(); j++)
		{
			type error1 = abs(lower_triangle_calcul[i][j] - lower_triangle_exp[i][j]);
			type error2 = abs(lower_triangle_calcul[i][j] + lower_triangle_exp[i][j]);
			if (std::min(error1, error2) > errormax)
			{
				im = i;
				jm = j;
				errormax = std::min(error1, error2);
			}
		}
	}
	//std::cout << im << ' ' << jm<<'\n';
	//std::cout << lower_triangle_calcul[im][jm] << '\n';
	//std::cout << lower_triangle_exp[im][jm] << '\n';
	return errormax;
}