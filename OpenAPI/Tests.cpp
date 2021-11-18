#include<omp.h>
#include<iostream>
#include<fstream>
#include<time.h>
#include<vector>
#include<string>
#include<chrono>
#include"Matrix.h"
#include"Matrix_func.h"
#include"Choletsky_block_algorithm.h"
#include"Tests.h"
#include<mkl.h>

using namespace std;

void Get_matrix_form_file(Matrix& a, Matrix& l)
{
	size_t N = a.sizec();

	string file_name = "Pos_def_Matrix_size_" + to_string(N);
	ifstream file;
	file.open("../Create_positive_def_Matix/" + file_name, ios::binary);
	if (file.is_open())
	{
		file >> l;
		file >> a;
	}
	else
		throw std::exception();
	file.close();
}

Matrix start_alg(Matrix& mat, int alg, double& time, int64_t block1, int block2, int block3)
{
	Matrix result(0,0);

	auto start = std::chrono::high_resolution_clock::now();
	
	switch (alg)
	{
	case 1:
		result = Cholesky_decomposition(mat);
		break;
	case 2:
		result = Cholesky_decomposition_block_with_matrixblock_mult(mat, block1, block2, block3);
		break;
	case 3:
		mklcholetsky_algorithm(mat);
		break;
	default:
		cerr << "Wrong algo in argv[2][0]\n";
		throw std::exception("Wrong algo in argv[2][0]");
		break;
	}
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> duration = end - start;
	time = duration.count();

	if (alg == 3)
	{
		return mat;
	}
	return result;
}




void start_home(const bool PRINT_MATRIX, const size_t N)
{
	srand(10);
	if (PRINT_MATRIX)
	{
		cout << fixed;
		cout.precision(3);
	}

	auto start = std::chrono::high_resolution_clock::now();

	Matrix a(N, N), l(N, N);
	try
	{
		Get_matrix_form_file(a, l);
	}
	catch (...)
	{
		l = create_Lower_triangle_matrix(N);
		a = sqr(l);
	}


	if (PRINT_MATRIX) { cout << l; }

	auto end = std::chrono::high_resolution_clock::now();
	cout << "Time to create matrix: ";
	std::chrono::duration<double> duration = end - start;
	cout << duration.count() << "\n\n";
	if (PRINT_MATRIX) { cout << a; }
	start = std::chrono::high_resolution_clock::now();
	Matrix b = Cholesky_decomposition(a);
	end = std::chrono::high_resolution_clock::now();
	duration = end - start;
	cout << "Cholesky decomposition algorithm time: " << duration.count() << "\n\n";

	start = std::chrono::high_resolution_clock::now();
	Matrix c = Cholesky_decomposition_block_with_matrixblock_mult(a);
	end = std::chrono::high_resolution_clock::now();
	duration = end - start;
	if (PRINT_MATRIX) { cout << b; cout << c; }


	cout << "Block Cholesky decomposition algorithm time: " << duration.count() << "\n\n";

	start = std::chrono::high_resolution_clock::now();
	mklcholetsky_algorithm(a);
	end = std::chrono::high_resolution_clock::now();
	duration = end - start;
	cout << "Mkl time: " << duration.count() << "\n\n";
	auto err = error_rate(l, b);

	cout << "Absolute error in Cholesky decomposition algorithm: " << err.first << '\n';
	cout << "Relative error in Cholesky decomposition algorithm: " << err.second << "%\n\n";

	auto err2 = error_rate(l, c);

	cout << "Absolute error in block Cholesky decomposition algorithm: " << err2.first << '\n';
	cout << "Relative error in block Cholesky decomposition algorithm: " << err2.second << "%\n\n";

	auto err3 = error_rate(l, c);

	cout << "Absolute error in mkl: " << err3.first << '\n';
	cout << "Relative error in mkl: " << err3.second << "%\n";
}

