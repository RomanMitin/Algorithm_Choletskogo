#include<iostream>
#include<fstream>
#include<time.h>
#include<vector>
#include<string>
#include<chrono>
#include<omp.h>
#include"alg_chol.h"
#include"matrix_func.h"
#include"..\\OpenAPI\Choletsky_block_algorithm.h"
#include"..\\OpenAPI\Tests.h"

using namespace std;

const bool PRINT_MATRIX = false;
const int N = 3000;

int main()
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
		throw 1;
		Get_matrix_form_file(a, l);
	}
	catch (...)
	{
		l = create_Lower_triangle_matrix(N);
		a = sqr(l);
	}
	

	if (PRINT_MATRIX) { l.output(); a.output(); }

	auto end = std::chrono::high_resolution_clock::now();
	cout << "Time to create matrix: ";
	std::chrono::duration<double> duration = end - start;
	cout << duration.count() << "\n\n";

	//if (PRINT_MATRIX) { a.output(); }
	start = std::chrono::high_resolution_clock::now();
	Matrix b = Cholesky_decomposition(a);
	end = std::chrono::high_resolution_clock::now();
	duration = end - start;

	if (PRINT_MATRIX) { b.output(); }
	cout << "Cholesky decomposition algorithm time: " << duration.count() << "\n\n";
	
	start = std::chrono::high_resolution_clock::now();
	Matrix d = Cholesky_decomposition_block_with_matrixblock_mult(a);
	end = std::chrono::high_resolution_clock::now();
	duration = end - start;
	cout << "Cholesky decomposition block_algorithm time: " << duration.count() << "\n\n";

	start = std::chrono::high_resolution_clock::now();
	Matrix c = Cholesky_decomposition_dpc(a);
	end = std::chrono::high_resolution_clock::now();
	duration = end - start;

	if (PRINT_MATRIX) { c.output(); }
	cout << "Cholesky decomposition_dpc++ algorithm time: " << duration.count() << "\n\n";


	auto err = error_rate(l, b);

	cout << "Absolute error in Cholesky decomposition algorithm: " << err.first << '\n';
	cout << "Relative error in Cholesky decomposition algorithm: " << err.second << "%\n\n";

	auto err2 = error_rate(l, c);

	cout << "Absolute error in block Cholesky decomposition_dpc++ algorithm: " << err2.first << '\n';
	cout << "Relative error in block Cholesky decomposition_dpc++ algorithm: " << err2.second << "%\n\n";
}
