#include<iostream>
#include<fstream>
#include<time.h>
#include<vector>
#include<string>
#include<chrono>
#include"alg_chol.h"
#include"matrix_func.h"

using namespace std;

const bool PRINT_MATRIX = true;
const int N = 10;

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

	l = create_Lower_triangle_matrix(N);
	a = sqr(l);

	if (PRINT_MATRIX) { l.output(); }

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
