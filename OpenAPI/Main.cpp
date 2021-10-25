#include<omp.h>
#include<iostream>
#include<time.h>
#include<vector>
#include"Matrix.h"
#include"Matrix_func.h"
#include"Choletsky_block_algorithm.h"

using namespace std;

const size_t N = 4000;

constexpr bool PRINT_MATRIX = 0;

int main() 
{
	srand(10);
	
	time_t start = clock();
	Matrix l = create_Lower_triangle_matrix(N);

	if constexpr (PRINT_MATRIX) { cout << l; }
	Matrix a = sqr(l);
	cout << "Time to multiply matrix: ";
	cout << (clock() - start) / double(CLOCKS_PER_SEC) << "\n\n";
	if constexpr (PRINT_MATRIX) { cout << a; }

	start = clock();
	Matrix b = Cholesky_decomposition(a);
	cout << "Cholesky decomposition algorithm time: " << (clock() - start) / double(CLOCKS_PER_SEC) << "\n\n";

	start = clock();
	Matrix c = Cholesky_decomposition_block(a);
	time_t finish = clock();
	if constexpr (PRINT_MATRIX) { cout << b; cout << c; }
	

	auto err = error_rate(l, b);
	
	cout << "Absolute error in Cholesky decomposition algorithm: " << err.first << '\n';
	cout << "Relative error in Cholesky decomposition algorithm: " << err.second << "%\n\n";

	auto err2 = error_rate(l, c);
	cout << "block Cholesky decomposition algorithm time: " << (finish - start) / 1000.0 << "\n\n";
	cout << "Absolute error in block Cholesky decomposition algorithm: " << err2.first << '\n';
	cout << "Relative error in block Cholesky decomposition algorithm: " << err2.second << "%\n";

	return 0;
}

