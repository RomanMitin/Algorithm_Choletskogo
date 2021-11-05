#include<omp.h>
#include<iostream>
#include<time.h>
#include<vector>
#include<chrono>
#include"Matrix.h"
#include"Matrix_func.h"
#include"Choletsky_block_algorithm.h"

using namespace std;

const size_t N = 3000;

constexpr bool PRINT_MATRIX = 0;

int main() 
{
	srand(10);
	cout << fixed;
	cout.precision(3);
	
	auto start = std::chrono::high_resolution_clock::now();
	Matrix l = create_Lower_triangle_matrix(N);

	if constexpr (PRINT_MATRIX) { cout << l; }
	Matrix a = sqr(l);
	cout << "Time to multiply matrix: ";
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> duration = end - start;
	cout << duration.count() << "\n\n";
	if constexpr (PRINT_MATRIX) { cout << a; }

	start = std::chrono::high_resolution_clock::now();
	Matrix b = Cholesky_decomposition(a);
	end = std::chrono::high_resolution_clock::now();
	duration = end - start;
	cout << "Cholesky decomposition algorithm time: " << duration.count() << "\n\n";

	start = std::chrono::high_resolution_clock::now();
	Matrix c = Cholesky_decomposition_block2(a);
	end = std::chrono::high_resolution_clock::now();
	duration = end - start;
	if constexpr (PRINT_MATRIX) { cout << b; cout << c; }
	

	auto err = error_rate(l, b);
	
	cout << "Absolute error in Cholesky decomposition algorithm: " << err.first << '\n';
	cout << "Relative error in Cholesky decomposition algorithm: " << err.second << "%\n\n";

	auto err2 = error_rate(l, c);
	cout << "block Cholesky decomposition algorithm time: " << duration.count() << "\n\n";
	cout << "Absolute error in block Cholesky decomposition algorithm: " << err2.first << '\n';
	cout << "Relative error in block Cholesky decomposition algorithm: " << err2.second << "%\n";

	return 0;
}

