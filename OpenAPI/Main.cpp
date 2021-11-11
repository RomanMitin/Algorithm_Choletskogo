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

using namespace std;

const size_t N = 1000;

constexpr bool PRINT_MATRIX = 0;

int main() 
{
	srand(10);
	cout << fixed;
	cout.precision(3);
	
	auto start = std::chrono::high_resolution_clock::now();

	Matrix a(N, N), l(N, N);
	if (N < 1000)
	{
		l = create_Lower_triangle_matrix(N);
		a = sqr(l);
	}
	else
	{
		string file_name = "Pos_def_Matrix_size_" + to_string(N);
		ifstream file;
		file.open("../Create_positive_def_Matix/" + file_name);
		if (file.is_open())
		{
			file >> l;
			file >> a;
		}
		else
			throw std::exception();
		file.close();
	}

	if constexpr (PRINT_MATRIX) { cout << l; }

	auto end = std::chrono::high_resolution_clock::now();
	cout << "Time to create matrix: ";
	std::chrono::duration<double> duration = end - start;
	cout << duration.count() << "\n\n";
	if constexpr (PRINT_MATRIX) { cout << a; }

	start = std::chrono::high_resolution_clock::now();
	Matrix b = Cholesky_decomposition(a);
	end = std::chrono::high_resolution_clock::now();
	duration = end - start;
	cout << "Cholesky decomposition algorithm time: " << duration.count() << "\n\n";

	start = std::chrono::high_resolution_clock::now();
	Matrix c = Cholesky_decomposition_block_with_matrixblock_mult(a);
	end = std::chrono::high_resolution_clock::now();
	duration = end - start;
	if constexpr (PRINT_MATRIX) { cout << b; cout << c; }
	
	cout << "Block Cholesky decomposition algorithm time: " << duration.count() << "\n\n";


	auto err = error_rate(l, b);

	cout << "Absolute error in Cholesky decomposition algorithm: " << err.first << '\n';
	cout << "Relative error in Cholesky decomposition algorithm: " << err.second << "%\n\n";

	auto err2 = error_rate(l, c);
		
	cout << "Absolute error in block Cholesky decomposition algorithm: " << err2.first << '\n';
	cout << "Relative error in block Cholesky decomposition algorithm: " << err2.second << "%\n";

	return 0;
}

