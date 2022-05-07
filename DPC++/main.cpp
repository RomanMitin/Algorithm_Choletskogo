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

constexpr bool PRINT_MATRIX = false;
constexpr int N = 2000;
bool HOME = false;

int main(int argc, char* argv[])
{
	// argv[1] = matrix size
	// argv[2] = choise algorithm 
	//	1 == simple OMP choletsky alg
	//	2 == block OMP choletsky alg
	//  3 == mkl
	//  4 == dpc simle algorithm
	//  5 == dpc block algprithm
	// argv[3] = Block size for choletsky alg
	// argv[4] = row block size for matrix multiplication 
	// argv[5] = collum block size for matrix multiplication 
	//

	srand(0);
	if (!HOME)
	{
		if (argc < 3)
		{
			cerr << "Few parametrs\n";
			exit(-1);
		}
		size_t N = atoi(argv[1]);
		Matrix l(N, N), mat(N, N);
		try
		{
			Get_matrix_form_file(mat, l);
		}
		catch (...)
		{
			cerr << "Creating new matrix\n";
			l = create_Lower_triangle_matrix(N);
			mat = sqr(l);
		}

		int alg = atoi(argv[2]);

		double time = -1.0;
		if (argc < 3)
		{
			start_alg(mat, alg, time);
		}
		else
		{
			int64_t bl1 = atoi(argv[3]);
			int bl2 = atoi(argv[4]);
			int bl3 = atoi(argv[5]);
			start_alg(mat, alg, time, bl1, bl2, bl3);
		}

		cout.precision(4);
		cout << N << " " << time << '\n';

	}
	else
	{
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
		cout << "Cholesky decomposition_dpc++ algorithm time: " << duration.count() << "\n\n";

		if (PRINT_MATRIX) { c.output(); }

		start = std::chrono::high_resolution_clock::now();
		Matrix g = Cholesky_decomposition_dpc_block(a);
		end = std::chrono::high_resolution_clock::now();
		duration = end - start;
		cout << "Cholesky decomposition_dpc++_block algorithm time: " << duration.count() << "\n\n";

		if constexpr (PRINT_MATRIX) { g.output(); }

		auto err = error_rate(l, b);

		cout << "Absolute error in Cholesky decomposition algorithm: " << err.first << '\n';
		cout << "Relative error in Cholesky decomposition algorithm: " << err.second << "%\n\n";

		auto err2 = error_rate(l, d);

		cout << "Absolute error in block Cholesky decomposition algorithm: " << err2.first << '\n';
		cout << "Relative error in block Cholesky decomposition algorithm: " << err2.second << "%\n\n";

		auto err3 = error_rate(l, c);

		cout << "Absolute error in block Cholesky decomposition_dpc++ algorithm: " << err3.first << '\n';
		cout << "Relative error in block Cholesky decomposition_dpc++ algorithm: " << err3.second << "%\n\n";

		auto err4 = error_rate(l, g);

		cout << "Absolute error in block Cholesky decomposition_dpc++_block algorithm: " << err4.first << '\n';
		cout << "Relative error in block Cholesky decomposition_dpc++_block algorithm: " << err4.second << "%\n\n";
	}
	return 0;
}
