#include<omp.h>
#include<iostream>
#include<time.h>
#include<vector>
#include"Matrix.h"
#include"Matrix_func.h"

using namespace std;

const size_t N = 2000;

int main() {

	srand(1);
	
	time_t start = clock();
	Matrix l = create_Lower_triangle_matrix(N);
	//cout << l;
	Matrix a = sqr(l);
	cout << "Time to multiply matrix: ";
	cout << (clock() - start) / 1000.0 << "\n\n";
	//cout << a;
	start = clock();
	Matrix b = Cholesky_decomposition(a);
	//cout << b;
	cout << "Cholesky decomposition algorithm time: " << (clock() - start) / 1000.0 << "\n\n";
	cout << "Error in Cholesky decomposition algorithm: " << error_rate(l, b) << '\n';

	return 0;
}