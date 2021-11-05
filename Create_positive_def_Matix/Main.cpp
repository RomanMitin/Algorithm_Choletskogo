#include<fstream>
#include<string>
#include<omp.h>
#include"../OpenAPI/Matrix.h"
#include"../OpenAPI/Matrix_func.h"

using namespace std;

int main()
{
	srand(0);
	#pragma omp parallel for
	for (int i = 1000; i <= 10000; i+=1000)
	{
		Matrix a = create_positive_definite_matrix(i);
		ofstream file;
		string file_name = "Pos_def_Matrix_size_" + to_string(i);
		file.open(file_name);
		file << a;
		file.close();
		std::cout << "Done: " << i << '\n';
	}
	return 0;
}