#include<fstream>
#include<string>
#include<omp.h>
#include"../OpenAPI/Matrix.h"
#include"../OpenAPI/Matrix_func.h"

using namespace std;

int main()
{
	srand(0);
	#pragma omp parallel for schedule(dynamic)
	for (int i = 8000; i > 0; i -= 1000)
	{
		_sleep(omp_get_thread_num() * 50);
		cout << omp_get_thread_num() << " is computing matrix with size " << i << '\n';
		Matrix l = create_Lower_triangle_matrix(i);
		Matrix a = sqr(l);
		ofstream file;
		string file_name = "Pos_def_Matrix_size_" + to_string(i);
		file.open(file_name,ios::binary);
		file << l << a;
		file.close();
		std::cout << "Done: " << i << '\n';
	}
	return 0;
}