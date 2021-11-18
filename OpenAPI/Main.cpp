#include<omp.h>
#include<iostream>
#include<fstream>
#include<time.h>
#include<vector>
#include<string>
#include<chrono>
#include<mkl.h>
#include"Matrix.h"
#include"Matrix_func.h"
#include"Choletsky_block_algorithm.h"
#include"Tests.h"


using namespace std;

const size_t N = 4;

constexpr bool PRINT_MATRIX = 1;
constexpr bool HOME = 1;

int main(int argc,char* argv[]) 
{
	// argv[0] = ������ ������ 
	// argv[1] = ����� �������, ���� 0 == ����� ������� �� ���������
	// argv[2] = ����� �������, ����� ������� ����������
	//	1 == ������� �������� ����������
	//	2 == ������� �������� � ������� ���������� ������
	//  3 == mkl
	// argv[3][0] = ������ ����� ��� �������� ��������� ����������
	// argv[3][1] = ������ ����� �� ������ ��� �������� ��������� ������
	// argv[3][2] = ������ ����� �� ������� ��� �������� ��������� ������
	//
	
	if constexpr (!HOME)
	{
		if (argc < 3)
		{
			cerr << "Too few parameters\n";
			throw std::exception("Too few parameters");
		}
		size_t N = argv[0][0] - '0';
		Matrix l(N, N), mat(N, N);
		try
		{
			Get_matrix_form_file(mat, l);
		}
		catch (...)
		{
			l = create_Lower_triangle_matrix(N);
			mat = sqr(l);
		}

		size_t num_threads = argv[1][0] - '0';
		int alg = argv[2][0] - '0';
		if (num_threads)
		{
			omp_set_num_threads(num_threads);
		}
		
		double time;
		if (argc != 4)
		{
			start_alg(mat, alg, time);
		}
		else
		{
			int64_t bl1 = argv[3][0] - '0';
			int bl2 = argv[3][1] - '0';
			int bl3 = argv[3][2] - '0';
			start_alg(mat, alg, time, bl1, bl2, bl3);
		}

	}
	else
	{
		start_home(PRINT_MATRIX,N);
	}
	
	return 0;
}

