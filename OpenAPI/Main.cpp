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

const size_t N = 10000;

constexpr bool PRINT_MATRIX = 0;
constexpr bool HOME = 1;

int main(int argc,char* argv[]) 
{
	// argv[1] = ������ ������ 
	// argv[2] = ����� �������, ���� 0 == ����� ������� �� ���������
	// argv[3] = ����� �������, ����� ������� ����������
	//	1 == ������� �������� ����������
	//	2 == ������� �������� � ������� ���������� ������
	//  3 == mkl
	// argv[4] = ������ ����� ��� �������� ��������� ����������
	// argv[5] = ������ ����� �� ������ ��� �������� ��������� ������
	// argv[6] = ������ ����� �� ������� ��� �������� ��������� ������
	//
	
	if (!HOME)
	{
		if (argc < 4)
		{
			cerr << "Too few parameters\n";
			throw std::exception();
		}
		size_t N = atoi(argv[1]);
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

		size_t num_threads = atoi(argv[2]);
		int alg = atoi(argv[3]);
		if (num_threads)
		{
			omp_set_num_threads(num_threads);
		}
		
		double time;
		if (argc < 5)
		{
			start_alg(mat, alg, time);
		}
		else
		{
			int64_t bl1 = atoi(argv[4]);
			int bl2 = atoi(argv[5]);
			int bl3 = atoi(argv[6]);
			start_alg(mat, alg, time, bl1, bl2, bl3);
		}

		string s = "alg_" + to_string(alg) + ".txt";
		ofstream out;
		out.open(s, ios::app);
		out.precision(4);
		out << N << " " << num_threads << " " << time << '\n';
		out.close();
	}
	else
	{
		start_home(PRINT_MATRIX, N);
	}
	
	return 0;
}

