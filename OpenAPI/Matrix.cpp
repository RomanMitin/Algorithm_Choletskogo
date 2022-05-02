#include<iostream>
#include<cmath>
#include<omp.h>
#include "Matrix.h"
#include"Matrix_func.h"
#ifndef RAND_MAX
#define RAND_MAX 0x7fff
#endif // !RAND_MAX

inline double randd()
{
	return (rand() % 100 - 50) / 10.0;
}

Matrix::Matrix(size_t _sizer, size_t _sizec, type val)
	:_sizer(_sizer), _sizec(_sizec)
{
	p = new type[_sizer * _sizec];


	memset(p, val, _sizer * _sizec * sizeof(type));
	/*int size = (_sizec * _sizer * sizeof(type));
	int n;
	#pragma omp parallel
	{
		n = omp_get_num_threads();
		int tmpsize = size / n;
		memset(p + tmpsize * omp_get_thread_num(), val, tmpsize);
	}
	this->output();
	memset(p + size * n, val, size % n);
	this->output();*/
}

Matrix::Matrix(const Matrix& second)
	:_sizer(second._sizer),_sizec(second._sizec)
{
	p = new type[_sizec * _sizer];
	//#pragma omp parallel for
	for (size_t i = 0; i < _sizec * _sizer; i++)
	{
		p[i] = second.p[i];
	}
}

Matrix::Matrix(Matrix&& second) noexcept
	:_sizer(second._sizer), _sizec(second._sizec), p(second.p)
{
	second.p = nullptr;
	second._sizec = 0;
	second._sizer = 0;
}

type* Matrix::operator[](size_t i) noexcept
{
	return p + i * _sizec;
}

const type* Matrix::operator[](size_t i) const noexcept
{
	return p + i * _sizec;
}

Matrix& Matrix::operator=(const Matrix& second)
{
	if (this == &second)
		return *this;

	_sizer = second._sizer;
	_sizec = second._sizec;

	type* temp = new type[_sizer * _sizec];
	delete[] p;
	p = temp;
	for (size_t i = 0; i < _sizec * _sizer; i++)
	{
		p[i] = second.p[i];
	}

	return *this;
}

Matrix& Matrix::operator=(Matrix&& second) noexcept
{
	if (this == &second)
		return *this;

	delete[] p;
	
	_sizer = second._sizer;
	_sizec = second._sizec;
	p = second.p;

	second.p = nullptr;
	second._sizec = 0;
	second._sizer = 0;

	return *this;
}

Matrix Matrix::operator*(const Matrix& second)
{
	if (second.sizer() != this->sizec())
		throw std::exception();

	Matrix result(this->sizer(), second.sizec());

#pragma omp parallel for
	for (int i = 0; i < this->sizer(); i++)
	{	
		for (int j = 0; j < this->sizec(); j++)
		{
			for (size_t k = 0; k < second.sizer(); k++)
			{
				result[i][j] += (*this)[i][k] * second[k][j];
			}
		}
	}
	return result;
}

Matrix Matrix::operator+(const Matrix& second)
{
	if (this->sizer() != second.sizer() || this->sizec() != second.sizec())
		throw std::exception();

	Matrix result(this->sizer(), this->sizec());

#pragma omp parallel for
	for (int64_t i = 0; i < result.sizer(); i++)
	{
		for (int64_t j = 0; j < result.sizec(); j++)
		{
			result[i][j] = (*this)[i][j] + second[i][j];
		}
	}

	return result;

}

Matrix Matrix::operator-(const Matrix& second)
{
	if (this->sizer() != second.sizer() || this->sizec() != second.sizec())
		throw std::exception();

	Matrix result(this->sizer(), this->sizec());

#pragma omp parallel for
	for (int64_t i = 0; i < result.sizer(); i++)
	{
		for (int64_t j = 0; j < result.sizec(); j++)
		{
			result[i][j] = (*this)[i][j] - second[i][j];
		}
	}

	return result;
}

Matrix& Matrix::transposition()
{
	Matrix tmp(this->sizec(), this->sizer());

#pragma omp parallel for
	for (int64_t i = 0; i < this->sizer(); i++)
	{
		for (size_t j = 0; j < this->sizec(); j++)
		{
			tmp[j][i] = (*this)[i][j];
		}
	}

	*this = std::move(tmp);
	return *this;
}

Matrix Matrix::submatrix(size_t row_first, size_t row_last, size_t collumn_first, size_t collumn_last) const
{
	if (row_last > this->sizer() || collumn_last > this->sizec() || row_first >= row_last || collumn_first >= collumn_last)
	{
		throw std::exception();
	}

	Matrix result(row_last - row_first, collumn_last - collumn_first);

#pragma omp parallel for
	for (int64_t i = 0; i < result.sizer(); i++)
	{
		for (int64_t j = 0; j < result.sizec(); j++)
		{
			result[i][j] = (*this)[i + row_first][j + collumn_first];
		}
	}

	return result;
}

void Matrix::insert_submatrix(const Matrix& submat, size_t row_start, size_t col_start)
{
	if (submat.sizer() + row_start > this->sizer() || submat.sizec() + col_start > this->sizec())
	{
		throw std::exception();
	}

#pragma omp parallel for
	for (int64_t i = 0; i < submat.sizer(); i++)
	{
		for (int64_t j = 0; j < submat.sizec(); j++)
		{
			(*this)[i + row_start][j + col_start] = submat[i][j];
		}
	}
}

std::ostream& operator<<(std::ostream& str,const Matrix& mat)
{
	str.write(reinterpret_cast<char*>(mat.p), sizeof(type) * mat.sizec() * mat.sizer());
	//for (size_t i = 0; i < mat.sizer(); i++)
	//{
	//	for (size_t j = 0; j < mat.sizec(); j++)
	//	{
	//		str << mat[i][j];// << ' ';
	//	}
	//	//str << '\n';
	//}
	////str << '\n';
	return str;
}

std::istream& operator>>(std::istream& str, Matrix& mat)
{
	str.read(reinterpret_cast<char*>(mat.p), sizeof(type) * mat.sizec() * mat.sizer());
	/*for (size_t i = 0; i < mat.sizer(); i++)
	{
		for (size_t j = 0; j < mat.sizec(); j++)
		{
			str >> mat[i][j];
		}
	}*/
	return str;
}

void Matrix::output()
{
	for (size_t i = 0; i < this->_sizer; i++)
	{
		for (size_t j = 0; j < this->_sizec; j++)
		{
			std::cout << (*this)[i][j] << '\t';
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

void Matrix::fillup_rand()
{
	for (size_t i = 0; i < _sizec * _sizer; i++)
	{
		p[i] = randd();
	}
}

 size_t Matrix::sizer() const
{
 	return _sizer;
}

 size_t Matrix::sizec() const
{
	return _sizec;
}

Matrix::~Matrix()
{
	delete[] p;
}
