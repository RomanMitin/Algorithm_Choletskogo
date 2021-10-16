#include<iostream>
#include "Matrix.h"
#ifndef RAND_MAX
#define RAND_MAX 0x7fff
#endif // !RAND_MAX

//typedef double type;

	//Matrix() {};

double randd();

Matrix::Matrix(size_t _size, type val)
	:_size(_size)
{
	p = new type[_size * _size];
	for (size_t i = 0; i < _size * _size; i++)
	{
		p[i] = val;
	}
}

Matrix::Matrix(const Matrix& second)
	:_size(second._size)
{
	p = new type[_size * _size];
	for (size_t i = 0; i < _size * _size; i++)
	{
		p[i] = second.p[i];
	}
}

type* Matrix::operator[](size_t i)
{
	return p + i * _size;
}

const type* Matrix::operator[](size_t i) const
{
	return p + i * _size;
}

Matrix& Matrix::operator=(const Matrix& second)
{
	if (this == &second)
		return *this;
	_size = second._size;
	type* temp = new type[_size * _size];
	delete[] p;
	p = temp;
	for (size_t i = 0; i < _size * _size; i++)
	{
		p[i] = second.p[i];
	}
}

Matrix Matrix::operator*(const Matrix& second)
{
	Matrix result(this->size());
	for (int i = 0; i < result.size(); i++)
	{
		//#pragma omp parallel for
		for (int j = 0; j < result.size(); j++)
		{
			for (size_t k = 0; k < result.size(); k++)
			{
				result[i][j] += (*this)[i][k] * second[k][j];
			}
		}
	}
	return result;
}

std::ostream& operator<<(std::ostream& str, Matrix mat)
{
	for (size_t i = 0; i < mat.size(); i++)
	{
		for (size_t j = 0; j < mat.size(); j++)
		{
			str << mat[i][j] << '\t';
		}
		str << '\n';
	}
	str << '\n';
	return str;
}

void Matrix::fillup_rand()
{
	for (size_t i = 0; i < _size * _size; i++)
	{
		p[i] = randd();
	}
}

size_t Matrix::size() const
{
	return _size;
}

Matrix::~Matrix()
{
	delete[] p;
}
