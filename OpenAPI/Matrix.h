#pragma once
#include<iostream>
#include<cstring>
#include<iomanip>

typedef float type;

class Matrix
{
public:
	size_t _sizer;
	size_t _sizec;
	type* p;

	//Matrix() {};
	Matrix(size_t _sizer, size_t _sizec, type val = 0);

	Matrix(const Matrix& second);

	Matrix(Matrix&& second) noexcept;

	__forceinline type* operator[](size_t i) noexcept
	{
		return p + i * _sizec;
	}

	__forceinline const type* operator[](size_t i) const noexcept
	{
		return p + i * _sizec;
	}

	Matrix& operator=(const Matrix& second);

	Matrix& operator=(Matrix&& second) noexcept;

	Matrix operator*(const Matrix& second);

	Matrix operator+(const Matrix& second);

	Matrix operator-(const Matrix& second);

	Matrix& transposition();

	Matrix submatrix(size_t row_first, size_t row_last, size_t collumn_first, size_t collumn_last) const;

	void insert_submatrix(const Matrix& submat, size_t row_start, size_t col_start);

	friend std::ostream& operator<<(std::ostream& str,const Matrix& mat);

	friend std::istream& operator>>(std::istream& str, Matrix& mat);

	void output();

	//friend Matrix mklcholetsky_algorithm(Matrix& mat);

	void fillup_rand();

	size_t sizec() const;

	size_t sizer() const;

	~Matrix();
};



