#pragma once
#include<iostream>
typedef double type;

//class Matrix
//{
//	size_t _size;
//	type* p;
//public:
//	//Matrix() {};
//	Matrix(size_t _size, type val = 0);
//
//	Matrix(const Matrix& second);
//
//	type* operator[](size_t i);
//
//	const type* operator[](size_t i) const;
//
//	Matrix& operator=(const Matrix& second);
//
//	Matrix operator*(const Matrix& second);
//
//	friend std::ostream& operator<<(std::ostream& str, Matrix mat);
//
//	void fillup_rand();
//
//	size_t size() const;
//
//	~Matrix();
//};

class Matrix
{
	size_t _sizer;
	size_t _sizec;
	type* p;
public:
	//Matrix() {};
	Matrix(size_t _sizer, size_t _sizec, type val = 0);

	Matrix(const Matrix& second);

	Matrix(Matrix&& second) noexcept;

	type* operator[](size_t i);

	const type* operator[](size_t i) const;

	Matrix& operator=(const Matrix& second);

	Matrix& operator=(Matrix&& second) noexcept;

	Matrix operator*(const Matrix& second);

	Matrix operator+(const Matrix& second);

	Matrix operator-(const Matrix& second);

	Matrix& transposition();

	Matrix submatrix(size_t row_first, size_t row_last, size_t collumn_first, size_t collumn_last) const;

	void insert_submatrix(const Matrix& submat, size_t row_start, size_t col_start);

	friend std::ostream& operator<<(std::ostream& str,const Matrix& mat);

	void fillup_rand();

	size_t sizec() const;

	size_t sizer() const;

	~Matrix();
};



