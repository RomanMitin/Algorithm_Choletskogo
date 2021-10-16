#pragma once
#include<iostream>
typedef double type;

class Matrix
{
	size_t _size;
	type* p;
public:
	//Matrix() {};
	Matrix(size_t _size, type val = 0);

	Matrix(const Matrix& second);

	type* operator[](size_t i);

	const type* operator[](size_t i) const;

	Matrix& operator=(const Matrix& second);

	Matrix operator*(const Matrix& second);

	friend std::ostream& operator<<(std::ostream& str, Matrix mat);

	void fillup_rand();

	size_t size() const;

	~Matrix();
};



