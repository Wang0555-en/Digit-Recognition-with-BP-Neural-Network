#ifndef MATRIX_H
#define MATRIX_H
#include <iostream>
#include <vector>
using namespace std;
class Matrix
{
	vector<vector<double>> data;
	int rows;
	int cols;
public:
	Matrix();
	Matrix(int r, int c);
	Matrix(int r, int c,double val);
	~Matrix(){}

	int getRows() const { return rows; }
	int getCols() const { return cols; }
	void setData(int r, int c, double val) { data[r][c] = val; }
	double getData(int r, int c) const { return data[r][c]; }
	
	Matrix operator+(const Matrix& other)const;
	Matrix operator-(const Matrix& other)const;
	Matrix operator*(const Matrix& other)const;
	
	Matrix transpose() const; 
	void print() const;
};

#endif
