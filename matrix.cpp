#include "matrix.h"
#include <iostream>
#include <cmath>
using namespace std;
Matrix::Matrix():rows(0),cols(0){}
Matrix::Matrix(int r, int c) :rows(r), cols(c)
{
	data.resize(rows, vector<double>(cols, 0.0));
}
Matrix::Matrix(int r, int c, double val) :rows(r), cols(c)
{
	data.resize(rows, vector<double>(cols, val));
}

Matrix Matrix::operator+(const Matrix& other)const
{
	if (cols != other.cols || rows != other.rows)
	{
		cout << "error:out of '+'range";
		return Matrix();
	}
	Matrix result(rows, cols);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			result.data[i][j] = data[i][j] + other.data[i][j];
		}
	}
	return result;
}	
Matrix Matrix::operator-(const Matrix& other)const
{
	if (cols != other.cols || rows != other.rows)
	{
		cout << "error:out of '-'range";
		return Matrix();
	}
	Matrix result(rows, cols);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			result.data[i][j] = data[i][j] - other.data[i][j];
		}
	}
	return result;
}

Matrix Matrix::operator*(const Matrix& other)const
{
	if (cols != other.rows)
	{
		cout << "error:out of '*'range";
		return Matrix();
	}
	Matrix result(rows, other.cols);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < other.cols; j++)
		{
			
			for (int k = 0; k < cols; k++)
			{
				result.data[i][j] += data[i][k] * other.data[k][j];
			}
		}
	}
	return result;
}
Matrix Matrix::transpose() const 
{
	Matrix result(cols, rows);  
	for (int i = 0; i < rows; i++) 
	{
		for (int j = 0; j < cols; j++) 
		{
			result.data[j][i] = data[i][j]; 
		}
	}
	return result;
}
void Matrix::print() const
{
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if (j == cols - 1) cout << data[i][j] << endl;
			else cout << data[i][j] << " ";
		}
	}
}