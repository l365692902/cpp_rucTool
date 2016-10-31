#include "TensorClass.hpp"
#include "LargestEigenvector.hpp"
#include <array>
#include "Common.hpp"
#include <iostream>
#include "TensorContraction.hpp"
#include "mkl.h"
#include <cmath>

namespace pwm
{
	// A_____ B_____ C_____ D_____ x__
	//    |      |      |      |     |
	//                               |
	//  __|__  __|__  __|__  __|__  _|
	// A      B      C      D
	//    2      1__
	// 1__|__3   2_|
	// under construction...
	void largestEigenvalue(
		std::array<tensor *, MaxNumTensor> T_in,
		tensor &x_in,
		double Converge_in,
		std::array<tensor *, MaxNumTensor> y_out,
		std::array<double *, MaxNumTensor>lam_out
		)
	{
		int cntT = 0;
		while (T_in[cntT] != 0)
		{
			cntT++;
		}
		tensor T_alan, Tx_bob;
		for (int i = cntT - 1; i >= 0; i--)
		{
			applyMPS('R', *T_in[i], x_in, *lam_out[i]);
			*y_out[i] = x_in;

		}
		return;
	}

	//  D_____ x__
	//     |     |
	//           |
	//   __|__  _|
	//  D
	//    2      1__
	// 1__|__3   2_|
	void applyMPS(char L_R, tensor &in, tensor &x, double &norm)
	{
		normalize(x.size, x.ptns);
		switch (L_R)
		{
		case 'L':
			tensorContract(x, in, x);
			tensorContract(2, in, x, x);
			break;
		case 'R':
			tensorContract(in, x, x);
			tensorContract(x, in, 2, x);
			break;
		default:
			std::cout << "control char mismatch" << std::endl;
			break;
		}
		norm = normalize(x.size, x.ptns);
		return;
	}

	double normalize(int size, double *&x)
	{
		double norm = cblas_ddot(size, x, 1, x, 1);
		double *out = (double *)MKL_calloc(size, sizeof(double), MKLalignment);
		norm = std::sqrt(norm);
		cblas_daxpy(size, 1 / norm, x, 1, out, 1);
		MKL_free(x);
		x = out;
		return norm;
	}

	double getNorm(int size, double *in)
	{
		double norm = 0;
#pragma omp parallel for reduction(+:norm)
		for (int i = 0; i < size; i++)
		{
			norm = norm + in[i] * in[i];
		}
		norm = std::sqrt(norm);
		return norm;
	}
}