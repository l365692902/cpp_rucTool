#include "TensorClass.hpp"
#include "LargestEigenvector.hpp"
#include <array>
#include "Common.hpp"
#include <iostream>
#include "TensorContraction.hpp"
#include "mkl.h"
#include <cmath>
#include <limits>
#include "omp.h"

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
		getNorm2(x.size, x.ptns);
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
		norm = getNorm2(x.size, x.ptns);
		return;
	}

	//************************************
	// Method:    getNorm
	// FullName:  pwm::getNorm
	// Access:    public 
	// Returns:   double
	// Qualifier:
	// Parameter: int size
	// Parameter: double * in
	// P.S:       in-place normalization, return coefficient, and perform normalization
	//			  correctness guaranteed, but in a relatively low accuracy
	// Note:      multiply won't affect accuracy, while add will
	//************************************
	double getNorm(int size, double *in)
	{
		double norm = 0, reciprocal = 1;
#pragma omp parallel for reduction(+:norm)
		for (int i = 0; i < size; i++)
		{
			norm = norm + in[i] * in[i];
		}
		norm = std::sqrt(norm);
		if (std::abs(norm - 1.0) < 1e-15)
		{
			return 1.0;
		}
		reciprocal = 1.0 / norm;
#pragma omp parallel for
		for (int i = 0; i < size; i++)
		{
			in[i] *= reciprocal;
		}
		return norm;
	}

	//************************************
	// Method:    getNorm2
	// FullName:  pwm::getNorm2
	// Access:    public 
	// Returns:   double
	// Qualifier:
	// Parameter: int size
	// Parameter: double * in
	// P.S:       can achieve high accuracy, using a strange algorithm copied from dnrm2.f
	//************************************
	double getNorm2(int size, double *in)
	{
		double norm = 0.0, reciprocal = 1.0;
#pragma omp parallel reduction(+:norm)
		{
			int thread_start, thread_finish, threads_total, thread_num, chunk, remain;
			threads_total = omp_get_num_threads();
			thread_num = omp_get_thread_num();
			remain = size%threads_total;
			chunk = size / threads_total;
			if (remain != 0 && thread_num < remain)
			{
				chunk++;
			}
			thread_start = thread_num*chunk;
			if (remain != 0 && thread_num >= remain)
			{
				thread_start += remain;
			}
			thread_finish = thread_start + chunk;

			double absxi = 0.0, scale = 0.0, ssq = 1.0;
			for (int i = thread_start; i < thread_finish; i++)
			{
				if (in[i] != 0.0)
				{
					absxi = std::abs(in[i]);
					if (scale < absxi)
					{
						ssq = 1.0 + ssq*(scale*scale) / (absxi*absxi);
						scale = absxi;
					}
					else
					{
						ssq += (absxi*absxi) / (scale*scale);
					}
				}
			}
			norm = scale*scale*ssq;
		}

		norm = std::sqrt(norm);
		if (std::abs(norm - 1.0) < std::numeric_limits<double>::epsilon())
		{
			return 1.0;
		}
		reciprocal = 1.0 / norm;
#pragma omp parallel for
		for (int i = 0; i < size; i++)
		{
			in[i] *= reciprocal;
		}
		return norm;
	}
}