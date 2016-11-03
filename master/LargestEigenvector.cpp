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
#include <cstring>

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
		int MaxIter,
		std::array<tensor *, MaxNumTensor> y_out,
		std::array<double *, MaxNumTensor> lam_out
		)
	{
		int cntT = 0, x_size = x_in.size;
		while (T_in[cntT] != 0)
		{
			cntT++;
		}
		tensor T_alan, Tx_bob;
		double error0 = 1.0, p_lam0 = 0.0, error_total;
		std::array<double, MaxNumTensor> p_lam;
		std::array<double *, MaxNumTensor> p_y;
		std::array<double, 2 * MaxNumTensor> error;
		while (error0 > Converge_in)
		{
			for (int i = cntT - 1; i >= 0; i--)//one time
			{
				applyOneMPS('R', *T_in[i], x_in, *lam_out[i]);
				//*y_out[i] = x_in;
			}
			error0 = std::abs(*lam_out[0] - p_lam0);
			p_lam0 = *lam_out[0];
			std::cout << "error: " << error0 << " lambda: " << *lam_out[0] << std::endl;
		}

		for (int i = 0; i < cntT; i++)//store results
		{
			p_lam[i] = *lam_out[i];
			p_y[i] = (double *)MKL_malloc(x_size*sizeof(double), MKLalignment);
		}

		for (int i = cntT - 1; i >= 0; i--)//one more time
		{
			applyOneMPS('R', *T_in[i], x_in, *lam_out[i]);
			*y_out[i] = x_in;
			std::memcpy(p_y[i], y_out[i], x_size*sizeof(double));
		}

		for (int i = 0; i < cntT; i++)
		{
			error[2 * i] = p_lam[i] - *lam_out[i];
			vdSub(x_size, p_y[i], y_out[i]->ptns, p_y[i]);
			error[2 * i + 1] = (p_y[i])[cblas_idamax(x_size, p_y[i], 1)];
		}

		error_total = error[cblas_idamax(cntT * 2, error.data(), 1)];
		std::cout << "total_error: " << error_total << std::endl;

		for (int i = 0; i < cntT; i++)
		{
			MKL_free(p_y[i]);
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
	void applyOneMPS(char L_R, tensor &in, tensor &x, double &norm)
	{
		//getNorm2(x.size, x.ptns);
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

	//************************************
	// Method:    getIdamax
	// FullName:  pwm::getIdamax
	// Access:    public 
	// Returns:   int
	// Qualifier:
	// Parameter: int size
	// Parameter: double * in
	// P.S:       find largest magnitude element's index
	//************************************
	int getIdamax(int size, double *in)
	{
		int cores = omp_get_max_threads();
		//int *index = (int *)MKL_malloc(cores*sizeof(int), MKLalignment);
		int *index = (int *)MKL_calloc(cores, sizeof(int), MKLalignment);
#pragma omp parallel
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

			double dmax = std::abs(in[thread_start]);
			//std::cout << thread_start << std::endl;
			double abs = 0.0;
			for (int i = thread_start + 1; i < thread_finish; i++)
			{
				//std::cout << i << std::endl;
				if ((abs = std::abs(in[i])) > dmax)
				{
					index[thread_num] = i;
					dmax = abs;
				}
			}
		}
		double outdmax = std::abs(in[index[0]]);
		//std::cout << index[0] << std::endl;
		double outabs = 0.0;
		int result = index[0];
		for (int i = 1; i < cores; i++)
		{
			//std::cout << index[i] << std::endl;
			if ((outabs = std::abs(in[index[i]])) > outdmax)
			{
				result = index[i];
				outdmax = outabs;
			}
		}
		MKL_free(index);
		return result;

	}

	double getMax(int size, double *in)
	{
		int cores = omp_get_max_threads();
		//int *index = (int *)MKL_malloc(cores*sizeof(int), MKLalignment);
		int *index = (int *)MKL_calloc(cores, sizeof(int), MKLalignment);

#pragma omp parallel
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

			double dmax = std::abs(in[thread_start]);
			//std::cout << thread_start << std::endl;
			double abs = 0.0;
			for (int i = thread_start + 1; i < thread_finish; i++)
			{
				//std::cout << i << std::endl;
				if ((abs = std::abs(in[i])) > dmax)
				{
					index[thread_num] = i;
					dmax = abs;
				}
			}
		}
		double outdmax = std::abs(in[index[0]]);
		//std::cout << index[0] << std::endl;
		double outabs = 0.0;
		int result = index[0];
		for (int i = 1; i < cores; i++)
		{
			//std::cout << index[i] << std::endl;
			if ((outabs = std::abs(in[index[i]])) > outdmax)
			{
				result = index[i];
				outdmax = outabs;
			}
		}
		MKL_free(index);
		return std::abs(in[result]);

	}

	double getDiff(char L_R_A_N, int size, double *L, double *R)
	{
		double *tmpL = NULL;
		double *tmpR = NULL;
		double result = 0;
		if (L == NULL)
		{
			if (R == NULL)
			{
				return 0.0;
			}
			else
			{
				//return std::abs(R[cblas_idamax(size, R, 1)]);
				return getMax(size, R);
			}
		}
		else if (R == NULL)
		{
			//return std::abs(L[cblas_idamax(size, L, 1)]);
			return getMax(size, L);
		}

		switch (L_R_A_N)
		{
		case 'A':
			//tmpL = (double *)std::realloc(tmpL, size*sizeof(double));
			tmpL = (double *)MKL_realloc(tmpL, size*sizeof(double));
			//tmpR = (double *)std::realloc(tmpR, size*sizeof(double));
			tmpR = (double *)MKL_realloc(tmpR, size*sizeof(double));
			vdAbs(size, L, tmpL);
			vdAbs(size, R, tmpR);
			vdSub(size, tmpL, tmpR, tmpL);
			//result = std::abs(tmpL[cblas_idamax(size, tmpL, 1)]);
			result = getMax(size, tmpL);
			break;
		case 'L':
			//tmpL = (double *)std::realloc(tmpL, size*sizeof(double));
			tmpL = (double *)MKL_realloc(tmpL, size*sizeof(double));
			vdAbs(size, L, tmpL);
			vdSub(size, tmpL, R, tmpL);
			//result = std::abs(tmpL[cblas_idamax(size, tmpL, 1)]);
			result = getMax(size, tmpL);
			break;
		case 'R':
			//tmpR = (double *)std::realloc(tmpR, size*sizeof(double));
			tmpR = (double *)MKL_realloc(tmpR, size*sizeof(double));
			vdAbs(size, R, tmpR);
			vdSub(size, L, tmpR, tmpR);
			//result = std::abs(tmpR[cblas_idamax(size, tmpR, 1)]);
			result = getMax(size, tmpR);
			break;
		default:
			//tmpL = (double *)std::realloc(tmpL, size*sizeof(double));
			tmpL = (double *)MKL_realloc(tmpL, size*sizeof(double));
			vdSub(size, L, R, tmpL);
			//result = std::abs(tmpL[cblas_idamax(size, tmpL, 1)]);
			result = getMax(size, tmpL);
			break;
		}
		//delete[] tmpL;
		MKL_free(tmpL);
		//delete[] tmpR;
		MKL_free(tmpR);
		return result;
	}

}