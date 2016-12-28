#include <iostream>
#include <array>
#include <cmath>
#include <limits>
#include <cstring>
#include "mkl.h"
#include "omp.h"
#include "Common.hpp"
#include "TensorClass.hpp"
#include "LargestEigenvector.hpp"
#include "TensorContraction.hpp"

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
		char L_R,
		std::array<tensor *, MaxNumTensor> T_in,
		tensor &x_in,
		double Converge_in,
		int MaxIter_in,
		std::array<tensor *, MaxNumTensor> y_out,
		std::array<double *, MaxNumTensor> lam_out
	)
	{
		int Tensor_cnt = 0; //x_size = x_in.size;
		int iterNo = 0;
		while (T_in[Tensor_cnt] != 0)
		{
			Tensor_cnt++;
		}
		tensor T_alan, Tx_bob;
		double error0 = 1.0, p_lam0 = 0.0, error_total = 1.0;
		std::array<double, MaxNumTensor> p_lam{};
		std::array<double *, MaxNumTensor> p_y;
		std::array<double, 2 * MaxNumTensor> error;
		std::array<int, MaxNumTensor> order;
		switch (L_R)
		{
		case 'L':
			for (int i = 0; i < Tensor_cnt; i++)
			{
				order[i] = i;//0,1,2,3,4,...
			}
			break;
		case 'R':
			for (int i = 0; i < Tensor_cnt; i++)
			{
				order[i] = Tensor_cnt - 1 - i;//n...4,3,2,1,0
			}
			break;
		default:
			std::cout << "largestEigenvalue::mod_mismatch" << std::endl;
			break;
		}

		for (int i = 0; i < Tensor_cnt; i++)//one time
		{
			p_y[i] = (double *)MKL_calloc(x_in.size, sizeof(double), MKLalignment);
			applyOneMPS(L_R, *T_in[order[i]], x_in, *lam_out[order[i]]);
		}
		iterNo++;

		while (error0 > Converge_in && iterNo < MaxIter_in)
		{
			for (int i = 0; i < Tensor_cnt; i++)//one time
			{
				applyOneMPS(L_R, *T_in[order[i]], x_in, *lam_out[order[i]]);
			}
			error0 = std::abs(*lam_out[0] - p_lam0);
			p_lam0 = *lam_out[0];
			iterNo++;
			//error0 = getMax(cntT, sub_lam.data());
			std::cout << "error: " << error0 << " lambda: " << *lam_out[0] << std::endl;
		}

		int error_cnt = 0;
		while (error_total > Converge_in && iterNo < MaxIter_in)
		{
			error_cnt = 0;
			for (int j = 0; j < Tensor_cnt; j++)
			{
				*y_out[order[j]] = x_in;
				applyOneMPS(L_R, *T_in[order[j]], x_in, *lam_out[order[j]]);
				error[error_cnt] = p_lam[order[j]] - *lam_out[order[j]];
				error_cnt++;
				p_lam[order[j]] = *lam_out[order[j]];//for next step
				vdSub(y_out[order[j]]->size, p_y[order[j]], y_out[order[j]]->ptns, p_y[order[j]]);
				error[error_cnt] = getMax(y_out[order[j]]->size, p_y[order[j]]);
				error_cnt++;
				std::memcpy(p_y[order[j]], y_out[order[j]]->ptns, y_out[order[j]]->size * sizeof(double));//for next step
			}
			error_total = getMax(error_cnt, error.data());
			iterNo++;
			std::cout << "error: " << error_total << " lambda: " << *lam_out[0] << std::endl;
		}

		std::cout << "total_error: " << error_total << std::endl;

		for (int i = 0; i < Tensor_cnt; i++)
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
	void applyOneMPS(char L_R, tensor &in, tensor &x_io, double &norm)
	{
		//getNorm2(x.size, x.ptns);
		switch (L_R)
		{
		case 'L':
			tensorContract(x_io, in, x_io);
			tensorContract(2, in, x_io, x_io);
			break;
		case 'R':
			tensorContract(in, x_io, x_io);
			tensorContract(x_io, in, 2, x_io);
			break;
		default:
			std::cout << "control char mismatch" << std::endl;
			break;
		}
		norm = getNorm2(x_io.size, x_io.ptns);
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

	//************************************
	// Method:    getMax
	// FullName:  pwm::getMax
	// Access:    public 
	// Returns:   double
	// Qualifier:
	// Parameter: int size
	// Parameter: double * in
	// P.S:       find largest magnitude element's magnitude
	//************************************
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
			tmpL = (double *)MKL_realloc(tmpL, size * sizeof(double));
			//tmpR = (double *)std::realloc(tmpR, size*sizeof(double));
			tmpR = (double *)MKL_realloc(tmpR, size * sizeof(double));
			vdAbs(size, L, tmpL);
			vdAbs(size, R, tmpR);
			vdSub(size, tmpL, tmpR, tmpL);
			//result = std::abs(tmpL[cblas_idamax(size, tmpL, 1)]);
			result = getMax(size, tmpL);
			break;
		case 'L':
			//tmpL = (double *)std::realloc(tmpL, size*sizeof(double));
			tmpL = (double *)MKL_realloc(tmpL, size * sizeof(double));
			vdAbs(size, L, tmpL);
			vdSub(size, tmpL, R, tmpL);
			//result = std::abs(tmpL[cblas_idamax(size, tmpL, 1)]);
			result = getMax(size, tmpL);
			break;
		case 'R':
			//tmpR = (double *)std::realloc(tmpR, size*sizeof(double));
			tmpR = (double *)MKL_realloc(tmpR, size * sizeof(double));
			vdAbs(size, R, tmpR);
			vdSub(size, L, tmpR, tmpR);
			//result = std::abs(tmpR[cblas_idamax(size, tmpR, 1)]);
			result = getMax(size, tmpR);
			break;
		default:
			//tmpL = (double *)std::realloc(tmpL, size*sizeof(double));
			tmpL = (double *)MKL_realloc(tmpL, size * sizeof(double));
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