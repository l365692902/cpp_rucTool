#include <iostream>
#include <cmath>
#include <string>
#include <iomanip>
#include <cassert>
#include "mkl.h"
#include "pwm_legacy.hpp"

namespace pwm
{
	double diff(char mod, tensor &L, tensor &R)
	{
		if (L.ptns == NULL)
		{
			if (R.ptns == NULL)
			{
				return 0.0;
			}
			else
			{
				return std::abs(R.ptns[cblas_idamax(R.calc_shp(1, R.shp.size()), R.ptns, 1)]);
			}
		}
		else if (R.ptns == NULL)
		{
			return std::abs(L.ptns[cblas_idamax(L.calc_shp(1, L.shp.size()), L.ptns, 1)]);
		}

		int size = 0;
		if (L.shp.size()) size = L.calc_shp(1, L.shp.size());
		if (R.shp.size() && (size != R.calc_shp(1, R.shp.size())))
		{
			std::cout << "[diff::char_tns_tns] unequal-size tensors" << std::endl;
			return 0;
		}
		if (size == 0) return 0;
		double *tmpL = NULL;
		double *tmpR = NULL;
		double result = 0;
		switch (mod)
		{
		case 'A':
			//tmpL = (double *)std::realloc(tmpL, size*sizeof(double));
			tmpL = (double *)MKL_realloc(tmpL, size*sizeof(double));
			//tmpR = (double *)std::realloc(tmpR, size*sizeof(double));
			tmpR = (double *)MKL_realloc(tmpR, size*sizeof(double));
			vdAbs(size, L.ptns, tmpL);
			vdAbs(size, R.ptns, tmpR);
			vdSub(size, tmpL, tmpR, tmpL);
			result = std::abs(tmpL[cblas_idamax(size, tmpL, 1)]);
			break;
		case 'L':
			//tmpL = (double *)std::realloc(tmpL, size*sizeof(double));
			tmpL = (double *)MKL_realloc(tmpL, size*sizeof(double));
			vdAbs(size, L.ptns, tmpL);
			vdSub(size, tmpL, R.ptns, tmpL);
			result = std::abs(tmpL[cblas_idamax(size, tmpL, 1)]);
			break;
		case 'R':
			//tmpR = (double *)std::realloc(tmpR, size*sizeof(double));
			tmpR = (double *)MKL_realloc(tmpR, size*sizeof(double));
			vdAbs(size, R.ptns, tmpR);
			vdSub(size, L.ptns, tmpR, tmpR);
			result = std::abs(tmpR[cblas_idamax(size, tmpR, 1)]);
			break;
		default:
			//tmpL = (double *)std::realloc(tmpL, size*sizeof(double));
			tmpL = (double *)MKL_realloc(tmpL, size*sizeof(double));
			vdSub(size, L.ptns, R.ptns, tmpL);
			result = std::abs(tmpL[cblas_idamax(size, tmpL, 1)]);
			break;
		}
		//delete[] tmpL;
		MKL_free(tmpL);
		//delete[] tmpR;
		MKL_free(tmpR);
		return result;
	}

	//preliminary tested
	//in: mod can be 'A', 'L', 'R' and 'N' meaning compute abs of all, left tensor, right tensor or none
	//can handle empty array
	double diff(char mod, int size, double *L, double *R)
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
				return std::abs(R[cblas_idamax(size, R, 1)]);
			}
		}
		else if (R == NULL)
		{
			return std::abs(L[cblas_idamax(size, L, 1)]);
		}

		switch (mod)
		{
		case 'A':
			//tmpL = (double *)std::realloc(tmpL, size*sizeof(double));
			tmpL = (double *)MKL_realloc(tmpL, size*sizeof(double));
			//tmpR = (double *)std::realloc(tmpR, size*sizeof(double));
			tmpR = (double *)MKL_realloc(tmpR, size*sizeof(double));
			vdAbs(size, L, tmpL);
			vdAbs(size, R, tmpR);
			vdSub(size, tmpL, tmpR, tmpL);
			result = std::abs(tmpL[cblas_idamax(size, tmpL, 1)]);
			break;
		case 'L':
			//tmpL = (double *)std::realloc(tmpL, size*sizeof(double));
			tmpL = (double *)MKL_realloc(tmpL, size*sizeof(double));
			vdAbs(size, L, tmpL);
			vdSub(size, tmpL, R, tmpL);
			result = std::abs(tmpL[cblas_idamax(size, tmpL, 1)]);
			break;
		case 'R':
			//tmpR = (double *)std::realloc(tmpR, size*sizeof(double));
			tmpR = (double *)MKL_realloc(tmpR, size*sizeof(double));
			vdAbs(size, R, tmpR);
			vdSub(size, L, tmpR, tmpR);
			result = std::abs(tmpR[cblas_idamax(size, tmpR, 1)]);
			break;
		default:
			//tmpL = (double *)std::realloc(tmpL, size*sizeof(double));
			tmpL = (double *)MKL_realloc(tmpL, size*sizeof(double));
			vdSub(size, L, R, tmpL);
			result = std::abs(tmpL[cblas_idamax(size, tmpL, 1)]);
			break;
		}
		//delete[] tmpL;
		MKL_free(tmpL);
		//delete[] tmpR;
		MKL_free(tmpR);
		return result;
	}

	void Shw_Mtx(std::string tag, double *in, int rows, int cols)
	{
		std::cout << std::endl << tag << std::endl;
		std::cout << /*std::setiosflags(_IOSfixed) <<*/ std::setprecision(5);
		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
			{
				std::cout << std::setw(9) << in[i*cols + j] << ' ';
			}
			std::cout << std::endl;
		}
	}

	//preliminary tested
	//mod could be '{', meaning merge leading indices.
	//				'}', meaning merge ending indices.
	//				'Z', meaning merge middle indices.
	void product(char mod, tensor &A, tensor &B, tensor &C)
	{
		int left, mid, right, lda, ldb;
		std::vector<int> shp_tmp;
		//typedef enum {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113} CBLAS_TRANSPOSE;
		CBLAS_TRANSPOSE tranA, tranB;
		switch (mod)
		{
		case '{':
			//checked
			//out("{");
			assert(A.shp.front() == B.shp.front());
			left = A.calc_shp(2, A.shp.size() - 1);
			mid = A.shp.front();
			right = B.calc_shp(2, B.shp.size() - 1);
			lda = left;
			ldb = right;
			tranA = CblasTrans;
			tranB = CblasNoTrans;
			shp_tmp.insert(shp_tmp.begin(), A.shp.begin() + 1, A.shp.end());
			shp_tmp.insert(shp_tmp.end(), B.shp.begin() + 1, B.shp.end());
			break;
		case '}':
			//checked
			//out("}");
			assert(A.shp.back() == B.shp.back());
			left = A.calc_shp(1, A.shp.size() - 1);
			mid = A.shp.back();
			right = B.calc_shp(1, B.shp.size() - 1);
			lda = mid;
			ldb = mid;
			tranA = CblasNoTrans;
			tranB = CblasTrans;
			shp_tmp.insert(shp_tmp.begin(), A.shp.begin(), A.shp.end() - 1);
			shp_tmp.insert(shp_tmp.end(), B.shp.begin(), B.shp.end() - 1);
			break;
		default:
			//checked
			//out("Z");
			assert(A.shp.back() == B.shp.front());
			left = A.calc_shp(1, A.shp.size() - 1);
			mid = A.shp.back();
			right = B.calc_shp(2, B.shp.size() - 1);
			lda = mid;
			ldb = right;
			tranA = CblasNoTrans;
			tranB = CblasNoTrans;
			shp_tmp.insert(shp_tmp.begin(), A.shp.begin(), A.shp.end() - 1);
			shp_tmp.insert(shp_tmp.end(), B.shp.begin() + 1, B.shp.end());
			break;
		}
		//double *c = new double[left*right];
		double *c = (double *)MKL_malloc(left*right*sizeof(double), MKLalignment);
		cblas_dgemm(CblasRowMajor, tranA, tranB, left, right, mid,
			1.0, A.ptns, lda, B.ptns, ldb, 0.0, c, right);
		//cpy_A_to_B(C.ptns, c, left*right);
		MKL_free(C.ptns);
		C.ptns = c;
		C.shp = shp_tmp;
		//delete[] c;
	}




}