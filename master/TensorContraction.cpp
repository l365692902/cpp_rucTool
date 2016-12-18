#include <cassert>
#include <string>
#include <cstring>
#include <array>
#include <iostream>
#include "TensorContraction.hpp"
#include "TensorClass.hpp"
#include "mkl.h"

namespace pwm
{
	//************************************
	// Method:    tensorContract
	// FullName:  pwm::tensorContract
	// Access:    public 
	// Returns:   void
	// Qualifier:
	// Parameter: tensor & A
	// Parameter: tensor & B
	// Parameter: int * permA
	// Parameter: int * permB
	// Parameter: int merg
	// Parameter: int * permC
	// Parameter: tensor & C
	// P.S:       underlying routine for user friendly tensor contract
	// Note:      in linux, do assignment after this tensor(or to say class) has been deconstructed,
	//            will throw out segmentation fault; while a pointer to a class can be reused.
	//************************************
	void tensorContract(tensor &A, tensor &B, int *permA, int *permB, int merg, int *permC, tensor &C)
	{
		tensor *alan = new tensor(), *bob = new tensor(), *cruz = new tensor();
		int left = 1, mid = 1, right = 1, lda = 1, ldb = 1;
		bool KeepAlan = true;
		if (permA != NULL)
		{
			A.permute(permA, *alan);
			KeepAlan = false;
		}
		else
		{
			alan->~tensor();
			alan = &A;
		}
		bool KeepBob = true;
		if (permB != NULL)
		{
			B.permute(permB, *bob);
			KeepBob = false;
		}
		else
		{
			bob->~tensor();
			bob = &B;
		}
		bool NeedCopyC = true;
		if ((&C != &A) && (&C != &B))
		{
			NeedCopyC = false;
			cruz->~tensor();
			cruz = &C;
		}
		cruz->shp.clear();
		for (int i = merg; i < alan->order; i++)
		{
			left *= alan->shp.at(i);
			cruz->shp.push_back(alan->shp.at(i));
		}
		for (int i = merg; i < bob->order; i++)
		{
			right *= bob->shp.at(i);
			cruz->shp.push_back(bob->shp.at(i));
		}
		mid = alan->size / left;
		assert(right == bob->size / mid);
		lda = left; ldb = right;

		cruz->size = left*right;
		cruz->order = cruz->shp.size();
		cruz->IsOrdered = true;
		cruz->unit = (int *)MKL_realloc(cruz->unit, cruz->order*sizeof(int));
		cruz->add_unit = (int *)MKL_realloc(cruz->add_unit, cruz->order*sizeof(int));
		cruz->ptns = (double *)MKL_realloc(cruz->ptns, cruz->size*sizeof(double));
		int cumprod = 1;
		cruz->unit[cruz->order - 1] = 1;
		cruz->add_unit[cruz->order - 1] = 1;
		for (int i = cruz->order - 2; i >= 0; i--)
		{
			cumprod *= cruz->shp.at(i + 1);
			cruz->unit[i] = cumprod;
			cruz->add_unit[i] = 1;
		}
		//d1*d1=d2 is OK, when d1*d2=d1 or d1*d2=d2 will be wrong
		cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, left, right, mid, 1.0, alan->ptns, lda, bob->ptns, ldb, 0.0, cruz->ptns, right);
		if (permC != NULL)
		{
			cruz->permute(permC);
		}
		if (NeedCopyC == true)
		{
			//C.~tensor();
			C = (*cruz);//segmentation fault in linux
			cruz->~tensor();
		}
		if (KeepAlan == false)
		{
			alan->~tensor();
		}
		if (KeepBob == false)
		{
			bob->~tensor();
		}
		return;
	}

	//basically abandoned
	void tensorContract(tensor &A, tensor &B, std::string idxA,
		std::string idxB, std::array<int, MaxOrder> permC, tensor &C)
	{
		int *permA = NULL;
		int *permB = NULL;
		int *permuteC = NULL;
		int com_cnt = 0;
		if (permC.front() != 0)
		{
			permuteC = permC.data();
		}
		resolve_perm(idxA, idxB, permA, permB, com_cnt);
		tensorContract(A, B, permA, permB, com_cnt, permuteC, C);

		return;
	}

	//friendly user interface, for fast develop
	void tensorContract(tensor &A, tensor &B, std::string idxA, std::string idxB, std::string idxC, tensor &C)
	{
		int *permA = NULL, *permB = NULL, *permC = NULL;
		int com_cnt = 0;
		resolve_perm(idxA, idxB, idxC, permA, permB, com_cnt, permC);
		tensorContract(A, B, permA, permB, com_cnt, permC, C);
		return;
	}

	//underlying routine for myself to construct mid-level routine
	void tensorContract(tensor &A, tensor &B, tensor &C)
	{
		assert(A.shp.back() == B.shp.front());
		tensor *cruz = new tensor();
		bool NeedCopyC = true;
		if ((&C != &A) && (&C != &B))
		{
			NeedCopyC = false;
			cruz->~tensor();
			cruz = &C;
		}

		int left, mid, right, lda, ldb;
		left = A.size / A.shp.back();
		mid = A.shp.back();
		right = B.size / B.shp.front();
		lda = mid;
		ldb = right;
		cruz->shp.clear();
		cruz->shp.assign(A.shp.begin(), A.shp.end() - 1);
		cruz->shp.insert(cruz->shp.end(), B.shp.begin() + 1, B.shp.end());//shp done
		cruz->size = left*right;
		cruz->order = cruz->shp.size();
		cruz->IsOrdered = true;
		cruz->unit = (int *)MKL_realloc(cruz->unit, cruz->order*sizeof(int));
		cruz->add_unit = (int *)MKL_realloc(cruz->add_unit, cruz->order*sizeof(int));
		cruz->ptns = (double *)MKL_realloc(cruz->ptns, cruz->size*sizeof(double));
		int cumprod = 1;
		cruz->unit[cruz->order - 1] = 1;
		cruz->add_unit[cruz->order - 1] = 1;
		for (int i = cruz->order - 2; i >= 0; i--)
		{
			cumprod *= cruz->shp.at(i + 1);
			cruz->unit[i] = cumprod;
			cruz->add_unit[i] = 1;
		}

		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, left, right, mid, 1.0, A.ptns, lda, B.ptns, ldb, 0.0, cruz->ptns, right);

		if (NeedCopyC == true)
		{
			C = (*cruz);
			cruz->~tensor();
		}
		return;
	}

	void tensorContract(tensor &A, int merg, tensor &B, tensor &C)
	{
		int left, mid = 1, right, lda, ldb;
		for (int i = 0; i < merg; i++)
		{
			assert(A.shp.at(A.order - merg + i) == B.shp.at(i));
			mid *= B.shp.at(i);
		}
		tensor *cruz = new tensor();
		bool NeedCopyC = true;
		if ((&C != &A) && (&C != &B))
		{
			NeedCopyC = false;
			cruz->~tensor();
			cruz = &C;
		}

		left = A.size / mid;
		right = B.size / mid;
		lda = mid;
		ldb = right;
		cruz->shp.clear();
		cruz->shp.assign(A.shp.begin(), A.shp.end() - merg);
		cruz->shp.insert(cruz->shp.end(), B.shp.begin() + merg, B.shp.end());//shp done
		cruz->size = left*right;
		cruz->order = cruz->shp.size();
		cruz->IsOrdered = true;
		cruz->unit = (int *)MKL_realloc(cruz->unit, cruz->order*sizeof(int));
		cruz->add_unit = (int *)MKL_realloc(cruz->add_unit, cruz->order*sizeof(int));
		cruz->ptns = (double *)MKL_realloc(cruz->ptns, cruz->size*sizeof(double));
		int cumprod = 1;
		cruz->unit[cruz->order - 1] = 1;
		cruz->add_unit[cruz->order - 1] = 1;
		for (int i = cruz->order - 2; i >= 0; i--)
		{
			cumprod *= cruz->shp.at(i + 1);
			cruz->unit[i] = cumprod;
			cruz->add_unit[i] = 1;
		}

		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, left, right, mid, 1.0, A.ptns, lda, B.ptns, ldb, 0.0, cruz->ptns, right);

		if (NeedCopyC == true)
		{
			C = (*cruz);
			cruz->~tensor();
		}
		return;
	}

	void tensorContract(int merg, tensor &A, tensor &B, tensor &C)
	{
		int left, mid = 1, right, lda, ldb;
		for (int i = 0; i < merg; i++)
		{
			assert(A.shp.at(i) == B.shp.at(i));
			mid *= B.shp.at(i);
		}
		tensor *cruz = new tensor();
		bool NeedCopyC = true;
		if ((&C != &A) && (&C != &B))
		{
			NeedCopyC = false;
			cruz->~tensor();
			cruz = &C;
		}
		left = A.size / mid;
		right = B.size / mid;
		lda = left;
		ldb = right;
		cruz->shp.clear();
		cruz->shp.assign(A.shp.begin() + merg, A.shp.end());
		cruz->shp.insert(cruz->shp.end(), B.shp.begin() + merg, B.shp.end());
		cruz->size = left*right;
		cruz->order = cruz->shp.size();
		cruz->IsOrdered = true;
		cruz->unit = (int *)MKL_realloc(cruz->unit, cruz->order*sizeof(int));
		cruz->add_unit = (int *)MKL_realloc(cruz->add_unit, cruz->order*sizeof(int));
		cruz->ptns = (double *)MKL_realloc(cruz->ptns, cruz->size*sizeof(double));
		int cumprod = 1;
		cruz->unit[cruz->order - 1] = 1;
		cruz->add_unit[cruz->order - 1] = 1;
		for (int i = cruz->order - 2; i >= 0; i--)
		{
			cumprod *= cruz->shp.at(i + 1);
			cruz->unit[i] = cumprod;
			cruz->add_unit[i] = 1;
		}
		cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, left, right, mid, 1.0, A.ptns, lda, B.ptns, ldb, 0.0, cruz->ptns, right);
		if (NeedCopyC == true)
		{
			C = (*cruz);
			cruz->~tensor();
		}
		return;
	}

	void tensorContract(tensor &A, tensor &B, int merg, tensor &C)
	{
		int left, mid = 1, right, lda, ldb;
		for (int i = 1; i <= merg; i++)
		{
			assert(A.shp.at(A.order - i) == B.shp.at(B.order - i));
			mid *= B.shp.at(B.order - i);
		}
		tensor *cruz = new tensor();
		bool NeedCopyC = true;
		if ((&C != &A) && (&C != &B))
		{
			NeedCopyC = false;
			cruz->~tensor();
			cruz = &C;
		}
		left = A.size / mid;
		right = B.size / mid;
		lda = mid;
		ldb = mid;
		cruz->shp.clear();
		cruz->shp.assign(A.shp.begin(), A.shp.end() - merg);
		cruz->shp.insert(cruz->shp.end(), B.shp.begin(), B.shp.end() - merg);
		cruz->size = left*right;
		cruz->order = cruz->shp.size();
		cruz->IsOrdered = true;
		cruz->unit = (int *)MKL_realloc(cruz->unit, cruz->order*sizeof(int));
		cruz->add_unit = (int *)MKL_realloc(cruz->add_unit, cruz->order*sizeof(int));
		cruz->ptns = (double *)MKL_realloc(cruz->ptns, cruz->size*sizeof(double));
		int cumprod = 1;
		cruz->unit[cruz->order - 1] = 1;
		cruz->add_unit[cruz->order - 1] = 1;
		for (int i = cruz->order - 2; i >= 0; i--)
		{
			cumprod *= cruz->shp.at(i + 1);
			cruz->unit[i] = cumprod;
			cruz->add_unit[i] = 1;
		}

		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, left, right, mid, 1.0, A.ptns, lda, B.ptns, ldb, 0.0, cruz->ptns, right);

		if (NeedCopyC == true)
		{
			C = (*cruz);
			cruz->~tensor();
		}
		return;
	}

	//idxA: eifjt-->efjit, 13425(permA)
	//idxB: fjgyen->efjgyn, 512346(permB); com_cnt=3
	//if no need to permute, return NULL
	void resolve_perm(std::string idxA, std::string idxB, int *&permA, int *&permB, int &com_cnt)
	{
		int *dumbA, *dumbB;
		dumbA = (int *)MKL_malloc(idxA.size()*sizeof(int), MKLalignment);
		permA = (int *)MKL_realloc(permA, idxA.size()*sizeof(int));
		dumbB = (int *)MKL_malloc(idxB.size()*sizeof(int), MKLalignment);
		permB = (int *)MKL_realloc(permB, idxB.size()*sizeof(int));
		for (int i = 0; i < idxA.size(); i++)
		{
			dumbA[i] = i + 1;
		}
		for (int i = 0; i < idxB.size(); i++)
		{
			dumbB[i] = i + 1;
		}
		std::string::size_type n;
		int A_cnt = 0, B_cnt = 0;

		for (int i = 0; i < idxA.size(); i++)
		{
			if (std::string::npos != (n = idxB.find(idxA.at(i))))
			{
				permA[A_cnt++] = dumbA[i];
				dumbA[i] = 0;
				permB[B_cnt++] = dumbB[n];
				dumbB[n] = 0;
			}
		}
		com_cnt = A_cnt;
		assert(A_cnt != 0);
		int orderA = 0, orderB = 0;
		for (int i = 0; i < idxA.size(); i++)
		{
			if (dumbA[i] != 0)
			{
				permA[A_cnt++] = dumbA[i];
			}
			if (permA[i] == i + 1)
			{
				orderA++;
			}
		}
		if (orderA == idxA.size())
		{
			MKL_free(permA);
			permA = NULL;
		}
		for (int i = 0; i < idxB.size(); i++)
		{
			if (dumbB[i] != 0)
			{
				permB[B_cnt++] = dumbB[i];
			}
			if (permB[i] == i + 1)
			{
				orderB++;
			}
		}
		if (orderB == idxB.size())
		{
			MKL_free(permB);
			permB = NULL;
		}

		MKL_free(dumbA);
		MKL_free(dumbB);
		return;
	}
	void resolve_perm(std::string idxA, std::string idxB, std::string idxC, int *&permA, int *&permB, int &com_cnt, int *&permC)
	{
		int *dumbA, *dumbB;
		dumbA = (int *)MKL_malloc(idxA.size()*sizeof(int), MKLalignment);
		permA = (int *)MKL_realloc(permA, idxA.size()*sizeof(int));
		dumbB = (int *)MKL_malloc(idxB.size()*sizeof(int), MKLalignment);
		permB = (int *)MKL_realloc(permB, idxB.size()*sizeof(int));
		permC = (int *)MKL_realloc(permC, idxC.size()*sizeof(int));
		for (int i = 0; i < idxA.size(); i++)
		{
			dumbA[i] = i + 1;
		}
		for (int i = 0; i < idxB.size(); i++)
		{
			dumbB[i] = i + 1;
		}
		std::string::size_type n;
		int A_cnt = 0, B_cnt = 0;

		for (int i = 0; i < idxA.size(); i++)
		{
			if (std::string::npos != (n = idxB.find(idxA.at(i))))
			{
				permA[A_cnt++] = dumbA[i];
				dumbA[i] = 0;
				permB[B_cnt++] = dumbB[n];
				dumbB[n] = 0;
			}
		}
		com_cnt = A_cnt;
		assert(A_cnt != 0);
		std::string idxM;
		int orderA = 0, orderB = 0, orderC = 0;//decide if A, B and C are originally in order
		for (int i = 0; i < idxA.size(); i++)
		{
			if (dumbA[i] != 0)
			{
				permA[A_cnt++] = dumbA[i];
				idxM.push_back(idxA.at(i));
			}
			if (permA[i] == i + 1)
			{
				orderA++;
			}
		}
		if (orderA == idxA.size())
		{
			MKL_free(permA);
			permA = NULL;
		}
		for (int i = 0; i < idxB.size(); i++)
		{
			if (dumbB[i] != 0)
			{
				permB[B_cnt++] = dumbB[i];
				idxM.push_back(idxB.at(i));
			}
			if (permB[i] == i + 1)
			{
				orderB++;
			}
		}
		if (orderB == idxB.size())
		{
			MKL_free(permB);
			permB = NULL;
		}
		assert(idxM.size() == idxC.size());
		for (int i = 0; i < idxC.size(); i++)
		{
			if (std::string::npos != (n = idxM.find(idxC.at(i))))
			{
				permC[i] = n + 1;
			}
			if (n == i)
			{
				orderC++;
			}
		}
		if (orderC == idxC.size())
		{
			MKL_free(permC);
			permC = NULL;
		}

		MKL_free(dumbA);
		MKL_free(dumbB);
		return;
	}


	//N stands for normal product
	//R stands for reciprocal, product the element-wise reciprocal of diag
	//under construction
	void tensorContractDiag(char N_R, tensor &A, double *diag, tensor &C)
	{
		int cols = A.shp.back();
		int rows = A.size / cols;
		bool keep__diag = false;
		if (&A != &C)
		{
			C = A;
		}
		double *__diag = NULL;
		switch (N_R)
		{
		case 'N':
			__diag = diag;
			keep__diag = true;
			break;
		case 'R':
			__diag = (double *)MKL_malloc(cols*sizeof(double), MKLalignment);
			vdInv(cols, diag, __diag);
			break;
		default:
			std::cout << "tensorContractDiag::mod_mismatch" << std::endl;
			break;
		}

		for (int i = 0; i < cols; i++)
		{
			cblas_daxpy(rows, __diag[i] - 1.0, C.ptns + i, cols, C.ptns + i, cols);
		}

		if (keep__diag == false)
		{
			MKL_free(__diag);
		}

		return;
	}
	void tensorContractDiag(char N_R, double *diag, tensor &B, tensor &C)
	{
		int rows = B.shp.front();
		int cols = B.size / rows;
		bool keep__diag = false;
		if (&B != &C)
		{
			C = B;
		}
		double *__diag = NULL;
		switch (N_R)
		{
		case 'N':
			__diag = diag;
			keep__diag = true;
			break;
		case 'R':
			__diag = (double *)MKL_malloc(rows*sizeof(double), MKLalignment);
			vdInv(rows, diag, __diag);
			break;
		default:
			std::cout << "tensorContractDiag::mod_mismatch" << std::endl;
			break;
		}

		for (int i = 0; i < rows; i++)
		{
			cblas_daxpy(cols, __diag[i] - 1.0, C.ptns + i*cols, 1, C.ptns + i*cols, 1);
		}

		if (keep__diag == false)
		{
			MKL_free(__diag);
		}
		return;
	}

}