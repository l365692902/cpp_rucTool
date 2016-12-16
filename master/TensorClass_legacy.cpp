#include <iostream>
#include <cassert>
#include "mkl.h"
#include "TensorClass.hpp"
#include "Common.hpp"

namespace pwm
{
	void tensor::merge(int begin, int end)
	{
		assert((begin > 0) && (end > begin) && (end <= shp.size()));
		int times = end - begin;
		for (int i = 0; i < times; i++)
		{
			shp.at(begin - 1) *= shp.at(begin);
			shp.erase(shp.begin() + begin);
		}
	}




	int tensor::calc_shp(int idx_1, int cnt_1)
	{
		int rslt = 1;
		idx_1--;
		if (cnt_1 < 1)
		{
			//std::cout << "error in calc_shp: cnt=" << cnt_1 << std::endl;
			return 1;
		}
		while (cnt_1--)
		{
			if (idx_1 < shp.size())
			{
				rslt *= shp.at(idx_1++);
			}
			else
			{
				rslt *= shp.at(idx_1 = 0);
			}
		}
		return rslt;
	}

	void tensor::operator<<(int idx)
	{
		//_ASSERT(idx <= shp.size());
		assert(idx <= shp.size());
		assert(sizeof(float) == sizeof(int));
		//if pull the last index, do transpose
		int legacy_size;//size of x+idx i.e. rows
		int remnant;//size of y i.e. cols
		int *k = NULL;
		float *fk = NULL;
		if (idx == 1)
		{
			return;
		}
		else if (idx == shp.size())
		{
			legacy_size = calc_shp(1, idx - 1);
			MKL_Dimatcopy('R', 'T', legacy_size, shp.back(), 1.0, ptns, shp.back(), legacy_size);//transpose
			shp.insert(shp.begin(), shp.back());
			shp.pop_back();
		}
		else
		{
			// |---x---||idx||--y--|
			//
			// separate rows and cols:
			// |---x---||idx|,|--y--|
			//
			// permute rows:
			// |idx||---x---|,|--y--|
			// 
			// permute rows means permute k(1,2,3,...), refer to mkl function ?lapmr
			// 
			legacy_size = calc_shp(1, idx);
			remnant = calc_shp(idx + 1, shp.size() - idx);
			//k = new int[size];
			k = (int *)MKL_malloc(legacy_size*sizeof(int), MKLalignment);
			//fk = new float[size];
			fk = (float *)MKL_malloc(legacy_size*sizeof(float), MKLalignment);
			//std::cout << "any thing wrong?" << std::endl;
			//memcpy_s(k, size*sizeof(int), kk, size*sizeof(int));
			for (int i = 0; i < legacy_size; i++)
			{
				((int *)fk)[i] = i + 1;
				//k[i] = i + 1;
			}
			// ?imatcopy accept only float or double array
			// but float occupy the same space as int which is 4 Byte(at least in most system)
			// so we can force the function accept the int array as a float array.
			// better to check if sizeof(float)==sizeof(int) in the first place
			//MKL_Simatcopy('R', 'T', size / shp.at(idx - 1),
			//	shp.at(idx - 1), 1.0, (float *)k, shp.at(idx - 1), size / shp.at(idx - 1));//transpose k

			MKL_Somatcopy('R', 'T', legacy_size / shp.at(idx - 1),
				shp.at(idx - 1), 1.0, fk, shp.at(idx - 1), (float *)k, legacy_size / shp.at(idx - 1));//transpose k
			//Shw_Mtx("k:", k, size / 6, 6);
			LAPACKE_dlapmr(CblasRowMajor, 1, legacy_size, remnant, ptns, remnant, k);//permute according k
			//delete[] k;
			MKL_free(k);
			//delete[] fk;
			MKL_free(fk);
			//trim shape
			shp.insert(shp.begin(), shp.at(idx - 1));
			shp.erase(shp.begin() + idx);
		}

	}

	void tensor::operator>>(int idx)
	{
		//_ASSERT(idx <= shp.size());
		assert(idx <= shp.size());
		assert(sizeof(float) == sizeof(int));
		int legacy_size;
		int x, y;
		int *k = NULL;
		float *fk = NULL;
		if (idx == shp.size())
		{
			return;
		}
		else if (idx == 1)
		{
			legacy_size = calc_shp(2, shp.size() - 1);
			MKL_Dimatcopy('R', 'T', shp.front(), legacy_size, 1.0, ptns, legacy_size, shp.front());//transpose
			shp.push_back(shp.front());
			shp.erase(shp.begin());
		}
		else
		{
			// |---x---||idx||--y--|
			x = calc_shp(1, idx - 1);
			y = calc_shp(idx + 1, shp.size() - idx);
			legacy_size = shp.at(idx - 1)*y;
			//k = new int[size];
			k = (int *)MKL_malloc(legacy_size*sizeof(int), MKLalignment);
			//fk = new float[size];
			fk = (float *)MKL_malloc(legacy_size*sizeof(float), MKLalignment);
			for (int i = 0; i < legacy_size; i++)
			{
				((int *)fk)[i] = i + 1;
				//k[i] = i + 1;
			}
			// k:|idx||--y--| --> |--y--||idx|
			MKL_Somatcopy('R', 'T', shp.at(idx - 1), y, 1.0, fk, y, (float *)k, shp.at(idx - 1));
			// |---x---||y + idx|:RowMajor; ld:y+idx
			// |y + idx||---x---|:ColMajor; ld:y+idx
			LAPACKE_dlapmr(CblasColMajor, 1, legacy_size, x, ptns, legacy_size, k);
			//delete[] k;
			MKL_free(k);
			//delete[] fk;
			MKL_free(fk);
			shp.insert(shp.end(), shp.at(idx - 1));
			shp.erase(shp.begin() + idx - 1);
		}
	}

		void tensor::legacy_svd(int idx4row, int preserve, double *&U, double *&LAM, double *&VT)
		{
			int rows = calc_shp(1, idx4row);
			int cols = calc_shp(idx4row + 1, shp.size() - idx4row);
			int size = rows*cols;
			int rank = (std::min)(rows, cols);
			if (preserve > rank)
			{
				preserve = rank;
			}
			double *u = new double[rows*rows];
			double *lam = new double[rank];
			double *vt = new double[cols*cols];
			//U = (double *)realloc(U, sizeof(double)*rows*rows);
			//lam = (double *)realloc(lam, sizeof(double)*rank);
			//VT = (double *)realloc(VT, sizeof(double)*cols*cols);
			//double *s = new double[rows];
			double *superb = new double[rank - 1];
			double *tmp = NULL;
			tmp = new double[size];
			std::memcpy(tmp, ptns, size*sizeof(double));
			//cpy_A_to_B(tmp, ptns, size);
			//after svd, A(tmp, ptns) will corrupt, so that using tmp
			LAPACKE_dgesvd(CblasRowMajor, 'S', 'S', rows, cols, tmp, cols, lam, u, rank, vt, cols, superb);

			U = (double *)std::realloc(U, sizeof(double)*rows*preserve);
			MKL_Domatcopy('C', 'N', preserve, rows, 1.0, u, rank, U, preserve);//this is U
			//MKL_Domatcopy('R', 'T', rows, preserve, 1.0, u, rank, U, rows);//actually UT
			LAM = (double *)std::realloc(LAM, preserve*sizeof(double));
			std::memcpy(LAM, lam, preserve*sizeof(double));
			//cpy_L_to_R(lam, LAM, preserve);
			VT = (double *)std::realloc(VT, preserve*cols*sizeof(double));
			std::memcpy(VT, vt, preserve*cols*sizeof(double));
			//cpy_L_to_R(vt, VT, preserve*cols);

			delete[] u;
			delete[] lam;
			delete[] vt;
			delete[] tmp;
			delete[] superb;
		}


}