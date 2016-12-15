#include <cstring>
#include <array>
#include "mkl.h"
#include "Common.hpp"
#include "TensorClass.hpp"

namespace pwm
{
	void tensor::svd(int idx_for_rows_in, std::array<tensor *, 3> ULV_out)
	{
		int rows = this->calc_shp(1, idx_for_rows_in);
		int cols = this->size / rows;
		int __rows = rows;
		int __cols = cols;
		int __rank = cols;
		bool big_endian = true;
		int mtx_layout = LAPACK_ROW_MAJOR;
		int lda = cols;
		double *u = NULL, *l = NULL, *vt = NULL;
		char jobu = 'N', jobvt = 'N';
		double *superb = NULL, *shadow = NULL;
		if (rows < cols)
		{
			big_endian = false;
			mtx_layout = LAPACK_COL_MAJOR;
			__rows = cols;
			__cols = rows;
			__rank = rows;
		}

		l = (double *)MKL_malloc(__rank*sizeof(double), MKLalignment);
		if (big_endian == true)
		{
			if (ULV_out[0] != 0)
			{
				u = (double *)MKL_malloc(__rows*__rank*sizeof(double), MKLalignment);
				jobu = 'S';
			}

			if (ULV_out[2] != 0)
			{
				vt = (double *)MKL_malloc(__cols*__cols*sizeof(double), MKLalignment);
				jobvt = 'S';
			}
		}
		else
		{
			if (ULV_out[2] != 0)
			{
				u = (double *)MKL_malloc(__rows*__rank*sizeof(double), MKLalignment);
				jobu = 'S';
			}

			if (ULV_out[0] != 0)
			{
				vt = (double *)MKL_malloc(__cols*__cols*sizeof(double), MKLalignment);
				jobvt = 'S';
			}
		}
		superb = (double *)MKL_malloc(__rank*sizeof(double), MKLalignment);
		shadow = (double *)MKL_malloc(this->size*sizeof(double), MKLalignment);
		std::memcpy(shadow, this->ptns, this->size*sizeof(double));

		LAPACKE_dgesvd(mtx_layout, jobu, jobvt, __rows, __cols, shadow, lda, l, u, __rows, vt, __rank, superb);

		ULV_out[1]->reset({ {__rank} }, l);
		if (big_endian == true)
		{
			if (jobu == 'S')
			{
				ULV_out[0]->reset({ {rows, cols} }, u);
			}

			if (jobvt == 'S')
			{
				ULV_out[2]->reset({ {__rank,__rank} }, vt);
			}
		}
		if (big_endian == false)
		{
			if (jobu == 'S')
			{
				ULV_out[2]->reset({ {rows, cols} }, u);
			}
			if (jobvt == 'S')
			{
				ULV_out[0]->reset({ {__rank, __rank} }, vt);
			}
		}

		MKL_free(u);
		MKL_free(l);
		MKL_free(vt);
		MKL_free(superb);
		MKL_free(shadow);
		return;
	}
}