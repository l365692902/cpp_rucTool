#include <cstdarg>
#include <vector>
#include <cstring>
#include <iostream>
#include <cassert>
#include <array>
#include "mkl.h"
#include "Common.hpp"
#include "Random.hpp"
#include "TensorClass.hpp"
#include "omp.h"

namespace pwm
{

	tensor::tensor()//default constructor
	{
		//ptns = NULL;
		//unit = NULL;
		//add_unit = NULL;

		//for better alignment
		ptns = (double *)MKL_malloc(sizeof(double), MKLalignment);
		unit = (int *)MKL_malloc(sizeof(int), MKLalignment);
		add_unit = (int *)MKL_malloc(sizeof(int), MKLalignment);
	}

	tensor::~tensor()//destructor
	{
		MKL_free(ptns);
		MKL_free(unit);
		MKL_free(add_unit);
		ptns = NULL;
		unit = NULL;
		add_unit = NULL;
	}

	tensor::tensor(int fst, ...)//constructor
	{
		int i_arg;
		va_list p_arg;
		va_start(p_arg, fst);
		shp.clear();
		shp.push_back(fst);
		size = 1;
		size *= fst;
		order = 1;
		while ((i_arg = va_arg(p_arg, int)) != 0)
		{
			shp.push_back(i_arg);
			size *= i_arg;
			order++;
		}
		va_end(p_arg);
		ptns = (double *)MKL_malloc(size * sizeof(double), MKLalignment);
		unit = (int *)MKL_malloc(order * sizeof(double), MKLalignment);
		add_unit = (int *)MKL_malloc(order * sizeof(double), MKLalignment);
		int cumprod = 1;
		unit[order - 1] = 1;
		add_unit[order - 1] = 1;
		for (int i = order - 2; i >= 0; i--)//get unit and add_unit based on shp
		{
			cumprod *= shp.at(i + 1);
			unit[i] = cumprod;
			add_unit[i] = 1;
		}
		IsOrdered = true;
	}

	tensor::tensor(std::array<int, MaxOrder> in)//constructor
	{
		assert(in[0] != 0);
		shp.clear();
		size = 1;
		int i = 0;
		while (in[i] != 0)
		{
			shp.push_back(in[i]);
			size *= in[i];
			i++;
		}
		order = shp.size();
		ptns = (double *)MKL_malloc(size * sizeof(double), MKLalignment);
		unit = (int *)MKL_malloc(order * sizeof(double), MKLalignment);
		add_unit = (int *)MKL_malloc(order * sizeof(double), MKLalignment);
		int cumprod = 1;
		unit[order - 1] = 1;
		add_unit[order - 1] = 1;
		for (int i = order - 2; i >= 0; i--)//get unit and add_unit based on shp
		{
			cumprod *= shp.at(i + 1);
			unit[i] = cumprod;
			add_unit[i] = 1;
		}
		IsOrdered = true;
	}

	void tensor::reset(std::array<int, MaxOrder> shape, double *pointer_tensor)
	{
		assert(shape[0] != 0);
		shp.clear();
		size = 1;
		int i = 0;
		while (shape[i] != 0)
		{
			shp.push_back(shape[i]);
			size *= shape[i];
			i++;
		}
		order = shp.size();
		ptns = (double *)MKL_realloc(ptns, size * sizeof(double));
		std::memcpy(ptns, pointer_tensor, size * sizeof(double));
		unit = (int *)MKL_malloc(order * sizeof(double), MKLalignment);
		add_unit = (int *)MKL_malloc(order * sizeof(double), MKLalignment);
		int cumprod = 1;
		unit[order - 1] = 1;
		add_unit[order - 1] = 1;
		for (int i = order - 2; i >= 0; i--)//get unit and add_unit based on shp
		{
			cumprod *= shp.at(i + 1);
			unit[i] = cumprod;
			add_unit[i] = 1;
		}
		IsOrdered = true;
		return;
	}



	void tensor::reset(std::array<int, MaxOrder> in)
	{
		assert(in[0] != 0);
		shp.clear();
		size = 1;
		int i = 0;
		while (in[i] != 0)
		{
			shp.push_back(in[i]);
			size *= in[i];
			i++;
		}
		order = shp.size();
		unit = (int *)MKL_malloc(order * sizeof(double), MKLalignment);
		add_unit = (int *)MKL_malloc(order * sizeof(double), MKLalignment);
		int cumprod = 1;
		unit[order - 1] = 1;
		add_unit[order - 1] = 1;
		for (int i = order - 2; i >= 0; i--)//get unit and add_unit based on shp
		{
			cumprod *= shp.at(i + 1);
			unit[i] = cumprod;
			add_unit[i] = 1;
		}
		IsOrdered = true;
		return;
	}

	void tensor::reset()
	{
		size = 1;
		for (auto i : shp)
		{
			size *= i;
		}
		order = shp.size();
		unit = (int *)MKL_malloc(order * sizeof(double), MKLalignment);
		add_unit = (int *)MKL_malloc(order * sizeof(double), MKLalignment);
		int cumprod = 1;
		unit[order - 1] = 1;
		add_unit[order - 1] = 1;
		for (int i = order - 2; i >= 0; i--)//get unit and add_unit based on shp
		{
			cumprod *= shp.at(i + 1);
			unit[i] = cumprod;
			add_unit[i] = 1;
		}
		IsOrdered = true;
		return;
	}

	void tensor::ini_rand(pwm::Rand &rand)
	{
		ptns = (double *)MKL_realloc(ptns, size * sizeof(double));
		for (int i = 0; i < size; i++)
		{
			ptns[i] = rand.doub();
		}
		return;
	}



	void tensor::ini_sequence()
	{
		ptns = (double *)MKL_realloc(ptns, size * sizeof(double));
		for (int i = 0; i < size; i++)
		{
			ptns[i] = double(i + 1.0) / 10.0;
		}
		return;
	}


	tensor& tensor::operator=(const tensor& in)
	{
		assert(this != &in);
		shp = in.shp;
		size = in.size;
		order = in.order;
		IsOrdered = in.IsOrdered;
		unit = (int *)MKL_realloc(unit, order * sizeof(int));
		std::memcpy(unit, in.unit, order * sizeof(int));
		add_unit = (int *)MKL_realloc(add_unit, order * sizeof(int));
		std::memcpy(add_unit, in.add_unit, order * sizeof(int));
		ptns = (double *)MKL_realloc(ptns, size * sizeof(double));
		std::memcpy(ptns, in.ptns, size * sizeof(double));
		return *this;
	}

	//locate by calculating pos += tns_pos[] * unit[]
	//permute indices outside loop
	void tensor::permute1(int fst, ...)
	{
		int *perm_idx = (int *)MKL_malloc(order * sizeof(int), MKLalignment);
		va_list p_arg;
		va_start(p_arg, fst);
		perm_idx[0] = fst;
		int cnt = 1;
		while ((perm_idx[cnt++] = va_arg(p_arg, int)) != 0);
		va_end(p_arg);

		int *temp_shp = (int *)MKL_malloc(order * sizeof(int), MKLalignment);
		int *temp_unit = (int *)MKL_malloc(order * sizeof(int), MKLalignment);
		int *tns_pos = (int *)MKL_calloc(order, sizeof(int), MKLalignment);//initialize by zeros

		for (int i = 0; i < order; i++)
		{
			temp_shp[i] = shp.at(perm_idx[i] - 1) - 1;//2,3,4 -> 2,4,3 -> 1,3,2
			add_unit[i] = temp_unit[i] = unit[perm_idx[i] - 1];
		}
		for (int i = 0; i < order; i++)
		{
			for (int j = 1; j < order - i; j++)
			{
				add_unit[i] -= ((temp_shp[i + j])*temp_unit[i + j]);
			}
		}

		double *out = (double *)MKL_malloc(size * sizeof(double), MKLalignment);
		int pos = 0;
		for (int i = 0; i < size; i++)
		{
			pos = 0;
			for (int j = 0; j < order; j++)
			{
				pos += tns_pos[j] * temp_unit[j];
			}
			out[i] = ptns[pos];

			//tns_pos increase
			tns_pos[order - 1]++;
			for (int j = order - 1; j > 0; j--)
			{
				if (tns_pos[j] > temp_shp[j])
				{
					tns_pos[j] = 0;
					tns_pos[j - 1]++;
				}
				else
				{
					break;
				}
			}
		}

		shp.at(0) = temp_shp[0] + 1;
		unit[order - 1] = 1;
		add_unit[order - 1] = 1;
		int cumprod = 1;
		for (int i = order - 2; i >= 0; i--)
		{
			cumprod *= shp.at(i + 1) = temp_shp[i + 1] + 1;
			unit[i] = cumprod;
			add_unit[i] = 1;
		}

		MKL_free(perm_idx);
		MKL_free(temp_shp);
		MKL_free(temp_unit);
		MKL_free(tns_pos);
		MKL_free(ptns);
		ptns = out;

		return;
	}

	//locate by calculating pos += tns_pos[] * unit []
	//but permute indices inside loop
	void tensor::permute2(int fst, ...)
	{
		int *perm_idx = (int *)MKL_malloc(order * sizeof(int), MKLalignment);
		va_list p_arg;
		va_start(p_arg, fst);
		perm_idx[0] = fst;
		int cnt = 1;
		while ((perm_idx[cnt++] = va_arg(p_arg, int)) != 0);
		va_end(p_arg);

		int *new_shp = (int *)MKL_malloc(order * sizeof(int), MKLalignment);
		int *old_shp = (int *)MKL_malloc(order * sizeof(int), MKLalignment);
		int *tns_pos = (int *)MKL_calloc(order, sizeof(int), MKLalignment);//initialize by zeros
		int *tns_pos_ = (int *)MKL_malloc(order * sizeof(int), MKLalignment);

		for (int i = 0; i < order; i++)
		{
			new_shp[i] = shp.at(perm_idx[i] - 1);//2,3,4 -> 2,4,3
			old_shp[i] = shp.at(i) - 1;//2,3,4 -> 1,2,3
		}
		int cumprod = 1;
		unit[order - 1] = 1;
		for (int i = order - 2; i >= 0; i--)
		{
			cumprod *= new_shp[i + 1];
			//new_shp[i + 1]--;
			unit[i] = cumprod;
		}
		//new_shp[order - 1]--;
		double *out = (double *)MKL_malloc(size * sizeof(double), MKLalignment);
		int pos = 0;
		for (int i = 0; i < size; i++)
		{
			pos = 0;
			for (int k = 0; k < order; k++)
			{
				tns_pos_[k] = tns_pos[perm_idx[k] - 1];
			}
			for (int j = 0; j < order; j++)
			{
				pos += tns_pos_[j] * unit[j];
			}
			out[pos] = ptns[i];
			//out[i] = ptns[pos];

			//tns_pos increase
			tns_pos[order - 1]++;
			for (int j = order - 1; j > 0; j--)
			{
				if (tns_pos[j] > old_shp[j])
				{
					tns_pos[j] = 0;
					tns_pos[j - 1]++;
				}
				else
				{
					break;
				}
			}
		}

		shp.clear();
		shp.assign(new_shp, new_shp + order);

		MKL_free(perm_idx);
		MKL_free(new_shp);
		MKL_free(old_shp);
		MKL_free(tns_pos);
		MKL_free(tns_pos_);
		MKL_free(ptns);
		ptns = out;

		return;
	}

	//locate by calculating pos += add_unit[]
	void tensor::permute3(int fst, ...)
	{
		int *perm_idx = (int *)MKL_malloc(order * sizeof(int), MKLalignment);
		va_list p_arg;
		va_start(p_arg, fst);
		perm_idx[0] = fst;
		int cnt = 1;
		while ((perm_idx[cnt++] = va_arg(p_arg, int)) != 0);
		va_end(p_arg);

		int *temp_shp = (int *)MKL_malloc(order * sizeof(int), MKLalignment);
		int *temp_unit = (int *)MKL_malloc(order * sizeof(int), MKLalignment);
		int *tns_pos = (int *)MKL_calloc(order, sizeof(int), MKLalignment);//initialize by zeros

		for (int i = 0; i < order; i++)
		{
			temp_shp[i] = shp.at(perm_idx[i] - 1) - 1;//2,3,4 -> 2,4,3 -> 1,3,2
			add_unit[i] = temp_unit[i] = unit[perm_idx[i] - 1];
		}
		for (int i = 0; i < order; i++)
		{
			for (int j = 1; j < order - i; j++)
			{
				add_unit[i] -= ((temp_shp[i + j])*temp_unit[i + j]);
			}
		}

		double *out = (double *)MKL_malloc(size * sizeof(double), MKLalignment);
		int pos = 0;
		int mark = 0;
		for (int i = 0; i < size; i++)
		{
			out[i] = ptns[pos];

			//tns_pos increase
			tns_pos[order - 1]++;
			mark = order - 1;
			for (int j = order - 1; j > 0; j--)
			{
				if (tns_pos[j] > temp_shp[j])
				{
					tns_pos[j] = 0;
					tns_pos[j - 1]++;
					mark = j - 1;
				}
				else
				{
					break;
				}
			}
			pos += add_unit[mark];
		}

		shp.at(0) = temp_shp[0] + 1;
		unit[order - 1] = 1;
		add_unit[order - 1] = 1;
		int cumprod = 1;
		for (int i = order - 2; i >= 0; i--)
		{
			cumprod *= shp.at(i + 1) = temp_shp[i + 1] + 1;
			unit[i] = cumprod;
			add_unit[i] = 1;
		}

		MKL_free(perm_idx);
		MKL_free(temp_shp);
		MKL_free(temp_unit);
		MKL_free(tns_pos);
		MKL_free(ptns);
		ptns = out;

		return;
	}

	//************************************
	// Method:    permute4
	// FullName:  pwm::tensor::permute4
	// Access:    public 
	// Returns:   void
	// Qualifier:
	// Parameter: int fst
	// Parameter: ...
	// P.S:       parallel version of permute3
	//************************************
	void tensor::permute4(int fst, ...)
	{
		int *perm_idx = (int *)MKL_malloc(order * sizeof(int), MKLalignment);
		va_list p_arg;
		va_start(p_arg, fst);
		perm_idx[0] = fst;
		int cnt = 1;
		while ((perm_idx[cnt++] = va_arg(p_arg, int)) != 0);
		va_end(p_arg);

		int *temp_shp = (int *)MKL_malloc(order * sizeof(int), MKLalignment);
		int *temp_unit = (int *)MKL_malloc(order * sizeof(int), MKLalignment);
		for (int i = 0; i < order; i++)
		{
			temp_shp[i] = shp.at(perm_idx[i] - 1) - 1;//2,3,4 -> 2,4,3 -> 1,3,2
			add_unit[i] = temp_unit[i] = unit[perm_idx[i] - 1];//so far temp_unit=12,1,4
		}
		for (int i = 0; i < order; i++)
		{
			for (int j = 1; j < order - i; j++)
			{
				add_unit[i] -= ((temp_shp[i + j])*temp_unit[i + j]);//add_unit=1,-7,4
			}
		}

		shp.at(0) = temp_shp[0] + 1;
		unit[order - 1] = 1;
		int cumprod = 1;
		for (int i = order - 2; i >= 0; i--)
		{
			cumprod *= shp.at(i + 1) = temp_shp[i + 1] + 1;
			unit[i] = cumprod;
		}



		double *out = (double *)MKL_malloc(size * sizeof(double), MKLalignment);
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

			int *tns_pos = (int *)MKL_calloc(order, sizeof(int), MKLalignment);//initialize by zeros
			int pos = 0, _pos = 0;
			int mark = 0;
			_pos = thread_start;
			for (int i = 0; i < order; i++)
			{
				tns_pos[i] = _pos / unit[i];//get tns_pos in form of 2,4,3(actual 1,3,2)
				_pos -= tns_pos[i] * unit[i];
				pos += tns_pos[i] * temp_unit[i];//get pos += tns_pos[] * temp_unit[]; temp_unit is 12,1,4
			}

			for (int i = thread_start; i < thread_finish; i++)//main loop
			{
				out[i] = ptns[pos];

				//tns_pos increase
				tns_pos[order - 1]++;
				mark = order - 1;
				for (int j = order - 1; j > 0; j--)
				{
					if (tns_pos[j] > temp_shp[j])
					{
						tns_pos[j] = 0;
						tns_pos[j - 1]++;
						mark = j - 1;
					}
					else
					{
						break;
					}
				}
				pos += add_unit[mark];
			}
			MKL_free(tns_pos);
		}

		for (int i = 0; i < order; i++)
		{
			add_unit[i] = 1;
		}

		MKL_free(perm_idx);
		MKL_free(temp_shp);
		MKL_free(temp_unit);
		MKL_free(ptns);
		ptns = out;

		return;
	}


	//************************************
	// Method:    permute
	// FullName:  pwm::tensor::permute
	// Access:    public 
	// Returns:   void
	// Qualifier:
	// Parameter: int * perm_idx
	// P.S:       based on permute4, accept array as input
	//************************************
	void tensor::permute(int *perm_idx)
	{
		int *temp_shp = (int *)MKL_malloc(order * sizeof(int), MKLalignment);
		int *temp_unit = (int *)MKL_malloc(order * sizeof(int), MKLalignment);
		for (int i = 0; i < order; i++)
		{
			temp_shp[i] = shp.at(perm_idx[i] - 1) - 1;//2,3,4 -> 2,4,3 -> 1,3,2
			add_unit[i] = temp_unit[i] = unit[perm_idx[i] - 1];//so far temp_unit=12,1,4
		}
		for (int i = 0; i < order; i++)
		{
			for (int j = 1; j < order - i; j++)
			{
				add_unit[i] -= ((temp_shp[i + j])*temp_unit[i + j]);//add_unit=1,-7,4
			}
		}

		shp.at(0) = temp_shp[0] + 1;
		unit[order - 1] = 1;
		int cumprod = 1;
		for (int i = order - 2; i >= 0; i--)
		{
			cumprod *= shp.at(i + 1) = temp_shp[i + 1] + 1;
			unit[i] = cumprod;
		}

		double *out = (double *)MKL_malloc(size * sizeof(double), MKLalignment);
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

			int *tns_pos = (int *)MKL_calloc(order, sizeof(int), MKLalignment);//initialize by zeros
			int pos = 0, _pos = 0;
			int mark = 0;
			_pos = thread_start;
			for (int i = 0; i < order; i++)
			{
				tns_pos[i] = _pos / unit[i];//get tns_pos in form of 2,4,3(actual 1,3,2)
				_pos -= tns_pos[i] * unit[i];
				pos += tns_pos[i] * temp_unit[i];//get pos += tns_pos[] * temp_unit[]; temp_unit is 12,1,4
			}

			for (int i = thread_start; i < thread_finish; i++)//main loop
			{
				out[i] = ptns[pos];

				//tns_pos increase
				tns_pos[order - 1]++;
				mark = order - 1;
				for (int j = order - 1; j > 0; j--)
				{
					if (tns_pos[j] > temp_shp[j])
					{
						tns_pos[j] = 0;
						tns_pos[j - 1]++;
						mark = j - 1;
					}
					else
					{
						break;
					}
				}
				pos += add_unit[mark];
			}
			MKL_free(tns_pos);
		}

		for (int i = 0; i < order; i++)
		{
			add_unit[i] = 1;
		}

		MKL_free(temp_shp);
		MKL_free(temp_unit);
		MKL_free(ptns);
		ptns = out;

		return;
	}

	//based on permute4
	//************************************
	// Method:    permute
	// FullName:  pwm::tensor::permute
	// Access:    public 
	// Returns:   void
	// Qualifier:
	// Parameter: int * perm_idx
	// Parameter: tensor & B
	// P.S:       reload of permute
	// Further optimize might come from 1) scilab::permute 2) OpenCV::Mat::reshape 3) matlab::permute
	// 4) Numpy 5) https://github.com/solomonik/ctf which is actually Cyclops mentioned by Bruce
	// 6) http://www.csc.lsu.edu/~gb/TCE/ which seems to be part of NWChem
	// 7) shuffle algorithm 8)https://github.com/smorita/mptensor which comes from nowhere
	//************************************
	void tensor::permute(int *perm_idx, tensor &B)
	{
		int *temp_shp = (int *)MKL_malloc(order * sizeof(int), MKLalignment);
		int *temp_unit = (int *)MKL_malloc(order * sizeof(int), MKLalignment);
		int *temp_add_unit = (int *)MKL_malloc(order * sizeof(int), MKLalignment);
		B.unit = (int *)MKL_realloc(B.unit, order * sizeof(int));
		B.add_unit = (int *)MKL_realloc(B.add_unit, order * sizeof(int));
		for (int i = 0; i < order; i++)
		{
			temp_shp[i] = shp.at(perm_idx[i] - 1) - 1;//2,3,4 -> 2,4,3 -> 1,3,2
			temp_add_unit[i] = temp_unit[i] = unit[perm_idx[i] - 1];//so far temp_unit=12,1,4
		}
		for (int i = 0; i < order; i++)
		{
			for (int j = 1; j < order - i; j++)
			{
				temp_add_unit[i] -= ((temp_shp[i + j])*temp_unit[i + j]);//add_unit=1,-7,4
			}
		}

		B.unit[order - 1] = 1;
		int cumprod = 1;
		for (int i = order - 2; i >= 0; i--)
		{
			cumprod *= (temp_shp[i + 1] + 1);
			B.unit[i] = cumprod;
		}

		double *out = (double *)MKL_malloc(size * sizeof(double), MKLalignment);
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

			int *tns_pos = (int *)MKL_calloc(order, sizeof(int), MKLalignment);//initialize by zeros
			int pos = 0, _pos = 0;
			int mark = 0;
			_pos = thread_start;
			for (int i = 0; i < order; i++)
			{
				tns_pos[i] = _pos / B.unit[i];//get tns_pos in form of 2,4,3(actual 1,3,2)
				_pos -= tns_pos[i] * B.unit[i];
				pos += tns_pos[i] * temp_unit[i];//get pos += tns_pos[] * temp_unit[]; temp_unit is 12,1,4
			}

			for (int i = thread_start; i < thread_finish; i++)//main loop
			{
				out[i] = ptns[pos];

				//tns_pos increase
				tns_pos[order - 1]++;
				mark = order - 1;
				for (int j = order - 1; j > 0; j--)
				{
					if (tns_pos[j] > temp_shp[j])
					{
						tns_pos[j] = 0;
						tns_pos[j - 1]++;
						mark = j - 1;
					}
					else
					{
						break;
					}
				}
				pos += temp_add_unit[mark];
			}
			MKL_free(tns_pos);
		}

		B.shp = shp;
		for (int i = 0; i < order; i++)
		{
			B.add_unit[i] = 1;
			B.shp.at(i) = shp.at(perm_idx[i] - 1);
		}

		B.size = size;
		B.order = order;
		B.IsOrdered = IsOrdered;

		MKL_free(temp_shp);
		MKL_free(temp_unit);
		MKL_free(temp_add_unit);
		MKL_free(B.ptns);
		B.ptns = out;

		return;
	}

	void tensor::permute(std::array<int, MaxOrder> perm_idx)
	{
		int *temp_shp = (int *)MKL_malloc(order * sizeof(int), MKLalignment);
		int *temp_unit = (int *)MKL_malloc(order * sizeof(int), MKLalignment);
		for (int i = 0; i < order; i++)
		{
			temp_shp[i] = shp.at(perm_idx[i] - 1) - 1;//2,3,4 -> 2,4,3 -> 1,3,2
			add_unit[i] = temp_unit[i] = unit[perm_idx[i] - 1];//so far temp_unit=12,1,4
		}
		for (int i = 0; i < order; i++)
		{
			for (int j = 1; j < order - i; j++)
			{
				add_unit[i] -= ((temp_shp[i + j])*temp_unit[i + j]);//add_unit=1,-7,4
			}
		}

		shp.at(0) = temp_shp[0] + 1;
		unit[order - 1] = 1;
		int cumprod = 1;
		for (int i = order - 2; i >= 0; i--)
		{
			cumprod *= shp.at(i + 1) = temp_shp[i + 1] + 1;
			unit[i] = cumprod;
		}

		double *out = (double *)MKL_malloc(size * sizeof(double), MKLalignment);
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

			int *tns_pos = (int *)MKL_calloc(order, sizeof(int), MKLalignment);//initialize by zeros
			int pos = 0, _pos = 0;
			int mark = 0;
			_pos = thread_start;
			for (int i = 0; i < order; i++)
			{
				tns_pos[i] = _pos / unit[i];//get tns_pos in form of 2,4,3(actual 1,3,2)
				_pos -= tns_pos[i] * unit[i];
				pos += tns_pos[i] * temp_unit[i];//get pos += tns_pos[] * temp_unit[]; temp_unit is 12,1,4
			}

			for (int i = thread_start; i < thread_finish; i++)//main loop
			{
				out[i] = ptns[pos];

				//tns_pos increase
				tns_pos[order - 1]++;
				mark = order - 1;
				for (int j = order - 1; j > 0; j--)
				{
					if (tns_pos[j] > temp_shp[j])
					{
						tns_pos[j] = 0;
						tns_pos[j - 1]++;
						mark = j - 1;
					}
					else
					{
						break;
					}
				}
				pos += add_unit[mark];
			}
			MKL_free(tns_pos);
		}

		for (int i = 0; i < order; i++)
		{
			add_unit[i] = 1;
		}

		MKL_free(temp_shp);
		MKL_free(temp_unit);
		MKL_free(ptns);
		ptns = out;

		return;
	}

	void tensor::permute(std::array<int, MaxOrder> perm_idx, tensor &B)
	{
		int *temp_shp = (int *)MKL_malloc(order * sizeof(int), MKLalignment);
		int *temp_unit = (int *)MKL_malloc(order * sizeof(int), MKLalignment);
		int *temp_add_unit = (int *)MKL_malloc(order * sizeof(int), MKLalignment);
		B.unit = (int *)MKL_realloc(B.unit, order * sizeof(int));
		B.add_unit = (int *)MKL_realloc(B.add_unit, order * sizeof(int));
		for (int i = 0; i < order; i++)
		{
			temp_shp[i] = shp.at(perm_idx[i] - 1) - 1;//2,3,4 -> 2,4,3 -> 1,3,2
			temp_add_unit[i] = temp_unit[i] = unit[perm_idx[i] - 1];//so far temp_unit=12,1,4
		}
		for (int i = 0; i < order; i++)
		{
			for (int j = 1; j < order - i; j++)
			{
				temp_add_unit[i] -= ((temp_shp[i + j])*temp_unit[i + j]);//add_unit=1,-7,4
			}
		}

		B.unit[order - 1] = 1;
		int cumprod = 1;
		for (int i = order - 2; i >= 0; i--)
		{
			cumprod *= (temp_shp[i + 1] + 1);
			B.unit[i] = cumprod;
		}

		double *out = (double *)MKL_malloc(size * sizeof(double), MKLalignment);
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

			int *tns_pos = (int *)MKL_calloc(order, sizeof(int), MKLalignment);//initialize by zeros
			int pos = 0, _pos = 0;
			int mark = 0;
			_pos = thread_start;
			for (int i = 0; i < order; i++)
			{
				tns_pos[i] = _pos / B.unit[i];//get tns_pos in form of 2,4,3(actual 1,3,2)
				_pos -= tns_pos[i] * B.unit[i];
				pos += tns_pos[i] * temp_unit[i];//get pos += tns_pos[] * temp_unit[]; temp_unit is 12,1,4
			}

			for (int i = thread_start; i < thread_finish; i++)//main loop
			{
				out[i] = ptns[pos];

				//tns_pos increase
				tns_pos[order - 1]++;
				mark = order - 1;
				for (int j = order - 1; j > 0; j--)
				{
					if (tns_pos[j] > temp_shp[j])
					{
						tns_pos[j] = 0;
						tns_pos[j - 1]++;
						mark = j - 1;
					}
					else
					{
						break;
					}
				}
				pos += temp_add_unit[mark];
			}
			MKL_free(tns_pos);
		}

		B.shp = shp;
		for (int i = 0; i < order; i++)
		{
			B.add_unit[i] = 1;
			B.shp.at(i) = shp.at(perm_idx[i] - 1);
		}

		B.size = size;
		B.order = order;
		B.IsOrdered = IsOrdered;

		MKL_free(temp_shp);
		MKL_free(temp_unit);
		MKL_free(temp_add_unit);
		MKL_free(B.ptns);
		B.ptns = out;

		return;
	}

	void tensor::times(double in)
	{
#pragma omp parallel for
		for (int i = 0; i < size; i++)
		{
			ptns[i] *= in;
		}
		return;
	}
}