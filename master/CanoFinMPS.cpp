#include <array>
#include <iostream>
#include <algorithm>
#include "Common.hpp"
#include "CanoFinMPS.hpp"
#include "TensorClass.hpp"
#include "TensorContraction.hpp"
#include "LargestEigenvector.hpp"
#include "mkl.h"

namespace pwm
{

	void getEnv(char L_R,
		std::array<tensor *, MaxNumTensor> T_in,
		std::array<tensor *, MaxNumTensor> Env_out)
	{
		int Tensor_cnt = 0, Mbond = 1;
		while (T_in[Tensor_cnt] != 0)
		{
			Tensor_cnt++;
		}
		Mbond = T_in[0]->shp.front();

		double *Identity = (double *)MKL_malloc(Mbond*sizeof(double), MKLalignment);
		std::fill_n(Identity, Mbond, 1.0);

		tensor *__x = new tensor(Mbond, Mbond, 0);
		std::memset(__x->ptns, 0, Mbond*Mbond*sizeof(double));
		cblas_dcopy(Mbond, Identity, 1, __x->ptns, Mbond + 1);

		std::array<int, MaxNumTensor> order;

		switch (L_R)
		{
		case 'L':
			*Env_out[0] = *__x;
			for (int i = 1; i < Tensor_cnt; i++)
			{
				pwm::tensorContract(*__x, *T_in[i], *__x);

			}
			break;
		case 'R':
			*Env_out[Tensor_cnt - 1] = *__x;
			break;
		default:
			break;
		}

		return;
	}

	// A_____ B_____ C_____ D_____ 
	//    |      |      |      |   
	//                             
	//  __|__  __|__  __|__  __|__ 
	// A      B      C      D
	//    2      
	// 1__|__3   
	void CanoFinMPS(
		std::array<pwm::tensor *, MaxNumTensor> T_io, 
		std::array<pwm::tensor *, MaxNumTensor> Gam_out, 
		std::array<double *, MaxNumTensor> coef_out)
	{
		int Tensor_cnt = 0;
		while (T_io[Tensor_cnt] != 0)
		{
			std::cout << T_io[Tensor_cnt] << std::endl;
			Tensor_cnt++;
		}
		//std::cout << "sizeof(pwm::tensor) " << sizeof(pwm::tensor) << std::endl;
		//std::cout << "sizeof(pwm::tensor *) " << sizeof(pwm::tensor *) << std::endl;
		std::array<tensor *, MaxNumTensor> Left_env{};
		std::array<tensor *, MaxNumTensor> Right_env{};
		std::array<double *, MaxNumTensor> redun_lam{};

		for (int i = 0; i < Tensor_cnt; i++)
		{
			Left_env[i] = new tensor();
			Right_env[i] = new tensor();
		}

		pwm::applyMPSsOnIdentity('L', T_io, Left_env);
		pwm::applyMPSsOnIdentity('R', T_io, Right_env);

		for (int i = 0; i < Tensor_cnt; i++)
		{

		}
		return;
	}

}