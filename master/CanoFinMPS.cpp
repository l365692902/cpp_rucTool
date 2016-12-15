#include <array>
#include <iostream>
#include "Common.hpp"
#include "CanoFinMPS.hpp"
#include "TensorClass.hpp"
#include "TensorContraction.hpp"
#include "LargestEigenvector.hpp"
#include "mkl.h"

namespace pwm
{

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
		std::cout << "test, here" << std::endl;
		return;
	}

}