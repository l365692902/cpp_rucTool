#include <array>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <ctime>
#include "Common.hpp"
#include "CanoFinMPS.hpp"
#include "TensorClass.hpp"
#include "TensorContraction.hpp"
#include "LargestEigenvector.hpp"
#include "mkl.h"

namespace pwm
{

	void CanoTransform(int Tensor_cnt,
		std::array<tensor *, MaxNumTensor> Left_in,
		std::array<tensor *, MaxNumTensor> Right_in,
		std::array<tensor *, MaxNumTensor> MPS_io,
		std::array<tensor *, MaxNumTensor> GamSVD_out)
	{
		//int Tensor_cnt = 0;
		//while (MPS_io[Tensor_cnt] != 0)
		//{
		//	Tensor_cnt++;
		//}
		tensor __x, U, L, V, P, Q;//P and Q can be saved
		for (int i = 0; i < Tensor_cnt - 1; i++)
		{
			pwm::tensorContract(*Left_in[i + 1], *Right_in[i], __x);
			__x.svd(1, { {&U,&L,&V} });
			*GamSVD_out[i] = L;
			pwm::getNorm2(GamSVD_out[i]->size, GamSVD_out[i]->ptns);
			vdInvSqrt(GamSVD_out[i]->size, GamSVD_out[i]->ptns, L.ptns);//L^-0.5
			V.permute({ {2,1} });
			tensorContractDiag('N', V, L.ptns, P);
			tensorContract(*Right_in[i], P, P);
			U.permute({ {2,1} });
			tensorContractDiag('N', L.ptns, U, Q);
			tensorContract(Q, *Left_in[i + 1], Q);
			tensorContract(*MPS_io[i], P, *MPS_io[i]);
			tensorContract(Q, *MPS_io[i + 1], *MPS_io[i + 1]);
		}
		pwm::tensorContract(*Left_in[0], *Right_in[Tensor_cnt - 1], __x);
		__x.svd(1, { {&U,&L,&V} });
		*GamSVD_out[Tensor_cnt - 1] = L;
		pwm::getNorm2(GamSVD_out[Tensor_cnt - 1]->size, GamSVD_out[Tensor_cnt - 1]->ptns);
		vdInvSqrt(GamSVD_out[Tensor_cnt - 1]->size, GamSVD_out[Tensor_cnt - 1]->ptns, L.ptns);
		V.permute({ {2,1} });
		tensorContractDiag('N', V, L.ptns, P);
		tensorContract(*Right_in[Tensor_cnt - 1], P, P);
		U.permute({ {2,1} });
		tensorContractDiag('N', L.ptns, U, Q);
		tensorContract(Q, *Left_in[0], Q);
		tensorContract(*MPS_io[Tensor_cnt - 1], P, *MPS_io[Tensor_cnt - 1]);
		tensorContract(Q, *MPS_io[0], *MPS_io[0]);

		return;
	}

	void NormalizeMPS(int Tensor_cnt,
		std::array<tensor *, MaxNumTensor> MPS_io,
		std::array<tensor *, MaxNumTensor> GamSVD_in,
		std::array<double *, MaxNumTensor> Coef_out)
	{
		//int Tensor_cnt = 0;
		//while (MPS_io[Tensor_cnt] != 0)
		//{
		//	Tensor_cnt++;
		//}
		tensor __x;
		for (int i = 0; i < Tensor_cnt - 1; i++)
		{
			pwm::tensorContractDiag('N', *MPS_io[i], GamSVD_in[i]->ptns, __x);
			pwm::tensorContract(*MPS_io[i], __x, 2, __x);
			*Coef_out[i] = std::sqrt(__x.ptns[0] / GamSVD_in[(i ? i : Tensor_cnt) - 1]->ptns[0]);
			MPS_io[i]->times(1.0 / *Coef_out[i]);
		}

		pwm::tensorContractDiag('N', GamSVD_in[Tensor_cnt - 2]->ptns, *MPS_io[Tensor_cnt - 1], __x);
		pwm::tensorContract(2, *MPS_io[Tensor_cnt - 1], __x, __x);
		*Coef_out[Tensor_cnt - 1] = std::sqrt(__x.ptns[0] / GamSVD_in[Tensor_cnt - 1]->ptns[0]);
		MPS_io[Tensor_cnt - 1]->times(1.0 / *Coef_out[Tensor_cnt - 1]);
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
			Tensor_cnt++;
		}
		int Mbond = T_io[0]->shp.front();
		//std::cout << "sizeof(pwm::tensor) " << sizeof(pwm::tensor) << std::endl;
		//std::cout << "sizeof(pwm::tensor *) " << sizeof(pwm::tensor *) << std::endl;
		std::array<tensor *, MaxNumTensor> Left_env{};
		std::array<tensor *, MaxNumTensor> Right_env{};
		std::array<double *, MaxNumTensor> coef{};
		for (int i = 0; i < Tensor_cnt; i++)
		{
			Left_env[i] = new tensor();
			Right_env[i] = new tensor();
			coef[i] = new double;
		}
		//getEnv('L', T_io, Left_env);//checked
		//getEnv('R', T_io, Right_env);//checked
		tensor __x(T_io[0]->shp.front(), T_io[0]->shp.front(), 0);
		pwm::Rand myrand(static_cast<unsigned long long>(time(0)));
		__x.ini_rand(myrand);
		pwm::largestEigenvalue('L', T_io, __x, 1e-13, 500, Left_env, coef);
		__x.renew({ {T_io[Tensor_cnt - 1]->shp.back(),T_io[Tensor_cnt - 1]->shp.back()} });
		__x.ini_rand(myrand);
		pwm::largestEigenvalue('R', T_io, __x, 1e-13, 500, Right_env, coef);
		tensor __u, __l, __v;
		for (int i = 0; i < Tensor_cnt; i++)
		{
			Left_env[i]->svd(1, { {&__u,&__l,&__v} });
			vdSqrt(__l.size, __l.ptns, __l.ptns);
			pwm::tensorContractDiag('N', __l.ptns, __v, *Left_env[i]);
			Right_env[i]->svd(1, { {&__u,&__l,&__v} });
			vdSqrt(__l.size, __l.ptns, __l.ptns);
			pwm::tensorContractDiag('N', __u, __l.ptns, *Right_env[i]);
		}
		CanoTransform(Tensor_cnt, Left_env, Right_env, T_io, Gam_out);
		NormalizeMPS(Tensor_cnt, T_io, Gam_out, coef_out);

		for (int i = 0; i < Tensor_cnt; i++)
		{
			Left_env[i]->~tensor();
			Right_env[i]->~tensor();
			delete coef[i];
		}
		return;
	}

}