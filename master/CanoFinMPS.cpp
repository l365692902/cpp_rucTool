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

		double *Identity = (double *)MKL_malloc(Mbond * sizeof(double), MKLalignment);
		std::fill_n(Identity, Mbond, 1.0);

		tensor __x(Mbond, Mbond, 0);
		tensor L, U_V;
		std::memset(__x.ptns, 0, Mbond*Mbond * sizeof(double));
		cblas_dcopy(Mbond, Identity, 1, __x.ptns, Mbond + 1);

		switch (L_R)
		{
		case 'L':
			*Env_out[0] = __x;
			for (int i = 1; i < Tensor_cnt; i++)
			{
				pwm::tensorContract(*Env_out[i - 1], *T_in[i - 1], __x);
				__x.svd(2, { {0,&L,&U_V} });
				L.times(1.0 / L.ptns[0]);
				pwm::tensorContractDiag('N', L.ptns, U_V, *Env_out[i]);
			}
			break;
		case 'R':
			*Env_out[Tensor_cnt - 1] = __x;
			for (int i = Tensor_cnt - 2; i >= 0; i--)
			{
				pwm::tensorContract(*T_in[i + 1], *Env_out[i + 1], __x);
				__x.svd(1, { {&U_V,&L,0} });
				L.times(1.0 / L.ptns[0]);
				pwm::tensorContractDiag('N', U_V, L.ptns, *Env_out[i]);
			}
			break;
		default:
			break;
		}

		MKL_free(Identity);
		return;
	}

	void CanoTransform(std::array<tensor *, MaxNumTensor> Left_in,
		std::array<tensor *, MaxNumTensor> Right_in,
		std::array<tensor *, MaxNumTensor> MPS_io,
		std::array<tensor *, MaxNumTensor> GamSVD_out)
	{
		int Tensor_cnt = 0;
		int Mbond = MPS_io[0]->shp.front();
		while (MPS_io[Tensor_cnt] != 0)
		{
			Tensor_cnt++;
		}
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

		//GamSVD_out[Tensor_cnt - 1]->ptns = (double *)MKL_malloc(Mbond * sizeof(double), MKLalignment);
		//std::fill_n(GamSVD_out[Tensor_cnt - 1]->ptns, Mbond, 1.0 / std::sqrt(Mbond));
		//GamSVD_out[Tensor_cnt - 1]->reset({ {Mbond} });
		return;
	}

	void NormalizeMPS(std::array<tensor *, MaxNumTensor> MPS_io,
		std::array<tensor *, MaxNumTensor> GamSVD_in,
		std::array<double *, MaxNumTensor> Coef_out)
	{
		int Tensor_cnt = 0;
		int Mbond = MPS_io[0]->shp.front();
		while (MPS_io[Tensor_cnt] != 0)
		{
			Tensor_cnt++;
		}
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
		tensor __x(Mbond, Mbond, 0);
		pwm::Rand myrand(static_cast<unsigned long long>(time(0)));
		__x.ini_rand(myrand);
		pwm::largestEigenvalue('L', T_io, __x, 1e-13, 500, Left_env, coef);
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
		CanoTransform(Left_env, Right_env, T_io, Gam_out);
		NormalizeMPS(T_io, Gam_out, coef_out);

		for (int i = 0; i < Tensor_cnt; i++)
		{
			Left_env[i]->~tensor();
			Right_env[i]->~tensor();
			delete coef[i];
		}
		return;
	}

}