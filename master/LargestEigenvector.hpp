#pragma once
#include "TensorClass.hpp"

namespace pwm
{
	void largestEigenvalue(
		char L_R,
		std::array<tensor *, MaxNumTensor> T_in,
		tensor &x_in,
		double Converge_in,
		int MaxIter_in,
		std::array<tensor *, MaxNumTensor> y_out,
		std::array<double *, MaxNumTensor> lam_out
	);
	void applyOneMPS(char L_R, tensor &in, tensor &x_io, double &norm);
	void applyMPSsOnIdentity(char L_R,
		std::array<tensor *, MaxNumTensor> T_in,
		std::array<tensor *, MaxNumTensor> Env_out);
	double getNorm(int size, double *in);
	double getNorm2(int size, double *in);
	int getIdamax(int size, double *in);
	double getMax(int size, double *in);
	double getDiff(char L_R_A_N, int size, double *L, double *R);

}