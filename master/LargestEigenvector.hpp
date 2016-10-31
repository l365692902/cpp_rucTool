#pragma once
#include "TensorClass.hpp"

namespace pwm
{
	void largestEigenvalue(
		std::array<tensor *, MaxNumTensor> T_in,
		tensor &x_in,
		double Converge_in,
		std::array<tensor *, MaxNumTensor> y_out,
		std::array<double *, MaxNumTensor>lam_out
		);
	void applyMPS(char L_R, tensor &in, tensor &x, double &norm);
	double normalize(int size, double *&x);
	double getNorm(int size, double *in);

}