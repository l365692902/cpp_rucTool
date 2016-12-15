#pragma once
#include <array>
#include "TensorClass.hpp"

namespace pwm
{
	void CanoFinMPS(
		std::array<tensor *, MaxNumTensor> T_io,
		std::array<tensor *, MaxNumTensor> Gam_out,
		std::array<double *, MaxNumTensor> coef_out);
}
