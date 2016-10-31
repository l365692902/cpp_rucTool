#include <string>
#include "TensorClass.hpp"

namespace pwm
{
	double diff(char mod, int size, double *L, double *R);
	double diff(char mod, tensor &L, tensor &R);
	
	void Shw_Mtx(std::string tag, double *in, int rows, int cols);
	void product(char mod, tensor &A, tensor &B, tensor &C);

}