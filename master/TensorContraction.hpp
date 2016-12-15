#pragma once
#include <array>
#include <string>
#include "TensorClass.hpp"

namespace pwm
{
	void tensorContract(tensor &A, tensor &B, int *permA, int *permB, int merg, int *permC, tensor &C);
	void tensorContract(tensor &A, tensor &B, std::string idxA,
		std::string idxB, std::array<int, MaxOrder> permC, tensor &C);
	void tensorContract(tensor &A, tensor &B, std::string idxA, std::string idxB, std::string idxC, tensor &C);
	void tensorContract(tensor &A, tensor &B, tensor &C);
	void tensorContract(tensor &A, tensor &B, int merg, tensor &C);
	void tensorContract(tensor &A, int merg, tensor &B, tensor &C);
	void tensorContract(int merg, tensor &A, tensor &B, tensor &C);


	void resolve_perm(std::string idxA, std::string idxB, int *&permA, int *&permB, int &com_cnt);
	void resolve_perm(std::string idxA, std::string idxB, std::string idxC, int *&permA, int *&permB, int &com_cnt, int *&permC);

	void tensorContractDiag(char N_R, tensor &A, double *diag, tensor &C);
	void tensorContractDiag(char N_R, double *diag, tensor &B, tensor &C);

}