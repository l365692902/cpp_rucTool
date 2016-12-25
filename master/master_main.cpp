#include "mkl.h"
#include "pwm.hpp"
#include "pwm_legacy.hpp"
#include <iostream>
#include "omp.h"
#include <cstring>

#include <cstdarg>

int main_largest2()
{
	pwm::Rand myrand(2);
	pwm::tensor TA(3, 2, 3, 0), TB(3, 2, 3, 0), TC(3, 2, 3, 0);
	pwm::tensor Ga, Gb, Gc;
	pwm::tensor x(3, 3, 0);
	double coefa, coefb, coefc;
	TA.ini_rand(myrand);
	TB.ini_rand(myrand);
	TC.ini_rand(myrand);
	x.ini_rand(myrand);
	pwm::largestEigenvalue('L', { {&TA,&TB,&TC} }, x, 1e-13, 500, { {&Ga,&Gb,&Gc} }, { {&coefa,&coefb,&coefc} });

	std::cout << std::endl;
	return 0;
}

int main()
{
	pwm::Rand myrand(2);
	pwm::tensor TA(3, 2, 3, 0), TB(3, 2, 3, 0), TC(3, 2, 3, 0);
	pwm::tensor Ga, Gb, Gc;
	double coefa, coefb, coefc;
	//TA.ini_sequence();
	//TB.ini_sequence();
	TA.ini_rand(myrand);
	TB.ini_rand(myrand);
	TC.ini_rand(myrand);
	TA.permute({ {3,2,1} });
	TB.permute({ {3,2,1} });
	TC.permute({ {3,2,1} });
	pwm::CanoFinMPS({ {&TA,&TB,&TC} }, { {&Ga,&Gb,&Gc} }, { {&coefa,&coefb,&coefc} });
	pwm::tensor temp;
	pwm::tensorContractDiag('N', Ga.ptns, TB, temp);
	pwm::tensorContract(2, temp, TB, temp);//should be equal to Gb
	pwm::tensorContract(2, TB, TB, temp);
	//pwm::getDiff('N',10)

	vdSqrt(TA.size, TA.ptns, TA.ptns);
	return 0;
}

int main_largest()
{
	int side = 40, mid = 3;
	pwm::tensor TA(side, mid, side, 0), TB(side, mid, side, 0), Tx(side, side, 0);
	pwm::tensor TAt(side, mid, side, 0), TBt(side, mid, side, 0), Txt(side, side, 0);
	pwm::tensor ya(side, side, 0), yb(side, side, 0);
	TA.ini_sequence();
	TB.ini_sequence();
	Tx.ini_sequence();
	TAt = TA;
	TBt = TB;
	Txt = Tx;
	double res1, res2, res1t, res2t;
	//pwm::applyOneMPS('L', TB, Tx, res1);
	pwm::largestEigenvalue('R', { { &TA, &TB } }, Tx, 1e-13, 1, { { &ya, &yb } }, { { &res1, &res2 } });

	pwm::tensorContract(TBt, Txt, "abi", "ci", "abc", Txt);
	pwm::tensorContract(TBt, Txt, "aij", "bij", "ab", Txt);
	res1t = pwm::getNorm2(Txt.size, Txt.ptns);
	std::cout << pwm::getDiff('N', 16, yb.ptns, Txt.ptns) << std::endl;

	//pwm::tensorContract(TAt, Txt, "abi", "ci", "abc", Txt);
	//pwm::tensorContract(TAt, Txt, "aij", "bij", "ab", Txt);
	pwm::tensorContract(TAt, Txt, Txt);
	pwm::tensorContract(Txt, TAt, 2, Txt);
	//pwm::tensorContract(Txt, TBt, "ai", "ibc", "abc", Txt);
	//pwm::tensorContract(TBt, Txt, "ija", "ijb", "ab", Txt);
	res2t = pwm::getNorm2(Txt.size, Txt.ptns);

	std::cout << pwm::getDiff('N', side*side, ya.ptns, Txt.ptns) << std::endl;
	return 0;
}