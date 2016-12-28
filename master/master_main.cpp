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
	
	x = Ga;
	pwm::applyOneMPS('L', TA, x, coefa);//=Gb
	pwm::applyOneMPS('L', TB, x, coefa);//=Gc
	pwm::applyOneMPS('L', TC, x, coefa);//=Ga

	pwm::largestEigenvalue('R', { {&TA,&TB,&TC} }, x, 1e-13, 500, { {&Ga,&Gb,&Gc} }, { {&coefa,&coefb,&coefc} });

	x = Ga;
	pwm::applyOneMPS('R', TA, x, coefa);//=Gc
	pwm::applyOneMPS('R', TC, x, coefa);//=Gb
	pwm::applyOneMPS('R', TB, x, coefa);//=Ga

	std::cout << std::endl;
	return 0;
}

int main()
{
	pwm::Rand myrand(2);
	//pwm::tensor TA(3, 2, 4, 0), TB(4, 2, 5, 0), TC(5, 2, 3, 0);
	pwm::tensor TA(4, 2, 3, 0), TB(5, 2, 4, 0), TC(3, 2, 5, 0);
	pwm::tensor Ga, Gb, Gc;
	double coefa, coefb, coefc;
	//TA.ini_sequence();
	//TB.ini_sequence();
	TA.ini_rand(myrand);
	TB.ini_rand(myrand);
	TC.ini_rand(myrand);
	TA.permute({ {3,2,1} });//3,2,4
	TB.permute({ {3,2,1} });//4,2,5
	TC.permute({ {3,2,1} });//5,2,3
	pwm::CanoFinMPS({ {&TA,&TB,&TC} }, { {&Ga,&Gb,&Gc} }, { {&coefa,&coefb,&coefc} });
	pwm::tensor temp;
	pwm::tensorContractDiag('N', TB, Gb.ptns, temp);
	pwm::tensorContract(temp, TB, 2, temp);//=Ga

	pwm::tensorContractDiag('N', Ga.ptns, TB, temp);
	pwm::tensorContract(2, temp, TB, temp);//=Gb right

	pwm::tensorContractDiag('N', TA, Ga.ptns, temp);
	pwm::tensorContract(TA, temp, 2, temp);//=Gc

	pwm::tensorContractDiag('N', Gc.ptns, TA, temp);
	pwm::tensorContract(2, temp, TA, temp);//=Ga right

	pwm::tensorContractDiag('N', TC, Gc.ptns, temp);
	pwm::tensorContract(TC, temp, 2, temp);//=Gb

	pwm::tensorContractDiag('N', Gb.ptns, TC, temp);
	pwm::tensorContract(2, temp, TC, temp);//=Gc right

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