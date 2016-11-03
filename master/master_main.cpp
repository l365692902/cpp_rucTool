#include "mkl.h"
#include "pwm.hpp"
#include "pwm_legacy.hpp"
#include <iostream>
#include "omp.h"
#include <cstring>

#include <cstdarg>

int main()
{
	pwm::tensor TA(4, 3, 4, 0), TB(4, 3, 4, 0), Tx(4, 4, 0);
	pwm::tensor TAt(4, 3, 4, 0), TBt(4, 3, 4, 0), Txt(4, 4, 0);
	pwm::tensor ya(4, 4, 0), yb(4, 4, 0);
	TA.ini_sequence();
	TB.ini_sequence();
	Tx.ini_sequence();
	TAt = TA;
	TBt = TB;
	Txt = Tx;
	double res1, res2, res1t, res2t;
	//pwm::applyOneMPS('L', TB, Tx, res1);
	pwm::largestEigenvalue({ { &TA, &TB } }, Tx, 1e-13, 1, { { &ya, &yb } }, { { &res1, &res2 } });

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

	std::cout << pwm::getDiff('N', 16, ya.ptns, Txt.ptns) << std::endl;
	return 0;
}