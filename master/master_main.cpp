#include "mkl.h"
#include "pwm.hpp"
#include "pwm_legacy.hpp"
#include <iostream>
#include "omp.h"
#include <cstring>

#include <cstdarg>

int main()
{
	pwm::tensor T1(4, 3, 4, 0), T2(4, 4, 0), T3, T11, T22, T33;
	T1.ini_sequence();
	T2.ini_sequence();
	T22 = T2;
	T33 = T2;
	double res1, res2, res3;
	res1 = cblas_dnrm2(T2.size, T2.ptns, 1);
	std::cout << res1 << std::endl;
	cblas_dscal(T2.size, 1 / res1, T2.ptns, 1);

	res2 = pwm::normalize(T22.size, T22.ptns);
	std::cout << res2 << std::endl;

	res3 = pwm::getNorm(T33.size, T33.ptns);
	std::cout << res2 << std::endl;

	return 0;
}


int main_test6()
{
	//pwm::tensor T1(500, 1000, 1500, 0), T2;
	pwm::tensor T1(20, 30, 40, 50, 0), T2(50, 30, 20, 40, 0), T3;
	pwm::tensor T11, T22, T33;
	double start, finish;
	pwm::Rand myrand(2);
	//T1.ini_rand(myrand);
	//T2.ini_rand(myrand);
	T1.ini_sequence();
	T2.ini_sequence();
	////////////////////////////
	T11 = T1;
	T22 = T2;
	T11 << 3; T11 << 3;//3,4,2,5
	T22 << 4; T22 << 3;//3,4,5,2
	T11.merge(1, 2);
	T22.merge(1, 2);
	pwm::product('{', T11, T22, T33);//2,5,5,2
	T33 << 2;//5,2,5,2//standard
	////////////////////////////
	pwm::tensorContract(T1, T2, "aijb", "cidj", "bacd", T3);
	////////////////////////////
	std::cout << "diff: " << pwm::diff('N', T3, T33) << std::endl;

	pwm::tensorContract(T1, T2, "aijb", "cidj", "bacd", T2);
	std::cout << "diff: " << pwm::diff('N', T2, T33) << std::endl;

	T11 = T1;
	pwm::tensorContract(T1, T11, "aijb", "cijd", "cbad", T2);//standard
	pwm::tensorContract(T1, T1, "aijb", "cijd", "cbad", T3);
	std::cout << "diff: " << pwm::diff('N', T2, T3) << std::endl;

	pwm::tensorContract(T1, T1, "aijb", "cijd", "cbad", T1);
	std::cout << "diff: " << pwm::diff('N', T2, T1) << std::endl;

	T11.~tensor();
	T11 = T1;
	std::cout << "assignment after deconstruct" << std::endl;
	return 0;
}

int main_test4()
{
	int cnt;
	int *permA = NULL, *permB = NULL, *permC = NULL;
	pwm::resolve_perm("eifjt", "efjgyn", "tiygn", permA, permB, cnt, permC);
	std::cout << cnt << std::endl;
	return 0;
}

int main_test7()
{
	pwm::tensor T1(13000, 13000, 0), T_result;
	pwm::tensor T2(500, 1000, 1500, 0);
	pwm::tensor T3(140, 160, 180, 200, 0);
	double t_start, t_finish;
	for (int k = 1; k <= 24; k++)
	{
		omp_set_dynamic(0);
		omp_set_num_threads(k);
		MKL_Set_Num_Threads(k);
#pragma omp parallel
		{
			if (omp_get_thread_num() == 0)
			{
				std::cout << "omp: " << omp_get_num_threads() << " mkl: " << MKL_Get_Max_Threads() << std::endl;
			}
		}
		t_start = dsecnd();
		pwm::tensorContract(T1, T1, "ab", "bc", "ca", T_result);
		t_finish = dsecnd();
		std::cout << "13000*13000_outplace " << t_finish - t_start << std::endl;
		t_start = dsecnd();
		pwm::tensorContract(T2, T2, "jia", "jib", "ba", T_result);
		t_finish = dsecnd();
		std::cout << "500*1000*15000_outplace " << t_finish - t_start << std::endl;
		t_start = dsecnd();
		pwm::tensorContract(T3, T3, "abij", "cdij", "dbca", T_result);
		t_finish = dsecnd();
		std::cout << "140*160*180*200_outplace " << t_finish - t_start << std::endl;
	}
}

int main_test5()
{
	pwm::tensor T1(13000, 13000, 0), T_result;
	double t_start, t_finish;
	t_start = dsecnd();
	pwm::tensorContract(T1, T1, "ab", "bc", "ca", T_result);
	t_finish = dsecnd();
	std::cout << "13000*13000_outplace " << t_finish - t_start << std::endl;
	t_start = dsecnd();
	pwm::tensorContract(T1, T1, "ab", "bc", "ca", T1);
	t_finish = dsecnd();
	std::cout << "13000*13000_in_place " << t_finish - t_start << std::endl;
	pwm::tensor T2(500, 1000, 1500, 0);
	t_start = dsecnd();
	pwm::tensorContract(T2, T2, "jia", "jib", "ba", T_result);
	t_finish = dsecnd();
	std::cout << "500*1000*15000_outplace " << t_finish - t_start << std::endl;
	t_start = dsecnd();
	pwm::tensorContract(T2, T2, "jia", "jib", "ba", T2);
	t_finish = dsecnd();
	std::cout << "500*1000*15000_in_place " << t_finish - t_start << std::endl;
	pwm::tensor T3(140, 160, 180, 200, 0);
	t_start = dsecnd();
	pwm::tensorContract(T3, T3, "abij", "cdij", "dbca", T_result);
	t_finish = dsecnd();
	std::cout << "140*160*180*200_outplace " << t_finish - t_start << std::endl;
	t_start = dsecnd();
	pwm::tensorContract(T3, T3, "abij", "cdij", "dbca", T3);
	t_finish = dsecnd();
	std::cout << "140*160*180*200_in_place " << t_finish - t_start << std::endl;

	return 0;
}