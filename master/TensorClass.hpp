#pragma once
#include <vector>
#include <array>
#include "Random.hpp"

namespace pwm
{
	class tensor
	{
	public:
		double *ptns;
		std::vector<int> shp;
		int *unit;
		int *add_unit;
		int size;
		int order;
		bool IsOrdered;

		//basic member functions
		~tensor();
		tensor();
		tensor(int fst, ...);
		tensor(std::array<int, MaxOrder> in);
		void ini_rand(pwm::Rand &rand);
		void ini_sequence();
		tensor& operator=(const tensor& in);

		//functional member functions
		void permute1(int fst, ...);
		void permute2(int fst, ...);
		void permute3(int fst, ...);
		void permute4(int fst, ...);
		void permute(int *perm_idx);
		void permute(int *perm_idx, tensor &B);
		void permute(std::array<int, MaxOrder> perm_idx);
		void permute(std::array<int, MaxOrder> perm_idx, tensor &B);
		tensor& permute_assign(int *perm_idx);

		//comparison with legacy
		int calc_shp(int idx_1, int cnt_1);
		void operator<<(int idx);
		void operator>>(int idx);
		void merge(int begin, int end);


	};

}