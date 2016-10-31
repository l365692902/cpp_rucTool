#pragma once
#include "Common.hpp"

namespace pwm
{
	//|----------------------------------------------------------------------------------------|
	//|random number generator recommended by "numerical recipes", period¡Ö3.138x10^57		   |
	//|How to use:																			   |
	//|create a object of Ran, like Ran myran(9); 9 in braket is random seed				   |
	//|Ran myran(static_cast<unsigned long long>(time(0))); local time as random seed		   |
	//|random seed can be any LL type except 4101842887655102017, (although I doubt about it)  |
	//|application:																			   |
	//|myran.int64(); return 64 bits unsigned type, around 10^20, 0~18446744073709551615	   |
	//|myran.int32(); return 32 bits unsigned type, around 10^10, 0~4294967295				   |
	//|myran.doub();  return double type between 0.0~1.0, 17 decimals						   |
	//|return integer between 1~n(include 1 and n):											   |
	//|1+myran.int64()%n;																	   |
	//|PS: to ensure quality of random number, use only one Ran object,						   |
	//|    when passing Ran object, use reference(&) to ensure continuity					   |
	//|version: 05/02/2015 1st Thu															   |
	//|----------------------------------------------------------------------------------------|
	struct Rand{
		unsigned long long u, v, w;
		Rand(unsigned long long j) :v(4101842887655102017LL), w(1){
			u = j^v; int64();
			v = u; int64();
			w = v; int64();
		}
		inline unsigned long long int64(){
			u = u * 2862933555777941757LL + 7046029254386353087LL;
			v ^= v >> 17; v ^= v << 31; v ^= v >> 8;
			w = 4294957665U * (w & 0xffffffff) + (w >> 32);
			unsigned long long x = u ^ (u << 21); x ^= x >> 35; x ^= x << 4;
			return (x + v) ^ w;
		}
		inline double doub(){ return 5.42101086242752217E-20 * int64(); }
		inline unsigned int int32(){ return (unsigned int)int64(); }
	};

}