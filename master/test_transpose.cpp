


#include <iostream>
#include <stdlib.h>
#include <vector>
#include <time.h>
#include <omp.h>

using namespace std ;

extern "C" {
	double omp_get_wtime() ;
	void omp_set_num_threads(int NThread) ;
}

void transpose_2() {
	const int d1 = 13000, d2 = 13000 ;
	const int N = d1 * d2 ;
	int Times = 10 ;
	double * A = new double[N] ; // A(d1,d2)
	double * B = new double[N] ; // B(d2,d1)

	int k, j, i ;
	int idx ;

	for ( i = 0; i < N ; i ++) {
		B[i] = i ;
	}

	double t0 = omp_get_wtime() ;

	vector<int> id2(d1) ;
#pragma omp parallel for
	for ( i = 0; i < d1; i ++ ) {
		id2[i] = i * d2 ;
	}
#pragma omp parallel default(shared) private(idx, i, j)
	for (k = 0; k < Times; k ++) {
		i = 0 ;
		j = 0 ;

#pragma omp for
		for ( idx = 0; idx < N; idx ++ ) {
			A[idx] = B[j + id2[i]] ;
			i ++ ;
			if ( i >= d1 ) {
				i = 0 ;
				j ++ ;
			}
		}
	}


	delete [] B ;
	double t1 = omp_get_wtime() ;
	delete [] A ;

	cout << t1 - t0 << " s, " << endl ;
}

int main() {
	
	int NThread = 24;
	for(int i=1;i<=24;i++)
	{
		omp_set_num_threads(i) ;
		transpose_2() ;
	}

	return 1 ;
}
