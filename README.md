# cpp_rucTool

## note
1) in MKL userguide they say BLAS level1 and level2 are threaded for intel Core 2 Duo and Intel Core i7 processors; I am not sure if these routines will experience performance decreasing or not on linux clusters which are most likely not core cpu.

	be aware of 
    * ?axpy in TensorContraction.cpp::tensorContractDiag

    * ?copy in LargestEigenvector.cpp::applyMPSsOnIdentity

	test them when shanghai cluster is ready.
2) threaded singular value decomposition routines are: ?gebrd, ?bdsqr  
and a number of other LAPACK rountines, based on threaded routines, make effective use of parallelism: ?gesvd  
does it mean I can use dgesvd without worries?

## log
* 16th dec 2016, modified or added
    * CanoFinMPS
        * all
    * LargestEigenvector
        * applyMPSsOnIdentity, useless, to be deleted
    * TensorClass
        * reset
    * TensorContraction
        * tensorContractDiag
    * TensorClass_SVD
        * all 

	but none of them are tested. 


* 15th dec 2016, applyMPSsOnIdentity, pretty reduntant
    * identity matrix can be omitted, just find a neat way to generate one for output.
	* now, the way generate a identity matrix is, generate a vector, fill it with '1', copy to a matrix. there must be a better way.
	* actually, I found this function useless.