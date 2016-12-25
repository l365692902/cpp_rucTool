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

3) prevent permute(or any other operation) applied on empty tensor, maybe add a empty flag in tensorclass?

4) now I strongly think, I should use "tensor" through this whole project, discard "double *"

5) questions about CanoFinMPS
    * what does CanoFinMPS mean?
    * should getLeftEnv, getRightEnv, CanoTransform, NormalizeMPS, these functions be available to outside canoFinMPS function?
	* there seems to be some discrepancies between matlab code and the graph on white board!?

6) a must be modified issue, in matlab, [U,S,V]=svd(A) means A = U * S * V', so I need one more transpose on V.

7) I need to reconsider the structre of project, I think should put all one-tensor operators below TensorClass, and intergrate entire vml into tensor etc.

8) svd might give exact "0" rather than a tiny digit, thus it can not be inversed. is this a problem?

## log

* 25th dec 2016, finished canoFinMPS, last question, do we need symmetry in normalization?

* 18th dec 2016, tested TensorClassSVD, small-scale test passed, below is the test code, using diff() to tell if they are equal
```
 	pwm::tensor TA(17, 13, 0);
	TA.ini_sequence();
	pwm::tensor U, L, Vt;
	double *lU = NULL, *lL = NULL, *lVt = NULL;
	TA.svd(1, { {0,&L,0} });
	//Vt.permute({ {2,1} });//should equal to lU
	//U.permute({ {2,1} });//should equal to lVt
	//TA.permute({ {2,1} });
	TA.legacy_svd(1, 13, lU, lL, lVt);
	return 0;
```


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