
/* CVUT 2016 -  some usefull mathematical methods, mostly from Numerical Recipies in C++ */
/* segmenter v1.4 */

#ifndef maths_c
#define maths_c

#include <cmath>
    /* \brief rounding of doubles. round() is not standard on some systems, thus we have re-implemented it */
    /* double round(double arg); not necessary ? */

#ifdef MISSING_MATHSF_FUNCTIONS
    float fabsf (float x);
    float sqrtf (float x);
    float cosf (float x);
    float sinf (float x);
    float logf (float x);
    float log10f (float x);
#endif



/*! evaluates continued fraction for incomplete beta function by modified
   Lentz's method, used by betai [NR C++ p. 227] */
    float betacf(float a, float b, float x);

/*! returns the incomplete beta function I_X(a,b) [NR C++ p. 227] */
    float betai(float a, float b, float x);

/*! Given a positive-definite symmetric matrix a[1..n][1..n], this
   routine constructs its Cholesky decomposition, A = L L^T.
   On input, only the upper triangle of a need be given; it is not
   modified. The Cholesky factor L is returned  in the lower triangle
   of a, except for its diagonal elements which are returned in p[1..n] */
    void  choldc(float **a, int n, float p[],int ExitOnError=1);

/*! Finds the inversion L^{-1} of the Cholesky factor of
    a = L L^T where a is a positive-definite symmetric matrix.
    a[1..n][1..n] and p[1..n] are input as the output of the choldc.
    The L^{-1} factor is returned in the lower  triangle. MH04 */
    void  cholinv(float **a, int n, float p[]);
   

/*! matrix transposition  MH04 */
    void  matrixTrans(float **a, int n);

/*! Given a Cholesky factor L (lower triangle) in matrix a[1..n][1..n]
    and diagonal elements in p[1..n] , this routine returns
    the original matrix A = L L^T in a.  MH04 */
    void  cholmul(float **a, int n, float p[]);
   
/*! Solves the set of n linear equations ax=b, where a is a
positive-definite symmetric matrix.  a[1..n][1..n] and p[1..n]
are input as the output of the choldc. Only the lower triangle
of a is accessed. b[1..n] is input as the right-hand side vector.
The solution vector is returned in x[1..n]. a,n, and p are not
modified and can be left in place for successive calls with
different right-hand sides b. b is not modified unless you identify
b and x in the calling sequence, which is allowed  */
    void  cholsl(float **a, int n, float p[],float b[], float x[]);
   
/*!  Expand in storage the covariance matrix covar, so as to take
    into account parameters that are being held fixed.
    (For the latter, return zero covariances.) */
template <class TREAL>   // float | double
    void  covsrt(TREAL ** covar, int ma,int lista[], int mfit);

/*! returns the complementary  error function erfc(x) with fractional
   error everywhere less than 1.2e-7 [NR C++ p. 221] */
    float erfcc(float x);

/*! \return returns ln(\Gamma(xx)) for xx>0 [NR C++ p. 214] */
    float gammln(float xx);

/*!  Linear solution by Gauss-Jordan elimination eq. (2.1.1) NRC++/39
    a[1..n][1..n] is the input matrix. b[1..n][1..m] is input
    containing the m right-hand side vectors. On output, a is
    replaced by its matrix inverse, and b is replaced by the
    corresponding set of solution vectors.                     */
template <class TREAL>   // float | double
    int   gaussj(TREAL **a,int n,TREAL **b,int m);
    //int   gaussj2(double **a,int n,double **b,int m);

/*! Given a set of data points x[1..ndat], y[1..ndat] with individual
   standard deviations of y sig[1..ndat] (if not known sig=1 ),
   use \chi^2 minimization to fit for some or all of the coefficients
   a[1..ma] of a function that depends linearly on a, y=sum a_i x afunc_i(x).
   The input array ia[1..ma] indicates by nonzero entries those components
   of a that should be fitted for, and by zero entries those components
   that should be held fixed at their input values. The program returns i
   values for a[1..ma], \chi^2, and the covariance matrix
   covar[1..ma][1..ma]. (Parameters held fixed return zero ovariances.)
   The user supplies a routine funcs(x,afunc,ma) that returns the ma basic
   functions evaluated at x=x in the array afunct[1..ma].
   void (*funcs)();  ANSI: void (*funcs)(float,float *,int); */
template <class TREAL>   // float | double
    void  lfit(float x[],float y[],float sig[],int ndat, float a[],
         int ia[], int ma,TREAL** covar,float* chisq,
         void (*funcs)(float,TREAL [],int) ); /*!< \brief  LS fitting using Normal equations */
    //void  lfit2(float x[],float y[],float sig[],int ndat, float a[],
    //     int ia[], int ma,double** covar,float* chisq,
    //     void (*funcs)(float,double [],int) ); /*!< \brief  The same as 'lfit' only 'covar' matrix is defined as double intead of float to suppres overflowing. */


/*! Given a set of data points x[1..ndata],y[1..ndata] with individual
    standard deviations  sig[1..ndata], use X^2 minimmization to determine
    the coefficients a[1..ma] of the fitting function y=\sum_i a_i * afunc_i(x).
    Here we solve the fitting equations using singular value decomposition of the ndata
    by ma matrix, as in (2.6.). Arrays u[1..ndata][1..ma], v[1..ma][1.ma], and w[1..ma]
    provide workspace on input; on output they define the singular value decomposition,
    and can be used to obtain the covariance matrix. The program returns values
    for the 'ma' fit parameters 'a', and X^2, 'chisq'. The user supplies a routine
    funcs(x,afunc,ma) that returns the 'ma' basis functions evaluated at
    x = 'x' in the array 'afunc[1..ma]. */
    void svdfit(float x[], float y[], float sig[], int ndata, float a[], int ma, float **u, float **v, float w[], float *chisq, void (*funcs)(float, float [], int)); /*! \brief LS fitting using SVD */

inline   float MAXIMUM(float a, float b) {  return(a > b ? a : b); }
inline   float MINIMUM(float a, float b) {  return(a < b ? a : b); }
inline   float SIGN(float a, float b) { return((b) >= 0.0 ? fabs(a) : -fabs(a)); }


/*!  given input arrays x[1..n] and y[1..n]  it returns
    the correlation coefficient r, the significance level at which the null
    hypothesis of zero correlation is disproved (prob whose small value
    indicates a significant correlation), and Fisher's z, whose value can
    be used in further statistical tests [NR C++ p. 638]
*/
    void  pearsn(float x[], float y[], int n, float* r, float* prob, float* z);

/*! Computes (a^2 + b^2)^(-1/2) without destructive underflow or overflow. */
    float PYTHAG(float a, float b);
    float SIGN(float a, float b);

/*!  Singular Value Decomposition (SVD, svdcmp.c  Numerical rec./67).
    Given a matrix a[1..m][1..n], this routine computes its singular
    value decomposition, A=U.W.V^T. The matrix U replaces a on output.
    The diagonal matrix of singular values W is output as a vector
    w[1..n]. The matrix V (not the transpose V^T) is output as
    v[1..n][1..n].
*/
    void  svdcmp(float **a,int m,int n,float *w,float **v); /*!< \brief SVD - singular values decomposition - Numerical Recepises*/

/*! Solves A.X=B for a vector X, where A is specified by the arrays u[1..m][1..n],
w[1..n],v[1..n][1..n] as returned by 'svdcmp'. 'm' and 'n' are the dimensions of a, and will be equal for square matrices. b[1..m] is the input right-hand side. x[1..n] is the output solution vector. No input quantities are destroyed, so the routine may be called sequentially with differrent b's. */
    void svbksb(float **u, float w[], float **v, int m, int n, float b[], float x[]); /*!< \brief Solves linear equations from U W V. Use after eigennumbers zeroing!*/

//! The ipow() function returns base raised to the exp power
    int   ipow(int base,int exp); /*!< \brief Compute root of base by integer number 'exp' */

//! The sqr() function returns base raised to 2
    float sqr(float base); /*!< returns base^2 */


/*! solves the set of lin. eq. A.X=B , a[idxshift..n-1+idxshift][idxshift..n-1+idxshift] LU decomp. of A     */
/*      (set idxshift to 0 or 1)  */
/* on input, indx[idxshift..n-1+idxshift] input permutation vector, b[idxshift,..,n-1+idxshift] on          */
/* input B returns the solution X , a,n not modified                    */
template <class TREAL>   // float | double
    void  lubksb(TREAL **a,int n,int *indx,TREAL b[], const int idxshift=1); // from TBImage

/*! makes LU (not Cholesky!) decomposition of nxn matrix a[idxshift..n+idxshift-1][idxshift..n+idxshift-1]
   (set idxshift to 0 or 1)
  a does not need to be symmetric and PD
  output consits in lower triangular part from L matrix and in upper
  part from U matrix  a=L*U
  indx[idxshift..n+idxshift-1] output vector recording row permutation ,d output +1 (-1)
  if number of row exchanges was even or odd                      */
template <class TREAL>   // float | double
    void  ludcmp(TREAL **a,int n,int *indx,TREAL *d, const int idxshift=1); // from TBImage

/*! double **a matrix inversion
  \param a[idxshift..n-1+idxshift][idxshift..n-1+idxshift] input matrix (set idxshift to 0 or 1)
  \return output inversion matrix                        */
template <class TREAL>   // float | double
    void  minv(TREAL **a,int n, const int idxshift=1); /*!< \brief Computes inverze of symetric matrix */ // from TBImage

/*! returns determinant of the double matrix a[idxshift..n+idxshift-1][idxshift..n+idxshift-1]
 (set idxshift to 0 or 1) */
template <class TREAL>   // float | double
    TREAL determinant(TREAL **a,int n, const int idxshift=1); /*!< returns determinant of the matrix a*/

/*! returns sqrt of abs determinant of the double matrix a[idxshift..n+idxshift-1][idxshift..n+idxshift-1]
 (set idxshift to 0 or 1) */
template <class TREAL>   // float | double
    TREAL determinant_abs05(TREAL **a,int n, const int idxshift=1); /*!< returns sqrt of determinant of the matrix a*/


/*! float matrix test print a[1..n][1..m]
    n<0 lower triangle, m<0 upper triangle */
    void matrixPri(float **a,char* name, int n, int m);
/*! float vector test print a[1..n] */
    void matrixPri(float *a,char* name, int n);
/*! Computing of first moment of matrix ma by the vector with midle value mv. */
/* \return Result is stored in mafm                                                  */
    void  gfirstm(float ***ma,float *mv,float **mafm, int NoOfPlanes, int nrl, int ncl); // firstm from TATransf
/*! Midle value of matrix ma is stored in vector mv */
    void  gmidvalue(float ***ma,float *mv, int NoOfPlanes, int nr, int nc); // midvalue from TATransf

/*! Computing of iversion matrix of matrix a, with size n x n eighnvalues are
 stored in d. Result is in v and used rotations are in nrot.               */
/*! \param input a[1-n][1-n] real symmetric matrix,
   \return ouput d[1-n] unsorted eigenvalues, v[1-n][1-n] normalized
   \note eigenvectors in columns, nrot no. of required Jacobi rotations */
template <class TREAL>   // float | double
    void  jacobi(TREAL **a,int n,TREAL d[],TREAL **v,int *nrot); // from TATransf


    void normalize(float **Arr,int StartRow,int EndRow,int StartColumn,int EndColumn,float AktMinimum,float AktDifference, float NewMinimum,float NewDifference,float NewVariance);
/*! Transformation of float matrix Mat vales to range 0..255        */
    void  gnormreal255(float **Mat, int nr, int nc, int const*const* invMask = 0); // normreal255 from TAPS
/*! Transformation of float matrix Mat vales to range 0..255        */
    void  gnormreal255(float ***Mat, int NoOfPlanes, int nr, int nc, float *norm_shift, float *norm_range, int const*const* invMask = 0); // normreal255 from TATransf
/*! Shift of the values to the original range */
    void  gdenormreal255(float ***Mat, int NoOfPlanes, int nr, int nc, float *norm_shift, float *norm_range, int const*const* invMask = 0); // denormreal255 from TBImage, originally from TSTransf
/*! Shift of the values to the original range */
    void  gdenormreal(float ***Mat, int NoOfPlanes, int nr, int nc, float *norm_shift, float *norm_range, int const*const* invMask = 0); // denormreal255 from TBImage, originally from TSTransf
/*! sets values of a 3D matrix into range 0-255 */
    void  gcutreal255(float ***Mat, int NoOfPlanes, int nr, int nc, int const*const* invMask = 0); // cutreal255 from TBImage

    void hist_equal(float **arr, int nr, int nc, int nrf=0, int nrl=-1, int ncf=0, int ncl=-1);

    void MatHist(float ***mainarr,int NoOfPlanes, int nr, int nc, const char *Coment);
// Perc <0,1>
    void MatHistCut(float ***mainarr, int NoOfPlanes, int nr, int nc, float Perc,const char *Coment,int PrintHistogram = 0);
    void MatHistPrint(float ***mainarr, int NoOfPlanes, int nr, int nc, const char *Coment);
    void MatStat(float ***mainarr,int NoOfPlanes, int nr, int nc, const char *Coment, int const*const* invMask = 0);
    void MatStat(float **mainarr, int nrf, int nrl, int ncf, int ncl, const char *Coment, int const*const* invMask = 0);


    /*! \brief The class collects static structures and routines for random generators*/
    class RandGen{
      protected:
        static int iset;
        static float gset;
        static long ix1,ix2,ix3;
        static float r[98];
        static int iff;
       //XS: shadow variables for context switching
        int _saved; int _iset; float _gset; long _ix1,_ix2,_ix3; float _r[98]; int _iff;
      public:
       //XS: context switching
        void init_ctx(); void save_ctx(); void load_ctx();
        RandGen(int save, int init=1) { init_ctx(); if (save) save_ctx(); if (init) initstatic(); };
        ~RandGen() { if (_saved) load_ctx(); }
       //XS: -----------------
        RandGen(){initstatic();init_ctx();};
/*! uniform random generator between 0 -1. */
        static float ran1(int *idum);
/*! standard (zero mean,unit cov.) Gaussian generator */
/*! Box -Muller method */
        static float gasdev(int *idum);
/*! Initialisation of static variables           */
        static void initstatic();
/** Returns as a floating-point number an integer value that is a random deviate drawn from a
Poisson distribution of mean xm, using ran1(idum) as a source of uniform random deviates.
Numerical recipes - version 2.10 */
        static float poidev(float xm, int *idum);
    };

/*! 1D Gaussian mixture generator */
    float gauss_mix(int *idum, int k, float *p, float *mu, float *sigma);
/*! nD Gaussian mixture generator (n<100) */
    int gauss_mix(int *idum, int k, int n, float *p, float **mean, float ***cov, float *vec);
/*! generator of the correlated 3D Gaussian field
   Y = \mu + L \tilde Y,   \Sigma = L L^T,   379/222
   \tilde Y white Gaussian vector
   mean[lf], cov[lf][lf] input Gaussian parameters */
    /** WARNING: In gauss functions, ncl, nrl indexes are ommited that differs from behaviour of Prog.cpp functions !!!*/
    int gauss(int lf,int nrf,int nrl,int ncf, int ncl,float ***a,float* mean,float** cov,int ExitOnError=1,int Seed=0);
/*! generator of the white Gaussian field */
    int gauss(int nrf,int nrl,int ncf, int ncl,float **a);
/*! discrete distribution generator between 0 - (nlev-1)
   discrete distribution given by the look-up table in prob[]         */
    int dis_gen(float *prob, int nlev, int *idum);


/*! \brief ndimensional Fourier transformation

    isign=1 replaces data[] by its ndim dimensional discrete Fourier tr.
            nn[1,...,ndim] int array containing the lengths of each dim.
            !! must be power of 2 !! data[] real array each complex elem.
            is stored [..,real,imaginary,...] and the rightmost index of
            array increases most rapidly as one proceeds along data[]
            i.e. for 2D rowwise storing, data[2*number of elements]
    isign=-1 data[] replaced by its inverse tr. x product of the lenghts
            of all dimensions !!!!
    subscripts range always from 1,...,Ni !!                             */
    void fourn(float data[],int nn[],int ndim,int isign); /*!< \brief Computes n-dimensional FFT - Numerical Recepises */

/*! \brief ndimensional Fourier transformation

    isign=1 replaces data[] by its ndim dimensional discrete Fourier tr.
            nn[1,...,ndim] int array containing the lengths of each dim.
            !! must be power of 2 !! data[] real array each complex elem.
            is stored [..,real,imaginary,...] and the rightmost index of
            array increases most rapidly as one proceeds along data[]
            i.e. for 2D rowwise storing, data[2*number of elements]
    isign=-1 data[] replaced by its inverse tr. x product of the lenghts
            of all dimensions !!!!
    subscripts range always from 1,...,Ni !!                             */
    void fourn(float data[],int nn1,int nn2,int isign); /*!< \brief Computes n-dimensional FFT - Numerical Recepises */

/*! Given a three-dimensional real array data[1. .nnl] [1. .nn2] [1. .nn3] (where nnl =
1 for the case of a logically two-dimensional array), this routine returns (for isign=1) the
complex fast Fourier transform as two complex arrays: On output, data contains the zero and
positive frequency values of the third frequency component, while
speq[1..nn1] [1..2*nn2] contains the Nyquist critical frequency values of the
third frequency component. First (and second) frequency components are stored for zero,
positive, and negative frequencies, in standard wrap-around order. See text for description
of how complex values are arranged. For isign=-1, the inverse transform (times nnl*nn2*nn3/2
as a constant multiplicative factor) is performed, with output data (viewed as a real array)
deriving from input data (viewed as complex) and "speq". The dimensions nn1, nn2, nn3 must always
be integer powers of 2.  */
    void rlft3(float ***data, float **speq, unsigned long nn1, unsigned long nn2, unsigned long nn3, int isign); /*!< \brief Function computes FFT (uses routine "fourn") -  Numerical Recepises */
    /*! \brief Creates real and imaginary image of FFT coef. with the lowest frequncy in the center*/

/*! Function enlarge computed FFT spectrum to whole image (the FFT function "rlft3" gives just only half of image with the lowest frequences in the corners). Thanks to spectrum symetry the function creates both whole images of real and imaginary part of spectra with  the lowest frequency in the middle of the image. Parametr "direct" determines if we want to do this (1) or inverse proces (0) - to create one image from two (real and imaginary) images and shifts the lowest frequencies back to the corners. JF. */
    void FFTCoefWraparound(float ***data, float **coefRe, float **coefIm, int nr, int nc, int direct);
//  void FFTCoefWraparound(float ***data, TBImage3 *coefRe, TBImage3 *coefIm, int nr, int nc, int direct);

/*! Function erase the circle part of spectra "data" (obtained from "rlft3")
  specified by diameter "pixelsLeft". If variable "lowPass" is (1)-lowpass filter, ie.
all values outside the circle are erased, if (0)-highpass filter, ie. all values in the circle are erased. JF*/
    /*! \brief Cuts specified part of FFT spectrum */
    void FFTImageFilter(float ***data, int pixelsLeft, int nr, int nc, int lowPass);

/*! Function computes amplitude fourier spectrum of "image" spectral layer
defined by "isp" and returns it back in "spectrum" JF */
    /*! \brief compute fourier amplitude spectrum of given image layer defined by "isp" */
    void FFTamplitude(float ***image, float **spectrum, int nr, int nc, int isp);

/*! (rho,theta) hough transformation for finding of lines parameters in edges in binary
  image. Returns accumulator image of size(rhos x thetas) JF */
    /*! \brief Returns accumulator matrix (image) computed by Hough transformation */
    void HoughTransform(float **binary, float **hough_space, int nr, int nc, int thetas, int rhos);


//    void eigs_nopivot(float **a, int n, float wr[], float wi[]);

/*! given eigenvalues d[1,..,n] and eigenvectors v[][1,..,n] from jacobi
   this routine sorts the eigenvalues into descending order and
   rearranges the eigenvectors of v correspondingly */
template <class TREAL>   // float | double
    void eigsrt(TREAL d[],TREAL **v,int n); /*!< \brief Sorts eigennumbers according to their size - Numerical Recepises */

/*! function sorts array of floats "data" from interval <"l","r"> */
    void quickSort(int l,int r,float *data); /*!< \brief Fast routine for sorting of 1 dimensional float array of numbers */

/*! function sorts array of floats "data" from interval <"l","r">
  returns moreover position index of sorted values in unsorted data. */
    void quickSort(int l,int r,float *data,int *pos); /*!< \brief Fast routine for sorting of 1 dimensional float array of numbers and it gives number position in unsorted array in "pos"*/

/*! function rounds float number to integer */
    int round(float co); /*!< \brief Function simply rounds float numbers */

    void MatrixMultipl(float **ma1, int nr1,int nc1,float **ma2, int nr2, int nc2, float **result);

/*!  Matrix multiplication  [result]=[ma1]*[ma2] where "nr1","nr2", "nc1","nc2"
    are numbers of rows and columns in matrices [ma1],[ma2].
    Result is stored in matrix [result] */
    void MatrixMultipl(float **ma1,int nr1,int nc1,float **ma2,int nc2,float **result);

/*!  Matrix multiplication  [result]=[ma1]*[ma2] where "nr1","nr2", "nc1","nc2"
    are numbers of rows and columns in matrices [ma1],[ma2].
    Result is stored in matrix [result] */
    //! \brief Matrix multiplication with progress outprint
    void MatrixMultiplPrintProgress(float **ma1,int nr1,int nc1,float **ma2,int nr2,int nc2,float **result);

/*!  Matrix multiplication  [result]=[ma]^T*[ma] where "nr","nc"
    are numbers of rows and columns in matrix [ma].
    Result is stored in matrix [result] */
    //! \brief computin symmetric matrix with progress outprint
    void SymMatrixMultiplPrintProgress(float **ma, int nr, int nc, float **result);

/*!  Triangular Matrix multiplication [result]=[ma1]*[ma2] where "nr1", "nc1",
    "nr2","nc2" are numbers of rows and columns in matrices [ma1],[ma2].
    if nr1<0 ma1 lower triangular
       nc1<0 ma1 upper triangular
       nr2<0 ma2 lower triangular
       nc2<0 ma2 upper triangular
    Result is stored in matrix [result] */
    void MatrixTrMultipl(float **ma1,int nr1,int nc1,float **ma2,int nr2,
                 int nc2, float **result);

/*! Used by mrqmin to evaluate the linearized fitting matrix alpha, and vector beta as in (15.5.8) and calculate x2. */
    void mrqcof(float **x, float y[], float sig[], int ndata, float a[], int ia[], int ma, float **alpha, float beta[], float *chisq, void (*funcs)(float **, int,  float [], float *, float [], int)); /*!< \brief Levenberg-Marquardt non linear parameters estimation routine - Numerical recepises */

/*! Levenberg-Marquardt method, attempting to reduce the value x2 of a fit between a set of data
points x[1..ndata], y[1..ndata] with individual standard deviations sig[1..ndata],
and a nonlinear function dependent on ma coefficients a[1..ma]. The input array ia[1..ma]
indicates by nonzero entries those components of a that should be fitted for, and by zero
entries those components that should be held fixed at their input values. The program
returns current best-fit values for the parameters a[1..ma], and x2 = chisq. The arrays
covar[1..ma][1..ma], alpha[1..ma][1..ma] are used as working space during most iterations.
Supply a routine funcs(x,a,yfit,dyda,ma) that evaluates the fitting function yfit,
and its derivatives dyda[l..ma] with respect to the fitting parameters a at x. On
the first call provide an initial guess for the parameters a, and set alamda<0 for
initialization (which then sets alamda=.001). If a step succeeds chisq becomes smaller
and alamda decreases by a factor of 10. If a step fails alamda grows by a factor of 10.
You must call this routine repeatedly until convergence is achieved. Then, make one final
call with alamda=0, so that covar[1..ma][1..ma] returns the covariance matrix, and alpha
the curvature matrix. (Parameters held fixed will return zero covariances.) */
    void mrqmin(float **x, float y[], float sig[], int ndata, float a[], int ia[], int ma, float **covar, float **alpha, float *chisq, void (*funcs)(float **, int, float [], float *, float [], int), float *alamda); /*!< \brief Levenberg-Marquardt non linear parameters estimation routine (uses routine "mrqcof" - Numerical recepises */

/*! \brief Kullback-Leiber distance of two distributions 'A' and 'B' of 'n' bins */
    /*! \brief Kullback-Leiber distance of two distributions 'A' and 'B' of 'n' bins */
    float distKullbackLeiber(float *A, float *B, int n);

/*! \brief  Jeffrey distance of two distributions 'A' and 'B' of 'n' bins */
    /*! \brief Jeffrey distance of two distributions 'A' and 'B' of 'n' bins */
    float distJeffrey(float *A, float *B, int n);

    //##### COLOURSPACE CONVERSION ###############################################

/*! \brief Conversion from RGB colour space into CIE XYZ
  for Observer. = 2degree, Illuminant = D65 */
    /*! \brief Conversion from RGB into CIE XYZ (D65) color space */
    int RGB2XYZ(float *R, float *G, float *B);
/*! \brief Conversion from CIE XYZ colour space into RGB
  for Observer. = 2degree, Illuminant = D65 */
    /*! \brief Conversion from CIE XYZ (D65) into RGB color space */
    int XYZ2RGB(float *X, float *Y, float *Z);

/*! \brief Conversion from RGB colour space into
  LogLu'v' (not L*u*v*!) according to [Ward98]
  for Observer. = 2degree, Illuminant = D65 */
    int RGB2LogLuv(float *R, float *G, float *B);
/*! \brief Conversion from LogLu'v' (not L*u*v*)  colour space according to [Ward98] into
  RGB colourspace for Observer. = 2degree, Illuminant = D65 */
    int LogLuv2RGB(float *L, float *u, float *v);
/*! \brief Conversion from RGB to YUV (also YCrCb)  colour space */
    int RGB2YUV(float *R, float *G, float *B);
/*! \brief Conversion from YUV (also YCrCb) to RGB colour space */
    int YUV2RGB(float *Y, float *U, float *V);

/*! \brief Conversion from RGB colour space into
  LogLu'v' (not L*u*v*!) according to [Ward98]
  for Observer. = 2degree, Illuminant = D65 */
int RGB2LogLuv(const float RGB[], float Luv[]);
/*! \brief Conversion from LogLu'v' (not L*u*v*)  colour space according to [Ward98] into
  RGB colourspace for Observer. = 2degree, Illuminant = D65 */
int LogLuv2RGB(const float Luv[], float RGB[]);
/*! \brief Conversion from RGB to YUV (also YCrCb)  colour space */
int RGB2YUV(const float RGB[], float YUV[]);
/*! \brief Conversion from YUV (also YCrCb) to RGB colour space */
int YUV2RGB(const float YUV[], float RGB[]);

/*! \brief Conversion from CIE XYZ 1976 colour space into
  perceptualy uniform CIE Lab space for Observer. = 2degree, Illuminant = D65 */
    /*! \brief Conversion from CIE XYZ (D65) into CIE LAB (D65) color space */
    int XYZ2Lab(float *X, float *Y, float *Z);
/*! \brief Conversion from perceptually uniform CIE Lab 1976 colour space into
   CIE XYZ space for Observer. = 2degree, Illuminant = D65 */
    /*! \brief Conversion from CIE LAB (D65) into CIE XYZ (D65) color space */
    int Lab2XYZ(float *L, float *a, float *b);

    int CIE76_L_from_Y(double Y, double Yn, double * L);
    int CIE76_Y_from_L(double L, double Yn, double * Y);
    int XYZ76_Lab(double  X,  double Y,  double Z,
                double  Xn, double Yn, double Zn,
                double *L,  double *a, double *b,
                double *h_ab,          double *Chroma_ab);
    int Lab76_XYZ(double  L,  double a,  double b,
                double  Xn, double Yn, double Zn,
                double  *X, double *Y, double *Z);
// compute CIE1976 Total Difference DE, CIE1976 a,b Chroma Difference DC, CIE1976 a,b Hue Difference DH, CIELab 2000 Total Color Difference DEexp
// values marked _samp are the unknown ("sample") color, _std are the reference ("standard") color
    void Lab_diff(double L_samp, double a_samp, double b_samp,
                  double L_std, double a_std, double b_std,
                  double *DE, double *DC, double *DH, double *DEexp );

    int CIE00_L_from_Y(double Y, double Yn, double * L);
    int CIE00_Y_from_L(double L, double Yn, double * Y);
    int XYZ00_Lab(double  X,  double Y,  double Z,
                double  Xn, double Yn, double Zn,
                double *L,  double *a, double *b,
                double *h_ab,          double *Chroma_ab);
    int Lab00_XYZ(double  L,  double a,  double b,
                double  Xn, double Yn, double Zn,
                double  *X, double *Y, double *Z);

    void HSV2RGB( float h, float s, float v, float *r, float *g, float *b );
/** Conversion from RGB to opponent colour space:
   O1: (R-G)/sqrt(2), O2: (R+G-2B)/sqrt(6), O3: (R+G+B)/sqrt(3)*/
    int RGB2Opponent( float r, float g, float b, float *o1, float *o2, float *o3 );

    /*! \brief Pixelwise image conversion from RGB int CIE XYZ (D65) colourspace
    "actualPlane" specifies which RGB tripple should be converted into CIE XYZ */
    int imgRGB2XYZ(float ***img, int nr, int nc, int actualPlane=0);
    /*! \brief Pixelwise image conversion from CIE XYZ (D65) into RGB colourspace
    "actualPlane" specifies which CIE XYZ tripple should be converted into RGB */
    int imgXYZ2RGB(float ***img, int nr, int nc, int actualPlane=0);
    /*! \brief Pixelwise image conversion from RGB int CIE Lab (D65) colourspace
    "actualPlane" specifies which RGB tripple should be converted into CIE Lab */
    int imgRGB2Lab(float ***img, int nr, int nc, int actualPlane=0);
    /*! \brief Pixelwise image conversion from CIE Lab (D65) into RGB colourspace
    "actualPlane" specifies which CIE Lab tripple should be converted into RGB */
    int imgLab2RGB(float ***img, int nr, int nc, int actualPlane=0);

    /*! \brief Pixelwise image conversion from RGB int CIE Lab (whitepoint - D65) colourspace
      "actualPlane" specifies which RGB tripple should be converted into CIE Lab
      PV: version with CIE Lab 2000 */
    int imgRGB2Lab2000(float ***img, int nr, int nc, int actualPlane=0);
    /*! \brief Pixelwise image conversion from CIE Lab (whitepoint - D65) to RGB colourspace
      "actualPlane" specifies which CIE Lab tripple should be converted into RGB
      if cutValues ise set, it cuts values into [0.255] interval*/
    int imgLab20002RGB(float ***img, int nr, int nc, int actualPlane=0, int cutValues=1);

/** \brief Pixelwise image conversion from RGB to opponent colour space:
   0: (R-G)/sqrt(2), 1: (R+G-2B)/sqrt(6), 2: (R+G+B)/sqrt(3)*/
    int imgRGB2Opponent(float ***img, int nr, int nc, int actualPlane=0);

    /*! \brief Computes L2 distance between two colours in CIELab space */
    float distCIELab(float L1, float a1, float b1, float L2, float a2, float b2);

//#### 1D and 2D SPLINES #############################################

/*! \brief Given arrays x[1..n] and y[1..n] containing a tabulated function,i.e. y_i=f(x_i), with
x1<x2<..<xN, and given values yp1 and ypn for the first derivative of the interpolating
function at points 1 and n, respectively, this routine returns an array y2[1..n] that
contains the second derivatives of the interpolating function at the tabulated points x_i.
If yp1 and/or ypn are equal to 1 x 10^30 or larger, the routine is signaled to set the
corresponding boundary condition for a natural spline, with zero second derivative on
that boundary. JF*/
void spline(float x[], float y[], int n, float yp1, float ypn, float y2[]);

  /*! \brief  Given the arrays xa[1..n] and ya[1..n], which tabulate a function (with the xa_i's
     in order), and given the array y2a[1..n], which is the output from "spline()" above,
     and given a value of x, this routine returns a cubic-spline interpolated value y. JF */
void splint(float xa[], float ya[], float y2a[], int n, float x, float *y);

  /*! \brief  Given an m by n tabulated function ya[1..m][1..n], and tabulated independent
variables x2a[1..n], this routine constructs one-dimensional natural cubic splines
of the rows of ya and returns the second-derivatives in the array y2a[1..m][1..n].
(The array x1a[1..m] is included in the argument list merely for consistency with
routine "splin2()"). JF */
void splie2(float x1a[], float x2a[], float **ya, int m, int n, float y2[]);

  /*! \brief  Given x1a, x2a,ya, m,n as described in "splie2()" and y2a as produced by that routine;
and given a desired interpolating point x1,x2; this routine returns an interpolated function
value y by cubic spline interpolation. JF*/
void splin2(float x1a[], float x2a[], float **ya, float **y2a, int m, int n, float x1, float x2, float *y);

    //################## from mRoller ###########################
    int IsMin(float** I, int nr, int nc, int x, int y, int size);
    float min(float *A, int l, int r);
    float max(float *A, int l, int r);

    float min(float a, float b);
    float max(float a, float b);

    float maxInArray(float **A, int l1, int r1, int l2, int r2, int &i1, int &i2);
    float maxInRow(float **A, int row, int l, int r, int &index);
    float maxInCol(float **A, int col, int l, int r, int &index);


/**Given a matrix a[1..n][1..n], this routine replaces it by a balanced matrix with identical
eigenvalues. A symmetric matrix is already balanced and is unaffected by this procedure. The
parameter RADIX should be the machine’s floating-point radix.*/
template <class TREAL>   // float | double
    void balanc(TREAL **a, int n);

/**Reduction to Hessenberg form by the elimination method. The real, nonsymmetric matrix
a[1..n][1..n] is replaced by an upper Hessenberg matrix with identical eigenvalues. Recommended,
but not required, is that this routine be preceded by balanc. On output, the
Hessenberg matrix is in elements a[i][j] with i ? j+1. Elements with i > j+1 are to be
thought of as zero, but are returned with random values. */
template <class TREAL>   // float | double
    void elmhes(TREAL **a, int n);

/**Finds all eigenvalues of an upper Hessenberg matrix a[1..n][1..n]. On input a can be
exactly as output from elmhes §11.5; on output it is destroyed. The real and imaginary parts
of the eigenvalues are returned in wr[1..n] and wi[1..n], respectively. */
template <class TREAL>   // float | double
    void hqr(TREAL **a, int n, TREAL wr[], TREAL wi[]);

/** Sort singular values in W from the biggest to the lowest and swap related vectors in U and V */
template <class TREAL>   // float | double
    void svdsrt(  TREAL ** u, TREAL *w, TREAL ** v, int nr, int nc );

/**Finds all eigenvalues of square real, even nonsymetrix matrix a[1..n][1..n]. On input is real matrix; on output it is destroyed.
 * The real and imaginary parts of the eigenvalues are returned in wr[1..n] and wi[1..n], respectively.
 * The method constists of balancing matrix, reduction upper Hessenberg matrix with identical eigenvalues by elimination method, followed by iterating QR decompositions for finding eigen values
 * The sort parameter controls sorting of eginvalues by it absolute values.
 */
template <class TREAL>   // float | double
    void eigs_qr(TREAL **a, int n, TREAL wr[], TREAL wi[], bool sort);

/** Sorts eigenvalues according to its absolute values (desceding), both [1..n]
 *
 * @param wr real parts
 * @param wi imaginary parts
 */
template <class TREAL>   // float | double
    void eigsrt(TREAL *wr, TREAL *wi, int n);

//    bool gauss_nopivot(float **a, int n);

/** \brief Two vectors of length 3 difference. */
    void vec_diff(float *x, float *y, float *res);
/** \brief 3D vector length. */
    float vec_len(float *x);
/** \brief Two vectors of length 3 cross product. */
    void vec_crossprod(float *x, float *y, float *res);
/** \brief 3D vector normalization. */
    int vec_norm(float *x);
/** \brief Trims the numrer into specified interval */
    void range(float *data, float low, float high);

// \brief  template functions for linear (lsfit) and non-linear (mrqcof,mrqmin) optimization
//==========================================================================================
template <class TREAL>   // float | double
void fpoly(float x, TREAL p[], int np);
//void fpolyd(float x, double p[], int np);

/*! Lafortune BRDF approximation:
   f_r(\hat u,\hat v) = \\sum_i \\rho_{s,i} (C_{x,i} u_x + C_{y,i} u_y + C_{z,i} u_z)^{n_i}

   x[j][3] -> u_x    a[i]   -> rho_{s,i}
   x[j][4] -> u_y    a[i+1] -> C_{x_i}
   x[j][5] -> u_z    a[i+2] -> C_{y_i}
                     a[i+3] -> C_{z_i}
                     a[i+4] -> n_i                                   */
void fbrdf(float **x, int j, float a[], float *y, float dyda[], int na);

/*! Lafortune BRDF approximation:
   f_r(\hat u,\hat v) = \\sum_i \\rho_{s,i} (C_{x,i} u_x v_x + C_{y,i} u_y v_x + C_{z,i} u_z v_x)^{n_i}

   x[j][3] -> v_x    a[i]   -> rho_{s,i}
   x[j][4] -> v_y    a[i+1] -> C_{x_i}
   x[j][5] -> v_z    a[i+2] -> C_{y_i}
   x[j][0] -> u_x    a[i+3] -> C_{z_i}
   x[j][1] -> u_y    a[i+4] -> n_i
   x[j][2] -> u_z                                         */
void fbrdf_iv(float **x, int j, float a[], float *y, float dyda[], int na);


/*! RGB to YCbCr space */
void RGBtoYCbCr(const float RGB[], float YCbCr[]);
/*! YCbCr to RGB space */
void YCbCrtoRGB(const float YCbCr[], float RGB[]);


inline float signum( const float x ) { if (x>0) return 1; else if (x==0) return 0; else return -1; };

#endif
