
//
// Python frontent for SVMHeavy
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//


#ifndef _pyheavy_h
#define _pyheavy_h

#include "ml_mutable.hpp"


class SVMHeavy
{
public:

    // ================================================================
    //     Mutation functions
    // ================================================================
    //
    // setMLTypeClean: set svm type without data retention, so new type
    //                 is a clean start.

    void setMLTypeClean(int newmlType);






    // ================================================================
    //     Common functions for all ML
    // ================================================================

    // Stuff

    int preallocsize(void) const;

    int  prealloc(int expectedN);
    void setmemsize(int memsize);

    // Information functions (training data):

    int N      (void)  const;
    int NNC    (int d) const;
    int type   (void)  const;
    int subtype(void)  const;

    int tspaceDim   (void) const;
    int xspaceDim   (void) const;
    int fspaceDim   (void) const;
    int tspaceSparse(void) const;
    int xspaceSparse(void) const;
    int numClasses  (void) const;
    int order       (void) const;

    int isTrained(void) const;
    int isMutable(void) const;
    int isPool   (void) const;

    char gOutType(void) const;
    char hOutType(void) const;
    char targType(void) const;

    double calcDistNul(                      int ia = -1, int db = 2);
    double calcDistInt(int    ha, int    hb, int ia = -1, int db = 2);
    double calcDistDbl(double ha, double hb, int ia = -1, int db = 2);

    int isUnderlyingScalar(void) const;
    int isUnderlyingVector(void) const;
    int isUnderlyingAnions(void) const;

    void ClassLabels       (int *res) const;
    int  getInternalClass  (int y)    const;
    int  numInternalClasses(void)     const;
    int  isenabled         (int i)    const;

    double C       (void)  const;
    double sigma   (void)  const;
    double eps     (void)  const;
    double Cclass  (int d) const;
    double epsclass(int d) const;

    int    memsize     (void) const;
    double zerotol     (void) const;
    double Opttol      (void) const;
    int    maxitcnt    (void) const;
    double maxtraintime(void) const;

    int    maxitermvrank(void) const;
    double lrmvrank     (void) const;
    double ztmvrank     (void) const;

    double betarank(void) const;

    double sparlvl(void) const;

    int isClassifier(void) const;
    int isRegression(void) const;

    // Kernel Modification

    int kernel_isFullNorm       (void) const;
    int kernel_isProd           (void) const;
    int kernel_isIndex          (void) const;
    int kernel_isShifted        (void) const;
    int kernel_isScaled         (void) const;
    int kernel_isShiftedScaled  (void) const;
    int kernel_isLeftPlain      (void) const;
    int kernel_isRightPlain     (void) const;
    int kernel_isLeftRightPlain (void) const;
    int kernel_isLeftNormal     (void) const;
    int kernel_isRightNormal    (void) const;
    int kernel_isLeftRightNormal(void) const;
    int kernel_isPartNormal     (void) const;
    int kernel_isAltDiff        (void) const;
    int kernel_needsmProd       (void) const;
    int kernel_wantsXYprod      (void) const;
    int kernel_suggestXYcache   (void) const;
    int kernel_isIPdiffered     (void) const;

    int kernel_size       (void) const;
    int kernel_getSymmetry(void) const;

    double kernel_cWeight(int q = 0) const;
    int    kernel_cType  (int q = 0) const;

    int kernel_isNormalised(int q = 0) const;
    int kernel_isChained   (int q = 0) const;
    int kernel_isSplit     (int q = 0) const;
    int kernel_isMulSplit  (int q = 0) const;
    int kernel_isMagTerm   (int q = 0) const;

    int kernel_numSplits   (void) const;
    int kernel_numMulSplits(void) const;

    double kernel_cRealConstants(int i, int q = 0) const;
    int    kernel_cIntConstants (int i, int q = 0) const;
    int    kernel_cRealOverwrite(int i, int q = 0) const;
    int    kernel_cIntOverwrite (int i, int q = 0) const;

    double kernel_getRealConstZero(int q = 0);
    int    kernel_getIntConstZero (int q = 0);

    int kernel_isKVarianceNZ(void) const

    void kernel_add   (int q);
    void kernel_remove(int q);
    void kernel_resize(int nsize);

    void kernel_setFullNorm       (void);
    void kernel_setNoFullNorm     (void);
    void kernel_setProd           (void);
    void kernel_setnonProd        (void);
    void kernel_setLeftPlain      (void);
    void kernel_setRightPlain     (void);
    void kernel_setLeftRightPlain (void);
    void kernel_setLeftNormal     (void);
    void kernel_setRightNormal    (void);
    void kernel_setLeftRightNormal(void);

    void kernel_setAltDiff       (int nv);
    void kernel_setsuggestXYcache(int nv);
    void kernel_setIPdiffered    (int nv);

    void kernel_setChained   (int q = 0);
    void kernel_setNormalised(int q = 0);
    void kernel_setSplit     (int q = 0);
    void kernel_setMulSplit  (int q = 0);
    void kernel_setMagTerm   (int q = 0);

    void kernel_setUnChained   (int q = 0);
    void kernel_setUnNormalised(int q = 0);
    void kernel_setUnSplit     (int q = 0);
    void kernel_setUnMulSplit  (int q = 0);
    void kernel_setUnMagTerm   (int q = 0);

    void kernel_setWeight (double nw,   int q = 0);
    void kernel_setType   (int ndtype,  int q = 0);
    void kernel_setAltCall(int newMLid, int q = 0);

    void kernel_setRealConstants(double ndRealConstants, int i, int q = 0);
    void kernel_setIntConstants (int    ndIntConstants,  int i, int q = 0);
    void kernel_setRealOverwrite(int    ndRealOverwrite, int i, int q = 0);
    void kernel_setIntOverwrite (int    ndIntOverwrite,  int i, int q = 0);

    void kernel_setRealConstZero(double nv, int q = 0);
    void kernel_setIntConstZero (int nv,    int q = 0);

    void fillCache(void);

    // Training set modification:

    int addTrainingVectorNul(int i,           const double *x, int dim, double Cweigh = 1, double epsweigh = 1);
    int addTrainingVectorInt(int i, int    z, const double *x, int dim, double Cweigh = 1, double epsweigh = 1);
    int addTrainingVectorDbl(int i, double z, const double *x, int dim, double Cweigh = 1, double epsweigh = 1);

    int removeTrainingVector (int i);
    int removeTrainingVectors(int i, int num);

    int setx(int i, const double *x, int dim);

    int setyInt(int i, double  y);
    int setyDbl(int i, double  y);

    int setCweight    (int i, double nv);
    int setCweightfuzz(int i, double nv);
    int setsigmaweight(int i, double nv);
    int setepsweight  (int i, double nv);

    int scaleCweight    (double s);
    int scaleCweightfuzz(double s);
    int scalesigmaweight(double s);
    int scaleepsweight  (double s);

    // Basis stuff

    int NbasisUU   (void);
    int basisTypeUU(void);
    int defProjUU  (void);

    int setBasisYUU           (void);
    int setBasisUUU           (void);
    int addToBasisUU          (int i, const double *o, int dim);
    int removeFromBasisUU     (int i);
    int setBasisUU            (int i, const double *o, int dim);
    int setDefaultProjectionUU(int d);
    int setBasisUU            (int n, int d);

    int NbasisVV   (void) const;
    int basisTypeVV(void) const;
    int defProjVV  (void) const;

    int setBasisYVV           (void);
    int setBasisUVV           (void);
    int addToBasisVV          (int i, const double *o, int dim);
    int removeFromBasisVV     (int i);
    int setBasisVV            (int i, const double *o, int dim);
    int setDefaultProjectionVV(int d);
    int setBasisVV            (int n, int d);

    int kernelUU_isFullNorm       (void) const;
    int kernelUU_isProd           (void) const;
    int kernelUU_isIndex          (void) const;
    int kernelUU_isShifted        (void) const;
    int kernelUU_isScaled         (void) const;
    int kernelUU_isShiftedScaled  (void) const;
    int kernelUU_isLeftPlain      (void) const;
    int kernelUU_isRightPlain     (void) const;
    int kernelUU_isLeftRightPlain (void) const;
    int kernelUU_isLeftNormal     (void) const;
    int kernelUU_isRightNormal    (void) const;
    int kernelUU_isLeftRightNormal(void) const;
    int kernelUU_isPartNormal     (void) const;
    int kernelUU_isAltDiff        (void) const;
    int kernelUU_needsmProd       (void) const;
    int kernelUU_wantsXYprod      (void) const;
    int kernelUU_suggestXYcache   (void) const;
    int kernelUU_isIPdiffered     (void) const;

    int kernelUU_size       (void) const;
    int kernelUU_getSymmetry(void) const;

    double kernelUU_cWeight(int q = 0) const;
    int    kernelUU_cType  (int q = 0) const;

    int kernelUU_isNormalised(int q = 0) const;
    int kernelUU_isChained   (int q = 0) const;
    int kernelUU_isSplit     (int q = 0) const;
    int kernelUU_isMulSplit  (int q = 0) const;
    int kernelUU_isMagTerm   (int q = 0) const;

    int kernelUU_numSplits   (void) const;
    int kernelUU_numMulSplits(void) const;

    double kernelUU_cRealConstants(int i, int q = 0) const;
    int    kernelUU_cIntConstants (int i, int q = 0) const;
    int    kernelUU_cRealOverwrite(int i, int q = 0) const;
    int    kernelUU_cIntOverwrite (int i, int q = 0) const;

    double kernelUU_getRealConstZero(int q = 0);
    int    kernelUU_getIntConstZero (int q = 0);

    int kernelUU_isKVarianceNZ(void) const

    void kernelUU_add   (int q);
    void kernelUU_remove(int q);
    void kernelUU_resize(int nsize);

    void kernelUU_setFullNorm       (void);
    void kernelUU_setNoFullNorm     (void);
    void kernelUU_setProd           (void);
    void kernelUU_setnonProd        (void);
    void kernelUU_setLeftPlain      (void);
    void kernelUU_setRightPlain     (void);
    void kernelUU_setLeftRightPlain (void);
    void kernelUU_setLeftNormal     (void);
    void kernelUU_setRightNormal    (void);
    void kernelUU_setLeftRightNormal(void);

    void kernelUU_setAltDiff       (int nv);
    void kernelUU_setsuggestXYcache(int nv);
    void kernelUU_setIPdiffered    (int nv);

    void kernelUU_setChained   (int q = 0);
    void kernelUU_setNormalised(int q = 0);
    void kernelUU_setSplit     (int q = 0);
    void kernelUU_setMulSplit  (int q = 0);
    void kernelUU_setMagTerm   (int q = 0);

    void kernelUU_setUnChained   (int q = 0);
    void kernelUU_setUnNormalised(int q = 0);
    void kernelUU_setUnSplit     (int q = 0);
    void kernelUU_setUnMulSplit  (int q = 0);
    void kernelUU_setUnMagTerm   (int q = 0);

    void kernelUU_setWeight (double nw,   int q = 0);
    void kernelUU_setType   (int ndtype,  int q = 0);
    void kernelUU_setAltCall(int newMLid, int q = 0);

    void kernelUU_setRealConstants(double ndRealConstants, int i, int q = 0);
    void kernelUU_setIntConstants (int    ndIntConstants,  int i, int q = 0);
    void kernelUU_setRealOverwrite(int    ndRealOverwrite, int i, int q = 0);
    void kernelUU_setIntOverwrite (int    ndIntOverwrite,  int i, int q = 0);

    void kernelUU_setRealConstZero(double nv, int q = 0);
    void kernelUU_setIntConstZero (int nv,    int q = 0);

    // General modification and autoset functions

    int randomise  (double sparsity);
    int autoen     (void);
    int renormalise(void);
    int realign    (void);

    int setzerotol     (double zt);
    int setOpttol      (double xopttol);
    int setmaxitcnt    (int xmaxitcnt);
    int setmaxtraintime(double xmaxtraintime);

    int setmaxitermvrank(int nv);
    int setlrmvrank     (double nv);
    int setztmvrank     (double nv);

    int setbetarank(double nv);

    int setC       (double xC);
    int setsigma   (double xsigma);
    int seteps     (double xeps);
    int setCclass  (int d, double xC);
    int setepsclass(int d, double xeps);

    int scale  (double a);
    int reset  (void);
    int restart(void);
    int home   (void);

    int settspaceDim    (int newdim);
    int addtspaceFeat   (int i);
    int removetspaceFeat(int i);
    int addxspaceFeat   (int i);
    int removexspaceFeat(int i);

    int setsubtype(int i);

    int setorder(int neword);
    int addclass(int label, int epszero = 0);

    // Sampling mode

    int isSampleMode (void) const;
    int setSampleMode(int nv, const double *xmin, const double *xmax, int dim, int Nsamp = DEFAULT_SAMPLES_SAMPLE, int sampSplit = 1, int sampType = 0, double sampScale = 1.0);

    // Training functions:

    void fudgeOn (void);
    void fudgeOff(void);

    int train(int &res);

    // Evaluation Functions:

    double ggTrainingVector(int i) const;
    double hhTrainingVector(int i) const;

    double varTrainingVector     (int i)                                                   const;
    double covTrainingVector     (int i, int j)                                            const;
    double stabProbTrainingVector(int i, int p, double pnrm, int rot, double mu, double B) const;

    double gg(const double *x, int dim) const;
    double hh(const double *x, int dim) const;

    double var     (const double *xa, int dim)                                         const;
    double cov     (const double *xa, const double *xb, int dim)                       const;
    double stabProb(const double *x, int p, double pnrm, int rot, double mu, double B) const;

    // Other functions

    int disable(int i);

    // Kernel normalisation function

    int normKernelZeroMeanUnitVariance  (int flatnorm = 0, int noshift = 0);
    int normKernelZeroMedianUnitVariance(int flatnorm = 0, int noshift = 0);
    int normKernelUnitRange             (int flatnorm = 0, int noshift = 0);









    // ================================================================
    //     Common functions for all SVMs
    // ================================================================

    // Constructors, destructors, assignment etc..

    int setAlpha(const double *newAlpha);
    int setBias (double newBias);

    // Information functions (training data):

    int NZ (void)  const;
    int NF (void)  const;
    int NS (void)  const;
    int NC (void)  const;
    int NLB(void)  const;
    int NLF(void)  const;
    int NUF(void)  const;
    int NUB(void)  const;

    int isLinearCost   (void)  const;
    int isQuadraticCost(void)  const;
    int is1NormCost    (void)  const;
    int isVarBias      (void)  const;
    int isPosBias      (void)  const;
    int isNegBias      (void)  const;
    int isFixedBias    (void)  const;

    int isNoMonotonicConstraints   (void) const;
    int isForcedMonotonicIncreasing(void) const;
    int isForcedMonotonicDecreasing(void) const;

    int isOptActive(void) const;
    int isOptSMO   (void) const;
    int isOptD2C   (void) const;
    int isOptGrad  (void) const;

    int isFixedTube (void) const;
    int isShrinkTube(void) const;

    int isRestrictEpsPos(void) const;
    int isRestrictEpsNeg(void) const;

    int isClassifyViaSVR(void) const;
    int isClassifyViaSVM(void) const;

    int is1vsA   (void) const;
    int is1vs1   (void) const;
    int isDAGSVM (void) const;
    int isMOC    (void) const;
    int ismaxwins(void) const;
    int isrecdiv (void) const;

    int isatonce(void) const;
    int isredbin(void) const;

    int isKreal  (void) const;
    int isKunreal(void) const;

    int isanomalyOn (void) const;
    int isanomalyOff(void) const;

    int isautosetOff         (void) const;
    int isautosetCscaled     (void) const;
    int isautosetCKmean      (void) const;
    int isautosetCKmedian    (void) const;
    int isautosetCNKmean     (void) const;
    int isautosetCNKmedian   (void) const;
    int isautosetLinBiasForce(void) const;

    double outerlr      (void) const;
    double outermom     (void) const;
    int    outermethod  (void) const;
    double outertol     (void) const;
    double outerovsc    (void) const;
    int    outermaxitcnt(void) const;
    int    outermaxcache(void) const;

    int    maxiterfuzzt(void) const;
    int    usefuzzt    (void) const;
    double lrfuzzt     (void) const;
    double ztfuzzt     (void) const;

    int m(void) const;

    double LinBiasForce (void)  const;
    double QuadBiasForce(void)  const;

    double nu    (void) const;
    double nuQuad(void) const;

    double theta  (void) const;
    int    simnorm(void) const;

    double anomalyNu   (void) const;
    int    anomalyClass(void) const;

    double autosetCval (void) const;
    double autosetnuval(void) const;

    int    anomclass      (void) const;
    int    singmethod     (void) const;
    double rejectThreshold(void) const;

    // Training set modification:

    int removeNonSupports(void);
    int trimTrainingSet  (int maxsize);

    // General modification and autoset functions

    int setLinearCost   (void);
    int setQuadraticCost(void);
    int set1NormCost    (void);
    int setVarBias      (void);
    int setPosBias      (void);
    int setNegBias      (void);
    int setFixedBias    (double newbias);

    int setNoMonotonicConstraints   (void);
    int setForcedMonotonicIncreasing(void);
    int setForcedMonotonicDecreasing(void);

    int setOptActive(void);
    int setOptSMO   (void);
    int setOptD2C   (void);
    int setOptGrad  (void);

    int setFixedTube (void);
    int setShrinkTube(void);

    int setRestrictEpsPos(void);
    int setRestrictEpsNeg(void);

    int setClassifyViaSVR(void);
    int setClassifyViaSVM(void);

    int set1vsA   (void);
    int set1vs1   (void);
    int setDAGSVM (void);
    int setMOC    (void);
    int setmaxwins(void);
    int setrecdiv (void);

    int setatonce(void);
    int setredbin(void);

    int setKreal  (void);
    int setKunreal(void);

    int anomalyOn (int danomalyClass, double danomalyNu);
    int anomalyOff(void);

    int setouterlr      (double xouterlr);
    int setoutermom     (double xoutermom);
    int setoutermethod  (int xoutermethod);
    int setoutertol     (double xoutertol);
    int setouterovsc    (double xouterovsc);
    int setoutermaxitcnt(int xoutermaxits);
    int setoutermaxcache(int xoutermaxcacheN);

    int setmaxiterfuzzt(int xmaxiterfuzzt);
    int setusefuzzt    (int xusefuzzt);
    int setlrfuzzt     (double xlrfuzzt);
    int setztfuzzt     (double xztfuzzt);

    int setm(int xm);

    int setLinBiasForce (double newval);
    int setQuadBiasForce(double newval);

    int setnu    (double xnu);
    int setnuQuad(double xnuQuad);

    int settheta  (double nv);
    int setsimnorm(int nv);

    int autosetOff         (void);
    int autosetCscaled     (double Cval);
    int autosetCKmean      (void);
    int autosetCKmedian    (void);
    int autosetCNKmean     (void);
    int autosetCNKmedian   (void);
    int autosetLinBiasForce(double nuval, double Cval = 1.0);

    void setanomalyclass   (int n);
    void setsingmethod     (int nv);
    void setRejectThreshold(double nv);

    // Likelihood

    double quasiloglikelihood(void) const;







    // ================================================================
    //     Common functions for all GPs
    // ================================================================

    // General modification and autoset functions

    int setmuWeight(const double *nv, int dim);
    int setmuBias  (double nv);

    void   muWeight(double *res) const;
    double muBias  (void)        const;

    int isZeromuBias(void) const;
    int isVarmuBias (void) const;

    int setZeromuBias(void);
    int setVarmuBias (void);

    // Likelihood

    double loglikelihood(void) const;







    // ================================================================
    //     Common functions for all LS-SVMs
    // ================================================================

    // Constructors, destructors, assignment etc..

    int setgamma(const double *newW, int dim);
    int setdelta(double newB);

    // Additional information

    int isVardelta (void) const;
    int isZerodelta(void) const;

    void   gamma(double *res) const;
    double delta(void)        const;

    // General modification and autoset functions

    int setVardelta (void);
    int setZerodelta(void);

    // Likelihood

    double lsvloglikelihood(void) const;








    // ================================================================
    //     Common functions for all KNNs
    // ================================================================

    // Information functions (training data):

    int k  (void) const;
    int ktp(void) const;

    // General modification and autoset functions

    int setk  (int xk);
    int setktp(int xk);





    // ================================================================
    //     Common functions for all BLKs
    // ================================================================

    // Bernstein polynomials

    int bernDegree(void) const;
    int bernIndex (void) const;

    int setBernDegree(int nv);
    int setBernIndex (int nv);

    // Battery modelling parameters

    void   battparam     (double *res) const;
    double batttmax      (void)        const;
    double battImax      (void)        const;
    double batttdelta    (void)        const;
    double battVstart    (void)        const;
    double battthetaStart(void)        const;

    int setbattparam     (const double *nv, int dim);
    int setbatttmax      (double nv);
    int setbattImax      (double nv);
    int setbatttdelta    (double nv);
    int setbattVstart    (double nv);
    int setbattthetaStart(double nv);


private:

    ML_Mutable thewhatsit;
};

//
// Type ranges:   0- 99 support vector machine (SVM)
//              100-199 one-layer layer neural network (ONN)
//              200-299 blocks (BLK)
//              300-399 k-nearest-neighbour machines (KNN)
//              400-499 Gaussian processes (GP)
//              500-599 Least-squares support vector machine (LSV)
//              600-699 Improvement measures (IMP)
//              700-799 Super-sparse support vector machine (SSV)
//              800-899 Type-II multi-layer kernel-machine (MLM)
//
// Type list:
//
// Types:  0 = Scalar SVM
//         1 = Binary SVM
//         2 = 1-class SVM
//         3 = Multiclass SVM
//         4 = Vector SVM
//         5 = Anionic SVM (real, complex, quaternion, octonion)
//         6 = auto-encoding SVM
//         7 = Density estimation SVM
//         8 = Pareto frontier SVM
//        12 = Binary score SVM
//        13 = Scalar Regression Score SVM
//        14 = Vector Regression Score SVM
//        15 = Generic target SVM
//        16 = planar SVM
//        17 = multi-expert rank SVM
//        18 = multi-user binary SVM
//        19 = similarity-learning SVM
//       100 = Scalar ONN
//       101 = Vector ONN
//       102 = Anion ONN
//       103 = Binary ONN
//       104 = Auto-encoding ONN
//       105 = Generic target ONN
//       200 = NOP machine (do nothing)
//       201 = Consensus machine (voting)
//       202 = Scalar average machine
//       203 = User-defined function (elementwise, g_i(x_i))
//       204 = User-I/O function
//       205 = Vector average machine
//       206 = Anion average machine
//       207 = User-defined function (vectorwise, g(x_0,x_1,...))
//       208 = Funcion callback machine
//       209 = Mex-defined function (elementwise, g_i(x_i))
//       210 = Mex-defined function (vectorwise, g(x_0,x_1,...))
//       211 = Mercer kernel inheritance block
//       212 = Multi ML averaging block
//       213 = system call block
//       214 = kernel block
//       215 = Bernstein kernel block
//       300 = KNN density estimator
//       301 = KNN binary classifier
//       302 = KNN generic regression
//       303 = KNN scalar regression
//       304 = KNN vector regression
//       305 = KNN anionic regression
//       306 = KNN autoencoder
//       307 = KNN multiclass classifier
//       400 = Scalar GP
//       401 = Vector GP
//       402 = Anion GP
//       408 = Generic target GP
//       409 = Binary classification GP
//       500 = Scalar LSV (LS-SVM)
//       501 = Vector LSV (LS-SVM)
//       502 = Anion LSV (LS-SVM)
//       505 = Scalar Regression Scoring LSV (LS-SVM)
//       506 = Vector Regression Scoring LSV (LS-SVM)
//       507 = auto-encoding LSV (LS-SVM)
//       508 = Generic target LSV (LS-SVM)
//       509 = Planar LSV (LS-SVM)
//       510 = Multi-expert rank LSV (LS-SVM)
//       600 = expected improvement (EI) IMP
//       601 = Pareto SVM 1-norm 1-class mono-surrogate
//       700 = SSV scalar regression
//       701 = SSV binary
//       701 = SSV 1-class
//       800 = SSV scalar regression
//

#endif


