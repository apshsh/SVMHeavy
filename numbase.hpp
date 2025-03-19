
//
// Non-standard functions for reals
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//
// USE_PMMINTRIN:  use avx2 immintrin.h intrinsics to do faster dot products
//                 Note that this is processor dependent

#include <math.h>
#include <cmath>
#include <limits.h>
#include <limits>
#include <stdlib.h>
#include "memdebug.hpp"
#include "niceassert.hpp"
#include "gslrefs.hpp"

#ifndef _numbase_h
#define _numbase_h

// Assume at least cpp14
#ifdef IS_CPP11
#undef IS_CPP11
#endif
#define IS_CPP11

#ifdef IS_CPP14
#undef IS_CPP14
#endif
#define IS_CPP14

#ifdef IS_CPP23
#ifdef IS_CPP20
#undef IS_CPP20
#endif
#define IS_CPP20
#endif

#ifdef IS_CPP26
#ifdef IS_CPP23
#undef IS_CPP23
#endif
#define IS_CPP23
#ifdef IS_CPP20
#undef IS_CPP20
#endif
#define IS_CPP20
#endif

// __restrict__ is good because it tells the compiler that pointers
// don't overlap, so you can use quicker assembly.  However it isn't
// strictly part of the c++ standard (it belongs to modern C) and
// visual studio refuses to recognize it.

#ifndef VISUAL_STU
#define svm_restrict __restrict__
#endif
#ifdef VISUAL_STU
#define svm_restrict
#endif



// Constants:
//
// NUMBASE_EULER:          euler's constant
// NUMBASE_E:              e
// NUMBASE_PI:             pi
// NUMBASE_PION2:          pi/2
// NUMBASE_PION4:          pi/4
// NUMBASE_1ONPI:          1/pi
// NUMBASE_2ONPI:          2/pi
// NUMBASE_SQRT2ONPI:      sqrt(2/pi)
// NUMBASE_SQRTPION2:      sqrt(pi/2)
// NUMBASE_SQRTPI:         sqrt(pi)
// NUMBASE_SQRTSQRTPI:     sqrt(sqrt(pi))
// NUMBASE_2ONSQRTPI:      2/sqrt(pi)
// NUMBASE_1ONSQRT2PI:     1/sqrt(2pi)
// NUMBASE_1ONSQRTSQRT2PI: 1/sqrt(sqrt(2pi))
// NUMBASE_LN2:            ln(2)
// NUMBASE_LNPI:           ln(pi)
// NUMBASE_LN10:           ln(10)
// NUMBASE_LOG2E:          log2(e)
// NUMBASE_LOG10E:         log10(e)
// NUMBASE_SQRT2:          sqrt(2)
// NUMBASE_SQRT1ON2:       sqrt(1/2)
// NUMBASE_SQRT3:          sqrt(3)

#define NUMBASE_EULER           0.57721566490153286060651209008
#define NUMBASE_E               2.71828182845904523536028747135
#define NUMBASE_PI              3.14159265358979323846264338328
#define NUMBASE_PION2           1.57079632679489661923132169164
#define NUMBASE_PION4           0.78539816339744830966156608458
#define NUMBASE_1ONPI           0.31830988618379067153776752675
#define NUMBASE_2ONPI           0.63661977236758134307553505349
#define NUMBASE_SQRT2ONPI       0.79788456080286541
#define NUMBASE_SQRTPION2       1.2533141373155001
#define NUMBASE_SQRTPI          1.77245385090551602729816748334
#define NUMBASE_SQRTSQRTPI      1.3313353638003897
#define NUMBASE_2ONSQRTPI       1.12837916709551257389615890312
#define NUMBASE_1ONSQRT2PI      0.39894228040143267793994605993
#define NUMBASE_1ONSQRTSQRT2PI  0.63161877774606467
#define NUMBASE_LN2             0.69314718055994530941723212146
#define NUMBASE_LNPI            1.14472988584940017414342735135
#define NUMBASE_LN10            2.30258509299404568401799145468
#define NUMBASE_LN2PI           1.8378770664093453
#define NUMBASE_LOG2E           1.44269504088896340735992468100
#define NUMBASE_LOG10E          0.43429448190325182765112891892
#define NUMBASE_SQRT2           1.41421356237309504880168872421
#define NUMBASE_SQRT1ON2        0.70710678118654752440084436210
#define NUMBASE_SQRT3           1.73205080756887729352744634151
#define NUMBASE_HALFLOG2PI      0.91893853320467274178032973640562






// NaN and inf tests

inline int testisvnan(double x);
inline int testisinf (double x);
inline int testispinf(double x);
inline int testisninf(double x);

// Representations of nan, inf and -inf

inline double valvnan(void);
inline double valpinf(void);
inline double valninf(void);

inline double valvnan(const char *pack);

inline double valvnan(void) { return NAN;       }
inline double valpinf(void) { return INFINITY;  }
inline double valninf(void) { return -INFINITY; }

inline double valvnan(const char *pack) { return std::nan(pack); }

inline int testisvnan(double x) { return std::isnan(x);             }
inline int testisinf (double x) { return std::isinf(x);             }
inline int testispinf(double x) { return testisinf(x) && ( x > 0 ); }
inline int testisninf(double x) { return testisinf(x) && ( x < 0 ); }






// Non-standard functions for real/int

inline constexpr double arg   (double a); // apparently needs to be out for visual studio (with c++11?)
inline           double abs1  (double a);
inline           double abs2  (double a);
inline           double absp  (double a, double p);
inline           double absinf(double a);
inline           double abs0  (double a);
inline constexpr double angle (double a);
inline constexpr double vangle(double a, double res = 0.0);
inline constexpr double conj  (double a);
inline constexpr double real  (double a);
inline constexpr double imag  (double a);
inline constexpr double inv   (double a);
inline           double norm1 (double a);
inline constexpr double norm2 (double a);
inline           double normp (double a, double p);
inline constexpr double sgn   (double a);
inline           double tenup (double a);

//inline int hypot (int a, int b);
//inline int arg   (int a);
inline           int abs1  (int a);
inline           int abs2  (int a);
inline           int absp  (int a, double p);
inline constexpr int angle (int a);
inline           int vangle(int a, double res = 0);
inline           int absinf(int a);
inline           int abs0  (int a);
inline           int conj  (int a);
inline constexpr int real  (int a);
inline constexpr int imag  (int a);
inline constexpr double inv(int a);
inline           int norm1 (int a);
inline constexpr int norm2 (int a);
inline           int normp (int a, double p);
inline constexpr int sgn   (int a);
inline           int tenup (int a);






inline double &scaladd(double &a, double b);
inline double &scaladd(double &a, double b, double c);
inline double &scaladd(double &a, double b, double c, double d);
inline double &scaladd(double &a, double b, double c, double d, double e);
inline double &scalsub(double &a, double b);
inline double &scalmul(double &a, double b);
inline double &scaldiv(double &a, double b);

inline double &scaladd(double &a, double b) { return a += b; }
inline double &scaladd(double &a, double b, double c) { return ( a = std::fma(b,c,a) ); } // { return a += (b*c); }
inline double &scaladd(double &a, double b, double c, double d) { return a += (b*c*d); }
inline double &scaladd(double &a, double b, double c, double d, double e) { return a += (b*c*d*e); }
inline double &scalsub(double &a, double b) { return a -= b; }
inline double &scalmul(double &a, double b) { return a *= b; }
inline double &scaldiv(double &a, double b) { return a /= b; }








// Borrowed from Fortran
//
// dsign returns a with sign of b (if b == 0 returns a)
//
// sppythag finds sqrt(a**2+b**2) without overflow or destructive underflow
// (translated from eispack function pythag.f)

inline constexpr double dsign(double a, double b);
inline constexpr double sppythag(double a, double b);









// Manual operations for double/int

inline double &leftmult (double &a, double  b);
inline double &rightmult(double  a, double &b);

inline int &leftmult (int &a, int  b);
inline int &rightmult(int  a, int &b);

inline unsigned int &leftmult (unsigned int &a, unsigned int  b);
inline unsigned int &rightmult(unsigned int  a, unsigned int &b);

inline size_t &leftmult (size_t &a, size_t  b);
inline size_t &rightmult(size_t  a, size_t &b);

inline char &leftmult (char &a, char  b);
inline char &rightmult(char  a, char &b);

inline std::string &leftmult (std::string &a, std::string  b); // throw
inline std::string &rightmult(std::string  a, std::string &b); // throw














// Inner products etc (used by vectors in template form)

inline double &oneProduct  (double &res, double a);
inline double &twoProduct  (double &res, double a, double b);
inline double &threeProduct(double &res, double a, double b, double c);
inline double &fourProduct (double &res, double a, double b, double c, double d);
inline double &mProduct    (double &res, int m, const double *svm_restrict a);

inline double &innerProduct       (double &res, double a, double b);
inline double &innerProductRevConj(double &res, double a, double b);

inline int &oneProduct  (int &res, int a);
inline int &twoProduct  (int &res, int a, int b);
inline int &threeProduct(int &res, int a, int b, int c);
inline int &fourProduct (int &res, int a, int b, int c, int d);
inline int &mProduct    (int &res, int m, const int *svm_restrict a);

inline int &innerProduct       (int &res, int a, int b);
inline int &innerProductRevConj(int &res, int a, int b);











// Trigonometric and other special functions
//
// Cas is used in the Hartley transform, so I've implemented it
// for some reason
//
// castrig(x)   = cos(x)+sin(x)
// casctrig(x)  = cos(x)-sin(x)
// acastrig(x)  = acos(x/sqrt(2))+pi/4
// acasctrig(x) = acos(x/sqrt(2))-pi/4
// cashyp(x)    = cosh(x)+sinh(x) = exp(x)
// caschyp(x)   = cosh(x)-sinh(x) = exp(-x)
// acashyp(x)   = ln(x)
// acaschyp(x)  = -ln(x)

inline double cosec    (double a);
inline double sec      (double a);
inline double cot      (double a);
inline double acosec   (double a);
inline double asec     (double a);
inline double acot     (double a);
inline double sinc     (double a);
inline double cosc     (double a);
inline double tanc     (double a);
inline double vers     (double a);
inline double covers   (double a);
inline double hav      (double a);
inline double excosec  (double a);
inline double exsec    (double a);
inline double avers    (double a);
inline double acovers  (double a);
inline double ahav     (double a);
inline double aexcosec (double a);
inline double aexsec   (double a);
inline double cosech   (double a);
inline double sech     (double a);
inline double coth     (double a);
inline double acosech  (double a);
inline double asech    (double a);
inline double acoth    (double a);
inline double sinhc    (double a);
inline double coshc    (double a);
inline double tanhc    (double a);
inline double versh    (double a);
inline double coversh  (double a);
inline double havh     (double a);
inline double excosech (double a);
inline double exsech   (double a);
inline double aversh   (double a);
inline double acovrsh  (double a);
inline double ahavh    (double a);
inline double aexcosech(double a);
inline double aexsech  (double a);
inline double sigm     (double a);
inline double gd       (double a);
inline double asigm    (double a);
inline double agd      (double a);
inline double castrg   (double a);
inline double casctrg  (double a);
inline double acastrg  (double a);
inline double acasctrg (double a);
inline double cashyp   (double a);
inline double caschyp  (double a);
inline double acashyp  (double a);
inline double acaschyp (double a);












// Hypervolume Calculation Functions
//
// spherevol(rsq,n): calculate the volume of a hypersphere of radius sqrt(rsq)
//                   in n-dimensional space.  Note that the volume of a
//                   sphere in zero-dimensional space is defined as 1.

double spherevol(double rsq, size_t n);








// signed nth root
//
// nthrt(x,n): returns the signed x^1/n (nan if n even and x negative)

double nthrt(double x, int n);







// Other obscure functions
//
// return value: non-zero if error, zero if good
// result: placed in res
//
// dawson:      Dawson function
// gamma:       Gamma function
// lngamma:     Log gamma function
// gamma_inc:   Incomplete gamma function
// psi, psn_n:  Psi functions
// erf,erfc:    Error and complementary error functions
// erfinv:      Inverse of the erf function
// zeta:        Reimann zeta function
// lambertW:    standard W0 branch (W>-1) of the Lambert W function
// lambertWx:   W1 branch (W<-1) of the Lambert W function
// j0,j1,jn:    Bessel functions of the first kind
// k0,k1,kn:    Bessel functions of the second kind
// i0,i1,in:    Modified Bessel functions of the first kind
// k0,k1,kn:    Modified Bessel functions of the second kind
// probit:      Probit function
// erfcx(x) = exp(x^2) erfc(x)
// normphionPhi(x) = normphi(x)/normPhi(x)

inline int numbase_dawson   (double &res,           double x);
inline int numbase_gamma_inc(double &res, double a, double x);
inline int numbase_psi      (double &res,           double x);
inline int numbase_psi_n    (double &res, int n,    double x);
inline int numbase_lambertW (double &res,           double x);
inline int numbase_lambertWx(double &res,           double x);

double numbase_gamma    (double x);
double numbase_lngamma  (double x);
double numbase_erfinv   (double x); // may return packed NaN
double numbase_probit   (double x); // may return packed NaN
double numbase_zeta     (double x);
double numbase_i0       (double x);
double numbase_i1       (double x);
double numbase_in(int n, double x);
double numbase_j0       (double x);
double numbase_j1       (double x);
double numbase_jn(int n, double x);
double numbase_k0       (double x);
double numbase_k1       (double x);
double numbase_kn(int n, double x);
double numbase_y0       (double x);
double numbase_y1       (double x);
double numbase_yn(int n, double x);
double numbase_erfcx    (double x);

inline double normPhi     (double x);
inline double normphi     (double x);
inline double normphionPhi(double x);







// Optimised maths code (AVX2 optional).  Included here so weird assembly
// shennanigans are safely isolated from regular code.
//
// fastmProduct: compute m-product
//
// fastAddTo: a += b      (or a += s*b)
// fastSubTo: a -= b      (or a -= s*b)
// fastMulBy: a *= b
// fastDivBy: a /= b
//
// if deindexing used then:
//
// a[ai[ab+(i*as)]] ?= b[bi[bb+(i*bs)]]
//
// etc

#ifdef USE_PMMINTRIN
#ifndef __AVX2__
#undef USE_PMMINTRIN
#endif
#endif

#ifdef USE_PMMINTRIN
#include "immintrin.h"
#endif

inline double fastoneProduct  (const double *svm_restrict a, int n) noexcept;
inline double fasttwoProduct  (const double *svm_restrict a, const double *svm_restrict b, int n) noexcept;
inline double fastthreeProduct(const double *svm_restrict a, const double *svm_restrict b, const double *svm_restrict c, int n) noexcept;
inline double fastfourProduct (const double *svm_restrict a, const double *svm_restrict b, const double *svm_restrict c, const double *svm_restrict d, int n) noexcept;
inline double fastfiveProduct (const double *svm_restrict a, const double *svm_restrict b, const double *svm_restrict c, const double *svm_restrict d, const double *svm_restrict e, int n) noexcept;

inline double fastoneProduct  (const double *svm_restrict a, const int *svm_restrict ai, int ab, int as, int n) noexcept;
inline double fasttwoProduct  (const double *svm_restrict a, const int *svm_restrict ai, int ab, int as, const double *svm_restrict b, const int *svm_restrict bi, int bb, int bs, int n) noexcept;
inline double fastthreeProduct(const double *svm_restrict a, const int *svm_restrict ai, int ab, int as, const double *svm_restrict b, const int *svm_restrict bi, int bb, int bs, const double *svm_restrict c, const int *svm_restrict ci, int cb, int cs, int n) noexcept;
inline double fastfourProduct (const double *svm_restrict a, const int *svm_restrict ai, int ab, int as, const double *svm_restrict b, const int *svm_restrict bi, int bb, int bs, const double *svm_restrict c, const int *svm_restrict ci, int cb, int cs, const double *svm_restrict d, const int *svm_restrict di, int db, int ds, int n) noexcept;
inline double fastfiveProduct (const double *svm_restrict a, const int *svm_restrict ai, int ab, int as, const double *svm_restrict b, const int *svm_restrict bi, int bb, int bs, const double *svm_restrict c, const int *svm_restrict ci, int cb, int cs, const double *svm_restrict d, const int *svm_restrict di, int db, int ds, const double *svm_restrict e, const int *svm_restrict ei, int eb, int es, int n) noexcept;

inline double fastoneProduct  (const double *svm_restrict a, int n, const double *svm_restrict scale) noexcept;
inline double fasttwoProduct  (const double *svm_restrict a, const double *svm_restrict b, int n, const double *svm_restrict scale) noexcept;
inline double fastthreeProduct(const double *svm_restrict a, const double *svm_restrict b, const double *svm_restrict c, int n, const double *svm_restrict scale) noexcept;
inline double fastfourProduct (const double *svm_restrict a, const double *svm_restrict b, const double *svm_restrict c, const double *svm_restrict d, int n, const double *svm_restrict scale) noexcept;
inline double fastfiveProduct (const double *svm_restrict a, const double *svm_restrict b, const double *svm_restrict c, const double *svm_restrict d, const double *svm_restrict e, int n, const double *svm_restrict scale) noexcept;

inline double fastoneProductSparse  (const double *svm_restrict a, const int *svm_restrict ai, int an) noexcept;
inline double fasttwoProductSparse  (const double *svm_restrict a, const int *svm_restrict ai, int an, const double *svm_restrict b, const int *svm_restrict bi, int bn) noexcept;
inline double fastthreeProductSparse(const double *svm_restrict a, const int *svm_restrict ai, int an, const double *svm_restrict b, const int *svm_restrict bi, int bn, const double *svm_restrict c, const int *svm_restrict ci, int cn) noexcept;
inline double fastfourProductSparse (const double *svm_restrict a, const int *svm_restrict ai, int an, const double *svm_restrict b, const int *svm_restrict bi, int bn, const double *svm_restrict c, const int *svm_restrict ci, int cn, const double *svm_restrict d, const int *svm_restrict di, int dn) noexcept;
inline double fastfiveProductSparse (const double *svm_restrict a, const int *svm_restrict ai, int an, const double *svm_restrict b, const int *svm_restrict bi, int bn, const double *svm_restrict c, const int *svm_restrict ci, int cn, const double *svm_restrict d, const int *svm_restrict di, int dn, const double *svm_restrict e, const int *svm_restrict ei, int en) noexcept;

inline void fastAddTo(double *svm_restrict a, const double *svm_restrict b,           int dim) noexcept;
inline void fastAddTo(double *svm_restrict a, const double *svm_restrict b, double s, int dim) noexcept;
inline void fastSubTo(double *svm_restrict a, const double *svm_restrict b,           int dim) noexcept;
inline void fastSubTo(double *svm_restrict a, const double *svm_restrict b, double s, int dim) noexcept;
inline void fastMulBy(double *svm_restrict a, const double *svm_restrict b,           int dim) noexcept;
inline void fastDivBy(double *svm_restrict a, const double *svm_restrict b,           int dim) noexcept;

inline void fastAddTo(double *svm_restrict a, double b, int dim) noexcept;
inline void fastSubTo(double *svm_restrict a, double b, int dim) noexcept;
inline void fastMulBy(double *svm_restrict a, double b, int dim) noexcept;
inline void fastDivBy(double *svm_restrict a, double b, int dim) noexcept;

inline void fastAddTo(double *svm_restrict a, const int *svm_restrict ai, int ab, int as, const double *svm_restrict b, const int *svm_restrict bi, int bb, int bs,           int dim) noexcept;
inline void fastAddTo(double *svm_restrict a, const int *svm_restrict ai, int ab, int as, const double *svm_restrict b, const int *svm_restrict bi, int bb, int bs, double s, int dim) noexcept;
inline void fastSubTo(double *svm_restrict a, const int *svm_restrict ai, int ab, int as, const double *svm_restrict b, const int *svm_restrict bi, int bb, int bs,           int dim) noexcept;
inline void fastSubTo(double *svm_restrict a, const int *svm_restrict ai, int ab, int as, const double *svm_restrict b, const int *svm_restrict bi, int bb, int bs, double s, int dim) noexcept;
inline void fastMulBy(double *svm_restrict a, const int *svm_restrict ai, int ab, int as, const double *svm_restrict b, const int *svm_restrict bi, int bb, int bs,           int dim) noexcept;
inline void fastDivBy(double *svm_restrict a, const int *svm_restrict ai, int ab, int as, const double *svm_restrict b, const int *svm_restrict bi, int bb, int bs,           int dim) noexcept;

inline void fastAddTo(double *svm_restrict a, const int *svm_restrict ai, int ab, int as, double b, int dim) noexcept;
inline void fastSubTo(double *svm_restrict a, const int *svm_restrict ai, int ab, int as, double b, int dim) noexcept;
inline void fastMulBy(double *svm_restrict a, const int *svm_restrict ai, int ab, int as, double b, int dim) noexcept;
inline void fastDivBy(double *svm_restrict a, const int *svm_restrict ai, int ab, int as, double b, int dim) noexcept;

















//inline int numbase_erf      (double &res,           double x);
//inline int numbase_erfc     (double &res,           double x);
//inline int numbase_Phi      (double &res,           double x);
//inline int numbase_phi      (double &res,           double x);


// Some maths

// These are used by numbase to define missing functions when required

#ifndef VISUAL_STU
#ifndef CYGWIN10
#define NO_ABS
#endif
#endif

#ifdef VISUAL_STU_NOERF
#define NO_ERF
#endif

#ifdef NO_HYPOT
inline double hypot(double a, double b);
inline double hypot(double a, double b)
{
    double fabsa = ( a < 0 ) ? -a : a;
    double fabsb = ( b < 0 ) ? -b : b;
    double maxab = ( fabsa > fabsb ) ? fabsa : fabsb;
    double minab = ( fabsa > fabsb ) ? fabsb : fabsa;

    double res = 0.0;

    if ( maxab > 0.0 )
    {
        res = maxab*sqrt(1.0+((minab/maxab)*(minab*maxab)));
    }

    return res;
}
#endif

//#ifdef NO_ABS
//inline double abs(double a);
//inline double abs(double a)
//{
//    return fabs(a);
//}
//#endif

#ifdef NO_ACOSH_ASINH_ATANH
inline double asinh(double a);
inline double acosh(double a);
inline double atanh(double a);
inline double asinh(double a) { return log(a+sqrt((a*a)+1)); }
inline double acosh(double a) { return log(a+sqrt((a*a)-1)); }
inline double atanh(double a) { return log((1+a)/(1-a))/2;   }
#endif

#ifdef NO_ERF
inline double erf(double x);
inline double erf(double x)
{
    // constants
    double a1 =  0.254829592;
    double a2 = -0.284496736;
    double a3 =  1.421413741;
    double a4 = -1.453152027;
    double a5 =  1.061405429;
    double p  =  0.3275911;

    // Save the sign of x
    int sign = ( x >= 0 ) ? 1 : -1;
    x = fabs(x);

    // A&S formula 7.1.26
    double t = 1.0/(1.0 + p*x);
    double y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x);

    return sign*y;
}
#endif

//inline double erfc(double x);
inline double erfc(double x)
{
    return 1-erf(x);
}

// Various maths functions
//
// roundnearest: round to nearest integer,
// xnfact: factorial function
// xnCr: n choose r using algorithm designed to minimise chances of overflow
// ceilintlog2: ceil(log2(x)) for integers, return 0 if x = 0
// upidiv: integer division that rounds upwards, not downwards

//inline                    int roundnearest(double x);
inline constexpr          int xnfact(int i);
inline constexpr          int xnmultifact(int i, int m);
inline constexpr          int xnCr(int n, int r);
inline constexpr unsigned int ceilintlog2(unsigned int x);
inline constexpr          int upidiv(int i, int j);

// "Make nice" function for maths equations.  Converts E to e and
// fixes + and - to ensure that + only appears as addition, - only
// as unary negation, and no long strings of ++---.  Returns 0 on
// success, nonzero of failure.

int makeMathsStringNice(std::string &dest, const std::string &src);

// Kronecker-Delta function

inline constexpr int krondel(int i, int j);

//inline int roundnearest(double x)
//{
//    return (int) std::round(x);
//}

/*
inline constexpr int roundnearest(double x)
{
    // round x to nearest integer

    int res = (int) x; // floor
    double rx = x-res; // remainder

    // Initial      Round required      Corrected result
    //
    // x = 1.1
    // res = 1           0                    1
    // rx = 0.1
    //
    // x = 1.1
    // res = 2           -1                   1
    // rx = -0.9
    //
    // x = 1.9
    // res = 1           +1                   2
    // rx = 0.9
    //
    // x = 1.9
    // res = 2           0                    2
    // rx = -0.1
    //
    // x = -1.1
    // res = -2          +1                   -1
    // rx = 0.9
    //
    // x = -1.1
    // res = -1          0                    -1
    // rx = -0.1
    //
    // x = -1.9
    // res = -1          -1                   -2
    // rx = 0.9
    //
    // x = -1.9
    // res = -2          0                    -2
    // rx = 0.1

//    if ( rx < -0.5 )
//    {
//        res -= 1;
//    }
//
//    else if ( rx > 0.5 )
//    {
//        res += 1;
//    }

    res = ( rx < -0.5 ) ? res-1 : res+1;

    return res;
}
*/

inline constexpr int xnfact(int i)
{
    int res = 1;

    while ( i > 0 )
    {
        res *= i--; // note use of post-decrement here
    }

    return res;
}
/*
    if ( i <= 0 )
    {
        return 1;
    }

    int j;
    int res = 1;

    for ( j = 1 ; j <= i ; ++j )
    {
        res *= j;
    }

    return res;
*/

inline constexpr int xnmultifact(int i, int m)
{
    int res = 1;

    while ( i > 0 )
    {
        res *= i;
        i -= m;
    }

    return res;
}

inline constexpr int xnCr(int n, int r)
{
    int result = 1;

    if ( ( n >= 0 ) && ( r >= 0 ) && ( n >= r ) )
    {
        // Recall: comb(0,0) = 1
        //         comb(n,0) = 1
        //         comb(n,n) = 1
        //         comb(n,r) = n!/(r!(n-r)!) (assuming r > n)

        result = 1;

        if ( ( r > 0 ) && ( n != r ) )
        {
            // Want r as large as possible

            if ( r > n/2 )
            {
                r = n-r;
            }

            for ( int k = 1 ; k <= r ; ++k )
            {
                result *= (n-r+k);
                result /= k;
            }

//          if ( (n-r) > r )
//          {
//              r = n-r;
//          }
//
//          for ( kk = r+1 ; kk <= n ; ++kk )
//          {
//              result *= kk;
//                result /= r;
//          }
        }
    }

    return result;
}

inline constexpr unsigned int ceilintlog2(unsigned int x)
{
    unsigned int zord = 0;
    unsigned int zsize = x;

    if ( x == 0 )
    {
        zord = 0; // special case (the ceil part)
    }

    else
    {
	// Calculate log2(x)

	while ( ( zsize >>= 1 ) ) ++zord;
    }

    // ceilintlog2(0) = 0
    // ceilintlog2(1) = 0
    // ceilintlog2(2) = 1
    // ceilintlog2(4) = 2
    // ceilintlog2(8) = 3
    // ...

    return zord;
}

// Upwards rounding integer division
// (haven't checked what happens with negative arguments)
//
// eg:
//
//  i   | j   | i/j | updiv(i,j)
// -----+-----+-----+------------
//  0   | 2   | 0   | 0
//  1   | 2   | 0   | 1
//  2   | 2   | 1   | 1
//  3   | 2   | 1   | 2
//  4   | 2   | 2   | 2
//  5   | 2   | 2   | 3
//  6   | 2   | 3   | 3
//  7   | 2   | 3   | 4
//  ... | ... | ... | ...


inline constexpr int upidiv(int i, int j)
{
    return (i%j) ? ((i/j)+1) : (i/j);
}

#ifdef VISUAL_STU
#ifndef VISUAL_STU_OLD
#define ALT_INF_DEF
#endif
#endif

//inline           int    testispinf(double x) { return ( x == valpinf() );                                }
//inline           int    testisninf(double x) { return ( x == valninf() );                                }

//inline double valvnan(void)             { return NAN; }
//inline double valvnan(const char *pack) { return std::nan(pack); }
//inline double valpinf(void)             { return INFINITY;  }
//inline double valninf(void)             { return -INFINITY; }

//inline constexpr int    testisvnan(double x) { return ( !( x > 0.0 ) && !( x < 0.0 ) && !( x == 0.0 ) ); }
//inline int testisvnan(double x) { return std::isnan(x);             }
//inline int testisinf (double x) { return std::isinf(x);             }
//inline int testispinf(double x) { return testisinf(x) && ( x > 0 ); }
//inline int testisninf(double x) { return testisinf(x) && ( x < 0 ); }
//#ifdef VISUAL_STU
//#include <limits>
//inline constexpr double valvnan(void)             { return std::numeric_limits<double>::quiet_NaN(); }
//inline constexpr double valvnan(const char *pack) { return std::numeric_limits<double>::quiet_NaN(); }
//inline constexpr double valpinf(void)             { return std::numeric_limits<double>::infinity();  }
//inline constexpr double valninf(void)             { return -std::numeric_limits<double>::infinity(); }
//#endif
//#ifndef VISUAL_STU
//inline constexpr double valvnan(void)             { return 0.0/0.0;  }
//inline constexpr double valvnan(const char *pack) { return 0.0/0.0;  }
//inline constexpr double valpinf(void)             { return 1.0/0.0;  }
//inline constexpr double valninf(void)             { return -1.0/0.0; }
//#endif

inline constexpr int krondel(int i, int j)
{
    return ( i == j ) ? 1 : 0;
}






































// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================

inline constexpr double dsign(double a, double b)
{
    return ( b >= 0.0 ) ? a : -a;
}

inline constexpr double sppythag(double a, double b)
{
    a = ( a < 0 ) ? -a : a; //fabs(a);
    b = ( b < 0 ) ? -b : b; //fabs(b);

    double p = ( a > b ) ? a : b;

    if ( p > 0.0 )
    {
        double r  = ( ( a < b ) ? a : b );
        r /= p;
        r *= r;
        double t = 4.0 + r;

        while ( t > 4.0 )
        {
            double s = r/t;
            double u = 1.0 + 2.0*s;
            p = u*p;
            t = (s/u);
            r *= t*t;
            t = 4.0 + r;
        }
    }

    return p;
}


inline           double abs1  (double a          ) { return fabs(a); }
inline           double abs2  (double a          ) { return fabs(a); }
inline           double absp  (double a, double p) { (void) p; return fabs(a); }
inline           double absinf(double a          ) { return fabs(a); }
inline           double abs0  (double a          ) { return fabs(a); }
inline constexpr double angle (double a          ) { return ( a > 0 ) ? +1 : ( ( a < 0 ) ? -1 : 0 ); }
inline constexpr double vangle(double a, double p) { (void) a; return p; }
inline constexpr double arg   (double a          ) { return ( a >= 0 ) ? 0 : NUMBASE_PI; }
inline constexpr double conj  (double a          ) { return a; }
inline constexpr double real  (double a          ) { return a; }
inline constexpr double imag  (double a          ) { (void) a; return 0; }
inline constexpr double inv   (double a          ) { return 1/a; }
inline constexpr double inv   (int a             ) { return inv((double) a); }
inline           double norm1 (double a          ) { return fabs(a); }
inline constexpr double norm2 (double a          ) { return a*a; }
inline           double normp (double a, double p) { return pow(fabs(a),p); }
inline constexpr double sgn   (double a          ) { return ( a > 0 ) ? +1 : ( ( a < 0 ) ? -1 : 0 ); }
inline           double tenup (double a          ) { return pow(10,a); }

//inline double arg (int    a          ) { return ( a >= 0 ) ? 0 : NUMBASE_PI; }

inline           int    abs1  (int    a          ) { return abs(a); }
inline           int    abs2  (int    a          ) { return abs(a); }
inline           int    absp  (int    a, double p) { (void) p; return abs(a); }
inline           int    absinf(int    a          ) { return abs(a); }
inline           int    abs0  (int    a          ) { return abs(a); }
inline constexpr int    angle (int    a          ) { return ( a > 0 ) ? +1 : ( ( a < 0 ) ? -1 : 0 ); }
inline           int    vangle(int    a, double p) { (void) a; int pp = (int) p; NiceAssert( p == pp ); return pp; }
inline           int    conj  (int    a          ) { return a; }
inline constexpr int    imag  (int    a          ) { (void) a; return 0; }
inline           int    norm1 (int    a          ) { return ( a < 0 ) ? -a : a; }
inline constexpr int    norm2 (int    a          ) { return a*a; }
inline           int    normp (int    a, double p) { int pp = (int) p; NiceAssert( p == pp ); NiceAssert( pp >= 0 ); return ( a < 0 ) ? normp(-a,pp) : ( ( pp > 0 ) ? a*normp(a,pp-1) : 1 ); }
inline constexpr int    real  (int    a          ) { return a; }
inline constexpr int    sgn   (int    a          ) { return ( a > 0 ) ? +1 : ( ( a < 0 ) ? -1 : 0 ); }
inline           int    tenup (int    a          ) { return (int) pow(10.0,(double) a); }

inline double &oneProduct  (double &res, double a)                               { res = a;       return res; }
inline double &twoProduct  (double &res, double a, double b)                     { res = a*b;     return res; }
inline double &threeProduct(double &res, double a, double b, double c)           { res = a*b*c;   return res; }
inline double &fourProduct (double &res, double a, double b, double c, double d) { res = a*b*c*d; return res; }
inline double &mProduct    (double &res, int m, const double *svm_restrict a)    { res = 1.0; NiceAssert( m >= 0 ); for ( int i = 0 ; i < m ; ++i ) { res *= a[i]; } return res; }

inline double &innerProduct       (double &res, double a, double b) { res = a*b; return res; }
inline double &innerProductRevConj(double &res, double a, double b) { res = a*b; return res; }

inline int &oneProduct  (int &res, int a)                            { res = a;       return res; }
inline int &twoProduct  (int &res, int a, int b)                     { res = a*b;     return res; }
inline int &threeProduct(int &res, int a, int b, int c)              { res = a*b*c;   return res; }
inline int &fourProduct (int &res, int a, int b, int c, int d)       { res = a*b*c*d; return res; }
inline int &mProduct    (int &res, int m, const int *svm_restrict a) { res = 1; NiceAssert( m >= 0 ); for ( int i = 0 ; i < m ; ++i ) { res *= a[i]; } return res; }

inline int &innerProduct       (int &res, int a, int b) { res = a*b; return res; }
inline int &innerProductRevConj(int &res, int a, int b) { res = a*b; return res; }

inline double &leftmult (double &a, double  b) { return a *= b; }
inline double &rightmult(double  a, double &b) { return b *= a; }

inline int &leftmult (int &a, int  b) { return a *= b; }
inline int &rightmult(int  a, int &b) { return b *= a; }

inline unsigned int &leftmult (unsigned int &a, unsigned int  b) { return a *= b; }
inline unsigned int &rightmult(unsigned int  a, unsigned int &b) { return b *= a; }

inline size_t &leftmult (size_t &a, size_t  b) { return a *= b; }
inline size_t &rightmult(size_t  a, size_t &b) { return b *= a; }

inline char &leftmult (char &a, char  b) { return a = static_cast<char>(a*b); }
inline char &rightmult(char  a, char &b) { return b = static_cast<char>(a*b); }

inline std::string &leftmult (std::string &a, std::string  b) { (void) b; NiceThrow("leftmult string is meaningless");  return a; }
inline std::string &rightmult(std::string  a, std::string &b) { (void) a; NiceThrow("rightmult string is meaningless"); return b; }

// yes sin(a)*inv(a) = inv(a)*sin(a), even for anions

inline double cosec    (double a) { return inv(sin(a)); }
inline double sec      (double a) { return inv(cos(a)); }
inline double cot      (double a) { return cos(a)*inv(sin(a)); }
inline double acosec   (double a) { return asin(inv(a)); }
inline double asec     (double a) { return acos(inv(a)); }
inline double acot     (double a) { return atan(inv(a)); }
inline double sinc     (double a) { return ( fabs(a) <= 1e-7 ) ? 1 : sin(a)*inv(a); }
inline double cosc     (double a) { return cos(a)*inv(a); }
inline double tanc     (double a) { return ( fabs(a) <= 1e-7 ) ? 1 : tan(a)*inv(a); }
inline double vers     (double a) { return 1-cos(a); }
inline double covers   (double a) { return 1-sin(a); }
inline double hav      (double a) { return vers(a)/2.0; }
inline double excosec  (double a) { return cosec(a)-1; }
inline double exsec    (double a) { return sec(a)-1; }
inline double avers    (double a) { return acos(a+1); }
inline double acovers  (double a) { return asin(a+1); }
inline double ahav     (double a) { return avers(2*a); }
inline double aexcosec (double a) { return acosec(a+1); }
inline double aexsec   (double a) { return asec(a+1); }
inline double cosech   (double a) { return inv(sinh(a)); }
inline double sech     (double a) { return inv(cosh(a)); }
inline double coth     (double a) { return cosh(a)*inv(sinh(a)); }
inline double acosech  (double a) { return asinh(inv(a)); }
inline double asech    (double a) { return acosh(inv(a)); }
inline double acoth    (double a) { return atanh(inv(a)); }
inline double sinhc    (double a) { return ( fabs(a) <= 1e-7 ) ? 1 : sinh(a)*inv(a); }
inline double coshc    (double a) { return cosh(a)*inv(a); }
inline double tanhc    (double a) { return ( fabs(a) <= 1e-7 ) ? 1 : tanh(a)*inv(a); }
inline double versh    (double a) { return 1-cosh(a); }
inline double coversh  (double a) { return 1-sinh(a); }
inline double havh     (double a) { return versh(a)/2.0; }
inline double excosech (double a) { return cosech(a)-1; }
inline double exsech   (double a) { return sech(a)-1; }
inline double aversh   (double a) { return acosh(a+1); }
inline double acovrsh  (double a) { return asinh(a+1); }
inline double ahavh    (double a) { return aversh(2*a); }
inline double aexcosech(double a) { return acosech(a+1); }
inline double aexsech  (double a) { return asech(a+1); }
inline double sigm     (double a) { return inv(1+exp(a)); }
inline double gd       (double a) { return 2*atan(tanh(a/2.0)); }
inline double asigm    (double a) { return log(inv(a)-1.0); }
inline double agd      (double a) { return 2*atanh(tan(a/2.0)); }
inline double castrg   (double a) { return cos(a)+sin(a); }
inline double casctrg  (double a) { return cos(a)-sin(a); }
inline double acastrg  (double a) { return acos(a/NUMBASE_SQRT2)+NUMBASE_PION4; }
inline double acasctrg (double a) { return acos(a/NUMBASE_SQRT2)-NUMBASE_PION4; }
inline double cashyp   (double a) { return exp(a); }
inline double caschyp  (double a) { return exp(-a); }
inline double acashyp  (double a) { return log(a); }
inline double acaschyp (double a) { return -log(a); }


inline double numbase_probit(double x) { return NUMBASE_SQRT2*numbase_erfinv((2*x)-1); }


inline int numbase_dawson   (double &res,           double x) { return gsl_dawson(res,x);      }
inline int numbase_gamma_inc(double &res, double a, double x) { return gsl_gamma_inc(res,a,x); }
inline int numbase_psi      (double &res,           double x) { return gsl_psi(res,x);         }
inline int numbase_psi_n    (double &res, int n,    double x) { return gsl_psi_n(res,n,x);     }
inline int numbase_lambertW (double &res,           double x) { return gsl_lambertW0(res,x);   }
inline int numbase_lambertWx(double &res,           double x) { return gsl_lambertW1(res,x);   }
//inline int numbase_erf      (double &res,           double x) { res = erf(x);                                                 return 0;    }
//inline int numbase_erfc     (double &res,           double x) { res = 1-erf(x);                                               return 0;    }
//inline int numbase_Phi      (double &res,           double x) { res = 0.5 + (0.5*erf(x*NUMBASE_SQRT1ON2));                    return 0;    }
//inline int numbase_phi      (double &res,           double x) { res = NUMBASE_1ONSQRT2PI*exp(-x*x/2);                         return 0;    }

#ifdef IS_CPP17
inline double numbase_zeta(double x) { return std::riemann_zeta(x); }
#endif
#ifndef IS_CPP17
inline double numbase_zeta(double x) { return gsl_zeta(x); }
#endif

inline double normPhi(double x)
{
    return 0.5 + (0.5*erf(x*NUMBASE_SQRT1ON2));
}

inline double normphi(double x)
{
    return NUMBASE_1ONSQRT2PI*exp(-x*x/2);
}

inline double normphionPhi(double x)
{
    // sqrt(2/pi) exp(-x^2/2) / ( 1 + erf(x/sqrt(2)) )
    // = sqrt(2/pi) exp(-x^2/2) / ( 1 - erf(-x/sqrt(2)) )
    // = sqrt(2/pi) exp(-x^2/2) / erfc(-x/sqrt(2))
    // = sqrt(2/pi) exp(-x^2/2) exp( -ln(erfc(-x/sqrt(2))) )
    // = sqrt(2/pi) exp(-x^2/2 - ln(erfc(-x/sqrt(2))) )
    //
    // ln(erfcx(x)) = ln(exp(x^2) erfc(x))
    //              = ln(erfc(x)) + x^2
    // => ln(erfc(x)) = ln(erfcx(x)) - x^2
    // => ln(erfc(-x)) = ln(erfcx(-x)) - x^2
    //
    // => sqrt(2/pi) exp(-x^2/2) / ( 1 + erf(x/sqrt(2)) )
    // = sqrt(2/pi) exp(-x^2/2 - ln(erfcx(-x/sqrt(2))) + x^2/2 )
    // = sqrt(2/pi) exp(-ln(erfcx(-x/sqrt(2))))
    // = sqrt(2/pi) 1/erfcx(-x/sqrt(2))

    return NUMBASE_SQRT2ONPI/numbase_erfcx(-x*NUMBASE_SQRT1ON2);
}











// Optimised m-product code (AVX2 optional)

inline void fastAddTo(double *svm_restrict a, const double *svm_restrict b, int dim) noexcept
{
    for ( int i = 0 ; i < dim ; ++i )
    {
        a[i] += b[i];
    }

    return;
}

inline void fastAddTo(double *svm_restrict a, const double *svm_restrict b, double s, int dim) noexcept
{
    for ( int i = 0 ; i < dim ; ++i )
    {
        //a[i] += s*b[i];
        a[i] =  std::fma(s,b[i],a[i]);
    }

    return;
}

inline void fastSubTo(double *svm_restrict a, const double *svm_restrict b, int dim) noexcept
{
    for ( int i = 0 ; i < dim ; ++i )
    {
        a[i] -= b[i];
    }

    return;
}

inline void fastSubTo(double *svm_restrict a, const double *svm_restrict b, double s, int dim) noexcept
{
    s *= -1;

    for ( int i = 0 ; i < dim ; ++i )
    {
        //a[i] += s*b[i];
        a[i] =  std::fma(s,b[i],a[i]);
    }

    return;
}

inline void fastMulBy(double *svm_restrict a, const double *svm_restrict b, int dim) noexcept
{
    for ( int i = 0 ; i < dim ; ++i )
    {
        a[i] *= b[i];
    }

    return;
}

inline void fastDivBy(double *svm_restrict a, const double *svm_restrict b, int dim) noexcept
{
    for ( int i = 0 ; i < dim ; ++i )
    {
        a[i] /= b[i];
    }

    return;
}

inline void fastAddTo(double *svm_restrict a, double b, int dim) noexcept
{
    for ( int i = 0 ; i < dim ; ++i )
    {
        a[i] += b;
    }

    return;
}

inline void fastSubTo(double *svm_restrict a, double b, int dim) noexcept
{
    for ( int i = 0 ; i < dim ; ++i )
    {
        a[i] -= b;
    }

    return;
}

inline void fastMulBy(double *svm_restrict a, double b, int dim) noexcept
{
    for ( int i = 0 ; i < dim ; ++i )
    {
        a[i] *= b;
    }

    return;
}

inline void fastDivBy(double *svm_restrict a, double b, int dim) noexcept
{
    for ( int i = 0 ; i < dim ; ++i )
    {
        a[i] /= b;
    }

    return;
}

inline void fastAddTo(double *svm_restrict a, const int *svm_restrict ai, int ab, int as, const double *svm_restrict b, const int *svm_restrict bi, int bb, int bs, int dim) noexcept
{
    for ( int i = 0 ; i < dim ; ++i )
    {
        a[ai[ab+(i*as)]] += b[bi[bb+(i*bs)]];
    }

    return;
}

inline void fastAddTo(double *svm_restrict a, const int *svm_restrict ai, int ab, int as, const double *svm_restrict b, const int *svm_restrict bi, int bb, int bs, double s, int dim) noexcept
{
    for ( int i = 0 ; i < dim ; ++i )
    {
        //a[ai[ab+(i*as)]] += s*b[bi[bb+(i*bs)]];
        a[ai[ab+(i*as)]] =  std::fma(s,b[bi[bb+(i*bs)]],a[ai[ab+(i*as)]]);
    }

    return;
}

inline void fastSubTo(double *svm_restrict a, const int *svm_restrict ai, int ab, int as, const double *svm_restrict b, const int *svm_restrict bi, int bb, int bs, int dim) noexcept
{
    for ( int i = 0 ; i < dim ; ++i )
    {
        a[ai[ab+(i*as)]] -= b[bi[bb+(i*bs)]];
    }

    return;
}

inline void fastSubTo(double *svm_restrict a, const int *svm_restrict ai, int ab, int as, const double *svm_restrict b, const int *svm_restrict bi, int bb, int bs, double s, int dim) noexcept
{
    s *= -1;

    for ( int i = 0 ; i < dim ; ++i )
    {
        //a[ai[ab+(i*as)]] += s*b[bi[bb+(i*bs)]];
        a[ai[ab+(i*as)]] =  std::fma(s,b[bi[bb+(i*bs)]],a[ai[ab+(i*as)]]);
    }

    return;
}

inline void fastMulBy(double *svm_restrict a, const int *svm_restrict ai, int ab, int as, const double *svm_restrict b, const int *svm_restrict bi, int bb, int bs, int dim) noexcept
{
    for ( int i = 0 ; i < dim ; ++i )
    {
        a[ai[ab+(i*as)]] *= b[bi[bb+(i*bs)]];
    }

    return;
}

inline void fastDivBy(double *svm_restrict a, const int *svm_restrict ai, int ab, int as, const double *svm_restrict b, const int *svm_restrict bi, int bb, int bs, int dim) noexcept
{
    for ( int i = 0 ; i < dim ; ++i )
    {
        a[ai[ab+(i*as)]] /= b[bi[bb+(i*bs)]];
    }

    return;
}

inline void fastAddTo(double *svm_restrict a, const int *svm_restrict ai, int ab, int as, double b, int dim) noexcept
{
    for ( int i = 0 ; i < dim ; ++i )
    {
        a[ai[ab+(i*as)]] += b;
    }

    return;
}

inline void fastSubTo(double *svm_restrict a, const int *svm_restrict ai, int ab, int as, double b, int dim) noexcept
{
    for ( int i = 0 ; i < dim ; ++i )
    {
        a[ai[ab+(i*as)]] -= b;
    }

    return;
}

inline void fastMulBy(double *svm_restrict a, const int *svm_restrict ai, int ab, int as, double b, int dim) noexcept
{
    for ( int i = 0 ; i < dim ; ++i )
    {
        a[ai[ab+(i*as)]] *= b;
    }

    return;
}

inline void fastDivBy(double *svm_restrict a, const int *svm_restrict ai, int ab, int as, double b, int dim) noexcept
{
    for ( int i = 0 ; i < dim ; ++i )
    {
        a[ai[ab+(i*as)]] /= b;
    }

    return;
}

#ifdef USE_PMMINTRIN
// source: stackoverflow
inline double fastoneProduct(const double *svm_restrict a, int n) noexcept
{
    __m256d vsum = _mm256_set_pd(0.0,0.0,0.0,0.0);

    double res = 0.0;

    // process 4 elements per iteration

    int k;

    for ( k = 0; k < n - 3; k += 4 )
    {
        __m256d va = _mm256_loadu_pd(&a[k]);

        vsum = _mm256_add_pd(vsum,va);
    }

    // horizontal sum of 4 partial dot products

    vsum = _mm256_hadd_pd(_mm256_permute2f128_pd(vsum,vsum,0x20),_mm256_permute2f128_pd(vsum,vsum,0x31));
    vsum = _mm256_hadd_pd(_mm256_permute2f128_pd(vsum,vsum,0x20),_mm256_permute2f128_pd(vsum,vsum,0x31));

    double tres[4];

    _mm256_storeu_pd(tres,vsum);

    res = *tres;

    // clean up any remaining elements

    for ( ; k < n; ++k )
    {
        res += a[k];
    }

    return res;
}

inline double fasttwoProduct(const double *svm_restrict a, const double *svm_restrict b, int n) noexcept
{
    __m256d vsum = _mm256_set_pd(0.0,0.0,0.0,0.0);

    double res = 0.0;

    int k;

    // process 4 elements per iteration

    for ( k = 0; k < n - 3; k += 4 )
    {
        __m256d va = _mm256_loadu_pd(&a[k]);
        __m256d vb = _mm256_loadu_pd(&b[k]);

        __m256d vs = _mm256_mul_pd(va,vb);

        vsum = _mm256_add_pd(vsum,vs);
    }

    // horizontal sum of 4 partial dot products

    vsum = _mm256_hadd_pd(_mm256_permute2f128_pd(vsum,vsum,0x20),_mm256_permute2f128_pd(vsum,vsum,0x31));
    vsum = _mm256_hadd_pd(_mm256_permute2f128_pd(vsum,vsum,0x20),_mm256_permute2f128_pd(vsum,vsum,0x31));

    double tres[4];

    _mm256_storeu_pd(tres,vsum);

    res = *tres;

    // clean up any remaining elements

    for ( ; k < n; ++k )
    {
        res += a[k]*b[k];
    }

    return res;
}

inline double fastthreeProduct(const double *svm_restrict a, const double *svm_restrict b, const double *svm_restrict c, int n) noexcept
{
    __m256d vsum = _mm256_set_pd(0.0,0.0,0.0,0.0);

    double res = 0.0;

    int k;

    // process 4 elements per iteration

    for ( k = 0; k < n - 3; k += 4 )
    {
        __m256d va = _mm256_loadu_pd(&a[k]);
        __m256d vb = _mm256_loadu_pd(&b[k]);
        __m256d vc = _mm256_loadu_pd(&c[k]);

        __m256d vs = _mm256_mul_pd(va,vb);
        __m256d vt = _mm256_mul_pd(vs,vc);

        vsum = _mm256_add_pd(vsum,vt);
    }

    // horizontal sum of 4 partial dot products

    vsum = _mm256_hadd_pd(_mm256_permute2f128_pd(vsum,vsum,0x20),_mm256_permute2f128_pd(vsum,vsum,0x31));
    vsum = _mm256_hadd_pd(_mm256_permute2f128_pd(vsum,vsum,0x20),_mm256_permute2f128_pd(vsum,vsum,0x31));

    double tres[4];

    _mm256_storeu_pd(tres,vsum);

    res = *tres;

    // clean up any remaining elements

    for ( ; k < n; ++k )
    {
        res += a[k]*b[k]*c[k];
    }

    return res;
}

inline double fastfourProduct(const double *svm_restrict a, const double *svm_restrict b, const double *svm_restrict c, const double *svm_restrict d, int n) noexcept
{
    __m256d vsum = _mm256_set_pd(0.0,0.0,0.0,0.0);

    double res = 0.0;

    int k;

    // process 4 elements per iteration

    for ( k = 0; k < n - 3; k += 4 )
    {
        __m256d va = _mm256_loadu_pd(&a[k]);
        __m256d vb = _mm256_loadu_pd(&b[k]);
        __m256d vc = _mm256_loadu_pd(&c[k]);
        __m256d vd = _mm256_loadu_pd(&d[k]);

        __m256d vs = _mm256_mul_pd(va,vb);
        __m256d vt = _mm256_mul_pd(vs,vc);
        __m256d vu = _mm256_mul_pd(vt,vd);

        vsum = _mm256_add_pd(vsum,vu);
    }

    // horizontal sum of 4 partial dot products

    vsum = _mm256_hadd_pd(_mm256_permute2f128_pd(vsum,vsum,0x20),_mm256_permute2f128_pd(vsum,vsum,0x31));
    vsum = _mm256_hadd_pd(_mm256_permute2f128_pd(vsum,vsum,0x20),_mm256_permute2f128_pd(vsum,vsum,0x31));

    double tres[4];

    _mm256_storeu_pd(tres,vsum);

    res = *tres;

    // clean up any remaining elements

    for ( ; k < n; ++k )
    {
        res += a[k]*b[k]*c[k]*d[k];
    }

    return res;
}

inline double fastfiveProduct(const double *svm_restrict a, const double *svm_restrict b, const double *svm_restrict c, const double *svm_restrict d, const double *svm_restrict e, int n) noexcept
{
    __m256d vsum = _mm256_set_pd(0.0,0.0,0.0,0.0);

    double res = 0.0;

    int k;

    // process 4 elements per iteration

    for ( k = 0; k < n - 3; k += 4 )
    {
        __m256d va = _mm256_loadu_pd(&a[k]);
        __m256d vb = _mm256_loadu_pd(&b[k]);
        __m256d vc = _mm256_loadu_pd(&c[k]);
        __m256d vd = _mm256_loadu_pd(&d[k]);
        __m256d ve = _mm256_loadu_pd(&e[k]);

        __m256d vs = _mm256_mul_pd(va,vb);
        __m256d vt = _mm256_mul_pd(vs,vc);
        __m256d vu = _mm256_mul_pd(vt,vd);
        __m256d vv = _mm256_mul_pd(vu,ve);

        vsum = _mm256_add_pd(vsum,vv);
    }

    // horizontal sum of 4 partial dot products

    vsum = _mm256_hadd_pd(_mm256_permute2f128_pd(vsum,vsum,0x20),_mm256_permute2f128_pd(vsum,vsum,0x31));
    vsum = _mm256_hadd_pd(_mm256_permute2f128_pd(vsum,vsum,0x20),_mm256_permute2f128_pd(vsum,vsum,0x31));

    double tres[4];

    _mm256_storeu_pd(tres,vsum);

    res = *tres;

    // clean up any remaining elements

    for ( ; k < n; ++k )
    {
        res += a[k]*b[k]*c[k]*d[k]*e[k];
    }

    return res;
}

inline double fastoneProduct(const double *svm_restrict a, const int *svm_restrict ai, int ab, int as, int n) noexcept
{
    __m256d vsum = _mm256_set_pd(0.0,0.0,0.0,0.0);

    double res = 0.0;

    int k;

    // process 4 elements per iteration

    for ( k = 0; k < n - 3; k += 4 )
    {
        __m256d va = _mm256_loadu_pd(&a[ai[ab+(k*as)]]);

        vsum = _mm256_add_pd(vsum,va);
    }

    // horizontal sum of 4 partial dot products

    vsum = _mm256_hadd_pd(_mm256_permute2f128_pd(vsum,vsum,0x20),_mm256_permute2f128_pd(vsum,vsum,0x31));
    vsum = _mm256_hadd_pd(_mm256_permute2f128_pd(vsum,vsum,0x20),_mm256_permute2f128_pd(vsum,vsum,0x31));

    double tres[4];

    _mm256_storeu_pd(tres,vsum);

    res = *tres;

    // clean up any remaining elements

    for ( ; k < n; ++k )
    {
        res += a[ai[ab+(k*as)]];
    }

    return res;
}

inline double fasttwoProduct(const double *svm_restrict a, const int *svm_restrict ai, int ab, int as, const double *svm_restrict b, const int *svm_restrict bi, int bb, int bs, int n) noexcept
{
    __m256d vsum = _mm256_set_pd(0.0,0.0,0.0,0.0);

    double res = 0.0;

    int k;

    // process 4 elements per iteration

    for ( k = 0; k < n - 3; k += 4 )
    {
        __m256d va = _mm256_loadu_pd(&a[ai[ab+(k*as)]]);
        __m256d vb = _mm256_loadu_pd(&b[bi[bb+(k*bs)]]);

        __m256d vs = _mm256_mul_pd(va,vb);

        vsum = _mm256_add_pd(vsum,vs);
    }

    // horizontal sum of 4 partial dot products

    vsum = _mm256_hadd_pd(_mm256_permute2f128_pd(vsum,vsum,0x20),_mm256_permute2f128_pd(vsum,vsum,0x31));
    vsum = _mm256_hadd_pd(_mm256_permute2f128_pd(vsum,vsum,0x20),_mm256_permute2f128_pd(vsum,vsum,0x31));

    double tres[4];

    _mm256_storeu_pd(tres,vsum);

    res = *tres;

    // clean up any remaining elements

    for ( ; k < n; ++k )
    {
        res += a[ai[ab+(k*as)]]*b[bi[bb+(k*bs)]];
    }

    return res;
}

inline double fastthreeProduct(const double *svm_restrict a, const int *svm_restrict ai, int ab, int as, const double *svm_restrict b, const int *svm_restrict bi, int bb, int bs, const double *svm_restrict c, const int *svm_restrict ci, int cb, int cs, int n) noexcept
{
    __m256d vsum = _mm256_set_pd(0.0,0.0,0.0,0.0);

    double res = 0.0;

    int k;

    // process 4 elements per iteration

    for ( k = 0; k < n - 3; k += 4 )
    {
        __m256d va = _mm256_loadu_pd(&a[ai[ab+(k*as)]]);
        __m256d vb = _mm256_loadu_pd(&b[bi[bb+(k*bs)]]);
        __m256d vc = _mm256_loadu_pd(&c[ci[cb+(k*cs)]]);

        __m256d vs = _mm256_mul_pd(va,vb);
        __m256d vt = _mm256_mul_pd(vs,vc);

        vsum = _mm256_add_pd(vsum,vt);
    }

    // horizontal sum of 4 partial dot products

    vsum = _mm256_hadd_pd(_mm256_permute2f128_pd(vsum,vsum,0x20),_mm256_permute2f128_pd(vsum,vsum,0x31));
    vsum = _mm256_hadd_pd(_mm256_permute2f128_pd(vsum,vsum,0x20),_mm256_permute2f128_pd(vsum,vsum,0x31));

    double tres[4];

    _mm256_storeu_pd(tres,vsum);

    res = *tres;

    // clean up any remaining elements

    for ( ; k < n; ++k )
    {
        res += a[ai[ab+(k*as)]]*b[bi[bb+(k*bs)]]*c[ci[cb+(k*cs)]];
    }

    return res;
}

inline double fastfourProduct(const double *svm_restrict a, const int *svm_restrict ai, int ab, int as, const double *svm_restrict b, const int *svm_restrict bi, int bb, int bs, const double *svm_restrict c, const int *svm_restrict ci, int cb, int cs, const double *svm_restrict d, const int *svm_restrict di, int db, int ds, int n) noexcept
{
    __m256d vsum = _mm256_set_pd(0.0,0.0,0.0,0.0);

    double res = 0.0;

    int k;

    // process 4 elements per iteration

    for ( k = 0; k < n - 3; k += 4 )
    {
        __m256d va = _mm256_loadu_pd(&a[ai[ab+(k*as)]]);
        __m256d vb = _mm256_loadu_pd(&b[bi[bb+(k*bs)]]);
        __m256d vc = _mm256_loadu_pd(&c[ci[cb+(k*cs)]]);
        __m256d vd = _mm256_loadu_pd(&d[di[db+(k*ds)]]);

        __m256d vs = _mm256_mul_pd(va,vb);
        __m256d vt = _mm256_mul_pd(vs,vc);
        __m256d vu = _mm256_mul_pd(vt,vd);

        vsum = _mm256_add_pd(vsum,vu);
    }

    // horizontal sum of 4 partial dot products

    vsum = _mm256_hadd_pd(_mm256_permute2f128_pd(vsum,vsum,0x20),_mm256_permute2f128_pd(vsum,vsum,0x31));
    vsum = _mm256_hadd_pd(_mm256_permute2f128_pd(vsum,vsum,0x20),_mm256_permute2f128_pd(vsum,vsum,0x31));

    double tres[4];

    _mm256_storeu_pd(tres,vsum);

    res = *tres;

    // clean up any remaining elements

    for ( ; k < n; ++k )
    {
        res += a[ai[ab+(k*as)]]*b[bi[bb+(k*bs)]]*c[ci[cb+(k*cs)]]*d[di[cb+(k*ds)]];
    }

    return res;
}

inline double fastfiveProduct(const double *svm_restrict a, const int *svm_restrict ai, int ab, int as, const double *svm_restrict b, const int *svm_restrict bi, int bb, int bs, const double *svm_restrict c, const int *svm_restrict ci, int cb, int cs, const double *svm_restrict d, const int *svm_restrict di, int db, int ds, const double *svm_restrict e, const int *svm_restrict ei, int eb, int es, int n) noexcept
{
    __m256d vsum = _mm256_set_pd(0.0,0.0,0.0,0.0);

    double res = 0.0;

    int k;

    // process 4 elements per iteration

    for ( k = 0; k < n - 3; k += 4 )
    {
        __m256d va = _mm256_loadu_pd(&a[ai[ab+(k*as)]]);
        __m256d vb = _mm256_loadu_pd(&b[bi[bb+(k*bs)]]);
        __m256d vc = _mm256_loadu_pd(&c[ci[cb+(k*cs)]]);
        __m256d vd = _mm256_loadu_pd(&d[di[db+(k*ds)]]);
        __m256d ve = _mm256_loadu_pd(&e[ei[eb+(k*es)]]);

        __m256d vs = _mm256_mul_pd(va,vb);
        __m256d vt = _mm256_mul_pd(vs,vc);
        __m256d vu = _mm256_mul_pd(vt,vd);
        __m256d vv = _mm256_mul_pd(vu,ve);

        vsum = _mm256_add_pd(vsum,vv);
    }

    // horizontal sum of 4 partial dot products

    vsum = _mm256_hadd_pd(_mm256_permute2f128_pd(vsum,vsum,0x20),_mm256_permute2f128_pd(vsum,vsum,0x31));
    vsum = _mm256_hadd_pd(_mm256_permute2f128_pd(vsum,vsum,0x20),_mm256_permute2f128_pd(vsum,vsum,0x31));

    double tres[4];

    _mm256_storeu_pd(tres,vsum);

    res = *tres;

    // clean up any remaining elements

    for ( ; k < n; ++k )
    {
        res += a[ai[ab+(k*as)]]*b[bi[bb+(k*bs)]]*c[ci[cb+(k*cs)]]*d[di[db+(k*ds)]]*e[ei[eb+(k*es)]];
    }

    return res;
}
#endif

#ifndef USE_PMMINTRIN
// We decouple operations here to enable the pipeline to do its thing
inline double fastoneProduct(const double *svm_restrict a, int n) noexcept
{
    double res = 0.0;

    for ( int i = 0 ; i < n ; ++i )
    {
        res += a[i];
    }

    return res;
}

inline double fasttwoProduct(const double *svm_restrict a, const double *svm_restrict b, int n) noexcept
{
    double res = 0.0;

    // Note the splitting of data-dependent operations to allow for pipelining of multiplication and assignment operations.

    for ( int i = 0 ; i < n ; ++i )
    {
        //res += a[i]*b[i];
        res = std::fma(a[i],b[i],res);
    }

    return res;
}

inline double fastthreeProduct(const double *svm_restrict a, const double *svm_restrict b, const double *svm_restrict c, int n) noexcept
{
    double res = 0.0;

    for ( int i = 0 ; i < n ; ++i )
    {
        //res += a[i]*b[i]*c[i];
        res = std::fma(a[i]*b[i],c[i],res);
    }

    return res;
}

inline double fastfourProduct(const double *svm_restrict a, const double *svm_restrict b, const double *svm_restrict c, const double *svm_restrict d, int n) noexcept
{
    double res = 0.0;

    for ( int i = 0 ; i < n ; ++i )
    {
        //res += a[i]*b[i]*c[i]*d[i];
        res = std::fma(a[i]*b[i],c[i]*d[i],res);
    }

    return res;
}

inline double fastfiveProduct(const double *svm_restrict a, const double *svm_restrict b, const double *svm_restrict c, const double *svm_restrict d, const double *svm_restrict e, int n) noexcept
{
    double res = 0.0;

    for ( int i = 0 ; i < n ; ++i )
    {
        //res += a[i]*b[i]*c[i]*d[i]*e[i];
        res = std::fma(a[i]*b[i],c[i]*d[i]*e[i],res);
    }

    return res;
}

inline double fastoneProduct(const double *svm_restrict a, int n, const double *svm_restrict scale) noexcept
{
    double res = 0.0;

    for ( int i = 0 ; i < n ; ++i )
    {
        res += a[i]/scale[i];
    }

    return res;
}

inline double fasttwoProduct(const double *svm_restrict a, const double *svm_restrict b, int n, const double *svm_restrict scale) noexcept
{
    double res = 0.0;

    // Note the splitting of data-dependent operations to allow for pipelining of multiplication and assignment operations.

    for ( int i = 0 ; i < n ; ++i )
    {
        //res += a[i]*b[i];
        res = std::fma(a[i]/scale[i],b[i]/scale[i],res);
    }

    return res;
}

inline double fastthreeProduct(const double *svm_restrict a, const double *svm_restrict b, const double *svm_restrict c, int n, const double *svm_restrict scale) noexcept
{
    double res = 0.0;

    for ( int i = 0 ; i < n ; ++i )
    {
        //res += a[i]*b[i]*c[i];
        res = std::fma((a[i]/scale[i])*(b[i]/scale[i]),c[i]/scale[i],res);
    }

    return res;
}

inline double fastfourProduct(const double *svm_restrict a, const double *svm_restrict b, const double *svm_restrict c, const double *svm_restrict d, int n, const double *svm_restrict scale) noexcept
{
    double res = 0.0;

    for ( int i = 0 ; i < n ; ++i )
    {
        //res += a[i]*b[i]*c[i]*d[i];
        res = std::fma((a[i]/scale[i])*(b[i]/scale[i]),(c[i]/scale[i])*(d[i]/scale[i]),res);
    }

    return res;
}

inline double fastfiveProduct(const double *svm_restrict a, const double *svm_restrict b, const double *svm_restrict c, const double *svm_restrict d, const double *svm_restrict e, int n, const double *svm_restrict scale) noexcept
{
    double res = 0.0;

    for ( int i = 0 ; i < n ; ++i )
    {
        //res += a[i]*b[i]*c[i]*d[i]*e[i];
        res = std::fma((a[i]/scale[i])*(b[i]/scale[i]),(c[i]/scale[i])*(d[i]/scale[i])*(e[i]/scale[i]),res);
    }

    return res;
}

inline double fastoneProduct(const double *svm_restrict a, const int *svm_restrict ai, int ab, int as, int n) noexcept
{
    double res = 0.0;

    for ( int i = 0 ; i < n ; ++i )
    {
        res += a[ai[ab+(i*as)]];
    }

    return res;
}

inline double fasttwoProduct(const double *svm_restrict a, const int *svm_restrict ai, int ab, int as, const double *svm_restrict b, const int *svm_restrict bi, int bb, int bs, int n) noexcept
{
    double res = 0.0;

    for ( int i = 0 ; i < n ; ++i )
    {
        //res += a[ai[ab+(i*as)]]*b[bi[bb+(i*bs)]];
        res = std::fma(a[ai[ab+(i*as)]],b[bi[bb+(i*bs)]],res);
    }

    return res;
}

inline double fastthreeProduct(const double *svm_restrict a, const int *svm_restrict ai, int ab, int as, const double *svm_restrict b, const int *svm_restrict bi, int bb, int bs, const double *svm_restrict c, const int *svm_restrict ci, int cb, int cs, int n) noexcept
{
    double res = 0.0;

    for ( int i = 0 ; i < n ; ++i )
    {
        //res += a[ai[ab+(i*as)]]*b[bi[bb+(i*bs)]]*c[ci[cb+(i*cs)]];
        res = std::fma(a[ai[ab+(i*as)]]*b[bi[bb+(i*bs)]],c[ci[cb+(i*cs)]],res);
    }

    return res;
}

inline double fastfourProduct(const double *svm_restrict a, const int *svm_restrict ai, int ab, int as, const double *svm_restrict b, const int *svm_restrict bi, int bb, int bs, const double *svm_restrict c, const int *svm_restrict ci, int cb, int cs, const double *svm_restrict d, const int *svm_restrict di, int db, int ds, int n) noexcept
{
    double res = 0.0;

    for ( int i = 0 ; i < n ; ++i )
    {
        //res += a[ai[ab+(i*as)]]*b[bi[bb+(i*bs)]]*c[ci[cb+(i*cs)]]*d[di[db+(i*ds)]];
        res = std::fma(a[ai[ab+(i*as)]]*b[bi[bb+(i*bs)]],c[ci[cb+(i*cs)]]*d[di[db+(i*ds)]],res);
    }

    return res;
}

inline double fastfiveProduct(const double *svm_restrict a, const int *svm_restrict ai, int ab, int as, const double *svm_restrict b, const int *svm_restrict bi, int bb, int bs, const double *svm_restrict c, const int *svm_restrict ci, int cb, int cs, const double *svm_restrict d, const int *svm_restrict di, int db, int ds, const double *svm_restrict e, const int *svm_restrict ei, int eb, int es, int n) noexcept
{
    double res = 0.0;

    for ( int i = 0 ; i < n ; ++i )
    {
        //res += a[ai[ab+(i*as)]]*b[bi[bb+(i*bs)]]*c[ci[cb+(i*cs)]]*d[di[db+(i*ds)]]*e[ei[eb+(i*es)]];
        res = std::fma(a[ai[ab+(i*as)]]*b[bi[bb+(i*bs)]],c[ci[cb+(i*cs)]]*d[di[db+(i*ds)]]*e[ei[eb+(i*es)]],res);
    }

    return res;
}
#endif

inline double fastoneProductSparse  (const double *svm_restrict a, const int *svm_restrict ai, int an) noexcept
{
    (void) ai;

    double res = 0.0;

    for ( int i = 0 ; i < an ; ++i )
    {
        res += a[i];
    }

    return res;
}

inline double fasttwoProductSparse  (const double *svm_restrict a, const int *svm_restrict ai, int an, const double *svm_restrict b, const int *svm_restrict bi, int bn) noexcept
{
    double res = 0.0;

    if ( an && bn )
    {
        int apos = 0;
        int bpos = 0;

	int aelm;
        int belm;

        while ( ( apos < an ) && ( bpos < bn ) )
	{
            aelm = ai[apos];
            belm = bi[bpos];

	    if ( aelm == belm )
	    {
                res = std::fma(a[apos],b[bpos],res);

		++apos;
                ++bpos;
	    }

	    else if ( aelm < belm )
	    {
		++apos;
	    }

	    else
	    {
                ++bpos;
	    }
	}
    }

    return res;
}

inline double fastthreeProductSparse(const double *svm_restrict a, const int *svm_restrict ai, int an, const double *svm_restrict b, const int *svm_restrict bi, int bn, const double *svm_restrict c, const int *svm_restrict ci, int cn) noexcept
{
    double res = 0.0;

    if ( an && bn && cn )
    {
        int apos = 0;
        int bpos = 0;
        int cpos = 0;

	int aelm;
        int belm;
        int celm;

        while ( ( apos < an ) && ( bpos < bn ) && ( cpos < cn ) )
	{
            aelm = ai[apos];
            belm = bi[bpos];
            celm = ci[cpos];

	    if ( ( aelm == belm ) && ( aelm == celm ) )
	    {
                res = std::fma(a[apos]*b[bpos],c[cpos],res);

		++apos;
                ++bpos;
                ++cpos;
	    }

            else if ( ( aelm <= belm ) && ( aelm <= celm ) )
	    {
		++apos;
	    }

            else if ( ( belm <= aelm ) && ( belm <= celm ) )
	    {
		++bpos;
	    }

            else
	    {
		++cpos;
	    }
	}
    }

    return res;
}

inline double fastfourProductSparse (const double *svm_restrict a, const int *svm_restrict ai, int an, const double *svm_restrict b, const int *svm_restrict bi, int bn, const double *svm_restrict c, const int *svm_restrict ci, int cn, const double *svm_restrict d, const int *svm_restrict di, int dn) noexcept
{
    double res = 0.0;

    if ( an && bn && cn && dn )
    {
        int apos = 0;
        int bpos = 0;
        int cpos = 0;
        int dpos = 0;

	int aelm;
        int belm;
        int celm;
        int delm;

        while ( ( apos < an ) && ( bpos < bn ) && ( cpos < cn ) && ( dpos < dn ) )
	{
            aelm = ai[apos];
            belm = bi[bpos];
            celm = ci[cpos];
            delm = di[dpos];

	    if ( ( aelm == belm ) && ( aelm == celm ) && ( aelm == delm ) )
	    {
                res = std::fma(a[apos]*b[bpos],c[cpos]*d[dpos],res);

		++apos;
                ++bpos;
                ++cpos;
                ++dpos;
	    }

            else if ( ( aelm <= belm ) && ( aelm <= celm ) && ( aelm <= delm ) )
	    {
		++apos;
	    }

            else if ( ( belm <= aelm ) && ( belm <= celm ) && ( belm <= delm ) )
	    {
		++bpos;
	    }

            else if ( ( celm <= aelm ) && ( celm <= belm ) && ( celm <= delm ) )
	    {
		++cpos;
	    }

	    else
	    {
                ++dpos;
	    }
	}
    }

    return res;
}

inline double fastfiveProductSparse (const double *svm_restrict a, const int *svm_restrict ai, int an, const double *svm_restrict b, const int *svm_restrict bi, int bn, const double *svm_restrict c, const int *svm_restrict ci, int cn, const double *svm_restrict d, const int *svm_restrict di, int dn, const double *svm_restrict e, const int *svm_restrict ei, int en) noexcept
{
    double res = 0.0;

    if ( an && bn && cn && dn && en )
    {
        int apos = 0;
        int bpos = 0;
        int cpos = 0;
        int dpos = 0;
        int epos = 0;

	int aelm;
        int belm;
        int celm;
        int delm;
        int eelm;

        while ( ( apos < an ) && ( bpos < bn ) && ( cpos < cn ) && ( dpos < dn ) && ( epos < en ) )
	{
            aelm = ai[apos];
            belm = bi[bpos];
            celm = ci[cpos];
            delm = di[dpos];
            eelm = ei[epos];

	    if ( ( aelm == belm ) && ( aelm == celm ) && ( aelm == delm ) && ( aelm == eelm ) )
	    {
                res = std::fma(a[apos]*b[bpos],c[cpos]*d[dpos]*e[epos],res);

		++apos;
                ++bpos;
                ++cpos;
                ++dpos;
                ++epos;
	    }

            else if ( ( aelm <= belm ) && ( aelm <= celm ) && ( aelm <= delm ) && ( aelm <= eelm ) )
	    {
		++apos;
	    }

            else if ( ( belm <= aelm ) && ( belm <= celm ) && ( belm <= delm ) && ( belm <= eelm ) )
	    {
		++bpos;
	    }

            else if ( ( celm <= aelm ) && ( celm <= belm ) && ( celm <= delm ) && ( celm <= eelm ) )
	    {
		++cpos;
	    }

            else if ( ( delm <= aelm ) && ( delm <= belm ) && ( delm <= celm ) && ( delm <= eelm ) )
	    {
		++dpos;
	    }

	    else
	    {
                ++epos;
	    }
	}
    }

    return res;
}




#endif
