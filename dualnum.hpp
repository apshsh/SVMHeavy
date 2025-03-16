
//
// Dual numbers
//
// Version: 1
// Date: 02/03/2021
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//
// Implements dual numbers x + y.d, where d^2 = 0
//


#ifndef _dualnum_h
#define _dualnum_h

#include <iostream>
#include <math.h>
#include <stdlib.h>
#include "basefn.hpp"


template <class T>
class dualnum;

template <class T>
void qswap(dualnum<T> &a, dualnum<T> &b);

template <class T>
dualnum<T> atodualnum(const char *qwerty, int len = -1);
template <class T>
int atodualnum(dualnum<T> &result, const char *qwerty, int len = -1);

// streams

template <class T>
std::ostream &operator<<(std::ostream &output, const dualnum<T> &source);
template <class T>
std::istream &operator>>(std::istream &input ,       dualnum<T> &destin);


template <class T>
class dualnum
{
    public:

    /*
       Constructors and destructors.
    */

    explicit dualnum()                         : value_re(0.0),          value_im(0.0)          {                        }
    explicit dualnum(const T          &src)    : value_re(src),          value_im(0.0)          {                        }
             dualnum(const dualnum<T> &src)    : value_re(src.value_re), value_im(src.value_im) {                        }
    explicit dualnum(const char       *src)    : value_re(0.0),          value_im(0.0)          { atodualnum(*this,src); }
    explicit dualnum(const T &re, const T &im) : value_re(re),           value_im(im)           {                        }

    /*
       Assignment operators
    */

    dualnum<T> &operator=(const T          &val) { value_re = val;          value_im = 0.0;          return *this; }
    dualnum<T> &operator=(const dualnum<T> &val) { value_re = src.value_re; value_im = src.value_im; return *this; }
    dualnum<T> &operator=(const char       *val) { atodualnum(*this,src);                            return *this; }

    /*
       Information functions.
    */

    int order(void)              const { return iscomplex() ? 1 : 0;   }
    int size(void)               const { return 1 << order();          }
    int isreal(void)             const { return !iscomplex();          }
    int iscomplex(void)          const { return value_im == 0 ? 0 : 1; }
    int iscommutative(void)      const { return 1;                     }
    int isassociative(void)      const { return 1;                     }
    int ispowerassociative(void) const { return 1;                     }
    int isindet(void)            const;

    const T &operator()(              int i) const { NiceAssert( ( i == 0 ) || ( i == 1 ) ); const static T zv(0.0); return !i ? &value_re : &value_im; }
          T &operator()(const char *, int i)       { NiceAssert( ( i == 0 ) || ( i == 1 ) ); const static T zv(0.0); return !i ? &value_re : &value_im; }

    /*
       String casting operator
    */

    std::string &tostring(std::string &dest) const

    //private: use these if you like

    T value_re;
    T value_im;
};

template <class T>
void qswap(dualnum<T> &a, dualnum<T> &b)
{
    // NOTE: we are not dealing with a single element here but rather a node
    // in a larger tree of elements (potentially).  Therefore we cannot simply
    // use the naive swap-all-elements approach.  Better to simply use value
    // copying and hope that this never gets used too often.

    qswap(a.value_re,value_re);
    qswap(a.value_im,value_im);

    return;
}


template <class T> const dualnum<T> &defaultdualnum(void) { const static dualnum<T> defval;       return defval;  }
template <class T> const dualnum<T> &zerodualnum   (void) { const static dualnum<T> zeroval(0.0); return zeroval; }

template <class T> dualnum<T> &oneProduct  (dualnum<T> &res, const dualnum<T> &a)                                                                { res = a;                               return res; }
template <class T> dualnum<T> &twoProduct  (dualnum<T> &res, const dualnum<T> &a, const dualnum<T> &b)                                           { res = a; res *= b;                     return res; }
template <class T> dualnum<T> &threeProduct(dualnum<T> &res, const dualnum<T> &a, const dualnum<T> &b, const dualnum<T> &c)                      { res = a; res *= b; res *= c            return res; }
template <class T> dualnum<T> &fourProduct (dualnum<T> &res, const dualnum<T> &a, const dualnum<T> &b, const dualnum<T> &c, const dualnum<T> &d) { res = a; res *= b; res *= c; res *= d; return res; }
template <class T> dualnum<T> &mProduct    (dualnum<T> &res, int m, const dualnum<T> *a);

template <class T> dualnum<T> &innerProduct       (dualnum<T> &res, const dualnum<T> &a, const dualnum<T> &b) { res = a; setconj(res); res *= b;         return res; }
template <class T> dualnum<T> &innerProductRevConj(dualnum<T> &res, const dualnum<T> &a, const dualnum<T> &b) { res = b; setconj(res); rightmult(a,res); return res; }

// + posation - unary, return rvalue
// - negation - unary, return rvalue

template <class T> dualnum<T> operator+(const dualnum<T> &left_op) {                             return left_op;        }
template <class T> dualnum<T> operator-(const dualnum<T> &left_op) { dualnum<T> result(left_op); return result *= -1.0; }

// + addition       - binary, return rvalue
// - subtraction    - binary, return rvalue

template <class T> dualnum<T> operator+(const T          &left_op, const dualnum<T> &right_op) { dualnum<T> temp(left_op); return temp += right_op; }
template <class T> dualnum<T> operator+(const dualnum<T> &left_op, const T          &right_op) { dualnum<T> temp(left_op); return temp += right_op; }
template <class T> dualnum<T> operator+(const dualnum<T> &left_op, const dualnum<T> &right_op) { dualnum<T> temp(left_op); return temp += right_op; }

template <class T> dualnum<T> operator-(const T          &left_op, const dualnum<T> &right_op) { dualnum<T> temp(left_op); return temp -= right_op; }
template <class T> dualnum<T> operator-(const dualnum<T> &left_op, const T          &right_op) { dualnum<T> temp(left_op); return temp -= right_op; }
template <class T> dualnum<T> operator-(const dualnum<T> &left_op, const dualnum<T> &right_op) { dualnum<T> temp(left_op); return temp -= right_op; }

// += additive       assignment - binary, return lvalue
// -= subtractive    assignment - binary, return lvalue

template <class T> dualnum<T> &operator+=(dualnum<T> &left_op, const dualnum<T> &right_op) { left_op.value_re += right_op.value_re; left_op.value_im  += right_op.value_im; return left_op; }
template <class T> dualnum<T> &operator+=(dualnum<T> &left_op, const T          &right_op) { left_op.value_re += right_op;                                                  return left_op; }

template <class T> dualnum<T> &operator-=(dualnum<T> &left_op, const dualnum<T> &right_op) { left_op.value_re -= right_op.value_re; left_op.value_im -= right_op.value_im;  return left_op; }
template <class T> dualnum<T> &operator-=(dualnum<T> &left_op, const T  &right_op) { left_op.value_re -= right_op;                                                  return left_op; }

// * multiplication - binary, return rvalue
// / division       - binary, return rvalue

template <class T> dualnum<T> operator*(const T  &left_op, const dualnum<T> &right_op) { dualnum<T> result(left_op); return ( result *= right_op ); }
template <class T> dualnum<T> operator*(const dualnum<T> &left_op, const T  &right_op) { dualnum<T> result(left_op); return ( result *= right_op ); }
template <class T> dualnum<T> operator*(const dualnum<T> &left_op, const dualnum<T> &right_op) { dualnum<T> result(left_op); return ( result *= right_op ); }

template <class T> dualnum<T> operator/(const T  &left_op, const dualnum<T> &right_op) { dualnum<T> result(left_op); return ( result /= right_op ); }
template <class T> dualnum<T> operator/(const dualnum<T> &left_op, const T  &right_op) { dualnum<T> result(left_op); return ( result /= right_op ); }
template <class T> dualnum<T> operator/(const dualnum<T> &left_op, const dualnum<T> &right_op) { dualnum<T> result(left_op); return ( result /= right_op ); }

// *= multiplicative assignment - binary, return lvalue
// /= divisive       assignment - binary, return lvalue
//
// leftmult:  overwrite left_op with left_op*right_op
// rightmult: overwrite right_op with left_op*right_op

template <class T> dualnum<T> &operator*=(dualnum<T> &left_op, const T  &right_op) { return leftmult(left_op,right_op); }
template <class T> dualnum<T> &operator*=(dualnum<T> &left_op, const dualnum<T> &right_op) { return leftmult(left_op,right_op); }

template <class T> dualnum<T> &operator/=(dualnum<T> &left_op, const T  &right_op) { left_op.value_re /= right_op; left_op.value_im /= right_op; return left_op; }
template <class T> dualnum<T> &operator/=(dualnum<T> &left_op, const dualnum<T> &right_op);

template <class T> dualnum<T> &leftmult(dualnum<T> &left_op, const T  &right_op) { left_op.value_re  *= right_op; left_op.value_im  *= right_op; return left_op;  }
template <class T> dualnum<T> &leftmult(dualnum<T> &left_op, const dualnum<T> &right_op);

template <class T> dualnum<T> &rightmult(const T  &left_op, dualnum<T> &right_op) { right_op.value_re *= left_op;  right_op.value_im *= left_op;  return right_op; }
template <class T> dualnum<T> &rightmult(const dualnum<T> &left_op, dualnum<T> &right_op);

// comparison operators

template <class T> int operator==(const dualnum<T> &left_op, const dualnum<T> &right_op) { return ( left_op.value_re == right_op.value_re ) && ( left_op.value_im == right_op.value_im ); }
template <class T> int operator!=(const dualnum<T> &left_op, const dualnum<T> &right_op) { return ( left_op.value_re != right_op.value_re ) || ( left_op.value_im != right_op.value_im ); }
template <class T> int operator<=(const dualnum<T> &left_op, const dualnum<T> &right_op) { return ( left_op.value_re < right_op.value_re ) || ( ( left_op.value_re == right_op.value_re ) && ( left_op.value_im <= right_op.value_im ) ); }
template <class T> int operator>=(const dualnum<T> &left_op, const dualnum<T> &right_op) { return ( left_op.value_re > right_op.value_re ) || ( ( left_op.value_re == right_op.value_re ) && ( left_op.value_im >= right_op.value_im ) ); }
template <class T> int operator< (const dualnum<T> &left_op, const dualnum<T> &right_op) { return ( left_op.value_re < right_op.value_re ) || ( ( left_op.value_re == right_op.value_re ) && ( left_op.value_im < right_op.value_im ) ); }
template <class T> int operator> (const dualnum<T> &left_op, const dualnum<T> &right_op) { return ( left_op.value_re > right_op.value_re ) || ( ( left_op.value_re == right_op.value_re ) && ( left_op.value_im > right_op.value_im ) ); }

template <class T> int operator==(const T  &left_op, const dualnum<T> &right_op) { return ( left_op == right_op.value_re ) && ( 0.0 == right_op.value_im ); }
template <class T> int operator!=(const T  &left_op, const dualnum<T> &right_op) { return ( left_op != right_op.value_re ) || ( 0.0 != right_op.value_im ); }
template <class T> int operator<=(const T  &left_op, const dualnum<T> &right_op) { return ( left_op < right_op.value_re ) || ( ( left_op == right_op.value_re ) && ( 0.0 <= right_op.value_im ) ); }
template <class T> int operator>=(const T  &left_op, const dualnum<T> &right_op) { return ( left_op > right_op.value_re ) || ( ( left_op == right_op.value_re ) && ( 0.0 >= right_op.value_im ) ); }
template <class T> int operator< (const T  &left_op, const dualnum<T> &right_op) { return ( left_op < right_op.value_re ) || ( ( left_op == right_op.value_re ) && ( 0.0 <  right_op.value_im ) ); }
template <class T> int operator> (const T  &left_op, const dualnum<T> &right_op) { return ( left_op > right_op.value_re ) || ( ( left_op == right_op.value_re ) && ( 0.0 >  right_op.value_im ) ); }

template <class T> int operator==(const dualnum<T> &left_op, const T  &right_op) { return ( left_op.value_re == right_op ) && ( left_op.value_im == 0.0 ); }
template <class T> int operator!=(const dualnum<T> &left_op, const T  &right_op) { return ( left_op.value_re != right_op ) || ( left_op.value_im != 0.0 ); }
template <class T> int operator<=(const dualnum<T> &left_op, const T  &right_op) { return ( left_op.value_re < right_op ) || ( ( left_op.value_re == right_op ) && ( left_op.value_im <= 0.0 ) ); }
template <class T> int operator>=(const dualnum<T> &left_op, const T  &right_op) { return ( left_op.value_re > right_op ) || ( ( left_op.value_re == right_op ) && ( left_op.value_im >= 0.0 ) ); }
template <class T> int operator< (const dualnum<T> &left_op, const T  &right_op) { return ( left_op.value_re < right_op ) || ( ( left_op.value_re == right_op ) && ( left_op.value_im <  0.0 ) ); }
template <class T> int operator> (const dualnum<T> &left_op, const T  &right_op) { return ( left_op.value_re > right_op ) || ( ( left_op.value_re == right_op ) && ( left_op.value_im >  0.0 ) ); }



// NaN and inf tests

template <class T> int testisvnan(const dualnum<T> &x) { return testisvnan(x.value_re) || testisvnan(x.value_im); }
template <class T> int testisinf (const dualnum<T> &x) { return testisinf (x.value_re) || testisinf (x.value_im); }
template <class T> int testispinf(const dualnum<T> &x) { return testispinf(x.value_re) || testisninf(x.value_im); }
template <class T> int testisninf(const dualnum<T> &x) { return testispinf(x.value_re) || testisninf(x.value_im); }

// Non-member functions:
//
// f(a+d.b) = f(a) + b.f'(a).d

T  abs1  (const dualnum<T> &a);
T  abs2  (const dualnum<T> &a);
T  absp  (const dualnum<T> &a, const T &p);
T  absinf(const dualnum<T> &a);
T  abs0  (const dualnum<T> &a);
T  absd  (const dualnum<T> &a);
T  norm1 (const dualnum<T> &a);
T  norm2 (const dualnum<T> &a);
T  normp (const dualnum<T> &a, const T &p);
T  normd (const dualnum<T> &a);
dualnum<T> sgn   (const dualnum<T> &a);

T  real(const dualnum<T> &a);
T  imag(const dualnum<T> &a);
dualnum<T> conj(const dualnum<T> &a);
dualnum<T> inv (const dualnum<T> &a);

dualnum<T> pow (const long    &a, const dualnum<T> &b);
dualnum<T> pow (const T  &a, const dualnum<T> &b);
dualnum<T> pow (const dualnum<T> &a, const long    &b);
dualnum<T> pow (const dualnum<T> &a, const T  &b);
dualnum<T> pow (const dualnum<T> &a, const dualnum<T> &b);
dualnum<T> sqrt(const dualnum<T> &a);

dualnum<T> exp  (const dualnum<T> &a);
dualnum<T> tenup(const dualnum<T> &a);
dualnum<T> log  (const dualnum<T> &a);
dualnum<T> log10(const dualnum<T> &a);
dualnum<T> logb (const long    &a, const dualnum<T> &b);
dualnum<T> logb (const T  &a, const dualnum<T> &b);
dualnum<T> logb (const dualnum<T> &a, const long   &b);
dualnum<T> logb (const dualnum<T> &a, const T  &b);
dualnum<T> logb (const dualnum<T> &a, const dualnum<T> &b);

dualnum<T> sin     (const dualnum<T> &a);
dualnum<T> cos     (const dualnum<T> &a);
dualnum<T> tan     (const dualnum<T> &a);
dualnum<T> cosec   (const dualnum<T> &a);
dualnum<T> sec     (const dualnum<T> &a);
dualnum<T> cot     (const dualnum<T> &a);
dualnum<T> asin    (const dualnum<T> &a);
dualnum<T> acos    (const dualnum<T> &a);
dualnum<T> atan    (const dualnum<T> &a);
dualnum<T> acosec  (const dualnum<T> &a);
dualnum<T> asec    (const dualnum<T> &a);
dualnum<T> acot    (const dualnum<T> &a);
dualnum<T> sinc    (const dualnum<T> &a);
dualnum<T> cosc    (const dualnum<T> &a);
dualnum<T> tanc    (const dualnum<T> &a);
dualnum<T> vers    (const dualnum<T> &a);
dualnum<T> covers  (const dualnum<T> &a);
dualnum<T> hav     (const dualnum<T> &a);
dualnum<T> excosec (const dualnum<T> &a);
dualnum<T> exsec   (const dualnum<T> &a);
dualnum<T> avers   (const dualnum<T> &a);
dualnum<T> acovers (const dualnum<T> &a);
dualnum<T> ahav    (const dualnum<T> &a);
dualnum<T> aexcosec(const dualnum<T> &a);
dualnum<T> aexsec  (const dualnum<T> &a);

dualnum<T> sinh     (const dualnum<T> &a);
dualnum<T> cosh     (const dualnum<T> &a);
dualnum<T> tanh     (const dualnum<T> &a);
dualnum<T> cosech   (const dualnum<T> &a);
dualnum<T> sech     (const dualnum<T> &a);
dualnum<T> coth     (const dualnum<T> &a);
dualnum<T> asinh    (const dualnum<T> &a);
dualnum<T> acosh    (const dualnum<T> &a);
dualnum<T> atanh    (const dualnum<T> &a);
dualnum<T> acosech  (const dualnum<T> &a);
dualnum<T> asech    (const dualnum<T> &a);
dualnum<T> acoth    (const dualnum<T> &a);
dualnum<T> sinhc    (const dualnum<T> &a);
dualnum<T> coshc    (const dualnum<T> &a);
dualnum<T> tanhc    (const dualnum<T> &a);
dualnum<T> versh    (const dualnum<T> &a);
dualnum<T> coversh  (const dualnum<T> &a);
dualnum<T> havh     (const dualnum<T> &a);
dualnum<T> excosech (const dualnum<T> &a);
dualnum<T> exsech   (const dualnum<T> &a);
dualnum<T> aversh   (const dualnum<T> &a);
dualnum<T> acovrsh  (const dualnum<T> &a);
dualnum<T> ahavh    (const dualnum<T> &a);
dualnum<T> aexcosech(const dualnum<T> &a);
dualnum<T> aexsech  (const dualnum<T> &a);

dualnum<T> sigm (const dualnum<T> &a);
dualnum<T> gd   (const dualnum<T> &a);
dualnum<T> asigm(const dualnum<T> &a);
dualnum<T> agd  (const dualnum<T> &a);



// Other functions for anions

template <class T> dualnum<T> &setident        (dualnum<T> &a) { a.value_re  =  1;    a.value_im  =  0;    return a; }
template <class T> dualnum<T> &setzero         (dualnum<T> &a) { a.value_re  =  0;    a.value_im  =  0;    return a; }
template <class T> dualnum<T> &setposate       (dualnum<T> &a) {                                           return a; }
template <class T> dualnum<T> &setnegate       (dualnum<T> &a) { a.value_re *= -1;    a.value_im *= -1;    return a; }
template <class T> dualnum<T> &setconj         (dualnum<T> &a) {                      a.value_im *= -1;    return a; }
template <class T> dualnum<T> &setrand         (dualnum<T> &a) { setrand(a.value_re); setrand(a.value_im); return a; }
template <class T> dualnum<T> &postProInnerProd(dualnum<T> &a) {                                           return a; }










// non-trivial inlines

template <class T> dualnum<T> &leftmult (dualnum<T> &left_op, const dualnum<T> &right_op)
{
    T leftre = left_op.value_re;
    T leftim = left_op.value_im;

    T rightre = right_op.value_re;
    T rightim = right_op.value_im;

    left_op.value_re = leftre*rightre;
    left_op.value_im = (leftre*rightim)+(leftim*rightre);

    return left_op;
}

template <class T> dualnum<T> &rightmult(const dualnum<T> &left_op, dualnum<T> &right_op)
{
    T leftre = left_op.value_re;
    T leftim = left_op.value_im;

    T rightre = right_op.value_re;
    T rightim = right_op.value_im;

    right_op.value_re = leftre*rightre;
    right_op.value_im = (leftre*rightim)+(leftim*rightre);

    return right_op;
}

template <class T> dualnum<T> &operator/=(dualnum<T> &left_op, const dualnum<T> &right_op)
{
    T leftre = left_op.value_re;
    T leftim = left_op.value_im;

    T rightre = right_op.value_re;
    T rightim = right_op.value_im;

    right_op.value_re = leftre/rightre;
    right_op.value_im = ((leftim*rightre)-(leftre*rightim))/(rightre*rightre);

    return right_op;
}


template <class T> dualnum<T> &mProduct(dualnum<T> &res, int m, const dualnum<T> *a)
{
    NiceAssert( m >= 0 );

    setident(res);

    if ( m > 0 )
    {
        int i;

        for ( i = 0 ; i < m ; ++i )
        {
            res *= a[i];
        }
    }

    return res;
}

#endif
