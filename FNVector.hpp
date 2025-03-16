//FOR [ x ~ xb ~ ... ] in FNVector, need to move the "up" part of the expansion K2 in ml_base.cc to mercer.h top-level K2 evaluation, including UPNTVI code (remember to add iupm, jupm == 1 test to 
//xymatrix shortcut generator code)



//
// Functional and RKHS Vector class
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _fnvector_h
#define _fnvector_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include <sstream>
#include "vector.hpp"
#include "sparsevector.hpp"
#include "gentype.hpp"
#include "mercer.hpp"
#include "mlcommon.hpp"


class FuncVector;
class RKHSVector;
class BernVector;


// These classes represent L2 functions on [0,1]^d.  
//
// FuncVector: - the function is defined directly as a function, and the 
//               dimension given.
//             - Format is [[ FN f: fn : d ]], where fn is the function and d the
//               dimension.
//             - Inner products etc are defined by approximating on a grid 
//               over this set.  
//             - If the function evaluates as nullptr at some point then this 
//               point is not included in the sum (etc), and the average is 
//               adjusted not to include this gridpoint.
//             - Lp inner products and norms also defined.
//
// RKHSVector: - the function is defined as sum_i alpha_i K(x,x_i)
//             - K, alpha_i and x_i therefore define the function.
//             - inner products etc are RKHS inner products
//               (sum_ij alpha_i alpha_j' K(x_i,x_j')
//             - Kernels must match for inner products etc!
//             - Lp inner products etc defined via m-kernels
//
//             - if m != 2 then this is an m-RKHS, so:
//               f(x) = sum_{i2,i3,...,im} alpha_i2 alpha_i3 ... alpha_im Km(x_i2,x_i3,...,x_im,x)
//               or, more generally, using the ~ feature in sparsevectors
//               f(x1 ~ x2 ~ ... ~ xn) = sum_{i2,i3,...,im} alpha_i2 alpha_i3 ... alpha_im K{m+n-1}(x_i2,x_i3,...,x_im,x1,x1,...,xn)
//               (unless the kernel is defined to use this feature for something else)
//             - The p-inner product is:
//               <v1,v1,...,vp>_p = sum_{i12,i13,...,i1m1,i22,i23,...,i2m2, ..., ip2,ip3,...,ipmp} alpha1_i12 alpha1_i23 ... alpha1_i1m1 alpha2_i22 alpha2_i23 ... alpha2_i2m2 ...
//                                                  ... alphap_ip2 alphap_ip3 ... alphap_ipmp Km(x1_i12,x1_i13,...,x1_i1m1,x2_i22,x2_i23,...,x2_i2m2,...,xp_ip2,xp_ip3,...,xp_ipmp)
//               where m = m1+m2+...+mp
//             - and the induced p-norm becomes:
//               ||v||_p^p = <v,v,...,v>_p
//
//             - use the alpha_as_vector feature for different weights:
//               f(x) = sum_{i2,i3,...,im} alpha_i2[0] alpha_i3[1] ... alpha_im[m-2] Km(x_i2,x_i3,...,x_im,x)
//
// BernVector: - the function is defined by weights wrt Bernstein basis
//             - the weight is w
//
// Notes:
//
// - The m-product between FuncVector and RKHSVector reverts to FuncVector
// - sums of vectors are defined, but...
//   - the sum of RKHSVectors are only defined if kernels match
//   - the sum of RKHSVector and FuncVector reverts to FuncVector



// Stream operators

std::ostream &operator<<(std::ostream &output, const FuncVector &src );
std::istream &operator>>(std::istream &input,        FuncVector &dest);
std::istream &streamItIn(std::istream &input,        FuncVector &dest, int processxyzvw = 1);

std::ostream &operator<<(std::ostream &output, const RKHSVector &src );
std::istream &operator>>(std::istream &input,        RKHSVector &dest);
std::istream &streamItIn(std::istream &input,        RKHSVector &dest, int processxyzvw = 1);

std::ostream &operator<<(std::ostream &output, const BernVector &src );
std::istream &operator>>(std::istream &input,        BernVector &dest);
std::istream &streamItIn(std::istream &input,        BernVector &dest, int processxyzvw = 1);

// Swap function

inline void qswap(const FuncVector *&a, const FuncVector *&b);
inline void qswap(FuncVector       *&a, FuncVector       *&b);
inline void qswap(FuncVector        &a, FuncVector        &b);

inline void qswap(const RKHSVector *&a, const RKHSVector *&b);
inline void qswap(RKHSVector       *&a, RKHSVector       *&b);
inline void qswap(RKHSVector        &a, RKHSVector        &b);

inline void qswap(const BernVector *&a, const BernVector *&b);
inline void qswap(BernVector       *&a, BernVector       *&b);
inline void qswap(BernVector        &a, BernVector        &b);

// Creation operators

void makeFuncVector(const std::string &typestring, Vector<gentype> *&res, std::istream &src);
void makeFuncVector(const std::string &typestring, Vector<gentype> *&res, std::istream &src, int processxyzvw);

// Calculate L2 distance squared from RKHSVector to function of given dimension,
// assuming a function of var(0,0), var(0,1), ..., var(0,dim-1)
//
// It is assume functions are over [0,1]^dim with gran steps per dimension
//
// scaleit 1 means L2 norm, scaleit2 means L2 norm * granularity
//
// dim = -1 means use dimension (for FuncVector)

double calcL2distsq(const Vector<gentype> &f, gentype &g, int dim, int scaleit = 1, int gran = DEFAULT_SAMPLES_SAMPLE);
double calcL2distsq(const gentype &f, gentype &g, int dim, int scaleit = 1, int gran = DEFAULT_SAMPLES_SAMPLE);


// This represents sum_i a_i a_i K(x,x_i)
//
// T can only be double or gentype, nothing else

template <> inline int aresame<gentype,gentype>(gentype *, gentype *);
template <> inline int aresame<gentype,gentype>(gentype *, gentype *) { return 1; }





// The class itself

class FuncVector : public Vector<gentype>
{
    friend class RKHSVector;
    friend class BernVector;
    friend void qswap(FuncVector &a, FuncVector &b);

public:

    // Constructors and Destructors

    FuncVector() : Vector<gentype>()  { fdim = 1; valfn = 0; }
    FuncVector(const FuncVector &src) : Vector<gentype>() { fdim = 1; valfn = 0; assign(src); } 
    virtual ~FuncVector() { }

    // Print and make duplicate

    virtual std::ostream &outstream(std::ostream &output) const;
    virtual std::istream &instream (std::istream &input );

    virtual std::istream &streamItIn(std::istream &input, int processxyzvw = 1);

    virtual Vector<gentype> *makeDup(void) const
    {
        FuncVector *dup;

        MEMNEW(dup,FuncVector(*this));

        return static_cast<Vector<gentype> *>(dup);
    }

    // Assignment

    FuncVector &operator=(const FuncVector &src) { return assign(src); }
    FuncVector &operator=(const gentype &src)    { return assign(src); }

    FuncVector &assign(const FuncVector &src);
    FuncVector &assign(const gentype &src);

    // Simple vector manipulations

    virtual Vector<gentype> &softzero(void) { unsample(); if ( extrapart.size() ) { int i; for ( i = 0 ; i < NE() ; ++i ) { MEMDEL(extrapart("&",i)); extrapart("&",i) = nullptr; } extrapart.resize(0); } valfn.zero();   return *this; }
    virtual Vector<gentype> &zero(void)     { unsample(); if ( extrapart.size() ) { int i; for ( i = 0 ; i < NE() ; ++i ) { MEMDEL(extrapart("&",i)); extrapart("&",i) = nullptr; } extrapart.resize(0); } valfn.zero();   return *this; }
    virtual Vector<gentype> &posate(void)   { unsample(); if ( extrapart.size() ) { int i; for ( i = 0 ; i < NE() ; ++i ) { (*extrapart("&",i)).posate(); } }                                           valfn.posate(); return *this; }
    virtual Vector<gentype> &negate(void)   { unsample(); if ( extrapart.size() ) { int i; for ( i = 0 ; i < NE() ; ++i ) { (*extrapart("&",i)).negate(); } }                                           valfn.negate(); return *this; }
    virtual Vector<gentype> &conj(void)     { unsample(); if ( extrapart.size() ) { int i; for ( i = 0 ; i < NE() ; ++i ) { (*extrapart("&",i)).conj();   } }                                           valfn.conj();   return *this; }
    virtual Vector<gentype> &rand(void)     { NiceThrow("Random functional vectors not implemented"); return *this; }

    // Access:
    //
    // - vector has the functional form f(x) = sum_{i=0}^{N-1} alpha_i K(x_i,x)
    // - to evaluate f(x) use operator()
    // - to access alpha_i use f.a(...)
    // - to access x_i use f.x(...)
    // - to access kernel use f.kern(...)

    virtual gentype &operator()(gentype &res, const              gentype  &i) const { SparseVector<gentype> ii; ii("&",0) = i; return (*this)(res,ii); }
    virtual gentype &operator()(gentype &res, const       Vector<gentype> &i) const { SparseVector<gentype> ii(i);             return (*this)(res,ii); }
    virtual gentype &operator()(gentype &res, const SparseVector<gentype> &i) const;

    const gentype &f(void) const        { NiceAssert( !ismixed() );                           return valfn; }
          gentype &f(const char *dummy) { NiceAssert( !ismixed() ); unsample(); (void) dummy; return valfn; }

    // Short-cut access:
    //
    // By calling sample, you can pre-generate a 1-d grid-evaluated version of the
    // vector for fast access.  This can then be accessed by the following:

    const Vector<gentype> &operator()(                        retVector<gentype> &tmp) const { if ( !samplesize() ) { sample(); } return precalcVec(tmp);          }
    const gentype         &operator()(int i                                          ) const { if ( !samplesize() ) { sample(); } return precalcVec(i);            }
    const Vector<gentype> &operator()(const Vector<int> &i,   retVector<gentype> &tmp) const { if ( !samplesize() ) { sample(); } return precalcVec(i,tmp);        }
    const Vector<gentype> &operator()(int ib, int is, int im, retVector<gentype> &tmp) const { if ( !samplesize() ) { sample(); } return precalcVec(ib,is,im,tmp); }

    // Information functions

    virtual int type(void) const { return 1; }

    virtual bool infsize(void) const { return true; }
    virtual bool ismixed(void) const { return NE(); }

    virtual int testsametype(std::string &typestring) { return typestring == "FN"; }

    // Function application - apply function fn to each element of vector.

    virtual Vector<gentype> &applyon(gentype (*fn)(gentype))                                      { NiceAssert( !ismixed() ); unsample(); valfn = (*fn)(valfn);   return *this; }
    virtual Vector<gentype> &applyon(gentype (*fn)(const gentype &))                              { NiceAssert( !ismixed() ); unsample(); valfn = (*fn)(valfn);   return *this; }
    virtual Vector<gentype> &applyon(gentype (*fn)(gentype, const void *), const void *a)         { NiceAssert( !ismixed() ); unsample(); valfn = (*fn)(valfn,a); return *this; }
    virtual Vector<gentype> &applyon(gentype (*fn)(const gentype &, const void *), const void *a) { NiceAssert( !ismixed() ); unsample(); valfn = (*fn)(valfn,a); return *this; }
    virtual Vector<gentype> &applyon(gentype &(*fn)(gentype &))                                   { NiceAssert( !ismixed() ); unsample();         (*fn)(valfn);   return *this; }
    virtual Vector<gentype> &applyon(gentype &(*fn)(gentype &, const void *), const void *a)      { NiceAssert( !ismixed() ); unsample();         (*fn)(valfn,a); return *this; }

    // Pre-allocation control.

    virtual void prealloc(int newallocsize)  { (void) newallocsize; }
    virtual void useStandardAllocation(void) {                      }
    virtual void useTightAllocation(void)    {                      }
    virtual void useSlackAllocation(void)    {                      }

    virtual bool array_norm (void) const { return true;  }
    virtual bool array_tight(void) const { return false; }
    virtual bool array_slack(void) const { return false; }





    // New stuff specific stuff

    virtual void sample(int Nsamp = DEFAULT_SAMPLES_SAMPLE) const;
    virtual void unsample(void) { precalcVec.resize(0); }
    virtual int samplesize(void) const { return precalcVec.size(); }

    int dim(void) const { return fdim; }
    void setdim(int nv) { fdim = nv; if ( NE() ) { int i; for ( i = 0 ; i < NE() ; ++i ) { (*extrapart("&",i)).setdim(nv); } } }

    int NE(void) const { return extrapart.size(); }





    // Inner-product functions for infsize vectors
    //
    // conj = 0: noConj
    //        1: normal
    //        2: revConj

    virtual gentype &inner1(gentype &res                                                                              ) const;
    virtual gentype &inner2(gentype &res, const Vector<gentype> &b, int conj = 1                                      ) const;
    virtual gentype &inner3(gentype &res, const Vector<gentype> &b, const Vector<gentype> &c                          ) const;
    virtual gentype &inner4(gentype &res, const Vector<gentype> &b, const Vector<gentype> &c, const Vector<gentype> &d) const;
    virtual gentype &innerp(gentype &res, const Vector<const Vector<gentype> *> &b                                    ) const;

    virtual double &inner1Real(double &res                                                                              ) const;
    virtual double &inner2Real(double &res, const Vector<gentype> &b, int conj = 1                                      ) const;
    virtual double &inner3Real(double &res, const Vector<gentype> &b, const Vector<gentype> &c                          ) const;
    virtual double &inner4Real(double &res, const Vector<gentype> &b, const Vector<gentype> &c, const Vector<gentype> &d) const;
    virtual double &innerpReal(double &res, const Vector<const Vector<gentype> *> &b                                    ) const;

    virtual double norm1(void)     const { double res; return inner1Real(res);       }
    virtual double norm2(void)     const { double res; return inner2Real(res,*this); }
    virtual double normp(double p) const { NiceAssert( ( (int) p ) == p ); NiceAssert( p > 0 ); double res; Vector<const Vector<gentype> *> b(((int) p)-1); b = this; return innerpReal(res,b); }

    virtual double absinf(void) const;
    virtual double abs0  (void) const;

    // Only the rudinemtary operators are defined: +=, -=, *=, /=, == (and by inference +,-,*,/)

    virtual Vector<gentype> &subit (const Vector<gentype> &b);
    virtual Vector<gentype> &addit (const Vector<gentype> &b);
    virtual Vector<gentype> &subit (const gentype         &b);
    virtual Vector<gentype> &addit (const gentype         &b);
    virtual Vector<gentype> &mulit (const Vector<gentype> &b);
    virtual Vector<gentype> &rmulit(const Vector<gentype> &b);
    virtual Vector<gentype> &divit (const Vector<gentype> &b);
    virtual Vector<gentype> &rdivit(const Vector<gentype> &b);
    virtual Vector<gentype> &mulit (const gentype         &b); // this*b
    virtual Vector<gentype> &rmulit(const gentype         &b); // b*this
    virtual Vector<gentype> &divit (const gentype         &b); // this/b
    virtual Vector<gentype> &rdivit(const gentype         &b); // b\this

    virtual bool iseq(const Vector<gentype> &b) { (void) b; NiceThrow("I don't know"); return false; }
    virtual bool iseq(const gentype         &b) { (void) b; NiceThrow("I really don't know"); return false; }





private:

    Vector<FuncVector *> extrapart; // nullptr except when you use FuncVector-RKHSVector, which results in a FuncVector with this non-null
    gentype valfn;
    int fdim;
    mutable Vector<gentype> precalcVec;
};

inline void qswap(FuncVector &a, FuncVector &b)
{
    // DON"T WANT THIS! qswap(static_cast<Vector<gentype> &>(a),static_cast<Vector<gentype> &>(b));

    qswap(a.valfn     ,b.valfn     );
    qswap(a.fdim      ,b.fdim      );
    qswap(a.precalcVec,b.precalcVec);
    qswap(a.extrapart ,b.extrapart );
}

inline void qswap(const FuncVector *&a, const FuncVector *&b)
{
    const FuncVector *c;

    c = a;
    a = b;
    b = c;
}

inline void qswap(FuncVector *&a, FuncVector *&b)
{
    FuncVector *c;

    c = a;
    a = b;
    b = c;
}





















Vector<gentype> &makeanRKHSVector(Vector<gentype> &res, const MercerKernel &kern, const SparseVector<gentype> &x, const gentype &a);
Vector<gentype> &makeanRKHSVector(Vector<gentype> &res, const MercerKernel &kern, const SparseVector<gentype> &x, const gentype &a, int m);
Vector<gentype> &makeanRKHSVector(Vector<gentype> &res, const MercerKernel &kern, const Vector<SparseVector<gentype> > &x, const Vector<vecInfo> &xinfo, const Vector<gentype> &a, int alphaasvec = 0, int m = 2);

class RKHSVector : public FuncVector
{
    friend void qswap(RKHSVector &a, RKHSVector &b);

public:

    // Constructors and Destructors

    RKHSVector() : FuncVector()  { mm = 2; alphaasvector = 0; revertToFunc = 0; }
    RKHSVector(const RKHSVector &src) : FuncVector(src) { mm = 2; alphaasvector = 0; revertToFunc = 0; assign(src); }
    virtual ~RKHSVector() { }

    // Special constructor for ML_Base

    RKHSVector &resetit(const MercerKernel &_spKern,
                        const Vector<SparseVector<gentype> > &_xx,
                        const Vector<vecInfo> &_xxinfo,
                        const Vector<gentype> &_alpha,
                        int _alphaasvector = 0,
                        int _mm = 2)
    {
        NiceAssert( _xx.size() == _xxinfo.size() );
        NiceAssert( _xx.size() == _alpha.size()  );

        spKern = _spKern;

        xx     = _xx;
        xxinfo = _xxinfo;
        alpha  = _alpha;

        xxinfook.resize(xx.size());
        xxinfook = 1;

        alphaasvector = _alphaasvector;
        mm            = _mm;
        revertToFunc  = 0;

        return *this;
    }

    // Print and make duplicate

    virtual std::ostream &outstream(std::ostream &output) const;
    virtual std::istream &instream (std::istream &input );

    virtual std::istream &streamItIn(std::istream &input, int processxyzvw = 1);

    virtual Vector<gentype> *makeDup(void) const
    {
        RKHSVector *dup;

        MEMNEW(dup,RKHSVector(*this));

        return static_cast<Vector<gentype> *>(dup);
    }

    // Assignment

    RKHSVector &operator=(const RKHSVector &src) { return assign(src); }
    RKHSVector &operator=(const gentype &src)    { return assign(src); }

    RKHSVector &assign(const RKHSVector &src)
    {
        FuncVector::assign(static_cast<const FuncVector &>(src));

        spKern        = src.spKern;
        alpha         = src.alpha;
        xx            = src.xx;
        xxinfo        = src.xxinfo;
        xxinfook      = src.xxinfook;
        mm            = src.mm;
        alphaasvector = src.alphaasvector;

        revertToFunc = src.revertToFunc;

        return *this;
    }

    RKHSVector &assign(const gentype &src)
    {
        (void) src;

        NiceThrow("No");

        return *this;
    }

    // Simple vector manipulations

    virtual Vector<gentype> &softzero(void) { unsample(); if ( revertToFunc ) { FuncVector::softzero(); } else { alpha.softzero(); } return *this; }
    virtual Vector<gentype> &zero(void)     { unsample(); if ( revertToFunc ) { FuncVector::zero();     } else { alpha.zero();     } return *this; }
    virtual Vector<gentype> &posate(void)   { unsample(); if ( revertToFunc ) { FuncVector::posate();   } else { alpha.posate();   } return *this; }
    virtual Vector<gentype> &negate(void)   { unsample(); if ( revertToFunc ) { FuncVector::negate();   } else { alpha.negate();   } return *this; }
    virtual Vector<gentype> &conj(void)     { unsample(); if ( revertToFunc ) { FuncVector::conj();     } else { alpha.conj();     } return *this; }
    virtual Vector<gentype> &rand(void)     { unsample(); if ( revertToFunc ) { FuncVector::rand();     } else { alpha.rand();     } return *this; }

    // Access:
    //
    // - vector has the functional form f(x) = sum_{i=0}^{N-1} alpha_i K(x_i,x)
    // - to evaluate f(x) use operator()
    // - to access alpha_i use f.a(...)
    // - to access x_i use f.x(...)
    // - to access kernel use f.kern(...)
    // - to evaluate f(x1,x2,...) use [ x1 ~ x2 ~ ... ] format sparsevectors
    //
    // NB: don't change N by resizing these references!

    virtual gentype &operator()(gentype &res, const              gentype  &i) const { if ( revertToFunc ) { return FuncVector::operator()(res,i); } SparseVector<gentype> ii; ii("&",0) = i; return (*this)(res,ii); }
    virtual gentype &operator()(gentype &res, const       Vector<gentype> &i) const { if ( revertToFunc ) { return FuncVector::operator()(res,i); } SparseVector<gentype> ii(i);             return (*this)(res,ii); }
    virtual gentype &operator()(gentype &res, const SparseVector<gentype> &i) const;

    const gentype &f(void) const        { if ( revertToFunc ) { return FuncVector::f();      } NiceThrow("Can't use f() on RKHSVector"); const static thread_local gentype rdummy; return rdummy; }
          gentype &f(const char *dummy) { if ( revertToFunc ) { return FuncVector::f(dummy); } NiceThrow("Can't use f() on RKHSVector");       static thread_local gentype rdummy; return rdummy; }

    Vector<gentype> &a(const char *dummy,                         retVector<gentype> &tmp) { unsample(); NiceAssert( !revertToFunc ); return alpha(dummy,tmp);          }
    gentype         &a(const char *dummy, int i                                          ) { unsample(); NiceAssert( !revertToFunc ); return alpha(dummy,i);            }
    Vector<gentype> &a(const char *dummy, const Vector<int> &i,   retVector<gentype> &tmp) { unsample(); NiceAssert( !revertToFunc ); return alpha(dummy,i,tmp);        }
    Vector<gentype> &a(const char *dummy, int ib, int is, int im, retVector<gentype> &tmp) { unsample(); NiceAssert( !revertToFunc ); return alpha(dummy,ib,is,im,tmp); }

    const Vector<gentype> &a(                        retVector<gentype> &tmp) const { NiceAssert( !revertToFunc ); return alpha(tmp);          }
    const gentype         &a(int i                                          ) const { NiceAssert( !revertToFunc ); return alpha(i);            }
    const Vector<gentype> &a(const Vector<int> &i,   retVector<gentype> &tmp) const { NiceAssert( !revertToFunc ); return alpha(i,tmp);        }
    const Vector<gentype> &a(int ib, int is, int im, retVector<gentype> &tmp) const { NiceAssert( !revertToFunc ); return alpha(ib,is,im,tmp); }

    Vector<SparseVector<gentype> > &x(const char *dummy,                         retVector<SparseVector<gentype> > &tmp) { unsample(); NiceAssert( !revertToFunc ); retVector<int> tmpvb; xxinfook(dummy,tmpvb)          = 0; return xx(dummy,tmp);          }
    SparseVector<gentype>          &x(const char *dummy, int i                                                         ) { unsample(); NiceAssert( !revertToFunc );                       xxinfook(dummy,i)              = 0; return xx(dummy,i);            }
    Vector<SparseVector<gentype> > &x(const char *dummy, const Vector<int> &i,   retVector<SparseVector<gentype> > &tmp) { unsample(); NiceAssert( !revertToFunc ); retVector<int> tmpvb; xxinfook(dummy,i,tmpvb)        = 0; return xx(dummy,i,tmp);        }
    Vector<SparseVector<gentype> > &x(const char *dummy, int ib, int is, int im, retVector<SparseVector<gentype> > &tmp) { unsample(); NiceAssert( !revertToFunc ); retVector<int> tmpvb; xxinfook(dummy,ib,is,im,tmpvb) = 0; return xx(dummy,ib,is,im,tmp); }

    const Vector<SparseVector<gentype> > &x(                        retVector<SparseVector<gentype> > &tmp) const { NiceAssert( !revertToFunc ); return getxx()(tmp);          }
    const SparseVector<gentype>          &x(int i                                                         ) const { NiceAssert( !revertToFunc ); return getxx(i);              }
    const Vector<SparseVector<gentype> > &x(const Vector<int> &i,   retVector<SparseVector<gentype> > &tmp) const { NiceAssert( !revertToFunc ); return getxx()(i,tmp);        }
    const Vector<SparseVector<gentype> > &x(int ib, int is, int im, retVector<SparseVector<gentype> > &tmp) const { NiceAssert( !revertToFunc ); return getxx()(ib,is,im,tmp); }

    const MercerKernel &kern(void) const        {             NiceAssert( !revertToFunc );               return getspKern(); }
          MercerKernel &kern(const char *dummy) { unsample(); NiceAssert( !revertToFunc ); (void) dummy; return spKern;      }

    // Information functions

    virtual int type(void) const { return 2; }

    virtual bool infsize(void) const { return true;                        }
    virtual bool ismixed(void) const { return revertToFunc ? true : false; }

    virtual int testsametype(std::string &typestring) { return typestring == "RKHS"; }

    // Function application - apply function fn to each element of vector.

    virtual Vector<gentype> &applyon(gentype (*fn)(gentype))                                      { if ( revertToFunc ) { return FuncVector::applyon(fn);   } NiceThrow("Can't apply function to RKHSVector"); return *this; }
    virtual Vector<gentype> &applyon(gentype (*fn)(const gentype &))                              { if ( revertToFunc ) { return FuncVector::applyon(fn);   } NiceThrow("Can't apply function to RKHSVector"); return *this; }
    virtual Vector<gentype> &applyon(gentype (*fn)(gentype, const void *), const void *a)         { if ( revertToFunc ) { return FuncVector::applyon(fn,a); } NiceThrow("Can't apply function to RKHSVector"); return *this; }
    virtual Vector<gentype> &applyon(gentype (*fn)(const gentype &, const void *), const void *a) { if ( revertToFunc ) { return FuncVector::applyon(fn,a); } NiceThrow("Can't apply function to RKHSVector"); return *this; }
    virtual Vector<gentype> &applyon(gentype &(*fn)(gentype &))                                   { if ( revertToFunc ) { return FuncVector::applyon(fn);   } NiceThrow("Can't apply function to RKHSVector"); return *this; }
    virtual Vector<gentype> &applyon(gentype &(*fn)(gentype &, const void *), const void *a)      { if ( revertToFunc ) { return FuncVector::applyon(fn,a); } NiceThrow("Can't apply function to RKHSVector"); return *this; }





    // RKHS specific stuff
    //
    // N is the basis size
    // resizeN changes this.  If decreasing size it will sparsify by removing the smallest alpha's (by 2-norm) first.

    virtual int N(void) const { return getN(); }
    virtual void resizeN(int N) { int oldN = N; xx.resize(N); xxinfo.resize(N); xxinfook.resize(N); alpha.resize(N); if ( N > oldN ) { retVector<int> tmpva; retVector<gentype> tmpvb; xxinfook("&",oldN,1,N-1,tmpva) = 0; alpha("&",oldN,1,N-1,tmpvb) = 0.0_gent; } }

    virtual int m(void) const { return mm; }
    virtual void setm(int nm) { NiceAssert( nm >= 1 ); mm = nm; }

    virtual int treatalphaasvector(void) const { return gettreatalphaasvector(); }
    virtual void settreatalphaasvector(int nv) { alphaasvector = nv; }

    const Vector<vecInfo> &xinfo(                        retVector<vecInfo> &tmp) const { NiceAssert( !revertToFunc ); makeinfo();          return getxxinfo()(tmp);          }
    const vecInfo         &xinfo(int i                                          ) const { NiceAssert( !revertToFunc ); makeinfo(i);         return getxxinfo(i);              }
    const Vector<vecInfo> &xinfo(const Vector<int> &i,   retVector<vecInfo> &tmp) const { NiceAssert( !revertToFunc ); makeinfo(i);         return getxxinfo()(i,tmp);        }
    const Vector<vecInfo> &xinfo(int ib, int is, int im, retVector<vecInfo> &tmp) const { NiceAssert( !revertToFunc ); makeinfo(ib,is,im);  return getxxinfo()(ib,is,im,tmp); }






    // Inner-product functions for RKHS
    //
    // conj = 0: noConj
    //        1: normal
    //        2: revConj

    virtual gentype &inner1(gentype &res                                                                              ) const;
    virtual gentype &inner2(gentype &res, const Vector<gentype> &b, int conj = 1                                      ) const;
    virtual gentype &inner3(gentype &res, const Vector<gentype> &b, const Vector<gentype> &c                          ) const;
    virtual gentype &inner4(gentype &res, const Vector<gentype> &b, const Vector<gentype> &c, const Vector<gentype> &d) const;
    virtual gentype &innerp(gentype &res, const Vector<const Vector<gentype> *> &b                                    ) const;

    virtual double &inner1Real(double &res                                                                              ) const;
    virtual double &inner2Real(double &res, const Vector<gentype> &b, int conj = 1                                      ) const;
    virtual double &inner3Real(double &res, const Vector<gentype> &b, const Vector<gentype> &c                          ) const;
    virtual double &inner4Real(double &res, const Vector<gentype> &b, const Vector<gentype> &c, const Vector<gentype> &d) const;
    virtual double &innerpReal(double &res, const Vector<const Vector<gentype> *> &b                                    ) const;

    virtual double norm1(void)     const { double res; return inner1Real(res); }
    virtual double norm2(void)     const { double res; return inner2Real(res,*this); }
    virtual double normp(double p) const { NiceAssert( ( (int) p ) == p ); NiceAssert( p > 0 ); double res; Vector<const Vector<gentype> *> b(((int) p)-1); b = this; return innerpReal(res,b); }

    virtual double absinf(void) const
    {
        NiceThrow("I don't know how to do that");

        return 0.0;
    }

    virtual double abs0(void) const
    {
        NiceThrow("Mmm mmm mmm mmm, mmm mmm mmm mmm, mmmmmmmmmmmmmmm");

        return 0.0;
    }

    //subit and addit are not efficient as they just append and fix sign
    virtual Vector<gentype> &subit (const Vector<gentype> &b);
    virtual Vector<gentype> &addit (const Vector<gentype> &b);
    virtual Vector<gentype> &subit (const gentype         &b);
    virtual Vector<gentype> &addit (const gentype         &b);
    virtual Vector<gentype> &mulit (const Vector<gentype> &b) { (void) b; NiceThrow("I'm sorry Dave, I don't know how to do that");   return *this; }
    virtual Vector<gentype> &rmulit(const Vector<gentype> &b) { (void) b; NiceThrow("I'm sorry Darren, I don't know how to do that"); return *this; }
    virtual Vector<gentype> &divit (const Vector<gentype> &b) { (void) b; NiceThrow("I'm sorry Garian, I don't know how to do that"); return *this; }
    virtual Vector<gentype> &rdivit(const Vector<gentype> &b) { (void) b; NiceThrow("I'm sorry Fred, I don't know how to do that");   return *this; }
    virtual Vector<gentype> &mulit (const gentype         &b);
    virtual Vector<gentype> &rmulit(const gentype         &b);
    virtual Vector<gentype> &divit (const gentype         &b);
    virtual Vector<gentype> &rdivit(const gentype         &b);

    virtual bool iseq(const Vector<gentype> &b) { (void) b; NiceThrow("I still don't know"); return false; }
    virtual bool iseq(const gentype         &b) { (void) b; NiceThrow("No seriously, I just don't know"); return false; }







private:

    MercerKernel spKern;
    Vector<SparseVector<gentype> > xx;
    mutable Vector<vecInfo> xxinfo; // can be constructed on the fly
    mutable Vector<int> xxinfook;   // can be constructed on the fly
    Vector<gentype> alpha;

    int alphaasvector;
    int mm; // order of m-kernel RKHS vector
    int revertToFunc; // if 1 then call back to FuncVector (required for -=,+=)

    // Private getters to enable trivial ml_base overload

    const MercerKernel &getspKern(void) const { return spKern; }

    const Vector<SparseVector<gentype> > &getxx(void)  const { return xx;    }
    const SparseVector<gentype> &getxx(int i) const { return xx(i); }

    const Vector<vecInfo> &getxxinfo(void)  const { return xxinfo;    }
    const vecInfo &getxxinfo(int i) const { return xxinfo(i); }

    const Vector<gentype> &getalpha(void) const { return alpha; }
    const gentype &getalpha(int i) const { return alpha(i); }

    int getN(void) const { return alpha.size(); }
    int gettreatalphaasvector(void) const { return alphaasvector; }

    // other stuff

    const gentype &al(int i, int j) const
    {
        return alphaasvector ? getalpha(i)(j) : getalpha(i);
    }

    void makeinfo(int i) const
    {
        if ( !xxinfook(i) )
        {
            xxinfook("&",i) = 1;

            kern().getvecInfo(xxinfo("&",i),xx(i));
        }
    }

    void makeinfo(const Vector<int> &i) const
    {
        int ii;

        for ( ii = 0 ; ii < i.size() ; ++ii )
        {
            makeinfo(i(ii));
        }
    }

    void makeinfo(int ib, int is, int im) const
    {
        int ii;

        if ( ( is > 0 ) && ( im >= ib ) )
        {
            for ( ii = ib ; ii <= im ; ii += is )
            {
                makeinfo(ii);
            }
        }

        else if ( ( is < 0 ) && ( im <= ib ) )
        {
            for ( ii = ib ; ii >= im ; ii += is )
            {
                makeinfo(ii);
            }
        }
    }

    void makeinfo(void) const
    {
        int ii;

        for ( ii = 0 ; ii < N() ; ++ii )
        {
            makeinfo(ii);
        }
    }


    // Base versions with m factors already taken into account

    virtual gentype &baseinner1(gentype &res                                                                              , int aind                              ) const;
    virtual gentype &baseinner2(gentype &res, const Vector<gentype> &b, int conj                                          , int aind, int bind                    ) const;
    virtual gentype &baseinner3(gentype &res, const Vector<gentype> &b, const Vector<gentype> &c                          , int aind, int bind, int cind          ) const;
    virtual gentype &baseinner4(gentype &res, const Vector<gentype> &b, const Vector<gentype> &c, const Vector<gentype> &d, int aind, int bind, int cind, int dind) const;
    virtual gentype &baseinnerp(gentype &res, const Vector<const Vector<gentype> *> &b                                    , int aind, const Vector<int> &bind     ) const;

    virtual double &baseinner1Real(double &res                                                                              , int aind                              ) const;
    virtual double &baseinner2Real(double &res, const Vector<gentype> &b, int conj                                          , int aind, int bind                    ) const;
    virtual double &baseinner3Real(double &res, const Vector<gentype> &b, const Vector<gentype> &c                          , int aind, int bind, int cind          ) const;
    virtual double &baseinner4Real(double &res, const Vector<gentype> &b, const Vector<gentype> &c, const Vector<gentype> &d, int aind, int bind, int cind, int dind) const;
    virtual double &baseinnerpReal(double &res, const Vector<const Vector<gentype> *> &b                                    , int aind, const Vector<int> &bind     ) const;
};

inline void qswap(RKHSVector &a, RKHSVector &b)
{
    qswap(static_cast<FuncVector &>(a),static_cast<FuncVector &>(b));

    qswap(a.spKern       ,b.spKern       );
    qswap(a.xx           ,b.xx           );
    qswap(a.alpha        ,b.alpha        );
    qswap(a.mm           ,b.mm           );
    qswap(a.alphaasvector,b.alphaasvector);

    qswap(a.revertToFunc,b.revertToFunc);
}

inline void qswap(const RKHSVector *&a, const RKHSVector *&b)
{
    const RKHSVector *c;

    c = a;
    a = b;
    b = c;
}

inline void qswap(RKHSVector *&a, RKHSVector *&b)
{
    RKHSVector *c;

    c = a;
    a = b;
    b = c;
}

















class BernVector : public FuncVector
{
    friend void qswap(BernVector &a, BernVector &b);

public:

    // Constructors and Destructors

    BernVector() : FuncVector()  { revertToFunc = 0; } 
    BernVector(const BernVector &src) : FuncVector(src) { revertToFunc = 0; assign(src); } 
    virtual ~BernVector() { }

    // Print and make duplicate

    virtual std::ostream &outstream(std::ostream &output) const;
    virtual std::istream &instream (std::istream &input );

    virtual std::istream &streamItIn(std::istream &input, int processxyzvw = 1);

    virtual Vector<gentype> *makeDup(void) const
    {
        BernVector *dup;

        MEMNEW(dup,BernVector(*this));

        return static_cast<Vector<gentype> *>(dup);
    }

    // Assignment

    BernVector &operator=(const BernVector &src) { return assign(src); }
    BernVector &operator=(const gentype &src)    { return assign(src); }

    BernVector &assign(const BernVector &src) 
    { 
        FuncVector::assign(static_cast<const FuncVector &>(src));

        ww = src.ww;

        revertToFunc = src.revertToFunc;

        return *this; 
    }

    BernVector &assign(const gentype &src) 
    { 
        (void) src;

        NiceThrow("Really, no");

        return *this; 
    }

    // Simple vector manipulations

    virtual Vector<gentype> &softzero(void) { unsample(); if ( revertToFunc ) { FuncVector::softzero(); } else { ww.softzero(); } return *this; }
    virtual Vector<gentype> &zero(void)     { unsample(); if ( revertToFunc ) { FuncVector::zero();     } else { ww.zero();     } return *this; }
    virtual Vector<gentype> &posate(void)   { unsample(); if ( revertToFunc ) { FuncVector::posate();   } else { ww.posate();   } return *this; }
    virtual Vector<gentype> &negate(void)   { unsample(); if ( revertToFunc ) { FuncVector::negate();   } else { ww.negate();   } return *this; }
    virtual Vector<gentype> &conj(void)     { unsample(); if ( revertToFunc ) { FuncVector::conj();     } else { ww.conj();     } return *this; }
    virtual Vector<gentype> &rand(void)     { unsample(); if ( revertToFunc ) { FuncVector::rand();     } else { ww.rand();     } return *this; }

    // Access:
    //
    // - vector has the functional form f(x) = sum_{i=0}^{N-1} alpha_i K(x_i,x)
    // - to evaluate f(x) use operator()
    // - to access alpha_i use f.a(...)
    // - to access x_i use f.x(...)
    // - to access kernel use f.kern(...)

    virtual gentype &operator()(gentype &res, const              gentype  &i) const { if ( revertToFunc ) { return FuncVector::operator()(res,i); } SparseVector<gentype> ii; ii("&",0) = i; return (*this)(res,ii); }
    virtual gentype &operator()(gentype &res, const       Vector<gentype> &i) const { if ( revertToFunc ) { return FuncVector::operator()(res,i); } SparseVector<gentype> ii(i);             return (*this)(res,ii); }
    virtual gentype &operator()(gentype &res, const SparseVector<gentype> &i) const;

    const gentype &f(void) const        { if ( revertToFunc ) { return FuncVector::f();      } NiceThrow("Can't use f() on BernVector"); const static thread_local gentype rdummy; return rdummy; }
          gentype &f(const char *dummy) { if ( revertToFunc ) { return FuncVector::f(dummy); } NiceThrow("Can't use f() on BernVector");       static thread_local gentype rdummy; return rdummy; }

    Vector<gentype> &w(const char *dummy,                         retVector<gentype> &tmp) { unsample(); NiceAssert( !revertToFunc ); return ww(dummy,tmp);          }
    gentype         &w(const char *dummy, int i                                          ) { unsample(); NiceAssert( !revertToFunc ); return ww(dummy,i);            }
    Vector<gentype> &w(const char *dummy, const Vector<int> &i,   retVector<gentype> &tmp) { unsample(); NiceAssert( !revertToFunc ); return ww(dummy,i,tmp);        }
    Vector<gentype> &w(const char *dummy, int ib, int is, int im, retVector<gentype> &tmp) { unsample(); NiceAssert( !revertToFunc ); return ww(dummy,ib,is,im,tmp); }

    const Vector<gentype> &w(                        retVector<gentype> &tmp) const { NiceAssert( !revertToFunc ); return ww(tmp);          }
    const gentype         &w(int i                                          ) const { NiceAssert( !revertToFunc ); return ww(i);            }
    const Vector<gentype> &w(const Vector<int> &i,   retVector<gentype> &tmp) const { NiceAssert( !revertToFunc ); return ww(i,tmp);        }
    const Vector<gentype> &w(int ib, int is, int im, retVector<gentype> &tmp) const { NiceAssert( !revertToFunc ); return ww(ib,is,im,tmp); }

    // Information functions

    virtual int type(void) const { return 3; }

    virtual bool infsize(void) const { return true;                        }
    virtual bool ismixed(void) const { return revertToFunc ? true : false; }

    virtual int testsametype(std::string &typestring) { return typestring == "Bern"; }

    // Function application - apply function fn to each element of vector.

    virtual Vector<gentype> &applyon(gentype (*fn)(gentype))                                      { if ( revertToFunc ) { return FuncVector::applyon(fn);   } NiceThrow("Can't apply function to BernVector"); return *this; }
    virtual Vector<gentype> &applyon(gentype (*fn)(const gentype &))                              { if ( revertToFunc ) { return FuncVector::applyon(fn);   } NiceThrow("Can't apply function to BernVector"); return *this; }
    virtual Vector<gentype> &applyon(gentype (*fn)(gentype, const void *), const void *a)         { if ( revertToFunc ) { return FuncVector::applyon(fn,a); } NiceThrow("Can't apply function to BernVector"); return *this; }
    virtual Vector<gentype> &applyon(gentype (*fn)(const gentype &, const void *), const void *a) { if ( revertToFunc ) { return FuncVector::applyon(fn,a); } NiceThrow("Can't apply function to BernVector"); return *this; }
    virtual Vector<gentype> &applyon(gentype &(*fn)(gentype &))                                   { if ( revertToFunc ) { return FuncVector::applyon(fn);   } NiceThrow("Can't apply function to BernVector"); return *this; }
    virtual Vector<gentype> &applyon(gentype &(*fn)(gentype &, const void *), const void *a)      { if ( revertToFunc ) { return FuncVector::applyon(fn,a); } NiceThrow("Can't apply function to BernVector"); return *this; }






    // Bernstein specific stuff
    //
    // Nw is the w size

    virtual int Nw(void) const { return ww.size()-1; }





    //subit and addit are not efficient as they just append and fix sign
    virtual Vector<gentype> &subit (const Vector<gentype> &b);
    virtual Vector<gentype> &addit (const Vector<gentype> &b);
    virtual Vector<gentype> &subit (const gentype         &b);
    virtual Vector<gentype> &addit (const gentype         &b);
    virtual Vector<gentype> &mulit (const Vector<gentype> &b) { (void) b; NiceThrow("I'm sorry Liam, I don't know how to do that");  return *this; }
    virtual Vector<gentype> &rmulit(const Vector<gentype> &b) { (void) b; NiceThrow("I'm sorry Ivy, I don't know how to do that");   return *this; }
    virtual Vector<gentype> &divit (const Vector<gentype> &b) { (void) b; NiceThrow("I'm sorry Cindy, I don't know how to do that"); return *this; }
    virtual Vector<gentype> &rdivit(const Vector<gentype> &b) { (void) b; NiceThrow("I'm sorry you, I don't know how to do that");   return *this; }
    virtual Vector<gentype> &mulit (const gentype         &b);
    virtual Vector<gentype> &rmulit(const gentype         &b);
    virtual Vector<gentype> &divit (const gentype         &b);
    virtual Vector<gentype> &rdivit(const gentype         &b);

    virtual bool iseq(const Vector<gentype> &b) { (void) b; NiceThrow("Umm."); return false; }
    virtual bool iseq(const gentype         &b) { (void) b; NiceThrow("Nah."); return false; }







private:

    Vector<gentype> ww;

    int revertToFunc; // if 1 then call back to FuncVector (required for -=,+=)
};

inline void qswap(BernVector &a, BernVector &b)
{
    qswap(static_cast<FuncVector &>(a),static_cast<FuncVector &>(b));

    qswap(a.ww,b.ww);

    qswap(a.revertToFunc,b.revertToFunc);
}

inline void qswap(const BernVector *&a, const BernVector *&b)
{
    const BernVector *c;

    c = a;
    a = b;
    b = c;
}

inline void qswap(BernVector *&a, BernVector *&b)
{
    BernVector *c;

    c = a;
    a = b;
    b = c;
}

















inline const FuncVector *&setident      (const FuncVector *&a);
inline const FuncVector *&setzero       (const FuncVector *&a);
inline const FuncVector *&setzeropassive(const FuncVector *&a);
inline const FuncVector *&setposate     (const FuncVector *&a);
inline const FuncVector *&setnegate     (const FuncVector *&a);
inline const FuncVector *&setconj       (const FuncVector *&a);
inline const FuncVector *&setrand       (const FuncVector *&a);

inline const FuncVector *&setident      (const FuncVector *&a) { return a = nullptr; }
inline const FuncVector *&setzero       (const FuncVector *&a) { return a = nullptr; }
inline const FuncVector *&setzeropassive(const FuncVector *&a) { return a = nullptr; }
inline const FuncVector *&setposate     (const FuncVector *&a) { return a = nullptr; }
inline const FuncVector *&setnegate     (const FuncVector *&a) { return a = nullptr; }
inline const FuncVector *&setconj       (const FuncVector *&a) { return a = nullptr; }
inline const FuncVector *&setrand       (const FuncVector *&a) { return a = nullptr; }

inline const RKHSVector *&setident      (const RKHSVector *&a);
inline const RKHSVector *&setzero       (const RKHSVector *&a);
inline const RKHSVector *&setzeropassive(const RKHSVector *&a);
inline const RKHSVector *&setposate     (const RKHSVector *&a);
inline const RKHSVector *&setnegate     (const RKHSVector *&a);
inline const RKHSVector *&setconj       (const RKHSVector *&a);
inline const RKHSVector *&setrand       (const RKHSVector *&a);

inline const RKHSVector *&setident      (const RKHSVector *&a) { return a = nullptr; }
inline const RKHSVector *&setzero       (const RKHSVector *&a) { return a = nullptr; }
inline const RKHSVector *&setzeropassive(const RKHSVector *&a) { return a = nullptr; }
inline const RKHSVector *&setposate     (const RKHSVector *&a) { return a = nullptr; }
inline const RKHSVector *&setnegate     (const RKHSVector *&a) { return a = nullptr; }
inline const RKHSVector *&setconj       (const RKHSVector *&a) { return a = nullptr; }
inline const RKHSVector *&setrand       (const RKHSVector *&a) { return a = nullptr; }

inline const BernVector *&setident      (const BernVector *&a);
inline const BernVector *&setzero       (const BernVector *&a);
inline const BernVector *&setzeropassive(const BernVector *&a);
inline const BernVector *&setposate     (const BernVector *&a);
inline const BernVector *&setnegate     (const BernVector *&a);
inline const BernVector *&setconj       (const BernVector *&a);
inline const BernVector *&setrand       (const BernVector *&a);

inline const BernVector *&setident      (const BernVector *&a) { return a = nullptr; }
inline const BernVector *&setzero       (const BernVector *&a) { return a = nullptr; }
inline const BernVector *&setzeropassive(const BernVector *&a) { return a = nullptr; }
inline const BernVector *&setposate     (const BernVector *&a) { return a = nullptr; }
inline const BernVector *&setnegate     (const BernVector *&a) { return a = nullptr; }
inline const BernVector *&setconj       (const BernVector *&a) { return a = nullptr; }
inline const BernVector *&setrand       (const BernVector *&a) { return a = nullptr; }



inline FuncVector *&setident      (FuncVector *&a);
inline FuncVector *&setzero       (FuncVector *&a);
inline FuncVector *&setzeropassive(FuncVector *&a);
inline FuncVector *&setposate     (FuncVector *&a);
inline FuncVector *&setnegate     (FuncVector *&a);
inline FuncVector *&setconj       (FuncVector *&a);
inline FuncVector *&setrand       (FuncVector *&a);

inline FuncVector *&setident      (FuncVector *&a) { return a = nullptr; }
inline FuncVector *&setzero       (FuncVector *&a) { return a = nullptr; }
inline FuncVector *&setzeropassive(FuncVector *&a) { return a = nullptr; }
inline FuncVector *&setposate     (FuncVector *&a) { return a = nullptr; }
inline FuncVector *&setnegate     (FuncVector *&a) { return a = nullptr; }
inline FuncVector *&setconj       (FuncVector *&a) { return a = nullptr; }
inline FuncVector *&setrand       (FuncVector *&a) { return a = nullptr; }

inline RKHSVector *&setident      (RKHSVector *&a);
inline RKHSVector *&setzero       (RKHSVector *&a);
inline RKHSVector *&setzeropassive(RKHSVector *&a);
inline RKHSVector *&setposate     (RKHSVector *&a);
inline RKHSVector *&setnegate     (RKHSVector *&a);
inline RKHSVector *&setconj       (RKHSVector *&a);
inline RKHSVector *&setrand       (RKHSVector *&a);

inline RKHSVector *&setident      (RKHSVector *&a) { return a = nullptr; }
inline RKHSVector *&setzero       (RKHSVector *&a) { return a = nullptr; }
inline RKHSVector *&setzeropassive(RKHSVector *&a) { return a = nullptr; }
inline RKHSVector *&setposate     (RKHSVector *&a) { return a = nullptr; }
inline RKHSVector *&setnegate     (RKHSVector *&a) { return a = nullptr; }
inline RKHSVector *&setconj       (RKHSVector *&a) { return a = nullptr; }
inline RKHSVector *&setrand       (RKHSVector *&a) { return a = nullptr; }

inline BernVector *&setident      (BernVector *&a);
inline BernVector *&setzero       (BernVector *&a);
inline BernVector *&setzeropassive(BernVector *&a);
inline BernVector *&setposate     (BernVector *&a);
inline BernVector *&setnegate     (BernVector *&a);
inline BernVector *&setconj       (BernVector *&a);
inline BernVector *&setrand       (BernVector *&a);

inline BernVector *&setident      (BernVector *&a) { return a = nullptr; }
inline BernVector *&setzero       (BernVector *&a) { return a = nullptr; }
inline BernVector *&setzeropassive(BernVector *&a) { return a = nullptr; }
inline BernVector *&setposate     (BernVector *&a) { return a = nullptr; }
inline BernVector *&setnegate     (BernVector *&a) { return a = nullptr; }
inline BernVector *&setconj       (BernVector *&a) { return a = nullptr; }
inline BernVector *&setrand       (BernVector *&a) { return a = nullptr; }



#endif


