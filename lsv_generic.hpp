
//
// LS-SVM base class
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _lsv_generic_h
#define _lsv_generic_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "ml_base.hpp"
#include "svm_scalar.hpp"


class LSV_Generic;
class LSV_Vector;

// Swap and zeroing (restarting) functions

inline void qswap(LSV_Generic &a, LSV_Generic &b);
inline LSV_Generic &setzero(LSV_Generic &a);

class LSV_Generic : public SVM_Scalar
{
    friend class LSV_Vector;

public:

    // Constructors, destructors, assignment etc..

    LSV_Generic();
    LSV_Generic(const LSV_Generic &src);
    LSV_Generic(const LSV_Generic &src, const ML_Base *srcx);
    LSV_Generic &operator=(const LSV_Generic &src) { assign(src); return *this; }
    virtual ~LSV_Generic() { return; }

    virtual int prealloc(int expectedN) override;
    virtual int preallocsize(void) const override;
    virtual void setmemsize(int memsize) override { SVM_Scalar::setmemsize(memsize); return; }

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0) override;
    virtual void semicopy(const ML_Base &src) override;
    virtual void qswapinternal(ML_Base &b) override;

    virtual int getparam (int ind, gentype         &val, const gentype         &xa, int ia, const gentype         &xb, int ib, charptr &desc) const override;
    virtual int egetparam(int ind, Vector<gentype> &val, const Vector<gentype> &xa, int ia, const Vector<gentype> &xb, int ib               ) const override;

    virtual std::ostream &printstream(std::ostream &output, int dep) const override;
    virtual std::istream &inputstream(std::istream &input ) override;

    virtual       ML_Base &getML     (void)       override { return static_cast<      ML_Base &>(getLSV     ()); }
    virtual const ML_Base &getMLconst(void) const override { return static_cast<const ML_Base &>(getLSVconst()); }

    // Information functions (training data):

    virtual double calcDist(const gentype &ha, const gentype &hb, int ia = -1, int db = 2) const override;

    virtual int isUnderlyingScalar(void) const override { return 0; }
    virtual int isUnderlyingVector(void) const override { return 0; }
    virtual int isUnderlyingAnions(void) const override { return 0; }

    virtual int getInternalClass(const gentype &y) const override { return ML_Base::getInternalClass(y); }

    virtual int isVarDefined(void) const override { return 0; }

    //virtual double eps(void)       const override { return 0.0;         }
    //virtual double epsclass(int d) const override { (void) d; return 1; }

    virtual const Vector<gentype> &y(void) const override { return alltraintarg; }

    virtual int isClassifier(void) const override { return 0; }
    virtual int isRegression(void) const override { return 1; }
    virtual int isPlanarType(void) const override { return 0; }

    // Kernel Modification

    // Training set modification:

    virtual int addTrainingVector (int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2) override;
    virtual int qaddTrainingVector(int i, const gentype &y,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2) override;

    virtual int addTrainingVector(int i,            double *xxa, int dima, double Cweigh = 1, double epsweigh = 1) override { return ML_Base::addTrainingVector(i,   xxa,dima,Cweigh,epsweigh); }
    virtual int addTrainingVector(int i, int zz,    double *xxa, int dima, double Cweigh = 1, double epsweigh = 1) override { return ML_Base::addTrainingVector(i,zz,xxa,dima,Cweigh,epsweigh); }
    virtual int addTrainingVector(int i, double zz, double *xxa, int dima, double Cweigh = 1, double epsweigh = 1) override { return ML_Base::addTrainingVector(i,zz,xxa,dima,Cweigh,epsweigh); }

    virtual int addTrainingVector (int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override;
    virtual int qaddTrainingVector(int i, const Vector<gentype> &y,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override;

    virtual int removeTrainingVector(int i                                      ) override { SparseVector<gentype> x; gentype y; return removeTrainingVector(i,y,x); }
    virtual int removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x) override;
    virtual int removeTrainingVector(int i, int num                             ) override;

    virtual int sety(int                i, const gentype         &y) override;
    virtual int sety(const Vector<int> &i, const Vector<gentype> &y) override;
    virtual int sety(                      const Vector<gentype> &y) override;

    virtual int sety(int                i, double                y) override { (void) i; (void) y; NiceThrow("sety fallback 1"); return 1; }
    virtual int sety(const Vector<int> &i, const Vector<double> &y) override { (void) i; (void) y; NiceThrow("sety fallback 2"); return 1; }
    virtual int sety(                      const Vector<double> &y) override {           (void) y; NiceThrow("sety fallback 3"); return 1; }

    virtual int sety(int                i, const Vector<double>          &y) override { (void) i; (void) y; NiceThrow("Whatever"); return 1; }
    virtual int sety(const Vector<int> &i, const Vector<Vector<double> > &y) override { (void) i; (void) y; NiceThrow("Whatever"); return 1; }
    virtual int sety(                      const Vector<Vector<double> > &y) override {           (void) y; NiceThrow("Whatever"); return 1; }

    virtual int sety(int                i, const d_anion         &y) override { (void) i; (void) y; NiceThrow("Whatever"); return 1; }
    virtual int sety(const Vector<int> &i, const Vector<d_anion> &y) override { (void) i; (void) y; NiceThrow("Whatever"); return 1; }
    virtual int sety(                      const Vector<d_anion> &y) override {           (void) y; NiceThrow("Whatever"); return 1; }

    virtual int setd(int                i, int                nd) override;
    virtual int setd(const Vector<int> &i, const Vector<int> &nd) override;
    virtual int setd(                      const Vector<int> &nd) override;

    virtual const gentype &y(int i) const override { if ( i >= 0 ) { return y()(i); } return SVM_Scalar::y(i); }

    virtual const Vector<gentype> &alphaVal(void)  const override { return gamma();             }
    virtual       double           alphaVal(int i) const override { return (double) gamma()(i); }

    // General modification and autoset functions

    virtual int randomise(double sparsity) override { SVM_Scalar::isStateOpt = 0; return ML_Base::randomise(sparsity); }

    virtual int scale(double a) override;
    virtual int reset(void) override;
    virtual int restart(void) override { return ML_Base::restart(); }

    virtual int settspaceDim(int newdim) override { return ML_Base::settspaceDim(newdim); }
    virtual int addtspaceFeat(int i)     override { return ML_Base::addtspaceFeat(i);     }
    virtual int removetspaceFeat(int i)  override { return ML_Base::removetspaceFeat(i);  }

    virtual int setorder(int neword) override { return ML_Base::setorder(neword); }

    // Training functions:

    virtual void fudgeOn(void)  override { return; }
    virtual void fudgeOff(void) override { return; }

    virtual int train(int &res) override { svmvolatile int killSwitch = 0; return LSV_Generic::train(res,killSwitch); }
    virtual int train(int &res, svmvolatile int &killSwitch) override;

    // Evaluation Functions:

    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = nullptr) const override { return ML_Base::ghTrainingVector(resh,resg,i,retaltg,pxyprodi); }

    virtual int covTrainingVector(gentype &resv, gentype &resmu, int i, int j, gentype ***pxyprodi = nullptr, gentype ***pxyprodj = nullptr, gentype **pxyprodij = nullptr) const override { return ML_Base::covTrainingVector(resv,resmu,i,j,pxyprodi,pxyprodj,pxyprodij); }

    virtual double eTrainingVector(int i) const override { return ML_Base::eTrainingVector(i); }

    virtual void dgTrainingVectorX(Vector<gentype> &resx, int i) const override;
    virtual void dgTrainingVectorX(Vector<double>  &resx, int i) const override;

    virtual void dgTrainingVectorX(Vector<gentype> &resx, const Vector<int> &i) const override { ML_Base::dgTrainingVectorX(resx,i); return; }
    virtual void dgTrainingVectorX(Vector<double>  &resx, const Vector<int> &i) const override { ML_Base::dgTrainingVectorX(resx,i); return; }

    virtual void dgTrainingVector(Vector<gentype>         &res, gentype        &resn, int i) const override;
    virtual void dgTrainingVector(Vector<double>          &res, double         &resn, int i) const override { ML_Base::dgTrainingVector(res,resn,i); return; }
    virtual void dgTrainingVector(Vector<Vector<double> > &res, Vector<double> &resn, int i) const override { ML_Base::dgTrainingVector(res,resn,i); return; }
    virtual void dgTrainingVector(Vector<d_anion>         &res, d_anion        &resn, int i) const override { ML_Base::dgTrainingVector(res,resn,i); return; }

    virtual void dgTrainingVector(Vector<gentype>         &res, const Vector<int> &i) const override { ML_Base::dgTrainingVector(res,i); return; }
    virtual void dgTrainingVector(Vector<double>          &res, const Vector<int> &i) const override { ML_Base::dgTrainingVector(res,i); return; }
    virtual void dgTrainingVector(Vector<Vector<double> > &res, const Vector<int> &i) const override { ML_Base::dgTrainingVector(res,i); return; }
    virtual void dgTrainingVector(Vector<d_anion>         &res, const Vector<int> &i) const override { ML_Base::dgTrainingVector(res,i); return; }





    // ================================================================
    //     Common functions for all LS-SVMs
    // ================================================================
    //
    // Technically there are a bunch of SVM functions inheritted here.  I
    // probably should make them private, but that would be too much work.
    // Just don't use them.

    virtual       LSV_Generic &getLSV     (void)       { return *this; }
    virtual const LSV_Generic &getLSVconst(void) const { return *this; }

    // General modification and autoset functions

    virtual int setVardelta (void);
    virtual int setZerodelta(void);

    // Constructors, destructors, assignment etc..
    //
    // VarApprox: m == -1: cov is the usual posterior covariance
    //            m == 0:  cov is the prior covariance (kernel)
    //            m > 0:   cov approx posterior using m nearest x's

    virtual int setgamma(const Vector<gentype> &newgamma);
    virtual int setdelta(const gentype         &newdelta);

    virtual int setvarApprox(const int m);

    // Additional information

    virtual int isVardelta (void) const { return SVM_Scalar::isVarBias();   }
    virtual int isZerodelta(void) const { return SVM_Scalar::isFixedBias(); }

    virtual const Vector<gentype> &gamma(void) const { return dalpha; }
    virtual const gentype         &delta(void) const { return dbias;  }

    virtual int varApprox(void) const { return covm; }

    virtual const Matrix<double> &lsvGp(void) const { return SVM_Scalar::Gp(); }






    // ================================================================
    //     Required by K2xfer
    // ================================================================
    //
    // K2xfer requires these, though they aren't really part of LSV.

    virtual const gentype         &bias (void) const override { return delta(); }
    virtual const Vector<gentype> &alpha(void) const override { return gamma(); }





protected:

    // Variables

    Vector<gentype> dalpha;
    gentype dbias;
    int covm;

    // Find vector closest (up to) m free training vectors
    // The vector res will be resized appropriately (but sizing set slack
    // to prevent allocation issues).

//FIXME
//    const Vector<int> &getNearest(int m, const SparseVector<gentype> &loc, Vector<int> &res);
//FIXME: use distK to evaluate kernel distance (for optimality)

    // Fast-coded solution

    mutable int fastdim;
    mutable double *fastweights;
    mutable double **fastxsums;

    mutable int fastdim_base;
    mutable double *fastweights_base; // This will basically be dalphaR in most (all?) cases, but have chosen to leave for future changes/flexibility
    mutable double **fastxsums_base;

    void killfasts(void)
    {
        if ( fastweights )
        {
            MEMDELARRAY(fastweights);
            fastweights = nullptr;
        }

        if ( fastxsums )
        {
            for ( int i = 0 ; i < fastdim ; i++ )
            {
                MEMDELARRAY(fastxsums[i]);
                fastxsums[i] = nullptr;
            }

            MEMDELARRAY(fastxsums);
            fastxsums = nullptr;
        }

        fastdim = 0;

        if ( fastweights_base )
        {
            MEMDELARRAY(fastweights_base);
            fastweights_base = nullptr;
        }

        if ( fastxsums_base )
        {
            for ( int i = 0 ; i < fastdim_base ; i++ )
            {
                MEMDELARRAY(fastxsums_base[i]);
                fastxsums_base[i] = nullptr;
            }

            MEMDELARRAY(fastxsums_base);
            fastxsums_base = nullptr;
        }

        fastdim_base = 0;
    }

    // Set alpha/bias internal

    template <class T> void setAlphaBiasLSV(const Vector<T> &alphais, T &bis)
    {
        // Assumes operator= is defined to convert T to gentype

        dbias = bis;

        int i;

        killfasts();

        dalpha.resize(alphais.size());

        for ( i = 0 ; i < alphais.size() ; ++i )
        {
            dalpha("&",i) = alphais(i);
        }

        return;
    }

    // Targets

    Vector<gentype> alltraintarg;

    // The definition of zero depends on the target type

    virtual gentype &makezero(gentype &val) 
    { 
        val.force_null(); 

        return val; 
    }

    Vector<gentype> &makeveczero(Vector<gentype> &val) 
    { 
        int i; 

        if ( val.size() )
        {
            for ( i = 0 ; i < val.size() ; ++i )
            {
                makezero(val("&",i));
            }
        }

        return val; 
    }
};

inline double norm2(const LSV_Generic &a);
inline double abs2 (const LSV_Generic &a);

inline double norm2(const LSV_Generic &a) { return a.RKHSnorm(); }
inline double abs2 (const LSV_Generic &a) { return a.RKHSabs();  }

inline void qswap(LSV_Generic &a, LSV_Generic &b)
{
    a.qswapinternal(b);

    return;
}

inline LSV_Generic &setzero(LSV_Generic &a)
{
    a.restart();

    return a;
}

inline void LSV_Generic::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    LSV_Generic &b = dynamic_cast<LSV_Generic &>(bb.getML());

    SVM_Scalar::qswapinternal(b);

    qswap(dalpha      ,b.dalpha      );
    qswap(dbias       ,b.dbias       );
    qswap(covm        ,b.covm        );
    qswap(alltraintarg,b.alltraintarg);

    killfasts();
    b.killfasts();

    return;
}

inline void LSV_Generic::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const LSV_Generic &b = dynamic_cast<const LSV_Generic &>(bb.getMLconst());

    SVM_Scalar::semicopy(b);

    dalpha = b.dalpha;
    dbias  = b.dbias;
    covm   = b.covm;
//    alltraintarg = b.alltraintarg;

    killfasts();

    return;
}

inline void LSV_Generic::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const LSV_Generic &src = dynamic_cast<const LSV_Generic &>(bb.getMLconst());

    SVM_Scalar::assign(src,onlySemiCopy);

    dalpha       = src.dalpha;
    dbias        = src.dbias;
    covm         = src.covm;
    alltraintarg = src.alltraintarg;

    killfasts();

    return;
}

#endif
