
//
// LS-SVM anionic class
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _lsv_anions_h
#define _lsv_anions_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "lsv_generic.hpp"


class LSV_Anions;

// Swap and zeroing (restarting) functions

inline void qswap(LSV_Anions &a, LSV_Anions &b);
inline LSV_Anions &setzero(LSV_Anions &a);

class LSV_Anions : public LSV_Generic
{
public:

    // Constructors, destructors, assignment etc..

    LSV_Anions();
    LSV_Anions(const LSV_Anions &src);
    LSV_Anions(const LSV_Anions &src, const ML_Base *srcx);
    LSV_Anions &operator=(const LSV_Anions &src) { assign(src); return *this; }
    virtual ~LSV_Anions() { return; }

    virtual int prealloc(int expectedN) override;

    virtual std::ostream &printstream(std::ostream &output, int dep) const override;
    virtual std::istream &inputstream(std::istream &input ) override;

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0) override;
    virtual void semicopy(const ML_Base &src) override;
    virtual void qswapinternal(ML_Base &b) override;

    // Information functions (training data):

    virtual int type(void)      const override { return 502; }
    virtual int subtype(void)   const override { return 0;   }
    virtual int tspaceDim(void) const override { return dbias.size(); }
    virtual int order(void)     const override { return dbias.order(); }

    virtual char gOutType(void) const override { return 'A'; }
    virtual char hOutType(void) const override { return 'A'; }
    virtual char targType(void) const override { return 'A'; }

    virtual int isUnderlyingScalar(void) const override { return 0; }
    virtual int isUnderlyingVector(void) const override { return 0; }
    virtual int isUnderlyingAnions(void) const override { return 1; }

    virtual int setorder(int neword) override;

    virtual int getInternalClass(const gentype &y) const override;

    // Training set modification:

    virtual int addTrainingVector (int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2) override;
    virtual int qaddTrainingVector(int i, const gentype &y,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2) override;

    virtual int addTrainingVector (int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override { return LSV_Generic::addTrainingVector (i,y,x,Cweigh,epsweigh); }
    virtual int qaddTrainingVector(int i, const Vector<gentype> &y,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override { return LSV_Generic::qaddTrainingVector(i,y,x,Cweigh,epsweigh); }

    virtual int removeTrainingVector(int i) override { SparseVector<gentype> x; gentype y; return removeTrainingVector(i,y,x); }
    virtual int removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x) override;
    virtual int removeTrainingVector(int i, int num) override { return ML_Base::removeTrainingVector(i,num); }

    virtual int sety(int                i, const gentype         &y) override;
    virtual int sety(const Vector<int> &i, const Vector<gentype> &y) override;
    virtual int sety(                      const Vector<gentype> &y) override;

    virtual int sety(int                i, const d_anion         &y) override;
    virtual int sety(const Vector<int> &i, const Vector<d_anion> &y) override;
    virtual int sety(                      const Vector<d_anion> &y) override;

    virtual int setd(int                i, int                nd) override;
    virtual int setd(const Vector<int> &i, const Vector<int> &nd) override;
    virtual int setd(                      const Vector<int> &nd) override;

    // General modification and autoset functions

    virtual int scale(double a) override;
    virtual int reset(void) override;
    virtual int restart(void) override { LSV_Anions temp; *this = temp; return 1; }

    // Training functions:

    virtual int train(int &res)                              override { svmvolatile int killSwitch = 0; return train(res,killSwitch); }
    virtual int train(int &res, svmvolatile int &killSwitch) override;

    // Use functions

    virtual int gh(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = nullptr) const override;
    virtual int gh(gentype &resh, gentype &resg, const SparseVector<gentype> &x, int retaltg = 0, const vecInfo *xinf = nullptr, gentype ***pxyprodx = nullptr) const override { return LSV_Generic::gh(resh,resg,x,retaltg,xinf,pxyprodx); }

    virtual double eTrainingVector(int i) const override;

    virtual double         &dedgTrainingVector(double         &res, int i) const override { return ML_Base::dedgTrainingVector(res,i); }
    virtual Vector<double> &dedgTrainingVector(Vector<double> &res, int i) const override { return ML_Base::dedgTrainingVector(res,i); }
    virtual d_anion        &dedgTrainingVector(d_anion        &res, int i) const override { return ML_Base::dedgTrainingVector(res,i); }
    virtual gentype        &dedgTrainingVector(gentype        &res, int i) const override;

    virtual double &d2edg2TrainingVector(double &res, int i) const override { return ML_Base::d2edg2TrainingVector(res,i); }

    virtual double dedKTrainingVector(int i, int j) const override { d_anion tmp(order()); d_anion resg(order()); return innerProduct(resg,dedgTrainingVector(tmp,i),alphaA()(j)).realpart(); }
    virtual Vector<double> &dedKTrainingVector(Vector<double> &res, int i) const override;
    virtual Matrix<double> &dedKTrainingVector(Matrix<double> &res) const override;

    // ================================================================
    //     Common functions for all LS-SVMs
    // ================================================================

    virtual int setgamma(const Vector<gentype> &newgamma) override;
    virtual int setdelta(const gentype         &newdelta) override;

    // ================================================================
    //     Required by K2xfer
    // ================================================================

    virtual const d_anion         &biasA (void) const override { return dbiasA;  }
    virtual const Vector<d_anion> &alphaA(void) const override { return dalphaA; }

private:

    virtual gentype &makezero(gentype &val) override
    { 
        val.force_anion().setorder(order()) *= 0.0;

        return val; 
    }

    Vector<d_anion> dalphaA;
    d_anion dbiasA;

    Vector<d_anion> alltraintargA;
};

inline double norm2(const LSV_Anions &a);
inline double abs2 (const LSV_Anions &a);

inline double norm2(const LSV_Anions &a) { return a.RKHSnorm(); }
inline double abs2 (const LSV_Anions &a) { return a.RKHSabs();  }

inline void qswap(LSV_Anions &a, LSV_Anions &b)
{
    a.qswapinternal(b);

    return;
}

inline LSV_Anions &setzero(LSV_Anions &a)
{
    a.restart();

    return a;
}

inline void LSV_Anions::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    LSV_Anions &b = dynamic_cast<LSV_Anions &>(bb.getML());

    LSV_Generic::qswapinternal(b);

    qswap(dalphaA      ,b.dalphaA      );
    qswap(dbiasA       ,b.dbiasA       );
    qswap(alltraintargA,b.alltraintargA);

    return;
}

inline void LSV_Anions::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const LSV_Anions &b = dynamic_cast<const LSV_Anions &>(bb.getMLconst());

    LSV_Generic::semicopy(b);

    dalphaA       = b.dalphaA;
    dbiasA        = b.dbiasA;
//    alltraintargA = b.alltraintargA;

    return;
}

inline void LSV_Anions::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const LSV_Anions &src = dynamic_cast<const LSV_Anions &>(bb.getMLconst());

    LSV_Generic::assign(src,onlySemiCopy);

    dalphaA       = src.dalphaA;
    dbiasA        = src.dbiasA;
    alltraintargA = src.alltraintargA;

    return;
}


#endif
