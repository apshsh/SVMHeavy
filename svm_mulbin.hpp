
//
// Multi-user Binary Classification SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _svm_mulbin_h
#define _svm_mulbin_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "svm_mvrank.hpp"








class SVM_MulBin;


// Swap function

inline void qswap(SVM_MulBin &a, SVM_MulBin &b);


class SVM_MulBin : public SVM_MvRank
{
public:

    SVM_MulBin();
    SVM_MulBin(const SVM_MulBin &src);
    SVM_MulBin(const SVM_MulBin &src, const ML_Base *xsrc);
    SVM_MulBin &operator=(const SVM_MulBin &src) { assign(src); return *this; }
    virtual ~SVM_MulBin();

    virtual int prealloc(int expectedN) override;
    virtual int preallocsize(void) const override;

    virtual int restart(void) override { SVM_MulBin temp; *this = temp; return 1; }

    virtual std::ostream &printstream(std::ostream &output, int dep) const override;
    virtual std::istream &inputstream(std::istream &input) override;

    // Information:

    virtual int type(void)    const override { return 18; }
    virtual int subtype(void) const override { return 0;  }

    virtual const Vector<gentype>         &y        (void) const override { return locy;  }
    virtual const Vector<double>          &yR       (void) const override { return loczR; }
    virtual const Vector<d_anion>         &yA       (void) const override { NiceThrow("yA not defined in svm_mulbin");  const static Vector<d_anion>         dummy; return dummy; }
    virtual const Vector<Vector<double> > &yV       (void) const override { NiceThrow("yV not defined in svm_mulbin");  const static Vector<Vector<double> > dummy; return dummy; }
    virtual const Vector<gentype>         &yp       (void) const override { NiceThrow("yp  not defined in svm_mulbin"); const static Vector<gentype>         dummy; return dummy; }
    virtual const Vector<double>          &ypR      (void) const override { NiceThrow("ypA not defined in svm_mulbin"); const static Vector<double>          dummy; return dummy; }
    virtual const Vector<d_anion>         &ypA      (void) const override { NiceThrow("ypA not defined in svm_mulbin"); const static Vector<d_anion>         dummy; return dummy; }
    virtual const Vector<Vector<double> > &ypV      (void) const override { NiceThrow("ypV not defined in svm_mulbin"); const static Vector<Vector<double> > dummy; return dummy; }
    virtual const Vector<double>  &zR(void) const override { return loczR; }

    virtual double zR(int i) const override { if ( i >= 0 ) { return zR()(i); } return 0; } // Tests always set zR(-1) = 0, so this is safe

    // Training set control

    virtual int addTrainingVector (int i, const gentype &z, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2) override;
    virtual int qaddTrainingVector(int i, const gentype &z,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2) override;

    virtual int addTrainingVector (int i, const Vector<gentype> &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override;
    virtual int qaddTrainingVector(int i, const Vector<gentype> &z,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override;

    virtual int addTrainingVector (int i, int d, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, double z = 0.0);
    virtual int qaddTrainingVector(int i, int d,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, double z = 0.0);

    virtual int addTrainingVector (int i, const Vector<int> &d, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<double> &z);
    virtual int qaddTrainingVector(int i, const Vector<int> &d,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<double> &z);

    virtual int removeTrainingVector(int i                                      ) override { SparseVector<gentype> x; gentype y; return removeTrainingVector(i,y,x); }
    virtual int removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x) override;
    virtual int removeTrainingVector(int i, int num                             ) override { return ML_Base::removeTrainingVector(i,num); }

    virtual int sety(int                i, double                z) override;
    virtual int sety(const Vector<int> &i, const Vector<double> &z) override;
    virtual int sety(                      const Vector<double> &z) override;

    virtual int sety(int                i, const gentype         &z) override;
    virtual int sety(const Vector<int> &i, const Vector<gentype> &z) override;
    virtual int sety(                      const Vector<gentype> &z) override;

    virtual int setd(int                i, int                d) override;
    virtual int setd(const Vector<int> &i, const Vector<int> &d) override;
    virtual int setd(                      const Vector<int> &d) override;

    // Other functions

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0) override;

    virtual void semicopy(const ML_Base &src) override;
    virtual void qswapinternal(ML_Base &b) override;

private:

    Vector<double> loczR;
    Vector<gentype> locy;
};

inline double norm2(const SVM_MulBin &a);
inline double abs2 (const SVM_MulBin &a);

inline double norm2(const SVM_MulBin &a) { return a.RKHSnorm(); }
inline double abs2 (const SVM_MulBin &a) { return a.RKHSabs();  }

inline void qswap(SVM_MulBin &a, SVM_MulBin &b)
{
    a.qswapinternal(b);

    return;
}

inline void SVM_MulBin::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    SVM_MulBin &b = dynamic_cast<SVM_MulBin &>(bb.getML());

    SVM_MvRank::qswapinternal(b);

    qswap(loczR,b.loczR);
    qswap(locy, b.locy );

    return;
}

inline void SVM_MulBin::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const SVM_MulBin &b = dynamic_cast<const SVM_MulBin &>(bb.getMLconst());

    SVM_MvRank::semicopy(b);

    //y,zR

    return;
}

inline void SVM_MulBin::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const SVM_MulBin &src = dynamic_cast<const SVM_MulBin &>(bb.getMLconst());

    SVM_MvRank::assign(static_cast<const SVM_MvRank &>(src),onlySemiCopy);

    locy  = src.locy;
    loczR = src.loczR;

    return;
}

#endif
