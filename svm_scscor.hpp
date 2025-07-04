
//
// Scalar Regression with Score SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

// Like Score SVM, but also allows for equality constraints.  For a given
// vector if the target is a scalar then this enforces the equality:
//
// g(xi) = yi
//
// whereas if the target is a vector then this is interpretted as a score
// and enforced as per svm_biscor.

#ifndef _svm_scscor_h
#define _svm_scscor_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "svm_scalar.hpp"




class SVM_ScScor;
template <class T> class SVM_Vector_redbin;


// Swap function

inline void qswap(SVM_ScScor &a, SVM_ScScor &b);


class SVM_ScScor : public SVM_Scalar
{
    friend class SVM_Vector_redbin<SVM_ScScor>;

public:

    // Constructors, destructors, assignment etc..

    SVM_ScScor();
    SVM_ScScor(const SVM_ScScor &src);
    SVM_ScScor(const SVM_ScScor &src, const ML_Base *xsrc);
    SVM_ScScor &operator=(const SVM_ScScor &src) { assign(src); return *this; }
    virtual ~SVM_ScScor();

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0) override;
    virtual void semicopy(const ML_Base &src) override;
    virtual void qswapinternal(ML_Base &b) override;

    virtual int restart(void) override { SVM_ScScor temp; *this = temp; return 1; }

    virtual std::ostream &printstream(std::ostream &output, int dep) const override;
    virtual std::istream &inputstream(std::istream &input) override;

    // Information functions (training data):
    //
    // targType is ? as each target can be either a vector (indicating a
    // score constraint) or a scalar (indicating an equality constraint).

    virtual int  N(void)        const override { return locN; }
    virtual int  type(void)     const override { return 13;   }
    virtual int  subtype(void)  const override { return 0;    }
    virtual char targType(void) const override { return '?';  }

    // We need to let these all be the wrong size.  Through polymorphism
    // they will be called by ml_base to access inequalities, and these
    // are stored past the locN boundary

    virtual const Vector<gentype>         &y (void) const override { return locz; }
    virtual const Vector<double>          &yR(void) const override { NiceThrow("yR not defined in svm_scscor"); const static Vector<double>          dummy; return dummy; }
    virtual const Vector<d_anion>         &yA(void) const override { NiceThrow("yA not defined in svm_scscor"); const static Vector<d_anion>         dummy; return dummy; }
    virtual const Vector<Vector<double> > &yV(void) const override { NiceThrow("yV not defined in svm_scscor"); const static Vector<Vector<double> > dummy; return dummy; }

    // Training set modification:
    //
    // setd is interpretted as "set d for all inequalities associated with
    // given vector", which is what is required for simple disable operation.
    // When active all d values for inequalities are set to 1 (everything
    // is done in terms of >= constraints)

    virtual int addTrainingVector (int i, const gentype &z, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2) override;
    virtual int qaddTrainingVector(int i, const gentype &z,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2) override;

    virtual int addTrainingVector (int i, const Vector<gentype> &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override;
    virtual int qaddTrainingVector(int i, const Vector<gentype> &z,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override;

    virtual int removeTrainingVector(int i                                      ) override;
    virtual int removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x) override;
    virtual int removeTrainingVector(int i, int num                             ) override;

    virtual int setx(int                i, const SparseVector<gentype>          &x) override;
    virtual int setx(const Vector<int> &i, const Vector<SparseVector<gentype> > &x) override;
    virtual int setx(                      const Vector<SparseVector<gentype> > &x) override;

    virtual int qswapx(int                i, SparseVector<gentype>          &x, int dontupdate = 0) override;
    virtual int qswapx(const Vector<int> &i, Vector<SparseVector<gentype> > &x, int dontupdate = 0) override;
    virtual int qswapx(                      Vector<SparseVector<gentype> > &x, int dontupdate = 0) override;

    virtual int sety(int                i, const gentype         &z) override;
    virtual int sety(const Vector<int> &i, const Vector<gentype> &z) override;
    virtual int sety(                      const Vector<gentype> &z) override;

    virtual int setd(int                i, int                d) override;
    virtual int setd(const Vector<int> &i, const Vector<int> &d) override;
    virtual int setd(                      const Vector<int> &d) override;

    virtual int setCweight(int                i, double                xCweight) override;
    virtual int setCweight(const Vector<int> &i, const Vector<double> &xCweight) override;
    virtual int setCweight(                      const Vector<double> &xCweight) override;

    virtual int setCweightfuzz(int                i, double                xCweight) override;
    virtual int setCweightfuzz(const Vector<int> &i, const Vector<double> &xCweight) override;
    virtual int setCweightfuzz(                      const Vector<double> &xCweight) override;

    virtual int setepsweight(int                i, double                xepsweight) override;
    virtual int setepsweight(const Vector<int> &i, const Vector<double> &xepsweight) override;
    virtual int setepsweight(                      const Vector<double> &xepsweight) override;

private:

    // Process inequalities for vector i and add them to the training set
    // for next level down

    int processz(int i);

    // locN: number of training vectors
    // locz: score vector

    int locN;
    Vector<gentype> locz;
    Vector<int> locd;
};

inline double norm2(const SVM_ScScor &a);
inline double abs2 (const SVM_ScScor &a);

inline double norm2(const SVM_ScScor &a) { return a.RKHSnorm(); }
inline double abs2 (const SVM_ScScor &a) { return a.RKHSabs();  }

inline void qswap(SVM_ScScor &a, SVM_ScScor &b)
{
    a.qswapinternal(b);

    return;
}

inline void SVM_ScScor::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    SVM_ScScor &b = dynamic_cast<SVM_ScScor &>(bb.getML());

    SVM_Scalar::qswapinternal(b);

    qswap(locN,b.locN);
    qswap(locz,b.locz);
    qswap(locd,b.locd);

    return;
}

inline void SVM_ScScor::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const SVM_ScScor &b = dynamic_cast<const SVM_ScScor &>(bb.getMLconst());

    SVM_Scalar::semicopy(b);

    locN = b.locN;
    locz = b.locz;
    locd = b.locd;

    return;
}

inline void SVM_ScScor::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const SVM_ScScor &src = dynamic_cast<const SVM_ScScor &>(bb.getMLconst());

    SVM_Scalar::assign(static_cast<const SVM_Scalar &>(src),onlySemiCopy);

    locN = src.locN;
    locz = src.locz;
    locd = src.locd;

    return;
}

#endif
