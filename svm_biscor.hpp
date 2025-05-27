
//
// Score SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

// The score SVM is an overlay for rank SVM.  The target is a vector of
// scores.  Based on this it generates inequalities.  For example the dataset
//
// [ 1 1 ] x1
// [ 1 2 ] x2
// [ 2 3 ] x3
// [ 5 4 ] x4
// [ 4 4 ] x5
//
// Generates the inequalities from component 1 of the target:
//
// g(x3) >= 1+g(x1)
// g(x3) >= 1+g(x2)
// g(x4) >= 1+g(x1)
// g(x4) >= 1+g(x2)
// g(x4) >= 1+g(x3)
// g(x5) >= 1+g(x1)
// g(x5) >= 1+g(x2)
// g(x5) >= 1+g(x3)
// 1+g(x5) <= g(x4) -> g(x4) >= 1+g(x5)
//
// and further from component 2 of the target:
//
// g(x2) >= 1+g(x1)
// g(x3) >= 1+g(x1)
// g(x3) >= 1+g(x2)
// g(x4) >= 1+g(x1)
// g(x4) >= 1+g(x2)
// g(x4) >= 1+g(x3)
// g(x5) >= 1+g(x1)
// g(x5) >= 1+g(x2)
// g(x5) >= 1+g(x3)
//
// Giving a net total of 18 inequalities.  This is transformed into a complete
// dataset of 22 vectors for svm_binary, which consists of 4 vectors (d=0)
// and 18 enforced inequalities (d=+1).
//
// 2016_12_19: Moddified to enforce g(xi) <= -1+g(xj).  This has the effect
// of forcing alpha negative, therefore all results tend negative for
// gaussian kernel

#ifndef _svm_biscor_h
#define _svm_biscor_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "svm_binary.hpp"




class SVM_BiScor;


// Swap function

inline void qswap(SVM_BiScor &a, SVM_BiScor &b);


class SVM_BiScor : public SVM_Binary
{
public:

    // Constructors, destructors, assignment etc..

    SVM_BiScor();
    SVM_BiScor(const SVM_BiScor &src);
    SVM_BiScor(const SVM_BiScor &src, const ML_Base *xsrc);
    SVM_BiScor &operator=(const SVM_BiScor &src) { assign(src); return *this; }
    virtual ~SVM_BiScor();

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0) override;
    virtual void semicopy(const ML_Base &src) override;
    virtual void qswapinternal(ML_Base &b) override;

    virtual int restart(void) override { SVM_BiScor temp; *this = temp; return 1; }

    virtual std::ostream &printstream(std::ostream &output, int dep) const override;
    virtual std::istream &inputstream(std::istream &input) override;

    // Information functions (training data):

    virtual int  N(void)        const override { return locN; }
    virtual int  type(void)     const override { return 12;   }
    virtual int  subtype(void)  const override { return 0;    }
    virtual char targType(void) const override { return 'V';  }

    // We need to let these all be the wrong size.  Through polymorphism
    // they will be called by ml_base to access inequalities, and these
    // are stored past the locN boundary

    virtual const Vector<gentype>         &y  (void) const override { return locz; }
    virtual const Vector<double>          &yR (void) const override { static thread_local Vector<double>          dummy; NiceThrow("yR not defined in svm_biscor"); return dummy; }
    virtual const Vector<d_anion>         &yA (void) const override { static thread_local Vector<d_anion>         dummy; NiceThrow("yA not defined in svm_biscor"); return dummy; }
    virtual const Vector<Vector<double> > &yV (void) const override { static thread_local Vector<Vector<double> > dummy; NiceThrow("yV not defined in svm_biscor"); return dummy; }
    virtual const Vector<gentype>         &yp (void) const override { static thread_local Vector<gentype>         dummy; NiceThrow("yp  not defined in svm_biscor"); return dummy; }
    virtual const Vector<double>          &ypR(void) const override { static thread_local Vector<double>          dummy; NiceThrow("ypR not defined in svm_biscor"); return dummy; }
    virtual const Vector<d_anion>         &ypA(void) const override { static thread_local Vector<d_anion>         dummy; NiceThrow("ypA not defined in svm_biscor"); return dummy; }
    virtual const Vector<Vector<double> > &ypV(void) const override { static thread_local Vector<Vector<double> > dummy; NiceThrow("ypV not defined in svm_biscor"); return dummy; }

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

    // ML_Base callback does sigma

    virtual int setepsweight(int                i, double                xepsweight) override;
    virtual int setepsweight(const Vector<int> &i, const Vector<double> &xepsweight) override;
    virtual int setepsweight(                      const Vector<double> &xepsweight) override;

    // Information functions (training data):

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

inline double norm2(const SVM_BiScor &a);
inline double abs2 (const SVM_BiScor &a);

inline double norm2(const SVM_BiScor &a) { return a.RKHSnorm(); }
inline double abs2 (const SVM_BiScor &a) { return a.RKHSabs();  }

inline void qswap(SVM_BiScor &a, SVM_BiScor &b)
{
    a.qswapinternal(b);

    return;
}

inline void SVM_BiScor::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    SVM_BiScor &b = dynamic_cast<SVM_BiScor &>(bb.getML());

    SVM_Binary::qswapinternal(b);
    
    qswap(locN,b.locN);
    qswap(locz,b.locz);
    qswap(locd,b.locd);

    return;
}

inline void SVM_BiScor::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const SVM_BiScor &b = dynamic_cast<const SVM_BiScor &>(bb.getMLconst());

    SVM_Binary::semicopy(b);

    locN = b.locN;
    locz = b.locz;
    locd = b.locd;

    return;
}

inline void SVM_BiScor::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const SVM_BiScor &src = dynamic_cast<const SVM_BiScor &>(bb.getMLconst());

    SVM_Binary::assign(static_cast<const SVM_Binary &>(src),onlySemiCopy);

    locN = src.locN;
    locz = src.locz;
    locd = src.locd;

    return;
}

#endif
