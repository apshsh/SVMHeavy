
//
// Density estimation SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _svm_densit_h
#define _svm_densit_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "svm_scalar.hpp"



// This is basically SVM_Scalar, except that targets are a function of "x".
// For example if x is 1-dimensional:
//
// [ 0 ], [ 1 ], [ 2 ], [ 3 ]
//
// then z will be
//
// 0,.33,.66,1
//
// If out of order, eg:
//
// [ 0 ], [ 1 ], [ 5 ], [ 3 ]
//
// then z will be
//
// 0,.33,1,.66
//
// that is, z depends on where x lies in smallest-largest range.  Multi-dimensional
// x is likewise supported, with ordering into general grid.  So it learns the cdf.
// Evaluating gives the gradient (dense derivative) - that is, it returns the pdf as
// the gradient of the learned pdf.
//
// Uses 1-norm on alpha, general kernels can be used, but by default kernel 400 used.
// Note that kernel must have dense derivatives defined.

class SVM_Densit;


// Swap function

inline void qswap(SVM_Densit &a, SVM_Densit &b);

class SVM_Densit : public SVM_Scalar
{
public:

    // Constructors, destructors, assignment operators and similar

    SVM_Densit();
    SVM_Densit(const SVM_Densit &src);
    SVM_Densit(const SVM_Densit &src, const ML_Base *xsrc);
    SVM_Densit &operator=(const SVM_Densit &src) { assign(src); return *this; }
    virtual ~SVM_Densit() { return; }

    virtual int restart(void) override { SVM_Densit temp; *this = temp; return 1; }

    virtual std::ostream &printstream(std::ostream &output, int dep) const override;
    virtual std::istream &inputstream(std::istream &input) override;

    // Information:

    virtual int tspaceDim(void)  const override { return 1; }
    virtual int numClasses(void) const override { return 0; }
    virtual int type(void)       const override { return 7; }
    virtual int subtype(void)    const override { return 0; }
    virtual int order(void)      const override { return 0; }

    virtual char gOutType(void) const override { return 'R'; }
    virtual char hOutType(void) const override { return 'R'; }
    virtual char targType(void) const override { return 'N'; }

    virtual int isClassifier(void) const override { return 0; }

    // Modification and autoset functions

    virtual int sety(int                i, double                z) override { (void) i; (void) z; NiceThrow("sety not defined for density estimation\n"); return 1; }
    virtual int sety(const Vector<int> &i, const Vector<double> &z) override { (void) i; (void) z; NiceThrow("sety not defined for density estimation\n"); return 1; }
    virtual int sety(                      const Vector<double> &z) override { (void) z;           NiceThrow("sety not defined for density estimation\n"); return 1; }

    // Training set control

    virtual int addTrainingVector (int i, double z, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2) override;
    virtual int qaddTrainingVector(int i, double z,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2) override;

    virtual int addTrainingVector (int i, const Vector<double> &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<int> &d) override;
    virtual int qaddTrainingVector(int i, const Vector<double> &z,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<int> &d) override;

    virtual int addTrainingVector (int i, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2);
    virtual int qaddTrainingVector(int i,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2);

    virtual int addTrainingVector (int i, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<int> &d);
    virtual int qaddTrainingVector(int i,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<int> &d);

    virtual int removeTrainingVector(int i                                      ) override { SparseVector<gentype> x; gentype y; return removeTrainingVector(i,y,x); }
    virtual int removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x) override;
    virtual int removeTrainingVector(int i, int num                             ) override { return ML_Base::removeTrainingVector(i,num); }

    virtual int addTrainingVector (int i, const gentype &z, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2) override;
    virtual int qaddTrainingVector(int i, const gentype &z,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2) override;

    virtual int addTrainingVector (int i, const Vector<gentype> &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override;
    virtual int qaddTrainingVector(int i, const Vector<gentype> &z,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override;

    virtual int sety(int                i, const gentype         &z) override { (void) i; (void) z; NiceThrow("sety  not defined for density estimation\n"); return 0; }
    virtual int sety(const Vector<int> &i, const Vector<gentype> &z) override { (void) i; (void) z; NiceThrow("sety  not defined for density estimation\n"); return 0; }
    virtual int sety(                      const Vector<gentype> &z) override { (void) z; NiceThrow("sety  not defined for density estimation\n"); return 0; }

    virtual int setd(int                i, int                d) override;
    virtual int setd(const Vector<int> &i, const Vector<int> &d) override;
    virtual int setd(                      const Vector<int> &d) override;

    // Train the SVM

    virtual int train(int &res)                              override { svmvolatile int killSwitch = 0; return train(res,killSwitch); }
    virtual int train(int &res, svmvolatile int &killSwitch) override;

    // Evaluation:

    virtual int isVarDefined(void) const override { return 0; }

    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = nullptr) const override;

    // Other functions

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0) override;

    virtual void semicopy(const ML_Base &src) override;
    virtual void qswapinternal(ML_Base &b) override;

private:

    virtual int gTrainingVector(double &res, int &unusedvar, int i, int raw = 0, gentype ***pxyprodi = nullptr) const override;

    void fixz(void);
};



inline double norm2(const SVM_Densit &a);
inline double abs2 (const SVM_Densit &a);

inline double norm2(const SVM_Densit &a) { return a.RKHSnorm(); }
inline double abs2 (const SVM_Densit &a) { return a.RKHSabs();  }

inline void qswap(SVM_Densit &a, SVM_Densit &b)
{
    a.qswapinternal(b);

    return;
}

inline void SVM_Densit::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    SVM_Densit &b = dynamic_cast<SVM_Densit &>(bb.getML());

    SVM_Scalar::qswapinternal(b);

    return;
}

inline void SVM_Densit::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const SVM_Densit &b = dynamic_cast<const SVM_Densit &>(bb.getMLconst());

    SVM_Scalar::semicopy(b);

    return;
}

inline void SVM_Densit::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const SVM_Densit &src = dynamic_cast<const SVM_Densit &>(bb.getMLconst());

    SVM_Scalar::assign(src,onlySemiCopy);

    return;
}

#endif
