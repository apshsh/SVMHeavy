
//
// Binary Classification SSV
//
// Version: 7
// Date: 01/12/2017
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _ssv_binary_h
#define _ssv_binary_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "ssv_scalar.hpp"








class SSV_Binary;


// Swap function

inline void qswap(SSV_Binary &a, SSV_Binary &b);


class SSV_Binary : public SSV_Scalar
{
public:

    SSV_Binary();
    SSV_Binary(const SSV_Binary &src);
    SSV_Binary(const SSV_Binary &src, const ML_Base *xsrc);
    SSV_Binary &operator=(const SSV_Binary &src) { assign(src); return *this; }
    virtual ~SSV_Binary();

    virtual int prealloc(int expectedN) override;
    virtual int preallocsize(void) const override;

    virtual int restart(void) override { SSV_Binary temp; *this = temp; return 1; }

    virtual std::ostream &printstream(std::ostream &output, int dep) const override;
    virtual std::istream &inputstream(std::istream &input) override;

    // Information:

    virtual int tspaceDim(void)  const override { return 1;   }
    virtual int numClasses(void) const override { return 2;   }
    virtual int type(void)       const override { return 701; }
    virtual int subtype(void)    const override { return 0;   }

    virtual char gOutType(void) const override { return 'R'; }
    virtual char hOutType(void) const override { return 'Z'; }
    virtual char targType(void) const override { return 'Z'; }
    virtual double calcDist(const gentype &ha, const gentype &hb, int ia = -1, int db = 2) const override;

    virtual int isClassifier(void) const override { return 1; }

    // Training set control

    virtual int addTrainingVector (int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1) override;
    virtual int qaddTrainingVector(int i, const gentype &y,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1) override;

    virtual int addTrainingVector (int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override;
    virtual int qaddTrainingVector(int i, const Vector<gentype> &y,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override;

    virtual int removeTrainingVector(int i                                      ) override { SparseVector<gentype> x; gentype y; return removeTrainingVector(i,y,x); }
    virtual int removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x) override;
    virtual int removeTrainingVector(int i, int num                             ) override { return ML_Base::removeTrainingVector(i,num); }

    virtual int sety(int i, const gentype &z) override;
    virtual int sety(const Vector<int> &i, const Vector<gentype> &z) override;
    virtual int sety(const Vector<gentype> &z) override;

    virtual int setd(int i, int d) override;
    virtual int setd(const Vector<int> &i, const Vector<int> &d) override;
    virtual int setd(const Vector<int> &d) override;

    // Evaluation:

    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = nullptr) const override;

    // Other functions

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0) override;

    virtual void semicopy(const ML_Base &src) override;
    virtual void qswapinternal(ML_Base &b) override;

private:

    // Largely just copied from SVM_Binary

    int setdinternal(int i, int d);
};

inline double norm2(const SSV_Binary &a);
inline double abs2 (const SSV_Binary &a);

inline double norm2(const SSV_Binary &a) { return a.RKHSnorm(); }
inline double abs2 (const SSV_Binary &a) { return a.RKHSabs();  }

inline void qswap(SSV_Binary &a, SSV_Binary &b)
{
    a.qswapinternal(b);

    return;
}

inline void SSV_Binary::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    SSV_Binary &b = dynamic_cast<SSV_Binary &>(bb.getML());

    SSV_Scalar::qswapinternal(b);

    return;
}

inline void SSV_Binary::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const SSV_Binary &b = dynamic_cast<const SSV_Binary &>(bb.getMLconst());

    SSV_Scalar::semicopy(b);

    return;
}

inline void SSV_Binary::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const SSV_Binary &src = dynamic_cast<const SSV_Binary &>(bb.getMLconst());

    SSV_Scalar::assign(static_cast<const SSV_Scalar &>(src),onlySemiCopy);

    return;
}

#endif
