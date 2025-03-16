
//
// 1 layer neural network scalar regression
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _onn_scalar_h
#define _onn_scalar_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "onn_generic.hpp"



class ONN_Scalar;


// Swap and zeroing (restarting) functions

inline void qswap(ONN_Scalar &a, ONN_Scalar &b);
inline ONN_Scalar &setzero(ONN_Scalar &a);

class ONN_Scalar : public ONN_Generic
{
public:

    // Constructors, destructors, assignment etc..

    ONN_Scalar();
    ONN_Scalar(const ONN_Scalar &src);
    ONN_Scalar(const ONN_Scalar &src, const ML_Base *xsrc);
    ONN_Scalar &operator=(const ONN_Scalar &src) { assign(src); return *this; }
    virtual ~ONN_Scalar();

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0) override;
    virtual void semicopy(const ML_Base &src) override;
    virtual void qswapinternal(ML_Base &b) override;

    virtual std::ostream &printstream(std::ostream &output, int dep) const override;
    virtual std::istream &inputstream(std::istream &input ) override;

    // Information functions (training data):

    virtual int NNC(int d)    const override { return classcnt(d+1); }
    virtual int type(void)    const override { return 100;           }
    virtual int subtype(void) const override { return 0;             }

    virtual int tspaceDim(void)  const override { return 1; }
    virtual int numClasses(void) const override { return 0; }
    virtual int order(void)      const override { return 0; }

    virtual char gOutType(void) const override { return 'R'; }
    virtual char hOutType(void) const override { return 'R'; }
    virtual char targType(void) const override { return 'R'; }
    virtual double calcDist(const gentype &ha, const gentype &hb, int ia = -1, int db = 2) const override;

    virtual const Vector<int> &ClassLabels(void)   const override { return classlabels; }
    virtual int getInternalClass(const gentype &y) const override { return ( ( (double) y ) < 0 ) ? 0 : 1; }
    virtual int numInternalClasses(void)           const override { return 2; }

    virtual int isUnderlyingScalar(void) const override { return 1; }
    virtual int isUnderlyingVector(void) const override { return 0; }
    virtual int isUnderlyingAnions(void) const override { return 0; }

    virtual int isClassifier(void) const override { return 0; }

    // Training set modification - need to overload to maintain counts

    virtual int addTrainingVector (int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1) override;
    virtual int qaddTrainingVector(int i, const gentype &y,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1) override;

    virtual int addTrainingVector (int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override;
    virtual int qaddTrainingVector(int i, const Vector<gentype> &y,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override;

    virtual int removeTrainingVector(int i                                      ) override { SparseVector<gentype> x; gentype y; return removeTrainingVector(i,y,x); }
    virtual int removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x) override;
    virtual int removeTrainingVector(int i, int num                             ) override { return ML_Base::removeTrainingVector(i,num); }

    virtual int sety(int i, const gentype &y) override;
    virtual int sety(const Vector<int> &i, const Vector<gentype> &y) override;
    virtual int sety(const Vector<gentype> &y) override;

    virtual int setd(int i, int d) override;
    virtual int setd(const Vector<int> &i, const Vector<int> &d) override;
    virtual int setd(const Vector<int> &d) override;







private:

    Vector<int> classlabels;
    Vector<int> classcnt;
};

inline double norm2(const ONN_Scalar &a);
inline double abs2 (const ONN_Scalar &a);

inline double norm2(const ONN_Scalar &a) { return a.RKHSnorm(); }
inline double abs2 (const ONN_Scalar &a) { return a.RKHSabs();  }

inline void qswap(ONN_Scalar &a, ONN_Scalar &b)
{
    a.qswapinternal(b);

    return;
}

inline ONN_Scalar &setzero(ONN_Scalar &a)
{
    a.restart();

    return a;
}

inline void ONN_Scalar::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    ONN_Scalar &b = dynamic_cast<ONN_Scalar &>(bb.getML());

    ONN_Generic::qswapinternal(b);

    qswap(classlabels,b.classlabels);
    qswap(classcnt   ,b.classcnt   );

    return;
}

inline void ONN_Scalar::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const ONN_Scalar &b = dynamic_cast<const ONN_Scalar &>(bb.getMLconst());

    ONN_Generic::semicopy(b);

    classlabels = b.classlabels;
    classcnt    = b.classcnt;

    return;
}

inline void ONN_Scalar::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const ONN_Scalar &src = dynamic_cast<const ONN_Scalar &>(bb.getMLconst());

    ONN_Generic::assign(src,onlySemiCopy);

    classlabels = src.classlabels;
    classcnt    = src.classcnt;

    return;
}

#endif
