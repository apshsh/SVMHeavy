
//
// Average result block
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _blk_avesca_h
#define _blk_avesca_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "blk_generic.hpp"


// Defines a very basic set of blocks for use in machine learning.


class BLK_AveSca;


// Swap and zeroing (restarting) functions

inline void qswap(BLK_AveSca &a, BLK_AveSca &b);


class BLK_AveSca : public BLK_Generic
{
public:

    // Assumptions: all virtual functions inherited from ML_Base are left
    // unchanged in this class.

    // Constructors, destructors, assignment etc..

    BLK_AveSca(int isIndPrune = 0);
    BLK_AveSca(const BLK_AveSca &src, int isIndPrune = 0)                      : BLK_Generic(isIndPrune) { setaltx(nullptr); assign(src,0);  return; }
    BLK_AveSca(const BLK_AveSca &src, const ML_Base *xsrc, int isIndPrune = 0) : BLK_Generic(isIndPrune) { setaltx(xsrc); assign(src,-1); return; }
    BLK_AveSca &operator=(const BLK_AveSca &src) { assign(src); return *this; }
    virtual ~BLK_AveSca() { return; }

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0) override;
    virtual void semicopy(const ML_Base &src) override;
    virtual void qswapinternal(ML_Base &b) override;

    virtual std::ostream &printstream(std::ostream &output, int dep) const override;
    virtual std::istream &inputstream(std::istream &input ) override;

    // Information functions (training data):

    virtual int NNC(int d)    const override { return classcnt(d+1); }
    virtual int type(void)    const override { return 202;           }
    virtual int subtype(void) const override { return 0;             }

    virtual int tspaceDim(void)    const override { return 1; }
    virtual int numClasses(void)   const override { return 0; }

    virtual char gOutType(void) const override { return 'R'; }
    virtual char hOutType(void) const override { return 'R'; }
    virtual char targType(void) const override { return 'R'; }
    virtual double calcDist(const gentype &ha, const gentype &hb, int ia = -1, int db = 2) const override;

    virtual const Vector<int> &ClassLabels(void)   const override { return classlabels; }
    virtual int getInternalClass(const gentype &y) const override { return ( ( (double) y ) < 0 ) ? 0 : 1; }

    virtual int isClassifier(void) const override { return 0; }

    // Training set modification - need to overload to maintain counts

    virtual int addTrainingVector (int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1) override;
    virtual int qaddTrainingVector(int i, const gentype &y,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1) override;

    virtual int addTrainingVector (int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override;
    virtual int qaddTrainingVector(int i, const Vector<gentype> &y,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override;

    virtual int removeTrainingVector(int i                                      ) override { SparseVector<gentype> x; gentype y; return removeTrainingVector(i,y,x); }
    virtual int removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x) override;
    virtual int removeTrainingVector(int i, int num                             ) override { return ML_Base::removeTrainingVector(i,num); }

    virtual int sety(int                i, const gentype         &y) override;
    virtual int sety(const Vector<int> &i, const Vector<gentype> &y) override;
    virtual int sety(                      const Vector<gentype> &y) override;

    virtual int setd(int                i, int                d) override;
    virtual int setd(const Vector<int> &i, const Vector<int> &d) override;
    virtual int setd(                      const Vector<int> &d) override;

    // Evaluation Functions:
    //
    // Output g(x) is average of input vector.
    // Output h(x) is g(x) with outfn applied to it (or g(x) if outfn null).
    // Raw output is sum of vectors (not average)

    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = nullptr) const override;

    virtual void dgTrainingVector(Vector<gentype>         &res, gentype        &resn, int i) const override;
    virtual void dgTrainingVector(Vector<double>          &res, double         &resn, int i) const override { ML_Base::dgTrainingVector(res,resn,i); return; }
    virtual void dgTrainingVector(Vector<Vector<double> > &res, Vector<double> &resn, int i) const override { ML_Base::dgTrainingVector(res,resn,i); return; }
    virtual void dgTrainingVector(Vector<d_anion>         &res, d_anion        &resn, int i) const override { ML_Base::dgTrainingVector(res,resn,i); return; }

    virtual void dgTrainingVector(Vector<gentype>         &res, const Vector<int> &i) const override;
    virtual void dgTrainingVector(Vector<double>          &res, const Vector<int> &i) const override { ML_Base::dgTrainingVector(res,i); return; }
    virtual void dgTrainingVector(Vector<Vector<double> > &res, const Vector<int> &i) const override { ML_Base::dgTrainingVector(res,i); return; }
    virtual void dgTrainingVector(Vector<d_anion>         &res, const Vector<int> &i) const override { ML_Base::dgTrainingVector(res,i); return; }

private:

    Vector<int> classlabels;
    Vector<int> classcnt;
};

inline double norm2(const BLK_AveSca &a);
inline double abs2 (const BLK_AveSca &a);

inline double norm2(const BLK_AveSca &a) { return a.RKHSnorm(); }
inline double abs2 (const BLK_AveSca &a) { return a.RKHSabs();  }

inline void qswap(BLK_AveSca &a, BLK_AveSca &b)
{
    a.qswapinternal(b);

    return;
}

inline void BLK_AveSca::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    BLK_AveSca &b = dynamic_cast<BLK_AveSca &>(bb.getML());

    BLK_Generic::qswapinternal(b);

    qswap(classlabels,b.classlabels);
    qswap(classcnt   ,b.classcnt   );

    return;
}

inline void BLK_AveSca::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const BLK_AveSca &b = dynamic_cast<const BLK_AveSca &>(bb.getMLconst());

    BLK_Generic::semicopy(b);

    classlabels = b.classlabels;
    classcnt    = b.classcnt;

    return;
}

inline void BLK_AveSca::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const BLK_AveSca &src = dynamic_cast<const BLK_AveSca &>(bb.getMLconst());

    BLK_Generic::assign(src,onlySemiCopy);

    classlabels = src.classlabels;
    classcnt    = src.classcnt;

    return;
}

#endif
