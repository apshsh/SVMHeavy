
//
// k-nearest-neighbour density estimation
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _knn_densit_h
#define _knn_densit_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "knn_generic.hpp"



class KNN_Densit;


// Swap and zeroing (restarting) functions

inline void qswap(KNN_Densit &a, KNN_Densit &b);
inline KNN_Densit &setzero(KNN_Densit &a);

class KNN_Densit : public KNN_Generic
{
public:

    // Constructors, destructors, assignment etc..

    KNN_Densit();
    KNN_Densit(const KNN_Densit &src);
    KNN_Densit(const KNN_Densit &src, const ML_Base *xsrc);
    KNN_Densit &operator=(const KNN_Densit &src) { assign(src); return *this; }
    virtual ~KNN_Densit();

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0) override;
    virtual void semicopy(const ML_Base &src) override;
    virtual void qswapinternal(ML_Base &b) override;

    virtual std::ostream &printstream(std::ostream &output, int dep) const override;
    virtual std::istream &inputstream(std::istream &input ) override;

    // Information functions (training data):

    virtual int NNC(int d)    const override;
    virtual int type(void)    const override { return 300; }
    virtual int subtype(void) const override { return 0;   }

    virtual int tspaceDim(void)    const override { return 1; }
    virtual int numClasses(void)   const override { return 0; }

    virtual char gOutType(void) const override { return 'R'; }
    virtual char hOutType(void) const override { return 'R'; }
    virtual char targType(void) const override { return 'N'; }
    virtual double calcDist(const gentype &ha, const gentype &hb, int ia = -1, int db = 2) const override;

    virtual const Vector<int> &ClassLabels(void)   const override;
    virtual int getInternalClass(const gentype &y) const override { return ( ( (double) y ) < 0 ) ? 0 : 1; }

    virtual int isClassifier(void) const override { return 0; }

    // Training set modification - need to overload to maintain counts

    virtual int addTrainingVector (int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1) override;
    virtual int qaddTrainingVector(int i, const gentype &y,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1) override;

    virtual int addTrainingVector (int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override;
    virtual int qaddTrainingVector(int i, const Vector<gentype> &y,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override;

    virtual int sety(int i, const gentype &y) override;
    virtual int sety(const Vector<int> &i, const Vector<gentype> &y) override;
    virtual int sety(const Vector<gentype> &y) override;

private:

    virtual void hfn(gentype &res, const Vector<gentype> &yk, const Vector<double> &kdistsq, const Vector<double> &weights, int Nnz, int effkay) const override;
};

inline double norm2(const KNN_Densit &a);
inline double abs2 (const KNN_Densit &a);

inline double norm2(const KNN_Densit &a) { return a.RKHSnorm(); }
inline double abs2 (const KNN_Densit &a) { return a.RKHSabs();  }

inline void qswap(KNN_Densit &a, KNN_Densit &b)
{
    a.qswapinternal(b);

    return;
}

inline KNN_Densit &setzero(KNN_Densit &a)
{
    a.restart();

    return a;
}

inline void KNN_Densit::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    KNN_Densit &b = dynamic_cast<KNN_Densit &>(bb.getML());

    KNN_Generic::qswapinternal(b);

    return;
}

inline void KNN_Densit::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const KNN_Densit &b = dynamic_cast<const KNN_Densit &>(bb.getMLconst());

    KNN_Generic::semicopy(b);

    return;
}

inline void KNN_Densit::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const KNN_Densit &src = dynamic_cast<const KNN_Densit &>(bb.getMLconst());

    KNN_Generic::assign(src,onlySemiCopy);

    return;
}

#endif
