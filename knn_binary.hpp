
//
// k-nearest-neighbour binary classifier
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _knn_binary_h
#define _knn_binary_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "knn_generic.hpp"



class KNN_Binary;


// Swap and zeroing (restarting) functions

inline void qswap(KNN_Binary &a, KNN_Binary &b);
inline KNN_Binary &setzero(KNN_Binary &a);

class KNN_Binary : public KNN_Generic
{
public:

    // Constructors, destructors, assignment etc..

    KNN_Binary();
    KNN_Binary(const KNN_Binary &src);
    KNN_Binary(const KNN_Binary &src, const ML_Base *xsrc);
    KNN_Binary &operator=(const KNN_Binary &src) { assign(src); return *this; }
    virtual ~KNN_Binary();

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0) override;
    virtual void semicopy(const ML_Base &src) override;
    virtual void qswapinternal(ML_Base &b) override;

    virtual std::ostream &printstream(std::ostream &output, int dep) const override;
    virtual std::istream &inputstream(std::istream &input ) override;

    // Information functions (training data):

    virtual int NNC(int d)    const override { return classcnt(d+1); }
    virtual int type(void)    const override { return 301;           }
    virtual int subtype(void) const override { return 0;             }

    virtual int tspaceDim(void)          const override { return 1; }
    virtual int numClasses(void)         const override { return 2; }

    virtual char gOutType(void) const override { return 'R'; }
    virtual char hOutType(void) const override { return 'Z'; }
    virtual char targType(void) const override { return 'Z'; }
    virtual double calcDist(const gentype &ha, const gentype &hb, int ia = -1, int db = 2) const override;

    virtual const Vector<int> &ClassLabels(void)   const override { return classlabels;     }
    virtual int getInternalClass(const gentype &y) const override { NiceAssert( y.isValInteger() ); NiceAssert( ( (int) y == -1 ) || ( (int) y == +1 ) ); return (((int) y)+1)/2; }

    virtual int isClassifier(void) const override { return 1; }

    // Training set modification - need to overload to maintain counts

    virtual int addTrainingVector (int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2) override;
    virtual int qaddTrainingVector(int i, const gentype &y,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2) override;

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

    // Randomisation: for KNN type, randomisation occurs in the target, as
    // this is the only variable available

    virtual int randomise(double sparsity) override;

private:

    Vector<int> classlabels;
    Vector<int> classcnt;

    virtual void hfn(gentype &res, const Vector<gentype> &yk, const Vector<double> &kdistsq, const Vector<double> &weights, int Nnz, int effkay) const override;
};

inline double norm2(const KNN_Binary &a);
inline double abs2 (const KNN_Binary &a);

inline double norm2(const KNN_Binary &a) { return a.RKHSnorm(); }
inline double abs2 (const KNN_Binary &a) { return a.RKHSabs();  }

inline void qswap(KNN_Binary &a, KNN_Binary &b)
{
    a.qswapinternal(b);

    return;
}

inline KNN_Binary &setzero(KNN_Binary &a)
{
    a.restart();

    return a;
}

inline void KNN_Binary::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    KNN_Binary &b = dynamic_cast<KNN_Binary &>(bb.getML());

    KNN_Generic::qswapinternal(b);

    qswap(classlabels,b.classlabels);
    qswap(classcnt   ,b.classcnt   );

    return;
}

inline void KNN_Binary::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const KNN_Binary &b = dynamic_cast<const KNN_Binary &>(bb.getMLconst());

    KNN_Generic::semicopy(b);

    classlabels = b.classlabels;
    classcnt    = b.classcnt;

    return;
}

inline void KNN_Binary::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const KNN_Binary &src = dynamic_cast<const KNN_Binary &>(bb.getMLconst());

    KNN_Generic::assign(src,onlySemiCopy);

    classlabels = src.classlabels;
    classcnt    = src.classcnt;

    return;
}

#endif
