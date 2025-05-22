
//
// k-nearest-neighbour anionic regressor
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _knn_anions_h
#define _knn_anions_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "knn_generic.hpp"



class KNN_Anions;


// Swap and zeroing (restarting) functions

inline void qswap(KNN_Anions &a, KNN_Anions &b);
inline KNN_Anions &setzero(KNN_Anions &a);

class KNN_Anions : public KNN_Generic
{
public:

    // Constructors, destructors, assignment etc..

    KNN_Anions();
    KNN_Anions(const KNN_Anions &src);
    KNN_Anions(const KNN_Anions &src, const ML_Base *xsrc);
    KNN_Anions &operator=(const KNN_Anions &src) { assign(src); return *this; }
    virtual ~KNN_Anions();

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0) override;
    virtual void semicopy(const ML_Base &src) override;
    virtual void qswapinternal(ML_Base &b) override;

    virtual std::ostream &printstream(std::ostream &output, int dep) const override;
    virtual std::istream &inputstream(std::istream &input ) override;

    // Information functions (training data):

    virtual int NNC(int d)    const override { return classcnt(d/2); }
    virtual int type(void)    const override { return 305;           }
    virtual int subtype(void) const override { return 0;             }

    virtual int tspaceDim(void)  const override { return 1<<order(); }
    virtual int numClasses(void) const override { return 0;          }
    virtual int order(void)      const override { return dorder;     }

    virtual char gOutType(void) const override { return 'A'; }
    virtual char hOutType(void) const override { return 'A'; }
    virtual char targType(void) const override { return 'A'; }
    virtual double calcDist(const gentype &ha, const gentype &hb, int ia = -1, int db = 2) const override;

    virtual const Vector<int> &ClassLabels(void)   const override { return classlabels; }
    virtual int getInternalClass(const gentype &y) const override { (void) y; return 0; }

    virtual int isUnderlyingScalar(void) const override { return 0; }
    virtual int isUnderlyingVector(void) const override { return 0; }
    virtual int isUnderlyingAnions(void) const override { return 1; }

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

    // Fast version of g(x)

    virtual int ggTrainingVector(d_anion &resg, int i, int retaltg = 0, gentype ***pxyprodi = nullptr) const override { return ggTrainingVectorInt(resg,i,retaltg,pxyprodi); }

    // Randomisation: for KNN type, randomisation occurs in the target, as
    // this is the only variable available

    virtual int randomise(double sparsity) override;

    virtual int settspaceDim(int newdim) override;
    virtual int addtspaceFeat(int i) override;
    virtual int removetspaceFeat(int i) override;
    virtual int setorder(int neword) override;

private:

    Vector<int> classlabels;
    Vector<int> classcnt;

    // Order reflects the largest order in the training data.

    int dorder;

    virtual void hfn(gentype &res, const Vector<gentype> &yk, const Vector<double> &kdistsq, const Vector<double> &weights, int Nnz, int effkay) const override;
    virtual void hfn(d_anion &res, const Vector<d_anion> &yk, const Vector<double> &kdistsq, const Vector<double> &weights, int Nnz, int effkay) const override;
};

inline double norm2(const KNN_Anions &a);
inline double abs2 (const KNN_Anions &a);

inline double norm2(const KNN_Anions &a) { return a.RKHSnorm(); }
inline double abs2 (const KNN_Anions &a) { return a.RKHSabs();  }

inline void qswap(KNN_Anions &a, KNN_Anions &b)
{
    a.qswapinternal(b);

    return;
}

inline KNN_Anions &setzero(KNN_Anions &a)
{
    a.restart();

    return a;
}

inline void KNN_Anions::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    KNN_Anions &b = dynamic_cast<KNN_Anions &>(bb.getML());

    KNN_Generic::qswapinternal(b);

    qswap(classlabels,b.classlabels);
    qswap(classcnt   ,b.classcnt   );
    qswap(dorder     ,b.dorder     );

    return;
}

inline void KNN_Anions::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const KNN_Anions &b = dynamic_cast<const KNN_Anions &>(bb.getMLconst());

    KNN_Generic::semicopy(b);

    classlabels = b.classlabels;
    classcnt    = b.classcnt;
    dorder      = b.dorder;

    return;
}

inline void KNN_Anions::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const KNN_Anions &src = dynamic_cast<const KNN_Anions &>(bb.getMLconst());

    KNN_Generic::assign(src,onlySemiCopy);

    classlabels = src.classlabels;
    classcnt    = src.classcnt;
    dorder      = src.dorder;

    return;
}


#endif
