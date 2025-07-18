
//
// k-nearest-neighbour scalar regression
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _knn_gentyp_h
#define _knn_gentyp_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "knn_generic.hpp"



class KNN_Gentyp;


// Swap and zeroing (restarting) functions

inline void qswap(KNN_Gentyp &a, KNN_Gentyp &b);
inline KNN_Gentyp &setzero(KNN_Gentyp &a);

class KNN_Gentyp : public KNN_Generic
{
public:

    // Constructors, destructors, assignment etc..

    KNN_Gentyp();
    KNN_Gentyp(const KNN_Gentyp &src);
    KNN_Gentyp(const KNN_Gentyp &src, const ML_Base *xsrc);
    KNN_Gentyp &operator=(const KNN_Gentyp &src) { assign(src); return *this; }
    virtual ~KNN_Gentyp();

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0) override;
    virtual void semicopy(const ML_Base &src) override;
    virtual void qswapinternal(ML_Base &b) override;

    virtual std::ostream &printstream(std::ostream &output, int dep) const override;
    virtual std::istream &inputstream(std::istream &input ) override;

    // Information functions (training data):

    virtual int NNC(int d)    const override { return classcnt(d+1); }
    virtual int type(void)    const override { return 302;           }
    virtual int subtype(void) const override { return 0;             }

    virtual int tspaceDim(void)    const override { return 1; }
    virtual int numClasses(void)   const override { return 0; }

    virtual char gOutType(void) const override { return '?'; }
    virtual char hOutType(void) const override { return '?'; }
    virtual char targType(void) const override { return '?'; }
    virtual double calcDist(const gentype &ha, const gentype &hb, int ia = -1, int db = 2) const override;

    virtual const Vector<int> &ClassLabels(void)   const override { return classlabels; }
    virtual int getInternalClass(const gentype &y) const override { (void) y; return 1; }

    virtual int isClassifier(void) const override { return 0; }

    // Training set modification - need to overload to maintain counts

    virtual int addTrainingVector (int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2) override;
    virtual int qaddTrainingVector(int i, const gentype &y,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2) override;

    virtual int addTrainingVector (int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override;
    virtual int qaddTrainingVector(int i, const Vector<gentype> &y,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override;

    virtual int removeTrainingVector(int i                                      ) override { SparseVector<gentype> x; gentype y; return removeTrainingVector(i,y,x); }
    virtual int removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x) override;
    virtual int removeTrainingVector(int i, int num                             ) override { return ML_Base::removeTrainingVector(i,num); }

    // Fast version of g(x)

    virtual int ggTrainingVector(double &resg, int i, int retaltg = 0, gentype ***pxyprodi = nullptr) const override { return ggTrainingVectorInt(resg,i,retaltg,pxyprodi); }

private:

    Vector<int> classlabels;
    Vector<int> classcnt;

    virtual void hfn(gentype &res, const Vector<gentype> &yk, const Vector<double> &kdistsq, const Vector<double> &weights, int Nnz, int effkay) const override;
};

inline double norm2(const KNN_Gentyp &a);
inline double abs2 (const KNN_Gentyp &a);

inline double norm2(const KNN_Gentyp &a) { return a.RKHSnorm(); }
inline double abs2 (const KNN_Gentyp &a) { return a.RKHSabs();  }

inline void qswap(KNN_Gentyp &a, KNN_Gentyp &b)
{
    a.qswapinternal(b);

    return;
}

inline KNN_Gentyp &setzero(KNN_Gentyp &a)
{
    a.restart();

    return a;
}

inline void KNN_Gentyp::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    KNN_Gentyp &b = dynamic_cast<KNN_Gentyp &>(bb.getML());

    KNN_Generic::qswapinternal(b);

    qswap(classlabels,b.classlabels);
    qswap(classcnt   ,b.classcnt   );

    return;
}

inline void KNN_Gentyp::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const KNN_Gentyp &b = dynamic_cast<const KNN_Gentyp &>(bb.getMLconst());

    KNN_Generic::semicopy(b);

    classlabels = b.classlabels;
    classcnt    = b.classcnt;

    return;
}

inline void KNN_Gentyp::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const KNN_Gentyp &src = dynamic_cast<const KNN_Gentyp &>(bb.getMLconst());

    KNN_Generic::assign(src,onlySemiCopy);

    classlabels = src.classlabels;
    classcnt    = src.classcnt;

    return;
}

#endif
