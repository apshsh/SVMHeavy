
//
// k-nearest-neighbour multi-class classifier
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _knn_multic_h
#define _knn_multic_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "knn_generic.hpp"
#include "idstore.hpp"



class KNN_MultiC;


// Swap and zeroing (restarting) functions

inline void qswap(KNN_MultiC &a, KNN_MultiC &b);
inline KNN_MultiC &setzero(KNN_MultiC &a);

class KNN_MultiC : public KNN_Generic
{
public:

    // Constructors, destructors, assignment etc..

    KNN_MultiC();
    KNN_MultiC(const KNN_MultiC &src);
    KNN_MultiC(const KNN_MultiC &src, const ML_Base *xsrc);
    KNN_MultiC &operator=(const KNN_MultiC &src) { assign(src); return *this; }
    virtual ~KNN_MultiC();

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0) override;
    virtual void semicopy(const ML_Base &src) override;
    virtual void qswapinternal(ML_Base &b) override;

    virtual std::ostream &printstream(std::ostream &output, int dep) const override;
    virtual std::istream &inputstream(std::istream &input ) override;

    // Information functions (training data):

    virtual int NNC(int d)    const override { return d ? Nnc(label_placeholder.findID(d)+1) : Nnc(0); }
    virtual int type(void)    const override { return 307;                                             }
    virtual int subtype(void) const override { return 0;                                               }

    virtual int tspaceDim(void)  const override { return 1;                        }
    virtual int numClasses(void) const override { return label_placeholder.size(); }

    virtual char gOutType(void) const override { return 'R'; }
    virtual char hOutType(void) const override { return 'Z'; }
    virtual char targType(void) const override { return 'Z'; }
    virtual double calcDist(const gentype &ha, const gentype &hb, int ia = -1, int db = 2) const override;

    virtual const Vector<int> &ClassLabels(void)   const override { return label_placeholder.getreftoID(); }
//    virtual int                findID(int ref)     const override { return label_placeholder.findID(ref);  }

    virtual int getInternalClass(const gentype &y) const override { return label_placeholder.findID((int) y); }
    virtual int numInternalClasses(void)           const override { return numClasses();                      }

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

    virtual int addclass(int label, int epszero = 0) override;

    // Randomisation: for KNN type, randomisation occurs in the target, as
    // this is the only variable available

    virtual int randomise(double sparsity) override;

private:

    IDStore label_placeholder;
    Vector<int> Nnc;

    virtual void hfn(gentype &res, const Vector<gentype> &yk, const Vector<double> &kdistsq, const Vector<double> &weights, int Nnz, int effkay) const override;
};

inline double norm2(const KNN_MultiC &a);
inline double abs2 (const KNN_MultiC &a);

inline double norm2(const KNN_MultiC &a) { return a.RKHSnorm(); }
inline double abs2 (const KNN_MultiC &a) { return a.RKHSabs();  }

inline void qswap(KNN_MultiC &a, KNN_MultiC &b)
{
    a.qswapinternal(b);

    return;
}

inline KNN_MultiC &setzero(KNN_MultiC &a)
{
    a.restart();

    return a;
}

inline void KNN_MultiC::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    KNN_MultiC &b = dynamic_cast<KNN_MultiC &>(bb.getML());

    KNN_Generic::qswapinternal(b);

    qswap(label_placeholder,b.label_placeholder);
    qswap(Nnc              ,b.Nnc              );

    return;
}

inline void KNN_MultiC::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const KNN_MultiC &b = dynamic_cast<const KNN_MultiC &>(bb.getMLconst());

    KNN_Generic::semicopy(b);

    label_placeholder = b.label_placeholder;
    Nnc               = b.Nnc;

    return;
}

inline void KNN_MultiC::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const KNN_MultiC &src = dynamic_cast<const KNN_MultiC &>(bb.getMLconst());

    KNN_Generic::assign(src,onlySemiCopy);

    label_placeholder = src.label_placeholder;
    Nnc               = src.Nnc;

    return;
}

#endif
