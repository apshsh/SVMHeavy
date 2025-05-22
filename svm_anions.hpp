
//
// Anionic regression SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _svm_anions_h
#define _svm_anions_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "svm_vector.hpp"











class SVM_Anions;

void AtoV(const d_anion &src, Vector<double>  &dest);
void AtoV(const d_anion &src, Vector<gentype> &dest);
void VtoA(const Vector<double> &src,  d_anion &dest);
void VtoA(const Vector<gentype> &src, d_anion &dest);


// Swap function

inline void qswap(SVM_Anions &a, SVM_Anions &b);


class SVM_Anions : public SVM_Vector
{
public:

    // Constructors, destructors, assignment operators and similar

    SVM_Anions();
    SVM_Anions(const SVM_Anions &src);
    SVM_Anions(const SVM_Anions &src, const ML_Base *xsrc);
    SVM_Anions &operator=(const SVM_Anions &src) { assign(src); return *this; }
    virtual ~SVM_Anions();

    virtual int prealloc(int expectedN) override;
    virtual int preallocsize(void) const override;

    virtual int scale(double a) override;
    virtual int reset(void) override;
    virtual int restart(void) override { SVM_Anions temp; *this = temp; return 1; }

    virtual int setAlphaA(const Vector<d_anion> &newAlpha) override;
    virtual int setBiasA(const d_anion &newBias) override;

    virtual std::ostream &printstream(std::ostream &output, int dep) const override;
    virtual std::istream &inputstream(std::istream &input ) override;

    // Information:

    virtual int type(void)    const override { return 5; }
    virtual int subtype(void) const override { return 0; }

    virtual char gOutType(void) const override { return 'A'; }
    virtual char hOutType(void) const override { return 'A'; }
    virtual char targType(void) const override { return 'A'; }
    virtual double calcDist(const gentype &ha, const gentype &hb, int ia = -1, int db = 2) const override;

    virtual int isUnderlyingScalar(void) const override { return 0; }
    virtual int isUnderlyingVector(void) const override { return 0; }
    virtual int isUnderlyingAnions(void) const override { return 1; }

    virtual const Vector<d_anion>         &zA(void)     const override { return traintarg;    }
    virtual const d_anion                 &biasA(void)  const override { return db;           }
    virtual const Vector<d_anion>         &alphaA(void) const override { return dalpha;       }
    virtual const Vector<gentype>         &y(void)      const override { return traintarggen; }
    virtual const Vector<double>          &yR(void)     const override { static thread_local Vector<double> dummy; NiceThrow("yR not defined in svm_anions"); return dummy;    }
    virtual const Vector<d_anion>         &yA(void)     const override { return traintarg;    }
    virtual const Vector<Vector<double> > &yV(void)     const override { static thread_local Vector<Vector<double> > dummy; NiceThrow("yV not defined in svm_anions");  return dummy;    }
    virtual const Vector<gentype>         &yp(void)     const override { static thread_local Vector<gentype>         dummy; NiceThrow("yp not defined in svm_anions");  return dummy;    }
    virtual const Vector<double>          &ypR(void)    const override { static thread_local Vector<double>          dummy; NiceThrow("ypR not defined in svm_anions"); return dummy;    }
    virtual const Vector<d_anion>         &ypA(void)    const override { static thread_local Vector<d_anion>         dummy; NiceThrow("ypA not defined in svm_anions"); return dummy;    }
    virtual const Vector<Vector<double> > &ypV(void)    const override { static thread_local Vector<Vector<double> > dummy; NiceThrow("ypV not defined in svm_anions"); return dummy;    }

    virtual const gentype &y(int i)  const override { return traintarggen(i); }
    virtual const d_anion &zA(int i) const override { return traintarg(i);    }
    //virtual const d_anion &zA(int i) const override { if ( i >= 0 ) { return zA()(i); } static d_anion badnotthreadsafe; VtoA((const Vector<double> &) y(i),badnotthreadsafe); return badnotthreadsafe; } // Never used afaict, so not too worreed about the thread-safe/multiple-simultaneous/caller issues

    virtual int isClassifier(void) const override { return 0; }

    // Modification:

    virtual int setLinearCost(void) override;
    virtual int setQuadraticCost(void) override;

    virtual int setC(double xC) override;

    virtual int sety(int                i, const d_anion         &z) override;
    virtual int sety(const Vector<int> &i, const Vector<d_anion> &z) override;
    virtual int sety(                      const Vector<d_anion> &z) override;

    virtual int settspaceDim(int newdim) override;
    virtual int addtspaceFeat(int i) override;
    virtual int removetspaceFeat(int i) override;
    virtual int setorder(int neword) override;

    // Train the SVM

    virtual int train(int &res, svmvolatile int &killSwitch) override;
    virtual int train(int &res) override { svmvolatile int killSwitch = 0; return train(res,killSwitch); }

    // Training set control:

    virtual int addTrainingVector (int i, const gentype &z, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1) override;
    virtual int qaddTrainingVector(int i, const gentype &z,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1) override;

    virtual int addTrainingVector (int i, const Vector<gentype> &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override;
    virtual int qaddTrainingVector(int i, const Vector<gentype> &z,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override;

    virtual int addTrainingVector( int i, const d_anion &z, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2);
    virtual int qaddTrainingVector(int i, const d_anion &z,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2);

    virtual int addTrainingVector( int i, const Vector<d_anion> &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<int> &d);
    virtual int qaddTrainingVector(int i, const Vector<d_anion> &z,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<int> &d);

    virtual int sety(int                i, const gentype         &z) override;
    virtual int sety(const Vector<int> &i, const Vector<gentype> &z) override;
    virtual int sety(                      const Vector<gentype> &z) override;

    virtual int removeTrainingVector(int i                                      ) override { SparseVector<gentype> x; gentype y; return removeTrainingVector(i,y,x); }
    virtual int removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x) override { traintarg.remove(i); traintarggen.remove(i); return SVM_Vector::removeTrainingVector(i,y,x); }
    virtual int removeTrainingVector(int i, int num)                              override;

    virtual int setd(int i, int d) override;

    virtual int setCweight(int                i, double                xCweight) override;
    virtual int setCweight(const Vector<int> &i, const Vector<double> &xCweight) override;
    virtual int setCweight(                      const Vector<double> &xCweight) override;

    virtual int setCweightfuzz(int                i, double                nw) override;
    virtual int setCweightfuzz(const Vector<int> &i, const Vector<double> &nw) override;
    virtual int setCweightfuzz(                      const Vector<double> &nw) override;

    virtual int scaleCweight(double scalefactor) override;
    virtual int scaleCweightfuzz(double scalefactor) override;

    virtual int randomise(double sparsity) override;

    // Evaluation:

    virtual int isVarDefined(void) const override { return 0; }

    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = nullptr) const override;

    virtual double eTrainingVector(int i) const override { return SVM_Vector::eTrainingVector(i); }

    virtual double         &dedgTrainingVector(double         &res, int i) const override { d_anion tmp; return ( res = dedgTrainingVector(tmp,i).realpart() );                                }
    virtual Vector<double> &dedgTrainingVector(Vector<double> &res, int i) const override { return SVM_Vector::dedgTrainingVector(res,i);                                                      }
    virtual d_anion        &dedgTrainingVector(d_anion        &res, int i) const override { Vector<double> tmp(tspaceDim()); SVM_Vector::dedgTrainingVector(tmp,i); VtoA(tmp,res); return res; }
    virtual gentype        &dedgTrainingVector(gentype        &res, int i) const override { gentype tmp; return ( res = dedgTrainingVector(tmp.force_anion(),i).realpart() );                  }

    virtual double &d2edg2TrainingVector(double &res, int i) const override { return ML_Base::d2edg2TrainingVector(res,i); }

    virtual double dedKTrainingVector(int i, int j) const override { return SVM_Vector::dedKTrainingVector(i,j); }
    virtual Vector<double> &dedKTrainingVector(Vector<double> &res, int i) const override { return SVM_Vector::dedKTrainingVector(res,i); }
    virtual Matrix<double> &dedKTrainingVector(Matrix<double> &res) const override { return SVM_Vector::dedKTrainingVector(res); }

    // Other functions

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0) override;
    virtual void semicopy(const ML_Base &src) override;
    virtual void qswapinternal(ML_Base &b) override;

private:

    Vector<d_anion> dalpha;
    d_anion db;
    Vector<d_anion> traintarg;
    Vector<gentype> traintarggen;

    void grabalpha(void);
    void grabdb(void);
    void grabtraintarg(void);

    // Blocked functions

    virtual int setAlphaV(const Vector<Vector<double> > &newAlpha) override;
    virtual int setBiasV(const Vector<double> &newBias) override;

    virtual int sety(int i, const Vector<double> &z) override;
    virtual int sety(const Vector<int> &i, const Vector<Vector<double> > &z) override;
    virtual int sety(const Vector<Vector<double> > &z) override;

    virtual int addTrainingVector( int i, const Vector<double> &z, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2) override;
    virtual int qaddTrainingVector(int i, const Vector<double> &z,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2) override;
};

inline double norm2(const SVM_Anions &a);
inline double abs2 (const SVM_Anions &a);

inline double norm2(const SVM_Anions &a) { return a.RKHSnorm(); }
inline double abs2 (const SVM_Anions &a) { return a.RKHSabs();  }

inline void qswap(SVM_Anions &a, SVM_Anions &b)
{
    a.qswapinternal(b);

    return;
}

inline void SVM_Anions::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    SVM_Anions &b = dynamic_cast<SVM_Anions &>(bb.getML());

    SVM_Vector::qswapinternal(b);

    qswap(dalpha,      b.dalpha      );
    qswap(db,          b.db          );
    qswap(traintarg,   b.traintarg   );
    qswap(traintarggen,b.traintarggen);

    return;
}

inline void SVM_Anions::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const SVM_Anions &b = dynamic_cast<const SVM_Anions &>(bb.getMLconst());

    SVM_Vector::semicopy(b);

    traintarg    = b.traintarg;
    traintarggen = b.traintarggen;

    dalpha = b.dalpha;
    db     = b.db;

    return;
}

inline void SVM_Anions::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const SVM_Anions &src = dynamic_cast<const SVM_Anions &>(bb.getMLconst());

    SVM_Vector::assign(src,onlySemiCopy);

    dalpha       = src.dalpha;
    db           = src.db;
    traintarg    = src.traintarg;
    traintarggen = src.traintarggen;

    return;
}


#endif
