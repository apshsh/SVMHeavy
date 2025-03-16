
//
// Binary Classification GP (by EP)
//
// Version: 7
// Date: 18/12/2019
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _gpr_binary_h
#define _gpr_binary_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "gpr_scalar.hpp"








class GPR_Binary;


// Swap function

inline void qswap(GPR_Binary &a, GPR_Binary &b);


class GPR_Binary : public GPR_Scalar
{
public:

    GPR_Binary();
    GPR_Binary(const GPR_Binary &src);
    GPR_Binary(const GPR_Binary &src, const ML_Base *xsrc);
    GPR_Binary &operator=(const GPR_Binary &src) { assign(src); return *this; }
    virtual ~GPR_Binary();

    virtual int prealloc(int expectedN) override;
    virtual int preallocsize(void) const override;

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0) override;
    virtual void semicopy(const ML_Base &src) override;
    virtual void qswapinternal(ML_Base &b) override;

    virtual std::ostream &printstream(std::ostream &output, int dep) const override;
    virtual std::istream &inputstream(std::istream &input) override;

    // Information:

    virtual int tspaceDim(void)  const override { return 1;   }
    virtual int numClasses(void) const override { return 2;   }
    virtual int type(void)       const override { return 409; }
    virtual int subtype(void)    const override { return 0;   }

    virtual char gOutType(void) const override { return 'R'; }
    virtual char hOutType(void) const override { return 'Z'; }
    virtual char targType(void) const override { return 'Z'; }
    virtual double calcDist(const gentype &ha, const gentype &hb, int ia = -1, int db = 2) const override;

    virtual int numInternalClasses(void) const override { return 2; }

    virtual const Vector<gentype> &y(void) const override { return bintraintarg;  }

    virtual int isClassifier(void) const override { return 1; }
    virtual int isRegression(void) const override { return 0; }

    // Modification:

    virtual int addTrainingVector (int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1) override;
    virtual int qaddTrainingVector(int i, const gentype &y,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1) override;

    virtual int removeTrainingVector(int i                                      ) override { SparseVector<gentype> x; gentype y; return removeTrainingVector(i,y,x); }
    virtual int removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x) override;
    virtual int removeTrainingVector(int i, int num                             ) override { return GPR_Generic::removeTrainingVector(i,num); }

    virtual int sety(int                i, const gentype         &nv) override { NiceAssert( nv.isCastableToIntegerWithoutLoss() ); NiceAssert( (int) nv >= -1 ); NiceAssert( (int) nv <= +1 ); return setd(i,(int) nv); }
    virtual int sety(const Vector<int> &i, const Vector<gentype> &nv) override;
    virtual int sety(                      const Vector<gentype> &nv) override;

    virtual int sety(int                i, double                nv) override { NiceAssert( nv == (int) nv ); NiceAssert( (int) nv >= -1 ); NiceAssert( (int) nv <= +1 ); return setd(i,(int) nv); }
    virtual int sety(const Vector<int> &i, const Vector<double> &nv) override;
    virtual int sety(                      const Vector<double> &nv) override;

    virtual int setd(int                i, int                nd) override;
    virtual int setd(const Vector<int> &i, const Vector<int> &nd) override;
    virtual int setd(                      const Vector<int> &nd) override;

    virtual int restart(void) override { GPR_Binary temp; *this = temp; return 1; }

    // Evaluation:

    virtual int hhTrainingVector(gentype &resh,                int i,                  gentype ***pxyprodi = nullptr) const override { gentype resg; return ghTrainingVector(resh,resg,i,0,      pxyprodi); }
    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = nullptr) const override;

    virtual int hh(gentype &resh,                const SparseVector<gentype> &x,                  const vecInfo *xinf = nullptr, gentype ***pxyprodx = nullptr) const override { gentype resg; return gh(resh,resg,x,0,xinf,pxyprodx); }
    virtual int gh(gentype &resh, gentype &resg, const SparseVector<gentype> &x, int retaltg = 0, const vecInfo *xinf = nullptr, gentype ***pxyprodx = nullptr) const override;



private:

    Vector<gentype> bintraintarg;
};

inline double norm2(const GPR_Binary &a);
inline double abs2 (const GPR_Binary &a);

inline double norm2(const GPR_Binary &a) { return a.RKHSnorm(); }
inline double abs2 (const GPR_Binary &a) { return a.RKHSabs();  }

inline void qswap(GPR_Binary &a, GPR_Binary &b)
{
    a.qswapinternal(b);

    return;
}

inline void GPR_Binary::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    GPR_Binary &b = dynamic_cast<GPR_Binary &>(bb.getML());

    GPR_Scalar::qswapinternal(b);

    qswap(bintraintarg,b.bintraintarg);

    return;
}

inline void GPR_Binary::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const GPR_Binary &b = dynamic_cast<const GPR_Binary &>(bb.getMLconst());

    GPR_Scalar::semicopy(b);

    bintraintarg = b.bintraintarg;

    return;
}

inline void GPR_Binary::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const GPR_Binary &src = dynamic_cast<const GPR_Binary &>(bb.getMLconst());

    GPR_Scalar::assign(static_cast<const GPR_Scalar &>(src),onlySemiCopy);

    bintraintarg = src.bintraintarg;

    return;
}

#endif
