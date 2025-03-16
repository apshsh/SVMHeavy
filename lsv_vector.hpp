
//
// LS-SVM vector class
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _lsv_vector_h
#define _lsv_vector_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "lsv_generic.hpp"
#include "lsv_generic_deref.hpp"
#include "svm_vector.hpp"


class LSV_Vector;

// This class is basically a front built over SVM_Vector,
// where SVM_Vector defaults to reduction to binary with
// Kreal/unreal left open for matrix-valued covariance, if
// desired.

// Swap and zeroing (restarting) functions

inline void qswap(LSV_Vector &a, LSV_Vector &b);
inline LSV_Vector &setzero(LSV_Vector &a);

class LSV_Vector : public LSV_Generic_Deref
{
public:

    LSV_Vector();
    LSV_Vector(const LSV_Vector &src);
    LSV_Vector(const LSV_Vector &src, const ML_Base *srcx);
    LSV_Vector &operator=(const LSV_Vector &src) { assign(src); return *this; }
    virtual ~LSV_Vector();

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0) override;
    virtual void semicopy(const ML_Base &src)                     override;
    virtual void qswapinternal(ML_Base &b)                        override;

    virtual std::ostream &printstream(std::ostream &output, int dep) const override;
    virtual std::istream &inputstream(std::istream &input          )       override;

    virtual       SVM_Generic &getQQQ     (void)       override { return QQQ; }
    virtual const SVM_Generic &getQQQconst(void) const override { return QQQ; }

    // Information functions (training data):

    virtual int type   (void)  const override { return 501; }
    virtual int subtype(void)  const override { return 0;   }

    virtual double eps(void)       const override { return 0.0;         }
    virtual double epsclass(int d) const override { (void) d; return 1; }

    // General modification and autoset functions

    virtual int seteps(double nv) override { (void) nv; return 0; }

    // Training functions:

    virtual void fudgeOn (void) override { return; }
    virtual void fudgeOff(void) override { return; }





    // LSV_Generic

    virtual       LSV_Generic &getLSV     (void)       { return *this; }
    virtual const LSV_Generic &getLSVconst(void) const { return *this; }

    // Constructors, destructors, assignment etc..

    virtual int setgamma(const Vector<gentype> &newgamma) { return getQQQ().setAlpha(newgamma); }
    virtual int setdelta(const gentype         &newdelta) { return getQQQ().setBias(newdelta);  }

    // Additional information

    virtual int isVardelta (void) const { return getQQQconst().isVarBias();   }
    virtual int isZerodelta(void) const { return getQQQconst().isFixedBias(); }

    virtual const Vector<gentype> &gamma(void) const { return getQQQconst().alpha(); }
    virtual const gentype         &delta(void) const { return getQQQconst().bias();  }

    virtual const Matrix<double> &lsvGp(void) const { return getQQQconst().Gp(); }

    // General modification and autoset functions

    virtual int setVardelta (void) { return getQQQ().setVarBias();      }
    virtual int setZerodelta(void) { return getQQQ().setFixedBias(0.0); }

    // Likelihood

    virtual double loglikelihood(void) const { return getQQQconst().loglikelihood(); }
    virtual double maxinfogain  (void) const { return getQQQconst().maxinfogain  (); }
    virtual double RKHSnorm     (void) const { return getQQQconst().RKHSnorm     (); }
    virtual double RKHSabs      (void) const { return getQQQconst().RKHSabs      (); }


private:

    SVM_Vector QQQ;
};

inline double norm2(const LSV_Vector &a);
inline double abs2 (const LSV_Vector &a);

inline double norm2(const LSV_Vector &a) { return a.RKHSnorm(); }
inline double abs2 (const LSV_Vector &a) { return a.RKHSabs();  }

inline void qswap(LSV_Vector &a, LSV_Vector &b)
{
    a.qswapinternal(b);

    return;
}

inline LSV_Vector &setzero(LSV_Vector &a)
{
    a.restart();

    return a;
}

inline void LSV_Vector::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    LSV_Vector &b = dynamic_cast<LSV_Vector &>(bb.getML());

    LSV_Generic::qswapinternal(b);

    qswap(QQQ,b.QQQ);

    return;
}

inline void LSV_Vector::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const LSV_Vector &b = dynamic_cast<const LSV_Vector &>(bb.getMLconst());

    LSV_Generic::semicopy(b);

    QQQ.semicopy(b.QQQ);

    return;
}

inline void LSV_Vector::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const LSV_Vector &src = dynamic_cast<const LSV_Vector &>(bb.getMLconst());

    LSV_Generic::assign(src,onlySemiCopy);

    QQQ.assign(src.QQQ);

    return;
}

#endif
