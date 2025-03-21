
//
// Super-Sparse SVM scalar class
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _ssv_scalar_h
#define _ssv_scalar_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "ssv_generic.hpp"


class SSV_Scalar;

// Swap and zeroing (restarting) functions

inline void qswap(SSV_Scalar &a, SSV_Scalar &b);

class SSV_Scalar : public SSV_Generic
{
public:

    // Constructors, destructors, assignment etc..

    SSV_Scalar()                                           : SSV_Generic() {                               return;       }
    SSV_Scalar(const SSV_Scalar &src)                      : SSV_Generic() {                assign(src,0); return;       }
    SSV_Scalar(const SSV_Scalar &src, const ML_Base *srcx) : SSV_Generic() { setaltx(srcx); assign(src,0); return;       }
    SSV_Scalar &operator=(const SSV_Scalar &src)                           {                assign(src,0); return *this; }
    virtual ~SSV_Scalar() { return; }

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0) override;
    virtual void semicopy(const ML_Base &src) override;
    virtual void qswapinternal(ML_Base &b) override;

    virtual std::ostream &printstream(std::ostream &output, int dep) const override;
    virtual std::istream &inputstream(std::istream &input) override;

    // Information functions (training data):

    virtual int type(void)      const override { return 700; }
    virtual int subtype(void)   const override { return 0;   }
    virtual int tspaceDim(void) const override { return 1;   }

    virtual char gOutType(void) const override { return 'R'; }
    virtual char hOutType(void) const override { return 'R'; }
    virtual char targType(void) const override { return 'R'; }

    virtual int isUnderlyingScalar(void) const override { return 1; }
    virtual int isUnderlyingVector(void) const override { return 0; }
    virtual int isUnderlyingAnions(void) const override { return 0; }

    // General modification and autoset functions

    virtual int restart(void) override { SSV_Scalar temp; *this = temp; return 1; }

    // Training functions:

    virtual int train(int &res, svmvolatile int &killSwitch) override;
    virtual int train(int &res) override { svmvolatile int killSwitch = 0; return train(res,killSwitch); }

private:

    virtual int entrain(SVM_Scalar &zzmodel, SVM_Scalar &betamodel, int &res, svmvolatile int &killSwitch);
};

inline double norm2(const SSV_Scalar &a);
inline double abs2 (const SSV_Scalar &a);

inline double norm2(const SSV_Scalar &a) { return a.RKHSnorm(); }
inline double abs2 (const SSV_Scalar &a) { return a.RKHSabs();  }

inline void qswap(SSV_Scalar &a, SSV_Scalar &b)
{
    a.qswapinternal(b);

    return;
}

inline void SSV_Scalar::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    SSV_Scalar &b = dynamic_cast<SSV_Scalar &>(bb.getML());

    SSV_Generic::qswapinternal(b);

    return;
}

inline void SSV_Scalar::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const SSV_Scalar &b = dynamic_cast<const SSV_Scalar &>(bb.getMLconst());

    SSV_Generic::semicopy(b);

    return;
}

inline void SSV_Scalar::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const SSV_Scalar &src = dynamic_cast<const SSV_Scalar &>(bb.getMLconst());

    SSV_Generic::assign(src,onlySemiCopy);

    return;
}

#endif
