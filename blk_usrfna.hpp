
//
// Simple function block
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _blk_usrfna_h
#define _blk_usrfna_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "blk_generic.hpp"


// Defines a very basic set of blocks for use in machine learning.
//
// g_i(x) = fn(x_i)
// sampling: applies finalisation to fn


class BLK_UsrFnA;


// Swap and zeroing (restarting) functions

inline void qswap(BLK_UsrFnA &a, BLK_UsrFnA &b);


class BLK_UsrFnA : public BLK_Generic
{
public:

    // Assumptions: all virtual functions inherited from ML_Base are left
    // unchanged in this class.

    // Constructors, destructors, assignment etc..

    BLK_UsrFnA(int xisIndPrune = 0)                                             : BLK_Generic(xisIndPrune) { setaltx(nullptr);                 return; }
    BLK_UsrFnA(const BLK_UsrFnA &src, int xisIndPrune = 0)                      : BLK_Generic(xisIndPrune) { setaltx(nullptr); assign(src,0);  return; }
    BLK_UsrFnA(const BLK_UsrFnA &src, const ML_Base *xsrc, int xisIndPrune = 0) : BLK_Generic(xisIndPrune) { setaltx(xsrc); assign(src,-1); return; }
    BLK_UsrFnA &operator=(const BLK_UsrFnA &src) { assign(src); return *this; }
    virtual ~BLK_UsrFnA() { return; }

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0) override;
    virtual void semicopy(const ML_Base &src) override;
    virtual void qswapinternal(ML_Base &b) override;

    virtual std::ostream &printstream(std::ostream &output, int dep) const override;
    virtual std::istream &inputstream(std::istream &input ) override;

    virtual int type(void)    const override { return 203; }
    virtual int subtype(void) const override { return 0;   }

    // Evaluation Functions:
    //
    // Output g(x) is the input, each element a processed form of the input.
    // Output h(x) is the input, each element a processed form of the input.

    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = nullptr) const override;
};

inline double norm2(const BLK_UsrFnA &a);
inline double abs2 (const BLK_UsrFnA &a);

inline double norm2(const BLK_UsrFnA &a) { return a.RKHSnorm(); }
inline double abs2 (const BLK_UsrFnA &a) { return a.RKHSabs();  }

inline void qswap(BLK_UsrFnA &a, BLK_UsrFnA &b)
{
    a.qswapinternal(b);

    return;
}

inline void BLK_UsrFnA::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    BLK_UsrFnA &b = dynamic_cast<BLK_UsrFnA &>(bb.getML());

    BLK_Generic::qswapinternal(b);

    return;
}

inline void BLK_UsrFnA::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const BLK_UsrFnA &b = dynamic_cast<const BLK_UsrFnA &>(bb.getMLconst());

    BLK_Generic::semicopy(b);

    return;
}

inline void BLK_UsrFnA::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const BLK_UsrFnA &src = dynamic_cast<const BLK_UsrFnA &>(bb.getMLconst());

    BLK_Generic::assign(src,onlySemiCopy);

    return;
}

#endif
