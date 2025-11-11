
//
// Simple MEX callback block (elementwise)
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _blk_mexfna_h
#define _blk_mexfna_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "blk_generic.hpp"


// Defines a very basic set of blocks for use in machine learning.


class BLK_MexFnA;


// Swap and zeroing (restarting) functions

inline void qswap(BLK_MexFnA &a, BLK_MexFnA &b);


class BLK_MexFnA : public BLK_Generic
{
public:

    // Assumptions: all virtual functions inherited from ML_Base are left
    // unchanged in this class.

    // Constructors, destructors, assignment etc..

    BLK_MexFnA(int xisIndPrune = 0)                                             : BLK_Generic(xisIndPrune) { setaltx(nullptr);                 return; }
    BLK_MexFnA(const BLK_MexFnA &src, int xisIndPrune = 0)                      : BLK_Generic(xisIndPrune) { setaltx(nullptr); assign(src,0);  return; }
    BLK_MexFnA(const BLK_MexFnA &src, const ML_Base *xsrc, int xisIndPrune = 0) : BLK_Generic(xisIndPrune) { setaltx(xsrc); assign(src,-1); return; }
    BLK_MexFnA &operator=(const BLK_MexFnA &src) { assign(src); return *this; }
    virtual ~BLK_MexFnA() { return; }

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0) override;
    virtual void semicopy(const ML_Base &src) override;
    virtual void qswapinternal(ML_Base &b) override;

    virtual std::ostream &printstream(std::ostream &output, int dep) const override;
    virtual std::istream &inputstream(std::istream &input ) override;

    virtual int type(void)    const override { return 209; }
    virtual int subtype(void) const override { return 0;   }

    // Evaluation Functions:
    //
    // Output g(x) is the input, each element a processed form of the input.
    // Output h(x) is the input, each element a processed form of the input.

    virtual int gh(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = nullptr) const override;
    virtual int gh(gentype &resh, gentype &resg, const SparseVector<gentype> &x, int retaltg = 0, const vecInfo *xinf = nullptr, gentype ***pxyprodx = nullptr) const override { return BLK_Generic::gh(resh,resg,x,retaltg,xinf,pxyprodx); }
};

inline double norm2(const BLK_MexFnA &a);
inline double abs2 (const BLK_MexFnA &a);

inline double norm2(const BLK_MexFnA &a) { return a.RKHSnorm(); }
inline double abs2 (const BLK_MexFnA &a) { return a.RKHSabs();  }

inline void qswap(BLK_MexFnA &a, BLK_MexFnA &b)
{
    a.qswapinternal(b);

    return;
}

inline void BLK_MexFnA::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    BLK_MexFnA &b = dynamic_cast<BLK_MexFnA &>(bb.getML());

    BLK_Generic::qswapinternal(b);

    return;
}

inline void BLK_MexFnA::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const BLK_MexFnA &b = dynamic_cast<const BLK_MexFnA &>(bb.getMLconst());

    BLK_Generic::semicopy(b);

    return;
}

inline void BLK_MexFnA::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const BLK_MexFnA &src = dynamic_cast<const BLK_MexFnA &>(bb.getMLconst());

    BLK_Generic::assign(src,onlySemiCopy);

    return;
}

#endif
