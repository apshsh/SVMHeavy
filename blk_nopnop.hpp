
//
// Do nothing block
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _blk_nopnop_h
#define _blk_nopnop_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "blk_generic.hpp"


// Defines a very basic set of blocks for use in machine learning.


class BLK_Nopnop;


// Swap and zeroing (restarting) functions

inline void qswap(BLK_Nopnop &a, BLK_Nopnop &b);
inline void qswap(BLK_Nopnop *&a, BLK_Nopnop *&b);



class BLK_Nopnop : public BLK_Generic
{
public:

    // Assumptions: all virtual functions inherited from ML_Base are left
    // unchanged in this class.

    // Constructors, destructors, assignment etc..

    BLK_Nopnop(int isIndPrune = 0)                                             : BLK_Generic(isIndPrune) { setaltx(nullptr);                 return; }
    BLK_Nopnop(const BLK_Nopnop &src, int isIndPrune = 0)                      : BLK_Generic(isIndPrune) { setaltx(nullptr); assign(src,0);  return; }
    BLK_Nopnop(const BLK_Nopnop &src, const ML_Base *xsrc, int isIndPrune = 0) : BLK_Generic(isIndPrune) { setaltx(xsrc); assign(src,-1); return; }
    BLK_Nopnop &operator=(const BLK_Nopnop &src) { assign(src); return *this; }
    virtual ~BLK_Nopnop() { return; }

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0) override;
    virtual void semicopy(const ML_Base &src) override;
    virtual void qswapinternal(ML_Base &b) override;

    virtual std::ostream &printstream(std::ostream &output, int dep) const override;
    virtual std::istream &inputstream(std::istream &input ) override;

    virtual int type(void)    const override { return 200; }
    virtual int subtype(void) const override { return 0;   }
};

inline double norm2(const BLK_Nopnop &a);
inline double abs2 (const BLK_Nopnop &a);

inline double norm2(const BLK_Nopnop &a) { return a.RKHSnorm(); }
inline double abs2 (const BLK_Nopnop &a) { return a.RKHSabs();  }

inline void qswap(BLK_Nopnop &a, BLK_Nopnop &b)
{
    a.qswapinternal(b);

    return;
}

inline void qswap(BLK_Nopnop *&a, BLK_Nopnop *&b)
{
    BLK_Nopnop *temp;

    temp = a;
    a = b;
    b = temp;

    return;
}

inline void BLK_Nopnop::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    BLK_Nopnop &b = dynamic_cast<BLK_Nopnop &>(bb.getML());

    BLK_Generic::qswapinternal(b);

    return;
}

inline void BLK_Nopnop::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const BLK_Nopnop &b = dynamic_cast<const BLK_Nopnop &>(bb.getMLconst());

    BLK_Generic::semicopy(b);

    return;
}

inline void BLK_Nopnop::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const BLK_Nopnop &src = dynamic_cast<const BLK_Nopnop &>(bb.getMLconst());

    BLK_Generic::assign(src,onlySemiCopy);

    return;
}

#endif
