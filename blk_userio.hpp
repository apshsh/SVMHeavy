
//
// User I/O Function
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _blk_userio_h
#define _blk_userio_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "blk_generic.hpp"


// Basic user I/O


class BLK_UserIO;


// Swap and zeroing (restarting) functions

inline void qswap(BLK_UserIO &a, BLK_UserIO &b);


class BLK_UserIO : public BLK_Generic
{
public:

    // Assumptions: all virtual functions inherited from ML_Base are left
    // unchanged in this class.

    // Constructors, destructors, assignment etc..

    BLK_UserIO(int isIndPrune = 0)                                             : BLK_Generic(isIndPrune) { setaltx(nullptr);                 return; }
    BLK_UserIO(const BLK_UserIO &src, int isIndPrune = 0)                      : BLK_Generic(isIndPrune) { setaltx(nullptr); assign(src,0);  return; }
    BLK_UserIO(const BLK_UserIO &src, const ML_Base *xsrc, int isIndPrune = 0) : BLK_Generic(isIndPrune) { setaltx(xsrc); assign(src,-1); return; }
    BLK_UserIO &operator=(const BLK_UserIO &src) { assign(src); return *this; }
    virtual ~BLK_UserIO() { return; }

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0) override;
    virtual void semicopy(const ML_Base &src) override;
    virtual void qswapinternal(ML_Base &b) override;

    virtual std::ostream &printstream(std::ostream &output, int dep) const override;
    virtual std::istream &inputstream(std::istream &input ) override;

    virtual int type(void)    const override { return 204; }
    virtual int subtype(void) const override { return 0;   }

    // Evaluation Functions:

    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = nullptr) const override;
};

inline double norm2(const BLK_UserIO &a);
inline double abs2 (const BLK_UserIO &a);

inline double norm2(const BLK_UserIO &a) { return a.RKHSnorm(); }
inline double abs2 (const BLK_UserIO &a) { return a.RKHSabs();  }

inline void qswap(BLK_UserIO &a, BLK_UserIO &b)
{
    a.qswapinternal(b);

    return;
}

inline void BLK_UserIO::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    BLK_UserIO &b = dynamic_cast<BLK_UserIO &>(bb.getML());

    BLK_Generic::qswapinternal(b);

    return;
}

inline void BLK_UserIO::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const BLK_UserIO &b = dynamic_cast<const BLK_UserIO &>(bb.getMLconst());

    BLK_Generic::semicopy(b);

    return;
}

inline void BLK_UserIO::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const BLK_UserIO &src = dynamic_cast<const BLK_UserIO &>(bb.getMLconst());

    BLK_Generic::assign(src,onlySemiCopy);

    return;
}

#endif
