
//
// Test function access block
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _blk_testfn_h
#define _blk_testfn_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "blk_generic.hpp"


// Defines a very basic set of blocks for use in machine learning.


class BLK_Testfn;


std::ostream &operator<<(std::ostream &output, const BLK_Testfn &src );
std::istream &operator>>(std::istream &input,        BLK_Testfn &dest);

// Swap and zeroing (restarting) functions

inline void qswap(BLK_Testfn &a, BLK_Testfn &b);


class BLK_Testfn : public BLK_Generic
{
public:

    // Assumptions: all virtual functions inherited from ML_Base are left
    // unchanged in this class.

    // Constructors, destructors, assignment etc..

    BLK_Testfn(int isIndPrune = 0);
    BLK_Testfn(const BLK_Testfn &src, int isIndPrune = 0)                      : BLK_Generic(isIndPrune) { localygood = 0; setaltx(nullptr); assign(src,0);  return; }
    BLK_Testfn(const BLK_Testfn &src, const ML_Base *xsrc, int isIndPrune = 0) : BLK_Generic(isIndPrune) { localygood = 0; setaltx(xsrc);    assign(src,-1); return; }
    BLK_Testfn &operator=(const BLK_Testfn &src) { assign(src); return *this; }
    virtual ~BLK_Testfn() { return; }

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0) override;
    virtual void semicopy(const ML_Base &src) override;
    virtual void qswapinternal(ML_Base &b) override;

    virtual std::ostream &printstream(std::ostream &output) const override;
    virtual std::istream &inputstream(std::istream &input ) override;

    // Information functions

    virtual int type(void)    const override { return 216; }
    virtual int subtype(void) const override { return 0;   }

    virtual int tspaceDim(void)  const override { return 1; }
    virtual int numClasses(void) const override { return 1; }

    virtual char gOutType(void) const override { return testFnType() ? 'V' : 'R'; }
    virtual char hOutType(void) const override { return testFnType() ? 'V' : 'R'; }
    virtual char targType(void) const override { return testFnType() ? 'V' : 'R'; }
    virtual double calcDist(const gentype &ha, const gentype &hb, int ia = -1, int db = 2) const override { (void) ia; return db ? ( ( (double) ha ) - ( (double) hb ) )*( ( (double) ha ) - ( (double) hb ) ) : 0; }

    virtual int isClassifier(void) const override { return 0; }
    virtual int isRegression(void) const override { return 1; }

    // Evaluation Functions:

    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = nullptr) const override;

    // Sampling stuff

    virtual int setSampleMode(int nv, const Vector<gentype> &xmin, const Vector<gentype> &xmax, int Nsamp, int sampSplit, int sampType, int xsampType, double sampScale, double sampSlack = 0) override
    {
        localygood    = 0;
        locsampleMode = nv;
        locxmin       = xmin;
        locxmax       = xmax;
        locxsampType  = xsampType;
        locNsamp      = Nsamp;
        locsampType   = sampType;
        locsampScale  = sampScale;
        locsampSlack  = sampSlack;

        return BLK_Generic::setSampleMode(nv,xmin,xmax,Nsamp,sampSplit,sampType,xsampType,sampScale,sampSlack);
    }

    // This is really only used in one place - see globalopt.h

    virtual const Vector<gentype> &y(void) const override;

    // Trips for y reset

    virtual int setTestFnType (int                   nv) override { localygood = 0; return BLK_Generic::setTestFnType (nv); }
    virtual int setTestFnNum  (int                   nv) override { localygood = 0; return BLK_Generic::setTestFnNum  (nv); }
    virtual int setTestFnA    (const Matrix<double> &nv) override { localygood = 0; return BLK_Generic::setTestFnA    (nv); }
    virtual int setTestFnAlpha(double                nv) override { localygood = 0; return BLK_Generic::setTestFnAlpha(nv); }

private:

    // For speed, these emulate blk_conect stuff

    mutable Vector<gentype> localy;
    mutable int localygood; // 0 not good, 1 good, -1 individual components good, sum bad

    // Need these for getting "y" (which is sample data, ymmv) of mixed models

    int locsampleMode;
    Vector<gentype> locxmin;
    Vector<gentype> locxmax;
    int locNsamp;
    int locsampType;
    int locsampScale;
    int locsampSlack;

};

inline double norm2(const BLK_Testfn &a);
inline double abs2 (const BLK_Testfn &a);

inline double norm2(const BLK_Testfn &a) { return a.RKHSnorm(); }
inline double abs2 (const BLK_Testfn &a) { return a.RKHSabs();  }

inline void qswap(BLK_Testfn &a, BLK_Testfn &b)
{
    a.qswapinternal(b);

    return;
}

inline void BLK_Testfn::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    BLK_Testfn &b = dynamic_cast<BLK_Testfn &>(bb.getML());

    qswap(localy       ,b.localy       );
    qswap(localygood   ,b.localygood   );
    qswap(locsampleMode,b.locsampleMode);
    qswap(locxmin      ,b.locxmin      );
    qswap(locxmax      ,b.locxmax      );
    qswap(locNsamp     ,b.locNsamp     );
    qswap(locsampType  ,b.locsampType  );
    qswap(locsampScale ,b.locsampScale );
    qswap(locsampSlack ,b.locsampSlack );

    BLK_Generic::qswapinternal(b);

    return;
}

inline void BLK_Testfn::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const BLK_Testfn &b = dynamic_cast<const BLK_Testfn &>(bb.getMLconst());

    BLK_Generic::semicopy(b);

    localy        = b.localy;
    localygood    = b.localygood;
    locsampleMode = b.locsampleMode;
    locxmin       = b.locxmin;
    locxmax       = b.locxmax;
    locNsamp      = b.locNsamp;
    locsampType   = b.locsampType;
    locsampScale  = b.locsampScale;
    locsampSlack  = b.locsampSlack;

    return;
}

inline void BLK_Testfn::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const BLK_Testfn &src = dynamic_cast<const BLK_Testfn &>(bb.getMLconst());

    BLK_Generic::assign(src,onlySemiCopy);

    localy        = src.localy;
    localygood    = src.localygood;
    locsampleMode = src.locsampleMode;
    locxmin       = src.locxmin;
    locxmax       = src.locxmax;
    locNsamp      = src.locNsamp;
    locsampType   = src.locsampType;
    locsampScale  = src.locsampScale;
    locsampSlack  = src.locsampSlack;

    return;
}

#endif
