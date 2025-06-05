
//
// Bernstein polynomial block
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _blk_bernst_h
#define _blk_bernst_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "blk_generic.hpp"


// Defines a very basic set of blocks for use in machine learning.


class BLK_Bernst;


// Swap and zeroing (restarting) functions

inline void qswap(BLK_Bernst &a, BLK_Bernst &b);


class BLK_Bernst : public BLK_Generic
{
public:

    // Assumptions: all virtual functions inherited from ML_Base are left
    // unchanged in this class.

    // Constructors, destructors, assignment etc..

    BLK_Bernst(int isIndPrune = 0);
    BLK_Bernst(const BLK_Bernst &src, int isIndPrune = 0)                      : BLK_Generic(isIndPrune) { localygood = 0; setaltx(nullptr); assign(src,0);  return; }
    BLK_Bernst(const BLK_Bernst &src, const ML_Base *xsrc, int isIndPrune = 0) : BLK_Generic(isIndPrune) { localygood = 0; setaltx(xsrc); assign(src,-1); return; }
    BLK_Bernst &operator=(const BLK_Bernst &src) { assign(src); return *this; }
    virtual ~BLK_Bernst() { return; }

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0);
    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);

    virtual std::ostream &printstream(std::ostream &output, int dep) const;
    virtual std::istream &inputstream(std::istream &input );

    // Information functions

    virtual int type   (void) const override { return 215; }
    virtual int subtype(void) const override { return 0;   }

    virtual int tspaceDim (void) const override { return 1; }
    virtual int numClasses(void) const override { return 1; }

    virtual char gOutType(void) const override { return 'R'; }
    virtual char hOutType(void) const override { return 'R'; }
    virtual char targType(void) const override { return 'R'; }
    virtual double calcDist(const gentype &ha, const gentype &hb, int ia = -1, int db = 2) const override { (void) ia; return db ? ( ( (double) ha ) - ( (double) hb ) )*( ( (double) ha ) - ( (double) hb ) ) : 0; }

    virtual int isClassifier(void) const override { return 0; }
    virtual int isRegression(void) const override { return 1; }

    // Evaluation Functions:

    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = nullptr) const;

    // Sampling stuff

    virtual int setSampleMode(int nv, const Vector<gentype> &xmin, const Vector<gentype> &xmax, int Nsamp, int sampSplit, int sampType, int xsampType, double sampScale, double sampSlack = 0) override
    {
        localygood    = 0;
        locsampleMode = nv;
        locxmin       = xmin;
        locxmax       = xmax;
        locxsampType  = xsampType;
        locNsamp      = Nsamp;
        locsampSplit  = sampSplit;
        locsampType   = sampType;
        locsampScale  = sampScale;
        locsampSlack  = sampSlack;

        return BLK_Generic::setSampleMode(nv,xmin,xmax,Nsamp,sampSplit,sampType,xsampType,sampScale,sampSlack);
    }

    // Trips for y update

    virtual int setBernDegree(const gentype &nv) { localygood = 0; return BLK_Generic::setBernDegree(nv); }
    virtual int setBernIndex (const gentype &nv) { localygood = 0; return BLK_Generic::setBernIndex(nv);  }

    // This is really only used in one place - see globalopt.h

    virtual const Vector<gentype>         &y (void) const override;
    virtual const Vector<double>          &yR(void) const override { static thread_local Vector<double>          dummy; NiceThrow("yR not available in blk_bernst"); return dummy; }
    virtual const Vector<d_anion>         &yA(void) const override { static thread_local Vector<d_anion>         dummy; NiceThrow("yA not available in blk_bernst"); return dummy; }
    virtual const Vector<Vector<double> > &yV(void) const override { static thread_local Vector<Vector<double> > dummy; NiceThrow("yV not available in blk_bernst"); return dummy; }

private:

    // For speed, these emulate blk_conect stuff

    mutable Vector<gentype> localy;
    mutable int localygood; // 0 not good, 1 good, -1 individual components good, sum bad

    // Need these for getting "y" (which is sample data, ymmv) of mixed models

    int locsampleMode;
    Vector<gentype> locxmin;
    Vector<gentype> locxmax;
    int locxsampType;
    int locNsamp;
    int locsampSplit;
    int locsampType;
    double locsampScale;
    double locsampSlack;
};

inline double norm2(const BLK_Bernst &a);
inline double abs2 (const BLK_Bernst &a);

inline double norm2(const BLK_Bernst &a) { return a.RKHSnorm(); }
inline double abs2 (const BLK_Bernst &a) { return a.RKHSabs();  }

inline void qswap(BLK_Bernst &a, BLK_Bernst &b)
{
    a.qswapinternal(b);

    return;
}

inline void BLK_Bernst::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    BLK_Bernst &b = dynamic_cast<BLK_Bernst &>(bb.getML());

    qswap(localy       ,b.localy       );
    qswap(localygood   ,b.localygood   );
    qswap(locsampleMode,b.locsampleMode);
    qswap(locxmin      ,b.locxmin      );
    qswap(locxmax      ,b.locxmax      );
    qswap(locxsampType ,b.locxsampType );
    qswap(locNsamp     ,b.locNsamp     );
    qswap(locsampSplit ,b.locsampSplit );
    qswap(locsampType  ,b.locsampType  );
    qswap(locsampScale ,b.locsampScale );
    qswap(locsampSlack ,b.locsampSlack );

    BLK_Generic::qswapinternal(b);

    return;
}

inline void BLK_Bernst::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const BLK_Bernst &b = dynamic_cast<const BLK_Bernst &>(bb.getMLconst());

    BLK_Generic::semicopy(b);

    localy        = b.localy;
    localygood    = b.localygood;
    locsampleMode = b.locsampleMode;
    locxmin       = b.locxmin;
    locxmax       = b.locxmax;
    locxsampType  = b.locxsampType;
    locNsamp      = b.locNsamp;
    locsampSplit  = b.locsampSplit;
    locsampType   = b.locsampType;
    locsampScale  = b.locsampScale;
    locsampSlack  = b.locsampSlack;

    return;
}

inline void BLK_Bernst::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const BLK_Bernst &src = dynamic_cast<const BLK_Bernst &>(bb.getMLconst());

    BLK_Generic::assign(src,onlySemiCopy);

    localy        = src.localy;
    localygood    = src.localygood;
    locsampleMode = src.locsampleMode;
    locxmin       = src.locxmin;
    locxmax       = src.locxmax;
    locxsampType  = src.locxsampType;
    locNsamp      = src.locNsamp;
    locsampSplit  = src.locsampSplit;
    locsampType   = src.locsampType;
    locsampScale  = src.locsampScale;
    locsampSlack  = src.locsampSlack;

    return;
}

#endif
