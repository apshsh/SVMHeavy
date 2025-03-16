
//
// Expected improvement (EHI for multi-objective)
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _imp_expect_h
#define _imp_expect_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "imp_generic.hpp"
#include "hyper_opt.hpp"




class IMP_Expect;


inline void qswap(IMP_Expect &a, IMP_Expect &b);


class IMP_Expect : public IMP_Generic
{
public:

    // Assumptions: all virtual functions inherited from ML_Base are left
    // unchanged in this class.

    // Constructors, destructors, assignment etc..

    IMP_Expect(int isIndPrune = 0);
    IMP_Expect(const IMP_Expect &src, int isIndPrune = 0);
    IMP_Expect(const IMP_Expect &src, const ML_Base *xsrc, int isIndPrune = 0);
    IMP_Expect &operator=(const IMP_Expect &src) { assign(src); return *this; }
    virtual ~IMP_Expect();

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0) override;
    virtual void semicopy(const ML_Base &src) override;
    virtual void qswapinternal(ML_Base &b) override;

    virtual std::ostream &printstream(std::ostream &output, int dep) const override;
    virtual std::istream &inputstream(std::istream &input ) override;

    virtual int type(void) const override { return 600; }

    // Information functions (training data):

    virtual char gOutType(void) const override { return 'R'; }
    virtual char hOutType(void) const override { return 'R'; }

    virtual int isUnderlyingScalar(void) const override { return 1; }
    virtual int isUnderlyingVector(void) const override { return 0; }
    virtual int isUnderlyingAnions(void) const override { return 0; }

    // Training function (pre-calculates min_i(x(i)) for x(i) enabled)

    virtual int train(int &res, svmvolatile int &killSwitch) override;
    virtual int train(int &res) override { svmvolatile int killSwitch = 0; return train(res,killSwitch); }

    // General modification and autoset functions

    virtual int reset(void)   override { untrain(); return 1;                     }
    virtual int restart(void) override { IMP_Expect temp; *this = temp; return 1; }

    // Evaluation Functions:
    //
    // Output g(x) = h(x) is the min(input,min_i(x(i))).
    // Output imp(E(x),var(x)) is expected decrease in g(x)
    //
    //

    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = nullptr) const override;
    virtual int imp(gentype &resi, gentype &resv, const SparseVector<gentype> &xxmean, const gentype &xxvar) const override;

private:

    // xmaxval: if xspaceDim() == 0,1 this stores maximum value of x
    // hc: pre-trained (partial) cache for optimised EHI calculation (or nullptr)
    // X: data (x) negated in alternative format (required for hc) (or nullptr)

    gentype xmaxval;
    hyper_cache *hc;
    double **X;

    virtual void untrain(void) override;
};

inline double norm2(const IMP_Expect &a);
inline double abs2 (const IMP_Expect &a);

inline double norm2(const IMP_Expect &a) { return a.RKHSnorm(); }
inline double abs2 (const IMP_Expect &a) { return a.RKHSabs();  }

inline void qswap(double **&a, double **&b);
inline void qswap(double **&a, double **&b)
{
    double **c;

    c = a;
    a = b;
    b = c;

    return;
}

inline void qswap(IMP_Expect &a, IMP_Expect &b)
{
    a.qswapinternal(b);

    return;
}

inline void IMP_Expect::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    IMP_Expect &b = dynamic_cast<IMP_Expect &>(bb.getML());

    qswap(xmaxval,b.xmaxval);
    qswap(hc     ,b.hc     );
    qswap(X      ,b.X      );

    IMP_Generic::qswapinternal(b);

    return;
}

inline void IMP_Expect::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const IMP_Expect &b = dynamic_cast<const IMP_Expect &>(bb.getMLconst());

    if ( hc )
    {
        untrain();
    }

    xmaxval = b.xmaxval;
    // hc,X must remain nullptr (no copy defined)

    IMP_Generic::semicopy(b);

    return;
}

inline void IMP_Expect::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const IMP_Expect &src = dynamic_cast<const IMP_Expect &>(bb.getMLconst());

    if ( hc )
    {
        untrain();
    }

    xmaxval = src.xmaxval;
    // hc,X must remain nullptr (no copy defined)

    IMP_Generic::assign(src,onlySemiCopy);

    return;
}

#endif
