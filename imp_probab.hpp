
//
// Probability of improvement.
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _imp_probab_h
#define _imp_probab_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "imp_generic.hpp"




class IMP_Probab;


inline void qswap(IMP_Probab &a, IMP_Probab &b);


class IMP_Probab : public IMP_Generic
{
public:

    // Assumptions: all virtual functions inherited from ML_Base are left
    // unchanged in this class.

    // Constructors, destructors, assignment etc..

    IMP_Probab(int isIndPrune = 0);
    IMP_Probab(const IMP_Probab &src, int isIndPrune = 0);
    IMP_Probab(const IMP_Probab &src, const ML_Base *xsrc, int isIndPrune = 0);
    IMP_Probab &operator=(const IMP_Probab &src) { assign(src); return *this; }
    virtual ~IMP_Probab();

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0) override;
    virtual void semicopy(const ML_Base &src) override;
    virtual void qswapinternal(ML_Base &b) override;

    virtual std::ostream &printstream(std::ostream &output, int dep) const override;
    virtual std::istream &inputstream(std::istream &input ) override;

    virtual int type(void) const override { return 601; }

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
    virtual int restart(void) override { IMP_Probab temp; *this = temp; return 1; }

    // Evaluation Functions:
    //
    // Output g(x) = h(x) is the -1 if input < min_i(x(i)), 0 otherwise
    // Output imp(E(x),var(x)) is probability decrease in g(x).  Note that
    // this result is actually negated so that it fits in with the minimisation
    // theme (so if you minimise this you maximise the PI)

    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = nullptr) const override;
    virtual int imp(gentype &resi, const SparseVector<gentype> &xxmean, const gentype &xxvar) const override;

private:

    gentype xminval;
};

inline double norm2(const IMP_Probab &a);
inline double abs2 (const IMP_Probab &a);

inline double norm2(const IMP_Probab &a) { return a.RKHSnorm(); }
inline double abs2 (const IMP_Probab &a) { return a.RKHSabs();  }

inline void qswap(IMP_Probab &a, IMP_Probab &b)
{
    a.qswapinternal(b);

    return;
}

inline void IMP_Probab::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    IMP_Probab &b = dynamic_cast<IMP_Probab &>(bb.getML());

    qswap(xminval,b.xminval);

    IMP_Generic::qswapinternal(b);

    return;
}

inline void IMP_Probab::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const IMP_Probab &b = dynamic_cast<const IMP_Probab &>(bb.getMLconst());

    xminval = b.xminval;

    IMP_Generic::semicopy(b);

    return;
}

inline void IMP_Probab::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const IMP_Probab &src = dynamic_cast<const IMP_Probab &>(bb.getMLconst());

    xminval = src.xminval;

    IMP_Generic::assign(src,onlySemiCopy);

    return;
}

#endif
