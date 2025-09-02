
//FIXME: need to passthrough all functionality to GPR_Scalar

//
// Random thompson sample form GP scalarisation
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _imp_nlsamp_h
#define _imp_nlsamp_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "imp_generic.hpp"
#include "gpr_scalar.hpp"



class IMP_NLSamp;
class BayesOptions;

inline void qswap(IMP_NLSamp &a, IMP_NLSamp &b);


class IMP_NLSamp : public IMP_Generic
{
    friend class BayesOptions;

public:

    // Assumptions: all virtual functions inherited from ML_Base are left
    // unchanged in this class.

    // Constructors, destructors, assignment etc..

    IMP_NLSamp(int isIndPrune = 0);
    IMP_NLSamp(const IMP_NLSamp &src, int isIndPrune = 0);
    IMP_NLSamp(const IMP_NLSamp &src, const ML_Base *xsrc, int isIndPrune = 0);
    IMP_NLSamp &operator=(const IMP_NLSamp &src) { assign(src); return *this; }
    virtual ~IMP_NLSamp();

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0) override;
    virtual void semicopy(const ML_Base &src) override;
    virtual void qswapinternal(ML_Base &b) override;

    virtual std::ostream &printstream(std::ostream &output, int dep) const override;
    virtual std::istream &inputstream(std::istream &input ) override;

    // Information functions (training data):

    virtual int type   (void) const override { return 603; }
    virtual int subtype(void) const override { return 0;   }

    // Block data addition from bayesopt

    virtual int addTrainingVector (int, const gentype &, const SparseVector<gentype> &, double a = 1, double b = 1, int d = 2) override { (void) a; (void) b; (void) d; return 0; }

    // Training function (pre-calculates min_i(x(i)) for x(i) enabled)

    virtual int train(int &res)                              override { svmvolatile int killSwitch = 0; return train(res,killSwitch); }
    virtual int train(int &res, svmvolatile int &killSwitch) override;

    // General modification and autoset functions

    virtual int reset(void)   override { untrain(); return 1;                     }
    virtual int restart(void) override { IMP_NLSamp temp; *this = temp; return 1; }









    // ================================================================
    //     Common to all IMPs
    // ================================================================

    // Evaluation Functions:

    virtual int imp(gentype &resi, gentype &resv, const SparseVector<gentype> &xxmean, const gentype &xxvar) const override;

    // Base-level stuff
    //
    // This is overloaded by children to return correct Q type

    virtual       ML_Base &getQ(void)            { return randscal ? *randscal : randscaltemplate; }
    virtual const ML_Base &getQconst(void) const { return randscal ? *randscal : randscaltemplate; }

private:

    GPR_Scalar randscaltemplate; // Underlying scalarisation model
    GPR_Scalar *randscal; // sampled version

    double dbias;
    double dscale;

    virtual void untrain(void) override;
};

inline void qswap(GPR_Scalar *&a, GPR_Scalar *&b);
inline void qswap(GPR_Scalar *&a, GPR_Scalar *&b) { GPR_Scalar *c(a); a = b; b = c; }

inline double norm2(const IMP_NLSamp &a);
inline double abs2 (const IMP_NLSamp &a);

inline double norm2(const IMP_NLSamp &a) { return a.RKHSnorm(); }
inline double abs2 (const IMP_NLSamp &a) { return a.RKHSabs();  }

inline void qswap(IMP_NLSamp &a, IMP_NLSamp &b)
{
    a.qswapinternal(b);

    return;
}

inline void IMP_NLSamp::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    IMP_NLSamp &b = dynamic_cast<IMP_NLSamp &>(bb.getML());

    qswap(randscaltemplate,b.randscaltemplate);
    qswap(randscal        ,b.randscal        );
    qswap(dbias           ,b.dbias           );
    qswap(dscale          ,b.dscale          );

    IMP_Generic::qswapinternal(b);

    return;
}

inline void IMP_NLSamp::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const IMP_NLSamp &b = dynamic_cast<const IMP_NLSamp &>(bb.getMLconst());

    randscaltemplate = b.randscaltemplate;

    if ( randscal && b.randscal )
    {
        (*randscal).semicopy(*(b.randscal));;
    }

    dbias  = b.dbias;
    dscale = b.dscale;

    IMP_Generic::semicopy(b);

    return;
}

inline void IMP_NLSamp::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const IMP_NLSamp &src = dynamic_cast<const IMP_NLSamp &>(bb.getMLconst());

    randscaltemplate = src.randscaltemplate;

    if ( randscal )
    {
        MEMDEL(randscal);
        randscal = nullptr;
    }

    if ( src.randscal )
    {
        MEMNEW(randscal,GPR_Scalar(*(src.randscal)));
    }

    dbias  = src.dbias;
    dscale = src.dscale;

    IMP_Generic::assign(src,onlySemiCopy);

    return;
}

#endif
