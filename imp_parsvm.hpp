
//
// 1-norm 1-class Pareto SVM measure
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _imp_parsvm_h
#define _imp_parsvm_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "imp_generic.hpp"
#include "svm_pfront.hpp"


// This is an IMP front-end to svm_pfront.  The modifications made by this
// layer are:
//
// - imp(x) is defined.  This approximates the variance by noting the
//   monotonicity of g(x) and uses that to approximate the variance of the
//   estimate g(x).
// - EI and PI are not well defined in this case.
// - raw and GP-UCB are defined.


class IMP_ParSVM;


inline void qswap(IMP_ParSVM &a, IMP_ParSVM &b);


class IMP_ParSVM : public IMP_Generic
{
public:

    // Constructors, destructors, assignment etc..

    IMP_ParSVM(int _isIndPrune = 0);
    IMP_ParSVM(const IMP_ParSVM &src, int _isIndPrune = 0);
    IMP_ParSVM(const IMP_ParSVM &src, const ML_Base *xsrc, int _isIndPrune = 0);
    IMP_ParSVM &operator=(const IMP_ParSVM &src) { assign(src); return *this; }
    virtual ~IMP_ParSVM();

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0) override;
    virtual void semicopy(const ML_Base &src)                     override;
    virtual void qswapinternal(ML_Base &b)                        override;

    virtual int getparam (int ind, gentype         &val, const gentype         &xa, int ia, const gentype         &xb, int ib, charptr &desc) const override { return IMP_Generic::getparam( ind,val,xa,ia,xb,ib,desc); }
    virtual int egetparam(int ind, Vector<gentype> &val, const Vector<gentype> &xa, int ia, const Vector<gentype> &xb, int ib               ) const override { return IMP_Generic::egetparam(ind,val,xa,ia,xb,ib     ); }

    virtual std::ostream &printstream(std::ostream &output, int dep) const override;
    virtual std::istream &inputstream(std::istream &input          )       override;

    // Information functions (training data):

    virtual int type   (void)  const override { return 601; }
    virtual int subtype(void)  const override { return 0;   }

    virtual double calcDistInt(int    ha, int    hb, int ia = -1, int db = 2) const override { return bypassml.calcDistInt(ha,hb,ia,db); }
    virtual double calcDistDbl(double ha, double hb, int ia = -1, int db = 2) const override { return bypassml.calcDistDbl(ha,hb,ia,db); }

    virtual const int *ClassLabelsInt(void) const override { return bypassml.ClassLabelsInt();       }
    virtual int  getInternalClassInt(int y) const override { return bypassml.getInternalClassInt(y); }

    // Training set modification:

    virtual int removeTrainingVector(int i, int num) override { return IMP_Generic::removeTrainingVector(i,num); }

    // Training functions:

    virtual int train(int &res)                              override { svmvolatile int killSwitch = 0; return train(res,killSwitch); }
    virtual int train(int &res, svmvolatile int &killSwitch) override { int resa = 0; int resb = 0; int ires = getQ().train(resa,killSwitch); ires |= IMP_Generic::train(resb,killSwitch); res = resa+(100*resb); return ires; }









    // ================================================================
    //     Common to all IMPs
    // ================================================================

    // Evaluation Functions:
    //
    // Output imp(E(x),var(x)) depends on svmmethod

    virtual int imp(gentype &resi, gentype &resv, const SparseVector<gentype> &xxmean, const gentype &xxvar) const override;

    virtual int needdg(void) const override { return 1; }





    // Base-level stuff
    //
    // This is overloaded by children to return correct Q type

    virtual       ML_Base &getQ(void)            { return content; }
    virtual const ML_Base &getQconst(void) const { return content; }

private:

    SVM_PFront content;

    virtual void untrain(void) override
    {
        IMP_Generic::untrain();
        return;
    }
};

inline double norm2(const IMP_ParSVM &a);
inline double abs2 (const IMP_ParSVM &a);

inline double norm2(const IMP_ParSVM &a) { return a.RKHSnorm(); }
inline double abs2 (const IMP_ParSVM &a) { return a.RKHSabs();  }

inline void qswap(IMP_ParSVM &a, IMP_ParSVM &b)
{
    a.qswapinternal(b);

    return;
}

inline void IMP_ParSVM::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    IMP_ParSVM &b = dynamic_cast<IMP_ParSVM &>(bb.getML());

    qswap(content,b.content);

    IMP_Generic::qswapinternal(b);

    return;
}

inline void IMP_ParSVM::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const IMP_ParSVM &b = dynamic_cast<const IMP_ParSVM &>(bb.getMLconst());

    content.semicopy(b.content);

    IMP_Generic::semicopy(b);

    return;
}

inline void IMP_ParSVM::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const IMP_ParSVM &src = dynamic_cast<const IMP_ParSVM &>(bb.getMLconst());

    content.assign(src.content,onlySemiCopy);
    IMP_Generic::assign(src,onlySemiCopy);

    return;
}

#endif
