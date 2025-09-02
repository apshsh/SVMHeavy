//runthese/svmheavy.err.log.14   - this is in the ehi run.  What went wrong?

//DONE FIXME: inherit from ml_base_deref, and then imp_parsvm and imp_nlsamp can simply redirect getQ
//FIXME: in both, stop redirecting gh.  let it go to the underlying class
//FIXME: when logging in bayesopt.h, need to reinstate the template over x to make it work!
//FIXME: when logging in bayesopt.h, add data to the surface plot!
//FIXME: wherever IMPs are used, untrain after use

//
// Improvement measure base class
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _imp_generic_h
#define _imp_generic_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "ml_base_deref.hpp"



// Defines blocks used as mono-surrogates in multitarget bayesian
// optimisation.  It is assumed that we are trying to solve the multi-
// objective minimisation problem:
//
// min_x f(x) = (f0(x), f1(x), ...)
//
// The output vectors f(x0), f(x1), ... are collected and added to this
// block as training data.  The *negated*  improvement from adding a new
// vector f(x) to the training set is given by the imp(...) function.  The
// smallest (most negative) improvement indicates the best candidate.
//
// Note that it is generally a good idea to enforce:
//
// f(x) <= 0
//
// This is only strictly required by imp_expect in the multi-objective
// case, but nevertheless it is a good idea for compatibility with this
// class.
//
// Note also that all x vectors should be in dense form rather than sparse.


class IMP_Generic;


// Swap and zeroing (restarting) functions

inline void qswap(IMP_Generic &a, IMP_Generic &b);
inline IMP_Generic &setzero(IMP_Generic &a);

class IMP_Generic : public ML_Base_Deref
{
public:

    // Assumptions: all virtual functions inherited from ML_Base are left
    // unchanged in this class.

    // Constructors, destructors, assignment etc..

    IMP_Generic(int isIndPrune = 0);
    IMP_Generic(const IMP_Generic &src, int isIndPrune = 0);
    IMP_Generic(const IMP_Generic &src, const ML_Base *xsrc, int isIndPrune = 0);
    IMP_Generic &operator=(const IMP_Generic &src) { assign(src); return *this; }
    virtual ~IMP_Generic();

    virtual void assign       (const ML_Base &src, int onlySemiCopy = 0) override;
    virtual void semicopy     (const ML_Base &src)                       override;
    virtual void qswapinternal(ML_Base &b)                               override;

    virtual int getparam (int ind, gentype         &val, const gentype         &xa, int ia, const gentype         &xb, int ib, charptr &desc) const override;
    virtual int egetparam(int ind, Vector<gentype> &val, const Vector<gentype> &xa, int ia, const Vector<gentype> &xb, int ib               ) const override;

    virtual std::ostream &printstream(std::ostream &output, int dep) const override;
    virtual std::istream &inputstream(std::istream &input          )       override;

    virtual       ML_Base &getML     (void)       override { return static_cast<      ML_Base &>(getIMP()     ); }
    virtual const ML_Base &getMLconst(void) const override { return static_cast<const ML_Base &>(getIMPconst()); }

    // Information functions (training data):

    virtual int  subtype (void)  const override { return 0;   }
    virtual char targType(void)  const override { return 'N'; }

    virtual int tspaceDim   (void)     const override { return 1;                          }
    virtual int xspaceDim   (int = -1) const override { return ( zxdim >= 0 ) ? zxdim : 0; }

    virtual int isTrained(void) const override { return disTrained; }

    // Data modification

    virtual int addTrainingVector (int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2) override;
    virtual int qaddTrainingVector(int i, const gentype &y,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2) override;

    virtual int addTrainingVector (int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override;
    virtual int qaddTrainingVector(int i, const Vector<gentype> &y,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override;

    virtual int removeTrainingVector(int i                                      ) override { SparseVector<gentype> x; gentype y; return removeTrainingVector(i,y,x); }
    virtual int removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x) override;
    virtual int removeTrainingVector(int i, int num                             ) override;

    virtual int setx(int                i, const SparseVector<gentype>          &x) override;
    virtual int setx(const Vector<int> &i, const Vector<SparseVector<gentype> > &x) override;
    virtual int setx(                      const Vector<SparseVector<gentype> > &x) override;

    virtual int qswapx(int                i, SparseVector<gentype>          &x, int dontupdate = 0) override;
    virtual int qswapx(const Vector<int> &i, Vector<SparseVector<gentype> > &x, int dontupdate = 0) override;
    virtual int qswapx(                      Vector<SparseVector<gentype> > &x, int dontupdate = 0) override;

    virtual int setd(int                i, int                nd) override;
    virtual int setd(const Vector<int> &i, const Vector<int> &nd) override;
    virtual int setd(                      const Vector<int> &nd) override;

    // General modification and autoset functions

    virtual int reset(void)   override { untrain(); return 1;       }
    virtual int restart(void) override { return getQ().restart(); }

    // Training functions:

    virtual int train(int &res)                    override { svmvolatile int killSwitch = 0; return train(res,killSwitch); }
    virtual int train(int &res, svmvolatile int &) override { res = 0; disTrained = 1;        return 0;                     }






    // ================================================================
    //     IMP Specific functions
    // ================================================================

    virtual       IMP_Generic &getIMP     (void)       { return *this; }
    virtual const IMP_Generic &getIMPconst(void) const { return *this; }

    // Improvement functions: given mean and variance of input x calculate
    // relevant measure of improvement/goodness of result.

    virtual int imp(gentype &resi, gentype &resv, const SparseVector<gentype> &xxmean, const gentype &xxvar) const { (void) resi; (void) resv; (void) xxmean; (void) xxvar; NiceThrow("Error: imp() not defined at generic level."); return 0; }

    // Information functions
    //
    // hypervol:  hypervolume of training set
    //
    // needdg:    does this require dg/dx?
    //            0: no
    //            1: yes
    //
    // zref:      zero reference used by EHI
    // ehimethod: EHI calculation method (for EI with vector input)
    //            0: use optimised EHI calculation with full cache
    //            1: use optimised EHI calculation with partial cache
    //            2: use optimised EHI calculation with no cache
    //            3: use Hupkens method
    //            4: use Couckuyt method
    //
    // scaltype:  Scalarisation types (RLSamp) are:
    //            0: random linear scalarisation
    //            1: random Chebyshev scalarisation
    //            2: augmented Chebyshev
    //            3: modified Chebyshev
    // scalalpha: alpha factor used by scaltype 2,3
    //
    // xdim:      assumed dimension of x vectors
    // Nsamp:     number of samples for IMP
    // sampSlack: sample slack for IMP

    virtual double hypervol(void) const;
    virtual int    needdg  (void) const { return 0; }

    virtual double zref     (void) const { return xzref;      }
    virtual int    ehimethod(void) const { return xehimethod; }
    virtual int    scaltype (void) const { return xscaltype;  }
    virtual double scalalpha(void) const { return xscalalpha; }
    virtual int    xdim     (void) const { return zxdim;      }
    virtual int    Nsamp    (void) const { return xNsamp;     }
    virtual double sampSlack(void) const { return xsampSlack; }

    // Modification function

    virtual int setzref     (double nv) { xzref      = nv; return 0; }
    virtual int setehimethod(int    nv) { xehimethod = nv; return 0; }
    virtual int setscaltype (int    nv) { xscaltype  = nv; return 0; }
    virtual int setscalalpha(double nv) { xscalalpha = nv; return 0; }
    virtual int setxdim     (int    nv) { zxdim      = nv; return 0; }
    virtual int setNsamp    (int    nv) { xNsamp     = nv; return 0; }
    virtual int setsampSlack(double nv) { xsampSlack = nv; return 0; }

    // Bypass to ML_Base

    virtual       ML_Base &getQ(void)            { return bypassml; }
    virtual const ML_Base &getQconst(void) const { return bypassml; }

protected:

    ML_Base bypassml;

    // Overload these in all derived classes

    virtual void untrain(void)
    {
        disTrained = 0;

        return;
    }

private:

    double xzref;
    int xehimethod;
    int xscaltype;
    double xscalalpha;
    int zxdim;
    int xNsamp;
    double xsampSlack;

    int disTrained;
};

inline double norm2(const IMP_Generic &a);
inline double abs2 (const IMP_Generic &a);

inline double norm2(const IMP_Generic &a) { return a.RKHSnorm(); }
inline double abs2 (const IMP_Generic &a) { return a.RKHSabs();  }

inline void qswap(IMP_Generic &a, IMP_Generic &b)
{
    a.qswapinternal(b);

    return;
}

inline IMP_Generic &setzero(IMP_Generic &a)
{
    a.restart();

    return a;
}

inline void IMP_Generic::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    IMP_Generic &b = dynamic_cast<IMP_Generic &>(bb.getML());

    qswap(xzref     ,b.xzref     );
    qswap(xehimethod,b.xehimethod);
    qswap(xscaltype ,b.xscaltype );
    qswap(xscalalpha,b.xscalalpha);
    qswap(zxdim     ,b.zxdim     );
    qswap(xNsamp    ,b.xNsamp    );
    qswap(xsampSlack,b.xsampSlack);
    qswap(disTrained,b.disTrained);

    getQ().qswapinternal(b);

    return;
}

inline void IMP_Generic::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const IMP_Generic &b = dynamic_cast<const IMP_Generic &>(bb.getMLconst());

    xzref      = b.xzref;
    xehimethod = b.xehimethod;
    xscaltype  = b.xscaltype;
    xscalalpha = b.xscalalpha;
    zxdim      = b.zxdim;
    xNsamp     = b.xNsamp;
    xsampSlack = b.xsampSlack;
    disTrained = b.disTrained;

    getQ().semicopy(b);

    return;
}

inline void IMP_Generic::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const IMP_Generic &src = dynamic_cast<const IMP_Generic &>(bb.getMLconst());

    xzref      = src.xzref;
    xehimethod = src.xehimethod;
    xscaltype  = src.xscaltype;
    xscalalpha = src.xscalalpha;
    zxdim      = src.zxdim;
    xNsamp     = src.xNsamp;
    xsampSlack = src.xsampSlack;
    disTrained = src.disTrained;

    getQ().assign(src,onlySemiCopy);

    return;
}

#endif
