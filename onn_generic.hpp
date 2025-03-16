//FIRST: finish AAAI paper!


//FIXME: make vector work as bunch of scalars (as per vector SVM)
//FIXME: make anion work of vector (as per anion SVM)
//FIXME: make binary work of scalar (as per binary SVM)
//FIXME: make scalar keep local copy of scalar vectors and scalar weight
//       and bias.  Give option to get vectors by callback (just for vector
//       method).  Training should then be very fast as it can leverage
//       fast, pre-calculated Kbiased that takes xyprod.

//DONE: have made mercer kernel version that takes x'y.  Use this here

//
// 1 layer neural network base class
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _onn_generic_h
#define _onn_generic_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "ml_base.hpp"


// General model:
//
// g(x) = K( <x,w> + b )
//
// where w is a sparse vector of gentypes and b is of type gentype.  For
// classification
//
// h(x) = sgn(g(x))
//
// and moreover for vectorial type outputs elements of w and b are vectors
// (likewise for anions).
//
// Hence:
//
// dg/dx = d/dx K(x,w)
//
// Training depends on types



class ONN_Generic;


// Swap and zeroing (restarting) functions

inline void qswap(ONN_Generic &a, ONN_Generic &b);
inline ONN_Generic &setzero(ONN_Generic &a);

class ONN_Generic : public ML_Base
{
public:

    // Constructors, destructors, assignment etc..

    ONN_Generic();
    ONN_Generic(const ONN_Generic &src);
    ONN_Generic(const ONN_Generic &src, const ML_Base *xsrc);
    ONN_Generic &operator=(const ONN_Generic &src) { assign(src); return *this; }
    virtual ~ONN_Generic();

    virtual int prealloc(int expectedN) override;
    virtual int preallocsize(void) const override;
    virtual void setmemsize(int memsize) override { ML_Base::setmemsize(memsize); return; }

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0) override;
    virtual void semicopy(const ML_Base &src) override;
    virtual void qswapinternal(ML_Base &b) override;

    virtual int getparam (int ind, gentype         &val, const gentype         &xa, int ia, const gentype         &xb, int ib, charptr &desc) const override;
    virtual int egetparam(int ind, Vector<gentype> &val, const Vector<gentype> &xa, int ia, const Vector<gentype> &xb, int ib               ) const override;

    virtual std::ostream &printstream(std::ostream &output, int dep) const override;
    virtual std::istream &inputstream(std::istream &input ) override;

    virtual       ML_Base &getML     (void)       override { return static_cast<      ML_Base &>(getONN());      }
    virtual const ML_Base &getMLconst(void) const override { return static_cast<const ML_Base &>(getONNconst()); }

    // Information functions (training data):

    virtual int tspaceDim(void) const override { return db.size();  }
    virtual int order(void)     const override { return db.order(); }

    virtual double C(void) const override { return dC; }

    virtual double zerotol(void)      const override { return dzt;                     }
    virtual double Opttol(void)       const override { return dot;                     }
    virtual double lr(void)           const override { return dlr;                     }
    virtual int    maxitcnt(void)     const override { return dmitcnt;                 }
    virtual double maxtraintime(void) const override { return dmtrtime;                }
    virtual double traintimeend(void) const override { return dmtrtimeend;             }

    virtual double sparlvl(void) const override { return xspaceDim() ? (1-(dW.indsize()/xspaceDim())) : 0; }

    // General modification and autoset functions

    virtual int randomise(double sparsity) override;

    virtual int setzerotol(double nv)      override { dzt         = nv; return 1; }
    virtual int setOpttol(double nv)       override { dot         = nv; return 1; }
    virtual int setlr(double lr)           override { dlr         = lr; return 1; }
    virtual int setmaxitcnt(int nv)        override { dmitcnt     = nv; return 1; }
    virtual int setmaxtraintime(double nv) override { dmtrtime    = nv; return 1; }
    virtual int settraintimeend(double nv) override { dmtrtimeend = nv; return 1; }

    virtual int setC(double xC) override { dC = xC; return 0; }

    virtual int scale(double a) override;
    virtual int reset(void) override;
    virtual int restart(void) override { return ML_Base::restart(); }

    virtual int settspaceDim(int newdim) override;
    virtual int addtspaceFeat(int i) override;
    virtual int removetspaceFeat(int i) override;
    virtual int addxspaceFeat(int i) override;
    virtual int removexspaceFeat(int i) override;

    virtual int setorder(int neword) override;

    // Training functions:

    virtual int train(int &res) override { svmvolatile int killSwitch = 0; return train(res,killSwitch); }
    virtual int train(int &res, svmvolatile int &killSwitch) override;

    // Evaluation Functions:

    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = nullptr) const override;



    // ================================================================
    //     Common functions for all ONNs
    // ================================================================

    virtual       ONN_Generic &getONN(void)            { return *this; }
    virtual const ONN_Generic &getONNconst(void) const { return *this; }

    // Information functions (training data):

    virtual const SparseVector<gentype> &W(void) const override { return dW; }
    virtual const gentype               &B(void) const { return db; }

    // General modification and autoset functions

    virtual int setW(const SparseVector<gentype> &Wsrc) { dW = Wsrc; getKernel().getvecInfo(dWinfo,dW); return 1; }
    virtual int setB(const gentype               &bsrc) { db = bsrc;                                    return 1; }

private:

    SparseVector<gentype> dW;
    gentype db;
    vecInfo dWinfo;

    double dC;
    double dzt;
    double dot;
    int dmitcnt;
    double dmtrtime;
    double dmtrtimeend;
    double dlr;

    virtual const vecInfo &getWinfo(void) const override { return dWinfo; }
};

inline double norm2(const ONN_Generic &a);
inline double abs2 (const ONN_Generic &a);

inline double norm2(const ONN_Generic &a) { return a.RKHSnorm(); }
inline double abs2 (const ONN_Generic &a) { return a.RKHSabs();  }

inline void qswap(ONN_Generic &a, ONN_Generic &b)
{
    a.qswapinternal(b);

    return;
}

inline ONN_Generic &setzero(ONN_Generic &a)
{
    a.restart();

    return a;
}

inline void ONN_Generic::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    ONN_Generic &b = dynamic_cast<ONN_Generic &>(bb.getML());

    ML_Base::qswapinternal(b);

    qswap(dC         ,b.dC         );
    qswap(dzt        ,b.dzt        );
    qswap(dot        ,b.dot        );
    qswap(dmitcnt    ,b.dmitcnt    );
    qswap(dmtrtime   ,b.dmtrtime   );
    qswap(dmtrtimeend,b.dmtrtimeend);
    qswap(dlr        ,b.dlr        );

    qswap(dW,b.dW);
    qswap(db,b.db);

    return;
}

inline void ONN_Generic::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const ONN_Generic &b = dynamic_cast<const ONN_Generic &>(bb.getMLconst());

    ML_Base::semicopy(b);

    dC          = b.dC;
    dzt         = b.dzt;
    dot         = b.dot;
    dmitcnt     = b.dmitcnt;
    dmtrtime    = b.dmtrtime;
    dmtrtimeend = b.dmtrtimeend;
    dlr         = b.dlr;

    dW = b.dW;
    db = b.db;

    return;
}

inline void ONN_Generic::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const ONN_Generic &src = dynamic_cast<const ONN_Generic &>(bb.getMLconst());

    ML_Base::assign(src,onlySemiCopy);

    dC          = src.dC;
    dzt         = src.dzt;
    dot         = src.dot;
    dmitcnt     = src.dmitcnt;
    dmtrtime    = src.dmtrtime;
    dmtrtimeend = src.dmtrtimeend;
    dlr         = src.dlr;

    dW = src.dW;
    db = src.db;

    return;
}

#endif
