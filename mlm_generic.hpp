
//
// Type-II multi-layer kernel-machine base class
//
// Version: 7
// Date: 06/07/2018
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

//
// This operates as a wrap-around for SVM_Generic, with some indirection,
// renaming, functional hijacking etc.
//
// These functions need to be provided overwritten in variants of MLM_Generic:
//
// virtual       SVM_Generic &getQ(void)            { return QQ; }
// virtual const SVM_Generic &getQconst(void) const { return QQ; }
//
// MLM_Generic();
// MLM_Generic(const MLM_Generic &src);
// MLM_Generic(const MLM_Generic &src, const ML_Base *srcx);
//
// virtual void assign(const ML_Base &src, int onlySemiCopy = 0);
// virtual void semicopy(const ML_Base &src);
// virtual void qswapinternal(ML_Base &b);
// virtual std::ostream &printstream(std::ostream &output) const;
// virtual std::istream &inputstream(std::istream &input );
//
// int type(void)    const { return 800; }
// int subtype(void) const { return 0;   }
//
// virtual int train(int &res, svmvolatile int &killSwitch);
//


#ifndef _mlm_generic_h
#define _mlm_generic_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "ml_base.hpp"
#include "ml_base_deref.hpp"
#include "svm_generic.hpp"
#include "svm_scalar.hpp"





class MLM_Generic;

// Swap and zeroing (restarting) functions

inline void qswap(MLM_Generic &a, MLM_Generic &b);
inline MLM_Generic &setzero(MLM_Generic &a);

class MLM_Generic : public ML_Base_Deref
{
public:

    // Constructors, destructors, assignment etc..

    MLM_Generic();
    MLM_Generic(const MLM_Generic &src);
    MLM_Generic(const MLM_Generic &src, const ML_Base *srcx);
    MLM_Generic &operator=(const MLM_Generic &src) { assign(src); return *this; }
    virtual ~MLM_Generic() { return; }

    virtual int  prealloc    (int expectedN)       override;
    virtual int  preallocsize(void)          const override;
    virtual void setmemsize  (int memsize)         override;

    virtual void assign       (const ML_Base &src, int onlySemiCopy = 0) override;
    virtual void semicopy     (const ML_Base &src)                       override;
    virtual void qswapinternal(      ML_Base &b)                         override;

    virtual int getparam (int ind, gentype         &val, const gentype         &xa, int ia, const gentype         &xb, int ib, charptr &desc) const override;
    virtual int egetparam(int ind, Vector<gentype> &val, const Vector<gentype> &xa, int ia, const Vector<gentype> &xb, int ib               ) const override;

    virtual std::ostream &printstream(std::ostream &output, int dep) const override;
    virtual std::istream &inputstream(std::istream &input          )       override;

    virtual       ML_Base &getML     (void)       override { return static_cast<      ML_Base &>(getMLM());      }
    virtual const ML_Base &getMLconst(void) const override { return static_cast<const ML_Base &>(getMLMconst()); }

    // Generate RKHS vector form of ML (if possible).

    virtual RKHSVector      &getvecforma(RKHSVector      &res) const override { return getQconst().getvecforma(res); }
    virtual Vector<gentype> &getvecformb(Vector<gentype> &res) const override { return getQconst().getvecformb(res); }
    virtual gentype         &getvecformc(gentype         &res) const override { return getQconst().getvecformc(res); }

    // Information functions (training data):

    virtual int type   (void) const override { return -1;                 }
    virtual int subtype(void) const override { return -1;                 }

    virtual int isTrained(void) const override { return getQconst().isTrained() && xistrained; }
    virtual int isSolGlob(void) const override { return 0; }

    virtual double calcDistInt(int    ha, int    hb, int ia = -1, int db = 2) const override { return ML_Base::calcDistInt(ha,hb,ia,db); }
    virtual double calcDistDbl(double ha, double hb, int ia = -1, int db = 2) const override { return ML_Base::calcDistDbl(ha,hb,ia,db); }

    virtual const int *ClassLabelsInt     (void)  const override { return ML_Base::ClassLabelsInt();       }
    virtual       int  getInternalClassInt(int y) const override { return ML_Base::getInternalClassInt(y); }

    // Kernel Modification

    virtual const MercerKernel &getKernel       (void) const override { return getKnumMLconst().getKernel();   }
    virtual       MercerKernel &getKernel_unsafe(void)       override { return getKnumML().getKernel_unsafe(); }
    virtual       void          prepareKernel   (void)       override {        getKnumML().prepareKernel();    }

    virtual double tuneKernel(int method, double xwidth, int tuneK = 1, int tuneP = 0, const tkBounds *tunebounds = nullptr) override { return getKnumML().tuneKernel(method,xwidth,tuneK,tuneP,tunebounds); }

    virtual int resetKernel(                             int modind = 1, int onlyChangeRowI = -1, int updateInfo = 1) override { int res = getKnumML().resetKernel(modind,onlyChangeRowI,updateInfo); fixMLTree(); return res; }
    virtual int setKernel  (const MercerKernel &xkernel, int modind = 1, int onlyChangeRowI = -1                    ) override { int res = getKnumML().setKernel(xkernel,modind,onlyChangeRowI);      fixMLTree(); return res; }

    virtual int isKreal  (void) const override { return getKnumMLconst().isKreal();   }
    virtual int isKunreal(void) const override { return getKnumMLconst().isKunreal(); }

    virtual int setKreal  (void) override { return getKnumML().setKreal();   }
    virtual int setKunreal(void) override { return getKnumML().setKunreal(); }

    // Training set modification:

    virtual int addTrainingVector (int i, const gentype &y, const SparseVector<gentype> &x, double sigmaweigh = 1, double epsweigh = 1, int d = 2) override;
    virtual int qaddTrainingVector(int i, const gentype &y,       SparseVector<gentype> &x, double sigmaweigh = 1, double epsweigh = 1, int d = 2) override;

    virtual int addTrainingVector (int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &sigmaweigh, const Vector<double> &epsweigh) override;
    virtual int qaddTrainingVector(int i, const Vector<gentype> &y,       Vector<SparseVector<gentype> > &x, const Vector<double> &sigmaweigh, const Vector<double> &epsweigh) override;

    virtual int removeTrainingVector(int i                                      ) override { SparseVector<gentype> x; gentype y; return removeTrainingVector(i,y,x); }
    virtual int removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x) override;
    virtual int removeTrainingVector(int i, int num                             ) override { return ML_Base::removeTrainingVector(i,num); }

    virtual int setx(int                i, const SparseVector<gentype>          &x) override { int res = getQ().setx(i,x); resetKernelTree(); return res; }
    virtual int setx(const Vector<int> &i, const Vector<SparseVector<gentype> > &x) override { int res = getQ().setx(i,x); resetKernelTree(); return res; }
    virtual int setx(                      const Vector<SparseVector<gentype> > &x) override { int res = getQ().setx(  x); resetKernelTree(); return res; }

    virtual int qswapx(int                i, SparseVector<gentype>          &x, int dontupdate = 0) override { int res = getQ().qswapx(i,x,dontupdate); resetKernelTree(); return res; }
    virtual int qswapx(const Vector<int> &i, Vector<SparseVector<gentype> > &x, int dontupdate = 0) override { int res = getQ().qswapx(i,x,dontupdate); resetKernelTree(); return res; }
    virtual int qswapx(                      Vector<SparseVector<gentype> > &x, int dontupdate = 0) override { int res = getQ().qswapx(  x,dontupdate); resetKernelTree(); return res; }

    virtual int setd(int                i, int                nd) override;
    virtual int setd(const Vector<int> &i, const Vector<int> &nd) override;
    virtual int setd(                      const Vector<int> &nd) override;

    virtual void xferx(const ML_Base &xsrc) override { getQ().xferx(xsrc); resetKernelTree(); return; }

    // General modification and autoset functions

    virtual int randomise  (double sparsity) override;
    virtual int renormalise(void)            override { return ML_Base::renormalise(); }
    virtual int realign    (void)            override;

    virtual int scale  (double a) override;
    virtual int reset  (void)     override;
    virtual int restart(void)     override;
    virtual int home   (void)     override;

    virtual ML_Base &operator*=(double sf) override { scale(sf); return *this; }

    virtual int scaleby(double sf) override { *this *= sf; return 1; }

    virtual int addxspaceFeat   (int i) override { int res = getQ().addxspaceFeat(i);    resetKernelTree(); return res; }
    virtual int removexspaceFeat(int i) override { int res = getQ().removexspaceFeat(i); resetKernelTree(); return res; }

    // Training functions:

    virtual int train(int &res)                              override { svmvolatile int killSwitch = 0; return train(res,killSwitch); }
    virtual int train(int &res, svmvolatile int &killSwitch) override { (void) res; killSwitch = 0; return 0; }

    // Other functions

    virtual void setaltx(const ML_Base *_altxsrc) override { getQ().setaltx(_altxsrc); resetKernelTree(); return; }

    virtual int disable(int i)                override;
    virtual int disable(const Vector<int> &i) override;







    // ================================================================
    //     Common functions for all MLMs
    // ================================================================

    virtual       MLM_Generic &getMLM     (void)       { return *this; }
    virtual const MLM_Generic &getMLMconst(void) const { return *this; }

    // Back-propogation control
    //
    // tsize: size (depth) of kernel network (not including output layer)
    // knum:  sets which layer does get/set/reset/prepareKernel return (-1 to tsize()-1).
    //        at -1 you should only set type of inheritance
    //
    // regtype:  regularisation type. 1 for 1-norm, 2 for 2-norm
    // regC:     "C" value for layer knum (C for output layer can also be set using setC)
    // mlmlr:    learning rate
    // diffstop: stop when average decrease in total training error for a step is less than this
    // lsparse:  sparsity for initial random layer initialisation (1 means dense, 0 means all zero)

    virtual int tsize(void) const { return mltree.size(); }
    virtual int knum (void) const { return xknum;         }

    virtual int    regtype (int l) const { NiceAssert( ( l >= 0 ) && ( l < tsize() ) ); return xregtype(l); }
    virtual double regC    (int l) const { NiceAssert( ( l >= 0 ) && ( l < tsize() ) ); return getKnumMLconst(l).C(); }
    virtual double mlmlr   (void)  const { return xmlmlr; }
    virtual double diffstop(void)  const { return xdiffstop; }
    virtual double lsparse (void)  const { return xlsparse; }

    virtual const Matrix<double> &GGp(int l) const { return (dynamic_cast<const SVM_Generic &>(getKnumMLconst(l))).Gp(); }

    virtual int settsize(int nv) { NiceAssert( ( nv >= 1 ) ); int ov = tsize(); xistrained = 0; mltree.resize(nv); xregtype.resize(nv); if ( ov > nv ) { retVector<int> tmpva; xregtype("&",ov,1,nv-1,tmpva) = DEFAULT_REGTYPE; } fixMLTree(); return 1; }
    virtual int setknum (int nv) { NiceAssert( ( nv >= -1 ) && ( nv < tsize() ) ); xknum = nv; return 0; }

    virtual int setregtype (int l, int    nv) { NiceAssert( ( l >= 0 ) && ( l < tsize() ) ); NiceAssert( ( nv == 1 ) || ( nv == 2 ) ); xistrained = 0; xregtype("&",l) = nv; return 1; }
    virtual int setregC    (int l, double nv) { NiceAssert( ( l >= 0 ) && ( l < tsize() ) ); NiceAssert( nv > 0 ); xistrained = 0; return getKnumML(l).setC(nv); }
    virtual int setmlmlr   (double nv)        { NiceAssert( nv > 0.0 ); xmlmlr = nv; return 0; }
    virtual int setdiffstop(double nv)        { NiceAssert( nv > 0.0 ); xdiffstop = nv; return 0; }
    virtual int setlsparse (double nv)        { NiceAssert( nv >= 0.0 ); xlsparse = nv; return 0; }

    // Base-level stuff
    //
    // This is overloaded by children to return correct Q type

    virtual       SVM_Generic &getQQ     (void)       { return QQ; }
    virtual const SVM_Generic &getQQconst(void) const { return QQ; }

    virtual       ML_Base &getQ     (void)       override { return static_cast<      ML_Base &>(getQQ());      }
    virtual const ML_Base &getQconst(void) const override { return static_cast<const ML_Base &>(getQQconst()); }

protected:

    SVM_Generic QQ;

    int xistrained;
    int xknum; // which kernel is referenced by get/set/resetKernel functions (-1 for QQ, otherwise mltree element)
    Vector<int> xregtype;
    double xmlmlr;
    double xdiffstop;
    double xlsparse;

    Vector<SVM_Scalar> mltree;

    void fixMLTree(int modind = 0);
    void resetKernelTree(int modind = 0);
    ML_Base &getKnumML(int ovr = -2);
    const ML_Base &getKnumMLconst(int ovr = -2) const;
};

inline double norm2(const MLM_Generic &a);
inline double abs2 (const MLM_Generic &a);

inline double norm2(const MLM_Generic &a) { return a.RKHSnorm(); }
inline double abs2 (const MLM_Generic &a) { return a.RKHSabs();  }

inline void qswap(MLM_Generic &a, MLM_Generic &b)
{
    a.qswapinternal(b);

    return;
}

inline MLM_Generic &setzero(MLM_Generic &a)
{
    a.restart();

    return a;
}

inline void MLM_Generic::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    MLM_Generic &b = dynamic_cast<MLM_Generic &>(bb.getML());

    ML_Base::qswapinternal(b);

    qswap(xregtype  ,b.xregtype  );
    qswap(xmlmlr    ,b.xmlmlr    );
    qswap(xdiffstop ,b.xdiffstop );
    qswap(xlsparse  ,b.xlsparse  );
    qswap(xknum     ,b.xknum     );
    qswap(xistrained,b.xistrained);
    qswap(mltree    ,b.mltree    );

    fixMLTree();

    return;
}

inline void MLM_Generic::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const MLM_Generic &b = dynamic_cast<const MLM_Generic &>(bb.getMLconst());

    ML_Base::semicopy(b);

    xistrained = b.xistrained;

    xregtype   = b.xregtype;
    xmlmlr     = b.xmlmlr;
    xdiffstop  = b.xdiffstop;
    xlsparse   = b.xlsparse;
    xknum      = b.xknum;
    //mltree     = b.mltree;

    return;
}

inline void MLM_Generic::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const MLM_Generic &src = dynamic_cast<const MLM_Generic &>(bb.getMLconst());

    ML_Base::assign(src,onlySemiCopy);

    xregtype   = src.xregtype;
    xmlmlr     = src.xmlmlr;
    xdiffstop  = src.xdiffstop;
    xlsparse   = src.xlsparse;
    xknum      = src.xknum;
    xistrained = src.xistrained;

    if ( !onlySemiCopy )
    {
        mltree = src.mltree;

        //fixMLTree() - need to do this (can't do it in constructor)
    }

    else
    {
        // Assume in virtual non-constructor!

        fixMLTree();

        int i;

        for ( i = 0 ; i < tsize() ; ++i )
        {
            mltree("&",i).assign(bb,onlySemiCopy);
        }
    }

    return;
}

#endif
