//FIXME: change to use cheatscache via kernel 800 callback.  Then you can get rid of K2 stuff here because the cache is *automatically* called direct from ML_Base.  Note that all kernel
// references including getKernel must be diverted appropriately to cheatscache: NOTHING must be allowed to change the kernel in SVM_Planar!

//
// Planar SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _svm_planar_h
#define _svm_planar_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "svm_scalar.hpp"



class SVM_Planar;

// Swap function

inline void qswap(SVM_Planar &a, SVM_Planar &b);


class SVM_Planar : public SVM_Scalar
{
public:

    // Constructors, destructors, assignment etc..

    SVM_Planar();
    SVM_Planar(const SVM_Planar &src);
    SVM_Planar(const SVM_Planar &src, const ML_Base *xsrc);
    SVM_Planar &operator=(const SVM_Planar &src) { assign(src); return *this; }
    virtual ~SVM_Planar();

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0) override;
    virtual void semicopy(const ML_Base &src) override;
    virtual void qswapinternal(ML_Base &b) override;

    virtual int restart(void) override { SVM_Planar temp; *this = temp; return 1; }

    virtual std::ostream &printstream(std::ostream &output, int dep) const override;
    virtual std::istream &inputstream(std::istream &input) override;

    // Information functions (training data):
    // 
    // calcDist: - i == -1 then this assumes that both ha and hb are vectors
    //           - i >= 0, i < N then assumes ha is a vector and hb a scalar
    //             and will first convert ha -> ha.x(i).f4(7) to get scalar
    //           - i >= N also assumes ha is a vector and hb a scalar but
    //             will convert ha -> ha.u(i-N)

    virtual int type(void)    const override { return 16; }
    virtual int subtype(void) const override { return 0;  }

    virtual int tspaceDim(void) const override { return bdim ; }

    virtual char gOutType(void) const override { return 'V'; }
    virtual char hOutType(void) const override { return ( defproj == -1 ) ? gOutType() : 'R'; }
    virtual char targType(void) const override { return 'R'; }
    virtual double calcDist(const gentype &ha, const gentype &hb, int i = -1, int db = 2) const override;

    virtual int isUnderlyingScalar(void) const override { return 1; }
    virtual int isUnderlyingVector(void) const override { return 0; }
    virtual int isUnderlyingAnions(void) const override { return 0; }

    virtual int isPlanarType(void) const override { return 1; }

    // Kernel Modification - this does all the work by changing how K is
    // evaluated.

    virtual gentype &K2(gentype &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo *xainf = nullptr, const vecInfo *xbinf = nullptr) const override { return ML_Base::K2(res,xa,xb,xainf,xbinf); }

    virtual gentype        &K2(              gentype        &res, int ia, int ib                          , const gentype **pxyprod = nullptr, const SparseVector<gentype> *xx = nullptr, const SparseVector<gentype> *yy = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *yyinfo = nullptr, int resmode = 0) const override;
    virtual gentype        &K2(              gentype        &res, int ia, int ib, const gentype &bias     , const gentype **pxyprod = nullptr, const SparseVector<gentype> *xx = nullptr, const SparseVector<gentype> *yy = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *yyinfo = nullptr, int resmode = 0) const override;
    virtual gentype        &K2(              gentype        &res, int ia, int ib, const MercerKernel &altK, const gentype **pxyprod = nullptr, const SparseVector<gentype> *xx = nullptr, const SparseVector<gentype> *yy = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *yyinfo = nullptr, int resmode = 0) const override;
    virtual double          K2(                                   int ia, int ib                          , const gentype **pxyprod = nullptr, const SparseVector<gentype> *xx = nullptr, const SparseVector<gentype> *yy = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *yyinfo = nullptr, int resmode = 0) const override;
    virtual Matrix<double> &K2(int spaceDim, Matrix<double> &res, int ia, int ib                          , const gentype **pxyprod = nullptr, const SparseVector<gentype> *xx = nullptr, const SparseVector<gentype> *yy = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *yyinfo = nullptr, int resmode = 0) const override;
    virtual d_anion        &K2(int order,    d_anion        &res, int ia, int ib                          , const gentype **pxyprod = nullptr, const SparseVector<gentype> *xx = nullptr, const SparseVector<gentype> *yy = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *yyinfo = nullptr, int resmode = 0) const override;

    // Add/remove data:

    virtual int addTrainingVector (int i, double z, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2) override;
    virtual int qaddTrainingVector(int i, double z,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2) override;

    virtual int addTrainingVector (int i, const Vector<double> &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<int> &d) override;
    virtual int qaddTrainingVector(int i, const Vector<double> &z,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<int> &d) override;

    virtual int addTrainingVector (int i, const gentype &z, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2) override;
    virtual int qaddTrainingVector(int i, const gentype &z,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2) override;

    virtual int addTrainingVector (int i, const Vector<gentype> &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override;
    virtual int qaddTrainingVector(int i, const Vector<gentype> &z,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override;

    virtual int removeTrainingVector(int i                                      ) override { SparseVector<gentype> x; gentype y; return removeTrainingVector(i,y,x); }
    virtual int removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x) override;
    virtual int removeTrainingVector(int i, int num                             ) override;

    virtual int setx(int                i, const SparseVector<gentype>          &x) override;
    virtual int setx(const Vector<int> &i, const Vector<SparseVector<gentype> > &x) override;
    virtual int setx(                      const Vector<SparseVector<gentype> > &x) override;

    virtual int setd(int                i, int                d) override;
    virtual int setd(const Vector<int> &i, const Vector<int> &d) override;
    virtual int setd(                      const Vector<int> &d) override;

    // Modification

    virtual int prealloc(int expectedN) override;
    virtual int preallocsize(void) const override;
    virtual void prepareKernel(void) override { return; }
    virtual int resetKernel(int modind = 1, int onlyChangeRowI = -1, int updateInfo = 1) override;
    virtual int setKernel(const MercerKernel &xkernel, int modind = 1, int onlyChangeRowI = -1) override;
    virtual void setmemsize(int memsize) override;

    virtual int setVarBias(void) override;
    virtual int setPosBias(void) override;
    virtual int setNegBias(void) override;
    virtual int setFixedBias(double newbias) override;
    virtual int setVarBias(int q) override;
    virtual int setPosBias(int q) override;
    virtual int setNegBias(int q) override;
    virtual int setFixedBias(int q, double newbias) override;

    // Training

    virtual int train(int &res)                              override { svmvolatile int killSwitch = 0; return train(res,killSwitch); }
    virtual int train(int &res, svmvolatile int &killSwitch) override;

    // Evaluation:

    virtual int isVarDefined(void) const override { return 0; }

    virtual int gh(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = nullptr) const override;
    virtual int gh(gentype &resh, gentype &resg, const SparseVector<gentype> &x, int retaltg = 0, const vecInfo *xinf = nullptr, gentype ***pxyprodx = nullptr) const override { return SVM_Scalar::gh(resh,resg,x,retaltg,xinf,pxyprodx); }

    // Output basis

    virtual int NbasisVV(void)    const override { return locbasis.size();       }
    virtual int basisTypeVV(void) const override { return 0;                     }
    virtual int defProjVV(void)   const override { return defproj;    }

    virtual const Vector<gentype> &VbasisVV(void) const override { return locbasisgt; }

    virtual int setBasisYVV(void) override { NiceThrow("Function setBasisY not available for this ML type"); return 0; }
    virtual int setBasisUVV(void) override { return 0; }
    virtual int addToBasisVV(int i, const gentype &o) override;
    virtual int removeFromBasisVV(int i) override;
    virtual int setBasisVV(int i, const gentype &o) override;
    virtual int setBasisVV(const Vector<gentype> &o) override;
    virtual int setDefaultProjectionVV(int d) override { defproj = d; return 1; }
    virtual int setBasisVV(int i, int d) override { return ML_Base::setBasisVV(i,d); }

protected:

    // Local basis and factorisation

    Vector<Vector<double> > locbasis;
    Vector<gentype> locbasisgt;
    Matrix<double> VV;

    // Protected function passthrough

    int maxFreeAlphaBias(void) { return SVM_Scalar::maxFreeAlphaBias(); }
    int fact_minverse(Vector<double> &xdalpha, Vector<double> &xdbeta, const Vector<double> &bAlpha, const Vector<double> &bBeta) { return SVM_Scalar::fact_minverse(xdalpha,xdbeta,bAlpha,bBeta); }

    void refactorVV(int updateGpn = 1);
    Vector<Vector<double> > &reflocbasis(void) { return locbasis; }
    void reconstructlocbasisgt(void);
    Vector<int> &reflocd(void) { return locd; }
    int getbdim(void) { return bdim; }
    virtual int setBasisVV(int i, const gentype &o, int updateU);

    // Force exhaustive evaluation of gh(i) for i >= 0 (used by svm_cyclic)
    // Assumed set briefly then reset immediately afterward

    int ghEvalFull;

    virtual int unsafesetDefaultProjectionVV(int d) const { defproj = d; return 1; }

private:

    // Blocked functions

    void setGpnExt(Matrix<double> *GpnExtOld, Matrix<double> *GpnExtNew) { (void) GpnExtOld; (void) GpnExtNew; NiceThrow("setGpnExt blocked in svm_Planar"); return; }
    void naivesetGpnExt(Matrix<double> *GpnExtVal) { (void) GpnExtVal; NiceThrow("naivesetGpnExt blocked in svm_Planar"); return; }

    void setbiasdim(int xbiasdim, int addpos, double addval, int rempos) { (void) xbiasdim; (void) addpos; (void) addval; (void) rempos; NiceThrow("setbiasdim blocked in svm_Planar"); return; }
    void setBiasVMulti(const Vector<double> &nwbias) { (void) nwbias; NiceThrow("setBiasVMulti blocked in svm_Planar"); return; }

    void setgn(const Vector<double> &gnnew) { (void) gnnew; NiceThrow("setgn blocked in svm_Planar"); return; }
    void setGn(const Matrix<double> &Gnnew) { (void) Gnnew; NiceThrow("setGn blocked in svm_Planar"); return; }

private:

    SVM_Scalar cheatscache;
    int midadd;

    // Gpn vector: bias does not affect the ranking constraint variables,
    //             so we need to set Gpn = 0 for these
    // locd: resetKernel does some strange things with setd at svm_scalar
    //       level, so we need to keep a local copy.

    Matrix<double> inGpn;
    Vector<int> locd;

    // Basis dimensions

    int bdim;
    mutable int defproj;

    // Helper functions

    void calcVVij(double &res, int i, int j) const;
    int rankcalcGpn(Vector<double> &res, int d, const SparseVector<gentype> &x, int i);
};

inline double norm2(const SVM_Planar &a);
inline double abs2 (const SVM_Planar &a);

inline double norm2(const SVM_Planar &a) { return a.RKHSnorm(); }
inline double abs2 (const SVM_Planar &a) { return a.RKHSabs();  }

inline void qswap(SVM_Planar &a, SVM_Planar &b)
{
    a.qswapinternal(b);

    return;
}

inline void SVM_Planar::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    SVM_Planar &b = dynamic_cast<SVM_Planar &>(bb.getML());

    SVM_Scalar::qswapinternal(b);
    
    qswap(inGpn         ,b.inGpn         );
    qswap(locd          ,b.locd          );
    qswap(locbasis      ,b.locbasis      );
    qswap(VV            ,b.VV            );
    qswap(bdim          ,b.bdim          );
    qswap(cheatscache   ,b.cheatscache   );
    qswap(midadd        ,b.midadd        );
    qswap(defproj       ,b.defproj       );

    SVM_Scalar::naivesetGpnExt(&inGpn);

    return;
}

inline void SVM_Planar::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const SVM_Planar &b = dynamic_cast<const SVM_Planar &>(bb.getMLconst());

    SVM_Scalar::semicopy(b);

    inGpn          = b.inGpn;
    locd           = b.locd;
    //locbasis       = b.locbasis;
    //VV             = b.VV;
    //bdim           = b.bdim;
    midadd         = b.midadd;
    defproj        = b.defproj;

    SVM_Scalar::naivesetGpnExt(&inGpn);

    return;
}

inline void SVM_Planar::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const SVM_Planar &src = dynamic_cast<const SVM_Planar &>(bb.getMLconst());

    SVM_Scalar::assign(static_cast<const SVM_Scalar &>(src),onlySemiCopy);

    inGpn          = src.inGpn;
    locd           = src.locd;
    locbasis       = src.locbasis;
    VV             = src.VV;
    bdim           = src.bdim;
    cheatscache    = src.cheatscache;
    midadd         = src.midadd;
    defproj        = src.defproj;

    SVM_Scalar::naivesetGpnExt(&inGpn);

    return;
}

#endif
