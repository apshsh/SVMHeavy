
// NB: the gradient of a GP is a GP
// mean of gradient is obtained from dg
// variance of gradient is obtained from dcov


//
// Gaussian Process (GP) base class
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

//
// Currently this is basically a wrap-around for a LS-SVR with C mapped to
// 1/sigma for noise regularisation.  This is equivalent to the standard
// GP regressor assuming Gaussian measurement noise.  By default the zero 
// mean case is assumed (translates to fixed bias), but you can change this
// and it will work for the general case (variance adjusted as per:
//
// Bull: Convergence Rates of Efficient Global Optimisation
//
// For gradients: http://mlg.eng.cam.ac.uk/mchutchon/DifferentiatingGPs.pdf
//

#ifndef _gpr_generic_h
#define _gpr_generic_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "ml_base.hpp"
#include "ml_base_deref.hpp"
#include "lsv_generic.hpp"




class GPR_Generic;

// Swap and zeroing (restarting) functions

inline void qswap(GPR_Generic &a, GPR_Generic &b);
inline void qswap(GPR_Generic *&a, GPR_Generic *&b);

inline GPR_Generic &setzero(GPR_Generic &a);

class GPR_Generic : public ML_Base_Deref
{
public:

    // Constructors, destructors, assignment etc..

    GPR_Generic();
    GPR_Generic(const GPR_Generic &src);
    GPR_Generic(const GPR_Generic &src, const ML_Base *srcx);
    GPR_Generic &operator=(const GPR_Generic &src) { assign(src); return *this; }
    virtual ~GPR_Generic() { return; }

    virtual int prealloc(int expectedN) override;
    virtual int preallocsize(void) const override { return getQconst().preallocsize(); }
    virtual void setmemsize(int memsize) override { return getQ().setmemsize(memsize); }

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0) override;
    virtual void semicopy(const ML_Base &src) override;
    virtual void qswapinternal(ML_Base &b) override;

    virtual int getparam (int ind, gentype         &val, const gentype         &xa, int ia, const gentype         &xb, int ib, charptr &desc) const override;
    virtual int egetparam(int ind, Vector<gentype> &val, const Vector<gentype> &xa, int ia, const Vector<gentype> &xb, int ib               ) const override;

    virtual std::ostream &printstream(std::ostream &output, int dep) const override;
    virtual std::istream &inputstream(std::istream &input ) override;

    virtual       ML_Base &getML     (void)       override { return static_cast<      ML_Base &>(getGPR());      }
    virtual const ML_Base &getMLconst(void) const override { return static_cast<const ML_Base &>(getGPRconst()); }

    // Information functions (training data):

    virtual int NNC(int d)    const override { return Nnc(d+1);           }
    virtual int type(void)    const override { return -1;                 }
    virtual int subtype(void) const override { return -1;                 }

    virtual int isSolGlob(void) const override { return 1; }

    virtual int isVarDefined(void) const override { return 1; }

    virtual double calcDistInt(int    ha, int    hb, int ia = -1, int db = 2) const override { return ML_Base::calcDistInt(ha,hb,ia,db); }
    virtual double calcDistDbl(double ha, double hb, int ia = -1, int db = 2) const override { return ML_Base::calcDistDbl(ha,hb,ia,db); }

    virtual double C(void)         const override { return 1/dsigma;                }
    virtual double sigma(void)     const override { return dsigma;                  }
    virtual double sigma_cut(void) const override { return dsigma_cut;              }
    virtual double Cclass(int d)   const override { (void) d; return 1.0;           }

    virtual const Vector<gentype>                &y          (void) const override { return dy;                        }
    virtual const Vector<int>                    &d          (void) const override { return xd;                        }
    virtual const Vector<double>                 &Cweight    (void) const override { return dCweight;                  }
    virtual const Vector<double>                 &sigmaweight(void) const override { return dsigmaweight;              }

    // Training set modification:

    virtual int addTrainingVector (int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2) override;
    virtual int qaddTrainingVector(int i, const gentype &y,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2) override;

    virtual int addTrainingVector (int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override;
    virtual int qaddTrainingVector(int i, const Vector<gentype> &y,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override;

    virtual int removeTrainingVector(int i                                      ) override { SparseVector<gentype> x; gentype y; return removeTrainingVector(i,y,x); }
    virtual int removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x) override;
    virtual int removeTrainingVector(int i, int num                             ) override;

    virtual int setepsweight(int i,                double nv               ) { return getQ().setepsweight(i,nv); }
    virtual int setepsweight(const Vector<int> &i, const Vector<double> &nv) { return getQ().setepsweight(i,nv); }
    virtual int setepsweight(                      const Vector<double> &nv) { return getQ().setepsweight(  nv); }

    virtual int seteps(double xeps) override { return getQ().seteps(xeps); }

    virtual int sety(int                i, const gentype         &nv) override { dy.set(i,nv); return getQ().sety(i,nv); }
    virtual int sety(const Vector<int> &i, const Vector<gentype> &nv) override { dy.set(i,nv); return getQ().sety(i,nv); }
    virtual int sety(                      const Vector<gentype> &nv) override { dy = nv;      return getQ().sety(  nv); }

    virtual int sety(int                i, double                nv) override {                           dy("&",i) = nv;                 return getQ().sety(i,nv); }
    virtual int sety(const Vector<int> &i, const Vector<double> &nv) override { retVector<gentype> tmpva; dy("&",i,tmpva).castassign(nv); return getQ().sety(i,nv); }
    virtual int sety(                      const Vector<double> &nv) override {                           dy.castassign(nv);              return getQ().sety(  nv); }

    virtual int sety(int                i, const Vector<double>          &nv) override { int ires = getQ().sety(i,nv);                           dy.set(i,getQconst().y()(i));       return ires; }
    virtual int sety(const Vector<int> &i, const Vector<Vector<double> > &nv) override { int ires = getQ().sety(i,nv); retVector<gentype> tmpvb; dy.set(i,getQconst().y()(i,tmpvb)); return ires; }
    virtual int sety(                      const Vector<Vector<double> > &nv) override { int ires = getQ().sety(  nv);                           dy = getQconst().y();               return ires; }

    virtual int sety(int                i, const d_anion         &nv) override { int ires = getQ().sety(i,nv);                           dy.set(i,getQconst().y()(i));       return ires; }
    virtual int sety(const Vector<int> &i, const Vector<d_anion> &nv) override { int ires = getQ().sety(i,nv); retVector<gentype> tmpvb; dy.set(i,getQconst().y()(i,tmpvb)); return ires; }
    virtual int sety(                      const Vector<d_anion> &nv) override { int ires = getQ().sety(  nv);                           dy = getQconst().y();               return ires; }

    virtual int setd(int                i, int                nd) override;
    virtual int setd(const Vector<int> &i, const Vector<int> &nd) override;
    virtual int setd(                      const Vector<int> &nd) override;

    virtual int setCweight(int i,                double nv               ) override;
    virtual int setCweight(const Vector<int> &i, const Vector<double> &nv) override;
    virtual int setCweight(                      const Vector<double> &nv) override;

    virtual int setCweightfuzz(int i,                double nv               ) override { (void) i; (void) nv; NiceThrow("Weight fuzzing not available for gpr models"); return 1; }
    virtual int setCweightfuzz(const Vector<int> &i, const Vector<double> &nv) override { (void) i; (void) nv; NiceThrow("Weight fuzzing not available for gpr models"); return 1; }
    virtual int setCweightfuzz(                      const Vector<double> &nv) override {           (void) nv; NiceThrow("Weight fuzzing not available for gpr models"); return 1; }

    virtual int setsigmaweight(int i,                double nv               ) override;
    virtual int setsigmaweight(const Vector<int> &i, const Vector<double> &nv) override;
    virtual int setsigmaweight(                      const Vector<double> &nv) override;

    virtual int scaleCweight    (double s) override;
    virtual int scaleCweightfuzz(double s) override { (void) s; NiceThrow("Weight fuzzing not available for gpr models"); return 1; }
    virtual int scalesigmaweight(double s) override;

    virtual const gentype &y(int i) const override { return ( i >= 0 ) ? y()(i) : getQconst().y(i); }

    // General modification and autoset functions

    virtual int setC        (double xC)         override { return setsigma(1/xC);                          }
    virtual int setsigma    (double xsigma)     override { dsigma = xsigma; return getQ().setC(1/sigma()); }
    virtual int setsigma_cut(double xsigma_cut) override { dsigma_cut = xsigma_cut; return getQ().setsigma_cut(xsigma_cut); }
    virtual int setCclass   (int d, double xC)  override { (void) d; (void) xC; NiceThrow("Weight classing not available for gpr models"); return 1; }

    virtual int scale(double a) override;

    virtual ML_Base &operator*=(double sf) override { scale(sf); return *this; }

    virtual int scaleby(double sf) override { *this *= sf; return 1; }

    // Sampling mode

    virtual int isSampleMode(void) const override { return sampleMode; }
    virtual int setSampleMode(int nv, const Vector<gentype> &xmin, const Vector<gentype> &xmax, int Nsamp, int sampSplit, int sampType, int xsampType, double xsampScale, double sampSlack = 0) override;

    // Training functions:

    virtual int train(int &res) override { svmvolatile int killSwitch = 0; return train(res,killSwitch); }
    virtual int train(int &res, svmvolatile int &killSwitch) override { return isLocked ? 0 : getQ().train(res,killSwitch); }

    // Evaluation Functions:

    virtual int gg(               gentype &resg, const SparseVector<gentype> &x,                  const vecInfo *xinf = nullptr, gentype ***pxyprodx = nullptr) const override;
    virtual int hh(gentype &resh,                const SparseVector<gentype> &x,                  const vecInfo *xinf = nullptr, gentype ***pxyprodx = nullptr) const override;
    virtual int gh(gentype &resh, gentype &resg, const SparseVector<gentype> &x, int retaltg = 0, const vecInfo *xinf = nullptr, gentype ***pxyprodx = nullptr) const override;

    virtual int cov(gentype &resv, gentype &resmu, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo *xainf = nullptr, const vecInfo *xbinf = nullptr, gentype ***pxyprodx = nullptr, gentype ***pxyprody = nullptr, gentype **pxyprodij = nullptr) const override;

    // var and covar functions

    virtual int var(gentype &resv, gentype &resmu, const SparseVector<gentype> &xa, const vecInfo *xainf = nullptr, gentype ***pxyprodx = nullptr, gentype **pxyprodxx = nullptr) const override;

    virtual int covarTrainingVector(Matrix<gentype> &resv, const Vector<int> &i) const override { return ML_Base::covarTrainingVector(resv,i); }
    virtual int covar(Matrix<gentype> &resv, const Vector<SparseVector<gentype> > &x) const override;

    virtual int noisevar(gentype &resv, gentype &resmu, const SparseVector<gentype> &xa, const SparseVector<gentype> &xvar, int u = -1, const vecInfo *xainf = nullptr, gentype ***pxyprodx = nullptr, gentype **pxyprodxx = nullptr) const override;
    virtual int noisecov(gentype &resv, gentype &resmu, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xvar, int u = -1, const vecInfo *xainf = nullptr, const vecInfo *xbinf = nullptr, gentype ***pxyprodx = nullptr, gentype ***pxyprody = nullptr, gentype **pxyprodxy = nullptr) const override;






    // ================================================================
    //     Common functions for all GPs
    // ================================================================

    virtual       GPR_Generic &getGPR     (void)       { return *this; }
    virtual const GPR_Generic &getGPRconst(void) const { return *this; }

    // LS-SVM <-> GPR Translations
    //
    // bias  (delta) is replaced by muBias (though note that zero mean == zero bias assumed unless otherwise set)
    // alpha (gamma) is replaced by muWeight
    //
    // sigma = 1/C

    virtual int setmuWeight(const Vector<gentype> &nv) { return getQQ().setgamma(nv); }
    virtual int setmuBias  (const gentype         &nv) { return getQQ().setdelta(nv); }

    virtual int setZeromuBias(void) { return getQQ().setZerodelta(); }
    virtual int setVarmuBias (void) { return getQQ().setVardelta();  }

    virtual int setvarApproxim(const int m) { return getQQ().setvarApprox(m); }

    virtual const Vector<gentype> &muWeight(void) const { return getQQconst().gamma(); }
    virtual const gentype         &muBias  (void) const { return getQQconst().delta(); }

    virtual int isZeromuBias(void) const { return getQQconst().isZerodelta(); }
    virtual int isVarmuBias (void) const { return getQQconst().isVardelta();  }

    virtual int varApproxim(void) const { return getQQconst().varApprox(); }

    virtual const Matrix<double> &gprGp(void) const { return getQQconst().lsvGp(); }

    // Constraint handling inheritted from LSV
    //
    // naiveConstraints: if set then rather than using expectation propogation
    //       for inequality constraints and classification we instead use the
    //       naive method inheritted from SVM theory.

    virtual int isNaiveConst(void) const { return 0; }
    virtual int isEPConst   (void) const { return 1; }

    virtual int setNaiveConst(void) { return 0; }
    virtual int setEPConst   (void) { return 0; }

    // Likelihood

    virtual double loglikelihood(void) const { return getQQconst().loglikelihood(); }
    virtual double maxinfogain  (void) const { return getQQconst().maxinfogain  (); }
    virtual double RKHSnorm     (void) const { return getQQconst().RKHSnorm     (); }
    virtual double RKHSabs      (void) const { return getQQconst().RKHSabs      (); }

    // Base-level stuff
    //
    // This is overloaded by children to return correct Q type

    virtual       LSV_Generic &getQQ(void)            { return QQ; }
    virtual const LSV_Generic &getQQconst(void) const { return QQ; }

    virtual       ML_Base &getQ(void)            override { return static_cast<      ML_Base &>(getQQ());      }
    virtual const ML_Base &getQconst(void) const override { return static_cast<const ML_Base &>(getQQconst()); }




    // Grid generation, available anywhere.  Returns number of samples

    int genSampleGrid(Vector<SparseVector<gentype> > &res, const Vector<gentype> &xmin, const Vector<gentype> &xmax, int Nsamp, int sampSplit, int xsampType, double sampSlack);

    int isLocked; // set 1 to prevent further (re)training
private:

    LSV_Generic QQ;

    double dsigma;
    double dsigma_cut;
    Vector<double> dsigmaweight;
    Vector<double> dCweight;
    Vector<gentype> dy;

    // Local copy of d.  "d" as passed into lsv_generic is 0/2, d kept here
    // maintains -1,+1.  These can then be used in EP to enforce inequality
    // constraints.

    Vector<int> xd;

    // class counts

    Vector<int> Nnc; // number of vectors in each class (-1,0,+1,+2)

    // sampleMode: 0 normal
    //             1 this is a sample, so all evaluations of cov return sigma
    //               and all evaluations of g(x) are drawn from the posterior 
    //               N(g(x),cov(x,x)), then added as training data.
    //             2 this is a pre-sample.  Some pre-calculation has been done,
    //               but sample not actually taken yet.
    // sampleScale: GP(mean,sampleScale^2*cov)

    int sampleMode;
    double sampleScale;
};

inline double norm2(const GPR_Generic &a);
inline double abs2 (const GPR_Generic &a);

inline double norm2(const GPR_Generic &a) { return a.RKHSnorm(); }
inline double abs2 (const GPR_Generic &a) { return a.RKHSabs();  }

inline void qswap(GPR_Generic &a, GPR_Generic &b)
{
    a.qswapinternal(b);

    return;
}

inline void qswap(GPR_Generic *&a, GPR_Generic *&b)
{
    GPR_Generic *tmp;

    tmp = a; a = b; b = tmp;

    return;
}

inline GPR_Generic &setzero(GPR_Generic &a)
{
    a.restart();

    return a;
}

inline void GPR_Generic::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    GPR_Generic &b = dynamic_cast<GPR_Generic &>(bb.getML());

    ML_Base::qswapinternal(b);

    qswap(isLocked,b.isLocked);

    qswap(xd          ,b.xd          );
    qswap(Nnc         ,b.Nnc         );
    qswap(dsigma      ,b.dsigma      );
    qswap(dsigma_cut  ,b.dsigma_cut  );
    qswap(dsigmaweight,b.dsigmaweight);
    qswap(dy          ,b.dy          );
    qswap(dCweight    ,b.dCweight    );
    qswap(sampleMode  ,b.sampleMode  );
    qswap(sampleScale ,b.sampleScale );

    return;
}

inline void GPR_Generic::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const GPR_Generic &b = dynamic_cast<const GPR_Generic &>(bb.getMLconst());

    ML_Base::semicopy(b);

    isLocked = b.isLocked;

    xd           = b.xd;
    Nnc          = b.Nnc;
    dsigma       = b.dsigma;
    dsigma_cut   = b.dsigma_cut;
    dsigmaweight = b.dsigmaweight;
    dy           = b.dy;
    dCweight     = b.dCweight;
    sampleMode   = b.sampleMode;
    sampleScale  = b.sampleScale;

    return;
}

inline void GPR_Generic::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const GPR_Generic &src = dynamic_cast<const GPR_Generic &>(bb.getMLconst());

    ML_Base::assign(src,onlySemiCopy);

    isLocked = src.isLocked;

    xd           = src.xd;
    Nnc          = src.Nnc;
    dsigma       = src.dsigma;
    dsigma_cut   = src.dsigma_cut;
    dsigmaweight = src.dsigmaweight;
    dy           = src.dy;
    dCweight     = src.dCweight;
    sampleMode   = src.sampleMode;
    sampleScale  = src.sampleScale;

    return;
}

#endif
