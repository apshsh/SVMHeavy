
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

    virtual int  prealloc    (int expectedN)       override;
    virtual int  preallocsize(void)          const override { return getQQconst().preallocsize(); }
    virtual void setmemsize  (int memsize)         override { return getQQ().setmemsize(memsize); }

    virtual void assign       (const ML_Base &src, int onlySemiCopy = 0) override;
    virtual void semicopy     (const ML_Base &src)                       override;
    virtual void qswapinternal(ML_Base &b)                               override;

    virtual int getparam (int ind, gentype         &val, const gentype         &xa, int ia, const gentype         &xb, int ib, charptr &desc) const override;
    virtual int egetparam(int ind, Vector<gentype> &val, const Vector<gentype> &xa, int ia, const Vector<gentype> &xb, int ib               ) const override;

    virtual std::ostream &printstream(std::ostream &output, int dep) const override;
    virtual std::istream &inputstream(std::istream &input          )       override;

    virtual       ML_Base &getML     (void)       override { return static_cast<      ML_Base &>(getGPR());      }
    virtual const ML_Base &getMLconst(void) const override { return static_cast<const ML_Base &>(getGPRconst()); }

    // Information functions (training data):

    virtual int NNC(int d)    const override { return Nnc(d+1); }
    virtual int type(void)    const override { return -1;       }
    virtual int subtype(void) const override { return -1;       }

    virtual int isTrained(void) const override { return xisTrained; }
    virtual int isSolGlob(void) const override { return 1;          }

    virtual double calcDistInt(int    ha, int    hb, int ia = -1, int db = 2) const override { return ML_Base::calcDistInt(ha,hb,ia,db); }
    virtual double calcDistDbl(double ha, double hb, int ia = -1, int db = 2) const override { return ML_Base::calcDistDbl(ha,hb,ia,db); }

    virtual int isVarDefined(void) const override { return 1; }

    virtual double C         (void)  const override {                                          return 1/dsigma;           }
    virtual double sigma     (void)  const override {                                          return dsigma;             }
    virtual double sigma_cut (void)  const override {                                          return dsigma_cut;         }
    virtual double eps       (void)  const override {                                          return 0;                  }
    virtual double Cclass    (int d) const override { NiceAssert( ( d >= -1 ) && ( d <= 2 ) ); return 1/xsigmaclass(d+1); }
    virtual double sigmaclass(int d) const override { NiceAssert( ( d >= -1 ) && ( d <= 2 ) ); return xsigmaclass(d+1);   }
    virtual double epsclass  (int)   const override {                                          return 0.0;                }

    virtual const Vector<gentype>         &y          (void) const override { return dy;           }
    virtual const Vector<double>          &yR         (void) const override { return dyR;          }
    virtual const Vector<d_anion>         &yA         (void) const override { return dyA;          }
    virtual const Vector<Vector<double> > &yV         (void) const override { return dyV;          }
    virtual const Vector<int>             &d          (void) const override { return xd;           }
    virtual const Vector<double>          &Cweight    (void) const override { return dCweight;     }
    virtual const Vector<double>          &sigmaweight(void) const override { return dsigmaweight; }
    virtual const Vector<double>          &epsweight  (void) const override { static thread_local Vector<double> xxepsweight; xxepsweight.resize(N()) = 1.0; return xxepsweight;   }

    virtual const gentype        &y (int i) const override { return ( i >= 0 ) ? y ()(i) : getQQconst().y (i); }
    virtual       double          yR(int i) const override { return ( i >= 0 ) ? yR()(i) : getQQconst().yR(i); }
    virtual const d_anion        &yA(int i) const override { return ( i >= 0 ) ? yA()(i) : getQQconst().yA(i); }
    virtual const Vector<double> &yV(int i) const override { return ( i >= 0 ) ? yV()(i) : getQQconst().yV(i); }

    // Kernel tuning - you need this to jump straight to base because this needs to call GPR training (for inequalities at least), *NOT* LSV training

    virtual double tuneKernel(int method, double xwidth, int tuneK = 1, int tuneP = 0, const tkBounds *tunebounds = nullptr, paraDef *probbnd = nullptr) override { xisTrained = 0; return ML_Base::tuneKernel(method,xwidth,tuneK,tuneP,tunebounds,probbnd); }
    virtual double evalkernel(int method, const paraDef &probbnd, const Vector<double> &ffull) override { xisTrained = 0; return ML_Base::evalkernel(method,probbnd,ffull); }

    virtual int resetKernel(                             int modind = 1, int onlyChangeRowI = -1, int updateInfo = 1) override { xisTrained = 0; return getQQ().resetKernel(modind,onlyChangeRowI,updateInfo); }
    virtual int setKernel  (const MercerKernel &xkernel, int modind = 1, int onlyChangeRowI = -1                    ) override { xisTrained = 0; return getQQ().setKernel(xkernel,modind,onlyChangeRowI); }

    virtual int setKreal  (void) { xisTrained = 0; return getQQ().setKreal(); }
    virtual int setKunreal(void) { xisTrained = 0; return getQQ().setKreal(); }

    // Training set modification:

    virtual int addTrainingVector (int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2) override;
    virtual int qaddTrainingVector(int i, const gentype &y,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2) override;

    virtual int addTrainingVector (int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override;
    virtual int qaddTrainingVector(int i, const Vector<gentype> &y,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override;

    virtual int removeTrainingVector(int i                                      ) override { xisTrained = 0; SparseVector<gentype> x; gentype y; return removeTrainingVector(i,y,x); }
    virtual int removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x) override;
    virtual int removeTrainingVector(int i, int num                             ) override;

    virtual int sety(int                i, const gentype         &nv) override;
    virtual int sety(const Vector<int> &i, const Vector<gentype> &nv) override;
    virtual int sety(                      const Vector<gentype> &nv) override;

    virtual int sety(int                i, double                nv) override;
    virtual int sety(const Vector<int> &i, const Vector<double> &nv) override;
    virtual int sety(                      const Vector<double> &nv) override;

    virtual int sety(int                i, const Vector<double>          &nv) override;
    virtual int sety(const Vector<int> &i, const Vector<Vector<double> > &nv) override;
    virtual int sety(                      const Vector<Vector<double> > &nv) override;

    virtual int sety(int                i, const d_anion         &nv) override;
    virtual int sety(const Vector<int> &i, const Vector<d_anion> &nv) override;
    virtual int sety(                      const Vector<d_anion> &nv) override;

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

    virtual int setepsweight(int,                 double                ) { NiceThrow("eps not included in GP methods"); return 0; }
    virtual int setepsweight(const Vector<int> &, const Vector<double> &) { NiceThrow("eps not included in GP methods"); return 0; }
    virtual int setepsweight(                     const Vector<double> &) { NiceThrow("eps not included in GP methods"); return 0; }

    virtual int scaleCweight    (double s) override;
    virtual int scaleCweightfuzz(double s) override { (void) s; NiceThrow("Weight fuzzing not available for gpr models"); return 1; }
    virtual int scalesigmaweight(double s) override;
    virtual int scaleepsweight  (double)   override { NiceThrow("eps not included in GP methods"); return 0; }

    // General modification and autoset functions

    virtual int setC         (double xC)          override { xisTrained = 0;                                                                      return setsigma(1/xC);                  }
    virtual int setsigma     (double xsigma)      override { xisTrained = 0;                                         dsigma     = xsigma;         return getQQ().setC(1/sigma());          }
    virtual int setsigma_cut (double xsigma_cut)  override { xisTrained = 0;                                         dsigma_cut = xsigma_cut;     return getQQ().setsigma_cut(xsigma_cut); }
    virtual int seteps       (double xeps)        override { xisTrained = 0;           (void) xeps; NiceThrow("eps not included in GP methods");  return 0;                               }
    virtual int setCclass    (int d, double xC)   override { xisTrained = 0; NiceAssert( ( d >= 0 ) && ( d <= 2 ) ); xsigmaclass("&",d+1) = 1/xC; return getQQ().setCclass(d,xC);          }
    virtual int setsigmaclass(int d, double xsig) override { xisTrained = 0; NiceAssert( ( d >= 0 ) && ( d <= 2 ) ); xsigmaclass("&",d+1) = xsig; return getQQ().setCclass(d,1/xsig);      }
    virtual int setepsclass  (int d, double xeps) override { xisTrained = 0; (void) d; (void) xeps; NiceThrow("eps not included in GP methods");  return 1;                               }

    virtual int setprim  (int nv)            override { xisTrained = 0; return getQQ().setprim(nv);   }
    virtual int setprival(const gentype &nv) override { xisTrained = 0; return getQQ().setprival(nv); }
    virtual int setpriml (const ML_Base *nv) override { xisTrained = 0; return getQQ().setpriml(nv);  }

    virtual int scale  (double a) override;
    virtual int reset  (void)     override { xisTrained = 0; return getQQ().reset();   }
    virtual int restart(void)     override { xisTrained = 0; return getQQ().restart(); }
    virtual int home   (void)     override { xisTrained = 0; return getQQ().home();    }

    virtual ML_Base &operator*=(double sf) override { scale(sf); return *this; }

    virtual int scaleby(double sf) override { *this *= sf; return 1; }

    // Sampling mode

    virtual int  isSampleMode(void) const override { return sampleMode; }
    virtual int setSampleMode(int nv, const Vector<gentype> &xmin, const Vector<gentype> &xmax, int Nsamp, int sampSplit, int sampType, int xsampType, double xsampScale, double sampSlack = 0) override;

    // Training functions:

    virtual int train(int &res)                              override { svmvolatile int killSwitch = 0; return train(res,killSwitch);                        }
    virtual int train(int &res, svmvolatile int &killSwitch) override { xisTrained = 1;                 return isLocked ? 0 : getQQ().train(res,killSwitch); }

    // Likelihood

    virtual double loglikelihood(void) const override { return getQQconst().loglikelihood(); }
    virtual double maxinfogain  (void) const override { return getQQconst().maxinfogain  (); }
    virtual double RKHSnorm     (void) const override { return getQQconst().RKHSnorm     (); }
    virtual double RKHSabs      (void) const override { return getQQconst().RKHSabs      (); }

    // Evaluation Functions:

    virtual int gg(               gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = nullptr) const override { return ML_Base_Deref::gg(     resg,i,retaltg,pxyprodi); }
    virtual int hh(gentype &resh,                int i,                  gentype ***pxyprodi = nullptr) const override { return ML_Base_Deref::hh(resh,     i,        pxyprodi); }
    virtual int gh(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodx = nullptr) const override { return ML_Base_Deref::gh(resh,resg,i,retaltg,pxyprodx); }

    virtual int gg(               gentype &resg, const SparseVector<gentype> &x,                  const vecInfo *xinf = nullptr, gentype ***pxyprodx = nullptr) const override;
    virtual int hh(gentype &resh,                const SparseVector<gentype> &x,                  const vecInfo *xinf = nullptr, gentype ***pxyprodx = nullptr) const override;
    virtual int gh(gentype &resh, gentype &resg, const SparseVector<gentype> &x, int retaltg = 0, const vecInfo *xinf = nullptr, gentype ***pxyprodx = nullptr) const override;

    // var and covar functions

    virtual int predcov(gentype &resv_pred, gentype &resv, gentype &resmu, int ia, int ib, int ii, double sigmaweighti = 1.0                                                                                                                                                                           ) const { return ML_Base_Deref::predcov(resv_pred,resv,resmu,ia,ib,ii,sigmaweighti); }
    virtual int predcov(gentype &resv_pred, gentype &resv, gentype &resmu, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xx, double sigmaweighti = 1.0, const vecInfo *xainf = nullptr, const vecInfo *xbinf = nullptr, const vecInfo *xxinf = nullptr) const;

    virtual int cov(gentype &resv, gentype &resmu, int i, int j,                                                                                                                     gentype ***pxyprodx = nullptr, gentype ***pxyprody = nullptr, gentype **pxyprodij = nullptr) const override { return ML_Base_Deref::cov(resv,resmu,i,j,pxyprodx,pxyprody,pxyprodij); }
    virtual int cov(gentype &resv, gentype &resmu, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo *xainf = nullptr, const vecInfo *xbinf = nullptr, gentype ***pxyprodx = nullptr, gentype ***pxyprody = nullptr, gentype **pxyprodij = nullptr) const override;

    virtual int predvar(gentype &resv_pred, gentype &resv, gentype &resmu, int ia, int ii, double sigmaweighti = 1.0                                                                                                                  ) const override { return ML_Base::predvar(resv_pred,resv,resmu,ia,ii,sigmaweighti); }
    virtual int predvar(gentype &resv_pred, gentype &resv, gentype &resmu, const SparseVector<gentype> &xa, const SparseVector<gentype> &xx, double sigmaweighti = 1.0, const vecInfo *xainf = nullptr, const vecInfo *xxinf = nullptr) const override;

    virtual int var(gentype &resv, gentype &resmu, int i,                                                           gentype ***pxyprodi = nullptr, gentype **pxyprodii = nullptr) const override { return ML_Base::var(resv,resmu,i,pxyprodi,pxyprodii); }
    virtual int var(gentype &resv, gentype &resmu, const SparseVector<gentype> &xa, const vecInfo *xainf = nullptr, gentype ***pxyprodx = nullptr, gentype **pxyprodxx = nullptr) const override;

    virtual int covarTrainingVector(Matrix<gentype> &resv, const Vector<int> &i)                    const override { return ML_Base::covarTrainingVector(resv,i); }
    virtual int covar              (Matrix<gentype> &resv, const Vector<SparseVector<gentype> > &x) const override;

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

    virtual int setmuWeight(const Vector<gentype> &nv) { xisTrained = 0; return getQQ().setgamma(nv); }
    virtual int setmuBias  (const gentype         &nv) { xisTrained = 0; return getQQ().setdelta(nv); }

    virtual int setZeromuBias(void) { xisTrained = 0; return getQQ().setZerodelta(); }
    virtual int setVarmuBias (void) { xisTrained = 0; return getQQ().setVardelta();  }

    virtual const Vector<gentype> &muWeight(void) const { return getQQconst().gamma(); }
    virtual const gentype         &muBias  (void) const { return getQQconst().delta(); }

    virtual int isZeromuBias(void) const { return getQQconst().isZerodelta(); }
    virtual int isVarmuBias (void) const { return getQQconst().isVardelta();  }

    virtual const Matrix<double> &gprGp(void) const { return getQQconst().lsvGp(); }

    // Constraint handling inheritted from LSV
    //
    // naiveConstraints: if set then rather than using expectation propogation
    //       for inequality constraints and classification we instead use the
    //       naive method inheritted from SVM theory.

    virtual int isNaiveConst  (void) const { return 0; }
    virtual int isEPConst     (void) const { return 0; }
    virtual int isLaplaceConst(void) const { return 1; }

    virtual int setNaiveConst  (void        ) { xisTrained = 0;              return 0; }
    virtual int setEPConst     (void        ) { xisTrained = 0;              return 0; }
    virtual int setLaplaceConst(int type = 1) { xisTrained = 0; (void) type; return 0; }

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
    Vector<double> xsigmaclass;   // classwise C weights (0 = -1, 1 = zero, 2 = +1, 3 = free)
    Vector<gentype> dy;
    Vector<double> dyR;
    Vector<d_anion> dyA;
    Vector<Vector<double> > dyV;

public:
    int xisTrained;
private:

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
    qswap(xsigmaclass ,b.xsigmaclass );
    qswap(dy          ,b.dy          );
    qswap(dyR         ,b.dyR         );
    qswap(dyA         ,b.dyA         );
    qswap(dyV         ,b.dyV         );
    qswap(dCweight    ,b.dCweight    );
    qswap(sampleMode  ,b.sampleMode  );
    qswap(sampleScale ,b.sampleScale );
    qswap(xisTrained  ,b.xisTrained  );

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
    xsigmaclass  = b.xsigmaclass;
    dy           = b.dy;
    dyR          = b.dyR;
    dyA          = b.dyA;
    dyV          = b.dyV;
    dCweight     = b.dCweight;
    sampleMode   = b.sampleMode;
    sampleScale  = b.sampleScale;
    xisTrained   = b.xisTrained;

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
    xsigmaclass  = src.xsigmaclass;
    dy           = src.dy;
    dyR          = src.dyR;
    dyA          = src.dyA;
    dyV          = src.dyV;
    dCweight     = src.dCweight;
    sampleMode   = src.sampleMode;
    sampleScale  = src.sampleScale;
    xisTrained   = src.xisTrained;

    return;
}

#endif
