
//
// ML averaging block
//
// g(x) = mean(gi(x))
// gv(x) = mean(gv(x)) + var(gi(x))
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _blk_conect_h
#define _blk_conect_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "blk_generic.hpp"
#include "blk_consen.hpp"
#include "svm_scalar.hpp"
#include "idstore.hpp"


// Defines a very basic set of blocks for use in machine learning.


class BLK_Conect;


// Swap and zeroing (restarting) functions

inline void qswap(BLK_Conect &a, BLK_Conect &b);


class BLK_Conect : public BLK_Generic
{
public:

    // Assumptions: all virtual functions inherited from ML_Base are left
    // unchanged in this class.

    // Constructors, destructors, assignment etc..

    BLK_Conect(int isIndPrune = 0)                                             : BLK_Generic(isIndPrune) { setaltx(nullptr);                 return; }
    BLK_Conect(const BLK_Conect &src, int isIndPrune = 0)                      : BLK_Generic(isIndPrune) { setaltx(nullptr); assign(src,0);  return; }
    BLK_Conect(const BLK_Conect &src, const ML_Base *xsrc, int isIndPrune = 0) : BLK_Generic(isIndPrune) { setaltx(xsrc); assign(src,-1); return; }
    BLK_Conect &operator=(const BLK_Conect &src) { assign(src); return *this; }
    virtual ~BLK_Conect() { return; }

    virtual int prealloc(int expectedN) override;
    virtual int preallocsize(void) const override;
    virtual void setmemsize(int memsize) override;

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0) override;
    virtual void semicopy(const ML_Base &src) override;
    virtual void qswapinternal(ML_Base &b) override;

    virtual std::ostream &printstream(std::ostream &output, int dep) const override;
    virtual std::istream &inputstream(std::istream &input ) override;

    // Information functions

    virtual int N(void)       const override { return getRepConst().N();    }
    virtual int NNC(int d)    const override { return getRepConst().NNC(d); }
    virtual int type(void)    const override { return 212; }
    virtual int subtype(void) const override { return 0;   }

    virtual int tspaceDim(void)    const override { return getRepConst().tspaceDim();    }
    virtual int xspaceDim(void)    const override { return getRepConst().xspaceDim();    }
    virtual int fspaceDim(void)    const override { return getRepConst().fspaceDim();    }
    virtual int tspaceSparse(void) const override { return getRepConst().tspaceSparse(); }
    virtual int xspaceSparse(void) const override { return getRepConst().xspaceSparse(); }
    virtual int numClasses(void)   const override { return getRepConst().numClasses();   }
    virtual int order(void)        const override { return getRepConst().order();        }

    virtual int isTrained(void) const override { return getRepConst().isTrained(); }
    virtual int isMutable(void) const override { return getRepConst().isMutable(); }
    virtual int isPool   (void) const override { return getRepConst().isPool();    }

    virtual char gOutType(void) const override { return getRepConst().gOutType(); }
    virtual char hOutType(void) const override { return getRepConst().hOutType(); }
    virtual char targType(void) const override { return getRepConst().targType(); }
    virtual double calcDist(const gentype &ha, const gentype &hb, int ia = -1, int db = 2) const override { return getRepConst().calcDist(ha,hb,ia,db); }

    virtual int isUnderlyingScalar(void) const override { return getRepConst().isUnderlyingScalar(); }
    virtual int isUnderlyingVector(void) const override { return getRepConst().isUnderlyingVector(); }
    virtual int isUnderlyingAnions(void) const override { return getRepConst().isUnderlyingAnions(); }

    virtual const Vector<int> &ClassLabels(void)   const override { return getRepConst().ClassLabels(); }
    virtual int getInternalClass(const gentype &y) const override { return getRepConst().getInternalClass(y); }
    virtual int numInternalClasses(void)           const override { return getRepConst().numInternalClasses(); }
    virtual int isenabled(int i)                   const override { return getRepConst().isenabled(i); }

    virtual double C        (void)  const override { return getRepConst().C();         }
    virtual double sigma    (void)  const override { return getRepConst().sigma();     }
    virtual double sigma_cut(void)  const override { return getRepConst().sigma_cut(); }
    virtual double eps      (void)  const override { return getRepConst().eps();       }
    virtual double Cclass   (int d) const override { return getRepConst().Cclass(d);   }
    virtual double epsclass (int d) const override { return getRepConst().epsclass(d); }

    virtual int    memsize     (void) const override { return getRepConst().memsize();      }
    virtual double zerotol     (void) const override { return getRepConst().zerotol();      }
    virtual double Opttol      (void) const override { return getRepConst().Opttol();       }
    virtual double Opttolb     (void) const override { return getRepConst().Opttolb();      }
    virtual double Opttolc     (void) const override { return getRepConst().Opttolc();      }
    virtual double Opttold     (void) const override { return getRepConst().Opttold();      }
    virtual double lr          (void) const override { return getRepConst().lr();           }
    virtual double lrb         (void) const override { return getRepConst().lrb();          }
    virtual double lrc         (void) const override { return getRepConst().lrc();          }
    virtual double lrd         (void) const override { return getRepConst().lrd();          }
    virtual int    maxitcnt    (void) const override { return getRepConst().maxitcnt();     }
    virtual double maxtraintime(void) const override { return getRepConst().maxtraintime(); }
    virtual double traintimeend(void) const override { return getRepConst().traintimeend(); }

    virtual int    maxitermvrank(void) const override { return getRepConst().maxitermvrank(); }
    virtual double lrmvrank(void)      const override { return getRepConst().lrmvrank();      }
    virtual double ztmvrank(void)      const override { return getRepConst().ztmvrank();      }

    virtual double betarank(void) const override { return getRepConst().betarank(); }

    virtual double sparlvl(void) const;

    virtual const Vector<SparseVector<gentype> > &x          (void) const override { return getRepConst().x();           }
    virtual const Vector<gentype>                &y          (void) const override { return getRepConst().y();           }
    virtual const Vector<vecInfo>                &xinfo      (void) const override { return getRepConst().xinfo();       }
    virtual const Vector<int>                    &d          (void) const;
    virtual const Vector<double>                 &Cweight    (void) const override { return getRepConst().Cweight();     }
    virtual const Vector<double>                 &Cweightfuzz(void) const override { return getRepConst().Cweightfuzz(); }
    virtual const Vector<double>                 &sigmaweight(void) const override { return getRepConst().sigmaweight(); }
    virtual const Vector<double>                 &epsweight  (void) const override { return getRepConst().epsweight();   }
    virtual const Vector<int>                    &alphaState (void) const;

    virtual const Vector<gentype> &alphaVal(void)  const override { return getRepConst().alphaVal();  }
    virtual       double           alphaVal(int i) const override { return getRepConst().alphaVal(i); }

    virtual int isClassifier(void) const override { return getRepConst().isClassifier(); }
    virtual int isRegression(void) const override { return getRepConst().isRegression(); }

    // Kernel Modification
    //
    // (need to be implemented)

    virtual void fillCache(int Ns = 0, int Ne = -1) override;

    // Training set modification:

    virtual int addTrainingVector (int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1) override;
    virtual int qaddTrainingVector(int i, const gentype &y,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1) override;

    virtual int addTrainingVector (int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override;
    virtual int qaddTrainingVector(int i, const Vector<gentype> &y,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override;

    virtual int removeTrainingVector(int i) override;
    virtual int removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x) override;
    virtual int removeTrainingVector(int i, int num) override;

    virtual int setx(int                i, const SparseVector<gentype>          &x) override;
    virtual int setx(const Vector<int> &i, const Vector<SparseVector<gentype> > &x) override;
    virtual int setx(                      const Vector<SparseVector<gentype> > &x) override;

    virtual int qswapx(int                i, SparseVector<gentype>          &x, int dontupdate = 0) override { (void) i; (void) x; (void) dontupdate; NiceThrow("blk_connect: qswapx not implemented here."); return 1; }
    virtual int qswapx(const Vector<int> &i, Vector<SparseVector<gentype> > &x, int dontupdate = 0) override { (void) i; (void) x; (void) dontupdate; NiceThrow("blk_connect: qswapx not implemented here."); return 1; }
    virtual int qswapx(                      Vector<SparseVector<gentype> > &x, int dontupdate = 0) override {           (void) x; (void) dontupdate; NiceThrow("blk_connect: qswapx not implemented here."); return 1; }

    virtual int sety(int                i, const gentype         &y) override;
    virtual int sety(const Vector<int> &i, const Vector<gentype> &y) override;
    virtual int sety(                      const Vector<gentype> &y) override;

    virtual int sety(int                i, double                z) override;
    virtual int sety(const Vector<int> &i, const Vector<double> &z) override;
    virtual int sety(                      const Vector<double> &z) override;

    virtual int sety(int                i, const Vector<double>          &z) override;
    virtual int sety(const Vector<int> &i, const Vector<Vector<double> > &z) override;
    virtual int sety(                      const Vector<Vector<double> > &z) override;

    virtual int sety(int                i, const d_anion         &z) override;
    virtual int sety(const Vector<int> &i, const Vector<d_anion> &z) override;
    virtual int sety(                      const Vector<d_anion> &z) override;

    virtual int setd(int                i, int                nd) override;
    virtual int setd(const Vector<int> &i, const Vector<int> &nd) override;
    virtual int setd(                      const Vector<int> &nd) override;

    virtual int setCweight(int i,                double nv               ) override;
    virtual int setCweight(const Vector<int> &i, const Vector<double> &nv) override;
    virtual int setCweight(                      const Vector<double> &nv) override;

    virtual int setCweightfuzz(int i,                double nv               ) override;
    virtual int setCweightfuzz(const Vector<int> &i, const Vector<double> &nv) override;
    virtual int setCweightfuzz(                      const Vector<double> &nv) override;

    virtual int setsigmaweight(int i,                double nv               ) override;
    virtual int setsigmaweight(const Vector<int> &i, const Vector<double> &nv) override;
    virtual int setsigmaweight(                      const Vector<double> &nv) override;

    virtual int setepsweight(int i,                double nv               ) override;
    virtual int setepsweight(const Vector<int> &i, const Vector<double> &nv) override;
    virtual int setepsweight(                      const Vector<double> &nv) override;

    virtual int scaleCweight    (double s) override;
    virtual int scaleCweightfuzz(double s) override;
    virtual int scalesigmaweight(double s) override;
    virtual int scaleepsweight  (double s) override;

    virtual void assumeConsistentX  (void) override;
    virtual void assumeInconsistentX(void) override;

    virtual int isXConsistent(void)        const override { return getRepConst().isXConsistent();        }
    virtual int isXAssumedConsistent(void) const override { return getRepConst().isXAssumedConsistent(); }

    virtual void xferx(const ML_Base &xsrc) override { (void) xsrc; NiceThrow("blk_connect: xferx not implemented here."); return; }

    virtual const vecInfo &xinfo          (int i)                               const override { return getRepConst().xinfo(i);                   }
    virtual int   xtang                   (int i)                               const override { return getRepConst().xtang(i);                   }
    virtual const SparseVector<gentype> &x(int i)                               const override { return getRepConst().x(i);                       }
    virtual const SparseVector<gentype> &x(int i, int altMLid)                  const override { return getRepConst().x(i,altMLid);               }
    virtual int   xisrank                 (int i)                               const override { return getRepConst().xisrank(i);                 }
    virtual int   xisgrad                 (int i)                               const override { return getRepConst().xisgrad(i);                 }
    virtual int   xisrankorgrad           (int i)                               const override { return getRepConst().xisrankorgrad(i);           }
    virtual int   xisclass                (int i, int defaultclass, int q = -1) const override { return getRepConst().xisclass(i,defaultclass,q); }
    virtual const gentype &y              (int i)                               const override { return getRepConst().y(i);                       }

    // Generic target controls:
    //
    // (need to implement these)

    // General modification and autoset functions

    virtual int randomise(double sparsity) override;
    virtual int autoen(void) override;
    virtual int renormalise(void) override;
    virtual int realign(void) override;

    virtual int setzerotol     (double zt)            override;
    virtual int setOpttol      (double xopttol)       override;
    virtual int setOpttolb     (double xopttol)       override;
    virtual int setOpttolc     (double xopttol)       override;
    virtual int setOpttold     (double xopttol)       override;
    virtual int setlr          (double xlr)           override;
    virtual int setlrb         (double xlr)           override;
    virtual int setlrc         (double xlr)           override;
    virtual int setlrd         (double xlr)           override;
    virtual int setmaxitcnt    (int    xmaxitcnt)     override;
    virtual int setmaxtraintime(double xmaxtraintime) override;
    virtual int settraintimeend(double xtraintimeend) override;

    virtual int setmaxitermvrank(int nv) override;
    virtual int setlrmvrank(double nv) override;
    virtual int setztmvrank(double nv) override;

    virtual int setbetarank(double nv) override;

    virtual int setC(double xC) override;
    virtual int setsigma(double xC) override;
    virtual int setsigma_cut(double xC) override;
    virtual int seteps(double xC) override;
    virtual int setCclass(int d, double xC) override;
    virtual int setepsclass(int d, double xC) override;

    virtual int scale(double a) override;
    virtual int reset(void) override;
    virtual int restart(void) override;
    virtual int home(void) override;

    virtual int settspaceDim(int newdim) override;
    virtual int addtspaceFeat(int i) override;
    virtual int removetspaceFeat(int i) override;
    virtual int addxspaceFeat(int i) override;
    virtual int removexspaceFeat(int i) override;

    virtual int setsubtype(int i) override;

    virtual int setorder(int neword) override;
    virtual int addclass(int label, int epszero = 0) override;

    // Training functions:

    virtual void fudgeOn(void) override;
    virtual void fudgeOff(void) override;

    virtual int train(int &res) override { svmvolatile int killSwitch = 0; return train(res,killSwitch); }
    virtual int train(int &res, svmvolatile int &killSwitch) override;

    // Evaluation Functions:

    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = nullptr) const override;

    virtual int covTrainingVector(gentype &resv, gentype &resmu, int i, int j, gentype ***pxyprodi = nullptr, gentype ***pxyprodj = nullptr, gentype **pxyprodij = nullptr) const override;

//    virtual void dgTrainingVector(Vector<gentype>         &res, gentype        &resn, int i) const override;
//    virtual void dgTrainingVector(Vector<double>          &res, double         &resn, int i) const override { ML_Base::dgTrainingVector(res,resn,i); return; }
//    virtual void dgTrainingVector(Vector<Vector<double> > &res, Vector<double> &resn, int i) const override { ML_Base::dgTrainingVector(res,resn,i); return; }
//    virtual void dgTrainingVector(Vector<d_anion>         &res, d_anion        &resn, int i) const override { ML_Base::dgTrainingVector(res,resn,i); return; }

//    virtual void dgTrainingVector(Vector<gentype>         &res, const Vector<int> &i) const override;
//    virtual void dgTrainingVector(Vector<double>          &res, const Vector<int> &i) const override { ML_Base::dgTrainingVector(res,i); return; }
//    virtual void dgTrainingVector(Vector<Vector<double> > &res, const Vector<int> &i) const override { ML_Base::dgTrainingVector(res,i); return; }
//    virtual void dgTrainingVector(Vector<d_anion>         &res, const Vector<int> &i) const override { ML_Base::dgTrainingVector(res,i); return; }

//    virtual void stabProbTrainingVector(double  &res, int i, int p, double pnrm, int rot, double mu, double B) const override;

    virtual int gg(               gentype &resg, const SparseVector<gentype> &x,                  const vecInfo *xinf = nullptr, gentype ***pxyprodx = nullptr) const override { gentype resh; return gh(resh,resg,x,0,xinf,pxyprodx); }
    virtual int hh(gentype &resh,                const SparseVector<gentype> &x,                  const vecInfo *xinf = nullptr, gentype ***pxyprodx = nullptr) const override { gentype resg; return gh(resh,resg,x,0,xinf,pxyprodx); }
    virtual int gh(gentype &resh, gentype &resg, const SparseVector<gentype> &x, int retaltg = 0, const vecInfo *xinf = nullptr, gentype ***pxyprodx = nullptr) const override;

    virtual int cov(gentype &resv, gentype &resmu, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo *xainf = nullptr, const vecInfo *xbinf = nullptr, gentype ***pxyprodx = nullptr, gentype ***pxyprody = nullptr, gentype **pxyprodij = nullptr) const override;

//    virtual void dgX(Vector<gentype> &resx, const SparseVector<gentype> &x, const vecInfo *xinf = nullptr) const override;
//    virtual void dgX(Vector<double>  &resx, const SparseVector<gentype> &x, const vecInfo *xinf = nullptr) const override;

    virtual int gg(double &resg,         const SparseVector<gentype> &x, int retaltg = 0, const vecInfo *xinf = nullptr, gentype ***pxyprodx = nullptr) const override { (void) retaltg; gentype res; int resi = gg(res,x,xinf,pxyprodx); resg = (double)                 res; return resi; }
    virtual int gg(Vector<double> &resg, const SparseVector<gentype> &x, int retaltg = 0, const vecInfo *xinf = nullptr, gentype ***pxyprodx = nullptr) const override { (void) retaltg; gentype res; int resi = gg(res,x,xinf,pxyprodx); resg = (const Vector<double> &) res;  return resi; }
    virtual int gg(d_anion &resg,        const SparseVector<gentype> &x, int retaltg = 0, const vecInfo *xinf = nullptr, gentype ***pxyprodx = nullptr) const override { (void) retaltg; gentype res; int resi = gg(res,x,xinf,pxyprodx); resg = (const d_anion &)        res;  return resi; }

//    virtual void dg(Vector<gentype>         &res, gentype        &resn, const SparseVector<gentype> &x, const vecInfo *xinf = nullptr) const override;
//    virtual void dg(Vector<double>          &res, double         &resn, const SparseVector<gentype> &x, const vecInfo *xinf = nullptr) const override;
//    virtual void dg(Vector<Vector<double> > &res, Vector<double> &resn, const SparseVector<gentype> &x, const vecInfo *xinf = nullptr) const override;
//    virtual void dg(Vector<d_anion>         &res, d_anion        &resn, const SparseVector<gentype> &x, const vecInfo *xinf = nullptr) const override;

//    virtual void stabProb(double  &res, const SparseVector<gentype> &x, int p, double pnrm, int rot, double mu, double B) const override;

    // var and covar functions

    virtual int varTrainingVector(gentype &resv, gentype &resmu, int i, gentype ***pxyprodi = nullptr, gentype **pxyprodii = nullptr) const override { return covTrainingVector(resv,resmu,i,i,pxyprodi,pxyprodi,pxyprodii); }
    virtual int var(gentype &resv, gentype &resmu, const SparseVector<gentype> &xa, const vecInfo *xainf = nullptr, gentype ***pxyprodx = nullptr, gentype **pxyprodxx = nullptr) const override { return cov(resv,resmu,xa,xa,xainf,xainf,pxyprodx,pxyprodx,pxyprodxx); }

    virtual int covarTrainingVector(Matrix<gentype> &resv, const Vector<int> &i) const override { return ML_Base::covarTrainingVector(resv,i); }
    virtual int covar(Matrix<gentype> &resv, const Vector<SparseVector<gentype> > &x) const override { return ML_Base::covar(resv,x); }

    virtual int noisevarTrainingVector(gentype &resv, gentype &resmu, int i, const SparseVector<gentype> &xvar, int u = -1, gentype ***pxyprodi = nullptr, gentype **pxyprodii = nullptr) const override { return ML_Base::noisevarTrainingVector(resv,resmu,i,xvar,u,pxyprodi,pxyprodii); }
    virtual int noisevar(gentype &resv, gentype &resmu, const SparseVector<gentype> &xa, const SparseVector<gentype> &xvar, int u = -1, const vecInfo *xainf = nullptr, gentype ***pxyprodx = nullptr, gentype **pxyprodxx = nullptr) const override { return ML_Base::noisevar(resv,resmu,xa,xvar,u,xainf,pxyprodx,pxyprodxx); }

    virtual int noisecovTrainingVector(gentype &resv, gentype &resmu, int i, int j, const SparseVector<gentype> &xvar, int u = -1, gentype ***pxyprodi = nullptr, gentype ***pxyprodj = nullptr, gentype **pxyprodij = nullptr) const override { return ML_Base::noisecovTrainingVector(resv,resmu,i,j,xvar,u,pxyprodi,pxyprodj,pxyprodij); }
    virtual int noisecov(gentype &resv, gentype &resmu, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xvar, int u = -1, const vecInfo *xainf = nullptr, const vecInfo *xbinf = nullptr, gentype ***pxyprodx = nullptr, gentype ***pxyprody = nullptr, gentype **pxyprodxy = nullptr) const override { return ML_Base::noisecov(resv,resmu,xa,xb,xvar,u,xainf,xbinf,pxyprodx,pxyprody,pxyprodxy); }

    // Training data tracking functions:

    virtual const Vector<int>          &indKey(void)          const override { return getRepConst().indKey();          }
    virtual const Vector<int>          &indKeyCount(void)     const override { return getRepConst().indKeyCount();     }
    virtual const Vector<int>          &dattypeKey(void)      const override { return getRepConst().dattypeKey();      }
    virtual const Vector<Vector<int> > &dattypeKeyBreak(void) const override { return getRepConst().dattypeKeyBreak(); }

    // Other functions

    virtual void setaltx(const ML_Base *_altxsrc) override { NiceAssert( !_altxsrc ); (void) _altxsrc; return; }

    virtual int disable(int i) override;
    virtual int disable(const Vector<int> &i) override;

private:

    ML_Base defbase;
    BLK_Consen combit;

    int numReps(void) const
    {
        return mlqlist().indsize();
    }

    const ML_Base &getRepConst(int i = -1) const
    {
        if ( numReps() )
        { 
            return *(mlqlist().direcref( ( i >= 0 ) ? i : 0 ));
        }

        return defbase;
    }

    ML_Base &getRep(int i = -1)
    {
        if ( numReps() )
        { 
            return *(mlqlist().direcref( ( i >= 0 ) ? i : 0 ));
        }

        return defbase;
    }

    double getRepWeight(int i = -1) const
    {
        double res = 1;

        if ( numReps() )
        {
            res = (mlqweight().direcref( ( i >= 0 ) ? i : 0 ));
        }

        return res;
    }

    mutable Vector<int> dscratch;
    mutable Vector<int> alphaStateScratch;
};

inline double norm2(const BLK_Conect &a);
inline double abs2 (const BLK_Conect &a);

inline double norm2(const BLK_Conect &a) { return a.RKHSnorm(); }
inline double abs2 (const BLK_Conect &a) { return a.RKHSabs();  }

inline void qswap(BLK_Conect &a, BLK_Conect &b)
{
    a.qswapinternal(b);

    return;
}

inline void BLK_Conect::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    BLK_Conect &b = dynamic_cast<BLK_Conect &>(bb.getML());

    BLK_Generic::qswapinternal(b);

    return;
}

inline void BLK_Conect::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const BLK_Conect &b = dynamic_cast<const BLK_Conect &>(bb.getMLconst());

    BLK_Generic::semicopy(b);

    return;
}

inline void BLK_Conect::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const BLK_Conect &src = dynamic_cast<const BLK_Conect &>(bb.getMLconst());

    BLK_Generic::assign(src,onlySemiCopy);

    return;
}

#endif
