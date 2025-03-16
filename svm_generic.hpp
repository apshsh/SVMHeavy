
//
// SVM base class
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _svm_generic_h
#define _svm_generic_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "ml_base.hpp"


// NB: functions indented by 2 spaces should not be overloaded.


class SVM_Scalar;
class SVM_Binary;
template <class T> class SVM_Vector_redbin;
class SVM_Vector_atonce;
class SVM_Vector_Mredbin;
class SVM_Vector_Matonce;
template <class T> class SVM_Vector_atonce_temp;
class SVM_MultiC_redbin;
class SVM_MultiC_atonce;
class SVM_Generic;
class SVM_KConst;



class KxferData;
class KxferData
{
public:
    KxferData() { gprevxvernum = -1; gprevgvernum = -1; gprevN = 0; gprevNb = 0; return; }

    KxferData &operator=(const KxferData &src)
    {
        allxaprev                  = src.allxaprev;
        allxadirectProdsFull       = src.allxadirectProdsFull;
        allxainnerProdsFull        = src.allxainnerProdsFull;
        allxbinnerProdsFull        = src.allxbinnerProdsFull;
        allxaxbinnerProd           = src.allxaxbinnerProd;
        allxaxainnerProd           = src.allxaxainnerProd;
        allxbxbinnerProd           = src.allxbxbinnerProd;
        alldiagkernsgentype        = src.alldiagkernsgentype;
        alldiagkernsdouble         = src.alldiagkernsdouble;
        allprevxbvernum            = src.allprevxbvernum;
        allipres                   = src.allipres;

        gxvdirectProdsFull       = src.gxvdirectProdsFull;
        gxvinnerProdsFull        = src.gxvinnerProdsFull;
        gaadirectProdsFull       = src.gaadirectProdsFull;
        gprevalphaState          = src.gprevalphaState;
        gprevbalphaState         = src.gprevbalphaState;

        gprevxvernum = src.gprevxvernum;
        gprevgvernum = src.gprevgvernum;
        gprevN       = src.gprevN;
        gprevNb      = src.gprevNb;

        return *this;
    }

    SparseVector<SparseVector<gentype> > allxaprev;                     // previous xa vector (if different then need to recalculate)
    SparseVector<Matrix<SparseVector<gentype> > > allxadirectProdsFull; // pre-calculated direct products between xa, x(j) and x(k)
    SparseVector<Vector<double> >  allxainnerProdsFull;                 // pre-calculated inner products between xa, x(j)
    SparseVector<Vector<double> >  allxbinnerProdsFull;                 // pre-calculated inner products between xa, x(j)
    SparseVector<double> allxaxbinnerProd;                              // pre-calculated inner products between xa, xb
    SparseVector<double> allxaxainnerProd;                              // pre-calculated inner products between xa, xa
    SparseVector<double> allxbxbinnerProd;                              // pre-calculated inner products between xb, xb
    SparseVector<SparseVector<gentype> > alldiagkernsgentype;           // cached evaluations of K(ia,ia), ia >= 0, if done
    SparseVector<SparseVector<double>  > alldiagkernsdouble;            // cached evaluations of K(ia,ia), ia >= 0, if done
    SparseVector<int> allprevxbvernum;                                  // version number of x
    SparseVector<gentype> allipres;                                     // scrachpad for inner products

    Matrix<SparseVector<gentype> > gxvdirectProdsFull; // pre-calculated direct products between x(j) and x(k)
    Matrix<double> gxvinnerProdsFull;                  // pre-calculated inner products between x(j) and x(k)
    Matrix<double> gaadirectProdsFull;                 // pre-calculated direct products between alpha(j) and alpha(k)
    Vector<int> gprevalphaState;                       // previous alpha state (element change used when updating above caches)
    Vector<int> gprevbalphaState;                      // like above, but used in different place

    int gprevxvernum; // version number of x for evaluator
    int gprevgvernum; // version number of alpha for evaluator
    int gprevN;       // N (most recent) for evaluator
    int gprevNb;      // like above, but used in different place

    void qswapinternal(KxferData &b);
};

inline void qswap(KxferData &a, KxferData &b);

// Swap and zeroing (restarting) functions

inline void qswap(SVM_Generic &a, SVM_Generic &b);
inline void qswap(SVM_Generic *&a, SVM_Generic *&b);

inline SVM_Generic &setzero(SVM_Generic &a);

class SVM_Generic : public ML_Base
{
public:

    friend class SVM_Scalar;
    friend class SVM_Binary;
    template <class T> friend class SVM_Vector_redbin;
    friend class SVM_Vector_atonce;
    friend class SVM_Vector_Mredbin;
    friend class SVM_Vector_Matonce;
    template <class T> friend class SVM_Vector_atonce_temp;
    friend class SVM_MultiC_atonce;
    friend class SVM_MultiC_redbin;
    friend class SVM_KConst;

    // Constructors, destructors, assignment etc..

    SVM_Generic()                                            : ML_Base() { xyvalid = 0; setaltx(nullptr);                 return;       }
    SVM_Generic(const SVM_Generic &src)                      : ML_Base() { xyvalid = 0; setaltx(nullptr); assign(src,0);  return;       }
    SVM_Generic(const SVM_Generic &src, const ML_Base *srcx) : ML_Base() { xyvalid = 0; setaltx(srcx);    assign(src,-1); return;       }
    SVM_Generic &operator=(const SVM_Generic &src)                       { xyvalid = 0;                   assign(src);    return *this; }
    virtual ~SVM_Generic() { return; }

    virtual int prealloc(int expectedN) override;
    virtual int preallocsize(void) const override;
    virtual void setmemsize(int memsize) override { ML_Base::setmemsize(memsize); return; }

    virtual void assign  (const ML_Base &src, int onlySemiCopy = 0) override;
    virtual void semicopy(const ML_Base &src)                       override;
    virtual void qswapinternal(ML_Base &b) override;

    virtual int getparam (int ind, gentype         &val, const gentype         &xa, int ia, const gentype         &xb, int ib, charptr &desc) const override;
    virtual int egetparam(int ind, Vector<gentype> &val, const Vector<gentype> &xa, int ia, const Vector<gentype> &xb, int ib               ) const override;

    virtual std::ostream &printstream(std::ostream &output, int dep) const override;
    virtual std::istream &inputstream(std::istream &input ) override;

    virtual       ML_Base &getML     (void)       override { return static_cast<      ML_Base &>(getSVM());      }
    virtual const ML_Base &getMLconst(void) const override { return static_cast<const ML_Base &>(getSVMconst()); }

    // Information functions (training data):

    virtual int N(void) const override { return y().size(); }

    virtual int getInternalClass  (const gentype &y) const override;
    virtual int numInternalClasses(void)             const override { return isanomalyOn() ? numClasses()+1 : numClasses(); }

    virtual double sparlvl(void) const override { return N()-NNC(0) ? ((double) NZ()-NNC(0))/((double) N()-NNC(0)) : 1; }

    virtual const Vector<int>     &alphaState(void)  const override { return xalphaState; }
    virtual const Vector<gentype> &alphaVal  (void)  const override { return alpha();     }
    virtual double                 alphaVal  (int i) const override { return alphaR()(i); }

    virtual int isClassifier(void) const override { return 0; }
    virtual int isRegression(void) const override { return 0; }

    // Kernel transfer

    virtual int isKVarianceNZ(void) const override;

    virtual void K0xfer(                                    gentype &res, int &minmaxind, int typeis, const gentype &xyprod, const gentype &yxprod, const gentype &diffis,                                                                                                                                                                                                                                                                 int xdim, int densetype, int resmode, int mlid) const override;
    virtual void K1xfer(                                    gentype &res, int &minmaxind, int typeis, const gentype &xyprod, const gentype &yxprod, const gentype &diffis, const SparseVector<gentype> &xa,                                                                                                    const vecInfo &xainfo,                                                                      int ia,                         int xdim, int densetype, int resmode, int mlid) const override;
    virtual void K2xfer(gentype &dxyprod, gentype &ddiffis, gentype &res, int &minmaxind, int typeis, const gentype &xyprod, const gentype &yxprod, const gentype &diffis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,                                                                   const vecInfo &xainfo, const vecInfo &xbinfo,                                               int ia, int ib,                 int xdim, int densetype, int resmode, int mlid) const override;
    virtual void K3xfer(                                    gentype &res, int &minmaxind, int typeis, const gentype &xyprod, const gentype &yxprod, const gentype &diffis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc,                                  const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo,                        int ia, int ib, int ic,         int xdim, int densetype, int resmode, int mlid) const override;
    virtual void K4xfer(                                    gentype &res, int &minmaxind, int typeis, const gentype &xyprod, const gentype &yxprod, const gentype &diffis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, int ia, int ib, int ic, int id, int xdim, int densetype, int resmode, int mlid) const override;
    virtual void Kmxfer(                                    gentype &res, int &minmaxind, int typeis, const gentype &xyprod, const gentype &yxprod, const gentype &diffis, Vector<const SparseVector<gentype> *> &x, Vector<const vecInfo *> &xzinfo, Vector<int> &i, int xdim, int m, int densetype, int resmode, int mlid) const override;

    virtual void K0xfer(                                  double &res, int &minmaxind, int typeis, double xyprod, double yxprod, double diffis,                                                                                                                                                                                                                                                                 int xdim, int densetype, int resmode, int mlid) const override;
    virtual void K1xfer(                                  double &res, int &minmaxind, int typeis, double xyprod, double yxprod, double diffis, const SparseVector<gentype> &xa,                                                                                                    const vecInfo &xainfo,                                                                      int ia,                         int xdim, int densetype, int resmode, int mlid) const override;
    virtual void K2xfer(double &dxyprod, double &ddiffis, double &res, int &minmaxind, int typeis, double xyprod, double yxprod, double diffis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,                                                                   const vecInfo &xainfo, const vecInfo &xbinfo,                                               int ia, int ib,                 int xdim, int densetype, int resmode, int mlid) const override;
    virtual void K3xfer(                                  double &res, int &minmaxind, int typeis, double xyprod, double yxprod, double diffis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc,                                  const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo,                        int ia, int ib, int ic,         int xdim, int densetype, int resmode, int mlid) const override;
    virtual void K4xfer(                                  double &res, int &minmaxind, int typeis, double xyprod, double yxprod, double diffis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, int ia, int ib, int ic, int id, int xdim, int densetype, int resmode, int mlid) const override;
    virtual void Kmxfer(                                  double &res, int &minmaxind, int typeis, double xyprod, double yxprod, double diffis, Vector<const SparseVector<gentype> *> &x, Vector<const vecInfo *> &xzinfo, Vector<int> &i, int xdim, int m, int densetype, int resmode, int mlid) const override;

    // Training set modification:

    virtual int addTrainingVector (int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1) override;
    virtual int qaddTrainingVector(int i, const gentype &y,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1) override;

    virtual int addTrainingVector(int i,            double *xxa, int dima, double Cweigh = 1, double epsweigh = 1) override { return ML_Base::addTrainingVector(i,   xxa,dima,Cweigh,epsweigh); }
    virtual int addTrainingVector(int i, int zz,    double *xxa, int dima, double Cweigh = 1, double epsweigh = 1) override { return ML_Base::addTrainingVector(i,zz,xxa,dima,Cweigh,epsweigh); }
    virtual int addTrainingVector(int i, double zz, double *xxa, int dima, double Cweigh = 1, double epsweigh = 1) override { return ML_Base::addTrainingVector(i,zz,xxa,dima,Cweigh,epsweigh); }

    virtual int addTrainingVector (int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override;
    virtual int qaddTrainingVector(int i, const Vector<gentype> &y,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override;

    virtual int removeTrainingVector(int i                                      ) override { SparseVector<gentype> x; gentype y; return removeTrainingVector(i,y,x); }
    virtual int removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x) override;
    virtual int removeTrainingVector(int i, int num                             ) override;

    virtual int sety(int                i, const gentype         &y) override { return ML_Base::sety(i,y); }
    virtual int sety(const Vector<int> &i, const Vector<gentype> &y) override { return ML_Base::sety(i,y); }
    virtual int sety(                      const Vector<gentype> &y) override { return ML_Base::sety(  y); }

    virtual int sety(int                i, double                y) override { return ML_Base::sety(i,y); }
    virtual int sety(const Vector<int> &i, const Vector<double> &y) override { return ML_Base::sety(i,y); }
    virtual int sety(                      const Vector<double> &y) override { return ML_Base::sety(  y); }

    virtual int sety(int                i, const Vector<double>          &y) override { return ML_Base::sety(i,y); }
    virtual int sety(const Vector<int> &i, const Vector<Vector<double> > &y) override { return ML_Base::sety(i,y); }
    virtual int sety(                      const Vector<Vector<double> > &y) override { return ML_Base::sety(  y); }

    virtual int sety(int                i, const d_anion         &y) override { return ML_Base::sety(i,y); }
    virtual int sety(const Vector<int> &i, const Vector<d_anion> &y) override { return ML_Base::sety(i,y); }
    virtual int sety(                      const Vector<d_anion> &y) override { return ML_Base::sety(  y); }

    virtual int setd(int                i, int                d) override { (void) i; (void) d; NiceThrow("Function setd not available for this SVM type."); return 0; }
    virtual int setd(const Vector<int> &i, const Vector<int> &d) override { (void) i; (void) d; NiceThrow("Function setd not available for this SVM type."); return 0; }
    virtual int setd(                      const Vector<int> &d) override {           (void) d; NiceThrow("Function setd not available for this SVM type."); return 0; }

    // Evaluation Functions:

    virtual void dgTrainingVector(Vector<gentype>         &res, gentype        &resn, int i) const override;
    virtual void dgTrainingVector(Vector<double>          &res, double         &resn, int i) const override { ML_Base::dgTrainingVector(res,resn,i); return; }
    virtual void dgTrainingVector(Vector<Vector<double> > &res, Vector<double> &resn, int i) const override { ML_Base::dgTrainingVector(res,resn,i); return; }
    virtual void dgTrainingVector(Vector<d_anion>         &res, d_anion        &resn, int i) const override { ML_Base::dgTrainingVector(res,resn,i); return; }

    virtual void dgTrainingVector(Vector<gentype>         &res, const Vector<int> &i) const override { ML_Base::dgTrainingVector(res,i); return; }
    virtual void dgTrainingVector(Vector<double>          &res, const Vector<int> &i) const override { ML_Base::dgTrainingVector(res,i); return; }
    virtual void dgTrainingVector(Vector<Vector<double> > &res, const Vector<int> &i) const override { ML_Base::dgTrainingVector(res,i); return; }
    virtual void dgTrainingVector(Vector<d_anion>         &res, const Vector<int> &i) const override { ML_Base::dgTrainingVector(res,i); return; }

    // kerndiag override

    virtual double k2diag(int ia) const override { return kerndiag()(ia); }



    // ================================================================
    //     Common functions for all SVMs
    // ================================================================

    virtual       SVM_Generic &getSVM     (void)       { return *this; }
    virtual const SVM_Generic &getSVMconst(void) const { return *this; }

    // Constructors, destructors, assignment etc..

    virtual int setAlpha(const Vector<gentype> &newAlpha);
    virtual int setBias (const gentype         &newBias );

    virtual int setAlphaR(const Vector<double>          &newAlpha) { (void) newAlpha; NiceThrow("Function setAlphaR not available for this SVM type."); return 0; }
    virtual int setAlphaV(const Vector<Vector<double> > &newAlpha) { (void) newAlpha; NiceThrow("Function setAlphaV not available for this SVM type."); return 0; }
    virtual int setAlphaA(const Vector<d_anion>         &newAlpha) { (void) newAlpha; NiceThrow("Function setAlphaA not available for this SVM type."); return 0; }

    virtual int setBiasR(      double          newBias) { (void) newBias; NiceThrow("Function setBiasR not available for this SVM type."); return 0; }
    virtual int setBiasV(const Vector<double> &newBias) { (void) newBias; NiceThrow("Function setBiasV not available for this SVM type."); return 0; }
    virtual int setBiasA(const d_anion        &newBias) { (void) newBias; NiceThrow("Function setBiasA not available for this SVM type."); return 0; }

    // Information functions (training data):
    //
    // NB: - it is important that isenabled() returns d(i).  This is used by
    //       errortest.h and is important for the scalar case.

    virtual int NZ (void)  const { return 0;               }
    virtual int NF (void)  const { return 0;               }
    virtual int NS (void)  const { return 0;               }
    virtual int NC (void)  const { return 0;               }
    virtual int NLB(void)  const { return 0;               }
    virtual int NLF(void)  const { return 0;               }
    virtual int NUF(void)  const { return 0;               }
    virtual int NUB(void)  const { return 0;               }
    virtual int NF (int q) const { (void) q; return 0;     }
    virtual int NZ (int q) const { (void) q; return 0;     }
    virtual int NS (int q) const { (void) q; return 0;     }
    virtual int NC (int q) const { (void) q; return 0;     }
    virtual int NLB(int q) const { (void) q; return 0;     }
    virtual int NLF(int q) const { (void) q; return 0;     }
    virtual int NUF(int q) const { (void) q; return 0;     }
    virtual int NUB(int q) const { (void) q; return 0;     }

    virtual const Vector<Vector<int> > &ClassRep(void)  const { const static Vector<Vector<int> > dummy; return dummy; }
    virtual int                         findID(int ref) const { (void) ref; return 0;                                  }

    virtual int isLinearCost   (void)  const { return 0;           }
    virtual int isQuadraticCost(void)  const { return 0;           }
    virtual int is1NormCost    (void)  const { return 0;           }
    virtual int isVarBias      (void)  const { return 0;           }
    virtual int isPosBias      (void)  const { return 0;           }
    virtual int isNegBias      (void)  const { return 0;           }
    virtual int isFixedBias    (void)  const { return 0;           }
    virtual int isVarBias      (int q) const { (void) q; return 0; }
    virtual int isPosBias      (int q) const { (void) q; return 0; }
    virtual int isNegBias      (int q) const { (void) q; return 0; }
    virtual int isFixedBias    (int q) const { (void) q; return 0; }

    virtual int isNoMonotonicConstraints   (void) const { return 0; }
    virtual int isForcedMonotonicIncreasing(void) const { return 0; }
    virtual int isForcedMonotonicDecreasing(void) const { return 0; }

    virtual int isOptActive(void) const { return 0; }
    virtual int isOptSMO   (void) const { return 0; }
    virtual int isOptD2C   (void) const { return 0; }
    virtual int isOptGrad  (void) const { return 0; }

    virtual int isFixedTube (void) const { return 0; }
    virtual int isShrinkTube(void) const { return 0; }

    virtual int isRestrictEpsPos(void) const { return 0; }
    virtual int isRestrictEpsNeg(void) const { return 0; }

    virtual int isClassifyViaSVR(void) const { return 0; }
    virtual int isClassifyViaSVM(void) const { return 0; }

    virtual int is1vsA   (void) const { return 0; }
    virtual int is1vs1   (void) const { return 0; }
    virtual int isDAGSVM (void) const { return 0; }
    virtual int isMOC    (void) const { return 0; }
    virtual int ismaxwins(void) const { return 0; }
    virtual int isrecdiv (void) const { return 0; }

    virtual int isatonce(void) const { return 0; }
    virtual int isredbin(void) const { return 0; }

    virtual int isanomalyOn (void) const { return 0; }
    virtual int isanomalyOff(void) const { return 1; }

    virtual int isautosetOff         (void) const { return 0; }
    virtual int isautosetCscaled     (void) const { return 0; }
    virtual int isautosetCKmean      (void) const { return 0; }
    virtual int isautosetCKmedian    (void) const { return 0; }
    virtual int isautosetCNKmean     (void) const { return 0; }
    virtual int isautosetCNKmedian   (void) const { return 0; }
    virtual int isautosetLinBiasForce(void) const { return 0; }

    virtual double outerlr      (void) const { return MULTINORM_OUTERSTEP;     }
    virtual double outermom     (void) const { return MULTINORM_OUTERMOMENTUM; }
    virtual int    outermethod  (void) const { return MULTINORM_OUTERMETHOD;   }
    virtual double outertol     (void) const { return MULTINORM_OUTERSTEP;     }
    virtual double outerovsc    (void) const { return MULTINORM_OUTEROVSC;     }
    virtual int    outermaxitcnt(void) const { return MULTINORM_MAXITS;        }
    virtual int    outermaxcache(void) const { return MULTINORM_FULLCACHE_N;   }

    virtual       int      maxiterfuzzt(void) const { return DEFAULT_MAXITERFUZZT;                                 }
    virtual       int      usefuzzt    (void) const { return 0;                                                    }
    virtual       double   lrfuzzt     (void) const { return DEFAULT_LRFUZZT;                                      }
    virtual       double   ztfuzzt     (void) const { return DEFAULT_ZTFUZZT;                                      }
    virtual const gentype &costfnfuzzt (void) const { const static gentype temp(DEFAULT_COSTFNFUZZT); return temp; }

    virtual int m(void) const { return DEFAULT_EMM; }

    virtual double LinBiasForce (void)  const { return 0;           }
    virtual double QuadBiasForce(void)  const { return 0;           }
    virtual double LinBiasForce (int q) const { (void) q; return 0; }
    virtual double QuadBiasForce(int q) const { (void) q; return 0; }

    virtual double nu    (void) const { return 0; }
    virtual double nuQuad(void) const { return 0; }

    virtual double theta  (void) const { return 0; }
    virtual int    simnorm(void) const { return 0; }

    virtual double anomalyNu   (void) const { return 0; }
    virtual int    anomalyClass(void) const { return 0; }

    virtual double autosetCval (void) const { return 0; }
    virtual double autosetnuval(void) const { return 0; }

    virtual int    anomclass      (void) const { return +1; }
    virtual int    singmethod     (void) const { return 0;  }
    virtual double rejectThreshold(void) const { return 0;  }

    virtual int kconstWeights(void) const { return 0; }

    virtual double D    (void) const { return DEFAULT_D;     }
    virtual double E    (void) const { return DEFAULT_C;     }
    virtual int    tunev(void) const { return DEFAULT_TUNEV; }
    virtual int    pegk (void) const { return 1;             }
    virtual double minv (void) const { return 1;             }
    virtual double F    (void) const { return 0;             }
    virtual double G    (void) const { return 1;             }

    virtual const Matrix<double>          &Gp         (void)        const {             NiceThrow("Function Gp not available for this SVM type.");          return dummyMret;      }
    virtual const Matrix<double>          &XX         (void)        const {             NiceThrow("Function XX not available for this SVM type.");          return dummyMret;      }
    virtual const Vector<double>          &kerndiag   (void)        const {             NiceThrow("Function kerndiag not available for this SVM type.");    return dummyVret;      }
    virtual const Vector<Vector<double> > &getu       (void)        const {             NiceThrow("Function getu not available for this SVM type.");        return dummyVVret;     }
    virtual const gentype                 &bias       (void)        const {                                                                                 return dbias;          }
    virtual const Vector<gentype>         &alpha      (void)        const {                                                                                 return dalpha;         }
    virtual const Vector<double>          &zR         (void)        const {             NiceThrow("Function zR not available for this SVM type.");          return dummyVret;      }
    virtual const Vector<Vector<double> > &zV         (void)        const {             NiceThrow("Function zV not available for this SVM type.");          return dummyVVret;     }
    virtual const Vector<d_anion>         &zA         (void)        const {             NiceThrow("Function zA not available for this SVM type.");          return dummyAret;      }
    virtual       double                   biasR      (void)        const {             NiceThrow("Function biasR not available for this SVM type.");       return dummyRret;      }
    virtual const Vector<double>          &biasV      (int raw = 0) const { (void) raw; NiceThrow("Function biasV not available for this SVM type.");       return dummyVret;      }
    virtual const d_anion                 &biasA      (void)        const {             NiceThrow("Function biasA not available for this SVM type.");       return dummyanion; }
    virtual const Vector<double>          &alphaR     (void)        const {             NiceThrow("Function alaphR not available for this SVM type.");      return dummyVret;      }
    virtual const Vector<Vector<double> > &alphaV     (int raw = 0) const { (void) raw; NiceThrow("Function alphaV not available for this SVM type.");      return dummyVVret;     }
    virtual const Vector<d_anion>         &alphaA     (void)        const {             NiceThrow("Function alphaA not available for this SVM type.");      return dummyAret;      }

    virtual       double          zR(int i) const { (void) i; NiceThrow("Function zR not available for this SVM type.");                                    return 0;     }
    virtual const Vector<double> &zV(int i) const { (void) i; NiceThrow("Function zV not available for this SVM type."); const static Vector<double> dummy; return dummy; }
    virtual const d_anion        &zA(int i) const { (void) i; NiceThrow("Function zA not available for this SVM type."); const static d_anion dummy(0.0);   return dummy; }

    // Training set modification:
    //
    // removeNonSupports: remove all non-support vectors from training set
    // trimTrainingSet: trim training set to desired size by removing smaller alphas

    virtual int removeNonSupports(void);
    virtual int trimTrainingSet  (int maxsize);

    // General modification and autoset functions
    //
    // NB: - Class 0,-2,-3,... are reserved, all other classes may be used.
    //     - Class 0 means alpha = 0 (restricted) when using setd
    //     - Class 2 (in the regression context) means unconstrained (normal)
    //
    //
    // Autoset functions
    //
    // These functions tell the SVM to set various parameters automatically
    // based the given method.  They are persistent, in-so-far as adjusting
    // the kernel or adding/removing data will cause them to update the
    // parameters accordingly.  To turn off this feature, either use the
    // autosetOff() function (preferred) or call setC(getC()) (slow fallback
    // option).
    //
    // Cscaled:      C = Cval/N
    // CKmean:       C = mean(diag(G))
    // CKmedian:     C = median(diag(G))
    // CNKmean:      C = N*mean(diag(G))
    // CNKmedian:    C = N*median(diag(G))
    // LinBiasForce: C = Cval/(N*nuval),   eps = 0,   LinBiasForce = -Cval
    //
    // addclass: check if label is already present, and add if not
    //
    // NOTE: class 0,-2,-3,... are reserved, all other classes may be used.
    //       Class 0 means alpha = 0

    virtual int setLinearCost   (void)                   {                           NiceThrow("Function setLinearCost not available for this SVM type.");    return 0; }
    virtual int setQuadraticCost(void)                   {                           NiceThrow("Function setQuadraticCost not available for this SVM type."); return 0; }
    virtual int set1NormCost    (void)                   {                           NiceThrow("Function set1NormCost not available for this SVM type.");     return 0; }
    virtual int setVarBias      (void)                   {                           NiceThrow("Function setVarBias not available for this SVM type.");       return 0; }
    virtual int setPosBias      (void)                   {                           NiceThrow("Function setPosBias not available for this SVM type.");       return 0; }
    virtual int setNegBias      (void)                   {                           NiceThrow("Function setNegBias not available for this SVM type.");       return 0; }
    virtual int setFixedBias    (double newbias)         { (void) newbias;           NiceThrow("Function setFixedBias not available for this SVM type.");     return 0; }
    virtual int setVarBias      (int q)                  { (void) q;                 NiceThrow("Function setVarBias not available for this SVM type.");       return 0; }
    virtual int setPosBias      (int q)                  { (void) q;                 NiceThrow("Function setPosBias not available for this SVM type.");       return 0; }
    virtual int setNegBias      (int q)                  { (void) q;                 NiceThrow("Function setNegBias not available for this SVM type.");       return 0; }
    virtual int setFixedBias    (int q, double newbias)  { (void) q; (void) newbias; NiceThrow("Function setFixedBias not available for this SVM type.");     return 0; }
    virtual int setFixedBias    (const gentype &newBias);

    virtual int setNoMonotonicConstraints   (void) { NiceThrow("Function setNoMonotonicConstraints not available for this SVM type.");    return 0; }
    virtual int setForcedMonotonicIncreasing(void) { NiceThrow("Function setForcedMonotonicIncreasing not available for this SVM type."); return 0; }
    virtual int setForcedMonotonicDecreasing(void) { NiceThrow("Function setForcedMonotonicDecreasing not available for this SVM type."); return 0; }

    virtual int setOptActive(void) { NiceThrow("Function setOptActive not available for this SVM type."); return 0; }
    virtual int setOptSMO   (void) { NiceThrow("Function setOptSMO not available for this SVM type.");    return 0; }
    virtual int setOptD2C   (void) { NiceThrow("Function setOptD2C not available for this SVM type.");    return 0; }
    virtual int setOptGrad  (void) { NiceThrow("Function setOptGrad not available for this SVM type.");   return 0; }

    virtual int setFixedTube (void) { NiceThrow("Function setFixedTube not available for this SVM type.");  return 0; }
    virtual int setShrinkTube(void) { NiceThrow("Function setShrinkTube not available for this SVM type."); return 0; }

    virtual int setRestrictEpsPos(void) { NiceThrow("Function setRestrictEpsPos not available for this SVM type."); return 0; }
    virtual int setRestrictEpsNeg(void) { NiceThrow("Function setRestrictEpsNeg not available for this SVM type."); return 0; }

    virtual int setClassifyViaSVR(void) { NiceThrow("Function setClassifyViaSVR not available for this SVM type."); return 0; }
    virtual int setClassifyViaSVM(void) { NiceThrow("Function setClassifyViaSVM not available for this SVM type."); return 0; }

    virtual int set1vsA   (void) { NiceThrow("Function set1vsA not available for this SVM type.");    return 0; }
    virtual int set1vs1   (void) { NiceThrow("Function set1vs1 not available for this SVM type.");    return 0; }
    virtual int setDAGSVM (void) { NiceThrow("Function setDAGSVM not available for this SVM type.");  return 0; }
    virtual int setMOC    (void) { NiceThrow("Function setMOC not available for this SVM type.");     return 0; }
    virtual int setmaxwins(void) { NiceThrow("Function setmaxwins not available for this SVM type."); return 0; }
    virtual int setrecdiv (void) { NiceThrow("Function setrecdiv not available for this SVM type.");  return 0; }

    virtual int setatonce(void) { NiceThrow("Function setatonce not available for this SVM type."); return 0; }
    virtual int setredbin(void) { NiceThrow("Function setredbin not available for this SVM type."); return 0; }

    virtual int anomalyOn (int danomalyClass, double danomalyNu) { (void) danomalyClass; (void) danomalyNu; NiceThrow("Function anomalyOn not available for this SVM type.");  return 0; }
    virtual int anomalyOff(void)                                 {                                          NiceThrow("Function anomalyOff not available for this SVM type."); return 0; }

    virtual int setouterlr      (double xouterlr)     { (void) xouterlr;        NiceThrow("Function setouterlr not available for this SVM type.");       return 0; }
    virtual int setoutermom     (double xoutermom)    { (void) xoutermom;       NiceThrow("Function setoutermom not available for this SVM type.");      return 0; }
    virtual int setoutermethod  (int xoutermethod)    { (void) xoutermethod;    NiceThrow("Function setoutermethod not available for this SVM type.");   return 0; }
    virtual int setoutertol     (double xoutertol)    { (void) xoutertol;       NiceThrow("Function setoutertol not available for this SVM type.");      return 0; }
    virtual int setouterovsc    (double xouterovsc)   { (void) xouterovsc;      NiceThrow("Function setouterovsc not available for this SVM type.");     return 0; }
    virtual int setoutermaxitcnt(int xoutermaxits)    { (void) xoutermaxits;    NiceThrow("Function setoutermaxitcnt not available for this SVM type."); return 0; }
    virtual int setoutermaxcache(int xoutermaxcacheN) { (void) xoutermaxcacheN; NiceThrow("Function setoutermaxcache not available for this SVM type."); return 0; }

    virtual int setmaxiterfuzzt(int xmaxiterfuzzt)               { (void) xmaxiterfuzzt; NiceThrow("Function setmaxiterfuzzt not available for this SVM type."); return 0; }
    virtual int setusefuzzt    (int xusefuzzt)                   { if ( xusefuzzt ) { NiceThrow("Function setusefuzzt not available for this SVM type."); }      return 0; }
    virtual int setlrfuzzt     (double xlrfuzzt)                 { (void) xlrfuzzt;      NiceThrow("Function setlrfuzzt not available for this SVM type.");      return 0; }
    virtual int setztfuzzt     (double xztfuzzt)                 { (void) xztfuzzt;      NiceThrow("Function setztfuzzt not available for this SVM type.");      return 0; }
    virtual int setcostfnfuzzt (const gentype &xcostfnfuzzt)     { (void) xcostfnfuzzt;  NiceThrow("Function setcostfnfuzzt not available for this SVM type.");  return 0; }
    virtual int setcostfnfuzzt (const std::string &xcostfnfuzzt) { (void) xcostfnfuzzt;  NiceThrow("Function setcostfnfuzzt not available for this SVM type.");  return 0; }

    virtual int setm(int xm) { (void) xm; NiceThrow("Function setm not available for this SVM type."); return 0; }

    virtual int setLinBiasForce (double newval)        { (void) newval;           NiceThrow("Function setLinBiasForce not available for this SVM type.");  return 0; }
    virtual int setQuadBiasForce(double newval)        { (void) newval;           NiceThrow("Function setQuadBiasForce not available for this SVM type."); return 0; }
    virtual int setLinBiasForce (int q, double newval) { (void) q; (void) newval; NiceThrow("Function setLinBiasForce not available for this SVM type.");  return 0; }
    virtual int setQuadBiasForce(int q, double newval) { (void) q; (void) newval; NiceThrow("Function setQuadBiasForce not available for this SVM type."); return 0; }

    virtual int setnu    (double xnu)     { (void) xnu;     NiceThrow("Function setnu not available for this SVM type.");     return 0; }
    virtual int setnuQuad(double xnuQuad) { (void) xnuQuad; NiceThrow("Function setnuQuad not available for this SVM type."); return 0; }

    virtual int settheta  (double nv) { (void) nv; NiceThrow("Function settheta not availble for this SVM type."); return 0; }
    virtual int setsimnorm(int    nv) { (void) nv; NiceThrow("Function setsimnorm not availble for this SVM type."); return 0; }

    virtual int autosetOff         (void)                            {                            NiceThrow("Function autosetOff not available for this SVM type.");          return 0; }
    virtual int autosetCscaled     (double Cval)                     { (void) Cval;               NiceThrow("Function autosetCscaled not available for this SVM type.");      return 0; }
    virtual int autosetCKmean      (void)                            {                            NiceThrow("Function autosetCKmean not available for this SVM type.");       return 0; }
    virtual int autosetCKmedian    (void)                            {                            NiceThrow("Function autosetCKmedian not available for this SVM type.");     return 0; }
    virtual int autosetCNKmean     (void)                            {                            NiceThrow("Function autosetCNKmean not available for this SVM type.");      return 0; }
    virtual int autosetCNKmedian   (void)                            {                            NiceThrow("Function ausosetCNKmedian not available for this SVM type.");    return 0; }
    virtual int autosetLinBiasForce(double nuval, double Cval = 1.0) { (void) nuval; (void) Cval; NiceThrow("Function autosetLinBiasForce not available for this SVM type."); return 0; }

    virtual void setanomalyclass   (int n)     { (void) n;  NiceThrow("Function setanomalyclass not available for this SVM type.");    return; }
    virtual void setsingmethod     (int nv)    { (void) nv; NiceThrow("Function setsingmethod not available for this SVM type.");      return; }
    virtual void setRejectThreshold(double nv) { (void) nv; NiceThrow("Function setRejectThreshold not available for this SVM type."); return; }

    virtual int setkconstWeights(int nv) { (void) nv; NiceThrow("Function kconstWeights not available for this SVM type."); return 0; }

    virtual int setD     (double nv) { (void) nv; return 0; }
    virtual int setE     (double nv) { (void) nv; return 0; }
    virtual int setF     (double nv) { (void) nv; return 0; }
    virtual int setG     (double nv) { (void) nv; return 0; }
    virtual int setsigmaD(double nv) { return setD(1/nv);   }
    virtual int setsigmaE(double nv) { return setE(1/nv);   }
    virtual int setsigmaF(double nv) { return setF(1/nv);   }
    virtual int setsigmaG(double nv) { return setG(1/nv);   }
    virtual int settunev (int    nv) { (void) nv; return 0; }
    virtual int setpegk  (int    nv) { (void) nv; return 0; }
    virtual int setminv  (double nv) { (void) nv; return 0; }

    // Pre-training funciton

    virtual int pretrain(void) { return 0; }



protected:
    // ================================================================
    //     Base level functions
    // ================================================================

    // SVM specific

    virtual void basesetalpha(const Vector<gentype> &newalpha) { incgvernum(); dalpha = newalpha;  return; }

    virtual void basesetbias (const gentype         &newbias ) { incgvernum(); dbias  = newbias;   return; }
    virtual void basesetbias (      double           newbias ) { incgvernum(); dbias  = newbias;   return; }
    virtual void basesetbias (const d_anion         &newbias ) { incgvernum(); dbias  = newbias;   return; }
    virtual void basesetbias (const Vector<double>  &newbias ) { incgvernum(); dbias  = newbias;   return; }

    virtual void basesetalpha(int i, const gentype        &newalpha) { incgvernum(); dalpha("&",i) = newalpha; return; }
    virtual void basesetalpha(int i,       double          newalpha) { incgvernum(); dalpha("&",i) = newalpha; return; }
    virtual void basesetalpha(int i, const d_anion        &newalpha) { incgvernum(); dalpha("&",i) = newalpha; return; }
    virtual void basesetalpha(int i, const Vector<double> &newalpha) { incgvernum(); dalpha("&",i) = newalpha; return; }

    virtual void basescalealpha(double a) { incgvernum(); dalpha.scale(a); return; }
    virtual void basescalebias (double a) { incgvernum(); dbias *= a;      return; }

    virtual void basesetAlphaBiasFromAlphaBiasR(void);
    virtual void basesetAlphaBiasFromAlphaBiasV(void);
    virtual void basesetAlphaBiasFromAlphaBiasA(void);

    // General modification and autoset functions

//    virtual void setN (int newN) { (void) newN; NiceThrow("Function setN undefined for SVM.");  return; }

    // Don't use this!

    virtual int grablinbfq(void) const { return -1; }

    // Because LS-SVM needs to access this

    virtual const Vector<double> &diagoffset(void) const { NiceThrow("Function diagoffset not available for this SVM type.");  const static Vector<double> dummy; return dummy; }

    // Kernel cache selective access for gradient calculation

    virtual double getvalIfPresent_v(int numi, int numj, int &isgood) const override { return ML_Base::getvalIfPresent_v(numi,numj,isgood); }

    // Inner-product cache: over-write this with a non-nullptr return in classes where
    // a kernel cache is available

    mutable int xyvalid; // also mlid of caller

    virtual int isxymat(const MercerKernel &altK) const override { (void) altK; return 0; }
    virtual const Matrix<double> &getxymat(const MercerKernel &altK) const override { (void) altK; NiceThrow("xymat not available here!"); const static Matrix<double> dummy; return dummy; }
    virtual const double &getxymatelm(const MercerKernel &altK, int i, int j) const override { return getxymat(altK)(i,j); }

    // Cached access to K matrix
public:
    virtual double getKval(int i, int j) const { (void) i; (void) j; NiceThrow("getKval not defined here"); return 0.0; }

    // Evaluation of 8x6 kernels

    virtual void fastg(gentype &res) const override;
    virtual void fastg(gentype &res, int ia, const SparseVector<gentype> &xa, const vecInfo &xainfo) const override;
    virtual void fastg(gentype &res, int ia, int ib, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo) const override;
    virtual void fastg(gentype &res, int ia, int ib, int ic, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo) const override;
    virtual void fastg(gentype &res, int ia, int ib, int ic, int id, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo) const override;
    virtual void fastg(gentype &res, Vector<int> &ia, Vector<const SparseVector<gentype> *> &xa, Vector<const vecInfo *> &xainfo) const override;

    virtual void fastg(double &res) const override;
    virtual void fastg(double &res, int ia, const SparseVector<gentype> &xa, const vecInfo &xainfo) const override;
    virtual void fastg(double &res, int ia, int ib, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo) const override;
    virtual void fastg(double &res, int ia, int ib, int ic, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo) const override;
    virtual void fastg(double &res, int ia, int ib, int ic, int id, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo) const override;
    virtual void fastg(double &res, Vector<int> &ia, Vector<const SparseVector<gentype> *> &xa, Vector<const vecInfo *> &xainfo) const override;

private:

    Vector<gentype> dalpha;
    gentype dbias;
    Vector<int> xalphaState;

    const gentype &alpha(int i, int j, const SparseVector<gentype> &dummy) const { (void) dummy; return alpha()(i)(j);                  }
          double   alpha(int i, int j, const SparseVector<double>  &dummy) const { (void) dummy; return (alpha()(i)(j)).cast_double(1); }
    const gentype &alpha(int i, int j, const Vector<gentype>       &dummy) const { (void) dummy; return alpha()(i)(j);                  }
          double   alpha(int i, int j, const Vector<double>        &dummy) const { (void) dummy; return (alpha()(i)(j)).cast_double(1); }

    // Cached stuff for accelerating evaluation of K2xfer

    mutable KxferData KxferDatStore;

    // ...

    Matrix<double> dummyMret;
    Vector<double> dummyVret;
    Vector<Vector<double> > dummyVVret;
    double dummyRret;
    d_anion dummyanion;
    Vector<d_anion> dummyAret;
};

inline double norm2(const SVM_Generic &a);
inline double abs2 (const SVM_Generic &a);

inline double norm2(const SVM_Generic &a) { return a.RKHSnorm(); }
inline double abs2 (const SVM_Generic &a) { return a.RKHSabs();  }

inline void qswap(KxferData &a, KxferData &b)
{
    a.qswapinternal(b);

    return;
}

inline void KxferData::qswapinternal(KxferData &b)
{
    qswap(allxaprev                 ,b.allxaprev                 );
    qswap(allxadirectProdsFull      ,b.allxadirectProdsFull      );
    qswap(allxainnerProdsFull       ,b.allxainnerProdsFull       );
    qswap(allxbinnerProdsFull       ,b.allxbinnerProdsFull       );
    qswap(allxaxbinnerProd          ,b.allxaxbinnerProd          );
    qswap(allxaxainnerProd          ,b.allxaxainnerProd          );
    qswap(allxbxbinnerProd          ,b.allxbxbinnerProd          );
    qswap(alldiagkernsgentype       ,b.alldiagkernsgentype       );
    qswap(alldiagkernsdouble        ,b.alldiagkernsdouble        );
    qswap(allprevxbvernum           ,b.allprevxbvernum           );
    qswap(allipres                  ,b.allipres                  );

    qswap(gxvdirectProdsFull      ,b.gxvdirectProdsFull);
    qswap(gxvinnerProdsFull       ,b.gxvinnerProdsFull );
    qswap(gaadirectProdsFull      ,b.gaadirectProdsFull);
    qswap(gprevalphaState         ,b.gprevalphaState   );
    qswap(gprevbalphaState        ,b.gprevbalphaState  );

    qswap(gprevxvernum,b.gprevxvernum);
    qswap(gprevgvernum,b.gprevgvernum);
    qswap(gprevN      ,b.gprevN      );
    qswap(gprevNb     ,b.gprevNb     );

    return;
}


inline void qswap(SVM_Generic &a, SVM_Generic &b)
{
    a.qswapinternal(b);

    return;
}

inline void qswap(SVM_Generic *&a, SVM_Generic *&b)
{
    SVM_Generic *temp;

    temp = a;
    a = b;
    b = temp;

    return;
}

inline SVM_Generic *&setident (SVM_Generic *&a) { NiceThrow("Whatever"); return a; }
inline SVM_Generic *&setzero  (SVM_Generic *&a) { return a = nullptr; }
inline SVM_Generic *&setposate(SVM_Generic *&a) { return a; }
inline SVM_Generic *&setnegate(SVM_Generic *&a) { NiceThrow("I reject your reality and substitute my own"); return a; }
inline SVM_Generic *&setconj  (SVM_Generic *&a) { NiceThrow("Mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm"); return a; }
inline SVM_Generic *&setrand  (SVM_Generic *&a) { NiceThrow("Blippity Blappity Blue"); return a; }
inline SVM_Generic *&postProInnerProd(SVM_Generic *&a) { return a; }

inline SVM_Generic &setzero(SVM_Generic &a)
{
    a.restart();

    return a;
}

inline void SVM_Generic::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    SVM_Generic &b = dynamic_cast<SVM_Generic &>(bb.getML());

    ML_Base::qswapinternal(b);

    qswap(dalpha     ,b.dalpha     );
    qswap(dbias      ,b.dbias      );
    qswap(xalphaState,b.xalphaState);

    qswap(xyvalid,b.xyvalid);

    return;
}

inline void SVM_Generic::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const SVM_Generic &b = dynamic_cast<const SVM_Generic &>(bb.getMLconst());

    ML_Base::semicopy(b);

    dalpha      = b.dalpha;
    dbias       = b.dbias;
    xalphaState = b.xalphaState;

    xyvalid = b.xyvalid;

    incgvernum();

    return;
}

inline void SVM_Generic::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const SVM_Generic &src = dynamic_cast<const SVM_Generic &>(bb.getMLconst());

    ML_Base::assign(src,onlySemiCopy);

    dalpha      = src.dalpha;
    dbias       = src.dbias;
    xalphaState = src.xalphaState;

    xyvalid       = src.xyvalid;
    KxferDatStore = KxferDatStore;

    return;
}



#endif
