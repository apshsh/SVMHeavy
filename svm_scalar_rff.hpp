
 //FIXME: different bias for each task, split inner-loop training into tasks and do in sequence, setting relevant bias part.  Also setbias needs to use setBiasV

//FIXME: add generate vector function to mercer

//
// Scalar Random-Fourier-Features SVM
//
// Version: 7
// Date: 23/02/2021
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//


//
// Method: we inherit from SVM_Scalar.  Data is stored as per
// usual, followed by random features, so:
//
// Gp = [ ... Qc' Qs' ]   ( Kij, followed by the feature map [ Qc' Qs' ] of xi under RFF)
//      [ Qc  ... ... ]
//      [ Qs  ... ... ]
//
// where Qc_ij = sqrt(2/M)*cos(<omega_i,x_j>)
// where Qs_ij = sqrt(2/M)*sin(<omega_i,x_j>)
//
// where kernel calculation is over-ridden to calculate Qc and Qs
// correctly.
//
// The training data (first N points) are never marked active,
// so these parts of the Gp matrix are never calculated, and
// hence that memory is never actually used or allocated in any
// way.  Training is extricated to this label and involves
// calculating alpha for random features part only.  Most
// functionality can therefore be passed back to the SVM_Scalar
// layer and should work just fine, with some minor adjustments
// at this level as documented in code.
//

#ifndef _svm_scalar_rff_h
#define _svm_scalar_rff_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "svm_scalar.hpp"
#include "adam.hpp"



// Very rough version of scalar SVM with RFF.  Note that
// there are a lot of things not finished here!  Generally
// leave the default quadratic cost, epsilon = 0 (default),
// only equality constraints, and it should work fine.


class SVM_Scalar_rff;
class SVM_Binary_rff;


// Swap function

inline void qswap(SVM_Scalar_rff &a, SVM_Scalar_rff &b);

#define INDIM ( ( locReOnly ? 1 : 2 ) * ( ( locNRffRep > 0 ) ? locNRffRep : 1 ) )


class SVM_Scalar_rff : public SVM_Scalar
{
    friend class SVM_Binary_rff;

public:

    // Constructors, destructors, assignment etc..

    SVM_Scalar_rff();
    SVM_Scalar_rff(const SVM_Scalar_rff &src);
    SVM_Scalar_rff(const SVM_Scalar_rff &src, const ML_Base *xsrc);
    SVM_Scalar_rff &operator=(const SVM_Scalar_rff &src) { assign(src); return *this; }
    virtual ~SVM_Scalar_rff();

    virtual void assign       (const ML_Base &src, int onlySemiCopy = 0) override;
    virtual void semicopy     (const ML_Base &src)                       override;
    virtual void qswapinternal(ML_Base &b)                               override;

    virtual int prealloc(int expectedN) override;

    virtual int restart(void) override { SVM_Scalar_rff temp; *this = temp; return 1; }

    virtual std::ostream &printstream(std::ostream &output, int dep) const override;
    virtual std::istream &inputstream(std::istream &input) override;

    // Information functions (training data):

    virtual int  N       (void) const override { return locN; }
    virtual int  type    (void) const override { return 22;   }
    virtual int  subtype (void) const override { return 0;    }
    virtual char targType(void) const override { return 'R';  }

    virtual int RFFordata(int i) const { return ( i >= locN ) ? 1 : 0; }

    // Technically speaking this should be 1, however, as we are using
    // gradient descent the previous solution sometimes "persists" in the
    // new solution, particularly if the algorithm is stopped early due
    // to time/iteration limits.  Thus it is preferable to start fresh
    // each time to avoid unexpected surprises.
    virtual int isSolGlob(void) const override { return 0; }

    virtual double C       (void)  const override { return locC;             }
    virtual double eps     (void)  const override { return loceps;           }
    virtual double Cclass  (int d) const override { return locCclass(d+1);   }
    virtual double epsclass(int d) const override { return locepsclass(d+1); }

    virtual double Opttol (void) const override { return locOpttol;  }
    virtual double Opttolb(void) const override { return locOpttolb; }

    virtual double lr (void) const override { return loclr;  }
    virtual double lrb(void) const override { return loclrb; }

    // These must be wrong size.  Polymorphism will be called by ml_base to access inequalities
    virtual const Vector<gentype>         &y          (void) const override { return locz;           }
    virtual const Vector<double>          &yR         (void) const override { return loczr;          }
    virtual const Vector<d_anion>         &yA         (void) const override { static thread_local Vector<d_anion>         dummy; NiceThrow("yA not defined in svm_scalar_rff"); return dummy; }
    virtual const Vector<Vector<double> > &yV         (void) const override { static thread_local Vector<Vector<double> > dummy; NiceThrow("yV not defined in svm_scalar_rff"); return dummy; }
    virtual const Vector<double>          &Cweight    (void) const override { return locCweight;     }
    virtual const Vector<double>          &epsweight  (void) const override { return locepsweight;   }
    virtual const Vector<double>          &Cweightfuzz(void) const override { return locCweightfuzz; }

    // Kernel transfer

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

    // Kernel Modification

    virtual int reset(void) override;

    virtual int resetKernel(                             int modind = 1, int onlyChangeRowI = -1, int updateInfo = 1) override;
    virtual int setKernel  (const MercerKernel &xkernel, int modind = 1, int onlyChangeRowI = -1)                     override;

    virtual void fillCache(int Ns = 0, int Ne = -1) override;

    virtual gentype &K2(gentype &res, int ia, int ib                          , const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, int resmode = 0) const override;
    virtual gentype &K2(gentype &res, int ia, int ib, const gentype &bias     , const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, int resmode = 0) const override;
    virtual gentype &K2(gentype &res, int ia, int ib, const MercerKernel &altK, const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, int resmode = 0) const override;
    virtual double   K2(              int ia, int ib                          , const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, int resmode = 0) const override;

    // Training set modification:

    virtual int addTrainingVector (int i, double z, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2) override;
    virtual int qaddTrainingVector(int i, double z,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2) override;

    virtual int addTrainingVector (int i, const Vector<double> &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<int> &d) override;
    virtual int qaddTrainingVector(int i, const Vector<double> &z,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<int> &d) override;

    virtual int addTrainingVector (int i, const gentype &z, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1) override;
    virtual int qaddTrainingVector(int i, const gentype &z,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1) override;

    virtual int addTrainingVector (int i, const Vector<gentype> &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override;
    virtual int qaddTrainingVector(int i, const Vector<gentype> &z,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override;

    virtual int removeTrainingVector(int i                                      ) override;
    virtual int removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x) override;
    virtual int removeTrainingVector(int i, int num                             ) override;

    virtual int setx(int                i, const SparseVector<gentype>          &x) override;
    virtual int setx(const Vector<int> &i, const Vector<SparseVector<gentype> > &x) override;
    virtual int setx(                      const Vector<SparseVector<gentype> > &x) override;

    virtual int sety(int                i, double                z) override;
    virtual int sety(const Vector<int> &i, const Vector<double> &z) override;
    virtual int sety(                      const Vector<double> &z) override;

    virtual int sety(int                i, const gentype         &z) override;
    virtual int sety(const Vector<int> &i, const Vector<gentype> &z) override;
    virtual int sety(                      const Vector<gentype> &z) override;

    virtual int setd(int                i, int                d) override;
    virtual int setd(const Vector<int> &i, const Vector<int> &d) override;
    virtual int setd(                      const Vector<int> &d) override;

    virtual int setCweight(int i,                double nv               ) override;
    virtual int setCweight(const Vector<int> &i, const Vector<double> &nv) override;
    virtual int setCweight(                      const Vector<double> &nv) override;

    virtual int setepsweight(int i,                double nv               ) override;
    virtual int setepsweight(const Vector<int> &i, const Vector<double> &nv) override;
    virtual int setepsweight(                      const Vector<double> &nv) override;

    virtual int setCweightfuzz(int i,                double nv               ) override;
    virtual int setCweightfuzz(const Vector<int> &i, const Vector<double> &nv) override;
    virtual int setCweightfuzz(                      const Vector<double> &nv) override;

    virtual int scaleCweight    (double s) override;
    virtual int scaleepsweight  (double s) override;
    virtual int scaleCweightfuzz(double s) override;

    // General modification and autoset functions

    virtual int setOpttol (double xopttol) override { locOpttol  = xopttol; locisTrained = 0; cholscratchcov  = 0; covlogdetcalced = 0; return 0; }
    virtual int setOpttolb(double xopttol) override { locOpttolb = xopttol; locisTrained = 0; cholscratchcov  = 0; covlogdetcalced = 0; return 0; }

    virtual int setlr (double xlr) override { loclr  = xlr; return 0; }
    virtual int setlrb(double xlr) override { loclrb = xlr; return 0; }

    virtual int setC     (double nv)        override;
    virtual int setCclass(int d, double nv) override;

    virtual int seteps     (double nv)        override;
    virtual int setepsclass(int d, double nv) override;

    // Training functions:

    virtual int train(int &res, svmvolatile int &killSwitch) override;
    virtual int train(int &res)                              override { svmvolatile int killSwitch = 0; return train(res,killSwitch); }








    // Bias related stuff (biased is stored locally)

    virtual int isVarBias  (void)  const override { return biastype == 0; }
    virtual int isFixedBias(void)  const override { return biastype == 3; }

    virtual int isVarBias  (int q) const override { return ( ( q == -1 ) || ( q == 0 ) ) && isVarBias();   }
    virtual int isFixedBias(int q) const override { return ( ( q == -1 ) || ( q == 0 ) ) && isFixedBias(); }

    virtual int setVarBias  (void)      override { biastype = 0;               return 1; }
    virtual int setFixedBias(double nv) override { biastype = 2; locbias = nv; return 1; }

    virtual int setVarBias  (int q)             override { (void) q; NiceAssert( q == -1 ); return setVarBias();     }
    virtual int setFixedBias(int q, double nv)  override { (void) q; NiceAssert( q == -1 ); return setFixedBias(nv); }

    virtual int setFixedBias(const gentype &nv) override { return setFixedBias((double) nv); }

    virtual int setBiasR(      double          nv) override { locbias = nv; return 1; }
    virtual int setBiasV(const Vector<double> &nv) override { locbias = nv; return 1; }

    virtual const gentype &bias(void) const override { if ( !locNRffRep ) { tempgbias.force_double() = locbias(0); } else { tempgbias = locbias; } return tempgbias;  }

    virtual       double          biasR(void)    const override { return locbias(0); }
    virtual const Vector<double> &biasV(int = 0) const override { return locbias;    }










    // Information functions (training data):
    //
    // NRff:    number of random features
    // NRffRep: number of tasks being completed
    // ReOnly:  0 means features are [ sin cos ], 1 means features are [ cos ]
    // inAdam:  Inner loop method
    //          0 = direct matrix inversion
    //          1 = adam
    //          2 = adam with 2 hotstart
    //          3 = stochastic adam
    //          4 = gradient
    //          5 = stochastic gradient
    //          6 = direct matrix inversion using offNaiveDiagChol
    //          7 = pegasos training
    //          8 = cheats method using standard SVM training with approximated kernel
    // outGrad: 0,2 means optimise outer loop for v^{-1}, 1,3 means optimise for v (1 uses independent chol, 3 borrows chol from svm_scalar)

    virtual int isLinearCost(void)    const override { return costtype == 0; }
    virtual int isQuadraticCost(void) const override { return costtype == 1; }
    virtual int is1NormCost(void)     const override { return 0;             }

    virtual int isOptActive(void) const override { return 0; }
    virtual int isOptSMO   (void) const override { return 0; }
    virtual int isOptD2C   (void) const override { return 0; }
    virtual int isOptGrad  (void) const override { return 1; }

    virtual int NRff   (void) const override { return locNRff;    }
    virtual int NRffRep(void) const override { return locNRffRep; }
    virtual int ReOnly (void) const override { return locReOnly;  }
    virtual int inAdam (void) const override { return innerAdam;  }
    virtual int outGrad(void) const override { return xoutGrad;   }

    virtual double D    (void) const override { return locD;            }
    virtual double E    (void) const override { return SVM_Scalar::C(); }
    virtual int    tunev(void) const override { return loctunev;        }
    virtual int    pegk (void) const override { return locpegk;         }
    virtual double minv (void) const override { return locminv;         }
    virtual double F    (void) const override { return locF;            }
    virtual double G    (void) const override { return locG;            }

    // General modification and autoset functions

    virtual int setLinearCost   (void) override { Bacgood = false; locisTrained = 0; cholscratchcov  = 0; covlogdetcalced = 0; costtype = 0; return 1; }
    virtual int setQuadraticCost(void) override { Bacgood = false; locisTrained = 0; cholscratchcov  = 0; covlogdetcalced = 0; costtype = 1; return 1; }
    virtual int set1NormCost    (void) override { NiceThrow("Can't do 1 norm cost in SVM RFF");                                                                    return 0; }

    virtual int setNRff   (int nv) override;
    virtual int setNRffRep(int nv) override;
    virtual int setReOnly (int nv) override;
    virtual int setinAdam (int nv) override { innerAdam = nv; return 1; }
    virtual int setoutGrad(int nv) override { xoutGrad  = nv; return 1; }

    virtual int setD     (double nv) override { locD = nv;            locisTrained = 0; cholscratchcov  = 0; covlogdetcalced = 0; return 1; }
    virtual int setE     (double nv) override { SVM_Scalar::setC(nv); locisTrained = 0; cholscratchcov  = 0; covlogdetcalced = 0; return 1; }
    virtual int setF     (double nv) override { locF = nv;            locisTrained = 0; cholscratchcov  = 0; covlogdetcalced = 0; return 1; }
    virtual int setG     (double nv) override { locG = nv;            locisTrained = 0; cholscratchcov  = 0; covlogdetcalced = 0; return 1; }
    virtual int setpegk  (int    nv) override { locpegk = nv;                                                                                           return 1; }
    virtual int settunev (int    nv) override;
    virtual int setminv  (double nv) override;

    // Pre-training funciton

    virtual int pretrain(void) override;

    virtual double loglikelihood(void) const override;
    virtual double maxinfogain  (void) const override;
    virtual double RKHSnorm     (void) const override;
    virtual double RKHSabs      (void) const override { return sqrt(RKHSnorm()); }

//private: - need to do this first in errortest so that all versions in cross-fold share the *same* random features (which are stored in x, which is shared)
    int fixupfeatures(void);

    // inintrain: for fixed v, find vw
    // intrain: optional outer loop that tunes v and calls inintrain at each iteration
    // method: 0 = direct matrix inversion
    //         1 = adam
    //         2 = adam with 2 hotstart
    //         3 = stochastic adam
    //         4 = gradient
    //         5 = stochastic gradient
    //         6 = direct matrix inversion using offNaiveDiagChol
    //         7 = pegasos

    double inintrain(int &res, svmvolatile int &killSwitch, Vector<double> &vw, Vector<double> &vwaug, Vector<double> &vv, double &b, double lambda, int method, int &notfirstcall, int &fbused);
    double intrain(  int &res, svmvolatile int &killSwitch, Vector<double> &vw, Vector<double> &vwaug, Vector<double> &vv, double &b, double lambda, double Lambda);












    // bit of a cheat

    const Vector<double> &yRR(void) const { return loczr; }

    virtual bool disableDiag(void) const override
    {
        return true;
    }

private:
    // LSRff:   lengthscale for random draws for TRF
    // RFFDist: 1:  just a linear kernel(ish)
    //          3:  draw from normal distribution (RFF on RBF kernel with given lengthscale)
    //          4:  draw from Cauchy distribution (RFF on Laplacian kernel with given lengthscale)
    //          13: draw from uniform distribution (RFF on Wave kernel with given lengthscale)
    //          19: draw from Exponential distribution (RFF on Cauchy kernel with given lengtshcale)

    double LSRff  (void) const { return xlsrff;     }
    int    RFFDist(void) const { return locddtype;  }

    int setLSRff  (double nv);
    int setRFFDist(int    nv);
    int setLSRffandRFFDist(double nv, int nvb);

    // asdkjlui

    int addingNow;

    bool Bacgood;
    bool featGood;
    bool midxset;

    double xlsrff;    // lengthscale for kernel features are drawn from
    double loclr;     // learning rate
    double loclrb;    // learning rate
    double locOpttol; // optima tolerance
    double locOpttolb; // optima tolerance
    int loctunev;  // v sign (0 no tuning, -1 negative, +1 positive, 2 anything)
    int locpegk;   // pegasos "k" value
    double locC;      // C value (1/lambda)
    double loceps;    // eps value
    double locD;      // D value (1/Lambda)
    double locminv;   // regularise with (v-minv)'.H.(v-minv)
    double locF;      // weght F for term F/2 ( 1'.v - G )^2
    double locG;      // target F for term F/2 ( 1'.v - G )^2

    int locN;         // N (number of training vectors)
    int locisTrained; // is trained flag
    int locNRff;
    int locddtype;    // type of kernel features are drawn from
    int locNRffRep;   // 0 for standard, > 0 for locNRffRep tasks (that is, repeat each random
                      // feature this many times, first time with :::: 7:[ 1 0 0 ... ] appended,
                      // second time with :::: 7:[ 0 1 0 ... ] appended.  These repetitions all
                      // share the same "v", but have different "w", so transfer occurs via "v".
    int locReOnly;    // 0 for real and imaginary, 1 for real only
    int innerAdam;    // 1 for tuning with adam on the inner loop, 0 for matrix inversion on the inner loop
    int xoutGrad;     // Outer ADAM gradient approach
                      // 0,2: optimise for modified gradient on v.^-1
                      // 1: optimise for gradient on v using Cholesky of H
                      // 3: like 2, but using precalculated Cholesky loaned from SVM_Scalar (default)
    int costtype;     // 0: linear
                      // 1: quadratic

    Vector<double> locCclass;      // classwise C weights (0 = -1 (not used), 1 = zero, 2 = +1 (not used), 3 = free)
    Vector<double> locepsclass;    // classwise eps weights (0 = -1 (not used), 1 = zero, 2 = +1 (not used), 3 = free)
    Vector<double> locCweight;     // elementwise C weights
    Vector<double> locepsweight;   // elementwise C weights
    Vector<double> locCweightfuzz; // not used (1 default)

    Vector<double> locaw; // This is v
    Vector<double> locphase; // Random between -pi and pi

    Vector<gentype> locz; // This is y in gentype form
    Vector<double> loczr; // This is y

    // inintrain solves:
    //
    // 1/2 [ vw ]' [ B + lambda.diag(inv(v))   a  ] [ vw ] - [ vw ]' [ c ]
    //     [ b  ]  [ a'                        NN ] [ b  ]   [ b  ]  [ d ]

    Matrix<double> locB;  // This is P
    Vector<double> loca;  // This is q
    Vector<double> locc;  // This is s

    Chol<double> locBF;   // Factorisation of locB
    Vector<double> BFoff; // Diagonal offset used for locBF factorisation

    Chol<double> locLB; // cholesky factorisation of P + lambda/minv

    Vector<int> isact; // for each training vector, 0 if inactive (not in Q etc), 1 if active

    double locNN; // This is r
    double locd;  // This is t

    double calcRefLengthScale(void) const { return xlsrff;                                                     }
    double calcCscalequick(int i)   const { return locCclass(trainclass(i)+1)*locCweight(i)*locCweightfuzz(i); }
    double calcepsscalequick(int i) const { return locepsclass(trainclass(i)+1)*locepsweight(i);               }
    double calcepsvalquick(int i)   const { return loceps*locepsclass(trainclass(i)+1)*locepsweight(i);        }




    // Bias related stuff

    // eTrainingVector works through gTrainingVector
    // ghTrainingVector either works through gTrainingVector or doesn't actually need the bias
    // gTrainingVector mostly works through biasR(), unless i >= 0 for non-gradient evaluation, in which case we need to correct the result
    // covTrainingVector doesn't go through gTrainingVector, so we need to do the bias adjust on this too.

    virtual int gTrainingVector(double &res, int &unusedvar, int i, int raw = 0, gentype ***pxyprodi = nullptr) const override;
    virtual int covTrainingVector(gentype &resv, gentype &resmu, int i, int j, gentype ***pxyprodx = nullptr, gentype ***pxyprody = nullptr, gentype **pxyprodij = nullptr) const override;

    int biastype;
    Vector<double> locbias;
    mutable gentype tempgbias;




    // Stuff that is kept for speed when training.  No need to copy, swap etc

    mutable Vector<double> ivscratch;
    Vector<double> vwoldscratch;
    Vector<double> gscratch;
    Vector<double> dgscratch;
    Vector<double> Qiiscratch;
    Vector<double> dvwscratch;
    Vector<double> sscratch;
    mutable Vector<double> yscratch;
    Vector<double> xscratch;
    mutable Matrix<double> cholscratch;
    mutable Matrix<double> cholscratchb;
    Vector<double> qscratch;
    Vector<double> epsscratch;
    Vector<double> Csscratch;
    Vector<int> Atscratch;
    Vector<int> Btscratch;
    mutable int cholscratchcov;  // true if cholscratch is initialised for covariance (reduced cubic to quadratic cost)
    mutable int covlogdetcalced; // true of determinant of covariance is pre-calculated
    mutable double covlogdet;

    Vector<double> vwstartpoint;
    double bstartpoint;

    ADAMscratch inadamscratchpad;
    ADAMscratch ininadamscratchpad;

    int setLSRffoirRFFDist(double lsrffscale = 1, int scalelsrff = 0);

    int calcwdim(void) const { return INDIM*NRff(); }
    int calcvdim(void) const { return NRff(); }

    double getGpRowRffPartNorm2(int i, const Vector<double> &ivl) const;
    Vector<double> &getGpRowRffPart(Vector<double> &res, int i) const;
    int setIsAct(int nv, int i, Vector<double> &Qii, bool xdiff = true, bool ydiff = true, bool cdiff = true, bool QiiIsPrecalced = false);

    // dir = -1 if removing, +1 if adding

    int addRemoveXInfluence(double dir, int i, Vector<double> &Qii, bool xdiff = true, bool ydiff = true, bool cdiff = true, bool QiiIsPrecalced = false);
    int addRemoveXInfluence(double dir, const Vector<int> &ii, Vector<double> &Qii, bool xdiff = true, bool ydiff = true, bool cdiff = true);
    int updateBAC(const Vector<int> &ii);
    int fixactiveset(Vector<double> &Qii, Vector<double> &g, Vector<double> &Eps, Vector<double> &vw, double b, int wdim, int Nval, int &notfirstcall);
};

inline double norm2(const SVM_Scalar_rff &a);
inline double abs2 (const SVM_Scalar_rff &a);

inline double norm2(const SVM_Scalar_rff &a) { return a.RKHSnorm(); }
inline double abs2 (const SVM_Scalar_rff &a) { return a.RKHSabs();  }

inline void qswap(SVM_Scalar_rff &a, SVM_Scalar_rff &b)
{
    a.qswapinternal(b);

    return;
}

inline void SVM_Scalar_rff::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    SVM_Scalar_rff &b = dynamic_cast<SVM_Scalar_rff &>(bb.getML());

    SVM_Scalar::qswapinternal(b);

    qswap(Bacgood ,b.Bacgood );
    qswap(featGood,b.featGood);
    qswap(midxset ,b.midxset );

    qswap(locOpttol ,b.locOpttol );
    qswap(locOpttolb,b.locOpttolb);
    qswap(loctunev  ,b.loctunev  );
    qswap(locpegk   ,b.locpegk   );
    qswap(locReOnly ,b.locReOnly );
    qswap(locNRff   ,b.locNRff   );
    qswap(locddtype ,b.locddtype );
    qswap(locNRffRep,b.locNRffRep);
    qswap(innerAdam ,b.innerAdam );
    qswap(xoutGrad  ,b.xoutGrad  );
    qswap(costtype  ,b.costtype  );
    qswap(loclr     ,b.loclr     );
    qswap(loclrb    ,b.loclrb    );
    qswap(xlsrff    ,b.xlsrff    );
    qswap(locC      ,b.locC      );
    qswap(loceps    ,b.loceps    );
    qswap(locD      ,b.locD      );
    qswap(locminv   ,b.locminv   );
    qswap(locF      ,b.locF      );
    qswap(locG      ,b.locG      );
    qswap(locN      ,b.locN      );

    qswap(locCclass     ,b.locCclass     );
    qswap(locepsclass   ,b.locepsclass   );
    qswap(locCweight    ,b.locCweight    );
    qswap(locepsweight  ,b.locepsweight  );
    qswap(locCweightfuzz,b.locCweightfuzz);

    qswap(locz ,b.locz );
    qswap(loczr,b.loczr);
    qswap(isact,b.isact);
    qswap(locaw,b.locaw);

    qswap(biastype ,b.biastype );
    qswap(locbias  ,b.locbias  );
    qswap(tempgbias,b.tempgbias);

    qswap(locB ,b.locB );
    qswap(locBF,b.locBF);
    qswap(BFoff,b.BFoff);
    qswap(loca ,b.loca );
    qswap(locc ,b.locc );
    qswap(locd ,b.locd );
    qswap(locNN,b.locNN);

    cholscratchcov  = 0;
    covlogdetcalced = 0;

    qswap(locisTrained,b.locisTrained);

    return;
}

inline void SVM_Scalar_rff::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const SVM_Scalar_rff &b = dynamic_cast<const SVM_Scalar_rff &>(bb.getMLconst());

    SVM_Scalar::semicopy(b);

    Bacgood  = b.Bacgood;
    featGood = b.featGood;
    midxset  = b.midxset;

    locOpttol  = b.locOpttol;
    locOpttolb = b.locOpttolb;
    loctunev   = b.loctunev;
    locpegk    = b.locpegk;
    locReOnly  = b.locReOnly;
    locNRff    = b.locNRff;
    locNRffRep = b.locNRffRep;
    locddtype  = b.locddtype;
    innerAdam  = b.innerAdam;
    xoutGrad   = b.xoutGrad;
    costtype   = b.costtype;
    loclr      = b.loclr;
    loclrb     = b.loclrb;
    xlsrff     = b.xlsrff;
    locC       = b.locC;
    loceps     = b.loceps;
    locD       = b.locD;
    locminv    = b.locminv;
    locF       = b.locF;
    locG       = b.locG;
    locN       = b.locN;

    locCclass   = b.locCclass;
    locepsclass = b.locepsclass;

    //locCweight     = b.locCweight;
    //locepsweight   = b.locepsweight;
    //locCweightfuzz = b.locCweightfuzz;

    locaw = b.locaw;

    //locz  = b.locz;
    //loczr = b.loczr;

    isact = b.isact;

    biastype  = b.biastype;
    locbias   = b.locbias;
    tempgbias = b.tempgbias;

    locB  = b.locB;
    locBF = b.locBF;
    BFoff = b.BFoff;
    loca  = b.loca;
    locc  = b.locc;
    locd  = b.locd;
    locNN = b.locNN;

    locisTrained    = b.locisTrained;
    cholscratchcov  = 0;
    covlogdetcalced = 0;

    return;
}

inline void SVM_Scalar_rff::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const SVM_Scalar_rff &src = dynamic_cast<const SVM_Scalar_rff &>(bb.getMLconst());

    SVM_Scalar::assign(static_cast<const SVM_Scalar &>(src),onlySemiCopy);

    Bacgood  = src.Bacgood;
    featGood = src.featGood;
    midxset  = src.midxset;

    locOpttol  = src.locOpttol;
    locOpttolb = src.locOpttolb;
    loctunev   = src.loctunev;
    locpegk    = src.locpegk;
    locReOnly  = src.locReOnly;
    locNRff    = src.locNRff;
    locNRffRep = src.locNRffRep;
    locddtype  = src.locddtype;
    innerAdam  = src.innerAdam;
    xoutGrad   = src.xoutGrad;
    costtype   = src.costtype;
    loclr      = src.loclr;
    loclrb     = src.loclrb;
    xlsrff     = src.xlsrff;
    locC       = src.locC;
    loceps     = src.loceps;
    locD       = src.locD;
    locminv    = src.locminv;
    locF       = src.locF;
    locG       = src.locG;
    locN       = src.locN;

    locCclass   = src.locCclass;
    locepsclass = src.locepsclass;

    locCweight     = src.locCweight;
    locepsweight   = src.locepsweight;
    locCweightfuzz = src.locCweightfuzz;

    locaw = src.locaw;

    isact = src.isact;

    locz  = src.locz;
    loczr = src.loczr;

    biastype  = src.biastype;
    locbias   = src.locbias;
    tempgbias = src.tempgbias;

    locB  = src.locB;
    locBF = src.locBF;
    BFoff = src.BFoff;
    loca  = src.loca;
    locc  = src.locc;
    locd  = src.locd;
    locNN = src.locNN;

    locisTrained    = src.locisTrained;
    cholscratchcov  = 0;
    covlogdetcalced = 0;

    return;
}

#endif
