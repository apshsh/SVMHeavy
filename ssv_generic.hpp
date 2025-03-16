//FIXME: assumption made: x vectors share indices, zmin/zmax also share same - DOCUMENT THIS!

//
// Super-sparse SVMs
//
// Version: 7
// Date: 14/11/2017
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//
// See: Demontis et al, Super-Sparse Learning in Similarity Spaces
//
// Note: Rather than u and lambda we use u[i] = Ci/Nnz (Ci including C,
//       Cweight as per SVM_Scalar).
//
//

#ifndef _ssv_generic_h
#define _ssv_generic_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "svm_scalar.hpp"
#include "lsv_scalar.hpp"


// 
// Loss is least-squares, model is: 
// 
// g(x) = sum_j alpha_j K (z_j,x) + b 
// 
// Learning of alpha is done by svm_scalar, learning of 
// z_j is done locally.  Optimisation is slightly modified
// to fit with SVM standards: 
// 
// min_{beta,b,z}   Q = 1/2 beta'.beta  +  C/2 sum_i C_i ( g(x_i) - y_i )^2 
// 
// In terms of beta,b: 
// 
// 1/2 [ beta ]' [ S_{xz}'.U.S_{xz} + I   S_{xz}'.U.1 ] [ beta ] - [ beta ]' [ S_{xz}'.U.y ] 
//     [  b   ]  [    1'.U.S_{xz}            1'.U.1   ] [  b   ]   [  b   ]  [   1'.U.y    ] 
//               +-------------------+----------------+                      +-----+-------+
//                               M{p,pn,n}                                       n{p,n}
// 
// where: 
// 
// U = diag(C.C_i)/Nnz
// S_{xz}_{ij} = K(x_i,z_j) 
// 
// This is solved exactly to find beta,b.  z_j terms are found using 
// gradient descent, where the gradient wrt z is: 
// 
// dQ/dz_j = 2.(g-y)'.U.( beta_j dS_{x,z_j}/dz_j + S_{xz} dbeta/dz_j + 1 db/dz_j ) + 2 beta'.dbeta/dz_j
// 
// where:
//
// g = [ g(x_0) ]
//     [ g(x_1) ]
//     [  ...   ]
//
// S_{x,z_j} = col(j) of S_{xz}
//
// dS_{x,z_j}/dz_j = [ dS_{x_0,z_j}/dz_{j0}  dS_{x_0,z_j}/dz_{j1}  ...  ]
//                   [ dS_{x_1,z_j}/dz_{j0}  dS_{x_1,z_j}/dz_{j1}  ...  ]
//                   [          ...                   ...          ...  ]
//
// and:
//
// [ dbeta/dz_j ] = -inv(M).( beta_j [ S_{xz}' ] + [ V' ] ).U.dS_{xz_j}/dz_j
// [ db/dz_j    ]           (        [  1'     ] + [ 0' ] )
//
// or, component by component:
//
// [ dbeta/dz_{jk} ] = -inv(M).( beta_j [ S_{xz}' ] + [ Vj' ] ).U.dS_{xz_j}/dz_{jk}	
// [ db/dz_{jk}    ]           (        [  1'     ] + [ 0'  ] )
// 
// where:
//
// Vj = [ 0 ... 0 g-y 0 ... 0 ]
//                 |
//              column j
//
// which can be shoe-horned back into the minimization problem:
//
// 1/2 [ dbeta/dz_{jk} ]' M [ dbeta/dz_{jk} ] - [ dbeta/dz_{jk} ]' [ r ] 
//     [ db/dz_{jk}    ]    [ db/dz_{jk}    ]   [ db/dz_{jk}    ]  [ s ] 
//
// where:
//
// [ r ] = -( beta_j [ S_{xz}' ] + [ Vj' ] ).U.dS_{xz_j}/dz_{jk}
// [ s ]    (        [  1'     ] + [ 0'  ] )
//
// (that is, we can use the same underlying SVM minimiser to do both parts)
//
// With regard to non-equality constraints (classification) we include
// in S_{xz}, U, g, y only those rows where the inequality requires
// enforcement (the active set on these).
//
//
//
//
// If we define:
//
// [ Q ] = [ U.S_{xz} ]
// [ t ]   [    1     ]
//
// Then the beta problem is:
// 
// 1/2 [ beta ]' [ Q'.inv(U).Q + I   Q'.1  ] [ beta ] + [ beta ]' [  -Q'.y  ] 
//     [  b   ]  [    1'.Q          1'.U.1 ] [  b   ]   [  b   ]  [ -1'.U.y ] 
// 
// or:
//
// 1/2 [ Q.beta ]' [ inv(U) + I     1   ] [ Q.beta ] + [ Q.beta ]' [   -y    ] 
//     [   b    ]  [   1'        1'.U.1 ] [   b    ]   [   b    ]  [ -1'.U.y ] 
// 
// and, likewise:
//
// 1/2 [ Q.dbeta/dz_{jk} ]' [ inv(U) + I     1   ] [ Q.dbeta/dz_{jk} ] - [ Q.dbeta/dz_{jk} ]' [ ... ] 
//     [    db/dz_{jk}   ]  [   1'        1'.U.1 ] [    db/dz_{jk}   ]   [    db/dz_{jk}   ]  [ ... ] 
//
// but this doesn't lead anywhere as the dimension is way too large!
//
//
//
//
//
//
// Note on M: we note that M is positive definite, so [ beta b ] must be 
// treated (for the sake of shoe-horning into SVM model) as alpha = [ beta b].  
// Furthermore we use least-squares SVM for speed.
//
//
//
//
//
//
// Adding new variants: currently set up for scalar targets, more generally
// you'll need to overwrite eTrainingVector and override functions.  Also
// you'll need to do training.
//







class SSV_Generic;
class kssv_callback : public kernPrecursor
{
public:
    virtual int isKVarianceNZ(void) const;

    virtual void K0xfer(gentype &res, int &minmaxind, int typeis,
                        const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                        int xdim, int densetype, int resmode, int mlid) const;

    virtual void K1xfer(gentype &res, int &minmaxind, int typeis,
                        const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                        const SparseVector<gentype> &xa, 
                        const vecInfo &xainfo, 
                        int ia, 
                        int xdim, int densetype, int resmode, int mlid) const;

    virtual void K2xfer(gentype &dxyprod, gentype &ddiffis, gentype &res, int &minmaxind, int typeis,
                        const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                        const vecInfo &xainfo, const vecInfo &xbinfo,
                        int ia, int ib,
                        int xdim, int densetype, int resmode, int mlid) const;

    virtual void K3xfer(gentype &res, int &minmaxind, int typeis,
                        const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, 
                        const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, 
                        int ia, int ib, int ic, 
                        int xdim, int densetype, int resmode, int mlid) const;

    virtual void K4xfer(gentype &res, int &minmaxind, int typeis,
                        const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                        const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
                        int ia, int ib, int ic, int id,
                        int xdim, int densetype, int resmode, int mlid) const;

    virtual void Kmxfer(gentype &res, int &minmaxind, int typeis,
                        const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                        Vector<const SparseVector<gentype> *> &x,
                        Vector<const vecInfo *> &xzinfo,
                        Vector<int> &ii,
                        int xdim, int m, int densetype, int resmode, int mlid) const;

    virtual void K0xfer(double &res, int &minmaxind, int typeis,
                        double xyprod, double yxprod, double diffis,
                        int xdim, int densetype, int resmode, int mlid) const;

    virtual void K1xfer(double &res, int &minmaxind, int typeis,
                        double xyprod, double yxprod, double diffis,
                        const SparseVector<gentype> &xa, 
                        const vecInfo &xainfo, 
                        int ia, 
                        int xdim, int densetype, int resmode, int mlid) const;

    virtual void K2xfer(double &dxyprod, double &ddiffis, double &res, int &minmaxind, int typeis,
                        double xyprod, double yxprod, double diffis,
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                        const vecInfo &xainfo, const vecInfo &xbinfo,
                        int ia, int ib,
                        int xdim, int densetype, int resmode, int mlid) const;

    virtual void K3xfer(double &res, int &minmaxind, int typeis,
                        double xyprod, double yxprod, double diffis,
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, 
                        const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, 
                        int ia, int ib, int ic, 
                        int xdim, int densetype, int resmode, int mlid) const;

    virtual void K4xfer(double &res, int &minmaxind, int typeis,
                        double xyprod, double yxprod, double diffis,
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                        const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
                        int ia, int ib, int ic, int id,
                        int xdim, int densetype, int resmode, int mlid) const;

    virtual void Kmxfer(double &res, int &minmaxind, int typeis,
                        double xyprod, double yxprod, double diffis,
                        Vector<const SparseVector<gentype> *> &x,
                        Vector<const vecInfo *> &xzinfo,
                        Vector<int> &ii,
                        int xdim, int m, int densetype, int resmode, int mlid) const;

    SSV_Generic *realOwner;
};


#define ZY(i) ( ( type() == 702 ) ? 0.0 : zy(i) )

// Swap function

inline void qswap(SSV_Generic &a, SSV_Generic &b);
inline SSV_Generic &setzero(SSV_Generic &a);


class SSV_Generic : public SVM_Scalar
{
    friend class kssv_callback;

public:

    // Constructors, destructors, assignment etc..

    SSV_Generic();
    SSV_Generic(const SSV_Generic &src);
    SSV_Generic(const SSV_Generic &src, const ML_Base *srcx);
    SSV_Generic &operator=(const SSV_Generic &src) { assign(src); return *this; }
    virtual ~SSV_Generic() { return; }





    virtual void assign(const ML_Base &src, int onlySemiCopy = 0) override;
    virtual void semicopy(const ML_Base &src) override;
    virtual void qswapinternal(ML_Base &b) override;

    virtual int getparam (int ind, gentype         &val, const gentype         &xa, int ia, const gentype         &xb, int ib, charptr &desc) const override;
    virtual int egetparam(int ind, Vector<gentype> &val, const Vector<gentype> &xa, int ia, const Vector<gentype> &xb, int ib               ) const override;

    virtual std::ostream &printstream(std::ostream &output, int dep) const override;
    virtual std::istream &inputstream(std::istream &input ) override;

    virtual       ML_Base &getML     (void)       override { return static_cast<      ML_Base &>(getSSV());      }
    virtual const ML_Base &getMLconst(void) const override { return static_cast<const ML_Base &>(getSSVconst()); }

    // Information functions (training data):

    virtual int N(void)    const override { return SVM_Scalar::N()-Nzs(); }
    virtual int NNC(int d) const override { return d ? SVM_Scalar::NNC(d) : SVM_Scalar::NNC(d)-Nzs(); }

    virtual double C(void)     const override { return 1/sigma();      }
    virtual double sigma(void) const override { return zmodel.sigma(); }
    virtual double sigma_cut(void) const override { return zmodel.sigma_cut(); }

    virtual int isSolGlob(void) const override { return 0; }

    // For some obscure but important reason y() is used in SVM_Generic for N(), so
    // we can't do these overloads!  (also xinfo messes things up for some reason).

/*    virtual const Vector<SparseVector<gentype> > &x          (void) const { return (SVM_Scalar::x          ())(0,1,N()-1); }
    virtual const Vector<gentype>                &y          (void) const { return zy;                                     }
    virtual const Vector<vecInfo>                &xinfo      (void) const { return (SVM_Scalar::xinfo      ())(0,1,N()-1); }
    virtual const Vector<int>                    &d          (void) const { return (SVM_Scalar::d          ())(0,1,N()-1); }
    virtual const Vector<double>                 &Cweight    (void) const { return (SVM_Scalar::Cweight    ())(0,1,N()-1); }
    virtual const Vector<double>                 &Cweightfuzz(void) const { return (SVM_Scalar::Cweightfuzz())(0,1,N()-1); }
    virtual const Vector<double>                 &sigmaweight(void) const { return (SVM_Scalar::sigmaweight())(0,1,N()-1); }
    virtual const Vector<double>                 &epsweight  (void) const { return (SVM_Scalar::epsweight  ())(0,1,N()-1); }
    virtual const Vector<int>                    &alphaState (void) const { return (SVM_Scalar::alphaState ())(0,1,N()-1); }
*/

    // Kernel Modification

    virtual void prepareKernel(void) override { return; }
    virtual int resetKernel(int modind = 1, int onlyChangeRowI = -1, int updateInfo = 1) override;
    virtual int setKernel(const MercerKernel &xkernel, int modind = 1, int onlyChangeRowI = -1) override;

    // Training set modification

    virtual int addTrainingVector (int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1) override;
    virtual int qaddTrainingVector(int i, const gentype &y,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1) override;

    virtual int addTrainingVector(int i,            double *xxa, int dima, double Cweigh = 1, double epsweigh = 1) override { return ML_Base::addTrainingVector(i,   xxa,dima,Cweigh,epsweigh); }
    virtual int addTrainingVector(int i, int zz,    double *xxa, int dima, double Cweigh = 1, double epsweigh = 1) override { return ML_Base::addTrainingVector(i,zz,xxa,dima,Cweigh,epsweigh); }
    virtual int addTrainingVector(int i, double zz, double *xxa, int dima, double Cweigh = 1, double epsweigh = 1) override { return ML_Base::addTrainingVector(i,zz,xxa,dima,Cweigh,epsweigh); }

    virtual int addTrainingVector (int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override;
    virtual int qaddTrainingVector(int i, const Vector<gentype> &y,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override;

    virtual int removeTrainingVector(int i                                      ) override { SparseVector<gentype> x; gentype y; return removeTrainingVector(i,y,x); }
    virtual int removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x) override;
    virtual int removeTrainingVector(int i, int num                             ) override { return ML_Base::removeTrainingVector(i,num); }

    virtual int setx(int                i, const SparseVector<gentype>          &x) override;
    virtual int setx(const Vector<int> &i, const Vector<SparseVector<gentype> > &x) override;
    virtual int setx(                      const Vector<SparseVector<gentype> > &x) override { retVector<int> tmpva; return SSV_Generic::setx(cntintvec(N(),tmpva),x); }

    virtual int qswapx(int                i, SparseVector<gentype>          &x, int dontupdate = 0) override;
    virtual int qswapx(const Vector<int> &i, Vector<SparseVector<gentype> > &x, int dontupdate = 0) override;
    virtual int qswapx(                      Vector<SparseVector<gentype> > &x, int dontupdate = 0) override { retVector<int> tmpva; return SSV_Generic::qswapx(cntintvec(N(),tmpva),x,dontupdate); }

    virtual int sety(int                i, const gentype         &y) override;
    virtual int sety(const Vector<int> &i, const Vector<gentype> &y) override;
    virtual int sety(                      const Vector<gentype> &y) override { retVector<int> tmpva; return SSV_Generic::sety(cntintvec(N(),tmpva),y);           }

    virtual int setd(int                i, int                d) override;
    virtual int setd(const Vector<int> &i, const Vector<int> &d) override;
    virtual int setd(                      const Vector<int> &d) override { retVector<int> tmpva; return SVM_Scalar::setd(cntintvec(N(),tmpva),d); }

    virtual int setCweight(int                i, double                nv) override;
    virtual int setCweight(const Vector<int> &i, const Vector<double> &nv) override;
    virtual int setCweight(                      const Vector<double> &nv) override { retVector<int> tmpva; return SSV_Generic::setCweight(cntintvec(N(),tmpva),nv); }

    virtual int setCweightfuzz(int i,                double nv               ) override;
    virtual int setCweightfuzz(const Vector<int> &i, const Vector<double> &nv) override;
    virtual int setCweightfuzz(                      const Vector<double> &nv) override { retVector<int> tmpva; return SSV_Generic::setCweightfuzz(cntintvec(N(),tmpva),nv); }

    virtual int setsigmaweight(int                i, double                nv) override;
    virtual int setsigmaweight(const Vector<int> &i, const Vector<double> &nv) override;
    virtual int setsigmaweight(                      const Vector<double> &nv) override { retVector<int> tmpva; return SSV_Generic::setsigmaweight(cntintvec(N(),tmpva),nv); }

    virtual int setepsweight(int                i, double                nv) override { return SVM_Scalar::setepsweight(i,nv);              }
    virtual int setepsweight(const Vector<int> &i, const Vector<double> &nv) override { return SVM_Scalar::setepsweight(i,nv);              }
    virtual int setepsweight(                      const Vector<double> &nv) override { retVector<int> tmpva; return SVM_Scalar::setepsweight(cntintvec(N(),tmpva),nv); }

    virtual const SparseVector<gentype> &x(int i) const override { return SVM_Scalar::x(i);     }
    virtual const SparseVector<gentype> &x(int i, int altMLid) const override { return SVM_Scalar::x(i,altMLid); }
    virtual const vecInfo &xinfo(int i)           const override { return SVM_Scalar::xinfo(i); }

    // General modification and autoset functions

    virtual int randomise(double sparsity) override { return SVM_Scalar::randomise(sparsity); }
    virtual int autoen(void)               override { return SVM_Scalar::autoen();            }
    virtual int renormalise(void)          override { return SVM_Scalar::renormalise();       }
    virtual int realign(void)              override { return SVM_Scalar::realign();           }

    virtual int setzerotol(double zt)                 override { return SVM_Scalar::setzerotol(zt)                 | zmodel.setzerotol(zt);                 }
    virtual int setOpttol(double xopttol)             override { return SVM_Scalar::setOpttol(xopttol)             | zmodel.setOpttol(xopttol);             }
    virtual int setOpttolb(double xopttol)            override { return SVM_Scalar::setOpttolb(xopttol)            | zmodel.setOpttolb(xopttol);            }
    virtual int setOpttolc(double xopttol)            override { return SVM_Scalar::setOpttolc(xopttol)            | zmodel.setOpttolc(xopttol);            }
    virtual int setOpttold(double xopttol)            override { return SVM_Scalar::setOpttold(xopttol)            | zmodel.setOpttold(xopttol);            }
    virtual int setlr(double xlr)                     override { return SVM_Scalar::setlr(xlr)                     | zmodel.setlr(xlr);                     }
    virtual int setlrb(double xlr)                    override { return SVM_Scalar::setlrb(xlr)                    | zmodel.setlrb(xlr);                    }
    virtual int setlrc(double xlr)                    override { return SVM_Scalar::setlrc(xlr)                    | zmodel.setlrc(xlr);                    }
    virtual int setlrd(double xlr)                    override { return SVM_Scalar::setlrd(xlr)                    | zmodel.setlrd(xlr);                    }
    virtual int setmaxitcnt(int xmaxitcnt)            override { return SVM_Scalar::setmaxitcnt(xmaxitcnt)         | zmodel.setmaxitcnt(xmaxitcnt);         }
    virtual int setmaxtraintime(double xmaxtraintime) override { return SVM_Scalar::setmaxtraintime(xmaxtraintime) | zmodel.setmaxtraintime(xmaxtraintime); }
    virtual int settraintimeend(double xtraintimeend) override { return SVM_Scalar::settraintimeend(xtraintimeend) | zmodel.settraintimeend(xtraintimeend); }

    virtual int setC(double xC) override { return setsigma(1/xC); }
    virtual int setsigma    (double xC) override;
    virtual int setsigma_cut(double xC) override { return zmodel.setsigma_cut(xC); }
    virtual int setCclass(int d, double xC) override;

    virtual int scale(double a) override;
    virtual int reset(void) override;
    virtual int restart(void) override { SSV_Generic temp; *this = temp; return 1; }

    // Evaluation Functions:

    virtual int ggTrainingVector(               gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = nullptr) const override { gentype resh; return ghTrainingVector(resh,resg,i,retaltg,pxyprodi); }
    virtual int hhTrainingVector(gentype &resh,                int i,                  gentype ***pxyprodi = nullptr) const override { gentype resg; return ghTrainingVector(resh,resg,i,0,      pxyprodi); }
    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = nullptr) const override;

    virtual int covTrainingVector(gentype &resv, gentype &resmu, int i, int j, gentype ***pxyprodi = nullptr, gentype ***pxyprodj = nullptr, gentype **pxyprodij = nullptr) const override { return SVM_Scalar::covTrainingVector(resv,resmu,i,j,pxyprodi,pxyprodj,pxyprodij); }

    virtual void dgTrainingVectorX(Vector<gentype> &resx, int i) const override { SVM_Scalar::dgTrainingVectorX(resx,i); return; }
    virtual void dgTrainingVectorX(Vector<double>  &resx, int i) const override { SVM_Scalar::dgTrainingVectorX(resx,i); return; }

    virtual int ggTrainingVector(double         &resg, int i, int retaltg = 0, gentype ***pxyprodi = nullptr) const override { return SVM_Scalar::ggTrainingVector(resg,i,retaltg,pxyprodi); }
    virtual int ggTrainingVector(Vector<double> &resg, int i, int retaltg = 0, gentype ***pxyprodi = nullptr) const override { return SVM_Scalar::ggTrainingVector(resg,i,retaltg,pxyprodi); }
    virtual int ggTrainingVector(d_anion        &resg, int i, int retaltg = 0, gentype ***pxyprodi = nullptr) const override { return SVM_Scalar::ggTrainingVector(resg,i,retaltg,pxyprodi); }

    virtual void dgTrainingVector(Vector<gentype>         &res, gentype        &resn, int i) const override { SVM_Scalar::dgTrainingVector(res,resn,i); return; }
    virtual void dgTrainingVector(Vector<double>          &res, double         &resn, int i) const override { SVM_Scalar::dgTrainingVector(res,resn,i); return; }
    virtual void dgTrainingVector(Vector<Vector<double> > &res, Vector<double> &resn, int i) const override { SVM_Scalar::dgTrainingVector(res,resn,i); return; }
    virtual void dgTrainingVector(Vector<d_anion>         &res, d_anion        &resn, int i) const override { SVM_Scalar::dgTrainingVector(res,resn,i); return; }




    // ================================================================
    //     Common functions for all SSVs
    // ================================================================

    virtual       SSV_Generic &getSSV     (void)       { return *this; }
    virtual const SSV_Generic &getSSVconst(void) const { return *this; }

    // General information and control
    // 
    // Nzs: number of support vectors
    // 
    // beta:   weights for support vectors
    // b:      bias
    // z:      support vectors
    // zmin:   minimum for support vectors
    // zmax:   maximum for support vectors
    // xstate: for training vectors which are active (in M) and which are not
    // xact:   indices of active (in M) training vectors
    // M:      M matrix
    // n:      n vector
    // 
    // isQuadRegul: beta regulation term is 2.||beta||_2^2 (standard)
    // isLinRegul:  beta regulation term is ||beta||_1

    virtual int Nzs(void) const { return zbeta.size(); }

    virtual const Vector<gentype>                &beta  (void) const { return zbeta;                                      }
    virtual const gentype                        &b     (void) const { return zb;                                         }
    virtual const Vector<SparseVector<gentype> > &z     (void) const { return (SVM_Scalar::x())(N(),1,N()+Nzs()-1,retva); }
    virtual const SparseVector<double>           &zmin  (void) const { return zxmin;                                      }
    virtual const SparseVector<double>           &zmax  (void) const { return zxmax;                                      }
    virtual const Vector<int>                    &xstate(void) const { return zxstate;                                    }
    virtual const Vector<int>                    &xact  (void) const { return zxact;                                      }
    virtual const Matrix<double>                 &M     (void) const { return zM;                                         }
    virtual const Vector<double>                 &n     (void) const { return zn;                                         }

    virtual const SparseVector<gentype> &z(int i) const { return SVM_Scalar::x(N()+i); }

    virtual int isQuadRegul(void) const { return zmodel.isQuadraticCost(); }
    virtual int isLinRegul (void) const { return zmodel.isLinearCost();    }

    virtual double biasForce(void) const { return xbiasForce; }
    virtual int anomalclass(void)  const { return +1; }

    // Control functions
    // 
    // setNzs: this should be done after adding data or it won't be able to correctly
    //         infer xspaceDim() and will hence give z vectors of the wrong dimension

    virtual int setbeta(const Vector<gentype> &newBeta);
    virtual int setb   (const gentype         &newb   );

    virtual int setbeta(const Vector<double> &newBeta);
    virtual int setb   (      double          newb   );

    virtual int setNzs(int nv);

    virtual int setzmin(const SparseVector<double> &nv) { zxmin = nv; return 1; }
    virtual int setzmax(const SparseVector<double> &nv) { zxmax = nv; return 1; }

    virtual int setQuadRegul(void);
    virtual int setLinRegul (void);

    virtual int setBiasForce(double nv);
    virtual int setanomalclass(int n) { (void) n; return 0; }

    // Training control (for outer loop)
    //
    // ssvlr       = learning rate
    // ssvmom      = momentum factor
    // ssvtol      = zero tolerance
    // ssvovsc     = overshoot scaleback factor
    // ssvmaxitcnt = maximum iterations
    // ssvmaxtime  = maximum training time

    virtual double ssvlr(void)       const { return zssvlr;       }
    virtual double ssvmom(void)      const { return zssvmom;      }
    virtual double ssvtol(void)      const { return zssvtol;      }
    virtual double ssvovsc(void)     const { return zssvovsc;     }
    virtual int    ssvmaxitcnt(void) const { return zssvmaxitcnt; }
    virtual double ssvmaxtime(void)  const { return zssvmaxtime;  }

    virtual int setssvlr(double nv)      { NiceAssert( nv >  0 ); zssvlr       = nv; return 1; }
    virtual int setssvmom(double nv)     { NiceAssert( nv >= 0 ); zssvmom      = nv; return 1; }
    virtual int setssvtol(double nv)     { NiceAssert( nv >  0 ); zssvtol      = nv; return 1; }
    virtual int setssvovsc(double nv)    { NiceAssert( nv >= 0 ); zssvovsc     = nv; return 1; }
    virtual int setssvmaxitcnt(int nv)   { NiceAssert( nv >= 0 ); zssvmaxitcnt = nv; return 1; }
    virtual int setssvmaxtime(double nv) { NiceAssert( nv >= 0 ); zssvmaxtime  = nv; return 1; }

protected:

    // Internal functions
    //
    // activate:   mark training vector as active (can be used in M matrices) and update
    // deactivate: mark training vector as inactive (cannot be used in M matrices) and update

    virtual int setz(int                j, const SparseVector<gentype>          &newz);
    virtual int setz(const Vector<int> &j, const Vector<SparseVector<gentype> > &newz);
    virtual int setz(                      const Vector<SparseVector<gentype> > &newz);

    virtual int qswapz(int                j, SparseVector<gentype>          &newz, int dontupdate = 0);
    virtual int qswapz(const Vector<int> &j, Vector<SparseVector<gentype> > &newz, int dontupdate = 0);
    virtual int qswapz(                      Vector<SparseVector<gentype> > &newz, int dontupdate = 0);

    virtual int activate(int i = -1);
    virtual int activate(const Vector<int> &i);

    virtual int deactivate(int i = -1);
    virtual int deactivate(const Vector<int> &i);

    // Model data

    Vector<double> zy;
    Vector<gentype> zbeta;
    gentype zb;

    Vector<int> zxstate;  // 1 if activated, 0 otherwise
    Vector<int> zxact;    // indices

    Matrix<double> zM;
    Vector<double> zn;

    SparseVector<double> zxmin;
    SparseVector<double> zxmax;

    int Nnz; // number of non-constrained training vectors

    // Override this with function to calculate C correctly

    virtual double calcCval(int i) const { return ((SVM_Scalar::calcCvalquick(i))*locCscale(i))/( ( Nnz > 0 ) ? Nnz : 1 ); }

    // Override with function to process changes to M

    virtual int updateM(int j = -1);
    virtual int updateM(const Vector<int> &j);

    // Override with function to process changes to n

    virtual int updaten(int j = -1);
    virtual int updaten(const Vector<int> &j);

    // Override with function to process changes in Nzs (increase or decrease by 1 only)
    //
    // Mp, Mpn will be resized before increasing Nzs
    // Mp, Mpn will be resized after decreasing Nzs
    // All others (Mn, np, nn) updated before calling

    virtual int updateNzs(int i, int oldNzs, int newNzs);

    // Model function used to calculate beta etc

    //LSV_Scalar zmodel; LSV is just a front over SVM, so it is actually faster to use SVM
    SVM_Scalar zmodel;

    // Prevents infinite loop in setKernel

    int inbypass;

    double xbiasForce;

    // C scales used during training

    Vector<double> locCscale;

    // Training control, z learning
    //
    // zssvlr       = learning rate
    // zssvmom      = momentum factor
    // zssvtol      = zero tolerance
    // zssvovsc     = overshoot scaleback factor
    // zssvmaxitcnt = maximum iterations
    // zssvmaxtime  = maximum training time

    double zssvlr;
    double zssvmom;
    double zssvtol;
    double zssvovsc;
    int zssvmaxitcnt;
    double zssvmaxtime;

    // M mods

    virtual int updatez(int j, int alsoupdateM = 1);
    virtual int updatez(const Vector<int> &j);
    virtual int updatez(void);

    // Stuff

    mutable retVector<SparseVector<gentype> > retva;

    // Kernel callback

    kssv_callback Kcall;

    void fixKcallback(void);
};

inline double norm2(const SSV_Generic &a);
inline double abs2 (const SSV_Generic &a);

inline double norm2(const SSV_Generic &a) { return a.RKHSnorm(); }
inline double abs2 (const SSV_Generic &a) { return a.RKHSabs();  }

inline void qswap(SSV_Generic &a, SSV_Generic &b)
{
    a.qswapinternal(b);

    return;
}

inline SSV_Generic &setzero(SSV_Generic &a)
{
    a.restart();

    return a;
}

inline void SSV_Generic::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    SSV_Generic &b = dynamic_cast<SSV_Generic &>(bb.getML());

    SVM_Scalar::qswapinternal(b);
    
    qswap(zy            ,b.zy            );
    qswap(zbeta         ,b.zbeta         );
    qswap(zb            ,b.zb            );
    qswap(zxstate       ,b.zxstate       );
    qswap(zxact         ,b.zxact         );
    qswap(zM            ,b.zM            );
    qswap(zn            ,b.zn            );
    qswap(zssvlr        ,b.zssvlr        );
    qswap(zssvmom       ,b.zssvmom       );
    qswap(zssvtol       ,b.zssvtol       );
    qswap(zssvovsc      ,b.zssvovsc      );
    qswap(zssvmaxitcnt  ,b.zssvmaxitcnt  );
    qswap(zssvmaxtime   ,b.zssvmaxtime   );
    qswap(Nnz           ,b.Nnz           );
    qswap(locCscale     ,b.locCscale     );
    qswap(xbiasForce    ,b.xbiasForce    );
    qswap(zmodel        ,b.zmodel        );

    return;
}

inline void SSV_Generic::semicopy(const ML_Base &bb)
{
    // Need full copy as z vectors must be changeable!

    assign(bb);

    return;
}

inline void SSV_Generic::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const SSV_Generic &src = dynamic_cast<const SSV_Generic &>(bb.getMLconst());

    SVM_Scalar::assign(static_cast<const SSV_Generic &>(src),onlySemiCopy);

    zy           = src.zy;
    zbeta        = src.zbeta;
    zb           = src.zb;
    zxstate      = src.zxstate;
    zxact        = src.zxact;
    zM           = src.zM;
    zn           = src.zn;
    zssvlr       = src.zssvlr;
    zssvmom      = src.zssvmom;
    zssvtol      = src.zssvtol;
    zssvovsc     = src.zssvovsc;
    zssvmaxitcnt = src.zssvmaxitcnt;
    zssvmaxtime  = src.zssvmaxtime;
    Nnz          = src.Nnz;
    locCscale    = src.locCscale;
    xbiasForce   = src.xbiasForce;
    zmodel       = src.zmodel;

    fixKcallback();

    return;
}

#endif
