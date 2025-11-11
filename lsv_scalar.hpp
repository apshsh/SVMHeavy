
//
// LS-SVM scalar class
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _lsv_scalar_h
#define _lsv_scalar_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "lsv_generic.hpp"


class LSV_Scalar;

// Swap and zeroing (restarting) functions

inline void qswap(LSV_Scalar &a, LSV_Scalar &b);
inline LSV_Scalar &setzero(LSV_Scalar &a);

class LSV_Scalar : public LSV_Generic
{
public:

    // Constructors, destructors, assignment etc..

    LSV_Scalar();
    LSV_Scalar(const LSV_Scalar &src);
    LSV_Scalar(const LSV_Scalar &src, const ML_Base *srcx);
    LSV_Scalar &operator=(const LSV_Scalar &src) { assign(src); return *this; }
    virtual ~LSV_Scalar() { return; }

    virtual int prealloc(int expectedN) override;

    virtual std::ostream &printstream(std::ostream &output, int dep) const override;
    virtual std::istream &inputstream(std::istream &input          )       override;

    virtual void assign       (const ML_Base &src, int onlySemiCopy = 0) override;
    virtual void semicopy     (const ML_Base &src)                       override;
    virtual void qswapinternal(      ML_Base &b)                         override;

    // Information functions (training data):

    virtual int type(void)      const override { return 500; }
    virtual int subtype(void)   const override { return 0;   }
    virtual int tspaceDim(void) const override { return 1;   }

    virtual int isVarDefined(void) const override { return 2; }

    virtual char gOutType(void) const override { return 'R'; }
    virtual char hOutType(void) const override { return 'R'; }
    virtual char targType(void) const override { return 'R'; }

    virtual int isUnderlyingScalar(void) const override { return 1; }
    virtual int isUnderlyingVector(void) const override { return 0; }
    virtual int isUnderlyingAnions(void) const override { return 0; }

    virtual int getInternalClass(const gentype &y) const override;

    // Training set modification:

    virtual int addTrainingVector (int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2) override;
    virtual int qaddTrainingVector(int i, const gentype &y,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2) override;

    virtual int addTrainingVector (int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override { return LSV_Generic::addTrainingVector (i,y,x,Cweigh,epsweigh); }
    virtual int qaddTrainingVector(int i, const Vector<gentype> &y,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override { return LSV_Generic::qaddTrainingVector(i,y,x,Cweigh,epsweigh); }

    virtual int removeTrainingVector(int i                                      ) override { SparseVector<gentype> x; gentype y; return removeTrainingVector(i,y,x); }
    virtual int removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x) override;
    virtual int removeTrainingVector(int i, int num                             ) override;

    virtual int sety(int                i, const gentype         &y) override;
    virtual int sety(const Vector<int> &i, const Vector<gentype> &y) override;
    virtual int sety(                      const Vector<gentype> &y) override;

    virtual int sety(int                i,       double          y) override;
    virtual int sety(const Vector<int> &i, const Vector<double> &y) override;
    virtual int sety(                      const Vector<double> &y) override;

    virtual int setd(int                i, int                nd) override;
    virtual int setd(const Vector<int> &i, const Vector<int> &nd) override;
    virtual int setd(                      const Vector<int> &nd) override;

    // General modification and autoset functions

    virtual int scale(double a) override;
    virtual int reset(void) override;
    virtual int restart(void) override { LSV_Scalar temp; *this = temp; return 1; }

    // Training functions:

    virtual int train(int &res)                              override { svmvolatile int killSwitch = 0; return train(res,killSwitch); }
    virtual int train(int &res, svmvolatile int &killSwitch) override;

    // Use functions

    virtual int gh(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = nullptr) const override;
    virtual int gh(gentype &resh, gentype &resg, const SparseVector<gentype> &x, int retaltg = 0, const vecInfo *xinf = nullptr, gentype ***pxyprodx = nullptr) const override { return LSV_Generic::gh(resh,resg,x,retaltg,xinf,pxyprodx); }

    virtual double eTrainingVector(int i) const override;

    virtual double         &dedgTrainingVector(double         &res, int i) const override;
    virtual Vector<double> &dedgTrainingVector(Vector<double> &res, int i) const override { dedgTrainingVector((res.resize(1))("&",0),i);   return res; }
    virtual d_anion        &dedgTrainingVector(d_anion        &res, int i) const override { dedgTrainingVector((res.setorder(0))("&",0),i); return res; }
    virtual gentype        &dedgTrainingVector(gentype        &res, int i) const override { dedgTrainingVector(res.force_double(),i);       return res;  }

    virtual double &d2edg2TrainingVector(double &res, int i) const override;

    virtual double dedKTrainingVector(int i, int j) const override { double tmp; return dedgTrainingVector(tmp,i)*alphaR()(j); }
    virtual Vector<double> &dedKTrainingVector(Vector<double> &res, int i) const override;
    virtual Matrix<double> &dedKTrainingVector(Matrix<double> &res) const override;

    virtual int predcov(gentype &resv_pred, gentype &resv, gentype &resmu, int ia, int ib, int ii, double sigmaweighti = 1.0                                                                                                                                                                           ) const;
    virtual int predcov(gentype &resv_pred, gentype &resv, gentype &resmu, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xx, double sigmaweighti = 1.0, const vecInfo *xainf = nullptr, const vecInfo *xbinf = nullptr, const vecInfo *xxinf = nullptr) const { return LSV_Generic::predcov(resv_pred,resv,resmu,xa,xb,xx,sigmaweighti,xainf,xbinf,xxinf); }

    virtual int cov(gentype &resv, gentype &resmu, int i, int j,                                                                                                                     gentype ***pxyprodi = nullptr, gentype ***pxyprodj = nullptr, gentype **pxyprodij = nullptr) const override;
    virtual int cov(gentype &resv, gentype &resmu, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo *xainf = nullptr, const vecInfo *xbinf = nullptr, gentype ***pxyprodx = nullptr, gentype ***pxyprody = nullptr, gentype **pxyprodij = nullptr) const override { return LSV_Generic::cov(resv,resmu,xa,xb,xainf,xbinf,pxyprodx,pxyprody,pxyprodij); }

    // ================================================================
    //     Common functions for all LS-SVMs
    // ================================================================

    virtual int setgamma(const Vector<gentype> &newgamma) override;
    virtual int setdelta(const gentype         &newdelta) override;

    // ================================================================
    //     Required by K2xfer
    // ================================================================

    virtual       double          biasR (void) const override { return dbiasR;  }
    virtual const Vector<double> &alphaR(void) const override { return dalphaR; }

private:

    double ghUnbiasedUnsquaredNotundirectedgradIneg(int i, int xtangi, gentype ***pxyprodi) const;
    double ghUnbiasedSquaredNotundirectedgradIneg(int i, int xtangi) const;

    virtual gentype &makezero(gentype &val)
    {
        val.force_double() = 0.0;

        return val;
    }

public:
    int localtrain(int &res, svmvolatile int &killSwitch); // just the training, avoid dealing with gentype
    void fintrain(void); // finish off, set stuff nicely
private:

    Vector<double> dalphaR;
    double dbiasR;

public: // for simplicity when training GPR_SCALAR with inequalities
    Vector<double> alltraintargR;
};

inline double norm2(const LSV_Scalar &a);
inline double abs2 (const LSV_Scalar &a);

inline double norm2(const LSV_Scalar &a) { return a.RKHSnorm(); }
inline double abs2 (const LSV_Scalar &a) { return a.RKHSabs();  }

inline void qswap(LSV_Scalar &a, LSV_Scalar &b)
{
    a.qswapinternal(b);

    return;
}

inline LSV_Scalar &setzero(LSV_Scalar &a)
{
    a.restart();

    return a;
}

inline void LSV_Scalar::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    LSV_Scalar &b = dynamic_cast<LSV_Scalar &>(bb.getML());

    LSV_Generic::qswapinternal(b);

    qswap(dalphaR      ,b.dalphaR      );
    qswap(dbiasR       ,b.dbiasR       );
    qswap(alltraintargR,b.alltraintargR);

    return;
}

inline void LSV_Scalar::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const LSV_Scalar &b = dynamic_cast<const LSV_Scalar &>(bb.getMLconst());

    LSV_Generic::semicopy(b);

    dalphaR       = b.dalphaR;
    dbiasR        = b.dbiasR;
//    alltraintargR = b.alltraintargR;

    return;
}

inline void LSV_Scalar::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const LSV_Scalar &src = dynamic_cast<const LSV_Scalar &>(bb.getMLconst());

    LSV_Generic::assign(src,onlySemiCopy);

    dalphaR       = src.dalphaR;
    dbiasR        = src.dbiasR;
    alltraintargR = src.alltraintargR;

    return;
}

#endif
