
//
// Generic target SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _svm_gentyp_h
#define _svm_gentyp_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "svm_vector.hpp"



class SVM_Gentyp;


// Swap and zeroing (restarting) functions

inline void qswap(SVM_Gentyp &a, SVM_Gentyp &b);
inline SVM_Gentyp &setzero(SVM_Gentyp &a);

class SVM_Gentyp : public SVM_Vector
{
public:

    // Constructors, destructors, assignment etc..

    SVM_Gentyp();
    SVM_Gentyp(const SVM_Gentyp &src);
    SVM_Gentyp(const SVM_Gentyp &src, const ML_Base *xsrc);
    SVM_Gentyp &operator=(const SVM_Gentyp &src) { assign(src); return *this; }
    virtual ~SVM_Gentyp();

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0) override;
    virtual void semicopy(const ML_Base &src) override;
    virtual void qswapinternal(ML_Base &b) override;

    virtual std::ostream &printstream(std::ostream &output, int dep) const override;
    virtual std::istream &inputstream(std::istream &input ) override;

    // Information functions (training data):

    virtual int type(void)    const override { return 15; }
    virtual int subtype(void) const override { return 0;  }

    virtual double calcDist(const gentype &ha, const gentype &hb, int ia = -1, int db = 2) const override;

    virtual char gOutType(void) const override { return '?'; }
    virtual char hOutType(void) const override { return '?'; }
    virtual char targType(void) const override { return '?'; }

    virtual const Vector<gentype>         &y (void) const override { return locyval; }
    virtual const Vector<double>          &yR(void) const override { static thread_local Vector<double>          dummy; NiceThrow("yR not defined in svm_gentyp"); return dummy; }
    virtual const Vector<d_anion>         &yA(void) const override { static thread_local Vector<d_anion>         dummy; NiceThrow("yA not defined in svm_gentyp"); return dummy; }
    virtual const Vector<Vector<double> > &yV(void) const override { static thread_local Vector<Vector<double> > dummy; NiceThrow("yV not defined in svm_gentyp"); return dummy; }
    virtual const Vector<gentype>         &yp (void) const override { static thread_local Vector<gentype>         dummy; NiceThrow("yp  not defined in svm_gentyp"); return dummy; }
    virtual const Vector<double>          &ypR(void) const override { static thread_local Vector<double>          dummy; NiceThrow("ypR not defined in svm_gentyp"); return dummy; }
    virtual const Vector<d_anion>         &ypA(void) const override { static thread_local Vector<d_anion>         dummy; NiceThrow("ypA not defined in svm_gentyp"); return dummy; }
    virtual const Vector<Vector<double> > &ypV(void) const override { static thread_local Vector<Vector<double> > dummy; NiceThrow("ypV not defined in svm_gentyp"); return dummy; }

    // Training set modification - need to overload to maintain counts

    virtual int addTrainingVector (int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2) override;
    virtual int qaddTrainingVector(int i, const gentype &y,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2) override;

    virtual int addTrainingVector (int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override;
    virtual int qaddTrainingVector(int i, const Vector<gentype> &y,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override;

    virtual int removeTrainingVector(int i                                      ) override { SparseVector<gentype> x; gentype y; return removeTrainingVector(i,y,x); }
    virtual int removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x) override;
    virtual int removeTrainingVector(int i, int num                             ) override;

    virtual int sety(int                i, const gentype         &z) override;
    virtual int sety(const Vector<int> &i, const Vector<gentype> &z) override;
    virtual int sety(                      const Vector<gentype> &z) override;

    virtual int sety(int                i, double                z) override { return ML_Base::sety(i,z); }
    virtual int sety(const Vector<int> &i, const Vector<double> &z) override { return ML_Base::sety(i,z); }
    virtual int sety(                      const Vector<double> &z) override { return ML_Base::sety(z);   }

    virtual int sety(int                i, const Vector<double>          &z) override { return ML_Base::sety(i,z); }
    virtual int sety(const Vector<int> &i, const Vector<Vector<double> > &z) override { return ML_Base::sety(i,z); }
    virtual int sety(                      const Vector<Vector<double> > &z) override { return ML_Base::sety(z);   }

    virtual int sety(int                i, const d_anion         &z) override { return ML_Base::sety(i,z); }
    virtual int sety(const Vector<int> &i, const Vector<d_anion> &z) override { return ML_Base::sety(i,z); }
    virtual int sety(                      const Vector<d_anion> &z) override { return ML_Base::sety(z);   }

    virtual const gentype        &y (int i) const override { return locyval(i); }
    virtual       double          yR(int i) const override { return yR()(i); }
    virtual const d_anion        &yA(int i) const override { return yA()(i); }
    virtual const Vector<double> &yV(int i) const override { return yV()(i); }

    // Evaluation Functions:

    virtual int isVarDefined(void) const override { return 0; }

    virtual int ggTrainingVector(               gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = nullptr) const override { gentype resh; return ghTrainingVector(resh,resg,i,retaltg,pxyprodi); }
    virtual int hhTrainingVector(gentype &resh,                int i,                  gentype ***pxyprodi = nullptr) const override { gentype resg; return ghTrainingVector(resh,resg,i,0,      pxyprodi); }
    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = nullptr) const override;

    virtual int ggTrainingVector(double &resg,         int i, int retaltg = 0, gentype ***pxyprodi = nullptr) const override { gentype resgg; int res = ggTrainingVector(resgg,i,retaltg,pxyprodi); resg = (double)                 resgg; return res; }
    virtual int ggTrainingVector(Vector<double> &resg, int i, int retaltg = 0, gentype ***pxyprodi = nullptr) const override { gentype resgg; int res = ggTrainingVector(resgg,i,retaltg,pxyprodi); resg = (const Vector<double> &) resgg; return res; }
    virtual int ggTrainingVector(d_anion &resg,        int i, int retaltg = 0, gentype ***pxyprodi = nullptr) const override { gentype resgg; int res = ggTrainingVector(resgg,i,retaltg,pxyprodi); resg = (const d_anion &)        resgg; return res; }

//    virtual int gg(               gentype &resg, const SparseVector<gentype> &x                 , const vecInfo *xinf = nullptr, gentype ***pxyprodx = nullptr) const override { gentype resh; return gh(resh,resg,x,0,xinf,pxyprodx); }
//    virtual int hh(gentype &resh,                const SparseVector<gentype> &x                 , const vecInfo *xinf = nullptr, gentype ***pxyprodx = nullptr) const override { gentype resg; return gh(resh,resg,x,0,xinf,pxyprodx); }
//    virtual int gh(gentype &resh, gentype &resg, const SparseVector<gentype> &x, int retaltg = 0, const vecInfo *xinf = nullptr, gentype ***pxyprodx = nullptr) const;




    // Generic target specific options
    //
    // Nbasis   = number of basis vectors
    // isBasisY = true (nz) if basis == y vectors, false if user setable
    // Vbasis   = basis vectors
    //
    // setBasisY: basis will be equal to targets
    // setBasisU: basis will be set by user
    // addToBasis: add element to basis
    // removeFromBasis: remove element from basis
    // setBasis: set basis, either single element or all
    // setDefaultProjection: h(x) = u_d.g(x)
    //
    // output kernel: this is used to compute similarity between basis and
    // target functions.

    virtual int NbasisUU(void)    const override { return locbasis.size(); }
    virtual int basisTypeUU(void) const override { return isBasisUser;     }
    virtual int defProjUU(void)   const override { return defbasis;        }

    virtual const Vector<gentype> &VbasisUU(void) const override { return locbasis; }

    virtual int setBasisYUU(void) override;
    virtual int setBasisUUU(void) override;
    virtual int addToBasisUU(int i, const gentype &o) override;
    virtual int removeFromBasisUU(int i) override;
    virtual int setBasisUU(int i, const gentype &o) override;
    virtual int setBasisUU(const Vector<gentype> &o) override;
    virtual int setDefaultProjectionUU(int d) override { defbasis = d; return 1; }
    virtual int setBasisUU(int i, int d) override { return ML_Base::setBasisUU(i,d); }

    // Output basis kernel

    virtual int setUUOutputKernel(const MercerKernel &xkernel, int modind = 1) override;

private:

    // Local y value store

    Vector<gentype> locyval;

    // Setting to control if basis is user controlled (0, default) or
    // fixed to equal y (ie target set).

    int isBasisUser;
    int defbasis;
    Vector<gentype> locbasis;

    // Local basis and factorisation

    Matrix<double> M;
    Chol<double> L;

    // Dummies needed to keep cholesky factorisation

    Matrix<double> dummyGpn;
    Matrix<double> dummyGn;

    // Helper functions

    void ConvertYtoVec(const gentype &src, Vector<double> &dest) const;
    void ConvertYtoOut(const Vector<double> &src, gentype &dest) const;

    void calcMij(double &res, int i, int j) const;
    void calcMiy(double &res, int i, const gentype &y) const;
    void calcMxy(double &res, const gentype &x, const gentype &y) const;
};

inline double norm2(const SVM_Gentyp &a);
inline double abs2 (const SVM_Gentyp &a);

inline double norm2(const SVM_Gentyp &a) { return a.RKHSnorm(); }
inline double abs2 (const SVM_Gentyp &a) { return a.RKHSabs();  }

inline void qswap(SVM_Gentyp &a, SVM_Gentyp &b)
{
    a.qswapinternal(b);

    return;
}

inline SVM_Gentyp &setzero(SVM_Gentyp &a)
{
    a.restart();

    return a;
}

inline void SVM_Gentyp::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    SVM_Gentyp &b = dynamic_cast<SVM_Gentyp &>(bb.getML());

    SVM_Vector::qswapinternal(b);

    qswap(locyval    ,b.locyval    );
    qswap(isBasisUser,b.isBasisUser);
    qswap(locbasis   ,b.locbasis   );
    qswap(M          ,b.M          );
    qswap(L          ,b.L          );
    qswap(dummyGpn   ,b.dummyGpn   );
    qswap(dummyGn    ,b.dummyGn    );
    qswap(defbasis   ,b.defbasis   );

    return;
}

inline void SVM_Gentyp::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const SVM_Gentyp &b = dynamic_cast<const SVM_Gentyp &>(bb.getMLconst());

    SVM_Vector::semicopy(b);

    isBasisUser = b.isBasisUser;
    locbasis    = b.locbasis;
    locyval     = b.locyval;
    M           = b.M;
    L           = b.L;
    dummyGpn    = b.dummyGpn;
    dummyGn     = b.dummyGn;
    defbasis    = b.defbasis;

    return;
}

inline void SVM_Gentyp::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const SVM_Gentyp &src = dynamic_cast<const SVM_Gentyp &>(bb.getMLconst());

    SVM_Vector::assign(src,onlySemiCopy);

    isBasisUser = src.isBasisUser;
    locbasis    = src.locbasis;
    locyval     = src.locyval;
    M           = src.M;
    L           = src.L;
    dummyGpn    = src.dummyGpn;
    dummyGn     = src.dummyGn;
    defbasis    = src.defbasis;

    return;
}

#endif
