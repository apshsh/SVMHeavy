
//
// Scalar SVM for kernel construction only
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//


//
// This is a bare-minimum implementation of SVM to enable kernel inheritance via Kxfer
//


#ifndef _svm_kconst_h
#define _svm_kconst_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <string.h>
#include <string>
#include "svm_generic.hpp"


// Swap function

class SVM_KConst;

inline void qswap(SVM_KConst &a, SVM_KConst &b);


class SVM_KConst : public SVM_Generic
{
public:

    // Constructors, destructors, assignment operators and similar

    SVM_KConst() : SVM_Generic() { useKwe = 0; ddbiasR = 0; return; }
    SVM_KConst(const SVM_KConst &src) : SVM_Generic(src) { useKwe = 0; ddbiasR = src.ddbiasR; ddalphaR = src.ddalphaR; return; }
    SVM_KConst(const SVM_KConst &src, const ML_Base *xsrc) : SVM_Generic(src,xsrc) { useKwe = 0; ddbiasR = src.ddbiasR; ddalphaR = src.ddalphaR; return; }
    SVM_KConst &operator=(const SVM_KConst &src) { assign(src); return *this; }
    virtual ~SVM_KConst() { return; }

    virtual int prealloc(int expectedN)  override { ddalphaR.prealloc(expectedN); return SVM_Generic::prealloc(expectedN); }
    virtual int preallocsize(void) const override {                               return SVM_Generic::preallocsize();      }
    virtual void setmemsize(int memsize) override {                               return SVM_Generic::setmemsize(memsize); }

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0) override;
    virtual void semicopy(const ML_Base &src) override;
    virtual void qswapinternal(ML_Base &b) override;

    virtual int getparam (int ind, gentype         &val, const gentype         &xa, int ia, const gentype         &xb, int ib, charptr &desc) const override { return SVM_Generic::getparam (ind,val,xa,ia,xb,ib,desc); }
    virtual int egetparam(int ind, Vector<gentype> &val, const Vector<gentype> &xa, int ia, const Vector<gentype> &xb, int ib               ) const override { return SVM_Generic::egetparam(ind,val,xa,ia,xb,ib     ); }

    virtual std::ostream &printstream(std::ostream &output, int dep) const override;
    virtual std::istream &inputstream(std::istream &input) override;

    virtual       ML_Base &getML     (void)       override { return static_cast<      ML_Base &>(getSVM());      }
    virtual const ML_Base &getMLconst(void) const override { return static_cast<const ML_Base &>(getSVMconst()); }

    // Information:

    virtual int NS (void)  const override { return N();  }
    virtual int NF (void)  const override { return N();  }
    virtual int NNC(int d) const override { return ( d == 2 ) ? N() : 0;  }

    virtual int type(void)       const override { return 21; }
    virtual int subtype(void)    const override { return 0;  }

    virtual const Vector<double> &alphaR(void) const override { return ddalphaR; }
    virtual       double          biasR (void) const override { return ddbiasR;  }

    virtual int addTrainingVector (int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2) override;
    virtual int qaddTrainingVector(int i, const gentype &y,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2) override;

    virtual int addTrainingVector (int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override;
    virtual int qaddTrainingVector(int i, const Vector<gentype> &y,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override;

    virtual int removeTrainingVector(int i                                      ) override { SparseVector<gentype> x; gentype y; return removeTrainingVector(i,y,x); }
    virtual int removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x) override;
    virtual int removeTrainingVector(int i, int num                             ) override { return ML_Base::removeTrainingVector(i,num); }

    virtual int isUnderlyingScalar(void) const override { return 1; }
    virtual int isUnderlyingVector(void) const override { return 0; }
    virtual int isUnderlyingAnions(void) const override { return 0; }

    virtual int kconstWeights(void) const override { return useKwe; }

    // The basics

    virtual int setAlphaR(const Vector<double> &newAlpha) override;
    virtual int setBiasR(double newBias) override;

    virtual int randomise(double sparsity) override;

    virtual int setkconstWeights(int nv) override { useKwe = nv; return 1; }

    // Evaluation (for inheritance)

    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodx = nullptr) const override;
    virtual double eTrainingVector(int i) const override;

protected:
    // Inner-product cache: used to accelerate kernel transfer

    virtual int isxymat(const MercerKernel &altK) const override
    {
        return ( xyvalid && ( altK.suggestXYcache() || altK.wantsXYprod() ) ) ? 1 : 0;
    }

    virtual const Matrix<double> &getxymat(const MercerKernel &altK) const override
    {
        (void) altK;

        NiceThrow("Can't do m-kernel xymat in svm_kconst (you can for m=0,1,2,3,4, but not in general).");

        const static Matrix<double> dummy;

        return dummy;
    }

    virtual const double &getxymatelm(const MercerKernel &altK, int i, int j) const override
    {
        (void) altK;

        NiceAssert( isxymat(altK) );

        if ( ( i >= 0 ) && ( j >= 0 ) )
        {
            return (KxferDatStore.gxvinnerProdsFull)(i,j);
        }

        else if ( ( i == -42 ) && ( j >= 0 ) )
        {
            return (KxferDatStore.allxainnerProdsFull)(xyvalid)(j);
        }

        else if ( ( i == -43 ) && ( j >= 0 ) )
        {
            return (KxferDatStore.allxbinnerProdsFull)(xyvalid)(j);
        }

        else if ( ( i >= 0 ) && ( i == -42 ) )
        {
            return (KxferDatStore.allxainnerProdsFull)(xyvalid)(i);
        }

        else if ( ( i >= 0 ) && ( i == -43 ) )
        {
            return (KxferDatStore.allxbinnerProdsFull)(xyvalid)(i);
        }

        else if ( ( i == -42 ) && ( j == -42 ) )
        {
            return (KxferDatStore.allxaxainnerProd)(xyvalid);
        }

        else if ( ( i == -43 ) && ( j == -43 ) )
        {
            return (KxferDatStore.allxbxbinnerProd)(xyvalid);
        }

        return (KxferDatStore.allxaxbinnerProd)(xyvalid);
    }

private:
    virtual int gTrainingVector(double &res, int &unusedvar, int i, int raw = 0, gentype ***pxyprodi = nullptr) const;

    virtual void fastg(double &res) const override;
    virtual void fastg(double &res, int ia, const SparseVector<gentype> &xa, const vecInfo &xainfo) const override;
    virtual void fastg(double &res, int ia, int ib, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo) const override;

    virtual void fastg(gentype &res) const override;
    virtual void fastg(gentype &res, int ia, const SparseVector<gentype> &xa, const vecInfo &xainfo) const override;
    virtual void fastg(gentype &res, int ia, int ib, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo) const override;

    Vector<double> ddalphaR;
    double ddbiasR;

    int useKwe; // set this and alphaR() is loaded into the kernel as weight, for use when tuning weights on 800 kernels
};

inline double norm2(const SVM_KConst &a);
inline double abs2 (const SVM_KConst &a);

inline double norm2(const SVM_KConst &a) { return a.RKHSnorm(); }
inline double abs2 (const SVM_KConst &a) { return a.RKHSabs();  }

inline void qswap(SVM_KConst &a, SVM_KConst &b)
{
    a.qswapinternal(b);

    return;
}

inline void SVM_KConst::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    SVM_KConst &b = dynamic_cast<SVM_KConst &>(bb.getML());

    SVM_Generic::qswapinternal(b);

    qswap(ddalphaR,b.ddalphaR);
    qswap(ddbiasR ,b.ddbiasR );
    qswap(useKwe  ,b.useKwe  );

    return;
}

inline void SVM_KConst::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const SVM_KConst &b = dynamic_cast<const SVM_KConst &>(bb.getMLconst());

    SVM_Generic::semicopy(b);

    ddalphaR = b.ddalphaR;
    ddbiasR  = b.ddbiasR;
    useKwe   = b.useKwe;

    return;
}

inline void SVM_KConst::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const SVM_KConst &src = dynamic_cast<const SVM_KConst &>(bb.getMLconst());

    SVM_Generic::assign(src,onlySemiCopy);

    ddalphaR = src.ddalphaR;
    ddbiasR  = src.ddbiasR;
    useKwe   = src.useKwe;

    return;
}

#endif
