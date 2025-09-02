
//
// Multi-directional ranking SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _svm_mvrank_h
#define _svm_mvrank_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "svm_planar.hpp"





class SVM_MvRank;

// Swap function

inline void qswap(SVM_MvRank &a, SVM_MvRank &b);


class SVM_MvRank : public SVM_Planar
{
public:

    // Constructors, destructors, assignment etc..

    SVM_MvRank();
    SVM_MvRank(const SVM_MvRank &src);
    SVM_MvRank(const SVM_MvRank &src, const ML_Base *xsrc);
    SVM_MvRank &operator=(const SVM_MvRank &src) { assign(src); return *this; }
    virtual ~SVM_MvRank();

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0) override;
    virtual void semicopy(const ML_Base &src) override;
    virtual void qswapinternal(ML_Base &b) override;

    virtual int restart(void) override { SVM_MvRank temp; *this = temp; return 1; }

    virtual std::ostream &printstream(std::ostream &output, int dep) const override;
    virtual std::istream &inputstream(std::istream &input) override;

    // Information functions (training data):

    virtual int type(void)    const override { return 17; }
    virtual int subtype(void) const override { return 0;  }

    virtual int    maxitermvrank(void) const override { return xmaxitermvrank; }
    virtual double lrmvrank(void)      const override { return xlrmvrank;      }
    virtual double ztmvrank(void)      const override { return xztmvrank;      }

    virtual double betarank(void) const override { return xbetarank; }

    // Modification

    virtual int setmaxitermvrank(int nv) override { xmaxitermvrank = nv; return 1; }
    virtual int setlrmvrank(double nv)   override { xlrmvrank      = nv; return 1; }
    virtual int setztmvrank(double nv)   override { xztmvrank      = nv; return 1; }

    virtual int setbetarank(double nv) override { xbetarank = nv; return 1; }

    // Training

    virtual int train(int &res)                              override { svmvolatile int killSwitch = 0; return train(res,killSwitch); }
    virtual int train(int &res, svmvolatile int &killSwitch) override;

protected:

    // Protected function passthrough

    void calcLalpha(Matrix<double> &res);

private:

    int xmaxitermvrank;
    double xlrmvrank;
    double xztmvrank;
    double xbetarank;
};

inline double norm2(const SVM_MvRank &a);
inline double abs2 (const SVM_MvRank &a);

inline double norm2(const SVM_MvRank &a) { return a.RKHSnorm(); }
inline double abs2 (const SVM_MvRank &a) { return a.RKHSabs();  }

inline void qswap(SVM_MvRank &a, SVM_MvRank &b)
{
    a.qswapinternal(b);

    return;
}

inline void SVM_MvRank::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    SVM_MvRank &b = dynamic_cast<SVM_MvRank &>(bb.getML());

    SVM_Planar::qswapinternal(b);
    
    qswap(xmaxitermvrank,b.xmaxitermvrank);
    qswap(xlrmvrank     ,b.xlrmvrank     );
    qswap(xztmvrank     ,b.xztmvrank     );
    qswap(xbetarank     ,b.xbetarank     );

    return;
}

inline void SVM_MvRank::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const SVM_MvRank &b = dynamic_cast<const SVM_MvRank &>(bb.getMLconst());

    SVM_Planar::semicopy(b);

    xmaxitermvrank = b.xmaxitermvrank;
    xlrmvrank      = b.xlrmvrank;
    xztmvrank      = b.xztmvrank;
    xbetarank      = b.xbetarank;

    return;
}

inline void SVM_MvRank::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const SVM_MvRank &src = dynamic_cast<const SVM_MvRank &>(bb.getMLconst());

    SVM_Planar::assign(static_cast<const SVM_Planar &>(src),onlySemiCopy);

    xmaxitermvrank = src.xmaxitermvrank;
    xlrmvrank      = src.xlrmvrank;
    xztmvrank      = src.xztmvrank;
    xbetarank      = src.xbetarank;

    return;
}

#endif
