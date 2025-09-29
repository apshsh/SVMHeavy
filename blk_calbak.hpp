
//
// Function callback block
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _blk_calbak_h
#define _blk_calbak_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "blk_generic.hpp"


// Basic user I/O


class BLK_CalBak;


// Swap and zeroing (restarting) functions

inline void qswap(BLK_CalBak &a, BLK_CalBak &b);


class BLK_CalBak : public BLK_Generic
{
public:

    // Assumptions: all virtual functions inherited from ML_Base are left
    // unchanged in this class.

    // Constructors, destructors, assignment etc..

    BLK_CalBak(int xisIndPrune = 0)                                             : BLK_Generic(xisIndPrune) { setaltx(nullptr);                 return; }
    BLK_CalBak(const BLK_CalBak &src, int xisIndPrune = 0)                      : BLK_Generic(xisIndPrune) { setaltx(nullptr); assign(src,0);  return; }
    BLK_CalBak(const BLK_CalBak &src, const ML_Base *xsrc, int xisIndPrune = 0) : BLK_Generic(xisIndPrune) { setaltx(xsrc); assign(src,-1); return; }
    BLK_CalBak &operator=(const BLK_CalBak &src) { assign(src); return *this; }
    virtual ~BLK_CalBak() { return; }

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0) override;
    virtual void semicopy(const ML_Base &src) override;
    virtual void qswapinternal(ML_Base &b) override;

    virtual std::ostream &printstream(std::ostream &output, int dep) const override;
    virtual std::istream &inputstream(std::istream &input ) override;

    virtual int type(void)    const override { return 208; }
    virtual int subtype(void) const override { return 0;   }

    // Kernel transfer by callback

    virtual void K0xfer(                                    gentype &res, int &minmaxind, int typeis, const gentype &xyprod, const gentype &yxprod, const gentype &diffis, int xdim, int densetype, int resmode, int mlid) const override;
    virtual void K1xfer(                                    gentype &res, int &minmaxind, int typeis, const gentype &xyprod, const gentype &yxprod, const gentype &diffis, const SparseVector<gentype> &xa, const vecInfo &xainfo, int ia, int xdim, int densetype, int resmode, int mlid) const override;
    virtual void K2xfer(gentype &dxyprod, gentype &ddiffis, gentype &res, int &minmaxind, int typeis, const gentype &xyprod, const gentype &yxprod, const gentype &diffis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, int ia, int ib, int xdim, int densetype, int resmode, int mlid) const override;
    virtual void K3xfer(                                    gentype &res, int &minmaxind, int typeis, const gentype &xyprod, const gentype &yxprod, const gentype &diffis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, int ia, int ib, int ic, int xdim, int densetype, int resmode, int mlid) const override;
    virtual void K4xfer(                                    gentype &res, int &minmaxind, int typeis, const gentype &xyprod, const gentype &yxprod, const gentype &diffis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, int ia, int ib, int ic, int id, int xdim, int densetype, int resmode, int mlid) const override;
    virtual void Kmxfer(                                    gentype &res, int &minmaxind, int typeis, const gentype &xyprod, const gentype &yxprod, const gentype &diffis, Vector<const SparseVector<gentype> *> &x, Vector<const vecInfo *> &xzinfo, Vector<int> &i, int xdim, int m, int densetype, int resmode, int mlid) const override;

    virtual void K0xfer(                                  double &res, int &minmaxind, int typeis, double xyprod, double yxprod, double diffis, int xdim, int densetype, int resmode, int mlid) const override;
    virtual void K1xfer(                                  double &res, int &minmaxind, int typeis, double xyprod, double yxprod, double diffis, const SparseVector<gentype> &xa, const vecInfo &xainfo, int ia, int xdim, int densetype, int resmode, int mlid) const override;
    virtual void K2xfer(double &dxyprod, double &ddiffis, double &res, int &minmaxind, int typeis, double xyprod, double yxprod, double diffis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, int ia, int ib, int xdim, int densetype, int resmode, int mlid) const override;
    virtual void K3xfer(                                  double &res, int &minmaxind, int typeis, double xyprod, double yxprod, double diffis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, int ia, int ib, int ic, int xdim, int densetype, int resmode, int mlid) const override;
    virtual void K4xfer(                                  double &res, int &minmaxind, int typeis, double xyprod, double yxprod, double diffis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, int ia, int ib, int ic, int id, int xdim, int densetype, int resmode, int mlid) const override;
    virtual void Kmxfer(                                  double &res, int &minmaxind, int typeis, double xyprod, double yxprod, double diffis, Vector<const SparseVector<gentype> *> &x, Vector<const vecInfo *> &xzinfo, Vector<int> &i, int xdim, int m, int densetype, int resmode, int mlid) const override;

    // Evaluation Functions:

    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = nullptr) const override;
};

inline double norm2(const BLK_CalBak &a);
inline double abs2 (const BLK_CalBak &a);

inline double norm2(const BLK_CalBak &a) { return a.RKHSnorm(); }
inline double abs2 (const BLK_CalBak &a) { return a.RKHSabs();  }

inline void qswap(BLK_CalBak &a, BLK_CalBak &b)
{
    a.qswapinternal(b);

    return;
}

inline void BLK_CalBak::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    BLK_CalBak &b = dynamic_cast<BLK_CalBak &>(bb.getML());

    BLK_Generic::qswapinternal(b);

    return;
}

inline void BLK_CalBak::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const BLK_CalBak &b = dynamic_cast<const BLK_CalBak &>(bb.getMLconst());

    BLK_Generic::semicopy(b);

    return;
}

inline void BLK_CalBak::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const BLK_CalBak &src = dynamic_cast<const BLK_CalBak &>(bb.getMLconst());

    BLK_Generic::assign(src,onlySemiCopy);

    return;
}

#endif
