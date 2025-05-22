
//
// Simple kernel block
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "blk_mercer.hpp"


#define MINUINTSIZE 16

std::ostream &BLK_Mercer::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Mercer kernel wrapper block\n";

    repPrint(output,'>',dep) << "Mercer cache: " << merCache << "\n";
    repPrint(output,'>',dep) << "Mercer hit:   " << merHit   << "\n";

    repPrint(output,'>',dep) << "Mercer cache (variance): " << merCacheVar << "\n";
    repPrint(output,'>',dep) << "Mercer hit (variance):   " << merHitVar   << "\n";

    return BLK_Generic::printstream(output,dep+1);
}

std::istream &BLK_Mercer::inputstream(std::istream &input )
{
    wait_dummy dummy;

    input >> dummy; input >> merCache;
    input >> dummy; input >> merHit;

    input >> dummy; input >> merCacheVar;
    input >> dummy; input >> merHitVar;

    return BLK_Generic::inputstream(input);
}

int BLK_Mercer::setmercachesize(int nv)
{
    NiceAssert( nv >= -1 );

    int s = ( nv >= 0 ) ? nv : 0;
    int sc = ( nv > 0 ) ? 1+((nv-1)/MINUINTSIZE) : 0;

    merCache.resize(s,s);
    merHit.resize(s,sc);

    merHit = 0;

    merCacheVar.resize(s,s);
    merHitVar.resize(s,sc);

    merHitVar = 0;

    return BLK_Generic::setmercachesize(nv);
}

int BLK_Mercer::isKVarianceNZ(void) const
{
    return ML_Base::isKVarianceNZ();
}

void BLK_Mercer::K0xfer(gentype &res, int &minmaxind, int typeis,
                        const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                        int xdim, int densetype, int resmode, int mlid) const
{
    // Can't cache 0-kernels

    ML_Base::K0xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xdim,densetype,resmode,mlid);

    return;
}

void BLK_Mercer::K0xfer(double &res, int &minmaxind, int typeis,
                        double xyprod, double yxprod, double diffis,
                        int xdim, int densetype, int resmode, int mlid) const
{
    // Can't cache 0-kernels

    ML_Base::K0xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xdim,densetype,resmode,mlid);

    return;
}

void BLK_Mercer::K1xfer(gentype &res, int &minmaxind, int typeis,
                        const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                        const SparseVector<gentype> &xa, 
                        const vecInfo &xainfo, 
                        int ia, 
                        int xdim, int densetype, int resmode, int mlid) const
{
    // Can't cache 4-kernels

    ML_Base::K1xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xainfo,ia,xdim,densetype,resmode,mlid);

    return;
}

void BLK_Mercer::K1xfer(double &res, int &minmaxind, int typeis,
                        double xyprod, double yxprod, double diffis,
                        const SparseVector<gentype> &xa, 
                        const vecInfo &xainfo, 
                        int ia, 
                        int xdim, int densetype, int resmode, int mlid) const
{
    // Can't cache 4-kernels

    ML_Base::K1xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xainfo,ia,xdim,densetype,resmode,mlid);

    return;
}

void BLK_Mercer::K2xfer(gentype &dxyprod, gentype &ddiffis, gentype &res, int &minmaxind, int typeis, 
                        const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                        const vecInfo &xainfo, const vecInfo &xbinfo,
                        int i, int j,
                        int xdim, int densetype, int resmode, int mlid) const
{
    if ( ( mercachesize() >= 0 ) && ( i >= 0 ) && ( j >= 0 ) && !resmode )
    {
        NiceAssert( i <= mercachesize() );
        NiceAssert( j <= mercachesize() );

        // If no hit then need to pre-calculate kernel first

        if ( !( merHit(i,j/MINUINTSIZE) & (1<<(j%MINUINTSIZE)) ) )
        {
            merHit("&",i,j/MINUINTSIZE) |= (1<<(j%MINUINTSIZE));

            if ( !mercachenorm() )
            {
                ML_Base::K2xfer(dxyprod,ddiffis,res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xainfo,xbinfo,i,j,xdim,densetype,resmode,mlid);
            }

            else
            {
                gentype tma;
                gentype tmb;

                ML_Base::K2xfer(dxyprod,ddiffis,res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xainfo,xbinfo,i,j,xdim,densetype,resmode,mlid);
                ML_Base::K2xfer(dxyprod,ddiffis,tma,minmaxind,typeis,xyprod,yxprod,diffis,xa,xa,xainfo,xainfo,i,i,xdim,densetype,resmode,mlid);
                ML_Base::K2xfer(dxyprod,ddiffis,tmb,minmaxind,typeis,xyprod,yxprod,diffis,xb,xb,xbinfo,xbinfo,j,j,xdim,densetype,resmode,mlid);

                OP_sqrt(tma);
                OP_sqrt(tmb);

                res /= tma;
                res /= tmb;
            }

errstream() << "?1";
            merCache("&",i,j) = res;
        }

        else
        {
            res = merCache(i,j);
        }
    }

    else if ( ( mercachesize() >= 0 ) && ( i >= 0 ) && ( j >= 0 ) && ( resmode == 0x80 ) )
    {
        NiceAssert( i <= mercachesize() );
        NiceAssert( j <= mercachesize() );
        NiceAssert( !mercachenorm() );

        // If no hit then need to pre-calculate kernel first

        if ( !( merHitVar(i,j/MINUINTSIZE) & (1<<(j%MINUINTSIZE)) ) )
        {
            merHitVar("&",i,j/MINUINTSIZE) |= (1<<(j%MINUINTSIZE));

            ML_Base::K2xfer(dxyprod,ddiffis,res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xainfo,xbinfo,i,j,xdim,densetype,resmode,mlid);

errstream() << "?2";
            merCacheVar("&",i,j) = res;
        }

        else
        {
            res = merCache(i,j);
        }
    }

    else
    {
errstream() << "?3";
        ML_Base::K2xfer(dxyprod,ddiffis,res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xainfo,xbinfo,i,j,xdim,densetype,resmode,mlid);
    }

    return;
}

void BLK_Mercer::K2xfer(double &dxyprod, double &ddiffis, double &res, int &minmaxind, int typeis,
                        double xyprod, double yxprod, double diffis,
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                        const vecInfo &xainfo, const vecInfo &xbinfo,
                        int i, int j,
                        int xdim, int densetype, int resmode, int mlid) const
{
    if ( ( mercachesize() >= 0 ) && ( i >= 0 ) && ( j >= 0 ) && !resmode )
    {
        NiceAssert( i <= mercachesize() );
        NiceAssert( j <= mercachesize() );

        // If no hit then need to pre-calculate kernel first

        if ( !( merHit(i,j/MINUINTSIZE) & (1<<(j%MINUINTSIZE)) ) )
        {
            merHit("&",i,j/MINUINTSIZE) |= (1<<(j%MINUINTSIZE));

            if ( !mercachenorm() )
            {
                ML_Base::K2xfer(dxyprod,ddiffis,res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xainfo,xbinfo,i,j,xdim,densetype,resmode,mlid);
            }

            else
            {
                double tma;
                double tmb;

                ML_Base::K2xfer(dxyprod,ddiffis,res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xainfo,xbinfo,i,j,xdim,densetype,resmode,mlid);
                ML_Base::K2xfer(dxyprod,ddiffis,tma,minmaxind,typeis,xyprod,yxprod,diffis,xa,xa,xainfo,xainfo,i,i,xdim,densetype,resmode,mlid);
                ML_Base::K2xfer(dxyprod,ddiffis,tmb,minmaxind,typeis,xyprod,yxprod,diffis,xb,xb,xbinfo,xbinfo,j,j,xdim,densetype,resmode,mlid);

                OP_sqrt(tma);
                OP_sqrt(tmb);

                res /= tma;
                res /= tmb;
            }

errstream() << "!1";
            merCache("&",i,j) = res;
        }

        else
        {
            res = (double) merCache(i,j);
        }
    }

    else if ( ( mercachesize() >= 0 ) && ( i >= 0 ) && ( j >= 0 ) && ( resmode == 0x80 ) )
    {
        NiceAssert( i <= mercachesize() );
        NiceAssert( j <= mercachesize() );
        NiceAssert( !mercachenorm() );

        // If no hit then need to pre-calculate kernel first

        if ( !( merHitVar(i,j/MINUINTSIZE) & (1<<(j%MINUINTSIZE)) ) )
        {
            merHitVar("&",i,j/MINUINTSIZE) |= (1<<(j%MINUINTSIZE));

            ML_Base::K2xfer(dxyprod,ddiffis,res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xainfo,xbinfo,i,j,xdim,densetype,resmode,mlid);

errstream() << "!2";
            merCacheVar("&",i,j) = res;
        }

        else
        {
            res = (double) merCache(i,j);
        }
    }

    else
    {
errstream() << "!3";
        ML_Base::K2xfer(dxyprod,ddiffis,res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xainfo,xbinfo,i,j,xdim,densetype,resmode,mlid);
    }

    return;
}

void BLK_Mercer::K3xfer(gentype &res, int &minmaxind, int typeis,
                        const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, 
                        const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, 
                        int ia, int ib, int ic, 
                        int xdim, int densetype, int resmode, int mlid) const
{
    // Can't cache 4-kernels

    ML_Base::K3xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xc,xainfo,xbinfo,xcinfo,ia,ib,ic,xdim,densetype,resmode,mlid);

    return;
}

void BLK_Mercer::K3xfer(double &res, int &minmaxind, int typeis,
                        double xyprod, double yxprod, double diffis,
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, 
                        const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, 
                        int ia, int ib, int ic, 
                        int xdim, int densetype, int resmode, int mlid) const
{
    // Can't cache 4-kernels

    ML_Base::K3xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xc,xainfo,xbinfo,xcinfo,ia,ib,ic,xdim,densetype,resmode,mlid);

    return;
}

void BLK_Mercer::K4xfer(gentype &res, int &minmaxind, int typeis,
                        const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                        const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
                        int ia, int ib, int ic, int id,
                        int xdim, int densetype, int resmode, int mlid) const
{
    // Can't cache 4-kernels

    ML_Base::K4xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,ia,ib,ic,id,xdim,densetype,resmode,mlid);

    return;
}

void BLK_Mercer::K4xfer(double &res, int &minmaxind, int typeis,
                        double xyprod, double yxprod, double diffis,
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                        const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
                        int ia, int ib, int ic, int id,
                        int xdim, int densetype, int resmode, int mlid) const
{
    // Can't cache 4-kernels

    ML_Base::K4xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,ia,ib,ic,id,xdim,densetype,resmode,mlid);

    return;
}

void BLK_Mercer::Kmxfer(gentype &res, int &minmaxind, int typeis,
                        const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                        Vector<const SparseVector<gentype> *> &x,
                        Vector<const vecInfo *> &xinfo,
                        Vector<int> &i,
                        int xdim, int m, int densetype, int resmode, int mlid) const
{
    // Can't cache m>2-kernels (back-calling automatic if m = 0,2,4)

    ML_Base::Kmxfer(res,minmaxind,typeis,xyprod,yxprod,diffis,x,xinfo,i,xdim,m,densetype,resmode,mlid);

    return;
}

void BLK_Mercer::Kmxfer(double &res, int &minmaxind, int typeis,
                        double xyprod, double yxprod, double diffis,
                        Vector<const SparseVector<gentype> *> &x,
                        Vector<const vecInfo *> &xinfo,
                        Vector<int> &i,
                        int xdim, int m, int densetype, int resmode, int mlid) const
{
    // Can't cache m>2-kernels (back-calling automatic if m = 0,2,4)

    ML_Base::Kmxfer(res,minmaxind,typeis,xyprod,yxprod,diffis,x,xinfo,i,xdim,m,densetype,resmode,mlid);

    return;
}
