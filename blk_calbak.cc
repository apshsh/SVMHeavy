
//
// Callback Function
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
#include "blk_calbak.hpp"


std::ostream &BLK_CalBak::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Callback wrapper block\n";

    return BLK_Generic::printstream(output,dep+1);
}

std::istream &BLK_CalBak::inputstream(std::istream &input )
{
    return BLK_Generic::inputstream(input);
}



































const double *getdbl(const SparseVector<gentype> &x, int &resdim);
const double *getdbl(const SparseVector<gentype> &x, int &resdim)
{
    // Note that nothing much gets created here.  We *might* make altcontent,
    // if it hasn't already been made, but that hangs around to be used over
    // and over, so whatever.  Otherwise we're just returning x in a "simpler"
    // form that can be used by the simple (read: pythonesque) callback.

    resdim = x.indsize();

    x.makealtcontent();

    NiceAssert( x.altcontent );

    return x.altcontent;
}




int BLK_CalBak::ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg, gentype ***pxyprodi) const
{
    NiceAssert( !retaltg );

    (void) retaltg;
    (void) pxyprodi;

    int res = 0;

    const SparseVector<gentype> &xx = x(i);

    if ( callback() )
    {
        res = (*callback())(resg,xx,callbackfndata());
    }

    else
    {
        int xxdim = 0;
        const double *xxx = getdbl(xx,xxdim);

        resg.force_double() = (*callbackalt())(xxx,xxdim,callbackfndata());
    }

    resh = resg;

    return res;
}






void BLK_CalBak::K0xfer(gentype &res, int &minmaxind, int typeis,
                        const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                        int xdim, int densetype, int resmode, int mlid) const
{
    if ( K0callback() )
    {
        (*K0callback())(res,minmaxind,typeis,xdim,densetype,resmode,mlid,K0callbackdata());
    }

    else if ( K0callbackalt() )
    {
        res.force_double() = (*K0callbackalt())(typeis,xdim,densetype,resmode,mlid,K0callbackdata());
    }

    else
    {
        res.force_double() = (*Kcallbackalt())(typeis,(double) (xyprod+yxprod)/2.0,(double) diffis,xdim,densetype,resmode,mlid,Kcallbackdata());
    }

    return;
}

void BLK_CalBak::K1xfer(gentype &res, int &minmaxind, int typeis,
                        const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                        const SparseVector<gentype> &xa,
                        const vecInfo &xainfo,
                        int ia,
                        int xdim, int densetype, int resmode, int mlid) const
{
    if ( K1callback() )
    {
        (*K1callback())(res,minmaxind,typeis,xa,xainfo,ia,xdim,densetype,resmode,mlid,K1callbackdata());
    }

    else if ( K1callbackalt() )
    {
        int xadim = 0;
        const double *xax = getdbl(xa,xadim);

        res.force_double() = (*K1callbackalt())(typeis,xax,xadim,ia,xdim,densetype,resmode,mlid,K1callbackdata());
    }

    else
    {
        res.force_double() = (*Kcallbackalt())(typeis,(double) (xyprod+yxprod)/2.0,(double) diffis,xdim,densetype,resmode,mlid,Kcallbackdata());
    }

    return;
}

void BLK_CalBak::K2xfer(gentype &dxyprod, gentype &ddiffis, gentype &res, int &minmaxind, int typeis,
                        const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                        const vecInfo &xainfo, const vecInfo &xbinfo,
                        int ia, int ib,
                        int xdim, int densetype, int resmode, int mlid) const
{
    (void) dxyprod;
    (void) ddiffis;

    if ( K2callback() )
    {
        (*K2callback())(res,minmaxind,typeis,xa,xb,xainfo,xbinfo,ia,ib,xdim,densetype,resmode,mlid,K2callbackdata());
    }

    else if ( K2callbackalt() )
    {
        int xadim = 0;
        const double *xax = getdbl(xa,xadim);

        int xbdim = 0;
        const double *xbx = getdbl(xb,xbdim);

        res.force_double() = (*K2callbackalt())(typeis,xax,xadim,xbx,xbdim,ia,ib,xdim,densetype,resmode,mlid,K2callbackdata());
    }

    else
    {
        res.force_double() = (*Kcallbackalt())(typeis,(double) (xyprod+yxprod)/2.0,(double) diffis,xdim,densetype,resmode,mlid,Kcallbackdata());
    }

    return;
}

void BLK_CalBak::K3xfer(gentype &res, int &minmaxind, int typeis,
                        const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc,
                        const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo,
                        int ia, int ib, int ic,
                        int xdim, int densetype, int resmode, int mlid) const
{
    if ( K3callback() )
    {
        (*K3callback())(res,minmaxind,typeis,xa,xb,xc,xainfo,xbinfo,xcinfo,ia,ib,ic,xdim,densetype,resmode,mlid,K3callbackdata());
    }

    else if ( K3callbackalt() )
    {
        int xadim = 0;
        const double *xax = getdbl(xa,xadim);

        int xbdim = 0;
        const double *xbx = getdbl(xb,xbdim);

        int xcdim = 0;
        const double *xcx = getdbl(xc,xcdim);

        res.force_double() = (*K3callbackalt())(typeis,xax,xadim,xbx,xbdim,xcx,xcdim,ia,ib,ic,xdim,densetype,resmode,mlid,K3callbackdata());
    }

    else
    {
        res.force_double() = (*Kcallbackalt())(typeis,(double) (xyprod+yxprod)/2.0,(double) diffis,xdim,densetype,resmode,mlid,Kcallbackdata());
    }

    return;
}

void BLK_CalBak::K4xfer(gentype &res, int &minmaxind, int typeis,
                        const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                        const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
                        int ia, int ib, int ic, int id,
                        int xdim, int densetype, int resmode, int mlid) const
{
    if ( K4callback() )
    {
        (*K4callback())(res,minmaxind,typeis,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,ia,ib,ic,id,xdim,densetype,resmode,mlid,K4callbackdata());
    }

    else if ( K4callbackalt() )
    {
        int xadim = 0;
        const double *xax = getdbl(xa,xadim);

        int xbdim = 0;
        const double *xbx = getdbl(xb,xbdim);

        int xcdim = 0;
        const double *xcx = getdbl(xc,xcdim);

        int xddim = 0;
        const double *xdx = getdbl(xd,xddim);

        res.force_double() = (*K4callbackalt())(typeis,xax,xadim,xbx,xbdim,xcx,xcdim,xdx,xddim,ia,ib,ic,id,xdim,densetype,resmode,mlid,K4callbackdata());
    }

    else
    {
        res.force_double() = (*Kcallbackalt())(typeis,(double) (xyprod+yxprod)/2.0,(double) diffis,xdim,densetype,resmode,mlid,Kcallbackdata());
    }

    return;
}

void BLK_CalBak::Kmxfer(gentype &res, int &minmaxind, int typeis,
                        const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                        Vector<const SparseVector<gentype> *> &x,
                        Vector<const vecInfo *> &xinfo,
                        Vector<int> &i,
                        int xdim, int m, int densetype, int resmode, int mlid) const
{
/*
    if ( m == 0 )
    {
        K0xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xdim,densetype,resmode,mlid);
    }

    else if ( m == 1 )
    {
        K1xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,*x(0),*xinfo(0),i(0),xdim,densetype,resmode,mlid);
    }

    else if ( m == 2 )
    {
        gentype dummy;

        K2xfer(dummy,dummy,res,minmaxind,typeis,xyprod,yxprod,diffis,*x(0),*x(1),*xinfo(0),*xinfo(1),i(0),i(1),xdim,densetype,resmode,mlid);
    }

    else if ( m == 3 )
    {
        K3xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,*x(0),*x(1),*x(2),*xinfo(0),*xinfo(1),*xinfo(2),i(0),i(1),i(2),xdim,densetype,resmode,mlid);
    }

    else if ( m == 4 )
    {
        K4xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,*x(0),*x(1),*x(2),*x(3),*xinfo(0),*xinfo(1),*xinfo(2),*xinfo(3),i(0),i(1),i(2),i(3),xdim,densetype,resmode,mlid);
    }

    else
*/
    if ( Kmcallback() )
    {
        (*Kmcallback())(res,minmaxind,typeis,x,xinfo,i,xdim,m,densetype,resmode,mlid,Kmcallbackdata());
    }

    else if ( Kmcallbackalt() )
    {
        int ii;

        double **xx = new double *[m];
        int *xxdim = new int[m];

        for ( ii = 0 ; ii < m ; ++ii )
        {
            xx[ii] = const_cast<double *>(getdbl(*(x(ii)),xxdim[ii]));
        }

        res.force_double() = (*Kmcallbackalt())(typeis,const_cast<const double **>(xx),xxdim,&(i(0)),xdim,m,densetype,resmode,mlid,Kmcallbackdata());

        delete[] xx;
        delete[] xxdim;
    }

    else
    {
        res.force_double() = (*Kcallbackalt())(typeis,(double) (xyprod+yxprod)/2.0,(double) diffis,xdim,densetype,resmode,mlid,Kcallbackdata());
    }

    return;
}

void BLK_CalBak::K0xfer(double &res, int &minmaxind, int typeis,
                        double xyprod, double yxprod, double diffis,
                        int xdim, int densetype, int resmode, int mlid) const
{
    if ( !K0callback() && !K0callbackalt() )
    {
        res = (*Kcallbackalt())(typeis,xyprod,diffis,xdim,densetype,resmode,mlid,Kcallbackdata());
    }

    else
    {
        gentype tempres;

        gentype gxyprod(xyprod);
        gentype gyxprod(yxprod);
        gentype gdiffis(diffis);

        K0xfer(tempres,minmaxind,typeis,gxyprod,gyxprod,gdiffis,xdim,densetype,resmode,mlid);

        res = (double) tempres;
    }

    return;
}

void BLK_CalBak::K1xfer(double &res, int &minmaxind, int typeis,
                        double xyprod, double yxprod, double diffis,
                        const SparseVector<gentype> &xa,
                        const vecInfo &xainfo,
                        int ia,
                        int xdim, int densetype, int resmode, int mlid) const
{
    if ( !K1callback() && !K1callbackalt() )
    {
        res = (*Kcallbackalt())(typeis,xyprod,diffis,xdim,densetype,resmode,mlid,Kcallbackdata());
    }

    else
    {
        gentype tempres;

        gentype gxyprod(xyprod);
        gentype gyxprod(yxprod);
        gentype gdiffis(diffis);

        K1xfer(tempres,minmaxind,typeis,gxyprod,gyxprod,gdiffis,xa,xainfo,ia,xdim,densetype,resmode,mlid);

        res = (double) tempres;
    }

    return;
}

void BLK_CalBak::K2xfer(double &dxyprod, double &ddiffis, double &res, int &minmaxind, int typeis,
                        double xyprod, double yxprod, double diffis,
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                        const vecInfo &xainfo, const vecInfo &xbinfo,
                        int ia, int ib,
                        int xdim, int densetype, int resmode, int mlid) const
{
    if ( !K2callback() && !K2callbackalt() )
    {
        res = (*Kcallbackalt())(typeis,xyprod,diffis,xdim,densetype,resmode,mlid,Kcallbackdata());
    }

    else
    {
        gentype tempres;
        gentype tempdxyprod;
        gentype tempddiffis;

        gentype gxyprod(xyprod);
        gentype gyxprod(yxprod);
        gentype gdiffis(diffis);

        K2xfer(tempdxyprod,tempddiffis,tempres,minmaxind,typeis,gxyprod,gyxprod,gdiffis,xa,xb,xainfo,xbinfo,ia,ib,xdim,densetype,resmode,mlid);

        res     = (double) tempres;
        dxyprod = (double) tempdxyprod;
        ddiffis = (double) tempddiffis;
    }

    return;
}

void BLK_CalBak::K3xfer(double &res, int &minmaxind, int typeis,
                        double xyprod, double yxprod, double diffis,
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc,
                        const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo,
                        int ia, int ib, int ic,
                        int xdim, int densetype, int resmode, int mlid) const
{
    if ( !K3callback() && !K3callbackalt() )
    {
        res = (*Kcallbackalt())(typeis,xyprod,diffis,xdim,densetype,resmode,mlid,Kcallbackdata());
    }

    else
    {
        gentype tempres;

        gentype gxyprod(xyprod);
        gentype gyxprod(yxprod);
        gentype gdiffis(diffis);

        K3xfer(tempres,minmaxind,typeis,gxyprod,gyxprod,gdiffis,xa,xb,xc,xainfo,xbinfo,xcinfo,ia,ib,ic,xdim,densetype,resmode,mlid);

        res = (double) tempres;
    }

    return;
}

void BLK_CalBak::K4xfer(double &res, int &minmaxind, int typeis,
                        double xyprod, double yxprod, double diffis,
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                        const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
                        int ia, int ib, int ic, int id,
                        int xdim, int densetype, int resmode, int mlid) const
{
    if ( !K4callback() && !K4callbackalt() )
    {
        res = (*Kcallbackalt())(typeis,xyprod,diffis,xdim,densetype,resmode,mlid,Kcallbackdata());
    }

    else
    {
        gentype tempres;

        gentype gxyprod(xyprod);
        gentype gyxprod(yxprod);
        gentype gdiffis(diffis);

        K4xfer(tempres,minmaxind,typeis,gxyprod,gyxprod,gdiffis,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,ia,ib,ic,id,xdim,densetype,resmode,mlid);

        res = (double) tempres;
    }

    return;
}

void BLK_CalBak::Kmxfer(double &res, int &minmaxind, int typeis,
                        double xyprod, double yxprod, double diffis,
                        Vector<const SparseVector<gentype> *> &x,
                        Vector<const vecInfo *> &xinfo,
                        Vector<int> &i,
                        int xdim, int m, int densetype, int resmode, int mlid) const
{
/*
    if ( m == 0 )
    {
        K0xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xdim,densetype,resmode,mlid);
    }

    else if ( m == 1 )
    {
        K1xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,*x(0),*xinfo(0),i(0),xdim,densetype,resmode,mlid);
    }

    else if ( m == 2 )
    {
        double dummy;

        K2xfer(dummy,dummy,res,minmaxind,typeis,xyprod,yxprod,diffis,*x(0),*x(1),*xinfo(0),*xinfo(1),i(0),i(1),xdim,densetype,resmode,mlid);
    }

    else if ( m == 3 )
    {
        K3xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,*x(0),*x(1),*x(2),*xinfo(0),*xinfo(1),*xinfo(2),i(0),i(1),i(2),xdim,densetype,resmode,mlid);
    }

    else if ( m == 4 )
    {
        K4xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,*x(0),*x(1),*x(2),*x(3),*xinfo(0),*xinfo(1),*xinfo(2),*xinfo(3),i(0),i(1),i(2),i(3),xdim,densetype,resmode,mlid);
    }

    else
*/
    if ( !Kmcallback() && !Kmcallbackalt() )
    {
        res = (*Kcallbackalt())(typeis,xyprod,diffis,xdim,densetype,resmode,mlid,Kcallbackdata());
    }

    else
    {
        gentype tempres;

        gentype gxyprod(xyprod);
        gentype gyxprod(yxprod);
        gentype gdiffis(diffis);

        Kmxfer(tempres,minmaxind,typeis,gxyprod,gyxprod,gdiffis,x,xinfo,i,xdim,m,densetype,resmode,mlid);

        res = (double) tempres;
    }

    return;
}



