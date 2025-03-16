
//
// Kernel specialisation block
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
#include "blk_kernel.hpp"


#define MINUINTSIZE 16

int BLK_Kernel::isKVarianceNZ(void) const
{
    return ML_Base::isKVarianceNZ();
}

std::ostream &BLK_Kernel::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Kernel specialisation block\n";

    return BLK_Generic::printstream(output,dep+1);
}

std::istream &BLK_Kernel::inputstream(std::istream &input )
{
    return BLK_Generic::inputstream(input);
}

void BLK_Kernel::K0xfer(gentype &res, int &minmaxind, int typeis,
                        const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                        int xdim, int densetype, int resmode, int mlid) const
{
    (void) res;
    (void) minmaxind;
    (void) typeis;
    (void) xyprod;
    (void) yxprod;
    (void) diffis;
    (void) xdim;
    (void) densetype;
    (void) resmode;
    (void) mlid;

    setzero(res);

    return;
}

void BLK_Kernel::K0xfer(double &res, int &minmaxind, int typeis,
                        double xyprod, double yxprod, double diffis,
                        int xdim, int densetype, int resmode, int mlid) const
{
    (void) res;
    (void) minmaxind;
    (void) typeis;
    (void) xyprod;
    (void) yxprod;
    (void) diffis;
    (void) xdim;
    (void) densetype;
    (void) resmode;
    (void) mlid;

    setzero(res);

    return;
}

void BLK_Kernel::K1xfer(gentype &res, int &minmaxind, int typeis,
                        const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                        const SparseVector<gentype> &xa, 
                        const vecInfo &xainfo, 
                        int ia, 
                        int xdim, int densetype, int resmode, int mlid) const
{
    // NB: xyprod,yxprod and diffis are not used in the inheritance, so no need to calculate

    setzero(res);

    int Nval = ( lambdaKB().numRows() < N() ) ? lambdaKB().numRows() : N();
    int Nrep = lambdaKB().numCols();

    if ( Nval && Nrep )
    {
        gentype temp;
        gentype dummy;

        int q,i;

        gentype ddiffis;

        for ( q = 0 ; q < Nrep ; ++q )
        {
            for ( i = 0 ; i < Nval ; ++i )
            {
                ML_Base::K2xfer(dummy,ddiffis,temp,minmaxind,typeis,xyprod,yxprod,diffis,xa,x(i),xainfo,xinfo(i),ia,i,xdim,densetype,resmode,mlid);

                temp *= lambdaKB()(i,q);

                res += temp;
            }
        }
    }

    return;
}

void BLK_Kernel::K1xfer(double &res, int &minmaxind, int typeis,
                        double xyprod, double yxprod, double diffis,
                        const SparseVector<gentype> &xa, 
                        const vecInfo &xainfo, 
                        int ia, 
                        int xdim, int densetype, int resmode, int mlid) const
{
    // NB: xyprod,yxprod and diffis are not used in the inheritance, so no need to calculate

    setzero(res);

    int Nval = ( lambdaKB().numRows() < N() ) ? lambdaKB().numRows() : N();
    int Nrep = lambdaKB().numCols();

    if ( Nval && Nrep )
    {
        double temp;
        double dummy;

        int q,i;

        double ddiffis;

        for ( q = 0 ; q < Nrep ; ++q )
        {
            for ( i = 0 ; i < Nval ; ++i )
            {
                ML_Base::K2xfer(dummy,ddiffis,temp,minmaxind,typeis,xyprod,yxprod,diffis,xa,x(i),xainfo,xinfo(i),ia,i,xdim,densetype,resmode,mlid);

                temp *= lambdaKB()(i,q);

                res += temp;
            }
        }
    }

    return;
}

void BLK_Kernel::K2xfer(gentype &dxyprod, gentype &ddiffis, gentype &res, int &minmaxind, int typeis, 
                        const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                        const vecInfo &xainfo, const vecInfo &xbinfo,
                        int ia, int ib,
                        int xdim, int densetype, int resmode, int mlid) const
{
    // NB: xyprod,yxprod and diffis are not used in the inheritance, so no need to calculate

    (void) dxyprod;
    (void) ddiffis;

    setzero(res);

    int Nval = ( lambdaKB().numRows() < N() ) ? lambdaKB().numRows() : N();
    int Nrep = lambdaKB().numCols();

    if ( Nval && Nrep )
    {
        gentype temp;

        int q,i,j;

        for ( q = 0 ; q < Nrep ; ++q )
        {
            for ( i = 0 ; i < Nval ; ++i )
            {
                for ( j = 0 ; j <= i ; ++j )
                {
                    ML_Base::K4xfer(temp,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,x(i),x(j),xainfo,xbinfo,xinfo(i),xinfo(j),ia,ib,i,j,xdim,densetype,resmode,mlid);

                    temp *= lambdaKB()(i,q)*lambdaKB()(j,q);

                    res += temp;

                    if ( j != i )
                    {
                        res += temp;
                    }
                }
            }
        }
    }

    return;
}

void BLK_Kernel::K2xfer(double &dxyprod, double &ddiffis, double &res, int &minmaxind, int typeis,
                        double xyprod, double yxprod, double diffis,
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                        const vecInfo &xainfo, const vecInfo &xbinfo,
                        int ia, int ib,
                        int xdim, int densetype, int resmode, int mlid) const
{
    // NB: xyprod,yxprod and diffis are not used in the inheritance, so no need to calculate

    (void) dxyprod;
    (void) ddiffis;

    setzero(res);

    int Nval = ( lambdaKB().numRows() < N() ) ? lambdaKB().numRows() : N();
    int Nrep = lambdaKB().numCols();

    if ( Nval && Nrep )
    {
        double temp;

        int q,i,j;

        for ( q = 0 ; q < Nrep ; ++q )
        {
            for ( i = 0 ; i < Nval ; ++i )
            {
                for ( j = 0 ; j <= i ; ++j )
                {
                    ML_Base::K4xfer(temp,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,x(i),x(j),xainfo,xbinfo,xinfo(i),xinfo(j),ia,ib,i,j,xdim,densetype,resmode,mlid);

                    temp *= lambdaKB()(i,q)*lambdaKB()(j,q);

                    res += temp;

                    if ( j != i )
                    {
                        res += temp;
                    }
                }
            }
        }
    }

    return;
}

void BLK_Kernel::K3xfer(gentype &res, int &minmaxind, int typeis,
                        const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, 
                        const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, 
                        int ia, int ib, int ic, 
                        int xdim, int densetype, int resmode, int mlid) const
{
    int m = 3;

    Vector<const SparseVector<gentype> *> xx(m);
    Vector<const vecInfo *> xxinfo(m);
    Vector<int> ii(m);

    int z = 0;

    xx("&",z) = &xa;
    xx("&",1) = &xb;
    xx("&",2) = &xc;

    xxinfo("&",z) = &xainfo;
    xxinfo("&",1) = &xbinfo;
    xxinfo("&",2) = &xcinfo;

    ii("&",z) = ia;
    ii("&",1) = ib;
    ii("&",2) = ic;

    Kmxfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xx,xxinfo,ii,xdim,m,densetype,resmode,mlid);

    return;
}

void BLK_Kernel::K3xfer(double &res, int &minmaxind, int typeis,
                        double xyprod, double yxprod, double diffis,
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, 
                        const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, 
                        int ia, int ib, int ic, 
                        int xdim, int densetype, int resmode, int mlid) const
{
    int m = 3;

    Vector<const SparseVector<gentype> *> xx(m);
    Vector<const vecInfo *> xxinfo(m);
    Vector<int> ii(m);

    int z = 0;

    xx("&",z) = &xa;
    xx("&",1) = &xb;
    xx("&",2) = &xc;

    xxinfo("&",z) = &xainfo;
    xxinfo("&",1) = &xbinfo;
    xxinfo("&",2) = &xcinfo;

    ii("&",z) = ia;
    ii("&",1) = ib;
    ii("&",2) = ic;

    Kmxfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xx,xxinfo,ii,xdim,m,densetype,resmode,mlid);

    return;
}

void BLK_Kernel::K4xfer(gentype &res, int &minmaxind, int typeis,
                        const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                        const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
                        int ia, int ib, int ic, int id,
                        int xdim, int densetype, int resmode, int mlid) const
{
    int m = 4;

    Vector<const SparseVector<gentype> *> xx(m);
    Vector<const vecInfo *> xxinfo(m);
    Vector<int> ii(m);

    int z = 0;

    xx("&",z) = &xa;
    xx("&",1) = &xb;
    xx("&",2) = &xc;
    xx("&",3) = &xd;

    xxinfo("&",z) = &xainfo;
    xxinfo("&",1) = &xbinfo;
    xxinfo("&",2) = &xcinfo;
    xxinfo("&",3) = &xdinfo;

    ii("&",z) = ia;
    ii("&",1) = ib;
    ii("&",2) = ic;
    ii("&",3) = id;

    Kmxfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xx,xxinfo,ii,xdim,m,densetype,resmode,mlid);

    return;
}

void BLK_Kernel::K4xfer(double &res, int &minmaxind, int typeis,
                        double xyprod, double yxprod, double diffis,
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                        const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
                        int ia, int ib, int ic, int id,
                        int xdim, int densetype, int resmode, int mlid) const
{
    int m = 4;

    Vector<const SparseVector<gentype> *> xx(m);
    Vector<const vecInfo *> xxinfo(m);
    Vector<int> ii(m);

    int z = 0;

    xx("&",z) = &xa;
    xx("&",1) = &xb;
    xx("&",2) = &xc;
    xx("&",3) = &xd;

    xxinfo("&",z) = &xainfo;
    xxinfo("&",1) = &xbinfo;
    xxinfo("&",2) = &xcinfo;
    xxinfo("&",3) = &xdinfo;

    ii("&",z) = ia;
    ii("&",1) = ib;
    ii("&",2) = ic;
    ii("&",3) = id;

    Kmxfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xx,xxinfo,ii,xdim,m,densetype,resmode,mlid);

    return;
}

void BLK_Kernel::Kmxfer(gentype &res, int &minmaxind, int typeis,
                        const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                        Vector<const SparseVector<gentype> *> &xx,
                        Vector<const vecInfo *> &xxinfo,
                        Vector<int> &ii,
                        int xdim, int m, int densetype, int resmode, int mlid) const
{
    // NB: xyprod,yxprod and diffis are not used in the inheritance, so no need to calculate

    setzero(res);

    int Nval = ( lambdaKB().numRows() < N() ) ? lambdaKB().numRows() : N();
    int Nrep = lambdaKB().numCols();

    if ( Nval && Nrep )
    {
        gentype temp;

        int q,j;
        Vector<int> i(m);
        int isdone = 0;

        Vector<const SparseVector<gentype> *> xxx(2*m);
        Vector<const vecInfo *> xxxinfo(2*m);
        Vector<int> iii(2*m);

        retVector<const SparseVector<gentype> *> tmpa;
        retVector<const vecInfo *> tmpb;
        retVector<int> tmpc;

        xxx("&",0,1,m-1,tmpa) = xx;
        xxxinfo("&",0,1,m-1,tmpb) = xxinfo;
        iii("&",0,1,m-1,tmpc) = ii;

        for ( q = 0 ; q < Nrep ; ++q )
        {
            for ( i = 0 ; !isdone ; isdone = getnext(i,Nval-1) )
            {
                for ( j = 0 ; j < m ; ++j )
                {
                    xxx("&",j+m) = &(x(i(j)));
                    xxxinfo("&",j+m) = &(xinfo(i(j)));
                    iii("&",j+m) = i(j);
                }

                ML_Base::Kmxfer(temp,minmaxind,typeis,xyprod,yxprod,diffis,xxx,xxxinfo,iii,xdim,2*m,densetype,resmode,mlid);

                for ( j = 0 ; j < m ; ++j )
                {
                    temp *= lambdaKB()(i(j),q);
                }

                res += temp;
            }
        }
    }

    return;
}

void BLK_Kernel::Kmxfer(double &res, int &minmaxind, int typeis,
                        double xyprod, double yxprod, double diffis,
                        Vector<const SparseVector<gentype> *> &xx,
                        Vector<const vecInfo *> &xxinfo,
                        Vector<int> &ii,
                        int xdim, int m, int densetype, int resmode, int mlid) const
{
    // NB: xyprod,yxprod and diffis are not used in the inheritance, so no need to calculate

    setzero(res);

    int Nval = ( lambdaKB().numRows() < N() ) ? lambdaKB().numRows() : N();
    int Nrep = lambdaKB().numCols();

    if ( Nval && Nrep )
    {
        double temp;

        int q,j;
        Vector<int> i(m);
        int isdone = 0;

        Vector<const SparseVector<gentype> *> xxx(2*m);
        Vector<const vecInfo *> xxxinfo(2*m);
        Vector<int> iii(2*m);

        retVector<const SparseVector<gentype> *> tmpa;
        retVector<const vecInfo *> tmpb;
        retVector<int> tmpc;

        xxx("&",0,1,m-1,tmpa) = xx;
        xxxinfo("&",0,1,m-1,tmpb) = xxinfo;
        iii("&",0,1,m-1,tmpc) = ii;

        for ( q = 0 ; q < Nrep ; ++q )
        {
            for ( i = 0 ; !isdone ; isdone = getnext(i,Nval-1) )
            {
                for ( j = 0 ; j < m ; ++j )
                {
                    xxx("&",j+m) = &(x(i(j)));
                    xxxinfo("&",j+m) = &(xinfo(i(j)));
                    iii("&",j+m) = i(j);
                }

                ML_Base::Kmxfer(temp,minmaxind,typeis,xyprod,yxprod,diffis,xxx,xxxinfo,iii,xdim,2*m,densetype,resmode,mlid);

                for ( j = 0 ; j < m ; ++j )
                {
                    temp *= lambdaKB()(i(j),q);
                }

                res += temp;
            }
        }
    }

    return;
}

int BLK_Kernel::train(int &res, svmvolatile int &killswitch)
{
    int Nblk = altMLidsKB().size(); // number of MLs being optimised for

    if ( Nblk )
    {
        const double lr(lrKB()); // learning rate
        const Vector<int> &altMLids(altMLidsKB()); // IDs of MLs being optimised
        const Vector<double> &MLweight(MLweightKB()); // weights for different MLs
        const double minstep(minstepKB());
        const int maxiter(maxiterKB());

        Vector<int> alphaState;
        Matrix<double> B;
        Matrix<double> Bnext;
        const ML_Base *firstML = nullptr;

        gentype tmp;
        int i,j,k,l,q;
        int notdone = 1;
        int firststep = 1;
        int itcnt = 0;

        NiceAssert( MLweight.size() == Nblk );

        Matrix<double> lambdaGrad(N());

        while ( !killswitch && notdone && ( itcnt < maxiter ) )
        {
            ++itcnt;

            if ( !firststep )
            {
                for ( q = 0 ; q < Nblk ; ++q )
                {
                    kernPrecursor *MLblk;
                    getaltML(MLblk,altMLids(q));
                    ML_Base &MLblock = dynamic_cast<ML_Base &>(*MLblk);

                    if ( !q )
                    {
                        firstML = &MLblock;

                        B = 0.0;
                        MLblock.dedKTrainingVector(B) *= MLweight(q);
                        alphaState = MLblock.alphaState();
                    }

                    else
                    {
                        Bnext = 0.0;
                        MLblock.dedKTrainingVector(Bnext) *= MLweight(q);
                        alphaState += MLblock.alphaState();

                        B += Bnext;
                    }
                }

                B *= (1.0/((double) (Nblk*B.numRows())));

                int Nval = ( lambdaKB().numRows() < N() ) ? lambdaKB().numRows() : N();
                int Nrep = lambdaKB().numCols();
                int Ninner = B.numRows();

                lambdaGrad = 0.0;

                for ( q = 0 ; q < Nrep ; ++q )
                {
                    for ( k = 0 ; k < Nval ; ++k )
                    {
                        for ( l = 0 ; l < Nval ; ++l )
                        {
                            for ( i = 0 ; i < Ninner ; ++i )
                            {
                                if ( alphaState(i) )
                                {
                                    for ( j = 0 ; j < Ninner ; ++j )
                                    {
                                        if ( alphaState(j) )
                                        {
                                            lambdaGrad("&",k,q) += lambdaKB()(l,q)*B(i,j)*((double) K4(tmp,(*firstML).x(i),(*firstML).x(j),x(k),x(l),&((*firstML).xinfo(i)),&((*firstML).xinfo(j)),&(xinfo(k)),&(xinfo(l))));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                double stepsize = 0;

                if ( ( itcnt >= 3 ) && ( sqsum(stepsize,lambdaGrad) < minstep ) )
                {
                    notdone = 0;
                }

                // take the step

                lambdaGrad *= -lr;
                lambdaGrad += lambdaKB();

                setlambdaKB(lambdaGrad);
            }

            for ( q = 0 ; q < Nblk ; ++q )
            {
                kernPrecursor *MLblk;
                getaltML(MLblk,altMLids(q));
                ML_Base &MLblock = dynamic_cast<ML_Base &>(*MLblk);

                MLblock.resetKernel();
                MLblock.train(res,killswitch);
            }

            firststep = 0;
        }
    }

    return 0;
}
