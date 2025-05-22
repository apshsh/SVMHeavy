//TO DO: for some reason d = +1 (indicated by > 0 in training set) is not getting passed through to here
//to trigger SVM_Scalar training fallback.  Find out why and fix it!

//
// LS-SVM scalar class
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
#include "lsv_scalar.hpp"

#define FASTXDIMMAX 32

std::ostream &LSV_Scalar::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Base training alphaR: " << dalphaR << "\n";
    repPrint(output,'>',dep) << "Base training biasR:  " << dbiasR  << "\n";

    LSV_Generic::printstream(output,dep+1);

    return output;
}

std::istream &LSV_Scalar::inputstream(std::istream &input )
{
    wait_dummy dummy;

    input >> dummy; input >> dalphaR;
    input >> dummy; input >> dbiasR;

    LSV_Generic::inputstream(input);

    return input;
}

LSV_Scalar::LSV_Scalar() : LSV_Generic()
{
    dbias.force_double() = 0.0;

    return;
}

LSV_Scalar::LSV_Scalar(const LSV_Scalar &src) : LSV_Generic()
{
    assign(src,0);
    dbias.force_double() = 0.0;

    return;
}

LSV_Scalar::LSV_Scalar(const LSV_Scalar &src, const ML_Base *srcx) : LSV_Generic()
{
    setaltx(srcx);

    assign(src,-1);
    dbias.force_double() = 0.0;

    return;
}

int LSV_Scalar::prealloc(int expectedN)
{
    LSV_Generic::prealloc(expectedN);
    dalphaR.prealloc(expectedN);
    alltraintargR.prealloc(expectedN);

    return 0;
}

int LSV_Scalar::getInternalClass(const gentype &y) const
{
    return y.isCastableToRealWithoutLoss() ? ( ( ( (double) y ) < 0 ) ? 0 : 1 ) : 0;
}

int LSV_Scalar::addTrainingVector(int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= N() );

    dalphaR.add(i);
    dalphaR("&",i) = 0.0;

    alltraintargR.add(i);
    alltraintargR("&",i) = (double) y;

    return LSV_Generic::addTrainingVector(i,y,x,Cweigh,epsweigh);
}

int LSV_Scalar::qaddTrainingVector(int i, const gentype &y, SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= N() );

    dalphaR.add(i);
    dalphaR("&",i) = 0.0;

    alltraintargR.add(i);
    alltraintargR("&",i) = (double) y;

    return LSV_Generic::qaddTrainingVector(i,y,x,Cweigh,epsweigh);
}

int LSV_Scalar::removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );

    int res = LSV_Generic::removeTrainingVector(i,y,x);

    dalphaR.remove(i);
    alltraintargR.remove(i);

    return res;
}

int LSV_Scalar::removeTrainingVector(int i, int num)
{
    int res = 0;
    gentype y;
    SparseVector<gentype> x;

    while ( num )
    {
        --num; 
        res |= removeTrainingVector(i+num,y,x);
    }

    return res;
}

int LSV_Scalar::sety(int i, const gentype &y)
{
    int res = LSV_Generic::sety(i,y);

    alltraintargR("&",i) = (double) y;

    return res;
}

int LSV_Scalar::sety(const Vector<int> &i, const Vector<gentype> &y)
{
    int j;
    int res = LSV_Generic::sety(i,y);

    if ( i.size() )
    {
        for ( j = 0 ; j < i.size() ; ++j )
        {
            alltraintargR("&",i(j)) = (double) y(j);
        }
    }

    return res;
}

int LSV_Scalar::sety(const Vector<gentype> &y)
{
    int j;
    int res = LSV_Generic::sety(y);

    if ( N() )
    {
        for ( j = 0 ; j < N() ; ++j )
        {
            alltraintargR("&",j) = (double) y(j);
        }
    }

    return res;
}

int LSV_Scalar::sety(int i, double y)
{
    gentype yy(y);

    return sety(i,yy);
}

int LSV_Scalar::sety(const Vector<int> &i, const Vector<double> &y)
{
    Vector<gentype> yy(i.size());

    int j;

    if ( i.size() )
    {
        for ( j = 0 ; j < i.size() ; ++j )
        {
            yy("&",j) = y(j);
        }
    }

    return sety(i,yy);
}

int LSV_Scalar::sety(const Vector<double> &y)
{
    Vector<gentype> yy(N());

    int j;

    if ( N() )
    {
        for ( j = 0 ; j < N() ; ++j )
        {
            yy("&",j) = y(j);
        }
    }

    return sety(yy);
}

int LSV_Scalar::setd(int i, int nd)
{
    int res = LSV_Generic::setd(i,nd);

    if ( !nd )
    {
        dalphaR("&",i) = 0.0;
    }

    return res;
}

int LSV_Scalar::setd(const Vector<int> &i, const Vector<int> &nd)
{
    int res = LSV_Generic::setd(i,nd);

    if ( i.size() )
    {
        int j;

        for ( j = 0 ; j < i.size() ; ++j )
        {
            if ( !nd(j) )
            {
                dalphaR("&",i(j)) = 0.0;
            }
        }
    }

    return res;
}

int LSV_Scalar::setd(const Vector<int> &nd)
{
    int res = LSV_Generic::setd(nd);

    if ( N() )
    {
        int j;

        for ( j = 0 ; j < N() ; ++j )
        {
            if ( !nd(j) )
            {
                dalphaR("&",j) = 0.0;
            }
        }
    }

    return res;
}

int LSV_Scalar::scale(double a)
{
    LSV_Generic::scale(a);
    dalphaR.scale(a);
    dbiasR *= a;

    return 1;
}

int LSV_Scalar::reset(void)
{
    LSV_Generic::reset();
    dalphaR = 0.0;
    dbiasR = 0.0;

    return 1;
}

int LSV_Scalar::setgamma(const Vector<gentype> &newW)
{
    int res = LSV_Generic::setgamma(newW);

    if ( N() )
    {
        int i;

        for ( i = 0 ; i < N() ; ++i )
        {
            dalphaR("&",i) = (double) newW(i);
        }
    }

    return res;
}

int LSV_Scalar::setdelta(const gentype &newB)
{
    int res = LSV_Generic::setdelta(newB);

    dbiasR = (double) newB;

    return res;
}

int LSV_Scalar::train(int &res, svmvolatile int &killSwitch)
{
    killfasts();

    incgvernum();

//errstream() << "phantomx lsvtrain 0: " << NNC(-1) << "," << NNC(+1) << "\n";
    if ( !NNC(-1) && !NNC(+1) && ( eps() == 0.0 ) )
    {
//errstream() << "phantomx lsvtrain 1\n";
        Vector<double> dybeta(1);
        Vector<double> dbetaR(1);
//errstream() << "phantomx lsvtrain 2\n";

        LSV_Generic::train(res,killSwitch);

        dybeta = 0.0;
        dbetaR = 0.0;

        dalphaR = 0.0;
        dbiasR  = 0.0;

//errstream() << "phantomx lsvtrain 3: " << alltraintargR << "\n";
//errstream() << "phantomx lsvtrain 4: " << dybeta << "\n";
//errstream() << "phantomx lsvtrain 4b: " << Gp() << "\n";
//errstream() << "phantomx lsvtrain 4c: " << *this << "\n";
        int badindex = -1;

//tryagain:
        if ( ( badindex = fact_minverse(dalphaR,dbetaR,alltraintargR,dybeta) ) >= 0 )
        {
//errstream() << "_" << badindex << "," << SVM_Scalar::Cweight()(badindex) << "_";
//SVM_Scalar::setCweight(badindex,Cweight()(badindex)/10);
//goto tryagain;
            goto fallback_method; // training has failed as Hessian is indefinite!
        }
//errstream() << "phantomx lsvtrain 5: " << dalphaR << "\n";
//errstream() << "phantomx lsvtrain 6: " << dbetaR << "\n";

        dbiasR = dbetaR(0);

        SVM_Scalar::setAlphaR(dalphaR);
        SVM_Scalar::setBiasR(dbiasR);
    }

    else
    {
fallback_method:
errstream() << "@";
        SVM_Scalar::isStateOpt = 0;
        SVM_Scalar::train(res,killSwitch);

        dalphaR = SVM_Scalar::alphaR();
        dbiasR  = SVM_Scalar::biasR();
//errstream() << "phantomxyz lsv alpha = " << dalphaR << "\n";
//errstream() << "phantomxyz lsv bias  = " << dbiasR << "\n";
    }

    if ( N() )
    {
//errstream() << "phantomx lsvtrain 1\n";
        int i;

        for ( i = 0 ; i < N() ; ++i )
        {
            dalpha("&",i).dir_double() = dalphaR(i);
        }
    }

    dbias.dir_double() = dbiasR;

    return 0;
}

int LSV_Scalar::ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg, gentype ***pxyprodi) const
{
    int xtangi = xtang(i);
    int dtv = xtangi & (7+32+64);
    int isloc = ( ( i >= 0 ) && d()(i) ) ? 1 : 0;

    setzero(resg);

    if ( isloc && ( eps() == 0.0 ) && !( retaltg & 2 ) )
    {
        double &res = resg.force_double();
        setzero(res);

        res = ( -((double) dalpha(i)) * ((diagoffset())(i)) ) + ((double) alltraintarg(i));

        if ( mpri() )
        {
            if ( i >= 0 )
            {
                res += ypR()(i);
            }

            else
            {
                gentype resprior;

                calcprior(resprior,x(i));

                res += (double) resprior;
            }
        }
    }

    else if ( ( dtv & 4 ) )
    {
        // Undirected gradient, result is *probably* a vector (unless xspaceDim() == 1, as noted below)

        NiceAssert( !( retaltg & 2 ) )

        int j;
        bool isfirst = true;
        gentype Kxj;

        for ( j = 0 ; j < N() ; ++j )
        {
            if ( d()(j) )
            {
                if ( i >= 0 )
                {
                    // This probably won't happen, but undirected gradient constraints are
                    // ok in the training set if xspaceDim() == 1, which is the case here.

                    Kxj = Gp()(i,j);

                    if ( i == j )
                    {
                        Kxj -= diagoffset()(i);
                    }
                }

                else
                {
                    K2(Kxj,i,j,pxyprodi ? (const gentype **) pxyprodi[j] : nullptr);
                }

                if ( isfirst )
                {
                    resg = Kxj*dalphaR(j);
                    isfirst = false;
                }

                else
                {
                    resg += Kxj*dalphaR(j);
                }
            }
        }

        if ( mpri() )
        {
            if ( i >= 0 )
            {
                resg += yp()(i);
            }

            else
            {
                gentype resprior;

                calcprior(resprior,x(i));

                resg += resprior;
            }
        }
    }

    else if ( !( retaltg & 2 ) )
    {
        double &res = resg.force_double();

        // NB: the derivative might not be double!

        bool biasZero = ( ( dtv & (3+64) ) || isZerodelta() ); // No bias if this is a gradient, dense or otherwise

        NiceAssert( ( dbiasR == 0.0 ) || !( dtv & 32 ) ); // can't combine non-zero bias with dense integration (result is infinite in this case)

        res  = biasZero ? 0.0 : dbiasR;
        res += ghTrainingVectorUnbiasedUnsquaredNotundirectedgradIneg(i,xtangi,pxyprodi);
    }

//    else if ( !dtv )
//    {
//        double &res = resg.force_double();
//
//        res  = isZerodelta() ? 0.0 : dbiasR;
//        res += ghTrainingVectorUnbiasedUnsquaredNotundirectedgradIneg(i,xtangi,pxyprodi);
//        res *= res;
//    }

    else
    {
        double &res = resg.force_double();

        // NB: the derivative might not be double!

        bool biasZero = ( ( dtv & (3+64) ) || isZerodelta() ); // No bias if this is a gradient, dense or otherwise

        NiceAssert( ( dbiasR == 0.0 ) || !( dtv & 32 ) ); // can't combine non-zero bias with dense integration (result is infinite in this case)

        res = 0.0; // bias comes later

        // See commentary in svm_scalar.cc on ::gTrainingVector function (in essence recall we want any operators to act *after* g(x) is squared, not before)

        if ( !biasZero )
        {
            double resl = ghTrainingVectorUnbiasedUnsquaredNotundirectedgradIneg(i,xtangi,pxyprodi);

            res = (dbiasR*dbiasR)+(2*dbiasR*resl);
        }

        res += ghTrainingVectorUnbiasedSquaredNotundirectedgradIneg(i,xtangi);
    }

    resh = resg;

    gentype tentype(sgn(resh));

    return tentype.isCastableToIntegerWithoutLoss() ? (int) tentype : 0;
}

double LSV_Scalar::ghTrainingVectorUnbiasedUnsquaredNotundirectedgradIneg(int i, int xtangi, gentype ***pxyprodi) const
{
    double res = 0;

    int j,k;
    double Kxj,diffnormis;

    int xdim = xspaceDim();
    bool evalherebase = ( i < 0 ) && !xtangi && ( fastdim_base || ( ( xdim <= FASTXDIMMAX ) && isXConsistent() && ( dattypeKey() == 4 ) && getKernel().unadornedRBFKernel() ) );

    if ( evalherebase )
    {
        const SparseVector<gentype> &xi = x(i);
        int NN = N();
        double r0 = getKernel().cRealConstants()(0);

        if ( !fastdim_base )
        {
            int k;

            MEMNEWARRAY(fastweights_base,double,NN);
            MEMNEWARRAY(fastxsums_base,double *,NN);

            for ( j = 0 ; j < NN ; ++j )
            {
                if ( d()(j) )
                {
                    const SparseVector<gentype> &xa = x(j);

                    MEMNEWARRAY(fastxsums_base[fastdim_base],double,xdim);

                    for ( k = 0 ; k < xdim ; ++k )
                    {
                        fastxsums_base[fastdim_base][k] = ((double) xa.direcref(k))/(r0*NUMBASE_SQRT2);
                    }

                    fastweights_base[fastdim_base] = dalphaR(j);

                    fastdim_base++;
                }
            }
        }

        double xxi[FASTXDIMMAX];

        for ( k = 0 ; k < xdim ; k++ )
        {
            xxi[k] = (xi.direcref(k))/(r0*NUMBASE_SQRT2);
        }

        for ( j = 0 ; j < fastdim_base ; j++ )
        {
            diffnormis = 0;

            for ( int k = 0 ; k < xdim ; ++k )
            {
                diffnormis += (xxi[k]-fastxsums_base[j][k])*(xxi[k]-fastxsums_base[j][k]);
            }

            res += exp(-diffnormis)*fastweights_base[j];
        }
    }

    else
    {
        for ( j = 0 ; j < N() ; ++j )
        {
            if ( d()(j) )
            {
                if ( i >= 0 )
                {
                    // It is *vital* that we go through the kernel cache here!  Otherwise
                    // in for example grid-search we will just end up calculating the same
                    // kernel evaluations over and over again!

                    Kxj = (double) (Gp()(i,j));

                    if ( i == j )
                    {
                        Kxj -= diagoffset()(i);
                    }
                }

                else
                {
                    Kxj = K2(i,j,pxyprodi ? (const gentype **) pxyprodi[j] : nullptr);
                }

                res += Kxj*dalphaR(j);
            }
        }
    }

        if ( mpri() )
        {
            if ( i >= 0 )
            {
                res += ypR()(i);
            }

            else
            {
                gentype resprior;

                calcprior(resprior,x(i));

                res += (double) resprior;
            }
        }

    return res;
}

double LSV_Scalar::ghTrainingVectorUnbiasedSquaredNotundirectedgradIneg(int i, int xtangi) const
{
    double res = 0;

    int j,ja,jb,k,q;
    double Kxj,Kxjq;

    int xdim = xspaceDim();

    // We want to accelerate squared integration for TS stuff
    //
    // evalhereA is a dense integral of a RBF kernel product
    // evalhereB is a directed gradient version of evalhereA

    bool evalhereA = ( i < 0 ) && ( xtangi == 32 ) && ( fastdim || ( ( xdim <= FASTXDIMMAX ) && isXConsistent() && ( dattypeKey() == 4 ) && getKernel().unadornedRBFKernel() ) );
    bool evalhereB = ( i < 0 ) && ( xtangi == 34 ) && ( fastdim || ( ( xdim <= FASTXDIMMAX ) && isXConsistent() && ( dattypeKey() == 4 ) && getKernel().unadornedRBFKernel() ) );

    if ( evalhereA || evalhereB )
    {
        // K(x,xa) = exp( - <x,x>/(2.r0.r0) - <xa,xa>/(2.r0.r0) + 2.<x,xa>/(2.r0.r0))
        // K(x,xb) = exp( - <x,x>/(2.r0.r0) - <xb,xb>/(2.r0.r0) + 2.<x,xb>/(2.r0.r0))
        // K(x,xa).K(x,xb) = exp( - 2.<x,x>/(2.r0.r0) - <xa,xa>/(2.r0.r0) - <xb,xb>/(2.r0.r0) + 2.<x,xa>/(2.r0.r0) + 2.<x,xb>/(2.r0.r0))
        //                 = exp( - 2.<x,x>/(2.r0.r0) + 4.<x,(xa+xb)/2>/(2.r0.r0) - 2.<(xa+xb)/2,(xa+xb)/2>/(2.r0.r0) - (1/2).<(xa-xb),(xa-xb)>/(2.r0.r0))
        //                 = exp( - 2.<x,x>/(2.r0.r0) + 4.<x,(xa+xb)/2>/(2.r0.r0) - 2.<(xa+xb)/2,(xa+xb)/2>/(2.r0.r0) ) . sqrt(exp( -<(xa-xb),(xa-xb)>/(2.r0.r0) ))
        //                 = exp( - ||x-(xa+xb)/2||_2^2/(r0.r0) ) . sqrt(exp( -||xa-xb||_2^2/(2.r0.r0) ))
        //               ( = exp( - ||x-(xa+xb)/2||_2^2/(r0.r0) ) . exp( -||(xa-xb)/2||_2^2/(2.r0.r0) )  )
        //
        // Then integrate the first term to get:
        //                 = ( prod_i int_{-inf}^{x_i} exp( -((z''/r0)-((xa_i+xb_i)/(2.r0)))^2 ) dz'' ).sqrt( exp( -||xa-xb||_2^2/(2.r0.r0) )
        //                   ( z' = z''/r0, so z'' = r0.z' and dz'' = r0.dz' )
        //                 = ( prod_i r0 int_{-inf}^{x_i/r0} exp( -(z'-((xa_i+xb_i)/(2.r0)))^2 ) dz' ).sqrt( exp( -||xa-xb||_2^2/(2.r0.r0) )
        //                   ( z = z'-((xa_i+xb_i)/(2.r0)), so z' = z+((xa_i+xb_i)/(2.r0)) and dz' = dz )
        //                 = ( prod_i r0 int_{-inf}^{(x_i/r0)-((xa_i+xb_i)/(2.r0))} exp( -z^2 ) dz ).sqrt( exp( -||xa-xb||_2^2/(2.r0.r0) )
        //                 = ( prod_i sqrt(pi)/2 r0 ( 1 + erf( (x_i/r0) - ((xa_i+xb_i)/(2.r0)) ) ) ).sqrt( exp( -||xa-xb||_2^2/(2.r0.r0) )
        //                 = ( prod_i ( 1 + erf( (x_i/r0) - ((xa_i+xb_i)/(2.r0)) ) ) ).( prod_i sqrt(pi)/2 r0 ).sqrt( exp( -||xa-xb||_2^2/(2.r0.r0) )
        //                 = ( prod_i ( 1 + erf( (x_i/r0) - ((xa_i+xb_i)/(2.r0)) ) ) ).W_ab
        //
        // W_ab = ( prod_i sqrt(pi)/2 r0 ).sqrt( exp( -||xa-xb||_2^2/(2.r0.r0) )
        //
        // In the directional derivative case:
        //                 = \sum_j r_j d/dx_j ( prod_i ( 1 + erf( (x_i/r0) - ((xa_i+xb_i)/(2.r0)) ) ) ).W_ab
        //                 = \sum_j r_j ( d/dx_j erf( (x_j/r0) - ((xa_j+xb_j)/(2.r0)) ) ) ( prod_{i \ne j} ( 1 + erf( (x_i/r0) - ((xa_i+xb_i)/(2.r0)) ) ) ).W_ab
        //                 = \sum_j 2/sqrt(pi) r_j/r0 ( exp( -((x_j/r0)-((xa_j+xb_j)/(2.r0)))^2 ) ) ( prod_{i \ne j} ( 1 + erf( (x_i/r0)-((xa_i+xb_i)/(2.r0)) ) ) ).W_ab

        const SparseVector<gentype> &xi = x(i);

        if ( !fastdim )
        {
            // W_ab = ( prod_i sqrt(pi)/2 r0 ).sqrt( exp( -||xa-xb||_2^2/(2.r0.r0) )

            int NN = N();
            double r0 = getKernel().cRealConstants()(0) / sqrt(2);

            MEMNEWARRAY(fastweights,double,NN*NN);
            MEMNEWARRAY(fastxsums,double *,NN*NN);

            for ( ja = 0 ; ja < NN ; ++ja )
            {
                if ( d()(ja) )
                {
                    const SparseVector<gentype> &xa = x(ja);

                    for ( jb = 0 ; jb < NN ; ++jb )
                    {
                        if ( d()(jb) )
                        {
                            const SparseVector<gentype> &xb = x(jb);

                            MEMNEWARRAY(fastxsums[fastdim],double,xdim);

                            Kxj = (double) (Gp()(ja,jb));

                            if ( ja == jb )
                            {
                                Kxj -= diagoffset()(ja);
                            }

                            Kxj = sqrt(Kxj);

                            // See Kbase case 2003

                            for ( k = 0 ; k < xdim ; ++k )
                            {
                                fastxsums[fastdim][k] = (((double) xa.direcref(k))+((double) xb.direcref(k)))/(2*r0);

                                Kxj *= NUMBASE_SQRTPI*r0/2;
                            }

                            fastweights[fastdim] = Kxj*dalphaR(ja)*dalphaR(jb);

                            fastdim++;
                        }
                    }
                }
            }
        }

        if ( evalhereA )
        {
            double r0 = getKernel().cRealConstants()(0) / sqrt(2);

            double xxi[FASTXDIMMAX];
            double jjres;

            for ( j = 0 ; j < xdim ; j++ )
            {
                xxi[j] = (xi.direcref(j))/r0;
            }

            for ( j = 0 ; j < fastdim ; j++ )
            {
                // W_ab.( prod_k ( 1 + erf( (x_k/r0) - ((xa_k+xb_k)/(2.r0)) ) ) )

                jjres = fastweights[j];

                for ( int k = 0 ; k < xdim ; ++k )
                {
                    jjres *= (1+erf(xxi[k]-fastxsums[j][k]));
                }

                res += jjres;
            }
        }

        else if ( evalhereB )
        {
            double r0 = getKernel().cRealConstants()(0) / sqrt(2);

            double xxi[FASTXDIMMAX];
            double xxiff[FASTXDIMMAX];
            double v[FASTXDIMMAX];

            for ( k = 0 ; k < xdim ; k++ )
            {
                xxi[k]   = (xi.direcref(k))/r0;
                xxiff[k] = (xi.f2().direcref(k))/r0;
            }

            for ( j = 0 ; j < fastdim ; j++ )
            {
                // W_ab.\sum_q 2/sqrt(pi) r_q/r0 ( exp( -((x_q/r0)-((xa_q+xb_q)/(2.r0)))^2 ) ) ( prod_{k \ne j} v_k )
                // v_k = 1 + erf( (x_k/r0)-((xa_k+xb_k)/(2.r0)) )

                for ( k = 0 ; k < xdim ; ++k )
                {
                    v[k] = (1+erf(xxi[k]-fastxsums[j][k]));
                }

                double jjres = 0;

                for ( q = 0 ; q < xdim ; ++q )
                {
                    Kxjq = xxiff[q];

                    for ( k = 0 ; k < xdim ; ++k )
                    {
                        Kxjq *= ( ( k == q ) ? exp(-(xxi[k]-fastxsums[j][k])*(xxi[k]-fastxsums[j][k])) : v[k] );
                    }

                    jjres += Kxjq;
                }

                res += NUMBASE_2ONSQRTPI*jjres*fastweights[j];
            }
        }
    }

    else
    {
        for ( ja = 0 ; ja < N() ; ++ja )
        {
            if ( d()(ja) )
            {
                for ( jb = 0 ; jb < N() ; ++jb )
                {
                    if ( d()(jb) )
                    {
                        Kxj = K2x2(i,ja,jb);

                        res += Kxj*dalphaR(ja)*dalphaR(jb);
                    }
                }
            }
        }
    }

        if ( mpri() )
        {
            if ( i >= 0 )
            {
                res += ypR()(i);
            }

            else
            {
                gentype resprior;

                calcprior(resprior,x(i));

                res += (double) resprior;
            }
        }

    return res;
}



double LSV_Scalar::eTrainingVector(int i) const
{
    gentype resg(0.0);

    ggTrainingVector(resg,i);

    double res = (double) resg;

        if ( mpri() )
        {
            if ( i >= 0 )
            {
                res -= ypR()(i);
            }

            else
            {
                gentype resprior;

                calcprior(resprior,x(i));

                res -= (double) resprior;
            }
        }

    if ( ( i < 0 ) || isenabled(i) )
    {
        res = (res-alltraintargR(i))*(res-alltraintargR(i))/2;
    }

    else
    {
        res = 0.0;
    }

    return res;
}


double &LSV_Scalar::dedgTrainingVector(double &res, int i) const
{
    gentype resg(0.0);

    ggTrainingVector(resg,i);

    res = (double) resg;

    if ( ( i < 0 ) || isenabled(i) )
    {
        res -= alltraintargR(i);
    }

    else if ( Q.alphaRestrict(i) == 1 )
    {
        res = 0;
    }

    return res;
}

double &LSV_Scalar::d2edg2TrainingVector(double &res, int i) const
{
    res = 0;

    if ( ( i < 0 ) || isenabled(i) )
    {
        res = 1;
    }

    return res;
}

Matrix<double> &LSV_Scalar::dedKTrainingVector(Matrix<double> &res) const
{
    res.resize(N(),N());

    double tmp;
    int i,j;

    for ( i = 0 ; i < N() ; ++i )
    {
        if ( alphaState()(i) )
        {
            dedgTrainingVector(tmp,i);

            for ( j = 0 ; j < N() ; ++j )
            {
                if ( alphaState()(j) )
                {
                    res("&",i,j) = tmp*alphaR()(j);
                }
            }
        }
    }

    return res;
}

Vector<double> &LSV_Scalar::dedKTrainingVector(Vector<double> &res, int i) const
{
    NiceAssert( i < N() );

    double tmp;
    int j;

    res.resize(N());

    {
        {
            dedgTrainingVector(tmp,i);

            for ( j = 0 ; j < N() ; ++j )
            {
                if ( alphaState()(j) )
                {
                    res("&",j) = tmp*alphaR()(j);
                }
            }
        }
    }

    return res;
}

//FIXME test to make sure posterior variance is still working!
//FIXME: do approximate version covm (see lsv_generic) here.  Find closest covm points, use these as observation points in variance calculation 
//FIXME: with explicit inverse (chol version with diagonal offsets).  Then bridge this though GPR modules.
int LSV_Scalar::covTrainingVector(gentype &resv, gentype &resmu, int ia, int ib, gentype ***pxyprodi, gentype ***pxyprodj, gentype **pxyprodij) const
{
//errstream() << "phantomxmeep 0: " << x(ia) << "," << x(ib) << "\n";
    int tres = 0;

    int dtva = xtang(ia) & (7+32+64);
    int dtvb = xtang(ib) & (7+32+64);

    int covm = ( ( ia != ib ) && !varApprox() ) ? -1 : varApprox();

    NiceAssert( dtva >= 0 );
    NiceAssert( dtvb >= 0 );

    // This is used elsewhere (ie not scalar), so the following is relevant

//FIXME: resmu
    if ( ( dtva & 4 ) || ( dtvb & 4 ) || !isUnderlyingScalar() )
    {
        if ( NNC(-1) || NNC(+1) || NNC(2) )
        {
            int j;

            Vector<gentype> Kia(N());
            Vector<gentype> Kib((( covm && ( ia != ib ) ) ? N() : 0));
            Vector<gentype> itsone(1);//isVarBias() ? 1 : 0);
            gentype Kiaib,Kiaia,Kibib;

            itsone("&",0) = 1.0;

            if ( ( ia >= 0 ) && ( ib >= 0 ) )
            {
                Kiaib  = Gp()(ia,ib);
                Kiaib -= ( ia == ib ) ? diagoffset()(ia) : 0.0;
            }

            else
            {
                K2(Kiaib,ia,ib,(const gentype **) pxyprodij);
            }

            if ( covm && ( ia >= 0 ) )
            {
                for ( j = 0 ; j < N() ; ++j )
                {
                    if ( alphaState()(j) ) //|| ( ia == j ) || ( ib == j ) )
                    {
                        Kia("&",j) = Gp()(ia,j);
                    }

                    else
                    {
                        Kia("&",j) = 0.0;
                    }
                }

                if ( alphaState()(ia) )
                {
                    Kia("&",ia) -= diagoffset()(ia);
                }

                Kiaia = Kia(ia);
            }

            else if ( covm )
            {
                for ( j = 0 ; j < N() ; ++j )
                {
                    if ( alphaState()(j) ) //|| ( ia == j ) || ( ib == j ) )
                    {
                        K2(Kia("&",j),ia,j,pxyprodi ? (const gentype **) pxyprodi[j] : nullptr);
                    }

                    else
                    {
                        Kia("&",j) = 0.0;
                    }
                }

                K2(Kiaia,ia,ia,nullptr);
            }

            if ( covm && ( ia == ib ) )
            {
                Kibib = Kiaia;
            }

            else if ( covm && ( ib >= 0 ) && ( ia != ib ) )
            {
                for ( j = 0 ; j < N() ; ++j )
                {
                    if ( alphaState()(j) ) //|| ( ia == j ) || ( ib == j ) )
                    {
                        Kib("&",j) = Gp()(j,ib);
                    }

                    else
                    {
                        Kib("&",j) = 0.0;
                    }
                }

                if ( alphaState()(ib) )
                {
                    Kib("&",ib) -= diagoffset()(ib);
                }

                Kibib = Kib(ib);
            }

            else if ( covm && ( ia != ib ) )
            {
                for ( j = 0 ; j < N() ; ++j )
                {
                    if ( alphaState()(j) ) //|| ( ia == j ) || ( ib == j ) )
                    {
                        //K2(Kib("&",j),j,ib,pxyprodj ? (const gentype **) pxyprodj[j] : nullptr);
                        K2(Kib("&",j),ib,j,pxyprodj ? (const gentype **) pxyprodj[j] : nullptr); // do this to suit the assumptions in mercer Kxfer (that the unknown "x" is first)
                        setconj(Kib("&",j));
                    }

                    else
                    {
                        Kib("&",j) = 0.0;
                    }
                }

                K2(Kibib,ib,ib,nullptr);
            }

            // covariance calculation

            if ( !covm )
            {
                // Calculate prior variance

                resv = Kiaib;
            }

            else if ( covm == -1 )
            {
                resv = Kiaib;

                // Calculate real posterior variance

                Vector<gentype> btemp(1);//isVarBias() ? 1 : 0);
                Vector<gentype> Kres(N());

                if ( ia == ib )
                {
                    //NB: this will automatically only do part corresponding to pivAlphaF
                    fact_minverse(Kres,btemp,Kia,itsone); // recall Kib not defined for ia == ib
                }

                else
                {
                    //NB: this will automatically only do part corresponding to pivAlphaF
                    fact_minverse(Kres,btemp,Kib,itsone);
                }

                for ( j = 0 ; j < N() ; ++j )
                {
                    // This is important or variance will be fracking negative!!!!

                    if ( ( alphaState()(j) == -1 ) || ( alphaState()(j) == +1 ) )
                    {
                        resv -= outerProd(Kia(j),Kres(j));
                    }
                }

                if ( isVarBias() )
                {
                    // This is the additional corrective factor

                    resv -= btemp(0);
                }
            }

            else
            {
//errstream() << "$"; //phantomxyzxyz
                // Calculate approximate posterior variance

                int realcovm = 0;

                Vector<double> vardist(covm+1); // deliberately oversize by one to simplify code
                Vector<int> varind(covm+1); // deliberately oversize by one to simplify code

                // First find the closest covm vectors

                gentype Kjj;
                int k;
                double distmax = 0;

                for ( j = 0 ; j < N() ; ++j )
                {
                    // This is important or variance will be fracking negative!!!!

                    if ( ( alphaState()(j) == -1 ) || ( alphaState()(j) == +1 ) )
                    {
                        Kjj  = Gp()(j,j);
                        Kjj -= diagoffset()(j);

                        double iajdist = 0;
                        double ibjdist = 0;
                        double jdist = 0;

                        if ( ia == ib )
                        {
                            jdist = (double) (Kiaia+Kjj-Kia(j)-Kia(j));
                        }

                        else
                        {
                            iajdist = (double) (Kiaia+Kjj-Kia(j)-Kia(j));
                            ibjdist = (double) (Kibib+Kjj-Kib(j)-Kib(j));

                            jdist = ( iajdist <= ibjdist ) ? iajdist : ibjdist;
                        }

                        if ( ( jdist < distmax ) || ( realcovm < covm ) )
                        {
                            for ( k = 0 ; k < realcovm ; k++ )
                            {
                                if ( jdist < vardist(k) )
                                {
                                    break;
                                }
                            }

                            //if ( ( k < realcovm ) || ( realcovm < covm ) )
                            {
                                int kk = k;

                                for ( k = realcovm ; k > kk ; k-- )
                                {
                                    vardist("&",k) = vardist(k-1);
                                    varind("&",k)  = varind(k-1);
                                }

                                vardist("&",kk) = jdist;
                                varind("&",kk) = j;

                                if ( realcovm < covm )
                                {
                                    realcovm++;
                                }
                            }

                            distmax = vardist(realcovm-1);
                        }
                    }
                }

                vardist.resize(realcovm);
                varind.resize(realcovm);

                // Now approximate the posterior covariance

                NiceAssert( !isVarBias() );

                resv = Kiaib;

                // Calculate real posterior variance

                Vector<gentype> Kres(realcovm);
                Matrix<double> Gpinv(realcovm,realcovm);

                retVector<gentype> tmpva;
                retMatrix<double> tmpma;

                Gp()(varind,varind,tmpma).inveSymm(Gpinv); // cubic cost, but better if realcovm is small

                if ( ia == ib )
                {
                    Kres = Kia(varind,tmpva);
                    Kres *= Gpinv;
                }

                else
                {
                    Kres = Kib(varind,tmpva);
                    Kres *= Gpinv;
                }

                for ( j = 0 ; j < realcovm ; ++j )
                {
                    // This is important or variance will be fracking negative!!!!

                    if ( ( alphaState()(varind(j)) == -1 ) || ( alphaState()(varind(j)) == +1 ) )
                    {
                        resv -= outerProd(Kia(varind(j)),Kres(j));
                    }
                }
            }

            // mu calculation

            int firstterm = 1;

            for ( j = 0 ; j < N() ; ++j )
            {
                if ( firstterm )
                {
                    if ( ( alphaState()(j) == -1 ) || ( alphaState()(j) == +1 ) )
                    {
                        resmu = Kia(j)*dalpha(j);

                        firstterm = 0;
                    }
                }

                else
                {
                    if ( ( alphaState()(j) == -1 ) || ( alphaState()(j) == +1 ) )
                    {
                        resmu += Kia(j)*dalpha(j);
                    }
                }
            }

            if ( !( dtva & 7 ) )
            {
                if ( firstterm )
                {
                    resmu = dbias;
                    firstterm = 0;
                }

                else
                {
                    resmu += dbias;
                }
            }

            else
            {
                if ( firstterm )
                {
                    resmu =  dbiasR;
                    resmu *= 0.0;
                    firstterm = 0;
                }
            }
        }

        else
        {
            if ( ( ia >= 0 ) && ( ib >= 0 ) )
            {
                resv  = Gp()(ia,ib);
                resv -= ( ia == ib ) ? diagoffset()(ia) : 0.0;
            }

            else
            {
                K2(resv,ia,ib,(const gentype **) pxyprodij);
            }

            if ( !( dtva & 7 ) )
            {
                resmu = dbias;
            }

            else
            {
                resmu  = dbias;
                resmu *= 0.0;
            }
        }

        if ( getKernel().isKVarianceNZ() )
        {
            NiceAssert( ia == ib );

            gentype addres(0.0);
            gentype Kxj;
            int j;

            for ( j = 0 ; j < N() ; ++j )
            {
                K2(Kxj,ia,j,nullptr,nullptr,nullptr,nullptr,nullptr,0x80);
                addres += ((double) dalpha(j))*((double) dalpha(j))*Kxj;
            }

            resv += addres;
        }

        if ( ( ia == ib ) && ( resv <= 0.0_gent ) ) //zerogentype() ) )
        {
            // Sometimes numerical issues can make the variance (ia == ib, so 
            // this is variance) negative.  This causes issues when eg bayesian
            // optimisation attempts to take the square-root and cast the
            // result to real.  Hence this "fix-fudge" to make it zero instead.

            resv = 0.0_gent; //zerogentype();
        }
    }

    else
    {
//errstream() << "phantomxyggg 0\n";
        double &resvv = resv.force_double();
        double &resgg = resmu.force_double();

        if ( dtva & 7 )
        {
            resgg = 0.0;
        }

        else
        {
            resgg = dbiasR;
        }

//errstream() << "phantomxyggg 1\n";
        if ( NNC(-1) || NNC(+1) || NNC(2) )
        {
//errstream() << "phantomxyggg 2\n";
//errstream() << "phantomxyggg 2 ia: " << x(ia) << "\n";
//errstream() << "phantomxyggg 2 ia: " << x(ib) << "\n";
//errstream() << "phantomxyggg 2 ia: " << xinfo(ia) << "\n";
//errstream() << "phantomxyggg 2 ia: " << xinfo(ib) << "\n";
            int j;

            Vector<double> Kia(N());
            Vector<double> Kib((( covm && ( ia != ib ) ) ? N() : 0));
            Vector<double> itsone(1);//isVarBias() ? 1 : 0);
            double Kiaib,Kiaia,Kibib;

            itsone("&",0) = 1.0;

            if ( ( ia >= 0 ) && ( ib >= 0 ) )
            {
                Kiaib  = Gp()(ia,ib);
                Kiaib -= ( ia == ib ) ? diagoffset()(ia) : 0.0;
            }

            else
            {
                Kiaib = K2(ia,ib,(const gentype **) pxyprodij);
            }

//errstream() << "phantomxyggg 2 Kiaib = " << Kiaib << "\n";
            if ( covm && ( ia >= 0 ) )
            {
                for ( j = 0 ; j < N() ; ++j )
                {
                    if ( alphaState()(j) ) //|| ( ia == j ) || ( ib == j ) )
                    {
                        Kia("&",j) = Gp()(ia,j);
                    }

                    else
                    {
                        Kia("&",j) = 0.0;
                    }
                }

                if ( alphaState()(ia) )
                {
                    Kia("&",ia) -= diagoffset()(ia);
                }

                Kiaia = Kia(ia);
            }

            else if ( covm )
            {
                for ( j = 0 ; j < N() ; ++j )
                {
                    if ( alphaState()(j) ) //|| ( ia == j ) || ( ib == j ) )
                    {
                        Kia("&",j) = K2(ia,j,pxyprodi ? (const gentype **) pxyprodi[j] : nullptr);
                    }

                    else
                    {
                        Kia("&",j) = 0.0;
                    }
                }

                Kiaia = K2(ia,ia,nullptr);
//errstream() << "phantomxyggg 2 Gp = " << Gp() << "\n";
//errstream() << "phantomxyggg 2 kernel = " << getKernel() << "\n";
//errstream() << "phantomxyggg 2 Kiaia = " << Kiaia << "\n";
//errstream() << "phantomxyggg 2 Kia = " << Kia << "\n";
            }

            Kibib = Kiaia;

            if ( covm && ( ia == ib ) )
            {
                ;
            }

            else if ( covm && ( ib >= 0 ) && ( ia != ib ) )
            {
                for ( j = 0 ; j < N() ; ++j )
                {
                    if ( alphaState()(j) ) //|| ( ia == j ) || ( ib == j ) )
                    {
                        Kib("&",j) = Gp()(j,ib);
                    }

                    else
                    {
                        Kib("&",j) = 0.0;
                    }
                }

                if ( alphaState()(ib) )
                {
                    Kib("&",ib) -= diagoffset()(ib);
                }

                Kibib = Kib(ib);
            }

            else if ( covm && ( ia != ib ) )
            {
                for ( j = 0 ; j < N() ; ++j )
                {
                    if ( alphaState()(j) ) //|| ( ia == j ) || ( ib == j ) )
                    {
                        //K2(Kib("&",j),j,ib,pxyprodj ? (const gentype **) pxyprodj[j] : nullptr); - see above
                        Kib("&",j) = K2(ib,j,pxyprodj ? (const gentype **) pxyprodj[j] : nullptr);
                    }

                    else
                    {
                        Kib("&",j) = 0.0;
                    }
                }

                Kibib = K2(ib,ib,nullptr);
            }

            // covariance calculation

            if ( !covm )
            {
                resvv = Kiaib;
            }

            else if ( covm == -1 )
            {
                resvv = Kiaib;

                Vector<double> btemp(1);//isVarBias() ? 1 : 0);
                Vector<double> Kres(N());

                if ( ia == ib )
                {
                    //NB: this will automatically only do part corresponding to pivAlphaF
                    fact_minverse(Kres,btemp,Kia,itsone); // Kib not defined at this point
                }

                else
                {
                    //NB: this will automatically only do part corresponding to pivAlphaF
                    fact_minverse(Kres,btemp,Kib,itsone);
                }

                for ( j = 0 ; j < N() ; ++j )
                {
                    // This is important or variance will be fracking negative!!!!

                    if ( ( alphaState()(j) == -1 ) || ( alphaState()(j) == +1 ) )
                    {
                        resvv -= Kia(j)*Kres(j);
                    }
                }

                if ( isVarBias() )
                {
                    // This is the additional corrective factor

                    resvv -= btemp(0);
                }
            }

            else
            {
//errstream() << "$"; //phantomxyzxyz
                // Calculate approximate posterior variance

                int realcovm = 0;

                Vector<double> vardist(covm+1); // deliberately oversize by one to simplify code
                Vector<int> varind(covm+1); // deliberately oversize by one to simplify code

                // First find the closest covm vectors

                double Kjj;
                int k;
                double distmax = 0;

                for ( j = 0 ; j < N() ; ++j )
                {
                    // This is important or variance will be fracking negative!!!!

                    if ( ( alphaState()(j) == -1 ) || ( alphaState()(j) == +1 ) )
                    {
                        Kjj = Gp()(j,j) - diagoffset()(j);

                        double iajdist = 0;
                        double ibjdist = 0;
                        double jdist = 0;

                        if ( ia == ib )
                        {
                            jdist = (Kiaia+Kjj-(2*Kia(j)));
                        }

                        else
                        {
                            iajdist = (Kiaia+Kjj-(2*Kia(j)));
                            ibjdist = (Kibib+Kjj-(2*Kib(j)));

                            jdist = ( iajdist <= ibjdist ) ? iajdist : ibjdist;
                        }

                        if ( ( jdist < distmax ) || ( realcovm < covm ) )
                        {
                            for ( k = 0 ; k < realcovm ; k++ )
                            {
                                if ( jdist < vardist(k) )
                                {
                                    break;
                                }
                            }

//                            if ( ( k < realcovm ) || ( realcovm < covm ) )
                            {
                                int kk = k;

                                for ( k = realcovm ; k > kk ; k-- )
                                {
                                    vardist("&",k) = vardist(k-1);
                                    varind("&",k)  = varind(k-1);
                                }

                                vardist("&",kk) = jdist;
                                varind("&",kk) = j;

                                if ( realcovm < covm )
                                {
                                    realcovm++;
                                }
                            }

                            distmax = vardist(realcovm-1);
                        }
                    }
                }

                vardist.resize(realcovm);
                varind.resize(realcovm);

                // Now approximate the posterior covariance

                NiceAssert( !isVarBias() );

                resvv = Kiaib;

                // Calculate real posterior variance

                Vector<double> Kres(realcovm);
                Matrix<double> Gpinv(realcovm,realcovm);

                retVector<double> tmpva;
                retMatrix<double> tmpma;

                Gp()(varind,varind,tmpma).inveSymm(Gpinv); // cubic cost, but better if realcovm is small

                if ( ia == ib )
                {
                    Kres = Kia(varind,tmpva);
                    Kres *= Gpinv;
                }

                else
                {
                    Kres = Kib(varind,tmpva);
                    Kres *= Gpinv;
                }

                for ( j = 0 ; j < realcovm ; ++j )
                {
                    // This is important or variance will be fracking negative!!!!

                    if ( ( alphaState()(varind(j)) == -1 ) || ( alphaState()(varind(j)) == +1 ) )
                    {
                        resvv -= Kia(varind(j))*Kres(j);
                    }
                }
            }

            // mu calculation

            for ( j = 0 ; j < N() ; ++j )
            {
                if ( alphaState()(j) )
                {
                    resgg += Kia(j)*dalphaR(j);
                }
            }
        }

        else
        {
//errstream() << "phantomxyggg 100\n";
            if ( ( ia >= 0 ) && ( ib >= 0 ) )
            {
                resvv  = Gp()(ia,ib);
                resvv -= ( ia == ib ) ? diagoffset()(ia) : 0.0;
            }

            else
            {
//errstream() << "phantomxyggg 101\n";
                resvv = K2(ia,ib,(const gentype **) pxyprodij);
            }

            if ( !( dtva & 7 ) )
            {
                resgg = dbias;
            }

            else
            {
                resgg  = dbias;
                resgg *= 0.0;
            }
//errstream() << "phantomxyggg 103\n";
        }

        if ( getKernel().isKVarianceNZ() )
        {
            NiceAssert( ia == ib );

            double addres = 0.0;
            double Kxj;
            int j;

            for ( j = 0 ; j < N() ; ++j )
            {
                Kxj = K2(ia,j,nullptr,nullptr,nullptr,nullptr,nullptr,0x80);
                addres += ((double) dalpha(j))*((double) dalpha(j))*Kxj;
            }

            resvv += addres;
        }

        if ( ( ia == ib ) && ( resvv <= 0.0 ) )
        {
            // Sometimes numerical issues can make the variance (ia == ib, so
            // this is variance) negative.  This causes issues when eg bayesian
            // optimisation attempts to take the square-root and cast the
            // result to real.  Hence this "fix-fudge" to make it zero instead.

            resvv = 0.0;
        }
    }

        if ( mpri() )
        {
            if ( ia >= 0 )
            {
                resmu += yp()(ia);
            }

            else
            {
                gentype resprior;

                calcprior(resprior,x(ia));

                resmu += resprior;
            }
        }

    return tres;
}










