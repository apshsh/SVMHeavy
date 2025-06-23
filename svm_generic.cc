

//
// SVM base class
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
//#ifdef ENABLE_THREADS
//#include <mutex>
//#endif
#include "svm_generic.hpp"

std::ostream &SVM_Generic::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Base training alpha:       " << dalpha      << "\n";
    repPrint(output,'>',dep) << "Base training bias:        " << dbias       << "\n";
    repPrint(output,'>',dep) << "Base training alpha state: " << xalphaState << "\n";

    ML_Base::printstream(output,dep+1);

    return output;
}

std::istream &SVM_Generic::inputstream(std::istream &input )
{
    wait_dummy dummy;

    input >> dummy; input >> dalpha;
    input >> dummy; input >> dbias;
    input >> dummy; input >> xalphaState;

    ML_Base::inputstream(input);

    return input;
}

int SVM_Generic::getInternalClass(const gentype &y) const
{
    int res = 0;

    if ( ( type() == 0 ) || ( type() == 22 ) )
    {
        res = y.isCastableToRealWithoutLoss() ? ( ( ( (double) y ) < 0 ) ? 0 : 1 ) : 0;
    }

    else if ( isClassifier() )
    {
        if ( isanomalyOn() )
        {
            if ( (int) y == anomalyClass() )
            {
                res = numClasses();
            }

            else
            {
                res = findID((int) y);
            }
        }

        else
        {
            res = findID((int) y);
        }

        if ( res == -1 )
        {
            if ( isanomalyOn() || ( ( type() == 3 ) && ( subtype() == 0 ) ) )
            {
                // In this case if we don't recognise a class *but* there is
                // an anomaly class defined *then* register this as an anomaly

                //res = dynamic_cast<const SVM_MultiC_atonce &>(*this).grablinbfq();
                res = grablinbfq();
            }
        }
    }

    return res;
}

int SVM_Generic::setFixedBias(const gentype &newBias)
{
    if ( isUnderlyingVector() )
    {
	Vector<gentype> simplebias((const Vector<gentype> &) newBias);

	int i;

	if ( simplebias.size() )
	{
	    for ( i = 0 ; i < simplebias.size() ; ++i )
	    {
                setFixedBias(i,(double) simplebias(i));
	    }
	}
    }

    else if ( isUnderlyingAnions() )
    {
        d_anion simplebias((const d_anion &) newBias);

        setBiasA(simplebias);
        setFixedBias((double) simplebias(0));
    }

    else
    {
        setFixedBias((double) newBias);
    }

    return 1;
}

int SVM_Generic::addTrainingVector (int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh, double epsweigh, int dval)
{
    incgvernum();

    NiceAssert( i >= 0 );
    //NiceAssert( i <= SVM_Generic::N() );

    int res = ML_Base::addTrainingVector(i,y,x,Cweigh,epsweigh,dval);

    dalpha.add(i);      dalpha("&",i) = 0;
    xalphaState.add(i); xalphaState("&",i) = dval ? 1 : 0; //1;

    return res;
}

int SVM_Generic::qaddTrainingVector(int i, const gentype &y,       SparseVector<gentype> &x, double Cweigh, double epsweigh, int dval)
{
    incgvernum();

    NiceAssert( i >= 0 );
    //NiceAssert( i <= SVM_Generic::N() );

    int res = ML_Base::qaddTrainingVector(i,y,x,Cweigh,epsweigh,dval);

    dalpha.add(i);      dalpha("&",i) = 0;
    xalphaState.add(i); xalphaState("&",i) = dval ? 1 : 0; //1;

    return res;
}

int SVM_Generic::addTrainingVector(int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    int Nadd = y.size();

    incgvernum();

    NiceAssert( i >= 0 );
    //NiceAssert( i <= SVM_Generic::N() );
    NiceAssert( Nadd == x.size() );
    NiceAssert( Nadd == Cweigh.size() );
    NiceAssert( Nadd == epsweigh.size() );

    int res = ML_Base::addTrainingVector(i,y,x,Cweigh,epsweigh);

    retVector<gentype> tmpvgen;
    retVector<int> tmpvint;

    dalpha.addpad(i,Nadd);      dalpha("&",i,1,i+Nadd-1,tmpvgen)      = 0_gent;
    xalphaState.addpad(i,Nadd); xalphaState("&",i,1,i+Nadd-1,tmpvint) = 1;

    return res;
}

int SVM_Generic::qaddTrainingVector(int i, const Vector<gentype> &y, Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    int Nadd = y.size();

    incgvernum();

    NiceAssert( i >= 0 );
    //NiceAssert( i <= SVM_Generic::N() );
    NiceAssert( Nadd == x.size() );
    NiceAssert( Nadd == Cweigh.size() );
    NiceAssert( Nadd == epsweigh.size() );

    int res = ML_Base::qaddTrainingVector(i,y,x,Cweigh,epsweigh);

    retVector<gentype> tmpvgen;
    retVector<int> tmpvint;

    dalpha.addpad(i,Nadd);      dalpha("&",i,1,i+Nadd-1,tmpvgen)      = 0_gent;
    xalphaState.addpad(i,Nadd); xalphaState("&",i,1,i+Nadd-1,tmpvint) = 1;

    return res;
}

int SVM_Generic::removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x)
{
    incgvernum();

    NiceAssert( i >= 0 );
    //NiceAssert( i < SVM_Generic::N() );

    int res = ML_Base::removeTrainingVector(i,y,x);
    dalpha.remove(i);
    xalphaState.remove(i);

    return res;
}

int SVM_Generic::removeTrainingVector(int i, int num)
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







// Kernel transfer

int SVM_Generic::isKVarianceNZ(void) const
{
    return getKernel().isKVarianceNZ();
}

void SVM_Generic::fastg(double &res) const
{
    SparseVector<gentype> x;

    gg(res,x);

    return;
}

void SVM_Generic::fastg(double &res, 
                        int ia, 
                        const SparseVector<gentype> &xa, 
                        const vecInfo &xainfo) const
{
    (void) ia;
    (void) xainfo;

    gg(res,xa);

    return;
}

void SVM_Generic::fastg(double &res, 
                        int ia, int ib, 
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, 
                        const vecInfo &xainfo, const vecInfo &xbinfo) const
{
    (void) xainfo;
    (void) xbinfo;

    (void) ia;
    (void) ib;

    SparseVector<gentype> x(xa);

    if ( xb.indsize() )
    {
        int i;

        for ( i = 0 ; i < xb.indsize() ; ++i )
        {
            //x("&",xb.ind(i)+DEFAULT_TUPLE_INDEX_STEP) = xb.direcref(i);
            x("&",xb.ind(i),1) = xb.direcref(i);
        }
    }

    gg(res,x);

    return;
}

void SVM_Generic::fastg(double &res, 
                        int ia, int ib, int ic, 
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, 
                        const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo) const
{
    (void) xainfo;
    (void) xbinfo;
    (void) xcinfo;

    (void) ia;
    (void) ib;
    (void) ic;

    SparseVector<gentype> x(xa);

    if ( xb.indsize() )
    {
        int i;

        for ( i = 0 ; i < xb.indsize() ; ++i )
        {
            //x("&",xb.ind(i)+DEFAULT_TUPLE_INDEX_STEP) = xb.direcref(i);
            x("&",xb.ind(i),1) = xb.direcref(i);
        }
    }

    if ( xc.indsize() )
    {
        int i;

        for ( i = 0 ; i < xc.indsize() ; ++i )
        {
            //x("&",xc.ind(i)+(2*DEFAULT_TUPLE_INDEX_STEP)) = xc.direcref(i);
            x("&",xc.ind(i),2) = xc.direcref(i);
        }
    }

    gg(res,x);

    return;
}

void SVM_Generic::fastg(double &res, 
                        int ia, int ib, int ic, int id,
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                        const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo) const
{
    (void) xainfo;
    (void) xbinfo;
    (void) xcinfo;
    (void) xdinfo;

    (void) ia;
    (void) ib;
    (void) ic;
    (void) id;

    SparseVector<gentype> x(xa);

    if ( xb.indsize() )
    {
        int i;

        for ( i = 0 ; i < xb.indsize() ; ++i )
        {
            //x("&",xb.ind(i)+DEFAULT_TUPLE_INDEX_STEP) = xb.direcref(i);
            x("&",xb.ind(i),1) = xb.direcref(i);
        }
    }

    if ( xc.indsize() )
    {
        int i;

        for ( i = 0 ; i < xc.indsize() ; ++i )
        {
            //x("&",xc.ind(i)+(2*DEFAULT_TUPLE_INDEX_STEP)) = xc.direcref(i);
            x("&",xc.ind(i),2) = xc.direcref(i);
        }
    }

    if ( xd.indsize() )
    {
        int i;

        for ( i = 0 ; i < xd.indsize() ; ++i )
        {
            //x("&",xd.ind(i)+(3*DEFAULT_TUPLE_INDEX_STEP)) = xd.direcref(i);
            x("&",xd.ind(i),3) = xd.direcref(i);
        }
    }

    gg(res,x);

    return;
}

void SVM_Generic::fastg(double &res,
                        Vector<int> &ia,
                        Vector<const SparseVector<gentype> *> &xa,
                        Vector<const vecInfo *> &xainfo) const
{
    (void) xainfo;
    (void) ia;

    SparseVector<gentype> x;

    if ( xa.size() )
    {
        int i,j;

        for ( j = 0 ; j < xa.size() ; ++j )
        {
            const SparseVector<gentype> &xb = (*(xa(j)));

            if ( xb.indsize() )
            {
                for ( i = 0 ; i < xb.indsize() ; ++i )
                {
                    //x("&",xb.ind(i)+(j*DEFAULT_TUPLE_INDEX_STEP)) = xb.direcref(i);
                    x("&",xb.ind(i),j) = xb.direcref(i);
                }
            }
        }
    }

    gg(res,x);

    return;
}


void SVM_Generic::fastg(gentype &res) const
{
    SparseVector<gentype> x;

    gg(res,x);

    return;
}

void SVM_Generic::fastg(gentype &res, 
                        int ia, 
                        const SparseVector<gentype> &xa, 
                        const vecInfo &xainfo) const
{
    (void) ia;
    (void) xainfo;

    gg(res,xa);

    return;
}

void SVM_Generic::fastg(gentype &res,
                        int ia, int ib,
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                        const vecInfo &xainfo, const vecInfo &xbinfo) const
{
    (void) xainfo;
    (void) xbinfo;

    (void) ia;
    (void) ib;

    SparseVector<gentype> x(xa);

    if ( xb.indsize() )
    {
        int i;

        for ( i = 0 ; i < xb.indsize() ; ++i )
        {
            //x("&",xb.ind(i)+DEFAULT_TUPLE_INDEX_STEP) = xb.direcref(i);
            x("&",xb.ind(i),1) = xb.direcref(i);
        }
    }

    gg(res,x);

    return;
}

void SVM_Generic::fastg(gentype &res,
                        int ia, int ib, int ic,
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc,
                        const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo) const
{
    (void) xainfo;
    (void) xbinfo;
    (void) xcinfo;

    (void) ia;
    (void) ib;
    (void) ic;

    SparseVector<gentype> x(xa);

    if ( xb.indsize() )
    {
        int i;

        for ( i = 0 ; i < xb.indsize() ; ++i )
        {
            //x("&",xb.ind(i)+DEFAULT_TUPLE_INDEX_STEP) = xb.direcref(i);
            x("&",xb.ind(i),1) = xb.direcref(i);
        }
    }

    if ( xc.indsize() )
    {
        int i;

        for ( i = 0 ; i < xc.indsize() ; ++i )
        {
            //x("&",xc.ind(i)+(2*DEFAULT_TUPLE_INDEX_STEP)) = xc.direcref(i);
            x("&",xc.ind(i),2) = xc.direcref(i);
        }
    }

    gg(res,x);

    return;
}

void SVM_Generic::fastg(gentype &res,
                        int ia, int ib, int ic, int id,
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                        const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo) const
{
    (void) xainfo;
    (void) xbinfo;
    (void) xcinfo;
    (void) xdinfo;

    (void) ia;
    (void) ib;
    (void) ic;
    (void) id;

    SparseVector<gentype> x(xa);

    if ( xb.indsize() )
    {
        int i;

        for ( i = 0 ; i < xb.indsize() ; ++i )
        {
            //x("&",xb.ind(i)+DEFAULT_TUPLE_INDEX_STEP) = xb.direcref(i);
            x("&",xb.ind(i),1) = xb.direcref(i);
        }
    }

    if ( xc.indsize() )
    {
        int i;

        for ( i = 0 ; i < xc.indsize() ; ++i )
        {
            //x("&",xc.ind(i)+(2*DEFAULT_TUPLE_INDEX_STEP)) = xc.direcref(i);
            x("&",xc.ind(i),2) = xc.direcref(i);
        }
    }

    if ( xd.indsize() )
    {
        int i;

        for ( i = 0 ; i < xd.indsize() ; ++i )
        {
            //x("&",xd.ind(i)+(3*DEFAULT_TUPLE_INDEX_STEP)) = xd.direcref(i);
            x("&",xd.ind(i),3) = xd.direcref(i);
        }
    }

    gg(res,x);

    return;
}

void SVM_Generic::fastg(gentype &res,
                        Vector<int> &ia,
                        Vector<const SparseVector<gentype> *> &xa,
                        Vector<const vecInfo *> &xainfo) const
{
    (void) xainfo;
    (void) ia;

    SparseVector<gentype> x;

    if ( xa.size() )
    {
        int i,j;

        for ( j = 0 ; j < xa.size() ; ++j )
        {
            const SparseVector<gentype> &xb = (*(xa(j)));

            if ( xb.indsize() )
            {
                for ( i = 0 ; i < xb.indsize() ; ++i )
                {
                    //x("&",xb.ind(i)+(j*DEFAULT_TUPLE_INDEX_STEP)) = xb.direcref(i);
                    x("&",xb.ind(i),j) = xb.direcref(i);
                }
            }
        }
    }

    gg(res,x);

    return;
}




void SVM_Generic::K0xfer(gentype &res, int &minmaxind, int typeis,
                         const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                         int xdim, int densetype, int resmode, int mlid) const
{
    NiceAssert( !densetype );

    gentype dummy;
    double dummyr = 0.0;

    switch ( typeis )
    {
        case 801:
        case 811:
        {
            res = 0.0;

            if ( NLB()+NUB()+NF() )
            {
                gentype Kij;

                int j,k;
                int docheat = ( ( ( resmode & 0x7F ) >= 0 ) && ( ( resmode & 0x7F ) <= 15 ) ) ? 1 : 0;

                for ( j = 0 ; j < N() ; ++j )
                {
                    if ( alphaState()(j) )
                    {
                        for ( k = 0 ; k < ( docheat ? j+1 : N() ) ; ++k )
                        {
                            if ( alphaState()(k) )
                            {
                                switch ( resmode & 0x7f )
                                {
                                    case 0:  case 1:  case 2:  case 3:
                                    case 4:  case 5:  case 6:  case 7:
                                    case 8:  case 9:  case 10: case 11:
                                    case 12: case 13: case 14: case 15:
                                    {
                                        K2(Kij,j,k,nullptr,nullptr,nullptr,nullptr,nullptr,resmode);

                                        break;
                                    }

                                    case 16: case 32: case 48:
                                    {
                                        // Can't do this as additional vectors <x,...> now present, does not fit with xygrad, xnormgrad return

                                        NiceThrow("K0xfer 801 not supported.");

                                        break;
                                    }

                                    default:
                                    {
                                        NiceThrow("K0xfer precursor specified resmode undefined at this level.");

                                        break;
                                    }
                                }

                                if ( isUnderlyingScalar() )
                                {
                                    Kij *= alphaR()(j)*alphaR()(k);
                                }

                                else if ( isUnderlyingVector() )
                                {
                                    Kij *= twoProduct(dummyr,alphaV()(j),alphaV()(k));
                                }

                                else
                                {
                                    Kij *= (double) real(innerProduct(dummy,alpha()(j),alpha()(k)));
                                }

                                if ( docheat )
                                {
                                    if ( k < j )
                                    {
                                        res += Kij;
                                        res += Kij;
                                    }

                                    else if ( k == j )
                                    {
                                        res += Kij;
                                    }
                                }

                                else
                                {
                                    res += Kij;
                                }
                            }
                        }
                    }
                }
            }

            break;
        }

        case 802:
        case 812:
        {
            NiceAssert( !(resmode & 0x80 ) );

            res = 0.0;

            if ( NLB()+NUB()+NF() )
            {
                gentype Kija,Kijb;

                int j,k;

                for ( j = 0 ; j < N() ; ++j )
                {
                    if ( alphaState()(j) )
                    {
                        for ( k = 0 ; k < N() ; ++k )
                        {
                            if ( alphaState()(k) )
                            {
                                switch ( resmode & 0x7f )
                                {
                                    case 0:  case 1:  case 2:  case 3:
                                    case 4:  case 5:  case 6:  case 7:
                                    case 8:  case 9:  case 10: case 11:
                                    case 12: case 13: case 14: case 15:
                                    {
                                        K1(Kija,j,nullptr,nullptr,nullptr,resmode);
                                        K1(Kijb,k,nullptr,nullptr,nullptr,resmode);

                                        Kija *= Kijb;

                                        break;
                                    }

                                    case 16: case 32: case 48:
                                    {
                                        // Can't do this as additional vectors <x,...> now present, does not fit with xygrad, xnormgrad return

                                        NiceThrow("K0xfer 802 not supported.");

                                        break;
                                    }

                                    case 64:
                                    {
                                        NiceThrow("K0xfer precursor second order derivatives not yet implemented.");

                                        break;
                                    }

                                    default:
                                    {
                                        NiceThrow("K0xfer precursor specified resmode undefined at this level.");

                                        break;
                                    }
                                }

                                if ( isUnderlyingScalar() )
                                {
                                    Kija *= alphaR()(j)*alphaR()(k);
                                }

                                else if ( isUnderlyingVector() )
                                {
                                    Kija *= twoProduct(dummyr,alphaV()(j),alphaV()(k));
                                }

                                else
                                {
                                    Kija *= (double) real(innerProduct(dummy,alpha()(j),alpha()(k)));
                                }

                                res += Kija;
                                res += Kija;
                            }
                        }
                    }
                }
            }

            break;
        }

        case 807:
        case 817:
        {
            gentype Kintemp;

            K0(Kintemp);

            res =  Kintemp;
            res *= Kintemp;
            res -= 1.0;
            res = -1.0/res;

            break;
        }

        default:
        {
            ML_Base::K0xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xdim,densetype,resmode,mlid);

            break;
        }
    }

    return;
}

void SVM_Generic::K0xfer(double &res, int &minmaxind, int typeis,
                         double xyprod, double yxprod, double diffis,
                         int xdim, int densetype, int resmode, int mlid) const
{
    NiceAssert( !densetype );

    if ( ( resmode & 0x7f ) >= 16 )
    {
        gentype tempa,tempb,tempc;

        gentype gxyprod(xyprod);
        gentype gyxprod(yxprod);
        gentype gdiffis(diffis);

        K0xfer(tempc,minmaxind,typeis,gxyprod,gyxprod,gdiffis,xdim,densetype,resmode,mlid);

        res = (double) tempc;

        return;
    }

    //gentype dummy;
    double dummyr = 0.0;

    switch ( typeis )
    {
        case 801:
        case 811:
        {
            res = 0.0;

            if ( NLB()+NUB()+NF() )
            {
                double Kij;

                int j,k;
                int docheat = ( ( ( resmode & 0x7F ) >= 0 ) && ( ( resmode & 0x7F ) <= 15 ) ) ? 1 : 0;

                for ( j = 0 ; j < N() ; ++j )
                {
                    if ( alphaState()(j) )
                    {
                        for ( k = 0 ; k < ( docheat ? j+1 : N() ) ; ++k )
                        {
                            if ( alphaState()(k) )
                            {
                                switch ( resmode & 0x7f )
                                {
                                    case 0:  case 1:  case 2:  case 3:
                                    case 4:  case 5:  case 6:  case 7:
                                    case 8:  case 9:  case 10: case 11:
                                    case 12: case 13: case 14: case 15:
                                    {
                                        Kij = K2(j,k,nullptr,nullptr,nullptr,nullptr,nullptr,resmode);

                                        break;
                                    }

                                    case 16: case 32: case 48:
                                    {
                                        // Can't do this as additional vectors <x,...> now present, does not fit with xygrad, xnormgrad return

                                        NiceThrow("K0xfer 801 not supported.");

                                        break;
                                    }

                                    default:
                                    {
                                        NiceThrow("K0xfer precursor specified resmode undefined at this level.");

                                        break;
                                    }
                                }

                                if ( isUnderlyingScalar() )
                                {
                                    Kij *= alphaR()(j)*alphaR()(k);
                                }

                                else if ( isUnderlyingVector() )
                                {
                                    Kij *= twoProduct(dummyr,alphaV()(j),alphaV()(k));
                                }

                                else
                                {
                                    Kij *= real(alphaA()(j)*conj(alphaA()(k)));
                                }

                                if ( docheat )
                                {
                                    if ( k < j )
                                    {
                                        res += Kij;
                                        res += Kij;
                                    }

                                    else if ( k == j )
                                    {
                                        res += Kij;
                                    }
                                }

                                else
                                {
                                    res += Kij;
                                }
                            }
                        }
                    }
                }
            }

            break;
        }

        case 802:
        case 812:
        {
            NiceAssert( !(resmode & 0x80 ) );

            res = 0.0;

            if ( NLB()+NUB()+NF() )
            {
                double Kija,Kijb;

                int j,k;

                for ( j = 0 ; j < N() ; ++j )
                {
                    if ( alphaState()(j) )
                    {
                        for ( k = 0 ; k < N() ; ++k )
                        {
                            if ( alphaState()(k) )
                            {
                                switch ( resmode & 0x7f )
                                {
                                    case 0:  case 1:  case 2:  case 3:
                                    case 4:  case 5:  case 6:  case 7:
                                    case 8:  case 9:  case 10: case 11:
                                    case 12: case 13: case 14: case 15:
                                    {
                                        Kija = K1(j,nullptr,nullptr,nullptr,resmode);
                                        Kijb = K1(k,nullptr,nullptr,nullptr,resmode);

                                        Kija *= Kijb;

                                        break;
                                    }

                                    case 16: case 32: case 48:
                                    {
                                        // Can't do this as additional vectors <x,...> now present, does not fit with xygrad, xnormgrad return

                                        NiceThrow("K0xfer 802 not supported.");

                                        break;
                                    }

                                    case 64:
                                    {
                                        NiceThrow("K0xfer precursor second order derivatives not yet implemented.");

                                        break;
                                    }

                                    default:
                                    {
                                        NiceThrow("K0xfer precursor specified resmode undefined at this level.");

                                        break;
                                    }
                                }

                                if ( isUnderlyingScalar() )
                                {
                                    Kija *= alphaR()(j)*alphaR()(k);
                                }

                                else if ( isUnderlyingVector() )
                                {
                                    Kija *= twoProduct(dummyr,alphaV()(j),alphaV()(k));
                                }

                                else
                                {
                                    Kija *= real(alphaA()(j)*conj(alphaA()(k)));
                                }

                                res += Kija;
                                res += Kija;
                            }
                        }
                    }
                }
            }

            break;
        }

        case 807:
        case 817:
        {
            double Kintemp;

            Kintemp = K0();

            res = 1/(1-Kintemp*Kintemp);

            break;
        }

        default:
        {
            ML_Base::K0xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xdim,densetype,resmode,mlid);

            break;
        }
    }

    return;
}






void SVM_Generic::K1xfer(gentype &res, int &minmaxind, int typeis,
                         const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                         const SparseVector<gentype> &xa, 
                         const vecInfo &xainfo, 
                         int ia, 
                         int xdim, int densetype, int resmode, int mlid) const
{
    NiceAssert( !densetype );

    (void) densetype;
    (void) xdim;
    gentype dummy;
    double dummyr = 0.0;

    ia = (typeis-(100*(typeis/100)))/10 ? ia : -42;

    switch ( typeis )
    {
        case 801:
        case 811:
        {
            if ( ia < 0 ) { setInnerWildpa(&xa,&xainfo); ia = -1; }

            Vector<const SparseVector<gentype> *> x(2);
            Vector<const vecInfo *> xinfo(2);
            Vector<int> i(2);

            x("&",0) = &xa;
            x("&",1)         = nullptr;

            xinfo("&",0) = &xainfo;
            xinfo("&",1)         = nullptr;

            i("&",0) = ia;
            i("&",1)         = 0;

            res = 0.0;

            if ( NLB()+NUB()+NF() )
            {
                gentype Kij;

                int j;


                for ( j = 0 ; j < N() ; ++j )
                {
                    if ( alphaState()(j) )
                    {
                                                i("&",1) = j;

                                                switch ( resmode & 0x7f )
                                                {
                                                    case 0:  case 1:  case 2:  case 3:
                                                    case 4:  case 5:  case 6:  case 7:
                                                    case 8:  case 9:  case 10: case 11:
                                                    case 12: case 13: case 14: case 15:
                                                    {
                                                        Km(2,Kij,i,nullptr,&x,&xinfo,resmode);

                                                        break;
                                                    }

                                                    case 64:
                                                    {
                                                        NiceThrow("K1xfer precursor second order derivatives not yet implemented.");

                                                        break;
                                                    }

                                                    default:
                                                    {
                                                        NiceThrow("K1xfer precursor specified resmode undefined at this level.");

                                                        break;
                                                    }
                                                }

                                                if ( isUnderlyingScalar() )
                                                {
                                                    Kij *= alphaR()(j);
                                                }

                                                else if ( isUnderlyingVector() )
                                                {
                                                    Kij *= oneProduct(dummyr,alphaV()(j));
                                                }

                                                else
                                                {
                                                    Kij *= (double) real(alpha()(j));
                                                }
                    }
                }
            }

            if ( ia < 0 ) { resetInnerWildp(); }

            break;
        }

        case 807:
        case 817:
        {
            gentype Kintemp;
            double dummyr;

            K1(Kintemp,xa,&xainfo);

            res = 0.0;

            if ( NLB()+NUB()+NF() )
            {
                gentype Kij;
                gentype temp;

                int j;

                for ( j = 0 ; j < N() ; ++j )
                {
                    if ( alphaState()(j) )
                    {
                        K1(Kij,j);

                        temp =  Kintemp;
                        temp *= Kij;
                        temp -= 1.0;
                        temp = -1.0/temp;

                        if ( isUnderlyingScalar() )
                        {
                            temp *= alphaR()(j);
                        }

                        else if ( isUnderlyingVector() )
                        {
                            temp *= oneProduct(dummyr,alphaV()(j));
                        }

                        else
                        {
                            temp *= (double) real(alpha()(j));
                        }

                        res += temp;
                    }
                }
            }

            break;
        }

        default:
        {
            ML_Base::K1xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xainfo,ia,xdim,densetype,resmode,mlid);

            break;
        }
    }

    return;
}

void SVM_Generic::K1xfer(double &res, int &minmaxind, int typeis,
                         double xyprod, double yxprod, double diffis,
                         const SparseVector<gentype> &xa, 
                         const vecInfo &xainfo, 
                         int ia, 
                         int xdim, int densetype, int resmode, int mlid) const
{
    NiceAssert( !densetype );

    if ( ( resmode & 0x7f ) >= 16 )
    {
        gentype temp;

        gentype gxyprod(xyprod);
        gentype gyxprod(yxprod);
        gentype gdiffis(diffis);

        K1xfer(temp,minmaxind,typeis,gxyprod,gyxprod,gdiffis,xa,xainfo,ia,xdim,densetype,resmode,mlid);

        return;
    }

    //gentype dummy;
    double dummyr = 0.0;

    ia = (typeis-(100*(typeis/100)))/10 ? ia : -42;

    switch ( typeis )
    {
        case 801:
        case 811:
        {
            if ( ia < 0 ) { setInnerWildpa(&xa,&xainfo); ia = -1; }

            Vector<const SparseVector<gentype> *> x(2);
            Vector<const vecInfo *> xinfo(2);
            Vector<int> i(2);

            x("&",0) = &xa;
            x("&",1)         = nullptr;

            xinfo("&",0) = &xainfo;
            xinfo("&",1)         = nullptr;

            i("&",0) = ia;
            i("&",1)         = 0;

            res = 0.0;

            if ( NLB()+NUB()+NF() )
            {
                gentype Kij;

                int j;


                for ( j = 0 ; j < N() ; ++j )
                {
                    if ( alphaState()(j) )
                    {
                                                i("&",1) = j;

                                                switch ( resmode & 0x7f )
                                                {
                                                    case 0:  case 1:  case 2:  case 3:
                                                    case 4:  case 5:  case 6:  case 7:
                                                    case 8:  case 9:  case 10: case 11:
                                                    case 12: case 13: case 14: case 15:
                                                    {
                                                        Km(2,Kij,i,nullptr,&x,&xinfo,resmode);

                                                        break;
                                                    }

                                                    case 64:
                                                    {
                                                        NiceThrow("K1xfer precursor second order derivatives not yet implemented.");

                                                        break;
                                                    }

                                                    default:
                                                    {
                                                        NiceThrow("K1xfer precursor specified resmode undefined at this level.");

                                                        break;
                                                    }
                                                }

                                                if ( isUnderlyingScalar() )
                                                {
                                                    Kij *= alphaR()(j);
                                                }

                                                else if ( isUnderlyingVector() )
                                                {
                                                    Kij *= oneProduct(dummyr,alphaV()(j));
                                                }

                                                else
                                                {
                                                    Kij *= real(alphaA()(j));
                                                }
                    }
                }
            }

            if ( ia < 0 ) { resetInnerWildp(); }

            break;
        }

        case 807:
        case 817:
        {
            gentype Kintemp;
            double dummyr;

            K1(Kintemp,xa,&xainfo);

            res = 0.0;

            if ( NLB()+NUB()+NF() )
            {
                gentype Kij;
                gentype temp;

                int j;

                for ( j = 0 ; j < N() ; ++j )
                {
                    if ( alphaState()(j) )
                    {
                        K1(Kij,j);

                        temp =  Kintemp;
                        temp *= Kij;
                        temp -= 1.0;
                        temp = -1.0/temp;

                        if ( isUnderlyingScalar() )
                        {
                            temp *= alphaR()(j);
                        }

                        else if ( isUnderlyingVector() )
                        {
                            temp *= oneProduct(dummyr,alphaV()(j));
                        }

                        else
                        {
                            temp *= real(alphaA()(j));
                        }

                        res += (double) temp;
                    }
                }
            }

            break;
        }

        default:
        {
            ML_Base::K1xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xainfo,ia,xdim,densetype,resmode,mlid);

            break;
        }
    }

    return;
}









void SVM_Generic::K2xfer(gentype &dxyprod, gentype &ddiffis, gentype &res, int &minmaxind, int typeis,
                         const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                         const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                         const vecInfo &xainfo, const vecInfo &xbinfo,
                         int ia, int ib,
                         int xdim, int densetype, int resmode, int mlid) const
{
errstream() << "BADBADBAD";
    NiceAssert( !densetype );

    gentype dummy;
    double dummyr = 0.0;

    int iacall = ia;
    int ibcall = ib;

    ia = (typeis-(100*(typeis/100)))/10 ? ia : -42; // ia if type == x1x,x2x,... (ie assuming shared dataset), -42 otherwise
    ib = (typeis-(100*(typeis/100)))/10 ? ib : -43; // ib if type == x1x,x2x,... (ie assuming shared dataset), -43 otherwise

    switch ( typeis )
    {
        case 800:
        case 810:
        {
//errstream() << "phantomx 0 gentype: " << xa << "," << xb << "\n";
            if ( !resmode && isUnderlyingScalar() && ( ia >= 0 ) && ( ib >= 0 ) )
            {
                res = ( ia == ib ) ? kerndiag()(ia) : Gp()(ia,ib);
            }

            ML_Base::K2xfer(dxyprod,ddiffis,res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xainfo,xbinfo,ia,ib,xdim,densetype,resmode,mlid);

            break;
        }

        case 801:
        case 811:
        {
            if ( ia < 0 ) { setInnerWildpa(&xa,&xainfo); ia = -1; }
            if ( ib < 0 ) { setInnerWildpb(&xb,&xbinfo); ib = -2; }

            res = 0.0;

            if ( NLB()+NUB()+NF() )
            {
                switch ( resmode & 0x7f )
                {
                    case 0:  case 1:  case 2:  case 3:
                    case 4:  case 5:  case 6:  case 7:
                    case 8:  case 9:  case 10: case 11:
                    case 12: case 13: case 14: case 15:
                    {
                        break;
                    }

                    case 16: case 32: case 48:
                    {
                        // Can't do this as additional vectors <x,...> now present, does not fit with xygrad, xnormgrad return

                        NiceThrow("K2xfer 801 not supported.");

                        break;
                    }

                    default:
                    {
                        NiceThrow("K2xfer precursor specified resmode undefined at this level.");

                        break;
                    }
                }

                if ( ( getKernel().isAltDiff() == 0 ) || ( getKernel().isAltDiff() == 1 ) ||  ( getKernel().isAltDiff() == 2 ) || ( getKernel().isAltDiff() == 5 ) )
                {
//#ifdef ENABLE_THREADS
//                    static std::mutex eyelock;
//                    eyelock.lock();
//#endif

//errstream() << "phantomx 0 gentype (" << mlid << ")\n";
                    // This bit of code is speed-critical.  To optimise we cache the
                    // direct products of <xa,X,X> (xa with dataset here, twice)
                    // as we note that, if for example we are calculating g(x) on an
                    // inheritted kernel, at least one of these if practically fixed.
                    // Then we can use standard inner-products, which are much quicker,
                    // to calculate the K4 evaluation.  We use xa for our cache as we
                    // note that g(x) almost always takes the form sum_i alpha_i K(x,xi),
                    // so in the inheritance it is xa that gets repeated for each part
                    // of this sum and xb that keeps on changing.

                    // Cache shared by all MLs

                    Matrix<SparseVector<gentype> > &xvdirectProdsFull = (KxferDatStore).gxvdirectProdsFull;
                    Matrix<double> &xvinnerProdsFull                  = (KxferDatStore).gxvinnerProdsFull;
                    Matrix<double> &aadirectProdsFull                 = (KxferDatStore).gaadirectProdsFull;
                    Vector<int> &prevalphaState                       = (KxferDatStore).gprevalphaState;
                    Vector<int> &prevbalphaState                      = (KxferDatStore).gprevbalphaState;

                    int &prevxvernum = (KxferDatStore).gprevxvernum;
                    int &prevgvernum = (KxferDatStore).gprevgvernum;
                    int &prevNN      = (KxferDatStore).gprevN;
                    int &prevNb      = (KxferDatStore).gprevNb;

                    // Initialise variables

                    if ( !((KxferDatStore).allprevxbvernum.isindpresent(mlid)) )
                    {
                        ((KxferDatStore).allprevxbvernum)("&",mlid) = -1;
                    }

                    // Cached values relating to the caller (mlid)

                    SparseVector<gentype> &xaprev                     = ((KxferDatStore).allxaprev)("&",mlid);                  // previous xa vector (if different then need to recalculate)
                    Matrix<SparseVector<gentype> > &xadirectProdsFull = ((KxferDatStore).allxadirectProdsFull)("&",mlid);       // pre-calculated direct products between xa, x(j) and x(k)
                    Vector<double> &xainnerProdsFull                  = ((KxferDatStore).allxainnerProdsFull)("&",mlid);        // pre-calculated direct products between xa, x(j)
                    Vector<double> &xbinnerProdsFull                  = ((KxferDatStore).allxbinnerProdsFull)("&",mlid);        // pre-calculated direct products between xb, x(j)
                    double &xaxbinnerProd                             = ((KxferDatStore).allxaxbinnerProd)("&",mlid);
                    double &xaxainnerProd                             = ((KxferDatStore).allxaxainnerProd)("&",mlid);
                    double &xbxbinnerProd                             = ((KxferDatStore).allxbxbinnerProd)("&",mlid);
                    SparseVector<gentype> &diagkerns                  = ((KxferDatStore).alldiagkernsgentype)("&",mlid);        // cached evaluations of K(ia,ia), ia >= 0, if done
                    int &prevxbvernum                                 = ((KxferDatStore).allprevxbvernum)("&",mlid);            // version number of x
                    gentype &ipres                                    = ((KxferDatStore).allipres)("&",mlid);                   // scratchpad for ipres

                    int NN = N();
                    int Nb = N();

                    // Have relevant state, can now continue

                    int j,k;

                    int detchange = 0;
                    int alchange  = 0;
                    int xchange   = 0;

//errstream() << "K2xfer: prevxbvernum = " << prevxbvernum << " and mlid = " << mlid << " and xvernum(mlid) = " << xvernum(mlid) << "\n";
                    if ( ( prevxbvernum == -1 ) || ( prevxbvernum != xvernum(mlid) ) )
                    {
                        // Caller's X has changed, so flush relevant cache

                        diagkerns.zero();

                        xadirectProdsFull.prealloc(Nb,Nb);
                        xadirectProdsFull.resize(Nb,Nb);

                        xainnerProdsFull.prealloc(Nb);
                        xainnerProdsFull.resize(Nb);

                        xbinnerProdsFull.prealloc(Nb);
                        xbinnerProdsFull.resize(Nb);

                        prevbalphaState = 0;

                        prevxbvernum = xvernum(mlid);
//errstream() << "K2xfer: Caller X reset.\n";
                    }

                    if ( ( ( prevxvernum == -1 ) && ( prevxvernum != xvernum() ) ) || ( ( prevgvernum == -1 ) && ( prevgvernum != gvernum() ) ) )
                    {
                        // First call, so everything needs to be calculated

                        prevNN = NN;
                        prevNb = Nb;

                        xvdirectProdsFull.prealloc(NN,NN);
                        xvdirectProdsFull.resize(NN,NN);

                        xvinnerProdsFull.prealloc(NN,NN);
                        xvinnerProdsFull.resize(NN,NN);

                        xadirectProdsFull.prealloc(Nb,Nb);
                        xadirectProdsFull.resize(Nb,Nb);

                        xainnerProdsFull.prealloc(Nb);
                        xainnerProdsFull.resize(Nb);

                        xbinnerProdsFull.prealloc(Nb);
                        xbinnerProdsFull.resize(Nb);

                        aadirectProdsFull.prealloc(NN,NN);
                        aadirectProdsFull.resize(NN,NN);

                        prevalphaState.prealloc(NN);
                        prevalphaState.resize(NN);
                        prevalphaState = 0;

                        prevbalphaState.prealloc(Nb);
                        prevbalphaState.resize(Nb);
                        prevbalphaState = 0;

                        prevxvernum = xvernum();
                        prevgvernum = gvernum();

                        diagkerns.zero();

                        detchange = 1;
                        alchange  = 1;
//errstream() << "K2xfer: First call setup.\n";
                    }

                    else if ( prevxvernum != xvernum() )
                    {
                        // X has changed, so everything needs to be calculated

                        prevNN = NN;
                        prevNb = Nb;

                        xvdirectProdsFull.prealloc(NN,NN);
                        xvdirectProdsFull.resize(NN,NN);

                        xvinnerProdsFull.prealloc(NN,NN);
                        xvinnerProdsFull.resize(NN,NN);

                        xadirectProdsFull.prealloc(Nb,Nb);
                        xadirectProdsFull.resize(Nb,Nb);

                        xainnerProdsFull.prealloc(Nb);
                        xainnerProdsFull.resize(Nb);

                        xbinnerProdsFull.prealloc(Nb);
                        xbinnerProdsFull.resize(Nb);

                        aadirectProdsFull.prealloc(NN,NN);
                        aadirectProdsFull.resize(NN,NN);

                        prevalphaState.prealloc(NN);
                        prevalphaState.resize(NN);
                        prevalphaState = 0;

                        prevbalphaState.prealloc(Nb);
                        prevbalphaState.resize(Nb);
                        prevbalphaState = 0;

                        prevxvernum = xvernum();
                        prevgvernum = gvernum();

                        diagkerns.zero();

                        detchange = 1;
                        alchange  = 1;
//errstream() << "K2xfer: Call to modified X.\n";
                    }

                    else if ( prevgvernum != gvernum() )
                    {
                        // alpha has changed but not X, so alphaState is still the same size, but we need to recalculate some bits

                        prevgvernum = gvernum();

                        // IMPORTANT NOTE: we need to reset *all* diagkerns!
                        //diagkerns.zero();

                        for ( j = 0 ; j < (KxferDatStore).alldiagkernsgentype.indsize() ; ++j )
                        {
                            ((KxferDatStore).alldiagkernsgentype.direref(j)).zero();
                        }

                        detchange = 1;
                        alchange  = 1;
//errstream() << "K2xfer: Call to modified alpha.\n";
                    }

                    xchange = 1;

                    //if ( xa == xaprev ) // The following is a marginal speedup for var() calculation in gpr, noting
                    // that (a) assignment copies vecID and (b) most often it will be called b.N times for each vector
                    //if ( ( xa.vecID() == xaprev.vecID() ) || ( ( xa.size() < 10 ) && ( xa == xaprev ) ) )
                    if ( xa.vecID() == xaprev.vecID() )
                    {
                        // xa has not changed

                        xchange = 0;
//errstream() << "K2xfer: New x.\n";
                    }

                    if ( detchange )
                    {
//errstream() << "K2xfer: Setup X otimes X product cache\n";
                        // X changed or first call

                        for ( j = 0 ; j < prevNN ; ++j )
                        {
                            if ( alphaState()(j) && !prevalphaState(j) )
                            {
                                // compute direct product xa,x(j) on outer loop to avoid repetition

                                for ( k = 0 ; k <= j ; ++k )
                                {
                                    if ( alphaState()(k) && !prevalphaState(k) )
                                    {
                                        if ( ( getKernel().isAltDiff() == 0 ) || ( getKernel().isAltDiff() == 1 ) )
                                        {
                                            xvdirectProdsFull("&",j,k) =  x(j);
                                            xvdirectProdsFull("&",j,k) *= x(k);

                                            xvdirectProdsFull("&",j,k).makealtcontent();
                                        }

                                        else
                                        {
                                            innerProductAssumeReal(xvinnerProdsFull("&",j,k),x(j),x(k));
                                            xvinnerProdsFull("&",k,j) = xvinnerProdsFull(j,k);
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if ( alchange )
                    {
//errstream() << "K2xfer: Setup alpha otimes alpha product cache\n";
                        // alpha changed or first call

                        for ( j = 0 ; j < prevNN ; ++j )
                        {
                            for ( k = 0 ; k <= j ; ++k )
                            {
                                if ( isUnderlyingScalar() )
                                {
                                    aadirectProdsFull("&",j,k) = alphaR()(j)*alphaR()(k);
                                }

                                else if ( isUnderlyingVector() )
                                {
                                    aadirectProdsFull("&",j,k) = twoProduct(dummyr,alphaV()(j),alphaV()(k));
                                }

                                else
                                {
                                    aadirectProdsFull("&",j,k) = real(alphaA()(j)*conj(alphaA()(k)));
                                    //aadirectProdsFull("&",j,k) = (double) real(innerProduct(dummy,alpha()(j),alpha()(k)));
                                }
                            }
                        }
                    }

                    if ( detchange )
                    {
                        for ( j = 0 ; j < prevNN ; ++j )
                        {
                            prevalphaState("&",j) |= alphaState()(j);
                        }
                    }

                    if ( prevNN != NN )
                    {
//errstream() << "K2xfer: Extend X otimes X product cache\n";
                        // extend cache if required.

                        xvdirectProdsFull.prealloc(NN,NN);
                        xvdirectProdsFull.resize(NN,NN);

                        xvinnerProdsFull.prealloc(NN,NN);
                        xvinnerProdsFull.resize(NN,NN);

                        aadirectProdsFull.prealloc(NN,NN);
                        aadirectProdsFull.resize(NN,NN);

                        for ( j = prevNN ; j < NN ; ++j )
                        {
                            for ( k = 0 ; k <= j ; ++k )
                            {
                                if ( ( getKernel().isAltDiff() == 0 ) || ( getKernel().isAltDiff() == 1 ) )
                                {
                                    xvdirectProdsFull("&",j,k) =  x(j);
                                    xvdirectProdsFull("&",j,k) *= x(k);

                                    xvdirectProdsFull("&",j,k).makealtcontent();
                                }

                                else
                                {
                                    innerProductAssumeReal(xvinnerProdsFull("&",j,k),x(j),x(k));
                                    xvinnerProdsFull("&",k,j) = xvinnerProdsFull(j,k);
                                }

                                if ( isUnderlyingScalar() )
                                {
                                    aadirectProdsFull("&",j,k) = alphaR()(j)*alphaR()(k);
                                }

                                else if ( isUnderlyingVector() )
                                {
                                    aadirectProdsFull("&",j,k) = twoProduct(dummyr,alphaV()(j),alphaV()(k));
                                }

                                else
                                {
                                    aadirectProdsFull("&",j,k) = real(alphaA()(j)*conj(alphaA()(k)));
                                    //aadirectProdsFull("&",j,k) = (double) real(innerProduct(dummy,alpha()(j),alpha()(k)));
                                }
                            }
                        }

                        retVector<int> tmpva;
                        retVector<int> tmpvb;

                        prevalphaState.prealloc(NN);
                        prevalphaState.resize(NN);
                        prevalphaState("&",prevNN,1,NN-1,tmpva) = alphaState()(prevNN,1,NN-1,tmpvb);

                        prevNN = NN;
                    }

                    if ( ( iacall == ibcall ) && ( iacall >= 0 ) && diagkerns.isindpresent(iacall) )
                    {
                        // This is K(x(ia),x(ib)), and we have calculated it before, so just use cached version

                        res = diagkerns(iacall);

                        NiceAssert( !testisvnan(res) );
                        NiceAssert( !testisinf(res) );
                    }

                    else if ( ( getKernel().isAltDiff() == 0 ) || ( getKernel().isAltDiff() == 1 ) )
                    {
                        if ( ( iacall < 0 ) || ( iacall != ibcall ) )
                        {
                            // Subtlety: non-diagonal kernel evaluation runs through kcache, so the *entire row* is calculated in a single hit during
                            //           a resetKernel call.  Hence using this should speed things up substantially!
                            // This is K(x,x(ib)), which is called for all x(ib) in g(x) calculation, so cache xadirectProds on first hit

                            int upastate = 0;
//int actdone = 0;

                            //if ( xchange || detchange ) - commented, as prevbalphaState is also a factor, and the comparision is done in the first loop anyhow
                            {
                                // Expensive, but only required *once* for each g(xa) evaluation!

                                for ( j = 0 ; j < prevNb ; ++j )
                                {
                                    upastate |= ( alphaState()(j) != prevbalphaState(j) );

                                    if ( alphaState()(j) && ( xchange || !prevbalphaState(j) ) )
                                    {
                                        // compute direct product xa,x(j) on outer loop to avoid repetition

                                        for ( k = 0 ; k <= j ; ++k )
                                        {
                                           if ( alphaState()(k) && ( xchange || !prevbalphaState(k) ) )
                                            {
                                                xadirectProdsFull("&",j,k)  = xa;
                                                xadirectProdsFull("&",j,k) *= xvdirectProdsFull(j,k);

                                                xadirectProdsFull("&",j,k).makealtcontent();
//actdone = 1;
                                            }
                                        }
                                    }
                                }

//if ( actdone ) { errstream() << "!"; }
                                if ( upastate )
                                {
                                    retVector<int> tmpva;

                                    prevbalphaState = alphaState()(0,1,prevNb-1,tmpva);
                                }

                                if ( xchange )
                                {
                                    xaprev = xa; // this will make the vecIDs the same, so the next comparison may be short-circuited
                                }
                            }

                            if ( prevNb != Nb )
                            {
//errstream() << "@";
                                // extend cache if required.

                                xadirectProdsFull.prealloc(Nb,Nb);
                                xadirectProdsFull.resize(Nb,Nb);

                                xainnerProdsFull.prealloc(Nb);
                                xainnerProdsFull.resize(Nb);

                                xbinnerProdsFull.prealloc(Nb);
                                xbinnerProdsFull.resize(Nb);

                                for ( j = prevNb ; j < Nb ; ++j )
                                {
                                    for ( k = 0 ; k <= j ; ++k )
                                    {
                                        xadirectProdsFull("&",j,k)  = xa;
                                        xadirectProdsFull("&",j,k) *= xvdirectProdsFull(j,k);

                                        xadirectProdsFull("&",j,k).makealtcontent();
                                    }
                                }

                                retVector<int> tmpva;
                                retVector<int> tmpvb;

                                prevbalphaState.prealloc(Nb);
                                prevbalphaState.resize(Nb);
                                prevbalphaState("&",prevNb,1,Nb-1,tmpva) = alphaState()(prevNb,1,Nb-1,tmpvb);

                                prevNb = Nb;
                            }

                            //gentype ipres(0.0);
                            ipres.force_double() = 0.0;
                            gentype kres(0.0);
                            const gentype *pxyprod[2];

                            pxyprod[0] = &ipres; // we will use this to pass the pre-calculated 4-product in directly
                            pxyprod[1] = nullptr;

                            for ( j = 0 ; j < Nb ; ++j )
                            {
                                // Not cheap, but can't see a way to avoid this

                                if ( alphaState()(j) )
                                {
                                    for ( k = 0 ; k <= j ; ++k )
                                    {
                                        if ( alphaState()(k) )
                                        {
                                            twoProduct(ipres,xadirectProdsFull(j,k),xb); // Assume no conjugation for speed here

                                            NiceAssert( !testisvnan(ipres) );
                                            NiceAssert( !testisinf(ipres) );

                                            // We have xyprod, and other norms will be inferred from xinfo.  Note that 
                                            // nullptrs are filled at ML_Base level as required, and relevant norms are 
                                            // always cached in xinfo (so only calc once).
                                            //
                                            // We can safely assume that xainfo and xbinfo exist.  xbinfo is probably the
                                            // xinfo() from the calling class, and xainfo is created by g(x).

                                            K4(kres,ia,ib,j,k,pxyprod,&xa,&xb,nullptr,nullptr,&xainfo,&xbinfo,nullptr,nullptr,resmode);

                                            if ( j != k )
                                            {
                                                kres *= 2.0; // there are two identical instances by symmetry, real by assumption and for speed
                                            }

                                            // Scale by alpha and add to result

                                            kres *= aadirectProdsFull(j,k);

                                            NiceAssert( !testisvnan(kres) );
                                            NiceAssert( !testisinf(kres) );

                                            res += kres;
                                        }
                                    }
                                }
                            }
                        }

                        else
                        {
//errstream() << "*(" << iacall << "," << ibcall << ")*";
                            // This calculation is one-hit, so for speed don't modify caches etc

                            // Note that iacall == ibcall >= 0

                            //gentype ipres(0.0);
                            ipres.force_double() = 0.0;
                            gentype kres(0.0);
                            const gentype *pxyprod[2];

                            pxyprod[0] = &ipres; // we will use this to pass the pre-calculated 4-product in directly
                            pxyprod[1] = nullptr;

                            SparseVector<gentype> xout(xa);

                            xout *= xb;

                            for ( j = 0 ; j < Nb ; ++j )
                            {
                                if ( alphaState()(j) )
                                {
                                    for ( k = 0 ; k <= j ; ++k )
                                    {
                                        if ( alphaState()(k) )
                                        {
                                            twoProduct(ipres,xvdirectProdsFull(j,k),xout); // Again we assume commutativity etc

                                            NiceAssert( !testisvnan(ipres) );
                                            NiceAssert( !testisinf(ipres) );

                                            K4(kres,ia,ib,j,k,pxyprod,&xa,&xb,nullptr,nullptr,&xainfo,&xbinfo,nullptr,nullptr,resmode);

                                            if ( j != k )
                                            {
                                                kres *= 2; // there are two identical instances by symmetry, real by assumption and for speed
                                            }

                                            // Scale by alpha and add to result

                                            kres *= aadirectProdsFull(j,k);

                                            NiceAssert( !testisvnan(kres) );
                                            NiceAssert( !testisinf(kres) );

                                            res += kres;
                                        }
                                    }
                                }
                            }

                            if ( ( iacall == ibcall ) && ( iacall >= 0 ) )
                            {
                                diagkerns("&",iacall) = res;
                            }

                            NiceAssert( !testisvnan(res) );
                            NiceAssert( !testisinf(res) );
                        }
                    }

                    else
                    {
                        int upastate = 0;
                        gentype kres;

                        for ( j = 0 ; j < prevNb ; ++j )
                        {
                            upastate |= ( alphaState()(j) != prevbalphaState(j) );

                            if ( ( getKernel().isAltDiff() == 2 ) && alphaState()(j) && ( xchange || !prevbalphaState(j) ) )
                            {
                                innerProductAssumeReal(xainnerProdsFull("&",j),xa,x(j));
                                innerProductAssumeReal(xbinnerProdsFull("&",j),xb,x(j));
                            }
                        }

                        if ( upastate )
                        {
                            retVector<int> tmpva;

                            prevbalphaState = alphaState()(0,1,prevNb-1,tmpva);
                        }

                        if ( xchange )
                        {
                            innerProductAssumeReal(xaxbinnerProd,xa,xb);

                            xaxainnerProd = getKernel().getmnorm(xainfo,xa,2,isXConsistent(),1);
                            xbxbinnerProd = getKernel().getmnorm(xbinfo,xb,2,isXConsistent(),1);
                        }

                        if ( prevNb != Nb )
                        {
                            xadirectProdsFull.prealloc(Nb,Nb);
                            xadirectProdsFull.resize(Nb,Nb);

                            xainnerProdsFull.prealloc(Nb);
                            xainnerProdsFull.resize(Nb);

                            xbinnerProdsFull.prealloc(Nb);
                            xbinnerProdsFull.resize(Nb);

                            if ( getKernel().isAltDiff() == 2 )
                            {
                                for ( j = prevNb ; j < Nb ; ++j )
                                {
                                    innerProductAssumeReal(xainnerProdsFull("&",j),xa,x(j));
                                    innerProductAssumeReal(xbinnerProdsFull("&",j),xb,x(j));
                                }
                            }

                            retVector<int> tmpva;
                            retVector<int> tmpvb;

                            prevbalphaState.prealloc(Nb);
                            prevbalphaState.resize(Nb);
                            prevbalphaState("&",prevNb,1,Nb-1,tmpva) = alphaState()(prevNb,1,Nb-1,tmpvb);

                            prevNb = Nb;
                        }

                        (xyvalid) = mlid; // xymat will now be used by K4 call, avoiding the requirement for any further inner products

                        for ( j = 0 ; j < Nb ; ++j )
                        {
                            if ( alphaState()(j) )
                            {
                                for ( k = 0 ; k <= j ; ++k )
                                {
                                    if ( alphaState()(k) )
                                    {
                                        K4(kres,ia,ib,j,k,nullptr,&xa,&xb,nullptr,nullptr,&xainfo,&xbinfo,nullptr,nullptr,resmode);

                                        if ( j != k )
                                        {
                                            kres *= 2.0; // there are two identical instances by symmetry, real by assumption and for speed
                                        }

                                        // Scale by alpha and add to result

                                        kres *= aadirectProdsFull(j,k);

                                        NiceAssert( !testisvnan(kres) );
                                        NiceAssert( !testisinf(kres) );

                                        res += kres;
                                    }
                                }
                            }
                        }

                        (xyvalid) = 0;

                        if ( ( iacall == ibcall ) && ( iacall >= 0 ) )
                        {
                            diagkerns("&",iacall) = res;
                        }
                    }

//errstream() << "phantomx 0 gentype res = " << res << "\n";
//#ifdef ENABLE_THREADS
//                    eyelock.unlock();
//#endif
                }

                else
                {
                    gentype Kij;

                    int j,k;
                    int docheat = ( ( ( resmode & 0x7F ) >= 0 ) && ( ( resmode & 0x7F ) <= 15 ) ) ? 1 : 0;

                    for ( j = 0 ; j < N() ; ++j )
                    {
                        if ( alphaState()(j) )
                        {
                            for ( k = 0 ; k < ( docheat ? j+1 : N() ) ; ++k )
                            {
                                if ( alphaState()(k) )
                                {
                                    K4(Kij,ia,ib,j,k,nullptr,&xa,&xb,nullptr,nullptr,&xainfo,&xbinfo,nullptr,nullptr,resmode);

                                    if ( isUnderlyingScalar() )
                                    {
                                        Kij *= alphaR()(j)*alphaR()(k);
                                    }

                                    else if ( isUnderlyingVector() )
                                    {
                                        Kij *= twoProduct(dummyr,alphaV()(j),alphaV()(k));
                                    }

                                    else
                                    {
                                        Kij *= (double) real(innerProduct(dummy,alpha()(j),alpha()(k)));
                                    }

                                    if ( docheat )
                                    {
                                        if ( k < j )
                                        {
                                            res += Kij;
                                            res += Kij;
                                        }

                                        else if ( k == j )
                                        {
                                            res += Kij;
                                        }
                                    }

                                    else
                                    {
                                        res += Kij;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if ( ( ia < 0 ) || ( ib < 0 ) ) { resetInnerWildp(); }

            break;
        }

        case 802:
        case 812:
        {
            if ( ia < 0 ) { setInnerWildpa(&xa,&xainfo); ia = -1; }
            if ( ib < 0 ) { setInnerWildpb(&xb,&xbinfo); ib = -2; }

            NiceAssert( !(resmode & 0x80 ) );

            res = 0.0;

            if ( NLB()+NUB()+NF() )
            {
                gentype Kija,Kijb;

                int j,k;

                for ( j = 0 ; j < N() ; ++j )
                {
                    if ( alphaState()(j) )
                    {
                        for ( k = 0 ; k < N() ; ++k )
                        {
                            if ( alphaState()(k) )
                            {
                                switch ( resmode & 0x7f )
                                {
                                    case 0:  case 1:  case 2:  case 3:
                                    case 4:  case 5:  case 6:  case 7:
                                    case 8:  case 9:  case 10: case 11:
                                    case 12: case 13: case 14: case 15:
                                    {
                                        K2(Kija,ia,j,nullptr,&xa,nullptr,&xainfo,nullptr,resmode);
                                        K2(Kijb,ib,k,nullptr,&xb,nullptr,&xbinfo,nullptr,resmode);

                                        Kija *= Kijb;

                                        break;
                                    }

                                    case 16: case 32: case 48:
                                    {
                                        // Can't do this as additional vectors <x,...> now present, does not fit with xygrad, xnormgrad return

                                        NiceThrow("K2xfer 802 not supported.");

                                        break;
                                    }

                                    case 64:
                                    {
                                        NiceThrow("K2xfer precursor second order derivatives not yet implemented.");

                                        break;
                                    }

                                    default:
                                    {
                                        NiceThrow("K2xfer precursor specified resmode undefined at this level.");

                                        break;
                                    }
                                }

                                if ( isUnderlyingScalar() )
                                {
                                    Kija *= alphaR()(j)*alphaR()(k);
                                }

                                else if ( isUnderlyingVector() )
                                {
                                    Kija *= twoProduct(dummyr,alphaV()(j),alphaV()(k));
                                }

                                else
                                {
                                    Kija *= (double) real(innerProduct(dummy,alpha()(j),alpha()(k)));
                                }

                                res += Kija;
                                res += Kija;
                            }
                        }
                    }
                }
            }

            if ( ( ia < 0 ) || ( ib < 0 ) ) { resetInnerWildp(); }

            break;
        }

        case 807:
        case 817:
        {
            gentype Kintemp;
            double dummyr;

            K1(Kintemp,xa,&xainfo);

            res = 0.0;

            if ( NLB()+NUB()+NF() )
            {
                gentype Kij;
                gentype temp;

                int j,k;

                for ( j = 0 ; j < N() ; ++j )
                {
                    if ( alphaState()(j) )
                    {
                        for ( k = 0 ; k < N() ; ++k )
                        {
                            if ( alphaState()(k) )
                            {
                                K2(Kij,j,k);

                                temp =  Kintemp;
                                temp *= Kij;
                                temp -= 1.0;
                                temp = -1.0/temp;

                                if ( isUnderlyingScalar() )
                                {
                                    temp *= alphaR()(j)*alphaR()(k);
                                }

                                else if ( isUnderlyingVector() )
                                {
                                    temp *= innerProduct(dummyr,alphaV()(j),alphaV()(k));
                                }

                                else
                                {
                                    temp *= (double) real(alpha()(j)*alpha()(k));
                                }

                                res += temp;
                            }
                        }
                    }
                }
            }

            break;
        }

        default:
        {
            ML_Base::K2xfer(dxyprod,ddiffis,res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xainfo,xbinfo,ia,ib,xdim,densetype,resmode,mlid);

            break;
        }
    }

    NiceAssert( !testisvnan(res) );
    NiceAssert( !testisinf(res) );

    return;
}

void SVM_Generic::K2xfer(double &dxyprod, double &ddiffis, double &res, int &minmaxind, int typeis,
                         double xyprod, double yxprod, double diffis,
                         const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                         const vecInfo &xainfo, const vecInfo &xbinfo,
                         int ia, int ib,
                         int xdim, int densetype, int resmode, int mlid) const
{
    NiceAssert( !densetype );

    if ( ( resmode & 0x7f ) >= 16 )
    {
        gentype tempa,tempb,tempc;

        gentype gxyprod(xyprod);
        gentype gyxprod(yxprod);
        gentype gdiffis(diffis);

        K2xfer(tempa,tempb,tempc,minmaxind,typeis,gxyprod,gyxprod,gdiffis,xa,xb,xainfo,xbinfo,ia,ib,xdim,densetype,resmode,mlid);

        dxyprod = (double) tempa;
        ddiffis = (double) tempb;

        res = (double) tempc;

        NiceAssert( !testisvnan(res) );
        NiceAssert( !testisinf(res) );

        return;
    }

    //gentype dummy;
    double dummyr = 0.0;

    if ( ( ia >= 0 ) && ( ib < 0 ) )
    {
        // To satisfy later assumptions

        K2xfer(dxyprod,ddiffis,res,minmaxind,typeis,xyprod,yxprod,diffis,xb,xa,xbinfo,xainfo,ib,ia,xdim,densetype,resmode,mlid);

        NiceAssert( !testisvnan(res) );
        NiceAssert( !testisinf(res) );

        return;
    }

    int iacall = ia;
    int ibcall = ib;

    ia = (typeis-(100*(typeis/100)))/10 ? ia : -42;
    ib = (typeis-(100*(typeis/100)))/10 ? ib : -43;

    switch ( typeis )
    {
        case 800:
        case 810:
        {
            if ( !resmode && isUnderlyingScalar() && ( ia >= 0 ) && ( ib >= 0 ) )
            {
                res = ( ia == ib ) ? kerndiag()(ia) : Gp()(ia,ib);
            }

//errstream() << "phantomx 1\n";
            ML_Base::K2xfer(dxyprod,ddiffis,res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xainfo,xbinfo,ia,ib,xdim,densetype,resmode,mlid);

            break;
        }

        case 801:
        case 811:
        {
//errstream() << "phantomx 0: " << getKernel().isAltDiff() << "," << NLB()+NUB()+NF() << "\n";
            if ( ia < 0 ) { setInnerWildpa(&xa,&xainfo); ia = -1; }
            if ( ib < 0 ) { setInnerWildpb(&xb,&xbinfo); ib = -2; }

            res = 0.0;

            if ( NLB()+NUB()+NF() )
            {
//errstream() << "phantomx 0b: " << getKernel().isAltDiff() << "\n";
                switch ( resmode & 0x7f )
                {
                    case 0:  case 1:  case 2:  case 3:
                    case 4:  case 5:  case 6:  case 7:
                    case 8:  case 9:  case 10: case 11:
                    case 12: case 13: case 14: case 15:
                    {
//errstream() << "phantomx 0c: " << getKernel().isAltDiff() << "\n";
                        break;
                    }

                    case 16: case 32: case 48:
                    {
//errstream() << "phantomx 0d: " << getKernel().isAltDiff() << "\n";
                        // Can't do this as additional vectors <x,...> now present, does not fit with xygrad, xnormgrad return

                        NiceThrow("K2xfer 801 not supported.");

                        break;
                    }

                    default:
                    {
//errstream() << "phantomx 0f: " << getKernel().isAltDiff() << "\n";
                        NiceThrow("K2xfer precursor specified resmode undefined at this level.");

                        break;
                    }
                }
//errstream() << "phantomx 0h: " << getKernel().isAltDiff() << "\n";

                if ( ( getKernel().isAltDiff() == 0 ) || ( getKernel().isAltDiff() == 1 ) ||  ( getKernel().isAltDiff() == 2 ) || ( getKernel().isAltDiff() == 5 ) )
                {
//errstream() << "phantomx 1: " << getKernel().isAltDiff() << "\n";
//#ifdef ENABLE_THREADS
//                    static std::mutex eyelock;
//                    eyelock.lock();
//#endif

                    // duplicate of gentype, with gentype -> double

//errstream() << "phantomx K2xfer 0 double (" << mlid << ") = ";
                    // This bit of code is speed-critical.  To optimise we cache the
                    // direct products of <xa,X,X> (xa with dataset here, twice)
                    // as we note that, if for example we are calculating g(x) on an
                    // inheritted kernel, at least one of these if practically fixed.
                    // Then we can use standard inner-products, which are much quicker,
                    // to calculate the K4 evaluation.  We use xa for our cache as we
                    // note that g(x) almost always takes the form sum_i alpha_i K(x,xi),
                    // so in the inheritance it is xa that gets repeated for each part
                    // of this sum and xb that keeps on changing.

                    // Cache shared by all MLs

                    Matrix<SparseVector<gentype> > &xvdirectProdsFull = (KxferDatStore).gxvdirectProdsFull;
                    Matrix<double> &xvinnerProdsFull                  = (KxferDatStore).gxvinnerProdsFull;
                    Matrix<double> &aadirectProdsFull                 = (KxferDatStore).gaadirectProdsFull;
                    Vector<int> &prevalphaState                       = (KxferDatStore).gprevalphaState;
                    Vector<int> &prevbalphaState                      = (KxferDatStore).gprevbalphaState;

                    int &prevxvernum = (KxferDatStore).gprevxvernum;
                    int &prevgvernum = (KxferDatStore).gprevgvernum;
                    int &prevNN      = (KxferDatStore).gprevN;
                    int &prevNb      = (KxferDatStore).gprevNb;

                    // Initialise variables

                    if ( !((KxferDatStore).allprevxbvernum.isindpresent(mlid)) )
                    {
                        ((KxferDatStore).allprevxbvernum)("&",mlid) = -1;
                    }

                    // Cached values relating to the caller (mlid)

                    SparseVector<gentype> &xaprev                     = ((KxferDatStore).allxaprev)("&",mlid);                 // previous xa vector (if different then need to recalculate)
                    Matrix<SparseVector<gentype> > &xadirectProdsFull = ((KxferDatStore).allxadirectProdsFull)("&",mlid);      // pre-calculated direct products between xa, x(j) and x(k)
                    Vector<double> &xainnerProdsFull                  = ((KxferDatStore).allxainnerProdsFull)("&",mlid);       // pre-calculated direct products between xa, x(j)
                    Vector<double> &xbinnerProdsFull                  = ((KxferDatStore).allxbinnerProdsFull)("&",mlid);       // pre-calculated direct products between xb, x(j)
                    double &xaxbinnerProd                             = ((KxferDatStore).allxaxbinnerProd)("&",mlid);
                    double &xaxainnerProd                             = ((KxferDatStore).allxaxainnerProd)("&",mlid);
                    double &xbxbinnerProd                             = ((KxferDatStore).allxbxbinnerProd)("&",mlid);
                    SparseVector<double> &diagkerns                   = ((KxferDatStore).alldiagkernsdouble)("&",mlid);        // cached evaluations of K(ia,ia), ia >= 0, if done
                    int &prevxbvernum                                 = ((KxferDatStore).allprevxbvernum)("&",mlid);           // version number of x
                    gentype &ipres                                    = ((KxferDatStore).allipres)("&",mlid);                  // scratchpad for ipres

                    int NN = N();
                    int Nb = N();

                    // Setup, let's go

                    int j,k;

                    int detchange = 0;
                    int alchange  = 0;
                    int xchange   = 0;
//errstream() << "phantomx K2xfer 4\n";

                    if ( ( prevxbvernum == -1 ) || ( prevxbvernum != xvernum(mlid) ) )
                    {
//errstream() << "phantomx K2xfer 5\n";
                        // Caller's X has changed, so flush relevant cache

                        diagkerns.zero();

                        xadirectProdsFull.prealloc(Nb,Nb);
                        xadirectProdsFull.resize(Nb,Nb);

                        xainnerProdsFull.prealloc(Nb);
                        xainnerProdsFull.resize(Nb);

                        xbinnerProdsFull.prealloc(Nb);
                        xbinnerProdsFull.resize(Nb);

                        prevbalphaState = 0;

                        prevxbvernum = xvernum(mlid);
//errstream() << "K2xfer: Caller X reset.\n";
                    }

                    if ( ( ( prevxvernum == -1 ) && ( prevxvernum != xvernum() ) ) || ( ( prevgvernum == -1 ) && ( prevgvernum != gvernum() ) ) )
                    {
//errstream() << "phantomx K2xfer 6\n";
                        // First call, so everything needs to be calculated

                        prevNN = NN;
                        prevNb = Nb;

                        xvdirectProdsFull.prealloc(NN,NN);
                        xvdirectProdsFull.resize(NN,NN);

                        xvinnerProdsFull.prealloc(NN,NN);
                        xvinnerProdsFull.resize(NN,NN);

                        xadirectProdsFull.prealloc(Nb,Nb);
                        xadirectProdsFull.resize(Nb,Nb);

                        xainnerProdsFull.prealloc(Nb);
                        xainnerProdsFull.resize(Nb);

                        xbinnerProdsFull.prealloc(Nb);
                        xbinnerProdsFull.resize(Nb);

                        aadirectProdsFull.prealloc(NN,NN);
                        aadirectProdsFull.resize(NN,NN);

                        prevalphaState.prealloc(NN);
                        prevalphaState.resize(NN);
                        prevalphaState = 0;

                        prevbalphaState.prealloc(Nb);
                        prevbalphaState.resize(Nb);
                        prevbalphaState = 0;

                        prevxvernum = xvernum();
                        prevgvernum = gvernum();

                        diagkerns.zero();

                        detchange = 1;
                        alchange  = 1;
//errstream() << "K2xfer: First call setup.\n";
                    }

                    else if ( prevxvernum != xvernum() )
                    {
//errstream() << "phantomx K2xfer 7\n";
                        // X has changed, so everything needs to be calculated

                        prevNN = NN;
                        prevNb = Nb;

                        xvdirectProdsFull.prealloc(NN,NN);
                        xvdirectProdsFull.resize(NN,NN);

                        xvinnerProdsFull.prealloc(NN,NN);
                        xvinnerProdsFull.resize(NN,NN);

                        xadirectProdsFull.prealloc(Nb,Nb);
                        xadirectProdsFull.resize(Nb,Nb);

                        xainnerProdsFull.prealloc(Nb);
                        xainnerProdsFull.resize(Nb);

                        xbinnerProdsFull.prealloc(Nb);
                        xbinnerProdsFull.resize(Nb);

                        aadirectProdsFull.prealloc(NN,NN);
                        aadirectProdsFull.resize(NN,NN);

                        prevalphaState.prealloc(NN);
                        prevalphaState.resize(NN);
                        prevalphaState = 0;

                        prevbalphaState.prealloc(Nb);
                        prevbalphaState.resize(Nb);
                        prevbalphaState = 0;

                        prevxvernum = xvernum();
                        prevgvernum = gvernum();

                        diagkerns.zero();

                        detchange = 1;
                        alchange  = 1;
//errstream() << "K2xfer: Call to modified X.\n";
                    }

                    else if ( prevgvernum != gvernum() )
                    {
//errstream() << "phantomx K2xfer 8\n";
                        // alpha has changed but not X, so alphaState is still the same size, but we need to recalculate some bits

                        prevgvernum = gvernum();

                        // IMPORTANT NOTE: we need to reset *all* diagkerns!
                        //diagkerns.zero();

                        for ( j = 0 ; j < (KxferDatStore).alldiagkernsdouble.indsize() ; ++j )
                        {
                            ((KxferDatStore).alldiagkernsdouble.direref(j)).zero();
                        }

                        detchange = 1;
                        alchange  = 1;
//errstream() << "K2xfer: Call to modified alpha.\n";
                    }

                    xchange = 1;

                    //if ( xa == xaprev ) // The following is a marginal speedup for var() calculation in gpr, noting
                    // that (a) assignment copies vecID and (b) most often it will be called b.N times for each vector
                    //if ( ( xa.vecID() == xaprev.vecID() ) || ( ( xa.size() < 10 ) && ( xa == xaprev ) ) )
                    if ( xa.vecID() == xaprev.vecID() )
                    {
                        // xa has not changed

                        xchange = 0;
//errstream() << "K2xfer: New x.\n";
                    }

//errstream() << "phantomxyz K2xfer x(0) = " << x(0) << "\n";
                    if ( detchange )
                    {
//errstream() << "phantomx K2xfer 10\n";
//errstream() << "K2xfer: Setup X otimes X product cache\n";
                        // X changed or first call

                        for ( j = 0 ; j < prevNN ; ++j )
                        {
                            if ( alphaState()(j) && !prevalphaState(j) )
                            {
                                // compute direct product xa,x(j) on outer loop to avoid repetition

                                for ( k = 0 ; k <= j ; ++k )
                                {
                                    if ( alphaState()(k) && !prevalphaState(k) )
                                    {
//errstream() << "!1!";
                                        if ( ( getKernel().isAltDiff() == 0 ) || ( getKernel().isAltDiff() == 1 ) )
                                        {
                                            xvdirectProdsFull("&",j,k) =  x(j);
                                            xvdirectProdsFull("&",j,k) *= x(k);

                                            xvdirectProdsFull("&",j,k).makealtcontent();
                                        }

                                        else
                                        {
                                            innerProductAssumeReal(xvinnerProdsFull("&",j,k),x(j),x(k));
                                            xvinnerProdsFull("&",k,j) = xvinnerProdsFull(j,k);
//errstream() << "phantomxyz K2xfer 1 xvinner(" << j << "," << k << ") = " << xvinnerProdsFull(j,k) << "\n";
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if ( alchange )
                    {
//errstream() << "K2xfer: Setup alpha otimes alpha product cache\n";
                        // alpha changed or first call

                        for ( j = 0 ; j < prevNN ; ++j )
                        {
                            for ( k = 0 ; k <= j ; ++k )
                            {
                                if ( isUnderlyingScalar() )
                                {
                                    aadirectProdsFull("&",j,k) = alphaR()(j)*alphaR()(k);
                                }

                                else if ( isUnderlyingVector() )
                                {
                                    aadirectProdsFull("&",j,k) = twoProduct(dummyr,alphaV()(j),alphaV()(k));
                                }

                                else
                                {
                                    aadirectProdsFull("&",j,k) = real(alphaA()(j)*conj(alphaA()(k)));
                                    //aadirectProdsFull("&",j,k) = (double) real(innerProduct(dummy,alpha()(j),alpha()(k)));
                                }
                            }
                        }
                    }

                    if ( detchange )
                    {
                        for ( j = 0 ; j < prevNN ; ++j )
                        {
                            prevalphaState("&",j) |= alphaState()(j);
                        }
                    }

                    if ( prevNN != NN )
                    {
//errstream() << "phantomx K2xfer 11\n";
//errstream() << "K2xfer: Extend X otimes X product cache\n";
                        // extend cache if required.

                        xvdirectProdsFull.prealloc(NN,NN);
                        xvdirectProdsFull.resize(NN,NN);

                        xvinnerProdsFull.prealloc(NN,NN);
                        xvinnerProdsFull.resize(NN,NN);

                        aadirectProdsFull.prealloc(NN,NN);
                        aadirectProdsFull.resize(NN,NN);

                        for ( j = prevNN ; j < NN ; ++j )
                        {
                            for ( k = 0 ; k <= j ; ++k )
                            {
//errstream() << "!2!";
                                if ( ( getKernel().isAltDiff() == 0 ) || ( getKernel().isAltDiff() == 1 ) )
                                {
                                    xvdirectProdsFull("&",j,k) =  x(j);
                                    xvdirectProdsFull("&",j,k) *= x(k);

                                    xvdirectProdsFull("&",j,k).makealtcontent();
                                }

                                else
                                {
                                    innerProductAssumeReal(xvinnerProdsFull("&",j,k),x(j),x(k));
                                    xvinnerProdsFull("&",k,j) = xvinnerProdsFull(j,k);
//errstream() << "phantomxyz K2xfer 2 xvinner(" << j << "," << k << ") = " << xvinnerProdsFull(j,k) << "\n";
                                }

                                if ( isUnderlyingScalar() )
                                {
                                    aadirectProdsFull("&",j,k) = alphaR()(j)*alphaR()(k);
                                }

                                else if ( isUnderlyingVector() )
                                {
                                    aadirectProdsFull("&",j,k) = twoProduct(dummyr,alphaV()(j),alphaV()(k));
                                }

                                else
                                {
                                    aadirectProdsFull("&",j,k) = real(alphaA()(j)*conj(alphaA()(k)));
                                    //aadirectProdsFull("&",j,k) = (double) real(innerProduct(dummy,alpha()(j),alpha()(k)));
                                }
                            }
                        }

                        retVector<int> tmpva;
                        retVector<int> tmpvb;

                        prevalphaState.resize(NN);
                        prevalphaState("&",prevNN,1,NN-1,tmpva) = alphaState()(prevNN,1,NN-1,tmpvb);

                        prevNN = NN;
                    }

                    if ( ( iacall == ibcall ) && ( iacall >= 0 ) && diagkerns.isindpresent(iacall) )
                    {
//errstream() << "phantomx K2xfer 12\n";
                        // This is K(x(ia),x(ib)), and we have calculated it before, so just use cached version

                        res = diagkerns(iacall);

                        NiceAssert( !testisvnan(res) );
                        NiceAssert( !testisinf(res) );
                    }

                    else if ( ( getKernel().isAltDiff() == 0 ) || ( getKernel().isAltDiff() == 1 ) )
                    {
                        if ( ( iacall < 0 ) || ( iacall != ibcall ) )
                        {
                            // Subtlety: non-diagonal kernel evaluation runs through kcache, so the *entire row* is calculated in a single hit during
                            //           a resetKernel call.  Hence using this should speed things up substantially!
                            // This is K(x,x(ib)), which is called for all x(ib) in g(x) calculation, so cache xadirectProds on first hit

                            int upastate = 0;
//int actdone = 0;

                            //if ( xchange || detchange ) - commented, as prevbalphaState is also a factor, and the comparision is done in the first loop anyhow
                            {
                                // Expensive, but only required *once* for each g(xa) evaluation!

                                for ( j = 0 ; j < prevNb ; ++j )
                                {
                                    upastate |= ( alphaState()(j) != prevbalphaState(j) );

                                    if ( alphaState()(j) && ( xchange || !prevbalphaState(j) ) )
                                    {
                                        // compute direct product xa,x(j) on outer loop to avoid repetition

                                        if ( ( getKernel().isAltDiff() == 0 ) || ( getKernel().isAltDiff() == 1 ) )
                                        {
                                            for ( k = 0 ; k <= j ; ++k )
                                            {
                                                if ( alphaState()(k) && ( xchange || !prevbalphaState(k) ) )
                                                {
                                                    xadirectProdsFull("&",j,k) =  xa;
                                                    xadirectProdsFull("&",j,k) *= xvdirectProdsFull(j,k);

                                                    xadirectProdsFull("&",j,k).makealtcontent();
//actdone = 1;
                                                }
                                            }
                                        }
                                    }
                                }

//if ( actdone ) { errstream() << "!"; }
                                if ( upastate )
                                {
                                    retVector<int> tmpva;

                                    prevbalphaState = alphaState()(0,1,prevNb-1,tmpva);
                                }

                                if ( xchange )
                                {
                                    xaprev = xa; // this will make the vecIDs the same, so the next comparison may be short-circuited
                                }
                            }

                            if ( prevNb != Nb )
                            {
//errstream() << "@ blahto";
                                // extend cache if required.

                                xadirectProdsFull.prealloc(Nb,Nb);
                                xadirectProdsFull.resize(Nb,Nb);

                                xainnerProdsFull.prealloc(Nb);
                                xainnerProdsFull.resize(Nb);

                                xbinnerProdsFull.prealloc(Nb);
                                xbinnerProdsFull.resize(Nb);

                                for ( j = prevNb ; j < Nb ; ++j )
                                {
                                    for ( k = 0 ; k <= j ; ++k )
                                    {
                                        xadirectProdsFull("&",j,k) =  xa;
                                        xadirectProdsFull("&",j,k) *= xvdirectProdsFull(j,k);

                                        xadirectProdsFull("&",j,k).makealtcontent();
//redoxadir = 1;
                                    }
                                }

                                retVector<int> tmpva;
                                retVector<int> tmpvb;

                                prevbalphaState.prealloc(Nb);
                                prevbalphaState.resize(Nb);
                                prevbalphaState("&",prevNb,1,Nb-1,tmpva) = alphaState()(prevNb,1,Nb-1,tmpvb);

                                prevNb = Nb;
                            }

//if ( redoxadir )
//{
//errstream() << "??" << iacall << "," << ibcall << "??";
//}
//errstream() << "phantomx K2xfer 20\n";
                            //gentype ipres(0.0);
                            ipres.force_double() = 0.0;
                            double kres = 0.0;
                            const gentype *pxyprod[2];

                            pxyprod[0] = &ipres; // we will use this to pass the pre-calculated 4-product in directly
                            pxyprod[1] = nullptr;

//errstream() << "phantomx K2xfer 21\n";
                            for ( j = 0 ; j < Nb ; ++j )
                            {
                                // Not cheap, but can't see a way to avoid this

                                if ( alphaState()(j) )
                                {
                                    for ( k = 0 ; k <= j ; ++k )
                                    {
                                        if ( alphaState()(k) )
                                        {
                                            innerProductAssumeReal(ipres.force_double(),xadirectProdsFull(j,k),xb); // Assume no conjugation for speed here
//errstream() << "phantomx K2xfer 21b: " << ipres << "\n";

                                            NiceAssert( !testisvnan(ipres) );
                                            NiceAssert( !testisinf(ipres) );

                                            // We have xyprod, and other norms will be inferred from xinfo.  Note that 
                                            // nullptrs are filled at ML_Base level as required, and relevant norms are 
                                            // always cached in xinfo (so only calc once).
                                            //
                                            // We can safely assume that xainfo and xbinfo exist.  xbinfo is probably the
                                            // xinfo() from the calling class, and xainfo is created by g(x).

                                            kres = K4(ia,ib,j,k,pxyprod,&xa,&xb,nullptr,nullptr,&xainfo,&xbinfo,nullptr,nullptr,resmode);
//errstream() << "phantomx K2xfer 21c: " << kres << "\n";

                                            NiceAssert( !testisvnan(kres) );
                                            NiceAssert( !testisinf(kres) );

                                            if ( j != k )
                                            {
                                                kres *= 2; // there are two identical instances by symmetry, real by assumption and for speed
                                            }

                                            // Scale by alpha and add to result

//errstream() << "phantomx K2xfer 21d: " << kres << "*" << aadirectProdsFull(j,k) << "=";
                                            kres *= aadirectProdsFull(j,k);
//errstream() << kres << "\n";

                                            NiceAssert( !testisvnan(kres) );
                                            NiceAssert( !testisinf(kres) );

                                            res += kres;
//errstream() << "phantomx K2xfer 21e: " << res << "\n";
                                        }
                                    }
                                }
                            }

                            NiceAssert( !testisvnan(res) );
                            NiceAssert( !testisinf(res) );
//errstream() << "phantomx K2xfer 22\n";
                        }

                        else
                        {
//errstream() << "*(" << iacall << "," << ibcall << ")*";
                            // This calculation is one-hit, so for speed don't modify caches etc

                            // Note that iacall == ibcall >= 0

//errstream() << "?-1-?";
                            //gentype ipres(0.0);
                            ipres.force_double() = 0.0;
                            double kres = 0.0;
                            const gentype *pxyprod[2];

                            pxyprod[0] = &ipres; // we will use this to pass the pre-calculated 4-product in directly
                            pxyprod[1] = nullptr;

                            SparseVector<gentype> xout(xa);

                            xout *= xb;

                            for ( j = 0 ; j < Nb ; ++j )
                            {
                                if ( alphaState()(j) )
                                {
                                    for ( k = 0 ; k <= j ; ++k )
                                    {
                                        if ( alphaState()(k) )
                                        {
                                            twoProduct(ipres,xvdirectProdsFull(j,k),xout); // Again we assume commutativity etc

                                            NiceAssert( !testisvnan(ipres) );
                                            NiceAssert( !testisinf(ipres) );

                                            kres = K4(ia,ib,j,k,pxyprod,&xa,&xb,nullptr,nullptr,&xainfo,&xbinfo,nullptr,nullptr,resmode);

                                            NiceAssert( !testisvnan(kres) );
                                            NiceAssert( !testisinf(kres) );

                                            if ( j != k )
                                            {
                                                kres *= 2; // there are two identical instances by symmetry, real by assumption and for speed
                                            }

                                            // Scale by alpha and add to result

                                            kres *= aadirectProdsFull(j,k);

                                            NiceAssert( !testisvnan(kres) );
                                            NiceAssert( !testisinf(kres) );

                                            res += kres;
                                        }
                                    }
                                }
                            }

                            NiceAssert( !testisvnan(res) );
                            NiceAssert( !testisinf(res) );

                            if ( ( iacall == ibcall ) && ( iacall >= 0 ) )
                            {
                                diagkerns("&",iacall) = res;
                            }
                        }
                    }

                    else
                    {
                        int upastate = 0;
                        double kres;

                        for ( j = 0 ; j < prevNb ; ++j )
                        {
                            upastate |= ( alphaState()(j) != prevbalphaState(j) );

                            if ( ( getKernel().isAltDiff() == 2 ) && alphaState()(j) && ( xchange || !prevbalphaState(j) ) )
                            {
                                innerProductAssumeReal(xainnerProdsFull("&",j),xa,x(j));
                                innerProductAssumeReal(xbinnerProdsFull("&",j),xb,x(j));
                            }
                        }

                        if ( upastate )
                        {
                            retVector<int> tmpva;

                            prevbalphaState = alphaState()(0,1,prevNb-1,tmpva);
                        }

                        if ( xchange )
                        {
                            innerProductAssumeReal(xaxbinnerProd,xa,xb);

                            xaxainnerProd = getKernel().getmnorm(xainfo,xa,2,isXConsistent(),1);
                            xbxbinnerProd = getKernel().getmnorm(xbinfo,xb,2,isXConsistent(),1);
                        }

                        if ( prevNb != Nb )
                        {
                            xadirectProdsFull.prealloc(Nb,Nb);
                            xadirectProdsFull.resize(Nb,Nb);

                            xainnerProdsFull.prealloc(Nb);
                            xainnerProdsFull.resize(Nb);

                            xbinnerProdsFull.prealloc(Nb);
                            xbinnerProdsFull.resize(Nb);

                            if ( getKernel().isAltDiff() == 2 )
                            {
                                for ( j = prevNb ; j < Nb ; ++j )
                                {
                                    innerProductAssumeReal(xainnerProdsFull("&",j),xa,x(j));
                                    innerProductAssumeReal(xbinnerProdsFull("&",j),xb,x(j));
                                }
                            }

                            retVector<int> tmpva;
                            retVector<int> tmpvb;

                            prevbalphaState.prealloc(Nb);
                            prevbalphaState.resize(Nb);
                            prevbalphaState("&",prevNb,1,Nb-1,tmpva) = alphaState()(prevNb,1,Nb-1,tmpvb);

                            prevNb = Nb;
                        }

//errstream() << "phantomxyz 0 K2xfer: (ia,ib) = " << iacall << "," << ibcall << "\n";
//errstream() << "phantomxyz 0 K2xfer: xainnerProdsFull = " << xainnerProdsFull << "\n";
//errstream() << "phantomxyz 1 K2xfer: xbinnerProdsFull = " << xbinnerProdsFull << "\n";
//errstream() << "phantomxyz 2 K2xfer: xaxbinnerProd = " << xaxbinnerProd << "\n";
//errstream() << "phantomxyz 3 K2xfer: xaxainnerProd = " << xaxainnerProd << "\n";
//errstream() << "phantomxyz 4 K2xfer: xbxbinnerProd = " << xbxbinnerProd << "\n";
//errstream() << "phantomxyz 5 K2xfer: xvinnerProdsFull = " << xvinnerProdsFull << "\n";
//errstream() << "phantomxyz 6 K2xfer: aadirectProdsFull = " << aadirectProdsFull << "\n";
                        (xyvalid) = mlid; // xymat will now be used by K4 call, avoiding the requirement for any further inner products

                        for ( j = 0 ; j < Nb ; ++j )
                        {
                            if ( alphaState()(j) )
                            {
                                for ( k = 0 ; k <= j ; ++k )
                                {
                                    if ( alphaState()(k) )
                                    {
                                        kres = K4(ia,ib,j,k,nullptr,&xa,&xb,nullptr,nullptr,&xainfo,&xbinfo,nullptr,nullptr,resmode);

                                        if ( j != k )
                                        {
                                            kres *= 2.0; // there are two identical instances by symmetry, real by assumption and for speed
                                        }

                                        // Scale by alpha and add to result

                                        kres *= aadirectProdsFull(j,k);

                                        NiceAssert( !testisvnan(kres) );
                                        NiceAssert( !testisinf(kres) );

                                        res += kres;
                                    }
                                }
                            }
                        }

                        (xyvalid) = 0;

                        if ( ( iacall == ibcall ) && ( iacall >= 0 ) )
                        {
                            diagkerns("&",iacall) = res;
                        }
                    }

//errstream() << "phantomx 999 K(" << xa << "," << xb << ") = " << res << "\n";
//#ifdef ENABLE_THREADS
//                    eyelock.unlock();
//#endif
                }

                else
                {
//errstream() << "phantomx 2\n";
                    double Kij;

                    int j,k;
                    int docheat = ( ( ( resmode & 0x7F ) >= 0 ) && ( ( resmode & 0x7F ) <= 15 ) ) ? 1 : 0;

                    for ( j = 0 ; j < N() ; ++j )
                    {
                        if ( alphaState()(j) )
                        {
                            for ( k = 0 ; k < ( docheat ? j+1 : N() ) ; ++k )
                            {
                                if ( alphaState()(k) )
                                {
                                    switch ( resmode & 0x7f )
                                    {
                                        case 0:  case 1:  case 2:  case 3:
                                        case 4:  case 5:  case 6:  case 7:
                                        case 8:  case 9:  case 10: case 11:
                                        case 12: case 13: case 14: case 15:
                                        {
                                            Kij = K4(ia,ib,j,k,nullptr,&xa,&xb,nullptr,nullptr,&xainfo,&xbinfo,nullptr,nullptr,resmode);

                                            break;
                                        }

                                        case 16: case 32: case 48:
                                        {
                                            //Can't do this as additional vectors <x,...> now present, does not fit with xygrad, xnormgrad return

                                            NiceThrow("K2xfer 801 not supported.");

                                            break;
                                        }

                                        default:
                                        {
                                            NiceThrow("K2xfer precursor specified resmode undefined at this level.");

                                            break;
                                        }
                                    }

                                    if ( isUnderlyingScalar() )
                                    {
                                        Kij *= alphaR()(j)*alphaR()(k);
                                    }

                                    else if ( isUnderlyingVector() )
                                    {
                                        Kij *= twoProduct(dummyr,alphaV()(j),alphaV()(k));
                                    }

                                    else
                                    {
                                        gentype dummy;

                                        Kij *= (double) real(innerProduct(dummy,alpha()(j),alpha()(k)));
                                    }

                                    if ( docheat )
                                    {
                                        if ( k < j )
                                        {
                                            res += Kij;
                                            res += Kij;
                                        }

                                        else if ( k == j )
                                        {
                                            res += Kij;
                                        }
                                    }

                                    else
                                    {
                                        res += Kij;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if ( ( ia < 0 ) || ( ib < 0 ) ) { resetInnerWildp(); }

            break;
        }

        case 802:
        case 812:
        {
            if ( ia < 0 ) { setInnerWildpa(&xa,&xainfo); ia = -1; }
            if ( ib < 0 ) { setInnerWildpb(&xb,&xbinfo); ib = -2; }

            NiceAssert( !(resmode & 0x80 ) );

            res = 0.0;

            if ( NLB()+NUB()+NF() )
            {
                double Kija,Kijb;

                int j,k;

                for ( j = 0 ; j < N() ; ++j )
                {
                    if ( alphaState()(j) )
                    {
                        for ( k = 0 ; k < N() ; ++k )
                        {
                            if ( alphaState()(k) )
                            {
                                switch ( resmode & 0x7f )
                                {
                                    case 0:  case 1:  case 2:  case 3:
                                    case 4:  case 5:  case 6:  case 7:
                                    case 8:  case 9:  case 10: case 11:
                                    case 12: case 13: case 14: case 15:
                                    {
                                        Kija = K2(ia,j,nullptr,&xa,nullptr,&xainfo,nullptr,resmode);
                                        Kijb = K2(ib,k,nullptr,&xb,nullptr,&xbinfo,nullptr,resmode);

                                        Kija *= Kijb;

                                        break;
                                    }

                                    case 16: case 32: case 48:
                                    {
                                        // Can't do this as additional vectors <x,...> now present, does not fit with xygrad, xnormgrad return

                                        NiceThrow("K2xfer 802 not supported.");

                                        break;
                                    }

                                    case 64:
                                    {
                                        NiceThrow("K2xfer precursor second order derivatives not yet implemented.");

                                        break;
                                    }

                                    default:
                                    {
                                        NiceThrow("K2xfer precursor specified resmode undefined at this level.");

                                        break;
                                    }
                                }

                                if ( isUnderlyingScalar() )
                                {
                                    Kija *= alphaR()(j)*alphaR()(k);
                                }

                                else if ( isUnderlyingVector() )
                                {
                                    Kija *= twoProduct(dummyr,alphaV()(j),alphaV()(k));
                                }

                                else
                                {
                                    Kija *= real(alphaA()(j)*conj(alphaA()(k)));
                                    //Kija *= (double) real(innerProduct(dummy,alpha()(j),alpha()(k)));
                                }

                                res += Kija;
                                res += Kija;
                            }
                        }
                    }
                }
            }

            if ( ( ia < 0 ) || ( ib < 0 ) ) { resetInnerWildp(); }

            break;
        }

        case 807:
        case 817:
        {
            gentype Kintemp;
            double dummyr;

            K1(Kintemp,xa,&xainfo);

            res = 0.0;

            if ( NLB()+NUB()+NF() )
            {
                gentype Kij;
                gentype temp;

                int j,k;

                for ( j = 0 ; j < N() ; ++j )
                {
                    if ( alphaState()(j) )
                    {
                        for ( k = 0 ; k < N() ; ++k )
                        {
                            if ( alphaState()(k) )
                            {
                                K2(Kij,j,k);

                                temp =  Kintemp;
                                temp *= Kij;
                                temp -= 1.0;
                                temp = -1.0/temp;

                                if ( isUnderlyingScalar() )
                                {
                                    temp *= alphaR()(j)*alphaR()(k);
                                }

                                else if ( isUnderlyingVector() )
                                {
                                    temp *= innerProduct(dummyr,alphaV()(j),alphaV()(k));
                                }

                                else
                                {
                                    temp *= (double) real(alpha()(j)*alpha()(k));
                                }

                                res += (double) temp;
                            }
                        }
                    }
                }
            }

            break;
        }

        default:
        {
            ML_Base::K2xfer(dxyprod,ddiffis,res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xainfo,xbinfo,ia,ib,xdim,densetype,resmode,mlid);

            break;
        }
    }

    NiceAssert( !testisvnan(res) );
    NiceAssert( !testisinf(res) );

    return;
}






















void SVM_Generic::K3xfer(gentype &res, int &minmaxind, int typeis,
                         const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                         const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, 
                         const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, 
                         int ia, int ib, int ic, 
                         int xdim, int densetype, int resmode, int mlid) const
{
    NiceAssert( !densetype );

    (void) densetype;
    (void) xdim;
    gentype dummy;
    double dummyr = 0.0;

    ia = (typeis-(100*(typeis/100)))/10 ? ia : -42;
    ib = (typeis-(100*(typeis/100)))/10 ? ib : -43;
    ic = (typeis-(100*(typeis/100)))/10 ? ic : -44;

    switch ( typeis )
    {
        case 801:
        case 811:
        {
            if ( ia < 0 ) { setInnerWildpa(&xa,&xainfo); ia = -1; }
            if ( ib < 0 ) { setInnerWildpb(&xb,&xbinfo); ib = -2; }
            if ( ic < 0 ) { setInnerWildpc(&xc,&xcinfo); ic = -3; }

            Vector<const SparseVector<gentype> *> x(6);
            Vector<const vecInfo *> xinfo(6);
            Vector<int> i(6);

            x("&",0) = &xa;
            x("&",1)         = &xb;
            x("&",2)         = &xc;
            x("&",3)         = nullptr;
            x("&",4)         = nullptr;
            x("&",5)         = nullptr;

            xinfo("&",0) = &xainfo;
            xinfo("&",1)         = &xbinfo;
            xinfo("&",2)         = &xcinfo;
            xinfo("&",3)         = nullptr;
            xinfo("&",4)         = nullptr;
            xinfo("&",5)         = nullptr;

            i("&",0) = ia;
            i("&",1)         = ib;
            i("&",2)         = ic;
            i("&",3)         = 0;
            i("&",4)         = 0;
            i("&",5)         = 0;

            res = 0.0;

            if ( NLB()+NUB()+NF() )
            {
                gentype Kij;

                int j,k,l;
                int docheat = ( ( ( resmode & 0x7F ) >= 0 ) && ( ( resmode & 0x7F ) <= 15 ) ) ? 1 : 0;

                for ( j = 0 ; j < N() ; ++j )
                {
                    if ( alphaState()(j) )
                    {
                        for ( k = 0 ; k < ( docheat ? j+1 : N() ) ; ++k )
                        {
                            if ( alphaState()(k) )
                            {
                                for ( l = 0 ; l < ( docheat ? k+1 : N() ) ; ++l )
                                {
                                    if ( alphaState()(l) )
                                    {
                                                i("&",3) = j;
                                                i("&",4) = k;
                                                i("&",5) = l;

                                                switch ( resmode & 0x7f )
                                                {
                                                    case 0:  case 1:  case 2:  case 3:
                                                    case 4:  case 5:  case 6:  case 7:
                                                    case 8:  case 9:  case 10: case 11:
                                                    case 12: case 13: case 14: case 15:
                                                    {
                                                        Km(6,Kij,i,nullptr,&x,&xinfo,resmode);

                                                        break;
                                                    }

                                                    case 64:
                                                    {
                                                        NiceThrow("K3xfer precursor second order derivatives not yet implemented.");

                                                        break;
                                                    }

                                                    default:
                                                    {
                                                        NiceThrow("K3xfer precursor specified resmode undefined at this level.");

                                                        break;
                                                    }
                                                }

                                                if ( isUnderlyingScalar() )
                                                {
                                                    Kij *= alphaR()(j)*alphaR()(k)*alphaR()(l);
                                                }
                                        
                                                else if ( isUnderlyingVector() )
                                                {
                                                    Kij *= threeProduct(dummyr,alphaV()(j),alphaV()(k),alphaV()(l));
                                                }
                                        
                                                else
                                                {
                                                    Kij *= (double) real(threeProduct(dummy,((const Vector<gentype> &) alpha()(j)),((const Vector<gentype> &) alpha()(k)),((const Vector<gentype> &) alpha()(l))));
                                                }

                                                if ( docheat )
                                                {
                                                    if ( ( j == k ) && ( j == l ) )
                                                    {
                                                        res += Kij;
                                                    }

                                                    else if ( ( ( j == k ) && ( j != l ) ) ||
                                                              ( ( j == l ) && ( j != k ) ) ||
                                                              ( ( k == l ) && ( k != j ) )    )
                                                    {
                                                        res += Kij;
                                                        res += Kij;
                                                        res += Kij;
                                                        res += Kij;
                                                    }

                                                    else if ( ( j != k ) && ( j != l ) && ( k != l ) )
                                                    {
                                                        res += Kij; 
                                                        res += Kij; 
                                                        res += Kij; 
                                                        res += Kij; 
                                                        res += Kij; 
                                                    }
                                                }

                                                else
                                                {
                                                    res += Kij;
                                                }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if ( ( ia < 0 ) || ( ib < 0 ) || ( ic < 0 ) ) { resetInnerWildp(); }

            break;
        }

        case 807:
        case 817:
        {
            gentype Kintemp;
            double dummyr;

            K1(Kintemp,xa,&xainfo);

            res = 0.0;

            if ( NLB()+NUB()+NF() )
            {
                gentype Kij;
                gentype temp;

                int j,k,l;

                for ( j = 0 ; j < N() ; ++j )
                {
                    if ( alphaState()(j) )
                    {
                        for ( k = 0 ; k < N() ; ++k )
                        {
                            if ( alphaState()(k) )
                            {
                                for ( l = 0 ; l < N() ; ++l )
                                {
                                    if ( alphaState()(l) )
                                    {
                                        K3(Kij,j,k,l);

                                        temp =  Kintemp;
                                        temp *= Kij;
                                        temp -= 1.0;
                                        temp = -1.0/temp;

                                        if ( isUnderlyingScalar() )
                                        {
                                            temp *= alphaR()(j)*alphaR()(k)*alphaR()(l);
                                        }

                                        else if ( isUnderlyingVector() )
                                        {
                                            temp *= threeProduct(dummyr,alphaV()(j),alphaV()(k),alphaV()(l));
                                        }

                                        else
                                        {
                                            temp *= (double) real(alpha()(j)*alpha()(k)*alpha()(l));
                                        }

                                        res += temp;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            break;
        }

        default:
        {
            ML_Base::K3xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xc,xainfo,xbinfo,xcinfo,ia,ib,ic,xdim,densetype,resmode,mlid);

            break;
        }
    }

    return;
}

void SVM_Generic::K3xfer(double &res, int &minmaxind, int typeis,
                         double xyprod, double yxprod, double diffis,
                         const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, 
                         const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, 
                         int ia, int ib, int ic, 
                         int xdim, int densetype, int resmode, int mlid) const
{
    NiceAssert( !densetype );

    if ( ( resmode & 0x7f ) >= 16 )
    {
        gentype temp;

        gentype gxyprod(xyprod);
        gentype gyxprod(yxprod);
        gentype gdiffis(diffis);

        K3xfer(temp,minmaxind,typeis,gxyprod,gyxprod,gdiffis,xa,xb,xc,xainfo,xbinfo,xcinfo,ia,ib,ic,xdim,densetype,resmode,mlid);

        return;
    }

    gentype dummy;
    double dummyr = 0.0;

    ia = (typeis-(100*(typeis/100)))/10 ? ia : -42;
    ib = (typeis-(100*(typeis/100)))/10 ? ib : -43;
    ic = (typeis-(100*(typeis/100)))/10 ? ic : -44;

    switch ( typeis )
    {
        case 801:
        case 811:
        {
            if ( ia < 0 ) { setInnerWildpa(&xa,&xainfo); ia = -1; }
            if ( ib < 0 ) { setInnerWildpb(&xb,&xbinfo); ib = -2; }
            if ( ic < 0 ) { setInnerWildpc(&xc,&xcinfo); ic = -3; }

            Vector<const SparseVector<gentype> *> x(6);
            Vector<const vecInfo *> xinfo(6);
            Vector<int> i(6);

            x("&",0) = &xa;
            x("&",1)         = &xb;
            x("&",2)         = &xc;
            x("&",3)         = nullptr;
            x("&",4)         = nullptr;
            x("&",5)         = nullptr;

            xinfo("&",0) = &xainfo;
            xinfo("&",1)         = &xbinfo;
            xinfo("&",2)         = &xcinfo;
            xinfo("&",3)         = nullptr;
            xinfo("&",4)         = nullptr;
            xinfo("&",5)         = nullptr;

            i("&",0) = ia;
            i("&",1)         = ib;
            i("&",2)         = ic;
            i("&",3)         = 0;
            i("&",4)         = 0;
            i("&",5)         = 0;

            res = 0.0;

            if ( NLB()+NUB()+NF() )
            {
                double Kij;

                int j,k,l;
                int docheat = ( ( ( resmode & 0x7F ) >= 0 ) && ( ( resmode & 0x7F ) <= 15 ) ) ? 1 : 0;

                for ( j = 0 ; j < N() ; ++j )
                {
                    if ( alphaState()(j) )
                    {
                        for ( k = 0 ; k < ( docheat ? j+1 : N() ) ; ++k )
                        {
                            if ( alphaState()(k) )
                            {
                                for ( l = 0 ; l < ( docheat ? k+1 : N() ) ; ++l )
                                {
                                    if ( alphaState()(l) )
                                    {
                                                i("&",3) = j;
                                                i("&",4) = k;
                                                i("&",5) = l;

                                                switch ( resmode & 0x7f )
                                                {
                                                    case 0:  case 1:  case 2:  case 3:
                                                    case 4:  case 5:  case 6:  case 7:
                                                    case 8:  case 9:  case 10: case 11:
                                                    case 12: case 13: case 14: case 15:
                                                    {
                                                        Kij = Km(6,i,nullptr,&x,&xinfo,resmode);

                                                        break;
                                                    }

                                                    case 64:
                                                    {
                                                        NiceThrow("K3xfer precursor second order derivatives not yet implemented.");

                                                        break;
                                                    }

                                                    default:
                                                    {
                                                        NiceThrow("K3xfer precursor specified resmode undefined at this level.");

                                                        break;
                                                    }
                                                }

                                                if ( isUnderlyingScalar() )
                                                {
                                                    Kij *= alphaR()(j)*alphaR()(k)*alphaR()(l);
                                                }
                                        
                                                else if ( isUnderlyingVector() )
                                                {
                                                    Kij *= threeProduct(dummyr,alphaV()(j),alphaV()(k),alphaV()(l));
                                                }
                                        
                                                else
                                                {
                                                    Kij *= (double) real(threeProduct(dummy,((const Vector<gentype> &) alpha()(j)),((const Vector<gentype> &) alpha()(k)),((const Vector<gentype> &) alpha()(l))));
                                                }

                                                if ( docheat )
                                                {
                                                    if ( ( j == k ) && ( j == l ) )
                                                    {
                                                        res += Kij;
                                                    }

                                                    else if ( ( ( j == k ) && ( j != l ) ) ||
                                                              ( ( j == l ) && ( j != k ) ) ||
                                                              ( ( k == l ) && ( k != j ) )    )
                                                    {
                                                        res += Kij;
                                                        res += Kij;
                                                        res += Kij;
                                                        res += Kij;
                                                    }

                                                    else if ( ( j != k ) && ( j != l ) && ( k != l ) )
                                                    {
                                                        res += Kij; 
                                                        res += Kij; 
                                                        res += Kij; 
                                                        res += Kij; 
                                                        res += Kij; 
                                                    }
                                                }

                                                else
                                                {
                                                    res += Kij;
                                                }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if ( ( ia < 0 ) || ( ib < 0 ) || ( ic < 0 ) ) { resetInnerWildp(); }

            break;
        }

        case 807:
        case 817:
        {
            gentype Kintemp;
            double dummyr;

            K1(Kintemp,xa,&xainfo);

            res = 0.0;

            if ( NLB()+NUB()+NF() )
            {
                gentype Kij;
                gentype temp;

                int j,k,l;

                for ( j = 0 ; j < N() ; ++j )
                {
                    if ( alphaState()(j) )
                    {
                        for ( k = 0 ; k < N() ; ++k )
                        {
                            if ( alphaState()(k) )
                            {
                                for ( l = 0 ; l < N() ; ++l )
                                {
                                    if ( alphaState()(l) )
                                    {
                                        K3(Kij,j,k,l);

                                        temp =  Kintemp;
                                        temp *= Kij;
                                        temp -= 1.0;
                                        temp = -1.0/temp;

                                        if ( isUnderlyingScalar() )
                                        {
                                            temp *= alphaR()(j)*alphaR()(k)*alphaR()(l);
                                        }

                                        else if ( isUnderlyingVector() )
                                        {
                                            temp *= threeProduct(dummyr,alphaV()(j),alphaV()(k),alphaV()(l));
                                        }

                                        else
                                        {
                                            temp *= (double) real(alpha()(j)*alpha()(k)*alpha()(l));
                                        }

                                        res += (double) temp;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            break;
        }

        default:
        {
            ML_Base::K3xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xc,xainfo,xbinfo,xcinfo,ia,ib,ic,xdim,densetype,resmode,mlid);

            break;
        }
    }

    return;
}






















void SVM_Generic::K4xfer(gentype &res, int &minmaxind, int typeis,
                         const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                         const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                         const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
                         int ia, int ib, int ic, int id,
                         int xdim, int densetype, int resmode, int mlid) const
{
    NiceAssert( !densetype );

    (void) densetype;
    (void) xdim;
    gentype dummy;
    double dummyr = 0.0;

    ia = (typeis-(100*(typeis/100)))/10 ? ia : -42;
    ib = (typeis-(100*(typeis/100)))/10 ? ib : -43;
    ic = (typeis-(100*(typeis/100)))/10 ? ic : -44;
    id = (typeis-(100*(typeis/100)))/10 ? id : -45;

    switch ( typeis )
    {
        case 801:
        case 811:
        {
            if ( ia < 0 ) { setInnerWildpa(&xa,&xainfo); ia = -1; }
            if ( ib < 0 ) { setInnerWildpb(&xb,&xbinfo); ib = -2; }
            if ( ic < 0 ) { setInnerWildpc(&xc,&xcinfo); ic = -3; }
            if ( id < 0 ) { setInnerWildpd(&xd,&xdinfo); id = -4; }

            Vector<const SparseVector<gentype> *> x(8);
            Vector<const vecInfo *> xinfo(8);
            Vector<int> i(8);

            x("&",0) = &xa;
            x("&",1)         = &xb;
            x("&",2)         = &xc;
            x("&",3)         = &xd;
            x("&",4)         = nullptr;
            x("&",5)         = nullptr;
            x("&",6)         = nullptr;
            x("&",7)         = nullptr;

            xinfo("&",0) = &xainfo;
            xinfo("&",1)         = &xbinfo;
            xinfo("&",2)         = &xcinfo;
            xinfo("&",3)         = &xdinfo;
            xinfo("&",4)         = nullptr;
            xinfo("&",5)         = nullptr;
            xinfo("&",6)         = nullptr;
            xinfo("&",7)         = nullptr;

            i("&",0) = ia;
            i("&",1)         = ib;
            i("&",2)         = ic;
            i("&",3)         = id;
            i("&",4)         = 0;
            i("&",5)         = 0;
            i("&",6)         = 0;
            i("&",7)         = 0;

            res = 0.0;

            if ( NLB()+NUB()+NF() )
            {
                gentype Kij;

                int j,k,l,m;
                int docheat = ( ( ( resmode & 0x7F ) >= 0 ) && ( ( resmode & 0x7F ) <= 15 ) ) ? 1 : 0;

                for ( j = 0 ; j < N() ; ++j )
                {
                    if ( alphaState()(j) )
                    {
                        for ( k = 0 ; k < ( docheat ? j+1 : N() ) ; ++k )
                        {
                            if ( alphaState()(k) )
                            {
                                for ( l = 0 ; l < ( docheat ? k+1 : N() ) ; ++l )
                                {
                                    if ( alphaState()(l) )
                                    {
                                        for ( m = 0 ; m < ( docheat ? l+1 : N() ) ; ++m )
                                        {
                                            if ( alphaState()(m) )
                                            {
                                                i("&",4) = j;
                                                i("&",5) = k;
                                                i("&",6) = l;
                                                i("&",7) = m;

                                                switch ( resmode & 0x7f )
                                                {
                                                    case 0:  case 1:  case 2:  case 3:
                                                    case 4:  case 5:  case 6:  case 7:
                                                    case 8:  case 9:  case 10: case 11:
                                                    case 12: case 13: case 14: case 15:
                                                    {
                                                        Km(8,Kij,i,nullptr,&x,&xinfo,resmode);

                                                        break;
                                                    }

                                                    case 64:
                                                    {
                                                        NiceThrow("K4xfer precursor second order derivatives not yet implemented.");

                                                        break;
                                                    }

                                                    default:
                                                    {
                                                        NiceThrow("K4xfer precursor specified resmode undefined at this level.");

                                                        break;
                                                    }
                                                }

                                                if ( isUnderlyingScalar() )
                                                {
                                                    Kij *= alphaR()(j)*alphaR()(k)*alphaR()(l)*alphaR()(m);
                                                }
                                        
                                                else if ( isUnderlyingVector() )
                                                {
                                                    Kij *= fourProduct(dummyr,alphaV()(j),alphaV()(k),alphaV()(l),alphaV()(m));
                                                }
                                        
                                                else
                                                {
                                                    Kij *= (double) real(fourProduct(dummy,alpha()(j),alpha()(k),alpha()(l),alpha()(m)));
                                                }

                                                if ( docheat )
                                                {
                                                    if ( ( j == k ) && ( j == l ) && ( j == m ) )
                                                    {
                                                        res += Kij;
                                                    }

                                                    else if ( ( ( j == k ) && ( j == l ) && ( j != m ) ) ||
                                                              ( ( k == l ) && ( k == m ) && ( k != j ) ) ||
                                                              ( ( l == m ) && ( l == j ) && ( l != k ) ) ||
                                                              ( ( m == j ) && ( m == k ) && ( m != l ) )    )
                                                    {
                                                        res += Kij;
                                                        res += Kij;
                                                    }

                                                    else if ( ( ( j == k ) && ( l != m ) && ( j != l ) ) ||
                                                              ( ( j == l ) && ( k != m ) && ( j != k ) ) ||
                                                              ( ( j == m ) && ( k != l ) && ( j != k ) ) ||
                                                              ( ( k == l ) && ( j != m ) && ( k != j ) ) ||
                                                              ( ( k == m ) && ( j != l ) && ( k != j ) ) ||
                                                              ( ( l == m ) && ( j != k ) && ( l != j ) )    )
                                                    {
                                                        res += Kij;
                                                        res += Kij;
                                                        res += Kij;
                                                        res += Kij;
                                                        res += Kij;
                                                        res += Kij;
                                                    }

                                                    else if ( ( j != k ) && ( j != l ) && ( j != m ) && ( k != l ) && ( k != m ) && ( l != m ) )
                                                    {
                                                        res += Kij; res += Kij; res += Kij;
                                                        res += Kij; res += Kij; res += Kij;
                                                        res += Kij; res += Kij; res += Kij;
                                                        res += Kij; res += Kij; res += Kij;
                                                        res += Kij; res += Kij; res += Kij;
                                                        res += Kij; res += Kij; res += Kij;
                                                        res += Kij; res += Kij; res += Kij;
                                                        res += Kij; res += Kij; res += Kij;
                                                    }
                                                }

                                                else
                                                {
                                                    res += Kij;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if ( ( ia < 0 ) || ( ib < 0 ) || ( ic < 0 ) || ( id < 0 ) ) { resetInnerWildp(); }

            break;
        }

        case 802:
        case 812:
        {
            if ( ia < 0 ) { setInnerWildpa(&xa,&xainfo); ia = -1; }
            if ( ib < 0 ) { setInnerWildpb(&xb,&xbinfo); ib = -2; }
            if ( ic < 0 ) { setInnerWildpc(&xc,&xcinfo); ic = -3; }
            if ( id < 0 ) { setInnerWildpd(&xd,&xdinfo); id = -4; }

            NiceAssert( !( resmode & 0x80 ) );

            res = 0.0;

            if ( NLB()+NUB()+NF() )
            {
                gentype Kij;

                gentype Kijaa,Kijab;
                gentype Kijba,Kijbb;
                gentype Kijca,Kijcb;
                gentype Kijda,Kijdb;
                gentype Kijea,Kijeb;
                gentype Kijfa,Kijfb;

                int j,k,l,m;

                for ( j = 0 ; j < N() ; ++j )
                {
                    if ( alphaState()(j) )
                    {
                        for ( k = 0 ; k < N() ; ++k )
                        {
                            if ( alphaState()(k) )
                            {
                                for ( l = 0 ; l < N() ; ++l )
                                {
                                    if ( alphaState()(l) )
                                    {
                                        for ( m = 0 ; m < N() ; ++m )
                                        {
                                            if ( alphaState()(m) )
                                            {
                                                switch ( resmode & 0x7f )
                                                {
                                                    case 0:  case 1:  case 2:  case 3:
                                                    case 4:  case 5:  case 6:  case 7:
                                                    case 8:  case 9:  case 10: case 11:
                                                    case 12: case 13: case 14: case 15:
                                                    {
                                                        K4(Kijaa,ia,ib,j,k,nullptr,&xa,&xb,nullptr,nullptr,&xainfo,&xbinfo,nullptr,nullptr); K4(Kijab,ic,id,l,m,nullptr,&xa,&xb,nullptr,nullptr,&xainfo,&xbinfo,nullptr,nullptr);
                                                        K4(Kijba,ia,ic,j,k,nullptr,&xa,&xc,nullptr,nullptr,&xainfo,&xcinfo,nullptr,nullptr); K4(Kijbb,ib,id,l,m,nullptr,&xa,&xc,nullptr,nullptr,&xainfo,&xcinfo,nullptr,nullptr);
                                                        K4(Kijca,ia,id,j,k,nullptr,&xa,&xd,nullptr,nullptr,&xainfo,&xdinfo,nullptr,nullptr); K4(Kijcb,ib,ic,l,m,nullptr,&xa,&xd,nullptr,nullptr,&xainfo,&xdinfo,nullptr,nullptr);
                                                        K4(Kijda,ib,ic,j,k,nullptr,&xb,&xc,nullptr,nullptr,&xbinfo,&xcinfo,nullptr,nullptr); K4(Kijdb,ia,id,l,m,nullptr,&xb,&xc,nullptr,nullptr,&xbinfo,&xcinfo,nullptr,nullptr);
                                                        K4(Kijea,ib,id,j,k,nullptr,&xb,&xd,nullptr,nullptr,&xbinfo,&xdinfo,nullptr,nullptr); K4(Kijeb,ia,id,l,m,nullptr,&xb,&xd,nullptr,nullptr,&xbinfo,&xdinfo,nullptr,nullptr);
                                                        K4(Kijfa,ic,id,j,k,nullptr,&xc,&xd,nullptr,nullptr,&xcinfo,&xdinfo,nullptr,nullptr); K4(Kijfb,ia,ib,l,m,nullptr,&xc,&xd,nullptr,nullptr,&xcinfo,&xdinfo,nullptr,nullptr);

                                                        Kijaa *= Kijab;
                                                        Kijba *= Kijbb;
                                                        Kijca *= Kijcb;
                                                        Kijda *= Kijdb;
                                                        Kijea *= Kijeb;
                                                        Kijfa *= Kijfb;

                                                        Kij  = Kijaa;
                                                        Kij += Kijba;
                                                        Kij += Kijca;
                                                        Kij += Kijda;
                                                        Kij += Kijea;
                                                        Kij += Kijfa;

                                                        break;
                                                    }

                                                    case 64:
                                                    {
                                                        NiceThrow("K4xfer precursor second order derivatives not yet implemented.");

                                                        break;
                                                    }

                                                    default:
                                                    {
                                                        NiceThrow("K4xfer precursor specified resmode undefined at this level.");

                                                        break;
                                                    }
                                                }

                                                if ( isUnderlyingScalar() )
                                                {
                                                    Kij *= alphaR()(j)*alphaR()(k)*alphaR()(l)*alphaR()(m);
                                                }
                                        
                                                else if ( isUnderlyingVector() )
                                                {
                                                    Kij *= fourProduct(dummyr,alphaV()(j),alphaV()(k),alphaV()(l),alphaV()(m));
                                                }
                                        
                                                else
                                                {
                                                    Kij *= (double) real(fourProduct(dummy,alpha()(j),alpha()(k),alpha()(l),alpha()(m)));
                                                }

                                                res += Kij;
                                                res += Kij;
                                                res += Kij;
                                                res += Kij;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if ( ( ia < 0 ) || ( ib < 0 ) || ( ic < 0 ) || ( id < 0 ) ) { resetInnerWildp(); }

            break;
        }

        case 807:
        case 817:
        {
            gentype Kintemp;
            double dummyr;

            K1(Kintemp,xa,&xainfo);

            res = 0.0;

            if ( NLB()+NUB()+NF() )
            {
                gentype Kij;
                gentype temp;

                int j,k,l,m;

                for ( j = 0 ; j < N() ; ++j )
                {
                    if ( alphaState()(j) )
                    {
                        for ( k = 0 ; k < N() ; ++k )
                        {
                            if ( alphaState()(k) )
                            {
                                for ( l = 0 ; l < N() ; ++l )
                                {
                                    if ( alphaState()(l) )
                                    {
                                        for ( m = 0 ; m < N() ; ++m )
                                        {
                                            if ( alphaState()(m) )
                                            {
                                                K4(Kij,j,k,l,m);

                                                temp =  Kintemp;
                                                temp *= Kij;
                                                temp -= 1.0;
                                                temp = -1.0/temp;

                                                if ( isUnderlyingScalar() )
                                                {
                                                    temp *= alphaR()(j)*alphaR()(k)*alphaR()(l)*alphaR()(m);
                                                }

                                                else if ( isUnderlyingVector() )
                                                {
                                                    temp *= fourProduct(dummyr,alphaV()(j),alphaV()(k),alphaV()(l),alphaV()(m));
                                                }

                                                else
                                                {
                                                    temp *= (double) real(alpha()(j)*alpha()(k)*alpha()(l)*alpha()(m));
                                                }

                                                res += temp;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            break;
        }

        default:
        {
            ML_Base::K4xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,ia,ib,ic,id,xdim,densetype,resmode,mlid);

            break;
        }
    }

    return;
}

void SVM_Generic::K4xfer(double &res, int &minmaxind, int typeis,
                         double xyprod, double yxprod, double diffis,
                         const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                         const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
                         int ia, int ib, int ic, int id,
                         int xdim, int densetype, int resmode, int mlid) const
{
    NiceAssert( !densetype );

    if ( ( resmode & 0x7f ) >= 16 )
    {
        gentype temp;

        gentype gxyprod(xyprod);
        gentype gyxprod(yxprod);
        gentype gdiffis(diffis);

        K4xfer(temp,minmaxind,typeis,gxyprod,gyxprod,gdiffis,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,ia,ib,ic,id,xdim,densetype,resmode,mlid);

        return;
    }

    gentype dummy;
    double dummyr = 0.0;

    ia = (typeis-(100*(typeis/100)))/10 ? ia : -42;
    ib = (typeis-(100*(typeis/100)))/10 ? ib : -43;
    ic = (typeis-(100*(typeis/100)))/10 ? ic : -44;
    id = (typeis-(100*(typeis/100)))/10 ? id : -45;

    switch ( typeis )
    {
        case 801:
        case 811:
        {
            if ( ia < 0 ) { setInnerWildpa(&xa,&xainfo); ia = -1; }
            if ( ib < 0 ) { setInnerWildpb(&xb,&xbinfo); ib = -2; }
            if ( ic < 0 ) { setInnerWildpc(&xc,&xcinfo); ic = -3; }
            if ( id < 0 ) { setInnerWildpd(&xd,&xdinfo); id = -4; }

            Vector<const SparseVector<gentype> *> x(8);
            Vector<const vecInfo *> xinfo(8);
            Vector<int> i(8);

            x("&",0) = &xa;
            x("&",1)         = &xb;
            x("&",2)         = &xc;
            x("&",3)         = &xd;
            x("&",4)         = nullptr;
            x("&",5)         = nullptr;
            x("&",6)         = nullptr;
            x("&",7)         = nullptr;

            xinfo("&",0) = &xainfo;
            xinfo("&",1)         = &xbinfo;
            xinfo("&",2)         = &xcinfo;
            xinfo("&",3)         = &xdinfo;
            xinfo("&",4)         = nullptr;
            xinfo("&",5)         = nullptr;
            xinfo("&",6)         = nullptr;
            xinfo("&",7)         = nullptr;

            i("&",0) = ia;
            i("&",1)         = ib;
            i("&",2)         = ic;
            i("&",3)         = id;
            i("&",4)         = 0;
            i("&",5)         = 0;
            i("&",6)         = 0;
            i("&",7)         = 0;

            res = 0.0;

            if ( NLB()+NUB()+NF() )
            {
                double Kij;

                int j,k,l,m;
                int docheat = ( ( ( resmode & 0x7F ) >= 0 ) && ( ( resmode & 0x7F ) <= 15 ) ) ? 1 : 0;

                for ( j = 0 ; j < N() ; ++j )
                {
                    if ( alphaState()(j) )
                    {
                        for ( k = 0 ; k < ( docheat ? j+1 : N() ) ; ++k )
                        {
                            if ( alphaState()(k) )
                            {
                                for ( l = 0 ; l < ( docheat ? k+1 : N() ) ; ++l )
                                {
                                    if ( alphaState()(l) )
                                    {
                                        for ( m = 0 ; m < ( docheat ? l+1 : N() ) ; ++m )
                                        {
                                            if ( alphaState()(m) )
                                            {
                                                i("&",4) = j;
                                                i("&",5) = k;
                                                i("&",6) = l;
                                                i("&",7) = m;

                                                switch ( resmode & 0x7f )
                                                {
                                                    case 0:  case 1:  case 2:  case 3:
                                                    case 4:  case 5:  case 6:  case 7:
                                                    case 8:  case 9:  case 10: case 11:
                                                    case 12: case 13: case 14: case 15:
                                                    {
                                                        Kij = Km(8,i,nullptr,&x,&xinfo,resmode);

                                                        break;
                                                    }

                                                    case 64:
                                                    {
                                                        NiceThrow("K4xfer precursor second order derivatives not yet implemented.");

                                                        break;
                                                    }

                                                    default:
                                                    {
                                                        NiceThrow("K4xfer precursor specified resmode undefined at this level.");

                                                        break;
                                                    }
                                                }

                                                if ( isUnderlyingScalar() )
                                                {
                                                    Kij *= alphaR()(j)*alphaR()(k)*alphaR()(l)*alphaR()(m);
                                                }
                                        
                                                else if ( isUnderlyingVector() )
                                                {
                                                    Kij *= fourProduct(dummyr,alphaV()(j),alphaV()(k),alphaV()(l),alphaV()(m));
                                                }
                                        
                                                else
                                                {
                                                    Kij *= (double) real(fourProduct(dummy,alpha()(j),alpha()(k),alpha()(l),alpha()(m)));
                                                }

                                                if ( docheat )
                                                {
                                                    if ( ( j == k ) && ( j == l ) && ( j == m ) )
                                                    {
                                                        res += Kij;
                                                    }

                                                    else if ( ( ( j == k ) && ( j == l ) && ( j != m ) ) ||
                                                              ( ( k == l ) && ( k == m ) && ( k != j ) ) ||
                                                              ( ( l == m ) && ( l == j ) && ( l != k ) ) ||
                                                              ( ( m == j ) && ( m == k ) && ( m != l ) )    )
                                                    {
                                                        res += Kij;
                                                        res += Kij;
                                                    }

                                                    else if ( ( ( j == k ) && ( l != m ) && ( j != l ) ) ||
                                                              ( ( j == l ) && ( k != m ) && ( j != k ) ) ||
                                                              ( ( j == m ) && ( k != l ) && ( j != k ) ) ||
                                                              ( ( k == l ) && ( j != m ) && ( k != j ) ) ||
                                                              ( ( k == m ) && ( j != l ) && ( k != j ) ) ||
                                                              ( ( l == m ) && ( j != k ) && ( l != j ) )    )
                                                    {
                                                        res += Kij;
                                                        res += Kij;
                                                        res += Kij;
                                                        res += Kij;
                                                        res += Kij;
                                                        res += Kij;
                                                    }

                                                    else if ( ( j != k ) && ( j != l ) && ( j != m ) && ( k != l ) && ( k != m ) && ( l != m ) )
                                                    {
                                                        res += Kij; res += Kij; res += Kij;
                                                        res += Kij; res += Kij; res += Kij;
                                                        res += Kij; res += Kij; res += Kij;
                                                        res += Kij; res += Kij; res += Kij;
                                                        res += Kij; res += Kij; res += Kij;
                                                        res += Kij; res += Kij; res += Kij;
                                                        res += Kij; res += Kij; res += Kij;
                                                        res += Kij; res += Kij; res += Kij;
                                                    }
                                                }

                                                else
                                                {
                                                    res += Kij;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if ( ( ia < 0 ) || ( ib < 0 ) || ( ic < 0 ) || ( id < 0 ) ) { resetInnerWildp(); }

            break;
        }

        case 802:
        case 812:
        {
            NiceAssert( !( resmode & 0x80 ) );

            if ( ia < 0 ) { setInnerWildpa(&xa,&xainfo); ia = -1; }
            if ( ib < 0 ) { setInnerWildpb(&xb,&xbinfo); ib = -2; }
            if ( ic < 0 ) { setInnerWildpc(&xc,&xcinfo); ic = -3; }
            if ( id < 0 ) { setInnerWildpd(&xd,&xdinfo); id = -4; }

            res = 0.0;

            if ( NLB()+NUB()+NF() )
            {
                double Kij;

                double Kijaa,Kijab;
                double Kijba,Kijbb;
                double Kijca,Kijcb;
                double Kijda,Kijdb;
                double Kijea,Kijeb;
                double Kijfa,Kijfb;

                int j,k,l,m;

                for ( j = 0 ; j < N() ; ++j )
                {
                    if ( alphaState()(j) )
                    {
                        for ( k = 0 ; k < N() ; ++k )
                        {
                            if ( alphaState()(k) )
                            {
                                for ( l = 0 ; l < N() ; ++l )
                                {
                                    if ( alphaState()(l) )
                                    {
                                        for ( m = 0 ; m < N() ; ++m )
                                        {
                                            if ( alphaState()(m) )
                                            {
                                                switch ( resmode & 0x7f )
                                                {
                                                    case 0:  case 1:  case 2:  case 3:
                                                    case 4:  case 5:  case 6:  case 7:
                                                    case 8:  case 9:  case 10: case 11:
                                                    case 12: case 13: case 14: case 15:
                                                    {
                                                        Kijaa = K4(ia,ib,j,k,nullptr,&xa,&xb,nullptr,nullptr,&xainfo,&xbinfo,nullptr,nullptr); Kijab = K4(ic,id,l,m,nullptr,&xa,&xb,nullptr,nullptr,&xainfo,&xbinfo,nullptr,nullptr);
                                                        Kijba = K4(ia,ic,j,k,nullptr,&xa,&xc,nullptr,nullptr,&xainfo,&xcinfo,nullptr,nullptr); Kijbb = K4(ib,id,l,m,nullptr,&xa,&xc,nullptr,nullptr,&xainfo,&xcinfo,nullptr,nullptr);
                                                        Kijca = K4(ia,id,j,k,nullptr,&xa,&xd,nullptr,nullptr,&xainfo,&xdinfo,nullptr,nullptr); Kijcb = K4(ib,ic,l,m,nullptr,&xa,&xd,nullptr,nullptr,&xainfo,&xdinfo,nullptr,nullptr);
                                                        Kijda = K4(ib,ic,j,k,nullptr,&xb,&xc,nullptr,nullptr,&xbinfo,&xcinfo,nullptr,nullptr); Kijdb = K4(ia,id,l,m,nullptr,&xb,&xc,nullptr,nullptr,&xbinfo,&xcinfo,nullptr,nullptr);
                                                        Kijea = K4(ib,id,j,k,nullptr,&xb,&xd,nullptr,nullptr,&xbinfo,&xdinfo,nullptr,nullptr); Kijeb = K4(ia,id,l,m,nullptr,&xb,&xd,nullptr,nullptr,&xbinfo,&xdinfo,nullptr,nullptr);
                                                        Kijfa = K4(ic,id,j,k,nullptr,&xc,&xd,nullptr,nullptr,&xcinfo,&xdinfo,nullptr,nullptr); Kijfb = K4(ia,ib,l,m,nullptr,&xc,&xd,nullptr,nullptr,&xcinfo,&xdinfo,nullptr,nullptr);

                                                        Kijaa *= Kijab;
                                                        Kijba *= Kijbb;
                                                        Kijca *= Kijcb;
                                                        Kijda *= Kijdb;
                                                        Kijea *= Kijeb;
                                                        Kijfa *= Kijfb;

                                                        Kij  = Kijaa;
                                                        Kij += Kijba;
                                                        Kij += Kijca;
                                                        Kij += Kijda;
                                                        Kij += Kijea;
                                                        Kij += Kijfa;

                                                        break;
                                                    }

                                                    case 64:
                                                    {
                                                        NiceThrow("K4xfer precursor second order derivatives not yet implemented.");

                                                        break;
                                                    }

                                                    default:
                                                    {
                                                        NiceThrow("K4xfer precursor specified resmode undefined at this level.");

                                                        break;
                                                    }
                                                }

                                                if ( isUnderlyingScalar() )
                                                {
                                                    Kij *= alphaR()(j)*alphaR()(k)*alphaR()(l)*alphaR()(m);
                                                }
                                        
                                                else if ( isUnderlyingVector() )
                                                {
                                                    Kij *= fourProduct(dummyr,alphaV()(j),alphaV()(k),alphaV()(l),alphaV()(m));
                                                }
                                        
                                                else
                                                {
                                                    Kij *= (double) real(fourProduct(dummy,alpha()(j),alpha()(k),alpha()(l),alpha()(m)));
                                                }

                                                res += Kij;
                                                res += Kij;
                                                res += Kij;
                                                res += Kij;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if ( ( ia < 0 ) || ( ib < 0 ) || ( ic < 0 ) || ( id < 0 ) ) { resetInnerWildp(); }

            break;
        }

        case 807:
        case 817:
        {
            gentype Kintemp;
            double dummyr;

            K1(Kintemp,xa,&xainfo);

            res = 0.0;

            if ( NLB()+NUB()+NF() )
            {
                gentype Kij;
                gentype temp;

                int j,k,l,m;

                for ( j = 0 ; j < N() ; ++j )
                {
                    if ( alphaState()(j) )
                    {
                        for ( k = 0 ; k < N() ; ++k )
                        {
                            if ( alphaState()(k) )
                            {
                                for ( l = 0 ; l < N() ; ++l )
                                {
                                    if ( alphaState()(l) )
                                    {
                                        for ( m = 0 ; m < N() ; ++m )
                                        {
                                            if ( alphaState()(m) )
                                            {
                                                K4(Kij,j,k,l,m);

                                                temp =  Kintemp;
                                                temp *= Kij;
                                                temp -= 1.0;
                                                temp = -1.0/temp;

                                                if ( isUnderlyingScalar() )
                                                {
                                                    temp *= alphaR()(j)*alphaR()(k)*alphaR()(l)*alphaR()(m);
                                                }

                                                else if ( isUnderlyingVector() )
                                                {
                                                    temp *= fourProduct(dummyr,alphaV()(j),alphaV()(k),alphaV()(l),alphaV()(m));
                                                }

                                                else
                                                {
                                                    temp *= (double) real(alpha()(j)*alpha()(k)*alpha()(l)*alpha()(m));
                                                }

                                                res += (double) temp;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            break;
        }

        default:
        {
            ML_Base::K4xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,ia,ib,ic,id,xdim,densetype,resmode,mlid);

            break;
        }
    }

    return;
}


















void SVM_Generic::Kmxfer(gentype &res, int &minmaxind, int typeis,
                         const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                         Vector<const SparseVector<gentype> *> &x,
                         Vector<const vecInfo *> &xinfo,
                         Vector<int> &iii,
                         int xdim, int m, int densetype, int resmode, int mlid) const
{
//    if ( ( m == 0 ) || ( m == 1 ) || ( m == 2 ) || ( m == 3 ) || ( m == 4 ) )
//    {
//        kernPrecursor::Kmxfer(res,minmaxind,typeis,xyprod,yxprod,diffis,x,xinfo,iii,xdim,m,densetype,resmode,mlid);
//        return;
//    }

    NiceAssert( !densetype );

    gentype dummy;

    Vector<int> i(iii);

    int iq;

    for ( iq = 0 ; iq < m ; ++iq )
    {
        i("&",iq) = (typeis-(100*(typeis/100)))/10 ? i(iq) : -42-iq;
    }

    switch ( typeis )
    {
        case 801:
        case 811:
        {
            Vector<SparseVector<gentype> > *xx = nullptr;

            if ( !( i >= 0 ) )
            {
                MEMNEW(xx,Vector<SparseVector<gentype> >(x.size()));

                int ir;

                for ( ir = 0 ; ir < x.size() ; ++ir )
                {
                    (*xx)("&",ir) = *(x(ir));
                }

                retVector<int> tmpva; 

                setInnerWildpx(xx); 
                i = cntintvec(m,tmpva);
                i += 1; 
                i *= -100;
            }

            Vector<const SparseVector<gentype> *> xa(2*m);
            Vector<const vecInfo *> xainfo(2*m);
            Vector<int> ia(2*m);

            retVector<const SparseVector<gentype> *> tmpva;
            retVector<const vecInfo *>               tmpvb;
            retVector<int>                           tmpvc;

            xa("&",0,1,m-1,tmpva)     = x;
            xa("&",m,1,(2*m)-1,tmpva) = nullptr;

            xainfo("&",0,1,m-1,tmpvb)     = xinfo;
            xainfo("&",m,1,(2*m)-1,tmpvb) = nullptr;

            ia("&",0,1,m-1,tmpvc)     = i;
            ia("&",m,1,(2*m)-1,tmpvc) = 0;

            res = 0.0;

            if ( NLB()+NUB()+NF() )
            {
                gentype Kij;

                int j;
                int isdone = 0;
                int isnz;

                while ( !isdone )
                {
                    isnz = 1;

                    for ( j = 0 ; j < m ; ++j )
                    {
                        if ( !alphaState()(ia(m+j)) )
                        {
                            isnz = 0;
                            break;
                        }
                    }

                    if ( isnz )
                    {
                        switch ( resmode & 0x7f )
                        {
                            case 0:  case 1:  case 2:  case 3:
                            case 4:  case 5:  case 6:  case 7:
                            case 8:  case 9:  case 10: case 11:
                            case 12: case 13: case 14: case 15:
                            {
                                Km(2*m,Kij,ia,nullptr,&xa,&xainfo,resmode);

                                break;
                            }

                            case 64:
                            {
                                NiceThrow("Kmxfer precursor second order derivatives not yet implemented.");

                                break;
                            }

                            default:
                            {
                                NiceThrow("Kmxfer precursor specified resmode undefined at this level.");

                                break;
                            }
                        }

                        if ( isUnderlyingScalar() )
                        {
                            for ( j = 0 ; j < m ; ++j )
                            {
                                Kij *= alphaR()(ia(j));
                            }
                        }

                        else
                        {
                            NiceThrow("m-products for vectors non pointer blah something.");
                        }

                        res += Kij;
                    }

                    isdone = 1;

                    for ( j = 0 ; j < m ; ++j )
                    {
                        ++(ia("&",m+j));

                        if ( ia(m+j) < N() )
                        {
                            isdone = 0;

                            break;
                        }

                        ia("&",m+j) = 0;
                    }
                }
            }

            if ( !( i >= 0 ) ) { resetInnerWildp(); MEMDEL(xx); }

            break;
        }

        case 802:
        case 812:
        {
            NiceAssert( !( resmode & 0x80 ) );
            NiceAssert( !(m%2) );

            Vector<SparseVector<gentype> > *xx = nullptr;

            if ( !( i >= 0 ) )
            {
                MEMNEW(xx,Vector<SparseVector<gentype> >(x.size()));

                int ir;

                for ( ir = 0 ; ir < x.size() ; ++ir )
                {
                    (*xx)("&",ir) = *(x(ir));
                }

                retVector<int> tmpva;

                setInnerWildpx(xx);
                i = cntintvec(m,tmpva);
                i += 1;
                i *= -100;
            }

            Vector<const SparseVector<gentype> *> xa(2*m);
            Vector<const vecInfo *> xainfo(2*m);
            Vector<int> ia(2*m);

            Vector<const SparseVector<gentype> *> xb(m);
            Vector<const vecInfo *> xbinfo(m);
            Vector<int> ib(m);

            Vector<const SparseVector<gentype> *> xc(m);
            Vector<const vecInfo *> xcinfo(m);
            Vector<int> ic(m);

            Vector<int> k(2*m);

            retVector<const SparseVector<gentype> *> tmpva;
            retVector<const vecInfo *>               tmpvb;
            retVector<int>                           tmpvc;

            xa("&",0,1,m-1,tmpva)     = x;
            xa("&",m,1,(2*m)-1,tmpva) = nullptr;

            xainfo("&",0,1,m-1,tmpvb)     = xinfo;
            xainfo("&",m,1,(2*m)-1,tmpvb) = nullptr;

            ia("&",0,1,m-1,tmpvc)     = i;
            ia("&",m,1,(2*m)-1,tmpvc) = 0;

            res = 0.0;

            if ( NLB()+NUB()+NF() )
            {
                gentype Kija,Kijb;

                gentype Kij;

                int j,l;
                int isdone = 0;
                int isfill = 0;
//                int ummbongo;
                int isnz,isun;

                while ( !isdone )
                {
                    isnz = 1;

                    for ( j = 0 ; j < m ; ++j )
                    {
                        if ( !alphaState()(ia(m+j)) )
                        {
                            isnz = 0;
                            break;
                        }
                    }

                    if ( isnz )
                    {
                        switch ( resmode & 0x7f )
                        {
                            case 0:  case 1:  case 2:  case 3:
                            case 4:  case 5:  case 6:  case 7:
                            case 8:  case 9:  case 10: case 11:
                            case 12: case 13: case 14: case 15:
                            {
                                k = 0;

                                isfill = 0;

                                // Outer loop over all combinations.
                                // Inner loop over all unique combinations

                                while ( !isfill )
                                {
                                    isun = 1;

                                    for ( j = 0 ; j < (2*m)-1 ; ++j )
                                    {
                                        for ( l = j+1 ; l < 2*m ; ++l )
                                        {
                                            if ( k(j) == k(l) )
                                            {
                                                isun = 0;
                                                break;
                                            }
                                        }
                                    }

                                    // Skip repetitions that dont put k halves in order.  This will also ensure nullptrs go to right

                                    if ( isun )
                                    {
                                        for ( j = 0 ; j < m-1 ; ++j )
                                        {
                                            if ( ( ib(j) >= ib(j+1) ) || ( ic(j) >= ic(j+1) ) )
                                            {
                                                isun = 0;
                                                break;
                                            }
                                        }
                                    }

                                    // Skip repetitions that don't evenly split nullptrs and non-nullptrs

                                    if ( isun && ( !xb((m/2)-1) || xb(m/2) || !xc((m/2)-1) || xc(m/2) ) )
                                    {
                                        isun = 0;
                                    }

                                    if ( isun )
                                    {
                                        for ( j = 0 ; j < m ; ++j )
                                        {
                                            xb("&",j)     = xa(k(j));
                                            xbinfo("&",j) = xainfo(k(j));
                                            ib("&",j)     = ia(k(j));

                                            xc("&",j)     = xa(k(j+m));
                                            xcinfo("&",j) = xainfo(k(j+m));
                                            ic("&",j)     = ia(k(j+m));
                                        }

                                        Km(m,Kija,ib,nullptr,&xb,&xbinfo,resmode);
                                        Km(m,Kijb,ic,nullptr,&xc,&xcinfo,resmode);

                                        Kija *= Kijb;

                                        Kij += Kija;
                                    }

                                    isfill = 1;

                                    for ( j = 0 ; j < 2*m ; ++j )
                                    {
                                        ++(k("&",j));

                                        if ( k(j) < 2*m )
                                        {
                                            isfill = 0;

                                            break;
                                        }

                                        k("&",j) = 0;
                                    }
                                }

                                break;
                            }

                            case 64:
                            {
                                NiceThrow("Kmxfer precursor second order derivatives not yet implemented.");

                                break;
                            }

                            default:
                            {
                                NiceThrow("Kmxfer precursor specified resmode undefined at this level.");

                                break;
                            }
                        }

                        if ( isUnderlyingScalar() )
                        {
                            for ( j = 0 ; j < m ; ++j )
                            {
                                Kij *= alphaR()(ia(j));
                            }
                        }

                        else
                        {
                            NiceThrow("m-products for vectors non pointer blah something.");
                        }

                        res += Kij;
                    }

                    isdone = 1;

                    for ( j = 0 ; j < m ; ++j )
                    {
                        ++(ia("&",m+j));

                        if ( ia(m+j) < N() )
                        {
                            isdone = 0;

                            break;
                        }

                        ia("&",m+j) = 0;
                    }
                }
            }

            if ( !( i >= 0 ) ) { resetInnerWildp(); MEMDEL(xx); }

            break;
        }

        case 807:
        case 817:
        {
            NiceThrow("Really?  Hyperkernels of order >4?");

            break;
        }

        default:
        {
            ML_Base::Kmxfer(res,minmaxind,typeis,xyprod,yxprod,diffis,x,xinfo,i,xdim,m,densetype,resmode,mlid);

            break;
        }
    }

    return;
}

void SVM_Generic::Kmxfer(double &res, int &minmaxind, int typeis,
                         double xyprod, double yxprod, double diffis,
                         Vector<const SparseVector<gentype> *> &x,
                         Vector<const vecInfo *> &xinfo,
                         Vector<int> &iii,
                         int xdim, int m, int densetype, int resmode, int mlid) const
{
//    if ( ( m == 0 ) || ( m == 1 ) || ( m == 2 ) || ( m == 3 ) || ( m == 4 ) )
//    {
//        kernPrecursor::Kmxfer(res,minmaxind,typeis,xyprod,yxprod,diffis,x,xinfo,iii,xdim,m,densetype,resmode,mlid);
//        return;
//    }

    NiceAssert( !densetype );

    if ( ( resmode & 0x7f ) >= 16 )
    {
        gentype temp;

        gentype gxyprod(xyprod);
        gentype gyxprod(yxprod);
        gentype gdiffis(diffis);

        Kmxfer(temp,minmaxind,typeis,gxyprod,gyxprod,gdiffis,x,xinfo,iii,xdim,m,densetype,resmode,mlid);

        return;
    }

    gentype dummy;

    Vector<int> i(iii);

    int iq;

    for ( iq = 0 ; iq < m ; ++iq )
    {
        i("&",iq) = (typeis-(100*(typeis/100)))/10 ? i(iq) : -42-iq;
    }

    switch ( typeis )
    {
        case 801:
        case 811:
        {
            Vector<SparseVector<gentype> > *xx = nullptr;

            if ( !( i >= 0 ) )
            {
                MEMNEW(xx,Vector<SparseVector<gentype> >(x.size()));

                int ir;

                for ( ir = 0 ; ir < x.size() ; ++ir )
                {
                    (*xx)("&",ir) = *(x(ir));
                }

                retVector<int> tmpva; 

                setInnerWildpx(xx); 
                i = cntintvec(m,tmpva);
                i += 1; 
                i *= -100;
            }

            Vector<const SparseVector<gentype> *> xa(2*m);
            Vector<const vecInfo *> xainfo(2*m);
            Vector<int> ia(2*m);

            retVector<const SparseVector<gentype> *> tmpva;
            retVector<const vecInfo *>               tmpvb;
            retVector<int>                           tmpvc;

            xa("&",0,1,m-1,tmpva)     = x;
            xa("&",m,1,(2*m)-1,tmpva) = nullptr;

            xainfo("&",0,1,m-1,tmpvb)     = xinfo;
            xainfo("&",m,1,(2*m)-1,tmpvb) = nullptr;

            ia("&",0,1,m-1,tmpvc)     = i;
            ia("&",m,1,(2*m)-1,tmpvc) = 0;

            res = 0.0;

            if ( NLB()+NUB()+NF() )
            {
                double Kij = 0.0;

                int j;
                int isdone = 0;
                int isnz;

                while ( !isdone )
                {
                    isnz = 1;

                    for ( j = 0 ; j < m ; ++j )
                    {
                        if ( !alphaState()(ia(m+j)) )
                        {
                            isnz = 0;
                            break;
                        }
                    }

                    if ( isnz )
                    {
                        switch ( resmode & 0x7f )
                        {
                            case 0:  case 1:  case 2:  case 3:
                            case 4:  case 5:  case 6:  case 7:
                            case 8:  case 9:  case 10: case 11:
                            case 12: case 13: case 14: case 15:
                            {
                                Kij = Km(2*m,ia,nullptr,&xa,&xainfo,resmode);

                                break;
                            }

                            case 64:
                            {
                                NiceThrow("Kmxfer precursor second order derivatives not yet implemented.");

                                break;
                            }

                            default:
                            {
                                NiceThrow("Kmxfer precursor specified resmode undefined at this level.");

                                break;
                            }
                        }

                        if ( isUnderlyingScalar() )
                        {
                            for ( j = 0 ; j < m ; ++j )
                            {
                                Kij *= alphaR()(ia(j));
                            }
                        }

                        else
                        {
                            NiceThrow("m-products for vectors non pointer blah something.");
                        }

                        res += Kij;
                    }

                    isdone = 1;

                    for ( j = 0 ; j < m ; ++j )
                    {
                        ++(ia("&",m+j));

                        if ( ia(m+j) < N() )
                        {
                            isdone = 0;

                            break;
                        }

                        ia("&",m+j) = 0;
                    }
                }
            }

            if ( !( i >= 0 ) ) { resetInnerWildp(); MEMDEL(xx); }

            break;
        }

        case 802:
        case 812:
        {
            NiceAssert( !( resmode & 0x80 ) );
            NiceAssert( !(m%2) );

            Vector<SparseVector<gentype> > *xx = nullptr;

            if ( !( i >= 0 ) )
            {
                MEMNEW(xx,Vector<SparseVector<gentype> >(x.size()));

                int ir;

                for ( ir = 0 ; ir < x.size() ; ++ir )
                {
                    (*xx)("&",ir) = *(x(ir));
                }

                retVector<int> tmpva;

                setInnerWildpx(xx);
                i = cntintvec(m,tmpva);
                i += 1;
                i *= -100;
            }

            Vector<const SparseVector<gentype> *> xa(2*m);
            Vector<const vecInfo *> xainfo(2*m);
            Vector<int> ia(2*m);

            Vector<const SparseVector<gentype> *> xb(m);
            Vector<const vecInfo *> xbinfo(m);
            Vector<int> ib(m);

            Vector<const SparseVector<gentype> *> xc(m);
            Vector<const vecInfo *> xcinfo(m);
            Vector<int> ic(m);

            Vector<int> k(2*m);

            retVector<const SparseVector<gentype> *> tmpva;
            retVector<const vecInfo *>               tmpvb;
            retVector<int>                           tmpvc;

            xa("&",0,1,m-1,tmpva)     = x;
            xa("&",m,1,(2*m)-1,tmpva) = nullptr;

            xainfo("&",0,1,m-1,tmpvb)     = xinfo;
            xainfo("&",m,1,(2*m)-1,tmpvb) = nullptr;

            ia("&",0,1,m-1,tmpvc)     = i;
            ia("&",m,1,(2*m)-1,tmpvc) = 0;

            res = 0.0;

            if ( NLB()+NUB()+NF() )
            {
                double Kija,Kijb;

                double Kij = 0.0;

                int j,l;
                int isdone = 0;
                int isfill = 0;
                int isnz,isun;

                while ( !isdone )
                {
                    isnz = 1;

                    for ( j = 0 ; j < m ; ++j )
                    {
                        if ( !alphaState()(ia(m+j)) )
                        {
                            isnz = 0;
                            break;
                        }
                    }

                    if ( isnz )
                    {
                        switch ( resmode & 0x7f )
                        {
                            case 0:  case 1:  case 2:  case 3:
                            case 4:  case 5:  case 6:  case 7:
                            case 8:  case 9:  case 10: case 11:
                            case 12: case 13: case 14: case 15:
                            {
                                k = 0;

                                isfill = 0;

                                // Outer loop over all combinations.
                                // Inner loop over all unique combinations

                                while ( !isfill )
                                {
                                    isun = 1;

                                    for ( j = 0 ; j < (2*m)-1 ; ++j )
                                    {
                                        for ( l = j+1 ; l < 2*m ; ++l )
                                        {
                                            if ( k(j) == k(l) )
                                            {
                                                isun = 0;
                                                break;
                                            }
                                        }
                                    }

                                    if ( isun )
                                    {
                                        for ( j = 0 ; j < m ; ++j )
                                        {
                                            xb("&",j)     = xa(k(j));
                                            xbinfo("&",j) = xainfo(k(j));
                                            ib("&",j)     = ia(k(j));

                                            xc("&",j)     = xa(k(j+m));
                                            xcinfo("&",j) = xainfo(k(j+m));
                                            ic("&",j)     = ia(k(j+m));
                                        }

                                        Kija = Km(m,ib,nullptr,&xb,&xbinfo,resmode);
                                        Kijb = Km(m,ic,nullptr,&xc,&xcinfo,resmode);

                                        Kija *= Kijb;

                                        Kij += Kija;
                                    }

                                    isfill = 1;

                                    for ( j = 0 ; j < 2*m ; ++j )
                                    {
                                        ++(k("&",j));

                                        if ( k(j) < 2*m )
                                        {
                                            isfill = 0;

                                            break;
                                        }

                                        k("&",j) = 0;
                                    }
                                }

                                break;
                            }

                            case 64:
                            {
                                NiceThrow("Kmxfer precursor second order derivatives not yet implemented.");

                                break;
                            }

                            default:
                            {
                                NiceThrow("Kmxfer precursor specified resmode undefined at this level.");

                                break;
                            }
                        }

                        if ( isUnderlyingScalar() )
                        {
                            for ( j = 0 ; j < m ; ++j )
                            {
                                Kij *= alphaR()(ia(j));
                            }
                        }

                        else
                        {
                            NiceThrow("m-products for vectors non pointer blah something.");
                        }

                        res += Kij;
                    }

                    isdone = 1;

                    for ( j = 0 ; j < m ; ++j )
                    {
                        ++(ia("&",m+j));

                        if ( ia(m+j) < N() )
                        {
                            isdone = 0;

                            break;
                        }

                        ia("&",m+j) = 0;
                    }
                }
            }

            if ( !( i >= 0 ) ) { resetInnerWildp(); MEMDEL(xx); }

            break;
        }

        case 807:
        case 817:
        {
            NiceThrow("Seriously?  Hyperkernels of order >4?");

            break;
        }

        default:
        {
            ML_Base::Kmxfer(res,minmaxind,typeis,xyprod,yxprod,diffis,x,xinfo,i,xdim,m,densetype,resmode,mlid);

            break;
        }
    }

    return;
}

















void SVM_Generic::basesetAlphaBiasFromAlphaBiasR(void)
{
    incgvernum();

    //int Nval = SVM_Generic::N();
    int Nval = alphaR().size();

    dalpha.resize(Nval);
    xalphaState.resize(Nval);

    xalphaState = 1;

    if ( Nval )
    {
        int i;

        for ( i = 0 ; i < Nval ; ++i )
        {
            dalpha("&",i).force_double() = alphaR()(i);
        }
    }

    dbias.force_double() = biasR();

    return;
}

void SVM_Generic::basesetAlphaBiasFromAlphaBiasV(void)
{
    incgvernum();

    //int Nval = SVM_Generic::N();
    int Nval = alphaV().size();

    dalpha.resize(Nval);
    xalphaState.resize(Nval);

    xalphaState = 1;

    if ( Nval )
    {
        int i;

        for ( i = 0 ; i < Nval ; ++i )
        {
            dalpha("&",i) = alphaV()(i);
        }
    }

    dbias = biasV();

    return;
}

void SVM_Generic::basesetAlphaBiasFromAlphaBiasA(void)
{
    incgvernum();

    //int Nval = SVM_Generic::N();
    int Nval = alphaA().size();

    dalpha.resize(Nval);
    xalphaState.resize(Nval);

    xalphaState = 1;

    if ( Nval )
    {
        int i;

        for ( i = 0 ; i < Nval ; ++i )
        {
            dalpha("&",i).force_anion() = alphaA()(i);
        }
    }

    dbias.force_anion() = biasA();

    return;
}

int SVM_Generic::removeNonSupports(void)
{
    int i;
    int res = 0;

    gentype y;
    SparseVector<gentype> x;

    while ( NZ() )
    {
	minabs(alphaState(),i);
        res |= removeTrainingVector(i,y,x);
    }

    return res;
}

int SVM_Generic::trimTrainingSet(int maxsize)
{
    NiceAssert( maxsize >= 0 );

    int i;
    int res = 0;

    gentype y;
    SparseVector<gentype> x;

    while ( SVM_Generic::N() > maxsize )
    {
	if ( NZ() )
	{
	    minabs(alphaState(),i);
            res |= removeTrainingVector(i,y,x);
	}

	else
	{
            minabs(alpha(),i);
            res |= removeTrainingVector(i,y,x);
	}
    }

    return res;
}

void SVM_Generic::dgTrainingVector(Vector<gentype> &res, gentype &resn, int i) const
{
    res.resize(SVM_Generic::N());

    if ( gOutType() == 'V' )
    {
        gentype zerotemplate('V');

        zerotemplate.dir_vector().resize(tspaceDim());

        setzero(zerotemplate);

        res  = zerotemplate;
        resn = zerotemplate;

        if ( SVM_Generic::N() )
        {
            int j;
            int dummyind;

            gentype xscale('R');
            gentype yscale('R');

            for ( j = 0 ; j < SVM_Generic::N() ; ++j )
            {
                if ( d()(j) )
                {
                    dK2delx(xscale,yscale,dummyind,i,j);

                    NiceAssert( dummyind < 0 );

                    if ( i >= 0 )
                    {
                        res("&",i).dir_vector().scaleAdd((double) xscale,(const Vector<gentype> &) dalpha(j));
                        res("&",j).dir_vector().scaleAdd((double) yscale,(const Vector<gentype> &) dalpha(j));
                    }

                    else
                    {
                        resn      .dir_vector().scaleAdd((double) xscale,(const Vector<gentype> &) dalpha(j));
                        res("&",j).dir_vector().scaleAdd((double) yscale,(const Vector<gentype> &) dalpha(j));
                    }
                }
            }
        }
    }

    else if ( gOutType() == 'A' )
    {
        gentype zerotemplate('A');

        zerotemplate.dir_anion().setorder(order());

        setzero(zerotemplate);

        res  = zerotemplate;
        resn = zerotemplate;

        if ( SVM_Generic::N() )
        {
            int j;
            int dummyind;

            gentype xscale('R');
            gentype yscale('R');

            for ( j = 0 ; j < SVM_Generic::N() ; ++j )
            {
                if ( d()(j) )
                {
                    dK2delx(xscale,yscale,dummyind,i,j);

                    NiceAssert( dummyind < 0 );

                    if ( i >= 0 )
                    {
                        res("&",i).dir_anion() += ((double) xscale)*((const d_anion &) dalpha(j));
                        res("&",j).dir_anion() += ((double) yscale)*((const d_anion &) dalpha(j));
                    }

                    else
                    {
                        resn      .dir_anion() += ((double) xscale)*((const d_anion &) dalpha(j));
                        res("&",j).dir_anion() += ((double) yscale)*((const d_anion &) dalpha(j));
                    }
                }
            }
        }
    }

    else
    {
        //gentype zerotemplate(0.0);

        res  = 0.0_gent;
        resn = 0.0_gent;

        if ( SVM_Generic::N() )
        {
            int j;
            int dummyind = -1;

            gentype xscale('R');
            gentype yscale('R');

            for ( j = 0 ; j < SVM_Generic::N() ; ++j )
            {
                if ( d()(j) )
                {
                    dK2delx(xscale,yscale,dummyind,i,j);

                    NiceAssert( dummyind < 0 );

                    if ( i >= 0 )
                    {
                        res("&",i).dir_double() += ((double) xscale)*((double) dalpha(j));
                        res("&",j).dir_double() += ((double) yscale)*((double) dalpha(j));
                    }

                    else
                    {
                        resn      .dir_double() += ((double) xscale)*((double) dalpha(j));
                        res("&",j).dir_double() += ((double) yscale)*((double) dalpha(j));
                    }
                }
            }
        }
    }

    return;
}

int SVM_Generic::setAlpha(const Vector<gentype> &newAlpha)
{
    incgvernum();

    //NiceAssert( newAlpha.size() == SVM_Generic::N() );

    // Does not set alpha, relies on function callback to internalsetAlpha

    int res = 0;

    if ( SVM_Generic::N() )
    {
        int i,j;

        if ( isUnderlyingScalar() )
        {
            Vector<double> nAlpha(newAlpha.size());

            for ( i = 0 ; i < nAlpha.size() ; ++i )
            {
                nAlpha("&",i) = ((double) newAlpha(i));
            }

            res |= setAlphaR(nAlpha);
        }

        else if ( isUnderlyingVector() )
        {
            Vector<Vector<double> > nAlpha(newAlpha.size());

            for ( i = 0 ; i < nAlpha.size() ; ++i )
            {
                nAlpha("&",i).resize(newAlpha(i).size());

                if ( newAlpha(i).size() )
                {
                    const Vector<gentype> &ghgh = (const Vector<gentype> &) newAlpha(i);

                    for ( j = 0 ; j < nAlpha(i).size() ; ++j )
                    {
                        nAlpha("&",i)("&",j) = ghgh(j);
                    }
                }
            }

            res |= setAlphaV(nAlpha);
        }

        else
        {
            Vector<d_anion> nAlpha(newAlpha.size());

            for ( i = 0 ; i < nAlpha.size() ; ++i )
            {
                nAlpha("&",i) = ((const d_anion &) newAlpha(i));
            }

            res |= setAlphaA(nAlpha);
        }
    }

    return res;
}

int SVM_Generic::setBias(const gentype &newBias)
{
    incgvernum();

    // Does not set alpha, relies on function callback to internalsetBias

    int res = 0;

    if ( isUnderlyingScalar() )
    {
        res |= setBiasR(((double) newBias));
    }

    else if ( isUnderlyingVector() )
    {
        int i;

        Vector<double> temp(newBias.size());

        if ( newBias.size() )
        {
            const Vector<gentype> &ghgh = (const Vector<gentype> &) newBias;

            for ( i = 0 ; i < newBias.size() ; ++i )
            {
                temp("&",i) = ghgh(i);
            }
        }

        res |= setBiasV(temp);
    }

    else
    {
        res |= setBiasA((const d_anion &) newBias);
    }

    return res;
}


int SVM_Generic::prealloc(int expectedN)
{
    dalpha.prealloc(expectedN);
    xalphaState.prealloc(expectedN);
    ML_Base::prealloc(expectedN);

    return 0;
}

int SVM_Generic::preallocsize(void) const
{
    return ML_Base::preallocsize();
}




int SVM_Generic::egetparam(int ind, Vector<gentype> &val, const Vector<gentype> &xa, int ia, const Vector<gentype> &xb, int ib) const
{
    int k,res = 0;
    const char *dummy = nullptr;

    NiceAssert( xa.size() == xb.size() );

    val.resize(xa.size());

    for ( k = 0 ; k < xa.size() ; ++k )
    {
        res |= getparam(ind,val("&",k),xa(k),ia,xb(k),ib,dummy);
    }

    return res;
}

int SVM_Generic::getparam(int ind, gentype &val, const gentype &xa, int ia, const gentype &xb, int ib, charptr &desc) const
{
    int res = 0;

    desc = "";

    // Function to access parameters via indexing

    NiceAssert( ia >= 0 );
    NiceAssert( ib >= 0 );

    val.force_null();

    switch ( ind )
    {
        case 9000: { val = NZ();                          desc = "SVM_Scalar::NZ"; break; }
        case 9001: { val = NF();                          desc = "SVM_Scalar::NF"; break; }
        case 9002: { val = NS();                          desc = "SVM_Scalar::NS"; break; }
        case 9003: { val = NC();                          desc = "SVM_Scalar::NC"; break; }
        case 9004: { val = NLB();                         desc = "SVM_Scalar::NLB"; break; }
        case 9005: { val = NLF();                         desc = "SVM_Scalar::NLF"; break; }
        case 9006: { val = NUF();                         desc = "SVM_Scalar::NUF"; break; }
        case 9007: { val = NUB();                         desc = "SVM_Scalar::NUB"; break; }
        case 9008: { val = isLinearCost();                desc = "SVM_Scalar::isLinearCost"; break; }
        case 9009: { val = isQuadraticCost();             desc = "SVM_Scalar::isQuadraticCost"; break; }
        case 9010: { val = is1NormCost();                 desc = "SVM_Scalar::is1NormCost"; break; }
        case 9011: { val = isVarBias();                   desc = "SVM_Scalar::isVarBias"; break; }
        case 9012: { val = isPosBias();                   desc = "SVM_Scalar::isPosBias"; break; }
        case 9013: { val = isNegBias();                   desc = "SVM_Scalar::isNegBias"; break; }
        case 9014: { val = isFixedBias();                 desc = "SVM_Scalar::isFixedBias"; break; }
        case 9015: { val = isOptActive();                 desc = "SVM_Scalar::isOptActive"; break; }
        case 9016: { val = isOptSMO();                    desc = "SVM_Scalar::isOptSMO"; break; }
        case 9017: { val = isOptD2C();                    desc = "SVM_Scalar::isOptD2C"; break; }
        case 9018: { val = isOptGrad();                   desc = "SVM_Scalar::isOptGrad"; break; }
        case 9019: { val = isFixedTube();                 desc = "SVM_Scalar::isFixedTube"; break; }
        case 9020: { val = isShrinkTube();                desc = "SVM_Scalar::isShrinkTube"; break; }
        case 9021: { val = isRestrictEpsPos();            desc = "SVM_Scalar::isRestrictEpsPos"; break; }
        case 9022: { val = isRestrictEpsNeg();            desc = "SVM_Scalar::isRestrictEpsNeg"; break; }
        case 9023: { val = isClassifyViaSVR();            desc = "SVM_Scalar::isClassifyViaSVR"; break; }
        case 9024: { val = isClassifyViaSVM();            desc = "SVM_Scalar::isClassifyViaSVM"; break; }
        case 9025: { val = is1vsA();                      desc = "SVM_Scalar::is1vsA"; break; }
        case 9026: { val = is1vs1();                      desc = "SVM_Scalar::is1vs1"; break; }
        case 9027: { val = isDAGSVM();                    desc = "SVM_Scalar::isDAGSVM"; break; }
        case 9028: { val = isMOC();                       desc = "SVM_Scalar::isMOC"; break; }
        case 9029: { val = ismaxwins();                   desc = "SVM_Scalar::ismaxwins"; break; }
        case 9030: { val = isrecdiv();                    desc = "SVM_Scalar::isrecdiv"; break; }
        case 9031: { val = isatonce();                    desc = "SVM_Scalar::isatonce"; break; }
        case 9032: { val = isredbin();                    desc = "SVM_Scalar::isredbin"; break; }
        case 9033: { val = isKreal();                     desc = "SVM_Scalar::isKreal"; break; }
        case 9034: { val = isKunreal();                   desc = "SVM_Scalar::isKunreal"; break; }
        case 9035: { val = isanomalyOn();                 desc = "SVM_Scalar::isanomalyOn"; break; }
        case 9036: { val = isanomalyOff();                desc = "SVM_Scalar::isanomalyOff"; break; }
        case 9037: { val = isautosetOff();                desc = "SVM_Scalar::isautosetOff"; break; }
        case 9038: { val = isautosetCscaled();            desc = "SVM_Scalar::isautosetCscaled"; break; }
        case 9039: { val = isautosetCKmean();             desc = "SVM_Scalar::isautosetCKmean"; break; }
        case 9040: { val = isautosetCKmedian();           desc = "SVM_Scalar::isCKmedian"; break; }
        case 9041: { val = isautosetCNKmean();            desc = "SVM_Scalar::isCNKmean"; break; }
        case 9042: { val = isautosetCNKmedian();          desc = "SVM_Scalar::isCNKmedian"; break; }
        case 9043: { val = isautosetLinBiasForce();       desc = "SVM_Scalar::isautosetLinBiasForce"; break; }
        case 9044: { val = outerlr();                     desc = "SVM_Scalar::outerlr"; break; }
        case 9045: { val = outermom();                    desc = "SVM_Scalar::outermom"; break; }
        case 9046: { val = outermethod();                 desc = "SVM_Scalar::outermethod"; break; }
        case 9047: { val = outertol();                    desc = "SVM_Scalar::outertol"; break; }
        case 9048: { val = outerovsc();                   desc = "SVM_Scalar::outerovsc"; break; }
        case 9049: { val = outermaxitcnt();               desc = "SVM_Scalar::outermaxitcnt"; break; }
        case 9050: { val = outermaxcache();               desc = "SVM_Scalar::outermaxcache"; break; }
        case 9051: { val = maxiterfuzzt();                desc = "SVM_Scalar::maxiterfuzzt"; break; }
        case 9052: { val = usefuzzt();                    desc = "SVM_Scalar::usefuzzt"; break; }
        case 9053: { val = lrfuzzt();                     desc = "SVM_Scalar::lrfuzzt"; break; }
        case 9054: { val = ztfuzzt();                     desc = "SVM_Scalar::ztfuzzt"; break; }
        case 9055: { val = costfnfuzzt();                 desc = "SVM_Scalar::costfnfuzzt"; break; }
        case 9056: { val = m();                           desc = "SVM_Scalar::m"; break; }
        case 9057: { val = LinBiasForce();                desc = "SVM_Scalar::LinBiasForce"; break; }
        case 9058: { val = QuadBiasForce();               desc = "SVM_Scalar::QuadBiasForce"; break; }
        case 9059: { val = nu();                          desc = "SVM_Scalar::nu"; break; }
        case 9060: { val = nuQuad();                      desc = "SVM_Scalar::nuQuad"; break; }
        case 9061: { val = theta();                       desc = "SVM_Scalar::theta"; break; }
        case 9062: { val = simnorm();                     desc = "SVM_Scalar::simnorm"; break; }
        case 9063: { val = anomalyNu();                   desc = "SVM_Scalar::anomalyNu"; break; }
        case 9064: { val = anomalyClass();                desc = "SVM_Scalar::anomalyClass"; break; }
        case 9065: { val = autosetCval();                 desc = "SVM_Scalar::autosetCval"; break; }
        case 9066: { val = autosetnuval();                desc = "SVM_Scalar::autosetnuval"; break; }
        case 9067: { val = anomclass();                   desc = "SVM_Scalar::anomclass"; break; }
        case 9068: { val = singmethod();                  desc = "SVM_Scalar::singmethod"; break; }
        case 9069: { val = rejectThreshold();             desc = "SVM_Scalar::rejectThreshold"; break; }
        case 9070: { val = Gp();                          desc = "SVM_Scalar::Gp"; break; }
        case 9071: { val = XX();                          desc = "SVM_Scalar::XX"; break; }
        case 9072: { val = kerndiag();                    desc = "SVM_Scalar::kerndiag"; break; }
        case 9073: { val = bias();                        desc = "SVM_Scalar::bias"; break; }
        case 9074: { val = alpha();                       desc = "SVM_Scalar::alpha"; break; }
        case 9075: { val = loglikelihood();               desc = "SVM_Scalar::loglikelihood"; break; }
        case 9076: { val = isNoMonotonicConstraints();    desc = "SVM_Scalar::isNoMonotonicCostraints"; break; }
        case 9077: { val = isForcedMonotonicIncreasing(); desc = "SVM_Scalar::isForcedMonotonicIncreasing"; break; }
        case 9078: { val = isForcedMonotonicDecreasing(); desc = "SVM_Scalar::isForcedMonotonicDecreasing"; break; }
        case 9079: { val = NRff();                        desc = "SVM_Scalar::NRff"; break; }
        case 9080: { val = NRffRep();                     desc = "SVM_Scalar::NRffRep"; break; }
        case 9081: { val = ReOnly();                      desc = "SVM_Scalar::ReOnly"; break; }
        case 9082: { val = inAdam();                      desc = "SVM_Scalar::inAdam"; break; }
        case 9083: { val = outGrad();                     desc = "SVM_Scalar::outGrad"; break; }
        case 9084: { val = D();                           desc = "SVM_Scalar::D"; break; }
        case 9085: { val = E();                           desc = "SVM_Scalar::E"; break; }
        case 9086: { val = F();                           desc = "SVM_Scalar::F"; break; }
        case 9087: { val = G();                           desc = "SVM_Scalar::G"; break; }
        case 9088: { val = tunev();                       desc = "SVM_Scalar::tunev"; break; }
        case 9089: { val = minv();                        desc = "SVM_Scalar::minv"; break; }

        case 9100: { val = NF((int) xa);            desc = "SVM_Scalar::NF";            break; }
        case 9101: { val = NZ((int) xa);            desc = "SVM_Scalar::NZ";            break; }
        case 9102: { val = NS((int) xa);            desc = "SVM_Scalar::NS";            break; }
        case 9103: { val = NC((int) xa);            desc = "SVM_Scalar::NC";            break; }
        case 9104: { val = NLB((int) xa);           desc = "SVM_Scalar::NLB";           break; }
        case 9105: { val = NLF((int) xa);           desc = "SVM_Scalar::NLF";           break; }
        case 9106: { val = NUF((int) xa);           desc = "SVM_Scalar::NUF";           break; }
        case 9107: { val = NUB((int) xa);           desc = "SVM_Scalar::NUB";           break; }
        case 9108: { val = ClassRep()((int) xa);    desc = "SVM_Scalar::ClassRep";      break; }
        case 9109: { val = findID((int) xa);        desc = "SVM_Scalar::findID";        break; }
        case 9110: { val = getu()((int) xa);        desc = "SVM_Scalar::getu";          break; }
        case 9111: { val = isVarBias((int) xa);     desc = "SVM_Scalar::isVarBias";     break; }
        case 9112: { val = isPosBias((int) xa);     desc = "SVM_Scalar::isPosBias";     break; }
        case 9113: { val = isNegBias((int) xa);     desc = "SVM_Scalar::isNegBias";     break; }
        case 9114: { val = isFixedBias((int) xa);   desc = "SVM_Scalar::isFixedBias";   break; }
        case 9115: { val = LinBiasForceclass((int) xa);  desc = "SVM_Scalar::LinBiasForceclass";  break; }
        case 9116: { val = QuadBiasForceclass((int) xa); desc = "SVM_Scalar::QuadBiasForceclass"; break; }

        case 9200: { val = Gp()((int) xa, (int) xb); desc = "SVM_Scalar::Gp"; break; }
        case 9201: { val = XX()((int) xa, (int) xb); desc = "SVM_Scalar::XX"; break; }

        default:
        {
            res = ML_Base::getparam(ind,val,xa,ia,xb,ib,desc);

            break;
        }
    }

    return res;
}

