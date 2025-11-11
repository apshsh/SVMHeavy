
//
// Improvement measure base class
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
#include "imp_generic.hpp"
#include "hyper_base.hpp"


std::ostream &IMP_Generic::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "zref:                           " << xzref      << "\n";
    repPrint(output,'>',dep) << "EHI method:                     " << xehimethod << "\n";
    repPrint(output,'>',dep) << "Scalarisation method:           " << xscaltype  << "\n";
    repPrint(output,'>',dep) << "Scalarisation alpha:            " << xscalalpha << "\n";
    repPrint(output,'>',dep) << "Override x dimension:           " << zxdim      << "\n";
    repPrint(output,'>',dep) << "Number of samples in TS sample: " << xNsamp     << "\n";
    repPrint(output,'>',dep) << "Sample slack in TS sample:      " << xsampSlack << "\n";
    repPrint(output,'>',dep) << "isTrained:                      " << disTrained << "\n";

    getQconst().printstream(output,dep+1);

    return output;
}

std::istream &IMP_Generic::inputstream(std::istream &input )
{
    wait_dummy dummy;

    input >> dummy; input >> xzref;
    input >> dummy; input >> xehimethod;
    input >> dummy; input >> xscaltype;
    input >> dummy; input >> xscalalpha;
    input >> dummy; input >> zxdim;
    input >> dummy; input >> xNsamp;
    input >> dummy; input >> xsampSlack;
    input >> dummy; input >> disTrained;

    getQ().inputstream(input);

    return input;
}


IMP_Generic::IMP_Generic(int _isIndPrune) : ML_Base_Deref(_isIndPrune)
{
    //setaltx(nullptr);

    xzref      = 0; // typically, at top level, we minimise f(x) in [-1,0]^d.  This is negated to get what we model here, so -f(x) in [0,1]^d > 0, so zref = 0
    xehimethod = 0;
    xscaltype  = 0;
    xscalalpha = 0.1;
    zxdim      = -1;
    xNsamp     = 20;
    xsampSlack = 0; // actually works better this way 0.5;
    disTrained = 0;

    return;
}

IMP_Generic::IMP_Generic(const IMP_Generic &src, int _isIndPrune) : ML_Base_Deref(_isIndPrune)
{
    //setaltx(nullptr);

    assign(src,0);

    return;
}

IMP_Generic::IMP_Generic(const IMP_Generic &src, const ML_Base *, int _isIndPrune) : ML_Base_Deref(_isIndPrune)
{
    //setaltx(xsrc);

    assign(src,-1);

    return;
}

IMP_Generic::~IMP_Generic()
{
    return;
}

int IMP_Generic::addTrainingVector(int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh, double epsweigh, int dval)
{
    untrain();
    getQ().addTrainingVector(i,y,x,Cweigh,epsweigh,dval);

    return 1;
}

int IMP_Generic::qaddTrainingVector(int i, const gentype &y, SparseVector<gentype> &x, double Cweigh, double epsweigh, int dval)
{
    untrain();
    getQ().qaddTrainingVector(i,y,x,Cweigh,epsweigh,dval);

    return 1;
}

int IMP_Generic::addTrainingVector(int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    NiceAssert( y.size() == x.size() );
    NiceAssert( y.size() == Cweigh.size() );
    NiceAssert( y.size() == epsweigh.size() );

    if ( y.size() )
    {
        untrain();
        getQ().addTrainingVector(i,y,x,Cweigh,epsweigh);
    }

    return 1;
}

int IMP_Generic::qaddTrainingVector(int i, const Vector<gentype> &y, Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    NiceAssert( y.size() == x.size() );
    NiceAssert( y.size() == Cweigh.size() );
    NiceAssert( y.size() == epsweigh.size() );

    if ( y.size() )
    {
        untrain();
        getQ().qaddTrainingVector(i,y,x,Cweigh,epsweigh);
    }

    return 1;
}

int IMP_Generic::removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x)
{
    untrain();
    getQ().removeTrainingVector(i,y,x);

    return 1;
}

int IMP_Generic::removeTrainingVector(int i, int num)
{
    untrain();
    getQ().removeTrainingVector(i,num);

    return 1;
}

int IMP_Generic::setx(int i, const SparseVector<gentype> &x)
{
    untrain();
    getQ().setx(i,x);

    return 1;
}

int IMP_Generic::setx(const Vector<int> &i, const Vector<SparseVector<gentype> > &x)
{
    untrain();
    getQ().setx(i,x);

    return 1;
}

int IMP_Generic::setx(const Vector<SparseVector<gentype> > &x)
{
    untrain();
    getQ().setx(x);

    return 1;
}

int IMP_Generic::qswapx(int i, SparseVector<gentype> &x, int dontupdate)
{
    untrain();
    getQ().qswapx(i,x,dontupdate);

    return 1;
}

int IMP_Generic::qswapx(const Vector<int> &i, Vector<SparseVector<gentype> > &x, int dontupdate)
{
    untrain();
    getQ().qswapx(i,x,dontupdate);

    return 1;
}

int IMP_Generic::qswapx(Vector<SparseVector<gentype> > &x, int dontupdate)
{
    untrain();
    getQ().qswapx(x,dontupdate);

    return 1;
}

int IMP_Generic::setd(int i, int nd)
{
    untrain();
    getQ().setd(i,nd);

    return 1;
}

int IMP_Generic::setd(const Vector<int> &i, const Vector<int> &nd)
{
    untrain();
    getQ().setd(i,nd);

    return 1;
}

int IMP_Generic::setd(const Vector<int> &nd)
{
    untrain();
    getQ().setd(nd);

    return 1;
}

double IMP_Generic::hypervol(void) const
{
    double retval = 0;
    gentype xminval;

    if ( N()-NNC(0) )
    {
        if ( xspaceDim() == 1 )
        {
            int i;
            gentype temp;

            xelm(xminval,0,0);

            for ( i = 1 ; i < N() ; ++i )
            {
                if ( isenabled(i) )
                {
                    if ( xelm(temp,i,0) < xminval )
                    {
                        xminval = temp;
                    }
                }
            }

            retval =  (double) xminval;
            retval -= zref();
        }

        else if ( xspaceDim() > 1 )
        {
            int M = N()-NNC(0);
            int n = xspaceDim();
            gentype temp;

            double **X;

            MEMNEWARRAY(X,double *,M+1);

            int i,j,k;

            for ( i = 0, j = 0 ; i < N() ; ++i )
            {
                if ( isenabled(i) )
                {
                    MEMNEWARRAY(X[j],double,xspaceDim());

                    for ( k = 0 ; k < xspaceDim() ; ++k )
                    {
                        xelm(temp,i,k);
                        X[j][k] = -(((double) temp)-zref());
                    }

                    ++j;
                }
            }

            retval = h(X,M,n);

            for ( i = 0, j = 0 ; i < N() ; ++i )
            {
                if ( isenabled(i) )
                {
                    MEMDELARRAY(X[j]); X[j] = nullptr;

                    ++j;
                }
            }

            MEMDELARRAY(X); X = nullptr;
        }
    }

    return retval;
}







int IMP_Generic::egetparam(int ind, Vector<gentype> &val, const Vector<gentype> &xa, int ia, const Vector<gentype> &xb, int ib) const
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

int IMP_Generic::getparam(int ind, gentype &val, const gentype &xa, int ia, const gentype &xb, int ib, charptr &desc) const
{
    int res = 0;

    desc = "";

    // Function to access parameters via indexing

    NiceAssert( ia >= 0 );
    NiceAssert( ib >= 0 );

    if ( ( ( ind >= 3000 ) && ( ind <= 3099 ) && !ia && !ib ) || ( ( ind >= 3100 ) && ( ind <= 3199 ) && !ib ) || ( ind <= 2999 ) )
    {
        switch ( ind )
        {
            case 3000: { val = zref();      desc = "IMP_Generic::zref"; break; }
            case 3001: { val = ehimethod(); desc = "IMP_Generic::ehimethod"; break; }
            case 3002: { val = needdg();    desc = "IMP_Generic::needdg"; break; }
            case 3003: { val = hypervol();  desc = "IMP_Generic::hypervol"; break; }

            case 3100:
            {
                SparseVector<gentype> xx;

                if ( convertSetToSparse(xx,xa,ia) )
                {
                    res = 1;
                }

                else
                {
                    if ( !ib )
                    {
                        gentype dummy;
                        imp(val,dummy,xx,xb);
                    }

                    else
                    {
                        val.force_null();
                    }
                }

                desc = "IMP_Generic::imp";

                break;
            }

            default:
            {
                res = getQconst().getparam(ind,val,xa,ia,xb,ib,desc);

                break;
            }
        }
    }

    else
    {
        val.force_null();
    }

    return res;
}

