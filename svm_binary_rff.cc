
//
// Binary Classification RFF SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "svm_binary_rff.hpp"
#include <iostream>
#include <sstream>
#include <string>
#include <typeinfo>


class SVM_Single;

// the zcalc macro effectively calculates what we want gp to be to take into
// account both z and tube factors, allowing hp == 0 which results in faster
// optimisation.  The functions below are those that need to use this translation
// macro.

#define DCALC(_d_)                                             ( ( isClassifyViaSVR() && (_d_) ) ? 2 : (_d_) )
#define ZCALC(_z_,_d_,_E_,_Eclass_,_xclass_,_Eweigh_)          ( ( isFixedTube() ? ( (_z_) + ( (_d_) * (_Eclass_)(((_xclass_)+1)/2) * (_E_) * (_Eweigh_) ) ) : (_z_) + ( isClassifyViaSVR() ? (_d_) : 0 ) ) )
#define CWCALC(_xclass_,_Cweight_,_Cweightfuzz_,_dthres_)      ( (_Cweight_) * (_Cweightfuzz_) * (binCclass((_xclass_)+1)) * ( ( ( (_dthres_) > 0 ) && ( (_dthres_) < 0.5 ) ) ? ( ( 1 - (_dthres_) ) / (_dthres_) ) : 1 ) )
#define CWCALCBASE(_xclass_,_Cweight_,_Cweightfuzz_,_dthres_)  ( (_Cweight_) * (_Cweightfuzz_) * (binCclass((_xclass_)+1)) )
#define CWCALCEXTRA(_xclass_,_Cweight_,_Cweightfuzz_,_dthres_) ( (_Cweight_) * (_Cweightfuzz_) * (binCclass((_xclass_)+1)) * ( ( ( (_dthres_) > 0 ) && ( (_dthres_) < 0.5 ) ) ? ( ( 1 - (_dthres_) ) / (_dthres_) ) - 1 : 0 ) )

#define DEFAULT_DTHRES 0.0

SVM_Binary_rff::SVM_Binary_rff() : SVM_Scalar_rff()
{
    setaltx(nullptr);

    SVM_Scalar_rff::seteps(0);
    bineps = 1;
    binepsclass.resize(3);
    binepsclass = 1.0;
    binCclass.resize(3);
    binCclass = 1.0;
    isSVMviaSVR = 1; // Default for SVM_Scalar_rff
    binNnc.resize(4);
    binNnc = 0;
    SVM_Scalar_rff::setRestrictEpsNeg();





    dthres = DEFAULT_DTHRES;

    return;
}

SVM_Binary_rff::SVM_Binary_rff(const SVM_Binary_rff &src) : SVM_Scalar_rff()
{
    setaltx(nullptr);

    SVM_Scalar_rff::seteps(0);
    bineps = 1;
    binepsclass.resize(3);
    binepsclass = 1.0;
    binCclass.resize(3);
    binCclass = 1.0;
    isSVMviaSVR = 1; // default for SVM_Scalar_rff
    binNnc.resize(4);
    binNnc = 0;
    SVM_Scalar_rff::setRestrictEpsNeg();





    dthres = DEFAULT_DTHRES;

    assign(src,0);

    return;
}

SVM_Binary_rff::SVM_Binary_rff(const SVM_Binary_rff &src, const ML_Base *xsrc) : SVM_Scalar_rff()
{
    setaltx(nullptr);

    SVM_Scalar_rff::seteps(0);
    bineps = 1;
    binepsclass.resize(3);
    binepsclass = 1.0;
    binCclass.resize(3);
    binCclass = 1.0;
    isSVMviaSVR = 1; // Default for SVM_Scalar_rff
    binNnc.resize(4);
    binNnc = 0;
    SVM_Scalar_rff::setRestrictEpsNeg();





    dthres = DEFAULT_DTHRES;
    setaltx(xsrc);
    assign(src,-1);

    return;
}

SVM_Binary_rff::~SVM_Binary_rff()
{
    return;
}

double SVM_Binary_rff::calcDist(const gentype &ha, const gentype &hb, int ia, int db) const
{
    (void) ia;

    double res = 0;

    if ( db )
    {
        res = ( ( (int) ha ) != ( (int) hb ) ) ? 1 : 0;
    }

    return res;
}

int SVM_Binary_rff::setC(double xC)
{

    return SVM_Scalar_rff::setC(xC);
}

int SVM_Binary_rff::seteps(double xeps)
{
    int i;
    int res = 0;






    bineps = xeps;

    if ( isFixedTube() )
    {
        if ( SVM_Binary_rff::N() )
	{
            for ( i = 0 ; i < SVM_Binary_rff::N() ; ++i )
	    {
                res |= SVM_Scalar_rff::sety(i,ZCALC(bintraintarg(i),bintrainclass(i),bineps,binepsclass,bintrainclass(i),binepsweight(i)));
                gentype yn(bintrainclass(i));
                res |= SVM_Generic::sety(i,yn);
	    }
	}
    }

    else
    {
	if ( isClassifyViaSVR() )
	{
            res |= SVM_Scalar_rff::seteps(bineps);
	}

	else
	{
            res |= SVM_Scalar_rff::seteps(-bineps);
	}
    }

    return res;
}

int SVM_Binary_rff::setepsclass(int d, double xeps)
{
    NiceAssert( ( d == -1 ) || ( d == 0 ) || ( d == +1 ) );

    int i;

    binepsclass("&",d+1) = xeps;

    int res = SVM_Scalar_rff::setepsclass(d,xeps);

    if ( isFixedTube() )
    {
        if ( SVM_Binary_rff::N() )
	{
            for ( i = 0 ; i < SVM_Binary_rff::N() ; ++i )
	    {
                if ( bintrainclass(i) == d )
		{
                    res |= SVM_Scalar_rff::sety(i,ZCALC(bintraintarg(i),bintrainclass(i),bineps,binepsclass,bintrainclass(i),binepsweight(i)));
                    gentype yn(bintrainclass(i));
                    res |= SVM_Generic::sety(i,yn);
		}
	    }
	}
    }

    return res;
}

void SVM_Binary_rff::setalleps(double xeps, const Vector<double> &xepsclass)
{
    NiceAssert( xepsclass.size() == 3 );

    int i;

    seteps(xeps);

    for ( i = 0 ; i < 3 ; ++i )
    {
        setepsclass(i,xepsclass(i));
    }

    return;
}

int SVM_Binary_rff::scaleepsweight(double scalefactor)
{
    int res = 0;

    if ( SVM_Binary_rff::N() )
    {
	int i;

        for ( i = 0 ; i < SVM_Binary_rff::N() ; ++i )
	{
            res |= setepsweight(i,binepsweight(i)*scalefactor);
	}
    }

    return res;
}

int SVM_Binary_rff::scaleCweight(double scalefactor)
{
    int res = 0;

    if ( SVM_Binary_rff::N() )
    {
	int i;

        for ( i = 0 ; i < SVM_Binary_rff::N() ; ++i )
	{
            res |= setCweight(i,binCweight(i)*scalefactor);
	}
    }

    return res;
}

int SVM_Binary_rff::scaleCweightfuzz(double scalefactor)
{
    int res = 0;

    if ( SVM_Binary_rff::N() )
    {
	int i;

        for ( i = 0 ; i < SVM_Binary_rff::N() ; ++i )
	{
            res |= setCweightfuzz(i,binCweightfuzz(i)*scalefactor);
	}
    }

    return res;
}

int SVM_Binary_rff::setCclass(int d, double xC)
{
    NiceAssert( ( d == -1 ) || ( d == 0 ) || ( d == +1 ) );

    int i;
    int res = 0;

    binCclass("&",d+1) = xC;

    if ( SVM_Binary_rff::N() )
    {
        for ( i = 0 ; i < SVM_Binary_rff::N() ; ++i )
	{
            if ( bintrainclass(i) == d )
	    {
                res |= SVM_Scalar_rff::setCweight(i,CWCALC(bintrainclass(i),binCweight(i),binCweightfuzz(i),dthres));
	    }
	}
    }

    return res;
}
























int SVM_Binary_rff::sety(int i, const gentype &zn)
{
    NiceAssert( zn.isCastableToIntegerWithoutLoss() );

    return setd(i,(int) zn);
}

int SVM_Binary_rff::sety(const Vector<int> &j, const Vector<gentype> &yn)
{
    NiceAssert( j.size() == yn.size() );

    int res = 0;

    if ( j.size() )
    {
        int i;

        for ( i = 0 ; i < j.size() ; ++i )
        {
            res |= sety(j(i),yn(i));
        }
    }

    return res;
}

int SVM_Binary_rff::sety(const Vector<gentype> &yn)
{
    NiceAssert( SVM_Binary_rff::N() == yn.size() );

    int res = 0;

    if ( SVM_Binary_rff::N() )
    {
        int i;

        for ( i = 0 ; i < SVM_Binary_rff::N() ; ++i )
        {
            res |= sety(i,yn(i));
        }
    }

    return res;
}

int SVM_Binary_rff::setd(int i, int xd)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < SVM_Binary_rff::N() );
    NiceAssert( ( xd == -1 ) || ( xd == 0 ) || ( xd == +1 ) );

    int res = 0;

    if ( xd != bintrainclass(i) )
    {
        res = 1;



        setdinternal(i,xd);





    }

    return res;
}

int SVM_Binary_rff::sety(int i, double xz)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < SVM_Binary_rff::N() );

    int res = 0;

    bintraintarg("&",i) = xz;

    res |= SVM_Scalar_rff::sety(i,ZCALC(bintraintarg(i),bintrainclass(i),bineps,binepsclass,bintrainclass(i),binepsweight(i)));
    gentype yn(bintrainclass(i));
    res |= SVM_Generic::sety(i,yn);

    return res;
}

int SVM_Binary_rff::setCweight(int i, double xCweight)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < SVM_Binary_rff::N() );

    binCweight("&",i) = xCweight;

    return SVM_Scalar_rff::setCweight(i,CWCALC(bintrainclass(i),binCweight(i),binCweightfuzz(i),dthres));
}

int SVM_Binary_rff::setCweightfuzz(int i, double xCweight)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < SVM_Binary_rff::N() );

    binCweightfuzz("&",i) = xCweight;

    return SVM_Scalar_rff::setCweight(i,CWCALC(bintrainclass(i),binCweight(i),binCweightfuzz(i),dthres));
}

int SVM_Binary_rff::setepsweight(int i, double xepsweight)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < SVM_Binary_rff::N() );

    binepsweight("&",i) = xepsweight;

    int res = SVM_Scalar_rff::setepsweight(i,xepsweight);

    if ( isFixedTube() )
    {
        res |= SVM_Scalar_rff::sety(i,ZCALC(bintraintarg(i),bintrainclass(i),bineps,binepsclass,bintrainclass(i),binepsweight(i)));
        gentype yn(bintrainclass(i));
        res |= SVM_Generic::sety(i,yn);
    }

    return res;
}

int SVM_Binary_rff::setd(const Vector<int> &j, const Vector<int> &d)
{
    NiceAssert( j.size() == d.size() );

    int res = 0;

    if ( j.size() )
    {
        res = 1;

        int i;

        for ( i = 0 ; i < j.size() ; ++i )
	{
            res |= setdinternal(j(i),d(i));
	}


    }

    return res;
}

int SVM_Binary_rff::sety(const Vector<int> &j, const Vector<double> &xz)
{
    NiceAssert( j.size() == xz.size() );

    int res = 0;

    retVector<double> tmpva;

    bintraintarg("&",j,tmpva) = xz;

    if ( j.size() )
    {
	int i;

        for ( i = 0 ; i < j.size() ; ++i )
	{
            res |= SVM_Scalar_rff::sety(j(i),ZCALC(bintraintarg(j(i)),bintrainclass(j(i)),bineps,binepsclass,bintrainclass(j(i)),binepsweight(j(i))));
            gentype yn(bintrainclass(j(i)));
            res |= SVM_Generic::sety(j(i),yn);
	}
    }

    return res;
}

int SVM_Binary_rff::setCweight(const Vector<int> &j, const Vector<double> &xCweight)
{
    NiceAssert( xCweight.size() == j.size() );

    int res = 0;

    if ( j.size() )
    {
        int i;

        for ( i = 0 ; i < j.size() ; ++i )
	{
            res |= setCweight(j(i),xCweight(i));
	}
    }

    return res;
}

int SVM_Binary_rff::setCweightfuzz(const Vector<int> &j, const Vector<double> &xCweight)
{
    NiceAssert( xCweight.size() == j.size() );

    int res = 0;

    if ( j.size() )
    {
        int i;

        for ( i = 0 ; i < j.size() ; ++i )
	{
            res |= setCweightfuzz(j(i),xCweight(i));
	}
    }

    return res;
}

int SVM_Binary_rff::setepsweight(const Vector<int> &j, const Vector<double> &xepsweight)
{
    NiceAssert( xepsweight.size() == j.size() );

    int res = 0;

    if ( j.size() )
    {
        int i;

        for ( i = 0 ; i < j.size() ; ++i )
	{
            res |= setepsweight(j(i),xepsweight(i));
	}
    }

    return res;
}

int SVM_Binary_rff::setd(const Vector<int> &d)
{
    NiceAssert( d.size() == SVM_Binary_rff::N() );

    int i;
    int res = 0;

    if ( SVM_Binary_rff::N() )
    {
        for ( i = 0 ; i < SVM_Binary_rff::N() ; ++i )
	{
            res |= setdinternal(i,d(i));
	}


    }

    return res;
}

int SVM_Binary_rff::sety(const Vector<double> &xz)
{
    NiceAssert( bintraintarg.size() == xz.size() );

    int res = 0;

    bintraintarg = xz;

    if ( SVM_Binary_rff::N() )
    {
	int i;

        for ( i = 0 ; i < SVM_Binary_rff::N() ; ++i )
	{
            res |= SVM_Scalar_rff::sety(i,ZCALC(bintraintarg(i),bintrainclass(i),bineps,binepsclass,bintrainclass(i),binepsweight(i)));
            gentype yn(bintrainclass(i));
            res |= SVM_Generic::sety(i,yn);
	}
    }

    return res;
}

int SVM_Binary_rff::setCweight(const Vector<double> &xCweight)
{
    NiceAssert( xCweight.size() == SVM_Binary_rff::N() );

    int res = 0;

    int i;

    if ( SVM_Binary_rff::N() )
    {
        for ( i = 0 ; i < SVM_Binary_rff::N() ; ++i )
	{
            res |= setCweight(i,xCweight(i));
	}
    }

    return res;
}

int SVM_Binary_rff::setCweightfuzz(const Vector<double> &xCweight)
{
    NiceAssert( xCweight.size() == SVM_Binary_rff::N() );

    int res = 0;

    int i;

    if ( SVM_Binary_rff::N() )
    {
        for ( i = 0 ; i < SVM_Binary_rff::N() ; ++i )
	{
            res |= setCweightfuzz(i,xCweight(i));
	}
    }

    return res;
}

int SVM_Binary_rff::setepsweight(const Vector<double> &xepsweight)
{
    NiceAssert( xepsweight.size() == SVM_Binary_rff::N() );

    int i;
    int res = 0;

    if ( SVM_Binary_rff::N() )
    {
        for ( i = 0 ; i < SVM_Binary_rff::N() ; ++i )
	{
            res |= setepsweight(i,xepsweight(i));
	}
    }

    return res;
}

int SVM_Binary_rff::setdinternal(int i, int xd)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < SVM_Binary_rff::N() );
    NiceAssert( ( xd == -1 ) || ( xd == 0 ) || ( xd == +1 ) );

    int res = 0;

    if ( xd != bintrainclass(i) )
    {
        res = 1;

        --(binNnc("&",bintrainclass(i)+1));
        ++(binNnc("&",xd+1));

        bintrainclass("&",i) = xd;

        res |= SVM_Scalar_rff::setd(i,DCALC(xd));
        res |= SVM_Scalar_rff::sety(i,ZCALC(bintraintarg(i),bintrainclass(i),bineps,binepsclass,bintrainclass(i),binepsweight(i)));
        gentype yn(bintrainclass(i));
        res |= SVM_Generic::sety(i,yn);
        res |= SVM_Scalar_rff::setCweight(i,CWCALC(bintrainclass(i),binCweight(i),binCweightfuzz(i),dthres));
    }

    return res;
}





































































int SVM_Binary_rff::addTrainingVector(int i, const gentype &z, const SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    return SVM_Binary_rff::addTrainingVector(i,(int) z,x,Cweigh,epsweigh);
}

int SVM_Binary_rff::qaddTrainingVector(int i, const gentype &z, SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    return SVM_Binary_rff::qaddTrainingVector(i,(int) z,x,Cweigh,epsweigh);
}

int SVM_Binary_rff::addTrainingVector(int i, const Vector<gentype> &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    Vector<int> zz(z.size());
    Vector<double> xz(z.size());

    xz = 0.0;

    if ( z.size() )
    {
        int j;

        for ( j = 0 ; j < z.size() ; ++j )
        {
            zz("&",j) = (int) z(j);
        }
    }

    return SVM_Binary_rff::addTrainingVector(i,zz,x,Cweigh,epsweigh,xz);
}

int SVM_Binary_rff::qaddTrainingVector(int i, const Vector<gentype> &z, Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    Vector<int> zz(z.size());
    Vector<double> xz(z.size());

    xz = 0.0;

    if ( z.size() )
    {
        int j;

        for ( j = 0 ; j < z.size() ; ++j )
        {
            zz("&",j) = (int) z(j);
        }
    }

    return SVM_Binary_rff::qaddTrainingVector(i,zz,x,Cweigh,epsweigh,xz);
}

int SVM_Binary_rff::addTrainingVector(int i, int xd, const SparseVector<gentype> &x, double xCweigh, double xepsweigh, double xz)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= SVM_Binary_rff::N() );

    ++(binNnc("&",xd+1));

    bintrainclass.add(i);
    bintraintarg.add(i);
    binepsweight.add(i);
    binCweight.add(i);
    binCweightfuzz.add(i);

    bintrainclass("&",i)  = xd;
    bintraintarg("&",i)   = xz;
    binepsweight("&",i)   = xepsweigh;
    binCweight("&",i)     = xCweigh;
    binCweightfuzz("&",i) = 1.0;

    int res = SVM_Scalar_rff::addTrainingVector(i,gentype(ZCALC(bintraintarg(i),bintrainclass(i),bineps,binepsclass,bintrainclass(i),binepsweight(i))),x,CWCALC(bintrainclass(i),binCweight(i),binCweightfuzz(i),dthres),binepsweight(i),DCALC(bintrainclass(i)));



    gentype yn(bintrainclass(i));
    res |= SVM_Generic::sety(i,yn);

    return res;
}

int SVM_Binary_rff::qaddTrainingVector(int i, int xd, SparseVector<gentype> &x, double xCweigh, double xepsweigh, double xz)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= SVM_Binary_rff::N() );

    ++(binNnc("&",xd+1));

    bintrainclass.add(i);
    bintraintarg.add(i);
    binepsweight.add(i);
    binCweight.add(i);
    binCweightfuzz.add(i);

    bintrainclass("&",i)  = xd;
    bintraintarg("&",i)   = xz;
    binepsweight("&",i)   = xepsweigh;
    binCweight("&",i)     = xCweigh;
    binCweightfuzz("&",i) = 1.0;

    int res = SVM_Scalar_rff::qaddTrainingVector(i,gentype(ZCALC(bintraintarg(i),bintrainclass(i),bineps,binepsclass,bintrainclass(i),binepsweight(i))),x,CWCALC(bintrainclass(i),binCweight(i),binCweightfuzz(i),dthres),binepsweight(i),DCALC(bintrainclass(i)));



    gentype yn(bintrainclass(i));
    res |= SVM_Generic::sety(i,yn);

    return res;
}

int SVM_Binary_rff::addTrainingVector(int i, const Vector<int> &xd, const Vector<SparseVector<gentype> > &xx, const Vector<double> &xCweigh, const Vector<double> &xepsweigh, const Vector<double> &xz)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= SVM_Binary_rff::N() );
    NiceAssert( xd.size() == xx.size() );
    NiceAssert( xd.size() == xCweigh.size() );
    NiceAssert( xd.size() == xepsweigh.size() );
    NiceAssert( xd.size() == xz.size() );

    int res = 0;

    if ( xd.size() )
    {
        int j;

        for ( j = 0 ; j < xd.size() ; ++j )
        {
            ++(binNnc("&",xd(j)+1));

            bintrainclass.add(i+j);
            bintraintarg.add(i+j);
            binepsweight.add(i+j);
            binCweight.add(i+j);
            binCweightfuzz.add(i+j);

            bintrainclass("&",i+j)  = xd(j);
            bintraintarg("&",i+j)   = xz(j);
            binepsweight("&",i+j)   = xepsweigh(j);
            binCweight("&",i+j)     = xCweigh(j);
            binCweightfuzz("&",i+j) = 1.0;

            res |= SVM_Scalar_rff::addTrainingVector(i+j,gentype(ZCALC(bintraintarg(i+j),bintrainclass(i+j),bineps,binepsclass,bintrainclass(i+j),binepsweight(i+j))),xx(j),CWCALC(bintrainclass(i+j),binCweight(i+j),binCweightfuzz(i+j),dthres),binepsweight(i+j),DCALC(bintrainclass(i+j)));

            gentype yn(bintrainclass(i+j));
            res |= SVM_Generic::sety(i+j,yn);
        }
    }



    return res;
}

int SVM_Binary_rff::qaddTrainingVector(int i, const Vector<int> &xd, Vector<SparseVector<gentype> > &xx, const Vector<double> &xCweigh, const Vector<double> &xepsweigh, const Vector<double> &xz)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= SVM_Binary_rff::N() );
    NiceAssert( xd.size() == xx.size() );
    NiceAssert( xd.size() == xCweigh.size() );
    NiceAssert( xd.size() == xepsweigh.size() );
    NiceAssert( xd.size() == xz.size() );

    int res = 0;

    if ( xd.size() )
    {
        int j;

        for ( j = 0 ; j < xd.size() ; ++j )
        {
            ++(binNnc("&",xd(j)+1));

            bintrainclass.add(i+j);
            bintraintarg.add(i+j);
            binepsweight.add(i+j);
            binCweight.add(i+j);
            binCweightfuzz.add(i+j);

            bintrainclass("&",i+j)  = xd(j);
            bintraintarg("&",i+j)   = xz(j);
            binepsweight("&",i+j)   = xepsweigh(j);
            binCweight("&",i+j)     = xCweigh(j);
            binCweightfuzz("&",i+j) = 1.0;

            res |= SVM_Scalar_rff::qaddTrainingVector(i+j,gentype(ZCALC(bintraintarg(i+j),bintrainclass(i+j),bineps,binepsclass,bintrainclass(i+j),binepsweight(i+j))),xx("&",j),CWCALC(bintrainclass(i+j),binCweight(i+j),binCweightfuzz(i+j),dthres),binepsweight(i+j),DCALC(bintrainclass(i+j)));

            gentype yn(bintrainclass(i+j));
            res |= SVM_Generic::sety(i+j,yn);
        }
    }



    return res;
}

int SVM_Binary_rff::removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < SVM_Binary_rff::N() );

    --(binNnc("&",bintrainclass(i)+1));

    bintrainclass.remove(i);
    bintraintarg.remove(i);
    binepsweight.remove(i);
    binCweight.remove(i);
    binCweightfuzz.remove(i);

    int res = SVM_Scalar_rff::removeTrainingVector(i,y,x);



    return res;
}

int SVM_Binary_rff::setClassifyViaSVR(void)
{
    int res = 0;

    if ( !isClassifyViaSVR() )
    {
	isSVMviaSVR = 1;
        //res |= SVM_Scalar_rff::setRestrictEpsPos();

	int i;

        for ( i = 0 ; i < SVM_Binary_rff::N() ; ++i )
	{
            res |= SVM_Scalar_rff::setd(i,DCALC(bintrainclass(i))); // No point using setdinternal here, as autosets in the scalar are not used
            res |= SVM_Scalar_rff::sety(i,ZCALC(bintraintarg(i),bintrainclass(i),bineps,binepsclass,bintrainclass(i),binepsweight(i)));
            gentype yn(bintrainclass(i));
            res |= SVM_Generic::sety(i,yn);
	}





    }

    return res;
}

int SVM_Binary_rff::setClassifyViaSVM(void)
{
    int res = 0;

    if ( !isClassifyViaSVM() )
    {
        isSVMviaSVR = 0;
        res |= SVM_Scalar_rff::setRestrictEpsNeg();

	int i;

        for ( i = 0 ; i < SVM_Binary_rff::N() ; ++i )
	{
            res |= SVM_Scalar_rff::setd(i,DCALC(bintrainclass(i))); // No point using setdinternal here, as autosets in the scalar base are not used
            res |= SVM_Scalar_rff::sety(i,ZCALC(bintraintarg(i),bintrainclass(i),bineps,binepsclass,bintrainclass(i),binepsweight(i)));
            gentype yn(bintrainclass(i));
            res |= SVM_Generic::sety(i,yn);
	}

	if ( isShrinkTube() )
	{
            res |= SVM_Scalar::seteps(-bineps);
	}
    }

    return res;
}








































































int SVM_Binary_rff::train(int &res, svmvolatile int &killSwitch)
{
    int dobartlett = ( ( dthres > 0 ) && ( dthres < 0.5 ) ) ? 1 : 0;
    int realN = N();
    int fakeN = 0;

    if ( dobartlett && N() )
    {
        int i;

        SparseVector<gentype> xnew;

        for ( i = 0 ; i < realN ; ++i )
        {
            if ( ( d()(i) == -1 ) || ( d()(i) == +1 ) )
            {
                xnew.f4("&",0) = i;

                addTrainingVector(realN+fakeN,d()(i),xnew,1.0,0.0); // eps == 0 for this one, assume z = 0

                ++fakeN;
            }
        }
    }

    int modmod = loctrain(res,killSwitch,realN);

    if ( dobartlett )
    {
        SVM_Generic::removeTrainingVector(realN,fakeN);
    }

    return modmod;
}

int SVM_Binary_rff::loctrain(int &res, svmvolatile int &killSwitch, int realN, int assumeDNZ)
{
    int dobartlett = ( ( dthres > 0 ) && ( dthres < 0.5 ) ) ? 1 : 0;

    Vector<double> altalpha;

    if ( dobartlett && N() )
    {
        int fakeN = 0;

        altalpha = alphaR();

        // Need to enforce usual cost on 0 <= y_i.g(x_i) <= 1
        // plus additional cost (Cweightmult) on y_i.g(x_i) <= 0
        //
        // That is (Bar4, section 2, not including existing C and Cweight):
        //
        // phi(z) = 1-z   if 0 <= y_i.g(x_i) <= 1
        //          1-az  if y_i.g(x_i) <= 0
        //        = standard_phi(z) + extra_phi(z)
        //
        // standard_phi(z) = 1-z      if y_i.g(x_i) <= 1
        // extra_phi   (z) = (1-a).z  if y_i.g(x_i) <= 0
        //
        // where: a = (1-d)/d
        //        a-1 = (1-2d)/d
        //
        // We assume that epsilon == 1 here for simplicity

        int i;

        SparseVector<gentype> xnew;

        for ( i = 0 ; i < realN ; ++i )
        {
            if ( assumeDNZ || ( d()(i) == -1 ) || ( d()(i) == +1 ) )
            {
                xnew.f4("&",0) = i;

                SVM_Scalar_rff::setCweight(realN+fakeN,CWCALCEXTRA(bintrainclass(i),binCweight(i),binCweightfuzz(i),dthres));
                SVM_Scalar_rff::setCweight(i,          CWCALCBASE( bintrainclass(i),binCweight(i),binCweightfuzz(i),dthres));

                altalpha("&",realN+fakeN) = altalpha(i)-alphaR()(i);
                altalpha("&",i)           = alphaR()(i);

                ++fakeN;
            }
        }
    }

    int modmod = SVM_Scalar_rff::train(res,killSwitch);

    if ( dobartlett )
    {
        int fakeN = 0;

        altalpha = alphaR();

        int i;

        for ( i = 0 ; i < realN ; ++i )
        {
            if ( assumeDNZ || ( d()(i) == -1 ) || ( d()(i) == +1 ) )
            {
                altalpha("&",i) = (SVM_Scalar_rff::alphaR())(i) + (SVM_Scalar_rff::alphaR())(realN+fakeN);

                SVM_Scalar_rff::setCweight(i,CWCALC(bintrainclass(i),binCweight(i),binCweightfuzz(i),dthres));

                ++fakeN;
            }
        }

        SVM_Scalar_rff::setAlphaR(altalpha);

        SVM_Scalar_rff::isStateOpt = 1;
    }














    return modmod;
}

int SVM_Binary_rff::ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg, gentype ***pxyprodi) const
{
    NiceAssert( !( retaltg & 2 ) );

    int unusedvar = 0;
    int tempresh = 0;
    double tempresg = 0;

    tempresh = SVM_Scalar_rff::gTrainingVector(tempresg,unusedvar,i,retaltg,pxyprodi);

    if ( ( tempresg > -dthres ) && ( tempresg < dthres ) )
    {
        // Bartlett's "classify with reject option"

        tempresh = 0;
    }

    resh = tempresh;

    if ( retaltg & 1 )
    {
        gentype negres(-tempresg);
        gentype posres(tempresg);

        Vector<gentype> tempresg(2);

        tempresg("&",0) = negres;
        tempresg("&",1) = posres;

        resg = tempresg;
    }

    else
    {
        resg = tempresg;
    }

    return tempresh;
}

double SVM_Binary_rff::eTrainingVector(int i) const
{
    double res = 0;

    if ( ( i < 0 ) || isenabled(i) )
    {
        double yval;
        int unusedvar = 0;

        gTrainingVector(yval,unusedvar,i);

        // e = 1/(1+exp(yg))
        // dedg = -y.exp(yg)/((1+exp(yg))^2)
        // d2edg2 = -(y^2).exp(yg)/((1+exp(yg))^2) + -y.exp(yg).-2y.exp(yg)/((1+exp(yg))^3)
        //        = -(y^2).exp(yg)/((1+exp(yg))^2) + 2.(y^2).exp(2yg)/((1+exp(yg))^3)
        //        = -(y^2).exp(yg).( 1 - 2.exp(yg)/(1+exp(yg)) )/((1+exp(yg))^2)
        //        = -(y^2).exp(yg).( (1+exp(yg)-2.exp(yg))/(1+exp(yg)) )/((1+exp(yg))^2)
        //        = -(y^2).exp(yg).( (1-exp(yg))/(1+exp(yg)) )/((1+exp(yg))^2)
        //        = -(y^2).exp(yg).(1-exp(yg))/((1+exp(yg))^3)
        //        = -exp(yg).(1-exp(yg))/((1+exp(yg))^3)

        double dval = (int) y()(i);
        double eyd = exp(yval*dval);

        res = 1/(1+eyd);

/*
        if ( isQuadraticCost() )
        {
            res = norm2(yval)/2.0;
        }

        else
        {
            res = abs1(yval);
        }
*/
    }

    return res;
}

double &SVM_Binary_rff::dedgTrainingVector(double &res, int i) const
{
//    res.resize(tspaceDim()) = 0.0;

    res = 0;

    if ( ( i < 0 ) || isenabled(i) )
    {
        double yval;
        int unusedvar = 0;

        gTrainingVector(yval,unusedvar,i);
        //Assume no arbitrary biasing

        // dedg = -y.exp(yg)/((1+exp(yg))^2)

        double dval = (int) y()(i);
        double eyd = exp(yval*dval);

        res = -dval*eyd/((1+eyd)*(1+eyd));

/*
        if ( isQuadraticCost() )
        {
            ;
        }

        else
        {
            int j;

            for ( j = 0 ; j < res.size() ; ++j )
            {
                res("&",j) = sgn(res(j));
            }
        }
*/
    }

    return res;
}

double &SVM_Binary_rff::d2edg2TrainingVector(double &res, int i) const
{
//    double res = 0;

    res = 0;

    if ( ( i < 0 ) || isenabled(i) )
    {
        double yval;
        int unusedvar = 0;

        gTrainingVector(yval,unusedvar,i);

        // d2edg2 = -exp(yg).(1-exp(yg))/((1+exp(yg))^3)

        double dval = (int) y()(i);
        double eyd = exp(yval*dval);

        res = -eyd*(1-eyd)/((1+eyd)*(1+eyd)*(1+eyd));

/*
        if ( isQuadraticCost() )
        {
            res = norm2(yval)/2.0;
        }

        else
        {
            res = abs1(yval);
        }
*/
    }

/*
    int unusedvar = 0;
    res = 0;

    if ( ( i < 0 ) || ( Q.alphaRestrict(i) == 0 ) )
    {
        if ( isQuadraticCost() )
        {
            res = 1;
        }
    }

    else if ( Q.alphaRestrict(i) == 1 )
    {
        gTrainingVector(res,unusedvar,i);

        if ( res < zR(i) )
        {
            if ( isQuadraticCost() )
            {
                res = 1;
            }

            else
            {
                res = 0;
            }
        }

        else
        {
            res = 0;
        }
    }

    else if ( Q.alphaRestrict(i) == 2 )
    {
        gTrainingVector(res,unusedvar,i);

        if ( res > zR(i) )
        {
            if ( isQuadraticCost() )
            {
                res = 1;
            }

            else
            {
                res = 0;
            }
        }

        else
        {
            res = 0;
        }
    }

    else
    {
        res = 0;
    }
*/

    return res;
}

Matrix<double> &SVM_Binary_rff::dedKTrainingVector(Matrix<double> &res) const
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

Vector<double> &SVM_Binary_rff::dedKTrainingVector(Vector<double> &res, int i) const
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









































































std::ostream &SVM_Binary_rff::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Binary SVM RFF\n\n";

    repPrint(output,'>',dep) << "epsilon:                         " << bineps                 << "\n";
    repPrint(output,'>',dep) << "d:                               " << bintrainclass          << "\n";
    repPrint(output,'>',dep) << "z:                               " << bintraintarg           << "\n";
    repPrint(output,'>',dep) << "classwise epsilon:               " << binepsclass            << "\n";
    repPrint(output,'>',dep) << "elementwise epsilon:             " << binepsweight           << "\n";
    repPrint(output,'>',dep) << "classwise C:                     " << binCclass              << "\n";
    repPrint(output,'>',dep) << "elementwise C:                   " << binCweight             << "\n";
    repPrint(output,'>',dep) << "elementwise C (fuzz):            " << binCweightfuzz         << "\n";
    repPrint(output,'>',dep) << "SVM as SVR:                      " << isSVMviaSVR            << "\n";
    repPrint(output,'>',dep) << "Nnc:                             " << binNnc                 << "\n";
    repPrint(output,'>',dep) << "dthres:                          " << dthres                 << "\n";



    repPrint(output,'>',dep) << "=====================================================================\n";
    repPrint(output,'>',dep) << "Base SVR:                        ";
    SVM_Scalar_rff::printstream(output,dep+1);
    repPrint(output,'>',dep) << "\n";
    repPrint(output,'>',dep) << "---------------------------------------------------------------------\n";

    return output;
}

std::istream &SVM_Binary_rff::inputstream(std::istream &input)
{
    wait_dummy dummy;

    input >> dummy; input >> bineps;
    input >> dummy; input >> bintrainclass;
    input >> dummy; input >> bintraintarg;
    input >> dummy; input >> binepsclass;
    input >> dummy; input >> binepsweight;
    input >> dummy; input >> binCclass;
    input >> dummy; input >> binCweight;
    input >> dummy; input >> binCweightfuzz;
    input >> dummy; input >> isSVMviaSVR;
    input >> dummy; input >> binNnc;
    input >> dummy; input >> dthres;



    input >> dummy;
    SVM_Scalar_rff::inputstream(input);

    return input;
}

int SVM_Binary_rff::prealloc(int expectedN)
{
    bintrainclass.prealloc(expectedN);
    bintraintarg.prealloc(expectedN);
    binepsclass.prealloc(expectedN);
    binepsweight.prealloc(expectedN);
    binCclass.prealloc(expectedN);
    binCweight.prealloc(expectedN);
    binCweightfuzz.prealloc(expectedN);
    SVM_Scalar_rff::prealloc(expectedN);

    return 0;
}

int SVM_Binary_rff::preallocsize(void) const
{
    return SVM_Scalar_rff::preallocsize();
}











int SVM_Binary_rff::disable(int i)
{
    if ( i < 0 )
    {
        return 0;
    }
//    i = ( i >= 0 ) ? i : ((-i-1)%(SVM_Binary_rff::N()));

// Only gets called from errortest, so whatever

    NiceAssert( i >= 0 );
    NiceAssert( i < SVM_Binary_rff::N() );


    int res = 0;

    if ( bintrainclass(i) )
    {
        res = 1;

        --(binNnc("&",bintrainclass(i)+1));
        ++(binNnc("&",1));

        bintrainclass("&",i) = 0;

        res |= SVM_Scalar_rff::setd(i,DCALC(0));
    }

    return res;
}

int SVM_Binary_rff::disable(const Vector<int> &ii)
{
    retVector<int> tmpva;

    int res = 0;
    int i,j;

    for ( j = 0 ; j < ii.size() ; ++j )
    {
        i = ii(j);

//        i = ( i >= 0 ) ? i : ((-i-1)%(SVM_Binary_rff::N()));

        if ( bintrainclass(i) && ( i >= 0 ) )
        {
            res |= 1;

            --(binNnc("&",bintrainclass(i)+1));
            ++(binNnc("&",1));

            bintrainclass("&",i) = 0;

            res |= SVM_Scalar_rff::setd(i,DCALC(0));
        }
    }

    return res;
}
