
//
// Binary Classification SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "lsv_binary.hpp"
#include <iostream>
#include <sstream>
#include <string>
#include <typeinfo>


class LSV_Single;

// the zcalc macro effectively calculates what we want gp to be to take into
// account both z and tube factors, allowing hp == 0 which results in faster
// optimisation.  The functions below are those that need to use this translation
// macro.

#define DCALC(_d_)                                             ( ( (_d_) ) ? 2 : (_d_) )
#define ZCALC(_z_,_d_,_E_,_Eclass_,_xclass_,_Eweigh_)          ( (_z_) + ( (_d_) * (_Eclass_)(((_xclass_)+1)/2) * (_E_) * (_Eweigh_) ) )
#define CWCALC(_xclass_,_Cweight_,_Cweightfuzz_)               ( (_Cweight_) * (_Cweightfuzz_) * (binCclass((_xclass_)+1)) )

LSV_Binary::LSV_Binary() : LSV_Scalar()
{
    setaltx(nullptr);

    LSV_Scalar::seteps(0);
    bineps = 1;
    binepsclass.resize(3);
    binepsclass = 1.0;
    binCclass.resize(3);
    binCclass = 1.0;
    binNnc.resize(4);
    binNnc = 0;
    LSV_Scalar::setRestrictEpsNeg();

    return;
}

LSV_Binary::LSV_Binary(const LSV_Binary &src) : LSV_Scalar()
{
    setaltx(nullptr);

    LSV_Scalar::seteps(0);
    bineps = 1;
    binepsclass.resize(3);
    binepsclass = 1.0;
    binCclass.resize(3);
    binCclass = 1.0;
    binNnc.resize(4);
    binNnc = 0;
    LSV_Scalar::setRestrictEpsNeg();

    assign(src,0);

    return;
}

LSV_Binary::LSV_Binary(const LSV_Binary &src, const ML_Base *xsrc) : LSV_Scalar()
{
    setaltx(xsrc);

    LSV_Scalar::seteps(0);
    bineps = 1;
    binepsclass.resize(3);
    binepsclass = 1.0;
    binCclass.resize(3);
    binCclass = 1.0;
    binNnc.resize(4);
    binNnc = 0;
    LSV_Scalar::setRestrictEpsNeg();

    assign(src,-1);

    return;
}

LSV_Binary::~LSV_Binary()
{
    return;
}

double LSV_Binary::calcDist(const gentype &ha, const gentype &hb, int ia, int db) const
{
    (void) ia;

    double res = 0;

    if ( db )
    {
        res = ( ( (int) ha ) != ( (int) hb ) ) ? 1 : 0;
    }

    return res;
}

int LSV_Binary::setC(double xC)
{
    return LSV_Scalar::setC(xC);
}

int LSV_Binary::seteps(double xeps)
{
    int i;
    int res = 0;

    bineps = xeps;

    {
        if ( LSV_Binary::N() )
	{
            for ( i = 0 ; i < LSV_Binary::N() ; ++i )
	    {
                res |= LSV_Scalar::sety(i,gentype(ZCALC(bintraintarg(i),bintrainclass(i),bineps,binepsclass,bintrainclass(i),binepsweight(i))));
                gentype yn(bintrainclass(i));
                res |= SVM_Generic::sety(i,yn);
	    }
	}
    }

    return res;
}

int LSV_Binary::setepsclass(int d, double xeps)
{
    NiceAssert( ( d == -1 ) || ( d == 0 ) || ( d == +1 ) );

    int i;

    binepsclass("&",d+1) = xeps;

    int res = LSV_Scalar::setepsclass(d,xeps);

    {
        if ( LSV_Binary::N() )
	{
            for ( i = 0 ; i < LSV_Binary::N() ; ++i )
	    {
                if ( bintrainclass(i) == d )
		{
                    res |= LSV_Scalar::sety(i,gentype(ZCALC(bintraintarg(i),bintrainclass(i),bineps,binepsclass,bintrainclass(i),binepsweight(i))));
                    gentype yn(bintrainclass(i));
                    res |= SVM_Generic::sety(i,yn);
		}
	    }
	}
    }

    return res;
}

void LSV_Binary::setalleps(double xeps, const Vector<double> &xepsclass)
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

int LSV_Binary::scaleepsweight(double scalefactor)
{
    int res = 0;

    if ( LSV_Binary::N() )
    {
	int i;

        for ( i = 0 ; i < LSV_Binary::N() ; ++i )
	{
            res |= setepsweight(i,binepsweight(i)*scalefactor);
	}
    }

    return res;
}

int LSV_Binary::scaleCweight(double scalefactor)
{
    int res = 0;

    if ( LSV_Binary::N() )
    {
	int i;

        for ( i = 0 ; i < LSV_Binary::N() ; ++i )
	{
            res |= setCweight(i,binCweight(i)*scalefactor);
	}
    }

    return res;
}

int LSV_Binary::scaleCweightfuzz(double scalefactor)
{
    int res = 0;

    if ( LSV_Binary::N() )
    {
	int i;

        for ( i = 0 ; i < LSV_Binary::N() ; ++i )
	{
            res |= setCweightfuzz(i,binCweightfuzz(i)*scalefactor);
	}
    }

    return res;
}

int LSV_Binary::setCclass(int d, double xC)
{
    NiceAssert( ( d == -1 ) || ( d == 0 ) || ( d == +1 ) );

    int i;
    int res = 0;

    binCclass("&",d+1) = xC;

    if ( LSV_Binary::N() )
    {
        for ( i = 0 ; i < LSV_Binary::N() ; ++i )
	{
            if ( bintrainclass(i) == d )
	    {
                res |= LSV_Scalar::setCweight(i,CWCALC(bintrainclass(i),binCweight(i),binCweightfuzz(i)));
	    }
	}
    }

    return res;
}

void LSV_Binary::prepareKernel(void)
{
    LSV_Scalar::prepareKernel();

    return;
}

int LSV_Binary::resetKernel(int modind, int onlyChangeRowI, int updateInfo)
{
    int res = LSV_Scalar::resetKernel(modind,onlyChangeRowI,updateInfo);

    return res;
}

int LSV_Binary::setKernel(const MercerKernel &xkernel, int modind, int onlyChangeRowI)
{
    int res = LSV_Scalar::setKernel(xkernel,modind,onlyChangeRowI);

    return res;
}

int LSV_Binary::sety(int i, const gentype &zn)
{
    NiceAssert( zn.isCastableToIntegerWithoutLoss() );

    return setd(i,(int) zn);
}

int LSV_Binary::sety(const Vector<int> &j, const Vector<gentype> &yn)
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

int LSV_Binary::sety(const Vector<gentype> &yn)
{
    NiceAssert( LSV_Binary::N() == yn.size() );

    int res = 0;

    if ( LSV_Binary::N() )
    {
        int i;

        for ( i = 0 ; i < LSV_Binary::N() ; ++i )
        {
            res |= sety(i,yn(i));
        }
    }

    return res;
}

int LSV_Binary::setd(int i, int xd)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < LSV_Binary::N() );
    NiceAssert( ( xd == -1 ) || ( xd == 0 ) || ( xd == +1 ) );

    int res = 0;

    if ( xd != bintrainclass(i) )
    {
        res = 1;

        setdinternal(i,xd);
    }

    return res;
}



int LSV_Binary::sety(int i, double xz)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < LSV_Binary::N() );

    int res = 0;

    bintraintarg("&",i) = xz;

    res |= LSV_Scalar::sety(i,gentype(ZCALC(bintraintarg(i),bintrainclass(i),bineps,binepsclass,bintrainclass(i),binepsweight(i))));
    gentype yn(bintrainclass(i));
    res |= SVM_Generic::sety(i,yn);

    return res;
}

int LSV_Binary::setCweight(int i, double xCweight)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < LSV_Binary::N() );

    binCweight("&",i) = xCweight;

    return LSV_Scalar::setCweight(i,CWCALC(bintrainclass(i),binCweight(i),binCweightfuzz(i)));
}

int LSV_Binary::setCweightfuzz(int i, double xCweight)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < LSV_Binary::N() );

    binCweightfuzz("&",i) = xCweight;

    return LSV_Scalar::setCweight(i,CWCALC(bintrainclass(i),binCweight(i),binCweightfuzz(i)));
}

int LSV_Binary::setepsweight(int i, double xepsweight)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < LSV_Binary::N() );

    binepsweight("&",i) = xepsweight;

    int res = LSV_Scalar::setepsweight(i,xepsweight);

    {
        res |= LSV_Scalar::sety(i,gentype(ZCALC(bintraintarg(i),bintrainclass(i),bineps,binepsclass,bintrainclass(i),binepsweight(i))));
        gentype yn(bintrainclass(i));
        res |= SVM_Generic::sety(i,yn);
    }

    return res;
}

int LSV_Binary::setd(const Vector<int> &j, const Vector<int> &d)
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

int LSV_Binary::sety(const Vector<int> &j, const Vector<double> &xz)
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
            res |= LSV_Scalar::sety(j(i),gentype(ZCALC(bintraintarg(j(i)),bintrainclass(j(i)),bineps,binepsclass,bintrainclass(j(i)),binepsweight(j(i)))));
            gentype yn(bintrainclass(j(i)));
            res |= SVM_Generic::sety(j(i),yn);
	}
    }

    return res;
}

int LSV_Binary::setCweight(const Vector<int> &j, const Vector<double> &xCweight)
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

int LSV_Binary::setCweightfuzz(const Vector<int> &j, const Vector<double> &xCweight)
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

int LSV_Binary::setepsweight(const Vector<int> &j, const Vector<double> &xepsweight)
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

int LSV_Binary::setd(const Vector<int> &d)
{
    NiceAssert( d.size() == LSV_Binary::N() );

    int i;
    int res = 0;

    if ( LSV_Binary::N() )
    {
        for ( i = 0 ; i < LSV_Binary::N() ; ++i )
	{
            res |= setdinternal(i,d(i));
	}
    }

    return res;
}

int LSV_Binary::sety(const Vector<double> &xz)
{
    NiceAssert( bintraintarg.size() == xz.size() );

    int res = 0;

    bintraintarg = xz;

    if ( LSV_Binary::N() )
    {
	int i;

        for ( i = 0 ; i < LSV_Binary::N() ; ++i )
	{
            res |= LSV_Scalar::sety(i,gentype(ZCALC(bintraintarg(i),bintrainclass(i),bineps,binepsclass,bintrainclass(i),binepsweight(i))));
            gentype yn(bintrainclass(i));
            res |= SVM_Generic::sety(i,yn);
	}
    }

    return res;
}

int LSV_Binary::setCweight(const Vector<double> &xCweight)
{
    NiceAssert( xCweight.size() == LSV_Binary::N() );

    int res = 0;

    int i;

    if ( LSV_Binary::N() )
    {
        for ( i = 0 ; i < LSV_Binary::N() ; ++i )
	{
            res |= setCweight(i,xCweight(i));
	}
    }

    return res;
}

int LSV_Binary::setCweightfuzz(const Vector<double> &xCweight)
{
    NiceAssert( xCweight.size() == LSV_Binary::N() );

    int res = 0;

    int i;

    if ( LSV_Binary::N() )
    {
        for ( i = 0 ; i < LSV_Binary::N() ; ++i )
	{
            res |= setCweightfuzz(i,xCweight(i));
	}
    }

    return res;
}

int LSV_Binary::setepsweight(const Vector<double> &xepsweight)
{
    NiceAssert( xepsweight.size() == LSV_Binary::N() );

    int i;
    int res = 0;

    if ( LSV_Binary::N() )
    {
        for ( i = 0 ; i < LSV_Binary::N() ; ++i )
	{
            res |= setepsweight(i,xepsweight(i));
	}
    }

    return res;
}

int LSV_Binary::setdinternal(int i, int xd)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < LSV_Binary::N() );
    NiceAssert( ( xd == -1 ) || ( xd == 0 ) || ( xd == +1 ) );

    int res = 0;

    if ( xd != bintrainclass(i) )
    {
        res = 1;

        --(binNnc("&",bintrainclass(i)+1));
        ++(binNnc("&",xd+1));

        bintrainclass("&",i) = xd;

        res |= LSV_Scalar::setd(i,DCALC(xd));
        res |= LSV_Scalar::sety(i,gentype(ZCALC(bintraintarg(i),bintrainclass(i),bineps,binepsclass,bintrainclass(i),binepsweight(i))));
        gentype yn(bintrainclass(i));
        res |= SVM_Generic::sety(i,yn);
        res |= LSV_Scalar::setCweight(i,CWCALC(bintrainclass(i),binCweight(i),binCweightfuzz(i)));
    }

    return res;
}

int LSV_Binary::addTrainingVector(int i, const gentype &z, const SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    return LSV_Binary::addTrainingVector(i,(int) z,x,Cweigh,epsweigh);
}

int LSV_Binary::qaddTrainingVector(int i, const gentype &z, SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    return LSV_Binary::qaddTrainingVector(i,(int) z,x,Cweigh,epsweigh);
}

int LSV_Binary::addTrainingVector(int i, const Vector<gentype> &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
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

    return LSV_Binary::addTrainingVector(i,zz,x,Cweigh,epsweigh,xz);
}

int LSV_Binary::qaddTrainingVector(int i, const Vector<gentype> &z, Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
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

    return LSV_Binary::qaddTrainingVector(i,zz,x,Cweigh,epsweigh,xz);
}

int LSV_Binary::addTrainingVector(int i, int xd, const SparseVector<gentype> &x, double xCweigh, double xepsweigh, double xz)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= LSV_Binary::N() );

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

    int res = LSV_Scalar::addTrainingVector(i,gentype(ZCALC(bintraintarg(i),bintrainclass(i),bineps,binepsclass,bintrainclass(i),binepsweight(i))),x,CWCALC(bintrainclass(i),binCweight(i),binCweightfuzz(i)),binepsweight(i));
    res |= LSV_Scalar::setd(i,DCALC(bintrainclass(i)));

    gentype yn(bintrainclass(i));
    res |= SVM_Generic::sety(i,yn);

    return res;
}

int LSV_Binary::qaddTrainingVector(int i, int xd, SparseVector<gentype> &x, double xCweigh, double xepsweigh, double xz)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= LSV_Binary::N() );

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

    int res = LSV_Scalar::qaddTrainingVector(i,gentype(ZCALC(bintraintarg(i),bintrainclass(i),bineps,binepsclass,bintrainclass(i),binepsweight(i))),x,CWCALC(bintrainclass(i),binCweight(i),binCweightfuzz(i)),binepsweight(i));
    res |= LSV_Scalar::setd(i,DCALC(bintrainclass(i)));

    gentype yn(bintrainclass(i));
    res |= SVM_Generic::sety(i,yn);

    return res;
}

int LSV_Binary::addTrainingVector(int i, const Vector<int> &xd, const Vector<SparseVector<gentype> > &xx, const Vector<double> &xCweigh, const Vector<double> &xepsweigh, const Vector<double> &xz)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= LSV_Binary::N() );
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

            res |= LSV_Scalar::addTrainingVector(i+j,gentype(ZCALC(bintraintarg(i+j),bintrainclass(i+j),bineps,binepsclass,bintrainclass(i+j),binepsweight(i+j))),xx(j),CWCALC(bintrainclass(i+j),binCweight(i+j),binCweightfuzz(i+j)),binepsweight(i+j));
            res |= LSV_Scalar::setd(i+j,DCALC(bintrainclass(i+j)));

            gentype yn(bintrainclass(i+j));
            res |= SVM_Generic::sety(i+j,yn);
        }
    }

    return res;
}

int LSV_Binary::qaddTrainingVector(int i, const Vector<int> &xd, Vector<SparseVector<gentype> > &xx, const Vector<double> &xCweigh, const Vector<double> &xepsweigh, const Vector<double> &xz)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= LSV_Binary::N() );
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

            res |= LSV_Scalar::qaddTrainingVector(i+j,gentype(ZCALC(bintraintarg(i+j),bintrainclass(i+j),bineps,binepsclass,bintrainclass(i+j),binepsweight(i+j))),xx("&",j),CWCALC(bintrainclass(i+j),binCweight(i+j),binCweightfuzz(i+j)),binepsweight(i+j));
            res |= LSV_Scalar::setd(i+j,DCALC(bintrainclass(i+j)));

            gentype yn(bintrainclass(i+j));
            res |= SVM_Generic::sety(i+j,yn);
        }
    }

    return res;
}

int LSV_Binary::removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < LSV_Binary::N() );

    --(binNnc("&",bintrainclass(i)+1));

    bintrainclass.remove(i);
    bintraintarg.remove(i);
    binepsweight.remove(i);
    binCweight.remove(i);
    binCweightfuzz.remove(i);

    int res = LSV_Scalar::removeTrainingVector(i,y,x);

    return res;
}

int LSV_Binary::train(int &res, svmvolatile int &killSwitch)
{
    return LSV_Scalar::train(res,killSwitch);
}

int LSV_Binary::ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg, gentype ***pxyprodi) const
{
    NiceAssert( !( retaltg & 2 ) );

    int tempresh = LSV_Scalar::ghTrainingVector(resh,resg,i,retaltg,pxyprodi);

    resh.force_int() = tempresh;

    if ( retaltg & 1 )
    {
        gentype negres(-resg);
        gentype posres(resg);

        Vector<gentype> tempresg(2);

        tempresg("&",0) = negres;
        tempresg("&",1) = posres;

        resg = tempresg;
    }

    return tempresh;
}

double LSV_Binary::eTrainingVector(int i) const
{
    double res = 0;

    if ( ( i < 0 ) || isenabled(i) )
    {
        gentype yyval;
        gentype unusedvar;

        LSV_Scalar::ghTrainingVector(yyval,unusedvar,i);

        double yval = (double) yyval;

        // e = 1/(1+exp(yg))
        // dedg = -y.exp(yg)/((1+exp(yg))^2)
        // d2edg2 = -(y^2).exp(yg)/((1+exp(yg))^2) + -y.exp(yg).-2y.exp(yg)/((1+exp(yg))^3)
        //        = -(y^2).exp(yg)/((1+exp(yg))^2) + 2.(y^2).exp(2yg)/((1+exp(yg))^3)
        //        = -(y^2).exp(yg).( 1 - 2.exp(yg)/(1+exp(yg)) )/((1+exp(yg))^2)
        //        = -(y^2).exp(yg).( (1+exp(yg)-2.exp(yg))/(1+exp(yg)) )/((1+exp(yg))^2)
        //        = -(y^2).exp(yg).( (1-exp(yg))/(1+exp(yg)) )/((1+exp(yg))^2)
        //        = -(y^2).exp(yg).(1-exp(yg))/((1+exp(yg))^3)
        //        = -exp(yg).(1-exp(yg))/((1+exp(yg))^3)

        double dval = (int) y(i);
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

double &LSV_Binary::dedgTrainingVector(double &res, int i) const
{
//    res.resize(tspaceDim()) = 0.0;

    res = 0;

    if ( ( i < 0 ) || isenabled(i) )
    {
        gentype yyval;
        gentype unusedvar;

        LSV_Scalar::ghTrainingVector(yyval,unusedvar,i);
        //Assume no arbitrary biasing

        double yval = (double) yyval;

        // dedg = -y.exp(yg)/((1+exp(yg))^2)

        double dval = (int) y(i);
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

double &LSV_Binary::d2edg2TrainingVector(double &res, int i) const
{
//    double res = 0;

    res = 0;

    if ( ( i < 0 ) || isenabled(i) )
    {
        gentype yyval;
        gentype unusedvar;

        LSV_Scalar::ghTrainingVector(yyval,unusedvar,i);

        double yval = (double) yyval;

        // d2edg2 = -exp(yg).(1-exp(yg))/((1+exp(yg))^3)

        double dval = (int) y(i);
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
        gentype yyval;
        gentype unusedvar = 0;

        LSV_Scalar::ghTrainingVector(yval,unusedvar,i);

        res = (double) yyval;

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
        gentype yyval;
        gentype unusedvar = 0;

        LSV_Scalar::ghTrainingVector(yval,unusedvar,i);

        res = (double) yyval;

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

Matrix<double> &LSV_Binary::dedKTrainingVector(Matrix<double> &res) const
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

Vector<double> &LSV_Binary::dedKTrainingVector(Vector<double> &res, int i) const
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


std::ostream &LSV_Binary::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Binary SVM\n\n";

    repPrint(output,'>',dep) << "epsilon:                         " << bineps                 << "\n";
    repPrint(output,'>',dep) << "d:                               " << bintrainclass          << "\n";
    repPrint(output,'>',dep) << "z:                               " << bintraintarg           << "\n";
    repPrint(output,'>',dep) << "classwise epsilon:               " << binepsclass            << "\n";
    repPrint(output,'>',dep) << "elementwise epsilon:             " << binepsweight           << "\n";
    repPrint(output,'>',dep) << "classwise C:                     " << binCclass              << "\n";
    repPrint(output,'>',dep) << "elementwise C:                   " << binCweight             << "\n";
    repPrint(output,'>',dep) << "elementwise C (fuzz):            " << binCweightfuzz         << "\n";
    repPrint(output,'>',dep) << "Nnc:                             " << binNnc                 << "\n";
    repPrint(output,'>',dep) << "=====================================================================\n";
    repPrint(output,'>',dep) << "Base SVR:                        ";
    LSV_Scalar::printstream(output,dep+1);
    repPrint(output,'>',dep) << "\n";
    repPrint(output,'>',dep) << "---------------------------------------------------------------------\n";

    return output;
}

std::istream &LSV_Binary::inputstream(std::istream &input)
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
    input >> dummy; input >> binNnc;
    input >> dummy;
    LSV_Scalar::inputstream(input);

    return input;
}

int LSV_Binary::prealloc(int expectedN)
{
    bintrainclass.prealloc(expectedN);
    bintraintarg.prealloc(expectedN);
    binepsclass.prealloc(expectedN);
    binepsweight.prealloc(expectedN);
    binCclass.prealloc(expectedN);
    binCweight.prealloc(expectedN);
    binCweightfuzz.prealloc(expectedN);
    LSV_Scalar::prealloc(expectedN);

    return 0;
}

int LSV_Binary::preallocsize(void) const
{
    return LSV_Scalar::preallocsize();
}


