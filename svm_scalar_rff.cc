
//FIXME: use a gradient killSwitch for intrain to stop if v gradient small
//FIXME: make -dia 7 the default method

/*
1/N sum_i y_i cs_i ( w'xi + b ) + \lambda/2 \sum_i w_i^2/v_i

d/db = 1/N sum_i y_i cs_i
d/dw = 1/N sum_i y_i cs_i xi + \lambda w./v


Additional term on s: -sum_i y_i cs_i xi
Additional term on t: -sum_i y_i cs_i




w' ( 1/N sum_i y_i cs_i xi ) + b ( 1/N sum_i y_i cs_i ) + \lambda/2 \sum_i w_i^2/v_i
*/

//TO DO: use minvertOffset with fallback to offNaive.  Don't forget fudgeOn!

//TO DO: make method = 1 visible, consider using it as standard!

//free alpha in feature range, never elsewhere
//don't need locz, remove it and use y()(0,1,N()-1) instead
//fix ranging on vector variable return from SVM_Scalar


//With tuning: .12...      ./svmheavyv7.exe -z R -nN 1000 -c 1 -bal -kt 3 -kg 1 -oe 0.001 -olr 0.015 -dc 10 -trf 1 -tr -AA tr1s.txt -N 1605+\(2*1000\) -s temp2r1.svm
//Without tuning: .15...   ./svmheavyv7.exe -z R -nN 1000 -c 1 -bal -kt 3 -kg 1 -oe 0.001 -olr 0.015 -dc 10 -trf 0 -tr -AA tr1s.txt -N 1605+\(2*1000\) -s temp2r1.svm
//
// TODO - make binary_rff version that does binary classification using classify via regression

//
// Scalar Random-Fourier-Features SVM
//
// Version: 7
// Date: 23/02/2021
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "svm_scalar_rff.hpp"
#include <iostream>
#include <sstream>
#include <string>
#include <typeinfo>
#include "basefn.hpp"
#include "memdebug.hpp"
#include "niceassert.hpp"
#include "adam.hpp"
#include "randfun.hpp"


//#define V1ON3  0.33333333333333333333333333
//#define V2ON3  0.66666666666666666666666666

//#define USE_ADAM   1
//#define ADAM_BETA1 0.9
//#define ADAM_BETA2 0.999
//#define ADAM_EPS   1e-8
//#define ADAM_EPS   0.5

#define FEEDBACK_CYCLE         50
#define MAJOR_FEEDBACK_CYCLE   10000
#define PEGASOS_FULLGRAD_CYCLE 50
#define DEFAULT_RFFLR          0.01

// Solve x^3 - x^2 + d = 0 for x, and calculate dx/dd
double cubicsolveczero(double d, double &dxdd);

SVM_Scalar_rff::SVM_Scalar_rff() : SVM_Scalar()
{
    Atscratch.useSlackAllocation();
    Btscratch.useSlackAllocation();

    biastype = 0;
    locbias.resize(1) = 0.0;

    SVM_Scalar::setQuadraticCost();            // diagonal offset is gamma (E)
    SVM_Scalar::setC(1);                       // alpha range unbounded
    SVM_Scalar::setFixedBias(0.0);             // bias stored locally
    SVM_Scalar::seteps(1);                     // hpzero for speed
    SVM_Scalar::setmaxitcnt(0);                // Unrestricted by default as inintrain may take many iterations!
    SVM_Scalar::getKernel_unsafe().setType(3);                      // Set kernel type RBF by default
    SVM_Scalar::getKernel_unsafe().setRealConstZero(DEFAULT_LSRFF); // Set kernel type RBF by default
    SVM_Scalar::resetKernel();                                      // ...

    locBF.fudgeOn();

    (SVM_Scalar::kerncache).setPreferLower(1); // This makes things faster - all caching should be done on the random-feature part of Gp

    setaltx(nullptr);

    locddtype = 3;
    xlsrff    = DEFAULT_LSRFF;

    locN       = 0;
    locC       = DEFAULT_C;
    loceps     = DEFAULTEPS;
    locD       = DEFAULT_D;
    locminv    = 1;
    locF       = 0;
    locG       = 1;
    loclr      = DEFAULT_RFFLR;  // 1
    loclrb     = DEFAULT_LR;     // 0.3
    locOpttol  = DEFAULT_OPTTOL; // 0.001
    locOpttolb = DEFAULT_OPTTOL; // 0.001
    loctunev   = DEFAULT_TUNEV;  // 1
    locpegk    = -1;
    locReOnly  = DEFAULT_REONLY; // 1
    locNRff    = 0;
    locNRffRep = 0;
    innerAdam  = DEFAULT_INADAM;  // 7
    xoutGrad   = DEFAULT_OUTGRAD; // 5
    costtype   = 1;

    locd  = 0;
    locNN = 0;

    Bacgood  = true; //false; // must start false as we don't know indices
    featGood = true; //false;
    midxset  = false;

    addingNow = 0;

    locCclass.resize(4);
    locCclass = 1.0;

    locepsclass.resize(4);
    locepsclass = 1.0;

    locisTrained    = 0;
    (cholscratchcov)  = 0;
    (covlogdetcalced) = 0;

    return;
}

SVM_Scalar_rff::SVM_Scalar_rff(const SVM_Scalar_rff &src) : SVM_Scalar()
{
    Atscratch.useSlackAllocation();
    Btscratch.useSlackAllocation();

    biastype = 0;
    locbias.resize(1) = 0.0;

    setaltx(nullptr);

    SVM_Scalar::setQuadraticCost();            // diagonal offset is gamma (E)
    SVM_Scalar::setC(1);                       // alpha range unbounded
    SVM_Scalar::setFixedBias(0.0);             // bias stored locally
    SVM_Scalar::seteps(1);                     // hpzero for speed
    SVM_Scalar::setmaxitcnt(0);                // Unrestricted by default as inintrain may take many iterations!
    SVM_Scalar::getKernel_unsafe().setType(3);                      // Set kernel type RBF by default
    SVM_Scalar::getKernel_unsafe().setRealConstZero(DEFAULT_LSRFF); // Set kernel type RBF by default
    SVM_Scalar::resetKernel();                                      // ...

    locBF.fudgeOn();

    (SVM_Scalar::kerncache).setPreferLower(1); // This makes things faster - all caching should be done on the random-feature part of Gp

    locddtype = 3;
    xlsrff    = DEFAULT_LSRFF;

    locN       = 0;
    locC       = DEFAULT_C;
    loceps     = DEFAULTEPS;
    locD       = DEFAULT_D;
    locminv    = 1;
    locF       = 0;
    locG       = 1;
    loclr      = DEFAULT_RFFLR;
    loclrb     = DEFAULT_LR;
    locOpttol  = DEFAULT_OPTTOL;
    locOpttolb = DEFAULT_OPTTOL;
    loctunev   = DEFAULT_TUNEV;
    locpegk    = -1;
    locReOnly  = DEFAULT_REONLY;
    locNRff    = 0;
    locNRffRep = 0;
    innerAdam  = DEFAULT_INADAM;
    xoutGrad   = DEFAULT_OUTGRAD;
    costtype   = 1;

    locd  = 0;
    locNN = 0;

    Bacgood  = true; //false; // must start false as we don't know indices
    featGood = true; //false;
    midxset  = false;

    addingNow = 0;

    locCclass.resize(4);
    locCclass = 1.0;

    locepsclass.resize(4);
    locepsclass = 1.0;

    locisTrained    = 0;
    (cholscratchcov)  = 0;
    (covlogdetcalced) = 0;

    assign(src,0);

    return;
}

SVM_Scalar_rff::SVM_Scalar_rff(const SVM_Scalar_rff &src, const ML_Base *xsrc) : SVM_Scalar()
{
    Atscratch.useSlackAllocation();
    Btscratch.useSlackAllocation();

    biastype = 0;
    locbias.resize(1) = 0.0;

    setaltx(xsrc);

    SVM_Scalar::setQuadraticCost();            // diagonal offset is gamma (E)
    SVM_Scalar::setC(1);                       // alpha range unbounded
    SVM_Scalar::setFixedBias(0.0);             // bias stored locally
    SVM_Scalar::seteps(1);                     // hpzero for speed
    SVM_Scalar::setmaxitcnt(0);                // Unrestricted by default as inintrain may take many iterations!
    SVM_Scalar::getKernel_unsafe().setType(3);                      // Set kernel type RBF by default
    SVM_Scalar::getKernel_unsafe().setRealConstZero(DEFAULT_LSRFF); // Set kernel type RBF by default
    SVM_Scalar::resetKernel();                                      // ...

    locBF.fudgeOn();

    (SVM_Scalar::kerncache).setPreferLower(1); // This makes things faster - all caching should be done on the random-feature part of Gp

    locddtype = 3;
    xlsrff    = DEFAULT_LSRFF;

    locN       = 0;
    locC       = DEFAULT_C;
    loceps     = DEFAULTEPS;
    locD       = DEFAULT_D;
    locminv    = 1;
    locF       = 0;
    locG       = 1;
    loclr      = DEFAULT_RFFLR;
    loclrb     = DEFAULT_LR;
    locOpttol  = DEFAULT_OPTTOL;
    locOpttolb = DEFAULT_OPTTOL;
    loctunev   = DEFAULT_TUNEV;
    locpegk    = -1;
    locReOnly  = DEFAULT_REONLY;
    locNRff    = 0;
    locNRffRep = 0;
    innerAdam  = DEFAULT_INADAM;
    xoutGrad   = DEFAULT_OUTGRAD;
    costtype   = 1;

    locd  = 0;
    locNN = 0;

    Bacgood  = true; //false; // must start false as we don't know indices
    featGood = true; //false;
    midxset  = false;

    addingNow = 0;

    locCclass.resize(4);
    locCclass = 1.0;

    locepsclass.resize(4);
    locepsclass = 1.0;

    locisTrained    = 0;
    (cholscratchcov)  = 0;
    (covlogdetcalced) = 0;

    assign(src,-1);

    return;
}

SVM_Scalar_rff::~SVM_Scalar_rff()
{
    return;
}

int SVM_Scalar_rff::prealloc(int expectedN)
{
    locz.prealloc(expectedN);
    loczr.prealloc(expectedN);
    isact.prealloc(expectedN);
    locCweight.prealloc(expectedN);
    locepsweight.prealloc(expectedN);
    locCweightfuzz.prealloc(expectedN);
    Atscratch.prealloc(expectedN);
    Btscratch.prealloc(expectedN);
    SVM_Scalar::prealloc(expectedN+NRff());

    return 0;
}

std::ostream &SVM_Scalar_rff::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Scalar Random-Fourier-Feature SVM\n\n";

    repPrint(output,'>',dep) << "C:                                 " << locC           << "\n";
    repPrint(output,'>',dep) << "eps:                               " << loceps         << "\n";
    repPrint(output,'>',dep) << "Cclass:                            " << locCclass      << "\n";
    repPrint(output,'>',dep) << "epsclass:                          " << locepsclass    << "\n";
    repPrint(output,'>',dep) << "Cweight:                           " << locCweight     << "\n";
    repPrint(output,'>',dep) << "epsweight:                         " << locepsweight   << "\n";
    repPrint(output,'>',dep) << "Cweightfuzz:                       " << locCweightfuzz << "\n";
    repPrint(output,'>',dep) << "D:                                 " << locD           << "\n";
    repPrint(output,'>',dep) << "minv:                              " << locminv        << "\n";
    repPrint(output,'>',dep) << "F:                                 " << locF           << "\n";
    repPrint(output,'>',dep) << "G:                                 " << locG           << "\n";
    repPrint(output,'>',dep) << "lr:                                " << loclr          << "\n";
    repPrint(output,'>',dep) << "lrb:                               " << loclrb         << "\n";
    repPrint(output,'>',dep) << "Opttol:                            " << locOpttol      << "\n";
    repPrint(output,'>',dep) << "Opttolb:                           " << locOpttolb     << "\n";
    repPrint(output,'>',dep) << "tunev:                             " << loctunev       << "\n";
    repPrint(output,'>',dep) << "pegk:                              " << locpegk        << "\n";
    repPrint(output,'>',dep) << "ReOnly:                            " << locReOnly      << "\n";
    repPrint(output,'>',dep) << "NRff:                              " << locNRff        << "\n";
    repPrint(output,'>',dep) << "NRffRep:                           " << locNRffRep     << "\n";
    repPrint(output,'>',dep) << "Distribution type:                 " << locddtype      << "\n";
    repPrint(output,'>',dep) << "Distribution lengthscale:          " << xlsrff         << "\n";
    repPrint(output,'>',dep) << "innerAdam:                         " << innerAdam      << "\n";
    repPrint(output,'>',dep) << "outGrad:                           " << xoutGrad       << "\n";
    repPrint(output,'>',dep) << "costtype:                          " << costtype       << "\n";
    repPrint(output,'>',dep) << "isTrained:                         " << locisTrained   << "\n";
    repPrint(output,'>',dep) << "N (actual, not including weights): " << locN           << "\n";
    repPrint(output,'>',dep) << "Training targets:                  " << locz           << "\n";
    repPrint(output,'>',dep) << "Training targets (real):           " << loczr          << "\n";
    repPrint(output,'>',dep) << "Random phase:                      " << locphase       << "\n";
    repPrint(output,'>',dep) << "isact:                             " << isact          << "\n";
    repPrint(output,'>',dep) << "Feature weights:                   " << locaw          << "\n";
    repPrint(output,'>',dep) << "Bacgood:                           " << Bacgood        << "\n";
    repPrint(output,'>',dep) << "featGood:                          " << featGood       << "\n";
    repPrint(output,'>',dep) << "midxset:                           " << midxset        << "\n";
    repPrint(output,'>',dep) << "locB:                              " << locB           << "\n";
    repPrint(output,'>',dep) << "locBF:                             " << locBF          << "\n";
    repPrint(output,'>',dep) << "BFoff:                             " << BFoff          << "\n";
    repPrint(output,'>',dep) << "loca:                              " << loca           << "\n";
    repPrint(output,'>',dep) << "locc:                              " << locc           << "\n";
    repPrint(output,'>',dep) << "locd:                              " << locd           << "\n";
    repPrint(output,'>',dep) << "locNN:                             " << locNN          << "\n";
    repPrint(output,'>',dep) << "biasType:                          " << biasType       << "\n";
    repPrint(output,'>',dep) << "locbias:                           " << locbias        << "\n";
    repPrint(output,'>',dep) << "=====================================================================\n";
    repPrint(output,'>',dep) << "Base SVR: ";
    SVM_Scalar::printstream(output,dep+1);
    repPrint(output,'>',dep) << "\n";
    repPrint(output,'>',dep) << "---------------------------------------------------------------------\n";

    return output;
}

std::istream &SVM_Scalar_rff::inputstream(std::istream &input)
{
    wait_dummy dummy;

    input >> dummy; input >> locC;
    input >> dummy; input >> loceps;
    input >> dummy; input >> locCclass;
    input >> dummy; input >> locepsclass;
    input >> dummy; input >> locCweight;
    input >> dummy; input >> locepsweight;
    input >> dummy; input >> locCweightfuzz;
    input >> dummy; input >> locD;
    input >> dummy; input >> locminv;
    input >> dummy; input >> locF;
    input >> dummy; input >> locG;
    input >> dummy; input >> loclr;
    input >> dummy; input >> loclrb;
    input >> dummy; input >> locOpttol;
    input >> dummy; input >> locOpttolb;
    input >> dummy; input >> loctunev;
    input >> dummy; input >> locpegk;
    input >> dummy; input >> locReOnly;
    input >> dummy; input >> locNRff;
    input >> dummy; input >> locNRffRep;
    input >> dummy; input >> locddtype;
    input >> dummy; input >> xlsrff;
    input >> dummy; input >> innerAdam;
    input >> dummy; input >> xoutGrad;
    input >> dummy; input >> costtype;
    input >> dummy; input >> locisTrained;
    input >> dummy; input >> locN;
    input >> dummy; input >> locz;
    input >> dummy; input >> loczr;
    input >> dummy; input >> locphase;
    input >> dummy; input >> isact;
    input >> dummy; input >> locaw;
    input >> dummy; input >> Bacgood;
    input >> dummy; input >> featGood;
    input >> dummy; input >> midxset;
    input >> dummy; input >> locB;
    input >> dummy; input >> locBF;
    input >> dummy; input >> BFoff;
    input >> dummy; input >> loca;
    input >> dummy; input >> locc;
    input >> dummy; input >> locd;
    input >> dummy; input >> locNN;
    input >> dummy; input >> biastype;
    input >> dummy; input >> locbias;
    input >> dummy;
    SVM_Scalar::inputstream(input);

    return input;
}

int SVM_Scalar_rff::gTrainingVector(double &res, int &unusedvar, int i, int retaltg, gentype ***pxyprodi) const
{
    NiceAssert( !( retaltg & 2 ) );

    int &raw = retaltg;

    // Need some minor adjustments to allow for local bias

    int j;

/*
if ( i >= 0 )
{
res = biasR();
int wdim = calcwdim();
for ( j = 0 ; j < wdim ; ++j )
{
res += alphaR()(N()+j)*Gp()(j+N(),i);
}
return ( unusedvar = ( res > 0 ) ? +1 : -1 );
}
*/

    SVM_Scalar::gTrainingVector(res,unusedvar,i,raw,pxyprodi);

    if ( !( xtang(i) & 7 ) )
    {
        if ( i < 0 )
        {
            res -= SVM_Scalar::biasR(); // remove influence of incorrect bias in SVM_Scalar (see SVM_Scalar::gTrainingVector - biasR() is called when calculating in this case!).
        }

        // Add correct bias

        int fxind = 0;

        NiceAssert( !x(i).isf4indpresent(7) || !locNRffRep );

        if ( x(i).isf4indpresent(7) )
        {
            const Vector<gentype> &taskind = x(i).f4(7).cast_vector();

            NiceAssert( locNRffRep == taskind.size() );

            // In the multitask case we need the bias relevant to the given task

            for ( j = 0 ; j < locNRffRep ; ++j )
            {
                if ( (int) taskind(j) )
                {
                    fxind = j;
                    break;
                }
            }
        }

        res += locbias(fxind);
    }

    return ( unusedvar = ( res > 0 ) ? +1 : -1 );
}

















int SVM_Scalar_rff::cov(gentype &resv, gentype &resmu, int ia, int ib, gentype ***pxyprodi, gentype ***pxyprodj, gentype **pxyprodij) const
{
    // FOR DETAILS: see section 3.2 of Daskalakis et al, "How Good are Low-Rank Approximations in Gaussian Process Regression?"

    // Modified version of SVM_Scalar call to allow for kernel differences in inverted block
    {
        // See svm_scalar for details here!

        int wdim = INDIM*NRff();
        int NN = SVM_Scalar::N();
        int Nloc = N();

        NiceAssert( ( ia >= -3 ) && ( ib != -2 ) );
        NiceAssert( ia < Nloc );
        NiceAssert( ( ib >= -3 ) && ( ib != -2 ) );
        NiceAssert( ib < Nloc );
        NiceAssert( emm == 2 );

        //int dtva = xtang(ia) & 7;
        //int dtvb = xtang(ib) & 7;

        NiceAssert( ( xtang(ia) & 7 ) >= 0 );
        NiceAssert( ( xtang(ib) & 7 ) >= 0 );

        // This is used elsewhere (ie not scalar), so the following is relevant

        NiceAssert( !( xtang(ia) & 4 ) );
        NiceAssert( !( xtang(ib) & 4 ) );
        NiceAssert( SVM_Scalar::isUnderlyingScalar() );

        int unusedvar = 0;

        double &resvv = resv.force_double();
        double &resgg = resmu.force_double();

        gTrainingVector(resgg,unusedvar,ia,0,pxyprodi);

        NiceAssert( Bacgood );

        resvv = 0.0;

        if ( SVM_Scalar::NS() )
        {
            int j;

            Vector<double> Kia(NN);
            Vector<double> Kib(NN);

            if ( ia >= 0 )
            {
                for ( j = 0 ; j < NN ; ++j )
                {
                    Kia("&",j) = Gp()(ia,j);
                }

                // Off-diagonal by definition, so no offsets!
            }

            else
            {
                for ( j = 0 ; j < NN ; ++j )
                {
                    Kia("&",j) = K2(ia,j,pxyprodi ? (const gentype **) pxyprodi[j] : nullptr);
                }
            }

            if ( ib == ia )
            {
                Kib = Kia;
            }

            else if ( ib >= 0 )
            {
                for ( j = 0 ; j < NN ; ++j )
                {
                    Kib("&",j) = Gp()(j,ib);
                }

                // Off-diagonal by definition, so no offsets!
            }

            else
            {
                for ( j = 0 ; j < NN ; ++j )
                {
                    Kib("&",j) = K2(ib,j,pxyprodj ? (const gentype **) pxyprodj[j] : nullptr);
                }
            }

            Vector<double> BKia(Kia);
            Vector<double> BKib(Kib);

            const Matrix<double> &B = locB;

            retVector<double> tmpkx;
            retVector<double> tmpkc;

            BKia("&",Nloc,1,Nloc+wdim-1,tmpkx) *= B;
            BKib("&",Nloc,1,Nloc+wdim-1,tmpkc) *= B;

            Vector<double> Kres(NN);

// MAIN DIFFERENCE HERE! - can't use diagonal block as per svm as this is reserved for L/H!
            //const Matrix<double> &B = locB;
            Vector<double> &ivl = (ivscratch);

            ivl.resize(wdim);
            (yscratch).resize(wdim);
            (cholscratch).resize(wdim,wdim);
            (cholscratchb).resize(wdim,wdim);

            if ( !(cholscratchcov) || !(covlogdetcalced) )
            {
                const Vector<double> &vv  = locaw;

                for ( int i = 0 ; i < wdim ; ++i )
                {
                    ivl("&",i)  = sigma() * ( ( vv(i) > locminv ) ? 1/vv(i) : 1/locminv );
                }
            }

            retVector<double> tmpka;
            retVector<double> tmpkb;

            B.naiveCholInve(Kres("&",Nloc,1,Nloc+wdim-1,tmpka),BKib(Nloc,1,Nloc+wdim-1,tmpkb),1.0,ivl,(yscratch),(cholscratch),0,(cholscratchcov));
            (cholscratchcov) = 1; // if you have a whole series of cov calls, only the first one should be cubic cost!
//retVector<double> tmpki;
//retVector<double> tmpkj;
//retVector<double> tmpkk;
//retVector<double> tmpkl;
//retVector<double> tmpkt;
//errstream() << "phantomx -2: Kia = " << Kia(Nloc,1,Nloc+wdim-1,tmpki) << "\n";
//errstream() << "phantomx -2: Kib = " << Kib(Nloc,1,Nloc+wdim-1,tmpkj) << "\n";
//errstream() << "phantomx -2: BKia = " << BKia(Nloc,1,Nloc+wdim-1,tmpkk) << "\n";
//errstream() << "phantomx -2: BKib = " << BKib(Nloc,1,Nloc+wdim-1,tmpkl) << "\n";
//errstream() << "phantomx -1: Kres = " << Kres(Nloc,1,Nloc+wdim-1,tmpkt) << "\n";
//errstream() << "phantomx 1: B = " << B << "\n";
//errstream() << "phantomx 2: ivl = " << ivl << "\n";

            resvv = 0.0;

            // Changes here too!
            for ( int cj = 0 ; cj < wdim ; ++cj )
            {
                resvv += ( Kia(Nloc+cj)* Kib(Nloc+cj));
                resvv -= ( Kia(Nloc+cj)*BKib(Nloc+cj))/ivl(cj);
                resvv += (BKia(Nloc+cj)*Kres(Nloc+cj))/ivl(cj);
            }
//errstream() << "phantomx 4: resvv = " << resvv << "\n";
        }

        else
        {
            if ( ( ia >= 0 ) && ( ib >= 0 ) )
            {
                retVector<double> tmpva;

                resvv  = Gp()(ia,ib);
                resvv -= ( ia == ib ) ? diagoffset()(tmpva)(ia) : 0.0;
            }

            else
            {
                resvv = K2(ia,ib,(const gentype **) pxyprodij);
            }
       }
    }





    // Need some minor adjustments to allow for local bias - see gTrainingVector

    int j;

    //SVM_Scalar::cov(resv,resmu,ia,ib,pxyprodi,pxyprodj,pxyprodij);

    if ( !( xtang(ia) & 7 ) )
    {
        double &res = resmu.force_double();

        if ( ia < 0 )
        {
            res -= SVM_Scalar::biasR(); // remove influence of incorrect bias in SVM_Scalar (see SVM_Scalar::gTrainingVector - biasR() is called when calculating in this case!).
        }

        // Add correct bias

        int fxind = 0;

        NiceAssert( !x(ia).isf4indpresent(7) || !locNRffRep );

        if ( x(ia).isf4indpresent(7) )
        {
            const Vector<gentype> &taskind = x(ia).f4(7).cast_vector();

            NiceAssert( locNRffRep == taskind.size() );

            // In the multitask case we need the bias relevant to the given task

            for ( j = 0 ; j < locNRffRep ; ++j )
            {
                if ( (int) taskind(j) )
                {
                    fxind = j;
                    break;
                }
            }
        }

        res += locbias(fxind);
    }

    return 0;
}

double SVM_Scalar_rff::loglikelihood(void) const
{
    double res = 0;

    // Close enough: res = -1/2 y.inv(Gp).y - 1/2 log(det(Gp)) - n/2 log 2pi
    // (works for fixed-bias GP, not defined well at this level, but meh)
    //
    // c/f section 3.2 of Daskalakis et al, "How Good are Low-Rank Approximations in Gaussian Process Regression?"
    // we can readily see that:
    //
    // 1/2 y.inv(Gp).y = ( ||y||_2^2 - (Z'.y)'.inv( B + sigma().I).(Z'.y) )/(2.sigma())
    // 1/2 log(det(Gp)) = 1/2 log(det(B+sigma I)) + 1/2 (Ntr - wdim) log(sigma)

    // FIXME: really should use ivl, not sigma, but this is close enough

    int wdim = INDIM*NRff();
    int Nloc = N();

    Vector<double> Zy(wdim);
    retMatrix<double> tmpma;

    mult(Zy,Gp()(Nloc,1,Nloc+wdim-1,0,1,Nloc-1,tmpma),loczr);

//    for ( int i = 0 ; i < wdim ; i++ )
//    {
//        Zy("&",i) = 0.0;
//
//        for ( int j = 0 ; j < Nloc ; j++ )
//        {
//            Zy("&",i) += Gp()(i+Nloc,j)*loczr(j);
//        }
//    }

    const Matrix<double> &B = locB;
    Vector<double> &ivl = (ivscratch);

    ivl.resize(wdim);
    (yscratch).resize(wdim);
    (cholscratch).resize(wdim,wdim);
    (cholscratchb).resize(wdim,wdim);

    if ( !(cholscratchcov) || !(covlogdetcalced) )
    {
        const Vector<double> &vv  = locaw;

        for ( int i = 0 ; i < wdim ; ++i )
        {
            ivl("&",i)  = sigma() * ( ( vv(i) > locminv ) ? 1/vv(i) : 1/locminv );
        }
    }

    if ( !(covlogdetcalced) )
    {
        (covlogdet) = locB.logdet_naiveChol((cholscratchb),1.0,ivl);
        (covlogdetcalced) = 1;
    }

    Vector<double> alp(wdim);

    B.naiveCholInve(alp,Zy,1.0,ivl,(yscratch),(cholscratch),0,(cholscratchcov));

    res = 0.0;

    double tmp;

    res -= norm2(loczr)/(2.0*sigma());
    res += twoProduct(tmp,Zy,alp)/(2.0*sigma());

    res -= (covlogdet)/2.0;
    res -= (Nloc-wdim)*log(sigma())/2.0;

    res -= NUMBASE_LN2PI*Nloc/2.0;

    return res;
}

double SVM_Scalar_rff::maxinfogain(void) const
{
    int wdim = INDIM*NRff();

    double res = 0;

    // Note that sigma here is sigma^2 in Srinivas
    //
    // Close enough: res = 1/2 log( det( sigma^{-1} K + I ) )
    //                   = 1/2 log( det( sigma^{-1} ( K + sigma I ) ) )
    //                   = 1/2 log( sigma^{-n} det( K + sigma I ) )
    //                   = 1/2 log( det( K + sigma I ) ) + 1/2 log( sigma^{-n} )
    //                   = 1/2 log( det( K + sigma I ) ) - n/2 log( sigma )
    // (works for fixed-bias GP, not defined well at this level, but meh)
    //
    // see section 3.2 in "How Good are Low-Rank Approximations in Gaussian Process Regression?"
    // for details about how to calculate the determinant
    //
    // 1/2 log(det(Gp)) = 1/2 log(det(B+sigma I)) + 1/2 (n - wdim) log(sigma)

    Vector<double> &ivl = (ivscratch);

    ivl.resize(wdim);
    (yscratch).resize(wdim);
    (cholscratch).resize(wdim,wdim);
    (cholscratchb).resize(wdim,wdim);

    if ( !(cholscratchcov) || !(covlogdetcalced) )
    {
        const Vector<double> &vv  = locaw;

        for ( int i = 0 ; i < wdim ; ++i )
        {
            ivl("&",i)  = sigma() * ( ( vv(i) > locminv ) ? 1/vv(i) : 1/locminv );
        }
    }

    if ( !(covlogdetcalced) )
    {
        (covlogdet) = locB.logdet_naiveChol((cholscratchb),1.0,ivl);
        (covlogdetcalced) = 1;
    }

    res  = (covlogdet)/2.0;
    res -= wdim*log(sigma())/2.0;

//errstream() << "phantomxyz 111 " << res << " = 1/2 log(" << Q.fact_det() << ") - " << (N()/2.0) << "*log(" << sigma() << ")\n";

    return res;
}

double SVM_Scalar_rff::RKHSnorm(void) const
{
    // res = w'.w, recalling that we essentially have the
    // weight vector in this case.

    int wdim = INDIM*NRff();
    int Nloc = N();

    retVector<double> tmpva;

    return norm2(SVM_Scalar::alphaR()(Nloc,1,Nloc+wdim-1,tmpva));
}






int SVM_Scalar_rff::addTrainingVector(int i, double zzz, const SparseVector<gentype> &xxxx, double Cweigh, double epsweigh, int d)
{
    SparseVector<gentype> tempx(xxxx);

    return qaddTrainingVector(i,zzz,tempx,Cweigh,epsweigh,d);
}

int SVM_Scalar_rff::qaddTrainingVector(int i, double zzz, SparseVector<gentype> &xxxx, double Cweigh, double epsweigh, int d)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= N() );

    int xdim = xspaceDim();
    int res = 0;

//if ( i == 4 )
//{
//errstream() << "phantomxyz add i        = " << i        << "\n";
//errstream() << "phantomxyz add zzz      = " << zzz      << "\n";
//errstream() << "phantomxyz add xxxx     = " << xxxx     << "\n";
//errstream() << "phantomxyz add Cweigh   = " << Cweigh   << "\n";
//errstream() << "phantomxyz add epsweigh = " << epsweigh << "\n";
//errstream() << "phantomxyz add d        = " << d        << "\n";
//}
    res |= fixupfeatures();

    loczr.add(i);          loczr("&",i)               = zzz;
    locz.add(i);           locz("&",i).force_double() = zzz;
    isact.add(i);          isact("&",i)               = ( ( ( d == 2 ) && isQuadraticCost() ) || ( d == 0 ) ) ? 1 : 0;
    locCweight.add(i);     locCweight("&",i)          = Cweigh;
    locepsweight.add(i);   locepsweight("&",i)        = epsweigh;
    locCweightfuzz.add(i); locCweightfuzz("&",i)      = 1;

    Vector<double> Qii;

    addingNow = 1; // N() is "off-by-one" for the purposes of the K2 callback that follows, so this is used to compensate
    res |= SVM_Scalar::qaddTrainingVector(i,zzz,xxxx,1e12,epsweigh,d); // Important not to include z here or it will "come back" through gTrainingVector(i) for i >= 0
    addingNow = 0;
    ++locN;
    res |= addRemoveXInfluence(+1,i,Qii);

    if ( xspaceDim() != xdim )
    {
        Bacgood  = false;
        featGood = false;

        res |= fixupfeatures();
    }

    locisTrained    = 0;
    (cholscratchcov)  = 0;
    (covlogdetcalced) = 0;

    return res;
}

int SVM_Scalar_rff::removeTrainingVector(int i, gentype &y, SparseVector<gentype> &xxxx)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );

    int xdim = xspaceDim();
    int res = 0;

    res |= fixupfeatures();

    Vector<double> Qii;

    gentype dummy;

    res |= addRemoveXInfluence(-1,i,Qii);
    --locN;
    res |= SVM_Scalar::removeTrainingVector(i,dummy,xxxx);

    y = locz(i);

    locz.remove(i);
    loczr.remove(i);
    isact.remove(i);
    locCweight.remove(i);
    locepsweight.remove(i);
    locCweightfuzz.remove(i);

    locisTrained    = 0;
    (cholscratchcov)  = 0;
    (covlogdetcalced) = 0;

    if ( xspaceDim() != xdim )
    {
        Bacgood  = false;
        featGood = false;

        res |= fixupfeatures();
    }

    return res;
}

int SVM_Scalar_rff::settunev(int nv)
{
    int i;
    int res = 0;

    res |= fixupfeatures();

    if ( loctunev != nv )
    {
        res = 1;

        locisTrained    = 0;
        (cholscratchcov)  = 0;
        (covlogdetcalced) = 0;

        loctunev = nv;

        for ( i = 0 ; i < locaw.size() ; ++i )
        {
            if ( ( loctunev == -1 ) && locaw(i) >= 0 )
            {
                locaw("&",i) = -locminv;
            }

            else if ( ( loctunev == +1 ) && locaw(i) <= 0 )
            {
                locaw("&",i) = +locminv;
            }
        }
    }

    return res;
}

int SVM_Scalar_rff::setminv(double nv)
{
    Vector<double> BFstep(BFoff);

    locminv = nv;
    BFoff = 1/(C()*locminv);

    if ( Bacgood )
    {
        BFstep.negate();
        BFstep += BFoff;

        locBF.diagoffset(BFstep,locB,BFoff);
    }

    locisTrained    = 0;
    (cholscratchcov)  = 0;
    (covlogdetcalced) = 0;

    return 1;
}

int SVM_Scalar_rff::setCclass(int xd, double xC)
{
    NiceAssert( ( xd == -1 ) || ( xd == 0 ) || ( xd == +1 ) || ( xd == 2 ) );
    NiceAssert( xC > 0 );

    int res = 0;

    res |= fixupfeatures();

    if ( locCclass(xd+1) == xC )
    {
        return res;
    }

    res = 1;
    int ii;
    Vector<int> i;

    for ( ii = 0 ; ii < N() ; ++ii )
    {
        if ( isenabled(ii) && isact(ii) && ( SVM_Scalar::d()(ii) == xd ) )
        {
            i.add(i.size());
            i("&",i.size()-1) = ii;
        }
    }

    Vector<double> Qii;

    res |= addRemoveXInfluence(-1,i,Qii,false,false,true);
    locCclass("&",xd+1) = xC;
    res |= addRemoveXInfluence(+1,i,Qii,false,false,true);

    locisTrained    = 0;
    (cholscratchcov)  = 0;
    (covlogdetcalced) = 0;

    return res;
}

int SVM_Scalar_rff::setepsclass(int xd, double xeps)
{
    NiceAssert( ( xd == -1 ) || ( xd == 0 ) || ( xd == +1 ) || ( xd == 2 ) );
    NiceAssert( xeps >= 0 );

    int res = 0;

    res |= fixupfeatures();

    if ( locepsclass(xd+1) == xeps )
    {
        return res;
    }

    res = 1;

    locepsclass("&",xd+1) = xeps;

    locisTrained    = 0;
    (cholscratchcov)  = 0;
    (covlogdetcalced) = 0;

    return res;
}

int SVM_Scalar_rff::setCweight(int i, double xC)
{
    int res = 0;

    res |= fixupfeatures();

    if ( locCweight(i) == xC )
    {
        return res;
    }

    NiceAssert( i >= 0 );
    NiceAssert( i < SVM_Scalar::N() );
    NiceAssert( xC > 0 );

    res = 1;

    Vector<double> Qii;

    res |= addRemoveXInfluence(-1,i,Qii,false,false,true);
    locCweight("&",i) = xC;
    res |= addRemoveXInfluence(+1,i,Qii,false,false,true);

    locisTrained    = 0;
    (cholscratchcov)  = 0;
    (covlogdetcalced) = 0;

    return res;
}

int SVM_Scalar_rff::setepsweight(int i, double xeps)
{
    int res = 0;

    res |= fixupfeatures();

    if ( locepsweight(i) == xeps )
    {
        return res;
    }

    NiceAssert( i >= 0 );
    NiceAssert( i < SVM_Scalar::N() );
    NiceAssert( xeps >= 0 );

    locepsweight("&",i) = xeps;

    locisTrained    = 0;
    (cholscratchcov)  = 0;
    (covlogdetcalced) = 0;

    return res;
}

int SVM_Scalar_rff::setCweightfuzz(int i, double xC)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < SVM_Scalar::N() );
    NiceAssert( xC > 0 );

    int res = 0;

    res |= fixupfeatures();

    if ( locCweightfuzz(i) == xC )
    {
        return res;
    }

    res = 1;

    Vector<double> Qii;

    res |= addRemoveXInfluence(-1,i,Qii,false,false,true);
    locCweightfuzz("&",i) = xC;
    res |= addRemoveXInfluence(+1,i,Qii,false,false,true);

    locisTrained    = 0;
    (cholscratchcov)  = 0;
    (covlogdetcalced) = 0;

    return res;
}

int SVM_Scalar_rff::scaleCweight(double s)
{
    int res = 0;

    res |= fixupfeatures();

    if ( s == 1 )
    {
        return res;
    }

    res = 1;

    loca  *= s;
    locc  *= s;
    locd  *= s;
    locNN *= s;

    locCweight *= s;

    if ( Bacgood )
    {
        locB *= s;

        Vector<double> oldBFoff(BFoff);

        oldBFoff *= s;
        locBF.scale(s,locB,oldBFoff); // effectively scaling both locB and BFoff
        oldBFoff.negate();
        oldBFoff += BFoff; // oldBFoff is now a step from s.BFoff to BFoff.
        locBF.diagoffset(oldBFoff,locB,BFoff); // Corrective step to retrieve unscaled BFoff
    }

    locisTrained    = 0;
    (cholscratchcov)  = 0;
    (covlogdetcalced) = 0;

    return res;
}

int SVM_Scalar_rff::scaleepsweight(double s)
{
    int res = 0;

    res |= fixupfeatures();

    if ( s == 1 )
    {
        return res;
    }

    locepsweight *= s;

    locisTrained    = 0;
    (cholscratchcov)  = 0;
    (covlogdetcalced) = 0;

    return res;
}

int SVM_Scalar_rff::scaleCweightfuzz(double s)
{
    int res = 0;

    res |= fixupfeatures();

    if ( s == 1 )
    {
        return res;
    }

    res = 1;

    loca  *= s;
    locc  *= s;
    locd  *= s;
    locNN *= s;

    locCweightfuzz *= s;

    if ( Bacgood )
    {
        locB *= s;

        Vector<double> oldBFoff(BFoff);

        oldBFoff *= s;
        locBF.scale(s,locB,oldBFoff); // effectively scaling both locB and BFoff
        oldBFoff.negate();
        oldBFoff += BFoff; // oldBFoff is now a step from s.BFoff to BFoff.
        locBF.diagoffset(oldBFoff,locB,BFoff); // Corrective step to retrieve unscaled BFoff
    }

    locisTrained    = 0;
    (cholscratchcov)  = 0;
    (covlogdetcalced) = 0;

    return res;
}

int SVM_Scalar_rff::setC(double nv)
{
    int res = 0;

    res |= fixupfeatures();

    if ( C() == nv )
    {
        return res;
    }

    res = 1;

    Vector<double> BFstep(BFoff);

    locC = nv;

    BFoff = 1/(C()*locminv);

    if ( Bacgood )
    {
        BFstep.negate();
        BFstep += BFoff;
        locBF.diagoffset(BFstep,locB,BFoff);
    }

    locisTrained    = 0;
    (cholscratchcov)  = 0;
    (covlogdetcalced) = 0;

    return res;
}

int SVM_Scalar_rff::seteps(double nv)
{
    int res = 0;

    res |= fixupfeatures();

    if ( eps() == nv )
    {
        return res;
    }

    res = 1;

    loceps = nv;

    locisTrained    = 0;
    (cholscratchcov)  = 0;
    (covlogdetcalced) = 0;

    return res;
}

int SVM_Scalar_rff::setx(int i, const SparseVector<gentype> &x)
{
    // Assume index set unchanged

    int res = 0;

    res |= fixupfeatures();

    Vector<double> Qii;

    res |= addRemoveXInfluence(-1,i,Qii,true,false,false);
    res |= SVM_Scalar::setx(i,x);
    res |= addRemoveXInfluence(+1,i,Qii,true,false,false);

    locisTrained    = 0;
    (cholscratchcov)  = 0;
    (covlogdetcalced) = 0;

    return res;
}

int SVM_Scalar_rff::sety(int i, const gentype &zz)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );

    //gentype zzz(0.0);

    int res = 0;

    res |= fixupfeatures();

    Vector<double> Qii;

    res |= addRemoveXInfluence(-1,i,Qii,false,true,false);
    loczr("&",i) = (double) ( locz("&",i) = zz );
    res |= SVM_Scalar::sety(i,0.0_gent);
    res |= addRemoveXInfluence(+1,i,Qii,false,true,false);

    locisTrained    = 0;
    (cholscratchcov)  = 0;
    (covlogdetcalced) = 0;

    return res;
}

int SVM_Scalar_rff::sety(int i, double zz)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < N() );

    //gentype zzz(0.0);

    int res = 0;

    res |= fixupfeatures();

    Vector<double> Qii;

    res |= addRemoveXInfluence(-1,i,Qii,false,true,false);
    locz("&",i) = ( loczr("&",i) = zz );
    res |= SVM_Scalar::sety(i,0.0_gent);
    res |= addRemoveXInfluence(+1,i,Qii,false,true,false);

    locisTrained    = 0;
    (cholscratchcov)  = 0;
    (covlogdetcalced) = 0;

    return res;
}

int SVM_Scalar_rff::resetKernel(int modind, int onlyChangeRowI, int updateInfo)
{
    int res = 0;

    locisTrained    = 0;
    (cholscratchcov)  = 0;
    (covlogdetcalced) = 0;

    res |= SVM_Scalar::resetKernel(modind,onlyChangeRowI,updateInfo);

//    NiceAssert( getKernel().size() == 1 ); - causes throw even when ununsed, need to move elsewhere

    int newlocddtype = getKernel().cType();
    double newLSRff = getKernel().getRealConstZero();

    if ( ( onlyChangeRowI < 0 ) && ( ( newlocddtype != locddtype ) || ( newLSRff != xlsrff ) ) )
    {
        setLSRffandRFFDist(newLSRff,newlocddtype);
    }

/*
    if ( newlocddtype != locddtype )
    {
        setRFFDist(newlocddtype);
    }

    if ( newLSRff != xlsrff )
    {
        setLSRff(newLSRff);
    }
*/

    return res;
}

int SVM_Scalar_rff::setKernel(const MercerKernel &xkernel, int modind, int onlyChangeRowI)
{
    int res = 0;

    locisTrained    = 0;
    (cholscratchcov)  = 0;
    (covlogdetcalced) = 0;

    res |= SVM_Scalar::setKernel(xkernel,modind,onlyChangeRowI);

//    NiceAssert( getKernel().size() == 1 ); - causes throw even when ununsed, need to move elsewhere

    int newlocddtype = getKernel().cType();
    double newLSRff = getKernel().getRealConstZero();

    if ( ( onlyChangeRowI < 0 ) && ( ( newlocddtype != locddtype ) || ( newLSRff != xlsrff ) ) )
    {
        setLSRffandRFFDist(newLSRff,newlocddtype);
    }

/*
    if ( newlocddtype != locddtype )
    {
        setRFFDist(newlocddtype);
    }

    if ( newLSRff != xlsrff )
    {
        setLSRff(newLSRff);
    }
*/

    return res;
}

int SVM_Scalar_rff::setd(int i, int nv)
{
    int res = 0;

    res |= fixupfeatures();

    if ( nv && !SVM_Scalar::d()(i) )
    {
        Vector<double> Qii;

        res |= addRemoveXInfluence(+1,i,Qii);
        res |= SVM_Scalar::setd(i,nv);

        locisTrained    = 0;
        (cholscratchcov)  = 0;
        (covlogdetcalced) = 0;
    }

    else if ( !nv && SVM_Scalar::d()(i) )
    {
        Vector<double> Qii;

        res |= addRemoveXInfluence(-1,i,Qii);
        res |= SVM_Scalar::setd(i,nv);

        locisTrained    = 0;
        (cholscratchcov)  = 0;
        (covlogdetcalced) = 0;
    }

    else if ( nv != SVM_Scalar::d()(i) )
    {
        res |= SVM_Scalar::setd(i,nv);

        locisTrained    = 0;
        (cholscratchcov)  = 0;
        (covlogdetcalced) = 0;
    }

    return res;
}

int SVM_Scalar_rff::setd(const Vector<int> &i, const Vector<int> &nv)
{
    NiceAssert( i.size() == nv.size() );

    int res = 0;
    int ii;

    res |= fixupfeatures();

    Vector<int> ipos;
    Vector<int> ineg;
    Vector<int> idiff;

    for ( ii = 0 ; ii < i.size() ; ++ii )
    {
        if ( nv(ii) && !SVM_Scalar::d()(i(ii)) )
        {
            ipos.add(ipos.size());
            ipos("&",ipos.size()-1) = i(ii);
        }

        else if ( !nv(ii) && SVM_Scalar::d()(i(ii)) )
        {
            ineg.add(ineg.size());
            ineg("&",ineg.size()-1) = i(ii);
        }

        else if ( nv(ii) != SVM_Scalar::d()(i(ii)) )
        {
            idiff.add(ineg.size());
            idiff("&",ineg.size()-1) = i(ii);
        }
    }

    Vector<double> Qii;

    res |= addRemoveXInfluence(+1,ipos,Qii);
    res |= addRemoveXInfluence(-1,ineg,Qii);
    res |= SVM_Scalar::setd(idiff,nv);

    locisTrained    = 0;
    (cholscratchcov)  = 0;
    (covlogdetcalced) = 0;

    return res;
}



int SVM_Scalar_rff::setNRff(int nv)
{
    int res = 0;
    int i,j,m;

    NiceAssert( nv >= 0 );

    res |= fixupfeatures();

    if ( locNRff > nv )
    {
        int reindim = ( ReOnly() ? INDIM : (INDIM/2) );
        int NRffval = locNRff;

        for ( j = NRffval-1 ; j >= nv ; --j )
        {
            if ( !ReOnly() )
            {
                for ( i = reindim-1 ; i >= 0 ; --i )
                {
                    m = (j*reindim)+i+(reindim*locNRff);

                    res |= SVM_Scalar::removeTrainingVector(N()+m);
                    locaw.remove(m);

                    locB.removeRowCol(m);
                    loca.remove(m);
                    locc.remove(m);
                    locphase.remove(m);
                    BFoff.remove(m);

                    if ( Bacgood )
                    {
                        locBF.remove(m,locB,BFoff);
                    }
                }
            }

            for ( i = reindim-1 ; i >= 0 ; --i )
            {
                m = ((j*reindim)+i);

                res |= SVM_Scalar::removeTrainingVector(N()+m);
                locaw.remove(m);

                locB.removeRowCol(m);
                loca.remove(m);
                locc.remove(m);
                locphase.remove(m);
                BFoff.remove(m);

                if ( Bacgood )
                {
                    locBF.remove(m,locB,BFoff);
                }

                locNRff--;
            }
        }

        locisTrained    = 0;
        (cholscratchcov)  = 0;
        (covlogdetcalced) = 0;
    }

    if ( locNRff < nv )
    {
        int reindim = ( ReOnly() ? INDIM : (INDIM/2) );
        int NRffval = locNRff;

        double lengthscale = calcRefLengthScale();

        Vector<int> ii;

        for ( j = NRffval ; j < nv ; ++j )
        {
            locNRff++; // Do this now so that it's set correctly when K2 callback occurs!

            SparseVector<gentype> xxa;

            xxa.indalign(indKey());

            for ( i = 0 ; i < xxa.nindsize() ; ++i )
            {
                if ( locddtype == 1 )
                {
                    ; // none
                }

                else if ( locddtype == 3 )
                {
                    //inline double &randnfill(double &res, double mu = 0, double sig = 1);     // Normal
                    randnfill(xxa.direref(i).force_double()); //,0,1/lengthscale);

                    xxa.direref(i).force_double() /= lengthscale;
                }

                else if ( locddtype == 4 )
                {
                    //inline double &randCfill(double &res, double a = 0, double b = 1);        // Cauchy
                    randCfill(xxa.direref(i).force_double()); //,0,1/lengthscale);

                    xxa.direref(i).force_double() /= lengthscale;
                }

                else if ( locddtype == 13 )
                {
                    //inline double &randufill(double &res, double a = 0, double b = 1);        // Uniform
                    //randufill(xxa.direref(i).force_double(),-NUMBASE_PI/lengthscale,NUMBASE_PI/lengthscale);
                    randufill(xxa.direref(i).force_double(),-NUMBASE_PI,NUMBASE_PI);

                    xxa.direref(i).force_double() /= lengthscale;
                }

                else
                {
                    NiceAssert( 19 == locddtype );

                    //inline double &randefill(double &res, double lambda = 1);                 // Exponential
                    //randefill(xxa.direref(i).force_double(),lengthscale);
                    randefill(xxa.direref(i).force_double());

                    xxa.direref(i).force_double() /= lengthscale;
                }
            }

            for ( i = 0 ; i < reindim ; ++i )
            {
                m = (j*reindim)+i;

                if ( NRffRep() )
                {
                    Vector<gentype> &depindpart = xxa.f4("&",7).force_vector(NRffRep());

                    depindpart = 0.0_gent;
                    depindpart("&",i%NRffRep()) = 1.0;
                }

                locaw.add(m);
                locaw("&",m) = ( tunev() == -1 ) ? -locminv : locminv;
                res |= SVM_Scalar::addTrainingVector(N()+m,0.0_gent,xxa);

                locB.addRowCol(m);
                loca.add(m);
                locc.add(m);
                locphase.add(m);
                randufill(locphase("&",m),-NUMBASE_PI,NUMBASE_PI);
                BFoff.append(m,1/(C()*locminv));
            }
        }

        if ( !ReOnly() )
        {
            for ( j = NRffval ; j < nv ; ++j )
            {
                for ( i = 0 ; i < reindim ; ++i )
                {
                    m = (j*reindim)+i+(reindim*locNRff);

                    SparseVector<gentype> xxx(x(N()+m-(reindim*locNRff)));
                    locaw.add(m);
                    locaw("&",m) = ( tunev() == -1 ) ? -locminv : locminv;
                    res |= SVM_Scalar::addTrainingVector(N()+m,0.0_gent,xxx);

                    locB.addRowCol(m);
                    loca.add(m);
                    locc.add(m);
                    locphase.add(m);
                    randufill(locphase("&",m),-NUMBASE_PI,NUMBASE_PI);
                    BFoff.append(m,1/(C()*locminv));
                }
            }
        }

        for ( j = NRffval ; j < nv ; ++j )
        {
            for ( i = 0 ; i < reindim ; ++i )
            {
                m = (j*reindim)+i;

                ii.add(ii.size());
                ii("&",ii.size()-1) = m;
            }

            if ( !ReOnly() )
            {
                for ( i = 0 ; i < reindim ; ++i )
                {
                    m = (j*reindim)+i+(reindim*nv);

                    ii.add(ii.size());
                    ii("&",ii.size()-1) = m;
                }
            }
        }

        res |= updateBAC(ii);

        locisTrained    = 0;
        (cholscratchcov)  = 0;
        (covlogdetcalced) = 0;
    }

    return res;
}

int SVM_Scalar_rff::setNRffRep(int nv)
{
    int res = 0;

    res |= fixupfeatures();

    if ( nv != NRffRep() )
    {
        res = 0;

        if ( ( ( NRffRep() == 0 ) && ( nv == 1 ) ) || ( ( NRffRep() == 1 ) && ( nv == 0 ) ) )
        {
            int reindim = ( ReOnly() ? INDIM : (INDIM/2) );
            int NRffval = NRff();

            // INDIM is unchanged, but :::: 7 is added or removed from random features
            // NOTE: need to change x as a block to ensure that :::: 7 is consistent!

            retVector<SparseVector<gentype> > tmpvj;
            Vector<SparseVector<gentype> > xxa(x()(N(),1,N()+calcwdim()-1,tmpvj));

            int i,j,m;

            for ( j = 0 ; j < NRffval ; ++j )
            {
                for ( i = 0 ; i < reindim ; ++i )
                {
                    m = (j*reindim)+i;

                    if ( !nv )
                    {
                        xxa("&",m).zerof4i(7);
                    }

                    else
                    {
                        Vector<gentype> &depindpart = xxa("&",m).f4("&",7).force_vector(nv);

                        depindpart = 0.0_gent;
                        depindpart("&",i%nv) = 1.0;
                    }

                    if ( !ReOnly() )
                    {
                        m += reindim*NRffval;

                        if ( !nv )
                        {
                            xxa("&",m).zerof4i(7);
                        }

                        else
                        {
                            Vector<gentype> &depindpart = xxa("&",m).f4("&",7).force_vector(nv);

                            depindpart = 0.0_gent;
                            depindpart("&",i%nv) = 1.0;
                        }
                    }
                }
            }

            midxset = true; // suppress errors in K2 temporarily
            retVector<int> tmpvjj;
            Vector<int> ii(cntintvec(calcwdim(),tmpvjj));
            ii += N();
            res |= SVM_Scalar::setx(ii,xxa);
            midxset = false; // reenable errors in K2

            locNRffRep = nv; // This won't change INDIM
            locbias.resize(nv ? nv : 1);
        }

        else if ( NRffRep() < nv )
        {
            int reindim = ( ReOnly() ? INDIM : (INDIM/2) );
            int NRffval = NRff();

            // First, we need to update the existing x vectors to fix dimension of :::: 7

            retVector<SparseVector<gentype> > tmpvj;
            Vector<SparseVector<gentype> > xxa(x()(N(),1,N()+calcwdim()-1,tmpvj));

            int i,j,m;

            for ( j = 0 ; j < NRffval ; ++j )
            {
                for ( i = 0 ; i < reindim ; ++i )
                {
                    m = (j*reindim)+i;

                    {
                        Vector<gentype> &depindpart = xxa("&",m).f4("&",7).force_vector(nv);

                        depindpart = 0.0_gent;
                        depindpart("&",i%nv) = 1.0;
                    }

                    if ( !ReOnly() )
                    {
                        m += reindim*NRffval;

                        {
                            Vector<gentype> &depindpart = xxa("&",m).f4("&",7).force_vector(nv);

                            depindpart = 0.0_gent;
                            depindpart("&",i%nv) = 1.0;
                        }
                    }
                }
            }

            midxset = true; // suppress errors in K2 temporarily
            retVector<int> tmpvjj;
            Vector<int> jj(cntintvec(calcwdim(),tmpvjj));
            jj += N();
            res |= SVM_Scalar::setx(jj,xxa);
            midxset = false; // reenable errors in K2

            // Now we can update the rest

            int oldreindim;

            locNRffRep = nv;
            locbias.resize(nv ? nv : 1);

            oldreindim = reindim;
            reindim    = ( ReOnly() ? INDIM : (INDIM/2) );

            Vector<int> ii;

            for ( j = 0 ; j < NRffval ; ++j )
            {
                for ( i = oldreindim ; i < reindim ; ++i )
                {
                    m = (j*reindim)+i;

                    SparseVector<gentype> xxb(x(N()+(j*reindim)));

                    Vector<gentype> &depindpart = xxb.f4("&",7).force_vector(nv);

                    depindpart = 0.0_gent;
                    depindpart("&",i%nv) = 1.0;

                    locaw.add(m);
                    locaw("&",m) = ( tunev() == -1 ) ? -locminv : locminv;
                    res |= SVM_Scalar::addTrainingVector(N()+m,0.0_gent,xxb);

                    locB.addRowCol(m);
                    loca.add(m);
                    locc.add(m);
                    locphase.add(m);
                    randufill(locphase("&",m),-NUMBASE_PI,NUMBASE_PI);
                    BFoff.append(m,1/(C()*locminv));

                    ii.add(ii.size());
                    ii("&",ii.size()-1) = m;
                }
            }

            if ( !ReOnly() )
            {
                for ( j = 0 ; j < NRffval ; ++j )
                {
                    for ( i = oldreindim ; i < reindim ; ++i )
                    {
                        m = (j*reindim)+i+(reindim*NRffval);

                        SparseVector<gentype> xxb(x(N()+m-1));

                        Vector<gentype> &depindpart = xxb.f4("&",7).force_vector(nv);

                        depindpart = 0.0_gent;
                        depindpart("&",i%nv) = 1.0;

                        locaw.add(m);
                        locaw("&",m) = ( tunev() == -1 ) ? -locminv : locminv;
                        res |= SVM_Scalar::addTrainingVector(N()+m,0.0_gent,xxb);

                        locB.addRowCol(m);
                        loca.add(m);
                        locc.add(m);
                        locphase.add(m);
                        randufill(locphase("&",m),-NUMBASE_PI,NUMBASE_PI);
                        BFoff.append(m,1/(C()*locminv));

                        ii.add(ii.size());
                        ii("&",ii.size()-1) = m;
                    }
                }
            }

            res |= updateBAC(ii);
        }

        else if ( NRffRep() > nv )
        {
            int oldreindim = INDIM;
            locNRffRep = nv;
            locbias.resize(nv ? nv : 1);
            int reindim = INDIM;
            int NRffval = NRff();

            retVector<SparseVector<gentype> > tmpvj;
            Vector<SparseVector<gentype> > xxa(x()(N(),1,N()+calcwdim()-1,tmpvj));

            int i,j,m;

            if ( !ReOnly() )
            {
                for ( j = NRffval-1 ; j >= 0 ; --j )
                {
                    for ( i = oldreindim-1 ; i >= 0 ; ++i )
                    {
                        m = (j*oldreindim)+i+(oldreindim*NRffval);

                        if ( i >= reindim )
                        {
                            xxa.remove(m);
                            res |= SVM_Scalar::removeTrainingVector(N()+m);
                        }

                        else if ( !nv )
                        {
                            xxa("&",m).zerof4i(7);
                        }

                        else
                        {
                            Vector<gentype> &depindpart = xxa("&",m).f4("&",7).force_vector(nv);

                            depindpart = 0.0_gent;
                            depindpart("&",i%nv) = 1.0;
                        }
                    }
                }
            }

            for ( j = NRffval-1 ; j >= 0 ; --j )
            {
                for ( i = oldreindim-1 ; i >= 0 ; ++i )
                {
                    m = (j*oldreindim)+i;

                    if ( i >= reindim )
                    {
                        xxa.remove(m);
                        res |= SVM_Scalar::removeTrainingVector(N()+m);
                    }

                    else if ( !nv )
                    {
                        xxa("&",m).zerof4i(7);
                    }

                    else
                    {
                        Vector<gentype> &depindpart = xxa("&",m).f4("&",7).force_vector(nv);

                        depindpart = 0.0_gent;
                        depindpart("&",i%nv) = 1.0;
                    }
                }
            }

            midxset = true; // suppress errors in K2 temporarily
            retVector<int> tmpvjj;
            Vector<int> ii(cntintvec(calcwdim(),tmpvjj));
            ii += N();
            res |= SVM_Scalar::setx(ii,xxa);
            midxset = false; // reenable errors in K2
        }

        locisTrained    = 0;
        (cholscratchcov)  = 0;
        (covlogdetcalced) = 0;
    }

    return res;
}



int SVM_Scalar_rff::setReOnly(int nv)
{
    int res = 0;

    NiceAssert( ( nv == 0 ) || ( nv == 1 ) );

    res |= fixupfeatures();

    if ( nv && !ReOnly() )
    {
        int oldwdim = calcwdim();
        int newwdim = oldwdim/2;

        res |= SVM_Scalar::removeTrainingVector(N()+newwdim,oldwdim-newwdim);

        int i;

        for ( i = oldwdim-1 ; i >= newwdim ; --i )
        {
            locaw.remove(i);
            locB.removeRowCol(i);
            loca.remove(i);
            locc.remove(i);
            locphase.remove(i);
            BFoff.remove(i);

            if ( Bacgood )
            {
                locBF.remove(i,locB,BFoff);
            }
        }

        locReOnly = nv;

        locisTrained    = 0;
        (cholscratchcov)  = 0;
        (covlogdetcalced) = 0;
    }

    else if ( !nv && ReOnly() )
    {
        int oldwdim = calcwdim();
        int newwdim = oldwdim*2;

        retVector<int> tmpvj;
        retVector<gentype> tmpvjj;
        retVector<SparseVector<gentype> > tmpvkk;
        retVector<double> tmpvjjj;
        retVector<double> tmpvkkk;
        Vector<int> jj(cntintvec(oldwdim,tmpvj));
        jj += N();
        Vector<gentype> yyy(SVM_Scalar::y()(jj,tmpvjj));
        Vector<SparseVector<gentype> > xxx(SVM_Scalar::x()(jj,tmpvkk));
        SVM_Scalar::addTrainingVector(N()+oldwdim,yyy,xxx,onedoublevec(oldwdim,tmpvjjj),onedoublevec(oldwdim,tmpvkkk));

        locaw.resize(newwdim);
        retVector<double> tmpvm;
        retVector<double> tmpvn;
        locaw("&",oldwdim,1,newwdim-1,tmpvm) = locaw(0,1,oldwdim-1,tmpvn);

        int i;

        for ( i = oldwdim ; i < newwdim ; ++i )
        {
            locB.addRowCol(i);
            loca.add(i);
            locc.add(i);
            locphase.add(i);
            randufill(locphase("&",i),-NUMBASE_PI,NUMBASE_PI);
            BFoff.append(i,1/(C()*locminv));
        }

        retVector<int> tmpvl;
        Vector<int> ii(cntintvec(newwdim-oldwdim,tmpvl));
        ii += oldwdim;

        locReOnly = nv;

        res |= updateBAC(ii);

        locisTrained    = 0;
        (cholscratchcov)  = 0;
        (covlogdetcalced) = 0;
    }

    return res;
}

int SVM_Scalar_rff::setLSRffoirRFFDist(double lsrffscale, int scalelsrff)
{
    int res = 0;

    if ( featGood )
    {
        res = 1;

        int i,j;
        int wdim = calcwdim();

        double lengthscale(calcRefLengthScale());

        retVector<SparseVector<gentype> > tmpvj;
        Vector<SparseVector<gentype> > xtemp(x()(N(),1,N()+wdim-1,tmpvj));

        errstream() << "@\b";

        for ( i = 0 ; i < wdim ; ++i )
        {
            for ( j = 0 ; j < xtemp(i).nindsize() ; ++j )
            {
                if ( locddtype == 1 )
                {
                    ; // none
                }

                else if ( locddtype == 3 )
                {
                    if ( scalelsrff )
                    {
                        xtemp("&",i).direref(j).force_double() /= lsrffscale;
                    }

                    else
                    {
                        //inline double &randnfill(double &res, double mu = 0, double sig = 1);     // Normal
                        randnfill(xtemp("&",i).direref(i).force_double()); //,0,1/lengthscale);

                        xtemp("&",i).direref(i).force_double() /= lengthscale;
                    }
                }

                else if ( locddtype == 4 )
                {
                    if ( scalelsrff )
                    {
                        xtemp("&",i).direref(j).force_double() /= lsrffscale;
                    }

                    else
                    {
                        //inline double &randCfill(double &res, double a = 0, double b = 1);        // Cauchy
                        randCfill(xtemp("&",i).direref(i).force_double()); //,0,1/lengthscale);

                        xtemp("&",i).direref(i).force_double() /= lengthscale;
                    }
                }

                else if ( locddtype == 13 )
                {
                    if ( scalelsrff )
                    {
                        xtemp("&",i).direref(j).force_double() /= lsrffscale;
                    }

                    else
                    {
                        //inline double &randufill(double &res, double a = 0, double b = 1);        // Uniform
                        //randufill(xtemp("&",i).direref(i).force_double(),-NUMBASE_PI/lengthscale,NUMBASE_PI/lengthscale);
                        randufill(xtemp("&",i).direref(i).force_double(),-NUMBASE_PI,NUMBASE_PI);

                        xtemp("&",i).direref(i).force_double() /= lengthscale;
                    }
                }

                else
                {
                    NiceAssert( 19 == locddtype );

                    if ( scalelsrff )
                    {
                        xtemp("&",i).direref(j).force_double() /= lsrffscale;
                    }

                    else
                    {
                        //inline double &randefill(double &res, double lambda = 1);                 // Exponential
                        //randefill(xtemp("&",i).direref(i).force_double(),lengthscale);
                        randefill(xtemp("&",i).direref(i).force_double());

                        xtemp("&",i).direref(i).force_double() /= lengthscale;
                    }
                }
            }
        }

        retVector<int> tmpvt;
        Vector<int> ii(cntintvec(wdim,tmpvt));
        ii += N();
        res |= SVM_Scalar::setx(ii,xtemp);
        ii -= N();
        res |= updateBAC(ii);
    }

    locisTrained    = 0;
    (cholscratchcov)  = 0;
    (covlogdetcalced) = 0;

    return res;
}

int SVM_Scalar_rff::setLSRff(double nv)
{
    NiceAssert( nv > 0 );

    int res = 0;

    res |= fixupfeatures();

    if ( xlsrff != nv )
    {
        double oldlsrff(LSRff());
        xlsrff = nv;

        res = setLSRffoirRFFDist(xlsrff/oldlsrff,1);
    }

    (cholscratchcov)  = 0;
    (covlogdetcalced) = 0;

    return res;
}

int SVM_Scalar_rff::setLSRffandRFFDist(double nv, int nvb)
{
    NiceAssert( nv > 0 );

    int res = 0;

    res |= fixupfeatures();

    if ( ( xlsrff != nv ) && ( locddtype != nv ) )
    {
        locddtype = nvb;

        double oldlsrff(LSRff());
        xlsrff = nv;

        res = setLSRffoirRFFDist(xlsrff/oldlsrff,1);
    }

    else if ( xlsrff != nv )
    {
        double oldlsrff(LSRff());
        xlsrff = nv;

        res = setLSRffoirRFFDist(xlsrff/oldlsrff,1);
    }

    else if ( locddtype != nv )
    {
        locddtype = nvb;

        res = setLSRffoirRFFDist();
    }

    (cholscratchcov)  = 0;
    (covlogdetcalced) = 0;

    return res;
}

int SVM_Scalar_rff::setRFFDist(int nv)
{
    NiceAssert( nv > 0 );

    int res = 0;

    res |= fixupfeatures();

    if ( locddtype != nv )
    {
        locddtype = nv;

        res = setLSRffoirRFFDist();
    }

    (cholscratchcov)  = 0;
    (covlogdetcalced) = 0;

    return res;
}


int SVM_Scalar_rff::fixupfeatures(void)
{
    int res = 0;

    featGood = ( Bacgood = ( featGood && Bacgood ) );

    if ( !NRff() || !N() || ( featGood && Bacgood ) )
    {
        Bacgood  = true;
        featGood = true;
    }

    else
    {
        res = 1;

        int reindim = ( ReOnly() ? INDIM : (INDIM/2) );

        int i,j,k,m;
        int wdim = calcwdim();
        int NRffval = NRff();

        double lengthscale(calcRefLengthScale());

        Vector<SparseVector<gentype> > xxa(wdim);

        for ( j = 0 ; j < NRffval ; ++j )
        {
            xxa("&",j*reindim).indalign(indKey());

            for ( k = 0 ; k < xxa(j*reindim).nindsize() ; ++k )
            {
                if ( locddtype == 1 )
                {
                    ; // none
                }

                else if ( locddtype == 3 )
                {
                    //inline double &randnfill(double &res, double mu = 0, double sig = 1);     // Normal
                    randnfill(xxa("&",j*reindim).direref(k).force_double(),0,1/lengthscale);
                }

                else if ( locddtype == 4 )
                {
                    //inline double &randCfill(double &res, double a = 0, double b = 1);        // Cauchy
                    randCfill(xxa("&",j*reindim).direref(k).force_double(),0,1/lengthscale);
                }

                else if ( locddtype == 13 )
                {
                    //inline double &randufill(double &res, double a = 0, double b = 1);        // Uniform
                    randufill(xxa("&",j*reindim).direref(k).force_double(),-NUMBASE_PI/lengthscale,NUMBASE_PI/lengthscale);
                }

                else
                {
                    NiceAssert( 19 == locddtype );

                    //inline double &randefill(double &res, double lambda = 1);                 // Exponential
                    randefill(xxa("&",j*reindim).direref(k).force_double(),lengthscale);
                }
            }

            for ( i = 0 ; i < reindim ; ++i )
            {
                m = (j*reindim)+i;

                if ( i )
                {
                    xxa("&",m) = xxa(j*reindim);
                }

                if ( NRffRep() )
                {
                    Vector<gentype> &depindpart = xxa("&",m).f4("&",7).force_vector(NRffRep());

                    depindpart = 0.0_gent;
                    depindpart("&",i%NRffRep()) = 1.0;
                }

                if ( !ReOnly() )
                {
                    xxa("&",m+(reindim*NRffval)) = xxa("&",m);
                }
            }
        }

        retVector<int> tmpvt;
        Vector<int> ii(cntintvec(wdim,tmpvt));
        ii += N();
        res |= SVM_Scalar::setx(ii,xxa);
        ii -= N();
        res |= updateBAC(ii);

        locisTrained    = 0;
        (cholscratchcov)  = 0;
        (covlogdetcalced) = 0;
    }

    return res;
}






    double SVM_Scalar_rff::getGpRowRffPartNorm2(int i, const Vector<double> &ivl) const
    {
        int wdim = calcwdim();
        int Nval = N();
        int j;
        double tmp,res = 0;

        // We do this elementwise to prevent caching outside of random features

        for ( j = 0 ; j < wdim ; ++j )
        {
            tmp =  Gp()(j+Nval,i);
            res += (tmp*tmp/ivl(j));
        }

        return res;
    }

    Vector<double> &SVM_Scalar_rff::getGpRowRffPart(Vector<double> &res, int i) const
    {
        int wdim = calcwdim();
        int Nval = N();
        int j;

        NiceAssert( res.size() == wdim );

        // We do this elementwise to prevent caching outside of random features

//if ( i == 4 )
//{
//errstream() << "phantomxyz Nval = " << Nval << "\n";
//errstream() << "phantomxyz wdim = " << wdim << "\n";
//}
        for ( j = 0 ; j < wdim ; ++j )
        {
//if ( i == 4 )
//{
//errstream() << "phantomxyz Qiicalc(" << j+Nval << "," << i << ") = ...";
//}
            res("&",j) = Gp()(j+Nval,i);
//if ( i == 4 )
//{
//errstream() << "..." << res(j) << "\n";
//}
        }

        return res;
    }

    int SVM_Scalar_rff::setIsAct(int nv, int i, Vector<double> &Qii, bool xdiff, bool ydiff, bool cdiff, bool QiiIsPrecalced)
    {
//QiiIsPrecalced = false;
        int res = 0;

        if ( isact(i) && nv && ( isact(i) != nv ) )
        {
            res = addRemoveXInfluence(-1,i,Qii,xdiff,ydiff,cdiff,QiiIsPrecalced);
            isact("&",i) = nv;
            res = addRemoveXInfluence(+1,i,Qii,xdiff,ydiff,cdiff,true);
        }

        else if ( !isact(i) && nv )
        {
            isact("&",i) = nv;
            res = addRemoveXInfluence(+1,i,Qii,xdiff,ydiff,cdiff,QiiIsPrecalced);
        }

        else if ( isact(i) && !nv )
        {
            res = addRemoveXInfluence(-1,i,Qii,xdiff,ydiff,cdiff,QiiIsPrecalced);
            isact("&",i) = nv;
        }

        return res;
    }

    // dir = -1 if removing, +1 if adding

    int SVM_Scalar_rff::addRemoveXInfluence(double dir, int i, Vector<double> &Qii, bool xdiff, bool ydiff, bool cdiff, bool QiiIsPrecalced)
    {
//QiiIsPrecalced = false;
        int res = 0;

        const Vector<int> &dd = SVM_Scalar::d();

        if ( dd(i) && isact(i) )
        {
            // (g-z)^2 = 1/2 (w'x - z)^2 = 1/2 w'.(xx').w - w'.z

            res = 1;

            double CCii = calcCscalequick(i); // only relative costs here, rest in lambda
            int ddd = ( dd(i) == 2 ) ? isact(i) : dd(i);
            double effz = ( ddd == -1 ) ? (loczr(i)+calcepsvalquick(i)) : (loczr(i)-calcepsvalquick(i));

            if ( Bacgood )
            {
                if ( !QiiIsPrecalced )
                {
                    Qii.resize(calcwdim());

                    getGpRowRffPart(Qii,i);
                }

                if ( ( xdiff || cdiff ) && isQuadraticCost() )
                {
//if ( i == 4 )
//{
//errstream() << "phantomxyz locB = " << locB << "\n";
//errstream() << "phantomxyz loca = " << loca << "\n";
//errstream() << "phantomxyz Qii = " << Qii << "\n";
//errstream() << "phantomxyz dir*CCii = " << dir*CCii << "\n";
//}
                    locB.rankone(dir*CCii,Qii,Qii);
                    loca.scaleAdd(dir*CCii,Qii);
                    locBF.rankone(Qii,dir*CCii,locB,BFoff);
//if ( i == 4 )
//{
//errstream() << "phantomxyz after locB = " << locB << "\n";
//errstream() << "phantomxyz after loca = " << loca << "\n";
//errstream() << "phantomxyz after Gp() = " << Gp() << "\n";
//}
                }

                if ( xdiff || ydiff || cdiff )
                {
                    if ( isQuadraticCost() )
                    {
                        locc.scaleAdd(dir*effz*CCii,Qii);
                    }

                    else
                    {
                        locc.scaleAdd(dir*ddd*CCii,Qii);
                    }
                }
            }

            if ( ydiff || cdiff )
            {
                if ( isQuadraticCost() )
                {
                    locd += dir*CCii*effz;
                }

                else
                {
                    locd += dir*ddd*effz;
                }
            }

            if ( cdiff && isQuadraticCost() )
            {
                locNN += dir*CCii;
            }

            (cholscratchcov)  = 0;
            (covlogdetcalced) = 0;
        }

        return res;
    }

    int SVM_Scalar_rff::addRemoveXInfluence(double dir, const Vector<int> &ii, Vector<double> &Qii, bool xdiff, bool ydiff, bool cdiff)
    {
        int j,res = 0;

        for ( j = 0 ; j < ii.size() ; ++j )
        {
            const Vector<int> &dd = SVM_Scalar::d();

            int i = ii(j);

            if ( dd(i) && isact(i) )
            {
                res = 1;

                double CCii = calcCscalequick(i); // only relative costs here, rest in lambda
                int ddd = ( dd(i) == 2 ) ? isact(i) : dd(i);
                double effz = ( ddd == -1 ) ? (loczr(i)+calcepsvalquick(i)) : (loczr(i)-calcepsvalquick(i));

                if ( Bacgood )
                {
                    Qii.resize(calcwdim());

                    getGpRowRffPart(Qii,i);

                    if ( ( xdiff || cdiff ) && isQuadraticCost() )
                    {
                        locB.rankone(dir*CCii,Qii,Qii);
                        loca.scaleAdd(dir*CCii,Qii);
                        locBF.rankone(Qii,dir*CCii,locB,BFoff);
                    }

                    if ( xdiff || ydiff || cdiff )
                    {
                        if ( isQuadraticCost() )
                        {
                            locc.scaleAdd(dir*effz*CCii,Qii);
                        }

                        else
                        {
                            locc.scaleAdd(dir*ddd*CCii,Qii);
                        }
                    }
               }

                if ( ydiff || cdiff )
                {
                    if ( isQuadraticCost() )
                    {
                        locd += dir*CCii*effz;
                    }

                    else
                    {
                        locd += dir*ddd*CCii;
                    }
                }

                if ( cdiff && isQuadraticCost() )
                {
                    locNN += dir*CCii;
                }
            }

            (cholscratchcov)  = 0;
            (covlogdetcalced) = 0;
        }

        return res;
    }

    int SVM_Scalar_rff::updateBAC(const Vector<int> &ii)
    {
        int res = 0;

        if ( Bacgood && ii.size() )
        {
            const Vector<int> &dd = SVM_Scalar::d();

            res = 1;

            Vector<int> nfixind;  // Assume either d = -1,1,2 or d = 0, only include those with d != 0
            Vector<int> nfixindx; // Like above, but only if d = 2 or d != 0, isQuadraticCost
            Vector<int> nfixindy; // Indices in nfixind but not nfixindx

            Vector<double> CC;
            Vector<double> ddd;
            Vector<double> effz(loczr);

            {
                nfixind.resize(0);
                CC.resize(N());
                ddd.resize(N());
                int i,j;

                for ( i = 0 ; i < N() ; ++i )
                {
                    if ( isenabled(i) && isact(i) )
                    {
                        j = nfixind.size();

                        nfixind.add(j);
                        nfixind("&",j) = i;

                        CC("&",i)   = calcCscalequick(i); // only relative costs here, rest in lambda
                        ddd("&",i)  = ( dd(i) == 2 ) ? isact(i) : dd(i);
                        effz("&",i) = ( ddd(i) == -1 ) ? (loczr(i)+calcepsvalquick(i)) : (loczr(i)-calcepsvalquick(i));

                        if ( isQuadraticCost() )
                        {
                            j = nfixindx.size();

                            nfixindx.add(j);
                            nfixindx("&",j) = i;
                        }

                        else
                        {
                            j = nfixindy.size();

                            nfixindy.add(j);
                            nfixindy("&",j) = i;
                        }
                    }
                }
            }

            retVector<double> tmpvtux;
            retVector<double> tmpvtuy;

            retVector<double> tmpvrr;
            retVector<double> tmpvrs;

            retVector<double> tmpvsr;
            retVector<double> tmpvss;

            const Vector<double> &CCsubx = CC(nfixindx,tmpvtux);
            const Vector<double> &CCsuby = CC(nfixindy,tmpvtuy);

            //const Vector<double> &dddsubx = ddd(nfixindx,tmpvrr);
            const Vector<double> &dddsuby = ddd(nfixindy,tmpvrs);

            const Vector<double> &effzsubx = effz(nfixindx,tmpvsr);
            //const Vector<double> &effzsuby = effz(nfixindy,tmpvss);

            retVector<double> tmpvix;
            retVector<double> tmpviy;

            retVector<double> tmpviix;
            retVector<double> tmpviiy;

            retVector<double> tmpvqx;
            retVector<double> tmpvqy;

            retVector<double> tmpvjx;
            retVector<double> tmpvjjx;
            retVector<double> tmpvrx;

            int i,m,k;
            double adae;

            for ( i = 0 ; i < ii.size() ; ++i )
            {
                m = ii(i);

                const Vector<double> &Qiix = Gp()(N()+m,nfixindx,tmpvix,tmpviix,tmpvqx);
                const Vector<double> &Qiiy = Gp()(N()+m,nfixindy,tmpviy,tmpviiy,tmpvqy);

                for ( k = 0 ; k <= m ; ++k )
                {
                    const Vector<double> &Qjx = Gp()(N()+k,nfixindx,tmpvjx,tmpvjjx,tmpvrx);

                    threeProduct(locB("&",m,k),Qiix,Qjx,CCsubx);

                    locB("&",k,m) = locB(m,k);
                }

                twoProduct(loca("&",m),Qiix,CCsubx);

                locc("&",m)  = threeProduct(adae,Qiix,effzsubx,CCsubx);
                locc("&",m) += threeProduct(adae,Qiiy,dddsuby, CCsuby);
            }

             locBF.remake(locB,BFoff,DEFAULT_ZTOL,1);

            (cholscratchcov)  = 0;
            (covlogdetcalced) = 0;
        }

        return res;
    }

    int SVM_Scalar_rff::fixactiveset(Vector<double> &Qii, Vector<double> &g, Vector<double> &Eps, Vector<double> &vw, double b, int wdim, int Nval, int &notfirstcall)
    {
        Qii.resize(wdim);
        g.resize(Nval);

        int res = 0;
        const Vector<int> &dd = SVM_Scalar::d();

        int i;

        for ( i = 0 ; i < Nval ; ++i )
        {
            if ( dd(i) )
            {
                getGpRowRffPart(Qii,i);

                twoProduct(g("&",i),Qii,vw);

                g("&",i) += b;

                if ( ( +1 == dd(i) ) && isact(i) && ( g(i) >= loczr(i)-Eps(i) ) )
                {
                    notfirstcall = 0; // method 2 factorisation no longer good if active set changes
                    res = 1;

                    setIsAct(0,i,Qii);
                }

                else if ( ( -1 == dd(i) ) && isact(i) && ( g(i) <= loczr(i)+Eps(i) ) )
                {
                    notfirstcall = 0; // method 2 factorisation no longer good if active set changes
                    res = 1;

                    setIsAct(0,i,Qii);
                }

                else if ( ( 2 == dd(i) ) && isact(i) && ( g(i) <= loczr(i)+Eps(i) ) && ( g(i) >= loczr(i)-Eps(i) ) )
                {
                    notfirstcall = 0; // method 2 factorisation no longer good if active set changes
                    res = 1;

                    setIsAct(0,i,Qii);
                }

                else if ( ( 2 == dd(i) ) && ( isact(i) == +1 ) && ( g(i) >= loczr(i)+Eps(i) ) )
                {
                    notfirstcall = 0; // method 2 factorisation no longer good if active set changes
                    res = 1;

                    setIsAct(-1,i,Qii);
                }

                else if ( ( 2 == dd(i) ) && ( isact(i) == -1 ) && ( g(i) <= loczr(i)-Eps(i) ) )
                {
                    notfirstcall = 0; // method 2 factorisation no longer good if active set changes
                    res = 1;

                    setIsAct(+1,i,Qii);
                }

                else if ( ( +1 == dd(i) ) && !isact(i) && ( g(i) < loczr(i)-Eps(i) ) )
                {
                    notfirstcall = 0; // method 2 factorisation no longer good if active set changes
                    res = 1;

                    setIsAct(+1,i,Qii);
                }

                else if ( ( -1 == dd(i) ) && !isact(i) && ( g(i) > loczr(i)+Eps(i) ) )
                {
                    notfirstcall = 0; // method 2 factorisation no longer good if active set changes
                    res = 1;

                    setIsAct(-1,i,Qii);
                }

                else if ( ( 2 == dd(i) ) && !isact(i) && ( g(i) < loczr(i)-Eps(i) ) )
                {
                    notfirstcall = 0; // method 2 factorisation no longer good if active set changes
                    res = 1;

                    setIsAct(+1,i,Qii);
                }

                else if ( ( 2 == dd(i) ) && !isact(i) && ( g(i) > loczr(i)+Eps(i) ) )
                {
                    notfirstcall = 0; // method 2 factorisation no longer good if active set changes
                    res = 1;

                    setIsAct(-1,i,Qii);
                }
            }
        }

        return res;
    }






void SVM_Scalar_rff::fillCache(int Ns, int Ne)
{
    // Only want to refresh the upper part of Gp!

    Ns = ( Ns < N() ) ? N() : Ns;

    SVM_Scalar::fillCache(Ns,Ne);

    return;
}

gentype &SVM_Scalar_rff::K2(gentype &res, int ia, int ib, const gentype **pxyprod, const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, const vecInfo *xainfo, const vecInfo *xbinfo, int resmode) const
{
    int realN = N()+addingNow; //SVM_Scalar::N() - locNRff;

    if ( ( ia >= realN ) && ( ib >= realN ) )
    {
        // This is L

        ML_Base::K2(res,ia,ib,pxyprod,xa,xb,xainfo,xbinfo,resmode);
    }

    else
    {
        res.force_double() = K2(ia,ib,pxyprod,xa,xb,xainfo,xbinfo,resmode);
    }

    return res;
}

double SVM_Scalar_rff::K2(int ia, int ib, const gentype **pxyprod, const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, const vecInfo *xainfo, const vecInfo *xbinfo, int resmode) const
{
    // Overall:
    //
    // Gp = [ ... Qc' Qs' ]   ( Kij, followed by the feature map [ Qc' Qs' ] of xi under RFF)
    //      [ Qc  ... ... ]
    //      [ Qs  ... ... ]
    //
    // where Qc_ij = sqrt(2/M)*cos(<omega_i,x_j>)
    // where Qs_ij = sqrt(2/M)*sin(<omega_i,x_j>)

    int realN = N()+addingNow; //SVM_Scalar::N() - locNRff;
    double res = 0;

//bool dumpit = ( ( ia == 4 ) || ( ib == 4 ) );
    if ( ( ia >= realN ) && ( ib >= realN ) )
    {
        // This is L
        return ML_Base::K2(ia,ib,pxyprod,xa,xb,xainfo,xbinfo,resmode);
    }

    if ( ( ia < realN ) && ( ib < realN ) )
    {
        // Return inner product of random feature vectors.

//errstream() << "botleft?";
        int wdim = calcwdim();
        int Nval = realN;
        int j;
        double tmpx,tmpy;

        double dres = 0;

        // We do this elementwise to prevent caching outside of random features

        const Vector<double> &v = locaw;

        for ( j = 0 ; j < wdim ; ++j )
        {
            tmpx = K2(ia,j+Nval,nullptr,xa,nullptr,xainfo,nullptr,resmode);
            tmpy = K2(j+Nval,ib,nullptr,nullptr,xb,nullptr,xbinfo,resmode);

            dres += (v(j)*tmpx*tmpy);
        }

        return dres;
    }

    if ( ia < realN )
    {
        // Restrict ourselves to bottom left for cache

        int ii = ia;                          ia     = ib;     ib     = ii;
        const SparseVector<gentype> *xx = xa; xa     = xb;     xb     = xx;
        const vecInfo *xxinfo = xainfo;       xainfo = xbinfo; xbinfo = xxinfo;
    }

    if ( !xa ) { xa = &x(ia); }
    if ( !xb ) { xb = &x(ib); }

    if ( !xainfo ) { xainfo = &xinfo(ia); }
    if ( !xbinfo ) { xbinfo = &xinfo(ib); }

    // Now we know that ia > N() >= ib
    // Ordering: cos first, then sin

    int M = NRff();
    //int iia = ((ia-N())%M)+N(); // By aliasing back we can take advantage of any stored inner products (in theory)

    // The next bit is cut-and-paste from ml_base, with altK.K2 replaced as required
    {
        const SparseVector<gentype> *xanear = nullptr;
        const SparseVector<gentype> *xbnear = nullptr;

        const SparseVector<gentype> *xafar = nullptr;
        const SparseVector<gentype> *xbfar = nullptr;

        const SparseVector<gentype> *xafarfar = nullptr;
        const SparseVector<gentype> *xbfarfar = nullptr;

        const SparseVector<gentype> *xafarfarfar = nullptr;
        const SparseVector<gentype> *xbfarfarfar = nullptr;

        const vecInfo *xanearinfo = nullptr;
        const vecInfo *xbnearinfo = nullptr;

        const vecInfo *xafarinfo = nullptr;
        const vecInfo *xbfarinfo = nullptr;

        int ixa,iia,xalr,xarr,xagr,xagrR,iaokr,iaok,adiagr,agmuL,agmuR,agradOrder,agradOrderR,iaplanr,iaplan,iaset,iadenseint,iadensederiv;
        int ixb,iib,xblr,xbrr,xbgr,xbgrR,ibokr,ibok,bdiagr,bgmuL,bgmuR,bgradOrder,bgradOrderR,ibplanr,ibplan,ibset,ibdenseint,ibdensederiv;

        double arankL,arankR;
        double brankL,brankR;

        const gentype *ixatup = nullptr;
        const gentype *iiatup = nullptr;

        const gentype *ixbtup = nullptr;
        const gentype *iibtup = nullptr;

        SparseVector<gentype> *xauntang = nullptr;
        SparseVector<gentype> *xbuntang = nullptr;

        vecInfo *xainfountang = nullptr;
        vecInfo *xbinfountang = nullptr;

        Vector<int> sumind;
        Vector<double> sumweight;

        double iadiagoffset = 0;
        double ibdiagoffset = 0;

        int iavectset = 0;
        int ibvectset = 0;

        int loctanga = detangle_x(xauntang,xainfountang,xanear,xafar,xafarfar,xafarfarfar,xanearinfo,xafarinfo,ixa,iia,ixatup,iiatup,xalr,xarr,xagr,xagrR,iaokr,iaok,arankL,arankR,agmuL,agmuR,ia,adiagr,xa,xainfo,agradOrder,agradOrderR,iaplanr,iaplan,iaset,iadenseint,iadensederiv,sumind,sumweight,iadiagoffset,iavectset);
        int loctangb = detangle_x(xbuntang,xbinfountang,xbnear,xbfar,xbfarfar,xbfarfarfar,xbnearinfo,xbfarinfo,ixb,iib,ixbtup,iibtup,xblr,xbrr,xbgr,xbgrR,ibokr,ibok,brankL,brankR,bgmuL,bgmuR,ib,bdiagr,xb,xbinfo,bgradOrder,bgradOrderR,ibplanr,ibplan,ibset,ibdenseint,ibdensederiv,sumind,sumweight,ibdiagoffset,ibvectset);

        (void) loctanga; NiceAssert ( !( loctanga & 2048 ) );
        (void) loctangb; NiceAssert ( !( loctangb & 2048 ) );

        int issameset = ( iavectset == ibvectset ) ? 1 : 0;

        const SparseVector<gentype> *xai = xauntang ? xauntang : xa;
        const SparseVector<gentype> *xbi = xbuntang ? xbuntang : xb;

        const vecInfo *xainfoi = xainfountang ? xainfountang : xainfo;
        const vecInfo *xbinfoi = xbinfountang ? xbinfountang : xbinfo;

        if ( issameset )
        {
//if ( !cholscratchcov ) { errstream() << "phantomxK2 xai " << *xai << "\n"; }
//if ( !cholscratchcov ) { errstream() << "phantomxK2 xaj " << *xbi << "\n"; }
//if ( !cholscratchcov ) { errstream() << "phantomxK2 ia " << ia << "\n"; }
//if ( !cholscratchcov ) { errstream() << "phantomxK2 ib " << ib << "\n"; }
//if ( dumpit ) { errstream() << "phantomxK2 xai " << *xai << "\n"; }
//if ( dumpit ) { errstream() << "phantomxK2 xaj " << *xbi << "\n"; }
//if ( dumpit ) { errstream() << "phantomxK2 ia " << ia << "\n"; }
//if ( dumpit ) { errstream() << "phantomxK2 ib " << ib << "\n"; }
            res = K2ip(ia,ib,nullptr,xai,xbi,xainfoi,xbinfoi);
//if ( !cholscratchcov ) { errstream() << "phantomxK2 ip " << res << "\n"; }
//if ( dumpit ) { errstream() << "phantomxK2 ip " << res << "\n"; }

            //bool isBiasPart = ( (ia-N()) >= (M*INDIM) );                    // any constants in the feature map go here
            bool isSinPart  = ( !ReOnly() && ( (ia-realN) >= (M*INDIM)/2 ) ); // put sin block at end for simplicity in code!

            //NiceAssert( !isBiasPart );

            if ( locddtype == 1 )
            {
                res = (*xbi)(ia-realN);
//if ( !cholscratchcov ) { errstream() << "phantomxK2 res1 " << res << "\n"; }
//if ( dumpit ) { errstream() << "phantomxK2 res1 " << res << "\n"; }
            }

            else if ( isSinPart )
            {
                res = sqrt(1.0/((double) M))*sin(res);
//if ( !cholscratchcov ) { errstream() << "phantomxK2 res2 correct " << res << "\n"; }
//if ( dumpit ) { errstream() << "phantomxK2 res2 correct 1/sqrt(" << M << ") sin(" << res << ") = " << res << "\n"; }
            }

            else if ( !ReOnly() )
            {
                res = sqrt(1.0/((double) M))*cos(res);
//if ( !cholscratchcov ) { errstream() << "phantomxK2 res3 " << res << "\n"; }
//if ( dumpit ) { errstream() << "phantomxK2 res3 " << res << "\n"; }
            }

            else
            {
                res = sqrt(1.0/((double) M))*cos(res+locphase(ia-realN)); // added random phase here!
//if ( !cholscratchcov ) { errstream() << "phantomxK2 res4 " << res << "\n"; }
//if ( dumpit ) { errstream() << "phantomxK2 res4 " << res << "\n"; }
            }
        }

        else
        {
            res = 0.0;
        }

        if ( ia == ib )
        {
            res += iadiagoffset;
        }

        if ( xauntang ) { delete xauntang; }
        if ( xbuntang ) { delete xbuntang; }

        if ( xainfountang ) { delete xainfountang; }
        if ( xbinfountang ) { delete xbinfountang; }

        if ( iaokr || ibokr )
        {
            Vector<int> iiokr(2);
            Vector<int> iiok(2);
            Vector<const gentype *> xxalt(2);

            iiokr("&",0) = iaokr;
            iiok("&",0)  = iaok;
            xxalt("&",0) = (*xa).isf4indpresent(3) ? &((*xa).f4(3)) : &nullgentype();

            iiokr("&",1) = ibokr;
            iiok("&",1)  = ibok;
            xxalt("&",1) = (*xb).isf4indpresent(3) ? &((*xb).f4(3)) : &nullgentype();

            gentype UUres;

            (*UUcallback)(UUres,2,*this,iiokr,iiok,xxalt,defbasisUU);

            res *= (double) UUres;
//if ( dumpit ) { errstream() << "phantomxK2 res*= " << res << "\n"; }
        }

        if ( iaplanr || ibplanr )
        {
            Vector<int> iiplanr(2);
            Vector<int> iiplan(2);
            Vector<const gentype *> xxalt(2);

            iiplanr("&",0) = iaplanr;
            iiplan("&",0)  = iaplan;
            xxalt("&",0)   = (*xa).isf4indpresent(7) ? &((*xa).f4(7)) : &nullgentype();

            iiplanr("&",1) = ibplanr;
            iiplan("&",1)  = ibplan;
            xxalt("&",1)   = (*xb).isf4indpresent(7) ? &((*xb).f4(7)) : &nullgentype();

            gentype VVres;
            gentype kval(res);

            (*VVcallback)(VVres,2,kval,*this,iiplanr,iiplan,xxalt,defbasisVV);

            if ( VVres.isCastableToReal() || !midxset )
            {
//if ( dumpit ) { errstream() << "phantomxK2 res*=222 " << res << "\n"; }
                res = (double) VVres;
            }
        }
    }

//if ( dumpit ) { errstream() << "phantomxK2 return " << res << "\n"; }
    return res;
}

gentype &SVM_Scalar_rff::K2(gentype &res, int ia, int ib, const MercerKernel &altK, const gentype **pxyprod, const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, const vecInfo *xainfo, const vecInfo *xbinfo, int resmode) const
{
    // See above, but with different kernel

    int realN = N()+addingNow; //SVM_Scalar::N() - locNRff;

    if ( ( ia >= realN ) && ( ib >= realN ) )
    {
        // This is L

        return ML_Base::K2(res,ia,ib,altK,pxyprod,xa,xb,xainfo,xbinfo,resmode);
    }

    else
    {
        res.force_double() = K2(ia,ib,pxyprod,xa,xb,xainfo,xbinfo,resmode);
    }

    return res;
}

gentype &SVM_Scalar_rff::K2(gentype &res, int ia, int ib, const gentype &bias, const gentype **pxyprod, const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, const vecInfo *xainfo, const vecInfo *xbinfo, int resmode) const
{
    // See above, but with bias

    int realN = N()+addingNow; //SVM_Scalar::N() - locNRff;

    if ( ( ia >= realN ) && ( ib >= realN ) )
    {
        // This is L

        return ML_Base::K2(res,ia,ib,bias,pxyprod,xa,xb,xainfo,xbinfo,resmode);
    }

    else
    {
        K2(res,ia,ib,pxyprod,xa,xb,xainfo,xbinfo,resmode);
    }

    return res;
}



int SVM_Scalar_rff::reset(void)
{
    locaw = ( tunev() == -1 ) ? -locminv : locminv;

    return SVM_Scalar::reset();
}

int SVM_Scalar_rff::pretrain(void)
{
    int wdim = INDIM*NRff();
    //int vdim = NRff();

    fixupfeatures();

    SVM_Scalar::freeRandomFeatures(wdim); // NRff (vdim) here, not dim, as we only need H, not [ H H ; H H ]

    return 1;
}

int SVM_Scalar_rff::train(int &res, svmvolatile int &killSwitch)
{
    locisTrained = 1;
    res = 0;

    int wdim = INDIM*NRff();
    int vdim = NRff();
    int Ndim = N();

    if ( !vdim || !Ndim )
    {
        retVector<double> tmpva;

        setAlphaR(zerodoublevec(Ndim+wdim,tmpva));

        if ( isVarBias() )
        {
            setBiasR(0.0);
        }

        return 0;
    }

    errstream("$");
    pretrain();
    errstream("%");

    NiceAssert( ( 3 != outGrad() ) || ( 8 != inAdam() ) );

    Atscratch.resize(Ndim);
    Btscratch.resize(Ndim);

    epsscratch.resize(Ndim);
    Csscratch.resize(Ndim);

    // Pre setup epsilon for speed

    Vector<double> &Eps = epsscratch;
    Vector<double> &Csc = Csscratch;

    int i;

    for ( i = 0 ; i < N() ; ++i )
    {
        Eps("&",i) = calcepsvalquick(i);
        Csc("&",i) = calcCscalequick(i);
    }

    if ( 8 == inAdam() )
    {
        // Cheats method using SVM_Scalar training routine directly

        for ( i = 0 ; i < N() ; ++i )
        {
            SVM_Scalar::setCweight(i,Csc(i)*locC/(SVM_Scalar::C()));
            SVM_Scalar::setepsweight(i,calcepsscalequick(i)*loceps/(SVM_Scalar::eps()));
        }
    }

    int tres = 0;

    retVector<double> tmpvsub;
    retVector<double> tmpvsubb;
    retVector<double> tmpvsubbb;

    Vector<double> fullalpha(Ndim+wdim+1); // includes "extra" at end
    Vector<double> &subalpha = fullalpha("&",0,1,Ndim-1,tmpvsubb); // no "extra" at end

    Vector<double> &vv = locaw;
    Vector<double> &vw = fullalpha("&",Ndim,1,Ndim+wdim-1,tmpvsub); // no "extra" at end

    Vector<double> &vwaug = fullalpha("&",Ndim,1,Ndim+wdim,tmpvsubbb); // includes "extra" at end

    // NB: vw and vwaug *mostly* point to the same data, it's just that vwaug has an "extra" at the end

    double b;

    subalpha = 0.0;

    retVector<double> tmpva;

    vw = alphaR()(N(),1,N()+wdim-1,tmpva);
    b  = biasR();

    retVector<double> lkhafdf;

    vwaug("&",wdim) = b;

    int notfirstcall = 0;
    int fbused = 0;

    fixactiveset(Qiiscratch,gscratch,Eps,vw,b,wdim,Ndim,notfirstcall);

    notfirstcall = 0;

    if ( !tunev() )
    {
//        errstream("Start training for fixed weight method\n");
errstream("fwm");

        inintrain(res,killSwitch,vw,vwaug,vv,b,sigma(),innerAdam,notfirstcall,fbused);
    }

    else
    {
        (cholscratchcov)  = 0;
        (covlogdetcalced) = 0;

//        errstream("Start training for tuned weight method\n");
errstream("twm\n");

        intrain(res,killSwitch,vw,vwaug,vv,b,sigma(),1/D());
    }

    if ( 8 == inAdam() )
    {
        // Cheats method using SVM_Scalar training routine directly
        //
        // First grab alpha and bias

/*
        vw = 0.0;
        b  = SVM_Scalar::biasR();

        for ( i = 0 ; i < Ndim ; ++i )
        {
            Qiiscratch.resize(wdim);

            getGpRowRffPart(Qiiscratch,i)

            vw.scaleAdd(SVM_Scalar::alphaR()(i),Qiiscratch);
        }

        vw *= locaw;
*/

        // Then revert SVM_Scalar to normal operation

        for ( i = 0 ; i < N() ; ++i )
        {
            SVM_Scalar::setCweight(i,1.0);
            SVM_Scalar::setepsweight(i,1.0);
        }

        SVM_Scalar::freeRandomFeatures(wdim); // NRff (vdim) here, not dim, as we only need H, not [ H H ; H H ]
    }

    // Set alpha and v

    retVector<double> tmpvui;

    //setAlphaR(fullalpha("&",0,1,Ndim+wdim-1,tmpvui));
    setAlphaRF(fullalpha("&",0,1,Ndim+wdim-1,tmpvui),false);

    if ( isVarBias() )
    {
//errstream() << "phantomxxx b: " << b << "\n";
        setBiasR(b);
    }

    fixactiveset(Qiiscratch,gscratch,Eps,vw,b,calcwdim(),N(),notfirstcall);

    errstream() << "!"; //"!\n";




//    int wdim = INDIM*NRff();
//    int Nloc = N();

    Vector<double> Zy(wdim);
    retMatrix<double> tmpma;

    mult(Zy,Gp()(Ndim,1,Ndim+wdim-1,0,1,Ndim-1,tmpma),loczr);

    const Matrix<double> &B = locB;
    Vector<double> &ivl = (ivscratch);

    ivl.resize(wdim);
    (yscratch).resize(wdim);
    (cholscratch).resize(wdim,wdim);
    (cholscratchb).resize(wdim,wdim);

    if ( !(cholscratchcov) || !(covlogdetcalced) )
    {
        const Vector<double> &vvv  = locaw;

        for ( int ci = 0 ; ci < wdim ; ++ci )
        {
            ivl("&",ci)  = sigma() * ( ( vvv(ci) > locminv ) ? 1/vvv(ci) : 1/locminv );
        }
    }

    if ( !(covlogdetcalced) )
    {
        (covlogdet) = locB.logdet_naiveChol((cholscratchb),1.0,ivl);
        (covlogdetcalced) = 1;
    }

    Vector<double> alp(wdim);

    B.naiveCholInve(alp,Zy,1.0,ivl,(yscratch),(cholscratch),0,(cholscratchcov));

    return tres;
}







class vwgradset
{
public:
    vwgradset(double xlambda, const Vector<double> &xvv, const Matrix<double> &xB, const Vector<double> &xa, const Vector<double> &xc, const Vector<double> &xEps, const Vector<double> &xCsc, const Vector<int> &xisact, const Vector<double> &xloczr, Vector<int> &xAt, Vector<int> &xBt, Vector<double> &xg, double xd, double xNN, double &xb, int xwdim, int xmethod, int xcosttype, int xk, SVM_Scalar_rff &xcaller) :
        lambda(xlambda), vv(xvv), B(xB), a(xa), c(xc), Eps(xEps), Csc(xCsc), isact(xisact), loczr(xloczr), At(xAt), Bt(xBt), g(xg), d(xd), NN(xNN), b(xb), wdim(xwdim), method(xmethod), costtype(xcosttype), kk(xk), resvalue(0), itcnt(0), caller(xcaller) { ; }

    double lambda;

    const Vector<double> &vv;
    const Matrix<double> &B;
    const Vector<double> &a;
    const Vector<double> &c;
    const Vector<double> &Eps;
    const Vector<double> &Csc;
    const Vector<int> &isact;
    const Vector<double> &loczr;

    Vector<int> &At;
    Vector<int> &Bt;
    Vector<double> &g;

    double d;
    double NN;

    double &b;

    int wdim;
    int method;
    int costtype;
    int kk; // batch size for pegasos

    double resvalue;
    int itcnt;

    SVM_Scalar_rff &caller;

    int calcObj(double &res, const Vector<double> &vwaug, Vector<double> &vwgradaug, Vector<double> &vwgradgradaug, svmvolatile int &, int *nostop)
    {
        resvalue -= 1e-5;
        ++itcnt;
        *nostop = 0;

        int i,j;
        bool isVarBias = ( wdim != vwaug.size() ); //caller.isVarBias();
        int numpm = (caller.NNC(-1))+(caller.NNC(1));
        int N = caller.N();
        int Nnz = N-caller.NNC(0);
        const Vector<int> &dd = caller.SVM_Scalar::d(); // don't want to be refered back to svm_binary_rff, as this may be wrong!
        const Vector<double> &yRR = caller.yRR();
        const Matrix<double> &Gp = caller.Gp();
        retVector<double> tmpva;
        retVector<double> tmpvb;
        retVector<double> tmpvc;
        const Vector<double> &vw = vwaug(0,1,wdim-1,tmpva);
        Vector<double> &vwgrad = vwgradaug("&",0,1,wdim-1,tmpvb);
        Vector<double> &vwgradgrad = vwgradgradaug("&",0,1,wdim-1,tmpvc);
        double dummygrad;
        double dummygradgrad;
        double &bgrad = isVarBias ? vwgradaug("&",wdim) : dummygrad;
        double &bgradgrad = isVarBias ? vwgradgradaug("&",wdim) : dummygradgrad;

        int k = ( kk == -1 ) ? N : kk;

        bgrad = 0.0;
        bgradgrad = 1.0;

        if ( isVarBias )
        {
            b = vwaug(wdim);
        }

        if ( !numpm && caller.isQuadraticCost() && ( costtype == 1 ) && ( method != 3 ) && ( method != 5 ) && ( method != 7 ) )
        {
            // Ridge regression case

            // Update b if variable bias

            if ( isVarBias )
            {
                double dummy;
                b = (d-innerProduct(dummy,a,vw))/NN;
            }

            // Objective is:
            //
            // 1/2 [ vw ]' [ B + lambda inv(diag(v))   a  ] [ vw ] - [ vw ]' [ c ]
            //     [ b  ]  [ a'                        NN ] [ b  ]   [ b  ]  [ d ]
            //
            // KKT conditions:
            //
            // [ B + lambda inv(diag(v))   a  ] [ vw ] - [ c ] = [ 0 ]
            // [ a'                        NN ] [ b  ]   [ d ]   [ 0 ]
            //
            // We did b above, so the gradient becomes:
            //
            // e = B.vw + lambda vw./v + a.b - c (= 0 for optimality)
            // f = a'.vw + NN.b - d (= 0 for optimality)
            //
            // The objective itself becomes:
            //
            // 1/2 [ vw ]' ( [ B + lambda inv(diag(v))   a  ] [ vw ] - [ vw ]' [ c ] ) - 1/2 [ vw ]' [ c ]
            //     [ b  ]    [ a'                        NN ] [ b  ]   [ b  ]  [ d ]         [ b  ]  [ d ]
            //
            // = 1/2 [ vw ]' [ e ] - 1/2 [ vw ]' [ c ]
            //       [ b  ]  [ f ]       [ b  ]  [ d ]
            //
            // = 1/2 vw'.(e-c) + b.(f-d) - bd/2

            vwgrad  = vw;
            vwgrad *= B;

            NiceAssert( B.isSquare() );
            NiceAssert( a.size() == B.numRows() );
            NiceAssert( c.size() == B.numRows() );
            NiceAssert( vw.size() == B.numRows() );
            NiceAssert( vv.size() == B.numRows() );

            for ( i = 0 ; i < wdim ; ++i )
            {
                vwgrad("&",i) += (a(i)*b)-c(i);
                vwgrad("&",i) += lambda*vw(i)/vv(i);

                vwgradgrad("&",i) = lambda/vv(i);
                vwgradgrad("&",i) = ( vwgradgrad(i) < 1 ) ? 1.0 : vwgradgrad(i);
            }

            //if ( isVarBias )
            //{
            //    innerProduct(bgrad,a,vw);
            //    bgrad += NN*b;
            //    bgrad -= d;
            //
            //    //bgradgrad = lambda;
            //    //bgradgrad = ( bgradgrad < 1 ) ? 1.0 : bgradgrad;
            //}

            // Calculate objective

            //res = -b*d/2;
            res = 0;

            for ( i = 0 ; i < wdim ; ++i )
            {
                res += vw(i)*(vwgrad(i)-c(i))/2;
            }

            if ( isVarBias )
            {
                res += b*(bgrad-d)/2;
            }
        }

        else if ( method == 6 )
        {
            // Linear or ridge regression with inequalities, may be stochastic (method == 3,5) or regular (method = 1,2)

            // f(x_j) = w'.z(x_j) + b
            //
            // cost = sum_j ell(f(x_j),y_j) + lambda/2 (|w|.^2)./v
            // grad = sum_j ell'(f(x_j),y_j) ([ z(x_j) ; 1 ]) + lambda w./v
            // gradgrad = lambda inv(v)

            vwgrad = 0.0;
            bgrad  = 0.0;

            vwgradgrad = 1.0;
            bgradgrad  = 1.0;

            res = 0.0;

            double fxj,fxjgrad;
            int numrolls;

            for ( j = 0 ; j < N ; ++j )
            {
                if ( ( 3 == method ) || ( 5 == method ) )
                {
                    // find random, non-optimal training vector to evaluate

                    numrolls = 0;
                    //j = svm_rand()%N;
                    j = rand()%N;

                    if ( dd(j) )
                    {
                        // work out f(x_j)

                        fxj = b-yRR(j);

                        for ( i = 0 ; i < wdim ; ++i )
                        {
                            fxj += vw(i)*Gp(i+N,j);
                        }
                    }

                    else
                    {
                        fxj = 0;
                    }

                    while ( ( numrolls < N ) && !( ( dd(j) == 2 ) || ( ( dd(j) == -1 ) && ( fxj > 0 ) ) || ( ( dd(j) == +1 ) && ( fxj < 0 ) ) ) )
                    {
                        ++numrolls;
                        j = (j+1)%N;

                        if ( dd(j) )
                        {
                            // work out f(x_j)

                            fxj = b-yRR(j);

                            for ( i = 0 ; i < wdim ; ++i )
                            {
                                fxj += vw(i)*Gp(i+N,j);
                            }
                        }

                        else
                        {
                            fxj = 0;
                        }
                    }

                    if ( numrolls == N )
                    {
                        // Nothing to do here, all seem good

                        break;
                    }

                    else
                    {
                        *nostop = 1;
                    }
                }

                else if ( dd(j) )
                {
                    // work out f(x_j)

                    fxj = b-yRR(j);

                    for ( i = 0 ; i < wdim ; ++i )
                    {
                        fxj += vw(i)*Gp(i+N,j); // z(x_j) stored here!
                    }
                }

                else
                {
                    fxj = 0;
                }

                //if ( 4 == method )
                //{
                //    // Pegasos
                //
                //    fxj /= Nnz;
                //}

                if ( ( dd(j) == 2 ) || ( ( dd(j) == -1 ) && ( fxj > 0 ) ) || ( ( dd(j) == +1 ) && ( fxj < 0 ) ) )
                {
                    if ( costtype == 0 )
                    {
                        // ell(f,y) = |x-y|
                        // ell'(f,y) = sgn(x-y)
                        // ell''(f,y) "=" 1

                        fxjgrad = ( fxj > 0 ) ? +1 : -1;
                        res += norm1(fxj);
                    }

                    else
                    {
                        //if ( costtype == 1 )

                        // ell(f,y) = 1/2 (f-y)^2
                        // ell'(f,y) = (f-y)
                        // ell''(f,y) = 1

                        fxjgrad = fxj;
                        res += norm2(fxjgrad)/2;
                    }

                    fxjgrad *= Csc(j);

                    //fxjgrad /= Nnz;

//errstream() << "fxjgrad = " << fxjgrad << "\n";
                    for ( i = 0 ; i < wdim ; ++i )
                    {
                        vwgrad("&",i) += fxjgrad*Gp(i+N,j); // z(x_j) stored here
                    }

                    if ( isVarBias )
                    {
                        bgrad += fxjgrad;
                    }
                }

                if ( ( ( 3 == method ) || ( 5 == method ) ) )
                {
                    // Only do one variable for stochastic version

                    vwgrad *= (double) Nnz; // "scale out" to act like all variables are equally bad
                    bgrad  *= (double) Nnz;

                    res *= Nnz;

                    break;
                }
            }

            vwgradgrad = 1.0;
            bgradgrad  = 1.0;

            for ( i = 0 ; i < wdim ; ++i )
            {
                vwgrad("&",i) += lambda*vw(i)/vv(i);

                //vwgradgrad("&",i) = lambda/vv(i);
                //vwgradgrad("&",i) = ( vwgradgrad(i) < 1 ) ? 1.0 : vwgradgrad(i);

                res += lambda*vw(i)*vw(i)/vv(i);
            }

            bgrad *= 100;

            if ( ( 3 == method ) || ( 5 == method ) )
            {
                // In this case res is basically meaningless, so we make sure the final value is returned!

                res = resvalue;
            }

            //if ( isVarBias )
            //{
            //    bgradgrad += lambda*b;
            //
            //    //bgradgrad = lambda;
            //    //bgradgrad = ( vwgradgrad(wdim) < 1 ) ? 1.0 : bgradgrad;
            //}
        }

        else
        {
//errstream() << "phantomxyzwtf 0d;jadwfhuWE: " << method << ", " << k << "\n";
            // Pegasos
            //
            // Generate A_t
            //
            // Start with 0:N-1
            // Remove vectors constrained to 0
            // Randomly prune until size is k
            // Remove vectors satisfying relevant constraints

            int loc_kval = (itcnt%PEGASOS_FULLGRAD_CYCLE) ? k : N; // every PEGASOS_FULLGRAD_CYCLE iterations we work out the complete gradient

            retVector<int> tmpvaa;

            int ii;

            {
                Bt.resize(N) = cntintvec(N,tmpvaa);
                At.resize(0);

                for ( i = Bt.size()-1 ; i >= 0 ; --i )
                {
                    if ( !dd(Bt(i)) )
                    {
                        Bt.remove(i);
                    }
                }

                // Bt is now the list of possible vectors.  Next we go through,
                // randomly remove elements from Bt, and add them to At if they
                // are non-optimal.

                bool gotone = false;

                while ( ( At.size() < loc_kval ) && ( Bt.size() ) )
                {
//errstream() << "phantomxyza -why\n";
                    //i = svm_rand()%(Bt.size());
                    i = rand()%(Bt.size());
//errstream() << "phantomxyza -why i = " << i << "\n";
                    j = Bt(i);
//errstream() << "phantomxyza -why j = " << j << "\n";

                    Bt.remove(i);
//errstream() << "phantomxyza bleurgh\n";

                    gotone = true;

                    if ( dd(j) )
                    {
                        g("&",j) = b;

                        for ( ii = 0 ; ii < wdim ; ++ii )
                        {
                            g("&",j) += vw(ii)*Gp(ii+N,j);
                        }

//errstream() << "phantomxyza g(" << j << "," << dd(j) << ") = " << g(j) << ( ( +1 == dd(j) ) ? ">=" : "<=" ) << " " << ( ( dd(j) == +1 ) ? loczr(j)+Eps(j) : loczr(j)-Eps(j) ) << "\n";
                        if ( ( +1 == dd(j) ) && ( g(j) < loczr(j)-Eps(j) ) )
                        {
                            //if ( costtype )
                            {
                                g("&",j) -= (loczr(j)-Eps(j));
                            }
//errstream() << "phantomxyza 3 g(" << j << ") = " << g(j) << "\n";
                        }

                        else if ( ( 2 == dd(j) ) && ( g(j) < loczr(j)-Eps(j) ) )
                        {
                            //if ( costtype )
                            {
                                g("&",j) -= (loczr(j)-Eps(j));
                            }
//errstream() << "phantomxyza 6 g(" << j << ") = " << g(j) << "\n";
                        }

                        else if ( ( -1 == dd(j) ) && ( g(j) > loczr(j)+Eps(j) ) )
                        {
                            //if ( costtype )
                            {
                                g("&",j) -= (loczr(j)+Eps(j));
                            }
//errstream() << "phantomxyza 4 g(" << j << ") = " << g(j) << "\n";
                        }

                        else if ( ( 2 == dd(j) ) && ( g(j) > loczr(j)+Eps(j) ) )
                        {
                            //if ( costtype )
                            {
                                g("&",j) -= (loczr(j)+Eps(j));
                            }
//errstream() << "phantomxyza 5 g(" << j << ") = " << g(j) << "\n";
                        }

                        else
                        {
                            gotone = false;
                        }
                    }

                    if ( gotone )
                    {
                        At.add(At.size());
                        At("&",At.size()-1) = j;
                    }
                }

                if ( Bt.size() )
                {
                    *nostop = 1;
                }
            }

            loc_kval = At.size();

            // Do gradient over At

            // f(x_j) = w'.z(x_j) + b
            //
            // cost = sum_j ell(f(x_j),y_j) + lambda/2 (|w|.^2)./v
            // grad = sum_j ell'(f(x_j),y_j) ([ z(x_j) ; 1 ]) + lambda w./v
            // gradgrad = lambda inv(v)

//errstream() << "phantomxyza At = " << At << "\n";
            vwgrad = 0.0;
            bgrad  = 0.0;

            vwgradgrad = 1.0;
            bgradgrad  = 1.0;

            res = resvalue;

            double fxj,fxjgrad;

            for ( i = 0 ; i < At.size() ; ++i )
            {
                j = At(i);
//errstream() << "phantomxyzab j = " << j << "\n";

                // work out f(x_j)

                fxj = g(j);

                if ( costtype == 0 )
                {
                    // ell(f,y) = |f-y|
                    // ell'(f,y) = sgn(f-y)
                    // ell''(f,y) "=" 1

                    fxjgrad = ( fxj > 0 ) ? +1 : -1;
                }

                else
                {
                    //if ( costtype == 1 )

                    // ell(f,y) = 1/2 (f-y)^2
                    // ell'(f,y) = (f-y)
                    // ell''(f,y) = 1

                    fxjgrad = fxj;
                }

                fxjgrad *= Csc(j);

                for ( ii = 0 ; ii < wdim ; ++ii )
                {
                    vwgrad("&",ii) += fxjgrad*Gp(ii+N,j)/loc_kval;
                }

                if ( isVarBias )
                {
                    bgrad += fxjgrad/loc_kval;
                }
            }
//errstream() << "phantomxyzab fxj = " << fxj << "\n";
//errstream() << "phantomxyzab fxjgrad = " << fxjgrad << "\n";
//errstream() << "phantomxyzab abs2(fxjgrad) = " << abs2(fxjgrad) << "\n";
//errstream() << "phantomxyzab bgrad = " << bgrad << "\n";

            vwgradgrad = 1.0;
            bgradgrad  = 1.0;

            for ( ii = 0 ; ii < wdim ; ++ii )
            {
                vwgrad("&",ii) += lambda*vw(ii)/vv(ii);
            }
//errstream() << "phantomxyzab vwgrad = " << vwgrad << "\n";
//if ( !(*nostop) )
//{
//errstream() << "phantomxyzab vwgrad = " << absinf(vwgrad) << "\n";
//}
        }

        return 0;
    }
};

int calcvwObj(double &res, const Vector<double> &x, Vector<double> &gradx, Vector<double> &gradgradx, svmvolatile int &killSwitch, int *nostop, void *objargs);
int calcvwObj(double &res, const Vector<double> &x, Vector<double> &gradx, Vector<double> &gradgradx, svmvolatile int &killSwitch, int *nostop, void *objargs)
{
    vwgradset &calcit = *((vwgradset *) objargs);

    return calcit.calcObj(res,x,gradx,gradgradx,killSwitch,nostop);
}












double SVM_Scalar_rff::inintrain(int &res, svmvolatile int &killSwitch, Vector<double> &vw, Vector<double> &vwaug, Vector<double> &vv, double &b, double lambda, int method, int &notfirstcall, int &fbused)
{
    (cholscratchcov)  = 0;
    (covlogdetcalced) = 0;

    double tres = 0;
    res = 0;

    int i;
    int wdim = INDIM*NRff();
    int vdim = NRff();
    int Nval = N();
    int numpm = SVM_Scalar::NNC(-1)+SVM_Scalar::NNC(1);
    //int numeq = SVM_Scalar::NNC(2);

    // Get cached information (will update if stale)

    const Vector<int> &dd = SVM_Scalar::d();

    Matrix<double> &B = locB;

    Vector<double> &a = loca;
    Vector<double> &c = locc;

    double d  = locd;
    double NN = locNN;

    double zzta  = 1e-12; //1e-6; //1e-3; //zerotol();
    double zztna = 1e-12; //1e-3; //100*zerotol();
    double zztma = 1e-12; //1e-3; //100*zerotol();

    Vector<double> &Eps = epsscratch;
    Vector<double> &Csc = Csscratch;

    for ( i = 0 ; i < Nval ; ++i )
    {
        SVM_Scalar::resetKernel(0,i,0);
    }

    if ( isLinearCost() && ( NNC(2) == 0 ) )
    {
        // regularise to prevent matrix singularity!

        NN = lambda/locminv;
        //NN = lambda/(10*locminv);
        //NN = 1e6;
    }

    // inintrain solves, isQuadraticCost() case:
    //
    // 1/2 [ vw ]' [ B + lambda.diag(inv(v))   a  ] [ vw ] - [ vw ]' [ c ]
    //     [ b  ]  [ a'                        NN ] [ b  ]   [ b  ]  [ d ]

    // inintrain solves, isLinearCost() case:
    //
    // 1/2 [ vw ]' [ B   a  ] [ vw ] + | vw |' [ c ] - [ vw ]' [ c ]
    //     [ b  ]  [ a'  NN ] [ b  ]                   [ b  ]  [ d ]

//errstream() << "phantomxyz 0\n";
    if ( !numpm && isQuadraticCost() )
    {
//errstream() << "phantomxyz 0 type 1\n";
        // Simple version - no inequality constraints, default to matrix inversion

        if ( ( 0 == method ) || ( 2 == method ) || ( 6 == method ) || ( 7 == method ) || ( 8 == method ) )
        {
            if ( ( 2 != method ) || !notfirstcall )
            {
                (yscratch).resize(wdim);
                (cholscratch).resize(wdim,wdim);

                Vector<double> &ivl = (ivscratch);
                Vector<double> &ssl = sscratch;

                Vector<double> &q = qscratch;

                ivl.resize(wdim);
                ssl.resize(wdim);

                q.resize(wdim);

                q = 0.0;

                for ( i = 0 ; i < wdim ; ++i )
                {
                    // inv(diag(v)) = diag(ivl) = diag(Boff) - diag(ssl)

                    ivl("&",i)  = lambda * ( ( vv(i) > locminv ) ? 1/vv(i) : 1/locminv );
                    ssl("&",i)  = ivl(i) - BFoff(i);
                }

                if ( isFixedBias() && b )
                {
                    // B.w + b = c

                    c -= b;
                }

//errstream() << "phantomxyz 1: ivl = " << ivl << "\n";
//errstream() << "phantomxyz 2: vv = " << vv << "\n";
                if ( isLinearCost() && ( NNC(2) == 0 ) )
                {
                    // Direct matrix inversion with diagonal B

                    vw = c;
                    vw /= ivl;
                }

                else if ( ( 0 == method ) || ( 2 == method ) )
                {
                    // Direct matrix inversion

                    B.naiveCholInve(vw,c,1.0,ivl,(yscratch),(cholscratch),0,notfirstcall); // first 0 says "don't zero the upper part of cholscratch", second 0 says "calculate cholscratch"

                    (cholscratchcov)  = notfirstcall;
                    (covlogdetcalced) = 0;

                    if ( method == 0 )
                    {
                        // In this case we do inversion each time.
                        // If method == 1 then this acts as the starting point for future iterations.

                        notfirstcall = 0;
                    }
                }

                else
                {
                    // Accelerated matrix inversion using series expansion of inv((B+diag(Boff))-diag(ssl) in terms of
                    // inv(B+diag(Boff)), which can be pre-computed once and then used for each iteration.

//errstream() << "phantomxyz 3 c = " << c << "\n";
//errstream() << "phantomxyz 3\n";
//errstream() << "phantomxyz 3\n";
                    locBF.minverseOffset(vw,c,1.0,ssl,B,BFoff,fbused); // fbused is incremented if fallback calculation is required
//errstream() << "phantomxyz 4: " << vw << "\n";
                }

                if ( isVarBias() )
                {
                    // BB = B + lambda.diag(inv(v))
                    //
                    // [ w ] = inv( [ BB  a  ] ) [ c ]
                    // [ b ]        [ a'  NN ]   [ d ]
                    //
                    //       = [ inv(BB) - inv(BB).a inv( a'.inv(BB).a - NN ) a'.inv(BB)   inv(BB).a inv( a'.inv(BB).a - NN ) ] [ c ]
                    //         [                     inv( a'.inv(BB).a - NN ) a'.inv(BB)            -inv( a'.inv(BB).a - NN ) ] [ d ]
                    //
                    //       = [ inv(BB) - q inv( q'.a - NN ) q'    q inv( q'.a - NN ) ] [ c ]
                    //         [             inv( q'.a - NN ) q'     -inv( q'.a - NN ) ] [ d ]
                    //
                    //       = [ inv(BB).c - q ( inv( q'.a - NN ) q'.c - inv( q'.a - NN ) d ) ]
                    //         [                 inv( q'.a - NN ) q'.c - inv( q'.a - NN ) d   ]
                    //
                    //       = [ r - qb ]
                    //         [      b ]
                    //
                    // where: q = inv(BB).a
                    //        r = inv(BB).c
                    //        b = ( q'.c - d )/( q'.a - NN )
                    //
                    // If linear cost classifier: BB is diagonal
                    //                            a is zero
                    //                            NN is nonzero
                    //
                    // so: q = 0
                    //     r = c/ivl
                    //     b = d/NN
                    // so:

                    double s,t;

                    if ( isLinearCost() && ( NNC(2) == 0 ) )
                    {
                        //q = 0.0;
                        ////q = a;
                        ////q /= ivl;
                    }

                    else if ( ( 0 == method ) || ( 2 == method ) )
                    {
                        B.naiveCholInve(q,a,1.0,ivl,(yscratch),(cholscratch),0,notfirstcall);

                        (cholscratchcov)  = notfirstcall;
                        (covlogdetcalced) = 0;
                    }

                    else
                    {
                        locBF.minverseOffset(q,a,1.0,ssl,B,BFoff,fbused);
                    }

                    if ( isLinearCost() && ( NNC(2) == 0 ) )
                    {
                        b = d/NN;
                    }

                    else
                    {
                        double tmpdiv;

                        tmpdiv = twoProduct(s,q,a);
                        tmpdiv -= NN;

                        if ( fabs(tmpdiv) < 1e-6 )
                        {
                            tmpdiv = ( tmpdiv <= 0 ) ? -1e-6 : 1e-6;
                        }

                        b = ((twoProduct(t,q,c)-d)/tmpdiv);
                        vw.scaleAdd(-b,q);
                    }
                }

                if ( isFixedBias() )
                {
                    // Q = 1/2 vw'.B.vw + lambda/2 vw'.(vv.*vw) - vw'.c
                    // e = B.vw + lambda vv.*vw - c = 0
                    // Q = 1/2 vw'.e - 1/2 vw'.c = -1/2 vw'.c

                    twoProduct(tres,vw,c) /= -2;
                }

                else
                {
                    // Q = -1/2 [ vw ]'.[ c ]
                    //          [ b  ]  [ d ]

                    twoProduct(tres,vw,c) /= -2;
                    tres -= (b*d/2);
                }

                if ( isFixedBias() && b )
                {
                    // B.w + b = c

                    c += b;
                }

                if ( 2 == method )
                {
                    vw = vwstartpoint;
                    b  = bstartpoint;
                }
            }

            if ( 2 == method )
            {
                // Remainder is as per method == 1, with modified starting point

                vw = vwstartpoint;
                b  = bstartpoint;

                //double opttolis = ( !( ( method == 3 ) || ( method == 5 ) || ( method == 7 ) ) || ( ( method == 7 ) && ( locpegk == -1 ) ) ) ? Opttolb() : -1;
                double opttolis = Opttolb();

                vwgradset optinfo(lambda,vv,B,a,c,Eps,Csc,isact,loczr,Atscratch,Btscratch,gscratch.resize(Nval),d,NN,b,wdim,method,costtype,locpegk,*this);
                ininadamscratchpad.resize(isVarBias() ? wdim+1 : wdim);

                stopCond sc;

                sc.maxitcnt   = maxitcnt();
                sc.maxruntime = maxtraintime();
                sc.runtimeend = traintimeend();
                sc.tol        = opttolis;

                res = ADAMopt(tres,( isVarBias() ? vwaug : vw ),calcvwObj,"Optimisation in SVM random fourier features (vw loop)",killSwitch,lrb(),(void *) &optinfo,ininadamscratchpad,USE_ADAM,sc,ADAM_BETA1,ADAM_BETA2,ADAM_EPS,2);

                b = vwaug(wdim);
            }

            vwaug("&",wdim) = b;
        }

        else if ( ( 1 == method ) || ( 3 == method ) || ( 4 == method ) || ( 5 == method ) )
        {
lazygoto:
            // Optimisation info

            vwgradset optinfo(lambda,vv,B,a,c,Eps,Csc,isact,loczr,Atscratch,Btscratch,gscratch.resize(Nval),d,NN,b,wdim,method,costtype,locpegk,*this);
            ininadamscratchpad.resize(isVarBias() ? wdim+1 : wdim);

            // Call ADAM to do its thing!

            int useadam = ( ( 1 == method ) || ( 3 == method ) ) ? USE_ADAM : 0;
            //double schedconst = useadam ? SCHEDCONST : 1.0;
            double schedconst = SCHEDCONST;
            double normmax = 0;
            //double opttolis = ( !( ( method == 3 ) || ( method == 5 ) || ( method == 7 ) ) || ( ( method == 7 ) && ( locpegk == -1 ) ) ) ? Opttolb() : -1;
            double opttolis = Opttolb();

            stopCond sc;

            sc.maxitcnt   = maxitcnt();
            sc.maxruntime = maxtraintime();
            sc.runtimeend = traintimeend();
            sc.tol        = opttolis;

            res = ADAMopt(tres,( isVarBias() ? vwaug : vw ),calcvwObj,"Optimisation in SVM random fourier features (vw loop)",killSwitch,lrb(),(void *) &optinfo,ininadamscratchpad,useadam,sc,ADAM_BETA1,ADAM_BETA2,ADAM_EPS,2,schedconst,normmax,locminv);

            b = vwaug(wdim);
        }
    }

    else if ( 8 == method )
    {
        Vector<double> &Qii = Qiiscratch;

        Qii.resize(wdim);

        //for ( i = 0 ; i < Nval ; ++i )
        //{
        //    SVM_Scalar::resetKernel(0,i,0);
        //}

//errstream() << "phantomxyzabc ==10: " << Gp() << "\n";
        SVM_Scalar::train(res,killSwitch);
//errstream() << "phantomxyzabc ==10: " << SVM_Scalar::alphaR() << "\n";
//errstream() << "phantomxyzabc ==10: " << *this << "\n";

        vw = 0.0;
        b  = SVM_Scalar::biasR();

        for ( i = 0 ; i < Nval ; ++i )
        {
            getGpRowRffPart(Qii,i);

            vw.scaleAdd(SVM_Scalar::alphaR()(i),Qii);
        }

        vw *= locaw;

        if ( isFixedBias() )
        {
            // Q = 1/2 vw'.B.vw + lambda/2 vw'.(vv.*vw) - vw'.c
            // e = B.vw + lambda vv.*vw - c = 0
            // Q = 1/2 vw'.e - 1/2 vw'.c = -1/2 vw'.c

            twoProduct(tres,vw,c) /= -2;
        }

        else
        {
            // Q = -1/2 [ vw ]'.[ c ]
            //          [ b  ]  [ d ]

            twoProduct(tres,vw,c) /= -2;
            tres -= (b*d/2);
        }

        vwaug("&",wdim) = b;
    }

    else if ( 6 == method )
    {
//errstream() << "phantomxyz 0 type 2\n";
        // Method with inequality constraints.  For simplicity we assume
        // either *all* inequalities or *no* inequalities, and linear cost.

        //NiceAssert( SVM_Scalar::NNC(2) == wdim );
        //NiceAssert( isLinearCost() );

        if ( ( 0 == method ) || ( 2 == method ) || ( 6 == method ) )
        {
            if ( ( 2 != method ) || !notfirstcall )
            {
                (yscratch).resize(wdim);
                (cholscratch).resize(wdim,wdim);

                Vector<double> &ivl = (ivscratch);
                Vector<double> &ssl = sscratch;

                Vector<double> &q = qscratch;

                Vector<double> &vwold = vwoldscratch;
                Vector<double> &dvw   = dvwscratch;

                Vector<double> &g  = gscratch;
                Vector<double> &dg = dgscratch;

                Vector<double> &Qii = Qiiscratch;

                double bold;
                double db = 0;

                ivl.resize(wdim);
                ssl.resize(wdim);

                q.resize(wdim);

                vwold.resize(wdim);
                dvw.resize(wdim);

                g.resize(Nval);
                dg.resize(Nval);

                Qii.resize(wdim);

                q = 0.0;

                // Make sure we start in the 1/lambda sphere

                for ( i = 0 ; i < wdim ; ++i )
                {
                    ivl("&",i)  = lambda * ( ( vv(i) > locminv ) ? 1/vv(i) : 1/locminv );
                    ssl("&",i)  = ivl(i) - BFoff(i);
                }

                if ( isFixedBias() && b )
                {
                    // B.w + b = c

                    c -= b;
                }

                // Initial setup of isact for first call

//FIXME TEMP                //if ( !notfirstcall )
                {
                    // Process is iterative in this case, so set startpoint zero and make sure isact is correctly set
                    // Also make the startpoint feasible?

                    g = 0.0;
                    dg = 0.0;

                    fixactiveset(Qii,g,Eps,vw,b,wdim,Nval,notfirstcall);
                }

                int k;
//errstream() << "phantomx 0: Activation set: " << isact << "\n";
//errstream() << "phantomx 0: dd: " << dd << "\n";
//errstream() << "phantomx 0: g: " << g << "\n";

                double xmaxitcnt = maxitcnt();
                double xmtrtime = maxtraintime();
                double xmtrtimeend = traintimeend();
                double vlr = lrb();
                double soltol = Opttolb();
                double *uservars[] = { &xmaxitcnt, &xmtrtime, &xmtrtimeend, &vlr, &soltol, nullptr };
                const char *varnames[] = { "itercount", "traintime", "traintimeend", "vlr", "soltol", nullptr };
                const char *vardescr[] = { "Maximum iteration count (0 for unlimited)", "Maximum training time (seconds, 0 for unlimited)", "Training stop time(absolute, sconds, -1 for na)", "Step scale (vlr)", "Solution tolerance", nullptr };

                bool isopt = false;
                time_used start_time = TIMECALL;
                time_used curr_time = start_time;
                size_t itcnt = 0;
                int timeout = 0;
                int bailout = 0;

                double prevscale = 1;
                int previblock = -1;

                while ( !killSwitch && !isopt && ( ( itcnt < (size_t) xmaxitcnt ) || !xmaxitcnt ) && !timeout && !bailout )
                {
                    isopt = false;

                    vwold = vw;
                    bold  = b;

                    // 1. Calculate step
                    // 2. Scale step based on inactive constraints
                    // 3. Take potentially scaled step
                    // 4. If step was scaled then activate relevant constraint and goto 1
                    // 5. Are active constraints all required?
                    // 6. If no then deactivate worst active constraint
                    // 7. Otherwise set isopt

                    // 1. Calculate step (Newton based)

//errstream() << "phantomxyzahfiii 0: g(370) " << g(370) << "\n";
                    if ( isLinearCost() )
                    {
                        dvw = c/ivl;
//errstream() << "phantomxyzaa 0: c = " << c << "\n";
//errstream() << "phantomxyzaa 0: ivl = " << ivl << "\n";

                        if ( isVarBias() )
                        {
                            if ( ( fabs(NN) < zzta ) && ( NN > 0 ) )
                            {
                                db = d/1e-6;
                            }

                            else if ( ( fabs(NN) < zzta ) && ( NN < 0 ) )
                            {
                                db = -d/1e-6;
                            }

                            else if ( ( fabs(NN) < zzta ) && ( d > 0 ) )
                            {
                                db = 1e6;
                            }

                            else if ( ( fabs(NN) < zzta ) && ( d < 0 ) )
                            {
                                db = -1e6;
                            }

                            else if ( fabs(NN) < zzta )
                            {
                                db = 0;
                            }

                            else
                            {
                                db = d/NN;
                            }
                        }
                    }

                    else
                    {
                        locBF.minverseOffset(dvw,c,1.0,ssl,B,BFoff,fbused);

                        if ( isVarBias() )
                        {
                            double s,t;
                            double tmpdiv;

                            locBF.minverseOffset(q,a,1.0,ssl,B,BFoff,fbused);

                            tmpdiv = twoProduct(s,q,a);
                            tmpdiv -= NN;

                            if ( fabs(tmpdiv) < zzta )
                            {
                                tmpdiv = ( tmpdiv <= 0 ) ? -zzta : zzta;
                            }

                            db = ((twoProduct(t,q,c)-d)/tmpdiv);
                            dvw.scaleAdd(-db,q);
                        }

                        else
                        {
                            db = bold;
                        }
                    }

                    dvw -= vwold;
                    db  -= bold;

//errstream() << "phantomxyzaa 0: dvw = " << dvw << "\n";
//errstream() << "phantomxyzaa 0: db = " << db << "\n";
                    // 2. Scale step based on inactive constraints

                    double scale = 1.0;
                    double xscale;

                    int iblock = -1;
                    int blocktype = 0;

                    dg = 0.0;

                    for ( i = 0 ; i < Nval ; ++i )
                    {
                        dg("&",i) = db;

                        for ( k = 0 ; k < wdim ; ++k )
                        {
                            dg("&",i) += Gp()(k+Nval,i)*dvw(k);
                        }

//errstream() << "phantomxyqwq -4 (" << i << "): " << g(i) << " + " << dg(i) << " = " << (g(i)+dg(i)) << " ??? " << loczr(i) << "\n";
                        if ( dd(i) && !isact(i) && dg(i) )
                        {
                            double effzn = (loczr(i)+Eps(i));
                            double effzp = (loczr(i)-Eps(i));
//errstream() << "phantomxy -4 (" << i << "): " << g(i) << " + " << dg(i) << " = " << (g(i)+dg(i)) << " ??? " << effzn << "," << effzp << "\n";

                            if ( ( ( +1 == dd(i) ) || ( +2 == dd(i) ) ) && ( g(i)+dg(i) < effzp-zzta ) && ( dg(i) < -zztna ) )
                            {
                                // Inactive constraint g_i >= z_i...
                                // ...becoming potentially active g_i+s.dg_i = z_i
                                //
                                // g_i + s.dg_i = z_i
                                // s = (z_i-g_i)/dg_i

                                xscale = (effzp-g(i))/dg(i);

                                if ( xscale < scale )
                                {
                                    scale = xscale;
                                    iblock = i;
                                    blocktype = +1;
//errstream() << "phantomxya -4 (" << i << "): " << g(i) << " + " << dg(i) << " = " << (g(i)+dg(i)) << " < " << effzp << "\n";
                                }
                            }

                            else if ( ( ( -1 == dd(i) ) || ( +2 == dd(i) ) ) && ( g(i)+dg(i) > effzn+zzta ) && ( dg(i) > zztma ) )
                            {
                                // Inactive constraint g_i <= z_i...
                                // ...becoming potentially active g_i+s.dg_i = z_i
                                //
                                // g_i + s.dg_i = z_i
                                // s = (z_i-g_i)/dg_i

                                xscale = (effzn-g(i))/dg(i);

                                if ( xscale < scale )
                                {
                                    scale = xscale;
                                    iblock = i;
                                    blocktype = -1;
//errstream() << "phantomxya -2 (" << i << "): " << g(i) << " + " << dg(i) << " = " << (g(i)+dg(i)) << " < " << effzn << "\n";
                                }
                            }
                        }

                        if ( dd(i) && isact(i) )
                        {
                            int ddd = ( dd(i) == 2 ) ? isact(i) : dd(i);
                            double effz = ( ddd == -1 ) ? (loczr(i)+Eps(i)) : (loczr(i)-Eps(i));
//errstream() << "phantomxy 13 (" << i << "): " << ddd << ", " << g(i) << " ??? " << effz << "\n";

                            if ( ( +1 == ddd ) && ( g(i)+dg(i) > effz-zztna ) && ( dg(i) > zztna ) )
                            {
                                // Active constraint g_i >= z_i...
                                // ...becoming potentially inactive
                                //
                                // g_i + s.dg_i = z_i
                                // s = (z_i-g_i)/dg_i

                                xscale = (effz-g(i))/dg(i);

                                if ( xscale < scale )
                                {
                                    scale = xscale;
                                    iblock = i;
                                    blocktype = +1;
//errstream() << "phantomxy -4 (" << i << "): " << g(i) << " > " << effz << "\n";
                                }
                            }

                            else if ( ( -1 == ddd ) && ( g(i)+dg(i) < effz+zztma ) && ( dg(i) < -zztma ) )
                            {
                                // Active constraint g_i <= z_i...
                                // ...becoming potentially inactive
                                //
                                // g_i + s.dg_i = z_i
                                // s = (z_i-g_i)/dg_i

                                xscale = (effz-g(i))/dg(i);

                                if ( xscale < scale )
                                {
                                    scale = xscale;
                                    iblock = i;
                                    blocktype = -1;
//errstream() << "phantomxy -2 (" << i << "): " << g(i) << " < " << effz << "\n";
                                }
                            }
                        }
                    }

                    if ( scale < 0 )
                    {
//errstream() << "phantomxq negative scale of " << scale << " corrected\n";
                        scale = 0;
                    }

                    // 3. Take potentially scaled step

//errstream() << "phantomxyzah 0: scale of " << scale << "\n";
                    vw.scaleAdd(scale,dvw);
//errstream() << "phantomxyzahf 0a: g(370) " << g(360) << "\n";
//errstream() << "phantomxyzahf 0b: dg(370) " << dg(370) << "\n";
                    g.scaleAdd(scale,dg);
//errstream() << "phantomxyzahf 0c: g(370) " << g(370) << "\n";
//errstream() << "phantomxyzahf 0d: d.dg(370) " << (blocktype*dg(370)) << "\n";
//errstream() << "phantomxyzahf 0e: d.dgnext(370) " << (blocktype*calcCscalequick(370)*getGpRowRffPartNorm2(370,ivl)) << "\n";

                    if ( isVarBias() )
                    {
                        b = bold+(scale*b);
                    }

                    // 4. If step was scaled then activate relevant constraint and goto 1

                    if ( blocktype && !isact(iblock) )
                    {
                        // activate constraint

                        setIsAct(blocktype,iblock,Qii);
//errstream() << "phantomx 12: have activated " << iblock << " type " << blocktype << "\n";

if ( ( iblock == previblock ) && ( scale <= zzta ) && ( prevscale <= zzta ) )
{
    isopt = true;
}

                        previblock = iblock;
                        prevscale = scale;
                    }

                    else if ( blocktype && ( isQuadraticCost() || ( scale > zztna ) || ( blocktype*dg(iblock) > (blocktype*calcCscalequick(iblock)*getGpRowRffPartNorm2(iblock,ivl)) ) ) )
                    {
                        // deactivate constraint on g_iset

                        setIsAct(0,iblock,Qii);
//errstream() << "phantomx 13: have DEactivated " << iblock << "\n";

                        previblock = iblock;
                        prevscale = scale;
                    }

                    else
                    {
                        // 5. Are active constraints all required?

/*
errstream() << "phantomxyzau 0: check active constraints\n";
                        double magover = 0;
                        double xmagover;

                        int iset = -1;
                        int itype = 0;

                        for ( i = 0 ; i < Nval ; ++i )
                        {
//errstream() << "phantomxy 12 (" << i << "): " << dd(i) << ", " << isact(i) << ", " << g(i) << " ??? " << loczr(i) << "\n";
                            if ( dd(i) && isact(i) )
                            {
                                int ddd = ( dd(i) == 2 ) ? isact(i) : dd(i);
                                double effz = ( ddd == -1 ) ? (loczr(i)+Eps(i)) : (loczr(i)-Eps(i));
//errstream() << "phantomxy 13 (" << i << "): " << ddd << ", " << g(i) << " ??? " << effz << "\n";

                                if ( ( +1 == ddd ) && ( g(i) > effz-zztna ) )
                                {
                                    // Inactive constraint g_i >= z_i...
                                    // ...becoming potentially active

                                    xmagover = fabs(g(i)-effz);

                                    if ( xmagover > magover )
                                    {
                                        magover = xmagover;
                                        iset = i;
                                        itype = +1;
errstream() << "phantomxy -4 (" << i << "): " << g(i) << " > " << effz << "\n";
                                    }
                                }

                                else if ( ( -1 == ddd ) && ( g(i) < effz+zztma ) )
                                {
                                    // Inactive constraint g_i <= z_i...
                                    // ...becoming potentially active

                                    xmagover = fabs(g(i)-effz);

                                    if ( xmagover > magover )
                                    {
                                        magover = xmagover;
                                        iset = i;
                                        itype = -1;
errstream() << "phantomxy -2 (" << i << "): " << g(i) << " < " << effz << "\n";
                                    }
                                }
                            }
                        }

                        // 6. If no then deactivate worst active constraint

                        if ( itype )
                        {
                            // deactivate constraint on g_iset

                            setIsAct(0,iset,Qii);
errstream() << "phantomx 13: have DEactivated " << iset << "\n";
                        }

                        else
*/
                        {
                            // 7. Otherwise set isopt

//errstream() << "phantomxyzau 0: done, apparently\n";
                            isopt = true;
                        }
                    }

                    if ( !isopt )
                    {
                        // Feedback

                        if ( !(++itcnt%FEEDBACK_CYCLE) )
                        {
                            if ( (itcnt/FEEDBACK_CYCLE)%4 == 0 )
                            {
outstream("+\b");
//                                errstream("+\b");
                            }

                            else if ( (itcnt/FEEDBACK_CYCLE)%4 == 1 )
                            {
outstream("x\b");
                                //errstream("x\b");
                            }

                            else if ( (itcnt/FEEDBACK_CYCLE)%4 == 2 )
                            {
outstream("-\b");
                                //errstream("-\b");
                            }

                            else if ( (itcnt/FEEDBACK_CYCLE)%4 == 3 )
                            {
outstream("x\b");
                                //errstream("*\b");
                            }
                        }

                        if ( !(itcnt%MAJOR_FEEDBACK_CYCLE) && isMainThread() )
                        {
                            char dummyb[1024];

outstream() << "\n\n" << itcnt << "\n\n";
                            sprintf(dummyb,"=%d=",(int) itcnt);

                            //nullPrint(errstream(),dummyb);
                        }

                        if ( xmtrtime > 1 )
                        {
                            curr_time = TIMECALL;

                            if ( TIMEDIFFSEC(curr_time,start_time) > xmtrtime )
                            {
                                timeout = 1;

                                errstream() << "\nTimeout in inner loop training.\n";
                            }
                        }

                        if ( xmtrtimeend > 0 )
                        {
                            curr_time = TIMECALL;

                            if ( TIMEABSSEC(curr_time) >= xmtrtimeend )
                            {
                                timeout = 1;

                                errstream() << "\nAbsolute timeout in inner loop training.\n";
                            }
                        }

                        if ( !timeout )
                        {
                            timeout = kbquitdet("SVM_Scalar_rff inner training loop",uservars,varnames,vardescr);
                        }
                    }
                }

                res = isopt ? 0 : ( bailout ? bailout : -1 );

                if ( isFixedBias() )
                {
                    // Q = 1/2 vw'.B.vw + lambda/2 vw'.(vv.*vw) - vw'.c
                    // e = B.vw + lambda vv.*vw - c = 0
                    // Q = 1/2 vw'.e - 1/2 vw'.c = -1/2 vw'.c

                    twoProduct(tres,vw,c) /= -2;
                }

                else
                {
                    // Q = -1/2 [ vw ]'.[ c ]
                    //          [ b  ]  [ d ]

                    twoProduct(tres,vw,c) /= -2;
                    tres -= (b*d/2);
                }

                if ( isFixedBias() && b )
                {
                    // B.w + b = c

                    c += b;
                }

                if ( 2 == method )
                {
                    vwstartpoint = vw;
                    bstartpoint  = b;
                }
            }

            else if ( 2 == method )
            {
                // Remainder is as per method == 1, with modified starting point

                vw = vwstartpoint;
                b  = bstartpoint;

                vwgradset optinfo(lambda,vv,B,a,c,Eps,Csc,isact,loczr,Atscratch,Btscratch,gscratch.resize(Nval),d,NN,b,wdim,method,costtype,locpegk,*this);
                ininadamscratchpad.resize(isVarBias() ? wdim+1 : wdim);

                double opttolis = ( !( ( method == 3 ) || ( method == 5 ) || ( method == 7 ) ) || ( ( method == 7 ) && ( locpegk == -1 ) ) ) ? Opttolb() : -1;

                stopCond sc;

                sc.maxitcnt   = maxitcnt();
                sc.maxruntime = maxtraintime();
                sc.runtimeend = traintimeend();
                sc.tol        = opttolis;

                res = ADAMopt(tres,( isVarBias() ? vwaug : vw ),calcvwObj,"Optimisation in SVM random fourier features (vw loop)",killSwitch,lrb(),(void *) &optinfo,ininadamscratchpad,USE_ADAM,sc,ADAM_BETA1,ADAM_BETA2,ADAM_EPS,2);
            }

            vwaug("&",wdim) = b;
        }

        else if ( ( 1 == method ) || ( 3 == method ) || ( 4 == method ) || ( 5 == method ) )
        {
            goto lazygoto;
        }
    }

    else
    {
//errstream() << "phantomxyz 0 type 3\n";
        // Pegasos

        double miniv = 1;
        double ivval;

        for ( i = 0 ; i < vdim ; ++i )
        {
            ivval = ( vv(i) > locminv ) ? 1/vv(i) : 1/locminv;

            if ( !i || ( ivval < miniv ) )
            {
                miniv = ivval;
            }
        }

        // Call ADAM to do his thing (which is gradient descent here)!

        // 1/2 ||w||^2 + C/N sum_i ell_i
        // 1/2 ||w||^2 + C() sum_i ell_i
        // 1/2 ||w||^2 + (N*C())/N sum_i ell_i
        // 1/(N*C()) 1/2 ||w||^2 + 1/N sum_i ell_i
        // lambda/2 ||w||^2 + 1/N sum_i ell_i
        //
        // so "lambda" = 1/Nnz.C for the purposes of pegasos
        // then scale by min 1/iv to allow for variable regularisation here!
        //
        //see https://ftp.esat.kuleuven.be/pub/SISTA/vjumutc/reports/weighted_pegasos_jumutc_suykens_PREMI2013.pdf (Jun1)

        int useadam = 0;
        double peglambda = lambda;
        double schedconst = SCHEDCONST;
        double lris = lrb();
        double normmax = 0;
        double opttolis = Opttolb();
        int kval = locpegk;

        if ( ( 1 == method ) || ( 2 == method ) )
        {
            useadam = USE_ADAM;
            peglambda = lambda;
            schedconst = SCHEDCONST;
            lris = lrb();
            normmax = 0;
            opttolis = Opttolb();
            kval = -1;
        }

        else if ( 3 == method )
        {
            useadam = USE_ADAM;
            peglambda = lambda;
            schedconst = SCHEDCONST;
            lris = lrb();
            normmax = 0;
            opttolis = Opttolb();
            kval = locpegk;
        }

        else if ( 4 == method )
        {
            useadam = 0;
            peglambda = lambda;
            schedconst = SCHEDCONST;
            lris = lrb();
            normmax = 0;
            opttolis = Opttolb();
            kval = -1;
        }

        else if ( 5 == method )
        {
            useadam = 0;
            peglambda = lambda;
            schedconst = SCHEDCONST;
            lris = lrb();
            normmax = 0;
            opttolis = Opttolb();
            kval = locpegk;
        }

        else if ( 7 == method )
        {
            //int Nnz = Nval-SVM_Scalar::NNC(0);

            useadam = 0;
            //peglambda  = miniv*lambda/Nnz; // lambda for the purposes of pegasos optimisation
            ////peglambda  = lambda*miniv/Nnz; // lambda for the purposes of pegasos optimisation
            peglambda  = lambda*miniv; // lambda for the purposes of pegasos optimisation
            schedconst = 1.0; // pegasos lr is 1/(lambda.t)
            lris = 1/peglambda; // pegasos lr is 1/(lambda.t)
            normmax = isVarBias() ? -1/sqrt(peglambda) : 1/sqrt(peglambda); // pegasos scaling regime
//            opttolis   = ( !( ( method == 3 ) || ( method == 5 ) || ( method == 7 ) ) || ( ( method == 7 ) && ( locpegk == -1 ) ) ) ? Opttolb() : -1;
            opttolis = Opttolb();
            kval = locpegk;
        }

        // Optimisation info

        //vwgradset optinfo(peglambda,vv,B,a,c,Eps,Csc,isact,loczr,Atscratch,Btscratch,gscratch.resize(Nval),d,NN,b,wdim,method,costtype,locpegk,*this);
        vwgradset optinfo(   lambda,vv,B,a,c,Eps,Csc,isact,loczr,Atscratch,Btscratch,gscratch.resize(Nval),d,NN,b,wdim,method,costtype,kval,*this);
        ininadamscratchpad.resize(isVarBias() ? wdim+1 : wdim);

        // Optimisation

        Vector<double> &xxx = ( isVarBias() ? vwaug : vw );

        stopCond sc;

        sc.maxitcnt   = maxitcnt();
        sc.maxruntime = maxtraintime();
        sc.runtimeend = traintimeend();
        sc.tol        = opttolis;

        res = ADAMopt(tres,xxx,calcvwObj,"Optimisation in SVM random fourier features (vw loop) - PEGASOS variant",killSwitch,lris,(void *) &optinfo,ininadamscratchpad,useadam,sc,ADAM_BETA1,ADAM_BETA2,ADAM_EPS,2,schedconst,normmax,locminv);

        b = vwaug(wdim);

/*
int j,ii;
                for ( j = Nval-1 ; j >= 0 ; --j )
                {
                    if ( dd(j) )
                    {
                        gscratch("&",j) = b;

                        for ( ii = 0 ; ii < wdim ; ++ii )
                        {
                            gscratch("&",j) += vw(ii)*Gp()(ii+Nval,j);
                        }

                        if ( ( +1 == dd(j) ) && ( gscratch(j) < loczr(j)+Eps(j) ) )
                        {
errstream() << "phantomx 9 +1: " << gscratch(j) << "-(" << loczr(j) << "+" << Eps(j) << " = " << gscratch(j)-(loczr(j)+Eps(j)) << "\n";
                        }

                        else if ( ( -1 == dd(j) ) && ( gscratch(j) > loczr(j)-Eps(j) ) )
                        {
errstream() << "phantomx 9 -1: " << gscratch(j) << "-(" << loczr(j) << "-" << Eps(j) << " = " << gscratch(j)-(loczr(j)-Eps(j)) << "\n";
                        }

                        else if ( ( 2 == dd(j) ) && ( gscratch(j) > loczr(j)+Eps(j) ) )
                        {
errstream() << "phantomx 9 2+1: " << gscratch(j) << "-(" << loczr(j) << "+" << Eps(j) << " = " << gscratch(j)-(loczr(j)+Eps(j)) << "\n";
                        }

                        else if ( ( 2 == dd(j) ) && ( gscratch(j) < loczr(j)-Eps(j) ) )
                        {
errstream() << "phantomx 9 2-1: " << gscratch(j) << "-(" << loczr(j) << "-" << Eps(j) << " = " << gscratch(j)-(loczr(j)-Eps(j)) << "\n";
                        }

                        else
                        {
errstream() << "phantomx other: " << gscratch(j) << "\n";
                        }
                    }
                }
*/
    }

//    errstream() << "?";

    for ( i = 0 ; i < wdim ; ++i )
    {
        if ( testisvnan(vw(i)) || testisinf(vw(i)) )
        {
            errstream() << "!wvinfnan!!" << i;
            vw("&",i) = 0.0;
        }

        if ( testisvnan(vv(i)) || testisinf(vv(i)) )
        {
            errstream() << "!vinfnan!!" << i;
            vv("&",i) = ( tunev() == -1 ) ? -1.0 : 1.0;
        }

        if ( ( vv(i) > 0 ) && ( tunev() == -1 ) )
        {
            errstream() << "!vover!!" << i;
            vv("&",i) = -1.0;
        }

        if ( ( vv(i) < 0 ) && ( tunev() == +1 ) )
        {
            errstream() << "!vunder!!" << i;
            vv("&",i) = 1.0;
        }
    }

    if ( testisvnan(b) || testisinf(b) )
    {
        errstream() << "!binfnan!!";
        b = 0.0;
    }

//    errstream() << "\b";
    return tres; // res =
}






class vgradset
{
public:

    vgradset(int xmethod, double xsf, double xlambda, double xLambda, double xlocminv, double xlocF, double xlocG,
             const Matrix<double> &xH, Matrix<double> &xHchol, int &xHcholcalced,
             Vector<double> &xvw, Vector<double> &xvwaug, Vector<double> &xvv, Vector<double> &xivgradbase, Vector<double> &xivmodbase,
             double &xb, int xinnerAdam, int &xnotfirstcall, int &xfbused, int &xitcnt,
             Vector<double> &xyinterm, Vector<double> &xivmod,
             int xvdim, int xwdim, int xpegk, int xcosttype, int xisVarBias,
             const Vector<double> &xEps, const Vector<double> &xCsc, const Vector<int> &xisact, const Vector<double> &xloczr,
             Vector<int> &xAt, Vector<int> &xBt, Vector<double> &xg, Vector<double> &xxgrad, Vector<double> &xivgradgradbase,
             SVM_Scalar_rff &xcaller) :
        method(xmethod), sf(xsf), lambda(xlambda), Lambda(xLambda), locminv(xlocminv), locF(xlocF), locG(xlocG),
        H(xH), Hchol(xHchol), Hcholcalced(xHcholcalced),
        vw(xvw), vwaug(xvwaug), vv(xvv), ivgradbase(xivgradbase), ivmodbase(xivmodbase),
        b(xb), innerAdam(xinnerAdam), notfirstcall(xnotfirstcall), fbused(xfbused), itcnt(xitcnt),
        yinterm(xyinterm), ivmod(xivmod),
        vdim(xvdim), wdim(xwdim), pegk(xpegk), costtype(xcosttype), isVarBias(xisVarBias),
        Eps(xEps), Csc(xCsc), isact(xisact), loczr(xloczr),
        At(xAt), Bt(xBt), g(xg), xgrad(xxgrad), ivgradgradbase(xivgradgradbase),
        caller(xcaller) { ; }

    int method; // 0 means work in iv, 1 means work in v

    double sf;
    double lambda;
    double Lambda;
    double locminv;
    double locF;
    double locG;

    const Matrix<double> &H;
    Matrix<double> &Hchol;

    int &Hcholcalced; // should be set 0 initially

    Vector<double> &vw;
    Vector<double> &vwaug;
    Vector<double> &vv;

    Vector<double> &ivgradbase;
    Vector<double> &ivmodbase;

    double &b;

    int innerAdam;
    int &notfirstcall;
    int &fbused;
    int &itcnt;

    Vector<double> &yinterm; // cache
    Vector<double> &ivmod;   // cache

    int vdim;
    int wdim;
    int pegk;
    int costtype;
    int isVarBias;

    const Vector<double> &Eps;
    const Vector<double> &Csc;
    const Vector<int> &isact;
    const Vector<double> &loczr;

    Vector<int> &At;
    Vector<int> &Bt;
    Vector<double> &g;
    Vector<double> &xgrad;
    Vector<double> &ivgradgradbase;

    SVM_Scalar_rff &caller;

    int calcObj(double &res, const Vector<double> &iv, Vector<double> &ivgrad, Vector<double> &ivgradgrad, svmvolatile int &killSwitch, int *nostop)
    {
        *nostop = 0;

        int tres = 0;

        // Methods: 0 = iv, two    level, ind chol
        //          2 = iv, two    level, svm chol
        //          4 = iv, single level, ind chol
        //          6 = iv, single level, svm chol
        //          1 = vv, two    level, ind chol
        //          3 = vv, two    level, svm chol
        //          5 = vv, single level, ind chol
        //          7 = vv, single level, svm chol

        if ( ( method == 0 ) || ( method == 2 ) || ( method == 4 ) || ( method == 6 ) )
        {
            // Optimising in terms of 1/v

            int i;

            // First we must update vv

            //int wdim = vw.size();
            //int vdim = vv.size();

            double vvsum = 0;

            for ( i = 0 ; i < vdim ; ++i )
            {
                vv("&",i) = 1/iv(i);
                vvsum += vv(i);
            }

            for ( i = vdim ; i < wdim ; ++i )
            {
                vv("&",i) = vv(i%vdim);
            }

            // Then we need to call the inner loop to calculate vw

            if ( method < 4 )
            {
                res = caller.inintrain(tres,killSwitch,vw,vwaug,vv,b,lambda,innerAdam,notfirstcall,fbused);
            }

            else
            {
                tres = calcinnergrad(res,iv,ivgrad,ivgradgrad,killSwitch,nostop);
            }

            // OLD: e = (lambda/(2*Lambda)) H.((|wv|.^2).*(iv.*iv)) - inv(iv) + 1
            // e = (lambda/(2*Lambda)) H.( (|wv|.^2).*(iv.^2) - (2*Lambda*F/lambda) ( (1.1').inv(iv) - G 1 )) - inv(iv) + minv

            for ( i = 0 ; i < vdim ; ++i )
            {
                if ( vdim == wdim )
                {
                    ivgrad("&",i) = (iv(i)*iv(i)*vw(i)*vw(i)) - ((2*Lambda*locF/lambda)*(vvsum-locG));
                }

                else
                {
                    ivgrad("&",i) = (iv(i)*iv(i)*((vw(i)*vw(i))+(vw(i+vdim)*vw(i+vdim)))) - ((2*Lambda*locF/lambda)*(vvsum-locG));
                }
            }

            ivgrad *= H;

            for ( i = 0 ; i < vdim ; ++i )
            {
                ivgrad("&",i) = (ivgrad(i)*sf*lambda/(2*Lambda))-vv(i)+locminv;
                ivgradgrad("&",i) = 1; // because I'm lazy
            }

            double xres = (lambda/2)*absinf(ivgrad);

            if ( isMainThread() && !(itcnt%FEEDBACK_CYCLE) )
            {
                char errbuffer[512];

                if ( fabs(xres) < 1e6 )
                {
                    sprintf(errbuffer,"      %d:    %lf  +  %lf  =  %lf   (%lf, %d)                                          ",itcnt,res,xres,res+xres,absinf(ivgradgrad),fbused);
                }

                else
                {
                    sprintf(errbuffer,"      %d:    %lf  +  BIGNUM  =  BIGNUM   (BIGNUM, %d)                                          ",itcnt,res,fbused);
                }

                nullPrint(errstream(),errbuffer);
            }

            ++itcnt;

            res += xres;
        }

        else
        {
            //else if ( method == 1 ) || ( method == 3 )
            // Optimising in terms of v

            int i;

            // First we must update vv

            //int wdim = vw.size();
            //int vdim = iv.size();

            double vvsum = 0;

            for ( i = 0 ; i < vdim ; ++i )
            {
                vv("&",i) = iv(i); // in this method, iv represents v
                vvsum += vv(i);
            }

            for ( i = vdim ; i < wdim ; ++i )
            {
                vv("&",i) = vv(i%vdim);
            }
//Vector<double> vvold(vv);

            // Then we need to call the inner loop to calculate vw

            if ( method < 4 )
            {
                res = caller.inintrain(tres,killSwitch,vw,vwaug,vv,b,lambda,innerAdam,notfirstcall,fbused);
//errstream() << "phantomxyzooo 0: vw = " << vw << "\n";
            }

            else
            {
                tres = calcinnergrad(res,iv,ivgrad,ivgradgrad,killSwitch,nostop);
            }

            // g = -(|wv|.^2).*(inv(v).^2) + 2Lambda/lambda inv(H).(v-1) + F (1.1').v - FG 1

            if ( ( 1 == method ) || ( 5 == method ) )
            {
                if ( !Hcholcalced )
                {
                    yinterm.resize(vdim);
                    //ivmod.resize(vdim); - need to do this elsewhere so we can setup ivmodbase correctly!
                    Hchol.resize(vdim,vdim);
                }
            }

            else
            {
                if ( !Hcholcalced )
                {
                    yinterm.resize(2);
                    //ivmod.resize(vdim); - need to do this elsewhere so we can setup ivmodbase correctly!
                }
            }

            for ( i = 0 ; i < vdim ; ++i )
            {
                ivmod("&",i%vdim) = iv(i)-locminv;
            }

//errstream() << "phantomxyzooo 1: vv-locminv = " << ivmod << "\n";
            if ( 1 == method )
            {
                H.naiveCholInve(ivgrad,ivmod,yinterm,Hchol,0,Hcholcalced);
            }

            else
            {
                retVector<double> tmpva;
                retVector<double> tmpvb;

                caller.fact_minverse(ivgradbase,yinterm("&",0,1,0,tmpva),ivmodbase,yinterm(1,1,1,tmpvb));
            }

            ivgrad.scale(2*Lambda/lambda);

            // Sparse approximate inverse for second order

            for ( i = 0 ; i < vdim ; ++i )
            {
                ivgradgrad("&",i) = ivgradgradbase(i);
            }

//errstream() << "phantomxyzooo 2: ivgrad = " << ivgrad << "\n";

            for ( i = 0 ; i < vdim ; ++i )
            {
                if ( vdim == wdim )
                {
                    ivgrad("&",i) += ((locF*vvsum)-(locF*locG)-(vw(i)*vw(i)/(vv(i)*vv(i))));
                    ivgradgrad("&",i) += ((2*vw(i)*vw(i)/(vv(i)*vv(i)*vv(i))));
                }

                else
                {
                    ivgrad("&",i) += ((locF*vvsum)-(locF*locG)-(((vw(i)*vw(i))+(vw(i+vdim)+vw(i+vdim)))/(vv(i)*vv(i))));
                    ivgradgrad("&",i) += ((2*((vw(i)*vw(i))+(vw(i+vdim)+vw(i+vdim)))/(vv(i)*vv(i)*vv(i))));
                }

                //ivgradgrad("&",i) = ( ivgradgrad(i) < 1 ) ? 1.0 : ivgradgrad(i);
            }
//errstream() << "phantomxyzooo 3: ivgrad = " << ivgrad << "\n";

            if ( method >= 4 )
            {
                for ( i = 0 ; i < vdim ; ++i )
                {
                    ivgrad("&",i) *= lambda/2;
                    ivgradgrad("&",i) *= lambda/2;
                }
            }

            double xres = absinf(ivgrad);

            if ( isMainThread() && !(itcnt%FEEDBACK_CYCLE) )
            {
                char errbuffer[512];

                if ( fabs(xres) < 1e6 )
                {
                    sprintf(errbuffer,"      %d:    %lf  +  %lf  =  %lf   (%lf, %lf, %d)                                          ",itcnt,res,xres,res+xres,absinf(ivgrad),absinf(ivgradgrad),fbused);
                }

                else
                {
                    sprintf(errbuffer,"      %d:    %lf  +  BIGNUM  =  BIGNUM   (BIGNUM, %d)                                          ",itcnt,res,fbused);
//errstream() << "phantomxyz ivgrad = " << ivgrad << "\n";
//errstream() << "phantomxyz vvold = " << vv << "\n";
//errstream() << "phantomxyz vv = " << iv << "\n";
//errstream() << "phantomxyz vw = " << vw << "\n";
//exit(1);
//errstream() << "                                           phantomxyz ivgrad = " << absinf(ivgrad) << "\n";
                }

                nullPrint(errstream(),errbuffer);
            }

            ++itcnt;

            res += xres;

            // Trickery to fix off-diagonals approximately
            //
            // (in essence treat hessian as 4 diagonal (approximated) blocks,
            // and do part of the Newton step here, leaving the second order
            // terms for later scaling).

//errstream() << "phantomxyz ivgradpre = " << ivgrad << "\n";
//errstream() << "phantomxyz absinf(ivgradpre) = " << absinf(ivgrad) << "\n";
//errstream() << "phantomxyz ivgradgrad = " << ivgradgrad << "\n";
//errstream() << "phantomxyz absinf(ivgradgrad) = " << absinf(ivgradgrad) << "\n";
ivgradgrad = 1.0;
/*
            if ( ( method >= 4 ) && ( vdim == wdim ) )
            {
                double tmpvgrad;
                double tmpwgrad;
                double tmpvvgradgrad;
                double tmpwwgradgrad;

                for ( i = 0 ; i < wdim ; ++i )
                {
                    tmpvgrad = (ivgradgrad(i+vdim)*ivgrad(i)) - (xgrad(i)*ivgrad(i+vdim));
                    tmpwgrad = (ivgradgrad(i)*ivgrad(i+vdim)) - (xgrad(i)*ivgrad(i));

                    tmpvvgradgrad = (ivgradgrad(i)*ivgradgrad(i+vdim)) - (xgrad(i)*xgrad(i));
                    tmpwwgradgrad = (ivgradgrad(i+vdim)*ivgradgrad(i)) - (xgrad(i)*xgrad(i));

//errstream() << "phantomxyz 0: tmpwwgragrad " << tmpwwgradgrad << "\n";

                    tmpvvgradgrad = ( tmpvvgradgrad < 1e-5 ) ? 1e-5 : tmpvvgradgrad;
                    tmpwwgradgrad = ( tmpwwgradgrad < 1e-5 ) ? 1e-5 : tmpwwgradgrad;

                    ivgrad("&",i)      = tmpvgrad;
                    ivgrad("&",i+vdim) = tmpwgrad;

                    ivgradgrad("&",i)      = tmpvvgradgrad;
                    ivgradgrad("&",i+vdim) = tmpwwgradgrad;

                    //ivgrad("&",i)      = tmpvgrad/tmpvvgradgrad;
                    //ivgrad("&",i+vdim) = tmpwgrad/tmpwwgradgrad;

                    //ivgradgrad("&",i)      = 1;
                    //ivgradgrad("&",i+vdim) = 1;
                }
            }
*/

//errstream() << "phantomxyz ivgrad = " << ivgrad << "\n";
//errstream() << "phantomxyz absinf(ivgrad) = " << absinf(ivgrad) << "\n";
//errstream() << "phantomxyz nostop = " << *nostop << "\n";
//errstream() << "phantomxyz vv = " << iv << "\n";
        }

//errstream() << "phantomxyz ivgrad = " << ivgrad << "\n";
//errstream() << "phantomxyz vvold = " << vv << "\n";
//errstream() << "phantomxyz vv = " << iv << "\n";
//errstream() << "phantomxyz method = " << method << "\n";
        return tres;
    }

    int calcinnergrad(double &res, const Vector<double> &iv, Vector<double> &ivgrad, Vector<double> &ivgradgrad, svmvolatile int &, int *nostop)
    {
            int i,j;
            int tres = 0;

            //int costtype = caller::costtype;
            //bool isVarBias = ( wdim != vwaug.size() ); //caller.isVarBias();
            //int numpm = (caller.NNC(-1))+(caller.NNC(1));
            int N = caller.N();
            //int Nnz = N-caller.NNC(0);

            const Vector<int> &dd = caller.SVM_Scalar::d(); // don't want to be refered back to svm_binary_rff, as this may be wrong!
            //const Vector<double> &yRR = caller.yRR();
            const Matrix<double> &Gp = caller.Gp();

            retVector<double> tmpvaa;
            retVector<double> tmpvb;
            retVector<double> tmpvc;

            double dummygrad = 0.0;
            double dummygradgrad = 0.0;

            vw = iv(vdim,1,vdim+wdim-1,tmpvaa);
            b = isVarBias ? iv(vdim+wdim) : 0;

//errstream() << "phantomxyz b = " << b << "\n";
            Vector<double> &vwgrad = ivgrad("&",vdim,1,vdim+wdim-1,tmpvb);
            double &bgrad = isVarBias ? ivgrad("&",vdim+wdim) : dummygrad;

            Vector<double> &vwgradgrad = ivgradgrad("&",vdim,1,vdim+wdim-1,tmpvc);
            double &bgradgrad = isVarBias ? ivgradgrad("&",vdim+wdim) : dummygradgrad;

            int kk = ( pegk == -1 ) ? N : pegk;
            bool dofull = itcnt%PEGASOS_FULLGRAD_CYCLE;
            int loc_kval = dofull ? kk : N; // every PEGASOS_FULLGRAD_CYCLE iterations we work out the complete gradient

            retVector<int> tmpva;

            int ii,Nk = 0;

            // Select random working set
            {
                Bt.resize(N) = cntintvec(N,tmpva);
                At.resize(0);

                for ( i = Bt.size()-1 ; i >= 0 ; --i )
                {
                    if ( !dd(Bt(i)) )
                    {
                        Bt.remove(i);
                    }
                }

                Nk = Bt.size();

                // Bt is now the list of possible vectors.  Next we go through,
                // randomly remove elements from Bt, and add them to At if they
                // are non-optimal.

                bool gotone = false;

                while ( ( At.size() < loc_kval ) && ( Bt.size() ) )
                {
                    //i = svm_rand()%(Bt.size());
                    i = rand()%(Bt.size());
                    j = Bt(i);

                    Bt.remove(i);

                    gotone = true;

                    if ( dd(j) )
                    {
                        g("&",j) = b;

                        for ( ii = 0 ; ii < wdim ; ++ii )
                        {
                            g("&",j) += vw(ii)*Gp(ii+N,j);
                        }

                        if ( ( +1 == dd(j) ) && ( g(j) < loczr(j)-Eps(j) ) )
                        {
                            g("&",j) -= (loczr(j)-Eps(j));
                        }

                        else if ( ( 2 == dd(j) ) && ( g(j) < loczr(j)-Eps(j) ) )
                        {
                            g("&",j) -= (loczr(j)-Eps(j));
                        }

                        else if ( ( -1 == dd(j) ) && ( g(j) > loczr(j)+Eps(j) ) )
                        {
                            g("&",j) -= (loczr(j)+Eps(j));
                        }

                        else if ( ( 2 == dd(j) ) && ( g(j) > loczr(j)+Eps(j) ) )
                        {
                            g("&",j) -= (loczr(j)+Eps(j));
                        }

                        else
                        {
                            gotone = false;
                        }
                    }

                    if ( gotone )
                    {
                        At.add(At.size());
                        At("&",At.size()-1) = j;
                    }
                }

                loc_kval = At.size();

                if ( Bt.size() )
                {
                    *nostop = 1;
                }
            }

//errstream() << "phantomxyz (" << costtype << "," << Nk << ") absinf(g) = " << absinf(g) << "\n";
//errstream() << "phantomxyz g = " << g << "\n";
//errstream() << "phantomxyz dd = " << dd << "\n";

            // Do gradient over At

            vwgrad = 0.0;
            bgrad  = 0.0;

            vwgradgrad = 0.0;
            bgradgrad  = 0.0;

            res = 0.0;

            double fxj,fxjgrad,fxjgradgrad;

            for ( i = 0 ; i < At.size() ; ++i )
            {
                j = At(i);
//errstream() << "phantomxyzab j = " << j << "\n";

                // work out f(x_j)

                fxj = g(j);

                if ( costtype == 0 )
                {
                    // ell(f,y) = |f|
                    // ell'(f,y) = sgn(f)
                    // ell''(f,y) "=" 0

//errstream() << "phantomxyzaaaaa g(" << j << ") = " << g(j) << " and dd(" << j << ") = " << dd(j) << "\t sign = " << g(j)*dd(j) << "\n";
                    res += fabs(fxj);
                    fxjgrad = ( fxj > 0 ) ? +1 : -1;
                    fxjgradgrad = 0;
                }

                else
                {
                    //if ( costtype == 1 )

                    // ell(f,y) = 1/2 (f-y)^2
                    // ell'(f,y) = (f-y)
                    // ell''(f,y) = 1

                    res += fxj*fxj/2;
                    fxjgrad = fxj;
                    fxjgradgrad = 1.0;
                }

                // dell/dwj = ell'.df/dwj
                //          = ell'.xi_j
                // d^2 ell/dw_j^2 = ell''.xi_j.df/dw
                //                = ell''.xi_j^2

                fxjgrad *= Csc(j);

                for ( ii = 0 ; ii < wdim ; ++ii )
                {
                    vwgrad("&",ii) += fxjgrad*Gp(ii+N,j); // /Nk;
                    vwgradgrad("&",ii) += fxjgradgrad*Gp(ii+N,j)*Gp(ii+N,j); // /Nk;
                }

                if ( isVarBias )
                {
                    bgrad += fxjgrad; // /Nk;
                    bgradgrad += fxjgradgrad; // /Nk;
                }
            }

            if ( !dofull && ( pegk > 0 ) )
            {
                for ( ii = 0 ; ii < wdim ; ++ii )
                {
                    vwgrad("&",ii) *= Nk/pegk;
                    vwgradgrad("&",ii) *= Nk/pegk;
                }

                if ( isVarBias )
                {
                    bgrad *= Nk/pegk;
                    bgradgrad += Nk/pegk;
                }
            }

//errstream() << "phantomxyzab fxj = " << fxj << "\n";
//errstream() << "phantomxyzab fxjgrad = " << fxjgrad << "\n";
//errstream() << "phantomxyzab abs2(fxjgrad) = " << abs2(fxjgrad) << "\n";
//errstream() << "phantomxyzab bgrad = " << bgrad << "\n";

            //vwgradgrad = 1.0;
            //bgradgrad  = 1.0;

            for ( ii = 0 ; ii < wdim ; ++ii )
            {
                res += lambda*vw(ii)*vw(ii)/(2*vv(ii));
                vwgrad("&",ii) += lambda*vw(ii)/vv(ii);
                vwgradgrad("&",ii) += lambda/vv(ii);
                xgrad("&",ii) = -lambda*vw(ii)/(vv(ii)*vv(ii));
            }
//errstream() << "phantomxyzab vwgrad = " << vwgrad << "\n";
//if ( !(*nostop) )
//{
//errstream() << "phantomxyzab vwgrad = " << absinf(vwgrad) << "\n";
//}

        return tres;
    }
};

int calcivObj(double &res, const Vector<double> &x, Vector<double> &gradx, Vector<double> &gradgradx, svmvolatile int &killSwitch, int *nostop, void *objargs);
int calcivObj(double &res, const Vector<double> &x, Vector<double> &gradx, Vector<double> &gradgradx, svmvolatile int &killSwitch, int *nostop, void *objargs)
{
    vgradset &calcit = *((vgradset *) objargs);

    return calcit.calcObj(res,x,gradx,gradgradx,killSwitch,nostop);
}


double SVM_Scalar_rff::intrain(int &xres, svmvolatile int &killSwitch, Vector<double> &vw, Vector<double> &vwaug, Vector<double> &vv, double &b, double lambda, double Lambda)
{
errstream("\n\nV tuning here\n\n");
    NiceAssert( tunev() );
    xres = 0;

    int notfirstcall = 0; // this indicates whether the call to inintrain is first or not
    int fbused = 0; // incrememted whenever offNaiveCholInve requires fallback to compute full inverse rather than shortcut method
    int i;
    int wdim = INDIM*NRff();
    int vdim = NRff();
    int Ndim = N();
    double res = 0;

    // Find v using ADAM optimiser as outer loop.  We want to minimise:
    //
    // Q = (|wv|.^2)'.inv(v) + Lambda/lambda (v-1)'.inv(H).(v-1) + (F/2) (1'v - G)^2
    //   = (|wv|.^2)'.inv(v) + Lambda/lambda (v-1)'.inv(H).(v-1) + (F/2) v'.(1.1').v - FG 1'v + FG^2/2
    //
    // where wv is a function of v (minimises the inner optimisation problem).
    // But we don't want to invert H, so, noting the gradient is:
    //
    // g = -(|wv|.^2).*(inv(v).^2) + 2Lambda/lambda inv(H).(v-1) + F (1.1').v - FG 1
    //
    // our goal becomes to zero this.  Rather than do straight-up gradient
    // descent, we can modify the gradient by multiplying by a positive
    // definite matrix and still find the minima (this is just a generalisation
    // of a scaling factor for 1-d gradient descent).  So, noting that H is
    // positive definite, we can use the scaled gradient (lambda/2lambda).H.g:
    //
    // gmod = -(lambda/(2*Lambda)) H.((|wv|.^2).*(inv(v).^2)) + v - 1 + F H.((1.1').v) - FG H.1
    //
    // Finally, rather than working in v we want to work with iv = inv(v).
    // Note that, if we have f(v), then:
    //
    // df/div = dv/div df/dv dv/div = -inv(iv.^2) df/dv
    //
    // So:
    //
    // dQ/div = -inv(iv.^2).*(dQ/dv) = -diag(iv.^2).*(dQ/dv)
    //
    // But once again diag(iv.^2) is positive definite, so we can just
    // as well do descent in the modified direction:
    //
    // -(dQ/dv)
    //
    // (and as an aside note that this will scale *much* better when iv
    // is small).  The upshot is that we can do descent in terms of iv
    // in the direction of modified gradient:
    //
    // e = gmodmod = (lambda/(2*Lambda)) H.( (|wv|.^2).*(iv.^2) - (2*Lambda/lambda) F (1.1').inv(iv) + (2*Lambda/lambda) FG 1) - inv(iv) + 1
    // dei/divi = (lambda/Lambda) H.( (|wv|.^2).*iv_i + (2*Lambda/lambda) F/(iv_i^2) + 1/(iv.^2)
    //
    // Our strategy then is to let ADAM do the gradient descent on iv,
    // where at each step we:
    //
    // 1. Calculate wv optimal for the given iv (converted to v)
    // 2. Calculate the modified gradient e.
    // 3. Pass this modified gradient back to ADAM and let it do its thing.
    //
    // The one slight modification we make to ADAM is to enforce a constraint
    // 0 < iv <= 1 (so 1 <= v < infty), which is in line with the theorems
    // in the paper that predict that this will just happen anyhow.  Note
    // that the term inv(iv) in e acts as a rather nice barrier function to
    // prevent iv from going negative!

    {
        // Setup H

        //SVM_Scalar::freeRandomFeatures(vdim); // NRff (vdim) here, not dim, as we only need H, not [ H H ; H H ]

        retMatrix<double> tmpmH;
        const Matrix<double> &H = (SVM_Scalar::Gp())(Ndim,1,Ndim+vdim-1,Ndim,1,Ndim+vdim-1,tmpmH);

        // Optimisation info

        int method = outGrad();
        int itcnt = 0;
        int ivdim = ( method < 4 ) ? vdim : ( isVarBias() ? vdim+wdim+1 : vdim+wdim );
//errstream() << "ivdim = " << ivdim << "\n";

        Matrix<double> Hchol;
        int Hcholcalced = 0;
        Vector<double> yinterm;
        Vector<double> xivmod(ivdim);

        Vector<double> iv(ivdim);

        inadamscratchpad.resize(ivdim);

        Vector<int> baseind(SVM_Scalar::N());

        retVector<int> tmpva;
        retVector<int> tmpvb;

        baseind("&",0,1,N()+(wdim-vdim)-1,tmpva) = 0;
        baseind("&",N()+(wdim-vdim),1,N()+wdim-1,tmpva) = cntintvec(vdim,tmpvb);

        retVector<double> tmpvc;
        retVector<double> tmpvd;

        Vector<double> &ivgradbase = (inadamscratchpad.gradx)("&",baseind,tmpvc);
        Vector<double> &ivmodbase = xivmod("&",baseind,tmpvd);

        Vector<double> &Eps = epsscratch;
        Vector<double> &Csc = Csscratch;

        Vector<double> xgrad(wdim);

        Vector<double> ivgradgradbase(vdim);

        // Sparse approximate inverse of diagonals of inv(H)

        for ( i = 0 ; i < vdim ; ++i )
        {
             retVector<double> tmpvaaa;
             retVector<double> tmpvaab;

             ivgradgradbase("&",i) = H(i,i)/norm2(H(i,tmpvaaa,tmpvaab));
        }

/*
for ( i = 0 ; i < N() ; ++i )
{
errstream() << "phantomxdfr Gp(" << i << ") = ";
int j;
for ( j = 0 ; j < wdim ; ++j )
{
errstream() << Gp()(j+Ndim,i) << "\t";
}
errstream() << "\n";
}
errstream() << "phantomxyzaa 0 ivgradgradbase = " << ivgradgradbase << "\n";
*/
        vgradset optinfo(method,1,lambda,Lambda,locminv,locF,locG,H,Hchol,Hcholcalced,vw,vwaug,vv,ivgradbase,ivmodbase,b,innerAdam,notfirstcall,fbused,itcnt,yinterm,xivmod,vdim,wdim,locpegk,costtype,isVarBias(),Eps,Csc,isact,loczr,Atscratch,Btscratch,gscratch.resize(N()),xgrad,ivgradgradbase,*this);

        // Transfer v to iv

        if ( ( method == 0 ) || ( method == 2 ) || ( method == 4 ) || ( method == 6 ) )
        {
            for ( i = 0 ; i < vdim ; ++i )
            {
                /*if ( vv(i) >= locminv )
                {
                    iv("&",i) = 1/vv(i);
                }

                else
                {
                    iv("&",i) = 1/locminv;
                }*/

                iv("&",i) = 1/locminv;
//randufill(iv("&",i),0.9/locminv,1/locminv);
            }
        }

        else
        {
            for ( i = 0 ; i < vdim ; ++i )
            {
                /*if ( vv(i) >= locminv )
                {
                    iv("&",i) = vv(i);
                }

                else
                {
                    iv("&",i) = locminv;
                }*/

                iv("&",i) = locminv;
//randufill(iv("&",i),1*locminv,1.1*locminv);
            }
        }

//errstream() << "phantomxyza b = " << b << "\n";
        if ( method >= 4 )
        {
            for ( i = 0 ; i < wdim ; ++i )
            {
                iv("&",vdim+i) = vw(i);
randufill(iv("&",vdim+i),-1,1);
            }

            if ( isVarBias() )
            {
                iv("&",vdim+wdim) = b;
randufill(iv("&",vdim+wdim),-1,1);
            }
        }

//errstream() << "phantomxyzab b = " << b << "\n";
        // Call ADAM to do its thing!

        //int xsgn = +1;
        int xsgn = ( ( 0 == method ) || ( 2 == method ) ) ? 4 : 3;
        double opttolis = Opttol();

        stopCond sc;

        sc.maxitcnt   = maxitcnt();
        sc.maxruntime = maxtraintime();
        sc.runtimeend = traintimeend();
        sc.tol        = opttolis;

        ADAMopt(res,iv,calcivObj,"Optimisation in SVM random fourier features (v loop)",killSwitch,lr(),(void *) &optinfo,inadamscratchpad,USE_ADAM,sc,ADAM_BETA1,ADAM_BETA2,ADAM_EPS,xsgn,SCHEDCONST,0,locminv,vdim);

        // Retrieve optimal v

//errstream() << "phantomxyzabc b = " << b << "\n";
//errstream() << "phantomxyzabc iv = " << iv << "\n";
        if ( ( method == 0 ) || ( method == 2 ) || ( method == 4 ) || ( method == 6 ) )
        {
            for ( i = 0 ; i < vdim ; ++i )
            {
                vv("&",i) = 1/iv(i);
            }
        }

        else
        {
            for ( i = 0 ; i < vdim ; ++i )
            {
                vv("&",i) = iv(i);
            }
        }

        if ( method >= 4 )
        {
            for ( i = 0 ; i < wdim ; ++i )
            {
                vw("&",i) = iv(vdim+i);
            }

            if ( isVarBias() )
            {
                b = iv(vdim+wdim);
            }
        }

        for ( i = vdim ; i < wdim ; ++i )
        {
            vv("&",i) = vv(i%vdim);
        }

//errstream() << "phantomxyz: vv (xsgn=" << xsgn << ") = " << vv << "\n";
        // Final call to inner loop to calculate optimal vw!

        if ( method < 4 )
        {
            res = inintrain(xres,killSwitch,vw,vwaug,vv,b,lambda,innerAdam,notfirstcall,fbused);
        }
    }

    return res;
}

























































int SVM_Scalar_rff::addTrainingVector(int i, const gentype &zi, const SparseVector<gentype> &x, double Cweigh, double epsweigh, int dval)
{
    return SVM_Scalar_rff::addTrainingVector(i,(double) zi,x,Cweigh,epsweigh,dval); //2);
}

int SVM_Scalar_rff::qaddTrainingVector(int i, const gentype &zi, SparseVector<gentype> &x, double Cweigh, double epsweigh, int dval)
{
    return SVM_Scalar_rff::qaddTrainingVector(i,(double) zi,x,Cweigh,epsweigh,dval); //2);
}

int SVM_Scalar_rff::addTrainingVector(int i, const Vector<gentype> &zi, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    Vector<double> zzi(zi.size());
    Vector<int> ddd(zi.size());

    ddd = 2;

    if ( zi.size() )
    {
        int j;

        for ( j = 0 ; j < zi.size() ; ++j )
        {
            zzi("&",j) = (double) zi(j);
        }
    }

    return SVM_Scalar_rff::addTrainingVector(i,zzi,x,Cweigh,epsweigh,ddd);
}

int SVM_Scalar_rff::qaddTrainingVector(int i, const Vector<gentype> &zi, Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    Vector<double> zzi(zi.size());
    Vector<int> ddd(zi.size());

    ddd = 2;

    if ( zi.size() )
    {
        int j;

        for ( j = 0 ; j < zi.size() ; ++j )
        {
            zzi("&",j) = (double) zi(j);
        }
    }

    return SVM_Scalar_rff::qaddTrainingVector(i,zzi,x,Cweigh,epsweigh,ddd);
}

int SVM_Scalar_rff::addTrainingVector(int i, const Vector<double> &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<int> &d)
{
    NiceAssert( d.size() == x.size() );
    NiceAssert( z.size() == x.size() );
    NiceAssert( z.size() == Cweigh.size() );
    NiceAssert( z.size() == epsweigh.size() );

    int res = 0;

    if ( z.size() )
    {
        int j;

        for ( j = 0 ; j < z.size() ; ++j )
        {
            res |= addTrainingVector(i+j,z(j),x(j),Cweigh(j),epsweigh(j),d(j));
        }
    }

    return res;
}

int SVM_Scalar_rff::qaddTrainingVector(int i, const Vector<double> &z, Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<int> &d)
{
    NiceAssert( d.size() == x.size() );
    NiceAssert( z.size() == x.size() );
    NiceAssert( z.size() == Cweigh.size() );
    NiceAssert( z.size() == epsweigh.size() );

    int res = 0;

    if ( z.size() )
    {
        int j;

        for ( j = 0 ; j < z.size() ; ++j )
        {
            res |= qaddTrainingVector(i+j,z(j),x("&",j),Cweigh(j),epsweigh(j),d(j));
        }
    }

    return res;
}

int SVM_Scalar_rff::removeTrainingVector(int i)
{
    gentype y;
    SparseVector<gentype> x;

    return removeTrainingVector(i,y,x);
}

int SVM_Scalar_rff::removeTrainingVector(int i, int num)
{
    int res = 0;

    if ( num > 0 )
    {
        int j;

        for ( j = num-1 ; j >= 0 ; --j )
        {
            res |= removeTrainingVector(i+j);
        }
    }

    return res;
}

int SVM_Scalar_rff::sety(const Vector<int> &i, const Vector<gentype> &z)
{
    NiceAssert( i.size() == z.size() );

    int res = 0;

    if ( i.size() )
    {
        int j;

        for ( j = 0 ; j < i.size() ; ++j )
        {
            res |= sety(i(j),z(j));
        }
    }

    return res;
}

int SVM_Scalar_rff::sety(const Vector<gentype> &z)
{
    retVector<int> tmpva;

    return sety(cntintvec(N(),tmpva),z);
}

int SVM_Scalar_rff::sety(const Vector<int> &i, const Vector<double> &z)
{
    NiceAssert( i.size() == z.size() );

    int res = 0;

    if ( i.size() )
    {
        int j;

        for ( j = 0 ; j < i.size() ; ++j )
        {
            res |= sety(i(j),z(j));
        }
    }

    return res;
}

int SVM_Scalar_rff::sety(const Vector<double> &z)
{
    retVector<int> tmpva;

    return sety(cntintvec(N(),tmpva),z);
}

/*
int SVM_Scalar_rff::setd(const Vector<int> &i, const Vector<int> &d)
{
    NiceAssert( i.size() == d.size() );

    int res = 0;

    if ( i.size() )
    {
        int j;

        for ( j = 0 ; j < i.size() ; ++j )
        {
            res |= setd(i(j),d(j));
        }
    }

    return res;
}
*/

int SVM_Scalar_rff::setd(const Vector<int> &d)
{
    retVector<int> tmpva;

    return setd(cntintvec(N(),tmpva),d);
}

int SVM_Scalar_rff::setx(const Vector<int> &i, const Vector<SparseVector<gentype> > &x)
{
    NiceAssert( i.size() == x.size() );

    int res = 0;

    if ( i.size() )
    {
        int j;

        for ( j = 0 ; j < i.size() ; ++j )
        {
            res |= setx(i(j),x(j));
        }
    }

    return res;
}

int SVM_Scalar_rff::setx(const Vector<SparseVector<gentype> > &x)
{
    retVector<int> tmpva;

    return setx(cntintvec(N(),tmpva),x);
}

int SVM_Scalar_rff::setCweight(const Vector<int> &i, const Vector<double> &cw)
{
    NiceAssert( i.size() == cw.size() );

    int res = 0;

    if ( i.size() )
    {
        int j;

        for ( j = 0 ; j < i.size() ; ++j )
        {
            res |= setCweight(i(j),cw(j));
        }
    }

    return res;
}


int SVM_Scalar_rff::setepsweight(const Vector<int> &i, const Vector<double> &cw)
{
    NiceAssert( i.size() == cw.size() );

    int res = 0;

    if ( i.size() )
    {
        int j;

        for ( j = 0 ; j < i.size() ; ++j )
        {
            res |= setepsweight(i(j),cw(j));
        }
    }

    return res;
}


int SVM_Scalar_rff::setCweight(const Vector<double> &cw)
{
    retVector<int> tmpva;

    return setCweight(cntintvec(N(),tmpva),cw);
}

int SVM_Scalar_rff::setepsweight(const Vector<double> &cw)
{
    retVector<int> tmpva;

    return setepsweight(cntintvec(N(),tmpva),cw);
}

int SVM_Scalar_rff::setCweightfuzz(const Vector<int> &i, const Vector<double> &cw)
{
    NiceAssert( i.size() == cw.size() );

    int res = 0;

    if ( i.size() )
    {
        int j;

        for ( j = 0 ; j < i.size() ; ++j )
        {
            res |= setCweightfuzz(i(j),cw(j));
        }
    }

    return res;
}

int SVM_Scalar_rff::setCweightfuzz(const Vector<double> &cw)
{
    retVector<int> tmpva;

    return setCweightfuzz(cntintvec(N(),tmpva),cw);
}

























































// Solve x^3 - x^2 + d = 0 for x, and calculate dx/dd
double cubicsolveczero(double d, double &dxdd)
{
    double disc,q,q2,r,dum1,dum2,s,t,s2,t2,term1,r13;
    double drdd,ddiscdd,dsdd,dtdd,ds2dd,dt2dd,dr13dd,ddum2dd;

    q = -1.0/9.0;               // dq/dd = 0
    r = (1.0/27.0) - (d/2.0);   // d=0: 1.0/27.0
                                // dr/dd = -1/2

    drdd = -1.0/2.0;

    disc = q*q*q + r*r;         // d=0: -3^{-6} + 3^{-6} = 0
                                // ddisc/dd = ddisc/dq dq/dd + ddisc/dr dr/dd
                                //          = 2.r.dr/dd

    ddiscdd = 2.0*r*drdd;

    NiceAssert( disc >= 0 );

    term1 = -1.0/3.0;           // d=0: -1.0/3.0
                                // dterm1/dd = 0

    if ( disc > 0 )
    {
        s = r + sqrt(disc);                                   // ds/dd = dr/dd + (1/2).(1/sqrt(disc)).ddisc/dd
        s2 = ( s < 0 ) ? -pow(-s,1.0/3.0) : pow(s,1.0/3.0);   // ds2/dd = (1/3).sgn(s).(1/|s|^{2/3}).ds/dd
        t = r - sqrt(disc);                                   // dt/dd = dr/dd - (1/2).(1/sqrt(disc)).ddisc/dd
        t2 = ( t < 0 ) ? -pow(-t,1.0/3.0) : pow(t,1.0/3.0);   // dt2/dd = (1/3).sgn(t).(1/|t|^{2/3}).dt/dd

        dsdd = drdd + (ddiscdd/(2*sqrt(disc)));
        ds2dd = ( s < 0 ) ? -(1.0/3.0)*pow(-s,-2.0/3.0)*dsdd : (1.0/3.0)*pow(s,-2.0/3.0)*dsdd;
        dtdd = drdd - (ddiscdd/(2*sqrt(disc)));
        dt2dd = ( t < 0 ) ? -(1.0/3.0)*pow(-t,-2.0/3.0)*dtdd : (1.0/3.0)*pow(t,-2.0/3.0)*dtdd;

        dxdd = ds2dd + dt2dd;

        return s2+t2-term1;                                   // dx/dd = ds2/dd + dt2/dd
    }

    if ( disc == 0 )
    {
        r13 = ( r < 0 ) ? -pow(-r,1.0/3.0) : pow(r,1.0/3.0);  // d=0: (3^{-3})^{1/3} = 1/3
                                                              // dr13/dd = (1/3).sgn(r).(1/|r|^{2/3}).dr/dd

        dr13dd = ( r < 0 ) ? -(1.0/3.0)*pow(-r,-2.0/3.0)*drdd : (1.0/3.0)*pow(r,-2.0/3.0)*drdd;

        dxdd = 2*dr13dd;

        return (2*r13) - term1;                               // d=0: 2/3 - -1/3 = 1
                                                              // dx/dd = 2.dr13/dd
    }

    q2 = -q;                    // dq2/dd = -dq/dd = 0
    dum1 = q2*q2*q2;            // ddum1/dd = 3.q2.dq2/dd = 0
    dum2 = acos(r/sqrt(dum1));  // ddum2/dd = -(1/sqrt(1-(r/sqrt(dum1)))).(1/sqrt(dum1)).dr/dd
    r13 = 2.0*sqrt(q2);         // (1/sqrt(q2))/dq2/dd = 0

    ddum2dd = -(1.0/sqrt(1.0-(r/sqrt(dum1))))*(1.0/sqrt(dum1))*drdd;

    dxdd = -r13*sin(dum2/3.0)*ddum2dd;

    return (r13*cos(dum2/3.0)) - term1; // -r13.sin(dum2/3).ddum2/dd
}


/* Original source
function cubicsolve(dataForm)
{
    var a = parseFloat(dataForm.aIn.value);
    var b = parseFloat(dataForm.bIn.value);
    var c = parseFloat(dataForm.cIn.value);
    var d = parseFloat(dataForm.dIn.value);
    if (a == 0)
    {
        alert("The coefficient of the cube of x is 0. Please use the utility for a SECOND degree quadratic. No further action taken.");
        return;
    } //End if a == 0

    if (d == 0)
    {
        alert("One root is 0. Now divide through by x and use the utility for a SECOND degree quadratic to solve the resulting equation for the other two roots. No further action taken.");
        return;
    } //End if d == 0
    b /= a;
    c /= a;
    d /= a;
    var disc, q, r, dum1, s, t, term1, r13;
    q = (3.0*c - (b*b))/9.0;
    r = -(27.0*d) + b*(9.0*c - 2.0*(b*b));
    r /= 54.0;
    disc = q*q*q + r*r;
    dataForm.x1Im.value = 0; //The first root is always real.
    term1 = (b/3.0);
    if (disc > 0) { // one root real, two are complex
        s = r + Math.sqrt(disc);
        s = ((s < 0) ? -Math.pow(-s, (1.0/3.0)) : Math.pow(s, (1.0/3.0)));
        t = r - Math.sqrt(disc);
        t = ((t < 0) ? -Math.pow(-t, (1.0/3.0)) : Math.pow(t, (1.0/3.0)));
        dataForm.x1Re.value = -term1 + s + t;
        term1 += (s + t)/2.0;
        dataForm.x3Re.value = dataForm.x2Re.value = -term1;
        term1 = Math.sqrt(3.0)*(-t + s)/2;
        dataForm.x2Im.value = term1;
        dataForm.x3Im.value = -term1;
        return;
    } 
    // End if (disc > 0)
    // The remaining options are all real
    dataForm.x3Im.value = dataForm.x2Im.value = 0;
    if (disc == 0){ // All roots real, at least two are equal.
        r13 = ((r < 0) ? -Math.pow(-r,(1.0/3.0)) : Math.pow(r,(1.0/3.0)));
        dataForm.x1Re.value = -term1 + 2.0*r13;
        dataForm.x3Re.value = dataForm.x2Re.value = -(r13 + term1);
        return;
    } // End if (disc == 0)
    // Only option left is that all roots are real and unequal (to get here, q < 0)
    q = -q;
    dum1 = q*q*q;
    dum1 = Math.acos(r/Math.sqrt(dum1));
    r13 = 2.0*Math.sqrt(q);
    dataForm.x1Re.value = -term1 + r13*Math.cos(dum1/3.0);
    dataForm.x2Re.value = -term1 + r13*Math.cos((dum1 + 2.0*Math.PI)/3.0);
    dataForm.x3Re.value = -term1 + r13*Math.cos((dum1 + 4.0*Math.PI)/3.0);
    return;
}  //End of cubicSolve
*/






void SVM_Scalar_rff::K0xfer(gentype &res, int &minmaxind, int typeis,
                         const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                         int xdim, int densetype, int resmode, int mlid) const
{
    if ( ( typeis != 808 ) && ( typeis != 818 ) )
    {
        return SVM_Scalar::K0xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xdim,densetype,resmode,mlid);
    }

    if ( resmode >= 8 )
    {
        NiceThrow("Haven't implemented this yet.");
    }

    res = 0.0;
}

void SVM_Scalar_rff::K0xfer(double &res, int &minmaxind, int typeis,
                         double xyprod, double yxprod, double diffis,
                         int xdim, int densetype, int resmode, int mlid) const
{
    if ( ( typeis != 808 ) && ( typeis != 818 ) )
    {
        return SVM_Scalar::K0xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xdim,densetype,resmode,mlid);
    }

    if ( resmode >= 8 )
    {
        NiceThrow("Haven't implemented this yet.");
    }

    res = 0.0;
}

void SVM_Scalar_rff::K1xfer(gentype &res, int &minmaxind, int typeis,
                         const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                         const SparseVector<gentype> &xa, 
                         const vecInfo &xainfo, 
                         int ia, 
                         int xdim, int densetype, int resmode, int mlid) const
{
    if ( ( typeis != 808 ) && ( typeis != 818 ) )
    {
        return SVM_Scalar::K1xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xainfo,ia,xdim,densetype,resmode,mlid);
    }

    if ( resmode >= 8 )
    {
        NiceThrow("Haven't implemented this yet.");
    }

    const Vector<double> &v = locaw;

    int vdim = NRff();
    int Ndim = N();
    int i;
    double vsc = sqrt(1.0/((double) vdim));

    double phia;

    ia = (typeis-(100*(typeis/100)))/10 ? ia : -42;

    res = 0.0;

    for ( i = 0 ; i < vdim ; ++i )
    {
        phia = K2ip(ia,i,nullptr,&xa,&x(i+Ndim).n(),&xainfo,&xinfo(i+Ndim));

        res += v(i)*vsc*cos(phia);

        if ( !ReOnly() )
        {
            res += v(i)*vsc*sin(phia);
        }
    }
}

void SVM_Scalar_rff::K1xfer(double &res, int &minmaxind, int typeis,
                         double xyprod, double yxprod, double diffis,
                         const SparseVector<gentype> &xa, 
                         const vecInfo &xainfo, 
                         int ia, 
                         int xdim, int densetype, int resmode, int mlid) const
{
    if ( ( typeis != 808 ) && ( typeis != 818 ) )
    {
        return SVM_Scalar::K1xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xainfo,ia,xdim,densetype,resmode,mlid);
    }

    if ( resmode >= 8 )
    {
        NiceThrow("Haven't implemented this yet.");
    }

    const Vector<double> &v = locaw;

    int vdim = NRff();
    int Ndim = N();
    int i;
    double vsc = sqrt(1.0/((double) vdim));

    double phia;

    ia = (typeis-(100*(typeis/100)))/10 ? ia : -42;

    res = 0.0;

    for ( i = 0 ; i < vdim ; ++i )
    {
        phia = K2ip(ia,i,nullptr,&xa,&x(i+Ndim).n(),&xainfo,&xinfo(i+Ndim));

        res += v(i)*vsc*cos(phia);

        if ( !ReOnly() )
        {
            res += v(i)*vsc*sin(phia);
        }
    }
}

void SVM_Scalar_rff::K2xfer(gentype &dxyprod, gentype &ddiffis, gentype &res, int &minmaxind, int typeis,
                         const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                         const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                         const vecInfo &xainfo, const vecInfo &xbinfo,
                         int ia, int ib,
                         int xdim, int densetype, int resmode, int mlid) const
{
    if ( ( typeis != 808 ) && ( typeis != 818 ) )
    {
        return SVM_Scalar::K2xfer(dxyprod,ddiffis,res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xainfo,xbinfo,ia,ib,xdim,densetype,resmode,mlid);
    }

    if ( resmode >= 8 )
    {
        NiceThrow("Haven't implemented this yet.");
    }

    const Vector<double> &v = locaw;

    int vdim = NRff();
    int Ndim = N();
    int i;
    double vsc = sqrt(1.0/((double) vdim));

    double phia;
    double phib;

    ia = (typeis-(100*(typeis/100)))/10 ? ia : -42;
    ib = (typeis-(100*(typeis/100)))/10 ? ib : -43;

    res = 0.0;

    for ( i = 0 ; i < vdim ; ++i )
    {
        phia = K2ip(ia,i,nullptr,&xa,&x(i+Ndim).n(),&xainfo,&xinfo(i+Ndim));
        phib = K2ip(ib,i,nullptr,&xb,&x(i+Ndim).n(),&xbinfo,&xinfo(i+Ndim));

        res += v(i)*vsc*cos(phia)*vsc*cos(phib);

        if ( !ReOnly() )
        {
            res += v(i)*vsc*sin(phia)*vsc*sin(phib);
        }
    }
}

void SVM_Scalar_rff::K2xfer(double &dxyprod, double &ddiffis, double &res, int &minmaxind, int typeis,
                         double xyprod, double yxprod, double diffis,
                         const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                         const vecInfo &xainfo, const vecInfo &xbinfo,
                         int ia, int ib,
                         int xdim, int densetype, int resmode, int mlid) const
{
    if ( ( typeis != 808 ) && ( typeis != 818 ) )
    {
        return SVM_Scalar::K2xfer(dxyprod,ddiffis,res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xainfo,xbinfo,ia,ib,xdim,densetype,resmode,mlid);
    }

    if ( resmode >= 8 )
    {
        NiceThrow("Haven't implemented this yet.");
    }

    const Vector<double> &v = locaw;

    int vdim = NRff();
    int Ndim = N();
    int i;
    double vsc = sqrt(1.0/((double) vdim));

    double phia;
    double phib;

    ia = (typeis-(100*(typeis/100)))/10 ? ia : -42;
    ib = (typeis-(100*(typeis/100)))/10 ? ib : -43;

    res = 0.0;

    for ( i = 0 ; i < vdim ; ++i )
    {
        phia = K2ip(ia,i,nullptr,&xa,&x(i+Ndim).n(),&xainfo,&xinfo(i+Ndim));
        phib = K2ip(ib,i,nullptr,&xb,&x(i+Ndim).n(),&xbinfo,&xinfo(i+Ndim));

        res += v(i)*vsc*cos(phia)*vsc*cos(phib);

        if ( !ReOnly() )
        {
            res += v(i)*vsc*sin(phia)*vsc*sin(phib);
        }
    }
}

void SVM_Scalar_rff::K3xfer(gentype &res, int &minmaxind, int typeis,
                         const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                         const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, 
                         const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, 
                         int ia, int ib, int ic, 
                         int xdim, int densetype, int resmode, int mlid) const
{
    if ( ( typeis != 808 ) && ( typeis != 818 ) )
    {
        return SVM_Scalar::K3xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xc,xainfo,xbinfo,xcinfo,ia,ib,ic,xdim,densetype,resmode,mlid);
    }

    if ( resmode >= 8 )
    {
        NiceThrow("Haven't implemented this yet.");
    }

    const Vector<double> &v = locaw;

    int vdim = NRff();
    int Ndim = N();
    int i;
    double vsc = sqrt(1.0/((double) vdim));

    double phia;
    double phib;
    double phic;

    ia = (typeis-(100*(typeis/100)))/10 ? ia : -42;
    ib = (typeis-(100*(typeis/100)))/10 ? ib : -43;
    ic = (typeis-(100*(typeis/100)))/10 ? ic : -44;

    res = 0.0;

    for ( i = 0 ; i < vdim ; ++i )
    {
        phia = K2ip(ia,i,nullptr,&xa,&x(i+Ndim).n(),&xainfo,&xinfo(i+Ndim));
        phib = K2ip(ib,i,nullptr,&xb,&x(i+Ndim).n(),&xbinfo,&xinfo(i+Ndim));
        phic = K2ip(ic,i,nullptr,&xc,&x(i+Ndim).n(),&xcinfo,&xinfo(i+Ndim));

        res += v(i)*vsc*cos(phia)*vsc*cos(phib)*vsc*cos(phic);

        if ( !ReOnly() )
        {
            res += v(i)*vsc*sin(phia)*vsc*sin(phib)*vsc*sin(phic);
        }
    }
}

void SVM_Scalar_rff::K3xfer(double &res, int &minmaxind, int typeis,
                         double xyprod, double yxprod, double diffis,
                         const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, 
                         const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, 
                         int ia, int ib, int ic, 
                         int xdim, int densetype, int resmode, int mlid) const
{
    if ( ( typeis != 808 ) && ( typeis != 818 ) )
    {
        return SVM_Scalar::K3xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xc,xainfo,xbinfo,xcinfo,ia,ib,ic,xdim,densetype,resmode,mlid);
    }

    if ( resmode >= 8 )
    {
        NiceThrow("Haven't implemented this yet.");
    }

    const Vector<double> &v = locaw;

    int vdim = NRff();
    int Ndim = N();
    int i;
    double vsc = sqrt(1.0/((double) vdim));

    double phia;
    double phib;
    double phic;

    ia = (typeis-(100*(typeis/100)))/10 ? ia : -42;
    ib = (typeis-(100*(typeis/100)))/10 ? ib : -43;
    ic = (typeis-(100*(typeis/100)))/10 ? ic : -44;

    res = 0.0;

    for ( i = 0 ; i < vdim ; ++i )
    {
        phia = K2ip(ia,i,nullptr,&xa,&x(i+Ndim).n(),&xainfo,&xinfo(i+Ndim));
        phib = K2ip(ib,i,nullptr,&xb,&x(i+Ndim).n(),&xbinfo,&xinfo(i+Ndim));
        phic = K2ip(ic,i,nullptr,&xc,&x(i+Ndim).n(),&xcinfo,&xinfo(i+Ndim));

        res += v(i)*vsc*cos(phia)*vsc*cos(phib)*vsc*cos(phic);

        if ( !ReOnly() )
        {
            res += v(i)*vsc*sin(phia)*vsc*sin(phib)*vsc*sin(phic);
        }
    }
}

void SVM_Scalar_rff::K4xfer(gentype &res, int &minmaxind, int typeis,
                         const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                         const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                         const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
                         int ia, int ib, int ic, int id,
                         int xdim, int densetype, int resmode, int mlid) const
{
    if ( ( typeis != 808 ) && ( typeis != 818 ) )
    {
        return SVM_Scalar::K4xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,ia,ib,ic,id,xdim,densetype,resmode,mlid);
    }

    if ( resmode >= 8 )
    {
        NiceThrow("Haven't implemented this yet.");
    }

    const Vector<double> &v = locaw;

    int vdim = NRff();
    int Ndim = N();
    int i;
    double vsc = sqrt(1.0/((double) vdim));

    double phia;
    double phib;
    double phic;
    double phid;

    ia = (typeis-(100*(typeis/100)))/10 ? ia : -42;
    ib = (typeis-(100*(typeis/100)))/10 ? ib : -43;
    ic = (typeis-(100*(typeis/100)))/10 ? ic : -44;
    id = (typeis-(100*(typeis/100)))/10 ? id : -45;

    res = 0.0;

    for ( i = 0 ; i < vdim ; ++i )
    {
        phia = K2ip(ia,i,nullptr,&xa,&x(i+Ndim).n(),&xainfo,&xinfo(i+Ndim));
        phib = K2ip(ib,i,nullptr,&xb,&x(i+Ndim).n(),&xbinfo,&xinfo(i+Ndim));
        phic = K2ip(ic,i,nullptr,&xc,&x(i+Ndim).n(),&xcinfo,&xinfo(i+Ndim));
        phid = K2ip(id,i,nullptr,&xd,&x(i+Ndim).n(),&xdinfo,&xinfo(i+Ndim));

        res += v(i)*vsc*cos(phia)*vsc*cos(phib)*vsc*cos(phic)*vsc*cos(phid);

        if ( !ReOnly() )
        {
            res += v(i)*vsc*sin(phia)*vsc*sin(phib)*vsc*sin(phic)*vsc*sin(phid);
        }
    }
}

void SVM_Scalar_rff::K4xfer(double &res, int &minmaxind, int typeis,
                         double xyprod, double yxprod, double diffis,
                         const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                         const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
                         int ia, int ib, int ic, int id,
                         int xdim, int densetype, int resmode, int mlid) const
{
    if ( ( typeis != 808 ) && ( typeis != 818 ) )
    {
        return SVM_Scalar::K4xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,ia,ib,ic,id,xdim,densetype,resmode,mlid);
    }

    if ( resmode >= 8 )
    {
        NiceThrow("Haven't implemented this yet.");
    }

    const Vector<double> &v = locaw;

    int vdim = NRff();
    int Ndim = N();
    int i;
    double vsc = sqrt(1.0/((double) vdim));

    double phia;
    double phib;
    double phic;
    double phid;

    ia = (typeis-(100*(typeis/100)))/10 ? ia : -42;
    ib = (typeis-(100*(typeis/100)))/10 ? ib : -43;
    ic = (typeis-(100*(typeis/100)))/10 ? ic : -44;
    id = (typeis-(100*(typeis/100)))/10 ? id : -45;

    res = 0.0;

    for ( i = 0 ; i < vdim ; ++i )
    {
        phia = K2ip(ia,i,nullptr,&xa,&x(i+Ndim).n(),&xainfo,&xinfo(i+Ndim));
        phib = K2ip(ib,i,nullptr,&xb,&x(i+Ndim).n(),&xbinfo,&xinfo(i+Ndim));
        phic = K2ip(ic,i,nullptr,&xc,&x(i+Ndim).n(),&xcinfo,&xinfo(i+Ndim));
        phid = K2ip(id,i,nullptr,&xd,&x(i+Ndim).n(),&xdinfo,&xinfo(i+Ndim));

        res += v(i)*vsc*cos(phia)*vsc*cos(phib)*vsc*cos(phic)*vsc*cos(phid);

        if ( !ReOnly() )
        {
            res += v(i)*vsc*sin(phia)*vsc*sin(phib)*vsc*sin(phic)*vsc*sin(phid);
        }
    }
}

void SVM_Scalar_rff::Kmxfer(gentype &res, int &minmaxind, int typeis,
                         const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                         Vector<const SparseVector<gentype> *> &xx,
                         Vector<const vecInfo *> &xxinfo,
                         Vector<int> &iii,
                         int xdim, int m, int densetype, int resmode, int mlid) const
{
    if ( ( typeis != 808 ) && ( typeis != 818 ) )
    {
        return SVM_Scalar::Kmxfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xx,xxinfo,iii,xdim,m,densetype,resmode,mlid);
    }

    if ( resmode >= 8 )
    {
        NiceThrow("Haven't implemented this yet.");
    }

    const Vector<double> &v = locaw;

    int vdim = NRff();
    int Ndim = N();
    int i,j;
    double vsc = sqrt(1.0/((double) vdim));

    double phi;

    res = 0.0;

    for ( i = 0 ; i < vdim ; ++i )
    {
        double tmpa = v(i);
        double tmpb = v(i);

        for ( j = 0 ; j < xx.size() ; ++j )
        {
            int ii = (typeis-(100*(typeis/100)))/10 ? iii(j) : -42-j;

            phi = K2ip(ii,i,nullptr,xx(j),&x(i+Ndim).n(),xxinfo(j),&xinfo(i+Ndim));

            tmpa *= vsc*cos(phi);
            tmpb *= vsc*sin(phi);
        }

        res += tmpa;

        if ( !ReOnly() )
        {
            res += tmpb;
        }
    }
}

void SVM_Scalar_rff::Kmxfer(double &res, int &minmaxind, int typeis,
                         double xyprod, double yxprod, double diffis,
                         Vector<const SparseVector<gentype> *> &xx,
                         Vector<const vecInfo *> &xxinfo,
                         Vector<int> &iii,
                         int xdim, int m, int densetype, int resmode, int mlid) const
{
    if ( ( typeis != 808 ) && ( typeis != 818 ) )
    {
        return SVM_Scalar::Kmxfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xx,xxinfo,iii,xdim,m,densetype,resmode,mlid);
    }

    if ( resmode >= 8 )
    {
        NiceThrow("Haven't implemented this yet.");
    }

    const Vector<double> &v = locaw;

    int vdim = NRff();
    int Ndim = N();
    int i,j;
    double vsc = sqrt(1.0/((double) vdim));

    double phi;

    res = 0.0;

    for ( i = 0 ; i < vdim ; ++i )
    {
        double tmpa = v(i);
        double tmpb = v(i);

        for ( j = 0 ; j < xx.size() ; ++j )
        {
            int ii = (typeis-(100*(typeis/100)))/10 ? iii(j) : -42-j;

            phi = K2ip(ii,i,nullptr,xx(j),&x(i+Ndim).n(),xxinfo(j),&xinfo(i+Ndim));

            tmpa *= vsc*cos(phi);
            tmpb *= vsc*sin(phi);
        }

        res += tmpa;

        if ( !ReOnly() )
        {
            res += tmpb;
        }
    }
}

