
//
// Basic kernel class
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
#include <math.h>
#ifdef ENABLE_THREADS
#include <mutex>
#endif
#include "mercer.hpp"
#include "randfun.hpp"



//Adding kernels: search for ADDHERE


SparseVector<kernPrecursor *>* kernPrecursor::fullmllist = nullptr;
#ifdef ENABLE_THREADS
std::mutex kernPrecursor::kerneyelock;
#endif



// Constructors and assignment operators

MercerKernel::MercerKernel()
{
    isind           = 0;
    backupisind     = 0;
    isshift         = 0;
    isprod          = 0;
    isdiffalt       = 1;
    isfullnorm      = 0;
    issymmset       = 0;
    xdenseZeroPoint = -1.0;

    leftplain  = 0;
    rightplain = 0;

    xisfast       = -1;
    xneedsInner   = -1;
    xneedsInnerm2 = -1;
    xneedsDiff    = -1;
    xneedsNorm    = -1;

    static const gentype tempsampdist("[ ]");

    xnumsamples   = DEFAULT_NUMKERNSAMP;
    xsampdist     = tempsampdist;
    xnumSplits    = 0;
    xnumMulSplits = 0;

    dtype.resize(1) = 2; // to stop valgrind complaining when we call setType below
    isnorm.resize(1);
    ischain.resize(1);
    issplit.resize(1);
    mulsplit.resize(1);
    ismagterm.resize(1);
    kernflags.resize(1);
    altcallback.resize(1);
    randFeats.resize(1);
    randFeatAngle.resize(1);
    randFeatReOnly.resize(1);
    randFeatNoAngle.resize(1);

    randFeats("&",0).resize(0);
    randFeatAngle("&",0).resize(0);
    randFeatReOnly("&",0) = 1;
    randFeatNoAngle("&",0) = 1;

    dRealConstants.resize(1);
    dIntConstants.resize(1);
    dRealConstantsLB.resize(1);
    dIntConstantsLB.resize(1);
    dRealConstantsUB.resize(1);
    dIntConstantsUB.resize(1);
    dRealOverwrite.resize(1);
    dIntOverwrite.resize(1);

    dShiftProd        = 0;
    dShiftProdNoConj  = 0;
    dShiftProdRevConj = 0;

    xproddepth      = 4; // Needs to be 4 for the K4 optimisations to work properly
    enchurn         = 0;
    xsuggestXYcache = 0;
    xisIPdiffered   = 0;

    haslinconstr = true;

    altcallback = MLid();

    setType(2,0);          // quadratic kernel, default parameters
    setUnNormalised(0);    // not normalised
    setUnChained(0);       // not chained
    setUnSplit(0);         // not split
    setUnMulSplit(0);      // not split
    setUnMagTerm(0);       // not magnitude term

    dRealOverwrite("&",0).zero();
    dIntOverwrite("&",0).zero();

    return;
}

MercerKernel::MercerKernel(const MercerKernel &src)
{
    *this = src;

    return;
}

MercerKernel::~MercerKernel()
{
    return;
}

MercerKernel &MercerKernel::operator=(const MercerKernel &src)
{
    isprod               = src.isprod;
    isind                = src.isind;
    isfullnorm           = src.isfullnorm;
    issymmset            = src.issymmset;
    isshift              = src.isshift;
    leftplain            = src.leftplain;
    rightplain           = src.rightplain;
    isdiffalt            = src.isdiffalt;
    xproddepth           = src.xproddepth;
    enchurn              = src.enchurn;
    xsuggestXYcache      = src.xsuggestXYcache;
    xisIPdiffered        = src.xisIPdiffered;
    xnumSplits           = src.xnumSplits;
    xnumMulSplits        = src.xnumMulSplits;
    xdenseZeroPoint      = src.xdenseZeroPoint;

    dtype                = src.dtype;
    isnorm               = src.isnorm;
    ischain              = src.ischain;
    issplit              = src.issplit;
    mulsplit             = src.mulsplit;
    ismagterm            = src.ismagterm;
    xranktype            = src.xranktype;
    dIndexes             = src.dIndexes;
    kernflags            = src.kernflags;
    dRealConstants       = src.dRealConstants;
    dIntConstants        = src.dIntConstants;
    dRealConstantsLB     = src.dRealConstantsLB;
    dIntConstantsLB      = src.dIntConstantsLB;
    dRealConstantsUB     = src.dRealConstantsUB;
    dIntConstantsUB      = src.dIntConstantsUB;
    dRealOverwrite       = src.dRealOverwrite;
    dIntOverwrite        = src.dIntOverwrite;
    altcallback          = src.altcallback;
    randFeats            = src.randFeats;
    randFeatAngle        = src.randFeatAngle;
    randFeatReOnly       = src.randFeatReOnly;
    randFeatNoAngle      = src.randFeatNoAngle;

    linGradOrd           = src.linGradOrd;
    linGradScal          = src.linGradScal;
    linGradScalTsp       = src.linGradScalTsp;
    haslinconstr         = src.haslinconstr;

    dShift               = src.dShift;
    dScale               = src.dScale;
    dShiftProd           = src.dShiftProd;
    dShiftProdNoConj     = src.dShiftProdNoConj;
    dShiftProdRevConj    = src.dShiftProdRevConj;

    xnumsamples          = src.xnumsamples;
    xindsub              = src.xindsub;
    xsampdist            = src.xsampdist;

    combinedOverwriteSrc      = src.combinedOverwriteSrc;
    backupisind               = src.backupisind;
    backupdIndexes            = src.backupdIndexes;

    xisfast       = src.xisfast;
    xneedsInner   = src.xneedsInner;
    xneedsInnerm2 = src.xneedsInnerm2;
    xneedsDiff    = src.xneedsDiff;
    xneedsNorm    = src.xneedsNorm;

    return *this;
}















double MercerKernel::effweight(int q) const
{
    NiceAssert( q < size() );

    double res = 1;

    do
    {
        res *= (double) cWeight(q);
        ++q;
    }
    while ( ( q < size() ) && ( isSplit(q-1) == 1 ) );

    return res;
}



int MercerKernel::getSymmetry(void) const
{
//ADDHERE - returns +1 if K(x,y) = conj(K(y,x))
//                  -1 if K(x,y) = -conj(K(y,x))
//                  0  otherwise
    int res = 2; // Initial value
    int q;

    //FIXME: technically symmetry can be maintained in more general cases
    //       for chained kernels.  This won't give bad results, but may
    //       report no symmetry when symmetry is in fact present.

    for ( q = 0 ; ( q < size() ) && res ; ++q )
    {
        switch ( cType(q) )
        {
            case 17:   case 27:
            case 1003: case 1038:
            case 2003: case 2038:
            {
                // Antisymmetric case

                if ( res == 2 )
                {
                    res = -1;
                }

                else
                {
                    res = 0;
                }

                break;
            }

            case 28:
            case 400: case 401: case 402: case 403: case 404:
            case 450: case 451: case 452: case 453: case 454:
            case 600: case 601: case 602: case 603: case 604:
            case 650: case 651: case 652: case 653: case 654:
            {
                // Asymmetric case

                res = 0;

                break;
            }

            default:
            {
                // Symmetric case

                if ( ( res == 2 ) || ( res == +1 ) )
                {
                    res = +1;
                }

                else
                {
                    res = 0;
                }

                break;
            }
        }
    }

    return ( res == 2 ) ? 0 : res;
}

int MercerKernel::iskern(int potind) const
{
//ADDHERE - returns 1 if kernel index exists, 0 otherwise
    switch ( potind )
    {
        case 0:    case 1:    case 2:    case 3:    case 4:
        case 5:    case 6:    case 7:    case 8:    case 9:
        case 10:   case 11:   case 12:   case 13:   case 14:
        case 15:   case 16:   case 17:   case 18:   case 19:
        case 20:   case 21:   case 22:   case 23:   case 24:
        case 25:   case 26:   case 27:   case 28:   case 29:
        case 30:   case 31:   case 32:   case 33:   case 34:
        case 35:   case 36:   case 37:   case 38:   case 39:
        case 40:   case 41:   case 42:   case 43:   case 44:
        case 45:   case 46:   case 47:   case 48:   case 49:
        case 50:   case 51:   case 52:   case 53:
        case 100:  case 101:  case 102:  case 103:  case 104:
        case 105:  case 106:
        case 200:  case 201:  case 202:  case 203:  case 204:
        case 205:  case 206:
        case 300:  case 301:  case 302:  case 303:  case 304:
        case 400:  case 401:  case 402:  case 403:  case 404:
        case 450:  case 451:  case 452:  case 453:  case 454:
        case 500:  case 501:  case 502:  case 503:  case 504:
        case 550:  case 551:  case 552:  case 553:  case 554:
        case 600:  case 601:  case 602:  case 603:  case 604:
        case 650:  case 651:  case 652:  case 653:  case 654:
        case 700:  case 701:  case 702:  case 703:  case 704:
        case 750:  case 751:  case 752:  case 753:  case 754:
        case 800:  case 801:  case 802:  case 803:  case 804:
        case 805:  case 806:  case 807:  case 808:  case 809:
        case 810:  case 811:  case 812:  case 813:  case 814:
        case 815:  case 816:  case 817:  case 818:  case 819:
        case 820:  case 821:  case 822:  case 823:  case 824:
        case 825:  case 826:  case 827:  case 828:  case 829:
        case 830:  case 831:  case 832:  case 833:  case 834:
        case 835:  case 836:  case 837:  case 838:  case 839:
        case 840:  case 841:  case 842:  case 843:  case 844:
        case 845:  case 846:  case 847:  case 848:  case 849:
        case 850:  case 851:  case 852:  case 853:  case 854:
        case 855:  case 856:  case 857:  case 858:  case 859:
        case 860:  case 861:  case 862:  case 863:  case 864:
        case 865:  case 866:  case 867:  case 868:  case 869:
        case 870:  case 871:  case 872:  case 873:  case 874:
        case 875:  case 876:  case 877:  case 878:  case 879:
        case 880:  case 881:  case 882:  case 883:  case 884:
        case 885:  case 886:  case 887:  case 888:  case 889:
        case 890:  case 891:  case 892:  case 893:  case 894:
        case 895:  case 896:  case 897:  case 898:  case 899:
        case 1003: case 2003:
        {
            return 1;
        }

        default:
        {
            break;
        }
    }

    return 0;
}

double constexpr calcDenseDerivPair(int &typeis, int adensetype, int bdensetype, int xdim)
{
    double symm = -1;

    if ( adensetype && bdensetype )
    {
        typeis = -1;
        return symm;
    }

//ADDHERE: need to register kernel pairs
    if ( xdim != 1 )
    {
        switch ( typeis )
        {
            case 3:    { typeis = 1003; break; }
            case 2003: { typeis = 3;    break; }
            case 400:  { typeis = 500;  break; }
            case 401:  { typeis = 501;  break; }
            case 402:  { typeis = 502;  break; }
            case 403:  { typeis = 503;  break; }
            case 404:  { typeis = 504;  break; }
            case 450:  { typeis = 550;  break; }
            case 451:  { typeis = 551;  break; }
            case 452:  { typeis = 552;  break; }
            case 453:  { typeis = 553;  break; }
            case 454:  { typeis = 554;  break; }
            case 600:  { typeis = 700;  break; }
            case 601:  { typeis = 701;  break; }
            case 602:  { typeis = 702;  break; }
            case 603:  { typeis = 703;  break; }
            case 604:  { typeis = 704;  break; }
            case 650:  { typeis = 750;  break; }
            case 651:  { typeis = 751;  break; }
            case 652:  { typeis = 752;  break; }
            case 653:  { typeis = 753;  break; }
            case 654:  { typeis = 754;  break; }
            default:   { typeis = -1;   break; }
        }
    }

    else
    {
        switch ( typeis )
        {
            case 3:    { typeis = 1003; break; }
            case 2003: { typeis = 3;    break; }
            case 400:  { typeis = 500;  break; }
            case 401:  { typeis = 501;  break; }
            case 402:  { typeis = 502;  break; }
            case 403:  { typeis = 503;  break; }
            case 404:  { typeis = 504;  break; }
            case 450:  { typeis = 550;  break; }
            case 451:  { typeis = 551;  break; }
            case 452:  { typeis = 552;  break; }
            case 453:  { typeis = 553;  break; }
            case 454:  { typeis = 554;  break; }
            case 600:  { typeis = 700;  break; }
            case 601:  { typeis = 701;  break; }
            case 602:  { typeis = 702;  break; }
            case 603:  { typeis = 703;  break; }
            case 604:  { typeis = 704;  break; }
            case 650:  { typeis = 750;  break; }
            case 651:  { typeis = 751;  break; }
            case 652:  { typeis = 752;  break; }
            case 653:  { typeis = 753;  break; }
            case 654:  { typeis = 754;  break; }

            case 38:   { typeis = 1038; break; }
            case 2038: { typeis = 38;   break; }
            default:   { typeis = -1;   break; }
        }
    }

    return adensetype ? 1 : symm;
}

double constexpr calcDenseIntPair(int &typeis, int adensetype, int bdensetype, int xdim)
{
    double symm = -1;

    if ( adensetype && bdensetype )
    {
        typeis = -1;
        return symm;
    }

//ADDHERE: need to register kernel pairs
    if ( xdim != 1 )
    {
        switch ( typeis )
        {
            case 3:    { typeis = 2003; break; }
            case 1003: { typeis = 3;    break; }
            case 500:  { typeis = 400;  break; }
            case 501:  { typeis = 401;  break; }
            case 502:  { typeis = 402;  break; }
            case 503:  { typeis = 403;  break; }
            case 504:  { typeis = 404;  break; }
            case 550:  { typeis = 450;  break; }
            case 551:  { typeis = 451;  break; }
            case 552:  { typeis = 452;  break; }
            case 553:  { typeis = 453;  break; }
            case 554:  { typeis = 454;  break; }
            case 700:  { typeis = 600;  break; }
            case 701:  { typeis = 601;  break; }
            case 702:  { typeis = 602;  break; }
            case 703:  { typeis = 603;  break; }
            case 704:  { typeis = 604;  break; }
            case 750:  { typeis = 650;  break; }
            case 751:  { typeis = 651;  break; }
            case 752:  { typeis = 652;  break; }
            case 753:  { typeis = 653;  break; }
            case 754:  { typeis = 654;  break; }
            default:   { typeis = -1;   break; }
        }
    }

    else
    {
        switch ( typeis )
        {
            case 3:    { typeis = 2003; break; }
            case 1003: { typeis = 3;    break; }
            case 500:  { typeis = 400;  break; }
            case 501:  { typeis = 401;  break; }
            case 502:  { typeis = 402;  break; }
            case 503:  { typeis = 403;  break; }
            case 504:  { typeis = 404;  break; }
            case 550:  { typeis = 450;  break; }
            case 551:  { typeis = 451;  break; }
            case 552:  { typeis = 452;  break; }
            case 553:  { typeis = 453;  break; }
            case 554:  { typeis = 454;  break; }
            case 700:  { typeis = 600;  break; }
            case 701:  { typeis = 601;  break; }
            case 702:  { typeis = 602;  break; }
            case 703:  { typeis = 603;  break; }
            case 704:  { typeis = 604;  break; }
            case 750:  { typeis = 650;  break; }
            case 751:  { typeis = 651;  break; }
            case 752:  { typeis = 652;  break; }
            case 753:  { typeis = 653;  break; }
            case 754:  { typeis = 654;  break; }

            case 38:   { typeis = 2038; break; }
            case 1038: { typeis = 38;   break; }
            default:   { typeis = -1;   break; }
        }
    }

    return ( adensetype > 0 ) ? 1 : symm;
}

double constexpr calcDensePair(int &typeis, int adensetype, int bdensetype, int xdim)
{
    if ( ( adensetype == +1 ) || ( bdensetype == +1 ) )
    {
        // Actually want dense derivative

        return calcDenseDerivPair(typeis,adensetype,bdensetype,xdim);
    }

    else if ( ( adensetype == +2 ) || ( bdensetype == +2 ) )
    {
        // Actually want dense integral

        return calcDenseIntPair(typeis,adensetype,bdensetype,xdim);
    }

    return 1;
}






// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================

// Modifiers:

MercerKernel &MercerKernel::add(int q)
{
    NiceAssert( ( q >= 0 ) && ( q <= size() ) );

    xisfast       = -1;
    xneedsInner   = -1;
    xneedsInnerm2 = -1;
    xneedsDiff    = -1;
    xneedsNorm    = -1;

    dtype.add(q);
    isnorm.add(q);
    ischain.add(q);
    issplit.add(q);
    mulsplit.add(q);
    ismagterm.add(q);
    kernflags.add(q);
    altcallback.add(q);
    randFeats.add(q);
    randFeatAngle.add(q);
    randFeatReOnly.add(q);
    randFeatNoAngle.add(q);

    randFeats("&",q).resize(0);
    randFeatAngle("&",q).resize(0);
    randFeatReOnly("&",q) = 1;
    randFeatNoAngle("&",q) = 1;

    dRealConstants.add(q);
    dIntConstants.add(q);
    dRealConstantsLB.add(q);
    dIntConstantsLB.add(q);
    dRealConstantsUB.add(q);
    dIntConstantsUB.add(q);
    dRealOverwrite.add(q);
    dIntOverwrite.add(q);

    altcallback("&",q) = MLid();

    setType(0,q);          // constant kernel, default parameters
    setUnNormalised(q);    // not normalised
    setUnChained(q);       // not chained
    setUnSplit(q);         // not split
    setUnMulSplit(q);      // not split
    setUnMagTerm(q);       // not magnitude term

    dRealOverwrite("&",q).zero();
    dIntOverwrite("&",q).zero();

    return *this;
}

MercerKernel &MercerKernel::remove(int q)
{
    NiceAssert( ( q >= 0 ) && ( q < size() ) );

    xisfast       = -1;
    xneedsInner   = -1;
    xneedsInnerm2 = -1;
    xneedsDiff    = -1;
    xneedsNorm    = -1;

    dtype.remove(q);
    isnorm.remove(q);
    ischain.remove(q);
    issplit.remove(q);
    mulsplit.remove(q);
    ismagterm.remove(q);
    kernflags.remove(q);
    altcallback.remove(q);
    randFeats.remove(q);
    randFeatAngle.remove(q);
    randFeatReOnly.remove(q);
    randFeatNoAngle.remove(q);

    dRealConstants.remove(q);
    dIntConstants.remove(q);
    dRealConstantsLB.remove(q);
    dIntConstantsLB.remove(q);
    dRealConstantsUB.remove(q);
    dIntConstantsUB.remove(q);
    dRealOverwrite.remove(q);
    dIntOverwrite.remove(q);

    fixcombinedOverwriteSrc();

    return *this;
}

MercerKernel &MercerKernel::resize(int nsize)
{
    NiceAssert( nsize >= 0 );

    xisfast       = -1;
    xneedsInner   = -1;
    xneedsInnerm2 = -1;
    xneedsDiff    = -1;
    xneedsNorm    = -1;

    int oldsize = size();
    int q;

    dtype.resize(nsize);
    isnorm.resize(nsize);
    ischain.resize(nsize);
    issplit.resize(nsize);
    mulsplit.resize(nsize);
    ismagterm.resize(nsize);
    kernflags.resize(nsize);
    altcallback.resize(nsize);
    randFeats.resize(nsize);
    randFeatAngle.resize(nsize);
    randFeatReOnly.resize(nsize);
    randFeatNoAngle.resize(nsize);

    dRealConstants.resize(nsize);
    dIntConstants.resize(nsize);
    dRealConstantsLB.resize(nsize);
    dIntConstantsLB.resize(nsize);
    dRealConstantsUB.resize(nsize);
    dIntConstantsUB.resize(nsize);
    dRealOverwrite.resize(nsize);
    dIntOverwrite.resize(nsize);

    if ( oldsize < size() )
    {
	for ( q = oldsize ; q < size() ; ++q )
	{
            altcallback("&",q) = MLid();

            dtype("&",q) = 0;

            setType(0,q);          // constant kernel, default parameters
	    setUnNormalised(q);    // not normalised
            setUnChained(q);       // not chained
            setUnSplit(q);         // not split
            setUnMulSplit(q);      // not split
            setUnMagTerm(q);       // not magnitude term

            randFeats("&",q).resize(0);
            randFeatAngle("&",q).resize(0);
            randFeatReOnly("&",q) = 1;
            randFeatNoAngle("&",q) = 1;

            dRealOverwrite("&",q).zero();
            dIntOverwrite("&",q).zero();
	}
    }

    fixcombinedOverwriteSrc();

    return *this;
}


void MercerKernel::recalcRandFeats(int q, int numFeats)
{
    // Currently a stub function, to be filled later when random features are incorporated fully here

    //Vector<Vector<SparseVector<gentype> > > randFeats;
    //Vector<Vector<double> > randFeatAngle;
    //Vector<int> randFeatReOnly;
    //Vector<int> randFeatAngleOnly;

    if ( q < 0 )
    {
        for ( q = 0 ; q < size() ; ++q )
        {
            recalcRandFeats(q,numFeats);
        }
    }

    else
    {
        if ( numFeats >= 0 )
        {
            randFeats("&",q).resize(numFeats);
            randFeatAngle("&",q).resize(numFeats);
        }

        numFeats = randFeats(q).size();

        if ( numFeats )
        {
            int i,r;
            const double lengthscale(getRealConstZero(q));

            for ( i = 0 ; i < numFeats ; ++i )
            {
                SparseVector<gentype> &xxa = randFeats("&",q)("&",i);
                double &xaa = randFeatAngle("&",q)("&",i);

                randnfill(xaa,0,2*NUMBASE_PI);

                xxa.indalign(defindKey());
                xxa = 0.0_gent;

                for ( r = 0 ; r < xxa.nindsize() ; ++r )
                {
                    if ( cType(q) == 3 )
                    {
                        randnfill(xxa.direref(r).force_double(),0,1/lengthscale); // Gaussian
                    }

                    else if ( cType(q) == 4 )
                    {
                        randCfill(xxa.direref(r).force_double(),0,1/lengthscale); // Cauchy
                    }

                    else if ( cType(q) == 13 )
                    {
                        randufill(xxa.direref(r).force_double(),-NUMBASE_PI/lengthscale,NUMBASE_PI/lengthscale); // Uniform
                    }

                    else if ( cType(q) == 19 )
                    {
                        randefill(xxa.direref(r).force_double(),1/lengthscale); // Exponential
                    }

                    else
                    {
                        NiceThrow("Random features not defined for this kernel type");
                    }
                }
            }
        }
    }

    return;
}










// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================


// Element retrieval

//gentype &MercerKernel::xelm(gentype &res, const SparseVector<gentype> &x, int i, int j) const
gentype &MercerKernel::xelm(gentype &res, const SparseVector<gentype> &x, int, int j) const
{
    //(void) i;

    // FIXME: being lazy here and assuming no vector-level modification

    int isvalnz = 1;
    int xindis = x.ind(j);

    if ( isIndex() )
    {
        isvalnz = 0;

        if ( dIndexes.size() )
        {
            int l;

            for ( l = 0 ; l < dIndexes.size() ; ++l )
            {
                if ( dIndexes(l) == xindis )
                {
                    isvalnz = 1;
                    break;
                }
            }
        }
    }

    if ( isvalnz )
    {
        res = x.direcref(j);

        if ( isShifted() || isShiftedScaled() )
        {
            res -= dShift(xindis);
        }

        if ( ( isScaled() || isShiftedScaled() ) && ( dScale.isindpresent(j) ) )
        {
            res /= dScale(xindis);
        }
    }

    else
    {
        res = 0.0;
    }

    return res;
}

//int MercerKernel::xindsize(const SparseVector<gentype> &x, int i) const
int MercerKernel::xindsize(const SparseVector<gentype> &x, int) const
{
    //(void) i;

    return x.nindsize();
}

































// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================



// ====================================================================================

double MercerKernel::distK(const SparseVector<gentype> &x, const SparseVector<gentype> &y,
                           const vecInfo &xinfo, const vecInfo &yinfo,
                           int i, int j,
                           int xdim, int xconsist, int mlid, const double *xy00, const double *xy10, const double *xy11, int assumreal) const
{
    double res = 0;
    const gentype bias(0);

    if ( !isSimpleDistKernel() )
    {
        res += K2(x,x,xinfo,xinfo,bias,nullptr,i,i,xdim,xconsist,0,mlid,xy00,xy00,xy00,assumreal);
        res += K2(y,y,yinfo,yinfo,bias,nullptr,j,j,xdim,xconsist,0,mlid,xy11,xy11,xy11,assumreal);
    }

    res -= 2*K2(x,y,xinfo,yinfo,bias,nullptr,i,j,xdim,xconsist,0,mlid,xy00,xy10,xy11,assumreal);

    return res;
}



void MercerKernel::ddistKdx(double &xscaleres, double &yscaleres,
                            int &minmaxind,
                            const SparseVector<gentype> &x, const SparseVector<gentype> &y,
                            const vecInfo &xinfo, const vecInfo &yinfo,
                            int i, int j,
                            int xdim, int xconsist, int mlid, const double *xy00, const double *xy10, const double *xy11, int assumreal) const
{
    gentype xres;
    gentype yres;
    gentype bres;
    gentype dummybias(0);

    dK2delx(xres,yres,minmaxind,x,y,xinfo,yinfo,dummybias,nullptr,i,j,xdim,xconsist,mlid,xy00,xy10,xy11,assumreal);

    xscaleres = -2*((double) xres);
    yscaleres = -2*((double) yres);

    if ( !isSimpleDistKernel() )
    {
        dK2delx(xres,yres,minmaxind,x,x,xinfo,xinfo,dummybias,nullptr,i,i,xdim,xconsist,mlid,xy00,xy10,xy11,assumreal);

        xscaleres += (double) xres;
        xscaleres += (double) yres;

        dK2delx(xres,yres,minmaxind,y,y,yinfo,yinfo,dummybias,nullptr,j,j,xdim,xconsist,mlid,xy00,xy10,xy11,assumreal);

        yscaleres += (double) xres;
        yscaleres += (double) yres;
    }
}

//phantomx
void MercerKernel::densedKdx(double &res, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, double bias, int i, int j, int xdim, int xconsist, int mlid, int assumreal) const
{
    NiceAssert( !isprod );

    // x now appropriately constructed as required

    gentype xyprod;
    gentype yxprod;

    if ( assumreal )
    {
        xyprod.force_double();
        yxprod.force_double();
    }

    innerProductDiverted(xyprod,x,y,xconsist,assumreal);
    innerProductDivertedRevConj(yxprod,xyprod,x,y,xconsist,assumreal);

    xyprod += bias;
    yxprod += bias;

    // NB: second last argument 1 to indicate dense derivative
    gentype temp;
    K2i(temp,xyprod,yxprod,xinfo,yinfo,getmnorm(xinfo,x,2,xconsist,assumreal),getmnorm(yinfo,y,2,xconsist,assumreal),x,y,i,j,xdim,1,0,0,mlid,0,size()-1,assumreal);
    res = (double) temp;
}

//phantomx
void MercerKernel::denseintK(double &res, const SparseVector<gentype> &x, const SparseVector<gentype> &y, const vecInfo &xinfo, const vecInfo &yinfo, double bias, int i, int j, int xdim, int xconsist, int mlid, int assumreal) const
{
    NiceAssert( !isprod );

    // x now appropriately constructed as required

    gentype xyprod;
    gentype yxprod;

    if ( assumreal )
    {
        xyprod.force_double();
        yxprod.force_double();
    }

    innerProductDiverted(xyprod,x,y,xconsist,assumreal);
    innerProductDivertedRevConj(yxprod,xyprod,x,y,xconsist,assumreal);

    xyprod += bias;
    yxprod += bias;

    // NB: second last argument x to indicate dense integral
    gentype temp;
    K2i(temp,xyprod,yxprod,xinfo,yinfo,getmnorm(xinfo,x,2,xconsist,assumreal),getmnorm(yinfo,y,2,xconsist,assumreal),x,y,i,j,xdim,2,0,0,mlid,0,size()-1,assumreal);
    res = (double) temp;
}

















// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================

double  MercerKernel::K0(double bias, const gentype **pxyprod,
                         int xdim, int xconsist, int xresmode, int mlid, int assumreal) const
{
    return yyyK0(bias,pxyprod,xdim,xconsist,xresmode,mlid,assumreal,0);
}

double  MercerKernel::K1(const SparseVector<gentype> &x,
                         const vecInfo &xinfo,
                         double bias,
                         const gentype **pxyprod,
                         int i,
                         int xdim, int xconsist, int resmode, int mlid,
                         const double *xy, int assumreal) const
{
    return yyyK1(x,xinfo,bias,pxyprod,i,xdim,xconsist,resmode,mlid,xy,assumreal,0);
}

double  MercerKernel::K2(const SparseVector<gentype> &x, const SparseVector<gentype> &y,
                         const vecInfo &xinfo, const vecInfo &yinfo,
                         double bias,
                         const gentype **pxyprod,
                         int i, int j,
                         int xdim, int xconsist, int resmode, int mlid, 
                         const double *xy00, const double *xy10, const double *xy11, int assumreal) const
{
    return yyyK2(x,y,xinfo,yinfo,bias,pxyprod,i,j,xdim,xconsist,resmode,mlid,xy00,xy10,xy11,assumreal,0);
}

double  MercerKernel::K2x2(const SparseVector<gentype> &x,  const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                           const vecInfo &xinfo, const vecInfo &xainfo, const vecInfo &xbinfo,
                           double bias,
                           int i, int ia, int ib,
                           int xdim, int xconsist, int resmode, int mlid,
                           const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22,
                           int assumreal) const
{
//    (void) xy21;
//
//    double resa;
//    double resb;
//
//    K2(resa,x,xa,xinfo,xainfo,bias,nullptr,i,ia,xdim,xconsist,resmode,mlid,xy00,xy10,xy11,assumreal);
//    K2(resb,x,xb,xinfo,xbinfo,bias,nullptr,i,ib,xdim,xconsist,resmode,mlid,xy00,xy20,xy22,assumreal);
//
//    return res = resa*resb;
    return yyyK2x2(x,xa,xb,xinfo,xainfo,xbinfo,bias,i,ia,ib,xdim,xconsist,resmode,mlid,xy00,xy10,xy11,xy20,xy21,xy22,assumreal,0);
}


double  MercerKernel::K3(const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, 
                         const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, 
                         double bias, const gentype **pxyprod,
                         int ia, int ib, int ic, 
                         int xdim, int xconsist, int xresmode, int mlid, 
                         const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, int assumreal) const
{
    return yyyK3(xa,xb,xc,xainfo,xbinfo,xcinfo,bias,pxyprod,ia,ib,ic,xdim,xconsist,xresmode,mlid,xy00,xy10,xy11,xy20,xy21,xy22,assumreal,0);
}

double  MercerKernel::K4(const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                         const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
                         double bias, const gentype **pxyprod,
                         int ia, int ib, int ic, int id,
                         int xdim, int xconsist, int xresmode, int mlid, 
                         const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, const double *xy30, const double *xy31, const double *xy32, const double *xy33, int assumreal) const
{
    return yyyK4(xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,pxyprod,ia,ib,ic,id,xdim,xconsist,xresmode,mlid,xy00,xy10,xy11,xy20,xy21,xy22,xy30,xy31,xy32,xy33,assumreal,0);
}

double  MercerKernel::Km(int m,
                         Vector<const SparseVector<gentype> *> &x,
                         Vector<const vecInfo *> &xinfo,
                         double bias,
                         Vector<int> &i,
                         const gentype **pxyprod,
                         int xdim, int xconsist, int resmode, int mlid, 
                         const Matrix<double> *xy, int assumreal) const
{
    return yyyKm(m,x,xinfo,bias,i,pxyprod,xdim,xconsist,resmode,mlid,xy,assumreal,0);
}

int MercerKernel::phim(int m, Vector<gentype> &res, const SparseVector<gentype> &xa, const vecInfo &xainfo, int ia, int allowfinite, int xdim, int xconsist, int assumreal) const
{
    return yyyphim(m,res,xa,xainfo,ia,allowfinite,xdim,xconsist,assumreal);
}

// ===================================================================================

gentype &MercerKernel::K0(gentype &res,
                          const gentype &bias,
                          const gentype **pxyprod,
                          int xdim, int xconsist, int xresmode, int mlid, int assumreal) const
{
    return yyyK0(res,bias,pxyprod,xdim,xconsist,xresmode,mlid,assumreal,0);
}

gentype &MercerKernel::K1(gentype &res,
                          const SparseVector<gentype> &x, 
                          const vecInfo &xinfo, 
                          const gentype &bias,
                          const gentype **pxyprod,
                          int i, 
                          int xdim, int xconsist, int resmode, int mlid, 
                          const double *xy, int assumreal) const
{
    return yyyK1(res,x,xinfo,bias,pxyprod,i,xdim,xconsist,resmode,mlid,xy,assumreal,0);
}

gentype &MercerKernel::K2(gentype &res,
                          const SparseVector<gentype> &x, const SparseVector<gentype> &y,
                          const vecInfo &xinfo, const vecInfo &yinfo,
                          const gentype &bias,
                          const gentype **pxyprod,
                          int i, int j,
                          int xdim, int xconsist, int resmode, int mlid, 
                          const double *xy00, const double *xy10, const double *xy11, int assumreal) const
{
    return yyyK2(res,x,y,xinfo,yinfo,bias,pxyprod,i,j,xdim,xconsist,resmode,mlid,xy00,xy10,xy11,assumreal,0);
}

gentype &MercerKernel::K2x2(gentype &res,
                            const SparseVector<gentype> &x,  const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                            const vecInfo &xinfo, const vecInfo &xainfo, const vecInfo &xbinfo,
                            const gentype &bias,
                            int i, int ia, int ib,
                            int xdim, int xconsist, int resmode, int mlid,
                            const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22,
                            int assumreal) const
{
//    (void) xy21;
//
//    gentype resa;
//    gentype resb;
//
//    K2(resa,x,xa,xinfo,xainfo,bias,nullptr,i,ia,xdim,xconsist,resmode,mlid,xy00,xy10,xy11,assumreal);
//    K2(resb,x,xb,xinfo,xbinfo,bias,nullptr,i,ib,xdim,xconsist,resmode,mlid,xy00,xy20,xy22,assumreal);
//
//    return res = resa*resb;
    return yyyK2x2(res,x,xa,xb,xinfo,xainfo,xbinfo,bias,i,ia,ib,xdim,xconsist,resmode,mlid,xy00,xy10,xy11,xy20,xy21,xy22,assumreal,0);
}

gentype &MercerKernel::K3(gentype &res,
                          const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, 
                          const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, 
                          const gentype &bias,
                          const gentype **pxyprod,
                          int ia, int ib, int ic, 
                          int xdim, int xconsist, int xresmode, int mlid, 
                          const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, int assumreal) const
{
    return yyyK3(res,xa,xb,xc,xainfo,xbinfo,xcinfo,bias,pxyprod,ia,ib,ic,xdim,xconsist,xresmode,mlid,xy00,xy10,xy11,xy20,xy21,xy22,assumreal,0);
}

gentype &MercerKernel::K4(gentype &res,
                          const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                          const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
                          const gentype &bias,
                          const gentype **pxyprod,
                          int ia, int ib, int ic, int id, 
                          int xdim, int xconsist, int xresmode, int mlid, 
                          const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, const double *xy30, const double *xy31, const double *xy32, const double *xy33, int assumreal) const
{
    return yyyK4(res,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,pxyprod,ia,ib,ic,id,xdim,xconsist,xresmode,mlid,xy00,xy10,xy11,xy20,xy21,xy22,xy30,xy31,xy32,xy33,assumreal,0);
}

gentype &MercerKernel::Km(int m, gentype &res,
                          Vector<const SparseVector<gentype> *> &x,
                          Vector<const vecInfo *> &xinfo,
                          const gentype &bias,
                          Vector<int> &i,
                          const gentype **pxyprod, int xdim, int xconsist, int resmode, int mlid, 
                          const Matrix<double> *xy, int assumreal) const
{
    return yyyKm(m,res,x,xinfo,bias,i,pxyprod,xdim,xconsist,resmode,mlid,xy,assumreal,0);
}

int MercerKernel::phim(int m, Vector<double>  &res, const SparseVector<gentype> &xa, const vecInfo &xainfo, int ia, int allowfinite, int xdim, int xconsist, int assumreal) const
{
    return yyyphim(m,res,xa,xainfo,ia,allowfinite,xdim,xconsist,assumreal);
}

double MercerKernel::K0ip(double bias, const gentype **pxyprod,
                          int xdim, int xconsist, int mlid, int assumreal) const
{
    return yyyK0(bias,pxyprod,xdim,xconsist,0,mlid,assumreal,1);
}

double MercerKernel::K1ip(const SparseVector<gentype> &x,
                          const vecInfo &xinfo,
                          double bias, const gentype **pxyprod,
                          int i,
                          int xdim, int xconsist, int mlid, int assumreal) const
{
    return yyyK1(x,xinfo,bias,pxyprod,i,xdim,xconsist,0,mlid,nullptr,assumreal,1);
}

double MercerKernel::K2ip(const SparseVector<gentype> &x, const SparseVector<gentype> &y,
                          const vecInfo &xinfo, const vecInfo &yinfo,
                          double bias, const gentype **pxyprod,
                          int i, int j,
                          int xdim, int xconsist, int mlid, int assumreal) const
{
    return yyyK2(x,y,xinfo,yinfo,bias,pxyprod,i,j,xdim,xconsist,0,mlid,nullptr,nullptr,nullptr,assumreal,1);
}

double MercerKernel::K3ip(const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc,
                          const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo,
                          double bias, const gentype **pxyprod,
                          int ia, int ib, int ic,
                          int xdim, int xconsist, int mlid, int assumreal) const
{
    return yyyK3(xa,xb,xc,xainfo,xbinfo,xcinfo,bias,pxyprod,ia,ib,ic,xdim,xconsist,0,mlid,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,assumreal,1);
}

double MercerKernel::K4ip(const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                          const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
                          double bias, const gentype **pxyprod,
                          int ia, int ib, int ic, int id,
                          int xdim, int xconsist, int mlid, int assumreal) const
{
    return yyyK4(xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,bias,pxyprod,ia,ib,ic,id,xdim,xconsist,0,mlid,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,assumreal,1);
}

double MercerKernel::Kmip(int m,
                          Vector<const SparseVector<gentype> *> &x,
                          Vector<const vecInfo *> &xinfo,
                          Vector<int> &i,
                          double bias, const gentype **pxyprod,
                          int xdim, int xconsist, int mlid, int assumreal) const
{
    return yyyKm(m,x,xinfo,bias,i,pxyprod,xdim,xconsist,0,mlid,nullptr,assumreal,1);
}










void castvectoreal(Vector<double> &dest, const Vector<gentype> &src);
void castvectoreal(Vector<double> &dest, const Vector<gentype> &src)
{
    int dim = src.size();

    dest.resize(dim);

    if ( dim )
    {
        int i;

        for ( i = 0 ; i < dim ; ++i )
        {
            dest("&",i) = (double) src(i);
        }
    }

    return;
}

double  MercerKernel::yyyK0(double bias,
                            const gentype **pxyprod,
                            int xdim, int xconsist, int xresmode, int mlid, int assumreal, int justcalcip) const
{
    double res = 0;

    return yyyaK0(res,bias,pxyprod,xdim,xconsist,xresmode,mlid,assumreal,justcalcip);
}

gentype &MercerKernel::yyyK0(gentype &res,
                             const gentype &bias,
                             const gentype **pxyprod,
                             int xdim, int xconsist, int xresmode, int mlid, int assumreal, int justcalcip) const
{
    return yyyaK0(res,bias,pxyprod,xdim,xconsist,xresmode,mlid,assumreal,justcalcip);
}

double  MercerKernel::yyyK1(const SparseVector<gentype> &xa,
                            const vecInfo &xainfo,
                            double bias,
                            const gentype **pxyprod,
                            int ia,
                            int xdim, int xconsist, int resmode, int mlid,
                            const double *xy00, int assumreal, int justcalcip) const
{
    double res = 0;

    int xaignorefarfarfar = 0;
    int xagradordaddR     = 0;

    if ( !sizeLinConstr() )
    {
        int xaignorefarfar = 0;
        int xagradordadd = 0;

        yyyaK1(res,xa,xainfo,xaignorefarfar,xaignorefarfarfar,xagradordadd,xagradordaddR,bias,pxyprod,ia,xdim,xconsist,resmode,mlid,xy00,assumreal,justcalcip);
    }

    else
    {
        static const SparseVector<gentype> dummy;

        int xafarfarpresent = xa.isf2offindpresent() ? 1 : 0;
        int xaind6present   = xa.isf4indpresent(6) && !(xa.f4(6).isValNull());
        int xagradOrder     = xaind6present ? ( (int) xa.f4(6) ) : ( xafarfarpresent ? 1 : 0 );

        const SparseVector<gentype> &xaffg = xafarfarpresent ? xa.f2() : dummy;
        retVector<gentype> tmpvff;
        Vector<double> xaff;
        castvectoreal(xaff,xaffg(tmpvff));
        int xaffdim = xaff.size() ? xaff.size() : ((int) std::round(pow(xdim,xagradOrder)));

        NiceAssert( !xagradOrder || xafarfarpresent ); // for this case the result must be a double, so require projected gradient

        int q;

        int xaignorefarfar = 1;
        int xagradordadd   = 0;

        gentype gra;
        gentype gbias(bias);
        double tmpr;

        Matrix<double> Aqalt;

        retVector<double> tmpva,tmpvb,tmpvc,tmpvd,tmpve;

        for ( q = 0 ; q < sizeLinConstr() ; ++q )
        {
            // Find relevant gradient

            xagradordadd = linGradOrd(q);

            yyyaK1(gra,xa,xainfo,xaignorefarfar,xaignorefarfarfar,xagradordadd,xagradordaddR,gbias,pxyprod,ia,xdim,xconsist,resmode,mlid,xy00,assumreal,justcalcip);

            // Cast result as vector of double

            const Vector<double> &grav = (const Vector<double> &) gra;

            // Get scaling matrix and record its size

            const Matrix<double> &Aqraw = linGradScal(q);

            const Matrix<double> *Aqptr = &Aqraw;

            // If we are evaluating a gradient we need to convert A to the gradient form!

            if ( xagradOrder && xafarfarpresent )
            {
                // Directional gradient, so need to increase gradient order and project down.
                // e \otimes A -> [ e \otimes A_{0:} ]

                Aqalt.resize(Aqraw.numRows(),xaffdim*(Aqraw.numCols()));

                kronprod(Aqalt("&",0,tmpva,tmpvb),xaff,Aqraw(0,tmpvd,tmpve));

                Aqptr = &Aqalt;
            }

            const Matrix<double> &Aq = *Aqptr;

            NiceAssert( Aq.numRows() == 1           );
            NiceAssert( Aq.numCols() == grav.size() );

            // Calculate the result

            if ( !q )
            {
                res  = twoProduct(tmpr,Aq(0,tmpva,tmpvb),grav);
            }

            else
            {
                res += twoProduct(tmpr,Aq(0,tmpva,tmpvb),grav);
            }
        }
    }

    return res;
}

gentype &MercerKernel::yyyK1(gentype &res,
                             const SparseVector<gentype> &xa,
                             const vecInfo &xainfo,
                             const gentype &bias,
                             const gentype **pxyprod,
                             int ia,
                             int xdim, int xconsist, int resmode, int mlid,
                             const double *xy00, int assumreal, int justcalcip) const
{
    int xaignorefarfarfar = 0;
    int xagradordaddR     = 0;

    if ( !sizeLinConstr() )
    {
        int xaignorefarfar = 0;
        int xagradordadd   = 0;

        yyyaK1(res,xa,xainfo,xaignorefarfar,xaignorefarfarfar,xagradordadd,xagradordaddR,bias,pxyprod,ia,xdim,xconsist,resmode,mlid,xy00,assumreal,justcalcip);
    }

    else
    {
        static const SparseVector<gentype> dummy;

        int xafarfarpresent = xa.isf2offindpresent() ? 1 : 0;
        int xaind6present   = xa.isf4indpresent(6) && !(xa.f4(6).isValNull());
        int xagradOrder     = xaind6present ? ( (int) xa.f4(6) ) : ( xafarfarpresent ? 1 : 0 );

        const SparseVector<gentype> &xaffg = xafarfarpresent ? xa.f2() : dummy;
        retVector<gentype> tmpvff;
        Vector<double> xaff;
        castvectoreal(xaff,xaffg(tmpvff));
        int xaffdim = xaff.size() ? xaff.size() : ((int) std::round(pow(xdim,xagradOrder)));
        xaff.resize(xaffdim);

        int j,k,q;

        int xaignorefarfar = 1;
        int xagradordadd   = 0;

        gentype gra;

        Matrix<double> Aqalt;

        retVector<double> tmpva,tmpvb,tmpvc,tmpvd,tmpve;

        for ( q = 0 ; q < sizeLinConstr() ; ++q )
        {
            // Find relevant gradient

            xagradordadd = linGradOrd(q);

            yyyaK1(gra,xa,xainfo,xaignorefarfar,xaignorefarfarfar,xagradordadd,xagradordaddR,bias,pxyprod,ia,xdim,xconsist,resmode,mlid,xy00,assumreal,justcalcip);

            // Cast result as vector of double

            const Vector<double> &grav = (const Vector<double> &) gra;

            // Get scaling matrix and record its size

            const Matrix<double> &Aqraw = linGradScal(q);

            const Matrix<double> *Aqptr = &Aqraw;

            // If we are evaluating a gradient we need to convert A to the gradient form!

            if ( xagradOrder && xafarfarpresent )
            {
                // Directional gradient, so need to increase gradient order and project down.
                // Like below, then take a weighted sum

                Aqalt.resize(Aqraw.numRows(),xaffdim*(Aqraw.numCols()));

                for ( j = 0 ; j < Aqraw.numRows() ; ++j )
                {
                    kronprod(Aqalt("&",j,tmpva,tmpvb),xaff,Aqraw(j,tmpvd,tmpve));
                }

                Aqptr = &Aqalt;
            }

            else if ( xagradOrder && !xafarfarpresent )
            {
                // Directional gradient, so need to increase gradient order and project down.
                // E \otimes A -> [ E_{0:} \otimes A_{0:} ]
                //                [ E_{0:} \otimes A_{1:} ]
                //                [          ...          ]
                //                [ E_{1:} \otimes A_{0:} ]
                //                [ E_{1:} \otimes A_{1:} ]
                //                [          ...          ]
                //                [          ...          ]
                // where E_{0:} = [ 1 0 0 0 ... ]
                //       E_{1:} = [ 0 1 0 0 ... ]
                //             ...

                Aqalt.resize(xaffdim*(Aqraw.numRows()),xaffdim*(Aqraw.numCols()));

                for ( k = 0 ; k < xaffdim ; ++k )
                {
                    xaff = 0.0;
                    xaff("&",k) = 1.0;

                    for ( j = 0 ; j < Aqraw.numRows() ; ++j )
                    {
                        // We are assuming non-sparse format here for simplicity!

                        kronprod(Aqalt("&",j+(k*(Aqraw.numRows())),tmpva,tmpvb),xaff,Aqraw(j,tmpvd,tmpve));
                    }
                }

                Aqptr = &Aqalt;
            }

            const Matrix<double> &Aq = *Aqptr;

            NiceAssert( Aq.numCols() == grav.size() );

            // Calculate the result

            if ( !q )
            {
                res  = (Aq*grav);
            }

            else
            {
                res += (Aq*grav);
            }
        }
    }

    return res;
}

double MercerKernel::LL2fast(const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                             const vecInfo &xainfo, const vecInfo &xbinfo,
                             double bias,
                             const gentype **pxyprod,
                             int ia, int ib,
                             int xdim, int xconsist, int assumreal,
                             const double *xy00, const double *xy10, const double *xy11) const
{
    double res = 0;

    double xyprod = 0.0;

    double aaprod = 0.0;
    double bbprod = 0.0;

    double diffis = 0.0;

    if ( ( ia == ib ) && !isMagTerm(0) )
    {
        // infer that xa == xb which makes things particularly simple

        if ( xy00 && xy11 )
        {
            aaprod = (*xy00)+bias;
            bbprod = aaprod;
            xyprod = aaprod;
        }

        else if ( xy10 )
        {
            aaprod = (*xy10)+bias;
            bbprod = aaprod;
            xyprod = aaprod;
        }

        else if ( pxyprod && pxyprod[0] )
        {
            aaprod = (*pxyprod[0]);
            bbprod = aaprod;
            xyprod = aaprod;
        }

        else if ( isNormalised(0) || needsNorm(0) || needsInner(0,2) )
        {
            aaprod = getmnorm(xainfo,xa,2,xconsist,assumreal)+bias; // this should be cached anyhow
            bbprod = aaprod;
            xyprod = aaprod;
        }

        // diffis == 0 in any case as xa == xb

        //KKpro(res,xyprod,diffis,ixy,0,1,xdim,2,logres,xyvals);
        {
            retVector<gentype> tmpva;

            KKprosinglediffiszero(res,xyprod,ia,ib,aaprod,bbprod,cType(0),cWeight(0),dRealConstants(0)(1,1,dRealConstants(0).size()-1,tmpva),dIntConstants(0));
        }
    }

    else
    {
        if ( xy00 && xy11 )
        {
            aaprod = (*xy00)+bias;
            bbprod = (*xy11)+bias;
        }

        else if ( isNormalised(0) || needsNorm(0) || needsDiff(0) )
        {
            aaprod = getmnorm(xainfo,xa,2,xconsist,assumreal)+bias;
            bbprod = getmnorm(xbinfo,xb,2,xconsist,assumreal)+bias;
        }

        if ( xy10 )
        {
            xyprod = (*xy10)+bias;
        }

        else if ( pxyprod && pxyprod[0] )
        {
            xyprod = (*pxyprod[0]);
        }

        else if ( needsInner(0,2) )
        {
            xyprod = getTwoProd(xa,xb,0,0,0,xconsist,assumreal)+bias;
        }

        if ( pxyprod && pxyprod[1] )
        {
            diffis = (*pxyprod[1]);
        }

        else if ( needsDiff(0) )
        {
            diffis = diff2norm(xyprod,aaprod,bbprod);
        }

        //KKpro(res,xyprod,diffis,ixy,0,1,xdim,2,logres,xyvals);
        {
            double xyvals[2] = { aaprod,bbprod };
            int ixy[2] = { ia,ib };

            retVector<gentype> tmpva;

            double logres = 0;
            int logresvalid = 0;

            KKprosingle(res,xyprod,diffis,ixy,xdim,2,logres,xyvals,cType(0),logresvalid,cWeight(0),dRealConstants(0)(1,1,dRealConstants(0).size()-1,tmpva),dIntConstants(0),isMagTerm(0));
        }
    }

    return res;
}

double  MercerKernel::yyyK2(const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                            const vecInfo &xainfo, const vecInfo &xbinfo,
                            double bias,
                            const gentype **pxyprod,
                            int ia, int ib,
                            int xdim, int xconsist, int resmode, int mlid,
                            const double *xy00, const double *xy10, const double *xy11, int assumreal, int justcalcip) const
{
    double res = 0;

    int xaignorefarfarfar = 0;
    int xagradordaddR     = 0;

    int xbignorefarfarfar = 0;
    int xbgradordaddR     = 0;

    if ( !sizeLinConstr() )
    {
        if ( !resmode &&
             !isprod &&
             !justcalcip &&
             ( size() == 1 ) &&
             isFastKernelSum() &&
             !isfullnorm &&
             !isNormalised(0) &&
             !numMulSplits() &&
             !(combinedOverwriteSrc.size()) &&
             ( ( !isShifted() && !isScaled() ) || isLeftRightPlain() ) &&
             !isIndex() &&
             ( xainfo.xusize() == 1 ) &&
             ( xbinfo.xusize() == 1 ) &&
             !xainfo.xiseqn() &&
             !xbinfo.xiseqn() &&
             xa.isnofaroffindpresent() &&
             xb.isnofaroffindpresent() )
        {
            res = LL2fast(xa,xb,xainfo,xbinfo,bias,pxyprod,ia,ib,xdim,xconsist,assumreal,xy00,xy10,xy11);
        }

        else
        {
            int xaignorefarfar = 0;
            int xagradordadd   = 0;

            int xbignorefarfar = 0;
            int xbgradordadd   = 0;

            yyyaK2(res,xa,xb,xainfo,xbinfo,xaignorefarfar,xbignorefarfar,xaignorefarfarfar,xbignorefarfarfar,xagradordadd,xbgradordadd,xagradordaddR,xbgradordaddR,bias,pxyprod,ia,ib,xdim,xconsist,resmode,mlid,xy00,xy10,xy11,assumreal,justcalcip);
        }
    }

    else
    {
        static const SparseVector<gentype> dummy;

        int xafarfarpresent = xa.isf2offindpresent() ? 1 : 0;
        int xaind6present   = xa.isf4indpresent(6) && !(xa.f4(6).isValNull());
        int xagradOrder     = xaind6present ? ( (int) xa.f4(6) ) : ( xafarfarpresent ? 1 : 0 );

        int xbfarfarpresent = xb.isf2offindpresent() ? 1 : 0;
        int xbind6present   = xb.isf4indpresent(6) && !(xb.f4(6).isValNull());
        int xbgradOrder     = xbind6present ? ( (int) xb.f4(6) ) : ( xbfarfarpresent ? 1 : 0 );

        const SparseVector<gentype> &xaffg = xafarfarpresent ? xa.f2() : dummy;
        retVector<gentype> tmpvff;
        Vector<double> xaff;
        castvectoreal(xaff,xaffg(tmpvff));
        int xaffdim = xaff.size() ? xaff.size() : ((int) std::round(pow(xdim,xagradOrder)));

        const SparseVector<gentype> &xbffg = xbfarfarpresent ? xb.f2() : dummy;
        retVector<gentype> tmpvgg;
        Vector<double> xbff;
        castvectoreal(xbff,xbffg(tmpvgg));
        int xbffdim = xbff.size() ? xbff.size() : ((int) std::round(pow(xdim,xbgradOrder)));

        NiceAssert( !xagradOrder || xafarfarpresent );
        NiceAssert( !xbgradOrder || xbfarfarpresent );

        int q,r;

        int xaignorefarfar = 1;
        int xagradordadd   = 0;

        int xbignorefarfar = 1;
        int xbgradordadd   = 0;

        gentype gbias(bias);
        gentype gra;
        double tempres;

        Matrix<double> Aqalt;
        Matrix<double> Aralt;

        retVector<double> tmpva,tmpvb,tmpvc,tmpvd,tmpve;

        Vector<double> gA;

        for ( q = 0 ; q < sizeLinConstr() ; ++q )
        {
            for ( r = 0 ; r < sizeLinConstr() ; ++r )
            {
                // Find relevant gradient

                xagradordadd = linGradOrd(q);
                xbgradordadd = linGradOrd(r);

                yyyaK2(gra,xa,xb,xainfo,xbinfo,xaignorefarfar,xbignorefarfar,xaignorefarfarfar,xbignorefarfarfar,xagradordadd,xbgradordadd,xagradordaddR,xbgradordaddR,gbias,pxyprod,ia,ib,xdim,xconsist,resmode,mlid,xy00,xy10,xy11,assumreal,justcalcip);

                // Cast result as vector of double

                const Matrix<double> &gram = (const Matrix<double> &) gra;

                // Get scaling matrix and record its size

                const Matrix<double> &Aqraw = linGradScal(q);
                const Matrix<double> &Arraw = linGradScal(r);

                const Matrix<double> *Aqptr = &Aqraw;
                const Matrix<double> *Arptr = &Arraw;

                if ( xagradOrder && xafarfarpresent )
                {
                    // Directional gradient, so need to increase gradient order and project down.
                    // e \otimes A -> [ e \otimes A_{0:} ]

                    Aqalt.resize(Aqraw.numRows(),xaffdim*(Aqraw.numCols()));

                    kronprod(Aqalt("&",0,tmpva,tmpvb),xaff,Aqraw(0,tmpvd,tmpve));

                    Aqptr = &Aqalt;
                }

                if ( xagradOrder && xafarfarpresent )
                {
                    // Directional gradient, so need to increase gradient order and project down.
                    // e \otimes A -> [ e \otimes A_{0:} ]

                    Aralt.resize(Arraw.numRows(),xbffdim*(Arraw.numCols()));

                    kronprod(Aralt("&",0,tmpva,tmpvb),xbff,Arraw(0,tmpvd,tmpve));

                    Arptr = &Aralt;
                }

                const Matrix<double> &Aq = *Aqptr;
                const Matrix<double> &Ar = *Arptr;

                NiceAssert( Aq.numRows() == 1              );
                NiceAssert( Aq.numCols() == gram.numRows() );

                NiceAssert( Ar.numRows() == 1              );
                NiceAssert( Ar.numCols() == gram.numCols() );

                // Calculate the result

                gA = gram*Ar(0,tmpva,tmpvb);

                if ( !q && !r )
                {
                    res  = twoProduct(tempres,Aq(0,tmpva,tmpvb),gA);
                }

                else
                {
                    res += twoProduct(tempres,Aq(0,tmpva,tmpvb),gA);
                }
            }
        }
    }

    return res;
}

gentype &MercerKernel::yyyK2(gentype &res,
                             const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                             const vecInfo &xainfo, const vecInfo &xbinfo,
                             const gentype &bias,
                             const gentype **pxyprod,
                             int ia, int ib,
                             int xdim, int xconsist, int resmode, int mlid,
                             const double *xy00, const double *xy10, const double *xy11, int assumreal, int justcalcip) const
{
    int xaignorefarfarfar = 0;
    int xagradordaddR     = 0;

    int xbignorefarfarfar = 0;
    int xbgradordaddR     = 0;

    if ( !sizeLinConstr() )
    {
        int xaignorefarfar = 0;
        int xagradordadd   = 0;

        int xbignorefarfar = 0;
        int xbgradordadd   = 0;

        yyyaK2(res,xa,xb,xainfo,xbinfo,xaignorefarfar,xbignorefarfar,xaignorefarfarfar,xbignorefarfarfar,xagradordadd,xbgradordadd,xagradordaddR,xbgradordaddR,bias,pxyprod,ia,ib,xdim,xconsist,resmode,mlid,xy00,xy10,xy11,assumreal,justcalcip);
    }

    else
    {
        static const SparseVector<gentype> dummy;

        int xafarfarpresent = xa.isf2offindpresent() ? 1 : 0;
        int xaind6present   = xa.isf4indpresent(6) && !(xa.f4(6).isValNull());
        int xagradOrder     = xaind6present ? ( (int) xa.f4(6) ) : ( xafarfarpresent ? 1 : 0 );

        int xbfarfarpresent = xb.isf2offindpresent() ? 1 : 0;
        int xbind6present   = xb.isf4indpresent(6) && !(xb.f4(6).isValNull());
        int xbgradOrder     = xbind6present ? ( (int) xb.f4(6) ) : ( xbfarfarpresent ? 1 : 0 );

        const SparseVector<gentype> &xaffg = xafarfarpresent ? xa.f2() : dummy;
        retVector<gentype> tmpvff;
        Vector<double> xaff;
        castvectoreal(xaff,xaffg(tmpvff));
        int xaffdim = xaff.size() ? xaff.size() : ((int) std::round(pow(xdim,xagradOrder)));
        xaff.resize(xaffdim);

        const SparseVector<gentype> &xbffg = xbfarfarpresent ? xb.f2() : dummy;
        retVector<gentype> tmpvgg;
        Vector<double> xbff;
        castvectoreal(xbff,xbffg(tmpvgg));
        int xbffdim = xbff.size() ? xbff.size() : ((int) std::round(pow(xdim,xbgradOrder)));
        xbff.resize(xbffdim);

        int j,k,q,r;

        int xaignorefarfar = 1;
        int xagradordadd   = 0;

        int xbignorefarfar = 1;
        int xbgradordadd   = 0;

        gentype gra;

        Matrix<double> Aqalt;
        Matrix<double> Aralt;

        retVector<double> tmpva,tmpvb,tmpvc,tmpvd,tmpve;

        for ( q = 0 ; q < sizeLinConstr() ; ++q )
        {
            for ( r = 0 ; r < sizeLinConstr() ; ++r )
            {
                // Find relevant gradient

                xagradordadd = linGradOrd(q);
                xbgradordadd = linGradOrd(r);

                yyyaK2(gra,xa,xb,xainfo,xbinfo,xaignorefarfar,xbignorefarfar,xaignorefarfarfar,xbignorefarfarfar,xagradordadd,xbgradordadd,xagradordaddR,xbgradordaddR,bias,pxyprod,ia,ib,xdim,xconsist,resmode,mlid,xy00,xy10,xy11,assumreal,justcalcip);

                // Cast result as matrix of double

                const Matrix<double> &gram = (const Matrix<double> &) gra;

                // Get scaling matrix and record its size

                const Matrix<double> &Aqraw = linGradScal(q);

                const Matrix<double> &Artspraw = linGradScalTsp(r);

                const Matrix<double> *Aqptr = &Aqraw;
                const Matrix<double> *Artspptr = &Artspraw;

                if ( xagradOrder && xafarfarpresent )
                {
                    // Directional gradient, so need to increase gradient order and project down.
                    // e \otimes A -> [ e \otimes A_{0:} ]
                    //                [ e \otimes A_{1:} ]
                    //                [     ...          ]

                    Aqalt.resize(Aqraw.numRows(),xaffdim*(Aqraw.numCols()));

                    for ( j = 0 ; j < Aqraw.numRows() ; ++j )
                    {
                        kronprod(Aqalt("&",j,tmpva,tmpvb),xaff,Aqraw(j,tmpvd,tmpve));
                    }

                    Aqptr = &Aqalt;
                }

                else if ( xagradOrder && !xafarfarpresent )
                {
                    // Directional gradient, so need to increase gradient order and project down.
                    // E \otimes A -> [ e_{0:} \otimes A_{0:} ]
                    //                [ e_{0:} \otimes A_{1:} ]
                    //                [          ...          ]
                    //                [ e_{1:} \otimes A_{0:} ]
                    //                [ e_{1:} \otimes A_{1:} ]
                    //                [          ...          ]
                    //                [          ...          ]

                    Aqalt.resize(xaffdim*(Aqraw.numRows()),xaffdim*(Aqraw.numCols()));

                    for ( k = 0 ; k < xaffdim ; ++k )
                    {
                        xaff = 0.0;
                        xaff("&",k) = 1.0;

                        for ( j = 0 ; j < Aqraw.numRows() ; ++j )
                        {
                            // We are assuming non-sparse format here for simplicity!

                            kronprod(Aqalt("&",j+(k*(Aqraw.numRows())),tmpva,tmpvb),xaff,Aqraw(j,tmpvd,tmpve));
                        }
                    }

                    Aqptr = &Aqalt;
                }

                // Fro Ar we need to transform then transpose.

                if ( xbgradOrder && xbfarfarpresent )
                {
                    const Matrix<double> &Arraw = linGradScal(r);

                    // Directional gradient, so need to increase gradient order and project down.
                    // e \otimes A -> [ e \otimes A_{0:} ]
                    //                [ e \otimes A_{1:} ]
                    //                [     ...          ]

                    Aralt.resize(Arraw.numRows(),xbffdim*(Arraw.numCols()));

                    for ( j = 0 ; j < Arraw.numRows() ; ++j )
                    {
                        kronprod(Aralt("&",j,tmpva,tmpvb),xbff,Arraw(j,tmpvd,tmpve));
                    }

                    Aralt.transpose();
                    Artspptr = &Aralt;
                }

                else if ( xbgradOrder && !xbfarfarpresent )
                {
                    const Matrix<double> &Arraw = linGradScal(r);

                    // Directional gradient, so need to increase gradient order and project down.
                    // E \otimes A -> [ e_{0:} \otimes A_{0:} ]
                    //                [ e_{0:} \otimes A_{1:} ]
                    //                [          ...          ]
                    //                [ e_{1:} \otimes A_{0:} ]
                    //                [ e_{1:} \otimes A_{1:} ]
                    //                [          ...          ]
                    //                [          ...          ]

                    Aralt.resize(xbffdim*(Arraw.numRows()),xbffdim*(Arraw.numCols()));

                    for ( k = 0 ; k < xbffdim ; ++k )
                    {
                        xbff = 0.0;
                        xbff("&",k) = 1.0;

                        for ( j = 0 ; j < Arraw.numRows() ; ++j )
                        {
                            // We are assuming non-sparse format here for simplicity!

                            kronprod(Aralt("&",j+(k*(Arraw.numRows())),tmpva,tmpvb),xbff,Arraw(j,tmpvd,tmpve));
                        }
                    }

                    Aralt.transpose();
                    Artspptr = &Aralt;
                }

                const Matrix<double> &Aq = *Aqptr;
                const Matrix<double> &Artsp = *Artspptr;

                NiceAssert( Aq.numCols()    == gram.numRows() );
                NiceAssert( Artsp.numRows() == gram.numCols() );

                // Calculate the result
                //
                // res = Aq.gram.Ar'

                if ( !q && !r )
                {
                    res  = (Aq*(gram*Artsp));
                }

                else
                {
                    res += (Aq*(gram*Artsp));
                }
            }
        }
    }

    return res;
}

double  MercerKernel::yyyK2x2(const SparseVector<gentype> &x, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                              const vecInfo &xinfo, const vecInfo &xainfo, const vecInfo &xbinfo,
                              double bias,
                              int i, int ia, int ib,
                              int xdim, int xconsist, int resmode, int mlid,
                              const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22,
                              int assumreal, int justcalcip) const
{
    NiceAssert( !sizeLinConstr() )

    int xignorefarfar = 0;
    int xgradordadd   = 0;

    int xaignorefarfar = 0;
    int xagradordadd   = 0;

    int xbignorefarfar = 0;
    int xbgradordadd   = 0;

    int xignorefarfarfar = 0;
    int xgradordaddR     = 0;

    int xaignorefarfarfar = 0;
    int xagradordaddR     = 0;

    int xbignorefarfarfar = 0;
    int xbgradordaddR     = 0;

    double res = 0;

//FIXME: implement linear constraint kernels somehow here
    yyyaK2x2(res,x,xa,xb,xinfo,xainfo,xbinfo,xignorefarfar,xaignorefarfar,xbignorefarfar,xignorefarfarfar,xaignorefarfarfar,xbignorefarfarfar,xgradordadd,xagradordadd,xbgradordadd,xgradordaddR,xagradordaddR,xbgradordaddR,bias,i,ia,ib,xdim,xconsist,resmode,mlid,xy00,xy10,xy11,xy20,xy21,xy22,assumreal,justcalcip);

    return res;
}

gentype &MercerKernel::yyyK2x2(gentype &res,
                               const SparseVector<gentype> &x, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                               const vecInfo &xinfo, const vecInfo &xainfo, const vecInfo &xbinfo,
                               const gentype &bias,
                               int i, int ia, int ib,
                               int xdim, int xconsist, int resmode, int mlid,
                               const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22,
                               int assumreal, int justcalcip) const
{
    NiceAssert( !sizeLinConstr() )

    int xignorefarfar = 0;
    int xgradordadd   = 0;

    int xaignorefarfar = 0;
    int xagradordadd   = 0;

    int xbignorefarfar = 0;
    int xbgradordadd   = 0;

    int xignorefarfarfar = 0;
    int xgradordaddR     = 0;

    int xaignorefarfarfar = 0;
    int xagradordaddR     = 0;

    int xbignorefarfarfar = 0;
    int xbgradordaddR     = 0;

    yyyaK2x2(res,x,xa,xb,xinfo,xainfo,xbinfo,xignorefarfar,xaignorefarfar,xbignorefarfar,xignorefarfarfar,xaignorefarfarfar,xbignorefarfarfar,xgradordadd,xagradordadd,xbgradordadd,xgradordaddR,xagradordaddR,xbgradordaddR,bias,i,ia,ib,xdim,xconsist,resmode,mlid,xy00,xy10,xy11,xy20,xy21,xy22,assumreal,justcalcip);

//FIXME: implement linear constraint kernels somehow here
    return res;
}

// Defining Jidling stuff for higher-orders would require tensorial return types.  These don't exist
// (well, technically they could for A.numRows() == 1, which collapses to a double, but this still
// won't do because the inner kernel part can't return a tensor), so we don't try.

double  MercerKernel::yyyK3(const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc,
                            const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo,
                            double bias,
                            const gentype **pxyprod,
                            int ia, int ib, int ic,
                            int xdim, int xconsist, int xresmode, int mlid,
                            const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, int assumreal, int justcalcip) const
{
    double res = 0;

    int xaignorefarfar = 0;
    int xbignorefarfar = 0;
    int xcignorefarfar = 0;

    int xagradordadd = 0;
    int xbgradordadd = 0;
    int xcgradordadd = 0;

    int xaignorefarfarfar = 0;
    int xbignorefarfarfar = 0;
    int xcignorefarfarfar = 0;

    int xagradordaddR = 0;
    int xbgradordaddR = 0;
    int xcgradordaddR = 0;

    NiceAssert( !sizeLinConstr() );

    yyyaK3(res,xa,xb,xc,xainfo,xbinfo,xcinfo,xaignorefarfar,xbignorefarfar,xcignorefarfar,xaignorefarfarfar,xbignorefarfarfar,xcignorefarfarfar,xagradordadd,xbgradordadd,xcgradordadd,xagradordaddR,xbgradordaddR,xcgradordaddR,bias,pxyprod,ia,ib,ic,xdim,xconsist,xresmode,mlid,xy00,xy10,xy11,xy20,xy21,xy22,assumreal,justcalcip);

    return res;
}

gentype &MercerKernel::yyyK3(gentype &res,
                             const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc,
                             const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo,
                             const gentype &bias,
                             const gentype **pxyprod,
                             int ia, int ib, int ic,
                             int xdim, int xconsist, int xresmode, int mlid,
                             const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, int assumreal, int justcalcip) const
{
    int xaignorefarfar = 0;
    int xbignorefarfar = 0;
    int xcignorefarfar = 0;

    int xagradordadd = 0;
    int xbgradordadd = 0;
    int xcgradordadd = 0;

    int xaignorefarfarfar = 0;
    int xbignorefarfarfar = 0;
    int xcignorefarfarfar = 0;

    int xagradordaddR = 0;
    int xbgradordaddR = 0;
    int xcgradordaddR = 0;

    NiceAssert( !sizeLinConstr() );

    yyyaK3(res,xa,xb,xc,xainfo,xbinfo,xcinfo,xaignorefarfar,xbignorefarfar,xcignorefarfar,xaignorefarfarfar,xbignorefarfarfar,xcignorefarfarfar,xagradordadd,xbgradordadd,xcgradordadd,xagradordaddR,xbgradordaddR,xcgradordaddR,bias,pxyprod,ia,ib,ic,xdim,xconsist,xresmode,mlid,xy00,xy10,xy11,xy20,xy21,xy22,assumreal,justcalcip);

    return res;
}

double  MercerKernel::yyyK4(const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                            const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
                            double bias,
                            const gentype **pxyprod,
                            int ia, int ib, int ic, int id,
                            int xdim, int xconsist, int xresmode, int mlid,
                            const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, const double *xy30, const double *xy31, const double *xy32, const double *xy33, int assumreal, int justcalcip) const
{
    double res = 0;

    int xaignorefarfar = 0;
    int xbignorefarfar = 0;
    int xcignorefarfar = 0;
    int xdignorefarfar = 0;

    int xagradordadd = 0;
    int xbgradordadd = 0;
    int xcgradordadd = 0;
    int xdgradordadd = 0;

    int xaignorefarfarfar = 0;
    int xbignorefarfarfar = 0;
    int xcignorefarfarfar = 0;
    int xdignorefarfarfar = 0;

    int xagradordaddR = 0;
    int xbgradordaddR = 0;
    int xcgradordaddR = 0;
    int xdgradordaddR = 0;

    NiceAssert( !sizeLinConstr() );

    yyyaK4(res,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,xaignorefarfar,xbignorefarfar,xcignorefarfar,xdignorefarfar,xaignorefarfarfar,xbignorefarfarfar,xcignorefarfarfar,xdignorefarfarfar,xagradordadd,xbgradordadd,xcgradordadd,xdgradordadd,xagradordaddR,xbgradordaddR,xcgradordaddR,xdgradordaddR,bias,pxyprod,ia,ib,ic,id,xdim,xconsist,xresmode,mlid,xy00,xy10,xy11,xy20,xy21,xy22,xy30,xy31,xy32,xy33,assumreal,justcalcip);

    return res;
}

gentype &MercerKernel::yyyK4(gentype &res,
                             const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                             const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
                             const gentype &bias,
                             const gentype **pxyprod,
                             int ia, int ib, int ic, int id,
                             int xdim, int xconsist, int xresmode, int mlid,
                             const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, const double *xy30, const double *xy31, const double *xy32, const double *xy33, int assumreal, int justcalcip) const
{
    int xaignorefarfar = 0;
    int xbignorefarfar = 0;
    int xcignorefarfar = 0;
    int xdignorefarfar = 0;

    int xagradordadd = 0;
    int xbgradordadd = 0;
    int xcgradordadd = 0;
    int xdgradordadd = 0;

    int xaignorefarfarfar = 0;
    int xbignorefarfarfar = 0;
    int xcignorefarfarfar = 0;
    int xdignorefarfarfar = 0;

    int xagradordaddR = 0;
    int xbgradordaddR = 0;
    int xcgradordaddR = 0;
    int xdgradordaddR = 0;

    NiceAssert( !sizeLinConstr() );

    yyyaK4(res,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,xaignorefarfar,xbignorefarfar,xcignorefarfar,xdignorefarfar,xaignorefarfarfar,xbignorefarfarfar,xcignorefarfarfar,xdignorefarfarfar,xagradordadd,xbgradordadd,xcgradordadd,xdgradordadd,xagradordaddR,xbgradordaddR,xcgradordaddR,xdgradordaddR,bias,pxyprod,ia,ib,ic,id,xdim,xconsist,xresmode,mlid,xy00,xy10,xy11,xy20,xy21,xy22,xy30,xy31,xy32,xy33,assumreal,justcalcip);

    return res;
}

double  MercerKernel::yyyKm(int m,
                            Vector<const SparseVector<gentype> *> &x,
                            Vector<const vecInfo *> &xinfo,
                            double bias,
                            Vector<int> &i,
                            const gentype **pxyprod, int xdim, int xconsist, int resmode, int mlid,
                            const Matrix<double> *xy, int assumreal, int justcalcip) const
{
    double res = 0;

    if ( !sizeLinConstr() )
    {
        Vector<int> xignorefarfar(x.size());
        Vector<int> xgradordadd(x.size());

        Vector<int> xignorefarfarfar(x.size());
        Vector<int> xgradordaddR(x.size());

        xignorefarfar = 0;
        xgradordadd = 0;

        xignorefarfarfar = 0;
        xgradordaddR = 0;

        yyyaKm(m,res,x,xinfo,xignorefarfar,xignorefarfarfar,xgradordadd,xgradordaddR,bias,i,pxyprod,xdim,xconsist,resmode,mlid,xy,assumreal,justcalcip);
    }

    else
    {
        NiceAssert( m <= 2 );

        if ( m == 0 )
        {
            res = yyyK0(bias,pxyprod,xdim,xconsist,resmode,mlid,assumreal,justcalcip);
        }

        else if ( m == 1 )
        {
            res = yyyK1(*(x(0)),*(xinfo(0)),bias,nullptr,i(0),xdim,xconsist,resmode,mlid,nullptr,assumreal,justcalcip);
        }

        else
        {
            res = yyyK2(*(x(0)),*(x(1)),*(xinfo(0)),*(xinfo(1)),bias,nullptr,i(0),i(1),xdim,xconsist,resmode,mlid,nullptr,nullptr,nullptr,assumreal,justcalcip);
        }
    }

    return res;
}

gentype &MercerKernel::yyyKm(int m, gentype &res,
                             Vector<const SparseVector<gentype> *> &x,
                             Vector<const vecInfo *> &xinfo,
                             const gentype &bias,
                             Vector<int> &i,
                             const gentype **pxyprod, int xdim, int xconsist, int resmode, int mlid,
                             const Matrix<double> *xy, int assumreal, int justcalcip) const
{
    if ( !sizeLinConstr() )
    {
        Vector<int> xignorefarfar(x.size());
        Vector<int> xgradordadd(x.size());

        Vector<int> xignorefarfarfar(x.size());
        Vector<int> xgradordaddR(x.size());

        xignorefarfar = 0;
        xgradordadd = 0;

        xignorefarfarfar = 0;
        xgradordaddR = 0;

        yyyaKm(m,res,x,xinfo,xignorefarfar,xignorefarfarfar,xgradordadd,xgradordaddR,bias,i,pxyprod,xdim,xconsist,resmode,mlid,xy,assumreal,justcalcip);
    }

    else
    {
        NiceAssert( m <= 2 );

        if ( m == 0 )
        {
            yyyK0(res,bias,pxyprod,xdim,xconsist,resmode,mlid,assumreal,justcalcip);
        }

        else if ( m == 1 )
        {
            yyyK1(res,*(x(0)),*(xinfo(0)),bias,nullptr,i(0),xdim,xconsist,resmode,mlid,nullptr,assumreal,justcalcip);
        }

        else
        {
            yyyK2(res,*(x(0)),*(x(1)),*(xinfo(0)),*(xinfo(1)),bias,nullptr,i(0),i(1),xdim,xconsist,resmode,mlid,nullptr,nullptr,nullptr,assumreal,justcalcip);
        }
    }

    return res;
}

int MercerKernel::yyyphim(int m, Vector<double> &res, const SparseVector<gentype> &xa, const vecInfo &xainfo, int ia, int allowfinite, int xdim, int xconsist, int assumreal) const
{
    NiceAssert( !sizeLinConstr() );

    int xaignorefarfar = 0;
    int xagradordadd = 0;

    int xaignorefarfarfar = 0;
    int xagradordaddR = 0;

    int dres = yyyaphim(m,res,xa,xainfo,xaignorefarfar,xaignorefarfarfar,xagradordadd,xagradordaddR,ia,allowfinite,xdim,xconsist,assumreal);

    return dres;
}

int MercerKernel::yyyphim(int m, Vector<gentype> &res, const SparseVector<gentype> &xa, const vecInfo &xainfo, int ia, int allowfinite, int xdim, int xconsist, int assumreal) const
{
    NiceAssert( !sizeLinConstr() );

    int xaignorefarfar = 0;
    int xagradordadd = 0;

    int xaignorefarfarfar = 0;
    int xagradordaddR = 0;

    int dres = yyyaphim(m,res,xa,xainfo,xaignorefarfar,xaignorefarfarfar,xagradordadd,xagradordaddR,ia,allowfinite,xdim,xconsist,assumreal);

    return dres;
}























double MercerKernel::density(const SparseVector<gentype> &xa, const vecInfo &xainfo, double bias, int ia, int xdim, int xconsist, int mlid, int assumreal) const
{
    (void) ia;
    (void) mlid;

    NiceAssert( isSimpleKernel() );

    double res = 0;

    // K(x,x) = int p(omega) exp(-j omega'.(x-x')) domega
    // K(x/s,x/s) = int p(omega) exp(-j omega'.((x/s)-(x'/s))) domega
    // K(x/s,x/s) = int p(omega) exp(-j (omega/s)'.(x-x')) domega    (omega = ws, domega = sdw)
    // K(x/s,x/s) = int p(sw) exp(-j w'.(x-x')) s dw
    // K(x/s,x/s) = int q(w) exp(-j w'.(x-x')) dw
    //
    // q(w) = s p(sw)

    switch ( cType() )
    {
        case 3:
        {
            // Gaussian: exp(-d/(2.r0.r0)-r1) = exp(-d/(2.r0.r0)).exp(-r1)
            // (so treat exp(-r1) as a separate scale factor)
            // Density: p(w) = (2pi)^{-D/2} exp(-(||w||_2^2)/2)
            // Density: q(w) = r0 (2pi)^{-D/2} exp(-(||w||_2^2)/(2.r0.r0))

            double r0 = (double) dRealConstants(0)(0+1);
            double r1 = (double) dRealConstants(0)(1+1);

            double xxprod;

            xxprod = getmnorm(xainfo,xa,2,xconsist,assumreal);
            xxprod += bias;

            res =  r0*pow(NUMBASE_1ONSQRT2PI,xdim)*exp(-xxprod/(2*r0*r0)-r1);

            break;
        }

        case 4:
        {
            // Laplacian: exp(-sqrt(d)/r0-r1) = exp(-sqrt(d)/(r0.r0)).exp(-r1)
            // (so treat exp(-r1) as a separate scale factor)
            // Density: p(w) = prod_i 1/(pi*(1+w_i^2))
            // Density: q(w) = r0 prod_i 1/(pi*(1+(w_i/r0)^2))

            double r0 = (double) dRealConstants(0)(0+1);
            double r1 = (double) dRealConstants(0)(1+1);

            int i;

            res = r0*exp(-r1);

            for ( i = 0 ; i < xa.indsize() ; ++i )
            {
                res *= 1/(NUMBASE_PI*(1+(((double) xa.direcref(i))/r0)*(((double) xa.direcref(i))/r0)));
            }

            break;
        }

        case 19:
        {
            // Cauchy: 1/(1+(d/(r0.r0)))
            // Density: p(w) = exp(-||w||_1) = exp(-sum_i |w_i|)
            // Density: q(w) = r0 exp(-sum_i |w_i|/r0)

            double r0 = (double) dRealConstants(0)(0+1);

            int i;

            res = r0;

            for ( i = 0 ; i < xa.indsize() ; ++i )
            {
                res *= exp(-abs1((double) xa.direcref(i))/r0);
            }

            break;
        }

        default:
        {
            NiceThrow("Sorry, density is only implemented for Gaussian, Laplacian and Cauchy");

            break;
        }
    }

    res *= (double) cWeight();

    return res;
}














// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================

void MercerKernel::dK(double &xygrad, double &xnormgrad, int &minmaxind,
                        const SparseVector<gentype> &x, const SparseVector<gentype> &y,
                        const vecInfo &xinfo, const vecInfo &yinfo,
                        double bias,
                        const gentype **pxyprod,
                        int i, int j,
                        int xdim, int xconsist, int mlid, 
                        const double *xy00, const double *xy10, const double *xy11, int deepDeriv, int assumreal) const
{
    yyydKK2(xygrad,xnormgrad,minmaxind,x,y,xinfo,yinfo,bias,pxyprod,i,j,xdim,xconsist,assumreal,mlid,xy00,xy10,xy11,deepDeriv);
}

void MercerKernel::dK(gentype &xygrad, gentype &xnormgrad, int &minmaxind,
                         const SparseVector<gentype> &x, const SparseVector<gentype> &y,
                         const vecInfo &xinfo, const vecInfo &yinfo,
                         const gentype &bias,
                         const gentype **pxyprod,
                         int i, int j,
                         int xdim, int xconsist, int mlid, 
                         const double *xy00, const double *xy10, const double *xy11, int deepDeriv, int assumreal) const
{
    yyydKK2(xygrad,xnormgrad,minmaxind,x,y,xinfo,yinfo,bias,pxyprod,i,j,xdim,xconsist,assumreal,mlid,xy00,xy10,xy11,deepDeriv);
}


// ===============================================================================================


void MercerKernel::dK2delx(gentype &xscaleres, gentype &yscaleres,  int &minmaxind,
                          const SparseVector<gentype> &x, const SparseVector<gentype> &y, 
                          const vecInfo &xinfo, const vecInfo &yinfo, 
                          const gentype &bias, 
                          const gentype **pxyprod,
                          int i, int j, 
                          int xdim, int xconsist, int mlid, 
                          const double *xy00, const double *xy10, const double *xy11, int assumreal) const
{
    gentype xygrad(0.0);
    gentype xnormgrad(0.0);

    dK(xygrad,xnormgrad,minmaxind,x,y,xinfo,yinfo,bias,pxyprod,i,j,xdim,xconsist,mlid,xy00,xy10,xy11,1,assumreal);

    xscaleres  = xnormgrad;
    xscaleres *= 2.0;
    yscaleres  = xygrad;
}

void MercerKernel::dK2delx(double &xscaleres, double &yscaleres,  int &minmaxind,
                          const SparseVector<gentype> &x, const SparseVector<gentype> &y, 
                          const vecInfo &xinfo, const vecInfo &yinfo, 
                          double bias,
                          const gentype **pxyprod, 
                          int i, int j,
                          int xdim, int xconsist, int mlid, 
                          const double *xy00, const double *xy10, const double *xy11, int assumreal) const
{
    double xygrad = 0.0;
    double xnormgrad = 0.0;

    dK(xygrad,xnormgrad,minmaxind,x,y,xinfo,yinfo,bias,pxyprod,i,j,xdim,xconsist,mlid,xy00,xy10,xy11,1,assumreal);

    xscaleres  = xnormgrad;
    xscaleres *= 2.0;
    yscaleres  = xygrad;
}

void MercerKernel::dnK2del(Vector<gentype> &sc, Vector<Vector<int> > &n, int &minmaxind, 
                          const Vector<int> &q, 
                          const SparseVector<gentype> &x, const SparseVector<gentype> &y, 
                          const vecInfo &xinfo, const vecInfo &yinfo, 
                          const gentype &bias, 
                          const gentype **pxyprod, 
                          int i, int j, 
                          int xdim, int xconsist, int mlid, 
                          const double *xy00, const double *xy10, const double *xy11, int deepDerive, int assumreal) const
{
    int z = 0;

    if ( q.size() == 0 )
    {
        // "no gradient" case

        sc.resize(1);
        n.resize(1);

        n("&",z).resize(z);

        K2(sc("&",z),x,y,xinfo,yinfo,bias,pxyprod,i,j,xdim,xconsist,0,mlid,xy00,xy10,xy11,assumreal);
    }

    else if ( q.size() == 1 )
    {
        if ( q(z) == 0 )
        {
            // d/dx case - result is sc(0).x + sc(1).y

            sc.resize(2);
            n.resize(2);

            n("&",z).resize(1);
            n("&",1).resize(1);

            n("&",z)("&",z) = z;
            n("&",1)("&",z) = 1;

            dK2delx(sc("&",z),sc("&",1),minmaxind,x,y,xinfo,yinfo,bias,pxyprod,i,j,xdim,xconsist,mlid,xy00,xy10,xy11,assumreal);
        }

        else
        {
            // d/dy case - result is sc(0).x + sc(1).y
            // We assume symmetry to evaluate this

            sc.resize(2);
            n.resize(2);

            n("&",z).resize(1);
            n("&",1).resize(1);

            n("&",z)("&",z) = z;
            n("&",1)("&",z) = 1;

            dK2delx(sc("&",1),sc("&",z),minmaxind,y,x,yinfo,xinfo,bias,nullptr,j,i,xdim,xconsist,mlid,nullptr,nullptr,nullptr,assumreal);
        }
    }

    else if ( q.size() == 2 )
    {
        if ( ( q(z) == 0 ) && ( q(1) == 0 ) )
        {
            // d^2/dx^2 case - result is sc(0).x.x' + sc(1).y.y' + sc(2).x.y' + sc(3).y.x' + sc(4).I

            sc.resize(5);
            n.resize(5);

            n("&",z).resize(2);
            n("&",1).resize(2);
            n("&",2).resize(2);
            n("&",3).resize(2);
            n("&",4).resize(2);

            n("&",z)("&",z) = z;  n("&",z)("&",1) = z;
            n("&",1)("&",z) = 1;  n("&",1)("&",1) = 1;
            n("&",2)("&",z) = z;  n("&",2)("&",1) = 1;
            n("&",3)("&",z) = 1;  n("&",3)("&",1) = z;
            n("&",4)("&",z) = -1; n("&",4)("&",1) = -1;

            d2K2delxdelx(sc("&",z),sc("&",1),sc("&",2),sc("&",3),sc("&",4),minmaxind,x,y,xinfo,yinfo,bias,pxyprod,i,j,xdim,xconsist,mlid,xy00,xy10,xy11,deepDerive,assumreal);
        }

        else if ( ( q(z) == 0 ) && ( q(1) == 1 ) )
        {
            // d/dx d/dy case - result is sc(0).x.x' + sc(1).y.y' + sc(2).x.y' + sc(3).y.x' + sc(4).I

            sc.resize(5);
            n.resize(5);

            n("&",z).resize(2);
            n("&",1).resize(2);
            n("&",2).resize(2);
            n("&",3).resize(2);
            n("&",4).resize(2);

            n("&",z)("&",z) = z;  n("&",z)("&",1) = z;
            n("&",1)("&",z) = 1;  n("&",1)("&",1) = 1;
            n("&",2)("&",z) = z;  n("&",2)("&",1) = 1;
            n("&",3)("&",z) = 1;  n("&",3)("&",1) = z;
            n("&",4)("&",z) = -1; n("&",4)("&",1) = -1;

            d2K2delxdely(sc("&",z),sc("&",1),sc("&",2),sc("&",3),sc("&",4),minmaxind,x,y,xinfo,yinfo,bias,pxyprod,i,j,xdim,xconsist,mlid,xy00,xy10,xy11,deepDerive,assumreal);
        }

        else if ( ( q(z) == 1 ) && ( q(1) == 0 ) )
        {
            // d/dy d/dx case - result is sc(0).x.x' + sc(1).y.y' + sc(2).x.y' + sc(3).y.x' + sc(4).I
            // We assume symmetry to evaluate this

            sc.resize(5);
            n.resize(5);

            n("&",z).resize(2);
            n("&",1).resize(2);
            n("&",2).resize(2);
            n("&",3).resize(2);
            n("&",4).resize(2);

            n("&",z)("&",z) = z;  n("&",z)("&",1) = z;
            n("&",1)("&",z) = 1;  n("&",1)("&",1) = 1;
            n("&",2)("&",z) = z;  n("&",2)("&",1) = 1;
            n("&",3)("&",z) = 1;  n("&",3)("&",1) = z;
            n("&",4)("&",z) = -1; n("&",4)("&",1) = -1;

            d2K2delxdely(sc("&",1),sc("&",z),sc("&",3),sc("&",2),sc("&",4),minmaxind,y,x,yinfo,xinfo,bias,nullptr,j,i,xdim,xconsist,mlid,nullptr,nullptr,nullptr,deepDerive,assumreal);
        }

        else
        {
            // d/dy d/dy case - result is sc(0).x.x' + sc(1).y.y' + sc(2).x.y' + sc(3).y.x' + sc(4).I
            // We assume symmetry to evaluate this

            sc.resize(5);
            n.resize(5);

            n("&",z).resize(2);
            n("&",1).resize(2);
            n("&",2).resize(2);
            n("&",3).resize(2);
            n("&",4).resize(2);

            n("&",z)("&",z) = z;  n("&",z)("&",1) = z;
            n("&",1)("&",z) = 1;  n("&",1)("&",1) = 1;
            n("&",2)("&",z) = z;  n("&",2)("&",1) = 1;
            n("&",3)("&",z) = 1;  n("&",3)("&",1) = z;
            n("&",4)("&",z) = -1; n("&",4)("&",1) = -1;

            d2K2delxdelx(sc("&",1),sc("&",z),sc("&",3),sc("&",2),sc("&",4),minmaxind,y,x,yinfo,xinfo,bias,nullptr,j,i,xdim,xconsist,mlid,nullptr,nullptr,nullptr,deepDerive,assumreal);
        }
    }

    else
    {
        yyydnKK2del(sc,n,minmaxind,q,x,y,xinfo,yinfo,bias,pxyprod,i,j,xdim,xconsist,assumreal,mlid,xy00,xy10,xy11,deepDerive);
    }
}

void MercerKernel::dnK2del(Vector<double>  &sc, Vector<Vector<int> > &n, int &minmaxind, 
                          const Vector<int> &q, 
                          const SparseVector<gentype> &x, const SparseVector<gentype> &y, 
                          const vecInfo &xinfo, const vecInfo &yinfo, 
                          double bias, 
                          const gentype **pxyprod, 
                          int i, int j, 
                          int xdim, int xconsist, int mlid, 
                          const double *xy00, const double *xy10, const double *xy11, int deepDerive, int assumreal) const
{
    int z = 0;

    if ( q.size() == 0 )
    {
        // "no gradient" case

        sc.resize(1);
        n.resize(1);

        n("&",z).resize(z);

        sc("&",z) = K2(x,y,xinfo,yinfo,bias,pxyprod,i,j,xdim,xconsist,0,mlid,xy00,xy10,xy11,assumreal);
    }

    else if ( q.size() == 1 )
    {
        if ( q(z) == 0 )
        {
            // d/dx case - result is sc(0).x + sc(1).y

            sc.resize(2);
            n.resize(2);

            n("&",z).resize(1);
            n("&",1).resize(1);

            n("&",z)("&",z) = z;
            n("&",1)("&",z) = 1;

            dK2delx(sc("&",z),sc("&",1),minmaxind,x,y,xinfo,yinfo,bias,pxyprod,i,j,xdim,xconsist,mlid,xy00,xy10,xy11,assumreal);
        }

        else
        {
            // d/dy case - result is sc(0).x + sc(1).y
            // We assume symmetry to evaluate this

            sc.resize(2);
            n.resize(2);

            n("&",z).resize(1);
            n("&",1).resize(1);

            n("&",z)("&",z) = z;
            n("&",1)("&",z) = 1;

            dK2delx(sc("&",1),sc("&",z),minmaxind,y,x,yinfo,xinfo,bias,nullptr,j,i,xdim,xconsist,mlid,nullptr,nullptr,nullptr,assumreal);
        }
    }

    else if ( q.size() == 2 )
    {
        if ( ( q(z) == 0 ) && ( q(1) == 0 ) )
        {
            // d^2/dx^2 case - result is sc(0).x.x' + sc(1).y.y' + sc(2).x.y' + sc(3).y.x' + sc(4).I

            sc.resize(5);
            n.resize(5);

            n("&",z).resize(2);
            n("&",1).resize(2);
            n("&",2).resize(2);
            n("&",3).resize(2);
            n("&",4).resize(2);

            n("&",z)("&",z) = z;  n("&",z)("&",1) = z;
            n("&",1)("&",z) = 1;  n("&",1)("&",1) = 1;
            n("&",2)("&",z) = z;  n("&",2)("&",1) = 1;
            n("&",3)("&",z) = 1;  n("&",3)("&",1) = z;
            n("&",4)("&",z) = -1; n("&",4)("&",1) = -1;

            d2K2delxdelx(sc("&",z),sc("&",1),sc("&",2),sc("&",3),sc("&",4),minmaxind,x,y,xinfo,yinfo,bias,pxyprod,i,j,xdim,xconsist,mlid,xy00,xy10,xy11,deepDerive,assumreal);
        }

        else if ( ( q(z) == 0 ) && ( q(1) == 1 ) )
        {
            // d/dx d/dy case - result is sc(0).x.x' + sc(1).y.y' + sc(2).x.y' + sc(3).y.x' + sc(4).I

            sc.resize(5);
            n.resize(5);

            n("&",z).resize(2);
            n("&",1).resize(2);
            n("&",2).resize(2);
            n("&",3).resize(2);
            n("&",4).resize(2);

            n("&",z)("&",z) = z;  n("&",z)("&",1) = z;
            n("&",1)("&",z) = 1;  n("&",1)("&",1) = 1;
            n("&",2)("&",z) = z;  n("&",2)("&",1) = 1;
            n("&",3)("&",z) = 1;  n("&",3)("&",1) = z;
            n("&",4)("&",z) = -1; n("&",4)("&",1) = -1;

            d2K2delxdely(sc("&",z),sc("&",1),sc("&",2),sc("&",3),sc("&",4),minmaxind,x,y,xinfo,yinfo,bias,pxyprod,i,j,xdim,xconsist,mlid,xy00,xy10,xy11,deepDerive,assumreal);
        }

        else if ( ( q(z) == 1 ) && ( q(1) == 0 ) )
        {
            // d/dy d/dx case - result is sc(0).x.x' + sc(1).y.y' + sc(2).x.y' + sc(3).y.x' + sc(4).I
            // We assume symmetry to evaluate this

            sc.resize(5);
            n.resize(5);

            n("&",z).resize(2);
            n("&",1).resize(2);
            n("&",2).resize(2);
            n("&",3).resize(2);
            n("&",4).resize(2);

            n("&",z)("&",z) = z;  n("&",z)("&",1) = z;
            n("&",1)("&",z) = 1;  n("&",1)("&",1) = 1;
            n("&",2)("&",z) = z;  n("&",2)("&",1) = 1;
            n("&",3)("&",z) = 1;  n("&",3)("&",1) = z;
            n("&",4)("&",z) = -1; n("&",4)("&",1) = -1;

            d2K2delxdely(sc("&",1),sc("&",z),sc("&",3),sc("&",2),sc("&",4),minmaxind,y,x,yinfo,xinfo,bias,nullptr,j,i,xdim,xconsist,mlid,nullptr,nullptr,nullptr,deepDerive,assumreal);
        }

        else
        {
            // d/dy d/dy case - result is sc(0).x.x' + sc(1).y.y' + sc(2).x.y' + sc(3).y.x' + sc(4).I
            // We assume symmetry to evaluate this

            sc.resize(5);
            n.resize(5);

            n("&",z).resize(2);
            n("&",1).resize(2);
            n("&",2).resize(2);
            n("&",3).resize(2);
            n("&",4).resize(2);

            n("&",z)("&",z) = z;  n("&",z)("&",1) = z;
            n("&",1)("&",z) = 1;  n("&",1)("&",1) = 1;
            n("&",2)("&",z) = z;  n("&",2)("&",1) = 1;
            n("&",3)("&",z) = 1;  n("&",3)("&",1) = z;
            n("&",4)("&",z) = -1; n("&",4)("&",1) = -1;

            d2K2delxdelx(sc("&",1),sc("&",z),sc("&",3),sc("&",2),sc("&",4),minmaxind,y,x,yinfo,xinfo,bias,nullptr,j,i,xdim,xconsist,mlid,nullptr,nullptr,nullptr,deepDerive,assumreal);
        }
    }

    else
    {
        yyydnKK2del(sc,n,minmaxind,q,x,y,xinfo,yinfo,bias,pxyprod,i,j,xdim,xconsist,assumreal,mlid,xy00,xy10,xy11,deepDerive);
    }
}

void MercerKernel::d2K2delxdelx(gentype &xxscaleres, gentype &yyscaleres, gentype &xyscaleres, gentype &yxscaleres, gentype &constres, int &minmaxind, 
                 const SparseVector<gentype> &x, const SparseVector<gentype> &y, 
                 const vecInfo &xinfo, const vecInfo &yinfo, 
                 const gentype &bias, 
                 const gentype **pxyprod, 
                 int i, int j, 
                 int xdim, int xconsist, int mlid, 
                 const double *xy00, const double *xy10, const double *xy11, int deepDeriv, int assumreal) const
{
    // Assume any kernel can be written as:
    //
    // K(x,y) = K(a,z,b)
    //
    // where a = ||x||^2
    //       b = ||y||^2
    //       z = x'y
    //
    // dK/dx_j = dK/da da/dx_j + dK/dz dz/dx_j
    //         = dK/da 2x_j + dK/dz y_j
    //
    // d2K/dx_idx_j = d2K/dada da/dx_i 2x_j + d2K/dzda dz/dx_i 2x_j + dK/da 2 delta_{ij} + d2K/dadz da/dx_i y_j + d2K/dzdz dz/dx_i y_j
    //              = d2K/dada 2x_i 2x_j + d2K/dzda y_i 2x_j + dK/da 2 delta_{ij} + d2K/dadz 2x_i y_j + d2K/dzdz y_i y_j
    //
    // d2K/dxdx = 4 d2K/dada x.x' + 2 d2K/dzda y.x' + 2 d2K/dadz x.y' + d2K/dzdz y.y' + 2 dK/da I

    gentype xygrad;
    gentype xnormgrad;
    gentype xyxygrad;
    gentype xyxnormgrad;
    gentype xyynormgrad;
    gentype xnormxnormgrad;
    gentype xnormynormgrad;
    gentype ynormynormgrad;

    d2K(xygrad,xnormgrad,xyxygrad,xyxnormgrad,xyynormgrad,xnormxnormgrad,xnormynormgrad,ynormynormgrad,minmaxind,x,y,xinfo,yinfo,bias,pxyprod,i,j,xdim,xconsist,mlid,xy00,xy10,xy11,deepDeriv,assumreal);

    xxscaleres = 4.0*xnormxnormgrad;
    xyscaleres = 2.0*xyxnormgrad;
    yxscaleres = xyscaleres;
    yyscaleres = xyxygrad;
    constres   = 2.0*xnormgrad;
}

void MercerKernel::d2K2delxdely(gentype &xxscaleres, gentype &yyscaleres, gentype &xyscaleres, gentype &yxscaleres, gentype &constres, int &minmaxind, 
                 const SparseVector<gentype> &x, const SparseVector<gentype> &y, 
                 const vecInfo &xinfo, const vecInfo &yinfo, 
                 const gentype &bias, 
                 const gentype **pxyprod, 
                 int i, int j, 
                 int xdim, int xconsist, int mlid, 
                 const double *xy00, const double *xy10, const double *xy11, int deepDeriv, int assumreal) const
{
    // Assume any kernel can be written as:
    //
    // K(x,y) = K(a,z,b)
    //
    // where a = ||x||^2
    //       b = ||y||^2
    //       z = x'y
    //
    // dK/dx_j = dK/da da/dx_j + dK/dz dz/dx_j
    //         = dK/da 2x_j + dK/dz y_j
    //
    // d2K/dy_idx_j = d2K/dzda dz/dy_i 2x_j + d2K/dbda db/dy_i 2x_j + d2K/dzdz dz/dy_i y_j + d2K/dbdz db/dy_i y_j + dK/dz delta_{ij}
    //              = d2K/dzda x_i     2x_j + d2K/dbda 2y_i    2x_j + d2K/dzdz x_i     y_j + d2K/dbdz 2y_i    y_j + dK/dz delta_{ij}
    //              = 2 d2K/dzda x_i x_j + 4 d2K/dbda y_i x_j + d2K/dzdz x_i y_j + 2 d2K/dbdz y_i y_j + dK/dz delta_{ij}
    //
    // d2K/dx_idy_j = 2 d2K/dzda x_i x_j + 4 d2K/dbda x_i y_j + d2K/dzdz y_i x_j + 2 d2K/dbdz y_i y_j + dK/dz delta_{ij}
    //
    // d2K/dxdy = 2 d2K/dzda x.x' + 4 d2K/dbda x.y' + d2K/dzdz y.x' + 2 d2K/dbdz y.y' + dK/dz I

    gentype xygrad;
    gentype xnormgrad;
    gentype xyxygrad;
    gentype xyxnormgrad;
    gentype xyynormgrad;
    gentype xnormxnormgrad;
    gentype xnormynormgrad;
    gentype ynormynormgrad;

    d2K(xygrad,xnormgrad,xyxygrad,xyxnormgrad,xyynormgrad,xnormxnormgrad,xnormynormgrad,ynormynormgrad,minmaxind,x,y,xinfo,yinfo,bias,pxyprod,i,j,xdim,xconsist,mlid,xy00,xy10,xy11,deepDeriv,assumreal);

    xxscaleres = 2.0*xyxnormgrad;
    xyscaleres = 4.0*xnormynormgrad;
    yxscaleres = xyxygrad;
    yyscaleres = 2.0*xyynormgrad;
    constres   = xygrad;
}

void MercerKernel::d2K2delxdelx(double &xxscaleres, double &yyscaleres, double &xyscaleres, double &yxscaleres, double &constres, int &minmaxind, 
                 const SparseVector<gentype> &x, const SparseVector<gentype> &y, 
                 const vecInfo &xinfo, const vecInfo &yinfo, 
                 double bias, 
                 const gentype **pxyprod, 
                 int i, int j, 
                 int xdim, int xconsist, int mlid, 
                 const double *xy00, const double *xy10, const double *xy11, int deepDeriv, int assumreal) const
{
    // Assume any kernel can be written as:
    //
    // K(x,y) = K(a,z,b)
    //
    // where a = ||x||^2
    //       b = ||y||^2
    //       z = x'y
    //
    // dK/dx_j = dK/da da/dx_j + dK/dz dz/dx_j
    //         = dK/da 2x_j + dK/dz y_j
    //
    // d2K/dx_idx_j = d2K/dada da/dx_i 2x_j + d2K/dzda dz/dx_i 2x_j + dK/da 2 delta_{ij} + d2K/dadz da/dx_i y_j + d2K/dzdz dz/dx_i y_j
    //              = d2K/dada 2x_i 2x_j + d2K/dzda y_i 2x_j + dK/da 2 delta_{ij} + d2K/dadz 2x_i y_j + d2K/dzdz y_i y_j
    //
    // d2K/dxdx = 4 d2K/dada x.x' + 2 d2K/dzda y.x' + 2 d2K/dadz x.y' + d2K/dzdz y.y' + 2 dK/da I

    double xygrad;
    double xnormgrad;
    double xyxygrad;
    double xyxnormgrad;
    double xyynormgrad;
    double xnormxnormgrad;
    double xnormynormgrad;
    double ynormynormgrad;

    d2K(xygrad,xnormgrad,xyxygrad,xyxnormgrad,xyynormgrad,xnormxnormgrad,xnormynormgrad,ynormynormgrad,minmaxind,x,y,xinfo,yinfo,bias,pxyprod,i,j,xdim,xconsist,mlid,xy00,xy10,xy11,deepDeriv,assumreal);

    xxscaleres = 4.0*xnormxnormgrad;
    xyscaleres = 2.0*xyxnormgrad;
    yxscaleres = xyscaleres;
    yyscaleres = xyxygrad;
    constres   = 2.0*xnormgrad;
}

void MercerKernel::d2K2delxdely(double &xxscaleres, double &yyscaleres, double &xyscaleres, double &yxscaleres, double &constres, int &minmaxind, 
                 const SparseVector<gentype> &x, const SparseVector<gentype> &y, 
                 const vecInfo &xinfo, const vecInfo &yinfo, 
                 double bias, 
                 const gentype **pxyprod, 
                 int i, int j, 
                 int xdim, int xconsist, int mlid, 
                 const double *xy00, const double *xy10, const double *xy11, int deepDeriv, int assumreal) const
{
    // Assume any kernel can be written as:
    //
    // K(x,y) = K(a,z,b)
    //
    // where a = ||x||^2
    //       b = ||y||^2
    //       z = x'y
    //
    // dK/dx_j = dK/da da/dx_j + dK/dz dz/dx_j
    //         = dK/da 2x_j + dK/dz y_j
    //
    // d2K/dy_idx_j = d2K/dzda dz/dy_i 2x_j + d2K/dbda db/dy_i 2x_j + d2K/dzdz dz/dy_i y_j + d2K/dbdz db/dy_i y_j + dK/dz delta_{ij}
    //              = d2K/dzda x_i     2x_j + d2K/dbda 2y_i    2x_j + d2K/dzdz x_i     y_j + d2K/dbdz 2y_i    y_j + dK/dz delta_{ij}
    //              = 2 d2K/dzda x_i x_j + 4 d2K/dbda y_i x_j + d2K/dzdz x_i y_j + 2 d2K/dbdz y_i y_j + dK/dz delta_{ij}
    //
    // d2K/dx_idy_j = 2 d2K/dzda x_i x_j + 4 d2K/dbda x_i y_j + d2K/dzdz y_i x_j + 2 d2K/dbdz y_i y_j + dK/dz delta_{ij}
    //
    // d2K/dxdy = 2 d2K/dzda x.x' + 4 d2K/dbda x.y' + d2K/dzdz y.x' + 2 d2K/dbdz y.y' + dK/dz I

    double xygrad;
    double xnormgrad;
    double xyxygrad;
    double xyxnormgrad;
    double xyynormgrad;
    double xnormxnormgrad;
    double xnormynormgrad;
    double ynormynormgrad;

    d2K(xygrad,xnormgrad,xyxygrad,xyxnormgrad,xyynormgrad,xnormxnormgrad,xnormynormgrad,ynormynormgrad,minmaxind,x,y,xinfo,yinfo,bias,pxyprod,i,j,xdim,xconsist,mlid,xy00,xy10,xy11,deepDeriv,assumreal);

    xxscaleres = 2.0*xyxnormgrad;
    xyscaleres = 4.0*xnormynormgrad;
    yxscaleres = xyxygrad;
    yyscaleres = 2.0*xyynormgrad;
    constres   = xygrad;
}


void MercerKernel::d2K(gentype &xygrad, gentype &xnormgrad, gentype &xyxygrad, gentype &xyxnormgrad, gentype &xyynormgrad, gentype &xnormxnormgrad, gentype &xnormynormgrad, gentype &ynormynormgrad, int &minmaxind, 
         const SparseVector<gentype> &x, const SparseVector<gentype> &y, 
         const vecInfo &xinfo, const vecInfo &yinfo, 
         const gentype &bias, const gentype **pxyprod, 
         int i, int j, 
         int xdim, int xconsist, int mlid, 
         const double *xy00, const double *xy10, const double *xy11, int deepDeriv, int assumreal) const
{
    yyyd2KK2(xygrad,xnormgrad,xyxygrad,xyxnormgrad,xyynormgrad,xnormxnormgrad,xnormynormgrad,ynormynormgrad,minmaxind,x,y,xinfo,yinfo,bias,pxyprod,i,j,xdim,xconsist,assumreal,mlid,xy00,xy10,xy11,deepDeriv);
}

void MercerKernel::d2K(double &xygrad, double &xnormgrad, double &xyxygrad, double &xyxnormgrad, double &xyynormgrad, double &xnormxnormgrad, double &xnormynormgrad, double &ynormynormgrad, int &minmaxind, 
         const SparseVector<gentype> &x, const SparseVector<gentype> &y, 
         const vecInfo &xinfo, const vecInfo &yinfo, 
         double bias, const gentype **pxyprod, 
         int i, int j, 
         int xdim, int xconsist, int mlid, 
         const double *xy00, const double *xy10, const double *xy11, int deepDeriv, int assumreal) const
{
    yyyd2KK2(xygrad,xnormgrad,xyxygrad,xyxnormgrad,xyynormgrad,xnormxnormgrad,xnormynormgrad,ynormynormgrad,minmaxind,x,y,xinfo,yinfo,bias,pxyprod,i,j,xdim,xconsist,assumreal,mlid,xy00,xy10,xy11,deepDeriv);
}








// ====================================================================================


// "Reversing" functions.
//
// For speed of operation it is sometimes helpful to retrieve either the
// inner product or distance from an evaluated kernel.  These functions
// let you do that
//
// isReversible: test if kernel is reversible.  Output is:
//     0: kernel cannot be reversed
//     1: kernel can be reversed to produce <x,y>+bias
//     2: kernel can be reversed to produce ||x-y||^2
//
// reverseK: reverse kernel as described by isReversible
//
// The result so produced can be fed back in via the pxyprod argument
// (appropriately set) to speed up calculation of results.  Use case
// could be quickly changing kernel parameters with minimal recalculation.
//
// As a general rule these only work with isSimpleFastKernel or
// isSimpleKernelChain, and then in limited cases.  For the chain case
// the result is the relevant (processed) output of the first layer.

//phantomx
//ADDHERE
int MercerKernel::isReversible(void) const
{
    int res = 0;

    if ( ( ( size() == 1 ) && isSimpleKernel() && churnInner() ) || ( ( size() == 2 ) && isSimpleKernelChain() && churnInner() ) )
    {
        const Vector<int> &ic = dIntConstants(size()-1);

        if (   ( cType(size()-1) == 1  )                          ||
             ( ( cType(size()-1) == 2  ) && !(ic(0)%2) && ic(0) ) ||
               ( cType(size()-1) == 7  )                             )
        {
            res = 1;
        }

        else if ( ( cType(size()-1) == 3  ) ||
                  ( cType(size()-1) == 4  ) ||
                  ( cType(size()-1) == 5  ) ||
                  ( cType(size()-1) == 14 ) ||
                  ( cType(size()-1) == 15 ) ||
                  ( cType(size()-1) == 23 ) ||
                  ( cType(size()-1) == 38 )    )
        {
            res = 2;
        }
    }

    return res;
}

//KERNELSHERE
//phantomx
gentype &MercerKernel::reverseK(gentype &res, const gentype &Kval) const
{
    if ( ( ( size() == 1 ) && isSimpleKernel() ) || ( ( size() == 2 ) && isSimpleKernelChain() ) )
    {
        res /= cWeight(size()-1);

        retVector<gentype> tmpva;

        const Vector<gentype> &r = dRealConstants(size()-1)(1,1,dRealConstants(size()-1).size()-1,tmpva);
        const Vector<int> &ic = dIntConstants(size()-1);

        if ( cType(size()-1) == 1 )
        {
            res  = Kval;
            res *= r(0);
            res *= r(0);
        }

        else if ( ( cType(size()-1) == 2  ) && !(ic(0)%2) && ic(0) )
        {
            res  = Kval;
            raiseto(res,1_gent/((double) ic(0)));
            //raiseto(res,oneintgentype()/((double) ic(0)));
            res -= r(1);
            res *= r(0);
            res *= r(0);
        }

        else if ( cType(size()-1) == 7 )
        {
            res  = Kval;
            OP_atanh(res);
            res -= r(1);
            res *= r(0);
            res *= r(0);
        }


        else if ( cType(size()-1) == 3  )
        {
            res  = Kval;
            OP_log(res);
            res += r(1);
            res.negate();
            res *= r(0);
            res *= r(0);
            res *= 2;
        }

        else if ( cType(size()-1) == 4  )
        {
            res  = Kval;
            OP_log(res);
            res += r(1);
            res.negate();
            res *= r(0);
            raiseto(res,2);
        }

        else if ( cType(size()-1) == 5  )
        {
            res  = Kval;
            OP_log(res);
            res += r(2);
            res.negate();
            res *= r(1);
            res *= pow(r(0),r(1));
            //raiseto(res,twointgentype()/r(1));
            raiseto(res,2_gent/r(1));
        }

        else if ( cType(size()-1) == 14 )
        {
            res  = Kval;
            res.negate();
            //raiseto(res,twointgentype()/r(1));
            raiseto(res,2_gent/r(1));
            res *= r(0);
            res *= r(0);
        }

        else if ( cType(size()-1) == 15 )
        {
            res  = Kval;
            res.negate();
            OP_exp(res);
            res -= 1;
            raiseto(res,2_gent/r(1));
            //raiseto(res,twointgentype()/r(1));
            res *= r(0);
            res *= r(0);
        }

        else if ( cType(size()-1) == 23 )
        {
            res  = Kval;
            res.inverse();
            res -= 1;
            res *= r(0);
            res *= r(0);
        }

        else if ( cType(size()-1) == 38 )
        {
            res  = Kval;
            OP_log(res);
            res.negate();
            res *= r(0);
            raiseto(res,2);
        }

        else if ( cType(size()-1) == 41 )
        {
            res  = Kval;
            //raiseto(res,twointgentype()*r(0)*r(0));
            raiseto(res,2_gent*r(0)*r(0));
        }

        else
        {
            NiceThrow("phooey");
        }

        // Secondary scaling

        if ( isSimpleKernelChain() )
        {
            res /= cWeight(0);
        }
    }

    else
    {
        NiceThrow("wassseirefwn");
    }

    return res;
}

//KERNELSHERE
//phantomx
double  &MercerKernel::reverseK(double &res, double Kval) const
{
    if ( ( ( size() == 1 ) && isSimpleKernel() ) || ( ( size() == 2 ) && isSimpleKernelChain() ) )
    {
        res /= (double) cWeight(size()-1);

        retVector<gentype> tmpva;

        const Vector<gentype> &r = dRealConstants(size()-1)(1,1,dRealConstants(size()-1).size()-1,tmpva);
        const Vector<int> &ic = dIntConstants(size()-1);

        if ( cType(size()-1) == 1  )
        {
            res =  Kval;
            res *= (double) r(0);
            res *= (double) r(0);
        }

        else if ( ( cType(size()-1) == 2  ) && !(ic(0)%2) && ic(0) )
        {
            res  = pow(Kval,1/((double) ic(0)));
            res -= (double) r(1);
            res *= (double) r(0);
            res *= (double) r(0);
        }

        else if ( cType(size()-1) == 7  )
        {
            res  = atanh(Kval);
            res -= (double) r(1);
            res /= (double) r(0);
        }


        else if ( cType(size()-1) == 3  )
        {
            res  = -log(Kval);
            res -= (double) r(1);
            res *= (double) r(0);
            res *= (double) r(0);
            res *= 2;
        }

        else if ( cType(size()-1) == 4  )
        {
            res  = -log(Kval);
            res -= (double) r(1);
            res *= (double) r(0);
            res *= res;
        }

        else if ( cType(size()-1) == 5  )
        {
            res  = -log(Kval);
            res -= (double) r(2);
            res *= (double) r(1);
            res *= (double) pow((double) r(0),(double) r(1));
            res  = pow(res,2/((double) r(1)));
        }

        else if ( cType(size()-1) == 14 )
        {
            res  = -Kval;
            res  = pow(res,2/((double) r(1)));
            res *= (double) r(0);
            res *= (double) r(0);
        }

        else if ( cType(size()-1) == 15 )
        {
            res  = -Kval;
            res  = exp(res);
            res -= 1;
            res  = pow(res,2/((double) r(1)));
            res *= (double) r(0);
            res *= (double) r(0);
        }

        else if ( cType(size()-1) == 23 )
        {
            res  = 1/Kval;
            res -= 1;
            res *= (double) r(0);
            res *= (double) r(0);
        }

        else if ( cType(size()-1) == 38 )
        {
            res  = -log(Kval);
            res *= (double) r(0);
            res *= res;
        }

        else if ( cType(size()-1) == 41 )
        {
            res  = pow(Kval,2*((double) r(0))*((double) r(0)));
        }

        else
        {
            NiceThrow("cor blimey!");
        }

        // Secondary scaling

        if ( isSimpleKernelChain() )
        {
            res /= (double) cWeight(0);
        }
    }

    else
    {
        gentype tempres(res);
        gentype tempKval(Kval);

        reverseK(tempres,tempKval);

        res = (double) tempres;
    }

    return res;
}














// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================

int MercerKernel::subSample(SparseVector<SparseVector<gentype> > &subval, gentype &a) const
{
    int res = 0;

    Vector<gentype> locdist(xsampdist);

    NiceAssert( locdist.size() == xindsub.size() );

    int j;

    for ( j = 0 ; j < locdist.size() ; ++j )
    {
        locdist("&",j).finalise();
        subval("&",0)("&",xindsub(j)) = locdist(j);
        subval("&",0)("&",xindsub(j)).finalise();
    }

    res += a.substitute(subval);
    res += a.finalise();

    return res;
}

int MercerKernel::subSample(SparseVector<SparseVector<gentype> > &subval, double &a) const
{
    int res = 0;

    (void) subval;
    (void) a;

    NiceThrow("This is weird.");

    return res;
}

int MercerKernel::subSample(SparseVector<SparseVector<gentype> > &subval, SparseVector<gentype> &x, vecInfo &xinfo) const
{
    int res = 0;

    Vector<gentype> locdist(xsampdist);

    NiceAssert( locdist.size() == xindsub.size() );

    int j;

    for ( j = 0 ; j < locdist.size() ; ++j )
    {
        locdist("&",j).finalise();
        subval("&",0)("&",xindsub(j)) = locdist(j);
        subval("&",0)("&",xindsub(j)).finalise();
    }

    for ( j = 0 ; j < x.indsize() ; ++j )
    {
        res += x.direref(j).substitute(subval);
        res += x.direref(j).finalise();
    }

    getvecInfo(xinfo,x);

    return res;
}



















// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================

const gentype &MercerKernel::getmnorm(const vecInfo &xinfo, const SparseVector<gentype> &x, int m, int xconsist, int assumreal) const
{
    m = isAltDiff() ? 2 : m;
    //m = isAltDiff() ? diffnormdefault() : m;

    if ( !(m%2) )
    {
        Vector<gentype> &xhalfmprod = **(xinfo.xhalfinda()); // strip the const

        int oldm = 2*(xhalfmprod.size());

        if ( m != oldm )
        {
            // Lock to allow sharing of vecinfo between threads

            xhalfmprod.resize(m/2);

            if ( ( m >= 2 ) && ( m > oldm ) )
            {
                gentype tmpres;

                twoProductDiverted(tmpres,x,x,xconsist,assumreal);

#ifdef ENABLE_THREADS
                static std::mutex eyelock;
                eyelock.lock();
#endif

                xhalfmprod("&",0) = tmpres;

#ifdef ENABLE_THREADS
                eyelock.unlock();
#endif

                oldm = 2;
            }

            if ( ( m >= 4 ) && ( m > oldm ) )
            {
                gentype tmpres;

                fourProductDiverted(xhalfmprod("&",1),x,x,x,x,xconsist,assumreal);

#ifdef ENABLE_THREADS
                static std::mutex eyelock;
                eyelock.lock();
#endif

                xhalfmprod("&",1) = tmpres;

#ifdef ENABLE_THREADS
                eyelock.unlock();
#endif

                oldm = 4;
            }

            if ( ( m >= 6 ) && ( m > oldm ) )
            {
                Vector<const SparseVector<gentype> *> aa(m);
                Vector<const vecInfo *> aainfo(m);

                aa     = &x;
                aainfo = &xinfo;

                int i;

                retVector<const SparseVector<gentype> *> tmpva;
                retVector<const vecInfo *>               tmpvb;

#ifdef ENABLE_THREADS
                static std::mutex eyelock;
                eyelock.lock();
#endif

                for ( i = oldm+2 ; i <= m ; i += 2 )
                {
                    mProductDiverted(i,xhalfmprod("&",(i/2)-1),aa(0,1,i-1,tmpva),xconsist,assumreal);
                }

#ifdef ENABLE_THREADS
                eyelock.unlock();
#endif

                oldm = m;
            }
        }

        return xhalfmprod((m/2)-1);
    }

    static thread_local Vector<gentype> scratch(100,nullptr,2);
    static thread_local int scratchind(99);

    int scrind = (int) ++scratchind;
    scrind %= 20;

    gentype &res = scratch("&",scrind);

    if ( m == 1 )
    {
        oneProductDiverted(res,x,xconsist,assumreal);
    }

    else if ( m == 3 )
    {
        threeProductDiverted(res,x,x,x,xconsist,assumreal);
    }

    else
    {
        Vector<const SparseVector<gentype> *> aa(m);
        Vector<const vecInfo *> aainfo(m);

        aa     = &x;
        aainfo = &xinfo;

        mProductDiverted(m,res,aa,xconsist,assumreal);
    }

    return res;
}

gentype &MercerKernel::getmnorm(vecInfo &xinfo, const SparseVector<gentype> &x, int m, int xconsist, int assumreal) const
{
    m = isAltDiff() ? 2 : m;
    //m = isAltDiff() ? diffnormdefault() : m;

    if ( !(m%2) )
    {
        Vector<gentype> &xhalfmprod = **(xinfo.xhalfinda()); // strip the const

        int oldm = 2*(xhalfmprod.size());

        if ( m != oldm )
        {
            xhalfmprod.resize(m/2);

            if ( ( m >= 2 ) && ( m > oldm ) )
            {
#ifdef ENABLE_THREADS
                static std::mutex eyelock;
                eyelock.lock();
#endif

                gentype tmpres;

                twoProductDiverted(tmpres,x,x,xconsist,assumreal);

                xhalfmprod("&",0) = tmpres;

#ifdef ENABLE_THREADS
                eyelock.unlock();
#endif

                oldm = 2;
            }

            if ( ( m >= 4 ) && ( m > oldm ) )
            {
#ifdef ENABLE_THREADS
                static std::mutex eyelock;
                eyelock.lock();
#endif

                gentype tmpres;

                fourProductDiverted(tmpres,x,x,x,x,xconsist,assumreal);

                xhalfmprod("&",1) = tmpres;

#ifdef ENABLE_THREADS
                eyelock.unlock();
#endif

                oldm = 4;
            }

            if ( ( m >= 6 ) && ( m > oldm ) )
            {
                Vector<const SparseVector<gentype> *> aa(m);
                Vector<const vecInfo *> aainfo(m);

                aa     = &x;
                aainfo = &xinfo;

                int i;

#ifdef ENABLE_THREADS
                static std::mutex eyelock;
                eyelock.lock();
#endif

                for ( i = oldm+2 ; i <= m ; i += 2 )
                {
                    retVector<const SparseVector<gentype> *> tmpva;
                    retVector<const vecInfo *>               tmpvb;

                    mProductDiverted(i,xhalfmprod("&",(i/2)-1),aa(0,1,i-1,tmpva),xconsist,assumreal);
                }

#ifdef ENABLE_THREADS
                eyelock.unlock();
#endif

                oldm = m;
            }
        }

        return xhalfmprod("&",(m/2)-1);
    }

    static thread_local Vector<gentype> scratch(100,nullptr,2);
    static thread_local int scratchind(99);

    int scrind = (int) ++scratchind;
    scrind %= 20;

    gentype &res = scratch("&",scrind);

    if ( m == 1 )
    {
        oneProductDiverted(res,x,xconsist,assumreal);
    }

    else if ( m == 3 )
    {
        threeProductDiverted(res,x,x,x,xconsist,assumreal);
    }

    else
    {
        Vector<const SparseVector<gentype> *> aa(m);
        Vector<const vecInfo *> aainfo(m);

        aa     = &x;
        aainfo = &xinfo;

        mProductDiverted(m,res,aa,xconsist,assumreal);
    }

    return res;
}




























// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================

//extern Vector<gentype> &makeanRKHSVector(Vector<gentype> &res, const MercerKernel &kern, const SparseVector<gentype> &x, const gentype &a, int m);
Vector<gentype> &makeanRKHSVector(Vector<gentype> &res, const MercerKernel &kern, const SparseVector<gentype> &x, const gentype &a);
Vector<gentype> &makeanRKHSVector(Vector<gentype> &res, const MercerKernel &kern, const SparseVector<gentype> &x, const gentype &a, int m);

Vector<gentype> &MercerKernel::local_makeanRKHSVector(Vector<gentype> &res, const MercerKernel &kern, const SparseVector<gentype> &x, const gentype &a, int m) const
{
    return makeanRKHSVector(res,kern,x,a,m);
}

Vector<double> &MercerKernel::local_makeanRKHSVector(Vector<double> &res, const MercerKernel &, const SparseVector<gentype> &, const gentype &, int) const
{
    NiceThrow("A Vector<double> RKHS doesn't exist.");

    return res;
}


//Vector<gentype> &MercerKernel::phim(int m, Vector<gentype> &res, const SparseVector<gentype> &x, int ia, int allowfinite, int xdim, int xconsist, int assumreal) const
/*
int MercerKernel::phim(int m, Vector<gentype> &res, const SparseVector<gentype> &x, int, int allowfinite, int xdim, int, int) const
{
    //(void) ia;
    //(void) xconsist;
    //(void) assumreal;

    retVector<gentype> tmpva;

    const Vector<gentype> &r = dRealConstants(size()-1)(1,1,dRealConstants(size()-1).size()-1,tmpva);
    const Vector<int> &ic = dIntConstants(size()-1);

    int i;
    int dres = -1; // This signifies default infdim vector, which is the default

    if ( allowfinite && ( ( size() == 1 ) && isVeryTrivialKernel() ) && ( cType(0) == 0 ) )
    {
        // K(x,y) = r1

        dres = 1;
    }

    else if ( allowfinite && ( ( size() == 1 ) && isVeryTrivialKernel() ) && ( ( cType(0) == 1 ) || ( cType(0) == 100 ) ) )
    {
        // K(x,y) = <x/r0,y/r0>

        dres = xdim;
    }

    else if ( allowfinite && ( ( size() == 1 ) && isVeryTrivialKernel() ) && ( cType(0) == 2 ) )
    {
        // K(x,y) = ( r1 + <x/r0,y/r0> )^i0

        dres = pow(xdim+1,ic(0));
    }

  if ( m >= 0 )
  {
    if ( allowfinite && ( ( size() == 1 ) && isVeryTrivialKernel() ) && ( cType(0) == 0 ) )
    {
        // K(x,y) = r1

        if ( res.infsize() )
        {
            const static Vector<gentype> temp(1);

            res = temp;
        }

        (res.resize(1))("&",0) = sqrt(r(1));
    }

    else if ( allowfinite && ( ( size() == 1 ) && isVeryTrivialKernel() ) && ( ( cType(0) == 1 ) || ( cType(0) == 100 ) ) )
    {
        // K(x,y) = <x/r0,y/r0>

        if ( res.infsize() )
        {
            const static Vector<gentype> temp;

            res = temp;
        }

        res.resize(xdim);

        for ( i = 0 ; i < xdim ; ++i )
        {
            res("&",i) = x(i);
            res("&",i) /= r(0);
        }
    }

    else if ( allowfinite && ( ( size() == 1 ) && isVeryTrivialKernel() ) && ( cType(0) == 2 ) )
    {
        // K(x,y) = ( r1 + <x/r0,y/r0> )^i0

        if ( res.infsize() )
        {
            const static Vector<gentype> temp;

            res = temp;
        }

        SparseVector<gentype> tmp(x);

        tmp /= r(0);
        tmp("&",xdim) = sqrt(r(1));

        kronpow(res,tmp,xdim+1,ic(0));
    }

    else
    {
        // Return as RKHS vector

        gentype a(1.0);

        makeanRKHSVector(res,*this,x,a,m);
    }
  }

    return dres;
}

//Vector<double> &MercerKernel::phim(int m, Vector<double> &res, const SparseVector<gentype> &x, int ia, int allowfinite, int xdim, int xconsist, int assumreal) const
int MercerKernel::phim(int, Vector<double> &res, const SparseVector<gentype> &x, int, int allowfinite, int xdim, int, int) const
{
    //(void) ia;
    //(void) xconsist;
    //(void) assumreal;
    //(void) m;

    retVector<gentype> tmpva;

    const Vector<gentype> &r = dRealConstants(size()-1)(1,1,dRealConstants(size()-1).size()-1,tmpva);
    const Vector<int> &ic = dIntConstants(size()-1);

    int i;
    int dres = -1; // This signifies default infdim vector, which is the default

    if ( allowfinite && ( ( size() == 1 ) && isVeryTrivialKernel() ) && ( cType(0) == 0 ) )
    {
        // K(x,y) = r1

        dres = 1;
    }

    else if ( allowfinite && ( ( size() == 1 ) && isVeryTrivialKernel() ) && ( ( cType(0) == 1 ) || ( cType(0) == 100 ) ) )
    {
        // K(x,y) = <x/r0,y/r0>

        dres = xdim;
    }

    else if ( allowfinite && ( ( size() == 1 ) && isVeryTrivialKernel() ) && ( cType(0) == 2 ) )
    {
        // K(x,y) = ( r1 + <x/r0,y/r0> )^i0

        dres = pow(xdim+1,ic(0));
    }

  if ( m >= 0 )
  {
    if ( allowfinite && ( ( size() == 1 ) && isVeryTrivialKernel() ) && ( cType(0) == 0 ) )
    {
        // K(x,y) = r1

        if ( res.infsize() )
        {
            const static Vector<double> temp(1);

            res = temp;
        }

        (res.resize(1))("&",0) = (double) sqrt(r(1));
    }

    else if ( allowfinite && ( ( size() == 1 ) && isVeryTrivialKernel() ) && ( ( cType(0) == 1 ) || ( cType(0) == 100 ) ) )
    {
        // K(x,y) = <x/r0,y/r0>

        if ( res.infsize() )
        {
            const static Vector<double> temp;

            res = temp;
        }

        res.resize(xdim);

        for ( i = 0 ; i < xdim ; ++i )
        {
            res("&",i)  = (double) x(i);
            res("&",i) /= (double) r(0);
        }
    }

    else if ( allowfinite && ( ( size() == 1 ) && isVeryTrivialKernel() ) && ( cType(0) == 2 ) )
    {
        // K(x,y) = ( r1 + <x/r0,y/r0> )^i0

        if ( res.infsize() )
        {
            const static Vector<double> temp;

            res = temp;
        }

        SparseVector<gentype> tmp(x);

        tmp /= r(0);
        tmp("&",xdim) = sqrt(r(1));

        Vector<gentype> tempres;

        kronpow(tempres,tmp,xdim+1,ic(0));

        res.castassign(tempres);
    }

    else
    {
        // Return as RKHS vector

        NiceThrow("Can't return a double RKHS vector (only gentype RKHS defined)");
    }
  }

    return dres;
}
*/


















// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================


void MercerKernel::K0i(gentype &res,
                     const gentype &xyprod,
                     int xdim, int resmode, int mlid, int indstart, int indend) const
{
    int startchain = 1;

    gentype Kxyres;
    gentype Kxyprev;

    gentype locdiffis;
    setzero(locdiffis);

    setzero(res);

    if ( indstart <= indend )
    {
        gentype diffis(0.0);

        if ( needsDiff() )
        {
            // Calculate ||x-y||^2 only as required

            diff0norm(diffis,xyprod);
        }

        int &q = indstart;

	for ( ; q <= indend ; ++q )
	{
            if ( isChained(q) )
            {
                NiceAssert( !( resmode & 0x80 ) );
                NiceAssert( !(kinf(q).usesVector) );
                NiceAssert( !(kinf(q).usesMinDiff) );
                NiceAssert( !(kinf(q).usesMaxDiff) );

                if ( startchain )
                {
                    startchain = 0;

                    K0(Kxyres,q,xyprod,diffis,0,xdim,resmode,mlid);
                }

                else
                {
                    NiceAssert( ( isAltDiff() <= 1 ) || ( isAltDiff() >= 100 ) );

                    qswap(Kxyprev,Kxyres);

                    //FIXME: check the Kyxres calculation here

                    K0(Kxyres,q,Kxyprev,locdiffis,1,xdim,resmode,mlid);
                }

                Kxyres *= cWeight(q);
            }

            else
            {
                if ( startchain )
                {
                    K0(Kxyres,q,xyprod,diffis,0,xdim,resmode,mlid);
                }

                else
                {
                    NiceAssert( !( resmode & 0x80 ) );
                    NiceAssert( ( isAltDiff() <= 1 ) || ( isAltDiff() >= 100 ) );
                    NiceAssert( !(kinf(q).usesVector) );
                    NiceAssert( !(kinf(q).usesMinDiff) );
                    NiceAssert( !(kinf(q).usesMaxDiff) );

                    startchain = 1;

                    qswap(Kxyprev,Kxyres);

                    K0(Kxyres,q,Kxyprev,locdiffis,1,xdim,resmode,mlid);
                }

                Kxyres *= cWeight(q);

                res += Kxyres;
            }

            //if ( isSplit(q) )
            //{
            //    ++q;
            //    break;
            //}
	}
    }

    return;
}

void MercerKernel::K2i(gentype &res, 
                     const gentype &xyprod, const gentype &yxprod,
                     const vecInfo &xinfo, const vecInfo &yinfo,
                     const gentype &xnorm, const gentype &ynorm,
                     const SparseVector<gentype> &x, const SparseVector<gentype> &y,
                     int i, int j,
                     int xdim, int adensetype, int bdensetype,
                     int resmode, int mlid, int indstart, int indend, int assumreal) const
{

    int startchain = 1;

    vecInfo Kxxresinfo,Kxxprevinfo;
    vecInfo Kyyresinfo,Kyyprevinfo;

    gentype Kxyres;
    gentype Kyxres;
    gentype &Kxxres = getmnorm(Kxxresinfo,x,2,0,assumreal);
    gentype &Kyyres = getmnorm(Kyyresinfo,y,2,0,assumreal);

    gentype Kxyprev;
    gentype Kyxprev;
    gentype &Kxxprev = getmnorm(Kxxprevinfo,x,2,0,assumreal);
    gentype &Kyyprev = getmnorm(Kxxprevinfo,x,2,0,assumreal);

    gentype locdiffis;
    setzero(locdiffis);

    //setzero(Kxyres);
    //setzero(Kyxres);
    //setzero(Kxxres);
    //setzero(Kyyres);

    //setzero(Kxyprev);
    //setzero(Kyxprev);
    //setzero(Kxxprev);
    //setzero(Kyyprev);

    setzero(res);

    if ( indstart <= indend )
    {
        gentype diffis(0.0);

        if ( needsDiff() )
        {
            // Calculate ||x-y||^2 only as required

            diff2norm(diffis,(xyprod+yxprod)/2.0,xnorm,ynorm);
        }

        int &q = indstart;

	for ( ; q <= indend ; ++q )
	{
            if ( isChained(q) )
            {
                NiceAssert( !( resmode & 0x80 ) );
                NiceAssert( !(kinf(q).usesVector) );
                NiceAssert( !(kinf(q).usesMinDiff) );
                NiceAssert( !(kinf(q).usesMaxDiff) );
                NiceAssert( !adensetype && !bdensetype );

                if ( startchain )
                {
                    startchain = 0;

                    K2(Kxyres,q,xyprod,yxprod,diffis,0,xinfo,yinfo,xnorm,ynorm,x,y,i,j,xdim,adensetype,bdensetype,resmode,mlid);
                    Kyxres = Kxyres;      //FIXME: clearly, this breaks anions
                    K2(Kxxres,q,xnorm ,xnorm ,locdiffis,1,xinfo,xinfo,xnorm,xnorm,x,x,i,i,xdim,adensetype,adensetype,resmode,mlid);
                    K2(Kyyres,q,ynorm ,ynorm ,locdiffis,1,yinfo,yinfo,ynorm,ynorm,y,y,j,j,xdim,bdensetype,bdensetype,resmode,mlid);
                }

                else
                {
                    NiceAssert( ( isAltDiff() <= 1 ) || ( isAltDiff() >= 100 ) );

                    qswap(Kxyprev,Kxyres);
                    qswap(Kyxprev,Kyxres);

                    qswap(Kxxprevinfo,Kxxresinfo);
                    qswap(Kyyprevinfo,Kyyresinfo);

                    //FIXME: check the Kyxres calculation here

                    K2(Kxyres,q,Kxyprev,Kyxprev,locdiffis,1,Kxxprevinfo,Kyyprevinfo,Kxxprev,Kyyprev,x,y,i,j,xdim,adensetype,bdensetype,resmode,mlid);
                    K2(Kyxres,q,Kyxprev,Kxyprev,locdiffis,1,Kyyprevinfo,Kxxprevinfo,Kyyprev,Kxxprev,y,x,j,i,xdim,bdensetype,adensetype,resmode,mlid);
                    K2(Kxxres,q,Kxxprev,Kxxprev,locdiffis,1,Kxxprevinfo,Kxxprevinfo,Kxxprev,Kxxprev,x,x,i,i,xdim,adensetype,adensetype,resmode,mlid);
                    K2(Kyyres,q,Kyyprev,Kyyprev,locdiffis,1,Kyyprevinfo,Kyyprevinfo,Kyyprev,Kyyprev,y,y,j,j,xdim,bdensetype,bdensetype,resmode,mlid);
                }

                Kxyres *= cWeight(q);
                Kyxres *= cWeight(q);
                Kxxres *= cWeight(q);
                Kyyres *= cWeight(q);
            }

            else
            {
                if ( startchain )
                {
                    K2(Kxyres,q,xyprod,yxprod,diffis,0,xinfo,yinfo,xnorm,ynorm,x,y,i,j,xdim,adensetype,bdensetype,resmode,mlid);
                }

                else
                {
                    NiceAssert( !( resmode & 0x80 ) );
                    NiceAssert( ( isAltDiff() <= 1 ) || ( isAltDiff() >= 100 ) );
                    NiceAssert( !(kinf(q).usesVector) );
                    NiceAssert( !(kinf(q).usesMinDiff) );
                    NiceAssert( !(kinf(q).usesMaxDiff) );
                    NiceAssert( !adensetype && !bdensetype );

                    startchain = 1;

                    qswap(Kxyprev,Kxyres);
                    qswap(Kyxprev,Kyxres);

                    qswap(Kxxprevinfo,Kxxresinfo);
                    qswap(Kyyprevinfo,Kyyresinfo);

                    K2(Kxyres,q,Kxyprev,Kyxprev,locdiffis,1,Kxxprevinfo,Kxxprevinfo,Kxxprev,Kyyprev,x,y,i,j,xdim,adensetype,bdensetype,resmode,mlid);
                }

                Kxyres *= cWeight(q);

                res += Kxyres;
            }

            //if ( isSplit(q) )
            //{
            //    ++q;
            //    break;
            //}
	}
    }

    return;
}

void MercerKernel::K4i(gentype &res, 
                      const gentype &xyprod, 
                      const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, 
                      const gentype &xanorm, const gentype &xbnorm, const gentype &xcnorm, const gentype &xdnorm, 
                      const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, 
                      int ia, int ib, int ic, int id, 
                      int xdim, int resmode, int mlid, 
                      double xy00, double xy10, double xy11, double xy20, double xy21, double xy22, double xy30, double xy31, double xy32, double xy33, const Vector<int> *s, int indstart, int indend, int assumreal) const
{
    int startchain = 1;

    vecInfo Kaaresinfo,Kaaprevinfo;
    vecInfo Kbbresinfo,Kbbprevinfo;
    vecInfo Kccresinfo,Kccprevinfo;
    vecInfo Kddresinfo,Kddprevinfo;

    gentype Kxyres;
    gentype &Kaares = getmnorm(Kaaresinfo,xa,4,0,assumreal);
    gentype &Kbbres = getmnorm(Kbbresinfo,xb,4,0,assumreal);
    gentype &Kccres = getmnorm(Kccresinfo,xc,4,0,assumreal);
    gentype &Kddres = getmnorm(Kddresinfo,xd,4,0,assumreal);

    gentype Kxyprev;
    gentype &Kaaprev = getmnorm(Kaaprevinfo,xa,4,0,assumreal);
    gentype &Kbbprev = getmnorm(Kbbprevinfo,xb,4,0,assumreal);
    gentype &Kccprev = getmnorm(Kccprevinfo,xc,4,0,assumreal);
    gentype &Kddprev = getmnorm(Kddprevinfo,xd,4,0,assumreal);

    gentype locdiffis;
    setzero(locdiffis);

    setzero(Kxyres);
    setzero(Kaares);
    setzero(Kbbres);
    setzero(Kccres);
    setzero(Kddres);

    setzero(Kxyprev);
    setzero(Kaaprev);
    setzero(Kbbprev);
    setzero(Kccprev);
    setzero(Kddprev);

    setzero(res);

    if ( indstart <= indend )
    {
        gentype diffis(0.0);

        if ( needsDiff() )
        {
            diff4norm(diffis,xyprod,xanorm,xbnorm,xcnorm,xdnorm,xy00,xy10,xy11,xy20,xy21,xy22,xy30,xy31,xy32,xy33,s);
        }

        int &q = indstart;

	for ( ; q <= indend ; ++q )
	{
            if ( isChained(q) )
            {
                NiceAssert( !( resmode & 0x80 ) );
                NiceAssert( !(kinf(q).usesVector) );
                NiceAssert( !(kinf(q).usesMinDiff) );
                NiceAssert( !(kinf(q).usesMaxDiff) );

                if ( startchain )
                {
                    startchain = 0;

                    K4(Kxyres,q,xyprod,diffis,0,xainfo,xbinfo,xcinfo,xdinfo,xanorm,xbnorm,xcnorm,xdnorm,xa,xb,xc,xd,ia,ib,ic,id,xdim,resmode,mlid,xy00,xy10,xy11,xy20,xy21,xy22,xy30,xy31,xy32,xy33,s);

                    K4(Kaares,q,xanorm,locdiffis,1,xainfo,xainfo,xainfo,xainfo,xanorm,xanorm,xanorm,xanorm,xa,xa,xa,xa,ia,ia,ia,ia,xdim,resmode,mlid,xy00,xy10,xy11,xy20,xy21,xy22,xy30,xy31,xy32,xy33,s);
                    K4(Kbbres,q,xbnorm,locdiffis,1,xbinfo,xbinfo,xbinfo,xbinfo,xbnorm,xbnorm,xbnorm,xbnorm,xb,xb,xb,xb,ib,ib,ib,ib,xdim,resmode,mlid,xy00,xy10,xy11,xy20,xy21,xy22,xy30,xy31,xy32,xy33,s);
                    K4(Kccres,q,xcnorm,locdiffis,1,xcinfo,xcinfo,xcinfo,xcinfo,xcnorm,xcnorm,xcnorm,xcnorm,xc,xc,xc,xc,ic,ic,ic,ic,xdim,resmode,mlid,xy00,xy10,xy11,xy20,xy21,xy22,xy30,xy31,xy32,xy33,s);
                    K4(Kddres,q,xdnorm,locdiffis,1,xdinfo,xdinfo,xdinfo,xdinfo,xdnorm,xdnorm,xdnorm,xdnorm,xd,xd,xd,xd,id,id,id,id,xdim,resmode,mlid,xy00,xy10,xy11,xy20,xy21,xy22,xy30,xy31,xy32,xy33,s);
                }

                else
                {
                    NiceAssert( ( isAltDiff() <= 1 ) || ( isAltDiff() >= 100 ) );
                    NiceAssert( !needsMatDiff() );

                    qswap(Kxyprev,Kxyres);

                    qswap(Kaaprevinfo,Kaaresinfo);
                    qswap(Kbbprevinfo,Kbbresinfo);
                    qswap(Kccprevinfo,Kccresinfo);
                    qswap(Kddprevinfo,Kddresinfo);

                    K4(Kxyres,q,Kxyprev,locdiffis,1,Kaaprevinfo,Kbbprevinfo,Kccprevinfo,Kddprevinfo,Kaaprev,Kbbprev,Kccprev,Kddprev,xa,xb,xc,xd,ia,ib,ic,id,xdim,resmode,mlid,xy00,xy10,xy11,xy20,xy21,xy22,xy30,xy31,xy32,xy33,s);
                    K4(Kaares,q,Kaaprev,locdiffis,1,Kaaprevinfo,Kaaprevinfo,Kaaprevinfo,Kaaprevinfo,Kaaprev,Kaaprev,Kaaprev,Kaaprev,xa,xa,xa,xa,ia,ia,ia,ia,xdim,resmode,mlid,xy00,xy10,xy11,xy20,xy21,xy22,xy30,xy31,xy32,xy33,s);
                    K4(Kbbres,q,Kbbprev,locdiffis,1,Kbbprevinfo,Kbbprevinfo,Kbbprevinfo,Kbbprevinfo,Kbbprev,Kbbprev,Kbbprev,Kbbprev,xb,xb,xb,xb,ib,ib,ib,ib,xdim,resmode,mlid,xy00,xy10,xy11,xy20,xy21,xy22,xy30,xy31,xy32,xy33,s);
                    K4(Kccres,q,Kccprev,locdiffis,1,Kccprevinfo,Kccprevinfo,Kccprevinfo,Kccprevinfo,Kccprev,Kccprev,Kccprev,Kccprev,xc,xc,xc,xc,ic,ic,ic,ic,xdim,resmode,mlid,xy00,xy10,xy11,xy20,xy21,xy22,xy30,xy31,xy32,xy33,s);
                    K4(Kddres,q,Kddprev,locdiffis,1,Kddprevinfo,Kddprevinfo,Kddprevinfo,Kddprevinfo,Kddprev,Kddprev,Kddprev,Kddprev,xd,xd,xd,xd,id,id,id,id,xdim,resmode,mlid,xy00,xy10,xy11,xy20,xy21,xy22,xy30,xy31,xy32,xy33,s);
                }

                Kxyres *= cWeight(q);
                Kaares *= cWeight(q);
                Kbbres *= cWeight(q);
                Kccres *= cWeight(q);
                Kddres *= cWeight(q);
            }

            else
            {
                if ( startchain )
                {
                    K4(Kxyres,q,xyprod,diffis,0,xainfo,xbinfo,xcinfo,xdinfo,xanorm,xbnorm,xcnorm,xdnorm,xa,xb,xc,xd,ia,ib,ic,id,xdim,resmode,mlid,xy00,xy10,xy11,xy20,xy21,xy22,xy30,xy31,xy32,xy33,s);
                }

                else
                {
                    NiceAssert( !( resmode & 0x80 ) );
                    NiceAssert( ( isAltDiff() <= 1 ) || ( isAltDiff() >= 100 ) );
                    NiceAssert( !needsMatDiff() );
                    NiceAssert( !(kinf(q).usesVector) );
                    NiceAssert( !(kinf(q).usesMinDiff) );
                    NiceAssert( !(kinf(q).usesMaxDiff) );

                    startchain = 1;

                    qswap(Kxyprev,Kxyres);

                    qswap(Kaaprevinfo,Kaaresinfo);
                    qswap(Kbbprevinfo,Kbbresinfo);
                    qswap(Kccprevinfo,Kccresinfo);
                    qswap(Kddprevinfo,Kddresinfo);

                    K4(Kxyres,q,Kxyprev,locdiffis,1,Kaaprevinfo,Kbbprevinfo,Kccprevinfo,Kddprevinfo,Kaaprev,Kbbprev,Kccprev,Kddprev,xa,xb,xc,xd,ia,ib,ic,id,xdim,resmode,mlid,xy00,xy10,xy11,xy20,xy21,xy22,xy30,xy31,xy32,xy33,s);
                }

                Kxyres *= cWeight(q);

                res += Kxyres;
            }

            //if ( isSplit(q) )
            //{
            //    ++q;
            //    break;
            //}
	}
    }

    return;
}

void MercerKernel::Kmi(gentype &res,
                      const gentype &xyprod,
                      Vector<const vecInfo *> &xinfo,
                      Vector<const gentype *> &xnorm,
                      Vector<const SparseVector<gentype> *> &x,
                      Vector<int> &i,
                      int xdim, int m, int resmode, int mlid,
                      const Matrix<double> &xy, const Vector<int> *s, int indstart, int indend, int assumreal) const
{
    int j;
    int startchain = 1;

    gentype Kxyres,Kxyprev;

    gentype locdiffis;
    setzero(locdiffis);

    Vector<vecInfo> Kxxresinfo;
    Vector<vecInfo> Kxxprevinfo;

    Vector<gentype> Kxxres;
    Vector<gentype> Kxxprev;

    Vector<gentype *> Kxxresp(xnorm.size());
    Vector<gentype *> Kxxprevp(xnorm.size());

    Vector<const vecInfo *> Kxxprevinfop(xnorm.size());
    Vector<const gentype *> Kxxprevpp(xnorm.size());

    Vector<const vecInfo *> xxinfo(xinfo);
    Vector<const gentype *> xxnorm(xnorm);
    Vector<const SparseVector<gentype> *> xx(x);
    Vector<int> ii(i);

    for ( j = 0 ; j < m ; ++j )
    {
        Kxxresp ("&",j) = &getmnorm(Kxxresinfo ("&",j),*(x(j)),m,0,assumreal);
        Kxxprevp("&",j) = &getmnorm(Kxxprevinfo("&",j),*(x(j)),m,0,assumreal);

        Kxxprevpp   ("&",j) =  (Kxxprevp   (j));
        Kxxprevinfop("&",j) = &(Kxxprevinfo(j));
    }

    setzero(res);

    if ( indstart <= indend )
    {
        gentype diffis(0.0);

        if ( needsDiff() )
        {
            // Calculate ||x-y||^2 only as required

            diffmnorm(m,diffis,xyprod,xnorm,xy,s);
        }

        int &q = indstart;

	for ( ; q <= indend ; ++q )
	{
            if ( isChained(q) )
            {
                NiceAssert( !( resmode & 0x80 ) );
                NiceAssert( !(kinf(q).usesVector) );
                NiceAssert( !(kinf(q).usesMinDiff) );
                NiceAssert( !(kinf(q).usesMaxDiff) );

                if ( startchain )
                {
                    startchain = 0;

                    Km(Kxyres,q,xyprod,diffis,0,xinfo,xnorm,x,i,xdim,m,resmode,mlid,xy,s);

                    for ( j = 0 ; j < m ; ++j )
                    {
                        xxinfo = xinfo(j);
                        xxnorm = xnorm(j);
                        xx     = x(j);
                        ii     = i(j);

                        Km(Kxxres("&",j),q,*(xnorm(j)),locdiffis,1,xxinfo,xxnorm,xx,ii,xdim,m,resmode,mlid,xy,s);
                    }
                }

                else
                {
                    NiceAssert( ( isAltDiff() <= 1 ) || ( isAltDiff() >= 100 ) );
                    NiceAssert( !needsMatDiff() );

                    qswap(Kxyres,Kxyprev);

                    qswap(Kxxresinfo,Kxxprevinfo);

                    Km(Kxyres,q,Kxyprev,locdiffis,1,Kxxprevinfop,Kxxprevpp,x,i,xdim,m,resmode,mlid,xy,s);

                    for ( j = 0 ; j < m ; ++j )
                    {
                        xxinfo = &Kxxprevinfo(j);
                        xxnorm = Kxxprevp(j);
                        xx     = x(j);
                        ii     = i(j);

                        Km(Kxxres("&",j),q,*(Kxxprevp(j)),locdiffis,1,xxinfo,xxnorm,xx,ii,xdim,m,resmode,mlid,xy,s);
                    }
                }

                Kxyres *= cWeight(q);

                for ( j = 0 ; j < m ; ++j )
                {
                    Kxxres("&",j) *= cWeight(q);
                }
            }

            else
            {
                if ( startchain )
                {
                    Km(Kxyres,q,xyprod,diffis,0,xinfo,xnorm,x,i,xdim,m,resmode,mlid,xy,s);
                }

                else
                {
                    NiceAssert( !( resmode & 0x80 ) );
                    NiceAssert( ( isAltDiff() <= 1 ) || ( isAltDiff() >= 100 ) );
                    NiceAssert( !needsMatDiff() );
                    NiceAssert( !(kinf(q).usesVector) );
                    NiceAssert( !(kinf(q).usesMinDiff) );
                    NiceAssert( !(kinf(q).usesMaxDiff) );

                    startchain = 1;

                    qswap(Kxyres,Kxyprev);

                    qswap(Kxxresinfo,Kxxprevinfo);

                    Km(Kxyres,q,Kxyprev,locdiffis,1,Kxxprevinfop,Kxxprevpp,x,i,xdim,m,resmode,mlid,xy,s);
                }

                Kxyres *= cWeight(q);

                res += Kxyres;
            }

            //if ( isSplit(q) )
            //{
            //    ++q;
            //    break;
            //}
	}
    }

    return;
}




















// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================

void MercerKernel::K0(gentype &res, int q,
                     const gentype &xyprod, gentype &diffis, int recalcdiffis,
                     int xdim, int resmode, int mlid) const
{
    NiceAssert( q >= 0 );
    NiceAssert( q < size() );

    K0unnorm(res,q,xyprod,diffis,recalcdiffis,xdim,resmode,mlid);

    return;
}

void MercerKernel::K2(gentype &res, int q,
                     const gentype &xyprod, const gentype &yxprod, gentype &diffis, int recalcdiffis,
                     const vecInfo &xinfo, const vecInfo &yinfo, const gentype &xnorm, const gentype &ynorm,
                     const SparseVector<gentype> &x, const SparseVector<gentype> &y,
                     int i, int j,
                     int xdim, int adensetype, int bdensetype, int resmode, int mlid) const
{
    NiceAssert( ( !adensetype && !bdensetype ) || !isNormalised(q) );

    NiceAssert( q >= 0 );
    NiceAssert( q < size() );

    K2unnorm(res,q,xyprod,yxprod,diffis,recalcdiffis,xinfo,yinfo,xnorm,ynorm,x,y,i,j,xdim,adensetype,bdensetype,resmode,mlid);

    if ( isNormalised(q) )
    {
        NiceAssert( !( resmode & 0x80 ) );

        gentype xkern,ykern;
        gentype locdiffis(0.0);

        K2unnorm(xkern,q,xnorm,xnorm,locdiffis,0,xinfo,xinfo,xnorm,xnorm,x,x,i,i,xdim,adensetype,adensetype,resmode,mlid);
        K2unnorm(ykern,q,ynorm,ynorm,locdiffis,0,yinfo,yinfo,ynorm,ynorm,y,y,j,j,xdim,bdensetype,bdensetype,resmode,mlid);

        if ( ( (double) abs2(xkern) <= BADZEROTOL ) || ( (double) abs2(ykern) <= BADZEROTOL ) )
        {
            res = angle(res);
            res = ( (double) abs2(res) <= BADZEROTOL ) ? 1.0 : res;
        }

        else
        {
            OP_sqrt(xkern);
            OP_sqrt(ykern);

            xkern.inverse();
            ykern.inverse();

            // Following method used to allow for matrix-valued kernels.
            //
            // inv(sqrt(K(x,x)))*K(x,y)*inv(sqrt(K(y,y)))

            rightmult(xkern,res);
            leftmult (res,ykern);

            if ( testisvnan(res) )
            {
                res = 1.0;
            }
        }
    }

    return;
}

void MercerKernel::K4(gentype &res, int q,
                      const gentype &xyprod, gentype &diffis, int recalcdiffis,
                      const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
                      const gentype &xanorm, const gentype &xbnorm, const gentype &xcnorm, const gentype &xdnorm,
                      const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                      int ia, int ib, int ic, int id,
                      int xdim, int resmode, int mlid,
                      double xy00, double xy10, double xy11, double xy20, double xy21, double xy22, double xy30, double xy31, double xy32, double xy33, const Vector<int> *s) const
{
    NiceAssert( q >= 0 );
    NiceAssert( q < size() );

    K4unnorm(res,q,xyprod,diffis,recalcdiffis,xainfo,xbinfo,xcinfo,xdinfo,xanorm,xbnorm,xcnorm,xdnorm,xa,xb,xc,xd,ia,ib,ic,id,xdim,resmode,mlid,xy00,xy10,xy11,xy20,xy21,xy22,xy30,xy31,xy32,xy33,s);

    if ( isNormalised(q) )
    {
        NiceAssert( !( resmode & 0x80 ) );

        gentype xakern,xbkern,xckern,xdkern;
        gentype locdiffis(0.0);

        retVector<int> tmpva;

        K4unnorm(xakern,q,xanorm,locdiffis,1,xainfo,xainfo,xainfo,xainfo,xanorm,xanorm,xanorm,xanorm,xa,xa,xa,xa,ia,ia,ia,ia,xdim,resmode,mlid,xy00,xy00,xy00,xy00,xy00,xy00,xy00,xy00,xy00,xy00,s);
        K4unnorm(xbkern,q,xbnorm,locdiffis,1,xbinfo,xbinfo,xbinfo,xbinfo,xbnorm,xbnorm,xbnorm,xbnorm,xb,xb,xb,xb,ib,ib,ib,ib,xdim,resmode,mlid,xy11,xy11,xy11,xy11,xy11,xy11,xy11,xy11,xy11,xy11,s);
        K4unnorm(xckern,q,xcnorm,locdiffis,1,xcinfo,xcinfo,xcinfo,xcinfo,xcnorm,xcnorm,xcnorm,xcnorm,xc,xc,xc,xc,ic,ic,ic,ic,xdim,resmode,mlid,xy22,xy22,xy22,xy22,xy22,xy22,xy22,xy22,xy22,xy22,s);
        K4unnorm(xdkern,q,xdnorm,locdiffis,1,xdinfo,xdinfo,xdinfo,xdinfo,xdnorm,xdnorm,xdnorm,xdnorm,xd,xd,xd,xd,id,id,id,id,xdim,resmode,mlid,xy33,xy33,xy33,xy33,xy33,xy33,xy33,xy33,xy33,xy33,s);

//FIXME: need to allow for zero norms
        OP_sqrt(xakern);
        OP_sqrt(xbkern);
        OP_sqrt(xckern);
        OP_sqrt(xdkern);

        OP_sqrt(xakern);
        OP_sqrt(xbkern);
        OP_sqrt(xckern);
        OP_sqrt(xdkern);

        xakern.inverse();
        xbkern.inverse();
        xckern.inverse();
        xdkern.inverse();

        res *= xakern;
        res *= xbkern;
        res *= xckern;
        res *= xdkern;

        if ( testisvnan(res) )
        {
            res = 1.0;
        }
    }

    return;
}


void MercerKernel::Km(gentype &res, int q,
                      const gentype &xyprod, gentype &diffis, int recalcdiffis,
                      Vector<const vecInfo *> &xinfo,
                      Vector<const gentype *> &xnorm,
                      Vector<const SparseVector<gentype> *> &x,
                      Vector<int> &i,
                      int xdim, int m, int resmode, int mlid,
                      const Matrix<double> &xy, const Vector<int> *s) const
{
    NiceAssert( ( xnorm.size() > 0 ) && !((xnorm.size())%2) );
    NiceAssert( q >= 0 );
    NiceAssert( q < size() );

    Kmunnorm(res,q,xyprod,diffis,recalcdiffis,xinfo,xnorm,x,i,xdim,m,resmode,mlid,xy,s);

    if ( isNormalised(q) )
    {
        NiceAssert( !( resmode & 0x80 ) );

	int j;
        gentype xxprod;
        gentype xkern;
        gentype locdiffis;
        Vector<const vecInfo *> xxinfo(xinfo);
        Vector<const gentype *> xxnorm(xnorm);
        Vector<const SparseVector<gentype> *> xx(x);
        Vector<int> ii(i);

        setzero(locdiffis);

//FIXME: need to allow for zero norms
        retMatrix<double> tmpma;
        retVector<int> tmpva;

	for ( j = 0 ; j < m ; ++j )
	{
            xxinfo = xinfo(j);
            xxnorm = xnorm(j);
            xx     = x(j);
            ii     = i(j);

            xxprod = *(xnorm(j));

            Kmunnorm(xkern,q,xxprod,locdiffis,1,xxinfo,xxnorm,xx,ii,xdim,m,resmode,mlid,xy(ii*oneintvec(( m <= xy.numRows() ) ? m : xy.numRows(),tmpva),ii*oneintvec(( m <= xy.numCols() ) ? m : xy.numCols(),tmpva),tmpma),s);

            gentype oneonm(1.0/m);

            res /= pow(abs2(xkern),oneonm);
	}

        if ( testisvnan(res) )
        {
            res = 1.0;
        }
    }

    return;
}





































// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================

void MercerKernel::K0unnorm(gentype &res, int q,
                            const gentype &xyprod, gentype &diffis, int recalcdiffis,
                            int xdim, int resmode, int mlid) const
{
    if ( recalcdiffis && needsDiff(q) )
    {
        // Calculate ||x-y||^2 only as required

        diff0norm(diffis,xyprod);
    }

    Vector<const vecInfo *> xxinfo(0);
    Vector<const SparseVector<gentype> *> xx(0);
    Vector<const gentype *> xxnorm(0);
    Vector<int> ii(0);

    return Kbase(res,q,cType(q),
                 xyprod,xyprod,diffis,
                 xx,
                 xxinfo,
                 xxnorm,
                 ii,
                 xdim,0,0,0,resmode,mlid);
}

void MercerKernel::K2unnorm(gentype &res, int q,
                           const gentype &xyprod, const gentype &yxprod, gentype &diffis, int recalcdiffis,
                           const vecInfo &xinfo, const vecInfo &yinfo,
                           const gentype &xnorm, const gentype &ynorm,
                           const SparseVector<gentype> &x, const SparseVector<gentype> &y,
                           int i, int j,
                           int xdim, int adensetype, int bdensetype, int resmode, int mlid) const
{
    if ( recalcdiffis && needsDiff(q) )
    {
        // Calculate ||x-y||^2 only as required

        diff2norm(diffis,(xyprod+yxprod)/2.0,xnorm,ynorm);
    }

    Vector<const vecInfo *> xxinfo(2);
    Vector<const SparseVector<gentype> *> xx(2);
    Vector<const gentype *> xxnorm(2);
    Vector<int> ii(2);

    xxinfo("&",0) = &xinfo;
    xxinfo("&",1) = &yinfo;

    xx("&",0) = &x;
    xx("&",1) = &y;

    xxnorm("&",0) = &xnorm;
    xxnorm("&",1) = &ynorm;

    ii("&",0) = i;
    ii("&",1) = j;

    return Kbase(res,q,cType(q),
                 xyprod,yxprod,diffis,
                 xx,
                 xxinfo,
                 xxnorm,
                 ii,
                 xdim,2,adensetype,bdensetype,resmode,mlid);
}

void MercerKernel::K4unnorm(gentype &res, int q,
                            const gentype &xyprod, gentype &diffis, int recalcdiffis,
                            const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
                            const gentype &xanorm, const gentype &xbnorm, const gentype &xcnorm, const gentype &xdnorm,
                            const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                            int ia, int ib, int ic, int id,
                            int xdim, int resmode, int mlid,
                            double xy00, double xy10, double xy11, double xy20, double xy21, double xy22, double xy30, double xy31, double xy32, double xy33, const Vector<int> *s) const
{
    if ( recalcdiffis && needsDiff(q) )
    {
        // Calculate ||x-y||^2 only as required

        diff4norm(diffis,xyprod,xanorm,xbnorm,xcnorm,xdnorm,xy00,xy10,xy11,xy20,xy21,xy22,xy30,xy31,xy32,xy33,s);
    }

    Vector<const vecInfo *> xxinfo(4);
    Vector<const SparseVector<gentype> *> xx(4);
    Vector<const gentype *> xxnorm(4);
    Vector<int> ii(4);

    xxinfo("&",0) = &xainfo;
    xxinfo("&",1) = &xbinfo;
    xxinfo("&",2) = &xcinfo;
    xxinfo("&",3) = &xdinfo;

    xx("&",0) = &xa;
    xx("&",1) = &xb;
    xx("&",2) = &xc;
    xx("&",3) = &xd;

    xxnorm("&",0) = &xanorm;
    xxnorm("&",1) = &xbnorm;
    xxnorm("&",2) = &xcnorm;
    xxnorm("&",3) = &xdnorm;

    ii("&",0) = ia;
    ii("&",1) = ib;
    ii("&",2) = ic;
    ii("&",3) = id;

    return Kbase(res,q,cType(q),
                 xyprod,xyprod,diffis,
                 xx,
                 xxinfo,
                 xxnorm,
                 ii,
                 xdim,4,0,0,resmode,mlid);
}

void MercerKernel::Kmunnorm(gentype &res, int q,
                            const gentype &xyprod, gentype &diffis, int recalcdiffis,
                            Vector<const vecInfo *> &xinfo,
                            Vector<const gentype *> &xnorm,
                            Vector<const SparseVector<gentype> *> &x,
                            Vector<int> &ii,
                            int xdim, int m, int resmode, int mlid,
                            const Matrix<double> &xy, const Vector<int> *s) const
{
    if ( recalcdiffis && needsDiff(q) )
    {
        // Calculate ||x-y||^2 only as required

        diffmnorm(m,diffis,xyprod,xnorm,xy,s);
    }

    Kbase(res,q,cType(q),
          xyprod,xyprod,diffis,
          x,
          xinfo,
          xnorm,
          ii,
          xdim,m,0,0,resmode,mlid);

    return;
}




































// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
































// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================








































// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================

// Evaluate kernel gradient dK/dx(x,y) and dK/dy(x,y)

// (inc double versions for speed)


















// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================



//phantomx
void MercerKernel::dKdaz(gentype &resda, gentype &resdz, int &minmaxind, 
                         const gentype &xyprod, const gentype &yxprod,
                         const vecInfo &xinfo, const vecInfo &yinfo, 
                         const gentype &xnorm, const gentype &ynorm, 
                         const SparseVector<gentype> &x, const SparseVector<gentype> &y, 
                         int i, int j, 
                         int xdim, int mlid, int assumreal) const
{
    minmaxind = -2;

    if ( isKernelDerivativeEasy() )
    {
        // Assume any kernel can be written as:
        //
        // K(x,y) = K(a,z,b)
        //
        // where a = ||x||^2
        //       b = ||y||^2
        //       z = x'y
        //
        // dK/dx = dK/da da/dx + dK/db db/dx + dK/dz dz/dx
        //       = dK/da 2x + dK/dz y

        gentype diffis(0.0);

        if ( kinf(0).numflagsset() == 0 )
        {
            // dK/dx = 0

            resdz = 0.0;
            resda = 0.0;

            minmaxind = -1;
        }

        else if ( needsDiff(0) )
        {
            // dK/dx = dK/da 2x + dK/dz y
            // 
            // But in this case we can simplify.  Note (see dKdaBase) that
            // dK/da = -1/2 dK/dz, we see that:
            //
            // dK/dx = -dK/dz x + dK/dz y
            //       = dK/dz (y-x)
            //
            // which implies the following quicker code

            diff2norm(diffis,(xyprod+yxprod)/2.0,xnorm,ynorm);

            Vector<const SparseVector<gentype> *> xx(2);
            Vector<const vecInfo *> xxinfo(2);
            Vector<const gentype *> xxnorm(2);
            Vector<int> ii(2);

            xx("&",0)     = &x;
            xxinfo("&",0) = &xinfo;
            xxnorm("&",0) = &xnorm;
            ii("&",0)     = i;

            xx("&",1)     = &y;
            xxinfo("&",1) = &yinfo;
            xxnorm("&",1) = &ynorm;
            ii("&",1)     = i;

            dKdzBase(resdz,minmaxind,0,xyprod,yxprod,diffis,xx,xxinfo,xxnorm,ii,xdim,2,mlid);
            resdz *= cWeight(0);

            resda  = resdz;
            resda *= -0.5;
        }

        else if ( needsInner(0,2) )
        {
            NiceAssert( kinf(0).usesInner || kinf(0).usesMinDiff || kinf(0).usesMaxDiff );

            // dK/dx = dK/dz y

            Vector<const SparseVector<gentype> *> xx(2);
            Vector<const vecInfo *> xxinfo(2);
            Vector<const gentype *> xxnorm(2);
            Vector<int> ii(2);

            xx("&",0)     = &x;
            xxinfo("&",0) = &xinfo;
            xxnorm("&",0) = &xnorm;
            ii("&",0)     = i;

            xx("&",1)     = &y;
            xxinfo("&",1) = &yinfo;
            xxnorm("&",1) = &ynorm;
            ii("&",1)     = i;

            dKdzBase(resdz,minmaxind,0,xyprod,yxprod,diffis,xx,xxinfo,xxnorm,ii,xdim,2,mlid);
            resdz *= cWeight(0);

            resda = 0.0;
        }

        else
        {
            resdz = 0.0;
            resda = 0.0;

            minmaxind = -1;
        }

        return;
    }

    // K(x,y) = K(a,z,b)
    //
    // where:
    //
    // a = x'x
    // z = x'y    (note this breaks for non-real kernels)
    // b = y'y
    //
    // we first work out gradients wrt a, b and z, taking into account
    // chaining etc.  Only once this is done do we convert to x and y
    // scales, noting that
    //
    // da/dx = 2x
    // dz/dx = y
    // db/dx = 0

    int startchain = 1;

    vecInfo Kxxresinfo,Kxxprevinfo;
    vecInfo Kyyresinfo,Kyyprevinfo;

    gentype Kxyres;
    gentype Kyxres;
    gentype &Kxxres = getmnorm(Kxxresinfo,x,2,0,assumreal);
    gentype &Kyyres = getmnorm(Kyyresinfo,y,2,0,assumreal);

    gentype Kxyprev;
    gentype Kyxprev;
    gentype &Kxxprev = getmnorm(Kxxprevinfo,x,2,0,assumreal);
    gentype &Kyyprev = getmnorm(Kxxprevinfo,y,2,0,assumreal);

    setzero(Kxyres);
    setzero(Kyxres);
    setzero(Kxxres);
    setzero(Kyyres);

    setzero(Kxyprev);
    setzero(Kyxprev);
    setzero(Kxxprev);
    setzero(Kyyprev);

    gentype dKada(0.0);
    gentype dKadz(0.0);
    gentype dKadb(0.0);
    gentype dKzda(0.0);
    gentype dKzdz(0.0);
    gentype dKzdb(0.0);
    gentype dKbda(0.0);
    gentype dKbdz(0.0);
    gentype dKbdb(0.0);

    gentype dKathisda(0.0);
    gentype dKathisdz(0.0);
    gentype dKathisdb(0.0);
    gentype dKzthisda(0.0);
    gentype dKzthisdz(0.0);
    gentype dKzthisdb(0.0);
    gentype dKbthisda(0.0);
    gentype dKbthisdz(0.0);
    gentype dKbthisdb(0.0);

    gentype dKamidda(0.0);
    gentype dKamiddz(0.0);
    gentype dKamiddb(0.0);
    gentype dKzmidda(0.0);
    gentype dKzmiddz(0.0);
    gentype dKzmiddb(0.0);
    gentype dKbmidda(0.0);
    gentype dKbmiddz(0.0);
    gentype dKbmiddb(0.0);

    gentype dKaprevda(0.0);
    gentype dKaprevdz(0.0);
    gentype dKaprevdb(0.0);
    gentype dKzprevda(0.0);
    gentype dKzprevdz(0.0);
    gentype dKzprevdb(0.0);
    gentype dKbprevda(0.0);
    gentype dKbprevdz(0.0);
    gentype dKbprevdb(0.0);

    gentype resdKda(0.0);
    gentype resdKdz(0.0);
    gentype resdKdb(0.0);

    gentype locdiffis(0.0);

    if ( size() )
    {
        gentype diffis(0.0);

        if ( needsDiff() )
        {
            // Calculate ||x-y||^2 only as required

            diff2norm(diffis,(xyprod+yxprod)/2.0,xnorm,ynorm);
        }

        int q;

	for ( q = 0 ; q <= size() ; ++q )
	{
            if ( isChained(q) )
            {
                NiceAssert( ( isAltDiff() <= 1 ) || ( isAltDiff() >= 100 ) );
                NiceAssert( !(kinf(q).usesVector) );
                NiceAssert( !(kinf(q).usesMinDiff) );
                NiceAssert( !(kinf(q).usesMaxDiff) );

                if ( startchain )
                {
                    startchain = 0;

                    // Calculate gradients of this layer
                    //
                    // Note dkdb is replaced with dkda, arguments reversed if needed (symmetry assumed)

                    dKda(dKathisda,minmaxind,q,xnorm ,xnorm ,locdiffis,1,xinfo,xinfo,xnorm,xnorm,x,x,i,i,xdim,mlid);
                    dKathisdz = 0.0;
                    dKathisdb = 0.0;
                    dKda(dKzthisda,minmaxind,q,xyprod,yxprod,locdiffis,1,xinfo,yinfo,xnorm,ynorm,x,y,i,j,xdim,mlid);
                    dKdz(dKzthisdz,minmaxind,q,xyprod,yxprod,locdiffis,1,xinfo,yinfo,xnorm,ynorm,x,y,i,j,xdim,mlid);
                    dKda(dKzthisdb,minmaxind,q,yxprod,xyprod,locdiffis,1,yinfo,xinfo,ynorm,xnorm,y,x,j,i,xdim,mlid); // dKdb, arguments reversed
                    dKbthisda = 0.0;
                    dKbthisdz = 0.0;
                    dKda(dKbthisdb,minmaxind,q,ynorm ,ynorm ,locdiffis,1,yinfo,yinfo,ynorm,ynorm,y,y,j,j,xdim,mlid);

                    // Update norms

                    K2(Kxyres,q,xyprod,yxprod,locdiffis,1,xinfo,yinfo,xnorm,ynorm,x,y,i,j,xdim,0,0,0,mlid);
                    Kyxres = Kxyres;      //FIXME: clearly, this breaks anions
                    K2(Kxxres,q,xnorm ,xnorm ,locdiffis,1,xinfo,xinfo,xnorm,xnorm,x,x,i,i,xdim,0,0,0,mlid);
                    K2(Kyyres,q,ynorm ,ynorm ,locdiffis,1,yinfo,yinfo,ynorm,ynorm,y,y,j,j,xdim,0,0,0,mlid);
                }

                else
                {
                    // Not start of chain, previous kernel exists, so inner
                    // product (and hence inner product gradients) are
                    // inherited from it

                    qswap(dKaprevda,dKathisda);
                    qswap(dKaprevdz,dKathisdz);
                    qswap(dKaprevdb,dKathisdb);
                    qswap(dKzprevda,dKzthisda);
                    qswap(dKzprevdz,dKzthisdz);
                    qswap(dKzprevdb,dKzthisdb);
                    qswap(dKbprevda,dKbthisda);
                    qswap(dKbprevdz,dKbthisdz);
                    qswap(dKbprevdb,dKbthisdb);

                    qswap(Kxyprev,Kxyres);
                    qswap(Kyxprev,Kyxres);

                    qswap(Kxxprevinfo,Kxxresinfo);
                    qswap(Kyyprevinfo,Kyyresinfo);

                    // Calculate gradients of this layer
                    //
                    // Note dkdb is replaced with dkda, arguments reversed if needed (symmetry assumed)

                    dKda(dKamidda,minmaxind,q,Kxxprev,Kxxprev,locdiffis,1,Kxxprevinfo,Kxxprevinfo,Kxxprev,Kxxprev,x,x,i,i,xdim,mlid);
                    dKdz(dKamiddz,minmaxind,q,Kxxprev,Kxxprev,locdiffis,1,Kxxprevinfo,Kxxprevinfo,Kxxprev,Kxxprev,x,x,i,i,xdim,mlid);
                    dKamiddb = dKamidda;
                    dKda(dKzmidda,minmaxind,q,Kxyprev,Kyxprev,locdiffis,1,Kxxprevinfo,Kyyprevinfo,Kxxprev,Kyyprev,x,y,i,j,xdim,mlid);
                    dKdz(dKzmiddz,minmaxind,q,Kxyprev,Kyxprev,locdiffis,1,Kxxprevinfo,Kyyprevinfo,Kxxprev,Kyyprev,x,y,i,j,xdim,mlid);
                    dKda(dKzmiddb,minmaxind,q,Kyxprev,Kxyprev,locdiffis,1,Kyyprevinfo,Kxxprevinfo,Kyyprev,Kxxprev,y,x,j,i,xdim,mlid); // dKdb, arguments reversed
                    dKda(dKbmidda,minmaxind,q,Kyyprev,Kyyprev,locdiffis,1,Kyyprevinfo,Kyyprevinfo,Kyyprev,Kyyprev,y,y,j,j,xdim,mlid);
                    dKdz(dKbmiddz,minmaxind,q,Kyyprev,Kyyprev,locdiffis,1,Kyyprevinfo,Kyyprevinfo,Kyyprev,Kyyprev,y,y,j,j,xdim,mlid);
                    dKbmiddb = dKbmidda;

                    // Gradient chaining

                    dKathisda = (dKamidda*dKaprevda);
                    dKathisdz = (dKamidda*dKaprevdz);
                    dKathisdb = (dKamidda*dKaprevdb);
                    dKzthisda = (dKzmidda*dKaprevda);
                    dKzthisdz = (dKzmidda*dKaprevdz);
                    dKzthisdb = (dKzmidda*dKaprevdb);
                    dKbthisda = (dKbmidda*dKaprevda);
                    dKbthisdz = (dKbmidda*dKaprevdz);
                    dKbthisdb = (dKbmidda*dKaprevdb);

                    dKathisda += (dKamiddz*dKzprevda);
                    dKathisdz += (dKamiddz*dKzprevdz);
                    dKathisdb += (dKamiddz*dKzprevdb);
                    dKzthisda += (dKzmiddz*dKzprevda);
                    dKzthisdz += (dKzmiddz*dKzprevdz);
                    dKzthisdb += (dKzmiddz*dKzprevdb);
                    dKbthisda += (dKbmiddz*dKzprevda);
                    dKbthisdz += (dKbmiddz*dKzprevdz);
                    dKbthisdb += (dKbmiddz*dKzprevdb);

                    dKathisda += (dKamiddb*dKbprevda);
                    dKathisdz += (dKamiddb*dKbprevdz);
                    dKathisdb += (dKamiddb*dKbprevdb);
                    dKzthisda += (dKzmiddb*dKbprevda);
                    dKzthisdz += (dKzmiddb*dKbprevdz);
                    dKzthisdb += (dKzmiddb*dKbprevdb);
                    dKbthisda += (dKbmiddb*dKbprevda);
                    dKbthisdz += (dKbmiddb*dKbprevdz);
                    dKbthisdb += (dKbmiddb*dKbprevdb);

                    // Update norms

                    K2(Kxyres,q,Kxyprev,Kyxprev,locdiffis,1,Kxxprevinfo,Kyyprevinfo,Kxxprev,Kyyprev,x,y,i,j,xdim,0,0,0,mlid);
                    K2(Kyxres,q,Kyxprev,Kxyprev,locdiffis,1,Kyyprevinfo,Kxxprevinfo,Kyyprev,Kxxprev,y,x,j,i,xdim,0,0,0,mlid);
                    K2(Kxxres,q,Kxxprev,Kxxprev,locdiffis,1,Kxxprevinfo,Kxxprevinfo,Kxxprev,Kxxprev,x,x,i,i,xdim,0,0,0,mlid);
                    K2(Kyyres,q,Kyyprev,Kyyprev,locdiffis,1,Kyyprevinfo,Kyyprevinfo,Kyyprev,Kyyprev,y,y,j,j,xdim,0,0,0,mlid);
                }

                // Apply weights to gradients of this layer

                Kxyres *= cWeight(q);
                Kyxres *= cWeight(q);
                Kxxres *= cWeight(q);
                Kyyres *= cWeight(q);

                dKathisda *= cWeight(q);
                dKzthisda *= cWeight(q);
                dKbthisda *= cWeight(q);

                dKathisdz *= cWeight(q);
                dKzthisdz *= cWeight(q);
                dKbthisdz *= cWeight(q);

                dKathisdb *= cWeight(q);
                dKzthisdb *= cWeight(q);
                dKbthisdb *= cWeight(q);
            }

            else
            {
                if ( startchain )
                {
                    // Calculate gradients of this layer

                    dKda(dKzthisda,minmaxind,q,xyprod,yxprod,diffis,0,xinfo,yinfo,xnorm,ynorm,x,y,i,j,xdim,mlid);
                    dKdz(dKzthisdz,minmaxind,q,xyprod,yxprod,diffis,0,xinfo,yinfo,xnorm,ynorm,x,y,i,j,xdim,mlid);
                    dKda(dKzthisdb,minmaxind,q,yxprod,xyprod,diffis,0,yinfo,xinfo,ynorm,xnorm,y,x,j,i,xdim,mlid); // dKdb, arguments reversed
                }


                else
                {
                    NiceAssert( ( isAltDiff() <= 1 ) || ( isAltDiff() >= 100 ) );
                    NiceAssert( !(kinf(q).usesVector) );
                    NiceAssert( !(kinf(q).usesMinDiff) );
                    NiceAssert( !(kinf(q).usesMaxDiff) );

                    // End of chain: add gradients to result now 

                    startchain = 1;

                    // Not start of chain, previous kernel exists, so inner
                    // product (and hence inner product gradients) are
                    // inherited from it

                    qswap(dKaprevda,dKathisda);
                    qswap(dKaprevdz,dKathisdz);
                    qswap(dKaprevdb,dKathisdb);
                    qswap(dKzprevda,dKzthisda);
                    qswap(dKzprevdz,dKzthisdz);
                    qswap(dKzprevdb,dKzthisdb);
                    qswap(dKbprevda,dKbthisda);
                    qswap(dKbprevdz,dKbthisdz);
                    qswap(dKbprevdb,dKbthisdb);

                    // Calculate gradients of this layer
                    //
                    // Note dkdb is replaced with dkda, arguments reversed if needed (symmetry assumed)

                    dKda(dKzmidda,minmaxind,q,Kxyprev,Kyxprev,locdiffis,1,Kxxprevinfo,Kyyprevinfo,Kxxprev,Kyyprev,x,y,i,j,xdim,mlid);
                    dKdz(dKzmiddz,minmaxind,q,Kxyprev,Kyxprev,locdiffis,1,Kxxprevinfo,Kyyprevinfo,Kxxprev,Kyyprev,x,y,i,j,xdim,mlid);
                    dKda(dKzmiddb,minmaxind,q,Kyxprev,Kxyprev,locdiffis,1,Kyyprevinfo,Kxxprevinfo,Kyyprev,Kxxprev,y,x,j,i,xdim,mlid); // dKdb, arguments reversed

                    // Gradient chaining

                    dKzthisda = (dKzmidda*dKaprevda);
                    dKzthisdz = (dKzmidda*dKaprevdz);
                    dKzthisdb = (dKzmidda*dKaprevdb);

                    dKzthisda += (dKzmiddz*dKzprevda);
                    dKzthisdz += (dKzmiddz*dKzprevdz);
                    dKzthisdb += (dKzmiddz*dKzprevdb);

                    dKzthisda += (dKzmiddb*dKbprevda);
                    dKzthisdz += (dKzmiddb*dKbprevdz);
                    dKzthisdb += (dKzmiddb*dKbprevdb);
                }

                dKathisdz *= cWeight(q);
                dKzthisdz *= cWeight(q);
                dKbthisdz *= cWeight(q);

                resdKda += dKzthisda;
                resdKdz += dKzthisdz;
                resdKdb += dKzthisdb;
            }

            //if ( isSplit(q) )
            //{
            //    break;
            //}
	}
    }

    resda = resdKda;
    resdz = resdKdz;

    return;
}








































// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================

void MercerKernel::dKda(gentype &res, int &minmaxind, int q, 
                        const gentype &xyprod, const gentype &yxprod, gentype &diffis, int recalcdiffis,
                        const vecInfo &xinfo, const vecInfo &yinfo, 
                        const gentype &xnorm, const gentype &ynorm, 
                        const SparseVector<gentype> &x, const SparseVector<gentype> &y, 
                        int i, int j, 
                        int xdim, int mlid) const
{
    NiceAssert( q >= 0 );
    NiceAssert( q < size() );

    if ( !isNormalised(q) )
    {
        dKunnormda(res,minmaxind,q,xyprod,yxprod,diffis,recalcdiffis,xinfo,yinfo,xnorm,ynorm,x,y,i,j,xdim,mlid);
    }

    else
    {
        gentype Kazb;
        gentype Kaaa;

        gentype dKda_azb;
        gentype dKda_aaa;
        gentype dKdz_aaa;

        gentype temp;

        double scalefact = 1.0;

        K2unnorm(Kazb,q,xyprod,yxprod,diffis,recalcdiffis,xinfo,yinfo,xnorm,ynorm,x,y,i,j,xdim,0,0,0,mlid);
        dKunnormda(dKda_azb,minmaxind,q,xyprod,yxprod,diffis,0,xinfo,yinfo,xnorm,ynorm,x,y,i,j,xdim,mlid);

//FIXME: need to allow for zero norms
        {
            gentype locdiffis(0.0);

            K2unnorm(temp,q,xnorm,xnorm,locdiffis,1,xinfo,xinfo,xnorm,xnorm,x,x,i,i,xdim,0,0,0,mlid);

            scalefact *= sqrt((double) abs2(temp));

            {
                dKunnormda(dKda_aaa,minmaxind,q,xnorm,xnorm,locdiffis,1,xinfo,xinfo,xnorm,xnorm,x,x,i,i,xdim,mlid);
                dKunnormdz(dKdz_aaa,minmaxind,q,xnorm,xnorm,locdiffis,1,xinfo,xinfo,xnorm,xnorm,x,x,i,i,xdim,mlid);

                Kaaa = temp;
                Kaaa.inverse();
            }

            K2unnorm(temp,q,ynorm,ynorm,locdiffis,1,yinfo,yinfo,ynorm,ynorm,y,y,j,j,xdim,0,0,0,mlid);

            scalefact *= sqrt((double) abs2(temp));
	}

        res =  dKda_aaa;
        res *= 2.0;
        res += dKdz_aaa;
        leftmult(res,Kaaa); // Kaaa has been inverted
        leftmult(res,Kazb);
        res *= -0.5;
        res += dKda_azb;
        res /= scalefact;
    }

    return;
}

void MercerKernel::dKdz(gentype &res, int &minmaxind, int q, 
                         const gentype &xyprod, const gentype &yxprod, gentype &diffis, int recalcdiffis, 
                         const vecInfo &xinfo, const vecInfo &yinfo, 
                         const gentype &xnorm, const gentype &ynorm, 
                         const SparseVector<gentype> &x, const SparseVector<gentype> &y, 
                         int i, int j, 
                         int xdim, int mlid) const
{
    NiceAssert( q >= 0 );
    NiceAssert( q < size() );

    dKunnormdz(res,minmaxind,q,xyprod,yxprod,diffis,recalcdiffis,xinfo,yinfo,xnorm,ynorm,x,y,i,j,xdim,mlid);

    if ( isNormalised(q) )
    {
        gentype xxprod;
        gentype xkern;
        gentype locdiffis(0.0);

//FIXME: need to allow for zero norms
        {
            K2unnorm(xkern,q,xnorm,xnorm,locdiffis,1,xinfo,xinfo,xnorm,xnorm,x,x,i,i,xdim,0,0,0,mlid);

            res /= sqrt((double) abs2(xkern));

            K2unnorm(xkern,q,ynorm,ynorm,locdiffis,1,yinfo,yinfo,ynorm,ynorm,y,y,j,j,xdim,0,0,0,mlid);

            res /= sqrt((double) abs2(xkern));
	}
    }

    return;
}





































// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================

void MercerKernel::dKunnormda(gentype &res, int &minmaxind, int q, 
                              const gentype &xyprod, const gentype &yxprod, gentype &diffis, int recalcdiffis,
                              const vecInfo &xinfo, const vecInfo &yinfo, 
                              const gentype &xnorm, const gentype &ynorm, 
                              const SparseVector<gentype> &x, const SparseVector<gentype> &y, 
                              int i, int j, 
                              int xdim, int mlid) const
{
    if ( recalcdiffis && needsDiff(q) )
    {
	// Calculate ||x-y||^2 only as required

        diff2norm(diffis,(xyprod+yxprod)/2.0,xnorm,ynorm);
    }

    Vector<const SparseVector<gentype> *> xx(2);
    Vector<const vecInfo *> xxinfo(2);
    Vector<const gentype *> xxnorm(2);
    Vector<int> ii(2);

    xx("&",0) = &x;
    xx("&",1) = &y;

    xxinfo("&",0) = &xinfo;
    xxinfo("&",1) = &yinfo;

    xxnorm("&",0) = &xnorm;
    xxnorm("&",1) = &ynorm;

    ii("&",0) = i;
    ii("&",1) = j;

    return dKdaBase(res,minmaxind,q,
                    xyprod,yxprod,diffis,
                    xx,
                    xxinfo,
                    xxnorm,
                    ii,
                    xdim,2,mlid);
}

void MercerKernel::dKunnormdz(gentype &res, int &minmaxind, int q, 
                              const gentype &xyprod, const gentype &yxprod, gentype &diffis, int recalcdiffis, 
                              const vecInfo &xinfo, const vecInfo &yinfo, 
                              const gentype &xnorm, const gentype &ynorm, 
                              const SparseVector<gentype> &x, const SparseVector<gentype> &y, 
                              int i, int j, 
                              int xdim, int mlid) const
{
    if ( recalcdiffis && needsDiff(q) )
    {
	// Calculate ||x-y||^2 only as required

        diff2norm(diffis,(xyprod+yxprod)/2.0,xnorm,ynorm);
    }

    Vector<const SparseVector<gentype> *> xx(2);
    Vector<const vecInfo *> xxinfo(2);
    Vector<const gentype *> xxnorm(2);
    Vector<int> ii(2);

    xx("&",0) = &x;
    xx("&",1) = &y;

    xxinfo("&",0) = &xinfo;
    xxinfo("&",1) = &yinfo;

    xxnorm("&",0) = &xnorm;
    xxnorm("&",1) = &ynorm;

    ii("&",0) = i;
    ii("&",1) = j;

    return dKdzBase(res,minmaxind,q,
                    xyprod,yxprod,diffis,
                    xx,
                    xxinfo,
                    xxnorm,
                    ii,
                    xdim,2,mlid);
}





































// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================




































// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================


















// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================













































// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
























// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================



























// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
































































//===========================================================================
//===========================================================================
//===========================================================================
//===========================================================================
//===========================================================================
//===========================================================================
//===========================================================================
//===========================================================================
//===========================================================================
//===========================================================================
//===========================================================================
//===========================================================================
//===========================================================================
//===========================================================================
//===========================================================================
//===========================================================================
//===========================================================================
//===========================================================================
//===========================================================================
//===========================================================================
//===========================================================================
//===========================================================================
//===========================================================================
//===========================================================================
//===========================================================================
//===========================================================================
//===========================================================================
//===========================================================================
//===========================================================================
//===========================================================================
//===========================================================================
//===========================================================================
//===========================================================================
//===========================================================================
//===========================================================================
//===========================================================================
//===========================================================================
//===========================================================================
//===========================================================================
//===========================================================================
//===========================================================================
//===========================================================================
//===========================================================================
//===========================================================================
//===========================================================================
//===========================================================================
//===========================================================================
//===========================================================================
//===========================================================================
//===========================================================================
//===========================================================================
//===========================================================================
//===========================================================================


// Function to shift, scale and index vectors prior to calling the callback
// function
//
// Note: this function is not fast.  Better hope the kernel cache is large
// enough.

SparseVector<gentype> &MercerKernel::preShiftScale(SparseVector<gentype> &res, const SparseVector<gentype> &x) const
{
    int i,j;

    if ( isShiftedScaled() && isIndex() )
    {
        res.zero();

        if ( dIndexes.size() )
        {
            for ( i = 0 ; i < dIndexes.size() ; ++i )
            {
                j = dIndexes(i);

                res("&",j) = x(j);
                res("&",j) += dShift(j);
                res("&",j) /= dScale(j);
            }
        }
    }

    else if ( isShifted() && isIndex() )
    {
        res.zero();

        if ( dIndexes.size() )
        {
            for ( i = 0 ; i < dIndexes.size() ; ++i )
            {
                j = dIndexes(i);

                res("&",j) = x(j);
                res("&",j) += dShift(j);
            }
        }
    }

    else if ( isScaled() && isIndex() )
    {
        res.zero();

        if ( dIndexes.size() )
        {
            for ( i = 0 ; i < dIndexes.size() ; ++i )
            {
                j = dIndexes(i);

                res("&",j)  = x(j);
                res("&",j) /= dScale(j);
            }
        }
    }

    else if ( isShiftedScaled() )
    {
        res  = x;
        res += dShift;
        res /= dScale;
    }

    else if ( isShifted() )
    {
        res  = x;
        res += dShift;
    }

    else if ( isScaled() )
    {
        res  = x;
        res /= dScale;
    }

    else if ( isIndex() )
    {
        res.zero();

        if ( dIndexes.size() )
        {
            for ( i = 0 ; i < dIndexes.size() ; ++i )
            {
                j = dIndexes(i);

                res("&",j) = x(j);
            }
        }
    }

    else
    {
        res = x;
    }

    return res;
}


void MercerKernel::fixShiftProd(void)
{
    if ( !isIndex() )
    {
        getTwoProd(dShiftProd,dShift,dShift,0,1,3,0,0);
        getTwoProd(dShiftProdNoConj,dShift,dShift,0,0,3,0,0);
        getTwoProd(dShiftProdRevConj,dShift,dShift,0,2,3,0,0);
    }

    else
    {
        getTwoProd(dShiftProd,dShift,dShift,1,1,3,0,0);
        getTwoProd(dShiftProdNoConj,dShift,dShift,1,0,3,0,0);
        getTwoProd(dShiftProdRevConj,dShift,dShift,1,2,3,0,0);
    }

    return;
}

vecInfo &MercerKernel::getvecInfo(vecInfo &res, const SparseVector<gentype> &x, const gentype *xmag, int xconsist, int assumreal) const
{
    int i;
    int z = 0;

//    int nearindsize = x.nindsize();
    int farindsize = x.f1indsize();

    NiceAssert( res.isloc );

    (*(res.content("&",z))).zero();
    (*(res.content("&",1))).zero();

    int nearusize = x.nupsize();
    int farusize  = x.f1upsize();

    res.usize_overwrite = 0;

    {
        for ( i = 0 ; i < nearusize ; ++i )
        {
            const SparseVector<gentype> &xx = x.nup(i);

            getvecInfo((*((res.content)("&",z)))("&",i),xx,!i ? xmag : nullptr,xconsist,assumreal);

            (*((res.content)("&",z)))("&",i).xusize = nearusize;
        }
    }

    if ( farindsize )
    {
        for ( i = 0 ; i < farusize ; ++i )
        {
            const SparseVector<gentype> &xx = x.f1up(i);

            //getvecInfo((*((res.content)("&",1)))("&",i),xx.f1(),nullptr,xconsist,assumreal);
            getvecInfo((*((res.content)("&",1)))("&",i),xx,nullptr,xconsist,assumreal);

            (*((res.content)("&",1)))("&",i).xusize = farusize;
        }
    }

    return res;
}

vecInfoBase &MercerKernel::getvecInfo(vecInfoBase &res, const SparseVector<gentype> &x, const gentype *xmag, int xconsist, int assumreal) const
{
    // NB: we need to correct all averages here.  By default, the mean
    // of a sparse vector uses nindsize() for scaling.  However for eg
    // if the vector
    // ( 0 0 1 0 )
    // is streamed into the SVM then this will be automatically converted
    // to
    // ( 2:1 3:0 )
    // and hence the mean will be 0.5, rather than 0.25 as required for
    // correct calculation.  We correct this by noting that size() of
    // the above sparse vector will be
    // 3+1 = 4
    // which is the correct value, so we need to rescale all averages
    // by nindsize()/size() (or 1 if size() == 0).

    int i = 0;

    res.hasbeenset = 1;

    res.xiseqn = 0;

    if ( x.nindsize() )
    {
        for ( i = 0 ; i < x.nindsize() ; ++i )
        {
            res.xiseqn |= x.direcref(i).isValEqn();
        }
    }

    // Can only do this now as may require means and variances

    Vector<gentype> &xhalfmprod = res.xhalfmprod;

    int m = xproddepth;
    int oldm = 0; // Need to calculate from scratch

    xhalfmprod.resize(m/2);

    if ( ( m >= 2 ) && ( m > oldm ) )
    {
        if ( !xmag )
        {
            twoProductDiverted(xhalfmprod("&",0),x,x,xconsist,assumreal);
        }

        else
        {
            xhalfmprod("&",0) = *xmag;
        }

        oldm = 2;
    }

    if ( ( m >= 4 ) && ( m > oldm ) )
    {
        fourProductDiverted(xhalfmprod("&",1),x,x,x,x,xconsist,assumreal);

        oldm = 4;
    }

    if ( ( m >= 6 ) && ( m/2 > xhalfmprod.size() ) )
    {
        Vector<const SparseVector<gentype> *> aa(m);

        aa = &x;

        int i;

        for ( i = oldm/2 ; i < m/2 ; ++i )
        {
            retVector<const SparseVector<gentype> *> tmpva;

            mProductDiverted(2*(i+1),xhalfmprod("&",i),aa(0,1,2*(i+1)-1,tmpva),xconsist,assumreal);
        }
    }

    return res;
}

void MercerKernel::processOverwrites(int q, const SparseVector<gentype> &x, const SparseVector<gentype> &y) const
{
    if ( isLeftRightPlain() )
    {
        return;
    }

    int i;

    if ( dIntOverwrite(q).nindsize() )
    {
        for ( i = 0 ; i < dIntOverwrite(q).nindsize() ; ++i )
        {
            gentype altdest;
            int &dest = dIntConstants("&",q)("&",dIntOverwrite(q).ind(i));
            int srcind = dIntOverwrite(q).direcref(i);

            setzero(altdest);

            if ( isLeftPlain() )
            {
                altdest = y(srcind);
            }

            else if ( isRightPlain() )
            {
                altdest = x(srcind);
            }

            else
            {
                altdest  = x(srcind);
                altdest *= y(srcind);
            }

            dest = (int) altdest;
        }
    }

    if ( dRealOverwrite(q).nindsize() )
    {
        for ( i = 0 ; i < dRealOverwrite(q).nindsize() ; ++i )
        {
            gentype &dest = (dRealConstants)("&",q)("&",dRealOverwrite(q).ind(i));
            int srcind = dRealOverwrite(q).direcref(i);

            setzero(dest);

            if ( isLeftPlain() )
            {
                dest = y(srcind);
            }

            else if ( isRightPlain() )
            {
                dest = x(srcind);
                setconj(dest);
            }

            else
            {
                dest = x(srcind);
                setconj(dest);
                dest *= y(srcind);
            }
        }
    }

    return;
}

void MercerKernel::fixcombinedOverwriteSrc(void)
{
    // Clear overwrite indices list

    combinedOverwriteSrc.resize(0);

    // Grab all indices

    if ( size() )
    {
        int q,i,j;

        for ( q = 0 ; q < size() ; ++q )
        {
            if ( dRealOverwrite(q).size() )
            {
                for ( i = 0 ; i < dRealOverwrite(q).nindsize() ; ++i )
                {
                    j = combinedOverwriteSrc.size();
                    combinedOverwriteSrc.add(j);
                    combinedOverwriteSrc("&",j) = dRealOverwrite(q).direcref(i);
                }
            }

            if ( dIntOverwrite(q).size() )
            {
                for ( i = 0 ; i < dIntOverwrite(q).nindsize() ; ++i )
                {
                    j = combinedOverwriteSrc.size();
                    combinedOverwriteSrc.add(j);
                    combinedOverwriteSrc("&",j) = dIntOverwrite(q).direcref(i);
                }
            }
        }
    }

    // Sort indices

    if ( combinedOverwriteSrc.size() > 1 )
    {
        int i,j;

        for ( i = 0 ; i < combinedOverwriteSrc.size()-1 ; ++i )
        {
            for ( j = i+1 ; j < combinedOverwriteSrc.size() ; ++j )
            {
                if ( combinedOverwriteSrc(j) < combinedOverwriteSrc(i) )
                {
                    qswap(combinedOverwriteSrc("&",i),combinedOverwriteSrc("&",j));
                    --i;
                    break;
                }
            }
        }
    }

    // Remove duplicates

    if ( combinedOverwriteSrc.size() > 1 )
    {
        int i;

        for ( i = 0 ; i < combinedOverwriteSrc.size()-1 ; ++i )
        {
            if ( combinedOverwriteSrc(i) == combinedOverwriteSrc(i+1) )
            {
                combinedOverwriteSrc.remove(i);
                --i;
            }
        }
    }

    return;
}

void MercerKernel::addinOverwriteInd(const SparseVector<gentype> &x, const SparseVector<gentype> &y) const
{
    if ( combinedOverwriteSrc.size() )
    {
        backupisind    = isind;
        backupdIndexes = dIndexes;
        isind          = 1;

        Vector<int> &redidIndexes = dIndexes;

        if ( !backupisind )
        {
            // Add indices for vectors

            int i;
            int j;

            i = 0;
            j = 0;

            while ( j < x.nindsize() )
            {
                if ( i < redidIndexes.size() )
                {
                    if ( redidIndexes(i) < x.ind(j) )
                    {
                        ++i;
                    }

                    else if ( redidIndexes(i) > x.ind(j) )
                    {
                        redidIndexes.add(i);
                        redidIndexes("&",i) = x.ind(j);
                        ++i;
                        ++j;
                    }

                    else
                    {
                        ++i;
                        ++j;
                    }
                }

                else
                {
                    NiceAssert( i == redidIndexes.size() );

                    redidIndexes.add(i);
                    redidIndexes("&",i) = x.ind(j);
                    ++i;
                    ++j;
                }
            }

            i = 0;
            j = 0;

            while ( j < y.nindsize() )
            {
                if ( i < redidIndexes.size() )
                {
                    if ( redidIndexes(i) < y.ind(j) )
                    {
                        ++i;
                    }

                    else if ( redidIndexes(i) > y.ind(j) )
                    {
                        redidIndexes.add(i);
                        redidIndexes("&",i) = y.ind(j);
                        ++i;
                        ++j;
                    }

                    else
                    {
                        ++i;
                        ++j;
                    }
                }

                else
                {
                    NiceAssert( i == redidIndexes.size() );

                    redidIndexes.add(i);
                    redidIndexes("&",i) = y.ind(j);
                    ++i;
                    ++j;
                }
            }
        }

        addinOverwriteInd();
    }

    return;
}

/*
void MercerKernel::addinOverwriteInd(const Vector<gentype> &x, const Vector<gentype> &y) const
{
    if ( combinedOverwriteSrc.size() )
    {
        backupisind    = isind;
        backupdIndexes = dIndexes;
        isind          = 1;

        Vector<int> &redidIndexes = dIndexes;

        if ( !backupisind )
        {
            // Add indices for vectors

            int i;
            int j = ( x.size() < y.size() ) ? x.size() : y.size();

            for ( i = 0 ; i < j ; ++i )
            {
                redidIndexes.add(i);
                redidIndexes("&",i) = i;
            }
        }

        addinOverwriteInd();
    }

    return;
}
*/

/*
void MercerKernel::addinOverwriteInd(const SparseVector<double> &x, const SparseVector<double> &y) const
{
    if ( combinedOverwriteSrc.size() )
    {
        backupisind    = isind;
        backupdIndexes = dIndexes;
        isind          = 1;

        Vector<int> &redidIndexes = dIndexes;

        if ( !backupisind )
        {
            // Add indices for vectors

            int i;
            int j;

            i = 0;
            j = 0;

            while ( j < x.nindsize() )
            {
                if ( i < redidIndexes.size() )
                {
                    if ( redidIndexes(i) < x.ind(j) )
                    {
                        ++i;
                    }

                    else if ( redidIndexes(i) > x.ind(j) )
                    {
                        redidIndexes.add(i);
                        redidIndexes("&",i) = x.ind(j);
                        ++i;
                        ++j;
                    }

                    else
                    {
                        ++i;
                        ++j;
                    }
                }

                else
                {
                    NiceAssert( i == redidIndexes.size() );

                    redidIndexes.add(i);
                    redidIndexes("&",i) = x.ind(j);
                    ++i;
                    ++j;
                }
            }

            i = 0;
            j = 0;

            while ( j < y.nindsize() )
            {
                if ( i < redidIndexes.size() )
                {
                    if ( redidIndexes(i) < y.ind(j) )
                    {
                        ++i;
                    }

                    else if ( redidIndexes(i) > y.ind(j) )
                    {
                        redidIndexes.add(i);
                        redidIndexes("&",i) = y.ind(j);
                        ++i;
                        ++j;
                    }

                    else
                    {
                        ++i;
                        ++j;
                    }
                }

                else
                {
                    NiceAssert( i == redidIndexes.size() );

                    redidIndexes.add(i);
                    redidIndexes("&",i) = y.ind(j);
                    ++i;
                    ++j;
                }
            }
        }

        addinOverwriteInd();
    }

    return;
}
*/

/*
void MercerKernel::addinOverwriteInd(const Vector<double> &x, const Vector<double> &y) const
{
    if ( combinedOverwriteSrc.size() )
    {
        backupisind    = isind;
        backupdIndexes = dIndexes;
        isind          = 1;

        Vector<int> &redidIndexes = dIndexes;

        if ( !backupisind )
        {
            // Add indices for vectors

            int i;
            int j = ( x.size() < y.size() ) ? x.size() : y.size();

            for ( i = 0 ; i < j ; ++i )
            {
                redidIndexes.add(i);
                redidIndexes("&",i) = i;
            }
        }

        addinOverwriteInd();
    }

    return;
}
*/

void MercerKernel::addinOverwriteInd(const SparseVector<gentype> &v) const
{
    if ( combinedOverwriteSrc.size() )
    {
        backupisind    = isind;
        backupdIndexes = dIndexes;
        isind          = 1;

        Vector<int> &redidIndexes = dIndexes;

        if ( !backupisind )
        {
            // Add indices for vectors

            int i;
            int j;

            i = 0;
            j = 0;

            while ( j < v.nindsize() )
            {
                if ( i < redidIndexes.size() )
                {
                    if ( redidIndexes(i) < v.ind(j) )
                    {
                        ++i;
                    }

                    else if ( redidIndexes(i) > v.ind(j) )
                    {
                        redidIndexes.add(i);
                        redidIndexes("&",i) = v.ind(j);
                        ++i;
                        ++j;
                    }

                    else
                    {
                        ++i;
                        ++j;
                    }
                }

                else
                {
                    NiceAssert( i == redidIndexes.size() );

                    redidIndexes.add(i);
                    redidIndexes("&",i) = v.ind(j);
                    ++i;
                    ++j;
                }
            }
        }

        addinOverwriteInd();
    }

    return;
}

void MercerKernel::addinOverwriteInd(const SparseVector<gentype> &v, const SparseVector<gentype> &w, const SparseVector<gentype> &x) const
{
    if ( combinedOverwriteSrc.size() )
    {
        backupisind    = isind;
        backupdIndexes = dIndexes;
        isind          = 1;

        Vector<int> &redidIndexes = dIndexes;

        if ( !backupisind )
        {
            // Add indices for vectors

            int i;
            int j;

            i = 0;
            j = 0;

            while ( j < v.nindsize() )
            {
                if ( i < redidIndexes.size() )
                {
                    if ( redidIndexes(i) < v.ind(j) )
                    {
                        ++i;
                    }

                    else if ( redidIndexes(i) > v.ind(j) )
                    {
                        redidIndexes.add(i);
                        redidIndexes("&",i) = v.ind(j);
                        ++i;
                        ++j;
                    }

                    else
                    {
                        ++i;
                        ++j;
                    }
                }

                else
                {
                    NiceAssert( i == redidIndexes.size() );

                    redidIndexes.add(i);
                    redidIndexes("&",i) = v.ind(j);
                    ++i;
                    ++j;
                }
            }

            i = 0;
            j = 0;

            while ( j < w.nindsize() )
            {
                if ( i < redidIndexes.size() )
                {
                    if ( redidIndexes(i) < w.ind(j) )
                    {
                        ++i;
                    }

                    else if ( redidIndexes(i) > w.ind(j) )
                    {
                        redidIndexes.add(i);
                        redidIndexes("&",i) = w.ind(j);
                        ++i;
                        ++j;
                    }

                    else
                    {
                        ++i;
                        ++j;
                    }
                }

                else
                {
                    NiceAssert( i == redidIndexes.size() );

                    redidIndexes.add(i);
                    redidIndexes("&",i) = w.ind(j);
                    ++i;
                    ++j;
                }
            }

            i = 0;
            j = 0;

            while ( j < x.nindsize() )
            {
                if ( i < redidIndexes.size() )
                {
                    if ( redidIndexes(i) < x.ind(j) )
                    {
                        ++i;
                    }

                    else if ( redidIndexes(i) > x.ind(j) )
                    {
                        redidIndexes.add(i);
                        redidIndexes("&",i) = x.ind(j);
                        ++i;
                        ++j;
                    }

                    else
                    {
                        ++i;
                        ++j;
                    }
                }

                else
                {
                    NiceAssert( i == redidIndexes.size() );

                    redidIndexes.add(i);
                    redidIndexes("&",i) = x.ind(j);
                    ++i;
                    ++j;
                }
            }
        }

        addinOverwriteInd();
    }

    return;
}

void MercerKernel::addinOverwriteInd(const SparseVector<gentype> &v, const SparseVector<gentype> &w, const SparseVector<gentype> &x, const SparseVector<gentype> &y) const
{
    if ( combinedOverwriteSrc.size() )
    {
        backupisind    = isind;
        backupdIndexes = dIndexes;
        isind          = 1;

        Vector<int> &redidIndexes = dIndexes;

        if ( !backupisind )
        {
            // Add indices for vectors

            int i;
            int j;

            i = 0;
            j = 0;

            while ( j < v.nindsize() )
            {
                if ( i < redidIndexes.size() )
                {
                    if ( redidIndexes(i) < v.ind(j) )
                    {
                        ++i;
                    }

                    else if ( redidIndexes(i) > v.ind(j) )
                    {
                        redidIndexes.add(i);
                        redidIndexes("&",i) = v.ind(j);
                        ++i;
                        ++j;
                    }

                    else
                    {
                        ++i;
                        ++j;
                    }
                }

                else
                {
                    NiceAssert( i == redidIndexes.size() );

                    redidIndexes.add(i);
                    redidIndexes("&",i) = v.ind(j);
                    ++i;
                    ++j;
                }
            }

            i = 0;
            j = 0;

            while ( j < w.nindsize() )
            {
                if ( i < redidIndexes.size() )
                {
                    if ( redidIndexes(i) < w.ind(j) )
                    {
                        ++i;
                    }

                    else if ( redidIndexes(i) > w.ind(j) )
                    {
                        redidIndexes.add(i);
                        redidIndexes("&",i) = w.ind(j);
                        ++i;
                        ++j;
                    }

                    else
                    {
                        ++i;
                        ++j;
                    }
                }

                else
                {
                    NiceAssert( i == redidIndexes.size() );

                    redidIndexes.add(i);
                    redidIndexes("&",i) = w.ind(j);
                    ++i;
                    ++j;
                }
            }

            i = 0;
            j = 0;

            while ( j < x.nindsize() )
            {
                if ( i < redidIndexes.size() )
                {
                    if ( redidIndexes(i) < x.ind(j) )
                    {
                        ++i;
                    }

                    else if ( redidIndexes(i) > x.ind(j) )
                    {
                        redidIndexes.add(i);
                        redidIndexes("&",i) = x.ind(j);
                        ++i;
                        ++j;
                    }

                    else
                    {
                        ++i;
                        ++j;
                    }
                }

                else
                {
                    NiceAssert( i == redidIndexes.size() );

                    redidIndexes.add(i);
                    redidIndexes("&",i) = x.ind(j);
                    ++i;
                    ++j;
                }
            }

            i = 0;
            j = 0;

            while ( j < y.nindsize() )
            {
                if ( i < redidIndexes.size() )
                {
                    if ( redidIndexes(i) < y.ind(j) )
                    {
                        ++i;
                    }

                    else if ( redidIndexes(i) > y.ind(j) )
                    {
                        redidIndexes.add(i);
                        redidIndexes("&",i) = y.ind(j);
                        ++i;
                        ++j;
                    }

                    else
                    {
                        ++i;
                        ++j;
                    }
                }

                else
                {
                    NiceAssert( i == redidIndexes.size() );

                    redidIndexes.add(i);
                    redidIndexes("&",i) = y.ind(j);
                    ++i;
                    ++j;
                }
            }
        }

        addinOverwriteInd();
    }

    return;
}

void MercerKernel::addinOverwriteInd(const Vector<const SparseVector<gentype> *> &a) const
{
    if ( combinedOverwriteSrc.size() )
    {
        backupisind    = isind;
        backupdIndexes = dIndexes;
        isind          = 1;

        Vector<int> &redidIndexes = dIndexes;

        if ( !backupisind )
        {
            // Add indices for vectors

            int i;
            int j;
            int k;

            if ( a.size() )
            {
                for ( k = 0 ; k < a.size() ; ++k )
                {
                    const SparseVector<gentype> &x = *a(k);

                    i = 0;
                    j = 0;

                    while ( j < x.nindsize() )
                    {
                        if ( i < redidIndexes.size() )
                        {
                            if ( redidIndexes(i) < x.ind(j) )
                            {
                                ++i;
                            }

                            else if ( redidIndexes(i) > x.ind(j) )
                            {
                                redidIndexes.add(i);
                                redidIndexes("&",i) = x.ind(j);
                                ++i;
                                ++j;
                            }

                            else
                            {
                                ++i;
                                ++j;
                            }
                        }

                        else
                        {
                            NiceAssert( i == redidIndexes.size() );

                            redidIndexes.add(i);
                            redidIndexes("&",i) = x.ind(j);
                            ++i;
                            ++j;
                        }
                    }
                }
            }
        }

        addinOverwriteInd();
    }

    return;
}

void MercerKernel::addinOverwriteInd(void) const
{
    if ( combinedOverwriteSrc.size() )
    {
        Vector<int> &redidIndexes = dIndexes;

        int i = 0;
        int j = 0;

        // Remove indices that are used for overwrite

        while ( ( i < redidIndexes.size() ) && ( j < combinedOverwriteSrc.size() ) )
        {
            if ( redidIndexes(i) < combinedOverwriteSrc(j) )
            {
                ++i;
            }

            else if ( redidIndexes(i) > combinedOverwriteSrc(j) )
            {
                ++j;
            }

            else
            {
                redidIndexes.remove(i);
            }
        }
    }

    return;
}

void MercerKernel::removeOverwriteInd(void) const
{
    if ( combinedOverwriteSrc.size() )
    {
        // Need to correct for indices

        isind    = backupisind;
        dIndexes = backupdIndexes;
    }

    return;
}



























// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================

int MercerKernel::innerProductDiverted(double &res, const SparseVector<gentype> &a, const SparseVector<gentype> &b, int xconsist, int assumreal) const
{
    int tres = 0;

    if ( !assumreal )
    {
        gentype temp(res);

        tres = innerProductDiverted(temp,a,b,xconsist,assumreal);

        res = tres ? 0.0 : (double) temp;
    }

    else
    {
        tres = twoProductDiverted(res,a,b,xconsist,assumreal);
    }

    return tres;
}

int MercerKernel::innerProductDiverted(gentype &result, const SparseVector<gentype> &x, const SparseVector<gentype> &y, int xconsist, int assumreal) const
{
//errstream() << "phantomxyzmercer 0\n";
    if ( assumreal )
    {
//errstream() << "phantomxyzmercer 1\n";
        return twoProductDiverted(result.force_double(),x,y,xconsist,assumreal);
//errstream() << "phantomxyzmercer 2\n";
    }

    addinOverwriteInd(x,y);

    // If x pre-processing is enabled:
    //
    // ( ( x - 1xstep )/xscale )' . ( y - 1ystep )/yscale
    // = ( x'.y - sum(x).ystep - conj(xstep).sum(y) + conj(xstep).ystep.N )/(xscale*yscale)
    //
    // where xstep, xscale, ystep and yscale are all scalars and already
    // include correction for shifting/scaling and indexing.
    //
    // leftPlain:  = ( x'.y - sum(x).ystep       )/yscale
    // rightPlain: = ( x'.y - conj(xstep).sum(y) )/xscale

    // Step 1 - common for all cases

    {
        if ( ( !isShifted() && !isScaled() ) || isLeftRightPlain() )
        {
            if ( !isIndex() )
            {
                getTwoProd(result,x,y,0,1,0,xconsist,assumreal);
            }

            else
            {
                getTwoProd(result,x,y,1,1,0,xconsist,assumreal);
            }
        }

        else if ( !isShifted() || isLeftRightPlain() )
        {
            if ( !isIndex() )
            {
                getTwoProd(result,x,y,0,1,3,xconsist,assumreal);
            }

            else
            {
                getTwoProd(result,x,y,1,1,3,xconsist,assumreal);
            }
        }

        else
        {
            // If isLeftPlain:
            //
            // x'(sc.*(y+sh)) = x'.diag(sc).(y+sh)
            //                = x'.diag(sc).y + x'.diag(sc).sh
            //                = x'.S.y + x'.S.sh
            //
            // If isRightPlain:
            //
            // (sc.*(x+sh))'y = (x+sh)'.diag(sc).y
            //                = x'.diag(sc).y + sh'.diag(sc).y
            //
            // else:
            //
            // (sc.*(x+sh))'(sc.*(y+sh)) = (x+sh)'.diag(sc.*sc).(y+sh)
            //                           = x'.diag(sc.*sc).y + x'.diag(sc.*sc).(sh./sc) + (sh./sc)'.diag(sc.*sc).y + sh'.diag(sc.*sc).sh
            //                           = x'.S.y + x'.S.dShift + dShift'.S.y + dShiftProd

            // NB: sc(i).*sc(i) = outerProd(sc(i),sc(i)) for vector-valued case

            if ( isLeftPlain() )
            {
                NiceAssert( isRightNormal() );

                gentype tempb;

                if ( !isIndex() )
                {
                    getTwoProd(result,x,y,0,1,2,xconsist,assumreal);
                    getTwoProd(tempb,dShift,y,0,1,2,xconsist,assumreal);
                }

                else
                {
                    getTwoProd(result,x,y,1,1,2,xconsist,assumreal);
                    getTwoProd(tempb,dShift,y,1,1,2,xconsist,assumreal);
                }

                result += tempb;
            }

            else if ( isRightPlain() )
            {
                NiceAssert( isLeftNormal() );

                gentype tempa;

                if ( !isIndex() )
                {
                    getTwoProd(result,x,y,0,1,1,xconsist,assumreal);
                    getTwoProd(tempa,x,dShift,0,1,1,xconsist,assumreal);
                }

                else
                {
                    getTwoProd(result,x,y,1,1,1,xconsist,assumreal);
                    getTwoProd(tempa,x,dShift,1,1,1,xconsist,assumreal);
                }

                result += tempa;
            }

            else
            {
                NiceAssert( isLeftRightNormal() );

                gentype tempa;
                gentype tempb;

                if ( !isIndex() )
                {
                    getTwoProd(result,x,y,0,1,3,xconsist,assumreal);
                    getTwoProd(tempa,x,dShift,0,1,3,xconsist,assumreal);
                    getTwoProd(tempb,dShift,y,0,1,3,xconsist,assumreal);
                }

                else
                {
                    getTwoProd(result,x,y,1,1,3,xconsist,assumreal);
                    getTwoProd(tempa,x,dShift,1,1,3,xconsist,assumreal);
                    getTwoProd(tempb,dShift,y,1,1,3,xconsist,assumreal);
                }

                result += tempa;
                result += tempb;
                result += dShiftProd;
            }
        }
    }

//errstream() << "phantomxyzmercer 100\n";
    removeOverwriteInd();

    return result.isValEqn();
}

int MercerKernel::innerProductDivertedRevConj(gentype &result, const gentype &xyres, const SparseVector<gentype> &x, const SparseVector<gentype> &y, int xconsist, int assumreal) const
{
    if ( xyres.isCommutative() )
    {
        result = xyres;
        result.conj();
        return result.isValEqn();
    }

    if ( assumreal )
    {
        return twoProductDiverted(result.force_double(),x,y,xconsist,assumreal);
    }

    addinOverwriteInd(x,y);

    if ( ( !isShifted() && !isScaled() ) || isLeftRightPlain() )
    {
        if ( !isIndex() )
	{
            getTwoProd(result,x,y,0,2,0,xconsist,assumreal);
	}

	else
	{
            getTwoProd(result,x,y,1,2,0,xconsist,assumreal);
	}
    }

    else if ( !isShifted() || isLeftRightPlain() )
    {
        if ( !isIndex() )
	{
            getTwoProd(result,x,y,0,2,3,xconsist,assumreal);
	}

	else
	{
            getTwoProd(result,x,y,1,2,3,xconsist,assumreal);
	}
    }

    else
    {
        // If isLeftPlain:
        //
        // x'(sc.*(y+sh)) = x'.diag(sc).(y+sh)
        //                = x'.diag(sc).y + x'.diag(sc).sh
        //                = x'.S.y + x'.S.sh
        //
        // If isRightPlain:
        //
        // (sc.*(x+sh))'y = (x+sh)'.diag(sc).y
        //                = x'.diag(sc).y + sh'.diag(sc).y
        //
        // else:
        //
	// ((x+sh).*sc)'((y+sh).*sc) = (x+sh)'.diag(sc.*sc).(y+sh)
	//                           = x'.diag(sc.*sc).y + x'.diag(sc.*sc).(sh./sc) + (sh./sc)'.diag(sc.*sc).y + sh'.diag(sc.*sc).sh
        //                           = x'.S.y + x'.S.dShift + dShift'.S.y + dShiftProd

        if ( isLeftPlain() )
        {
            NiceAssert( isRightNormal() );

            gentype tempb;

            if ( !isIndex() )
            {
                getTwoProd(result,x,y,0,2,2,xconsist,assumreal);
                getTwoProd(tempb,dShift,y,0,2,2,xconsist,assumreal);
            }

            else
            {
                getTwoProd(result,x,y,1,2,2,xconsist,assumreal);
                getTwoProd(tempb,dShift,y,1,2,2,xconsist,assumreal);
            }

            result += tempb;
        }

        else if ( isRightPlain() )
        {
            NiceAssert( isLeftNormal() );

            gentype tempa;

            if ( !isIndex() )
            {
                getTwoProd(result,x,y,0,2,1,xconsist,assumreal);
                getTwoProd(tempa,x,dShift,0,2,1,xconsist,assumreal);
            }

            else
            {
                getTwoProd(result,x,y,1,2,1,xconsist,assumreal);
                getTwoProd(tempa,x,dShift,1,2,1,xconsist,assumreal);
            }

            result += tempa;
        }

        else
        {
            NiceAssert( isLeftRightNormal() );

            gentype tempa;
            gentype tempb;

            if ( !isIndex() )
            {
                getTwoProd(result,x,y,0,2,3,xconsist,assumreal);
                getTwoProd(tempa,x,dShift,0,2,3,xconsist,assumreal);
                getTwoProd(tempb,dShift,y,0,2,3,xconsist,assumreal);
            }

            else
            {
                getTwoProd(result,x,y,1,2,3,xconsist,assumreal);
                getTwoProd(tempa,x,dShift,1,2,3,xconsist,assumreal);
                getTwoProd(tempb,dShift,y,1,2,3,xconsist,assumreal);
            }

            result += tempa;
            result += tempb;
            result += dShiftProdRevConj;
        }
    }

    removeOverwriteInd();

    return result.isValEqn();
}

int MercerKernel::oneProductDiverted(double &result, const SparseVector<gentype> &v, int xconsist, int assumreal) const
{
  int tres = 0;

  if ( !assumreal )
  {
    gentype temp(result);

    tres = oneProductDiverted(temp,v,xconsist,assumreal);

    result = tres ? 0.0 : (double) temp;
  }

  else
  {
    // FIXME: modify so that copy constructor not needed.

    addinOverwriteInd(v);

    NiceAssert( isLeftRightNormal() );

    if ( !isShifted() && !isScaled() )
    {
        if ( !isIndex() )
	{
            result = getOneProd(v,0,0,xconsist,assumreal);
	}

	else
	{
            result = getOneProd(v,1,0,xconsist,assumreal);
	}
    }

    else if ( !isShifted() )
    {
        if ( !isIndex() )
	{
            result = getOneProd(v,0,3,xconsist,assumreal);
	}

	else
	{
            result = getOneProd(v,1,3,xconsist,assumreal);
	}
    }

    else
    {
        SparseVector<gentype> vv(v);

	vv += dShift;
        vv /= dScale;

        if ( !isIndex() )
	{
            result = getOneProd(vv,0,0,xconsist,assumreal);
	}

	else
	{
            result = getOneProd(vv,1,0,xconsist,assumreal);
	}
    }

    removeOverwriteInd();
  }

  return tres;
}

int MercerKernel::oneProductDiverted(gentype &result, const SparseVector<gentype> &v, int xconsist, int assumreal) const
{
    // FIXME: modify so that copy constructor not needed.

    addinOverwriteInd(v);

    NiceAssert( isLeftRightNormal() );

    if ( !isShifted() && !isScaled() )
    {
        if ( !isIndex() )
	{
            getOneProd(result,v,0,0,xconsist,assumreal);
	}

	else
	{
            getOneProd(result,v,1,0,xconsist,assumreal);
	}
    }

    else if ( !isShifted() )
    {
        if ( !isIndex() )
	{
            getOneProd(result,v,0,3,xconsist,assumreal);
	}

	else
	{
            getOneProd(result,v,1,3,xconsist,assumreal);
	}
    }

    else
    {
        SparseVector<gentype> vv(v);

	vv += dShift;
        vv /= dScale;

        if ( !isIndex() )
	{
            getOneProd(result,vv,0,0,xconsist,assumreal);
	}

	else
	{
            getOneProd(result,vv,1,0,xconsist,assumreal);
	}
    }

    removeOverwriteInd();

    return result.isValEqn();
}

int MercerKernel::twoProductDiverted(double  &res, const SparseVector<gentype> &a, const SparseVector<gentype> &b, int xconsist, int assumreal) const
{
  int tres = 0;

  if ( !assumreal )
  {
    gentype temp(res);

    tres = twoProductDiverted(temp,a,b,xconsist,assumreal);

    res = tres ? 0.0 : (double) temp;
  }

  else
  {
//errstream() << "phantomxyzmerceraa 0\n";
    addinOverwriteInd(a,b);

    if ( ( !isShifted() && !isScaled() ) || isLeftRightPlain() )
    {
        if ( !isIndex() )
	{
            res = getTwoProd(a,b,0,0,0,xconsist,assumreal);
	}

	else
	{
            res = getTwoProd(a,b,1,0,0,xconsist,assumreal);
	}
    }

    else if ( !isShifted() || isLeftRightPlain() )
    {
        if ( !isIndex() )
	{
            res = getTwoProd(a,b,0,0,3,xconsist,assumreal);
	}

	else
	{
            res = getTwoProd(a,b,1,0,3,xconsist,assumreal);
	}
    }

    else
    {
        // If isLeftPlain:
        //
        // x'(sc.*(y+sh)) = x'.diag(sc).(y+sh)
        //                = x'.diag(sc).y + x'.diag(sc).sh
        //                = x'.S.y + x'.S.sh
        //
        // If isRightPlain:
        //
        // (sc.*(x+sh))'y = (x+sh)'.diag(sc).y
        //                = x'.diag(sc).y + sh'.diag(sc).y
        //
        // else:
        //
	// ((x+sh).*sc)'((y+sh).*sc) = (x+sh)'.diag(sc.*sc).(y+sh)
	//                           = x'.diag(sc.*sc).y + x'.diag(sc.*sc).(sh./sc) + (sh./sc)'.diag(sc.*sc).y + sh'.diag(sc.*sc).sh
        //                           = x'.S.y + x'.S.dShift + dShift'.S.y + dShiftProd

        if ( isLeftPlain() )
        {
            NiceAssert( isRightNormal() );

            double tempb;

            if ( !isIndex() )
            {
                res   = getTwoProd(a,b,0,0,2,xconsist,assumreal);
                tempb = getTwoProd(dShift,b,0,0,2,xconsist,assumreal);
            }

            else
            {
                res   = getTwoProd(a,b,1,0,2,xconsist,assumreal);
                tempb = getTwoProd(dShift,b,1,0,2,xconsist,assumreal);
            }

            res += tempb;
        }

        else if ( isRightPlain() )
        {
            NiceAssert( isLeftNormal() );

            double tempa;

            if ( !isIndex() )
            {
                res   = getTwoProd(a,b,0,0,1,xconsist,assumreal);
                tempa = getTwoProd(a,dShift,0,0,1,xconsist,assumreal);
            }

            else
            {
                res   = getTwoProd(a,b,1,0,1,xconsist,assumreal);
                tempa = getTwoProd(a,dShift,1,0,1,xconsist,assumreal);
            }

            res += tempa;
        }

        else
        {
            NiceAssert( isLeftRightNormal() );

            double tempa;
            double tempb;

            if ( !isIndex() )
            {
                res   = getTwoProd(a,b,0,0,3,xconsist,assumreal);
                tempa = getTwoProd(a,dShift,0,0,3,xconsist,assumreal);
                tempb = getTwoProd(dShift,b,0,0,3,xconsist,assumreal);
            }

            else
            {
                res   = getTwoProd(a,b,1,0,3,xconsist,assumreal);
                tempa = getTwoProd(a,dShift,1,0,3,xconsist,assumreal);
                tempb = getTwoProd(dShift,b,1,0,3,xconsist,assumreal);
            }

            res += tempa;
            res += tempb;
            res += (double) dShiftProdNoConj;
        }
    }

    removeOverwriteInd();
  }

  return tres;
}

int MercerKernel::twoProductDiverted(gentype &result, const SparseVector<gentype> &x, const SparseVector<gentype> &y, int xconsist, int assumreal) const
{
//errstream() << "phantomxyzmerceraa 0\n";
    addinOverwriteInd(x,y);

    if ( ( !isShifted() && !isScaled() ) || isLeftRightPlain() )
    {
        if ( !isIndex() )
	{
            getTwoProd(result,x,y,0,0,0,xconsist,assumreal);
	}

	else
	{
            getTwoProd(result,x,y,1,0,0,xconsist,assumreal);
	}
    }

    else if ( !isShifted() || isLeftRightPlain() )
    {
        if ( !isIndex() )
	{
            getTwoProd(result,x,y,0,0,3,xconsist,assumreal);
	}

	else
	{
            getTwoProd(result,x,y,1,0,3,xconsist,assumreal);
	}
    }

    else
    {
        // If isLeftPlain:
        //
        // x'(sc.*(y+sh)) = x'.diag(sc).(y+sh)
        //                = x'.diag(sc).y + x'.diag(sc).sh
        //                = x'.S.y + x'.S.sh
        //
        // If isRightPlain:
        //
        // (sc.*(x+sh))'y = (x+sh)'.diag(sc).y
        //                = x'.diag(sc).y + sh'.diag(sc).y
        //
        // else:
        //
	// ((x+sh).*sc)'((y+sh).*sc) = (x+sh)'.diag(sc.*sc).(y+sh)
	//                           = x'.diag(sc.*sc).y + x'.diag(sc.*sc).(sh./sc) + (sh./sc)'.diag(sc.*sc).y + sh'.diag(sc.*sc).sh
        //                           = x'.S.y + x'.S.dShift + dShift'.S.y + dShiftProd

        if ( isLeftPlain() )
        {
            NiceAssert( isRightNormal() );

            gentype tempb;

            if ( !isIndex() )
            {
                getTwoProd(result,x,y,0,0,2,xconsist,assumreal);
                getTwoProd(tempb,dShift,y,0,0,2,xconsist,assumreal);
            }

            else
            {
                getTwoProd(result,x,y,1,0,2,xconsist,assumreal);
                getTwoProd(tempb,dShift,y,1,0,2,xconsist,assumreal);
            }

            result += tempb;
        }

        else if ( isRightPlain() )
        {
            NiceAssert( isLeftNormal() );

            gentype tempa;

            if ( !isIndex() )
            {
                getTwoProd(result,x,y,0,0,1,xconsist,assumreal);
                getTwoProd(tempa,x,dShift,0,0,1,xconsist,assumreal);
            }

            else
            {
                getTwoProd(result,x,y,1,0,1,xconsist,assumreal);
                getTwoProd(tempa,x,dShift,1,0,1,xconsist,assumreal);
            }

            result += tempa;
        }

        else
        {
            NiceAssert( isLeftRightNormal() );

            gentype tempa;
            gentype tempb;

            if ( !isIndex() )
            {
                getTwoProd(result,x,y,0,0,3,xconsist,assumreal);
                getTwoProd(tempa,x,dShift,0,0,3,xconsist,assumreal);
                getTwoProd(tempb,dShift,y,0,0,3,xconsist,assumreal);
            }

            else
            {
                getTwoProd(result,x,y,1,0,3,xconsist,assumreal);
                getTwoProd(tempa,x,dShift,1,0,3,xconsist,assumreal);
                getTwoProd(tempb,dShift,y,1,0,3,xconsist,assumreal);
            }

            result += tempa;
            result += tempb;
            result += dShiftProdNoConj;
        }
    }

    removeOverwriteInd();

//errstream() << "phantomxyzmerceraa 1\n";
    return result.isValEqn();
}

int MercerKernel::threeProductDiverted(double &result, const SparseVector<gentype> &v, const SparseVector<gentype> &w, const SparseVector<gentype> &x, int xconsist, int assumreal) const
{
  int tres = 0;

  if ( !assumreal )
  {
    gentype temp(result);

    tres = threeProductDiverted(temp,v,w,x,xconsist,assumreal);

    result = tres ? 0.0 : (double) temp;
  }

  else
  {
    // FIXME: modify so that copy constructor not needed.

    addinOverwriteInd(v,w,x);

    NiceAssert( isLeftRightNormal() );

    if ( !isShifted() && !isScaled() )
    {
        if ( !isIndex() )
	{
            result = getThreeProd(v,w,x,0,0,xconsist,assumreal);
	}

	else
	{
            result = getThreeProd(v,w,x,1,0,xconsist,assumreal);
	}
    }

    else if ( !isShifted() )
    {
        if ( !isIndex() )
	{
            result = getThreeProd(v,w,x,0,3,xconsist,assumreal);
	}

	else
	{
            result = getThreeProd(v,w,x,1,3,xconsist,assumreal);
	}
    }

    else
    {
        SparseVector<gentype> vv(v);
        SparseVector<gentype> ww(w);
        SparseVector<gentype> xx(x);

	vv += dShift;
        vv /= dScale;

	ww += dShift;
        ww /= dScale;

	xx += dShift;
        xx /= dScale;

        if ( !isIndex() )
	{
            result = getThreeProd(vv,ww,xx,0,0,xconsist,assumreal);
	}

	else
	{
            result = getThreeProd(vv,ww,xx,1,0,xconsist,assumreal);
	}
    }

    removeOverwriteInd();
  }

  return tres;
}

int MercerKernel::threeProductDiverted(gentype &result, const SparseVector<gentype> &v, const SparseVector<gentype> &w, const SparseVector<gentype> &x, int xconsist, int assumreal) const
{
    // FIXME: modify so that copy constructor not needed.

    addinOverwriteInd(v,w,x);

    NiceAssert( isLeftRightNormal() );

    if ( !isShifted() && !isScaled() )
    {
        if ( !isIndex() )
	{
            getThreeProd(result,v,w,x,0,0,xconsist,assumreal);
	}

	else
	{
            getThreeProd(result,v,w,x,1,0,xconsist,assumreal);
	}
    }

    else if ( !isShifted() )
    {
        if ( !isIndex() )
	{
            getThreeProd(result,v,w,x,0,3,xconsist,assumreal);
	}

	else
	{
            getThreeProd(result,v,w,x,1,3,xconsist,assumreal);
	}
    }

    else
    {
        SparseVector<gentype> vv(v);
        SparseVector<gentype> ww(w);
        SparseVector<gentype> xx(x);

	vv += dShift;
        vv /= dScale;

	ww += dShift;
        ww /= dScale;

	xx += dShift;
        xx /= dScale;

        if ( !isIndex() )
	{
            getThreeProd(result,vv,ww,xx,0,0,xconsist,assumreal);
	}

	else
	{
            getThreeProd(result,vv,ww,xx,1,0,xconsist,assumreal);
	}
    }

    removeOverwriteInd();

    return result.isValEqn();
}

int MercerKernel::fourProductDiverted(double &res, const SparseVector<gentype> &a, const SparseVector<gentype> &b, const SparseVector<gentype> &c, const SparseVector<gentype> &d, int xconsist, int assumreal) const
{
  int tres = 0;

  if ( !assumreal )
  {
    gentype temp(res);

    tres = fourProductDiverted(temp,a,b,c,d,xconsist,assumreal);

    res = tres ? 0.0 : (double) temp;
  }

  else
  {
    addinOverwriteInd(a,b,c,d);

    NiceAssert( isLeftRightNormal() );

    if ( !isShifted() && !isScaled() )
    {
        if ( !isIndex() )
	{
            res = getFourProd(a,b,c,d,0,0,xconsist,assumreal);
	}

	else
	{
            res = getFourProd(a,b,c,d,1,0,xconsist,assumreal);
	}
    }

    else if ( !isShifted() )
    {
        if ( !isIndex() )
	{
            res = getFourProd(a,b,c,d,0,3,xconsist,assumreal);
	}

	else
	{
            res = getFourProd(a,b,c,d,1,3,xconsist,assumreal);
	}
    }

    else
    {
        SparseVector<gentype> vv(a);
        SparseVector<gentype> ww(b);
        SparseVector<gentype> xx(c);
        SparseVector<gentype> yy(d);

	vv += dShift;
        vv /= dScale;

	ww += dShift;
        ww /= dScale;

	xx += dShift;
        xx /= dScale;

	yy += dShift;
        yy /= dScale;

        if ( !isIndex() )
	{
            res = getFourProd(vv,ww,xx,yy,0,0,xconsist,assumreal);
	}

	else
	{
            res = getFourProd(vv,ww,xx,yy,1,0,xconsist,assumreal);
	}
    }

    removeOverwriteInd();
  }

  return tres;
}

int MercerKernel::fourProductDiverted(gentype &result, const SparseVector<gentype> &v, const SparseVector<gentype> &w, const SparseVector<gentype> &x, const SparseVector<gentype> &y, int xconsist, int assumreal) const
{
    // FIXME: modify so that copy constructor not needed.

    addinOverwriteInd(v,w,x,y);

    NiceAssert( isLeftRightNormal() );

    if ( !isShifted() && !isScaled() )
    {
        if ( !isIndex() )
	{
            getFourProd(result,v,w,x,y,0,0,xconsist,assumreal);
	}

	else
	{
            getFourProd(result,v,w,x,y,1,0,xconsist,assumreal);
	}
    }

    else if ( !isShifted() )
    {
        if ( !isIndex() )
	{
            getFourProd(result,v,w,x,y,0,3,xconsist,assumreal);
	}

	else
	{
            getFourProd(result,v,w,x,y,1,3,xconsist,assumreal);
	}
    }

    else
    {
        SparseVector<gentype> vv(v);
        SparseVector<gentype> ww(w);
        SparseVector<gentype> xx(x);
        SparseVector<gentype> yy(y);

	vv += dShift;
        vv /= dScale;

	ww += dShift;
        ww /= dScale;

	xx += dShift;
        xx /= dScale;

	yy += dShift;
        yy /= dScale;

        if ( !isIndex() )
	{
            getFourProd(result,vv,ww,xx,yy,0,0,xconsist,assumreal);
	}

	else
	{
            getFourProd(result,vv,ww,xx,yy,1,0,xconsist,assumreal);
	}
    }

    removeOverwriteInd();

    return result.isValEqn();
}

int MercerKernel::mProductDiverted(int m, double &result, const Vector<const SparseVector<gentype> *> &a, int xconsist, int assumreal) const
{
  int tres = 0;

  if ( !assumreal )
  {
    gentype temp(result);

    tres = mProductDiverted(m,temp,a,xconsist,assumreal);

    result = tres ? 0.0 : (double) temp;
  }

  else
  {
    NiceAssert( ( m >= 0 ) && ( m <= a.size() ) );

    addinOverwriteInd(a);

    // FIXME: modify so that copy constructor not needed

    NiceAssert( isLeftRightNormal() );

    {
        if ( !isIndex() )
	{
            retVector<const SparseVector<gentype> *> tmpva;

            result = getmProd(a(0,1,m-1,tmpva),0,0,xconsist,assumreal);
	}

	else
	{
            retVector<const SparseVector<gentype> *> tmpva;

            result = getmProd(a(0,1,m-1,tmpva),1,0,xconsist,assumreal);
	}
    }

    removeOverwriteInd();
  }

  return tres;
}

int MercerKernel::mProductDiverted(int m, gentype &result, const Vector<const SparseVector<gentype> *> &a, int xconsist, int assumreal) const
{
    NiceAssert( ( m >= 0 ) && ( m <= a.size() ) );

    addinOverwriteInd(a);

    // FIXME: modify so that copy constructor not needed

    NiceAssert( isLeftRightNormal() );

    {
        if ( !isIndex() )
	{
            retVector<const SparseVector<gentype> *> tmpva;

            getmProd(result,a(0,1,m-1,tmpva),0,0,xconsist,assumreal);
	}

	else
	{
            retVector<const SparseVector<gentype> *> tmpva;

            getmProd(result,a(0,1,m-1,tmpva),1,0,xconsist,assumreal);
	}
    }

    removeOverwriteInd();

    return result.isValEqn();
}

















void MercerKernel::fillXYMatrix(double &altxyr00, double &altxyr10, double &altxyr11, double &altxyr20, double &altxyr21, double &altxyr22, 
                                const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, 
                                const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, 
                                const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, 
                                int doanyhow, int assumreal) const
{
    NiceAssert( needsMatDiff() != -1 );

    if ( xy00 ) { altxyr00 = *xy00; }
    if ( xy10 ) { altxyr10 = *xy10; }
    if ( xy11 ) { altxyr11 = *xy11; }
    if ( xy20 ) { altxyr20 = *xy20; }
    if ( xy21 ) { altxyr21 = *xy21; }
    if ( xy22 ) { altxyr22 = *xy22; }

    if ( !xy00 && ( needsMatDiff() || doanyhow ) ) { altxyr00 = (double) getmnorm(xainfo,xa,2,0,assumreal); }
    if ( !xy10 && ( needsMatDiff() || doanyhow ) ) { innerProductDiverted(altxyr10,xb,xa);                    }
    if ( !xy11 && ( needsMatDiff() || doanyhow ) ) { altxyr11 = (double) getmnorm(xbinfo,xb,2,0,assumreal); }
    if ( !xy20 && ( needsMatDiff() || doanyhow ) ) { innerProductDiverted(altxyr20,xc,xa);                    }
    if ( !xy21 && ( needsMatDiff() || doanyhow ) ) { innerProductDiverted(altxyr21,xc,xb);                    }
    if ( !xy22 && ( needsMatDiff() || doanyhow ) ) { altxyr22 = (double) getmnorm(xcinfo,xc,2,0,assumreal); }

    return;
}

void MercerKernel::fillXYMatrix(double &altxyr00, double &altxyr10, double &altxyr11, double &altxyr20, double &altxyr21, double &altxyr22, double &altxyr30, double &altxyr31, double &altxyr32, double &altxyr33, 
                                const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, 
                                const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, 
                                const double *xy00, const double *xy10, const double *xy11, const double *xy20, const double *xy21, const double *xy22, const double *xy30, const double *xy31, const double *xy32, const double *xy33, 
                                int doanyhow, int assumreal) const
{
    if ( xy00 ) { altxyr00 = *xy00; }
    if ( xy10 ) { altxyr10 = *xy10; }
    if ( xy11 ) { altxyr11 = *xy11; }
    if ( xy20 ) { altxyr20 = *xy20; }
    if ( xy21 ) { altxyr21 = *xy21; }
    if ( xy22 ) { altxyr22 = *xy22; }
    if ( xy30 ) { altxyr30 = *xy30; }
    if ( xy31 ) { altxyr31 = *xy31; }
    if ( xy32 ) { altxyr32 = *xy32; }
    if ( xy33 ) { altxyr32 = *xy33; }

    if ( !xy00 && (   needsMatDiff()         || doanyhow ) ) { altxyr00 = (double) getmnorm(xainfo,xa,2,0,assumreal); }
    if ( !xy10 && (   needsMatDiff()         || doanyhow ) ) { innerProductDiverted(altxyr10,xb,xa);                    }
    if ( !xy11 && (   needsMatDiff()         || doanyhow ) ) { altxyr11 = (double) getmnorm(xbinfo,xb,2,0,assumreal); }
    if ( !xy20 && ( ( needsMatDiff() == +1 ) || doanyhow ) ) { innerProductDiverted(altxyr20,xc,xa);                    }
    if ( !xy21 && ( ( needsMatDiff() == +1 ) || doanyhow ) ) { innerProductDiverted(altxyr21,xc,xb);                    }
    if ( !xy22 && (    needsMatDiff()        || doanyhow ) ) { altxyr22 = (double) getmnorm(xcinfo,xc,2,0,assumreal); }
    if ( !xy30 && ( ( needsMatDiff() == +1 ) || doanyhow ) ) { innerProductDiverted(altxyr30,xd,xa);                    }
    if ( !xy31 && ( ( needsMatDiff() == +1 ) || doanyhow ) ) { innerProductDiverted(altxyr31,xd,xb);                    }
    if ( !xy32 && (    needsMatDiff()        || doanyhow ) ) { innerProductDiverted(altxyr32,xd,xc);                    }
    if ( !xy33 && (    needsMatDiff()        || doanyhow ) ) { altxyr33 = (double) getmnorm(xdinfo,xd,2,0,assumreal); }

    return;
}

const Matrix<double> &MercerKernel::fillXYMatrix(int m, Matrix<double> &altres, Vector<const SparseVector<gentype> *> &x, Vector<const vecInfo *> &xinfo, const Matrix<double> *optionCache, int doanyhow, int assumreal) const
{
    NiceAssert( !optionCache || ( ( (*optionCache).numRows() == m ) && ( (*optionCache).numCols() == m ) ) );

//    if ( !optionCache && ( ( ( m >= 4 ) && needsMatDiff() ) || doanyhow ) )
    if ( !optionCache && ( needsMatDiff() || doanyhow ) )
    {
        altres.resize(m,m);

        int i,j;

        for ( i = 0 ; i < m ; ++i )
        {
            altres("&",i,i) = (double) getmnorm(*(xinfo(i)),*(x(i)),2,0,assumreal);

            if ( i )
            {
                for ( j = 0 ; j < i ; ++j )
                {
                    if ( doanyhow || ( needsMatDiff() == +1 ) || ( ( needsMatDiff() == -1 ) && ( j = i-1 ) && ( !(i%2) ) ) )
                    {
                        innerProductDiverted(altres("&",i,j),*(x(i)),*(x(j)));
                        altres("&",j,i) = altres("&",i,j);
                    }
                }
            }
        }
    }

    return optionCache ? *optionCache : altres;
}


//void MercerKernel::diff3norm(gentype &res, const gentype &xyprod, const gentype &xanorm, const gentype &xbnorm, const gentype &xcnorm, double xy00, double xy10, double xy11, double xy20, double xy21, double xy22, const Vector<int> *s) const
void MercerKernel::diff3norm(gentype &res, const gentype &xyprod, const gentype &xanorm, const gentype &xbnorm, const gentype &xcnorm, double xy00, double xy10, double xy11, double xy20, double xy21, double xy22, const Vector<int> *) const
{
    //(void) s;

    // xanorm, ... are ||xa||_m^m if isAltDiff == 0, ||xa||_2^2 otherwise

    if ( isAltDiff() == 0 )
    {
        res  = xyprod;
        res *= -3.0;
        res += xanorm;
        res += xbnorm;
        res += xcnorm;
    }

    else if ( isAltDiff() == 1 )
    {
        res  = xyprod;
        res *= -2.0;
        res += xanorm;
        res += xbnorm;
        res += xcnorm;
    }

    else if ( isAltDiff() == 2 )
    {
        res  = xanorm;
        res += xbnorm;
        res += xcnorm;

        res -= ( xy00 + xy10 + xy20 )/3.0;
        res -= ( xy10 + xy11 + xy21 )/3.0;
        res -= ( xy20 + xy21 + xy22 )/3.0;

        res *= 2.0;
    }

    else
    {
        NiceThrow("diff3norm not defined for altdiff != 0,1,2 (including 5)");
    }

    return;
}

void MercerKernel::diff4norm(gentype &res, const gentype &xyprod, const gentype &xanorm, const gentype &xbnorm, const gentype &xcnorm, const gentype &xdnorm, double xy00, double xy10, double xy11, double xy20, double xy21, double xy22, double xy30, double xy31, double xy32, double xy33, const Vector<int> *s) const
{
    // xanorm, ... are ||xa||_m^m if isAltDiff == 0, ||xa||_2^2 otherwise

    int z = 0;

    if ( isAltDiff() == 0 )
    {
        res  = xyprod;
        res *= -4.0;

        res += xanorm;
        res += xbnorm;
        res += xcnorm;
        res += xdnorm;
    }

    else if ( isAltDiff() == 1 )
    {
        res  = xyprod;
        res *= -2.0;

        res += xanorm;
        res += xbnorm;
        res += xcnorm;
        res += xdnorm;
    }

    else if ( isAltDiff() == 2 )
    {
        res  = xanorm;
        res += xbnorm;
        res += xcnorm;
        res += xdnorm;

        res -= ( xy00 + xy10 + xy20 + xy30)/4.0;
        res -= ( xy10 + xy11 + xy21 + xy31)/4.0;
        res -= ( xy20 + xy21 + xy22 + xy32)/4.0;
        res -= ( xy30 + xy31 + xy32 + xy33)/4.0;

        res *= 2.0;
    }

    else if ( isAltDiff() == 5 )
    {
        res  = -2.0*(xy10+xy32);

        res += xanorm;
        res += xbnorm;
        res += xcnorm;
        res += xdnorm;
    }

    else if ( isAltDiff() == 103 )
    {
        NiceAssert(s);

        res  = ((*s)(z)*(*s)(z)*xy00) + ((*s)(z)*(*s)(1)*xy10) + ((*s)(z)*(*s)(2)*xy20) + ((*s)(z)*(*s)(3)*xy30);
        res += ((*s)(1)*(*s)(z)*xy10) + ((*s)(1)*(*s)(1)*xy11) + ((*s)(1)*(*s)(2)*xy21) + ((*s)(1)*(*s)(3)*xy31);
        res += ((*s)(2)*(*s)(z)*xy20) + ((*s)(2)*(*s)(1)*xy21) + ((*s)(2)*(*s)(2)*xy22) + ((*s)(2)*(*s)(3)*xy32);
        res += ((*s)(3)*(*s)(z)*xy30) + ((*s)(3)*(*s)(1)*xy31) + ((*s)(3)*(*s)(2)*xy32) + ((*s)(3)*(*s)(3)*xy33);
    }

    else if ( isAltDiff() == 104 )
    {
        NiceAssert(s);

        Matrix<double> xxyy(4,4);

        xxyy("&",z,z) = xy00; xxyy("&",z,1) = xy10; xxyy("&",z,2) = xy20; xxyy("&",z,3) = xy30;
        xxyy("&",1,z) = xy10; xxyy("&",1,1) = xy11; xxyy("&",1,2) = xy21; xxyy("&",1,3) = xy31;
        xxyy("&",2,z) = xy20; xxyy("&",2,1) = xy21; xxyy("&",2,2) = xy22; xxyy("&",2,3) = xy32;
        xxyy("&",3,z) = xy30; xxyy("&",3,1) = xy31; xxyy("&",3,2) = xy32; xxyy("&",3,3) = xy33;

        res  =  xxyy((*s)(z),(*s)(z)) - xxyy((*s)(z),(*s)(1));
        res += -xxyy((*s)(1),(*s)(z)) + xxyy((*s)(1),(*s)(1));

        res +=  xxyy((*s)(2),(*s)(2)) - xxyy((*s)(2),(*s)(3));
        res += -xxyy((*s)(3),(*s)(2)) + xxyy((*s)(3),(*s)(3));
    }

    return;
}

void MercerKernel::diffmnorm(int m, gentype &res, const gentype &xyprod, const Vector<const gentype *> &xanorm, const Matrix<double> &xxyy, const Vector<int> *ss) const
{
    // xanorm, ... are ||xa||_m^m if isAltDiff == 0, ||xa||_2^2 otherwise

    int i,j;

    if ( isAltDiff() == 0 )
    {
        res  = xyprod;
        res *= -m;

        if ( m )
        {
            int i;

            for ( i = 0 ; i < m ; ++i )
            {
                res += (*(xanorm(i)));
            }
        }
    }

    else if ( isAltDiff() == 1 )
    {
        res  = xyprod;
        res *= -2.0;

        if ( m )
        {
            int i;

            for ( i = 0 ; i < m ; ++i )
            {
                res += (*(xanorm(i)));
            }
        }
    }

    else if ( isAltDiff() == 2 )
    {
        res = 0.0;

        if ( m )
        {
            for ( i = 0 ; i < m ; ++i )
            {
                res += (*(xanorm(i)));
            }

            for ( i = 0 ; i < m ; ++i )
            {
                for ( j = 0 ; j < m ; ++j )
                {
                    res -= (xxyy(i,j))/((double) m);
                }
            }
        }

        res *= 2.0;
    }

    else if ( isAltDiff() == 5 )
    {
        NiceAssert( !(m%2) );

        res = 0.0;

        if ( m )
        {
            for ( i = 0 ; i < m ; i += 2 )
            {
                res += -2.0*xxyy(i,i+1);

                res += (*(xanorm(i)));
                res += (*(xanorm(i+1)));
            }
        }
    }

    else if ( isAltDiff() == 103 )
    {
        res = 0.0;

        NiceAssert(ss);

        for ( i = 0 ; i < m ; ++i )
        {
            for ( j = 0 ; j < m ; ++j )
            {
                res += (*ss)(i)*(*ss)(j)*xxyy(i,j);
            }
        }
    }

    else if ( isAltDiff() == 104 )
    {
        res = 0.0;

        NiceAssert(ss);

        for ( i = 0 ; i < m ; i += 2 )
        {
            res +=  xxyy((*ss)(i  ),(*ss)(i  )) - xxyy((*ss)(i  ),(*ss)(i+1));
            res += -xxyy((*ss)(i+1),(*ss)(i  )) + xxyy((*ss)(i+1),(*ss)(i+1));
        }
    }

    return;
}

//void MercerKernel::diff3norm(double &res, const double &xyprod, const double &xanorm, const double &xbnorm, const double &xcnorm, double xy00, double xy10, double xy11, double xy20, double xy21, double xy22, const Vector<int> *s) const
double MercerKernel::diff3norm(double xyprod, double xanorm, double xbnorm, double xcnorm, double xy00, double xy10, double xy11, double xy20, double xy21, double xy22, const Vector<int> *) const
{
    double res = 0;

    //(void) s;

    // xanorm, ... are ||xa||_m^m if isAltDiff == 0, ||xa||_2^2 otherwise

    if ( isAltDiff() == 0 )
    {
        res = xanorm+xbnorm+xcnorm-(3*xyprod);
    }

    else if ( isAltDiff() == 1 )
    {
        res = xanorm+xbnorm+xcnorm-(2*xyprod);
    }

    else if ( isAltDiff() == 2 )
    {
        res = xanorm+xbnorm+xcnorm;

        res -= ( xy00 + xy10 + xy20 )/3.0;
        res -= ( xy10 + xy11 + xy21 )/3.0;
        res -= ( xy10 + xy21 + xy22 )/3.0;

        res *= 2.0;
    }

    else
    {
        NiceThrow("diff3norm not defined for altdiff != 0,1,2 or 5 (m odd)");
    }

    return res;
}

double MercerKernel::diff4norm(double xyprod, double xanorm, double xbnorm, double xcnorm, double xdnorm, double xy00, double xy10, double xy11, double xy20, double xy21, double xy22, double xy30, double xy31, double xy32, double xy33, const Vector<int> *s) const
{
    double res = 0;

    // xanorm, ... are ||xa||_m^m if isAltDiff == 0, ||xa||_2^2 otherwise

    int z = 0;

    if ( isAltDiff() == 0 )
    {
        res = xanorm+xbnorm+xcnorm+xdnorm-(4*xyprod);
//errstream() << "phantomxyzmer 0: " << res << " = " << xanorm << "+" << xbnorm << "+" << xcnorm << "+" << xdnorm << "-(4*" << xyprod << ")\n";
    }

    else if ( isAltDiff() == 1 )
    {
        res = xanorm+xbnorm+xcnorm+xdnorm-(2*xyprod);
    }

    else if ( isAltDiff() == 2 )
    {
        res = xanorm+xbnorm+xcnorm+xdnorm;

        res -= ( xy00 + xy10 + xy20 + xy30)/4.0;
        res -= ( xy10 + xy11 + xy21 + xy31)/4.0;
        res -= ( xy20 + xy21 + xy22 + xy32)/4.0;
        res -= ( xy30 + xy31 + xy32 + xy33)/4.0;

        res *= 2.0;
    }

    else if ( isAltDiff() == 5 )
    {
        res  = -2.0*(xy10+xy32);

        res += xanorm;
        res += xbnorm;
        res += xcnorm;
        res += xdnorm;
//errstream() << "phantomxyzmer: " << res << " = -2.(" << xy10 << "+" << xy32 << ")+" << xanorm << "+" << xbnorm << "+" << xcnorm << "+" << xdnorm << "\n";
    }

    else if ( isAltDiff() == 103 )
    {
        NiceAssert(s);

        res  = ((*s)(z)*(*s)(z)*xy00) + ((*s)(z)*(*s)(1)*xy10) + ((*s)(z)*(*s)(2)*xy20) + ((*s)(z)*(*s)(3)*xy30);
        res += ((*s)(1)*(*s)(z)*xy10) + ((*s)(1)*(*s)(1)*xy11) + ((*s)(1)*(*s)(2)*xy21) + ((*s)(1)*(*s)(3)*xy31);
        res += ((*s)(2)*(*s)(z)*xy20) + ((*s)(2)*(*s)(1)*xy21) + ((*s)(2)*(*s)(2)*xy22) + ((*s)(2)*(*s)(3)*xy32);
        res += ((*s)(3)*(*s)(z)*xy30) + ((*s)(3)*(*s)(1)*xy31) + ((*s)(3)*(*s)(2)*xy32) + ((*s)(3)*(*s)(3)*xy33);
    }

    else if ( isAltDiff() == 104 )
    {
        NiceAssert(s);

        Matrix<double> xxyy(4,4);

        xxyy("&",z,z) = xy00; xxyy("&",z,1) = xy10; xxyy("&",z,2) = xy20; xxyy("&",z,3) = xy30;
        xxyy("&",1,z) = xy10; xxyy("&",1,1) = xy11; xxyy("&",1,2) = xy21; xxyy("&",1,3) = xy31;
        xxyy("&",2,z) = xy20; xxyy("&",2,1) = xy21; xxyy("&",2,2) = xy22; xxyy("&",2,3) = xy32;
        xxyy("&",3,z) = xy30; xxyy("&",3,1) = xy31; xxyy("&",3,2) = xy32; xxyy("&",3,3) = xy33;

        res  =  xxyy((*s)(z),(*s)(z)) - xxyy((*s)(z),(*s)(1));
        res += -xxyy((*s)(1),(*s)(z)) + xxyy((*s)(1),(*s)(1));

        res +=  xxyy((*s)(2),(*s)(2)) - xxyy((*s)(2),(*s)(3));
        res += -xxyy((*s)(3),(*s)(2)) + xxyy((*s)(3),(*s)(3));
    }

    return res;
}

double MercerKernel::diffmnorm(int m, double xyprod, const Vector<const double *> &xanorm, const Matrix<double> &xxyy, const Vector<int> *ss) const
{
    double res = 0;

    // xanorm, ... are ||xa||_m^m if isAltDiff == 0, ||xa||_2^2 otherwise

    int i,j;

    if ( isAltDiff() == 0 )
    {
        res = -m*xyprod;

        if ( m )
        {
            int i;

            for ( i = 0 ; i < m ; ++i )
            {
                res += *(xanorm(i));
            }
        }
    }

    else if ( isAltDiff() == 1 )
    {
        res = -2*xyprod;

        if ( m )
        {
            int i;

            for ( i = 0 ; i < m ; ++i )
            {
                res += *(xanorm(i));
            }
        }
    }

    else if ( isAltDiff() == 2 )
    {
        res = 0.0;

        if ( m )
        {
            for ( i = 0 ; i < m ; ++i )
            {
                res += (*xanorm(i));
            }

            for ( i = 0 ; i < m ; ++i )
            {
                for ( j = 0 ; j < m ; ++j )
                {
                    res -= xxyy(i,j)/m;
                }
            }
        }

        res *= 2.0;
    }

    else if ( isAltDiff() == 5 )
    {
        NiceAssert( !(m%2) );

        res = 0.0;

        if ( m )
        {
            for ( i = 0 ; i < m ; i += 2 )
            {
                res += -2.0*xxyy(i,i+1);

                res += (*(xanorm(i)));
                res += (*(xanorm(i+1)));
            }
        }
    }

    else if ( isAltDiff() == 103 )
    {
        res = 0.0;

        NiceAssert(ss);

        for ( i = 0 ; i < m ; ++i )
        {
            for ( j = 0 ; j < m ; ++j )
            {
                res += (*ss)(i)*(*ss)(j)*xxyy(i,j);
            }
        }
    }

    else if ( isAltDiff() == 104 )
    {
        res = 0.0;

        NiceAssert(ss);

        for ( i = 0 ; i < m ; i += 2 )
        {
            res +=  xxyy((*ss)(i  ),(*ss)(i  )) - xxyy((*ss)(i  ),(*ss)(i+1));
            res += -xxyy((*ss)(i+1),(*ss)(i  )) + xxyy((*ss)(i+1),(*ss)(i+1));
        }
    }

    return res;
}








































// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================

void Bn(gentype &res, int n, const gentype &z);
void Jn(gentype &res, int n, const gentype &z);

void Bn(gentype &res, int n, const gentype &z)
{
    int k;
    gentype x,xx,xxx;

    res = 0;

    for ( k = 0 ; k <= n+1 ; ++k )
    {
        x =  z;
        x += (((double) n)+1)/2.0;
        x -= (double) k;

        xx = (double) n;
        xxx = 0.0;

        res += ((real(x)>=xxx)?pow(x,xx):xxx)*(xnCr(n+1,k)*((k%2)?-1.0:+1.0)/((double) xnfact(n)));
    }

    return;
}


void Jn(gentype &res, int n, const gentype &z)
{
    // Jn(x) = sin^(2n+1) (-1/sin(x) d/dx)^n (pi-x)/sin(x)
    //
    // See: Youngmin Cho, Lawrence K. Saul - Kernel Methods for Deep Learning

    NiceAssert( n >= 0 );

    switch ( n )
    {
        case 0:
        {
            // pi - z

            res =  NUMBASE_PI;
            res -= z;

            break;
        }

        case 1:
        {
            // sin(z) + (pi-z).cos(z)
         
            res =  NUMBASE_PI;
            res -= z;
            res *= cos(z);
            res += sin(z);

            break;
        }

        case 2:
        {
            // 3.sin(z).cos(z) + (pi-z).( 1 + 2.cos(z).cos(z) )

            gentype cosz;
            gentype sinz;
            gentype piminusz;
            gentype threesincos;

            cosz = cos(z);
            sinz = sin(z);

            piminusz =  NUMBASE_PI;
            piminusz -= z;

            threesincos =  cosz;
            threesincos *= sinz;
            threesincos *= 3.0;

            res =  cosz;
            res *= cosz;
            res *= 2.0;
            res += 1.0;
            res *= piminusz;
            res += threesincos;

            break;
        }

        default:
        {
            static thread_local SparseVector<gentype> JnFuncsVol;
            SparseVector<gentype> &JnFuncs = JnFuncsVol;

            const static thread_local gentype sinfn("sin(x)");
            const static thread_local gentype cosecfn("cosec(x)");

            // We need to calculate the function.  Because this is slow,
            // we want to keep that function once calculated.  The sparse
            // vector JnFuncs does this, and gentype allows us to do the
            // requisit algebra to calculate the function.
            //
            // General form is:
            //
            // Jn(x) = (-1)^n sin(x)^(2n+1) (1/sin(x) d/dx)^n (pi-x)/sin(x)
            //       = (-1)^n sin(x)^(2n+1) (cosec(x) d/dx)^n (pi-x).cosec(x)
            //
            // Change of variables:
            //
            // d/dy = 1/sin(x) d/dx
            //

            if ( !(JnFuncs.isindpresent(n)) )
            {
                JnFuncs("&",n) = "(pi()-x).*cosec(x)";

                int i;

                if ( n )
                {
                    for ( i = 0 ; i < n ; ++i )
                    {
                        JnFuncs("&",n).realDeriv(0,0);
                        JnFuncs("&",n) *= cosecfn;
                    }
                }

                for ( i = 0 ; i < (2*n)+1 ; ++i )
                {
                    JnFuncs("&",n) *= sinfn;
                }

                if ( n%2 )
                {
                    JnFuncs("&",n).negate();
                }
            }

            gentype zz(z);

            // One catch: gentype cannot simplify the equations, and in
            // particular it will leave sin(x) factors on the denominator.
            // This can lead to divide by zero errors.  The solution is to
            // slightly perturb z if |sin(z)| is too small.
            //
            // Note that for sufficiently small z sin(z) = z, or close enough.
            // Note also the periodicity of z, and the need to keep the angle

            if ( (double) abs2(sin(zz)) < BADZEROTOL )
            {
                if ( (double) abs2(zz) < BADZEROTOL )
                {
                    // No angle to reliably preserve

                    zz = BADZEROTOL;
                }

                else
                {
                    // Preserve the angle

                    while ( (double) abs2(sin(zz)) < BADZEROTOL )
                    {
                        zz *= (1+BADZEROTOL);
                    }
                }
            }

            res = JnFuncs(n)(zz);

            break;
        }
    }

    return;
}















// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================


















































// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================


















































// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================


















































// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================

//KERNELSHERE

kernInfo &fillOutInfo(kernInfo &res, Vector<gentype> &r, Vector<int> &ic, int ktype);

void MercerKernel::Kbase(gentype &res, int q, int typeis,
                         const gentype &txyprod, const gentype &tyxprod, const gentype &diffis,
                         Vector<const SparseVector<gentype> *> &x,
                         Vector<const vecInfo *> &xinfo,
                         Vector<const gentype *> &xnorm,
                         Vector<int> &iiii,
                         int xdim, int m, int adensetype, int bdensetype, int resmode, int mlid) const
{
    // Design decision: ii(0),ii(1) form i,j pair for base, rest ignored

  NiceAssert( q >= 0 );
  NiceAssert( q < size() );
  NiceAssert( typeis != -1 );

    double symm = +1;
    bool isdoubleintverRBF = false;

  if ( ( typeis >= 800 ) && ( typeis <= 899 ) )
  {
    kernel8xx(q,res,q,typeis,txyprod,tyxprod,diffis,x,xinfo,iiii,xdim,m,resmode,mlid);
  }

  else if ( resmode == 0x80 )
  {
    res = 0.0;
  }

  else if ( resmode == 0 )
  {
    int KusesVector  = kinf(q).usesVector;
    int KusesMinDiff = kinf(q).usesMinDiff;
    int KusesMaxDiff = kinf(q).usesMaxDiff;

    if ( adensetype || bdensetype )
    {
        // Actually want dense derivative

        if ( ( typeis == 3 ) && ( adensetype == 2 ) && ( bdensetype == 2 ) )
        {
            isdoubleintverRBF = true; // this is a special case
        }

        else
        {
            // The integrated RBF is a special case

            symm = calcDensePair(typeis,adensetype,bdensetype,xdim);
        }

        NiceAssert( typeis >= 0 );

        kernInfo altkinf;
        Vector<gentype> rrr; // dummy vector
        Vector<int> icic;    // dummy vector

        fillOutInfo(altkinf,rrr,icic,typeis);

        KusesVector  = altkinf.usesVector;
        KusesMinDiff = altkinf.usesMinDiff;
        KusesMaxDiff = altkinf.usesMaxDiff;
    }

    gentype xyprod;

    xyprod =  txyprod;
    xyprod += tyxprod;
    xyprod /= 2.0;

    // (I suspect this will make the kernel Mercer, but haven't proven it)

    // Apply to first two only (design decision)

    processOverwrites(q,*(x(0)),*(x(1)));

    SparseVector<gentype> xx;
    SparseVector<gentype> yy;

    int i,j;

    if ( KusesVector || KusesMinDiff || KusesMaxDiff )
    {
        if ( m != 2 )
        {
            NiceThrow("Vector-function kernels are not implemented for m != 2");
        }

        xx = *(x(0));
        yy = *(x(1));

        if ( isLeftNormal()  ) { preShiftScale(xx,*(x(0))); }
        if ( isRightNormal() ) { preShiftScale(yy,*(x(1)));         }

        xx.conj();

        xx -= yy;
        xx += yy; // xx now has all indices
        yy -= xx;
        yy += xx; // yy now has all indices, shared with xx
    }

    retVector<gentype> tmpva;

    const Vector<gentype> &r = dRealConstants(q)(1,1,dRealConstants(q).size()-1,tmpva);
    const Vector<int> &ic = dIntConstants(q);

//ADDHERE - new kernel implementations go here
    switch ( typeis )
    {
        // Kernels descriptions:
        //
        // rj = real constant j
        // ij = integer constant j
        // a = x'x
        // b = y'y
        // z = x'y
        // d = ||x-y||^2

        case 0:
        {
            res = r(1);
            break;
        }

        case 1:
        {
            res = xyprod;
            res /= r(0);
            res /= r(0);
            break;
        }

        case 2:
        {
            res = xyprod;
            res /= r(0);
            res /= r(0);
            res += r(1);
            raiseto(res,ic(0));

            break;
        }

        case 3:
        case 1003:
        {
            NiceAssert( !isdoubleintverRBF || ( typeis == 3 ) );

            if ( !isdoubleintverRBF )
            {
                res = diffis;
                res /= -2.0;
                res /= r(0);
                res /= r(0);
                res -= r(1);
                OP_exp(res);
                res *= AltDiffNormConst(xdim,m,r(0));

                if ( typeis == 1003 )
                {
                    for ( i = 0 ; i < yy.nindsize() ; ++i )
                    {
                        yy.direref(i) -= xx.direcref(i);

                        res *= yy.direcref(i);
                        res /= r(0);
                        res /= r(0);
                    }
                }
            }

            else
            {
                // See case 2003 below, but this is the double integral case

                // see calcKRBFSymmKern

                res =  r(1);
                res *= -1.0;
                OP_exp(res);
                res *= AltDiffNormConst(xdim,m,r(0));

                // Recall that xx and yy are already set up with a common index set

                gentype zp(xdenseZeroPoint);

                xx /= r(0);
                yy /= r(0);

                SparseVector<gentype> xdiff(xx);
                SparseVector<gentype> &xalb = yy;
                SparseVector<gentype> &xblb = xx;

                xdiff -= yy;

                xalb -= zp;
                xalb.negate();

                xblb -= zp;

                for ( i = 0 ; i < xx.nindsize() ; ++i )
                {
                    //res *= (NUMBASE_SQRTPI/2)*r(0)*r(0)*((xblb[i]*erf(xblb[i]))+(xalb[i]*erf(xalb[i]))-(xdiff[i]*erf(xdiff[i]))+((exp(-xalb[i]*xalb[i])+exp(-xblb[i]*xblb[i])-exp(-xdiff[i]*xdiff[i])-1)/NUMBASE_SQRTPI));
                    res *= (NUMBASE_SQRTPI/2)*r(0)*r(0)*((xblb.direcref(i)*erf(xblb.direcref(i)))+(xalb.direcref(i)*erf(xalb.direcref(i)))-(xdiff.direcref(i)*erf(xdiff.direcref(i)))+((exp(-xalb.direcref(i)*xalb.direcref(i))+exp(-xblb.direcref(i)*xblb.direcref(i))-exp(-xdiff.direcref(i)*xdiff.direcref(i))-1.0_gent)/NUMBASE_SQRTPI));
                }
            }

            break;
        }

        case 2003:
        {
            // see calcKRBFSymmKern

            res =  r(1);
            res *= -1.0;
            OP_exp(res);
            res *= AltDiffNormConst(xdim,m,r(0));

            if ( symm == +1 )
            {
                // Recall that xx and yy are already set up with a common index set

                gentype zp(xdenseZeroPoint);

                xx /= r(0);
                yy /= r(0);

                xx -= yy; // This is xdiff in calcKRBFSymmKern

                yy -= zp;
                yy.negate(); // This is xalb in calcKRBFSymmKern

                for ( i = 0 ; i < xx.nindsize() ; ++i )
                {
                    //res *= (NUMBASE_SQRTPI/2)*r(0)*(erf(xdiff[i])-erf(xalb[i]));
                    res *= (NUMBASE_SQRTPI/2);
                    res *= r(0);
                    res *= (erf(xx.direcref(i))-erf(yy.direcref(i)));
                }
            }

            else
            {
                NiceAssert( symm == -1 );

                // Recall that xx and yy are already set up with a common index set

                gentype zp(xdenseZeroPoint);

                xx /= r(0);
                yy /= r(0);

                yy -= xx;
                yy.negate(); // This is xdiff in calcKRBFSymmKern

                xx -= zp; // This is xblb in calcKRBFSymmKern

                for ( i = 0 ; i < xdim ; ++i )
                {
                    //res *= (NUMBASE_SQRTPI/2)*r(0)*(erf(xblb[i])-erf(xdiff[i]));
                    res *= (NUMBASE_SQRTPI/2);
                    res *= r(0);
                    res *= (erf(xx.direcref(i))-erf(yy.direcref(i)));
                }

                res *= symm; // pre-correct for negation just before return
            }

            break;
        }

        case 4:
        {
            res = diffis;
            OP_sqrt(res);
            res.negate();
            res /= r(0);
            res -= r(1);
            OP_exp(res);

            break;
        }

        case 5:
        {
            res = diffis;
            OP_sqrt(res);
            res = epow(res,r(1));
            res.negate();
            res /= epow(r(0),r(1));
            res /= r(1);
            res -= r(2);
            OP_exp(res);

            break;
        }

        case 6:
        {
            res = 0.0;

            if ( xx.nindsize() )
            {
                for ( i = 0 ; i < xx.nindsize() ; ++i )
                {
                    xx("&",xx.ind(i)) = epow(xx.direcref(i)/r(0),r(1));
                    yy("&",yy.ind(i)) = epow(yy.direcref(i)/r(0),r(1));
                    xx("&",xx.ind(i)) -= yy(yy.ind(i));
                    res += exp(-(r(4)*epow(xx.direcref(i),r(2))));
                }

                res = epow(res,r(3));
            }

            break;
        }

        case 7:
        {
            res = xyprod;
            res /= r(0);
            res /= r(0);
            res += r(1);
            OP_tanh(res);

            break;
        }

        case 8:
        {
            res  = diffis;
            res /= 2.0;
            res /= r(0);
            res /= r(0);
            res /= r(1);
            res += 1.0;
            res = epow(res,-r(1));

            break;
        }

        case 9:
        {
            res = diffis;
            res /= r(0);
            res /= r(0);
            res += r(1)*r(1);
            OP_sqrt(res);

            break;
        }

        case 10:
        {
            res = diffis;
            res /= r(0);
            res /= r(0);
            res += r(1)*r(1);
            OP_sqrt(res);
            OP_einv(res);

            break;
        }

        case 11:
        {
            res = 0.0_gent; //zerogentype();

            gentype tempres(diffis);
            OP_sqrt(tempres);
            tempres /= r(0);
            tempres.negate();

            if ( (double) abs2(tempres) < 1.0 )
            {
                res = tempres;
                res *= tempres;
                res.negate();
                res += 1.0;
                OP_sqrt(res);
                res *= tempres;
                res += acos(tempres);
                res *= NUMBASE_2ONPI;
            }

            break;
        }

        case 12:
        {
            res = 0.0_gent; //zerogentype();

            gentype tempres(diffis);
            OP_sqrt(tempres);

            {
                res = tempres;
                res *= tempres;
                res *= tempres;
                res /= 2.0;
                res /= r(0);
                res /= r(0);
                res /= r(0);
                res *= 0.6666666666666666666666;
                res *= r(0);
                res -= tempres;
                res /= 0.6666666666666666666666;
                res /= r(0);
                res += 1.0;
            }

            break;
        }

        case 13:
        {
            res = diffis;
            OP_sqrt(res);
            res /= r(0);
            OP_sinc(res);

            break;
        }

        case 14:
        {
            res = diffis;
            res /= r(0);
            res /= r(0);
            OP_sqrt(res);
            res  = epow(res,r(1));
            res *= -1.0;
            break;
         }

        case 15:
        {
            res = diffis;
            res /= r(0);
            res /= r(0);
            OP_sqrt(res);
            res  = epow(res,r(1));
            res += 1.0;
            OP_log(res);
            res.negate();

            break;
        }

        case 16:
        {
            gentype xymin;
            gentype tempres;
            gentype tempras;

            res = 1.0;

            if ( xx.nindsize() )
            {
                for ( i = 0 ; i < xx.nindsize() ; ++i )
                {
                    xymin = ( real(xx.direcref(i)) < real(yy.direcref(i)) ) ? xx.direcref(i) : yy.direcref(i);
                    xymin /= r(0);

                    tempres = 1.0;
                    tempres += tempres;

                    tempras = xx.direcref(i);
                    tempras /= r(0);
                    tempras *= yy.direcref(i);
                    tempras /= r(0);
                    tempras *= xymin;

                    tempres += tempras;

                    tempras *= xymin;
                    tempras /= 2.0;

                    tempres += tempras;

                    tempras = xymin;
                    tempras *= xymin;
                    tempras *= xymin;
                    tempras /= 3.0;

                    tempres += tempras;

                    res *= tempres;
                }
            }

            break;
        }

        case 17:
        {
            res = 1.0;

            if ( xx.nindsize() )
            {
                gentype temp;

                for ( i = 0 ; i < xx.nindsize() ; ++i )
                {
                    Bn(temp,(2*ic(0))+1,(xx.direcref(i)/r(0))-(yy.direcref(i)/r(0)));

                    res *= temp;
                }
            }

            break;
        }

        case 18:
        {
            int i0 = ic(0);
            double r0 = (double) r(0);
            double r1 = (double) r(1);
            double d = (double) diffis;

            res.force_double() = numbase_jn(i0+1,(r1*sqrt((double) d)/r0)/(pow(sqrt(d)/r0,-i0*(r1+1))));

            break;
        }

        case 19:
        {
            res = diffis;
            res /= r(0);
            res /= r(0);
            res += 1.0;
            OP_einv(res);

            break;
        }

        case 20:
        {
            gentype tempres;

            res = 1.0;

            if ( xx.nindsize() )
            {
                for ( i = 0 ; i < xx.nindsize() ; ++i )
                {
                    tempres = xx.direcref(i);
                    tempres *= yy.direcref(i);
                    tempres /= xx.direcref(i)+yy.direcref(i);
                    tempres *= 2.0;
                    tempres /= r(0);

                    res -= tempres;
                }
            }

            break;
        }

        case 21:
        {
            gentype xymin;

            res = 0.0;

            if ( xx.nindsize() )
            {
                for ( i = 0 ; i < xx.nindsize() ; ++i )
                {
                    xymin = ( real(xx.direcref(i)) < real(yy.direcref(i)) ) ? xx.direcref(i) : yy.direcref(i); 
                    xymin /= r(0);

                    res += xymin;
                }
            }

            break;
        }

        case 22:
        {
            gentype xymin;
            gentype xxx,yyy;

            res = 0.0;

            if ( xx.nindsize() )
            {
                for ( i = 0 ; i < xx.nindsize() ; ++i )
                {
                    xxx = eabs2(xx.direcref(i));
                    yyy = eabs2(yy.direcref(i));

                    xxx /= r(0);
                    yyy /= r(0);

                    xxx = epow(xxx,r(1));
                    yyy = epow(yyy,r(2));

                    xymin = ( xxx < yyy ) ? xxx : yyy;

                    res += xymin;
                }
            }

            break;
        }

        case 23:
        {
            res = diffis;
            OP_sqrt(res);
            res /= r(0);
            res = epow(res,r(1));
            res += 1.0;
            OP_einv(res);

            break;
        }

        case 24:
        {
            gentype tmp = xyprod;

            tmp /= r(0);
            tmp /= r(0);

            res = tmp;
            raiseto(res,ic(0));
            res -= 1.0;
            res /= (tmp-1.0);

            break;
        }

        case 25:
        {
            res = diffis;
            OP_sqrt(res);
            res /= -r(0);
            res += NUMBASE_PI;
            OP_cosh(res);
            res *= NUMBASE_PI;

            break;
        }

        case 26:
        {
            res = diffis;
            res /= r(0);
            res = epow(res,r(1)+0.5);

            break;
         }

        case 27:
        {
            gentype tempres;

            res = diffis;
            res /= r(0);

            tempres = res;
            OP_sqrt(tempres);
            OP_log(tempres);

            res = epow(res,r(1));
            res *= tempres;

            break;
         }

        case 28:
        {
            SparseVector<SparseVector<gentype> > t;

            t("&",0)("&",0)  = m;
            t("&",0)("&",1)  = txyprod;
            t("&",0)("&",2)  = tyxprod;
            t("&",0)("&",3)  = diffis;
            t("&",0)("&",4)  = *(xnorm(0));
            t("&",0)("&",5)  = *(xnorm(1));
            t("&",1)("&",0)  = r(0);
            t("&",1)("&",1)  = r(1);
            t("&",1)("&",2)  = r(2);
            t("&",1)("&",3)  = r(3);
            t("&",1)("&",4)  = r(4);
            t("&",1)("&",5)  = r(5);
            t("&",1)("&",6)  = r(6);
            t("&",1)("&",7)  = r(7);
            t("&",1)("&",8)  = r(8);
            t("&",1)("&",9)  = r(9);
            t("&",1)("&",10) = r(10);
            t("&",3) = xx;
            t("&",4) = yy;

            SparseVector<SparseVector<int> > vused(r(10).varsUsed());

            if ( vused(2).nindsize() )
            {
                for ( i = 0 ; i < vused(2).nindsize() ; ++i )
                {
                    j = vused(2).ind(i);

                    Kbase(t("&",2)("&",j),q,j,txyprod,tyxprod,diffis,x,xinfo,xnorm,iiii,xdim,m,0,0,0,mlid);
                }
            }

            res = r(10)(t);

            break;
        }

        case 29:
        {
            double xrnorm = sqrt((double) abs2(*(xnorm(0))));
            double yrnorm = sqrt((double) abs2(*(xnorm(0))));

            double tmp = pow(xrnorm*yrnorm,ic(0))/NUMBASE_PI;

            if ( ( xrnorm < BADZEROTOL ) || ( yrnorm < BADZEROTOL ) )
            {
                //Jn(res,ic(0),acosonedblgentype());
                Jn(res,ic(0),acos(1.0_gent));
                res *= tmp;
            }

            else
            {
                gentype tempres;

                tempres = xyprod;
                tempres /= (xrnorm*yrnorm);
                OP_acos(tempres);

                Jn(res,ic(0),tempres);
                res *= tmp;
            }

            res *= r(0);
            res *= r(0);

            break;
        }

        case 30:
        {
            int i0 = ic(0);

            if ( i0 < 0 )
            {
                NiceThrow("Chaotic logistic kernel assumes i0 >= 0.");
            }

            gentype temp;

            if ( i0 )
            {
                temp =  1.0;
                temp += r(1);
                temp += r(1);
                OP_einv(temp);

                if ( xx.nindsize() )
                {
                    for ( i = 0 ; i < xx.nindsize() ; ++i )
                    {
                        xx.direref(i) /= r(0);
                        xx.direref(i) += r(2);
                        xx.direref(i) *= temp;
                        yy.direref(i) /= r(0);
                        yy.direref(i) += r(2);
                        yy.direref(i) *= temp;
                    }
                }
            }

            while ( i0 > 0 )
            {
                if ( xx.nindsize() )
                {
                    for ( i = 0 ; i < xx.nindsize() ; ++i )
                    {
                        temp = 2.0;
                        temp -= xx.direcref(i);
                        xx.direref(i) *= temp;
                        xx.direref(i) *= r(1);

                        temp = 2.0;
                        temp -= yy.direcref(i);
                        yy.direref(i) *= temp;
                        yy.direref(i) *= r(1);
                    }
                }

                --i0;
            }

            getTwoProd(res,xx,yy,0,1,0,0,0);
            break;
        }

        case 31:
        {
            int i0 = ic(0);

            if ( i0 < 0 )
            {
                NiceThrow("Summed Chaotic logistic kernel assumes i0 >= 0.");
            }

            gentype temp;
            getTwoProd(res,xx,yy,0,1,0,0,0);

            if ( i0 )
            {
                temp = 1.0;
                temp += r(1);
                temp += r(1);
                OP_einv(temp);

                if ( xx.nindsize() )
                {
                    for ( i = 0 ; i < xx.nindsize() ; ++i )
                    {
                        xx.direref(i) /= r(0);
                        xx.direref(i) += r(2);
                        xx.direref(i) *= temp;
                        yy.direref(i) /= r(0);
                        yy.direref(i) += r(2);
                        yy.direref(i) *= temp;
                    }
                }
            }

            while ( i0 > 0 )
            {
                if ( xx.nindsize() )
                {
                    for ( i = 0 ; i < xx.nindsize() ; ++i )
                    {
                        temp = 2.0;
                        temp -= xx.direcref(i);
                        xx.direref(i) *= temp;
                        xx.direref(i) *= r(1);

                        temp = 2.0;
                        temp -= yy.direcref(i);
                        yy.direref(i) *= temp;
                        yy.direref(i) *= r(1);
                    }
                }

                getTwoProd(temp,xx,yy,0,1,0,0,0);
                res += temp;
                --i0;
            }

            break;
        }

        case 32:
        {
            res = ( ( iiii(0) == iiii(1) ) && ( iiii(0) >= 0 ) ) ? r(1) : 0.0_gent; //zerogentype();
            break;
        }

        case 33:
        {
            res = ( real(sqrt(diffis)) < real(r(0)) ) ? 0.5/r(0) : 0.0_gent; //zerogentype();
            break;
        }

        case 34:
        {
            res = diffis;
            OP_sqrt(res);
            res = ( real(res) < real(r(0)) ) ? (1.0-(sqrt(res)/r(0)))/r(0) : 0.0_gent; //zerogentype();
            break;
        }

        case 35:
        {
            // ((2^(1-r1))/gamma(r1)).((sqrt(2.r1).sqrt(d)/r0)^r1).K_r1(sqrt(2.r1).sqrt(d)/r0)
            // (2^(1-r1)).(dd^r1).K_r1(dd)/gamma(r1)
            // 2.((dd/2)^r1).K_r1(dd)/gamma(r1)

            int i0 = (int) ic(0);
            double r0 = (double) r(0);
            double dd = sqrt(2*i0*((double) diffis))/r0;
            double qq = 0;
            double gg = 0;

            qq = numbase_kn(i0,dd);
            gg = numbase_gamma(i0);

            res = ( dd < 1e-6 ) ? 1.0 : 2*pow(dd/2,i0)*qq/gg;

            break;
        }

        case 36:
        {
            res = 1.0;

            if ( xx.nindsize() )
            {
                for ( i = 0 ; i < xx.nindsize() ; ++i )
                {
                    res *= ( real(xx.direcref(i)/r(0)) < real(yy.direcref(i)/r(0)) ) ? xx.direcref(i)/r(0) : yy.direcref(i)/r(0);
                }
            }

            break;
        }

        case 37:
        {
            gentype tempq;
            gentype tempr;
            gentype tempa;
            gentype tempb;

            int p = ic(0);
            double nu = p+0.5;

            NiceAssert( p >= 0 );

            tempq  = diffis;
            tempq *= nu;
            tempq *= 8.0;
            OP_sqrt(tempq);
            tempq /= r(0);

            tempr  = 1.0; // this will be multiplied by tempq after each
                          // iteration so will equal tempq^(p-i) at the
                          // start of each iteration.
            tempb  = 0.0;

            for ( i = p ; i >= 0 ; --i )
            {
                tempa  = tempr; // = tempq^(p-i)
                tempa *= (((double) xnfact(p+i))/(((double) xnfact(p-i))*((double) xnfact(i))));

                tempb += tempa;
                tempr *= tempq;
            }

            double tempc;
            double tempd;

            tempc = numbase_gamma(p+1);
            tempd = numbase_gamma((2*p)+1);

            tempb *= (tempc/tempd);

            res  = tempq; // No need to re-calculate this
            res /= -2.0;
            OP_exp(res);

            res *= tempb;

            break;
        }

        case 38:
        case 1038:
        {
            res  = diffis;
            OP_sqrt(res);
            res /= r(0);
            res.negate();
            OP_exp(res);

            if ( typeis == 1038 )
            {
                for ( i = 0 ; i < yy.nindsize() ; ++i )
                {
                    yy.direref(i) -= xx.direcref(i);

                    OP_sgn(yy.direref(i));
                    res *= yy.direcref(i);
                    res /= r(0);
                    res.negate();
                }
            }

            break;
        }

        case 2038:
        {
            NiceAssert( xdim == 1 );

            double xxx = (double) xx.direcref(0);
            double yyy = (double) yy.direcref(0);
            double r0 = (double) r(0);

            if ( xxx < yyy )
            {
                res = (r0*exp(-(yyy-xxx)/r0));
            }

            else
            {
                res = (r0*(2-exp(-(xxx-yyy)/r0)));
            }

            break;
        }

        case 39:
        {
            gentype tempa;
            gentype tempb;

            tempb = 1.0;

            tempa  = diffis;
            tempa *= 3.0;
            OP_sqrt(tempa);
            tempa /= r(0);

            tempb += tempa;

            res = tempa; // No need to repeat this
            res.negate();
            OP_exp(res);

            res *= tempb;

            break;
        }

        case 40:
        {
            gentype tempa;
            gentype tempb;

            tempb = 1.0;

            tempa  = diffis;
            tempa *= 5.0;
            OP_sqrt(tempa);
            tempa /= r(0);

            tempb += tempa;

            res = tempa; // do now to save recalculation
            res.negate();
            OP_exp(res);

            tempa *= tempa;
            tempa /= 3.0;

            tempb += tempa;

            res *= tempb;

            break;
        }

        case 41:
        {
            res = xyprod;
            //raiseto(res,halfdblgentype()/(r(0)*r(0)));
            raiseto(res,0.5_gent/(r(0)*r(0)));

            break;
        }

        case 42:
        {
            res = xyprod;
            res /= r(0);
            res /= r(0);
            OP_agd(res);

            break;
        }

        case 43:
        {
            gentype tempa(xyprod);

            tempa  = xyprod;
            tempa /= r(0);
            tempa /= r(0);

            gentype tempb(tempa);

            tempa += 1.0;
            tempb -= 1.0;

            res  = tempa;
            res /= tempb;

            OP_log(res);

            break;
        }

        case 44:
        {
            res = xyprod;
            res /= r(0);
            res /= r(0);
            res -= r(1);
            OP_exp(res);

            break;
        }

        case 45:
        {
            res = xyprod;
            res /= r(0);
            res /= r(0);
            OP_sinh(res);

            break;
        }

        case 46:
        {
            res = xyprod;
            res /= r(0);
            res /= r(0);
            OP_cosh(res);

            break;
        }

        case 47:
        {
            gentype dd(diffis);

            OP_sqrt(dd);

            res = dd;
            res /= r(0);
            OP_sinc(res);

            gentype tmpa(dd);

            tmpa *= (2*NUMBASE_PI);
            tmpa /= r(1);
            OP_cos(tmpa);

            res *= tmpa;

            break;
        }

        case 48:
        {
            NiceAssert( xx(0).isCastableToIntegerWithoutLoss() );
            NiceAssert( yy(0).isCastableToIntegerWithoutLoss() );

            int ix = (int) xx(0);
            int iy = (int) yy(0);

            const gentype &r1 = r(1);

            if ( r1.isValMatrix() )
            {
                res = r1(ix,iy);
            }

            else if ( ix == iy )
            {
                res = 1.0;
            }

            else
            {
                res = r1;
            }

            break;
        }

        case 49:
        {
            gentype tmp(diffis);

            tmp /= -2.0;
            tmp /= r(0);
            tmp /= r(0);
            tmp -= r(1);
            OP_exp(tmp);
            tmp *= AltDiffNormConst(xdim,m,r(0));
            tmp *= r(2);
            tmp -= 1.0;

            res  = r(2);
            res -= 1.0;
            res /= tmp;

            break;
        }

        case 50:
        {
            res = xyprod;
            res /= r(0);
            res /= r(0);
            OP_acos(res);
            res *= -1.0;
            res += NUMBASE_PI;

            break;
        }

        case 51:
        {
            res = xyprod;
            res /= r(0);
            res /= r(0);
            res *= -1.0;
            res += 2.0;
            OP_einv(res);

            break;
        }

        case 52:
        {
            // K = sqrt(a).sqrt(b)/(r0.r0)

            res = 1.0;

            for ( int j = 0 ; j < m ; ++j )
            {
                res *= *(xnorm(j));
            }

            gentype oneonm(1.0/m);
            res = pow(res,oneonm);
            res /= r(0);
            res /= r(0);

            break;
        }

        case 53:
        {
            // K = (1-(1-sqrt(a)^r1)^r2).(1-(1-sqrt(b)^r1)^r2)/(r0.r0)

            gentype oneonm(1.0/m);

            res = 1.0;

            for ( int j = 0 ; j < m ; ++j )
            {
                gentype tempres(*(xnorm(j)));

                tempres = pow(tempres,oneonm);

                tempres = pow(tempres,r(1));
                tempres *= -1.0;
                tempres += 1.0;
                tempres = pow(tempres,r(2));
                tempres *= -1.0;
                tempres += 1.0;

                res *= tempres;
            }

            res /= r(0);
            res /= r(0);

            break;
        }

        case 100:
        {
            res = xyprod;
            res /= r(0);
            res /= r(0);
            break;
        }

        case 101:
        {
            res = xyprod;
            res /= r(0);
            res /= r(0);
            res.negate();
            OP_exp(res);
            res += 1.0;
            OP_einv(res);

            break;
        }

        case 102:
        {
            res = xyprod;
            res -= r(3);
            res /= r(0);
            res /= r(0);
            res *= r(2);
            res.negate();
            OP_exp(res);
            res *= r(1);
            res += 1.0;
            res = epow(res,inv(r(2)));
            OP_einv(res);

            break;
        }

        case 103:
        {
            //res = ( real(xyprod) > zerogentype() ) ? 1.0 : 0.0;
            res = ( real(xyprod) > 0.0_gent ) ? 1.0 : 0.0;
            break;
        }

        case 104:
        {
            //res = ( real(xyprod) > zerogentype() ) ? xyprod : zerogentype();
            res = ( real(xyprod) > 0.0_gent ) ? xyprod : 0.0_gent; //zerogentype();
            res /= r(0);
            res /= r(0);
            break;
        }

        case 105:
        {
            res = xyprod;
            res /= r(0);
            res /= r(0);
            OP_exp(res);
            res += r(1);
            OP_log(res);

            break;
        }

        case 106:
        {
            //if ( real(xyprod) > zerogentype() )
            if ( real(xyprod) > 0.0_gent )
            {
                //res = ( real(xyprod) > zerogentype() ) ? xyprod : zerogentype();
                res = ( real(xyprod) > 0.0_gent ) ? xyprod : 0.0_gent;
                res /= r(0);
                res /= r(0);
            }

            else
            {
                //res = ( real(xyprod) > zerogentype() ) ? xyprod : zerogentype();
                res = ( real(xyprod) > 0.0_gent ) ? xyprod : 0.0_gent;
                res /= r(0);
                res /= r(0);
                res *= r(1);
            }

            break;
        }

        case 200:
        {
            res = xyprod;
            res /= r(0);
            res /= r(0);
            res -= 1.0;

            break;
        }

        case 201:
        {
            res = xyprod;
            res /= r(0);
            res /= r(0);
            res.negate();
            OP_exp(res);
            res += 1.0;
            OP_einv(res);
            res *= 2.0;
            res -= 1.0;

            break;
        }

        case 202:
        {
            res = xyprod;
            res -= r(3);
            res /= r(0);
            res /= r(0);
            res *= r(2);
            res.negate();
            OP_exp(res);
            res *= r(1);
            res += 1.0;
            res = epow(res,inv(r(2)));
            OP_einv(res);
            res *= 2.0;
            res -= 1.0;

            break;
        }

        case 203:
        {
            //res = ( real(xyprod) > zerogentype() ) ? onedblgentype() : negonedblgentype();
            res = ( real(xyprod) > 0.0_gent ) ? 1.0_gent : -1.0_gent;
            break;
        }

        case 204:
        {
            //res = ( real(xyprod) > zerogentype() ) ? xyprod-onedblgentype() : negonedblgentype();
            res = ( real(xyprod) > 0.0_gent ) ? xyprod-1.0_gent : -1.0_gent;
            res /= r(0);
            res /= r(0);
            break;
        }

        case 205:
        {
            res = xyprod;
            res /= r(0);
            res /= r(0);
            OP_exp(res);
            res += r(1);
            OP_log(res);
            res *= 2.0;
            res -= 1.0;

            break;
        }

        case 206:
        {
            //if ( real(xyprod) > zerogentype() )
            if ( real(xyprod) > 0.0_gent )
            {
                //res = xyprod-onedblgentype();
                res = xyprod-1.0_gent;
                res /= r(0);
                res /= r(0);
            }

            else
            {
                //res = (r(1)*xyprod)-onedblgentype();
                res = (r(1)*xyprod)-1.0_gent;
                res /= r(0);
                res /= r(0);
            }

            break;
        }

        case 300:
        {
            res = diffis;
            res *= -0.5;
            res /= r(0);
            res /= r(0);

            break;
        }

        case 301:
        {
            xx -= yy;
            res = abs1(xx);
            res *= res;
            res *= -0.5;
            res /= r(0);
            res /= r(0);

            break;
        }

        case 302:
        {
            xx -= yy;
            res = absinf(xx);
            res *= res;
            res *= -0.5;
            res /= r(0);
            res /= r(0);

            break;
        }

        case 303:
        {
            xx -= yy;
            res = abs0(xx);
            res *= res;
            res *= -0.5;
            res /= r(0);
            res /= r(0);

            break;
        }

        case 304:
        {
            xx -= yy;
            res = absp(xx,(double) r(1));
            res *= res;
            res *= -0.5;
            res /= r(0);
            res /= r(0);

            break;
        }

        case 400:
        case 450:
        case 600:
        case 650:
        {
            res = 1.0;

            if ( yy.nindsize() )
            {
                for ( i = 0 ; i < yy.nindsize() ; ++i )
                {
                    yy.direref(i) -= xx.direcref(i);

                    if ( (typeis/10)%10 == 5 )
                    {
                        yy.direref(i).negate();
                    }

                    yy.direref(i) /= r(0);
                    OP_exp(yy.direref(i));
                    yy.direref(i) += 1.0;
                    OP_einv(yy.direref(i));
                    yy.direref(i) *= 2.0;
                    yy.direref(i) -= r(1);
                    res *= yy.direcref(i);
                }
            }

            if ( typeis/100 == 4 )
            {
                res += 1.0;
                res *= 0.5;
            }

            break;
        }

        case 401:
        case 451:
        case 601:
        case 651:
        {
            res = 1.0;

            if ( yy.nindsize() )
            {
                for ( i = 0 ; i < yy.nindsize() ; ++i )
                {
                    yy.direref(i) -= xx.direcref(i);

                    if ( (typeis/10)%10 == 5 )
                    {
                        yy.direref(i).negate();
                    }

                    yy.direref(i) /= r(0);
                    yy.direref(i).negate();
                    yy.direref(i) = erf(yy.direcref(i));
                    yy.direref(i) -= r(1);
                    res *= yy.direcref(i);
                }
            }

            if ( typeis/100 == 4 )
            {
                res += 1.0;
                res *= 0.5;
            }

            break;
        }

        case 402:
        case 452:
        case 602:
        case 652:
        {
            res = 1.0;

            if ( xx.nindsize() )
            {
                double temp = 0;
                double minval = 0;

                for ( i = 0 ; i < xx.nindsize() ; ++i )
                {
                    temp  = (double) xx.direcref(i);
                    temp -= (double) yy.direcref(i);

                    if ( (typeis/10)%10 == 5 )
                    {
                        temp = -temp;
                    }

                    if ( !i || ( temp < minval ) )
                    {
                        minval = temp;
                    }
                }

                res = minval;
                res /= r(0);
                res.negate();
                OP_exp(res);
                res += 1.0;
                OP_einv(res);
                res *= 2.0;
                res -= r(1);
            }

            if ( typeis/100 == 4 )
            {
                res += 1.0;
                res *= 0.5;
            }

            break;
        }

        case 403:
        case 453:
        case 603:
        case 653:
        {
            res = 1.0;

            if ( xx.nindsize() )
            {
                double temp = 0;
                double minval = 0;

                for ( i = 0 ; i < xx.nindsize() ; ++i )
                {
                    temp  = (double) xx.direcref(i);
                    temp -= (double) yy.direcref(i);

                    if ( (typeis/10)%10 == 5 )
                    {
                        temp = -temp;
                    }

                    if ( !i || ( temp < minval ) )
                    {
                        minval = temp;
                    }
                }

                res = ( minval <= 1e-6 ) ? -1.0 : 1.0;
            }

            if ( typeis/100 == 4 )
            {
                res += 1.0;
                res *= 0.5;
            }

            break;
        }

        case 404:
        case 454:
        case 604:
        case 654:
        {
            res = 1.0;

            if ( xx.nindsize() )
            {
                double temp = 0;
                double maxval = 0;

                for ( i = 0 ; i < xx.nindsize() ; ++i )
                {
                    temp  = (double) xx.direcref(i);
                    temp -= (double) yy.direcref(i);

                    if ( (typeis/10)%10 == 5 )
                    {
                        temp = -temp;
                    }

                    if ( !i || ( temp > maxval ) )
                    {
                        maxval = temp;
                    }
                }

                res = maxval;
                res /= r(0);
            }

            if ( typeis/100 == 4 )
            {
                res += 1.0;
                res *= 0.5;
            }

            break;
        }


        case 500:
        case 550:
        case 700:
        case 750:
        {
            res = 1.0;

            if ( xx.nindsize() )
            {
                for ( i = 0 ; i < xx.nindsize() ; ++i )
                {
                    xx.direref(i) -= yy.direcref(i);
                    xx.direref(i) /= r(0);

                    if ( (typeis/10)%10 == 5 )
                    {
                        yy.direref(i).negate();
                    }

                    xx.direref(i).negate();
                    OP_exp(xx.direref(i));
                    res *= xx.direcref(i);
                    res /= r(0);
                    res *= 2.0;
                    xx.direref(i) += 1.0;
                    OP_einv(xx.direref(i));
                    res *= xx.direcref(i);
                    res *= xx.direcref(i);

                    if ( (typeis/10)%10 == 5 )
                    {
                        res.negate(); // do this on each axis
                    }
                }
            }

            if ( typeis/100 == 5 )
            {
                res *= 0.5;
            }

            break;
        }

        case 501:
        case 551:
        case 701:
        case 751:
        {
            gentype scalefact(2.0/NUMBASE_SQRTPI);

            res = 1.0;

            if ( xx.nindsize() )
            {
                for ( i = 0 ; i < xx.nindsize() ; ++i )
                {
                    xx.direref(i) -= yy.direcref(i);
                    xx.direref(i) /= r(0);

                    if ( (typeis/10)%10 == 5 )
                    {
                        xx.direref(i).negate();
                    }

                    xx.direref(i) *= xx.direcref(i);
                    xx.direref(i).negate();
                    OP_exp(xx.direref(i));
                    xx.direref(i) /= r(0);
                    xx.direref(i) *= scalefact;
                    res *= yy.direcref(i);

                    if ( (typeis/10)%10 == 5 )
                    {
                        res.negate(); // do this on each axis
                    }
                }
            }

            if ( typeis/100 == 5 )
            {
                res *= 0.5;
            }

            break;
        }

        case 502:
        case 552:
        case 702:
        case 752:
        {
            res = 0.0;

            if ( xx.nindsize() )
            {
                double temp = 0;
                double minval = 0;

                for ( i = 0 ; i < yy.nindsize() ; ++i )
                {
                    temp  = (double) xx.direcref(i);
                    temp -= (double) yy.direcref(i);

                    if ( (typeis/10)%10 == 5 )
                    {
                        temp = -temp;
                    }

                    if ( !i || ( temp < minval ) )
                    {
                        minval = temp;
                    }
                }

                gentype tempb;

                tempb = minval;
                tempb /= r(0);
                tempb.negate();
                OP_exp(tempb);
                res = tempb;
                tempb += 1.0;
                OP_einv(tempb);
                res *= tempb;
                res *= tempb;
                res /= r(0);
                res *= 2.0;

                if ( (typeis/10)%10 == 5 )
                {
                    res.negate();
                }

                if ( typeis/100 == 5 )
                {
                    res *= 0.5;
                }
            }

            break;
        }

        case 503:
        case 553:
        case 703:
        case 753:
        {
            res = 0.0;

            break;
        }

        case 504:
        case 554:
        case 704:
        case 754:
        {
            res = 1.0;

            if ( (typeis/10)%10 == 5 )
            {
                res.negate();
            }

            if ( typeis/100 == 5 )
            {
                res *= 0.5;
                res /= r(0);
            }

            break;
        }

        default:
	{
	    NiceThrow("Unknown kernel type.\n");

	    break;
	}
    }
  }

  else if ( resmode & 8 )
  {
    // dKdr gradient calculation

    NiceAssert( q >= 0 );
    NiceAssert( q < size() );
    NiceAssert( !adensetype && !bdensetype );

    Vector<gentype> &vres = res.force_vector();

    gentype xyprod;

    xyprod =  txyprod;
    xyprod += tyxprod;
    xyprod /= 2.0;

    vres.resize(dRealConstants(q).size()-1);

    processOverwrites(q,*(x(0)),*(x(1)));

    retVector<gentype> tmpva;

    const Vector<gentype> &r = dRealConstants(q)(1,1,dRealConstants(q).size()-1,tmpva);
    const Vector<int> &ic = dIntConstants(q);

//ADDHERE - kernel derivative wrt parameter vector goes here
    if ( dRealConstants(q).size()-1 )
    {
      // Use -1 pseudotype to force fallthrough to manual calculation

      switch ( !resmode ? typeis : -1 )
      {
        case 0:
        {
            vres("&",0) = 0.0;
            vres("&",1) = 1.0;

            break;
        }

        case 1:
        {
            vres("&",0) = -2.0;
            vres("&",0) /= r(0);
            vres("&",0) /= r(0);
            vres("&",0) /= r(0);

            break;
        }

        case 2:
	{
            vres("&",1) = xyprod;
            vres("&",1) /= r(0);
            vres("&",1) /= r(0);
            vres("&",1) += r(1);
            raiseto(vres("&",1),ic(0)-1);
            vres("&",1) *= ic(0);

            vres("&",0) = vres(1);
            vres("&",0) *= -2.0;
            vres("&",0) *= xyprod;
            vres("&",0) /= r(0);
            vres("&",0) /= r(0);
            vres("&",0) /= r(0);

            break;
	}

        case 3:
	{
            vres("&",0) = diffis;
            vres("&",0) /= -2.0;
            vres("&",0) /= r(0);
            vres("&",0) /= r(0);
            vres("&",0) -= r(1);
            OP_exp(vres("&",0));
            vres("&",0) *= diffis;
            vres("&",0) /= r(0);
            vres("&",0) /= r(0);
            vres("&",0) /= r(0);

            vres("&",1) = diffis;
            vres("&",1) /= -2.0;
            vres("&",1) /= r(0);
            vres("&",1) /= r(0);
            vres("&",1) -= r(1);
            OP_exp(vres("&",1));

            break;
	}

        case 4:
	{
            gentype temp(diffis);

            OP_sqrt(temp);

            vres("&",0) = temp;
            vres("&",0).negate();
            vres("&",0) /= r(0);
            vres("&",0) -= r(1);
            OP_exp(vres("&",0));
            vres("&",0) *= temp;
            vres("&",0) /= r(0);
            vres("&",0) /= r(0);

            vres("&",1) = temp;
            vres("&",1).negate();
            vres("&",1) /= r(0);
            vres("&",1) -= r(1);
            OP_exp(vres("&",1));

            break;
	}

        case 5:
	{
            gentype tempa(diffis);
            OP_sqrt(tempa);
            tempa /= r(0);

            vres("&",0) = tempa;
            OP_log(vres("&",0));
            vres("&",0).negate();
            vres("&",0) += inv(r(1));
            vres("&",0) /= r(1);

            vres("&",1) = r(0);
            vres("&",1).inverse();

            tempa = epow(tempa,r(1));

            vres("&",0) *= tempa;
            vres("&",1) *= tempa;

            tempa /= r(1);
            tempa.negate();
            OP_exp(tempa);

            vres("&",0) *= tempa;
            vres("&",1) *= tempa;

// fuck it, who cares about vres(0,2)

            break;
	}

        case 7:
	{
            vres("&",1) = xyprod;
            vres("&",1) /= r(0);
            vres("&",1) /= r(0);
            vres("&",1) += r(1);
            OP_sech(vres("&",1));
            vres("&",1) *= vres(1);

            vres("&",0) = vres(1);
            vres("&",0) *= xyprod;
            vres("&",0) *= -2.0;
            vres("&",0) /= r(0);
            vres("&",0) /= r(0);
            vres("&",0) /= r(0);

            break;
	}

        case 8:
	{
            vres("&",0)  = diffis;
            vres("&",0) /= 2.0;
            vres("&",0) /= r(0);
            vres("&",0) /= r(0);
            vres("&",0) /= r(1);
            vres("&",0) += 1.0;
            vres("&",0) = epow(res,-r(1));

            break;
	}

        case 9:
	{
            vres("&",0) = diffis;
            vres("&",0) /= r(0);
            vres("&",0) /= r(0);
            vres("&",0) += r(1)*r(1);
            OP_sqrt(vres("&",0));
            OP_einv(vres("&",0));
            vres("&",0) *= -2.0;
            vres("&",0) *= diffis;
            vres("&",0) /= r(0);
            vres("&",0) /= r(0);
            vres("&",0) /= r(0);

            vres("&",1) = diffis;
            vres("&",1) /= r(0);
            vres("&",1) /= r(0);
            vres("&",1) += r(1)*r(1);
            OP_sqrt(vres("&",1));
            OP_einv(vres("&",1));
            vres("&",1) *= r(1);

            break;
	}

        case 10:
	{
            vres("&",0) = diffis;
            vres("&",0) /= r(0);
            vres("&",0) /= r(0);
            vres("&",0) += r(1)*r(1);
            OP_sqrt(vres("&",0));
            raiseto(vres("&",0),3);
            OP_einv(vres("&",0));
            vres("&",0) *= -2.0;
            vres("&",0) *= diffis;
            vres("&",0) /= r(0);
            vres("&",0) /= r(0);
            vres("&",0) /= r(0);

            vres("&",1) = diffis;
            vres("&",1) /= r(0);
            vres("&",1) /= r(0);
            vres("&",1) += r(1)*r(1);
            OP_sqrt(vres("&",1));
            raiseto(vres("&",1),3);
            OP_einv(vres("&",1));
            vres("&",1) *= r(1);

            break;
	}

        case 11:
	{
            gentype tempa(diffis);

            OP_sqrt(tempa);
            tempa /= r(0);

            gentype tempb(tempa);

            tempb.negate();
            tempb *= tempa;
            tempb += 1.0;
            OP_sqrt(tempb);

            vres("&",0) = tempa;
            vres("&",0) *= tempa;
            vres("&",0) *= tempa;
            vres("&",0) /= tempb;
            vres("&",0) /= r(0);
            vres("&",0) *= (4/NUMBASE_PI);

            break;
	}

        case 12:
	{
            gentype tempa(diffis);

            OP_sqrt(tempa);
            tempa /= r(0);

            vres("&",0) = tempa;
            vres("&",0) *= tempa;
            vres("&",0) += 1.0;
            vres("&",0) *= tempa;
            vres("&",0) /= r(0);
            vres("&",0) *= -1.5;

            break;
	}

        case 13:
	{
            gentype tempa(diffis);

            OP_sqrt(tempa);
            tempa /= r(0);

            gentype tempb(tempa);

            OP_cos(tempa);
            OP_sinc(tempb);

            vres("&",0) = tempa;
            vres("&",0) -= tempb;
            vres("&",0) /= r(0);
            vres("&",0).negate();

            break;
	}

        case 18:
        {
            NiceThrow("Bessel kernel not implemented yet");
            break;
        }

        case 32:
	{
            vres("&",0) = 0.0;
            vres("&",1) = ( ( iiii(0) == iiii(1) ) && ( iiii(0) >= 0 ) ) ? 1.0 : 0.0;

            break;
	}

        case 35:
        {
            NiceThrow("Matern kernel not implemented yet");
            break;
        }

        case 103:
        {
            vres("&",0) = 0.0;
            break;
        }

        case 203:
        {
            vres("&",0) = 0.0;
            break;
        }

        default:
	{
            if ( dRealConstants(q).size() <= 1 )
            {
                break;
            }

            // Revert to symbolic mathematics approach
            //
            // 1. call Kbase in remode 2 to get K(x,y) as equation in terms of
            //    var(0,0) = x'x, var(0,1) = y'y, var(0,2) = x'y, var(1,i) = ri
            // 2. substitute var(0,0), var(0,1), var(0,2)
            // 3. take derivative wrt var(1,i)
            // 4. substitute var(1,i)
            // 5. assert that end result is a number.
    
            gentype restemp;
            SparseVector<SparseVector<gentype> > subarray;

            // Calculate equational form of kernel, making sure not to sub
            // out real constants

            Kbase(restemp,q,typeis,txyprod,tyxprod,diffis,x,xinfo,xnorm,iiii,xdim,m,0,0,resmode|4,mlid);

            vres = restemp;

            // First stage substitutions

            if ( !( resmode & 1 ) )
            {
                subarray("&",0)("&",0) =  *(xnorm(0));
                subarray("&",0)("&",1) =  *(xnorm(1));
                subarray("&",0)("&",2) =  txyprod;
                subarray("&",0)("&",2) += tyxprod;
                subarray("&",0)("&",2) /= 2.0;
            }

            int ij;

            if ( !( resmode & 2 ) && ( dIntConstants(q).size() > 0 ) )
            {
                for ( ij = 0 ; ij < dIntConstants(q).size() ; ++ij )
                {
                    subarray("&",2)("&",ij) = dIntConstants(q)(ij);
                }
            }

            for ( ij = 0 ; ij < dRealConstants(q).size()-1 ; ++ij )
            {
                vres("&",ij).substitute(subarray);
            }

            // Calculate derivative vector

            for ( ij = 0 ; ij < dRealConstants(q).size()-1 ; ++ij )
            {
                vres("&",ij).realDeriv(1,ij);
            }

            // Second stage substitutions

            subarray.zero();

            if ( !( resmode & 4 ) && ( dRealConstants(q).size()-1 > 0 ) )
            {
                for ( ij = 0 ; ij < dRealConstants(q).size()-1 ; ++ij )
                {
                    subarray("&",1)("&",ij) = dRealConstants(q)(ij+1);
                }
            }

            for ( ij = 0 ; ij < dRealConstants(q).size()-1 ; ++ij )
            {
                vres("&",ij).substitute(subarray);
            }

	    break;
	}
      }
    }
  }

  else
  {
    //NiceAssert( !densetype );

//    int KusesVector  = kinf(q).usesVector;
//    int KusesMinDiff = kinf(q).usesMinDiff;
//    int KusesMaxDiff = kinf(q).usesMaxDiff;

    if ( adensetype || bdensetype )
    {
        // Actually want dense derivative

        symm = calcDensePair(typeis,adensetype,bdensetype,xdim);

        NiceAssert( typeis >= 0 );

//        kernInfo altkinf;
//        Vector<gentype> rrr; // dummy vector
//        Vector<int> icic;    // dummy vector

//        fillOutInfo(altkinf,rrr,icic,typeis);

//        KusesVector  = altkinf.usesVector;
//        KusesMinDiff = altkinf.usesMinDiff;
//        KusesMaxDiff = altkinf.usesMaxDiff;
    }

    switch ( typeis )
    {
        case   0: { res = "var(1,1)"; break; }
        case   1: { res = "z/(var(1,0)*var(1,0))"; break; }
        case   2: { res = "(var(1,1)+(z/(var(1,0)*var(1,0))))^var(2,0)"; break; }
        case   3: { res = "exp((-var(0,3)/(2*var(1,0)*var(1,0)))-var(1,1))"; break; }
        case   4: { res = "exp((-sqrt(var(0,3))/var(1,0))-var(1,1))"; break; }
        case   5: { res = "exp((-(var(0,3)^(var(1,1)/2))/(var(1,0)*var(1,1)))-var(1,2))"; break; }
        case   6: { res.makeError("Error: ANOVA kernel cannot be returned as equation"); break; }
        case   7: { res = "tanh(z/(var(1,0)*var(1,0))+var(1,1))"; break; }
        case   8: { res = "(1+(var(0,3)/(2*var(1,0)*var(1,0)*var(1,1))))^(-var(1,1))"; break; }
        case   9: { res = "sqrt((var(0,3)/(var(1,0)*var(1,0)))+var(1,1)^2)"; break; }
        case  10: { res = "1/sqrt((var(0,3)/(var(1,0)*var(1,0)))+var(1,1)^2)"; break; }
        case  11: { res = "ifthenelse(lt(var(0,3),0),(2/pi())*(acos(-sqrt(var(0,3))/var(1,0))-((sqrt(var(0,3))/var(1,0))*sqrt(1-var(0,3)/var(1,0)))),0)"; break; }
        case  12: { res = "1-(1.5*sqrt(var(0,3))/var(1,0))(0.5*(var(0,3)^3)/(var(1,0)^3))"; break; }
        case  13: { res = "sinc(sqrt(var(0,3))/var(1,0))"; break; }
        case  14: { res = "sqrt((var(0,3)/(var(1,0)*var(1,0))))^var(1,1)"; break; }
        case  15: { res = "log((sqrt((var(0,3)/(var(1,0)*var(1,0))))^var(1,1))+1)"; break; }
        case  16: { res.makeError("Error: Spline kernel cannot be returned as equation"); break; }
        case  17: { res.makeError("Error: B-Spline kernel cannot be returned as equation"); break; }
        case  18: { res.makeError("Error: Bessel kernel not yet implemented"); break; }
        case  19: { res = "1/(1+(var(0,3)/(var(1,0)*var(1,0))))"; break; }
        case  20: { res.makeError("Error: Chi-Square kernel cannot be returned as equation"); break; }
        case  21: { res.makeError("Error: Histogram kernel cannot be returned as equation"); break; }
        case  22: { res.makeError("Error: Generalised histogram kernel cannot be returned as equation"); break; }
        case  23: { res = "1/(1+((sqrt(var(0,3))/var(1,0))^var(1,1)))"; break; }
        case  24: { res = "(1-(z/(var(1,0)*var(1,0)))^var(2,0))/(1-(z/(var(1,0)*var(1,0))))"; break; }
        case  25: { res = "pi()*cosh(pi()-sqrt(var(0,3))/var(1,0))"; break; }
        case  26: { res = "(var(0,3)/var(1,0))^(var(1,1)+0.5)"; break; }
        case  27: { res = "((var(0,3)/var(1,0))^(var(1,1)+0.5))*log(sqrt(var(0,3)/var(1,0)))"; break; }
        case  28: { res.makeError("Error: General kernel cannot be returned as equation"); break; }
        case  29: { res.makeError("Error: Arccosine kernel cannot be returned as equation"); break; }
        case  30: { res.makeError("Error: Chaotic logistic kernel cannot be returned as equation"); break; }
        case  31: { res.makeError("Error: Summed Chaotic logistic kernel cannot be returned as equation"); break; }
        case  32: { res.makeError("Error: Diagonal kernel cannot be returned as equation"); break; }
        case  33: { res = "ifthenelse(lt(real(sqrt(x+y-2*z)),var(1,0)),1/(2*var(1,0)),0)"; break; }
        case  34: { res = "ifthenelse(lt(real(sqrt(x+y-2*z)),var(1,0)),(1-real(sqrt(x+y-2*z)/var(1,0))/var(1,0),0)"; break; }
        case  35: { res.makeError("Error: General Matern kernel not yet implemented"); break; }
        case  36: { res.makeError("Error: Weiner kernel cannot be returned as equation"); break; }
        case  37: { res.makeError("Error: Half-Integer Matern kernel cannot be returned as an equation"); break; }
        case  38: { res = "exp(-sqrt(var(0,3))/var(1,0))"; break; }
        case  39: { res = "(1+(sqrt(3*var(0,3))/var(1,0)))*exp(-sqrt(3*var(0,3))/var(1,0))"; break; }
        case  40: { res = "(1+(sqrt(5*var(0,3))/var(1,0))+(5*var(0,3)/(3*var(1,0)*var(1,0))))*exp(-sqrt(5*var(0,3))/var(1,0))"; break; }
        case  41: { res = "z^(1/(2*var(1,0)*var(1,0)))"; break; }
        case  42: { res = "agd(z/(var(1,0)*var(1,0)))"; break; }
        case  43: { res = "log((1+(z/(var(1,0)*var(1,0))))/(1-(z/(var(1,0)*var(1,0)))))"; break; }
        case  44: { res = "exp((z/(var(1,0)*var(1,0)))-var(1,1))"; break; }
        case  45: { res = "sinh(z/(var(1,0)*var(1,0)))"; break; }
        case  46: { res = "cosh(z/(var(1,0)*var(1,0)))"; break; }
        case  47: { res = "sinc(sqrt(var(0,3))/var(1,0))*sinc(2*pi()*sqrt(var(0,3))/(var(1,0)*var(1,1)))"; break; }
        case  48: { res.makeError("Error: LUT kernel cannot be returned as equation"); break; }
        case  49: { res = "(1-var(1,2))/(1-var(1,2)*exp((-var(0,3)/(2*var(1,0)*var(1,0)))-var(1,1)))"; break; }
        case  50: { res = "pi()-acos(z/(var(1,0)*var(1,0)))"; break; }
        case  51: { res = "1/(2-z/(var(1,0)*var(1,0)))"; break; }
        case  52: { res.makeError("Error: Radius kernel cannot be returned as equation"); break; }
        case  53: { res.makeError("Error: Radius kernel cannot be returned as equation"); break; }
        case 100: { res = "z/(var(1,0)*var(1,0))"; break; }
        case 101: { res = "1/(1+exp(-z/(var(1,0)*var(1,0))))"; break; }
        case 102: { res = "1/(1+var(1,1)*exp(-var(1,2)*(z-var(1,3))/(var(1,0)*var(1,0))))^(1/var(1,2))"; break; }
        case 103: { res = "ifthenelse(lt(real(z),0),0,1)"; break; }
        case 104: { res = "ifthenelse(lt(real(z),0),0,z/(var(1,0)*var(1,0)))"; break; }
        case 105: { res = "log(var(1,1)+exp(z/(var(1,0)*var(1,0))))"; break; }
        case 106: { res = "ifthenelse(lt(real(z),0),(var(1,1)*z)/(var(1,0)*var(1,0)),z/(var(1,0)*var(1,0)))"; break; }
        case 200: { res = "(z/(var(1,0)*var(1,0)))-1"; break; }
        case 201: { res = "(2/(1+exp(-z/(var(1,0)*var(1,0)))))-1"; break; }
        case 202: { res = "(2/(1+var(1,1)*exp(-var(1,2)*(z-var(1,3))/(var(1,0)*var(1,0))))^(1/var(1,2)))-1"; break; }
        case 203: { res = "ifthenelse(lt(real(z),0),-1,1)"; break; }
        case 204: { res = "ifthenelse(lt(real(z),0),-1,(z/(var(1,0)*var(1,0)))-1)"; break; }
        case 205: { res = "(2*log(var(1,1)+exp(z/(var(1,0)*var(1,0)))))-1"; break; }
        case 206: { res = "ifthenelse(lt(real(z),0),((var(1,1)*z)/(var(1,0)*var(1,0)))-1,(z/(var(1,0)*var(1,0)))-1)"; break; }
        case 300: { res = "-(x+y-2*z)/(2*var(1,0)*var(1,0))"; break; }
        case 301: { res.makeError("Error: 1-norm distance kernel cannot be returned as equation"); break; }
        case 302: { res.makeError("Error: inf-norm distance kernel cannot be returned as equation"); break; }
        case 303: { res.makeError("Error: 0-norm distance kernel cannot be returned as equation"); break; }
        case 304: { res.makeError("Error: r0-norm distance kernel cannot be returned as equation"); break; }
        case 400: { res.makeError("Error: monotonic 0/1 density kernel 1 cannot be returned as equation"); break; }
        case 401: { res.makeError("Error: monotonic 0/1 density kernel 2 cannot be returned as equation"); break; }
        case 402: { res.makeError("Error: monotonic 0/1 density kernel 3 cannot be returned as equation"); break; }
        case 403: { res.makeError("Error: monotonic 0/1 density kernel 4 cannot be returned as equation"); break; }
        case 404: { res.makeError("Error: monotonic 0/1 density kernel 5 cannot be returned as equation"); break; }
        case 450: { res.makeError("Error: monotonic 0/1 density kernel 1 (reversed order) cannot be returned as equation"); break; }
        case 451: { res.makeError("Error: monotonic 0/1 density kernel 2 (reversed order) cannot be returned as equation"); break; }
        case 452: { res.makeError("Error: monotonic 0/1 density kernel 3 (reversed order) cannot be returned as equation"); break; }
        case 453: { res.makeError("Error: monotonic 0/1 density kernel 4 (reversed order) cannot be returned as equation"); break; }
        case 454: { res.makeError("Error: monotonic 0/1 density kernel 5 (reversed order) cannot be returned as equation"); break; }
        case 500: { res.makeError("Error: monotonic 0/1 density derivative kernel 1 cannot be returned as equation"); break; }
        case 501: { res.makeError("Error: monotonic 0/1 density derivative kernel 2 cannot be returned as equation"); break; }
        case 502: { res.makeError("Error: monotonic 0/1 density derivative kernel 3 cannot be returned as equation"); break; }
        case 503: { res.makeError("Error: monotonic 0/1 density derivative kernel 4 cannot be returned as equation"); break; }
        case 504: { res.makeError("Error: monotonic 0/1 density derivative kernel 5 cannot be returned as equation"); break; }
        case 550: { res.makeError("Error: monotonic 0/1 density derivative kernel 1 cannot be returned as equation"); break; }
        case 551: { res.makeError("Error: monotonic 0/1 density derivative kernel 2 cannot be returned as equation"); break; }
        case 552: { res.makeError("Error: monotonic 0/1 density derivative kernel 3 cannot be returned as equation"); break; }
        case 553: { res.makeError("Error: monotonic 0/1 density derivative kernel 4 cannot be returned as equation"); break; }
        case 554: { res.makeError("Error: monotonic 0/1 density derivative kernel 5 cannot be returned as equation"); break; }
        case 600: { res.makeError("Error: monotonic -1/1 density kernel 1 cannot be returned as equation"); break; }
        case 601: { res.makeError("Error: monotonic -1/1 density kernel 2 cannot be returned as equation"); break; }
        case 602: { res.makeError("Error: monotonic -1/1 density kernel 3 cannot be returned as equation"); break; }
        case 603: { res.makeError("Error: monotonic -1/1 density kernel 4 cannot be returned as equation"); break; }
        case 604: { res.makeError("Error: monotonic -1/1 density kernel 5 cannot be returned as equation"); break; }
        case 650: { res.makeError("Error: monotonic -1/1 density kernel 1 (reversed order) cannot be returned as equation"); break; }
        case 651: { res.makeError("Error: monotonic -1/1 density kernel 2 (reversed order) cannot be returned as equation"); break; }
        case 652: { res.makeError("Error: monotonic -1/1 density kernel 3 (reversed order) cannot be returned as equation"); break; }
        case 653: { res.makeError("Error: monotonic -1/1 density kernel 4 (reversed order) cannot be returned as equation"); break; }
        case 654: { res.makeError("Error: monotonic -1/1 density kernel 5 (reversed order) cannot be returned as equation"); break; }
        case 700: { res.makeError("Error: monotonic -1/1 density derivative kernel 1 cannot be returned as equation"); break; }
        case 701: { res.makeError("Error: monotonic -1/1 density derivative kernel 2 cannot be returned as equation"); break; }
        case 702: { res.makeError("Error: monotonic -1/1 density derivative kernel 3 cannot be returned as equation"); break; }
        case 703: { res.makeError("Error: monotonic -1/1 density derivative kernel 4 cannot be returned as equation"); break; }
        case 704: { res.makeError("Error: monotonic -1/1 density derivative kernel 5 cannot be returned as equation"); break; }
        case 750: { res.makeError("Error: monotonic -1/1 density derivative kernel 1 cannot be returned as equation"); break; }
        case 751: { res.makeError("Error: monotonic -1/1 density derivative kernel 2 cannot be returned as equation"); break; }
        case 752: { res.makeError("Error: monotonic -1/1 density derivative kernel 3 cannot be returned as equation"); break; }
        case 753: { res.makeError("Error: monotonic -1/1 density derivative kernel 4 cannot be returned as equation"); break; }
        case 754: { res.makeError("Error: monotonic -1/1 density derivative kernel 5 cannot be returned as equation"); break; }
        case 1003: { res.makeError("Error: Radius kernel cannot be returned as equation"); break; }
        case 1038: { res.makeError("Error: Radius kernel cannot be returned as equation"); break; }
        case 2003: { res.makeError("Error: Radius kernel cannot be returned as equation"); break; }
        case 2038: { res.makeError("Error: Radius kernel cannot be returned as equation"); break; }

        default:
	{
	    NiceThrow("Unknown kernel type.\n");

	    break;
	}
    }

    // Substitute as required
    //
    // always substitute var(0,3) = x+y-2*z
    // sub out integer constants if resmode == 2,3
    // sub out real constants if resmode == 3

    SparseVector<SparseVector<gentype> > subarray;

    subarray("&",0)("&",3) = "x+y-2*z";
    res.substitute(subarray);
    subarray("&",0).zero();
    subarray.zero();

    if ( !( resmode & 1 ) )
    {
        subarray("&",0)("&",0) =  *(xnorm(0));
        subarray("&",0)("&",1) =  *(xnorm(1));
        subarray("&",0)("&",2) =  txyprod;
        subarray("&",0)("&",2) += tyxprod;
        subarray("&",0)("&",2) /= 2.0;
    }

    if ( !( resmode & 2 ) && ( dIntConstants(q).size() > 0 ) )
    {
        int ij;

        for ( ij = 0 ; ij < dIntConstants(q).size() ; ++ij )
        {
            subarray("&",2)("&",ij) = dIntConstants(q)(ij);
        }
    }

    if ( !( resmode & 4 ) && ( dRealConstants(q).size()-1 > 0 ) )
    {
        int ij;

        for ( ij = 0 ; ij < dRealConstants(q).size()-1 ; ++ij )
        {
            subarray("&",1)("&",ij) = dRealConstants(q)(ij+1);
        }
    }

    res.substitute(subarray);

    NiceAssert( !res.isValStrErr() );
  }

  res *= symm;

  return;
}

//KERNELSHERE
int MercerKernel::reverseEngK(int m, gentype &res, const Vector<const vecInfo *> &xinfo, const Vector<const SparseVector<gentype> *> &x, double Kres) const
{
    if ( isProd() ||
         ( size() != 1 ) ||
         kinf(0).usesVector ||
         kinf(0).usesNorm ||
         kinf(0).usesMinDiff ||
         kinf(0).usesMaxDiff ||
         ( kinf(0).numflagsset() != 1) ||
         ( cType(0) == 0 ) ||
         ( cType(0) == 32 ) ||
         ( cType(0) == 501 ) )
    {
        return 1;
    }

    res = Kres;
    res /= cWeight(0);

    int q = 0;

    retVector<gentype> tmpva;

    const Vector<gentype> &r = dRealConstants(q)(1,1,dRealConstants(q).size()-1,tmpva);
    const Vector<int> &ic = dIntConstants(q);

//ADDHERE - kernel reverse engineering here (diffis and inner only)
    switch ( cType(0) )
    {
        case 1:
	{
            res *= r(0);
            res *= r(0);

	    break;
	}

        case 2:
	{
            if ( ic(0)%2 )
            {
                gentype tempiz(ic(0));

                OP_einv(tempiz);

                res = epow(res,tempiz);
                res -= r(1);
                res *= r(0);
                res *= r(0);
            }

            else
            {
                // sign ambiguity means we can't do this case
                goto failure_fallthrough;
            }

	    break;
	}

        case 3:
	{
            OP_log(res);
            res += r(1);
            res *= -2.0;
            res *= r(0);
            res *= r(0);

            break;
	}

        case 4:
	{
            OP_log(res);
            res += r(1);
            res.negate();
            res *= r(0);
            res *= res;

            break;
	}

        case 5:
	{
            gentype tempr(r(1));

            OP_log(res);
            res += r(2);
            res.negate();
            res *= epow(r(0),r(1));
            res *= r(1);

            OP_einv(tempr);
            tempr *= 0.5;

            res = epow(res,tempr);

            break;
	}

        case 7:
	{
            OP_atanh(res);
            res -= r(1);
            res *= r(0);
            res *= r(0);

            break;
	}

        case 9:
	{
            res *= res;
            res -= r(1)*r(1);
            res *= r(0);
            res *= r(0);

            break;
	}

        case 10:
	{
            OP_einv(res);
            res *= res;
            res -= r(1)*r(1);
            res *= r(0);
            res *= r(0);

            break;
	}

        case 14:
	{
            res.negate();
            res = epow(res,0.5/r(1));
            res *= r(0);
            res *= r(0);

            break;
	}

        case 15:
	{
            res.negate();
            OP_exp(res);
            res -= 1.0;
            res = epow(res,0.5/r(1));
            res *= r(0);
            res *= r(0);

            break;
	}

        case 19:
	{
            OP_einv(res);
            res -= 1.0;
            res *= r(0);
            res *= r(0);

            break;
	}

        case 25:
	{
            res /= NUMBASE_PI;
            OP_acosh(res);
            res -= NUMBASE_PI;
            res.negate();
            res *= r(0);
            res *= res;

            break;
	}

        case 26:
	{
            res = epow(res,0.5/r(1));
            res *= r(0);

            break;
	}

        case 42:
	{
            OP_gd(res);
            res *= r(0);
            res *= r(0);

            break;
	}

        case 100:
	{
            res *= r(0);
            res *= r(0);

	    break;
	}

        case 101:
	{
            OP_einv(res);
            res -= 1.0;
            OP_log(res);
            res *= r(0);
            res *= r(0);
            res.negate();

	    break;
	}

        case 102:
	{
            OP_einv(res);
            res = epow(res,r(2));
            res -= 1.0;
            res /= r(1);
            OP_log(res);
            res.negate();
            res /= r(2);
            res *= r(0);
            res *= r(0);
            res += r(3);

	    break;
	}

        case 105:
	{
            OP_exp(res);
            res -= r(1);
            OP_log(res);
            res *= r(0);
            res *= r(0);

	    break;
	}

        case 200:
	{
            res += 1.0;
            res *= r(0);
            res *= r(0);

	    break;
	}

        case 201:
	{
            res += 1.0;
            res *= 0.5;
            OP_einv(res);
            res -= 1.0;
            OP_log(res);
            res *= r(0);
            res *= r(0);
            res.negate();

	    break;
	}

        case 202:
	{
            res += 1.0;
            res *= 0.5;
            OP_einv(res);
            res = epow(res,r(2));
            res -= 1.0;
            res /= r(1);
            OP_log(res);
            res.negate();
            res /= r(2);
            res *= r(0);
            res *= r(0);
            res += r(3);

	    break;
	}

        case 205:
	{
            res += 1.0;
            res *= 0.5;
            OP_exp(res);
            res -= r(1);
            OP_log(res);
            res *= r(0);
            res *= r(0);

	    break;
	}

        case 300:
	{
            res *= -2.0;
            res *= r(0);
            res *= r(0);

            break;
	}

        default:
	{
            failure_fallthrough:
            return 1;

	    break;
	}
    }

    if ( needsDiff(0) && ( kinf(0).numflagsset() == 1 ) )
    {
        // Currently res = ||x-y||^2
        //               = ||x||^2 + ||y||^2 - 2<x,y>
        // <x,y> = (||x||^2+||y||^2-res)/2
        //
        // This generalises to:
        //
        // ||x1,x2,...||^2 = ||x1||^2 + ||x2||^2 + ... - m.<x1,x2,...>

        res.negate();

        if ( m )
        {
            int i;

            for ( i = 0 ; i < m ; ++i )
            {
                res += getmnorm(*(xinfo(i)),*(x(i)),m);
            }
        }

        res /= m;
    }

    return 0;
}


//KERNELSHERE


MercerKernel &MercerKernel::setType(int ndtype, int q)
{
    NiceAssert( ndtype >= 0 );

    xisfast       = -1;
    xneedsInner   = -1;
    xneedsInnerm2 = -1;
    xneedsDiff    = -1;
    xneedsNorm    = -1;

    if ( ( dtype(q)/100 == 8 ) || ( ndtype/100 == 8 ) )
    {
        xisIPdiffered = 1;
    }

    dtype("&",q) = ndtype;

    // sourcelist: http://crsouza.blogspot.com/2010/03/kernel-functions-for-machine-learning.html

    Vector<gentype> &r = dRealConstants("&",q);
    Vector<int> &ic = dIntConstants("&",q);
    kernInfo &ki = kernflags("&",q);

    fillOutInfo(ki,r,ic,cType(q));

    r("&",0) = 1.0;

    dRealConstantsLB("&",q).resize(dRealConstants(q).size());
    dRealConstantsUB("&",q).resize(dRealConstants(q).size());

    dRealConstantsLB("&",q) = nullgentype();
    dRealConstantsUB("&",q) = nullgentype();

    dIntConstantsLB("&",q).resize(dIntConstants(q).size());
    dIntConstantsUB("&",q).resize(dIntConstants(q).size());

    dIntConstantsLB("&",q) = 0;
    dIntConstantsUB("&",q) = 0;

    recalcRandFeats(q);

    return *this;
}


kernInfo &fillOutInfo(kernInfo &ki, Vector<gentype> &r, Vector<int> &ic, int ktype)
{
//ADDHERE - new kernel initialisation function goes here
    switch ( ktype )
    {
        case   0: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+2);                 r("&",1+0) = 1.0; r("&",1+1) = 1.0;                                         break; }
        case   1: { ki.usesDiff = 0; ki.usesInner = 1; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                           break; }
        case   2: { ki.usesDiff = 0; ki.usesInner = 1; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(1); r.resize(1+2);  ic("&",0) = 2; r("&",1+0) = 1.0; r("&",1+1) = 1.0;                                         break; }
        case   3: { ki.usesDiff = 1; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+2);                 r("&",1+0) = 1.0; r("&",1+1) = 0.0;                                         break; }
        case   4: { ki.usesDiff = 1; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+2);                 r("&",1+0) = 1.0; r("&",1+1) = 0.0;                                         break; }
        case   5: { ki.usesDiff = 1; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+3);                 r("&",1+0) = 1.0; r("&",1+1) = 3.0; r("&",1+2) = 0.0;                       break; }
        case   6: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+5);                 r("&",1+0) = 1.0; r("&",1+1) = 1.0; r("&",1+2) = 2.0;  r("&",1+3) = 3.0; r("&",1+4) = 1.0;    break; }
        case   7: { ki.usesDiff = 0; ki.usesInner = 1; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+2);                 r("&",1+0) = 1.0; r("&",1+1) = 1.0;                                         break; }
        case   8: { ki.usesDiff = 1; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+2);                 r("&",1+0) = 1.0; r("&",1+1) = 1.0;                                         break; }
        case   9: { ki.usesDiff = 1; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+2);                 r("&",1+0) = 1.0; r("&",1+1) = 1.0;                                         break; }
        case  10: { ki.usesDiff = 1; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+2);                 r("&",1+0) = 1.0; r("&",1+1) = 1.0;                                         break; }
        case  11: { ki.usesDiff = 1; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                           break; }
        case  12: { ki.usesDiff = 1; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                           break; }
        case  13: { ki.usesDiff = 1; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                           break; }
        case  14: { ki.usesDiff = 1; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+2);                 r("&",1+0) = 1.0; r("&",1+1) = 1.0;                                         break; }
        case  15: { ki.usesDiff = 1; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+2);                 r("&",1+0) = 1.0; r("&",1+1) = 1.0;                                         break; }
        case  16: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                           break; }
        case  17: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(1); r.resize(1+1);  ic("&",0) = 1; r("&",1+0) = 1.0;                                                           break; }
        case  18: { ki.usesDiff = 1; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(1); r.resize(1+2);  ic("&",0) = 1; r("&",1+0) = 1.0; r("&",1+1) = 1.0;                                         break; }
        case  19: { ki.usesDiff = 1; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                           break; }
        case  20: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                           break; }
        case  21: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                           break; }
        case  22: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+3);                 r("&",1+0) = 1.0; r("&",1+1) = 1.0; r("&",1+2) = 1.0;                      break; }
        case  23: { ki.usesDiff = 1; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+2);                 r("&",1+0) = 1.0; r("&",1+1) = 1.0;                                        break; }
        case  24: { ki.usesDiff = 0; ki.usesInner = 1; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(1); r.resize(1+1);  ic("&",0) = 2; r("&",1+0) = 1.0;                                                          break; }
        case  25: { ki.usesDiff = 1; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                           break; }
        case  26: { ki.usesDiff = 1; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+2);                 r("&",1+0) = 1.0; r("&",1+1) = 1;                                          break; }
        case  27: { ki.usesDiff = 1; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+2);                 r("&",1+0) = 1.0; r("&",1+1) = 1;                                          break; }
        case  28: { ki.usesDiff = 1; ki.usesInner = 1; ki.usesNorm = 1; ki.usesVector = 1; ki.usesMinDiff = 1; ki.usesMaxDiff = 1; ic.resize(0); r.resize(1+11);                r("&",1+0) = 1.0; r("&",1+1) = 1.0; r("&",1+2) = 1.0;  r("&",1+3) = 1.0; r("&",1+4) = 1.0; r("&",1+5) = 1.0; r("&",1+6) = 1.0; r("&",1+7) = 1.0; r("&",1+8) = 1.0; r("&",1+9) = 1.0; r("&",1+10) = "(var(0,1)+var(0,2))/2"; break; }
        case  29: { ki.usesDiff = 0; ki.usesInner = 1; ki.usesNorm = 1; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(1); r.resize(1+1);  ic("&",0) = 0; r("&",1+0) = 1.0;                                                            break; }
        case  30: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(1); r.resize(1+3);  ic("&",0) = 0; r("&",1+0) = 1.0; r("&",1+1) = 1.8; r("&",1+2) = 1e-5;                      break; }
        case  31: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(1); r.resize(1+2);  ic("&",0) = 0; r("&",1+2) = 1.0; r("&",1+1) = 1.8; r("&",1+2) = 1e-5;                      break; }
        case  32: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+2);                 r("&",1+0) = 1.0; r("&",1+1) = 1.0;                                        break; }
        case  33: { ki.usesDiff = 1; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                           break; }
        case  34: { ki.usesDiff = 1; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                           break; }
        case  35: { ki.usesDiff = 1; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(1); r.resize(1+1);  ic("&",0) = 1; r("&",1+0) = 1.0;                                                           break; }
        case  36: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                           break; }
        case  37: { ki.usesDiff = 1; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(1); r.resize(1+1);  ic("&",0) = 0; r("&",1+0) = 1.0;                                                           break; }
        case  38: { ki.usesDiff = 1; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                           break; }
        case  39: { ki.usesDiff = 1; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                           break; }
        case  40: { ki.usesDiff = 1; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                           break; }
        case  41: { ki.usesDiff = 0; ki.usesInner = 1; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                           break; }
        case  42: { ki.usesDiff = 0; ki.usesInner = 1; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                          break; }
        case  43: { ki.usesDiff = 0; ki.usesInner = 1; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                          break; }
        case  44: { ki.usesDiff = 0; ki.usesInner = 1; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+2);                 r("&",1+0) = 1.0; r("&",1+1) = 0.0;                                         break; }
        case  45: { ki.usesDiff = 0; ki.usesInner = 1; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                           break; }
        case  46: { ki.usesDiff = 0; ki.usesInner = 1; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                           break; }
        case  47: { ki.usesDiff = 1; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+2);                 r("&",1+0) = 1.0; r("&",1+1) = 1.0;                                           break; }
        case  48: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+2);                 r("&",1+0) = 1.0; r("&",1+1) = 1.0;                                                                                                                                                                                         break; }
        case  49: { ki.usesDiff = 1; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+3);                 r("&",1+0) = 1.0; r("&",1+1) = 0.0;  r("&",1+2) = 0.5;                      break; }
        case  50: { ki.usesDiff = 0; ki.usesInner = 1; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                           break; }
        case  51: { ki.usesDiff = 0; ki.usesInner = 1; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                           break; }
        case  52: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 1; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                           break; }
        case  53: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 1; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+3);                 r("&",1+0) = 1.0; r("&",1+1) = 1.0; r("&",1+2) = 1.0;                        break; }
        case 100: { ki.usesDiff = 0; ki.usesInner = 1; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                            break; }
        case 101: { ki.usesDiff = 0; ki.usesInner = 1; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                           break; }
        case 102: { ki.usesDiff = 0; ki.usesInner = 1; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+4);                 r("&",1+0) = 1.0; r("&",1+1) = 1.0; r("&",1+2) = 1.0;  r("&",1+3) = 0.0; break; }
        case 103: { ki.usesDiff = 0; ki.usesInner = 1; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                            break; }
        case 104: { ki.usesDiff = 0; ki.usesInner = 1; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                             break; }
        case 105: { ki.usesDiff = 0; ki.usesInner = 1; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+2);                 r("&",1+0) = 1.0; r("&",1+1) = 1.0;                                        break; }
        case 106: { ki.usesDiff = 0; ki.usesInner = 1; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+2);                 r("&",1+0) = 1.0; r("&",1+1) = 0.001;                                         break; }
        case 200: { ki.usesDiff = 0; ki.usesInner = 1; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                             break; }
        case 201: { ki.usesDiff = 0; ki.usesInner = 1; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                           break; }
        case 202: { ki.usesDiff = 0; ki.usesInner = 1; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+4);                 r("&",1+0) = 1.0; r("&",1+1) = 1.0; r("&",1+2) = 1.0;  r("&",1+3) = 0.0; break; }
        case 203: { ki.usesDiff = 0; ki.usesInner = 1; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                             break; }
        case 204: { ki.usesDiff = 0; ki.usesInner = 1; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                             break; }
        case 205: { ki.usesDiff = 0; ki.usesInner = 1; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+2);                 r("&",1+0) = 1.0; r("&",1+1) = 1.0;                                           break; }
        case 206: { ki.usesDiff = 0; ki.usesInner = 1; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+2);                 r("&",1+0) = 1.0; r("&",1+1) = 0.001;                                         break; }
        case 300: { ki.usesDiff = 1; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                             break; }
        case 301: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                             break; }
        case 302: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                             break; }
        case 303: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                             break; }
        case 304: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+2);                 r("&",1+0) = 1.0; r("&",1+1) = 2.0;                                          break; }
        case 400: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+2);                 r("&",1+0) = 1.0; r("&",1+1) = 1.0;                                        break; }
        case 401: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+2);                 r("&",1+0) = 1.0; r("&",1+1) = 1.0;                                        break; }
        case 402: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 1; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+2);                 r("&",1+0) = 1.0; r("&",1+1) = 1.0;                                        break; }
        case 403: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 1; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                             break; }
        case 404: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 1; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                             break; }
        case 450: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+2);                 r("&",1+0) = 1.0; r("&",1+1) = 1.0;                                        break; }
        case 451: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+2);                 r("&",1+0) = 1.0; r("&",1+1) = 1.0;                                        break; }
        case 452: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 1; ic.resize(0); r.resize(1+2);                 r("&",1+0) = 1.0; r("&",1+1) = 1.0;                                        break; }
        case 453: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 1; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                             break; }
        case 454: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 1; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                             break; }
        case 500: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                           break; }
        case 501: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                           break; }
        case 502: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 1; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                           break; }
        case 503: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                             break; }
        case 504: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                             break; }
        case 550: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                           break; }
        case 551: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                           break; }
        case 552: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 1; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                           break; }
        case 553: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                             break; }
        case 554: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                             break; }
        case 600: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+2);                 r("&",1+0) = 1.0; r("&",1+1) = 1.0;                                        break; }
        case 601: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+2);                 r("&",1+0) = 1.0; r("&",1+1) = 1.0;                                        break; }
        case 602: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 1; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+2);                 r("&",1+0) = 1.0; r("&",1+1) = 1.0;                                        break; }
        case 603: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 1; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                             break; }
        case 604: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 1; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                             break; }
        case 650: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+2);                 r("&",1+0) = 1.0; r("&",1+1) = 1.0;                                        break; }
        case 651: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+2);                 r("&",1+0) = 1.0; r("&",1+1) = 1.0;                                        break; }
        case 652: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 1; ic.resize(0); r.resize(1+2);                 r("&",1+0) = 1.0; r("&",1+1) = 1.0;                                        break; }
        case 653: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 1; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                             break; }
        case 654: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 1; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                             break; }
        case 700: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                           break; }
        case 701: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                           break; }
        case 702: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 1; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                           break; }
        case 703: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                             break; }
        case 704: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                             break; }
        case 750: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                           break; }
        case 751: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                           break; }
        case 752: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 1; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                           break; }
        case 753: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                             break; }
        case 754: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+1);                 r("&",1+0) = 1.0;                                                             break; }

        case 800: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 801: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 802: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 803: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 804: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 805: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 806: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 807: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 808: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 809: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 810: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 811: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 812: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 813: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 814: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 815: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 816: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 817: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 818: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 819: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 820: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 821: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 822: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 823: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 824: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 825: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 826: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 827: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 828: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 829: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 830: { ki.usesDiff = 0; ki.usesInner = 1; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 831: { ki.usesDiff = 0; ki.usesInner = 1; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 832: { ki.usesDiff = 0; ki.usesInner = 1; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 833: { ki.usesDiff = 0; ki.usesInner = 1; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 834: { ki.usesDiff = 0; ki.usesInner = 1; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 835: { ki.usesDiff = 0; ki.usesInner = 1; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 836: { ki.usesDiff = 0; ki.usesInner = 1; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 837: { ki.usesDiff = 0; ki.usesInner = 1; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 838: { ki.usesDiff = 0; ki.usesInner = 1; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 839: { ki.usesDiff = 0; ki.usesInner = 1; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 840: { ki.usesDiff = 0; ki.usesInner = 1; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 841: { ki.usesDiff = 0; ki.usesInner = 1; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 842: { ki.usesDiff = 0; ki.usesInner = 1; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 843: { ki.usesDiff = 0; ki.usesInner = 1; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 844: { ki.usesDiff = 0; ki.usesInner = 1; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 845: { ki.usesDiff = 0; ki.usesInner = 1; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 846: { ki.usesDiff = 0; ki.usesInner = 1; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 847: { ki.usesDiff = 0; ki.usesInner = 1; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 848: { ki.usesDiff = 0; ki.usesInner = 1; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 849: { ki.usesDiff = 0; ki.usesInner = 1; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 850: { ki.usesDiff = 0; ki.usesInner = 1; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 851: { ki.usesDiff = 0; ki.usesInner = 1; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 852: { ki.usesDiff = 0; ki.usesInner = 1; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 853: { ki.usesDiff = 0; ki.usesInner = 1; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 854: { ki.usesDiff = 0; ki.usesInner = 1; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 855: { ki.usesDiff = 0; ki.usesInner = 1; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 856: { ki.usesDiff = 0; ki.usesInner = 1; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 857: { ki.usesDiff = 0; ki.usesInner = 1; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 858: { ki.usesDiff = 0; ki.usesInner = 1; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 859: { ki.usesDiff = 0; ki.usesInner = 1; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 860: { ki.usesDiff = 1; ki.usesInner = 1; ki.usesNorm = 1; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 861: { ki.usesDiff = 1; ki.usesInner = 1; ki.usesNorm = 1; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 862: { ki.usesDiff = 1; ki.usesInner = 1; ki.usesNorm = 1; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 863: { ki.usesDiff = 1; ki.usesInner = 1; ki.usesNorm = 1; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 864: { ki.usesDiff = 1; ki.usesInner = 1; ki.usesNorm = 1; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 865: { ki.usesDiff = 1; ki.usesInner = 1; ki.usesNorm = 1; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 866: { ki.usesDiff = 1; ki.usesInner = 1; ki.usesNorm = 1; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 867: { ki.usesDiff = 1; ki.usesInner = 1; ki.usesNorm = 1; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 868: { ki.usesDiff = 1; ki.usesInner = 1; ki.usesNorm = 1; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 869: { ki.usesDiff = 1; ki.usesInner = 1; ki.usesNorm = 1; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 870: { ki.usesDiff = 1; ki.usesInner = 1; ki.usesNorm = 1; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 871: { ki.usesDiff = 1; ki.usesInner = 1; ki.usesNorm = 1; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 872: { ki.usesDiff = 1; ki.usesInner = 1; ki.usesNorm = 1; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 873: { ki.usesDiff = 1; ki.usesInner = 1; ki.usesNorm = 1; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 874: { ki.usesDiff = 1; ki.usesInner = 1; ki.usesNorm = 1; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 875: { ki.usesDiff = 1; ki.usesInner = 1; ki.usesNorm = 1; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 876: { ki.usesDiff = 1; ki.usesInner = 1; ki.usesNorm = 1; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 877: { ki.usesDiff = 1; ki.usesInner = 1; ki.usesNorm = 1; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 878: { ki.usesDiff = 1; ki.usesInner = 1; ki.usesNorm = 1; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 879: { ki.usesDiff = 1; ki.usesInner = 1; ki.usesNorm = 1; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 880: { ki.usesDiff = 1; ki.usesInner = 1; ki.usesNorm = 1; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 881: { ki.usesDiff = 1; ki.usesInner = 1; ki.usesNorm = 1; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 882: { ki.usesDiff = 1; ki.usesInner = 1; ki.usesNorm = 1; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 883: { ki.usesDiff = 1; ki.usesInner = 1; ki.usesNorm = 1; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 884: { ki.usesDiff = 1; ki.usesInner = 1; ki.usesNorm = 1; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 885: { ki.usesDiff = 1; ki.usesInner = 1; ki.usesNorm = 1; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 886: { ki.usesDiff = 1; ki.usesInner = 1; ki.usesNorm = 1; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 887: { ki.usesDiff = 1; ki.usesInner = 1; ki.usesNorm = 1; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 888: { ki.usesDiff = 1; ki.usesInner = 1; ki.usesNorm = 1; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 889: { ki.usesDiff = 1; ki.usesInner = 1; ki.usesNorm = 1; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 890: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 891: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 892: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 893: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 894: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 895: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 896: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 897: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 898: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }
        case 899: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 0; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+0);                                                                                               break; }

        case 1003: { ki.usesDiff = 1; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+2);                 r("&",1+0) = 1.0; r("&",1+1) = 0.0;                                         break; }
        case 1038: { ki.usesDiff = 1; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+2);                 r("&",1+0) = 1.0; r("&",1+1) = 0.0;                                         break; }

        case 2003: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+2);                 r("&",1+0) = 1.0; r("&",1+1) = 0.0;                                         break; }
        case 2038: { ki.usesDiff = 0; ki.usesInner = 0; ki.usesNorm = 0; ki.usesVector = 1; ki.usesMinDiff = 0; ki.usesMaxDiff = 0; ic.resize(0); r.resize(1+2);                 r("&",1+0) = 1.0; r("&",1+1) = 0.0;                                         break; }

        default:
	{
	    NiceThrow("Unknown kernel type.\n");

            break;
	}
    }

    return ki;
}






























// ----------------------------------------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------------------------------------------------

int MercerKernel::reverseEngK(gentype &res, const vecInfo &xinfo, const vecInfo &yinfo, const SparseVector<gentype> &x, const SparseVector<gentype> &y, double Kres) const
{
    Vector<const vecInfo *> xyinfo(2);
    Vector<const SparseVector<gentype> *> xy(2);

    xyinfo("&",0) = &xinfo;
    xyinfo("&",1)         = &yinfo;

    xy("&",0) = &x;
    xy("&",1)         = &y;

    return reverseEngK(2,res,xyinfo,xy,Kres);
}

int MercerKernel::reverseEngK(gentype &res, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, double Kres) const
{
    Vector<const vecInfo *> xyinfo(4);
    Vector<const SparseVector<gentype> *> xy(4);

    xyinfo("&",0) = &xainfo;
    xyinfo("&",1)         = &xbinfo;
    xyinfo("&",2)         = &xcinfo;
    xyinfo("&",3)         = &xdinfo;

    xy("&",0) = &xa;
    xy("&",1)         = &xb;
    xy("&",2)         = &xc;
    xy("&",3)         = &xd;

    return reverseEngK(4,res,xyinfo,xy,Kres);
}






//KERNELSHERRE

void MercerKernel::dKdaBase(gentype &res, int &minmaxind, int q, 
                            const gentype &xyprod, const gentype &yxprod, const gentype &diffis, 
                            Vector<const SparseVector<gentype> *> &x,
                            Vector<const vecInfo *> &xinfo,
                            Vector<const gentype *> &xnorm,
                            Vector<int> &i,
                            int xdim, int m, int mlid) const
{
    NiceAssert( q >= 0 );
    NiceAssert( q < size() );

    switch ( cType(q) )
    {
        case 0:
        case 1:
        case 2:
        case 7:
        case 24:
        case 32:
        case 100:
        case 101:
        case 102:
        case 103:
        case 104:
        case 105:
        case 106:
        case 200:
        case 201:
        case 202:
        case 203:
        case 204:
        case 205:
        case 206:
        {
            // Inner product kernel, gradient zero

            res = 0.0;

            minmaxind = combineminmaxind(minmaxind,-2);

            break;
        }

        case 3:
        case 4:
        case 5:
        case 8:
        case 9:
        case 10:
        case 11:
        case 12:
        case 13:
        case 14:
        case 15:
        case 18:
        case 19:
        case 23:
        case 25:
        case 26:
        case 27:
        case 33:
        case 34:
        case 35:
        case 37:
        case 38:
        case 39:
        case 40:
        case 47:
        case 300:
        case 501:
        case 701:
        {
            // For kernels that are functions of diffis we can simply note that:
            //
            // dK(diffis)/dz = dK(diffis)/ddiffis . ddiffis/dz
            // dK(diffis)/da = dK(diffis)/ddiffis . ddiffis/da
            //
            // so: dK(diffis)/da = -1/2 dK(diffis)/dz
            // as diffis = a+b-2z

            dKdzBase(res,minmaxind,q,xyprod,yxprod,diffis,x,xinfo,xnorm,i,xdim,m,mlid);

            res /= -2.0;

            break;
        }

        case 29:
        {
            // Arc-cosine (no gradient)

            NiceThrow("Arc-cosine gradient not implemented");

            break;
        }

        case 800: case 801: case 802: case 803: case 804: case 805: case 806: case 807: case 808: case 809:
        case 810: case 811: case 812: case 813: case 814: case 815: case 816: case 817: case 818: case 819:
        case 820: case 821: case 822: case 823: case 824: case 825: case 826: case 827: case 828: case 829:
        case 830: case 831: case 832: case 833: case 834: case 835: case 836: case 837: case 838: case 839:
        case 840: case 841: case 842: case 843: case 844: case 845: case 846: case 847: case 848: case 849:
        case 850: case 851: case 852: case 853: case 854: case 855: case 856: case 857: case 858: case 859:
        case 860: case 861: case 862: case 863: case 864: case 865: case 866: case 867: case 868: case 869:
        case 870: case 871: case 872: case 873: case 874: case 875: case 876: case 877: case 878: case 879:
        case 880: case 881: case 882: case 883: case 884: case 885: case 886: case 887: case 888: case 889:
        case 890: case 891: case 892: case 893: case 894: case 895: case 896: case 897: case 898: case 899:
        {
            kernel8xx(q,res,minmaxind,cType(q),xyprod,yxprod,diffis,x,xinfo,i,xdim,m,16,mlid);

            break;
        }

        default:
        {
            NiceThrow("Only inner product and norm difference kernels have kernel x/y gradients defined.");

            break;
        }
    }

    return;
}


//KERNELSHERE

void MercerKernel::dKdzBase(gentype &res, int &minmaxind, int q, 
                            const gentype &txyprod, const gentype &tyxprod, const gentype &diffis, 
                            Vector<const SparseVector<gentype> *> &x,
                            Vector<const vecInfo *> &xinfo,
                            Vector<const gentype *> &xnorm,
                            Vector<int> &ii,
                            int xdim, int m, int mlid) const
{
    NiceAssert( q >= 0 );
    NiceAssert( q < size() );

    int minmaxres = -1;

    gentype xyprod;

    xyprod =  txyprod;
    xyprod += tyxprod;
    xyprod /= 2.0;

    // Apply to 0,1 only (design decision)

    processOverwrites(q,*(x(0)),*(x(1)));

    SparseVector<gentype> xx;
    SparseVector<gentype> yy;

    int i;

    if ( kinf(q).usesVector || kinf(q).usesMinDiff || kinf(q).usesMaxDiff )
    {
        if ( m != 2 )
        {
            NiceThrow("Vector-function kernels are not implemented for m != 2");
        }

        xx = *(x(0));
        yy = *(x(1));

        if ( isLeftNormal()  ) { preShiftScale(xx,*(x(0))); }
        if ( isRightNormal() ) { preShiftScale(yy,*(x(1))); }

        xx.conj();

        xx -= yy;
        xx += yy; // xx now has all indices
        yy -= xx;
        yy += xx; // yy now has all indices, shared with xx
    }

    retVector<gentype> tmpva;

    const Vector<gentype> &r = dRealConstants(q)(1,1,dRealConstants(q).size()-1,tmpva);
    const Vector<int> &ic = dIntConstants(q);

    int typeis = cType(q);

//ADDHERE - kernel derivative is implemented here (either ||x-y||^2 or <x,y>)
    switch ( typeis )
    {
        case 0:
        {
            res = 0.0;
            break;
        }

        case 1:
        {
            res =  1.0;
            res /= r(0);
            res /= r(0);

            break;
        }

        case 2:
        {
            res  = xyprod;
            res /= r(0);
            res /= r(0);
            res += r(1);
            raiseto(res,ic(0)-1);
            res /= r(0);
            res /= r(0);
            res *= ic(0);

            break;
        }

        case 3:
        {
            Kbase(res,q,cType(q),txyprod,tyxprod,diffis,x,xinfo,xnorm,ii,xdim,m,0,0,0,mlid);

            res /= r(0);
            res /= r(0);

            break;
        }

        case 4:
        {
            if ( (double) abs2(diffis) >= BADZEROTOL*BADZEROTOL )
            {
                Kbase(res,q,cType(q),txyprod,tyxprod,diffis,x,xinfo,xnorm,ii,xdim,m,0,0,0,mlid);
                res /= r(0);
                res /= sqrt(diffis);
            }

            else
            {
                res = 1.0;
            }

            break;
        }

        case 5:
        {
            gentype temp(diffis);
            gentype tempb;

            OP_sqrt(temp);

            tempb = r(1);
            tempb -= 2.0;
            tempb.negate();

            if ( (double) abs2(temp) >= BADZEROTOL )
            {
                temp = epow(temp,tempb);

                Kbase(res,q,cType(q),txyprod,tyxprod,diffis,x,xinfo,xnorm,ii,xdim,m,0,0,0,mlid);

                res /= epow(r(0),r(1));
                res /= temp;
            }

            else
            {
                res = 1.0;
            }

            break;
        }

        case 7:
        {
            res = xyprod;
            res /= r(0);
            res /= r(0);
            res += r(1);
            OP_sech(res);
            res *= res;
            res /= r(0);
            res /= r(0);

            break;
        }

        case 8:
        {
            res  = diffis;
            res /= 2.0;
            res /= r(0);
            res /= r(0);
            res /= r(1);
            res += 1.0;
            res = epow(res,-r(1)-1.0);
            res /= -2.0;
            res /= r(0);
            res /= r(0);

//OLD            res = diffis;
//OLD            res += r(0);
//OLD            res *= res;
//OLD            OP_einv(res);
//OLD            res *= r(0);
//OLD            res *= 2.0;

            break;
        }

        case 9:
        {
            gentype temp;

            Kbase(temp,q,cType(q),txyprod,tyxprod,diffis,x,xinfo,xnorm,ii,xdim,m,0,0,0,mlid);

            res = -1.0;
            res /= temp;
            res /= r(0);
            res /= r(0);

            break;
        }

        case 10:
        {
            gentype temp;

            Kbase(temp,q,cType(q),txyprod,tyxprod,diffis,x,xinfo,xnorm,ii,xdim,m,0,0,0,mlid);

            res = temp;
            raiseto(res,3);
            res /= r(0);
            res /= r(0);

            break;
        }

        case 11:
        {
            res = r(0);
            res *= r(0);
            res -= diffis;
            res /= diffis;
            OP_einv(res);
            OP_sqrt(res);
            res *= 4.0/NUMBASE_PI;
            res /= r(0);
            res /= r(0);

            break;
        }

        case 12:
        {
            res = diffis;
            OP_sqrt(res);
            res *= r(0);
            res *= r(0);
            res -= sqrt(diffis);
            res *= 1.5;
            res /= r(0);
            res /= r(0);
            res /= r(0);

            break;
        }

        case 13:
        {
            if ( (double) abs2(diffis) >= BADZEROTOL*BADZEROTOL )
            {
                gentype qq(diffis);

                OP_sqrt(qq);
                qq /= r(0);

                gentype rr(qq);

                OP_sinc(rr);

                res = qq;
                OP_cos(res);
                res -= rr;
                res /= qq;
                res /= 2.0;
                res /= r(0);
                res /= r(0);
                res /= qq;
                res *= -2.0;
            }

            else
            {
                res = 1.0;
            }

            break;
        }

        case 14:
        {
            gentype raiseto(r(1));

            raiseto /= 2.0;
            raiseto -= 1.0;

            res = diffis;
            res /= r(0);
            res /= r(0);
            res = epow(res,raiseto);
            res *= r(1);
            res /= r(0);
            res /= r(0);

            break;
        }

        case 15:
        {
            gentype raisetoa(r(1));

            raisetoa /= 2.0;

            gentype raisetob(raisetoa);

            raisetob -= 1.0;

            gentype temp = diffis;

            temp /= r(0);
            temp /= r(0);

            temp = epow(temp,raisetoa);

            temp += 1.0;

            res = epow(diffis,raisetob);
            res /= temp;
            res *= r(1);

            res /= r(0);
            res /= r(0);

            break;
        }

        case 18:
        {
            NiceThrow("Bessel kernel not implemented yet");
            break;
        }

        case 19:
        {
            Kbase(res,q,cType(q),txyprod,tyxprod,diffis,x,xinfo,xnorm,ii,xdim,m,0,0,0,mlid);

            res *= res;
            res *= 2.0;
            res /= r(0);
            res /= r(0);

            break;
        }

        case 23:
        {
            gentype raiseto(r(1));

            raiseto -= 1.0;

            Kbase(res,q,cType(q),txyprod,tyxprod,diffis,x,xinfo,xnorm,ii,xdim,m,0,0,0,mlid);

            res *= res;
            res *= 2.0;
            res /= r(0);
            res *= r(1);
            res *= epow(diffis,raiseto);

            break;
        }

        case 24:
        {
            gentype tmp = xyprod;

            tmp /= r(0);
            tmp /= r(0);

            gentype temp(1.0);

            temp -= tmp;

            gentype tempb(tmp);

            raiseto(tempb,ic(0)-1);
            tempb *= ic(0);

            res = tmp;
            raiseto(res,ic(0));
            res -= 1.0;
            res /= temp;
            res -= tempb;
            res /= temp;
            res /= temp;

            res /= r(0);
            res /= r(0);

            break;
        }

        case 25:
        {
            gentype qq(diffis);

            OP_sqrt(qq);

            res = qq;
            res /= r(0);
            res -= NUMBASE_PI;
            res.negate();
            OP_sinh(res);
            res *= NUMBASE_PI;
            res /= r(0);
            res /= qq;

            break;
        }

        case 26:
        {
            gentype raiseto(r(0));
            gentype multby(r(0));

            raiseto -= 0.5;
            multby += 0.5;

            res = diffis;
            res /= r(0);
            res = epow(res,r(0)-0.5);
            res *= (r(0)+0.5);
            res /= r(0);
            res *= -2.0;

            break;
        }

        case 27:
        {
            gentype qq(diffis);

            qq /= r(0);
            res = epow(qq,r(1));
            res *= log(sqrt(qq));
            res += 0.5;
            res /= qq;
            res *= r(1);
            res *= -2.0;
            res /= r(0);

            break;
        }

        case 32:
        {
            res = 0.0;
            break;
        }

        case 33:
        {
            res = 0.0;
            break;
        }

        case 34:
        {
            res = diffis;
            OP_sqrt(res);
            if ( real(res) < real(r(0)) )
            {
                res *= r(0);
                res *= r(0);
                OP_einv(res);
            }

            else
            {
                res = 0.0;
            }

            break;
        }

        case 42:
        {
            res  = xyprod;
            res /= r(0);
            res /= r(0);

            gentype tzsq = res;
            gentype czsq = res;

            OP_tan(tzsq);
            OP_sec(czsq);

            tzsq *= tzsq;
            czsq *= czsq;

            tzsq -= 1.0;
            tzsq *= -1.0;

            res  = czsq;
            res /= tzsq;
            res /= r(0);
            res /= r(0);

            break;
        }

        case 43:
        {
            res  = xyprod;
            res /= r(0);
            res /= r(0);
            res -= 1.0;
            res *= res;
            OP_einv(res);
            res /= r(0);
            res /= r(0);
            res /= r(0);
            res /= r(0);

            break;
        }

        case 44:
        {
            Kbase(res,q,cType(q),txyprod,tyxprod,diffis,x,xinfo,xnorm,ii,xdim,m,0,0,0,mlid);

            res /= r(0);
            res /= r(0);

            break;
        }

        case 45:
        {
            Kbase(res,q,cType(q),txyprod,tyxprod,diffis,x,xinfo,xnorm,ii,xdim,m,0,0,0,mlid);

            res /= r(0);
            res /= r(0);

            break;
        }

        case 46:
        {
            Kbase(res,q,cType(q),txyprod,tyxprod,diffis,x,xinfo,xnorm,ii,xdim,m,0,0,0,mlid);

            res /= r(0);
            res /= r(0);

            break;
        }

        case 100:
        {
            res = 1.0;

            res /= r(0);
            res /= r(0);

            break;
        }

        case 101:
        {
            gentype temp;

            Kbase(temp,q,cType(q),txyprod,tyxprod,diffis,x,xinfo,xnorm,ii,xdim,m,0,0,0,mlid);

            res = temp;
            temp.negate();
            temp += 1.0;
            res *= temp;
            res /= r(0);
            res /= r(0);

            break;
        }

        case 102:
        {
            gentype temp;

            Kbase(temp,q,cType(q),txyprod,tyxprod,diffis,x,xinfo,xnorm,ii,xdim,m,0,0,0,mlid);

            res = temp;
            raiseto(temp,2);
            temp.negate();
            temp += 1.0;
            res *= temp;
            res /= r(0);
            res /= r(0);

            break;
        }

        case 103:
        {
            res = 0.0;
            break;
        }

        case 104:
        {
            res = 0.0;

            if ( (double) real(xyprod) > BADZEROTOL )
            {
                res = 1.0;
                res /= r(0);
                res /= r(0);
            }

            break;
        }

        case 105:
        {
            gentype temp(xyprod);

            temp /= r(0);
            temp /= r(0);
            OP_exp(temp);
            res = temp;
            res += r(1);
            OP_einv(res);
            res *= temp;
            res /= r(0);
            res /= r(0);

            break;
        }

        case 106:
        {
            res = 0.0;

            if ( (double) real(xyprod) > BADZEROTOL )
            {
                res = 1.0;
                res /= r(0);
                res /= r(0);
            }

            else
            {
                res = r(1);
                res /= r(0);
                res /= r(0);
            }

            break;
        }

        case 200:
        {
            res = 1.0;

            res /= r(0);
            res /= r(0);

            break;
        }

        case 201:
        {
            gentype temp;

            Kbase(temp,q,cType(q),txyprod,tyxprod,diffis,x,xinfo,xnorm,ii,xdim,m,0,0,0,mlid);

            res = temp;
            res += 1.0;
            temp.negate();
            temp += 1.0;
            res *= temp;
            res /= r(0);
            res /= r(0);
            res *= 0.5;

            break;
        }

        case 202:
        {
            gentype temp;

            Kbase(temp,q,102,txyprod,tyxprod,diffis,x,xinfo,xnorm,ii,xdim,m,0,0,0,mlid);

            res = temp;
            temp = epow(temp,r(2));
            temp.negate();
            temp += 1.0;
            res *= temp;
            res /= r(0);
            res /= r(0);
            res *= 2.0;

            break;
        }

        case 203:
        {
            res = 0.0;
            break;
        }

        case 204:
        {
            res = 0.0;

            if ( (double) real(xyprod) > BADZEROTOL )
            {
                res = 1.0;
                res /= r(0);
                res /= r(0);
            }

            break;
        }

        case 205:
        {
            gentype temp(xyprod);

            temp /= r(0);
            temp /= r(0);
            OP_exp(temp);

            res = temp;
            res += r(1);
            OP_einv(res);
            res *= temp;
            res /= r(0);
            res /= r(0);
            res *= 2.0;

            break;
        }

        case 206:
        {
            res = 0.0;

            if ( (double) real(xyprod) > BADZEROTOL )
            {
                res = 1.0;
                res /= r(0);
                res /= r(0);
            }

            else
            {
                res = r(1);
                res /= r(0);
                res /= r(0);
            }

            break;
        }

        case 300:
        {
            res = diffis;
            OP_sqrt(res);
            OP_einv(res);
            res *= 0.5;
            res /= r(0);
            res /= r(0);

            break;
        }

        case 503:
        case 504:
        case 553:
        case 554:
        case 703:
        case 704:
        case 753:
        case 754:
        {
            res = 0.0;
            break;
        }

        case 402: 
        case 452: 
        case 602: 
        case 652: 
        {
            // K = 2/(1+exp(-r0*min_k(x_k-y_k))) - r1
            //   = 2/(1+exp(-r0*z)) - r1
            //
            // dK/dz =  2/(1+exp(-r0*z)) - r1

            res = 0.0;

            if ( xx.nindsize() )
            {
                double temp = 0;
                double minval = 0;

                for ( i = 0 ; i < xx.nindsize() ; ++i )
                {
                    temp  = (double) xx.direcref(i);
                    temp -= (double) yy.direcref(i);

                    if ( (typeis/10)%10 == 5 )
                    {
                        temp = -temp;
                    }

                    if ( !i || ( temp < minval ) )
                    {
                        minval = temp;
                        minmaxres = xx.ind(i);
                    }
                }

                gentype tempb(minval);

                tempb *- r(0);
                tempb.negate();
                OP_exp(tempb);

                res  = tempb;
                res += 1.0;
                res *= res;
                OP_einv(res);
                res *= 2.0;
                res *= r(0);
                res *= tempb;
            }

            if ( typeis/100 == 4 )
            {
                res *= 0.5;
            }

            break;
        }

        case 403: 
        case 453: 
        case 603: 
        case 653: 
        {
            res = 0.0;

            if ( xx.nindsize() )
            {
                double temp = 0;
                double minval = 0;

                for ( i = 0 ; i < xx.nindsize() ; ++i )
                {
                    temp  = (double) xx.direcref(i);
                    temp -= (double) yy.direcref(i);

                    if ( (typeis/10)%10 == 5 )
                    {
                        temp = -temp;
                    }

                    if ( !i || ( temp < minval ) )
                    {
                        minval = temp;
                        minmaxres = xx.ind(i);
                    }
                }
            }

            break;
        }

        case 404: 
        case 454: 
        case 604: 
        case 654: 
        {
            res = 1.0;

            if ( xx.nindsize() )
            {
                double temp = 0;
                double maxval = 0;

                for ( i = 0 ; i < xx.nindsize() ; ++i )
                {
                    temp  = (double) xx.direcref(i);
                    temp -= (double) yy.direcref(i);

                    if ( (typeis/10)%10 == 5 )
                    {
                        temp = -temp;
                    }

                    if ( !i || ( temp > maxval ) )
                    {
                        maxval = temp;
                        minmaxres = xx.ind(i);
                    }
                }
            }

            break;
        }

        case 800: case 801: case 802: case 803: case 804: case 805: case 806: case 807: case 808: case 809:
        case 810: case 811: case 812: case 813: case 814: case 815: case 816: case 817: case 818: case 819:
        case 820: case 821: case 822: case 823: case 824: case 825: case 826: case 827: case 828: case 829:
        case 830: case 831: case 832: case 833: case 834: case 835: case 836: case 837: case 838: case 839:
        case 840: case 841: case 842: case 843: case 844: case 845: case 846: case 847: case 848: case 849:
        case 850: case 851: case 852: case 853: case 854: case 855: case 856: case 857: case 858: case 859:
        case 860: case 861: case 862: case 863: case 864: case 865: case 866: case 867: case 868: case 869:
        case 870: case 871: case 872: case 873: case 874: case 875: case 876: case 877: case 878: case 879:
        case 880: case 881: case 882: case 883: case 884: case 885: case 886: case 887: case 888: case 889:
        case 890: case 891: case 892: case 893: case 894: case 895: case 896: case 897: case 898: case 899:
        {
            kernel8xx(q,res,minmaxind,cType(q),txyprod,tyxprod,diffis,x,xinfo,ii,xdim,m,32,mlid);

            break;
        }

        default:
        {
            NiceThrow("Kernel does not have defined derivative.\n");
            break;
        }
    }

    minmaxind = combineminmaxind(minmaxind,minmaxres);

    return;
}






























































// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
































// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================


















































// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================

//phantomxyz
void MercerKernel::kernel8xx(int q, gentype &res, int &minmaxind, int typeis,
                             const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                             int xdim, int resmode, int mlid) const
{
    {
        (*(getAltCall(q,mlid))).K0xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xdim,0,resmode,mlid);
    }

    return;
}

void MercerKernel::kernel8xx(int q, gentype &res, int &minmaxind, int typeis,
                             const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                             const SparseVector<gentype> &xa,
                             const vecInfo &xainfo, 
                             int ia, 
                             int xdim, int resmode, int mlid) const
{
    {
        (*(getAltCall(q,mlid))).K1xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xainfo,ia,xdim,0,resmode,mlid);
    }

    return;
}

void MercerKernel::kernel8xx(int q, gentype &res, int &minmaxind, int typeis,
                             const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                             const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                             const vecInfo &xainfo, const vecInfo &xbinfo,
                             int ia, int ib,
                             int xdim, int resmode, int mlid) const
{
    {
        gentype dummy;

        (*(getAltCall(q,mlid))).K2xfer(dummy,dummy,res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xainfo,xbinfo,ia,ib,xdim,0,resmode,mlid);
    }

    return;
}

void MercerKernel::kernel8xx(int q, gentype &res, int &minmaxind, int typeis,
                             const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                             const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, 
                             const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, 
                             int ia, int ib, int ic, 
                             int xdim, int resmode, int mlid) const
{
    {
        (*(getAltCall(q,mlid))).K3xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xc,xainfo,xbinfo,xcinfo,ia,ib,ic,xdim,0,resmode,mlid);
    }

    return;
}

void MercerKernel::kernel8xx(int q, gentype &res, int &minmaxind, int typeis,
                             const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                             const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                             const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
                             int ia, int ib, int ic, int id,
                             int xdim, int resmode, int mlid) const
{
    {
        (*(getAltCall(q,mlid))).K4xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,ia,ib,ic,id,xdim,0,resmode,mlid);
    }

    return;
}

void MercerKernel::kernel8xx(int q, gentype &res, int &minmaxind, int typeis,
                             const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                             Vector<const SparseVector<gentype> *> &x,
                             Vector<const vecInfo *> &xinfo,
                             Vector<int> &i,
                             int xdim, int m, int resmode, int mlid) const
{
    if ( m == 0 )
    {
        (*(getAltCall(q,mlid))).K0xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xdim,0,resmode,mlid);
    }

    else if ( m == 1 )
    {
        (*(getAltCall(q,mlid))).K1xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,*(x(0)),*(xinfo(0)),i(0),xdim,0,resmode,mlid);
    }

    else if ( m == 2 )
    {
        gentype dummy;

        (*(getAltCall(q,mlid))).K2xfer(dummy,dummy,res,minmaxind,typeis,xyprod,yxprod,diffis,*(x(0)),*(x(1)),*(xinfo(0)),*(xinfo(1)),i(0),i(1),xdim,0,resmode,mlid);
    }

    else if ( m == 3 )
    {
        (*(getAltCall(q,mlid))).K3xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,*(x(0)),*(x(1)),*(x(2)),*(xinfo(0)),*(xinfo(1)),*(xinfo(2)),i(0),i(1),i(2),xdim,0,resmode,mlid);
    }

    else if ( m == 4 )
    {
        (*(getAltCall(q,mlid))).K4xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,*(x(0)),*(x(1)),*(x(2)),*(x(3)),*(xinfo(0)),*(xinfo(1)),*(xinfo(2)),*(xinfo(3)),i(0),i(1),i(2),i(3),xdim,0,resmode,mlid);
    }

    else
    {
        (*(getAltCall(q,mlid))).Kmxfer(res,minmaxind,typeis,xyprod,yxprod,diffis,x,xinfo,i,xdim,m,0,resmode,mlid);
    }

    return;
}




void MercerKernel::kernel8xx(int q, double &res, int &minmaxind, int typeis,
                             double xyprod, double yxprod, double diffis,
                             int xdim, int resmode, int mlid) const
{
    {
        (*(getAltCall(q,mlid))).K0xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xdim,0,resmode,mlid);
    }

    return;
}

void MercerKernel::kernel8xx(int q, double &res, int &minmaxind, int typeis,
                             double xyprod, double yxprod, double diffis,
                             const SparseVector<gentype> &xa, 
                             const vecInfo &xainfo, 
                             int ia, 
                             int xdim, int resmode, int mlid) const
{
    {
        (*(getAltCall(q,mlid))).K1xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xainfo,ia,xdim,0,resmode,mlid);
    }

    return;
}

void MercerKernel::kernel8xx(int q, double &res, int &minmaxind, int typeis,
                             double xyprod, double yxprod, double diffis,
                             const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                             const vecInfo &xainfo, const vecInfo &xbinfo,
                             int ia, int ib,
                             int xdim, int resmode, int mlid) const
{
    {
        double dummy = 0;

        (*(getAltCall(q,mlid))).K2xfer(dummy,dummy,res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xainfo,xbinfo,ia,ib,xdim,0,resmode,mlid);
    }

    return;
}

void MercerKernel::kernel8xx(int q, double &res, int &minmaxind, int typeis,
                             double xyprod, double yxprod, double diffis,
                             const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, 
                             const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, 
                             int ia, int ib, int ic, 
                             int xdim, int resmode, int mlid) const
{
    {
        (*(getAltCall(q,mlid))).K3xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xc,xainfo,xbinfo,xcinfo,ia,ib,ic,xdim,0,resmode,mlid);
    }

    return;
}

void MercerKernel::kernel8xx(int q, double &res, int &minmaxind, int typeis,
                             double xyprod, double yxprod, double diffis,
                             const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                             const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
                             int ia, int ib, int ic, int id,
                             int xdim, int resmode, int mlid) const
{
    {
        (*(getAltCall(q,mlid))).K4xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,ia,ib,ic,id,xdim,0,resmode,mlid);
    }

    return;
}

void MercerKernel::kernel8xx(int q, double &res, int &minmaxind, int typeis,
                             double xyprod, double yxprod, double diffis,
                             Vector<const SparseVector<gentype> *> &x,
                             Vector<const vecInfo *> &xinfo,
                             Vector<int> &i,
                             int xdim, int m, int resmode, int mlid) const
{
    if ( m == 0 )
    {
        (*(getAltCall(q,mlid))).K0xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xdim,0,resmode,mlid);
    }

    else if ( m == 1 )
    {
        (*(getAltCall(q,mlid))).K1xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,*(x(0)),*(xinfo(0)),i(0),xdim,0,resmode,mlid);
    }

    else if ( m == 2 )
    {
        double dummy = 0;

        (*(getAltCall(q,mlid))).K2xfer(dummy,dummy,res,minmaxind,typeis,xyprod,yxprod,diffis,*(x(0)),*(x(1)),*(xinfo(0)),*(xinfo(1)),i(0),i(1),xdim,0,resmode,mlid);
    }

    else if ( m == 3 )
    {
        (*(getAltCall(q,mlid))).K3xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,*(x(0)),*(x(1)),*(x(2)),*(xinfo(0)),*(xinfo(1)),*(xinfo(2)),i(0),i(1),i(2),xdim,0,resmode,mlid);
    }

    else if ( m == 4 )
    {
        (*(getAltCall(q,mlid))).K4xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,*(x(0)),*(x(1)),*(x(2)),*(x(3)),*(xinfo(0)),*(xinfo(1)),*(xinfo(2)),*(xinfo(3)),i(0),i(1),i(2),i(3),xdim,0,resmode,mlid);
    }

    else
    {
        (*(getAltCall(q,mlid))).Kmxfer(res,minmaxind,typeis,xyprod,yxprod,diffis,x,xinfo,i,xdim,m,0,resmode,mlid);
    }

    return;
}



void MercerKernel::dkernel8xx(int q, double &dxyprod, double &ddiffis, double &res, int &minmaxind, int typeis,
                             double xyprod, double yxprod, double diffis,
                             const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                             const vecInfo &xainfo, const vecInfo &xbinfo,
                             int ia, int ib,
                             int xdim, int resmode, int mlid) const
{
    (*(getAltCall(q,mlid))).K2xfer(dxyprod,ddiffis,res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xainfo,xbinfo,ia,ib,xdim,0,resmode,mlid);

    return;
}
void MercerKernel::dkernel8xx(int q, gentype &dxyprod, gentype &ddiffis, gentype &res, int &minmaxind, int typeis,
                             const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                             const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                             const vecInfo &xainfo, const vecInfo &xbinfo,
                             int ia, int ib,
                             int xdim, int resmode, int mlid) const
{
    (*(getAltCall(q,mlid))).K2xfer(dxyprod,ddiffis,res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xainfo,xbinfo,ia,ib,xdim,0,resmode,mlid);

    return;
}



























































































// ----------------------------------------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------------------------------------------------
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================
// ========================================================================


// Stream operators

std::ostream &MercerKernel::printstream(std::ostream &output, int dep) const
{
    (void) dep;

    const MercerKernel &src = *this;

    gentype tempsampdist("[ ]");

    if ( src.isVeryTrivialKernel() )
    {
        output << "K" << src.cType() << "(" << src.cWeight() << ";" << src.kinf(0) << ";";

        int i;

        for ( i = 0 ; i < src.cRealConstants().size() ; ++i )
        {
            if ( i < src.cRealConstants().size()-1 )
            {
                output << src.cRealConstants()(i) << ",";
            }

            else
            {
                output << src.cRealConstants()(i);
            }
        }

        output << ";";

        for ( i = 0 ; i < src.cIntConstants().size() ; ++i )
        {
            if ( i < src.cIntConstants().size()-1 )
            {
                output << src.cIntConstants()(i) << ",";
            }

            else
            {
                output << src.cIntConstants()(i) << ";";
            }
        }

        output << ";";
        output << src.suggestXYcache() << ";" << src.isIPdiffered() << ";" << src.defindKey().size() << ")";
    }

    else if ( src.isTrivialKernel() && !src.sizeLinConstr() )
    {
        output << "k" << src.cType() << "(" << src.cWeight() << ";" << src.kinf(0) << ";";

        int i;

        for ( i = 0 ; i < src.cRealConstants().size() ; ++i )
        {
            if ( i < src.cRealConstants().size()-1 )
            {
                output << src.cRealConstants()(i) << ",";
            }

            else
            {
                output << src.cRealConstants()(i);
            }
        }

        output << ";";

        for ( i = 0 ; i < src.cIntConstants().size() ; ++i )
        {
            if ( i < src.cIntConstants().size()-1 )
            {
                output << src.cIntConstants()(i) << ",";
            }

            else
            {
                output << src.cIntConstants()(i) << ";";
            }
        }

        output << ";";
        output << src.suggestXYcache() << ";" << src.isIPdiffered() << ";";
        printoneline(output,src.defindKey());
        output << ")";
    }

    else
    {
        output << "Presumed x index key:  " << src.xdefindKey          << "\n";
        output << "Product:               " << src.isprod              << "\n";
        output << "Indexed:               " << src.isind               << "\n";
        output << "Full Norm:             " << src.isfullnorm          << "\n";
        output << "Symmetrised setwise:   " << src.issymmset           << "\n";
        output << "Left Plain:            " << src.leftplain           << "\n";
        output << "Right Plain:           " << src.rightplain          << "\n";
        output << "Alternative Diff:      " << src.isdiffalt           << "\n";
        output << "Shifting/scaling:      " << src.isshift             << "\n";
        output << "Indexing:              " << src.dIndexes            << "\n";
        output << "Shift factor:          " << src.dShift              << "\n";
        output << "Scale factor:          " << src.dScale              << "\n";
        output << "Suggest xy cache:      " << src.xsuggestXYcache     << "\n";
        output << "IP differed:           " << src.xisIPdiffered       << "\n";
        output << "Number of samples:     " << src.xnumsamples         << "\n";
        output << "Sample indices:        " << src.xindsub             << "\n";
        output << "Sample distribution:   " << src.xsampdist           << "\n";
        output << "Dense integr z point:  " << src.xdenseZeroPoint     << "\n";

        output << "Kernel type:           " << src.dtype               << "\n";
        output << "Kernel flags:          " << src.kernflags           << "\n";
        output << "Normalisation:         " << src.isnorm              << "\n";
        output << "Chained:               " << src.ischain             << "\n";
        output << "Split:                 " << src.issplit             << "\n";
        output << "Multiplicative Split:  " << src.mulsplit            << "\n";
        output << "Magnitude Terms:       " << src.ismagterm           << "\n";
        output << "Rank Type Terms:       " << src.xranktype           << "\n";
        output << "Real constants:        " << src.dRealConstants      << "\n";
        output << "Integer constants:     " << src.dIntConstants       << "\n";
        output << "Real constants LB:     " << src.dRealConstantsLB    << "\n";
        output << "Integer constants LB:  " << src.dIntConstantsLB     << "\n";
        output << "Real constants UB:     " << src.dRealConstantsUB    << "\n";
        output << "Integer constants UB:  " << src.dIntConstantsUB     << "\n";
        output << "Real overwrites:       " << src.dRealOverwrite      << "\n";
        output << "Integer overwrites:    " << src.dIntOverwrite       << "\n";
        output << "Alt callback:          " << src.altcallback         << "\n";
        output << "Random Features:       " << src.randFeats           << "\n";
        output << "Random Features angle: " << src.randFeatAngle       << "\n";
        output << "RFF Re part only:      " << src.randFeatReOnly      << "\n\n";
        output << "RFF Re part only:      " << src.randFeatNoAngle     << "\n\n";

        output << "Linear constraint gradient order:        " << src.linGradOrd   << "\n";
        output << "Linear constraint gradient matrices:     " << src.linGradScal  << "\n";
        output << "Linear constraint gradient matrices tsp: " << src.linGradScalTsp  << "\n";
        output << "Linear constraint has constants:         " << src.haslinconstr << "\n\n";

        output << "Enable <xy> retrieval: " << src.enchurn         << "\n";
    }

    return output;
}

std::istream &MercerKernel::inputstream(std::istream &input)
{
    MercerKernel &dest = *this;

    wait_dummy dummy;

    int i;
    char tt;

    while ( isspace(input.peek()) )
    {
        input.get(tt);
    }

    input.get(tt);

    if ( ( tt == 'K' ) || ( tt == 'k' ) )
    {
        char ttt = tt;

        gentype tempsampdist("[ ]");

        dest.resize(1);

        dest.isprod              = 0;
        dest.isind               = 0;
        dest.isfullnorm          = 0;
        dest.issymmset           = 0;
        dest.leftplain           = 0;
        dest.rightplain          = 0;
        dest.isdiffalt           = 1;
        dest.isshift             = 0;
        dest.dIndexes            = 0;
        dest.dShift.zero();
        dest.dScale.zero();
        dest.xnumsamples         = DEFAULT_NUMKERNSAMP;
        dest.xsampdist           = tempsampdist;
        dest.xdenseZeroPoint     = -1.0;

        dest.xindsub.zero();

        dest.isnorm      = 0;
        dest.ischain     = 0;
        dest.issplit     = 0;
        dest.mulsplit    = 0;
        dest.ismagterm   = 0;
        dest.xranktype   = 0;
        dest.enchurn     = 0;
        dest.altcallback = dest.MLid();

        dest.dRealOverwrite("&",0).zero();
        dest.dIntOverwrite("&",0).zero();

        std::stringstream sbuff;

        dest.dRealConstants("&",0).resize(1);
        dest.dIntConstants("&",0).resize(1);
        dest.dRealConstantsLB("&",0).resize(1);
        dest.dIntConstantsLB("&",0).resize(1);
        dest.dRealConstantsUB("&",0).resize(1);
        dest.dIntConstantsUB("&",0).resize(1);

        dest.dRealConstantsLB("&",0)("&",0).makeNull();
        dest.dRealConstantsUB("&",0)("&",0).makeNull();

        dest.dIntConstantsLB("&",0)("&",0) = 0;
        dest.dIntConstantsLB("&",0)("&",0) = 0;

        while ( input.peek() != '(' )
        {
            input.get(tt);
            sbuff << tt;
        }

        sbuff >> dest.dtype("&",0);

        input.get(tt);

        while ( input.peek() != ';' )
        {
            input.get(tt);
            sbuff << tt;
        }

        sbuff >> dest.dRealConstants("&",0)("&",0);

        input.get(tt);

        while ( input.peek() != ';' )
        {
            input.get(tt);
            sbuff << tt;
        }

        sbuff >> dest.kernflags("&",0);

        input.get(tt);

        i = 0;

        while ( input.peek() != ';' )
        {
            ++i;

            while ( ( input.peek() != ',' ) && ( input.peek() != ';' ) )
            {
                input.get(tt);
                sbuff << tt;
            }

            dest.dRealConstants("&",0).add(i);
            dest.dRealConstantsLB("&",0).add(i);
            dest.dRealConstantsUB("&",0).add(i);

            sbuff >> dest.dRealConstants("&",0)("&",i);

            dest.dRealConstantsLB("&",0)("&",i).makeNull();
            dest.dRealConstantsUB("&",0)("&",i).makeNull();

            if ( input.peek() == ',' )
            {
                input.get(tt);
            }
        }

        input.get(tt);

        i = 0;

        while ( input.peek() != ';' )
        {
            ++i;

            while ( ( input.peek() != ',' ) && ( input.peek() != ';' ) )
            {
                input.get(tt);
                sbuff << tt;
            }

            dest.dIntConstants("&",0).add(i);
            dest.dIntConstantsLB("&",0).add(i);
            dest.dIntConstantsUB("&",0).add(i);

            sbuff >> dest.dIntConstants("&",0)("&",i);

            dest.dIntConstantsLB("&",0)("&",i) = 0;
            dest.dIntConstantsUB("&",0)("&",i) = 0;

            if ( input.peek() == ',' )
            {
                input.get(tt);
            }
        }

        input.get(tt);

        while ( input.peek() != ';' )
        {
            input.get(tt);
            sbuff << tt;
        }

        sbuff >> dest.xsuggestXYcache;

        input.get(tt);

        while ( input.peek() != ';' )
        {
            input.get(tt);
            sbuff << tt;
        }

        sbuff >> dest.xisIPdiffered;

        input.get(tt);

        while ( input.peek() != ')' )
        {
            input.get(tt);
            sbuff << tt;
        }

        if ( ttt == 'K' )
        {
            int xbasexdim = 0;

            sbuff >> xbasexdim;

            retVector<int> tmpva;

            dest.xdefindKey = oneintvec(xbasexdim,tmpva);

            input.get(tt);
        }

        else
        {
            sbuff >> dest.xdefindKey;

            input.get(tt);
        }
    }

    else
    {
        input >> dummy; input >> dest.xdefindKey;
        input >> dummy; input >> dest.isprod;
        input >> dummy; input >> dest.isind;
        input >> dummy; input >> dest.isfullnorm;
        input >> dummy; input >> dest.issymmset;
        input >> dummy; input >> dest.leftplain;
        input >> dummy; input >> dest.rightplain;
        input >> dummy; input >> dest.isdiffalt;
        input >> dummy; input >> dest.isshift;
        input >> dummy; input >> dest.dIndexes;
        input >> dummy; input >> dest.dShift;
        input >> dummy; input >> dest.dScale;
        input >> dummy; input >> dest.xsuggestXYcache;
        input >> dummy; input >> dest.xisIPdiffered;
        input >> dummy; input >> dest.xnumsamples;
        input >> dummy; input >> dest.xindsub;
        input >> dummy; input >> dest.xsampdist;
        input >> dummy; input >> dest.xdenseZeroPoint;

        input >> dummy; input >> dest.dtype;
        input >> dummy; input >> dest.kernflags;
        input >> dummy; input >> dest.isnorm;
        input >> dummy; input >> dest.ischain;
        input >> dummy; input >> dest.issplit;
        input >> dummy; input >> dest.mulsplit;
        input >> dummy; input >> dest.ismagterm;
        input >> dummy; input >> dest.xranktype;
        input >> dummy; input >> dest.dRealConstants;
        input >> dummy; input >> dest.dIntConstants;
        input >> dummy; input >> dest.dRealConstantsLB;
        input >> dummy; input >> dest.dIntConstantsLB;
        input >> dummy; input >> dest.dRealConstantsUB;
        input >> dummy; input >> dest.dIntConstantsUB;
        input >> dummy; input >> dest.dRealOverwrite;
        input >> dummy; input >> dest.dIntOverwrite;
        input >> dummy; input >> dest.altcallback;
        input >> dummy; input >> dest.randFeats;
        input >> dummy; input >> dest.randFeatAngle;
        input >> dummy; input >> dest.randFeatReOnly;
        input >> dummy; input >> dest.randFeatNoAngle;

        input >> dummy; input >> dest.linGradOrd;
        input >> dummy; input >> dest.linGradScal;
        input >> dummy; input >> dest.linGradScalTsp;
        input >> dummy; input >> dest.haslinconstr;

        input >> dummy; input >> dest.enchurn;
    }

    retVector<int> tmp;

    dest.xproddepth    = 4; // default back, only matters for optimisation
    dest.xnumSplits    = dest.calcnumSplits();
    dest.xnumMulSplits = dest.calcnumMulSplits();

    dest.fixcombinedOverwriteSrc();
    dest.fixShiftProd();

    dest.xisfast       = -1;
    dest.xneedsInner   = -1;
    dest.xneedsInnerm2 = -1;
    dest.xneedsDiff    = -1;
    dest.xneedsNorm    = -1;

    return input;
}

std::ostream &operator<<(std::ostream &output, const vecInfoBase &src)
{
    output << "2-product:            " << src.xhalfmprod << "\n";
//    output << "Mean:                 " << src.xmean      << "\n";
//    output << "Median:               " << src.xmedian    << "\n";
//    output << "Squared mean:         " << src.xsqmean    << "\n";
//    output << "Variance:             " << src.xvari      << "\n";
//    output << "Standard deviation:   " << src.xstdev     << "\n";
//    output << "Maximum:              " << src.xmax       << "\n";
//    output << "Minimum:              " << src.xmin       << "\n";
//    output << "Range:                " << src.xrange     << "\n";
//    output << "Absolute maximum:     " << src.xmaxabs    << "\n";
    output << "Upsize:               " << src.xusize     << "\n";
    output << "Has been initialised: " << src.hasbeenset << "\n";

    return output;
}

std::istream &operator>>(std::istream &input, vecInfoBase &dest)
{
    wait_dummy dummy;

    input >> dummy; input >> dest.xhalfmprod;
//    input >> dummy; input >> dest.xmean;
//    input >> dummy; input >> dest.xmedian;
//    input >> dummy; input >> dest.xsqmean;
//    input >> dummy; input >> dest.xvari;
//    input >> dummy; input >> dest.xstdev;
//    input >> dummy; input >> dest.xmax;
//    input >> dummy; input >> dest.xmin;
//    input >> dummy; input >> dest.xrange;
//    input >> dummy; input >> dest.xmaxabs;
    input >> dummy; input >> dest.xusize;
    input >> dummy; input >> dest.hasbeenset;

    return input;
}

std::ostream &operator<<(std::ostream &output, const vecInfo &src)
{
    int z = 0;

    output << "Info near: " << (*((src.content)(z))) << "\n";
    output << "Info far:  " << (*((src.content)(1))) << "\n";

    return output;
}

std::istream &operator>>(std::istream &input, vecInfo &dest)
{
    int z = 0;

    wait_dummy dummy;

    if ( !dest.isloc )
    {
        MEMNEW((dest.content)("&",z),SparseVector<vecInfoBase>);
        MEMNEW((dest.content)("&",1),SparseVector<vecInfoBase>);

        (*((dest.content)("&",z)))("&",z);
        (*((dest.content)("&",1)))("&",z);
    }

    dest.isloc = 1;

    dest.minind = 0;
    dest.majind = 1;

    dest.usize_overwrite = 0;

    input >> dummy; input >> (*((dest.content)(z)));
    input >> dummy; input >> (*((dest.content)(1)));

    return input;
}

int operator==(const MercerKernel &leftop, const MercerKernel &rightop)
{
    if ( !( leftop.isprod               == rightop.isprod               ) ) { return 0; }
    if ( !( leftop.isind                == rightop.isind                ) ) { return 0; }
    if ( !( leftop.leftplain            == rightop.leftplain            ) ) { return 0; }
    if ( !( leftop.rightplain           == rightop.rightplain           ) ) { return 0; }
    if ( !( leftop.isshift              == rightop.isshift              ) ) { return 0; }
    if ( !( leftop.dtype                == rightop.dtype                ) ) { return 0; }
    if ( !( leftop.kernflags            == rightop.kernflags            ) ) { return 0; }
    if ( !( leftop.isnorm               == rightop.isnorm               ) ) { return 0; }
    if ( !( leftop.isdiffalt            == rightop.isdiffalt            ) ) { return 0; }
    if ( !( leftop.ischain              == rightop.ischain              ) ) { return 0; }
    if ( !( leftop.issplit              == rightop.issplit              ) ) { return 0; }
    if ( !( leftop.mulsplit             == rightop.mulsplit             ) ) { return 0; }
    if ( !( leftop.ismagterm            == rightop.ismagterm            ) ) { return 0; }
    if ( !( leftop.xranktype            == rightop.xranktype            ) ) { return 0; }
    if ( !( leftop.xnumSplits           == rightop.xnumSplits           ) ) { return 0; }
    if ( !( leftop.xnumMulSplits        == rightop.xnumMulSplits        ) ) { return 0; }
//    if ( !( leftop.altcallback          == rightop.altcallback          ) ) { return 0; }
    if ( !( leftop.dIndexes             == rightop.dIndexes             ) ) { return 0; }
    if ( !( leftop.dShift               == rightop.dShift               ) ) { return 0; }
    if ( !( leftop.dScale               == rightop.dScale               ) ) { return 0; }
    if ( !( leftop.dShiftProd           == rightop.dShiftProd           ) ) { return 0; }
    if ( !( leftop.dShiftProdNoConj     == rightop.dShiftProdNoConj     ) ) { return 0; }
    if ( !( leftop.dShiftProdRevConj    == rightop.dShiftProdRevConj    ) ) { return 0; }
    if ( !( leftop.dRealConstants       == rightop.dRealConstants       ) ) { return 0; }
    if ( !( leftop.dIntConstants        == rightop.dIntConstants        ) ) { return 0; }
//    if ( !( leftop.dRealConstantsLB     == rightop.dRealConstantsLB     ) ) { return 0; }
//    if ( !( leftop.dIntConstantsLB      == rightop.dIntConstantsLB      ) ) { return 0; }
//    if ( !( leftop.dRealConstantsUB     == rightop.dRealConstantsUB     ) ) { return 0; }
//    if ( !( leftop.dIntConstantsUB      == rightop.dIntConstantsUB      ) ) { return 0; }
    if ( !( leftop.dRealOverwrite       == rightop.dRealOverwrite       ) ) { return 0; }
    if ( !( leftop.dIntOverwrite        == rightop.dIntOverwrite        ) ) { return 0; }
    if ( !( leftop.combinedOverwriteSrc == rightop.combinedOverwriteSrc ) ) { return 0; }
    if ( !( leftop.linGradOrd           == rightop.linGradOrd           ) ) { return 0; }
    if ( !( leftop.linGradScal          == rightop.linGradScal          ) ) { return 0; }

    return 1;
}

void MercerKernel::getOneProd(gentype &res,
                              const SparseVector<gentype> &x,
                              int inding, int scaling, int xconsist, int assumreal) const
{
    if ( ( x.altcontent || x.altcontentsp ) && xconsist )
    {
        // In this case the xconsist shortcut slows us down, so don't use it!

        xconsist = 0;
    }

    if ( !x.isnofaroffindpresent() && xconsist )
    {
        // faroff content is not compatible with xconst shortcut

        xconsist = 0;
    }

    if ( !xconsist )
    {
        if ( assumreal && !inding && !scaling )
        {
            oneProductAssumeReal(res.force_double(),x);
        }

        else
        {
                 if ( ( inding == 0 ) && ( scaling == 0 ) ) {        oneProduct      (res,         x       ); }
            else if ( ( inding == 0 ) && ( scaling == 3 ) ) {        oneProductScaled(res,         x,dScale); }
            else if ( ( inding == 1 ) && ( scaling == 0 ) ) { indexedoneProduct      (res,dIndexes,x       ); }
            else if ( ( inding == 1 ) && ( scaling == 3 ) ) { indexedoneProductScaled(res,dIndexes,x,dScale); }

            else
            {
               NiceThrow("Unknown one product type.");
            }
        }
    }

    else
    {
        retVector<gentype> tmpvx;

        const Vector<gentype> &xx = x(x.ind(),tmpvx);

        if ( assumreal && !inding && !scaling )
        {
            oneProductAssumeReal(res.force_double(),xx);
        }

        else
        {
            retVector<gentype> tmpva;

                 if ( ( inding == 0 ) && ( scaling == 0 ) ) {        oneProduct      (res,         xx                           ); }
            else if ( ( inding == 0 ) && ( scaling == 3 ) ) {        oneProductScaled(res,         xx,dScale(dScale.ind(),tmpva)); }
            else if ( ( inding == 1 ) && ( scaling == 0 ) ) { indexedoneProduct      (res,dIndexes,xx                           ); }
            else if ( ( inding == 1 ) && ( scaling == 3 ) ) { indexedoneProductScaled(res,dIndexes,xx,dScale(dScale.ind(),tmpva)); }

            else
            {
               NiceThrow("Unknown one product type.");
            } 
        }
    }

    return;
}

double MercerKernel::getOneProd(const SparseVector<gentype> &x,
                                int inding, int scaling, int xconsist, int assumreal) const
{
    double res = 0;

    if ( ( x.altcontent || x.altcontentsp ) && xconsist )
    {
        // In this case the xconsist shortcut slows us down, so don't use it!

        xconsist = 0;
    }

    if ( !x.isnofaroffindpresent() && xconsist )
    {
        // faroff content is not compatible with xconst shortcut

        xconsist = 0;
    }

    if ( !xconsist )
    {
        if ( assumreal && !inding && !scaling )
        {
            oneProductAssumeReal(res,x);
        }

        else
        {
            if ( ( inding == 0 ) && ( scaling == 0 ) ) { oneProductAssumeReal(res,x); }

            else
            {
                gentype temp(res);

                getOneProd(temp,x,inding,scaling,xconsist,assumreal);

                res = (double) temp;
            }
        }
    }

    else
    {
        retVector<gentype> tmpvx;

        const Vector<gentype> &xx = x(x.ind(),tmpvx);

        if ( assumreal && !inding && !scaling )
        {
            oneProductAssumeReal(res,xx);
        }

        else
        {
            retVector<gentype> tmpva;

            if ( ( inding == 0 ) && ( scaling == 0 ) ) { oneProductAssumeReal(res,xx); }

            else
            {
                gentype temp(res);

                getOneProd(temp,x,inding,scaling,xconsist,assumreal);

                res = (double) temp;
            }
        }
    }

    return res;
}

void MercerKernel::getTwoProd(gentype &res,
                                const SparseVector<gentype> &x,
                                const SparseVector<gentype> &y,
                                int inding, int conj, int scaling, int xconsist, int assumreal) const
{
//errstream() << "phantomxyzmerceraabb 0\n";
    if ( ( x.altcontent || x.altcontentsp ) && ( y.altcontent || y.altcontentsp ) && xconsist )
    {
        // In this case the xconsist shortcut slows us down, so don't use it!

        xconsist = 0;
    }

    if ( ( !x.isnofaroffindpresent() || !y.isnofaroffindpresent() ) && xconsist )
    {
        // faroff content is not compatible with xconst shortcut

        xconsist = 0;
    }

    if ( assumreal )
    {
        conj = 0;
    }

//errstream() << "phantomxyzmerceraabb 1\n";
    if ( !xconsist )
    {
//errstream() << "phantomxyzmerceraabb 2\n";
        if ( assumreal && !inding && !scaling )
        {
//errstream() << "phantomxyzmerceraabb 2a\n";
            innerProductAssumeReal(res.force_double(),x,y);
        }

        else
        {
                 if ( ( inding == 0 ) && ( conj == 0 ) && ( scaling == 0 ) ) {        twoProduct            (res,         x,y       ); }
            else if ( ( inding == 0 ) && ( conj == 0 ) && ( scaling == 1 ) ) {        twoProductLeftScaled  (res,         x,y,dScale); }
            else if ( ( inding == 0 ) && ( conj == 0 ) && ( scaling == 2 ) ) {        twoProductRightScaled (res,         x,y,dScale); }
            else if ( ( inding == 0 ) && ( conj == 0 ) && ( scaling == 3 ) ) {        twoProductScaled      (res,         x,y,dScale); }
            else if ( ( inding == 0 ) && ( conj == 1 ) && ( scaling == 0 ) ) {        innerProduct                  (res,         x,y       ); }
            else if ( ( inding == 0 ) && ( conj == 1 ) && ( scaling == 1 ) ) {        innerProductLeftScaled        (res,         x,y,dScale); }
            else if ( ( inding == 0 ) && ( conj == 1 ) && ( scaling == 2 ) ) {        innerProductRightScaled       (res,         x,y,dScale); }
            else if ( ( inding == 0 ) && ( conj == 1 ) && ( scaling == 3 ) ) {        innerProductScaled            (res,         x,y,dScale); }
            else if ( ( inding == 0 ) && ( conj == 2 ) && ( scaling == 0 ) ) {        innerProductRevConj           (res,         x,y       ); }
            else if ( ( inding == 0 ) && ( conj == 2 ) && ( scaling == 1 ) ) {        innerProductLeftScaledRevConj (res,         x,y,dScale); }
            else if ( ( inding == 0 ) && ( conj == 2 ) && ( scaling == 2 ) ) {        innerProductRightScaledRevConj(res,         x,y,dScale); }
            else if ( ( inding == 0 ) && ( conj == 2 ) && ( scaling == 3 ) ) {        innerProductScaledRevConj     (res,         x,y,dScale); }
            else if ( ( inding == 1 ) && ( conj == 0 ) && ( scaling == 0 ) ) { indexedtwoProduct            (res,dIndexes,x,y       ); }
            else if ( ( inding == 1 ) && ( conj == 0 ) && ( scaling == 1 ) ) { indexedtwoProductLeftScaled  (res,dIndexes,x,y,dScale); }
            else if ( ( inding == 1 ) && ( conj == 0 ) && ( scaling == 2 ) ) { indexedtwoProductRightScaled (res,dIndexes,x,y,dScale); }
            else if ( ( inding == 1 ) && ( conj == 0 ) && ( scaling == 3 ) ) { indexedtwoProductScaled      (res,dIndexes,x,y,dScale); }
            else if ( ( inding == 1 ) && ( conj == 1 ) && ( scaling == 0 ) ) { indexedinnerProduct                  (res,dIndexes,x,y       ); }
            else if ( ( inding == 1 ) && ( conj == 1 ) && ( scaling == 1 ) ) { indexedinnerProductLeftScaled        (res,dIndexes,x,y,dScale); }
            else if ( ( inding == 1 ) && ( conj == 1 ) && ( scaling == 2 ) ) { indexedinnerProductRightScaled       (res,dIndexes,x,y,dScale); }
            else if ( ( inding == 1 ) && ( conj == 1 ) && ( scaling == 3 ) ) { indexedinnerProductScaled            (res,dIndexes,x,y,dScale); }
            else if ( ( inding == 1 ) && ( conj == 2 ) && ( scaling == 0 ) ) { indexedinnerProductRevConj           (res,dIndexes,x,y       ); }
            else if ( ( inding == 1 ) && ( conj == 2 ) && ( scaling == 1 ) ) { indexedinnerProductLeftScaledRevConj (res,dIndexes,x,y,dScale); }
            else if ( ( inding == 1 ) && ( conj == 2 ) && ( scaling == 2 ) ) { indexedinnerProductRightScaledRevConj(res,dIndexes,x,y,dScale); }
            else if ( ( inding == 1 ) && ( conj == 2 ) && ( scaling == 3 ) ) { indexedinnerProductScaledRevConj     (res,dIndexes,x,y,dScale); }

            else
            {
                NiceThrow("Unknown inner product type.");
            }
        }
//errstream() << "phantomxyzmerceraabb 3\n";
    }

    else
    {
//errstream() << "phantomxyzmerceraabb 4\n";
        retVector<gentype> tmpvx;
        retVector<gentype> tmpvy;

        const Vector<gentype> &xx = x(x.ind(),tmpvx);
        const Vector<gentype> &yy = y(y.ind(),tmpvy);

        if ( assumreal && !inding && !scaling )
        {
            innerProductAssumeReal(res.force_double(),xx,yy);
        }

        else
        {
            retVector<gentype> tmpva;

                 if ( ( inding == 0 ) && ( conj == 0 ) && ( scaling == 0 ) ) {        twoProduct            (res,         xx,yy                           ); }
            else if ( ( inding == 0 ) && ( conj == 0 ) && ( scaling == 1 ) ) {        twoProductLeftScaled  (res,         xx,yy,dScale(dScale.ind(),tmpva)); }
            else if ( ( inding == 0 ) && ( conj == 0 ) && ( scaling == 2 ) ) {        twoProductRightScaled (res,         xx,yy,dScale(dScale.ind(),tmpva)); }
            else if ( ( inding == 0 ) && ( conj == 0 ) && ( scaling == 3 ) ) {        twoProductScaled      (res,         xx,yy,dScale(dScale.ind(),tmpva)); }
            else if ( ( inding == 0 ) && ( conj == 1 ) && ( scaling == 0 ) ) {        innerProduct                  (res,         xx,yy                           ); }
            else if ( ( inding == 0 ) && ( conj == 1 ) && ( scaling == 1 ) ) {        innerProductLeftScaled        (res,         xx,yy,dScale(dScale.ind(),tmpva)); }
            else if ( ( inding == 0 ) && ( conj == 1 ) && ( scaling == 2 ) ) {        innerProductRightScaled       (res,         xx,yy,dScale(dScale.ind(),tmpva)); }
            else if ( ( inding == 0 ) && ( conj == 1 ) && ( scaling == 3 ) ) {        innerProductScaled            (res,         xx,yy,dScale(dScale.ind(),tmpva)); }
            else if ( ( inding == 0 ) && ( conj == 2 ) && ( scaling == 0 ) ) {        innerProductRevConj           (res,         xx,yy                           ); }
            else if ( ( inding == 0 ) && ( conj == 2 ) && ( scaling == 1 ) ) {        innerProductLeftScaledRevConj (res,         xx,yy,dScale(dScale.ind(),tmpva)); }
            else if ( ( inding == 0 ) && ( conj == 2 ) && ( scaling == 2 ) ) {        innerProductRightScaledRevConj(res,         xx,yy,dScale(dScale.ind(),tmpva)); }
            else if ( ( inding == 0 ) && ( conj == 2 ) && ( scaling == 3 ) ) {        innerProductScaledRevConj     (res,         xx,yy,dScale(dScale.ind(),tmpva)); }
            else if ( ( inding == 1 ) && ( conj == 0 ) && ( scaling == 0 ) ) { indexedtwoProduct            (res,dIndexes,xx,yy                           ); }
            else if ( ( inding == 1 ) && ( conj == 0 ) && ( scaling == 1 ) ) { indexedtwoProductLeftScaled  (res,dIndexes,xx,yy,dScale(dScale.ind(),tmpva)); }
            else if ( ( inding == 1 ) && ( conj == 0 ) && ( scaling == 2 ) ) { indexedtwoProductRightScaled (res,dIndexes,xx,yy,dScale(dScale.ind(),tmpva)); }
            else if ( ( inding == 1 ) && ( conj == 0 ) && ( scaling == 3 ) ) { indexedtwoProductScaled      (res,dIndexes,xx,yy,dScale(dScale.ind(),tmpva)); }
            else if ( ( inding == 1 ) && ( conj == 1 ) && ( scaling == 0 ) ) { indexedinnerProduct                  (res,dIndexes,xx,yy                           ); }
            else if ( ( inding == 1 ) && ( conj == 1 ) && ( scaling == 1 ) ) { indexedinnerProductLeftScaled        (res,dIndexes,xx,yy,dScale(dScale.ind(),tmpva)); }
            else if ( ( inding == 1 ) && ( conj == 1 ) && ( scaling == 2 ) ) { indexedinnerProductRightScaled       (res,dIndexes,xx,yy,dScale(dScale.ind(),tmpva)); }
            else if ( ( inding == 1 ) && ( conj == 1 ) && ( scaling == 3 ) ) { indexedinnerProductScaled            (res,dIndexes,xx,yy,dScale(dScale.ind(),tmpva)); }
            else if ( ( inding == 1 ) && ( conj == 2 ) && ( scaling == 0 ) ) { indexedinnerProductRevConj           (res,dIndexes,xx,yy                           ); }
            else if ( ( inding == 1 ) && ( conj == 2 ) && ( scaling == 1 ) ) { indexedinnerProductLeftScaledRevConj (res,dIndexes,xx,yy,dScale(dScale.ind(),tmpva)); }
            else if ( ( inding == 1 ) && ( conj == 2 ) && ( scaling == 2 ) ) { indexedinnerProductRightScaledRevConj(res,dIndexes,xx,yy,dScale(dScale.ind(),tmpva)); }
            else if ( ( inding == 1 ) && ( conj == 2 ) && ( scaling == 3 ) ) { indexedinnerProductScaledRevConj     (res,dIndexes,xx,yy,dScale(dScale.ind(),tmpva)); }

            else
            {
                NiceThrow("Unknown inner product type.");
            }
        }
//errstream() << "phantomxyzmerceraabb 5\n";
    }

//errstream() << "phantomxyzmerceraabb 6\n";
    return;
}

double MercerKernel::getTwoProd(const SparseVector<gentype> &x,
                                const SparseVector<gentype> &y,
                                int inding, int conj, int scaling, int xconsist, int assumreal) const
{
    double res = 0;

//errstream() << "phantomxyzmerceraabb 0\n";
    if ( ( x.altcontent || x.altcontentsp ) && ( y.altcontent || y.altcontentsp ) && xconsist )
    {
        // In this case the xconsist shortcut slows us down, so don't use it!

        xconsist = 0;
    }

    if ( ( !x.isnofaroffindpresent() || !y.isnofaroffindpresent() ) && xconsist )
    {
        // faroff content is not compatible with xconst shortcut

        xconsist = 0;
    }

    if ( assumreal )
    {
        conj = 0;
    }

//errstream() << "phantomxyzmerceraabb 1\n";
    if ( !xconsist )
    {
//errstream() << "phantomxyzmerceraabb 2\n";
        if ( assumreal && !inding && !scaling )
        {
//errstream() << "phantomxyzmerceraabb 2a\n";
            innerProductAssumeReal(res,x,y);
        }

        else
        {
                 if ( ( inding == 0 ) && ( conj == 0 ) && ( scaling == 0 ) ) { innerProductAssumeReal(res,x,y); }
            else if ( ( inding == 0 ) && ( conj == 1 ) && ( scaling == 0 ) ) { innerProductAssumeReal(res,x,y); }
            else if ( ( inding == 0 ) && ( conj == 2 ) && ( scaling == 0 ) ) { innerProductAssumeReal(res,x,y); }

            else
            {
                gentype temp(res);

                getTwoProd(temp,x,y,inding,conj,scaling,xconsist,assumreal);

                res = (double) temp;
            }
        }
//errstream() << "phantomxyzmerceraabb 3\n";
    }

    else
    {
//errstream() << "phantomxyzmerceraabb 4\n";
        retVector<gentype> tmpvx;
        retVector<gentype> tmpvy;

        const Vector<gentype> &xx = x(x.ind(),tmpvx);
        const Vector<gentype> &yy = y(y.ind(),tmpvy);

        if ( assumreal && !inding && !scaling )
        {
            innerProductAssumeReal(res,xx,yy);
        }

        else
        {
            retVector<gentype> tmpva;

                 if ( ( inding == 0 ) && ( conj == 0 ) && ( scaling == 0 ) ) { innerProductAssumeReal(res,xx,yy); }
            else if ( ( inding == 0 ) && ( conj == 1 ) && ( scaling == 0 ) ) { innerProductAssumeReal(res,xx,yy); }
            else if ( ( inding == 0 ) && ( conj == 2 ) && ( scaling == 0 ) ) { innerProductAssumeReal(res,xx,yy); }

            else
            {
                gentype temp(res);

                getTwoProd(temp,x,y,inding,conj,scaling,xconsist,assumreal);

                res = (double) temp;
            }
        }
//errstream() << "phantomxyzmerceraabb 5\n";
    }

//errstream() << "phantomxyzmerceraabb 6\n";
    return res;
}

void MercerKernel::getThreeProd(gentype &res,
                                const SparseVector<gentype> &xa,
                                const SparseVector<gentype> &xb,
                                const SparseVector<gentype> &xc,
                                int inding, int scaling, int xconsist, int assumreal) const
{
    if ( (xa.altcontent) && (xb.altcontent) && (xc.altcontent) && xconsist )
    {
        // In this case the xconsist shortcut slows us down, so don't use it!

        xconsist = 0;
    }

    if ( ( !xa.isnofaroffindpresent() || !xb.isnofaroffindpresent() || !xc.isnofaroffindpresent() ) && xconsist )
    {
        // faroff content is not compatible with xconst shortcut

        xconsist = 0;
    }

    if ( !xconsist )
    {
        if ( assumreal && !inding && !scaling )
        {
            threeProductAssumeReal(res.force_double(),xa,xb,xc);
        }

        else
        {
                 if ( ( inding == 0 ) && ( scaling == 0 ) ) {        threeProduct      (res,         xa,xb,xc       ); }
            else if ( ( inding == 0 ) && ( scaling == 3 ) ) {        threeProductScaled(res,         xa,xb,xc,dScale); }
            else if ( ( inding == 1 ) && ( scaling == 0 ) ) { indexedthreeProduct      (res,dIndexes,xa,xb,xc       ); }
            else if ( ( inding == 1 ) && ( scaling == 3 ) ) { indexedthreeProductScaled(res,dIndexes,xa,xb,xc,dScale); }

            else
            {
                NiceThrow("Unknown three product type.");
            }
        }
    }

    else
    {
        retVector<gentype> tmpvxa;
        retVector<gentype> tmpvxb;
        retVector<gentype> tmpvxc;

        const Vector<gentype> &xxa = xa(xa.ind(),tmpvxa);
        const Vector<gentype> &xxb = xb(xb.ind(),tmpvxb);
        const Vector<gentype> &xxc = xc(xc.ind(),tmpvxc);

        if ( assumreal && !inding && !scaling )
        {
            threeProductAssumeReal(res.force_double(),xxa,xxb,xxc);
        }

        else
        {
            retVector<gentype> tmpva;

                 if ( ( inding == 0 ) && ( scaling == 0 ) ) {        threeProduct      (res,         xxa,xxb,xxc                           ); }
            else if ( ( inding == 0 ) && ( scaling == 3 ) ) {        threeProductScaled(res,         xxa,xxb,xxc,dScale(dScale.ind(),tmpva)); }
            else if ( ( inding == 1 ) && ( scaling == 0 ) ) { indexedthreeProduct      (res,dIndexes,xxa,xxb,xxc                           ); }
            else if ( ( inding == 1 ) && ( scaling == 3 ) ) { indexedthreeProductScaled(res,dIndexes,xxa,xxb,xxc,dScale(dScale.ind(),tmpva)); }

            else
            {
                NiceThrow("Unknown three product type.");
            }
        }
    }

    return;
}

double MercerKernel::getThreeProd(const SparseVector<gentype> &xa,
                                const SparseVector<gentype> &xb,
                                const SparseVector<gentype> &xc,
                                int inding, int scaling, int xconsist, int assumreal) const
{
    double res = 0;

    if ( (xa.altcontent) && (xb.altcontent) && (xc.altcontent) && xconsist )
    {
        // In this case the xconsist shortcut slows us down, so don't use it!

        xconsist = 0;
    }

    if ( ( !xa.isnofaroffindpresent() || !xb.isnofaroffindpresent() || !xc.isnofaroffindpresent() ) && xconsist )
    {
        // faroff content is not compatible with xconst shortcut

        xconsist = 0;
    }

    if ( !xconsist )
    {
        if ( assumreal && !inding && !scaling )
        {
            threeProductAssumeReal(res,xa,xb,xc);
        }

        else
        {
            if ( ( inding == 0 ) && ( scaling == 0 ) ) { threeProductAssumeReal(res,xa,xb,xc); }

            else
            {
                gentype temp(res);

                getThreeProd(temp,xa,xb,xc,inding,scaling,xconsist,assumreal);

                res = (double) temp;
            }
        }
    }

    else
    {
        retVector<gentype> tmpvxa;
        retVector<gentype> tmpvxb;
        retVector<gentype> tmpvxc;

        const Vector<gentype> &xxa = xa(xa.ind(),tmpvxa);
        const Vector<gentype> &xxb = xb(xb.ind(),tmpvxb);
        const Vector<gentype> &xxc = xc(xc.ind(),tmpvxc);

        if ( assumreal && !inding && !scaling )
        {
            threeProductAssumeReal(res,xxa,xxb,xxc);
        }

        else
        {
            retVector<gentype> tmpva;

            if ( ( inding == 0 ) && ( scaling == 0 ) ) { threeProductAssumeReal(res,xxa,xxb,xxc); }

            else
            {
                gentype temp(res);

                getThreeProd(temp,xa,xb,xc,inding,scaling,xconsist,assumreal);

                res = (double) temp;
            }
        }
    }

    return res;
}

void MercerKernel::getFourProd(gentype &res,
                               const SparseVector<gentype> &xa,
                               const SparseVector<gentype> &xb,
                               const SparseVector<gentype> &xc,
                               const SparseVector<gentype> &xd,
                               int inding, int scaling, int xconsist, int assumreal) const
{
    if ( (xa.altcontent) && (xb.altcontent) && (xc.altcontent) && (xd.altcontent) && xconsist )
    {
        // In this case the xconsist shortcut slows us down, so don't use it!

        xconsist = 0;
    }

    if ( ( !xa.isnofaroffindpresent() || !xb.isnofaroffindpresent() || !xc.isnofaroffindpresent() || !xd.isnofaroffindpresent() ) && xconsist )
    {
        // faroff content is not compatible with xconst shortcut

        xconsist = 0;
    }

    if ( !xconsist )
    {
        if ( assumreal && !inding && !scaling )
        {
            fourProductAssumeReal(res.force_double(),xa,xb,xc,xd);
        }

        else
        {
                 if ( ( inding == 0 ) && ( scaling == 0 ) ) {        fourProduct      (res,         xa,xb,xc,xd       ); }
            else if ( ( inding == 0 ) && ( scaling == 3 ) ) {        fourProductScaled(res,         xa,xb,xc,xd,dScale); }
            else if ( ( inding == 1 ) && ( scaling == 0 ) ) { indexedfourProduct      (res,dIndexes,xa,xb,xc,xd       ); }
            else if ( ( inding == 1 ) && ( scaling == 3 ) ) { indexedfourProductScaled(res,dIndexes,xa,xb,xc,xd,dScale); }

            else
            {
                NiceThrow("Unknown four product type.");
            }
        }
    }

    else
    {
        retVector<gentype> tmpvxa;
        retVector<gentype> tmpvxb;
        retVector<gentype> tmpvxc;
        retVector<gentype> tmpvxd;

        const Vector<gentype> &xxa = xa(xa.ind(),tmpvxa);
        const Vector<gentype> &xxb = xb(xb.ind(),tmpvxb);
        const Vector<gentype> &xxc = xc(xc.ind(),tmpvxc);
        const Vector<gentype> &xxd = xd(xd.ind(),tmpvxd);

        if ( assumreal && !inding && !scaling )
        {
            fourProductAssumeReal(res.force_double(),xxa,xxb,xxc,xxd);
        }

        else
        {
            retVector<gentype> tmpva;

                 if ( ( inding == 0 ) && ( scaling == 0 ) ) {        fourProduct      (res,         xxa,xxb,xxc,xxd                           ); }
            else if ( ( inding == 0 ) && ( scaling == 3 ) ) {        fourProductScaled(res,         xxa,xxb,xxc,xxd,dScale(dScale.ind(),tmpva)); }
            else if ( ( inding == 1 ) && ( scaling == 0 ) ) { indexedfourProduct      (res,dIndexes,xxa,xxb,xxc,xxd                           ); }
            else if ( ( inding == 1 ) && ( scaling == 3 ) ) { indexedfourProductScaled(res,dIndexes,xxa,xxb,xxc,xxd,dScale(dScale.ind(),tmpva)); }

            else
            {
                NiceThrow("Unknown four product type.");
            }
        }
    }

    return;
}

double MercerKernel::getFourProd(const SparseVector<gentype> &xa,
                               const SparseVector<gentype> &xb,
                               const SparseVector<gentype> &xc,
                               const SparseVector<gentype> &xd,
                               int inding, int scaling, int xconsist, int assumreal) const
{
    double res = 0;

    if ( (xa.altcontent) && (xb.altcontent) && (xc.altcontent) && (xd.altcontent) && xconsist )
    {
        // In this case the xconsist shortcut slows us down, so don't use it!

        xconsist = 0;
    }

    if ( ( !xa.isnofaroffindpresent() || !xb.isnofaroffindpresent() || !xc.isnofaroffindpresent() || !xd.isnofaroffindpresent() ) && xconsist )
    {
        // faroff content is not compatible with xconst shortcut

        xconsist = 0;
    }

    if ( !xconsist )
    {
        if ( assumreal && !inding && !scaling )
        {
            fourProductAssumeReal(res,xa,xb,xc,xd);
        }

        else
        {
            if ( ( inding == 0 ) && ( scaling == 0 ) ) { fourProductAssumeReal(res,xa,xb,xc,xd); }

            else
            {
                gentype temp(res);

                getFourProd(temp,xa,xb,xc,xd,inding,scaling,xconsist,assumreal);

                res = (double) temp;
            }
        }
    }

    else
    {
        retVector<gentype> tmpvxa;
        retVector<gentype> tmpvxb;
        retVector<gentype> tmpvxc;
        retVector<gentype> tmpvxd;

        const Vector<gentype> &xxa = xa(xa.ind(),tmpvxa);
        const Vector<gentype> &xxb = xb(xb.ind(),tmpvxb);
        const Vector<gentype> &xxc = xc(xc.ind(),tmpvxc);
        const Vector<gentype> &xxd = xd(xd.ind(),tmpvxd);

        if ( assumreal && !inding && !scaling )
        {
            fourProductAssumeReal(res,xxa,xxb,xxc,xxd);
        }

        else
        {
            retVector<gentype> tmpva;

            if ( ( inding == 0 ) && ( scaling == 0 ) ) { fourProductAssumeReal(res,xxa,xxb,xxc,xxd); }

            else
            {
                gentype temp(res);

                getFourProd(temp,xa,xb,xc,xd,inding,scaling,xconsist,assumreal);

                res = (double) temp;
            }
        }
    }

    return res;
}

//void MercerKernel::getmProd(gentype &res,
//                            const Vector<const SparseVector<gentype> *> &x,
//                            int inding, int scaling, int xconsist, int assumreal) const
void MercerKernel::getmProd(gentype &res,
                            const Vector<const SparseVector<gentype> *> &x,
                            int inding, int scaling, int, int assumreal) const
{
    //xconsist = 0; // Otherwise we need to search through all x, which is too time consuming

    //(void) xconsist;

    if ( assumreal && !inding && !scaling )
    {
        mProductAssumeReal(res.force_double(),x);
    }

    else
    {
             if ( ( inding == 0 ) && ( scaling == 0 ) ) {        mProduct      (res,         x       ); }
        else if ( ( inding == 0 ) && ( scaling == 3 ) ) {        mProductScaled(res,         x,dScale); }
        else if ( ( inding == 1 ) && ( scaling == 0 ) ) { indexedmProduct      (res,dIndexes,x       ); }
        else if ( ( inding == 1 ) && ( scaling == 3 ) ) { indexedmProductScaled(res,dIndexes,x,dScale); }

        else
        {
            NiceThrow("Unknown m product type.");
        }
    }

    return;
}

double MercerKernel::getmProd(const Vector<const SparseVector<gentype> *> &x,
                            int inding, int scaling, int xconsist, int assumreal) const
{
    double res = 0;

    xconsist = 0; // Otherwise we need to search through all x, which is too time consuming

    if ( assumreal && !inding && !scaling )
    {
        mProductAssumeReal(res,x);
    }

    else
    {
        if ( ( inding == 0 ) && ( scaling == 0 ) ) { mProductAssumeReal(res,x); }

        else
        {
            gentype temp(res);

            getmProd(temp,x,inding,scaling,xconsist,assumreal);

            res = (double) temp;
        }
    }

    return res;
}








kernInfo &operator+=(kernInfo &a, const kernInfo &b)
{
    a.usesDiff    |= b.usesDiff;
    a.usesInner   |= b.usesInner;
    a.usesNorm    |= b.usesNorm;
    a.usesVector  |= b.usesVector;
    a.usesMinDiff |= b.usesMinDiff;
    a.usesMaxDiff |= b.usesMaxDiff;

    return a;
}

std::ostream &operator<<(std::ostream &output, const kernInfo &src)
{
    output << ( src.usesDiff    ? "D" : "_" );
    output << ( src.usesInner   ? "I" : "_" );
    output << ( src.usesNorm    ? "M" : "_" );
    output << ( src.usesVector  ? "V" : "_" );
    output << ( src.usesMinDiff ? "L" : "_" );
    output << ( src.usesMaxDiff ? "U" : "_" );

    return output;
}

std::istream &operator>>(std::istream &input, kernInfo &dest)
{
    std::string buffer;

    input >> buffer;

    NiceAssert( buffer.length() == 6 );

    dest.usesDiff    = ( buffer[0] == 'D' ) ? 1 : 0;
    dest.usesInner   = ( buffer[1] == 'I' ) ? 1 : 0;
    dest.usesNorm    = ( buffer[2] == 'M' ) ? 1 : 0;
    dest.usesVector  = ( buffer[3] == 'V' ) ? 1 : 0;
    dest.usesMinDiff = ( buffer[4] == 'L' ) ? 1 : 0;
    dest.usesMaxDiff = ( buffer[5] == 'U' ) ? 1 : 0;

    return input;
}

int operator==(const kernInfo &a, const kernInfo &b)
{
    if ( a.usesDiff    != b.usesDiff    ) { return 0; }
    if ( a.usesInner   != b.usesInner   ) { return 0; }
    if ( a.usesNorm    != b.usesNorm    ) { return 0; }
    if ( a.usesVector  != b.usesVector  ) { return 0; }
    if ( a.usesMinDiff != b.usesMinDiff ) { return 0; }
    if ( a.usesMaxDiff != b.usesMaxDiff ) { return 0; }

    return 1;
}


int MercerKernel::egetparam(int ind, Vector<gentype> &val, const Vector<gentype> &xa, int ia, const Vector<gentype> &xb, int ib) const
{
    int k,res = 0;
    const char *dummy = "";

    NiceAssert( xa.size() == xb.size() );

    val.resize(xa.size());

    for ( k = 0 ; k < xa.size() ; ++k )
    {
        res |= getparam(ind,val("&",k),xa(k),ia,xb(k),ib,dummy);
    }

    return res;
}

int MercerKernel::getparam(int ind, gentype &val, const gentype &xa, int ia, const gentype &xb, int ib, charptr &desc) const
{
    int res = 0;

    desc = "";

    // Function to access parameters via indexing

    NiceAssert( ia >= 0 );
    NiceAssert( ib >= 0 );

    (void) xb;
    (void) ib;

    if ( ia || ib )
    {
        val.force_null();
    }

    else if ( ind < 50 )
    {
        retVector<gentype> tmpva;

        switch ( ind )
        {
            case   0: { val = isFullNorm();           desc = "mercer::isFullNorm"; break; }
            case   1: { val = isProd();               desc = "mercer::isProd"; break; }
            case   2: { val = isIndex();              desc = "mercer::isIndex"; break; }
            case   3: { val = isShiftedScaled();      desc = "mercer::isShiftedScaled"; break; }
            case   4: { val = isLeftPlain();          desc = "mercer::isLeftPlain"; break; }
            case   5: { val = isRightPlain();         desc = "mercer::isRightPlain"; break; }
            case   6: { val = isLeftRightPlain();     desc = "mercer::isLeftRightPlain"; break; }
            case   7: { val = isLeftNormal();         desc = "mercer::isLeftNormal"; break; }
            case   8: { val = isRightNormal();        desc = "mercer::ifRightNormal"; break; }
            case   9: { val = isLeftRightNormal();    desc = "mercer::isLeftRightNormal"; break; }
            case  10: { val = isPartNormal();         desc = "mercer::isPartNormal"; break; }
            case  16: { val = isAltDiff();            desc = "mercer::isAltDiff"; break; }
            case  17: { val = needsmProd();           desc = "mercer::needsmProd"; break; }
            case  18: { val = wantsXYprod();          desc = "mercer::wantsXYprod"; break; }
            case  19: { val = suggestXYcache();       desc = "mercer::suggestXYcache"; break; }
            case  20: { val = isIPdiffered();         desc = "mercer::isIPdiffered"; break; }
            case  22: { val = size();                 desc = "mercer::size"; break; }
            case  23: { val = getSymmetry();          desc = "mercer::getSymmetry"; break; }
            case  24: { val = cIndexes();             desc = "mercer::cIndexes"; break; }
            case  25: { val = cShift()(tmpva);        desc = "mercer::cShift"; break; }
            case  26: { val = cScale()(tmpva);        desc = "mercer::cScale"; break; }
            case  29: { val = churnInner();           desc = "mercer::churnInner"; break; }
            case  30: { val = isKVarianceNZ();        desc = "mercer::isKVarianceNZ"; break; }
            case  31: { val = isShifted();            desc = "mercer::isShifted"; break; }
            case  32: { val = isScaled();             desc = "mercer::isScaled"; break; }

            default:
            {
                val.force_null();
                break;
            }
        }
    }

    else if ( xa.isCastableToIntegerWithoutLoss() && ( (int) xa < size() ) )
    {
        retVector<int> tmpva;

        switch ( ind )
        {
            case  50: { val = cWeight((int) xa);               desc = "mercer::cWeight"; break; }
            case  51: { val = cType((int) xa);                 desc = "mercer::cType"; break; }
            case  52: { val = isNormalised((int) xa);          desc = "mercer::isNormalised"; break; }
            case  54: { val = isChained((int) xa);             desc = "mercer::isChained"; break; }
            case  56: { val = cRealConstants((int) xa);        desc = "mercer::cRealConstants"; break; }
            case  57: { val = cIntConstants((int) xa);         desc = "mercer::cIntConstants"; break; }
            case  58: { val = cRealOverwrite((int) xa)(tmpva); desc = "mercer::cRealOverwrite"; break; }
            case  59: { val = cIntOverwrite((int) xa)(tmpva);  desc = "mercer::cIntOverwrite"; break; }
            case  60: { val = getRealConstZero((int) xa);      desc = "mercer::getRealConstZero"; break; }
            case  61: { val = getIntConstZero((int) xa);       desc = "mercer::getIntConstZero"; break; }
            case  62: { val = isSplit((int) xa);               desc = "mercer::isSplit"; break; }
            case  64: { val = isMagTerm((int) xa);             desc = "mercer::isMagTerm"; break; }
            case  66: { val = isMulSplit((int) xa);            desc = "mercer::isMulSplit"; break; }

            default:
            {
                val.force_null();
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






