
//
// Functional block base class
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
#include "blk_generic.hpp"


int dummygetsetExtVar(gentype &res, const gentype &src, int num);
int dummygetsetExtVar(gentype &res, const gentype &src, int num)
{
    (void) res;
    (void) src;
    (void) num;

    NiceThrow("No external variables in this context!");

    return -1;
}

BLK_Generic::mexcallsyn BLK_Generic::getsetExtVar = dummygetsetExtVar;


std::ostream &BLK_Generic::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Mercer cache size:     " << xmercachesize         << "\n";
    repPrint(output,'>',dep) << "Mercer cache norm:     " << xmercachenorm         << "\n";
    repPrint(output,'>',dep) << "Averaging mode:        " << xmlqmode              << "\n";
    repPrint(output,'>',dep) << "Sigma:                 " << locsigma              << "\n";
    repPrint(output,'>',dep) << "Output function:       " << doutfn                << "\n";
    repPrint(output,'>',dep) << "Is this a sample:      " << xissample             << "\n";
    repPrint(output,'>',dep) << "MEX callback:          " << mexfn                 << "\n";
    repPrint(output,'>',dep) << "MEX callback id:       " << mexfnid               << "\n";
    repPrint(output,'>',dep) << "SYSTEM callback:       " << sysfn                 << "\n";
    repPrint(output,'>',dep) << "SYSTEM x file:         " << xfname                << "\n";
    repPrint(output,'>',dep) << "SYSTEM y file:         " << yfname                << "\n";
    repPrint(output,'>',dep) << "SYSTEM xy file:        " << xyfname               << "\n";
    repPrint(output,'>',dep) << "SYSTEM yx file:        " << yxfname               << "\n";
    repPrint(output,'>',dep) << "SYSTEM r file:         " << rfname                << "\n";
    repPrint(output,'>',dep) << "KB learning rate:      " << KBlr                  << "\n";
    repPrint(output,'>',dep) << "KB min step:           " << KBminstep             << "\n";
    repPrint(output,'>',dep) << "KB max iters:          " << KBmaxiter             << "\n";
    repPrint(output,'>',dep) << "KB ML ids:             " << KBaltMLids            << "\n";
    repPrint(output,'>',dep) << "KB weights:            " << KBMLweight            << "\n";
    repPrint(output,'>',dep) << "KB lambda:             " << KBlambda              << "\n";
    repPrint(output,'>',dep) << "ML weight:             " << xmlqweight            << "\n";
    repPrint(output,'>',dep) << "Bernstein degree:      " << berndeg               << "\n";
    repPrint(output,'>',dep) << "Bernstein index:       " << bernind               << "\n";
    repPrint(output,'>',dep) << "Battery parameters:    " << xbattParam            << "\n";
    repPrint(output,'>',dep) << "Battery t_max:         " << xbatttmax             << "\n";
    repPrint(output,'>',dep) << "Battery I_max:         " << xbattImax             << "\n";
    repPrint(output,'>',dep) << "Battery t_delta:       " << xbatttdelta           << "\n";
    repPrint(output,'>',dep) << "Battery V_start:       " << xbattVstart           << "\n";
    repPrint(output,'>',dep) << "Battery theta_start:   " << xbattthetaStart       << "\n";
    repPrint(output,'>',dep) << "Battery non-parasitic: " << xbattneglectParasitic << "\n";
    repPrint(output,'>',dep) << "Battery fixed-temp:    " << xbattfixedTheta       << "\n";

    ML_Base::printstream(output,dep+1);

    return output;
}

std::istream &BLK_Generic::inputstream(std::istream &input )
{
    wait_dummy dummy;

    input >> dummy; input >> xmercachesize;
    input >> dummy; input >> xmercachenorm;
    input >> dummy; input >> xmlqmode;
    input >> dummy; input >> locsigma;
    input >> dummy; input >> doutfn;
    input >> dummy; input >> xissample;
    input >> dummy; input >> mexfn;
    input >> dummy; input >> mexfnid;
    input >> dummy; input >> sysfn;
    input >> dummy; input >> xfname;
    input >> dummy; input >> yfname;
    input >> dummy; input >> xyfname;
    input >> dummy; input >> yxfname;
    input >> dummy; input >> rfname;
    input >> dummy; input >> KBlr;
    input >> dummy; input >> KBminstep;
    input >> dummy; input >> KBmaxiter;
    input >> dummy; input >> KBaltMLids;
    input >> dummy; input >> KBMLweight;
    input >> dummy; input >> KBlambda;
    input >> dummy; input >> xmlqweight;
    input >> dummy; input >> berndeg;
    input >> dummy; input >> bernind;
    input >> dummy; input >> xbattParam;
    input >> dummy; input >> xbatttmax;
    input >> dummy; input >> xbattImax;
    input >> dummy; input >> xbatttdelta;
    input >> dummy; input >> xbattVstart;
    input >> dummy; input >> xbattthetaStart;
    input >> dummy; input >> xbattneglectParasitic;
    input >> dummy; input >> xbattfixedTheta;

    ML_Base::inputstream(input);

    return input;
}

int gcallbackdummy(gentype &res, const SparseVector<gentype> &x, void *fndata)
{
    (void) res;
    (void) x;
    (void) fndata;

    return 0;
}

void K0callbackdummy(gentype &res, int &minmaxind, int typeis, int xdim, int densetype, int resmode, int mlid, void *fndata)
{
    (void) res;
    (void) minmaxind;
    (void) typeis;
    (void) xdim;
    (void) densetype;
    (void) resmode;
    (void) mlid;
    (void) fndata;

    return;
}

void K1callbackdummy(gentype &res, int &minmaxind, int typeis, const SparseVector<gentype> &xa, const vecInfo &xainfo, int ia, int xdim, int densetype, int resmode, int mlid, void *fndata)
{
    (void) res;
    (void) minmaxind;
    (void) typeis;
    (void) xa;
    (void) xainfo;
    (void) ia;
    (void) xdim;
    (void) densetype;
    (void) resmode;
    (void) mlid;
    (void) fndata;

    return;
}

void K2callbackdummy(gentype &res, int &minmaxind, int typeis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, int ia, int ib,int xdim, int densetype, int resmode, int mlid, void *fndata)
{
    (void) res;
    (void) minmaxind;
    (void) typeis;
    (void) xa;
    (void) xb;
    (void) xainfo;
    (void) xbinfo;
    (void) ia;
    (void) ib;
    (void) xdim;
    (void) densetype;
    (void) resmode;
    (void) mlid;
    (void) fndata;

    return;
}

void K3callbackdummy(gentype &res, int &minmaxind, int typeis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, int ia, int ib, int ic, int xdim, int densetype, int resmode, int mlid, void *fndata)
{
    (void) res;
    (void) minmaxind;
    (void) typeis;
    (void) xa;
    (void) xb;
    (void) xc;
    (void) xainfo;
    (void) xbinfo;
    (void) xcinfo;
    (void) ia;
    (void) ib;
    (void) ic;
    (void) xdim;
    (void) densetype;
    (void) resmode;
    (void) mlid;
    (void) fndata;

    return;
}

void K4callbackdummy(gentype &res, int &minmaxind, int typeis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, int ia, int ib, int ic, int id, int xdim, int densetype, int resmode, int mlid, void *fndata)
{
    (void) res;
    (void) minmaxind;
    (void) typeis;
    (void) xa;
    (void) xb;
    (void) xc;
    (void) xd;
    (void) xainfo;
    (void) xbinfo;
    (void) xcinfo;
    (void) xdinfo;
    (void) ia;
    (void) ib;
    (void) ic;
    (void) id;
    (void) xdim;
    (void) densetype;
    (void) resmode;
    (void) mlid;
    (void) fndata;

    return;
}

void Kmcallbackdummy(gentype &res, int &minmaxind, int typeis, Vector<const SparseVector<gentype> *> &x, Vector<const vecInfo *> &xinfo, Vector<int> &i, int xdim, int m, int densetype, int resmode, int mlid, void *fndata)
{
    (void) res;
    (void) minmaxind;
    (void) typeis;
    (void) x;
    (void) xinfo;
    (void) i;
    (void) xdim;
    (void) m;
    (void) densetype;
    (void) resmode;
    (void) mlid;
    (void) fndata;

    return;
}

double gcallbackaltdummy(const double *xa, int xadim, void *fndata)
{
    (void) xa;
    (void) xadim;
    (void) fndata;

    return 0;
}

double K0callbackaltdummy(int typeis, int xdim, int densetype, int resmode, int mlid, void *fndata)
{
    (void) typeis;
    (void) xdim;
    (void) densetype;
    (void) resmode;
    (void) mlid;
    (void) fndata;

    return 0;
}

double K1callbackaltdummy(int typeis, const double *xa, int xadim, int ia, int xdim, int densetype, int resmode, int mlid, void *fndata)
{
    (void) typeis;
    (void) xa;
    (void) ia;
    (void) xadim;
    (void) xdim;
    (void) densetype;
    (void) resmode;
    (void) mlid;
    (void) fndata;

    return 0;
}

double K2callbackaltdummy(int typeis, const double *xa, int xadim, const double *xb, int xbdim, int ia, int ib, int xdim, int densetype, int resmode, int mlid, void *fndata)
{
    (void) typeis;
    (void) xa;
    (void) xb;
    (void) ia;
    (void) ib;
    (void) xadim;
    (void) xbdim;
    (void) xdim;
    (void) densetype;
    (void) resmode;
    (void) mlid;
    (void) fndata;

    return 0;
}

double K3callbackaltdummy(int typeis, const double *xa, int xadim, const double *xb, int xbdim, const double *xc, int xcdim, int ia, int ib, int ic, int xdim, int densetype, int resmode, int mlid, void *fndata)
{
    (void) typeis;
    (void) xa;
    (void) xb;
    (void) xc;
    (void) ia;
    (void) ib;
    (void) ic;
    (void) xadim;
    (void) xbdim;
    (void) xcdim;
    (void) xdim;
    (void) densetype;
    (void) resmode;
    (void) mlid;
    (void) fndata;

    return 0;
}

double K4callbackaltdummy(int typeis, const double *xa, int xadim, const double *xb, int xbdim, const double *xc, int xcdim, const double *xd, int xddim, int ia, int ib, int ic, int id, int xdim, int densetype, int resmode, int mlid, void *fndata)
{
    (void) typeis;
    (void) xa;
    (void) xb;
    (void) xc;
    (void) xd;
    (void) ia;
    (void) ib;
    (void) ic;
    (void) id;
    (void) xadim;
    (void) xbdim;
    (void) xcdim;
    (void) xddim;
    (void) xdim;
    (void) densetype;
    (void) resmode;
    (void) mlid;
    (void) fndata;

    return 0;
}

double Kmcallbackaltdummy(int typeis, const double **xx, const int *xxdim, const int *ix, int xdim, int m, int densetype, int resmode, int mlid, void *fndata)
{
    (void) typeis;
    (void) xx;
    (void) ix;
    (void) xxdim;
    (void) xdim;
    (void) m;
    (void) densetype;
    (void) resmode;
    (void) mlid;
    (void) fndata;

    return 0;
}

double Kcallbackaltdummy(int typeis, double xprod, double diffis, int xdim, int densetype, int resmode, int mlid, void *fndata)
{
    (void) typeis;
    (void) xprod;
    (void) diffis;
    (void) xdim;
    (void) densetype;
    (void) resmode;
    (void) mlid;
    (void) fndata;

    return 0;
}


BLK_Generic::BLK_Generic(int isIndPrune) : ML_Base(isIndPrune)
{
    setaltx(nullptr);

    static std::string dummy = "";

    xuseristream = &promptstream(dummy);
    xuserostream = &promptoutstream();

    xcallback = nullptr; xcallbackfndata = nullptr;

    xK0callback = nullptr; xK0callbackdata = nullptr;
    xK1callback = nullptr; xK1callbackdata = nullptr;
    xK2callback = nullptr; xK2callbackdata = nullptr;
    xK3callback = nullptr; xK3callbackdata = nullptr;
    xK4callback = nullptr; xK4callbackdata = nullptr;
    xKmcallback = nullptr; xKmcallbackdata = nullptr;

    xKcallbackdata = nullptr;

    xcallbackalt = &gcallbackaltdummy;

    xK0callbackalt = K0callbackaltdummy;
    xK1callbackalt = K1callbackaltdummy;
    xK2callbackalt = K2callbackaltdummy;
    xK3callbackalt = K3callbackaltdummy;
    xK4callbackalt = K4callbackaltdummy;
    xKmcallbackalt = Kmcallbackaltdummy;

    xKcallbackalt = Kcallbackaltdummy;

    xmercachesize = -1;
    xmercachenorm = 0;

    xmlqmode = 0;

    locsigma = 1.0/DEFAULT_C;

    doutfn.makeNull();

    mexfnid = -3;

    xissample = 0;

    berndeg.force_null();
    bernind.force_null();

    // see blk_batter.cc

    xbattParam.resize(21);

    xbattParam("&",0)  = 261.9; // conversion Ah -> coulombs
    xbattParam("&",1)  = 1.18;
    xbattParam("&",2)  = -40;
    xbattParam("&",3)  = 1.29;
    xbattParam("&",4)  = 1.40;
    xbattParam("&",5)  = 49;
    xbattParam("&",6)  = 2.135;
    xbattParam("&",7)  = 0.58e-3;
    xbattParam("&",8)  = 5000;
    xbattParam("&",9)  = 2e-3;
    xbattParam("&",10) = 0.7e-3;
    xbattParam("&",11) = 15e-3;
    xbattParam("&",12) = -0.3;
    xbattParam("&",13) = -8;
    xbattParam("&",14) = -8.45;
    xbattParam("&",15) = 1.95;
    xbattParam("&",16) = 0.1;
    xbattParam("&",17) = 2;
    xbattParam("&",18) = 2e-12;
    xbattParam("&",19) = 15;
    xbattParam("&",20) = 0.2;

    xbatttmax             = 60*60;
    xbattImax             = 30;
    xbatttdelta           = 0.05;
    xbattVstart           = 2.135;
    xbattthetaStart       = 20;
    xbattneglectParasitic = 0;
    xbattfixedTheta       = -1000;

    KBlr      = 0.3;
    KBminstep = 1e-3;
    KBmaxiter = 50;

    return;
}

BLK_Generic::BLK_Generic(const BLK_Generic &src, int isIndPrune) : ML_Base(isIndPrune)
{
    xmercachesize = -1;
    xmercachenorm = 0;

    xissample = 0;

    setaltx(nullptr);

    assign(src,0);

    return;
}

BLK_Generic::BLK_Generic(const BLK_Generic &src, const ML_Base *xsrc, int isIndPrune) : ML_Base(isIndPrune)
{
    xmercachesize = -1;
    xmercachenorm = 0;

    xissample = 0;

    setaltx(xsrc);

    assign(src,-1);

    return;
}






int BLK_Generic::egetparam(int ind, Vector<gentype> &val, const Vector<gentype> &xa, int ia, const Vector<gentype> &xb, int ib) const
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

int BLK_Generic::getparam(int ind, gentype &val, const gentype &xa, int ia, const gentype &xb, int ib, charptr &desc) const
{
    int res = 0;

    desc = "";

    // Function to access parameters via indexing

    NiceAssert( ia >= 0 );
    NiceAssert( ib >= 0 );

    int isfallback = 0;

    switch ( ind )
    {
        case 1000: { val = outfn();                     desc = "BLK_Generic::outfn"; break; }
        case 1001: { val = outfngrad();                 desc = "BLK_Generic::outfngrad"; break; }
        case 1002: { val.force_string() = getmexcall(); desc = "BLK_Generic::getmexcall"; break; }
        case 1003: { val = getmexcallid();              desc = "BLK_Generic::getmexcallid"; break; }
        case 1004: { val = mercachesize();              desc = "BLK_Generic::mercachesize"; break; }
        case 1005: { val = mercachenorm();              desc = "BLK_Generic::mercachenorm"; break; }
        case 1006: { val.force_string() = getsyscall(); desc = "BLK_Generic::getsyscall"; break; }
        case 1007: { val = bernDegree();                desc = "BLK_Generic::bernDegree"; break; }
        case 1008: { val = bernIndex();                 desc = "BLK_Generic::bernIndex"; break; }
        case 1009: { val = battparam();                 desc = "BLK_Generic::battparam"; break; }
        case 1010: { val = batttmax();                  desc = "BLK_Generic::batttmax"; break; }
        case 1011: { val = battImax();                  desc = "BLK_Generic::battImax"; break; }
        case 1012: { val = batttdelta();                desc = "BLK_Generic::batttdelta"; break; }
        case 1013: { val = battVstart();                desc = "BLK_Generic::battVstart"; break; }
        case 1014: { val = battthetaStart();            desc = "BLK_Generic::battthetaStart"; break; }
        case 1015: { val = battneglectParasitic();      desc = "BLK_Generic::battneglectParasitic"; break; }
        case 1016: { val = battfixedTheta();            desc = "BLK_Generic::battfixedTheta"; break; }

        default:
        {
            isfallback = 1;
            res = ML_Base::getparam(ind,val,xa,ia,xb,ib,desc);

            break;
        }
    }

    if ( ( ia || ib ) && !isfallback )
    {
        val.force_null();
    }

    return res;
}





