
//
// Functional block base class
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _blk_generic_h
#define _blk_generic_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "ml_base.hpp"



// Defines a very basic set of blocks for use in machine learning.


class BLK_Generic;


typedef int (*gcallback)(gentype &res, const SparseVector<gentype> &x, void *fndata);

typedef void (*K0callbackfn)(gentype &res, int &minmaxind, int typeis, int xdim, int densetype, int resmode, int mlid, void *fndata);
typedef void (*K1callbackfn)(gentype &res, int &minmaxind, int typeis, const SparseVector<gentype> &xa, const vecInfo &xainfo, int ia, int xdim, int densetype, int resmode, int mlid, void *fndata);
typedef void (*K2callbackfn)(gentype &res, int &minmaxind, int typeis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, int ia, int ib,int xdim, int densetype, int resmode, int mlid, void *fndata);
typedef void (*K3callbackfn)(gentype &res, int &minmaxind, int typeis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, int ia, int ib, int ic, int xdim, int densetype, int resmode, int mlid, void *fndata);
typedef void (*K4callbackfn)(gentype &res, int &minmaxind, int typeis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, int ia, int ib, int ic, int id, int xdim, int densetype, int resmode, int mlid, void *fndata);
typedef void (*Kmcallbackfn)(gentype &res, int &minmaxind, int typeis, Vector<const SparseVector<gentype> *> &x, Vector<const vecInfo *> &xzinfo, Vector<int> &i, int xdim, int m, int densetype, int resmode, int mlid, void *fndata);

typedef double (*gcallbackalt)(const double *xa, int xadim, void *fndata);

typedef double (*K0callbackfnalt)(int typeis, int xdim, int densetype, int resmode, int mlid, void *fndata);
typedef double (*K1callbackfnalt)(int typeis, const double *xa, int xadim, int ia, int xdim, int densetype, int resmode, int mlid, void *fndata);
typedef double (*K2callbackfnalt)(int typeis, const double *xa, int xadim, const double *xb, int xbdim, int ia, int ib, int xdim, int densetype, int resmode, int mlid, void *fndata);
typedef double (*K3callbackfnalt)(int typeis, const double *xa, int xadim, const double *xb, int xbdim, const double *xc, int xcdim, int ia, int ib, int ic, int xdim, int densetype, int resmode, int mlid, void *fndata);
typedef double (*K4callbackfnalt)(int typeis, const double *xa, int xadim, const double *xb, int xbdim, const double *xc, int xcdim, const double *xd, int xddim, int ia, int ib, int ic, int id, int xdim, int densetype, int resmode, int mlid, void *fndata);
typedef double (*Kmcallbackfnalt)(int typeis, const double **xx, const int *xxdim, const int *ix, int xdim, int m, int densetype, int resmode, int mlid, void *fndata);

typedef double (*Kcallbackfnalt)(int typeis, double xprod, double diffis, int xdim, int densetype, int resmode, int mlid, void *fndata);

int gcallbackdummy(gentype &res, const SparseVector<gentype> &x, void *fndata);

void K0callbackdummy(gentype &res, int &minmaxind, int typeis, int xdim, int densetype, int resmode, int mlid, void *fndata);
void K1callbackdummy(gentype &res, int &minmaxind, int typeis, const SparseVector<gentype> &xa, const vecInfo &xainfo, int ia, int xdim, int densetype, int resmode, int mlid, void *fndata);
void K2callbackdummy(gentype &res, int &minmaxind, int typeis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, int ia, int ib,int xdim, int densetype, int resmode, int mlid, void *fndata);
void K3callbackdummy(gentype &res, int &minmaxind, int typeis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, int ia, int ib, int ic, int xdim, int densetype, int resmode, int mlid, void *fndata);
void K4callbackdummy(gentype &res, int &minmaxind, int typeis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, int ia, int ib, int ic, int id, int xdim, int densetype, int resmode, int mlid, void *fndata);
void Kmcallbackdummy(gentype &res, int &minmaxind, int typeis, Vector<const SparseVector<gentype> *> &x, Vector<const vecInfo *> &xzinfo, Vector<int> &i, int xdim, int m, int densetype, int resmode, int mlid, void *fndata);

double gcallbackaltdummy(const double *xa, int xadim, void *fndata);

double K0callbackaltdummy(int typeis, int xdim, int densetype, int resmode, int mlid, void *fndata);
double K1callbackaltdummy(int typeis, const double *xa, int xadim, int ia, int xdim, int densetype, int resmode, int mlid, void *fndata);
double K2callbackaltdummy(int typeis, const double *xa, int xadim, const double *xb, int xbdim, int ia, int ib, int xdim, int densetype, int resmode, int mlid, void *fndata);
double K3callbackaltdummy(int typeis, const double *xa, int xadim, const double *xb, int xbdim, const double *xc, int xcdim, int ia, int ib, int ic, int xdim, int densetype, int resmode, int mlid, void *fndata);
double K4callbackaltdummy(int typeis, const double *xa, int xadim, const double *xb, int xbdim, const double *xc, int xcdim, const double *xd, int xddim, int ia, int ib, int ic, int id, int xdim, int densetype, int resmode, int mlid, void *fndata);
double Kmcallbackaltdummy(int typeis, const double **xx, const int *xxdim, const int *ix, int xdim, int m, int densetype, int resmode, int mlid, void *fndata);

double Kcallbackaltdummy(int typeis, double xprod, double diffis, int xdim, int densetype, int resmode, int mlid, void *fndata);

// Swap and zeroing (restarting) functions

inline void qswap(BLK_Generic &a, BLK_Generic &b);
inline BLK_Generic &setzero(BLK_Generic &a);

class BLK_Generic : public ML_Base
{
public:

    // Assumptions: all virtual functions inherited from ML_Base are left
    // unchanged in this class.

    // Constructors, destructors, assignment etc..

    BLK_Generic(int isIndPrune = 0);
    BLK_Generic(const BLK_Generic &src, int isIndPrune = 0);
    BLK_Generic(const BLK_Generic &src, const ML_Base *xsrc, int isIndPrune = 0);
    BLK_Generic &operator=(const BLK_Generic &src) { assign(src); return *this; }
    virtual ~BLK_Generic() { return; }





    virtual void assign(const ML_Base &src, int onlySemiCopy = 0) override;
    virtual void semicopy(const ML_Base &src) override;
    virtual void qswapinternal(ML_Base &b) override;

    virtual int getparam (int ind, gentype         &val, const gentype         &xa, int ia, const gentype         &xb, int ib, charptr &desc) const override;
    virtual int egetparam(int ind, Vector<gentype> &val, const Vector<gentype> &xa, int ia, const Vector<gentype> &xb, int ib               ) const override;

    virtual std::ostream &printstream(std::ostream &output, int dep) const override;
    virtual std::istream &inputstream(std::istream &input ) override;

    virtual       ML_Base &getML     (void)       override { return static_cast<      ML_Base &>(getBLK()     ); }
    virtual const ML_Base &getMLconst(void) const override { return static_cast<const ML_Base &>(getBLKconst()); }

    virtual int isSampleMode(void) const override { return xissample; }
    virtual int setSampleMode(int nv, const Vector<gentype> &xmin, const Vector<gentype> &xmax, int Nsamp, int sampSplit, int sampType, int xsampType, double sampScale, double sampSlack = 0) override
    {
        int res = ( xissample != nv ) ? 1 : 0;

        if ( ( xissample = nv ) )
        {
            doutfn.finalise();
        }

        return res | ML_Base::setSampleMode(nv,xmin,xmax,Nsamp,sampSplit,sampType,xsampType,sampScale,sampSlack);
    }

    // Make this null so that it can be included as a prior without wasting time in optimising... nothing
    virtual double tuneKernel(int, double, int = 1) override { return 0; }


    // ================================================================
    //     BLK Specific functions
    // ================================================================

    virtual       BLK_Generic &getBLK     (void)       { return *this; }
    virtual const BLK_Generic &getBLKconst(void) const { return *this; }

    // Information functions (training data):

    virtual const gentype &outfn    (void) const { return doutfn; }
    virtual const gentype &outfngrad(void) const { outgrad = outfn(); outgrad.realDeriv(0,0); return outgrad; }

    // General modification and autoset functions

    virtual int setoutfn(const gentype &newoutfn)     { doutfn = newoutfn; return 1; }
    virtual int setoutfn(const std::string &newoutfn) { doutfn = newoutfn; return 1; }

    // Streams used by userio

    virtual int setuseristream(std::istream &src) { xuseristream = &src; return 1; }
    virtual int setuserostream(std::ostream &dst) { xuserostream = &dst; return 1; }

    virtual std::istream &useristream(void) const { return *xuseristream; }
    virtual std::ostream &userostream(void) const { return *xuserostream; }

    // Callback function used by calbak
    //
    // if Kxcallback is set then this is used...
    // failing that, if Kxcallbackalt is set then this is used...
    // and failing that, Kcallbackalt is used.

    virtual int setcallback(int (*ncallback)(gentype &, const SparseVector<gentype> &, void *), void *ncallbackfndata) { xcallback = ncallback ? ncallback : gcallbackdummy ; xcallbackfndata = ncallbackfndata; return 1; }

    virtual int setK0callback(K0callbackfn nK0callback, void *nK0callbackdata) { xK0callback = nK0callback ? nK0callback : K0callbackdummy; xK0callbackdata = nK0callbackdata; return 1; }
    virtual int setK1callback(K1callbackfn nK1callback, void *nK1callbackdata) { xK1callback = nK1callback ? nK1callback : K1callbackdummy; xK1callbackdata = nK1callbackdata; return 1; }
    virtual int setK2callback(K2callbackfn nK2callback, void *nK2callbackdata) { xK2callback = nK2callback ? nK2callback : K2callbackdummy; xK2callbackdata = nK2callbackdata; return 1; }
    virtual int setK3callback(K3callbackfn nK3callback, void *nK3callbackdata) { xK3callback = nK3callback ? nK3callback : K3callbackdummy; xK3callbackdata = nK3callbackdata; return 1; }
    virtual int setK4callback(K4callbackfn nK4callback, void *nK4callbackdata) { xK4callback = nK4callback ? nK4callback : K4callbackdummy; xK4callbackdata = nK4callbackdata; return 1; }
    virtual int setKmcallback(Kmcallbackfn nKmcallback, void *nKmcallbackdata) { xKmcallback = nKmcallback ? nKmcallback : Kmcallbackdummy; xKmcallbackdata = nKmcallbackdata; return 1; }

    virtual int setcallbackalt(double (*ncallbackalt)(const double *, int, void *), void *ncallbackfndata) { xcallback = nullptr; xcallbackalt = ncallbackalt ? ncallbackalt : gcallbackaltdummy; xcallbackfndata = ncallbackfndata; return 1; }

    virtual int setK0callbackalt(double (*nK0callbackalt)(int,                                                                                                         int, int, int, int, void *), void *nK0callbackdata) { xK0callback = nullptr; xK0callbackalt = nK0callbackalt ? nK0callbackalt : K0callbackaltdummy; xK0callbackdata = nK0callbackdata; return 1; }
    virtual int setK1callbackalt(double (*nK1callbackalt)(int, const double *, int, int,                                                                               int, int, int, int, void *), void *nK0callbackdata) { xK1callback = nullptr; xK1callbackalt = nK1callbackalt ? nK1callbackalt : K1callbackaltdummy; xK1callbackdata = nK0callbackdata; return 1; }
    virtual int setK2callbackalt(double (*nK2callbackalt)(int, const double *, int, const double *, int, int, int,                                                     int, int, int, int, void *), void *nK0callbackdata) { xK2callback = nullptr; xK2callbackalt = nK2callbackalt ? nK2callbackalt : K2callbackaltdummy; xK2callbackdata = nK0callbackdata; return 1; }
    virtual int setK3callbackalt(double (*nK3callbackalt)(int, const double *, int, const double *, int, const double *, int, int, int, int,                           int, int, int, int, void *), void *nK0callbackdata) { xK3callback = nullptr; xK3callbackalt = nK3callbackalt ? nK3callbackalt : K3callbackaltdummy; xK3callbackdata = nK0callbackdata; return 1; }
    virtual int setK4callbackalt(double (*nK4callbackalt)(int, const double *, int, const double *, int, const double *, int, const double *, int, int, int, int, int, int, int, int, int, void *), void *nK0callbackdata) { xK4callback = nullptr; xK4callbackalt = nK4callbackalt ? nK4callbackalt : K4callbackaltdummy; xK4callbackdata = nK0callbackdata; return 1; }
    virtual int setKmcallbackalt(double (*nKmcallbackalt)(int, const double **, const int *, const int *, int,                                                         int, int, int, int, void *), void *nK0callbackdata) { xKmcallback = nullptr; xKmcallbackalt = nKmcallbackalt ? nKmcallbackalt : Kmcallbackaltdummy; xKmcallbackdata = nK0callbackdata; return 1; }

    virtual int setKcallbackalt(double (*nKcallbackalt)(int, double, double, int, int, int, int, void *), void *nKcallbackdata) { xK0callback    = nullptr; xK1callback    = nullptr; xK2callback    = nullptr; xK3callback    = nullptr; xK4callback    = nullptr; xKmcallback    = nullptr; 
                                                                                                                                  xK0callbackalt = nullptr; xK1callbackalt = nullptr; xK2callbackalt = nullptr; xK3callbackalt = nullptr; xK4callbackalt = nullptr; xKmcallbackalt = nullptr; 
                                                                                                                                  xKcallbackalt = nKcallbackalt ? nKcallbackalt : Kcallbackaltdummy; xKcallbackdata = nKcallbackdata; return 1; }

    virtual gcallback callback(void)   const { return xcallback; }

    virtual K0callbackfn K0callback(void) const { return xK0callback; }
    virtual K1callbackfn K1callback(void) const { return xK1callback; }
    virtual K2callbackfn K2callback(void) const { return xK2callback; }
    virtual K3callbackfn K3callback(void) const { return xK3callback; }
    virtual K4callbackfn K4callback(void) const { return xK4callback; }
    virtual Kmcallbackfn Kmcallback(void) const { return xKmcallback; }

    virtual gcallbackalt callbackalt(void)   const { return xcallbackalt; }

    virtual K0callbackfnalt K0callbackalt(void) const { return xK0callbackalt; }
    virtual K1callbackfnalt K1callbackalt(void) const { return xK1callbackalt; }
    virtual K2callbackfnalt K2callbackalt(void) const { return xK2callbackalt; }
    virtual K3callbackfnalt K3callbackalt(void) const { return xK3callbackalt; }
    virtual K4callbackfnalt K4callbackalt(void) const { return xK4callbackalt; }
    virtual Kmcallbackfnalt Kmcallbackalt(void) const { return xKmcallbackalt; }

    virtual Kcallbackfnalt Kcallbackalt(void) const { return xKcallbackalt; }

    virtual void *callbackfndata(void) const { return xcallbackfndata; }

    virtual void *K0callbackdata(void) const { return xK0callbackdata; }
    virtual void *K1callbackdata(void) const { return xK1callbackdata; }
    virtual void *K2callbackdata(void) const { return xK2callbackdata; }
    virtual void *K3callbackdata(void) const { return xK3callbackdata; }
    virtual void *K4callbackdata(void) const { return xK4callbackdata; }
    virtual void *Kmcallbackdata(void) const { return xKmcallbackdata; }

    virtual void *Kcallbackdata(void) const { return xKcallbackdata; }

    // Callback string used by MEX interface

    virtual int setmexcall  (const std::string &xmexfn) { mexfn   = xmexfn;   return 1; }
    virtual int setmexcallid(int xmexfnid)              { mexfnid = xmexfnid; return 1; }
    virtual const std::string &getmexcall  (void) const { return mexfn;                 }
    virtual int                getmexcallid(void) const { return mexfnid;               }

    // Callback string used by sytem call interface
    //
    // This is cast as a (gentype) function and then evaluated given x
    //
    // xfilename:  datafile containing x data (not written if string empty)
    // yfilename:  datafile containing y data (not written if string empty)
    // xyfilename: datafile containing xy (target at end) data (not written if string empty)
    // yxfilename: datafile containing yx (target at start) data (not written if string empty)
    // rfilename:  name of file where result is retrieved (nullptr if string empty)

    virtual int setsyscall(const std::string &xsysfn)   { sysfn   = xsysfn; return 1; }
    virtual int setxfilename(const std::string &fname)  { xfname  = fname;  return 1; }
    virtual int setyfilename(const std::string &fname)  { yfname  = fname;  return 1; }
    virtual int setxyfilename(const std::string &fname) { xyfname = fname;  return 1; }
    virtual int setyxfilename(const std::string &fname) { yxfname = fname;  return 1; }
    virtual int setrfilename(const std::string &fname)  { rfname  = fname;  return 1; }

    virtual const std::string &getsyscall(void)    const { return sysfn;   }
    virtual const std::string &getxfilename(void)  const { return xfname;  }
    virtual const std::string &getyfilename(void)  const { return yfname;  }
    virtual const std::string &getxyfilename(void) const { return xyfname; }
    virtual const std::string &getyxfilename(void) const { return yxfname; }
    virtual const std::string &getrfilename(void)  const { return rfname;  }

    // MEX function: mex does not actually exist as far as this code is concerned.
    // Hence for the mex callback blocks you need to give it a funciton to call.
    // You'll need to set the following globally here to access it.  Operation is
    // assumed to be:
    //
    // getsetExtVar: - get or set external (typically mex) variable.
    //               - if num >= 0 then loads extvar num into res.  If extvar is
    //                 a function handle then src acts as an argument (optional,
    //                 not used if null, multiple arguments if set).
    //               - if num == -1 then loads external variable named in res
    //                 (res must be string) into res before returning.  In this
    //                 case src gives preferred type if result interpretation is
    //                 ambiguous (type of res will attempt to copy gentype of
    //                 src).
    //               - if num == -2 then loads contents of src into external
    //                 variable named in res before returning.
    //               - if num == -3 then evaluates fn(v) where fn is a matlab
    //                 function named by res, v is the set of arguments (see
    //                 num >= 0) and the result is stored in res.
    //               - returns 0 on success, -1 on failure.
    //
    // Call is (*getsetExtVar)(res,src,mexfnid), where res = mexfn is set prior
    // to call.

    // Mercer cache size: set -1 for no cache, N >= 0 for cache of size N
    //
    // fill cache: pre-calculates all elements in cache for later use
    // norm cache: normalise cache so that diagonals are all 1 in K2

    virtual int mercachesize   (void)   const { return xmercachesize; }
    virtual int setmercachesize(int nv)       { NiceAssert( nv >= -1 ); xmercachesize = nv; return 1; }

    virtual int mercachenorm   (void)   const { return xmercachenorm; }
    virtual int setmercachenorm(int nv)       { xmercachenorm = nv; return 1; }

    // ML block averaging: set/remove element in list of ML blocks being averaged
    //
    // mlqmode: 0 = normal weighed average
    //          1 = stack of priors.  Each model acts as prior mean to the next
    //              (0 is prior mean to 1, etc etc).  Only the model on top of
    //              the stack contributes to posterior variance.
    //          2 = like 1, but kernel tuning only applies to the final block

    virtual int setmlqmode(int nv)       { xmlqmode = nv; return xmlqmode; }
    virtual int getmlqmode(void)   const {                return xmlqmode; }

    virtual int setmlqlist(int i, ML_Base &src)          { xmlqlist("[]",i) = &src; xmlqweight("[]",i) = 1.0;                    return 1; }
    virtual int setmlqlist(const Vector<ML_Base *> &src) { xmlqlist = src; xmlqweight.indalign(xmlqlist); xmlqweight = 1.0_gent; return 1; }

    virtual int setmlqweight(int i, const gentype &w)  { xmlqweight("[]",i) = w; return 1; }
    virtual int setmlqweight(const Vector<gentype> &w) { xmlqweight = w;         return 1; }

    virtual int addmlqlist(int i, ML_Base &src) { xmlqlist.add(i); xmlqlist("[]",i) = &src; xmlqweight.add(i); xmlqweight("[]",i) = 1.0; return 1; }
    virtual int removemlqlist(int i)            { xmlqlist.remove(i);                       xmlqweight.remove(i);                        return 1; }

    const SparseVector<ML_Base *> mlqlist  (void) const { return xmlqlist; }
    const SparseVector<gentype>   mlqweight(void) const { return xmlqweight; }

    // Kernel training (hyper/m- kenels):
    //
    // K(z_1,z_1,...,z_{m-1}) = sum_{i_0,j_1,...,i_{m-1}} = 1...N} sum_q lambda_{i_0,q} lambda_{i_1,q} ... lambda_{i_{m-1},q} K(x_{i_0},x_{i_1}...,x_{i_{m-1}},z_0,z_1,...,z_{m-1})
    //
    // Set altMLKB to reference a set of MLs and MLweightKB their weights if you want to use bi-quadratic optimisation on lambda

    virtual double minstepKB(void) const { return KBminstep; }
    virtual int    maxiterKB(void) const { return KBmaxiter; }
    virtual double lrKB     (void) const { return KBlr;      }

    virtual const Vector<int>    altMLidsKB(void) const { return KBaltMLids; }
    virtual const Vector<double> MLweightKB(void) const { return KBMLweight; }
    virtual const Matrix<double> &lambdaKB (void) const { return KBlambda;   }

    virtual int setminstepKB(double nv) { KBminstep = nv; return 1; }
    virtual int setmaxiterKB(int    nv) { KBmaxiter = nv; return 1; }
    virtual int setlrKB     (double nv) { KBlr = nv;      return 1; }

    virtual int setaltMLidsKB(const Vector<int>    &nv) { KBaltMLids = nv; return 1; }
    virtual int setMLweightKB(const Vector<double> &nv) { KBMLweight = nv; return 1; }
    virtual int setlambdaKB  (const Matrix<double> &nv) { KBlambda = nv;   return 1; }

    // Bernstein polynomials
    //
    // degree and index can be either null, int or Vector<int>.

    virtual const gentype &bernDegree(void) const { return berndeg; }
    virtual const gentype &bernIndex(void)  const { return bernind; }

    virtual int setBernDegree(const gentype &nv) { berndeg = nv; return 1; }
    virtual int setBernIndex(const gentype &nv)  { bernind = nv; return 1; }

    typedef int (*mexcallsyn)(gentype &, const gentype &, int);
    static mexcallsyn getsetExtVar;

    // Battery modelling parameters
    //
    // battparam: 21-d vector of battery parameters
    // batttmax: total simulation time (sec)
    // battImax: max charge/discharge current (amps)
    // batttdelta: time granunaliry (sec)
    // battVstart: start voltage (V)
    // battthetaStart: start temperature (deg)
    // battneglectParasitic: neglect parasitic branch if set
    // battfixedTheta: if >-1000 then use this fixed theta

    virtual const Vector<double> &battparam(void)            const { return xbattParam;            }
    virtual       double          batttmax(void)             const { return xbatttmax;             }
    virtual       double          battImax(void)             const { return xbattImax;             }
    virtual       double          batttdelta(void)           const { return xbatttdelta;           }
    virtual       double          battVstart(void)           const { return xbattVstart;           }
    virtual       double          battthetaStart(void)       const { return xbattthetaStart;       }
    virtual       int             battneglectParasitic(void) const { return xbattneglectParasitic; }
    virtual       double          battfixedTheta(void)       const { return xbattfixedTheta;       }

    virtual int setbattparam(const Vector<gentype> &nv)
    {
        Vector<double> nnv(xbattParam);

        NiceAssert( nv.size() == nnv.size() );

        int i;

        for ( i = 0 ; i < nv.size() ; ++i )
        {
            if ( !nv(i).isValNull() )
            {
                nnv.sv(i,(double) nv(i));
            }
        }

        xbattParam = nnv;

        return 1;
    }

    virtual int setbatttmax(double nv)          { xbatttmax             = nv; return 1; }
    virtual int setbattImax(double nv)          { xbattImax             = nv; return 1; }
    virtual int setbatttdelta(double nv)        { xbatttdelta           = nv; return 1; }
    virtual int setbattVstart(double nv)        { xbattVstart           = nv; return 1; }
    virtual int setbattthetaStart(double nv)    { xbattthetaStart       = nv; return 1; }
    virtual int setbattneglectParasitic(int nv) { xbattneglectParasitic = nv; return 1; }
    virtual int setbattfixedTheta(double nv)    { xbattfixedTheta       = nv; return 1; }

private:

    int xissample;

    int xmercachesize;
    int xmercachenorm;

    int xmlqmode;

    gentype doutfn;

    std::istream *xuseristream;
    std::ostream *xuserostream;

    gcallback xcallback; void *xcallbackfndata;

    K0callbackfn xK0callback; void *xK0callbackdata;
    K1callbackfn xK1callback; void *xK1callbackdata;
    K2callbackfn xK2callback; void *xK2callbackdata;
    K3callbackfn xK3callback; void *xK3callbackdata;
    K4callbackfn xK4callback; void *xK4callbackdata;
    Kmcallbackfn xKmcallback; void *xKmcallbackdata;

    void *xKcallbackdata;

    gcallbackalt xcallbackalt;

    K0callbackfnalt xK0callbackalt;
    K1callbackfnalt xK1callbackalt;
    K2callbackfnalt xK2callbackalt;
    K3callbackfnalt xK3callbackalt;
    K4callbackfnalt xK4callbackalt;
    Kmcallbackfnalt xKmcallbackalt;

    Kcallbackfnalt xKcallbackalt;

    std::string mexfn;
    int mexfnid;

    std::string sysfn;
    std::string xfname;
    std::string yfname;
    std::string xyfname;
    std::string yxfname;
    std::string rfname;

    mutable gentype outgrad;

    // ML block averaging

    SparseVector<ML_Base *> xmlqlist;
    SparseVector<gentype> xmlqweight;

    // Kernel training

    double KBlr;
    double KBminstep;
    int KBmaxiter;
    Vector<int> KBaltMLids;
    Vector<double> KBMLweight;
    Matrix<double> KBlambda;

    // Bernstein

    gentype berndeg;
    gentype bernind;

    // Battery sims

    Vector<double> xbattParam;
    double xbatttmax;
    double xbattImax;
    double xbatttdelta;
    double xbattVstart;
    double xbattthetaStart;
    int xbattneglectParasitic;
    double xbattfixedTheta;
};

inline double norm2(const BLK_Generic &a);
inline double abs2 (const BLK_Generic &a);

inline double norm2(const BLK_Generic &a) { return a.RKHSnorm(); }
inline double abs2 (const BLK_Generic &a) { return a.RKHSabs();  }

inline void qswap(BLK_Generic &a, BLK_Generic &b)
{
    a.qswapinternal(b);

    return;
}

inline BLK_Generic &setzero(BLK_Generic &a)
{
    a.restart();

    return a;
}

inline void BLK_Generic::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    BLK_Generic &b = dynamic_cast<BLK_Generic &>(bb.getML());

    qswap(xissample,b.xissample);

    qswap(xmlqmode,b.xmlqmode);

    qswap(doutfn       ,b.doutfn       );
    qswap(xmercachesize,b.xmercachesize);
    qswap(xmercachenorm,b.xmercachenorm);

    std::istream *xistream;
    std::ostream *xostream;

    xistream = xuseristream; xuseristream = b.xuseristream; b.xuseristream = xistream;
    xostream = xuserostream; xuserostream = b.xuserostream; b.xuserostream = xostream;

    gcallback qcallback; void *qcallbackfndata;

    K0callbackfn qK0callback; void *qK0callbackdata;
    K1callbackfn qK1callback; void *qK1callbackdata;
    K2callbackfn qK2callback; void *qK2callbackdata;
    K3callbackfn qK3callback; void *qK3callbackdata;
    K4callbackfn qK4callback; void *qK4callbackdata;
    Kmcallbackfn qKmcallback; void *qKmcallbackdata;

    K0callbackfnalt qK0callbackalt;
    K1callbackfnalt qK1callbackalt;
    K2callbackfnalt qK2callbackalt;
    K3callbackfnalt qK3callbackalt;
    K4callbackfnalt qK4callbackalt;
    Kmcallbackfnalt qKmcallbackalt;

    Kcallbackfnalt qKcallbackalt;

    qcallback = xcallback; xcallback = b.xcallback; b.xcallback = qcallback;

    qK0callback = xK0callback; xK0callback = b.xK0callback; b.xK0callback = qK0callback;
    qK1callback = xK1callback; xK1callback = b.xK1callback; b.xK1callback = qK1callback;
    qK2callback = xK2callback; xK2callback = b.xK2callback; b.xK2callback = qK2callback;
    qK3callback = xK3callback; xK3callback = b.xK3callback; b.xK3callback = qK3callback;
    qK4callback = xK4callback; xK4callback = b.xK4callback; b.xK4callback = qK4callback;
    qKmcallback = xKmcallback; xKmcallback = b.xKmcallback; b.xKmcallback = qKmcallback;

    qK0callbackalt = xK0callbackalt; xK0callbackalt = b.xK0callbackalt; b.xK0callbackalt = qK0callbackalt;
    qK1callbackalt = xK1callbackalt; xK1callbackalt = b.xK1callbackalt; b.xK1callbackalt = qK1callbackalt;
    qK2callbackalt = xK2callbackalt; xK2callbackalt = b.xK2callbackalt; b.xK2callbackalt = qK2callbackalt;
    qK3callbackalt = xK3callbackalt; xK3callbackalt = b.xK3callbackalt; b.xK3callbackalt = qK3callbackalt;
    qK4callbackalt = xK4callbackalt; xK4callbackalt = b.xK4callbackalt; b.xK4callbackalt = qK4callbackalt;
    qKmcallbackalt = xKmcallbackalt; xKmcallbackalt = b.xKmcallbackalt; b.xKmcallbackalt = qKmcallbackalt;

    qKcallbackalt = xKcallbackalt; xKcallbackalt = b.xKcallbackalt; b.xKcallbackalt = qKcallbackalt;

    qcallbackfndata = xcallbackfndata; xcallbackfndata = b.xcallbackfndata; b.xcallbackfndata = qcallbackfndata;

    qK0callbackdata = xK0callbackdata; xK0callbackdata = b.xK0callbackdata; b.xK0callbackdata = qK0callbackdata;
    qK1callbackdata = xK1callbackdata; xK1callbackdata = b.xK1callbackdata; b.xK1callbackdata = qK1callbackdata;
    qK2callbackdata = xK2callbackdata; xK2callbackdata = b.xK2callbackdata; b.xK2callbackdata = qK2callbackdata;
    qK3callbackdata = xK3callbackdata; xK3callbackdata = b.xK3callbackdata; b.xK3callbackdata = qK3callbackdata;
    qK4callbackdata = xK4callbackdata; xK4callbackdata = b.xK4callbackdata; b.xK4callbackdata = qK4callbackdata;
    qKmcallbackdata = xKmcallbackdata; xKmcallbackdata = b.xKmcallbackdata; b.xKmcallbackdata = qKmcallbackdata;

    qswap(mexfn  ,b.mexfn  );
    qswap(mexfnid,b.mexfnid);

    qswap(sysfn  ,b.sysfn  );
    qswap(xfname ,b.xfname );
    qswap(yfname ,b.yfname );
    qswap(xyfname,b.xyfname);
    qswap(yxfname,b.yxfname);
    qswap(rfname ,b.rfname );

    qswap(KBlr      ,b.KBlr      );
    qswap(KBminstep ,b.KBminstep );
    qswap(KBmaxiter ,b.KBmaxiter );
    qswap(KBaltMLids,b.KBaltMLids);
    qswap(KBMLweight,b.KBMLweight);
    qswap(KBlambda  ,b.KBlambda  );

    qswap(xmlqlist  ,b.xmlqlist  );
    qswap(xmlqweight,b.xmlqweight);

    qswap(berndeg,b.berndeg);
    qswap(bernind,b.bernind);

    qswap(xbattParam           ,b.xbattParam           );
    qswap(xbatttmax            ,b.xbatttmax            );
    qswap(xbattImax            ,b.xbattImax            );
    qswap(xbatttdelta          ,b.xbatttdelta          );
    qswap(xbattVstart          ,b.xbattVstart          );
    qswap(xbattthetaStart      ,b.xbattthetaStart      );
    qswap(xbattneglectParasitic,b.xbattneglectParasitic);
    qswap(xbattfixedTheta      ,b.xbattfixedTheta      );

    ML_Base::qswapinternal(b);

    return;
}

inline void BLK_Generic::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const BLK_Generic &b = dynamic_cast<const BLK_Generic &>(bb.getMLconst());

    xissample = b.xissample;

    xmlqmode = b.xmlqmode;

    doutfn        = b.doutfn;
    xmercachesize = b.xmercachesize;
    xmercachenorm = b.xmercachenorm;

    xuseristream = b.xuseristream;
    xuserostream = b.xuserostream;

    xcallback = b.xcallback; xcallbackfndata = b.xcallbackfndata;

    xK0callback = b.xK0callback; xK0callbackdata = b.xK0callbackdata;
    xK1callback = b.xK1callback; xK1callbackdata = b.xK1callbackdata;
    xK2callback = b.xK2callback; xK2callbackdata = b.xK2callbackdata;
    xK3callback = b.xK3callback; xK3callbackdata = b.xK3callbackdata;
    xK4callback = b.xK4callback; xK4callbackdata = b.xK4callbackdata;
    xKmcallback = b.xKmcallback; xKmcallbackdata = b.xKmcallbackdata;

    xcallbackalt = b.xcallbackalt;

    xK0callbackalt = b.xK0callbackalt;
    xK1callbackalt = b.xK1callbackalt;
    xK2callbackalt = b.xK2callbackalt;
    xK3callbackalt = b.xK3callbackalt;
    xK4callbackalt = b.xK4callbackalt;
    xKmcallbackalt = b.xKmcallbackalt;

    xKcallbackalt = b.xKcallbackalt;

    mexfn   = b.mexfn;
    mexfnid = b.mexfnid;

    sysfn   = b.sysfn;
    xfname  = b.xfname;
    yfname  = b.yfname;
    xyfname = b.xyfname;
    yxfname = b.yxfname;
    rfname  = b.rfname;

    KBlr       = b.KBlr;
    KBminstep  = b.KBminstep;
    KBmaxiter  = b.KBmaxiter;
    KBaltMLids = b.KBaltMLids;
    KBMLweight = b.KBMLweight;
    KBlambda   = b.KBlambda;

    xmlqlist   = b.xmlqlist;
    xmlqweight = b.xmlqweight;

    berndeg = b.berndeg;
    bernind = b.bernind;

    xbattParam            = b.xbattParam;
    xbatttmax             = b.xbatttmax;
    xbattImax             = b.xbattImax;
    xbatttdelta           = b.xbatttdelta;
    xbattVstart           = b.xbattVstart;
    xbattthetaStart       = b.xbattthetaStart;
    xbattneglectParasitic = b.xbattneglectParasitic;
    xbattfixedTheta       = b.xbattfixedTheta;

    ML_Base::semicopy(b);

    return;
}

inline void BLK_Generic::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const BLK_Generic &src = dynamic_cast<const BLK_Generic &>(bb.getMLconst());

    xissample = src.xissample;

    xmlqmode = src.xmlqmode;

    doutfn        = src.doutfn;
    xmercachesize = src.xmercachesize;
    xmercachenorm = src.xmercachenorm;

    xuseristream = src.xuseristream;
    xuserostream = src.xuserostream;

    xcallback = src.xcallback; xcallbackfndata = src.xcallbackfndata;

    xK0callback = src.xK0callback; xK0callbackdata = src.xK0callbackdata;
    xK1callback = src.xK1callback; xK1callbackdata = src.xK1callbackdata;
    xK2callback = src.xK2callback; xK2callbackdata = src.xK2callbackdata;
    xK3callback = src.xK3callback; xK3callbackdata = src.xK3callbackdata;
    xK4callback = src.xK4callback; xK4callbackdata = src.xK4callbackdata;
    xKmcallback = src.xKmcallback; xKmcallbackdata = src.xKmcallbackdata;

    xcallbackalt = src.xcallbackalt;

    xK0callbackalt = src.xK0callbackalt;
    xK1callbackalt = src.xK1callbackalt;
    xK2callbackalt = src.xK2callbackalt;
    xK3callbackalt = src.xK3callbackalt;
    xK4callbackalt = src.xK4callbackalt;
    xKmcallbackalt = src.xKmcallbackalt;

    xKcallbackalt = src.xKcallbackalt;

    mexfn   = src.mexfn;
    mexfnid = src.mexfnid;

    sysfn   = src.sysfn;
    xfname  = src.xfname;
    yfname  = src.yfname;
    xyfname = src.xyfname;
    yxfname = src.yxfname;
    rfname  = src.rfname;

    KBlr       = src.KBlr;
    KBminstep  = src.KBminstep;
    KBmaxiter  = src.KBmaxiter;
    KBaltMLids = src.KBaltMLids;
    KBMLweight = src.KBMLweight;
    KBlambda   = src.KBlambda;

    xmlqlist   = src.xmlqlist;
    xmlqweight = src.xmlqweight;

    berndeg = src.berndeg;
    bernind = src.bernind;

    xbattParam            = src.xbattParam;
    xbatttmax             = src.xbatttmax;
    xbattImax             = src.xbattImax;
    xbatttdelta           = src.xbatttdelta;
    xbattVstart           = src.xbattVstart;
    xbattthetaStart       = src.xbattthetaStart;
    xbattneglectParasitic = src.xbattneglectParasitic;
    xbattfixedTheta       = src.xbattfixedTheta;

    ML_Base::assign(src,onlySemiCopy);

    return;
}

#endif
