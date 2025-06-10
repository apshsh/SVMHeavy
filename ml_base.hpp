

//
// ML (machine learning) base type
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _ml_base_h
#define _ml_base_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include <cmath>
//#ifdef ENABLE_THREADS
//#include <mutex>
//#endif
#include "mercer.hpp"
#include "vector.hpp"
#include "sparsevector.hpp"
#include "matrix.hpp"
#include "gentype.hpp"
#include "mlcommon.hpp"
#include "basefn.hpp"
#include "memdebug.hpp"
#include "niceassert.hpp"
#include "numbase.hpp"
#include "FNVector.hpp"

class ML_Base;
class SVM_Planar;
class SVM_Scalar;
class SVM_Generic;
class SVM_KConst;
class SVM_Scalar_rff;
class ONN_Generic;
class KNN_Scalar;
class KNN_Vector;
class KNN_MultiC;
class KNN_Binary;
class KNN_Anions;
class BLK_AveVec;
class BLK_AveAni;

#include "errortest.hpp"

#define NUMXTYPES 12

inline std::ostream &operator<<(std::ostream &output, const ML_Base &src );
inline std::istream &operator>>(std::istream &input,        ML_Base &dest);

// Compatibility functions

int isSemicopyCompat(const ML_Base &a, const ML_Base &b);
int isQswapCompat(const ML_Base &a, const ML_Base &b);
int isAssignCompat(const ML_Base &a, const ML_Base &b);

// Swap and zeroing (restarting) functions

inline void qswap(ML_Base &a, ML_Base &b);
inline void qswap(ML_Base *&a, ML_Base *&b);
inline void qswap(const ML_Base *&a, const ML_Base *&b);

inline ML_Base &setident (ML_Base &a) { NiceThrow("something"); return a; }
inline ML_Base &setzero  (ML_Base &a);
inline ML_Base &setposate(ML_Base &a) { return a; }
inline ML_Base &setnegate(ML_Base &a) { NiceThrow("something"); return a; }
inline ML_Base &setconj  (ML_Base &a) { NiceThrow("something"); return a; }
inline ML_Base &setrand  (ML_Base &a) { NiceThrow("something"); return a; }
inline ML_Base &postProInnerProd(ML_Base &a) { return a; }

inline ML_Base *&setident (ML_Base *&a) { NiceThrow("something"); return a; }
inline ML_Base *&setzero  (ML_Base *&x);
inline ML_Base *&setposate(ML_Base *&a) { return a; }
inline ML_Base *&setnegate(ML_Base *&a) { NiceThrow("something"); return a; }
inline ML_Base *&setconj  (ML_Base *&a) { NiceThrow("something"); return a; }
inline ML_Base *&setrand  (ML_Base *&a) { NiceThrow("something"); return a; }
inline ML_Base *&postProInnerProd(ML_Base *&a) { return a; }

inline const ML_Base *&setident (const ML_Base *&a) { NiceThrow("something"); return a; }
inline const ML_Base *&setzero  (const ML_Base *&x);
inline const ML_Base *&setposate(const ML_Base *&a) { return a; }
inline const ML_Base *&setnegate(const ML_Base *&a) { NiceThrow("something"); return a; }
inline const ML_Base *&setconj  (const ML_Base *&a) { NiceThrow("something"); return a; }
inline const ML_Base *&setrand  (const ML_Base *&a) { NiceThrow("something"); return a; }
inline const ML_Base *&postProInnerProd(const ML_Base *&a) { return a; }


// Training vector conversion (ONLY for use in getparam function)
//
// convertSetToSparse: res = { [ s0 : s1 :: s2 ... ] if src = { s0, s1, s2, ... }
//                           { s0                    if src = s0
// convertSparseToSet: res = { { s0, s1, s2, ... } } if src = [ s0 : s1 :: s2 ... ]
//                           { s0                    if src = [ s0 ]
//
// Key assumption: s0, s1, ... non-sparse
//
// If idiv > 0 then res.f4(6) += idiv (additional idiv derivatives)
//
// Return value: 1 if function but not scalar function 0 otherwise.

int convertSetToSparse(SparseVector<gentype> &res, const gentype &src, int idiv = 0);
int convertSparseToSet(gentype &res, const SparseVector<gentype> &src);

// Similarity callbacks - UU uses output kernel and m can be anything, VV has no kernel and assumed m = 2

gentype &UUcallbacknon(gentype &res, int m, const ML_Base &caller, Vector<int> &iokr, Vector<int> &iok, Vector<const gentype *> xalt, int defbasis);
gentype &UUcallbackdef(gentype &res, int m, const ML_Base &caller, Vector<int> &iokr, Vector<int> &iok, Vector<const gentype *> xalt, int defbasis);

const gentype &VVcallbacknon(gentype &res, int m, const gentype &kval, const ML_Base &caller, Vector<int> &iokr, Vector<int> &iok, Vector<const gentype *> xalt, int defbasis);
const gentype &VVcallbackdef(gentype &res, int m, const gentype &kval, const ML_Base &caller, Vector<int> &iokr, Vector<int> &iok, Vector<const gentype *> xalt, int defbasis);

#define DEFCMIN 0.01
#define DEFCMAX 10
#define DEFEPSMIN 1e-5
#define DEFEPSMAX 2
#define NUMZOOMS   2
#define ZOOMFACTOR 0.3
//#define MAXADIM 10000
#define MAXADIM 15000

class tkBounds
{
public:
    explicit tkBounds()
    {
        Cmin = 0.01;
        Cmax = 100;
        epsmin = 0;
        epsmax = 1;
    }

    explicit tkBounds(const MercerKernel &kerntemp)
    {
        wlb.resize(kerntemp.size()) = -INFINITY;
        wub.resize(kerntemp.size()) = INFINITY;

        klb.resize(kerntemp.size());
        kub.resize(kerntemp.size());

        for ( int q = 0 ; q < kerntemp.size() ; ++q )
        {
            klb("&",q).resize(kerntemp.cRealConstants(q).size()) = -INFINITY;
            kub("&",q).resize(kerntemp.cRealConstants(q).size()) = INFINITY;
        }

        Cmin = DEFCMIN;
        Cmax = DEFCMAX;

        epsmin = DEFEPSMIN;
        epsmax = DEFEPSMAX;

        numzooms   = NUMZOOMS;
        zoomfactor = ZOOMFACTOR;
    }

    Vector<double> wlb; // kernel weight lower bounds
    Vector<double> wub; // kernel weight upper bounds
    Vector<Vector<double> > klb; // kernel constant lower bounds
    Vector<Vector<double> > kub; // kernel constant upper bounds

    double Cmin;
    double Cmax;
    double epsmin;
    double epsmax;

    int    numzooms;
    double zoomfactor;
};


class ML_Base : public kernPrecursor
{
    friend class SVM_Planar;
    friend class SVM_Scalar;
    friend class SVM_Generic;
    friend class SVM_KConst;
    friend class SVM_Scalar_rff;
    friend class ONN_Generic;
    friend class KNN_Scalar;
    friend class KNN_Vector;
    friend class KNN_MultiC;
    friend class KNN_Binary;
    friend class KNN_Anions;
    friend class BLK_AveVec;
    friend class BLK_AveAni;

public:

    // Constructors, destructors, assignment etc..
    //
    // prealloc: call this to set the expected size of the training set.
    //     this allows memory to be preallocated as a single block rather
    //     than incrementally on a point-by-point basis.  Using this is both
    //     quicker and more memory efficient than the alternative.
    // preallocsize: returns the current preallocation size (0 if none).
    // setmemsize: set size of kernel etc caches.
    //
    // assign: copy constructor
    // semicopy: this is like an assignment operator, but assumes that dest
    //     has all training data and therefore only copies state information.
    //     It should be used when variables have been constrained zero
    //     temporarily (eg when doing cross-fold validation) and you want to
    //     quickly regain old state.
    // qswapinternal: qswap function
    //
    // getparam: get parameter via indexed callback.  Return 0 on success, 1
    //     is answer cannot be resolved (ie is still a function).
    //
    // printstream: print ML to output stream.
    // inputstream: get ML from input stream.
    //
    // NB: - all of these actually dynamic the objects based on the type()
    //       argument and then call the relevant cast version.
    //     - print and input functions are just placeholders called by the
    //       stream operators << and >> on this class.  Polymorph as needed.

    ML_Base(int _isIndPrune = 0);
    ML_Base &operator=(const ML_Base &src) { assign(src); return *this; }
    virtual ~ML_Base();

    virtual int  prealloc    (int expectedN);
    virtual int  preallocsize(void)          const { return xpreallocsize; }
    virtual void setmemsize  (int memsize);

    virtual void assign       (const ML_Base &src, int onlySemiCopy = 0);
    virtual void semicopy     (const ML_Base &src);
    virtual void qswapinternal(      ML_Base &b);

    virtual int getparam (int ind, gentype         &val, const gentype         &xa, int ia, const gentype         &xb, int ib, charptr &desc) const override;
    virtual int egetparam(int ind, Vector<gentype> &val, const Vector<gentype> &xa, int ia, const Vector<gentype> &xb, int ib               ) const override;

    virtual std::ostream &printstream(std::ostream &output, int dep) const override;
    virtual std::istream &inputstream(std::istream &input          )       override;

    virtual       ML_Base &getML     (void)       { return *this; }
    virtual const ML_Base &getMLconst(void) const { return *this; }

    // Generate RKHS vector form of ML (if possible).

    virtual RKHSVector      &getvecforma(RKHSVector      &res) const { res.resetit(getKernel(),x(),xinfo(),alphaVal(),isUnderlyingVector());          return res; }
    virtual Vector<gentype> &getvecformb(Vector<gentype> &res) const { makeanRKHSVector(res,getKernel(),x(),xinfo(),alphaVal(),isUnderlyingVector()); return res; }
    virtual gentype         &getvecformc(gentype         &res) const { getvecformb(res.force_vector());                                               return res; }

    // Information functions (training data):
    //
    // N:    the number of training vectors
    // NNC:  the number of training vectors in a given internal class
    // type: the type of the machine.
    // Nw:   size of w vector
    // NwS:  number of non-zero elements in w
    // NwZ:  number of zero elements in w
    //
    // tspaceDim:  the dimensionality of target space.
    // xspaceDim:  index dimensionality of input space (u=-1 gives overall dim, u>=0 gives only dimension for relevant minor/up type - see sparsevector).
    // fspaceDim:  dimensionality of feature space (-1 if infinite)
    // numClasses: the number of classes
    // order:      log2(tspaceDim)
    //
    // isTrained: true if machine is trained and up to date
    // isMutable: true if machine is mutable (type can be changed)
    // isPool:    true if machine is ML_Pool
    //
    // getML:      reference to "actual" ML (differs from this if mutable)
    // getMLconst: reference to "actual" ML (differs from this if mutable)
    //
    // isUnderlyingScalar: true if underlying weight type is double
    // isUnderlyingVector: true if underlying weight type is vector
    // isUnderlyingAnions: true if underlying weight type is anions
    //
    // mpri:   mean prior type: 0 (none), 1 (gentype evaluated), 2 (ML)
    // prival: prior mean function (mpri == 1)
    // priml:  prior ML_Base object (mpri == 2)
    //
    // ClassLabels: returns a vector of all class labels.
    // getInternalClass: for classifiers this returns the internal class
    //     representation number (0 for regressor).  For all classifiers each
    //     class is assigned a number 0,1,...,m, where m = numInternalClasses,
    //     which is the number of actual classes plus the anomaly class, if
    //     there is one.
    // numInternalClasses: number of internal classes
    //
    // gOutType: unprocessed g(x) output type of machine (see gentype)
    // hOutType: processed h(x) output type of machine (see gentype)
    // targType: target data y type of machine (see gentype)
    // calcDist: given processed outputs ha, hb calculate, for this ML, the
    //     norm  squared error.  db applies to scalar types and is ignored
    //     elsewhere.  +1 indicates lower bound only, -1 upper bound only.
    //     For all types 0 means don't include.
    //
    // sparlvl: sparsity level (1 completely sparse, 0 non-sparse)
    //
    // isSVM.../isONN...: returns true if type matches
    //
    // zerotol:      zero tolerance
    // Opttol:       optimality tolerance
    // maxitcnt:     maximum iteration count for training
    // maxtraintime: maximum training time
    // traintimeend: absolute training time end
    //
    // x,y,Cweight,epsweight: training data
    // b,w,ws,v,vn: trained machine
    // ws_i = 0 if w_i = 0, nz otherwise
    // vn_i = the norm of v_i
    //
    // x is the training vectors
    // y is the targets
    //
    // alphaState: has usual meaning for svm_generic, but also used to
    //             indicate if a particular training pair has any influence
    //             on the trained machine.  By default 1 for all training
    //             variables
    // alphaVal: like alphaState
    //
    // RFFordata: return 1 if vector i is actually RFF, 0 otherwise (used
    //            when evaluating the kernel.  If this tests 1 for all
    //            vectors then the RFFKernel is used)
    //
    // isVarDefined: 0 posterior variance not defined
    //               1 posterior variance defined
    //               2 pseudo-posterior variance defined (like an LS-SVM, which can be interpretted as a GP but really isn't)

    virtual int  N       (void)  const          { return locNcalc();     }
    virtual int  NNC     (int d) const          { return d ? 0 : xdzero; }
    virtual int  type    (void)  const override { return -1;             }
    virtual int  subtype (void)  const override { return 0;              }
    virtual char gOutType(void)  const          { return '?';            }
    virtual char hOutType(void)  const          { return '?';            }
    virtual char targType(void)  const          { return '?';            }

    virtual int tspaceDim   (void)       const { return 1;                                                             }
    virtual int xspaceDim   (int u = -1) const { return ( wildxdim > indKey(u).size() ) ? wildxdim : indKey(u).size(); }
    virtual int fspaceDim   (void)       const { return getKernel().phidim(1,xspaceDim());                             }
    virtual int tspaceSparse(void)       const { return 0;                                                             }
    virtual int xspaceSparse(void)       const { return 1;                                                             }
    virtual int numClasses  (void)       const { return 0;                                                             }
    virtual int order       (void)       const { return ceilintlog2(tspaceDim());                                      }

    virtual int isTrained(void) const { return 0; }
    virtual int isSolGlob(void) const { return 1; } // 1 if solution is global, 0 if solution is a function of initial state
    virtual int isMutable(void) const { return 0; }
    virtual int isPool   (void) const { return 0; }

    virtual double calcDist(const gentype &ha, const gentype &hb, int ia = -1, int db = 2) const { (void) ha; (void) hb; (void) ia; (void) db; NiceThrow("calcDist undefined at this level."); return 0; }

    virtual double calcDistInt(int    ha, int    hb, int ia = -1, int db = 2) const { gentype haa(ha); gentype hbb(hb); return getMLconst().calcDistInt(haa,hbb,ia,db); }
    virtual double calcDistDbl(double ha, double hb, int ia = -1, int db = 2) const { gentype haa(ha); gentype hbb(hb); return getMLconst().calcDistDbl(haa,hbb,ia,db); }

    virtual int isUnderlyingScalar(void) const { return 1; }
    virtual int isUnderlyingVector(void) const { return 0; }
    virtual int isUnderlyingAnions(void) const { return 0; }

    virtual const Vector<int> &ClassLabels       (void)             const { const static Vector<int> temp; return temp;         }
    virtual       int          getInternalClass  (const gentype &)  const {                                return 0;            }
    virtual       int          numInternalClasses(void)             const {                                return numClasses(); }
    virtual       int          isenabled         (int i)            const {                                return d()(i);       }
    virtual       int          isVarDefined      (void)             const {                                return 0;            }

    virtual const int *ClassLabelsInt     (void)  const {                return &(getMLconst().ClassLabels()(0)); }
    virtual       int  getInternalClassInt(int y) const { gentype yy(y); return getInternalClass(yy);             }

    virtual double C        (void) const { return 1/locsigma;        } // Originally there was no sigma here and the code was designed assuming C() was defined locally,
    virtual double sigma    (void) const { return 1.0/C();           } // sigma relied on the base version. We can't change this, so we have the roundabout bit here.
    virtual double sigma_cut(void) const { return DEFAULT_SIGMA_CUT; }
    virtual double eps      (void) const { return DEFAULTEPS;        }
    virtual double Cclass   (int)  const { return 1;                 }
    virtual double epsclass (int)  const { return 1;                 }

    virtual       int      mpri  (void) const { return xmuprior;    }
    virtual const gentype &prival(void) const { return xmuprior_gt; }
    virtual const ML_Base *priml (void) const { return xmuprior_ml; }

    virtual void calcprior   (gentype &res, const SparseVector<gentype> &x, const vecInfo *xinf = nullptr) const;
    virtual void calcallprior(void);

    virtual int    memsize     (void) const { return DEFAULT_MEMSIZE;      }
    virtual double zerotol     (void) const { return globalzerotol;        }
    virtual double Opttol      (void) const { return DEFAULT_OPTTOL;       }
    virtual double Opttolb     (void) const { return DEFAULT_OPTTOL;       }
    virtual double Opttolc     (void) const { return DEFAULT_OPTTOL;       }
    virtual double Opttold     (void) const { return DEFAULT_OPTTOL;       }
    virtual double lr          (void) const { return loclr;                }
    virtual double lrb         (void) const { return loclrb;               }
    virtual double lrc         (void) const { return loclrc;               }
    virtual double lrd         (void) const { return loclrd;               }
    virtual int    maxitcnt    (void) const { return DEFAULT_MAXITCNT;     }
    virtual double maxtraintime(void) const { return DEFAULT_MAXTRAINTIME; }
    virtual double traintimeend(void) const { return DEFAULT_TRAINTIMEEND; }

    virtual int    maxitermvrank(void) const { return DEFAULT_MAXITERMVRANK; }
    virtual double lrmvrank     (void) const { return DEFAULT_LRMVRANK;      }
    virtual double ztmvrank     (void) const { return DEFAULT_ZTMVRANK;      }

    virtual double betarank(void) const { return DEFAULT_BETARANK; }

    virtual double sparlvl(void) const { return 0; }

    virtual const Vector<SparseVector<gentype> > &x          (void) const { return altxsrc ? (*altxsrc).x() : allxdatagent;  }
    virtual const Vector<gentype>                &y          (void) const { return alltraintarg;                             }
    virtual const Vector<double>                 &yR         (void) const { return alltraintargR;                            }
    virtual const Vector<d_anion>                &yA         (void) const { return alltraintargA;                            }
    virtual const Vector<Vector<double> >        &yV         (void) const { return alltraintargV;                            }
    virtual const Vector<gentype>                &yp         (void) const { return alltraintargp;                            }
    virtual const Vector<double>                 &ypR        (void) const { return alltraintargpR;                           }
    virtual const Vector<d_anion>                &ypA        (void) const { return alltraintargpA;                           }
    virtual const Vector<Vector<double> >        &ypV        (void) const { return alltraintargpV;                           }
    virtual const Vector<vecInfo>                &xinfo      (void) const { return altxsrc ? (*altxsrc).xinfo() : traininfo; }
    virtual const Vector<int>                    &xtang      (void) const { return altxsrc ? (*altxsrc).xtang() : traintang; }
    virtual const Vector<int>                    &d          (void) const { return xd;                                       }
    virtual const Vector<double>                 &Cweight    (void) const { return xCweight;                                 }
    virtual const Vector<double>                 &Cweightfuzz(void) const { return xCweightfuzz;                             }
    virtual const Vector<double>                 &sigmaweight(void) const;
    virtual const Vector<double>                 &epsweight  (void) const { return xepsweight;                               }
    virtual const Vector<gentype>                &alphaVal   (void) const { NiceThrow("alphaVal has no meaning here"); static const Vector<gentype> dummy; return dummy; }
    virtual const Vector<int>                    &alphaState (void) const { return xalphaState;                              }

    virtual const SparseVector<gentype> &x       (int i)              const override { return xgetloc(i); }
    virtual const SparseVector<gentype> &x       (int i, int altMLid) const override { kernPrecursor *tmp = nullptr; getaltML(tmp,altMLid);  NiceAssert(tmp); return (*tmp).x(i);  }
    virtual const gentype               &y       (int i)              const          { if ( i >= 0 ) { return y()(i);   } return ytargdata;   }
    virtual       double                 yR      (int i)              const          { if ( i >= 0 ) { return yR()(i);  } return ytargdataR;  }
    virtual const d_anion               &yA      (int i)              const          { if ( i >= 0 ) { return yA()(i);  } return ytargdataA;  }
    virtual const Vector<double>        &yV      (int i)              const          { if ( i >= 0 ) { return yV()(i);  } return ytargdataV;  }
    virtual const gentype               &yp      (int i)              const          { if ( i >= 0 ) { return yp()(i);  } return ytargdatap;  }
    virtual       double                 ypR     (int i)              const          { if ( i >= 0 ) { return ypR()(i); } return ytargdatapR; }
    virtual const d_anion               &ypA     (int i)              const          { if ( i >= 0 ) { return ypA()(i); } return ytargdatapA; }
    virtual const Vector<double>        &ypV     (int i)              const          { if ( i >= 0 ) { return ypV()(i); } return ytargdatapV; }
    virtual const vecInfo               &xinfo   (int i)              const          { return locxinfo(i); }
    virtual       int                    xtang   (int i)              const          { return locxtang(i); }
    virtual       double                 alphaVal(int)                const          { NiceThrow("alphaVal has no meaning here"); return 0.0; }

    virtual int xisrank      (int i)                               const { const SparseVector<gentype> &xres = x(i); return xres.isf1offindpresent() || xres.isf4indpresent(1);  }
    virtual int xisgrad      (int i)                               const { const SparseVector<gentype> &xres = x(i); return xres.isf2offindpresent(); }
    virtual int xisrankorgrad(int i)                               const { const SparseVector<gentype> &xres = x(i); return xres.isf1offindpresent() || xres.isf4indpresent(1) || xres.isf2offindpresent(); }
    virtual int xisclass     (int i, int defaultclass, int q = -1) const { const SparseVector<gentype> &xres = x(i); return ( q == -1 ) ? defaultclass : ( xres.isf4indpresent((100*q)+0) ? ( (int) xres.f4((100*q)+0) ) : defaultclass ); }

    virtual int RFFordata(int) const { return 0; }

    virtual void npCweight    (double **res, int *dim) const { *dim = Cweight().size();     yCweight     = Cweight();     *res = &yCweight("&",0);     }
    virtual void npCweightfuzz(double **res, int *dim) const { *dim = Cweightfuzz().size(); yCweightfuzz = Cweightfuzz(); *res = &yCweightfuzz("&",0); }
    virtual void npsigmaweight(double **res, int *dim) const { *dim = sigmaweight().size(); ysigmaweight = sigmaweight(); *res = &ysigmaweight("&",0); }
    virtual void npepsweight  (double **res, int *dim) const { *dim = epsweight().size();   yepsweight   = epsweight();   *res = &yepsweight("&",0);   }

    virtual int isClassifier(void) const { return 0;               }
    virtual int isRegression(void) const { return !isClassifier(); }
    virtual int isPlanarType(void) const { return 0;               }

    // Random features stuff:
    //
    // NRff:    number of random features
    // NRffRep: number of tasks being completed
    // ReOnly:  0 means features are [ sin cos ], 1 means features are [ cos ]
    // inAdam:  Inner loop method
    //          0 = direct matrix inversion
    //          1 = adam
    //          2 = adam with 2 hotstart
    //          3 = stochastic adam
    //          4 = gradient
    //          5 = stochastic gradient
    //          6 = direct matrix inversion using offNaiveDiagChol
    //          7 = pegasos training
    //          8 = cheats method using standard SVM training with approximated kernel
    // outGrad: 0,2 means optimise outer loop for v^{-1}, 1,3 means optimise for v (1 uses independent chol, 3 borrows chol from svm_scalar)

    virtual int NRff   (void) const { return DEFAULT_NRFF;    }
    virtual int NRffRep(void) const { return 0;               }
    virtual int ReOnly (void) const { return DEFAULT_REONLY;  }
    virtual int inAdam (void) const { return DEFAULT_INADAM;  }
    virtual int outGrad(void) const { return DEFAULT_OUTGRAD; }

    // Version numbers, ML ids etc
    //
    // xvernum(): An integer that gets incremented whenever x is changed in a
    //            non-simple fashion.  Non-simple changes are anything except
    //            adding new vectors to the end of the dataset.
    //            Starts at 0, handy if you want to cache something x related.
    // xvernum(altmlid): gives x version number for a different ML
    // incxvernum(): increments xvernum() (for this ML)
    // gvernum(): An int that gets incremented whenever gh is changed.
    // gvernum(altmlid): gives g version number for a different ML
    // incgvernum(): increments gvernum() (for this ML)
    // getaltML(): get reference to ML with given ID.  Return 0 on success, 1 if nullptr.

    virtual int MLid    (void)                             const override { return kernPrecursor::MLid();                }
    virtual int setMLid (int nv)                                 override;
    virtual int getaltML(kernPrecursor *&res, int altMLid) const override { return kernPrecursor::getaltML(res,altMLid); }

    virtual int xvernum(void) const;
    virtual int gvernum(void) const;

    virtual int xvernum(int altMLid) const;
    virtual int gvernum(int altMLid) const;

    virtual int incxvernum(void);
    virtual int incgvernum(void);

    // Kernel Modification
    //
    // The safe way to modify the kernel is to use k = getKernel() to get a
    // copy of the current kernel, modify the copy, and then use the function
    // setKernel(k) to update (set) the kernel used by the ML.  Which can be
    // slow due to all the copying of kernels required.
    //
    // An faster alternative (unsafe) method is to use getKernel_unsafe to
    // obtain a reference to the actual kernel being used by the ML and modify
    // it directly, then call resetKernel() to force make the ML aware that
    // changes have been made, so for example:
    //
    // SVM_Scalar x
    // ...
    // (x.getKernel_unsafe()).setType(4,1);
    // x.resetKernel();
    //
    // is functionally equivalent to the slower alternative:
    //
    // SVM_Scalar x;
    // MercerKernel k;
    // ...
    // k = x.getKernel();
    // k.setType(4,1);
    // x.setKernel(k);
    //
    // The latter requires 3 calls to MercerKernel's copy constructor
    // plus memory to store k.  The former requires no calls to the copy
    // constructor and no additional memory to store the kernel.
    //
    // The modind argument may be set 0 when calling setKernel or resetKernel
    // provided that indexing has not been switched on, switched off, or the
    // indexes themselves changed.
    //
    // The onlyChangeRowI argument is used to indicate that the change
    // only affects that row.  Set -1 if change affects all rows (default).
    //
    // The updateInfo argument may be used to suppress updating traininfo
    // if not required.  By default this is 1 (do update), set 0 if modind
    // or shift/scale are unchanged (these are the only kernel attributes
    // that will have an impact on traininfo).
    //
    // Note that the kernel is present in all kernel methods.  In non-kernel
    // based methods it must remain linear but may still be used for
    // shifting and scaling of data for normalisation purposes.
    //
    // prepareKernel: if using the resetKernel trick and making changes that
    // don't change x or its inner product (indexing, shift/scale, 8xx kernels)
    // then you can call this first to save information and time.
    //
    // Kalt: evaluate K for alternative kernel function.
    // K2xfer: used to borrow "learnt" kernels
    //
    // 800: Trivial:     K(x,y) = Kx(x,y)
    // 801: m-inner:     K(x,y) = sum_ij a_i a_j Kx(x_i,x_j,x,y)
    //                   (eg. for SVM a_i = alpha_i)
    // 802: Moment:      K(x,y) = sum_ij a_i a_j Kx(x_i,x_j)
    //                   (eg. for SVM a_i = alpha_i)
    // 803: k-learn:     K(x,y) = sum_ij a_i Kx(x_i,(x,y))  - indices not passed through
    //                   (eg. for SVM a_i = alpha_i.  Typically x_i = (xa_i,xb_i))
    // 804: k-learn:     K(x,y) = sum_ij a_i Kx(x_i,(x,y))  - indices passed through
    //                   (eg. for SVM a_i = alpha_i.  Typically x_i = (xa_i,xb_i))
    // 805: k2-learn:    K(x,y) = (sum_i a_i Kx(x_i,(x,y)))^2  - indices not passed through
    //                   (eg. for SVM a_i = alpha_i.  Typically x_i = (xa_i,xb_i))
    // 806: MLN:         K(x,y) = K(f(x),f(y))
    // 807: hyperkernel: K(x,y) = sum_{i,j} alpha_i alpha_j 1/(1+K(x,y)K(x_i,x_j))
    //                   (eg Ong's paper on hyperkernels)
    // 808: RFF learn:   K(x,y) = \sum_q v_q phi_q(x_i) phi_q(x_j)
    //                   (eg Ong's paper on hyperkernels)
    // 81x: like above but with indice pass-through
    //
    // k2diag: like K2(i,i), but unlike K2 (which always calculates fresh), this can be
    //         cached through eg kerndiag in svm_generic.
    //
    // fillCache: runs through all vectors and calls K(res,i,j).  This is handy
    //            to pre-fill any kernel caches (blk type, not Gp type).
    //
    // K2bypass: if matrix (of non-zero size) is put here then K2 function with i,j>=0
    //           is taken directly from this matrix
    //
    // Notes: - K4 does not calculate gradient constraints
    //        - Km does not calculate gradient constraints or rank constraints
    //        - Km assumes any nullptrs in xx (and xzinfo) are at the end, not the start
    //        - K4 if xa == xb == nullptr, xc == xd given then assumes xc := xa.*xb, xd := xc.xd
    //        - K2ip calculates the inner product as used by the kernel calculation.
    //
    // Polymorphing K2xfer:
    //
    // - pass kernel 800 back to ML_Base.
    // - always add elements to end of i,xx,xzinfo.
    // - if you change the ordering of these remember to change them back!
    //
    // Gradients:
    //
    // - dK calculates gradient wrt <x,y> (or K2xfer(x,y)) and ||x||^2 (or K2xfer(x,x))
    //   (set deepDeriv 1 for derivative of <x,y>, ||x|| regardless)
    // - dK2delx calculates gradient wrt x and returns result as:
    //   xscaleres.x + yscaleres.y   (or xscaleres.x(minmaxind) + yscaleres.y(minmaxind)
    //                                if minmaxind >= 0 in result)
    // - d2K2delxdelx calculates the drivative d/dx d/dx K(x,y) and returns result as:
    //   xxscaleres.x.x' + xyscaleres.x.y' + yxscaleres.y.x' + yyscaleres.y.y' + constres.I
    // - d2K2delxdely calculates the drivative d/dx d/dy K(x,y) and returns result as:
    //   xxscaleres.x.x' + xyscaleres.x.y' + yxscaleres.y.x' + yyscaleres.y.y' + constres.I
    // - dnK2del calculates the derivative:
    //   d/dxq0 d/dxq1 ... K(x0,x1)
    //   and returns a vectorised result of the form:
    //   sum_i sc_i kronprod_j [ x{nn_ij}   if nn_ij == 0,1
    //                         [ kd{nn_ij}  if nn_ij < 0
    //   where kd{a} kd_{a} is the vectorised identity matrix
    //
    // setKreal/setKunreal: in rare cases (eg SVM_Vector) the kernel can evaluate
    //     as a matrix to model interactions between different axis of the vector.
    //
    //
    //
    // tuneKernel: Generic kernel tuning function
    //
    // method: 1 = max-likelihood
    //         2 = loo error
    //         3 = recall
    // xwidth: maximum length-scale
    // tuneK:  0 = don't tune kernel
    //         1 = tune kernel
    // tuneP:  0 = don't tune model parameters
    //         1 = tune model parameter C (NOT IMPLEMENTED YET)
    //         2 = tune model parameter eps (NOT IMPLEMENTED YET)
    //         3 = tune model parameters C and eps (NOT IMPLEMENTED YET)
    // tunebounds: optional lower and upper bound overrides for tuning. See datatypes
    //
    // Note that this is very basic.  It tunes continuous variables
    // only, parameter bounds are arbitrary, and grid sizes fixed
    //
    // Returns best error

    virtual const MercerKernel &getKernel       (void) const { return kernel; }
    virtual       MercerKernel &getKernel_unsafe(void)       { return kernel; }
    virtual       void          prepareKernel   (void)       {                }

    virtual double tuneKernel(int method, double xwidth, int tuneK = 1, int tuneP = 0, const tkBounds *tunebounds = nullptr);

    virtual int resetKernel(                             int modind = 1, int onlyChangeRowI = -1, int updateInfo = 1);
    virtual int setKernel  (const MercerKernel &xkernel, int modind = 1, int onlyChangeRowI = -1                    );

    virtual int isKreal  (void) const { return 1; }
    virtual int isKunreal(void) const { return 0; }

    virtual int setKreal  (void) {                                                         return 0; }
    virtual int setKunreal(void) { NiceThrow("Can't set unreal kernels for this ML type"); return 1; }

    virtual double k2diag(int ia) const { return K2(ia,ia); }

    virtual void fillCache(int Ns = 0, int Ne = -1);

    virtual void K2bypass(const Matrix<gentype> &nv) { K2mat = nv; }

    virtual gentype &Keqn(gentype &res,                           int resmode = 1) const;
    virtual gentype &Keqn(gentype &res, const MercerKernel &altK, int resmode = 1) const;

    virtual gentype &K1(gentype &res, const SparseVector<gentype> &xa, const vecInfo *xainf = nullptr) const { setInnerWildpa(&xa,xainf); K1(res,-1); resetInnerWildp(( xainf == nullptr )); return res; }
    virtual gentype &K2(gentype &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo *xainf = nullptr, const vecInfo *xbinf = nullptr) const { setInnerWildpa(&xa,xainf); setInnerWildpb(&xb,xbinf); K2(res,-1,-3); resetInnerWildp(( xainf == nullptr ),( xbinf == nullptr )); return res; }
    virtual gentype &K3(gentype &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo *xainf = nullptr, const vecInfo *xbinf = nullptr, const vecInfo *xcinf = nullptr) const { setInnerWildpa(&xa,xainf); setInnerWildpb(&xb,xbinf); setInnerWildpc(&xc,xcinf); K3(res,-1,-3,-4);  resetInnerWildp(( xainf == nullptr ),( xbinf == nullptr ),( xcinf == nullptr )); return res; }
    virtual gentype &K4(gentype &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo *xainf = nullptr, const vecInfo *xbinf = nullptr, const vecInfo *xcinf = nullptr, const vecInfo *xdinf = nullptr) const { setInnerWildpa(&xa,xainf); setInnerWildpb(&xb,xbinf); setInnerWildpc(&xc,xcinf); setInnerWildpd(&xd,xdinf); K4(res,-1,-3,-4,-5); resetInnerWildp(( xainf == nullptr ),( xbinf == nullptr ),(xcinf == nullptr),(xdinf == nullptr)); return res; }
    virtual gentype &Km(gentype &res, const Vector<SparseVector<gentype> > &xx) const { int m = xx.size(); setInnerWildpx(&xx); retVector<int> tmpva; Vector<int> ii(cntintvec(m,tmpva)); ii += 1; ii *= -100; Km(m,res,ii); resetInnerWildp(); return res; }

    virtual double  K2ip(const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo *xainf = nullptr, const vecInfo *xbinf = nullptr) const { setInnerWildpa(&xa,xainf); setInnerWildpb(&xb,xbinf); double res = K2ip(-1,-3,0.0); resetInnerWildp(( xainf == nullptr ),( xbinf == nullptr )); return res; }
    virtual double distK(const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo *xainf = nullptr, const vecInfo *xbinf = nullptr) const { setInnerWildpa(&xa,xainf); setInnerWildpb(&xb,xbinf); double res = distK(-1,-3); resetInnerWildp(( xainf == nullptr ),( xbinf == nullptr )); return res; }

    virtual Vector<gentype> &phi2(Vector<gentype> &res,         const SparseVector<gentype> &xa,           const vecInfo *xainf = nullptr) const { setInnerWildpa(&xa,xainf); phi2(res,-1); resetInnerWildp(( xainf == nullptr )); return res; }
    virtual Vector<gentype> &phi2(Vector<gentype> &res, int ia, const SparseVector<gentype> *xa = nullptr, const vecInfo *xainf = nullptr) const;

    virtual Vector<double> &phi2(Vector<double> &res,         const SparseVector<gentype> &xa,           const vecInfo *xainf = nullptr) const { setInnerWildpa(&xa,xainf); phi2(res,-1); resetInnerWildp(( xainf == nullptr )); return res; }
    virtual Vector<double> &phi2(Vector<double> &res, int ia, const SparseVector<gentype> *xa = nullptr, const vecInfo *xainf = nullptr) const;

    virtual double K0ip(                                const gentype **pxyprod = nullptr)                                                                                                                                                                                                                                                                                                                 const { return KK0ip(0.0,pxyprod);                                                     }
    virtual double K1ip(int ia,                         const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr,                                                                                                                                  const vecInfo *xainfo = nullptr)                                                                                                    const { return KK1ip(ia,0.0,pxyprod,xa,xainfo);                                        }
    virtual double K2ip(int ia, int ib,                 const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr,                                                                                       const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr)                                                                   const { return KK2ip(ia,ib,0.0,pxyprod,xa,xb,xainfo,xbinfo);                           }
    virtual double K3ip(int ia, int ib, int ic,         const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const SparseVector<gentype> *xc = nullptr,                                            const vecInfo *xainfo = nullptr,           const vecInfo *xbinfo = nullptr, const vecInfo *xcinfo = nullptr)                        const { return KK3ip(ia,ib,ic,0.0,pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo);              }
    virtual double K4ip(int ia, int ib, int ic, int id, const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const SparseVector<gentype> *xc = nullptr, const SparseVector<gentype> *xd = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, const vecInfo *xcinfo = nullptr, const vecInfo *xdinfo = nullptr) const { return KK4ip(ia,ib,ic,id,0.0,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo); }
    virtual double Kmip(int m, Vector<int> &i, const gentype **pxyprod = nullptr, Vector<const SparseVector<gentype> *> *xx = nullptr, Vector<const vecInfo *> *xzinfo = nullptr) const { return KKmip(m,i,0.0,pxyprod,xx,xzinfo); }

    virtual double K0ip(                                double bias, const gentype **pxyprod = nullptr)                                                                                                                                                                                                                                                                                                                 const { return KK0ip(bias,pxyprod); }
    virtual double K1ip(int ia,                         double bias, const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr,                                                                                                                                  const vecInfo *xainfo = nullptr)                                                                                                    const { return KK1ip(ia,bias,pxyprod,xa,xainfo); }
    virtual double K2ip(int ia, int ib,                 double bias, const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr,                                                                                       const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr)                                                                   const { return KK2ip(ia,ib,bias,pxyprod,xa,xb,xainfo,xbinfo); }
    virtual double K3ip(int ia, int ib, int ic,         double bias, const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const SparseVector<gentype> *xc = nullptr,                                            const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, const vecInfo *xcinfo = nullptr)                                  const { return KK3ip(ia,ib,ic,bias,pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo); }
    virtual double K4ip(int ia, int ib, int ic, int id, double bias, const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const SparseVector<gentype> *xc = nullptr, const SparseVector<gentype> *xd = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, const vecInfo *xcinfo = nullptr, const vecInfo *xdinfo = nullptr) const { return KK4ip(ia,ib,ic,id,bias,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo); }
    virtual double Kmip(int m, Vector<int> &i, double bias, const gentype **pxyprod = nullptr, Vector<const SparseVector<gentype> *> *xx = nullptr, Vector<const vecInfo *> *xzinfo = nullptr) const { return KKmip(m,i,bias,pxyprod,xx,xzinfo); }

    virtual gentype        &K0(              gentype        &res                          , const gentype **pxyprod = nullptr, int resmode = 0) const;
    virtual gentype        &K0(              gentype        &res, const gentype &bias     , const gentype **pxyprod = nullptr, int resmode = 0) const;
    virtual gentype        &K0(              gentype        &res, const MercerKernel &altK, const gentype **pxyprod = nullptr, int resmode = 0) const;
    virtual double          K0(                                                             const gentype **pxyprod = nullptr, int resmode = 0) const;
    virtual Matrix<double> &K0(int spaceDim, Matrix<double> &res                          , const gentype **pxyprod = nullptr, int resmode = 0) const;
    virtual d_anion        &K0(int order,    d_anion        &res                          , const gentype **pxyprod = nullptr, int resmode = 0) const;

    virtual gentype        &K1(              gentype        &res, int ia                          , const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const vecInfo *xainfo = nullptr, int resmode = 0) const;
    virtual gentype        &K1(              gentype        &res, int ia, const gentype &bias     , const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const vecInfo *xainfo = nullptr, int resmode = 0) const;
    virtual gentype        &K1(              gentype        &res, int ia, const MercerKernel &altK, const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const vecInfo *xainfo = nullptr, int resmode = 0) const;
    virtual double          K1(                                   int ia                          , const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const vecInfo *xainfo = nullptr, int resmode = 0) const;
    virtual Matrix<double> &K1(int spaceDim, Matrix<double> &res, int ia                          , const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const vecInfo *xainfo = nullptr, int resmode = 0) const;
    virtual d_anion        &K1(int order,    d_anion        &res, int ia                          , const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const vecInfo *xainfo = nullptr, int resmode = 0) const;

    virtual gentype        &K2(              gentype        &res, int ia, int ib                          , const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, int resmode = 0) const;
    virtual gentype        &K2(              gentype        &res, int ia, int ib, const gentype &bias     , const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, int resmode = 0) const;
    virtual gentype        &K2(              gentype        &res, int ia, int ib, const MercerKernel &altK, const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, int resmode = 0) const;
    virtual double          K2(                                   int ia, int ib                          , const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, int resmode = 0) const;
    virtual Matrix<double> &K2(int spaceDim, Matrix<double> &res, int ia, int ib                          , const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, int resmode = 0) const;
    virtual d_anion        &K2(int order,    d_anion        &res, int ia, int ib                          , const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, int resmode = 0) const;

    virtual gentype        &K2x2(              gentype        &res, int i, int ia, int ib                          , const SparseVector<gentype> *x = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const vecInfo *xinfo = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, int resmode = 0) const;
    virtual gentype        &K2x2(              gentype        &res, int i, int ia, int ib, const gentype &bias     , const SparseVector<gentype> *x = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const vecInfo *xinfo = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, int resmode = 0) const;
    virtual gentype        &K2x2(              gentype        &res, int i, int ia, int ib, const MercerKernel &altK, const SparseVector<gentype> *x = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const vecInfo *xinfo = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, int resmode = 0) const;
    virtual double          K2x2(                                   int i, int ia, int ib                          , const SparseVector<gentype> *x = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const vecInfo *xinfo = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, int resmode = 0) const;
    virtual Matrix<double> &K2x2(int spaceDim, Matrix<double> &res, int i, int ia, int ib                          , const SparseVector<gentype> *x = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const vecInfo *xinfo = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, int resmode = 0) const;
    virtual d_anion        &K2x2(int order,    d_anion        &res, int i, int ia, int ib                          , const SparseVector<gentype> *x = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const vecInfo *xinfo = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, int resmode = 0) const;

    virtual gentype        &K3(              gentype        &res, int ia, int ib, int ic                          , const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const SparseVector<gentype> *xc = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, const vecInfo *xcinfo = nullptr, int resmode = 0) const;
    virtual gentype        &K3(              gentype        &res, int ia, int ib, int ic, const gentype &bias     , const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const SparseVector<gentype> *xc = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, const vecInfo *xcinfo = nullptr, int resmode = 0) const;
    virtual gentype        &K3(              gentype        &res, int ia, int ib, int ic, const MercerKernel &altK, const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const SparseVector<gentype> *xc = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, const vecInfo *xcinfo = nullptr, int resmode = 0) const;
    virtual double          K3(                                   int ia, int ib, int ic                          , const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const SparseVector<gentype> *xc = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, const vecInfo *xcinfo = nullptr, int resmode = 0) const;
    virtual Matrix<double> &K3(int spaceDim, Matrix<double> &res, int ia, int ib, int ic                          , const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const SparseVector<gentype> *xc = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, const vecInfo *xcinfo = nullptr, int resmode = 0) const;
    virtual d_anion        &K3(int order,    d_anion        &res, int ia, int ib, int ic                          , const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const SparseVector<gentype> *xc = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, const vecInfo *xcinfo = nullptr, int resmode = 0) const;

    virtual gentype        &K4(              gentype        &res, int ia, int ib, int ic, int id                          , const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const SparseVector<gentype> *xc = nullptr, const SparseVector<gentype> *xd = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, const vecInfo *xcinfo = nullptr, const vecInfo *xdinfo = nullptr, int resmode = 0) const;
    virtual gentype        &K4(              gentype        &res, int ia, int ib, int ic, int id, const gentype &bias     , const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const SparseVector<gentype> *xc = nullptr, const SparseVector<gentype> *xd = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, const vecInfo *xcinfo = nullptr, const vecInfo *xdinfo = nullptr, int resmode = 0) const;
    virtual gentype        &K4(              gentype        &res, int ia, int ib, int ic, int id, const MercerKernel &altK, const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const SparseVector<gentype> *xc = nullptr, const SparseVector<gentype> *xd = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, const vecInfo *xcinfo = nullptr, const vecInfo *xdinfo = nullptr, int resmode = 0) const;
    virtual double          K4(                                   int ia, int ib, int ic, int id                          , const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const SparseVector<gentype> *xc = nullptr, const SparseVector<gentype> *xd = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, const vecInfo *xcinfo = nullptr, const vecInfo *xdinfo = nullptr, int resmode = 0) const;
    virtual Matrix<double> &K4(int spaceDim, Matrix<double> &res, int ia, int ib, int ic, int id                          , const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const SparseVector<gentype> *xc = nullptr, const SparseVector<gentype> *xd = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, const vecInfo *xcinfo = nullptr, const vecInfo *xdinfo = nullptr, int resmode = 0) const;
    virtual d_anion        &K4(int order,    d_anion        &res, int ia, int ib, int ic, int id                          , const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const SparseVector<gentype> *xc = nullptr, const SparseVector<gentype> *xd = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, const vecInfo *xcinfo = nullptr, const vecInfo *xdinfo = nullptr, int resmode = 0) const;

    virtual gentype        &Km(int m              , gentype        &res, Vector<int> &i                          , const gentype **pxyprod = nullptr, Vector<const SparseVector<gentype> *> *xx = nullptr, Vector<const vecInfo *> *xzinfo = nullptr, int resmode = 0) const;
    virtual gentype        &Km(int m              , gentype        &res, Vector<int> &i, const gentype &bias     , const gentype **pxyprod = nullptr, Vector<const SparseVector<gentype> *> *xx = nullptr, Vector<const vecInfo *> *xzinfo = nullptr, int resmode = 0) const;
    virtual gentype        &Km(int m              , gentype        &res, Vector<int> &i, const MercerKernel &altK, const gentype **pxyprod = nullptr, Vector<const SparseVector<gentype> *> *xx = nullptr, Vector<const vecInfo *> *xzinfo = nullptr, int resmode = 0) const;
    virtual double          Km(int m              ,                      Vector<int> &i                          , const gentype **pxyprod = nullptr, Vector<const SparseVector<gentype> *> *xx = nullptr, Vector<const vecInfo *> *xzinfo = nullptr, int resmode = 0) const;
    virtual Matrix<double> &Km(int m, int spaceDim, Matrix<double> &res, Vector<int> &i                          , const gentype **pxyprod = nullptr, Vector<const SparseVector<gentype> *> *xx = nullptr, Vector<const vecInfo *> *xzinfo = nullptr, int resmode = 0) const;
    virtual d_anion        &Km(int m, int order   , d_anion        &res, Vector<int> &i                          , const gentype **pxyprod = nullptr, Vector<const SparseVector<gentype> *> *xx = nullptr, Vector<const vecInfo *> *xzinfo = nullptr, int resmode = 0) const;

    virtual void dK(gentype &xygrad, gentype &xnormgrad, int ia, int ib, const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, int deepDeriv = 0) const;
    virtual void dK(double  &xygrad, double  &xnormgrad, int ia, int ib, const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, int deepDeriv = 0) const;

    virtual void d2K(gentype &xygrad, gentype &xnormgrad, gentype &xyxygrad, gentype &xyxnormgrad, gentype &xyynormgrad, gentype &xnormxnormgrad, gentype &xnormynormgrad, gentype &ynormynormgrad, int &minmaxind, int i, int j, const gentype **pxyprod = nullptr, const SparseVector<gentype> *xx = nullptr, const SparseVector<gentype> *yy = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *yyinfo = nullptr) const;
    virtual void d2K(double  &xygrad, double  &xnormgrad, double  &xyxygrad, double  &xyxnormgrad, double  &xyynormgrad, double  &xnormxnormgrad, double  &xnormynormgrad, double  &ynormynormgrad, int &minmaxind, int i, int j, const gentype **pxyprod = nullptr, const SparseVector<gentype> *xx = nullptr, const SparseVector<gentype> *yy = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *yyinfo = nullptr) const;

    virtual void dK2delx(gentype &xscaleres, gentype &yscaleres, int &minmaxind, int i, int j, const gentype **pxyprod = nullptr, const SparseVector<gentype> *xx = nullptr, const SparseVector<gentype> *yy = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *yyinfo = nullptr) const;
    virtual void dK2delx(double  &xscaleres, double  &yscaleres, int &minmaxind, int i, int j, const gentype **pxyprod = nullptr, const SparseVector<gentype> *xx = nullptr, const SparseVector<gentype> *yy = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *yyinfo = nullptr) const;

    virtual void d2K2delxdelx(gentype &xxscaleres, gentype &yyscaleres, gentype &xyscaleres, gentype &yxscaleres, gentype &constres, int &minmaxind, int i, int j, const gentype **pxyprod = nullptr, const SparseVector<gentype> *xx = nullptr, const SparseVector<gentype> *yy = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *yyinfo = nullptr) const;
    virtual void d2K2delxdely(gentype &xxscaleres, gentype &yyscaleres, gentype &xyscaleres, gentype &yxscaleres, gentype &constres, int &minmaxind, int i, int j, const gentype **pxyprod = nullptr, const SparseVector<gentype> *xx = nullptr, const SparseVector<gentype> *yy = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *yyinfo = nullptr) const;

    virtual void d2K2delxdelx(double  &xxscaleres, double  &yyscaleres, double  &xyscaleres, double  &yxscaleres, double  &constres, int &minmaxind, int i, int j, const gentype **pxyprod = nullptr, const SparseVector<gentype> *xx = nullptr, const SparseVector<gentype> *yy = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *yyinfo = nullptr) const;
    virtual void d2K2delxdely(double  &xxscaleres, double  &yyscaleres, double  &xyscaleres, double  &yxscaleres, double  &constres, int &minmaxind, int i, int j, const gentype **pxyprod = nullptr, const SparseVector<gentype> *xx = nullptr, const SparseVector<gentype> *yy = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *yyinfo = nullptr) const;

    virtual void dnK2del(Vector<gentype> &sc, Vector<Vector<int> > &n, int &minmaxind, const Vector<int> &q, int i, int j, const gentype **pxyprod = nullptr, const SparseVector<gentype> *xx = nullptr, const SparseVector<gentype> *yy = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *yyinfo = nullptr) const;
    virtual void dnK2del(Vector<double>  &sc, Vector<Vector<int> > &n, int &minmaxind, const Vector<int> &q, int i, int j, const gentype **pxyprod = nullptr, const SparseVector<gentype> *xx = nullptr, const SparseVector<gentype> *yy = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *yyinfo = nullptr) const;

    virtual double distK(int i, int j) const;

    virtual void densedKdx(double &res, int i, int j) const { return densedKdx(res,i,j,0.0); }
    virtual void denseintK(double &res, int i, int j) const { return denseintK(res,i,j,0.0); }

    virtual void densedKdx(double &res, int i, int j, double bias) const;
    virtual void denseintK(double &res, int i, int j, double bias) const;

    virtual void ddistKdx(double &xscaleres, double &yscaleres, int &minmaxind, int i, int j) const;

    virtual int isKVarianceNZ(void) const override;

    virtual void K0xfer(                                    gentype &res, int &minmaxind, int typeis, const gentype &xyprod, const gentype &yxprod, const gentype &diffis, int xdim, int densetype, int resmode, int mlid) const override;
    virtual void K1xfer(                                    gentype &res, int &minmaxind, int typeis, const gentype &xyprod, const gentype &yxprod, const gentype &diffis, const SparseVector<gentype> &xa, const vecInfo &xainfo, int ia, int xdim, int densetype, int resmode, int mlid) const override;
    virtual void K2xfer(gentype &dxyprod, gentype &ddiffis, gentype &res, int &minmaxind, int typeis, const gentype &xyprod, const gentype &yxprod, const gentype &diffis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, int ia, int ib, int xdim, int densetype, int resmode, int mlid) const override;
    virtual void K3xfer(                                    gentype &res, int &minmaxind, int typeis, const gentype &xyprod, const gentype &yxprod, const gentype &diffis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, int ia, int ib, int ic, int xdim, int densetype, int resmode, int mlid) const override;
    virtual void K4xfer(                                    gentype &res, int &minmaxind, int typeis, const gentype &xyprod, const gentype &yxprod, const gentype &diffis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, int ia, int ib, int ic, int id, int xdim, int densetype, int resmode, int mlid) const override;
    virtual void Kmxfer(                                    gentype &res, int &minmaxind, int typeis, const gentype &xyprod, const gentype &yxprod, const gentype &diffis, Vector<const SparseVector<gentype> *> &x, Vector<const vecInfo *> &xzinfo, Vector<int> &i, int xdim, int m, int densetype, int resmode, int mlid) const override;

    virtual void K0xfer(                                  double &res, int &minmaxind, int typeis, double xyprod, double yxprod, double diffis,                                                                                                                                                                                                                                                                 int xdim, int densetype, int resmode, int mlid) const override;
    virtual void K1xfer(                                  double &res, int &minmaxind, int typeis, double xyprod, double yxprod, double diffis, const SparseVector<gentype> &xa,                                                                                                    const vecInfo &xainfo,                                                                      int ia,                         int xdim, int densetype, int resmode, int mlid) const override;
    virtual void K2xfer(double &dxyprod, double &ddiffis, double &res, int &minmaxind, int typeis, double xyprod, double yxprod, double diffis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,                                                                   const vecInfo &xainfo, const vecInfo &xbinfo,                                               int ia, int ib,                 int xdim, int densetype, int resmode, int mlid) const override;
    virtual void K3xfer(                                  double &res, int &minmaxind, int typeis, double xyprod, double yxprod, double diffis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc,                                  const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo,                        int ia, int ib, int ic,         int xdim, int densetype, int resmode, int mlid) const override;
    virtual void K4xfer(                                  double &res, int &minmaxind, int typeis, double xyprod, double yxprod, double diffis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, int ia, int ib, int ic, int id, int xdim, int densetype, int resmode, int mlid) const override;
    virtual void Kmxfer(                                  double &res, int &minmaxind, int typeis, double xyprod, double yxprod, double diffis, Vector<const SparseVector<gentype> *> &x, Vector<const vecInfo *> &xzinfo, Vector<int> &i, int xdim, int m, int densetype, int resmode, int mlid) const override;

    virtual const gentype &xelm    (gentype &res, int i, int j) const;
    virtual       int      xindsize(int i)                      const;

    // Training set modification:
    //
    // addTrainingVector:  add training vector to training set.
    // qaddTrainingVector: like addTrainingVector, but uses qswap for speed.
    //
    // removeTraingVector: remove training vector from training set.
    //
    // NB: - x is not preserved by qaddTrainingVector
    //     - if x,y are included in removeTrainingVector then these are
    //       qswapped out of data before removal
    //     - all functions that modify the ML return 0 if the trained machine
    //       is unchanged, 1 otherwise
    //
    // qswapx swaps vectors rather than overwriting them
    //  - set dontupdate to prevent any updates being processed
    //  - see also assumeConsistentX if you want fast.
    //
    // x can also be accessed directly with the x_unsafe function.
    // If any changes are made you also need to call update_x.
    //
    // Optimality note: if x vectors have 3ent indices (ie non-sparse,
    // same dimension, or sparse but with the same zeros and non-zeros for
    // all cases) then you can set assumeConsistentX.  This will only set
    // or change indexKey when vector x(0) is modified, and will lead to
    // speedups elsewhere.

    virtual int addTrainingVector (int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2);
    virtual int qaddTrainingVector(int i, const gentype &y,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2);

    virtual int addTrainingVector (int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);
    virtual int qaddTrainingVector(int i, const Vector<gentype> &y,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);

    virtual int removeTrainingVector(int i                                      ) { SparseVector<gentype> x; gentype y; return removeTrainingVector(i,y,x); }
    virtual int removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x);
    virtual int removeTrainingVector(int i, int num                             );

    virtual int setx(int                i, const SparseVector<gentype>          &x);
    virtual int setx(const Vector<int> &i, const Vector<SparseVector<gentype> > &x);
    virtual int setx(                      const Vector<SparseVector<gentype> > &x);

    virtual int qswapx(int                i, SparseVector<gentype>          &x, int dontupdate = 0);
    virtual int qswapx(const Vector<int> &i, Vector<SparseVector<gentype> > &x, int dontupdate = 0);
    virtual int qswapx(                      Vector<SparseVector<gentype> > &x, int dontupdate = 0);

    virtual int sety(int                i, const gentype         &y);
    virtual int sety(const Vector<int> &i, const Vector<gentype> &y);
    virtual int sety(                      const Vector<gentype> &y);

    virtual int sety(int                i, double                z);
    virtual int sety(const Vector<int> &i, const Vector<double> &z);
    virtual int sety(                      const Vector<double> &z);

    virtual int sety(int                i, const Vector<double>          &z);
    virtual int sety(const Vector<int> &i, const Vector<Vector<double> > &z);
    virtual int sety(                      const Vector<Vector<double> > &z);

    virtual int sety(int                i, const d_anion         &z);
    virtual int sety(const Vector<int> &i, const Vector<d_anion> &z);
    virtual int sety(                      const Vector<d_anion> &z);

    virtual int setd(int                i, int                d);
    virtual int setd(const Vector<int> &i, const Vector<int> &d);
    virtual int setd(                      const Vector<int> &d);

    virtual int setCweight(int                i, double                nv);
    virtual int setCweight(const Vector<int> &i, const Vector<double> &nv);
    virtual int setCweight(                      const Vector<double> &nv);

    virtual int setCweightfuzz(int                i, double                nv);
    virtual int setCweightfuzz(const Vector<int> &i, const Vector<double> &nv);
    virtual int setCweightfuzz(                      const Vector<double> &nv);

    virtual int setsigmaweight(int                i, double                nv);
    virtual int setsigmaweight(const Vector<int> &i, const Vector<double> &nv);
    virtual int setsigmaweight(                      const Vector<double> &nv);

    virtual int setepsweight(int                i, double                nv);
    virtual int setepsweight(const Vector<int> &i, const Vector<double> &nv);
    virtual int setepsweight(                      const Vector<double> &nv);

    virtual int scaleCweight    (double s);
    virtual int scaleCweightfuzz(double s);
    virtual int scalesigmaweight(double s) { return scaleCweight(1/s); }
    virtual int scaleepsweight  (double s);

    virtual void assumeConsistentX  (void) { xassumedconsist = 1; xconsist = 0;              }
    virtual void assumeInconsistentX(void) { xassumedconsist = 0; xconsist = testxconsist(); }

    virtual int isXConsistent       (void) const { return xassumedconsist || xconsist; }
    virtual int isXAssumedConsistent(void) const { return xassumedconsist;             }

    virtual void xferx(const ML_Base &xsrc);

    // Generic target controls: in some generic target classes the output
    // is restricted to lie in the span of a particular basis.  In this case
    // these functions control contents of this basis.  The output kernel
    // specifies the similarity measure between basis elements.
    //
    // basisType: 0 = gentype basis defined by user
    //            1 = basis same as y() vector
    //
    // setBasis(n,d) sets random 1-norm unit basis of n elements, each a
    // d-dimensional real-valued 1-norm unit vector.

    virtual int NbasisUU   (void) const { return locbasisUU.size(); }
    virtual int basisTypeUU(void) const { return isBasisUserUU;     }
    virtual int defProjUU  (void) const { return defbasisUU;        }

    virtual const Vector<gentype> &VbasisUU(void) const { return locbasisUU; }

    virtual int setBasisYUU           (void);
    virtual int setBasisUUU           (void);
    virtual int addToBasisUU          (int i, const gentype &o);
    virtual int removeFromBasisUU     (int i);
    virtual int setBasisUU            (int i, const gentype &o);
    virtual int setBasisUU            (const Vector<gentype> &o);
    virtual int setDefaultProjectionUU(int d)                    { int res = defbasisUU; defbasisUU = d; return res; }
    virtual int setBasisUU            (int n, int d);

    virtual int NbasisVV   (void) const { return locbasisVV.size(); }
    virtual int basisTypeVV(void) const { return isBasisUserVV;     }
    virtual int defProjVV  (void) const { return defbasisVV;        }

    virtual const Vector<gentype> &VbasisVV(void) const { return locbasisVV; }

    virtual int setBasisYVV           (void);
    virtual int setBasisUVV           (void);
    virtual int addToBasisVV          (int i, const gentype &o);
    virtual int removeFromBasisVV     (int i);
    virtual int setBasisVV            (int i, const gentype &o);
    virtual int setBasisVV            (const Vector<gentype> &o);
    virtual int setDefaultProjectionVV(int d)                    { int res = defbasisVV; defbasisVV = d; return res; }
    virtual int setBasisVV            (int n, int d);

    virtual const MercerKernel &getUUOutputKernel       (void)                                        const { return UUoutkernel;                                     }
    virtual       MercerKernel &getUUOutputKernel_unsafe(void)                                              { return UUoutkernel;                                     }
    virtual int                 resetUUOutputKernel     (int modind = 1)                                    { return setUUOutputKernel(getUUOutputKernel(),modind);   }
    virtual int                 setUUOutputKernel       (const MercerKernel &xkernel, int modind = 1)       { UUoutkernel = xkernel; return resetKernel(modind,-1,0); }

    // RFF Similarity in random feature space

    virtual const MercerKernel &getRFFKernel       (void)                                        const { return RFFkernel;                                     }
    virtual       MercerKernel &getRFFKernel_unsafe(void)                                              { return RFFkernel;                                     }
    virtual int                 resetRFFKernel     (int modind = 1)                                    { return setRFFKernel(getRFFKernel(),modind);           }
    virtual int                 setRFFKernel       (const MercerKernel &xkernel, int modind = 1)       { RFFkernel = xkernel; return resetKernel(modind,-1,0); }

    // General modification and autoset functions
    //
    // scale:   scale y and K
    // reset:   set w = 0, b = 0, start as per starting state
    // restart: reset to state immediately after construction
    // home:    for serial/parallel blocks this sets the active element to
    //          -1, which is the parent (overall) view.
    //
    // set...: set various things
    //
    // randomise: randomise "weights" to uniform with given fraction set zero
    //            (sparsity).  If sparsity > 0 then ranges are [0,1 or C] for
    //            positive weights, [-1 or -C,0] for negative weights, [-1 or
    //            -C,1 or C] for unconstrained weights, where C may be used if
    //            the magnitude of the weight is constrained.  If sparsity is
    //            < 0 then |sparsity| is used and unconstrained weights are
    //            selected in [0,1 or C] (see eg xferml for usage).
    // autoen: set targets equal to inputs and train.  If output dimension
    //         does not match input then trim or pad with zeros.  If output
    //         type does not match input type then attempt "closest copy"
    //         (no guarantees that this means anything).
    // renormalise: randomisation can lead to very large training outputs.
    //            this scales so that training outputs range from zero to 1
    // realign:   set targets equal to output of system for training set.
    //
    // addxspaceFeat/removexspaceFeat: These functions are for adding and
    // removing dimensions from input space.  Now input sparse is made up of
    // sparse features, so their dimensionality is actually undefined.
    // However the features used by the training set are well defined, and
    // in some cases it is important to know which of these are being used
    // (for neural nets, for example, need to know this to set up the
    // weights).  By default these functions do nothing.
    //
    // When vectors are added/removed the code checks for changes to input
    // space and calls these functions to reflect changes.  They can be
    // polymorphed by child functions where this knowledge is important.

    virtual int randomise  (double sparsity) { (void) sparsity; return 0; }
    virtual int autoen     (void);
    virtual int renormalise(void);
    virtual int realign    (void);

    virtual int setzerotol     (double zt);
    virtual int setOpttol      (double xopttol)       { (void) xopttol;       return 0; }
    virtual int setOpttolb     (double xopttol)       { (void) xopttol;       return 0; }
    virtual int setOpttolc     (double xopttol)       { (void) xopttol;       return 0; }
    virtual int setOpttold     (double xopttol)       { (void) xopttol;       return 0; }
    virtual int setlr          (double xlr)           { loclr  = xlr;         return 1; }
    virtual int setlrb         (double xlr)           { loclrb = xlr;         return 1; }
    virtual int setlrc         (double xlr)           { loclrc = xlr;         return 1; }
    virtual int setlrd         (double xlr)           { loclrd = xlr;         return 1; }
    virtual int setmaxitcnt    (int    xmaxitcnt)     { (void) xmaxitcnt;     return 0; }
    virtual int setmaxtraintime(double xmaxtraintime) { (void) xmaxtraintime; return 0; }
    virtual int settraintimeend(double xtraintimeend) { (void) xtraintimeend; return 0; }

    virtual int setmaxitermvrank(int    nv) { (void) nv; NiceThrow("Function setmaxitermvrank not available for this ML type."); return 0; }
    virtual int setlrmvrank     (double nv) { (void) nv; NiceThrow("Function setlrmvrank not available for this ML type.");      return 0; }
    virtual int setztmvrank     (double nv) { (void) nv; NiceThrow("Function setztmvrank not available for this ML type.");      return 0; }

    virtual int setbetarank(double nv) { (void) nv; NiceThrow("Function setbetarank not available for this ML type."); return 0; }

    virtual int setC        (double xC)          { locsigma = 1/xC;             return 1; }
    virtual int setsigma    (double xsigma)      { return setC(1/((xsigma<1e12)?1e-12:xsigma));                 }
    virtual int setsigma_cut(double xsigma_cut)  {           (void) xsigma_cut; return 0; }
    virtual int seteps      (double xeps)        {           (void) xeps;       return 0; }
    virtual int setCclass   (int d, double xC)   { (void) d; (void) xC;         return 0; }
    virtual int setepsclass (int d, double xeps) { (void) d; (void) xeps;       return 0; }

    virtual int setmpri  (int nv)            { xmuprior    = nv; calcallprior(); return 0; }
    virtual int setprival(const gentype &nv) { xmuprior_gt = nv; calcallprior(); return 0; }
    virtual int setpriml (const ML_Base *nv) { xmuprior_ml = nv; calcallprior(); return 0; }

    virtual int scale  (double a) { (void) a; return 0; }
    virtual int reset  (void)     {           return 0; }
    virtual int restart(void)     {           return 0; }
    virtual int home   (void)     {           return 0; }

    virtual ML_Base &operator*=(double sf) { scale(sf); return *this; }

    virtual int scaleby(double sf) { *this *= sf; return 1; }

    virtual int settspaceDim    (int newdim);
    virtual int addtspaceFeat   (int i);
    virtual int removetspaceFeat(int i);
    virtual int addxspaceFeat   (int) { return 0; }
    virtual int removexspaceFeat(int) { return 0; }

    virtual int setsubtype(int i) { (void) i; NiceAssert( i == 0 ); return 0; }

    virtual int setorder(int neword);
    virtual int addclass(int label, int epszero = 0) { (void) label; (void) epszero; NiceThrow("Function addclass not available for this ML type"); return 0; }

    virtual int setNRff   (int nv) { (void) nv; NiceThrow("Function setNRff not available for this ML type.");    return 0; }
    virtual int setNRffRep(int nv) { (void) nv; NiceThrow("Function setNRffRep not available for this ML type."); return 0; }
    virtual int setReOnly (int nv) { (void) nv; NiceThrow("Function setReOnly not available for this ML type.");  return 0; }
    virtual int setinAdam (int nv) { (void) nv; NiceThrow("Function setInAdam not available for this ML type.");  return 0; }
    virtual int setoutGrad(int nv) { (void) nv; NiceThrow("Function setoutGrad not available for this ML type."); return 0; }

    // Sampling mode
    //
    // For stochastic models, setting sample mode 1 makes
    // the model act like a sample from the distribution (so for example
    // a GP in sample mode, upon evaluating gg, takes a sample from
    // the posterior and adds it to the prior):
    //
    // nv = 0: non-sample mode (generally this does nothing.  An exception
    //         is the imp type, where sampling actually duplicates the base
    //         and setting sample mode zero will remove the sampled version).
    // nv = 1: sample on a grid (slow for higher dimensions)
    // nv = 2: do pre-calculations for sampling but does not take the
    //         actual sample.  This can save on calculations if you
    //         pre-sample a source and, make lots of copies and sample
    //         the copies.
    // nv = 3: JIT version.  Does no samples but sets flag, so evaluating
    //         the model will cause a sample at that point.  Note that
    //         only calls to evaluate through var() will be sampled, and
    //         currently only in GPR models.  Not compatible with sampType != 0..
    // nv = 4: JIT version with samples forced positive.
    // nv = 5: JIT version with samples forced negative.
    // Nsamp: number of samples (negative for random samples, 0 for auto
    //         random as per Kandasami TS-BO paper, 10jd^2, j = N(), d = dim)
    // sampSplit: let x = [ x1 ~ x2 ~ ... ~ xn ] where n = sampSplit
    //            Thus the resulting function can be treated as a kernel
    //            0 - use true random when drawing x (default is fixed
    //                random sequence to enable comparison between models)
    //           -n - use the same fixed random sequence every time).
    // sampType:  0 - unbounded draw
    //            1 - positive (definite/symm) draw by clip max(0,y).
    //            2 - positive (definite/symm) draw by flip |y|.
    //            3 - negative (definite/symm) draw by clip min(0,y).
    //            4 - negative (definite/symm) draw by flip -|-y|.
    //            5 - unbounded (symmetric) draw.
    //            6 - just sample alpha from unit normal distribution
    //            7 - just sample alpha from unit universal distribution
    //           1x - sampleType x, but force alpha +ve after
    //           2x - sampleType x, but force alpha -ve after
    //          1xx - sampleType x, but square the resulting function using
    //                kernel tricks and data transformation.  Existing
    //                observations are priors on the pre-squared function.
    //          2xx - like 1xx, except existing observations are priors
    //                on the squared function.
    //          3xx - like 2xx, but don't actually square the result.
    // xsampType: method for generating x sample
    //            0 - "true" pseudo-random
    //            1 - pre-defined sequence of random samples, generated sequentially
    //            2 - pre-defined sequence of random samples, same everytime
    //            3 - grid of Nsamp^dim samples
    //
    // sampScale: sample is from GP(mean,sampScale^2.covariance)
    // sampSlack: if > 0 then the sample box is expanded to [-xmax,xmax] along
    //            the edges, [-sampSlack,0] on the boundaries of the extra boxes.

    virtual int  isSampleMode(void) const { return 0; }
    virtual int setSampleMode(int nv, const Vector<gentype> &xmin, const Vector<gentype> &xmax, int Nsamp, int sampSplit, int sampType, int xsampType, double sampScale, double sampSlack = 0) { (void) nv; (void) xmin; (void) xmax; (void) Nsamp; (void) sampSplit; (void) sampType; (void) xsampType; (void) sampScale; (void) sampSlack; return 0; }

    // Training functions:
    //
    // res not changed on success, set nz on fail (so set zero before calling)
    // returns 0 if trained machine unchanged, 1 otherwise
    //
    // NB: - killSwitch is polled periodically, and training will terminate
    //       early if it is set.  This is designed for multi-thread use.

    virtual void fudgeOn (void) { }
    virtual void fudgeOff(void) { }

    virtual int train(int &res)                              { svmvolatile int killSwitch = 0; return train(res,killSwitch); }
    virtual int train(int &res, svmvolatile int &killSwitch) { (void) res;     killSwitch = 0; return 0;                     }

    // Information functions:
    //
    // NB: - loglikelihood is not guaranteed to be defined or "actual" log
    //       likelihood.  For example in gpr models this is the log likelihood,
    //       but in svm models this is a "quasi" log likelihood
    //     - ditto maxinfogain (maximum information gain)

    virtual double loglikelihood(void) const { return valvnan(); }
    virtual double maxinfogain  (void) const { return valvnan(); }
    virtual double RKHSnorm     (void) const { return valvnan(); }
    virtual double RKHSabs      (void) const { return valvnan(); }

    // Evaluation Functions:
    //
    // - gg(resg ,y): writes unprocessed result to resg
    // - hh(resh ,y): writes processed result to resh
    // - gh(rh,rg,y): writes both processed and unprocessed results
    //
    // - cov(res,mu,x,y): calculate covariance in resg (if available)
    //                Note that this is always scalar (variance is a function
    //                of x, which is shared by all axis in the scalar case).
    //                If y is a vector of vectors the result is a covariance
    //                matrix.  mu is also returned because this is a negligable-
    //                cost operation in most cases.  mu relates to x.
    // -  e(res,y,x): writes smoothed error to res
    // - dg(res,y,x): writes dg/dx to res (average if i is a vector)
    // - de(res,y,x): gradients for error (scaler version is de/dg)
    //
    // - stabProb: probability of mu_{1:p} stability, using ||.||_pnrm (or rotated inf-norm if rot != 0)
    //
    // By default gh is the prior mean and cov is the kernel.
    //
    // The error is q(x)-y, where q(x) = h(x) for the regression types, and
    // some version of the sigmoid function for classification.  Note that
    // the error may be a scalar, vector or anion.  The gradient de is a
    // sparse vector gradient wrt raw output.  Note that this is not currently
    // well defined if raw output includes things like g and x.
    //
    // Gradients: grad = sum_i res_i mod_vj(x_i) + resn x_-1
    //            mod_vj(x_i) = x_i          if vj <  0
    //                          [ 0        ]
    //                        = [ x_{i,vj} ] if vj >= 0
    //                          [ 0        ]
    //           note there is no -1 for vectorial i (q variants)
    //
    // e: error for vector
    // dedg: gradient of error wrt g (could be any number of types)
    // dedK: returns dei/dKj.  In SVM, this is dedg multiplied by alpha
    //       (always a double), where "alpha" is in sum_j alpha_j K(x,x_j).
    // dedK: vector version only fills those spots for which alphaState()(j) is nonzero
    // dedK: matrix verison only fills in those spots for which alphaState()(i) and alphaState()(j) are nonzero
    //
    // pxyprod: if you know the inner products (or diff) of a vector wrt all training vectors you
    //          can put it here to speed up calculations.  This is a vector of pxyprod pointers as
    //          described previously.
    //
    // retaltg: if bit 1 then returns g as a "value-per-class" vector
    //          if bit 2 set then returns g(x)^2 using 4-kernel trick

    virtual int ggTrainingVector(               gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = nullptr) const { gentype resh; return ghTrainingVector(resh,resg,i,retaltg,pxyprodi); }
    virtual int hhTrainingVector(gentype &resh,                int i,                  gentype ***pxyprodi = nullptr) const { gentype resg; return ghTrainingVector(resh,resg,i,0,      pxyprodi); }
    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = nullptr) const { resg = yp(i); resh = resg; (void) retaltg; (void) pxyprodi; return 0; }

    virtual double eTrainingVector(int i) const { (void) i; NiceThrow("eTrainingVector not implemented for this ML"); return 0.0; }

    virtual int covTrainingVector(gentype &resv, gentype &resmu, int i, int j, gentype ***pxyprodi = nullptr, gentype ***pxyprodj = nullptr, gentype **pxyprodij = nullptr) const { K2(resv,i,j); resmu = yp(i); (void) pxyprodi; (void) pxyprodj; (void) pxyprodij; (void) pxyprodij; return 0; }

    virtual double         &dedgTrainingVector(double         &res, int i) const { gentype gres; dedgTrainingVector(gres,i); res = (double) gres;                 return res; }
    virtual Vector<double> &dedgTrainingVector(Vector<double> &res, int i) const { gentype gres; dedgTrainingVector(gres,i); res = (const Vector<double> &) gres; return res; }
    virtual d_anion        &dedgTrainingVector(d_anion        &res, int i) const { gentype gres; dedgTrainingVector(gres,i); res = (const d_anion &) gres;        return res; }
    virtual gentype        &dedgTrainingVector(gentype        &res, int i) const { (void) res; (void) i; NiceThrow("dedgTrainingVector not implemented for this ML"); return res; }

    virtual double &d2edg2TrainingVector(double &res, int i) const { (void) res; (void) i; NiceThrow("d2db2TrainingVector not implemented for this ML"); return res; }

    virtual double          dedKTrainingVector(int i, int j)               const { (void) i; (void) j; NiceThrow("dedKTrainingVector not implemented for this ML"); return 0.0; }
    virtual Vector<double> &dedKTrainingVector(Vector<double> &res, int i) const { (void) i; NiceThrow("dedKTrainingVector not implemented for this ML");           return res; }
    virtual Matrix<double> &dedKTrainingVector(Matrix<double> &res)        const { NiceThrow("dedKTrainingVector not implemented for this ML");                     return res; }

    virtual void dgTrainingVectorX(Vector<gentype> &resx, int i) const;
    virtual void dgTrainingVectorX(Vector<double>  &resx, int i) const;

    virtual void deTrainingVectorX(Vector<gentype> &resx, int i) const;

    virtual void dgTrainingVectorX(Vector<gentype> &resx, const Vector<int> &i) const;
    virtual void dgTrainingVectorX(Vector<double>  &resx, const Vector<int> &i) const;

    virtual void deTrainingVectorX(Vector<gentype> &resx, const Vector<int> &i) const;

    virtual int ggTrainingVector(double         &resg, int i, int retaltg = 0, gentype ***pxyprodi = nullptr) const { (void) pxyprodi; gentype res; int resi = ggTrainingVector(res,i,retaltg); resg = (double)                 res; return resi; }
    virtual int ggTrainingVector(Vector<double> &resg, int i, int retaltg = 0, gentype ***pxyprodi = nullptr) const { (void) pxyprodi; gentype res; int resi = ggTrainingVector(res,i,retaltg); resg = (const Vector<double> &) res; return resi; }
    virtual int ggTrainingVector(d_anion        &resg, int i, int retaltg = 0, gentype ***pxyprodi = nullptr) const { (void) pxyprodi; gentype res; int resi = ggTrainingVector(res,i,retaltg); resg = (const d_anion &)        res; return resi; }

    virtual void dgTrainingVector(Vector<gentype>         &res, gentype        &resn, int i) const { (void) res; (void) resn; (void) i; NiceThrow("Function dgTrainingVector not available for this ML type."); }
    virtual void dgTrainingVector(Vector<double>          &res, double         &resn, int i) const;
    virtual void dgTrainingVector(Vector<Vector<double> > &res, Vector<double> &resn, int i) const;
    virtual void dgTrainingVector(Vector<d_anion>         &res, d_anion        &resn, int i) const;

    virtual void deTrainingVector(Vector<gentype> &res, gentype &resn, int i) const { dgTrainingVector(res,resn,i); double scale = 0.0; dedgTrainingVector(scale,i); res.scale(scale); resn *= scale; }

    virtual void dgTrainingVector(Vector<gentype>         &res, const Vector<int> &i) const;
    virtual void dgTrainingVector(Vector<double>          &res, const Vector<int> &i) const;
    virtual void dgTrainingVector(Vector<Vector<double> > &res, const Vector<int> &i) const;
    virtual void dgTrainingVector(Vector<d_anion>         &res, const Vector<int> &i) const;

    virtual void deTrainingVector(Vector<gentype> &res, const Vector<int> &i) const;

    virtual void stabProbTrainingVector(double  &res, int i, int p, double pnrm, int rot, double mu, double B) const;

    virtual int gg(               gentype &resg, const SparseVector<gentype> &x,                  const vecInfo *xinf = nullptr, gentype ***pxyprodx = nullptr) const { gentype resh; return gh(resh,resg,x,0,xinf,pxyprodx); }
    virtual int hh(gentype &resh,                const SparseVector<gentype> &x,                  const vecInfo *xinf = nullptr, gentype ***pxyprodx = nullptr) const { gentype resg; return gh(resh,resg,x,0,xinf,pxyprodx); }
    virtual int gh(gentype &resh, gentype &resg, const SparseVector<gentype> &x, int retaltg = 0, const vecInfo *xinf = nullptr, gentype ***pxyprodx = nullptr) const { setInnerWildpa(&x,xinf); int res = ghTrainingVector(resh,resg,-1,retaltg,pxyprodx); resetInnerWildp(xinf == nullptr); return res; }

    virtual double e(const gentype &y, const SparseVector<gentype> &x, const vecInfo *xinf = nullptr) const { setInnerWildpa(&x,xinf); setWildTargpp(y); double res = eTrainingVector(-1); resetInnerWildp(xinf == nullptr); return res; }

    virtual int cov(gentype &resv, gentype &resmu, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo *xainf = nullptr, const vecInfo *xbinf = nullptr, gentype ***pxyprodxa = nullptr, gentype ***pxyprodxb = nullptr, gentype **pxyprodij = nullptr) const { setInnerWildpa(&xa,xainf); setInnerWildpb(&xb,xbinf); int res = covTrainingVector(resv,resmu,-1,-3,pxyprodxa,pxyprodxb,pxyprodij); resetInnerWildp(( xainf == nullptr ),( xbinf == nullptr )); return res; }

    virtual void dedg(double         &res, const gentype &y, const SparseVector<gentype> &x, const vecInfo *xinf = nullptr) const { setInnerWildpa(&x,xinf); setWildTargpp(y); dedgTrainingVector(res,-1); resetInnerWildp(xinf == nullptr); }
    virtual void dedg(Vector<double> &res, const gentype &y, const SparseVector<gentype> &x, const vecInfo *xinf = nullptr) const { setInnerWildpa(&x,xinf); setWildTargpp(y); dedgTrainingVector(res,-1); resetInnerWildp(xinf == nullptr); }
    virtual void dedg(d_anion        &res, const gentype &y, const SparseVector<gentype> &x, const vecInfo *xinf = nullptr) const { setInnerWildpa(&x,xinf); setWildTargpp(y); dedgTrainingVector(res,-1); resetInnerWildp(xinf == nullptr); }
    virtual void dedg(gentype        &res, const gentype &y, const SparseVector<gentype> &x, const vecInfo *xinf = nullptr) const { setInnerWildpa(&x,xinf); setWildTargpp(y); dedgTrainingVector(res,-1); resetInnerWildp(xinf == nullptr); }

    virtual double &d2edg2(double &res, const gentype &y, const SparseVector<gentype> &x, const vecInfo *xinf = nullptr) const { (void) res; (void) y; (void) x; (void) xinf; NiceThrow("d2db2 not implemented for this ML"); return res; }

    virtual void dgX(Vector<gentype> &resx, const SparseVector<gentype> &x, const vecInfo *xinf = nullptr) const { setInnerWildpa(&x,xinf); dgTrainingVectorX(resx,-1); resetInnerWildp(xinf == nullptr); }
    virtual void dgX(Vector<double>  &resx, const SparseVector<gentype> &x, const vecInfo *xinf = nullptr) const { setInnerWildpa(&x,xinf); dgTrainingVectorX(resx,-1); resetInnerWildp(xinf == nullptr); }

    virtual void deX(Vector<gentype> &resx, const gentype &y, const SparseVector<gentype> &x, const vecInfo *xinf = nullptr) const { setInnerWildpa(&x,xinf); setWildTargpp(y); deTrainingVectorX(resx,-1); resetInnerWildp(xinf == nullptr); }

    virtual int gg(double         &resg, const SparseVector<gentype> &x, int retaltg = 0, const vecInfo *xinf = nullptr, gentype ***pxyprodx = nullptr) const { setInnerWildpa(&x,xinf); int resi = ggTrainingVector(resg,-1,retaltg,pxyprodx); resetInnerWildp(xinf == nullptr); return resi; }
    virtual int gg(Vector<double> &resg, const SparseVector<gentype> &x, int retaltg = 0, const vecInfo *xinf = nullptr, gentype ***pxyprodx = nullptr) const { setInnerWildpa(&x,xinf); int resi = ggTrainingVector(resg,-1,retaltg,pxyprodx); resetInnerWildp(xinf == nullptr); return resi; }
    virtual int gg(d_anion        &resg, const SparseVector<gentype> &x, int retaltg = 0, const vecInfo *xinf = nullptr, gentype ***pxyprodx = nullptr) const { setInnerWildpa(&x,xinf); int resi = ggTrainingVector(resg,-1,retaltg,pxyprodx); resetInnerWildp(xinf == nullptr); return resi; }

    virtual void dg(Vector<gentype>         &res, gentype        &resn, const SparseVector<gentype> &x, const vecInfo *xinf = nullptr) const { setInnerWildpa(&x,xinf); dgTrainingVector(res,resn,-1); resetInnerWildp(xinf == nullptr); }
    virtual void dg(Vector<double>          &res, double         &resn, const SparseVector<gentype> &x, const vecInfo *xinf = nullptr) const { setInnerWildpa(&x,xinf); dgTrainingVector(res,resn,-1); resetInnerWildp(xinf == nullptr); }
    virtual void dg(Vector<Vector<double> > &res, Vector<double> &resn, const SparseVector<gentype> &x, const vecInfo *xinf = nullptr) const { setInnerWildpa(&x,xinf); dgTrainingVector(res,resn,-1); resetInnerWildp(xinf == nullptr); }
    virtual void dg(Vector<d_anion>         &res, d_anion        &resn, const SparseVector<gentype> &x, const vecInfo *xinf = nullptr) const { setInnerWildpa(&x,xinf); dgTrainingVector(res,resn,-1); resetInnerWildp(xinf == nullptr); }

    virtual void de(Vector<gentype> &res, gentype &resn, const gentype &y, const SparseVector<gentype> &x, const vecInfo *xinf = nullptr) const { setInnerWildpa(&x,xinf); setWildTargpp(y); deTrainingVector(res,resn,-1); resetInnerWildp(xinf == nullptr); }

    virtual void stabProb(double  &res, const SparseVector<gentype> &x, int p, double pnrm, int rot, double mu, double B) const { setInnerWildpa(&x); stabProbTrainingVector(res,-1,p,pnrm,rot,mu,B); resetInnerWildp(); }

    // var and covar functions

    virtual int varTrainingVector(gentype &resv, gentype &resmu, int i, gentype ***pxyprodi = nullptr, gentype **pxyprodii = nullptr) const { return covTrainingVector(resv,resmu,i,i,pxyprodi,pxyprodi,pxyprodii); }
    virtual int var(gentype &resv, gentype &resmu, const SparseVector<gentype> &xa, const vecInfo *xainf = nullptr, gentype ***pxyprodx = nullptr, gentype **pxyprodxx = nullptr) const { setInnerWildpa(&xa,xainf); int res = covTrainingVector(resv,resmu,-1,-1,pxyprodx,pxyprodx,pxyprodxx); resetInnerWildp(xainf == nullptr); return res; }

    virtual int covarTrainingVector(Matrix<gentype> &resv, const Vector<int> &i) const;
    virtual int covar(Matrix<gentype> &resv, const Vector<SparseVector<gentype> > &x) const;

    // Input-Output noise calculation
    //
    // Calculates variance of output including noise from input (in addition to normal output noise):
    //
    // McHutchon, Rasmussen: Gaussian Process Training with Input Noise
    //
    // var = var + sum_j df/dx_i var_i df/dx_i
    //
    // if noise in one only one of [ x0var ~ x1var ~ ... ] use u >= 0 (-1 for all, -2 for all but x0var)
    //
    // (basically we model output noise as input noise times gradient,
    // assuming input noise is "small" relative to gradient variation).

    virtual int noisevarTrainingVector(gentype &resv, gentype &resmu, int i, const SparseVector<gentype> &xvar, int u = -1, gentype ***pxyprodi = nullptr, gentype **pxyprodii = nullptr) const;
    virtual int noisevar(gentype &resv, gentype &resmu, const SparseVector<gentype> &xa, const SparseVector<gentype> &xvar, int u = -1, const vecInfo *xainf = nullptr, gentype ***pxyprodx = nullptr, gentype **pxyprodxx = nullptr) const { setInnerWildpa(&xa,xainf); int res = noisevarTrainingVector(resv,resmu,-1,xvar,u,pxyprodx,pxyprodxx); resetInnerWildp(xainf == nullptr); return res; }

    virtual int noisecovTrainingVector(gentype &resv, gentype &resmu, int i, int j, const SparseVector<gentype> &xvar, int u = -1, gentype ***pxyprodi = nullptr, gentype ***pxyprodj = nullptr, gentype **pxyprodij = nullptr) const;
    virtual int noisecov(gentype &resv, gentype &resmu, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xvar, int u = -1, const vecInfo *xainf = nullptr, const vecInfo *xbinf = nullptr, gentype ***pxyprodx = nullptr, gentype ***pxyprody = nullptr, gentype **pxyprodxy = nullptr) const { setInnerWildpa(&xa,xainf); setInnerWildpb(&xb,xbinf); int res = noisecovTrainingVector(resv,resmu,-1,-3,xvar,u,pxyprodx,pxyprody,pxyprodxy); resetInnerWildp(( xainf == nullptr ),( xbinf == nullptr )); return res; }

    // Training data tracking functions:
    //
    // The indexing and type functions are described below.  They give
    // information about the contents of the training set - which features
    // are  used, how often each feature is used, and the type of data in any
    // given feature.  Full description is below.
    //
    // Fucntions to translate to/from sparse form are also defined.
    //
    // (u=-1 for overall, u>=0 gives only dimension for relevant minor/up type - see sparsevector)

    virtual const Vector<int>          &indKey         (int u = -1) const { if ( indexKey.isindpresent(u+1)      ) { return indexKey(u+1);      } static thread_local Vector<int>          dummy; return dummy; }
    virtual const Vector<int>          &indKeyCount    (int u = -1) const { if ( indexKeyCount.isindpresent(u+1) ) { return indexKeyCount(u+1); } static thread_local Vector<int>          dummy; return dummy; }
    virtual const Vector<int>          &dattypeKey     (int u = -1) const { if ( typeKey.isindpresent(u+1)       ) { return typeKey(u+1);       } static thread_local Vector<int>          dummy; return dummy; }
    virtual const Vector<Vector<int> > &dattypeKeyBreak(int u = -1) const { if ( typeKeyBreak.isindpresent(u+1)  ) { return typeKeyBreak(u+1);  } static thread_local Vector<Vector<int> > dummy; return dummy; }

    // Other functions
    //
    // setaltx: if non-null, set alternative x source (nullptr to reset)
    //          Function is naive - it does not update anything to reflect
    //          possible changes in x from the new source.  Use resetKernel
    //          to propogate such changes manually (with updateInfo = 1).
    // disable: removes influence of points without removing them from the
    //          training set.  Note that this additionally goes through
    //          all altx sources (and so on) and disables x in those as
    //          well.  If i<0 then it disables ((-i)-1)%N, with one exception:
    //          blk_conect.  For blk_conect i<0 means disable in top model
    //          but not lower ones.

    virtual void setaltx(const ML_Base *_altxsrc) { incxvernum(); altxsrc = _altxsrc; }

    virtual int disable(int i);
    virtual int disable(const Vector<int> &i);

    // ================================================================
    //     Common functions for all ML
    // ================================================================

    // Training data information functions (all assume no far/farfar/farfarfar or multivectors)

    virtual const SparseVector<gentype> &xsum   (SparseVector<gentype> &res) const;
    virtual const SparseVector<gentype> &xmean  (SparseVector<gentype> &res) const;
    virtual const SparseVector<gentype> &xmeansq(SparseVector<gentype> &res) const;
    virtual const SparseVector<gentype> &xsqsum (SparseVector<gentype> &res) const;
    virtual const SparseVector<gentype> &xsqmean(SparseVector<gentype> &res) const;
    virtual const SparseVector<gentype> &xmedian(SparseVector<gentype> &res) const;
    virtual const SparseVector<gentype> &xvar   (SparseVector<gentype> &res) const;
    virtual const SparseVector<gentype> &xstddev(SparseVector<gentype> &res) const;
    virtual const SparseVector<gentype> &xmax   (SparseVector<gentype> &res) const;
    virtual const SparseVector<gentype> &xmin   (SparseVector<gentype> &res) const;

    // Kernel normalisation function
    // =============================
    //
    // Effectively Normalise the training data (zero mean, unit variance) by
    // setting shifting/scaling in the kernel as follows:
    //
    // xmean   = (1/N) sum_i x_i
    // xmeansq = (1/N) sum_i x_i.^2
    // xvar    = (1/N) sum_i (x_i-xmean).^2
    //         = (1/N) sum_i x_i.^2 + (1/N) sum_i xmean.^2 - (2/N) sum_i x_i.*xmean
    //         = xmeansq + xmean.^2 - 2 xmean.*((1/N) sum_i x_i)
    //         = xmeansq + xmean.^2 - 2 xmean.*xmean
    //         = xmeansq - xmean.^2
    //
    // xshift = -xmean
    // xscale = 1./sqrt(xvar)
    //
    // Individual components may be any one of the types supported by gentype.
    // This includes:
    //
    // - real (int or double)
    // - anion (assume only real, complex, quaternion or octonian)
    // - vector (assume only of real, anion, vector or matrix)
    // - matrix (assume only of real, anion, vector or matrix)
    // - set
    // - dgraph
    // - string
    //
    // Now:
    //
    // - sets, dgraphs and strings cannot be normalised (it makes no sense), so
    //   we need to detect any feature having this type of argument and then
    //   place a 0 in the relevant mean, 1 in the relevant variance
    // - reals evaluate trivially
    //
    // Vectors and matrices are more complicated.  For such "scalars":
    //
    // mean = (1/N) sum_i y_i
    // var  = (1/N) sum_i (y_i-mean).(y_i-mean)'
    //
    // which is an outer product, ' means conjugate transpose.
    //
    // (y_i-mean)  -> A.(y_i-mean)
    // (y_i-mean)' -> (y_i-mean)'.A'
    //
    // so: var -> newvar = A.var.A' = I
    //     var = BB', B = chol(var)
    //
    // choose: A = inv(B)
    // => newvar = I
    //
    // normalisation: y -> A.(y-mean)
    //
    // Treatment of missing features:
    //
    // By default, "missing" features (ie indices present in some vectors but
    // not others) are treated as 0s.  An alternative approach may be selected
    // by setting replaceMissingFeatures = 1.  Under this alternative scheme
    // missing features are treated as not present and replaced (in the ML)
    // by the mean value of this feature for those vectors in which it is
    // present.  This is done prior to calculation of shifting and scaling
    // factors.
    //
    // Unit range: applies to reals/integers/nulls only.  Asserts range of input
    // must lie between 0 (minimum value) and 1 (maximum value).  Thus
    //
    // shift = min(x)
    // scale = 1/(max(x)-min(x))
    //
    // Option: flatnorm: rather than work on a per-feature basis, this sets
    //         the scale to the min scale.
    //         noshift: do not apply shifting, only scaling.

    virtual int normKernelNone                  (void);
    virtual int normKernelZeroMeanUnitVariance  (int flatnorm = 0, int noshift = 0);
    virtual int normKernelZeroMedianUnitVariance(int flatnorm = 0, int noshift = 0);
    virtual int normKernelUnitRange             (int flatnorm = 0, int noshift = 0);

    // Helper functions for sparse variables
    //
    // These functions convert to/from sparse vectors.  It is assumed that
    // indexing in the sparse vectors follows the index key defined for
    // this training set.  makeFullSparse ensures that all indices are
    // present in the sparse vector without setting them (so that they
    // default to zero).
    //
    // Assumptions: this assumes no far/farfar/farfarfar elements and no [ ... ~ ... ] style multi-vectors

    virtual SparseVector<gentype> &xlateToSparse(SparseVector<gentype> &dest, const Vector<gentype>      &src) const;
    virtual SparseVector<gentype> &xlateToSparse(SparseVector<gentype> &dest, const Vector<double>       &src) const;
    virtual SparseVector<gentype> &xlateToSparse(SparseVector<gentype> &dest, const SparseVector<double> &src) const;

    virtual Vector<gentype> &xlateFromSparse(Vector<gentype> &dest, const SparseVector<gentype> &src) const;
    virtual Vector<double>  &xlateFromSparse(Vector<double>  &dest, const SparseVector<gentype> &src) const;
    virtual Vector<double>  &xlateFromSparse(Vector<double>  &dest, const SparseVector<double>  &src) const;
    virtual Vector<double>  &xlateFromSparse(Vector<double>  &dest, const Vector<gentype>       &src) const;
    virtual Vector<double>  &xlateFromSparse(Vector<double>  &dest, const Vector<double>        &src) const;

    virtual Vector<double>  &xlateFromSparseTrainingVector(Vector<double>  &dest, int i) const { return xlateFromSparse(dest,x(i)); }
    virtual Vector<gentype> &xlateFromSparseTrainingVector(Vector<gentype> &dest, int i) const { return xlateFromSparse(dest,x(i)); }

    virtual SparseVector<gentype> &makeFullSparse(SparseVector<gentype> &dest) const;

    // x detangling
    //
    // x "vectors" can get a little confusing, as they can refer to "normal" vectors,
    // rank constraints, gradient constraints and other things.  To keep it together
    // the following function disentangles it all.  Given i, xx and xzinfo this function
    // returns the following:
    //
    // 0:    xnear{info} points to x vector {info}, inear is index (xnear always set)
    // 1:    xfar{info} and ifar refer to "other side" of rank constraint
    // 2:    xfarfar refers to direction of gradient constraint, gradOrder > 0 *or* xfarfarfar refers to direction of gradient constraint, gradOrderR > 0
    // 4:    gradOrder > 0, but xfarfar not set *or* gradOrderR > 0 but xfarfarfar not set
    // 8:    treat distributions as sample from sets, allowing whole-set constraints
    // 16:   idiagr set
    // 32:   evaluate dense integral
    // 64:   evaluate dense derivative
    // 128:  xfarfar refers to direction of gradient constraint, gradOrder > 0
    // 256:  gradOrder > 0, but xfarfar not set
    // 512:  xfarfarfar refers to direction of gradient constraint, gradOrderR > 0
    // 1024: gradOrderR > 0, but xfarfarfar not set
    // 2048: actually a weighted sum of vectors
    // 3,5,9,10,11,12,13,16,32,64: allowable combinations (128,256,512,1024 also follow if 2,4 set)
    // NOT ANY MORE: -1: idiagr set (simple version only)
    //
    // For tuple disambiguation will detect and set ineartup, ifartup non-nullptr if found.
    // inear/ifar, xnear/xfar, xnearinfo/xfarinfo not to be trusted if ineartup/ifartup set non-nullptr
    //
    // Notes:
    //
    // - If xx (and xzinfo) are non-nullptr the result is always 0, xnear{info} = xx{info}
    // - if iokr set then kernel evaluation should be scaled by UU product with index iok.
    // - xnear{info}, xfar{info}, xfarfar are never left nullptr.
    // - case 3 cannot be properly dealt with at present
    // - usextang: set if you want to use precalculated xtang, otherwise calculate from scratch

    virtual int detangle_x(int i, int usextang = 0) const
    {
        //xx     = xx     ? xx     : &xgetloc(i);
        //xzinfo = xzinfo ? xzinfo : &locxinfo(i);

        int res = 0;

        if ( !usextang )
        {
            const SparseVector<gentype> *xx     = &x(i);
            const vecInfo               *xzinfo = &locxinfo(i);

            const SparseVector<gentype> *xnear      = nullptr;
            const SparseVector<gentype> *xfar       = nullptr;
            const SparseVector<gentype> *xfarfar    = nullptr;
            const SparseVector<gentype> *xfarfarfar = nullptr;
            const vecInfo *xnearinfo = nullptr;
            const vecInfo *xfarinfo  = nullptr;
            int inear = 0;
            int ifar  = 0;
            const gentype *ineartup = nullptr;
            const gentype *ifartup  = nullptr;
            int iokr   = 0;
            int iok    = 0;
            int idiagr = 0;
            int igradOrder  = 0;
            int igradOrderR = 0;
            int iplanr = 0;
            int iplan  = 0;
            int iset   = 0;
            int idenseint   = 0;
            int idensederiv = 0;
            int ilr;
            int irr;
            int igr;
            int igrR;
            double rankL;
            double rankR;
            int gmuL;
            int gmuR;
            Vector<int> sumind;
            Vector<double> sumweight;
            double diagoffset = 0;
            int ivectset = 0;

            SparseVector<gentype> *xuntang = nullptr;
            vecInfo *xzinfountang = nullptr;

            //Final 0 here suppresses allocation of xuntang/xuntanginfo
            res = detangle_x(xuntang,xzinfountang,xnear,xfar,xfarfar,xfarfarfar,xnearinfo,xfarinfo,inear,ifar,ineartup,ifartup,ilr,irr,igr,igrR,iokr,iok,rankL,rankR,gmuL,gmuR,i,idiagr,xx,xzinfo,igradOrder,igradOrderR,iplanr,iplan,iset,idenseint,idensederiv,sumind,sumweight,diagoffset,ivectset,usextang,0);
        }

        else
        {
            res = locxtang(i);
        }

        return res;
        //return idiagr ? -1 : res;
    }

    // if the vector itself needs to change (redirection of xnear/xfar etc)
    // then xuntang will be allocated and pointed to otherwise it will just be nullptr.

    virtual int detangle_x(SparseVector<gentype> *&xuntang, vecInfo *&xzinfountang,
                   const SparseVector<gentype> *&xnear, const SparseVector<gentype> *&xfar, const SparseVector<gentype> *&xfarfar, const SparseVector<gentype> *&xfarfarfar,
                   const vecInfo *&xnearinfo, const vecInfo *&xfarinfo,
                   int &inear, int &ifar,
                   const gentype *&ineartup, const gentype *&ifartup,
                   int &ilr, int &irr, int &igr, int &igrR,
                   int &iokr, int &iok,
                   double &rankL, double &rankR,
                   int &gmuL, int &gmuR,
                   int i, int &idiagr,
                   const SparseVector<gentype> *xx, const vecInfo *xzinfo,
                   int &gradOrder, int &gradOrderR,
                   int &iplanr, int &iplan, int &iset,
                   int &idenseint, int &idensederiv,
                   Vector<int> &sumind, Vector<double> &sumweight,
                   double &diagoffset, int &ivectset,
                   int usextang = 1, int allocxuntangifneeded = 1) const
    {
        NiceAssert( xx ); //&& xzinfo );

        int methodkey = 0;

        // Base references (no indirection yet)

        const SparseVector<gentype> &xib = *xx;

        xzinfo = xzinfo ? xzinfo : &xinfo(i);

        xnearinfo = xzinfo;
        xfarinfo  = xzinfo;

        (void) usextang;

        {
            // ilr:    is (leftside  of rank) index redirected?
            // irr:    is (rightside of rank) index redirected?
            // igr:    is gradient index redirected?
            // iokr:   is output kernel invoked?
            // idiagr: is diagonal kernel bypass invoked?
            // itup:   is il or ir a tuple?
            // iset:   are distributions treated as distributions (0) or samples from set(s) (1)

            int z = 0;
            int ind0present  = xib.isf4indpresent(z)  && !(xib.f4(z).isValNull());
            int ind1present  = xib.isf4indpresent(1)  && !(xib.f4(1).isValNull());
            int ind2present  = xib.isf4indpresent(2)  && !(xib.f4(2).isValNull());
            int ind3present  = xib.isf4indpresent(3)  && !(xib.f4(3).isValNull());
            int ind4present  = xib.isf4indpresent(4)  && !(xib.f4(4).isValNull());
//            int ind5present  = xib.isf4indpresent(5)  && !(xib.f4(5).isValNull());
            int ind6present  = xib.isf4indpresent(6)  && !(xib.f4(6).isValNull());
            int ind7present  = xib.isf4indpresent(7)  && !(xib.f4(7).isValNull());
            int ind8present  = xib.isf4indpresent(8)  && !(xib.f4(8).isValNull());
            int ind9present  = xib.isf4indpresent(9)  && !(xib.f4(9).isValNull());
            int ind10present = xib.isf4indpresent(10) && !(xib.f4(10).isValNull());
            int ind11present = xib.isf4indpresent(11) && !(xib.f4(11).isValNull());
            int ind12present = xib.isf4indpresent(12) && !(xib.f4(12).isValNull());
            int ind13present = xib.isf4indpresent(13) && !(xib.f4(13).isValNull());
            int ind14present = xib.isf4indpresent(14) && !(xib.f4(14).isValNull());
            int ind15present = xib.isf4indpresent(15) && !(xib.f4(15).isValNull());
            int ind16present = xib.isf4indpresent(16) && !(xib.f4(16).isValNull());
//            int ind17present = xib.isf4indpresent(17) && !(xib.f4(17).isValNull());
//            int ind18present = xib.isf4indpresent(18) && !(xib.f4(18).isValNull());
            int ind19present = xib.isf4indpresent(19) && !(xib.f4(19).isValNull());
            int ind20present = xib.isf4indpresent(20) && !(xib.f4(20).isValNull());

            ilr         = ind0present  ? 1 : 0;
            irr         = ind1present  ? 1 : 0;
            igr         = ind2present  ? 1 : 0;
            igrR        = ind15present ? 1 : 0;
            iokr        = ind3present  ? 1 : 0;
            idiagr      = ind4present  ? 1 : 0;
            iplanr      = ind7present  ? 1 : 0;
            iset        = ind8present  ? 1 : 0;
            idenseint   = ind11present ? 1 : 0;
            idensederiv = ind12present ? 1 : 0;
            ivectset    = ind20present ? xib.f4(20) : 0;

            diagoffset = 0;  if ( ind19present ) { diagoffset = (double) xib.f4(19); }
            ivectset   = 0;  if ( ind20present ) { ivectset   = (int) xib.f4(20);    }

            ineartup = nullptr;
            ifartup  = nullptr;

            if ( ilr && (xib.f4(0)).isValVector() )
            {
                ineartup = &xib.f4(0);
            }

            if ( irr && (xib.f4(1)).isValVector() )
            {
                ifartup = &xib.f4(1);
            }

            // il:    (leftside of rank) index
            // ir:    (rightside of rank) index
            // ig:    gradient index
            // iok:   basis references for output kernel
            // gmuL:  which instance does gradient of xnear refer to in multi-instance format
            // gmuR:  which instance does gradient of xfar refer to in multi-instance format
            // rankL: kernel weight for near part
            // rankR: kernel weight for far part

            int il    = ( ilr && !ineartup ) ? ( (int) xib.f4(0) ) : i;
            int ir    = ( irr && !ifartup  ) ? ( (int) xib.f4(1) ) : i;
            int ig    = igr  ? ( (int) xib.f4(2)  ) : i;
            int igR   = igrR ? ( (int) xib.f4(15) ) : i;
                iok   = ( iokr   && (xib.f4(3)).isValInteger() ) ? ( (int) xib.f4(3) ) : -1;
                iplan = ( iplanr && (xib.f4(7)).isValInteger() ) ? ( (int) xib.f4(7) ) : -1;
                gmuL  = ( ind9present  && (xib.f4(9)).isValInteger()  ) ? ((int) xib.f4(9))  : 0;
                gmuR  = ( ind10present && (xib.f4(10)).isValInteger() ) ? ((int) xib.f4(10)) : 0;
                rankL = ( ind13present && (xib.f4(13)).isCastableToRealWithoutLoss() ) ? ((double) xib.f4(13)) : 1.0;
                rankR = ( ind14present && (xib.f4(14)).isCastableToRealWithoutLoss() ) ? ((double) xib.f4(14)) : 1.0;

            // ilfar:    is (leftside  of rank) a far reference?
            // irfar:    is (rightside of rank) a far reference?
            // igfarfar: is gradient a farfar reference?

            int ilfar       = 0;
            int irfar       = ( !irr  && xib.isf1offindpresent() ) ? 1 : 0;
            int igfarfar    = ( !igr  && xib.isf2offindpresent() ) ? 2 : 0;
            int igfarfarfar = ( !igrR && xib.isf3offindpresent() ) ? 3 : 0;

            // gradient order calculations

            gradOrder  = ind6present  ? ( (int) xib.f4(6)  ) : ( ( igfarfar    || igr  ) ? 1 : 0 );
            gradOrderR = ind16present ? ( (int) xib.f4(16) ) : ( ( igfarfarfar || igrR ) ? 1 : 0 );

            // What are the method keys for indexes?

            methodkey = ( ( irfar || irr ) ? 1 : 0 )
                      | ( ( (  ( igfarfar || igr ) && ( gradOrder > 0 ) ) || (  ( igfarfarfar || igrR ) && ( gradOrderR > 0 ) ) ) ? 2 : 0 )
                      | ( ( ( !( igfarfar || igr ) && ( gradOrder > 0 ) ) || ( !( igfarfarfar || igrR ) && ( gradOrderR > 0 ) ) ) ? 4 : 0 )
                      | ( iset ? 8 : 0 )
                      | ( idiagr ? 16 : 0 )
                      | ( idenseint ? 32 : 0 )
                      | ( idensederiv ? 64 : 0 )
                      | ( (  ( igfarfar || igr ) && ( gradOrder > 0 ) ) ? 128 : 0 )
                      | ( ( !( igfarfar || igr ) && ( gradOrder > 0 ) ) ? 256 : 0 )
                      | ( (  ( igfarfarfar || igrR ) && ( gradOrderR > 0 ) ) ? 512  : 0 )
                      | ( ( !( igfarfarfar || igrR ) && ( gradOrderR > 0 ) ) ? 1024 : 0 );

            // NB: idiagr over-rides all other options here.

            if ( !idiagr )
            {
                // xnear:   (leftside  of rank) vector references
                // xfar:    (rightside of rank) vector references
                // xfarfar: gradient vector references
                //
                // Notes: - ternary operator short-circuits, so only relevant branch evaluated

                const SparseVector<gentype> &xxl  = ( i == il  ) ? *xx : x(il);
                const SparseVector<gentype> &xxr  = ( i == ir  ) ? *xx : x(ir);
                const SparseVector<gentype> &xxg  = ( i == ig  ) ? *xx : x(ig);
                const SparseVector<gentype> &xxgR = ( i == igR ) ? *xx : x(igR);

                xnear      = ( !methodkey ? &(xxl.n())  : ( ilfar       ? &(xxl.f1())  : &(xxl.n())  ) );
                xfar       = ( !methodkey ? &(xxr.n())  : ( irfar       ? &(xxr.f1())  : &(xxr.n())  ) );
                xfarfar    = ( !methodkey ? &(xxg.n())  : ( igfarfar    ? &(xxg.f2())  : &(xxg.n())  ) );
                xfarfarfar = ( !methodkey ? &(xxgR.n()) : ( igfarfarfar ? &(xxgR.f3()) : &(xxgR.n()) ) );

                // xnearinfo: (leftside  of rank) information
                // xfarinfo:  (rightside of rank) information

                xnearinfo = ( i == il ) ? xzinfo : &(xinfo(il));
                xfarinfo  = ( i == ir ) ? xzinfo : &(xinfo(ir));

                xnearinfo = &((*xnearinfo)(0,-1));
                xfarinfo  = &((*xfarinfo )(1,-1));

                // (leftside  of rank) index rename and recalc
                // (rightside of rank) index rename and recalc

                inear = il;
                ifar  = -(((ir+1)*100)+1);
            }
        }

        xuntang      = nullptr;
        xzinfountang = nullptr;

        if ( allocxuntangifneeded && !idiagr && ( ineartup && ifartup ) )
        {
            int q;

            // Process indirections

            xuntang      = new SparseVector<gentype>(xib);
            xzinfountang = new vecInfo;

                          (*xuntang).zeron();
                          (*xuntang).zerof1();
            if ( igr  ) { (*xuntang).overwritef2(*xfarfar);    }
            if ( igrR ) { (*xuntang).overwritef3(*xfarfarfar); }

            const Vector<gentype> &iain = (*ineartup).cast_vector();
            const Vector<gentype> &iaif = (*ifartup).cast_vector();

            int iains = iain.size();
            int iaifs = iaif.size();

            for ( q = 0 ; q < iains ; ++q )
            {
                (*xuntang).overwriten(x((int) iain(q)),q);
            }

            for ( q = 0 ; q < iaifs ; ++q )
            {
                (*xuntang).overwritef1(x((int) iaif(q)),q);
            }

            getKernel().getvecInfo((*xzinfountang),(*xuntang));
        }

        else if ( allocxuntangifneeded && !idiagr && ( ineartup && irr ) )
        {
            int q;

            // Process indirections

            xuntang      = new SparseVector<gentype>(xib);
            xzinfountang = new vecInfo;

                          (*xuntang).zeron();
            if ( ilr  ) { (*xuntang).overwritef1(*xfar);       }
            if ( igr  ) { (*xuntang).overwritef2(*xfarfar);    }
            if ( igrR ) { (*xuntang).overwritef3(*xfarfarfar); }

            const Vector<gentype> &iain = (*ineartup).cast_vector();

            int iains = iain.size();

            for ( q = 0 ; q < iains ; ++q )
            {
                (*xuntang).overwriten(x((int) iain(q)),q);
            }

            getKernel().getvecInfo((*xzinfountang),(*xuntang));
        }

        else if ( allocxuntangifneeded && !idiagr && ( ilr && ifartup ) )
        {
            int q;

            // Process indirections

            xuntang      = new SparseVector<gentype>(xib);
            xzinfountang = new vecInfo;

            if ( ilr  ) { (*xuntang).overwriten(*xnear);       }
                          (*xuntang).zerof1();
            if ( igr  ) { (*xuntang).overwritef2(*xfarfar);    }
            if ( igrR ) { (*xuntang).overwritef3(*xfarfarfar); }

            const Vector<gentype> &iaif = (*ifartup).cast_vector();

            int iaifs = iaif.size();

            for ( q = 0 ; q < iaifs ; ++q )
            {
                (*xuntang).overwritef1(x((int) iaif(q)),q);
            }

            getKernel().getvecInfo((*xzinfountang),(*xuntang));
        }

        else if ( allocxuntangifneeded && !idiagr && ( ilr || irr || igr || igrR ) )
        {
            // Process indirections

            xuntang      = new SparseVector<gentype>(xib);
            xzinfountang = new vecInfo;

            if ( ilr  ) { (*xuntang).overwriten(*xnear);       }
            if ( irr  ) { (*xuntang).overwritef1(*xfar);       }
            if ( igr  ) { (*xuntang).overwritef2(*xfarfar);    }
            if ( igrR ) { (*xuntang).overwritef3(*xfarfarfar); }

            getKernel().getvecInfo((*xzinfountang),(*xuntang));
        }

        if ( ( !xnear      || !(*xnear).indsize()      ) &&
             ( !xfar       || !(*xfar).indsize()       ) &&
             ( !xfarfar    || !(*xfarfar).indsize()    ) &&
             ( !xfarfarfar || !(*xfarfarfar).indsize() ) &&
             ( xib.indsize() == 2 ) &&
             ( xib.isindpresent(17) && xib(17).isValVector() ) &&
             ( xib.isindpresent(18) && xib(18).isValVector() ) &&
             ( xib(17).size() == xib(18).size() ) )
        {
            // This is a weighted sum

            methodkey = 2048;

            const Vector<gentype> &sumindraw    = (const Vector<gentype> &) xib(17);
            const Vector<gentype> &sumweightraw = (const Vector<gentype> &) xib(18);

            NiceAssert( sumindraw.size() == sumweightraw.size() );

            sumind.resize(sumindraw.size());
            sumweight.resize(sumweightraw.size());

            int ii;

            for ( ii = 0 ; ii < sumind.size() ; ii++ )
            {
                sumind.sv(ii,(int) sumindraw(ii));
                sumweight.sv(ii,(double) sumweightraw(ii));
            }

            Vector<int> dummysumind;
            Vector<double> dummysumweight;

            int qinear,qifar;
            int qilr,qirr,qigr,qigrR,qiok,qiokr,qgmuL,qgmuR,qidiagr;
            double qrankL,qrankR;
            int qgradOrder,qgradOrderR,qiplanr,qiplan,qiset,qidenseint,qidensederiv;
            double qdiagoffset;
            int qivectset;

            SparseVector<gentype> *qxuntang = nullptr;
            vecInfo *qxzinfountang = nullptr;

            const SparseVector<gentype> *qxnear = nullptr;
            const SparseVector<gentype> *qxfar = nullptr;
            const SparseVector<gentype> *qxfarfar = nullptr;
            const SparseVector<gentype> *qxfarfarfar = nullptr;
            const vecInfo *qxnearinfo = nullptr;
            const vecInfo *qxfarinfo = nullptr;

            const gentype *qineartup = nullptr;
            const gentype *qifartup = nullptr;

            for ( ii = 0 ; ii < sumind.size() ; ii++ )
            {
                methodkey |= detangle_x(qxuntang,qxzinfountang,qxnear,qxfar,qxfarfar,qxfarfarfar,qxnearinfo,qxfarinfo,
                   qinear,qifar,qineartup,qifartup,qilr,qirr,qigr,qigrR,qiokr,qiok,qrankL,qrankR,qgmuL,qgmuR,sumind(ii),qidiagr,&(x()(sumind(ii))),nullptr,
                   qgradOrder,qgradOrderR,qiplanr,qiplan,qiset,qidenseint,qidensederiv,dummysumind,dummysumweight,qdiagoffset,qivectset,usextang,0);

                idenseint   |= qidenseint;
                idensederiv |= qidensederiv;
            }
        }

        return methodkey;
    }





protected:

    // Kernel cache access
    //
    // If a kernel cache is attached to this then this will set isgood = 0 if
    // the value is not in the cache, otherwise isgood = 1.

    virtual double getvalIfPresent_v(int numi, int numj, int &isgood) const;

    // Inner-product cache: over-write this with a non-nullptr return in classes where
    // a kernel cache is available

    virtual int isxymat(const MercerKernel &altK) const { (void) altK; return 0; }
    virtual const Matrix<double> &getxymat(const MercerKernel &altK) const { (void) altK; NiceThrow("xymat not available here!"); const static Matrix<double> dummy; return dummy; }
    virtual const double &getxymatelm(const MercerKernel &altK, int i, int j) const { return getxymat(altK)(i,j); }


private:

    int locNcalc(void) const
    {
        //return altxsrc ? (altxsrc->locNcalc()) : (alltraintarg.size());
        return altxsrc ? (*altxsrc).N() : allxdatagent.size();
    }

    int xvecdim(const SparseVector<gentype> &xa) const
    {
        int xdm = 0;

        if ( ( xa.nupsize() > 1 ) || ( xa.f1upsize() > 1 ) || xa.isf1offindpresent() || xa.isf2offindpresent() || xa.isf4offindpresent() )
        {
            int i,s;
            int dim = 0;

            s = xa.nupsize();

            for ( i = 0 ; i < s ; ++i )
            {
                dim = xa.nupsize(i);
                xdm = ( dim > xdm ) ? dim : xdm;
            }

            s = xa.f1upsize();

            for ( i = 0 ; i < s ; ++i )
            {
                dim = xa.f1upsize(i);
                xdm = ( dim > xdm ) ? dim : xdm;
            }
        }

        else if ( xa.nindsize() )
        {
            xdm = xa.nupsize(0);
        }

        return xdm;
    }

    // Templated to limit code redundancy

    template <class T> T &K0(T &res, const T &bias, const MercerKernel &Kx, const gentype **pxyprod, int resmode) const;
    template <class T> T &K1(T &res, int ia, const T &bias, const MercerKernel &Kx, const gentype **pxyprod, const SparseVector<gentype> *xa, const vecInfo *xainfo, int resmode) const;
    template <class T> T &K2(T &res, int ia, int ib, const T &bias, const MercerKernel &Kx, const gentype **pxyprod, const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, const vecInfo *xainfo, const vecInfo *xbinfo, int resmode) const;
    template <class T> T &K3(T &res, int ia, int ib, int ic, const T &bias, const MercerKernel &Kx, const gentype **pxyprod, const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, const SparseVector<gentype> *xc, const vecInfo *xainfo, const vecInfo *xbinfo, const vecInfo *xcinfo, int resmode) const;
    template <class T> T &K4(T &res, int ia, int ib, int ic, int id, const T &bias, const MercerKernel &Kx, const gentype **pxyprod, const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, const SparseVector<gentype> *xc, const SparseVector<gentype> *xd, const vecInfo *xainfo, const vecInfo *xbinfo, const vecInfo *xcinfo, const vecInfo *xdinfo, int resmode) const;
    template <class T> T &Km(int m, T &res, Vector<int> &ii, const T &bias, const MercerKernel &Kx, const gentype **pxyprod, Vector<const SparseVector<gentype> *> *xx, Vector<const vecInfo *> *xzinfo, int resmode) const;

    // ...and *UNTEMPLATED* later for speed (we don't need to take in a double reference, set it and return it

    double K0(                                double bias, const MercerKernel &altK, const gentype **pxyprod,                                                                                                                                                                                                                                 int resmode) const;
    double K1(int ia,                         double bias, const MercerKernel &altK, const gentype **pxyprod, const SparseVector<gentype> *xa,                                                                                                    const vecInfo *xainfo,                                                                      int resmode) const;
    double K2(int ia, int ib,                 double bias, const MercerKernel &altK, const gentype **pxyprod, const SparseVector<gentype> *xa, const SparseVector<gentype> *xb,                                                                   const vecInfo *xainfo, const vecInfo *xbinfo,                                               int resmode) const;
    double K3(int ia, int ib, int ic,         double bias, const MercerKernel &altK, const gentype **pxyprod, const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, const SparseVector<gentype> *xc,                                  const vecInfo *xainfo, const vecInfo *xbinfo, const vecInfo *xcinfo,                        int resmode) const;
    double K4(int ia, int ib, int ic, int id, double bias, const MercerKernel &altK, const gentype **pxyprod, const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, const SparseVector<gentype> *xc, const SparseVector<gentype> *xd, const vecInfo *xainfo, const vecInfo *xbinfo, const vecInfo *xcinfo, const vecInfo *xdinfo, int resmode) const;
    double Km(int m, Vector<int> &i,          double bias, const MercerKernel &altK, const gentype **pxyprod, Vector<const SparseVector<gentype> *> *xxx, Vector<const vecInfo *> *xxxinfo, int resmode) const;
    double K2x2(int ia, int ib, int ic, double bias, const MercerKernel &altK, const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, const SparseVector<gentype> *xc, const vecInfo *xainfo, const vecInfo *xbinfo, const vecInfo *xcinfo, int resmode) const;


    template <class T> T &K2x2(T &res, int i, int ia, int ib, const T &bias, const MercerKernel &Kx, const SparseVector<gentype> *x, const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, const vecInfo *xinfo, const vecInfo *xainfo, const vecInfo *xbinfo, int resmode) const;

    virtual double KK0ip(                                double bias, const gentype **pxyprod) const;
    virtual double KK1ip(int ia,                         double bias, const gentype **pxyprod, const SparseVector<gentype> *xa, const vecInfo *xainfo) const;
    virtual double KK2ip(int ia, int ib,                 double bias, const gentype **pxyprod, const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, const vecInfo *xainfo, const vecInfo *xbinfo) const;
    virtual double KK3ip(int ia, int ib, int ic,         double bias, const gentype **pxyprod, const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, const SparseVector<gentype> *xc, const vecInfo *xainfo, const vecInfo *xbinfo, const vecInfo *xcinfo) const;
    virtual double KK4ip(int ia, int ib, int ic, int id, double bias, const gentype **pxyprod, const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, const SparseVector<gentype> *xc, const SparseVector<gentype> *xd, const vecInfo *xainfo, const vecInfo *xbinfo, const vecInfo *xcinfo, const vecInfo *xdinfo) const;
    virtual double KKmip(int m, Vector<int> &i, double bias, const gentype **pxyprod, Vector<const SparseVector<gentype> *> *xx, Vector<const vecInfo *> *xzinfo) const;

    template <class T> void dK(T &xygrad, T &xnormgrad, int i, int j, const T &bias, const MercerKernel &Kx, const gentype **pxyprod, const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, const vecInfo *xzinfo, const vecInfo *yyinfo, int deepDeriv) const;
    template <class T> void dK2delx(T &xscaleres, T &yscaleres, int &minmaxind, int i, int j, const T &bias, const MercerKernel &Kx, const gentype **pxyprod, const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, const vecInfo *xainfo, const vecInfo *yyinfo) const;

    template <class T> void d2K(T &xygrad, T &xnormgrad, T &xyxygrad, T &xyxnormgrad, T &xyynormgrad, T &xnormxnormgrad, T &xnormynormgrad, T &ynormynormgrad, int &minmaxind, int i, int j, const T &bias, const MercerKernel &altK, const gentype **pxyprod, const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, const vecInfo *xainfo, const vecInfo *yyinfo) const;
    template <class T> void d2K2delxdelx(T &xxscaleres, T &yyscaleres, T &xyscaleres, T & yxscaleres, T &constres, int &minmaxind, int i, int j, const T &bias, const MercerKernel &Kx, const gentype **pxyprod, const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, const vecInfo *xainfo, const vecInfo *yyinfo) const;
    template <class T> void d2K2delxdely(T &xxscaleres, T &yyscaleres, T &xyscaleres, T & yxscaleres, T &constres, int &minmaxind, int i, int j, const T &bias, const MercerKernel &Kx, const gentype **pxyprod, const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, const vecInfo *xainfo, const vecInfo *yyinfo) const;

    template <class T> void dnK2del(Vector<T> &sc, Vector<Vector<int> > &n, int &minmaxind, const Vector<int> &q, int i, int j, const T &bias, const MercerKernel &Kx, const gentype **pxyprod, const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, const vecInfo *xainfo, const vecInfo *yyinfo) const;

    // Base data

    Vector<SparseVector<gentype> > allxdatagent;
//FIXME    Vector<const SparseVector<gentype> *> allxdatagentp;
    MercerKernel kernel;
    mutable gentype ytargdata;
    mutable double ytargdataR;
    mutable d_anion ytargdataA;
    mutable Vector<double> ytargdataV;
    mutable gentype ytargdatap;
    mutable double ytargdatapR;
    mutable d_anion ytargdatapA;
    mutable Vector<double> ytargdatapV;
    Vector<gentype> alltraintarg;
    Vector<double> alltraintargR;
    Vector<d_anion> alltraintargA;
    Vector<Vector<double> > alltraintargV;
    Vector<gentype> alltraintargp;
    Vector<double> alltraintargpR;
    Vector<d_anion> alltraintargpA;
    Vector<Vector<double> > alltraintargpV;
    Vector<vecInfo> traininfo;
    Vector<int> traintang;
//FIXME    Vector<const vecInfo *> traininfop;
    Vector<int> xd;
    Vector<double> xCweight;
    Vector<double> xCweightfuzz;
    Vector<double> xepsweight;
    Vector<int> xalphaState;
    Matrix<gentype> K2mat;

    int xdzero; // number of elements in xd that are zero

    int xpreallocsize;

    int xmuprior;
    gentype xmuprior_gt;
    const ML_Base *xmuprior_ml;

    // default sigma (variance)

    double locsigma;

    // Base data extended

    double loclr;
    double loclrb;
    double loclrc;
    double loclrd;

    // Output kernel

    MercerKernel UUoutkernel;

    // RFF feature kernel

    MercerKernel RFFkernel;

    // isBasisUser: controls if basis is user controlled (0, default) or fixed to equal y (ie target set).
    // defbasis: -1 if not defined, otherwise default basis onto which projection is done (assumed unit)
    // locbasis: basis

    int isBasisUserUU;
    int defbasisUU;
    Vector<gentype> locbasisUU;

    int isBasisUserVV;
    int defbasisVV;
    Vector<gentype> locbasisVV;

    gentype &(*UUcallback)(gentype &res, int m, const ML_Base &caller, Vector<int> &iokr, Vector<int> &iok, Vector<const gentype *> xalt, int defbasis);
    const gentype &(*VVcallback)(gentype &res, int m, const gentype &kval, const ML_Base &caller, Vector<int> &iokr, Vector<int> &iok, Vector<const gentype *> xalt, int defbasis);

    // Not used in older classes that implemented zerotol locally

    double globalzerotol;

    // Temporary store so that callback knows what data element -1 is

    mutable const SparseVector<gentype> *wildxgenta;
    mutable const SparseVector<gentype> *wildxgentb;
    mutable const SparseVector<gentype> *wildxgentc;
    mutable const SparseVector<gentype> *wildxgentd;

    mutable const vecInfo *wildxinfoa;
    mutable const vecInfo *wildxinfob;
    mutable const vecInfo *wildxinfoc;
    mutable const vecInfo *wildxinfod;

    mutable vecInfo *allocxinfoa;
    mutable vecInfo *allocxinfob;
    mutable vecInfo *allocxinfoc;
    mutable vecInfo *allocxinfod;

    mutable int wildxtanga;
    mutable int wildxtangb;
    mutable int wildxtangc;
    mutable int wildxtangd;

    mutable int wildxdima;
    mutable int wildxdimb;
    mutable int wildxdimc;
    mutable int wildxdimd;

    mutable const Vector<SparseVector<gentype> > *wildxxgent;
    mutable Vector<vecInfo> wildxxinfo;
    mutable Vector<int> wildxxtang;
    mutable int wildxxdim;

    mutable int wildxdim;

    // cachey stuff

    mutable Vector<double> yCweight;
    mutable Vector<double> yCweightfuzz;
    mutable Vector<double> ysigmaweight;
    mutable Vector<double> yepsweight;
    mutable Vector<double> xsigmaweightonfly;

    // Training data information:
    //
    // indexKey: which indices (in the sparse training vectors) are in use.
    // indexKeyCount: how often each index is used.
    // typeKey: what data type is in each index
    // typeKeyBreak: a detailed breakdown of the number of vectors in each type.
    //
    // isIndPrune: set 1 to indicate that all vectors should have the same
    //             index (null full as required) and that indices that only
    //             have nulls should be removed.
    //
    // typeKey: - 0:  mixture of the below categories
    //          - 1:  all null
    //          - 2:  all binary integer 0/1 (or null (0))
    //          - 3:  all integers (or null (0))
    //          - 4:  all doubles (or integers or null)
    //          - 5:  all anions (or doubles or integers or null)
    //          - 6:  all vectors of doubles
    //          - 7:  all matrices of doubles
    //          - 8:  all sets
    //          - 9:  all dgraphs
    //          - 10: all strings
    //          - 11: all equations
    //
    // typeKeyBreak: - 0:  sum of the below categories
    //               - 1:  null
    //               - 2:  binary integer 1
    //               - 3:  integer not one
    //               - 4:  double
    //               - 5:  anion
    //               - 6:  vector
    //               - 7:  matrix
    //               - 8:  set
    //               - 9:  dgraph
    //               - 10: string
    //               - 11: equations

    SparseVector<Vector<int> > indexKey;
    SparseVector<Vector<int> > indexKeyCount;
    SparseVector<Vector<int> > typeKey;
    SparseVector<Vector<Vector<int> > > typeKeyBreak;

    // Each ML_Base instantiated has a unique ID, and corresponding to
    // that ID are x and g version numbers (see above).  These are
    // shared.

    static thread_local SparseVector<int> xvernumber;
    static thread_local SparseVector<int> gvernumber;
//#ifdef ENABLE_THREADS
//    static std::mutex mleyelock;
//#endif

    // indPrune: 0 by default, 1 to indicate that x should be "filled out"
    //           with nulls so that each training vector has the same
    //           index vector.  This will also cause pruning of indexes
    //           that contain only nulls.
    // xassumedconsist: 0 by default, 1 to indicate that x can safely be assumed to
    //           be consistent throughout (ie all sparse vectors have the
    //           same index vector)

    int isIndPrune;
    int xassumedconsist;
    int xconsist;

    virtual int indPrune(void) const { NiceAssert( ( isIndPrune == 0 ) || ( isIndPrune == 1 ) ); return isIndPrune; }

    // unfillIndex: Call this function to remove an index from all vectors
    //           in the training set.  Note that this does not actually
    //           remove the index itself (that is the job of the caller).
    // fillIndex: Call this function to ensure an index is present in all
    //           training vectors.  Note that index must be present before
    //           calling.

    void unfillIndex(int i);
    void fillIndex(int i);

    // Functions to update index information for addition/removal
    // (u=-1 gives overall dim, u>=0 gives only dimension for relevant minor/up type - see sparsevector).

    void addToIndexKeyAndUpdate(const SparseVector<gentype> &newz, int u = -1);
    void removeFromIndexKeyAndUpdate(const SparseVector<gentype> &oldz, int u = -1);

    // Returns the appropriate index type corresponding to variable y

    int gettypeind(const gentype &y) const;

    // Alternate x source (nullptr if data local)

    const ML_Base *altxsrc;

    ML_Base **that;

    // Fixes x pointer vector

    void fixpvects(void)
    {
;
//FIXME
/*
        if ( allxdatagent.size() )
        {
            int i;

            allxdatagentp.resize(allxdatagent.size());

            for ( i = 0 ; i < allxdatagent.size() ; ++i )
            {
                allxdatagentp("&",i) = &allxdatagent(i);
            }
        }

        if ( traininfo.size() )
        {
            int i;

            traininfop.resize(traininfo.size());

            for ( i = 0 ; i < traininfo.size() ; ++i )
            {
                traininfop("&",i) = &traininfo(i);
            }
        }
*/
    }

    // Test for x consistency (same indices for all)

    int testxconsist(void)
    {
        if ( indKeyCount().size() )
        {
            return indKeyCount() == indKeyCount()(0);
        }

        return 1;
    }

    // Functions to control wilds

    virtual void setInnerWildpa(const SparseVector<gentype> *xl, const vecInfo *xinf = nullptr) const
    {
        (*xl).makealtcontent();

        wildxgenta = xl;

        if ( ( xinf == nullptr ) && ( type() == 216 ) )
        {
            // For BLK_Batter a *lot* of data could be in x and xinfo is not used, so don't waste time here!

            MEMNEW(allocxinfoa,vecInfo);
            wildxinfoa = allocxinfoa;
        }

        else if ( xinf == nullptr )
        {
            MEMNEW(allocxinfoa,vecInfo);
            wildxinfoa = allocxinfoa;

            getKernel().getvecInfo(*allocxinfoa,*xl);
        }

        else
        {
            wildxinfoa = xinf;
        }

        wildxdima = xvecdim(*wildxgenta);

        wildxdim = 0;
        wildxdim = ( wildxdima > wildxdim ) ? wildxdima : wildxdim;
        wildxdim = ( wildxdimb > wildxdim ) ? wildxdimb : wildxdim;
        wildxdim = ( wildxdimc > wildxdim ) ? wildxdimc : wildxdim;
        wildxdim = ( wildxdimd > wildxdim ) ? wildxdimd : wildxdim;
        wildxdim = ( wildxxdim > wildxdim ) ? wildxxdim : wildxdim;

        wildxtanga = detangle_x(-1);
        wildxaReal = isxreal(-1);

        if ( !wildxaReal )
        {
            calcSetAssumeReal(0);
        }

        {
            calcprior(ytargdatap,*wildxgenta,wildxinfoa);

            ytargdatapR = (double) ytargdatap;
            ytargdatapA = (const d_anion &) ytargdatap;
            ytargdatapV = (const Vector<double> &) ytargdatap;
        }
    }

    virtual void setInnerWildpb(const SparseVector<gentype> *xl, const vecInfo *xinf = nullptr) const
    {
        (*xl).makealtcontent();

        wildxgentb = xl;

        if ( ( xinf == nullptr ) && ( type() == 216 ) )
        {
            // For BLK_Batter a *lot* of data could be in x and xinfo is not used, so don't waste time here!

            MEMNEW(allocxinfob,vecInfo);
            wildxinfob = allocxinfob;
        }

        else if ( xinf == nullptr )
        {
            MEMNEW(allocxinfob,vecInfo);
            wildxinfob = allocxinfob;

            getKernel().getvecInfo(*allocxinfob,*xl);
        }

        else
        {
            wildxinfob = xinf;
        }

        wildxdimb = xvecdim(*wildxgentb);

        wildxdim = 0;
        wildxdim = ( wildxdima > wildxdim ) ? wildxdima : wildxdim;
        wildxdim = ( wildxdimb > wildxdim ) ? wildxdimb : wildxdim;
        wildxdim = ( wildxdimc > wildxdim ) ? wildxdimc : wildxdim;
        wildxdim = ( wildxdimd > wildxdim ) ? wildxdimd : wildxdim;
        wildxdim = ( wildxxdim > wildxdim ) ? wildxxdim : wildxdim;

        wildxtangb = detangle_x(-3);
        wildxbReal = isxreal(-3);

        if ( !wildxbReal )
        {
            calcSetAssumeReal(0);
        }
    }

    virtual void setInnerWildpc(const SparseVector<gentype> *xl, const vecInfo *xinf = nullptr) const
    {
        (*xl).makealtcontent();

        wildxgentc = xl;

        if ( ( xinf == nullptr ) && ( type() == 216 ) )
        {
            // For BLK_Batter a *lot* of data could be in x and xinfo is not used, so don't waste time here!

            MEMNEW(allocxinfoc,vecInfo);
            wildxinfoc = allocxinfoc;
        }

        else if ( xinf == nullptr )
        {
            MEMNEW(allocxinfoc,vecInfo);
            wildxinfoc = allocxinfoc;

            getKernel().getvecInfo(*allocxinfoc,*xl);
        }

        else
        {
            wildxinfoc = xinf;
        }

        wildxdimc = xvecdim(*wildxgentc);

        wildxdim = 0;
        wildxdim = ( wildxdima > wildxdim ) ? wildxdima : wildxdim;
        wildxdim = ( wildxdimb > wildxdim ) ? wildxdimb : wildxdim;
        wildxdim = ( wildxdimc > wildxdim ) ? wildxdimc : wildxdim;
        wildxdim = ( wildxdimd > wildxdim ) ? wildxdimd : wildxdim;
        wildxdim = ( wildxxdim > wildxdim ) ? wildxxdim : wildxdim;

        wildxtangc = detangle_x(-4);
        wildxcReal = isxreal(-4);

        if ( !wildxcReal )
        {
            calcSetAssumeReal(0);
        }
    }

    virtual void setInnerWildpd(const SparseVector<gentype> *xl, const vecInfo *xinf = nullptr) const
    {
        (*xl).makealtcontent();

        wildxgentd = xl;

        if ( ( xinf == nullptr ) && ( type() == 216 ) )
        {
            // For BLK_Batter a *lot* of data could be in x and xinfo is not used, so don't waste time here!

            MEMNEW(allocxinfod,vecInfo);
            wildxinfod = allocxinfod;
        }

        else if ( xinf == nullptr )
        {
            MEMNEW(allocxinfod,vecInfo);
            wildxinfod = allocxinfod;

            getKernel().getvecInfo(*allocxinfod,*xl);
        }

        else
        {
            wildxinfod = xinf;
        }

        wildxdimd = xvecdim(*wildxgentd);

        wildxdim = 0;
        wildxdim = ( wildxdima > wildxdim ) ? wildxdima : wildxdim;
        wildxdim = ( wildxdimb > wildxdim ) ? wildxdimb : wildxdim;
        wildxdim = ( wildxdimc > wildxdim ) ? wildxdimc : wildxdim;
        wildxdim = ( wildxdimd > wildxdim ) ? wildxdimd : wildxdim;
        wildxdim = ( wildxxdim > wildxdim ) ? wildxxdim : wildxdim;

        wildxtangd = detangle_x(-5);
        wildxdReal = isxreal(-5);

        if ( !wildxdReal )
        {
            calcSetAssumeReal(0);
        }
    }

    virtual void setInnerWildpx(const Vector<SparseVector<gentype> > *xl) const
    {
        int i;

        wildxxgent = xl;
        wildxxinfo.resize((*xl).size());
        wildxxtang.resize((*xl).size());

        for ( i = 0 ; i < (*xl).size() ; ++i )
        {
            ((*xl)(i)).makealtcontent();

            getKernel().getvecInfo(wildxxinfo("&",i),(*xl)(i));
        }

        wildxxdim = 0;

        //if ( (*wildxxgent).size() )
        {
            int dimx;

            for ( i = 0 ; i < (*wildxxgent).size() ; ++i )
            {
                dimx = xvecdim((*wildxxgent)(i));
                wildxxdim = ( wildxxdim > dimx ) ? wildxxdim : dimx;
            }
        }

        wildxdim = 0;
        wildxdim = ( wildxdima > wildxdim ) ? wildxdima : wildxdim;
        wildxdim = ( wildxdimb > wildxdim ) ? wildxdimb : wildxdim;
        wildxdim = ( wildxdimc > wildxdim ) ? wildxdimc : wildxdim;
        wildxdim = ( wildxdimd > wildxdim ) ? wildxdimd : wildxdim;
        wildxdim = ( wildxxdim > wildxdim ) ? wildxxdim : wildxdim;

        wildxxReal = 1;

        for ( i = 0 ; i < (*xl).size() ; ++i )
        {
            wildxxtang.sv(i,detangle_x(-100*(i+1)));
            wildxxReal = ( wildxxReal && isxreal(-100*(i+1)) );
        }

        if ( !wildxxReal )
        {
            calcSetAssumeReal(0);
        }
    }

    virtual void setWildTargpp(const gentype &yI) const
    {
        ytargdata  = yI;

        ytargdataR = (double) ytargdata;
        ytargdataA = (const d_anion &) ytargdata;
        ytargdataV = (const Vector<double> &) ytargdata;
    }

    virtual void resetInnerWildp(int wasnulla = 0, int wasnullb = 0, int wasnullc = 0, int wasnulld = 0) const
    {
        wildxgenta = nullptr;
        wildxgentb = nullptr;
        wildxgentc = nullptr;
        wildxgentd = nullptr;
        wildxxgent = nullptr;

        if ( allocxinfoa && wasnulla )
        {
            MEMDEL(allocxinfoa);
        }

        if ( allocxinfob && wasnullb )
        {
            MEMDEL(allocxinfob);
        }

        if ( allocxinfoc && wasnullc )
        {
            MEMDEL(allocxinfoc);
        }

        if ( allocxinfod && wasnulld )
        {
            MEMDEL(allocxinfod);
        }

        wildxinfoa = nullptr;
        wildxinfob = nullptr;
        wildxinfoc = nullptr;
        wildxinfod = nullptr;

        allocxinfoa = nullptr;
        allocxinfob = nullptr;
        allocxinfoc = nullptr;
        allocxinfod = nullptr;

        wildxdim = 0;

        wildxdima = 0;
        wildxdimb = 0;
        wildxdimc = 0;
        wildxdimd = 0;
        wildxxdim = 0;

        wildxaReal = 1;
        wildxbReal = 1;
        wildxcReal = 1;
        wildxdReal = 1;
        wildxxReal = 1;

        calcSetAssumeReal(0);

        //(wildxxinfo).resize(0); - slight speedup by not resizing as the same size is often repeated
    }


    // Local x retrieval function

    virtual const SparseVector<gentype> &xgetloc(int i) const
    {
        if ( ( i >= 0 ) && altxsrc )
        {
            return (*altxsrc).x(i);
        }

        else if ( i >= 0 )
        {
            return allxdatagent(i);
        }

        else if ( i == -1 )
        {
            // Testing vector

            return *wildxgenta;
        }

//        else if ( i == -2 )
//        {
//            // ONN weight vector (error-throwing placeholder unless ONN type).
//
//            return W();
//        }

        else if ( i == -3 )
        {
            // Testing vector

            return *wildxgentb;
        }

        else if ( i == -4 )
        {
            // Testing vector

            return *wildxgentc;
        }

        else if ( i == -5 )
        {
            // Testing vector

            return *wildxgentd;
        }

        else if ( ( i <= -100 ) && !((-i)%100) )
        {
            // Testing vector
            //
            // -100 -> 0
            // -200 -> 1
            // -300 -> 2
            // -400 -> 3
            //   ...

            return (*wildxxgent)((-(i+100))/100);
        }

        else if ( ( i <= -100 ) && !((-(i+1))%100) )
        {
            // faroff part to be used (but return as usual)
            //
            // -101 -> 0
            // -201 -> 1
            // -301 -> 2
            // -401 -> 3
            //   ...

            return xgetloc((-(i+101))/100);
        }

        NiceThrow("Error: xgetloc index not valid");

        const static SparseVector<gentype> temp;

        return temp;
    }

    virtual int locxtang(int i) const
    {
        if ( ( i >= 0 ) && altxsrc )
        {
            return (*altxsrc).xtang(i);
        }

        else if ( i >= 0 )
        {
            return xtang()(i);
        }

        else if ( i == -1 )
        {
            return wildxtanga;
        }

//        else if ( i == -2 )
//        {
//            return getWtang();
//        }

        else if ( i == -3 )
        {
            return wildxtangb;
        }

        else if ( i == -4 )
        {
            return wildxtangc;
        }

        else if ( i == -5 )
        {
            return wildxtangd;
        }

        else if ( ( i <= -100 ) && !((-i)%100) )
        {
            // Testing vector
            //
            // -100 -> 0
            // -200 -> 1
            // -300 -> 2
            // -400 -> 3
            //   ...

            return wildxxtang((-(i+100))/100);
        }

        else if ( ( i <= -100 ) && !((-(i+1))%100) )
        {
            // faroff part to be used (but return as usual)
            //
            // -101 -> 0
            // -201 -> 1
            // -301 -> 2
            // -401 -> 3
            //   ...

            return locxtang((-(i+101))/100);
        }

        NiceThrow("Error: x info for invalid index requested");

        return -1;
    }

    virtual const vecInfo &locxinfo(int i) const
    {
        if ( ( i >= 0 ) && altxsrc )
        {
            return (*altxsrc).xinfo(i);
        }

        else if ( i >= 0 )
        {
            return traininfo(i);
        }

        else if ( i == -1 )
        {
            return *wildxinfoa;
        }

//        else if ( i == -2 )
//        {
//            return getWinfo();
//        }

        else if ( i == -3 )
        {
            return *wildxinfob;
        }

        else if ( i == -4 )
        {
            return *wildxinfoc;
        }

        else if ( i == -5 )
        {
            return *wildxinfod;
        }

        else if ( ( i <= -100 ) && !((-i)%100) )
        {
            // Testing vector
            //
            // -100 -> 0
            // -200 -> 1
            // -300 -> 2
            // -400 -> 3
            //   ...

            return wildxxinfo((-(i+100))/100);
        }

        else if ( ( i <= -100 ) && !((-(i+1))%100) )
        {
            // faroff part to be used (but return as usual)
            //
            // -101 -> 0
            // -201 -> 1
            // -301 -> 2
            // -401 -> 3
            //   ...

            return locxinfo((-(i+101))/100);
        }

        NiceThrow("Error: x info for invalid index requested");

        const static vecInfo temp;

        return temp;
    }

    mutable int assumeReal;
    mutable int trainingDataReal;

    mutable int wildxaReal;
    mutable int wildxbReal;
    mutable int wildxcReal;
    mutable int wildxdReal;
    mutable int wildxxReal;

    int isxreal(int i) const
    {
        int res = 1;
        int j,k;

        const SparseVector<gentype> &xx = x(i);

        // If altcontent is set then clearly we know that the vector is real
        // If the vector is empty then it is effectively real by default.

        if ( ( xx.altcontent == nullptr ) && ( xx.altcontentsp == nullptr ) && xx.indsize() )
        {
            for ( j = 0 ; j < xx.indsize() ; ++j )
            {
                k = gettypeind(xx.direcref(j));

                if ( !( ( k >= 2 ) && ( k <= 4 ) ) )
                {
                    res = 0;

                    break;
                }
            }
        }

        return res;
    }

    void calcSetXdim(void)
    {
        getKernel_unsafe().setdefindKey(indKey());
    }

    void calcSetAssumeReal(int fulltest = 1, int assumeUnreal = 0) const
    {
        if ( fulltest )
        {
            if ( dattypeKey().size() == 0 )
            {
                trainingDataReal = 1;
            }

            else if ( ( dattypeKey() <= 4 ) && ( dattypeKey() >= 2 ) )
            {
                trainingDataReal = 1;
            }

            else
            {
                trainingDataReal = 0;
            }
        }

        if ( !assumeReal && ( !assumeUnreal && trainingDataReal && wildxaReal && wildxbReal && wildxcReal && wildxdReal && wildxxReal ) )
        {
            assumeReal = 1;
        }

        else if ( assumeReal && !( !assumeUnreal && trainingDataReal && wildxaReal && wildxbReal && wildxcReal && wildxdReal && wildxxReal ) )
        {
            assumeReal = 0;
        }
    }

public:
    // Evaluation of 8x6 kernels

    virtual void fastg(gentype &res) const;
    virtual void fastg(gentype &res, int ia, const SparseVector<gentype> &xa, const vecInfo &xainfo) const;
    virtual void fastg(gentype &res, int ia, int ib, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo) const;
    virtual void fastg(gentype &res, int ia, int ib, int ic, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo) const;
    virtual void fastg(gentype &res, int ia, int ib, int ic, int id, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo) const;
    virtual void fastg(gentype &res, Vector<int> &ia, Vector<const SparseVector<gentype> *> &xa, Vector<const vecInfo *> &xainfo) const;

    virtual void fastg(double &res) const;
    virtual void fastg(double &res, int ia, const SparseVector<gentype> &xa, const vecInfo &xainfo) const;
    virtual void fastg(double &res, int ia, int ib, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo) const;
    virtual void fastg(double &res, int ia, int ib, int ic, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo) const;
    virtual void fastg(double &res, int ia, int ib, int ic, int id, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo) const;
    virtual void fastg(double &res, Vector<int> &ia, Vector<const SparseVector<gentype> *> &xa, Vector<const vecInfo *> &xainfo) const;
};



inline double norm2(const ML_Base &a);
inline double abs2 (const ML_Base &a);

inline double norm2(const ML_Base &a) { return a.RKHSnorm(); }
inline double abs2 (const ML_Base &a) { return a.RKHSabs();  }

inline void qswap(ML_Base &a, ML_Base &b)
{
    a.qswapinternal(b);
}

inline ML_Base &setzero(ML_Base &a)
{
    a.restart();

    return a;
}

inline std::ostream &operator<<(std::ostream &output, const ML_Base &src)
{
    return src.printstream(output,0);
}

inline std::istream &operator>>(std::istream &input, ML_Base &dest)
{
    return dest.inputstream(input);
}

inline ML_Base *&setzero(ML_Base *&x)
{
    return ( x = nullptr );
}

inline const ML_Base *&setzero(const ML_Base *&x)
{
    return ( x = nullptr );
}

inline void qswap(ML_Base *&a, ML_Base *&b)
{
    ML_Base *temp;

    temp = a;
    a = b;
    b = temp;
}

inline int isQswapCompat(const ML_Base &a, const ML_Base &b)
{
    if ( ( a.isPool()    == b.isPool()    ) &&
         ( a.isMutable() == b.isMutable() ) &&
         ( a.isPool()    || a.isMutable() )    )
    {
        return 1;
    }

    return ( a.type()    == b.type()    ) &&
           ( a.subtype() == b.subtype() );
}

inline int isSemicopyCompat(const ML_Base &a, const ML_Base &b)
{
    return ( a.type()    == b.type()    ) &&
           ( a.subtype() == b.subtype() );
}

inline int isAssignCompat(const ML_Base &a, const ML_Base &b)
{
    if ( a.isPool() && b.isPool() )
    {
        return 1;
    }

    if ( a.isMutable() )
    {
        return 1;
    }

    switch ( a.type() + 10000*b.type() )
    {
        case     0:  { return 1; break; }
        case     1:  { return 0; break; }
        case     2:  { return 0; break; }
        case 10000:  { return 1; break; }
        case 10001:  { return 1; break; }
        case 10002:  { return 0; break; }
        case 20000:  { return 1; break; }
        case 20001:  { return 1; break; }
        case 20002:  { return 1; break; }

        default:
        {
            break;
        }
    }

    return ( a.type() == b.type() );
}

inline void qswap(const ML_Base *&a, const ML_Base *&b)
{
    const ML_Base *temp;

    temp = a;
    a = b;
    b = temp;
}

inline void ML_Base::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    ML_Base &b = bb.getML();

    kernPrecursor::qswapinternal(b);

    qswap(kernel         ,b.kernel         );
    qswap(UUoutkernel    ,b.UUoutkernel    );
    qswap(RFFkernel      ,b.RFFkernel      );
    qswap(allxdatagent   ,b.allxdatagent   );
//FIXME    qswap(allxdatagentp  ,b.allxdatagentp  );
    qswap(ytargdata      ,b.ytargdata      );
    qswap(ytargdataR     ,b.ytargdataR     );
    qswap(ytargdataA     ,b.ytargdataA     );
    qswap(ytargdataV     ,b.ytargdataV     );
    qswap(ytargdatap     ,b.ytargdatap     );
    qswap(ytargdatapR    ,b.ytargdatapR    );
    qswap(ytargdatapA    ,b.ytargdatapA    );
    qswap(ytargdatapV    ,b.ytargdatapV    );
    qswap(alltraintarg   ,b.alltraintarg   );
    qswap(alltraintargp  ,b.alltraintargp  );
    qswap(alltraintargR  ,b.alltraintargR  );
    qswap(alltraintargA  ,b.alltraintargA  );
    qswap(alltraintargV  ,b.alltraintargV  );
    qswap(alltraintargpR ,b.alltraintargpR );
    qswap(alltraintargpA ,b.alltraintargpA );
    qswap(alltraintargpV ,b.alltraintargpV );
    qswap(xd             ,b.xd             );
    qswap(xdzero         ,b.xdzero         );
    qswap(locsigma       ,b.locsigma       );
    qswap(loclr          ,b.loclr          );
    qswap(loclrb         ,b.loclrb         );
    qswap(loclrc         ,b.loclrc         );
    qswap(loclrd         ,b.loclrd         );
    qswap(xCweight       ,b.xCweight       );
    qswap(xCweightfuzz   ,b.xCweightfuzz   );
    qswap(xepsweight     ,b.xepsweight     );
    qswap(traininfo      ,b.traininfo      );
    qswap(traintang      ,b.traintang      );
//FIXME    qswap(traininfop     ,b.traininfop     );
    qswap(xalphaState    ,b.xalphaState    );
    qswap(indexKey       ,b.indexKey       );
    qswap(indexKeyCount  ,b.indexKeyCount  );
    qswap(typeKey        ,b.typeKey        );
    qswap(typeKeyBreak   ,b.typeKeyBreak   );
    qswap(isIndPrune     ,b.isIndPrune     );
    qswap(xassumedconsist,b.xassumedconsist);
    qswap(xconsist       ,b.xconsist       );
    qswap(globalzerotol  ,b.globalzerotol  );
    qswap(isBasisUserUU  ,b.isBasisUserUU  );
    qswap(defbasisUU     ,b.defbasisUU     );
    qswap(locbasisUU     ,b.locbasisUU     );
    qswap(isBasisUserVV  ,b.isBasisUserVV  );
    qswap(defbasisVV     ,b.defbasisVV     );
    qswap(locbasisVV     ,b.locbasisVV     );
    qswap(xpreallocsize  ,b.xpreallocsize  );
    qswap(K2mat          ,b.K2mat          );

    qswap(xmuprior   ,b.xmuprior);
    qswap(xmuprior_gt,b.xmuprior_gt);
    qswap(xmuprior_ml,b.xmuprior_ml);

    qswap(assumeReal      ,b.assumeReal      );
    qswap(trainingDataReal,b.trainingDataReal);
    qswap(wildxaReal      ,b.wildxaReal      );
    qswap(wildxbReal      ,b.wildxbReal      );
    qswap(wildxcReal      ,b.wildxcReal      );
    qswap(wildxdReal      ,b.wildxdReal      );
    qswap(wildxxReal      ,b.wildxxReal      );

    //incxvernum();
    //b.incxvernum();

    //incgvernum();
    //b.incgvernum();

    gentype &(*UUcallbackxx)(gentype &res, int m, const ML_Base &caller, Vector<int> &iokr, Vector<int> &iok, Vector<const gentype *> xalt, int defbasis);
    const gentype &(*VVcallbackxx)(gentype &res, int m, const gentype &kval, const ML_Base &caller, Vector<int> &iokr, Vector<int> &iok, Vector<const gentype *> xalt, int defbasis);

    UUcallbackxx = UUcallback;
    UUcallback   = b.UUcallback;
    b.UUcallback = UUcallbackxx;

    VVcallbackxx = VVcallback;
    VVcallback   = b.VVcallback;
    b.VVcallback = VVcallbackxx;
}

inline void ML_Base::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const ML_Base &b = bb.getMLconst();

    kernPrecursor::semicopy(b);

    //kernel
    //UUoutkernel
    //RFFkernel
    //traininfo
    //allxdatagent
    //ytargdata
    //ytargdataR
    //ytargdataA
    //ytargdataV
    //ytargdatap
    //ytargdatapR
    //ytargdatapA
    //ytargdatapV
    //indexKey
    //indexKeyCount
    //typeKey
    //typeKeyBreak
    //isIndPrune
    //xassumedconsist
    //xconsist
    //xCweight
    //xCweightfuzz
    //xepsweight
    //isBasisUser
    //locbasis
    //K2mat

    xmuprior    = b.xmuprior;
    xmuprior_gt = b.xmuprior_gt;
    xmuprior_ml = b.xmuprior_ml;

    xd             = b.xd;
    xdzero         = b.xdzero;
    locsigma       = b.locsigma;
    loclr          = b.loclr;
    loclrb         = b.loclrb;
    loclrc         = b.loclrc;
    loclrd         = b.loclrd;
    alltraintarg   = b.alltraintarg;
    alltraintargp  = b.alltraintargp; // evaluated prior
    alltraintargR  = b.alltraintargR;
    alltraintargA  = b.alltraintargA;
    alltraintargV  = b.alltraintargV;
    alltraintargpR = b.alltraintargpR; // evaluated prior
    alltraintargpA = b.alltraintargpA; // evaluated prior
    alltraintargpV = b.alltraintargpV; // evaluated prior
    globalzerotol  = b.globalzerotol;
    defbasisUU     = b.defbasisUU;
    defbasisVV     = b.defbasisVV;
    UUcallback     = b.UUcallback;
    VVcallback     = b.VVcallback;

    // These are in svm_generic, where it is relevant
    //incxvernum();
    //incgvernum();
}

inline void ML_Base::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const ML_Base &src = bb.getMLconst();

    kernPrecursor::assign(src);

    kernel        = src.kernel;
    UUoutkernel   = src.UUoutkernel;
    RFFkernel     = src.RFFkernel;
    xalphaState   = src.xalphaState;
    globalzerotol = src.globalzerotol;

    isBasisUserUU = src.isBasisUserUU;
    defbasisUU    = src.defbasisUU;
    locbasisUU    = src.locbasisUU;
    isBasisUserVV = src.isBasisUserVV;
    defbasisVV    = src.defbasisVV;
    locbasisVV    = src.locbasisVV;
    UUcallback    = src.UUcallback;
    VVcallback    = src.VVcallback;

    assumeReal       = src.assumeReal;
    trainingDataReal = src.trainingDataReal;
    wildxaReal       = src.wildxaReal;
    wildxbReal       = src.wildxbReal;
    wildxcReal       = src.wildxcReal;
    wildxdReal       = src.wildxdReal;
    wildxxReal       = src.wildxxReal;

    if ( !onlySemiCopy || ( ( -1 == onlySemiCopy ) && !altxsrc ) )
    {
        // ful copy

        allxdatagent  = src.allxdatagent;
        traininfo     = src.traininfo;
        traintang     = src.traintang;
        ytargdata     = src.ytargdata;
        ytargdataR    = src.ytargdataR;
        ytargdataA    = src.ytargdataA;
        ytargdataV    = src.ytargdataV;
        ytargdatap    = src.ytargdatap;
        ytargdatapR   = src.ytargdatapR;
        ytargdatapA   = src.ytargdatapA;
        ytargdatapV   = src.ytargdatapV;

        alltraintarg   = src.alltraintarg;
        alltraintargp  = src.alltraintargp;
        alltraintargR  = src.alltraintargR;
        alltraintargA  = src.alltraintargA;
        alltraintargV  = src.alltraintargV;
        alltraintargpR = src.alltraintargpR;
        alltraintargpA = src.alltraintargpA;
        alltraintargpV = src.alltraintargpV;

        xd           = src.xd;
        xdzero       = src.xdzero;
        locsigma     = src.locsigma;
        loclr        = src.loclr;
        loclrb       = src.loclrb;
        loclrc       = src.loclrc;
        loclrd       = src.loclrd;
        xCweight     = src.xCweight;
        xCweightfuzz = src.xCweightfuzz;
        xepsweight   = src.xepsweight;
        K2mat        = src.K2mat;

        xmuprior    = src.xmuprior;
        xmuprior_gt = src.xmuprior_gt;
        xmuprior_ml = src.xmuprior_ml;

        fixpvects();
    }

    else if ( -1 == onlySemiCopy )
    {
        // full copy, but without x data (which is in altxsrc)

        allxdatagent.resize((src.allxdatagent).size());
        traininfo.resize((src.traininfo).size());
        traintang.resize((src.traintang).size());
        ytargdata   = src.ytargdata;
        ytargdataR  = src.ytargdataR;
        ytargdataA  = src.ytargdataA;
        ytargdataV  = src.ytargdataV;
        ytargdatap  = src.ytargdatap;
        ytargdatapR = src.ytargdatapR;
        ytargdatapA = src.ytargdatapA;
        ytargdatapV = src.ytargdatapV;

        alltraintarg   = src.alltraintarg;
        alltraintargp  = src.alltraintargp;
        alltraintargR  = src.alltraintargR;
        alltraintargA  = src.alltraintargA;
        alltraintargV  = src.alltraintargV;
        alltraintargpR = src.alltraintargpR;
        alltraintargpA = src.alltraintargpA;
        alltraintargpV = src.alltraintargpV;

        xd           = src.xd;
        xdzero       = src.xdzero;
        locsigma     = src.locsigma;
        loclr        = src.loclr;
        loclrb       = src.loclrb;
        loclrc       = src.loclrc;
        loclrd       = src.loclrd;
        xCweight     = src.xCweight;
        xCweightfuzz = src.xCweightfuzz;
        xepsweight   = src.xepsweight;
        K2mat        = src.K2mat;

        xmuprior    = src.xmuprior;
        xmuprior_gt = src.xmuprior_gt;
        xmuprior_ml = src.xmuprior_ml;

        fixpvects();
    }

    else
    {
        // semi-copy version

        allxdatagent.resize((src.allxdatagent).size());
        traininfo.resize((src.traininfo).size());
        traintang.resize((src.traintang).size());
        alltraintarg   = src.alltraintarg;
        alltraintargp  = src.alltraintargp;
        alltraintargR  = src.alltraintargR;
        alltraintargA  = src.alltraintargA;
        alltraintargV  = src.alltraintargV;
        alltraintargpR = src.alltraintargpR;
        alltraintargpA = src.alltraintargpA;
        alltraintargpV = src.alltraintargpV;

        xd           = src.xd;
        xdzero       = src.xdzero;
        locsigma     = src.locsigma;
        loclr        = src.loclr;
        loclrb       = src.loclrb;
        loclrc       = src.loclrc;
        loclrd       = src.loclrd;
        xCweight     = src.xCweight;
        xCweightfuzz = src.xCweightfuzz;
        xepsweight   = src.xepsweight;

        fixpvects();
    }

    indexKey      = src.indexKey;
    indexKeyCount = src.indexKeyCount;
    typeKey       = src.typeKey;
    typeKeyBreak  = src.typeKeyBreak;

    isIndPrune      = src.isIndPrune;
    xassumedconsist = src.xassumedconsist;
    xconsist        = src.xconsist;

    // xpreallocsize unchanged!
}

#endif

