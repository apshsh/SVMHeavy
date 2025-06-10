
//
// LSV base class indirection
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _lsv_generic_deref_h
#define _lsv_generic_deref_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "lsv_generic.hpp"


class LSV_Generic_Deref;

// Swap and zeroing (restarting) functions

inline void qswap(LSV_Generic_Deref &a, LSV_Generic_Deref &b);
inline LSV_Generic_Deref &setzero(LSV_Generic_Deref &a);






class LSV_Generic_Deref : public LSV_Generic
{
public:

    virtual       SVM_Generic &getQQQ     (void)       { return *this; }
    virtual const SVM_Generic &getQQQconst(void) const { return *this; }

    virtual       ML_Base &getQ     (void)       { return static_cast<      ML_Base &>(getQQQ());      }
    virtual const ML_Base &getQconst(void) const { return static_cast<const ML_Base &>(getQQQconst()); }

    // Constructors, destructors, assignment etc..

    LSV_Generic_Deref() : LSV_Generic() { return; }
    LSV_Generic_Deref(const LSV_Generic_Deref &src) : LSV_Generic() { assign(src,0); return; }
    LSV_Generic_Deref(const LSV_Generic_Deref &src, const ML_Base *srcx) : LSV_Generic() { setaltx(srcx); assign(src,-1); return; }
    LSV_Generic_Deref &operator=(const LSV_Generic_Deref &src) { assign(src); return *this; }
    virtual ~LSV_Generic_Deref() { return; }

    virtual int  prealloc    (int expectedN)       override { return getQ().prealloc(expectedN);  }
    virtual int  preallocsize(void)          const override { return getQconst().preallocsize();  }
    virtual void setmemsize  (int memsize)         override { getQ().setmemsize(memsize); return; }

    virtual void assign       (const ML_Base &src, int onlySemiCopy = 0) override { getQ().assign(src,onlySemiCopy); return; }
    virtual void semicopy     (const ML_Base &src)                       override { getQ().semicopy(src);            return; }
    virtual void qswapinternal(      ML_Base &b)                         override { getQ().qswapinternal(b);         return; }

    virtual int getparam (int ind, gentype         &val, const gentype         &xa, int ia, const gentype         &xb, int ib, charptr &desc) const override { return getQconst().getparam( ind,val,xa,ia,xb,ib,desc); }
    virtual int egetparam(int ind, Vector<gentype> &val, const Vector<gentype> &xa, int ia, const Vector<gentype> &xb, int ib               ) const override { return getQconst().egetparam(ind,val,xa,ia,xb,ib     ); }

    virtual std::ostream &printstream(std::ostream &output, int dep) const override { return getQconst().printstream(output,dep); }
    virtual std::istream &inputstream(std::istream &input          )       override { return getQ().inputstream(input);           }

    virtual       ML_Base &getML     (void)       override { return static_cast<      ML_Base &>(getLSV());      }
    virtual const ML_Base &getMLconst(void) const override { return static_cast<const ML_Base &>(getLSVconst()); }

    // Generate RKHS vector form of ML (if possible).

    virtual RKHSVector      &getvecforma(RKHSVector      &res) const override { return getQconst().getvecforma(res); }
    virtual Vector<gentype> &getvecformb(Vector<gentype> &res) const override { return getQconst().getvecformb(res); }
    virtual gentype         &getvecformc(gentype         &res) const override { return getQconst().getvecformc(res); }

    // Information functions (training data):

    virtual int  N       (void)  const override { return getQconst().N();        }
    virtual int  NNC     (int d) const override { return getQconst().NNC(d);     }
    virtual int  type    (void)  const override { return getQconst().type();     }
    virtual int  subtype (void)  const override { return getQconst().subtype();  }
    virtual char gOutType(void)  const override { return getQconst().gOutType(); }
    virtual char hOutType(void)  const override { return getQconst().hOutType(); }
    virtual char targType(void)  const override { return getQconst().targType(); }

    virtual int tspaceDim   (void)       const override { return getQconst().tspaceDim();    }
    virtual int xspaceDim   (int u = -1) const override { return getQconst().xspaceDim(u);   }
    virtual int fspaceDim   (void)       const override { return getQconst().fspaceDim();    }
    virtual int tspaceSparse(void)       const override { return getQconst().tspaceSparse(); }
    virtual int xspaceSparse(void)       const override { return getQconst().xspaceSparse(); }
    virtual int numClasses  (void)       const override { return getQconst().numClasses();   }
    virtual int order       (void)       const override { return getQconst().order();        }

    virtual int isTrained(void) const override { return getQconst().isTrained(); }
    virtual int isSolGlob(void) const override { return getQconst().isSolGlob(); }
    virtual int isMutable(void) const override { return getQconst().isMutable(); }
    virtual int isPool   (void) const override { return getQconst().isPool   (); }

    virtual double calcDist(const gentype &ha, const gentype &hb, int ia = -1, int db = 2) const override { return getQconst().calcDist(ha,hb,ia,db); }

    virtual double calcDistInt(int    ha, int    hb, int ia = -1, int db = 2) const override { return getQconst().calcDistInt(ha,hb,ia,db); }
    virtual double calcDistDbl(double ha, double hb, int ia = -1, int db = 2) const override { return getQconst().calcDistDbl(ha,hb,ia,db); }

    virtual int isUnderlyingScalar(void) const override { return getQconst().isUnderlyingScalar(); }
    virtual int isUnderlyingVector(void) const override { return getQconst().isUnderlyingVector(); }
    virtual int isUnderlyingAnions(void) const override { return getQconst().isUnderlyingAnions(); }

    virtual const Vector<int> &ClassLabels(void)             const override { return getQconst().ClassLabels();        }
    virtual int   getInternalClass        (const gentype &y) const override { return getQconst().getInternalClass(y);  }
    virtual int   numInternalClasses      (void)             const override { return getQconst().numInternalClasses(); }
    virtual int   isenabled               (int i)            const override { return getQconst().isenabled(i);         }
    virtual int   isVarDefined            (void)             const override { return getQconst().isVarDefined();       }

    virtual const int *ClassLabelsInt     (void)  const override { return getQconst().ClassLabelsInt();       }
    virtual       int  getInternalClassInt(int y) const override { return getQconst().getInternalClassInt(y); }

    virtual double C        (void)  const override { return getQconst().C();         }
    virtual double sigma    (void)  const override { return getQconst().sigma();     }
    virtual double sigma_cut(void)  const override { return getQconst().sigma_cut(); }
    virtual double eps      (void)  const override { return getQconst().eps();       }
    virtual double Cclass   (int d) const override { return getQconst().Cclass(d);   }
    virtual double epsclass (int d) const override { return getQconst().epsclass(d); }

    virtual       int      mpri  (void) const override { return getQconst().mpri();   }
    virtual const gentype &prival(void) const override { return getQconst().prival(); }
    virtual const ML_Base *priml (void) const override { return getQconst().priml();  }

    virtual void calcprior   (gentype &res, const SparseVector<gentype> &x, const vecInfo *xinf = nullptr) const override { return getQconst().calcprior(res,x,xinf); }
    virtual void calcallprior(void)                                                                              override { return getQ().calcallprior();             }

    virtual int    memsize     (void) const override { return getQconst().memsize();      }
    virtual double zerotol     (void) const override { return getQconst().zerotol();      }
    virtual double Opttol      (void) const override { return getQconst().Opttol();       }
    virtual double Opttolb     (void) const override { return getQconst().Opttolb();      }
    virtual double Opttolc     (void) const override { return getQconst().Opttolc();      }
    virtual double Opttold     (void) const override { return getQconst().Opttold();      }
    virtual double lr          (void) const override { return getQconst().lr();           }
    virtual double lrb         (void) const override { return getQconst().lrb();          }
    virtual double lrc         (void) const override { return getQconst().lrc();          }
    virtual double lrd         (void) const override { return getQconst().lrd();          }
    virtual int    maxitcnt    (void) const override { return getQconst().maxitcnt();     }
    virtual double maxtraintime(void) const override { return getQconst().maxtraintime(); }
    virtual double traintimeend(void) const override { return getQconst().traintimeend(); }

    virtual int    maxitermvrank(void) const override { return getQconst().maxitermvrank(); }
    virtual double lrmvrank     (void) const override { return getQconst().lrmvrank();      }
    virtual double ztmvrank     (void) const override { return getQconst().ztmvrank();      }

    virtual double betarank(void) const override { return getQconst().betarank(); }

    virtual double sparlvl(void) const override { return getQconst().sparlvl(); }

    virtual const Vector<SparseVector<gentype> > &x          (void) const override { return getQconst().x();           }
    virtual const Vector<gentype>                &y          (void) const override { return getQconst().y();           }
    virtual const Vector<double>                 &yR         (void) const override { return getQconst().yR();          }
    virtual const Vector<d_anion>                &yA         (void) const override { return getQconst().yA();          }
    virtual const Vector<Vector<double> >        &yV         (void) const override { return getQconst().yV();          }
    virtual const Vector<gentype>                &yp         (void) const override { return getQconst().yp();          }
    virtual const Vector<double>                 &ypR        (void) const override { return getQconst().ypR();         }
    virtual const Vector<d_anion>                &ypA        (void) const override { return getQconst().ypA();         }
    virtual const Vector<Vector<double> >        &ypV        (void) const override { return getQconst().ypV();         }
    virtual const Vector<vecInfo>                &xinfo      (void) const override { return getQconst().xinfo();       }
    virtual const Vector<int>                    &xtang      (void) const override { return getQconst().xtang();       }
    virtual const Vector<int>                    &d          (void) const override { return getQconst().d();           }
    virtual const Vector<double>                 &Cweight    (void) const override { return getQconst().Cweight();     }
    virtual const Vector<double>                 &Cweightfuzz(void) const override { return getQconst().Cweightfuzz(); }
    virtual const Vector<double>                 &sigmaweight(void) const override { return getQconst().sigmaweight(); }
    virtual const Vector<double>                 &epsweight  (void) const override { return getQconst().epsweight();   }
    virtual const Vector<gentype>                &alphaVal   (void) const override { return getQconst().alphaVal();    }
    virtual const Vector<int>                    &alphaState (void) const override { return getQconst().alphaState();  }

    virtual const SparseVector<gentype> &x       (int i)              const override { return getQconst().x(i);         }
    virtual const SparseVector<gentype> &x       (int i, int altMLid) const override { return getQconst().x(i,altMLid); }
    virtual const gentype               &y       (int i)              const override { return getQconst().y(i);         }
    virtual       double                 yR      (int i)              const override { return getQconst().yR(i);        }
    virtual const d_anion               &yA      (int i)              const override { return getQconst().yA(i);        }
    virtual const Vector<double>        &yV      (int i)              const override { return getQconst().yV(i);        }
    virtual const gentype               &yp      (int i)              const override { return getQconst().yp(i);        }
    virtual       double                 ypR     (int i)              const override { return getQconst().ypR(i);       }
    virtual const d_anion               &ypA     (int i)              const override { return getQconst().ypA(i);       }
    virtual const Vector<double>        &ypV     (int i)              const override { return getQconst().ypV(i);       }
    virtual const vecInfo               &xinfo   (int i)              const override { return getQconst().xinfo(i);     }
    virtual       int                    xtang   (int i)              const override { return getQconst().xtang(i);     }
    virtual       double                 alphaVal(int i)              const override { return getQconst().alphaVal(i);  }

    virtual int xisrank      (int i)                               const override { return getQconst().xisrank(i);                 }
    virtual int xisgrad      (int i)                               const override { return getQconst().xisgrad(i);                 }
    virtual int xisrankorgrad(int i)                               const override { return getQconst().xisrankorgrad(i);           }
    virtual int xisclass     (int i, int defaultclass, int q = -1) const override { return getQconst().xisclass(i,defaultclass,q); }

    virtual int RFFordata(int i) const { return getQconst().RFFordata(i); }

    virtual void npCweight    (double **res, int *dim) const override { getQconst().npCweight    (res,dim); return; }
    virtual void npCweightfuzz(double **res, int *dim) const override { getQconst().npCweightfuzz(res,dim); return; }
    virtual void npsigmaweight(double **res, int *dim) const override { getQconst().npsigmaweight(res,dim); return; }
    virtual void npepsweight  (double **res, int *dim) const override { getQconst().npepsweight  (res,dim); return; }

    virtual int isClassifier(void) const override { return getQconst().isClassifier(); }
    virtual int isRegression(void) const override { return getQconst().isRegression(); }
    virtual int isPlanarType(void) const override { return getQconst().isPlanarType(); }

    // Random features stuff:

    virtual int NRff   (void) const override { return getQconst().NRff   (); }
    virtual int NRffRep(void) const override { return getQconst().NRffRep(); }
    virtual int ReOnly (void) const override { return getQconst().ReOnly (); }
    virtual int inAdam (void) const override { return getQconst().inAdam (); }
    virtual int outGrad(void) const override { return getQconst().outGrad(); }

    // Version numbers

    virtual int MLid    (void)                             const override { return getQconst().MLid();                }
    virtual int setMLid (int nv)                                 override { return getQ().setMLid(nv);                }
    virtual int getaltML(kernPrecursor *&res, int altMLid) const override { return getQconst().getaltML(res,altMLid); }

    virtual int xvernum(void) const override { return getQconst().xvernum(); }
    virtual int gvernum(void) const override { return getQconst().gvernum(); }

    virtual int xvernum(int altMLid) const override { return getQconst().xvernum(altMLid); }
    virtual int gvernum(int altMLid) const override { return getQconst().gvernum(altMLid); }

    virtual int incxvernum(void) override { return getQ().incxvernum(); }
    virtual int incgvernum(void) override { return getQ().incgvernum(); }

    // Kernel Modification

    virtual const MercerKernel &getKernel       (void) const override { return getQconst().getKernel();   }
    virtual       MercerKernel &getKernel_unsafe(void)       override { return getQ().getKernel_unsafe(); }
    virtual       void          prepareKernel   (void)       override {        getQ().prepareKernel();    }

    virtual double tuneKernel(int method, double xwidth, int tuneK = 1, int tuneP = 0, const tkBounds *tunebounds = nullptr) override { return getQ().tuneKernel(method,xwidth,tuneK,tuneP,tunebounds); }

    virtual int resetKernel(                             int modind = 1, int onlyChangeRowI = -1, int updateInfo = 1) override { return getQ().resetKernel(modind,onlyChangeRowI,updateInfo); }
    virtual int setKernel  (const MercerKernel &xkernel, int modind = 1, int onlyChangeRowI = -1                    ) override { return getQ().setKernel(xkernel,modind,onlyChangeRowI);      }

    virtual int isKreal  (void) const override { return getQconst().isKreal();   }
    virtual int isKunreal(void) const override { return getQconst().isKunreal(); }

    virtual int setKreal  (void) override { return getQ().setKreal();   }
    virtual int setKunreal(void) override { return getQ().setKunreal(); }

    virtual double k2diag(int ia) const override { return getQconst().k2diag(ia); }

    virtual void fillCache(int Ns = 0, int Ne = -1) override { getQ().fillCache(Ns,Ne); return; }

    virtual void K2bypass(const Matrix<gentype> &nv) override { getQ().K2bypass(nv); return; }

    virtual gentype &Keqn(gentype &res,                           int resmode = 1) const override { return getQconst().Keqn(res,     resmode); }
    virtual gentype &Keqn(gentype &res, const MercerKernel &altK, int resmode = 1) const override { return getQconst().Keqn(res,altK,resmode); }

    virtual gentype &K1(gentype &res, const SparseVector<gentype> &xa, const vecInfo *xainf = nullptr) const override { return getQconst().K1(res,xa,xainf); }
    virtual gentype &K2(gentype &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo *xainf = nullptr, const vecInfo *xbinf = nullptr) const override { return getQconst().K2(res,xa,xb,xainf,xbinf); }
    virtual gentype &K3(gentype &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo *xainf = nullptr, const vecInfo *xbinf = nullptr, const vecInfo *xcinf = nullptr) const override { return getQconst().K3(res,xa,xb,xc,xainf,xbinf,xcinf); }
    virtual gentype &K4(gentype &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo *xainf = nullptr, const vecInfo *xbinf = nullptr, const vecInfo *xcinf = nullptr, const vecInfo *xdinf = nullptr) const override { return getQconst().K4(res,xa,xb,xc,xd,xainf,xbinf,xcinf,xdinf); }
    virtual gentype &Km(gentype &res, const Vector<SparseVector<gentype> > &xx) const override { return getQconst().Km(res,xx); }

    virtual double  K2ip(const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo *xainf = nullptr, const vecInfo *xbinf = nullptr) const override { return getQconst().K2ip(xa,xb,xainf,xbinf); }
    virtual double distK(const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo *xainf = nullptr, const vecInfo *xbinf = nullptr) const override { return getQconst().distK(xa,xb,xainf,xbinf); }

    virtual Vector<gentype> &phi2(Vector<gentype> &res, const SparseVector<gentype> &xa, const vecInfo *xainf = nullptr) const override { return getQconst().phi2(res,xa,xainf); }
    virtual Vector<gentype> &phi2(Vector<gentype> &res, int ia, const SparseVector<gentype> *xa = nullptr, const vecInfo *xainf = nullptr) const override { return getQconst().phi2(res,ia,xa,xainf); }

    virtual Vector<double> &phi2(Vector<double> &res, const SparseVector<gentype> &xa, const vecInfo *xainf = nullptr) const override { return getQconst().phi2(res,xa,xainf); }
    virtual Vector<double> &phi2(Vector<double> &res, int ia, const SparseVector<gentype> *xa = nullptr, const vecInfo *xainf = nullptr) const override { return getQconst().phi2(res,ia,xa,xainf); }

    virtual double K0ip(                                       const gentype **pxyprod = nullptr) const override { return getQconst().K0ip(pxyprod); }
    virtual double K1ip(       int ia,                         const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const vecInfo *xainfo = nullptr) const override { return  getQconst().K1ip(ia,pxyprod,xa,xainfo); }
    virtual double K2ip(       int ia, int ib,                 const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr) const override { return  getQconst().K2ip(ia,ib,pxyprod,xa,xb,xainfo,xbinfo); }
    virtual double K3ip(       int ia, int ib, int ic,         const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const SparseVector<gentype> *xc = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, const vecInfo *xcinfo = nullptr) const override { return getQconst().K3ip(ia,ib,ic,pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo); }
    virtual double K4ip(       int ia, int ib, int ic, int id, const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const SparseVector<gentype> *xc = nullptr, const SparseVector<gentype> *xd = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, const vecInfo *xcinfo = nullptr, const vecInfo *xdinfo = nullptr) const override { return getQconst().K4ip(ia,ib,ic,id,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo); }
    virtual double Kmip(int m, Vector<int> &i, const gentype **pxyprod = nullptr, Vector<const SparseVector<gentype> *> *xx = nullptr, Vector<const vecInfo *> *xzinfo = nullptr) const override { return getQconst().Kmip(m,i,pxyprod,xx,xzinfo); }

    virtual double K0ip(                                       double bias, const gentype **pxyprod = nullptr) const override { return getQconst().K0ip(bias,pxyprod); }
    virtual double K1ip(       int ia,                         double bias, const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const vecInfo *xainfo = nullptr) const override { return getQconst().K1ip(ia,bias,pxyprod,xa,xainfo); }
    virtual double K2ip(       int ia, int ib,                 double bias, const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr) const override { return getQconst().K2ip(ia,ib,bias,pxyprod,xa,xb,xainfo,xbinfo); }
    virtual double K3ip(       int ia, int ib, int ic,         double bias, const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const SparseVector<gentype> *xc = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, const vecInfo *xcinfo = nullptr) const override { return getQconst().K3ip(ia,ib,ic,bias,pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo); }
    virtual double K4ip(       int ia, int ib, int ic, int id, double bias, const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const SparseVector<gentype> *xc = nullptr, const SparseVector<gentype> *xd = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, const vecInfo *xcinfo = nullptr, const vecInfo *xdinfo = nullptr) const override { return getQconst().K4ip(ia,ib,ic,id,bias,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo); }
    virtual double Kmip(int m, Vector<int> &i, double bias, const gentype **pxyprod = nullptr, Vector<const SparseVector<gentype> *> *xx = nullptr, Vector<const vecInfo *> *xzinfo = nullptr) const override { return getQconst().Kmip(m,i,bias,pxyprod,xx,xzinfo); }

    virtual gentype        &K0(              gentype        &res                          , const gentype **pxyprod = nullptr, int resmode = 0) const override { return getQconst().K0(         res     ,pxyprod,resmode); }
    virtual gentype        &K0(              gentype        &res, const gentype &bias     , const gentype **pxyprod = nullptr, int resmode = 0) const override { return getQconst().K0(         res,bias,pxyprod,resmode); }
    virtual gentype        &K0(              gentype        &res, const MercerKernel &altK, const gentype **pxyprod = nullptr, int resmode = 0) const override { return getQconst().K0(         res,altK,pxyprod,resmode); }
    virtual double          K0(                                                             const gentype **pxyprod = nullptr, int resmode = 0) const override { return getQconst().K0(                  pxyprod,resmode); }
    virtual Matrix<double> &K0(int spaceDim, Matrix<double> &res                          , const gentype **pxyprod = nullptr, int resmode = 0) const override { return getQconst().K0(spaceDim,res     ,pxyprod,resmode); }
    virtual d_anion        &K0(int order,    d_anion        &res                          , const gentype **pxyprod = nullptr, int resmode = 0) const override { return getQconst().K0(order   ,res     ,pxyprod,resmode); }

    virtual gentype        &K1(              gentype        &res, int ia                          , const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const vecInfo *xainfo = nullptr, int resmode = 0) const override { return getQconst().K1(         res,ia     ,pxyprod,xa,xainfo,resmode); }
    virtual gentype        &K1(              gentype        &res, int ia, const gentype &bias     , const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const vecInfo *xainfo = nullptr, int resmode = 0) const override { return getQconst().K1(         res,ia,bias,pxyprod,xa,xainfo,resmode); }
    virtual gentype        &K1(              gentype        &res, int ia, const MercerKernel &altK, const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const vecInfo *xainfo = nullptr, int resmode = 0) const override { return getQconst().K1(         res,ia,altK,pxyprod,xa,xainfo,resmode); }
    virtual double          K1(                                   int ia                          , const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const vecInfo *xainfo = nullptr, int resmode = 0) const override { return getQconst().K1(             ia     ,pxyprod,xa,xainfo,resmode); }
    virtual Matrix<double> &K1(int spaceDim, Matrix<double> &res, int ia                          , const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const vecInfo *xainfo = nullptr, int resmode = 0) const override { return getQconst().K1(spaceDim,res,ia     ,pxyprod,xa,xainfo,resmode); }
    virtual d_anion        &K1(int order,    d_anion        &res, int ia                          , const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const vecInfo *xainfo = nullptr, int resmode = 0) const override { return getQconst().K1(order   ,res,ia     ,pxyprod,xa,xainfo,resmode); }

    virtual gentype        &K2(              gentype        &res, int ia, int ib                          , const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, int resmode = 0) const override { return getQconst().K2(         res,ia,ib     ,pxyprod,xa,xb,xainfo,xbinfo,resmode); }
    virtual gentype        &K2(              gentype        &res, int ia, int ib, const gentype &bias     , const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, int resmode = 0) const override { return getQconst().K2(         res,ia,ib,bias,pxyprod,xa,xb,xainfo,xbinfo,resmode); }
    virtual gentype        &K2(              gentype        &res, int ia, int ib, const MercerKernel &altK, const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, int resmode = 0) const override { return getQconst().K2(         res,ia,ib,altK,pxyprod,xa,xb,xainfo,xbinfo,resmode); }
    virtual double          K2(                                   int ia, int ib                          , const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, int resmode = 0) const override { return getQconst().K2(             ia,ib     ,pxyprod,xa,xb,xainfo,xbinfo,resmode); }
    virtual Matrix<double> &K2(int spaceDim, Matrix<double> &res, int ia, int ib                          , const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, int resmode = 0) const override { return getQconst().K2(spaceDim,res,ia,ib     ,pxyprod,xa,xb,xainfo,xbinfo,resmode); }
    virtual d_anion        &K2(int order,    d_anion        &res, int ia, int ib                          , const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, int resmode = 0) const override { return getQconst().K2(order,   res,ia,ib     ,pxyprod,xa,xb,xainfo,xbinfo,resmode); }

    virtual gentype        &K2x2(              gentype        &res, int i, int ia, int ib                          , const SparseVector<gentype> *x = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const vecInfo *xinfo = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, int resmode = 0) const override { return getQconst().K2x2(         res,i,ia,ib,     x,xa,xb,xinfo,xainfo,xbinfo,resmode); }
    virtual gentype        &K2x2(              gentype        &res, int i, int ia, int ib, const gentype &bias     , const SparseVector<gentype> *x = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const vecInfo *xinfo = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, int resmode = 0) const override { return getQconst().K2x2(         res,i,ia,ib,bias,x,xa,xb,xinfo,xainfo,xbinfo,resmode); }
    virtual gentype        &K2x2(              gentype        &res, int i, int ia, int ib, const MercerKernel &altK, const SparseVector<gentype> *x = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const vecInfo *xinfo = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, int resmode = 0) const override { return getQconst().K2x2(         res,i,ia,ib,altK,x,xa,xb,xinfo,xainfo,xbinfo,resmode); }
    virtual double          K2x2(                                   int i, int ia, int ib                          , const SparseVector<gentype> *x = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const vecInfo *xinfo = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, int resmode = 0) const override { return getQconst().K2x2(             i,ia,ib,     x,xa,xb,xinfo,xainfo,xbinfo,resmode); }
    virtual Matrix<double> &K2x2(int spaceDim, Matrix<double> &res, int i, int ia, int ib                          , const SparseVector<gentype> *x = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const vecInfo *xinfo = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, int resmode = 0) const override { return getQconst().K2x2(spaceDim,res,i,ia,ib,     x,xa,xb,xinfo,xainfo,xbinfo,resmode); }
    virtual d_anion        &K2x2(int order,    d_anion        &res, int i, int ia, int ib                          , const SparseVector<gentype> *x = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const vecInfo *xinfo = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, int resmode = 0) const override { return getQconst().K2x2(order,   res,i,ia,ib,     x,xa,xb,xinfo,xainfo,xbinfo,resmode); }

    virtual gentype        &K3(              gentype        &res, int ia, int ib, int ic                          , const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const SparseVector<gentype> *xc = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, const vecInfo *xcinfo = nullptr, int resmode = 0) const override { return getQconst().K3(         res,ia,ib,ic     ,pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo,resmode); }
    virtual gentype        &K3(              gentype        &res, int ia, int ib, int ic, const gentype &bias     , const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const SparseVector<gentype> *xc = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, const vecInfo *xcinfo = nullptr, int resmode = 0) const override { return getQconst().K3(         res,ia,ib,ic,bias,pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo,resmode); }
    virtual gentype        &K3(              gentype        &res, int ia, int ib, int ic, const MercerKernel &altK, const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const SparseVector<gentype> *xc = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, const vecInfo *xcinfo = nullptr, int resmode = 0) const override { return getQconst().K3(         res,ia,ib,ic,altK,pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo,resmode); }
    virtual double          K3(                                   int ia, int ib, int ic                          , const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const SparseVector<gentype> *xc = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, const vecInfo *xcinfo = nullptr, int resmode = 0) const override { return getQconst().K3(             ia,ib,ic     ,pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo,resmode); }
    virtual Matrix<double> &K3(int spaceDim, Matrix<double> &res, int ia, int ib, int ic                          , const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const SparseVector<gentype> *xc = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, const vecInfo *xcinfo = nullptr, int resmode = 0) const override { return getQconst().K3(spaceDim,res,ia,ib,ic     ,pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo,resmode); }
    virtual d_anion        &K3(int order,    d_anion        &res, int ia, int ib, int ic                          , const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const SparseVector<gentype> *xc = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, const vecInfo *xcinfo = nullptr, int resmode = 0) const override { return getQconst().K3(order   ,res,ia,ib,ic     ,pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo,resmode); }

    virtual gentype        &K4(              gentype        &res, int ia, int ib, int ic, int id                          , const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const SparseVector<gentype> *xc = nullptr, const SparseVector<gentype> *xd = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, const vecInfo *xcinfo = nullptr, const vecInfo *xdinfo = nullptr, int resmode = 0) const override { return getQconst().K4(         res,ia,ib,ic,id     ,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,resmode); }
    virtual gentype        &K4(              gentype        &res, int ia, int ib, int ic, int id, const gentype &bias     , const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const SparseVector<gentype> *xc = nullptr, const SparseVector<gentype> *xd = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, const vecInfo *xcinfo = nullptr, const vecInfo *xdinfo = nullptr, int resmode = 0) const override { return getQconst().K4(         res,ia,ib,ic,id,bias,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,resmode); }
    virtual gentype        &K4(              gentype        &res, int ia, int ib, int ic, int id, const MercerKernel &altK, const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const SparseVector<gentype> *xc = nullptr, const SparseVector<gentype> *xd = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, const vecInfo *xcinfo = nullptr, const vecInfo *xdinfo = nullptr, int resmode = 0) const override { return getQconst().K4(         res,ia,ib,ic,id,altK,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,resmode); }
    virtual double          K4(                                   int ia, int ib, int ic, int id                          , const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const SparseVector<gentype> *xc = nullptr, const SparseVector<gentype> *xd = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, const vecInfo *xcinfo = nullptr, const vecInfo *xdinfo = nullptr, int resmode = 0) const override { return getQconst().K4(             ia,ib,ic,id     ,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,resmode); }
    virtual Matrix<double> &K4(int spaceDim, Matrix<double> &res, int ia, int ib, int ic, int id                          , const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const SparseVector<gentype> *xc = nullptr, const SparseVector<gentype> *xd = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, const vecInfo *xcinfo = nullptr, const vecInfo *xdinfo = nullptr, int resmode = 0) const override { return getQconst().K4(spaceDim,res,ia,ib,ic,id     ,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,resmode); }
    virtual d_anion        &K4(int order,    d_anion        &res, int ia, int ib, int ic, int id                          , const gentype **pxyprod = nullptr, const SparseVector<gentype> *xa = nullptr, const SparseVector<gentype> *xb = nullptr, const SparseVector<gentype> *xc = nullptr, const SparseVector<gentype> *xd = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *xbinfo = nullptr, const vecInfo *xcinfo = nullptr, const vecInfo *xdinfo = nullptr, int resmode = 0) const override { return getQconst().K4(order   ,res,ia,ib,ic,id     ,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,resmode); }

    virtual gentype        &Km(int m              , gentype        &res, Vector<int> &i                          , const gentype **pxyprod = nullptr, Vector<const SparseVector<gentype> *> *xx = nullptr, Vector<const vecInfo *> *xainfo = nullptr, int resmode = 0) const override { return getQconst().Km(m         ,res,i,pxyprod     ,xx,xainfo,resmode); }
    virtual gentype        &Km(int m              , gentype        &res, Vector<int> &i, const gentype &bias     , const gentype **pxyprod = nullptr, Vector<const SparseVector<gentype> *> *xx = nullptr, Vector<const vecInfo *> *xainfo = nullptr, int resmode = 0) const override { return getQconst().Km(m         ,res,i,bias,pxyprod,xx,xainfo,resmode); }
    virtual gentype        &Km(int m              , gentype        &res, Vector<int> &i, const MercerKernel &altK, const gentype **pxyprod = nullptr, Vector<const SparseVector<gentype> *> *xx = nullptr, Vector<const vecInfo *> *xainfo = nullptr, int resmode = 0) const override { return getQconst().Km(m         ,res,i,altK,pxyprod,xx,xainfo,resmode); }
    virtual double          Km(int m              ,                      Vector<int> &i                          , const gentype **pxyprod = nullptr, Vector<const SparseVector<gentype> *> *xx = nullptr, Vector<const vecInfo *> *xainfo = nullptr, int resmode = 0) const override { return getQconst().Km(m             ,i,pxyprod     ,xx,xainfo,resmode); }
    virtual Matrix<double> &Km(int m, int spaceDim, Matrix<double> &res, Vector<int> &i                          , const gentype **pxyprod = nullptr, Vector<const SparseVector<gentype> *> *xx = nullptr, Vector<const vecInfo *> *xainfo = nullptr, int resmode = 0) const override { return getQconst().Km(m,spaceDim,res,i,pxyprod     ,xx,xainfo,resmode); }
    virtual d_anion        &Km(int m, int order   , d_anion        &res, Vector<int> &i                          , const gentype **pxyprod = nullptr, Vector<const SparseVector<gentype> *> *xx = nullptr, Vector<const vecInfo *> *xainfo = nullptr, int resmode = 0) const override { return getQconst().Km(m,order   ,res,i,pxyprod     ,xx,xainfo,resmode); }

    virtual void dK(gentype &xygrad, gentype &xnormgrad, int i, int j, const gentype **pxyprod = nullptr, const SparseVector<gentype> *xx = nullptr, const SparseVector<gentype> *yy = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *yyinfo = nullptr, int deepDeriv = 0) const override { getQconst().dK(xygrad,xnormgrad,i,j,     pxyprod,xx,yy,xainfo,yyinfo,deepDeriv); return; }
    virtual void dK(double  &xygrad, double  &xnormgrad, int i, int j, const gentype **pxyprod = nullptr, const SparseVector<gentype> *xx = nullptr, const SparseVector<gentype> *yy = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *yyinfo = nullptr, int deepDeriv = 0) const override { getQconst().dK(xygrad,xnormgrad,i,j,     pxyprod,xx,yy,xainfo,yyinfo,deepDeriv); return; }

    virtual void d2K(gentype &xygrad, gentype &xnormgrad, gentype &xyxygrad, gentype &xyxnormgrad, gentype &xyynormgrad, gentype &xnormxnormgrad, gentype &xnormynormgrad, gentype &ynormynormgrad, int &minmaxind, int i, int j, const gentype **pxyprod = nullptr, const SparseVector<gentype> *xx = nullptr, const SparseVector<gentype> *yy = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *yyinfo = nullptr) const override { getQconst().d2K(xygrad,xnormgrad,xyxygrad,xyxnormgrad,xyynormgrad,xnormxnormgrad,xnormynormgrad,ynormynormgrad,minmaxind,i,j,pxyprod,xx,yy,xainfo,yyinfo); return; }
    virtual void d2K(double  &xygrad, double  &xnormgrad, double  &xyxygrad, double  &xyxnormgrad, double  &xyynormgrad, double  &xnormxnormgrad, double  &xnormynormgrad, double  &ynormynormgrad, int &minmaxind, int i, int j, const gentype **pxyprod = nullptr, const SparseVector<gentype> *xx = nullptr, const SparseVector<gentype> *yy = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *yyinfo = nullptr) const override { getQconst().d2K(xygrad,xnormgrad,xyxygrad,xyxnormgrad,xyynormgrad,xnormxnormgrad,xnormynormgrad,ynormynormgrad,minmaxind,i,j,pxyprod,xx,yy,xainfo,yyinfo); return; }

    virtual void dK2delx(gentype &xscaleres, gentype &yscaleres, int &minmaxind, int i, int j, const gentype **pxyprod = nullptr, const SparseVector<gentype> *xx = nullptr, const SparseVector<gentype> *yy = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *yyinfo = nullptr) const override { getQconst().dK2delx(xscaleres,yscaleres,minmaxind,i,j,     pxyprod,xx,yy,xainfo,yyinfo); return; }
    virtual void dK2delx(double  &xscaleres, double  &yscaleres, int &minmaxind, int i, int j, const gentype **pxyprod = nullptr, const SparseVector<gentype> *xx = nullptr, const SparseVector<gentype> *yy = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *yyinfo = nullptr) const override { getQconst().dK2delx(xscaleres,yscaleres,minmaxind,i,j,     pxyprod,xx,yy,xainfo,yyinfo); return; }

    virtual void d2K2delxdelx(gentype &xxscaleres, gentype &yyscaleres, gentype &xyscaleres, gentype &yxscaleres, gentype &constres, int &minmaxind, int i, int j, const gentype **pxyprod = nullptr, const SparseVector<gentype> *xx = nullptr, const SparseVector<gentype> *yy = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *yyinfo = nullptr) const override { getQconst().d2K2delxdelx(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,minmaxind,i,j,pxyprod,xx,yy,xainfo,yyinfo); return; }
    virtual void d2K2delxdely(gentype &xxscaleres, gentype &yyscaleres, gentype &xyscaleres, gentype &yxscaleres, gentype &constres, int &minmaxind, int i, int j, const gentype **pxyprod = nullptr, const SparseVector<gentype> *xx = nullptr, const SparseVector<gentype> *yy = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *yyinfo = nullptr) const override { getQconst().d2K2delxdely(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,minmaxind,i,j,pxyprod,xx,yy,xainfo,yyinfo); return; }

    virtual void d2K2delxdelx(double  &xxscaleres, double  &yyscaleres, double  &xyscaleres, double  &yxscaleres, double  &constres, int &minmaxind, int i, int j, const gentype **pxyprod = nullptr, const SparseVector<gentype> *xx = nullptr, const SparseVector<gentype> *yy = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *yyinfo = nullptr) const override { getQconst().d2K2delxdelx(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,minmaxind,i,j,pxyprod,xx,yy,xainfo,yyinfo); return; }
    virtual void d2K2delxdely(double  &xxscaleres, double  &yyscaleres, double  &xyscaleres, double  &yxscaleres, double  &constres, int &minmaxind, int i, int j, const gentype **pxyprod = nullptr, const SparseVector<gentype> *xx = nullptr, const SparseVector<gentype> *yy = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *yyinfo = nullptr) const override { getQconst().d2K2delxdely(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,minmaxind,i,j,pxyprod,xx,yy,xainfo,yyinfo); return; }

    virtual void dnK2del(Vector<gentype> &sc, Vector<Vector<int> > &n, int &minmaxind, const Vector<int> &q, int i, int j, const gentype **pxyprod = nullptr, const SparseVector<gentype> *xx = nullptr, const SparseVector<gentype> *yy = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *yyinfo = nullptr) const override { getQconst().dnK2del(sc,n,minmaxind,q,i,j,pxyprod,xx,yy,xainfo,yyinfo); return; }
    virtual void dnK2del(Vector<double>  &sc, Vector<Vector<int> > &n, int &minmaxind, const Vector<int> &q, int i, int j, const gentype **pxyprod = nullptr, const SparseVector<gentype> *xx = nullptr, const SparseVector<gentype> *yy = nullptr, const vecInfo *xainfo = nullptr, const vecInfo *yyinfo = nullptr) const override { getQconst().dnK2del(sc,n,minmaxind,q,i,j,pxyprod,xx,yy,xainfo,yyinfo); return; }

    virtual double distK(int i, int j) const override { return getQconst().distK(i,j); }

    virtual void densedKdx(double &res, int i, int j) const override { return getQconst().densedKdx(res,i,j); }
    virtual void denseintK(double &res, int i, int j) const override { return getQconst().denseintK(res,i,j); }

    virtual void densedKdx(double &res, int i, int j, double bias) const override { return getQconst().densedKdx(res,i,j,bias); }
    virtual void denseintK(double &res, int i, int j, double bias) const override { return getQconst().denseintK(res,i,j,bias); }

    virtual void ddistKdx(double &xscaleres, double &yscaleres, int &minmaxind, int i, int j) const override { getQconst().ddistKdx(xscaleres,yscaleres,minmaxind,i,j); return; }

    virtual int isKVarianceNZ(void) const override { return getQconst().isKVarianceNZ(); }

    virtual void K0xfer(                                    gentype &res, int &minmaxind, int typeis, const gentype &xyprod, const gentype &yxprod, const gentype &diffis, int xdim, int densetype, int resmode, int mlid) const override { getQconst().K0xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xdim,densetype,resmode,mlid); return; }
    virtual void K1xfer(                                    gentype &res, int &minmaxind, int typeis, const gentype &xyprod, const gentype &yxprod, const gentype &diffis, const SparseVector<gentype> &xa, const vecInfo &xainfo, int ia, int xdim, int densetype, int resmode, int mlid) const override { getQconst().K1xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xainfo,ia,xdim,densetype,resmode,mlid); return; }
    virtual void K2xfer(gentype &dxyprod, gentype &ddiffis, gentype &res, int &minmaxind, int typeis, const gentype &xyprod, const gentype &yxprod, const gentype &diffis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, int ia, int ib, int xdim, int densetype, int resmode, int mlid) const override { getQconst().K2xfer(dxyprod,ddiffis,res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xainfo,xbinfo,ia,ib,xdim,densetype,resmode,mlid); return; }
    virtual void K3xfer(                                    gentype &res, int &minmaxind, int typeis, const gentype &xyprod, const gentype &yxprod, const gentype &diffis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, int ia, int ib, int ic, int xdim, int densetype, int resmode, int mlid) const override { getQconst().K3xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xc,xainfo,xbinfo,xcinfo,ia,ib,ic,xdim,densetype,resmode,mlid); return; }
    virtual void K4xfer(                                    gentype &res, int &minmaxind, int typeis, const gentype &xyprod, const gentype &yxprod, const gentype &diffis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, int ia, int ib, int ic, int id, int xdim, int densetype, int resmode, int mlid) const override { getQconst().K4xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,ia,ib,ic,id,xdim,densetype,resmode,mlid); return; }
    virtual void Kmxfer(                                    gentype &res, int &minmaxind, int typeis, const gentype &xyprod, const gentype &yxprod, const gentype &diffis, Vector<const SparseVector<gentype> *> &x, Vector<const vecInfo *> &xzinfo, Vector<int> &i, int xdim, int m, int densetype, int resmode, int mlid) const override { getQconst().Kmxfer(res,minmaxind,typeis,xyprod,yxprod,diffis,x,xzinfo,i,xdim,m,densetype,resmode,mlid); return; }

    virtual void K0xfer(                                  double &res, int &minmaxind, int typeis, double xyprod, double yxprod, double diffis, int xdim, int densetype, int resmode, int mlid) const override { getQconst().K0xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xdim,densetype,resmode,mlid); return; }
    virtual void K1xfer(                                  double &res, int &minmaxind, int typeis, double xyprod, double yxprod, double diffis, const SparseVector<gentype> &xa, const vecInfo &xainfo, int ia,  int xdim, int densetype, int resmode, int mlid) const override { getQconst().K1xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xainfo,ia,xdim,densetype,resmode,mlid); return; }
    virtual void K2xfer(double &dxyprod, double &ddiffis, double &res, int &minmaxind, int typeis, double xyprod, double yxprod, double diffis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, int ia, int ib, int xdim, int densetype, int resmode, int mlid) const override { getQconst().K2xfer(dxyprod,ddiffis,res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xainfo,xbinfo,ia,ib,xdim,densetype,resmode,mlid); return; }
    virtual void K3xfer(                                  double &res, int &minmaxind, int typeis, double xyprod, double yxprod, double diffis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, int ia, int ib, int ic, int xdim, int densetype, int resmode, int mlid) const override { getQconst().K3xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xc,xainfo,xbinfo,xcinfo,ia,ib,ic,xdim,densetype,resmode,mlid); return; }
    virtual void K4xfer(                                  double &res, int &minmaxind, int typeis, double xyprod, double yxprod, double diffis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, int ia, int ib, int ic, int id, int xdim, int densetype, int resmode, int mlid) const override { getQconst().K4xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,ia,ib,ic,id,xdim,densetype,resmode,mlid); return; }
    virtual void Kmxfer(                                  double &res, int &minmaxind, int typeis, double xyprod, double yxprod, double diffis, Vector<const SparseVector<gentype> *> &x, Vector<const vecInfo *> &xzinfo, Vector<int> &i, int xdim, int m, int densetype, int resmode, int mlid) const override { getQconst().Kmxfer(res,minmaxind,typeis,xyprod,yxprod,diffis,x,xzinfo,i,xdim,m,densetype,resmode,mlid); return; }

    virtual const gentype &xelm    (gentype &res, int i, int j) const override { return getQconst().xelm(res,i,j); }
    virtual       int      xindsize(int i)                      const override { return getQconst().xindsize(i); }

    // Training set modification:

    virtual int addTrainingVector (int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2) override { return  getQ().addTrainingVector(i,y,x,Cweigh,epsweigh,d); }
    virtual int qaddTrainingVector(int i, const gentype &y,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2) override { return getQ().qaddTrainingVector(i,y,x,Cweigh,epsweigh,d); }

    virtual int addTrainingVector (int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override { return  getQ().addTrainingVector(i,y,x,Cweigh,epsweigh); }
    virtual int qaddTrainingVector(int i, const Vector<gentype> &y,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override { return getQ().qaddTrainingVector(i,y,x,Cweigh,epsweigh); }

    virtual int removeTrainingVector(int i)                                       override { return getQ().removeTrainingVector(i);     }
    virtual int removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x) override { return getQ().removeTrainingVector(i,y,x); }
    virtual int removeTrainingVector(int i, int num)                              override { return getQ().removeTrainingVector(i,num); }

    virtual int setx(int                i, const SparseVector<gentype>          &x) override { return getQ().setx(i,x); }
    virtual int setx(const Vector<int> &i, const Vector<SparseVector<gentype> > &x) override { return getQ().setx(i,x); }
    virtual int setx(                      const Vector<SparseVector<gentype> > &x) override { return getQ().setx(  x); }

    virtual int qswapx(int                i, SparseVector<gentype>          &x, int dontupdate = 0) override { return getQ().qswapx(i,x,dontupdate); }
    virtual int qswapx(const Vector<int> &i, Vector<SparseVector<gentype> > &x, int dontupdate = 0) override { return getQ().qswapx(i,x,dontupdate); }
    virtual int qswapx(                      Vector<SparseVector<gentype> > &x, int dontupdate = 0) override { return getQ().qswapx(  x,dontupdate); }

    virtual int sety(int                i, const gentype         &nv) override { return getQ().sety(i,nv); }
    virtual int sety(const Vector<int> &i, const Vector<gentype> &nv) override { return getQ().sety(i,nv); }
    virtual int sety(                      const Vector<gentype> &nv) override { return getQ().sety(  nv); }

    virtual int sety(int                i, double                nv) override { return getQ().sety(i,nv); }
    virtual int sety(const Vector<int> &i, const Vector<double> &nv) override { return getQ().sety(i,nv); }
    virtual int sety(                      const Vector<double> &nv) override { return getQ().sety(  nv); }

    virtual int sety(int                i, const Vector<double>          &nv) override { return getQ().sety(i,nv); }
    virtual int sety(const Vector<int> &i, const Vector<Vector<double> > &nv) override { return getQ().sety(i,nv); }
    virtual int sety(                      const Vector<Vector<double> > &nv) override { return getQ().sety(  nv); }

    virtual int sety(int                i, const d_anion         &nv) override { return getQ().sety(i,nv); }
    virtual int sety(const Vector<int> &i, const Vector<d_anion> &nv) override { return getQ().sety(i,nv); }
    virtual int sety(                      const Vector<d_anion> &nv) override { return getQ().sety(  nv); }

    virtual int setd(int                i, int                nd) override { return getQ().setd(i,nd); }
    virtual int setd(const Vector<int> &i, const Vector<int> &nd) override { return getQ().setd(i,nd); }
    virtual int setd(                      const Vector<int> &nd) override { return getQ().setd(  nd); }

    virtual int setCweight(int                i, double                nv) override { return getQ().setCweight(i,nv); }
    virtual int setCweight(const Vector<int> &i, const Vector<double> &nv) override { return getQ().setCweight(i,nv); }
    virtual int setCweight(                      const Vector<double> &nv) override { return getQ().setCweight(  nv); }

    virtual int setCweightfuzz(int                i, double                nv) override { return getQ().setCweightfuzz(i,nv); }
    virtual int setCweightfuzz(const Vector<int> &i, const Vector<double> &nv) override { return getQ().setCweightfuzz(i,nv); }
    virtual int setCweightfuzz(                      const Vector<double> &nv) override { return getQ().setCweightfuzz(  nv); }

    virtual int setsigmaweight(int                i, double                nv) override { return getQ().setsigmaweight(i,nv); }
    virtual int setsigmaweight(const Vector<int> &i, const Vector<double> &nv) override { return getQ().setsigmaweight(i,nv); }
    virtual int setsigmaweight(                      const Vector<double> &nv) override { return getQ().setsigmaweight(  nv); }

    virtual int setepsweight(int                i, double                nv) override { return getQ().setepsweight(i,nv); }
    virtual int setepsweight(const Vector<int> &i, const Vector<double> &nv) override { return getQ().setepsweight(i,nv); }
    virtual int setepsweight(                      const Vector<double> &nv) override { return getQ().setepsweight(  nv); }

    virtual int scaleCweight    (double s) override { return getQ().scaleCweight(s);     }
    virtual int scaleCweightfuzz(double s) override { return getQ().scaleCweightfuzz(s); }
    virtual int scalesigmaweight(double s) override { return getQ().scalesigmaweight(s); }
    virtual int scaleepsweight  (double s) override { return getQ().scaleepsweight(s);   }

    virtual void assumeConsistentX  (void) override { getQ().assumeConsistentX();   return; }
    virtual void assumeInconsistentX(void) override { getQ().assumeInconsistentX(); return; }

    virtual int isXConsistent       (void) const override { return getQconst().isXConsistent();        }
    virtual int isXAssumedConsistent(void) const override { return getQconst().isXAssumedConsistent(); }

    virtual void xferx(const ML_Base &xsrc) override { getQ().xferx(xsrc); return; }

    // Basis stuff

    virtual int NbasisUU   (void) const override { return getQconst().NbasisUU();    }
    virtual int basisTypeUU(void) const override { return getQconst().basisTypeUU(); }
    virtual int defProjUU  (void) const override { return getQconst().defProjUU();   }

    virtual const Vector<gentype> &VbasisUU(void) const override { return getQconst().VbasisUU(); }

    virtual int setBasisYUU           (void)                     override { return getQ().setBasisYUU();             }
    virtual int setBasisUUU           (void)                     override { return getQ().setBasisUUU();             }
    virtual int addToBasisUU          (int i, const gentype &o)  override { return getQ().addToBasisUU(i,o);         }
    virtual int removeFromBasisUU     (int i)                    override { return getQ().removeFromBasisUU(i);      }
    virtual int setBasisUU            (int i, const gentype &o)  override { return getQ().setBasisUU(i,o);           }
    virtual int setBasisUU            (const Vector<gentype> &o) override { return getQ().setBasisUU(o);             }
    virtual int setDefaultProjectionUU(int d)                    override { return getQ().setDefaultProjectionUU(d); }
    virtual int setBasisUU            (int n, int d)             override { return getQ().setBasisUU(n,d);           }

    virtual int NbasisVV   (void) const override { return getQconst().NbasisVV();    }
    virtual int basisTypeVV(void) const override { return getQconst().basisTypeVV(); }
    virtual int defProjVV  (void) const override { return getQconst().defProjVV();   }

    virtual const Vector<gentype> &VbasisVV(void) const override { return getQconst().VbasisVV(); }

    virtual int setBasisYVV           (void)                     override { return getQ().setBasisYVV();             }
    virtual int setBasisUVV           (void)                     override { return getQ().setBasisUVV();             }
    virtual int addToBasisVV          (int i, const gentype &o)  override { return getQ().addToBasisVV(i,o);         }
    virtual int removeFromBasisVV     (int i)                    override { return getQ().removeFromBasisVV(i);      }
    virtual int setBasisVV            (int i, const gentype &o)  override { return getQ().setBasisVV(i,o);           }
    virtual int setBasisVV            (const Vector<gentype> &o) override { return getQ().setBasisVV(o);             }
    virtual int setDefaultProjectionVV(int d)                    override { return getQ().setDefaultProjectionVV(d); }
    virtual int setBasisVV            (int n, int d)             override { return getQ().setBasisVV(n,d);           }

    virtual const MercerKernel &getUUOutputKernel       (void)                                        const override { return getQconst().getUUOutputKernel();          }
    virtual       MercerKernel &getUUOutputKernel_unsafe(void)                                              override { return getQ().getUUOutputKernel_unsafe();        }
    virtual int                 resetUUOutputKernel     (int modind = 1)                                    override { return getQ().resetUUOutputKernel(modind);       }
    virtual int                 setUUOutputKernel       (const MercerKernel &xkernel, int modind = 1)       override { return getQ().setUUOutputKernel(xkernel,modind); }

    // RFF Similarity in random feature space

    virtual const MercerKernel &getRFFKernel       (void)                                        const override { return getQconst().getRFFKernel();          }
    virtual       MercerKernel &getRFFKernel_unsafe(void)                                              override { return getQ().getRFFKernel_unsafe();        }
    virtual int                 resetRFFKernel     (int modind = 1)                                    override { return getQ().resetRFFKernel(modind);       }
    virtual int                 setRFFKernel       (const MercerKernel &xkernel, int modind = 1)       override { return getQ().setRFFKernel(xkernel,modind); }

    // General modification and autoset functions

    virtual int randomise  (double sparsity) override { return getQ().randomise(sparsity); }
    virtual int autoen     (void)            override { return getQ().autoen();            }
    virtual int renormalise(void)            override { return getQ().renormalise();       }
    virtual int realign    (void)            override { return getQ().realign();           }

    virtual int setzerotol     (double zt)            override { return getQ().setzerotol(zt);                 }
    virtual int setOpttol      (double xopttol)       override { return getQ().setOpttol(xopttol);             }
    virtual int setOpttolb     (double xopttol)       override { return getQ().setOpttolb(xopttol);            }
    virtual int setOpttolc     (double xopttol)       override { return getQ().setOpttolc(xopttol);            }
    virtual int setOpttold     (double xopttol)       override { return getQ().setOpttold(xopttol);            }
    virtual int setlr          (double xlr)           override { return getQ().setlr(xlr);                     }
    virtual int setlrb         (double xlr)           override { return getQ().setlrb(xlr);                    }
    virtual int setlrc         (double xlr)           override { return getQ().setlrc(xlr);                    }
    virtual int setlrd         (double xlr)           override { return getQ().setlrd(xlr);                    }
    virtual int setmaxitcnt    (int    xmaxitcnt)     override { return getQ().setmaxitcnt(xmaxitcnt);         }
    virtual int setmaxtraintime(double xmaxtraintime) override { return getQ().setmaxtraintime(xmaxtraintime); }
    virtual int settraintimeend(double xtraintimeend) override { return getQ().settraintimeend(xtraintimeend); }

    virtual int setmaxitermvrank(int    nv) override { return getQ().setmaxitermvrank(nv); }
    virtual int setlrmvrank     (double nv) override { return getQ().setlrmvrank(nv);      }
    virtual int setztmvrank     (double nv) override { return getQ().setztmvrank(nv);      }

    virtual int setbetarank(double nv) override { return getQ().setbetarank(nv); }

    virtual int setC        (double xC)          override { return getQ().setC(xC);                 }
    virtual int setsigma    (double xsigma)      override { return getQ().setsigma(xsigma);         }
    virtual int setsigma_cut(double xsigma_cut)  override { return getQ().setsigma_cut(xsigma_cut); }
    virtual int seteps      (double xeps)        override { return getQ().seteps(xeps);             }
    virtual int setCclass   (int d, double xC)   override { return getQ().setCclass(d,xC);          }
    virtual int setepsclass (int d, double xeps) override { return getQ().setepsclass(d,xeps);      }

    virtual int setmpri  (int nv)            override { return getQ().setmpri(nv);   }
    virtual int setprival(const gentype &nv) override { return getQ().setprival(nv); }
    virtual int setpriml (const ML_Base *nv) override { return getQ().setpriml(nv);  }

    virtual int scale  (double a) override { return getQ().scale(a);  }
    virtual int reset  (void)     override { return getQ().reset();   }
    virtual int restart(void)     override { return getQ().restart(); }
    virtual int home   (void)     override { return getQ().home();    }

    virtual ML_Base &operator*=(double sf) override { return ( getQ() *= sf ); }

    virtual int scaleby(double sf) override { return getQ().scaleby(sf); }

    virtual int settspaceDim    (int newdim) override { return getQ().settspaceDim(newdim); }
    virtual int addtspaceFeat   (int i)      override { return getQ().addtspaceFeat(i);     }
    virtual int removetspaceFeat(int i)      override { return getQ().removetspaceFeat(i);  }
    virtual int addxspaceFeat   (int i)      override { return getQ().addxspaceFeat(i);     }
    virtual int removexspaceFeat(int i)      override { return getQ().removexspaceFeat(i);  }

    virtual int setsubtype(int i) override { return getQ().setsubtype(i); }

    virtual int setorder(int neword)                 override { return getQ().setorder(neword);        }
    virtual int addclass(int label, int epszero = 0) override { return getQ().addclass(label,epszero); }

    virtual int setNRff   (int nv) override { return getQ().setNRff   (nv); }
    virtual int setNRffRep(int nv) override { return getQ().setNRffRep(nv); }
    virtual int setReOnly (int nv) override { return getQ().setReOnly (nv); }
    virtual int setinAdam (int nv) override { return getQ().setinAdam (nv); }
    virtual int setoutGrad(int nv) override { return getQ().setoutGrad(nv); }

    // Sampling mode

    virtual int  isSampleMode(void) const override { return getQconst().isSampleMode(); }
    virtual int setSampleMode(int nv, const Vector<gentype> &xmin, const Vector<gentype> &xmax, int Nsamp, int sampSplit, int sampType, int xsampType, double sampScale, double sampSlack = 0) override { return getQ().setSampleMode(nv,xmin,xmax,Nsamp,sampSplit,sampType,xsampType,sampScale,sampSlack); }

    // Training functions:

    virtual void fudgeOn (void) override { getQ().fudgeOn();  return; }
    virtual void fudgeOff(void) override { getQ().fudgeOff(); return; }

    virtual int train(int &res)                              override { return getQ().train(res);            }
    virtual int train(int &res, svmvolatile int &killSwitch) override { return getQ().train(res,killSwitch); }

    // Information functions:

    virtual double loglikelihood(void) const { return getQconst().loglikelihood(); }
    virtual double maxinfogain  (void) const { return getQconst().maxinfogain  (); }
    virtual double RKHSnorm     (void) const { return getQconst().RKHSnorm     (); }
    virtual double RKHSabs      (void) const { return getQconst().RKHSabs      (); }

    // Evaluation Functions:

    virtual int ggTrainingVector(               gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = nullptr) const override { return getQconst().ggTrainingVector(     resg,i,retaltg,pxyprodi); }
    virtual int hhTrainingVector(gentype &resh,                int i,                  gentype ***pxyprodi = nullptr) const override { return getQconst().hhTrainingVector(resh,     i,        pxyprodi); }
    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = nullptr) const override { return getQconst().ghTrainingVector(resh,resg,i,retaltg,pxyprodi); }

    virtual double eTrainingVector(int i) const override { return getQconst().eTrainingVector(i); }

    virtual int covTrainingVector(gentype &resv, gentype &resmu, int i, int j, gentype ***pxyprodi = nullptr, gentype ***pxyprodj = nullptr, gentype **pxyprodij = nullptr) const override { return getQconst().covTrainingVector(resv,resmu,i,j,pxyprodi,pxyprodj,pxyprodij); }

    virtual double         &dedgTrainingVector(double         &res, int i) const override { return getQconst().dedgTrainingVector(res,i); }
    virtual Vector<double> &dedgTrainingVector(Vector<double> &res, int i) const override { return getQconst().dedgTrainingVector(res,i); }
    virtual d_anion        &dedgTrainingVector(d_anion        &res, int i) const override { return getQconst().dedgTrainingVector(res,i); }
    virtual gentype        &dedgTrainingVector(gentype        &res, int i) const override { return getQconst().dedgTrainingVector(res,i); }

    virtual double &d2edg2TrainingVector(double &res, int i) const override { return getQconst().d2edg2TrainingVector(res,i); }

    virtual double          dedKTrainingVector(int i, int j)               const override { return getQconst().dedKTrainingVector(i,j);   }
    virtual Vector<double> &dedKTrainingVector(Vector<double> &res, int i) const override { return getQconst().dedKTrainingVector(res,i); }
    virtual Matrix<double> &dedKTrainingVector(Matrix<double> &res)        const override { return getQconst().dedKTrainingVector(res);   }

    virtual void dgTrainingVectorX(Vector<gentype> &resx, int i) const override { getQconst().dgTrainingVectorX(resx,i); return; }
    virtual void dgTrainingVectorX(Vector<double>  &resx, int i) const override { getQconst().dgTrainingVectorX(resx,i); return; }

    virtual void deTrainingVectorX(Vector<gentype> &resx, int i) const override { getQconst().deTrainingVectorX(resx,i); return; }

    virtual void dgTrainingVectorX(Vector<gentype> &resx, const Vector<int> &i) const override { getQconst().dgTrainingVectorX(resx,i); return; }
    virtual void dgTrainingVectorX(Vector<double>  &resx, const Vector<int> &i) const override { getQconst().dgTrainingVectorX(resx,i); return; }

    virtual void deTrainingVectorX(Vector<gentype> &resx, const Vector<int> &i) const override { getQconst().deTrainingVectorX(resx,i); return; }

    virtual int ggTrainingVector(double         &resg, int i, int retaltg = 0, gentype ***pxyprodi = nullptr) const override { return getQconst().ggTrainingVector(resg,i,retaltg,pxyprodi); }
    virtual int ggTrainingVector(Vector<double> &resg, int i, int retaltg = 0, gentype ***pxyprodi = nullptr) const override { return getQconst().ggTrainingVector(resg,i,retaltg,pxyprodi); }
    virtual int ggTrainingVector(d_anion        &resg, int i, int retaltg = 0, gentype ***pxyprodi = nullptr) const override { return getQconst().ggTrainingVector(resg,i,retaltg,pxyprodi); }

    virtual void dgTrainingVector(Vector<gentype>         &res, gentype        &resn, int i) const override { getQconst().dgTrainingVector(res,resn,i); return; }
    virtual void dgTrainingVector(Vector<double>          &res, double         &resn, int i) const override { getQconst().dgTrainingVector(res,resn,i); return; }
    virtual void dgTrainingVector(Vector<Vector<double> > &res, Vector<double> &resn, int i) const override { getQconst().dgTrainingVector(res,resn,i); return; }
    virtual void dgTrainingVector(Vector<d_anion>         &res, d_anion        &resn, int i) const override { getQconst().dgTrainingVector(res,resn,i); return; }

    virtual void deTrainingVector(Vector<gentype> &res, gentype &resn, int i) const override { getQconst().deTrainingVector(res,resn,i); return; }

    virtual void dgTrainingVector(Vector<gentype>         &res, const Vector<int> &i) const override { getQconst().dgTrainingVector(res,i); return; }
    virtual void dgTrainingVector(Vector<double>          &res, const Vector<int> &i) const override { getQconst().dgTrainingVector(res,i); return; }
    virtual void dgTrainingVector(Vector<Vector<double> > &res, const Vector<int> &i) const override { getQconst().dgTrainingVector(res,i); return; }
    virtual void dgTrainingVector(Vector<d_anion>         &res, const Vector<int> &i) const override { getQconst().dgTrainingVector(res,i); return; }

    virtual void deTrainingVector(Vector<gentype> &res, const Vector<int> &i) const override { getQconst().deTrainingVector(res,i); return; }

    virtual void stabProbTrainingVector(double  &res, int i, int p, double pnrm, int rot, double mu, double B) const override { getQconst().stabProbTrainingVector(res,i,p,pnrm,rot,mu,B); return; }

    virtual int gg(               gentype &resg, const SparseVector<gentype> &x                 , const vecInfo *xinf = nullptr, gentype ***pxyprodx = nullptr) const override { return getQconst().gg(     resg,x,        xinf,pxyprodx); }
    virtual int hh(gentype &resh,                const SparseVector<gentype> &x                 , const vecInfo *xinf = nullptr, gentype ***pxyprodx = nullptr) const override { return getQconst().hh(resh,     x,        xinf,pxyprodx); }
    virtual int gh(gentype &resh, gentype &resg, const SparseVector<gentype> &x, int retaltg = 0, const vecInfo *xinf = nullptr, gentype ***pxyprodx = nullptr) const override { return getQconst().gh(resh,resg,x,retaltg,xinf,pxyprodx); }

    virtual double e(const gentype &y, const SparseVector<gentype> &x, const vecInfo *xinf = nullptr) const override { return getQconst().e(y,x,xinf); }

    virtual int cov(gentype &resv, gentype &resmu, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo *xainf = nullptr, const vecInfo *xbinf = nullptr, gentype ***pxyprodx = nullptr, gentype ***pxyprody = nullptr, gentype **pxyprodij = nullptr) const override { return getQconst().cov(resv,resmu,xa,xb,xainf,xbinf,pxyprodx,pxyprody,pxyprodij); }

    virtual void dedg(double         &res, const gentype &y, const SparseVector<gentype> &x, const vecInfo *xinf = nullptr) const override { getQconst().dedg(res,y,x,xinf); return; }
    virtual void dedg(Vector<double> &res, const gentype &y, const SparseVector<gentype> &x, const vecInfo *xinf = nullptr) const override { getQconst().dedg(res,y,x,xinf); return; }
    virtual void dedg(d_anion        &res, const gentype &y, const SparseVector<gentype> &x, const vecInfo *xinf = nullptr) const override { getQconst().dedg(res,y,x,xinf); return; }
    virtual void dedg(gentype        &res, const gentype &y, const SparseVector<gentype> &x, const vecInfo *xinf = nullptr) const override { getQconst().dedg(res,y,x,xinf); return; }

    virtual double &d2edg2(double &res, const gentype &y, const SparseVector<gentype> &x, const vecInfo *xinf = nullptr) const override { return getQconst().d2edg2(res,y,x,xinf); }

    virtual void dgX(Vector<gentype> &resx, const SparseVector<gentype> &x, const vecInfo *xinf = nullptr) const override { getQconst().dgX(resx,x,xinf); return; }
    virtual void dgX(Vector<double>  &resx, const SparseVector<gentype> &x, const vecInfo *xinf = nullptr) const override { getQconst().dgX(resx,x,xinf); return; }

    virtual void deX(Vector<gentype> &resx, const gentype &y, const SparseVector<gentype> &x, const vecInfo *xinf = nullptr) const override { getQconst().deX(resx,y,x,xinf); return; }

    virtual int gg(double         &resg, const SparseVector<gentype> &x, int retaltg = 0, const vecInfo *xinf = nullptr, gentype ***pxyprodx = nullptr) const override { return getQconst().gg(resg,x,retaltg,xinf,pxyprodx); }
    virtual int gg(Vector<double> &resg, const SparseVector<gentype> &x, int retaltg = 0, const vecInfo *xinf = nullptr, gentype ***pxyprodx = nullptr) const override { return getQconst().gg(resg,x,retaltg,xinf,pxyprodx); }
    virtual int gg(d_anion        &resg, const SparseVector<gentype> &x, int retaltg = 0, const vecInfo *xinf = nullptr, gentype ***pxyprodx = nullptr) const override { return getQconst().gg(resg,x,retaltg,xinf,pxyprodx); }

    virtual void dg(Vector<gentype>         &res, gentype        &resn, const SparseVector<gentype> &x, const vecInfo *xinf = nullptr) const override { getQconst().dg(res,resn,x,xinf); return; }
    virtual void dg(Vector<double>          &res, double         &resn, const SparseVector<gentype> &x, const vecInfo *xinf = nullptr) const override { getQconst().dg(res,resn,x,xinf); return; }
    virtual void dg(Vector<Vector<double> > &res, Vector<double> &resn, const SparseVector<gentype> &x, const vecInfo *xinf = nullptr) const override { getQconst().dg(res,resn,x,xinf); return; }
    virtual void dg(Vector<d_anion>         &res, d_anion        &resn, const SparseVector<gentype> &x, const vecInfo *xinf = nullptr) const override { getQconst().dg(res,resn,x,xinf); return; }

    virtual void de(Vector<gentype> &res, gentype &resn, const gentype &y, const SparseVector<gentype> &x, const vecInfo *xinf = nullptr) const override { getQconst().de(res,resn,y,x,xinf); return; }

    virtual void stabProb(double  &res, const SparseVector<gentype> &x, int p, double pnrm, int rot, double mu, double B) const override { getQconst().stabProb(res,x,p,pnrm,rot,mu,B); return; }

    // var and covar functions

    virtual int varTrainingVector(gentype &resv, gentype &resmu, int i, gentype ***pxyprodi = nullptr, gentype **pxyprodii = nullptr) const override { return getQconst().varTrainingVector(resv,resmu,i,pxyprodi,pxyprodii); }
    virtual int var(gentype &resv, gentype &resmu, const SparseVector<gentype> &xa, const vecInfo *xainf = nullptr, gentype ***pxyprodx = nullptr, gentype **pxyprodxx = nullptr) const override { return getQconst().var(resv,resmu,xa,xainf,pxyprodx,pxyprodxx); }

    virtual int covarTrainingVector(Matrix<gentype> &resv, const Vector<int> &i) const override { return getQconst().covarTrainingVector(resv,i); }
    virtual int covar(Matrix<gentype> &resv, const Vector<SparseVector<gentype> > &x) const override { return getQconst().covar(resv,x); }

    // Input-Output noise calculation

    virtual int noisevarTrainingVector(gentype &resv, gentype &resmu, int i, const SparseVector<gentype> &xvar, int u = -1, gentype ***pxyprodi = nullptr, gentype **pxyprodii = nullptr) const override { return getQconst().noisevarTrainingVector(resv,resmu,i,xvar,u,pxyprodi,pxyprodii); }
    virtual int noisevar(gentype &resv, gentype &resmu, const SparseVector<gentype> &xa, const SparseVector<gentype> &xvar, int u = -1, const vecInfo *xainf = nullptr, gentype ***pxyprodx = nullptr, gentype **pxyprodxx = nullptr) const override { return getQconst().noisevar(resv,resmu,xa,xvar,u,xainf,pxyprodx,pxyprodxx); }

    virtual int noisecovTrainingVector(gentype &resv, gentype &resmu, int i, int j, const SparseVector<gentype> &xvar, int u = -1, gentype ***pxyprodi = nullptr, gentype ***pxyprodj = nullptr, gentype **pxyprodij = nullptr) const override { return getQconst().noisecovTrainingVector(resv,resmu,i,j,xvar,u,pxyprodi,pxyprodj,pxyprodij); }
    virtual int noisecov(gentype &resv, gentype &resmu, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xvar, int u = -1, const vecInfo *xainf = nullptr, const vecInfo *xbinf = nullptr, gentype ***pxyprodx = nullptr, gentype ***pxyprody = nullptr, gentype **pxyprodxy = nullptr) const override { return getQconst().noisecov(resv,resmu,xa,xb,xvar,u,xainf,xbinf,pxyprodx,pxyprody,pxyprodxy); }

    // Training data tracking functions:

    virtual const Vector<int>          &indKey         (int u = -1) const override { return getQconst().indKey(u);          }
    virtual const Vector<int>          &indKeyCount    (int u = -1) const override { return getQconst().indKeyCount(u);     }
    virtual const Vector<int>          &dattypeKey     (int u = -1) const override { return getQconst().dattypeKey(u);      }
    virtual const Vector<Vector<int> > &dattypeKeyBreak(int u = -1) const override { return getQconst().dattypeKeyBreak(u); }

    // Other functions

    virtual void setaltx(const ML_Base *_altxsrc) override { getQ().setaltx(_altxsrc); return; }

    virtual int disable(int i)                override { return getQ().disable(i); }
    virtual int disable(const Vector<int> &i) override { return getQ().disable(i); }

    // Training data information functions (all assume no far/farfar/farfarfar or multivectors)

    virtual const SparseVector<gentype> &xsum   (SparseVector<gentype> &res) const override { return getQconst().xsum(res);    }
    virtual const SparseVector<gentype> &xmean  (SparseVector<gentype> &res) const override { return getQconst().xmean(res);   }
    virtual const SparseVector<gentype> &xmeansq(SparseVector<gentype> &res) const override { return getQconst().xmeansq(res); }
    virtual const SparseVector<gentype> &xsqsum (SparseVector<gentype> &res) const override { return getQconst().xsqsum(res);  }
    virtual const SparseVector<gentype> &xsqmean(SparseVector<gentype> &res) const override { return getQconst().xsqmean(res); }
    virtual const SparseVector<gentype> &xmedian(SparseVector<gentype> &res) const override { return getQconst().xmedian(res); }
    virtual const SparseVector<gentype> &xvar   (SparseVector<gentype> &res) const override { return getQconst().xvar(res);    }
    virtual const SparseVector<gentype> &xstddev(SparseVector<gentype> &res) const override { return getQconst().xstddev(res); }
    virtual const SparseVector<gentype> &xmax   (SparseVector<gentype> &res) const override { return getQconst().xmax(res);    }
    virtual const SparseVector<gentype> &xmin   (SparseVector<gentype> &res) const override { return getQconst().xmin(res);    }

    // Kernel normalisation function

    virtual int normKernelNone                  (void)                              override { return getQ().normKernelNone();                                   }
    virtual int normKernelZeroMeanUnitVariance  (int flatnorm = 0, int noshift = 0) override { return getQ().normKernelZeroMeanUnitVariance(flatnorm,noshift);   }
    virtual int normKernelZeroMedianUnitVariance(int flatnorm = 0, int noshift = 0) override { return getQ().normKernelZeroMedianUnitVariance(flatnorm,noshift); }
    virtual int normKernelUnitRange             (int flatnorm = 0, int noshift = 0) override { return getQ().normKernelUnitRange(flatnorm,noshift);              }

    // Helper functions for sparse variables

    virtual SparseVector<gentype> &xlateToSparse(SparseVector<gentype> &dest, const Vector<gentype>      &src) const override { return getQconst().xlateToSparse(dest,src); }
    virtual SparseVector<gentype> &xlateToSparse(SparseVector<gentype> &dest, const Vector<double>       &src) const override { return getQconst().xlateToSparse(dest,src); }
    virtual SparseVector<gentype> &xlateToSparse(SparseVector<gentype> &dest, const SparseVector<double> &src) const override { return getQconst().xlateToSparse(dest,src); }

    virtual Vector<gentype> &xlateFromSparse(Vector<gentype> &dest, const SparseVector<gentype> &src) const override { return getQconst().xlateFromSparse(dest,src); }
    virtual Vector<double>  &xlateFromSparse(Vector<double>  &dest, const SparseVector<gentype> &src) const override { return getQconst().xlateFromSparse(dest,src); }
    virtual Vector<double>  &xlateFromSparse(Vector<double>  &dest, const SparseVector<double>  &src) const override { return getQconst().xlateFromSparse(dest,src); }
    virtual Vector<double>  &xlateFromSparse(Vector<double>  &dest, const Vector<gentype>       &src) const override { return getQconst().xlateFromSparse(dest,src); }
    virtual Vector<double>  &xlateFromSparse(Vector<double>  &dest, const Vector<double>        &src) const override { return getQconst().xlateFromSparse(dest,src); }

    virtual Vector<double>  &xlateFromSparseTrainingVector(Vector<double>  &dest, int i) const override { return getQconst().xlateFromSparseTrainingVector(dest,i); }
    virtual Vector<gentype> &xlateFromSparseTrainingVector(Vector<gentype> &dest, int i) const override { return getQconst().xlateFromSparseTrainingVector(dest,i); }

    virtual SparseVector<gentype> &makeFullSparse(SparseVector<gentype> &dest) const override { return getQconst().makeFullSparse(dest); }

    // x detangling

    virtual int detangle_x(int i, int usextang = 0) const override
    {
        return getQconst().detangle_x(i,usextang);
    }

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
                   int usextang = 1, int allocxuntangifneeded = 1) const override
    {
        return getQconst().detangle_x(xuntang,xzinfountang,xnear,xfar,xfarfar,xfarfarfar,xnearinfo,xfarinfo,inear,ifar,ineartup,ifartup,ilr,irr,igr,igrR,iokr,iok,rankL,rankR,gmuL,gmuR,i,idiagr,xx,xzinfo,gradOrder,gradOrderR,iplanr,iplan,iset,idenseint,idensederiv,sumind,sumweight,diagoffset,ivectset,usextang,allocxuntangifneeded);
    }

















    // LSV_Generic specific stuff

    virtual       LSV_Generic &getLSV     (void)       override { return *this; }
    virtual const LSV_Generic &getLSVconst(void) const override { return *this; }

    // Constructors, destructors, assignment etc..

    virtual int setgamma(const Vector<gentype> &newgamma) { return getLSV().setgamma(newgamma); }
    virtual int setdelta(const gentype         &newdelta) { return getLSV().setdelta(newdelta); }

    // Additional information

    virtual int isVardelta (void) const { return getLSVconst().isVardelta();  }
    virtual int isZerodelta(void) const { return getLSVconst().isZerodelta(); }

    virtual const Vector<gentype> &gamma(void) const { return getLSVconst().gamma(); }
    virtual const gentype         &delta(void) const { return getLSVconst().delta(); }

    virtual const Matrix<double> &lsvGp(void) const { return getLSVconst().lsvGp(); }

    // General modification and autoset functions

    virtual int setVardelta (void) { return getLSV().setVardelta();  }
    virtual int setZerodelta(void) { return getLSV().setZerodelta(); }

};

inline void qswap(LSV_Generic_Deref &a, LSV_Generic_Deref &b)
{
    a.qswapinternal(b);

    return;
}

inline LSV_Generic_Deref &setzero(LSV_Generic_Deref &a)
{
    a.restart();

    return a;
}

#endif
