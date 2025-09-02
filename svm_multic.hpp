
//
// Multiclass classification SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _svm_multic_h
#define _svm_multic_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "svm_generic.hpp"
#include "svm_generic_deref.hpp"
#include "svm_multic_redbin.hpp"
#include "svm_multic_atonce.hpp"







class SVM_MultiC;


// Swap function

inline void qswap(SVM_MultiC &a, SVM_MultiC &b);




class SVM_MultiC : public SVM_Generic_Deref
{
public:

    // Constructors, destructors, assignment etc..

    SVM_MultiC() : SVM_Generic_Deref() { setaltx(nullptr); isQatonce = 1; return; }
    SVM_MultiC(const SVM_MultiC &src) : SVM_Generic_Deref() { setaltx(nullptr); assign(src,0); return; }
    SVM_MultiC(const SVM_MultiC &src, const ML_Base *xsrc) : SVM_Generic_Deref() { setaltx(xsrc); assign(src,-1); return; }
    SVM_MultiC &operator=(const SVM_MultiC  &src) { assign(src); return *this; }
    virtual ~SVM_MultiC() { return; }

    virtual int prealloc(int expectedN) override;
    virtual int preallocsize(void) const override;
    virtual void setmemsize(int memsize) override { Qatonce.setmemsize(memsize); Qredbin.setmemsize(memsize); return; }

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0) override;
    virtual void semicopy(const ML_Base &src) override;
    virtual void qswapinternal(ML_Base &b) override;

    virtual int getparam (int ind, gentype         &val, const gentype         &xa, int ia, const gentype         &xb, int ib, charptr &desc) const override { return  SVM_Generic::getparam(ind,val,xa,ia,xb,ib,desc); }
    virtual int egetparam(int ind, Vector<gentype> &val, const Vector<gentype> &xa, int ia, const Vector<gentype> &xb, int ib               ) const override { return SVM_Generic::egetparam(ind,val,xa,ia,xb,ib     ); }

    virtual std::ostream &printstream(std::ostream &output, int dep) const override;
    virtual std::istream &inputstream(std::istream &input) override;

    virtual       ML_Base &getML     (void)       override { return static_cast<      ML_Base &>(getSVM());      }
    virtual const ML_Base &getMLconst(void) const override { return static_cast<const ML_Base &>(getSVMconst()); }

    // Information functions (training data):

    virtual double calcDistInt(int    ha, int    hb, int ia = -1, int db = 2) const override { return ML_Base::calcDistInt(ha,hb,ia,db); }
    virtual double calcDistDbl(double ha, double hb, int ia = -1, int db = 2) const override { return ML_Base::calcDistDbl(ha,hb,ia,db); }

    virtual const int *ClassLabelsInt(void) const override { return ML_Base::ClassLabelsInt();       }
    virtual int  getInternalClassInt(int y) const override { return ML_Base::getInternalClassInt(y); }

    // Kernel Modification

    virtual int resetKernel(int modind = 1, int onlyChangeRowI = -1, int updateInfo = 1)        override { int res = Qatonce.resetKernel(modind,onlyChangeRowI,updateInfo); res |= Qredbin.resetKernel(modind,-1,updateInfo); return res; }
    virtual int setKernel(const MercerKernel &xkernel, int modind = 1, int onlyChangeRowI = -1) override { int res = Qatonce.setKernel(xkernel,modind,onlyChangeRowI);      res |= Qredbin.setKernel(xkernel,modind);         return res; }

    // Training set modification:

    virtual int addTrainingVector (int i, const gentype &z, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2) override;
    virtual int qaddTrainingVector(int i, const gentype &z,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2) override;

    virtual int addTrainingVector (int i, const Vector<gentype> &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweight, const Vector<double> &epsweight) override;
    virtual int qaddTrainingVector(int i, const Vector<gentype> &z,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweight, const Vector<double> &epsweight) override;

    // General modification and autoset functions

    virtual int setzerotol     (double zt)            override { int res = Qatonce.setzerotol(zt);                 res |= Qredbin.setzerotol(zt);                 return res; }
    virtual int setOpttol      (double xopttol)       override { int res = Qatonce.setOpttol(xopttol);             res |= Qredbin.setOpttol(xopttol);             return res; }
    virtual int setOpttolb     (double xopttol)       override { int res = Qatonce.setOpttolb(xopttol);            res |= Qredbin.setOpttolb(xopttol);            return res; }
    virtual int setOpttolc     (double xopttol)       override { int res = Qatonce.setOpttolc(xopttol);            res |= Qredbin.setOpttolc(xopttol);            return res; }
    virtual int setOpttold     (double xopttol)       override { int res = Qatonce.setOpttold(xopttol);            res |= Qredbin.setOpttold(xopttol);            return res; }
    virtual int setlr          (double xlr)           override { int res = Qatonce.setlr(xlr);                     res |= Qredbin.setlr(xlr);                     return res; }
    virtual int setlrb         (double xlr)           override { int res = Qatonce.setlrb(xlr);                    res |= Qredbin.setlrb(xlr);                    return res; }
    virtual int setlrc         (double xlr)           override { int res = Qatonce.setlrc(xlr);                    res |= Qredbin.setlrc(xlr);                    return res; }
    virtual int setlrd         (double xlr)           override { int res = Qatonce.setlrd(xlr);                    res |= Qredbin.setlrd(xlr);                    return res; }
    virtual int setmaxitcnt    (int    xmaxitcnt)     override { int res = Qatonce.setmaxitcnt(xmaxitcnt);         res |= Qredbin.setmaxitcnt(xmaxitcnt);         return res; }
    virtual int setmaxtraintime(double xmaxtraintime) override { int res = Qatonce.setmaxtraintime(xmaxtraintime); res |= Qredbin.setmaxtraintime(xmaxtraintime); return res; }
    virtual int settraintimeend(double xtraintimeend) override { int res = Qatonce.settraintimeend(xtraintimeend); res |= Qredbin.settraintimeend(xtraintimeend); return res; }

    virtual int setC         (double xC)          override { int res = Qatonce.setC(xC);              res |= Qredbin.setC(xC);              return res; }
    virtual int setsigma     (double xC)          override { int res = Qatonce.setsigma(xC);          res |= Qredbin.setsigma(xC);          return res; }
    virtual int setsigma_cut (double xC)          override { int res = Qatonce.setsigma_cut(xC);      res |= Qredbin.setsigma_cut(xC);      return res; }
    virtual int seteps       (double xeps)        override { int res = Qatonce.seteps(xeps);          res |= Qredbin.seteps(xeps);          return res; }
    virtual int setCclass    (int d, double xC)   override { int res = Qatonce.setCclass(d,xC);       res |= Qredbin.setCclass(d,xC);       return res; }
    virtual int setsigmaclass(int d, double xsig) override { int res = Qatonce.setsigmaclass(d,xsig); res |= Qredbin.setsigmaclass(d,xsig); return res; }
    virtual int setepsclass  (int d, double xeps) override { int res = Qatonce.setepsclass(d,xeps);   res |= Qredbin.setepsclass(d,xeps);   return res; }

    virtual int scale  (double a) override { int res = Qredbin.scale(a); res |= Qatonce.scale(a); return res; }
    virtual int reset  (void)     override { int res = Qredbin.reset();  res |= Qatonce.reset();  return res; }
    virtual int restart(void)     override { SVM_MultiC temp; *this = temp; return 1; }

    virtual ML_Base &operator*=(double sf) override { scale(sf); return *this; }

    virtual int scaleby(double sf) override { *this *= sf; return 1; }

    virtual int setsubtype(int i) override;

    virtual int addclass(int label, int epszero = 0) override { NiceAssert( !epszero || isatonce() ); int res = Qatonce.addclass(label,epszero); res |= Qredbin.addclass(label); return res; }

    // Training functions:

    virtual void fudgeOn (void) override { Qatonce.fudgeOn();  Qredbin.fudgeOn();  return; }
    virtual void fudgeOff(void) override { Qatonce.fudgeOff(); Qredbin.fudgeOff(); return; }

    // Other functions

    virtual void setaltx(const ML_Base *_altxsrc) override { Qatonce.setaltx(_altxsrc); Qredbin.setaltx(_altxsrc); return; }










    // ================================================================
    //     Common functions for all SVMs
    // ================================================================

    virtual       SVM_Generic &getSVM     (void)       override { return *this; }
    virtual const SVM_Generic &getSVMconst(void) const override { return *this; }

    // Information functions (training data):

    virtual int findID(int ref) const override { return getQQconst().findID(ref);   }

    // General modification and autoset functions

    virtual int setLinearCost   (void) override { int res = Qatonce.setLinearCost();    res |= Qredbin.setLinearCost();    return res; }
    virtual int setQuadraticCost(void) override { int res = Qatonce.setQuadraticCost(); res |= Qredbin.setQuadraticCost(); return res; }
    virtual int set1NormCost    (void) override { int res = Qatonce.set1NormCost();     res |= Qredbin.set1NormCost();     return res; }

    virtual int setOptActive(void) override { int res = Qatonce.setOptActive(); res |= Qredbin.setOptActive(); return res; }
    virtual int setOptSMO   (void) override { int res = Qatonce.setOptSMO();    res |= Qredbin.setOptSMO();    return res; }
    virtual int setOptD2C   (void) override { int res = Qatonce.setOptD2C();    res |= Qredbin.setOptD2C();    return res; }
    virtual int setOptGrad  (void) override { int res = Qatonce.setOptGrad();   res |= Qredbin.setOptGrad();   return res; }

    virtual int set1vsA   (void) override { int res = setredbin(); res |= Qredbin.set1vsA();    return res; }
    virtual int set1vs1   (void) override { int res = setredbin(); res |= Qredbin.set1vs1();    return res; }
    virtual int setDAGSVM (void) override { int res = setredbin(); res |= Qredbin.setDAGSVM();  return res; }
    virtual int setMOC    (void) override { int res = setredbin(); res |= Qredbin.setMOC();     return res; }
    virtual int setmaxwins(void) override { int res = setatonce(); res |= Qatonce.setmaxwins(); return res; }
    virtual int setrecdiv (void) override { int res = setatonce(); res |= Qatonce.setrecdiv();  return res; }

    virtual int setatonce(void) override;
    virtual int setredbin(void) override;

    virtual int setKreal  (void) override { int res = Qatonce.setKreal();   res |= Qredbin.setKreal();   return res; }
    virtual int setKunreal(void) override { int res = Qatonce.setKunreal(); res |= Qredbin.setKunreal(); return res; }

    virtual int anomalyOn (int danomalyClass, double danomalyNu) override { int res = Qatonce.anomalyOn(danomalyClass,danomalyNu); res |= Qredbin.anomalyOn(danomalyClass,danomalyNu); return res; }
    virtual int anomalyOff(void)                                 override { int res = Qatonce.anomalyOff();                        res |= Qredbin.anomalyOff();                        return res; }

    virtual int setouterlr      (double xouterlr)     override { int res = Qatonce.setouterlr(xouterlr);              res |= Qredbin.setouterlr(xouterlr);              return res; }
    virtual int setoutermom     (double xoutermom)    override { int res = Qatonce.setoutermom(xoutermom);            res |= Qredbin.setoutermom(xoutermom);            return res; }
    virtual int setoutermethod  (int xoutermethod)    override { int res = Qatonce.setoutermethod(xoutermethod);      res |= Qredbin.setoutermethod(xoutermethod);      return res; }
    virtual int setoutertol     (double xoutertol)    override { int res = Qatonce.setoutertol(xoutertol);            res |= Qredbin.setoutertol(xoutertol);            return res; }
    virtual int setouterovsc    (double xouterovsc)   override { int res = Qatonce.setouterovsc(xouterovsc);          res |= Qredbin.setouterovsc(xouterovsc);          return res; }
    virtual int setoutermaxitcnt(int xoutermaxits)    override { int res = Qatonce.setoutermaxitcnt(xoutermaxits);    res |= Qredbin.setoutermaxitcnt(xoutermaxits);    return res; }
    virtual int setoutermaxcache(int xoutermaxcacheN) override { int res = Qatonce.setoutermaxcache(xoutermaxcacheN); res |= Qredbin.setoutermaxcache(xoutermaxcacheN); return res; }

    virtual int setmaxiterfuzzt(int xmaxiterfuzzt)               override { int res = Qatonce.setmaxiterfuzzt(xmaxiterfuzzt); res |= Qredbin.setmaxiterfuzzt(xmaxiterfuzzt); return res; }
    virtual int setusefuzzt    (int xusefuzzt)                   override { int res = Qatonce.setusefuzzt(xusefuzzt);         res |= Qredbin.setusefuzzt(xusefuzzt);         return res; }
    virtual int setlrfuzzt     (double xlrfuzzt)                 override { int res = Qatonce.setlrfuzzt(xlrfuzzt);           res |= Qredbin.setlrfuzzt(xlrfuzzt);           return res; }
    virtual int setztfuzzt     (double xztfuzzt)                 override { int res = Qatonce.setztfuzzt(xztfuzzt);           res |= Qredbin.setztfuzzt(xztfuzzt);           return res; }
    virtual int setcostfnfuzzt (const gentype &xcostfnfuzzt)     override { int res = Qatonce.setcostfnfuzzt(xcostfnfuzzt);   res |= Qredbin.setcostfnfuzzt(xcostfnfuzzt);   return res; }
    virtual int setcostfnfuzzt (const std::string &xcostfnfuzzt) override { int res = Qatonce.setcostfnfuzzt(xcostfnfuzzt);   res |= Qredbin.setcostfnfuzzt(xcostfnfuzzt);   return res; }

    virtual int setLinBiasForce      (double newval)         override { int res = Qatonce.setLinBiasForce(newval);          res |= Qredbin.setLinBiasForce(newval);         return res; }
    virtual int setQuadBiasForce     (double newval)         override { int res = getQQ().setQuadBiasForce(newval);                                                         return res; }
    virtual int setLinBiasForceclass (int dq, double newval) override { int res = Qatonce.setLinBiasForceclass(dq,newval);  res |= Qredbin.setLinBiasForceclass(dq,newval); return res; }
    virtual int setQuadBiasForceclass(int dq, double newval) override { int res = getQQ().setQuadBiasForceclass(dq,newval);                                                 return res; }

    virtual int autosetOff      (void)        override { int res = Qatonce.autosetOff();         res |= Qredbin.autosetOff();         return res; }
    virtual int autosetCscaled  (double Cval) override { int res = Qatonce.autosetCscaled(Cval); res |= Qredbin.autosetCscaled(Cval); return res; }
    virtual int autosetCKmean   (void)        override { int res = Qatonce.autosetCKmean();      res |= Qredbin.autosetCKmean();      return res; }
    virtual int autosetCKmedian (void)        override { int res = Qatonce.autosetCKmedian();    res |= Qredbin.autosetCKmedian();    return res; }
    virtual int autosetCNKmean  (void)        override { int res = Qatonce.autosetCNKmean();     res |= Qredbin.autosetCNKmean();     return res; }
    virtual int autosetCNKmedian(void)        override { int res = Qatonce.autosetCNKmedian();   res |= Qredbin.autosetCNKmedian();   return res; }








protected:
    // ================================================================
    //     Base level functions
    // ================================================================

    // SVM specific

    virtual int addTrainingVector (int i, int d, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);
    virtual int qaddTrainingVector(int i, int d,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);

    virtual int addTrainingVector (int i, const Vector<int> &d, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);
    virtual int qaddTrainingVector(int i, const Vector<int> &d,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);

private:

    virtual       SVM_Generic &getQQ     (void)       override { if ( isQatonce ) { return static_cast<      SVM_Generic &>(Qatonce); } return static_cast<      SVM_Generic &>(Qredbin); }
    virtual const SVM_Generic &getQQconst(void) const override { if ( isQatonce ) { return static_cast<const SVM_Generic &>(Qatonce); } return static_cast<const SVM_Generic &>(Qredbin); }

    virtual       ML_Base &getQ     (void)       override { return static_cast<      ML_Base &>(getQQ());      }
    virtual const ML_Base &getQconst(void) const override { return static_cast<const ML_Base &>(getQQconst()); }

    int isQatonce;

    SVM_MultiC_redbin Qredbin;
    SVM_MultiC_atonce Qatonce;
};

inline double norm2(const SVM_MultiC &a);
inline double abs2 (const SVM_MultiC &a);

inline double norm2(const SVM_MultiC &a) { return a.RKHSnorm(); }
inline double abs2 (const SVM_MultiC &a) { return a.RKHSabs();  }

inline void qswap(SVM_MultiC &a, SVM_MultiC &b)
{
    a.qswapinternal(b);

    return;
}

inline void SVM_MultiC::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    SVM_MultiC &b = dynamic_cast<SVM_MultiC &>(bb.getML());

    SVM_Generic::qswapinternal(b);

    qswap(Qredbin  ,b.Qredbin  );
    qswap(Qatonce  ,b.Qatonce  );
    qswap(isQatonce,b.isQatonce);

    return;
}

inline void SVM_MultiC::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const SVM_MultiC &b = dynamic_cast<const SVM_MultiC &>(bb.getMLconst());

    SVM_Generic::semicopy(b);

    if ( isQatonce ) { Qatonce.semicopy(b.Qatonce); }
    else             { Qredbin.semicopy(b.Qredbin); }

    return;
}

inline void SVM_MultiC::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const SVM_MultiC &src = dynamic_cast<const SVM_MultiC &>(bb.getMLconst());

    SVM_Generic::assign(src,onlySemiCopy);

    isQatonce = src.isQatonce;

    Qredbin.assign(src.Qredbin,onlySemiCopy);
    Qatonce.assign(src.Qatonce,onlySemiCopy);

    return;
}

#endif
