
//
// Vector regression SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _svm_vector_h
#define _svm_vector_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "svm_generic.hpp"
#include "svm_generic_deref.hpp"
#include "svm_scalar.hpp"
#include "svm_vector_atonce.hpp"
#include "svm_vector_redbin.hpp"
#include "svm_vector_matonce.hpp"
#include "svm_vector_mredbin.hpp"



class SVM_Vector;
class LSV_Gentyp;


// Swap function

inline void qswap(SVM_Vector &a, SVM_Vector &b);


class SVM_Vector : public SVM_Generic_Deref
{
    friend class LSV_Gentyp;

public:

    // Constructors, destructors, assignment etc..

    SVM_Vector() : SVM_Generic_Deref() { setaltx(nullptr); isQatonce = 0; isQreal   = 1; return; }
    SVM_Vector(const SVM_Vector &src) : SVM_Generic_Deref() { setaltx(nullptr); assign(src,0); return; }
    SVM_Vector(const SVM_Vector &src, const ML_Base *xsrc) : SVM_Generic_Deref() { setaltx(xsrc); assign(src,-1); return; }
    SVM_Vector &operator=(const SVM_Vector  &src) { assign(src); return *this; }
    virtual ~SVM_Vector() { return; }

    virtual int prealloc(int expectedN) override;
    virtual int preallocsize(void) const override;
    virtual void setmemsize(int memsize) override { Qatonce.setmemsize(memsize); Qredbin.setmemsize(memsize); QMatonce.setmemsize(memsize); QMredbin.setmemsize(memsize); return; }

    virtual void assign(const ML_Base &src, int isOnlySemi = 0) override;
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

    virtual int resetKernel(int modind = 1, int onlyChangeRowI = -1, int updateInfo = 1)        override { int res = Qatonce.resetKernel(modind,onlyChangeRowI,updateInfo); res |= Qredbin.resetKernel(modind,onlyChangeRowI,updateInfo); return res; }
    virtual int setKernel(const MercerKernel &xkernel, int modind = 1, int onlyChangeRowI = -1) override { int res = Qatonce.setKernel(xkernel,modind,onlyChangeRowI);      res |= Qredbin.setKernel(xkernel,modind,onlyChangeRowI);      return res; }

    // Training set modification:

    virtual int addTrainingVector (int i, const gentype &z, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2) override;
    virtual int qaddTrainingVector(int i, const gentype &z,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2) override;

    virtual int addTrainingVector (int i, const Vector<gentype> &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override;
    virtual int qaddTrainingVector(int i, const Vector<gentype> &z,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override;

    // General modification and autoset functions

    virtual int setzerotol     (double zt)            override { int res = Qatonce.setzerotol(zt);                 res |= Qredbin.setzerotol(zt);                 res |= QMatonce.setzerotol(zt);                 res |= QMredbin.setzerotol(zt);                 return res; }
    virtual int setOpttol      (double xopttol)       override { int res = Qatonce.setOpttol(xopttol);             res |= Qredbin.setOpttol(xopttol);             res |= QMatonce.setOpttol(xopttol);             res |= QMredbin.setOpttol(xopttol);             return res; }
    virtual int setOpttolb     (double xopttol)       override { int res = Qatonce.setOpttolb(xopttol);            res |= Qredbin.setOpttolb(xopttol);            res |= QMatonce.setOpttolb(xopttol);            res |= QMredbin.setOpttolb(xopttol);            return res; }
    virtual int setOpttolc     (double xopttol)       override { int res = Qatonce.setOpttolc(xopttol);            res |= Qredbin.setOpttolc(xopttol);            res |= QMatonce.setOpttolc(xopttol);            res |= QMredbin.setOpttolc(xopttol);            return res; }
    virtual int setOpttold     (double xopttol)       override { int res = Qatonce.setOpttold(xopttol);            res |= Qredbin.setOpttold(xopttol);            res |= QMatonce.setOpttold(xopttol);            res |= QMredbin.setOpttold(xopttol);            return res; }
    virtual int setlr          (double xlr)           override { int res = Qatonce.setlr(xlr);                     res |= Qredbin.setlr(xlr);                     res |= QMatonce.setlr(xlr);                     res |= QMredbin.setlr(xlr);                     return res; }
    virtual int setlrb         (double xlr)           override { int res = Qatonce.setlrb(xlr);                    res |= Qredbin.setlrb(xlr);                    res |= QMatonce.setlrb(xlr);                    res |= QMredbin.setlrb(xlr);                    return res; }
    virtual int setlrc         (double xlr)           override { int res = Qatonce.setlrc(xlr);                    res |= Qredbin.setlrc(xlr);                    res |= QMatonce.setlrc(xlr);                    res |= QMredbin.setlrc(xlr);                    return res; }
    virtual int setlrd         (double xlr)           override { int res = Qatonce.setlrd(xlr);                    res |= Qredbin.setlrd(xlr);                    res |= QMatonce.setlrd(xlr);                    res |= QMredbin.setlrd(xlr);                    return res; }
    virtual int setmaxitcnt    (int    xmaxitcnt)     override { int res = Qatonce.setmaxitcnt(xmaxitcnt);         res |= Qredbin.setmaxitcnt(xmaxitcnt);         res |= QMatonce.setmaxitcnt(xmaxitcnt);         res |= QMredbin.setmaxitcnt(xmaxitcnt);         return res; }
    virtual int setmaxtraintime(double xmaxtraintime) override { int res = Qatonce.setmaxtraintime(xmaxtraintime); res |= Qredbin.setmaxtraintime(xmaxtraintime); res |= QMatonce.setmaxtraintime(xmaxtraintime); res |= QMredbin.setmaxtraintime(xmaxtraintime); return res; }
    virtual int settraintimeend(double xtraintimeend) override { int res = Qatonce.settraintimeend(xtraintimeend); res |= Qredbin.settraintimeend(xtraintimeend); res |= QMatonce.settraintimeend(xtraintimeend); res |= QMredbin.settraintimeend(xtraintimeend); return res; }

    virtual int setC         (double xC)          override { int res = Qatonce.setC(xC);              res |= Qredbin.setC(xC);              res |= QMatonce.setC(xC);              res |= QMredbin.setC(xC);              return res; }
    virtual int setsigma     (double xC)          override { int res = Qatonce.setsigma(xC);          res |= Qredbin.setsigma(xC);          res |= QMatonce.setsigma(xC);          res |= QMredbin.setsigma(xC);          return res; }
    virtual int setsigma_cut (double xC)          override { int res = Qatonce.setsigma_cut(xC);      res |= Qredbin.setsigma_cut(xC);      res |= QMatonce.setsigma_cut(xC);      res |= QMredbin.setsigma_cut(xC);      return res; }
    virtual int seteps       (double xeps)        override { int res = Qatonce.seteps(xeps);          res |= Qredbin.seteps(xeps);          res |= QMatonce.seteps(xeps);          res |= QMredbin.seteps(xeps);          return res; }
    virtual int setCclass    (int d, double xC)   override { int res = Qatonce.setCclass(d,xC);       res |= Qredbin.setCclass(d,xC);       res |= QMatonce.setCclass(d,xC);       res |= QMredbin.setCclass(d,xC);       return res; }
    virtual int setsigmaclass(int d, double xsig) override { int res = Qatonce.setsigmaclass(d,xsig); res |= Qredbin.setsigmaclass(d,xsig); res |= QMatonce.setsigmaclass(d,xsig); res |= QMredbin.setsigmaclass(d,xsig); return res; }
    virtual int setepsclass  (int d, double xeps) override { int res = Qatonce.setepsclass(d,xeps);   res |= Qredbin.setepsclass(d,xeps);   res |= QMatonce.setepsclass(d,xeps);   res |= QMredbin.setepsclass(d,xeps);   return res; }

    virtual int scale  (double a) override { int res = Qatonce.scale(a); res |= Qredbin.scale(a); res |= QMatonce.scale(a); res |= QMredbin.scale(a); return res; }
    virtual int reset  (void)     override { int res = Qatonce.reset();  res |= Qredbin.reset();  res |= QMatonce.reset();  res |= QMredbin.reset();  return res; }
    virtual int restart(void)     override { SVM_Vector temp; *this = temp; return 1; }

    virtual ML_Base &operator*=(double sf) override { scale(sf); return *this; }

    virtual int scaleby(double sf) override { *this *= sf; return 1; }

    virtual int settspaceDim    (int newdim) override { int res = Qatonce.settspaceDim(newdim); res |= Qredbin.settspaceDim(newdim); res |= QMatonce.settspaceDim(newdim); res |= QMredbin.settspaceDim(newdim); return res; }
    virtual int addtspaceFeat   (int i)      override { int res = Qatonce.addtspaceFeat(i);     res |= Qredbin.addtspaceFeat(i);     res |= QMatonce.addtspaceFeat(i);     res |= QMredbin.addtspaceFeat(i);     return res; }
    virtual int removetspaceFeat(int i)      override { int res = Qatonce.removetspaceFeat(i);  res |= Qredbin.removetspaceFeat(i);  res |= QMatonce.removetspaceFeat(i);  res |= QMredbin.removetspaceFeat(i);  return res; }
    virtual int addxspaceFeat   (int i)      override { int res = Qatonce.addxspaceFeat(i);     res |= Qredbin.addxspaceFeat(i);     res |= QMatonce.addxspaceFeat(i);     res |= QMredbin.addxspaceFeat(i);     return res; }
    virtual int removexspaceFeat(int i)      override { int res = Qatonce.removexspaceFeat(i);  res |= Qredbin.removexspaceFeat(i);  res |= QMatonce.removexspaceFeat(i);  res |= QMredbin.removexspaceFeat(i);  return res; }

    virtual int setsubtype(int i) override;

    virtual int setorder(int neword)                 override { int res = Qatonce.setorder(neword);         res |= Qredbin.setorder(neword);         res |= QMatonce.setorder(neword);         res |= QMredbin.setorder(neword);        return res; }
    virtual int addclass(int label, int epszero = 0) override { int res = Qatonce.addclass(label,epszero);  res |= Qredbin.addclass(label,epszero);  res |= QMatonce.addclass(label,epszero);  res |= QMredbin.addclass(label,epszero); return res; }

    // Training functions:

    virtual void fudgeOn (void) override { Qatonce.fudgeOn();  Qredbin.fudgeOn();  return; }
    virtual void fudgeOff(void) override { Qatonce.fudgeOff(); Qredbin.fudgeOff(); return; }

    // Other functions

    virtual void setaltx(const ML_Base *_altxsrc) override { Qatonce.setaltx(_altxsrc); Qredbin.setaltx(_altxsrc); QMatonce.setaltx(_altxsrc); QMredbin.setaltx(_altxsrc); return; }










    // ================================================================
    //     Common functions for all SVMs
    // ================================================================

    virtual       SVM_Generic &getSVM     (void)       override { return *this; }
    virtual const SVM_Generic &getSVMconst(void) const override { return *this; }

    // Information functions (training data):

    virtual int findID(int ref) const override;

    // General modification and autoset functions

    virtual int setLinearCost   (void) override { int res = Qatonce.setLinearCost();    res |= Qredbin.setLinearCost();    res |= QMatonce.setLinearCost();    res |= QMredbin.setLinearCost();    return res; }
    virtual int setQuadraticCost(void) override { int res = Qatonce.setQuadraticCost(); res |= Qredbin.setQuadraticCost(); res |= QMatonce.setQuadraticCost(); res |= QMredbin.setQuadraticCost(); return res; }
    virtual int set1NormCost    (void) override { int res = Qatonce.set1NormCost();     res |= Qredbin.set1NormCost();     res |= QMatonce.set1NormCost();     res |= QMredbin.set1NormCost();     return res; }
    virtual int setVarBias      (void) override { int res = Qatonce.setVarBias();       res |= Qredbin.setVarBias();       res |= QMatonce.setVarBias();       res |= QMredbin.setVarBias();       return res; }
    virtual int setPosBias      (void) override { int res = Qatonce.setPosBias();       res |= Qredbin.setPosBias();       res |= QMatonce.setPosBias();       res |= QMredbin.setPosBias();       return res; }
    virtual int setNegBias      (void) override { int res = Qatonce.setNegBias();       res |= Qredbin.setNegBias();       res |= QMatonce.setNegBias();       res |= QMredbin.setNegBias();       return res; }

    //virtual int setVarBias  (int  q)                       override { int res = Qatonce.setVarBias(q);           res |= Qredbin.setVarBias(q);           res |= QMatonce.setVarBias(q);           res |= QMredbin.setVarBias(q);           return res; }
    //virtual int setPosBias  (int  q)                       override { int res = Qatonce.setPosBias(q);           res |= Qredbin.setPosBias(q);           res |= QMatonce.setPosBias(q);           res |= QMredbin.setPosBias(q);           return res; }
    //virtual int setNegBias  (int  q)                       override { int res = Qatonce.setNegBias(q);           res |= Qredbin.setNegBias(q);           res |= QMatonce.setNegBias(q);           res |= QMredbin.setNegBias(q);           return res; }
    //virtual int setFixedBias(int  q, double newbias = 0.0) override { int res = Qatonce.setFixedBias(q,newbias); res |= Qredbin.setFixedBias(q,newbias); res |= QMatonce.setFixedBias(q,newbias); res |= QMredbin.setFixedBias(q,newbias); return res; }
    //virtual int setFixedBias(const gentype &newbias)       override { int res = Qatonce.setFixedBias(newbias);   res |= Qredbin.setFixedBias(newbias);   res |= QMatonce.setFixedBias(newbias);   res |= QMredbin.setFixedBias(newbias);   return res; }

    virtual int setVarBias  (int  q)                 override { int res = Qredbin.setVarBias(q);           res |= QMredbin.setVarBias(q);           return res; }
    virtual int setPosBias  (int  q)                 override { int res = Qredbin.setPosBias(q);           res |= QMredbin.setPosBias(q);           return res; }
    virtual int setNegBias  (int  q)                 override { int res = Qredbin.setNegBias(q);           res |= QMredbin.setNegBias(q);           return res; }
    virtual int setFixedBias(int  q, double newbias) override { int res = Qredbin.setFixedBias(q,newbias); res |= QMredbin.setFixedBias(q,newbias); return res; }
    virtual int setFixedBias(const gentype &newbias) override { int res = Qredbin.setFixedBias(newbias);   res |= QMredbin.setFixedBias(newbias);   return res; }

    virtual int setOptActive(void) override { int res = Qatonce.setOptActive(); res |= Qredbin.setOptActive(); res |= QMatonce.setOptActive(); res |= QMredbin.setOptActive(); return res; }
    virtual int setOptSMO   (void) override { int res = Qatonce.setOptSMO();    res |= Qredbin.setOptSMO();    res |= QMatonce.setOptSMO();    res |= QMredbin.setOptSMO();    return res; }
    virtual int setOptD2C   (void) override { int res = Qatonce.setOptD2C();    res |= Qredbin.setOptD2C();    res |= QMatonce.setOptD2C();    res |= QMredbin.setOptD2C();    return res; }
    virtual int setOptGrad  (void) override { int res = Qatonce.setOptGrad();   res |= Qredbin.setOptGrad();   res |= QMatonce.setOptGrad();   res |= QMredbin.setOptGrad();   return res; }

    virtual int setFixedTube (void) override { int res = Qatonce.setFixedTube();  res |= Qredbin.setFixedTube();  res |= QMatonce.setFixedTube();  res |= QMredbin.setFixedTube();  return res; }
    virtual int setShrinkTube(void) override { int res = Qatonce.setShrinkTube(); res |= Qredbin.setShrinkTube(); res |= QMatonce.setShrinkTube(); res |= QMredbin.setShrinkTube(); return res; }

    virtual int setRestrictEpsPos(void) override { int res = Qatonce.setRestrictEpsPos(); res |= Qredbin.setRestrictEpsPos(); res |= QMatonce.setRestrictEpsPos(); res |= QMredbin.setRestrictEpsPos(); return res; }
    virtual int setRestrictEpsNeg(void) override { int res = Qatonce.setRestrictEpsNeg(); res |= Qredbin.setRestrictEpsNeg(); res |= QMatonce.setRestrictEpsNeg(); res |= QMredbin.setRestrictEpsNeg(); return res; }

    virtual int setClassifyViaSVR(void) override { int res = Qatonce.setClassifyViaSVR(); res |= Qredbin.setClassifyViaSVR(); res |= QMatonce.setClassifyViaSVR(); res |= QMredbin.setClassifyViaSVR(); return res; }
    virtual int setClassifyViaSVM(void) override { int res = Qatonce.setClassifyViaSVM(); res |= Qredbin.setClassifyViaSVM(); res |= QMatonce.setClassifyViaSVM(); res |= QMredbin.setClassifyViaSVM(); return res; }

    virtual int set1vsA   (void) override { int res = Qatonce.set1vsA();    res |= Qredbin.set1vsA();    res |= QMatonce.set1vsA();    res |= QMredbin.set1vsA();    return res; }
    virtual int set1vs1   (void) override { int res = Qatonce.set1vs1();    res |= Qredbin.set1vs1();    res |= QMatonce.set1vs1();    res |= QMredbin.set1vs1();    return res; }
    virtual int setDAGSVM (void) override { int res = Qatonce.setDAGSVM();  res |= Qredbin.setDAGSVM();  res |= QMatonce.setDAGSVM();  res |= QMredbin.setDAGSVM();  return res; }
    virtual int setMOC    (void) override { int res = Qatonce.setMOC();     res |= Qredbin.setMOC();     res |= QMatonce.setMOC();     res |= QMredbin.setMOC();     return res; }
    virtual int setmaxwins(void) override { int res = Qatonce.setmaxwins(); res |= Qredbin.setmaxwins(); res |= QMatonce.setmaxwins(); res |= QMredbin.setmaxwins(); return res; }
    virtual int setrecdiv (void) override { int res = Qatonce.setrecdiv();  res |= Qredbin.setrecdiv();  res |= QMatonce.setrecdiv();  res |= QMredbin.setrecdiv();  return res; }

    virtual int setatonce(void) override;
    virtual int setredbin(void) override;

    virtual int setKreal  (void) override;
    virtual int setKunreal(void) override;

    virtual int anomalyOn (int danomalyClass, double danomalyNu) override { int res = Qatonce.anomalyOn(danomalyClass,danomalyNu); res |= Qredbin.anomalyOn(danomalyClass,danomalyNu); res |= QMatonce.anomalyOn(danomalyClass,danomalyNu); res |= QMredbin.anomalyOn(danomalyClass,danomalyNu); return res; }
    virtual int anomalyOff(void)                                 override { int res = Qatonce.anomalyOff();                        res |= Qredbin.anomalyOff();                        res |= QMatonce.anomalyOff();                        res |= QMredbin.anomalyOff();                        return res; }

    virtual int setouterlr      (double xouterlr)     override { int res = Qatonce.setouterlr(xouterlr);              res |= Qredbin.setouterlr(xouterlr);              res |= QMatonce.setouterlr(xouterlr);              res |= QMredbin.setouterlr(xouterlr);              return res; }
    virtual int setoutermom     (double xoutermom)    override { int res = Qatonce.setoutermom(xoutermom);            res |= Qredbin.setoutermom(xoutermom);            res |= QMatonce.setoutermom(xoutermom);            res |= QMredbin.setoutermom(xoutermom);            return res; }
    virtual int setoutermethod  (int xoutermethod)    override { int res = Qatonce.setoutermethod(xoutermethod);      res |= Qredbin.setoutermethod(xoutermethod);      res |= QMatonce.setoutermethod(xoutermethod);      res |= QMredbin.setoutermethod(xoutermethod);      return res; }
    virtual int setoutertol     (double xoutertol)    override { int res = Qatonce.setoutertol(xoutertol);            res |= Qredbin.setoutertol(xoutertol);            res |= QMatonce.setoutertol(xoutertol);            res |= QMredbin.setoutertol(xoutertol);            return res; }
    virtual int setouterovsc    (double xouterovsc)   override { int res = Qatonce.setouterovsc(xouterovsc);          res |= Qredbin.setouterovsc(xouterovsc);          res |= QMatonce.setouterovsc(xouterovsc);          res |= QMredbin.setouterovsc(xouterovsc);          return res; }
    virtual int setoutermaxitcnt(int xoutermaxits)    override { int res = Qatonce.setoutermaxitcnt(xoutermaxits);    res |= Qredbin.setoutermaxitcnt(xoutermaxits);    res |= QMatonce.setoutermaxitcnt(xoutermaxits);    res |= QMredbin.setoutermaxitcnt(xoutermaxits);    return res; }
    virtual int setoutermaxcache(int xoutermaxcacheN) override { int res = Qatonce.setoutermaxcache(xoutermaxcacheN); res |= Qredbin.setoutermaxcache(xoutermaxcacheN); res |= QMatonce.setoutermaxcache(xoutermaxcacheN); res |= QMredbin.setoutermaxcache(xoutermaxcacheN); return res; }

    virtual int setmaxiterfuzzt(int xmaxiterfuzzt)               override { int res = Qatonce.setmaxiterfuzzt(xmaxiterfuzzt); res |= Qredbin.setmaxiterfuzzt(xmaxiterfuzzt); res |= QMatonce.setmaxiterfuzzt(xmaxiterfuzzt); res |= QMredbin.setmaxiterfuzzt(xmaxiterfuzzt); return res; }
    virtual int setusefuzzt    (int xusefuzzt)                   override { int res = Qatonce.setusefuzzt(xusefuzzt);         res |= Qredbin.setusefuzzt(xusefuzzt);         res |= QMatonce.setusefuzzt(xusefuzzt);         res |= QMredbin.setusefuzzt(xusefuzzt);         return res; }
    virtual int setlrfuzzt     (double xlrfuzzt)                 override { int res = Qatonce.setlrfuzzt(xlrfuzzt);           res |= Qredbin.setlrfuzzt(xlrfuzzt);           res |= QMatonce.setlrfuzzt(xlrfuzzt);           res |= QMredbin.setlrfuzzt(xlrfuzzt);           return res; }
    virtual int setztfuzzt     (double xztfuzzt)                 override { int res = Qatonce.setztfuzzt(xztfuzzt);           res |= Qredbin.setztfuzzt(xztfuzzt);           res |= QMatonce.setztfuzzt(xztfuzzt);           res |= QMredbin.setztfuzzt(xztfuzzt);           return res; }
    virtual int setcostfnfuzzt (const gentype &xcostfnfuzzt)     override { int res = Qatonce.setcostfnfuzzt(xcostfnfuzzt);   res |= Qredbin.setcostfnfuzzt(xcostfnfuzzt);   res |= QMatonce.setcostfnfuzzt(xcostfnfuzzt);   res |= QMredbin.setcostfnfuzzt(xcostfnfuzzt);   return res; }
    virtual int setcostfnfuzzt (const std::string &xcostfnfuzzt) override { int res = Qatonce.setcostfnfuzzt(xcostfnfuzzt);   res |= Qredbin.setcostfnfuzzt(xcostfnfuzzt);   res |= QMatonce.setcostfnfuzzt(xcostfnfuzzt);   res |= QMredbin.setcostfnfuzzt(xcostfnfuzzt);   return res; }

    virtual int setm(int xm) override { int res = Qatonce.setm(xm); res |= Qredbin.setm(xm); res |= QMatonce.setm(xm); res |= QMredbin.setm(xm); return res; }

    virtual int setLinBiasForce      (        double newval) override { int res = Qatonce.setLinBiasForce(newval);         res |= Qredbin.setLinBiasForce(newval);         res |= QMatonce.setLinBiasForce(newval);         res |= QMredbin.setLinBiasForce(newval);         return res; }
    virtual int setQuadBiasForce     (        double newval) override { int res = Qatonce.setQuadBiasForce(newval);        res |= Qredbin.setQuadBiasForce(newval);        res |= QMatonce.setQuadBiasForce(newval);        res |= QMredbin.setQuadBiasForce(newval);        return res; }
    virtual int setLinBiasForceclass (int  q, double newval) override { int res = Qatonce.setLinBiasForceclass(q,newval);  res |= Qredbin.setLinBiasForceclass(q,newval);  res |= QMatonce.setLinBiasForceclass(q,newval);  res |= QMredbin.setLinBiasForceclass(q,newval);  return res; }
    virtual int setQuadBiasForceclass(int  q, double newval) override { int res = Qatonce.setQuadBiasForceclass(q,newval); res |= Qredbin.setQuadBiasForceclass(q,newval); res |= QMatonce.setQuadBiasForceclass(q,newval); res |= QMredbin.setQuadBiasForceclass(q,newval); return res; }

    virtual int setnu    (double xnu)     override { int res = Qatonce.setnu(xnu);         res |= Qredbin.setnu(xnu);         res |= QMatonce.setnu(xnu);         res |= QMredbin.setnu(xnu);         return res; }
    virtual int setnuQuad(double xnuQuad) override { int res = Qatonce.setnuQuad(xnuQuad); res |= Qredbin.setnuQuad(xnuQuad); res |= QMatonce.setnuQuad(xnuQuad); res |= QMredbin.setnuQuad(xnuQuad); return res; }

    virtual int settheta  (double nv) override { int res = Qatonce.settheta(nv);   res |= Qredbin.settheta(nv);   res |= QMatonce.settheta(nv);   res |= QMredbin.settheta(nv);   return res; }
    virtual int setsimnorm(int    nv) override { int res = Qatonce.setsimnorm(nv); res |= Qredbin.setsimnorm(nv); res |= QMatonce.setsimnorm(nv); res |= QMredbin.setsimnorm(nv); return res; }

    virtual int autosetOff         (void)                            override { int res = Qatonce.autosetOff();         res |= Qredbin.autosetOff();         res |= QMatonce.autosetOff();                    res |= QMredbin.autosetOff();                    return res; }
    virtual int autosetCscaled     (double Cval)                     override { int res = Qatonce.autosetCscaled(Cval); res |= Qredbin.autosetCscaled(Cval); res |= QMatonce.autosetCscaled(Cval);            res |= QMredbin.autosetCscaled(Cval);            return res; }
    virtual int autosetCKmean      (void)                            override { int res = Qatonce.autosetCKmean();      res |= Qredbin.autosetCKmean();      res |= QMatonce.autosetCKmean();                 res |= QMredbin.autosetCKmean();                 return res; }
    virtual int autosetCKmedian    (void)                            override { int res = Qatonce.autosetCKmedian();    res |= Qredbin.autosetCKmedian();    res |= QMatonce.autosetCKmedian();               res |= QMredbin.autosetCKmedian();               return res; }
    virtual int autosetCNKmean     (void)                            override { int res = Qatonce.autosetCNKmean();     res |= Qredbin.autosetCNKmean();     res |= QMatonce.autosetCNKmean();                res |= QMredbin.autosetCNKmean();                return res; }
    virtual int autosetCNKmedian   (void)                            override { int res = Qatonce.autosetCNKmedian();   res |= Qredbin.autosetCNKmedian();   res |= QMatonce.autosetCNKmedian();              res |= QMredbin.autosetCNKmedian();              return res; }
    virtual int autosetLinBiasForce(double nuval, double Cval = 1.0) override { int res = Qatonce.autosetCNKmedian();   res |= Qredbin.autosetCNKmedian();   res |= QMatonce.autosetLinBiasForce(nuval,Cval); res |= QMredbin.autosetLinBiasForce(nuval,Cval); return res; }








protected:
    // ================================================================
    //     Base level functions
    // ================================================================

    // SVM specific

    virtual int addTrainingVector (int i, const Vector<double> &z, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2);
    virtual int qaddTrainingVector(int i, const Vector<double> &z,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2);

    virtual int addTrainingVector (int i, const Vector<Vector<double> > &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<int> &d);
    virtual int qaddTrainingVector(int i, const Vector<Vector<double> > &z,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<int> &d);

private:

    virtual       SVM_Generic &getQQ     (void)       override { if ( isQatonce &&  isQreal ) { return static_cast<      SVM_Generic &>(Qatonce);  } else if ( isQatonce && !isQreal ) { return static_cast<      SVM_Generic &>(QMatonce); } else if ( !isQatonce && isQreal ) { return static_cast<      SVM_Generic &>(Qredbin); } return static_cast<      SVM_Generic &>(QMredbin); }
    virtual const SVM_Generic &getQQconst(void) const override { if ( isQatonce &&  isQreal ) { return static_cast<const SVM_Generic &>(Qatonce);  } else if ( isQatonce && !isQreal ) { return static_cast<const SVM_Generic &>(QMatonce); } else if ( !isQatonce && isQreal ) { return static_cast<const SVM_Generic &>(Qredbin); } return static_cast<const SVM_Generic &>(QMredbin); }

    virtual       ML_Base &getQ     (void)       override { return static_cast<      ML_Base &>(getQQ());      }
    virtual const ML_Base &getQconst(void) const override { return static_cast<const ML_Base &>(getQQconst()); }

    int isQatonce;
    int isQreal;

    SVM_Vector_atonce Qatonce;
    SVM_Vector_redbin<SVM_Scalar> Qredbin;

    SVM_Vector_Matonce QMatonce;
    SVM_Vector_Mredbin QMredbin;
};

inline double norm2(const SVM_Vector &a);
inline double abs2 (const SVM_Vector &a);

inline double norm2(const SVM_Vector &a) { return a.RKHSnorm(); }
inline double abs2 (const SVM_Vector &a) { return a.RKHSabs();  }

inline void qswap(SVM_Vector &a, SVM_Vector &b)
{
    a.qswapinternal(b);

    return;
}

inline void SVM_Vector::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    SVM_Vector &b = dynamic_cast<SVM_Vector &>(bb.getML());

    SVM_Generic::qswapinternal(b);

    qswap(isQatonce,b.isQatonce);
    qswap(isQreal  ,b.isQreal  );
    qswap(Qatonce  ,b.Qatonce  );
    qswap(Qredbin  ,b.Qredbin  );
    qswap(QMatonce ,b.QMatonce );
    qswap(QMredbin ,b.QMredbin );

    return;
}

inline void SVM_Vector::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const SVM_Vector &b = dynamic_cast<const SVM_Vector &>(bb.getMLconst());

    SVM_Generic::semicopy(b);

         if (  isQatonce &&  isQreal ) { Qatonce. semicopy(b.Qatonce ); }
    else if (  isQatonce && !isQreal ) { QMatonce.semicopy(b.QMatonce); }
    else if ( !isQatonce &&  isQreal ) { Qredbin. semicopy(b.Qredbin ); }
    else                               { QMredbin.semicopy(b.QMredbin); }

    return;
}

inline void SVM_Vector::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const SVM_Vector &src = dynamic_cast<const SVM_Vector &>(bb.getMLconst());

    SVM_Generic::assign(src,onlySemiCopy);

    isQatonce = src.isQatonce;
    isQreal   = src.isQreal;

    Qatonce.assign(src.Qatonce,onlySemiCopy);
    Qredbin.assign(src.Qredbin,onlySemiCopy);
    QMatonce.assign(src.QMatonce,onlySemiCopy);
    QMredbin.assign(src.QMredbin,onlySemiCopy);

    return;
}

#endif
