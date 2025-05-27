
//
// Vector regression SVM (matrix reduction to binary)
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _svm_vector_mredbin_h
#define _svm_vector_mredbin_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "svm_generic.hpp"
#include "svm_scalar.hpp"



class SVM_Vector_Mredbin;
class scalar_callback : public kernPrecursor
{
public:
    virtual int isKVarianceNZ(void) const;

    virtual void K0xfer(gentype &res, int &minmaxind, int typeis,
                        const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                       int xdim, int densetype, int resmode, int mlid) const;

    virtual void K1xfer(gentype &res, int &minmaxind, int typeis,
                        const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                        const SparseVector<gentype> &xa, 
                        const vecInfo &xainfo, 
                        int ia, 
                        int xdim, int densetype, int resmode, int mlid) const;

    virtual void K2xfer(gentype &dxyprod, gentype &ddiffis, gentype &res, int &minmaxind, int typeis,
                        const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                        const vecInfo &xainfo, const vecInfo &xbinfo,
                        int ia, int ib,
                        int xdim, int densetype, int resmode, int mlid) const;

    virtual void K3xfer(gentype &res, int &minmaxind, int typeis,
                        const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, 
                        const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, 
                        int ia, int ib, int ic, 
                        int xdim, int densetype, int resmode, int mlid) const;

    virtual void K4xfer(gentype &res, int &minmaxind, int typeis,
                        const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                        const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
                        int ia, int ib, int ic, int id,
                        int xdim, int densetype, int resmode, int mlid) const;

    virtual void Kmxfer(gentype &res, int &minmaxind, int typeis,
                        const gentype &xyprod, const gentype &yxprod, const gentype &diffis,
                        Vector<const SparseVector<gentype> *> &x,
                        Vector<const vecInfo *> &xzinfo,
                        Vector<int> &ii,
                        int xdim, int m, int densetype, int resmode, int mlid) const;

    virtual void K0xfer(double &res, int &minmaxind, int typeis,
                        double xyprod, double yxprod, double diffis,
                        int xdim, int densetype, int resmode, int mlid) const;

    virtual void K1xfer(double &res, int &minmaxind, int typeis,
                        double xyprod, double yxprod, double diffis,
                        const SparseVector<gentype> &xa, 
                        const vecInfo &xainfo, 
                        int ia, 
                        int xdim, int densetype, int resmode, int mlid) const;

    virtual void K2xfer(double &dxyprod, double &ddiffis, double &res, int &minmaxind, int typeis,
                        double xyprod, double yxprod, double diffis,
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                        const vecInfo &xainfo, const vecInfo &xbinfo,
                        int ia, int ib,
                        int xdim, int densetype, int resmode, int mlid) const;

    virtual void K3xfer(double &res, int &minmaxind, int typeis,
                        double xyprod, double yxprod, double diffis,
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, 
                        const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, 
                        int ia, int ib, int ic, 
                        int xdim, int densetype, int resmode, int mlid) const;

    virtual void K4xfer(double &res, int &minmaxind, int typeis,
                        double xyprod, double yxprod, double diffis,
                        const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                        const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
                        int ia, int ib, int ic, int id,
                        int xdim, int densetype, int resmode, int mlid) const;

    virtual void Kmxfer(double &res, int &minmaxind, int typeis,
                        double xyprod, double yxprod, double diffis,
                        Vector<const SparseVector<gentype> *> &x,
                        Vector<const vecInfo *> &xzinfo,
                        Vector<int> &ii,
                        int xdim, int m, int densetype, int resmode, int mlid) const;

    SVM_Vector_Mredbin *realOwner;
};



// Swap function

inline void qswap(SVM_Vector_Mredbin &a, SVM_Vector_Mredbin &b);



class SVM_Vector_Mredbin : public SVM_Generic
{
    friend class scalar_callback;

public:

    // Constructors, destructors, assignment operators and similar

    SVM_Vector_Mredbin();
    SVM_Vector_Mredbin(const SVM_Vector_Mredbin &src);
    SVM_Vector_Mredbin(const SVM_Vector_Mredbin &src, const ML_Base *xsrc);
    SVM_Vector_Mredbin &operator=(const SVM_Vector_Mredbin &src) { assign(src); return *this; }
    virtual ~SVM_Vector_Mredbin();

    virtual int prealloc(int expectedN) override;
    virtual int preallocsize(void) const override;
    virtual void setmemsize(int memsize) override { Q.setmemsize(memsize); return; }

    virtual int scale(double a) override;
    virtual int reset(void) override;
    virtual int restart(void) override { SVM_Vector_Mredbin temp; *this = temp; return 1; }

    virtual int setAlphaV(const Vector<Vector<double> > &newAlpha) override;
    virtual int setBiasV(const Vector<double>  &newBias) override;

    virtual std::ostream &printstream(std::ostream &output, int dep) const override;
    virtual std::istream &inputstream(std::istream &input) override;

    // Information:

    virtual int isTrained(void) const override { return Q.isTrained(); }

    virtual int N  (void)  const override { return aN;                 }
    virtual int NS (void)  const override { return aNS;                }
    virtual int NZ (void)  const override { return aNZ;                }
    virtual int NF (void)  const override { return aNF;                }
    virtual int NC (void)  const override { return aNC;                }
    virtual int NLB(void)  const override { return 0;                  }
    virtual int NLF(void)  const override { return 0;                  }
    virtual int NUF(void)  const override { return aNF;                }
    virtual int NUB(void)  const override { return aNC-aNZ;            }
    virtual int NNC(int d) const override { return (Q.NNC(d))/order(); }
    virtual int NS (int q) const override { (void) q; return NS();     }
    virtual int NZ (int q) const override { (void) q; return NZ();     }
    virtual int NF (int q) const override { (void) q; return NF();     }
    virtual int NC (int q) const override { (void) q; return NC();     }
    virtual int NLB(int q) const override { (void) q; return NLB();    }
    virtual int NLF(int q) const override { (void) q; return NLF();    }
    virtual int NUF(int q) const override { (void) q; return NUF();    }
    virtual int NUB(int q) const override { (void) q; return NUB();    }

    virtual int tspaceDim(void)  const override { return dbiasA.size();  }
    virtual int numClasses(void) const override { return 0;              }
    virtual int type(void)       const override { return 4;              }
    virtual int subtype(void)    const override { return 3;              }

    virtual int numInternalClasses(void) const override { return 1; }

    virtual char gOutType(void) const override { return 'V'; }
    virtual char hOutType(void) const override { return 'V'; }
    virtual char targType(void) const override { return 'V'; }
    virtual double calcDist(const gentype &ha, const gentype &hb, int ia = -1, int db = 2) const override;

    virtual const Vector<int>          &ClassLabels(void)   const override { return Q.ClassLabels(); }
    virtual const Vector<Vector<int> > &ClassRep(void)      const override { return Q.ClassRep();    }
    virtual int                         findID(int ref)     const override { return Q.findID(ref);   }

    virtual int isLinearCost(void)      const override { return Q.isLinearCost();    }
    virtual int isQuadraticCost(void)   const override { return Q.isQuadraticCost(); }
    virtual int is1NormCost(void)       const override { return 0;                   }
    virtual int isVarBias(void)         const override { return Q.isVarBias();       }
    virtual int isPosBias(void)         const override { return Q.isPosBias();       }
    virtual int isNegBias(void)         const override { return Q.isNegBias();       }
    virtual int isFixedBias(void)       const override { return Q.isFixedBias();     }
    virtual int isVarBias(int q)        const override { return Q.isVarBias(q);      }
    virtual int isPosBias(int q)        const override { return Q.isPosBias(q);      }
    virtual int isNegBias(int q)        const override { return Q.isNegBias(q);      }
    virtual int isFixedBias(int q)      const override { return Q.isFixedBias(q);    }

    virtual int isOptActive(void) const override { return Q.isOptActive(); }
    virtual int isOptSMO(void)    const override { return Q.isOptSMO();    }
    virtual int isOptD2C(void)    const override { return Q.isOptD2C();    }
    virtual int isOptGrad(void)   const override { return Q.isOptGrad();   }

    virtual int m(void) const override { return Q.m(); }

    virtual int NRff(void) const override { return 0; }

    virtual double C(void)            const override { return Q.C();            }
    virtual double eps(void)          const override { return Q.eps();          }
    virtual double Cclass(int d)      const override { return Q.Cclass(d);      }
    virtual double epsclass(int d)    const override { return Q.epsclass(d);    }

    virtual int    memsize     (void) const override { return Q.memsize();      }
    virtual double zerotol     (void) const override { return Q.zerotol();      }
    virtual double Opttol      (void) const override { return Q.Opttol();       }
    virtual double Opttolb     (void) const override { return Q.Opttolb();      }
    virtual double Opttolc     (void) const override { return Q.Opttolc();      }
    virtual double Opttold     (void) const override { return Q.Opttold();      }
    virtual double lr          (void) const override { return Q.lr();           }
    virtual double lrb         (void) const override { return Q.lrb();          }
    virtual double lrc         (void) const override { return Q.lrc();          }
    virtual double lrd         (void) const override { return Q.lrd();          }
    virtual int    maxitcnt    (void) const override { return Q.maxitcnt();     }
    virtual double maxtraintime(void) const override { return Q.maxtraintime(); }
    virtual double traintimeend(void) const override { return Q.traintimeend(); }

    virtual double outerlr(void)      const override { return Q.outerlr();      }
    virtual double outertol(void)     const override { return Q.outertol();     }

    virtual       int      maxiterfuzzt(void) const override { return Q.maxiterfuzzt(); }
    virtual       int      usefuzzt(void)     const override { return Q.usefuzzt();     }
    virtual       double   lrfuzzt(void)      const override { return Q.lrfuzzt();      }
    virtual       double   ztfuzzt(void)      const override { return Q.ztfuzzt();      }
    virtual const gentype &costfnfuzzt(void)  const override { return Q.costfnfuzzt();  }

    virtual double LinBiasForce(void)        const override { return Q.LinBiasForce();   }
    virtual double QuadBiasForce(void)       const override { return Q.QuadBiasForce();  }
    virtual double LinBiasForce(int q)       const override { return Q.LinBiasForce(q);  }
    virtual double QuadBiasForce(int q)      const override { return Q.QuadBiasForce(q); }

    virtual int isFixedTube(void)  const override { return Q.isFixedTube();  }
    virtual int isShrinkTube(void) const override { return Q.isShrinkTube(); }

    virtual int isRestrictEpsPos(void) const override { return Q.isRestrictEpsPos(); }
    virtual int isRestrictEpsNeg(void) const override { return Q.isRestrictEpsNeg(); }

    virtual double nu(void)     const override { return Q.nu();     }
    virtual double nuQuad(void) const override { return Q.nuQuad(); }

    virtual int isClassifyViaSVR(void) const override { return Q.isClassifyViaSVR(); }
    virtual int isClassifyViaSVM(void) const override { return Q.isClassifyViaSVM(); }

    virtual int is1vsA(void)    const override { return 0; }
    virtual int is1vs1(void)    const override { return 0; }
    virtual int isDAGSVM(void)  const override { return 0; }
    virtual int isMOC(void)     const override { return 0; }
    virtual int ismaxwins(void) const override { return 0; }
    virtual int isrecdiv(void)  const override { return 0; }

    virtual int isatonce(void) const override { return 0; }
    virtual int isredbin(void) const override { return 1; }

    virtual int isKreal(void)   const override { return 0; }
    virtual int isKunreal(void) const override { return 1; }

    virtual int isClassifier(void) const override { return 0; }

    virtual int isUnderlyingScalar(void) const override { return 0; }
    virtual int isUnderlyingVector(void) const override { return 1; }
    virtual int isUnderlyingAnions(void) const override { return 0; }

    virtual int isanomalyOn(void)  const override { return 0; }
    virtual int isanomalyOff(void) const override { return 1; }

    virtual double anomalyNu(void)    const override { return 0; }
    virtual int    anomalyClass(void) const override { return 0; }

    virtual int isautosetOff(void)          const override { return Q.isautosetOff();       }
    virtual int isautosetCscaled(void)      const override { return Q.isautosetCscaled();   }
    virtual int isautosetCKmean(void)       const override { return Q.isautosetCKmean();    }
    virtual int isautosetCKmedian(void)     const override { return Q.isautosetCKmedian();  }
    virtual int isautosetCNKmean(void)      const override { return Q.isautosetCNKmean();   }
    virtual int isautosetCNKmedian(void)    const override { return Q.isautosetCNKmedian(); }
    virtual int isautosetLinBiasForce(void) const override { return 0;                      }

    virtual double autosetCval(void)  const override { return Q.autosetCval()/tspaceDim(); }
    virtual double autosetnuval(void) const override { return 0;                           }

    virtual const Vector<int>                  &d          (void)      const override { return (Q.d())(interlace,retva);         }
    virtual const Vector<double>               &Cweight    (void)      const override { return (Q.Cweight())(interlace,retvb);   }
    virtual const Vector<double>               &Cweightfuzz(void)      const override { return onedvec;                          }
    virtual const Vector<double>               &epsweight  (void)      const override { return (Q.epsweight())(interlace,retvc); }
    virtual const Matrix<double>               &Gp         (void)      const override { return Q.Gp();                           }
    virtual const Vector<double>               &kerndiag   (void)      const override { return Q.kerndiag();                     }
    virtual const Vector<int>                  &alphaState (void)      const override { return xalphaState;                      }
    virtual const Vector<Vector<double> >      &zV         (void)      const override { return traintarg;                        }
    virtual const Vector<double>               &biasV      (int raw=0) const override { (void) raw; return dbiasA;               }
    virtual const Vector<Vector<double> >      &alphaV     (int raw=0) const override { (void) raw; return dalphaA;              }
    virtual const Vector<Vector<double> >      &getu       (void)      const override { return Q.getu();                         }

    virtual const Vector<double> &zV(int i) const override { return ( i >= 0 ) ? zV()(i) : ((const Vector<double> &) y(i)); }

    // Modification:

    virtual int setLinearCost(void)    override { return Q.setLinearCost();    }
    virtual int setQuadraticCost(void) override { return Q.setQuadraticCost(); }

    virtual int setC(double xC)     override { return Q.setC(xC);     }
    virtual int seteps(double xeps) override { return Q.seteps(xeps); }

    virtual int setOptActive(void) override { return Q.setOptActive(); }
    virtual int setOptSMO(void)    override { return Q.setOptSMO();    }
    virtual int setOptD2C(void)    override { return Q.setOptD2C();    }
    virtual int setOptGrad(void)   override { return Q.setOptGrad();   }

    virtual int setzerotol     (double zt)            override { return Q.setzerotol(zt);                 }
    virtual int setOpttol      (double xopttol)       override { return Q.setOpttol(xopttol);             }
    virtual int setOpttolb     (double xopttol)       override { return Q.setOpttolb(xopttol);            }
    virtual int setOpttolc     (double xopttol)       override { return Q.setOpttolc(xopttol);            }
    virtual int setOpttold     (double xopttol)       override { return Q.setOpttold(xopttol);            }
    virtual int setlr          (double xlr)           override { return Q.setlr(xlr);                     }
    virtual int setlrb         (double xlr)           override { return Q.setlrb(xlr);                    }
    virtual int setlrc         (double xlr)           override { return Q.setlrc(xlr);                    }
    virtual int setlrd         (double xlr)           override { return Q.setlrd(xlr);                    }
    virtual int setmaxitcnt    (int    xmaxitcnt)     override { return Q.setmaxitcnt(xmaxitcnt);         }
    virtual int setmaxtraintime(double xmaxtraintime) override { return Q.setmaxtraintime(xmaxtraintime); }
    virtual int settraintimeend(double xtraintimeend) override { return Q.settraintimeend(xtraintimeend); }

    virtual int setouterlr(double xouterlr)           override { return Q.setouterlr(xouterlr);           }
    virtual int setoutertol(double xoutertol)         override { return Q.setoutertol(xoutertol);         }

    virtual int randomise(double sparsity) override;

    virtual int sety(int                i, const Vector<double>          &z) override;
    virtual int sety(const Vector<int> &i, const Vector<Vector<double> > &z) override;
    virtual int sety(                      const Vector<Vector<double> > &z) override;

    virtual int autosetOff(void)            override { return Q.autosetOff();                     }
    virtual int autosetCscaled(double Cval) override { return Q.autosetCscaled(Cval*tspaceDim()); }
    virtual int autosetCKmean(void)         override { return Q.autosetCKmean();                  }
    virtual int autosetCKmedian(void)       override { return Q.autosetCKmedian();                }
    virtual int autosetCNKmean(void)        override { return Q.autosetCNKmean();                 }
    virtual int autosetCNKmedian(void)      override { return Q.autosetCNKmedian();               }

    virtual int settspaceDim(int newdim) override;
    virtual int addtspaceFeat(int i) override;
    virtual int removetspaceFeat(int i) override;

    // Kernel Modification

    virtual void prepareKernel(void) override { return; }
    virtual int resetKernel(int modind = 1, int onlyChangeRowI = -1, int updateInfo = 1) override;
    virtual int setKernel(const MercerKernel &xkernel, int modind = 1, int onlyChangeRowI = -1) override;

    virtual void fillCache(int Ns = 0, int Ne = -1) override { Q.fillCache(Ns,Ne); return; }

    // Training set control

    virtual int addTrainingVector (int i, const gentype &z, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2) override;
    virtual int qaddTrainingVector(int i, const gentype &z,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2) override;

    virtual int addTrainingVector (int i, const Vector<gentype> &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override;
    virtual int qaddTrainingVector(int i, const Vector<gentype> &z,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override;

    virtual int addTrainingVector (int i, const Vector<double>  &z, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2);
    virtual int qaddTrainingVector(int i, const Vector<double>  &z,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2);

    virtual int addTrainingVector (int i, const Vector<Vector<double> >  &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<int> &d);
    virtual int qaddTrainingVector(int i, const Vector<Vector<double> >  &z,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<int> &d);

    virtual int removeTrainingVector(int i                                      ) override { SparseVector<gentype> x; gentype y; return removeTrainingVector(i,y,x); }
    virtual int removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x) override;
    virtual int removeTrainingVector(int i, int num                             ) override { return ML_Base::removeTrainingVector(i,num); }

    virtual int sety(int                i, const gentype         &z) override;
    virtual int sety(const Vector<int> &i, const Vector<gentype> &z) override;
    virtual int sety(                      const Vector<gentype> &z) override;

    virtual int setd(int                i, int                d) override;
    virtual int setd(const Vector<int> &i, const Vector<int> &d) override;
    virtual int setd(                      const Vector<int> &d) override;

    virtual int setCweight(int                i, double                xCweight) override;
    virtual int setCweight(const Vector<int> &i, const Vector<double> &xCweight) override;
    virtual int setCweight(                      const Vector<double> &xCweight) override;

    virtual int setCweightfuzz(int                i, double                xCweight) override;
    virtual int setCweightfuzz(const Vector<int> &i, const Vector<double> &xCweight) override;
    virtual int setCweightfuzz(                      const Vector<double> &xCweight) override;

    virtual int setepsweight(int                i, double                xepsweight) override;
    virtual int setepsweight(const Vector<int> &i, const Vector<double> &xepsweight) override;
    virtual int setepsweight(                      const Vector<double> &xepsweight) override;

    virtual int scaleCweight(double scalefactor)     override { return Q.scaleCweight(scalefactor);     }
    virtual int scaleCweightfuzz(double scalefactor) override { return Q.scaleCweightfuzz(scalefactor); }
    virtual int scaleepsweight(double scalefactor)   override { return Q.scaleepsweight(scalefactor);   }

    // Train the SVM

    virtual void fudgeOn(void)  override { Q.fudgeOn();  return; }
    virtual void fudgeOff(void) override { Q.fudgeOff(); return; }

    virtual int train(int &res, svmvolatile int &killSwitch) override;
    virtual int train(int &res) override { svmvolatile int killSwitch = 0; return train(res,killSwitch); }

    // Evaluation:

    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = nullptr) const override;

    virtual double eTrainingVector(int i) const override;

    virtual double         &dedgTrainingVector(double         &res, int i) const override { return ML_Base::dedgTrainingVector(res,i); }
    virtual Vector<double> &dedgTrainingVector(Vector<double> &res, int i) const override;
    virtual d_anion        &dedgTrainingVector(d_anion        &res, int i) const override { return ML_Base::dedgTrainingVector(res,i); }
    virtual gentype        &dedgTrainingVector(gentype        &res, int i) const override { Vector<double> tmp(tspaceDim()); return ( res = dedgTrainingVector(tmp,i) ); }

    virtual double &d2edg2TrainingVector(double &res, int i) const override { return ML_Base::d2edg2TrainingVector(res,i); }

    virtual double dedKTrainingVector(int i, int j) const override { Vector<double> tmp(tspaceDim()); double res; return innerProduct(res,dedgTrainingVector(tmp,i),alphaV()(j)); }
    virtual Vector<double> &dedKTrainingVector(Vector<double> &res, int i) const override;
    virtual Matrix<double> &dedKTrainingVector(Matrix<double> &res) const override;

    // Other functions

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0) override;
    virtual void semicopy(const ML_Base &src) override;
    virtual void qswapinternal(ML_Base &b) override;

    // Training set control:

    virtual int setFixedBias(const Vector<double> &newbias);
    virtual int setFixedBias(int q, double newbias)  override { return SVM_Generic::setFixedBias(q,newbias); }
    virtual int setFixedBias(const gentype &newbias) override { return SVM_Generic::setFixedBias(  newbias); }
    virtual int setFixedBias(double newbias)         override { return SVM_Generic::setFixedBias(  newbias); }

private:

    virtual int gTrainingVector(Vector<double> &gproject, int &dummy, int i, int raw = 0, gentype ***pxyprodi = nullptr) const;

    int aN;
    int aNS;
    int aNZ;
    int aNF;
    int aNC;

    Vector<int> interlace; // [ 0 m 2m ... ] (m=tspaceDim)

    Vector<Vector<double> > traintarg;
    Vector<int> xalphaState;
    Vector<double> onedvec;
    Vector<Vector<double> > dalphaA;
    Vector<double> dbiasA;
    Matrix<double> Gpn;

    SVM_Scalar Q;

    scalar_callback Kcall;

    int ixsplit;
    int iqsplit;
    int ixskip;
    int ixskipc;

    mutable retVector<int> retva;
    mutable retVector<double> retvb;
    mutable retVector<double> retvc;

    void updateBias(void);
    void updateAlpha(void);
    void fixKcallback(void);
    int qtaddTrainingVector(int i, const Vector<double>  &z, double Cweigh = 1, double epsweigh = 1, int d = 2);
    void locnaivesetGpnExt(void);
    int inintrain(int &res, svmvolatile int &killSwitch);
    int intrain(int &res, svmvolatile int &killSwitch);
};

inline double norm2(const SVM_Vector_Mredbin &a);
inline double abs2 (const SVM_Vector_Mredbin &a);

inline double norm2(const SVM_Vector_Mredbin &a) { return a.RKHSnorm(); }
inline double abs2 (const SVM_Vector_Mredbin &a) { return a.RKHSabs();  }

inline void qswap(SVM_Vector_Mredbin &a, SVM_Vector_Mredbin &b)
{
    a.qswapinternal(b);

    return;
}

inline void SVM_Vector_Mredbin::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    SVM_Vector_Mredbin &b = dynamic_cast<SVM_Vector_Mredbin &>(bb.getML());

    SVM_Generic::qswapinternal(b);

    qswap(aN         ,b.aN         );
    qswap(aNS        ,b.aNS        );
    qswap(aNZ        ,b.aNZ        );
    qswap(aNF        ,b.aNF        );
    qswap(aNC        ,b.aNC        );
    qswap(interlace  ,b.interlace  );
    qswap(traintarg  ,b.traintarg  );
    qswap(xalphaState,b.xalphaState);
    qswap(onedvec    ,b.onedvec    );
    qswap(dalphaA    ,b.dalphaA    );
    qswap(dbiasA     ,b.dbiasA     );
    qswap(Gpn        ,b.Gpn        );
    qswap(Q          ,b.Q          );
    qswap(ixsplit    ,b.ixsplit    );
    qswap(iqsplit    ,b.iqsplit    );
    qswap(ixskip     ,b.ixskip     );
    qswap(ixskipc    ,b.ixskipc    );

    locnaivesetGpnExt();
    b.locnaivesetGpnExt();

    return;
}

inline void SVM_Vector_Mredbin::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const SVM_Vector_Mredbin &b = dynamic_cast<const SVM_Vector_Mredbin &>(bb.getMLconst());

    SVM_Generic::semicopy(b);

    //interlace
    //Gpn

    traintarg = b.traintarg;

    ixsplit = b.ixsplit;
    iqsplit = b.iqsplit;
    ixskip  = b.ixskip;
    ixskipc = b.ixskipc;

    aN  = b.aN;
    aNS = b.aNS;
    aNZ = b.aNZ;
    aNF = b.aNF;
    aNC = b.aNC;

    xalphaState = b.xalphaState;
    onedvec     = b.onedvec;
    dalphaA     = b.dalphaA;
    dbiasA      = b.dbiasA;

    Q.semicopy(b.Q);

    return;
}

inline void SVM_Vector_Mredbin::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const SVM_Vector_Mredbin &src = dynamic_cast<const SVM_Vector_Mredbin &>(bb.getMLconst());

    aN  = src.aN;
    aNS = src.aNS;
    aNZ = src.aNZ;
    aNF = src.aNF;
    aNC = src.aNC;

    interlace = src.interlace;

    SVM_Generic::assign(src,onlySemiCopy);

    traintarg   = src.traintarg;
    xalphaState = src.xalphaState;
    onedvec     = src.onedvec;
    dalphaA     = src.dalphaA;
    dbiasA      = src.dbiasA;
    Gpn         = src.Gpn;

    Q.assign(src.Q,onlySemiCopy);

    locnaivesetGpnExt();

    ixsplit = src.ixsplit;
    iqsplit = src.iqsplit;
    ixskip  = src.ixskip;
    ixskipc = src.ixskipc;

    fixKcallback();

    return;
}

#endif
