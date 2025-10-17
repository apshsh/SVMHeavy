
//
// Multiclass classification SVM (reduction to binary)
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _svm_multic_redbin_h
#define _svm_multic_redbin_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "svm_generic.hpp"
#include "svm_binary.hpp"
#include "svm_single.hpp"
#include "kcache.hpp"
#include "idstore.hpp"

void evalKSVM_MultiC_redbin(double &res, int i, int j, const gentype **pxyprod, const void *owner);
void evalXYSVM_MultiC_redbin(double &res, int i, int j, const gentype **pxyprod, const void *owner);
void evalSigmaSVM_MultiC_redbin(double &res, int i, int j, const gentype **pxyprod, const void *owner);



class SVM_MultiC_redbin;
class SVM_MultiC;

// Swap function

inline void qswap(SVM_MultiC_redbin &a, SVM_MultiC_redbin &b);


class SVM_MultiC_redbin : public SVM_Generic
{
    friend void evalKSVM_MultiC_redbin(double &res, int i, int j, const gentype **pxyprod, const void *owner);
    friend void evalXYSVM_MultiC_redbin(double &res, int i, int j, const gentype **pxyprod, const void *owner);
    friend void evalsigmaSVM_MultiC_redbin(double &res, int i, int j, const gentype **pxyprod, const void *owner);
    friend class SVM_MultiC;

public:

    // Constructors, destructors, assignment operators and similar

    SVM_MultiC_redbin();
    SVM_MultiC_redbin(const SVM_MultiC_redbin &src);
    SVM_MultiC_redbin(const SVM_MultiC_redbin &src, const ML_Base *xsrc);
    SVM_MultiC_redbin &operator=(const SVM_MultiC_redbin &src) { assign(src); return *this; }
    virtual ~SVM_MultiC_redbin();

    virtual int prealloc(int expectedN) override;
    virtual int preallocsize(void) const override;
    virtual void setmemsize(int memsize) override;

    virtual int scale(double a) override;
    virtual int reset(void) override;
    virtual int restart(void) override { SVM_MultiC_redbin temp; *this = temp; return 1; }

    virtual int setAlphaV(const Vector<Vector<double> > &newAlpha) override;
    virtual int setBiasV(const Vector<double> &newBias) override;

    virtual std::ostream &printstream(std::ostream &output, int dep) const override;
    virtual std::istream &inputstream(std::istream &input) override;

    // Information (use q = -3 to find information about anomaly class, q == -2 for combined info if available):

    virtual int isTrained(void) const override { return isStateOpt; }

    virtual int N  (void)  const override { return trainclass.size();                                                                   }
    virtual int NS (void)  const override { return NS(-2);                                                                              }
    virtual int NZ (void)  const override { return NZ(-2);                                                                              }
    virtual int NF (void)  const override { return NF(0);                                                                               }
    virtual int NC (void)  const override { return NC(0);                                                                               }
    virtual int NLB(void)  const override { return NLB(0);                                                                              }
    virtual int NLF(void)  const override { return NLF(0);                                                                              }
    virtual int NUF(void)  const override { return NUF(0);                                                                              }
    virtual int NUB(void)  const override { return NUB(0);                                                                              }
    virtual int NNC(int d) const override { return ( d == anomalyd ) ? QA.NNC(1) : ( d ? Nnc(label_placeholder.findID(d)+1) : Nnc(0) ); }
    virtual int NS (int q) const override { return ( q == -3 ) ? QA.NS()  : ( ( q == -2 ) ? Ns : (NF(q)+NLB(q)+NUB(q)) );               }
    virtual int NZ (int q) const override { return ( q == -3 ) ? QA.NZ()  : ( ( q == -2 ) ? N()-NS() : Q(q).NZ() );                     }
    virtual int NF (int q) const override { return ( q == -3 ) ? QA.NF()  : Q(q).NF();                                                  }
    virtual int NC (int q) const override { return ( q == -3 ) ? QA.NC()  : Q(q).NC();                                                  }
    virtual int NLB(int q) const override { return ( q == -3 ) ? QA.NLB() : Q(q).NLB();                                                 }
    virtual int NLF(int q) const override { return ( q == -3 ) ? QA.NLF() : Q(q).NLF();                                                 }
    virtual int NUF(int q) const override { return ( q == -3 ) ? QA.NUF() : Q(q).NUF();                                                 }
    virtual int NUB(int q) const override { return ( q == -3 ) ? QA.NUB() : Q(q).NUB();                                                 }

    virtual int tspaceDim(void)  const override { return db.size();                }
    virtual int numClasses(void) const override { return label_placeholder.size(); }
    virtual int type(void)       const override { return 3;                        }
    virtual int subtype(void)    const override { return 1;                        }

    virtual char gOutType(void) const override { return 'V'; }
    virtual char hOutType(void) const override { return 'Z'; }
    virtual char targType(void) const override { return 'Z'; }
    virtual double calcDist(const gentype &ha, const gentype &hb, int ia = -1, int db = 2) const override;

    virtual const Vector<int>          &ClassLabels(void)   const override { return label_placeholder.getreftoID(); }
    virtual const Vector<Vector<int> > &ClassRep(void)      const override { return classRepval;                    }
    virtual int                         findID(int ref)     const override { return label_placeholder.findID(ref);  }

    virtual int isLinearCost(void)      const override { return costType == 0;      }
    virtual int isQuadraticCost(void)   const override { return costType == 1;      }
    virtual int is1NormCost(void)       const override { return 0;                  }
    virtual int isVarBias(void)         const override { return isVarBias(0);       }
    virtual int isPosBias(void)         const override { return isPosBias(0);       }
    virtual int isNegBias(void)         const override { return isNegBias(0);       }
    virtual int isFixedBias(void)       const override { return isFixedBias(0);     }
    virtual int isVarBias(int q)        const override { return Q(q).isVarBias();   }
    virtual int isPosBias(int q)        const override { return Q(q).isPosBias();   }
    virtual int isNegBias(int q)        const override { return Q(q).isNegBias();   }
    virtual int isFixedBias(int q)      const override { return Q(q).isFixedBias(); }

    virtual int isOptActive(void) const override { return optType == 0; }
    virtual int isOptSMO(void)    const override { return optType == 1; }
    virtual int isOptD2C(void)    const override { return optType == 2; }
    virtual int isOptGrad(void)   const override { return optType == 3; }

    virtual int m(void) const override { return 2; }

    virtual int NRff (void) const override { return 0; }

    virtual double C(void)         const override { return CNval;                                                            }
    virtual double eps(void)       const override { return epsval;                                                           }
    virtual double Cclass(int d)   const override { return ( d == anomalyd ) ? 1 : mulCclass(label_placeholder.findID(d));   }
    virtual double epsclass(int d) const override { return ( d == anomalyd ) ? 1 : mulepsclass(label_placeholder.findID(d)); }

    virtual int    memsize     (void) const override { return kerncache.get_memsize(); }
    virtual double zerotol     (void) const override { return Q(0).zerotol();          }
    virtual double Opttol      (void) const override { return Q(0).Opttol();           }
    virtual double Opttolb     (void) const override { return Q(0).Opttolb();          }
    virtual double Opttolc     (void) const override { return Q(0).Opttolc();          }
    virtual double Opttold     (void) const override { return Q(0).Opttold();          }
    virtual double lr          (void) const override { return Q(0).lr();               }
    virtual double lrb         (void) const override { return Q(0).lrb();              }
    virtual double lrc         (void) const override { return Q(0).lrc();              }
    virtual double lrd         (void) const override { return Q(0).lrd();              }
    virtual int    maxitcnt    (void) const override { return Q(0).maxitcnt();         }
    virtual double maxtraintime(void) const override { return Q(0).maxtraintime();     }
    virtual double traintimeend(void) const override { return Q(0).traintimeend();     }

    virtual double outerlr(void)      const override { return MULTINORM_OUTERSTEP;  }
    virtual double outertol(void)     const override { return MULTINORM_OUTERACCUR; }

    virtual       int      maxiterfuzzt(void) const override { return Q(0).maxiterfuzzt(); }
    virtual       int      usefuzzt(void)     const override { return Q(0).usefuzzt();     }
    virtual       double   lrfuzzt(void)      const override { return Q(0).lrfuzzt();      }
    virtual       double   ztfuzzt(void)      const override { return Q(0).ztfuzzt();      }
    virtual const gentype &costfnfuzzt(void)  const override { return Q(0).costfnfuzzt();  }

    virtual double LinBiasForce(void)        const override { return LinBiasForceclass(0); }
    virtual double QuadBiasForce(void)       const override { return QuadBiasForceclass(0);}
    virtual double LinBiasForceclass(int q)  const override { return Q(q).LinBiasForce();  }
    virtual double QuadBiasForceclass(int q) const override { return Q(q).QuadBiasForce(); }

    virtual int isFixedTube(void)  const override { return Q(0).isFixedTube();  }
    virtual int isShrinkTube(void) const override { return Q(0).isShrinkTube(); }

    virtual int isRestrictEpsPos(void) const override { return 0; }
    virtual int isRestrictEpsNeg(void) const override { return 0; }

    virtual double nu(void)     const override { return Q(0).nu(); }
    virtual double nuQuad(void) const override { return 0;            }

    virtual int isClassifyViaSVR(void) const override { return Q(0).isClassifyViaSVR(); }
    virtual int isClassifyViaSVM(void) const override { return Q(0).isClassifyViaSVM(); }

    virtual int is1vsA(void)    const override { return multitype == 0; }
    virtual int is1vs1(void)    const override { return multitype == 1; }
    virtual int isDAGSVM(void)  const override { return multitype == 2; }
    virtual int isMOC(void)     const override { return multitype == 3; }
    virtual int ismaxwins(void) const override { return 0;              }
    virtual int isrecdiv(void)  const override { return 0;              }

    virtual int isatonce(void) const override { return 0; }
    virtual int isredbin(void) const override { return 1; }

    virtual int isKreal(void)   const override { return 0; }
    virtual int isKunreal(void) const override { return 0; }

    virtual int isUnderlyingScalar(void) const override { return 0; }
    virtual int isUnderlyingVector(void) const override { return 1; }
    virtual int isUnderlyingAnions(void) const override { return 0; }

    virtual int isanomalyOn(void)  const override { return anomalyd != -3;    }
    virtual int isanomalyOff(void) const override { return anomalyd == -3;    }

    virtual double anomalyNu(void)    const override { return QA.autosetnuval(); }
    virtual int    anomalyClass(void) const override { return anomalyd;          }

    virtual int isautosetOff(void)          const override { return autosetLevel == 0; }
    virtual int isautosetCscaled(void)      const override { return autosetLevel == 1; }
    virtual int isautosetCKmean(void)       const override { return autosetLevel == 2; }
    virtual int isautosetCKmedian(void)     const override { return autosetLevel == 3; }
    virtual int isautosetCNKmean(void)      const override { return autosetLevel == 4; }
    virtual int isautosetCNKmedian(void)    const override { return autosetLevel == 5; }
    virtual int isautosetLinBiasForce(void) const override { return 0;                 }

    virtual double autosetCval(void)  const override { return autosetCvalx;  }
    virtual double autosetnuval(void) const override { return autosetnuvalx; }

    virtual const Vector<int>                  &d          (void)      const override { return trainclass;                                  }
    virtual const Vector<double>               &Cweight    (void)      const override { return Cweightval;                                  }
    virtual const Vector<double>               &Cweightfuzz(void)      const override { return Cweightvalfuzz;                              }
    virtual const Vector<double>               &epsweight  (void)      const override { return epsweightval;                                }
    virtual const Matrix<double>               &Gp         (void)      const override { return *Gpval;                                      }
    virtual const Vector<double>               &kerndiag   (void)      const override { return xxkerndiag;                                  }
    virtual const Vector<int>                  &alphaState (void)      const override { return dalphaState;                                 }
    virtual const Vector<double>               &biasV      (int raw=0) const override { (void) raw; return db;                              }
    virtual const Vector<Vector<double> >      &alphaV     (int raw=0) const override { (void) raw; return dalpha;                          }
    virtual const Vector<Vector<double> >      &getu       (void)      const override { return notu;                                        }

    virtual int isClassifier(void) const override { return 1; }
    virtual int singmethod(void) const override { return QA.singmethod(); }
    virtual double rejectThreshold(void) const override { return Q(0).rejectThreshold(); }

    // Modification:

    virtual int setLinearCost(void) override;
    virtual int setQuadraticCost(void) override;

    virtual int setVarBias(void)             override { return setVarBias(-2);           }
    virtual int setPosBias(void)             override { return setPosBias(-2);           }
    virtual int setNegBias(void)             override { return setNegBias(-2);           }
    virtual int setFixedBias(double newbias) override { return setFixedBias(-2,newbias); }
    virtual int setVarBias(int q)            override;
    virtual int setPosBias(int q)            override;
    virtual int setNegBias(int q)            override;
    virtual int setFixedBias(int q, double newbias) override;

    virtual int setC(double xC) override;
    virtual int seteps(double xeps) override;
    virtual int setCclass(int d, double xC) override;
    virtual int setepsclass(int d, double xeps) override;

    virtual int setOptActive(void) override;
    virtual int setOptSMO(void) override;
    virtual int setOptD2C(void) override;
    virtual int setOptGrad(void) override;

    virtual int setzerotol     (double zt)            override;
    virtual int setOpttol      (double xopttol)       override;
    virtual int setOpttolb     (double xopttol)       override { (void) xopttol; return 0; }
    virtual int setOpttolc     (double xopttol)       override { (void) xopttol; return 0; }
    virtual int setOpttold     (double xopttol)       override { (void) xopttol; return 0; }
    virtual int setlr          (double xlr)           override { (void) xlr; return 0; }
    virtual int setlrb         (double xlr)           override { (void) xlr; return 0; }
    virtual int setlrc         (double xlr)           override { (void) xlr; return 0; }
    virtual int setlrd         (double xlr)           override { (void) xlr; return 0; }
    virtual int setmaxitcnt    (int    xmaxitcnt)     override;
    virtual int setmaxtraintime(double xmaxtraintime) override;
    virtual int settraintimeend(double xtraintimeend) override;

    virtual int setouterlr(double xouterlr)   override { (void) xouterlr;  return 0; }
    virtual int setoutertol(double xoutertol) override { (void) xoutertol; return 0; }

    virtual int setmaxiterfuzzt(int xmaxiterfuzzt) override;
    virtual int setusefuzzt(int xusefuzzt) override;
    virtual int setlrfuzzt(double xlrfuzzt) override;
    virtual int setztfuzzt(double xztfuzzt) override;
    virtual int setcostfnfuzzt(const gentype &xcostfnfuzzt) override;
    virtual int setcostfnfuzzt(const std::string &xcostfnfuzzt) override;

    virtual int setLinBiasForce      (       double newval) override { return setLinBiasForceclass(0,newval);  }
    virtual int setQuadBiasForce     (       double newval) override { return setQuadBiasForceclass(0,newval); }
    virtual int setLinBiasForceclass (int q, double newval) override;
    virtual int setQuadBiasForceclass(int q, double newval) override;

    virtual int setFixedTube(void) override;
    virtual int setShrinkTube(void) override;

    virtual int setnu(double xnu) override;

    virtual int setClassifyViaSVR(void) override;
    virtual int setClassifyViaSVM(void) override;

    virtual int set1vsA(void) override;
    virtual int set1vs1(void) override;
    virtual int setDAGSVM(void) override;
    virtual int setMOC(void) override;

    virtual int anomalyOn(int danomalyClass, double danomalyNu) override;
    virtual int anomalyOff(void) override;

    virtual int randomise(double sparsity) override;

    virtual int autosetOff(void)            override { autosetLevel = 0; return 0; }
    virtual int autosetCscaled(double Cval) override { return autosetCscaled(Cval,0); }
    virtual int autosetCKmean(void)         override { return autosetCKmean(0);       }
    virtual int autosetCKmedian(void)       override { return autosetCKmedian(0);     }
    virtual int autosetCNKmean(void)        override { return autosetCNKmean(0);      }
    virtual int autosetCNKmedian(void)      override { return autosetCNKmedian(0);    }

    virtual int addclass(int label, int epszero = 0) override;
    virtual void setsingmethod(int nv) override { QA.setsingmethod(nv); return; }
    virtual void setRejectThreshold(double nv) override;

    // Kernel Modification

    virtual void prepareKernel(void) override { return; }
    virtual int resetKernel(int modind = 1, int onlyChangeRowI = -1, int updateInfo = 1) override;
    virtual int setKernel(const MercerKernel &xkernel, int modind = 1, int onlyChangeRowI = -1) override;

    virtual void fillCache(int Ns = 0, int Ne = -1) override;

    // Training set control

    virtual int addTrainingVector (int i, const gentype &z, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2) override;
    virtual int qaddTrainingVector(int i, const gentype &z,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2) override;

    virtual int addTrainingVector (int i, const Vector<gentype> &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override;
    virtual int qaddTrainingVector(int i, const Vector<gentype> &z,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override;

    virtual int addTrainingVector (int i, int d, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);
    virtual int qaddTrainingVector(int i, int d,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);

    virtual int addTrainingVector (int i, const Vector<int> &d, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);
    virtual int qaddTrainingVector(int i, const Vector<int> &d,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);

    virtual int removeTrainingVector(int i                                      ) override { SparseVector<gentype> x; gentype y; return removeTrainingVector(i,y,x); }
    virtual int removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x) override;
    virtual int removeTrainingVector(int i, int num                             ) override { return ML_Base::removeTrainingVector(i,num); }

    virtual int sety(int                i, const gentype         &y) override;
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

    virtual int scaleCweight(double scalefactor) override;
    virtual int scaleCweightfuzz(double scalefactor) override;
    virtual int scaleepsweight(double scalefactor) override;

    // Train the SVM

    virtual void fudgeOn(void) override;
    virtual void fudgeOff(void) override;

    virtual int train(int &res)                              override { svmvolatile int killSwitch = 0; return train(res,killSwitch); }
    virtual int train(int &res, svmvolatile int &killSwitch) override;

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

    virtual void setaltx(const ML_Base *_altxsrc) override { QA.setaltx(_altxsrc); ML_Base::setaltx(_altxsrc); return; }

private:

    virtual int gTrainingVector(Vector<double> &gproject, int &locclassrep, int i, int raw = 0, gentype ***pxyprodi = nullptr) const;
    virtual int gTrainingVector(Vector<gentype> &gproject, int &locclassrep, int i, int raw = 0, gentype ***pxyprodi = nullptr) const;

    int costType;            // 0 = linear, 1 = LS
    int multitype;           // 0 = 1vsA, 1 = 1vs1, 2 = DAGSVM, 3 = MOC
    int optType;             // 0 = active set, 1 = SMO, 2 D2C, 3 grad

    double CNval;               // C (tradeoff) value
    double epsval;              // eps (tube) value
    Vector<double> mulCclass;   // stored classlabel-wise C weights (0 = -1, 1 = zero, 2 = +1, 3 = free)
    Vector<double> mulepsclass; // stored classlabel-wise eps weights

    Kcache<double> xycache;           // xy cache
    Kcache<double> kerncache;         // kernel cache
    Kcache<double> sigmacache;        // sigma cache
    Vector<double> xxkerndiag;        // kernel diagonals
    Vector<double> diagoff;           // diagonal offset for hessian (used by quadratic cost)

    int autosetLevel;     // 0 = none, 1 = C/N, 2 = Cmean, 3 = Cmedian, 4 = CNmean, 5 = CNmedian
    double autosetnuvalx; // Cval used if autosetLevel == 1
    double autosetCvalx;  // Cval used if autosetLevel == 1

    // Optimisation state
    //
    // label_placeholder: The user defines class in terms of labels.  They may be any int.  These are translated into classlabels, namely 0,1,2,..., by label_placeholders.
    // classRep: Each class has a corresponding ternary representation (eg class 2 might be -1,+1,0,-1, where 0 is "don't care").  These are stored here.
    //           The number of classes is given by classRep.size()
    // Ns: number of support vectors

    Vector<SVM_Binary> Q;           // stored classrep-wise.
    Vector<Vector<double> > dalpha; // stored classrep-wise
    Vector<int> dalphaState;
    Vector<double> db;              // stored classrep-wise
    IDStore label_placeholder;
    Vector<Vector<int> > classRepval;  // stored classlabel-wise.
    Vector<Vector<double> > notu;   // irrelevant placeholder
    int Ns;                         // number of support vectors
    Vector<int> Nnc;                // number of vector not constrained at zero
    int isStateOpt;                 // set if SVM is in optimal state

    // Anomaly detection (if used)

    int anomalyd;
    SVM_Single QA;

    // Training data

//    Vector<SparseVector<gentype> > traindata;
    Vector<int> trainclass;
    Vector<double> Cweightval;
    Vector<double> Cweightvalfuzz;
    Vector<double> epsweightval;

    // Quadratic program definition

    Matrix<double> *xyval;
    Matrix<double> *Gpval;
    Matrix<double> *Gpsigma;

    // label: class numbers given by the user, which are any non-consecutive set of integers
    // classlabel: labels translated to 0,1,2,...
    // classrep: underlying SVMs, which are classes expanded using classRep.

    // Internal functions
    //
    // locsetGp: call setGp for all binary SVMs Q
    // changemultitype: change multitype, and make necessary adjustments
    // classify: place gproject into appropriate class
    // recalcdiagoff(i): recalculate and update diagoff(i), or all of diagoff if i == -1

    void locsetGp(int refactsol = 1);
    void changemultitype(int newmultitype);
    int classify(int &locclassrep, const Vector<double> &gproject) const;
    void recalcdiagoff(int ival);

    int fixautosettings(int kernchange, int Nchange, int ncut = 0); // ncut is subtracted from N
    int setdinternal(int i, int d); // like setd, but without fixing auto settings
    double autosetkerndiagmean(void);
    double autosetkerndiagmedian(void);

    int qtaddTrainingVector(int i, int d, double Cweigh = 1, double epsweigh = 1);

    // ncut versions

    virtual int autosetCscaled(double Cval, int ncut) { NiceAssert( Cval > 0 ); autosetCvalx = Cval; int res = setC( (N()-NNC(0)-ncut) ? (Cval/((N()-NNC(0)-ncut))) : 1.0); autosetLevel = 1; return res; }
    virtual int autosetCKmean(int ncut)               { double diagsum = ( (N()-NNC(0)-ncut) ? autosetkerndiagmean()                     : 1   ); int res = setC( ( abs2(diagsum) > zerotol() ) ? (1/diagsum) : 1 ); autosetLevel = 2; return res; }
    virtual int autosetCKmedian(int ncut)             { double diagsum = ( (N()-NNC(0)-ncut) ? autosetkerndiagmedian()                   : 1   ); int res = setC( ( abs2(diagsum) > zerotol() ) ? (1/diagsum) : 1 ); autosetLevel = 3; return res; }
    virtual int autosetCNKmean(int ncut)              { double diagsum = ( (N()-NNC(0)-ncut) ? (N()-NNC(0)-ncut)*autosetkerndiagmean()   : 1   ); int res = setC( ( abs2(diagsum) > zerotol() ) ? (1/diagsum) : 1 ); autosetLevel = 4; return res; }
    virtual int autosetCNKmedian(int ncut)            { double diagsum = ( (N()-NNC(0)-ncut) ? (N()-NNC(0)-ncut)*autosetkerndiagmedian() : 1   ); int res = setC( ( abs2(diagsum) > zerotol() ) ? (1/diagsum) : 1 ); autosetLevel = 5; return res; }
    virtual int autosetLinBiasForce(double nuval, double Cval, int ncut)
    {
        NiceAssert( ( Cval > 0 ) && ( nuval >= 0.0 ) && ( nuval <= 1.0 ) );

        int res = QA.setC(( (N()-NNC(0)-ncut) ) ? ((1.0/nuval)*(Cval/((double) ((N()-NNC(0)-ncut))))) : 1.0);
        res |= QA.seteps(0.0); QA.setLinBiasForce(Cval);
        autosetLevel = 6;

        return res;
    }
};

// 1vsa:     for each Q, Cclass and epsclass are set to 1 and Cweight, epsweight are set instead.
// 1vs1:     for each Q, Cclass and epsclass are set to 1 and Cweight, epsweight are set instead.  All vectors are added to all Qs, but only relevant ones have d != 0
// MOC:      for each Q, Cclass and epsclass are set to 1 and Cweight, epsweight are set instead.  All vectors are added to all Qs, but only relevant ones have d != 0
//
// 1vs1: has n(n-1)/2 elements in Q, namely: 1vs0, 2vs0, 2vs1, 3vs0, 3vs1, 3vs2, ..., (n-1)vs0, (n-1)vs1, ..., (n-1)vs(n-2)  (+vs-)
// MOC:  has ceil(log2(n)) elements in Q with least significant bit (LSB) first.

inline double norm2(const SVM_MultiC_redbin &a);
inline double abs2 (const SVM_MultiC_redbin &a);

inline double norm2(const SVM_MultiC_redbin &a) { return a.RKHSnorm(); }
inline double abs2 (const SVM_MultiC_redbin &a) { return a.RKHSabs();  }

inline void qswap(SVM_MultiC_redbin &a, SVM_MultiC_redbin &b)
{
    a.qswapinternal(b);

    return;
}

inline void SVM_MultiC_redbin::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    SVM_MultiC_redbin &b = dynamic_cast<SVM_MultiC_redbin &>(bb.getML());

    SVM_Generic::qswapinternal(b);

    qswap(costType         ,b.costType         );
    qswap(multitype        ,b.multitype        );
    qswap(optType          ,b.optType          );
    qswap(CNval            ,b.CNval            );
    qswap(epsval           ,b.epsval           );
    qswap(mulCclass        ,b.mulCclass        );
    qswap(mulepsclass      ,b.mulepsclass      );
    qswap(xycache          ,b.xycache          );
    qswap(kerncache        ,b.kerncache        );
    qswap(sigmacache       ,b.sigmacache       );
    qswap(xxkerndiag       ,b.xxkerndiag       );
    qswap(diagoff          ,b.diagoff          );
    qswap(autosetLevel     ,b.autosetLevel     );
    qswap(autosetnuvalx    ,b.autosetnuvalx    );
    qswap(autosetCvalx     ,b.autosetCvalx     );
    qswap(Q                ,b.Q                );
    qswap(anomalyd         ,b.anomalyd         );
    qswap(QA               ,b.QA               );
    qswap(dalpha           ,b.dalpha           );
    qswap(dalphaState      ,b.dalphaState      );
    qswap(db               ,b.db               );
    qswap(label_placeholder,b.label_placeholder);
    qswap(classRepval      ,b.classRepval      );
    qswap(Ns               ,b.Ns               );
    qswap(Nnc              ,b.Nnc              );
    qswap(isStateOpt       ,b.isStateOpt       );
    qswap(trainclass       ,b.trainclass       );
    qswap(Cweightval       ,b.Cweightval       );
    qswap(Cweightvalfuzz   ,b.Cweightvalfuzz   );
    qswap(epsweightval     ,b.epsweightval     );

    Matrix<double> *txy;
    Matrix<double> *tGp;
    Matrix<double> *tGpsigma;

    txy      = xyval;   xyval   = b.xyval;   b.xyval   = txy;
    tGp      = Gpval;   Gpval   = b.Gpval;   b.Gpval   = tGp;
    tGpsigma = Gpsigma; Gpsigma = b.Gpsigma; b.Gpsigma = tGpsigma;

    // The kernel (and sigma) cache, as well as Gp (and Gpsigma) will have
    // been messed around by the above switching.  We need to make sure that
    // their pointers are set to rights before we continue.

    (xycache).cheatSetEvalArg((void *) this);
    (kerncache).cheatSetEvalArg((void *) this);
    (sigmacache).cheatSetEvalArg((void *) this);

    (xyval)->cheatsetcdref((void *) &(xycache));
    (Gpval)->cheatsetcdref((void *) &(kerncache));
    (Gpsigma)->cheatsetcdref((void *) &(sigmacache));

    (b.xycache).cheatSetEvalArg((void *) &b);
    (b.kerncache).cheatSetEvalArg((void *) &b);
    (b.sigmacache).cheatSetEvalArg((void *) &b);

    (b.xyval)->cheatsetcdref((void *) &(b.xycache));
    (b.Gpval)->cheatsetcdref((void *) &(b.kerncache));
    (b.Gpsigma)->cheatsetcdref((void *) &(b.sigmacache));

    return;
}

inline void SVM_MultiC_redbin::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const SVM_MultiC_redbin &b = dynamic_cast<const SVM_MultiC_redbin &>(bb.getMLconst());

    SVM_Generic::semicopy(b);

    int q;

    //kerncache;
    //sigmacache;

    //Gpval;
    //Gpsigma;

    //label_placeholder;
    //classRepval;
    //notu;

    costType  = b.costType;
    multitype = b.multitype;
    optType   = b.optType;

    mulCclass   = b.mulCclass;
    mulepsclass = b.mulepsclass;

    autosetLevel  = b.autosetLevel;
    autosetnuvalx = b.autosetnuvalx;
    autosetCvalx  = b.autosetCvalx;

    anomalyd = b.anomalyd;

    Cweightval     = b.Cweightval;
    Cweightvalfuzz = b.Cweightvalfuzz;
    epsweightval   = b.epsweightval;

    isStateOpt = b.isStateOpt;

    CNval       = b.CNval;
    epsval      = b.epsval;

    xxkerndiag = b.xxkerndiag;
    diagoff    = b.diagoff;

    dalpha      = b.dalpha;
    dalphaState = b.dalphaState;
    db          = b.db;
    Ns          = b.Ns;
    Nnc         = b.Nnc;

    trainclass = b.trainclass;

    xycache.recalcDiag();
    kerncache.recalcDiag();
    sigmacache.recalcDiag();

    QA.semicopy(b.QA);
    Q.resize(b.Q.size()); // as Q might never have been set

    for ( q = 0 ; q < Q.size() ; ++q )
    {
        Q("&",q).semicopy((b.Q)(q));
    }

    return;
}

inline void SVM_MultiC_redbin::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const SVM_MultiC_redbin &src = dynamic_cast<const SVM_MultiC_redbin &>(bb.getMLconst());

    isStateOpt = src.isStateOpt;

    costType  = src.costType;
    multitype = src.multitype;
    optType   = src.optType;

    CNval       = src.CNval;
    epsval      = src.epsval;
    mulCclass   = src.mulCclass;
    mulepsclass = src.mulepsclass;

    xycache    = src.xycache;
    kerncache  = src.kerncache;
    sigmacache = src.sigmacache;
    xxkerndiag = src.xxkerndiag;
    diagoff    = src.diagoff;

    autosetLevel  = src.autosetLevel;
    autosetnuvalx = src.autosetnuvalx;
    autosetCvalx  = src.autosetCvalx;

    int i;

    Q.resize(src.Q.size());

    for ( i = 0 ; i < Q.size() ; ++i )
    {
        Q("&",i).assign((src.Q)(i),onlySemiCopy);
    }

    dalpha      = src.dalpha;
    dalphaState = src.dalphaState;
    db          = src.db;

    anomalyd = src.anomalyd;
    QA.assign(src.QA,onlySemiCopy);

    label_placeholder = src.label_placeholder;
    classRepval       = src.classRepval;

    Ns  = src.Ns;
    Nnc = src.Nnc;

    SVM_Generic::assign(src,onlySemiCopy);

    trainclass = src.trainclass;

    Cweightval     = src.Cweightval;
    Cweightvalfuzz = src.Cweightvalfuzz;
    epsweightval   = src.epsweightval;

    xycache.cheatSetEvalArg((void *) this);
    kerncache.cheatSetEvalArg((void *) this);
    sigmacache.cheatSetEvalArg((void *) this);

    if ( Gpval != nullptr )
    {
	MEMDEL(xyval);   xyval   = nullptr;
	MEMDEL(Gpval);   Gpval   = nullptr;
	MEMDEL(Gpsigma); Gpsigma = nullptr;
    }

    MEMNEW(xyval  ,Matrix<double>(Kcache_celm_v_double,Kcache_celm_double,Kcache_crow_double,(void *) &xycache   ,trainclass.size(),trainclass.size()));
    MEMNEW(Gpval  ,Matrix<double>(Kcache_celm_v_double,Kcache_celm_double,Kcache_crow_double,(void *) &kerncache ,trainclass.size(),trainclass.size()));
    MEMNEW(Gpsigma,Matrix<double>(Kcache_celm_v_double,Kcache_celm_double,Kcache_crow_double,(void *) &sigmacache,trainclass.size(),trainclass.size()));

    locsetGp();

    return;
}

#endif

