
//
// Python frontent for SVMHeavy
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//


#include "pyheavy.hpp"

void setMLTypeClean(int nv) { thewhatsit.setMLTypeClean(nv); return; }

int preallocsize(void) const { return thewhatsit.preallocsize(); }

int  prealloc    (int expectedN) { return thewhatsit.prealloc(expectedN); }
void setmemsize  (int memsize)   {        thewhatsit.setmemsize(memsize); }

int N      (void)  const { return thewhatsit.N();       }
int NNC    (int d) const { return thewhatsit.NNC(d);    }
int type   (void)  const { return thewhatsit.type();    }
int subtype(void)  const { return thewhatsit.subtype(); }

int tspaceDim   (void) const { return thewhatsit.tspaceDim();    }
int xspaceDim   (void) const { return thewhatsit.xspaceDim();    }
int fspaceDim   (void) const { return thewhatsit.fspaceDim();    }
int tspaceSparse(void) const { return thewhatsit.tspaceSparse(); }
int xspaceSparse(void) const { return thewhatsit.xspaceSparse(); }
int numClasses  (void) const { return thewhatsit.numClasses();   }
int order       (void) const { return thewhatsit.order();        }

int isTrained(void) const { return thewhatsit.isTrained(); }
int isMutable(void) const { return thewhatsit.isMutable(); }
int isPool   (void) const { return thewhatsit.isPool();    }

char gOutType(void) const { return thewhatsit.gOutType(); }
char hOutType(void) const { return thewhatsit.hOutType(); }
char targType(void) const { return thewhatsit.targType(); }

int isUnderlyingScalar(void) const { return thewhatsit.isUnderlyingScalar(); }
int isUnderlyingVector(void) const { return thewhatsit.isUnderlyingVector(); }
int isUnderlyingAnions(void) const { return thewhatsit.isUnderlyingAnions(); }

int  getInternalClass  (int y)    const { return thewhatsit.getInternalClass(y);  }
int  numInternalClasses(void)     const { return thewhatsit.getInternalClasses(); }
int  isenabled         (int i)    const { return thewhatsit.isenabled(i);         }

double C       (void)  const { return thewhatsit.C();         }
double sigma   (void)  const { return thewhatsit.sigma();     }
double eps     (void)  const { return thewhatsit.eps();       }
double Cclass  (int d) const { return thewhatsit.Cclass(d);   }
double epsclass(int d) const { return thewhatsit.epsclass(d); }

int    memsize     (void) const { return thewhatsit.memsize();      }
double zerotol     (void) const { return thewhatsit.zerotol();      }
double Opttol      (void) const { return thewhatsit.Opttol();       }
int    maxitcnt    (void) const { return thewhatsit.maxitcnt();     }
double maxtraintime(void) const { return thewhatsit.maxtraintime(); }

int    maxitermvrank(void) const { return thewhatsit.maxitermvrank(); }
double lrmvrank     (void) const { return thewhatsit.lrmvrank();      }
double ztmvrank     (void) const { return thewhatsit.ztmvrank();      }

double betarank(void) const { return thewhatsit.betarank(); }

double sparlvl(void) const { return thewhatsit.sparlvl; }

int isClassifier(void) const { return thewhatsit.isClassifier(); }
int isRegression(void) const { return thewhatsit.isRegression(); }

void fillCache(void) { thewhatsit.fillCache(); return; }

int removeTrainingVector (int i)          { return thewhatsit.removeTrainingVector(i);      }
int removeTrainingVectors(int i, int num) { return thewhatsit.removeTrainingVectors(i,num); }

int setCweight    (int i, double nv) { return thewhatsit.setCweight(i,nv);     }
int setCweightfuzz(int i, double nv) { return thewhatsit.setCweightfuzz(i,nv); }
int setsigmaweight(int i, double nv) { return thewhatsit.setsigmaweight(i,nv); }
int setepsweight  (int i, double nv) { return thewhatsit.setepsweight(i,nv);   }

int scaleCweight    (double s) { return thewhatsit.scaleCweight(s);     }
int scaleCweightfuzz(double s) { return thewhatsit.scaleCweightfuzz(s); }
int scalesigmaweight(double s) { return thewhatsit.scalesigmaweight(s); }
int scaleepsweight  (double s) { return thewhatsit.scaleepsweight(s);   }

int NbasisUU   (void) { return thewhatsit.NbasisUU();    }
int basisTypeUU(void) { return thewhatsit.basisTypeUU(); }
int defProjUU  (void) { return thewhatsit.defProjUU();   }

int setBasisYUU           (void)         { return thewhatsit.setBasisYUU();             }
int setBasisUUU           (void)         { return thewhatsit.setBasisUUU();             }
int removeFromBasisUU     (int i)        { return thewhatsit.removeFromBasisUU(i);      }
int setDefaultProjectionUU(int d)        { return thewhatsit.setDefaultProjectionUU(d); }
int setBasisUU            (int n, int d) { return thewhatsit.setBasisUU(n,d);           }

int NbasisVV   (void) const { return thewhatsit.NbasisVV();    }
int basisTypeVV(void) const { return thewhatsit.basistypeVV(); }
int defProjVV  (void) const { return thewhatsit.defProjVV();   }

int setBasisYVV           (void)         { return thewhatsit.setBasisYVV();             }
int setBasisUVV           (void)         { return thewhatsit.setBasisUVV();             }
int removeFromBasisVV     (int i)        { return thewhatsit.removeFromBasisVV(i);      }
int setDefaultProjectionVV(int d)        { return thewhatsit.setDefaultProjectionVV(d); }
int setBasisVV            (int n, int d) { return thewhatsit.setBasisVV(n,d);           }

int randomise  (double sparsity) { return thewhatsit.randomise(sparsity); }
int autoen     (void)            { return thewhatsit.autoen();            }
int renormalise(void)            { return thewhatsit.renormalise();       }
int realign    (void)            { return thewhatsit.realign();           }

int setzerotol     (double zt)            { return thewhatsit.setzerotol(zt);                 }
int setOpttol      (double xopttol)       { return thewhatsit.setOpttol(xopttol);             }
int setmaxitcnt    (int xmaxitcnt)        { return thewhatsit.setmaxitcnt(xmaxitcnt);         }
int setmaxtraintime(double xmaxtraintime) { return thewhatsit.setmaxtraintime(xmaxtraintime); }

int setmaxitermvrank(int nv)    { return thewhatsit.setmaxitermvrank(nv); }
int setlrmvrank     (double nv) { return thewhatsit.setlrmvrank(nv);      }
int setztmvrank     (double nv) { return thewhatsit.setztmvrank(nv);      }

int setbetarank(double nv) { return thewhatsit.setbetarank(nv); }

int setC       (double xC)          { return thewhatsit.setC(xC);            }
int setsigma   (double xsigma)      { return thewhatsit.setsigma(xsigma);    }
int seteps     (double xeps)        { return thewhatsit.seteps(xeps);        }
int setCclass  (int d, double xC)   { return thewhatsit.setCclass(d,xC);     }
int setepsclass(int d, double xeps) { return thewhatsit.setepsclass(d,xeps); }

int scale  (double a) { return thewhatsit.scale(a);  }
int reset  (void)     { return thewhatsit.reset();   }
int restart(void)     { return thewhatsit.restart(); }
int home   (void)     { return thewhatsit.home();    }

int settspaceDim    (int newdim) { return thewhatsit.settspaceDim(newdim); }
int addtspaceFeat   (int i)      { return thewhatsit.addtspaceFeat(i);     }
int removetspaceFeat(int i)      { return thewhatsit.removetspaceFeat(i);  }
int addxspaceFeat   (int i)      { return thewhatsit.addxspaceFeat(i);     }
int removexspaceFeat(int i)      { return thewhatsit.removexspaceFeat(i);  }

int setsubtype(int i) { return thewhatsit.setsubtype(i); }

int setorder(int neword)             { return thewhatsit.setorder(neword);        }
int addclass(int label, int epszero) { return thewhatsit.addclass(label,epszero); }

int isSampleMode(void) const { return thewhatsit.isSampleMode(); }

void fudgeOn (void) { thewhatsit.fudgeOn();  return; }
void fudgeOff(void) { thewhatsit.fudgeOff(); return; }

int disable(int i) { return thewhatsit.disable(i); }

int normKernelZeroMeanUnitVariance  (int flatnorm, int noshift) { return thewhatsit.normKernelZeroMeanUnitVariance(flatnorm,noshift);   }
int normKernelZeroMedianUnitVariance(int flatnorm, int noshift) { return thewhatsit.normKernelZeroMedianUnitVariance(flatnorm,noshift); }
int normKernelUnitRange             (int flatnorm, int noshift) { return thewhatsit.normKernelUnitRange(flatnorm,noshift);              }

int NZ (void)  const { return thewhatsit.NZ();   }
int NF (void)  const { return thewhatsit.NF();   }
int NS (void)  const { return thewhatsit.NS();   }
int NC (void)  const { return thewhatsit.NC();   }
int NLB(void)  const { return thewhatsit.NLB();  }
int NLF(void)  const { return thewhatsit.NLF();  }
int NUF(void)  const { return thewhatsit.NUF();  }
int NUB(void)  const { return thewhatsit.NUB();  }

int isLinearCost   (void)  const { return thewhatsit.isLinearCost();    }
int isQuadraticCost(void)  const { return thewhatsit.isQuadraticCost(); }
int is1NormCost    (void)  const { return thewhatsit.is1NormCost();     }
int isVarBias      (void)  const { return thewhatsit.isVarBias();       }
int isPosBias      (void)  const { return thewhatsit.isPosBias();       }
int isNegBias      (void)  const { return thewhatsit.isNegBias();       }
int isFixedBias    (void)  const { return thewhatsit.isFixedBias();     }

int isNoMonotonicConstraints   (void) const { return thewhatsit.isMonotonicConstraints();      }
int isForcedMonotonicIncreasing(void) const { return thewhatsit.isForcedMonotonicIncreasing(); }
int isForcedMonotonicDecreasing(void) const { return thewhatsit.isForcedMonotonicDecreasing(); }

int isOptActive(void) const { return thewhatsit.isOptActive(); }
int isOptSMO   (void) const { return thewhatsit.isOptSMO();    }
int isOptD2C   (void) const { return thewhatsit.isOptD2C();    }
int isOptGrad  (void) const { return thewhatsit.isOptGrad();   }

int isFixedTube (void) const { return thewhatsit.isFixedTube();  }
int isShrinkTube(void) const { return thewhatsit.isShrinkTube(); }

int isRestrictEpsPos(void) const { return thewhatsit.isRestrictEpsPos(); }
int isRestrictEpsNeg(void) const { return thewhatsit.isRestrictEpsNeg(); }

int isClassifyViaSVR(void) const { return thewhatsit.isClassifyViaSVR(); }
int isClassifyViaSVM(void) const { return thewhatsit.isClassifyViaSVM(); }

int is1vsA   (void) const { return thewhatsit.is1vsA();    }
int is1vs1   (void) const { return thewhatsit.is1vs1();    }
int isDAGSVM (void) const { return thewhatsit.isDAGSVM();  }
int isMOC    (void) const { return thewhatsit.isMOC();     }
int ismaxwins(void) const { return thewhatsit.ismaxwins(); }
int isrecdiv (void) const { return thewhatsit.isrecdiv();  }

int isatonce(void) const { return thewhatsit.isatonce(); }
int isredbin(void) const { return thewhatsit.isredbin(); }

int isKreal  (void) const { return thewhatsit.isKreal();   }
int isKunreal(void) const { return thewhatsit.isKunreal(); }

int isanomalyOn (void) const { return thewhatsit.isanomalyOn();  }
int isanomalyOff(void) const { return thewhatsit.isanomalyOff(); }

int isautosetOff         (void) const { return thewhatsit.isautosetOff();          }
int isautosetCscaled     (void) const { return thewhatsit.isautosetCscaled();      }
int isautosetCKmean      (void) const { return thewhatsit.isautosetCKmean();       }
int isautosetCKmedian    (void) const { return thewhatsit.isautosetCKmedian();     }
int isautosetCNKmean     (void) const { return thewhatsit.isautosetCNKmean();      }
int isautosetCNKmedian   (void) const { return thewhatsit.isautosetCNKmedian();    }
int isautosetLinBiasForce(void) const { return thewhatsit.isautosetLinBiasForce(); }

double outerlr      (void) const { return thewhatsit.outerlr();       }
double outermom     (void) const { return thewhatsit.outermom();      }
int    outermethod  (void) const { return thewhatsit.outermethod();   }
double outertol     (void) const { return thewhatsit.outertol();      }
double outerovsc    (void) const { return thewhatsit.outerovsc();     }
int    outermaxitcnt(void) const { return thewhatsit.outermaxitcnt(); }
int    outermaxcache(void) const { return thewhatsit.outermaxcache(); }

int    maxiterfuzzt(void) const { return thewhatsit.maxiterfuzzt(); }
int    usefuzzt    (void) const { return thewhatsit.usefzzt();      }
double lrfuzzt     (void) const { return thewhatsit.lrfuzzt();      }
double ztfuzzt     (void) const { return thewhatsit.ztfufft();      }

int m(void) const { return thewhatsit.m(); }

double LinBiasForce (void)  const { return thewhatsit.LinBiasForce();  }
double QuadBiasForce(void)  const { return thewhatsit.QuadBiasForce(); }

double nu    (void) const { return thewhatsit.nu();     }
double nuQuad(void) const { return thewhatsit.nuQuad(); }

double theta  (void) const { return thewhatsit.theta();   }
int    simnorm(void) const { return thewhatsit.simnorm(); }

double anomalyNu   (void) const { return thewhatsit.anomalyNu();    }
int    anomalyClass(void) const { return thewhatsit.anomalyClass(); }

double autosetCval (void) const { return thewhatsit.autosetCval();  }
double autosetnuval(void) const { return thewhatsit.autosetnuval(); }

int    anomclass      (void) const { return thewhatsit.anomclass();       }
int    singmethod     (void) const { return thewhatsit.singmethod();      }
double rejectThreshold(void) const { return thewhatsit.rejectThreshold(); }

int removeNonSupports(void)        { return thewhatsit.removeNonSupports();      }
int trimTrainingSet  (int maxsize) { return thewhatsit.trimTrainingSet(maxsize); }

int setLinearCost   (void) { return thewhatsit.setLinearCost();    }
int setQuadraticCost(void) { return thewhatsit.setQuadraticCost(); }
int set1NormCost    (void) { return thewhatsit.set1NormCost();     }
int setVarBias      (void) { return thewhatsit.setVarBias();       }
int setPosBias      (void) { return thewhatsit.setPosBias();       }
int setNegBias      (void) { return thewhatsit.setNegBias();       }

int setNoMonotonicConstraints   (void) { return thewhatsit.
int setForcedMonotonicIncreasing(void) { return thewhatsit.
int setForcedMonotonicDecreasing(void) { return thewhatsit.

int setOptActive(void) { return thewhatsit.
int setOptSMO   (void) { return thewhatsit.
int setOptD2C   (void) { return thewhatsit.
int setOptGrad  (void) { return thewhatsit.

int setFixedTube (void) { return thewhatsit.
int setShrinkTube(void) { return thewhatsit.

int setRestrictEpsPos(void) { return thewhatsit.
int setRestrictEpsNeg(void) { return thewhatsit.

int setClassifyViaSVR(void) { return thewhatsit.
int setClassifyViaSVM(void) { return thewhatsit.

int set1vsA   (void) { return thewhatsit.
int set1vs1   (void) { return thewhatsit.
int setDAGSVM (void) { return thewhatsit.
int setMOC    (void) { return thewhatsit.
int setmaxwins(void) { return thewhatsit.
int setrecdiv (void) { return thewhatsit.

int setatonce(void) { return thewhatsit.
int setredbin(void) { return thewhatsit.

int setKreal  (void) { return thewhatsit.
int setKunreal(void) { return thewhatsit.

int anomalyOn (int danomalyClass, double danomalyNu) { return thewhatsit.
int anomalyOff(void)                                 { return thewhatsit.

int setouterlr      (double xouterlr)     { return thewhatsit.
int setoutermom     (double xoutermom)    { return thewhatsit.
int setoutermethod  (int xoutermethod)    { return thewhatsit.
int setoutertol     (double xoutertol)    { return thewhatsit.
int setouterovsc    (double xouterovsc)   { return thewhatsit.
int setoutermaxitcnt(int xoutermaxits)    { return thewhatsit.
int setoutermaxcache(int xoutermaxcacheN) { return thewhatsit.

int setmaxiterfuzzt(int xmaxiterfuzzt)               { return thewhatsit.
int setusefuzzt    (int xusefuzzt)                   { return thewhatsit.
int setlrfuzzt     (double xlrfuzzt)                 { return thewhatsit.
int setztfuzzt     (double xztfuzzt)                 { return thewhatsit.

int setm(int xm) { return thewhatsit.

int setLinBiasForce (double newval)        { return thewhatsit.
int setQuadBiasForce(double newval)        { return thewhatsit.

int setnu    (double xnu)     { return thewhatsit.
int setnuQuad(double xnuQuad) { return thewhatsit.

int settheta  (double nv) { return thewhatsit.
int setsimnorm(int nv)    { return thewhatsit.

int autosetOff         (void)                      { return thewhatsit.
int autosetCscaled     (double Cval)               { return thewhatsit.
int autosetCKmean      (void)                      { return thewhatsit.
int autosetCKmedian    (void)                      { return thewhatsit.
int autosetCNKmean     (void)                      { return thewhatsit.
int autosetCNKmedian   (void)                      { return thewhatsit.
int autosetLinBiasForce(double nuval, double Cval) { return thewhatsit.

void setanomalyclass   (int n)     { thewhatsit.
void setsingmethod     (int nv)    { thewhatsit.
void setRejectThreshold(double nv) { thewhatsit.

double quasiloglikelihood(void) const { return thewhatsit.

int isZeromuBias(void) const { return thewhatsit.
int isVarmuBias (void) const { return thewhatsit.

int setZeromuBias(void) { return thewhatsit.
int setVarmuBias (void) { return thewhatsit.

double loglikelihood(void) const { return thewhatsit.

int isVardelta (void) const { return thewhatsit.
int isZerodelta(void) const { return thewhatsit.

int setVardelta (void) { return thewhatsit.
int setZerodelta(void) { return thewhatsit.

double lsvloglikelihood(void) const { return thewhatsit.

int k  (void) const { return thewhatsit.
int ktp(void) const { return thewhatsit.

int setk  (int xk) { return thewhatsit.
int setktp(int xk) { return thewhatsit.

int bernDegree(void) const { return thewhatsit.
int bernIndex (void) const { return thewhatsit.

int setBernDegree(int nv) { return thewhatsit.
int setBernIndex (int nv) { return thewhatsit.

double batttmax      (void) const { return thewhatsit.
double battImax      (void) const { return thewhatsit.
double batttdelta    (void) const { return thewhatsit.
double battVstart    (void) const { return thewhatsit.
double battthetaStart(void) const { return thewhatsit.

int setbatttmax      (double nv) { return thewhatsit.
int setbattImax      (double nv) { return thewhatsit.
int setbatttdelta    (double nv) { return thewhatsit.
int setbattVstart    (double nv) { return thewhatsit.
int setbattthetaStart(double nv) { return thewhatsit.



















int kernel_isFullNorm       (void) const
int kernel_isProd           (void) const
int kernel_isIndex          (void) const
int kernel_isShifted        (void) const
int kernel_isScaled         (void) const
int kernel_isShiftedScaled  (void) const
int kernel_isLeftPlain      (void) const
int kernel_isRightPlain     (void) const
int kernel_isLeftRightPlain (void) const
int kernel_isLeftNormal     (void) const
int kernel_isRightNormal    (void) const
int kernel_isLeftRightNormal(void) const
int kernel_isPartNormal     (void) const
int kernel_isAltDiff        (void) const
int kernel_needsmProd       (void) const
int kernel_wantsXYprod      (void) const
int kernel_suggestXYcache   (void) const
int kernel_isIPdiffered     (void) const

int kernel_size       (void) const
int kernel_getSymmetry(void) const

double kernel_cWeight(int q) const
int    kernel_cType  (int q) const

int kernel_isNormalised(int q) const
int kernel_isChained   (int q) const
int kernel_isSplit     (int q) const
int kernel_isMulSplit  (int q) const
int kernel_isMagTerm   (int q) const

int kernel_numSplits   (void) const
int kernel_numMulSplits(void) const

double kernel_cRealConstants(int i, int q) const
int    kernel_cIntConstants (int i, int q) const
int    kernel_cRealOverwrite(int i, int q) const
int    kernel_cIntOverwrite (int i, int q) const

double kernel_getRealConstZero(int q)
int    kernel_getIntConstZero (int q)

int kernel_isKVarianceNZ(void) const

void kernel_add   (int q)
void kernel_remove(int q)
void kernel_resize(int nsize)

void kernel_setFullNorm       (void)
void kernel_setNoFullNorm     (void)
void kernel_setProd           (void)
void kernel_setnonProd        (void)
void kernel_setLeftPlain      (void)
void kernel_setRightPlain     (void)
void kernel_setLeftRightPlain (void)
void kernel_setLeftNormal     (void)
void kernel_setRightNormal    (void)
void kernel_setLeftRightNormal(void)

void kernel_setAltDiff       (int nv)
void kernel_setsuggestXYcache(int nv)
void kernel_setIPdiffered    (int nv)

void kernel_setChained   (int q)
void kernel_setNormalised(int q)
void kernel_setSplit     (int q)
void kernel_setMulSplit  (int q)
void kernel_setMagTerm   (int q)

void kernel_setUnChained   (int q)
void kernel_setUnNormalised(int q)
void kernel_setUnSplit     (int q)
void kernel_setUnMulSplit  (int q)
void kernel_setUnMagTerm   (int q)

void kernel_setWeight (double nw,   int q)
void kernel_setType   (int ndtype,  int q)
void kernel_setAltCall(int newMLid, int q)

void kernel_setRealConstants(double ndRealConstants, int i, int q)
void kernel_setIntConstants (int    ndIntConstants,  int i, int q)
void kernel_setRealOverwrite(int    ndRealOverwrite, int i, int q)
void kernel_setIntOverwrite (int    ndIntOverwrite,  int i, int q)

void kernel_setRealConstZero(double nv, int q)
void kernel_setIntConstZero (int nv,    int q)















int kernelUU_isFullNorm       (void) const
int kernelUU_isProd           (void) const
int kernelUU_isIndex          (void) const
int kernelUU_isShifted        (void) const
int kernelUU_isScaled         (void) const
int kernelUU_isShiftedScaled  (void) const
int kernelUU_isLeftPlain      (void) const
int kernelUU_isRightPlain     (void) const
int kernelUU_isLeftRightPlain (void) const
int kernelUU_isLeftNormal     (void) const
int kernelUU_isRightNormal    (void) const
int kernelUU_isLeftRightNormal(void) const
int kernelUU_isPartNormal     (void) const
int kernelUU_isAltDiff        (void) const
int kernelUU_needsmProd       (void) const
int kernelUU_wantsXYprod      (void) const
int kernelUU_suggestXYcache   (void) const
int kernelUU_isIPdiffered     (void) const

int kernelUU_size       (void) const
int kernelUU_getSymmetry(void) const

double kernelUU_cWeight(int q) const
int    kernelUU_cType  (int q) const

int kernelUU_isNormalised(int q) const
int kernelUU_isChained   (int q) const
int kernelUU_isSplit     (int q) const
int kernelUU_isMulSplit  (int q) const
int kernelUU_isMagTerm   (int q) const

int kernelUU_numSplits   (void) const
int kernelUU_numMulSplits(void) const

double kernelUU_cRealConstants(int i, int q) const
int    kernelUU_cIntConstants (int i, int q) const
int    kernelUU_cRealOverwrite(int i, int q) const
int    kernelUU_cIntOverwrite (int i, int q) const

double kernelUU_getRealConstZero(int q)
int    kernelUU_getIntConstZero (int q)

int kernelUU_isKVarianceNZ(void) const

void kernelUU_add   (int q)
void kernelUU_remove(int q)
void kernelUU_resize(int nsize)

void kernelUU_setFullNorm       (void)
void kernelUU_setNoFullNorm     (void)
void kernelUU_setProd           (void)
void kernelUU_setnonProd        (void)
void kernelUU_setLeftPlain      (void)
void kernelUU_setRightPlain     (void)
void kernelUU_setLeftRightPlain (void)
void kernelUU_setLeftNormal     (void)
void kernelUU_setRightNormal    (void)
void kernelUU_setLeftRightNormal(void)

void kernelUU_setAltDiff       (int nv)
void kernelUU_setsuggestXYcache(int nv)
void kernelUU_setIPdiffered    (int nv)

void kernelUU_setChained   (int q)
void kernelUU_setNormalised(int q)
void kernelUU_setSplit     (int q)
void kernelUU_setMulSplit  (int q)
void kernelUU_setMagTerm   (int q)

void kernelUU_setUnChained   (int q)
void kernelUU_setUnNormalised(int q)
void kernelUU_setUnSplit     (int q)
void kernelUU_setUnMulSplit  (int q)
void kernelUU_setUnMagTerm   (int q)

void kernelUU_setWeight (double nw,   int q)
void kernelUU_setType   (int ndtype,  int q)
void kernelUU_setAltCall(int newMLid, int q)

void kernelUU_setRealConstants(double ndRealConstants, int i, int q)
void kernelUU_setIntConstants (int    ndIntConstants,  int i, int q)
void kernelUU_setRealOverwrite(int    ndRealOverwrite, int i, int q)
void kernelUU_setIntOverwrite (int    ndIntOverwrite,  int i, int q)

void kernelUU_setRealConstZero(double nv, int q)
void kernelUU_setIntConstZero (int nv,    int q)














int train(int &res) { return thewhatsit.train(res);

int setFixedBias    (double newbias) { return thewhatsit.

double calcDistNul(                      int ia, int db)
double calcDistInt(int    ha, int    hb, int ia, int db)
double calcDistDbl(double ha, double hb, int ia, int db)

void ClassLabels(int *res) const

int addTrainingVectorNul(int i,           const double *x, int dim, double Cweigh, double epsweigh)
int addTrainingVectorInt(int i, int    z, const double *x, int dim, double Cweigh, double epsweigh)
int addTrainingVectorDbl(int i, double z, const double *x, int dim, double Cweigh, double epsweigh)

int setx(int i, const double *x, int dim)

int setyInt(int i, double  y)
int setyDbl(int i, double  y)

int addToBasisUU(int i, const double *o, int dim)
int setBasisUU  (int i, const double *o, int dim)

int addToBasisVV(int i, const double *o, int dim)
int setBasisVV  (int i, const double *o, int dim)

int setSampleMode(int nv, const double *xmin, const double *xmax, int dim, int Nsamp, int sampSplit, int sampType, double sampScale)

double ggTrainingVector(int i) const
double hhTrainingVector(int i) const

double varTrainingVector     (int i)                                                   const
double covTrainingVector     (int i, int j)                                            const
double stabProbTrainingVector(int i, int p, double pnrm, int rot, double mu, double B) const

double gg(const double *x, int dim) const
double hh(const double *x, int dim) const

double var     (const double *xa, int dim)                                         const
double cov     (const double *xa, const double *xb, int dim)                       const
double stabProb(const double *x, int p, double pnrm, int rot, double mu, double B) const

int setAlpha(const double *newAlpha)
int setBias (double newBias)

int setmuWeight(const double *nv, int dim)
int setmuBias  (double nv)

void   muWeight(double *res) const
double muBias(void)          const

int setgamma(const double *newW, int dim)
int setdelta(double newB)

void gamma(double *res) const
double delta(void) const

int setbattparam(const double *nv, int dim)
void battparam(double *res) const














