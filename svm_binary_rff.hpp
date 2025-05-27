
//
// Binary Classification RFF SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _svm_binary_rff_h
#define _svm_binary_rff_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "svm_scalar_rff.hpp"








class SVM_Binary_rff;


// Swap function

inline void qswap(SVM_Binary_rff &a, SVM_Binary_rff &b);


class SVM_Binary_rff : public SVM_Scalar_rff
{
public:

    SVM_Binary_rff();
    SVM_Binary_rff(const SVM_Binary_rff &src);
    SVM_Binary_rff(const SVM_Binary_rff &src, const ML_Base *xsrc);
    SVM_Binary_rff &operator=(const SVM_Binary_rff &src) { assign(src); return *this; }
    virtual ~SVM_Binary_rff();

    virtual int prealloc(int expectedN) override;
    virtual int preallocsize(void) const override;

    virtual int restart(void) override { SVM_Binary_rff temp; *this = temp; return 1; }

    virtual std::ostream &printstream(std::ostream &output, int dep) const override;
    virtual std::istream &inputstream(std::istream &input) override;

    // Information:

    virtual int NNC(int d) const override { return binNnc(d+1); }

    virtual int tspaceDim(void)  const override { return 1;  }
    virtual int numClasses(void) const override { return 2;  }
    virtual int type(void)       const override { return 23; }
    virtual int subtype(void)    const override { return 0;  }

    virtual char gOutType(void) const override { return 'R'; }
    virtual char hOutType(void) const override { return 'Z'; }
    virtual char targType(void) const override { return 'Z'; }
    virtual double calcDist(const gentype &ha, const gentype &hb, int ia = -1, int db = 2) const override;

    virtual int numInternalClasses(void) const override { return SVM_Generic::numInternalClasses(); }

    virtual double eps(void)          const override { return bineps;           }
    virtual double Cclass(int d)      const override { return binCclass(d+1);   }
    virtual double epsclass(int d)    const override { return binepsclass(d+1); }





    virtual int isClassifyViaSVR(void) const override { return isSVMviaSVR;  }
    virtual int isClassifyViaSVM(void) const override { return !isSVMviaSVR; }

    virtual int is1vsA(void)    const override { return 0; }
    virtual int is1vs1(void)    const override { return 1; }
    virtual int isDAGSVM(void)  const override { return 1; }
    virtual int isMOC(void)     const override { return 1; }
    virtual int ismaxwins(void) const override { return 0; }
    virtual int isrecdiv(void)  const override { return 1; }












    virtual const Vector<int>    &d(void)           const override { return bintrainclass;  }
    virtual const Vector<double> &Cweight(void)     const override { return binCweight;     }
    virtual const Vector<double> &Cweightfuzz(void) const override { return binCweightfuzz; }
    virtual const Vector<double> &epsweight(void)   const override { return binepsweight;   }
    virtual const Vector<double> &zR(void)          const override { return bintraintarg;   }

    virtual double zR(int i) const override { if ( i >= 0 ) { return zR()(i); } return 0; } // Tests always set zR(-1) = 0, so this is safe

    virtual int isClassifier(void) const override { return 1; }

    // Modification:

    virtual int setC(double xC) override;
    virtual int seteps(double xeps) override;
    virtual int setCclass(int d, double xC) override;
    virtual int setepsclass(int d, double xeps) override;

    virtual int sety(int                i, double                z) override;
    virtual int sety(const Vector<int> &i, const Vector<double> &d) override;
    virtual int sety(                      const Vector<double> &z) override;














    virtual int setClassifyViaSVR(void) override;
    virtual int setClassifyViaSVM(void) override;









    // Classify-with-reject:
    //
    // Set threshold zero for normal operation, otherwise this is "d" in Bartlett

    virtual double rejectThreshold(void) const override { return dthres; }
    virtual void setRejectThreshold(double nv) override { NiceAssert( ( nv >= 0 ) || ( nv <= 0.5 ) ); dthres = nv; return; }







    // Training set control

    virtual int addTrainingVector (int i, const gentype &z, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2) override;
    virtual int qaddTrainingVector(int i, const gentype &z,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2) override;

    virtual int addTrainingVector (int i, const Vector<gentype> &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override;
    virtual int qaddTrainingVector(int i, const Vector<gentype> &z,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override;

    virtual int addTrainingVector (int i, int d, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, double z = 0.0);
    virtual int qaddTrainingVector(int i, int d,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, double z = 0.0);

    virtual int addTrainingVector (int i, const Vector<int> &d, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<double> &z);
    virtual int qaddTrainingVector(int i, const Vector<int> &d,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<double> &z);

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

    virtual int scaleCweight(double scalefactor) override;
    virtual int scaleCweightfuzz(double scalefactor) override;
    virtual int scaleepsweight(double scalefactor) override;

    // Train the SVM

    virtual int train(int &res, svmvolatile int &killSwitch) override;
    virtual int train(int &res) override { svmvolatile int killSwitch = 0; return train(res,killSwitch); }

    // Evaluation:

    virtual int isVarDefined(void) const override { return 0; }

    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = nullptr) const override;

    virtual double eTrainingVector(int i) const override;

    virtual double         &dedgTrainingVector(double         &res, int i) const override;
    virtual Vector<double> &dedgTrainingVector(Vector<double> &res, int i) const override { dedgTrainingVector((res.resize(1))("&",0),i);   return res; }
    virtual d_anion        &dedgTrainingVector(d_anion        &res, int i) const override { dedgTrainingVector((res.setorder(0))("&",0),i); return res; }
    virtual gentype        &dedgTrainingVector(gentype        &res, int i) const override { dedgTrainingVector(res.force_double(),i);       return res; }

    virtual double &d2edg2TrainingVector(double &res, int i) const override;

    virtual double dedKTrainingVector(int i, int j) const override { double tmp; return dedgTrainingVector(tmp,i)*alphaR()(j); }
    virtual Vector<double> &dedKTrainingVector(Vector<double> &res, int i) const override;
    virtual Matrix<double> &dedKTrainingVector(Matrix<double> &res) const override;

    // Other functions

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0) override;

    virtual void semicopy(const ML_Base &src) override;
    virtual void qswapinternal(ML_Base &b) override;

    virtual int disable(int i) override;
    virtual int disable(const Vector<int> &i) override;

private:

    // Legacy from svm_binary

    double bineps;
    Vector<int>    bintrainclass;
    Vector<double> bintraintarg;
    Vector<double> binepsclass;
    Vector<double> binepsweight;
    Vector<double> binCclass;
    Vector<double> binCweight;     // rely on ML_Base callback to do sigma stuff
    Vector<double> binCweightfuzz; // rely on ML_Base callback to do sigma stuff





    // This flag forces the SVM to act as a regressor with targets +-1 and
    // no insensitive zero (zero tube width).  Combining this with quadratic
    // cost implements the binary LS-SVM described by Suykens

    int isSVMviaSVR;

    // Information on class counts (must be kept locally, as the scalar
    // SVM version will be inaccurate if running SVM as SVR)

    Vector<int> binNnc; // number of vectors in each class (-1,0,+1)

    // Bartlett's "classification with reject option" "d" threshold
    // 
    // d = 0: not using reject, so do nothing special
    // d > 0: during training we need to duplicate a bunch of alphas 
    //        (cheat on x) that are y_i g(x_i) >= 0 (no boundary eps), 
    //        train, then recombine alpha

    double dthres;

    // Internal functions


    int setdinternal(int i, int d);


    void setalleps(double xeps, const Vector<double> &xepsclass);

public:
    // Train, with reject, but assuming extra bits already added

    virtual int loctrain(int &res, svmvolatile int &killSwitch, int realN, int assumeDNZ = 0);
};

inline SVM_Binary_rff &setident (SVM_Binary_rff &a) { NiceThrow("something"); return a; }
inline SVM_Binary_rff &setzero  (SVM_Binary_rff &a) { a.restart(); return a; }
inline SVM_Binary_rff &setposate(SVM_Binary_rff &a) { return a; }
inline SVM_Binary_rff &setnegate(SVM_Binary_rff &a) { NiceThrow("something"); return a; }
inline SVM_Binary_rff &setconj  (SVM_Binary_rff &a) { NiceThrow("something"); return a; }
inline SVM_Binary_rff &setrand  (SVM_Binary_rff &a) { NiceThrow("something"); return a; }
inline SVM_Binary_rff &postProInnerProd(SVM_Binary_rff &a) { return a; }


inline double norm2(const SVM_Binary_rff &a);
inline double abs2 (const SVM_Binary_rff &a);

inline double norm2(const SVM_Binary_rff &a) { return a.RKHSnorm(); }
inline double abs2 (const SVM_Binary_rff &a) { return a.RKHSabs();  }

inline void qswap(SVM_Binary_rff &a, SVM_Binary_rff &b)
{
    a.qswapinternal(b);

    return;
}

inline void SVM_Binary_rff::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    SVM_Binary_rff &b = dynamic_cast<SVM_Binary_rff &>(bb.getML());

    SVM_Scalar_rff::qswapinternal(b);

    qswap(bineps          ,b.bineps          );
    qswap(bintrainclass   ,b.bintrainclass   );
    qswap(bintraintarg    ,b.bintraintarg    );
    qswap(binepsclass     ,b.binepsclass     );
    qswap(binepsweight    ,b.binepsweight    );
    qswap(binCclass       ,b.binCclass       );
    qswap(binCweight      ,b.binCweight      );
    qswap(binCweightfuzz  ,b.binCweightfuzz  );
    qswap(isSVMviaSVR    ,b.isSVMviaSVR    );
    qswap(binNnc         ,b.binNnc         );
    qswap(dthres         ,b.dthres         );

    return;
}

inline void SVM_Binary_rff::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const SVM_Binary_rff &b = dynamic_cast<const SVM_Binary_rff &>(bb.getMLconst());

    SVM_Scalar_rff::semicopy(b);

    bineps         = b.bineps;
    binepsclass    = b.binepsclass;
    binepsweight   = b.binepsweight;
    binCclass      = b.binCclass;
    binCweight     = b.binCweight;
    binCweightfuzz = b.binCweightfuzz;

    isSVMviaSVR = b.isSVMviaSVR;

    binNnc           = b.binNnc;
    bintraintarg     = b.bintraintarg;
    bintrainclass    = b.bintrainclass;

    dthres = b.dthres;

    return;
}

inline void SVM_Binary_rff::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const SVM_Binary_rff &src = dynamic_cast<const SVM_Binary_rff &>(bb.getMLconst());

    SVM_Scalar_rff::assign(static_cast<const SVM_Scalar_rff &>(src),onlySemiCopy);

    bineps         = src.bineps;
    bintrainclass  = src.bintrainclass;
    bintraintarg   = src.bintraintarg;
    binepsclass    = src.binepsclass;
    binepsweight   = src.binepsweight;
    binCclass      = src.binCclass;
    binCweight     = src.binCweight;
    binCweightfuzz = src.binCweightfuzz;
    isSVMviaSVR    = src.isSVMviaSVR;
    binNnc         = src.binNnc;

    dthres = src.dthres;

    return;
}

#endif
