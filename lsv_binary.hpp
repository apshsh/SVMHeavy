
//
// Binary Classification SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _lsv_binary_h
#define _lsv_binary_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "lsv_scalar.hpp"








class LSV_Binary;


// Swap function

inline void qswap(LSV_Binary &a, LSV_Binary &b);


class LSV_Binary : public LSV_Scalar
{
public:

    LSV_Binary();
    LSV_Binary(const LSV_Binary &src);
    LSV_Binary(const LSV_Binary &src, const ML_Base *xsrc);
    LSV_Binary &operator=(const LSV_Binary &src) { assign(src); return *this; }
    virtual ~LSV_Binary();

    virtual int prealloc    (int expectedN)       override;
    virtual int preallocsize(void)          const override;

    virtual void assign       (const ML_Base &src, int onlySemiCopy = 0) override;
    virtual void semicopy     (const ML_Base &src)                       override;
    virtual void qswapinternal(ML_Base &b)                               override;

    virtual std::ostream &printstream(std::ostream &output, int dep) const override;
    virtual std::istream &inputstream(std::istream &input) override;

    // Information:

    virtual int NNC(int d)      const override { return binNnc(d+1); }
    virtual int type(void)      const override { return 511;         }
    virtual int subtype(void)   const override { return 0;           }
    virtual char gOutType(void) const override { return 'R';         }
    virtual char hOutType(void) const override { return 'Z';         }
    virtual char targType(void) const override { return 'Z';         }

    virtual int tspaceDim(void)  const override { return 1; }
    virtual int numClasses(void) const override { return 2; }

    virtual double calcDist(const gentype &ha, const gentype &hb, int ia = -1, int db = 2) const override;

    virtual int numInternalClasses(void) const override { return LSV_Generic::numInternalClasses(); }

    virtual double eps(void)          const override { return bineps;           }
    virtual double Cclass(int d)      const override { return binCclass(d+1);   }
    virtual double epsclass(int d)    const override { return binepsclass(d+1); }

    virtual const Vector<int>    &d(void)           const override { return bintrainclass;  }
    virtual const Vector<double> &Cweight(void)     const override { return binCweight;     }
    virtual const Vector<double> &Cweightfuzz(void) const override { return binCweightfuzz; }
    virtual const Vector<double> &epsweight(void)   const override { return binepsweight;   }

    virtual int isClassifier(void) const override { return 1; }
    virtual int isRegression(void) const override { return 0; }

    // Kernel Modification

    virtual void prepareKernel(void) override;

    virtual int resetKernel(                             int modind = 1, int onlyChangeRowI = -1, int updateInfo = 1) override;
    virtual int setKernel  (const MercerKernel &xkernel, int modind = 1, int onlyChangeRowI = -1                    ) override;

    // Training set control

    virtual int addTrainingVector (int i, const gentype &z, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2) override;
    virtual int qaddTrainingVector(int i, const gentype &z,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2) override;

    virtual int addTrainingVector (int i, const Vector<gentype> &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override;
    virtual int qaddTrainingVector(int i, const Vector<gentype> &z,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override;

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

    virtual int scaleCweight    (double scalefactor) override;
    virtual int scaleCweightfuzz(double scalefactor) override;
    virtual int scaleepsweight  (double scalefactor) override;

    // General modification

    virtual int setC(double xC) override;
    virtual int seteps(double xeps) override;
    virtual int setCclass(int d, double xC) override;
    virtual int setepsclass(int d, double xeps) override;

    virtual int restart(void) override { LSV_Binary temp; *this = temp; return 1; }

    // Train the SVM

    virtual int train(int &res, svmvolatile int &killSwitch) override;
    virtual int train(int &res) override { svmvolatile int killSwitch = 0; return train(res,killSwitch); }

    // Evaluation:

    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = nullptr) const override;

    virtual double eTrainingVector(int i) const override;

    virtual double         &dedgTrainingVector(double         &res, int i) const override;
    virtual Vector<double> &dedgTrainingVector(Vector<double> &res, int i) const override { dedgTrainingVector((res.resize(1))("&",0),i);   return res; }
    virtual d_anion        &dedgTrainingVector(d_anion        &res, int i) const override { dedgTrainingVector((res.setorder(0))("&",0),i); return res; }
    virtual gentype        &dedgTrainingVector(gentype        &res, int i) const override { dedgTrainingVector(res.force_double(),i);       return res; }

    virtual double &d2edg2TrainingVector(double &res, int i) const override;

    virtual double          dedKTrainingVector(int i, int j)               const override { double tmp; return dedgTrainingVector(tmp,i)*alphaR()(j); }
    virtual Vector<double> &dedKTrainingVector(Vector<double> &res, int i) const override;
    virtual Matrix<double> &dedKTrainingVector(Matrix<double> &res)        const override;













private:

    // For speed reasons we want to put tube information in gp (via z) rather
    // than hp (via eps in the scalar regression level).  For this reason we
    // need to keep a local version of the following variables and translate
    // them down for the regression level.

    double bineps;
    Vector<int>    bintrainclass;
    Vector<double> bintraintarg;
    Vector<double> binepsclass;
    Vector<double> binepsweight;
    Vector<double> binCclass;
    Vector<double> binCweight;     // rely on ML_Base callback to do sigma stuff
    Vector<double> binCweightfuzz; // rely on ML_Base callback to do sigma stuff

    // Information on class counts (must be kept locally, as the scalar
    // SVM version will be inaccurate if running SVM as SVR)

    Vector<int> binNnc; // number of vectors in each class (-1,0,+1)

    // Internal functions

    int setdinternal(int i, int d);
    void setalleps(double xeps, const Vector<double> &xepsclass);
    double zR(int i) const override { if ( i >= 0 ) { return bintraintarg(i); } return 0; } // Tests always set zR(-1) = 0, so this is safe

    virtual int addTrainingVector (int i, int d, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, double z = 0.0);
    virtual int qaddTrainingVector(int i, int d,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, double z = 0.0);

    virtual int addTrainingVector (int i, const Vector<int> &d, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<double> &z);
    virtual int qaddTrainingVector(int i, const Vector<int> &d,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<double> &z);

    virtual int sety(int                i, double                z) override;
    virtual int sety(const Vector<int> &i, const Vector<double> &d) override;
    virtual int sety(                      const Vector<double> &z) override;
};

inline LSV_Binary &setident (LSV_Binary &a) { NiceThrow("something"); return a; }
inline LSV_Binary &setzero  (LSV_Binary &a) { a.restart(); return a; }
inline LSV_Binary &setposate(LSV_Binary &a) { return a; }
inline LSV_Binary &setnegate(LSV_Binary &a) { NiceThrow("something"); return a; }
inline LSV_Binary &setconj  (LSV_Binary &a) { NiceThrow("something"); return a; }
inline LSV_Binary &setrand  (LSV_Binary &a) { NiceThrow("something"); return a; }
inline LSV_Binary &postProInnerProd(LSV_Binary &a) { return a; }


inline double norm2(const LSV_Binary &a);
inline double abs2 (const LSV_Binary &a);

inline double norm2(const LSV_Binary &a) { return a.RKHSnorm(); }
inline double abs2 (const LSV_Binary &a) { return a.RKHSabs();  }

inline void qswap(LSV_Binary &a, LSV_Binary &b)
{
    a.qswapinternal(b);

    return;
}

inline void LSV_Binary::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    LSV_Binary &b = dynamic_cast<LSV_Binary &>(bb.getML());

    LSV_Scalar::qswapinternal(b);

    qswap(bineps         ,b.bineps         );
    qswap(bintrainclass  ,b.bintrainclass  );
    qswap(bintraintarg   ,b.bintraintarg   );
    qswap(binepsclass    ,b.binepsclass    );
    qswap(binepsweight   ,b.binepsweight   );
    qswap(binCclass      ,b.binCclass      );
    qswap(binCweight     ,b.binCweight     );
    qswap(binCweightfuzz ,b.binCweightfuzz );
    qswap(binNnc         ,b.binNnc         );

    return;
}

inline void LSV_Binary::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const LSV_Binary &b = dynamic_cast<const LSV_Binary &>(bb.getMLconst());

    LSV_Scalar::semicopy(b);

    bineps         = b.bineps;
    binepsclass    = b.binepsclass;
    binepsweight   = b.binepsweight;
    binCclass      = b.binCclass;
    binCweight     = b.binCweight;
    binCweightfuzz = b.binCweightfuzz;

    binNnc        = b.binNnc;
    bintraintarg  = b.bintraintarg;
    bintrainclass = b.bintrainclass;

    return;
}

inline void LSV_Binary::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const LSV_Binary &src = dynamic_cast<const LSV_Binary &>(bb.getMLconst());

    LSV_Scalar::assign(static_cast<const LSV_Scalar &>(src),onlySemiCopy);

    bineps         = src.bineps;
    bintrainclass  = src.bintrainclass;
    bintraintarg   = src.bintraintarg;
    binepsclass    = src.binepsclass;
    binepsweight   = src.binepsweight;
    binCclass      = src.binCclass;
    binCweight     = src.binCweight;
    binCweightfuzz = src.binCweightfuzz;
    binNnc         = src.binNnc;

    return;
}

#endif
