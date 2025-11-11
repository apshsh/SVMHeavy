
//
// Scalar regression GP (RFF)
//
// Version: 7
// Date: 29/11/2022
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

// See notes in gpr_scalar.h, this is more or less the same.

#ifndef _gpr_scalar_rff_h
#define _gpr_scalar_rff_h

#include "gpr_generic.hpp"
#include "lsv_scalar_rff.hpp"




class GPR_Scalar_rff;

// Swap and zeroing (restarting) functions

inline void qswap(GPR_Scalar_rff &a, GPR_Scalar_rff &b);
inline GPR_Scalar_rff &setzero(GPR_Scalar_rff &a);

class GPR_Scalar_rff : public GPR_Generic
{
public:

    // Constructors, destructors, assignment etc..

    GPR_Scalar_rff();
    GPR_Scalar_rff(const GPR_Scalar_rff &src);
    GPR_Scalar_rff(const GPR_Scalar_rff &src, const ML_Base *srcx);
    GPR_Scalar_rff &operator=(const GPR_Scalar_rff &src) { assign(src); return *this; }
    virtual ~GPR_Scalar_rff() { return; }

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0) override;
    virtual void semicopy(const ML_Base &src) override;
    virtual void qswapinternal(ML_Base &b) override;

    virtual std::ostream &printstream(std::ostream &output, int dep) const override;
    virtual std::istream &inputstream(std::istream &input ) override;


    // ================================================================
    //     Common functions for all ML
    // ================================================================

    // Information functions

    virtual int type(void)    const override { return 410; }
    virtual int subtype(void) const override { return 0;   }

    virtual int isClassifier(void) const override { return 0; }

    // Training functions - we need to overwrite this enable EP if inequality constraints are present

    virtual int train(int &res)                              override { svmvolatile int killSwitch = 0; return train(res,killSwitch); }
    virtual int train(int &res, svmvolatile int &killSwitch) override;

    // Evaluation Functions:

    virtual int gg(               gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = nullptr) const override { return GPR_Generic::gg(     resg,i,retaltg,pxyprodi); }
    virtual int hh(gentype &resh,                int i,                  gentype ***pxyprodi = nullptr) const override { return GPR_Generic::hh(resh,     i,        pxyprodi); }
    virtual int gh(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = nullptr) const override { return GPR_Generic::gh(resh,resg,i,retaltg,pxyprodi); }

    virtual int gg(               gentype &resg, const SparseVector<gentype> &x,                  const vecInfo *xinf = nullptr, gentype ***pxyprodx = nullptr) const override;
    virtual int hh(gentype &resh,                const SparseVector<gentype> &x,                  const vecInfo *xinf = nullptr, gentype ***pxyprodx = nullptr) const override;
    virtual int gh(gentype &resh, gentype &resg, const SparseVector<gentype> &x, int retaltg = 0, const vecInfo *xinf = nullptr, gentype ***pxyprodx = nullptr) const override;



    // ================================================================
    //     Common functions for all GPs
    // ================================================================

    // Information functions (training data):

    virtual       GPR_Generic &getGPR(void)            override { return *this; }
    virtual const GPR_Generic &getGPRconst(void) const override { return *this; }

    // General modification and autoset functions

    virtual       ML_Base &getML(void)            override { return static_cast<      ML_Base &>(getGPR());      }
    virtual const ML_Base &getMLconst(void) const override { return static_cast<const ML_Base &>(getGPRconst()); }

    // Constraint handling inheritted from LSV
    //
    // naiveConstraints: if set then rather than using expectation propogation
    //       for inequality constraints and classification we instead use the
    //       naive method inheritted from SVM theory.

    virtual int isNaiveConst(void) const override { return xNaiveConst;  }
    virtual int isEPConst   (void) const override { return !xNaiveConst; }

    virtual int setNaiveConst(void) override;
    virtual int setEPConst   (void) override;


    // Base-level stuff

    virtual       LSV_Generic &getQQ(void)            override { return QQ; }
    virtual const LSV_Generic &getQQconst(void) const override { return QQ; }




private:

    int getQsetsigmaweight(Vector<int> i, Vector<double> nv)
    {
        NiceAssert( i.size() == nv.size() );

        if ( i.size() )
        {
            int j;

            for ( j = 0 ; j < i.size() ; ++j )
            {
                getQQ().setCweight(i(j),1/nv(j));
            }
        }

        return 1;
    }

    int xNaiveConst;
    LSV_Scalar_rff QQ;
};

inline double norm2(const GPR_Scalar_rff &a);
inline double abs2 (const GPR_Scalar_rff &a);

inline double norm2(const GPR_Scalar_rff &a) { return a.RKHSnorm(); }
inline double abs2 (const GPR_Scalar_rff &a) { return a.RKHSabs();  }

inline void qswap(GPR_Scalar_rff &a, GPR_Scalar_rff &b)
{
    a.qswapinternal(b);

    return;
}

inline GPR_Scalar_rff &setzero(GPR_Scalar_rff &a)
{
    a.restart();

    return a;
}

inline void GPR_Scalar_rff::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    GPR_Scalar_rff &b = dynamic_cast<GPR_Scalar_rff &>(bb.getML());

    GPR_Generic::qswapinternal(b);

    qswap(getQQ(),    b.getQQ()    );
    qswap(xNaiveConst,b.xNaiveConst);

    return;
}

inline void GPR_Scalar_rff::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const GPR_Scalar_rff &b = dynamic_cast<const GPR_Scalar_rff &>(bb.getMLconst());

    GPR_Generic::semicopy(b);

    getQQ().semicopy(b.getQQconst());

    xNaiveConst = b.xNaiveConst;

    return;
}

inline void GPR_Scalar_rff::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const GPR_Scalar_rff &src = dynamic_cast<const GPR_Scalar_rff &>(bb.getMLconst());

    GPR_Generic::assign(src,onlySemiCopy);
    getQQ().assign(src.getQQconst(),onlySemiCopy);

    xNaiveConst = src.xNaiveConst;

    return;
}

#endif
