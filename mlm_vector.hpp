
//
// Vector regression Type-II multi-layer kernel-machine
//
// Version: 7
// Date: 07/07/2018
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

//
// Currently this is basically a wrap-around for a LS-SVR with C mapped to
// 1/sigma for noise regularisation.  This is equivalent to the standard
// GP regressor assuming Gaussian measurement noise.
//

#ifndef _mlm_vector_h
#define _mlm_vector_h

#include "mlm_generic.hpp"
#include "svm_vector.hpp"




class MLM_Vector;

// Swap and zeroing (restarting) functions

inline void qswap(MLM_Vector &a, MLM_Vector &b);
inline MLM_Vector &setzero(MLM_Vector &a);

class MLM_Vector : public MLM_Generic
{
public:

    // Constructors, destructors, assignment etc..

    MLM_Vector();
    MLM_Vector(const MLM_Vector &src);
    MLM_Vector(const MLM_Vector &src, const ML_Base *srcx);
    MLM_Vector &operator=(const MLM_Vector &src) { assign(src); return *this; }
    virtual ~MLM_Vector() { return; }

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0) override;
    virtual void semicopy(const ML_Base &src) override;
    virtual void qswapinternal(ML_Base &b) override;

    virtual std::ostream &printstream(std::ostream &output, int dep) const override;
    virtual std::istream &inputstream(std::istream &input ) override;


    // ================================================================
    //     Common functions for all ML
    // ================================================================

    // Information functions

    virtual int type(void)    const override { return 802; }
    virtual int subtype(void) const override { return 0;   }

    virtual int train(int &res, svmvolatile int &killSwitch) override;
    virtual int train(int &res) override { svmvolatile int killSwitch = 0; return train(res,killSwitch); }



    // ================================================================
    //     Common functions for all GPs
    // ================================================================

    // Information functions (training data):

    virtual       MLM_Generic &getMLM(void)            override { return *this; }
    virtual const MLM_Generic &getMLMconst(void) const override { return *this; }

    // General modification and autoset functions

    virtual       ML_Base &getML(void)            override { return static_cast<      ML_Base &>(getMLM());      }
    virtual const ML_Base &getMLconst(void) const override { return static_cast<const ML_Base &>(getMLMconst()); }



    // Base-level stuff

    virtual       SVM_Generic &getQQ(void)            override { return QQQ; }
    virtual const SVM_Generic &getQQconst(void) const override { return QQQ; }

private:

    SVM_Vector QQQ;
};

inline double norm2(const MLM_Vector &a);
inline double abs2 (const MLM_Vector &a);

inline double norm2(const MLM_Vector &a) { return a.RKHSnorm(); }
inline double abs2 (const MLM_Vector &a) { return a.RKHSabs();  }

inline void qswap(MLM_Vector &a, MLM_Vector &b)
{
    a.qswapinternal(b);

    return;
}

inline MLM_Vector &setzero(MLM_Vector &a)
{
    a.restart();

    return a;
}

inline void MLM_Vector::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    MLM_Vector &b = dynamic_cast<MLM_Vector &>(bb.getML());

    MLM_Generic::qswapinternal(b);

    qswap(getQQ(),b.getQQ());

    return;
}

inline void MLM_Vector::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const MLM_Vector &b = dynamic_cast<const MLM_Vector &>(bb.getMLconst());

    MLM_Generic::semicopy(b);

    getQQ().semicopy(b.getQQconst());

    return;
}

inline void MLM_Vector::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const MLM_Vector &src = dynamic_cast<const MLM_Vector &>(bb.getMLconst());

    MLM_Generic::assign(src,onlySemiCopy);
    getQQ().assign(src.getQQconst(),onlySemiCopy);

    if ( !onlySemiCopy )
    {
        fixMLTree();
    }

    return;
}

#endif
