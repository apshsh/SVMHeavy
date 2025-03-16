
//
// Vector regression GP
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

//
// Currently this is basically a wrap-around for a LS-SVR with C mapped to
// 1/sigma for noise regularisation.  This is equivalent to the standard
// GP regressor assuming Gaussian measurement noise.
//

#ifndef _gpr_vector_h
#define _gpr_vector_h

#include "gpr_generic.hpp"
#include "lsv_vector.hpp"




class GPR_Vector;

// Swap and zeroing (restarting) functions

inline void qswap(GPR_Vector &a, GPR_Vector &b);
inline GPR_Vector &setzero(GPR_Vector &a);

class GPR_Vector : public GPR_Generic
{
public:

    // Constructors, destructors, assignment etc..

    GPR_Vector();
    GPR_Vector(const GPR_Vector &src);
    GPR_Vector(const GPR_Vector &src, const ML_Base *srcx);
    GPR_Vector &operator=(const GPR_Vector &src) { assign(src); return *this; }
    virtual ~GPR_Vector() { return; }

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0) override;
    virtual void semicopy(const ML_Base &src) override;
    virtual void qswapinternal(ML_Base &b) override;

    virtual std::ostream &printstream(std::ostream &output, int dep) const override;
    virtual std::istream &inputstream(std::istream &input ) override;


    // ================================================================
    //     Common functions for all ML
    // ================================================================

    // Information functions

    virtual int type(void)    const override { return 401; }
    virtual int subtype(void) const override { return 0;   }

    virtual int isClassifier(void) const override { return 0; }



    // ================================================================
    //     Common functions for all GPs
    // ================================================================

    // Constraint handling inheritted from LSV

    virtual int isNaiveConst(void) const override { return 1; }
    virtual int isEPConst   (void) const override { return 0; }

    virtual int setNaiveConst(void) override {                                               return 0; }
    virtual int setEPConst   (void) override { NiceThrow("EP not available for gpr_vector"); return 0; }

    // Information functions (training data):

    virtual       GPR_Generic &getGPR(void)            override { return *this; }
    virtual const GPR_Generic &getGPRconst(void) const override { return *this; }

    // General modification and autoset functions

    virtual       ML_Base &getML(void)            override { return static_cast<      ML_Base &>(getGPR());      }
    virtual const ML_Base &getMLconst(void) const override { return static_cast<const ML_Base &>(getGPRconst()); }



    // Base-level stuff

    virtual       LSV_Generic &getQQ(void)            override { return QQ; }
    virtual const LSV_Generic &getQQconst(void) const override { return QQ; }




private:

    LSV_Vector QQ;
};

inline double norm2(const GPR_Vector &a);
inline double abs2 (const GPR_Vector &a);

inline double norm2(const GPR_Vector &a) { return a.RKHSnorm(); }
inline double abs2 (const GPR_Vector &a) { return a.RKHSabs();  }

inline void qswap(GPR_Vector &a, GPR_Vector &b)
{
    a.qswapinternal(b);

    return;
}

inline GPR_Vector &setzero(GPR_Vector &a)
{
    a.restart();

    return a;
}

inline void GPR_Vector::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    GPR_Vector &b = dynamic_cast<GPR_Vector &>(bb.getML());

    GPR_Generic::qswapinternal(b);

    qswap(getQQ(),b.getQQ());

    return;
}

inline void GPR_Vector::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const GPR_Vector &b = dynamic_cast<const GPR_Vector &>(bb.getMLconst());

    GPR_Generic::semicopy(b);

    getQQ().semicopy(b.getQQconst());

    return;
}

inline void GPR_Vector::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const GPR_Vector &src = dynamic_cast<const GPR_Vector &>(bb.getMLconst());

    GPR_Generic::assign(src,onlySemiCopy);
    getQQ().assign(src.getQQconst(),onlySemiCopy);

    return;
}

#endif
