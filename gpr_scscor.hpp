
//
// Scalar regression with ranking GP
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

#ifndef _gpr_scscor_h
#define _gpr_scscor_h

#include "gpr_generic.hpp"
#include "lsv_scscor.hpp"




class GPR_ScScor;

std::ostream &operator<<(std::ostream &output, const GPR_ScScor &src );
std::istream &operator>>(std::istream &input,        GPR_ScScor &dest);

// Swap and zeroing (restarting) functions

inline void qswap(GPR_ScScor &a, GPR_ScScor &b);
inline GPR_ScScor &setzero(GPR_ScScor &a);

class GPR_ScScor : public GPR_Generic
{
public:

    // Constructors, destructors, assignment etc..

    GPR_ScScor();
    GPR_ScScor(const GPR_ScScor &src);
    GPR_ScScor(const GPR_ScScor &src, const ML_Base *srcx);
    GPR_ScScor &operator=(const GPR_ScScor &src) { assign(src); return *this; }
    virtual ~GPR_ScScor() { return; }

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0) override;
    virtual void semicopy(const ML_Base &src) override;
    virtual void qswapinternal(ML_Base &b) override;

    virtual std::ostream &printstream(std::ostream &output) const override;
    virtual std::istream &inputstream(std::istream &input ) override;


    // ================================================================
    //     Common functions for all ML
    // ================================================================

    // Information functions

    virtual int type(void)    const override { return 405; }
    virtual int subtype(void) const override { return 0;   }

    virtual int isClassifier(void) const override { return 0; }



    // ================================================================
    //     Common functions for all GPs
    // ================================================================

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

    LSV_ScScor QQ;
};

inline double norm2(const GPR_ScScor &a);
inline double abs2 (const GPR_ScScor &a);

inline double norm2(const GPR_ScScor &a) { return a.RKHSnorm(); }
inline double abs2 (const GPR_ScScor &a) { return a.RKHSabs();  }

inline void qswap(GPR_ScScor &a, GPR_ScScor &b)
{
    a.qswapinternal(b);

    return;
}

inline GPR_ScScor &setzero(GPR_ScScor &a)
{
    a.restart();

    return a;
}

inline void GPR_ScScor::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    GPR_ScScor &b = dynamic_cast<GPR_ScScor &>(bb.getML());

    GPR_Generic::qswapinternal(b);

    qswap(getQQ(),b.getQQ());

    return;
}

inline void GPR_ScScor::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const GPR_ScScor &b = dynamic_cast<const GPR_ScScor &>(bb.getMLconst());

    GPR_Generic::semicopy(b);

    getQQ().semicopy(b.getQQconst());

    return;
}

inline void GPR_ScScor::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const GPR_ScScor &src = dynamic_cast<const GPR_ScScor &>(bb.getMLconst());

    GPR_Generic::assign(src,onlySemiCopy);
    getQQ().assign(src.getQQconst(),onlySemiCopy);

    return;
}

#endif
