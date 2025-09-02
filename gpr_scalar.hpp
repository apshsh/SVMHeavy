
//
// Scalar regression GP
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

//
// Inequalities here are enforced using EP (see Rasmussen Ras2) - that is, we
// approximate a logistic prior with a Gaussian.  The distinctions are that
// we allow for general m in logistic (see section 3.9.  Ras2 basically sets
// m = 0, but we let m = y(i)) and let the user set the steepness (nu) in the
// logistic (where Ras2 just sets nu = 1).  Ras2 is fine for classification, 
// but because we want to enforce regression inequalities more strictly we
// recquire nu = 0.1 default here (it gets defaulted back to 1 in GPR_Binary).
//

#ifndef _gpr_scalar_h
#define _gpr_scalar_h

#include "gpr_generic.hpp"
#include "lsv_scalar.hpp"




class GPR_Scalar;

// Swap and zeroing (restarting) functions

inline void qswap(GPR_Scalar &a, GPR_Scalar &b);
inline GPR_Scalar &setzero(GPR_Scalar &a);

class GPR_Scalar : public GPR_Generic
{
public:

    // Constructors, destructors, assignment etc..

    GPR_Scalar();
    GPR_Scalar(const GPR_Scalar &src);
    GPR_Scalar(const GPR_Scalar &src, const ML_Base *srcx);
    GPR_Scalar &operator=(const GPR_Scalar &src) { assign(src); return *this; }
    virtual ~GPR_Scalar() { return; }

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0) override;
    virtual void semicopy(const ML_Base &src) override;
    virtual void qswapinternal(ML_Base &b) override;

    virtual std::ostream &printstream(std::ostream &output, int dep) const override;
    virtual std::istream &inputstream(std::istream &input ) override;


    // ================================================================
    //     Common functions for all ML
    // ================================================================

    // Information functions

    virtual int type(void)    const override { return 400; }
    virtual int subtype(void) const override { return 0;   }

    virtual int isClassifier(void) const override { return 0; }

    // Training functions - we need to overwrite this enable EP if inequality constraints are present

    virtual int train(int &res)                              override { svmvolatile int killSwitch = 0; return train(res,killSwitch); }
    virtual int train(int &res, svmvolatile int &killSwitch) override;



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

    virtual int isNaiveConst  (void) const override { return xNaiveConst;                                                          }
    virtual int isEPConst     (void) const override { return ( ( xNaiveConst == 0 ) && ( xEPorLaplace == 0 ) ) ? 1            : 0; }
    virtual int isLaplaceConst(void) const override { return ( ( xNaiveConst == 0 ) && ( xEPorLaplace != 0 ) ) ? xEPorLaplace : 0; }

    virtual int setNaiveConst  (void        ) override;
    virtual int setEPConst     (void        ) override;
    virtual int setLaplaceConst(int type = 1) override;


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

    int xNaiveConst;  // 1 nothing, 0 EP or Laplace
    int xEPorLaplace; // 0 for EP, 1 for Laplace with Gaussian CDF, 2 for Laplace with logistic
    LSV_Scalar QQ;
};

inline double norm2(const GPR_Scalar &a);
inline double abs2 (const GPR_Scalar &a);

inline double norm2(const GPR_Scalar &a) { return a.RKHSnorm(); }
inline double abs2 (const GPR_Scalar &a) { return a.RKHSabs();  }

inline void qswap(GPR_Scalar &a, GPR_Scalar &b)
{
    a.qswapinternal(b);

    return;
}

inline GPR_Scalar &setzero(GPR_Scalar &a)
{
    a.restart();

    return a;
}

inline void GPR_Scalar::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    GPR_Scalar &b = dynamic_cast<GPR_Scalar &>(bb.getML());

    GPR_Generic::qswapinternal(b);

    qswap(getQQ(),     b.getQQ()     );
    qswap(xNaiveConst, b.xNaiveConst );
    qswap(xEPorLaplace,b.xEPorLaplace);

    return;
}

inline void GPR_Scalar::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const GPR_Scalar &b = dynamic_cast<const GPR_Scalar &>(bb.getMLconst());

    GPR_Generic::semicopy(b);

    getQQ().semicopy(b.getQQconst());

    xNaiveConst  = b.xNaiveConst;
    xEPorLaplace = b.xEPorLaplace;

    return;
}

inline void GPR_Scalar::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const GPR_Scalar &src = dynamic_cast<const GPR_Scalar &>(bb.getMLconst());

    GPR_Generic::assign(src,onlySemiCopy);
    getQQ().assign(src.getQQconst(),onlySemiCopy);

    xNaiveConst  = src.xNaiveConst;
    xEPorLaplace = src.xEPorLaplace;

    return;
}

#endif
