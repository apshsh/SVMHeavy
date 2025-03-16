//FIXME: need to set K order here
//
// Vector (at once) regression SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _svm_vector_matonce_h
#define _svm_vector_matonce_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "svm_vector_atonce_template.hpp"

double evalKSVM_Vector_Matonce(int i, int j, const gentype **pxyprod, const void *owner);
double evalSigmaSVM_Vector_Matonce(int i, int j, const gentype **pxyprod, const void *owner);



class SVM_Vector_Matonce;


// Swap function

void qswap(SVM_Vector_Matonce &a, SVM_Vector_Matonce &b);


class SVM_Vector_Matonce : public SVM_Vector_atonce_temp<Matrix<double> >
{
public:

    // Constructors, destructors, assignment operators and similar

    SVM_Vector_Matonce();
    SVM_Vector_Matonce(const SVM_Vector_Matonce &src);
    SVM_Vector_Matonce(const SVM_Vector_Matonce &src, const ML_Base *xsrc);
    SVM_Vector_Matonce &operator=(const SVM_Vector_Matonce &src) { assign(src); return *this; }
    virtual ~SVM_Vector_Matonce();

    virtual std::ostream &printstream(std::ostream &output, int dep) const override;
    virtual std::istream &inputstream(std::istream &input) override;

    // Not-kosher Gp return - all matrices globbed together

    virtual const Matrix<double> &Gp(void) const
    {
        Matrix<double> &Gpp = GpRepresent;

        Gpp.resize(N()*tspaceDim(),N()*tspaceDim());

        int i,j;

        retMatrix<double> tmpma;

        for ( i = 0 ; i < N() ; ++i )
        {
            for ( j = 0 ; j < N() ; ++j )
            {
                Gpp("&",i*tspaceDim(),1,((i+1)*tspaceDim())-1,j*tspaceDim(),1,((j+1)*tspaceDim())-1,tmpma) = getGpGradTemp()(i,j);
            }
        }

        return Gpp;
    }

    // Other functions

    virtual void assign(const ML_Base &src, int isOnlySemi = 0) override;
    virtual void semicopy(const ML_Base &src) override;
    virtual void qswapinternal(ML_Base &b) override;

    virtual int subtype(void)    const override { return 2;              }

private:
    mutable Matrix<double> GpRepresent; // used in Gp function
};

inline double norm2(const SVM_Vector_Matonce &a);
inline double abs2 (const SVM_Vector_Matonce &a);

inline double norm2(const SVM_Vector_Matonce &a) { return a.RKHSnorm(); }
inline double abs2 (const SVM_Vector_Matonce &a) { return a.RKHSabs();  }

inline void qswap(SVM_Vector_Matonce &a, SVM_Vector_Matonce &b)
{
    a.qswapinternal(b);

    return;
}

inline void SVM_Vector_Matonce::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    SVM_Vector_Matonce &b = dynamic_cast<SVM_Vector_Matonce &>(bb.getML());

    SVM_Vector_atonce_temp<Matrix<double> >::qswapinternal(b);

    return;
}

inline void SVM_Vector_Matonce::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const SVM_Vector_Matonce &b = dynamic_cast<const SVM_Vector_Matonce &>(bb.getMLconst());

    SVM_Vector_atonce_temp<Matrix<double> >::semicopy(b);

    return;
}

inline void SVM_Vector_Matonce::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const SVM_Vector_Matonce &src = dynamic_cast<const SVM_Vector_Matonce &>(bb.getMLconst());

    SVM_Vector_atonce_temp<Matrix<double> >::assign(src,onlySemiCopy);

    return;
}

#endif
