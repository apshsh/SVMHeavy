
//
// Vector regression Type-II multi-layer kernel-machine
//
// Version: 7
// Date: 07/07/2018
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "mlm_vector.hpp"

// Boilerplate

MLM_Vector::MLM_Vector() : MLM_Generic()
{
    QQQ.setredbin();

    fixMLTree();

    return;
}

MLM_Vector::MLM_Vector(const MLM_Vector &src) : MLM_Generic()
{
    QQQ.setredbin();

    fixMLTree();

    QQQ.setredbin();

    assign(src,0);

    return;
}

MLM_Vector::MLM_Vector(const MLM_Vector &src, const ML_Base *srcx) : MLM_Generic()
{
    QQQ.setredbin();

    fixMLTree();

    setaltx(srcx);
    assign(src,-1);

    return;
}

std::ostream &MLM_Vector::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Vector MLM\n\n";

    MLM_Generic::printstream(output,dep+1);

    return output;
}

std::istream &MLM_Vector::inputstream(std::istream &input )
{
    MLM_Generic::inputstream(input);

    return input;
}


// Actual stuff

int MLM_Vector::train(int &res, svmvolatile int &killSwitch)
{
    QQQ.train(res,killSwitch);

    xistrained = 1;

    return res;
}

