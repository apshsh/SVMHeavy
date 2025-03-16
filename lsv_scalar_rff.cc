
//
// LS-SVM scalar RFF class
//
// Version: 7
// Date: 13/11/2022
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "lsv_scalar_rff.hpp"



LSV_Scalar_rff::LSV_Scalar_rff() : LSV_Generic_Deref()
{
    QQQ.setinAdam(0);
    QQQ.settunev(0);
    QQQ.setReOnly(0);

    return;
}

LSV_Scalar_rff::LSV_Scalar_rff(const LSV_Scalar_rff &src) : LSV_Generic_Deref()
{
    QQQ.setinAdam(0);
    QQQ.settunev(0);
    QQQ.setReOnly(0);

    assign(src,0);

    return;
}

LSV_Scalar_rff::LSV_Scalar_rff(const LSV_Scalar_rff &src, const ML_Base *srcx) : LSV_Generic_Deref()
{
    setaltx(srcx);

    QQQ.setinAdam(0);
    QQQ.settunev(0);
    QQQ.setReOnly(0);

    assign(src,-1);

    return;
}

LSV_Scalar_rff::~LSV_Scalar_rff()
{
    return;
}

std::ostream &LSV_Scalar_rff::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Scalar RFF LS-SVM\n";

    LSV_Generic_Deref::printstream(output,dep+1);

    return output;
}

std::istream &LSV_Scalar_rff::inputstream(std::istream &input )
{
    LSV_Generic_Deref::inputstream(input);

    return input;
}

