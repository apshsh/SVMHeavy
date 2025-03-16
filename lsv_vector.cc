
//
// LS-SVM vector class
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "lsv_vector.hpp"



LSV_Vector::LSV_Vector() : LSV_Generic_Deref()
{
    QQQ.setQuadraticCost();
    QQQ.seteps(0.0);
    QQQ.fudgeOn();

    return;
}

LSV_Vector::LSV_Vector(const LSV_Vector &src) : LSV_Generic_Deref()
{
    QQQ.setQuadraticCost();
    QQQ.seteps(0.0);
    QQQ.fudgeOn();

    assign(src,0);

    return;
}

LSV_Vector::LSV_Vector(const LSV_Vector &src, const ML_Base *srcx) : LSV_Generic_Deref()
{
    setaltx(srcx);

    QQQ.setQuadraticCost();
    QQQ.seteps(0.0);
    QQQ.fudgeOn();

    assign(src,-1);

    return;
}

LSV_Vector::~LSV_Vector()
{
    return;
}

std::ostream &LSV_Vector::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "Vector-valued LS-SVM\n";

    LSV_Generic_Deref::printstream(output,dep+1);

    return output;
}

std::istream &LSV_Vector::inputstream(std::istream &input )
{
    LSV_Generic_Deref::inputstream(input);

    return input;
}

