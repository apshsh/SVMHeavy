
//
// Scalar regression with scoring LSV
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "lsv_scscor.hpp"

LSV_ScScor::LSV_ScScor() : LSV_Scalar()
{
    QQ.setQuadraticCost();
    QQ.seteps(0.0);
    QQ.fudgeOn();

    setaltx(nullptr);

    return;
}

LSV_ScScor::LSV_ScScor(const LSV_ScScor &src) : LSV_Scalar()
{
    QQ.setQuadraticCost();
    QQ.seteps(0.0);
    QQ.fudgeOn();

    setaltx(nullptr);
    assign(src,0);

    return;
}

LSV_ScScor::LSV_ScScor(const LSV_ScScor &src, const ML_Base *srcx) : LSV_Scalar()
{
    QQ.setQuadraticCost();
    QQ.seteps(0.0);
    QQ.fudgeOn();

    setaltx(srcx);
    assign(src,-1);

    return;
}

std::ostream &LSV_ScScor::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "LSV Vector Scoring\n";

    repPrint(output,'>',dep) << "Underlying LSV: " << QQ << "\n";

    LSV_Scalar::printstream(output,dep+1);

    return output;
}

std::istream &LSV_ScScor::inputstream(std::istream &input )
{
    wait_dummy dummy;

    input >> dummy; input >> QQ;

    LSV_Scalar::inputstream(input);

    return input;
}

