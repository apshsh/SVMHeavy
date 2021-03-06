
//
// Simple function block
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
#include "blk_usrfnb.h"


BLK_UsrFnB::BLK_UsrFnB(int isIndPrune) : BLK_Generic(isIndPrune)
{
    setaltx(NULL);

    return;
}

BLK_UsrFnB::BLK_UsrFnB(const BLK_UsrFnB &src, int isIndPrune) : BLK_Generic(isIndPrune)
{
    setaltx(NULL);

    assign(src,0);

    return;
}

BLK_UsrFnB::BLK_UsrFnB(const BLK_UsrFnB &src, const ML_Base *xsrc, int isIndPrune) : BLK_Generic(isIndPrune)
{
    setaltx(xsrc);

    assign(src,0);

    return;
}

BLK_UsrFnB::~BLK_UsrFnB()
{
    return;
}

std::ostream &BLK_UsrFnB::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "User wrapper block\n";

    return BLK_Generic::printstream(output,dep+1);
}

std::istream &BLK_UsrFnB::inputstream(std::istream &input )
{
    return BLK_Generic::inputstream(input);
}





































int BLK_UsrFnB::ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg, gentype ***pxyprodi) const
{
    (void) retaltg;
    (void) pxyprodi;

    retVector<gentype> tmpva;

    gentype xarg(x(i)(tmpva));

    resg = outfn()(xarg);
    resg.finalise(2);
    resg.finalise(1);
    resg.finalise();
    resh = resg;

    return 0;
}

