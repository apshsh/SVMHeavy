
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
#include "blk_usrfnb.hpp"


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
    NiceAssert( !retaltg );

    (void) retaltg;
    (void) pxyprodi;

    const SparseVector<gentype> &xx = x(i);

    NiceAssert ( !xx.isf2offindpresent() );
    NiceAssert ( !xx.isf3offindpresent() );
    NiceAssert ( !xx.isf4offindpresent() );

    NiceAssert ( xx.nupsize () <= 1 );
    NiceAssert ( xx.f1upsize() <= 1 );

    {
        const SparseVector<gentype> &xxx = xx.nup(0);

        retVector<gentype> tmpva;

        gentype xarg(xxx(tmpva));

        resg = outfn()(xarg);
        resg.finalise(2);
        resg.finalise(1);
        resg.finalise();
    }

    if ( xx.isf1offindpresent() )
    {
        const SparseVector<gentype> &xxx = xx.f1up(0);

        retVector<gentype> tmpva;

        gentype xarg(xxx(tmpva));

        gentype resgoff;

        resgoff = outfn()(xarg);
        resgoff.finalise(2);
        resgoff.finalise(1);
        resgoff.finalise();

        resg -= resgoff;
    }

    resh = resg;

    return 0;
}

