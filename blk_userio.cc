
//
// User I/O Function
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
#include "blk_userio.hpp"


std::ostream &BLK_UserIO::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "User wrapper block\n";

    return BLK_Generic::printstream(output,dep+1);
}

std::istream &BLK_UserIO::inputstream(std::istream &input )
{
    return BLK_Generic::inputstream(input);
}





































int BLK_UserIO::ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg, gentype ***pxyprodi) const
{
    NiceAssert( !retaltg );

    (void) retaltg;
    (void) pxyprodi;

    userostream() << "g(" << x(i) << ") = ";
    useristream() >> resg;

    resh = resg;

    return 0;
}

