
//
// Do nothing block
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
#include "blk_nopnop.hpp"


std::ostream &BLK_Nopnop::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "No-op wrapper block\n";

    return BLK_Generic::printstream(output,dep+1);
}

std::istream &BLK_Nopnop::inputstream(std::istream &input )
{
    return BLK_Generic::inputstream(input);
}
