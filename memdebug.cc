
//
// Debugger version of new and delete operators.  These allow you do
// detect and report simple memory errors (double-deletion of memory
// etc) with informative errors.
//
// Version: split off basefn
// Date: 11/09/2024
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "memdebug.hpp"


#ifdef DEBUG_MEM_CHEAP
size_t _global_alloccnt = 0;
size_t _global_maxalloccnt = 0;
#endif


#ifdef DEBUG_MEM
const char *updatedesc(const char *addendum, size_t &oldallocdesclen)
{
    static char allocdesc[8192];
    static size_t allocdesclen = 0;

    if ( addendum )
    {
        size_t addlen = strlen(addendum);

        oldallocdesclen = allocdesclen;

        {
            for ( size_t i = 0 ; i < addlen ; ++i )
            {
                allocdesc[allocdesclen+i] = addendum[i];
            }
        }

        allocdesclen += addlen;
    }

    else
    {
        allocdesclen = oldallocdesclen;
    }

    allocdesc[allocdesclen] = '\0';

    return allocdesc;
}
#endif
