
//
// Dynamic array class.  Features:
//
// Version: 6
// Date: 05/12/2014
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//
// - can be resized dynamically at any time to any size without loss of data.
// - elements can be accessed either by reference, using standard ("&",)
//   operations, or by value, using the () operator.  Access by value
//   is a const operation.
// - uses flexible over-allocation to allow for repeated size modification
//   without excessive performance hit copying data.
//




#include "dynarray.hpp"
//#ifdef ENABLE_THREADS
//#include <mutex>
//#endif

#define MAXOLDCONTENT 1000000

template <>
void DynArray<int>::locresize(int size, int suggestedallocsize, const int *fillval, bool leavemem)
{
    // NB: THERE IS A SPECIALISED AND NON-SPECIALISED VERSION, SO UPDATE BOTH!

    bool locarrayhold = ( suggestedallocsize == -3 ) ? true : array_hold();
    suggestedallocsize = ( suggestedallocsize == -3 ) ? -1 : suggestedallocsize;

    NiceAssert( suggestedallocsize && ( size >= 0 ) && ( ( suggestedallocsize >= size ) || ( suggestedallocsize == -1 ) || ( suggestedallocsize == -2 ) ) );

    if ( suggestedallocsize > 0 )
    {
        holdalloc = true;
        // but don't reset if -1, as this is probably just a standard
        // resize call.

        if ( size > suggestedallocsize )
        {
            suggestedallocsize = size;
        }
    }

    int stepsizedown   = (int) (((double) allocsize)/DOWNSIZEFRAC);
    int stepsizeup     = (int) (ALLOCAHEADFRAC*size);
    int stepsizeupclip = ( stepsizeup > MINSIZE ) ? stepsizeup : MINSIZE;

    //if ( !allocsize && size ) - actually, you need to test content != nullptr to allow for enZeroExt
    if ( !dyncontent && size )
    {
        // JIT allocation occurs here
        // On first run we just allocate the size requested - no alloc
        // ahead - as there is a strong possibility that this could
        // possibly be a fixed size array.

        dsize     = size;
        allocsize = ( ( suggestedallocsize == -1 ) || ( suggestedallocsize == -2 ) ) ? size : suggestedallocsize;

        NiceAssert( allocsize >= 0 );

        MEMNEWARRAY(dyncontent,int,allocsize+1);

        setzero(DEREFMEMARRAY(dyncontent,0));
//        setzero(dyncontent[0]);

        NiceAssert(dyncontent);
        #ifndef IGNOREMEM
        memcount((allocsize*sizeof(int)),+1);
        #endif

        if ( fillval )
        {
            for ( int i = 0 ; i < allocsize ; ++i )
            {
                DEREFMEMARRAY(dyncontent,i+1) = *fillval;
//                dyncontent[i+1] = *fillval;
            }
        }

        else if ( suggestedallocsize == -2 )
        {
            for ( int i = 0 ; i < allocsize ; ++i )
            {
                DEREFMEMARRAY(dyncontent,i+1) = i;
//                dyncontent[i+1] = i;
            }
        }
    }

    else if ( ( suggestedallocsize >= 0              ) ||
              ( size > allocsize                     ) ||
              (  array_norm()                     &&
                !locarrayhold                     &&
                ( size           < stepsizedown ) &&
                ( stepsizeupclip < stepsizedown )    ) ||
              (  array_tight()                    &&
                !locarrayhold                     &&
                ( size           < allocsize    )    )
            )
    {
        // Resize array

        // NB: this is called (in zerointarry et al) in a multi-threaded
        // context.  This is not ideal, but only occurs when extending the
        // array.  So long as we take care to ensure that all variables are
        // ready to go before over-writing then it should (fingers crossed)
        // work OK.

        int *newcontent = nullptr;

        int newdsize      = dsize;
        int newallocsize  = allocsize;

        bool newholdalloc  = holdalloc;

        if ( size > suggestedallocsize )
        {
            // prealloc bound broken
            newholdalloc = false;
            suggestedallocsize = ( suggestedallocsize == -2 ) ? -2 : -1;
        }

        #ifndef IGNOREMEM
        int oldallocsize = allocsize;
        #endif

        int *oldcontent = dyncontent;

        int copysize;
        int modallocsize;

        copysize     = ( size < newdsize ) ? size : newdsize;
        modallocsize = array_tight() ? size : stepsizeupclip;
        newallocsize = ( ( suggestedallocsize == -1 ) || ( suggestedallocsize == -2 ) ) ? modallocsize : suggestedallocsize;

        newdsize = size;

        NiceAssert( newallocsize >= newdsize );
        NiceAssert( newallocsize >= 0        );
        NiceAssert( newdsize     >= 0        );

        MEMNEWARRAY(newcontent,int,newallocsize+1);

        setzero(DEREFMEMARRAY(newcontent,0));
//        setzero(newcontent[0]);

        NiceAssert(newcontent);
        #ifndef IGNOREMEM
        memcount((newallocsize*sizeof(int)),+1);
        #endif

        if ( fillval )
        {
            for ( int i = 0 ; i < newallocsize ; ++i )
            {
                DEREFMEMARRAY(newcontent,i+1) = *fillval;
//                newcontent[i+1] = *fillval;
            }
        }

        else if ( suggestedallocsize == -2 )
        {
            for ( int i = 0 ; i < newallocsize ; ++i )
            {
                DEREFMEMARRAY(newcontent,i+1) = i;
//                newcontent[i+1] = i;
            }
        }

        else if ( copysize+((int) enZeroExt) && oldcontent )
        {
            for ( int i = 0 ; i < copysize ; ++i )
            {
                // NB: for simple types like this it is actually faster to copy!

                DEREFMEMARRAY(newcontent,i+1) = DEREFMEMARRAY(oldcontent,i+1);
//                newcontent[i+1] = oldcontent[i+1];
            }
        }

//NB: this is always called with leavemem true, so it shouldn't really matter is this gets disordered
//
//#ifdef ENABLE_THREADS
//        if ( suggestedallocsize == -2 )
//        {
//            static std::mutex eyelock;
//            eyelock.lock();
//
//            content   = newcontent;
//            dsize     = newdsize;
//            allocsize = newallocsize;
//            holdalloc = newholdalloc;
//
//            eyelock.unlock();
//        }
//
//        else
//#endif
        {
            dyncontent = newcontent;
            dsize      = newdsize;
            allocsize  = newallocsize;
            holdalloc  = newholdalloc;
        }

        if ( !notdelcontent && !leavemem && oldcontent )
        {
            MEMDELARRAY(oldcontent); oldcontent = nullptr;
        }

        oldcontent = nullptr;

        #ifndef IGNOREMEM
        memcount((oldallocsize*sizeof(int)),-1);
        #endif
    }

    else
    {
        dsize = size;
    }
}




