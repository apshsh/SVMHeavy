
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

#ifndef _dynarray_h
#define _dynarray_h

#include <memory>
#include <cmath> // for the INFINITY macro
#include "qswapbase.hpp"
#include "memdebug.hpp"
#include "niceassert.hpp"
#ifndef IGNOREMEM
#include "basefn.hpp"
#endif

// 3/12/2015 was 1.2 for both fractions
#define MINSIZE        10
#define ALLOCAHEADFRAC 1.05
#define DOWNSIZEFRAC   2

template <class T> class DynArray;

// zerointarray: returns a pointer to an array ( 0 ).
// oneintarray:  returns a pointer to an array ( 1 ).
// cntintarray:  returns a pointer to an array ( 0 1 2 ... size-1 ).
//
// The result is fixed, though it's content may be extended as required.
// Note that the old contents are not deleted, just abandonded until exit.

inline const DynArray<int> *zerointarray(void);
inline const DynArray<int> *oneintarray (void);
inline const DynArray<int> *cntintarray (int size);

inline const DynArray<double> *zerodoublearray(void);
inline const DynArray<double> *onedoublearray (void);
inline const DynArray<double> *ninfdoublearray(void);
inline const DynArray<double> *pinfdoublearray(void);

//template <class T>
//class DynArray
template <class T>
struct DynArray
{
//public:

    // Constructors and Destructors

    ~DynArray()
    {
        if ( !notdelcontent && content )
        {
            MEMDELARRAY(content);
        }

        //content = nullptr;

        #ifndef IGNOREMEM
        memcount((allocsize*sizeof(T)),-1);
        #endif
    }

    // Access:
    //
    // - ("&",i) access returns a reference
    // - (i) access returns a value (const reference)
    // - v(i) access returns by value
    // - sv(i,x) set element i by value
    // - aref can access complete allocated range, not just dsize elements

    T &operator()(const char *, int i)
    {
        NiceAssert( content );
        NiceAssert( i >= -((int) enZeroExt) );
        NiceAssert( i <  dsize      );

        return content[i+1];
    }

    const T &operator()(int i) const
    {
        NiceAssert( content );
        NiceAssert( i >= -((int) enZeroExt) );
        NiceAssert( i <  dsize      );

        return content[i+1];
    }

    void set(int i, const T &src)
    {
        NiceAssert( content );
        NiceAssert( i >= -((int) enZeroExt) );
        NiceAssert( i <  dsize      );

        content[i+1] = src;
    }

    T v(int i) const
    {
        NiceAssert( content );
        NiceAssert( i >= -((int) enZeroExt) );
        NiceAssert( i <  dsize      );

        return content[i+1];
    }

    void sv(int i, T x) const
    {
        NiceAssert( content );
        NiceAssert( i >= -((int) enZeroExt) );
        NiceAssert( i <  dsize      );

        content[i+1] = x;
    }

    T &aref(int i)
    {
        NiceAssert( content );
        NiceAssert( i >= -((int) enZeroExt) );
        NiceAssert( i <  allocsize  );

        return content[i+1];
    }

    // Information
    //
    // size:  size of array
    // alloc: number of elements allocated to array (may exceed size)
    // hold:  true if pre-allocation has occured (expected size known)
    // tight: true if "tight" allocation is used (no allocation-ahead)
    // norm:  true if standard allocation strategy used
    // slack: true if "slack" allocation is used (no de-allocation when
    //        array shrinks, so array memory can only ever grow)
    // zeExt: true if zero-extend enabled (that is, an extra element
    //        at location -1 in the array that may be used for "zero"
    //        padding in vectors), false otherwise.

    int array_size (void) const { return dsize;     }
    int array_alloc(void) const { return allocsize; }

    bool array_hold (void) const { return holdalloc;       }
    bool array_norm (void) const { return tightalloc == 0; }
    bool array_tight(void) const { return tightalloc == 1; }
    bool array_slack(void) const { return tightalloc == 2; }
    bool array_zeExt(void) const { return enZeroExt;       }

    // Resize operation (add to or remove from end to achieve target size)
    //
    // suggestedallocsize is an optional argument that can be used to force
    // the dynamic array to pre-allocate a given amount of memory.  Set -1
    // for standard automatic operation instead.  Set fillval != nullptr to
    // assign all values as this.  Set -3 to force array_hold() behaviour.
    //
    // int only: - if you set suggestedallocsize == -2 then this sets
    //       contents to 0,1,2,... (-1 for "zero" element) and does
    //       rudimentary threaded memory safety.
    //     - If leavemem set then does not dealloc previous memory.

    DynArray<T> &resize(int size, int suggestedallocsize = -1, const T *fillval = nullptr, bool leavemem = false)
    {
        NiceAssert( !notdelcontent );

        locresize(size,suggestedallocsize,fillval,leavemem);

        return *this;
    }

    // Pre-allocate operation.  This is different that resize, as it simply
    // sets the amount of memory pre-allocated for the vector and not the
    // actual size of the vector.

    void prealloc(int newallocsize)
    {
        NiceAssert( !notdelcontent );
        NiceAssert( ( newallocsize == -1 ) || ( newallocsize >= 0 ) );

        if ( !newallocsize )
        {
            if ( dsize || enZeroExt )
            {
                // newallocsize is not allowed to be less than dsize

                newallocsize = dsize;
                holdalloc = true;
            }

            else
            {
                // requested newallocsize zero, no memory currently
                // allocated, so revert to unallocated state.  Reallocation
                // will occur JIT.

                if ( content )
                {
                    MEMDELARRAY(content);
                }

                content = nullptr;

                dsize     = 0;
                allocsize = 0;
                holdalloc = false;
            }
        }

        else if ( newallocsize > 0 )
        {
            // newallocsize is not allowed to be less than dsize

            newallocsize = ( dsize > newallocsize ) ? dsize : newallocsize;
            holdalloc = true;
        }

        else
        {
            NiceAssert( newallocsize == -1 );

            holdalloc = false;
        }

        if ( newallocsize )
        {
            // Allocation will be completed by resize call.

            resize(dsize,newallocsize);
        }
    }

    // applyOnAll: applies the given function to *all* elements allocated
    // here, including those that have been preallocated (but which are not
    // directly accessible to the user)

    void applyOnAll(void (*fn)(T &, int), int argx)
    {
        if ( allocsize+((int) enZeroExt) )
        {
            NiceAssert( content );

            for ( int i = -((int) enZeroExt) ; i < allocsize ; ++i )
            {
                fn(content[i+1],argx);
            }
        }
    }

    // Allocation strategy:
    //
    // Normally the dynamic array uses an alloc-ahead strategy that allocated
    // a certain fraction more than requested. This allows that array to grow
    // (and shrink) within these set bounds without needing time-expensive
    // re-allocation and copy operations.  An alternative strategy is tight
    // allocation which only allocations strictly what is required with no
    // preallocation.  This is more memory conserving, but may be slower
    // as every resize operation must then reallocate and copy contents

    void useStandardAllocation(void) { tightalloc = 0; }
    void useTightAllocation   (void) { tightalloc = 1; }
    void useSlackAllocation   (void) { tightalloc = 2; }

    // Zero extension:
    //
    // If enabled, zeroExtend allocates an additional element at index -1
    // in the array that may be used as a "zero padding" element in vectors
    // and such.  Disabled by default (though memory is allocated in any
    // case).

    void enZeroExtend(void)
    {
        NiceAssert( !notdelcontent );

        enZeroExt = true;

        if ( !content )
        {
            MEMNEWARRAY(content,T,allocsize+1);
            NiceAssert(content);
        }

        setzero(content[0]);
    }

    void noZeroExtend(void)
    {
        NiceAssert( !notdelcontent );

        enZeroExt = false;
    }


















//private: - don't abuse this, treat as private!  We commented this out to enable aggregate initialisation in zerointbasic etc

    // Need specialised versions, so no default arguments allowed!

    void locresize(int size, int suggestedallocsize, const T *fillval, bool leavemem)
    {
        NiceAssert( suggestedallocsize != -2 );
        NiceAssert( !leavemem );

        bool locarrayhold = ( suggestedallocsize == -3 ) ? true : array_hold();
        suggestedallocsize = ( suggestedallocsize == -3 ) ? -1 : suggestedallocsize;

        // NB: THERE IS A SPECIALISED AND NON-SPECIALISED VERSION, SO UPDATE BOTH!

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
        if ( !content && size )
        {
            // JIT allocation occurs here
            // On first run we just allocate the size requested - no alloc
            // ahead - as there is a strong possibility that this could
            // possibly be a fixed size array.

            dsize     = size;
            allocsize = ( ( suggestedallocsize == -1 ) || ( suggestedallocsize == -2 ) ) ? size : suggestedallocsize;

            NiceAssert( allocsize >= 0 );

            MEMNEWARRAY(content,T,allocsize+1);

            NiceAssert(content);
            #ifndef IGNOREMEM
            memcount((allocsize*sizeof(T)),+1);
            #endif

            if ( fillval )
            {
                for ( int i = -1 ; i < allocsize ; ++i )
                {
                    content[i+1] = *fillval;
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

            T *newcontent = nullptr;

            int newdsize     = dsize;
            int newallocsize = allocsize;

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

            T *oldcontent = content;

            int copysize;
            int modallocsize;

            copysize     = ( size < newdsize ) ? size : newdsize;
            modallocsize = array_tight() ? size : stepsizeupclip;
            newallocsize = ( ( suggestedallocsize == -1 ) || ( suggestedallocsize == -2 ) ) ? modallocsize : suggestedallocsize;

            newdsize = size;

            NiceAssert( newallocsize >= newdsize );
            NiceAssert( newallocsize >= 0        );
            NiceAssert( newdsize     >= 0        );

            MEMNEWARRAY(newcontent,T,newallocsize+1);

            NiceAssert(newcontent);
            #ifndef IGNOREMEM
            memcount((newallocsize*sizeof(T)),+1);
            #endif

            if ( fillval )
            {
                for ( int i = -1 ; i < newallocsize ; ++i )
                {
                    newcontent[i+1] = *fillval;
                }
            }

            else if ( copysize+((int) enZeroExt) && oldcontent )
            {
                for ( int i = -((int) enZeroExt) ; i < copysize ; ++i )
                {
                    // Note use of qswap here.  If the contents of the array
                    // are non-trivial (eg sparse vectors) then using a
                    // copy here would result in a serious performance hit.

                    //newcontent[i+1] = oldcontent[i+1];
                    qswap(newcontent[i+1],oldcontent[i+1]);
                }
            }

            content = newcontent;

            dsize     = newdsize;
            allocsize = newallocsize;

            holdalloc  = newholdalloc;

            if ( !leavemem && oldcontent )
            {
                MEMDELARRAY(oldcontent);
            }

            oldcontent = nullptr;

            #ifndef IGNOREMEM
            memcount((oldallocsize*sizeof(T)),-1);
            #endif
        }

        else
        {
            dsize = size;
        }
    }

    // content: contents of array
    //
    // dsize:      size of array
    // allocsize:  elements allocated to array
    // tightalloc: set if tight allocation strategy used (no allocahead)
    //
    // holdalloc:  set if pre-allocation has occured (allocation size known)
    // enZeroExt:  set if we allow an additional "zero" vector at index -1

    T *content; // default initialisation to 0 == NULL... = nullptr;

    int dsize;      // default initialised to 0... = 0;
    int allocsize;  // default initialised to 0... = 0;
    int tightalloc; // default initialised to 0... = 0;

    bool holdalloc;  // default initialised to 0... = false;
    bool enZeroExt;  // default initialised to 0... = false;

    // Only delete content if set - *only* set false for zerointarray etc.

    bool notdelcontent; // default initialised to 0... = false;
};

template <> void DynArray<int>::locresize(   int size, int suggestedallocsize, const int    *fillval, bool leavemem);




inline const DynArray<int> *zerointarray(void)
{
    static int content[2] = { 0, 0 };
    static DynArray<int> zeroarray = { content,1,1,1,true,true,true }; // final notdelcontent

    return &zeroarray;
}

inline const DynArray<int> *oneintarray(void)
{
    static int content[2] = { 0, 1 };
    static DynArray<int> onearray = { content,1,1,1,true,true,true }; // final notdelcontent

    return &onearray;
}

inline const DynArray<double> *zerodoublearray(void)
{
    static double content[2] = { 0, 0 };
    static DynArray<double> zeroarray = { content,1,1,1,true,true,true }; // final notdelcontent

    return &zeroarray;
}

inline const DynArray<double> *onedoublearray(void)
{
    static double content[2] = { 0, 1 };
    static DynArray<double> onearray = { content,1,1,1,true,true,true }; // final notdelcontent

    return &onearray;
}

inline const DynArray<double> *ninfdoublearray(void)
{
    static double content[2] = { 0, -INFINITY };
    static DynArray<double> ninfarray = { content,1,1,1,true,true,true }; // final notdelcontent

    return &ninfarray;
}

inline const DynArray<double> *pinfdoublearray(void)
{
    static double content[2] = { 0, INFINITY };
    static DynArray<double> pinfarray = { content,1,1,1,true,true,true }; // final notdelcontent

    return &pinfarray;
}

#define MAXHACKYHEAD 1000000

inline const DynArray<int> *cntintarray(int size)
{
    static int maxsize = 0;
    static std::unique_ptr<DynArray<int> > vogonpoetry(new DynArray<int>({nullptr,0,0,0,false,false,false})); // deletion is automatic thanks to unique_ptr ownership
    static DynArray<int> *cntarray = vogonpoetry.get();

    NiceAssert( size >= 0 );

    if ( ( size > maxsize ) || !maxsize )
    {
        int newsize = size+MAXHACKYHEAD;

        (*cntarray).resize(newsize,-2,nullptr,true); // -2 used here to force the resize function to do the counting

        maxsize = newsize;
    }

    return cntarray;
}

#endif
