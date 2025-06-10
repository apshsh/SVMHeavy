
//
// Kernel cache class
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _kcache_h
#define _kcache_h

#include "vector.hpp"
#include "gentype.hpp"
#include "basefn.hpp"
//Design decision: don't share kcache over thread boundary
//#ifdef ENABLE_THREADS
//#include <mutex>
//#endif

template <class T> class Klink;
template <class T> class Kcache;

template <class T> std::ostream &operator<<(std::ostream &output, const Kcache<T> &src );
template <class T> std::istream &operator>>(std::istream &input,        Kcache<T> &dest);

// LARGE_TRAIN_BOUNDARY: number of training vectors past which the dataset is "large"

#define LARGE_TRAIN_BOUNDARY       5000
// SEE ALSO ML_BASE.H

// Swap function
//
// NB: while it is OK to swap pointers to Klink, swapping actual elements would
//     not work as this is a linked list, and the pointers would end up dangling.
//     (*& is a reference to a pointer).  We need this when quick swapping the
//     lookup vector, which is a vector of pointers to Klink<T>.

template <class T> void qswap(Klink<T> *&a, Klink<T> *&b);
template <class T> void qswap(Kcache<T> &a, Kcache<T> &b);

template <class T>
class Klink
{
    /*
       kernel_row = the kernel row.
       row_ident  = row number (or -1 if not used).
    */

    public:

    Klink(int newallocsize = -1);
    Klink(const Vector<T> &xkernel_row, int xrow_ident = -1, int newallocsize = -1);

    void prealloc(int newallocsize)
    {
        kernel_row.prealloc(newallocsize);
        //inner_row.prealloc(newallocsize); - don't do this.  For large datasets this is a waste of space
        //dist_row.prealloc(newallocsize);
    }

    Vector<T> kernel_row;
    int row_ident;

    // Extra infor: temp stores for <x,y>+bias and ||x-y||^2 in midst of kernel reset
    // (zero size when not filled)

    Vector<T> inner_row;
    Vector<T> dist_row;

    Klink<T> *next;
    Klink<T> *prev;
};

template <class T> Klink<T> *&setident (Klink<T> *&a) { NiceThrow("What do you mean by setident"); return a; }
template <class T> Klink<T> *&setzero  (Klink<T> *&a) { return a = nullptr; }
template <class T> Klink<T> *&setposate(Klink<T> *&a) { return a; }
template <class T> Klink<T> *&setnegate(Klink<T> *&a) { NiceThrow("I reject your reality and substitute my own"); return a; }
template <class T> Klink<T> *&setconj  (Klink<T> *&a) { NiceThrow("Mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm"); return a; }
template <class T> Klink<T> *&setrand  (Klink<T> *&a) { NiceThrow("Blippity Blappity Blue"); return a; }
template <class T> Klink<T> *&postProInnerProd(Klink<T> *&a) { return a; }

template <class T>
class Kcache
{
    template <class S> friend std::ostream &operator<<(std::ostream &output, const Kcache<S> &src );
    template <class S> friend std::istream &operator>>(std::istream &input,        Kcache<S> &dest);

    template <class S> friend void qswap(Kcache<S> &a, Kcache<S> &b);

public:

    /*
       Constructors and destructor:

       - allocate all vectors etc.  Must be followed by a called to
         setmemsize before use (by default, the constructor sets
	 everything to 0, and vectors empty).  If the evalCache
	 is not set then it must be set later with the reset function.
         The evalCache function is used to evaluate values in the cache.
    */

    Kcache();
    Kcache(int xtrainsize, void (*xevalCache)(T &, int, int, const gentype **, const void *), void *xevalArg, int xsymmetry = +1, int xpreferLower = 0);
    Kcache(const Kcache &src);
    Kcache<T> &operator=(const Kcache<T> &src);
    ~Kcache();

    void prealloc(int newallocsize);

    /*
       reset()

       - puts cache back into state pre doing anything.  Can also set
         evalCache and evalArg.  Must be followed by a call to
	 setmemsize.

       setmemsize(memsize,min_rowdim)

       - this function does two things.  Firstly, rowdim is recalculated,
         and is the larger of min_rowdim and the training set size N.  If this
         value has changed then the dimension of all elements of kernel_rows
         will be increased to suit (junk left as padding).

         Secondly, numrows is calculated.  This is the number of rows of
         size rowdim which can be fitted into memsize MB of memory
         (roughly, anyhow - there will be some bookkeeping overhead which
         is ignored).  If numrows is less than the size of kernel_rows then
         the oldest rows will be removed so that the values coincide.
         Similarly, if numrows exceeds the size of kernel_rows then blank
	 (filler) rows with row_idents set to zero will be added.

       setEvalArg(void *xevalArg)

       - sets evalArg and then recalculates the diagonals.  Nothing else.
         This is useful as diagonals are lost by the assignment operator.

       setSymmetry(int nv)

       - sets symmetry.  You'll need to clear or recalc after this as this
         function does not (cannot) account for change.

       setPreferLower(int nv)

       - sets lower preference in calculation.

       clear()

       - clears all rows in cache.  This implies setting lookup = nullptr,
         recalculating diagvals, and setting row_ident = -1 in all elements
         of the linked list.  If there is inner_row or dist_row then this
         is used to speed up calculations.  Also frees any memory for cheats
         and resets is...Cheat (=0).

       recalc(rownum)

       - recalculates all stored values in a particular row/column.  Actually
         removes row rownum and recalculates relevant values in columns.
         This does not incur additional cost and may save flops if the
         row is not required again.

       set...Cheat:

       - uses given function to calculate inner product or distance required
         to quickly calculate the kernel for all kernels in cache and saves
         these values.  Subsequent calls to clear will then restore the
         values using accelerated calculation leveraging the stored inner or 
         distance values.  Use with care, and only on kernels that support
         the relevant reverseK operations.

    */

    void reset(int xpreallocsize = -1);
    void reset(int xtrainsize, int xpreallocsize);
    void reset(int xtrainsize, void (*xevalCache)(T &, int, int, const gentype **, const void *), void *xevalArg, int xpreallocsize = -1);
    void setEvalArg(void *xevalArg);
    void cheatSetEvalArg(void *xevalArg) { evalArg = xevalArg;   }
    void cheatSetEvalCache(void (*xevalCache)(T &, int, int, const gentype **, const void *)) { evalCache = xevalCache; }
    void setSymmetry(int xsymmetry)      { symmetry = xsymmetry; }
    void setPreferLower(int nv)          { preferLower = nv;     }
    void setmemsize(int xmemsize, int xmin_rowdim, int modprealloc = 0);
    void clear(void);
    void recalc(int rownum);
    void setInnerCheat(void (*reverseK)(T &, const T &, void *), void *carg);
    void setDistCheat (void (*reverseK)(T &, const T &, void *), void *carg);

    /*
       remove(num)

       - if row num is currently stored, then remove it (actually a null
         operation on kernel_rows, but row_idents will be written with a
         zero).  This assumes that remove has been called because a row/
         column has been removed from the training set.  Hence not only will
         the row be removed, but the elements of the other kernel_rows
         corresponding this this will be removed (and those elements lying
         after this moved back one).

         It is assumed that remove will be called after the relevant data
         has been removed from caller.  An implicit call will be made to
         setmemsize after this function is complete.

       add(num)

       - add new row/column to cache, adding the relevant value to the
         rows already in the cache.

         It is assumed that add will be called after the relevant data
         has been added to caller.  An implicit call will be made to
         setmemsize before this function is started.

       recalcDiag()

       - recalculate the diagonals of the matrix.  If i is given then
         only the (i,i) diagonal is updated.

       symmetry:

       - set symmetry value (0 for asymmetric, 1 for symmetric, -1 for
         skew-symmetry (i,j)=-(j,i))

       preferLower:

       - 0 for normal, 1 makes getRow return the lower part
    */

    void remove(int num);
    void add(int num);
    void recalcDiag(void);
    void recalcDiag(int i);
    void padCol(int n);

    /*
       getval(numi,numj):

       - returns the value for elements numi,numj.  If the element is stored
         (which can be told easily by looking for numi (or numj) in the
         row_idents vector) then will simply return the value.

	 If the value is not stored the relevant row will be constructed and
	 filled out, and the relevant element returned.  If memory is already
	 full before this then the oldest row (last in the linked list) will
	 be overwritten.

         When constructing a new row, if possible kernel values will be
         taken from existing rows instead of re-calculating them.

	 numi is used for getrow calls if possible.

       getrow(numi):

       - like getval, but returns (vector) row numi.

       isRowInCache(numi):

       - returns 1 if row is in cache, zero otherwise

       getvalIfPresent(numi, numj, isgood);

       - if value is present in cache then isgood is set and the value
         (reference) returned.  Otherwise isgood is reset (0) and a 
         dummy (static) reference returned.
    */

    const T &getval(int numi, int numj, retVector<T> &tmpva);
    const T getval_v(int numi, int numj, retVector<T> &tmpva);
    const T *getvalIfPresent(int numi, int numj) const;
    const T getvalIfPresent_v(int numi, int numj, int &isgood) const;
    const Vector<T> &getrow(int numi, retVector<T> &tmp);
    int isRowInCache(int numi);

    /*
       Information functions:
    */

    int get_trainsize(void)   const { return trainsize;   }
    int get_padsize(void)     const { return padsize;     }
    int get_padreserve(void)  const { return padreserve;  }
    int get_maxrows(void)     const { return maxrows;     }
    int get_rowdim(void)      const { return rowdim;      }
    int get_min_rowdim(void)  const { return min_rowdim;  }
    int get_memsize(void)     const { return memsize;     }
    int get_symmetry(void)    const { return symmetry;    }
    int get_preferLower(void) const { return preferLower; }

private:

    /*
       The kernel data is as follows:

       first_element = first (most recently accessed) kernel cache row.
       last_element  = last kernel cache row (may be unused).

       trainsize   = the size of the current training set.  Kept b/c it is
                     necessary to know when it has changed in the setmemsize
		     function.
       padsize     = number of zero elements appended to each vector
                     when row is returned.
       padreserve  = padsize that is maintained on each row
       maxrows     = maximum number of rows that can fit into current cache.
       numrows     = the number of rows in the cache (including empty rows).
       rowdim      = size of each row vector in the cache.  If rowdim exceeds
                     trainsize then random padding will be present.
       memsize     = size of memory (in MB) which the cache must fit into.
                     (-1 means unlimited)
       min_rowdim  = minimum allowable dimension of rows in the cache.

       evalCache = function evaluated to find value in cache
       evalArg   = void * argument passed to evalCache

       lookup = if a row is in the cache then lookup(rownum) will point to the
                relevant Klink element.  Otherwise it will be nullptr.

       symmetry = +1 for symmetric matrix (i,j) = (j,i)
                  -1 for anti-symmetric matrix (i,j) = -(j,i)
                  0  for a-symmetric (no assumptions)

       preferLower = 1 when possible try to use rows lwoer in the matrix
                     0 don't do that

       isInnerCheat: 1 if there is "cheat" <x,y>+bias ready for clear, 0 otherwise
       isDistCheat:  1 if there is "cheat" ||x-y||^2  ready for clear, 0 otherwise
    */

    Klink<T> *first_element;
    Klink<T> *last_element;

    int trainsize;
    int padsize;
    int padreserve;
    int maxrows;
    int rowdim;
    int memsize;
    int min_rowdim;
    int symmetry;
    int preferLower;
    int preallocsize;
    int isInnerCheat;
    int isDistCheat;

    void (*evalCache)(T &, int, int, const gentype **, const void *);
    void *evalArg;

    Vector<Klink<T> *> lookup;
    Vector<T> diagvals;

    void evalcacheind(T &, int i, int j);

    // We need to lock getrow for parallel operation so that
    // row accesses don't collide or operate partially.  Note
    // that memsize == -1 must be assumed for parallel operation
    // (or at least trainsize <= cache length).

//#ifdef ENABLE_THREADS
//    mutable std::mutex cachelock;
//#endif
};

template <class T> void qswap(Klink<T> *&a, Klink<T> *&b)
{
    Klink<T> *x(a); a = b; b = x;
}

template <class T> void qswap(Kcache<T> &a, Kcache<T> &b)
{
    qswap(a.first_element,b.first_element);
    qswap(a.last_element ,b.last_element );

    qswap(a.trainsize   ,b.trainsize   );
    qswap(a.padsize     ,b.padsize     );
    qswap(a.padreserve  ,b.padreserve  );
    qswap(a.maxrows     ,b.maxrows     );
    qswap(a.rowdim      ,b.rowdim      );
    qswap(a.memsize     ,b.memsize     );
    qswap(a.min_rowdim  ,b.min_rowdim  );
    qswap(a.symmetry    ,b.symmetry    );
    qswap(a.preferLower ,b.preferLower );
    qswap(a.preallocsize,b.preallocsize);
    qswap(a.isInnerCheat,b.isInnerCheat);
    qswap(a.isDistCheat ,b.isDistCheat );

    void (*evalCache)(T &, int, int, const gentype **, const void *);
    void *evalArg;

    evalCache = a.evalCache; a.evalCache = b.evalCache; b.evalCache = evalCache;
    evalArg   = a.evalArg;   a.evalArg   = b.evalArg;   b.evalArg   = evalArg;

    qswap(a.lookup  ,b.lookup  );
    qswap(a.diagvals,b.diagvals);
}


// Return functions for virtual matrices.

template <class T>
const Vector<T> &Kcache_crow(int numi, const void *owner, retVector<T> &);

template <class T>
const T &Kcache_celm(int numi, int numj, const void *owner, retVector<T> &);

const Vector<double >         &Kcache_crow_double (int numi, const void *owner, retVector<double>          &tmp);
const Vector<gentype>         &Kcache_crow_gentype(int numi, const void *owner, retVector<gentype>         &tmp);
const Vector<Matrix<double> > &Kcache_crow_matrix (int numi, const void *owner, retVector<Matrix<double> > &tmp);

      double          Kcache_celm_v_double(int numi, int numj, const void *owner, retVector<double>          &);
const double         &Kcache_celm_double  (int numi, int numj, const void *owner, retVector<double>          &);
const gentype        &Kcache_celm_gentype (int numi, int numj, const void *owner, retVector<gentype>         &);
const Matrix<double> &Kcache_celm_matrix  (int numi, int numj, const void *owner, retVector<Matrix<double> > &);



#define MIN_MAXROWS 5
//phantomx#define MIN_MAXROWS 701
#define MAX_MAXROWS 1000000
#define DIAGTHRESHOLD 1e-3
#define DEFAULT_PADSIZE 0
#define DEFAULT_PADRESERVE 20


template <class T>
void defaultEvalCache(T &res, int i, int j, const gentype **pxyprod, const void *evalArg);

//template <class T>
//void defaultEvalCache(T &res, int i, int j, const gentype **pxyprod, const void *evalArg)
template <class T>
void defaultEvalCache(T &res, int, int, const gentype **, const void *)
{
    res = 0.0;
}


template <class T>
Klink<T>::Klink(int newallocsize)
{
    row_ident = -1;

    next = nullptr;
    prev = nullptr;

    if ( newallocsize != -1 )
    {
        kernel_row.prealloc(newallocsize);
    }
}

template <class T>
Klink<T>::Klink(const Vector<T> &xkernel_row, int xrow_ident, int newallocsize)
{
    if ( newallocsize != -1 )
    {
        kernel_row.prealloc(newallocsize);
    }

    kernel_row = xkernel_row;
    row_ident  = xrow_ident;

    next = nullptr;
    prev = nullptr;
}

template <class T>
Kcache<T>::Kcache() :
    first_element(nullptr),
    last_element(nullptr),
    trainsize(0),
    padsize(DEFAULT_PADSIZE),
    padreserve(DEFAULT_PADRESERVE),
    maxrows(0),
    rowdim(0),
    memsize(0),
    min_rowdim(0),
    symmetry(1),
    preferLower(0),
    preallocsize(-1),
    isInnerCheat(0),
    isDistCheat(0),
    evalCache(&defaultEvalCache),
    evalArg(nullptr),
    lookup(),
    diagvals()
{
}



template <class T>
Kcache<T>::Kcache(int xtrainsize, void (*xevalCache)(T &, int, int, const gentype **, const void *), void *xevalArg, int xsymmetry, int xpreferLower) :
    first_element(nullptr),
    last_element(nullptr),
    trainsize(xtrainsize),
    padsize(DEFAULT_PADSIZE),
    padreserve(DEFAULT_PADRESERVE),
    maxrows(0),
    rowdim(0),
    memsize(0),
    min_rowdim(0),
    symmetry(xsymmetry),
    preferLower(xpreferLower),
    preallocsize(-1),
    isInnerCheat(0),
    isDistCheat(0),
    evalCache(xevalCache),
    evalArg(xevalArg),
    lookup(xtrainsize),
    diagvals(xtrainsize)
{
//    trainsize    = xtrainsize;
//    padsize      = DEFAULT_PADSIZE;
//    padreserve   = DEFAULT_PADRESERVE;
//    maxrows      = 0;
//    rowdim       = 0;
//    memsize      = 0;
//    min_rowdim   = 0;
//    symmetry     = xsymmetry;
//    preferLower  = xpreferLower;
//    preallocsize = -1;
//    isInnerCheat = 0;
//    isDistCheat  = 0;

//    first_element = nullptr;
//    last_element  = nullptr;

//    evalCache = xevalCache;
//    evalArg   = xevalArg;

//    lookup.resize(trainsize);
//    diagvals.resize(trainsize);

    lookup = nullptr;

    if ( trainsize )
    {
	int i;

	for ( i = 0 ; i < trainsize ; ++i )
	{
            (*evalCache)(diagvals("&",i),i,i,nullptr,evalArg);
	}
    }
}

template <class T>
void Kcache<T>::setInnerCheat(void (*reverseK)(T &, const T &, void *), void *carg)
{
    if ( trainsize && ( trainsize < LARGE_TRAIN_BOUNDARY ) )
    {
        isInnerCheat = 1;

        int i,j;

        for ( i = 0 ; i < trainsize ; ++i )
        {
            if ( lookup.v(i) )
            {
                ((*lookup("&",i)).inner_row).resize(trainsize);

                for ( j = 0 ; j < trainsize ; ++j )
                {
                    reverseK(((*lookup("&",i)).inner_row)("&",j),((*lookup(i)).kernel_row)(j),carg);
                }
            }
        }
    }
}

template <class T>
void Kcache<T>::setDistCheat(void (*reverseK)(T &, const T &, void *), void *carg)
{
    if ( trainsize && ( trainsize < LARGE_TRAIN_BOUNDARY ) )
    {
        isDistCheat = 1;

        int i,j;

        for ( i = 0 ; i < trainsize ; ++i )
        {
            if ( lookup.v(i) )
            {
                ((*lookup("&",i)).dist_row).resize(trainsize);

                for ( j = 0 ; j < trainsize ; ++j )
                {
                    reverseK(((*lookup("&",i)).dist_row)("&",j),((*lookup(i)).kernel_row)(j),carg);
                }
            }
        }
    }
}

template <class T>
void Kcache<T>::clear(void)
{
    if ( isInnerCheat || isDistCheat )
    {
        // Do quick recalc on all values, free cheat storage and reset all cheats

        if ( trainsize )
        {
	    int i,j;

            for ( i = 0 ; i < trainsize ; ++i )
            {
                (*evalCache)(diagvals("&",i),i,i,nullptr,evalArg);

                if ( lookup.v(i) )
                {
                    gentype pxyinner;
                    gentype pxydist;

                    //gentype **pxyprodbleh;
                    //
                    //MEMNEWARRAY(pxyprodbleh,gentype *,2);
                    //
                    //pxyprodbleh[0] = isInnerCheat ? &pxyinner : nullptr;
                    //pxyprodbleh[1] = isDistCheat  ? &pxydist  : nullptr;

                    gentype *pxyprodbleh[2];

                    pxyprodbleh[0] = isInnerCheat ? &pxyinner : nullptr;
                    pxyprodbleh[1] = isDistCheat  ? &pxydist  : nullptr;

                    const gentype **pxyprod = (const gentype **) ((void *) pxyprodbleh);

                    NiceAssert( ( isInnerCheat && ((*lookup(i)).inner_row).size() ) || ( isDistCheat && ((*lookup(i)).dist_row).size() ) );

                    for ( j = 0 ; j < trainsize ; ++j )
                    {
                        if ( i != j )
                        {
                            if ( isInnerCheat )
                            {
                                pxyinner = ((*lookup(i)).inner_row)(j);
                            }

                            if ( isDistCheat )
                            {
                                pxydist = ((*lookup(i)).dist_row)(j);
                            }

//phantomx - potential twice-work here, use symmetry to avoid double-calculation
                            (*evalCache)(((*lookup("&",i)).kernel_row)("&",j),i,j,pxyprod,evalArg);
                        }

                        else
                        {
                            ((*lookup("&",i)).kernel_row).set(j,diagvals(i));
                        }
                    }

                    ((*lookup("&",i)).inner_row).resize(0);
                    ((*lookup("&",i)).dist_row ).resize(0);

                    //MEMDELARRAY(pxyprodbleh);
                }
            }
        }

        isInnerCheat = 0;
        isDistCheat  = 0;
    }

    else
    {
        // Clear lookup

        lookup = nullptr;

        // Reset diagonal values

        if ( trainsize )
        {
	    int i;

            for ( i = 0 ; i < trainsize ; ++i )
            {
                (*evalCache)(diagvals("&",i),i,i,nullptr,evalArg);
            }
        }

        // Set row_ident == 0

        Klink<T> *posptr = first_element;

        while ( posptr != nullptr )
        {
	    posptr->row_ident = -1;
            posptr = posptr->next;
        }
    }
}

template <class T>
void Kcache<T>::recalc(int rownum)
{
    NiceAssert( rownum >= 0 );
    NiceAssert( rownum < trainsize );

    // Reset diagonal

    (*evalCache)(diagvals("&",rownum),rownum,rownum,nullptr,evalArg);

    // Remove row from lookup

    Klink<T> *pos_ptr  = nullptr;
    Klink<T> *pos_hold = nullptr;
    Klink<T> *pos_now  = nullptr;

    lookup.sv(rownum,nullptr);

    // Find row in linked list (if it exists), mark it unused and put
    // it at the end of the list.

    pos_ptr = first_element;

    while ( pos_ptr != nullptr )
    {
        pos_now = pos_ptr;

        if ( pos_ptr->row_ident == rownum )
	{
	    pos_ptr->row_ident = -1;

	    if ( ( pos_ptr == first_element ) && ( pos_ptr != last_element ) )
	    {
		first_element = first_element->next;
		first_element->prev = nullptr;

		pos_ptr->next = nullptr;
		pos_ptr->prev = last_element;

		last_element->next = pos_ptr;
		last_element = pos_ptr;

		pos_ptr = first_element;
	    }

	    else if ( pos_ptr != last_element )
	    {
		pos_hold = pos_ptr->next;

		(pos_ptr->prev)->next = pos_ptr->next;
		(pos_ptr->next)->prev = pos_ptr->prev;

		pos_ptr->next = nullptr;
		pos_ptr->prev = last_element;

		last_element->next = pos_ptr;
		last_element = pos_ptr;

		pos_ptr = pos_hold;
	    }

            else
            {
                pos_ptr = pos_ptr->next;
            }
	}

        else
        {
            pos_ptr = pos_ptr->next;
        }

        if ( pos_now->row_ident != -1 )
	{
            (*evalCache)((pos_now->kernel_row)("&",rownum),pos_now->row_ident,rownum,nullptr,evalArg);
        }
    }
}

template <class T>
Kcache<T>::Kcache(const Kcache<T> &src) :
    first_element(nullptr),
    last_element(nullptr),
    trainsize(src.trainsize),
    padsize(src.padsize),
    padreserve(src.padreserve),
    maxrows(0),
    rowdim(0),
    memsize(0),
    min_rowdim(0),
    symmetry(src.symmetry),
    preferLower(src.preferLower),
    preallocsize(src.preallocsize),
    isInnerCheat(0),
    isDistCheat(0),
    evalCache(src.evalCache),
    evalArg(src.evalArg),
    lookup(),
    diagvals(src.diagvals)
{
//    trainsize    = src.trainsize;
//    padsize      = src.padsize;
//    padreserve   = src.padreserve;
//    maxrows      = 0;
//    rowdim       = 0;
//    memsize      = 0;
//    min_rowdim   = 0;
//    symmetry     = src.symmetry;
//    preferLower   = src.preferLower;
//    preallocsize = src.preallocsize;
//    isInnerCheat = 0;
//    isDistCheat  = 0;

    if ( preallocsize != -1 )
    {
        diagvals.prealloc(preallocsize);
        lookup.prealloc(preallocsize);
    }

//    diagvals = src.diagvals;

//    first_element = nullptr;
//    last_element  = nullptr;

//    evalCache = src.evalCache;
//    evalArg   = src.evalArg;

    lookup.resize(trainsize);
    lookup = nullptr;

    setmemsize(src.memsize,src.min_rowdim,1);
}

template <class T>
Kcache<T> &Kcache<T>::operator=(const Kcache<T> &src)
{
    reset();

    trainsize    = src.trainsize;
    padsize      = src.padsize;
    padreserve   = src.padreserve;
    maxrows      = 0;
    rowdim       = 0;
    memsize      = 0;
    min_rowdim   = 0;
    symmetry     = src.symmetry;
    preferLower  = src.preferLower;
    preallocsize = src.preallocsize;
    isInnerCheat = 0;
    isDistCheat  = 0;

    if ( preallocsize != -1 )
    {
        diagvals.prealloc(preallocsize);
        lookup.prealloc(preallocsize);
    }

    diagvals = src.diagvals;

    first_element = nullptr;
    last_element  = nullptr;

    evalCache = src.evalCache;
    evalArg   = src.evalArg;

    lookup.resize(trainsize);
    lookup = nullptr;

    setmemsize(src.memsize,src.min_rowdim,1);

    return *this;
}

template <class T>
Kcache<T>::~Kcache()
{
    Klink<T> *pos_ptr;
    Klink<T> *lagger;

    pos_ptr = first_element;

    while ( pos_ptr != nullptr )
    {
        lagger  = pos_ptr;
        pos_ptr = pos_ptr->next;

        MEMDEL(lagger);
    }
}

template <class T>
void Kcache<T>::prealloc(int newallocsize)
{
    NiceAssert( ( newallocsize == -1 ) || ( newallocsize >= 0 ) );

    lookup.prealloc(newallocsize);
    diagvals.prealloc(newallocsize);

    preallocsize = newallocsize;

    setmemsize(memsize,min_rowdim,1);
}

template <class T>
void Kcache<T>::reset(int xpreallocsize)
{
    NiceAssert( ( xpreallocsize == -1 ) || ( xpreallocsize > 0 ) );

    Klink<T> *pos_ptr;
    Klink<T> *lagger;

    pos_ptr = first_element;

    while ( pos_ptr != nullptr )
    {
        lagger  = pos_ptr;
        pos_ptr = pos_ptr->next;

        MEMDEL(lagger);
    }

    evalCache = nullptr;
    evalArg   = nullptr;

    trainsize    = 0;
    maxrows      = 0;
    rowdim       = 0;
    memsize      = 0;
    min_rowdim   = 0;
    preallocsize = xpreallocsize;

    if ( preallocsize != -1 )
    {
        diagvals.prealloc(preallocsize);
        lookup.prealloc(preallocsize);
    }

    diagvals.resize(0);

    first_element = nullptr;
    last_element  = nullptr;

    // don't change evalCache or evalArg

    lookup.resize(0);
}

template <class T>
void Kcache<T>::reset(int xtrainsize, void (*xevalCache)(T &, int, int, const gentype **, const void *), void *xevalArg, int xpreallocsize)
{
    reset(xpreallocsize);

    trainsize = xtrainsize;
    evalCache = xevalCache;
    evalArg   = xevalArg;

    lookup.resize(trainsize);
    lookup = nullptr;

    diagvals.resize(trainsize);

    if ( trainsize )
    {
	int i;

	for ( i = 0 ; i < trainsize ; ++i )
	{
            (*evalCache)(diagvals("&",i),i,i,nullptr,evalArg);
	}
    }
}

template <class T>
void Kcache<T>::reset(int xtrainsize, int xpreallocsize)
{
    reset(xpreallocsize);

    T zeroval;

    zeroval = 0.0;

    trainsize = xtrainsize;

    lookup.resize(trainsize);
    lookup = nullptr;

    diagvals.resize(trainsize);
    diagvals = zeroval;
}

template <class T>
void Kcache<T>::setmemsize(int xmemsize, int xmin_rowdim, int modprealloc)
{
    NiceAssert( ( xmemsize > 0 ) || ( xmemsize == -1 ) );
    NiceAssert( xmin_rowdim > 0 );

    if ( preallocsize < trainsize )
    {
        preallocsize = -1;
        modprealloc = 1;
    }

    int rowalloc;
    int effallocsize = ( preallocsize == -1 ) ? trainsize : preallocsize;

    // Cannot let effallocsize be zero or the linked-list blows up (no first/last points)

    effallocsize = ( effallocsize > MIN_MAXROWS  ) ? effallocsize : MIN_MAXROWS;

    int i;
    Klink<T> *pos_ptr;

    int oldmemsize = memsize;

    memsize    = xmemsize;
    min_rowdim = xmin_rowdim;

    if ( oldmemsize == 0 )
    {
        /*
           Set all the relevant values.
        */

        rowdim   = ( trainsize > min_rowdim ) ? trainsize : min_rowdim;
        rowalloc = ( rowdim > preallocsize ) ? rowdim : preallocsize;
        //maxrows  = ( memsize == -1 ) ? MAX_MAXROWS : ( (int) ( ((double) memsize*1024*1024) / ((double) rowalloc*sizeof(T)) ) );
        maxrows  = ( memsize == -1 ) ? trainsize : ( (int) ( ((double) memsize*1024*1024) / ((double) rowalloc*sizeof(T)) ) );
        maxrows  = ( maxrows > MIN_MAXROWS  ) ? maxrows : MIN_MAXROWS;
        maxrows  = ( maxrows < MAX_MAXROWS  ) ? maxrows : MAX_MAXROWS;
        maxrows  = ( maxrows < effallocsize ) ? maxrows : effallocsize;

        /*
           Allocate the vectors.
        */

        Vector<T> temp;

        temp.prealloc(preallocsize+padreserve);
        temp.resize(trainsize); // was rowdim
        temp.pad((rowdim-trainsize)+padreserve); // was padreserve

        MEMNEW(first_element,Klink<T>(temp,-1,preallocsize+padreserve));
        NiceAssert( first_element );

        pos_ptr = first_element;

        if ( maxrows >= 2 )
        {
            for ( i = 1 ; i < maxrows ; ++i )
            {
                MEMNEW(pos_ptr->next,Klink<T>(temp,-1,preallocsize+padreserve));
                NiceAssert( pos_ptr->next );
                (pos_ptr->next)->prev = pos_ptr;
                pos_ptr = pos_ptr->next;
            }
        }

	last_element = pos_ptr;

        first_element->prev = nullptr;
        last_element->next = nullptr;
    }

    else
    {
        int new_rowdim;
        int new_maxrows;

        new_rowdim   = ( trainsize > min_rowdim ) ? trainsize : min_rowdim;
        rowalloc     = ( new_rowdim > preallocsize ) ? new_rowdim : preallocsize;
        //new_maxrows  = ( memsize == -1 ) ? MAX_MAXROWS : ( (int) ( ((double) memsize*1024*1024) / ((double) rowalloc*sizeof(T)) ) );
        new_maxrows  = ( memsize == -1 ) ? trainsize : ( (int) ( ((double) memsize*1024*1024) / ((double) rowalloc*sizeof(T)) ) );
        new_maxrows  = ( new_maxrows > MIN_MAXROWS  ) ? new_maxrows : MIN_MAXROWS;
        new_maxrows  = ( new_maxrows < MAX_MAXROWS  ) ? new_maxrows : MAX_MAXROWS;
        new_maxrows  = ( new_maxrows < effallocsize ) ? new_maxrows : effallocsize;

        if ( ( new_rowdim != rowdim ) && modprealloc )
        {
            pos_ptr = first_element;

            while ( pos_ptr != nullptr )
            {
                (pos_ptr->kernel_row).prealloc(preallocsize+padreserve);
                (pos_ptr->kernel_row).resize(trainsize); // was new_rowdim
                (pos_ptr->kernel_row).pad((new_rowdim-trainsize)+padreserve); // was padreserve

                pos_ptr = pos_ptr->next;
            }
        }

        else if ( modprealloc )
        {
            pos_ptr = first_element;

            while ( pos_ptr != nullptr )
            { 
                (pos_ptr->kernel_row).prealloc(preallocsize+padreserve);

                pos_ptr = pos_ptr->next;
            }
        }

        else if ( new_rowdim != rowdim )
        {
            pos_ptr = first_element;

            while ( pos_ptr != nullptr )
            {
                (pos_ptr->kernel_row).resize(trainsize); // was new_rowdim
                (pos_ptr->kernel_row).pad((new_rowdim-trainsize)+padreserve); // was padreserve

                pos_ptr = pos_ptr->next;
            }
        }

        rowdim = new_rowdim;

        if ( new_maxrows > maxrows )
        {
            Vector<T> temp;

            temp.prealloc(preallocsize+padreserve);
            temp.resize(trainsize); // was rowdim
            temp.pad((rowdim-trainsize)+padreserve); // was padreserve

            pos_ptr = last_element;

            for ( i = maxrows ; i < new_maxrows ; ++i )
            {
                MEMNEW(pos_ptr->next,Klink<T>(temp,-1,preallocsize+padreserve));
                NiceAssert( pos_ptr->next );

                (pos_ptr->next)->prev = pos_ptr;
                (pos_ptr->next)->next = nullptr;

                pos_ptr = pos_ptr->next;
            }

            last_element = pos_ptr;
        }

        else if ( new_maxrows < maxrows )
        {
            Klink<T> *temp;

            pos_ptr = first_element;
            temp    = first_element;

            if ( new_maxrows )
            {
                for ( i = 0 ; i < new_maxrows ; ++i )
                {
                    temp = pos_ptr;
                    pos_ptr = pos_ptr->next;
                }
            }

            last_element = temp;
            last_element->next = nullptr;

            for ( i = new_maxrows ; i < maxrows ; ++i )
            {
                temp = pos_ptr;
		pos_ptr = pos_ptr->next;

		if ( temp->row_ident != -1 )
		{
		    lookup.sv(temp->row_ident,nullptr);
		}

                MEMDEL(temp);
            }
        }

        maxrows = new_maxrows;
    }
}

template <class T>
void Kcache<T>::add(int num)
{
    NiceAssert( num >= 0 );
    NiceAssert( num <= trainsize );

    Klink<T> *pos_ptr = first_element;

    ++trainsize;

    lookup.add(num);
    lookup.sv(num,nullptr);
    diagvals.add(num);
    (*evalCache)(diagvals("&",num),num,num,nullptr,evalArg);

    setmemsize(memsize,min_rowdim);

    pos_ptr = first_element;

    while ( pos_ptr != nullptr )
    {
	if ( pos_ptr->row_ident != -1 )
	{
            (pos_ptr->kernel_row).blockswap(trainsize,num); // was rowdim-1,num
            //evalcacheind((pos_ptr->kernel_row)("&",num),pos_ptr->row_ident,num); - original position, pretty sure this was a mistake as x was
            //                                                                       updated before this, so if we do this here then the first
            //                                                                       index will be off-by-one.  Bug only became obvious in
            //                                                                       svm_scalar_rff when adding vectors after NRff set (that is,
            //                                                                       adding a vector, in effect, "in the middle"

            if ( pos_ptr->row_ident >= num )
            {
                ++(pos_ptr->row_ident);
            }

            evalcacheind((pos_ptr->kernel_row)("&",num),pos_ptr->row_ident,num); // new position.
	}

        else
        {
            break;
        }

	pos_ptr = pos_ptr->next;
    }
}

template <class T>
void Kcache<T>::padCol(int n)
{
    NiceAssert(n >= 0 );

    padsize = n;

    int padextend = padsize-padreserve;

    if ( padextend > 0 )
    {
        padreserve = padsize;

        Klink<T> *pos_ptr = first_element;

        while ( pos_ptr != nullptr )
        {
            (pos_ptr->kernel_row).pad(padextend);

            pos_ptr = pos_ptr->next;
        }
    }
}

template <class T>
void Kcache<T>::remove(int num)
{
    NiceAssert( num >= 0 );
    NiceAssert( num < trainsize );

    Klink<T> *pos_ptr = nullptr;
    Klink<T> *pos_hold = nullptr;
    Klink<T> *pos_now = nullptr;

    --trainsize;

    lookup.remove(num);
    diagvals.remove(num);

    pos_ptr = first_element;

    while ( pos_ptr != nullptr )
    {
        pos_now = pos_ptr;

	if ( pos_ptr->row_ident == num )
	{
	    pos_ptr->row_ident = -1;

	    if ( ( pos_ptr == first_element ) && ( pos_ptr != last_element ) )
	    {
		first_element = first_element->next;
		first_element->prev = nullptr;

		pos_ptr->next = nullptr;
		pos_ptr->prev = last_element;

		last_element->next = pos_ptr;
		last_element = pos_ptr;

		pos_ptr = first_element;
	    }

	    else if ( pos_ptr != last_element )
	    {
		pos_hold = pos_ptr->next;

		(pos_ptr->prev)->next = pos_ptr->next;
		(pos_ptr->next)->prev = pos_ptr->prev;

		pos_ptr->next = nullptr;
		pos_ptr->prev = last_element;

		last_element->next = pos_ptr;
		last_element = pos_ptr;

		pos_ptr = pos_hold;
	    }

            else
            {
                pos_ptr = pos_ptr->next;
            }
	}

        else
        {
            pos_ptr = pos_ptr->next;
        }

        if ( pos_now->row_ident != -1 )
        {
            (pos_now->kernel_row).blockswap(num,trainsize);

            if ( pos_now->row_ident > num )
            {
                --(pos_now->row_ident);
            }
        }

        else
        {
            break;
        }
    }

    setmemsize(memsize,min_rowdim);
}

template <class T>
void Kcache<T>::recalcDiag(void)
{
    Klink<T> *pos_ptr = first_element;

    if ( trainsize )
    {
	int i;

	for ( i = 0 ; i < trainsize ; ++i )
	{
            (*evalCache)(diagvals("&",i),i,i,nullptr,evalArg);
	}
    }

    while ( pos_ptr != nullptr )
    {
	if ( pos_ptr->row_ident == -1 )
	{
            break;
	}

        evalcacheind((pos_ptr->kernel_row)("&",pos_ptr->row_ident),pos_ptr->row_ident,pos_ptr->row_ident);

        pos_ptr = pos_ptr->next;
    }
}

template <class T>
void Kcache<T>::recalcDiag(int i)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < trainsize );

    (*evalCache)(diagvals("&",i),i,i,nullptr,evalArg);

    Klink<T> *pos_ptr = lookup.v(i);

    if ( pos_ptr != nullptr )
    {
        evalcacheind((pos_ptr->kernel_row)("&",pos_ptr->row_ident),pos_ptr->row_ident,pos_ptr->row_ident);
    }
}

template <class T>
void Kcache<T>::setEvalArg(void *xevalArg)
{
    evalArg = xevalArg;

    if ( trainsize )
    {
	int i;

	for ( i = 0 ; i < trainsize ; ++i )
	{
            (*evalCache)(diagvals("&",i),i,i,nullptr,evalArg);
	}
    }
}

template <class T>
const T &Kcache<T>::getval(int numi, int numj, retVector<T> &tmp)
{
    NiceAssert( numi >= 0 );
    NiceAssert( numi < trainsize );
    NiceAssert( numj >= 0 );
    NiceAssert( numj < trainsize+padsize );

    const T *resval = getvalIfPresent(numi,numj);

//FIXME
//phantomx - have "jog" that does first calculation as one-off, subsequent as calc whole row.
//phantomx - and what about calculating only active part?
    return resval ? *resval : ( ( ( numi >= numj ) || ( symmetry != 1 ) || !preferLower ) ? (getrow(numi,tmp))(numj) : (getrow(numj,tmp))(numi) );
}

template <class T>
const T Kcache<T>::getval_v(int numi, int numj, retVector<T> &tmp)
{
    NiceAssert( numi >= 0 );
    NiceAssert( numi < trainsize );
    NiceAssert( numj >= 0 );
    NiceAssert( numj < trainsize+padsize );

    int haveval = 0;
    T resval = getvalIfPresent_v(numi,numj,haveval);

//FIXME
//phantomx - have "jog" that does first calculation as one-off, subsequent as calc whole row.
//phantomx - and what about calculating only active part?
    return haveval ? resval : ( ( ( numi >= numj ) || ( symmetry != 1 ) || !preferLower ) ? (getrow(numi,tmp)).v(numj) : (getrow(numj,tmp)).v(numi) );
}

//#define ISVALPRESENT ( ( numi == numj ) || ( lookup.v(numi) != nullptr ) || ( ( numj < trainsize ) && ( lookup.v(numi) == nullptr ) && ( lookup.v(numj) != nullptr ) && ( symmetry == +1 ) ) )

template <class T>
const T *Kcache<T>::getvalIfPresent(int numi, int numj) const
{
    NiceAssert( numi >= 0 );
    NiceAssert( numi < trainsize );
    NiceAssert( numj >= 0 );
    NiceAssert( numj < trainsize+padsize );

    Klink<T> *pos_ptr = lookup.v(numi);

    if ( numi == numj )
    {
        return &diagvals(numi);
    }

    else if ( ( pos_ptr = lookup.v(numi) ) != nullptr )
    {
        return &((pos_ptr->kernel_row)(numj));
    }

    //else if ( ( numj < trainsize ) && ( lookup.v(numi) == nullptr ) && ( ( pos_ptr = lookup.v(numj) ) != nullptr ) && ( symmetry == +1 ) )
    else if ( ( numj < trainsize ) && ( ( pos_ptr = lookup.v(numj) ) != nullptr ) && ( symmetry == +1 ) )
    {
        return &((pos_ptr->kernel_row)(numi));
    }

    return nullptr;
}

template <class T>
const T Kcache<T>::getvalIfPresent_v(int numi, int numj, int &isgood) const
{
    NiceAssert( numi >= 0 );
    NiceAssert( numi < trainsize );
    NiceAssert( numj >= 0 );
    NiceAssert( numj < trainsize+padsize );

    Klink<double> *pos_ptr = lookup.v(numi);

    isgood = 0;

    if ( numi == numj )
    {
        isgood = 1;

        return diagvals.v(numi);
    }

    else if ( ( pos_ptr = lookup.v(numi) ) != nullptr )
    {
        isgood = 1;

        return ((pos_ptr->kernel_row).v(numj));
    }

    //else if ( ( numj < trainsize ) && ( lookup.v(numi) == nullptr ) && ( ( pos_ptr = lookup.v(numj) ) != nullptr ) && ( symmetry == +1 ) )
    else if ( ( numj < trainsize ) && ( ( pos_ptr = lookup.v(numj) ) != nullptr ) && ( symmetry == +1 ) )
    {
        isgood = 1;

        return ((pos_ptr->kernel_row).v(numi));
    }

    return diagvals.v(numi); // if reached isgood = 0, so return value never actually used
}

template <> inline const Vector<double> &Kcache<double>::getrow(int numi, retVector<double> &tmp);
template <> inline const Vector<double> &Kcache<double>::getrow(int numi, retVector<double> &tmp)
{
    NiceAssert( numi >= 0 );
    NiceAssert( numi < trainsize );

    int j;
    Klink<double> *pos_ptr = lookup.v(numi);

    if ( !pos_ptr )
    {
        // This block must be thread-safe!
//#ifdef ENABLE_THREADS
//        cachelock.lock();
//#endif

        pos_ptr = lookup.v(numi); // in case this has changed (race condition)

        if ( !pos_ptr )
        {
	    /*
	       If not found, construct the relevant row.
	    */

            if ( last_element->row_ident != -1 )
	    {
                // NB: for threaded operation we *must not* do this
                //     operation.  Either need memsize == -1 or big
                //     enough to make sure that this never gets hit!

	        lookup.sv(last_element->row_ident,nullptr);
	    }

            // Last element is now hanging, so we can set it up.
            //
            // Need to do all other operations first before we mess with lookup
            // or another thread could jump in, see lookup(numi) set, and then
            // use the partially formed row.

            for ( j = 0 ; j < trainsize ; ++j )
	    {
	        if ( ( lookup.v(j) == nullptr ) || ( numi == j ) || !symmetry )
	        {
                    evalcacheind((last_element->kernel_row)("&",j),numi,j);
	        }

	        else
	        {
                    (last_element->kernel_row).sv(j,((double) symmetry)*(((lookup.v(j))->kernel_row).v(numi)));
	        }
	    }

            pos_ptr = last_element;

            last_element = last_element->prev;
            last_element->next = nullptr;

            pos_ptr->next = first_element;
            pos_ptr->prev = nullptr;

            first_element->prev = pos_ptr;
            first_element = pos_ptr;

            first_element->row_ident  = numi;
            //first_element->kernel_row = ... see above;

            lookup.sv(numi,first_element);
        }

//#ifdef ENABLE_THREADS
//        cachelock.unlock();
//#endif
    }

    else if ( ( memsize != -1 ) && ( maxrows >= trainsize ) && ( pos_ptr != first_element ) )
    {
        // Lock for safety, but note we really shouldn't be here!
//#ifdef ENABLE_THREADS
//        cachelock.lock();
//#endif
        if ( pos_ptr != first_element )
        {
	    /*
	       Put the row at the top of the list
	    */

	    if ( pos_ptr == last_element )
	    {
	        last_element = last_element->prev;
	        last_element->next = nullptr;

	        pos_ptr->next = first_element;
	        pos_ptr->prev = nullptr;

	        first_element->prev = pos_ptr;
	        first_element = pos_ptr;
	    }

	    else if ( pos_ptr != first_element )
	    {
	        (pos_ptr->prev)->next = pos_ptr->next;
	        (pos_ptr->next)->prev = pos_ptr->prev;

	        pos_ptr->next = first_element;
	        pos_ptr->prev = nullptr;

	        first_element->prev = pos_ptr;
	        first_element = pos_ptr;
	    }
	}

//#ifdef ENABLE_THREADS
//        cachelock.unlock();
//#endif
    }

    return (pos_ptr->kernel_row)(0,1,trainsize+padsize-1,tmp);
}

template <class T>
const Vector<T> &Kcache<T>::getrow(int numi, retVector<T> &tmp)
{
    NiceAssert( numi >= 0 );
    NiceAssert( numi < trainsize );

    int j;
    Klink<T> *pos_ptr = lookup.v(numi);

    if ( !pos_ptr )
    {
        // This block must be thread-safe!
//#ifdef ENABLE_THREADS
//        cachelock.lock();
//#endif

        pos_ptr = lookup.v(numi); // in case this has changed (race condition)

        if ( !pos_ptr )
        {
	    /*
	       If not found, construct the relevant row.
	    */

            if ( last_element->row_ident != -1 )
	    {
                // NB: for threaded operation we *must not* do this
                //     operation.  Either need memsize == -1 or big
                //     enough to make sure that this never gets hit!

	        lookup.sv(last_element->row_ident,nullptr);
	    }

            // Last element is now hanging, so we can set it up.
            //
            // Need to do all other operations first before we mess with lookup
            // or another thread could jump in, see lookup(numi) set, and then
            // use the partially formed row.

            for ( j = 0 ; j < trainsize ; ++j )
	    {
	        if ( ( lookup.v(j) == nullptr ) || ( numi == j ) || !symmetry )
	        {
                    evalcacheind((last_element->kernel_row)("&",j),numi,j);
	        }

	        else
	        {
                    (last_element->kernel_row).sv(j,((double) symmetry)*(((lookup.v(j))->kernel_row)(numi)));
	        }
	    }

            pos_ptr = last_element;

            last_element = last_element->prev;
            last_element->next = nullptr;

            pos_ptr->next = first_element;
            pos_ptr->prev = nullptr;

            first_element->prev = pos_ptr;
            first_element = pos_ptr;

            first_element->row_ident  = numi;
            //first_element->kernel_row = ... see above;

            lookup.sv(numi,first_element);
        }

//#ifdef ENABLE_THREADS
//        cachelock.unlock();
//#endif
    }

    else if ( ( memsize != -1 ) && ( maxrows >= trainsize ) && ( pos_ptr != first_element ) )
    {
        // Lock for safety, but note we really shouldn't be here!
//#ifdef ENABLE_THREADS
//        cachelock.lock();
//#endif
        if ( pos_ptr != first_element )
        {
	    /*
	       Put the row at the top of the list
	    */

	    if ( pos_ptr == last_element )
	    {
	        last_element = last_element->prev;
	        last_element->next = nullptr;

	        pos_ptr->next = first_element;
	        pos_ptr->prev = nullptr;

	        first_element->prev = pos_ptr;
	        first_element = pos_ptr;
	    }

	    else if ( pos_ptr != first_element )
	    {
	        (pos_ptr->prev)->next = pos_ptr->next;
	        (pos_ptr->next)->prev = pos_ptr->prev;

	        pos_ptr->next = first_element;
	        pos_ptr->prev = nullptr;

	        first_element->prev = pos_ptr;
	        first_element = pos_ptr;
	    }
	}

//#ifdef ENABLE_THREADS
//        cachelock.unlock();
//#endif
    }

    return (pos_ptr->kernel_row)(0,1,trainsize+padsize-1,tmp);
}

template <class T>
int Kcache<T>::isRowInCache(int numi)
{
    NiceAssert( numi >= 0 );
    NiceAssert( numi < trainsize );

//    int res = 0;
//
//    if ( lookup.v(numi) != nullptr )
//    {
//        res = 1;
//    }
//
//    return res;

    return ( lookup.v(numi) != nullptr ) ? 1 : 0;
}

template <class T>
void Kcache<T>::evalcacheind(T &res, int i, int j)
{
    if ( i != j )
    {
        (*evalCache)(res,i,j,nullptr,evalArg);
    }

    else
    {
        res = diagvals(i);
    }
}

template <class T>
const Vector<T> &Kcache_crow(int numi, const void *owner, retVector<T> &tmp)
{
    Kcache<T> *typed_owner = (Kcache<T> *) owner;

    return typed_owner->getrow(numi,tmp);
}

template <class T>
const T &Kcache_celm(int numi, int numj, const void *owner, retVector<T> &tmp)
{
    Kcache<T> *typed_owner = (Kcache<T> *) owner;

    return typed_owner->getval(numi,numj,tmp);
}




template <class T>
std::ostream &operator<<(std::ostream &output, const Kcache<T> &src )
{
    output << "Training set size:  " << src.trainsize    << "\n";
    output << "Zero padding size:  " << src.padsize      << "\n";
    output << "Zero pad reserve:   " << src.padreserve   << "\n";
    output << "Target memory size: " << src.memsize      << "\n";
    output << "Minimum row dim:    " << src.min_rowdim   << "\n";
    output << "Symmetry:           " << src.symmetry     << "\n";
    output << "Prefer Lower:       " << src.preferLower   << "\n";
    output << "Preallocation size: " << src.preallocsize << "\n";

    return output;
}


template <class T>
std::istream &operator>>(std::istream &input,        Kcache<T> &dest)
{
    wait_dummy dummy;
    int trainsize;
    int memsize;
    int min_rowdim;

    input >> dummy; input >> trainsize;
    input >> dummy; input >> dest.padsize;
    input >> dummy; input >> dest.padreserve;
    input >> dummy; input >> memsize;
    input >> dummy; input >> min_rowdim;
    input >> dummy; input >> dest.symmetry;
    input >> dummy; input >> dest.preferLower;
    input >> dummy; input >> dest.preallocsize;

    dest.isInnerCheat = 0;
    dest.isDistCheat  = 0;

    dest.reset(trainsize);
    dest.setmemsize(memsize,min_rowdim,1);

    return input;
}

#endif
