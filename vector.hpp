
//
// Vector class
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _vvector_h
#define _vvector_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include <sstream>
#include <type_traits>
#include "randfun.hpp"
#include "qswapbase.hpp"
#include "dynarray.hpp"
#include "memdebug.hpp"
#include "niceassert.hpp"
#include "numbase.hpp"


// constexpr_if is handy if available, but not strictly necessary

#ifdef IS_CPP20
#define svm_constexpr_if constexpr
#endif

#ifndef IS_CPP20
#define svm_constexpr_if
#endif

// constexpr with variables in function is available in c++14 and after

//#ifdef IS_CPP14
//#define svm_constexpr constexpr
//#endif

//#ifndef IS_CPP14
//#define svm_constexpr
//#endif




#ifndef DEFAULT_SAMPLES_SAMPLE
#define DEFAULT_SAMPLES_SAMPLE     100
#endif


template <class T> class Vector;
template <class T> class retVector;

//Spoilers
template <class T> class Matrix;
template <class T> class retMatrix;
template <class T> class SparseVector;
class gentype;

// Stream operators

template <class T> std::ostream &operator<<(  std::ostream &output, const Vector<T> &src );
template <class T> std::ostream &printoneline(std::ostream &output, const Vector<T> &src );
template <class T> std::istream &operator>>(  std::istream &input,        Vector<T> &dest);
template <class T> std::istream &streamItIn(  std::istream &input,        Vector<T> &dest, int processxyzvw = 1);

//Spoilers
template <class T> std::istream &streamItIn(std::istream &input, SparseVector<T> &dest, int processxyzvw = 1);
                   std::istream &streamItIn(std::istream &input, gentype         &dest, int processxyzvw = 1);

// Swap function

template <class T> void qswap(const Vector<T> *&a, const Vector<T> *&b);
template <class T> void qswap(      Vector<T> *&a,       Vector<T> *&b);
template <class T> void qswap(      Vector<T>  &a,       Vector<T>  &b);

// For retVector

template <class T> void qswap(retVector<T> &a, retVector<T> &b);

// Base share function

template <class S, class U> bool shareBase(const Vector<S> &thus, const Vector<U> &that);

// Vector return handle.

template <class T> class retVector;

// Need to be defined first

template <class S> double norm1(const Vector<S> &a);
template <class S> double norm2(const Vector<S> &a);
template <class S> double normp(const Vector<S> &a, double p);

template <class S> double abs1  (const Vector<S> &a);
template <class S> double abs2  (const Vector<S> &a);
template <class S> double absp  (const Vector<S> &a, double p);
template <class S> double absinf(const Vector<S> &a);
template <class S> double abs0  (const Vector<S> &a);


template <class T>
class retVector : public Vector<T>
{
public:
    explicit retVector() : Vector<T>("&") { ; }

    // This function resets the return vector to clean-slate.  No need to
    // call this as it gets called when required by operator()

    retVector<T> &reset(void); // this version deletes pivot (if !pbase) and sets pivot = null

    retVector<T> &reset(Vector<T> &cover); // this version deletes pivot (if !pbase) and sets pivot = null
    DynArray<int> *reset_p(Vector<T> &cover, int pivotsize); // this version allocates pivot(pivotsize) (or resizes if pbase)

    retVector<T> &creset(const Vector<T> &cover); // this version deletes pivot (if !pbase) and sets pivot = null
    DynArray<int> *creset_p(const Vector<T> &cover, int pivotsize); // this version allocates pivot(pivotsize) (or resizes if pbase)
};

// Handy fixed vectors
//
// zeroxvec: ( 0 0 0 0 ... )
// onexvec:  ( 1 1 1 1 ... )
// cntxvec:  ( 0 1 2 3 ... )

inline const Vector<int> &zerointvec(int size, retVector<int> &tmpv);
inline const Vector<int> &oneintvec (int size, retVector<int> &tmpv);
inline const Vector<int> &cntintvec (int size, retVector<int> &tmpv);

inline const Vector<double> &zerodoublevec(int size, retVector<double> &tmpv);
inline const Vector<double> &onedoublevec (int size, retVector<double> &tmpv);
//inline const Vector<double> &cntdoublevec (int size, retVector<double> &tmpv);

inline const Vector<double> &ninfdoublevec(int size, retVector<double> &tmpv);
inline const Vector<double> &pingdoublevec(int size, retVector<double> &tmpv);
//inline const Vector<double> &cntdoublevec (int size, retVector<double> &tmpv);

// Basic versions

inline const Vector<int> &zerointvecbasic(void);
inline const Vector<int> &oneintvecbasic (void);

inline const Vector<double> &zerodoublevecbasic(void);
inline const Vector<double> &onedoublevecbasic (void);

inline const Vector<double> &ninfdoublevecbasic(void);
inline const Vector<double> &pinfdoublevecbasic(void);

//// Cooked UDL stuff
//
//inline Vector<double> operator"" _zerovect(unsigned long long int dim);
//inline Vector<double> operator"" _onevect (unsigned long long int dim);
//inline Vector<double> operator"" _ninfvect(unsigned long long int dim);
//inline Vector<double> operator"" _pinfvect(unsigned long long int dim);
//
//inline Vector<double> operator"" _zerovect(unsigned long long int dim) { retVector<double> tmpv; return zerodoublevec((int) dim,tmpv); }
//inline Vector<double> operator"" _onevect (unsigned long long int dim) { retVector<double> tmpv; return onedoublevec ((int) dim,tmpv); };
//inline Vector<double> operator"" _ninfvect(unsigned long long int dim) { retVector<double> tmpv; return ninfdoublevec((int) dim,tmpv); }
//inline Vector<double> operator"" _pinfvect(unsigned long long int dim) { retVector<double> tmpv; return pinfdoublevec((int) dim,tmpv); }

// The class itself

template <class T>
class Vector
{
    friend class retVector<T>;

    template <class> friend class Vector;
    template <class> friend class Matrix;

    template <class S> friend void qswap(   Vector<S> &a,    Vector<S> &b);
    template <class S> friend void qswap(retVector<S> &a, retVector<S> &b);

    friend inline const Vector<int> &zerointvec(int size, retVector<int> &tmpv);
    friend inline const Vector<int> &oneintvec (int size, retVector<int> &tmpv);
    friend inline const Vector<int> &cntintvec (int size, retVector<int> &tmpv);

    friend inline const Vector<double> &zerodoublevec(int size, retVector<double> &tmpv);
    friend inline const Vector<double> &onedoublevec (int size, retVector<double> &tmpv);
//    friend inline const Vector<double> &cntdoublevec (int size, retVector<double> &tmpv);

    friend inline const Vector<double> &ninfdoublevec(int size, retVector<double> &tmpv);
    friend inline const Vector<double> &pingdoublevec(int size, retVector<double> &tmpv);
//    friend inline const Vector<double> &cntdoublevec (int size, retVector<double> &tmpv);


    friend inline const Vector<int> &zerointvecbasic(void);
    friend inline const Vector<int> &oneintvecbasic (void);

    friend inline const Vector<double> &zerodoublevecbasic(void);
    friend inline const Vector<double> &onedoublevecbasic (void);

    friend inline const Vector<double> &ninfdoublevecbasic(void);
    friend inline const Vector<double> &pinfdoublevecbasic(void);

    template <class S, class U> friend bool shareBase(const Vector<S> &thus, const Vector<U> &that);

public:

    // Constructors and Destructors
    //
    // tightalloc: 0 = normal
    //             1 = tight
    //             2 = slack

    explicit Vector() : newln('\n'), nbase(false), pbase(true), dsize(0), defaulttightalloc(0), iib(0), iis(1), imoverhere(nullptr), bkref(nullptr), content(nullptr), ccontent(nullptr), pivot(nullptr) { ; }
    explicit Vector(int size, const T *src = nullptr, int tightalloc = 0);
    explicit Vector(int size, const T &src, int tightalloc = 0);
    explicit Vector(const char *, int tightalloc) : newln('\n'), nbase(false), pbase(true), dsize(0), defaulttightalloc(tightalloc), iib(0), iis(1), imoverhere(nullptr), bkref(nullptr), content(nullptr), ccontent(nullptr), pivot(nullptr) { ; }
             Vector(const Vector<T> &src);

    virtual ~Vector();

    // Print and make duplicate

    virtual Vector<T> *makeDup(void) const
    {
        Vector<T> *dup;

        MEMNEW(dup,Vector<T>(*this));

        return dup;
    }

    // Assignment
    //
    // - vector assignment: unless this vector is a temporary vector created
    //   to refer to parts of another vector then we do not require that sizes
    //   align but rather completely overwrite the destination, resetting the
    //   size to that of the source.
    // - scalar assignment: in this case the size of the vector remains
    //   unchanged, but all elements will be overwritten.
    // - behaviour is undefined if scalar is an element of this.
    // - assignment from a matrix only possible for 1*d or d*1 matrix.
    // - the assign function allows you to infer assignment between different
    //   template classes.  This is not defined as operator= to ensure that
    //   the second and third forms of operator= work without ambiguity

    Vector<T> &operator=(const Vector<T> &src) { return assign(src); }
    Vector<T> &operator=(const T &src)         { return assign(src); }

    virtual Vector<T> &assign(const Vector<T> &src);
    virtual Vector<T> &assign(const T         &src);

    Vector<T> &operator=(const Matrix<T> &src);

    template <class S> Vector<T> &castassign(const Vector<S> &src);

    // Simple vector manipulations
    //
    // ident:  apply setident() to all elements of the vector
    // zero:   apply setzero() to all elements of the vector
    //        (apply setzero() to element i if argument given).
    //
    // softzero: equivalent to zero (for consistency with sparsevector)
    // posate: apply setposate() to all elements of the vector
    // negate: apply setnegate() to all elements of the vector
    // conj:   apply setconj() to all elements of the vector
    // rand:   apply .rand() to all elements of the vector
    // offset: amoff > 0: insert amoff elements at the start of the vector
    //         amoff < 0: remove amoff elements from the start of the vector
    //
    //
    //
    //
    //
    // each returns a reference to *this

    Vector<T> &ident      (void);
    Vector<T> &zero       (int i);
    Vector<T> &zeropassive(void);
    Vector<T> &offset     (int amoff);

    virtual Vector<T> &softzero(void);
    virtual Vector<T> &zero    (void);
    virtual Vector<T> &posate  (void);
    virtual Vector<T> &negate  (void);
    virtual Vector<T> &conj    (void);
    virtual Vector<T> &rand    (void);

    // Access:
    //
    // - ("&",i) access a reference to element i.
    // - (i) access a const reference to element i.
    // - v(i) is the return by value form of (i).
    // - sv(i,x) set i by value..
    // - direcref(i) is functionally equivalent to (ind(i)).
    // - direref(i) is functionally equivalent to ("&",ind(i)).
    //
    // - ("&") returns are non-const reference to this
    // - () returns a const reference to this.
    //
    // - zeroExtDeref(i): like (i), where i is a vector, but in this case
    //   there is an additional index -1 that may be used.  Element "-1" is
    //   set zero upon calling.  This is handy for "padding" a vector with
    //   zeros (for example in sparse vectors or extended caches).
    //
    // Variants:
    //
    // - if i is of type Vector<int> then the reference returned is to the
    //   elements specified in i.
    // - if ib,is>0,im is given then this is the same as a vector i being used
    //   specified by: ( ib ib+is ib+is+is ... max_n(i=ib+(n*s)|i<=im) )
    //   (and if im < ib and is > 0 then an empty reference is returned)
    // - if ib,is<0,im is given then this is the same as a vector i being used
    //   specified by: ( ib ib+is ib+is+is ... min_n(i=ib+(n*s)|i>=im) )
    //   (and if im > ib and is < 0 then an empty reference is returned)
    // - if (ivec,ib,is,im) is given this is equivalent to (ivec)(ib,is,im)
    //
    // Notes:
    //
    // - direcref and direref variants are included for consistency with sparse
    //   vector type.
    //
    // Scope of result:
    //
    // - The scope of the returned reference is the minimum of the scope of
    //   retVector &tmp or *this.
    // - retVector &tmp may or may not be used depending on, so
    //   never *assume* that it will be!
    // - The returned reference is actually *this through a layer of indirection,
    //   so any changes to it will be reflected in *this (and vice-versa).

    Vector<T> &operator()(const char *dummy,                         retVector<T> &tmp) { NiceAssert( !infsize() ); if ( imoverhere ) { return overhere()(dummy,tmp); } return *this; }
    T         &operator()(const char *dummy, int i                                    ) { NiceAssert( !infsize() ); NiceAssert( i >= 0 ); NiceAssert( i < dsize ); if ( !ccontent ) { fix(); } NiceAssert( content ); NiceAssert( pivot ); return (*content)(dummy,(*pivot).v(iib+(i*iis))); }
    Vector<T> &operator()(const char *dummy, const Vector<int> &i,   retVector<T> &tmp);
    Vector<T> &operator()(const char *dummy, int ib, int is, int im, retVector<T> &tmp);
    Vector<T> &operator()(const char *dummy, const Vector<int> &i,   int ib, int is, int im, retVector<T> &tmp);

    const Vector<T> &operator()(                        retVector<T> &tmp) const { if ( imoverhere ) { return overhere()(tmp); } return *this; }
    const T         &operator()(int i                                    ) const { NiceAssert( !infsize() ); NiceAssert( i >= 0 ); NiceAssert( i < dsize ); NiceAssert( ccontent ); NiceAssert( pivot ); return (*ccontent)((*pivot).v(iib+(i*iis))); }
    const Vector<T> &operator()(const Vector<int> &i,   retVector<T> &tmp) const;
    const Vector<T> &operator()(int ib, int is, int im, retVector<T> &tmp) const;
    const Vector<T> &operator()(const Vector<int> &i,   int ib, int is, int im, retVector<T> &tmp) const;

    T     v(int i)      const { NiceAssert( !infsize() ); NiceAssert( i >= 0 ); NiceAssert( i < dsize ); NiceAssert( ccontent ); NiceAssert( pivot ); return (*ccontent).v( (*pivot).v(iib+(i*iis))  ); }
    void sv(int i, T x) const { NiceAssert( !infsize() ); NiceAssert( i >= 0 ); NiceAssert( i < dsize ); NiceAssert(  content ); NiceAssert( pivot );         (*content).sv((*pivot).v(iib+(i*iis)),x); }

    void set(                        const Vector<T> &src) { *this = src; }
    void set(int i,                  const        T  &src) { NiceAssert( !infsize() ); NiceAssert( i >= 0 ); NiceAssert( i < dsize ); NiceAssert( content ); NiceAssert( pivot ); (*content).set((*pivot).v(iib+(i*iis)),src); }
    void set(const Vector<int> &i,   const Vector<T> &src);
    void set(int ib, int is, int im, const Vector<T> &src);

    void set(                        const T &src) { *this = src; }
    void set(const Vector<int> &i,   const T &src);
    void set(int ib, int is, int im, const T &src);

    // For FuncVector

    virtual T &operator()(T &res, const        T  &i) const { if ( imoverhere ) { return overhere()(res,i); } NiceThrow("Continuous dereference not allowed for finite dimensional vector."); return res; }
    virtual T &operator()(T &res, const Vector<T> &i) const { if ( imoverhere ) { return overhere()(res,i); } NiceThrow("Continuous dereference not allowed for finite dimensional vector."); return res; }

    // Information functions
    //
    // type():    returns 0 for standard, 1 for FuncVector, 2 for RKHSVector etc
    // size():    returns the size of the vector (if finite)
    // order():   returns angry geckos
    // infsize(): returns 1 for infinite size, 0 otherwise
    // ismixed(): return 1 if "mixed type" vector (sum of different types, no defined inner products etc)

    virtual int type (void) const { return imoverhere ? overhere().type() : 0; }
            int size (void) const { return dsize;                              }
            int order(void) const { return ceilintlog2(size());                }

    virtual bool infsize(void) const { return imoverhere ? overhere().infsize() : false; }
    virtual bool ismixed(void) const { return false;                                     }

    virtual int testsametype(std::string &typestring)
    {
        return typestring == "";
    }

    // Vector scaling:
    //
    // Apply (*this)("&",i) *= a for all i, or (*this)("&",i) *= a(i) for the
    // vectorial version.  This is useful for scaling vectors of vectors.

    template <class S> Vector<T>  &scale(const S         &a);
    template <class S> Vector<T>  &scale(const Vector<S> &a);
    template <class S> Vector<T> &lscale(const S         &a);
    template <class S> Vector<T> &lscale(const Vector<S> &a);

    // Scaled addition:
    //
    // The following is functionally equivalent to *this += (a*b).  However
    // it is quicker and uses less memory as no temporary variables are
    // constructed.
    //
    // sqscaleAdd: like above, but a*b*b (vector part is squared)
    // vscaleAdd:  like above, but *this += (a.*b)

    template <class S> Vector<T> &scaleAdd  (const S         &a, const Vector<T> &b);
    template <class S> Vector<T> &scaleAddR (const Vector<T> &a, const S         &b);
    template <class S> Vector<T> &scaleAddB (const T         &a, const Vector<S> &b);
    template <class S> Vector<T> &scaleAddBR(const Vector<S> &b, const T         &a);

    template <class S> Vector<T> &sqscaleAdd  (const S         &a, const Vector<T> &b);
    template <class S> Vector<T> &sqscaleAddR (const Vector<T> &a, const S         &b);
    template <class S> Vector<T> &sqscaleAddB (const T         &a, const Vector<S> &b);
    template <class S> Vector<T> &sqscaleAddBR(const Vector<S> &b, const T         &a);

    Vector<T> &vscaleAdd(const Vector<T> &a, const Vector<T> &b);

    // Add and remove element functions
    //
    // add:    ( c ) (i)          ( c ) (i)
    //         ( d ) (...)  ->    ( 0 ) (1)
    //                            ( d ) (...)
    // addpad: ( c ) (i)          ( c ) (i)
    //         ( d ) (...)  ->    ( 0 ) (num)
    //                            ( d ) (...)
    // remove: ( c ) (i)          ( c ) (i)
    //         ( d ) (1)    ->    ( d ) (...)
    //         ( e ) (...)
    // resize: either add to end or remove from end until desired size is
    //         obtained.  See dynarray for suggestedallocsize usage.
    // append: add a to end of vector at position i >= size()
    // pad: add n elements to end of vector
    // qadd: add element given (it gets qswapped - ie destroyed)
    //
    // Note that these may not be applied to temporary vectors.

    Vector<T> &add (int i);
    Vector<T> &qadd(int i, T &src);
    Vector<T> &add (int i, int ipos) { NiceAssert( i == ipos ); (void) ipos; return add(i); }
    //Vector<T> &add(int i, int) { NiceAssert( i == ipos ); return add(i); }
    Vector<T> &addpad(int i, int num);
    Vector<T> &remove(int i);
    Vector<T> &remove(const Vector<int> &i);
    Vector<T> &resize(int i);
    Vector<T> &resize(int i, int suggestedallocsize);
    Vector<T> &pad(int n);
    template <class S> Vector<T> &resize(const Vector<S> &sizeTemplateUsed) { return resize(sizeTemplateUsed.size()); }
    Vector<T> &setorder(int i) { return resize(1<<i); }
    Vector<T> &append(int i) { add(i,size()); return *this; } // { add(i,indsize()); return *this; }
    Vector<T> &append(int i, const T &a);
    Vector<T> &append(int i, const Vector<T> &a);

    // Function application - apply function fn to each element of vector.

    virtual Vector<T> &applyon(T (*fn)(T));
    virtual Vector<T> &applyon(T (*fn)(const T &));
    virtual Vector<T> &applyon(T (*fn)(T, const void *), const void *a);
    virtual Vector<T> &applyon(T (*fn)(const T &, const void *), const void *a);
    virtual Vector<T> &applyon(T &(*fn)(T &));
    virtual Vector<T> &applyon(T &(*fn)(T &, const void *), const void *a);

    // Various swap functions
    //
    // blockswap  ( i < j ): ( c ) (i)          ( c ) (i)
    //                       ( e ) (1)    ->    ( d ) (j-i)
    //                       ( d ) (j-i)        ( e ) (1)
    //                       ( f ) (...)        ( f ) (...)
    //
    // blockswap  ( i > j ): ( c ) (j)          ( c ) (j)
    //                       ( d ) (i-j)  ->    ( e ) (1)
    //                       ( e ) (1)          ( d ) (i-j)
    //                       ( f ) (...)        ( f ) (...)
    //
    // squareswap ( i > j ): ( c ) (i)          ( c ) (i)
    //                       ( d ) (1)          ( f ) (1)
    //                       ( e ) (j-i)  ->    ( e ) (j-i)
    //                       ( f ) (1)          ( d ) (1)
    //                       ( g ) (...)        ( g ) (...)
    //
    // squareswap ( i < j ): ( c ) (j)          ( c ) (j)
    //                       ( d ) (1)          ( f ) (1)
    //                       ( e ) (i-j)  ->    ( e ) (i-j)
    //                       ( f ) (1)          ( d ) (1)
    //                       ( g ) (...)        ( g ) (...)
    //
    // It should be noted that these function actually move the contents of
    // the vector.  Hence if the contents are nontrivial (say you're dealing
    // with Vector<Vector<double>> or similar) then qswap operators will
    // be called repeatedly and lots of memory shuffling will occur.  Hence it
    // is usually better to keep a pivot vector of type Vector<int> on which
    // these operations are carried out.  For example given:
    //
    // Vector<Vector<double>> x;
    // Vector<int> pivot;
    //
    // it is better to make pivot = ( 0 1 2 3 ... ) and then do blockswap,
    // squareswap type operations on pivot and then use x(pivot) in
    // calculations.  This will have the same effect as doing blockswap etc.
    // operations directly on x but without the subsequent time penalty.

    Vector<T> &blockswap (int i, int j);
    Vector<T> &squareswap(int i, int j);

    // Pre-allocation control.  Vectors are pre-assigned with a given buffer
    // size to "grow" into, which can stretch and shrink dynamically as the
    // vector size evolves over time.  The downside of this is that memory
    // can be wasted when the class over-estimates the amount of memory
    // required, and speed can suffer if it under-estimates repeatedly (say
    // a vector grows from very small to very large) and the buffer must
    // be repeatedly reallocated and the contents copied over.
    //
    // To overcome this the following function lets you preset the size of
    // the allocate-ahead buffer if you know (or have a bound on) the final
    // size of the vector.  Note that this *only* works on base vector, not
    // children.
    //
    // To restore standard behaviour set newallocsize == -1
    //
    // The function applyonall applies an operation to all allocated members,
    // which may include pre-allocated members.
    //
    // Other functions here provide direct access to dynamic array base

    virtual void prealloc(int newallocsize);

    virtual void useStandardAllocation(void);
    virtual void useTightAllocation   (void);
    virtual void useSlackAllocation   (void);

    void applyOnAll(void (*fn)(T &, int), int argx);

    virtual bool array_norm (void) const { return imoverhere ? overhere().array_norm()  : ( content ? (*content).array_norm()  : true  ); }
    virtual bool array_tight(void) const { return imoverhere ? overhere().array_tight() : ( content ? (*content).array_tight() : false ); }
    virtual bool array_slack(void) const { return imoverhere ? overhere().array_slack() : ( content ? (*content).array_slack() : false ); }

    // Slight complication: whenever an assignment is called we need to deal
    // with possibilities like a *= a(pivot) or b = b(pivot), which will
    // royally screw things up if for example pivot = ( 3 2 1 ) and we start
    // naively doing the assignment element by element top to bottom.  To
    // deal with this we need to check if the source and destination share
    // the same root, and if they do make a temporary copy of the source and
    // then call the assignment operator for this temporary copy.  The
    // following function facillitates this by testing if this instance sharesFuncVector
    // a common root with another.
    //
    // Contiguous: &x(0) points to a simple array containing all elements

    template <class S>
    bool shareBase(const Vector<S> &that) const
    {
        return ::shareBase(*this,that);
    }

    bool base              (void) const { return nbase;                                                         }
    bool contiguous        (void) const { return !nbase || ( ( iis == 1 ) && ( pivot == cntintarray(dsize) ) ); }
    bool contentalloced    (void) const { return ( content ? true : false );                                    }
    bool contentarray_hold (void) const { NiceAssert( content ); return content->array_hold();                  }
    int  contentarray_alloc(void) const { NiceAssert( content ); return content->array_alloc();                 }

    // If friends need to grab ccontent, content or pivot, call this first to ensure it is allocated!

    //void fix(void) const { if ( !ccontent ) { resize(0); } NiceAssert( ccontent ); } - so we don't need mutable!
    void fix(void)       { if ( !ccontent ) { resize(0); } NiceAssert( ccontent ); }

    // Control how the output looks

    char getnewln(void) const { return newln; }
    char setnewln(char srcvl) { NiceAssert(isspace(srcvl)); return ( newln = srcvl ); }




    // Operators filled-in by functional (****don't use in code, just placeholders for now****)
    //
    // conj = 0: noConj
    //        1: normal
    //        2: revConj

    virtual std::ostream &outstream(std::ostream &output) const { NiceThrow("no"); return output; } // DO NOT USE THIS!
    virtual std::istream &instream (std::istream &input )       { NiceThrow("no"); return input;  } // DO NOT USE THIS!

    virtual std::istream &streamItIn(std::istream &input, int processxyzvw = 1) { (void) processxyzvw; NiceThrow("no"); return input; } // DON'T USE THIS!

    virtual T &inner1(T &res                                                            ) const {                                           if ( imoverhere ) { overhere().inner1(res);       } else { NiceThrow("Not an FuncVector 1"); } return res; }
    virtual T &inner2(T &res, const Vector<T> &b, int conj = 1                          ) const { (void) b; (void) conj;                    if ( imoverhere ) { overhere().inner2(res,b);     } else if ( b.imoverhere ) { b.inner2(res,*this,( conj == 1 ) ? 2 : ( ( conj == 2 ) ? 1 : 0 )); } else { NiceThrow("Not an FuncVector 2"); } return res; }
    virtual T &inner3(T &res, const Vector<T> &b, const Vector<T> &c                    ) const { (void) b; (void) c; (void) res;           if ( imoverhere ) { overhere().inner3(res,b,c);   } else if ( b.imoverhere ) { b.inner3(res,*this,c); } else if ( c.imoverhere ) { b.inner3(res,b,*this); } else { NiceThrow("Not an FuncVector 3"); } return res; }
    virtual T &inner4(T &res, const Vector<T> &b, const Vector<T> &c, const Vector<T> &d) const { (void) b; (void) c; (void) d; (void) res; if ( imoverhere ) { overhere().inner4(res,b,c,d); } else if ( b.imoverhere ) { b.inner4(res,*this,c,d); } else if ( c.imoverhere ) { c.inner4(res,b,*this,d); } else if ( d.imoverhere ) { d.inner4(res,b,c,*this); } else { NiceThrow("Not an FuncVector 4"); } return res; }
    virtual T &innerp(T &res, const Vector<const Vector<T> *> &b                        ) const
    {
//        (void) b; - really weird MS bug!  Uncomment this and microsoft visual c-c++ compiler, compile strfns.cc and the compiler basically chews
//                    up all available memory and crashes the entire system!  I have no fucking *idea* why!
        (void) res;

        if ( imoverhere )
        {
            overhere().innerp(res,b);
        }

        else
        {
             NiceThrow("Not an FuncVector 5");
        }

        return res;
    }

    virtual double &inner1Real(double &res                                                            ) const {              if ( imoverhere ) { overhere().inner1Real(res);       } else { NiceThrow("Not an FuncVector 6");  } return res; }
    virtual double &inner2Real(double &res, const Vector<T> &b, int conj = 1                          ) const { (void) conj; if ( imoverhere ) { overhere().inner2Real(res,b);     } else { NiceThrow("Not an FuncVector 7");  } return res; }
    virtual double &inner3Real(double &res, const Vector<T> &b, const Vector<T> &c                    ) const {              if ( imoverhere ) { overhere().inner3Real(res,b,c);   } else { NiceThrow("Not an FuncVector 8");  } return res; }
    virtual double &inner4Real(double &res, const Vector<T> &b, const Vector<T> &c, const Vector<T> &d) const {              if ( imoverhere ) { overhere().inner4Real(res,b,c,d); } else { NiceThrow("Not an FuncVector 9");  } return res; }
    virtual double &innerpReal(double &res, const Vector<const Vector<T> *> &b                        ) const {              if ( imoverhere ) { overhere().innerpReal(res,b);     } else { NiceThrow("Not an FuncVector 10"); } return res; }

    virtual double norm1(void)     const { double res = 0.0;           if ( imoverhere ) { res = overhere().norm1();  } else { NiceThrow("Not an FuncVector 11"); } return res; }
    virtual double norm2(void)     const { double res = 0.0;           if ( imoverhere ) { res = overhere().norm2();  } else { NiceThrow("Not an FuncVector 12"); } return res; }
    virtual double normp(double p) const { double res = 0.0; (void) p; if ( imoverhere ) { res = overhere().normp(p); } else { NiceThrow("Not an FuncVector 13"); } return res; }

    virtual double abs1(void)     const { return norm1(); }
    virtual double abs2(void)     const { return sqrt(norm2()); }
    virtual double absp(double p) const { return pow(normp(p),1/p); }

    virtual double absinf(void) const { double res = 0.0; if ( imoverhere ) { res = overhere().absinf(); } else { NiceThrow("Not an FuncVector 14a"); } return res; }
    virtual double abs0  (void) const { double res = 0.0; if ( imoverhere ) { res = overhere().abs0();   } else { NiceThrow("Not an FuncVector 14b"); } return res; }

    virtual Vector<T> &subit (const Vector<T> &b) { if ( imoverhere ) { overhere().subit(b);  } else { NiceThrow("Not in FuncVector 15");  } return *this; }
    virtual Vector<T> &addit (const Vector<T> &b) { if ( imoverhere ) { overhere().addit(b);  } else { NiceThrow("Not in FuncVector 16");  } return *this; }
   virtual Vector<T> &subit (const T         &b) { if ( imoverhere ) { overhere().subit(b);  } else { NiceThrow("Not in FuncVector 15a"); } return *this; }
   virtual Vector<T> &addit (const T         &b) { if ( imoverhere ) { overhere().addit(b);  } else { NiceThrow("Not in FuncVector 16a"); } return *this; }
   virtual Vector<T> &mulit (const Vector<T> &b) { if ( imoverhere ) { overhere().mulit(b);  } else { NiceThrow("Not in FuncVector 17a");  } return *this; } // this*b
   virtual Vector<T> &rmulit(const Vector<T> &b) { if ( imoverhere ) { overhere().rmulit(b); } else { NiceThrow("Not in FuncVector 18a");  } return *this; } // b*this
   template <class Y> Vector<T> &divit (const Vector<Y> &b) { if ( imoverhere ) { overhere().divit(b);  } else { NiceThrow("Not in FuncVector 19a");  } return *this; } // this/b
   virtual Vector<T> &rdivit(const Vector<T> &b) { if ( imoverhere ) { overhere().rdivit(b); } else { NiceThrow("Not in FuncVector 20a");  } return *this; } // b\this
    virtual Vector<T> &mulit (const T         &b) { if ( imoverhere ) { overhere().mulit(b);  } else { NiceThrow("Not in FuncVector 17");   } return *this; } // this*b
    virtual Vector<T> &rmulit(const T         &b) { if ( imoverhere ) { overhere().rmulit(b); } else { NiceThrow("Not in FuncVector 18");   } return *this; } // b*this
    virtual Vector<T> &divit (const T         &b) { if ( imoverhere ) { overhere().divit(b);  } else { NiceThrow("Not in FuncVector 19");   } return *this; } // this/b
    virtual Vector<T> &rdivit(const T         &b) { if ( imoverhere ) { overhere().rdivit(b); } else { NiceThrow("Not in FuncVector 20");   } return *this; } // b\this

    virtual bool iseq(const Vector<T> &b) { if ( imoverhere ) { return overhere().iseq(b); } else { NiceThrow("Not in FuncVector 21"); } return 0; }

private:

    // dsize: size of vector
    // fixsize: if set then the size cannot be changed
    // defaulttightalloc: default tightalloc value for content dynarray
    //
    // nbase: 0 if content is local, 1 if it points elsewhere
    //        (NB: if nbase == 0 then pivot = ( 0 1 2 ... ))
    // pbase: 0 if pivot is local, 1 if it points elsewhere
    //        (NB: if nbase == 0 then pbase == 0 by definition)
    //
    // iib: constant added to indexes
    // iis: step for indexes
    //
    // bkref: if nbase, this is the vector derived from (and pointed to).  This
    //        is used by the shareBase function.
    // content: contents of vector
    // ccontent: constant pointer to content
    // pivot: pivotting used to access contents
    //
    // newln: usually vectors are printed:
    //                   ( x0 ;
    //                   x1 ;
    //                   ... ;
    //                   xn-1 )
    //        where the newline is given by newln.  By setting for example
    //        newln = ' ' you would get the alternative format
    //                   ( x0 ; x1 ; ... ; xn-1 )

    char newln; // = '\n';

    bool nbase; // = 0;
    bool pbase; // = 1;

    int dsize; // = 0; - mutable to enable JIT
//    int fixsize = 0;
    int defaulttightalloc; // = 0;

    int iib; // = 0;
    int iis; // = 1; (set zero by constructor, as it should be)

    // The vector class was written a long time ago, *before* I knew about polymorphism.  
    // It got built in to a lot of things.  Much later I wanted to add support for RKHS, 
    // which is an infinite-dimensional vector with an inner product that isn't just an
    // integral.  The easiest was to do this was polymorphism... except I couldn't,
    // because in *many* places (eg gentype) the vectors are alloced *inside* old code,
    // so are by default of type Vector<T>.  To get around this I built the polymorphed 
    // class (RKHSVector) and then added this pointer.  If it's nullptr then everything
    // acts like you would expect.  If it's not then virtual functions will detect this
    // and redirect to this pointer.  Thus if this points to RKHSVector then, *even if
    // this is of type Vector<T>*, virtual functions will be appropriately redirected, so
    // it will *act like* it is of type RKHSVector.

public: // because fuck it
    Vector<T> *imoverhere = nullptr;

    const Vector<T> &overhere(void) const
    {
        NiceAssert(imoverhere);

        const Vector<T> *overthere = imoverhere;

        while ( (*overthere).imoverhere )
        {
            overthere = (*overthere).imoverhere;
        }

        return *overthere;
    }

    Vector<T> &overhere(void)
    {
        NiceAssert(imoverhere);

        Vector<T> *overthere = imoverhere;

        while ( (*overthere).imoverhere )
        {
            overthere = (*overthere).imoverhere;
        }

        return *overthere;
    }

    const T *unsafeccontent(void) const { NiceAssert( !infsize() ); if ( !ccontent ) { static thread_local Vector<T> altres;      return altres.unsafecontent(); } return &((*ccontent)(0));    }
          T *unsafecontent (void)       { NiceAssert( !infsize() ); fix(); NiceAssert( content );                                                                  return &((*content)("&",0)); }
    const int *unsafepivot (void) const { NiceAssert( !infsize() ); if ( !pivot    ) { static thread_local Vector<int> altres(0); return altres.unsafepivot();   } return &((*pivot)(0));       }

    int unsafeib(void) const { return iib; }
    int unsafeis(void) const { return iis; }
private:

    const Vector<T> &resize(int i) const; // literally identical to the non-const, but not "nice, so don't use this

    // This may be returned by ind() call.  Might cause issues but unlikely

    const Vector<T> *bkref;      // = nullptr
          DynArray<T> *content;  // = nullptr
    const DynArray<T> *ccontent; // = nullptr
    const DynArray<int> *pivot;  // = nullptr

    // Internal dereferencing operators

          Vector<T> &operator()(const char *dummy, const DynArray<int> &i, int isize, retVector<T> &tmp);
    const Vector<T> &operator()(                   const DynArray<int> &i, int isize, retVector<T> &tmp) const;

    // These internal versions do two steps in one to prevent the need for two retVector arguments.
    // The steps are: res = ((*this)(i))(ib,is,im)

          Vector<T> &operator()(const char *dummy, const DynArray<int> &i, int ib, int is, int im, retVector<T> &tmp);
    const Vector<T> &operator()(                   const DynArray<int> &i, int ib, int is, int im, retVector<T> &tmp) const;

    // Blind constructor: does no allocation, just sets bkref and defaults

    explicit Vector(const char *dummy, const Vector<T> &src) : newln(src.newln), nbase(false), pbase(true), dsize(0), defaulttightalloc(0), iib(0), iis(0), imoverhere(nullptr), bkref(src.bkref), content(nullptr), ccontent(nullptr), pivot(nullptr) { (void) dummy; }
    explicit Vector(const char *dummy)                       : newln('\n'),      nbase(false), pbase(true), dsize(0), defaulttightalloc(0), iib(0), iis(0), imoverhere(nullptr), bkref(nullptr),      content(nullptr), ccontent(nullptr), pivot(nullptr) { (void) dummy; }

    // Dynarray constructor - constructs a (const) vector refering to an
    // external dynamic array.  Result is nominally constant: use with care

    explicit Vector(const DynArray<T> *ccontentsrc) : newln('\n'), nbase(true), pbase(true), dsize(ccontentsrc->array_size()), defaulttightalloc(0), iib(0), iis(1), imoverhere(nullptr), bkref(this), content(nullptr), ccontent(ccontentsrc), pivot(cntintarray(ccontentsrc->array_size())) { NiceAssert( ccontentsrc ); }

    // Like above but assumes ccontentsrc is *basic*, so iis is set zero.

    explicit Vector(const DynArray<T> *ccontentsrc, const char *) : newln('\n'), nbase(true), pbase(true), dsize(INT_MAX-1), defaulttightalloc(0), iib(0), iis(0), imoverhere(nullptr), bkref(this), content(nullptr), ccontent(ccontentsrc), pivot(zerointarray()) { NiceAssert( ccontentsrc ); }

    // Unsafe assignment operator - literally just a raw copy, only to be
    // used in very limited matrix functions!

    virtual Vector<T> &assignover(const Vector<T> &src);

    // Fix bkref

    void fixbkreftree(const Vector<T> *newbkref);
};

template <class T> void qswap(Vector<T> &a, Vector<T> &b)
{
    NiceAssert( a.nbase == false );
    NiceAssert( b.nbase == false );

    qswap(a.dsize            ,b.dsize            );
//    qswap(a.fixsize          ,b.fixsize          );
    qswap(a.nbase            ,b.nbase            );
    qswap(a.pbase            ,b.pbase            );
    qswap(a.iib              ,b.iib              );
    qswap(a.iis              ,b.iis              );
    qswap(a.defaulttightalloc,b.defaulttightalloc);

    qswap(a.newln,     b.newln     );
    qswap(a.imoverhere,b.imoverhere);

    const Vector<T> *bkref;
    DynArray<T> *content;
    const DynArray<T> *ccontent;
    const DynArray<int> *pivot;

    bkref    = a.bkref;    a.bkref    = b.bkref;    b.bkref    = bkref;
    content  = a.content;  a.content  = b.content;  b.content  = content;
    ccontent = a.ccontent; a.ccontent = b.ccontent; b.ccontent = ccontent;
    pivot    = a.pivot;    a.pivot    = b.pivot;    b.pivot    = pivot;

    // The above will have messed up one important thing, namely bkref and
    // bkref in any child vectors.  We must now repair the child trees if
    // they exist

    a.fixbkreftree(&a);
    b.fixbkreftree(&b);
}

template <class T> void qswap(retVector<T> &a, retVector<T> &b)
{
    // Don't want to assert nbase == 0, because it may not be
    // Just reset: this should only be used when a,b are *not* in active use!
    // (eg if you have Vector<retVector<T> >, like in kcache)

    a.reset();
    b.reset();
}



template <class T> void qswap(const Vector<T> *&a, const Vector<T> *&b)
{
    const Vector<T> *c;

    c = a;
    a = b;
    b = c;
}

template <class T> void qswap(Vector<T> *&a, Vector<T> *&b)
{
    Vector<T> *c;

    c = a;
    a = b;
    b = c;
}



// Various functions
//
// max: find max element, put index in i.  If two vectors are given then finds max a-b
// min: find min element, put index in i.  If two vectors are given then finds min a-b
// maxabs: find the |max| element, put index in i.
// minabs: find the |min| element, put index in i.
// sqabsmax: find the |max|*|max| element, put index in i.
// sqabsmin: find the |min|*|min| element, put index in i.
// sum: find the sum of elements in a vector
// orall: find logical OR of all elements in vector
// andall: find logical OR of all elements in vector
// prod: find the product of elements in a vector, top to bottom
// Prod: find the product of elements in a vector, bottom to top
// mean: calculate the mean of.  Ill-defined if vector empty.
// median: calculate the median.  Put the index into i.
//
// innerProduct: calculate the inner product of two vectors conj(a)'.b
// innerProductRevConj: calculate the inner product of two vectors a'.conj(b)
//
// twoProduct: calculate the inner product of two vectors but without conjugating a
// fourProduct: calculate the four-product sum_i (a_i.b_i).(c_i.d_i)  (note order of operation)
// mProduct: calculate the m-product sum_i prod_j (a_{2j,i}.a_{2j,i+1})  (note order of operation)
// Indexed versions: a -> a(n), b -> b(n), c -> c(n), ....  It is assumed that n is sorted from smallest to largest.
// Scaled versions: a -> a./scale, b -> b./scale, c -> c./scale ...
//
// setident: call a.ident()
// setzero: call a.zero()
// setposate: call a.posate()
// setnegate: call a.negate()
// setconj: call a.conj()
//
// angle:    calculate a/abs2(a) (0 if abs2(0) = 0)
// vangle:   calculate a/abs2(a) (defsign if abs2(0) = 0
// seteabs2: calculate the elemetwise absolute of a (ie a_i = ||a_i||)
//
// abs1:   return the 1-norm of the vector
// abs2:   return the square root of the 2-norm of the vector
// absp:   return the p-root of the p-norm of the vector
// absinf: return the inf-norm of the vector
// norm1:  return the 1-norm of the vector ||a||_1
// norm2:  return the 2-norm of the vector ||a||_2^2
// normp:  return the p-norm of the vector ||a||_p^p
//
// sum and mean can be weighted using second argument

template <class T> const T &sum (T &res, const Vector<T> &a, const Vector<double> &weights);
template <class T> const T &mean(T &res, const Vector<T> &a, const Vector<double> &weights);

template <class S, class T> const T &sumb(T &result, const Vector<S> &left_op, const Vector<T> &right_op);

template <class T> T max     (const Vector<T> &a, const Vector<T> &b, int &i);
template <class T> T min     (const Vector<T> &a, const Vector<T> &b, int &i);
template <class T> T maxabs  (const Vector<T> &a, const Vector<T> &b, int &i);
template <class T> T maxabs  (const Vector<T> &a,                     int &i);
template <class T> T minabs  (const Vector<T> &a,                     int &i);
template <class T> T sqabsmax(const Vector<T> &a);
template <class T> T sqabsmin(const Vector<T> &a);
template <class T> T sum     (const Vector<T> &a);
template <class T> T sqsum   (const Vector<T> &a);
template <class T> T orall   (const Vector<T> &a);
template <class T> T andall  (const Vector<T> &a);
template <class T> T logsum  (const Vector<T> &a); // sum of the logs of the vector
template <class T> T prod    (const Vector<T> &a);
template <class T> T Prod    (const Vector<T> &a);
template <class T> T mean    (const Vector<T> &a);
template <class T> T sqmean  (const Vector<T> &a);
template <class T> T vari    (const Vector<T> &a);
template <class T> T stdev   (const Vector<T> &a);

template <class T> T logsum  (const Vector<T> &a, int maxsize); // sum of the logs of the vector up to no more than length maxsize
template <class T> T prod    (const Vector<T> &a, int maxsize); // only do product to max of vector size and maxsize

template <class T> const T &max   (const Vector<T> &a, int &i);
template <class T> const T &min   (const Vector<T> &a, int &i);
template <class T> const T &median(const Vector<T> &a, int &i);

template <class T> const T &max     (T &res, const Vector<T> &a, const Vector<T> &b, int &i);
template <class T> const T &min     (T &res, const Vector<T> &a, const Vector<T> &b, int &i);
template <class T> const T &maxabs  (T &res, const Vector<T> &a,                     int &i);
template <class T> const T &minabs  (T &res, const Vector<T> &a,                     int &i);
template <class T> const T &sqabsmax(T &res, const Vector<T> &a);
template <class T> const T &sqabsmin(T &res, const Vector<T> &a);
template <class T> const T &sum     (T &res, const Vector<T> &a);
template <class T> const T &sqsum   (T &res, const Vector<T> &a);
template <class T> const T &orall   (T &res, const Vector<T> &a);
template <class T> const T &andall  (T &res, const Vector<T> &a);
template <class T> const T &prod    (T &res, const Vector<T> &a);
template <class T> const T &Prod    (T &res, const Vector<T> &a);
template <class T> const T &mean    (T &res, const Vector<T> &a);
template <class T> const T &sqmean  (T &res, const Vector<T> &a);
template <class T> const T &vari    (T &res, const Vector<T> &a);
template <class T> const T &stdev   (T &res, const Vector<T> &a);

template <> inline const double &sum  (double &result, const Vector<double> &left_op);
template <> inline const double &sum  (double &result, const Vector<double> &left_op, const Vector<double> &right_op);
template <> inline const double &sqsum(double &result, const Vector<double> &left_op);
template <> inline const double &sumb (double &result, const Vector<double> &left_op, const Vector<double> &right_op);

template <class T>       T       abssum(             const Vector<T>      &a);
template <class T> const T      &abssum(T      &res, const Vector<T>      &a);
template <> inline const double &abssum(double &res, const Vector<double> &a);

template <class T> T &innerProduct                         (T &res,                       const Vector<T> &a, const Vector<T> &b                        );
template <class T> T &innerProductRevConj                  (T &res,                       const Vector<T> &a, const Vector<T> &b                        );
template <class T> T &innerProductScaled                   (T &res,                       const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale);
template <class T> T &innerProductScaledRevConj            (T &res,                       const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale);
template <class T> T &innerProductRightScaled              (T &res,                       const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale);
template <class T> T &innerProductRightScaledRevConj       (T &res,                       const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale);
template <class T> T &innerProductLeftScaled               (T &res,                       const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale);
template <class T> T &innerProductLeftScaledRevConj        (T &res,                       const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale);

template <class T> T &indexedinnerProduct                  (T &res, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b                        );
template <class T> T &indexedinnerProductScaled            (T &res, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale);
template <class T> T &indexedinnerProductRevConj           (T &res, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b                        );
template <class T> T &indexedinnerProductScaledRevConj     (T &res, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale);
template <class T> T &indexedinnerProductRightScaled       (T &res, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale);
template <class T> T &indexedinnerProductRightScaledRevConj(T &res, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale);
template <class T> T &indexedinnerProductLeftScaled        (T &res, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale);
template <class T> T &indexedinnerProductLeftScaledRevConj (T &res, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale);

template <class T> T &twoProductRightScaled        (T &res,                       const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale);
template <class T> T &twoProductLeftScaled         (T &res,                       const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale);
template <class T> T &indexedtwoProductRightScaled (T &res, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale);
template <class T> T &indexedtwoProductLeftScaled  (T &res, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale);

template <class T> T &oneProduct  (T &res, const Vector<T> &a);
template <class T> T &twoProduct  (T &res, const Vector<T> &a, const Vector<T> &b);
template <class T> T &threeProduct(T &res, const Vector<T> &a, const Vector<T> &b, const Vector<T> &c);
template <class T> T &fourProduct (T &res, const Vector<T> &a, const Vector<T> &b, const Vector<T> &c, const Vector<T> &d);
template <class T> T &mProduct    (T &res, const Vector<const Vector <T> *> &a);

template <class T> T &oneProductScaled  (T &res, const Vector<T> &a, const Vector<T> &scale);
template <class T> T &twoProductScaled  (T &res, const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale);
template <class T> T &threeProductScaled(T &res, const Vector<T> &a, const Vector<T> &b, const Vector<T> &c, const Vector<T> &scale);
template <class T> T &fourProductScaled (T &res, const Vector<T> &a, const Vector<T> &b, const Vector<T> &c, const Vector<T> &d, const Vector<T> &scale);
template <class T> T &mProductScaled    (T &res, const Vector<const Vector <T> *> &a, const Vector<T> &scale);

template <class T> double &innerProductAssumeReal(double &res, const Vector<T> &a, const Vector<T> &b);

template <class T> double &oneProductAssumeReal  (double &res, const Vector<T> &a);
template <class T> double &twoProductAssumeReal  (double &res, const Vector<T> &a, const Vector<T> &b);
template <class T> double &threeProductAssumeReal(double &res, const Vector<T> &a, const Vector<T> &b, const Vector<T> &c);
template <class T> double &fourProductAssumeReal (double &res, const Vector<T> &a, const Vector<T> &b, const Vector<T> &c, const Vector<T> &d);
template <class T> double &mProductAssumeReal    (double &res, const Vector<const Vector <T> *> &a);

template <class T> T &indexedoneProduct  (T &res, const Vector<int> &n, const Vector<T> &a);
template <class T> T &indexedtwoProduct  (T &res, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b);
template <class T> T &indexedthreeProduct(T &res, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b, const Vector<T> &c);
template <class T> T &indexedfourProduct (T &res, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b, const Vector<T> &c, const Vector<T> &d);
template <class T> T &indexedmProduct    (T &res, const Vector<int> &n, const Vector<const Vector <T> *> &a);

template <class T> T &indexedoneProductScaled  (T &res, const Vector<int> &n, const Vector<T> &a, const Vector<T> &scale);
template <class T> T &indexedtwoProductScaled  (T &res, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale);
template <class T> T &indexedthreeProductScaled(T &res, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b, const Vector<T> &c, const Vector<T> &scale);
template <class T> T &indexedfourProductScaled (T &res, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b, const Vector<T> &c, const Vector<T> &d, const Vector<T> &scale);
template <class T> T &indexedmProductScaled    (T &res, const Vector<int> &n, const Vector<const Vector <T> *> &a, const Vector<T> &scale);

template <> inline double &innerProductAssumeReal(double &res, const Vector<double> &a, const Vector<double> &b);

template <> inline double &oneProductAssumeReal  (double &res, const Vector<double> &a);
template <> inline double &twoProductAssumeReal  (double &res, const Vector<double> &a, const Vector<double> &b);
template <> inline double &threeProductAssumeReal(double &res, const Vector<double> &a, const Vector<double> &b, const Vector<double> &c);
template <> inline double &fourProductAssumeReal (double &res, const Vector<double> &a, const Vector<double> &b, const Vector<double> &c, const Vector<double> &d);
template <> inline double &mProductAssumeReal    (double &res, const Vector<const Vector <double> *> &a);

template <> inline double &oneProduct  (double &res, const Vector<double> &a);
template <> inline double &twoProduct  (double &res, const Vector<double> &a, const Vector<double> &b);
template <> inline double &threeProduct(double &res, const Vector<double> &a, const Vector<double> &b, const Vector<double> &c);
template <> inline double &fourProduct (double &res, const Vector<double> &a, const Vector<double> &b, const Vector<double> &c, const Vector<double> &d);
template <> inline double &mProduct    (double &res, const Vector<const Vector <double> *> &a);

template <> inline double &innerProduct       (double &res, const Vector<double> &a, const Vector<double> &b);
template <> inline double &innerProductRevConj(double &res, const Vector<double> &a, const Vector<double> &b);

template <class T> Vector<T> &setident        (Vector<T> &a);
template <class T> Vector<T> &setzero         (Vector<T> &a);
template <class T> Vector<T> &setzeropassive  (Vector<T> &a);
template <class T> Vector<T> &setposate       (Vector<T> &a);
template <class T> Vector<T> &setnegate       (Vector<T> &a);
template <class T> Vector<T> &setconj         (Vector<T> &a);
template <class T> Vector<T> &setrand         (Vector<T> &a);
template <class T> Vector<T> &postProInnerProd(Vector<T> &a) { return a; }

template <class T> Vector<T> *&setident (Vector<T> *&a) { NiceThrow("Whatever"); return a; }
template <class T> Vector<T> *&setzero  (Vector<T> *&a) { return a = nullptr; }
template <class T> Vector<T> *&setposate(Vector<T> *&a) { return a; }
template <class T> Vector<T> *&setnegate(Vector<T> *&a) { NiceThrow("I reject your reality and substitute my own"); return a; }
template <class T> Vector<T> *&setconj  (Vector<T> *&a) { NiceThrow("Mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm"); return a; }
template <class T> Vector<T> *&setrand  (Vector<T> *&a) { NiceThrow("Blippity Blappity Blue"); return a; }

template <class T> Vector<T> *&postProInnerProd(Vector<T> *&a) { return a; }

template <class T> const Vector<T> *&setident (const Vector<T> *&a) { NiceThrow("Whatever"); return a; }
template <class T> const Vector<T> *&setzero  (const Vector<T> *&a) { return a = nullptr; }
template <class T> const Vector<T> *&setposate(const Vector<T> *&a) { return a; }
template <class T> const Vector<T> *&setnegate(const Vector<T> *&a) { NiceThrow("I reject your reality and substitute my own"); return a; }
template <class T> const Vector<T> *&setconj  (const Vector<T> *&a) { NiceThrow("Mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm"); return a; }
template <class T> const Vector<T> *&setrand  (const Vector<T> *&a) { NiceThrow("Blippity Blappity Blue"); return a; }

template <class T> const Vector<T> *&postProInnerProd(const Vector<T> *&a) { return a; }

template <class S> Vector<S>       angle   (                const Vector<S> &a);
template <class S> Vector<S>      &angle   (Vector<S> &res, const Vector<S> &a);
template <class S> Vector<double>  eabs2   (                const Vector<S> &a);
template <class S> Vector<S>       vangle  (                const Vector<S> &a, const Vector<S> &defsign);
template <class S> Vector<S>      &vangle  (Vector<S> &res, const Vector<S> &a, const Vector<S> &defsign);
template <class S> Vector<double> &seteabs2(                      Vector<S> &a);




// Elementwise pow

template <class T> Vector<T> &epow(Vector<T> &res, const Vector<T> &a, T c);



// Kronecker products

template <class T> Vector<T> &kronprod(Vector<T> &res, const Vector<T> &a, const Vector<T> &b);
template <class T> Vector<T> &kronpow (Vector<T> &res, const Vector<T> &a, int n);



// NaN and inf tests

template <class T> int testisvnan(const Vector<T> &x);
template <class T> int testisinf (const Vector<T> &x);
template <class T> int testispinf(const Vector<T> &x);
template <class T> int testisninf(const Vector<T> &x);



// Second-rate incrementor related stuff - return 1 if looped back to zero

inline int getnext(Vector<int> &i, int max);
inline int getnext(Vector<int> &i, int max)
{
    int j = 0;
    int notdone = 1;

    while ( notdone && ( j < i.size() ) )
    {
        ++(i("&",j));

        if ( i.v(j) > max )
        {
            i.sv(j,0);
            ++j;
        }

        else
        {
            notdone = 0;
        }
    }

    return notdone;
}



// Random permutation function and random fill

inline Vector<int> &randPerm(Vector<int> &res);

template <class T> Vector<T> &randrfill(Vector<T> &res);
template <class T> Vector<T> &randbfill(Vector<T> &res);
template <class T> Vector<T> &randBfill(Vector<T> &res);
template <class T> Vector<T> &randgfill(Vector<T> &res);
template <class T> Vector<T> &randpfill(Vector<T> &res);
template <class T> Vector<T> &randufill(Vector<T> &res);
template <class T> Vector<T> &randefill(Vector<T> &res);
template <class T> Vector<T> &randGfill(Vector<T> &res);
template <class T> Vector<T> &randwfill(Vector<T> &res);
template <class T> Vector<T> &randxfill(Vector<T> &res);
template <class T> Vector<T> &randnfill(Vector<T> &res);
template <class T> Vector<T> &randlfill(Vector<T> &res);
template <class T> Vector<T> &randcfill(Vector<T> &res);
template <class T> Vector<T> &randCfill(Vector<T> &res);
template <class T> Vector<T> &randffill(Vector<T> &res);
template <class T> Vector<T> &randtfill(Vector<T> &res);



// Conversion from strings

template <class T> Vector<T> &atoVector(Vector<T> &dest, const std::string &src);

// Mathematical operator overloading
//
// NB: in general it is wise to avoid use of non-assignment operators (ie.
//     those which do not return a reference) as there may be a
//     computational hit when constructors (and possibly copy constructors)
//     are called.
//
// +  posation          - unary, return rvalue
// -  negation          - unary, return rvalue
//
// NB: - all unary operators are elementwise

template <class T> Vector<T>  operator+ (const Vector<T> &left_op);
template <class T> Vector<T>  operator- (const Vector<T> &left_op);

// +  addition           - binary, return rvalue
// -  subtraction        - binary, return rvalue
// *  multiplication     - binary, return rvalue
// /  division           - binary, return rvalue
//
// NB: - adding a scalar to a vector adds the scalar to all elements of the
//       vector.
//     - we don't assume commutativity over T, so division is not well defined
//     - multiplying two vectors performs element-wise multiplication.
//     - division: vector/vector will do elementwise division                (and return reference to left_op)
//                 vector/scalar will do right division (vector*inv(scalar)) (and return reference to left_op)
//                 scalar/vector will do left division (inv(scalar)*vector)  (and return reference to right_op)
//
//

template <class T> Vector<T>  operator+ (const Vector<T> &left_op, const Vector<T> &right_op);
template <class T> Vector<T>  operator- (const Vector<T> &left_op, const Vector<T> &right_op);
template <class T> Vector<T>  operator* (const Vector<T> &left_op, const Vector<T> &right_op);
template <class T> Vector<T>  operator/ (const Vector<T> &left_op, const Vector<T> &right_op);

template <class T> Vector<T>  operator+ (const Vector<T> &left_op, const T         &right_op);
template <class T> Vector<T>  operator- (const Vector<T> &left_op, const T         &right_op);
template <class T> Vector<T>  operator* (const Vector<T> &left_op, const T         &right_op);
template <class T> Vector<T>  operator/ (const Vector<T> &left_op, const T         &right_op);

template <class T> Vector<T>  operator+ (const T         &left_op, const Vector<T> &right_op);
template <class T> Vector<T>  operator- (const T         &left_op, const Vector<T> &right_op);
template <class T> Vector<T>  operator* (const T         &left_op, const Vector<T> &right_op);
template <class T> Vector<T>  operator/ (const T         &left_op, const Vector<T> &right_op);

// +=  additive       assignment - binary, return lvalue
// -=  subtractive    assignment - binary, return lvalue
// *=  multiplicative assignment - binary, return lvalue
// /=  divisive       assignment - binary, return lvalue
//
// NB: - adding a scalar to a vector adds the scalar to all elements of the
//       vector.
//     - left-shift and right-shift operate elementwise.
//     - when left_op is not a vector, the result is stored in right_op and returned as a reference
//     - it is assumed that addition and subtraction are commutative
//     - scalar /= vector does left division (that is, vector = inv(scalar)*vector).
//
//

template <         class T> Vector<T> &operator+=(Vector<T> &left_op, const Vector<T> &right_op);
template <         class T> Vector<T> &operator-=(Vector<T> &left_op, const Vector<T> &right_op);
template <class S, class T> Vector<S> &operator*=(Vector<S> &left_op, const Vector<T> &right_op);
template <class S, class T> Vector<S> &operator/=(Vector<S> &left_op, const Vector<T> &right_op); // elementwise for chol.hpp

template <class T> Vector<T> &operator+=(Vector<T> &left_op, const T &right_op);
template <class T> Vector<T> &operator-=(Vector<T> &left_op, const T &right_op);
template <class T> Vector<T> &operator*=(Vector<T> &left_op, const T &right_op);
template <class T> Vector<T> &operator/=(Vector<T> &left_op, const T &right_op);

template <class T> Vector<T> &operator+=(const T &left_op, Vector<T> &right_op);
template <class T> Vector<T> &operator-=(const T &left_op, Vector<T> &right_op);
template <class T> Vector<T> &operator*=(const T &left_op, Vector<T> &right_op);
template <class T> Vector<T> &operator/=(const T &left_op, Vector<T> &right_op);

template <> inline Vector<double> &operator+=(Vector<double> &left_op, const Vector<double> &right_op);
template <> inline Vector<double> &operator-=(Vector<double> &left_op, const Vector<double> &right_op);
template <> inline Vector<double> &operator*=(Vector<double> &left_op, const Vector<double> &right_op);
template <> inline Vector<double> &operator/=(Vector<double> &left_op, const Vector<double> &right_op); // elementwise for chol.hpp

template <> inline Vector<double> &operator+=(Vector<double> &left_op, const double &right_op);
template <> inline Vector<double> &operator-=(Vector<double> &left_op, const double &right_op);
template <> inline Vector<double> &operator*=(Vector<double> &left_op, const double &right_op);
template <> inline Vector<double> &operator/=(Vector<double> &left_op, const double &right_op);

template <> inline Vector<double> &operator*=(const double &left_op, Vector<double> &right_op);
template <> inline Vector<double> &operator/=(const double &left_op, Vector<double> &right_op);

// Related non-commutative operations
//
// leftmult:  equivalent to *=
// rightmult: like *=, but result is stored in right_op and ref to right_op is returned

template <class T> Vector<T> &leftmult (Vector<T> &left_op, const Vector<T> &right_op);
template <class T> Vector<T> &leftmult (Vector<T> &left_op, const T         &right_op);
template <class T> Vector<T> &rightmult(const Vector<T> &left_op, Vector<T> &right_op);
template <class T> Vector<T> &rightmult(const T         &left_op, Vector<T> &right_op);

//template <> inline Vector<double> &leftmult (Vector<double> &left_op, const Vector<double> &right_op);
//template <> inline Vector<double> &leftmult (Vector<double> &left_op, const double         &right_op);
template <> inline Vector<double> &rightmult(const Vector<double> &left_op, Vector<double> &right_op);
template <> inline Vector<double> &rightmult(const double         &left_op, Vector<double> &right_op);

// Relational operator overloading

template <class T> int operator==(const Vector<T> &left_op, const Vector<T> &right_op);
template <class T> int operator==(const Vector<T> &left_op, const T         &right_op);
template <class T> int operator==(const T         &left_op, const Vector<T> &right_op);

template <class T> int operator!=(const Vector<T> &left_op, const Vector<T> &right_op);
template <class T> int operator!=(const Vector<T> &left_op, const T         &right_op);
template <class T> int operator!=(const T         &left_op, const Vector<T> &right_op);

template <class T> int operator< (const Vector<T> &left_op, const Vector<T> &right_op);
template <class T> int operator< (const Vector<T> &left_op, const T         &right_op);
template <class T> int operator< (const T         &left_op, const Vector<T> &right_op);

template <class T> int operator<=(const Vector<T> &left_op, const Vector<T> &right_op);
template <class T> int operator<=(const Vector<T> &left_op, const T         &right_op);
template <class T> int operator<=(const T         &left_op, const Vector<T> &right_op);

template <class T> int operator> (const Vector<T> &left_op, const Vector<T> &right_op);
template <class T> int operator> (const Vector<T> &left_op, const T         &right_op);
template <class T> int operator> (const T         &left_op, const Vector<T> &right_op);

template <class T> int operator>=(const Vector<T> &left_op, const Vector<T> &right_op);
template <class T> int operator>=(const Vector<T> &left_op, const T         &right_op);
template <class T> int operator>=(const T         &left_op, const Vector<T> &right_op);











// return 1 if all elements in vector fit within range

inline int checkRange(int lb, int ub, const Vector<int> &x);
inline int checkRange(int lb, int ub, const Vector<int> &x)
{
    for ( int i = 0 ; i < x.size() ; ++i )
    {
        if ( ( x.v(i) < lb ) || ( x.v(i) > ub ) )
        {
            return 0;
        }
    }

    return 1;
}


template <class S, class U>
bool shareBase(const Vector<S> &thus, const Vector<U> &that)
{
    return ( (void *) thus.bkref == (void *) that.bkref );
}



























// Now for the actual code


template <class T>
retVector<T> &retVector<T>::reset(void)
{
    if ( !(Vector<T>::pbase) && Vector<T>::pivot )
    {
        MEMDEL(Vector<T>::pivot);
    }

    Vector<T>::pbase = true;
    Vector<T>::pivot = nullptr;

    if ( !(Vector<T>::nbase) && Vector<T>::content )
    {
        MEMDEL(Vector<T>::content);
    }

    return *this;
}

template <class T>
retVector<T> &retVector<T>::reset(Vector<T> &cover)
{
    if ( !(Vector<T>::pbase) && Vector<T>::pivot )
    {
        MEMDEL(Vector<T>::pivot);
    }

    Vector<T>::pbase = true;
    Vector<T>::pivot = nullptr;

    if ( !(Vector<T>::nbase) && Vector<T>::content )
    {
        MEMDEL(Vector<T>::content);
    }

    Vector<T>::bkref             = cover.bkref;
    Vector<T>::nbase             = true;
    Vector<T>::content           = cover.content;
    Vector<T>::ccontent          = cover.ccontent;
    Vector<T>::newln             = cover.newln;
    Vector<T>::imoverhere        = nullptr;
    Vector<T>::defaulttightalloc = 0;

    return *this;
}

template <class T>
DynArray<int> *retVector<T>::reset_p(Vector<T> &cover, int pivotsize)
{
    // this version allocates pivot(pivotsize) (or resizes if pbase) and sets pbase = true

    DynArray<int> *resval = nullptr;

    if ( !(Vector<T>::pbase) && Vector<T>::pivot )
    {
        resval = &((const_cast<DynArray<int> &>(*(Vector<T>::pivot)).resize(pivotsize)));
    }

    else
    {
	MEMNEW(resval,DynArray<int>);
        (*resval) = { nullptr,0,0,0,false,false,false };
        (*resval).resize(pivotsize);
        (*resval).useSlackAllocation();

        Vector<T>::pbase = false;
        Vector<T>::pivot = resval;
    }

    if ( !(Vector<T>::nbase) && Vector<T>::content )
    {
        MEMDEL(Vector<T>::content);
    }

    Vector<T>::bkref             = cover.bkref;
    Vector<T>::nbase             = true;
    Vector<T>::content           = cover.content;
    Vector<T>::ccontent          = cover.ccontent;
    Vector<T>::newln             = cover.newln;
    Vector<T>::imoverhere        = nullptr;
    Vector<T>::defaulttightalloc = 0;

    return resval;
}


template <class T>
retVector<T> &retVector<T>::creset(const Vector<T> &cover)
{
    if ( !(Vector<T>::pbase) && Vector<T>::pivot )
    {
        MEMDEL(Vector<T>::pivot);
    }

    Vector<T>::pbase = true;
    Vector<T>::pivot = nullptr;

    if ( !(Vector<T>::nbase) && Vector<T>::content )
    {
        MEMDEL(Vector<T>::content);
    }

    Vector<T>::bkref             = cover.bkref;
    Vector<T>::nbase             = true;
    Vector<T>::content           = nullptr;
    Vector<T>::ccontent          = cover.ccontent;
    Vector<T>::newln             = cover.newln;
    Vector<T>::imoverhere        = nullptr;
    Vector<T>::defaulttightalloc = 0;

    return *this;
}

template <class T>
DynArray<int> *retVector<T>::creset_p(const Vector<T> &cover, int pivotsize)
{
    // this version allocates pivot(pivotsize) (or resizes if pbase) and sets pbase = true

    DynArray<int> *resval = nullptr;

    if ( !(Vector<T>::pbase) && Vector<T>::pivot )
    {
        resval = &((const_cast<DynArray<int> &>(*(Vector<T>::pivot)).resize(pivotsize)));
    }

    else
    {
	MEMNEW(resval,DynArray<int>);
        (*resval) = { nullptr,0,0,0,false,false,false };
        (*resval).resize(pivotsize);
        (*resval).useSlackAllocation();

        Vector<T>::pbase = false;
        Vector<T>::pivot = resval;
    }

    if ( !(Vector<T>::nbase) && Vector<T>::content )
    {
        MEMDEL(Vector<T>::content);
    }

    Vector<T>::bkref             = cover.bkref;
    Vector<T>::nbase             = true;
    Vector<T>::content           = nullptr;
    Vector<T>::ccontent          = cover.ccontent;
    Vector<T>::newln             = cover.newln;
    Vector<T>::imoverhere        = nullptr;
    Vector<T>::defaulttightalloc = 0;

    return resval;
}



template <class T>
void Vector<T>::fixbkreftree(const Vector<T> *newbkref)
{
    bkref = newbkref;
}


// Constructors and Destructors

template <class T>
Vector<T>::Vector(int size, const T *src, int tightalloc) : newln('\n'),
                                                            nbase(false),
                                                            pbase(true),
                                                            dsize(size),
                                                            defaulttightalloc(tightalloc),
                                                            iib(0),
                                                            iis(1),
                                                            imoverhere(nullptr),
                                                            bkref(this),
                                                            content(nullptr),
                                                            ccontent(nullptr),
                                                            pivot(cntintarray(size))
{
    NiceAssert( size >= 0 );

    MEMNEW(content,DynArray<T>);
    (*content) = { nullptr,0,0,defaulttightalloc,false,false,false };
    (*content).resize(dsize);
    ccontent = content;

    NiceAssert( content );

    if ( src )
    {
        for ( int i = 0 ; i < size ; ++i )
        {
            (*this).set(i,src[i]);
        }
    }
}

template <class T>
Vector<T>::Vector(int size, const T &src, int tightalloc) : newln('\n'),
                                                            nbase(false),
                                                            pbase(true),
                                                            dsize(size),
                                                            defaulttightalloc(tightalloc),
                                                            iib(0),
                                                            iis(1),
                                                            imoverhere(nullptr),
                                                            bkref(this),
                                                            content(nullptr),
                                                            ccontent(nullptr),
                                                            pivot(cntintarray(size))
{
    NiceAssert( size >= 0 );

    MEMNEW(content,DynArray<T>);
    (*content) = { nullptr,0,0,defaulttightalloc,false,false,false };
    (*content).resize(dsize);
    ccontent = content;

    NiceAssert( content );

    {
        for ( int i = 0 ; i < size ; ++i )
        {
            (*this).set(i,src);
        }
    }
}

template <class T>
Vector<T>::Vector(const Vector<T> &src) : newln(src.newln),
                                          nbase(false),
                                          pbase(true),
                                          dsize(src.size()),
                                          defaulttightalloc(src.defaulttightalloc),
                                          iib(0),
                                          iis(1),
                                          imoverhere(nullptr),
                                          bkref(this),
                                          content(nullptr),
                                          ccontent(nullptr),
                                          pivot(cntintarray(src.size()))
{
    if ( src.imoverhere )
    {
        imoverhere = (src.overhere()).makeDup();
    }

    else if ( src.infsize() )
    {
        imoverhere = src.makeDup();
    }

    MEMNEW(content,DynArray<T>);
    (*content) = { nullptr,0,0,defaulttightalloc,false,false,false };
    (*content).resize(dsize);
    ccontent = content;

    NiceAssert( content );

    *this = src;
}

template <class T>
Vector<T>::~Vector()
{
    if ( imoverhere )
    {
        MEMDEL(imoverhere);
    }

    if ( !nbase && content )
    {
        MEMDEL(content);
    }

    if ( !pbase && pivot )
    {
        MEMDEL(pivot);
    }
}



// Assignment

template <> inline Vector<double> &Vector<double>::assign(const Vector<double> &src);
template <> inline Vector<double> &Vector<double>::assign(const Vector<double> &src)
{
    defaulttightalloc = src.defaulttightalloc;

    if ( imoverhere && src.imoverhere )
    {
              Vector<double> &thisover = overhere();
        const Vector<double> &srcover  = src.overhere();

        if ( thisover.type() == srcover.type() )
        {
            thisover.assign(srcover);
        }

        else
        {
            MEMDEL(imoverhere);
            imoverhere = srcover.makeDup();
        }
    }

    else if ( imoverhere && src.infsize() )
    {
        Vector<double> &thisover = overhere();

        if ( thisover.type() == src.type() )
        {
            thisover.assign(src);
        }

        else
        {
            MEMDEL(imoverhere);
            imoverhere = src.makeDup();
        }
    }

    else if ( !imoverhere && src.imoverhere )
    {
        const Vector<double> &srcover = src.overhere();

        resize(0);
        imoverhere = srcover.makeDup();
    }

    else if ( !imoverhere && src.infsize() )
    {
        resize(0);
        imoverhere = src.makeDup();
    }

    else if ( !content )
    {
        fix();

        assign(src);
    }

    else if ( shareBase(src) )
    {
        Vector<double> temp;

        temp  = src;
        assign(temp);
    }

    else
    {
        if ( imoverhere )
        {
            MEMDEL(imoverhere);
            imoverhere = nullptr;
        }

        int srcsize = src.size();

        if ( !nbase )
        {
            resize(srcsize);

            if ( !(src.base()) && content && src.contentalloced() )
            {
                if ( src.contentarray_hold() )
                {
                    // Design decision: preallocation is duplicated

                    content->prealloc(src.contentarray_alloc());
                }
            }
        }

        NiceAssert( dsize == srcsize );

        for ( int i = 0 ; i < dsize ; ++i )
        {
            //(*this)("&",i) = src(i);
            sv(i,src.v(i));
        }
    }

    return *this;
}

template <> inline Vector<int> &Vector<int>::assign(const Vector<int> &src);
template <> inline Vector<int> &Vector<int>::assign(const Vector<int> &src)
{
    defaulttightalloc = src.defaulttightalloc;

    if ( imoverhere && src.imoverhere )
    {
              Vector<int> &thisover = overhere();
        const Vector<int> &srcover  = src.overhere();

        if ( thisover.type() == srcover.type() )
        {
            thisover.assign(srcover);
        }

        else
        {
            MEMDEL(imoverhere);
            imoverhere = srcover.makeDup();
        }
    }

    else if ( imoverhere && src.infsize() )
    {
        Vector<int> &thisover = overhere();

        if ( thisover.type() == src.type() )
        {
            thisover.assign(src);
        }

        else
        {
            MEMDEL(imoverhere);
            imoverhere = src.makeDup();
        }
    }

    else if ( !imoverhere && src.imoverhere )
    {
        const Vector<int> &srcover  = src.overhere();

        resize(0);
        imoverhere = srcover.makeDup();
    }

    else if ( !imoverhere && src.infsize() )
    {
        resize(0);
        imoverhere = src.makeDup();
    }

    else if ( !content )
    {
        fix();

        assign(src);
    }

    else if ( shareBase(src) )
    {
        Vector<int> temp;

        temp  = src;
        assign(temp);
    }

    else
    {
        if ( imoverhere )
        {
            MEMDEL(imoverhere);
            imoverhere = nullptr;
        }

        int srcsize = src.size();

        if ( !nbase )
        {
            resize(srcsize);

            if ( !(src.base()) && content && src.contentalloced() )
            {
                if ( src.contentarray_hold() )
                {
                    // Design decision: preallocation is duplicated

                    content->prealloc(src.contentarray_alloc());
                }
            }
        }

        NiceAssert( dsize == srcsize );

        for ( int i = 0 ; i < dsize ; ++i )
        {
            //(*this)("&",i) = src(i);
            sv(i,src.v(i));
        }
    }

    return *this;
}

template <class T>
Vector<T> &Vector<T>::assign(const Vector<T> &src)
{
    defaulttightalloc = src.defaulttightalloc;

    if ( imoverhere && src.imoverhere )
    {
              Vector<T> &thisover = overhere();
        const Vector<T> &srcover  = src.overhere();

        if ( thisover.type() == srcover.type() )
        {
            thisover.assign(srcover);
        }

        else
        {
            MEMDEL(imoverhere);
            imoverhere = srcover.makeDup();
        }
    }

    else if ( imoverhere && src.infsize() )
    {
        Vector<T> &thisover = overhere();

        if ( thisover.type() == src.type() )
        {
            thisover.assign(src);
        }

        else
        {
            MEMDEL(imoverhere);
            imoverhere = src.makeDup();
        }
    }

    else if ( !imoverhere && src.imoverhere )
    {
        const Vector<T> &srcover  = src.overhere();

        resize(0);
        imoverhere = srcover.makeDup();
    }

    else if ( !imoverhere && src.infsize() )
    {
        resize(0);
        imoverhere = src.makeDup();
    }

    else if ( !content )
    {
        fix();

        assign(src);
    }

    else if ( shareBase(src) )
    {
        Vector<T> temp;

        temp  = src;
        assign(temp);
    }

    else
    {
        if ( imoverhere )
        {
            MEMDEL(imoverhere);
            imoverhere = nullptr;
        }

        int srcsize = src.size();
        int i;

        if ( !nbase )
        {
            resize(srcsize);

            if ( !(src.base()) && content && src.contentalloced() )
            {
                if ( src.contentarray_hold() )
                {
                    // Design decision: preallocation is duplicated

                    content->prealloc(src.contentarray_alloc());
                }
            }
        }

        NiceAssert( dsize == srcsize );

        for ( i = 0 ; i < dsize ; ++i )
        {
            (*this).set(i,src(i));
        }
    }

    return *this;
}

template <class T>
Vector<T> &Vector<T>::assignover(const Vector<T> &src)
{
    dsize = src.dsize;
    defaulttightalloc = src.defaulttightalloc;

    iib = src.iib;
    iis = src.iis;

    nbase = src.nbase;
    pbase = src.pbase;

    newln = src.newln;

    imoverhere = src.imoverhere;

    bkref = src.bkref;

    content  = src.content;
    ccontent = src.ccontent;

    pivot = src.pivot;

    return *this;
}

template <class S, class T>
inline int aresame(T *, S *);

template <class S, class T>
inline int aresame(T *, S *)
{
    return 0;
}

/*  This can't work.  The cast (const T &) will fail (eg if S == double), and it will fail silently!  Instead each case of this is specialised as required (see gentype.h, end of, for use)
template <class T>
template <class S>
Vector<T> &Vector<T>::castassign(const Vector<S> &src)
{
    NiceAssert( !infsize() );

    if ( imoverhere )
    {
        MEMDEL(imoverhere);
        imoverhere = nullptr;
    }

    if ( src.imoverhere )
    {
        T *a = nullptr;
        S *b = nullptr;

        if ( !aresame(a,b) )
        {
            NiceThrow("Can't do that");
        }

        else
        {
            Vector<S> *tmp;

            tmp = (*(src.imoverhere)).makeDup();

            imoverhere = (Vector<T> *) ((void *) tmp);
        }
    }

    else if ( src.infsize() )
    {
        T *a = nullptr;
        S *b = nullptr;

        if ( !aresame(a,b) )
        {
            NiceThrow("Can't do that");
        }

        else
        {
            Vector<S> *tmp;

            tmp = src.makeDup();

            imoverhere = (Vector<T> *) ((void *) tmp);
        }
    }

    else
    {
        if ( shareBase(src) )
        {
            Vector<S> temp;

            temp  = src;
            castassign(temp);
        }

        else
        {
            int srcsize = src.size();
            int i;

            if ( !nbase )
            {
                resize(srcsize);

                if ( !(src.base()) && content && src.contentalloced() )
                {
                    if ( src.contentarray_hold() )
                    {
                        // Design decision: preallocation is duplicated

                        content->prealloc(src.contentarray_alloc());
                    }
                }
            }

            NiceAssert( dsize == srcsize );

            //if ( dsize )
            {
                for ( i = 0 ; i < dsize ; ++i )
                {
                    (*this)("&",i) = (const T &) src(i);
                }
            }
        }
    }

    return *this;
}
*/

template <> inline Vector<double> &Vector<double>::assign(const double &src);
template <> inline Vector<double> &Vector<double>::assign(const double &src)
{
    NiceAssert( !infsize() );

    if ( imoverhere )
    {
        overhere().assign(src);
    }

    else
    {
        double lsrc(src);

        for ( int i = 0 ; i < dsize ; ++i )
        {
            (*this).sv(i,lsrc);
        }
    }

    return *this;
}

template <> inline Vector<int> &Vector<int>::assign(const int &src);
template <> inline Vector<int> &Vector<int>::assign(const int &src)
{
    NiceAssert( !infsize() );

    if ( imoverhere )
    {
        overhere().assign(src);
    }

    else
    {
        int lsrc(src);

        for ( int i = 0 ; i < dsize ; ++i )
        {
            (*this).sv(i,lsrc);
        }
    }

    return *this;
}

template <class T>
Vector<T> &Vector<T>::assign(const T &src)
{
    NiceAssert( !infsize() );

    if ( imoverhere )
    {
        overhere().assign(src);
    }

    else
    {
        for ( int i = 0 ; i < dsize ; ++i )
        {
            (*this).set(i,src);
        }
    }

    return *this;
}


// Basic operations.

template <> inline Vector<double> &Vector<double>::ident(void);
template <> inline Vector<double> &Vector<double>::ident(void)
{
    NiceAssert( !infsize() );

    for ( int i = 0 ; i < dsize ; ++i )
    {
        sv(i,1);
    }

    return *this;
}

template <> inline Vector<int> &Vector<int>::ident(void);
template <> inline Vector<int> &Vector<int>::ident(void)
{
    NiceAssert( !infsize() );

    for ( int i = 0 ; i < dsize ; ++i )
    {
        sv(i,1);
    }

    return *this;
}

template <class T>
Vector<T> &Vector<T>::ident(void)
{
    NiceAssert( !infsize() );

    for ( int i = 0 ; i < dsize ; ++i )
    {
        setident((*this)("&",i));
    }

    return *this;
}

template <> inline Vector<double> &Vector<double>::zero(void);
template <> inline Vector<double> &Vector<double>::zero(void)
{
    if ( imoverhere )
    {
        overhere().zero();
    }

    else
    {
        for ( int i = 0 ; i < dsize ; ++i )
        {
            sv(i,0);
        }
    }

    return *this;
}

template <> inline Vector<int> &Vector<int>::zero(void);
template <> inline Vector<int> &Vector<int>::zero(void)
{
    if ( imoverhere )
    {
        overhere().zero();
    }

    else
    {
        for ( int i = 0 ; i < dsize ; ++i )
        {
            sv(i,0);
        }
    }

    return *this;
}

template <class T>
Vector<T> &Vector<T>::zero(void)
{
    if ( imoverhere )
    {
        overhere().zero();
    }

    else
    {
        for ( int i = 0 ; i < dsize ; ++i )
        {
            setzero((*this)("&",i));
        }
    }

    return *this;
}

template <class T>
Vector<T> &Vector<T>::zeropassive(void)
{
    NiceAssert( !infsize() );

    for ( int i = 0 ; i < dsize ; ++i )
    {
        setzeropassive((*this)("&",i));
    }

    return *this;
}

template <class T>
Vector<T> &Vector<T>::zero(int i)
{
    NiceAssert( !infsize() );
    NiceAssert( i >= 0 );
    NiceAssert( i < size() );

    setzero((*this)("&",i));

    return *this;
}

template <class T>
Vector<T> &Vector<T>::softzero(void)
{
    NiceAssert( !infsize() );

    return zero();
}

template <class T>
Vector<T> &Vector<T>::offset(int amoff)
{
    NiceAssert( !infsize() );
    NiceAssert( amoff >= -size() );

    if ( amoff )
    {
        int i;

        if ( !ccontent )
        {
            fix();
        }

        NiceAssert( !nbase );
        NiceAssert( content );
//        NiceAssert( !fixsize );

        if ( amoff < 0 )
        {
            for ( i = 0 ; i < dsize+amoff ; ++i )
            {
                qswap((*this)("&",i-amoff),(*this)("&",i));
            }
        }

        dsize = dsize+amoff;

        (*content).resize(dsize);
        pivot = cntintarray(dsize);

        if ( dsize && ( amoff > 0 ) )
        {
            for ( i = dsize-1 ; i >= amoff ; --i )
            {
                qswap((*this)("&",i-amoff),(*this)("&",i));
            }

            for ( i = amoff-1 ; i >= 0 ; --i )
            {
                zero(i);
            }
        }
    }

    return *this;
}

template <> inline Vector<double> &Vector<double>::posate(void);
template <> inline Vector<double> &Vector<double>::posate(void)
{
    return *this;
}

template <> inline Vector<int> &Vector<int>::posate(void);
template <> inline Vector<int> &Vector<int>::posate(void)
{
    return *this;
}

template <class T>
Vector<T> &Vector<T>::posate(void)
{
    if ( imoverhere )
    {
        overhere().posate();
    }

    else
    {
        for ( int i = 0 ; i < dsize ; ++i )
        {
            setposate((*this)("&",i));
	}
    }

    return *this;
}

template <> inline Vector<double> &Vector<double>::negate(void);
template <> inline Vector<double> &Vector<double>::negate(void)
{
    if ( imoverhere )
    {
        overhere().negate();
    }

    else
    {
	for ( int i = 0 ; i < dsize ; ++i )
	{
            (*this)("&",i) *= -1;
	}
    }

    return *this;
}

template <> inline Vector<int> &Vector<int>::negate(void);
template <> inline Vector<int> &Vector<int>::negate(void)
{
    if ( imoverhere )
    {
        overhere().negate();
    }

    else
    {
	for ( int i = 0 ; i < dsize ; ++i )
	{
            (*this)("&",i) *= -1;
	}
    }

    return *this;
}

template <class T>
Vector<T> &Vector<T>::negate(void)
{
    if ( imoverhere )
    {
        overhere().negate();
    }

    else
    {
	for ( int i = 0 ; i < dsize ; ++i )
	{
            setnegate((*this)("&",i));
	}
    }

    return *this;
}

template <> inline Vector<double> &Vector<double>::conj(void);
template <> inline Vector<double> &Vector<double>::conj(void)
{
    return *this;
}

template <> inline Vector<int> &Vector<int>::conj(void);
template <> inline Vector<int> &Vector<int>::conj(void)
{
    return *this;
}

template <class T>
Vector<T> &Vector<T>::conj(void)
{
    if ( imoverhere )
    {
        overhere().conj();
    }

    else
    {
        for ( int i = 0 ; i < dsize ; ++i )
        {
            setconj((*this)("&",i));
	}
    }

    return *this;
}

template <class T>
Vector<T> &Vector<T>::rand(void)
{
    if ( imoverhere )
    {
        overhere().rand();
    }

    else
    {
	for ( int i = 0 ; i < dsize ; ++i )
	{
            setrand((*this)("&",i));
	}
    }

    return *this;
}



// Access:


template <> inline void Vector<double>::set(const Vector<int> &i, const double &src);
template <> inline void Vector<double>::set(const Vector<int> &i, const double &src)
{
    NiceAssert( !infsize() );
    NiceAssert( checkRange(0,dsize-1,i) );

    int isize = i.size();

    if ( !ccontent )
    {
        fix();
    }

    NiceAssert( content );
    NiceAssert( pivot );

    {
        int k = 0;
        double lsrc(src);

        for ( int j = 0 ; k < isize ; j++, k++ )
        {
            int m = i(j);

            (*content).sv((*pivot).v(iib+(m*iis)),lsrc);
        }
    }
}

template <> inline void Vector<int>::set(const Vector<int> &i, const int &src);
template <> inline void Vector<int>::set(const Vector<int> &i, const int &src)
{
    NiceAssert( !infsize() );
    NiceAssert( checkRange(0,dsize-1,i) );

    int isize = i.size();

    if ( !ccontent )
    {
        fix();
    }

    NiceAssert( content );
    NiceAssert( pivot );

    {
        int k = 0;
        int lsrc(src);

        for ( int j = 0 ; k < isize ; j++, k++ )
        {
            int m = i(j);

            (*content).sv((*pivot).v(iib+(m*iis)),lsrc);
        }
    }
}

template <class T>
void Vector<T>::set(const Vector<int> &i,   const T &src)
{
    NiceAssert( !infsize() );
    NiceAssert( checkRange(0,dsize-1,i) );

    int isize = i.size();

    if ( !ccontent )
    {
        fix();
    }

    NiceAssert( content );
    NiceAssert( pivot );

    {
        int k = 0;

        for ( int j = 0 ; k < isize ; j++, k++ )
        {
            int m = i(j);

            (*content).set((*pivot).v(iib+(m*iis)),src);
        }
    }
}

template <> inline void Vector<double>::set(int ib, int is, int im, const double &src);
template <> inline void Vector<double>::set(int ib, int is, int im, const double &src)
{
    NiceAssert( !infsize() );
    NiceAssert( ( ( is > 0 ) && ( ib >= 0 ) ) || ( ( is < 0 ) && ( ib < dsize ) ) );
    NiceAssert( ( ( is < 0 ) && ( im >= 0 ) ) || ( ( is > 0 ) && ( im < dsize ) ) );
    NiceAssert( is );

    int isize = ((im-ib)/is)+1;

    isize = ( isize < 0 ) ? 0 : isize;

    if ( !ccontent )
    {
        fix();
    }

    NiceAssert( content );
    NiceAssert( pivot );

    {
        int k = 0;
        double lsrc(src);

        for ( int m = ib ; k < isize ; m += is, k++ )
        {
            (*content).sv((*pivot).v(iib+(m*iis)),lsrc);
        }
    }
}

template <> inline void Vector<int>::set(int ib, int is, int im, const int &src);
template <> inline void Vector<int>::set(int ib, int is, int im, const int &src)
{
    NiceAssert( !infsize() );
    NiceAssert( ( ( is > 0 ) && ( ib >= 0 ) ) || ( ( is < 0 ) && ( ib < dsize ) ) );
    NiceAssert( ( ( is < 0 ) && ( im >= 0 ) ) || ( ( is > 0 ) && ( im < dsize ) ) );
    NiceAssert( is );

    int isize = ((im-ib)/is)+1;

    isize = ( isize < 0 ) ? 0 : isize;

    if ( !ccontent )
    {
        fix();
    }

    NiceAssert( content );
    NiceAssert( pivot );

    {
        int k = 0;
        int lsrc(src);

        for ( int m = ib ; k < isize ; m += is, k++ )
        {
            (*content).sv((*pivot).v(iib+(m*iis)),lsrc);
        }
    }
}

template <class T>
void Vector<T>::set(int ib, int is, int im, const T &src)
{
    NiceAssert( !infsize() );
    NiceAssert( ( ( is > 0 ) && ( ib >= 0 ) ) || ( ( is < 0 ) && ( ib < dsize ) ) );
    NiceAssert( ( ( is < 0 ) && ( im >= 0 ) ) || ( ( is > 0 ) && ( im < dsize ) ) );
    NiceAssert( is );

    int isize = ((im-ib)/is)+1;

    isize = ( isize < 0 ) ? 0 : isize;

    if ( !ccontent )
    {
        fix();
    }

    NiceAssert( content );
    NiceAssert( pivot );

    {
        int k = 0;

        for ( int m = ib ; k < isize ; m += is, k++ )
        {
            (*content).sv((*pivot).v(iib+(m*iis)),src);
        }
    }
}

template <> inline void Vector<double>::set(const Vector<int> &i, const Vector<double> &src);
template <> inline void Vector<double>::set(const Vector<int> &i, const Vector<double> &src)
{
    NiceAssert( !infsize() );
    NiceAssert( checkRange(0,dsize-1,i) );

    int isize = i.size();

    NiceAssert( ( src.size() == isize ) || !src.size() );

    if ( !ccontent )
    {
        fix();
    }

    NiceAssert( content );
    NiceAssert( pivot );

    if ( src.size() )
    {
        int k = 0;

        for ( int j = 0 ; k < isize ; j++, k++ )
        {
            int m = i(j);

            (*content).sv((*pivot).v(iib+(m*iis)),src.v(k));
        }
    }
}

template <> inline void Vector<int>::set(const Vector<int> &i, const Vector<int> &src);
template <> inline void Vector<int>::set(const Vector<int> &i, const Vector<int> &src)
{
    NiceAssert( !infsize() );
    NiceAssert( checkRange(0,dsize-1,i) );

    int isize = i.size();

    NiceAssert( ( src.size() == isize ) || !src.size() );

    if ( !ccontent )
    {
        fix();
    }

    NiceAssert( content );
    NiceAssert( pivot );

    if ( src.size() )
    {
        int k = 0;

        for ( int j = 0 ; k < isize ; j++, k++ )
        {
            int m = i(j);

            (*content).sv((*pivot).v(iib+(m*iis)),src.v(k));
        }
    }
}

template <class T>
void Vector<T>::set(const Vector<int> &i, const Vector<T> &src)
{
    NiceAssert( !infsize() );
    NiceAssert( checkRange(0,dsize-1,i) );

    int isize = i.size();

    NiceAssert( ( src.size() == isize ) || !src.size() );

    if ( !ccontent )
    {
        fix();
    }

    NiceAssert( content );
    NiceAssert( pivot );

    if ( src.size() )
    {
        int k = 0;

        for ( int j = 0 ; k < isize ; j++, k++ )
        {
            int m = i(j);

            (*content).set((*pivot).v(iib+(m*iis)),src(k));
        }
    }
}

template <> inline void Vector<double>::set(int ib, int is, int im, const Vector<double> &src);
template <> inline void Vector<double>::set(int ib, int is, int im, const Vector<double> &src)
{
    NiceAssert( !infsize() );
    NiceAssert( ( ( is > 0 ) && ( ib >= 0 ) ) || ( ( is < 0 ) && ( ib < dsize ) ) );
    NiceAssert( ( ( is < 0 ) && ( im >= 0 ) ) || ( ( is > 0 ) && ( im < dsize ) ) );
    NiceAssert( is );

    int isize = ((im-ib)/is)+1;

    isize = ( isize < 0 ) ? 0 : isize;

    NiceAssert( ( src.size() == isize ) || !src.size() );

    if ( !ccontent )
    {
        fix();
    }

    NiceAssert( content );
    NiceAssert( pivot );

    if ( src.size() )
    {
        int k = 0;

        for ( int m = ib ; k < isize ; m += is, k++ )
        {
            (*content).sv((*pivot).v(iib+(m*iis)),src.v(k));
        }
    }
}

template <> inline void Vector<int>::set(int ib, int is, int im, const Vector<int> &src);
template <> inline void Vector<int>::set(int ib, int is, int im, const Vector<int> &src)
{
    NiceAssert( !infsize() );
    NiceAssert( ( ( is > 0 ) && ( ib >= 0 ) ) || ( ( is < 0 ) && ( ib < dsize ) ) );
    NiceAssert( ( ( is < 0 ) && ( im >= 0 ) ) || ( ( is > 0 ) && ( im < dsize ) ) );
    NiceAssert( is );

    int isize = ((im-ib)/is)+1;

    isize = ( isize < 0 ) ? 0 : isize;

    NiceAssert( ( src.size() == isize ) || !src.size() );

    if ( !ccontent )
    {
        fix();
    }

    NiceAssert( content );
    NiceAssert( pivot );

    if ( src.size() )
    {
        int k = 0;

        for ( int m = ib ; k < isize ; m += is, k++ )
        {
            (*content).sv((*pivot).v(iib+(m*iis)),src.v(k));
        }
    }
}

template <class T>
void Vector<T>::set(int ib, int is, int im, const Vector<T> &src)
{
    NiceAssert( !infsize() );
    NiceAssert( ( ( is > 0 ) && ( ib >= 0 ) ) || ( ( is < 0 ) && ( ib < dsize ) ) );
    NiceAssert( ( ( is < 0 ) && ( im >= 0 ) ) || ( ( is > 0 ) && ( im < dsize ) ) );
    NiceAssert( is );

    int isize = ((im-ib)/is)+1;

    isize = ( isize < 0 ) ? 0 : isize;

    NiceAssert( ( src.size() == isize ) || !src.size() );

    if ( !ccontent )
    {
        fix();
    }

    NiceAssert( content );
    NiceAssert( pivot );

    if ( src.size() )
    {
        int k = 0;

        for ( int m = ib ; k < isize ; m += is, k++ )
        {
            (*content).set((*pivot).v(iib+(m*iis)),src(k));
        }
    }
}



template <class T>
Vector<T> &Vector<T>::operator()(const char *, const Vector<int> &i, retVector<T> &res)
{
    NiceAssert( !infsize() );
    NiceAssert( checkRange(0,dsize-1,i) );

    int isize = i.size(); //((im-ib)/is)+1;

    //isize = ( isize < 0 ) ? 0 : isize;

    if ( ( !nbase || ( ( iib == 0 ) && ( iis == 1 ) && ( pivot == cntintarray(0) ) ) ) && !(i.base()) )
    {
        res.reset(*this);

        if ( !ccontent )
        {
            fix();
        }

	res.dsize = isize;
	res.iib   = 0; //iib;
	res.iis   = 1; //iis;
        res.pivot = res.dsize ? i.ccontent : cntintarray(res.iib);

    }

    else if ( ( iib == 0 ) && ( iis == 0 ) && ( pivot == zerointarray() ) )
    {
        res.reset(*this);

        if ( !ccontent )
        {
            fix();
        }

	res.dsize = isize;
	res.iib   = 0;
	res.iis   = 0;
        res.pivot = zerointarray();
    }

    else
    {
        DynArray<int> *tpiv = res.reset_p(*this,i.size());

        if ( !ccontent )
        {
            fix();
        }

	res.dsize = isize;
	res.iib   = 0;
	res.iis   = 1;

        for ( int ii = 0 ; ii < res.dsize ; ++ii )
        {
            (*tpiv).sv(ii,(*pivot).v(iib+(iis*(i.v(ii)))));
	}
    }

    return res;
}

template <class T>
Vector<T> &Vector<T>::operator()(const char *, const DynArray<int> &i, int isize, retVector<T> &res)
{
    NiceAssert( !infsize() );
    NiceAssert( isize >= 0 );

    //int isize = i.size(); //((im-ib)/is)+1;

    //isize = ( isize < 0 ) ? 0 : isize;

#ifndef NDEBUG
    for ( int ij = 0 ; ij < isize ; ++ij )
    {
        NiceAssert( i.v(ij) >= 0 );
        NiceAssert( i.v(ij) < dsize );
    }
#endif

    if ( !nbase || ( ( iib == 0 ) && ( iis == 1 ) && ( pivot == cntintarray(0) ) ) )
    {
        res.reset(*this);

        if ( !ccontent )
        {
            fix();
        }

	res.dsize = isize;
	res.iib   = 0; //iib;
	res.iis   = 1; //iis;
        res.pivot = res.dsize ? &i : cntintarray(res.iib);
    }

    else if ( ( iib == 0 ) && ( iis == 0 ) && ( pivot == zerointarray() ) )
    {
        res.reset(*this);

        if ( !ccontent )
        {
            fix();
        }

	res.dsize = isize;
	res.iib   = 0;
	res.iis   = 0;
        res.pivot = zerointarray();
    }

    else
    {
        DynArray<int> *tpiv = res.reset_p(*this,isize);

        if ( !ccontent )
        {
            fix();
        }

	res.dsize = isize;
	res.iib   = 0;
	res.iis   = 1;

        for ( int ii = 0 ; ii < res.dsize ; ++ii )
        {
            (*tpiv).sv(ii,(*pivot).v(iib+(iis*(i.v(ii)))));
	}
    }

    return res;
}

template <class T>
Vector<T> &Vector<T>::operator()(const char *, const Vector<int> &i, int ib, int is, int im, retVector<T> &res)
{
    NiceAssert( !infsize() );
    NiceAssert( is );

    int isize = ((im-ib)/is)+1;

    isize = ( isize < 0 ) ? 0 : isize;

#ifndef NDEBUG
    for ( int ij = 0 ; ij < isize ; ++ij )
    {
        NiceAssert( i.v(ij) >= 0 );
        NiceAssert( i.v(ij) < dsize );
    }
#endif

    if ( ( !nbase || ( ( iib == 0 ) && ( iis == 1 ) && ( pivot == cntintarray(0) ) ) ) && !(i.base()) )
    {
        res.reset(*this);

        if ( !ccontent )
        {
            fix();
        }

	res.dsize = isize;
        res.iib   = ib; //iib+(iis*ib);
        res.iis   = is; //iis*is;
        res.pivot = res.dsize ? i.ccontent : cntintarray(res.iib);
    }

    else if ( ( iib == 0 ) && ( iis == 0 ) && ( pivot == zerointarray() ) )
    {
        res.reset(*this);

        if ( !ccontent )
        {
            fix();
        }

	res.dsize = isize;
        res.iib   = 0;
        res.iis   = 0;
        res.pivot = zerointarray();
    }

    else
    {
        DynArray<int> *tpiv = res.reset_p(*this,isize);

        if ( !ccontent )
        {
            fix();
        }

	res.dsize = isize;
	res.iib   = 0;
	res.iis   = 1;

        for ( int ii = 0 ; ii < res.dsize ; ++ii )
        {
            (*tpiv).sv(ii,(*pivot).v(iib+(iis*(i.v(ib+(is*ii))))));
	}
    }

    return res;
}

template <class T>
Vector<T> &Vector<T>::operator()(const char *, const DynArray<int> &i, int ib, int is, int im, retVector<T> &res)
{
    NiceAssert( !infsize() );
    NiceAssert( is );

    int isize = ((im-ib)/is)+1;

    isize = ( isize < 0 ) ? 0 : isize;

#ifndef NDEBUG
    for ( int ij = 0 ; ij < isize ; ++ij )
    {
        NiceAssert( i.v(ij) >= 0 );
        NiceAssert( i.v(ij) < dsize );
    }
#endif

    if ( !nbase || ( ( iib == 0 ) && ( iis == 1 ) && ( pivot == cntintarray(0) ) ) )
    {
        res.reset(*this);

        if ( !ccontent )
        {
            fix();
        }

	res.dsize = isize;
        res.iib   = ib; //iib+(iis*ib);
        res.iis   = is; //iis*is;
        res.pivot = res.dsize ? &i : cntintarray(res.iib);
    }

    else if ( ( iib == 0 ) && ( iis == 0 ) && ( pivot == zerointarray() ) )
    {
        res.reset(*this);

        if ( !ccontent )
        {
            fix();
        }

	res.dsize = isize;
        res.iib   = 0;
        res.iis   = 0;
        res.pivot = zerointarray();
    }

    else
    {
        DynArray<int> *tpiv = res.reset_p(*this,isize);

        if ( !ccontent )
        {
            fix();
        }

	res.dsize = isize;
	res.iib   = 0;
	res.iis   = 1;

        for ( int ii = 0 ; ii < res.dsize ; ++ii )
        {
            (*tpiv).sv(ii,(*pivot).v(iib+(iis*(i.v(ib+(is*ii))))));
	}
    }

    return res;
}

template <class T>
Vector<T> &Vector<T>::operator()(const char *, int ib, int is, int im, retVector<T> &res)
{
    NiceAssert( !infsize() );
    NiceAssert( ( ( is > 0 ) && ( ib >= 0 ) ) || ( ( is < 0 ) && ( ib < dsize ) ) );
    NiceAssert( ( ( is < 0 ) && ( im >= 0 ) ) || ( ( is > 0 ) && ( im < dsize ) ) );
    NiceAssert( is );

    int isize = ((im-ib)/is)+1;

    isize = ( isize < 0 ) ? 0 : isize;

    res.reset(*this);

    if ( !ccontent )
    {
        fix();
    }

    res.dsize = isize;
    res.iib   = iib+(iis*ib);
    res.iis   = iis*is;
    res.pivot = pivot;

    return res;
}

template <class T>
const Vector<T> &Vector<T>::operator()(const Vector<int> &i, retVector<T> &res) const
{
    NiceAssert( !infsize() );
    NiceAssert( checkRange(0,dsize-1,i) );

    if ( !ccontent )
    {
        // We need ccontent for this to make sense, but we don't have it, and
        // we can't allocate it (without mutable) in the const context, so we
        // have an alternative vector (which we would be constructing) and use that.

        static thread_local Vector<T> altres;

        return altres("&",i,res); // note *non*-const return so that fix can be called.
    }

    int isize = i.size(); //((im-ib)/is)+1;

    //isize = ( isize < 0 ) ? 0 : isize;

    if ( ( !nbase || ( ( iib == 0 ) && ( iis == 1 ) && ( pivot == cntintarray(0) ) ) ) && !(i.base()) )
    {
        res.creset(*this);

	res.dsize = isize;
	res.iib   = 0; //iib;
	res.iis   = 1; //iis;
        res.pivot = res.dsize ? i.ccontent : cntintarray(res.iib);
    }

    else if ( ( iib == 0 ) && ( iis == 0 ) && ( pivot == zerointarray() ) )
    {
        res.creset(*this);

	res.dsize = isize;
	res.iib   = 0;
	res.iis   = 0;
        res.pivot = zerointarray();
    }

    else
    {
        DynArray<int> *tpiv = res.creset_p(*this,i.size());

	res.dsize = isize;
	res.iib   = 0;
	res.iis   = 1;

        for ( int ii = 0 ; ii < res.dsize ; ++ii )
        {
            (*tpiv).sv(ii,(*pivot).v(iib+(iis*(i.v(ii)))));
	}
    }

    return res;
}

template <class T>
const Vector<T> &Vector<T>::operator()(const Vector<int> &i, int ib, int is, int im, retVector<T> &res) const
{
    NiceAssert( !infsize() );
    NiceAssert( is );

    if ( !ccontent )
    {
        // We need ccontent for this to make sense, but we don't have it, and
        // we can't allocate it (without mutable) in the const context, so we
        // have an alternative vector (which we would be constructing) and use that.

        static thread_local Vector<T> altres;

        return altres("&",i,ib,is,im,res); // note *non*-const return so that fix can be called.
    }

    int isize = ((im-ib)/is)+1;

    isize = ( isize < 0 ) ? 0 : isize;

#ifndef NDEBUG
    for ( int ij = 0 ; ij < isize ; ++ij )
    {
        NiceAssert( i.v(ij) >= 0 );
        NiceAssert( i.v(ij) < dsize );
    }
#endif

    if ( ( !nbase || ( ( iib == 0 ) && ( iis == 1 ) && ( pivot == cntintarray(0) ) ) ) && !(i.base()) )
    {
        res.creset(*this);

	res.dsize = isize;
        res.iib   = ib; //iib+(iis*ib);
        res.iis   = is; //iis*is;
        res.pivot = res.dsize ? i.ccontent : cntintarray(res.iib);
    }

    else if ( ( iib == 0 ) && ( iis == 0 ) && ( pivot == zerointarray() ) )
    {
        res.creset(*this);

	res.dsize = isize;
        res.iib   = 0;
        res.iis   = 0;
        res.pivot = zerointarray();
    }

    else
    {
        DynArray<int> *tpiv = res.creset_p(*this,isize);

	res.dsize = isize;
	res.iib   = 0;
	res.iis   = 1;

        for ( int ii = 0 ; ii < res.dsize ; ++ii )
        {
            (*tpiv).sv(ii,(*pivot).v(iib+(iis*(i.v(ib+(is*ii))))));
	}
    }

    return res;
}

template <class T>
const Vector<T> &Vector<T>::operator()(const DynArray<int> &i, int isize, retVector<T> &res) const
{
    NiceAssert( !infsize() );
    NiceAssert( isize >= 0 );

    if ( !ccontent )
    {
        // We need ccontent for this to make sense, but we don't have it, and
        // we can't allocate it (without mutable) in the const context, so we
        // have an alternative vector (which we would be constructing) and use that.

        static thread_local Vector<T> altres;

        return altres("&",i,isize,res); // note *non*-const return so that fix can be called.
    }

#ifndef NDEBUG
    for ( int ij = 0 ; ij < isize ; ++ij )
    {
        NiceAssert( i.v(ij) >= 0 );
        NiceAssert( i.v(ij) < dsize );
    }
#endif

    if ( !nbase || ( ( iib == 0 ) && ( iis == 1 ) && ( pivot == cntintarray(0) ) ) )
    {
        res.creset(*this);

	res.dsize = isize;
	res.iib   = 0; //iib;
	res.iis   = 1; //iis;
        res.pivot = res.dsize ? &i : cntintarray(res.iib);
    }

    else if ( ( iib == 0 ) && ( iis == 0 ) && ( pivot == zerointarray() ) )
    {
        res.creset(*this);

	res.dsize = isize;
	res.iib   = 0;
	res.iis   = 0;
        res.pivot = zerointarray();
    }

    else
    {
        DynArray<int> *tpiv = res.creset_p(*this,isize);

	res.dsize = isize;
	res.iib   = 0;
	res.iis   = 1;

        for ( int ii = 0 ; ii < res.dsize ; ++ii )
        {
            (*tpiv).sv(ii,(*pivot).v(iib+(iis*(i.v(ii)))));
	}
    }

    return res;
}

template <class T>
const Vector<T> &Vector<T>::operator()(const DynArray<int> &i, int ib, int is, int im, retVector<T> &res) const
{
    NiceAssert( !infsize() );

    if ( !ccontent )
    {
        // We need ccontent for this to make sense, but we don't have it, and
        // we can't allocate it (without mutable) in the const context, so we
        // have an alternative vector (which we would be constructing) and use that.

        static thread_local Vector<T> altres;

        return altres("&",i,ib,is,im,res); // note *non*-const return so that fix can be called.
    }

    int isize = ((im-ib)/is)+1;

    isize = ( isize < 0 ) ? 0 : isize;

#ifndef NDEBUG
    for ( int ij = 0 ; ij < isize ; ++ij )
    {
        NiceAssert( i.v(ij) >= 0 );
        NiceAssert( i.v(ij) < dsize );
    }
#endif

    if ( !nbase || ( ( iib == 0 ) && ( iis == 1 ) && ( pivot == cntintarray(0) ) ) )
    {
        res.creset(*this);

	res.dsize = isize;
        res.iib   = ib; //iib+(iis*ib);
        res.iis   = is; //iis*is;
        res.pivot = res.dsize ? &i : cntintarray(res.iib);
    }

    else if ( ( iib == 0 ) && ( iis == 0 ) && ( pivot == zerointarray() ) )
    {
        res.creset(*this);

	res.dsize = isize;
        res.iib   = 0;
        res.iis   = 0;
        res.pivot = zerointarray();
    }

    else
    {
        DynArray<int> *tpiv = res.creset_p(*this,isize);

	res.dsize = isize;
	res.iib   = 0;
	res.iis   = 1;

        for ( int ii = 0 ; ii < res.dsize ; ++ii )
        {
            (*tpiv).sv(ii,(*pivot).v(iib+(iis*(i.v(ib+(is*ii))))));
	}
    }

    return res;
}

template <class T>
const Vector<T> &Vector<T>::operator()(int ib, int is, int im, retVector<T> &res) const
{
    NiceAssert( !infsize() );
    NiceAssert( ( ( is > 0 ) && ( ib >= 0 ) ) || ( ( is < 0 ) && ( ib < dsize ) ) );
    NiceAssert( ( ( is < 0 ) && ( im >= 0 ) ) || ( ( is > 0 ) && ( im < dsize ) ) );
    NiceAssert( is );

    if ( !ccontent )
    {
        // We need ccontent for this to make sense, but we don't have it, and
        // we can't allocate it (without mutable) in the const context, so we
        // have an alternative vector (which we would be constructing) and use that.

        static thread_local Vector<T> altres;

        return altres("&",ib,is,im,res); // note *non*-const return so that fix can be called.
    }

    int isize = ((im-ib)/is)+1;

    isize = ( isize < 0 ) ? 0 : isize;

    res.creset(*this);

    res.dsize = isize;
    res.iib   = iib+(iis*ib);
    res.iis   = iis*is;
    res.pivot = pivot;

    return res;
}























// Scaled addition:

template <> template <> inline Vector<double> &Vector<double>::scaleAdd(const double &a, const Vector<double> &b);
template <> template <> inline Vector<double> &Vector<double>::scaleAdd(const double &a, const Vector<double> &b)
{
    NiceAssert( !size() || !b.size() || ( b.size() == size() ) );

    if ( size() && b.size() && contiguous() && b.contiguous() ) // NB fastAddTo will be fine here as no indexing (wrap-over), so even if this == b we're good
    {
        fastAddTo(&(*this)("&",0),&b(0),a,size());
    }

    else if ( !shareBase(b) && size() && b.size() ) // May not be able to fastAddTo here depending on indexing
    {
        fastAddTo(unsafecontent(),unsafepivot(),unsafeib(),unsafeis(),b.unsafeccontent(),b.unsafepivot(),b.unsafeib(),b.unsafeis(),a,size());
    }

    else if ( size() && shareBase(b) )
    {
        Vector<double> temp(b);
        scaleAdd(a,temp);
    }

    else if ( !size() && b.size() )
    {
        *this  = b;
        *this *= a;
    }

    return *this;
}


template <> template <> inline Vector<double> &Vector<double>::scaleAddR(const Vector<double> &a, const double &b);
template <> template <> inline Vector<double> &Vector<double>::scaleAddR(const Vector<double> &a, const double &b)
{
    return scaleAdd(b,a);
}

template <> template <> inline Vector<double> &Vector<double>::scaleAddB(const double &a, const Vector<double> &b);
template <> template <> inline Vector<double> &Vector<double>::scaleAddB(const double &a, const Vector<double> &b)
{
    return scaleAdd(a,b);
}

template <> template <> inline Vector<double> &Vector<double>::scaleAddBR(const Vector<double> &a, const double &b);
template <> template <> inline Vector<double> &Vector<double>::scaleAddBR(const Vector<double> &a, const double &b)
{
    return scaleAdd(b,a);
}

template <> template <> inline Vector<double> &Vector<double>::sqscaleAdd(const double &a, const Vector<double> &b);
template <> template <> inline Vector<double> &Vector<double>::sqscaleAdd(const double &a, const Vector<double> &b)
{
    NiceAssert( !size() || !b.size() || ( b.size() == size() ) );

    if ( size() && b.size() && contiguous() && b.contiguous() )
    {
        double *rr = &((*this)("&",0));
        const double *bb = &(b(0));

        for ( int i = 0 ; i < size() ; ++i )
        {
            rr[i] = a*bb[i]*bb[i];
        }
    }

    else if ( !shareBase(b) && size() && b.size() )
    {
        for ( int i = 0 ; i < size() ; ++i )
        {
            sv(i,a*b.v(i)*b.v(i));
        }
    }

    else if ( size() && shareBase(b) )
    {
        Vector<double> temp(b);

        sqscaleAdd(a,temp);
    }

    else if ( !size() && b.size() )
    {
        resize(b.size());
        zero();
        sqscaleAdd(a,b);
    }

    return *this;
}

template <> template <> inline Vector<double> &Vector<double>::sqscaleAddR(const Vector<double> &a, const double &b);
template <> template <> inline Vector<double> &Vector<double>::sqscaleAddR(const Vector<double> &a, const double &b)
{
    return sqscaleAdd(b,a);
}

template <> template <> inline Vector<double> &Vector<double>::sqscaleAddB(const double &a, const Vector<double> &b);
template <> template <> inline Vector<double> &Vector<double>::sqscaleAddB(const double &a, const Vector<double> &b)
{
    return sqscaleAdd(a,b);
}

template <> template <> inline Vector<double> &Vector<double>::sqscaleAddBR(const Vector<double> &a, const double &b);
template <> template <> inline Vector<double> &Vector<double>::sqscaleAddBR(const Vector<double> &a, const double &b)
{
    return sqscaleAdd(b,a);
}

template <class T>
Vector<T> &Vector<T>::vscaleAdd(const Vector<T> &a, const Vector<T> &b)
{
    NiceAssert( !infsize() );

    if ( a.size() && shareBase(a) && b.size() && shareBase(b) )
    {
        Vector<T> tempa(a);
        Vector<T> tempb(b);

        vscaleAdd(tempa,tempb);
    }

    else if ( a.size() && shareBase(a) )
    {
        Vector<T> tempa(a);

        vscaleAdd(tempa,b);
    }

    else if ( b.size() && shareBase(b) )
    {
        Vector<T> tempb(b);

        vscaleAdd(a,tempb);
    }

    else if ( !(a.size()) && !(b.size()) )
    {
        ;
    }

    else if ( !(a.size()) && b.size() )
    {
        *this += b;
    }

    else if ( a.size() && !(b.size()) )
    {
        *this += a;
    }

    else if ( !size() )
    {
        NiceAssert( a.size() == b.size() );

        resize(a.size());

        vscaleAdd(a,b);
    }

    else
    {
        NiceAssert( size() == a.size() );
        NiceAssert( size() == b.size() );

        for ( int i = 0 ; i < size() ; ++i )
        {
            (*this)("&",i) += ((a(i))*(b(i)));
        }
    }

    return *this;
}

template <class T>
template <class S> Vector<T> &Vector<T>::scaleAdd(const S &a, const Vector<T> &b)
{
    NiceAssert( !infsize() );

    if ( b.size() && shareBase(b) )
    {
        Vector<T> temp(b);

        scaleAdd(a,temp);
    }

    else
    {
        NiceAssert( ( size() == b.size() ) || !size() || !(b.size()) );

	if ( !size() && b.size() )
	{
            resize(b.size());
            zero();
            scaleAdd(a,b);
	}

	else if ( b.size() )
	{
	    for ( int i = 0 ; i < size() ; ++i )
	    {
		(*this)("&",i) += (a*(b(i)));
	    }
	}
    }

    return *this;
}

template <class T>
template <class S> Vector<T> &Vector<T>::scaleAddR(const Vector<T> &a, const S &b)
{
    NiceAssert( !infsize() );

    if ( b.size() && shareBase(a) )
    {
        Vector<T> temp(a);

        scaleAddR(temp,b);
    }

    else
    {
        NiceAssert( ( size() == a.size() ) || !size() || !(a.size()) );

	if ( !size() && a.size() )
	{
            resize(a.size());
            zero();
            scaleAddR(a,b);
	}

	else if ( a.size() )
	{
	    for ( int i = 0 ; i < size() ; ++i )
	    {
		(*this)("&",i) += ((a(i))*b);
	    }
	}
    }

    return *this;
}

template <class T>
template <class S> Vector<T> &Vector<T>::scaleAddB(const T &a, const Vector<S> &b)
{
    NiceAssert( !infsize() );
    NiceAssert( ( size() == b.size() ) || !size() || !(b.size()) );

    if ( b.size() && !size() )
    {
        resize(b.size());
        zero();
        scaleAddB(a,b);
    }

    else if ( b.size() )
    {
	for ( int i = 0 ; i < size() ; ++i )
	{
	    (*this)("&",i) += (a*(b(i)));
	}
    }

    return *this;
}

template <class T>
template <class S> Vector<T> &Vector<T>::scaleAddBR(const Vector<S> &a, const T &b)
{
    NiceAssert( !infsize() );
    NiceAssert( ( size() == a.size() ) || !size() || !(a.size()) );

    if ( a.size() && !size() )
    {
        resize(a.size());
        zero();
        scaleAddBR(a,b);
    }

    else if ( a.size() )
    {
	for ( int i = 0 ; i < size() ; ++i )
	{
	    (*this)("&",i) += ((a(i))*b);
	}
    }

    return *this;
}

template <class T>
template <class S> Vector<T> &Vector<T>::sqscaleAdd(const S &a, const Vector<T> &b)
{
    NiceAssert( !infsize() );

    if ( b.size() && shareBase(b) )
    {
        Vector<T> temp(b);

        sqscaleAdd(a,temp);
    }

    else
    {
        NiceAssert( ( size() == b.size() ) || !size() || !(b.size()) );

	if ( !size() && b.size() )
	{
            resize(b.size());
            zero();
            sqscaleAdd(a,b);
	}

	else if ( b.size() )
	{
	    for ( int i = 0 ; i < size() ; ++i )
	    {
		(*this)("&",i) += (a*(b(i))*(b(i)));
	    }
	}
    }

    return *this;
}

template <class T>
template <class S> Vector<T> &Vector<T>::sqscaleAddR(const Vector<T> &a, const S &b)
{
    NiceAssert( !infsize() );

    if ( b.size() && shareBase(a) )
    {
        Vector<T> temp(a);

        sqscaleAddR(temp,b);
    }

    else
    {
        NiceAssert( ( size() == a.size() ) || !size() || !(a.size()) );

	if ( !size() && a.size() )
	{
            resize(a.size());
            zero();
            sqscaleAddR(a,b);
	}

	else if ( a.size() )
	{
	    for ( int i = 0 ; i < size() ; ++i )
	    {
		(*this)("&",i) += ((a(i))*(a(i))*b);
	    }
	}
    }

    return *this;
}

template <class T>
template <class S> Vector<T> &Vector<T>::sqscaleAddB(const T &a, const Vector<S> &b)
{
    NiceAssert( !infsize() );
    NiceAssert( ( size() == b.size() ) || !size() || !(b.size()) );

    if ( b.size() && !size() )
    {
        resize(b.size());
        zero();
        sqscaleAddB(a,b);
    }

    else if ( b.size() )
    {
	for ( int i = 0 ; i < size() ; ++i )
	{
	    (*this)("&",i) += (a*(b(i))*(b(i)));
	}
    }

    return *this;
}

template <class T>
template <class S> Vector<T> &Vector<T>::sqscaleAddBR(const Vector<S> &a, const T &b)
{
    NiceAssert( !infsize() );
    NiceAssert( ( size() == a.size() ) || !size() || !(a.size()) );

    if ( a.size() && !size() )
    {
        resize(a.size());
        zero();
        sqscaleAddBR(a,b);
    }

    else if ( a.size() )
    {
	for ( int i = 0 ; i < size() ; ++i )
	{
	    (*this)("&",i) += ((a(i))*(a(i))*b);
	}
    }

    return *this;
}

template <> template <> inline Vector<double> &Vector<double>::scale(const double &a);
template <> template <> inline Vector<double> &Vector<double>::scale(const double &a)
{
    return *this *= a;
}

template <> template <> inline Vector<double> &Vector<double>::lscale(const double &a);
template <> template <> inline Vector<double> &Vector<double>::lscale(const double &a)
{
    return *this *= a;
}

template <class T>
template <class S> Vector<T> &Vector<T>::scale(const S &a)
{
    NiceAssert( !infsize() );

    for ( int i = 0 ; i < size() ; ++i )
    {
        (*this)("&",i) *= a;
    }

    return *this;
}

template <class T>
template <class S> Vector<T> &Vector<T>::lscale(const S &a)
{
    NiceAssert( !infsize() );

    for ( int i = 0 ; i < size() ; ++i )
    {
        rightmult(a,(*this)("&",i));
    }

    return *this;
}

template <> template <> inline Vector<double> &Vector<double>::scale(const Vector<double> &a);
template <> template <> inline Vector<double> &Vector<double>::scale(const Vector<double> &a)
{
    return *this *= a;
}

template <> template <> inline Vector<double> &Vector<double>::lscale(const Vector<double> &a);
template <> template <> inline Vector<double> &Vector<double>::lscale(const Vector<double> &a)
{
    return *this *= a;
}

template <class T>
template <class S> Vector<T> &Vector<T>::scale(const Vector<S> &a)
{
    NiceAssert( !infsize() );
    NiceAssert( ( size() == a.size() ) || !(size()) || !(a.size()) );

    if ( !size() && a.size() )
    {
	resize(a.size());
        zero();
    }

    else if ( a.size() )
    {
	for ( int i = 0 ; i < size() ; ++i )
	{
	    (*this)("&",i) *= a(i);
	}
    }

    return *this;
}

template <class T>
template <class S> Vector<T> &Vector<T>::lscale(const Vector<S> &a)
{
    NiceAssert( !infsize() );
    NiceAssert( ( size() == a.size() ) || !(size()) || !(a.size()) );

    if ( !size() && a.size() )
    {
	resize(a.size());
        zero();
    }

    else if ( a.size() )
    {
	for ( int i = 0 ; i < size() ; ++i )
	{
            rightmult(a(i),(*this)("&",i));
	}
    }

    return *this;
}




// Various swap functions

template <class T>
Vector<T> &Vector<T>::blockswap(int i, int j)
{
    NiceAssert( !infsize() );
    NiceAssert( !nbase );
    NiceAssert( i >= 0 );
    NiceAssert( i < size() );
    NiceAssert( j >= 0 );
    NiceAssert( j < size() );

    // blockswap  ( i < j ): ( c ) (i)          ( c ) (i)
    //                       ( e ) (1)    ->    ( d ) (j-i)
    //                       ( d ) (j-i)        ( e ) (1)
    //                       ( f ) (...)        ( f ) (...)
    //
    // blockswap  ( i > j ): ( c ) (j)          ( c ) (j)
    //                       ( d ) (i-j)  ->    ( e ) (1)
    //                       ( e ) (1)          ( d ) (i-j)
    //                       ( f ) (...)        ( f ) (...)

    for ( int k = i ; k > j ; --k )
    {
        qswap((*this)("&",k),(*this)("&",k-1));
    }

    for ( int k = i ; k < j ; ++k )
    {
        qswap((*this)("&",k),(*this)("&",k+1));
    }

    return *this;
}

template <class T>
Vector<T> &Vector<T>::squareswap(int i, int j)
{
    NiceAssert( !infsize() );
    NiceAssert( i >= 0 );
    NiceAssert( i < size() );
    NiceAssert( j >= 0 );
    NiceAssert( j < size() );

    // squareswap ( i > j ): ( c ) (i)          ( c ) (i)
    //                       ( d ) (1)          ( f ) (1)
    //                       ( e ) (j-i)  ->    ( e ) (j-i)
    //                       ( f ) (1)          ( d ) (1)
    //                       ( g ) (...)        ( g ) (...)
    //
    // squareswap ( i < j ): ( c ) (j)          ( c ) (j)
    //                       ( d ) (1)          ( f ) (1)
    //                       ( e ) (i-j)  ->    ( e ) (i-j)
    //                       ( f ) (1)          ( d ) (1)
    //                       ( g ) (...)        ( g ) (...)

    if ( i != j )
    {
	qswap((*this)("&",i),(*this)("&",j));
    }

    return *this;
}

/*
template <class T>
Vector<T> &Vector<T>::blockswap(int i, int j)
{
    NiceAssert( !infsize() );
    NiceAssert( !nbase );
    NiceAssert( i >= 0 );
    NiceAssert( i < dsize );
    NiceAssert( j >= 0 );
    NiceAssert( j < dsize );

    if ( i > j )
    {
        NiceAssert( content );

	T temp;

        temp = (*this)(i);

	for ( int k = i ; k > j ; --k )
	{
            (*this)("&",k) = (*this)(k-1);
	}

        (*this)(j) = temp;
    }

    else if ( i < j )
    {
        NiceAssert( content );

	T temp;;

	temp = (*this)(i);

	for ( int k = i ; k < j ; ++k )
	{
            (*this)("&",k) = (*this)(k+1);
	}

        (*this)(j) = temp;
    }

    return *this;
}

template <class T>
Vector<T> &Vector<T>::squareswap(int i, int j)
{
    NiceAssert( !infsize() );
    NiceAssert( !nbase );
    NiceAssert( i >= 0 );
    NiceAssert( i < dsize );
    NiceAssert( j >= 0 );
    NiceAssert( j < dsize );

    if ( i != j )
    {
        NiceAssert( content );

	T temp;

	temp           = (*this)(i);
	(*this)("&",i) = (*this)(j);
        (*this)("&",j) = temp;
    }

    return *this;
}
*/



// Add and remove element functions

template <class T>
Vector<T> &Vector<T>::add(int i)
{
    if ( !ccontent )
    {
        fix();
    }

    NiceAssert( !infsize() );
    NiceAssert( !nbase );
//    NiceAssert( !fixsize );
    NiceAssert( i >= 0 );
    NiceAssert( i <= dsize );
    NiceAssert( content );

    ++dsize;

    NiceAssert( content );

    (*content).resize(dsize);
    pivot = cntintarray(dsize);

    blockswap(dsize-1,i);

    return *this;
}

template <class T>
Vector<T> &Vector<T>::qadd(int i, T &src)
{
    if ( !ccontent )
    {
        fix();
    }

    NiceAssert( !infsize() );
    NiceAssert( !nbase );
//    NiceAssert( !fixsize );
    NiceAssert( i >= 0 );
    NiceAssert( i <= dsize );
    NiceAssert( content );

    ++dsize;

    NiceAssert( content );

    (*content).resize(dsize);
    qswap((*this)("&",dsize-1),src);
    pivot = cntintarray(dsize);

    blockswap(dsize-1,i);

    return *this;
}

template <class T>
Vector<T> &Vector<T>::addpad(int i, int num)
{
    NiceAssert( !infsize() );
    NiceAssert( num >= 0 );

    if ( !num ) { fix(); }

    while ( num > 0 )
    {
        add(i);
        ++i;
        --num;
    }

    return *this;
}

template <class T>
Vector<T> &Vector<T>::remove(int i)
{
    if ( !ccontent )
    {
        fix();
    }

    NiceAssert( !infsize() );
    NiceAssert( !nbase );
//    NiceAssert( !fixsize );
    NiceAssert( i >= 0 );
    NiceAssert( i < dsize );
    NiceAssert( content );

    blockswap(i,dsize-1);

    --dsize;
    pivot = cntintarray(dsize);

    (*content).resize(dsize);

    return *this;
}

template <class T>
Vector<T> &Vector<T>::remove(const Vector<int> &i)
{
    NiceAssert( !infsize() );

    Vector<int> ii(i);
    int j;

    while ( ii.size() )
    {
        remove(max(ii,j));
        ii.remove(j);
    }

    return *this;
}

template <class T>
Vector<T> &Vector<T>::resize(int tsize)
{
    NiceAssert( !infsize() || !tsize );
    NiceAssert( tsize >= 0 );

    if ( !ccontent && !dsize && !nbase )
    {
        bkref    = this;
        MEMNEW(content,DynArray<T>);
        (*content) = { nullptr,0,0,defaulttightalloc,false,false,false };
        ccontent = content;
        pivot    = cntintarray(0);

        NiceAssert( content );
    }

    if ( dsize != tsize )
    {
        NiceAssert( !nbase );
//        NiceAssert( !fixsize );
        NiceAssert( content );

        dsize = tsize;

        (*content).resize(dsize);
        pivot = cntintarray(dsize);
    }

    return *this;
}

template <class T>
Vector<T> &Vector<T>::resize(int tsize, int suggestedallocsize)
{
    NiceAssert( !infsize() || !tsize );
    NiceAssert( tsize >= 0 );

    if ( !ccontent && !dsize && !nbase )
    {
        bkref    = this;
        MEMNEW(content,DynArray<T>);
        (*content) = { nullptr,0,0,defaulttightalloc,false,false,false };
        ccontent = content;
        pivot    = cntintarray(0);

        NiceAssert( content );
    }

    if ( dsize != tsize )
    {
        NiceAssert( !nbase );
//        NiceAssert( !fixsize );
        NiceAssert( content );

        dsize = tsize;

        (*content).resize(dsize,suggestedallocsize);
        pivot = cntintarray(dsize);
    }

    return *this;
}

template <class T>
const Vector<T> &Vector<T>::resize(int tsize) const
{
    NiceAssert( !infsize() || !tsize );
    NiceAssert( tsize >= 0 );

    if ( !ccontent && !dsize && !nbase )
    {
        bkref    = this;
        MEMNEW(content,DynArray<T>);
        (*content) = { nullptr,0,0,defaulttightalloc,false,false,false };
        ccontent = content;
        pivot    = cntintarray(0);

        NiceAssert( content );
    }

    if ( dsize != tsize )
    {
        NiceAssert( !nbase );
//        NiceAssert( !fixsize );
        NiceAssert( content );

        dsize = tsize;

        (*content).resize(dsize);
        pivot = cntintarray(dsize);
    }

    return *this;
}

template <class T>
Vector<T> &Vector<T>::pad(int n)
{
    NiceAssert( !infsize() );

    if ( !n ) { fix(); }

    if ( n > 0 )
    {
        resize(dsize+n);

        for ( int i = dsize-n ; i < dsize ; ++i )
        {
            setzero((*this)("&",i));
        }
    }

    else if ( n < 0 )
    {
        resize(dsize+n);
    }

    return *this;
}

template <class T>
void Vector<T>::prealloc(int newallocsize)
{
    if ( imoverhere )
    {
        overhere().prealloc(newallocsize);
    }

    else if ( content && !nbase )
    {
        NiceAssert( ( newallocsize >= 0 ) || ( newallocsize == -1 ) );

        (*content).prealloc(newallocsize);
    }
}

template <class T>
void Vector<T>::useStandardAllocation(void)
{
    if ( imoverhere )
    {
        overhere().useStandardAllocation();
    }

    else if ( content && !nbase )
    {
        (*content).useStandardAllocation();
    }
}

template <class T>
void Vector<T>::useTightAllocation(void)
{
    if ( imoverhere )
    {
        overhere().useTightAllocation();
    }

    else if ( content && !nbase )
    {
        (*content).useTightAllocation();
    }
}

template <class T>
void Vector<T>::useSlackAllocation(void)
{
    if ( imoverhere )
    {
        overhere().useSlackAllocation();
    }

    else if ( content && !nbase )
    {
        (*content).useSlackAllocation();
    }
}

template <class T>
void Vector<T>::applyOnAll(void (*fn)(T &, int), int argx)
{
    if ( !ccontent )
    {
        fix();
    }

    NiceAssert( !infsize() );
    NiceAssert( !nbase );
    NiceAssert( content );

    (*content).applyOnAll(fn,argx);
}

template <class T>
Vector<T> &Vector<T>::append(int i, const T &a)
{
    i = ( i >= 0 ) ? i : size();

    NiceAssert( !infsize() );
    NiceAssert( i >= size() );

    resize(i+1);

    (*this).set(i,a);

    return *this;
}

template <class T>
Vector<T> &Vector<T>::append(int i, const Vector<T> &a)
{
    i = ( i >= 0 ) ? i : size();

    NiceAssert( !infsize() );
    NiceAssert( i >= size() );

    //if ( a.size() )
    {
        resize(i+(a.size()));

        for ( int j = 0 ; j < a.size() ; ++j )
        {
            (*this).set(i+j,a(j));
        }
    }

    return *this;
}


// Function application

template <class T>
Vector<T> &Vector<T>::applyon(T (*fn)(T))
{
    if ( imoverhere )
    {
        overhere().applyon(fn);
    }

    else
    {
	for ( int i = 0 ; i < dsize ; ++i )
	{
            (*this).set(i,fn((*this)(i)));
	}
    }

    return *this;
}

template <class T>
Vector<T> &Vector<T>::applyon(T (*fn)(const T &))
{
    if ( imoverhere )
    {
        overhere().applyon(fn);
    }

    else
    {
	for ( int i = 0 ; i < dsize ; ++i )
	{
            (*this).set(i,fn((*this)(i)));
	}
    }

    return *this;
}

template <class T>
Vector<T> &Vector<T>::applyon(T &(*fn)(T &))
{
    if ( imoverhere )
    {
        overhere().applyon(fn);
    }

    else
    {
	for ( int i = 0 ; i < dsize ; ++i )
	{
            fn((*this)("&",i));
	}
    }

    return *this;
}

template <class T>
Vector<T> &Vector<T>::applyon(T (*fn)(T, const void *), const void *a)
{
    if ( imoverhere )
    {
        overhere().applyon(fn,a);
    }

    else
    {
	for ( int i = 0 ; i < dsize ; ++i )
	{
            (*this).set(i,fn((*this)(i),a));
	}
    }

    return *this;
}

template <class T>
Vector<T> &Vector<T>::applyon(T (*fn)(const T &, const void *), const void *a)
{
    if ( imoverhere )
    {
        overhere().applyon(fn,a);
    }

    else
    {
	for ( int i = 0 ; i < dsize ; ++i )
	{
            (*this).set(i,fn((*this)(i),a));
	}
    }

    return *this;
}

template <class T>
Vector<T> &Vector<T>::applyon(T &(*fn)(T &, const void *), const void *a)
{
    if ( imoverhere )
    {
        overhere().applyon(fn,a);
    }

    else
    {
	for ( int i = 0 ; i < dsize ; ++i )
	{
            fn((*this)("&",i),a);
	}
    }

    return *this;
}


















































































































template <class T>
const T &max(T &res, const Vector<T> &a, const Vector<T> &b, int &ii)
{
    NiceAssert( a.size() == b.size() );
    NiceAssert( a.size() );

    T locval;

    ii = 0;

    res  = a(0);
    res -= b(0);

    for ( int i = 1 ; i < a.size() ; ++i )
    {
        locval  = a(i);
        locval -= b(i);

        if ( locval > res )
        {
            res = locval;
            ii = i;
        }
    }

    return res;
}


template <class T>
const T &max(const Vector<T> &a, int &ii)
{
    NiceAssert( a.size() );

    ii = 0;

    for ( int i = 1 ; i < a.size() ; ++i )
    {
        if ( a(i) > a(ii) )
        {
            ii = i;
        }
    }

    return a(ii);
}

template <class T>
T max(const Vector<T> &a, const Vector<T> &b, int &ii)
{
    T res;

    return max(res,a,b,ii);
}



























template <class T>
const T &min(T &res, const Vector<T> &a, const Vector<T> &b, int &ii)
{
    NiceAssert( a.size() == b.size() );
    NiceAssert( a.size() );

    T locval;

    ii = 0;

    res  = a(0);
    res -= b(0);

    for ( int i = 1 ; i < a.size() ; ++i )
    {
        locval  = a(i);
        locval -= b(i);

        if ( locval < res )
        {
            res = locval;
            ii = i;
        }
    }

    return res;
}

template <class T>
const T &min(const Vector<T> &a, int &ii)
{
    NiceAssert( a.size() );

    ii = 0;

    for ( int i = 1 ; i < a.size() ; ++i )
    {
        if ( a(i) < a(ii) )
        {
            ii = i;
        }
    }

    return a(ii);
}

template <class T>
T min(const Vector<T> &a, const Vector<T> &b, int &ii)
{
    T res;

    return min(res,a,b,ii);
}


























template <class T>
const T &maxabs(T &res, const Vector<T> &a, int &ii)
{
    NiceAssert( a.size() );

    ii = 0;

    for ( int i = 1 ; i < a.size() ; ++i )
    {
        if ( abs2(a(i)) > abs2(a(ii)) )
        {
            ii = i;
        }
    }

    res = abs2(a(ii));

    return res;
}

template <class T>
T maxabs(const Vector<T> &a, int &ii)
{
    T res;

    return maxabs(res,a,ii);
}

template <class T>
const T &minabs(T &res, const Vector<T> &a, int &ii)
{
    NiceAssert( a.size() );

    ii = 0;

    for ( int i = 1 ; i < a.size() ; ++i )
    {
        if ( abs2(a(i)) < abs2(a(ii)) )
        {
            ii = i;
        }
    }

    res = abs2(a(ii));

    return res;
}

template <class T>
T minabs(const Vector<T> &a, int &ii)
{
    T res;

    return minabs(res,a,ii);
}

template <class T>
const T &sqabsmax(T &res, const Vector<T> &a)
{
    NiceAssert( a.size() );

    int ii = 0;

    for ( int i = 1 ; i < a.size() ; ++i )
    {
        if ( norm2(a(i)) > norm2(a(ii)) )
        {
            ii = i;
        }
    }

    res = norm2(a(ii));

    return res;
}

template <class T>
T sqabsmax(const Vector<T> &a)
{
    T res;

    return sqabsmax(res,a);
}

template <class T>
const T &sqabsmin(T &res, const Vector<T> &a)
{
    NiceAssert( a.size() );

    int ii = 0;

    for ( int i = 1 ; i < a.size() ; ++i )
    {
        if ( norm2(a(i)) < norm2(a(ii)) )
        {
            ii = i;
        }
    }

    res = norm2(a(ii));

    return res;
}

template <class T>
T sqabsmin(const Vector<T> &a)
{
    T res;

    return sqabsmin(res,a);
}

template <class T>
T sum(const Vector<T> &a)
{
    T res;

    return sum(res,a);
}

template <class T>
T sqsum(const Vector<T> &a)
{
    T res;

    return sqsum(res,a);
}

template <class T>
T orall(const Vector<T> &a)
{
    T res;

    return orall(res,a);
}

template <class T>
T andall(const Vector<T> &a)
{
    T res;

    return andall(res,a);
}

template <class T>
const T &sum(T &res, const Vector<T> &a)
{
    if ( a.infsize() )
    {
        NiceThrow("Sorry, this aint defined 'ere");
    }

    if ( a.size() )
    {
        res = a(0);

        for ( int i = 1 ; i < a.size() ; ++i )
        {
            res += a(i);
        }
    }

    else
    {
        setzero(res);
    }

    return res;
}

template <> inline const double &sum(double &res, const Vector<double> &a)
{
    return oneProduct(res,a);
}

template <class T>
const T &sum(T &res, const Vector<T> &a, const Vector<double> &weights)
{
    if ( a.infsize() || weights.infsize() )
    {
        NiceThrow("Sorry, this aint defined 'ere");
    }

    else if ( a.size() )
    {
        res  = a(0);
        res *= weights(0);

        if ( a.size() > 1 )
        {
            T temp;

            for ( int i = 1 ; i < a.size() ; ++i )
            {
                temp  = a(i);
                temp *= weights(i);

                res += temp;
            }
        }
    }

    else
    {
        setzero(res);
    }

    return res;
}

template <> inline const double &sum(double &res, const Vector<double> &a, const Vector<double> &b)
{
    return innerProduct(res,a,b);
}

template <class S, class T>
const T &sumb(T &result, const Vector<S> &left_op, const Vector<T> &right_op)
{
    if ( left_op.infsize() || right_op.infsize() )
    {
        NiceThrow("Sorry, this aint defined 'ere");
    }

    else
    {
        NiceAssert( left_op.size() == right_op.size() );

        T temp;

        setzero(result);
        setzero(temp);

        if ( left_op.size() )
        {
            result = right_op(0);
            rightmult(left_op(0),result);

            //if ( left_op.size() > 1 )
            {
                for ( int i = 1 ; i < left_op.size() ; ++i )
 	        {
                    temp = right_op(i);
                    rightmult(left_op(i),temp);

                    result += temp;
	        }
	    }
        }

        postProInnerProd(result);
    }

    return result;
}

template <> inline const double &sumb(double &res, const Vector<double> &a, const Vector<double> &b)
{
    return innerProduct(res,a,b);
}

template <class T>
const T &sqsum(T &res, const Vector<T> &a)
{
    T temp;

    if ( a.size() )
    {
        res =  a(0);
        res *= a(0);

        //if ( a.size() > 1 )
        {
            for ( int i = 1 ; i < a.size() ; ++i )
            {
                temp =  a(i);
                temp *= a(i);

                res += temp;
            }
        }
    }

    else
    {
        setzero(res);
    }

    return res;
}

template <class T>
T abssum(const Vector<T> &a)
{
    T res;

    return abssum(res,a);
}

template <class T>
const T &abssum(T &res, const Vector<T> &a)
{
    if ( a.size() )
    {
        res = abs2(a(0));

        //if ( a.size() > 1 )
        {
            for ( int i = 1 ; i < a.size() ; ++i )
            {
                res += abs2(a(i));
            }
        }
    }

    else
    {
        setzero(res);
    }

    return res;
}

template <> inline const double &abssum(double &res, const Vector<double> &a)
{
    res = 0;

    //if ( a.size() )
    {
        for ( int i = 0 ; i < a.size() ; ++i )
        {
            res += abs2(a.v(i));
        }
    }

    return res;
}

template <> inline const double &sqsum(double &res, const Vector<double> &a)
{
    return sumb(res,a,a);
}

template <class T>
const T &orall(T &res, const Vector<T> &a)
{
    T temp;

    if ( a.size() )
    {
        res =  a(0);

        //if ( a.size() > 1 )
        {
            for ( int i = 1 ; i < a.size() ; ++i )
            {
                res |= a(i);
            }
        }
    }

    else
    {
        setzero(res);
    }

    return res;
}

template <class T>
const T &andall(T &res, const Vector<T> &a)
{
    T temp;

    if ( a.size() )
    {
        res =  a(0);

        //if ( a.size() > 1 )
        {
            for ( int i = 1 ; i < a.size() ; ++i )
            {
                res &= a(i);
            }
        }
    }

    else
    {
        setzero(res);
    }

    return res;
}

template <class T>
const T &prod(T &res, const Vector<T> &a)
{
    if ( a.infsize() )
    {
        NiceThrow("Good try old chap");
    }

    else if ( a.size() )
    {
        res = a(0);

        for ( int i = 1 ; i < a.size() ; ++i )
        {
            res *= a(i);
	}
    }

    else
    {
        setident(res);
    }

    return res;
}

template <class T>
const T logsum(const Vector<T> &a)
{
    T res;

    if ( a.infsize() )
    {
        NiceThrow("Yaaarrrrrr");
    }

    else if ( a.size() )
    {
        res = log(a(0));

        for ( int i = 1 ; i < a.size() ; ++i )
        {
            res += log(a(i));
        }
    }

    else
    {
        setzero(res);
    }

    return res;
}

template <class T>
const T logsum(const Vector<T> &a, int maxsize)
{
    T res;

    if ( a.infsize() )
    {
        NiceThrow("Yaaarrrrrr");
    }

    else if ( a.size() && maxsize )
    {
        res = log(a(0));

        for ( int i = 1 ; ( i < a.size() ) && ( i < maxsize ) ; ++i )
        {
            res += log(a(i));
        }
    }

    else
    {
        setzero(res);
    }

    return res;
}

template <class T>
T prod(const Vector<T> &a)
{
    T res;

    if ( a.infsize() )
    {
        NiceThrow("Good try old chap");
    }

    else if ( a.size() )
    {
        res = a(0);

        for ( int i = 1 ; i < a.size() ; ++i )
        {
            res *= a(i);
	}
    }

    else
    {
        setident(res);
    }

    return res;
}

template <class T>
T prod(const Vector<T> &a, int maxsize)
{
    T res;

    if ( a.infsize() )
    {
        NiceThrow("Good try old chap");
    }

    else if ( a.size() && maxsize )
    {
        res = a(0);

        for ( int i = 1 ; ( i < a.size() ) && ( i < maxsize ) ; ++i )
        {
            res *= a(i);
	}
    }

    else
    {
        setident(res);
    }

    return res;
}

template <class T>
const T &Prod(T &res, const Vector<T> &a)
{
    if ( a.size() )
    {
        res = a(a.size()-1);

        //if ( a.size() > 1 )
	{
            for ( int i = a.size()-2 ; i >= 0 ; --i )
	    {
                res *= a(i);
	    }
	}
    }

    else
    {
        setident(res);
    }

    return res;
}

template <class T>
T Prod(const Vector<T> &a)
{
    T res;

    return Prod(res,a);
}

template <class T>
const T &mean(T &res, const Vector<T> &a)
{
    if ( a.size() == 1 )
    {
        res = a(0);
    }

    else if ( a.size() )
    {
        sum(res,a);
        res *= 1/((double) a.size());
    }

    else
    {
	setzero(res);
    }

    return res;
}

template <class T>
const T &mean(T &res, const Vector<T> &a, const Vector<double> &weights)
{
    NiceAssert( a.size() == weights.size() );

    if ( a.size() )
    {
        sum(res,a,weights);
        res *= 1/((double) a.size());
    }

    else
    {
	setzero(res);
    }

    return res;
}

template <class T>
T mean(const Vector<T> &a)
{
    T res;

    return mean(res,a);
}

template <class T>
const T &sqmean(T &res, const Vector<T> &a)
{
    if ( a.size() )
    {
        sqsum(res,a);
        res *= 1/((double) a.size());
    }

    else
    {
	setzero(res);
    }

    return res;
}

template <class T>
T sqmean(const Vector<T> &a)
{
    T res;

    return sqmean(res,a);
}

template <class T>
const T &vari(T &res, const Vector<T> &a)
{
    // mean(a) = 1/N sum_i a_i
    // sqmean(a) = 1/N sum_i a_i^2
    //
    // vari(a) = 1/N sum_i ( a_i - mean(a) )^2
    //         = 1/N sum_i a_i^2 - 2 mean(a) 1/N sum_i a_i + mean(a)^2 1/N sum_i 1
    //         = sqmean(a) - 2 mean(a)^2 + mean(a)^2
    //         = sqmean(a) - mean(a)^2

    T ameansq;

    res =  sqmean(a);
    ameansq =  mean(a);
    ameansq *= ameansq;
    res -= ameansq;

    return res;
}

template <class T>
T vari(const Vector<T> &a)
{
    T res;

    return vari(res,a);
}

template <class T>
const T &stdev(T &res, const Vector<T> &a)
{
    res = sqrt(vari(a));

    return res;
}

template <class T>
T stdev(const Vector<T> &a)
{
    T res;

    return stdev(res,a);
}

template <class T>
const T &median(const Vector<T> &a, int &ii)
{
    const T *res = nullptr;

    ii = 0;

    if ( a.size() == 1 )
    {
        res = &(a(0));
    }

    else if ( a.size() > 1 )
    {
        // Aim: a(outdex) should be arranged from largest to smallest

        Vector<int> outdex;

        int i,j;

        for ( i = a.size()-1 ; i >= 0 ; --i )
        {
            j = 0;

            if ( outdex.size() )
            {
                for ( j = 0 ; j < outdex.size() ; ++j )
                {
                    if ( a(outdex.v(j)) <= a(i) )
                    {
                        break;
                    }
                }
            }

            outdex.add(j);
            outdex.sv(j,i);
        }

        ii = outdex(a.size()/2);

        res = &(a(ii));
    }

    else
    {
        static thread_local int frun = 1;
        static thread_local T defres;

        ii = 0;

        if ( frun )
        {
            setzero(defres);
            frun = 0;
        }

        res = &defres;
    }

    return *res;
}

template <class S> Vector<S> angle(const Vector<S> &a)
{
    Vector<S> temp(a);

    double tempabs = abs2(temp);

    if ( tempabs != 0.0 )
    {
        temp.scale(1/tempabs);
    }

    return temp;
}

template <class S> Vector<S> &angle(Vector<S> &res, const Vector<S> &a)
{
    res = a;

    double tempabs = abs2(res);

    if ( tempabs != 0.0 )
    {
        res.scale(1/tempabs);
    }

    return res;
}

template <class S> Vector<double> eabs2(const Vector<S> &a)
{
    Vector<double> temp(a.size());

    //if ( temp.size() )
    {
        for ( int i = 0 ; i < temp.size() ; ++i )
        {
            temp.sv(i,abs2(temp(i)));
        }
    }

    return temp;
}

template <class S> Vector<S> vangle(const Vector<S> &a, const Vector<S> &defsign)
{
    Vector<S> temp(a);

    double tempabs = abs2(temp);

    if ( tempabs != 0.0 )
    {
        temp.scale(1/tempabs);
    }

    else
    {
        temp = defsign;
    }

    return temp;
}

template <class S> Vector<S> &vangle(Vector<S> &res, const Vector<S> &a, const Vector<S> &defsign)
{
    res = a;

    double tempabs = abs2(res);

    if ( tempabs != 0.0 )
    {
        res.scale(1/tempabs);
    }

    else
    {
        res = defsign;
    }

    return res;
}

template <class S> double abs1(const Vector<S> &a)
{
    return norm1(a);
}

template <class S> double abs2(const Vector<S> &a)
{
    return sqrt(norm2(a));
}

template <class S> double absp(const Vector<S> &a, double p)
{
    return pow(normp(a,p),1/p);
}

template <class S> double norm1(const Vector<S> &a)
{
    int size = a.size();
    double result = 0;

    if ( a.infsize() )
    {
        result = a.norm1();
    }

    else
    {
        for ( int i = 0 ; i < size ; ++i )
        {
            result += (double) norm1(a(i));
        }
    }

    return result;
}

template <class S> double norm2(const Vector<S> &a)
{
    int size = a.size();
    double result = 0;

    if ( a.infsize() )
    {
        result = a.norm2();
    }

    else
    {
        for ( int i = 0 ; i < size ; ++i )
        {
            result += (double) norm2(a(i));
	}
    }

    return result;
}

template <class S> double normp(const Vector<S> &a, double p)
{
    int size = a.size();
    double result = 0;

    if ( a.infsize() )
    {
        result = a.normp(p);
    }

    else
    {
        for ( int i = 0 ; i < size ; ++i )
        {
            result += (double) normp(a(i),p);
        }
    }

    return result;
}

template <class S> double absinf(const Vector<S> &a)
{
    int size = a.size();
    double maxval = 0;
    double locval;

    if ( a.infsize() )
    {
        maxval = a.absinf();
    }

    else
    {
        for ( int i = 0 ; i < size ; ++i )
        {
            locval = (double) absinf(a(i));

            if ( locval > maxval )
            {
                maxval = locval;
	    }
	}
    }

    return maxval;
}

template <class S> double abs0(const Vector<S> &a)
{
    int size = a.size();
    double maxval = 0;
    double locval;

    if ( a.infsize() )
    {
        maxval = a.absinf();
    }

    else
    {
        for ( int i = 0 ; i < size ; ++i )
        {
            locval = (double) abs0(a(i));

            if ( !i || ( locval < maxval ) )
            {
                maxval = locval;
	    }
	}
    }

    return maxval;
}

template <class T> Vector<T> &setident(Vector<T> &a)
{
    return a.ident();
}

template <class S> Vector<double> &seteabs2(Vector<S> &a)
{
    //if ( a.size() )
    {
        for ( int i = 0 ; i < a.size() ; ++i )
        {
            a("&",i) = abs2(a(i));
        }
    }

    return a;
}

template <class T> Vector<T> &setzero(Vector<T> &a)
{
    return a.zero();
}

template <class T> Vector<T> &setzeropassive(Vector<T> &a)
{
    return a.zeropassive();
}

template <class T> Vector<T> &setposate(Vector<T> &a)
{
    return a.posate();
}

template <class T> Vector<T> &setnegate(Vector<T> &a)
{
    return a.negate();
}

template <class T> Vector<T> &setconj(Vector<T> &a)
{
    return a.conj();
}

template <class T> Vector<T> &setrand(Vector<T> &a)
{
    return a.rand();
}




















template <class T> int testisvnan(const Vector<T> &x)
{
    int res = 0;

    //if ( x.size() )
    {
        for ( int i = 0 ; !res && ( i < x.size() ) ; ++i )
        {
            if ( testisvnan(x(i)) )
            {
                res = 1;
                break;
            }
        }
    }

    return res;
}

template <class T> int testisinf (const Vector<T> &x)
{
    int res = 0;

    //if ( x.size() && !testisvnan(x) )
    if ( !testisvnan(x) )
    {
        for ( int i = 0 ; !res && ( i < x.size() ) ; ++i )
        {
            if ( testisinf(x(i)) )
            {
                res = 1;
                break;
            }
        }
    }

    return res;
}

template <class T> int testispinf(const Vector<T> &x)
{
    int pinfcnt = 0;

    //if ( x.size() )
    {
        for ( int i = 0 ; i < x.size() ; ++i )
        {
            if ( testispinf(x(i)) )
            {
                ++pinfcnt;
                break;
            }
        }
    }

    return ( ( pinfcnt == x.size() ) && x.size() ) ? 1 : 0;
}

template <class T> int testisninf(const Vector<T> &x)
{
    int ninfcnt = 0;

    //if ( x.size() )
    {
        for ( int i = 0 ; i < x.size() ; ++i )
        {
            if ( testisninf(x(i)) )
            {
                ++ninfcnt;
                break;
            }
        }
    }

    return ( ( ninfcnt == x.size() ) && x.size() ) ? 1 : 0;
}



























//phantomxyz
//----------------------------------------------


template <class T>
double &oneProductAssumeReal(double &res, const Vector<T> &a)
{
    if ( a.infsize() )
    {
        a.inner1Real(res);
    }

    else
    {
        int dim = a.size();

        res = 0;

        for ( int i = 0 ; i < dim ; ++i )
        {
            res += (double) a(i);
	}
    }

    return res;
}

template <> inline double &oneProductAssumeReal(double &res, const Vector<double> &a)
{
    int size = a.size();

    res = 0;

    if ( size && a.contiguous() )
    {
        res = fastoneProduct(&a(0),size);
    }

    else if ( size )
    {
        res = fastoneProduct(a.unsafeccontent(),a.unsafepivot(),a.unsafeib(),a.unsafeis(),size);
    }

    return res;
}

template <class T>
T &oneProduct(T &result, const Vector<T> &a)
{
    if ( a.infsize() )
    {
        a.inner1(result);
    }

    else
    {
        int dim = a.size();

        if ( dim )
        {
            result = a(0);

            for ( int i = 1 ; i < dim ; ++i )
            {
                result += a(i);
	    }
	}

        else
        {
            setzero(result);
        }

        postProInnerProd(result);
    }

    return result;
}

template <> inline double &oneProduct(double &result, const Vector<double> &a)
{
    return oneProductAssumeReal(result,a);
}

template <class T>
T &oneProductScaled(T &result, const Vector<T> &a, const Vector<T> &scale)
{
    if ( a.infsize() )
    {
        NiceThrow("Scaled FuncVector inner product not supported!");
    }

    else
    {
        setzero(result);

        int dim = a.size();

        if ( dim )
        {
            result  = a(0);
            result /= scale(0);

            for ( int i = 1 ; i < dim ; ++i )
            {
                result += a(i)/scale(i);
	    }
	}

        postProInnerProd(result);
    }

    return result;
}

template <class T>
T &indexedoneProduct(T &result, const Vector<int> &n, const Vector<T> &a)
{
    if ( a.infsize() )
    {
        NiceThrow("Index FuncVector inner product not supported!");
    }

    else
    {
        int nsize = n.size();
        int asize = a.size(); //indsize();

        setzero(result);

        if ( nsize && asize )
        {
            int npos = 0;
	    int apos = 0;

            int nelm;
	    int aelm;

	    while ( ( npos < nsize ) && ( apos < asize ) )
	    {
                nelm = n.v(npos);
                aelm = apos; //a.ind(apos);

  	        if ( aelm == nelm )
	        {
                    // NB: we process the product in pairs, so if T == gentype
                    //     then ("a"*"a")*("b"*"b") will "correctly" evaluate
                    //     to 1, not error (ie symbolic multiplication).

                    result += a(apos);

                    ++npos;
		    ++apos;
	        }

                else if ( nelm <= aelm )
	        {
		    ++npos;
	        }

                else
	        {
		    ++apos;
	        }
	    }
	}

        postProInnerProd(result);
    }

    return result;
}

template <class T>
T &indexedoneProductScaled(T &result, const Vector<int> &n, const Vector<T> &a, const Vector<T> &scale)
{
    if ( a.infsize() )
    {
        NiceThrow("Index FuncVector inner product not supported!");
    }

    else
    {
        int nsize = n.size();
        int asize = a.size(); //indsize();

        setzero(result);

        if ( nsize && asize )
        {
            int npos = 0;
	    int apos = 0;

            int nelm;
	    int aelm;

	    while ( ( npos < nsize ) && ( apos < asize ) )
	    {
                nelm = n.v(npos);
                aelm = apos; //a.ind(apos);

	        if ( aelm == nelm )
	        {
                    // NB: we process the product in pairs, so if T == gentype
                    //     then ("a"*"a")*("b"*"b") will "correctly" evaluate
                    //     to 1, not error (ie symbolic multiplication).

                    result += a(apos)/scale(apos);

                    ++npos;
		    ++apos;
	        }

                else if ( nelm <= aelm )
	        {
		    ++npos;
	        }

                else
	        {
		    ++apos;
	        }
	    }
        }

        postProInnerProd(result);
    }

    return result;
}

//----------------------------------------------

template <class T>
T &innerProduct(T &result, const Vector<T> &a, const Vector<T> &b)
{
  if ( a.infsize() || b.infsize() )
  {
    NiceAssert( a.infsize() && b.infsize() );

    a.inner2(result,b,1);
  }

  else
  {
    NiceAssert( a.size() == b.size() );

    int dim = a.size();

    if svm_constexpr_if ( !(std::is_floating_point<T>::value) && !(std::is_integral<T>::value) )
    {
        // standard

        T temp;

        setzero(temp);
        setzero(result);

        if ( dim )
        {
            result = b(0);
            setconj(result);
            result *= a(0);
            setconj(result);

            for ( int i = 1 ; i < dim ; ++i )
            {
                // conj(l).r = conj(conj(r).l) (staying type T at all times)

                temp = b(i);
                setconj(temp);
                temp *= a(i);
                setconj(temp);

                result += temp;
	    }
        }

        postProInnerProd(result);
    }

    else
    {
        // fast

        result = 0;

        for ( int i = 0 ; i < dim ; ++i )
        {
            result += a(i)*b(i);
        }
    }
  }

  return result;
}

template <> inline double &innerProduct(double &res, const Vector<double> &a, const Vector<double> &b)
{
    int size = a.size();

    res = 0;

    NiceAssert( size == b.size() );

    if ( size && a.contiguous() && b.contiguous() )
    {
        res = fasttwoProduct(&a(0),&b(0),size);
    }

    else if ( size )
    {
        res = fasttwoProduct(a.unsafeccontent(),a.unsafepivot(),a.unsafeib(),a.unsafeis(),b.unsafeccontent(),b.unsafepivot(),b.unsafeib(),b.unsafeis(),size);
    }

    return res;
}

template <class T>
T &twoProduct(T &result, const Vector<T> &a, const Vector<T> &b)
{
  if ( a.infsize() || b.infsize() )
  {
    NiceAssert( a.infsize() && b.infsize() );

    a.inner2(result,b,1);
  }

  else
  {
    NiceAssert( a.size() == b.size() );

    int dim = a.size();

    if svm_constexpr_if ( !(std::is_floating_point<T>::value) && !(std::is_integral<T>::value) )
    {
        T temp;

        setzero(result);
        setzero(temp);

        if ( dim )
        {
            result = b(0);
            rightmult(a(0),result);

            for ( int i = 1 ; i < dim ; ++i )
            {
                temp = b(i);
                rightmult(a(i),temp);

                result += temp;
            }
        }

        postProInnerProd(result);
    }

    else
    {
        // fast

        result = 0;

        for ( int i = 0 ; i < dim ; ++i )
        {
            result += a(i)*b(i);
        }
    }
  }

  return result;
}

template <> inline double &twoProduct(double &result, const Vector<double> &a, const Vector<double> &b)
{
    return innerProduct(result,a,b);
}

template <class T>
double &innerProductAssumeReal(double &res, const Vector<T> &a, const Vector<T> &b)
{
    if ( a.infsize() || b.infsize() )
    {
        NiceAssert( b.infsize() && a.infsize() );

        a.inner2Real(res,b,0);
    }

    else
    {
        int dim = a.size();

        NiceAssert( b.size() == dim );

        res = 0;

        for ( int i = 0 ; i < dim ; ++i )
        {
            scaladd(res,a(i),b(i));
        }
    }

    return res;
}

template <class T>
double &twoProductAssumeReal(double &res, const Vector<T> &a, const Vector<T> &b)
{
    return innerProductAssumeReal(res,a,b);
}

template <> inline double &innerProductAssumeReal(double &result, const Vector<double> &a, const Vector<double> &b)
{
    return twoProduct(result,a,b);
}

template <> inline double &twoProductAssumeReal(double &result, const Vector<double> &a, const Vector<double> &b)
{
    return twoProduct(result,a,b);
}

template <class T>
T &innerProductRevConj(T &result, const Vector<T> &a, const Vector<T> &b)
{
    if ( a.infsize() || b.infsize() )
    {
        NiceAssert( a.infsize() && b.infsize() );

        return a.inner2(result,b,2);
    }

    NiceAssert( a.size() == b.size() );

    int dim = a.size();

    if svm_constexpr_if ( !(std::is_floating_point<T>::value) && !(std::is_integral<T>::value) )
    {
        T temp;

        setzero(result);
        setzero(temp);

        if ( dim )
        {
            result = b(0);
            setconj(result);
            rightmult(a(0),result);

            for ( int i = 1 ; i < dim ; ++i )
            {
                // conj(l).r = conj(conj(r).l) (staying type T at all times)

                temp = b(i);
                setconj(temp);
                rightmult(a(i),temp);

                result += temp;
	    }
	}
    }

    else
    {
        // fast

        result = 0;

        for ( int i = 0 ; i < dim ; ++i )
        {
            result += a(i)*b(i);
        }

        return result;
    }

    return postProInnerProd(result);
}

template <> inline double &innerProductRevConj(double &result, const Vector<double> &a, const Vector<double> &b)
{
    return innerProduct(result,a,b);
}

template <class T>
T &innerProductScaled(T &result, const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale)
{
    if ( a.infsize() || b.infsize() )
    {
        NiceThrow("Scaled FuncVector inner product not supported.");

        return result;
    }

    NiceAssert( a.size() == b.size() );
    NiceAssert( a.size() == scale.size() );

    int dim = a.size();

    if svm_constexpr_if ( !(std::is_floating_point<T>::value) && !(std::is_integral<T>::value) )
    {
        T tempa;
        T tempb;

        setzero(result);
        setzero(tempa);
        setzero(tempb);

        if ( dim )
        {
            tempa  = a(0);
            tempa /= scale(0);

            tempb  = b(0);
            tempb /= scale(0);

            setconj(tempa);
            //setconj(tempb);
            rightmult(tempa,tempb);

            result = tempb;

            for ( int i = 1 ; i < dim ; ++i )
            {
                // conj(l).r = conj(conj(r).l) (staying type T at all times)

                tempa  = a(i);
                tempa /= scale(i);

                tempb  = b(i);
                tempb /= scale(i);

                setconj(tempa);
                //setconj(tempb);
                rightmult(tempa,tempb);

                result += tempb;
            }
	}
    }

    else
    {
        result = 0;

        for ( int i = 0 ; i < dim ; ++i )
        {
            result += a(i)*b(i)/(scale(i)*scale(i));
	}

        return result;
    }

    return postProInnerProd(result);
}

template <class T>
T &twoProductScaled(T &result, const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale)
{
    if ( a.infsize() || b.infsize() )
    {
        NiceThrow("Scaled FuncVector inner product not supported.");

        return result;
    }

    NiceAssert( a.size() == b.size() );
    NiceAssert( a.size() == scale.size() );

    int dim = a.size();

    if svm_constexpr_if ( !(std::is_floating_point<T>::value) && !(std::is_integral<T>::value) )
    {
        T tempa;
        T tempb;

        setzero(result);
        setzero(tempa);
        setzero(tempb);

        if ( dim )
        {
            tempa  = a(0);
            tempa /= scale(0);

            tempb  = b(0);
            tempb /= scale(0);

            //setconj(tempa);
            //setconj(tempb);
            rightmult(tempa,tempb);

            result = tempb;

            for ( int i = 1 ; i < dim ; ++i )
	    {
                // conj(l).r = conj(conj(r).l) (staying type T at all times)

                tempa  = a(i);
                tempa /= scale(i);

                tempb  = b(i);
                tempb /= scale(i);

                //setconj(tempa);
                //setconj(tempb);
                rightmult(tempa,tempb);

                result += tempb;
	    }
	}
    }

    else
    {
        result = 0;

        for ( int i = 0 ; i < dim ; ++i )
        {
            result += a(i)*b(i)/(scale(i)*scale(i));
	}

        return result;
    }

    return postProInnerProd(result);
}

template <class T>
T &innerProductScaledRevConj(T &result, const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale)
{
    if ( a.infsize() || b.infsize() )
    {
        NiceThrow("Scaled FuncVector inner product not supported.");

        return result;
    }

    NiceAssert( a.size() == b.size() );
    NiceAssert( a.size() == scale.size() );

    int dim = a.size();

    if svm_constexpr_if ( !(std::is_floating_point<T>::value) && !(std::is_integral<T>::value) )
    {
        T tempa;
        T tempb;

        setzero(result);
        setzero(tempa);
        setzero(tempb);

        if ( dim )
        {
            tempa  = a(0);
            tempa /= scale(0);

            tempb  = b(0);
            tempb /= scale(0);

            //setconj(tempa);
            setconj(tempb);
            rightmult(tempa,tempb);

            result = tempb;

            for ( int i = 1 ; i < dim ; ++i )
	    {
                // conj(l).r = conj(conj(r).l) (staying type T at all times)

                tempa  = a(i);
                tempa /= scale(i);

                tempb  = b(i);
                tempb /= scale(i);

                //setconj(tempa);
                setconj(tempb);
                rightmult(tempa,tempb);

                result += tempb;
	    }
	}
    }

    else
    {
        result = 0;

        for ( int i = 0 ; i < dim ; ++i )
        {
            result += a(i)*b(i)/(scale(i)*scale(i));
	}

        return result;
    }

    return postProInnerProd(result);
}

template <class T>
T &innerProductRightScaled(T &result, const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale)
{
    if ( a.infsize() || b.infsize() )
    {
        NiceThrow("Scaled FuncVector inner product not supported.");

        return result;
    }

    NiceAssert( a.size() == b.size() );
    NiceAssert( a.size() == scale.size() );

    int dim = a.size();

    if svm_constexpr_if ( !(std::is_floating_point<T>::value) && !(std::is_integral<T>::value) )
    {
        T tempa;
        T tempb;

        setzero(result);
        setzero(tempa);
        setzero(tempb);

        if ( dim )
        {
            tempa  = a(0);
            //tempa /= scale(0);

            tempb  = b(0);
            tempb /= scale(0);

            setconj(tempa);
            //setconj(tempb);
            rightmult(tempa,tempb);

            result = tempb;

            for ( int i = 1 ; i < dim ; ++i )
	    {
                // conj(l).r = conj(conj(r).l) (staying type T at all times)

                tempa  = a(i);
                //tempa /= scale(i);

                tempb  = b(i);
                tempb /= scale(i);

                setconj(tempa);
                //setconj(tempb);
                rightmult(tempa,tempb);

                result += tempb;
	    }
	}
    }

    else
    {
        result = 0;

        for ( int i = 0 ; i < dim ; ++i )
        {
            result += a(i)*b(i)/scale(i);
	}

        return result;
    }

    return postProInnerProd(result);
}

template <class T>
T &twoProductRightScaled(T &result, const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale)
{
    if ( a.infsize() || b.infsize() )
    {
        NiceThrow("Scaled FuncVector inner product not supported.");

        return result;
    }

    NiceAssert( a.size() == b.size() );
    NiceAssert( a.size() == scale.size() );

    int dim = a.size();

    if svm_constexpr_if ( !(std::is_floating_point<T>::value) && !(std::is_integral<T>::value) )
    {
        T tempa;
        T tempb;

        setzero(result);
        setzero(tempa);
        setzero(tempb);

        if ( dim )
        {
            tempa  = a(0);
            //tempa /= scale(0);

            tempb  = b(0);
            tempb /= scale(0);

            //setconj(tempa);
            //setconj(tempb);
            rightmult(tempa,tempb);

            result = tempb;

            for ( int i = 1 ; i < dim ; ++i )
	    {
                // conj(l).r = conj(conj(r).l) (staying type T at all times)

                tempa  = a(i);
                //tempa /= scale(i);

                tempb  = b(i);
                tempa /= scale(i);

                //setconj(tempa);
                //setconj(tempb);
                rightmult(tempa,tempb);

                result += tempb;
	    }
	}
    }

    else
    {
        result = 0;

        for ( int i = 0 ; i < dim ; ++i )
        {
            result += a(i)*b(i)/scale(i);
	}

        return result;
    }

    return postProInnerProd(result);
}

template <class T>
T &innerProductRightScaledRevConj(T &result, const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale)
{
    if ( a.infsize() || b.infsize() )
    {
        NiceThrow("Scaled FuncVector inner product not supported.");

        return result;
    }

    NiceAssert( a.size() == b.size() );
    NiceAssert( a.size() == scale.size() );

    int dim = a.size();

    if svm_constexpr_if ( !(std::is_floating_point<T>::value) && !(std::is_integral<T>::value) )
    {
        T tempa;
        T tempb;

        setzero(result);
        setzero(tempa);
        setzero(tempb);

        if ( dim )
        {
            tempa  = a(0);
            //tempa /= scale(0);

            tempb  = b(0);
            tempb /= scale(0);

            //setconj(tempa);
            setconj(tempb);
            rightmult(tempa,tempb);

            result = tempb;

            for ( int i = 1 ; i < dim ; ++i )
	    {
                // conj(l).r = conj(conj(r).l) (staying type T at all times)

                tempa  = a(i);
                //tempa /= scale(i);

                tempb  = b(i);
                tempb /= scale(i);

                //setconj(tempa);
                setconj(tempb);
                rightmult(tempa,tempb);

                result += tempb;
	    }
	}
    }

    else
    {
        result = 0;

        for ( int i = 0 ; i < dim ; ++i )
        {
            result += a(i)*b(i)/scale(i);
	}

        return result;
    }

    return postProInnerProd(result);
}

template <class T>
T &innerProductLeftScaled(T &result, const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale)
{
    if ( a.infsize() || b.infsize() )
    {
        NiceThrow("Scaled FuncVector inner product not supported.");

        return result;
    }

    NiceAssert( a.size() == b.size() );
    NiceAssert( a.size() == scale.size() );

    int dim = a.size();

    if svm_constexpr_if ( !(std::is_floating_point<T>::value) && !(std::is_integral<T>::value) )
    {
        T tempa;
        T tempb;

        setzero(result);
        setzero(tempa);
        setzero(tempb);

        if ( dim )
        {
            tempa  = a(0);
            tempa /= scale(0);

            tempb  = b(0);
            //tempb /= scale(0);

            setconj(tempa);
            //setconj(tempb);
            rightmult(tempa,tempb);

            result = tempb;

            for ( int i = 1 ; i < dim ; ++i )
	    {
                // conj(l).r = conj(conj(r).l) (staying type T at all times)

                tempa  = a(i);
                tempa /= scale(i);

                tempb  = b(i);
                //tempb /= scale(i);

                setconj(tempa);
                //setconj(tempb);
                rightmult(tempa,tempb);

                result += tempb;
	    }
	}
    }

    else
    {
        result = 0;

        for ( int i = 0 ; i < dim ; ++i )
        {
            result += a(i)*b(i)/scale(i);
	}

        return result;
    }

    return postProInnerProd(result);
}

template <class T>
T &twoProductLeftScaled(T &result, const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale)
{
    if ( a.infsize() || b.infsize() )
    {
        NiceThrow("Scaled FuncVector inner product not supported.");

        return result;
    }

    NiceAssert( a.size() == b.size() );
    NiceAssert( a.size() == scale.size() );

    int dim = a.size();

    if svm_constexpr_if ( !(std::is_floating_point<T>::value) && !(std::is_integral<T>::value) )
    {
        T tempa;
        T tempb;

        setzero(result);
        setzero(tempa);
        setzero(tempb);

        if ( dim )
        {
            tempa  = a(0);
            tempa /= scale(0);

            tempb  = b(0);
            //tempb /= scale(0);

            //setconj(tempa);
            //setconj(tempb);
            rightmult(tempa,tempb);

            result = tempb;

            for ( int i = 1 ; i < dim ; ++i )
	    {
                // conj(l).r = conj(conj(r).l) (staying type T at all times)

                tempa  = a(i);
                tempa /= scale(i);

                tempb  = b(i);
                //tempb /= scale(i);

                //setconj(tempa);
                //setconj(tempb);
                rightmult(tempa,tempb);

                result += tempb;
	    }
	}
    }

    else
    {
        result = 0;

        for ( int i = 0 ; i < dim ; ++i )
        {
            result += a(i)*b(i)/scale(i);
	}

        return result;
    }

    return postProInnerProd(result);
}

template <class T>
T &innerProductLeftScaledRevConj(T &result, const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale)
{
    if ( a.infsize() || b.infsize() )
    {
        NiceThrow("Scaled FuncVector inner product not supported.");

        return result;
    }

    NiceAssert( a.size() == b.size() );
    NiceAssert( a.size() == scale.size() );

    int dim = a.size();

    if svm_constexpr_if ( !(std::is_floating_point<T>::value) && !(std::is_integral<T>::value) )
    {
        T tempa;
        T tempb;

        setzero(result);
        setzero(tempa);
        setzero(tempb);

        if ( dim )
        {
            tempa  = a(0);
            tempa /= scale(0);

            tempb  = b(0);
            //tempb /= scale(0);

            //setconj(tempa);
            setconj(tempb);
            rightmult(tempa,tempb);

            result = tempb;

            for ( int i = 1 ; i < dim ; ++i )
	    {
                // conj(l).r = conj(conj(r).l) (staying type T at all times)

                tempa  = a(i);
                tempa /= scale(i);

                tempb  = b(i);
                //tempb /= scale(i);

                //setconj(tempa);
                setconj(tempb);
                rightmult(tempa,tempb);

                result += tempb;
	    }
	}
    }

    else
    {
        result = 0;

        for ( int i = 0 ; i < dim ; ++i )
        {
            result += a(i)*b(i)/scale(i);
	}

        return result;
    }

    return postProInnerProd(result);
}

template <class T>
T &indexedinnerProduct(T &result, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b)
{
    if ( a.infsize() || b.infsize() )
    {
        NiceThrow("Indexed FuncVector inner product not supported.");

        return result;
    }

    int nsize = n.size();
    int asize = a.size(); //indsize();
    int bsize = b.size(); //indsize();

    T temp;

    setzero(result);
    setzero(temp);

    if ( nsize && asize && bsize )
    {
        int npos = 0;
	int apos = 0;
	int bpos = 0;

        int nelm;
	int aelm;
	int belm;

	while ( ( npos < nsize ) && ( apos < asize ) && ( bpos < bsize ) )
	{
            nelm = n.v(npos);
            aelm = apos; //a.ind(apos);
	    belm = bpos; //b.ind(bpos);

	    if ( ( aelm == nelm ) && ( belm == nelm ) )
	    {
                // conj(l).r = conj(conj(r).l) (staying type T at all times)

                temp = b(bpos);
                setconj(temp);
                temp *= a(apos);
                setconj(temp);

                result += temp;

                ++npos;
		++apos;
		++bpos;
	    }

            else if ( ( nelm <= aelm ) && ( nelm <= belm )  )
	    {
		++npos;
	    }

            else if ( ( aelm <= nelm ) && ( aelm <= belm )  )
	    {
		++apos;
	    }

	    else
	    {
                ++bpos;
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &indexedtwoProduct(T &result, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b)
{
    if ( a.infsize() || b.infsize() )
    {
        NiceThrow("Indexed FuncVector inner product not supported.");

        return result;
    }

    int nsize = n.size();
    int asize = a.size(); //indsize();
    int bsize = b.size(); //indsize();

    T temp;

    setzero(result);
    setzero(temp);

    if ( nsize && asize && bsize )
    {
        int npos = 0;
	int apos = 0;
	int bpos = 0;

        int nelm;
	int aelm;
	int belm;

	while ( ( npos < nsize ) && ( apos < asize ) && ( bpos < bsize ) )
	{
            nelm = n.v(npos);
            aelm = apos; //a.ind(apos);
	    belm = bpos; //b.ind(bpos);

	    if ( ( aelm == nelm ) && ( belm == nelm ) )
	    {
                temp = b(bpos);
                rightmult(a(apos),temp);

                result += temp;

                ++npos;
		++apos;
		++bpos;
	    }

            else if ( ( nelm <= aelm ) && ( nelm <= belm )  )
	    {
		++npos;
	    }

            else if ( ( aelm <= nelm ) && ( aelm <= belm )  )
	    {
		++apos;
	    }

	    else
	    {
                ++bpos;
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &indexedinnerProductRevConj(T &result, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b)
{
    if ( a.infsize() || b.infsize() )
    {
        NiceThrow("Indexed FuncVector inner product not supported.");

        return result;
    }

    int nsize = n.size();
    int asize = a.size(); //indsize();
    int bsize = b.size(); //indsize();

    T temp;

    setzero(result);
    setzero(temp);

    if ( nsize && asize && bsize )
    {
        int npos = 0;
	int apos = 0;
	int bpos = 0;

        int nelm;
	int aelm;
	int belm;

	while ( ( npos < nsize ) && ( apos < asize ) && ( bpos < bsize ) )
	{
            nelm = n.v(npos);
            aelm = apos; //a.ind(apos);
	    belm = bpos; //b.ind(bpos);

	    if ( ( aelm == nelm ) && ( belm == nelm ) )
	    {
                temp = b(bpos);
                setconj(temp);
                rightmult(a(apos),temp);

                result += temp;

                ++npos;
		++apos;
		++bpos;
	    }

            else if ( ( nelm <= aelm ) && ( nelm <= belm )  )
	    {
		++npos;
	    }

            else if ( ( aelm <= nelm ) && ( aelm <= belm )  )
	    {
		++apos;
	    }

	    else
	    {
                ++bpos;
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &indexedinnerProductScaled(T &result, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale)
{
    if ( a.infsize() || b.infsize() )
    {
        NiceThrow("Indexed FuncVector inner product not supported.");

        return result;
    }

    // Basically just a four product with conjugation on a and scale included twice

    int nsize = n.size();
    int asize = a.size(); //indsize();
    int bsize = b.size(); //indsize();
    int csize = scale.size(); //indsize();
    int dsize = scale.size(); //indsize();

    T tempa;
    T tempb;

    setzero(result);
    setzero(tempa);
    setzero(tempb);

    if ( nsize && asize && bsize && csize && dsize )
    {
        int npos = 0;
	int apos = 0;
	int bpos = 0;
	int cpos = 0;
        int dpos = 0;
        int nelm;
	int aelm;
	int belm;
	int celm;
        int delm;

	while ( ( npos < nsize ) && ( apos < asize ) && ( bpos < bsize ) && ( cpos < csize ) && ( dpos < dsize ) )
	{
            nelm = n.v(npos);
            aelm = apos; //a.ind(apos);
	    belm = bpos; //b.ind(bpos);
	    celm = cpos; //scale.ind(cpos);
	    delm = dpos; //scale.ind(dpos);

	    if ( ( aelm == nelm ) && ( belm == nelm ) && ( celm == nelm ) && ( delm == nelm ) )
	    {
                tempa  = a(apos);
                tempa /= scale(cpos);

                tempb  = b(bpos);
                tempb /= scale(dpos);

                setconj(tempa);
                //setconj(tempb);
                rightmult(tempa,tempb);

                result += tempb;

                ++npos;
		++apos;
		++bpos;
		++cpos;
                ++dpos;
	    }

            else if ( ( nelm <= aelm ) && ( nelm <= belm ) && ( nelm <= celm ) && ( nelm <= delm ) )
	    {
		++npos;
	    }

            else if ( ( aelm <= nelm ) && ( aelm <= belm ) && ( aelm <= celm ) && ( aelm <= delm ) )
	    {
		++apos;
	    }

            else if ( ( belm <= nelm ) && ( belm <= aelm ) && ( belm <= celm ) && ( belm <= delm ) )
	    {
		++bpos;
	    }

            else if ( ( celm <= nelm ) && ( celm <= aelm ) && ( celm <= belm ) && ( celm <= delm ) )
	    {
		++cpos;
	    }

	    else
	    {
                ++dpos;
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &indexedtwoProductScaled(T &result, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale)
{
    if ( a.infsize() || b.infsize() )
    {
        NiceThrow("Indexed FuncVector inner product not supported.");

        return result;
    }

    // Basically just a four product with conjugation on a and scale included twice

    int nsize = n.size();
    int asize = a.size(); //indsize();
    int bsize = b.size(); //indsize();
    int csize = scale.size(); //indsize();
    int dsize = scale.size(); //indsize();

    T tempa;
    T tempb;

    setzero(result);
    setzero(tempa);
    setzero(tempb);

    if ( nsize && asize && bsize && csize && dsize )
    {
        int npos = 0;
	int apos = 0;
	int bpos = 0;
	int cpos = 0;
        int dpos = 0;
        int nelm;
	int aelm;
	int belm;
	int celm;
        int delm;

	while ( ( npos < nsize ) && ( apos < asize ) && ( bpos < bsize ) && ( cpos < csize ) && ( dpos < dsize ) )
	{
            nelm = n.v(npos);
            aelm = apos; //a.ind(apos);
	    belm = bpos; //b.ind(bpos);
	    celm = cpos; //scale.ind(cpos);
	    delm = dpos; //scale.ind(dpos);

	    if ( ( aelm == nelm ) && ( belm == nelm ) && ( celm == nelm ) && ( delm == nelm ) )
	    {
                tempa  = a(apos);
                tempa /= scale(cpos);

                tempb  = b(bpos);
                tempb /= scale(dpos);

                //setconj(tempa);
                //setconj(tempb);
                rightmult(tempa,tempb);

                result += tempb;

                ++npos;
		++apos;
		++bpos;
		++cpos;
                ++dpos;
	    }

            else if ( ( nelm <= aelm ) && ( nelm <= belm ) && ( nelm <= celm ) && ( nelm <= delm ) )
	    {
		++npos;
	    }

            else if ( ( aelm <= nelm ) && ( aelm <= belm ) && ( aelm <= celm ) && ( aelm <= delm ) )
	    {
		++apos;
	    }

            else if ( ( belm <= nelm ) && ( belm <= aelm ) && ( belm <= celm ) && ( belm <= delm ) )
	    {
		++bpos;
	    }

            else if ( ( celm <= nelm ) && ( celm <= aelm ) && ( celm <= belm ) && ( celm <= delm ) )
	    {
		++cpos;
	    }

	    else
	    {
                ++dpos;
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &indexedinnerProductScaledRevConj(T &result, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale)
{
    if ( a.infsize() || b.infsize() )
    {
        NiceThrow("Indexed FuncVector inner product not supported.");

        return result;
    }

    // Basically just a four product with conjugation on a and scale included twice

    int nsize = n.size();
    int asize = a.size(); //indsize();
    int bsize = b.size(); //indsize();
    int csize = scale.size(); //indsize();
    int dsize = scale.size(); //indsize();

    T tempa;
    T tempb;

    setzero(result);
    setzero(tempa);
    setzero(tempb);

    if ( nsize && asize && bsize && csize && dsize )
    {
        int npos = 0;
	int apos = 0;
	int bpos = 0;
	int cpos = 0;
        int dpos = 0;
        int nelm;
	int aelm;
	int belm;
	int celm;
        int delm;

	while ( ( npos < nsize ) && ( apos < asize ) && ( bpos < bsize ) && ( cpos < csize ) && ( dpos < dsize ) )
	{
            nelm = n.v(npos);
            aelm = apos; //a.ind(apos);
	    belm = bpos; //b.ind(bpos);
	    celm = cpos; //scale.ind(cpos);
	    delm = dpos; //scale.ind(dpos);

	    if ( ( aelm == nelm ) && ( belm == nelm ) && ( celm == nelm ) && ( delm == nelm ) )
	    {
                tempa  = a(apos);
                tempa /= scale(cpos);

                tempb  = b(bpos);
                tempb /= scale(dpos);

                //setconj(tempa);
                setconj(tempb);
                rightmult(tempa,tempb);

                result += tempb;

                ++npos;
		++apos;
		++bpos;
		++cpos;
                ++dpos;
	    }

            else if ( ( nelm <= aelm ) && ( nelm <= belm ) && ( nelm <= celm ) && ( nelm <= delm ) )
	    {
		++npos;
	    }

            else if ( ( aelm <= nelm ) && ( aelm <= belm ) && ( aelm <= celm ) && ( aelm <= delm ) )
	    {
		++apos;
	    }

            else if ( ( belm <= nelm ) && ( belm <= aelm ) && ( belm <= celm ) && ( belm <= delm ) )
	    {
		++bpos;
	    }

            else if ( ( celm <= nelm ) && ( celm <= aelm ) && ( celm <= belm ) && ( celm <= delm ) )
	    {
		++cpos;
	    }

	    else
	    {
                ++dpos;
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &indexedinnerProductLeftScaled(T &result, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale)
{
    if ( a.infsize() || b.infsize() )
    {
        NiceThrow("Indexed FuncVector inner product not supported.");

        return result;
    }

    // Basically just a four product with conjugation on a and scale included twice

    int nsize = n.size();
    int asize = a.size(); //indsize();
    int bsize = b.size(); //indsize();
    int csize = scale.size(); //indsize();
    int dsize = scale.size(); //indsize();

    T tempa;
    T tempb;

    setzero(result);
    setzero(tempa);
    setzero(tempb);

    if ( nsize && asize && bsize && csize && dsize )
    {
        int npos = 0;
	int apos = 0;
	int bpos = 0;
	int cpos = 0;
        int dpos = 0;
        int nelm;
	int aelm;
	int belm;
	int celm;
        int delm;

	while ( ( npos < nsize ) && ( apos < asize ) && ( bpos < bsize ) && ( cpos < csize ) && ( dpos < dsize ) )
	{
            nelm = n.v(npos);
            aelm = apos; //a.ind(apos);
	    belm = bpos; //b.ind(bpos);
	    celm = cpos; //scale.ind(cpos);
	    delm = dpos; //scale.ind(dpos);

	    if ( ( aelm == nelm ) && ( belm == nelm ) && ( celm == nelm ) && ( delm == nelm ) )
	    {
                tempa  = a(apos);
                tempa /= scale(cpos);

                tempb  = b(bpos);
                //tempb /= scale(dpos); - only half scaling

                setconj(tempa);
                //setconj(tempb);
                rightmult(tempa,tempb);

                result += tempb;

                ++npos;
		++apos;
		++bpos;
		++cpos;
                ++dpos;
	    }

            else if ( ( nelm <= aelm ) && ( nelm <= belm ) && ( nelm <= celm ) && ( nelm <= delm ) )
	    {
		++npos;
	    }

            else if ( ( aelm <= nelm ) && ( aelm <= belm ) && ( aelm <= celm ) && ( aelm <= delm ) )
	    {
		++apos;
	    }

            else if ( ( belm <= nelm ) && ( belm <= aelm ) && ( belm <= celm ) && ( belm <= delm ) )
	    {
		++bpos;
	    }

            else if ( ( celm <= nelm ) && ( celm <= aelm ) && ( celm <= belm ) && ( celm <= delm ) )
	    {
		++cpos;
	    }

	    else
	    {
                ++dpos;
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &indexedtwoProductLeftScaled(T &result, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale)
{
    if ( a.infsize() || b.infsize() )
    {
        NiceThrow("Indexed FuncVector inner product not supported.");

        return result;
    }

    // Basically just a four product with conjugation on a and scale included twice

    int nsize = n.size();
    int asize = a.size(); //indsize();
    int bsize = b.size(); //indsize();
    int csize = scale.size(); //indsize();
    int dsize = scale.size(); //indsize();

    T tempa;
    T tempb;

    setzero(result);
    setzero(tempa);
    setzero(tempb);

    if ( nsize && asize && bsize && csize && dsize )
    {
        int npos = 0;
	int apos = 0;
	int bpos = 0;
	int cpos = 0;
        int dpos = 0;
        int nelm;
	int aelm;
	int belm;
	int celm;
        int delm;

	while ( ( npos < nsize ) && ( apos < asize ) && ( bpos < bsize ) && ( cpos < csize ) && ( dpos < dsize ) )
	{
            nelm = n.v(npos);
            aelm = apos; //a.ind(apos);
	    belm = bpos; //b.ind(bpos);
	    celm = cpos; //scale.ind(cpos);
	    delm = dpos; //scale.ind(dpos);

	    if ( ( aelm == nelm ) && ( belm == nelm ) && ( celm == nelm ) && ( delm == nelm ) )
	    {
                tempa  = a(apos);
                tempa /= scale(cpos);

                tempb  = b(bpos);
                //tempb /= scale(dpos); - only half scaling

                //setconj(tempa);
                //setconj(tempb);
                rightmult(tempa,tempb);

                result += tempb;

                ++npos;
		++apos;
		++bpos;
		++cpos;
                ++dpos;
	    }

            else if ( ( nelm <= aelm ) && ( nelm <= belm ) && ( nelm <= celm ) && ( nelm <= delm ) )
	    {
		++npos;
	    }

            else if ( ( aelm <= nelm ) && ( aelm <= belm ) && ( aelm <= celm ) && ( aelm <= delm ) )
	    {
		++apos;
	    }

            else if ( ( belm <= nelm ) && ( belm <= aelm ) && ( belm <= celm ) && ( belm <= delm ) )
	    {
		++bpos;
	    }

            else if ( ( celm <= nelm ) && ( celm <= aelm ) && ( celm <= belm ) && ( celm <= delm ) )
	    {
		++cpos;
	    }

	    else
	    {
                ++dpos;
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &indexedinnerProductLeftScaledRevConj(T &result, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale)
{
    if ( a.infsize() || b.infsize() )
    {
        NiceThrow("Indexed FuncVector inner product not supported.");

        return result;
    }

    // Basically just a four product with conjugation on a and scale included twice

    int nsize = n.size();
    int asize = a.size(); //indsize();
    int bsize = b.size(); //indsize();
    int csize = scale.size(); //indsize();
    int dsize = scale.size(); //indsize();

    T tempa;
    T tempb;

    setzero(result);
    setzero(tempa);
    setzero(tempb);

    if ( nsize && asize && bsize && csize && dsize )
    {
        int npos = 0;
	int apos = 0;
	int bpos = 0;
	int cpos = 0;
        int dpos = 0;
        int nelm;
	int aelm;
	int belm;
	int celm;
        int delm;

	while ( ( npos < nsize ) && ( apos < asize ) && ( bpos < bsize ) && ( cpos < csize ) && ( dpos < dsize ) )
	{
            nelm = n.v(npos);
            aelm = apos; //a.ind(apos);
	    belm = bpos; //b.ind(bpos);
	    celm = cpos; //scale.ind(cpos);
	    delm = dpos; //scale.ind(dpos);

	    if ( ( aelm == nelm ) && ( belm == nelm ) && ( celm == nelm ) && ( delm == nelm ) )
	    {
                tempa  = a(apos);
                tempa /= scale(cpos);

                tempb = b(bpos);
                //tempb /= scale(dpos); - only half scaling

                //setconj(tempa);
                setconj(tempb);
                rightmult(tempa,tempb);

                result += tempb;

                ++npos;
		++apos;
		++bpos;
		++cpos;
                ++dpos;
	    }

            else if ( ( nelm <= aelm ) && ( nelm <= belm ) && ( nelm <= celm ) && ( nelm <= delm ) )
	    {
		++npos;
	    }

            else if ( ( aelm <= nelm ) && ( aelm <= belm ) && ( aelm <= celm ) && ( aelm <= delm ) )
	    {
		++apos;
	    }

            else if ( ( belm <= nelm ) && ( belm <= aelm ) && ( belm <= celm ) && ( belm <= delm ) )
	    {
		++bpos;
	    }

            else if ( ( celm <= nelm ) && ( celm <= aelm ) && ( celm <= belm ) && ( celm <= delm ) )
	    {
		++cpos;
	    }

	    else
	    {
                ++dpos;
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &indexedinnerProductRightScaled(T &result, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale)
{
    if ( a.infsize() || b.infsize() )
    {
        NiceThrow("Indexed FuncVector inner product not supported.");

        return result;
    }

    // Basically just a four product with conjugation on a and scale included twice

    int nsize = n.size();
    int asize = a.size(); //indsize();
    int bsize = b.size(); //indsize();
    int csize = scale.size(); //indsize();
    int dsize = scale.size(); //indsize();

    T tempa;
    T tempb;

    setzero(result);
    setzero(tempa);
    setzero(tempb);

    if ( nsize && asize && bsize && csize && dsize )
    {
        int npos = 0;
	int apos = 0;
	int bpos = 0;
	int cpos = 0;
        int dpos = 0;
        int nelm;
	int aelm;
	int belm;
	int celm;
        int delm;

	while ( ( npos < nsize ) && ( apos < asize ) && ( bpos < bsize ) && ( cpos < csize ) && ( dpos < dsize ) )
	{
            nelm = n.v(npos);
            aelm = apos; //a.ind(apos);
	    belm = bpos; //b.ind(bpos);
	    celm = cpos; //scale.ind(cpos);
	    delm = dpos; //scale.ind(dpos);

	    if ( ( aelm == nelm ) && ( belm == nelm ) && ( celm == nelm ) && ( delm == nelm ) )
	    {
                tempa  = a(apos);
                //tempa /= scale(cpos); - only half scaling

                tempb  = b(bpos);
                tempb /= scale(dpos);

                setconj(tempa);
                //setconj(tempb);
                rightmult(tempa,tempb);

                result += tempb;

                ++npos;
		++apos;
		++bpos;
		++cpos;
                ++dpos;
	    }

            else if ( ( nelm <= aelm ) && ( nelm <= belm ) && ( nelm <= celm ) && ( nelm <= delm ) )
	    {
		++npos;
	    }

            else if ( ( aelm <= nelm ) && ( aelm <= belm ) && ( aelm <= celm ) && ( aelm <= delm ) )
	    {
		++apos;
	    }

            else if ( ( belm <= nelm ) && ( belm <= aelm ) && ( belm <= celm ) && ( belm <= delm ) )
	    {
		++bpos;
	    }

            else if ( ( celm <= nelm ) && ( celm <= aelm ) && ( celm <= belm ) && ( celm <= delm ) )
	    {
		++cpos;
	    }

	    else
	    {
                ++dpos;
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &indexedtwoProductRightScaled(T &result, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale)
{
    if ( a.infsize() || b.infsize() )
    {
        NiceThrow("Indexed FuncVector inner product not supported.");

        return result;
    }

    // Basically just a four product with conjugation on a and scale included twice

    int nsize = n.size();
    int asize = a.size(); //indsize();
    int bsize = b.size(); //indsize();
    int csize = scale.size(); //indsize();
    int dsize = scale.size(); //indsize();

    T tempa;
    T tempb;

    setzero(result);
    setzero(tempa);
    setzero(tempb);

    if ( nsize && asize && bsize && csize && dsize )
    {
        int npos = 0;
	int apos = 0;
	int bpos = 0;
	int cpos = 0;
        int dpos = 0;
        int nelm;
	int aelm;
	int belm;
	int celm;
        int delm;

	while ( ( npos < nsize ) && ( apos < asize ) && ( bpos < bsize ) && ( cpos < csize ) && ( dpos < dsize ) )
	{
            nelm = n.v(npos);
            aelm = apos; //a.ind(apos);
	    belm = bpos; //b.ind(bpos);
	    celm = cpos; //scale.ind(cpos);
	    delm = dpos; //scale.ind(dpos);

	    if ( ( aelm == nelm ) && ( belm == nelm ) && ( celm == nelm ) && ( delm == nelm ) )
	    {
                tempa  = a(apos);
                //tempa /= scale(cpos); - only half scaling

                tempb  = b(bpos);
                tempb /= scale(dpos);

                //setconj(tempa);
                //setconj(tempb);
                rightmult(tempa,tempb);

                result += tempb;

                ++npos;
		++apos;
		++bpos;
		++cpos;
                ++dpos;
	    }

            else if ( ( nelm <= aelm ) && ( nelm <= belm ) && ( nelm <= celm ) && ( nelm <= delm ) )
	    {
		++npos;
	    }

            else if ( ( aelm <= nelm ) && ( aelm <= belm ) && ( aelm <= celm ) && ( aelm <= delm ) )
	    {
		++apos;
	    }

            else if ( ( belm <= nelm ) && ( belm <= aelm ) && ( belm <= celm ) && ( belm <= delm ) )
	    {
		++bpos;
	    }

            else if ( ( celm <= nelm ) && ( celm <= aelm ) && ( celm <= belm ) && ( celm <= delm ) )
	    {
		++cpos;
	    }

	    else
	    {
                ++dpos;
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &indexedinnerProductRightScaledRevConj(T &result, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b, const Vector<T> &scale)
{
    if ( a.infsize() || b.infsize() )
    {
        NiceThrow("Indexed FuncVector inner product not supported.");

        return result;
    }

    // Basically just a four product with conjugation on a and scale included twice

    int nsize = n.size();
    int asize = a.size(); //indsize();
    int bsize = b.size(); //indsize();
    int csize = scale.size(); //indsize();
    int dsize = scale.size(); //indsize();

    T tempa;
    T tempb;

    setzero(result);
    setzero(tempa);
    setzero(tempb);

    if ( nsize && asize && bsize && csize && dsize )
    {
        int npos = 0;
	int apos = 0;
	int bpos = 0;
	int cpos = 0;
        int dpos = 0;
        int nelm;
	int aelm;
	int belm;
	int celm;
        int delm;

	while ( ( npos < nsize ) && ( apos < asize ) && ( bpos < bsize ) && ( cpos < csize ) && ( dpos < dsize ) )
	{
            nelm = n.v(npos);
            aelm = apos; //a.ind(apos);
	    belm = bpos; //b.ind(bpos);
	    celm = cpos; //scale.ind(cpos);
	    delm = dpos; //scale.ind(dpos);

	    if ( ( aelm == nelm ) && ( belm == nelm ) && ( celm == nelm ) && ( delm == nelm ) )
	    {
                tempa  = a(apos);
                //tempa /= scale(cpos); - only half scaling

                tempb  = b(bpos);
                tempb /= scale(dpos);

                //setconj(tempa);
                setconj(tempb);
                rightmult(tempa,tempb);

                result += tempb;

                ++npos;
		++apos;
		++bpos;
		++cpos;
                ++dpos;
	    }

            else if ( ( nelm <= aelm ) && ( nelm <= belm ) && ( nelm <= celm ) && ( nelm <= delm ) )
	    {
		++npos;
	    }

            else if ( ( aelm <= nelm ) && ( aelm <= belm ) && ( aelm <= celm ) && ( aelm <= delm ) )
            {
		++apos;
	    }

            else if ( ( belm <= nelm ) && ( belm <= aelm ) && ( belm <= celm ) && ( belm <= delm ) )
	    {
		++bpos;
	    }

            else if ( ( celm <= nelm ) && ( celm <= aelm ) && ( celm <= belm ) && ( celm <= delm ) )
	    {
		++cpos;
	    }

	    else
	    {
                ++dpos;
	    }
	}
    }

    return postProInnerProd(result);
}

//----------------------------------------------

template <class T>
double &threeProductAssumeReal(double &res, const Vector<T> &a, const Vector<T> &b, const Vector<T> &c)
{
    if ( a.infsize() || b.infsize() || c.infsize() )
    {
        NiceAssert( a.infsize() && b.infsize() && c.infsize() );

        return a.inner3Real(res,b,c);
    }

    int dim = a.size();

    NiceAssert( b.size() == dim );
    NiceAssert( c.size() == dim );

    res = 0;

    //if ( dim )
    {
        for ( int i = 0 ; i < dim ; ++i )
        {
            res += (double) (a(i)*b(i)*c(i));
	}
    }

    return res;
}

template <> inline double &threeProductAssumeReal(double &res, const Vector<double> &a, const Vector<double> &b, const Vector<double> &c)
{
    return threeProduct(res,a,b,c);
}

template <class T>
T &threeProduct(T &result, const Vector<T> &a, const Vector<T> &b, const Vector<T> &c)
{
    if ( a.infsize() || b.infsize() || c.infsize() )
    {
        NiceAssert( a.infsize() && b.infsize() && c.infsize() );

        return a.inner3(result,b,c);
    }

    NiceAssert( a.size() == b.size() );
    NiceAssert( a.size() == c.size() );

    setzero(result);

    if ( a.size() )
    {
        result = (a(0)*b(0))*c(0);

        //if ( a.size() > 1 )
	{
            for ( int i = 1 ; i < a.size() ; ++i )
	    {
                result += (a(i)*b(i))*c(i);
	    }
	}
    }

    return postProInnerProd(result);
}

template <> inline double &threeProduct(double &res, const Vector<double> &a, const Vector<double> &b, const Vector<double> &c)
{
    int size = a.size();

    res = 0;

    NiceAssert( size == b.size() );
    NiceAssert( size == c.size() );

    if ( size && a.contiguous() && b.contiguous() && c.contiguous() )
    {
        res = fastthreeProduct(&a(0),&b(0),&c(0),size);
    }

    else if ( size )
    {
        res = fastthreeProduct(a.unsafeccontent(),a.unsafepivot(),a.unsafeib(),a.unsafeis(),b.unsafeccontent(),b.unsafepivot(),b.unsafeib(),b.unsafeis(),c.unsafeccontent(),c.unsafepivot(),c.unsafeib(),c.unsafeis(),size);
    }

//    else
//    {
//        for ( int i = 0 ; i < size ; ++i )
//        {
//            res += a(i)*b(i)*c(i);
//        }
//    }

    return res;
}

template <class T>
T &threeProductScaled(T &result, const Vector<T> &a, const Vector<T> &b, const Vector<T> &c, const Vector<T> &scale)
{
    if ( a.infsize() || b.infsize() || c.infsize() )
    {
        NiceThrow("Scaled 3-FuncVector-product not supported");

        return result;
    }

    NiceAssert( a.size() == b.size() );
    NiceAssert( a.size() == c.size() );

    setzero(result);

    if ( a.size() )
    {
        result = ((a(0)/scale(0))*(b(0)/scale(0)))*(c(0)/scale(0));

        //if ( a.size() > 1 )
	{
            for ( int i = 1 ; i < a.size() ; ++i )
	    {
                result += ((a(i)/scale(i))*(b(i)/scale(i)))*(c(i)/scale(i));
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &indexedthreeProduct(T &result, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b, const Vector<T> &c)
{
    if ( a.infsize() || b.infsize() || c.infsize() )
    {
        NiceThrow("Indexed FuncVector three-product not supported");

        return result;
    }

    int nsize = n.size();
    int asize = a.size(); //indsize();
    int bsize = b.size(); //indsize();
    int csize = c.size(); //indsize();

    setzero(result);

    if ( nsize && asize && bsize && csize )
    {
        int npos = 0;
	int apos = 0;
	int bpos = 0;
	int cpos = 0;

        int nelm;
	int aelm;
	int belm;
	int celm;

	while ( ( npos < nsize ) && ( apos < asize ) && ( bpos < bsize ) && ( cpos < csize ) )
	{
            nelm = n.v(npos);
            aelm = apos; //a.ind(apos);
	    belm = bpos; //b.ind(bpos);
	    celm = cpos; //c.ind(cpos);

	    if ( ( aelm == nelm ) && ( belm == nelm ) && ( celm == nelm ) )
	    {
                // NB: we process the product in pairs, so if T == gentype
                //     then ("a"*"a")*("b"*"b") will "correctly" evaluate
                //     to 1, not error (ie symbolic multiplication).

                result += (a(apos)*b(bpos))*c(cpos);

                ++npos;
		++apos;
		++bpos;
		++cpos;
	    }

            else if ( ( nelm <= aelm ) && ( nelm <= belm ) && ( nelm <= celm ) )
	    {
		++npos;
	    }

            else if ( ( aelm <= nelm ) && ( aelm <= belm ) && ( aelm <= celm ) )
	    {
		++apos;
	    }

            else if ( ( belm <= nelm ) && ( belm <= aelm ) && ( belm <= celm ) )
	    {
		++bpos;
	    }

            else
	    {
		++cpos;
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &indexedthreeProductScaled(T &result, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b, const Vector<T> &c, const Vector<T> &scale)
{
    if ( a.infsize() || b.infsize() || c.infsize() )
    {
        NiceThrow("Indexed FuncVector three-product not supported");

        return result;
    }

    int nsize = n.size();
    int asize = a.size(); //indsize();
    int bsize = b.size(); //indsize();
    int csize = c.size(); //indsize();

    setzero(result);

    if ( nsize && asize && bsize && csize )
    {
        int npos = 0;
	int apos = 0;
	int bpos = 0;
	int cpos = 0;

        int nelm;
	int aelm;
	int belm;
	int celm;

	while ( ( npos < nsize ) && ( apos < asize ) && ( bpos < bsize ) && ( cpos < csize ) )
	{
            nelm = n.v(npos);
            aelm = apos; //a.ind(apos);
	    belm = bpos; //b.ind(bpos);
	    celm = cpos; //c.ind(cpos);

	    if ( ( aelm == nelm ) && ( belm == nelm ) && ( celm == nelm ) )
	    {
                // NB: we process the product in pairs, so if T == gentype
                //     then ("a"*"a")*("b"*"b") will "correctly" evaluate
                //     to 1, not error (ie symbolic multiplication).

                result += ((a(apos)/scale(apos))*(b(bpos)/scale(apos)))*(c(cpos)/scale(cpos));

                ++npos;
		++apos;
		++bpos;
		++cpos;
	    }

            else if ( ( nelm <= aelm ) && ( nelm <= belm ) && ( nelm <= celm ) )
	    {
		++npos;
	    }

            else if ( ( aelm <= nelm ) && ( aelm <= belm ) && ( aelm <= celm ) )
	    {
		++apos;
	    }

            else if ( ( belm <= nelm ) && ( belm <= aelm ) && ( belm <= celm ) )
	    {
		++bpos;
	    }

            else
	    {
		++cpos;
	    }
	}
    }

    return postProInnerProd(result);
}

//----------------------------------------------

template <class T>
double &fourProductAssumeReal(double &res, const Vector<T> &a, const Vector<T> &b, const Vector<T> &c, const Vector<T> &d)
{
    if ( a.infsize() || b.infsize() || c.infsize() || d.infsize() )
    {
        NiceAssert( a.infsize() && b.infsize() && c.infsize() && d.infsize() );

        return a.inner4Real(res,b,c,d);
    }

    int dim = a.size();

    NiceAssert( b.size() == dim );
    NiceAssert( c.size() == dim );
    NiceAssert( d.size() == dim );

    res = 0;

    //if ( dim )
    {
        for ( int i = 0 ; i < dim ; ++i )
        {
            scaladd(res,a(i),b(i),c(i),d(i));
	}
    }

    return res;
}

template <> inline double &fourProductAssumeReal(double &res, const Vector<double> &a, const Vector<double> &b, const Vector<double> &c, const Vector<double> &d)
{
    return fourProduct(res,a,b,c,d);
}

template <class T>
T &fourProduct(T &result, const Vector<T> &a, const Vector<T> &b, const Vector<T> &c, const Vector<T> &d)
{
    if ( a.infsize() || b.infsize() || c.infsize() || d.infsize() )
    {
        NiceAssert( a.infsize() && b.infsize() && c.infsize() && d.infsize() );

        return a.inner4(result,b,c,d);
    }

    NiceAssert( a.size() == b.size() );
    NiceAssert( a.size() == c.size() );
    NiceAssert( a.size() == d.size() );

    setzero(result);

    if ( a.size() )
    {
        result = (a(0)*b(0))*(c(0)*d(0));

        //if ( a.size() > 1 )
	{
            for ( int i = 1 ; i < a.size() ; ++i )
	    {
                result += (a(i)*b(i))*(c(i)*d(i));
	    }
	}
    }

    return postProInnerProd(result);
}

template <> inline double &fourProduct(double &res, const Vector<double> &a, const Vector<double> &b, const Vector<double> &c, const Vector<double> &d)
{
    int size = a.size();

    res = 0;

    NiceAssert( size == b.size() );
    NiceAssert( size == c.size() );
    NiceAssert( size == d.size() );

    if ( size && a.contiguous() && b.contiguous() && c.contiguous() && d.contiguous() )
    {
        res = fastfourProduct(&a(0),&b(0),&c(0),&d(0),size);
    }

    else if ( size )
    {
        res = fastfourProduct(a.unsafeccontent(),a.unsafepivot(),a.unsafeib(),a.unsafeis(),b.unsafeccontent(),b.unsafepivot(),b.unsafeib(),b.unsafeis(),c.unsafeccontent(),c.unsafepivot(),c.unsafeib(),c.unsafeis(),d.unsafeccontent(),d.unsafepivot(),d.unsafeib(),d.unsafeis(),size);
    }

//    else
//    {
//        for ( int i = 0 ; i < size ; ++i )
//        {
//            res += a(i)*b(i)*c(i)*d(i);
//        }
//    }

    return res;
}

template <class T>
T &fourProductScaled(T &result, const Vector<T> &a, const Vector<T> &b, const Vector<T> &c, const Vector<T> &d, const Vector<T> &scale)
{
    if ( a.infsize() || b.infsize() || c.infsize() || d.infsize() )
    {
        NiceThrow("Scaled 4-FuncVector-product not supported");

        return a.inner4(result,b,c,d);
    }

    NiceAssert( a.size() == b.size() );
    NiceAssert( a.size() == c.size() );
    NiceAssert( a.size() == d.size() );

    setzero(result);

    if ( a.size() )
    {
        result = ((a(0)/scale(0))*(b(0)/scale(0)))*((c(0)/scale(0))*(d(0)/scale(0)));

        //if ( a.size() > 1 )
	{
            for ( int i = 1 ; i < a.size() ; ++i )
	    {
                result += ((a(i)/scale(i))*(b(i)/scale(i)))*((c(i)/scale(i))*(d(i)/scale(i)));
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &indexedfourProduct(T &result, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b, const Vector<T> &c, const Vector<T> &d)
{
    if ( a.infsize() || b.infsize() || c.infsize() || d.infsize() )
    {
        NiceThrow("Indexed FuncVector four-product not supported.");

        return result;
    }

    int nsize = n.size();
    int asize = a.size(); //indsize();
    int bsize = b.size(); //indsize();
    int csize = c.size(); //indsize();
    int dsize = d.size(); //indsize();

    setzero(result);

    if ( nsize && asize && bsize && csize && dsize )
    {
        int npos = 0;
	int apos = 0;
	int bpos = 0;
	int cpos = 0;
	int dpos = 0;

        int nelm;
	int aelm;
	int belm;
	int celm;
        int delm;

	while ( ( npos < nsize ) && ( apos < asize ) && ( bpos < bsize ) && ( cpos < csize ) && ( dpos < dsize ) )
	{
            nelm = n.v(npos);
            aelm = apos; //a.ind(apos);
	    belm = bpos; //b.ind(bpos);
	    celm = cpos; //c.ind(cpos);
	    delm = dpos; //d.ind(dpos);

	    if ( ( aelm == nelm ) && ( belm == nelm ) && ( celm == nelm ) && ( delm == nelm ) )
	    {
                // NB: we process the product in pairs, so if T == gentype
                //     then ("a"*"a")*("b"*"b") will "correctly" evaluate
                //     to 1, not error (ie symbolic multiplication).

                result += (a(apos)*b(bpos))*(c(cpos)*d(dpos));

                ++npos;
		++apos;
		++bpos;
		++cpos;
                ++dpos;
	    }

            else if ( ( nelm <= aelm ) && ( nelm <= belm ) && ( nelm <= celm ) && ( nelm <= delm ) )
	    {
		++npos;
	    }

            else if ( ( aelm <= nelm ) && ( aelm <= belm ) && ( aelm <= celm ) && ( aelm <= delm ) )
	    {
		++apos;
	    }

            else if ( ( belm <= nelm ) && ( belm <= aelm ) && ( belm <= celm ) && ( belm <= delm ) )
	    {
		++bpos;
	    }

            else if ( ( celm <= nelm ) && ( celm <= aelm ) && ( celm <= belm ) && ( celm <= delm ) )
	    {
		++cpos;
	    }

	    else
	    {
                ++dpos;
	    }
	}
    }

    return postProInnerProd(result);
}

template <class T>
T &indexedfourProductScaled(T &result, const Vector<int> &n, const Vector<T> &a, const Vector<T> &b, const Vector<T> &c, const Vector<T> &d, const Vector<T> &scale)
{
    if ( a.infsize() || b.infsize() || c.infsize() || d.infsize() )
    {
        NiceThrow("Indexed FuncVector four-product not supported.");

        return result;
    }

    int nsize = n.size();
    int asize = a.size(); //indsize();
    int bsize = b.size(); //indsize();
    int csize = c.size(); //indsize();
    int dsize = d.size(); //indsize();

    setzero(result);

    if ( nsize && asize && bsize && csize && dsize )
    {
        int npos = 0;
	int apos = 0;
	int bpos = 0;
	int cpos = 0;
	int dpos = 0;

        int nelm;
	int aelm;
	int belm;
	int celm;
        int delm;

	while ( ( npos < nsize ) && ( apos < asize ) && ( bpos < bsize ) && ( cpos < csize ) && ( dpos < dsize ) )
	{
            nelm = n.v(npos);
            aelm = apos; //a.ind(apos);
	    belm = bpos; //b.ind(bpos);
	    celm = cpos; //c.ind(cpos);
	    delm = dpos; //d.ind(dpos);

	    if ( ( aelm == nelm ) && ( belm == nelm ) && ( celm == nelm ) && ( delm == nelm ) )
	    {
                // NB: we process the product in pairs, so if T == gentype
                //     then ("a"*"a")*("b"*"b") will "correctly" evaluate
                //     to 1, not error (ie symbolic multiplication).

                result += ((a(apos)/scale(apos))*(b(bpos)/scale(bpos)))*((c(cpos)/scale(cpos))*(d(dpos)/scale(dpos)));

                ++npos;
		++apos;
		++bpos;
		++cpos;
                ++dpos;
	    }

            else if ( ( nelm <= aelm ) && ( nelm <= belm ) && ( nelm <= celm ) && ( nelm <= delm ) )
	    {
		++npos;
	    }

            else if ( ( aelm <= nelm ) && ( aelm <= belm ) && ( aelm <= celm ) && ( aelm <= delm ) )
	    {
		++apos;
	    }

            else if ( ( belm <= nelm ) && ( belm <= aelm ) && ( belm <= celm ) && ( belm <= delm ) )
	    {
		++bpos;
	    }

            else if ( ( celm <= nelm ) && ( celm <= aelm ) && ( celm <= belm ) && ( celm <= delm ) )
	    {
		++cpos;
	    }

	    else
	    {
                ++dpos;
	    }
	}
    }

    return postProInnerProd(result);
}

//----------------------------------------------

template <class T>
double &mProductAssumeReal(double &res, const Vector<const Vector <T> *> &x)
{
    res = 0;

    int m = x.size();

    if ( m )
    {
        int i,j;
        int isinf = 0;

        for ( j = 0 ; j < m ; ++j )
        {
            isinf += ( (*(x(j))).infsize() ? 1 : 0 );
        }

        if ( isinf )
        {
            NiceAssert( isinf == m );

            if ( m == 1 )
            {
                return (*(x(0))).inner1Real(res);
            }

            else
            {
                retVector<const Vector <T> *> tmpva;

                return (*(x(0))).innerpReal(res,x(1,1,m-1,tmpva));
            }
        }

        int dim = (*(x(0))).size();

        if ( dim )
        {
            double temp;

            for ( i = 0 ; i < dim ; ++i )
            {
                temp = 1;

                for ( j = 0 ; j < m ; ++j )
                {
                    NiceAssert( (*(x(j))).size() == dim );

                    scalmul(temp,(*(x(j)))(i));
                }

                res += temp;
            }
        }
    }

    return res;
}

template <> inline double &mProductAssumeReal(double &res, const Vector<const Vector <double> *> &x)
{
    return mProduct(res,x);
}

template <class T>
T &mProduct(T &result, const Vector<const Vector <T> *> &x)
{
    setzero(result);

    if ( x.size() )
    {
        int i,j;
        int isinf = 0;

        for ( j = 0 ; j < x.size() ; ++j )
        {
            isinf += ( (*(x(j))).infsize() ? 1 : 0 );
        }

        if ( isinf )
        {
            NiceAssert( isinf == x.size() );

            if ( x.size() == 1 )
            {
                return (*(x(0))).inner1(result);
            }

            else
            {
                retVector<const Vector <T> *> tmpva;

                return (*(x(0))).innerp(result,x(1,1,x.size()-1,tmpva));
            }
        }

        Vector<T> temp(*(x(0)));

	if ( x.size() > 1 )
	{
            // NB: we process the product in pairs, so if T == gentype
            //     then ("a"*"a")*("b"*"b") will "correctly" evaluate
            //     to 1, not error (ie symbolic multiplication).

            temp *= *(x(1));

            if ( x.size() > 2 )
            {
	        for ( i = 2 ; i < x.size() ; i += 2 )
	        {
                    if ( i == x.size()-1 )
                    {
                        temp *= *(x(i));
                    }

                    else
                    {
                        temp *= ((*(x(i)))*(*(x(i+1))));
                    }
                }
	    }
	}

        result = sum(temp);
    }

    return postProInnerProd(result);
}

template <> inline double &mProduct(double &result, const Vector<const Vector <double> *> &x)
{
    int size = x.size();

    result = 0.0;

    if ( size )
    {
        int i,j;
        int isinf = 0;

        for ( j = 0 ; j < size ; ++j )
        {
            isinf += ( (*(x.v(j))).infsize() ? 1 : 0 );
        }

        if ( isinf )
        {
            NiceAssert( isinf == size );

            if ( size == 1 )
            {
                return (*(x.v(0))).inner1(result);
            }

            else
            {
                retVector<const Vector <double> *> tmpva;

                return (*(x.v(0))).innerp(result,x(1,1,x.size()-1,tmpva));
            }
        }

        Vector<double> temp(*(x(0)));

	if ( x.size() > 1 )
	{
            // NB: we process the product in pairs, so if T == gentype
            //     then ("a"*"a")*("b"*"b") will "correctly" evaluate
            //     to 1, not error (ie symbolic multiplication).

            temp *= *(x(1));

            if ( x.size() > 2 )
            {
	        for ( i = 2 ; i < x.size() ; i += 2 )
	        {
                    if ( i == x.size()-1 )
                    {
                        temp *= *(x.v(i));
                    }

                    else
                    {
                        temp *= ((*(x.v(i)))*(*(x.v(i+1))));
                    }
                }
	    }
	}

        result = sum(temp);
    }

    return result;
}

template <class T>
T &mProductScaled(T &result, const Vector<const Vector <T> *> &x, const Vector<T> &scale)
{
    setzero(result);

    if ( x.size() )
    {
        int i,j;
        int isinf = 0;

        for ( j = 0 ; j < x.size() ; ++j )
        {
            isinf += ( (*(x(j))).infsize() ? 1 : 0 );
        }

        if ( isinf )
        {
            NiceThrow("Scaled m-FuncVector-product not supported");

            return result;
        }

        Vector<T> temp(*(x(0)));
        temp /= scale;

	if ( x.size() > 1 )
	{
            // NB: we process the product in pairs, so if T == gentype
            //     then ("a"*"a")*("b"*"b") will "correctly" evaluate
            //     to 1, not error (ie symbolic multiplication).

            temp *= *(x(1));
            temp /= scale;

            if ( x.size() > 2 )
            {
	        for ( i = 2 ; i < x.size() ; i += 2 )
	        {
                    if ( i == x.size()-1 )
                    {
                        temp *= *(x(i));
                        temp /= scale;
                    }

                    else
                    {
                        temp *= (((*(x(i)))/scale)*((*(x(i+1)))/scale));
                    }
                }
	    }
	}

        result = sum(temp);
    }

    return postProInnerProd(result);
}

template <class T>
T &indexedmProduct(T &result, const Vector<int> &n, const Vector<const Vector <T> *> &x)
{
    setzero(result);

    if ( x.size() )
    {
        NiceAssert( !((*(x(0))).infsize()) );

	int i;
        Vector<T> a(*(x(0)));

	int nsize = n.size();
	int asize = a.size(); //indsize();

        int npos = 0;
	int apos = 0;

        int nelm;
	int aelm;

	while ( ( npos < nsize ) && ( apos < asize ) )
	{
            nelm = n.v(npos);
            aelm = apos; //a.ind(apos);

	    if ( aelm == nelm )
	    {
                ++npos;
		++apos;
	    }

	    else if ( nelm < aelm )
	    {
		++npos;
	    }

            else
	    {
                a.zero(apos); //a.ind(apos));
		++apos;
	    }
	}

	while ( apos < asize )
	{
	    a.zero(apos); //a.ind(apos));
	    ++apos;
	}

	if ( x.size() > 1 )
	{
            NiceAssert( !((*(x(1))).infsize()) );

            // NB: we process the product in pairs, so if T == gentype
            //     then ("a"*"a")*("b"*"b") will "correctly" evaluate
            //     to 1, not error (ie symbolic multiplication).

            a *= *(x(1));

            if ( x.size() > 2 )
            {
	        for ( i = 2 ; i < x.size() ; i += 2 )
	        {
                    NiceAssert( !((*(x(i))).infsize()) );

                    if ( i == x.size()-1 )
                    {
                        a *= *(x(i));
                    }

                    else
                    {
                        a *= ((*(x(i)))*(*(x(i+1))));
                    }
                }
	    }
	}

        result = sum(a);
    }

    return postProInnerProd(result);
}

template <class T>
T &indexedmProductScaled(T &result, const Vector<int> &n, const Vector<const Vector <T> *> &x, const Vector<T> &scale)
{
    setzero(result);

    if ( x.size() )
    {
        NiceAssert( !((*(x(0))).infsize()) );

	int i;
        Vector<T> a(*(x(0)));
        a /= scale;

	int nsize = n.size();
	int asize = a.size(); //indsize();

        int npos = 0;
	int apos = 0;

        int nelm;
	int aelm;

	while ( ( npos < nsize ) && ( apos < asize ) )
	{
            nelm = n.v(npos);
            aelm = apos; //a.ind(apos);

	    if ( aelm == nelm )
	    {
                ++npos;
		++apos;
	    }

	    else if ( nelm < aelm )
	    {
		++npos;
	    }

            else
	    {
                a.zero(apos); //a.ind(apos));
		++apos;
	    }
	}

	while ( apos < asize )
	{
	    a.zero(apos); //a.ind(apos));
	    ++apos;
	}

	if ( x.size() > 1 )
	{
            NiceAssert( !((*(x(1))).infsize()) );

            // NB: we process the product in pairs, so if T == gentype
            //     then ("a"*"a")*("b"*"b") will "correctly" evaluate
            //     to 1, not error (ie symbolic multiplication).

            a *= *(x(1));
            a /= scale;

            if ( x.size() > 2 )
            {
	        for ( i = 2 ; i < x.size() ; i += 2 )
	        {
                    NiceAssert( !((*(x(i))).infsize()) );

                    if ( i == x.size()-1 )
                    {
                        a *= *(x(i));
                        a /= scale;
                    }

                    else
                    {
                        a *= (((*(x(i)))/scale)*((*(x(i+1)))/scale));
                    }
                }
	    }
	}

        result = sum(a);
    }

    return postProInnerProd(result);
}





























// Mathematical operator overloading

template <class T> Vector<T>  operator+ (const Vector<T> &left_op, const Vector<T> &right_op)
{
    if ( !left_op.infsize() && right_op.infsize() )
    {
        return right_op+left_op;
    }

    Vector<T> res(left_op);

    return ( res += right_op );
}

template <class T> Vector<T>  operator+ (const Vector<T> &left_op, const T         &right_op)
{
    Vector<T> res(left_op);

    return ( res += right_op );
}

template <class T> Vector<T>  operator+ (const T         &left_op, const Vector<T> &right_op)
{
    Vector<T> res(right_op);

    return ( res += left_op );
}

template <class T> Vector<T>  operator- (const Vector<T> &left_op, const Vector<T> &right_op)
{
    if ( !left_op.infsize() && right_op.infsize() )
    {
        Vector<T> right_alt(right_op);

        right_alt.negate();

        return right_alt+left_op;
    }

    Vector<T> res(left_op);

    return ( res -= right_op );
}

template <class T> Vector<T>  operator- (const Vector<T> &left_op, const T         &right_op)
{
    Vector<T> res(left_op);

    return ( res -= right_op );
}

template <class T> Vector<T>  operator- (const T         &left_op, const Vector<T> &right_op)
{
    Vector<T> res(right_op);

    res.negate();

    return ( res += left_op );
}

template <class T> Vector<T>  operator* (const Vector<T> &left_op, const Vector<T> &right_op)
{
    Vector<T> res(left_op);

    return ( res *= right_op );
}

template <class T> Vector<T>  operator* (const Vector<T> &left_op, const T         &right_op)
{
    Vector<T> res(left_op);

    return ( res *= right_op );
}

template <class T> Vector<T>  operator* (const T         &left_op, const Vector<T> &right_op)
{
    Vector<T> res(right_op);

    return ( left_op *= res );
}

template <class T> Vector<T>  operator/ (const Vector<T> &left_op, const Vector<T> &right_op)
{
    Vector<T> res(left_op);

    return ( res /= right_op );
}

template <class T> Vector<T>  operator/ (const Vector<T> &left_op, const T         &right_op)
{
    Vector<T> res(left_op);

    return ( res /= right_op );
}

template <class T> Vector<T>  operator/ (const T         &left_op, const Vector<T> &right_op)
{
    Vector<T> res(right_op);

    return ( left_op /= res );
}

template <class T>          Vector<T> &operator+= (const T         &left_op,       Vector<T> &right_op)
{
    return ( right_op += left_op );
}

template <class T>          Vector<T> &operator-= (const T         &left_op,       Vector<T> &right_op)
{
    right_op.negate();

    return ( right_op += left_op );
}

template <class T> Vector<T> &leftmult (      Vector<T> &left_op, const Vector<T> &right_op)
{
    return ( left_op *= right_op );
}

template <class T> Vector<T> &leftmult (      Vector<T> &left_op, const T         &right_op)
{
    return ( left_op *= right_op );
}

template <class T> Vector<T>  operator+ (const Vector<T> &left_op)
{
    Vector<T> res(left_op);

    if svm_constexpr_if ( !(std::is_floating_point<T>::value) && !(std::is_integral<T>::value) )
    {
        if ( res.infsize() )
        {
            res.posate();
        }

        else
        {
            for ( int i = 0 ; i < left_op.size() ; ++i )
	    {
                setposate(res("&",i));
	    }
	}
    }

    return res;
}

template <class T> Vector<T>  operator- (const Vector<T> &left_op)
{
    Vector<T> res(left_op);

    if ( res.infsize() )
    {
        res.negate();
    }

    else
    {
        for ( int i = 0 ; i < left_op.size() ; ++i )
        {
            setnegate(res("&",i));
        }
    }

    return res;
}














template <class S, class T> Vector<S> &operator*=(Vector<S> &left_op, const Vector<T> &right_op)
{
    if ( right_op.size() && left_op.shareBase(right_op) )
    {
        Vector<T> temp(right_op);

        left_op *= temp;
    }

    else
    {
	// We treat empty vectors as additive identities (ie. zero)

        NiceAssert( !left_op.infsize() && !right_op.infsize() );
        NiceAssert( ( left_op.size() == right_op.size() ) || !(left_op.size()) || !(right_op.size()) );

	if ( !(left_op.size()) && right_op.size() )
	{
	    left_op.resize(right_op.size());
            left_op.zero();
	}

	else if ( !(right_op.size()) )
	{
	    left_op.zero();
	}

	else
	{
	    for ( int i = 0 ; i < left_op.size() ; ++i )
	    {
		left_op("&",i) *= right_op(i);
	    }
	}
    }

    return left_op;
}

template <class T> Vector<T> &operator*=(      Vector<T> &left_op, const T         &right_op)
{
    if ( left_op.infsize() )
    {
        left_op.mulit(right_op);
    }

    else
    {
	for ( int i = 0 ; i < left_op.size() ; ++i )
	{
            left_op("&",i) *= right_op;
	}
    }

    return left_op;
}

template <class T>          Vector<T> &operator*= (const T         &left_op,       Vector<T> &right_op)
{
    if ( right_op.infsize() )
    {
        right_op.rmulit(left_op);
    }

    else
    {
	for ( int i = 0 ; i < right_op.size() ; ++i )
	{
            rightmult(left_op,right_op("&",i));
	}
    }

    return right_op;
}

// elementwise for chol.hpp
template <class S, class T> Vector<S> &operator/=(Vector<S> &left_op, const Vector<T> &right_op)
{
    if ( right_op.size() && left_op.shareBase(right_op) )
    {
        Vector<T> temp(right_op);

        left_op /= temp;
    }

    else if ( left_op.infsize() || right_op.infsize() )
    {
        left_op.divit(right_op);
    }

    else
    {
	// We treat empty vectors as additive identities (ie. zero)

        NiceAssert( ( left_op.size() == right_op.size() ) || !(left_op.size()) || !(right_op.size()) );

	if ( !(left_op.size()) && right_op.size() )
	{
	    left_op.resize(right_op.size());
            left_op.zero();
	}

	else if ( !(right_op.size()) )
	{
	    left_op.zero();
	}

	else
	{
	    for ( int i = 0 ; i < left_op.size() ; ++i )
	    {
		left_op("&",i) /= right_op(i);
	    }
	}
    }

    return left_op;
}

template <class T> Vector<T> &operator/= (      Vector<T> &left_op, const T         &right_op)
{
    if ( left_op.infsize() )
    {
        left_op.divit(right_op);
    }

    else
    {
	for ( int i = 0 ; i < left_op.size() ; ++i )
	{
            left_op("&",i) /= right_op;
	}
    }

    return left_op;
}

template <class T>          Vector<T> &operator/= (const T         &left_op,       Vector<T> &right_op)
{
    T leftinv(inv(left_op));

    if ( right_op.infsize() )
    {
        right_op.rdivit(left_op);
    }

    else
    {
	for ( int i = 0 ; i < right_op.size() ; ++i )
	{
            rightmult(leftinv,right_op("&",i));
	}
    }

    return right_op;
}

template <class T> Vector<T> &operator+=(      Vector<T> &left_op, const Vector<T> &right_op)
{
    if ( right_op.size() && left_op.shareBase(right_op) )
    {
        Vector<T> temp(right_op);

        left_op += temp;
    }

    else if ( left_op.infsize() )
    {
        left_op.addit(right_op);
    }

    else if ( right_op.infsize() )
    {
        Vector<T> right_alt(left_op);

        left_op =  right_op;
        left_op += right_alt;
    }

    else
    {
	// We treat empty vectors as additive identities (ie. zero)

        NiceAssert( ( left_op.size() == right_op.size() ) || !(left_op.size()) || !(right_op.size()) );

	if ( left_op.size() && right_op.size() )
	{
	    for ( int i = 0 ; i < left_op.size() ; ++i )
	    {
		left_op("&",i) += right_op(i);
	    }
	}

	else if ( right_op.size() )
	{
            left_op = right_op;
	}
    }

    return left_op;
}

template <class T> Vector<T> &operator+=(      Vector<T> &left_op, const T         &right_op)
{
    //if ( left_op.size() )
    {
	for ( int i = 0 ; i < left_op.size() ; ++i )
	{
	    left_op("&",i) += right_op;
	}
    }

    return left_op;
}

template <class T> Vector<T> &operator-=(      Vector<T> &left_op, const Vector<T> &right_op)
{
    if ( right_op.size() && left_op.shareBase(right_op) )
    {
        Vector<T> temp(right_op);

        left_op -= temp;
    }

    else if ( left_op.infsize() )
    {
        left_op.subit(right_op);
    }

    else if ( right_op.infsize() )
    {
        Vector<T> right_alt(left_op);

        left_op =  right_op;
        left_op.negate();
        left_op += right_alt;
    }

    else
    {
	// We treat empty vectors as additive identities (ie. zero)

        NiceAssert( ( left_op.size() == right_op.size() ) || !(left_op.size()) || !(right_op.size()) );

	if ( left_op.size() && right_op.size() )
	{
	    for ( int i = 0 ; i < left_op.size() ; ++i )
	    {
		left_op("&",i) -= right_op(i);
	    }
	}

	else if ( right_op.size() )
	{
            left_op = -right_op;
	}
    }

    return left_op;
}

template <class T> Vector<T> &operator-=(      Vector<T> &left_op, const T         &right_op)
{
    //if ( left_op.size() )
    {
	for ( int i = 0 ; i < left_op.size() ; ++i )
	{
	    left_op("&",i) -= right_op;
	}
    }

    return left_op;
}

template <class T> Vector<T> &rightmult(const T         &left_op,       Vector<T> &right_op)
{
    return ( left_op *= right_op );
}

template <> inline Vector<double> &rightmult(const Vector<double> &left_op, Vector<double> &right_op)
{
    return ( right_op *= left_op );
}

template <> inline Vector<double> &rightmult(const double &left_op, Vector<double> &right_op)
{
    return ( right_op *= left_op );
}

template <> inline Vector<double> &operator*= (const double &left_op, Vector<double> &right_op)
{
    return right_op *= left_op;
}

template <> inline Vector<double> &operator/= (const double &left_op, Vector<double> &right_op)
{
    return right_op /= left_op;
}

template <> inline Vector<double> &operator*=(Vector<double> &left_op, const Vector<double> &right_op)
{
    int size = left_op.size();

    NiceAssert( !size || !right_op.size() || ( right_op.size() == size ) );

    if ( size && right_op.size() && left_op.contiguous() && right_op.contiguous() )
    {
        fastMulBy(&left_op("&",0),&right_op(0),size);
    }

    else if ( !(left_op.shareBase(right_op)) && size && right_op.size() )
    {
        fastMulBy(left_op.unsafecontent(),left_op.unsafepivot(),left_op.unsafeib(),left_op.unsafeis(),right_op.unsafeccontent(),right_op.unsafepivot(),right_op.unsafeib(),right_op.unsafeis(),size);
    }

    else if ( size && left_op.shareBase(right_op) )
    {
        Vector<double> temp(right_op);

        left_op *= temp;
    }

    else if ( !size && right_op.size() )
    {
        left_op.resize(right_op.size());
        left_op.zero();
    }

    else if ( size && !(right_op.size()) )
    {
        left_op.zero();
    }

    return left_op;
}

template <> inline Vector<double> &operator*=(Vector<double> &left_op, const double &right_op)
{
    int size = left_op.size();

    if ( size && left_op.contiguous() )
    {
        fastMulBy(&left_op("&",0),right_op,left_op.size());
    }

    else if ( size )
    {
        fastMulBy(left_op.unsafecontent(),left_op.unsafepivot(),left_op.unsafeib(),left_op.unsafeis(),right_op,left_op.size());
    }

    return left_op;
}


// elementwise for chol.hpp
template <> inline Vector<double> &operator/=(Vector<double> &left_op, const Vector<double> &right_op)
{
    int size = left_op.size();

    NiceAssert( !size || !right_op.size() || ( right_op.size() == size ) );

    if ( size && right_op.size() && left_op.contiguous() && right_op.contiguous() )
    {
        fastDivBy(&left_op("&",0),&right_op(0),size);
    }

    else if ( !(left_op.shareBase(right_op)) && size && right_op.size() )
    {
        fastDivBy(left_op.unsafecontent(),left_op.unsafepivot(),left_op.unsafeib(),left_op.unsafeis(),right_op.unsafeccontent(),right_op.unsafepivot(),right_op.unsafeib(),right_op.unsafeis(),size);
    }

    else if ( size && left_op.shareBase(right_op) )
    {
        Vector<double> temp(right_op);

        left_op /= temp;
    }

    else if ( !size && right_op.size() )
    {
        left_op.resize(right_op.size());
        left_op.zero();
    }

    else if ( size && !(right_op.size()) )
    {
        left_op.zero();
    }

    return left_op;
}

template <> inline Vector<double> &operator/=(Vector<double> &left_op, const double &right_op)
{
    int size = left_op.size();

    if ( size && left_op.contiguous() )
    {
        fastDivBy(&left_op("&",0),right_op,left_op.size());
    }

    else if ( size )
    {
        fastDivBy(left_op.unsafecontent(),left_op.unsafepivot(),left_op.unsafeib(),left_op.unsafeis(),right_op,left_op.size());
    }

    return left_op;
}


template <> inline Vector<double> &operator+=(Vector<double> &left_op, const Vector<double> &right_op)
{
    int size = left_op.size();

    NiceAssert( !size || !right_op.size() || ( right_op.size() == size ) );

    if ( size && right_op.size() && left_op.contiguous() && right_op.contiguous() )
    {
        fastAddTo(&left_op("&",0),&right_op(0),size);
    }

    else if ( !(left_op.shareBase(right_op)) && size && right_op.size() )
    {
        fastAddTo(left_op.unsafecontent(),left_op.unsafepivot(),left_op.unsafeib(),left_op.unsafeis(),right_op.unsafeccontent(),right_op.unsafepivot(),right_op.unsafeib(),right_op.unsafeis(),size);
    }

    else if ( size && left_op.shareBase(right_op) )
    {
        Vector<double> temp(right_op);

        left_op += temp;
    }

    else if ( !size && right_op.size() )
    {
        left_op = right_op;
    }

    return left_op;
}

template <> inline Vector<double> &operator+=(Vector<double> &left_op, const double &right_op)
{
    int size = left_op.size();

    if ( size && left_op.contiguous() )
    {
        fastAddTo(&left_op("&",0),right_op,size);
    }

    else if ( size )
    {
        fastAddTo(left_op.unsafecontent(),left_op.unsafepivot(),left_op.unsafeib(),left_op.unsafeis(),right_op,size);
    }

    return left_op;
}

template <> inline Vector<double> &operator-=(Vector<double> &left_op, const Vector<double> &right_op)
{
    int size = left_op.size();

    NiceAssert( !size || !right_op.size() || ( right_op.size() == size ) );

    if ( size && right_op.size() && left_op.contiguous() && right_op.contiguous() )
    {
        fastSubTo(&left_op("&",0),&right_op(0),size);
    }

    else if ( !(left_op.shareBase(right_op)) && size && right_op.size() )
    {
        fastSubTo(left_op.unsafecontent(),left_op.unsafepivot(),left_op.unsafeib(),left_op.unsafeis(),right_op.unsafeccontent(),right_op.unsafepivot(),right_op.unsafeib(),right_op.unsafeis(),size);
    }

    else if ( size && left_op.shareBase(right_op) )
    {
        Vector<double> temp(right_op);

        left_op -= temp;
    }

    else if ( !size && right_op.size() )
    {
        left_op = right_op;
        left_op.negate();
    }

    return left_op;
}

template <> inline Vector<double> &operator-=(Vector<double> &left_op, const double &right_op)
{
    int size = left_op.size();

    if ( size && left_op.contiguous() )
    {
        fastSubTo(&left_op("&",0),right_op,size);
    }

    else if ( size )
    {
        fastSubTo(left_op.unsafecontent(),left_op.unsafepivot(),left_op.unsafeib(),left_op.unsafeis(),right_op,size);
    }

    return left_op;
}


template <class T>
Vector<T> &kronprod(Vector<T> &res, const Vector<T> &a, const Vector<T> &b)
{
    NiceAssert( !a.infsize() );
    NiceAssert( !b.infsize() );

    if ( shareBase(res,a) )
    {
        Vector<T> tmpa(a);

        return kronprod(res,tmpa,b);
    }

    if ( shareBase(res,b) )
    {
        Vector<T> tmpb(b);

        return kronprod(res,a,tmpb);
    }

    int i,j;

    res.resize((a.size())*(b.size())).zero();

    for ( i = 0 ; i < a.size() ; ++i )
    {
        for ( j = 0 ; j < b.size() ; ++j )
        {
            res.set((i*(b.size()))+j,a(i)*b(j));
        }
    }

    return res;
}

template <class T>
Vector<T> &kronpow(Vector<T> &res, const Vector<T> &a, int n)
{
    NiceAssert( n >= 0 );

    if ( n == 0 )
    {
        setident((res.resize(1))("&",0));
    }

    else if ( n == 1 )
    {
        res = a;
    }

    else
    {
        Vector<T> b;

        kronprod(res,a,kronpow(b,a,n-1));
    }

    return res;
}

template <class T> Vector<T> &epow(Vector<T> &res, const Vector<T> &a, T c)
{
    NiceAssert( !a.infsize() );

    res.resize(a.size());

    for ( int i = 0 ; i < a.size() ; ++i )
    {
        res.set(i,pow(a(i),c));
    }

    return res;
}


template <class T> int operator==(const Vector<T> &left_op, const Vector<T> &right_op)
{
    if ( left_op.size() != right_op.size() )
    {
        return 0;
    }

    for ( int i = 0 ; i < left_op.size() ; ++i )
    {
        if ( !( left_op(i) == right_op(i) ) )
        {
            return 0;
        }
    }

    return 1;
}

template <class T> int operator==(const Vector<T> &left_op, const T         &right_op)
{
    //if ( left_op.size() )
    {
	for ( int i = 0 ; i < left_op.size() ; ++i )
	{
	    if ( !( left_op(i) == right_op ) )
	    {
                return 0;
	    }
	}
    }

    return 1;
}

template <class T> int operator==(const T         &left_op, const Vector<T> &right_op)
{
    //if ( right_op.size() )
    {
	for ( int i = 0 ; i < right_op.size() ; ++i )
	{
	    if ( !( left_op == right_op(i) ) )
	    {
                return 0;
	    }
	}
    }

    return 1;
}

template <class T> int operator!=(const Vector<T> &left_op, const Vector<T> &right_op)
{
    return !( left_op == right_op );
}

template <class T> int operator!=(const Vector<T> &left_op, const T         &right_op)
{
    return !( left_op == right_op );
}

template <class T> int operator!=(const T         &left_op, const Vector<T> &right_op)
{
    return !( left_op == right_op );
}

template <class T> int operator< (const Vector<T> &left_op, const Vector<T> &right_op)
{
    if ( left_op.size() != right_op.size() )
    {
        return 0;
    }

    //if ( left_op.size() )
    {
        for ( int i = 0 ; i < left_op.size() ; ++i )
        {
            if ( !( left_op(i) < right_op(i) ) )
            {
                return 0;
            }
        }
    }

    return 1;
}

template <class T> int operator< (const Vector<T> &left_op, const T         &right_op)
{
    //if ( left_op.size() )
    {
	for ( int i = 0 ; i < left_op.size() ; ++i )
	{
	    if ( !( left_op(i) < right_op ) )
	    {
                return 0;
	    }
	}
    }

    return 1;
}

template <class T> int operator< (const T         &left_op, const Vector<T> &right_op)
{
    //if ( right_op.size() )
    {
	for ( int i = 0 ; i < right_op.size() ; ++i )
	{
	    if ( !( left_op < right_op(i) ) )
	    {
                return 0;
	    }
	}
    }

    return 1;
}

template <class T> int operator<=(const Vector<T> &left_op, const Vector<T> &right_op)
{
    if ( left_op.size() != right_op.size() )
    {
        return 0;
    }

    //if ( left_op.size() )
    {
	for ( int i = 0 ; i < left_op.size() ; ++i )
	{
	    if ( !( left_op(i) <= right_op(i) ) )
	    {
                return 0;
	    }
	}
    }

    return 1;
}

template <class T> int operator<=(const Vector<T> &left_op, const T         &right_op)
{
    //if ( left_op.size() )
    {
	for ( int i = 0 ; i < left_op.size() ; ++i )
	{
	    if ( !( left_op(i) <= right_op ) )
	    {
                return 0;
	    }
	}
    }

    return 1;
}

template <class T> int operator<=(const T         &left_op, const Vector<T> &right_op)
{
    //if ( right_op.size() )
    {
	for ( int i = 0 ; i < right_op.size() ; ++i )
	{
	    if ( !( left_op <= right_op(i) ) )
	    {
                return 0;
	    }
	}
    }

    return 1;
}

template <class T> int operator> (const Vector<T> &left_op, const Vector<T> &right_op)
{
    if ( left_op.size() != right_op.size() )
    {
        return 0;
    }

    //if ( left_op.size() )
    {
	for ( int i = 0 ; i < left_op.size() ; ++i )
	{
	    if ( !( left_op(i) > right_op(i) ) )
	    {
                return 0;
	    }
	}
    }

    return 1;
}

template <class T> int operator> (const Vector<T> &left_op, const T         &right_op)
{
    //if ( left_op.size() )
    {
	for ( int i = 0 ; i < left_op.size() ; ++i )
	{
	    if ( !( left_op(i) > right_op ) )
	    {
                return 0;
	    }
	}
    }

    return 1;
}

template <class T> int operator> (const T         &left_op, const Vector<T> &right_op)
{
    //if ( right_op.size() )
    {
	for ( int i = 0 ; i < right_op.size() ; ++i )
	{
	    if ( !( left_op > right_op(i) ) )
	    {
                return 0;
	    }
	}
    }

    return 1;
}

template <class T> int operator>=(const Vector<T> &left_op, const Vector<T> &right_op)
{
    if ( left_op.size() != right_op.size() )
    {
        return 0;
    }

    //if ( left_op.size() )
    {
	for ( int i = 0 ; i < left_op.size() ; ++i )
	{
	    if ( !( left_op(i) >= right_op(i) ) )
	    {
                return 0;
	    }
	}
    }

    return 1;
}

template <class T> int operator>=(const Vector<T> &left_op, const T         &right_op)
{
    //if ( left_op.size() )
    {
	for ( int i = 0 ; i < left_op.size() ; ++i )
	{
	    if ( !( left_op(i) >= right_op ) )
	    {
                return 0;
	    }
	}
    }

    return 1;
}

template <class T> int operator>=(const T         &left_op, const Vector<T> &right_op)
{
    //if ( right_op.size() )
    {
	for ( int i = 0 ; i < right_op.size() ; ++i )
	{
	    if ( !( left_op >= right_op(i) ) )
	    {
                return 0;
	    }
	}
    }

    return 1;
}




// Conversion from strings

template <class T> Vector<T> &atoVector(Vector<T> &dest, const std::string &src)
{
    std::istringstream srcmod(src);

    srcmod >> dest;

    return dest;
}





inline Vector<int> &randPerm(Vector<int> &res)
{
    if ( res.size() )
    {
        Vector<int> temp(res.size());

        int i,j;

        for ( i = 0 ; i < res.size() ; ++i )
        {
            temp.sv(i,i);
        }

        for ( i = 0 ; i < res.size() ; ++i )
        {
            //j = svm_rand()%(temp.size());
            j = rand()%(temp.size());

            res.sv(i,temp.v(j));
            temp.remove(j);
        }
    }

    return res;
}

template <class T> Vector<T> &randrfill(Vector<T> &res) { return res.applyon(randrfill); }
template <class T> Vector<T> &randbfill(Vector<T> &res) { return res.applyon(randbfill); }
template <class T> Vector<T> &randBfill(Vector<T> &res) { return res.applyon(randBfill); }
template <class T> Vector<T> &randgfill(Vector<T> &res) { return res.applyon(randgfill); }
template <class T> Vector<T> &randpfill(Vector<T> &res) { return res.applyon(randpfill); }
template <class T> Vector<T> &randufill(Vector<T> &res) { return res.applyon(randufill); }
template <class T> Vector<T> &randefill(Vector<T> &res) { return res.applyon(randefill); }
template <class T> Vector<T> &randGfill(Vector<T> &res) { return res.applyon(randGfill); }
template <class T> Vector<T> &randwfill(Vector<T> &res) { return res.applyon(randwfill); }
template <class T> Vector<T> &randxfill(Vector<T> &res) { return res.applyon(randxfill); }
template <class T> Vector<T> &randnfill(Vector<T> &res) { return res.applyon(randnfill); }
template <class T> Vector<T> &randlfill(Vector<T> &res) { return res.applyon(randlfill); }
template <class T> Vector<T> &randcfill(Vector<T> &res) { return res.applyon(randcfill); }
template <class T> Vector<T> &randCfill(Vector<T> &res) { return res.applyon(randCfill); }
template <class T> Vector<T> &randffill(Vector<T> &res) { return res.applyon(randffill); }
template <class T> Vector<T> &randtfill(Vector<T> &res) { return res.applyon(randtfill); }

#define MINZEROSIZE        1000000
#define ZEROALLOCAHEADFRAC 1.2

inline const Vector<int> &zerointvecbasic(void)
{
    const static Vector<int> zerores(zerointarray(),"&");

    return zerores;
}

inline const Vector<int> &oneintvecbasic(void)
{
    const static Vector<int> oneres(oneintarray(),"&");

    return oneres;
}

inline const Vector<double> &zerodoublevecbasic(void)
{
    const static Vector<double> zerores(zerodoublearray(),"&");

    return zerores;
}

inline const Vector<double> &onedoublevecbasic(void)
{
    const static Vector<double> oneres(onedoublearray(),"&");

    return oneres;
}

inline const Vector<double> &ninfdoublevecbasic(void)
{
    const static Vector<double> ninfres(ninfdoublearray(),"&");

    return ninfres;
}

inline const Vector<double> &pinfdoublevecbasic(void)
{
    const static Vector<double> pinfres(pinfdoublearray(),"&");

    return pinfres;
}

inline const Vector<int> &zerointvec(int size, retVector<int> &tmpv)
{
    return zerointvecbasic()(0,1,size-1,tmpv);
}

inline const Vector<double> &zerodoublevec(int size, retVector<double> &tmpv)
{
    return zerodoublevecbasic()(0,1,size-1,tmpv);
}

inline const Vector<int> &oneintvec(int size, retVector<int> &tmpv)
{
    return oneintvecbasic()(0,1,size-1,tmpv);
}

inline const Vector<double> &onedoublevec(int size, retVector<double> &tmpv)
{
    return onedoublevecbasic()(0,1,size-1,tmpv);
}

inline const Vector<double> &ninfdoublevec(int size, retVector<double> &tmpv)
{
    return ninfdoublevecbasic()(0,1,size-1,tmpv);
}

inline const Vector<double> &pinfdoublevec(int size, retVector<double> &tmpv)
{
    return pinfdoublevecbasic()(0,1,size-1,tmpv);
}

inline const Vector<int> &cntintvec(int size, retVector<int> &tmpv)
{
    static thread_local Vector<int> cntres(cntintarray(MINZEROSIZE));

    NiceAssert( size >= 0 );

    if ( size > cntres.dsize )
    {
        cntres.dsize = cntintarray(size)->array_size();
    }

    const Vector<int> &res = cntres(0,1,size-1,tmpv);

    return res;
}

//inline const Vector<double> &cntdoublevec(int size, retVector<double> &tmpv)
//{
//    static thread_local Vector<double> cntres(cntdoublearray(MINZEROSIZE));
//
//    NiceAssert( size >= 0 );
//
//    if ( size > cntres.dsize )
//    {
//        cntres.dsize = cntdoublearray(size)->array_size();
//    }
//
//    const Vector<double> &res = cntres(0,1,size-1,tmpv);
//
//    return res;
//}


























//template <class T> inline void makeFuncVector(const std::string &typestring, Vector<T> *&res, std::istream &src); // operator>> analog
//template <class T> inline void makeFuncVector(const std::string &typestring, Vector<T> *&res, std::istream &src, int processxyzvw); // streamItIn analog

//temp placeholder while FuncVector gets written
//template <class T> inline void makeFuncVector(const std::string &typestring, Vector<T> *&res, std::istream &src) { NiceThrow("makeFuncVector stub called"); (void) typestring; (void) src; res = nullptr; return; }
//template <class T> inline void makeFuncVector(const std::string &typestring, Vector<T> *&res, std::istream &src, int processxyzvw) { NiceThrow("makeFuncVector stub b called"); (void) typestring; (void) src; (void) processxyzvw; res = nullptr; return; }

// NOTE: the templated version above "misses" specialisations (sometimes, not always) for some reason I don't understand, so I do the following tedious method instead

#define OVERLAYMAKEFNVECTOR(type) \
\
inline void makeFuncVector(const std::string &typestring, Vector<type > *&res, std::istream &src); \
inline void makeFuncVector(const std::string &typestring, Vector<type > *&res, std::istream &src, int processxyzvw); \
\
inline void makeFuncVector(const std::string &typestring, Vector<type > *&res, std::istream &src) { NiceThrow("makeFuncVector stub called"); (void) typestring; (void) src; res = nullptr; } \
inline void makeFuncVector(const std::string &typestring, Vector<type > *&res, std::istream &src, int processxyzvw) { NiceThrow("makeFuncVector stub b called"); (void) typestring; (void) src; (void) processxyzvw; res = nullptr; }

OVERLAYMAKEFNVECTOR(int)
OVERLAYMAKEFNVECTOR(double)
OVERLAYMAKEFNVECTOR(Vector<int>)
OVERLAYMAKEFNVECTOR(Vector<double>)
OVERLAYMAKEFNVECTOR(Vector<SparseVector<gentype> >)



// Stream operators

template <class T>
std::ostream &operator<<(std::ostream &output, const Vector<T> &src)
{
    if ( src.imoverhere )
    {
        (src.overhere()).outstream(output);
    }

    else if ( src.infsize() )
    {
        src.outstream(output);
    }

    else
    {
        int xsize = src.size();

        output << "[ ";

        //if ( xsize )
        {
            for ( int i = 0 ; i < xsize ; ++i )
	    {
	        if ( i > 0 )
	        {
                    output << "  ";
	        }

	        if ( i < xsize-1 )
	        {
                    output << src(i) << "\t;" << src.getnewln();
	        }

	        else
	        {
		    output << src(i) << "\t";
                }
	    }
	}

        output << "  ]";
    }

    return output;
}

template <class T>
std::ostream &printoneline(std::ostream &output, const Vector<T> &src)
{
    if ( src.imoverhere )
    {
        (src.overhere()).outstream(output);
    }

    else if ( src.infsize() )
    {
        src.outstream(output);
    }

    else
    {
        int xsize = src.size();

        output << "[ ";

        //if ( xsize )
        {
            for ( int i = 0 ; i < xsize ; ++i )
	    {
	        if ( i > 0 )
	        {
                    output << "  ";
	        }

	        if ( i < xsize-1 )
	        {
                    output << src(i) << "\t; ";
	        }

	        else
	        {
		    output << src(i) << "\t";
                }
	    }
	}

        output << "  ]";
    }

    return output;
}

template <class T>
std::istream &operator>>(std::istream &input, Vector<T> &dest)
{
    // Note: we don't actually insist on ;'s being included in the input
    //       stream, so ( 1 5 2.2 ) is perfectly acceptable (as indeed is
    //       ( 2 2 ; 5 ; 1 2 3 ), though I would strongly advise against
    //       the latter... it probably won't be supported later).

    int i;
    char tt;

    // old version: pipe to buffer, NiceAssert(!strcmp(buffer,"["))

    while ( isspace(input.peek()) )
    {
	input.get(tt);
    }

    input.get(tt);

    if ( ( tt == '[' ) && ( input.peek() == '[' ) )
    {
        input.get(tt);
        NiceAssert( tt == '[' );

        while ( isspace(input.peek()) )
        {
            input.get(tt);
        }

        std::string typestring;

        input >> typestring;

        if ( dest.imoverhere )
        {
            // dest is a front for an FuncVector

            if ( (dest.overhere()).testsametype(typestring) )
            {
                (dest.overhere()).instream(input);
            }

            else
            {
                MEMDEL(dest.imoverhere);
                makeFuncVector(typestring,dest.imoverhere,input);
            }
        }

        else if ( dest.infsize() && dest.testsametype(typestring) )
        {
            dest.instream(input);
        }

        else if ( dest.infsize() )
        {
            NiceThrow("Vector type mismatch in stream attempt");
        }

        else
        {
            // make dest a front for an FuncVector

            makeFuncVector(typestring,dest.imoverhere,input);
        }
    }

    else if ( tt == '[' )
    {
        if ( dest.imoverhere )
        {
            MEMDEL(dest.imoverhere);
            dest.imoverhere = nullptr;
        }

        int xsize = 0;

        while ( 1 )
        {
            while ( ( isspace(input.peek()) ) || ( input.peek() == ';' ) || ( input.peek() == ',' ) )
            {
                input.get(tt);
            }

            if ( input.peek() == ']' )
            {
                input.get(tt);

                break;
            }

            if ( dest.size() == xsize )
            {
                dest.add(xsize);
            }

            input >> dest("&",xsize);

            ++xsize;
        }

        if ( dest.size() > xsize )
        {
	    for ( i = dest.size()-1 ; i >= xsize ; --i )
	    {
                dest.remove(xsize);
            }
	}
    }

    else
    {
        NiceThrow("Attempting to stream non-vector data to vector\n");
    }

    return input;
}

template <class T>
std::istream &streamItIn(std::istream &input, Vector<T> &dest, int processxyzvw)
{
    // Note: we don't actually insist on ;'s being included in the input
    //       stream, so ( 1 5 2.2 ) is perfectly acceptable (as indeed is
    //       ( 2 2 ; 5 ; 1 2 3 ), though I would strongly advise against
    //       the latter... it probably won't be supported later).

    int i;
    char tt;

    // old version: pipe to buffer, NiceAssert(!strcmp(buffer,"["))

    while ( isspace(input.peek()) )
    {
	input.get(tt);
    }

    input.get(tt);

    if ( ( tt == '[' ) && ( input.peek() == '[' ) )
    {
        input.get(tt);
        NiceAssert( tt == '[' );

        while ( isspace(input.peek()) )
        {
            input.get(tt);
        }

        std::string typestring;

        input >> typestring;

        if ( dest.imoverhere )
        {
            // dest is a front for an FuncVector

            if ( (*(dest.imoverhere)).testsametype(typestring) )
            {
                (dest.overhere()).streamItIn(input,processxyzvw);
            }

            else
            {
                MEMDEL(dest.imoverhere);
                makeFuncVector(typestring,dest.imoverhere,input,processxyzvw);
            }
        }

        else if ( dest.infsize() && dest.testsametype(typestring) )
        {
            dest.streamItIn(input,processxyzvw);
        }

        else if ( dest.infsize() )
        {
            NiceThrow("Vector type mismatch in stream attempt");
        }

        else
        {
            // make dest a front for an FuncVector

            makeFuncVector(typestring,dest.imoverhere,input,processxyzvw);
        }
    }

    else if ( tt == '[' )
    {
        if ( dest.imoverhere )
        {
            MEMDEL(dest.imoverhere);
            dest.imoverhere = nullptr;
        }

        int xsize = 0;

        while ( 1 )
        {
            while ( ( isspace(input.peek()) ) || ( input.peek() == ';' ) || ( input.peek() == ',' ) )
            {
                input.get(tt);
            }

            if ( input.peek() == ']' )
            {
                input.get(tt);

                break;
            }

            if ( dest.size() == xsize )
            {
                dest.add(xsize);
            }

            ::streamItIn(input,dest("&",xsize),processxyzvw);

            ++xsize;
        }

        if ( dest.size() > xsize )
        {
	    for ( i = dest.size()-1 ; i >= xsize ; --i )
	    {
                dest.remove(xsize);
            }
	}
    }

    else
    {
        NiceThrow("Attempting to streamItIn non-vector data to vector\n");
    }

    return input;
}




#endif


