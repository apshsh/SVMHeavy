
//
// Set class
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//
// Note that we can have a nominal "infinite set" - see rules in comments.
// These allow you to have "any value" elements in training vectors.
// Notation: {*}
//

#ifndef _set_h
#define _set_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include <sstream>
#include "vector.hpp"

template <class T> class Set;

// Stream operators

template <class T> std::ostream &operator<<(std::ostream &output, const Set<T> &src                       );
template <class T> std::istream &operator>>(std::istream &input,        Set<T> &dest                      );
template <class T> std::istream &streamItIn(std::istream &input,        Set<T> &dest, int processxyzvw = 1);

// Swap function

template <class T> void qswap(const Set<T> *a, const Set<T> *b);
template <class T> void qswap(Set<T> &a, Set<T> &b);

// The class itself

template <class T>
class Set
{
    template <class S> friend std::istream &operator>>(std::istream &input, Set<S> &dest);
    template <class S> friend void qswap(Set<S> &a, Set<S> &b);

public:

    // Constructor:

    explicit Set(bool xisinf = false, bool xisneg = false) { isinf = xisinf; isneg = xisneg; }
             Set(const Set<T>    &src) { contents = src.contents;                   isinf = src.isinf; isneg = src.isneg; }
    explicit Set(const Vector<T> &src) { contents = src;                            isinf = false;     isneg = false;     }
    explicit Set(const T         &src) { contents.resize(1); contents("&",0) = src; isinf = false;     isneg = false;     }

    // Assignment:

    Set<T> &operator=(const Set<T> &src) { contents = src.contents; isinf = src.isinf; isneg = src.isneg; return *this; }

    // simple set manipulations
    //
    // ident: set size of set to zero
    // zero: set size of set to zero
    // posate: apply setposate() to all elements of the set
    // negate: apply setnegate() to all elements of the set
    // conj: apply setconj() to all elements of the set
    //
    // each returns a reference to *this

    Set<T> &ident (void) { contents.resize(0); isinf = false; isneg = false;  return *this; }
    Set<T> &zero  (void) { contents.resize(0); isinf = false; isneg = false;  return *this; }
    Set<T> &posate(void) {                                                    return *this; }
    Set<T> &negate(void) {                                    isneg = !isneg; return *this; }
    Set<T> &conj  (void) {                                                    return *this; }
    Set<T> &rand  (void) { contents.rand();                   isneg = false;  return *this; }

    // Access:
    //
    // - all: returns a vector containing all elements in set
    // - contains(x): returns 1 if x is in set, 0 otherwise
    // - isinfset: this is a "contains everything" set (heuristic)
    // - isnegset: this is a negated set: eg a-b -> a+-b means {a0,a1,...}\{b0,b1,...}

    const Vector<T> &all(void) const
    {
        return contents;
    }

    bool contains(const T &x) const
    {
        if ( isinf )
        {
            return true;
        }

        else
        {
            for ( int i = 0 ; i < contents.size() ; ++i )
            {
                if ( contents(i) == x )
                {
                    return true;
                }
            }
        }

        return false;
    }

    bool isinfset(void) const { return isinf; }
    bool isnegset(void) const { return isneg; }

    // Add (if not present) and remove (if present) elements:

    Set<T> &add(const T &x)
    {
        if ( !isinf )
        {
            contents.append(size(),x);
        }

        return *this;
    }

    Set<T> &remove(const T &x)
    {
        if ( !isinf )
        {
            for ( int i = size()-1 ; i >= 0 ; --i )
            {
                if ( contents(i) == x )
                {
                    contents.remove(i);
                }
            }
        }

        return *this;
    }

    // Information (-1 is a placeholder for infinite size)

    int size(void) const { return isinf ? -1 : contents.size(); }

    // Function application - apply function fn to each element of set.

    Set<T> &applyon(T (*fn)(T))                                      { if ( !isinf ) { contents.applyon(fn);   } return *this; }
    Set<T> &applyon(T (*fn)(const T &))                              { if ( !isinf ) { contents.applyon(fn);   } return *this; }
    Set<T> &applyon(T (*fn)(T, const void *), const void *a)         { if ( !isinf ) { contents.applyon(fn,a); } return *this; }
    Set<T> &applyon(T (*fn)(const T &, const void *), const void *a) { if ( !isinf ) { contents.applyon(fn,a); } return *this; }
    Set<T> &applyon(T &(*fn)(T &))                                   { if ( !isinf ) { contents.applyon(fn);   } return *this; }
    Set<T> &applyon(T &(*fn)(T &, const void *), const void *a)      { if ( !isinf ) { contents.applyon(fn,a); } return *this; }

    // Don't use this

    Vector<T> &ncall(void) { return contents; }

private:

    Vector<T> contents;
    bool isinf;
    bool isneg;

    void removeDuplicates(void)
    {
        if ( !isinf )
        {
            for ( int i = 0 ; i < size()-1 ; ++i )
            {
                for ( int j = i+1 ; j < size() ; ++j )
                {
                    if ( contents(i) == contents(j) )
                    {
                        contents.remove(j);
                        --j;
                    }
                }
            }
        }
    }
};

template <class T> void qswap(Set<T> &a, Set<T> &b)
{
    qswap(a.contents,b.contents);
    qswap(a.isinf   ,b.isinf   );
}

template <class T> void qswap(const Set<T> *a, const Set<T> *b)
{
    const Set<T> *c;

    c = a;
    a = b;
    b = c;
}

// Various functions
//
// max: find max element, put index in i.  If two sets are given then finds max element difference
// min: find min element, put index in i.  If two sets are given then finds min element difference
// maxabs: find the |max| element.
// minabs: find the |min| element.
// sqabsmax: find the |max|*|max| element, put index in i.
// sqabsmin: find the |min|*|min| element, put index in i.
// sum: find the sum of elements in a set
// prod: find the product of elements in a set (arbitrary order, not good for non-commutative elements)
// Prod: find the product of elements in a set (arbitrary order, not good for non-commutative elements)
// mean: mean of elements
// median: median of elements
//
// setident: call a.ident()
// setzero: call a.zero()
// setposate: call a.posate()
// setnegate: call a.negate()
// setconj: call a.conj()
//
// norms, inner products
// =====================
//
// These are constructed by identifying the sets with sparse vectors.  To
// be precise, suppose every possible element of any finite set is represented
// by a single non-negative integer.  Then any given finite set can be
// identified with a unique binary-valued sparse vector:
//
// { a,b,c } <-> [ i_a:1 i_b:1 i_c:1 ]
//
// where i_a = the unique integer label associated with set element a.
//
// Let the inner product of two sets equal the inner product of the sparse
// vectors associated with the two sets under the above mapping.  Then:
//
// <A,B> = #(I(A,B)) = number of elements that A and B have in common
//
// (where # is the element count function and I is the intersection).  From
// this we can readily identify:
//
// ||A||_2^2 = <A,A> = #(A)
// ||A|| = sqrt(#(A))
//
// For the non-euclidean norms, we simply associate the set norms with the
// sparse vector representation norms, so:
//
// norm1(A)   = #(A)
// norm2(A)   = #(A)
// normp(A,p) = #(A)
// abs1(A)    = #(A)
// abs2(A)    = sqrt(#(A))
// absp(A,p)  = #(A)^{1/p}
//
// and, finally:
//
// absinf(A) = 0 if A is empty, 1 otherwise.
//norm(infset a) is inf, dist(a,infset b) = dist(infset a,b) = inf, so exp(-dist) = 0, dist(infset a, infset b) = 0
//dist(infset a,b) = norm2(infset a) + norm2(b) - 2<infset a,b>. So we need norm2(infset a) = inf, <infset a,b> = norm2(b)
//dist(a,infset b) = norm2(a) + norm2(infset b) - 2<a,infset b>. So we need norm2(infset b) = inf, <a,infset b> = norm2(a)
//dist(infset a,infset b) = norm2(infset a) + norm2(infset b) - 2<infset a,infset b>. So we need norm2(infset a) = norm2(infset b) = inf, <infset a,infset b> = inf

template <class T> T max(const Set<T> &a, const Set<T> &b);
template <class T> T min(const Set<T> &a, const Set<T> &b);

template <class T> const T &max     (const Set<T> &a);
template <class T> const T &min     (const Set<T> &a);
template <class T>       T  maxabs  (const Set<T> &a);
template <class T>       T  minabs  (const Set<T> &a);
template <class T>       T  sqabsmax(const Set<T> &a);
template <class T>       T  sqabsmin(const Set<T> &a);
template <class T>       T  sum     (const Set<T> &a);
template <class T>       T  prod    (const Set<T> &a);
template <class T>       T  Prod    (const Set<T> &a);
template <class T>       T  mean    (const Set<T> &a);
template <class T> const T &median  (const Set<T> &a);

template <class T> Set<T> &setident (Set<T> &a);
template <class T> Set<T> &setzero  (Set<T> &a);
template <class T> Set<T> &setposate(Set<T> &a);
template <class T> Set<T> &setnegate(Set<T> &a);
template <class T> Set<T> &setconj  (Set<T> &a);
template <class T> Set<T> &setrand  (Set<T> &a);
template <class T> Set<T> &postProInnerProd(Set<T> &a) { return a; }

template <class T> double abs1  (const Set<T> &a);
template <class T> double abs2  (const Set<T> &a);
template <class T> double absp  (const Set<T> &a, double p);
template <class T> double absinf(const Set<T> &a);
template <class T> double abs0  (const Set<T> &a);
template <class T> double norm1 (const Set<T> &a);
template <class T> double norm2 (const Set<T> &a);
template <class T> double normp (const Set<T> &a, double p);

template <class T> double &oneProduct  (double &res, const Set<T> &a);
template <class T> double &twoProduct  (double &res, const Set<T> &a, const Set<T> &b);
template <class T> double &threeProduct(double &res, const Set<T> &a, const Set<T> &b, const Set<T> &c);
template <class T> double &fourProduct (double &res, const Set<T> &a, const Set<T> &b, const Set<T> &c, const Set<T> &d);
template <class T> double &mProduct    (double &res, int m, const Set<T> *a);

template <class T> double &innerProduct       (double &res, const Set<T> &a, const Set<T> &b);
template <class T> double &innerProductRevConj(double &res, const Set<T> &a, const Set<T> &b);

// Conversion from strings

template <class T> Set<T> &atoSet(Set<T> &dest, const std::string &src);

// Set arithmetic
// ==============
//
// The arithmetic operations are tricky to define for sets.  They are not
// in fact used in the SVM code, but we still define them as follows:
//
// 1. Operations on individual set elements (sets A,B,..., scalars a,b,...):
//
//   +   posation                  - elementwise, unary,  return rvalue
//   -   negation                  - elementwise, unary,  return rvalue
//   ~   bitwise not               - elementwise, unary,  return rvalue
//     (A+a,a+A cases, operates on each element of A)
//   +   scalar addition           - elementwise, binary, return rvalue
//   -   scalar subtraction        - elementwise, binary, return rvalue
//   +   scalar multiplication     - elementwise, binary, return rvalue
//   /   scalar division           - elementwise, binary, return rvalue
//     (A+=a -> A = A+a, a+=A -> A = a+A)
//   +=  additive       assignment - elementwise, binary, return lvalue
//   -=  subtractive    assignment - elementwise, binary, return lvalue
//   +=  multiplicative assignment - elementwise, binary, return lvalue
//   /=  divisive       assignment - elementwise, binary, return lvalue
//
//    note that a/A  = inv(a)*A (left-division, exception to rule)
//    note that a/=A = a/A = inv(a)*A (left-division, exception to rule)
//
// 2. "True" set-wise operations
//
//   A+B = union of sets A and B
//   A-B = intersection of sets A and B
//   A*B = { [ai;bj] : A = { a0,a1,...}, B = { b0,b1,... } }
//   A+=B and A-=B are also defined, but A*=B is not
//
//   Note however that A*B will not necessarily operate precisely as
//   expected.  For example A*B*C = (A*B)*C is not well defined, and
//   moreover even A*B is not defined unless A and B have the same type
//   of scalar elements.  To fix this, we need to define a dissimilar n-tuple
//   type, and I don't know how to do that.
//
// FIXME: Implement some sort of dissimilar n-tuple type and fix A*B.
//        In a limited way you could define A*B for:
//        Set<int>*Set<int> = Set<ntuple<2>>
//        Set<int>*Set<ntuple<n>> = Set<ntuple<n+1>>
//        Set<ntuple<n>>*Set<int> = Set<ntuple<n+1>>
//        Set<ntuple<n>>*Set<ntupe<m>> = Set<ntuple<n+m>>

template <class T> Set<T> operator+(const Set<T> &left_op);
template <class T> Set<T> operator-(const Set<T> &left_op);

template <class T> Set<T>          operator+(const Set<T> &left_op, const Set<T> &right_op);
template <class T> Set<T>          operator+(const Set<T> &left_op, const T      &right_op);
template <class T> Set<T>          operator+(const T      &left_op, const Set<T> &right_op);
template <class T> Set<T>          operator-(const Set<T> &left_op, const Set<T> &right_op);
template <class T> Set<T>          operator-(const Set<T> &left_op, const T      &right_op);
template <class T> Set<Vector<T> > operator*(const Set<T> &left_op, const Set<T> &right_op);
template <class T> Set<Vector<T> > operator*(const Set<T> &left_op, const T      &right_op);
template <class T> Set<Vector<T> > operator*(const T      &left_op, const Set<T> &right_op);

template <class T> Set<T> &operator+=(Set<T> &left_op, const Set<T> &right_op);
template <class T> Set<T> &operator+=(Set<T> &left_op, const T      &right_op);
template <class T> Set<T> &operator-=(Set<T> &left_op, const Set<T> &right_op);
template <class T> Set<T> &operator-=(Set<T> &left_op, const T      &right_op);

// Relational operator overloading
// ===============================
//
// a == b: sets contain same elements
// a != b: logical negation of a == b
// a <  b: max(a) <  min(b)
// a <= b: max(a) <= min(b)
// a >  b: max(a) >  min(b)
// a >= b: max(a) >= min(b)
//
// single elements are evaluated as a set of one

template <class T> int operator==(const Set<T> &left_op, const Set<T> &right_op);
template <class T> int operator==(const Set<T> &left_op, const T      &right_op);
template <class T> int operator==(const T      &left_op, const Set<T> &right_op);

template <class T> int operator!=(const Set<T> &left_op, const Set<T> &right_op);
template <class T> int operator!=(const Set<T> &left_op, const T      &right_op);
template <class T> int operator!=(const T      &left_op, const Set<T> &right_op);

template <class T> int operator< (const Set<T> &left_op, const Set<T> &right_op);
template <class T> int operator< (const Set<T> &left_op, const T      &right_op);
template <class T> int operator< (const T      &left_op, const Set<T> &right_op);

template <class T> int operator<=(const Set<T> &left_op, const Set<T> &right_op);
template <class T> int operator<=(const Set<T> &left_op, const T      &right_op);
template <class T> int operator<=(const T      &left_op, const Set<T> &right_op);

template <class T> int operator> (const Set<T> &left_op, const Set<T> &right_op);
template <class T> int operator> (const Set<T> &left_op, const T      &right_op);
template <class T> int operator> (const T      &left_op, const Set<T> &right_op);

template <class T> int operator>=(const Set<T> &left_op, const Set<T> &right_op);
template <class T> int operator>=(const Set<T> &left_op, const T      &right_op);
template <class T> int operator>=(const T      &left_op, const Set<T> &right_op);




template <class T> int testisvnan(const Set<T> &x) { (void) x; return 0; }
template <class T> int testisinf (const Set<T> &x) { (void) x; return 0; }
template <class T> int testispinf(const Set<T> &x) { (void) x; return 0; }
template <class T> int testisninf(const Set<T> &x) { (void) x; return 0; }

template <class T> T max(const Set<T> &a, const Set<T> &b) { static T infres(valpinf()); return ( a.isinfset() || b.isinfset() ) ? infres : max(a.all(),b.all()); }
template <class T> T min(const Set<T> &a, const Set<T> &b) { static T infres(valpinf()); return ( a.isinfset() || b.isinfset() ) ? infres : min(a.all(),b.all()); }

template <class T> const T &max     (const Set<T> &a) { int i = 0; static T infres(valpinf()); return a.isinfset() ? infres : max     (a.all(),i); }
template <class T> const T &min     (const Set<T> &a) { int i = 0; static T infres(valninf()); return a.isinfset() ? infres : min     (a.all(),i); }
template <class T>       T  maxabs  (const Set<T> &a) { int i = 0; static T infres(valpinf()); return a.isinfset() ? infres : maxabs  (a.all(),i); }
template <class T>       T  minabs  (const Set<T> &a) { int i = 0; static T infres(0);         return a.isinfset() ? infres : minabs  (a.all(),i); }
template <class T>       T  sqmaxabs(const Set<T> &a) {            static T infres(valpinf()); return a.isinfset() ? infres : sqmaxabs(a.all());   }
template <class T>       T  sqminabs(const Set<T> &a) {            static T infres(0);         return a.isinfset() ? infres : sqminabs(a.all());   }
template <class T>       T  sum     (const Set<T> &a) {            static T infres(valvnan()); return a.isinfset() ? infres : sum     (a.all());   }
template <class T>       T  prod    (const Set<T> &a) {            static T infres(valvnan()); return a.isinfset() ? infres : prod    (a.all());   }
template <class T>       T  Prod    (const Set<T> &a) {            static T infres(valvnan()); return a.isinfset() ? infres : Prod    (a.all());   }
template <class T>       T  mean    (const Set<T> &a) {            static T infres(valvnan()); return a.isinfset() ? infres : mean    (a.all());   }
template <class T> const T &median  (const Set<T> &a) { int i = 0; static T infres(valvnan()); return a.isinfset() ? infres : median  (a.all(),i); }

template <class T> Set<T> &setident (Set<T> &a) { return a.ident();  }
template <class T> Set<T> &setzero  (Set<T> &a) { return a.zero();   }
template <class T> Set<T> &setposate(Set<T> &a) { return a.posate(); }
template <class T> Set<T> &setnegate(Set<T> &a) { return a.negate(); }
template <class T> Set<T> &setconj  (Set<T> &a) { return a.conj();   }
template <class T> Set<T> &setrand  (Set<T> &a) { return a.rand();   }

template <class S> double abs2  (const Set<S> &a)           { return a.isinfset() ? valpinf() : sqrt(norm2(a));      }
template <class S> double abs1  (const Set<S> &a)           { return a.isinfset() ? valpinf() : norm1(a);            }
template <class S> double absp  (const Set<S> &a, double p) { return a.isinfset() ? valpinf() : pow(normp(a,p),1/p); }
template <class S> double absinf(const Set<S> &a)           { return a.size() ? 1 : 0;                               }
template <class S> double abs0  (const Set<S> &a)           { return a.size() ? 1 : 0;                               }

template <class S> double norm2(const Set<S> &a)         { return a.isinfset() ? valpinf() : a.size(); }
template <class S> double norm1(const Set<S> &a)         { return a.isinfset() ? valpinf() : a.size(); }
template <class S> double normp(const Set<S> &a, double) { return a.isinfset() ? valpinf() : a.size(); }

template <class T> double &innerProduct       (double &res, const Set<T> &a, const Set<T> &b) { return twoProduct(res,a,b); }
template <class T> double &innerProductRevConj(double &res, const Set<T> &a, const Set<T> &b) { return twoProduct(res,a,b); }

template <class T> double &oneProduct(double &res, const Set<T> &a)
{
    res = a.isinfset() ? valpinf() : a.size();

    return res;
}

template <class T> double &twoProduct(double &res, const Set<T> &a, const Set<T> &b)
{
    res = 0;

    if ( a.isinfset() && b.isinfset() )
    {
        res = valpinf();
    }

    else if ( a.isinfset() )
    {
        oneProduct(res,b);
    }

    else if ( b.isinfset() )
    {
        oneProduct(res,a);
    }

    else if ( a.size() && b.size() )
    {
        for ( int i = 0 ; i < a.size() ; ++i )
        {
            res += b.contains((a.all())(i));
        }
    }

    return res;
}

template <class T> double &threeProduct(double &res, const Set<T> &a, const Set<T> &b, const Set<T> &c)
{
    res = 0;

    if ( a.isinfset() && b.isinfset() && c.isinfset() )
    {
        res = valpinf();
    }

    else if ( a.isinfset() )
    {
        twoProduct(res,b,c);
    }

    else if ( b.isinfset() )
    {
        twoProduct(res,a,c);
    }

    else if ( c.isinfset() )
    {
        twoProduct(res,a,b);
    }

    else if ( a.size() && b.size() && c.size() )
    {
        for ( int i = 0 ; i < a.size() ; ++i )
        {
            res += ( b.contains((a.all())(i)) && c.contains((a.all())(i)) );
        }
    }

    return res;
}

template <class T> double &fourProduct(double &res, const Set<T> &a, const Set<T> &b, const Set<T> &c, const Set<T> &d)
{
    res = 0;

    if ( a.isinfset() && b.isinfset() && c.isinfset() && d.isinfset() )
    {
        res = valpinf();
    }

    else if ( a.isinfset() )
    {
        threeProduct(res,b,c,d);
    }

    else if ( b.isinfset() )
    {
        threeProduct(res,a,c,d);
    }

    else if ( c.isinfset() )
    {
        threeProduct(res,a,b,d);
    }

    else if ( d.isinfset() )
    {
        threeProduct(res,a,b,c);
    }

    else if ( a.size() && b.size() && c.size() && d.size() )
    {
        for ( int i = 0 ; i < a.size() ; ++i )
        {
            res += ( b.contains((a.all())(i)) && c.contains((a.all())(i)) && d.contains((a.all())(i)) );
        }
    }

    return res;
}

template <class T> double &mProduct(double &res, int m, const Set<T> *a)
{
    res = 0;

    if ( m && (*a).size() )
    {
        if ( (a[0]).isinfset() )
        {
            if ( m == 1 )
            {
                res = valpinf();
            }

            else
            {
                mProduct(res,--m,++a);
            }
        }

        else
        {
            for ( int i = 0 ; i < (a[0]).size() ; ++i )
            {
                int tmpres = 1;

                for ( int j = 1 ; j < m ; ++j )
                {
                    if ( !(a[j]).size() )
                    {
                        res = 0.0;
                        return res;
                    }

                    if ( !(a[j]).contains((a.all())(i)) )
                    {
                        tmpres = 0;
                        break;
                    }
                }

                res += tmpres;
            }
        }
    }

    return res;
}



// Mathematical operator overloading

template <class T> Set<T> operator+(const Set<T> &left_op) { Set<T> res(left_op); return res.posate(); }
template <class T> Set<T> operator-(const Set<T> &left_op) { Set<T> res(left_op); return res.negate(); }

template <class T> Set<Vector<T> > operator*(const T &left_op, const Set<T> &right_op) { Set<T> ll(left_op);  return ll*right_op; }
template <class T> Set<Vector<T> > operator*(const Set<T> &left_op, const T &right_op) { Set<T> rr(right_op); return left_op*rr;  }

template <class T> Set<T> operator+(const Set<T> &left_op, const Set<T> &right_op) { Set<T> res(left_op); return res += right_op; }
template <class T> Set<T> operator+(const T      &left_op, const Set<T> &right_op) { Set<T> ll(left_op);  return ll+right_op;     }
template <class T> Set<T> operator+(const Set<T> &left_op, const T      &right_op) { Set<T> rr(right_op); return left_op+rr;      }

template <class T> Set<T> operator-(const Set<T> &left_op, const Set<T> &right_op) { Set<T> res(left_op); return res -= right_op; }
template <class T> Set<T> operator-(const Set<T> &left_op, const T      &right_op) { Set<T> rr(right_op); return left_op-rr;      }

template <class T> Set<T> &operator+=(Set<T> &left_op, const T &right_op) { Set<T> rr(right_op); return left_op += rr; }
template <class T> Set<T> &operator-=(Set<T> &left_op, const T &right_op) { Set<T> rr(right_op); return left_op -= rr; }

template <class T> Set<Vector<T> > operator*(const Set<T> &left_op, const Set<T> &right_op)
{
    Set<Vector<T> > res(left_op.isinfset() || right_op.isinfset(), (!left_op.isnegset() != !right_op.isnegset()));

    if ( !res.isinfset() )
    {
        if ( left_op.size() && right_op.size() )
        {
            for ( int i = 0 ; i < left_op.size() ; ++i )
            {
                for ( int j = 0 ; j < right_op.size() ; ++j )
                {
                    Vector<T> temp(2);

                    temp("&",0) = (left_op.all())(i);
                    temp("&",1) = (right_op.all())(j);

                    res.add(temp);
                }
            }
        }
    }

    return res;
}

template <class T> Set<T> &operator+=(Set<T> &left_op, const Set<T> &right_op)
{
    // !left_op.isneg() && !right_op.isneg(): left_op union right_op
    // !left_op.isneg() &&  right_op.isneg(): left_up \ right_op
    //  left_op.isneg() && !right_op.isneg(): right_op \ left_op
    //  left_op.isneg() &&  right_op.isneg(): -( left_op union right_op )

    if ( !left_op.isnegset() && !right_op.isnegset() )
    {
        // left_op union right_op

        if ( left_op.isinfset() )
        {
            ;
        }

        else if ( right_op.isinfset() )
        {
            left_op = right_op;
        }

        else
        {
            for ( int i = 0 ; i < right_op.size() ; ++i )
            {
                if ( !left_op.contains((right_op.all())(i)) )
                {
                    left_op.add((right_op.all())(i));
                }
            }
        }
    }

    else if ( !left_op.isnegset() && right_op.isnegset() )
    {
        // left_op \ right_op

        if ( right_op.isinfset() )
        {
            left_op.zero();
        }

        else if ( left_op.isinfset() )
        {
            ;
        }

        else
        {
            for ( int i = (left_op.size())-1 ; i >= 0 ; --i )
            {
                if ( !(right_op.contains((left_op.all())(i))) )
                {
                    left_op.remove((left_op.all())(i));
                }
            }
        }
    }

    else if ( left_op.isnegset() && !right_op.isnegset() )
    {
        // right_op \ left_op

        if ( left_op.isinfset() )
        {
            left_op.zero();
        }

        else if ( right_op.isinfset() )
        {
            left_op = right_op;
        }

        else
        {
            left_op = (right_op+left_op);
        }
    }

    else if ( left_op.isnegset() && right_op.isnegset() )
    {
        // -(left_op union right_op)

        if ( left_op.isinfset() )
        {
            ;
        }

        else if ( right_op.isinfset() )
        {
            left_op = right_op;
        }

        else
        {
            for ( int i = 0 ; i < right_op.size() ; ++i )
            {
                if ( !left_op.contains((right_op.all())(i)) )
                {
                    left_op.add((right_op.all())(i));
                }
            }
        }

        // left_op is already negated
    }

    return left_op;
}

template <class T> Set<T> &operator-=(Set<T> &left_op, const Set<T> &right_op)
{
    // !left_op.isneg() && !right_op.isneg(): left_op \ right_op
    // !left_op.isneg() &&  right_op.isneg(): left_up union right_op
    // !left_op.isneg() && !right_op.isneg(): -( left_op union right_op )
    // !left_op.isneg() && !right_op.isneg(): right_op \ left_op

    if ( !left_op.isnegset() && !right_op.isnegset() )
    {
        // left_op \ right_op

        if ( right_op.isinfset() )
        {
            left_op.zero();
        }

        else if ( left_op.isinfset() )
        {
            ;
        }

        else
        {
            for ( int i = (left_op.size())-1 ; i >= 0 ; --i )
            {
                if ( !(right_op.contains((left_op.all())(i))) )
                {
                    left_op.remove((left_op.all())(i));
                }
            }
        }
    }

    else if ( !left_op.isnegset() && right_op.isnegset() )
    {
        // left_op union right_op

        if ( left_op.isinfset() )
        {
            ;
        }

        else if ( right_op.isinfset() )
        {
            left_op = right_op;
            left_op.negate();
        }

        else
        {
            for ( int i = 0 ; i < right_op.size() ; ++i )
            {
                if ( !left_op.contains((right_op.all())(i)) )
                {
                    left_op.add((right_op.all())(i));
                }
            }
        }
    }

    else if ( left_op.isnegset() && !right_op.isnegset() )
    {
        // -(left_op union right_op)

        if ( left_op.isinfset() )
        {
            ;
        }

        else if ( right_op.isinfset() )
        {
            left_op = right_op;
            left_op.negate();
        }

        else
        {
            for ( int i = 0 ; i < right_op.size() ; ++i )
            {
                if ( !left_op.contains((right_op.all())(i)) )
                {
                    left_op.add((right_op.all())(i));
                }
            }
        }
    }

    else if ( left_op.isnegset() && right_op.isnegset() )
    {
        // right_op \ left_op

        if ( left_op.isinfset() )
        {
            left_op.zero();
        }

        else if ( right_op.isinfset() )
        {
            left_op = right_op;
            left_op.negate();
        }

        else
        {
            Set<T> rr(right_op);

            rr.negate();

            left_op = (right_op+left_op); // recall left_op is negated already, so this will do right_op \ left_op, then assign and fix the sign
        }
    }

    return left_op;
}

// Logical operator overloading

template <class T> int operator==(const Set<T> &left_op, const Set<T> &right_op)
{
    if ( left_op.size() != right_op.size() )
    {
        return 0;
    }

    if ( left_op.isinfset() && right_op.isinfset() )
    {
        return 1;
    }

    for ( int i = 0 ; i < left_op.size() ; ++i )
    {
        if ( !(right_op.contains((left_op.all())(i))) )
        {
            return 0;
        }
    }

    return 1;
}

template <class T> int operator==(const Set<T> &left_op, const T      &right_op) { return ( ( left_op.size()  == 1 ) && left_op. contains(right_op) ); }
template <class T> int operator==(const T      &left_op, const Set<T> &right_op) { return ( ( right_op.size() == 1 ) && right_op.contains(left_op)  ); }

template <class T> int operator!=(const Set<T> &left_op, const Set<T> &right_op) { return !( left_op == right_op ); }
template <class T> int operator!=(const Set<T> &left_op, const T      &right_op) { return !( left_op == right_op ); }
template <class T> int operator!=(const T      &left_op, const Set<T> &right_op) { return !( left_op == right_op ); }

template <class T> int operator< (const Set<T> &left_op, const Set<T> &right_op)
{
    if ( !(left_op.size()) || !(right_op.size()) )
    {
        return ( left_op.size() == right_op.size() );
    }

    return max(left_op) < min(right_op);
}

template <class T> int operator< (const Set<T> &left_op, const T &right_op)
{
    if ( !(left_op.size()) )
    {
        return 0;
    }

    return max(left_op) < right_op;
}

template <class T> int operator< (const T &left_op, const Set<T> &right_op)
{
    if ( !(right_op.size()) )
    {
        return 0;
    }

    return left_op < min(right_op);
}

template <class T> int operator<=(const Set<T> &left_op, const Set<T> &right_op)
{
    if ( !(left_op.size()) || !(right_op.size()) )
    {
        return ( left_op.size() == right_op.size() );
    }

    return max(left_op) <= min(right_op);
}

template <class T> int operator<=(const Set<T> &left_op, const T      &right_op)
{
    if ( !(left_op.size()) )
    {
        return 0;
    }

    return max(left_op) <= right_op;
}

template <class T> int operator<=(const T      &left_op, const Set<T> &right_op)
{
    if ( !(right_op.size()) )
    {
        return 0;
    }

    return left_op <= min(right_op);
}

template <class T> int operator> (const Set<T> &left_op, const Set<T> &right_op) { return right_op <  left_op; }
template <class T> int operator> (const Set<T> &left_op, const T      &right_op) { return right_op <  left_op; }
template <class T> int operator> (const T      &left_op, const Set<T> &right_op) { return right_op <  left_op; }
template <class T> int operator>=(const Set<T> &left_op, const Set<T> &right_op) { return right_op <= left_op; }
template <class T> int operator>=(const Set<T> &left_op, const T      &right_op) { return right_op <= left_op; }
template <class T> int operator>=(const T      &left_op, const Set<T> &right_op) { return right_op <= left_op; }

// Conversion from strings

template <class T> Set<T> &atoSet(Set<T> &dest, const std::string &src)
{
    std::istringstream srcmod(src);

    srcmod >> dest;

    return dest;
}

// Stream operators

template <class T>
std::ostream &operator<<(std::ostream &output, const Set<T> &src)
{
    // This is just a copy of the vector streamer with some mods

    int size = src.size();

    if ( src.isnegset() )
    {
        output << "-";
    }

    if ( src.isinfset() )
    {
        output << "{*}";
    }

    else
    {
        output << "{ ";

	for ( int i = 0 ; i < size ; ++i )
	{
	    if ( i < size-1 )
	    {
                output << (src.all())(i) << " ; ";
	    }

	    else
	    {
                output << (src.all())(i);
	    }
	}

        output << "  }";
    }

    return output;
}

template <class T>
std::istream &operator>>(std::istream &input, Set<T> &dest)
{
    // This is just a copy of the vector streamer with some mods

    (dest.contents).resize(0);
    dest.isinf = false;

    char tt;

    while ( isspace(input.peek()) )
    {
	input.get(tt);
    }

    input.get(tt);

    dest.isneg = false;

    if ( tt == '-' )
    {
        dest.isneg = true;
        input.get(tt);
    }

    StrucAssert( tt == '{' );

    if ( input.peek() == '*' )
    {
        input.get(tt);
        StrucAssert( tt == '*' );
        input.get(tt);
        StrucAssert( tt == '}' );

        dest.isinf = true;
    }

    else
    {
        int size = 0;

        while ( 1 )
        {
            while ( ( isspace(input.peek()) ) || ( input.peek() == ';' ) || ( input.peek() == ',' ) )
            {
                input.get(tt);
            }

            if ( input.peek() == '}' )
            {
                input.get(tt);

                break;
            }

            (dest.contents).add(size);
            input >> (dest.contents)("&",size);

            ++size;
        }

        //dest.removeDuplicates();
    }

    return input;
}

template <class T>
std::istream &streamItIn(std::istream &input, Set<T> &dest, int processxyzvw)
{
    // This is just a copy of the vector streamItIn with some mods

    (dest.contents).resize(0);
    dest.isinf = false;

    char tt;

    while ( isspace(input.peek()) )
    {
	input.get(tt);
    }

    input.get(tt);

    dest.isneg = false;

    if ( tt == '-' )
    {
        dest.isneg = true;
        input.get(tt);
    }

    NiceAssert( tt == '{' );

    if ( input.peek() == '*' )
    {
        input.get(tt);
        StrucAssert( tt == '*' );
        input.get(tt);
        StrucAssert( tt == '}' );

        dest.isinf = true;
    }

    else
    {
        int size = 0;

        while ( 1 )
        {
            while ( ( isspace(input.peek()) ) || ( input.peek() == ';' ) || ( input.peek() == ',' ) )
            {
                input.get(tt);
            }

            if ( input.peek() == '}' )
            {
                input.get(tt);

                break;
            }

            (dest.contents).add(size);
            streamItIn(input,(dest.contents)("&",size),processxyzvw);

            ++size;
        }

        //dest.removeDuplicates();
    }

    return input;
}

#endif

