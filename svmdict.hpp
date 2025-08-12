
//
// Dictionary class
//
// Version: 7
// Date: 18/09/2019
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _dict_h
#define _dict_h

#include <iostream>
#include "vector.hpp"

typedef std::string dictkey;

template <class K, class T> class Dict;

// Stream operators

template <class K, class T> std::ostream &operator<<(std::ostream &output, const Dict<K,T> &src);
template <class K, class T> std::istream &operator>>(std::istream &input, Dict<K,T> &dest);
template <class K, class T> std::istream &streamItIn(std::istream &input, Dict<K,T> &dest, int processxyzvw = 1);

// Swap function

template <class K, class T> void qswap(      Dict<K,T>  &a,       Dict<K,T>  &b);
template <class K, class T> void qswap(const Dict<K,T> *&a, const Dict<K,T> *&b);
template <class K, class T> void qswap(      Dict<K,T> *&a,       Dict<K,T> *&b);

// The class itself

template <class K, class T>
class Dict
{
    template <class sK, class tT> friend void qswap(Dict<sK,tT> &a, Dict<sK,tT> &b);

    template <class sK, class tT> friend std::ostream &operator<<(std::ostream &output, const Dict<sK,tT> &src);
    template <class sK, class tT> friend std::istream &operator>>(std::istream &input, Dict<sK,tT> &dest);
    template <class sK, class tT> friend std::istream &streamItIn(std::istream &input, Dict<sK,tT> &dest, int processxyzvw);

public:

    // Constructors and Destructors:

    Dict(const Dict<K,T> &src) { assign(src); }
    Dict() { values.resize(1); setzero(values("&",0)); }
    ~Dict() { ; }

    // Assignment:

    Dict<K,T> &operator=(const Dict<K,T> &src) { return assign(src); }
    Dict<K,T> &assign(const Dict<K,T> &src) { values = src.values; keys = src.keys; return *this; }

    // Manipulations:

    Dict<K,T> &zero       (void) { keys.zero(); values.resize(1); setzero(values("&",0)); return *this; }
    Dict<K,T> &ident      (void) { retVector<K> tmpva; (*this)("&",tmpva).ident();        return *this; }
    Dict<K,T> &softzero   (void) { retVector<K> tmpva; (*this)("&",tmpva).zero();         return *this; }
    Dict<K,T> &zeropassive(void) { retVector<K> tmpva; (*this)("&",tmpva).zeropassive();  return *this; }
    Dict<K,T> &posate     (void) { retVector<K> tmpva; (*this)("&",tmpva).posate();       return *this; }
    Dict<K,T> &negate     (void) { retVector<K> tmpva; (*this)("&",tmpva).negate();       return *this; }
    Dict<K,T> &conj       (void) { retVector<K> tmpva; (*this)("&",tmpva).conj();         return *this; }
    Dict<K,T> &rand       (void) { retVector<K> tmpva; (*this)("&",tmpva).rand();         return *this; }

    // Access:

          K &operator()(const char *dummy, const T &i)       { return values(dummy,findaddkey(i)); }
    const K &operator()(                   const T &i) const { return values(      findkey   (i)); }

          Vector<K> &operator()(const char *dummy, retVector<K> &tmp)       { return values(dummy,0,1,size()-1,tmp); }
    const Vector<K> &operator()(                   retVector<K> &tmp) const { return values(      0,1,size()-1,tmp); }

          K &val(const char *dummy, int i)       { return values(dummy,i); }
    const K &val(                   int i) const { return values(      i); }

          T &key(const char *dummy, int i)       { return keys(dummy,i); }
    const T &key(                   int i) const { return keys(      i); }

    const Vector<T> &key(void) const { return keys; }

    // Information:

    int size(void) const { return keys.size(); }
    bool iskeypresent(const T &i) const { return ( findkey(i) < size() ) ? true : false; }

    // Add and remove:

    Dict<K,T> &add(T &i) { setzero((*this)("&",i)); return *this; }
    Dict<K,T> &remove(T &i) { int pos = findkey(i); if ( pos < size() ) { keys.remove(pos); values.remove(pos); } return *this; }

    // Function application:

    Dict<K,T> &applyon(K (*fn)(K))                                      { values.applyon(fn);   return *this; }
    Dict<K,T> &applyon(K (*fn)(const K &))                              { values.applyon(fn);   return *this; }
    Dict<K,T> &applyon(K (*fn)(K, const void *), const void *a)         { values.applyon(fn,a); return *this; }
    Dict<K,T> &applyon(K (*fn)(const K &, const void *), const void *a) { values.applyon(fn,a); return *this; }
    Dict<K,T> &applyon(K &(*fn)(K &))                                   { values.applyon(fn);   return *this; }
    Dict<K,T> &applyon(K &(*fn)(K &, const void *), const void *a)      { values.applyon(fn,a); return *this; }

    // Pre-allocation and allocation strategy:

    void prealloc(int newallocsize)  { keys.prealloc(newallocsize);  values.prealloc(newallocsize+1); }
    void useStandardAllocation(void) { keys.useStandardAllocation(); values.useStandardAllocation();  }
    void useTightAllocation   (void) { keys.useTightAllocation();    values.useTightAllocation();     }
    void useSlackAllocation   (void) { keys.useSlackAllocation();    values.useSlackAllocation();     }

    bool array_norm (void) const { return values.array_norm(); }
    bool array_tight(void) const { return values.array_tight(); }
    bool array_slack(void) const { return values.array_slack(); }

private:
    Vector<K> values;
    Vector<T> keys;

    int findkey(const T &i) const
    {
        int pos = 0;

        for ( pos = 0 ; pos < size() ; ++pos )
        {
            if ( keys(pos) == i )
            {
                break;
            }
        }

        return pos;
    }

    int findaddkey(const T &i)
    {
        int pos = findkey(i);

        if ( pos == size() )
        {
            keys.add(pos);
            keys("&",pos) = i;
            values.add(pos);
            setzero(values("&",pos));
        }

        return pos;
    }
};



// Operations:

template <class K, class T> Dict<K,T> &setident (Dict<K,T> &a) { return a.ident();  }
template <class K, class T> Dict<K,T> &setzero  (Dict<K,T> &a) { return a.zero();   }
template <class K, class T> Dict<K,T> &setposate(Dict<K,T> &a) { return a.posate(); }
template <class K, class T> Dict<K,T> &setnegate(Dict<K,T> &a) { return a.negate(); }
template <class K, class T> Dict<K,T> &setconj  (Dict<K,T> &a) { return a.conj();   }
template <class K, class T> Dict<K,T> &setrand  (Dict<K,T> &a) { return a.rand();   }

template <class K, class T> Dict<K,T> &postProInnerProd(Dict<K,T> &a) { return a; }

template <class K, class T> Dict<K,T> *&setident (Dict<K,T> *&a) { NiceThrow("wefgknf"); return a; }
template <class K, class T> Dict<K,T> *&setzero  (Dict<K,T> *&a) {                       return a = nullptr; }
template <class K, class T> Dict<K,T> *&setposate(Dict<K,T> *&a) {                       return a; }
template <class K, class T> Dict<K,T> *&setnegate(Dict<K,T> *&a) { NiceThrow("wefgknf"); return a; }
template <class K, class T> Dict<K,T> *&setconj  (Dict<K,T> *&a) { NiceThrow("wefgknf"); return a; }
template <class K, class T> Dict<K,T> *&setrand  (Dict<K,T> *&a) { NiceThrow("wefgknf"); return a; }

template <class K, class T> Dict<K,T> *&postProInnerProd(Dict<K,T> *&a) { return a; }

template <class K, class T> const Dict<K,T> *&setident (const Dict<K,T> *&a) { NiceThrow("wefgknf"); return a; }
template <class K, class T> const Dict<K,T> *&setzero  (const Dict<K,T> *&a) {                       return a = nullptr; }
template <class K, class T> const Dict<K,T> *&setposate(const Dict<K,T> *&a) {                       return a; }
template <class K, class T> const Dict<K,T> *&setnegate(const Dict<K,T> *&a) { NiceThrow("wefgknf"); return a; }
template <class K, class T> const Dict<K,T> *&setconj  (const Dict<K,T> *&a) { NiceThrow("wefgknf"); return a; }
template <class K, class T> const Dict<K,T> *&setrand  (const Dict<K,T> *&a) { NiceThrow("wefgknf"); return a; }

template <class K, class T> const Dict<K,T> *&postProInnerProd(const Dict<K,T> *&a) { return a; }

// Operators

template <class K, class T> int operator==(const Dict<K,T> &left_op, const Dict<K,T> &right_op);
template <class K, class T> int operator==(const Dict<K,T> &left_op, const T         &right_op) { return left_op() == right_op;   }
template <class K, class T> int operator==(const T         &left_op, const Dict<K,T> &right_op) { return left_op   == right_op(); }

template <class K, class T> int operator!=(const Dict<K,T> &left_op, const Dict<K,T> &right_op) { return left_op   != right_op;   }
template <class K, class T> int operator!=(const Dict<K,T> &left_op, const T         &right_op) { return left_op() != right_op;   }
template <class K, class T> int operator!=(const T         &left_op, const Dict<K,T> &right_op) { return left_op   != right_op(); }





























template <class K, class T>
void qswap(Dict<K,T> &a, Dict<K,T> &b)
{
    qswap(a.keys,b.keys);
    qswap(a.values,b.values);
}

template <class K, class T>
void qswap(const Dict<K,T> *&a, const Dict<K,T> *&b)
{
    const Dict<K,T> *x(a); a = b; b = x;
}

template <class K, class T>
void qswap(Dict<K,T> *&a, Dict<K,T> *&b)
{
    Dict<K,T> *x(a); a = b; b = x;
}

template <class K, class T>
int operator==(const Dict<K,T> &left_op, const Dict<K,T> &right_op)
{
    if ( left_op.size() != right_op.size() )
    {
        return 0;
    }

    for ( int pos = 0 ; pos < left_op.size() ; ++pos )
    {
        if ( right_op(left_op.key(pos)) != left_op.val(pos) )
        {
            return 0;
        }
    }

    return 1;
}











// Just... believe this is necessary. c++ is exhausting sometimes

template <class T> bool isTaString(const T &dummy);
template <> inline bool isTaString(const std::string &dummy);

template <class T> bool isTaString(const T &dummy)
{
    (void) dummy;
    return false;
}

template <> inline bool isTaString(const std::string &dummy)
{
    (void) dummy;
    return true;
}

template <class K, class T>
std::ostream &operator<<(std::ostream &output, const Dict<K,T> &src)
{
    int size = src.size();
    static T temp;
    static std::string quoteish = isTaString(temp) ? "\"" : "";

    output << "{{ ";

    for ( int i = 0 ; i < size ; ++i )
    {
        if ( i < size-1 )
        {
            output << quoteish << (src.keys)(i) << quoteish << " : " << (src.values)(i) << " ; ";
        }

        else
        {
            output << quoteish << (src.keys)(i) << quoteish << " : " << (src.values)(i);
        }
    }

    output << "  }}";

    return output;
}

template <class K, class T>
std::istream &operator>>(std::istream &input, Dict<K,T> &dest)
{
    (dest.values).resize(0);
    (dest.keys).resize(0);

    std::string dummy;
    char tt;
    static T temp;
    static bool stringcase = isTaString(temp);

    while ( isspace(input.peek()) )
    {
	input.get(tt);
    }

    input.get(tt);
    NiceAssert( tt == '{' );
    input.get(tt);
    NiceAssert( tt == '{' );

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
            input.get(tt);

            break;
        }

        (dest.values).add(size);
        (dest.keys).add(size);

        if ( stringcase )
        {
            std::string buffer;
            input >> buffer;
            NiceAssert( buffer.size() >= 2 );
            NiceAssert( buffer[0] == '\"' );
            NiceAssert( buffer[buffer.size()-2] == '\"' );
            (dest.keys)("&",size) = buffer.substr(1,buffer.size()-2);
        }

        else
        {
            input >> (dest.keys)("&",size);
        }

        input >> dummy;
        NiceAssert( dummy == ":" );
        input >> (dest.values)("&",size);

        ++size;
    }

    return input;
}

template <class K, class T>
std::istream &streamItIn(std::istream &input, Dict<K,T> &dest, int processxyzvw)
{
    (dest.values).resize(0);
    (dest.keys).resize(0);

    std::string dummy;
    char tt;
    static T temp;
    static bool stringcase = isTaString(temp);

    while ( isspace(input.peek()) )
    {
	input.get(tt);
    }

    input.get(tt);
    NiceAssert( tt == '{' );
    input.get(tt);
    NiceAssert( tt == '{' );

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
            input.get(tt);

            break;
        }

        (dest.values).add(size);
        (dest.keys).add(size);

        if ( stringcase )
        {
            std::string buffer;
            input >> buffer;
            NiceAssert( buffer.size() >= 2 );
            NiceAssert( buffer[0] == '\"' );
            NiceAssert( buffer[buffer.size()-2] == '\"' );
            (dest.keys)("&",size) = buffer.substr(1,buffer.size()-2);
        }

        else
        {
            input >> (dest.keys)("&",size);
        }

        input >> dummy;
        NiceAssert( dummy == ":" );
        streamitin(input,(dest.values)("&",size),processxyzvw);

        ++size;
    }

    return input;
}
















#endif
