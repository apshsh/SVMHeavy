
//
// "Nicer" throw/assert macros
//
// Version: split off basefn
// Date: 11/09/2024
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _niceassert_h
#define _nicrassert_h


#include <assert.h>
#include <string>
#ifndef NDEBUG
#include "basefn.hpp"
#endif

// NiceAssert:  "Nice" assert macro.  This one will throw an exception that can be caught.
// StrucAssert: "Nice, structural" assert macro.  Like NiceAssert, but persists even when NDEBUG is defined
// QuietAssert: Like NiceAssert, but exits by recursing the assert.
// NiceThrow:   Like throw for printables, but prints before throwing
// STRTHROW:    Construct std::string and throw this.
//
// Flags: NDEBUG turns of assert and throw macros

#define xS(x) #x
#define xS_(x) xS(x)
#define xS__LINE__ xS_(__LINE__)
#define PLACEMARK "Line " xS__LINE__ " in file " __FILE__
#define S(x) #x
#define S_(x) S(x)
#define S__LINE__ S_(__LINE__)
#define THROWSTRINGDEF(cond) "Assertion " #cond " failed at line " S__LINE__ " in file " __FILE__


#ifdef NDEBUG
#ifdef IS_CPP23
#define NiceAssert( cond ) [[assume(!(cond))]];
#endif
#ifndef IS_CPP23
#define NiceAssert( cond )
#endif
#define NiceThrow( what ) throw(what)
#endif


#ifndef NDEBUG
#define NiceAssert( cond ) \
if ( !(cond) ) \
{ \
    errstream() << THROWSTRINGDEF(cond) << "\n"; \
    throw(THROWSTRINGDEF(cond)); \
}
#define QuietAssert( cond ) \
if ( !(cond) ) \
{ \
    errstream() << THROWSTRINGDEF(cond) << "\n"; \
    assert(cond); \
}
#define NiceThrow( what ) \
{ \
    errstream() << PLACEMARK << " has thrown: " << what << "\n"; \
    throw(what); \
}
#endif

#define STRTHROW(_x_errstr_x_) \
std::string _x_errstring_x_ = _x_errstr_x_; \
throw _x_errstring_x_;


#define StrucAssert( cond ) \
if ( !(cond) ) \
{ \
    errstream() << THROWSTRINGDEF(cond) << "\n"; \
    throw(THROWSTRINGDEF(cond)); \
}





#endif
