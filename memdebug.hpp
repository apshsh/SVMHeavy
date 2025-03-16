
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

#ifndef _memdebug_h
#define _memdebug_h

#include <memory>
#include <string>
#include "niceassert.hpp"
#ifdef DEBUG_MEM
#include "basefn.hpp"
#endif

// #ifndef DEBUG_MEM - for full memory debugging
// #ifndef DEBUG_MEM_CHEAP - for rudimentary memory debugging

// MEMNEW(a,b)            macro version of a = new b
// MEMNEWARRAY(a,b,c)     macro version of a = new b[c]
// MEMNEWVOID(a,b)        macro version of a = (void *) new b
// MEMNEWVOIDARRAY(a,b,c) macro version of a = (void *) new b[c]
// MEMDEL(a)              macro version of delete a
// MEMDELARRAY(a)         macro version of delete[] a
// MEMDELVOID(a)          macro version of delete a
// MEMDELVOIDARRAY(a)     macro version of delete[] a


#ifndef DEBUG_MEM
#ifndef DEBUG_MEM_CHEAP

#define MEMNEW(_a_,_b_)          _a_ = new _b_
#define MEMNEWARRAY(_a_,_b_,_c_) _a_ = new _b_[_c_]

#define MEMNEWVOID(_a_,_b_)          _a_ = (void *) new _b_
#define MEMNEWVOIDARRAY(_a_,_b_,_c_) _a_ = (void *) new _b_[_c_]

#define MEMDEL(_a_)      delete   _a_
#define MEMDELARRAY(_a_) delete[] _a_

#define MEMDELVOID(_a_)      delete   _a_
#define MEMDELVOIDARRAY(_a_) delete[] _a_

#endif
#endif








#ifndef DEBUG_MEM
#ifdef DEBUG_MEM_CHEAP

extern size_t _global_alloccnt;
extern size_t _global_maxalloccnt;

#define MEMCUTPNT 16384

#define MEMNEW(_a_,_b_) \
{ \
    ++_global_alloccnt; \
\
    if ( _global_alloccnt/MEMCUTPNT > _global_maxalloccnt ) \
    { \
        _global_maxalloccnt = _global_alloccnt/MEMCUTPNT; \
        errstream() << "!"  << _global_maxalloccnt << "," << PLACEMARK << "!"; \
    } \
} _a_ = new _b_

#define MEMNEWARRAY(_a_,_b_,_c_) \
{ \
    ++_global_alloccnt; \
\
    if ( _global_alloccnt/MEMCUTPNT > _global_maxalloccnt ) \
    { \
        _global_maxalloccnt = _global_alloccnt/MEMCUTPNT; \
        errstream() << "!," << PLACEMARK << _global_maxalloccnt << "!"; \
    } \
} _a_ = new _b_[_c_]

#define MEMNEWVOID(_a_,_b_) \
{ \
    ++_global_alloccnt; \
\
    if ( _global_alloccnt/MEMCUTPNT > _global_maxalloccnt ) \
    { \
        _global_maxalloccnt = _global_alloccnt/MEMCUTPNT; \
        errstream() << "!," << _global_maxalloccnt << "," << PLACEMARK << "!"; \
    } \
} _a_ = (void *) new _b_

#define MEMNEWVOIDARRAY(_a_,_b_,_c_) \
{ \
    ++_global_alloccnt; \
\
    if ( _global_alloccnt/MEMCUTPNT > _global_maxalloccnt ) \
    { \
        _global_maxalloccnt = _global_alloccnt/MEMCUTPNT; \
        errstream() << "!," << _global_maxalloccnt << PLACEMARK << "!"; \
    } \
} _a_ = (void *) new _b_[_c_]

#define MEMDEL(_a_) \
{ \
    if ( _global_alloccnt > 0 ) \
    { \
        --_global_alloccnt; \
    } \
\
    if ( _global_alloccnt/MEMCUTPNT < _global_maxalloccnt ) \
    { \
        _global_maxalloccnt = _global_alloccnt/MEMCUTPNT; \
        errstream() << "?_" << _global_maxalloccnt << "_?"; \
    } \
} delete _a_

#define MEMDELARRAY(_a_) \
{ \
    if ( _global_alloccnt > 0 ) \
    { \
        --_global_alloccnt; \
    } \
\
    if ( _global_alloccnt/MEMCUTPNT < _global_maxalloccnt ) \
    { \
        _global_maxalloccnt = _global_alloccnt/MEMCUTPNT; \
        errstream() << "?_" << _global_maxalloccnt << "_?"; \
    } \
} delete[] _a_

#define MEMDELVOID(_a_) \
{ \
    if ( _global_alloccnt > 0 ) \
    { \
        --_global_alloccnt; \
    } \
\
    if ( _global_alloccnt/MEMCUTPNT < _global_maxalloccnt ) \
    { \
        _global_maxalloccnt = _global_alloccnt/MEMCUTPNT; \
        errstream() << "?_" << _global_maxalloccnt << "_?"; \
    } \
} delete _a_

#define MEMDELVOIDARRAY(_a_) \
{ \
    if ( _global_alloccnt > 0 ) \
    { \
        --_global_alloccnt; \
    } \
\
    if ( _global_alloccnt/MEMCUTPNT < _global_maxalloccnt ) \
    { \
        _global_maxalloccnt = _global_alloccnt/MEMCUTPNT; \
        errstream() << "?_" << _global_maxalloccnt << "_?"; \
    } \
} delete[] _a_

#endif
#endif












#ifdef DEBUG_MEM

// Function that keeps track of pointers:
//
// addr is pointer in question
// newdel: 1 if new, 0 of delete, -1 to dump report and cleanup, -2 to dump report without cleanup
// type: 0 is pointer, 1 is array
// size: size of array
//
// return: 0 = success
//         1 = attempt to delete an unallocated error code
//         2 = attempt to delete array with non-array delete
//         3 = attempt to delete non-array with array delete

//int addremoveptr(void *addr, int newdel, int type, int size, const char *desc);

// use addendum = nullptr to remove most recent addition
const char *updatedesc(const char *addendum, size_t &oldallocdesclen);

#define MEMNEW(_a_,_b_) \
{ \
    _a_ = new _b_; \
\
    NiceAssert( _a_ ); \
\
    size_t oldallocdesclen = 0; \
    const char *allocdesc = updatedesc(PLACEMARK,oldallocdesclen); \
\
    int _qq_ = addremoveptr((void *) _a_,1,0,1,allocdesc); \
\
    if ( _qq_ == 1 ) \
    { \
        errstream() << "Delete non-allocated memory"; \
    } \
\
    if ( _qq_ == 2 ) \
    { \
        errstream() << "Delete array with non"; \
    } \
\
    if ( _qq_ == 3 ) \
    { \
        errstream() << "Delete non with array"; \
    } \
\
    QuietAssert( !_qq_ ); \
\
    updatedesc(nullptr,oldallocdesclen); \
}

#define MEMNEWARRAY(_a_,_b_,_c_) \
{ \
    _a_ = new _b_[_c_]; \
\
    NiceAssert( _a_ ); \
\
    size_t oldallocdesclen = 0; \
    const char *allocdesc = updatedesc(PLACEMARK,oldallocdesclen); \
\
    int _qq_ = addremoveptr((void *) _a_,1,1,_c_,allocdesc); \
\
    if ( _qq_ == 1 ) \
    { \
        errstream() << "Delete non-allocated memory"; \
    } \
\
    if ( _qq_ == 2 ) \
    { \
        errstream() << "Delete array with non"; \
    } \
\
    if ( _qq_ == 3 ) \
    { \
        errstream() << "Delete non with array"; \
    } \
\
    QuietAssert( !_qq_ ); \
\
    updatedesc(nullptr,oldallocdesclen); \
}

#define MEMNEWVOID(_a_,_b_) \
{ \
    _a_ = (void *) new _b_; \
\
    NiceAssert( _a_ ); \
\
    size_t oldallocdesclen = 0; \
    const char *allocdesc = updatedesc(PLACEMARK,oldallocdesclen); \
\
    int _qq_ = addremoveptr((void *) _a_,1,0,1,allocdesc); \
\
    if ( _qq_ == 1 ) \
    { \
        errstream() << "Delete non-allocated memory"; \
    } \
\
    if ( _qq_ == 2 ) \
    { \
        errstream() << "Delete array with non"; \
    } \
\
    if ( _qq_ == 3 ) \
    { \
        errstream() << "Delete non with array"; \
    } \
\
    QuietAssert( !_qq_ ); \
\
    updatedesc(nullptr,oldallocdesclen); \
}

#define MEMNEWVOIDARRAY(_a_,_b_,_c_) \
{ \
    _a_ = (void *) new _b_[_c_]; \
\
    QuietAssert( _a_ ); \
\
    size_t oldallocdesclen = 0; \
    const char *allocdesc = updatedesc(PLACEMARK,oldallocdesclen); \
\
    int _qq_ = addremoveptr((void *) _a_,1,1,_c_,allocdesc); \
\
    if ( _qq_ == 1 ) \
    { \
        errstream() << "Delete non-allocated memory"; \
    } \
\
    if ( _qq_ == 2 ) \
    { \
        errstream() << "Delete array with non"; \
    } \
\
    if ( _qq_ == 3 ) \
    { \
        errstream() << "Delete non with array"; \
    } \
\
    QuietAssert( !_qq_ ); \
\
    updatedesc(nullptr,oldallocdesclen); \
}

#define MEMDEL(_a_) \
{ \
  QuietAssert( _a_ ); \
\
  if ( _a_ ) \
  { \
    int _qq_ = addremoveptr((void *) _a_,0,0,0,PLACEMARK); \
\
    if ( _qq_ == 1 ) \
    { \
        errstream() << "Delete non-allocated memory"; \
    } \
\
    if ( _qq_ == 2 ) \
    { \
        errstream() << "Delete array with non"; \
    } \
\
    if ( _qq_ == 3 ) \
    { \
        errstream() << "Delete non with array"; \
    } \
\
    QuietAssert( !_qq_ ); \
\
    if ( !_qq_ ) \
    { \
        delete _a_; \
    } \
  } \
}

#define MEMDELARRAY(_a_) \
{ \
  QuietAssert( _a_ ); \
\
  if ( _a_ ) \
  { \
    int _qq_ = addremoveptr((void *) _a_,0,1,0,PLACEMARK); \
\
    if ( _qq_ == 1 ) \
    { \
        errstream() << "Delete non-allocated memory"; \
    } \
\
    if ( _qq_ == 2 ) \
    { \
        errstream() << "Delete array with non"; \
    } \
\
    if ( _qq_ == 3 ) \
    { \
        errstream() << "Delete non with array"; \
    } \
\
    QuietAssert( !_qq_ ); \
\
    if ( !_qq_ ) \
    { \
        delete[] _a_; \
    } \
 } \
}

//    _a_ = nullptr;

#define MEMDELVOID(_a_) \
{ \
  QuietAssert( _a_ ); \
\
  if ( _a_ ) \
  { \
    int _qq_ = addremoveptr((void *) _a_,0,0,0,PLACEMARK); \
\
    if ( _qq_ == 1 ) \
    { \
        errstream() << "Delete non-allocated memory"; \
    } \
\
    if ( _qq_ == 2 ) \
    { \
        errstream() << "Delete array with non"; \
    } \
\
    if ( _qq_ == 3 ) \
    { \
        errstream() << "Delete non with array"; \
    } \
\
    QuietAssert( !_qq_ ); \
\
    if ( !_qq_ ) \
    { \
        delete _a_; \
    } \
  } \
}

#define MEMDELVOIDARRAY(_a_) \
{ \
  QuietAssert( _a_ ); \
\
  if ( _a_ ) \
  { \
    int _qq_ = addremoveptr((void *) _a_,0,1,0,PLACEMARK); \
\
    if ( _qq_ == 1 ) \
    { \
        errstream() << "Delete non-allocated memory"; \
    } \
\
    if ( _qq_ == 2 ) \
    { \
        errstream() << "Delete array with non"; \
    } \
\
    if ( _qq_ == 3 ) \
    { \
        errstream() << "Delete non with array"; \
    } \
\
    QuietAssert( !_qq_ ); \
\
    if ( !_qq_ ) \
    { \
        delete[] _a_; \
    } \
  } \
}

#endif



#endif
