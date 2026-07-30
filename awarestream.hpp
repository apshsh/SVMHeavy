
//
// Very slightly smarter stream class
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

// What is it: a vector with push and pop operations added on top
// and some other stuff.  It also works as istream/ostream.

#ifndef _awarestream_h
#define _awarestream_h

#include <iostream>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <time.h>
#include <ctype.h>
#include <string.h>
#include <string>
#include <math.h>
#include <streambuf>
#include "clockbase.hpp"
#include "basefn.hpp"
#include "memdebug.hpp"
#include "niceassert.hpp"





inline char *strncpy_safe_b(char *dest, const char *src, size_t len);
inline char *strncpy_safe_b(char *dest, const char *src, size_t len)
{
    strncpy(dest,src,len);
    dest[len] = '\0';
    return dest;
}



// Rudimentary string fifo buffer

class stringfifo;
class stringfifo
{
public:
    explicit stringfifo(void) : nodeval(nullptr), next(nullptr), killflag(0) { ; }

    ~stringfifo(void)
    {
        if ( nodeval ) { MEMDELARRAY(nodeval); nodeval = nullptr; }
        if ( next    ) { MEMDEL(next);         next    = nullptr; }
    }

    // Push string onto fifo.  Note that this ignores the kill flag
    //
    // Success return: killflag (0 or 1)
    // Fail return: -killflag-2
    //              (-2 means empty fifo, so can just wait for it to fill)
    //
    // The return value is used, so don't redefine behaviour

    int push(const std::string &src) svmvolatile
    {
        int ires = killflag;

        std::stringstream ss(src);
        std::string partsrc;

        while ( ss.good() && !ires )
        {
            ss >> partsrc;
            if ( partsrc.length() ) { ires |= push(partsrc.c_str(),(size_t) partsrc.length()); }
        }

        return ires;
    }

    // Pop string off fifo.  Note that this ignores the kill flag
    //
    // Success return: killflag (0 or 1)
    // Fail return: -killflag-2
    //
    // The return value is used, so don't redefine behaviour

    int pop(std::string &res) svmvolatile
    {
        // nodeval is typically volatile char * volatile
        // this makes it tricky to cast away volatility for assignment.
        // note however that this code is only accessed by one thread at
        // a time, so the following hack should work fine as nodeval
        // can't change midway through.

        NiceAssert( !next || next->next || nodeval );

        int ires = killflag;

        if      ( !next         ) { ires = -ires-2;                                                                                           }
        else if ( !(next->next) ) { res = (char *) ((void *) nodeval); MEMDELARRAY(nodeval); nodeval = nullptr; MEMDEL(next); next = nullptr; }
        else                      { ires = next->pop(res);                                                                                    }

        return ires;
    }

    // Returns: -1: kill flag is set
    //           0: fifo is empty
    //          >1: number of strings on fifo

    int size(void) svmvolatile
    {
        int res = 0;

        if      ( killflag ) { res = -1;               }
        else if ( next     ) { res = (next->size())+1; }

        return res;
    }

    // Set/unset "kill flag"

    void   setkillflag(void) svmvolatile { killflag = 1; if ( next ) { next->  setkillflag(); } }
    void unsetkillflag(void) svmvolatile { killflag = 0; if ( next ) { next->unsetkillflag(); } }

private:
    svmvolatile char *nodeval; // assumed empty if next = nullptr;
    svmvolatile stringfifo *next;
    svmvolatile int killflag;

    explicit stringfifo(int _killflag) : nodeval(nullptr), next(nullptr), killflag(_killflag) { }

    int push(const char *src, size_t len) svmvolatile
    {
        int ires = killflag;

        stringfifo *newnext; MEMNEW(newnext,stringfifo(killflag)); // need to pass killflag

        newnext->next    = next;
        newnext->nodeval = nodeval;

        next    = newnext;
        MEMNEWARRAY(nodeval,char,len+1);

        if ( !nodeval ) { ires = -ires-2;                                                                    }
        else            { for ( size_t i = 0 ; i < len ; ++i ) { nodeval[i] = src[i]; } nodeval[len] = '\0'; }

        return ires;
    }
};

// Rudimentary sparse vector of string fifos

class fifolist;
class fifolist
{
public:
    explicit fifolist(void) : fifonum(0), thisfifo(), next(nullptr) { ; }
    ~fifolist(void) { if ( next ) { MEMDEL(next); next = nullptr; } }

    // push: push string onto given fifo
    // pop: pop string off given fifo
    // size: return size of fifo
    // setkillflag: set kill flag for given fifo

    int push(size_t num, const std::string &src) svmvolatile { return getnum(num).push(src); }
    int pop (size_t num,       std::string &res) svmvolatile { return getnum(num).pop (res); }

    int size(size_t num)         svmvolatile { return getnum(num).size(); }
    void setkillflag(size_t num) svmvolatile { getnum(num).setkillflag(); }

    // set kill flag for all

    void setkillflag(void) svmvolatile
    {
        thisfifo.setkillflag();
        if ( next ) { next->setkillflag(); }
    }

    // return number of fifos in sparse vector

    int indsize(void) svmvolatile
    {
        if ( next ) { return 1+next->indsize(); }
        return 0;
    }

private:
    svmvolatile size_t fifonum;
    svmvolatile stringfifo thisfifo;
    svmvolatile fifolist *next;

    // This constructor is only used internally

    explicit fifolist(size_t num) : fifonum(num), thisfifo(), next(nullptr) { }

    // Retrieve element in fifo, assigning first if need be

    svmvolatile stringfifo &getnum(size_t num) svmvolatile
    {
        NiceAssert( num >= fifonum );

        svmvolatile stringfifo *res = nullptr;

        if ( fifonum == num ) { res = &thisfifo; }
        else                  { if      ( !next               ) { MEMNEW(next,fifolist(num)); }
                                else if ( next->fifonum > num ) { svmvolatile fifolist *afternext = next;
                                                                  MEMNEW(next,fifolist(num));
                                                                  next->next = afternext; }
                                res = &(next->getnum(num)); }

        return *res;
    }

};













class awarestream;

// Examples of stream binding:
//
// awarestream sbuf(...);
//
// std::istream sin(&sbuf);
// std::ostream sout(&sbuf);
//
// std::string a;
// gentype b;
//
// sout << a;
// sin >> b;

#define FIFO_BASESLEEP         100000
#define FIFO_ADDSLEEP_RAND     10000

// Note that we inherit from streambuf to make this trivially streamable

class awarestream : public std::streambuf
{
public:

    // Constructor for standard istream type
    //
    // _istr       = stream that data will be coming from
    // _ideletable = 0 if stream is to be left open always (eg std::cin)
    //               1 if stream should be closed once done (eg filestream)
    //
    // (_std default was std::cin)

    awarestream(std::istream *_istr = nullptr, int _ideletable = 0) : srcsel(_istr ? 1 : 0),
                                                                      istr(_istr),
                                                                      ostr(nullptr),
                                                                      ideletable(_istr ? _ideletable : 0),
                                                                      odeletable(0),
                                                                      fifoind(-1) { ; }

    awarestream(const char *, std::ostream *_ostr = nullptr, int _odeletable = 0): srcsel(1),
                                                                                   istr(nullptr),
                                                                                   ostr(_ostr ? _ostr : &outstream()),
                                                                                   ideletable(0),
                                                                                   odeletable(_odeletable),
                                                                                   fifoind(-1) { ; }

    awarestream(std::istream *_istr, std::ostream *_ostr, int _ideletable, int _odeletable) : srcsel(1),
                                                                                              istr(_istr),
                                                                                              ostr(_ostr),
                                                                                              ideletable(_ideletable),
                                                                                              odeletable(_odeletable),
                                                                                              fifoind(-1) { ; }

    // Constructor for shared string fifo (cross-thread comms)
    //
    // _fifoind = which fifo is used

    awarestream(const char *, const char *, int _fifoind) : srcsel(4),
                                                            istr(nullptr),
                                                            ostr(nullptr),
                                                            ideletable(0),
                                                            odeletable(0),
                                                            fifoind(_fifoind) { NiceAssert( _fifoind >= 0 ); }

    // Copy constructor.  This will always throw as there is no defined concept of copy here

//    awarestream(const awarestream &src)
//    {
//        *this = src;
//        return;
//    }

    // Destructor

    ~awarestream()
    {
        if ( ideletable && ( srcsel == 1 ) ) { MEMDEL(istr); istr = nullptr; }
        if ( odeletable && ( srcsel == 1 ) ) { MEMDEL(ostr); ostr = nullptr; }
    }

    // Assignment operator - will always throw as assignment ill-defined

    awarestream &operator=(const awarestream &)
    {
        NiceThrow("Cannot duplicate awarestream");
        return *this;
    }

    // vogon pipes a string to the output stream. srcsel == 0 acts like /dev/null
    //
    // Success return: 0
    // Fail return: 1 for successful push onto killed fifo
    //             -1 for failed push onto fifo
    //             -2 for failed push onto killed fifo

    int vogon(const std::string &src)
    {
        int res = 0;

        if      ( ( srcsel == 1 ) && ( ostr != nullptr ) ) { *ostr << src; res = 0;           }
        else if (   srcsel == 4                          ) { res = strfifo.push(fifoind,src); }

        return res;
    }

    // Skim pipes a string off the input stream.
    //
    // Success return: 0
    // Fail return: 4 for attempted read from output only stream
    //              1 for successful pop from killed fifo
    //             -1 for failed pop from fifo
    //             -2 for failed pop from killed fifo

    int skim(std::string &dest)
    {
        int res = 0;

        if      ( srcsel == 0 ) { dest = "";     res = 4; }
        else if ( srcsel == 1 ) { *istr >> dest; res = 0; }
        else if ( srcsel == 4 )
        {
            int isdone = 0;

            while ( !isdone )
            {
                int fifosizeval = strfifo.size(fifoind);

                if ( fifosizeval > 0 )
                {
                    // There are strings in the fifo, so pop one of them.

                    res = strfifo.pop(fifoind,dest);

                    isdone = 1;
                }

                else if ( fifosizeval == 0 )
                {
                    // There are no strings in the fifo, but fifo live
                    // Unlock fifo, wait a random interval to give other
                    // threads a chance to push onto fifo or kill it, then
                    // retry.

                    //svm_usleep(FIFO_BASESLEEP+(svm_rand()%FIFO_ADDSLEEP_RAND));
                    svm_usleep(FIFO_BASESLEEP+(rand()%FIFO_ADDSLEEP_RAND));
                }

                else
                {
                    // fifo has kill-flag set, so treat it like a dead
                    // socket.

                    dest = "";
                    res = fifosizeval;

                    isdone = 1;
                }
            }
        }

        return res;
    }

    // Returns 1 if there (might be) more data in the buffer.  May not be
    // reliable for sockets

    int good(void)
    {
        int res = 0;

        if      ( srcsel == 0 ) { res = 0;                                      }
        else if ( srcsel == 1 ) { res = istr->good();                           }
        else if ( srcsel == 4 ) { res = ( strfifo.size(fifoind) >= 0 ) ? 1 : 0; }

        return res;
    }

    // Streaming stuff

    virtual std::streambuf::int_type underflow()
    {
        if ( gptr() == egptr() )
        {
            std::string tmpbuf;

            if ( skim(tmpbuf) ) { return std::streambuf::traits_type::eof(); }
            tmpbuf += "\n"; // Make sure the buffer gets flushed!

            std::streamsize size = tmpbuf.size();

            strncpy_safe_b(inputbuffer,tmpbuf.c_str(),tmpbuf.size());
            setg(inputbuffer,inputbuffer,inputbuffer+size);
        }

        return ( gptr() == egptr() ) ? std::streambuf::traits_type::eof() : std::streambuf::traits_type::to_int_type(*gptr());
    }

    virtual std::streambuf::int_type overflow(std::streambuf::int_type c)
    {
        size_t sizeis = pptr()-pbase();
        size_t i;

        for ( i = 0 ; i < sizeis ; ++i ) { outputbuffer += pbase()[i]; }

        // Newline or \0 flushes the buffer to the socket/stream

        if ( ( c == '\n' ) || ( c == '\0' ) )
        {
            if ( outputbuffer.length() )
            {
                outputbuffer += '\n';
                if ( vogon(outputbuffer) ) { return std::streambuf::traits_type::eof(); }
                outputbuffer = "";
            }
        }

        else { outputbuffer += std::streambuf::traits_type::to_char_type(c); }

        dummybuffer[0] = std::streambuf::traits_type::to_char_type(c);
        setp(dummybuffer,dummybuffer); // Keep this empty!

        return traits_type::not_eof(c);
    }

    // Kill one or all string fifos.  Will affect all threads using fifos.

    void killfifo(int num) { strfifo.setkillflag(num); }
    void killfifo(void)    { strfifo.setkillflag(); svm_usleep(5*(FIFO_BASESLEEP+FIFO_ADDSLEEP_RAND)); } // Sleep long enough to ensure that all threads waiting for data on fifo receive kill signal

private:

    int srcsel;    // 0 none, 1 stream, 2 UDP, 3 TCP, 4 shared fifo
    std::istream *istr;
    std::ostream *ostr;
    int ideletable;
    int odeletable;
    int fifoind;
    svmvolatile static fifolist strfifo;

    char inputbuffer[2048];
    char dummybuffer[2];
    std::string outputbuffer;
};


inline awarestream *&setident (awarestream *&a) { NiceThrow("Whatever"); return a; }
inline awarestream *&setzero  (awarestream *&a) { return a = nullptr; }
inline awarestream *&setposate(awarestream *&a) { return a; }
inline awarestream *&setnegate(awarestream *&a) { NiceThrow("I reject your reality and substitute my own"); return a; }
inline awarestream *&setconj  (awarestream *&a) { NiceThrow("Mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm"); return a; }
inline awarestream *&setrand  (awarestream *&a) { NiceThrow("Blippity Blappity Blue"); return a; }
inline awarestream *&postProInnerProd(awarestream *&a) { return a; }


inline void qswap(awarestream *&a, awarestream *&b);
inline void qswap(awarestream *&a, awarestream *&b)
{
    awarestream *x;

    x = a; a = b; b = x;
}


#endif



