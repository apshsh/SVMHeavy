//TO DO: now that pthread_t has been generalised to svm_pthread_t, do this:
//
// - add unique ID to svm_pthread_t
// - record these IDs to a global variable (linked list?)
// - have option in interactive mode to list active threads, pause, kill, edit variables etc
//
// see for example
//
// void pthread_cleanup_pop(int execute);
// void pthread_cleanup_push(void (*routine)(void*), void *arg); [Option End]

//
// Miscellaneous stuff
// (aka a bunch of random code and ugly hacks gathered together in one place)
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _basefn_h
#define _basefn_h


#define INSVMCONTEXT


// Assumed minimum c++ version: c++14

// =======================================================================
//
// Comment these out for different compilation systems.
//
// DJGPP_MATHS:    bessel functions available from DJGPP maths library (and
//                 some other stuff).  Used in numbase.cc
// SPECFN_ASUPP:   special functions in cmath from c++11 (eg gamma) are
//                 supported (see numbase.h).  HOWEVER: SUPPORT IS HOPELESSLY
//                 PATCHY SO USE ALTERNATIVES BY DEFAULT.
// ALLOW_SOCKETS:  sockets are used in awarestream for TCP and UDP streams.
// ENABLE_THREADS: threads are used for background-training and mutexes.
//                 (mutexes are still supported without this, but non-posix)
// CYGWIN_BUILD:   uncomment for cygwin.  Locates various libaries in
//                 different places.  Only required for threads, specifies
//                 location of un.h header file.
// CYGWIN10:       cygwin, in windows 10, has two oddities - no usleep
//                 function (so use nanosleep instead) and abs *is* defined
//                 for doubles (latter no longer relevant since moving
//                 everything to abs2).
// VISUAL_STU:     visual studio compile.  Disables various features taken
//                 from unistd.h, redefines certain things (eg inf number
//                 macros), allows for "interesting" variations in the maths
//                 library.
// VISUAL_STU_OLD: still more modifications for older versions of visual stu.
// VISUAL_STU_NOERF: because MS 2012 doesn't have erf... obviously :roll:
// HAVE_CONIO:     conio.h is available, so use this for some functions.
// HAVE_TERMIOS:   termios.h is available, so use this for some functions.
// DEBUG_MEM:      new / delete have extra debugging
// DISABLE_KB_BY_DEF: define to disable interactive keyboard by default
// USE_HOPDM:      Use system call to hopdm for linear optimisation rather than
//                 the default internal optimisation routine
// IS_CPP11:       enable c++11 features (true by default)
// IS_CPP14:       enable c++14 features (true by default)
// IS_CPP17:       enable c++17 features
// IS_CPP20:       enable c++20 features (automatically sets IS_CPP17)
// IS_CPP23:       enable c++23 features (automatically sets IS_CPP20, IS_CPP17)
//
// =======================================================================

// =======================================================================
//
// Multithreading note: there is nothing in the C++ standard about whether
// initialisation of static local variables is threadsafe.  The initialisation
// occurs the first time that a segment of code is reached, but if two threads
// hit the static variable at the same time then *in theory* both will call
// the initialisation function.
//
// In practice, gcc does put locks around static variable initialisation, so
// this won't be an issue in unix environments (though I can't speak for clang).
// However visual c++ does not put locks around the same code, so there could
// be an issue here.
//
// UPDATE: for c++11 static local variable initialisation is threadsafe.  If
// the variable is being initialised in one thread and a second thread reaches
// the point where it may initialise it then instead it will wait until the
// first thread has finished initialising and then use the variable so
// initialised.  I assume visual now follows this behaviour (gcc always has).
//
//
// ***************************
// MULTITHREADING AND GENTYPE:
// ***************************
//
// Multithreaded initialisation function: initgentype
//
// Initialises all derivatives in one block.  This is not required for single
// threaded operation, but for multithreaded use call this function first
// before starting any new threads.
//
// =======================================================================

#ifndef SPECIFYSYSVIAMAKE

// dos/djgpp (no longer supported)

// #define DJGPP_MATHS
// #define HAVE_CONIO

// cygwin/gcc

// #define ALLOW_SOCKETS
// #define ENABLE_THREADS
// #define CYGWIN_BUILD
// #define HAVE_TERMIOS

// cygwin/gcc modern

// #define ALLOW_SOCKETS
// #define ENABLE_THREADS
// #define CYGWIN_BUILD
// #define HAVE_TERMIOS

// linux/gcc

// #define ALLOW_SOCKETS
// #define ENABLE_THREADS
// #define HAVE_TERMIOS

// linux/gcc modern

// #define ALLOW_SOCKETS
// #define ENABLE_THREADS
// #define HAVE_TERMIOS

// Linux/gcc unthreaded unthreaded

// #define ALLOW_SOCKETS
// #define ENABLE_THREADS
// #define HAVE_TERMIOS

// Visual Studio

//#define VISUAL_STU
//#define VISUAL_STU_OLD
//#define VISUAL_STU_NOERF
//#define HAVE_CONIO
//#define IGNOREMEM
//#define ALLOW_SOCKETS
//#define _CRT_SECURE_NO_WARNINGS 1

// Visual Studio modern

//#define VISUAL_STU
//#define VISUAL_STUDIO_BESSEL
//#define VISUAL_STU_NOERF
//#define HAVE_CONIO
//#define IGNOREMEM
//#define ALLOW_SOCKETS
//#define ENABLE_THREADS
//#define _CRT_SECURE_NO_WARNINGS 1
//#pragma warning(disable:4996)
//#pragma warning(disable:4244)
//#pragma warning(disable:4756)

// Mex old

//#define USE_MEX
//#define VISUAL_STU
//#define VISUAL_STU_NOERF
//#define HAVE_CONIO
//#define IGNOREMEM
//#define ALLOW_SOCKETS

// Mex modern

///#define USE_MEX
//#define VISUAL_STU
//#define VISUAL_STUDIO_BESSEL
//#define VISUAL_STU_NOERF
//#define HAVE_CONIO
//#define NDEBUG
//// Uncomment to enable threads (currently not supported)
////#define ENABLE_THREADS
//// Comment to debug
////#define ALLOW_SOCKETS
//#define IGNOREMEM
//// Uncomment to debug memory
////#define DEBUG_MEM
//#ifdef _CRT_SECURE_NO_WARNINGS
//#undef _CRT_SECURE_NO_WARNINGS
//#endif
//#define _CRT_SECURE_NO_WARNINGS 1
////#pragma warning(disable:4996)
////#pragma warning(disable:4244)
//#pragma warning(disable:4005)
////#pragma warning(disable:4756)


// Mex modern on *nix

#define USE_MEX
#define HAVE_TERMIOS
#define HAVE_IOCTL
#define NOPURGE
//#define IS_CPP20
#define IGNOREMEM
#define NDEBUG


#endif


// Assume at least cpp14
#define IS_CPP11
#define IS_CPP14

#ifdef IS_CPP26
#define IS_CPP23
#endif

#ifdef IS_CPP23
#define IS_CPP20
#endif

#ifdef IS_CPP20
#define IS_CPP17
#endif

// #define USE_HOPDM






// =======================================================================
//
// Includes can depend on the OS, CPU version etc.
//
// =======================================================================


#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <string.h>
#include <string>
#include <time.h>
#include <math.h>
#include <cmath>
#include <limits.h>
#include <limits>
#include <random>
#include <chrono>
#include <vector>
#include <complex>

#ifdef VISUAL_STU
#include <windows.h>
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif
#endif

#ifndef VISUAL_STU
#include <unistd.h>
#include <sys/select.h>
#endif

#ifdef ALLOW_SOCKETS
#include <errno.h>
#ifdef CYGWIN_BUILD
#include <sys/un.h>
#endif
#ifndef VISUAL_STU
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#ifndef CYGWIN_BUILD
#include <linux/un.h>
#endif
#endif
#endif

#ifdef ENABLE_THREADS
#include <mutex>
#include <thread>
#include <atomic>
#endif

// spoilers...

inline int isMainThread(int val = 0);
inline int getThreadID(void);








#define COMMONOPDEF(_classname_) \
inline _classname_ &setident (_classname_ &a) { throw("something"); return a; } \
inline _classname_ &setposate(_classname_ &a) {                     return a; } \
inline _classname_ &setnegate(_classname_ &a) { throw("something"); return a; } \
inline _classname_ &setconj  (_classname_ &a) { throw("something"); return a; } \
inline _classname_ &setrand  (_classname_ &a) { throw("something"); return a; } \
inline _classname_ &postProInnerProd(_classname_ &a) { return a; }

#define COMMONOPDEFTEMP(_classname_) \
template <class T> inline _classname_ &setident (_classname_ &a) { throw("something"); return a; } \
template <class T> inline _classname_ &setposate(_classname_ &a) {                     return a; } \
template <class T> inline _classname_ &setnegate(_classname_ &a) { throw("something"); return a; } \
template <class T> inline _classname_ &setconj  (_classname_ &a) { throw("something"); return a; } \
template <class T> inline _classname_ &setrand  (_classname_ &a) { throw("something"); return a; } \
template <class T> inline _classname_ &postProInnerProd(_classname_ &a) { return a; }

#define COMMONOPDEFPT(_classname_) \
inline _classname_ *&setzero  (_classname_ *&a) { return a = nullptr; } \
inline _classname_ *&setident (_classname_ *&a) { throw("something"); return a; } \
inline _classname_ *&setposate(_classname_ *&a) { return a; } \
inline _classname_ *&setnegate(_classname_ *&a) { throw("something"); return a; } \
inline _classname_ *&setconj  (_classname_ *&a) { throw("something"); return a; } \
inline _classname_ *&setrand  (_classname_ *&a) { throw("something"); return a; } \
inline _classname_ *&postProInnerProd(_classname_ *&a) { return a; } \
inline void qswap(_classname_ *&a, _classname_ *&b) {  _classname_ *x = a; a = b; b = x; }


template <class T> inline std::vector<T> &setzero         (std::vector<T> &a) { throw("Operation not defined"); return a; }
template <class T> inline std::vector<T> &setzeropassive  (std::vector<T> &a) { throw("Operation not defined"); return a; }

template <class T> inline void qswap(std::vector<T> &a, std::vector<T> &b);
template <class T> inline void qswap(std::vector<T> &a, std::vector<T> &b)
{
    std::vector<T> x = a; a = b; b = x;
}

inline void qswap(std::complex<double> &a, std::complex<double> &b);
inline void qswap(std::complex<double> &a, std::complex<double> &b)
{
    std::complex<double> x = a; a = b; b = x;
}

COMMONOPDEFTEMP(std::vector<T>)
COMMONOPDEFPT(std::string)
COMMONOPDEFPT(std::complex<double>)
COMMONOPDEFPT(double)




















// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// Replacement for atexit
//
// ring = 0: run after ring 1
//        1: run after ring 2
//        2: run first

int svm_atexit(void (*func)(void), const char *desc, int ring);

// The above function uses atexit by default (with some modifications).  However
// this is not suitable in some environments (eg mex).  The following function
// can be used to replace atexit (the standard library function
// used to define functions that will be run on exit) with an alternative
// such as mexAtExit.  Returns the atexit function.  If xfn = nullptr then will
// keep current function.  Default is atexit.
//
// Note that this must be called prior to *ALL OTHER FUNCTIONS*.

typedef int (*atexitfn)(void (*)(void));

atexitfn svm_setatexitfn(atexitfn xfn = nullptr);

// Included here because reasons

int addremoveptr(void *addr, int newdel, int type, int size, const char *desc);

















// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// System command
//
// svm_execall: Call executable.  If runbg then attempt to leave it running in the background
// svm_pycall: Call python script.  If runbg then attempt to leave it running in the background
//
// Versions with result (res) string read this from pyres.txt, which must
// be created by the command.
//
// NB: these are disabled in HEADLESS mode

int svm_system (const char *command);
int svm_execall(const std::string &command, bool runbg);
int svm_pycall (const std::string &command, bool runbg);
int svm_execall(std::string &res, const std::string &command);
int svm_pycall (std::string &res, const std::string &command);

// The above function uses system by default.  However
// this is not suitable in some environments (eg mex).  The following function
// can be used to replace system (the standard library function) with an alternative.
// Returns the system function.  If xfn = nullptr then will
// keep current function.  Default is system.

typedef int (*systemfn)(const char *);

systemfn svm_setsystemfn(systemfn xfn = nullptr);



























// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// Sockets stuff

// --- If sockets available include relevant libraries ---

#ifdef ALLOW_SOCKETS

#ifdef VISUAL_STU
// windows doesn't define this, but inferring from final argument of recvfrom
#define socklen_t int
//class sockaddr_un;
//class sockaddr_un
//{
//    public:
//
//    int sun_family;
//    char *sun_path;
//};
#define UNIX_PATH_MAX 256
#define SHUT_RDWR     2
struct sockaddr_un
{
    int sun_family;
    char *sun_path;

    sockaddr_un()
    {
        sun_family = 0;
        sun_path = new char[UNIX_PATH_MAX+1];
        sun_path[0] = '\0';
    }

    ~sockaddr_un()
    {
        delete[] sun_path;
    }
};
//#define SHUT_RDWR     SD_BOTH
inline int close(int a);
inline int close(int a) { return closesocket(a); }
#pragma comment(lib, "Ws2_32.lib")
#endif

#define UDPBUFFERLEN 1024

// Alias everything

#define SVM_SOCK_STREAM   SOCK_STREAM
#define SVM_SOCK_DGRAM    SOCK_DGRAM
#define SVM_MAX_RETRIES   MAX_RETRIES
#define SVM_AF_INET       AF_INET
#define SVM_AF_UNIX       AF_UNIX
#define SVM_UNIX_PATH_MAX UNIX_PATH_MAX
#define SVM_INADDR_ANY    INADDR_ANY
#define SVM_SHUT_WR       SHUT_WR
#define SVM_SHUT_RDWR     SHUT_RDWR
#define SVM_UDPBUFFERLEN  UDPBUFFERLEN
#define svm_socklen_t     socklen_t
#define svm_sockaddr_in   sockaddr_in
#define svm_sockaddr_un   sockaddr_un
#define svm_sockaddr      sockaddr

inline int svm_send(int a, const char *b, size_t c, int d);
inline int svm_send(int a, const char *b, size_t c, int d) { return (int) send(a,b,c,d); }

inline int svm_recvfrom(int a, char *b, int c, int d, svm_sockaddr *e, svm_socklen_t *f);
inline int svm_recvfrom(int a, char *b, int c, int d, svm_sockaddr *e, svm_socklen_t *f) { return (int) recvfrom(a,b,c,d,e,f); }

inline int svm_htons(int a);
inline int svm_htons(int a) { return htons((uint16_t) a); }

inline int svm_htonl(int a);
inline int svm_htonl(int a) { return htonl(a); }

inline int svm_inet_addr(const char *a);
inline int svm_inet_addr(const char *a) { return inet_addr(a); }

inline int svm_shutdown(int a, int b);
inline int svm_shutdown(int a, int b) { return shutdown(a,b); }

inline int svm_close(int a);
inline int svm_close(int a) { return close(a); }

inline int svm_socket(int a, int b, int c);
inline int svm_socket(int a, int b, int c) { return socket(a,b,c); }

inline int svm_bind(int a, svm_sockaddr *b, int c);
inline int svm_bind(int a, svm_sockaddr *b, int c) { return bind(a,b,c); }

inline int svm_accept(int a, svm_sockaddr *b, svm_socklen_t *c);
inline int svm_accept(int a, svm_sockaddr *b, svm_socklen_t *c) { return accept(a,b,c); }

inline int svm_connect(int a, svm_sockaddr *b, int c);
inline int svm_connect(int a, svm_sockaddr *b, int c) { return connect(a,b,c); }

inline int svm_listen(int a, int b);
inline int svm_listen(int a, int b) { return listen(a,b); }

#endif

// --- If sockets not possible define stubs and fake classes to allow ---
// --- compilation and return error codes if sockets used.            ---

#ifndef ALLOW_SOCKETS

#define SVM_SOCK_STREAM   0
#define SVM_SOCK_DGRAM    0
#define SVM_MAX_RETRIES   5
#define SVM_AF_INET       0
#define SVM_AF_UNIX       0
#define SVM_UNIX_PATH_MAX 0
#define SVM_INADDR_ANY    0
#define SVM_SHUT_WR       0
#define SVM_SHUT_RDWR     0
#define SVM_UDPBUFFERLEN  1024
#define svm_socklen_t     int

struct svm_saddr;
struct svm_saddr
{
    public:

    int ws_addr;
    int wS_un; // something windows uses apparently
};

struct svm_sockaddr_in;
struct svm_sockaddr_in
{
    public:

    int sin_family;
    int sin_port;
    struct svm_saddr sin_addr;
};

struct svm_sockaddr_un;
struct svm_sockaddr_un
{
    public:

    int sun_family;
    char *sun_path;
};

struct svm_sockaddr;
struct svm_sockaddr
{
    public:

    int sin_family;
    int sin_port;
    struct svm_saddr sin_addr;
};

inline int svm_send(int a, const char *b, size_t c, int d);
inline int svm_send(int,   const char *,  size_t,   int) { return -1; }

inline int svm_recvfrom(int a, char *b, int c, int d, svm_sockaddr *e, svm_socklen_t *f);
inline int svm_recvfrom(int,   char *,  int,   int,   svm_sockaddr *,  svm_socklen_t *)  { return -1; }

inline int svm_htons(int a);
inline int svm_htons(int) { return -1; }

inline int svm_htonl(int a);
inline int svm_htonl(int) { return -1; }

inline int svm_inet_addr(const char *a);
inline int svm_inet_addr(const char *) { return -1; }

inline int svm_shutdown(int a, int b);
inline int svm_shutdown(int,   int) { return -1; }

inline int svm_close(int a);
inline int svm_close(int) { return -1; }

inline int svm_socket(int a, int b, int c);
inline int svm_socket(int,   int,   int) { return -1; }

inline int svm_bind(int a, svm_sockaddr *b, int c);
inline int svm_bind(int,   svm_sockaddr *,  int) { return -1; }

inline int svm_accept(int a, svm_sockaddr *b, svm_socklen_t *c);
inline int svm_accept(int,   svm_sockaddr *,  svm_socklen_t *) { return -1; }

inline int svm_connect(int a, svm_sockaddr *b, int c);
inline int svm_connect(int,   svm_sockaddr *,  int) { return -1; }

inline int svm_listen(int a, int b);
inline int svm_listen(int,   int) { return -1; }

#endif








































// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// mutex and thread stuff
//
// mutex:   very basic mutex stuff, typically maps to c++11 or pthreads.
// threads: pthreads if available, or similar on windows (if available).
//
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
//
// Threads:
//
// - implements basic posix functions only.
//
//
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
//
// isMainThread: Thread ID recorder
//
//     Call with val == +1 to set current thread as main
//     Call with val == 0  to test if current thread is main thread (nz is true)
//     Call with val == -1 to set no main thread (so isMainThread will always return true)
//
// getThreadID: generates, sets and returns a unique ID for this thread

//inline int isMainThread(int val = 0);
//inline int getThreadID(void);

// No point in using volatile if threads not available (we use them sparingly
// for thread-related tasks only)

#ifdef ENABLE_THREADS
#define svmvolatile volatile
#endif

#ifndef ENABLE_THREADS
#define svmvolatile
#endif

#ifdef ENABLE_THREADS

// ------------------------------------------------------------------------
// Threads
// ------------------------------------------------------------------------

class svm_pthread_t
{
    public:
    std::thread *a = nullptr;
};

inline void qswap(svm_pthread_t &a, svm_pthread_t &b);
inline void qswap(svm_pthread_t &a, svm_pthread_t &b)
{
    std::thread *x = a.a; a.a = b.a; b.a = x;
}

inline void qswap(svm_pthread_t *&a, svm_pthread_t *&b);
inline void qswap(svm_pthread_t *&a, svm_pthread_t *&b)
{
    svm_pthread_t *x = a; a = b; b = x;
}

COMMONOPDEFPT(svm_pthread_t);


//typedef std::thread svm_pthread_t;
typedef std::thread::id svm_pthread_id;
//#define svm_pthread_id std::thread::id

// local aliases

inline int svm_pthread_create(svm_pthread_t *a, void *(*c)(void *), void *d);
inline int svm_pthread_create(svm_pthread_t *a, void *(*c)(void *), void *d)
{
    a->a = new std::thread(c,d);

    return 0;
}

inline int svm_pthread_join(svm_pthread_t a, void **retval);
inline int svm_pthread_join(svm_pthread_t a, void **)
{
    (a.a)->join();

    return 0;
}

inline svm_pthread_id svm_pthread_self(void);
inline svm_pthread_id svm_pthread_self(void)
{
    return std::this_thread::get_id();
}

inline int svm_pthread_same_as_this(const svm_pthread_id &a);
inline int svm_pthread_same_as_this(const svm_pthread_id &a)
{
    return ( a == std::this_thread::get_id() );
}

inline int svm_pthread_same(const svm_pthread_id &a, const svm_pthread_id &b);
inline int svm_pthread_same(const svm_pthread_id &a, const svm_pthread_id &b)
{
    return ( a == b );
}

inline size_t numThreadsGuess(void);
inline size_t numThreadsGuess(void)
{
    return std::thread::hardware_concurrency();
}

//inline int isMainThread(int val)
inline int isMainThread(int val)
{
    static thread_local int mainThread = 0; // this will be set per thread, remember, and should
                                            // only get set/reset by the main thread

    if ( val == 1 )
    {
        mainThread = 1;
    }

    else if ( val == -1 )
    {
        mainThread = 0;
    }

    return mainThread;
/*
    int res = 1;

    static int mainThreadSet = 0;
    static svm_pthread_id currthread;

    if ( val == 1 )
    {
// Only the main thread will call this, so no need to lock

//        static std::mutex eyelock;
//        eyelock.lock();

        mainThreadSet = 1;
        currthread = svm_pthread_self();

//        eyelock.unlock();
    }

    else if ( val == -1 )
    {
        mainThreadSet = 0;
    }

    if ( mainThreadSet )
    {
        res = ( ( val == 2 ) || svm_pthread_same_as_this(currthread) );
    }

    return res;
*/
}

//inline int getThreadID(void);
inline int getThreadID(void)
{
    static thread_local int threadID = -1; // this holds our ID in thread_local storage. -1 indicates not yet set

    if ( threadID == -1 )
    {
        std::mutex cachelock;
        cachelock.lock();

        static std::atomic<int> loccnt(-1); // global and atomic, so safe to use

        threadID = ++loccnt; // increment atomic count and use this as threadID

        cachelock.unlock();
    }

    return threadID;
}

#endif



// --- If threads not present then define relevant stubs and functions ---
// --- to enable compilation and return relevant error codes etc.      ---

#ifndef ENABLE_THREADS
//inline int isMainThread(int val)
inline int isMainThread(int)
{
    return 1;
}

//inline int getThreadID(void);
inline int getThreadID(void)
{
    return 0;
}

typedef size_t svm_pthread_t;
typedef size_t svm_pthread_id;

/* should be elsewhere - see size_t versions in qswapbase
inline void qswap(svm_pthread_t &a, svm_pthread_t &b);
inline void qswap(svm_pthread_t &,  svm_pthread_t &)
{
    ;
}

inline void qswap(svm_pthread_t *&a, svm_pthread_t *&b);
inline void qswap(svm_pthread_t *&a, svm_pthread_t *&b)
{
    svm_pthread_t *c = a; a = b; b = c;
}

inline svm_pthread_t *&setident (svm_pthread_t *&a) { throw("something"); return a; }
inline svm_pthread_t *&setzero  (svm_pthread_t *&a) { return a = nullptr; }
inline svm_pthread_t *&setposate(svm_pthread_t *&a) { return a; }
inline svm_pthread_t *&setnegate(svm_pthread_t *&a) { throw("something"); return a; }
inline svm_pthread_t *&setconj  (svm_pthread_t *&a) { throw("something"); return a; }
inline svm_pthread_t *&setrand  (svm_pthread_t *&a) { throw("something"); return a; }
inline svm_pthread_t *&postProInnerProd(svm_pthread_t *&a) { return a; }
*/

inline int svm_pthread_create(svm_pthread_t *a, void *(*c)(void *), void *d);
inline int svm_pthread_create(svm_pthread_t *a, void *(*)(void *),  void *) { svm_pthread_t onlythreadid = 0; *a = onlythreadid; return -1; }

inline int svm_pthread_join(svm_pthread_t a, void **retval);
inline int svm_pthread_join(svm_pthread_t,   void **) { return -1; }

inline svm_pthread_id svm_pthread_self(void);
inline svm_pthread_id svm_pthread_self(void) { return 0; }

inline int svm_pthread_same_as_this(const svm_pthread_id &a);
inline int svm_pthread_same_as_this(const svm_pthread_id &) { return 1; }

inline int svm_pthread_same(const svm_pthread_id &a, const svm_pthread_id &b);
inline int svm_pthread_same(const svm_pthread_id &a, const svm_pthread_id &b) { return a == b; }

inline size_t numThreadsGuess(void);
inline size_t numThreadsGuess(void) { return 1; }

#endif










































// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// UUID generator (non-zero unique int)

inline int genUUID(void);

inline int genUUID(void)
{
    //NB zero result not allowed
#ifdef ENABLE_THREADS
    static std::atomic<int> nextUUID(1);
#endif
#ifndef ENABLE_THREADS
    static int nextUUID(1);
#endif

    //FIXME: extrememly unlikely bug may result when the UUID wraps back
    // through negatives and reaches zero.

    return (int) nextUUID++;
}



































// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// Binary number support (compile-time conversion)
//
// Converts 8 bit binary representation to int
//
// eg. binary(0110) = 6
//     binary(10000) = 16

#define BIGTYPE  size_t

template<BIGTYPE N>
class tobinary
{
public:

    enum
    {
        value = (N % 8) + (tobinary<N / 8>::value << 1)
    };
};

template<>
class tobinary<0>
{
public:

    enum
    {
        value = 0
    };
};

#define binnum(n) tobinary<0##n>::value
































// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// Stream clearing - keeps removing elements from the stream until a ':' is
// encountered, removes the ':' and return the stream
//
// usage example:
//
// wait_dummy blah;
// int readvar;
// stream >> blah; stream >> readvar;
//
// when given "this is a variable: 10" will ignore the preceeding string and
// set readvar = 10.

class wait_dummy
{
//    void *****ha_ha_ha;
};

std::istream &operator>>(std::istream &input, wait_dummy &come_on);

// stream to dev/nullptr

class NullStreamBuf : public std::streambuf
{
    char dummy[128];

protected:

    virtual int overflow(int c)
    {
        setp(dummy,dummy+sizeof(dummy));
        return ( c == traits_type::eof() ) ? '\0' : c;
    }
};

class NullOStream : private NullStreamBuf, public std::ostream
{
public:

    NullOStream() : std::ostream(this) {}
    NullStreamBuf* rdbuf() { return this; }
};






























// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// Error logging class and stream redirection
//
// errstream: get error stream
// seterrstream: sets alternative error stream to std::cerr
//
// outstream: get standard output stream
// setoutstream: sets alternative output stream to std::cout
//
// instream: get standard input stream
// setinstream: sets alternative input stream to std::cin
//
// promptstream: get input stream with extra optional argument (prompt) that goes to output stream
// setpromptstream: sets prompt stream
//
// NB: This is not necessarily thread-safe in two ways:
//
// - setting the error stream is *NOT* thread-safe.  Do it once at the start
//   of the code *before* you get into threaded code, not later.
// - whatever you put in here must be thread-safe if you want your code to
//   be thread-safe.
//
// LoggingOstreamErr: this class, if constructed with xcallback = nullptr, just
// sends whatever is streamed into it to std::cerr.  If xcallback is set
// to a function then anything streamed into this will then be sent to
// that function as a stream of characters.  For example, you could link
// the callback function to printf to make the output stream just go to
// std::cout by a rather convoluated path.  Or you could set the function
// to do nothing to create a stream that acts like /dev/null etc.
//
// LoggingOstreamOut: like LoggingOstream, but defaults to std::cout
//
// LoggingIstream: in this case callback is called when input is
// required by istream.  Usually just gets a char from std::cin, but
// can be used to get input from any source.
//
// Multiple streams: you can set multiple pipes in a stream that can 
// (in principle) go to different targets.  The index i tells you
// which one you're referring to.  For simplicity you can have a 
// maximum of 128 streams (0-127).


class LoggingOstreamErr : public std::ostream, public std::streambuf
{
public:
    LoggingOstreamErr(void (*xcallback)(char c)) : std::ostream(this),
                                                   callback(xcallback)
    {
        //setbuf(0,0);
    }

    int overflow(int c)
    {
        // This is a virtual function that will be called from ostream

        justprintit((char) c);
        return 0;
    }

    void justprintit(char c)
    {
        // This splits to either standard printing to cerr or
        // calling the callback function

        if ( callback == nullptr )
        {
;
#ifndef HEADLESS
            std::cerr.put(c);
#endif
        }

        else
        {
            (*callback)(c);
        }
    }

    void setcallback(void (*xcallback)(char c) = nullptr)
    {
        // This allows you to set callback

        callback = xcallback;
    }

    static bool suppressStreamCout; // if true then prints to this stream don't appear on the screen
    static bool suppressStreamFile; // if true then prints to this stream don't get logged to a file

private:

    // Function callback to print to alterative destination.

    void (*callback)(char c);
};

class LoggingOstreamOut : public std::ostream, public std::streambuf
{
public:
    LoggingOstreamOut(void (*xcallback)(char c)) : std::ostream(this),
                                                   callback(xcallback)
    {
        setbuf(0,0);
    }

    int overflow(int c)
    {
        // This is a virtual function that will be called from ostream

        justprintit((char) c);
        return 0;
    }

    void justprintit(char c)
    {
        // This splits to either standard printing to cout (not cerr for this one) or
        // calling the callback function

        if ( callback == nullptr )
        {
;
#ifndef HEADLESS
            std::cout.put(c);
#endif
        }

        else
        {
            (*callback)(c);
        }
    }

    void setcallback(void (*xcallback)(char c) = nullptr)
    {
        // This allows you to set callback

        callback = xcallback;
    }

    static bool suppressStreamCout; // if true then prints to this stream don't appear on the screen
    static bool suppressStreamFile; // if true then prints to this stream don't get logged to a file

private:

    // Function callback to print to alterative destination.

    void (*callback)(char c);
};

class LoggingIstream : public std::istream, public std::streambuf
{
public:
    LoggingIstream(char (*xcallback)(void)) : std::istream(this),
                                              callback(xcallback)
    {
        setbuf(0,0);
    }

    int underflow(void)
    {
        // This is a virtual function that will be called from istream

        setg(buffer,buffer,buffer+1);
        *gptr() = justscanit();
        return *gptr();
    }

    char justscanit()
    {
        // This splits to either standard input from cin or
        // calling the callback function

        char c;

        if ( callback == nullptr )
        {
;
#ifndef HEADLESS
            c = (char) std::cin.get();
#endif
        }

        else
        {
            c = (*callback)();
        }

        return c;
    }

    void setcallback(char (*xcallback)(void) = nullptr)
    {
        // This allows you to set callback

        callback = xcallback;
    }

private:

    // Function callback to print to scan alternative input

    char (*callback)(void);
    char buffer[1];
};

inline std::ostream &errstreamunlogged(void);
#ifndef HEADLESS
inline std::ostream &errstreamunlogged(void) { return std::cerr; }
#endif
#ifdef HEADLESS
inline std::ostream &errstreamunlogged(void) { static thread_local std::ostream cnull(0); return cnull; }
#endif

inline std::ostream &outstreamunlogged(void);
#ifndef HEADLESS
inline std::ostream &outstreamunlogged(void) { return std::cout; }
#endif
#ifdef HEADLESS
inline std::ostream &outstreamunlogged(void) { static thread_local std::ostream cnull(0); return cnull; }
#endif

std::ostream &errstream(int i = 0);
inline void errstream(const char *src);
inline void errstream(const char *src) { if ( isMainThread() ) { errstream() << src; } }
void seterrstream(LoggingOstreamErr *altdest, int i = 0);

void   suppresserrstreamcout(void);
void   suppresserrstreamfile(void);
void unsuppresserrstreamcout(void);
void unsuppresserrstreamfile(void);

inline void errstreamunlogged(const char *src);
#ifndef HEADLESS
inline void errstreamunlogged(const char *src) { std::cerr << src; }
#endif
#ifdef HEADLESS
inline void errstreamunlogged(const char *) { ; }
#endif

std::ostream &outstream(int i = 0);
inline void outstream(const char *src);
inline void outstream(const char *src) { if ( isMainThread() ) { outstream() << src; } }
void setoutstream(LoggingOstreamOut *altdest, int i = 0);

void   suppressoutstreamcout(void);
void   suppressoutstreamfile(void);
void unsuppressoutstreamcout(void);
void unsuppressoutstreamfile(void);

void   suppressallstreamcout(void);
void   suppressallstreamfile(void);
void unsuppressallstreamcout(void);
void unsuppressallstreamfile(void);

std::istream &instream(int i = 0);
void setinstream(LoggingIstream *altsrc, int i = 0);

std::istream &promptstream(const std::string &prompt, int i = 0);
std::ostream &promptoutstream(int i = 0);
void setpromptstream(LoggingIstream *altsrc, LoggingOstreamOut *altdest, int i = 0);


// streamItIn: equivalent to input >> dest.  processxyzvw used elsewhere 
// streamItOut: equivalent to output << src.  if retainTypeMarker set then 
//              src will retain its essential "typeness" when printed (eg
//              double will always contain . or e).

#define STREAMINDUMMY(X) \
inline std::istream &streamItIn(std::istream &input, X& dest, int processxyzvw = 1); \
inline std::istream &streamItIn(std::istream &input, X& dest, int processxyzvw) \
{ \
    (void) dest; \
    (void) processxyzvw; \
    throw("Just no"); \
    return input; \
} \
inline std::ostream &operator<<(std::ostream &output, X& src); \
inline std::ostream &operator<<(std::ostream &output, X& src) \
{ \
    (void) src; \
    throw("Just no"); \
    return output; \
} \
inline std::istream &operator>>(std::istream &input, X& dest); \
inline std::istream &operator>>(std::istream &input, X& dest) \
{ \
    (void) dest; \
    throw("Just no"); \
    return input; \
}

//STREAMINDUMMY(const double *);
//STREAMINDUMMY(const int *);
//STREAMINDUMMY(const std::string *);

inline std::istream &streamItIn(std::istream &input, int&         dest, int processxyzvw = 1);
inline std::istream &streamItIn(std::istream &input, double&      dest, int processxyzvw = 1);
inline std::istream &streamItIn(std::istream &input, char&        dest, int processxyzvw = 1);
inline std::istream &streamItIn(std::istream &input, char*        dest, int processxyzvw = 1);
inline std::istream &streamItIn(std::istream &input, std::string& dest, int processxyzvw = 1);

inline std::ostream &streamItOut(std::ostream &output, const int&         src, int retainTypeMarker = 0);
inline std::ostream &streamItOut(std::ostream &output, const double&      src, int retainTypeMarker = 0);
inline std::ostream &streamItOut(std::ostream &output, const char&        src, int retainTypeMarker = 0);
inline std::ostream &streamItOut(std::ostream &output, char* const&       src, int retainTypeMarker = 0);
inline std::ostream &streamItOut(std::ostream &output, const std::string& src, int retainTypeMarker = 0);

inline std::istream &streamItIn(std::istream &input, double &dest, int)
{
    input >> dest;

    return input;
}

inline std::istream &streamItIn(std::istream &input, int &dest, int)
{
    input >> dest;

    return input;
}

inline std::istream &streamItIn(std::istream &input, char &dest, int)
{
    input >> dest;

    return input;
}

inline std::istream &streamItIn(std::istream &input, char *, int)
{
    //input >> dest;

    return input;
}

inline std::istream &streamItIn(std::istream &input, std::string &dest, int)
{
    input >> dest;

    return input;
}

inline std::ostream &streamItOut(std::ostream &output, const double &src, int retainTypeMarker)
{
    char tempres[100];
    sprintf(tempres,"%.17g",src);
    std::string tempresb(tempres);
    output << tempresb;

    if ( retainTypeMarker && !tempresb.find(".") && !tempresb.find("e") && !tempresb.find("E") )
    {
        output << ".0";
    }

    return output;
}

inline std::ostream &streamItOut(std::ostream &output, const int &src, int)
{
    output << src;

    return output;
}

inline std::ostream &streamItOut(std::ostream &output, const char &src, int)
{
    output << src;

    return output;
}

inline std::ostream &streamItOut(std::ostream &output, char* const& src, int)
{
    output << src;

    return output;
}

inline std::ostream &streamItOut(std::ostream &output, const std::string& src, int)
{
    output << src;

    return output;
}




































// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
//
// nullPrint: Print string to stream, then backspace back to start of string (or a few chars after if offset set positive).  Returns number of characters printed (so you can blackPrint to cover over)
// blankPrint: Print n spaces, then backspace back over them
// repPrint: print character n times
// wideprint: Print string and pad with spaces to get to desired width, leaving cursor on right.  Return number of unprinted characters

size_t nullPrint(std::ostream &dest, const std::string &src, size_t offset = 0);
size_t nullPrint(std::ostream &dest, const char        *src, size_t offset = 0);
size_t nullPrint(std::ostream &dest, const int          src, size_t offset = 0);
size_t widePrint(std::ostream &dest, const std::string &src, size_t width);

std::ostream &blankPrint(std::ostream &dest, size_t n);
std::ostream &repPrint(std::ostream &dest, const char c, size_t n);


// prompttoi: prompted user input to integer.  Result must be a number (not NaN), in range
//            min <= dest <= max (unless min > max).  Will give up after maxtries attempts.
//            Uses stream i
// prompttoi: prompted user input to double.

#define MAXTRIES 10

int prompttoi(int &dest,    int    min, int    max, const std::string &prompt, int maxtries = MAXTRIES, int i = 0);
int prompttod(double &dest, double min, double max, const std::string &prompt, int maxtries = MAXTRIES, int i = 0);


















































// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// Screen control, user input, CLI stuff
//
// enternonblockmode:  put keyboard in non-blocking mode
// exitnonblockmode:   take keyboard out of non-blocking mode.
// testinnonblockmode: true if in non-blocking mode, false otherwise.
// svm_getch_nonblock: check for keypress (if any) in non-blocking mode.
//                     This function leaves blocking mode the same as when entered.
//
// svmclrscr(fastver):      clear screen (skipped if fastver set) and put cursor at location (0,0)
// svmcurs(linenum,colnum): put cursor at linenum,colnum, which are indexed from 0
// getscrrowcol(rows,cols): find screen size and store it in rows, cols
//
// Note on svm_getch_nonblock and special characters:
//
// On windows arrow keys and such are reported as multiple keystrokes:
//
// 0xe0 0x4b = LEFT_ARROW
// 0xe0 0x4d = RIGHT_ARROW
// 0xe0 0x48 = UP_ARROW
// 0xe0 0x50 = DOWN_ARROW
//
// On *nix they get reported as multiple keystrokes:
//
// \033[D - arrow left
// \033[C - arrow right
// \033[A - arrow up
// \033[B - arrow down

void enternonblockmode (void);
void exitnonblockmode  (void);
int testinnonblockmode (void);
char svm_getch_nonblock(void);

void svmclrscr(int fastver);
void svmcurs(int linenum, int colnum);
void getscrrowcol(int &rows, int &cols);













































































// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// Asynchronous keyboard quit detection
//
// kbquitdet:
//
// If no keys hit will do nothing and return 0
// If key hit will print prompt and wait for user
// - if user input is quit then returns 1
// - if user input is anything else then returns 0
//
// uservars: either nullptr, or nullptr-terminated list of adjustable variables
// varnames: names of above
// vardescr: descriptions of above
// goanyhow: if set then assume key pressed
//
// disablekbquitdet: disable function
// enablekbquitdet:  enable function
// triggerkbquitdet: set kbquitdet to act as if key has been pressed on next call
//
// haskeybeenpressed: functionally equivalent to kbhit in conio.h but for
//                    a wider variety of systems (but not all)
//
// interactmenu: what gets called if key is pressed.  Dummy may be modified but
//               just ignore this.
//
// setintercalc: Function to set calculator function callback
// setkbcallback: set callback function to test if interupt key (eg ctrl-c in matlab) has been pressed.
//
// gintercalc: call(back) internal calculator function
// ginterobs: call(back) internal ML examiner function
// gkbcallback: call(back) interupt key test function

inline bool kbquitdet(const char *stateDescr, double **uservars = nullptr, const char **varnames = nullptr, const char **vardescr = nullptr, int goanyhow = 0);
inline void disablekbquitdet(void);
inline void enablekbquitdet(void);
inline void triggerkbquitdet(void);
inline void clearkbquitdet(void);
       void setintercalc(void (*intercalccall)(std::ostream &, std::istream &));
inline void setkbcallback(int (*kbcallback)(void) = nullptr);
inline int gkbcallback(int (*kbcallback)(void) = nullptr);
inline void gintercalc(std::ostream &output, std::istream &input, void (*intercalccall)(std::ostream &, std::istream &) = nullptr);
inline void ginterobs(std::ostream &output, std::istream &input, void (*interobscall)(std::ostream &, std::istream &) = nullptr);

int interactmenu(int &dummy, int &dostep, const char *stateDescr, double **uservars = nullptr, const char **varnames = nullptr, const char **vardescr = nullptr);
inline int haskeybeenpressed(void);
const char *randomquote(void);

void snakes(int gamewidth, int gameheight, int numrabbits, int startsnakelen, int addrate, int usleeptime);

// Stuff in linux but not windows

#ifndef STDIN_FILENO
#define STDIN_FILENO 1
#endif

#ifndef STDOUT_FILENO
#define STDOUT_FILENO 0
#endif

#ifdef HAVE_CONIO
#include <conio.h>
#endif

#ifdef HAVE_TERMIOS
#include <termios.h>
#include <fcntl.h>
#endif

#ifdef HAVE_IOCTL
#include <sys/ioctl.h>
#endif

inline int haskeybeenpressed(void)
{
    int res = 0;

    // If ncurses present use that

    #ifdef HAVE_NCURSES
FIXME - has a key been pressed
    #endif

    // If conio present use kbhit variant

    #ifndef HAVE_NCURSES
    #ifdef HAVE_CONIO
    #ifdef VISUAL_STU
    #ifdef VISUAL_STU_OLD
    res = kbhit();
    #endif
    #ifndef VISUAL_STU_OLD
    res = _kbhit();
    #endif
    #endif
    #ifndef VISUAL_STU
    res = kbhit();
    #endif
    #endif
    #endif

    // Fallback method (only detects enter and maybe whitespace, nothing else, unless in non-blocking mode and IOCTL available)

    #ifndef HAVE_NCURSES
    #ifndef HAVE_CONIO
//    #ifdef HAVE_IOCTL
//    if ( innonblockmode() )
//    {
//        ioctl(STDIN_FILENO,FIONREAD,&res);  // 0 is STDIN
//    }
//    else
//    #endif
    {
        struct timeval tv;
        fd_set fds;
        tv.tv_sec = 0;
        tv.tv_usec = 0;
        FD_ZERO(&fds);
        FD_SET(STDIN_FILENO,&fds); //STDIN_FILENO is 0
        select(STDIN_FILENO+1,&fds,nullptr,nullptr,&tv);
        res = FD_ISSET(STDIN_FILENO,&fds);
    }
    #endif
    #endif

    return res;
}



// Don't use the function retkeygriggerandclear
inline int retkeytriggerandclear(int val = 0);
inline int retkeytriggerandclear(int val)
{
    static thread_local int trval = 0; // trigger not set by default
    int retval = trval;   // return is current trigger state

    trval = val; // Set new value

    return retval; // return old value
}

// Don't use setgetkbstate
inline int setgetkbstate(int x = 2);
inline int setgetkbstate(int x)
{
/*
    #ifdef DISABLE_KB_BY_DEF
    svmvolatile static int status = 0; // 1 enabled, 0 disabled
    #endif

    #ifndef DISABLE_KB_BY_DEF
    svmvolatile static int status = 1; // 1 enabled, 0 disabled
    #endif
*/

    #ifdef DISABLE_KB_BY_DEF
    static thread_local int status = 0; // 1 enabled, 0 disabled
    #endif

    #ifndef DISABLE_KB_BY_DEF
    static thread_local int status = 1; // 1 enabled, 0 disabled
    #endif

    if ( x == 0 ) { status = 0; }
    if ( x == 1 ) { status = 1; }

    return status;
}


inline int gkbcallback(int (*kbcallback)(void))
{
    static thread_local int (*lockbcallback)(void) = nullptr;
    int res = 0;

    if ( kbcallback )
    {
        lockbcallback = kbcallback;
    }

    else if ( lockbcallback )
    {
        res = lockbcallback();
    }

    return res;
}

inline void gintercalc(std::ostream &output, std::istream &input, void (*intercalccall)(std::ostream &, std::istream &))
{
    static void (*loccalc)(std::ostream &, std::istream &) = nullptr;

    if ( intercalccall )
    {
        loccalc = intercalccall;
    }

    else if ( loccalc )
    {
        loccalc(output,input);
    }

    else
    {
        output << "Calculator not fitted.\n";
    }
}

inline void ginterobs(std::ostream &output, std::istream &input, void (*interobscall)(std::ostream &, std::istream &))
{
    static void (*locobs)(std::ostream &, std::istream &) = nullptr;

    if ( interobscall )
    {
        locobs = interobscall;
    }

    else if ( locobs )
    {
        locobs(output,input);
    }

    else
    {
        output << "ML examiner not fitted.\n";
    }
}

inline void setinterobs(void (*interobscall)(std::ostream &, std::istream &))
{
    ginterobs(outstream(),instream(),interobscall);
}

inline void setkbcallback(int (*kbcallback)(void))
{
    gkbcallback(kbcallback);
}

inline void disablekbquitdet(void)
{
   setgetkbstate(0);
}

inline void enablekbquitdet(void)
{
   setgetkbstate(1);
}

inline void triggerkbquitdet(void)
{
    retkeytriggerandclear(1); // set trigger
}

inline void clearkbquitdet(void)
{
    retkeytriggerandclear(0); // clear trigger
}

inline bool kbquitdet(const char *stateDescr, double **uservars, const char **varnames, const char **vardescr, int goanyhow)
{
    (void) stateDescr;
    (void) uservars;
    (void) varnames;
    (void) vardescr;
    (void) goanyhow;

#ifndef USERLESS
#ifndef HEADLESS
    static thread_local int goupone = 0;
    static thread_local int dostep = 0;
    static thread_local int reallymainthread = isMainThread(); // only need to call once this way
#endif
#endif
    bool res = false;

#ifndef USERLESS
#ifndef HEADLESS
    // short-circuit logic, ordered by ease of evaluation
    if ( reallymainthread && setgetkbstate() && ( dostep || goupone || goanyhow || retkeytriggerandclear() || gkbcallback() || haskeybeenpressed() ) ) // && isMainThread() )
    {
        dostep = 0;

        if ( goupone )
        {
            --goupone;
        }

        if ( goupone )
        {
            res = true;
        }

        else
        {
            res = interactmenu(goupone,dostep,stateDescr,uservars,varnames,vardescr);
        }
    }
#endif
#endif

    return res;
}




























































// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
//
// Memory count function
//
// Note that memcount is neither (a) accurate or (b) thread-safe.

inline size_t memcount(size_t incsize = 0, int direction = -1);
inline size_t memcount(size_t incsize,     int direction)
{
#ifdef ENABLE_THREADS
    static std::atomic<size_t> memused(0);
#endif
#ifndef ENABLE_THREADS
    static size_t memused(0);
#endif

    if ( direction > 0 )
    {
        memused += incsize;
    }

    else if ( direction < 0 )
    {
        if ( incsize < memused )
        {
            memused -= incsize;
        }

        else
        {
            memused -= memused;
        }
    }

    else
    {
        memused = incsize;
    }

    return memused;
}





















// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
//
// "Unsafe" function isolation

//inline char *strncpy_safe(char *dest, const char *src, size_t len);
//inline char *strncpy_safe(char *dest, const char *src, size_t len)
//{
//    strncpy(dest,src,len);
//    dest[len] = '\0';
//    return dest;
//}

















// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// File functions: because earlier versions of c++ don't support "does
// this file exist" functionality, we have this
//
// getUniqueFile: construct a filename string pre+UUID+post where the file
// does not exist.  UUID is in hex, because hex is cool.

int fileExists(const std::string &fname);
const std::string &getUniqueFile(std::string &res, const std::string &pre, const std::string &post);

































//------------------------------------------------------------------------
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//************************************************************************
//------------------------------------------------------------------------
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//************************************************************************
//------------------------------------------------------------------------
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//************************************************************************
//------------------------------------------------------------------------
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//************************************************************************
//------------------------------------------------------------------------
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//******************************************************************

#endif

