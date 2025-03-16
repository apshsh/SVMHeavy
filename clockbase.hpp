
//
// Simple time functions
//
// Version: split off basefn
// Date: 11/09/2024
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _clockbase_h
#define _clockbase_h

#include <time.h>
#include <chrono>

// Time measurement:
//
// The aim is to measure elapsed time in seconds, with roughly ms precision, 
// with result in a double giving seconds elapsed.  Originally I used:
//
// typedef clock_t     time_used;
// typedef long double timediffunits;
//
// #define TIMEDIFFSEC(a,b) (((double) (a-b))/CLOCKS_PER_SEC)
// #define TIMECALL clock();
//
// (with apologies for the bad function naming).  But this fails for long
// itervales as clock_t is typically signed 32 bit and overflows.
//
// Instead we keep both clock ticks and timestamp.  If timestamp difference
// is less than CROSSOVER_TIME_SEC seconds (600 seconds == 10 minutes) then
// clock tick difference is used, which is fine.  Otherwise the more
// coarse-grained but non-overflowing timestamp difference is used.
//
//
// TIMEDIFFSEC: time difference between a and b in seconds
// TIMEABSSEC:  time difference between a and some arbitrary "zero" time
//              (the time this is first called)
// gettimeuse:  grabs timestamp of type time_used
// svm_usleep:  sleep for (possibly rounded off) stm microseconds.
// svm_msleep:  sleep for (possibly rounded off) stm milliseconds.
// svm_sleep:   sleep for (possibly rounded off) stm seconds.
// ZEROTIMEDIFF: set time difference to zero

#define CROSSOVER_TIME_SEC 600
#define TIMECALL gettimeuse()

typedef double timediffunits;

#define ZEROTIMEDIFF(x) x = 0

class time_used
{
    public:

    time_used(void) { }
    time_used(const time_used &src) : alttime(src.alttime) { }

    time_used &operator=(const time_used &src)
    {
        alttime = src.alttime;

        return *this;
    }

    std::chrono::steady_clock::time_point alttime;
};

inline double TIMEDIFFSEC(const time_used &a, const time_used &b);
inline double TIMEABSSEC(const time_used &b);
inline time_used gettimeuse(void);

void svm_usleep(int stm);
void svm_msleep(int stm);
void svm_sleep (int stm);

int svm_clock_year (void); // year local time
int svm_clock_month(void); // month local time
int svm_clock_day  (void); // day local time
int svm_clock_hour (void); // hour local time
int svm_clock_min  (void); // minutes local time
int svm_clock_sec  (void); // seconds local time


inline double TIMEDIFFSEC(const time_used &a, const time_used &b)
{
    return ((double) std::chrono::duration_cast<std::chrono::milliseconds>(a.alttime - b.alttime).count())/1000.0;
}

inline double TIMEABSSEC(const time_used &b)
{
    static time_used a = b; // this will only initialise once, so the value
                            // will be whatever b is on the first call to
                            // this function.

    return TIMEDIFFSEC(b,a);
}

inline time_used gettimeuse(void)
{
    time_used res;

    res.alttime = std::chrono::steady_clock::now();

    return res;
}


#endif

