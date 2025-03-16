
//
// Quickswap functions for base types
//
// Version: split off basefn
// Date: 11/09/2024
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//


#include "niceassert.hpp"

#ifndef _qswapbase_h
#define _qswapbase_h


inline void qswap(         bool    &a,          bool    &b);
inline void qswap(         char    &a,          char    &b);
inline void qswap(         int     &a,          int     &b);
inline void qswap(unsigned int     &a, unsigned int     &b);
inline void qswap(         size_t  &a,          size_t  &b);
inline void qswap(         size_t *&a,         size_t *&b);
inline void qswap(volatile int     &a, volatile int     &b);
inline void qswap(         double  &a,          double  &b);
inline void qswap(std::string      &a, std::string      &b);
inline void qswap(const    double *&a, const    double *&b);
inline void qswap(const    int    *&a, const    int    *&b);
inline void qswap(         int    *&a,          int    *&b);
inline void qswap(const    char   *&a, const    char   *&b);
inline void qswap(std::istream    *&a, std::istream    *&b);

inline void qswap(         bool    &a,          bool    &b) {          bool    c(a); a = b; b = c; }
inline void qswap(         char    &a,          char    &b) {          char    c(a); a = b; b = c; }
inline void qswap(         int     &a,          int     &b) {          int     c(a); a = b; b = c; }
inline void qswap(unsigned int     &a, unsigned int     &b) { unsigned int     c(a); a = b; b = c; }
inline void qswap(         size_t  &a,          size_t  &b) {          size_t  c(a); a = b; b = c; }
inline void qswap(         size_t *&a,          size_t *&b) {          size_t *c(a); a = b; b = c; }
inline void qswap(volatile int     &a, volatile int     &b) { volatile int     c(a); a = b; b = c; }
inline void qswap(         double  &a,          double  &b) {          double  c(a); a = b; b = c; }
inline void qswap(std::string      &a, std::string      &b) { std::string      c(a); a = b; b = c; }
inline void qswap(const    double *&a, const    double *&b) { const    double *c(a); a = b; b = c; }
inline void qswap(const    int    *&a, const    int    *&b) { const    int    *c(a); a = b; b = c; }
inline void qswap(         int    *&a,          int    *&b) {          int    *c(a); a = b; b = c; }
inline void qswap(const    char   *&a, const    char   *&b) { const    char   *c(a); a = b; b = c; }
inline void qswap(std::istream    *&a, std::istream    *&b) { std::istream    *c(a); a = b; b = c; }

inline double &setident (double &a);
inline double &setzero  (double &a);
inline double &setposate(double &a);
inline double &setnegate(double &a);
inline double &setconj  (double &a);
inline double &setrand  (double &a); // uniform 0 to 1
inline double &postProInnerProd(double &a);

inline int &setident (int &a);
inline int &setzero  (int &a);
inline int &setposate(int &a);
inline int &setnegate(int &a);
inline int &setconj  (int &a);
inline int &setrand  (int &a); // random -1 or 1
inline int &postProInnerProd(int &a);

inline unsigned int &setident (unsigned int &a);
inline unsigned int &setzero  (unsigned int &a);
inline unsigned int &setposate(unsigned int &a);
inline unsigned int &setnegate(unsigned int &a);
inline unsigned int &setconj  (unsigned int &a);
inline unsigned int &setrand  (unsigned int &a); // random -1 or 1
inline unsigned int &postProInnerProd(unsigned int &a);

inline size_t &setident (size_t &a);
inline size_t &setzero  (size_t &a);
inline size_t &setposate(size_t &a);
inline size_t &setnegate(size_t &a);
inline size_t &setconj  (size_t &a);
inline size_t &setrand  (size_t &a); // random -1 or 1
inline size_t &postProInnerProd(size_t &a);

inline char &setident (char &a);
inline char &setzero  (char &a);
inline char &setposate(char &a);
inline char &setnegate(char &a);
inline char &setconj  (char &a);
inline char &setrand  (char &a); // random -1 or 1
inline char &postProInnerProd(char &a);

inline std::string &setident (std::string &a); // throw
inline std::string &setzero  (std::string &a); // empty string
inline std::string &setposate(std::string &a);
inline std::string &setnegate(std::string &a); // throw
inline std::string &setconj  (std::string &a); // throw
inline std::string &setrand  (std::string &a); // throw
inline std::string &postProInnerProd(std::string &a);

inline const char *&setident (const char *&a);
inline const char *&setzero  (const char *&a);
inline const char *&setposate(const char *&a);
inline const char *&setnegate(const char *&a);
inline const char *&setconj  (const char *&a);
inline const char *&setrand  (const char *&a);

inline int *&setident (int *&a);
inline int *&setzero  (int *&a);
inline int *&setposate(int *&a);
inline int *&setnegate(int *&a);
inline int *&setconj  (int *&a);
inline int *&setrand  (int *&a);

inline size_t *&setident (size_t *&a);
inline size_t *&setzero  (size_t *&a);
inline size_t *&setposate(size_t *&a);
inline size_t *&setnegate(size_t *&a);
inline size_t *&setconj  (size_t *&a);
inline size_t *&setrand  (size_t *&a);

inline const char *&setident (const char *&a) { a = nullptr;                      return a; }
inline const char *&setzero  (const char *&a) { a = nullptr;                      return a; }
inline const char *&setposate(const char *&a) {                                   return a; }
inline const char *&setnegate(const char *&a) { NiceThrow("don't do that");       return a; }
inline const char *&setconj  (const char *&a) {                                   return a; }
inline const char *&setrand  (const char *&a) { NiceThrow("that's a silly idea"); return a; }

inline int *&setident (int *&a) { a = nullptr;                      return a; }
inline int *&setzero  (int *&a) { a = nullptr;                      return a; }
inline int *&setposate(int *&a) {                                   return a; }
inline int *&setnegate(int *&a) { NiceThrow("don't do that");       return a; }
inline int *&setconj  (int *&a) {                                   return a; }
inline int *&setrand  (int *&a) { NiceThrow("that's a silly idea"); return a; }

inline size_t *&setident (size_t *&a) { a = nullptr;                      return a; }
inline size_t *&setzero  (size_t *&a) { a = nullptr;                      return a; }
inline size_t *&setposate(size_t *&a) {                                   return a; }
inline size_t *&setnegate(size_t *&a) { NiceThrow("don't do that");       return a; }
inline size_t *&setconj  (size_t *&a) {                                   return a; }
inline size_t *&setrand  (size_t *&a) { NiceThrow("that's a silly idea"); return a; }

inline double *&setident (double *&a) { NiceThrow("mummble"); return a; }
inline double *&setzero  (double *&a) { a = nullptr;          return a; }
inline double *&setposate(double *&a) {                       return a; }
inline double *&setnegate(double *&a) { NiceThrow("mummble"); return a; }
inline double *&setconj  (double *&a) { NiceThrow("mummble"); return a; }
inline double *&setrand  (double *&a) { NiceThrow("mummble"); return a; }
inline double *&postProInnerProd(double *&a) { return a; }

inline const double *&setident (const double *&a) { NiceThrow("mummble"); return a; }
inline const double *&setzero  (const double *&a) { a = nullptr;          return a; }
inline const double *&setposate(const double *&a) {                       return a; }
inline const double *&setnegate(const double *&a) { NiceThrow("mummble"); return a; }
inline const double *&setconj  (const double *&a) { NiceThrow("mummble"); return a; }
inline const double *&setrand  (const double *&a) { NiceThrow("mummble"); return a; }
inline const double *&postProInnerProd(const double *&a) { return a; }

inline double &setident (double &a           ) { return ( a = 1 ); }
inline double &setzero  (double &a           ) { return ( a = 0 ); }
inline double &setposate(double &a           ) { return a; }
inline double &setnegate(double &a           ) { return ( a *= -1 ); }
inline double &setconj  (double &a           ) { return a; }
inline double &setrand  (double &a           ) { return ( a = ((double) rand())/RAND_MAX ); } //return ( a = ((double) svm_rand())/SVM_RAND_MAX ); }
inline double &postProInnerProd(double &a) { return a; }

inline int &setident (int &a        ) { return ( a = 1 ); }
inline int &setzero  (int &a        ) { return ( a = 0 ); }
inline int &setposate(int &a        ) { return a; }
inline int &setnegate(int &a        ) { return ( a *= -1 ); }
inline int &setconj  (int &a        ) { return a; }
inline int &setrand  (int &a        ) { return ( a = (2*(rand()%2))-1 ); } //return ( a = (2*(svm_rand()%2))-1 ); }
inline int &postProInnerProd(int &a)  { return a; }

inline unsigned int &setident (unsigned int &a                 ) { return ( a = 1 ); }
inline unsigned int &setzero  (unsigned int &a                 ) { return ( a = 0 ); }
inline unsigned int &setposate(unsigned int &a                 ) { return a; }
inline unsigned int &setnegate(unsigned int &a                 ) { NiceAssert( a == 0 ); return a = 0; }
inline unsigned int &setconj  (unsigned int &a                 ) { return a; }
inline unsigned int &setrand  (unsigned int &a                 ) { return ( a = (rand()%2) ); } //return ( a = (svm_rand()%2) ); }
inline unsigned int &postProInnerProd(unsigned int &a)           { return a; }

inline size_t &setident (size_t &a           ) { return ( a = 1 ); }
inline size_t &setzero  (size_t &a           ) { return ( a = 0 ); }
inline size_t &setposate(size_t &a           ) { return a; }
inline size_t &setnegate(size_t &a           ) { NiceAssert( a == 0 ); return a = 0; }
inline size_t &setconj  (size_t &a           ) { return a; }
inline size_t &setrand  (size_t &a           ) { return ( a = (rand()%2) ); } //return ( a = (svm_rand()%2) ); }
inline size_t &postProInnerProd(size_t &a)     { return a; }

inline char &setident (char &a         ) { return a = 1; }
inline char &setzero  (char &a         ) { return a = 0; }
inline char &setposate(char &a         ) { return a; }
inline char &setnegate(char &a         ) { return a = static_cast<char>(-a); }
inline char &setconj  (char &a         ) { return a; }
inline char &setrand  (char &a         ) { return a = static_cast<char>((2*(rand()%2))-1); } //return a = static_cast<char>((2*(svm_rand()%2))-1); }
inline char &postProInnerProd(char &a)   { return a; }

inline std::string &setident (std::string &a                ) { NiceThrow("setident string is meaningless");            return a; }
inline std::string &setzero  (std::string &a                ) { a = "";                                                 return a; }
inline std::string &setposate(std::string &a                ) {                                                         return a; }
inline std::string &setnegate(std::string &a                ) { NiceThrow("setnegate string is meaningless");           return a; }
inline std::string &setconj  (std::string &a                ) { NiceThrow("setconj string is meaningless");             return a; }
inline std::string &setrand  (std::string &a                ) { NiceThrow("setrand string is meaningless");             return a; }
inline std::string &postProInnerProd(std::string &a)          { return a; }


#endif

