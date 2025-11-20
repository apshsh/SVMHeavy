
//
// Matrix class
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _matrix_h
#define _matrix_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <sstream>
#include <limits.h>
#include "randfun.hpp"
#include "vector.hpp"
#include "sparsevector.hpp"




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



template <class T> class Matrix;
template <class T> class retMatrix;

#define MATRIX_ZTOL 1e-10
// max recursion depth when calculating determinant before invoking LUP decomposition alternative method
#define MATRIX_DETRECURSEMAX 6
#define CONVSCALE 1e-4

OVERLAYMAKEFNVECTOR(Matrix<int>)
OVERLAYMAKEFNVECTOR(Matrix<double>)
OVERLAYMAKEFNVECTOR(Vector<Matrix<int> >)
OVERLAYMAKEFNVECTOR(Vector<Matrix<double> >)

// Constant type matrices
//
// zero*matrix:  [ 0 0 0 ... ]
//               [ 0 0 0 ... ]
//               [ : : :     ]
// one*matrix:   [ 1 1 1 ... ]
//               [ 1 1 1 ... ]
//               [ : : :     ]
// cnt*matrix:   [ 0 1 2 ... ]
//               [ 0 1 2 ... ]
//               [ : : :     ]
// delta*matrix: [ 0 0 ... 0 1 0 ... ]
//               [ 0 0 ... 0 1 0 ... ]
//               [ : :     : : :     ]

inline const Matrix<int> &zerointmatrix(int numRows, int numCols, retMatrix<int> &tmpm);
inline const Matrix<int> &oneintmatrix (int numRows, int numCols, retMatrix<int> &tmpm);

inline const Matrix<double> &zerodoublematrix(int numRows, int numCols, retMatrix<double> &tmpm);
inline const Matrix<double> &onedoublematrix (int numRows, int numCols, retMatrix<double> &tmpm);

// basic versions

inline const Matrix<int> &zerointmatrixbasic(void);
inline const Matrix<int> &oneintmatrixbasic (void);

inline const Matrix<double> &zerodoublematrixbasic(void);
inline const Matrix<double> &onedoublematrixbasic (void);

// Swap function

template <class T> void qswap(Matrix<T> &a, Matrix<T> &b);
template <class T> void qswap(Matrix<T> *&a, Matrix<T> *&b);
template <class T> void qswap(const Matrix<T> *&a, const Matrix<T> *&b);

// Matrix return handle.

template <class T> class retMatrix;

template <class T>
class retMatrix : public Matrix<T>
{
public:
    explicit retMatrix() : Matrix<T>("&") { }

    // This function resets the return matrix to clean-slate.  No need to
    // call this as it gets called when required by operator().

    void reset(Matrix<T> &cover);
    DynArray<int> *reset_r(Matrix<T> &cover, int rowpivotsize);
    DynArray<int> *reset_c(Matrix<T> &cover, int colpivotsize);
    DynArray<int> *reset_rc(Matrix<T> &cover, DynArray<int> *&tpivCol, int rowpivotsize, int colpivotsize); // tpivRow returned

    void creset(const Matrix<T> &cover);
    DynArray<int> *creset_r(const Matrix<T> &cover, int rowpivotsize);
    DynArray<int> *creset_c(const Matrix<T> &cover, int colpivotsize);
    DynArray<int> *creset_rc(const Matrix<T> &cover, DynArray<int> *&tpivCol, int rowpivotsize, int colpivotsize); // tpivRow returned
};

// The class itself

template <class T>
class Matrix
{
    friend class retMatrix<T>;

    template <class S> friend void qswap(Matrix<S> &a, Matrix<S> &b);

    friend inline const Matrix<int> &zerointmatrix(int numRows, int numCols, retMatrix<int> &tmpm);
    friend inline const Matrix<int> &oneintmatrix (int numRows, int numCols, retMatrix<int> &tmpm);

    friend inline const Matrix<double> &zerodoublematrix(int numRows, int numCols, retMatrix<double> &tmpm);
    friend inline const Matrix<double> &onedoublematrix (int numRows, int numCols, retMatrix<double> &tmpm);

    friend inline const Matrix<int> &zerointmatrixbasic(void);
    friend inline const Matrix<int> &oneintmatrixbasic (void);

    friend inline const Matrix<double> &zerodoublematrixbasic(void);
    friend inline const Matrix<double> &onedoublematrixbasic (void);

public:

    // Constructors and Destructors
    //
    // - (numRows,numCols): matrix is size numRows * numCols (default 0*0)
    // - (elm,row,dref,celm,crow,cdref,numRows,numCols): like above, but
    //   rather than being stored locally the contents are stored elsewhere
    //   and can be retrieved by calling elm(i,j,dref), row(i,dref)
    //   (writable) or celm(i,j,cdref), crow(i,cdref) (fixed).
    // - (src): copy constructor

    explicit Matrix();
    explicit Matrix(const T &); // this one just makes an empty matrix, but the additional ("zero") argument is required by kcache
    explicit Matrix(int numRows, int numCols = 0);
    explicit Matrix(                                                     const T &(*celm)(int, int, const void *, retVector<T> &), const Vector<T> &(*crow)(int, const void *, retVector<T> &), const void *cdref = nullptr, int numRows = 0, int numCols = 0,                                   void (*xcdelfn)(const void *, void *) = nullptr, void *dref = nullptr, const Vector<T> *_Lweight = nullptr, bool _useLweight = false);
    explicit Matrix(T (*celm_v)(int, int, const void *, retVector<T> &), const T &(*celm)(int, int, const void *, retVector<T> &), const Vector<T> &(*crow)(int, const void *, retVector<T> &), const void *cdref = nullptr, int numRows = 0, int numCols = 0,                                   void (*xcdelfn)(const void *, void *) = nullptr, void *dref = nullptr, const Vector<T> *_Lweight = nullptr, bool _useLweight = false);
    explicit Matrix(T &(*elm)(int, int, void *, retVector<T> &), Vector<T> &(*row)(int, void *, retVector<T> &), void *dref,
                                                                         const T &(*celm)(int, int, const void *, retVector<T> &), const Vector<T> &(*crow)(int, const void *, retVector<T> &), const void *cdref = nullptr, int numRows = 0, int numCols = 0, void (*xdelfn)(void *) = nullptr, void (*xcdelfn)(const void *, void *) = nullptr,                       const Vector<T> *_Lweight = nullptr, bool _useLweight = false);
             Matrix(const Matrix<T> &src);

    ~Matrix();

    // Assignment
    //
    // - matrix assignment: unless this matrix is a temporary matrix created
    //   to refer to parts of another matrix then we do not require that sizes
    //   align but rather completely overwrite *this, resetting the size to
    //   that of the source.
    // - vector assignment: if numRows == 1 and numCols = the size of the src
    //   vector then this acts like assignment to the matrix from a row
    //   vector.  Otherwise acts as assignment from a column vector, resizing
    //   if required (if possible - ie. this isn't a temporary matrix created
    //   to refer to parts of another matrix).
    // - scalar assignment: in this case the size of the matrix remains
    //   unchanged, but all elements will be overwritten.
    // - behaviour is undefined if scalar is an element of this.

    Matrix<T> &operator=(const Matrix<T> &src);
    Matrix<T> &operator=(const Vector<T> &src);
    Matrix<T> &operator=(const T         &src);

    // simple matrix manipulations
    //
    // ident:      apply ident to diagonal elements of matrix and zero to off-
    //             diagonal elements
    // zero:       apply zero to all elements of the matrix (vectorially)
    // posate:     apply posate to all elements of the matrix (vectorially)
    // negate:     apply negate to all elements of the matrix (vectorially)
    // conj:       apply conj to all elements of the matrix (vectorially)
    // transpose:  transpose matrix
    // symmetrise: set this = 1/2 ( this + transpose(this) )
    //
    // all return *this

    Matrix<T> &ident(void);
    Matrix<T> &zero(void);
    Matrix<T> &posate(void);
    Matrix<T> &negate(void);
    Matrix<T> &conj(void);
    Matrix<T> &rand(void);
    Matrix<T> &transpose(void);
    Matrix<T> &symmetrise(void);

    // Access:
    //
    // - ("&",i,j) - access a reference to scalar element i,j
    // - (i,j)      - access a const reference to scalar element i,j
    //
    // Variants:
    //
    // - if i/j is of type Vector<int> then the reference returned is to the
    //   elements specified in i/j.
    // - if ib,is,im is given then this is the same as a vector i being used
    //   specified by: ( ib ib+is ib+is+is ... max_n(i=ib+(n*s)|i<im) )
    //   (and if im < ib then an empty reference is returned).  Same for j.
    // - if i is a vector or in ib/is/im form and j an int then for technical
    //   reasons (because the matrix is stored as row vectors) we must return
    //   a column matrix, not a vector.
    // - for disambiguation if i is in ib/is/im form and j is an int then an
    //   additional dummy ("&") argument is required as the last argument.
    // - the argument j is optional.  If j is not present then it is assumed
    //   that all columns are included.
    //
    // Scope of result:
    //
    // - The scope of the returned reference is the minimum of the scope of
    //   retVector/retMatrix &tmp or *this (or tmpa if present).
    // - retVector/retMatrix &tmp may or may not be used depending on, so
    //   never *assume* that it will be!
    // - The returned reference is actually *this through a layer of indirection,
    //   so any changes to it will be reflected in *this (and vice-versa).

    Matrix<T> &operator()(const char *                                                                                                                ) { return (*this);                                                   }
    Vector<T> &operator()(const char *dummy, int i,                                          retVector<T> &tmp, retVector<T> &tmpb                    );
    Matrix<T> &operator()(const char *dummy, const Vector<int> &i,                           retMatrix<T> &res                                        ) { return (*this)(dummy,i,              0,1,numCols()-1,res       ); }
    Matrix<T> &operator()(const char *dummy, int ib, int is, int im,                         retMatrix<T> &res                                        ) { return (*this)(dummy,ib,is,im,       0,1,numCols()-1,res       ); }
    Matrix<T> &operator()(const char *dummy,                         int j,                  retMatrix<T> &res, const char *                          ) { return (*this)(dummy,0,1,numRows()-1,j,1,j,          res       ); }
    T         &operator()(const char *dummy, int i,                  int j                                                                            );
    Matrix<T> &operator()(const char *dummy, const Vector<int> &i,   int j,                  retMatrix<T> &res                                        ) { return (*this)(dummy,i,              j,1,j,          res       ); }
    Matrix<T> &operator()(const char *dummy, int ib, int is, int im, int j,                  retMatrix<T> &res, const char *                          ) { return (*this)(dummy,ib,is,im,       j,1,j,          res       ); }
    Matrix<T> &operator()(const char *dummy,                         const Vector<int> &j,   retMatrix<T> &res, const char *                          ) { return (*this)(dummy,0,1,numRows()-1,j,              res       ); }
    Vector<T> &operator()(const char *dummy, int i,                  const Vector<int> &j,   retVector<T> &tmp, retVector<T> &tmpb, retVector<T> &tmpc);
    Matrix<T> &operator()(const char *dummy, const Vector<int> &i,   const Vector<int> &j,   retMatrix<T> &res                                        );
    Matrix<T> &operator()(const char *dummy, int ib, int is, int im, const Vector<int> &j,   retMatrix<T> &res                                        );
    Matrix<T> &operator()(const char *dummy,                         int jb, int js, int jm, retMatrix<T> &res, const char *                          ) { return (*this)(dummy,0,1,numRows()-1,jb,js,jm,       res       ); }
    Vector<T> &operator()(const char *dummy, int i,                  int jb, int js, int jm, retVector<T> &tmp, retVector<T> &tmpb                    );
    Matrix<T> &operator()(const char *dummy, const Vector<int> &i,   int jb, int js, int jm, retMatrix<T> &res                                        );
    Matrix<T> &operator()(const char *dummy, int ib, int is, int im, int jb, int js, int jm, retMatrix<T> &res                                        );

    const Matrix<T> &operator()(void                                                                                                     ) const { return (*this);                                           }
    const Vector<T> &operator()(int i,                                          retVector<T> &tmp, retVector<T> &tmpb                    ) const;
    const Matrix<T> &operator()(const Vector<int> &i,                           retMatrix<T> &res                                        ) const { return (*this)(i,              0,1,numCols()-1,res     ); }
    const Matrix<T> &operator()(int ib, int is, int im,                         retMatrix<T> &res                                        ) const { return (*this)(ib,is,im,       0,1,numCols()-1,res     ); }
    const Matrix<T> &operator()(                        int j,                  retMatrix<T> &res, const char *                          ) const { return (*this)(0,1,numRows()-1,j,1,j,          res     ); }
    const T         &operator()(int i,                  int j                                                                            ) const;
    const Matrix<T> &operator()(const Vector<int> &i,   int j,                  retMatrix<T> &res                                        ) const { return (*this)(i,              j,1,j,          res     ); }
    const Matrix<T> &operator()(int ib, int is, int im, int j,                  retMatrix<T> &res, const char *                          ) const { return (*this)(ib,is,im,       j,1,j,          res     ); }
    const Matrix<T> &operator()(                        const Vector<int> &j,   retMatrix<T> &res, const char *                          ) const { return (*this)(0,1,numRows()-1,j,              res     ); }
    const Vector<T> &operator()(int i,                  const Vector<int> &j,   retVector<T> &tmp, retVector<T> &tmpb, retVector<T> &tmpc) const;
    const Matrix<T> &operator()(const Vector<int> &i,   const Vector<int> &j,   retMatrix<T> &res                                        ) const;
    const Matrix<T> &operator()(int ib, int is, int im, const Vector<int> &j,   retMatrix<T> &res                                        ) const;
    const Matrix<T> &operator()(                        int jb, int js, int jm, retMatrix<T> &res, const char *                          ) const { return (*this)(0,1,numRows()-1,jb,js,jm,       res     ); }
    const Vector<T> &operator()(int i,                  int jb, int js, int jm, retVector<T> &tmp, retVector<T> &tmpb                    ) const;
    const Matrix<T> &operator()(const Vector<int> &i,   int jb, int js, int jm, retMatrix<T> &res                                        ) const;
    const Matrix<T> &operator()(int ib, int is, int im, int jb, int js, int jm, retMatrix<T> &res                                        ) const;

    Matrix<T> &operator()(const char *dummy, const Vector<int> &i,                           retMatrix<T> &res,                                         int ib, int is, int im                        ) { return (*this)(dummy,i,0,1,numCols()-1,res,ib,is,im); }
    Matrix<T> &operator()(const char *dummy, const Vector<int> &i,   int j,                  retMatrix<T> &res,                                         int ib, int is, int im                        ) { return (*this)(dummy,i,j,1,j,res,ib,is,im); }
    Matrix<T> &operator()(const char *dummy,                         const Vector<int> &j,   retMatrix<T> &res, const char *,                                                   int jb, int js, int jm) { return (*this)(dummy,0,1,numRows()-1,j,res,jb,js,jm); }
    Vector<T> &operator()(const char *dummy, int i,                  const Vector<int> &j,   retVector<T> &tmp, retVector<T> &tmpb, retVector<T> &tmpc,                         int jb, int js, int jm);
    Matrix<T> &operator()(const char *dummy, const Vector<int> &i,   const Vector<int> &j,   retMatrix<T> &res,                                         int ib, int is, int im, int jb, int js, int jm);
    Matrix<T> &operator()(const char *dummy, const Vector<int> &i,   const Vector<int> &j,   retMatrix<T> &res,                                         int ib, int is, int im                        );
    Matrix<T> &operator()(const char *dummy, const Vector<int> &i,   const Vector<int> &j,   retMatrix<T> &res,                                         const char *dummyb,     int jb, int js, int jm);
    Matrix<T> &operator()(const char *dummy, int ib, int is, int im, const Vector<int> &j,   retMatrix<T> &res,                                                                 int jb, int js, int jm);
    Matrix<T> &operator()(const char *dummy, const Vector<int> &i,   int jb, int js, int jm, retMatrix<T> &res,                                         int ib, int is, int im                        );

    const Matrix<T> &operator()(const Vector<int> &i,                           retMatrix<T> &res,                                         int ib, int is, int im                        ) const { return (*this)(i,0,1,numCols()-1,res,ib,is,im); }
    const Matrix<T> &operator()(const Vector<int> &i,   int j,                  retMatrix<T> &res,                                         int ib, int is, int im                        ) const { return (*this)(i,j,1,j,res,ib,is,im); }
    const Matrix<T> &operator()(                        const Vector<int> &j,   retMatrix<T> &res, const char *,                                                   int jb, int js, int jm) const { return (*this)(0,1,numRows()-1,j,res,jb,js,jm); }
    const Vector<T> &operator()(int i,                  const Vector<int> &j,   retVector<T> &tmp, retVector<T> &tmpb, retVector<T> &tmpc,                         int jb, int js, int jm) const;
    const Matrix<T> &operator()(const Vector<int> &i,   const Vector<int> &j,   retMatrix<T> &res,                                         int ib, int is, int im, int jb, int js, int jm) const;
    const Matrix<T> &operator()(const Vector<int> &i,   const Vector<int> &j,   retMatrix<T> &res,                                         int ib, int is, int im                        ) const;
    const Matrix<T> &operator()(const Vector<int> &i,   const Vector<int> &j,   retMatrix<T> &res,                                         const char *dummyb,     int jb, int js, int jm) const;
    const Matrix<T> &operator()(int ib, int is, int im, const Vector<int> &j,   retMatrix<T> &res,                                                                 int jb, int js, int jm) const;
    const Matrix<T> &operator()(const Vector<int> &i,   int jb, int js, int jm, retMatrix<T> &res,                                         int ib, int is, int im                        ) const;

    T     v(int i, int j     ) const;
    void sv(int i, int j, T x);

    // row-concantenated linear access

          T &r(const char *dummy, int i)       { return (*this)(dummy,i/dnumCols,i%dnumCols); }
    const T &r(                   int i) const { return (*this)(i/dnumCols,i%dnumCols);       }

    T     v(int i     ) const { return v (i/dnumCols,i%dnumCols  ); }
    void sv(int i, T x)       {        sv(i/dnumCols,i%dnumCols,x); }

    // Row and column norms

    double getColNorm(int i) const;
    double getRowNorm(int i) const;
    double getRowColNorm(void) const;

    double getColAbs(int i) const;
    double getRowAbs(int i) const;
    double getRowColAbs(void) const;

    // Add and remove element functions.
    //
    // Note that these may not be applied to temporary matrices.  Each
    // returns a reference to *this
    //
    // PadCol: adds n columns to right and zeros them
    // PadRow: adds n rows to bottom and zeros them
    // PadRowCol: adds n columns to right and bottom and zeros them

    Matrix<T> &addRow   (int i);
    Matrix<T> &removeRow(int i);

    Matrix<T> &addCol   (int i);
    Matrix<T> &removeCol(int i);

    Matrix<T> &addRowCol   (int i);
    Matrix<T> &removeRowCol(int i);

    Matrix<T> &resize(int targNumRows, int targNumCols);
    template <class S> Matrix<T> &resize(const Matrix<S> &sizeTemplateUsed) { return resize(sizeTemplateUsed.numRows(),sizeTemplateUsed.numCols()); }

    Matrix<T> &appendRow(int rowStart, const Matrix<T> &src);
    Matrix<T> &appendCol(int colStart, const Matrix<T> &src);

    Matrix<T> &padCol   (int n);
    Matrix<T> &padRow   (int n);
    Matrix<T> &padRowCol(int n);

    // Function application - apply function fn to each element of matrix

    Matrix<T> &applyon(T (*fn)(T));
    Matrix<T> &applyon(T (*fn)(const T &));
    Matrix<T> &applyon(T (*fn)(T, const void *), const void *a);
    Matrix<T> &applyon(T (*fn)(const T &, const void *), const void *a);
    Matrix<T> &applyon(T &(*fn)(T &));
    Matrix<T> &applyon(T &(*fn)(T &, const void *), const void *a);

    // Other functions
    //
    // - rankone adds c*outerproduct(a,b) to the matrix without wasting memory
    //   by calling outerproduct(a,b), creating a temporary matrix and then
    //   adding that.
    // - diagoffset G += diag(d) or G += c.diag(d)
    // - sqdiagoffset G += diag(d).diag(d) or G += diag(c.d).diag(c.d)
    // - SVD: computes singular value decomposition, assuming real, to find
    //   UDV, where U is left-orthogonal, V right-orthogonal and D diagonal.
    //
    // - naiveChol: naive cholesky factorisation.  Calculates L and stores it
    //   in dest, where G = L.L'.  The matrix is assumed to be symmetric
    //   positive definite and no isnan checking is done. The result is written
    //   into the lower triangular part of dest.  A ref to dest is returned.
    //   * diagsign: if given then G = L.diag(d).L', where d = diagsign is a
    //     vector of +-1 elements.
    //   * diagoffscal, diagoffset: if given then G + c.diag(o) = L.L', where
    //     c = diagoffscal and o is diagoffset.
    //   * zeroupper: if set then the upper triangular part of the result is
    //     zeroed, otherwise the upper triangular part is undefined (which may
    //     be OK so long as you're not using it).
    // - naivepartChol: Like naiveChol, but does maximal partial factorisation
    //   a pivotted version of G.  It sets the pivot vector p and the
    //   factorisation size n (n is maxed to the largest value for which the
    //   factorisation is defined), so G(p,p) = L(p,p(0:1:n-1))*L(p,p(0:1:n-1))'.
    //
    // - forwardElim: Solve L*y = b for y, where L is the lower triangular
    //   part of the matrix (must by square).  Returns reference to y.
    //   * implicitTranspose: if set use L = transpose of upper triangular
    //     part of the matrix.
    // - backwardSubst: Solve U*y = b for y, where U is the upper triangular
    //   part of the matrix (must by square).  Returns reference to y.
    //   * implicitTranspose: if set use U = transpose of lower triangular
    //     part of the matrix.
    //
    // - naiveCholInve: constructs the cholesky factorisation using
    //   naiveChol, assuming symmetric positive definite (and no
    //   checking) and then uses forwardElim and backwardSubst to find y,
    //   where G.y = b:
    //   * yinterm: if given then the intermediate result yinterm, where
    //     L.yinterm = b, is stored in yinterm (avoiding constructors).
    //   * cholres: if given then the cholesky factorisation is stored
    //     here (avoiding constructors).
    //   * cholresisprecalced: if set then assume that the Cholesky
    //     factorisation is already pre-calculated and stored in cholres,
    //     thus avoiding the (expensive, N^3) operation of calculating
    //     the factorisation.
    //   * zeroupper, diagsign, diagoffscal, diagoffset: as per naiveChol.
    //   * dummy is an unused argument to differentiate between versions
    //
    // - offNaiveCholInve: This aims to find x, where (G-diag(s)).y = b.
    //   However, unlike naiveCholInve, this works by factorising G
    //   rather than G-diag(s).  By doing this, you can do lots of
    //   inversions for different s vectors at the cost of a single
    //   cholesky factorisation.  It works as follows:
    //
    //        y = inv(G-diag(s)).b
    //          = inv(I-inv(G).diag(s)).(inv(G).b)
    //          = inv(I-inv(G).diag(s)).x0
    //
    //   where:
    //
    //        x0 = inv(G).b (this is done by factorising G)
    //
    //   hence, using the Woodbury matrix identity:
    //
    //        inv(I-UCV) = I - U.inv(-inv(C)+VU).V
    //
    //   let C=V=I, so:
    //
    //        inv(I-U) = I - U.inv(-I+U)
    //                 = I + U.inv(I-U)
    //
    //   and, letting U = inv(G).diag(s):
    //
    //        inv(I-inv(G).diag(s)) = I + inv(G).diag(s).inv(I-(inv(G).diag(s)))
    //                              = I + inv(G).diag(s).( I + inv(G).diag(s).inv(I-(inv(G).diag(s))) )
    //                              = I + inv(G).diag(s) + (inv(G).diag(s)).^2.inv(I-(inv(G).diag(s)))
    //                              = I + inv(G).diag(s) + (inv(G).diag(s)).^2.( I + inv(G).diag(s).inv(I-(inv(G).diag(s))) )
    //                              = I + inv(G).diag(s) + (inv(G).diag(s)).^2 + (inv(G).diag(s))^3 + inv(G).diag(s).inv(I-(inv(G).diag(s)))
    //                                ...
    //                              = I + inv(G).diag(s) + (inv(G).diag(s)).^2 + (inv(G).diag(s))^3 + (inv(G).diag(s))^4 + ...
    //
    //   from which we see that:
    //
    //        y = x0 + x1 + x2 + ...
    //
    //   where:
    //
    //        x0 = inv(G).b          (this is done by factorising G)
    //        x1 = inv(G).diag(s).x0 (this is done by factorising G)
    //        x2 = inv(G).diag(s).x1 (this is done by factorising G)
    //        x3 = inv(G).diag(s).x2 (this is done by factorising G)
    //             ...
    //
    //   Each step in this process is cost n^2.  Provided we can terminate this
    //   series in fewer than n steps, the computational cost will be less than
    //   the "usual" factorisation if the cholesky factorisation of G is given.
    //
    //   With regard to convergence, suppose that G is positive definite and
    //   diagonally offset by r.  Then we form a factorisation:
    //
    //        L.L' = G + diag(r)
    //
    //   Provided 0 <= s <= r, each step x0 -> x1 -> x2 -> ... will result in a
    //   decrease in the norm of the next vector, so convergence *should* work.
    //   NB: *No attempt is made to enforce convergence, so use with care!*

    Matrix<T> &rankone(const T &c, const Vector<T> &a, const Vector<T> &b);

    Matrix<T> &diagoffset(const T &d);
    Matrix<T> &diagoffset(const Vector<T> &d);
    Matrix<T> &diagoffset(const T &c, const Vector<T> &d);
    Matrix<T> &diagoffset(const Matrix<T> &d);

    Matrix<T> &sqdiagoffset(const T &d);
    Matrix<T> &sqdiagoffset(const Vector<T> &d);
    Matrix<T> &sqdiagoffset(const T &c, const Vector<T> &d);
    Matrix<T> &sqdiagoffset(const Matrix<T> &d);

    Matrix<T> &SVD(Matrix<T> &u, Vector<T> &d, Matrix<T> &v) const;

    Matrix<T> &naiveChol(                           Matrix<T> &dest,                                                  int zeroupper = 1, double ztol = MATRIX_ZTOL) const;
    Matrix<T> &naiveChol(                           Matrix<T> &dest, double diagoffscal, const Vector<T> &diagoffset, int zeroupper = 1, double ztol = MATRIX_ZTOL) const;
    Matrix<T> &naiveChol(const Vector<T> &diagsign, Matrix<T> &dest,                                                  int zeroupper = 1, double ztol = MATRIX_ZTOL) const;
    Matrix<T> &naiveChol(const Vector<T> &diagsign, Matrix<T> &dest, double diagoffscal, const Vector<T> &diagoffset, int zeroupper = 1, double ztol = MATRIX_ZTOL) const;

    Matrix<T> &naivepartChol(                           Matrix<T> &dest,                                                  Vector<int> &p, int &n, int zeroupper = 1, double ztol = MATRIX_ZTOL) const;
    Matrix<T> &naivepartChol(                           Matrix<T> &dest, double diagoffscal, const Vector<T> &diagoffset, Vector<int> &p, int &n, int zeroupper = 1, double ztol = MATRIX_ZTOL) const;
    Matrix<T> &naivepartChol(const Vector<T> &diagsign, Matrix<T> &dest,                                                  Vector<int> &p, int &n, int zeroupper = 1, double ztol = MATRIX_ZTOL) const;
    Matrix<T> &naivepartChol(const Vector<T> &diagsign, Matrix<T> &dest, double diagoffscal, const Vector<T> &diagoffset, Vector<int> &p, int &n, int zeroupper = 1, double ztol = MATRIX_ZTOL) const;

    template <class S> Vector<S> &forwardElim  (Vector<S> &y, const Vector<S> &b, int implicitTranspose = 0) const;
    template <class S> Vector<S> &backwardSubst(Vector<S> &y, const Vector<S> &b, int implicitTranspose = 0) const;

    template <class S> Vector<S> &naiveCholInve(                                              Vector<S> &y, const Vector<S> &b                                                                                                                                  ) const;
    template <class S> Vector<S> &naiveCholInve(                                              Vector<S> &y, const Vector<S> &b,                                                  Vector<S> &yinterm                                                             ) const;
    template <class S> Vector<S> &naiveCholInve(                                              Vector<S> &y, const Vector<S> &b,                                                                      Matrix<T> & cholres, int zeroupper, int &cholresisprecalced) const;
    template <class S> Vector<S> &naiveCholInve(                                              Vector<S> &y, const Vector<S> &b,                                                  Vector<S> &yinterm, Matrix<T> & cholres, int zeroupper, int &cholresisprecalced) const;
    template <class S> Vector<S> &naiveCholInve(                                              Vector<S> &y, const Vector<S> &b, double diagoffscal, const Vector<T> &diagoffset                                                                                 ) const;
    template <class S> Vector<S> &naiveCholInve(                                              Vector<S> &y, const Vector<S> &b, double diagoffscal, const Vector<T> &diagoffset, Vector<S> &yinterm                                                             ) const;
    template <class S> Vector<S> &naiveCholInve(                                              Vector<S> &y, const Vector<S> &b, double diagoffscal, const Vector<T> &diagoffset,                     Matrix<T> & cholres, int zeroupper, int &cholresisprecalced) const;
    template <class S> Vector<S> &naiveCholInve(                                              Vector<S> &y, const Vector<S> &b, double diagoffscal, const Vector<T> &diagoffset, Vector<S> &yinterm, Matrix<T> & cholres, int zeroupper, int &cholresisprecalced) const;
    template <class S> Vector<S> &naiveCholInve(const char *dummy, const Vector<T> &diagsign, Vector<S> &y, const Vector<S> &b                                                                                                                                  ) const;
    template <class S> Vector<S> &naiveCholInve(const char *dummy, const Vector<T> &diagsign, Vector<S> &y, const Vector<S> &b,                                                  Vector<S> &yinterm                                                             ) const;
    template <class S> Vector<S> &naiveCholInve(const char *dummy, const Vector<T> &diagsign, Vector<S> &y, const Vector<S> &b,                                                                      Matrix<T> & cholres, int zeroupper, int &cholresisprecalced) const;
    template <class S> Vector<S> &naiveCholInve(const char *dummy, const Vector<T> &diagsign, Vector<S> &y, const Vector<S> &b,                                                  Vector<S> &yinterm, Matrix<T> & cholres, int zeroupper, int &cholresisprecalced) const;
    template <class S> Vector<S> &naiveCholInve(const char *dummy, const Vector<T> &diagsign, Vector<S> &y, const Vector<S> &b, double diagoffscal, const Vector<T> &diagoffset                                                                                 ) const;
    template <class S> Vector<S> &naiveCholInve(const char *dummy, const Vector<T> &diagsign, Vector<S> &y, const Vector<S> &b, double diagoffscal, const Vector<T> &diagoffset, Vector<S> &yinterm                                                             ) const;
    template <class S> Vector<S> &naiveCholInve(const char *dummy, const Vector<T> &diagsign, Vector<S> &y, const Vector<S> &b, double diagoffscal, const Vector<T> &diagoffset,                     Matrix<T> & cholres, int zeroupper, int &cholresisprecalced) const;
    template <class S> Vector<S> &naiveCholInve(const char *dummy, const Vector<T> &diagsign, Vector<S> &y, const Vector<S> &b, double diagoffscal, const Vector<T> &diagoffset, Vector<S> &yinterm, Matrix<T> & cholres, int zeroupper, int &cholresisprecalced) const;

    template <class S> Vector<S> &offNaiveCholInve(                                              Vector<S> &y, const Vector<S> &b                                                                                                              , double sscal, const Vector<T> &s                                                                                   ) const;
    template <class S> Vector<S> &offNaiveCholInve(                                              Vector<S> &y, const Vector<S> &b,                                                  Vector<S> &yinterm, Vector<S> &xinterm,                      double sscal, const Vector<T> &s                                                                                   ) const;
    template <class S> Vector<S> &offNaiveCholInve(                                              Vector<S> &y, const Vector<S> &b,                                                                                          Matrix<T> & cholres, double sscal, const Vector<T> &s, int zeroupper, int &cholresisprecalced, int &fbused, double convScale = CONVSCALE) const;
    template <class S> Vector<S> &offNaiveCholInve(                                              Vector<S> &y, const Vector<S> &b,                                                  Vector<S> &yinterm, Vector<S> &xinterm, Matrix<T> & cholres, double sscal, const Vector<T> &s, int zeroupper, int &cholresisprecalced, int &fbused, double convScale = CONVSCALE) const;
    template <class S> Vector<S> &offNaiveCholInve(                                              Vector<S> &y, const Vector<S> &b, double diagoffscal, const Vector<T> &diagoffset                                                             , double sscal, const Vector<T> &s                                                                                   ) const;
    template <class S> Vector<S> &offNaiveCholInve(                                              Vector<S> &y, const Vector<S> &b, double diagoffscal, const Vector<T> &diagoffset, Vector<S> &yinterm, Vector<S> &xinterm,                      double sscal, const Vector<T> &s                                                                                   ) const;
    template <class S> Vector<S> &offNaiveCholInve(                                              Vector<S> &y, const Vector<S> &b, double diagoffscal, const Vector<T> &diagoffset,                                         Matrix<T> & cholres, double sscal, const Vector<T> &s, int zeroupper, int &cholresisprecalced, int &fbused, double convScale = CONVSCALE) const;
    template <class S> Vector<S> &offNaiveCholInve(                                              Vector<S> &y, const Vector<S> &b, double diagoffscal, const Vector<T> &diagoffset, Vector<S> &yinterm, Vector<S> &xinterm, Matrix<T> & cholres, double sscal, const Vector<T> &s, int zeroupper, int &cholresisprecalced, int &fbused, double convScale = CONVSCALE) const;
    template <class S> Vector<S> &offNaiveCholInve(const char *dummy, const Vector<T> &diagsign, Vector<S> &y, const Vector<S> &b                                                                                                              , double sscal, const Vector<T> &s                                                                                   ) const;
    template <class S> Vector<S> &offNaiveCholInve(const char *dummy, const Vector<T> &diagsign, Vector<S> &y, const Vector<S> &b,                                                  Vector<S> &yinterm, Vector<S> &xinterm,                      double sscal, const Vector<T> &s                                                                                   ) const;
    template <class S> Vector<S> &offNaiveCholInve(const char *dummy, const Vector<T> &diagsign, Vector<S> &y, const Vector<S> &b,                                                                                          Matrix<T> & cholres, double sscal, const Vector<T> &s, int zeroupper, int &cholresisprecalced, int &fbused, double convScale = CONVSCALE) const;
    template <class S> Vector<S> &offNaiveCholInve(const char *dummy, const Vector<T> &diagsign, Vector<S> &y, const Vector<S> &b,                                                  Vector<S> &yinterm, Vector<S> &xinterm, Matrix<T> & cholres, double sscal, const Vector<T> &s, int zeroupper, int &cholresisprecalced, int &fbused, double convScale = CONVSCALE) const;
    template <class S> Vector<S> &offNaiveCholInve(const char *dummy, const Vector<T> &diagsign, Vector<S> &y, const Vector<S> &b, double diagoffscal, const Vector<T> &diagoffset                                                             , double sscal, const Vector<T> &s                                                                                   ) const;
    template <class S> Vector<S> &offNaiveCholInve(const char *dummy, const Vector<T> &diagsign, Vector<S> &y, const Vector<S> &b, double diagoffscal, const Vector<T> &diagoffset, Vector<S> &yinterm, Vector<S> &xinterm,                      double sscal, const Vector<T> &s                                                                                   ) const;
    template <class S> Vector<S> &offNaiveCholInve(const char *dummy, const Vector<T> &diagsign, Vector<S> &y, const Vector<S> &b, double diagoffscal, const Vector<T> &diagoffset,                                         Matrix<T> & cholres, double sscal, const Vector<T> &s, int zeroupper, int &cholresisprecalced, int &fbused, double convScale = CONVSCALE) const;
    template <class S> Vector<S> &offNaiveCholInve(const char *dummy, const Vector<T> &diagsign, Vector<S> &y, const Vector<S> &b, double diagoffscal, const Vector<T> &diagoffset, Vector<S> &yinterm, Vector<S> &xinterm, Matrix<T> & cholres, double sscal, const Vector<T> &s, int zeroupper, int &cholresisprecalced, int &fbused, double convScale = CONVSCALE) const;

    // Determinants, inverses, pseudoinverses etc.
    //
    // minor appears to be taken by some sort of macro (in gcc at least),
    // so using miner in its place.  Sorry 'bout that
    //
    // invtrace calculated 1/trace(inv(A)) = det()/sum_i(miner(i,i))
    //
    // LUPDecompose: construct LU decomposition.  A(P) = LU
    //                                         L lower triangular
    //                                         L has ones on diag
    //                                         U upper triangular
    //                                         P is the pivot vector
    //            res = ((L-I)+U)(P)
    //            where P is a permutation matrix returned as a vector
    //            returns S, where S is the number of permutations
    //            (if res missing then does it on this matrix)
    //            returns -1 on failure
    //
    // tridiag: calculate tridiagonal form, assuming real symmetric
    //
    //        d contains the diagonal elements of the tridiagonal matrix.
    //
    //        e contains the subdiagonal elements of the tridiagonal
    //          matrix in its last n-1 positions.  e(1) is set to zero.
    //
    //        z contains the orthogonal transformation matrix
    //          produced in the reduction.
    //
    //        e2 contains the squares of the corresponding elements of e.
    //          e2 may coincide with e if the squares are not needed.
    //
    // eig: calculates eigenvalues/vectors
    //
    //        w  contains the eigenvalues in ascending order.
    //
    //        z  contains the eigenvectors (eigenvectors are in columns).
    //
    //        return:  is an integer output variable set equal to an error
    //           completion code described in the documentation for tqlrat
    //           and tql2.  the normal completion code is zero.
    //
    //        fv1  and  fv2  are temporary storage arrays.
    //
    // projpsd: project onto nearest positive semidefinite matrix.  fv... are temporary
    //          method = 0 for spectral clip, 1 for spectral flip
    // projnsd: project onto nearest negative semidefinite matrix.  fv... are temporary
    //          method = 0 for spectral clip, 1 for spectral flip
    //
    // det_naiveChol: determinant calculation via naiveChol (see naiveChol)
    // logdet_naiveChol: log determinant calculation via naiveChol (see naiveChol)
    //
    // NB: - I make no guarantee that these will work for non-doubles.
    //     - Inverting matrices is a bad idea (tm) in general.
    //     - For non-square matrices, inverse will return conjugate transpose
    //       of the pseudoinverse

    T trace   (int maxsize = INT_MAX) const; // the argument is an optional "max size stop" (so do upper-left of no more than this size)
    T tracelog(int maxsize = INT_MAX) const; // sum of logs of diagonals
    T diagprod(int maxsize = INT_MAX) const;
    T det     (void) const;
    T invtrace(void) const;
    T miner(int i, int j) const;
    T cofactor(int i, int j) const;
    Matrix<T> &adj( Matrix<T> &res) const;
    Matrix<T> &inve(Matrix<T> &res) const;
    Matrix<T> &inveSymm(Matrix<T> &res) const;
    int LUPDecompose(Matrix<T> &res, Vector<int> &p, double ztol = MATRIX_ZTOL) const;
    int LUPDecompose(Vector<int> &p, double ztol);
    int LUPDecompose(double ztol = MATRIX_ZTOL);
    void tridiag(Vector<T> &d, Vector<T> &e, Vector<T> &e2);
    void tridiag(Vector<T> &d, Vector<T> &e, Matrix<T> &z ) const;
    int eig(Vector<T> &w, Vector<T> &fv1, Vector<T> &fv2) const;
    int eig(Vector<T> &w, Matrix<T> &z,   Vector<T> &fv1) const;
    int projpsd(Matrix<T> &res, Vector<T> &fv1, Matrix<T> &fv2, Vector<T> &fv3, int method = 0) const;
    int projnsd(Matrix<T> &res, Vector<T> &fv1, Matrix<T> &fv2, Vector<T> &fv3, int method = 0) const;

    Matrix<T> adj( void) const;
    Matrix<T> inve(void) const;
    Matrix<T> inveSymm(void) const;

    T det_naiveChol(void                                                                      ) const;
    T det_naiveChol(                           double diagoffscal, const Vector<T> &diagoffset) const;
    T det_naiveChol(const Vector<T> &diagsign                                                 ) const;
    T det_naiveChol(const Vector<T> &diagsign, double diagoffscal, const Vector<T> &diagoffset) const;

    T det_naiveChol(                           Matrix<T> &factScratch                                                 ) const;
    T det_naiveChol(                           Matrix<T> &factScratch, double diagoffscal, const Vector<T> &diagoffset) const;
    T det_naiveChol(const Vector<T> &diagsign, Matrix<T> &factScratch                                                 ) const;
    T det_naiveChol(const Vector<T> &diagsign, Matrix<T> &factScratch, double diagoffscal, const Vector<T> &diagoffset) const;

    T logdet_naiveChol(void                                                                      ) const;
    T logdet_naiveChol(                           double diagoffscal, const Vector<T> &diagoffset) const;
    T logdet_naiveChol(const Vector<T> &diagsign                                                 ) const;
    T logdet_naiveChol(const Vector<T> &diagsign, double diagoffscal, const Vector<T> &diagoffset) const;

    T logdet_naiveChol(                           Matrix<T> &factScratch                                                 ) const;
    T logdet_naiveChol(                           Matrix<T> &factScratch, double diagoffscal, const Vector<T> &diagoffset) const;
    T logdet_naiveChol(const Vector<T> &diagsign, Matrix<T> &factScratch                                                 ) const;
    T logdet_naiveChol(const Vector<T> &diagsign, Matrix<T> &factScratch, double diagoffscal, const Vector<T> &diagoffset) const;

    // Other stuff:
    //
    // rowsum: sum of rows in matrix
    // colsum: sum of columns in matrix
    // vertsum: sum of elements in column j
    // horizsum: sum of elements in row i
    // scale: scale matrix by amount (*this *= a)

    const Vector<T> &rowsum(Vector<T> &res) const;
    const Vector<T> &colsum(Vector<T> &res) const;
    const T &vertsum(int j, T &res) const;
    const T &horizsum(int i, T &res) const;
    template <class S> Matrix<T> &scale(const S &a);
    template <class S> Matrix<T> &scaleAdd(const S &a, const Matrix<T> &b);

    // Information
    //
    // numRows()  = number of rows in matrix
    // numCols()  = number of columns in matrix
    // size()     = max(numRows(),numCols())
    // isSquare() = numRows() == numCols()
    // isEmpty()  = !numRows() && !numCols()

    int numRows(void)  const { return dnumRows;                                      }
    int numCols(void)  const { return dnumCols;                                      }
    int size(void)     const { return ( dnumCols > dnumRows ) ? dnumCols : dnumRows; }

    bool isSquare(void) const { return ( dnumRows == dnumCols );                      }
    bool isEmpty(void)  const { return ( !dnumCols && !dnumRows );                    }

    // pre-allocation function (see vector.h for more detail)

    void prealloc(int newallocrows, int newalloccols);
    void useStandardAllocation(void);
    void useTightAllocation(void);
    void useSlackAllocation(void);

    // Slight complication from vector.h

    bool shareBase(const Matrix<T> *that) const { return ( bkref == that->bkref ); }

    // Casting operator used by vector regression template
    //
    // This is actually needed when the elements of the kernel matrix in an 
    // SVM are themselves matrices (MSVR, division-algebraic SVR) but we
    // still need some measure of mean/median diagonals for automatic
    // parameter tuning.

    operator T() const { return (*this)(0,0); }

    // "Cheat" ways of setting the external evaluation arguments.  These are
    // needed when we start messing with swap functions at higher levels.

    void cheatsetdref(void *newdref)         { dref  = newdref;  }
    void cheatsetcdref(const void *newcdref) { cdref = newcdref; }

private:

    // dnumRows: the height of the matrix
    // dnumCols: the width of the matrix
    //
    // nbase: 0 if content is local, 1 if it points elsewhere
    //        (NB: if nbase == 0 then pivotRow = pivotCol = ( 0 1 2 ... ))
    // pbaseRow: 0 if pivotRow is local, 1 if it points elsewhere
    //           (NB: if nbase == 0 then pbaseRow == 0 by definition)
    // pbaseCol: 0 if pivotCol is local, 1 if it points elsewhere
    //           (NB: if nbase == 0 then pbaseCol == 0 by definition)
    //
    // iibRow: constant added to row indices
    // iisRow: step for row indices
    //
    // iibCol: constant added to column indices
    // iisCol: step for column indices
    //
    // bkref: if nbase, this is the matrix derived from (and pointed to)
    // content: contents of matrix
    // ccontent: constant pointer to content
    // pivotRow: row pivotting used to access contents
    // pivotCol: column pivotting used to access contents
    //
    // iscover: 0 if matrix normal, 1 if it gets data from elsewhere
    // elmfn: if iscover then this function is called to access data
    // rowfn: if iscover then this function is called to access rows of data
    // dref: argument passed to elmfn and rowfn, presumably giving details
    // celmfn: if iscover then this fn is called to access (const) data by reference
    // celmfn_v: if iscover then this fn is called to access (const) data by value
    // crowfn: if iscover then this fn is called to access rows of const data
    // cdref: argument passed to celmfn and crowfn, presumably giving details
    // Lweight: diagonal weight matrix Lweight.G.Lweight if non-null

    int dnumRows;
    int dnumCols;

    int iibRow;
    int iisRow;

    int iibCol;
    int iisCol;

    bool nbase;
    bool pbaseRow;
    bool pbaseCol;
    bool iscover;

    const Matrix<T> *bkref;
    Vector<Vector<T> > *content;
    const Vector<Vector<T> > *ccontent;
    const DynArray<int> *pivotRow;
    const DynArray<int> *pivotCol;

    T &(*elmfn)(int, int, void *, retVector<T> &);
    Vector<T> &(*rowfn)(int, void *, retVector<T> &);
    void (*delfn)(void *);
    void *dref;
    const T &(*celmfn)(int, int, const void *, retVector<T> &);
    T (*celmfn_v)(int, int, const void *, retVector<T> &);
    const Vector<T> &(*crowfn)(int, const void *, retVector<T> &);
    void (*cdelfn)(const void *, void *);
    const void *cdref;
    const Vector<T> *Lweight;
    bool useLweight;

    // Blind constructor: does no allocation, just sets bkref and defaults

    explicit Matrix(const char *dummy, const Matrix<T> &src);
    explicit Matrix(const char *dummy);
    explicit Matrix(const char *dummy, const Vector<T> &ccondummy); // zeromatrix etc

    // Fix bkref

    void fixbkreftree(const Matrix<T> *newbkref);

    // Fast (but could cause problem X) internal version

    Vector<T> &operator()(const char *dummy, int i, const char *dummyb, retVector<T> &tempdonotuse, retVector<T> &tmp);

    // I HAVE NO IDEA WHY THIS IS REQUIRED!  For some reason, member
    // functions can't "see" the overloads of conj contained in numbase.h.
    // It's almost as if the void member overloads of conj in this class
    // blocks the compiler from seeing the other options.  Hopefully this is
    // a temporary bug to be fixed in future releases of gcc.  Until now, we
    // need to do this.

    double conj(double a) const { return a; }
};

template <class T> void qswap(Matrix<T> &a, Matrix<T> &b)
{
    NiceAssert( a.nbase == false );
    NiceAssert( b.nbase == false );

    qswap(a.dnumRows  ,b.dnumRows  );
    qswap(a.dnumCols  ,b.dnumCols  );
    qswap(a.nbase     ,b.nbase     );
    qswap(a.pbaseRow  ,b.pbaseRow  );
    qswap(a.pbaseCol  ,b.pbaseCol  );
    qswap(a.iibRow    ,b.iibRow    );
    qswap(a.iisRow    ,b.iisRow    );
    qswap(a.iibCol    ,b.iibCol    );
    qswap(a.iisCol    ,b.iisCol    );
    qswap(a.iscover   ,b.iscover   );
    qswap(a.useLweight,b.useLweight);

    const Matrix<T> *bkref;
    Vector<Vector<T> > *content;
    const Vector<Vector<T> > *ccontent;
    const DynArray<int> *pivotRow;
    const DynArray<int> *pivotCol;

    bkref    = a.bkref;    a.bkref    = b.bkref;    b.bkref    = bkref;
    content  = a.content;  a.content  = b.content;  b.content  = content;
    ccontent = a.ccontent; a.ccontent = b.ccontent; b.ccontent = ccontent;
    pivotRow = a.pivotRow; a.pivotRow = b.pivotRow; b.pivotRow = pivotRow;
    pivotCol = a.pivotCol; a.pivotCol = b.pivotCol; b.pivotCol = pivotCol;

    T &(*elmfn)(int, int, void *, retVector<T> &);
    Vector<T> &(*rowfn)(int, void *, retVector<T> &);
    void (*delfn)(void *);
    void *dref;
    const T &(*celmfn)(int, int, const void *, retVector<T> &);
    T (*celmfn_v)(int, int, const void *, retVector<T> &);
    const Vector<T> &(*crowfn)(int, const void *, retVector<T> &);
    void (*cdelfn)(const void *, void *);
    const void *cdref;
    const Vector<T> *Lweight;

    elmfn    = a.elmfn;    a.elmfn    = b.elmfn;    b.elmfn    = elmfn;
    rowfn    = a.rowfn;    a.rowfn    = b.rowfn;    b.rowfn    = rowfn;
    delfn    = a.delfn;    a.delfn    = b.delfn;    b.delfn    = delfn;
    dref     = a.dref;     a.dref     = b.dref;     b.dref     = dref;
    celmfn   = a.celmfn;   a.celmfn   = b.celmfn;   b.celmfn   = celmfn;
    celmfn_v = a.celmfn_v; a.celmfn_v = b.celmfn_v; b.celmfn_v = celmfn_v;
    crowfn   = a.crowfn;   a.crowfn   = b.crowfn;   b.crowfn   = crowfn;
    cdelfn   = a.cdelfn;   a.cdelfn   = b.cdelfn;   b.cdelfn   = cdelfn;
    cdref    = a.cdref;    a.cdref    = b.cdref;    b.cdref    = cdref;
    Lweight  = a.Lweight;  a.Lweight  = b.Lweight;  b.Lweight  = Lweight;

    // The above will have messed up one important thing, namely bkref and
    // bkref in any child vectors.  We must now repair the child trees if
    // they exist

    a.fixbkreftree(&a);
    b.fixbkreftree(&b);
}

template <class T>
void qswap(Matrix<T> *&a, Matrix<T> *&b)
{
    Matrix<T> *c(a); a = b; b = c;
}

template <class T>
void qswap(const Matrix<T> *&a, const Matrix<T> *&b)
{
    const Matrix<T> *c(a); a = b; b = c;
}

template <class T>
void Matrix<T>::fixbkreftree(const Matrix<T> *newbkref)
{
    bkref = newbkref;
}





// Various functions
//
// max: find max element, put index in i,j.
// min: find min element, put index in i,j.
// maxdiag: find max diagonal element, put index in i,j.
// mindiag: find min diagonal element, put index in i,j.
// maxabs: find the |max| element, put index in i,j.
// minabs: find the |min| element, put index in i,j.
// maxabsdiag: find the |max| diagonal element, put index in i,j.
// minabsdiag: find the |min| diagonal element, put index in i,j.
// outerProduct: calculate the outer product of two vectors
//
// sum: find the sum of elements in a matrix.
// mean: calculate the mean of elements in a matrix.
// median: calculate the median of elements in a matrix.
//
// setident: apply ident to diagonal elements of matrix and zero to off-diagonal elements
// setzero: apply zero t0 all elements of matrix (vectorially)
// setposate: apply posate to all element of matrix (vectorially)
// setnegate: apply negate to all element of matrix (vectorially)
// setconj: apply conj to all element of matrix (vectorially)
// settranspose: apply transpose to all element of matrix (vectorially)
//
// inv: find the inverse of the matrix
// abs2: returns sum of abs2 of elements
// distF: returns the Frobenius distance

template <class T> const T &max(const Matrix<T> &right_op, int &i, int &j);
template <class T> const T &min(const Matrix<T> &right_op, int &i, int &j);
template <class T> const T &maxdiag(const Matrix<T> &right_op, int &i, int &j);
template <class T> const T &mindiag(const Matrix<T> &right_op, int &i, int &j);
template <class T> T maxabs(const Matrix<T> &right_op, int &i, int &j);
template <class T> T minabs(const Matrix<T> &right_op, int &i, int &j);
template <class T> T maxabsdiag(const Matrix<T> &right_op, int &i, int &j);
template <class T> T minabsdiag(const Matrix<T> &right_op, int &i, int &j);
template <class T> Matrix<T> outerProduct(const Vector<T> &left_op, const Vector<T> &right_op);
template <class T> const Matrix<T> &takeProduct(Matrix<T> &res, const Matrix<T> &a, const Matrix<T> &b);

template <class T> T diagsum(const Matrix<T> &right_op);
template <class T> T sum(const Matrix<T> &right_op);
template <class T> T sqsum(const Matrix<T> &right_op);
template <class T> T sqsum(T &res, const Matrix<T> &right_op);
template <class T> T mean(const Matrix<T> &right_op);
template <class T> const T &median(const Matrix<T> &right_op, int &i, int &j);

template <class T> Matrix<T> &setident(Matrix<T> &a);
template <class T> Matrix<T> &setzero(Matrix<T> &a);
template <class T> Matrix<T> &setposate(Matrix<T> &a);
template <class T> Matrix<T> &setnegate(Matrix<T> &a);
template <class T> Matrix<T> &setconj(Matrix<T> &a);
template <class T> Matrix<T> &setrand(Matrix<T> &a);
template <class T> Matrix<T> &settranspose(Matrix<T> &a);
template <class T> Matrix<T> &postProInnerProd(Matrix<T> &a) { return a; }

template <class T> Matrix<T> inv(const Matrix<T> &src);
template <class T> double distF(const Matrix<T> &a, const Matrix<T> &b);

template <class S> double norm1(const Matrix<S> &a);
template <class S> double norm2(const Matrix<S> &a);
template <class S> double normp(const Matrix<S> &a, double p);
template <class S> double normF(const Matrix<S> &a);

template <class S> double abs1  (const Matrix<S> &a);
template <class S> double abs2  (const Matrix<S> &a);
template <class S> double absp  (const Matrix<S> &a, double p);
template <class S> double absinf(const Matrix<S> &a);
template <class S> double abs0  (const Matrix<S> &a);
template <class S> double absF  (const Matrix<S> &a);

template <class S> Matrix<S> angle(const Matrix<S> &a);



// offeiginv: Let B be a matrix, expressed as an eigendecomposition:
//
// B = Q.diag(e).Q'
//
// where Q and e are given.  Assume a positive vector x, and consider:
//
// B = Q.diag(e).Q' + diag(x)
//
// Given r, we want to find:
//
// y = inv(B+diag(x)).r
//
//
// invoffeiginv: Let B be a matrix, expressed as an eigendecomposition:
//
// B = Q.diag(e).Q'
//
// where Q and e are given.  Assume a positive vector x, and consider:
//
// B = Q.diag(e).Q' + inv(diag(x))
//
// Given r, we want to find:
//
// y = inv(B+inv(diag(x))).r
//
//
// t,u are scratch vectors

template <class T> Vector<T> &eiginv      (const Matrix<T> &Q, const Vector<T> &e, Vector<T> &y, const Vector<T> &r);
template <class T> Vector<T> &offeiginv   (const Matrix<T> &Q, const Vector<T> &e, Vector<T> &y, const Vector<T> &r, const Vector<T> &x,    Vector<T> &t, Vector<T> &u);
template <class T> Vector<T> &offinveiginv(const Matrix<T> &Q, const Vector<T> &e, Vector<T> &y, const Vector<T> &r, const Vector<T> &xinv, Vector<T> &t, Vector<T> &u);


// NaN and inf tests

template <class T> int testisvnan(const Matrix<T> &x);
template <class T> int testisinf (const Matrix<T> &x);
template <class T> int testispinf(const Matrix<T> &x);
template <class T> int testisninf(const Matrix<T> &x);


// Conversion from strings

template <class T> Matrix<T> &atoMatrix(Matrix<T> &dest, const std::string &src);
template <class T> Matrix<T> &atoMatrix(Matrix<T> &dest, const std::string &src)
{
    std::istringstream srcmod(src);

    srcmod >> dest;

    return dest;
}

// Random permutation function and random fill
//
// ltfill: lhsres(i,j) = 1 if lhsres(i,j) <  rhs(i,j), 0 otherwise
// gtfill: lhsres(i,j) = 1 if lhsres(i,j) >  rhs(i,j), 0 otherwise
// lefill: lhsres(i,j) = 1 if lhsres(i,j) <= rhs(i,j), 0 otherwise
// gefill: lhsres(i,j) = 1 if lhsres(i,j) >= rhs(i,j), 0 otherwise

template <class T> Matrix<T> &randrfill(Matrix<T> &res);
template <class T> Matrix<T> &randbfill(Matrix<T> &res);
template <class T> Matrix<T> &randBfill(Matrix<T> &res);
template <class T> Matrix<T> &randgfill(Matrix<T> &res);
template <class T> Matrix<T> &randpfill(Matrix<T> &res);
template <class T> Matrix<T> &randufill(Matrix<T> &res);
template <class T> Matrix<T> &randefill(Matrix<T> &res);
template <class T> Matrix<T> &randGfill(Matrix<T> &res);
template <class T> Matrix<T> &randwfill(Matrix<T> &res);
template <class T> Matrix<T> &randxfill(Matrix<T> &res);
template <class T> Matrix<T> &randnfill(Matrix<T> &res);
template <class T> Matrix<T> &randlfill(Matrix<T> &res);
template <class T> Matrix<T> &randcfill(Matrix<T> &res);
template <class T> Matrix<T> &randCfill(Matrix<T> &res);
template <class T> Matrix<T> &randffill(Matrix<T> &res);
template <class T> Matrix<T> &randtfill(Matrix<T> &res);

inline Matrix<double> &ltfill(Matrix<double> &lhsres, const Matrix<double> &rhs);
inline Matrix<double> &gtfill(Matrix<double> &lhsres, const Matrix<double> &rhs);
inline Matrix<double> &lefill(Matrix<double> &lhsres, const Matrix<double> &rhs);
inline Matrix<double> &gefill(Matrix<double> &lhsres, const Matrix<double> &rhs);



// Mathematical operator overloading
//
// NB: in general it is wise to avoid use of non-assignment operators (ie.
//     those which do not return a reference) as there may be a
//     computational hit when constructors (and possibly copy constructors)
//     are called.
//
// + posation - unary, return rvalue
// - negation - unary, return rvalue

template <class T> Matrix<T>  operator+(const Matrix<T> &left_op);
template <class T> Matrix<T>  operator-(const Matrix<T> &left_op);

// + addition    - binary, return rvalue
// - subtraction - binary, return rvalue
//
// NB: adding a scalar to a matrix adds the scalar to all elements of the
//     matrix.

template <class T> Matrix<T>  operator+ (const Matrix<T> &left_op, const Matrix<T> &right_op);
template <class T> Matrix<T>  operator+ (const Matrix<T> &left_op, const T         &right_op);
template <class T> Matrix<T>  operator+ (const T         &left_op, const Matrix<T> &right_op);
template <class T> Matrix<T>  operator- (const Matrix<T> &left_op, const Matrix<T> &right_op);
template <class T> Matrix<T>  operator- (const Matrix<T> &left_op, const T         &right_op);
template <class T> Matrix<T>  operator- (const T         &left_op, const Matrix<T> &right_op);

// += additive    assignment - binary, return lvalue
// -= subtractive assignment - binary, return lvalue

template <class T> Matrix<T> &operator+=(      Matrix<T> &left_op, const Matrix<T> &right_op);
template <class T> Matrix<T> &operator+=(      Matrix<T> &left_op, const T         &right_op);
template <class T> Matrix<T> &operator-=(      Matrix<T> &left_op, const Matrix<T> &right_op);
template <class T> Matrix<T> &operator-=(      Matrix<T> &left_op, const T         &right_op);

// * multiplication - binary, return rvalue
//
// NB: if A,B are matrices, b a vector and c a scalar
//     A*B = A.B  is standard matrix-matrix multiplication
//     A*b = A.b  is standard matrix-vector multiplication
//     A*c = A.c  is standard matrix-scalar multiplication
//     b*A = b'.A is standard matrix-vector multiplication, where b' is the conjugate transpose
//     c*A = c.A  is standard matrix-scalar multiplication

template <         class T> Matrix<T>  operator* (const Matrix<T> &left_op, const Matrix<T> &right_op);
template <class S, class T> Vector<S>  operator* (const Vector<S> &left_op, const Matrix<T> &right_op);
template <         class T> Matrix<T>  operator* (const Matrix<T> &left_op, const T         &right_op);
template <class S, class T> Vector<S>  operator* (const Matrix<T> &left_op, const Vector<S> &right_op);
template <         class T> Matrix<T>  operator* (const T         &left_op, const Matrix<T> &right_op);

// *= multiplicative assignment - binary, return lvalue
//
// NB: if A,B are matrices, b a vector and c a scalar
//     A *= B sets A  := A*B  and returns a reference to A
//     A *= c sets A  := A*c  and returns a reference to A
//     b *= A sets b' := b'*A and returns a reference to b
//
// leftmult:  sets A = A*B and returns a reference to A
// rightmult: sets B = A*B and returns a reference to B
//
// mult: A = B*C

template <         class T> Matrix<T> &operator*=(      Matrix<T> &left_op, const Matrix<T> &right_op);
template <class S, class T> Vector<S> &operator*=(      Vector<S> &left_op, const Matrix<T> &right_op);
template <         class T> Matrix<T> &operator*=(      Matrix<T> &left_op, const T         &right_op);

template <         class T> Matrix<T> &leftmult (      Matrix<T> &left_op, const Matrix<T> &right_op);
template <class S, class T> Vector<S> &leftmult (      Vector<S> &left_op, const Matrix<T> &right_op);
template <         class T> Matrix<T> &leftmult (      Matrix<T> &left_op, const T         &right_op);
template <         class T> Matrix<T> &rightmult(const Matrix<T> &left_op,       Matrix<T> &right_op);
template <class S, class T> Vector<S> &rightmult(const Matrix<T> &left_op,       Vector<S> &right_op);
template <         class T> Matrix<T> &rightmult(const T         &left_op,       Matrix<T> &right_op);

template <class T> Matrix<T> &mult(Matrix<T> &A, const Matrix<T> &B, const Matrix<T> &C);
template <class T> Vector<T> &mult(Vector<T> &a, const Vector<T> &b, const Matrix<T> &C);
template <class T> Vector<T> &mult(Vector<T> &a, const Matrix<T> &B, const Vector<T> &c);

template <class T> Vector<T> &multtrans(Vector<T> &a, const Vector<T> &b, const Matrix<T> &C);
template <class T> Vector<T> &multtrans(Vector<T> &a, const Matrix<T> &B, const Vector<T> &c);

// Relational operator overloading

template <class T> int operator==(const Matrix<T> &left_op, const Matrix<T> &right_op);
template <class T> int operator==(const Matrix<T> &left_op, const T         &right_op);
template <class T> int operator==(const T         &left_op, const Matrix<T> &right_op);

template <class T> int operator!=(const Matrix<T> &left_op, const Matrix<T> &right_op);
template <class T> int operator!=(const Matrix<T> &left_op, const T         &right_op);
template <class T> int operator!=(const T         &left_op, const Matrix<T> &right_op);

template <class T> int operator< (const Matrix<T> &left_op, const T &right_op);
template <class T> int operator< (const Matrix<T> &left_op, const T         &right_op);
template <class T> int operator< (const T         &left_op, const Matrix<T> &right_op);

template <class T> int operator<=(const Matrix<T> &left_op, const T &right_op);
template <class T> int operator<=(const Matrix<T> &left_op, const T         &right_op);
template <class T> int operator<=(const T         &left_op, const Matrix<T> &right_op);

template <class T> int operator> (const Matrix<T> &left_op, const T &right_op);
template <class T> int operator> (const Matrix<T> &left_op, const T         &right_op);
template <class T> int operator> (const T         &left_op, const Matrix<T> &right_op);

template <class T> int operator>=(const Matrix<T> &left_op, const T &right_op);
template <class T> int operator>=(const Matrix<T> &left_op, const T         &right_op);
template <class T> int operator>=(const T         &left_op, const Matrix<T> &right_op);


// Stream operators

template <class T> std::ostream &operator<<(std::ostream &output, const Matrix<T> &src );
template <class T> std::istream &operator>>(std::istream &input,        Matrix<T> &dest);

template <class T> inline std::istream &streamItIn(std::istream &input, Matrix<T>& dest, int processxyzvw = 1);
template <class T> inline std::ostream &streamItOut(std::ostream &output, const Matrix<T>& src, int retainTypeMarker = 0);






/*
      subroutine tql1(n,d,e,ierr)
c
      integer i,j,l,m,n,ii,l1,l2,mml,ierr
      double precision d(n),e(n)
      double precision c,c2,c3,dl1,el1,f,g,h,p,r,s,s2,tst1,tst2,pythag
c
c     this subroutine is a translation of the algol procedure tql1,
c     num. math. 11, 293-306(1968) by bowdler, martin, reinsch, and
c     wilkinson.
c     handbook for auto. comp., vol.ii-linear algebra, 227-240(1971).
c
c     this subroutine finds the eigenvalues of a symmetric
c     tridiagonal matrix by the ql method.
c
c     on input
c
c        n is the order of the matrix.
c
c        d contains the diagonal elements of the input matrix.
c
c        e contains the subdiagonal elements of the input matrix
c          in its last n-1 positions.  e(1) is arbitrary.
c
c      on output
c
c        d contains the eigenvalues in ascending order.  if an
c          error exit is made, the eigenvalues are correct and
c          ordered for indices 1,2,...ierr-1, but may not be
c          the smallest eigenvalues.
c
c        e has been destroyed.
c
c        ierr is set to
c          zero       for normal return,
c          j          if the j-th eigenvalue has not been
c                     determined after 30 iterations.
c
c     calls pythag for  dsqrt(a*a + b*b) .
c
c     questions and comments should be directed to burton s. garbow,
c     mathematics and computer science div, argonne national laboratory
c
c     this version dated august 1983.
c
c     ------------------------------------------------------------------
c
*/

template <class T>
int tql1(Vector<T> &d, Vector<T> &e);

/*
      subroutine tql2(nm,n,d,e,z,ierr)
c
      integer i,j,k,l,m,n,ii,l1,l2,nm,mml,ierr
      double precision d(n),e(n),z(nm,n)
      double precision c,c2,c3,dl1,el1,f,g,h,p,r,s,s2,tst1,tst2,pythag
c
c     this subroutine is a translation of the algol procedure tql2,
c     num. math. 11, 293-306(1968) by bowdler, martin, reinsch, and
c     wilkinson.
c     handbook for auto. comp., vol.ii-linear algebra, 227-240(1971).
c
c     this subroutine finds the eigenvalues and eigenvectors
c     of a symmetric tridiagonal matrix by the ql method.
c     the eigenvectors of a full symmetric matrix can also
c     be found if  tred2  has been used to reduce this
c     full matrix to tridiagonal form.
c
c     on input
c
c        nm must be set to the row dimension of two-dimensional
c          array parameters as declared in the calling program
c          dimension statement.
c
c        n is the order of the matrix.
c
c        d contains the diagonal elements of the input matrix.
c
c        e contains the subdiagonal elements of the input matrix
c          in its last n-1 positions.  e(1) is arbitrary.
c
c        z contains the transformation matrix produced in the
c          reduction by  tred2, if performed.  if the eigenvectors
c          of the tridiagonal matrix are desired, z must contain
c          the identity matrix.
c
c      on output
c
c        d contains the eigenvalues in ascending order.  if an
c          error exit is made, the eigenvalues are correct but
c          unordered for indices 1,2,...,ierr-1.
c
c        e has been destroyed.
c
c        z contains orthonormal eigenvectors of the symmetric
c          tridiagonal (or full) matrix.  if an error exit is made,
c          z contains the eigenvectors associated with the stored
c          eigenvalues.
c
c        ierr is set to
c          zero       for normal return,
c          j          if the j-th eigenvalue has not been
c                     determined after 30 iterations.
c
c     calls pythag for  dsqrt(a*a + b*b) .
c
c     questions and comments should be directed to burton s. garbow,
c     mathematics and computer science div, argonne national laboratory
c
c     this version dated august 1983.
c
c     ------------------------------------------------------------------
c
*/

//int tql2(int nm, int n, Vector<double> &d, Vector<double> &e, Matrix<double> &z);
template <class T>
inline int tql2(Vector<T> &d, Vector<T> &e, Matrix<T> &z);







































template <class T>
void retMatrix<T>::reset(Matrix<T> &cover)
{
    if ( !Matrix<T>::pbaseRow && Matrix<T>::pivotRow )
    {
        MEMDEL(Matrix<T>::pivotRow); Matrix<T>::pivotRow = nullptr;
    }

    Matrix<T>::pbaseRow = true;
    Matrix<T>::pivotRow = nullptr;

    if ( !Matrix<T>::pbaseCol && Matrix<T>::pivotCol )
    {
        MEMDEL(Matrix<T>::pivotCol); Matrix<T>::pivotCol = nullptr;
    }

    Matrix<T>::pbaseCol   = true;
    Matrix<T>::pivotCol   = nullptr;
    Matrix<T>::bkref      = cover.bkref;
    Matrix<T>::nbase      = true;
    Matrix<T>::content    = cover.content;
    Matrix<T>::ccontent   = cover.ccontent;
    Matrix<T>::iscover    = cover.iscover;
    Matrix<T>::elmfn      = cover.elmfn;
    Matrix<T>::rowfn      = cover.rowfn;
    Matrix<T>::delfn      = cover.delfn;
    Matrix<T>::dref       = cover.dref;
    Matrix<T>::celmfn     = cover.celmfn;
    Matrix<T>::celmfn_v   = cover.celmfn_v;
    Matrix<T>::crowfn     = cover.crowfn;
    Matrix<T>::cdelfn     = cover.cdelfn;
    Matrix<T>::cdref      = cover.cdref;
    Matrix<T>::Lweight    = cover.Lweight;
    Matrix<T>::useLweight = cover.useLweight;

    return;
}

template <class T>
DynArray<int> *retMatrix<T>::reset_r(Matrix<T> &cover, int rowpivotsize)
{
    DynArray<int> *resval = nullptr;

    if ( !Matrix<T>::pbaseRow && Matrix<T>::pivotRow )
    {
        resval = &((const_cast<DynArray<int> &>(*Matrix<T>::pivotRow).resize(rowpivotsize)));
    }

    else
    {
	MEMNEW(resval,DynArray<int>);
        (*resval) = { nullptr,0,0,0,false,false,false };
        (*resval).resize(rowpivotsize);
        (*resval).useSlackAllocation();

        Matrix<T>::pbaseRow = false;
        Matrix<T>::pivotRow = resval;
    }

    if ( !Matrix<T>::pbaseCol && Matrix<T>::pivotCol )
    {
        MEMDEL(Matrix<T>::pivotCol); Matrix<T>::pivotCol = nullptr;
    }

    Matrix<T>::pbaseCol   = true;
    Matrix<T>::pivotCol   = nullptr;
    Matrix<T>::bkref      = cover.bkref;
    Matrix<T>::nbase      = true;
    Matrix<T>::content    = cover.content;
    Matrix<T>::ccontent   = cover.ccontent;
    Matrix<T>::iscover    = cover.iscover;
    Matrix<T>::elmfn      = cover.elmfn;
    Matrix<T>::rowfn      = cover.rowfn;
    Matrix<T>::delfn      = cover.delfn;
    Matrix<T>::dref       = cover.dref;
    Matrix<T>::celmfn     = cover.celmfn;
    Matrix<T>::celmfn_v   = cover.celmfn_v;
    Matrix<T>::crowfn     = cover.crowfn;
    Matrix<T>::cdelfn     = cover.cdelfn;
    Matrix<T>::cdref      = cover.cdref;
    Matrix<T>::Lweight    = cover.Lweight;
    Matrix<T>::useLweight = cover.useLweight;

    return resval;
}

template <class T>
DynArray<int> *retMatrix<T>::reset_c(Matrix<T> &cover, int colpivotsize)
{
    if ( !Matrix<T>::pbaseRow && Matrix<T>::pivotRow )
    {
        MEMDEL(Matrix<T>::pivotRow); Matrix<T>::pivotRow = nullptr;
    }

    Matrix<T>::pbaseRow = true;
    Matrix<T>::pivotRow = nullptr;

    DynArray<int> *resval = nullptr;

    if ( !Matrix<T>::pbaseCol && Matrix<T>::pivotCol )
    {
        resval = &((const_cast<DynArray<int> &>(*Matrix<T>::pivotCol).resize(colpivotsize)));
    }

    else
    {
	MEMNEW(resval,DynArray<int>);
        (*resval) = { nullptr,0,0,0,false,false,false };
        (*resval).resize(colpivotsize);
        (*resval).useSlackAllocation();

        Matrix<T>::pbaseCol = false;
        Matrix<T>::pivotCol = resval;
    }

    Matrix<T>::bkref      = cover.bkref;
    Matrix<T>::nbase      = true;
    Matrix<T>::content    = cover.content;
    Matrix<T>::ccontent   = cover.ccontent;
    Matrix<T>::iscover    = cover.iscover;
    Matrix<T>::elmfn      = cover.elmfn;
    Matrix<T>::rowfn      = cover.rowfn;
    Matrix<T>::delfn      = cover.delfn;
    Matrix<T>::dref       = cover.dref;
    Matrix<T>::celmfn     = cover.celmfn;
    Matrix<T>::celmfn_v   = cover.celmfn_v;
    Matrix<T>::crowfn     = cover.crowfn;
    Matrix<T>::cdelfn     = cover.cdelfn;
    Matrix<T>::cdref      = cover.cdref;
    Matrix<T>::Lweight    = cover.Lweight;
    Matrix<T>::useLweight = cover.useLweight;

    return resval;
}

template <class T>
DynArray<int> *retMatrix<T>::reset_rc(Matrix<T> &cover, DynArray<int> *&tpivCol, int rowpivotsize, int colpivotsize)
{
    DynArray<int> *tpivRow = nullptr;

    if ( !Matrix<T>::pbaseRow && Matrix<T>::pivotRow )
    {
        tpivRow = &((const_cast<DynArray<int> &>(*Matrix<T>::pivotRow).resize(rowpivotsize)));
    }

    else
    {
	MEMNEW(tpivRow,DynArray<int>);
        (*tpivRow) = { nullptr,0,0,0,false,false,false };
        (*tpivRow).resize(rowpivotsize);
        (*tpivRow).useSlackAllocation();

        Matrix<T>::pbaseRow = false;
        Matrix<T>::pivotRow = tpivRow;
    }

    if ( !Matrix<T>::pbaseCol && Matrix<T>::pivotCol )
    {
        tpivCol = &((const_cast<DynArray<int> &>(*Matrix<T>::pivotCol).resize(colpivotsize)));
    }

    else
    {
	MEMNEW(tpivCol,DynArray<int>);
        (*tpivCol) = { nullptr,0,0,0,false,false,false };
        (*tpivCol).resize(colpivotsize);
        (*tpivCol).useSlackAllocation();

        Matrix<T>::pbaseCol = false;
        Matrix<T>::pivotCol = tpivCol;
    }

    Matrix<T>::bkref      = cover.bkref;
    Matrix<T>::nbase      = true;
    Matrix<T>::content    = cover.content;
    Matrix<T>::ccontent   = cover.ccontent;
    Matrix<T>::iscover    = cover.iscover;
    Matrix<T>::elmfn      = cover.elmfn;
    Matrix<T>::rowfn      = cover.rowfn;
    Matrix<T>::delfn      = cover.delfn;
    Matrix<T>::dref       = cover.dref;
    Matrix<T>::celmfn     = cover.celmfn;
    Matrix<T>::celmfn_v   = cover.celmfn_v;
    Matrix<T>::crowfn     = cover.crowfn;
    Matrix<T>::cdelfn     = cover.cdelfn;
    Matrix<T>::cdref      = cover.cdref;
    Matrix<T>::Lweight    = cover.Lweight;
    Matrix<T>::useLweight = cover.useLweight;

    return tpivRow;
}


template <class T>
void retMatrix<T>::creset(const Matrix<T> &cover)
{
    if ( !Matrix<T>::pbaseRow && Matrix<T>::pivotRow )
    {
        MEMDEL(Matrix<T>::pivotRow); Matrix<T>::pivotRow = nullptr;
    }

    Matrix<T>::pbaseRow = true;
    Matrix<T>::pivotRow = nullptr;

    if ( !Matrix<T>::pbaseCol && Matrix<T>::pivotCol )
    {
        MEMDEL(Matrix<T>::pivotCol); Matrix<T>::pivotCol = nullptr;
    }

    Matrix<T>::pbaseCol = true;
    Matrix<T>::pivotCol = nullptr;

    Matrix<T>::bkref      = cover.bkref;
    Matrix<T>::nbase      = true;
    Matrix<T>::content    = nullptr;
    Matrix<T>::ccontent   = cover.ccontent;
    Matrix<T>::iscover    = cover.iscover;
    Matrix<T>::elmfn      = nullptr;
    Matrix<T>::rowfn      = nullptr;
    Matrix<T>::delfn      = nullptr;
    Matrix<T>::dref       = nullptr;
    Matrix<T>::celmfn     = cover.celmfn;
    Matrix<T>::celmfn_v   = cover.celmfn_v;
    Matrix<T>::crowfn     = cover.crowfn;
    Matrix<T>::cdelfn     = cover.cdelfn;
    Matrix<T>::cdref      = cover.cdref;
    Matrix<T>::Lweight    = cover.Lweight;
    Matrix<T>::useLweight = cover.useLweight;

    return;
}

template <class T>
DynArray<int> *retMatrix<T>::creset_r(const Matrix<T> &cover, int rowpivotsize)
{
    DynArray<int> *resval = nullptr;

    if ( !Matrix<T>::pbaseRow && Matrix<T>::pivotRow )
    {
        resval = &((const_cast<DynArray<int> &>(*Matrix<T>::pivotRow).resize(rowpivotsize)));
    }

    else
    {
	MEMNEW(resval,DynArray<int>);
        (*resval) = { nullptr,0,0,0,false,false,false };
        (*resval).resize(rowpivotsize);
        (*resval).useSlackAllocation();

        Matrix<T>::pbaseRow = false;
        Matrix<T>::pivotRow = resval;
    }

    if ( !Matrix<T>::pbaseCol && Matrix<T>::pivotCol )
    {
        MEMDEL(Matrix<T>::pivotCol); Matrix<T>::pivotCol = nullptr;
    }

    Matrix<T>::pbaseCol   = true;
    Matrix<T>::pivotCol   = nullptr;
    Matrix<T>::bkref      = cover.bkref;
    Matrix<T>::nbase      = true;
    Matrix<T>::content    = nullptr;
    Matrix<T>::ccontent   = cover.ccontent;
    Matrix<T>::iscover    = cover.iscover;
    Matrix<T>::elmfn      = nullptr;
    Matrix<T>::rowfn      = nullptr;
    Matrix<T>::delfn      = nullptr;
    Matrix<T>::dref       = nullptr;
    Matrix<T>::celmfn     = cover.celmfn;
    Matrix<T>::celmfn_v   = cover.celmfn_v;
    Matrix<T>::crowfn     = cover.crowfn;
    Matrix<T>::cdelfn     = cover.cdelfn;
    Matrix<T>::cdref      = cover.cdref;
    Matrix<T>::Lweight    = cover.Lweight;
    Matrix<T>::useLweight = cover.useLweight;

    return resval;
}

template <class T>
DynArray<int> *retMatrix<T>::creset_c(const Matrix<T> &cover, int colpivotsize)
{
    if ( !Matrix<T>::pbaseRow && Matrix<T>::pivotRow )
    {
        MEMDEL(Matrix<T>::pivotRow); Matrix<T>::pivotRow = nullptr;
    }

    Matrix<T>::pbaseRow = true;
    Matrix<T>::pivotRow = nullptr;

    DynArray<int> *resval = nullptr;

    if ( !Matrix<T>::pbaseCol && Matrix<T>::pivotCol )
    {
        resval = &((const_cast<DynArray<int> &>(*Matrix<T>::pivotCol).resize(colpivotsize)));
    }

    else
    {
	MEMNEW(resval,DynArray<int>);
        (*resval) = { nullptr,0,0,0,false,false,false };
        (*resval).resize(colpivotsize);
        (*resval).useSlackAllocation();

        Matrix<T>::pbaseCol = false;
        Matrix<T>::pivotCol = resval;
    }

    Matrix<T>::bkref      = cover.bkref;
    Matrix<T>::nbase      = true;
    Matrix<T>::content    = nullptr;
    Matrix<T>::ccontent   = cover.ccontent;
    Matrix<T>::iscover    = cover.iscover;
    Matrix<T>::elmfn      = nullptr;
    Matrix<T>::rowfn      = nullptr;
    Matrix<T>::delfn      = nullptr;
    Matrix<T>::dref       = nullptr;
    Matrix<T>::celmfn     = cover.celmfn;
    Matrix<T>::celmfn_v   = cover.celmfn_v;
    Matrix<T>::crowfn     = cover.crowfn;
    Matrix<T>::cdelfn     = cover.cdelfn;
    Matrix<T>::cdref      = cover.cdref;
    Matrix<T>::Lweight    = cover.Lweight;
    Matrix<T>::useLweight = cover.useLweight;

    return resval;
}

template <class T>
DynArray<int> *retMatrix<T>::creset_rc(const Matrix<T> &cover, DynArray<int> *&tpivCol, int rowpivotsize, int colpivotsize)
{
    DynArray<int> *tpivRow = nullptr;

    if ( !Matrix<T>::pbaseRow && Matrix<T>::pivotRow )
    {
        tpivRow = &((const_cast<DynArray<int> &>(*Matrix<T>::pivotRow).resize(rowpivotsize)));
    }

    else
    {
	MEMNEW(tpivRow,DynArray<int>);
        (*tpivRow) = { nullptr,0,0,0,false,false,false };
        (*tpivRow).resize(rowpivotsize);
        (*tpivRow).useSlackAllocation();

        Matrix<T>::pbaseRow = false;
        Matrix<T>::pivotRow = tpivRow;
    }

    if ( !Matrix<T>::pbaseCol && Matrix<T>::pivotCol )
    {
        tpivCol = &((const_cast<DynArray<int> &>(*Matrix<T>::pivotCol).resize(colpivotsize)));
    }

    else
    {
	MEMNEW(tpivCol,DynArray<int>);
        (*tpivCol) = { nullptr,0,0,0,false,false,false };
        (*tpivCol).resize(colpivotsize);
        (*tpivCol).useSlackAllocation();

        Matrix<T>::pbaseCol = false;
        Matrix<T>::pivotCol = tpivCol;
    }

    Matrix<T>::bkref      = cover.bkref;
    Matrix<T>::nbase      = true;
    Matrix<T>::content    = nullptr;
    Matrix<T>::ccontent   = cover.ccontent;
    Matrix<T>::iscover    = cover.iscover;
    Matrix<T>::elmfn      = nullptr;
    Matrix<T>::rowfn      = nullptr;
    Matrix<T>::delfn      = nullptr;
    Matrix<T>::dref       = nullptr;
    Matrix<T>::celmfn     = cover.celmfn;
    Matrix<T>::celmfn_v   = cover.celmfn_v;
    Matrix<T>::crowfn     = cover.crowfn;
    Matrix<T>::cdelfn     = cover.cdelfn;
    Matrix<T>::cdref      = cover.cdref;
    Matrix<T>::Lweight    = cover.Lweight;
    Matrix<T>::useLweight = cover.useLweight;

    return tpivRow;
}


// Constructors and Destructors

template <class T>
Matrix<T>::Matrix() : dnumRows(0),
                      dnumCols(0),
                      iibRow(0),
                      iisRow(1),
                      iibCol(0),
                      iisCol(1),
                      nbase(false),
                      pbaseRow(true),
                      pbaseCol(true),
                      iscover(false),
                      bkref(this),
                      content(nullptr),
                      ccontent(nullptr),
                      pivotRow(cntintarray(0)),
                      pivotCol(cntintarray(0)),
                      elmfn(nullptr),
                      rowfn(nullptr),
                      delfn(nullptr),
                      dref(nullptr),
                      celmfn(nullptr),
                      celmfn_v(nullptr),
                      crowfn(nullptr),
                      cdelfn(nullptr),
                      cdref(nullptr),
                      Lweight(nullptr),
                      useLweight(false)
{
    MEMNEW(content,Vector<Vector<T> >(dnumRows));
    ccontent = content;

    NiceAssert( content );
}

template <class T>
Matrix<T>::Matrix(const T &) : dnumRows(0),
                      dnumCols(0),
                      iibRow(0),
                      iisRow(1),
                      iibCol(0),
                      iisCol(1),
                      nbase(false),
                      pbaseRow(true),
                      pbaseCol(true),
                      iscover(false),
                      bkref(this),
                      content(nullptr),
                      ccontent(nullptr),
                      pivotRow(cntintarray(0)),
                      pivotCol(cntintarray(0)),
                      elmfn(nullptr),
                      rowfn(nullptr),
                      delfn(nullptr),
                      dref(nullptr),
                      celmfn(nullptr),
                      celmfn_v(nullptr),
                      crowfn(nullptr),
                      cdelfn(nullptr),
                      cdref(nullptr),
                      Lweight(nullptr),
                      useLweight(false)
{
    MEMNEW(content,Vector<Vector<T> >(dnumRows));
    ccontent = content;

    NiceAssert( content );
}

template <class T>
Matrix<T>::Matrix(int numRows, int numCols) : dnumRows(numRows),
                                              dnumCols(numCols),
                                              iibRow(0),
                                              iisRow(1),
                                              iibCol(0),
                                              iisCol(1),
                                              nbase(false),
                                              pbaseRow(true),
                                              pbaseCol(true),
                                              iscover(false),
                                              bkref(this),
                                              content(nullptr),
                                              ccontent(nullptr),
                                              pivotRow(cntintarray(numRows)),
                                              pivotCol(cntintarray(numCols)),
                                              elmfn(nullptr),
                                              rowfn(nullptr),
                                              delfn(nullptr),
                                              dref(nullptr),
                                              celmfn(nullptr),
                                              celmfn_v(nullptr),
                                              crowfn(nullptr),
                                              cdelfn(nullptr),
                                              cdref(nullptr),
                                              Lweight(nullptr),
                                              useLweight(false)
{
    int i;

    NiceAssert( numRows >= 0 );
    NiceAssert( numCols >= 0 );

    MEMNEW(content,Vector<Vector<T> >(dnumRows));
    ccontent = content;

    NiceAssert( content );

    for ( i = 0 ; i < dnumRows ; ++i )
    {
        (*content)("&",i).resize(dnumCols);
    }
}

template <class T>
Matrix<T>::Matrix(const T &(*celm)(int, int, const void *, retVector<T> &), const Vector<T> &(*crow)(int, const void *, retVector<T> &), const void *cxdref, int numRows, int numCols, void (*xcdelfn)(const void *, void *), void *xdref, const Vector<T> *_Lweight, bool _useLweight)
                      : dnumRows(numRows),
                        dnumCols(numCols),
                        iibRow(0),
                        iisRow(1),
                        iibCol(0),
                        iisCol(1),
                        nbase(false),
                        pbaseRow(true),
                        pbaseCol(true),
                        iscover(true),
                        bkref(this),
                        content(nullptr),
                        ccontent(nullptr),
                        pivotRow(cntintarray(numRows)),
                        pivotCol(cntintarray(numCols)),
                        elmfn(nullptr),
                        rowfn(nullptr),
                        delfn(nullptr),
                        dref(xdref),
                        celmfn(celm),
                        celmfn_v(nullptr),
                        crowfn(crow),
                        cdelfn(xcdelfn),
                        cdref(cxdref),
                        Lweight(_Lweight),
                        useLweight(_useLweight)
{
    NiceAssert( numRows >= 0 );
    NiceAssert( numCols >= 0 );
}

template <class T>
Matrix<T>::Matrix(T (*celm_v)(int, int, const void *, retVector<T> &), const T &(*celm)(int, int, const void *, retVector<T> &), const Vector<T> &(*crow)(int, const void *, retVector<T> &), const void *cxdref, int numRows, int numCols, void (*xcdelfn)(const void *, void *), void *xdref, const Vector<T> *_Lweight, bool _useLweight)
                      : dnumRows(numRows),
                        dnumCols(numCols),
                        iibRow(0),
                        iisRow(1),
                        iibCol(0),
                        iisCol(1),
                        nbase(false),
                        pbaseRow(true),
                        pbaseCol(true),
                        iscover(true),
                        bkref(this),
                        content(nullptr),
                        ccontent(nullptr),
                        pivotRow(cntintarray(numRows)),
                        pivotCol(cntintarray(numCols)),
                        elmfn(nullptr),
                        rowfn(nullptr),
                        delfn(nullptr),
                        dref(xdref),
                        celmfn(celm),
                        celmfn_v(celm_v),
                        crowfn(crow),
                        cdelfn(xcdelfn),
                        cdref(cxdref),
                        Lweight(_Lweight),
                        useLweight(_useLweight)
{
    NiceAssert( numRows >= 0 );
    NiceAssert( numCols >= 0 );
}

template <class T>
Matrix<T>::Matrix(T &(*elm)(int, int, void *, retVector<T> &), Vector<T> &(*row)(int, void *, retVector<T> &), void *xdref, const T &(*celm)(int, int, const void *, retVector<T> &), const Vector<T> &(*crow)(int, const void *, retVector<T> &), const void *cxdref, int numRows, int numCols, void (*xdelfn)(void *), void (*xcdelfn)(const void *, void *), const Vector<T> *_Lweight, bool _useLweight)
                      : dnumRows(numRows),
                        dnumCols(numCols),
                        iibRow(0),
                        iisRow(1),
                        iibCol(0),
                        iisCol(1),
                        nbase(false),
                        pbaseRow(true),
                        pbaseCol(true),
                        iscover(true),
                        bkref(this),
                        content(nullptr),
                        ccontent(nullptr),
                        pivotRow(cntintarray(numRows)),
                        pivotCol(cntintarray(numCols)),
                        elmfn(elm),
                        rowfn(row),
                        delfn(xdelfn),
                        dref(xdref),
                        celmfn(celm),
                        celmfn_v(nullptr),
                        crowfn(crow),
                        cdelfn(xcdelfn),
                        cdref(cxdref),
                        Lweight(_Lweight),
                        useLweight(_useLweight)
{
    NiceAssert( numRows >= 0 );
    NiceAssert( numCols >= 0 );
}

template <class T>
Matrix<T>::Matrix(const Matrix<T> &src) : dnumRows(0),
                                          dnumCols(0),
                                          iibRow(0),
                                          iisRow(1),
                                          iibCol(0),
                                          iisCol(1),
                                          nbase(false),
                                          pbaseRow(true),
                                          pbaseCol(true),
                                          iscover(false),
                                          bkref(this),
                                          content(nullptr),
                                          ccontent(nullptr),
                                          pivotRow(cntintarray(0)),
                                          pivotCol(cntintarray(0)),
                                          elmfn(nullptr),
                                          rowfn(nullptr),
                                          delfn(nullptr),
                                          dref(nullptr),
                                          celmfn(nullptr),
                                          celmfn_v(nullptr),
                                          crowfn(nullptr),
                                          cdelfn(nullptr),
                                          cdref(nullptr),
                                          Lweight(nullptr),
                                          useLweight(false)
{
    MEMNEW(content,Vector<Vector<T> >);
    ccontent = content;

    NiceAssert( content );

    *this = src;
}

template <class T>
Matrix<T>::Matrix(const char *, const Matrix<T> &src) : dnumRows(0),
                                                             dnumCols(0),
                                                             iibRow(0),
                                                             iisRow(0),
                                                             iibCol(0),
                                                             iisCol(0),
                                                             nbase(false),
                                                             pbaseRow(true),
                                                             pbaseCol(true),
                                                             iscover(false),
                                                             bkref(src.bkref),
                                                             content(nullptr),
                                                             ccontent(nullptr),
                                                             pivotRow(nullptr),
                                                             pivotCol(nullptr),
                                                             elmfn(nullptr),
                                                             rowfn(nullptr),
                                                             delfn(nullptr),
                                                             dref(nullptr),
                                                             celmfn(nullptr),
                                                             celmfn_v(nullptr),
                                                             crowfn(nullptr),
                                                             cdelfn(nullptr),
                                                             cdref(nullptr),
                                                             Lweight(nullptr),
                                                             useLweight(false)
{
    ;
}

template <class T>
Matrix<T>::Matrix(const char *) : dnumRows(0),
                                       dnumCols(0),
                                       iibRow(0),
                                       iisRow(0),
                                       iibCol(0),
                                       iisCol(0),
                                       nbase(false),
                                       pbaseRow(true),
                                       pbaseCol(true),
                                       iscover(false),
                                       bkref(nullptr),
                                       content(nullptr),
                                       ccontent(nullptr),
                                       pivotRow(nullptr),
                                       pivotCol(nullptr),
                                       elmfn(nullptr),
                                       rowfn(nullptr),
                                       delfn(nullptr),
                                       dref(nullptr),
                                       celmfn(nullptr),
                                       celmfn_v(nullptr),
                                       crowfn(nullptr),
                                       cdelfn(nullptr),
                                       cdref(nullptr),
                                       Lweight(nullptr),
                                       useLweight(false)
{
    ;
}

template <class T>
Matrix<T>::Matrix(const char *, const Vector<T> &ccondummy) : dnumRows(INT_MAX-1),
                                       dnumCols(INT_MAX-1),
                                       iibRow(0),
                                       iisRow(0),
                                       iibCol(0),
                                       iisCol(0),
                                       nbase(true),
                                       pbaseRow(true),
                                       pbaseCol(true),
                                       iscover(false),
                                       bkref(nullptr),
                                       content(nullptr),
                                       ccontent(nullptr),
                                       pivotRow(zerointarray()),
                                       pivotCol(zerointarray()),
                                       elmfn(nullptr),
                                       rowfn(nullptr),
                                       delfn(nullptr),
                                       dref(nullptr),
                                       celmfn(nullptr),
                                       celmfn_v(nullptr),
                                       crowfn(nullptr),
                                       cdelfn(nullptr),
                                       cdref(nullptr),
                                       Lweight(nullptr),
                                       useLweight(false)
{
    bkref = this;

    Vector<Vector<T> > *cccontent = nullptr;

    MEMNEW(cccontent,Vector<Vector<T> >(1));
    (*cccontent)("&",0).assignover(ccondummy);
    ccontent = cccontent;
}

template <class T>
Matrix<T>::~Matrix()
{
    if ( !nbase && !iscover && content )
    {
	MEMDEL(content); content = nullptr;
    }

    if ( !nbase && iscover )
    {
	if ( delfn )
	{
	    delfn(dref);
	}

	if ( cdelfn )
	{
	    cdelfn(cdref,dref);
	}
    }

    if ( !pbaseRow && pivotRow )
    {
        MEMDEL(pivotRow); pivotRow = nullptr;
    }

    if ( !pbaseCol && pivotCol )
    {
        MEMDEL(pivotCol); pivotCol = nullptr;
    }
}




// Assignment

template <class T>
Matrix<T> &Matrix<T>::operator=(const Matrix<T> &src)
{
    if ( shareBase(&src) )
    {
	Matrix<T> temp(src);

	*this = temp;
    }

    else
    {
	int srcnumRows = src.numRows();
	int srcnumCols = src.numCols();
	int i;

	// Fix size if this is not a reference

	if ( !nbase )
	{
	    resize(srcnumRows,srcnumCols);
	}

        NiceAssert( dnumRows == srcnumRows );
        NiceAssert( dnumCols == srcnumCols );

        retVector<T> tmpva;
        retVector<T> tmpvb;
        retVector<T> tmpvc;
        retVector<T> tmpvd;

	if ( dnumRows && dnumCols )
	{
	    for ( i = 0 ; i < dnumRows ; ++i )
	    {
		(*this)("&",i,"&",tmpva,tmpvc) = src(i,tmpvb,tmpvd);
	    }
	}
    }

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::operator=(const Vector<T> &src)
{
    if ( ( dnumRows == 1 ) && ( dnumCols == src.size() ) )
    {
        for ( int i = 0 ; i < src.size() ; ++i )
        {
            (*this)("&",0,i) = src(i);
	}
    }

    else if ( ( dnumRows == src.size() ) && ( dnumCols == 1 ) )
    {
        for ( int i = 0 ; i < src.size() ; ++i )
        {
            (*this)("&",i,0) = src(i);
	}
    }

    else
    {
        NiceAssert( !nbase );

	resize(src.size(),1);

        *this = src;
    }

    return *this;
}

template <class T>
Vector<T> &Vector<T>::operator=(const Matrix<T> &src)
{
    NiceAssert( !infsize() );
    NiceAssert( ( src.numRows() == 1 ) || ( src.numCols() == 1 ) );

    if ( src.numRows() == 1 )
    {
	// Row vector

	int srcsize = src.numCols();

	if ( !nbase )
	{
	    resize(srcsize);
	}

        NiceAssert( dsize == srcsize );

        for ( int i = 0 ; i < dsize ; ++i )
        {
            (*this)("&",i) = src(0,i);
	}
    }

    else
    {
	// Column vector

	int srcsize = src.numRows();

	if ( !nbase )
	{
	    resize(srcsize);
	}

        NiceAssert( dsize == srcsize );

        for ( int i = 0 ; i < dsize ; ++i )
        {
            (*this)("&",i) = src(i,0);
	}
    }

    return *this;
}

template <class T>
SparseVector<T> &SparseVector<T>::operator=(const Matrix<T> &src)
{
    NiceAssert( ( src.numRows() == 1 ) || ( src.numCols() == 1 ) );

    zero();

    if ( src.numRows() == 1 )
    {
	// Row vector

	int i;

        retVector<T> tmpva;

        for ( i = 0 ; i < size() ; ++i )
        {
            (*this)("&",i,tmpva) = src(0,i);
	}
    }

    else
    {
	// Column vector

	int i;

        retVector<T> tmpva;

        for ( i = 0 ; i < size() ; ++i )
        {
            (*this)("&",i,tmpva) = src(i,0);
	}
    }

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::operator=(const T &src)
{
    int i,j;

    // Copy over

    if ( dnumCols )
    {
	for ( i = 0 ; i < dnumRows ; ++i )
	{
	    for ( j = 0 ; j < dnumCols ; ++j )
	    {
	        (*this)("&",i,j) = src;
	    }
	}
    }

    return *this;
}


// Access:

template <class T>
T &Matrix<T>::operator()(const char *dummy, int i, int j)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < dnumRows );
    NiceAssert( j >= 0 );
    NiceAssert( j < dnumCols );
    NiceAssert( dummy[0] );
    NiceAssert( pivotRow );
    NiceAssert( pivotCol );

    NiceAssert( !iscover );

    if ( iscover )
    {
        static thread_local retVector<T> tmpva;

        NiceAssert( elmfn );
        NiceAssert( !Lweight || !useLweight );

        return (*elmfn)((*pivotRow).v(iibRow+(i*iisRow)),(*pivotCol).v(iibCol+(j*iisCol)),dref,tmpva);
    }

    NiceAssert( content );

    return ((*content)(dummy,(*pivotRow).v(iibRow+(i*iisRow))))(dummy,(*pivotCol).v(iibCol+(j*iisCol)));
}

template <class T>
const T &Matrix<T>::operator()(int i, int j) const
{
    NiceAssert( i >= 0 );
    NiceAssert( i < dnumRows );
    NiceAssert( j >= 0 );
    NiceAssert( j < dnumCols );
    NiceAssert( pivotRow );
    NiceAssert( pivotCol );

    if ( iscover && ( !Lweight || !useLweight ) )
    {
        static thread_local retVector<T> tmpva;

        NiceAssert( celmfn );

        return (*celmfn)((*pivotRow).v(iibRow+(i*iisRow)),(*pivotCol).v(iibCol+(j*iisCol)),cdref,tmpva);
    }

    else if ( iscover )
    {
        static thread_local retVector<T> tmpva;
        static thread_local T tmpres;

        NiceAssert( celmfn );

        tmpres = ((*Lweight).v((*pivotRow).v(iibRow+(i*iisRow))))*((*Lweight).v((*pivotCol).v(iibCol+(j*iisCol))))*(*celmfn)((*pivotRow).v(iibRow+(i*iisRow)),(*pivotCol).v(iibCol+(j*iisCol)),cdref,tmpva);

        return tmpres;
    }

    NiceAssert( ccontent );

    return ((*ccontent)((*pivotRow).v(iibRow+(i*iisRow))))((*pivotCol).v(iibCol+(j*iisCol)));
}

template <class T>
T Matrix<T>::v(int i, int j) const
{
    NiceAssert( i >= 0 );
    NiceAssert( i < dnumRows );
    NiceAssert( j >= 0 );
    NiceAssert( j < dnumCols );
    NiceAssert( pivotRow );
    NiceAssert( pivotCol );

    if ( iscover && celmfn_v && ( !Lweight || !useLweight ) )
    {
        static thread_local retVector<T> tmpva;

        NiceAssert( celmfn_v );

        return (*celmfn_v)((*pivotRow).v(iibRow+(i*iisRow)),(*pivotCol).v(iibCol+(j*iisCol)),cdref,tmpva);
    }

    else if ( iscover && celmfn_v )
    {
        static thread_local retVector<T> tmpva;

        NiceAssert( celmfn_v );

        return ((*Lweight).v((*pivotRow).v(iibRow+(i*iisRow))))*((*Lweight).v((*pivotCol).v(iibCol+(j*iisCol))))*(*celmfn_v)((*pivotRow).v(iibRow+(i*iisRow)),(*pivotCol).v(iibCol+(j*iisCol)),cdref,tmpva);
    }

    else if ( iscover && ( !Lweight || !useLweight ) )
    {
        static thread_local retVector<T> tmpva;

        NiceAssert( celmfn );

        return (*celmfn)((*pivotRow).v(iibRow+(i*iisRow)),(*pivotCol).v(iibCol+(j*iisCol)),cdref,tmpva);
    }

    else if ( iscover )
    {
        static thread_local retVector<T> tmpva;

        NiceAssert( celmfn );

        return ((*Lweight).v((*pivotRow).v(iibRow+(i*iisRow))))*((*Lweight).v((*pivotCol).v(iibCol+(j*iisCol))))*(*celmfn)((*pivotRow).v(iibRow+(i*iisRow)),(*pivotCol).v(iibCol+(j*iisCol)),cdref,tmpva);
    }

    NiceAssert( ccontent );

    return ((*ccontent)((*pivotRow).v(iibRow+(i*iisRow)))).v((*pivotCol).v(iibCol+(j*iisCol)));
}

template <class T>
void Matrix<T>::sv(int i, int j, T x)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < dnumRows );
    NiceAssert( j >= 0 );
    NiceAssert( j < dnumCols );
    NiceAssert( pivotRow );
    NiceAssert( pivotCol );

    NiceAssert( !iscover );

    if ( iscover )
    {
        retVector<T> tmpva;

        NiceAssert( elmfn );
        NiceAssert( !Lweight || useLweight );

        (*elmfn)((*pivotRow).v(iibRow+(i*iisRow)),(*pivotCol).v(iibCol+(j*iisCol)),dref,tmpva) = x;
    }

    NiceAssert( content );

    return ((*content)("&",(*pivotRow).v(iibRow+(i*iisRow)))).sv((*pivotCol).v(iibCol+(j*iisCol)),x);
}

template <class T>
Vector<T> &Matrix<T>::operator()(const char *dummy, int i, retVector<T> &tmpb, retVector<T> &tmp)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < dnumRows );
    NiceAssert( pivotRow );
    NiceAssert( pivotCol );

    int jnumCols = numCols();

    NiceAssert( !iscover );

    if ( iscover )
    {
        NiceAssert( rowfn );
        NiceAssert( !Lweight || !useLweight );

	if ( !nbase )
	{
	    return ((*rowfn)((*pivotRow).v(iibRow+(i*iisRow)),dref,tmpb));
	}

        int ressize = numCols();

        int iib = iibCol;
        int iis = iisCol;
        int iim = iib+((ressize-1)*iis);

	return ((*rowfn)((*pivotRow).v(iibRow+(i*iisRow)),dref,tmp))(dummy,*pivotCol,iib,iis,iim,tmpb);
    }

    NiceAssert( content );

    if ( !nbase )
    {
	return ((*content)(dummy,(*pivotRow).v(iibRow+(i*iisRow))));
    }

    int ressize = jnumCols;

    int iib = iibCol;
    int iis = iisCol;
    int iim = iib+((ressize-1)*iis);

    return ((*content)(dummy,(*pivotRow).v(iibRow+(i*iisRow))))(dummy,*pivotCol,iib,iis,iim,tmpb);
}

template <class T>
Vector<T> &Matrix<T>::operator()(const char *dummy, int i, int jb, int js, int jm, retVector<T> &res, retVector<T> &tmp)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < dnumRows );
    NiceAssert( ( ( js > 0 ) && ( jb >= 0 ) ) || ( ( js < 0 ) && ( jb < dnumCols ) ) );
    NiceAssert( ( ( js < 0 ) && ( jm >= 0 ) ) || ( ( js > 0 ) && ( jm < dnumCols ) ) );
    NiceAssert( js );
    NiceAssert( pivotRow );
    NiceAssert( pivotCol );

    int jnumCols = ((jm-jb)/js)+1;

    jnumCols = ( jnumCols < 0 ) ? 0 : jnumCols;

    NiceAssert( !iscover );

    if ( iscover )
    {
        NiceAssert( rowfn );
        NiceAssert( !Lweight || !useLweight );

	if ( !nbase )
	{
	    return ((*rowfn)((*pivotRow).v(iibRow+(i*iisRow)),dref,tmp))(dummy,jb,js,jm,res);
	}

        int ressize = ( jb > jm ) ? 0 : ( ((jm-jb)/js)+1 );

        int iib0 = iibCol;
        int iis0 = iisCol;
        //int iim0 = iibCol+((dnumCols-1)*iisCol);

        int iib1 = jb;
        int iis1 = js;
        //int iim1 = jm;

        int iib = iib0+(iis0*iib1);
        int iis = iis0*iis1;
        int iim = iib+((ressize-1)*iis);

	return ((*rowfn)((*pivotRow).v(iibRow+(i*iisRow)),dref,tmp))(dummy,*pivotCol,iib,iis,iim,res);
    }

    NiceAssert( content );

    if ( !nbase )
    {
	return ((*content)(dummy,(*pivotRow).v(iibRow+(i*iisRow))))(dummy,jb,js,jm,res);
    }

    //return (((*ccontent)((*pivotRow)(iibRow+(i*iisRow))))(*pivotCol,iibCol,iisCol,iibCol+((dnumCols-1)*iisCol)))(jb,js,jm,res);

    int ressize = jnumCols;

    int iib0 = iibCol;
    int iis0 = iisCol;
    //int iim0 = iibCol+((dnumCols-1)*iisCol);

    int iib1 = jb;
    int iis1 = js;
    //int iim1 = jm;

    int iib = iib0+(iis0*iib1);
    int iis = iis0*iis1;
    int iim = iib+((ressize-1)*iis);

    return ((*content)(dummy,(*pivotRow).v(iibRow+(i*iisRow))))(dummy,*pivotCol,iib,iis,iim,res);
}

template <class T>
const Vector<T> &Matrix<T>::operator()(int i, retVector<T> &res, retVector<T> &tmp) const
{
    NiceAssert( i >= 0 );
    NiceAssert( i < dnumRows );
    NiceAssert( pivotRow );
    NiceAssert( pivotCol );

    int jnumCols = numCols();

    if ( iscover && ( !Lweight || !useLweight ) )
    {
        NiceAssert( crowfn );

	if ( !nbase )
	{
	    return ((*crowfn)((*pivotRow).v(iibRow+(i*iisRow)),cdref,tmp));
	}

        int ressize = numCols();

        int iib = iibCol;
        int iis = iisCol;
        int iim = iib+((ressize-1)*iis);

	return ((*crowfn)((*pivotRow).v(iibRow+(i*iisRow)),cdref,tmp))(*pivotCol,iib,iis,iim,res);
    }

    else if ( iscover )
    {
        NiceAssert( crowfn );

	if ( !nbase )
	{
            retVector<T> tmpres;

	    ((*crowfn)((*pivotRow).v(iibRow+(i*iisRow)),cdref,tmpres));

            res.reset("&",tmpres); // this is the overwrite form

            static_cast<Vector<T> &>(res).Lscale((*Lweight).v((*pivotRow).v(iibRow+(i*iisRow))));
            static_cast<Vector<T> &>(res).Rscale((*Lweight));

            return res;
	}

        int ressize = numCols();

        int iib = iibCol;
        int iis = iisCol;
        int iim = iib+((ressize-1)*iis);

        retVector<T> tmpres;
        retVector<T> tmpva;

	((*crowfn)((*pivotRow).v(iibRow+(i*iisRow)),cdref,tmp))(*pivotCol,iib,iis,iim,tmpres);

        res.reset("&",tmpres); // this is the overwrite form

        static_cast<Vector<T> &>(res).Lscale((*Lweight).v((*pivotRow).v(iibRow+(i*iisRow))));
        static_cast<Vector<T> &>(res).Rscale((*Lweight)(*pivotCol,iib,iis,iim,tmpva));

        return res;
    }

    NiceAssert( ccontent );

    if ( !nbase )
    {
	return ((*ccontent)((*pivotRow).v(iibRow+(i*iisRow))));
    }

    //return (((*ccontent)((*pivotRow)(iibRow+(i*iisRow))))(*pivotCol,iibCol,iisCol,iibCol+((dnumCols-1)*iisCol)))(jb,js,jm,res);

    int ressize = jnumCols;

    int iib = iibCol;
    int iis = iisCol;
    int iim = iib+((ressize-1)*iis);

    return ((*ccontent)((*pivotRow).v(iibRow+(i*iisRow))))(*pivotCol,iib,iis,iim,res);
}

template <class T>
const Vector<T> &Matrix<T>::operator()(int i, int jb, int js, int jm, retVector<T> &res, retVector<T> &tmp) const
{
    NiceAssert( i >= 0 );
    NiceAssert( i < dnumRows );
    NiceAssert( ( ( js > 0 ) && ( jb >= 0 ) ) || ( ( js < 0 ) && ( jb < dnumCols ) ) );
    NiceAssert( ( ( js < 0 ) && ( jm >= 0 ) ) || ( ( js > 0 ) && ( jm < dnumCols ) ) );
    NiceAssert( js );
    NiceAssert( pivotRow );
    NiceAssert( pivotCol );

    int jnumCols = ((jm-jb)/js)+1;

    jnumCols = ( jnumCols < 0 ) ? 0 : jnumCols;

    if ( iscover && ( !Lweight || !useLweight ) )
    {
        NiceAssert( crowfn );

	if ( !nbase )
	{
	    return ((*crowfn)((*pivotRow).v(iibRow+(i*iisRow)),cdref,tmp))(jb,js,jm,res);
	}

	//return (((*crowfn)((*pivotRow)(iibRow+(i*iisRow)),cdref,tmp))(*pivotCol,iibCol,iisCol,iibCol+((dnumCols-1)*iisCol)))(jb,js,jm,res);

        int ressize = ( jb > jm ) ? 0 : ( ((jm-jb)/js)+1 );

        int iib0 = iibCol;
        int iis0 = iisCol;
        //int iim0 = iibCol+((dnumCols-1)*iisCol);

        int iib1 = jb;
        int iis1 = js;
        //int iim1 = jm;

        int iib = iib0+(iis0*iib1);
        int iis = iis0*iis1;
        int iim = iib+((ressize-1)*iis);

	return ((*crowfn)((*pivotRow).v(iibRow+(i*iisRow)),cdref,tmp))(*pivotCol,iib,iis,iim,res);
    }

    else if ( iscover )
    {
        NiceAssert( crowfn );

	if ( !nbase )
	{
            retVector<T> tmpres;
            retVector<T> tmpva;

	    ((*crowfn)((*pivotRow).v(iibRow+(i*iisRow)),cdref,tmp))(jb,js,jm,tmpres);

            res.reset("&",tmpres); // this is the overwrite form

            static_cast<Vector<T> &>(res).Lscale((*Lweight).v((*pivotRow).v(iibRow+(i*iisRow))));
            static_cast<Vector<T> &>(res).Rscale((*Lweight)(jb,js,jm,tmpva));

            return res;
	}

	//return (((*crowfn)((*pivotRow)(iibRow+(i*iisRow)),cdref,tmp))(*pivotCol,iibCol,iisCol,iibCol+((dnumCols-1)*iisCol)))(jb,js,jm,res);

        int ressize = ( jb > jm ) ? 0 : ( ((jm-jb)/js)+1 );

        int iib0 = iibCol;
        int iis0 = iisCol;
        //int iim0 = iibCol+((dnumCols-1)*iisCol);

        int iib1 = jb;
        int iis1 = js;
        //int iim1 = jm;

        int iib = iib0+(iis0*iib1);
        int iis = iis0*iis1;
        int iim = iib+((ressize-1)*iis);

        retVector<T> tmpres;
        retVector<T> tmpva;

	((*crowfn)((*pivotRow).v(iibRow+(i*iisRow)),cdref,tmp))(*pivotCol,iib,iis,iim,tmpres);

        res.reset("&",tmpres); // this is the overwrite form

        static_cast<Vector<T> &>(res).Lscale((*Lweight).v((*pivotRow).v(iibRow+(i*iisRow))));
        static_cast<Vector<T> &>(res).Rscale((*Lweight)(*pivotCol,iib,iis,iim,tmpva));

        return res;
    }

    NiceAssert( ccontent );

    if ( !nbase )
    {
	return ((*ccontent)((*pivotRow).v(iibRow+(i*iisRow))))(jb,js,jm,res);
    }

    //return (((*ccontent)((*pivotRow)(iibRow+(i*iisRow))))(*pivotCol,iibCol,iisCol,iibCol+((dnumCols-1)*iisCol)))(jb,js,jm,res);

    int ressize = jnumCols;

    int iib0 = iibCol;
    int iis0 = iisCol;
    //int iim0 = iibCol+((dnumCols-1)*iisCol);

    int iib1 = jb;
    int iis1 = js;
    //int iim1 = jm;

    int iib = iib0+(iis0*iib1);
    int iis = iis0*iis1;
    int iim = iib+((ressize-1)*iis);

    return ((*ccontent)((*pivotRow).v(iibRow+(i*iisRow))))(*pivotCol,iib,iis,iim,res);
}

template <class T>
Vector<T> &Matrix<T>::operator()(const char *dummy, int i, const Vector<int> &j, retVector<T> &tmp, retVector<T> &res, retVector<T> &tmpb)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < dnumRows );
    NiceAssert( checkRange(0,dnumCols-1,j) );
    NiceAssert( pivotRow );
    NiceAssert( pivotCol );

    NiceAssert( !iscover );

    if ( iscover )
    {
        NiceAssert( rowfn );
        NiceAssert( !Lweight || !useLweight );

	if ( !nbase )
	{
	    return ((*rowfn)((*pivotRow).v(iibRow+(i*iisRow)),dref,tmpb))(dummy,j,res);
	}

	return (((*rowfn)((*pivotRow).v(iibRow+(i*iisRow)),dref,tmpb))(dummy,*pivotCol,iibCol,iisCol,iibCol+((dnumCols-1)*iisCol),tmp))(dummy,j,res);
    }

    NiceAssert( content );

    if ( !nbase )
    {
	return ((*content)(dummy,(*pivotRow).v(iibRow+(i*iisRow))))(dummy,j,res);
    }

    return (((*content)(dummy,(*pivotRow).v(iibRow+(i*iisRow))))(dummy,*pivotCol,iibCol,iisCol,iibCol+((dnumCols-1)*iisCol),tmp))(dummy,j,res);
}

template <class T>
Vector<T> &Matrix<T>::operator()(const char *dummy, int i, const Vector<int> &j, retVector<T> &tmp, retVector<T> &tmpb, retVector<T> &res, int jb, int js, int jm)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < dnumRows );
    NiceAssert( checkRange(0,dnumCols-1,j) );
    NiceAssert( pivotRow );
    NiceAssert( pivotCol );

    NiceAssert( !iscover );

    if ( iscover )
    {
        NiceAssert( rowfn );
        NiceAssert( !Lweight || !useLweight );

	if ( !nbase )
	{
	    return ((*rowfn)((*pivotRow).v(iibRow+(i*iisRow)),dref,tmpb))(dummy,j,jb,js,jm,res);
	}

	return (((*rowfn)((*pivotRow).v(iibRow+(i*iisRow)),dref,tmpb))(dummy,*pivotCol,iibCol,iisCol,iibCol+((dnumCols-1)*iisCol),tmp))(dummy,j,jb,js,jm,res);
    }

    NiceAssert( content );

    if ( !nbase )
    {
	return ((*content)(dummy,(*pivotRow).v(iibRow+(i*iisRow))))(dummy,j,jb,js,jm,res);
    }

    return (((*content)(dummy,(*pivotRow).v(iibRow+(i*iisRow))))(dummy,*pivotCol,iibCol,iisCol,iibCol+((dnumCols-1)*iisCol),tmp))(dummy,j,jb,js,jm,res);
}

template <class T>
const Vector<T> &Matrix<T>::operator()(int i, const Vector<int> &j, retVector<T> &tmp, retVector<T> &res, retVector<T> &tmpb) const
{
    NiceAssert( i >= 0 );
    NiceAssert( i < dnumRows );
    NiceAssert( checkRange(0,dnumCols-1,j) );
    NiceAssert( pivotRow );
    NiceAssert( pivotCol );

    if ( iscover && ( !Lweight || !useLweight ) )
    {
        NiceAssert( crowfn );

	if ( !nbase )
	{
	    return ((*crowfn)((*pivotRow).v(iibRow+(i*iisRow)),cdref,tmpb))(j,res);
	}

	return (((*crowfn)((*pivotRow).v(iibRow+(i*iisRow)),cdref,tmpb))(*pivotCol,iibCol,iisCol,iibCol+((dnumCols-1)*iisCol),tmp))(j,res);
    }

    else if ( iscover )
    {
        NiceAssert( crowfn );

	if ( !nbase )
	{
            retVector<T> tmpres;
            retVector<T> tmpva;

	    ((*crowfn)((*pivotRow).v(iibRow+(i*iisRow)),cdref,tmpb))(j,tmpres);

            res.reset("&",tmpres); // this is the overwrite form

            static_cast<Vector<T> &>(res).Lscale((*Lweight).v((*pivotRow).v(iibRow+(i*iisRow))));
            static_cast<Vector<T> &>(res).Rscale((*Lweight)(j,tmpva));

            return res;
	}

        retVector<T> tmpres;
        retVector<T> tmpva;
        retVector<T> tmpvb;

        (((*crowfn)((*pivotRow).v(iibRow+(i*iisRow)),cdref,tmpb))(*pivotCol,iibCol,iisCol,iibCol+((dnumCols-1)*iisCol),tmp))(j,tmpres);

        res.reset("&",tmpres); // this is the overwrite form

        static_cast<Vector<T> &>(res).Lscale((*Lweight).v((*pivotRow).v(iibRow+(i*iisRow))));
        static_cast<Vector<T> &>(res).Rscale((*Lweight)(*pivotCol,iibCol,iisCol,iibCol+((dnumCols-1)*iisCol),tmpvb)(j,tmpva));

        return res;
    }

    NiceAssert( ccontent );

    if ( !nbase )
    {
	return ((*ccontent)((*pivotRow).v(iibRow+(i*iisRow))))(j,res);
    }

    return (((*ccontent)((*pivotRow).v(iibRow+(i*iisRow))))(*pivotCol,iibCol,iisCol,iibCol+((dnumCols-1)*iisCol),tmp))(j,res);
}

template <class T>
const Vector<T> &Matrix<T>::operator()(int i, const Vector<int> &j, retVector<T> &tmp, retVector<T> &tmpb, retVector<T> &res, int jb, int js, int jm) const
{
    NiceAssert( i >= 0 );
    NiceAssert( i < dnumRows );
    NiceAssert( checkRange(0,dnumCols-1,j) );
    NiceAssert( pivotRow );
    NiceAssert( pivotCol );

    if ( iscover && ( !Lweight || !useLweight ) )
    {
        NiceAssert( crowfn );

	if ( !nbase )
	{
	    return ((*crowfn)((*pivotRow).v(iibRow+(i*iisRow)),cdref,tmpb))(j,jb,js,jm,res);
	}

	return (((*crowfn)((*pivotRow).v(iibRow+(i*iisRow)),cdref,tmpb))(*pivotCol,iibCol,iisCol,iibCol+((dnumCols-1)*iisCol),tmp))(j,jb,js,jm,res);
    }

    else if ( iscover )
    {
        NiceAssert( crowfn );

	if ( !nbase )
	{
            retVector<T> tmpres;
            retVector<T> tmpva;

	    ((*crowfn)((*pivotRow).v(iibRow+(i*iisRow)),cdref,tmpb))(j,jb,js,jm,tmpres);

            res.reset("&",tmpres); // this is the overwrite form

            static_cast<Vector<T> &>(res).Lscale((*Lweight).v((*pivotRow).v(iibRow+(i*iisRow))));
            static_cast<Vector<T> &>(res).Rscale((*Lweight)(j,jb,js,jm,tmpva));

            return res;
	}

        retVector<T> tmpres;
        retVector<T> tmpva;
        retVector<T> tmpvb;

	(((*crowfn)((*pivotRow).v(iibRow+(i*iisRow)),cdref,tmpb))(*pivotCol,iibCol,iisCol,iibCol+((dnumCols-1)*iisCol),tmp))(j,jb,js,jm,tmpres);

        res.reset("&",tmpres); // this is the overwrite form

        static_cast<Vector<T> &>(res).Lscale((*Lweight).v((*pivotRow).v(iibRow+(i*iisRow))));
        static_cast<Vector<T> &>(res).Rscale((*Lweight)(*pivotCol,iibCol,iisCol,iibCol+((dnumCols-1)*iisCol),tmpvb)(j,jb,js,jm,tmpva));

        return res;
    }

    NiceAssert( ccontent );

    if ( !nbase )
    {
	return ((*ccontent)((*pivotRow).v(iibRow+(i*iisRow))))(j,jb,js,jm,res);
    }

    return (((*ccontent)((*pivotRow).v(iibRow+(i*iisRow))))(*pivotCol,iibCol,iisCol,iibCol+((dnumCols-1)*iisCol),tmp))(j,jb,js,jm,res);
}

//template <class T>
//Vector<T> &Matrix<T>::operator()(const char *dummy, int i, retVector<T> &res)
//{
//    return (*this)(dummy,i,0,1,dnumCols-1,res);
//}
//    NiceAssert( i >= 0 );
//    NiceAssert( i < dnumRows );
//    NiceAssert( pivotRow );
//    NiceAssert( pivotCol );
//
//    if ( iscover )
//    {
//      NiceAssert( rowfn );
//
//	if ( !nbase )
//	{
//	    return (*rowfn)((*pivotRow)(iibRow+(i*iisRow)),dref);
//	}
//
//	return (((*rowfn)((*pivotRow)(iibRow+(i*iisRow)),dref))(dummy,*pivotCol,iibCol+((dnumCols-1)*iisCol)+1))(dummy,iibCol,iisCol,iibCol+((dnumCols-1)*iisCol));
//    }
//
//    NiceAssert( content );
//
//    if ( !nbase )
//    {
//// problem X at end	((*content)(dummy,(*pivotRow)(iibRow+(i*iisRow)))).fixsize = 1;
//
//	return (*content)(dummy,(*pivotRow)(iibRow+(i*iisRow)));
//    }
//
////problem X at end    ((((*content)(dummy,(*pivotRow)(iibRow+(i*iisRow))))(dummy,*pivotCol,iibCol+((dnumCols-1)*iisCol)+1))(dummy,iibCol,iisCol,iibCol+((dnumCols-1)*iisCol))).fixsize = 1;
//
//    return (((*content)(dummy,(*pivotRow)(iibRow+(i*iisRow))))(dummy,*pivotCol,iibCol+((dnumCols-1)*iisCol)+1))(dummy,iibCol,iisCol,iibCol+((dnumCols-1)*iisCol));
//
//
// Problem X: when you refer to the row of a matrix, ie
//
// G(i) or G("&",i)
//
// then we should set fixsize on that row to protect it from being
// resized, thereby messing up the matrix (one row is too long or too
// short and doesn't match dnumCols).  However if you do set fixsize
// then *it stays set*.  So when you come to add or remove a row, or
// Alistair Shilton 2014 (c) wrote this code
// shuffle rows, the required resize call within that will assert
// that !fixsize and then fail via a throw.
//
// This can come up with the following code:
//
// G("&",i) = whatever;
// G.addRow(0);
//
// The problem is that if we try to record if/what fixsize in contents
// have been set and reset them when required then we end up stuck in a
// constant thrash of resetting them, which is a waste of computational
// time.  The alternative is to return a child of the vector, but this
// is computationally even worse.
//
// Current workaround: don't set fixsize, assume that the user doesn't
// do anything suicidal.
//
// Possible solution: have a "fixsize set" bit and a vector of integers
// for which vectors fixsize is set for.  Whenever a call to add, remove,
// resize, shuffle etc is made if this bit is set then all of the
// relevant fixsize's are reset and the bit cleared.
//
// SOLUTION: rather than use fixsize, return a child.  It's a bit slower,
// but will prevent add/remove being called successfully without resorting
// to fixsize.  Hence the commenting out and replacement of the above code.
//}

template <class T>
Vector<T> &Matrix<T>::operator()(const char *dummy, int i, const char *dummyb, retVector<T> &tempdonotassume, retVector<T> &tmpb)
{
    // This version is like the above, but doesn't care about problem X as it
    // is for internal use only "&"), so we can use it for speed where we
    // know that it will be safe.

    NiceAssert( i >= 0 );
    NiceAssert( i < dnumRows );
    NiceAssert( pivotRow );
    NiceAssert( pivotCol );

    if ( iscover )
    {
        NiceAssert( rowfn );

	if ( !nbase )
	{
	    return (*rowfn)((*pivotRow).v(iibRow+(i*iisRow)),dref,tmpb);
	}

        int iib = iibCol;
        int iis = iisCol;
        int iim = iibCol+((dnumCols-1)*iisCol);

	return  ((*rowfn)((*pivotRow).v(iibRow+(i*iisRow)),dref,tmpb))(dummy,*pivotCol,iib,iis,iim,tempdonotassume);

	//return (((*rowfn)((*pivotRow).v(iibRow+(i*iisRow)),dref,tmpb))(dummy,*pivotCol,iibCol+((dnumCols-1)*iisCol)+1))(dummy,iibCol,iisCol,iibCol+((dnumCols-1)*iisCol),tempdonotassume);
    }

    NiceAssert( content );

    if ( !nbase )
    {
// problem X at end	((*content)(dummy,(*pivotRow).v(iibRow+(i*iisRow)))).fixsize = 1;
	return (*content)(dummy,(*pivotRow).v(iibRow+(i*iisRow)));
    }

    int iib = iibCol;
    int iis = iisCol;
    int iim = iibCol+((dnumCols-1)*iisCol);

    return ((*content)(dummyb,(*pivotRow).v(iibRow+(i*iisRow))))(dummy,*pivotCol,iib,iis,iim,tempdonotassume);

////problem X at end    ((((*content)(dummy,(*pivotRow)(iibRow+(i*iisRow))))(dummy,*pivotCol,iibCol+((dnumCols-1)*iisCol)+1))(dummy,iibCol,iisCol,iibCol+((dnumCols-1)*iisCol))).fixsize = 1;
//    return (((*content)(dummyb,(*pivotRow)(iibRow+(i*iisRow))))(dummy,*pivotCol,iibCol+((dnumCols-1)*iisCol)+1))(dummy,iibCol,iisCol,iibCol+((dnumCols-1)*iisCol),tempdonotassume);
}

























template <class T>
Matrix<T> &Matrix<T>::operator()(const char *, int ib, int is, int im, int jb, int js, int jm, retMatrix<T> &res)
{
    NiceAssert( ( ( is > 0 ) && ( ib >= 0 ) ) || ( ( is < 0 ) && ( ib < dnumRows ) ) );
    NiceAssert( ( ( is < 0 ) && ( im >= 0 ) ) || ( ( is > 0 ) && ( im < dnumRows ) ) );
    NiceAssert( is );
    NiceAssert( ( ( js > 0 ) && ( jb >= 0 ) ) || ( ( js < 0 ) && ( jb < dnumCols ) ) );
    NiceAssert( ( ( js < 0 ) && ( jm >= 0 ) ) || ( ( js > 0 ) && ( jm < dnumCols ) ) );
    NiceAssert( js );

    int inumRows = ((im-ib)/is)+1;
    int jnumCols = ((jm-jb)/js)+1;

    inumRows = ( inumRows < 0 ) ? 0 : inumRows;
    jnumCols = ( jnumCols < 0 ) ? 0 : jnumCols;

    res.reset(*this);

    res.dnumRows = inumRows;
    res.iibRow   = iibRow+(iisRow*ib);
    res.iisRow   = iisRow*is;
    res.pivotRow = pivotRow;

    res.dnumCols = jnumCols;
    res.iibCol   = iibCol+(iisCol*jb);
    res.iisCol   = iisCol*js;
    res.pivotCol = pivotCol;

    return res;
}

template <class T>
const Matrix<T> &Matrix<T>::operator()(int ib, int is, int im, int jb, int js, int jm, retMatrix<T> &res) const
{
    NiceAssert( ( ( is > 0 ) && ( ib >= 0 ) ) || ( ( is < 0 ) && ( ib < dnumRows ) ) );
    NiceAssert( ( ( is < 0 ) && ( im >= 0 ) ) || ( ( is > 0 ) && ( im < dnumRows ) ) );
    NiceAssert( is );
    NiceAssert( ( ( js > 0 ) && ( jb >= 0 ) ) || ( ( js < 0 ) && ( jb < dnumCols ) ) );
    NiceAssert( ( ( js < 0 ) && ( jm >= 0 ) ) || ( ( js > 0 ) && ( jm < dnumCols ) ) );
    NiceAssert( js );

    int inumRows = ((im-ib)/is)+1;
    int jnumCols = ((jm-jb)/js)+1;

    inumRows = ( inumRows < 0 ) ? 0 : inumRows;
    jnumCols = ( jnumCols < 0 ) ? 0 : jnumCols;

    res.creset(*this);

    res.dnumRows = inumRows;
    res.iibRow   = iibRow+(iisRow*ib);
    res.iisRow   = iisRow*is;
    res.pivotRow = pivotRow;

    res.dnumCols = jnumCols;
    res.iibCol   = iibCol+(iisCol*jb);
    res.iisCol   = iisCol*js;
    res.pivotCol = pivotCol;

    return res;
}

template <class T>
Matrix<T> &Matrix<T>::operator()(const char *, int ib, int is, int im, const Vector<int> &j, retMatrix<T> &res)
{
    NiceAssert( ( ( is > 0 ) && ( ib >= 0 ) ) || ( ( is < 0 ) && ( ib < dnumRows ) ) );
    NiceAssert( ( ( is < 0 ) && ( im >= 0 ) ) || ( ( is > 0 ) && ( im < dnumRows ) ) );
    NiceAssert( is );
    NiceAssert( checkRange(0,dnumCols-1,j) );

    int inumRows = ((im-ib)/is)+1;
    int jnumCols = j.size();

    inumRows = ( inumRows < 0 ) ? 0 : inumRows;
    //jnumCols = ( jnumCols < 0 ) ? 0 : jnumCols;

    //if ( !nbase && !(j.base()) && ( iibCol == 0 ) && ( iisCol == 1 ) )
    if ( ( !nbase || ( ( iibCol == 0 ) && ( iisCol == 1 ) && ( pivotCol == cntintarray(0) ) ) ) && !(j.base()) )
    {
        res.reset(*this);

        res.dnumRows = inumRows;
        res.iibRow   = iibRow+(iisRow*ib);
        res.iisRow   = iisRow*is;
        res.pivotRow = pivotRow;

	res.dnumCols = jnumCols;
	res.iibCol   = 0;
	res.iisCol   = 1;
        res.pivotCol = res.dnumCols ? j.ccontent : cntintarray(res.iibCol);
    }

    else
    {
        DynArray<int> *tpiv = res.reset_c(*this,jnumCols);

        res.dnumRows = inumRows;
        res.iibRow   = iibRow+(iisRow*ib);
        res.iisRow   = iisRow*is;
        res.pivotRow = pivotRow;

	res.dnumCols = jnumCols;
	res.iibCol   = 0;
	res.iisCol   = 1;

        for ( int jj = 0 ; jj < res.dnumCols ; ++jj )
        {
            (*tpiv).sv(jj,(*pivotCol).v(iibCol+(iisCol*(j.v(jj)))));
	}
    }

    return res;
}

template <class T>
Matrix<T> &Matrix<T>::operator()(const char *, int ib, int is, int im, const Vector<int> &j, retMatrix<T> &res, int jb, int js, int jm)
{
    NiceAssert( ( ( is > 0 ) && ( ib >= 0 ) ) || ( ( is < 0 ) && ( ib < dnumRows ) ) );
    NiceAssert( ( ( is < 0 ) && ( im >= 0 ) ) || ( ( is > 0 ) && ( im < dnumRows ) ) );
    NiceAssert( is );
    NiceAssert( ( ( js > 0 ) && ( jb >= 0 ) ) || ( ( js < 0 ) && ( jb < j.size() ) ) );
    NiceAssert( ( ( js < 0 ) && ( jm >= 0 ) ) || ( ( js > 0 ) && ( jm < j.size() ) ) );
    NiceAssert( js );

    int inumRows = ((im-ib)/is)+1;
    int jnumCols = ((jm-jb)/js)+1;

    inumRows = ( inumRows < 0 ) ? 0 : inumRows;
    jnumCols = ( jnumCols < 0 ) ? 0 : jnumCols;

    //if ( !nbase && !(j.base()) && ( iibCol == 0 ) && ( iisCol == 1 ) )
    if ( ( !nbase || ( ( iibCol == 0 ) && ( iisCol == 1 ) && ( pivotCol == cntintarray(0) ) ) ) && !(j.base()) )
    {
        res.reset(*this);

        res.dnumRows = inumRows;
        res.iibRow   = iibRow+(iisRow*ib);
        res.iisRow   = iisRow*is;
        res.pivotRow = pivotRow;

	res.dnumCols = jnumCols;
	res.iibCol   = jb;
	res.iisCol   = js;
        res.pivotCol = res.dnumCols ? j.ccontent : cntintarray(res.iibCol);
    }

    else
    {
        DynArray<int> *tpiv = res.reset_c(*this,jnumCols);

        res.dnumRows = inumRows;
        res.iibRow   = iibRow+(iisRow*ib);
        res.iisRow   = iisRow*is;
        res.pivotRow = pivotRow;

	res.dnumCols = jnumCols;
	res.iibCol   = jb;
	res.iisCol   = js;

        for ( int jj = 0 ; jj < jnumCols ; ++jj )
        {
            (*tpiv).sv(jj,(*pivotCol).v(iibCol+(iisCol*(j.v(jj)))));
	}
    }

    return res;
}

template <class T>
const Matrix<T> &Matrix<T>::operator()(int ib, int is, int im, const Vector<int> &j, retMatrix<T> &res) const
{
    NiceAssert( ( ( is > 0 ) && ( ib >= 0 ) ) || ( ( is < 0 ) && ( ib < dnumRows ) ) );
    NiceAssert( ( ( is < 0 ) && ( im >= 0 ) ) || ( ( is > 0 ) && ( im < dnumRows ) ) );
    NiceAssert( is );
    NiceAssert( checkRange(0,dnumCols-1,j) );

    int inumRows = ((im-ib)/is)+1;
    int jnumCols = j.size();

    inumRows = ( inumRows < 0 ) ? 0 : inumRows;
    //jnumCols = ( jnumCols < 0 ) ? 0 : jnumCols;

    //if ( !nbase && !(j.base()) && ( iibCol == 0 ) && ( iisCol == 1 ) )
    if ( ( !nbase || ( ( iibCol == 0 ) && ( iisCol == 1 ) && ( pivotCol == cntintarray(0) ) ) ) && !(j.base()) )
    {
        res.creset(*this);

        res.dnumRows = inumRows;
        res.iibRow   = iibRow+(iisRow*ib);
        res.iisRow   = iisRow*is;
        res.pivotRow = pivotRow;

	res.dnumCols = jnumCols;
	res.iibCol   = 0;
	res.iisCol   = 1;
        res.pivotCol = res.dnumCols ? j.ccontent : cntintarray(res.iibCol);
    }

    else
    {
        DynArray<int> *tpiv = res.creset_c(*this,jnumCols);

        res.dnumRows = inumRows;
        res.iibRow   = iibRow+(iisRow*ib);
        res.iisRow   = iisRow*is;
        res.pivotRow = pivotRow;

	res.dnumCols = jnumCols;
	res.iibCol   = 0;
	res.iisCol   = 1;

        for ( int jj = 0 ; jj < res.dnumCols ; ++jj )
        {
            (*tpiv).sv(jj,(*pivotCol).v(iibCol+(iisCol*(j.v(jj)))));
	}
    }

    return res;
}

template <class T>
const Matrix<T> &Matrix<T>::operator()(int ib, int is, int im, const Vector<int> &j, retMatrix<T> &res, int jb, int js, int jm) const
{
    NiceAssert( ( ( is > 0 ) && ( ib >= 0 ) ) || ( ( is < 0 ) && ( ib < dnumRows ) ) );
    NiceAssert( ( ( is < 0 ) && ( im >= 0 ) ) || ( ( is > 0 ) && ( im < dnumRows ) ) );
    NiceAssert( is );
    NiceAssert( ( ( js > 0 ) && ( jb >= 0 ) ) || ( ( js < 0 ) && ( jb < j.size() ) ) );
    NiceAssert( ( ( js < 0 ) && ( jm >= 0 ) ) || ( ( js > 0 ) && ( jm < j.size() ) ) );
    NiceAssert( js );

    int inumRows = ((im-ib)/is)+1;
    int jnumCols = ((jm-jb)/js)+1;

    inumRows = ( inumRows < 0 ) ? 0 : inumRows;
    jnumCols = ( jnumCols < 0 ) ? 0 : jnumCols;

    //if ( !nbase && !(j.base()) && ( iibCol == 0 ) && ( iisCol == 1 ) )
    if ( ( !nbase || ( ( iibCol == 0 ) && ( iisCol == 1 ) && ( pivotCol == cntintarray(0) ) ) ) && !(j.base()) )
    {
        res.creset(*this);

        res.dnumRows = inumRows;
        res.iibRow   = iibRow+(iisRow*ib);
        res.iisRow   = iisRow*is;
        res.pivotRow = pivotRow;

	res.dnumCols = jnumCols;
	res.iibCol   = jb;
	res.iisCol   = js;
        res.pivotCol = res.dnumCols ? j.ccontent : cntintarray(res.iibCol);
    }

    else
    {
        DynArray<int> *tpiv = res.creset_c(*this,jnumCols);

        res.dnumRows = inumRows;
        res.iibRow   = iibRow+(iisRow*ib);
        res.iisRow   = iisRow*is;
        res.pivotRow = pivotRow;

	res.dnumCols = jnumCols;
	res.iibCol   = jb;
	res.iisCol   = js;

        for ( int jj = 0 ; jj < jnumCols ; ++jj )
        {
            (*tpiv).sv(jj,(*pivotCol).v(iibCol+(iisCol*(j.v(jj)))));
	}
    }

    return res;
}


template <class T>
Matrix<T> &Matrix<T>::operator()(const char *, const Vector<int> &i, int jb, int js, int jm, retMatrix<T> &res)
{
    NiceAssert( checkRange(0,dnumRows-1,i) );
    NiceAssert( ( ( js > 0 ) && ( jb >= 0 ) ) || ( ( js < 0 ) && ( jb < dnumCols ) ) );
    NiceAssert( ( ( js < 0 ) && ( jm >= 0 ) ) || ( ( js > 0 ) && ( jm < dnumCols ) ) );
    NiceAssert( js );

    int inumRows = i.size();
    int jnumCols = ((jm-jb)/js)+1;

    //inumRows = ( inumRows < 0 ) ? 0 : inumRows;
    jnumCols = ( jnumCols < 0 ) ? 0 : jnumCols;

    //if ( !nbase && !(i.base()) && ( iibRow == 0 ) && ( iisRow == 1 ) )
    if ( ( !nbase || ( ( iibRow == 0 ) && ( iisRow == 1 ) && ( pivotRow == cntintarray(0) ) ) ) && !(i.base()) )
    {
        res.reset(*this);

	res.dnumRows = inumRows;
	res.iibRow   = 0;
	res.iisRow   = 1;
        res.pivotRow = res.dnumRows ? i.ccontent : cntintarray(res.iibRow);

        res.dnumCols = jnumCols;
        res.iibCol   = iibCol+(iisCol*jb);
        res.iisCol   = iisCol*js;
        res.pivotCol = pivotCol;
    }

    else
    {
        DynArray<int> *tpiv = res.reset_r(*this,inumRows);

	res.dnumRows = inumRows;
	res.iibRow   = 0;
	res.iisRow   = 1;

        for ( int ii = 0 ; ii < res.dnumRows ; ++ii )
        {
            (*tpiv).sv(ii,(*pivotRow).v(iibRow+(iisRow*(i.v(ii)))));
        }

        res.dnumCols = jnumCols;
        res.iibCol   = iibCol+(iisCol*jb);
        res.iisCol   = iisCol*js;
        res.pivotCol = pivotCol;
    }

    return res;
}

template <class T>
Matrix<T> &Matrix<T>::operator()(const char *, const Vector<int> &i, int jb, int js, int jm, retMatrix<T> &res, int ib, int is, int im)
{
    NiceAssert( ( ( is > 0 ) && ( ib >= 0 ) ) || ( ( is < 0 ) && ( ib < i.size() ) ) );
    NiceAssert( ( ( is < 0 ) && ( im >= 0 ) ) || ( ( is > 0 ) && ( im < i.size() ) ) );
    NiceAssert( is );
    NiceAssert( ( ( js > 0 ) && ( jb >= 0 ) ) || ( ( js < 0 ) && ( jb < dnumRows ) ) );
    NiceAssert( ( ( js < 0 ) && ( jm >= 0 ) ) || ( ( js > 0 ) && ( jm < dnumRows ) ) );
    NiceAssert( js );

    int inumRows = ((im-ib)/is)+1;
    int jnumCols = ((jm-jb)/js)+1;

    inumRows = ( inumRows < 0 ) ? 0 : inumRows;
    jnumCols = ( jnumCols < 0 ) ? 0 : jnumCols;

    //if ( !nbase && !(i.base()) && ( iibRow == 0 ) && ( iisRow == 1 ) )
    if ( ( !nbase || ( ( iibRow == 0 ) && ( iisRow == 1 ) && ( pivotRow == cntintarray(0) ) ) ) && !(i.base()) )
    {
        res.reset(*this);

	res.dnumRows = inumRows;
	res.iibRow   = ib;
	res.iisRow   = is;
        res.pivotRow = res.dnumRows ? i.ccontent : cntintarray(res.iibRow);

        res.dnumCols = jnumCols;
        res.iibCol   = iibCol+(iisCol*jb);
        res.iisCol   = iisCol*js;
        res.pivotCol = pivotCol;
    }

    else
    {
        DynArray<int> *tpiv = res.reset_r(*this,inumRows);

	res.dnumRows = inumRows;
	res.iibRow   = ib;
	res.iisRow   = is;

        for ( int ii = 0 ; ii < res.dnumRows ; ++ii )
        {
            (*tpiv).sv(ii,(*pivotRow).v(iibRow+(iisRow*(i.v(ii)))));
        }

        res.dnumCols = jnumCols;
        res.iibCol   = iibCol+(iisCol*jb);
        res.iisCol   = iisCol*js;
        res.pivotCol = pivotCol;
    }

    return res;
}

template <class T>
const Matrix<T> &Matrix<T>::operator()(const Vector<int> &i, int jb, int js, int jm, retMatrix<T> &res) const
{
    NiceAssert( checkRange(0,dnumRows-1,i) );
    NiceAssert( ( ( js > 0 ) && ( jb >= 0 ) ) || ( ( js < 0 ) && ( jb < dnumCols ) ) );
    NiceAssert( ( ( js < 0 ) && ( jm >= 0 ) ) || ( ( js > 0 ) && ( jm < dnumCols ) ) );
    NiceAssert( js );

    int inumRows = i.size();
    int jnumCols = ((jm-jb)/js)+1;

    //inumRows = ( inumRows < 0 ) ? 0 : inumRows;
    jnumCols = ( jnumCols < 0 ) ? 0 : jnumCols;

    //if ( !nbase && !(i.base()) && ( iibRow == 0 ) && ( iisRow == 1 ) )
    if ( ( !nbase || ( ( iibRow == 0 ) && ( iisRow == 1 ) && ( pivotRow == cntintarray(0) ) ) ) && !(i.base()) )
    {
        res.creset(*this);

	res.dnumRows = inumRows;
	res.iibRow   = 0;
	res.iisRow   = 1;
        res.pivotRow = res.dnumRows ? i.ccontent : cntintarray(res.iibRow);

        res.dnumCols = jnumCols;
        res.iibCol   = iibCol+(iisCol*jb);
        res.iisCol   = iisCol*js;
        res.pivotCol = pivotCol;
    }

    else
    {
        DynArray<int> *tpiv = res.creset_r(*this,inumRows);

	res.dnumRows = inumRows;
	res.iibRow   = 0;
	res.iisRow   = 1;

        for ( int ii = 0 ; ii < res.dnumRows ; ++ii )
        {
            (*tpiv).sv(ii,(*pivotRow).v(iibRow+(iisRow*(i.v(ii)))));
        }

        res.dnumCols = jnumCols;
        res.iibCol   = iibCol+(iisCol*jb);
        res.iisCol   = iisCol*js;
        res.pivotCol = pivotCol;
    }

    return res;
}

template <class T>
const Matrix<T> &Matrix<T>::operator()(const Vector<int> &i, int jb, int js, int jm, retMatrix<T> &res, int ib, int is, int im) const
{
    NiceAssert( ( ( is > 0 ) && ( ib >= 0 ) ) || ( ( is < 0 ) && ( ib < dnumRows ) ) );
    NiceAssert( ( ( is < 0 ) && ( im >= 0 ) ) || ( ( is > 0 ) && ( im < dnumRows ) ) );
    NiceAssert( is );
//    NiceAssert( ( ( js > 0 ) && ( jb >= 0 ) ) || ( ( js < 0 ) && ( jb < j.size() ) ) );
//    NiceAssert( ( ( js < 0 ) && ( jm >= 0 ) ) || ( ( js > 0 ) && ( jm < j.size() ) ) );
    NiceAssert( js );

    int inumRows = ((im-ib)/is)+1;
    int jnumCols = ((jm-jb)/js)+1;

    inumRows = ( inumRows < 0 ) ? 0 : inumRows;
    jnumCols = ( jnumCols < 0 ) ? 0 : jnumCols;

    //if ( !nbase && !(i.base()) && ( iibRow == 0 ) && ( iisRow == 1 ) )
    if ( ( !nbase || ( ( iibRow == 0 ) && ( iisRow == 1 ) && ( pivotRow == cntintarray(0) ) ) ) && !(i.base()) )
    {
        res.creset(*this);

	res.dnumRows = inumRows;
	res.iibRow   = ib;
	res.iisRow   = is;
        res.pivotRow = res.dnumRows ? i.ccontent : cntintarray(res.iibRow);

        res.dnumCols = jnumCols;
        res.iibCol   = iibCol+(iisCol*jb);
        res.iisCol   = iisCol*js;
        res.pivotCol = pivotCol;
    }

    else
    {
        DynArray<int> *tpiv = res.creset_r(*this,inumRows);

	res.dnumRows = inumRows;
	res.iibRow   = ib;
	res.iisRow   = is;

        for ( int ii = 0 ; ii < res.dnumRows ; ++ii )
        {
            (*tpiv).sv(ii,(*pivotRow).v(iibRow+(iisRow*(i.v(ii)))));
        }

        res.dnumCols = jnumCols;
        res.iibCol   = iibCol+(iisCol*jb);
        res.iisCol   = iisCol*js;
        res.pivotCol = pivotCol;
    }

    return res;
}




template <class T>
Matrix<T> &Matrix<T>::operator()(const char *, const Vector<int> &i, const Vector<int> &j, retMatrix<T> &res)
{
    NiceAssert( checkRange(0,dnumRows-1,i) );
    NiceAssert( checkRange(0,dnumCols-1,j) );

    int inumRows = i.size(); //((im-ib)/is)+1;
    int jnumCols = j.size(); //((jm-jb)/js)+1;

    //inumRows = ( inumRows < 0 ) ? 0 : inumRows;
    //jnumCols = ( jnumCols < 0 ) ? 0 : jnumCols;

    //if ( !nbase && !(i.base()) && !(j.base()) && ( iibRow == 0 ) && ( iisRow == 1 ) && ( iibCol == 0 ) && ( iisCol == 1 ) )
    if ( ( ( !nbase || ( ( iibRow == 0 ) && ( iisRow == 1 ) && ( pivotRow == cntintarray(0) ) ) ) && !(i.base()) ) &&
         ( ( !nbase || ( ( iibCol == 0 ) && ( iisCol == 1 ) && ( pivotCol == cntintarray(0) ) ) ) && !(j.base()) )    )
    {
        res.reset(*this);

	res.dnumRows = inumRows;
	res.iibRow   = 0;
	res.iisRow   = 1;
        res.pivotRow = res.dnumRows ? i.ccontent : cntintarray(res.iibRow);

	res.dnumCols = jnumCols;
	res.iibCol   = 0;
	res.iisCol   = 1;
        res.pivotCol = res.dnumCols ? j.ccontent : cntintarray(res.iibCol);
    }

    //else if ( !nbase && !(i.base()) && ( iibRow == 0 ) && ( iisRow == 1 ) )
    else if ( ( !nbase || ( ( iibRow == 0 ) && ( iisRow == 1 ) && ( pivotRow == cntintarray(0) ) ) ) && !(i.base()) )
    {
        DynArray<int> *tpiv = res.reset_c(*this,jnumCols);

	res.dnumRows = inumRows;
	res.iibRow   = 0;
	res.iisRow   = 1;
        res.pivotRow = res.dnumRows ? i.ccontent : cntintarray(res.iibRow);

	res.dnumCols = jnumCols;
	res.iibCol   = 0;
	res.iisCol   = 1;

        for ( int jj = 0 ; jj < res.dnumCols ; ++jj )
        {
            (*tpiv).sv(jj,(*pivotCol).v(iibCol+(iisCol*(j.v(jj)))));
	}
    }

    //else if ( !nbase && !(j.base()) && ( iibCol == 0 ) && ( iisCol == 1 ) )
    else if ( ( !nbase || ( ( iibCol == 0 ) && ( iisCol == 1 ) && ( pivotCol == cntintarray(0) ) ) ) && !(j.base()) )
    {
        DynArray<int> *tpiv = res.reset_r(*this,inumRows);

	res.dnumRows = inumRows;
	res.iibRow   = 0;
	res.iisRow   = 1;

        for ( int ii = 0 ; ii < res.dnumRows ; ++ii )
        {
            (*tpiv).sv(ii,(*pivotRow).v(iibRow+(iisRow*(i.v(ii)))));
	}

	res.dnumCols = jnumCols;
	res.iibCol   = 0;
	res.iisCol   = 1;
        res.pivotCol = res.dnumCols ? j.ccontent : cntintarray(res.iibCol);
    }

    else
    {
        DynArray<int> *tpivCol = nullptr;
        DynArray<int> *tpivRow = res.reset_rc(*this,tpivCol,inumRows,jnumCols);

	res.dnumRows = inumRows;
	res.iibRow   = 0;
	res.iisRow   = 1;

        for ( int ii = 0 ; ii < res.dnumRows ; ++ii )
        {
            (*tpivRow).sv(ii,(*pivotRow).v(iibRow+(iisRow*(i.v(ii)))));
	}

	res.dnumCols = jnumCols;
	res.iibCol   = 0;
	res.iisCol   = 1;

        for ( int jj = 0 ; jj < res.dnumCols ; ++jj )
        {
            (*tpivCol).sv(jj,(*pivotCol).v(iibCol+(iisCol*(j.v(jj)))));
	}
    }

    return res;
}

template <class T>
Matrix<T> &Matrix<T>::operator()(const char *, const Vector<int> &i, const Vector<int> &j, retMatrix<T> &res, int ib, int is, int im)
{
    NiceAssert( ( ( is > 0 ) && ( ib >= 0 ) ) || ( ( is < 0 ) && ( ib < dnumRows ) ) );
    NiceAssert( ( ( is < 0 ) && ( im >= 0 ) ) || ( ( is > 0 ) && ( im < dnumRows ) ) );
    NiceAssert( is );
    NiceAssert( checkRange(0,dnumCols-1,j) );

    int inumRows = ((im-ib)/is)+1;
    int jnumCols = j.size(); //((jm-jb)/js)+1;

    inumRows = ( inumRows < 0 ) ? 0 : inumRows;
    //jnumCols = ( jnumCols < 0 ) ? 0 : jnumCols;

    //if ( !nbase && !(i.base()) && !(j.base()) && ( iibRow == 0 ) && ( iisRow == 1 ) && ( iibCol == 0 ) && ( iisCol == 1 ) )
    if ( ( ( !nbase || ( ( iibRow == 0 ) && ( iisRow == 1 ) && ( pivotRow == cntintarray(0) ) ) ) && !(i.base()) ) &&
         ( ( !nbase || ( ( iibCol == 0 ) && ( iisCol == 1 ) && ( pivotCol == cntintarray(0) ) ) ) && !(j.base()) )    )
    {
        res.reset(*this);

	res.dnumRows = inumRows;
	res.iibRow   = ib;
	res.iisRow   = is;
        res.pivotRow = res.dnumRows ? i.ccontent : cntintarray(res.iibRow);

	res.dnumCols = jnumCols;
	res.iibCol   = 0;
	res.iisCol   = 1;
        res.pivotCol = res.dnumCols ? j.ccontent : cntintarray(res.iibCol);
    }

    //else if ( !nbase && !(i.base()) && ( iibRow == 0 ) && ( iisRow == 1 ) )
    else if ( ( !nbase || ( ( iibRow == 0 ) && ( iisRow == 1 ) && ( pivotRow == cntintarray(0) ) ) ) && !(i.base()) )
    {
        DynArray<int> *tpiv = res.reset_c(*this,jnumCols);

	res.dnumRows = inumRows;
	res.iibRow   = ib;
	res.iisRow   = is;
        res.pivotRow = res.dnumRows ? i.ccontent : cntintarray(res.iibRow);

	res.dnumCols = jnumCols;
	res.iibCol   = 0;
	res.iisCol   = 1;

        for ( int jj = 0 ; jj < res.dnumCols ; ++jj )
        {
            (*tpiv).sv(jj,(*pivotCol).v(iibCol+(iisCol*(j.v(jj)))));
	}
    }

    //else if ( !nbase && !(j.base()) && ( iibCol == 0 ) && ( iisCol == 1 ) )
    else if ( ( !nbase || ( ( iibCol == 0 ) && ( iisCol == 1 ) && ( pivotCol == cntintarray(0) ) ) ) && !(j.base()) )
    {
        DynArray<int> *tpiv = res.reset_r(*this,inumRows);

	res.dnumRows = inumRows;
	res.iibRow   = ib;
	res.iisRow   = is;

        for ( int ii = 0 ; ii < res.dnumRows ; ++ii )
        {
            (*tpiv).sv(ii,(*pivotRow).v(iibRow+(iisRow*(i.v(ii)))));
	}

	res.dnumCols = jnumCols;
	res.iibCol   = 0;
	res.iisCol   = 1;
        res.pivotCol = res.dnumCols ? j.ccontent : cntintarray(res.iibCol);
    }

    else
    {
        DynArray<int> *tpivCol = nullptr;
        DynArray<int> *tpivRow = res.reset_rc(*this,tpivCol,inumRows,jnumCols);

	res.dnumRows = inumRows;
	res.iibRow   = ib;
	res.iisRow   = is;

        for ( int ii = 0 ; ii < res.dnumRows ; ++ii )
        {
            (*tpivRow).sv(ii,(*pivotRow).v(iibRow+(iisRow*(i.v(ii)))));
	}

	res.dnumCols = jnumCols;
	res.iibCol   = 0;
	res.iisCol   = 1;

        for ( int jj = 0 ; jj < res.dnumCols ; ++jj )
        {
            (*tpivCol).sv(jj,(*pivotCol).v(iibCol+(iisCol*(j.v(jj)))));
	}
    }

    return res;
}

template <class T>
Matrix<T> &Matrix<T>::operator()(const char *, const Vector<int> &i, const Vector<int> &j, retMatrix<T> &res, const char *, int jb, int js, int jm)
{
    NiceAssert( checkRange(0,dnumRows-1,i) );
    NiceAssert( ( ( js > 0 ) && ( jb >= 0 ) ) || ( ( js < 0 ) && ( jb < j.size() ) ) );
    NiceAssert( ( ( js < 0 ) && ( jm >= 0 ) ) || ( ( js > 0 ) && ( jm < j.size() ) ) );
    NiceAssert( js );

    int inumRows = i.size(); //((im-ib)/is)+1;
    int jnumCols = ((jm-jb)/js)+1;

    //inumRows = ( inumRows < 0 ) ? 0 : inumRows;
    jnumCols = ( jnumCols < 0 ) ? 0 : jnumCols;

    //if ( !nbase && !(i.base()) && !(j.base()) && ( iibRow == 0 ) && ( iisRow == 1 ) && ( iibCol == 0 ) && ( iisCol == 1 ) )
    if ( ( ( !nbase || ( ( iibRow == 0 ) && ( iisRow == 1 ) && ( pivotRow == cntintarray(0) ) ) ) && !(i.base()) ) &&
         ( ( !nbase || ( ( iibCol == 0 ) && ( iisCol == 1 ) && ( pivotCol == cntintarray(0) ) ) ) && !(j.base()) )    )
    {
        res.reset(*this);

	res.dnumRows = inumRows;
	res.iibRow   = 0;
	res.iisRow   = 1;
        res.pivotRow = res.dnumRows ? i.ccontent : cntintarray(res.iibRow);

	res.dnumCols = jnumCols;
	res.iibCol   = jb;
	res.iisCol   = js;
        res.pivotCol = res.dnumCols ? j.ccontent : cntintarray(res.iibCol);
    }

    //else if ( !nbase && !(i.base()) && ( iibRow == 0 ) && ( iisRow == 1 ) )
    else if ( ( !nbase || ( ( iibRow == 0 ) && ( iisRow == 1 ) && ( pivotRow == cntintarray(0) ) ) ) && !(i.base()) )
    {
        DynArray<int> *tpiv = res.reset_c(*this,jnumCols);

	res.dnumRows = inumRows;
	res.iibRow   = 0;
	res.iisRow   = 1;
        res.pivotRow = res.dnumRows ? i.ccontent : cntintarray(res.iibRow);

	res.dnumCols = jnumCols;
	res.iibCol   = jb;
	res.iisCol   = js;

        for ( int jj = 0 ; jj < res.dnumCols ; ++jj )
        {
            (*tpiv).sv(jj,(*pivotCol).v(iibCol+(iisCol*(j.v(jj)))));
	}
    }

    //else if ( !nbase && !(j.base()) && ( iibCol == 0 ) && ( iisCol == 1 ) )
    else if ( ( !nbase || ( ( iibCol == 0 ) && ( iisCol == 1 ) && ( pivotCol == cntintarray(0) ) ) ) && !(j.base()) )
    {
        DynArray<int> *tpiv = res.reset_r(*this,inumRows);

	res.dnumRows = inumRows;
	res.iibRow   = 0;
	res.iisRow   = 1;

        for ( int ii = 0 ; ii < res.dnumRows ; ++ii )
        {
            (*tpiv).sv(ii,(*pivotRow).v(iibRow+(iisRow*(i.v(ii)))));
	}

	res.dnumCols = jnumCols;
	res.iibCol   = jb;
	res.iisCol   = js;
        res.pivotCol = res.dnumCols ? j.ccontent : cntintarray(res.iibCol);
    }

    else
    {
        DynArray<int> *tpivCol = nullptr;
        DynArray<int> *tpivRow = res.reset_rc(*this,tpivCol,inumRows,jnumCols);

	res.dnumRows = inumRows;
	res.iibRow   = 0;
	res.iisRow   = 1;

        for ( int ii = 0 ; ii < res.dnumRows ; ++ii )
        {
            (*tpivRow).sv(ii,(*pivotRow).v(iibRow+(iisRow*(i.v(ii)))));
	}

	res.dnumCols = jnumCols;
	res.iibCol   = jb;
	res.iisCol   = js;

        for ( int jj = 0 ; jj < res.dnumCols ; ++jj )
        {
            (*tpivCol).sv(jj,(*pivotCol).v(iibCol+(iisCol*(j.v(jj)))));
	}
    }

    return res;
}

template <class T>
Matrix<T> &Matrix<T>::operator()(const char *, const Vector<int> &i, const Vector<int> &j, retMatrix<T> &res, int ib, int is, int im, int jb, int js, int jm)
{
    NiceAssert( ( ( is > 0 ) && ( ib >= 0 ) ) || ( ( is < 0 ) && ( ib < dnumRows ) ) );
    NiceAssert( ( ( is < 0 ) && ( im >= 0 ) ) || ( ( is > 0 ) && ( im < dnumRows ) ) );
    NiceAssert( is );
    NiceAssert( ( ( js > 0 ) && ( jb >= 0 ) ) || ( ( js < 0 ) && ( jb < j.size() ) ) );
    NiceAssert( ( ( js < 0 ) && ( jm >= 0 ) ) || ( ( js > 0 ) && ( jm < j.size() ) ) );
    NiceAssert( js );

    int inumRows = ((im-ib)/is)+1;
    int jnumCols = ((jm-jb)/js)+1;

    inumRows = ( inumRows < 0 ) ? 0 : inumRows;
    jnumCols = ( jnumCols < 0 ) ? 0 : jnumCols;

    //if ( !nbase && !(i.base()) && !(j.base()) && ( iibRow == 0 ) && ( iisRow == 1 ) && ( iibCol == 0 ) && ( iisCol == 1 ) )
    if ( ( ( !nbase || ( ( iibRow == 0 ) && ( iisRow == 1 ) && ( pivotRow == cntintarray(0) ) ) ) && !(i.base()) ) &&
         ( ( !nbase || ( ( iibCol == 0 ) && ( iisCol == 1 ) && ( pivotCol == cntintarray(0) ) ) ) && !(j.base()) )    )
    {
        res.reset(*this);

	res.dnumRows = inumRows;
	res.iibRow   = ib;
	res.iisRow   = is;
        res.pivotRow = res.dnumRows ? i.ccontent : cntintarray(res.iibRow);

	res.dnumCols = jnumCols;
	res.iibCol   = jb;
	res.iisCol   = js;
        res.pivotCol = res.dnumCols ? j.ccontent : cntintarray(res.iibCol);
    }

    //else if ( !nbase && !(i.base()) && ( iibRow == 0 ) && ( iisRow == 1 ) )
    else if ( ( !nbase || ( ( iibRow == 0 ) && ( iisRow == 1 ) && ( pivotRow == cntintarray(0) ) ) ) && !(i.base()) )
    {
        DynArray<int> *tpiv = res.reset_c(*this,jnumCols);

	res.dnumRows = inumRows;
	res.iibRow   = ib;
	res.iisRow   = is;
        res.pivotRow = res.dnumRows ? i.ccontent : cntintarray(res.iibRow);

	res.dnumCols = jnumCols;
	res.iibCol   = jb;
	res.iisCol   = js;

        for ( int jj = 0 ; jj < res.dnumCols ; ++jj )
        {
            (*tpiv).sv(jj,(*pivotCol).v(iibCol+(iisCol*(j.v(jj)))));
	}
    }

    //else if ( !nbase && !(j.base()) && ( iibCol == 0 ) && ( iisCol == 1 ) )
    else if ( ( !nbase || ( ( iibCol == 0 ) && ( iisCol == 1 ) && ( pivotCol == cntintarray(0) ) ) ) && !(j.base()) )
    {
        DynArray<int> *tpiv = res.reset_r(*this,inumRows);

	res.dnumRows = inumRows;
	res.iibRow   = ib;
	res.iisRow   = is;

        for ( int ii = 0 ; ii < res.dnumRows ; ++ii )
        {
            (*tpiv).sv(ii,(*pivotRow).v(iibRow+(iisRow*(i.v(ii)))));
	}

	res.dnumCols = jnumCols;
	res.iibCol   = jb;
	res.iisCol   = js;
        res.pivotCol = res.dnumCols ? j.ccontent : cntintarray(res.iibCol);
    }

    else
    {
        DynArray<int> *tpivCol = nullptr;
        DynArray<int> *tpivRow = res.reset_rc(*this,tpivCol,inumRows,jnumCols);

	res.dnumRows = inumRows;
	res.iibRow   = ib;
	res.iisRow   = is;

        for ( int ii = 0 ; ii < res.dnumRows ; ++ii )
        {
            (*tpivRow).sv(ii,(*pivotRow).v(iibRow+(iisRow*(i.v(ii)))));
	}

	res.dnumCols = jnumCols;
	res.iibCol   = jb;
	res.iisCol   = js;

        for ( int jj = 0 ; jj < res.dnumCols ; ++jj )
        {
            (*tpivCol).sv(jj,(*pivotCol).v(iibCol+(iisCol*(j.v(jj)))));
	}
    }

    return res;
}



template <class T>
const Matrix<T> &Matrix<T>::operator()(const Vector<int> &i, const Vector<int> &j, retMatrix<T> &res) const
{
    NiceAssert( checkRange(0,dnumRows-1,i) );
    NiceAssert( checkRange(0,dnumCols-1,j) );

    int inumRows = i.size(); //((im-ib)/is)+1;
    int jnumCols = j.size(); //((jm-jb)/js)+1;

    //inumRows = ( inumRows < 0 ) ? 0 : inumRows;
    //jnumCols = ( jnumCols < 0 ) ? 0 : jnumCols;

    //if ( !nbase && !(i.base()) && !(j.base()) && ( iibRow == 0 ) && ( iisRow == 1 ) && ( iibCol == 0 ) && ( iisCol == 1 ) )
    if ( ( ( !nbase || ( ( iibRow == 0 ) && ( iisRow == 1 ) && ( pivotRow == cntintarray(0) ) ) ) && !(i.base()) ) &&
         ( ( !nbase || ( ( iibCol == 0 ) && ( iisCol == 1 ) && ( pivotCol == cntintarray(0) ) ) ) && !(j.base()) )    )
    {
        res.creset(*this);

	res.dnumRows = inumRows;
	res.iibRow   = 0;
	res.iisRow   = 1;
        res.pivotRow = res.dnumRows ? i.ccontent : cntintarray(res.iibRow);

	res.dnumCols = jnumCols;
	res.iibCol   = 0;
	res.iisCol   = 1;
        res.pivotCol = res.dnumCols ? j.ccontent : cntintarray(res.iibCol);
    }

    //else if ( !nbase && !(i.base()) && ( iibRow == 0 ) && ( iisRow == 1 ) )
    else if ( ( !nbase || ( ( iibRow == 0 ) && ( iisRow == 1 ) && ( pivotRow == cntintarray(0) ) ) ) && !(i.base()) )
    {
        DynArray<int> *tpiv = res.creset_c(*this,jnumCols);

	res.dnumRows = inumRows;
	res.iibRow   = 0;
	res.iisRow   = 1;
        res.pivotRow = res.dnumRows ? i.ccontent : cntintarray(res.iibRow);

	res.dnumCols = jnumCols;
	res.iibCol   = 0;
	res.iisCol   = 1;

        for ( int jj = 0 ; jj < res.dnumCols ; ++jj )
        {
            (*tpiv).sv(jj,(*pivotCol).v(iibCol+(iisCol*(j.v(jj)))));
	}
    }

    //else if ( !nbase && !(j.base()) && ( iibCol == 0 ) && ( iisCol == 1 ) )
    else if ( ( !nbase || ( ( iibCol == 0 ) && ( iisCol == 1 ) && ( pivotCol == cntintarray(0) ) ) ) && !(j.base()) )
    {
        DynArray<int> *tpiv = res.creset_r(*this,inumRows);

	res.dnumRows = inumRows;
	res.iibRow   = 0;
	res.iisRow   = 1;

        for ( int ii = 0 ; ii < res.dnumRows ; ++ii )
        {
            (*tpiv).sv(ii,(*pivotRow).v(iibRow+(iisRow*(i.v(ii)))));
	}

	res.dnumCols = jnumCols;
	res.iibCol   = 0;
	res.iisCol   = 1;
        res.pivotCol = res.dnumCols ? j.ccontent : cntintarray(res.iibCol);
    }

    else
    {
        DynArray<int> *tpivCol = nullptr;
        DynArray<int> *tpivRow = res.creset_rc(*this,tpivCol,inumRows,jnumCols);

	res.dnumRows = inumRows;
	res.iibRow   = 0;
	res.iisRow   = 1;

        for ( int ii = 0 ; ii < res.dnumRows ; ++ii )
        {
            (*tpivRow).sv(ii,(*pivotRow).v(iibRow+(iisRow*(i.v(ii)))));
	}

	res.dnumCols = jnumCols;
	res.iibCol   = 0;
	res.iisCol   = 1;

        for ( int jj = 0 ; jj < res.dnumCols ; ++jj )
        {
            (*tpivCol).sv(jj,(*pivotCol).v(iibCol+(iisCol*(j.v(jj)))));
	}
    }

    return res;
}


template <class T>
const Matrix<T> &Matrix<T>::operator()(const Vector<int> &i, const Vector<int> &j, retMatrix<T> &res, int ib, int is, int im) const
{
    NiceAssert( ( ( is > 0 ) && ( ib >= 0 ) ) || ( ( is < 0 ) && ( ib < dnumRows ) ) );
    NiceAssert( ( ( is < 0 ) && ( im >= 0 ) ) || ( ( is > 0 ) && ( im < dnumRows ) ) );
    NiceAssert( is );
    NiceAssert( checkRange(0,dnumCols-1,j) );

    int inumRows = ((im-ib)/is)+1;
    int jnumCols = j.size(); //((jm-jb)/js)+1;

    inumRows = ( inumRows < 0 ) ? 0 : inumRows;
    //jnumCols = ( jnumCols < 0 ) ? 0 : jnumCols;

    //if ( !nbase && !(i.base()) && !(j.base()) && ( iibRow == 0 ) && ( iisRow == 1 ) && ( iibCol == 0 ) && ( iisCol == 1 ) )
    if ( ( ( !nbase || ( ( iibRow == 0 ) && ( iisRow == 1 ) && ( pivotRow == cntintarray(0) ) ) ) && !(i.base()) ) &&
         ( ( !nbase || ( ( iibCol == 0 ) && ( iisCol == 1 ) && ( pivotCol == cntintarray(0) ) ) ) && !(j.base()) )    )
    {
        res.creset(*this);

	res.dnumRows = inumRows;
	res.iibRow   = ib;
	res.iisRow   = is;
        res.pivotRow = res.dnumRows ? i.ccontent : cntintarray(res.iibRow);

	res.dnumCols = jnumCols;
	res.iibCol   = 0;
	res.iisCol   = 1;
        res.pivotCol = res.dnumCols ? j.ccontent : cntintarray(res.iibCol);
    }

    //else if ( !nbase && !(i.base()) && ( iibRow == 0 ) && ( iisRow == 1 ) )
    else if ( ( !nbase || ( ( iibRow == 0 ) && ( iisRow == 1 ) && ( pivotRow == cntintarray(0) ) ) ) && !(i.base()) )
    {
        DynArray<int> *tpiv = res.creset_c(*this,jnumCols);

	res.dnumRows = inumRows;
	res.iibRow   = ib;
	res.iisRow   = is;
        res.pivotRow = res.dnumRows ? i.ccontent : cntintarray(res.iibRow);

	res.dnumCols = jnumCols;
	res.iibCol   = 0;
	res.iisCol   = 1;

        for ( int jj = 0 ; jj < res.dnumCols ; ++jj )
        {
            (*tpiv).sv(jj,(*pivotCol).v(iibCol+(iisCol*(j.v(jj)))));
	}
    }

    //else if ( !nbase && !(j.base()) && ( iibCol == 0 ) && ( iisCol == 1 ) )
    else if ( ( !nbase || ( ( iibCol == 0 ) && ( iisCol == 1 ) && ( pivotCol == cntintarray(0) ) ) ) && !(j.base()) )
    {
        DynArray<int> *tpiv = res.creset_r(*this,inumRows);

	res.dnumRows = inumRows;
	res.iibRow   = ib;
	res.iisRow   = is;

        for ( int ii = 0 ; ii < res.dnumRows ; ++ii )
        {
            (*tpiv).sv(ii,(*pivotRow).v(iibRow+(iisRow*(i.v(ii)))));
	}

	res.dnumCols = jnumCols;
	res.iibCol   = 0;
	res.iisCol   = 1;
        res.pivotCol = res.dnumCols ? j.ccontent : cntintarray(res.iibCol);
    }

    else
    {
        DynArray<int> *tpivCol = nullptr;
        DynArray<int> *tpivRow = res.creset_rc(*this,tpivCol,inumRows,jnumCols);

	res.dnumRows = inumRows;
	res.iibRow   = ib;
	res.iisRow   = is;

        for ( int ii = 0 ; ii < res.dnumRows ; ++ii )
        {
            (*tpivRow).sv(ii,(*pivotRow).v(iibRow+(iisRow*(i.v(ii)))));
	}

	res.dnumCols = jnumCols;
	res.iibCol   = 0;
	res.iisCol   = 1;

        for ( int jj = 0 ; jj < res.dnumCols ; ++jj )
        {
            (*tpivCol).sv(jj,(*pivotCol).v(iibCol+(iisCol*(j.v(jj)))));
	}
    }

    return res;
}

template <class T>
const Matrix<T> &Matrix<T>::operator()(const Vector<int> &i, const Vector<int> &j, retMatrix<T> &res, const char *, int jb, int js, int jm) const
{
    NiceAssert( checkRange(0,dnumRows-1,i) );
    NiceAssert( ( ( js > 0 ) && ( jb >= 0 ) ) || ( ( js < 0 ) && ( jb < j.size() ) ) );
    NiceAssert( ( ( js < 0 ) && ( jm >= 0 ) ) || ( ( js > 0 ) && ( jm < j.size() ) ) );
    NiceAssert( js );

    int inumRows = i.size(); //((im-ib)/is)+1;
    int jnumCols = ((jm-jb)/js)+1;

    //inumRows = ( inumRows < 0 ) ? 0 : inumRows;
    jnumCols = ( jnumCols < 0 ) ? 0 : jnumCols;

    //if ( !nbase && !(i.base()) && !(j.base()) && ( iibRow == 0 ) && ( iisRow == 1 ) && ( iibCol == 0 ) && ( iisCol == 1 ) )
    if ( ( ( !nbase || ( ( iibRow == 0 ) && ( iisRow == 1 ) && ( pivotRow == cntintarray(0) ) ) ) && !(i.base()) ) &&
         ( ( !nbase || ( ( iibCol == 0 ) && ( iisCol == 1 ) && ( pivotCol == cntintarray(0) ) ) ) && !(j.base()) )    )
    {
        res.creset(*this);

	res.dnumRows = inumRows;
	res.iibRow   = 0;
	res.iisRow   = 1;
        res.pivotRow = res.dnumRows ? i.ccontent : cntintarray(res.iibRow);

	res.dnumCols = jnumCols;
	res.iibCol   = jb;
	res.iisCol   = js;
        res.pivotCol = res.dnumCols ? j.ccontent : cntintarray(res.iibCol);
    }

    //else if ( !nbase && !(i.base()) && ( iibRow == 0 ) && ( iisRow == 1 ) )
    else if ( ( !nbase || ( ( iibRow == 0 ) && ( iisRow == 1 ) && ( pivotRow == cntintarray(0) ) ) ) && !(i.base()) )
    {
        DynArray<int> *tpiv = res.creset_c(*this,jnumCols);

	res.dnumRows = inumRows;
	res.iibRow   = 0;
	res.iisRow   = 1;
        res.pivotRow = res.dnumRows ? i.ccontent : cntintarray(res.iibRow);

	res.dnumCols = jnumCols;
	res.iibCol   = jb;
	res.iisCol   = js;

        for ( int jj = 0 ; jj < res.dnumCols ; ++jj )
        {
            (*tpiv).sv(jj,(*pivotCol).v(iibCol+(iisCol*(j.v(jj)))));
	}
    }

    //else if ( !nbase && !(j.base()) && ( iibCol == 0 ) && ( iisCol == 1 ) )
    else if ( ( !nbase || ( ( iibCol == 0 ) && ( iisCol == 1 ) && ( pivotCol == cntintarray(0) ) ) ) && !(j.base()) )
    {
        DynArray<int> *tpiv = res.creset_r(*this,inumRows);

	res.dnumRows = inumRows;
	res.iibRow   = 0;
	res.iisRow   = 1;

        for ( int ii = 0 ; ii < res.dnumRows ; ++ii )
        {
            (*tpiv).sv(ii,(*pivotRow).v(iibRow+(iisRow*(i.v(ii)))));
	}

	res.dnumCols = jnumCols;
	res.iibCol   = jb;
	res.iisCol   = js;
        res.pivotCol = res.dnumCols ? j.ccontent : cntintarray(res.iibCol);
    }

    else
    {
        DynArray<int> *tpivCol = nullptr;
        DynArray<int> *tpivRow = res.creset_rc(*this,tpivCol,inumRows,jnumCols);

	res.dnumRows = inumRows;
	res.iibRow   = 0;
	res.iisRow   = 1;

        for ( int ii = 0 ; ii < res.dnumRows ; ++ii )
        {
            (*tpivRow).sv(ii,(*pivotRow).v(iibRow+(iisRow*(i.v(ii)))));
	}

	res.dnumCols = jnumCols;
	res.iibCol   = jb;
	res.iisCol   = js;

        for ( int jj = 0 ; jj < res.dnumCols ; ++jj )
        {
            (*tpivCol).sv(jj,(*pivotCol).v(iibCol+(iisCol*(j.v(jj)))));
	}
    }

    return res;
}

template <class T>
const Matrix<T> &Matrix<T>::operator()(const Vector<int> &i, const Vector<int> &j, retMatrix<T> &res, int ib, int is, int im, int jb, int js, int jm) const
{
    NiceAssert( ( ( is > 0 ) && ( ib >= 0 ) ) || ( ( is < 0 ) && ( ib < dnumRows ) ) );
    NiceAssert( ( ( is < 0 ) && ( im >= 0 ) ) || ( ( is > 0 ) && ( im < dnumRows ) ) );
    NiceAssert( is );
    NiceAssert( ( ( js > 0 ) && ( jb >= 0 ) ) || ( ( js < 0 ) && ( jb < j.size() ) ) );
    NiceAssert( ( ( js < 0 ) && ( jm >= 0 ) ) || ( ( js > 0 ) && ( jm < j.size() ) ) );
    NiceAssert( js );

    int inumRows = ((im-ib)/is)+1;
    int jnumCols = ((jm-jb)/js)+1;

    inumRows = ( inumRows < 0 ) ? 0 : inumRows;
    jnumCols = ( jnumCols < 0 ) ? 0 : jnumCols;

    //if ( !nbase && !(i.base()) && !(j.base()) && ( iibRow == 0 ) && ( iisRow == 1 ) && ( iibCol == 0 ) && ( iisCol == 1 ) )
    if ( ( ( !nbase || ( ( iibRow == 0 ) && ( iisRow == 1 ) && ( pivotRow == cntintarray(0) ) ) ) && !(i.base()) ) &&
         ( ( !nbase || ( ( iibCol == 0 ) && ( iisCol == 1 ) && ( pivotCol == cntintarray(0) ) ) ) && !(j.base()) )    )
    {
        res.creset(*this);

	res.dnumRows = inumRows;
	res.iibRow   = ib;
	res.iisRow   = is;
        res.pivotRow = res.dnumRows ? i.ccontent : cntintarray(res.iibRow);

	res.dnumCols = jnumCols;
	res.iibCol   = jb;
	res.iisCol   = js;
        res.pivotCol = res.dnumCols ? j.ccontent : cntintarray(res.iibCol);
    }

    //else if ( !nbase && !(i.base()) && ( iibRow == 0 ) && ( iisRow == 1 ) )
    else if ( ( !nbase || ( ( iibRow == 0 ) && ( iisRow == 1 ) && ( pivotRow == cntintarray(0) ) ) ) && !(i.base()) )
    {
        DynArray<int> *tpiv = res.creset_c(*this,jnumCols);

	res.dnumRows = inumRows;
	res.iibRow   = ib;
	res.iisRow   = is;
        res.pivotRow = res.dnumRows ? i.ccontent : cntintarray(res.iibRow);

	res.dnumCols = jnumCols;
	res.iibCol   = jb;
	res.iisCol   = js;

        for ( int jj = 0 ; jj < res.dnumCols ; ++jj )
        {
            (*tpiv).sv(jj,(*pivotCol).v(iibCol+(iisCol*(j.v(jj)))));
	}
    }

    //else if ( !nbase && !(j.base()) && ( iibCol == 0 ) && ( iisCol == 1 ) )
    else if ( ( !nbase || ( ( iibCol == 0 ) && ( iisCol == 1 ) && ( pivotCol == cntintarray(0) ) ) ) && !(j.base()) )
    {
        DynArray<int> *tpiv = res.creset_r(*this,inumRows);

	res.dnumRows = inumRows;
	res.iibRow   = ib;
	res.iisRow   = is;

        for ( int ii = 0 ; ii < res.dnumRows ; ++ii )
        {
            (*tpiv).sv(ii,(*pivotRow).v(iibRow+(iisRow*(i.v(ii)))));
	}

	res.dnumCols = jnumCols;
	res.iibCol   = jb;
	res.iisCol   = js;
        res.pivotCol = res.dnumCols ? j.ccontent : cntintarray(res.iibCol);
    }

    else
    {
        DynArray<int> *tpivCol = nullptr;
        DynArray<int> *tpivRow = res.creset_rc(*this,tpivCol,inumRows,jnumCols);

	res.dnumRows = inumRows;
	res.iibRow   = ib;
	res.iisRow   = is;

        for ( int ii = 0 ; ii < res.dnumRows ; ++ii )
        {
            (*tpivRow).sv(ii,(*pivotRow).v(iibRow+(iisRow*(i.v(ii)))));
	}

	res.dnumCols = jnumCols;
	res.iibCol   = jb;
	res.iisCol   = js;

        for ( int jj = 0 ; jj < res.dnumCols ; ++jj )
        {
            (*tpivCol).sv(jj,(*pivotCol).v(iibCol+(iisCol*(j.v(jj)))));
	}
    }

    return res;
}





// Column access - NO LONGER NEEDED
//
// The matrix is stored in rows.  To access a single row and treat it like
// a vector is therefore easy (see functions above) - you just return a
// reference to the relevant vector.  To access a single column and treat
// it like a vector is harder: the operator() functions cheat a little here
// and return a matrix with a single column.  You can overwrite with a vector
// easily enough, but its still a matrix.  So (for a matrix A, vector x):
//
// A("&",0,1,A.numRows()-1,2,"&") = x   - overwrites column 2 with vector x
// x = A("&",0,1,A.numRows()-1,2,"&")   - fails to compile.
//
// To get around this you could:
//
// - construct a vector and return it.  This would take time and memory
//   and may or may not incur a significant CPU time penalty due to calls
//   to constructors and copy operators when it is returned depending on
//   optimisation.  Also you couldn't just edit the vector returned and
//   see the results in the matrix, so const returns only.
// - make vectors that can use a function to access elements.  This would
//   be elegant, but bad for speed.  Currently vectors are fast because
//   they just dereference memory: if you add in a "is this a function"
//   check then things like inner products and multiplications would slow
//   down significantly.
// - other?
//
// or you could use these functions.
//
// getCol(dest,i) - overwrite dest with column i (return reference to dest)
// setCol(i,src)  - overwrite column i with src (return reference to src)
//
//Vector<T> &getCol(Vector<T> &dest, int i) const;
//Alistair Shilton 2014 (c) wrote this code
//const Vector<T> &setCol(int i, const Vector<T> &src);
//
//template <class T>
//Vector<T> &Matrix<T>::getCol(Vector<T> &dest, int j) const
//{
//    NiceAssert( j >= 0 );
//    NiceAssert( j < dnumCols );
//
//    dest.resize(dnumRows);
//
//    if ( dnumRows )
//    {
//	int i;
//
//	for ( i = 0 ; i < dnumRows ; ++i )
//	{
//            dest("&",i) = (*this)(i,j);
//	}
//    }
//
//    return dest;
//}
//
//template <class T>
//const Vector<T> &Matrix<T>::setCol(int j, const Vector<T> &src)
//{
//    NiceAssert( j >= 0 );
//    NiceAssert( j < dnumCols );
//    NiceAssert( src.size() == dnumRows );
//
//    if ( dnumRows )
//    {
//	int i;
//
//	for ( i = 0 ; i < dnumRows ; ++i )
//	{
//	    (*this)("&",i,j) = src(i);
//	}
//    }
//
//    return src;
//}






























#define MAXHACKYHEAD 1000000

inline const Matrix<int> &zerointmatrixbasic(void)
{
    const static Matrix<int> zerores("&",zerointvecbasic());

    return zerores;
}

inline const Matrix<int> &oneintmatrixbasic(void)
{
    const static Matrix<int> oneres("&",oneintvecbasic());

    return oneres;
}

inline const Matrix<double> &zerodoublematrixbasic(void)
{
    const static Matrix<double> zerores("&",zerodoublevecbasic());

    return zerores;
}

inline const Matrix<double> &onedoublematrixbasic(void)
{
    const static Matrix<double> oneres("&",onedoublevecbasic());

    return oneres;
}

inline const Matrix<int> &zerointmatrix(int numRows, int numCols, retMatrix<int> &tmpm)
{
    return zerointmatrixbasic()(0,1,numRows-1,0,1,numCols-1,tmpm);
}

inline const Matrix<int> &oneintmatrix(int numRows, int numCols, retMatrix<int> &tmpm)
{
    return oneintmatrixbasic()(0,1,numRows-1,0,1,numCols-1,tmpm);
}

inline const Matrix<double> &zerodoublematrix(int numRows, int numCols, retMatrix<double> &tmpm)
{
    return zerodoublematrixbasic()(0,1,numRows-1,0,1,numCols-1,tmpm);
}

inline const Matrix<double> &onedoublematrix(int numRows, int numCols, retMatrix<double> &tmpm)
{
    return onedoublematrixbasic()(0,1,numRows-1,0,1,numCols-1,tmpm);
}

















/*
inline const Matrix<int> &cntintmatrix(int numRows, int numCols, retMatrix<int> &tmpm)
{
    static thread_local bool firstcall = true;
    static thread_local Matrix<int> cntres("&");

    NiceAssert( numRows >= 0 );
    NiceAssert( numCols >= 0 );

    if ( firstcall || ( numRows > cntres.dnumRows ) || ( numCols > cntres.dnumCols ) )
    {
        static thread_local Vector<Vector<int> > *ccontent = nullptr;
        static thread_local retVector<int> zvecstore;

        int locnumRows = ( firstcall || ( numRows > cntres.dnumRows ) ) ? numRows+MAXHACKYHEAD : cntres.dnumRows;
        int locnumCols = numCols;

        if ( firstcall )
        {
            cntres.bkref = &cntres;

            cntres.nbase    = true;
            cntres.pbaseRow = true;
            cntres.pbaseCol = true;

            cntres.iibRow = 0;
            cntres.iisRow = 0;

            cntres.iibCol = 0;
            cntres.iisCol = 1;

            MEMNEW(ccontent,Vector<Vector<int> >(1));

            (*ccontent)("&",0).assignover(cntintvec(locnumCols,zvecstore));
            (*ccontent)("&",0).dsize = ( locnumCols = (*((*ccontent)(0)).ccontent).array_size() );

            cntres.content  = nullptr;
            cntres.ccontent = ccontent;

            cntres.iscover = false;
            cntres.elmfn   = nullptr;
            cntres.rowfn   = nullptr;
            cntres.delfn   = nullptr;
            cntres.dref    = nullptr;
            cntres.celmfn  = nullptr;
            cntres.celmfn_v = nullptr_v;
            cntres.crowfn  = nullptr;
            cntres.cdelfn  = nullptr;
            cntres.cdref   = nullptr;

            cntres.pivotRow = cntintarray(locnumRows);
            cntres.pivotCol = cntintarray(locnumCols);

            cntres.dnumRows = locnumRows;
            cntres.dnumCols = locnumCols;

            firstcall = false;
        }

        else
        {
            if ( locnumCols >= cntres.dnumCols )
            {
                (*ccontent)("&",0).assignover(cntintvec(locnumCols,zvecstore));
                (*ccontent)("&",0).dsize = ( locnumCols = (*((*ccontent)(0)).ccontent).array_size() );
            }

            cntres.dnumRows = locnumRows;
            cntres.dnumCols = locnumCols;
        }
    }

    const Matrix<int> &res = cntres(0,1,numRows-1,0,1,numCols-1,tmpm);

    return res;
}

inline const Matrix<int> &deltaintmatrix(int pos, int numRows, int numCols, retMatrix<int> &tmpm)
{
    static thread_local bool firstcall = true;
    static thread_local Matrix<int> deltares("&");

    NiceAssert( numRows >= 0 );
    NiceAssert( numCols >= 0 );

    if ( firstcall || ( numRows > deltares.dnumRows ) || ( numCols > deltares.dnumCols ) )
    {
        static thread_local Vector<Vector<int> > *ccontent = nullptr;
        static thread_local retVector<int> zvecstore;

        int locnumRows = ( firstcall || ( numRows > deltares.dnumRows ) ) ? numRows+MAXHACKYHEAD : deltares.dnumRows;
        int locnumCols = numCols;

        if ( firstcall )
        {
            deltares.bkref = &deltares;

            deltares.nbase    = true;
            deltares.pbaseRow = true;
            deltares.pbaseCol = true;

            deltares.iibRow = 0;
            deltares.iisRow = 0;

            deltares.iibCol = 0;
            deltares.iisCol = 1;

            MEMNEW(ccontent,Vector<Vector<int> >(1));

            (*ccontent)("&",0).assignover(deltaintvec(pos,locnumCols,zvecstore));
            (*ccontent)("&",0).dsize = ( locnumCols = (*((*ccontent)(0)).ccontent).array_size() );

            deltares.content  = nullptr;
            deltares.ccontent = ccontent;

            deltares.iscover = false;
            deltares.elmfn   = nullptr;
            deltares.rowfn   = nullptr;
            deltares.delfn   = nullptr;
            deltares.dref    = nullptr;
            deltares.celmfn  = nullptr;
            deltares.celmfn_v = nullptr;
            deltares.crowfn  = nullptr;
            deltares.cdelfn  = nullptr;
            deltares.cdref   = nullptr;

            deltares.pivotRow = cntintarray(locnumRows);
            deltares.pivotCol = cntintarray(locnumCols);

            deltares.dnumRows = locnumRows;
            deltares.dnumCols = locnumCols;

            firstcall = false;
        }

        else
        {
            if ( locnumCols >= deltares.dnumCols )
            {
                (*ccontent)("&",0).assignover(deltaintvec(pos,locnumCols,zvecstore));
                (*ccontent)("&",0).dsize = ( locnumCols = (*((*ccontent)(0)).ccontent).array_size() );
            }

            deltares.dnumRows = locnumRows;
            deltares.dnumCols = locnumCols;
        }
    }

    const Matrix<int> &res = deltares(0,1,numRows-1,0,1,numCols-1,tmpm);

    return res;
}

inline const Matrix<int> &identintmatrix(int numRows, int numCols, retMatrix<int> &tmpm)
{
    int pos = numCols-1; // nominal, for size calculations (cause worst-case calculation)

    // This could be large, so we don't use thread_local to (a) save memory and (b) avoid
    // "churn" when a new thread first comes here.

    static thread_local bool firstcall = true;
    static thread_local Matrix<int> identres("&");

    //FIXME: currently ccontent and zvecstore are never deleted, need to fix this.

    NiceAssert( numRows >= 0 );
    NiceAssert( numCols >= 0 );

    if ( firstcall || ( numRows > identres.dnumRows ) || ( numCols > identres.dnumCols ) )
    {
        static thread_local Vector<Vector<int> > *ccontent = nullptr;
        static thread_local Vector<retVector<int> > *zvecstore = nullptr;
        static thread_local retVector<int> zzvecst;

        if ( firstcall || ( numRows > identres.dnumRows ) || ( numCols > identres.dnumCols ) )
        {
            int locnumRows = ( firstcall || ( numRows > identres.dnumRows ) ) ? numRows+MAXHACKYHEAD : identres.dnumRows;
            int locnumCols = (*((deltaintvec(pos,numCols,zzvecst)).ccontent)).array_size();

            if ( firstcall )
            {
                identres.bkref = &identres;

                identres.nbase    = true;
                identres.pbaseRow = true;
                identres.pbaseCol = true;

                identres.iibRow = 0;
                identres.iisRow = 1; // We need 1 here as each row is distinct!

                identres.iibCol = 0;
                identres.iisCol = 1;

                MEMNEW(ccontent,Vector<Vector<int> >(locnumRows));
                MEMNEW(zvecstore,Vector<retVector<int> >(locnumRows));

                for ( int i = 0 ; i < locnumRows ; ++i )
                {
                    (*ccontent)("&",i).assignover(deltaintvec(i,locnumCols,(*zvecstore)("&",i))); // the previous call to deltaintvec ensures that this doesn't change any sizes
                    (*ccontent)("&",i).dsize = locnumCols;
                }

                identres.content  = nullptr;
                identres.ccontent = ccontent; // purely ccontent, so not deleted by vector destructor

                identres.iscover = false;
                identres.elmfn   = nullptr;
                identres.rowfn   = nullptr;
                identres.delfn   = nullptr;
                identres.dref    = nullptr;
                identres.celmfn  = nullptr;
                identres.celmfn_v = nullptr;
                identres.crowfn  = nullptr;
                identres.cdelfn  = nullptr;
                identres.cdref   = nullptr;

                identres.pivotRow = cntintarray(locnumRows);
                identres.pivotCol = cntintarray(locnumCols);

                identres.dnumRows = locnumRows;
                identres.dnumCols = locnumCols;

                firstcall = false;
            }

            else
            {
                if ( locnumCols >= identres.dnumCols )
                {
                    MEMNEW(ccontent,Vector<Vector<int> >(locnumRows));
                    MEMNEW(zvecstore,Vector<retVector<int> >(locnumRows));

                    for ( int i = 0 ; i < locnumRows ; ++i )
                    {
                        (*ccontent)("&",i).assignover(deltaintvec(i,locnumCols,(*zvecstore)("&",i)));
                        (*ccontent)("&",i).dsize = locnumCols;
                    }

                    identres.ccontent = ccontent;
                }

                identres.dnumRows = locnumRows;
                identres.dnumCols = locnumCols;
            }
        }
    }

    const Matrix<int> &res = identres(0,1,numRows-1,0,1,numCols-1,tmpm);

    return res;
}
*/















/*
inline const Matrix<double> &cntdoublematrix(int numRows, int numCols, retMatrix<double> &tmpm)
{
    static thread_local bool firstcall = true;
    static thread_local Matrix<double> cntres("&");

    NiceAssert( numRows >= 0 );
    NiceAssert( numCols >= 0 );

    if ( firstcall || ( numRows > cntres.dnumRows ) || ( numCols > cntres.dnumCols ) )
    {
        static thread_local Vector<Vector<double> > *ccontent = nullptr;
        static thread_local retVector<double> zvecstore;

        int locnumRows = ( firstcall || ( numRows > cntres.dnumRows ) ) ? numRows+MAXHACKYHEAD : cntres.dnumRows;
        int locnumCols = numCols;

        if ( firstcall )
        {
            cntres.bkref = &cntres;

            cntres.nbase    = true;
            cntres.pbaseRow = true;
            cntres.pbaseCol = true;

            cntres.iibRow = 0;
            cntres.iisRow = 0;

            cntres.iibCol = 0;
            cntres.iisCol = 1;

            MEMNEW(ccontent,Vector<Vector<double> >(1));

            (*ccontent)("&",0).assignover(cntdoublevec(locnumCols,zvecstore));
            (*ccontent)("&",0).dsize = ( locnumCols = (*((*ccontent)(0)).ccontent).array_size() );

            cntres.content  = nullptr;
            cntres.ccontent = ccontent;

            cntres.iscover = false;
            cntres.elmfn   = nullptr;
            cntres.rowfn   = nullptr;
            cntres.delfn   = nullptr;
            cntres.dref    = nullptr;
            cntres.celmfn  = nullptr;
            cntres.celmfn_v = nullptr;
            cntres.crowfn  = nullptr;
            cntres.cdelfn  = nullptr;
            cntres.cdref   = nullptr;

            cntres.pivotRow = cntintarray(locnumRows);
            cntres.pivotCol = cntintarray(locnumCols);

            cntres.dnumRows = locnumRows;
            cntres.dnumCols = locnumCols;

            firstcall = false;
        }

        else
        {
            if ( locnumCols >= cntres.dnumCols )
            {
                (*ccontent)("&",0).assignover(cntdoublevec(locnumCols,zvecstore));
                (*ccontent)("&",0).dsize = ( locnumCols = (*((*ccontent)(0)).ccontent).array_size() );
            }

            cntres.dnumRows = locnumRows;
            cntres.dnumCols = locnumCols;
        }
    }

    const Matrix<double> &res = cntres(0,1,numRows-1,0,1,numCols-1,tmpm);

    return res;
}

inline const Matrix<double> &deltadoublematrix(int pos, int numRows, int numCols, retMatrix<double> &tmpm)
{
    static thread_local bool firstcall = true;
    static thread_local Matrix<double> deltares("&");

    NiceAssert( numRows >= 0 );
    NiceAssert( numCols >= 0 );

    if ( firstcall || ( numRows > deltares.dnumRows ) || ( numCols > deltares.dnumCols ) )
    {
        static thread_local Vector<Vector<double> > *ccontent = nullptr;
        static thread_local retVector<double> zvecstore;

        int locnumRows = ( firstcall || ( numRows > deltares.dnumRows ) ) ? numRows+MAXHACKYHEAD : deltares.dnumRows;
        int locnumCols = numCols;

        if ( firstcall )
        {
            deltares.bkref = &deltares;

            deltares.nbase    = true;
            deltares.pbaseRow = true;
            deltares.pbaseCol = true;

            deltares.iibRow = 0;
            deltares.iisRow = 0;

            deltares.iibCol = 0;
            deltares.iisCol = 1;

            MEMNEW(ccontent,Vector<Vector<double> >(1));

            (*ccontent)("&",0).assignover(deltadoublevec(pos,locnumCols,zvecstore));
            (*ccontent)("&",0).dsize = ( locnumCols = (*((*ccontent)(0)).ccontent).array_size() );

            deltares.content  = nullptr;
            deltares.ccontent = ccontent;

            deltares.iscover = false;
            deltares.elmfn   = nullptr;
            deltares.rowfn   = nullptr;
            deltares.delfn   = nullptr;
            deltares.dref    = nullptr;
            deltares.celmfn  = nullptr;
            deltares.celmfn_v = nullptr;
            deltares.crowfn  = nullptr;
            deltares.cdelfn  = nullptr;
            deltares.cdref   = nullptr;

            deltares.pivotRow = cntintarray(locnumRows);
            deltares.pivotCol = cntintarray(locnumCols);

            deltares.dnumRows = locnumRows;
            deltares.dnumCols = locnumCols;

            firstcall = false;
        }

        else
        {
            if ( locnumCols >= deltares.dnumCols )
            {
                (*ccontent)("&",0).assignover(deltadoublevec(pos,locnumCols,zvecstore));
                (*ccontent)("&",0).dsize = ( locnumCols = (*((*ccontent)(0)).ccontent).array_size() );
            }

            deltares.dnumRows = locnumRows;
            deltares.dnumCols = locnumCols;
        }
    }

    const Matrix<double> &res = deltares(0,1,numRows-1,0,1,numCols-1,tmpm);

    return res;
}

inline const Matrix<double> &identdoublematrix(int numRows, int numCols, retMatrix<double> &tmpm)
{
    int pos = numCols-1; // nominal, for size calculations (cause worst-case calculation)

    static thread_local bool firstcall = true;
    static thread_local Matrix<double> identres("&");

    //FIXME: currently ccontent and zvecstore are never deleted, need to fix this.

    NiceAssert( numRows >= 0 );
    NiceAssert( numCols >= 0 );

    if ( firstcall || ( numRows > identres.dnumRows ) || ( numCols > identres.dnumCols ) )
    {
        static thread_local Vector<Vector<double> > *ccontent = nullptr;
        static thread_local Vector<retVector<double> > *zvecstore = nullptr;
        static thread_local retVector<double> zzvecst;

        if ( firstcall || ( numRows > identres.dnumRows ) || ( numCols > identres.dnumCols ) )
        {
            int locnumRows = ( firstcall || ( numRows > identres.dnumRows ) ) ? numRows+MAXHACKYHEAD : identres.dnumRows;
            int locnumCols = (*((deltadoublevec(pos,numCols,zzvecst)).ccontent)).array_size();

            if ( firstcall )
            {
                identres.bkref = &identres;

                identres.nbase    = true;
                identres.pbaseRow = true;
                identres.pbaseCol = true;

                identres.iibRow = 0;
                identres.iisRow = 1; // We need 1 here as each row is distinct!

                identres.iibCol = 0;
                identres.iisCol = 1;

                MEMNEW(ccontent,Vector<Vector<double> >(locnumRows));
                MEMNEW(zvecstore,Vector<retVector<double> >(locnumRows));

                for ( int i = 0 ; i < locnumRows ; ++i )
                {
                    (*ccontent)("&",i).assignover(deltadoublevec(i,locnumCols,(*zvecstore)("&",i))); // the previous call to deltadoublevec ensures that this doesn't change any sizes
                    (*ccontent)("&",i).dsize = locnumCols;
                }

                identres.content  = nullptr;
                identres.ccontent = ccontent; // purely ccontent, so not deleted by vector destructor

                identres.iscover = false;
                identres.elmfn   = nullptr;
                identres.rowfn   = nullptr;
                identres.delfn   = nullptr;
                identres.dref    = nullptr;
                identres.celmfn  = nullptr;
                identres.celmfn_v = nullptr;
                identres.crowfn  = nullptr;
                identres.cdelfn  = nullptr;
                identres.cdref   = nullptr;

                identres.pivotRow = cntintarray(locnumRows);
                identres.pivotCol = cntintarray(locnumCols);

                identres.dnumRows = locnumRows;
                identres.dnumCols = locnumCols;

                firstcall = false;
            }

            else
            {
                if ( locnumCols >= identres.dnumCols )
                {
                    MEMNEW(ccontent,Vector<Vector<double> >(locnumRows));
                    MEMNEW(zvecstore,Vector<retVector<double> >(locnumRows));

                    for ( int i = 0 ; i < locnumRows ; ++i )
                    {
                        (*ccontent)("&",i).assignover(deltadoublevec(i,locnumCols,(*zvecstore)("&",i)));
                        (*ccontent)("&",i).dsize = locnumCols;
                    }

                    identres.ccontent = ccontent;
                }

                identres.dnumRows = locnumRows;
                identres.dnumCols = locnumCols;
            }
        }
    }

    const Matrix<double> &res = identres(0,1,numRows-1,0,1,numCols-1,tmpm);

    return res;
}
*/
































// Add and remove element functions

template <class T>
Matrix<T> &Matrix<T>::addRow(int i)
{
    NiceAssert( !nbase );
    NiceAssert( i >= 0 );
    NiceAssert( i <= dnumRows );

    ++dnumRows;

    if ( !iscover )
    {
        (*content).add(i);
//	(*content)("&",i).fixsize = 0;
        (*content)("&",i).resize(dnumCols);
    }

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::addCol(int i)
{
    NiceAssert( !nbase );
    NiceAssert( i >= 0 );
    NiceAssert( i <= dnumCols );

    ++dnumCols;

    if ( !iscover )
    {
	//if ( dnumRows )
	{
	    for ( int j = 0 ; j < dnumRows ; ++j )
	    {
//		((*content)("&",j)).fixsize = 0;
		((*content)("&",j)).add(i);
	    }
	}
    }

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::padCol(int n)
{
    NiceAssert( !nbase );

    if ( !iscover )
    {
	//if ( dnumRows )
	{
	    for ( int j = 0 ; j < dnumRows ; ++j )
	    {
                ((*content)("&",j)).pad(n);
	    }
	}
    }

    dnumCols += n;

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::padRow(int n)
{
    NiceAssert( !nbase );

    int i,j;

    while ( n )
    {
        i = numRows();

        addRow(i);

        //if ( numCols() )
        {
            for ( j = 0 ; j < numCols() ; ++j )
            {
                setzero((*this)("&",i,j));
            }
        }

        --n;
    }

    return *this;
}

template <class T> 
Matrix<T> &Matrix<T>::padRowCol(int n) 
{ 
    padRow(n);
    padCol(n);

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::addRowCol(int i)
{
    NiceAssert( !nbase );
    NiceAssert( i >= 0 );
    NiceAssert( i <= dnumRows );
    NiceAssert( i >= 0 );
    NiceAssert( i <= dnumCols );

    // Speed choice: addCol takes o(numRows) operations to complete, addRow
    // takes o(1) operations (both leverage qswap for speed).  So by calling
    // addCol first things are marginally faster.

    addCol(i);
    addRow(i);

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::removeRow(int i)
{
    NiceAssert( !nbase );
    NiceAssert( i >= 0 );
    NiceAssert( i < dnumRows );

    --dnumRows;

    if ( !iscover )
    {
//	((*content)("&",i)).fixsize = 0;
	//FIXME: not sure if this is a good idea or not (speed vs memory)
        //       ((*content)("&",i)).resize(0);
	//       speed: leave the vector as it is, even though it may be big
        //              and may remain in memory for a little bit
	//       memory: it may remain in memory after removal, so downsize
        //               *right* *now*
	//DECISION: speed >> memory
	(*content).remove(i);
    }

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::removeCol(int i)
{
    NiceAssert( !nbase );
    NiceAssert( i >= 0 );
    NiceAssert( i < dnumCols );

    --dnumCols;

    if ( !iscover )
    {
	//if ( dnumRows )
	{
	    for ( int j = 0 ; j < dnumRows ; ++j )
	    {
//		((*content)("&",j)).fixsize = 0;
		((*content)("&",j)).remove(i);
	    }
	}
    }

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::removeRowCol(int i)
{
    NiceAssert( !nbase );
    NiceAssert( i >= 0 );
    NiceAssert( i < dnumRows );
    NiceAssert( i >= 0 );
    NiceAssert( i < dnumCols );

    // Speed choice: run the addRowCol argument in reverse.

    removeRow(i);
    removeCol(i);

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::appendRow(int rowStart, const Matrix<T> &src)
{
    NiceAssert( !nbase );
    NiceAssert( rowStart >= 0 );
    NiceAssert( rowStart <= dnumRows );
    NiceAssert( src.numCols() == numCols() );

    dnumRows += src.numRows();

    if ( !iscover && src.numRows() )
    {
        retVector<T> tmpva;

        for ( int i = rowStart ; i < rowStart+src.numRows() ; ++i )
        {
            (*content).add(i);

//          (*content)("&",i).fixsize = 0;
            (*content).set(i,src(i-rowStart,tmpva));
        }
    }

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::appendCol(int colStart, const Matrix<T> &src)
{
    NiceAssert( !nbase );
    NiceAssert( colStart >= 0 );
    NiceAssert( colStart <= dnumCols );
    NiceAssert( src.numRows() == numRows() );

    dnumCols += src.numCols();

    if ( !iscover && src.numCols() && dnumRows )
    {
        int i,j;

        for ( i = 0 ; i < dnumRows ; ++i )
        {
            for ( j = colStart ; j < colStart+src.numCols() ; ++j )
            {
//              ((*content)("&",i)).fixsize = 0;
                ((*content)("&",i)).add(j);
                ((*content)("&",i))("&",j) = src(i,j-colStart);
	    }
	}
    }

    return *this;
}


template <class T>
Matrix<T> &Matrix<T>::resize(int targNumRows, int targNumCols)
{
    int i;

    if ( !iscover )
    {
        (*content).resize(targNumRows);

        if ( ( dnumRows < targNumRows ) && ( dnumCols == targNumCols ) )
        {
            for ( i = dnumRows ; i < targNumRows ; ++i )
            {
                (*content)("&",i).resize(targNumCols);
            }
        }

        dnumRows = targNumRows;

        if ( dnumRows && ( targNumCols != dnumCols ) )
        {
            for ( i = 0 ; i < dnumRows ; ++i )
            {
                (*content)("&",i).resize(targNumCols);
            }
        }

        if ( (*content).array_slack() )
        {
            useSlackAllocation();
        }

        if ( (*content).array_tight() )
        {
            useTightAllocation();
        }
    }

    dnumRows = targNumRows;
    dnumCols = targNumCols;

    return *this;
}


// Function application

template <class T>
Matrix<T> &Matrix<T>::applyon(T (*fn)(T))
{
    int i;

    if ( dnumRows && dnumCols )
    {
        retVector<T> tmpva;

	for ( i = 0 ; i < dnumRows ; ++i )
	{
            (*this)("&",i,"&",tmpva).applyon(fn);
	}
    }

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::applyon(T (*fn)(const T &))
{
    int i;

    if ( dnumRows && dnumCols )
    {
        retVector<T> tmpva;
        retVector<T> tmpvb;

	for ( i = 0 ; i < dnumRows ; ++i )
	{
            (*this)("&",i,"&",tmpva,tmpvb).applyon(fn);
	}
    }

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::applyon(T &(*fn)(T &))
{
    if ( dnumRows && dnumCols )
    {
        int i;

        retVector<T> tmpva;
        retVector<T> tmpvc;

	for ( i = 0 ; i < dnumRows ; ++i )
	{
            (*this)("&",i,"&",tmpva,tmpvc).applyon(fn);
	}
    }

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::applyon(T (*fn)(T, const void *), const void *a)
{
    int i;

    if ( dnumRows && dnumCols )
    {
        retVector<T> tmpva;
        retVector<T> tmpvc;

	for ( i = 0 ; i < dnumRows ; ++i )
	{
            (*this)("&",i,"&",tmpva,tmpvc).applyon(fn,a);
	}
    }

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::applyon(T (*fn)(const T &, const void *), const void *a)
{
    int i;

    if ( dnumRows && dnumCols )
    {
        retVector<T> tmpva;
        retVector<T> tmpvc;

	for ( i = 0 ; i < dnumRows ; ++i )
	{
            (*this)("&",i,"&",tmpva,tmpvc).applyon(fn,a);
	}
    }

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::applyon(T &(*fn)(T &, const void *), const void *a)
{
    int i;

    if ( dnumRows && dnumCols )
    {
        retVector<T> tmpva;
        retVector<T> tmpvc;

	for ( i = 0 ; i < dnumRows ; ++i )
	{
            (*this)("&",i,"&",tmpva,tmpvc).applyon(fn,a);
	}
    }

    return *this;
}


// Other functions

template <class T>
Matrix<T> &Matrix<T>::rankone(const T &c, const Vector<T> &a, const Vector<T> &b)
{
    NiceAssert( dnumRows == a.size() );
    NiceAssert( dnumCols == b.size() );

    int i;

    Vector<T> bb(b);
    bb.conj();

    if ( dnumRows && dnumCols )
    {
        retVector<T> tmpva;
        retVector<T> tmpvc;

	for ( i = 0 ; i < dnumRows ; ++i )
	{
            (*this)("&",i,"&",tmpva,tmpvc).scaleAdd(a(i)*c,bb);
	}
    }

    return *this;
}

template <> inline Matrix<double> &Matrix<double>::rankone(const double &c, const Vector<double> &a, const Vector<double> &b);
template <> inline Matrix<double> &Matrix<double>::rankone(const double &c, const Vector<double> &a, const Vector<double> &b)
{
    NiceAssert( dnumRows == a.size() );
    NiceAssert( dnumCols == b.size() );

    int i;

    if ( dnumRows && dnumCols )
    {
        retVector<double> tmpva;
        retVector<double> tmpvb;

	for ( i = 0 ; i < dnumRows ; ++i )
	{
            (*this)("&",i,"&",tmpva,tmpvb).scaleAdd(a.v(i)*c,b);
	}
    }

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::diagoffset(const Vector<T> &d)
{
    int minsize = ( dnumRows < dnumCols ) ? dnumRows : dnumCols;

    NiceAssert( minsize == d.size() );

    //if ( minsize )
    {
	for ( int i = 0 ; i < minsize ; ++i )
	{
            (*this)("&",i,i) += d(i);
	}
    }

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::diagoffset(const T &c, const Vector<T> &d)
{
    int minsize = ( dnumRows < dnumCols ) ? dnumRows : dnumCols;

    NiceAssert( minsize == d.size() );

    //if ( minsize )
    {
	for ( int i = 0 ; i < minsize ; ++i )
	{
            (*this)("&",i,i) += c*d(i);
	}
    }

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::diagoffset(const Matrix<T> &d)
{
    int minsize = ( dnumRows < dnumCols ) ? dnumRows : dnumCols;

    NiceAssert( minsize == ( ( d.numRows() < d.numCols() ) ? d.numRows() : d.numCols() ) );

    //if ( minsize )
    {
	for ( int i = 0 ; i < minsize ; ++i )
	{
            (*this)("&",i,i) += d(i,i);
	}
    }

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::diagoffset(const T &d)
{
    int minsize = ( dnumRows < dnumCols ) ? dnumRows : dnumCols;

    //if ( minsize )
    {
	for ( int i = 0 ; i < minsize ; ++i )
	{
            (*this)("&",i,i) += d;
	}
    }

    return *this;
}













template <class T>
Matrix<T> &Matrix<T>::sqdiagoffset(const Vector<T> &d)
{
    int minsize = ( dnumRows < dnumCols ) ? dnumRows : dnumCols;

    NiceAssert( minsize == d.size() );

    //if ( minsize )
    {
	for ( int i = 0 ; i < minsize ; ++i )
	{
            (*this)("&",i,i) += conj(d(i))*d(i);
	}
    }

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::sqdiagoffset(const T &c, const Vector<T> &d)
{
    int minsize = ( dnumRows < dnumCols ) ? dnumRows : dnumCols;

    NiceAssert( minsize == d.size() );

    //if ( minsize )
    {
	for ( int i = 0 ; i < minsize ; ++i )
	{
            (*this)("&",i,i) += conj(c*d(i))*conj(c*d(i));
	}
    }

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::sqdiagoffset(const Matrix<T> &d)
{
    int minsize = ( dnumRows < dnumCols ) ? dnumRows : dnumCols;

    NiceAssert( minsize == ( ( d.numRows() < d.numCols() ) ? d.numRows() : d.numCols() ) );

    //if ( minsize )
    {
	for ( int i = 0 ; i < minsize ; ++i )
	{
            (*this)("&",i,i) += conj(d(i,i))*d(i,i);
	}
    }

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::sqdiagoffset(const T &d)
{
    int minsize = ( dnumRows < dnumCols ) ? dnumRows : dnumCols;

    //if ( minsize )
    {
	for ( int i = 0 ; i < minsize ; ++i )
	{
            (*this)("&",i,i) += conj(d)*d;
	}
    }

    return *this;
}










template <> inline Matrix<double> &Matrix<double>::sqdiagoffset(const Vector<double> &d);
template <> inline Matrix<double> &Matrix<double>::sqdiagoffset(const Vector<double> &d)
{
    int minsize = ( dnumRows < dnumCols ) ? dnumRows : dnumCols;

    NiceAssert( minsize == d.size() );

    for ( int i = 0 ; i < minsize ; ++i )
    {
        (*this)("&",i,i) += d.v(i)*d.v(i);
    }

    return *this;
}

template <> inline Matrix<double> &Matrix<double>::sqdiagoffset(const double &c, const Vector<double> &d);
template <> inline Matrix<double> &Matrix<double>::sqdiagoffset(const double &c, const Vector<double> &d)
{
    int minsize = ( dnumRows < dnumCols ) ? dnumRows : dnumCols;

    NiceAssert( minsize == d.size() );

    for ( int i = 0 ; i < minsize ; ++i )
    {
        (*this)("&",i,i) += c*d.v(i)*c*d.v(i);
    }

    return *this;
}

template <> inline Matrix<double> &Matrix<double>::sqdiagoffset(const Matrix<double> &d);
template <> inline Matrix<double> &Matrix<double>::sqdiagoffset(const Matrix<double> &d)
{
    int minsize = ( dnumRows < dnumCols ) ? dnumRows : dnumCols;

    NiceAssert( minsize == ( ( d.numRows() < d.numCols() ) ? d.numRows() : d.numCols() ) );

    for ( int i = 0 ; i < minsize ; ++i )
    {
        (*this)("&",i,i) += d.v(i,i)*d.v(i,i);
    }

    return *this;
}

template <> inline Matrix<double> &Matrix<double>::sqdiagoffset(const double &d);
template <> inline Matrix<double> &Matrix<double>::sqdiagoffset(const double &d)
{
    int minsize = ( dnumRows < dnumCols ) ? dnumRows : dnumCols;

    for ( int i = 0 ; i < minsize ; ++i )
    {
        (*this)("&",i,i) += d*d;
    }

    return *this;
}








template <class T>
Matrix<T> &Matrix<T>::naiveChol(Matrix<T> &dest, int zeroupper, double ztol) const
{
    NiceAssert( isSquare() );
    NiceAssert( dnumRows == dest.numRows() );
    NiceAssert( dnumCols == dest.numCols() );

    if ( dnumRows )
    {
	int j,k;

	if ( zeroupper )
	{
            dest.zero();
	}

	for ( int ii = 0, i = 0 ; i < dnumRows ; ++i )
	{
	    dest("&",i,i) = (*this)(i,i);

	    //if ( i > 0 )
	    {
		for ( k = 0 ; k < i ; ++k )
		{
		    dest("&",i,i) -= dest(i,k)*conj(dest(i,k));
		}
	    }

            if ( (double) dest(i,i) < ztol )
            {
                dest("&",i,i) = ztol;
                ztol *= exp(pow(0.5,ii+1)-pow(0.5,ii));
                ii++;
            }

	    dest("&",i,i) = sqrt(dest(i,i));

	    //if ( i < dnumRows-1 )
	    {
		for ( j = i+1 ; j < dnumRows ; ++j )
		{
		    dest("&",j,i) = (*this)(j,i);

		    //if ( i > 0 )
		    {
			for ( k = 0 ; k < i ; ++k )
			{
			    dest("&",j,i) -= dest(j,k)*conj(dest(i,k));
			}
		    }

		    dest("&",j,i) /= real(dest(i,i));
		}
	    }
	}
    }

    return dest;
}

template <class T>
Matrix<T> &Matrix<T>::naiveChol(const Vector<T> &diagsign, Matrix<T> &dest, int zeroupper, double ztol) const
{
    NiceAssert( isSquare() );
    NiceAssert( dnumRows == dest.numRows() );
    NiceAssert( dnumCols == dest.numCols() );
    NiceAssert( dnumRows == diagsign.size() );

    if ( dnumRows )
    {
	int j,k;

	if ( zeroupper )
	{
            dest.zero();
	}

	for ( int ii = 0, i = 0 ; i < dnumRows ; ++i )
	{
	    dest("&",i,i) = (*this)(i,i);

	    //if ( i > 0 )
	    {
		for ( k = 0 ; k < i ; ++k )
		{
		    dest("&",i,i) -= dest(i,k)*diagsign(k)*conj(dest(i,k));
		}
	    }

            dest("&",i,i) *= diagsign(i);

            if ( (double) dest(i,i) < ztol )
            {
                dest("&",i,i) = ztol;
                ztol *= exp(pow(0.5,ii+1)-pow(0.5,ii));
                ii++;
            }

	    dest("&",i,i) = sqrt(dest(i,i));

	    //if ( i < dnumRows-1 )
	    {
		for ( j = i+1 ; j < dnumRows ; ++j )
		{
		    dest("&",j,i) = (*this)(j,i);

		    //if ( i > 0 )
		    {
			for ( k = 0 ; k < i ; ++k )
			{
			    dest("&",j,i) -= dest(j,k)*diagsign(k)*conj(dest(i,k));
			}
		    }

                    dest("&",j,i) *= diagsign(i);
		    dest("&",j,i) /= real(dest(i,i));
		}
	    }
	}
    }

    return dest;
}

template <class T>
Matrix<T> &Matrix<T>::naiveChol(Matrix<T> &dest, double diagoffscal, const Vector<T> &diagoffset, int zeroupper, double ztol) const
{
    NiceAssert( isSquare() );
    NiceAssert( dnumRows == dest.numRows() );
    NiceAssert( dnumCols == dest.numCols() );
    NiceAssert( dnumRows == diagoffset.size() );

    if ( dnumRows )
    {
	int j,k;

	if ( zeroupper )
	{
            dest.zero();
	}

	for ( int ii = 0, i = 0 ; i < dnumRows ; ++i )
	{
	    dest("&",i,i) = (*this)(i,i);
	    dest("&",i,i) += diagoffscal*diagoffset(i);

	    //if ( i > 0 )
	    {
		for ( k = 0 ; k < i ; ++k )
		{
		    dest("&",i,i) -= dest(i,k)*conj(dest(i,k));
		}
	    }

            if ( (double) dest(i,i) < ztol )
            {
                dest("&",i,i) = ztol;
                ztol *= exp(pow(0.5,ii+1)-pow(0.5,ii));
                ii++;
            }

	    dest("&",i,i) = sqrt(dest(i,i));

	    //if ( i < dnumRows-1 )
	    {
		for ( j = i+1 ; j < dnumRows ; ++j )
		{
		    dest("&",j,i) = (*this)(j,i);

		    //if ( i > 0 )
		    {
			for ( k = 0 ; k < i ; ++k )
			{
			    dest("&",j,i) -= dest(j,k)*conj(dest(i,k));
			}
		    }

		    dest("&",j,i) /= real(dest(i,i));
		}
	    }
	}
    }

    return dest;
}

template <class T>
Matrix<T> &Matrix<T>::naiveChol(const Vector<T> &diagsign, Matrix<T> &dest, double diagoffscal, const Vector<T> &diagoffset, int zeroupper, double ztol) const
{
    NiceAssert( isSquare() );
    NiceAssert( dnumRows == dest.numRows() );
    NiceAssert( dnumCols == dest.numCols() );
    NiceAssert( dnumRows == diagoffset.size() );
    NiceAssert( dnumRows == diagsign.size() );

    if ( dnumRows )
    {
	int j,k;

	if ( zeroupper )
	{
            dest.zero();
	}

        // [ la     ] [ sa     ] [ la  lb' ] = [ ga  gb' ]
        // [ lb  Lc ] [     Sc ] [     Lc' ]   [ gb  Gc  ]
        //
        // [ la.sa        ] [ la  lb' ] = [ ga  gb' ]
        // [ lb.sa  Lc.Sc ] [     Lc' ]   [ gb  Gc  ]
        //
        // [ la.sa.la   la.sa.lb'             ] = [ ga  gb' ]
        // [ lb.sa.la   lb.sa.lb' + Lc.Sc.Lc' ]   [ gb  Gc' ]
        //
        // [ la.sa.la   la.sa.lb' ] = [ ga   gb'             ]
        // [ lb.sa.la   Lc.Sc.Lc' ]   [ gb   Gc' - lb.sa.lb' ]
        //
        // la = sqrt(sa.ga)
        // lb = gb/(sa.la)
        //
        // And repeat for: Lc.Sc.Lc' = Gc' - lb.sa.lb'

	for ( int ii = 0, i = 0 ; i < dnumRows ; ++i )
	{
	    dest("&",i,i) = (*this)(i,i);
	    dest("&",i,i) += diagsign(i)*diagoffscal*diagoffset(i);

	    //if ( i > 0 )
	    {
		for ( k = 0 ; k < i ; ++k )
		{
		    dest("&",i,i) -= dest(i,k)*diagsign(k)*conj(dest(i,k));
		}
	    }

            dest("&",i,i) *= diagsign(i);

            if ( (double) dest(i,i) < ztol )
            {
                dest("&",i,i) = ztol;
                ztol *= exp(pow(0.5,ii+1)-pow(0.5,ii));
                ii++;
            }

	    dest("&",i,i) = sqrt(dest(i,i));

	    //if ( i < dnumRows-1 )
	    {
		for ( j = i+1 ; j < dnumRows ; ++j )
		{
		    dest("&",j,i) = (*this)(j,i);

		    //if ( i > 0 )
		    {
			for ( k = 0 ; k < i ; ++k )
			{
			    dest("&",j,i) -= dest(j,k)*diagsign(k)*conj(dest(i,k));
			}
		    }

                    dest("&",j,i) *= diagsign(i);
		    dest("&",j,i) /= real(dest(i,i));
		}
	    }
	}
    }

    return dest;
}

template <> inline Matrix<double> &Matrix<double>::naiveChol(Matrix<double> &dest, int zeroupper, double ztol) const;
template <> inline Matrix<double> &Matrix<double>::naiveChol(Matrix<double> &dest, int zeroupper, double ztol) const
{
    NiceAssert( isSquare() );
    NiceAssert( dnumRows == dest.numRows() );
    NiceAssert( dnumCols == dest.numCols() );

    if ( dnumRows )
    {
	int j,k;

	if ( zeroupper )
	{
            dest.zero();
	}

	for ( int ii = 0, i = 0 ; i < dnumRows ; ++i )
	{
	    dest("&",i,i) = (*this).v(i,i);

            for ( k = 0 ; k < i ; ++k )
	    {
	        dest("&",i,i) -= dest.v(i,k)*dest.v(i,k);
	    }

            if ( dest(i,i) < ztol )
            {
                dest("&",i,i) = ztol;
                ztol *= exp(pow(0.5,ii+1)-pow(0.5,ii));
                ii++;
            }

	    dest("&",i,i) = sqrt(dest(i,i));

            for ( j = i+1 ; j < dnumRows ; ++j )
	    {
	        dest("&",j,i) = (*this).v(j,i);

		for ( k = 0 ; k < i ; ++k )
		{
		    dest("&",j,i) -= dest.v(j,k)*dest.v(i,k);
		}

	        dest("&",j,i) /= dest.v(i,i);
	    }
	}
    }

    return dest;
}

template <> inline Matrix<double> &Matrix<double>::naiveChol(Matrix<double> &dest, double diagoffscal, const Vector<double> &diagoffset, int zeroupper, double ztol) const;
template <> inline Matrix<double> &Matrix<double>::naiveChol(Matrix<double> &dest, double diagoffscal, const Vector<double> &diagoffset, int zeroupper, double ztol) const
{
    NiceAssert( isSquare() );
    NiceAssert( dnumRows == dest.numRows() );
    NiceAssert( dnumCols == dest.numCols() );
    NiceAssert( dnumRows == diagoffset.size() );

    if ( dnumRows )
    {
	int j,k;

	if ( zeroupper )
	{
            dest.zero();
	}

	for ( int ii = 0, i = 0 ; i < dnumRows ; ++i )
	{
	    dest("&",i,i) = (*this).v(i,i);
	    dest("&",i,i) += diagoffscal*diagoffset(i);

	    for ( k = 0 ; k < i ; ++k )
	    {
	        dest("&",i,i) -= dest.v(i,k)*dest.v(i,k);
	    }

            if ( dest(i,i) < ztol )
            {
                dest("&",i,i) = ztol;
                ztol *= exp(pow(0.5,ii+1)-pow(0.5,ii));
                ii++;
            }

	    dest("&",i,i) = sqrt(dest.v(i,i));

	    for ( j = i+1 ; j < dnumRows ; ++j )
	    {
	        dest("&",j,i) = (*this).v(j,i);

	        for ( k = 0 ; k < i ; ++k )
	        {
	            dest("&",j,i) -= dest.v(j,k)*dest.v(i,k);
	        }

	        dest("&",j,i) /= dest.v(i,i);
	    }
	}
    }

    return dest;
}

template <> inline Matrix<double> &Matrix<double>::naiveChol(const Vector<double> &diagsign, Matrix<double> &dest, int zeroupper, double ztol) const;
template <> inline Matrix<double> &Matrix<double>::naiveChol(const Vector<double> &diagsign, Matrix<double> &dest, int zeroupper, double ztol) const
{
    NiceAssert( isSquare() );
    NiceAssert( dnumRows == dest.numRows() );
    NiceAssert( dnumCols == dest.numCols() );
    NiceAssert( dnumRows == diagsign.size() );

    if ( dnumRows )
    {
	int j,k;

	if ( zeroupper )
	{
            dest.zero();
	}

	for ( int ii = 0, i = 0 ; i < dnumRows ; ++i )
	{
	    dest("&",i,i) = (*this).v(i,i);

	    for ( k = 0 ; k < i ; ++k )
	    {
	        dest("&",i,i) -= dest.v(i,k)*diagsign.v(k)*dest.v(i,k);
	    }

            dest("&",i,i) *= diagsign(i);

            if ( dest(i,i) < ztol )
            {
                dest("&",i,i) = ztol;
                ztol *= exp(pow(0.5,ii+1)-pow(0.5,ii));
                ii++;
            }

	    dest("&",i,i) = sqrt(dest(i,i));

	    for ( j = i+1 ; j < dnumRows ; ++j )
	    {
	        dest("&",j,i) = (*this).v(j,i);

	        for ( k = 0 ; k < i ; ++k )
	        {
	            dest("&",j,i) -= dest.v(j,k)*diagsign.v(k)*dest.v(i,k);
	        }

                dest("&",j,i) *= diagsign.v(i);
	        dest("&",j,i) /= dest.v(i,i);
	    }
	}
    }

    return dest;
}

template <> inline Matrix<double> &Matrix<double>::naiveChol(const Vector<double> &diagsign, Matrix<double> &dest, double diagoffscal, const Vector<double> &diagoffset, int zeroupper, double ztol) const;
template <> inline Matrix<double> &Matrix<double>::naiveChol(const Vector<double> &diagsign, Matrix<double> &dest, double diagoffscal, const Vector<double> &diagoffset, int zeroupper, double ztol) const
{
    NiceAssert( isSquare() );
    NiceAssert( dnumRows == dest.numRows() );
    NiceAssert( dnumCols == dest.numCols() );
    NiceAssert( dnumRows == diagoffset.size() );
    NiceAssert( dnumRows == diagsign.size() );

    if ( dnumRows )
    {
	int j,k;

	if ( zeroupper )
	{
            dest.zero();
	}

	for ( int ii = 0, i = 0 ; i < dnumRows ; ++i )
	{
	    dest("&",i,i) = (*this).v(i,i);
	    dest("&",i,i) += diagsign.v(i)*diagoffscal*diagoffset.v(i);

	    //if ( i > 0 )
	    {
		for ( k = 0 ; k < i ; ++k )
		{
		    dest("&",i,i) -= dest.v(i,k)*diagsign.v(k)*dest.v(i,k);
		}
	    }

            dest("&",i,i) *= diagsign.v(i);

            if ( dest(i,i) < ztol )
            {
                dest("&",i,i) = ztol;
                ztol *= exp(pow(0.5,ii+1)-pow(0.5,ii));
                ii++;
            }

	    dest("&",i,i) = sqrt(dest.v(i,i));

	    //if ( i < dnumRows-1 )
	    {
		for ( j = i+1 ; j < dnumRows ; ++j )
		{
		    dest("&",j,i) = (*this).v(j,i);

		    //if ( i > 0 )
		    {
			for ( k = 0 ; k < i ; ++k )
			{
			    dest("&",j,i) -= dest.v(j,k)*diagsign.v(k)*dest.v(i,k);
			}
		    }

                    dest("&",j,i) *= diagsign.v(i);
		    dest("&",j,i) /= dest.v(i,i);
		}
	    }
	}
    }

    return dest;
}

template <class T>
Matrix<T> &Matrix<T>::naivepartChol(Matrix<T> &dest, Vector<int> &p, int &n, int zeroupper, double ztol) const
{
    NiceAssert( isSquare() );
    NiceAssert( dnumRows == p.size() );
    NiceAssert( dnumRows == dest.numRows() );
    NiceAssert( dnumCols == dest.numCols() );

    retVector<int> tmpva;

    n = dnumCols;
    p = cntintvec(n,tmpva);

    if ( dnumRows )
    {
        int i,j,k;

	if ( zeroupper )
	{
            dest.zero();
	}

        for ( i = 0 ; i < n ; ++i )
        {
            int pi = p.v(i);

            dest("&",pi,pi) = (*this)(pi,pi);

            //if ( i > 0 )
            {
                for ( k = 0 ; k < i ; ++k )
                {
                    int pk = p.v(k);

                    dest("&",pi,pi) -= dest(pi,pk)*conj(dest(pi,pk));
                }
            }

            if ( ((double) dest(pi,pi)) < ztol )
            {
                for ( j = i ; j < dnumRows ; ++j )
                {
                    int pj = p.v(j);

                    dest("&",pj,pi) = 0.0;
                }

                p.blockswap(i,n-1);

                --n;
                --i;

                pi = p.v(i);
            }

            else
            {
                dest("&",pi,pi) = sqrt(dest(pi,pi));

                //if ( i < dnumRows-1 )
                {
                    for ( j = i+1 ; j < dnumRows ; ++j )
                    {
                        int pj = p.v(j);

                        dest("&",pj,pi) = (*this)(pj,pi);

                        //if ( i > 0 )
                        {
                            for ( k = 0 ; k < i ; ++k )
                            {
                                int pk = p.v(k);

                                dest("&",pj,pi) -= dest(pj,pk)*conj(dest(pi,pk));
                            }
                        }

                        dest("&",pj,pi) /= real(dest(pi,pi));
                    }
                }
            }
        }
    }

    return dest;
}

template <class T>
Matrix<T> &Matrix<T>::naivepartChol(Matrix<T> &dest, double diagoffscal, const Vector<T> &diagoffset, Vector<int> &p, int &n, int zeroupper, double ztol) const
{
    NiceAssert( isSquare() );
    NiceAssert( dnumRows == p.size() );
    NiceAssert( dnumRows == dest.numRows() );
    NiceAssert( dnumCols == dest.numCols() );
    NiceAssert( dnumRows == diagoffset.size() );

    retVector<int> tmpva;

    n = dnumCols;
    p = cntintvec(n,tmpva);

    if ( dnumRows )
    {
        int i,j,k;

	if ( zeroupper )
	{
            dest.zero();
	}

        for ( i = 0 ; i < n ; ++i )
        {
            dest("&",p(i),p(i)) = (*this)(p(i),p(i));
            dest("&",p(i),p(i)) += diagoffscal*diagoffset(p(i));

            //if ( i > 0 )
            {
                for ( k = 0 ; k < i ; ++k )
                {
                    dest("&",p(i),p(i)) -= dest(p(i),p(k))*conj(dest(p(i),p(k)));
                }
            }

            if ( ((double) dest(p(i),p(i))) < ztol )
            {
                for ( j = i ; j < dnumRows ; ++j )
                {
                    dest("&",p(j),p(i)) = 0.0;
                }

                p.blockswap(i,n-1);

                --n;
                --i;
            }

            else
            {
                dest("&",p(i),p(i)) = sqrt(dest(p(i),p(i)));

                //if ( i < dnumRows-1 )
                {
                    for ( j = i+1 ; j < dnumRows ; ++j )
                    {
                        dest("&",p(j),p(i)) = (*this)(p(j),p(i));

                        //if ( i > 0 )
                        {
                            for ( k = 0 ; k < i ; ++k )
                            {
                                dest("&",p(j),p(i)) -= dest(p(j),p(k))*conj(dest(p(i),p(k)));
                            }
                        }

                        dest("&",p(j),p(i)) /= real(dest(p(i),p(i)));
                    }
                }
            }
        }
    }

    return dest;
}

template <class T>
Matrix<T> &Matrix<T>::naivepartChol(const Vector<T> &diagsign, Matrix<T> &dest, Vector<int> &p, int &n, int zeroupper, double ztol) const
{
    NiceAssert( isSquare() );
    NiceAssert( dnumRows == p.size() );
    NiceAssert( dnumRows == dest.numRows() );
    NiceAssert( dnumCols == dest.numCols() );
    NiceAssert( dnumRows == diagsign.size() );

    retVector<int> tmpva;

    n = dnumCols;
    p = cntintvec(n,tmpva);

    if ( dnumRows )
    {
        int i,j,k;

	if ( zeroupper )
	{
            dest.zero();
	}

        for ( i = 0 ; i < n ; ++i )
        {
            dest("&",p(i),p(i)) = (*this)(p(i),p(i));

            //if ( i > 0 )
            {
                for ( k = 0 ; k < i ; ++k )
                {
                    dest("&",p(i),p(i)) -= dest(p(i),p(k))*diagsign(p(k))*conj(dest(p(i),p(k)));
                }
            }

            dest("&",p(i),p(i)) *= diagsign(p(i));

            if ( ((double) dest(p(i),p(i))) < ztol )
            {
                for ( j = i ; j < dnumRows ; ++j )
                {
                    dest("&",p(j),p(i)) = 0.0;
                }

                p.blockswap(i,n-1);

                --n;
                --i;
            }

            else
            {
                dest("&",p(i),p(i)) = sqrt(dest(p(i),p(i)));

                //if ( i < dnumRows-1 )
                {
                    for ( j = i+1 ; j < dnumRows ; ++j )
                    {
                        dest("&",p(j),p(i)) = (*this)(p(j),p(i));

                        //if ( i > 0 )
                        {
                            for ( k = 0 ; k < i ; ++k )
                            {
                                dest("&",p(j),p(i)) -= dest(p(j),p(k))*diagsign(p(k))*conj(dest(p(i),p(k)));
                            }
                        }

                        dest("&",p(j),p(i)) *= diagsign(p(i));
                        dest("&",p(j),p(i)) /= real(dest(p(i),p(i)));
                    }
                }
            }
        }
    }

    return dest;
}

template <class T>
Matrix<T> &Matrix<T>::naivepartChol(const Vector<T> &diagsign, Matrix<T> &dest, double diagoffscal, const Vector<T> &diagoffset, Vector<int> &p, int &n, int zeroupper, double ztol) const
{
    NiceAssert( isSquare() );
    NiceAssert( dnumRows == p.size() );
    NiceAssert( dnumRows == dest.numRows() );
    NiceAssert( dnumCols == dest.numCols() );
    NiceAssert( dnumRows == diagoffset.size() );
    NiceAssert( dnumRows == diagsign.size() );

    retVector<int> tmpva;

    n = dnumCols;
    p = cntintvec(n,tmpva);

    if ( dnumRows )
    {
        int i,j,k;

	if ( zeroupper )
	{
            dest.zero();
	}

        for ( i = 0 ; i < n ; ++i )
        {
            dest("&",p(i),p(i)) = (*this)(p(i),p(i));
            dest("&",p(i),p(i)) += diagsign(p(i))*diagoffscal*diagoffset(p(i));

            //if ( i > 0 )
            {
                for ( k = 0 ; k < i ; ++k )
                {
                    dest("&",p(i),p(i)) -= dest(p(i),p(k))*diagsign(p(k))*conj(dest(p(i),p(k)));
                }
            }

            dest("&",p(i),p(i)) *= diagsign(p(i));

            if ( ((double) dest(p(i),p(i))) < ztol )
            {
                for ( j = i ; j < dnumRows ; ++j )
                {
                    dest("&",p(j),p(i)) = 0.0;
                }

                p.blockswap(i,n-1);

                --n;
                --i;
            }

            else
            {
                dest("&",p(i),p(i)) = sqrt(dest(p(i),p(i)));

                //if ( i < dnumRows-1 )
                {
                    for ( j = i+1 ; j < dnumRows ; ++j )
                    {
                        dest("&",p(j),p(i)) = (*this)(p(j),p(i));

                        //if ( i > 0 )
                        {
                            for ( k = 0 ; k < i ; ++k )
                            {
                                dest("&",p(j),p(i)) -= dest(p(j),p(k))*diagsign(p(k))*conj(dest(p(i),p(k)));
                            }
                        }

                        dest("&",p(j),p(i)) *= diagsign(p(i));
                        dest("&",p(j),p(i)) /= real(dest(p(i),p(i)));
                    }
                }
            }
        }
    }

    return dest;
}

// Based on something on github by phillip Burckhardt, which was adapted from
// code by Luke Tierney and David Betz.

#define SVDSIGN(a, b) ((b) >= 0.0 ? abs2(a) : -abs2(a))
#define SVDMAX(x,y) ((x)>(y)?(x):(y))

template <class T> 
double SVDPYTHAG(const T &a, const T &b)
{
    double at = abs2(a);
    double bt = abs2(b);
    double ct,result;

    if (at > bt)       { ct = bt / at; result = at * sqrt(1.0 + ct * ct); }
    else if (bt > 0.0) { ct = at / bt; result = bt * sqrt(1.0 + ct * ct); }
    else               { result = 0.0; }

    return result;
}

template <class T>
Matrix<T> &Matrix<T>::SVD(Matrix<T> &a, Vector<T> &w, Matrix<T> &v) const
{
    int m = numRows();
    int n = numCols();

    NiceAssert( m >= n );

    a = *this;
    a.resize(m,m);

    w.resize(n);
    v.resize(n,n);

    int flag, i, its, j, jj, k, l, nm;
    double c, f, h, s, x, y, z;
    double anorm = 0.0, g = 0.0, scale = 0.0;

    Vector<T> rv1(n);
  
/* Householder reduction to bidiagonal form */
    for (i = 0; i < n; ++i) 
    {
        /* left-hand reduction */
        l = i + 1;
        rv1("&",i) = scale * g;
        g = s = scale = 0.0;
        if (i < m) 
        {
            for (k = i; k < m; ++k) 
                scale += abs2((double)a(k,i));
            if (scale) 
            {
                for (k = i; k < m; ++k) 
                {
                    a("&",k,i) = (double)((double)a(k,i)/scale);
                    s += ((double)a(k,i) * (double)a(k,i));
                }
                f = (double)a(i,i);
                g = -SVDSIGN(sqrt(s), f);
                h = f * g - s;
                a("&",i,i) = (double)(f - g);
                if (i != n - 1) 
                {
                    for (j = l; j < n; ++j) 
                    {
                        for (s = 0.0, k = i; k < m; ++k) 
                            s += ((double)a(k,i) * (double)a(k,j));
                        f = s / h;
                        for (k = i; k < m; ++k) 
                            a("&",k,j) += (double)(f * (double)a(k,i));
                    }
                }
                for (k = i; k < m; ++k) 
                    a("&",k,i) = (double)((double)a(k,i)*scale);
            }
        }
        w("&",i) = (double)(scale * g);
    
        /* right-hand reduction */
        g = s = scale = 0.0;
        if (i < m && i != n - 1) 
        {
            for (k = l; k < n; ++k) 
                scale += abs2((double)a(i,k));
            if (scale) 
            {
                for (k = l; k < n; ++k) 
                {
                    a("&",i,k) = (double)((double)a(i,k)/scale);
                    s += ((double)a(i,k) * (double)a(i,k));
                }
                f = (double)a(i,l);
                g = -SVDSIGN(sqrt(s), f);
                h = f * g - s;
                a("&",i,l) = (double)(f - g);
                for (k = l; k < n; ++k) 
                    rv1("&",k) = (double)a(i,k) / h;
                if (i != m - 1) 
                {
                    for (j = l; j < m; ++j) 
                    {
                        for (s = 0.0, k = l; k < n; ++k) 
                            s += ((double)a(j,k) * (double)a(i,k));
                        for (k = l; k < n; ++k) 
                            a("&",j,k) += (double)(s * rv1(k));
                    }
                }
                for (k = l; k < n; ++k) 
                    a("&",i,k) = (double)((double)a(i,k)*scale);
            }
        }
        anorm = SVDMAX(anorm, (abs2((double)w(i)) + abs2(rv1(i))));
    }
  
    /* accumulate the right-hand transformation */
    for (i = n - 1; i >= 0; --i) 
    {
        if (i < n - 1) 
        {
            if (g) 
            {
                for (j = l; j < n; ++j)
                    v("&",j,i) = (double)(((double)a(i,j) / (double)a(i,l)) / g);
                    /* double division to avoid underflow */
                for (j = l; j < n; ++j) 
                {
                    for (s = 0.0, k = l; k < n; ++k) 
                        s += ((double)a(i,k) * (double)v(k,j));
                    for (k = l; k < n; ++k) 
                        v("&",k,j) += (double)(s * (double)v(k,i));
                }
            }
            for (j = l; j < n; ++j) 
                v("&",i,j) = v("&",j,i) = 0.0;
        }
        v("&",i,i) = 1.0;
        g = rv1(i);
        l = i;
    }
  
    /* accumulate the left-hand transformation */
    for (i = n - 1; i >= 0; --i) 
    {
        l = i + 1;
        g = (double)w(i);
        if (i < n - 1) 
            for (j = l; j < n; ++j) 
                a("&",i,j) = 0.0;
        if (g) 
        {
            g = 1.0 / g;
            if (i != n - 1) 
            {
                for (j = l; j < n; ++j) 
                {
                    for (s = 0.0, k = l; k < m; ++k) 
                        s += ((double)a(k,i) * (double)a(k,j));
                    f = (s / (double)a(i,i)) * g;
                    for (k = i; k < m; ++k) 
                        a("&",k,j) += (double)(f * (double)a(k,i));
                }
            }
            for (j = i; j < m; ++j) 
                a("&",j,i) = (double)((double)a(j,i)*g);
        }
        else 
        {
            for (j = i; j < m; ++j) 
                a("&",j,i) = 0.0;
        }
        ++a("&",i,i);
    }

    /* diagonalize the bidiagonal form */
    for (k = n - 1; k >= 0; --k) 
    {                             /* loop over singular values */
        for (its = 0; its < 30; ++its) 
        {                         /* loop over allowed iterations */
            flag = 1;
            for (l = k; l >= 0; --l) 
            {                     /* test for splitting */
                nm = l - 1;
                if (abs2(rv1(l)) + anorm == anorm) 
                {
                    flag = 0;
                    break;
                }
                if (abs2((double)w(nm)) + anorm == anorm) 
                    break;
            }
            if (flag) 
            {
                c = 0.0;
                s = 1.0;
                for (i = l; i <= k; ++i) 
                {
                    f = s * rv1(i);
                    if (abs2(f) + anorm != anorm) 
                    {
                        g = (double)w(i);
                        h = SVDPYTHAG(f, g);
                        w("&",i) = (double)h; 
                        h = 1.0 / h;
                        c = g * h;
                        s = (- f * h);
                        for (j = 0; j < m; ++j) 
                        {
                            y = (double)a(j,nm);
                            z = (double)a(j,i);
                            a("&",j,nm) = (double)(y * c + z * s);
                            a("&",j,i) = (double)(z * c - y * s);
                        }
                    }
                }
            }
            z = (double)w(k);
            if (l == k) 
            {                  /* convergence */
                if (z < 0.0) 
                {              /* make singular value nonnegative */
                    w("&",k) = (double)(-z);
                    for (j = 0; j < n; ++j) 
                        v("&",j,k) = (-v(j,k));
                }
                break;
            }
            if (its >= 30) {
                NiceThrow("No convergence after 30,000! iterations");
            }
    
            /* shift from bottom 2 x 2 minor */
            x = (double)w(l);
            nm = k - 1;
            y = (double)w(nm);
            g = rv1(nm);
            h = rv1(k);
            f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
            g = SVDPYTHAG(f, 1.0);
            f = ((x - z) * (x + z) + h * ((y / (f + SVDSIGN(g, f))) - h)) / x;
          
            /* next QR transformation */
            c = s = 1.0;
            for (j = l; j <= nm; ++j) 
            {
                i = j + 1;
                g = rv1(i);
                y = (double)w(i);
                h = s * g;
                g = c * g;
                z = SVDPYTHAG(f, h);
                rv1("&",j) = z;
                c = f / z;
                s = h / z;
                f = x * c + g * s;
                g = g * c - x * s;
                h = y * s;
                y = y * c;
                for (jj = 0; jj < n; ++jj) 
                {
                    x = (double)v(jj,j);
                    z = (double)v(jj,i);
                    v("&",jj,j) = (double)(x * c + z * s);
                    v("&",jj,i) = (double)(z * c - x * s);
                }
                z = SVDPYTHAG(f, h);
                w("&",j) = (double)z;
                if (z) 
                {
                    z = 1.0 / z;
                    c = f * z;
                    s = h * z;
                }
                f = (c * g) + (s * y);
                x = (c * y) - (s * g);
                for (jj = 0; jj < m; ++jj) 
                {
                    y = (double)a(jj,j);
                    z = (double)a(jj,i);
                    a("&",jj,j) = (double)(y * c + z * s);
                    a("&",jj,i) = (double)(z * c - y * s);
                }
            }
            rv1("&",l) = 0.0;
            rv1("&",k) = f;
            w("&",k) = (double)x;
        }
    }

    return v;
}


template <class T>
template <class S> Vector<S> &Matrix<T>::forwardElim(Vector<S> &y, const Vector<S> &b, int implicitTranspose) const
{
//SEE ALSO SPECIALISATION BELOW
    // G.y = b  know G and b, calculate y

    NiceAssert( isSquare() );
    NiceAssert( ( dnumRows == b.size() ) || isEmpty() );

    if ( y.shareBase(b) )
    {
	Vector<S> bb(b);

        forwardElim(y,bb,implicitTranspose);
    }

    else if ( b.size() )
    {
        if ( isEmpty() )
        {
            y = b;
        }

        else
        {
            if ( !implicitTranspose )
            {
                int zer = 0;
                int i;
                S temp;

                y = b;

                for ( i = 0 ; i < dnumRows ; ++i )
                {
                    y("&",i) -= twoProduct(temp,(*this)(i,zer,1,i-1),y(zer,1,i-1));
                    y("&",i) = (inv((*this)(i,i))*y(i));
                }
            }

            else
            {
                int i,j;

                y = b;

                for ( i = 0 ; i < dnumRows ; ++i )
                {
                    //if ( i )
                    {
                        for ( j = 0 ; j < i ; ++j )
                        {
                            y("&",i) -= (*this)(j,i)*y(j);
                        }
                    }

                    y.set(i,(inv((*this)(i,i))*y(i)));
                }
            }
        }
    }

    return y;
}

#define MINDIAGLOC 1e-6

template <> template <> inline Vector<double> &Matrix<double>::forwardElim(Vector<double> &y, const Vector<double> &b, int implicitTranspose) const;
template <> template <> inline Vector<double> &Matrix<double>::forwardElim(Vector<double> &y, const Vector<double> &b, int implicitTranspose) const
{
    // G.y = b  know G and b, calculate y

    NiceAssert( isSquare() );
    NiceAssert( ( dnumRows == b.size() ) || isEmpty() );

    if ( y.shareBase(b) && ( &y != &b ) )
    {
	Vector<double> bb(b);

        forwardElim(y,bb,implicitTranspose);
    }

    else if ( b.size() )
    {
        if ( isEmpty() )
        {
            if ( &y != &b )
            {
                y = b;
            }
        }

        else
        {
            if ( !implicitTranspose )
            {
                int zer = 0;
                int i;
                double temp;

                if ( &y != &b )
                {
                    y = b;
                }

                retVector<double> tmpva;
                retVector<double> tmpvb;
                retVector<double> tmpvc;

                for ( i = 0 ; i < dnumRows ; ++i )
                {
                    y("&",i) -= twoProduct(temp,(*this)(i,zer,1,i-1,tmpva,tmpvc),y(zer,1,i-1,tmpvb));

                    double thisii = ( (*this).v(i,i) < MINDIAGLOC ) ? MINDIAGLOC : (*this).v(i,i);
                    double ywas = y.v(i);

tryagaina:
                    y("&",i) /= thisii;

                    if ( ( thisii < 1 ) && ( testisvnan(y.v(i)) || testisinf(y.v(i)) ) )
                    {
                        thisii *= 2;
                        y.sv(i,ywas);

                        goto tryagaina;
                    }
                }
            }

            else
            {
                int i,j;

                y = b;

                for ( i = 0 ; i < dnumRows ; ++i )
                {
                    //if ( i )
                    {
                        for ( j = 0 ; j < i ; ++j )
                        {
                            y("&",i) -= (*this).v(j,i)*y.v(j);
                        }
                    }

                    double thisii = ( (*this).v(i,i) < MINDIAGLOC ) ? MINDIAGLOC : (*this).v(i,i);
                    double ywas = y.v(i);

tryagainb:
                    y("&",i) /= thisii;

                    if ( ( thisii < 1 ) && ( testisvnan(y.v(i)) || testisinf(y.v(i)) ) )
                    {
                        thisii *= 2;
                        y.sv(i,ywas);

                        goto tryagainb;
                    }
                }
            }
        }
    }

    return y;
}

template <class T>
template <class S> Vector<S> &Matrix<T>::backwardSubst(Vector<S> &y, const Vector<S> &b, int implicitTranspose) const
{
//SEE ALSO SPECIALISATION BELOW
    // G.y = b  know G and b, calculate y

    NiceAssert( isSquare() );
    NiceAssert( ( dnumRows == b.size() ) || isEmpty() );

    if ( y.shareBase(b) )
    {
	Vector<S> bb(b);

        backwardSubst(y,bb,implicitTranspose);
    }

    else if ( b.size() )
    {
        if ( isEmpty() )
        {
            y = b;
        }

        else
        {
            if ( !implicitTranspose )
            {
                int i;
                S temp;

                y = b;

                for ( i = dnumRows-1 ; i >= 0 ; --i )
                {
                    y("&",i) -= twoProduct(temp,(*this)(i,i+1,1,dnumRows-1),y(i+1,1,dnumRows-1));
                    y.set(i,(inv((*this)(i,i))*y(i)));
                }
            }

            else
            {
                int i,j;

                y = b;

                for ( i = dnumRows-1 ; i >= 0 ; --i )
                {
                    //if ( i+1 < dnumRows )
                    {
                        for ( j = i+1 ; j < dnumRows ; ++j )
                        {
                            y("&",i) -= (*this)(j,i)*y(j);
                        }
                    }

                    y.set(i,(inv((*this)(i,i))*y(i)));
                }
            }
        }
    }

    return y;
}

template <> template <> inline Vector<double> &Matrix<double>::backwardSubst(Vector<double> &y, const Vector<double> &b, int implicitTranspose) const;
template <> template <> inline Vector<double> &Matrix<double>::backwardSubst(Vector<double> &y, const Vector<double> &b, int implicitTranspose) const
{
    // G.y = b  know G and b, calculate y

    NiceAssert( isSquare() );
    NiceAssert( ( dnumRows == b.size() ) || isEmpty() );

    if ( y.shareBase(b) && ( &y != &b ) )
    {
	Vector<double> bb(b);

        backwardSubst(y,bb,implicitTranspose);
    }

    else if ( b.size() )
    {
        if ( isEmpty() )
        {
            if ( &y != &b )
            {
                y = b;
            }
        }

        else
        {
            if ( !implicitTranspose )
            {
                int i;
                double temp;

                if ( &y != &b )
                {
                    y = b;
                }

                retVector<double> tmpva;
                retVector<double> tmpvb;
                retVector<double> tmpvc;

                for ( i = dnumRows-1 ; i >= 0 ; --i )
                {
                    y("&",i) -= twoProduct(temp,(*this)(i,i+1,1,dnumRows-1,tmpva,tmpvc),y(i+1,1,dnumRows-1,tmpvb));

                    double thisii = ( (*this).v(i,i) < MINDIAGLOC ) ? MINDIAGLOC : (*this).v(i,i);
                    double ywas = y.v(i);

tryagaina:
                    y("&",i) /= thisii;

                    if ( ( thisii < 1 ) && ( testisvnan(y.v(i)) || testisinf(y.v(i)) ) )
                    {
                        thisii *= 2;
                        y.sv(i,ywas);

                        goto tryagaina;
                    }
                }
            }

            else
            {
                int i,j;

                y = b;

                for ( i = dnumRows-1 ; i >= 0 ; --i )
                {
                    //if ( i+1 < dnumRows )
                    {
                        for ( j = i+1 ; j < dnumRows ; ++j )
                        {
                            y("&",i) -= (*this).v(j,i)*y.v(j);
                        }
                    }

                    double thisii = ( (*this).v(i,i) < MINDIAGLOC ) ? MINDIAGLOC : (*this).v(i,i);
                    double ywas = y.v(i);

tryagainb:
                    y("&",i) /= thisii;

                    if ( ( thisii < 1 ) && ( testisvnan(y.v(i)) || testisinf(y.v(i)) ) )
                    {
                        thisii *= 2;
                        y.sv(i,ywas);

                        goto tryagainb;
                    }
                }
            }
        }
    }

    return y;
}






template <class T>
template <class S> Vector<S> &Matrix<T>::naiveCholInve(Vector<S> &y, const Vector<S> &b) const
{
    Vector<S> yinterm(y);
    Matrix<T> cholres(numRows(),numCols());
    int crisprecalc = 0;

    return naiveCholInve(y,b,yinterm,cholres,0,crisprecalc);
}

template <class T>
template <class S> Vector<S> &Matrix<T>::naiveCholInve(Vector<S> &y, const Vector<S> &b, Vector<S> &yinterm) const
{
    Matrix<T> cholres(numRows(),numCols());
    int crisprecalc = 0;

    return naiveCholInve(y,b,yinterm,cholres,0,crisprecalc);
}

template <class T>
template <class S> Vector<S> &Matrix<T>::naiveCholInve(Vector<S> &y, const Vector<S> &b, Matrix<T> & cholres, int zeroupper, int &crisprecalc) const
{
    Vector<S> yinterm(y);

    return naiveCholInve(y,b,yinterm,cholres,zeroupper,crisprecalc);
}

template <class T>
template <class S> Vector<S> &Matrix<T>::naiveCholInve(Vector<S> &y, const Vector<S> &b, double diagoffscal, const Vector<T> &diagoffset) const
{
    Vector<S> yinterm(y);
    Matrix<T> cholres(numRows(),numCols());
    int crisprecalc = 0;

    return naiveCholInve(y,b,diagoffscal,diagoffset,yinterm,cholres,0,crisprecalc);
}

template <class T>
template <class S> Vector<S> &Matrix<T>::naiveCholInve(Vector<S> &y, const Vector<S> &b, double diagoffscal, const Vector<T> &diagoffset, Vector<S> &yinterm) const
{
    Matrix<T> cholres(numRows(),numCols());
    int crisprecalc = 0;

    return naiveCholInve(y,b,diagoffscal,diagoffset,yinterm,cholres,0,crisprecalc);
}

template <class T>
template <class S> Vector<S> &Matrix<T>::naiveCholInve(Vector<S> &y, const Vector<S> &b, double diagoffscal, const Vector<T> &diagoffset, Matrix<T> & cholres, int zeroupper, int &crisprecalc) const
{
    Vector<S> yinterm(y);

    return naiveCholInve(y,b,diagoffscal,diagoffset,yinterm,cholres,zeroupper,crisprecalc);
}

template <class T>
template <class S> Vector<S> &Matrix<T>::naiveCholInve(const char *dummy, const Vector<T> &diagsign, Vector<S> &y, const Vector<S> &b) const
{
    Vector<S> yinterm(y);
    Matrix<T> cholres(numRows(),numCols());
    int crisprecalc = 0;

    return naiveCholInve(dummy,diagsign,y,b,yinterm,cholres,0,crisprecalc);
}

template <class T>
template <class S> Vector<S> &Matrix<T>::naiveCholInve(const char *dummy, const Vector<T> &diagsign, Vector<S> &y, const Vector<S> &b, Vector<S> &yinterm) const
{
    Matrix<T> cholres(numRows(),numCols());
    int crisprecalc = 0;

    return naiveCholInve(dummy,diagsign,y,b,yinterm,cholres,0,crisprecalc);
}

template <class T>
template <class S> Vector<S> &Matrix<T>::naiveCholInve(const char *dummy, const Vector<T> &diagsign, Vector<S> &y, const Vector<S> &b, Matrix<T> & cholres, int zeroupper, int &crisprecalc) const
{
    Vector<S> yinterm(y);

    return naiveCholInve(dummy,diagsign,y,b,yinterm,cholres,zeroupper,crisprecalc);
}

template <class T>
template <class S> Vector<S> &Matrix<T>::naiveCholInve(const char *dummy, const Vector<T> &diagsign, Vector<S> &y, const Vector<S> &b, double diagoffscal, const Vector<T> &diagoffset) const
{
    Vector<S> yinterm(y);
    Matrix<T> cholres(numRows(),numCols());
    int crisprecalc = 0;

    return naiveCholInve(dummy,diagsign,y,b,diagoffscal,diagoffset,yinterm,cholres,0,crisprecalc);
}

template <class T>
template <class S> Vector<S> &Matrix<T>::naiveCholInve(const char *dummy, const Vector<T> &diagsign, Vector<S> &y, const Vector<S> &b, double diagoffscal, const Vector<T> &diagoffset, Vector<S> &yinterm) const
{
    Matrix<T> cholres(numRows(),numCols());
    int crisprecalc = 0;

    return naiveCholInve(dummy,diagsign,y,b,diagoffscal,diagoffset,yinterm,cholres,0,crisprecalc);
}

template <class T>
template <class S> Vector<S> &Matrix<T>::naiveCholInve(const char *dummy, const Vector<T> &diagsign, Vector<S> &y, const Vector<S> &b, double diagoffscal, const Vector<T> &diagoffset, Matrix<T> & cholres, int zeroupper, int &crisprecalc) const
{
    Vector<S> yinterm(y);

    return naiveCholInve(dummy,diagsign,y,b,diagoffscal,diagoffset,yinterm,cholres,zeroupper,crisprecalc);
}








template <class T>
template <class S> Vector<S> &Matrix<T>::naiveCholInve(Vector<S> &y, const Vector<S> &b, Vector<S> &yinterm, Matrix<T> & cholres, int zeroupper, int &crisprecalc) const
{
    if ( !crisprecalc )
    {
        naiveChol(cholres,zeroupper);

        crisprecalc = 1;
    }

    NiceAssert( cholres.numRows() == numRows() );
    NiceAssert( cholres.numCols() == numCols() );

    cholres.forwardElim(yinterm,b);
    cholres.backwardSubst(y,yinterm,1); // implicit transpose here to use lower triangular part of cholres

    return y;
}

template <class T>
template <class S> Vector<S> &Matrix<T>::naiveCholInve(Vector<S> &y, const Vector<S> &b, double diagoffscal, const Vector<T> &diagoffset, Vector<S> &yinterm, Matrix<T> & cholres, int zeroupper, int &crisprecalc) const
{
    if ( !crisprecalc )
    {
        naiveChol(cholres,diagoffscal,diagoffset,zeroupper);

        crisprecalc = 1;
    }

    NiceAssert( cholres.numRows() == numRows() );
    NiceAssert( cholres.numCols() == numCols() );

    cholres.forwardElim(yinterm,b);
    cholres.backwardSubst(y,yinterm,1); // see above

    return y;
}

template <class T>
template <class S> Vector<S> &Matrix<T>::naiveCholInve(const char *, const Vector<T> &diagsign, Vector<S> &y, const Vector<S> &b, Vector<S> &yinterm, Matrix<T> & cholres, int zeroupper, int &crisprecalc) const
{
    if ( !crisprecalc )
    {
        naiveChol(diagsign,cholres,zeroupper);

        crisprecalc = 1;
    }

    NiceAssert( cholres.numRows() == numRows() );
    NiceAssert( cholres.numCols() == numCols() );

    cholres.forwardElim(yinterm,b);
    yinterm *= diagsign;
    cholres.backwardSubst(y,yinterm,1); // see above

    return y;
}

template <class T>
template <class S> Vector<S> &Matrix<T>::naiveCholInve(const char *, const Vector<T> &diagsign, Vector<S> &y, const Vector<S> &b, double diagoffscal, const Vector<T> &diagoffset, Vector<S> &yinterm, Matrix<T> & cholres, int zeroupper, int &crisprecalc) const
{
    if ( !crisprecalc )
    {
        naiveChol(diagsign,cholres,diagoffscal,diagoffset,zeroupper);

        crisprecalc = 1;
    }

    NiceAssert( cholres.numRows() == numRows() );
    NiceAssert( cholres.numCols() == numCols() );

    cholres.forwardElim(yinterm,b);
    yinterm *= diagsign;
    cholres.backwardSubst(y,yinterm,1); // see above

    return y;
}









template <class T>
template <class S> Vector<S> &Matrix<T>::offNaiveCholInve(Vector<S> &y, const Vector<S> &b, double sscal, const Vector<T> &s) const
{
    Vector<S> yinterm(y);
    Vector<S> xinterm(y);
    Matrix<T> cholres(numRows(),numCols());
    retVector<double> tmpva;
    int crisprecalc = 0;
    int fbused = 0;
    double convScale = CONVSCALE;

    return offNaiveCholInve(y,b,0.0,zerodoublevec(y.size(),tmpva),yinterm,xinterm,cholres,sscal,s,0,crisprecalc,fbused,convScale);
}

template <class T>
template <class S> Vector<S> &Matrix<T>::offNaiveCholInve(Vector<S> &y, const Vector<S> &b, Vector<S> &yinterm, Vector<S> &xinterm, double sscal, const Vector<T> &s) const
{
    Matrix<T> cholres(numRows(),numCols());
    retVector<double> tmpva;
    int crisprecalc = 0;
    int fbused = 0;
    double convScale = CONVSCALE;

    return offNaiveCholInve(y,b,0.0,zerodoublevec(y.size(),tmpva),yinterm,xinterm,cholres,sscal,s,0,crisprecalc,fbused,convScale);
}

template <class T>
template <class S> Vector<S> &Matrix<T>::offNaiveCholInve(Vector<S> &y, const Vector<S> &b, Matrix<T> & cholres, double sscal, const Vector<T> &s, int zeroupper, int &crisprecalc, int &fbused, double convScale) const
{
    Vector<S> yinterm(y);
    Vector<S> xinterm(y);
    retVector<double> tmpva;

    return offNaiveCholInve(y,b,0.0,zerodoublevec(y.size(),tmpva),yinterm,xinterm,cholres,sscal,s,zeroupper,crisprecalc,fbused,convScale);
}

template <class T>
template <class S> Vector<S> &Matrix<T>::offNaiveCholInve(Vector<S> &y, const Vector<S> &b, double diagoffscal, const Vector<T> &diagoffset, double sscal, const Vector<T> &s) const
{
    Vector<S> yinterm(y);
    Vector<S> xinterm(y);
    Matrix<T> cholres(numRows(),numCols());
    int crisprecalc = 0;
    int fbused = 0;
    double convScale = CONVSCALE;

    return offNaiveCholInve(y,b,diagoffscal,diagoffset,yinterm,xinterm,cholres,sscal,s,0,crisprecalc,fbused,convScale);
}

template <class T>
template <class S> Vector<S> &Matrix<T>::offNaiveCholInve(Vector<S> &y, const Vector<S> &b, double diagoffscal, const Vector<T> &diagoffset, Vector<S> &yinterm, Vector<S> &xinterm, double sscal, const Vector<T> &s) const
{
    Matrix<T> cholres(numRows(),numCols());
    int crisprecalc = 0;
    int fbused = 0;
    double convScale = CONVSCALE;

    return offNaiveCholInve(y,b,diagoffscal,diagoffset,yinterm,xinterm,cholres,sscal,s,0,crisprecalc,fbused,convScale);
}

template <class T>
template <class S> Vector<S> &Matrix<T>::offNaiveCholInve(Vector<S> &y, const Vector<S> &b, double diagoffscal, const Vector<T> &diagoffset, Matrix<T> & cholres, double sscal, const Vector<T> &s, int zeroupper, int &crisprecalc, int &fbused, double convScale) const
{
    Vector<S> yinterm(y);
    Vector<S> xinterm(y);

    return offNaiveCholInve(y,b,diagoffscal,diagoffset,yinterm,xinterm,cholres,sscal,s,zeroupper,crisprecalc,fbused,convScale);
}

template <class T>
template <class S> Vector<S> &Matrix<T>::offNaiveCholInve(Vector<S> &y, const Vector<S> &b, Vector<S> &yinterm, Vector<S> &xinterm, Matrix<T> & cholres, double sscal, const Vector<T> &s, int zeroupper, int &crisprecalc, int &fbused, double convScale) const
{
    retVector<double> tmpva;

    return offNaiveCholInve(y,b,0.0,zerodoublevec(y.size(),tmpva),yinterm,xinterm,cholres,sscal,s,zeroupper,crisprecalc,fbused,convScale);
}

template <class T>
template <class S> Vector<S> &Matrix<T>::offNaiveCholInve(const char *dummy, const Vector<T> &diagsign, Vector<S> &y, const Vector<S> &b, double sscal, const Vector<T> &s) const
{
    Vector<S> yinterm(y);
    Vector<S> xinterm(y);
    Matrix<T> cholres(numRows(),numCols());
    retVector<double> tmpva;
    int crisprecalc = 0;
    int fbused = 0;
    double convScale = CONVSCALE;

    return offNaiveCholInve(dummy,diagsign,y,b,0.0,zerodoublevec(y.size(),tmpva),yinterm,xinterm,cholres,sscal,s,0,crisprecalc,fbused,convScale);
}

template <class T>
template <class S> Vector<S> &Matrix<T>::offNaiveCholInve(const char *dummy, const Vector<T> &diagsign, Vector<S> &y, const Vector<S> &b, Vector<S> &yinterm, Vector<S> &xinterm, double sscal, const Vector<T> &s) const
{
    Matrix<T> cholres(numRows(),numCols());
    retVector<double> tmpva;
    int crisprecalc = 0;
    int fbused = 0;
    double convScale = CONVSCALE;

    return offNaiveCholInve(dummy,diagsign,y,b,0.0,zerodoublevec(y.size(),tmpva),yinterm,xinterm,cholres,sscal,s,0,crisprecalc,fbused,convScale);
}

template <class T>
template <class S> Vector<S> &Matrix<T>::offNaiveCholInve(const char *dummy, const Vector<T> &diagsign, Vector<S> &y, const Vector<S> &b, Matrix<T> & cholres, double sscal, const Vector<T> &s, int zeroupper, int &crisprecalc, int &fbused,double convScale) const
{
    Vector<S> yinterm(y);
    Vector<S> xinterm(y);
    retVector<double> tmpva;

    return offNaiveCholInve(dummy,diagsign,y,b,0.0,zerodoublevec(y.size(),tmpva),yinterm,xinterm,cholres,sscal,s,zeroupper,crisprecalc,fbused,convScale);
}

template <class T>
template <class S> Vector<S> &Matrix<T>::offNaiveCholInve(const char *dummy, const Vector<T> &diagsign, Vector<S> &y, const Vector<S> &b, double diagoffscal, const Vector<T> &diagoffset, double sscal, const Vector<T> &s) const
{
    Vector<S> yinterm(y);
    Vector<S> xinterm(y);
    Matrix<T> cholres(numRows(),numCols());
    int crisprecalc = 0;
    int fbused = 0;
    double convScale = CONVSCALE;

    return offNaiveCholInve(dummy,diagsign,y,b,diagoffscal,diagoffset,yinterm,xinterm,cholres,sscal,s,0,crisprecalc,fbused,convScale);
}

template <class T>
template <class S> Vector<S> &Matrix<T>::offNaiveCholInve(const char *dummy, const Vector<T> &diagsign, Vector<S> &y, const Vector<S> &b, double diagoffscal, const Vector<T> &diagoffset, Vector<S> &yinterm, Vector<S> &xinterm, double sscal, const Vector<T> &s) const
{
    Matrix<T> cholres(numRows(),numCols());
    int crisprecalc = 0;
    int fbused = 0;
    double convScale = CONVSCALE;

    return offNaiveCholInve(dummy,diagsign,y,b,diagoffscal,diagoffset,yinterm,xinterm,cholres,sscal,s,0,crisprecalc,fbused,convScale);
}

template <class T>
template <class S> Vector<S> &Matrix<T>::offNaiveCholInve(const char *dummy, const Vector<T> &diagsign, Vector<S> &y, const Vector<S> &b, double diagoffscal, const Vector<T> &diagoffset, Matrix<T> & cholres, double sscal, const Vector<T> &s, int zeroupper, int &crisprecalc, int &fbused,double convScale) const
{
    Vector<S> yinterm(y);
    Vector<S> xinterm(y);

    return offNaiveCholInve(dummy,diagsign,y,b,diagoffscal,diagoffset,yinterm,xinterm,cholres,sscal,s,zeroupper,crisprecalc,fbused,convScale);
}

template <class T>
template <class S> Vector<S> &Matrix<T>::offNaiveCholInve(const char *dummy, const Vector<T> &diagsign, Vector<S> &y, const Vector<S> &b, Vector<S> &yinterm, Vector<S> &xinterm, Matrix<T> & cholres, double sscal, const Vector<T> &s, int zeroupper, int &crisprecalc, int &fbused, double convScale) const
{
    retVector<double> tmpva;

    return offNaiveCholInve(dummy,diagsign,y,b,0.0,zerodoublevec(y.size(),tmpva),yinterm,xinterm,cholres,sscal,s,zeroupper,crisprecalc,fbused,convScale);
}










template <class T>
template <class S> Vector<S> &Matrix<T>::offNaiveCholInve(Vector<S> &y, const Vector<S> &b, double diagoffscal, const Vector<T> &diagoffset, Vector<S> &yinterm, Vector<S> &xinterm, Matrix<T> & cholres, double sscal, const Vector<T> &s, int zeroupper, int &crisprecalc, int &fbused, double convScale) const
{
    // Calculate x0 and ensure that the factorisation is ready for use

    naiveCholInve(y,b,diagoffscal,diagoffset,yinterm,cholres,zeroupper,crisprecalc);

    xinterm = y; // this will be used to store xi (currently x0)

    int i = 0;

    // Starting point: y = x0, xinterm = x0

    bool isconverged = false;
    bool isdiverged  = false;

    int badcnt = 0;

    double xiMag = abs2(xinterm);
    double xiMagstart = xiMag;

    while ( !isconverged && !isdiverged )
    {
        // We have xi stored in xinterm.  Our next step is to use
        // the recursion x{i+1} = (inv(G).diag(s)).xi to find
        // x{i+1}.

        ( xinterm *= s ) *= sscal;
        cholres.forwardElim(yinterm,xinterm);
        cholres.backwardSubst(xinterm,yinterm,1);

        // Add to y to update solution

        y += xinterm;

        // Convergence test

        xiMag = abs2(xinterm);
        ++i;

        if ( xiMag >= xiMagstart )
        {
            ++badcnt;
        }

        isconverged = ( xiMag < xiMagstart*convScale );
        isdiverged  = ( badcnt > size()/20 ) || ( i > size()/10 );
    }

    if ( isdiverged )
    {
        // Fallback method if no convergence has occurred

        Vector<double> totoffset(diagoffset);
        totoffset *= diagoffscal;
        totoffset.scaleAdd(-sscal,s);

        naiveCholInve(y,b,1.0,totoffset,yinterm);

        ++fbused;
    }

    return y;
}

template <class T>
template <class S> Vector<S> &Matrix<T>::offNaiveCholInve(const char *dummy, const Vector<T> &diagsign, Vector<S> &y, const Vector<S> &b, double diagoffscal, const Vector<T> &diagoffset, Vector<S> &yinterm, Vector<S> &xinterm, Matrix<T> & cholres, double sscal, const Vector<T> &s, int zeroupper, int &crisprecalc, int &fbused, double convScale) const
{
    // Calculate x0 and ensure that the factorisation is ready for use

    naiveCholInve(dummy,diagsign,y,b,diagoffscal,diagoffset,yinterm,cholres,zeroupper,crisprecalc);

    xinterm = y; // this will be used to store xi (currently x0)

    int i = 0;

    // Starting point: y = x0, xinterm = x0

    bool isconverged = false;
    bool isdiverged  = false;

    int badcnt = 0;

    double xiMag = abs2(xinterm);
    double xiMagstart = xiMag;

    while ( !isconverged && !isdiverged )
    {
        // We have xi stored in xinterm.  Our next step is to use
        // the recursion x{i+1} = (inv(G).diag(s)).xi to find
        // x{i+1}.

        ( xinterm *= s ) *= sscal;
        cholres.forwardElim(yinterm,xinterm);
        yinterm *= diagsign;
        cholres.backwardSubst(xinterm,yinterm,1);

        // Add to y to update solution

        y += xinterm;

        // Convergence test

        xiMag = abs2(xinterm);
        ++i;

        if ( xiMag >= xiMagstart )
        {
            ++badcnt;
        }

        isconverged = ( xiMag < xiMagstart*convScale );
        isdiverged  = ( badcnt > size()/20 ) || ( i > size()/10 );
    }

    if ( isdiverged )
    {
        // Fallback method if no convergence has occurred

        Vector<double> totoffset(diagoffset);
        totoffset *= diagoffscal;
        totoffset.scaleAdd(-sscal,s);

        naiveCholInve(dummy,diagsign,y,b,1.0,totoffset,yinterm);

        ++fbused;
    }

    return y;
}











// Determinants, inverses etc

template <class T>
T Matrix<T>::det(void) const
{
    NiceAssert( isSquare() );

    T res;

    if ( dnumRows == 0 )
    {
	setident(res); // this will be 1 if matrix of scalars, empty matrix if matrix of matrices, error if matrix of vectors
    }

    else if ( dnumRows == 1 )
    {
        // Trivial result

	res = (*this)(0,0);
    }

    else if ( dnumRows <= MATRIX_DETRECURSEMAX )
    {
        // For small matrices use method of cofactors

	int i;

	res = ( (*this)(0,0) * cofactor(0,0) );

	for ( i = 1 ; i < dnumCols ; ++i )
	{
	    res += ( (*this)(0,i) * cofactor(0,i) );
	}
    }

    else
    {
        // For large matrices use LUP decomposition method

        //if ( !matbuff )
        //{
        //    MEMNEW(matbuff,Matrix<T>);
        //}

        //Matrix<T> &locmatbuff = *matbuff;
        static thread_local Matrix<T> locmatbuff;
        Vector<int> p;

        locmatbuff = *this;

        int s = locmatbuff.LUPDecompose(p,MATRIX_ZTOL);
        int i;

        //const Vector<int> &p = *(locmatbuff.pbuff);

        if ( s >= 0 )
        {
            res = locmatbuff(p.v(0),0);

            for ( i = 1 ; i < dnumRows ; ++i )
            {
                res *= locmatbuff(p.v(i),i);
            }

            if ( s % 2 )
            {
                setnegate(res);
            }
        }

        else
        {
            setzero(res);
        }
    }

    return res;
}

template <class T>
T Matrix<T>::det_naiveChol(void) const
{
    Matrix<T> factScratch;

    return det_naiveChol(factScratch);
}

template <class T>
T Matrix<T>::det_naiveChol(double diagoffscal, const Vector<T> &diagoffset) const
{
    Matrix<T> factScratch;

    return det_naiveChol(factScratch,diagoffscal,diagoffset);
}

template <class T>
T Matrix<T>::det_naiveChol(const Vector<T> &diagsign) const
{
    Matrix<T> factScratch;

    return det_naiveChol(diagsign,factScratch);
}

template <class T>
T Matrix<T>::det_naiveChol(const Vector<T> &diagsign, double diagoffscal, const Vector<T> &diagoffset) const
{
    Matrix<T> factScratch;

    return det_naiveChol(diagsign,factScratch,diagoffscal,diagoffset);
}

template <class T>
T Matrix<T>::logdet_naiveChol(void) const
{
    Matrix<T> factScratch;

    return logdet_naiveChol(factScratch);
}

template <class T>
T Matrix<T>::logdet_naiveChol(double diagoffscal, const Vector<T> &diagoffset) const
{
    Matrix<T> factScratch;

    return logdet_naiveChol(factScratch,diagoffscal,diagoffset);
}

template <class T>
T Matrix<T>::logdet_naiveChol(const Vector<T> &diagsign) const
{
    Matrix<T> factScratch;

    return logdet_naiveChol(diagsign,factScratch);
}

template <class T>
T Matrix<T>::logdet_naiveChol(const Vector<T> &diagsign, double diagoffscal, const Vector<T> &diagoffset) const
{
    Matrix<T> factScratch;

    return logdet_naiveChol(diagsign,factScratch,diagoffscal,diagoffset);
}

template <class T>
T Matrix<T>::det_naiveChol(Matrix<T> &factScratch) const
{
    T res = naiveChol(factScratch,0).diagprod();

    return res*res;
}

template <class T>
T Matrix<T>::det_naiveChol(Matrix<T> &factScratch, double diagoffscal, const Vector<T> &diagoffset) const
{
    T res = naiveChol(factScratch,diagoffscal,diagoffset,0).diagprod();

    return res*res;
}

template <class T>
T Matrix<T>::det_naiveChol(const Vector<T> &diagsign, Matrix<T> &factScratch) const
{
    T res = naiveChol(diagsign,factScratch,0).diagprod();

    return res*res;
}

template <class T>
T Matrix<T>::det_naiveChol(const Vector<T> &diagsign, Matrix<T> &factScratch, double diagoffscal, const Vector<T> &diagoffset) const
{
    T res = naiveChol(diagsign,factScratch,diagoffscal,diagoffset,0).diagprod();

    return res*res;
}

template <class T>
T Matrix<T>::logdet_naiveChol(Matrix<T> &factScratch) const
{
    return 2*naiveChol(factScratch,0).tracelog();
}

template <class T>
T Matrix<T>::logdet_naiveChol(Matrix<T> &factScratch, double diagoffscal, const Vector<T> &diagoffset) const
{
    return 2*naiveChol(factScratch,diagoffscal,diagoffset,0).tracelog();
}

template <class T>
T Matrix<T>::logdet_naiveChol(const Vector<T> &diagsign, Matrix<T> &factScratch) const
{
    return 2*naiveChol(diagsign,factScratch,0).tracelog();
}

template <class T>
T Matrix<T>::logdet_naiveChol(const Vector<T> &diagsign, Matrix<T> &factScratch, double diagoffscal, const Vector<T> &diagoffset) const
{
    return 2*naiveChol(diagsign,factScratch,diagoffscal,diagoffset,0).tracelog();
}



template <class T>
T Matrix<T>::trace(int maxsize) const
{
    T res;

    if ( !dnumRows || !dnumCols || !maxsize )
    {
        setzero(res);
    }

    else
    {
	res = (*this)(0,0);

        for ( int i = 1 ; ( i < dnumRows ) && ( i < dnumCols ) && ( i < maxsize ) ; ++i )
        {
            res += (*this)(i,i);
	}
    }

    return res;
}

template <class T>
T Matrix<T>::tracelog(int maxsize) const
{
    T res;

    if ( !dnumRows || !dnumCols || !maxsize )
    {
        setzero(res);
    }

    else
    {
	res = log((*this)(0,0));

        for ( int i = 1 ; ( i < dnumRows ) && ( i < dnumCols ) && ( i < maxsize ) ; ++i )
        {
	    res += log((*this)(i,i));
	}
    }

    return res;
}

template <class T>
T Matrix<T>::diagprod(int maxsize) const
{
    T res;

    if ( !dnumRows || !dnumCols || !maxsize )
    {
        setident(res);
    }

    else
    {
	res = (*this)(0,0);

        for ( int i = 1 ; ( i < dnumRows ) && ( i < dnumCols ) && ( i < maxsize ) ; ++i )
        {
            res *= (*this)(i,i);
	}
    }

    return res;
}

template <class T>
T Matrix<T>::invtrace(void) const
{
    NiceAssert( isSquare() );

    T res;

    // Calculate 1/tr(inv(A)) = det(A)/

    if ( !dnumRows )
    {
        // We treat this case as an analogy of dnumRows == 1

	setzero(res);
    }

    else if ( dnumRows == 1 )
    {
        // In this case 1/tr(inv(A)) = 1/(1/A00) = A00

	res = (*this)(0,0);
    }

    else
    {
        // In this case 1/tr(inv()) = 1/(tr(adj()/det()))
        //                          = 1/(sum_i(miner(i,i)/det()))
        //                          = det()/(sum_i(miner(i,i)))

        int i;
        T ressum;

        Vector<int> rowcolsel(dnumRows-1);

        // The following is the miner function specialised for diagonals.
        // rowcolsel is the index vector with current diagonal removed.
        // It is set once at start and then can be updated incrementally.

        for ( i = 1 ; i < dnumRows ; ++i )
        {
            rowcolsel.sv(i-1,i);
        }

        retMatrix<T> tmpma;

        for ( i = 0 ; i < dnumCols ; ++i )
        {
            if ( i )
            {
                --(rowcolsel("&",i-1));
            }

            if ( !i )
            {
                ressum = ((*this)(rowcolsel,rowcolsel,tmpma)).det();
            }

            else
            {
                ressum += ((*this)(rowcolsel,rowcolsel,tmpma)).det();
            }
        }

        res =  det();
        res /= ressum;
    }

    return res;
}

template <class T>
T Matrix<T>::miner(int i, int j) const
{
    NiceAssert( isSquare() );
    NiceAssert( dnumRows );

    T res;

    if ( dnumRows == 1 )
    {
	// Need a special case here.  If the matrix
	// is a 1x1 matrix then the miner of the only
	// element should be the identity matrix.  However to
	// ensure the size is compatible with other elements we
	// want the identity the size of that element.

	res = (*this)(0,0);
        setident(res);
    }

    else
    {
	Vector<int> rowsel(dnumRows-1);
	Vector<int> colsel(dnumCols-1);

	int ii,jj,kk;

	ii = 0;
	jj = 0;

	for ( kk = 0 ; kk < dnumRows ; ++kk )
	{
	    if ( kk != i )
	    {
		rowsel.sv(ii,kk);

		++ii;
	    }

	    if ( kk != j )
	    {
		colsel.sv(jj,kk);

		++jj;
	    }
	}

        retMatrix<T> tmpma;

	res = ((*this)(rowsel,colsel,tmpma)).det();
    }

    return res;
}

template <class T>
T Matrix<T>::cofactor(int i, int j) const
{
    NiceAssert( isSquare() );
    NiceAssert( dnumRows );

    T res = miner(i,j);

    if ( (i+j)%2 )
    {
        setnegate(res);
    }

    return res;
}

template <class T>
Matrix<T> &Matrix<T>::adj(Matrix<T> &res) const
{
    NiceAssert( isSquare() );

    res.resize(dnumRows,dnumCols);

    if ( dnumRows && dnumCols )
    {
	int i,j;

	for ( i = 0 ; i < dnumRows ; ++i )
	{
	    for ( j = 0 ; j < dnumCols ; ++j )
	    {
		res("&",i,j) = cofactor(j,i);
	    }
	}
    }

    return res;
}

template <class T>
Matrix<T> &Matrix<T>::inve(Matrix<T> &res) const
{
    res.resize(dnumRows,dnumCols);

    if ( dnumRows && dnumCols )
    {
        if ( isSquare() )
	{
	    T matdet;
	    matdet = det();

	    if ( dnumRows == 1 )
	    {
		res("&",0,0) = inv(matdet);
	    }

            else if ( dnumRows == 2 )
            {
                res("&",0,0) = (*this)(1,1)*inv(matdet);
                res("&",0,1) = -(*this)(0,1)*inv(matdet);
                res("&",1,0) = -(*this)(1,0)*inv(matdet);
                res("&",1,1) = (*this)(0,0)*inv(matdet);
            }

	    else
	    {
		int i,j;

		for ( i = 0 ; i < dnumRows ; ++i )
		{
		    for ( j = 0 ; j < dnumCols ; ++j )
		    {
			res("&",i,j) = (cofactor(j,i)*inv(matdet));
		    }
		}
	    }
	}

	else if ( dnumRows > dnumCols )
	{
	    // Tall matrix
	    //
	    // A+ = ((A*.A)+).(A*)
	    // A+* = A.((A*.A)+*)
	    //     = A.(((A*.A)*)+) (return this)

	    Matrix<T> submatr(dnumCols,dnumCols);
	    Matrix<T> subinv(dnumCols,dnumCols);
	    T leftargx;
            T rightargx;

	    int i,j,k;

	    for ( i = 0 ; i < dnumCols ; ++i )
	    {
		for ( j = 0 ; j < dnumCols ; ++j )
		{
		    for ( k = 0 ; k < dnumRows ; ++k )
		    {
			////submatr("&",i,j) += conj((*this)(k,i))*((*this)(k,j));
			//submatr("&",j,i) += conj((*this)(k,j))*((*this)(k,i));

			leftargx  = conj((*this)(k,j));
			rightargx = ((*this)(k,i));

			submatr("&",j,i) += (leftargx*rightargx);
		    }
		}
	    }

	    //submatr.conj();
	    //submatr.transpose();

            res = *this;
            leftmult(res,submatr.inve(subinv));
	}

	else
	{
	    // Wide matrix
	    //
	    // A+ = (A*).((A.A*)+)
	    // A+* = ((A.A*)+*).A
	    //     = (((A.A*)*)+).A (return this)

	    Matrix<T> submatr(dnumRows,dnumRows);
	    Matrix<T> subinv(dnumRows,dnumRows);
	    T leftargx;
            T rightargx;

	    int i,j,k;

	    for ( i = 0 ; i < dnumRows ; ++i )
	    {
		for ( j = 0 ; j < dnumRows ; ++j )
		{
		    for ( k = 0 ; k < dnumCols ; ++k )
		    {
			////submatr("&",i,j) += ((*this)(i,k))*conj((*this)(j,k));
			//submatr("&",j,i) += ((*this)(j,k))*conj((*this)(i,k));

			leftargx  = ((*this)(j,k));
			rightargx = conj((*this)(i,k));

			submatr("&",j,i) += (leftargx*rightargx);
		    }
		}
	    }

	    //submatr.conj();
            //submatr.transpose();

            res = *this;
            rightmult(submatr.inve(subinv),res);
	}
    }

    else
    {
        res = *this;
        res.transpose();
    }

    return res;
}

template <class T>
Matrix<T> &Matrix<T>::inveSymm(Matrix<T> &res) const
{
    NiceAssert( isSquare() );

    res.resize(dnumRows,dnumCols);

    if ( dnumRows )
    {
	T matdet;
	matdet = det();

	if ( dnumRows == 1 )
	{
	    res("&",0,0) = inv(matdet);
	}

        else if ( dnumRows == 2 )
        {
            res("&",0,0) = (*this)(1,1)*inv(matdet);
            res("&",0,1) = -(*this)(0,1)*inv(matdet);
            res("&",1,0) = -(*this)(1,0)*inv(matdet);
            res("&",1,1) = (*this)(0,0)*inv(matdet);
        }

	else
	{
	    int i,j;

	    for ( i = 0 ; i < dnumRows ; ++i )
	    {
		for ( j = 0 ; j <= i ; ++j )
		{
                    res("&",i,j) = (cofactor(j,i)*inv(matdet));

                    if ( i != j )
                    {
                        res("&",j,i) = conj(res("&",i,j));
                    }
		}
	    }
	}
    }

    return res;
}

template <> inline Matrix<double> &Matrix<double>::inveSymm(Matrix<double> &res) const;
template <> inline Matrix<double> &Matrix<double>::inveSymm(Matrix<double> &res) const
{
    NiceAssert( isSquare() );

    res.resize(dnumRows,dnumCols);

    if ( dnumRows )
    {
	double matdet = det();

	if ( dnumRows == 1 )
	{
	    res("&",0,0) = 1/matdet;
	}

        else if ( dnumRows == 2 )
        {
            res("&",0,0) =  (*this).v(1,1)/matdet;
            res("&",0,1) = -(*this).v(0,1)/matdet;
            res("&",1,0) = -(*this).v(1,0)/matdet;
            res("&",1,1) =  (*this).v(0,0)/matdet;
        }

	else
	{
	    int i,j;

	    for ( i = 0 ; i < dnumRows ; ++i )
	    {
		for ( j = 0 ; j <= i ; ++j )
		{
                    res("&",i,j) = cofactor(j,i)/matdet;
                    res("&",j,i) = res.v(i,j);
		}
	    }
	}
    }

    return res;
}

template <class T>
int Matrix<T>::LUPDecompose(Matrix<T> &res, Vector<int> &p, double ztol) const
{
    res = *this;

    int s = res.LUPDecompose(p,ztol);

    //p = *(res.pbuff);

    return s;
}

template <class T>
int Matrix<T>::LUPDecompose(double ztol)
{
    static thread_local Vector<int> p("&",2);

    return LUPDecompose(ztol,p);
}

template <class T>
int Matrix<T>::LUPDecompose(Vector<int> &p, double ztol)
{
    NiceAssert( isSquare() );

    int s = 0;

    // Based on wikipedia

    int N = dnumRows;
    int i,j,k,imax;
    double maxA,absA;

    retVector<int> tmpva;

    //if ( !pbuff )
    //{
    //    MEMNEW(pbuff,Vector<int>);
    //}
    //
    // *pbuff = cntintvec(N,tmpva);
    p = cntintvec(N,tmpva);

    for ( i = 0 ; i < N ; ++i )
    {
        maxA = 0.0;
        imax = i;

        for ( k = i ; k < N ; ++k )
        {
            //absA = abs2((double) (*this)((*pbuff).v(k),i));
            absA = abs2((double) (*this)(p.v(k),i));

            if ( absA > maxA )
            {
                maxA = absA;
                imax = k;
            }
        }

        if ( maxA < ztol )
        {
            // Degenerate case, computation failed

            return -1;
        }

        if ( imax != i )
        {
            //(*pbuff).squareswap(i,imax);
            p.squareswap(i,imax);

            ++s;
        }

        for ( j = i+1 ; j < N ; ++j )
        {
            //(*this)("&",(*pbuff).v(j),i) /= (*this)((*pbuff).v(i),i);
            (*this)("&",p.v(j),i) /= (*this)(p.v(i),i);

            for ( k = i+1 ; k < N ; ++k )
            {
                //(*this)("&",(*pbuff).v(j),k) -= (*this)((*pbuff).v(j),i)*(*this)((*pbuff).v(i),k);
                (*this)("&",p.v(j),k) -= (*this)(p.v(j),i)*(*this)(p.v(i),k);
            }
        }
    }

    return s;
}

template <class T>
void Matrix<T>::tridiag(Vector<T> &d, Vector<T> &e, Vector<T> &e2)
{
    Matrix<T> &a = *this;

/*
      subroutine tred1(nm,n,a,d,e,e2)
c
      integer i,j,k,l,n,ii,nm,jp1
      double precision a(nm,n),d(n),e(n),e2(n)
      double precision f,g,h,scale
c
c     this subroutine is a translation of the algol procedure tred1,
c     num. math. 11, 181-195(1968) by martin, reinsch, and wilkinson.
c     handbook for auto. comp., vol.ii-linear algebra, 212-226(1971).
c
c     this subroutine reduces a real symmetric matrix
c     to a symmetric tridiagonal matrix using
c     orthogonal similarity transformations.
c
c     on input
c
c        nm must be set to the row dimension of two-dimensional
c          array parameters as declared in the calling program
c          dimension statement.
c
c        n is the order of the matrix.
c
c        a contains the real symmetric input matrix.  only the
c          lower triangle of the matrix need be supplied.
c
c     on output
c
c        a contains information about the orthogonal trans-
c          formations used in the reduction in its strict lower
c          triangle.  the full upper triangle of a is unaltered.
c
c        d contains the diagonal elements of the tridiagonal matrix.
c
c        e contains the subdiagonal elements of the tridiagonal
c          matrix in its last n-1 positions.  e(1) is set to zero.
c
c        e2 contains the squares of the corresponding elements of e.
c          e2 may coincide with e if the squares are not needed.
c
c     questions and comments should be directed to burton s. garbow,
c     mathematics and computer science div, argonne national laboratory
c
c     this version dated august 1983.
c
c     ------------------------------------------------------------------
c
*/

////void tred1(int nm, int n, Matrix<double> &a, Vector<double> &d, Vector<double> &e, Vector<double> &e2);
//void tred1(Matrix<double> &a, Vector<double> &d, Vector<double> &e, Vector<double> &e2);


    NiceAssert( a.isSquare() );

    int n = a.size();

    // Save diagonal

    int ix,jx;

    Vector<T> adiag(n);

    for ( ix = 0 ; ix < n ; ++ix )
    {
        adiag.set(ix,a(ix,ix));
    }

    d.resize(n);
    e.resize(n);
    e2.resize(n);

    int i,j,k,l,ii,jp1;
    double f,g,h,scale;

    for ( i = 1 ; i <= n ; ++i )
    {
        d("&",i-1) = a(n-1,i-1);
        a("&",n-1,i-1) = a(i-1,i-1);
    }

    //c     .......... for i=n step -1 until 1 do -- ..........

    for ( ii = 1 ; ii <= n ; ++ii )
    {
        i = n + 1 - ii;
        l = i - 1;
        h = 0.0;
        scale = 0.0;

        if ( l < 1 )
        {
            goto l130;
        }

        //c     .......... scale row (algol tol then not needed) ..........

        for ( k = 1 ; k <= l ; ++k )
        {
            scale = scale + abs2(d(k-1));
        }

        if ( scale != 0.0 )
        {
            goto l140;
        }

        for ( j = 1 ; j <= l ; ++j )
        {
            d("&",j-1) = a(l-1,j-1);
            a("&",l-1,j-1) = a(i-1,j-1);
            a("&",i-1,j-1) = 0.0;
        }

l130:
        e("&",i-1) = 0.0;
        e2("&",i-1) = 0.0;

        goto l300;

l140:
        for ( k = 1 ; k <= l ; ++k )
        {
            d("&",k-1) = d(k-1) / scale;
            h = h + d(k-1) * d(k-1);
        }

        e2("&",i-1) = scale * scale * h;
        f = d(l-1);
        g = -dsign(sqrt(h),f);
        e("&",i-1) = scale * g;
        h = h - f * g;
        d("&",l-1) = f - g;

        if ( l == 1 )
        {
            goto l285;
        }

        //c     .......... form a*u ..........

        for ( j = 1 ; j <= l ; ++j )
        {
            e("&",j-1) = 0.0;
        }

        for ( j = 1 ; j <= l ; ++j )
        {
            f = d(j-1);

            g = e(j-1) + a(j-1,j-1) * f;
            jp1 = j + 1;

            if ( l < jp1 )
            {
                goto l220;
            }

            for ( k = jp1 ; k <= l ; ++k )
            {
                g = g + a(k-1,j-1) * d(k-1);
                e("&",k-1) = e(k-1) + a(k-1,j-1) * f;
            }

l220:
            e("&",j-1) = g;
        }


        //c     .......... form p ..........

        f = 0.0;

        for ( j = 1 ; j <= l ; ++j )
        {
            e("&",j-1) = e(j-1) / h;
            f = f + e(j-1) * d(j-1);
        }

        h = f / (h + h);

        //c     .......... form q ..........

        for ( j = 1 ; j <= l ; ++j )
        {
            e("&",j-1) = e(j-1) - h * d(j-1);
        }

        //c     .......... form reduced a ..........

        for ( j = 1 ; j <= l ; ++j )
        {
            f = d(j-1);
            g = e(j-1);

            for ( k = j ; k <= l ; ++k )
            {
                a("&",k-1,j-1) = a(k-1,j-1) - f * e(k-1) - g * d(k-1);
            }
        }

l285:
        for ( j = 1 ; j <= l ; ++j )
        {
            f = d(j-1);
            d("&",j-1) = a(l-1,j-1);
            a("&",l-1,j-1) = a(i-1,j-1);
            a("&",i-1,j-1) = f * scale;
        }
    }

    // Reconstruct lower-triangular part

l300:
    for ( ix = 0 ; ix < n ; ++ix )
    {
        a("&",ix,ix) = adiag(ix);

        for ( jx = ix+1 ; jx < n ; ++jx )
        {
            a("&",jx,ix) = a(ix,jx);
        }
    }
}

template <class T>
void Matrix<T>::tridiag(Vector<T> &d, Vector<T> &e, Matrix<T> &z) const
{
    const Matrix<T> &a = *this;

/*
      subroutine tred2(nm,n,a,d,e,z)
c
      integer i,j,k,l,n,ii,nm,jp1
      double precision a(nm,n),d(n),e(n),z(nm,n)
      double precision f,g,h,hh,scale
c
c     this subroutine is a translation of the algol procedure tred2,
c     num. math. 11, 181-195(1968) by martin, reinsch, and wilkinson.
c     handbook for auto. comp., vol.ii-linear algebra, 212-226(1971).
c
c     this subroutine reduces a real symmetric matrix to a
c     symmetric tridiagonal matrix using and accumulating
c     orthogonal similarity transformations.
c
c     on input
c
c        nm must be set to the row dimension of two-dimensional
c          array parameters as declared in the calling program
c          dimension statement.
c
c        n is the order of the matrix.
c
c        a contains the real symmetric input matrix.  only the
c          lower triangle of the matrix need be supplied.
c
c     on output
c
c        d contains the diagonal elements of the tridiagonal matrix.
c
c        e contains the subdiagonal elements of the tridiagonal
c          matrix in its last n-1 positions.  e(1) is set to zero.
c
c        z contains the orthogonal transformation matrix
c          produced in the reduction.
c
c        a and z may coincide.  if distinct, a is unaltered.
c
c     questions and comments should be directed to burton s. garbow,
c     mathematics and computer science div, argonne national laboratory
c
c     this version dated august 1983.
c
c     ------------------------------------------------------------------
c
*/

////void tred2(int nm, int n, Matrix<double> &a, Vector<double> &d, Vector<double> &e, Matrix<double> &z);
//void tred2(const Matrix<double> &a, Vector<double> &d, Vector<double> &e, Matrix<double> &z);

    NiceAssert( a.isSquare() );

    int n = a.numRows();

    d.resize(n);
    e.resize(n);
    z.resize(n,n);

    int i,j,k,l,ii,jp1;
    double f,g,h,hh,scale;

    for ( i = 1 ; i <= n ; ++i )
    {
        for ( j = i ; j <= n ; ++j )
        {
            z("&",j-1,i-1) = a(j-1,i-1);
        }

//a("&",n-1,i-1) = a(i-1,i-1);
        d("&",i-1) = a(n-1,i-1);
    }

    if ( n == 1 )
    {
        goto l510;
    }

    //c     .......... for i=n step -1 until 2 do -- ..........

    for ( ii = 2 ; ii <= n ; ++ii )
    {
        i = n + 2 - ii;
        l = i - 1;
        h = 0.0;
        scale = 0.0;

        if ( l < 2 )
        {
            goto l130;
        }

        //c     .......... scale row (algol tol then not needed) ..........

        for ( k = 1 ; k <= l ; ++k )
        {
            scale = scale + abs2(d(k-1));
        }

        if ( scale != 0.0 )
        {
            goto l140;
        }

l130:
        e("&",i-1) = d(l-1);

        for ( j = 1 ; j <= l ; ++j )
        {
            d("&",j-1) = z(l-1,j-1);
            z("&",i-1,j-1) = 0.0;
            z("&",j-1,i-1) = 0.0;
        }

        goto l290;

l140:
        for ( k = 1 ; k <= l ; ++k )
        {
            d("&",k-1) = d(k-1) / scale;
            h = h + d(k-1) * d(k-1);
        }

        f = d(l-1);
        g = -dsign(sqrt(h),f);
        e("&",i-1) = scale * g;
        h = h - f * g;
        d("&",l-1) = f - g;

        //c     .......... form a*u ..........

        for ( j = 1 ; j <= l ; ++j )
        {
            e("&",j-1) = 0.0;
        }

        for ( j = 1 ; j <= l ; ++j )
        {
            f = d(j-1);
            z("&",j-1,i-1) = f;
            g = e(j-1) + z(j-1,j-1) * f;
            jp1 = j + 1;

            if ( l < jp1 )
            {
                goto l220;
            }

            for ( k = jp1 ; k <= l ; ++k )
            {
                g = g + z(k-1,j-1) * d(k-1);
                e("&",k-1) = e(k-1) + z(k-1,j-1) * f;
            }

l220:
            e("&",j-1) = g;
        }

        //c     .......... form p ..........

        f = 0.0;

        for ( j = 1 ; j <= l ; ++j )
        {
            e("&",j-1) = e(j-1) / h;
            f = f + e(j-1) * d(j-1);
        }

        hh = f / ( h + h );

        //c     .......... form q ..........

        for ( j = 1 ; j <= l ; ++j )
        {
            e("&",j-1) = e(j-1) - hh * d(j-1);
        }

        //c     .......... form reduced a ..........
        for ( j = 1 ; j <= l ; ++j )
        {
            f = d(j-1);
            g = e(j-1);

            for ( k = j ; k <= l ; ++k )
            {
                z("&",k-1,j-1) = z(k-1,j-1) - f * e(k-1) - g * d(k-1);
            }

            d("&",j-1) = z(l-1,j-1);
            z("&",i-1,j-1) = 0.0;
        }

l290:
        d("&",i-1) = h;
    }

    //c     .......... accumulation of transformation matrices ..........

    for ( i = 2 ; i <= n ; ++i )
    {
        l = i - 1;
        z("&",n-1,l-1) = z(l-1,l-1);
        z("&",l-1,l-1) = 1.0;
        h = d(i-1);

        if ( h == 0.0 )
        {
            goto l380;
        }

        for ( k = 1 ; k <= l ; ++k )
        {
            d("&",k-1) = z(k-1,i-1) / h;
        }

        for ( j = 1 ; j <= l ; ++j )
        {
            g = 0.0;

            for ( k = 1 ; k <= l ; ++k )
            {
                g = g + z(k-1,i-1) * z(k-1,j-1);
            }


            for ( k = 1 ; k <= l ; ++k )
            {
                z("&",k-1,j-1) = z(k-1,j-1) - g * d(k-1);
            }
        }

l380:
        for ( k = 1 ; k <= l ; ++k )
        {
            z("&",k-1,i-1) = 0.0;
        }
    }

l510:
    for ( i = 1 ; i <= n ; ++i )
    {
        d("&",i-1) = z(n-1,i-1);
        z("&",n-1,i-1) = 0.0;
    }

    z("&",n-1,n-1) = 1.0;
    e("&",1-1) = 0.0;
}

template <class T>
int tql1(Vector<T> &d, Vector<T> &e)
{
    NiceAssert( d.size() == e.size() );

    int n = d.size();

    int ierr = 0;

    int i,j,l,m,ii,l1,l2,mml;
    T c,c2,c3,dl1,el1,f,g,h,p,r,s,s2,tst1,tst2;

    s2 = 0.0; // Not strictly necessary (s2 is set in loop before being used as mml >= 1 is guaranteed), but stops gcc from complaining
    c3 = 0.0;

    if ( n == 1 )
    {
        goto l1001;
    }

    for ( i = 2 ; i <= n ; ++i )
    {
        e("&",i-1-1) = e(i-1);
    }

    f = 0.0;
    tst1 = 0.0;
    e("&",n-1) = 0.0;

    for ( l = 1 ; l <= n ; ++l )
    {
        j = 0;
        h = abs2(d(l-1)) + abs2(e(l-1));

        if ( tst1 < h )
        {
            tst1 = h;
        }

        //c     .......... look for small sub-diagonal element ..........

        for ( m = l ; m <= n ; ++m )
        {
            tst2 = tst1 + abs2(e(m-1));

            if ( tst2 == tst1 )
            {
                break;
            }

            //c     .......... e(n) is always zero, so there is no exit
            //c                through the bottom of the loop ..........
        }

        if ( m == l )
        {
            goto l210;
        }

l130:
        if ( j == 30 )
        {
            goto l1000;
        }

        j = j + 1;

        //c     .......... form shift ..........

        l1 = l + 1;
        l2 = l1 + 1;
        g = d(l-1);
        p = ( d(l1-1) - g ) / ( 2.0 * e(l-1) );
        r = sppythag(p,1.0);
        d("&",l-1) = e(l-1) / ( p + dsign(r,p) );
        d("&",l1-1) = e(l-1) * ( p + dsign(r,p) );
        dl1 = d(l1-1);
        h = g - d(l-1);

        if ( l2 > n )
        {
            goto l145;
        }

        for ( i = l2 ; i <= n ; ++i )
        {
            d("&",i-1) -= h;
        }

l145:
        f += h;

        //c     .......... ql transformation ..........

        p = d(m-1);
        c = 1.0;
        c2 = c;
        el1 = e(l1-1);
        s = 0.0;
        mml = m - l;

        //c     .......... for i=m-1 step -1 until l do -- ..........

        for ( ii = 1 ; ii <= mml ; ++ii )
        {
            c3 = c2;
            c2 = c;
            s2 = s;
            i = m - ii;
            g = c * e(i-1);
            h = c * p;
            r = sppythag(p,e(i-1));
            e("&",i+1-1) = s * r;
            s = e(i-1) / r;
            c = p / r;
            p = c * d(i-1) - s * g;
            d("&",i+1-1) = h + s * ( c * g + s * d(i-1) );
        }

        p = -s * s2 * c3 * el1 * e(l-1) / dl1;
        e("&",l-1) = s * p;
        d("&",l-1) = c * p;
        tst2 = tst1 + abs2(e(l-1));

        if ( tst2 > tst1 )
        {
            goto l130;
        }

l210:
        p = d(l-1) + f;

        //c     .......... order eigenvalues ..........

        if ( l == 1 )
        {
            goto l250;
        }

        //c     .......... for i=l step -1 until 2 do -- ..........

        for ( ii = 2 ; ii <= l ; ++ii )
        {
            i = l + 2 - ii;

            if ( p >= d(i-1-1) )
            {
                goto l270;
            }

            d("&",i-1) = d(i-1-1);
        }

l250:
        i = 1;
l270:
        d("&",i-1) = p;

    }

    goto l1001;

    //c     .......... set error -- no convergence to an
    //c                eigenvalue after 30 iterations ..........

l1000:
    ierr = l;
l1001:
    return ierr;
}

template <class T>
int tql2(Vector<T> &d, Vector<T> &e, Matrix<T> &z)
{
    NiceAssert( z.isSquare() );
    NiceAssert( d.size() == e.size() );
    NiceAssert( d.size() == z.size() );

    int n = d.size();

    int i,j,k,l,m,ii,l1,l2,mml;
    T c,c2,c3,dl1,el1,f,g,h,p,r,s,s2,tst1,tst2;

    s2 = 0.0; // See comment in tql1
    c3 = 0.0;

    int ierr = 0;

    if ( n == 1 )
    {
        goto l1001;
    }

    for ( i = 2 ; i <= n ; ++i )
    {
        e("&",i-1-1) = e(i-1);
    }

    f = 0.0;
    tst1 = 0.0;
    e("&",n-1) = 0.0;

    for ( l = 1 ; l <= n ; ++l )
    {
        j = 0;
        h = abs2(d(l-1)) + abs2(e(l-1));

        if ( tst1 < h )
        {
            tst1 = h;
        }

        //c     .......... look for small sub-diagonal element ..........

        for ( m = l ; m <= n ; ++m )
        {
            tst2 = tst1 + abs2(e(m-1));

            if ( tst2 == tst1 )
            {
                break;
            }

            //c     .......... e(n) is always zero, so there is no exit
            //c                through the bottom of the loop ..........
        }

        if ( m == l )
        {
            goto l220;
        }

l130:
        if ( j == 30 )
        {
            goto l1000;
        }

        ++j;

        //c     .......... form shift ..........

        l1 = l + 1;
        l2 = l1 + 1;
        g = d(l-1);
        p = ( d(l1-1) - g ) / ( 2.0 * e(l-1) );
        r = sppythag(p,1.0);
        d("&",l-1) = e(l-1) / ( p + dsign(r,p) );
        d("&",l1-1) = e(l-1) * ( p + dsign(r,p) );
        dl1 = d(l1-1);
        h = g - d(l-1);

        if (l2 > n)
        {
            goto l145;
        }

        for ( i = l2 ; i <= n ; ++i )
        {
            d("&",i-1) -= h;
        }

l145:
        f = f + h;

        //c     .......... ql transformation ..........

        p = d(m-1);
        c = 1.0;
        c2 = c;
        el1 = e(l1-1);
        s = 0.0;
        mml = m - l;

        //c     .......... for i=m-1 step -1 until l do -- ..........

        for ( ii = 1 ; ii <= mml ; ++ii )
        {
            c3 = c2;
            c2 = c;
            s2 = s;
            i = m - ii;
            g = c * e(i-1);
            h = c * p;
            r = sppythag(p,e(i-1));
            e("&",i+1-1) = s * r;
            s = e(i-1) / r;
            c = p / r;
            p = c * d(i-1) - s * g;
            d("&",i+1-1) = h + s * ( c * g + s * d(i-1) );

            //c     .......... form vector ..........

            for ( k = 1 ; k <= n ; ++k )
            {
                h = z(k-1,i+1-1);
                z("&",k-1,i+1-1) = s * z(k-1,i-1) + c * h;
                z("&",k-1,i-1) = c * z(k-1,i-1) - s * h;
            }
        }

         p = -s * s2 * c3 * el1 * e(l-1) / dl1;
         e("&",l-1) = s * p;
         d("&",l-1) = c * p;
         tst2 = tst1 + abs2(e(l-1));

         if ( tst2 > tst1 )
         {
             goto l130;
         }

l220:
        d("&",l-1) += f;
    }

    //c     .......... order eigenvalues and eigenvectors ..........

    for ( ii = 2 ; ii <= n ; ++ii )
    {
        i = ii - 1;
        k = i;
        p = d(i-1);

        for ( j = ii ; j <= n ; ++j )
        {
            if ( d(j-1) >= p )
            {
                break;
            }

            k = j;
            p = d(j-1);
        }

        if ( k == i )
        {
            break;
        }

        d("&",k-1) = d(i-1);
        d("&",i-1) = p;

        for ( j = 1 ; j <= n ; ++j )
        {
            p = z(j-1,i-1);
            z("&",j-1,i-1) = z(j-1,k-1);
            z("&",j-1,k-1) = p;
        }
    }

    goto l1001;

    //c     .......... set error -- no convergence to an
    //c                eigenvalue after 30 iterations ..........

l1000:
    ierr = l;
l1001:
    return ierr;
}

template <class T>
int Matrix<T>::eig(Vector<T> &w, Vector<T> &fv1, Vector<T> &fv2) const
{
//    const Matrix<T> &a = *this;

/*
      subroutine rs(nm,n,a,w,matz,z,fv1,fv2,ierr)
c
      integer n,nm,ierr,matz
      double precision a(nm,n),w(n),z(nm,n),fv1(n),fv2(n)
c
c     this subroutine calls the recommended sequence of
c     subroutines from the eigensystem subroutine package (eispack)
c     to find the eigenvalues and eigenvectors (if desired)
c     of a real symmetric matrix.
c
c     on input
c
c        nm  must be set to the row dimension of the two-dimensional
c        array parameters as declared in the calling program
c        dimension statement.
c
c        n  is the order of the matrix  a.
c
c        a  contains the real symmetric matrix.
c
c        matz  is an integer variable set equal to zero if
c        only eigenvalues are desired.  otherwise it is set to
c        any non-zero integer for both eigenvalues and eigenvectors.
c
c     on output
c
c        w  contains the eigenvalues in ascending order.
c
c        z  contains the eigenvectors if matz is not zero (eigenvectors are in columns).
c
c        ierr  is an integer output variable set equal to an error
c           completion code described in the documentation for tqlrat
c           and tql2.  the normal completion code is zero.
c
c        fv1  and  fv2  are temporary storage arrays.
c
c     questions and comments should be directed to burton s. garbow,
c     mathematics and computer science div, argonne national laboratory
c
c     this version dated august 1983.
c
c     ------------------------------------------------------------------
*/

    //c     .......... find eigenvalues only ..........

    tridiag(w,fv1,fv2);
//NOT WORKING?Matrix<double> z;
//NOT WORKING?(void) fv2;
//NOT WORKING?     tridiag(w,fv1,z);

    //*  tqlrat encounters catastrophic underflow on the Vax
    //*     call  tqlrat(n,w,fv2,ierr)

    return tql1(w,fv1);
}

template <class T>
int Matrix<T>::eig(Vector<T> &w, Matrix<T> &z, Vector<T> &fv1) const
{
//    const Matrix<T> &a = *this;

/*
      subroutine rs(nm,n,a,w,matz,z,fv1,fv2,ierr)
c
      integer n,nm,ierr,matz
      double precision a(nm,n),w(n),z(nm,n),fv1(n),fv2(n)
c
c     this subroutine calls the recommended sequence of
c     subroutines from the eigensystem subroutine package (eispack)
c     to find the eigenvalues and eigenvectors (if desired)
c     of a real symmetric matrix.
c
c     on input
c
c        nm  must be set to the row dimension of the two-dimensional
c        array parameters as declared in the calling program
c        dimension statement.
c
c        n  is the order of the matrix  a.
c
c        a  contains the real symmetric matrix.
c
c        matz  is an integer variable set equal to zero if
c        only eigenvalues are desired.  otherwise it is set to
c        any non-zero integer for both eigenvalues and eigenvectors.
c
c     on output
c
c        w  contains the eigenvalues in ascending order.
c
c        z  contains the eigenvectors if matz is not zero (eigenvectors are in columns).
c
c        ierr  is an integer output variable set equal to an error
c           completion code described in the documentation for tqlrat
c           and tql2.  the normal completion code is zero.
c
c        fv1  and  fv2  are temporary storage arrays.
c
c     questions and comments should be directed to burton s. garbow,
c     mathematics and computer science div, argonne national laboratory
c
c     this version dated august 1983.
c
c     ------------------------------------------------------------------
*/

    //c     .......... find both eigenvalues and eigenvectors ..........

     tridiag(w,fv1,z);

     return tql2(w,fv1,z);
}

template <class T>
int Matrix<T>::projpsd(Matrix<T> &res, Vector<T> &w, Matrix<T> &z, Vector<T> &fv1, int method) const
{
    const Matrix<T> &a = *this;

    int n = a.numRows();

    w.resize(n);
    z.resize(n,n);
    fv1.resize(n);

    int ierr = eig(w,z,fv1);

    if ( !ierr )
    {
        res.resize(n,n);
        res = 0.0;

        int i,j,k;

        for ( k = 0 ; k < n ; ++k )
        {
            double wk = method ? abs2(w(k)) : w(k);

            if ( wk > 0.0 )
            {
                for ( i = 0 ; i < n ; ++i )
                {
                    for ( j = 0 ; j < n ; ++j )
                    {
                        res("&",i,j) += wk*z(i,k)*z(j,k);
                    }
                }
            }
        }
    }

    return ierr;
}

template <class T>
int Matrix<T>::projnsd(Matrix<T> &res, Vector<T> &w, Matrix<T> &z, Vector<T> &fv1, int method) const
{
    const Matrix<T> &a = *this;

    NiceAssert( a.isSquare() );

    int n = a.numRows();

    w.resize(n);
    z.resize(n,n);
    fv1.resize(n);

    int ierr = eig(w,z,fv1);

    if ( !ierr )
    {
        res.resize(n,n);
        res = 0.0;

        int i,j,k;

        for ( k = 0 ; k < n ; ++k )
        {
            double wk = method ? -abs2(w(k)) : w(k);

            if ( wk < 0.0 )
            {
                for ( i = 0 ; i < n ; ++i )
                {
                    for ( j = 0 ; j < n ; ++j )
                    {
                        res("&",i,j) += wk*z(i,k)*z(j,k);
                    }
                }
            }
        }
    }

    return ierr;
}

template <class T>
Matrix<T> Matrix<T>::adj(void) const
{
    Matrix<T> res;

    adj(res);

    return res;
}

template <class T>
Matrix<T> Matrix<T>::inve(void) const
{
    Matrix<T> res;

    inve(res);

    return res;
}

template <class T>
Matrix<T> Matrix<T>::inveSymm(void) const
{
    Matrix<T> res;

    inveSymm(res);

    return res;
}

template <class T>
double Matrix<T>::getColNorm(int j) const
{
    int i;
    double result = 0;

    if ( numRows() )
    {
	result = norm2((*this)(0,j));

	//if ( numRows() > 1 )
	{
	    for ( i = 1 ; i < numRows() ; ++i )
	    {
		result += norm2((*this)(i,j));
	    }
	}
    }

    return result;
}

template <class T>
double Matrix<T>::getRowNorm(int i) const
{
    int j;
    double result = 0;

    if ( numCols() )
    {
	result = norm2((*this)(i,0));

	//if ( numCols() > 1 )
	{
	    for ( j = 1 ; j < numCols() ; ++j )
	    {
		result += norm2((*this)(i,j));
	    }
	}
    }

    return result;
}

template <class T>
double Matrix<T>::getRowColNorm(void) const
{
    int i,j;
    double result = 0;

    if ( numCols() && numRows() )
    {
        for ( i = 0 ; i < numCols() ; ++i )
        {
            for ( j = 0 ; j < numCols() ; ++j )
	    {
		result += norm2((*this)(i,j));
	    }
	}
    }

    return result;
}

template <class T>
double Matrix<T>::getColAbs(int j) const
{
    int i;
    double result = 0;

    if ( numRows() )
    {
        result = abs2((*this)(0,j));

	//if ( numRows() > 1 )
	{
	    for ( i = 1 ; i < numRows() ; ++i )
	    {
                result += abs2((*this)(i,j));
	    }
	}
    }

    return result;
}

template <class T>
double Matrix<T>::getRowAbs(int i) const
{
    int j;
    double result = 0;

    if ( numCols() )
    {
        result = abs2((*this)(i,0));

	//if ( numCols() > 1 )
	{
	    for ( j = 1 ; j < numCols() ; ++j )
	    {
                result += abs2((*this)(i,j));
	    }
	}
    }

    return result;
}

template <class T>
double Matrix<T>::getRowColAbs(void) const
{
    int i,j;
    double result = 0;

    if ( numCols() && numRows() )
    {
        for ( i = 0 ; i < numCols() ; ++i )
        {
            for ( j = 0 ; j < numCols() ; ++j )
	    {
                result += abs2((*this)(i,j));
	    }
	}
    }

    return result;
}

template <class T>
const Vector<T> &Matrix<T>::rowsum(Vector<T> &res) const
{
    res.resize(numCols());
    res.zero();

    //if ( numRows() )
    {
        for ( int i = 0 ; i < numRows() ; ++i )
        {
            res += (*this)(i);
        }
    }

    return res;
}

template <class T>
const Vector<T> &Matrix<T>::colsum(Vector<T> &res) const
{
    res.resize(numRows());
    res.zero();

    if ( numCols() && numRows() )
    {
        int i,j;

        for ( i = 0 ; i < numCols() ; ++i )
        {
            for ( j = 0 ; j < numRows() ; ++j )
            {
                res("&",j) += (*this)(j,i);
            }
        }
    }

    return res;
}

template <class T>
const T &Matrix<T>::vertsum(int j, T &res) const
{
    setzero(res);

    if ( numRows() && numCols() )
    {
        int i;

        for ( i = 0 ; i < numRows() ; ++i )
        {
            res += (*this)(i,j);
        }
    }

    return res;
}

template <class T>
const T &Matrix<T>::horizsum(int i, T &res) const
{
    setzero(res);

    if ( numRows() && numCols() )
    {
        int j;

        for ( j = 0 ; j < numCols() ; ++j )
        {
            res += (*this)(i,j);
        }
    }

    return res;
}

template <class T>
template <class S> Matrix<T> &Matrix<T>::scale(const S &a)
{
//    if ( numRows() && numCols() )
//    {
//	int i;
//
//        retVector<T> tmpva;
//
//	for ( i = 0 ; i < numRows() ; ++i )
//	{
//	    (*this)("&",i,tmpva).scale(a);
//        }
//    }

    if ( numCols() && numRows() )
    {
        int i,j;

        for ( i = 0 ; i < numRows() ; ++i )
        {
            for ( j = 0 ; j < numCols() ; ++j )
            {
                (*this)("&",i,j) *= a;
            }
        }
    }

    return *this;
}

template <class T>
template <class S> Matrix<T> &Matrix<T>::scaleAdd(const S &a, const Matrix<T> &b)
{
//    NiceAssert( ( numRows() == b.numRows() ) || isEmpty() || b.isEmpty() );
//    NiceAssert( ( numCols() == b.numCols() ) || isEmpty() || b.isEmpty() );
//
//    if ( numRows() && numCols() && !isEmpty() && !(b.isEmpty()) )
//    {
//        if ( shareBase(&b) )
//        {
//	    Matrix<T> temp(b);
//
//            scaleAdd(a,temp);
//	}
//
//	else
//	{
//	    int i;
//
//            retVector<T> tmpva;
//            retVector<T> tmpvb;
//
//	    for ( i = 0 ; i < numRows() ; ++i )
//	    {
//		(*this)("&",i,tmpva).scaleAdd(a,b(i,tmpvb));
//	    }
//        }
//    }
//
//    else if ( isEmpty() )
//    {
//        resize(b.numRows(),b.numCols());
//        zero();
//        scaleAdd(a,b);
//    }
//
//    return *this;



    if ( shareBase(&b) )
    {
        Matrix<T> temp(b);

        scaleAdd(a,temp);
    }

    else
    {
        if ( !size() )
	{
            resize(b.numRows(),b.numCols());
            zero();
	}

        NiceAssert( ( numRows() == b.numRows() ) );
        NiceAssert( ( numCols() == b.numCols() ) );

        if ( numRows() && numCols() )
	{
            int i,j;

            for ( i = 0 ; i < numRows() ; ++i )
	    {
                for ( j = 0 ; j < numCols() ; ++j )
                {
                    (*this)("&",i,j) += (a*(b(i,j)));
                }
            }
	}
    }

    return *this;
}

































// Complexity





// Various functions

template <class T>
const T &max(const Matrix<T> &right_op, int &ii, int &jj)
{
    NiceAssert( right_op.numRows() && right_op.numCols() );

    int i,j;

    ii = 0;
    jj = 0;

    retVector<T> tmpva;
    retVector<T> tmpvc;

    for ( i = 0 ; i < right_op.numRows() ; ++i )
    {
        j = 0;

        if ( max(right_op(i,tmpva,tmpvc),j) > right_op(ii,jj) )
	{
            ii = i;
            jj = j;
	}
    }

    return right_op(ii,jj);
}

template <class T>
const T &min(const Matrix<T> &right_op, int &ii, int &jj)
{
    NiceAssert( right_op.numRows() && right_op.numCols() );

    int i,j;

    ii = 0;
    jj = 0;

    retVector<T> tmpva;
    retVector<T> tmpvc;

    for ( i = 0 ; i < right_op.numRows() ; ++i )
    {
        j = 0;

        if ( max(right_op(i,tmpva,tmpvc),j) < right_op(ii,jj) )
	{
            ii = i;
            jj = j;
	}
    }

    return right_op(ii,jj);
}

template <class T>
const T &maxdiag(const Matrix<T> &right_op, int &ii, int &jj)
{
    NiceAssert( right_op.numRows() && right_op.numCols() );
    NiceAssert( right_op.numRows() == right_op.numCols() );

    int i;

    ii = 0;
    jj = 0;

    for ( i = 0 ; i < right_op.numRows() ; ++i )
    {
        if ( right_op(i,i) > right_op(ii,jj) )
	{
            ii = i;
            jj = i;
	}
    }

    return right_op(ii,jj);
}

template <class T>
const T &mindiag(const Matrix<T> &right_op, int &ii, int &jj)
{
    NiceAssert( right_op.numRows() && right_op.numCols() );
    NiceAssert( right_op.numRows() == right_op.numCols() );

    int i;

    ii = 0;
    jj = 0;

    for ( i = 0 ; i < right_op.numRows() ; ++i )
    {
        if ( right_op(i,i) < right_op(ii,jj) )
	{
            ii = i;
            jj = i;
	}
    }

    return right_op(ii,jj);
}

template <class T>
T maxabs(const Matrix<T> &right_op, int &ii, int &jj)
{
    int dnumRows = right_op.numRows();
    //int dnumCols = right_op.numCols();

    NiceAssert( dnumRows && right_op.numCols() );

    T maxrow;
    T maxval;
    int maxargi = 0;
    int maxargj = 0;

    maxval = abs2(right_op(0,0));

    int i,j;

    retVector<T> tmpva;
    retVector<T> tmpvc;

    for ( i = 0 ; i < dnumRows ; ++i )
    {
	maxrow = maxabs(right_op(i,tmpva,tmpvc),j);

	if ( maxrow > maxval )
	{
	    maxval  = maxrow;
	    maxargi = i;
            maxargj = j;
	}
    }

    ii = maxargi;
    jj = maxargj;

    return maxval;
}

template <class T>
T minabs(const Matrix<T> &right_op, int &ii, int &jj)
{
    int dnumRows = right_op.numRows();
    //int dnumCols = right_op.numCols();

    NiceAssert( dnumRows && right_op.numCols() );

    T minrow;
    T minval;
    int minargi = 0;
    int minargj = 0;

    minval = abs2(right_op(0,0));

    int i,j;

    retVector<T> tmpva;
    retVector<T> tmpvc;

    for ( i = 0 ; i < dnumRows ; ++i )
    {
	minrow = minabs(right_op(i,tmpva,tmpvc),j);

	if ( minrow < minval )
	{
	    minval  = minrow;
	    minargi = i;
            minargj = j;
	}
    }

    ii = minargi;
    jj = minargj;

    return minval;
}

template <class T>
T maxabsdiag(const Matrix<T> &right_op, int &ii, int &jj)
{
    int dnumRows = right_op.numRows();
    //int dnumCols = right_op.numCols();

    NiceAssert( dnumRows && right_op.numCols() );
    NiceAssert( dnumRows == right_op.numCols() );

    T maxval;
    int maxargi = 0;

    maxval = abs2(right_op(0,0));

    int i;

    for ( i = 0 ; i < dnumRows ; ++i )
    {
	if ( abs2(right_op(i,i)) > maxval )
	{
	    maxval  = abs2(right_op(i,i));
	    maxargi = i;
	}
    }

    ii = maxargi;
    jj = maxargi;

    return maxval;
}

template <class T>
T minabsdiag(const Matrix<T> &right_op, int &ii, int &jj)
{
    int dnumRows = right_op.numRows();
    //int dnumCols = right_op.numCols();

    NiceAssert( dnumRows && right_op.numCols() );
    NiceAssert( dnumRows == right_op.numCols() );

    T minval;
    int minargi = 0;

    minval = abs2(right_op(0,0));

    int i;

    for ( i = 0 ; i < dnumRows ; ++i )
    {
	if ( abs2(right_op(i,i)) < minval )
	{
	    minval  = abs2(right_op(i,i));
	    minargi = i;
	}
    }

    ii = minargi;
    jj = minargi;

    return minval;
}



template <class T>
T diagsum(const Matrix<T> &right_op)
{
    int i;

    int size = right_op.size();
    T res;

    if ( size )
    {
	res = right_op(0,0);

	//if ( size > 1 )
	{
	    for ( i = 1 ; i < size ; ++i )
	    {
		res += right_op(i,i);
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
T sum(const Matrix<T> &right_op)
{
    int i;

    int size = right_op.numRows();
    T res;

    if ( size )
    {
        retVector<T> tmpva;

	res = sum(right_op(0,tmpva));

        for ( i = 1 ; i < size ; ++i )
        {
            res += sum(right_op(i,tmpva));
	}
    }

    else
    {
        setzero(res);
    }

    return res;
}

template <class T>
T sqsum(const Matrix<T> &right_op)
{
    int i;

    int size = right_op.numRows();
    T res;

    if ( size )
    {
        retVector<T> tmpva;

	res = sqsum(right_op(0,tmpva));

        for ( i = 1 ; i < size ; ++i )
        {
            res += sqsum(right_op(i,tmpva));
	}
    }

    else
    {
        setzero(res);
    }

    return res;
}

template <class T>
T sqsum(T &res, const Matrix<T> &right_op)
{
    int i;

    int size = right_op.numRows();

    if ( size )
    {
        retVector<T> tmpva;
        retVector<T> tmpvb;

	res = sqsum(right_op(0,tmpva,tmpvb));

        for ( i = 1 ; i < size ; ++i )
        {
            res += sqsum(right_op(i,tmpva,tmpvb));
	}
    }

    else
    {
        setzero(res);
    }

    return res;
}

template <class T>
T mean(const Matrix<T> &right_op)
{
    T res;

    if ( ((right_op.numRows()) && (right_op.numCols())) )
    {
	res  = sum(right_op);
        res /= (double) ((right_op.numRows())*(right_op.numCols()));
    }

    else
    {
	setzero(res);
    }

    return res;
}

template <class T>
const T &median(const Matrix<T> &right_op, int &miniv, int &minjv)
{
    if ( ( right_op.numRows() == 1 ) && ( right_op.numCols() == 1 ) )
    {
        miniv = 0;
        minjv = 0;

        return right_op(miniv,minjv);
    }

    else if ( right_op.numRows() && right_op.numCols() )
    {
        // Aim: right_op(outdexi,outdexj) should be arranged from largest to smallest

        Vector<int> outdexi;
        Vector<int> outdexj;

        int i,j,k;

	for ( i = 0 ; i < right_op.numRows() ; ++i )
	{
	    for ( j = 0 ; j < right_op.numCols() ; ++j )
	    {
                k = 0;

                //if ( outdexi.size() )
                {
                    for ( k = 0 ; k < outdexi.size() ; ++k )
                    {
                        if ( right_op(outdexi.v(k),outdexj.v(k)) <= right_op(i,j) )
                        {
                            break;
                        }
                    }
                }

                outdexi.add(k);
                outdexi.sv(k,i);

                outdexj.add(k);
                outdexj.sv(k,j);
            }
        }

        miniv = outdexi.v(outdexi.size()/2);
        minjv = outdexj.v(outdexj.size()/2);

        return right_op(miniv,minjv);
    }

    static thread_local int frun = 1;
    static thread_local T defres;

    miniv = 0;
    minjv = 0;

    if ( frun )
    {
        setzero(defres);
        frun = 0;
    }

    return defres;
}

template <class T>
Matrix<T> outerProduct(const Vector<T> &left_op, const Vector<T> &right_op)
{
    Matrix<T> res(left_op.size(),right_op.size());

    T oneval; oneval = 1;

    res.zero();
    res.rankone(oneval,left_op,right_op);

    return res;
}

template <class T>
const Matrix<T> &takeProduct(Matrix<T> &res, const Matrix<T> &a, const Matrix<T> &b)
{
    NiceAssert( a.numCols() == b.numRows() );

    int numRows = a.numRows();
    int innerdim = a.numCols();
    int numCols = b.numCols();

    res.resize(numRows,numCols);
    res.zero();

    if ( numRows && numCols )
    {
        int i,j,k;

        for ( i = 0 ; i < numRows ; ++i )
        {
            for ( j = 0 ; j < numCols ; ++j )
            {
                for ( k = 0 ; k < innerdim ; ++k )
                {
                    res("&",i,j) += (a(i,k)*b(k,j));
                }
            }
        }
    }

    return res;
}


//template <class T> double abs2(const Matrix<T> &a)
//{
//    return a.getRowColAbs();
//}

template <class T> double normF(const Matrix<T> &a)
{
    return norm2(a);
}

template <class S> double norm1(const Matrix<S> &a)
{
    int i,j;
    double result = 0;
    //T temp;

    //if ( a.numCols() && a.numRows() )
    {
        for ( i = 0 ; i < a.numCols() ; ++i )
        {
            for ( j = 0 ; j < a.numCols() ; ++j )
	    {
		result = norm1(a(i,j));
	    }
	}
    }

    return result;
}

template <class S> double norm2(const Matrix<S> &a)
{
    int i,j;
    double result = 0;
    //T temp;

    //if ( a.numCols() && a.numRows() )
    {
        for ( i = 0 ; i < a.numCols() ; ++i )
        {
            for ( j = 0 ; j < a.numCols() ; ++j )
	    {
		result = norm2(a(i,j));
	    }
	}
    }

    return result;
}

template <class S> double normp(const Matrix<S> &a, double p)
{
    int i,j;
    double result = 0;
    //T temp;

    if ( a.numCols() && a.numRows() )
    {
        for ( i = 0 ; i < a.numCols() ; ++i )
        {
            for ( j = 0 ; j < a.numCols() ; ++j )
	    {
		result = normp(a(i,j),p);
	    }
	}
    }

    return result;
}

template <class T> double absF(const Matrix<T> &a)
{
    return abs2(a);
}

template <class S> double abs1(const Matrix<S> &a)
{
    return norm1(a);
}

template <class S> double abs2(const Matrix<S> &a)
{
    return sqrt(norm2(a));
}

template <class S> double absp(const Matrix<S> &a, double p)
{
    return pow(normp(a,p),1/p);
}

template <class S> double absinf(const Matrix<S> &a)
{
    int i,j;
    double dres = 0.0;
    double dtemp = 0.0;

    if ( a.numRows() && a.numCols() )
    {
	for ( i = 0 ; i < a.numRows() ; ++i )
	{
	    for ( j = 0 ; i < a.numCols() ; ++j )
	    {
		dtemp = absinf(a(i,j));
                dres = ( dtemp > dres ) ? dtemp : dres;
	    }
	}
    }

    return dres;
}

template <class S> double abs0(const Matrix<S> &a)
{
    int i,j;
    double temp;
    double dres = 0.0;
    double dtemp = 0.0;

    if ( a.numRows() && a.numCols() )
    {
	for ( i = 0 ; i < a.numRows() ; ++i )
	{
	    for ( j = 0 ; i < a.numCols() ; ++j )
	    {
		dtemp = abs0(a(i,j));
                dres = ( ( !i && !j ) || ( dtemp < dres ) ) ? dtemp : dres;
	    }
	}
    }

    return dres;
}

template <class T> double distF(const Matrix<T> &a, const Matrix<T> &b)
{
    NiceAssert( a.numRows() == b.numRows() );
    NiceAssert( a.numCols() == b.numCols() );

    int i,j;
    double result = 0;
    T temp;

    if ( a.numCols() && a.numRows() )
    {
        for ( i = 0 ; i < a.numCols() ; ++i )
        {
            for ( j = 0 ; j < a.numCols() ; ++j )
	    {
                innerProduct(temp,a(i,j),b(i,j));

		result += norm2(a(i,j));
		result += norm2(b(i,j));
		result -= abs2(temp);
	    }
	}
    }

    return result;
}

template <class S> Matrix<S> angle(const Matrix<S> &a)
{
    Matrix<S> temp(a);

    double tempabs = abs2(temp);

    if ( tempabs != 0.0 )
    {
        temp.scale(1/tempabs);
    }

    return temp;
}

template <class T> Matrix<T> &setident(Matrix<T> &a)
{
    return a.ident();
}

template <class T> Matrix<T> &setzero(Matrix<T> &a)
{
    return a.zero();
}

template <class T> Matrix<T> &setposate(Matrix<T> &a)
{
    return a.posate();
}

template <class T> Matrix<T> &setnegate(Matrix<T> &a)
{
    return a.negate();
}

template <class T> Matrix<T> &setconj(Matrix<T> &a)
{
    return a.conj();
}

template <class T> Matrix<T> &setrand(Matrix<T> &a)
{
    return a.rand();
}

template <class T> Matrix<T> &settranspose(Matrix<T> &a)
{
    return a.transpose();
}

template <class T> Matrix<T> inv(const Matrix<T> &src)
{
    return src.inve();
}




template <class T>
Matrix<T> &Matrix<T>::ident(void)
{
    if ( dnumRows && dnumCols )
    {
	int i;

        retVector<T> tmpva;
        retVector<T> tmpvb;

	for ( i = 0 ; i < dnumRows ; ++i )
	{
	    setzero((*this)("&",i,"&",tmpva,tmpvb));

	    if ( i < dnumCols )
	    {
		setident((*this)("&",i,i));
	    }
	}
    }

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::zero(void)
{
    if ( dnumRows && dnumCols )
    {
	int i;

        retVector<T> tmpva;
        retVector<T> tmpvb;

	for ( i = 0 ; i < dnumRows ; ++i )
	{
            setzero((*this)("&",i,"&",tmpva,tmpvb));
	}
    }

    return *this;
}

template <> inline Matrix<double> &Matrix<double>::posate(void);
template <> inline Matrix<double> &Matrix<double>::posate(void)
{
    return *this;
}

template <> inline Matrix<int> &Matrix<int>::posate(void);
template <> inline Matrix<int> &Matrix<int>::posate(void)
{
    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::posate(void)
{
    if ( dnumRows && dnumCols )
    {
        int i;

        retVector<T> tmpva;
        retVector<T> tmpvb;

        for ( i = 0 ; i < dnumRows ; ++i )
        {
            setposate((*this)("&",i,"&",tmpva,tmpvb));
        }
    }

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::negate(void)
{
    if ( dnumRows && dnumCols )
    {
	int i;

        retVector<T> tmpva;
        retVector<T> tmpvb;

	for ( i = 0 ; i < dnumRows ; ++i )
	{
            setnegate((*this)("&",i,"&",tmpva,tmpvb));
	}
    }

    return *this;
}

template <> inline Matrix<double> &Matrix<double>::conj(void);
template <> inline Matrix<double> &Matrix<double>::conj(void)
{
    return *this;
}

template <> inline Matrix<int> &Matrix<int>::conj(void);
template <> inline Matrix<int> &Matrix<int>::conj(void)
{
    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::conj(void)
{
    if ( dnumRows && dnumCols )
    {
        int i;

        retVector<T> tmpva;
        retVector<T> tmpvb;

        for ( i = 0 ; i < dnumRows ; ++i )
        {
            setconj((*this)("&",i,"&",tmpva,tmpvb));
	}
    }

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::rand(void)
{
    if ( dnumRows && dnumCols )
    {
	int i;

        retVector<T> tmpva;
        retVector<T> tmpvb;

	for ( i = 0 ; i < dnumRows ; ++i )
	{
            setrand((*this)("&",i,"&",tmpva,tmpvb));
	}
    }

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::transpose(void)
{
    if ( dnumRows || dnumCols )
    {
        int maxwide = ( dnumRows > dnumCols ) ? dnumRows : dnumCols;
	int newnumRows = dnumCols;
	int newnumCols = dnumRows;

	resize(maxwide,maxwide);

	//T temp;

	int i,j;

	for ( i = 0 ; i < maxwide ; ++i )
	{
	    for ( j = i ; j < maxwide ; ++j )
	    {
		if ( i != j )
		{
                    qswap((*this)("&",i,j),(*this)("&",j,i));
		    //temp              = (*this)(i,j);
		    //(*this)("&",i,j) = (*this)(j,i);
		    //(*this)("&",j,i) = temp;
		}
	    }
	}

        resize(newnumRows,newnumCols);
    }

    return *this;
}

template <class T>
Matrix<T> &Matrix<T>::symmetrise(void)
{
    if ( dnumRows || dnumCols )
    {
        int maxwide = ( dnumRows > dnumCols ) ? dnumRows : dnumCols;
	int oldnumRows = dnumRows;
	int oldnumCols = dnumCols;

	resize(maxwide,maxwide);

	//T temp;

	int i,j;

        if ( oldnumRows != oldnumCols )
        {
            for ( i = 0 ; i < maxwide ; ++i )
            {
                for ( j = 0 ; j < maxwide ; ++j )
                {
                    if ( ( i >= oldnumRows ) || ( j >= oldnumCols ) )
                    {
                        setzero((*this)("&",i,j));
                    } 
                }
            }
	}

	for ( i = 0 ; i < maxwide ; ++i )
	{
	    for ( j = i+1 ; j < maxwide ; ++j )
	    {
                (*this)("&",i,j) += (*this)(j,i);
                (*this)("&",i,j) /= 2;
                (*this)("&",j,i)  = (*this)(i,j);
	    }
	}
    }

    return *this;
}




// Mathematical operator overloading

template <class T> Matrix<T>  operator+ (const Matrix<T> &left_op, const Matrix<T> &right_op)
{
    Matrix<T> res(left_op);

    return ( res += right_op );
}

template <class T> Matrix<T>  operator+ (const Matrix<T> &left_op, const T         &right_op)
{
    Matrix<T> res(left_op);

    return ( res += right_op );
}

template <class T> Matrix<T>  operator+ (const T         &left_op, const Matrix<T> &right_op)
{
    Matrix<T> res(right_op);

    return ( res += left_op );
}

template <class T> Matrix<T>  operator- (const Matrix<T> &left_op, const Matrix<T> &right_op)
{
    Matrix<T> res(left_op);

    return ( res -= right_op );
}

template <class T> Matrix<T>  operator- (const Matrix<T> &left_op, const T         &right_op)
{
    Matrix<T> res(left_op);

    return ( res -= right_op );
}

template <class T> Matrix<T>  operator- (const T         &left_op, const Matrix<T> &right_op)
{
    Matrix<T> res(right_op);
    setnegate(res);

    return ( res += left_op );
}

template <         class T> Matrix<T>  operator* (const Matrix<T> &left_op, const Matrix<T> &right_op)
{
    Matrix<T> res(left_op);

    return leftmult(res,right_op);
}

template <class S, class T> Vector<S>  operator* (const Vector<S> &left_op, const Matrix<T> &right_op)
{
    Vector<S> res(left_op);

    return leftmult(res,right_op);
}

template <         class T> Matrix<T>  operator* (const Matrix<T> &left_op, const T         &right_op)
{
    Matrix<T> res(left_op);

    return leftmult(res,right_op);
}

template <class S, class T> Vector<S>  operator* (const Matrix<T> &left_op, const Vector<S> &right_op)
{
    Vector<S> res(right_op);

    return rightmult(left_op,res);
}

template <         class T> Matrix<T>  operator* (const T         &left_op, const Matrix<T> &right_op)
{
    Matrix<T> res(right_op);

    return rightmult(left_op,res);
}

template <class T> Matrix<T> &operator*=(      Matrix<T> &left_op, const Matrix<T> &right_op)
{
    return leftmult(left_op,right_op);
}

template <class S, class T> Vector<S> &operator*=(      Vector<S> &left_op, const Matrix<T> &right_op)
{
    return leftmult(left_op,right_op);
}

template <class T> Matrix<T> &operator*=(      Matrix<T> &left_op, const T         &right_op)
{
    return leftmult(left_op,right_op);
}





template <> inline Matrix<double> operator+(const Matrix<double> &left_op);
template <> inline Matrix<double> operator+(const Matrix<double> &left_op)
{
    Matrix<double> res(left_op);

    return res;
}

template <> inline Matrix<int> operator+(const Matrix<int> &left_op);
template <> inline Matrix<int> operator+(const Matrix<int> &left_op)
{
    Matrix<int> res(left_op);

    return res;
}

template <class T> Matrix<T>  operator+(const Matrix<T> &left_op)
{
    int i;
    Matrix<T> res(left_op);

    if ( left_op.numRows() && left_op.numCols() )
    {
        retVector<T> tmpva;

        for ( i = 0 ; i < left_op.numRows() ; ++i )
        {
            setposate(res("&",i,tmpva));
	}
    }

    return res;
}

template <> inline Matrix<double> operator-(const Matrix<double> &left_op);
template <> inline Matrix<double> operator-(const Matrix<double> &left_op)
{
    Matrix<double> res(left_op);

    res.negate();

    return res;
}

template <class T> Matrix<T>  operator-(const Matrix<T> &left_op)
{
    int i;
    Matrix<T> res(left_op);

    if ( left_op.numRows() && left_op.numCols() )
    {
        retVector<T> tmpva;
        retVector<T> tmpvc;

	for ( i = 0 ; i < left_op.numRows() ; ++i )
	{
            setnegate(res("&",i,tmpva,tmpvc));
        }
    }

    return res;
}

template <class T> Matrix<T> &operator+=(      Matrix<T> &left_op, const Matrix<T> &right_op)
{
    NiceAssert( ( left_op.numRows() == right_op.numRows() ) || left_op.isEmpty() || right_op.isEmpty() );
    NiceAssert( ( left_op.numCols() == right_op.numCols() ) || left_op.isEmpty() || right_op.isEmpty() );

    if ( left_op.numRows() && left_op.numCols() && !(left_op.isEmpty()) && !(right_op.isEmpty()) )
    {
        if ( left_op.shareBase(&right_op) )
        {
	    Matrix<T> temp(right_op);

            left_op += temp;
	}

	else
	{
	    int i;

            retVector<T> tmpva;
            retVector<T> tmpvb;
            retVector<T> tmpvc;
            retVector<T> tmpvd;

 	    for ( i = 0 ; i < left_op.numRows() ; ++i )
	    {
		left_op("&",i,tmpva,tmpvc) += right_op(i,tmpvb,tmpvd);
            }
        }
    }

    else if ( left_op.isEmpty() )
    {
        left_op = right_op;
    }

    return left_op;
}

template <class T> Matrix<T> &operator+=(      Matrix<T> &left_op, const T         &right_op)
{
    if ( left_op.numRows() && left_op.numCols() )
    {
	int i;
        int mindim = ( left_op.numRows() <= left_op.numCols() ) ? left_op.numRows() : left_op.numCols();

        for ( i = 0 ; i < mindim ; ++i )
	{
            left_op("&",i,i) += right_op;
	}
    }

    return left_op;
}

template <class T> Matrix<T> &operator-=(      Matrix<T> &left_op, const Matrix<T> &right_op)
{
    NiceAssert( ( left_op.numRows() == right_op.numRows() ) || left_op.isEmpty() || right_op.isEmpty() );
    NiceAssert( ( left_op.numCols() == right_op.numCols() ) || left_op.isEmpty() || right_op.isEmpty() );

    if ( left_op.numRows() && left_op.numCols() && !(left_op.isEmpty()) && !(right_op.isEmpty()) )
    {
        if ( left_op.shareBase(&right_op) )
        {
	    Matrix<T> temp(right_op);

            left_op += temp;
	}

	else
	{
	    int i;

            retVector<T> tmpva;
            retVector<T> tmpvb;
            retVector<T> tmpvc;
            retVector<T> tmpvd;

   	    for ( i = 0 ; i < left_op.numRows() ; ++i )
	    {
                left_op("&",i,tmpva,tmpvc) -= right_op(i,tmpvb,tmpvd);
	    }
        }
    }

    else if ( left_op.isEmpty() )
    {
        left_op = right_op;
        left_op.negate();
    }

    return left_op;
}

template <class T> Matrix<T> &operator-=(      Matrix<T> &left_op, const T         &right_op)
{
    if ( left_op.numRows() && left_op.numCols() )
    {
	int i;
        int mindim = ( left_op.numRows() <= left_op.numCols() ) ? left_op.numRows() : left_op.numCols();

        for ( i = 0 ; i < mindim ; ++i )
	{
            left_op("&",i,i) -= right_op;
	}
    }

    return left_op;
}

template <> inline Matrix<double> &leftmult(      Matrix<double> &left_op, const Matrix<double> &right_op);
template <> inline Matrix<double> &leftmult(      Matrix<double> &left_op, const Matrix<double> &right_op)
{
    NiceAssert( ( left_op.numCols() == right_op.numRows() ) || left_op.isEmpty() || right_op.isEmpty() );

    if ( left_op.isEmpty() )
    {
        left_op = right_op;
    }

    else if ( right_op.isEmpty() )
    {
        ;
    }

    else if ( left_op.shareBase(&right_op) )
    {
        Matrix<double> temp(right_op);

        return leftmult(left_op,temp);
    }

    else
    {
        int i,j,k;
        int innerdim = left_op.numCols();
        int resnumRows = left_op.numRows();
        int resnumCols = right_op.numCols();

        if ( resnumRows && resnumCols && innerdim )
        {
            if ( resnumCols > left_op.numCols() )
            {
                left_op.resize(resnumRows,resnumCols);
            }

            static thread_local Vector<double> leftrow(resnumCols,nullptr,2);
            leftrow.resize(resnumCols);
            double tmpres;

            for ( i = 0 ; i < resnumRows ; ++i )
            {
                for ( j = 0 ; j < resnumCols ; ++j )
                {
                    tmpres = 0.0;

                    for ( k = 0 ; k < innerdim ; k++ )
                    {
                        //tmpres += left_op.v(i,k)*right_op.v(k,j);
                        tmpres = std::fma(left_op.v(i,k),right_op.v(k,j),tmpres);
                    }

                    leftrow.sv(j,tmpres);
                }

                for ( j = 0 ; j < resnumCols ; j++ )
                {
                    left_op.sv(i,j,leftrow.v(j));
                }
            }

            left_op.resize(resnumRows,resnumCols);
        }

        else
        {
            left_op.resize(resnumRows,resnumCols);
            left_op.zero();
        }
    }

    return left_op;
}

template <class T> Matrix<T> &leftmult(      Matrix<T> &left_op, const Matrix<T> &right_op)
{
    NiceAssert( ( left_op.numCols() == right_op.numRows() ) || left_op.isEmpty() || right_op.isEmpty() );

    if ( left_op.isEmpty() )
    {
        left_op = right_op;
    }

    else if ( right_op.isEmpty() )
    {
        ;
    }

    else if ( left_op.shareBase(&right_op) )
    {
        Matrix<T> temp(right_op);

        return leftmult(left_op,temp);
    }

    else
    {
        int i,j,k;
        int innerdim = left_op.numCols();
        int resnumRows = left_op.numRows();
        int resnumCols = right_op.numCols();

        if ( resnumRows && resnumCols && innerdim )
        {
            if ( resnumCols > left_op.numCols() )
            {
                left_op.resize(resnumRows,resnumCols);
            }

            //Vector<T> leftrow(resnumCols);
            static thread_local Vector<T> leftrow(resnumCols,nullptr,2);
            leftrow.resize(resnumCols);

            for ( i = 0 ; i < resnumRows ; ++i )
            {
                for ( j = 0 ; j < resnumCols ; ++j )
                {
                    setzero(leftrow("&",j));

                    for ( k = 0 ; k < innerdim ; k++ )
                    {
                        leftrow("&",j) += left_op(i,k)*right_op(k,j);
                    }
                }

                for ( j = 0 ; j < resnumCols ; j++ )
                {
                    left_op("&",i,j) = leftrow(j);
                }
            }

            left_op.resize(resnumRows,resnumCols);
        }

        else
        {
            left_op.resize(resnumRows,resnumCols);
            left_op.zero();
        }
    }

    return left_op;
}

template <> inline Vector<double> &leftmult (      Vector<double> &left_op, const Matrix<double> &right_op);
template <> inline Vector<double> &leftmult (      Vector<double> &left_op, const Matrix<double> &right_op)
{
    NiceAssert( ( left_op.size() == right_op.numRows() ) || right_op.isEmpty() );

    if ( right_op.isEmpty() )
    {
        ;
    }

    else
    {
        int i,j;
        int innerdim = right_op.numRows();
        int resnumCols = right_op.numCols();

        static thread_local Vector<double> res(resnumCols,nullptr,2);
        res.resize(resnumCols);
        double innerres;

        for ( i = 0 ; i < resnumCols ; ++i )
        {
            innerres = 0;

            for ( j = 0 ; j < innerdim ; ++j )
            {
                //innerres += left_op.v(j)*right_op.v(j,i);
                innerres = std::fma(left_op.v(j),right_op.v(j,i),innerres);
            }

            res.sv(i,innerres);
        }

        left_op = res;
    }

    return left_op;
}

template <class S, class T> Vector<S> &leftmult (      Vector<S> &left_op, const Matrix<T> &right_op)
{
    NiceAssert( ( left_op.size() == right_op.numRows() ) || right_op.isEmpty() );

    if ( right_op.isEmpty() )
    {
        ;
    }

    else
    {
        int i,j;
        int innerdim = right_op.numRows();
        int resnumCols = right_op.numCols();

        //Vector<S> res(resnumCols);
        static thread_local Vector<S> res(resnumCols,nullptr,2);
        res.resize(resnumCols);

        if ( resnumCols )
        {
            //if svm_constexpr_if ( ( !(std::is_floating_point<T>::value) && !(std::is_integral<T>::value) ) || ( !(std::is_floating_point<S>::value) && !(std::is_integral<S>::value) ) )
            //{
            //    for ( i = 0 ; i < resnumCols ; ++i )
            //    {
            //        setzero(res("&",i));
            //
            //        for ( j = 0 ; j < innerdim ; ++j )
            //        {
            //            res("&",i) += left_op(j)*right_op(j,i);
            //        }
            //    }
            //}

            //else
            {
                for ( i = 0 ; i < resnumCols ; ++i )
                {
                    setzero(res("&",i));

                    for ( j = 0 ; j < innerdim ; ++j )
                    {
                        res("&",i) += left_op(j)*right_op(j,i);
                    }
                }
            }
        }

        left_op = res;
    }

    return left_op;
}

template <         class T> Matrix<T> &leftmult (      Matrix<T> &left_op, const T         &right_op)
{
    if ( left_op.numRows() && left_op.numCols() )
    {
	int i;

        retVector<T> tmpva;
        retVector<T> tmpvb;

	for ( i = 0 ; i < left_op.numRows() ; ++i )
	{
	    left_op("&",i,tmpva,tmpvb) *= right_op;
        }
    }

    return left_op;
}

template <> inline Matrix<double> &rightmult(const Matrix<double> &left_op, Matrix<double> &right_op);
template <> inline Matrix<double> &rightmult(const Matrix<double> &left_op, Matrix<double> &right_op)
{
    NiceAssert( ( left_op.numCols() == right_op.numRows() ) || left_op.isEmpty() || right_op.isEmpty() );

    if ( left_op.isEmpty() )
    {
        ;
    }

    else if ( right_op.isEmpty() )
    {
        right_op = left_op;
    }

    else if ( right_op.shareBase(&left_op) )
    {
        Matrix<double> temp(left_op);

        return rightmult(temp,right_op);
    }

    else
    {
        int i,j,k;
        int innerdim = left_op.numCols();
        int resnumRows = left_op.numRows();
        int resnumCols = right_op.numCols();

        if ( resnumRows && resnumCols && innerdim )
        {
            if ( resnumRows > right_op.numCols() )
            {
                right_op.resize(resnumRows,resnumCols);
            }

            static thread_local Vector<double> rightcol(resnumRows,nullptr,2);
            rightcol.resize(resnumRows);
            double tmpres;

            for ( j = 0 ; j < resnumCols ; ++j )
            {
                for ( i = 0 ; i < resnumRows ; ++i )
                {
                    tmpres = 0;

                    for ( k = 0 ; k < innerdim ; k++ )
                    {
                        //tmpres += (left_op.v(i,k)*right_op.v(k,j));
                        tmpres = std::fma(left_op.v(i,k),right_op.v(k,j),tmpres);
                    }

                    rightcol.sv(i,tmpres);
                }

                for ( i = 0 ; i < resnumRows ; ++i )
                {
                    right_op.sv(i,j,rightcol(i));
                }
            }

            right_op.resize(resnumRows,resnumCols);
        }

        else
        {
            right_op.resize(resnumRows,resnumCols);
            right_op.zero();
        }
    }

    return right_op;
}

template <class T> Matrix<T> &rightmult(const Matrix<T> &left_op, Matrix<T> &right_op)
{
    NiceAssert( ( left_op.numCols() == right_op.numRows() ) || left_op.isEmpty() || right_op.isEmpty() );

    if ( left_op.isEmpty() )
    {
        ;
    }

    else if ( right_op.isEmpty() )
    {
        right_op = left_op;
    }

    else if ( right_op.shareBase(&left_op) )
    {
        Matrix<T> temp(left_op);

        return rightmult(temp,right_op);
    }

    else
    {
        int i,j,k;
        int innerdim = left_op.numCols();
        int resnumRows = left_op.numRows();
        int resnumCols = right_op.numCols();

        if ( resnumRows && resnumCols && innerdim )
        {
            if ( resnumRows > right_op.numCols() )
            {
                right_op.resize(resnumRows,resnumCols);
            }

            //Vector<T> rightcol(resnumRows);
            static thread_local Vector<T> rightcol(resnumRows,nullptr,2);
            rightcol.resize(resnumRows);

            for ( j = 0 ; j < resnumCols ; ++j )
            {
                for ( i = 0 ; i < resnumRows ; ++i )
                {
                    setzero(rightcol("&",i));

                    for ( k = 0 ; k < innerdim ; k++ )
                    {
                        rightcol("&",i) += (left_op(i,k)*right_op(k,j));
                    }
                }

                for ( i = 0 ; i < resnumRows ; ++i )
                {
                    right_op("&",i,j) = rightcol(i);
                }
            }

            right_op.resize(resnumRows,resnumCols);
        }

        else
        {
            right_op.resize(resnumRows,resnumCols);
            right_op.zero();
        }
    }

    return right_op;
}

//NB: YOU DONT NEED TO SPECIALISE THIS.  SUMB IS ALREADY OPTIMISED TO THE Nth DEGREE, VALUE
//SEMANTICS ALREADY HAPPEN, NO NEED TO SPECIALISE
template <class S, class T> Vector<S> &rightmult(const Matrix<T> &left_op,       Vector<S> &right_op)
{
    NiceAssert( ( left_op.numCols() == right_op.size() ) || left_op.isEmpty() );

    if ( left_op.isEmpty() )
    {
        ;
    }

    else
    {
        int i;
        //int innerdim = left_op.numCols();
        int resnumRows = left_op.numRows();

        //Vector<S> res(resnumRows);
        static thread_local Vector<S> res(resnumRows,nullptr,2);
        res.resize(resnumRows);

        if ( resnumRows )
        {
            retVector<T> tmpva;
            retVector<T> tmpvb;

            for ( i = 0 ; i < resnumRows ; ++i )
            {
                // Do this even if innerdim == 0 to ensure zeroing

                sumb(res("&",i),left_op(i,tmpva,tmpvb),right_op);
            }
        }

        right_op = res;
    }

    return right_op;
}

template <         class T> Matrix<T> &rightmult(const T         &left_op,       Matrix<T> &right_op)
{
    if ( right_op.numRows() && right_op.numCols() )
    {
	int i;

        retVector<T> tmpva;
        retVector<T> tmpvb;
        retVector<T> tmpvc;
        retVector<T> tmpvd;

	for ( i = 0 ; i < right_op.numRows() ; ++i )
	{
	    right_op("&",i,tmpva,tmpvd) = ( left_op * right_op(i,tmpvb,tmpvc) );
        }
    }

    return right_op;
}

template <> inline Matrix<double> &mult(Matrix<double> &A, const Matrix<double> &B, const Matrix<double> &C);
template <> inline Matrix<double> &mult(Matrix<double> &A, const Matrix<double> &B, const Matrix<double> &C)
{
    NiceAssert( ( B.numCols() == C.numRows() ) || B.isEmpty() || C.isEmpty() );

    if ( B.isEmpty() )
    {
        A = C;
    }

    else if ( C.isEmpty() )
    {
        A = B;
    }

    else
    {
        int i,j,k;
        int resnumRows = B.numRows();
        int innerdim   = B.numCols();
        int resnumCols = C.numCols();

        A.resize(resnumRows,resnumCols);

        if ( resnumRows && resnumCols && innerdim )
        {
            for ( i = 0 ; i < resnumRows ; ++i )
            {
                for ( j = 0 ; j < resnumCols ; ++j )
                {
                    double innerres = 0;

                    for ( k = 0 ; k < innerdim ; ++k )
                    {
                        innerres += B.v(i,k)*C.v(k,j);
                    }

                    A.sv(i,j,innerres);
                }
            }
        }

        else
        {
            A.zero();
        }
    }

    return A;
}

template <class T> Matrix<T> &mult(Matrix<T> &A, const Matrix<T> &B, const Matrix<T> &C)
{
    NiceAssert( ( B.numCols() == C.numRows() ) || B.isEmpty() || C.isEmpty() );

    if ( B.isEmpty() )
    {
        A = C;
    }

    else if ( C.isEmpty() )
    {
        A = B;
    }

    else
    {
        int i,j,k;
        int resnumRows = B.numRows();
        int innerdim   = B.numCols();
        int resnumCols = C.numCols();

        A.resize(resnumRows,resnumCols);

        if ( resnumRows && resnumCols && innerdim )
        {
            //if svm_constexpr_if ( !(std::is_floating_point<T>::value) && !(std::is_integral<T>::value) )
            //{
            //    for ( i = 0 ; i < resnumRows ; ++i )
            //    {
            //        for ( j = 0 ; j < resnumCols ; ++j )
            //        {
            //            setzero(A("&",i,j));
            //
            //            for ( k = 0 ; k < innerdim ; ++k )
            //            {
            //                A("&",i,j) += B(i,k)*C(k,j);
            //            }
            //        }
            //    }
            //}

            //else
            {
                for ( i = 0 ; i < resnumRows ; ++i )
                {
                    for ( j = 0 ; j < resnumCols ; ++j )
                    {
                        setzero(A("&",i,j));

                        for ( k = 0 ; k < innerdim ; ++k )
                        {
                            A("&",i,j) += B(i,k)*C(k,j);
                        }
                    }
                }
            }
        }

        else
        {
            A.zero();
        }
    }

    return A;
}

template <> inline Vector<double> &mult(Vector<double> &a, const Vector<double> &b, const Matrix<double> &C);
template <> inline Vector<double> &mult(Vector<double> &a, const Vector<double> &b, const Matrix<double> &C)
{
    NiceAssert( ( b.size() == C.numRows() ) || C.isEmpty() );

    if ( C.isEmpty() )
    {
        a = b;
    }

    else
    {
        int i,k;
        int innerdim = b.size();
        int ressize  = C.numCols();

        a.resize(ressize);

        if ( ressize && innerdim )
        {
            {
                for ( i = 0 ; i < ressize ; ++i )
                {
                    double innerres = 0;

                    for ( k = 0 ; k < innerdim ; ++k )
                    {
                        innerres += b.v(k)*C.v(k,i);
                    }

                    a.sv(i,innerres);
                }
            }
        }

        else
        {
            a.zero();
        }
    }

    return a;
}

template <class T> Vector<T> &mult(Vector<T> &a, const Vector<T> &b, const Matrix<T> &C)
{
    NiceAssert( ( b.size() == C.numRows() ) || C.isEmpty() );

    if ( C.isEmpty() )
    {
        a = b;
    }

    else
    {
        int i,k;
        int innerdim = b.size();
        int ressize  = C.numCols();

        a.resize(ressize);

        if ( ressize && innerdim )
        {
            //if svm_constexpr_if ( !(std::is_floating_point<T>::value) && !(std::is_integral<T>::value) )
            //{
            //    for ( i = 0 ; i < ressize ; ++i )
            //    {
            //        setzero(a("&",i));
            //
            //        for ( k = 0 ; k < innerdim ; ++k )
            //        {
            //            a("&",i) += b(k)*C(k,i);
            //        }
            //    }
            //}

            //else
            {
                for ( i = 0 ; i < ressize ; ++i )
                {
                    setzero(a("&",i));

                    for ( k = 0 ; k < innerdim ; ++k )
                    {
                        a("&",i) += b(k)*C(k,i);
                    }
                }
            }
        }

        else
        {
            a.zero();
        }
    }

    return a;
}

template <> inline Vector<double> &mult(Vector<double> &a, const Matrix<double> &B, const Vector<double> &c);
template <> inline Vector<double> &mult(Vector<double> &a, const Matrix<double> &B, const Vector<double> &c)
{
    NiceAssert( ( B.numCols() == c.size() ) || B.isEmpty() );

    if ( B.isEmpty() )
    {
        a = c;
    }

    else
    {
        int i,k;
        int ressize  = B.numRows();
        int innerdim = c.size();

        a.resize(ressize);

        if ( ressize && innerdim )
        {
            {
                for ( i = 0 ; i < ressize ; ++i )
                {
                    double innerres = 0;

                    for ( k = 0 ; k < innerdim ; ++k )
                    {
                        innerres += B.v(i,k)*c.v(k);
                    }

                    a.sv(i,innerres);
                }
            }
        }

        else
        {
            a.zero();
        }
    }

    return a;
}

template <class T> Vector<T> &mult(Vector<T> &a, const Matrix<T> &B, const Vector<T> &c)
{
    NiceAssert( ( B.numCols() == c.size() ) || B.isEmpty() );

    if ( B.isEmpty() )
    {
        a = c;
    }

    else
    {
        int i,k;
        int ressize  = B.numRows();
        int innerdim = c.size();

        a.resize(ressize);

        if ( ressize && innerdim )
        {
            //if svm_constexpr_if ( !(std::is_floating_point<T>::value) && !(std::is_integral<T>::value) )
            //{
            //    for ( i = 0 ; i < ressize ; ++i )
            //    {
            //        setzero(a("&",i));
            //
            //        for ( k = 0 ; k < innerdim ; ++k )
            //        {
            //            a("&",i) += B(i,k)*c(k);
            //        }
            //    }
            //}

            //else
            {
                for ( i = 0 ; i < ressize ; ++i )
                {
                    setzero(a("&",i));

                    for ( k = 0 ; k < innerdim ; ++k )
                    {
                        a("&",i) += B(i,k)*c(k);
                    }
                }
            }
        }

        else
        {
            a.zero();
        }
    }

    return a;
}

template <> inline Vector<double> &multtrans(Vector<double> &a, const Vector<double> &b, const Matrix<double> &C);
template <> inline Vector<double> &multtrans(Vector<double> &a, const Vector<double> &b, const Matrix<double> &C)
{
    NiceAssert( ( b.size() == C.numCols() ) || C.isEmpty() );

    if ( C.isEmpty() )
    {
        a = b;
    }

    else
    {
        int i,k;
        int innerdim = b.size();
        int ressize  = C.numRows();

        a.resize(ressize);

        if ( ressize && innerdim )
        {
            {
                for ( i = 0 ; i < ressize ; ++i )
                {
                    double innerres = 0;

                    for ( k = 0 ; k < innerdim ; ++k )
                    {
                        innerres += b.v(k)*C.v(i,k);
                    }

                    a.sv(i,innerres);
                }
            }
        }

        else
        {
            a.zero();
        }
    }

    return a;
}

template <class T> Vector<T> &multtrans(Vector<T> &a, const Vector<T> &b, const Matrix<T> &C)
{
    NiceAssert( ( b.size() == C.numCols() ) || C.isEmpty() );

    if ( C.isEmpty() )
    {
        a = b;
    }

    else
    {
        int i,k;
        int innerdim = b.size();
        int ressize  = C.numRows();

        a.resize(ressize);

        if ( ressize && innerdim )
        {
            //if svm_constexpr_if ( !(std::is_floating_point<T>::value) && !(std::is_integral<T>::value) )
            //{
            //    for ( i = 0 ; i < ressize ; ++i )
            //    {
            //        setzero(a("&",i));
            //
            //        for ( k = 0 ; k < innerdim ; ++k )
            //        {
            //            a("&",i) += b(k)*C(i,k);
            //        }
            //    }
            //}

            //else
            {
                for ( i = 0 ; i < ressize ; ++i )
                {
                    setzero(a("&",i));

                    for ( k = 0 ; k < innerdim ; ++k )
                    {
                        a("&",i) += b(k)*C(i,k);
                    }
                }
            }
        }

        else
        {
            a.zero();
        }
    }

    return a;
}

template <> inline Vector<double> &multtrans(Vector<double> &a, const Matrix<double> &B, const Vector<double> &c);
template <> inline Vector<double> &multtrans(Vector<double> &a, const Matrix<double> &B, const Vector<double> &c)
{
    NiceAssert( ( B.numRows() == c.size() ) || B.isEmpty() );

    if ( B.isEmpty() )
    {
        a = c;
    }

    else
    {
        int i,k;
        int ressize  = B.numCols();
        int innerdim = c.size();

        a.resize(ressize);

        if ( ressize && innerdim )
        {
            {
                for ( i = 0 ; i < ressize ; ++i )
                {
                    double innerres = 0;

                    for ( k = 0 ; k < innerdim ; ++k )
                    {
                        innerres += B.v(k,i)*c.v(k);
                    }

                    a.sv(i,innerres);
                }
            }
        }

        else
        {
            a.zero();
        }
    }

    return a;
}

template <class T> Vector<T> &multtrans(Vector<T> &a, const Matrix<T> &B, const Vector<T> &c)
{
    NiceAssert( ( B.numRows() == c.size() ) || B.isEmpty() );

    if ( B.isEmpty() )
    {
        a = c;
    }

    else
    {
        int i,k;
        int ressize  = B.numCols();
        int innerdim = c.size();

        a.resize(ressize);

        if ( ressize && innerdim )
        {
            //if svm_constexpr_if ( !(std::is_floating_point<T>::value) && !(std::is_integral<T>::value) )
            //{
            //    for ( i = 0 ; i < ressize ; ++i )
            //    {
            //        setzero(a("&",i));
            //
            //        for ( k = 0 ; k < innerdim ; ++k )
            //        {
            //            a("&",i) += B(k,i)*c(k);
            //        }
            //    }
            //}

            //else
            {
                for ( i = 0 ; i < ressize ; ++i )
                {
                    setzero(a("&",i));

                    for ( k = 0 ; k < innerdim ; ++k )
                    {
                        a("&",i) += B(k,i)*c(k);
                    }
                }
            }
        }

        else
        {
            a.zero();
        }
    }

    return a;
}



template <class T> int operator==(const Matrix<T> &left_op, const Matrix<T> &right_op)
{
    if ( ( left_op.numRows() != right_op.numRows() ) || ( left_op.numCols() != right_op.numCols() ) )
    {
        return 0;
    }

    if ( left_op.numRows() && left_op.numCols() )
    {
        int i;

        retVector<T> tmpva;
        retVector<T> tmpvb;
        retVector<T> tmpvc;
        retVector<T> tmpvd;

	for ( i = 0 ; i < left_op.numRows() ; ++i )
	{
	    if ( left_op(i,tmpva,tmpvb) != right_op(i,tmpvb,tmpvd) )
	    {
                return 0;
	    }
	}
    }

    return 1;
}

template <class T> int operator==(const Matrix<T> &left_op, const T         &right_op)
{
    if ( left_op.numRows() && left_op.numCols() )
    {
        int i;

        retVector<T> tmpva;
        retVector<T> tmpvc;

	for ( i = 0 ; i < left_op.numRows() ; ++i )
	{
	    if ( left_op(i,tmpva,tmpvc) != right_op )
	    {
                return 0;
	    }
        }
    }

    return 1;
}

template <class T> int operator==(const T         &left_op, const Matrix<T> &right_op)
{
    return ( right_op == left_op );
}

template <class T> int operator!=(const Matrix<T> &left_op, const Matrix<T> &right_op)
{
    return !( left_op == right_op );
}

template <class T> int operator!=(const Matrix<T> &left_op, const T         &right_op)
{
    return !( left_op == right_op );
}

template <class T> int operator!=(const T         &left_op, const Matrix<T> &right_op)
{
    return !( left_op == right_op );
}

template <class T> int operator< (const Matrix<T> &left_op, const Matrix<T> &right_op)
{
    if ( ( left_op.numRows() != right_op.numRows() ) || ( left_op.numCols() != right_op.numCols() ) )
    {
        return 0;
    }

    if ( left_op.numRows() && left_op.numCols() )
    {
        int i;

        retVector<T> tmpva;
        retVector<T> tmpvb;
        retVector<T> tmpvc;
        retVector<T> tmpvd;

	for ( i = 0 ; i < left_op.numRows() ; ++i )
	{
	    if ( !( left_op(i,tmpva,tmpvc) <  right_op(i,tmpvb,tmpvd) ) )
	    {
                return 0;
	    }
        }
    }

    return 1;
}

template <class T> int operator< (const Matrix<T> &left_op, const T         &right_op)
{
    if ( left_op.numRows() && left_op.numCols() )
    {
        int i;

        retVector<T> tmpva;
        retVector<T> tmpvc;

	for ( i = 0 ; i < left_op.numRows() ; ++i )
	{
	    if ( !( left_op(i,tmpva,tmpvc) <  right_op ) )
	    {
                return 0;
	    }
        }
    }

    return 1;
}

template <class T> int operator< (const T         &left_op, const Matrix<T> &right_op)
{
    if ( right_op.numRows() && right_op.numCols() )
    {
        int i;

        retVector<T> tmpva;
        retVector<T> tmpvc;

	for ( i = 0 ; i < right_op.numRows() ; ++i )
	{
	    if ( !( left_op <  right_op(i,tmpva,tmpvc) ) )
	    {
                return 0;
	    }
        }
    }

    return 1;
}

template <class T> int operator<=(const Matrix<T> &left_op, const Matrix<T> &right_op)
{
    if ( ( left_op.numRows() != right_op.numRows() ) || ( left_op.numCols() != right_op.numCols() ) )
    {
        return 0;
    }

    if ( left_op.numRows() && left_op.numCols() )
    {
        int i;

        retVector<T> tmpva;
        retVector<T> tmpvb;
        retVector<T> tmpvc;
        retVector<T> tmpvd;

	for ( i = 0 ; i < left_op.numRows() ; ++i )
	{
	    if ( !( left_op(i,tmpva,tmpvc) <= right_op(i,tmpvb,tmpvd) ) )
	    {
                return 0;
	    }
        }
    }

    return 1;
}

template <class T> int operator<=(const Matrix<T> &left_op, const T         &right_op)
{
    if ( left_op.numRows() && left_op.numCols() )
    {
        int i;

        retVector<T> tmpva;
        retVector<T> tmpvc;

	for ( i = 0 ; i < left_op.numRows() ; ++i )
	{
	    if ( !( left_op(i,tmpva,tmpvc) <= right_op ) )
	    {
                return 0;
	    }
        }
    }

    return 1;
}

template <class T> int operator<=(const T         &left_op, const Matrix<T> &right_op)
{
    if ( right_op.numRows() && right_op.numCols() )
    {
        int i;

        retVector<T> tmpva;
        retVector<T> tmpvc;

	for ( i = 0 ; i < right_op.numRows() ; ++i )
	{
	    if ( !( left_op <= right_op(i,tmpva,tmpvc) ) )
	    {
                return 0;
	    }
        }
    }

    return 1;
}

template <class T> int operator> (const Matrix<T> &left_op, const Matrix<T> &right_op)
{
    if ( ( left_op.numRows() != right_op.numRows() ) || ( left_op.numCols() != right_op.numCols() ) )
    {
        return 0;
    }

    if ( left_op.numRows() && left_op.numCols() )
    {
        int i;

        retVector<T> tmpva;
        retVector<T> tmpvb;
        retVector<T> tmpvc;
        retVector<T> tmpvd;

	for ( i = 0 ; i < left_op.numRows() ; ++i )
	{
	    if ( !( left_op(i,tmpva,tmpvc) >  right_op(i,tmpvb,tmpvd) ) )
	    {
                return 0;
	    }
        }
    }

    return 1;
}

template <class T> int operator> (const Matrix<T> &left_op, const T         &right_op)
{
    if ( left_op.numRows() && left_op.numCols() )
    {
        int i;

        retVector<T> tmpva;
        retVector<T> tmpvc;

	for ( i = 0 ; i < left_op.numRows() ; ++i )
	{
	    if ( !( left_op(i,tmpva,tmpvc) >  right_op ) )
	    {
                return 0;
	    }
        }
    }

    return 1;
}

template <class T> int operator> (const T         &left_op, const Matrix<T> &right_op)
{
    if ( right_op.numRows() && right_op.numCols() )
    {
        int i;

        retVector<T> tmpva;
        retVector<T> tmpvc;

	for ( i = 0 ; i < right_op.numRows() ; ++i )
	{
	    if ( !( left_op >  right_op(i,tmpva,tmpvc) ) )
	    {
                return 0;
	    }
        }
    }

    return 1;
}

template <class T> int operator>=(const Matrix<T> &left_op, const Matrix<T> &right_op)
{
    if ( ( left_op.numRows() != right_op.numRows() ) || ( left_op.numCols() != right_op.numCols() ) )
    {
        return 0;
    }

    if ( left_op.numRows() && left_op.numCols() )
    {
        int i;

        retVector<T> tmpva;
        retVector<T> tmpvb;
        retVector<T> tmpvc;
        retVector<T> tmpvd;

	for ( i = 0 ; i < left_op.numRows() ; ++i )
	{
	    if ( !( left_op(i,tmpva,tmpvc) >= right_op(i,tmpvb,tmpvd) ) )
	    {
                return 0;
	    }
        }
    }

    return 1;
}

template <class T> int operator>=(const Matrix<T> &left_op, const T         &right_op)
{
    if ( left_op.numRows() && left_op.numCols() )
    {
        int i;

        retVector<T> tmpva;
        retVector<T> tmpvc;

	for ( i = 0 ; i < left_op.numRows() ; ++i )
	{
	    if ( !( left_op(i,tmpva,tmpvc) >= right_op ) )
	    {
                return 0;
	    }
        }
    }

    return 1;
}

template <class T> int operator>=(const T         &left_op, const Matrix<T> &right_op)
{
    if ( right_op.numRows() && right_op.numCols() )
    {
        int i;

        retVector<T> tmpva;
        retVector<T> tmpvc;

	for ( i = 0 ; i < right_op.numRows() ; ++i )
	{
	    if ( !( left_op >= right_op(i,tmpva,tmpvc) ) )
	    {
                return 0;
	    }
        }
    }

    return 1;
}

template <class T> Matrix<T> &randrfill(Matrix<T> &res) { return res.applyon(randrfill); }
template <class T> Matrix<T> &randbfill(Matrix<T> &res) { return res.applyon(randbfill); }
template <class T> Matrix<T> &randBfill(Matrix<T> &res) { return res.applyon(randBfill); }
template <class T> Matrix<T> &randgfill(Matrix<T> &res) { return res.applyon(randgfill); }
template <class T> Matrix<T> &randpfill(Matrix<T> &res) { return res.applyon(randpfill); }
template <class T> Matrix<T> &randufill(Matrix<T> &res) { return res.applyon(randufill); }
template <class T> Matrix<T> &randefill(Matrix<T> &res) { return res.applyon(randefill); }
template <class T> Matrix<T> &randGfill(Matrix<T> &res) { return res.applyon(randGfill); }
template <class T> Matrix<T> &randwfill(Matrix<T> &res) { return res.applyon(randwfill); }
template <class T> Matrix<T> &randxfill(Matrix<T> &res) { return res.applyon(randxfill); }
template <class T> Matrix<T> &randnfill(Matrix<T> &res) { return res.applyon(randnfill); }
template <class T> Matrix<T> &randlfill(Matrix<T> &res) { return res.applyon(randlfill); }
template <class T> Matrix<T> &randcfill(Matrix<T> &res) { return res.applyon(randcfill); }
template <class T> Matrix<T> &randCfill(Matrix<T> &res) { return res.applyon(randCfill); }
template <class T> Matrix<T> &randffill(Matrix<T> &res) { return res.applyon(randffill); }
template <class T> Matrix<T> &randtfill(Matrix<T> &res) { return res.applyon(randtfill); }

inline Matrix<double> &ltfill(Matrix<double> &lhsres, const Matrix<double> &rhs)
{
    NiceAssert( lhsres.numRows() == rhs.numRows() );
    NiceAssert( lhsres.numCols() == rhs.numCols() );

    if ( lhsres.numRows() && lhsres.numCols() )
    {
        int i,j;

        for ( i = 0 ; i < lhsres.numRows() ; ++i )
        {
            for ( j = 0 ; j < lhsres.numCols() ; ++j )
            {
                lhsres("&",i,j) = ( lhsres(i,j) < rhs(i,j) ) ? 1 : 0;
            }
        }
    }

    return lhsres;
}

inline Matrix<double> &gtfill(Matrix<double> &lhsres, const Matrix<double> &rhs)
{
    NiceAssert( lhsres.numRows() == rhs.numRows() );
    NiceAssert( lhsres.numCols() == rhs.numCols() );

    if ( lhsres.numRows() && lhsres.numCols() )
    {
        int i,j;

        for ( i = 0 ; i < lhsres.numRows() ; ++i )
        {
            for ( j = 0 ; j < lhsres.numCols() ; ++j )
            {
                lhsres("&",i,j) = ( lhsres(i,j) > rhs(i,j) ) ? 1 : 0;
            }
        }
    }

    return lhsres;
}

inline Matrix<double> &lefill(Matrix<double> &lhsres, const Matrix<double> &rhs)
{
    NiceAssert( lhsres.numRows() == rhs.numRows() );
    NiceAssert( lhsres.numCols() == rhs.numCols() );

    if ( lhsres.numRows() && lhsres.numCols() )
    {
        int i,j;

        for ( i = 0 ; i < lhsres.numRows() ; ++i )
        {
            for ( j = 0 ; j < lhsres.numCols() ; ++j )
            {
                lhsres("&",i,j) = ( lhsres(i,j) <= rhs(i,j) ) ? 1 : 0;
            }
        }
    }

    return lhsres;
}

inline Matrix<double> &gefill(Matrix<double> &lhsres, const Matrix<double> &rhs)
{
    NiceAssert( lhsres.numRows() == rhs.numRows() );
    NiceAssert( lhsres.numCols() == rhs.numCols() );

    if ( lhsres.numRows() && lhsres.numCols() )
    {
        int i,j;

        for ( i = 0 ; i < lhsres.numRows() ; ++i )
        {
            for ( j = 0 ; j < lhsres.numCols() ; ++j )
            {
                lhsres("&",i,j) = ( lhsres(i,j) >= rhs(i,j) ) ? 1 : 0;
            }
        }
    }

    return lhsres;
}


template <class T> void preallocsubfn(Vector<T> &x, int newalloccols);
template <class T> void preallocsubfn(Vector<T> &x, int newalloccols)
{
    x.prealloc(newalloccols);
}

template <class T> void useStandardAllocationsubfn(Vector<T> &x, int);
template <class T> void useStandardAllocationsubfn(Vector<T> &x, int)
{
    x.useStandardAllocation();
}

template <class T> void useTightAllocationsubfn(Vector<T> &x, int);
template <class T> void useTightAllocationsubfn(Vector<T> &x, int)
{
    x.useTightAllocation();
}

template <class T> void useSlackAllocationsubfn(Vector<T> &x, int);
template <class T> void useSlackAllocationsubfn(Vector<T> &x, int)
{
    x.useSlackAllocation();
}

template <class T>
void Matrix<T>::prealloc(int newallocrows, int newalloccols)
{
    NiceAssert( !nbase );
    NiceAssert( ( newallocrows >= 0 ) || ( newallocrows == -1 ) );
    NiceAssert( ( newalloccols >= 0 ) || ( newalloccols == -1 ) );

    if ( !iscover && content )
    {
        (*content).prealloc(newallocrows);
        (*content).applyOnAll(preallocsubfn,newalloccols);
    }
}

template <class T>
void Matrix<T>::useStandardAllocation(void)
{
    NiceAssert( !nbase );

    if ( !iscover && content )
    {
        (*content).useStandardAllocation();
        (*content).applyOnAll(useStandardAllocationsubfn,0);
    }
}

template <class T>
void Matrix<T>::useTightAllocation(void)
{
    NiceAssert( !nbase );

    if ( !iscover && content )
    {
        (*content).useTightAllocation();
        (*content).applyOnAll(useTightAllocationsubfn,0);
    }
}

template <class T>
void Matrix<T>::useSlackAllocation(void)
{
    NiceAssert( !nbase );

    if ( !iscover && content )
    {
        (*content).useSlackAllocation();
        (*content).applyOnAll(useSlackAllocationsubfn,0);
    }
}














// We have:
//
// inv(B) = Q.inv(diag(e)).Q'
//
// inv(diag(x)+B) = inv(diag(x) + Q.diag(e).Q')
//                = inv(diag(x)) - inv(diag(x)).Q.inv(inv(diag(e))+Q'.inv(diag(x))).Q).Q'.inv(diag(x))
//                  (Q columns are orthonormal)
//                = inv(diag(x)) - inv(diag(x)).Q.inv(inv(diag(e))+inv(diag(x)))).Q'.inv(diag(x))
//                = inv(diag(x)).( diag(x) - Q.inv(inv(diag(e))+inv(diag(x)))).Q' ).inv(diag(x))
//
// Let: t = inv(inv(diag(e)) + inv(diag(x)))    (ti = 0 if either ei = 0 or xi = 0)
// Define: s  = r./x
//         s2 = Q'.s
//         s3 = s2.*t
//         s4 = Q.s3
//         u  = s4./x
//
// y = inv(diag(x)+B).r = ( inv(diag(x)) - inv(diag(x)).Q.inv(inv(diag(e))+inv(diag(x)))).Q'.inv(diag(x)) ).r
//                      = s - inv(diag(x)).Q.diag(t).Q'.s
//                      = s - inv(diag(x)).Q.diag(t).s2
//                      = s - inv(diag(x)).Q.s3
//                      = s - inv(diag(x)).s4
//                      = s - u
//
// Code: loop to calculate t  (include assert( ti != 0 ) and assert( ei != 0 ), and set ti = 0 if inverse sum infinite)
//       t = inv(inv(diag(e)) + inv(diag(x)))
//       y  = r;
//       y /= x;
//       u  = y;
//       leftmult(u,Q);
//       u *= t;
//       rightmult(Q,u);
//       u /= x;
//       y -= u;

template <class T> Vector<T> &eiginv(const Matrix<T> &Q, const Vector<T> &e, Vector<T> &y, const Vector<T> &r)
{
    NiceAssert( Q.isSquare() );
    NiceAssert( Q.numRows() == e.size() );
    NiceAssert( Q.numRows() == r.size() );

    y = r;

    leftmult(y,Q);
    y /= e;
    rightmult(Q,y);

    return y;
}

template <class T> Vector<T> &offeiginv(const Matrix<T> &Q, const Vector<T> &e, Vector<T> &y, const Vector<T> &r, const Vector<T> &x, Vector<T> &t, Vector<T> &u)
{
    NiceAssert( Q.isSquare() );
    NiceAssert( Q.numRows() == e.size() );
    NiceAssert( Q.numRows() == r.size() );

    int i,dim = e.size();

    t.resize(dim);

    for ( i = 0 ; i < dim ; ++i )
    {
        t.set(i,inv(inv(e(i)) + inv(x(i))));
    }

    y  = r;
    y /= x;

    u = y;
    leftmult(u,Q);
    u *= t;
    rightmult(Q,u);
    u /= x;

    y -= u;

    return y;
}

template <class T> Vector<T> &offinveiginv(const Matrix<T> &Q, const Vector<T> &e, Vector<T> &y, const Vector<T> &r, const Vector<T> &xinv, Vector<T> &t, Vector<T> &u)
{
    NiceAssert( Q.isSquare() );
    NiceAssert( Q.numRows() == e.size() );
    NiceAssert( Q.numRows() == r.size() );

    int i,dim = e.size();

    t.resize(dim);

    for ( i = 0 ; i < dim ; ++i )
    {
        t.set(i,inv(inv(e(i)) + xinv(i)));
    }

    y  = r;
    y *= xinv;

    u = y;
    leftmult(u,Q);
    u *= t;
    rightmult(Q,u);
    u *= xinv;

    y -= u;

    return y;
}













template <class T> int testisvnan(const Matrix<T> &x)
{
    int res = 0;

    if ( x.numRows() && x.numCols() )
    {
        int i,j;

        for ( i = 0 ; !res && ( i < x.numRows() ) ; ++i )
        {
            for ( j = 0 ; !res && ( j < x.numCols() ) ; ++j )
            {
                if ( testisvnan(x(i,j)) )
                {
                    res = 1;
                }
            }
        }
    }

    return res;
}

template <class T> int testisinf (const Matrix<T> &x)
{
    int res = 0;

    if ( x.numRows() && x.numCols() && !testisvnan(x) )
    {
        int i,j;

        for ( i = 0 ; !res && ( i < x.numRows() ) ; ++i )
        {
            for ( j = 0 ; !res && ( j < x.numCols() ) ; ++j )
            {
                if ( testisinf(x(i,j)) )
                {
                    res = 1;
                }
            }
        }
    }

    return res;
}

template <class T> int testispinf(const Matrix<T> &x)
{
    int pinfcnt = 0;

    if ( x.numRows() && x.numCols() )
    {
        int i,j;

        for ( i = 0 ; i < x.numRows() ; ++i )
        {
            for ( j = 0 ; j < x.numCols() ; ++j )
            {
                if ( testispinf(x(i,j)) )
                {
                    ++pinfcnt;
                }
            }
        }
    }

    return ( ( pinfcnt == ((x.numRows())*(x.numCols())) ) && ( ((x.numRows())*(x.numCols())) > 0 ) ) ? 1 : 0;
}

template <class T> int testisninf(const Matrix<T> &x)
{
    int ninfcnt = 0;

    if ( x.numRows() && x.numCols() )
    {
        int i,j;

        for ( i = 0 ; i < x.numRows() ; ++i )
        {
            for ( j = 0 ; j < x.numRows() ; ++j )
            {
                if ( testisninf(x(i,j)) )
                {
                    ++ninfcnt;
                }
            }
        }
    }

    return ( ( ninfcnt == ((x.numRows())*(x.numCols())) ) && ( ((x.numRows())*(x.numCols())) > 0 ) ) ? 1 : 0;
}










template <class T>
std::ostream &operator<<(std::ostream &output, const Matrix<T> &src)
{
    int numRows = src.numRows();
    int numCols = src.numCols();

    int i,j;

    output << "[ ";

    if ( numRows )
    {
	for ( i = 0 ; i < numRows ; ++i )
	{
	    if ( numCols )
	    {
		for ( j = 0 ; j < numCols ; ++j )
		{
		    if ( j < numCols-1 )
		    {
			output << src(i,j) << " \t";
		    }

		    else
		    {
			output << src(i,j) << " \t";
		    }
		}

		if ( i < numRows-1 )
		{
		    output << ";\n  ";
		}

		else
		{
		    output << "  ";
		}
	    }

	    else
	    {
                output << ";  ";
	    }
	}
    }

    else
    {
	//if ( numCols )
	{
	    for ( j = 0 ; j < numCols ; ++j )
	    {
                output << ",  ";
	    }
	}
    }

    output << "]";

    return output;
}

template <class T>
std::istream &operator>>(std::istream &input, Matrix<T> &dest)
{
    char ss,tt;

    //OLD VERSION input >> buffer;
    //OLD VERSION 
    //OLD VERSION NiceAssert( !strcmp(buffer,"[") );

    while ( isspace(input.peek()) )
    {
	input.get(ss);
    }

    input.get(tt);

    // Need to allow for matrices in gentyp with format M:[ ... ], not [ ... ]

    if ( tt == 'M' )
    {
        input.get(tt);
        NiceAssert( tt = ':' );
        input.get(tt);
    }

    NiceAssert( tt == '[' );

    int numRows = 0;
    int numCols = 0;
    int colcnt  = 0;
    int elmread = 0;

    while ( isspace(input.peek()) )
    {
        input.get(ss);
    }

    if ( input.peek() != ']' )
    {
        while ( input.peek() != ']' )
	{
            if ( input.peek() == ';' )
	    {
                input.get(tt);

		if ( !numRows && !colcnt )
		{
		    goto semicoloncount;
		}

                NiceAssert( elmread );
                (void) elmread;

		if ( !numRows )
		{
		    numCols = colcnt;
		}

                NiceAssert( colcnt == numCols );

		++numRows;

                elmread = 0;
		colcnt  = 0;
	    }

            else if ( input.peek() == ',' )
	    {
                input.get(tt);

		while ( isspace(input.peek()) )
		{
		    input.get(ss);
		}

		if ( !numRows && !colcnt )
		{
		    goto commacount;
		}

                NiceAssert( elmread );

                elmread = 0;
	    }

	    else
	    {
		if ( dest.numRows() == numRows )
		{
		    dest.addRow(numRows);
		}

		if ( ( dest.numCols() == colcnt ) && !numRows )
		{
		    dest.addCol(colcnt);
		}

		input >> dest("&",numRows,colcnt);

                elmread = 1;
		++colcnt;
	    }

            while ( isspace(input.peek()) )
            {
                input.get(ss);
            }
	}

	if ( !numRows )
	{
	    numCols = colcnt;
	}

        NiceAssert( colcnt == numCols );

	++numRows;
    }

    input.get(tt);

    NiceAssert( tt == ']' );

    dest.resize(numRows,numCols);

    return input;

semicoloncount:

    numCols = 0;
    numRows = 0;

    while ( tt == ';' )
    {
	++numRows;

	while ( isspace(input.peek()) )
	{
	    input.get(ss);
	}

        input.get(tt);
    }

    NiceAssert( tt == ']' );

    dest.resize(numRows,numCols);

    return input;

commacount:

    numCols = 0;
    numRows = 0;

    while ( tt == ',' )
    {
	++numCols;

	while ( isspace(input.peek()) )
	{
	    input.get(ss);
	}

        input.get(tt);
    }

    NiceAssert( tt == ']' );

    dest.resize(numRows,numCols);

    return input;
}


template <class T> 
std::istream &streamItIn(std::istream &input, Matrix<T>& dest, int processxyzvw)
{
    (void) processxyzvw;

    input >> dest;

    return input;
}

template <class T> 
std::ostream &streamItOut(std::ostream &output, const Matrix<T>& src, int retainTypeMarker)
{
    (void) retainTypeMarker;

    output << src;

    return output;
}

#endif

