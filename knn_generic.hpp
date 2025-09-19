
//
// k-nearest-neighbour base class
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _knn_generic_h
#define _knn_generic_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "ml_base.hpp"
#include "kcache.hpp"



class KNN_Generic;


// Swap and zeroing (restarting) functions

void evalKKNN_dist(double &res, int i, int j, const gentype **pxyprod, const void *owner);

inline void qswap(KNN_Generic &a, KNN_Generic &b);
inline KNN_Generic &setzero(KNN_Generic &a);

class KNN_Generic : public ML_Base
{
public:

    // Constructors, destructors, assignment etc..

    KNN_Generic();
    KNN_Generic(const KNN_Generic &src);
    KNN_Generic(const KNN_Generic &src, const ML_Base *xsrc);
    KNN_Generic &operator=(const KNN_Generic &src) { assign(src); return *this; }
    virtual ~KNN_Generic();

    virtual int  prealloc    (int expectedN)       override;
    virtual int  preallocsize(void)          const override;
    virtual void setmemsize  (int memsize)         override { kerncache.setmemsize(memsize,kerncache.get_min_rowdim()); return; }

    virtual void assign       (const ML_Base &src, int onlySemiCopy = 0) override;
    virtual void semicopy     (const ML_Base &src)                       override;
    virtual void qswapinternal(ML_Base &b)                               override;

    virtual int getparam (int ind, gentype         &val, const gentype         &xa, int ia, const gentype         &xb, int ib, charptr &desc) const override;
    virtual int egetparam(int ind, Vector<gentype> &val, const Vector<gentype> &xa, int ia, const Vector<gentype> &xb, int ib               ) const override;

    virtual std::ostream &printstream(std::ostream &output, int dep) const override;
    virtual std::istream &inputstream(std::istream &input          )       override;

    virtual       ML_Base &getML     (void)       override { return static_cast<      ML_Base &>(getKNN());      }
    virtual const ML_Base &getMLconst(void) const override { return static_cast<const ML_Base &>(getKNNconst()); }

    // Information functions (training data):

    virtual int tspaceDim   (void)       const override { return 1; }
    virtual int tspaceSparse(void)       const override { return 0; }
    virtual int xspaceSparse(void)       const override { return 1; }

    virtual int isTrained(void) const override { return 1; }

    virtual int isUnderlyingScalar(void) const override { return 1; }
    virtual int isUnderlyingVector(void) const override { return 0; }
    virtual int isUnderlyingAnions(void) const override { return 0; }

    virtual int numInternalClasses(void)  const override;
    virtual int isenabled         (int i) const override { return d()(i); }

    virtual int memsize(void) const override { return kerncache.get_memsize(); }

    virtual double sparlvl(void) const override { return N()-NNC(0) ? ( ( k() > N()-NNC(0) ) ? k() : N()-NNC(0) )/((double) N()-NNC(0)) : 1; }

    virtual const Vector<int>     &d         (void) const override { return dd;     }
    virtual const Vector<int>     &alphaState(void) const override { return onevec; }
    virtual const Vector<gentype> &alphaVal  (void) const override { NiceThrow("alphaVal has no meaning here"); static const Vector<gentype> dummy; return dummy; }

    virtual double alphaVal(int i) const override { (void) i; NiceThrow("alpha has no meaning here"); return 0.0; }

    // Kernel Modification

    virtual int resetKernel(int modind = 1, int onlyChangeRowI = -1, int updateInfo = 1) override;
    virtual int setKernel(const MercerKernel &xkernel, int modind = 1, int onlyChangeRowI = -1) override;
    virtual void prepareKernel(void) override { return; }
    virtual double tuneKernel(int, double, int = 1, int = 0, const tkBounds * = nullptr, paraDef * = nullptr) override { return 0; }
    virtual double evalkernel(int, const paraDef &, const Vector<double> &) override { return 0; }

    // Training set modification:

    virtual int addTrainingVector (int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2) override;
    virtual int qaddTrainingVector(int i, const gentype &y,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2) override;

    virtual int addTrainingVector (int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override;
    virtual int qaddTrainingVector(int i, const Vector<gentype> &y,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) override;

    virtual int removeTrainingVector(int i                                      ) override { SparseVector<gentype> x; gentype y; return removeTrainingVector(i,y,x); }
    virtual int removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x) override;
    virtual int removeTrainingVector(int i, int num                             ) override { return ML_Base::removeTrainingVector(i,num); }

    virtual int setd(int                i, int                d) override;
    virtual int setd(const Vector<int> &i, const Vector<int> &d) override;
    virtual int setd(                      const Vector<int> &d) override;

    // General modification and autoset functions

    virtual int realign(void) override;

    // Evaluation Functions:

    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = nullptr) const override;

    virtual void dgTrainingVector(Vector<gentype>         &res, gentype        &resn, int i) const override;
    virtual void dgTrainingVector(Vector<double>          &res, double         &resn, int i) const override { ML_Base::dgTrainingVector(res,resn,i); return; }
    virtual void dgTrainingVector(Vector<Vector<double> > &res, Vector<double> &resn, int i) const override { ML_Base::dgTrainingVector(res,resn,i); return; }
    virtual void dgTrainingVector(Vector<d_anion>         &res, d_anion        &resn, int i) const override { ML_Base::dgTrainingVector(res,resn,i); return; }

    virtual void dgTrainingVector(Vector<gentype>         &res, const Vector<int> &i) const override { ML_Base::dgTrainingVector(res,i); return; }
    virtual void dgTrainingVector(Vector<double>          &res, const Vector<int> &i) const override { ML_Base::dgTrainingVector(res,i); return; }
    virtual void dgTrainingVector(Vector<Vector<double> > &res, const Vector<int> &i) const override { ML_Base::dgTrainingVector(res,i); return; }
    virtual void dgTrainingVector(Vector<d_anion>         &res, const Vector<int> &i) const override { ML_Base::dgTrainingVector(res,i); return; }

    // ================================================================
    //     KNN Specific functions
    // ================================================================

    virtual       KNN_Generic &getKNN     (void)       { return *this; }
    virtual const KNN_Generic &getKNNconst(void) const { return *this; }

    // Information functions (training data):

    virtual int k  (void) const { return kay; }
    virtual int ktp(void) const { return wkt; }

    // General modification and autoset functions

    virtual int setk  (int xk) { NiceAssert( xk >  0 ); incgvernum(); kay = xk; return 1; }
    virtual int setktp(int xk) { NiceAssert( xk >= 0 ); incgvernum(); wkt = xk; return 1; }

    // Not sure how this fits

    virtual const Matrix<double> &Gp(void) const { return *Gpdist; }

protected:

    // Core of method to be overloaded: given the class labels and distances
    // (squared) to the k nearest neighbours, calculate result res.

    virtual void hfn(gentype &res, const Vector<gentype> &yk, const Vector<double> &kdistsq, const Vector<double> &weights, int Nnz, int effkay) const
    {
        (void) res;
        (void) yk;
        (void) kdistsq;
        (void) effkay;
        (void) Nnz;
        (void) weights;

        NiceThrow("KNN generic container has no specifics");

        return;
    }

    virtual void hfn(double &res, const Vector<double> &yk, const Vector<double> &kdistsq, const Vector<double> &weights, int Nnz, int effkay) const
    {
        (void) res;
        (void) yk;
        (void) kdistsq;
        (void) effkay;
        (void) Nnz;
        (void) weights;

        NiceThrow("KNN generic container has no specifics");

        return;
    }

    virtual void hfn(Vector<double> &res, const Vector<Vector<double> > &yk, const Vector<double> &kdistsq, const Vector<double> &weights, int Nnz, int effkay) const
    {
        (void) res;
        (void) yk;
        (void) kdistsq;
        (void) effkay;
        (void) Nnz;
        (void) weights;

        NiceThrow("KNN generic container has no specifics");

        return;
    }

    virtual void hfn(d_anion &res, const Vector<d_anion> &yk, const Vector<double> &kdistsq, const Vector<double> &weights, int Nnz, int effkay) const
    {
        (void) res;
        (void) yk;
        (void) kdistsq;
        (void) effkay;
        (void) Nnz;
        (void) weights;

        NiceThrow("KNN generic container has no specifics");

        return;
    }

    // Fast versions: enable as required

    virtual int ggTrainingVectorInt(double         &resg, int i, int retaltg = 0, gentype ***pxyprodi = nullptr) const;
    virtual int ggTrainingVectorInt(Vector<double> &resg, int i, int retaltg = 0, gentype ***pxyprodi = nullptr) const;
    virtual int ggTrainingVectorInt(d_anion        &resg, int i, int retaltg = 0, gentype ***pxyprodi = nullptr) const;

private:

    int kay;
    protected:
    Vector<int> dd;
    Vector<int> onevec;
    mutable Vector<double> kdistscr;
    mutable Vector<int> iiscr;
    private:

    // Calculate distance of xx to all points, then sort from smallest to
    // largest, putting distances (unsorted) in kdist and indices (sorted) in
    // ii.  Sorting terminates after finding first effkay smallest distances,
    // so ii(0,1,effkay-1) are the indices of the first effkay smallest
    // distances and kdist(ii(0,1,effkay-1)) the distances themselves.
    // Points constrained to zero are not included in calculation, and will
    // be excluded form ii, so on return ii.size() will equal the number of
    // unconstrained x's.  Will return effkay taking this into account.

    //int distcalc(int effkay, Vector<int> &ii, Vector<double> &kdist, const SparseVector<gentype> &xx, int &Nnz) const;
    int distcalcTrainingVector(int effkay, Vector<int> &ii, Vector<double> &kdist, int j, int &Nnz) const;
    void ddistdxcalcTrainingVector(Vector<double> &igrad, Vector<double> &jgrad, const Vector<int> &ii, int j) const;
    double calcweight(double dist) const;
    double calcweightgrad(double dist) const;

    // Distances matrix (training vectors) and associated cache

    Kcache<double> kerncache;
    Matrix<double> *Gpdist;

    // Weight calculations: the regressor has the generic form:
    //
    // 1/k sum_i K(dist_i)/Kbar y_i
    //
    // where Kbar is the sum of K and dist_i is the distance.  *K is not
    // stored in the kernel in the normal sense - the distance metric is*
    //
    // 0: Rectangular: K(d) = 1/2 I(|d| <= 1)
    // 1: Triangular:  K(d) = (1-|d|).I(|d| <= 1)
    // 2: Epanechnikov: K(d) = 3/4 (1-d^2) I(|d| <= 1)
    // 3: Quartic/biweight: K(d) = 15/16  (1-d^2)^2 I(|d| <= 1)
    // 4: Triweight: K(d) = 35/32 (1-d^2)^3 I...
    // 5: Cosine: K(d) = pi/4 cos(d.pi/2) I...
    // 6: Gauss: K(d) = 1/sqrt(2.pi) exp(-d^2/2)
    // 7: Inversion: K(d) = 1/|d|

    int wkt;
};

inline double norm2(const KNN_Generic &a);
inline double abs2 (const KNN_Generic &a);

inline double norm2(const KNN_Generic &a) { return a.RKHSnorm(); }
inline double abs2 (const KNN_Generic &a) { return a.RKHSabs();  }

inline void qswap(KNN_Generic &a, KNN_Generic &b)
{
    a.qswapinternal(b);

    return;
}

inline KNN_Generic &setzero(KNN_Generic &a)
{
    a.restart();

    return a;
}

inline void KNN_Generic::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    KNN_Generic &b = dynamic_cast<KNN_Generic &>(bb.getML());

    ML_Base::qswapinternal(b);

    qswap(kay      ,b.kay      );
    qswap(dd       ,b.dd       );
    qswap(onevec   ,b.onevec   );
    qswap(kdistscr ,b.kdistscr );
    qswap(iiscr    ,b.iiscr    );
    qswap(kerncache,b.kerncache);
    qswap(wkt      ,b.wkt      );

    Matrix<double> *tGp;

    tGp = Gpdist; Gpdist = b.Gpdist; b.Gpdist = tGp;

    (kerncache).cheatSetEvalArg((void *) this);
    (Gpdist)->cheatsetcdref((void *) &(kerncache));

    (Gpdist)->cheatsetcdref((void *) &(kerncache));

    (b.kerncache).cheatSetEvalArg((void *) &b);
    (b.Gpdist)->cheatsetcdref((void *) &(b.kerncache));

    (b.Gpdist)->cheatsetcdref((void *) &(b.kerncache));

    return;
}

inline void KNN_Generic::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const KNN_Generic &b = dynamic_cast<const KNN_Generic &>(bb.getMLconst());

    ML_Base::semicopy(b);

    kay      = b.kay;
    dd       = b.dd;
    onevec   = b.onevec;
    kdistscr = b.kdistscr;
    iiscr    = b.iiscr;
    wkt      = b.wkt;

    return;
}

inline void KNN_Generic::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const KNN_Generic &src = dynamic_cast<const KNN_Generic &>(bb.getMLconst());

    ML_Base::assign(src,onlySemiCopy);

    kay      = src.kay;
    dd       = src.dd;
    onevec   = src.onevec;
    kdistscr = src.kdistscr;
    iiscr    = src.iiscr;
    wkt      = src.wkt;

    MEMDEL(Gpdist);
    Gpdist = nullptr;

    kerncache = src.kerncache;
    kerncache.cheatSetEvalArg((void *) this);

    MEMNEW(Gpdist,Matrix<double>(Kcache_celm_v_double,Kcache_celm_double,Kcache_crow_double,(void *) &kerncache,dd.size(),dd.size()));

    return;
}

#endif
