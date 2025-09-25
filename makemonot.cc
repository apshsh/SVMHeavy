
//
// Add inducing-point constraints to enforce monotonicity
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include <cmath>
#include "makemonot.hpp"
#include "randfun.hpp"

ML_Base &makeMonotone(ML_Base &ml,
                      int n,
                      int t,
                      const SparseVector<gentype> &xtemp,
                      const SparseVector<double> &lb,
                      const SparseVector<double> &ub,
                      int d,
                      gentype y,
                      double Cweigh,
                      double epsweigh)
{
    // Generate index vector

    SparseVector<double> indref(lb);
    indref += ub;

    // Get dimension

    int dim = indref.indsize();

    // Default n

    n = ( n == -1 ) ? ((int) std::ceil(std::pow(10.0,(double) dim))) : n;

    // Number of points per axis (if relevant)
    // (m+1)^dim = n -> m = ceil(n^(1/dim)) - 1
    // Round up n if relevant

    int m = (int) std::ceil(std::pow(((double) n),1.0/((double) dim))) - 1;

    if ( !t )
    {
        n = (int) std::pow(m+1,dim);
    }

    // Generate random or grid points
    // Note that we let all x be the template xtemp to
    // ensure relevant gradient etc constraints are present

    Vector<SparseVector<gentype>> x(n,xtemp);

    if ( !t )
    {
        // Working on a grid

        Vector<int> gridind(dim,0);

        for ( int i = 0 ; i < n ; ++i )
        {
            for ( int j = 0 ; j < dim ; ++j )
            {
                x("&",i)("&",indref.ind(j)).force_double() = lb(indref.ind(j)) + ( (ub(indref.ind(j))-lb(indref.ind(j))) * (((double) gridind(j))/((double) m)) );
            }

            for ( int j = 0 ; j < dim ; ++j )
            {
                if ( ++gridind("&",j) <= m )
                {
                    break;
                }

                gridind("&",j) = 0;
            }
        }
    }

    else
    {
        // Random spread

        for ( int i = 0 ; i < n ; ++i )
        {
            for ( int j = 0 ; j < dim ; ++j )
            {
                randufill(x("&",i)("&",indref.ind(j)).force_double(),lb(indref.ind(j)),ub(indref.ind(j)));
            }
        }
    }

    // Add grid to ML, setting d as we go

    gentype yy(y);

    yy.isNomConst = true; // suppress counting when working out errors etc!

    for ( int i = 0 ; i < n ; ++i )
    {
        ml.qaddTrainingVector(ml.N(),yy,x("&",i),Cweigh,epsweigh,d);
        ml.y()(ml.N()-1).isNomConst = true;
    }

    // and we're done

    return ml;
}
