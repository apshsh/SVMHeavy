
//
// Kernel cache class
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "kcache.hpp"

const Vector<double>  &Kcache_crow_double (int numi, const void *owner, retVector<double> &tmp)
{
    Kcache<double> *typed_owner = (Kcache<double> *) owner;

    return typed_owner->getrow(numi,tmp);
}

const Vector<gentype> &Kcache_crow_gentype(int numi, const void *owner, retVector<gentype> &tmp)
{
    Kcache<gentype> *typed_owner = (Kcache<gentype> *) owner;

    return typed_owner->getrow(numi,tmp);
}

const Vector<Matrix<double> > &Kcache_crow_matrix(int numi, const void *owner, retVector<Matrix<double> > &tmp)
{
    Kcache<Matrix<double> > *typed_owner = (Kcache<Matrix<double> > *) owner;

    return typed_owner->getrow(numi,tmp);
}

const double  &Kcache_celm_double (int numi, int numj, const void *owner, retVector<double> &tmp)
{
    Kcache<double> *typed_owner = (Kcache<double> *) owner;

    return typed_owner->getval(numi,numj,tmp);
}

double  Kcache_celm_v_double (int numi, int numj, const void *owner, retVector<double> &tmp)
{
    Kcache<double> *typed_owner = (Kcache<double> *) owner;

    return typed_owner->getval_v(numi,numj,tmp);
}

const gentype &Kcache_celm_gentype(int numi, int numj, const void *owner, retVector<gentype> &tmp)
{
    Kcache<gentype> *typed_owner = (Kcache<gentype> *) owner;

    return typed_owner->getval(numi,numj,tmp);
}

const Matrix<double> &Kcache_celm_matrix(int numi, int numj, const void *owner, retVector<Matrix<double> > &tmp)
{
    Kcache<Matrix<double> > *typed_owner = (Kcache<Matrix<double> > *) owner;

    return typed_owner->getval(numi,numj,tmp);
}
