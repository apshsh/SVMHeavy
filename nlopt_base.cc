
#include "nlopt_base.hpp"

std::ostream &directstream(void)
{
//    return std::cerr;
    static NullOStream devnullstream;

    return devnullstream;
//    return errstream();
}

/* utility routines to implement the various stopping criteria */

int relstop(double vold, double vnew, double reltol, double abstol)
{
     if (nlopt_isinf(vold)) return 0;
     return(fabs(vnew - vold) < abstol 
	    || fabs(vnew - vold) < reltol * (fabs(vnew) + fabs(vold)) * 0.5
	    || (reltol > 0 && vnew == vold)); /* catch vnew == vold == 0 */
}

int nlopt_stop_ftol(const nlopt_stopping *s, double f, double oldf)
{
     return (relstop(oldf, f, s->ftol_rel, s->ftol_abs));
}

int nlopt_stop_f(const nlopt_stopping *s, double f, double oldf)
{
     return (f <= s->minf_max || nlopt_stop_ftol(s, f, oldf));
}

int nlopt_stop_x(const nlopt_stopping *s, const double *x, const double *oldx)
{
     unsigned i;
     for (i = 0; i < s->n; ++i)
	  if (!relstop(oldx[i], x[i], s->xtol_rel, s->xtol_abs[i]))
	       return 0;
     return 1;
}

int nlopt_stop_dx(const nlopt_stopping *s, const double *x, const double *dx)
{
     unsigned i;
     for (i = 0; i < s->n; ++i)
	  if (!relstop(x[i] - dx[i], x[i], s->xtol_rel, s->xtol_abs[i]))
	       return 0;
     return 1;
}

static double sc(double x, double smin, double smax)
{
     return smin + x * (smax - smin);
}

/* some of the algorithms rescale x to a unit hypercube, so we need to
   scale back before we can compare to the tolerances */
int nlopt_stop_xs(const nlopt_stopping *s,
		  const double *xs, const double *oldxs,
		  const double *scale_min, const double *scale_max)
{
     unsigned i;
     for (i = 0; i < s->n; ++i)
	  if (relstop(sc(oldxs[i], scale_min[i], scale_max[i]), 
		      sc(xs[i], scale_min[i], scale_max[i]),
		      s->xtol_rel, s->xtol_abs[i]))
	       return 1;
     return 0;
}

int nlopt_stop_evals(const nlopt_stopping *s)
{
     return (s->maxeval > 0 && s->nevals >= s->maxeval);
}

int nlopt_stop_evalstime(const TIMESTAMPTYPE &starttime, const nlopt_stopping *stop)
{
     return nlopt_stop_evals(stop) || nlopt_stop_time(starttime,stop);
}

int nlopt_stop_time_(const TIMESTAMPTYPE &starttime, double xmtrtime)
{
//(void) starttime;
//(void) xmtrtime;
//    return 0;

    TIMESTAMPTYPE curr_time;
    int timeout = 0;
    double *uservars[] = { nullptr };
    const char *varnames[] = { nullptr };
    const char *vardescr[] = { nullptr };

    if ( xmtrtime > 1 )
    {
        curr_time = TIMECALL;

        if ( TIMEDIFFSEC(curr_time,starttime) > xmtrtime )
        {
            timeout = 1;
        }
    }

    if ( !timeout )
    {
        timeout = kbquitdet("Nelder-Mead optimisation",uservars,varnames,vardescr);
    }

    return timeout;
}

int nlopt_stop_time(const TIMESTAMPTYPE &starttime, const nlopt_stopping *s)
{
     return nlopt_stop_time_(starttime, s->maxtime);
}

int nlopt_stop_forced(const nlopt_stopping *stop)
{
     return stop->killSwitch && *(stop->killSwitch);
}

unsigned nlopt_count_constraints(unsigned p, const nlopt_constraint *c)
{
     unsigned i, count = 0;
     for (i = 0; i < p; ++i)
	  count += c[i].m;
     return count;
}

unsigned nlopt_max_constraint_dim(unsigned p, const nlopt_constraint *c)
{
     unsigned i, max_dim = 0;
     for (i = 0; i < p; ++i)
	  if (c[i].m > max_dim)
	       max_dim = c[i].m;
     return max_dim;
}

void nlopt_eval_constraint(double *result, double *grad,
			   const nlopt_constraint *c,
			   unsigned n, const double *x)
{
     if (c->f)
	  result[0] = c->f(n, x, grad, c->f_data);
     else
	  c->mf(c->m, result, n, x, grad, c->f_data);
}

















