#ifndef nlopt_base_h
#define nlopt_base_h

#include <stdlib.h>
#include <stddef.h> /* for ptrdiff_t */
#include "basefn.hpp"
#include "memdebug.hpp"
#include "niceassert.hpp"
#include "clockbase.hpp"


std::ostream &directstream(void);



// Something

typedef double (*nlopt_func)(unsigned n, const double *x,
                             double *gradient, /* nullptr if not needed */
                             void *func_data);

typedef void (*nlopt_mfunc)(unsigned m, double *result,
                            unsigned n, const double *x,
                             double *gradient, /* nullptr if not needed */
                             void *func_data);

typedef struct {
     unsigned m; /* dimensional of constraint: mf maps R^n -> R^m */
     nlopt_func f; /* one-dimensional constraint, requires m == 1 */
     nlopt_mfunc mf;
     //nlopt_precond pre; /* preconditioner for f (nullptr if none or if mf) */
     void *f_data;
     double *tol;
} nlopt_constraint;

// Stopping crit

typedef struct {
     unsigned n; // problem dimension
     double minf_max;
     double ftol_rel;
     double ftol_abs;
     double xtol_rel;
     const double *xtol_abs;
     int nevals; // 0
     int maxeval;
     double maxtime;
     svmvolatile int *killSwitch;
} nlopt_stopping;

// Return codes

typedef enum {
     NLOPT_FAILURE = -1, /* generic failure code */
     NLOPT_INVALID_ARGS = -2,
     NLOPT_OUT_OF_MEMORY = -3,
     NLOPT_ROUNDOFF_LIMITED = -4,
     NLOPT_FORCED_STOP = -5,
     NLOPT_SUCCESS = 1, /* generic success code */
     NLOPT_STOPVAL_REACHED = 2,
     NLOPT_FTOL_REACHED = 3,
     NLOPT_XTOL_REACHED = 4,
     NLOPT_MAXEVAL_REACHED = 5,
     NLOPT_MAXTIME_REACHED = 6
} nlopt_result;







#define NLOPT_MINF_MAX_REACHED NLOPT_STOPVAL_REACHED

inline int nlopt_isinf(double x);
inline int nlopt_isinf(double x) {
     return fabs(x) >= HUGE_VAL * 0.99
#ifdef HAVE_ISINF
          || testisinf(x)
#endif
          ;
}

// Timey-wimey stuff

//#define TIMESTAMPTYPE double
#define TIMESTAMPTYPE time_used

inline TIMESTAMPTYPE getstarttime(void);
inline TIMESTAMPTYPE getstarttime(void)
{
    return TIMECALL;
}

int relstop(double vold, double vnew, double reltol, double abstol);
int nlopt_stop_time(const TIMESTAMPTYPE &starttime, const nlopt_stopping *stop);
int nlopt_stop_time_(const TIMESTAMPTYPE &starttime, double xmtrtime);
int nlopt_stop_forced(const nlopt_stopping *stop);
int nlopt_stop_evals(const nlopt_stopping *stop);
int nlopt_stop_ftol(const nlopt_stopping *stop, double f, double oldf);
int nlopt_stop_x(const nlopt_stopping *stop,
                 const double *x, const double *oldx);

/* re-entrant qsort */
void nlopt_qsort_r(void *base_, size_t nmemb, size_t size, void *thunk,
                   int (*compar)(void *, const void *, const void *));

typedef double *rb_key; /* key type ... double* is convenient for us,
                           but of course this could be cast to anything
                           desired (although void* would look more generic) */

typedef enum { RED, BLACK } rb_color;
typedef struct rb_node_s {
     struct rb_node_s *p, *r, *l; /* parent, right, left */
     rb_key k; /* key (and data) */
     rb_color c;
} rb_node;

/* Red-black tree stuff because why not */

typedef int (*rb_compare)(rb_key k1, rb_key k2);

typedef struct {
     rb_compare compare;
     rb_node *root;
     int N; /* number of nodes */
} rb_tree;

void rb_tree_init(rb_tree *t, rb_compare compare);
void rb_tree_destroy(rb_tree *t);
void rb_tree_destroy_with_keys(rb_tree *t);
rb_node *rb_tree_insert(rb_tree *t, rb_key k);
int rb_tree_check(rb_tree *t);
rb_node *rb_tree_find(rb_tree *t, rb_key k);
rb_node *rb_tree_find_le(rb_tree *t, rb_key k);
rb_node *rb_tree_find_lt(rb_tree *t, rb_key k);
rb_node *rb_tree_find_gt(rb_tree *t, rb_key k);
rb_node *rb_tree_resort(rb_tree *t, rb_node *n);
rb_node *rb_tree_min(rb_tree *t);
rb_node *rb_tree_max(rb_tree *t);
rb_node *rb_tree_succ(rb_node *n);
rb_node *rb_tree_pred(rb_node *n);
void rb_tree_shift_keys(rb_tree *t, ptrdiff_t kshift);

/* To change a key, use rb_tree_find+resort.  Removing a node
   currently wastes memory unless you change the allocation scheme
   in redblack.c */
rb_node *rb_tree_remove(rb_tree *t, rb_node *n);





/* constraints */

unsigned nlopt_count_constraints(unsigned p, const nlopt_constraint *c);
unsigned nlopt_max_constraint_dim(unsigned p, const nlopt_constraint *c);
void nlopt_eval_constraint(double *result, double *grad, const nlopt_constraint *c, unsigned n, const double *x);








#endif
