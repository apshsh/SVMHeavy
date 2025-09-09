
//
// Various simplified variants of random number generators.
//
// Version: split off basefn
// Date: 11/09/2024
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//


#ifndef _randfun_h
#define _randfun_h


// randufill: uniform random number between zero and one
// randnfill: gaussian normal N(0,1) random number
// randrfill: random -1/+1 draw
// rand_fill: other distributions indicated by _, see code below
//
// randfill: generate random number from distribution disttype
//           disttype is distribution type (see code)
//           if disttype == 'S' then seed the RNG with seedval (use true random if seedval == -2)


double &randfill(double &res, char disttype, int seedval = -1, double a = 0, double b = 1, int t = 10, double p = 0.5);

// Seeding

inline void randseed(int seedval);

// Versions defined in randfill (using default parameters)

inline double &randrfill(double &res, double p);                  // Rademacher
inline double &randEfill(double &res, double p);                  // Bernoulli
inline double &randbfill(double &res, int t, double p);           // Binomial
inline double &randBfill(double &res, int k, double p);           // Negative binomial
inline double &randgfill(double &res, double p);                  // Geometric
inline double &randpfill(double &res, double mu);                 // Poisson
inline double &randefill(double &res, double lambda);             // Exponential
inline double &randGfill(double &res, double alpha, double beta); // Gamma
inline double &randwfill(double &res, double a, double b);        // Weibull
inline double &randufill(double &res, double a, double b);        // Uniform
inline double &randxfill(double &res, double a, double b);        // Extreme-value
inline double &randnfill(double &res, double mu, double sig);     // Normal
inline double &randlfill(double &res, double m, double s);        // lognormal
inline double &randcfill(double &res, double n);                  // Chi-squared
inline double &randCfill(double &res, double a, double b);        // Cauchy
inline double &randffill(double &res, double m, double n);        // Fisher F-distribution
inline double &randtfill(double &res, double n);                  // Student-t

inline double &randrfill(double &res, double p)                  { return randfill(res,'r',-1,0 ,1     ,10,p    ); }
inline double &randEfill(double &res, double p)                  { return randfill(res,'E',-1,0 ,1     ,10,p    ); }
inline double &randbfill(double &res, int t, double p)           { return randfill(res,'b',-1,0 ,1     ,t ,p    ); }
inline double &randBfill(double &res, int k, double p)           { return randfill(res,'B',-1,0 ,1     ,k ,p    ); }
inline double &randgfill(double &res, double p)                  { return randfill(res,'g',-1,0 ,1     ,10,p    ); }
inline double &randpfill(double &res, double mu)                 { return randfill(res,'p',-1,0 ,mu             ); }
inline double &randefill(double &res, double lambda)             { return randfill(res,'e',-1,0 ,lambda         ); }
inline double &randGfill(double &res, double alpha, double beta) { return randfill(res,'G',-1,0 ,beta  ,10,alpha); }
inline double &randwfill(double &res, double a, double b)        { return randfill(res,'w',-1,0 ,b     ,10,a    ); }
inline double &randufill(double &res, double a, double b)        { return randfill(res,'u',-1,a ,b              ); }
inline double &randxfill(double &res, double a, double b)        { return randfill(res,'x',-1,a ,b              ); }
inline double &randnfill(double &res, double mu, double sig)     { return randfill(res,'n',-1,mu,sig            ); }
inline double &randlfill(double &res, double m, double s)        { return randfill(res,'l',-1,m ,s              ); }
inline double &randcfill(double &res, double n)                  { return randfill(res,'c',-1,0 ,n              ); }
inline double &randCfill(double &res, double a, double b)        { return randfill(res,'C',-1,a ,b              ); }
inline double &randffill(double &res, double m, double n)        { return randfill(res,'f',-1,0 ,n     ,10,m    ); }
inline double &randtfill(double &res, double n)                  { return randfill(res,'t',-1,0 ,n              ); }

inline double &randrfill(double &res);
inline double &randEfill(double &res);
inline double &randbfill(double &res);
inline double &randBfill(double &res);
inline double &randgfill(double &res);
inline double &randpfill(double &res);
inline double &randefill(double &res);
inline double &randGfill(double &res);
inline double &randwfill(double &res);
inline double &randufill(double &res);
inline double &randxfill(double &res);
inline double &randnfill(double &res);
inline double &randlfill(double &res);
inline double &randcfill(double &res);
inline double &randCfill(double &res);
inline double &randffill(double &res);
inline double &randtfill(double &res);

inline double &randrfill(double &res) { return randrfill(res,0.5);    }
inline double &randEfill(double &res) { return randEfill(res,0.5);    }
inline double &randbfill(double &res) { return randbfill(res,10,0.5); }
inline double &randBfill(double &res) { return randBfill(res,10,0.5); }
inline double &randgfill(double &res) { return randgfill(res,0.5);    }
inline double &randpfill(double &res) { return randpfill(res,1);      }
inline double &randefill(double &res) { return randefill(res,1);      }
inline double &randGfill(double &res) { return randGfill(res,1,1);    }
inline double &randwfill(double &res) { return randwfill(res,0.5,1);  }
inline double &randufill(double &res) { return randufill(res,0,1);    }
inline double &randxfill(double &res) { return randxfill(res,0,1);    }
inline double &randnfill(double &res) { return randnfill(res,0,1);    }
inline double &randlfill(double &res) { return randlfill(res,0,1);    }
inline double &randcfill(double &res) { return randcfill(res,1);      }
inline double &randCfill(double &res) { return randCfill(res,0,1);    }
inline double &randffill(double &res) { return randffill(res,0.5,1);  }
inline double &randtfill(double &res) { return randtfill(res,1);      }

inline void randseed(int seedval)
{
    double dummy = 0;
    randfill(dummy,'S',seedval);
}


#endif


