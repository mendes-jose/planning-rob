#include "bspline.h"

Eigen::Array< double, 1, Eigen::Dynamic > Bspline::gen_knots(const double t_init, const double t_final, const int spl_degree, const int no_interv_nn, bool nonuniform)
{
    // TODO no_interv_nn < 2 => error

    double d = (t_final-t_init)/(4+(no_interv_nn-2));
    // d is the nonuniform interval base value (spacing produce intervals like this: 2*d, d,... , d, 2*d)

    Eigen::Array< double, 1, Eigen::Dynamic > knots(spl_degree*2 + no_interv_nn+1);

    // first and last knots
    knots.head(spl_degree) = Eigen::Array< double, 1, Eigen::Dynamic >::Constant(spl_degree, t_init);
    knots.tail(spl_degree) = Eigen::Array< double, 1, Eigen::Dynamic >::Constant(spl_degree, t_final);
    
    // intermediaries knots
    if(nonuniform)
    {    
        knots(spl_degree) = t_init;
        knots(spl_degree+1) = t_init+2*d;

        auto i = 0;
        for(i = 0; i < no_interv_nn-2; ++i)
        {
            knots(spl_degree+i+2) = knots(spl_degree+i+1)+d;
        }

        knots(spl_degree+2+i) = t_final; // = knots(spl_degree+2+i-1) + 2*d
    }
    else // uniform
    {
        knots.segment(spl_degree, no_interv_nn+1) = Eigen::Array< double, 1, Eigen::Dynamic >::LinSpaced(no_interv_nn+1, t_init, t_final);
    }
    return knots;
}