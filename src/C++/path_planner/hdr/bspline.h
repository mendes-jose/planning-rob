#ifndef BSPLINE_H
#define BSPLINE_H

#include <unsupported/Eigen/Splines>
#include <Eigen/Dense> //3.2.4

namespace Bspline
{
    Eigen::Array< double, 1, Eigen::Dynamic > gen_knots(const double t_init, const double t_final, const int spl_degree, const int no_interv_nn, bool nonuniform);
}

#endif