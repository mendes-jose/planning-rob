#include <unsupported/Eigen/Splines>
#include <Eigen/Dense> //3.2.4
#include <iostream>
#include "b-spline.h"

#ifdef _WIN32
#define ON_WINDOWS 1
#else
#define ON_WINDOWS 0
#endif

using namespace Eigen;

VectorXd gen_knots(const double t_init, const double t_final, const int spl_degree, const int no_interv_nn, bool nonuniform)
{
    // TODO no_interv_nn < 2 => error
    double d = (t_final-t_init)/(4+(no_interv_nn-2));
    VectorXd knots(spl_degree*2 + no_interv_nn+1);

    // first and last knots
    knots.head(spl_degree) = VectorXd::Constant(spl_degree, t_init);
    knots.tail(spl_degree) = VectorXd::Constant(spl_degree, t_final);
    
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
        knots.segment(spl_degree, no_interv_nn+1) = VectorXd::LinSpaced(no_interv_nn+1, t_init, t_final);
    }
    return knots;
}

int main(int argc, char** argv)
{
    const int flatoutput_dim = 2;
    const int flatoutput_deriv = 2;
    const int defoort_degree = flatoutput_deriv+2;
    const int spline_degree = defoort_degree - 1;
    const int no_interv_nn = 5;
    const int Ns = 8; //spl_degree*2 + no_interv_nn+1
    const int t_init = 0.0;
    const int t_final = 1.0;
	
    Array<double, flatoutput_dim, Ns> points;

    for(auto i = 0; i < flatoutput_dim; ++i)
    {
        points.row(i) = RowVectorXd::LinSpaced(Ns, 0.0, 2.0);
    }

    Spline<double, flatoutput_dim, spline_degree> b;
    // attention to degree in the thesis and elsewhere
    // underlying type, curve dim, degree (by default is Dynamic)

    b = SplineFitting<Spline<double, flatoutput_dim, spline_degree>>::Interpolate(
            points, spline_degree); // should add knots?
    //std::cout << SplineFitting<Spline<double, flatoutput_dim, spline_degree>>::Interpolate(
    //    points, spline_degree,  gen_knots(t_init, t_final, spline_degree, no_interv_nn)).ctrls() << std::endl; // should add knots?
    //std::cout << SplineFitting<Spline<double, flatoutput_dim, spline_degree>>::Interpolate(
    //    points, spline_degree).ctrls() << std::endl; // should add knots?

    //std::cout << "Control points: " << std::endl << b.ctrls() << std::endl;
    std::cout << "Knots gen: " << std::endl << gen_knots(t_init, t_final, spline_degree, 0, true).transpose() << std::endl;
    std::cout << "Knots eigen: " << std::endl << b.knots() << std::endl;
    //std::cout << "inter: " << std::endl << VectorXd::LinSpaced(no_interv_nn+1, t_init, t_final) << std::endl;

    //if(ON_WINDOWS) system("pause");
    return 0;
}
