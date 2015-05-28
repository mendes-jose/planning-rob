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

int main(int argc, char** argv)
{
    const int flatoutput_dim = 2;
    const int flatoutput_deriv = 2;
    const int Ns = 10;

    Array<double, flatoutput_dim, Ns> points;

    for(auto i = 0; i<flatoutput_dim; ++i)
    {
        points.row(i) = RowVectorXd::LinSpaced(Ns, 0.0, 1.0);
    }

	Spline<double, flatoutput_dim, flatoutput_deriv+2> b;
    // attention to degree in the thesis and elsewhere
    // underlying type, curve dim, degree (by default is Dynamic)

    b = SplineFitting<Spline<double, flatoutput_dim, flatoutput_deriv+2>>::Interpolate(
            points, flatoutput_deriv+2); // should add knots?

    std::cout << "Control points: " << std::endl << b.ctrls() << std::endl;

    if(ON_WINDOWS) system("pause");
	return 0;
}
