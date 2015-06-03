#include <iostream>
#include <fstream>
#include "mpl.h"
#include "unicyclekm.h"
#include "circleobstacle.h"

#include <Eigen/Dense> //3.2.4
#include <unsupported/Eigen/Splines>

#ifdef _WIN32
#define ON_WINDOWS 1
#else
#define ON_WINDOWS 0
#endif

int main(int argc, char** argv)
{
    const int flatoutput_dim = 2;
    const int flatoutput_deriv_needed = 2;
    const int defoort_degree = flatoutput_deriv_needed+2;
    const int spline_dim = flatoutput_dim+1;
    const int spline_degree = defoort_degree - 1;
    const int no_interv_nn = 4;
    const int no_ctrlpts = no_interv_nn + spline_degree;
    const int Ns = 50;
    const double t_init = 0.0;
    const double t_final = 10.0;
    MPL mpl;
    UnicycleKM km;
    //Obstacle obst(VectorXd::Random(2));
    CircleObst cobst(VectorXd::Random(2), 2.0);

    mpl.set_init_state(VectorXd::Random(3));
    mpl.set_final_state(VectorXd::Random(3));
    std::cout << km.phi0(mpl.get_init_state()) << std::endl;
    std::cout << km.q_dim << std::endl;
    std::cout << cobst.get_radius() << std::endl;
    
    //mpl.set_kine_model();
    //mpl.set_obstacles();
    mpl.plan();

    if(ON_WINDOWS) system("pause");
    return 0;
}
