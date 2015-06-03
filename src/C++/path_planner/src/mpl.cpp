#include "mpl.h"

#include <iostream>
#include <unsupported/Eigen/Splines>

MPL::MPL()
{
    nlopt::opt opt(nlopt::LD_MMA, 2);
    std::vector<double> lb(2);
    lb[0] = -HUGE_VAL; lb[1] = 0;
    opt.set_lower_bounds(lb);
}
MPL::~MPL(){}

void MPL::set_init_state ( const VectorXd & q_init )
{
    //std::cout << q_init << std::endl;
    this->q_init = q_init;
    return;
}
void MPL::set_init_state ( const std::vector< double > & q_init )
{
    for ( unsigned i = 0; i < q_init.size(); ++i )
    {
        this->q_init(i) = q_init[i];
    }
    return;
}
void MPL::set_final_state ( const VectorXd & q_final)
{
    //std::cout << q_final << std::endl;
    this->q_init = q_final;
    return;
}
void MPL::set_final_state ( const std::vector< double > & q_final )
{
    return;
}
//void MPL::set_kine_model ( UnicycleKM )
//{
//    return;
//}
void MPL::set_obstacles ( std::vector< Obstacle > & obstacles_list )
{
    return;
}
void MPL::plan ( )
{
    return;
}
