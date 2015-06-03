#ifndef MPL_H
#define MPL_H

#include <Eigen/Dense> //3.2.4
#include <nlopt.hpp>
#include "unicyclekinemodel.h"
#include "obstacle.h"

#pragma comment(lib, "libnlopt-0.lib")

using namespace Eigen;

class MPL
{
private:
    VectorXd q_init;
    VectorXd q_final;
public:
    MPL();
    ~MPL();
    void set_init_state ( const VectorXd & );
    void set_init_state ( const std::vector< double > & );
    void set_final_state ( const VectorXd & );
    void set_final_state ( const std::vector< double > & );
    void set_kine_model ( UnicycleKineModel );
    void set_obstacles ( std::vector< Obstacle > & );
    void plan ( );
};

#endif //MPL_H
