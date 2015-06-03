#ifndef MPL_H
#define MPL_H

#include <Eigen/Dense> //3.2.4
#include <nlopt.hpp>
#include "unicyclekm.h"
#include "obstacle.h"

#pragma comment(lib, "libnlopt-0.lib")

using namespace Eigen;

class MPL
{
private:
    VectorXd q_init;
    VectorXd q_final;
    KineModel km;
public:
    MPL();
    ~MPL();
    void set_init_state ( const VectorXd & );
    void set_init_state ( const std::vector< double > & );
    void set_final_state ( const VectorXd & );
    void set_final_state ( const std::vector< double > & );
    void set_kine_model ( KineModel );
    void set_obstacles ( std::vector< Obstacle > & );
    inline const VectorXd & get_init_state () const { return q_init; }
    inline const VectorXd & get_final_state () const { return q_final; }
    void plan ( );
};

#endif //MPL_H
