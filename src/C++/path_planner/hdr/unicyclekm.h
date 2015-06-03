#ifndef UNICYCLEKM_H
#define UNICYCLEKM_H

#include <Eigen/Dense>
#include <cmath>

using namespace Eigen;

class UnicycleKM
{
public:
    static const int z_dim = 2;
    static const int q_dim = 3;
    static const int u_dim = 2;
    static const int z_deriv_needed = 2;

    Matrix< double, z_dim, 1> phi0 ( const Matrix< double, q_dim, 1 > & q );
    Matrix< double, q_dim, 1> phi1 ( const Matrix< double, z_dim, z_deriv_needed+1 > & zl );
    Matrix< double, u_dim, 1> phi2 ( const Matrix< double, z_dim, z_deriv_needed+1 > & zl );
    Matrix< double, u_dim, 1> phi3 ( const Matrix< double, z_dim, z_deriv_needed+2 > & zl );
};

#endif
