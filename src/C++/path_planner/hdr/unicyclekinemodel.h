#ifndef UNICYCLEKINEMODEL_H
#define UNICYCLEKINEMODEL_H

#include <Eigen/Dense>
#include <cmath>

using namespace Eigen;

#define Z_DIM 2
#define Q_DIM 3
#define U_DIM 2
#define Z_DERIV_NEEDED 2

class UnicycleKineModel
{
public:
    Matrix< double, Z_DIM, 1> phi0 ( Matrix< double, Q_DIM, 1 > q );
    Matrix< double, Q_DIM, 1> phi1 ( Matrix< double, Z_DIM, Z_DERIV_NEEDED+1 > zl );
    Matrix< double, U_DIM, 1> phi2 ();
    Matrix< double, U_DIM, 1> phi3 ();
};

#endif
