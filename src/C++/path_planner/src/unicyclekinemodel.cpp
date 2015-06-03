#include "unicyclekinemodel.h"

Eigen::Matrix< double, Z_DIM, 1 > UnicycleKineModel::phi0 ( Matrix< double, Q_DIM, 1 > q )
{
    return q.block< Z_DIM, 1 >(0,0);
}
Matrix< double, Q_DIM, 1 > UnicycleKineModel::phi1 ( Matrix< double, Z_DIM, Z_DERIV_NEEDED+1 > zl )
{
    return (Matrix< double, Q_DIM, 1 >() <<
            zl.leftCols(1),
            atan2 ( zl(1,1), zl(0,1) )
            ).finished();
}
Matrix< double, U_DIM, 1 > UnicycleKineModel::phi2 (){}
Matrix< double, U_DIM, 1 > UnicycleKineModel::phi3 (){}
