#include "unicyclekm.h"
#include <limits>

Matrix< double, UnicycleKM::z_dim, 1 > UnicycleKM::phi0 ( Matrix< double, UnicycleKM::q_dim, 1 > q )
{
    return q.block< UnicycleKM::z_dim, 1 >(0,0);
}

Matrix< double, UnicycleKM::q_dim, 1 > UnicycleKM::phi1 ( Matrix< double, UnicycleKM::z_dim, UnicycleKM::z_deriv_needed+1 > zl )
{
    return ( Matrix< double, UnicycleKM::q_dim, 1 >() <<
            zl.leftCols(1),
            atan2 ( zl(1,1), zl(0,1) )
            ).finished();
}

Matrix< double, UnicycleKM::u_dim, 1 > UnicycleKM::phi2 ( Matrix< double, UnicycleKM::z_dim, UnicycleKM::z_deriv_needed+1 > zl )
{
    double den = pow(zl(0,1), 2) + pow(zl(1,1), 2) + std::numeric_limits< float >::epsilon();
    return ( Matrix< double, UnicycleKM::u_dim, 1>() <<
}
        
Matrix< double, UnicycleKM::u_dim, 1> phi3 ( const Matrix< double, UnicycleKM::z_dim, UnicycleKM::z_deriv_needed+2 > & zl )
{
    
}
