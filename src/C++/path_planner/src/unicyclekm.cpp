#include "unicyclekm.h"
#include <cmath>
#include <limits>
//#include <iostream>

Matrix< double, UnicycleKM::z_dim, 1 > UnicycleKM::phi0 ( const Matrix< double, UnicycleKM::q_dim, 1 > &q ) const
{
    return q.block< UnicycleKM::z_dim, 1 >(0,0);
}

Matrix< double, UnicycleKM::q_dim, 1 > UnicycleKM::phi1 ( const Matrix< double, UnicycleKM::z_dim, UnicycleKM::z_deriv_needed+1 > &zl ) const
{
    return ( Matrix< double, UnicycleKM::q_dim, 1 >() <<
            zl.leftCols(1),
            atan2 ( zl(1,1), zl(0,1) )
            ).finished();
}

Matrix< double, UnicycleKM::u_dim, 1 > UnicycleKM::phi2 ( const Matrix< double, UnicycleKM::z_dim, UnicycleKM::z_deriv_needed+1 > &zl ) const
{
    double den = pow(zl(0,1), 2) + pow(zl(1,1), 2) + std::numeric_limits< float >::epsilon(); //+eps so no /0
    return ( Matrix< double, UnicycleKM::u_dim, 1>() <<
            zl.block< UnicycleKM::z_dim, 1>(0,1).norm(),
            (zl(0,1)*zl(1,2)-zl(1,1)*zl(0,2))/den
            ).finished();
}
        
Matrix< double, UnicycleKM::u_dim, 1 > UnicycleKM::phi3 ( const Matrix< double, UnicycleKM::z_dim, UnicycleKM::z_deriv_needed+2 > &zl ) const
{
    double dz_norm = zl.block< UnicycleKM::z_dim, 1 >(0,1).norm();
    double dz_norm_den = dz_norm + std::numeric_limits< float >::epsilon();
    double dv = (zl(0,1)*zl(0,2)+zl(1,1)*zl(1,2))/dz_norm_den;
    double dw = ((zl(0,2)*zl(1,2)+zl(1,3)*zl(0,1) -
            (zl(1,2)*zl(0,2)+zl(0,3)*zl(1,1)))*(pow(dz_norm, 2)) -
            (zl(0,1)*zl(1,2)-zl(1,1)*zl(0,2))*2*dz_norm*dv)/pow(dz_norm_den, 4);
    return (Matrix< double, UnicycleKM::u_dim, 1 >() <<
            dv,
            dw
            ).finished();
}
