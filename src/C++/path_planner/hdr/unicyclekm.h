#ifndef UNICYCLEKM_H
#define UNICYCLEKM_H

#include <Eigen/Dense>

using namespace Eigen;

class UnicycleKM
{
public:
    static const int z_dim = 2;
    static const int q_dim = 3;
    static const int u_dim = 2;
    static const int z_deriv_needed = 2;

    /*!
        \brief
        Return q given z.
      
        Return \f$[x\\ y\\ \\theta]^T\f$ given \f$[z\\ \dot{z}\\ \dotsc\\ z^{(l)}]\f$
        (only \f$z\f$ and \f$\dot{z}\f$ are used). \f$\\theta\f$ is in the range
        \f$(-\pi, \pi]\f$.
     
        \f[
            \begin{array}{l}
            \varphi_1(z(t_k),\dotsc,z^{(l)}(t_k))=\\
            \left[\begin{array}{c}
            x\\
            y\\
            \omega
            \end{array}\right]
            \left[\begin{array}{c}
            z_1\\
            z_2\\
            \arctan(\dot{z}_2/\dot{z}_1)\\
            \end{array}\right]
            \end{array}
        \f]

        \param q
        State vector.
     
        \returns
        The flatoutput.
     */
    Matrix< double, z_dim, 1 > phi0 ( const Matrix< double, q_dim, 1 > &q ) const;
    Matrix< double, q_dim, 1 > phi1 ( const Matrix< double, z_dim, z_deriv_needed+1 > &zl ) const;
    Matrix< double, u_dim, 1 > phi2 ( const Matrix< double, z_dim, z_deriv_needed+1 > &zl ) const;
    Matrix< double, u_dim, 1 > phi3 ( const Matrix< double, z_dim, z_deriv_needed+2 > &zl ) const;
};

#endif
