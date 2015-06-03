#ifndef OBSTACLE_H
#define OBSTACLE_H

#include <Eigen/Dense>

using namespace Eigen;

class Obstacle
{
protected:
    VectorXd centroid;
public:
    inline Obstacle ( const VectorXd & centroid ): centroid(centroid) {}
    inline virtual ~Obstacle () =0 {};
    inline virtual double pt2obst ( const VectorXd &pt, double offset ) const { return (centroid - pt).norm() - offset; }
    inline virtual double detected_dist ( const VectorXd &pt ) const { return (centroid - pt).norm(); }
};

#endif