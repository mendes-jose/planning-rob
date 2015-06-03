#ifndef CIRCLEOBSTACLE_H
#define CIRCLEOBSTACLE_H

#include "obstacle.h"

class PolygonObst: public Obstacle
{
private:
    double radius;
public:
    PolygonObst ( const VectorXd &centroid, const double &radius );
    inline double pt2obst ( const VectorXd &pt, double offset ) const { return (centroid - pt).norm() - offset; }
    inline double detected_dist ( const VectorXd &pt ) const { return (centroid - pt).norm(); }
    inline const double & get_radius () { return radius; }
};

#endif