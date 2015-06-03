#include "circleobstacle.h"

CircleObst::CircleObst ( const VectorXd &centroid, const double &radius ):
    Obstacle(centroid), radius(radius) {}
