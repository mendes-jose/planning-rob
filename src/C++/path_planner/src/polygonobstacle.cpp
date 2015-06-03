#include "polygonobstacle.h"

PolygonObst::PolygonObst ( const VectorXd &centroid, const double &radius ):
    Obstacle(centroid), radius(radius) {}
