#include "obst.h"

class Rectangle: public Polygon {
  public:
    Rectangle(int a,int b) : Polygon(a,b) {}
    ~Rectangle(){}
    int area()
      { return width*height; }
};