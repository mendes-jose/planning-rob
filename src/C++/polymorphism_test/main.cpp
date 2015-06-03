// dynamic allocation and polymorphism
#include <iostream>
#include "nobst.h"

using namespace std;

int main () {
  Polygon * ppoly1 = new Rectangle (4,5);
  //Polygon * ppoly2 = new Triangle (4,5);
  ppoly1->printarea();
  //ppoly2->printarea();
  delete ppoly1;
  //delete ppoly2;
  return 0;
}