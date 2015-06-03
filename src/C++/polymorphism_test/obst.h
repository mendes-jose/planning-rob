class Polygon {
  protected:
    int width, height;
  public:
    Polygon (int a, int b) : width(a), height(b) {}
    virtual ~Polygon() =0;
    virtual int area (void){return 0;}
    void printarea()
      { }
};