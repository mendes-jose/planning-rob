#include <unsupported/Eigen/Splines>
#include <Eigen/Dense> //3.2.4
#include <iostream>
#include <math.h>
#include <nlopt.hpp>
#include <fstream>

#ifdef _WIN32
#define ON_WINDOWS 1
#else
#define ON_WINDOWS 0
#endif

Eigen::Array< double, 1, Eigen::Dynamic > gen_knots(const double t_init, const double t_final, const int spl_degree, const int no_interv_nn, bool nonuniform)
{
    // TODO no_interv_nn < 2 => error

    double d = (t_final-t_init)/(4+(no_interv_nn-2));
    // d is the nonuniform interval base value (spacing produce intervals like this: 2*d, d,... , d, 2*d)

    Eigen::Array< double, 1, Eigen::Dynamic > knots(spl_degree*2 + no_interv_nn+1);

    // first and last knots
    knots.head(spl_degree) = Eigen::Array< double, 1, Eigen::Dynamic >::Constant(spl_degree, t_init);
    knots.tail(spl_degree) = Eigen::Array< double, 1, Eigen::Dynamic >::Constant(spl_degree, t_final);
    
    // intermediaries knots
    if(nonuniform)
    {    
        knots(spl_degree) = t_init;
        knots(spl_degree+1) = t_init+2*d;

        auto i = 0;
        for(i = 0; i < no_interv_nn-2; ++i)
        {
            knots(spl_degree+i+2) = knots(spl_degree+i+1)+d;
        }

        knots(spl_degree+2+i) = t_final; // = knots(spl_degree+2+i-1) + 2*d
    }
    else // uniform
    {
        knots.segment(spl_degree, no_interv_nn+1) = Eigen::Array< double, 1, Eigen::Dynamic >::LinSpaced(no_interv_nn+1, t_init, t_final);
    }
    return knots;
}


int main(int argc, char** argv)
{
    nlopt::opt opt(nlopt::LD_MMA, 2);
    std::vector<double> lb(2);
    lb[0] = -HUGE_VAL; lb[1] = 0;
    opt.set_lower_bounds(lb);
    const int flatoutput_dim = 2;
    const int flatoutput_deriv_needed = 2;
    const int defoort_degree = flatoutput_deriv_needed+2;
    const int spline_dim = flatoutput_dim+1;
    const int spline_degree = defoort_degree - 1;
    const int no_interv_nn = 4;
    const int no_ctrlpts = no_interv_nn + spline_degree;
    const int Ns = 50;
    const double t_init = 0.0;
    const double t_final = 10.0;

    // Spline typedefs
    typedef Eigen::Spline< double, spline_dim, spline_degree > MySpline;
    typedef MySpline::PointType PointType;
    typedef MySpline::KnotVectorType KnotVectorType;
    typedef MySpline::ControlPointVectorType ControlPointVectorType;

    // Real time and interpolation time
    Eigen::RowVectorXd m_time(Eigen::RowVectorXd::LinSpaced(Ns, t_init, t_final));
    Eigen::RowVectorXd interp_time(Eigen::RowVectorXd::LinSpaced(no_ctrlpts, t_init, t_final));

    // Interpolation points (no_ctrlpts points)
    ControlPointVectorType points = ControlPointVectorType::Random(spline_dim, no_ctrlpts);
    points.topRows(1) = interp_time;
    
    // Get chords lengths from interpolation time
    KnotVectorType chord_lengths;
    Eigen::ChordLengths(interp_time, chord_lengths);

    // Create three splines:
    // one by interpolation
    // and two other by using the control points from the first spline (and the knots for one of them)
    MySpline spline = Eigen::SplineFitting<MySpline>::Interpolate(points, 3, chord_lengths);
    MySpline spline_ctrl(gen_knots(t_init, t_final, spline_degree, no_interv_nn, true), spline.ctrls());
    MySpline spline_ctrl_knots(spline.knots(), spline.ctrls());

    std::cout << "Ctrl points size " << std::endl << spline.ctrls().cols() << std::endl << spline.ctrls().rows() << std::endl;
    std::cout << "Ctrl points " << std::endl << spline.ctrls() << std::endl;
    std::cout << "knots size " << std::endl << spline.knots().cols() << std::endl << spline.knots().rows() << std::endl;
    std::cout << "knots " << std::endl << spline.knots() << std::endl;
    std::cout << "chords " << std::endl << chord_lengths << std::endl;
    std::cout << "points " << std::endl << points << std::endl;

    // Verifying that the three splines are equivalent

    // get chords lengths for the more sampled time vector
    KnotVectorType new_chord_lengths; // knot parameters
    Eigen::ChordLengths(m_time, new_chord_lengths);

    std::ofstream points_ts;
    points_ts.open ("points_ts.csv");
    for (auto i=0; i < Ns; ++i)
    {
        points_ts << m_time(i) << ",";
    }
    points_ts << std::endl;

    int pt1pt2_counter=0;
    int ptpt2_counter=0;
    for (Eigen::DenseIndex i=0; i<m_time.cols(); ++i)
    {
        PointType pt_1 = spline_ctrl( m_time(i) );
        PointType pt_2 = spline_ctrl( m_time(i) );
        PointType pt = spline( new_chord_lengths(i) );
        if( (pt_1 - pt_2).matrix().norm() < 1e-6 )
        {
            pt1pt2_counter++;
        }
        if( (pt - pt_2).matrix().norm() < 1e-6 )
        {
            ptpt2_counter++;
        }
        points_ts << pt(1,i) << ",";
    }
    points_ts << std::endl;

    std::cout << "N pt1pt2 " << std::endl << pt1pt2_counter << std::endl;
    std::cout << "N ptpt2 " << std::endl << ptpt2_counter << std::endl;
    std::cout << "Ns " << std::endl << Ns << std::endl;

    std::ofstream intervdata_ts;
    intervdata_ts.open ("intervdata_ts.csv");
    for (auto i=0; i < no_ctrlpts; ++i)
    {
        intervdata_ts << interp_time(i) << ",";
    }
    intervdata_ts << std::endl;
    for (auto i=0; i < no_ctrlpts; ++i)
    {
        intervdata_ts << points(1,i) << ",";
    }
    intervdata_ts << std::endl;

    if(ON_WINDOWS) system("pause");
    return 0;
}
