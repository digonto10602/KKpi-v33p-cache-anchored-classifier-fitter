#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace digonto_v3 {

struct Params {
    double monotone_tol = 0.02;
    double min_drop_fraction = 0.15;
    double min_pole_rise_fraction = 0.15;
    bool require_both_shoulders = false;
};

struct Candidate {
    int left_index = -1;
    int right_index = -1;
    double E_left = std::numeric_limits<double>::quiet_NaN();
    double E_right = std::numeric_limits<double>::quiet_NaN();
    double E_zero_linear = std::numeric_limits<double>::quiet_NaN();
    double y_left = std::numeric_limits<double>::quiet_NaN();
    double y_right = std::numeric_limits<double>::quiet_NaN();
    std::string label = "uncertain"; // true_zero, pole, uncertain
    std::string reason;
    int left_points_used = 0;
    int right_points_used = 0;
    double confidence = 0.0;
};

inline int sign(double x) { return (x > 0.0) ? 1 : ((x < 0.0) ? -1 : 0); }
inline bool finite(double x) { return std::isfinite(x); }

inline double linear_zero(double x1, double y1, double x2, double y2) {
    const double d = y2 - y1;
    if(std::isfinite(d) && std::abs(d) > 0.0) {
        const double z = x1 - y1 * (x2 - x1) / d;
        if(std::isfinite(z) && z >= std::min(x1,x2) && z <= std::max(x1,x2)) return z;
    }
    return 0.5*(x1+x2);
}

struct ShoulderResult {
    std::string kind = "uncertain"; // zero, pole, uncertain
    std::string reason;
    int points_used = 0;
    double score = 0.0;
};

inline bool same_nonzero_sign(double a, double b) {
    const int sa = sign(a), sb = sign(b);
    return sa != 0 && sa == sb;
}

inline ShoulderResult classify_shoulder(std::vector<double> y_far_to_near,
                                        const Params& p) {
    ShoulderResult r;
    if(y_far_to_near.size() < 2) { r.reason = "too_few_points"; return r; }

    // Remove far points until the shoulder segment adjacent to the flip has one polarity.
    while(y_far_to_near.size() > 2 && !same_nonzero_sign(y_far_to_near[y_far_to_near.size()-2], y_far_to_near.back())) {
        y_far_to_near.erase(y_far_to_near.begin());
    }
    while(y_far_to_near.size() > 2 && !same_nonzero_sign(y_far_to_near.front(), y_far_to_near.back())) {
        y_far_to_near.erase(y_far_to_near.begin());
    }
    if(y_far_to_near.size() < 2) { r.reason = "same_sign_trim_removed_window"; return r; }

    std::vector<double> a;
    a.reserve(y_far_to_near.size());
    for(double y: y_far_to_near) a.push_back(std::abs(y));

    // Dynamic 3 -> 2 shoulder trim.  If the middle point is a local extremum,
    // keep only the two points closest to the sign flip.  This implements the
    // user's example: y[i-2] > y[i-1] and y[i] > y[i-1] => use [i-1,i].
    if(a.size() >= 3) {
        const bool middle_min = (a[0] > a[1]*(1.0+p.monotone_tol) && a[2] > a[1]*(1.0+p.monotone_tol));
        const bool middle_max = (a[1] > a[0]*(1.0+p.monotone_tol) && a[1] > a[2]*(1.0+p.monotone_tol));
        if(middle_min || middle_max) {
            y_far_to_near.erase(y_far_to_near.begin());
            a.erase(a.begin());
        }
    }

    r.points_used = static_cast<int>(a.size());
    const double eps = 1e-300;
    bool zero_like = true;
    bool pole_like = true;
    double zero_score = 1e300;
    double pole_score = 1e300;
    for(std::size_t k=0; k+1<a.size(); ++k) {
        const double far = a[k] + eps;
        const double near = a[k+1] + eps;
        const double drop = (far - near) / std::max(far, eps);
        const double rise = (near - far) / std::max(far, eps);
        zero_like = zero_like && (drop >= p.min_drop_fraction);
        pole_like = pole_like && (rise >= p.min_pole_rise_fraction);
        zero_score = std::min(zero_score, drop);
        pole_score = std::min(pole_score, rise);
    }
    if(zero_like) {
        r.kind = "zero";
        r.score = zero_score;
        r.reason = (r.points_used >= 3) ? "monotone_abs_decrease_3pt" : "monotone_abs_decrease_2pt";
        return r;
    }
    if(pole_like) {
        r.kind = "pole";
        r.score = pole_score;
        r.reason = (r.points_used >= 3) ? "monotone_abs_rise_3pt" : "monotone_abs_rise_2pt";
        return r;
    }
    r.reason = "nonmonotone_or_weak_shoulder";
    return r;
}

inline Candidate classify_one_flip(const std::vector<double>& x,
                                   const std::vector<double>& y,
                                   int i,
                                   const Params& p) {
    Candidate c;
    c.left_index = i;
    c.right_index = i+1;
    c.E_left = x.at(static_cast<std::size_t>(i));
    c.E_right = x.at(static_cast<std::size_t>(i+1));
    c.y_left = y.at(static_cast<std::size_t>(i));
    c.y_right = y.at(static_cast<std::size_t>(i+1));
    c.E_zero_linear = linear_zero(c.E_left, c.y_left, c.E_right, c.y_right);

    if(i < 1 || i+2 >= static_cast<int>(y.size())) {
        c.label = "uncertain";
        c.reason = "insufficient_six_point_window";
        return c;
    }

    double maxabs = 0.0;
    for(int j=std::max(0,i-2); j<=std::min(static_cast<int>(y.size())-1,i+3); ++j) {
        if(finite(y[static_cast<std::size_t>(j)])) maxabs = std::max(maxabs, std::abs(y[static_cast<std::size_t>(j)]));
    }
    if(!(maxabs > 0.0)) {
        c.label = "uncertain";
        c.reason = "bad_local_scale";
        return c;
    }

    std::vector<double> left, right;
    // left far -> near: i-2, i-1, i
    for(int j=i-2; j<=i; ++j) if(j>=0 && finite(y[static_cast<std::size_t>(j)])) left.push_back(y[static_cast<std::size_t>(j)] / maxabs);
    // right far -> near: i+3, i+2, i+1
    for(int j=i+3; j>=i+1; --j) if(j<static_cast<int>(y.size()) && finite(y[static_cast<std::size_t>(j)])) right.push_back(y[static_cast<std::size_t>(j)] / maxabs);

    ShoulderResult L = classify_shoulder(left, p);
    ShoulderResult R = classify_shoulder(right, p);
    c.left_points_used = L.points_used;
    c.right_points_used = R.points_used;
    const bool left_zero = L.kind == "zero";
    const bool right_zero = R.kind == "zero";
    const bool left_pole = L.kind == "pole";
    const bool right_pole = R.kind == "pole";

    if((p.require_both_shoulders ? (left_zero && right_zero) : (left_zero || right_zero)) && !(left_pole && right_pole)) {
        c.label = "true_zero";
        c.reason = std::string("v3_") + (left_zero && right_zero ? "both_zero_shoulders" : (left_zero ? "left_zero_shoulder" : "right_zero_shoulder"))
                 + "__L=" + L.reason + "__R=" + R.reason;
        c.confidence = std::min(1.0, 0.55 + 0.2*(left_zero?1:0) + 0.2*(right_zero?1:0)
                                     + 0.05*std::max(0, L.points_used-2) + 0.05*std::max(0, R.points_used-2));
    } else if(left_pole || right_pole) {
        c.label = "pole";
        c.reason = std::string("v3_pole_like__L=") + L.reason + "__R=" + R.reason;
        c.confidence = std::min(1.0, 0.55 + 0.2*(left_pole?1:0) + 0.2*(right_pole?1:0));
    } else {
        c.label = "uncertain";
        c.reason = std::string("v3_uncertain__L=") + L.reason + "__R=" + R.reason;
        c.confidence = 0.2;
    }
    return c;
}

inline std::vector<Candidate> classify_series(const std::vector<double>& x,
                                              const std::vector<double>& y,
                                              const Params& p) {
    std::vector<int> flip_indices;
    for(int i=0; i+1<static_cast<int>(y.size()); ++i) {
        if(!finite(y[static_cast<std::size_t>(i)]) || !finite(y[static_cast<std::size_t>(i+1)])) continue;
        const int s1 = sign(y[static_cast<std::size_t>(i)]);
        const int s2 = sign(y[static_cast<std::size_t>(i+1)]);
        if(s1 != 0 && s2 != 0 && s1*s2 < 0) flip_indices.push_back(i);
    }
    std::vector<Candidate> out(flip_indices.size());
    #pragma omp parallel for schedule(dynamic,64)
    for(int k=0; k<static_cast<int>(flip_indices.size()); ++k) {
        out[static_cast<std::size_t>(k)] = classify_one_flip(x, y, flip_indices[static_cast<std::size_t>(k)], p);
    }
    std::sort(out.begin(), out.end(), [](const Candidate& a, const Candidate& b){ return a.E_zero_linear < b.E_zero_linear; });
    return out;
}

} // namespace digonto_v3
