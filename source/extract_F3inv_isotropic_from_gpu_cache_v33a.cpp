// v33a helper: read v32zu/v33d GPU F3inv/Vsel caches and write scalar diagnostics.
// Usage:
//   ./bin/extract_F3inv_isotropic_from_gpu_cache_v33a <cache.bin> <out.dat> <Lbyas> <xi> <irrep>
// Output columns:
//   grid_i Ecm En total_dim proj_dim F3inv_isotropic_rayleigh F3inv_proj_00_real det_scaled_projF3inv_real signed_logabsdet_scaled
//
// The current F3inv_isotropic_rayleigh follows the legacy full-basis isotropic
// convention: unit weight on the plm block and 1/sqrt(2) on the klm block,
// projected into the irrep basis before taking the Rayleigh quotient.
#include <Eigen/Dense>
#include <cstdint>
#include <complex>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

using comp = std::complex<double>;

template<class T>
static T read_scalar(std::ifstream& is) {
    T x{};
    is.read(reinterpret_cast<char*>(&x), sizeof(T));
    if(!is) throw std::runtime_error("truncated cache");
    return x;
}

static double scaled_signed_logdet(const Eigen::MatrixXcd& M, double scale) {
    if(M.rows()==0 || M.rows()!=M.cols()) return std::numeric_limits<double>::quiet_NaN();
    Eigen::MatrixXcd S = M / comp(scale,0.0);
    Eigen::PartialPivLU<Eigen::MatrixXcd> lu(S);
    comp detc = lu.determinant();
    int sign = (detc.real()>0.0) ? 1 : ((detc.real()<0.0) ? -1 : 0);
    const auto& LU = lu.matrixLU();
    double lga=0.0;
    bool ok=true;
    for(int i=0;i<LU.rows();++i) {
        double a=std::abs(LU(i,i));
        if(!(a>0.0) || !std::isfinite(a)) { ok=false; break; }
        lga += std::log(a);
    }
    if(!ok) {
        double a=std::abs(detc);
        if(a>0.0 && std::isfinite(a)) lga=std::log(a);
        else return std::numeric_limits<double>::quiet_NaN();
    }
    return sign ? sign*lga : std::numeric_limits<double>::quiet_NaN();
}

static Eigen::VectorXcd isotropic_projected_vector(int dim1, int dim2, int n, int vdim, const Eigen::MatrixXcd& Vsel) {
    Eigen::VectorXcd vfull = Eigen::VectorXcd::Zero(n);
    const double klm_weight = 1.0 / std::sqrt(2.0);
    for(int i=0; i<dim1 && i<n; ++i) vfull(i) = 1.0;
    for(int i=dim1; i<dim1 + dim2 && i<n; ++i) vfull(i) = klm_weight;
    Eigen::VectorXcd u = Vsel.adjoint() * vfull;
    const double nrm = u.norm();
    if(!(nrm > 0.0) || !std::isfinite(nrm)) {
        u = Eigen::VectorXcd::Ones(vdim);
        const double nrm2 = u.norm();
        if(nrm2 > 0.0) u /= nrm2;
        return u;
    }
    return u / nrm;
}

int main(int argc, char** argv) {
    if(argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <cache.bin> <out.dat> <Lbyas> <xi> <irrep>\n";
        return 2;
    }
    const std::string cache_path = argv[1];
    const std::string out_path = argv[2];
    const double Lbyas = std::stod(argv[3]);
    const double xi = std::stod(argv[4]);
    const std::string irrep = argv[5];
    const double scale = std::pow(Lbyas*xi, 6.0);

    std::ifstream is(cache_path, std::ios::binary);
    if(!is) { std::cerr << "cannot open cache: " << cache_path << "\n"; return 1; }
    std::filesystem::create_directories(std::filesystem::path(out_path).parent_path());
    std::ofstream os(out_path);
    if(!os) { std::cerr << "cannot open output: " << out_path << "\n"; return 1; }
    os << std::setprecision(17);
    os << "# cache " << cache_path << "\n";
    os << "# Lbyas " << Lbyas << " xi " << xi << " irrep " << irrep << " scale_(Lxi)^6 " << scale << "\n";
    os << "# columns: grid_i Ecm En total_dim proj_dim F3inv_isotropic_rayleigh F3inv_proj_00_real det_scaled_projF3inv_real signed_logabsdet_scaled\n";

    std::string header;
    std::getline(is, header);
    if(header != "KKPI_F3INV_VSEL_GPU_V32ZT_EXACT_V32Y 1") {
        std::cerr << "bad header: " << header << "\n";
        return 1;
    }
    const std::uint64_t rec_magic = 0x5653325a4f524543ULL;
    std::uint64_t nrec=0;
    while(is.peek()!=EOF) {
        std::uint64_t magic = read_scalar<std::uint64_t>(is);
        if(magic != rec_magic) { std::cerr << "bad record magic at record " << nrec << "\n"; return 1; }
        std::int32_t grid_i = read_scalar<std::int32_t>(is);
        std::int32_t dim1   = read_scalar<std::int32_t>(is);
        std::int32_t dim2   = read_scalar<std::int32_t>(is);
        std::int32_t n      = read_scalar<std::int32_t>(is);
        std::int32_t vdim   = read_scalar<std::int32_t>(is);
        double Ecm = read_scalar<double>(is);
        double En  = read_scalar<double>(is);
        (void)dim1; (void)dim2;
        if(n<0 || vdim<0) { std::cerr << "negative dimensions\n"; return 1; }
        Eigen::MatrixXcd F3inv(n,n), Vsel(n,vdim);
        for(int col=0; col<n; ++col) for(int row=0; row<n; ++row) {
            double re=read_scalar<double>(is), im=read_scalar<double>(is);
            F3inv(row,col)=comp(re,im);
        }
        for(int col=0; col<vdim; ++col) for(int row=0; row<n; ++row) {
            double re=read_scalar<double>(is), im=read_scalar<double>(is);
            Vsel(row,col)=comp(re,im);
        }
        if(vdim<=0) {
            os << grid_i << ' ' << Ecm << ' ' << En << ' ' << n << ' ' << vdim
               << " nan nan nan nan\n";
            ++nrec;
            continue;
        }
        Eigen::MatrixXcd P = Vsel.adjoint() * F3inv * Vsel;
        Eigen::VectorXcd u = isotropic_projected_vector(dim1, dim2, n, vdim, Vsel);
        double rayleigh = (u.adjoint() * P * u)(0,0).real();
        double p00 = P(0,0).real();
        Eigen::PartialPivLU<Eigen::MatrixXcd> lu(P / comp(scale,0.0));
        double det_scaled_re = lu.determinant().real();
        double slog = scaled_signed_logdet(P, scale);
        os << grid_i << ' ' << Ecm << ' ' << En << ' ' << n << ' ' << vdim << ' '
           << rayleigh << ' ' << p00 << ' ' << det_scaled_re << ' ' << slog << "\n";
        ++nrec;
    }
    std::cout << "[extract-v33a] wrote " << out_path << " rows=" << nrec << "\n";
    return 0;
}
