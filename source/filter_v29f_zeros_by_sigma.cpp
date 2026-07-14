#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct Row {
    std::string label;
    int nPx=0, nPy=0, nPz=0;
    std::string irrep, irrep_tag;
    int cluster_index=-1;
    double E_min=std::numeric_limits<double>::quiet_NaN();
    double E_max=std::numeric_limits<double>::quiet_NaN();
    double E_center=std::numeric_limits<double>::quiet_NaN();
    int members=0;
    std::string branch_ids;
    double cluster_width=std::numeric_limits<double>::quiet_NaN();
    double min_singular_cluster=std::numeric_limits<double>::quiet_NaN();
    double min_endpoint_abs_lambda=std::numeric_limits<double>::quiet_NaN();
    double max_overlap=std::numeric_limits<double>::quiet_NaN();
    int old_accepted=0;
    std::string old_reason;
};

static double parse_double(const std::string& s, const std::string& name) {
    try { size_t p=0; double x=std::stod(s,&p); if(p!=s.size()) throw std::runtime_error("bad"); return x; }
    catch(...) { throw std::runtime_error("Could not parse double for " + name + ": " + s); }
}
static int parse_int(const std::string& s, const std::string& name) {
    try { size_t p=0; int x=std::stoi(s,&p); if(p!=s.size()) throw std::runtime_error("bad"); return x; }
    catch(...) { throw std::runtime_error("Could not parse int for " + name + ": " + s); }
}

static void usage(const char* p) {
    std::cerr << "Usage:\n"
              << "  " << p << " --input clusters.dat --sigma-tol X --sorted-output sorted.dat --final-output final.dat\n\n"
              << "Reads v29f/v29e eigenbranch_zero_clusters.dat, sorts clusters by min_singular_cluster,\n"
              << "and writes final zeros with min_singular_cluster < sigma_tol.\n";
}

int main(int argc, char** argv) {
    try {
        std::string input, sorted_output, final_output;
        double sigma_tol = std::numeric_limits<double>::quiet_NaN();
        bool strict_less = true;

        for(int i=1;i<argc;++i) {
            std::string a=argv[i];
            auto need=[&](const std::string& opt)->std::string{
                if(i+1>=argc) throw std::runtime_error("Missing value after " + opt);
                return argv[++i];
            };
            if(a=="--input") input=need(a);
            else if(a=="--sigma-tol") sigma_tol=parse_double(need(a), a);
            else if(a=="--sorted-output") sorted_output=need(a);
            else if(a=="--final-output") final_output=need(a);
            else if(a=="--less-equal") strict_less = (parse_int(need(a),a)==0); // default strict; pass 1? backwards-safe unused
            else if(a=="--help" || a=="-h") { usage(argv[0]); return 0; }
            else throw std::runtime_error("Unknown option: " + a);
        }
        if(input.empty() || sorted_output.empty() || final_output.empty() || !std::isfinite(sigma_tol)) {
            usage(argv[0]);
            throw std::runtime_error("Missing required arguments");
        }

        std::ifstream fin(input);
        if(!fin) throw std::runtime_error("Could not open input: " + input);
        std::vector<Row> rows;
        std::string line;
        while(std::getline(fin,line)) {
            if(line.empty() || line[0]=='#') continue;
            std::istringstream ss(line);
            Row r;
            if(!(ss >> r.label >> r.nPx >> r.nPy >> r.nPz >> r.irrep >> r.irrep_tag
                  >> r.cluster_index >> r.E_min >> r.E_max >> r.E_center >> r.members >> r.branch_ids
                  >> r.cluster_width >> r.min_singular_cluster >> r.min_endpoint_abs_lambda >> r.max_overlap
                  >> r.old_accepted >> r.old_reason)) {
                continue;
            }
            rows.push_back(r);
        }
        std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b){
            const bool af=std::isfinite(a.min_singular_cluster), bf=std::isfinite(b.min_singular_cluster);
            if(af != bf) return af > bf;
            if(a.min_singular_cluster != b.min_singular_cluster) return a.min_singular_cluster < b.min_singular_cluster;
            return a.E_center < b.E_center;
        });

        std::ofstream fs(sorted_output), ff(final_output);
        if(!fs) throw std::runtime_error("Could not open sorted output: " + sorted_output);
        if(!ff) throw std::runtime_error("Could not open final output: " + final_output);
        fs << std::setprecision(17);
        ff << std::setprecision(17);
        fs << "# v29f sigma-sorted zero clusters\n";
        fs << "# input = " << input << "\n";
        fs << "# sigma_tol = " << sigma_tol << "\n";
        fs << "# sorted by min_singular_cluster ascending\n";
        fs << "# columns: sigma_rank label nPx nPy nPz irrep irrep_tag cluster_index E_center E_min E_max members branch_ids min_singular_cluster min_endpoint_abs_lambda cluster_width max_overlap pass_sigma old_accepted old_reason\n";
        ff << "# v29f final zeros selected by singular-value tolerance\n";
        ff << "# input = " << input << "\n";
        ff << "# sigma_tol = " << sigma_tol << "\n";
        ff << "# criterion: min_singular_cluster < sigma_tol\n";
        ff << "# columns: label nPx nPy nPz irrep irrep_tag final_zero_index E_center E_min E_max members branch_ids min_singular_cluster min_endpoint_abs_lambda cluster_width max_overlap reason\n";

        int final_i=0;
        for(size_t i=0;i<rows.size();++i) {
            const auto& r=rows[i];
            const bool pass = std::isfinite(r.min_singular_cluster) && (r.min_singular_cluster < sigma_tol);
            fs << i << ' ' << r.label << ' ' << r.nPx << ' ' << r.nPy << ' ' << r.nPz << ' ' << r.irrep << ' ' << r.irrep_tag << ' '
               << r.cluster_index << ' ' << r.E_center << ' ' << r.E_min << ' ' << r.E_max << ' ' << r.members << ' ' << r.branch_ids << ' '
               << r.min_singular_cluster << ' ' << r.min_endpoint_abs_lambda << ' ' << r.cluster_width << ' ' << r.max_overlap << ' '
               << (pass?1:0) << ' ' << r.old_accepted << ' ' << r.old_reason << '\n';
            if(pass) {
                ff << r.label << ' ' << r.nPx << ' ' << r.nPy << ' ' << r.nPz << ' ' << r.irrep << ' ' << r.irrep_tag << ' '
                   << final_i++ << ' ' << r.E_center << ' ' << r.E_min << ' ' << r.E_max << ' ' << r.members << ' ' << r.branch_ids << ' '
                   << r.min_singular_cluster << ' ' << r.min_endpoint_abs_lambda << ' ' << r.cluster_width << ' ' << r.max_overlap << ' '
                   << "accepted_by_sigma_tol;source_cluster=" << r.cluster_index << '\n';
            }
        }
        std::cout << "[v29f-filter] read_clusters=" << rows.size()
                  << " sigma_tol=" << sigma_tol
                  << " final_zeros=" << final_i << "\n";
        std::cout << "[v29f-filter] sorted_output=" << sorted_output << "\n";
        std::cout << "[v29f-filter] final_output=" << final_output << "\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
