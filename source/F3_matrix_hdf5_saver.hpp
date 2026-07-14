#ifndef F3_MATRIX_HDF5_SAVER_HPP
#define F3_MATRIX_HDF5_SAVER_HPP

/*
    F3_matrix_hdf5_saver.hpp

    Standalone HDF5 helper for saving/loading Eigen::MatrixXcd objects
    from the F3 GPU/OpenMP pipeline.

    Recommended use:

        #include "F3_matrix_hdf5_saver.hpp"

        F3MatrixHdf5Saver saver(
            "F3_matrix_dump_000_A1m_L20.h5",
            F3Hdf5OpenMode::Truncate,
            4,      // gzip level
            'y'     // debug
        );

        F3EnergyMetadata meta;
        meta.epoint_index = i;
        meta.Ecm_real = Ecm.real();
        meta.Ecm_imag = Ecm.imag();
        meta.En_real  = En.real();
        meta.En_imag  = En.imag();
        meta.L        = L;
        meta.Lbyas    = Lbyas;
        meta.nnP0     = nnP[0];
        meta.nnP1     = nnP[1];
        meta.nnP2     = nnP[2];
        meta.irrep    = I;
        meta.irrep_tag = irrep_tag_for_file;

        saver.save_energy_point(
            meta,
            F2,
            G,
            K2inv,
            F3,
            Vsel,
            true,   // save F2
            true,   // save G
            true,   // save K2inv
            true,   // save F3
            true    // save Vsel
        );

    File layout:

        /global_metadata/created_by
        /global_metadata/format_version
        /epoint_0/Ecm
        /epoint_0/En
        /epoint_0/L
        /epoint_0/Lbyas
        /epoint_0/nnP
        /epoint_0/irrep
        /epoint_0/irrep_tag
        /epoint_0/F2
        /epoint_0/G
        /epoint_0/K2inv
        /epoint_0/F3
        /epoint_0/Vsel
        /epoint_1/...

    Complex matrices are stored as real-valued HDF5 datasets with shape:

        [rows, cols, 2]

    where:

        data(i,j,0) = real part
        data(i,j,1) = imaginary part

    Compile example:

        nvcc -O3 -std=c++17 -DEIGEN_NO_CUDA --expt-relaxed-constexpr \
             -Xcompiler -fopenmp -lineinfo \
             -I/usr/include/eigen3 \
             -I/usr/include/hdf5/serial \
             your_file.cu \
             -L/usr/lib/x86_64-linux-gnu/hdf5/serial \
             -lhdf5_cpp -lhdf5 -lcublas -lcudart -o test_gpu

    If using g++ for CPU-only tests:

        g++ -O3 -std=c++17 -I/usr/include/eigen3 \
            -I/usr/include/hdf5/serial test.cpp \
            -L/usr/lib/x86_64-linux-gnu/hdf5/serial \
            -lhdf5_cpp -lhdf5 -o test
*/

#include <H5Cpp.h>
#include <Eigen/Dense>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using F3Hdf5Complex = std::complex<double>;

enum class F3Hdf5OpenMode
{
    Truncate,
    AppendOrCreate,
    ReadOnly
};

struct F3EnergyMetadata
{
    int epoint_index = -1;

    double Ecm_real = std::numeric_limits<double>::quiet_NaN();
    double Ecm_imag = 0.0;

    double En_real = std::numeric_limits<double>::quiet_NaN();
    double En_imag = 0.0;

    double L = std::numeric_limits<double>::quiet_NaN();
    double Lbyas = std::numeric_limits<double>::quiet_NaN();

    int nnP0 = 0;
    int nnP1 = 0;
    int nnP2 = 0;

    std::string irrep;
    std::string irrep_tag;

    int total_dim = -1;
    int vdim = -1;
};

struct F3SavedEnergyPoint
{
    F3EnergyMetadata meta;

    Eigen::MatrixXcd F2;
    Eigen::MatrixXcd G;
    Eigen::MatrixXcd K2inv;
    Eigen::MatrixXcd F3;
    Eigen::MatrixXcd Vsel;

    bool has_F2 = false;
    bool has_G = false;
    bool has_K2inv = false;
    bool has_F3 = false;
    bool has_Vsel = false;
};

namespace f3hdf5_detail
{
    inline bool hdf5_link_exists(const H5::H5File& file, const std::string& path)
    {
        htri_t exists = H5Lexists(file.getId(), path.c_str(), H5P_DEFAULT);
        return exists > 0;
    }

    inline bool hdf5_link_exists(const H5::Group& group, const std::string& name)
    {
        htri_t exists = H5Lexists(group.getId(), name.c_str(), H5P_DEFAULT);
        return exists > 0;
    }

    inline H5::Group create_or_open_group(H5::H5File& file, const std::string& path)
    {
        if (hdf5_link_exists(file, path))
        {
            return file.openGroup(path);
        }
        return file.createGroup(path);
    }

    inline H5::Group create_or_open_group(H5::Group& parent, const std::string& name)
    {
        if (hdf5_link_exists(parent, name))
        {
            return parent.openGroup(name);
        }
        return parent.createGroup(name);
    }

    inline void delete_link_if_exists(H5::Group& group, const std::string& name)
    {
        if (hdf5_link_exists(group, name))
        {
            H5Ldelete(group.getId(), name.c_str(), H5P_DEFAULT);
        }
    }

    inline void delete_link_if_exists(H5::H5File& file, const std::string& path)
    {
        if (hdf5_link_exists(file, path))
        {
            H5Ldelete(file.getId(), path.c_str(), H5P_DEFAULT);
        }
    }

    inline std::string epoint_group_name(int epoint_index)
    {
        if (epoint_index < 0)
        {
            throw std::runtime_error("epoint_index must be >= 0");
        }
        return "epoint_" + std::to_string(epoint_index);
    }

    inline H5::StrType variable_string_type()
    {
        H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
        return str_type;
    }

    inline void write_string_dataset(H5::Group& group, const std::string& name, const std::string& value)
    {
        delete_link_if_exists(group, name);

        H5::StrType str_type = variable_string_type();
        hsize_t dims[1] = {1};
        H5::DataSpace space(1, dims);

        H5::DataSet ds = group.createDataSet(name, str_type, space);
        const char* cstr = value.c_str();
        ds.write(&cstr, str_type);
    }

    inline std::string read_string_dataset(H5::Group& group, const std::string& name)
    {
        H5::DataSet ds = group.openDataSet(name);
        H5::StrType str_type = variable_string_type();

        char* raw = nullptr;
        ds.read(&raw, str_type);

        std::string out;
        if (raw != nullptr)
        {
            out = raw;
            free(raw);
        }
        return out;
    }

    template <class T>
    inline void write_scalar_dataset(
        H5::Group& group,
        const std::string& name,
        const T& value,
        const H5::PredType& h5_type)
    {
        delete_link_if_exists(group, name);

        hsize_t dims[1] = {1};
        H5::DataSpace space(1, dims);
        H5::DataSet ds = group.createDataSet(name, h5_type, space);
        ds.write(&value, h5_type);
    }

    template <class T>
    inline T read_scalar_dataset(
        H5::Group& group,
        const std::string& name,
        const H5::PredType& h5_type)
    {
        T value{};
        H5::DataSet ds = group.openDataSet(name);
        ds.read(&value, h5_type);
        return value;
    }

    inline void write_int_vector_dataset(
        H5::Group& group,
        const std::string& name,
        const std::vector<int>& values)
    {
        delete_link_if_exists(group, name);

        hsize_t dims[1] = {static_cast<hsize_t>(values.size())};
        H5::DataSpace space(1, dims);
        H5::DataSet ds = group.createDataSet(name, H5::PredType::NATIVE_INT, space);
        ds.write(values.data(), H5::PredType::NATIVE_INT);
    }

    inline std::vector<int> read_int_vector_dataset(H5::Group& group, const std::string& name)
    {
        H5::DataSet ds = group.openDataSet(name);
        H5::DataSpace space = ds.getSpace();
        hsize_t dims[1] = {0};
        space.getSimpleExtentDims(dims, nullptr);

        std::vector<int> values(static_cast<size_t>(dims[0]));
        ds.read(values.data(), H5::PredType::NATIVE_INT);
        return values;
    }

    inline void write_double_pair_dataset(
        H5::Group& group,
        const std::string& name,
        double real_part,
        double imag_part)
    {
        delete_link_if_exists(group, name);

        double data[2] = {real_part, imag_part};
        hsize_t dims[1] = {2};
        H5::DataSpace space(1, dims);
        H5::DataSet ds = group.createDataSet(name, H5::PredType::NATIVE_DOUBLE, space);
        ds.write(data, H5::PredType::NATIVE_DOUBLE);
    }

    inline std::complex<double> read_double_pair_dataset(H5::Group& group, const std::string& name)
    {
        double data[2] = {0.0, 0.0};
        H5::DataSet ds = group.openDataSet(name);
        ds.read(data, H5::PredType::NATIVE_DOUBLE);
        return std::complex<double>(data[0], data[1]);
    }

    inline hsize_t safe_chunk_dim(hsize_t x)
    {
        if (x == 0) return 1;
        return std::min<hsize_t>(x, 256);
    }

    inline void write_complex_matrix_dataset(
        H5::Group& group,
        const std::string& name,
        const Eigen::MatrixXcd& M,
        int gzip_level)
    {
        delete_link_if_exists(group, name);

        if (M.rows() < 0 || M.cols() < 0)
        {
            throw std::runtime_error("Invalid matrix dimensions for " + name);
        }

        const hsize_t rows = static_cast<hsize_t>(M.rows());
        const hsize_t cols = static_cast<hsize_t>(M.cols());

        std::vector<double> data(static_cast<size_t>(rows * cols * 2));

        /*
            Store in row-major logical order [i, j, re/im], independent of
            Eigen's internal column-major storage. This makes Python loading
            intuitive:

                M = raw[:, :, 0] + 1j * raw[:, :, 1]
        */
        for (hsize_t i = 0; i < rows; ++i)
        {
            for (hsize_t j = 0; j < cols; ++j)
            {
                const auto z = M(static_cast<int>(i), static_cast<int>(j));
                const size_t idx = static_cast<size_t>((i * cols + j) * 2);
                data[idx + 0] = z.real();
                data[idx + 1] = z.imag();
            }
        }

        hsize_t dims[3] = {rows, cols, 2};
        H5::DataSpace space(3, dims);

        H5::DSetCreatPropList plist;

        hsize_t chunk_dims[3] = {
            safe_chunk_dim(rows),
            safe_chunk_dim(cols),
            2
        };
        plist.setChunk(3, chunk_dims);

        if (gzip_level > 0)
        {
            int level = std::max(0, std::min(9, gzip_level));
            plist.setDeflate(level);
        }

        H5::DataSet ds = group.createDataSet(
            name,
            H5::PredType::NATIVE_DOUBLE,
            space,
            plist
        );

        if (!data.empty())
        {
            ds.write(data.data(), H5::PredType::NATIVE_DOUBLE);
        }
    }

    inline Eigen::MatrixXcd read_complex_matrix_dataset(H5::Group& group, const std::string& name)
    {
        H5::DataSet ds = group.openDataSet(name);
        H5::DataSpace space = ds.getSpace();

        int rank = space.getSimpleExtentNdims();
        if (rank != 3)
        {
            throw std::runtime_error("Dataset " + name + " must have rank 3: [rows, cols, 2]");
        }

        hsize_t dims[3] = {0, 0, 0};
        space.getSimpleExtentDims(dims, nullptr);

        const hsize_t rows = dims[0];
        const hsize_t cols = dims[1];
        const hsize_t two = dims[2];

        if (two != 2)
        {
            throw std::runtime_error("Dataset " + name + " third dimension must be 2");
        }

        std::vector<double> data(static_cast<size_t>(rows * cols * 2));
        if (!data.empty())
        {
            ds.read(data.data(), H5::PredType::NATIVE_DOUBLE);
        }

        Eigen::MatrixXcd M(static_cast<int>(rows), static_cast<int>(cols));

        for (hsize_t i = 0; i < rows; ++i)
        {
            for (hsize_t j = 0; j < cols; ++j)
            {
                const size_t idx = static_cast<size_t>((i * cols + j) * 2);
                M(static_cast<int>(i), static_cast<int>(j)) =
                    std::complex<double>(data[idx + 0], data[idx + 1]);
            }
        }

        return M;
    }

    inline void write_metadata_to_group(H5::Group& group, const F3EnergyMetadata& meta)
    {
        write_scalar_dataset(group, "epoint_index", meta.epoint_index, H5::PredType::NATIVE_INT);

        write_double_pair_dataset(group, "Ecm", meta.Ecm_real, meta.Ecm_imag);
        write_double_pair_dataset(group, "En",  meta.En_real,  meta.En_imag);

        write_scalar_dataset(group, "L", meta.L, H5::PredType::NATIVE_DOUBLE);
        write_scalar_dataset(group, "Lbyas", meta.Lbyas, H5::PredType::NATIVE_DOUBLE);

        write_int_vector_dataset(group, "nnP", {meta.nnP0, meta.nnP1, meta.nnP2});

        write_string_dataset(group, "irrep", meta.irrep);
        write_string_dataset(group, "irrep_tag", meta.irrep_tag);

        write_scalar_dataset(group, "total_dim", meta.total_dim, H5::PredType::NATIVE_INT);
        write_scalar_dataset(group, "vdim", meta.vdim, H5::PredType::NATIVE_INT);
    }

    inline F3EnergyMetadata read_metadata_from_group(H5::Group& group)
    {
        F3EnergyMetadata meta;

        meta.epoint_index = read_scalar_dataset<int>(group, "epoint_index", H5::PredType::NATIVE_INT);

        auto Ecm = read_double_pair_dataset(group, "Ecm");
        auto En  = read_double_pair_dataset(group, "En");

        meta.Ecm_real = Ecm.real();
        meta.Ecm_imag = Ecm.imag();
        meta.En_real  = En.real();
        meta.En_imag  = En.imag();

        meta.L = read_scalar_dataset<double>(group, "L", H5::PredType::NATIVE_DOUBLE);
        meta.Lbyas = read_scalar_dataset<double>(group, "Lbyas", H5::PredType::NATIVE_DOUBLE);

        std::vector<int> nnP = read_int_vector_dataset(group, "nnP");
        if (nnP.size() >= 3)
        {
            meta.nnP0 = nnP[0];
            meta.nnP1 = nnP[1];
            meta.nnP2 = nnP[2];
        }

        meta.irrep = read_string_dataset(group, "irrep");
        meta.irrep_tag = read_string_dataset(group, "irrep_tag");

        meta.total_dim = read_scalar_dataset<int>(group, "total_dim", H5::PredType::NATIVE_INT);
        meta.vdim = read_scalar_dataset<int>(group, "vdim", H5::PredType::NATIVE_INT);

        return meta;
    }
}

class F3MatrixHdf5Saver
{
private:
    std::string filename_;
    int gzip_level_ = 4;
    char debug_ = 'n';
    H5::H5File file_;

public:
    F3MatrixHdf5Saver(
        const std::string& filename,
        F3Hdf5OpenMode mode = F3Hdf5OpenMode::AppendOrCreate,
        int gzip_level = 4,
        char debug = 'n')
        : filename_(filename),
          gzip_level_(std::max(0, std::min(9, gzip_level))),
          debug_(debug),
          file_(open_file(filename, mode))
    {
        write_global_metadata_if_missing();
    }

    ~F3MatrixHdf5Saver()
    {
        try
        {
            file_.flush(H5F_SCOPE_GLOBAL);
            file_.close();
        }
        catch (...)
        {
            // Destructors must not throw.
        }
    }

    F3MatrixHdf5Saver(const F3MatrixHdf5Saver&) = delete;
    F3MatrixHdf5Saver& operator=(const F3MatrixHdf5Saver&) = delete;

    F3MatrixHdf5Saver(F3MatrixHdf5Saver&&) = delete;
    F3MatrixHdf5Saver& operator=(F3MatrixHdf5Saver&&) = delete;

    static H5::H5File open_file(const std::string& filename, F3Hdf5OpenMode mode)
    {
        if (mode == F3Hdf5OpenMode::Truncate)
        {
            return H5::H5File(filename, H5F_ACC_TRUNC);
        }

        if (mode == F3Hdf5OpenMode::ReadOnly)
        {
            return H5::H5File(filename, H5F_ACC_RDONLY);
        }

        // AppendOrCreate
        try
        {
            return H5::H5File(filename, H5F_ACC_RDWR);
        }
        catch (...)
        {
            return H5::H5File(filename, H5F_ACC_TRUNC);
        }
    }

    void write_global_metadata_if_missing()
    {
        H5::Group meta_group = f3hdf5_detail::create_or_open_group(file_, "/global_metadata");

        if (!f3hdf5_detail::hdf5_link_exists(meta_group, "format_version"))
        {
            f3hdf5_detail::write_string_dataset(meta_group, "format_version", "F3_matrix_hdf5_v1");
        }

        if (!f3hdf5_detail::hdf5_link_exists(meta_group, "created_by"))
        {
            f3hdf5_detail::write_string_dataset(meta_group, "created_by", "F3MatrixHdf5Saver");
        }

        if (!f3hdf5_detail::hdf5_link_exists(meta_group, "complex_layout"))
        {
            f3hdf5_detail::write_string_dataset(meta_group, "complex_layout", "dataset[rows, cols, 2] = real, imag");
        }
    }

    bool has_energy_point(int epoint_index)
    {
        std::string path = "/" + f3hdf5_detail::epoint_group_name(epoint_index);
        return f3hdf5_detail::hdf5_link_exists(file_, path);
    }

    void save_energy_point(
        F3EnergyMetadata meta,
        const Eigen::MatrixXcd& F2,
        const Eigen::MatrixXcd& G,
        const Eigen::MatrixXcd& K2inv,
        const Eigen::MatrixXcd& F3,
        const Eigen::MatrixXcd& Vsel,
        bool save_F2 = true,
        bool save_G = true,
        bool save_K2inv = true,
        bool save_F3 = true,
        bool save_Vsel = true,
        bool overwrite_existing_epoint = true)
    {
        if (meta.epoint_index < 0)
        {
            throw std::runtime_error("save_energy_point: meta.epoint_index must be >= 0");
        }

        if (meta.total_dim < 0)
        {
            if (F2.rows() > 0) meta.total_dim = static_cast<int>(F2.rows());
            else if (F3.rows() > 0) meta.total_dim = static_cast<int>(F3.rows());
        }

        if (meta.vdim < 0)
        {
            meta.vdim = static_cast<int>(Vsel.cols());
        }

        std::string group_name = f3hdf5_detail::epoint_group_name(meta.epoint_index);
        std::string group_path = "/" + group_name;

        if (f3hdf5_detail::hdf5_link_exists(file_, group_path))
        {
            if (!overwrite_existing_epoint)
            {
                throw std::runtime_error("Energy point already exists and overwrite_existing_epoint=false: " + group_path);
            }
            f3hdf5_detail::delete_link_if_exists(file_, group_path);
        }

        H5::Group group = file_.createGroup(group_path);

        f3hdf5_detail::write_metadata_to_group(group, meta);

        f3hdf5_detail::write_scalar_dataset(group, "has_F2",    int(save_F2),    H5::PredType::NATIVE_INT);
        f3hdf5_detail::write_scalar_dataset(group, "has_G",     int(save_G),     H5::PredType::NATIVE_INT);
        f3hdf5_detail::write_scalar_dataset(group, "has_K2inv", int(save_K2inv), H5::PredType::NATIVE_INT);
        f3hdf5_detail::write_scalar_dataset(group, "has_F3",    int(save_F3),    H5::PredType::NATIVE_INT);
        f3hdf5_detail::write_scalar_dataset(group, "has_Vsel",  int(save_Vsel),  H5::PredType::NATIVE_INT);

        if (save_F2)
        {
            f3hdf5_detail::write_complex_matrix_dataset(group, "F2", F2, gzip_level_);
        }

        if (save_G)
        {
            f3hdf5_detail::write_complex_matrix_dataset(group, "G", G, gzip_level_);
        }

        if (save_K2inv)
        {
            f3hdf5_detail::write_complex_matrix_dataset(group, "K2inv", K2inv, gzip_level_);
        }

        if (save_F3)
        {
            f3hdf5_detail::write_complex_matrix_dataset(group, "F3", F3, gzip_level_);
        }

        if (save_Vsel)
        {
            f3hdf5_detail::write_complex_matrix_dataset(group, "Vsel", Vsel, gzip_level_);
        }

        file_.flush(H5F_SCOPE_GLOBAL);

        if (debug_ == 'y')
        {
            std::cout << "[HDF5] saved " << group_name
                      << " Ecm = " << std::setprecision(17) << meta.Ecm_real
                      << " total_dim = " << meta.total_dim
                      << " vdim = " << meta.vdim
                      << " file = " << filename_
                      << '\n';
        }
    }

    void save_energy_point_minimal(
        F3EnergyMetadata meta,
        const Eigen::MatrixXcd& F2,
        const Eigen::MatrixXcd& G,
        const Eigen::MatrixXcd& K2inv,
        const Eigen::MatrixXcd& Vsel,
        bool overwrite_existing_epoint = true)
    {
        static const Eigen::MatrixXcd empty;
        save_energy_point(
            meta,
            F2,
            G,
            K2inv,
            empty,
            Vsel,
            true,
            true,
            true,
            false,
            true,
            overwrite_existing_epoint
        );
    }

    F3SavedEnergyPoint load_energy_point(int epoint_index)
    {
        std::string group_path = "/" + f3hdf5_detail::epoint_group_name(epoint_index);
        if (!f3hdf5_detail::hdf5_link_exists(file_, group_path))
        {
            throw std::runtime_error("Energy point does not exist: " + group_path);
        }

        H5::Group group = file_.openGroup(group_path);

        F3SavedEnergyPoint out;
        out.meta = f3hdf5_detail::read_metadata_from_group(group);

        out.has_F2 = f3hdf5_detail::hdf5_link_exists(group, "F2");
        out.has_G = f3hdf5_detail::hdf5_link_exists(group, "G");
        out.has_K2inv = f3hdf5_detail::hdf5_link_exists(group, "K2inv");
        out.has_F3 = f3hdf5_detail::hdf5_link_exists(group, "F3");
        out.has_Vsel = f3hdf5_detail::hdf5_link_exists(group, "Vsel");

        if (out.has_F2)
        {
            out.F2 = f3hdf5_detail::read_complex_matrix_dataset(group, "F2");
        }
        if (out.has_G)
        {
            out.G = f3hdf5_detail::read_complex_matrix_dataset(group, "G");
        }
        if (out.has_K2inv)
        {
            out.K2inv = f3hdf5_detail::read_complex_matrix_dataset(group, "K2inv");
        }
        if (out.has_F3)
        {
            out.F3 = f3hdf5_detail::read_complex_matrix_dataset(group, "F3");
        }
        if (out.has_Vsel)
        {
            out.Vsel = f3hdf5_detail::read_complex_matrix_dataset(group, "Vsel");
        }

        return out;
    }

    std::string filename() const
    {
        return filename_;
    }
};

inline void save_single_energy_matrices_to_hdf5(
    const std::string& filename,
    int epoint_index,
    double Ecm_real,
    double Ecm_imag,
    double En_real,
    double En_imag,
    double L,
    double Lbyas,
    int nnP0,
    int nnP1,
    int nnP2,
    const std::string& irrep,
    const std::string& irrep_tag,
    const Eigen::MatrixXcd& F2,
    const Eigen::MatrixXcd& G,
    const Eigen::MatrixXcd& K2inv,
    const Eigen::MatrixXcd& F3,
    const Eigen::MatrixXcd& Vsel,
    bool save_F2 = true,
    bool save_G = true,
    bool save_K2inv = true,
    bool save_F3 = true,
    bool save_Vsel = true,
    int gzip_level = 4,
    char debug = 'n')
{
    F3MatrixHdf5Saver saver(filename, F3Hdf5OpenMode::AppendOrCreate, gzip_level, debug);

    F3EnergyMetadata meta;
    meta.epoint_index = epoint_index;
    meta.Ecm_real = Ecm_real;
    meta.Ecm_imag = Ecm_imag;
    meta.En_real = En_real;
    meta.En_imag = En_imag;
    meta.L = L;
    meta.Lbyas = Lbyas;
    meta.nnP0 = nnP0;
    meta.nnP1 = nnP1;
    meta.nnP2 = nnP2;
    meta.irrep = irrep;
    meta.irrep_tag = irrep_tag;
    meta.total_dim = static_cast<int>(F2.rows() > 0 ? F2.rows() : F3.rows());
    meta.vdim = static_cast<int>(Vsel.cols());

    saver.save_energy_point(
        meta,
        F2,
        G,
        K2inv,
        F3,
        Vsel,
        save_F2,
        save_G,
        save_K2inv,
        save_F3,
        save_Vsel,
        true
    );
}

inline std::string make_matrix_dump_filename(
    int nPx,
    int nPy,
    int nPz,
    const std::string& irrep_tag,
    double Lbyas,
    const std::string& prefix = "F3_matrix_dump")
{
    std::ostringstream os;
    os << prefix << "_"
       << nPx << nPy << nPz
       << "_" << irrep_tag
       << "_L" << std::setprecision(12) << Lbyas
       << ".h5";
    return os.str();
}

#endif // F3_MATRIX_HDF5_SAVER_HPP
