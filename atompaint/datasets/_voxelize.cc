// How to compile
// ==============
// - A relatively fast recompilation command:
//
// 	   $ pip install --no-deps --no-build-isolation --editable .
//
// 	 This only works if `setuptools` and `pybind11` are installed in the
// 	 virtual environment in question.
//
// - After one `pip install` fails, you can copy the compiler command from the 
//   error message.  This doesn't actually update the python installation, 
//   though, so after compilation succeeds, you need to run `pip install`.

#include <overlap.hpp>

#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/eigen.h>

#include <string>
#include <vector>
#include <tuple>
#include <stdexcept>
#include <sstream>
#include <iostream>

#define watch(x) std::cerr << (#x) << std::endl << (x) << std::endl << std::endl

namespace py = pybind11;

using Eigen::DenseBase;
using Eigen::Array;
using Eigen::Dynamic;

// Note that `vector_t` is defined in `overlap.hpp`.  Unlike all the array 
// types defined here, it's a matrix.  I was tempted to make these types 
// matrices as well, for the sake of consistency, but the array API is much 
// more useful for the kinds of calculations I'm doing.

typedef Array<int, 3, Dynamic> indices_t;
typedef Array<scalar_t, 3, Dynamic> vectors_t;
typedef std::tuple<int, int, int> grid_shape_t;

scalar_t static const FUDGE_FACTOR = 1 + 1e-6;

struct Atom {

	Atom(Sphere sphere, int channel, float occupancy):
		sphere(sphere), channel(channel), occupancy(occupancy) {}

	Sphere const sphere;
	int const channel;
	float const occupancy;
};

struct Grid {

	Grid(
        int length_voxels,
        scalar_t resolution_A,
        vector_t const & center_A = {0, 0, 0}
    ):
		length_voxels(length_voxels),
		resolution_A(resolution_A),
		length_A(length_voxels * resolution_A),
		center_A(center_A) {}

	grid_shape_t get_shape() const {
		return std::make_tuple(length_voxels, length_voxels, length_voxels);
	}

	int const length_voxels;
	scalar_t const resolution_A;
	scalar_t const length_A;
	vector_t const center_A;
};


std::ostream& operator << (std::ostream& os, vector_t const & vec) {
	Eigen::IOFormat repr(
			Eigen::StreamPrecision,
			Eigen::DontAlignCols,
			", ",
			", ",
			"",
			"",
			"[",
			"]");

	os << vec.transpose().format(repr);
    return os;
}

std::ostream& operator << (std::ostream& os, Sphere const & sphere) {
    os << "Sphere(center_A=" << sphere.center <<
               ", radius_A=" << sphere.radius << ")";
    return os;
}

std::ostream& operator << (std::ostream& os, Atom const & atom) {
    os << "Atom(sphere=" << atom.sphere <<
             ", channel=" << atom.channel <<
             ", occupancy=" << atom.occupancy << ")";
    return os;
}

std::ostream& operator << (std::ostream& os, Grid const & grid) {
    os << "Grid(length_voxels=" << grid.length_voxels <<
             ", resolution_A=" << grid.resolution_A <<
             ", center_A=" << grid.center_A << ")";
    return os;
}

template < class T >
std::ostream& operator << (std::ostream& os, const std::vector<T>& v) {
    os << "[";
    for (typename std::vector<T>::const_iterator ii = v.begin(); ii != v.end(); ++ii)
    {
        os << " " << *ii;
    }
    os << "]";
    return os;
}


template <typename T>
indices_t
_find_voxels_containing_coords(
		Grid const & grid,
		DenseBase<T> const & coords_A) {

	auto center_to_coords_A = coords_A.colwise() - grid.center_A.array();
	auto origin_to_center_A = grid.resolution_A * (grid.length_voxels - 1) * 0.5;
	auto origin_to_coords_A = origin_to_center_A + center_to_coords_A;

	auto ijk = origin_to_coords_A / grid.resolution_A;
	return ijk.rint().template cast<int>().eval();
}

indices_t
_find_voxels_possibly_contacting_sphere(
		Grid const & grid,
		Sphere const & sphere) {

	scalar_t r = sphere.radius;
	Array<scalar_t, 3, 6> probe_rel_coords_A = {
		{ r, -r,  0,  0,  0,  0},
		{ 0,  0,  r, -r,  0,  0},
		{ 0,  0,  0,  0,  r, -r},
	};
	auto probe_coords_A = probe_rel_coords_A.colwise() + sphere.center.array();
	auto probe_voxels = _find_voxels_containing_coords(grid, probe_coords_A);

	auto ijk_min = probe_voxels.rowwise().minCoeff();
	auto ijk_max = probe_voxels.rowwise().maxCoeff();

	int n = (ijk_max - ijk_min + 1).prod();
	indices_t hits(3, n);

	int m = 0;
	for (int i = ijk_min(0); i <= ijk_max(0); i++) {
		for (int j = ijk_min(1); j <= ijk_max(1); j++) {
			for (int k = ijk_min(2); k <= ijk_max(2); k++) {
					hits.col(m) = Eigen::Vector3i {i, j, k};
					m++;
			}
		}
	}

	return hits;
}

template <typename T>
indices_t
_discard_voxels_outside_image(
		Grid const & grid,
		DenseBase<T> const & voxels) {

	// I think this function needs to return an instantiated array, rather than 
	// an expression, because the mask indices are stored in a local variable 
	// that will go out-of-scope when this function returns.  If an expression 
	// tried to hold onto those indices, it'd segfault when it eventually tried 
	// to read them.

	int const n_voxels = voxels.cols();
	std::vector<int> within_image;
	within_image.reserve(n_voxels);

	for (int i = 0; i < n_voxels; i++) {
		auto voxel = voxels.col(i);
		bool not_too_low = (voxel.minCoeff() >= 0);
		bool not_too_high = (voxel.maxCoeff() < grid.length_voxels);

		if (not_too_low && not_too_high) {
			within_image.push_back(i);
		}
	}

	return voxels(Eigen::all, within_image);
}	

template <typename T>
auto
_get_voxel_center_coords(
		Grid const & grid,
		DenseBase<T> const & voxels) {

	scalar_t center_offset = 0.5 * (grid.length_voxels - 1);
	auto center_coords = (voxels.template cast<scalar_t>() - center_offset) * grid.resolution_A;
	return center_coords.colwise() + grid.center_A.array();
}

template <typename T>
Hexahedron
_get_voxel_cube(
		Grid const & grid,
		DenseBase<T> const & voxel) {

	vector_t center_A = _get_voxel_center_coords(grid, voxel);

	// Coordinates based on CGNS conventions, but really just copied from the 
	// example provided by the `overlap` library:
	// https://github.com/severinstrobl/overlap
	// https://cgns.github.io/CGNS_docs_current/sids/conv.html#unst_hexa
	scalar_t x = grid.resolution_A / 2;
	Array<scalar_t, 3, 8> origin_verts {
		{-x,  x,  x, -x, -x,  x,  x, -x},
		{-x, -x,  x,  x, -x, -x,  x,  x},
		{-x, -x, -x, -x,  x,  x,  x,  x},
	};

	auto verts = origin_verts.colwise() + center_A.array();
	return {
			verts.col(0),
			verts.col(1),
			verts.col(2),
			verts.col(3),
			verts.col(4),
			verts.col(5),
			verts.col(6),
			verts.col(7),
	};
}

void
_add_atom_to_image(
		py::array_t<float> img,
		Grid const & grid,
		Atom const & atom) {

	auto img_accessor = img.mutable_unchecked<4>();

	auto voxels = _find_voxels_possibly_contacting_sphere(grid, atom.sphere);
	auto voxels_within_img = _discard_voxels_outside_image(grid, voxels);

	scalar_t total_overlap_A3 = 0;
	std::vector<scalar_t> overlaps;

	for (auto const & voxel: voxels_within_img.colwise()) {
		Hexahedron cube = _get_voxel_cube(grid, voxel);
		scalar_t overlap_A3 = overlap(atom.sphere, cube);
		total_overlap_A3 += overlap_A3;
		overlaps.push_back(overlap_A3);

		img_accessor(atom.channel, voxel(0), voxel(1), voxel(2)) += 
			atom.occupancy * overlap_A3 / atom.sphere.volume;
	}

	// I included the following check because of a claim made in the source code 
	// of the `voxelize` package, which also uses `overlap` to calculate 
	// sphere/cube intersection volumes.  The claim is that, although `overlap` 
	// puts an emphasis on numerical stability, it's still possible to get 
	// inaccurate results.  I haven't experienced these errors myself, even after 
	// quite long training runs, but it seems prudent to at least check for 
	// impossible results.

	if (
			(total_overlap_A3 > atom.sphere.volume * FUDGE_FACTOR) || (
				(total_overlap_A3 < atom.sphere.volume / FUDGE_FACTOR) &&
				(voxels.cols() == voxels_within_img.cols())
			)
	) {
		// Note that this function is meant to run simultaneously in many different 
		// subprocesses, so it's possible for this message to get garbled with 
		// other messages.  This isn't a real concern, though, because I don't 
		// really expect this message to ever be printed.

		std::cerr << "numerical instability in overlap calculation: "
            << "sum of all overlap volumes (" << total_overlap_A3 << " A^3) "
            << "differs from sphere volume (" << atom.sphere.volume << " A^3)"
            << std::endl;
	}
}


PYBIND11_MODULE(_voxelize, m) {
	// The classes exposed by this binding were originally frozen dataclasses, so 
	// my goal was to implement that same API.  Some notes:
	//
	// - Frozen dataclasses implement `__eq__()` and `__hash__()`.  However, 
	//   neither of these methods worked for the original classes, because they 
	//   all contained numpy array attributes, and numpy arrays (i) don't 
	//   implement equality in the usual way and (ii) aren't hashable.
	//
	//   I thought about implementing these methods anyways.  In this case, the 
	//   arrays are all 3D vectors, so equality is pretty intuitive and hash 
	//   algorithms are easy to find.  But I decided not to, because I really 
	//   don't have any need for it.

	py::class_<Sphere>(m, "Sphere")
		.def(
				py::init<vector_t, scalar_t>(),
				py::arg("center_A"),
				py::arg("radius_A"))
		.def(
				"__repr__",
				[](Sphere const & sphere) { 
                	std::stringstream ss;
                    ss << sphere;
                    return ss.str();
                },
				py::is_operator())
		.def(
				py::pickle(
					[](Sphere const & sphere) {
						return py::make_tuple(sphere.center, sphere.radius);
					},
					[](py::tuple state) {
						if (state.size() != 2) {
							throw std::runtime_error("can't unpickle sphere");
						}
						return Sphere {
								state[0].cast<vector_t>(), 
								state[1].cast<scalar_t>()};
					}))
		.def_readonly("center_A", &Sphere::center)
		.def_readonly("radius_A", &Sphere::radius)
		.def_readonly("volume_A3", &Sphere::volume);

	py::class_<Atom>(m, "Atom")
		.def(
				py::init<Sphere, int, float>(),
				py::arg("sphere"),
				py::arg("channel"),
				py::arg("occupancy"))
		.def(
                "__repr__",
                [](Atom const & atom) {
                	std::stringstream ss;
                    ss << atom;
                    return ss.str();
                },
				py::is_operator())
		.def(
				py::pickle(
					[](Atom const & atom) {
						return py::make_tuple(
								atom.sphere,
								atom.channel,
								atom.occupancy);
					},
					[](py::tuple state) {
						if (state.size() != 3) {
							throw std::runtime_error("can't unpickle grid");
						}
						return Atom {
								state[0].cast<Sphere>(), 
								state[1].cast<int>(), 
								state[2].cast<float>()};
					}))
		.def_readonly("sphere", &Atom::sphere)
		.def_readonly("channel", &Atom::channel)
		.def_readonly("occupancy", &Atom::occupancy);
	
	py::class_<Grid>(m, "Grid")
		.def(
				py::init<int, scalar_t, vector_t>(),
				py::arg("length_voxels"),
				py::arg("resolution_A"),
				py::arg("center_A") = vector_t {0,0,0})
		.def(
                "__repr__",
                [](Grid const & grid) {
                	std::stringstream ss;
                    ss << grid;
                    return ss.str();
                },
				py::is_operator())
		.def(
				py::pickle(
					[](Grid const & grid) {
						return py::make_tuple(
								grid.length_voxels,
								grid.resolution_A,
								grid.center_A);
					},
					[](py::tuple state) {
						if (state.size() != 3) {
							throw std::runtime_error("can't unpickle grid");
						}
						return Grid {
								state[0].cast<int>(), 
								state[1].cast<scalar_t>(), 
								state[2].cast<vector_t>()};
					}))
		.def_readonly("length_voxels", &Grid::length_voxels)
		.def_readonly("length_A", &Grid::length_A)
		.def_readonly("resolution_A", &Grid::resolution_A)
		.def_readonly("center_A", &Grid::center_A)
		.def_property_readonly("shape", &Grid::get_shape);

	m.def(
			"_add_atom_to_image",
			&_add_atom_to_image,
			py::arg("img").noconvert(),
			py::arg("grid"),
			py::arg("atom")); 

	m.def(
			"_find_voxels_possibly_contacting_sphere",
			&_find_voxels_possibly_contacting_sphere,
			py::arg("grid"),
			py::arg("sphere"));

	// The following functions are all templates, so that they can operate on 
	// expressions as well as plain matrices/arrays.  The binding code requires a 
	// non-templated function, so we use lambdas.

	m.def(
			"_find_voxels_containing_coords",
			[](Grid const & grid, vectors_t const & coords_A) {
				return _find_voxels_containing_coords(grid, coords_A);
			},
			py::arg("grid"),
			py::arg("coords_A"));
			
	m.def(
			"_discard_voxels_outside_image",
			[](Grid const & grid, indices_t const & voxels) {
				return _discard_voxels_outside_image(grid, voxels);
			},
			py::arg("grid"),
			py::arg("voxels"));

	m.def(
			"_get_voxel_center_coords",
			[](Grid const & grid, indices_t const & voxels) -> vectors_t {
				return _get_voxel_center_coords(grid, voxels);
			},
			py::arg("grid"),
			py::arg("voxels"));

}
