#include "ops.h"
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Regular_triangulation_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/Regular_triangulation_cell_base_3.h>
#include <CGAL/Bbox_3.h>
#include <CGAL/Cartesian_converter.h>
#include <CGAL/Regular_triangulation_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <CGAL/Regular_triangulation_face_base_2.h>
#include <chrono>
#include <memory>

// typedef CGAL::Exact_predicates_exact_constructions_kernel        K;      // change to this for exact computation
typedef CGAL::Exact_predicates_inexact_constructions_kernel         K;
typedef CGAL::Exact_predicates_inexact_constructions_kernel         K2;
typedef CGAL::Cartesian_converter<K, K2>                            ExactToFloat;

/*
Dimension 3
*/
typedef CGAL::Regular_triangulation_vertex_base_3<K>                Vb0_3;
typedef CGAL::Triangulation_vertex_base_with_info_3<int, K, Vb0_3>  Vb_3;
typedef CGAL::Regular_triangulation_cell_base_3<K>                  Cb_3;

typedef CGAL::Triangulation_data_structure_3<Vb_3, Cb_3>                        Tds_3;
typedef CGAL::Triangulation_data_structure_3<Vb_3, Cb_3, CGAL::Parallel_tag>    pTds_3;

typedef CGAL::Regular_triangulation_3<K, Tds_3>             Triangulation_3;
typedef CGAL::Regular_triangulation_3<K, pTds_3>            pTriangulation_3;

typedef K::FT                                               Weight;
typedef K::Point_3                                          Point_3;
typedef K::Weighted_point_3                                 Weighted_point_3;

/*
Dimension 2
*/
typedef CGAL::Regular_triangulation_vertex_base_2<K>                Vb0_2;
typedef CGAL::Triangulation_vertex_base_with_info_2<int, K, Vb0_2>  Vb_2;
typedef CGAL::Regular_triangulation_face_base_2<K>                  Fb_2;

typedef CGAL::Triangulation_data_structure_2<Vb_2, Fb_2>            Tds_2;

typedef CGAL::Regular_triangulation_2<K, Tds_2>             Triangulation_2;

typedef K::Point_2                                          Point_2;
typedef K::Weighted_point_2                                 Weighted_point_2;

namespace CGALDDT {

    DTResult WDT3D(int num_points,
                const float* positions,
                const float* weights,
                const bool compute_cc) {

        // @TODO: ugly initialization, faster way?
        std::vector<std::pair<Weighted_point_3, Triangulation_3::Vertex::Info>> weighted_points;
        weighted_points.reserve(num_points);
        float inf = std::numeric_limits<float>::infinity();
        CGAL::Bbox_3 bbox(inf, inf, inf, -inf, -inf, -inf);
            
        int dimension = 3;
        for (int i = 0; i < num_points; i++) {
            Point_3 p(
                positions[i * dimension + 0], 
                positions[i * dimension + 1], 
                positions[i * dimension + 2]);
            Weight w(weights[i]);
            weighted_points.push_back(std::make_pair(Weighted_point_3(p, w), i));
            bbox = bbox + p.bbox();
        }

        // run DT
        DTResult result;
        ExactToFloat exact_to_float;
        
        auto triangulation_start_time = std::chrono::high_resolution_clock::now();

        Triangulation_3 triangulator;
        triangulator.insert(weighted_points.begin(), weighted_points.end());

        auto triangulation_end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> triangulation_time = (triangulation_end_time - triangulation_start_time);
        result.time_sec = triangulation_time.count();
        
        assert(triangulator.is_valid());
        assert(triangulator.dimension() == 3);

        result.num_tri = triangulator.number_of_finite_cells();
        result.tri_verts_idx = new int[result.num_tri * 4];
        if (compute_cc)
            result.tri_cc = new float[result.num_tri * 3];

        // TODO: Faster way?
        int cnt = 0;
        int cnt2 = 0;
        for (auto it = triangulator.finite_cells_begin(); 
                it != triangulator.finite_cells_end(); 
                it++) {
            for (int i = 0; i < 4; i++)
                result.tri_verts_idx[cnt++] = it->vertex(i)->info();

            if (compute_cc) {
                auto cc = triangulator.dual(it);
                result.tri_cc[cnt2 * 3 + 0] = exact_to_float(cc.x());
                result.tri_cc[cnt2 * 3 + 1] = exact_to_float(cc.y());
                result.tri_cc[cnt2 * 3 + 2] = exact_to_float(cc.z());
                cnt2++;
            }
        }
        
        return result;
    }

    DTResult WDT2D(int num_points,
                const float* positions,
                const float* weights,
                const bool compute_cc) {
        
        // @TODO: ugly initialization, faster way?
        std::vector<std::pair<Weighted_point_2, Triangulation_2::Vertex::Info>> weighted_points;
        weighted_points.reserve(num_points);
            
        int dimension = 2;
        for (int i = 0; i < num_points; i++) {
            Point_2 p(
                positions[i * dimension + 0], 
                positions[i * dimension + 1]);
            Weight w(weights[i]);
            weighted_points.push_back(std::make_pair(Weighted_point_2(p, w), i));
        }

        DTResult result;
        ExactToFloat exact_to_float;
        
        auto triangulation_start_time = std::chrono::high_resolution_clock::now();

        Triangulation_2 triangulator;
        triangulator.insert(weighted_points.begin(), weighted_points.end());

        auto triangulation_end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> triangulation_time = (triangulation_end_time - triangulation_start_time);
        result.time_sec = triangulation_time.count();
        
        assert(triangulator.is_valid());
        assert(triangulator.dimension() == dimension);

        result.num_tri = triangulator.number_of_faces();
        result.tri_verts_idx = new int[result.num_tri * 3];
        if (compute_cc)
            result.tri_cc = new float[result.num_tri * 2];

        // TODO: Faster way?
        int cnt = 0;
        int cnt2 = 0;
        for (auto it = triangulator.finite_faces_begin(); 
                it != triangulator.finite_faces_end(); 
                it++) {
            for (int i = 0; i < 3; i++)
                result.tri_verts_idx[cnt++] = it->vertex(i)->info();

            if (compute_cc) {
                auto cc = triangulator.dual(it);
                result.tri_cc[cnt2 * 2 + 0] = exact_to_float(cc.x());
                result.tri_cc[cnt2 * 2 + 1] = exact_to_float(cc.y());
                cnt2++;
            }
        }
        
        return result;
    }

    DTResult WDT(int num_points, int dimension,
                const float* positions,
                const float* weights,
                const bool weighted,
                const bool compute_cc) {
        
        assert(weighted);
        assert(dimension == 3 || dimension == 2);
        assert(!parallelize);       // not parallelizable for now...
        
        if (dimension == 3)
            return WDT3D(num_points, positions, weights, compute_cc);
        else
            return WDT2D(num_points, positions, weights, compute_cc);
    }
}