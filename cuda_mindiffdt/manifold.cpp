#include "manifold.h"
#include <map>
#include <set>
#include <queue>

/*
Processor for removing non-manifold edges
*/
class FaceWithNMEScore {
public:
    int face_id;
    int nm_score;       // non-manifoldness score
                        // +4 for each non-manifold edge
                        // +1 for each boundary edge
                        // (this is to prevent removing faces that are adjacent to manifold edges as much as possible)
    int weight;         // importance of the face in reconstruction, higher means more important

    FaceWithNMEScore() {
        this->face_id = -1;
        this->nm_score = 0;
        this->weight = 0;
    }
    FaceWithNMEScore(int face_id, int nm_score, int weight) {
        this->face_id = face_id;
        this->nm_score = nm_score;
        this->weight = weight;
    }
    FaceWithNMEScore(const FaceWithNMEScore& other) {
        this->face_id = other.face_id;
        this->nm_score = other.nm_score;
        this->weight = other.weight;
    }

    bool operator<(const FaceWithNMEScore& other) const {
        if (nm_score != other.nm_score) {
            // 1. First compare non-manifoldness score. Higher score means higher priority when using (less) set.
            return nm_score > other.nm_score;
        } else if (weight != other.weight) {
            // 2. Second compare weight. Lower weight means higher priority when using (less) set.
            return weight < other.weight;
        } else {
            // 3. Finally compare face id. Lower face id means higher priority when using (less) set.
            return face_id < other.face_id;
        }
    }
};

class NonManifoldEdgeProcessor {
public:
    torch::Tensor verts;
    torch::Tensor faces;
    torch::Tensor face_existence;

    int num_non_manifold_edges;

    typedef std::set<int> Edge;
    std::map<Edge, std::set<int>> edge_face_adjacency;
    std::map<int, std::set<Edge>> face_edge_adjacency;
    std::map<Edge, int> edge_face_count;            // number of adjacent faces per edge
    
    std::set<FaceWithNMEScore> face_score_queue;
    // map from face id to score
    std::map<int, FaceWithNMEScore> face_score_map;  

    // function to call to get face weight
    py::function face_weight_callback;
    // function to call to remove a face
    py::function remove_face_callback;

    NonManifoldEdgeProcessor(
        const torch::Tensor& verts,
        const torch::Tensor& faces,
        const torch::Tensor& face_existence,
        py::object& layered_renderer
    ) {
        this->verts = verts;
        this->faces = faces;
        this->face_existence = face_existence;

        this->num_non_manifold_edges = 0;

        this->face_weight_callback = layered_renderer.attr("get_face_weight");
        this->remove_face_callback = layered_renderer.attr("remove_face");

        this->init_adjacency();
        this->init_face_score();
    }
    ~NonManifoldEdgeProcessor() {}

    // init adjacency info
    void init_adjacency() {
        this->edge_face_adjacency = std::map<Edge, std::set<int>>();
        this->face_edge_adjacency = std::map<int, std::set<Edge>>();
        this->edge_face_count = std::map<Edge, int>();

        torch::Tensor exist_face_idx = torch::nonzero(face_existence).squeeze(1);
        for (int i = 0; i < exist_face_idx.size(0); i++) {
            const int face_id = exist_face_idx[i].item<int>();
            const int v0 = faces[face_id][0].item<int>();
            const int v1 = faces[face_id][1].item<int>();
            const int v2 = faces[face_id][2].item<int>();

            const Edge e0 = {v0, v1};
            const Edge e1 = {v1, v2};
            const Edge e2 = {v2, v0};

            edge_face_adjacency[e0].insert(face_id);
            edge_face_adjacency[e1].insert(face_id);
            edge_face_adjacency[e2].insert(face_id);

            face_edge_adjacency[face_id].insert(e0);
            face_edge_adjacency[face_id].insert(e1);
            face_edge_adjacency[face_id].insert(e2);

            if (edge_face_count.find(e0) == edge_face_count.end())
                edge_face_count[e0] = 0;
            edge_face_count[e0]++;

            if (edge_face_count.find(e1) == edge_face_count.end())
                edge_face_count[e1] = 0;
            edge_face_count[e1]++;

            if (edge_face_count.find(e2) == edge_face_count.end())
                edge_face_count[e2] = 0;
            edge_face_count[e2]++;
        }

        // count non-manifold edges
        for (const auto& it : edge_face_count) {
            if (it.second > 2) {
                this->num_non_manifold_edges++;
            }
        }
    }

    // init face score info
    void init_face_score() {
        this->face_score_queue = std::set<FaceWithNMEScore>();
        this->face_score_map = std::map<int, FaceWithNMEScore>();
        py::object face_weight_dict_obj = this->face_weight_callback();
        std::map<int, int> face_weight_dict = face_weight_dict_obj.cast<std::map<int, int>>();

        torch::Tensor exist_face_idx = torch::nonzero(face_existence).squeeze(1);
        for (int i = 0; i < exist_face_idx.size(0); i++) {
            int face_id = exist_face_idx[i].item<int>();
            int nm_score = compute_face_score(face_id);
            int weight = 0;
            if (face_weight_dict.find(face_id) != face_weight_dict.end()) {
                weight = face_weight_dict[face_id];
            }
            FaceWithNMEScore face_with_score(face_id, nm_score, weight);
            face_score_queue.insert(face_with_score);
            face_score_map[face_id] = face_with_score;
        }
    }

    // compute face score based on [face_edge_adjacency] and [edge_face_count]
    int compute_face_score(int face_id) {
        int score = 0;
        for (const Edge& e : face_edge_adjacency[face_id]) {
            if (edge_face_count[e] > 2) {
                score += 4;
            } else if (edge_face_count[e] == 1) {
                score += 1;
            } else if (edge_face_count[e] == 2) {
                // do nothing
            } else {
                printf("Num. Adj. Faces to an Edge = %d\n", edge_face_count[e]);
                printf("Should not happen, quit\n");
                exit(1);
            }
        }
        return score;
    }

    void remove_face(const int face_id) {
        bool face_exists = face_existence[face_id].item<bool>();
        if (!face_exists) {
            printf("face_id = %d is already removed\n", face_id);
            return;
        }

        // call the remove face callback and update face weight dict
        remove_face_callback(face_id);
        py::object face_weight_dict_obj = this->face_weight_callback();
        std::map<int, int> face_weight_dict = face_weight_dict_obj.cast<std::map<int, int>>();

        // update existence
        face_existence[face_id] = false;

        // update adjacency info
        std::set<Edge> adj_edges = face_edge_adjacency[face_id];
        for (const Edge& e : adj_edges) {
            int prev_edge_face_count = edge_face_count[e];
            int curr_edge_face_count = prev_edge_face_count - 1;
            if (prev_edge_face_count >= 3 && curr_edge_face_count < 3) {
                this->num_non_manifold_edges--;
            }
            edge_face_count[e]--;
            edge_face_adjacency[e].erase(face_id);
        }
        face_edge_adjacency.erase(face_id);

        // update face score info
        // have to consider all adjacent faces to the given face
        for (const Edge& e : adj_edges) {
            for (const int adj_face_id : edge_face_adjacency[e]) {
                // remove old score from the queue
                const auto& old_face_score = face_score_map[adj_face_id];
                face_score_queue.erase(old_face_score);
                
                // insert new score to the queue
                int new_nm_score = compute_face_score(adj_face_id);
                int new_weight = 0;
                if (face_weight_dict.find(adj_face_id) != face_weight_dict.end()) {
                    new_weight = face_weight_dict[adj_face_id];
                }
                auto new_face_score = FaceWithNMEScore(adj_face_id, new_nm_score, new_weight);
                face_score_queue.insert(new_face_score);

                // update face score map
                face_score_map[adj_face_id] = new_face_score;
            }
        }
        // remove the face from the queue
        const auto& target_face_score = face_score_map[face_id];
        face_score_queue.erase(target_face_score);
        face_score_map.erase(face_id);
    }

    void remove_non_manifold_edges() {
        // remove faces with the highest score first
        while (face_score_queue.size() > 0) {
            auto it = face_score_queue.begin();
            int face_id = it->face_id;
            int nm_score = it->nm_score;
            int weight = it->weight;

            if (nm_score < 4) {
                // if there is no more non-manifold edge, stop
                break;
            }

            remove_face(face_id);
            std::cout << "\r\033[K Num. Non Manifold Edges = " << this->num_non_manifold_edges << std::flush;
        }
        std::cout << std::endl;
    }
};

/*
Processor for removing non-manifold vertices
*/
class Fan {
public:
    int vertex;
    int fan_id;             // id in the vertex's fan list
    std::unordered_set<int> faces;
    bool is_complete;       // whether the fan is complete (i.e. no boundary edge)
    int weight;             // sum of face weights in the fan

    Fan() {
        this->vertex = -1;
        this->fan_id = -1;
        this->faces = std::unordered_set<int>();
        this->is_complete = false;
        this->weight = 0;
    }
    Fan(const Fan& other) {
        this->vertex = other.vertex;
        this->fan_id = other.fan_id;
        this->faces = other.faces;
        this->is_complete = other.is_complete;
        this->weight = other.weight;
    }
};

class FaceWithNMVScore {
public:
    int face_id;
    std::array<bool, 3> is_vert_nmv;        // whether the vertex is non-manifold
    bool is_in_complete_fan;                // whether the face is in a complete fan
    int weight;                             // importance of the face in reconstruction, higher means more important
    
    FaceWithNMVScore() {
        this->face_id = -1;
        this->is_vert_nmv = {false, false, false};
        this->is_in_complete_fan = false;
        this->weight = 0;
    }

    FaceWithNMVScore(const FaceWithNMVScore& other) {
        this->face_id = other.face_id;
        this->is_vert_nmv = other.is_vert_nmv;
        this->is_in_complete_fan = other.is_in_complete_fan;
        this->weight = other.weight;
    }

    bool operator<(const FaceWithNMVScore& other) const {
        bool this_is_vert_nmv = is_vert_nmv[0] || is_vert_nmv[1] || is_vert_nmv[2];
        bool other_is_vert_nmv = other.is_vert_nmv[0] || other.is_vert_nmv[1] || other.is_vert_nmv[2];
        // 1. First compare whether the vertex is non-manifold
        if (this_is_vert_nmv != other_is_vert_nmv) {
            return this_is_vert_nmv > other_is_vert_nmv;
        }

        // 2. Second compare whether the fan is complete
        if (is_in_complete_fan != other.is_in_complete_fan) {
            return is_in_complete_fan < other.is_in_complete_fan;
        }

        // 3. Finally compare face weight
        if (weight != other.weight) {
            return weight < other.weight;
        }

        return face_id < other.face_id;
    }
};

class NonManifoldVertexProcessor {
public:
    torch::Tensor verts;
    torch::Tensor faces;
    torch::Tensor face_existence;

    std::map<int, std::vector<Fan>> vertex_fan_map;     // map from vertex id to fans, unlimited number of fans per vertex
    std::map<int, std::array<Fan, 3>> face_fan_map;     // map from face id to fans, 3 fans per face, as there are 3 vertices per face
                                                        // and this face can belong to one fan per vertex
    std::map<int, int> vertex_fan_count;
    int num_non_manifold_verts;
    
    std::set<FaceWithNMVScore> face_score_queue;
    // map from face id to score
    std::map<int, FaceWithNMVScore> face_score_map;  

    // function to call to get face weight
    py::function face_weight_callback;
    // function to call to remove a face
    py::function remove_face_callback; 

    NonManifoldVertexProcessor(
        const torch::Tensor& verts,
        const torch::Tensor& faces,
        const torch::Tensor& face_existence,
        py::object& layered_renderer
    ) {
        this->verts = verts;
        this->faces = faces;
        this->face_existence = face_existence;

        this->num_non_manifold_verts = 0;

        this->face_weight_callback = layered_renderer.attr("get_face_weight");
        this->remove_face_callback = layered_renderer.attr("remove_face");

        this->init_fan();
        this->init_face_score();
    }

    void _construct_adjacency_info_for_vertex(
        int vidx, 
        const std::unordered_set<int>& adj_faces, 
        std::unordered_map<int, std::vector<std::pair<int, int>>>& adj_info,
        std::unordered_set<int>& adj_verts) {
        
        // for each vertex connected to the central vertex, find the other vertex on the face and the face id
        // e.g. for a face [1, 2, 4], where [1] is the central vertex (vidx), the other vertex is 2 and 4
        // and the face id is the face id of the face, which we set 0 in this case
        // so insert (2, 0) and (4, 0) to tmp_fan_map[4] and tmp_fan_map[2], respectively
        for (int tidx : adj_faces) {
            int v0 = faces[tidx][0].item<int>();
            int v1 = faces[tidx][1].item<int>();
            int v2 = faces[tidx][2].item<int>();
            if (v0 != vidx && v1 != vidx) {
                adj_info[v0].emplace_back(std::pair<int, int>(v1, tidx));
                adj_info[v1].emplace_back(std::pair<int, int>(v0, tidx));
                adj_verts.emplace(v0);
                adj_verts.emplace(v1);
            }
            else if (v1 != vidx && v2 != vidx) {
                adj_info[v1].emplace_back(std::pair<int, int>(v2, tidx));
                adj_info[v2].emplace_back(std::pair<int, int>(v1, tidx));
                adj_verts.emplace(v1);
                adj_verts.emplace(v2);
            }
            else if (v2 != vidx && v0 != vidx) {
                adj_info[v2].emplace_back(std::pair<int, int>(v0, tidx));
                adj_info[v0].emplace_back(std::pair<int, int>(v2, tidx));
                adj_verts.emplace(v2);
                adj_verts.emplace(v0);
            }
        }
    }

    void _construct_fans_from_adjacency_info(
        int vidx,
        const std::unordered_map<int, std::vector<std::pair<int, int>>>& adj_info,
        const std::unordered_set<int>& adj_verts,
        const std::map<int, int>& face_weight_dict,
        std::vector<Fan>& fans
    ) {
        fans.clear();

        std::unordered_set<int> visited;
        std::unordered_set<int> not_visited = adj_verts;
        std::unordered_set<int> visited_faces;
        
        while (true) {
            Fan n_fan;
            n_fan.vertex = vidx;
            n_fan.fan_id = fans.size();

            std::queue<int> next;
            int start = *not_visited.begin();
            next.push(start);
            visited.emplace(start);
            not_visited.erase(start);    

            bool is_complete = true;
            while (!next.empty()) {
                int next_vert = next.front();
                next.pop();

                const auto& next_info = adj_info.at(next_vert);

                // if the number of next vertices is not 2, it is a boundary edge -> not complete
                // the number of next vertices is at most 2, since we already removed non-manifold edges
                if (next_info.size() < 2) {
                    is_complete = false;
                }

                for (const auto& next2 : next_info) {
                    int next_vert2 = next2.first;
                    int next_face_id = next2.second;

                    if (visited_faces.count(next_face_id) == 0) {
                        visited.emplace(next_vert2);
                        not_visited.erase(next_vert2);
                        next.emplace(next_vert2);

                        // add face to the fan
                        n_fan.faces.emplace(next_face_id);
                        visited_faces.emplace(next_face_id);
                    }
                }
            }
            n_fan.is_complete = is_complete;

            // set fan weight
            n_fan.weight = 0;
            for (int face_id : n_fan.faces) {
                int curr_face_weight = 0;
                if (face_weight_dict.find(face_id) != face_weight_dict.end()) {
                    curr_face_weight = face_weight_dict.at(face_id);
                }
                n_fan.weight += curr_face_weight;
            }

            fans.emplace_back(n_fan);

            if (not_visited.size() == 0) {
                break;
            }
        }
    }

    void init_fan() {
        // fetch face weights
        py::object face_weight_dict_obj = this->face_weight_callback();
        std::map<int, int> face_weight_dict = face_weight_dict_obj.cast<std::map<int, int>>();

        // Modified code from https://github.com/SonSang/Open3D/blob/ba2a6b189d947deace165a1b9f890650daa2cbeb/cpp/open3d/geometry/TriangleMesh.cpp#L1300
        std::vector<std::unordered_set<int>> vert_to_faces(this->verts.size(0));
        
        auto exist_faces = torch::nonzero(face_existence).squeeze(1);
        for (int i = 0; i < exist_faces.size(0); i++) {
            const int face_id = exist_faces[i].item<int>();
            const int v0 = faces[face_id][0].item<int>();
            const int v1 = faces[face_id][1].item<int>();
            const int v2 = faces[face_id][2].item<int>();

            vert_to_faces[v0].emplace(face_id);
            vert_to_faces[v1].emplace(face_id);
            vert_to_faces[v2].emplace(face_id);
        }

        std::vector<int> non_manifold_verts;
        for (int vidx = 0; vidx < int(this->verts.size(0)); ++vidx) {
            const auto &v_faces = vert_to_faces[vidx];
            if (v_faces.size() == 0) {
                continue;
            }

            // find adjacent vertices and faces
            std::unordered_map<int, std::vector<std::pair<int, int>>> tmp_fan_map;
            std::unordered_set<int> nb_verts;
            _construct_adjacency_info_for_vertex(vidx, v_faces, tmp_fan_map, nb_verts);

            // find fans connected to the central vertex
            std::vector<Fan> fans;
            _construct_fans_from_adjacency_info(vidx, tmp_fan_map, nb_verts, face_weight_dict, fans);

            // add fan to the face fan map
            for (const Fan& n_fan : fans) {
                for (int face_id : n_fan.faces) {
                    int v0 = faces[face_id][0].item<int>();
                    int v1 = faces[face_id][1].item<int>();
                    int v2 = faces[face_id][2].item<int>();

                    if (v0 == vidx) {
                        face_fan_map[face_id][0] = n_fan;
                    } else if (v1 == vidx) {
                        face_fan_map[face_id][1] = n_fan;
                    } else if (v2 == vidx) {
                        face_fan_map[face_id][2] = n_fan;
                    }
                }
            }

            // add fans to the vertex fan map
            vertex_fan_map[vidx] = fans;
            vertex_fan_count[vidx] = fans.size();

            // if the vertex is non-manifold, add it to the list 
            if (fans.size() > 1) {
                non_manifold_verts.push_back(vidx);
            }
        }
        this->num_non_manifold_verts = non_manifold_verts.size();
    }

    void init_face_score() {
        this->face_score_queue = std::set<FaceWithNMVScore>();

        // face weight
        py::object face_weight_dict_obj = this->face_weight_callback();
        std::map<int, int> face_weight_dict = face_weight_dict_obj.cast<std::map<int, int>>();

        auto exist_faces = torch::nonzero(face_existence).squeeze(1);
        for (int i = 0; i < exist_faces.size(0); i++) {
            int face_id = exist_faces[i].item<int>();
            FaceWithNMVScore face_with_score;
            face_with_score.face_id = face_id;

            // check if the vertices are non-manifold
            for (int vi = 0; vi < 3; vi++) {
                int vidx = faces[face_id][vi].item<int>();
                face_with_score.is_vert_nmv[vi] = (vertex_fan_count[vidx] > 1);
            }

            // check if the face is in a complete fan
            face_with_score.is_in_complete_fan = false;
            for (int fi = 0; fi < 3; fi++) {
                const auto& fan = face_fan_map[face_id][fi];
                if (fan.is_complete) {
                    face_with_score.is_in_complete_fan = true;
                    break;
                }
            }

            // set face weight
            face_with_score.weight = 0;
            if (face_weight_dict.find(face_id) != face_weight_dict.end()) {
                face_with_score.weight = face_weight_dict[face_id];
            }

            face_score_queue.insert(face_with_score);
            face_score_map[face_id] = face_with_score;
        }
    }

    void remove_face(const int face_id) {
        bool face_exists = face_existence[face_id].item<bool>();
        if (!face_exists) {
            printf("face_id = %d is already removed\n", face_id);
            return;
        }

        // call the remove face callback and update face weight dict
        int remove_face_weight = face_score_map[face_id].weight;
        remove_face_callback(face_id);
        py::object face_weight_dict_obj = this->face_weight_callback();
        std::map<int, int> face_weight_dict = face_weight_dict_obj.cast<std::map<int, int>>();

        // update existence
        face_existence[face_id] = false;

        /*
        Update fan info
        */
        // remove info from [face_fan_map]
        const auto fans = face_fan_map[face_id];
        face_fan_map.erase(face_id);
        
        // update info from [vertex_fan_map]
        for (int fi = 0; fi < 3; fi++) {
            // 1. remove the fan from the vertex fan map
            int vert_id = fans[fi].vertex;
            int fan_id = fans[fi].fan_id;
            if (vert_id < 0 || vert_id >= verts.size(0)) {
                printf("Vertex id out of range, quit\n");
                exit(1);
            }
            if (fan_id < 0) {
                printf("Fan id out of range: %d, quit\n", fan_id);
                exit(1);
            }
            vertex_fan_map[vert_id].erase(vertex_fan_map[vert_id].begin() + fan_id);

            // update fan id of the fans after the removed fan
            for (int ffi = fan_id; ffi < vertex_fan_map[vert_id].size(); ffi++) {
                vertex_fan_map[vert_id][ffi].fan_id = ffi;

                // update fan id of the fans in the face fan map
                for (int tfi : vertex_fan_map[vert_id][ffi].faces) {
                    int v0 = faces[tfi][0].item<int>();
                    int v1 = faces[tfi][1].item<int>();
                    int v2 = faces[tfi][2].item<int>();

                    if (v0 == vert_id) {
                        face_fan_map[tfi][0].fan_id = ffi;
                    } else if (v1 == vert_id) {
                        face_fan_map[tfi][1].fan_id = ffi;
                    } else if (v2 == vert_id) {
                        face_fan_map[tfi][2].fan_id = ffi;
                    }
                    else {
                        printf("Should not happen, quit\n");
                        exit(1);
                    }
                }
            }

            // 2. add new fan(s) to the vertex fan map
            const auto& fan = fans[fi];
            auto faces_in_fan = fan.faces;
            faces_in_fan.erase(face_id);

            if (faces_in_fan.size() > 0) {
                std::unordered_map<int, std::vector<std::pair<int, int>>> adj_info;
                std::unordered_set<int> adj_verts;
                _construct_adjacency_info_for_vertex(vert_id, faces_in_fan, adj_info, adj_verts);

                std::vector<Fan> new_fans;
                _construct_fans_from_adjacency_info(vert_id, adj_info, adj_verts, face_weight_dict, new_fans);

                // set fan id correctly
                for (int nfi = 0; nfi < new_fans.size(); nfi++) {
                    new_fans[nfi].fan_id = vertex_fan_map[vert_id].size() + nfi;
                }

                for (const Fan& n_fan : new_fans) {
                    // add new fans to the vertex fan map
                    vertex_fan_map[vert_id].emplace_back(n_fan);

                    // add new fans to the face fan map
                    for (int tfi : n_fan.faces) {
                        int v0 = faces[tfi][0].item<int>();
                        int v1 = faces[tfi][1].item<int>();
                        int v2 = faces[tfi][2].item<int>();

                        if (v0 == vert_id) {
                            face_fan_map[tfi][0] = n_fan;
                        } else if (v1 == vert_id) {
                            face_fan_map[tfi][1] = n_fan;
                        } else if (v2 == vert_id) {
                            face_fan_map[tfi][2] = n_fan;
                        } else {
                            printf("Should not happen, quit\n");
                            exit(1);
                        }
                    }
                }
            }

            // update vertex fan count
            int prev_vertex_fan_count = vertex_fan_count[vert_id];
            int curr_vertex_fan_count = vertex_fan_map[vert_id].size();
            if (prev_vertex_fan_count < 2 && curr_vertex_fan_count >= 2) {
                this->num_non_manifold_verts++;
            } else if (prev_vertex_fan_count >= 2 && curr_vertex_fan_count < 2) {
                this->num_non_manifold_verts--;
            }
            vertex_fan_count[vert_id] = curr_vertex_fan_count;
        }
    
        /*
        Update face score info
        */
        // collect faces adjacent to the vertices of the removed face
        std::unordered_set<int> adj_faces;
        for (int vi = 0; vi < 3; vi++) {
            int vidx = faces[face_id][vi].item<int>();
            const auto& fans = vertex_fan_map[vidx];
            for (const Fan& fan : fans) {
                for (int adj_face_id : fan.faces) {
                    adj_faces.insert(adj_face_id);
                }
            }
        }
    
        // remove the face from the queue
        for (const auto& adj_face_id : adj_faces) {
            const auto& old_face_score = face_score_map[adj_face_id];
            face_score_queue.erase(old_face_score);

            // insert new score to the queue
            FaceWithNMVScore new_face_score;
            new_face_score.face_id = adj_face_id;

            // check if the vertices are non-manifold
            for (int vi = 0; vi < 3; vi++) {
                int vidx = faces[adj_face_id][vi].item<int>();
                new_face_score.is_vert_nmv[vi] = (vertex_fan_count[vidx] > 1);
            }

            // check if the face is in a complete fan
            new_face_score.is_in_complete_fan = false;
            for (int fi = 0; fi < 3; fi++) {
                const auto& fan = face_fan_map[adj_face_id][fi];
                if (fan.is_complete) {
                    new_face_score.is_in_complete_fan = true;
                    break;
                }
            }

            // set face weight
            new_face_score.weight = 0;
            if (face_weight_dict.find(adj_face_id) != face_weight_dict.end()) {
                new_face_score.weight = face_weight_dict[adj_face_id];
            }

            face_score_queue.insert(new_face_score);
            face_score_map[adj_face_id] = new_face_score;
        }
    }

    void remove_non_manifold_verts() {
        // remove faces with the highest score first
        while (face_score_queue.size() > 0) {
            auto it = face_score_queue.begin();
            
            int face_id = it->face_id;
            auto is_vert_nmv = it->is_vert_nmv;
            auto weight = it->weight;
            auto is_in_complete_fan = it->is_in_complete_fan;

            face_score_queue.erase(it);
            
            bool face_adj_to_nmv = is_vert_nmv[0] || is_vert_nmv[1] || is_vert_nmv[2];
            if (!face_adj_to_nmv) {
                // if there is no more non-manifold vertex, stop
                break;
            }
            
            remove_face(face_id);
            std::cout << "\r\033[K Num. Non Manifold Verts = " << this->num_non_manifold_verts << std::flush;
        }
        std::cout << std::endl;
    }
};

torch::Tensor MINDIFFDT::remove_non_manifoldness(
    const torch::Tensor& verts,
    const torch::Tensor& faces,
    const torch::Tensor& face_existence,
    py::object& layered_renderer
) {
    printf("1. Remove non-manifold edges\n");
    NonManifoldEdgeProcessor edge_processor(verts, faces, face_existence, layered_renderer);
    edge_processor.remove_non_manifold_edges();
    const auto& n_face_existence_1 = edge_processor.face_existence;

    printf("2. Remove non-manifold vertices\n");
    NonManifoldVertexProcessor vert_processor(verts, faces, n_face_existence_1, layered_renderer);
    vert_processor.remove_non_manifold_verts();
    const auto& n_face_existence_2 = vert_processor.face_existence;

    return n_face_existence_2;
}