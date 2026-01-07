#ifndef BVH_SD_H
#define BVH_SD_H

#include "sphere_sd.h"
#include "hitable_sd.h"
#include "hitable.h"
#include "aabb.h"
#include <vector>
#include <float.h>
#include <algorithm>

/**
 * POD (Plain Old Data) representation of a BVH node.
 * Designed for efficient host-to-device transfer and iterative traversal.
 */
struct BVHNodeData {
    aabb box;
    int left;   // Index of the left child, -1 for leaf nodes
    int right;  // Index of the right child, -1 for leaf nodes
    int start;  // Starting index of primitives in the reordered array
    int count;  // Number of primitives contained in this node (0 for internal nodes)
};

/**
 * Flat BVH structure for GPU traversal.
 * Uses a manual stack to perform iterative traversal, bypassing GPU recursion limits.
 */
class bvh_flat_world {
public:
    __device__ bvh_flat_world(const BVHNodeData* nodes_, int node_count_, sphere_opt **prims_)
        : nodes(nodes_), node_count(node_count_), prims(prims_) {}

    /**
     * Core BVH traversal logic.
     * Complexity: O(log n).
     * Employs an iterative approach with a fixed-size stack to ensure thread safety and performance.
     */
    __device__ bool hit(const ray& r, float t_min, float t_max, hit_record_opt& rec) const {
        int stack[64];
        int stack_ptr = 0;
        stack[stack_ptr++] = 0; // Root node index

        bool hit_anything = false;
        float closest = t_max;

        while (stack_ptr) {
            int node_idx = stack[--stack_ptr]; 
            if (node_idx < 0 || node_idx >= node_count) continue; 
            
            const BVHNodeData& node = nodes[node_idx];
            if (!node.box.hit(r, t_min, closest)) continue; 

            if (node.count > 0) { // Leaf node handling
                for (int i = 0; i < node.count; i++) {
                    hit_record_opt temp_rec;
                    if (prims[node.start + i]->hit(r, t_min, closest, temp_rec)) {
                        hit_anything = true;
                        closest = temp_rec.t;
                        rec = temp_rec;
                    }
                }
            } else { // Internal node handling: traverse children
                stack[stack_ptr++] = node.left;
                stack[stack_ptr++] = node.right;
            }
        }
        return hit_anything;
    }

    __device__ virtual aabb bounding_box() const {
        return nodes[0].box;
    }

private:
    const BVHNodeData* nodes;
    int node_count;
    sphere_opt **prims;
};

/**
 * Constructs the BVH hierarchy on the CPU using the Surface Area Heuristic (SAH).
 */
int build_sah_bvh(std::vector<int>& prim_indices,
                  int start,
                  int end,
                  const std::vector<aabb> & prim_boxes,
                  std::vector<BVHNodeData>& nodes,
                  int leaf_size = 4) {
    int n = end - start;

    aabb bounds = prim_boxes[prim_indices[start]];
    for (int i = start + 1; i < end; i++) {
        aabb b = prim_boxes[prim_indices[i]];
        bounds = surrounding_box(bounds, b);
    }

    if (n <= leaf_size) {
        BVHNodeData node{};
        node.box = bounds;
        node.left = -1;
        node.right = -1;
        node.start = start;
        node.count = n;
        nodes.push_back(node);
        return static_cast<int>(nodes.size()) - 1;
    }

    int best_axis = 0;
    int best_split = -1;
    float best_cost = FLT_MAX;

    for (int axis = 0; axis < 3; axis++) {
        std::sort(prim_indices.begin() + start, prim_indices.begin() + end, [&](int a, int b){
            vec3 ca = 0.5f * (prim_boxes[a].min() + prim_boxes[a].max());
            vec3 cb = 0.5f * (prim_boxes[b].min() + prim_boxes[b].max());
            return ca[axis] < cb[axis];
        });

        std::vector<aabb> prefix(n);
        std::vector<aabb> suffix(n);

        prefix[0] = prim_boxes[prim_indices[start]];
        for (int i = 1; i < n; i++) {
            prefix[i] = surrounding_box(prefix[i-1], prim_boxes[prim_indices[start + i]]);
        }

        suffix[n-1] = prim_boxes[prim_indices[start + n - 1]];
        for (int i = n - 2; i >= 0; i--) {
            suffix[i] = surrounding_box(suffix[i+1], prim_boxes[prim_indices[start + i]]);
        }

        for (int i = 0; i < n - 1; i++) {
            float left_area = prefix[i].surface_area();
            float right_area = suffix[i+1].surface_area();
            float cost = left_area * (i + 1) + right_area * (n - i - 1);
            if (cost < best_cost) {
                best_cost = cost;
                best_axis = axis;
                best_split = i;
            }
        }
    }

    std::sort(prim_indices.begin() + start, prim_indices.begin() + end, [&](int a, int b){
        vec3 ca = 0.5f * (prim_boxes[a].min() + prim_boxes[a].max());
        vec3 cb = 0.5f * (prim_boxes[b].min() + prim_boxes[b].max());
        return ca[best_axis] < cb[best_axis];
    });

    int mid = start + best_split + 1;

    BVHNodeData node{};
    node.box = bounds;
    node.start = -1;
    node.count = 0;
    int node_index = static_cast<int>(nodes.size());
    nodes.push_back(node);

    int left_index = build_sah_bvh(prim_indices, start, mid, prim_boxes, nodes, leaf_size);
    int right_index = build_sah_bvh(prim_indices, mid, end, prim_boxes, nodes, leaf_size);

    nodes[node_index].left = left_index;
    nodes[node_index].right = right_index;
    nodes[node_index].box = bounds;
    return node_index;
}

#endif