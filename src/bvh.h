#ifndef BVHH
#define BVHH

#include "hitable.h"
#include "aabb.h"
#include <vector>
#include <float.h>
#include <algorithm>

// POD representation of a BVH node, built on the CPU and copied to the GPU.
struct BVHNodeData {
    aabb box;
    int left;   // child index, -1 for leaf
    int right;  // child index, -1 for leaf
    int start;  // start index in primitive array for leaf
    int count;  // number of primitives for leaf
};

class bvh_flat_world : public hitable {
public:
    __device__ bvh_flat_world(const BVHNodeData* nodes_, int node_count_, hitable **prims_)
        : nodes(nodes_), node_count(node_count_), prims(prims_) {}

    /**
     * BVH hit intersection test, thanks to BVH, instead of O(n) we get O(log n) complexity, which lead to faster ray-tracing
     * @param r The ray to test
     * @param tmin Minimum t value
     * @param tmax Maximum t value
     * @param rec Hit record if hit
     * @return True if the ray hits any object in the BVH, false otherwise
     */
    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
        // Iterative traversal with small fixed stack to avoid recursion on device.
        int stack[64];
        int stack_ptr = 0;
        stack[stack_ptr++] = 0; // root index

        bool hit_anything = false;
        float closest = t_max;

        while (stack_ptr) {
            int node_idx = stack[--stack_ptr]; // get next node index
            if (node_idx < 0 || node_idx >= node_count) continue; // skip invalid nodes
            const BVHNodeData& node = nodes[node_idx];  //get node data
            if (!node.box.hit(r, t_min, closest)) continue; // skip if ray misses box

            if (node.count > 0) { // if leaf
                for (int i = 0; i < node.count; i++) { // test all hitables in leaf and return the closest hit
                    hit_record temp_rec;
                    if (prims[node.start + i]->hit(r, t_min, closest, temp_rec)) {
                        hit_anything = true;
                        closest = temp_rec.t;
                        rec = temp_rec;
                    }
                }
            } else { //if not leaf, push children to stack to traverse in next iterations
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
    hitable **prims;
};

/**
 * Build a BHV usign SAH on the CPU
 * @param prim_indices Indices of primitives to build the BVH for
 * @param start Start index in prim_indices
 * @param end End index in prim_indices
 * @param prim_boxes The list of primitive bounding boxes
 * @param nodes Output list of BVH nodes
 * @param leaf_size Maximum number of primitives per leaf
 * @return Index of the created BVH node
 */
int build_sah_bvh(std::vector<int>& prim_indices,
                  int start,
                  int end,
                  const std::vector<aabb> & prim_boxes,
                  std::vector<BVHNodeData>& nodes,
                  int leaf_size = 4) {
    int n = end - start;

    // Compute bounds
    aabb bounds = prim_boxes[prim_indices[start]];
    for (int i = start + 1; i < end; i++) {
        aabb b = prim_boxes[prim_indices[i]];
        bounds = surrounding_box(bounds, b);
    }

    // Create leaf if small enough.
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

    // Choose split by SAH.
    int best_axis = 0;
    int best_split = -1;
    float best_cost = FLT_MAX;

    // Try splitting along each axis.
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

    // Resort with chosen axis
    std::sort(prim_indices.begin() + start, prim_indices.begin() + end, [&](int a, int b){
        vec3 ca = 0.5f * (prim_boxes[a].min() + prim_boxes[a].max());
        vec3 cb = 0.5f * (prim_boxes[b].min() + prim_boxes[b].max());
        return ca[best_axis] < cb[best_axis];
    });

    int mid = start + best_split + 1;

    // Build the node
    BVHNodeData node{};
    node.box = bounds;
    node.start = -1;
    node.count = 0;
    int node_index = static_cast<int>(nodes.size());
    nodes.push_back(node);

    // Recursively build children
    int left_index = build_sah_bvh(prim_indices, start, mid, prim_boxes, nodes, leaf_size);
    int right_index = build_sah_bvh(prim_indices, mid, end, prim_boxes, nodes, leaf_size);

    nodes[node_index].left = left_index;
    nodes[node_index].right = right_index;
    nodes[node_index].box = bounds;
    return node_index;
}

#endif
