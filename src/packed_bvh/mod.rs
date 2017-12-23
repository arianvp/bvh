//! Module defines [`BVH`] and [`BVHNode`] and functions for building and traversing it.
//! It uses a packed representation that allowes two nodes to fit in a cache line,
//! and allocates nodes from a node pool that has been aligned to cache boundaries.
use EPSILON;
use std::mem;
use std::heap;
use std::heap::Alloc;
use std::ops::Deref;
use std::ops::DerefMut;
use aabb::{AABB, Bounded};
use ray::Ray;
use ray::Intersection;
use bounding_hierarchy::{BoundingHierarchy, BHShape};
use utils::Bucket;
use std::ptr::Unique;

/// BVH Node. either internal or leaf
#[derive(Copy, Clone, Debug)]
pub struct BVHNode {
    /// bounding box of
    bounds: AABB,

    /// the left node if internal, right = left + 1.
    /// otherwise the start index in the shape array
    index: u32,

    /// `0` if internal. Otherwise the number of shapes in the leaf node
    len: u32,
}

fn grow_convex_hull(convex_hull: (AABB, AABB), shape_aabb: &AABB) -> (AABB, AABB) {
    let center = &shape_aabb.center();
    let convex_hull_aabbs = &convex_hull.0;
    let convex_hull_centroids = &convex_hull.1;
    (
        convex_hull_aabbs.join(shape_aabb),
        convex_hull_centroids.grow(center),

    )
}

impl BVHNode {
    fn intersect<'a, Shape: BHShape>(nodes: &[BVHNode], ray: &Ray, indices: &Vec<usize>, shapes: &'a[Shape]) -> Option<(&'a Shape, Intersection)> {
        let mut stack = Vec::with_capacity(128);
        stack.push(0);
        let mut answer: Option<(&Shape, Intersection)> = None;
        while let Some(index) = stack.pop() {
            let node = nodes[index];
            if ray.intersects_aabb(&node.bounds).is_none() {
                continue
            }
            if node.len == 0 {
                let left = node.index as usize;
                let l = ray.intersects_aabb(&nodes[left].bounds);
                let r = ray.intersects_aabb(&nodes[left + 1].bounds);
                match (l, r) {
                    (Some(t1), Some(t2)) => {
                        if t1 <= t2 {
                            stack.push(left);
                            stack.push(left+1);
                        } else {
                            stack.push(left+1);
                            stack.push(left);
                        }
                    }
                    (Some(_), None) => {
                        stack.push(left);
                    }
                    (None, Some(_)) => {
                        stack.push(left + 1);
                    }
                    (None, None) => {}
                }
            } else {
                for i in node.index..node.index+node.len {
                    let idx = indices[i as usize];
                    let shape = &shapes[idx];
                    let intersection = shape.intersect(ray);
                    match answer {
                        Some(ref mut answer) => {
                                if intersection.distance < answer.1.distance {
                                *answer = (shape, intersection)
                            }
                        },
                        None => { answer = Some((shape, intersection)) },
                    }
                }
            }
        }
        answer
    }
    fn intersect_recursive<'a, Shape: BHShape>(nodes: &[BVHNode], node_index: usize, ray: &Ray, indices: &Vec<usize>, shapes: &'a[Shape]) -> Option<(&'a Shape, Intersection)> {
        let node = nodes[node_index];
        if node.len == 0 {
            let left = node.index as usize;
            let l = ray.intersects_aabb(&nodes[left].bounds);
            let r = ray.intersects_aabb(&nodes[left + 1].bounds);
            match (l, r) {
                (Some(t1), Some(t2)) => {
                    if t1 <= t2 {
                        match (BVHNode::intersect_recursive(nodes, left, ray, indices, shapes), BVHNode::intersect_recursive(nodes, left + 1, ray, indices, shapes)) {
                            (Some(x),Some(y)) => Some(if x.1.distance < y.1.distance { x } else { y }),
                            (Some(x), _) => Some(x),
                            (_, Some(y)) => Some(y),
                            _ => None,

                        }
                    } else {
                        match (BVHNode::intersect_recursive(nodes, left + 1, ray, indices, shapes), BVHNode::intersect_recursive(nodes, left, ray, indices, shapes)) {
                            (Some(x),Some(y)) => Some(if x.1.distance < y.1.distance { x } else { y }),
                            (Some(x), _) => Some(x),
                            (_, Some(y)) => Some(y),
                            _ => None,
                        }
                    }
                }
                (Some(_), None) => {
                    BVHNode::intersect_recursive(nodes, left, ray, indices, shapes)
                }
                (None, Some(_)) => {
                    BVHNode::intersect_recursive(nodes, left + 1, ray, indices, shapes)
                }
                (None, None) => None
            }
        } else {
            let mut answer: Option<(&Shape, Intersection)> = None;
            for i in node.index..node.index+node.len {
                let idx = indices[i as usize];
                let shape = &shapes[idx];
                let intersection = shape.intersect(ray);
                match answer {
                    Some(ref mut answer) => {
                            if intersection.distance < answer.1.distance {
                            *answer = (shape, intersection)
                        }
                    },
                    None => { answer = Some((shape, intersection)) },
                }
            }
            answer
        }
    }

    fn traverse_recursive(
        nodes: &[BVHNode],
        node_index: usize,
        ray: &Ray,
        indices: &Vec<usize>,
        rindices: &mut Vec<usize>,
    ) {
        let node = nodes[node_index];
        if node.len == 0 {
            let left = node.index as usize;
            let l = ray.intersects_aabb(&nodes[left].bounds);
            let r = ray.intersects_aabb(&nodes[left + 1].bounds);
            match (l, r) {
                (Some(t1), Some(t2)) => {
                    if t1 <= t2 {
                        BVHNode::traverse_recursive(nodes, left, ray, indices, rindices);
                        BVHNode::traverse_recursive(nodes, left + 1, ray, indices, rindices);
                    } else {
                        BVHNode::traverse_recursive(nodes, left + 1, ray, indices, rindices);
                        BVHNode::traverse_recursive(nodes, left, ray, indices, rindices);
                    }
                }
                (Some(_), None) => {
                    BVHNode::traverse_recursive(nodes, left, ray, indices, rindices);
                }
                (None, Some(_)) => {
                    BVHNode::traverse_recursive(nodes, left + 1, ray, indices, rindices);
                }
                (None, None) => {}
            }
        } else {
            // TODO we should not allocate here! It's slow
            rindices.extend((node.index..node.index+node.len).map(|i| indices[i as usize]).collect::<Vec<usize>>())
        }
    }

    /// Splits a node at `node_index` into to child nodes, by splitting
    /// on a splitplane given by the surface area heuristic evaluated by bins
    fn partition<T: BHShape>(
        shapes: &mut [T],
        indices: &mut [usize],
        nodes: &mut BVHNodePool,
        node_index: usize,
    ) -> bool {
        let node = nodes[node_index];

        let mut convex_hull = Default::default();
        for i in node.index .. node.index+node.len {
            let index = indices[i as usize];
            convex_hull = grow_convex_hull(convex_hull, &shapes[index].aabb());
        }

        // TODO figure out why this was even used in the previous version?
        let (_aabb_bounds, centroid_bounds) = convex_hull;


        let split_axis = centroid_bounds.largest_axis();
        let split_axis_size = centroid_bounds.max[split_axis] - centroid_bounds.min[split_axis];

        // Nothing to split, we return false, causing this node not to be split further
        if split_axis_size < EPSILON {
            return false;
        }

        const NUM_BUCKETS: usize = 6;
        let mut buckets = [Bucket::empty(); NUM_BUCKETS];

        for i in node.index .. node.index + node.len {
            let shape = &shapes[indices[i as usize]];
            let shape_aabb = shape.aabb();
            let shape_center = shape_aabb.center();


            // Get the relative position of the shape centroid `[0.0..1.0]`.
            let bucket_num_relative =
                (shape_center[split_axis] - centroid_bounds.min[split_axis]) / split_axis_size;

            // Convert that to the actual `Bucket` number.
            let bucket_num = (bucket_num_relative * (NUM_BUCKETS as f32 - 0.01)) as usize;

            // Extend the selected `Bucket` 
            buckets[bucket_num].add_aabb(&shape_aabb)


        }

        let mut min_cost = node.bounds.surface_area() * node.len as f32;
        let mut min_bucket = None;
        let mut l_aabb = AABB::empty();
        let mut r_aabb = AABB::empty();

        for i in 0..(NUM_BUCKETS - 1) {
            let (l_buckets, r_buckets) = buckets.split_at(i + 1);
            let child_l = l_buckets.iter().fold(Bucket::empty(), Bucket::join_bucket);
            let child_r = r_buckets.iter().fold(Bucket::empty(), Bucket::join_bucket);

            let new_cost = (child_l.size as f32 * child_l.aabb.surface_area())  + (child_r.size as f32 * child_r.aabb.surface_area());

            if new_cost < min_cost {
                min_bucket = Some(i);
                min_cost = new_cost;
                l_aabb = child_l.aabb;
                r_aabb = child_r.aabb;
            }
        }

        match min_bucket {
            Some(min_bucket) => {
                let split_value = centroid_bounds.min[split_axis] + (1 + min_bucket) as f32 * (split_axis_size / NUM_BUCKETS as f32);
                let mut pivot_index = node.index as usize;
                for i in node.index.. node.index + node.len {
                    let center = shapes[indices[i as usize]].aabb().center()[split_axis];
                    if center <= split_value {
                        indices.swap(pivot_index, i as usize);
                        pivot_index += 1;
                    }
                }

                let left_node = BVHNode {
                    bounds: l_aabb,
                    index: node.index as u32,
                    len: (pivot_index - node.index as usize) as u32,
                };

                let left_index = nodes.push(left_node) as u32;
                
                let right_node = BVHNode {
                    bounds: r_aabb,
                    index: pivot_index as u32,
                    len: (node.len as usize - (pivot_index - node.index as usize)) as u32,
                };
                nodes.push(right_node);

                nodes[node_index].index = left_index;
                nodes[node_index].len = 0;
                true
                    

            },
            None => false
        }

    }
    /// Recursively subdivides nodes until a tree is formed
    fn subdivide<T: BHShape>(
        shapes: &mut [T],
        indices: &mut [usize],
        nodes: &mut BVHNodePool,
        node_index: usize,
    ) {

        // if the node we visited contains less than 3 nodes,
        // there is no need to split
        if nodes[node_index].len < 5 {
            shapes[nodes[node_index].index as usize].set_bh_node_index(node_index);
            return;
        }


        if BVHNode::partition(shapes, indices, nodes, node_index) {
            let left = nodes[node_index].index as usize;
            BVHNode::subdivide(shapes, indices, nodes, left);
            BVHNode::subdivide(shapes, indices, nodes, left + 1);
        }

    }
}


/// BVHNodePool is a collection of BVH nodes that are allocated
/// aligned on a cache line. Two nodes fit in one cache line
pub struct BVHNodePool {
    idx: isize,
    len: usize,
    ptr: Unique<BVHNode>,
}


impl Deref for BVHNodePool {
    type Target = [BVHNode];
    fn deref(&self) -> &[BVHNode] {
        unsafe { ::std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl DerefMut for BVHNodePool {
    fn deref_mut(&mut self) -> &mut [BVHNode] {
        unsafe { ::std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for BVHNodePool {
    fn drop(&mut self) {
        let size = mem::size_of::<BVHNode>();
        let align = 2 * size;
        let layout = heap::Layout::from_size_align(self.len * size, align).unwrap();
        let mut allocator = heap::Heap::default();
        unsafe { allocator.dealloc(self.ptr.as_ptr() as *mut _, layout) }
    }
}

impl BVHNodePool {
    /// Allocates a BVH with a pool size of [`pool_size`]
    pub fn new(len: usize) -> BVHNodePool {
        let size = mem::size_of::<BVHNode>();
        let align = 2 * size;
        let layout = heap::Layout::from_size_align(len * size, align).unwrap();
        let mut allocator = heap::Heap::default();
        let ptr = unsafe { allocator.alloc_zeroed(layout) };
        match ptr {
            Err(e) => allocator.oom(e),
            Ok(ptr) => BVHNodePool {
                idx: -1,
                len,
                ptr: unsafe { Unique::new_unchecked(ptr as *mut _) },
            },
        }
    }

    /// is the pool empty?
    pub fn empty(&self) -> bool {
        self.idx == -1
    }

    #[inline(always)]

    /// moves `node` into the pool. returns the pool index where it was stored
    pub fn push(&mut self, node: BVHNode) -> usize {
        // skip one allocation so the root node's childre are aligned
        if self.idx == 0 {
            self.idx += 1;
        }
        self.idx += 1;
        self[self.idx as usize] = node;
        self.idx as usize

    }
}

/// The [`BVH`] data structure. contains the list of [`BVHNode`]s.
pub struct BVH {
    nodes: BVHNodePool,
    indices: Vec<usize>,
}

impl BVH {
    /// Creates a new [`BVH`] from the `shapes slice.
    ///
    /// [`BVH`]: struct.BVH.html
    pub fn build<Shape: BHShape>(shapes: &mut [Shape]) -> BVH {
        let expected_node_count = shapes.len() * 2;
        let mut nodes = BVHNodePool::new(expected_node_count);
        let mut indices = (0..shapes.len()).collect::<Vec<_>>();

        let mut convex_hull = Default::default();
        for index in &indices {
            convex_hull = grow_convex_hull(convex_hull, &shapes[*index].aabb());
        }
        let (aabb_bounds, _centroid_bounds) = convex_hull;

        let node_index = nodes.push(BVHNode {
            index: 0,
            len: indices.len() as u32,
            bounds: aabb_bounds,
        });
        BVHNode::subdivide(shapes, &mut indices, &mut nodes, node_index);

        BVH { nodes, indices }
    }

    /// lol
    pub fn traverse<'a, Shape: Bounded>(&'a self, ray: &Ray, shapes: &'a [Shape]) -> Vec<&Shape> {
        let mut indices = Vec::new();
        BVHNode::traverse_recursive(&self.nodes, 0, ray, &self.indices, &mut indices);
        indices
            .iter()
            .map(|index| &shapes[*index])
            .collect::<Vec<_>>()
    }

    fn intersect<'a, Shape: BHShape>(&'a self, ray: &Ray, shapes: &'a[Shape]) -> Option<(&Shape, Intersection)> {
        BVHNode::intersect(&self.nodes, ray, &self.indices, shapes)
    }
}

impl BoundingHierarchy for BVH {
    fn build<Shape: BHShape>(shapes: &mut [Shape]) -> BVH {
        BVH::build(shapes)
    }

    fn traverse<'a, Shape: Bounded>(&'a self, ray: &Ray, shapes: &'a [Shape]) -> Vec<&Shape> {
        self.traverse(ray, shapes)
    }

    fn intersect<'a, Shape: BHShape>(&'a self, ray: &Ray, shapes: &'a[Shape]) -> Option<(&Shape, Intersection)> {
        self.intersect(ray, shapes)
    }
      

    fn pretty_print(&self) {
    }
}



#[cfg(test)]
mod tests {
    use std::mem;
    use packed_bvh::BVH;
    use packed_bvh::BVHNode;
    use packed_bvh::BVHNodePool;
    use nalgebra::Point3;

    use testbase::{build_1200_triangles_bh,
                   build_12k_triangles_bh, build_120k_triangles_bh, intersect_1200_triangles_bh,
                   intersect_12k_triangles_bh, intersect_120k_triangles_bh, load_sponza_scene,
                   intersect_bh};

    #[test]
    fn bvhnode_allocation_works_as_expected() {
        let mut pool = BVHNodePool::new(64);
        let root = pool.push(BVHNode{
            index: 0,
            len: 0,
            bounds: Default::default(),
        });
        let left = pool.push(BVHNode{
            index: 0,
            len: 0,
            bounds: Default::default(),
        });
        let right = pool.push(BVHNode{
            index: 0,
            len: 0,
            bounds: Default::default(),
        });

        assert_eq!(root, 0, "root = 0");
        assert_eq!(left, 2, "root = 0");
        assert_eq!(right, 3, "root = 0");
    }

    #[test]
    fn bvhnode_is_cache_aligned() {
        let size = mem::size_of::<BVHNode>();
        let align = 2 * size;
        assert_eq!(size, 32, "expect BVHNode to be 32 in size");
        assert_eq!(align, 64, "expect alignment of 64 (cache line)");
    }

    #[test]
    fn bvhnodepool_allocates_safely() {
        let pool = BVHNodePool::new(64);
        for node in pool.iter() {
            assert_eq!(node.bounds.min, Point3::new(0., 0., 0.));
            assert_eq!(node.bounds.max, Point3::new(0., 0., 0.));
            assert_eq!(node.index, 0);
            assert_eq!(node.len, 0);
        }
    }


    #[bench]
    /// Benchmark the construction of a `BVH` with 1,200 triangles.
    fn bench_build_1200_triangles_packed_bvh(mut b: &mut ::test::Bencher) {
        build_1200_triangles_bh::<BVH>(&mut b);
    }

    #[bench]
    /// Benchmark the construction of a `BVH` with 12,000 triangles.
    fn bench_build_12k_triangles_packed_bvh(mut b: &mut ::test::Bencher) {
        build_12k_triangles_bh::<BVH>(&mut b);
    }

    #[bench]
    /// Benchmark the construction of a `BVH` with 120,000 triangles.
    fn bench_build_120k_triangles_packed_bvh(mut b: &mut ::test::Bencher) {
        build_120k_triangles_bh::<BVH>(&mut b);
    }

    #[bench]
    /// Benchmark the construction of a `BVH` for the Sponza scene.
    fn bench_build_sponza_packed_bvh(b: &mut ::test::Bencher) {
        let (mut triangles, _) = load_sponza_scene();
        b.iter(|| { BVH::build(&mut triangles); });
    }

    #[bench]
    /// Benchmark intersecting 1,200 triangles using the recursive `BVH`.
    fn bench_intersect_1200_triangles_packed_bvh(mut b: &mut ::test::Bencher) {
        intersect_1200_triangles_bh::<BVH>(&mut b);
    }

    #[bench]
    /// Benchmark intersecting 12,000 triangles using the recursive `BVH`.
    fn bench_intersect_12k_triangles_packed_bvh(mut b: &mut ::test::Bencher) {
        intersect_12k_triangles_bh::<BVH>(&mut b);
    }

    #[bench]
    /// Benchmark intersecting 120,000 triangles using the recursive `BVH`.
    fn bench_intersect_120k_triangles_packed_bvh(mut b: &mut ::test::Bencher) {
        intersect_120k_triangles_bh::<BVH>(&mut b);
    }

    #[bench]
    /// Benchmark the traversal of a `BVH` with the Sponza scene.
    fn bench_intersect_sponza_packed_bvh(b: &mut ::test::Bencher) {
        let (mut triangles, bounds) = load_sponza_scene();
        let bvh = BVH::build(&mut triangles);
        intersect_bh(&bvh, &triangles, &bounds, b)
    }

}
