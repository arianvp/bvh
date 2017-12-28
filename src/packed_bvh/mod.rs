//! Module defines [`BVH`] and [`BVHNode`] and functions for building and traversing it.
//! It uses a packed representation that allowes two nodes to fit in a cache line,
//! and allocates nodes from a node pool that has been aligned to cache boundaries.
use EPSILON;
use std::f32;
use std::mem;
use std::heap;
use std::heap::Alloc;
use std::ops::Deref;
use std::ops::DerefMut;
use nalgebra::Point3;
use stdsimd::simd::f32x4;
use stdsimd::vendor::{_mm_max_ps,_mm_min_ps, _mm_comige_ss, _mm_movehl_ps, _mm_max_ss, _mm_min_ss, _mm_shuffle_ps};
use aabb::{AABB, Bounded};
use ray::Ray;
use ray::Intersection;
use bounding_hierarchy::{BoundingHierarchy, BHShape};
use utils::Bucket;
use std::ptr::Unique;


#[derive(Copy, Clone)]
struct MinAndIndex {
    min: Point3<f32>,
    index: u32,
}

#[derive(Copy, Clone)]
struct MaxAndLen {
    max: Point3<f32>,
    len: u32,
}

#[derive(Copy, Clone)]
union MinOrIndex {
    min4: f32x4,
    min_and_index: MinAndIndex,
}

#[derive(Copy, Clone)]
union MaxOrLen {
    max4: f32x4,
    max_and_len: MaxAndLen,
}

/// BVH Node. either internal or leaf
#[derive(Copy, Clone)]
pub struct BVHNode {
    /// a
    min_or_index: MinOrIndex,

    /// b
    max_or_len: MaxOrLen,
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
    #[inline(always)]
    fn new(index: u32, len: u32, bounds: AABB) -> BVHNode {
        let min = bounds.min;
        let max = bounds.max;
        BVHNode {
            min_or_index: MinOrIndex{ min_and_index: MinAndIndex { min, index }},
            max_or_len: MaxOrLen{max_and_len: MaxAndLen { max, len }},
        }
    }
    #[inline(always)]
    fn set_bounds(&mut self, bounds: AABB) {
        unsafe {
            self.min_or_index.min_and_index.min = bounds.min;
            self.max_or_len.max_and_len.max = bounds.max;
        }
    }
    #[inline(always)]
    fn bounds(&self) -> AABB {
        unsafe {
            AABB { min: self.min_or_index.min_and_index.min, max: self.max_or_len.max_and_len.max }
        }
    }
    #[inline(always)]
    fn set_index(&mut self, index: u32) {
        unsafe { self.min_or_index.min_and_index.index = index }
    }
    #[inline(always)]
    fn index(&self) -> u32 {
        unsafe { self.min_or_index.min_and_index.index }
    }
    #[inline(always)]
    fn set_len(&mut self, len: u32) {
        unsafe { self.max_or_len.max_and_len.len = len }
    }
    #[inline(always)]
    fn len(&self) -> u32 {
        unsafe { self.max_or_len.max_and_len.len }
    }

    #[inline(always)]
    fn intersects_ray(&self, ray: &Ray) -> Option<f32> {
        unsafe {
            let origin = ray.origin.xyz0;
            let invdir = ray.inv_direction.xyz0;
            let boxmin = self.min_or_index.min4;
            let boxmax = self.max_or_len.max4;

            let l1 = (boxmin - origin) * invdir;
            let l2 = (boxmax - origin) * invdir;

            let l1a      = _mm_min_ps(l1, f32x4::splat(f32::INFINITY));
            let l2a      = _mm_min_ps(l2, f32x4::splat(f32::INFINITY));

            let l1b      = _mm_max_ps(l1, f32x4::splat(-f32::INFINITY));
            let l2b      = _mm_max_ps(l2, f32x4::splat(-f32::INFINITY));

            let mut lmax = _mm_max_ps(l1a, l2a);
            let mut lmin = _mm_min_ps(l1b, l2b);

            let lmax0    = _mm_shuffle_ps(lmax, lmax, 0x39);
            let lmin0    = _mm_shuffle_ps(lmin, lmin, 0x39);

            lmax         = _mm_min_ss(lmax, lmax0);
            lmin         = _mm_max_ss(lmin, lmin0);

            let lmax1    = _mm_movehl_ps(lmax, lmax);
            let lmin1    = _mm_movehl_ps(lmin, lmin);

            lmax         = _mm_min_ss(lmax, lmax1);
            lmin         = _mm_max_ss(lmin, lmin1);

            // TODO use these for early out traversal
            let t_far = lmin.extract(0);

            if (_mm_comige_ss(lmax, f32x4::splat(0.)) & _mm_comige_ss(lmax, lmin)) != 0 {
                Some(t_far)
            } else {
                None
            }

            
        }
    }


    #[inline]
    fn intersect<'a, Shape: BHShape>(nodes: &[BVHNode], ray: &Ray, indices: &Vec<usize>, shapes: &'a[Shape]) -> Option<(&'a Shape, Intersection)> {
        let mut stack = Vec::with_capacity(12);
        stack.push(0);
        let mut answer: Option<(&Shape, Intersection)> = None;
        while let Some(index) = stack.pop() {
            let node = nodes[index];
            if node.len() == 0 {
                let left = node.index() as usize;
                let a = nodes[left].intersects_ray(ray);
                let b = nodes[left + 1].intersects_ray(ray);
                match(a,b) {
                    (Some(t1), Some(t2)) => if t1 <= t2 {
                        stack.push(left + 1);
                        stack.push(left);
                    } else { 
                        stack.push(left);
                        stack.push(left + 1);
                    },
                    (Some(_), None) =>  stack.push(left),
                    (None, Some(_)) => stack.push(left + 1),
                    (None, None) => {},
                }
            } else {
                for i in node.index()..node.index()+node.len() {
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
        if node.len() == 0 {
            let left = node.index() as usize;
            let l = ray.intersects_aabb(&nodes[left].bounds());
            let r = ray.intersects_aabb(&nodes[left + 1].bounds());
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
            for i in node.index()..node.index()+node.len() {
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
        if node.len() == 0 {
            let left = node.index() as usize;
            let l = ray.intersects_aabb(&nodes[left].bounds());
            let r = ray.intersects_aabb(&nodes[left + 1].bounds());
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
            rindices.extend((node.index()..node.index()+node.len()).map(|i| indices[i as usize]).collect::<Vec<usize>>())
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
        for i in node.index() .. node.index()+node.len() {
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

        const NUM_BUCKETS: usize = 8;
        let mut buckets = [Bucket::empty(); NUM_BUCKETS];

        for i in node.index() .. node.index() + node.len() {
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

        let mut min_cost = node.bounds().surface_area() * node.len() as f32;
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
                let split_value = centroid_bounds.min[split_axis] + (1 + min_bucket) as f32 * (split_axis_size / (NUM_BUCKETS as f32 - 0.01));
                let mut pivot_index = node.index() as usize;
                for i in node.index().. node.index() + node.len() {
                    let center = shapes[indices[i as usize]].aabb().center()[split_axis];
                    if center <= split_value {
                        indices.swap(pivot_index, i as usize);
                        pivot_index += 1;
                    }
                }

                let left_node = BVHNode::new(
                    node.index() as u32,
                    (pivot_index - node.index() as usize) as u32,
                    l_aabb,
                );

                let left_index = nodes.push(left_node) as u32;
                
                let right_node = BVHNode::new(
                    pivot_index as u32,
                    (node.len() as usize - (pivot_index - node.index() as usize)) as u32,
                    r_aabb,
                );

                nodes.push(right_node);

                nodes[node_index].set_index(left_index);
                nodes[node_index].set_len(0);
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
        if nodes[node_index].len() < 3 {
            shapes[nodes[node_index].index() as usize].set_bh_node_index(node_index);
            return;
        }

        if BVHNode::partition(shapes, indices, nodes, node_index) {
            let left = nodes[node_index].index() as usize;
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

        let node_index = nodes.push(BVHNode::new(
            0,
            indices.len() as u32,
            aabb_bounds,
        ));
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
        let root = pool.push(BVHNode::new(0, 0, Default::default()));
        let left = pool.push(BVHNode::new(0, 0, Default::default()));
        let right = pool.push(BVHNode::new(0, 0, Default::default()));

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
            assert_eq!(node.bounds().min, Point3::new(0., 0., 0.));
            assert_eq!(node.bounds().max, Point3::new(0., 0., 0.));
            assert_eq!(node.index(), 0);
            assert_eq!(node.len(), 0);
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
