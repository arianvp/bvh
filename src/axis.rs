//! Axis enum for indexing three-dimensional structures.

use std::ops::{Index, IndexMut};
use std::fmt::{Display, Formatter, Result};
use nalgebra::{Point3, Vector3};

/// An `Axis` in a three-dimensional coordinate system.
/// Used to access `Vector3`/`Point3` structs via index.
///
/// # Examples
/// ```
/// use bvh::axis::Axis;
///
/// let mut position = [1.0, 0.5, 42.0];
/// position[Axis::Y] *= 4.0;
///
/// assert_eq!(position[Axis::Y], 2.0);
/// ```
///
/// `nalgebra` structures are also indexable using `Axis`.
/// For reference see [the documentation]
/// (http://nalgebra.org/doc/nalgebra/struct.Vector3.html#method.index).
///
/// ```
/// extern crate bvh;
/// extern crate nalgebra;
///
/// use bvh::axis::Axis;
/// use nalgebra::Point3;
///
/// # fn main() {
/// let mut position: Point3<f32> = Point3::new(1.0, 2.0, 3.0);
/// position[Axis::X] = 1000.0;
///
/// assert_eq!(position[Axis::X], 1000.0);
/// # }
/// ```
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Axis {
    /// Index of the X axis.
    X = 0,

    /// Index of the Y axis.
    Y = 1,

    /// Index of the Z axis.
    Z = 2,
}

/// Display implementation for `Axis`.
impl Display for Axis {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f,
               "{}",
               match *self {
                   Axis::X => "x",
                   Axis::Y => "y",
                   Axis::Z => "z",
               })
    }
}

/// Make slices indexable by `Axis`.
impl Index<Axis> for [f32] {
    type Output = f32;

    fn index(&self, axis: Axis) -> &f32 {
        &self[axis as usize]
    }
}

/// Make `Point3` indexable by `Axis`.
impl Index<Axis> for Point3<f32> {
    type Output = f32;

    fn index(&self, axis: Axis) -> &f32 {
        match axis {
            Axis::X => &self.x,
            Axis::Y => &self.y,
            Axis::Z => &self.z,
        }
    }
}

/// Make `Vector3` indexable by `Axis`.
impl Index<Axis> for Vector3<f32> {
    type Output = f32;

    fn index(&self, axis: Axis) -> &f32 {
        match axis {
            Axis::X => &self.x,
            Axis::Y => &self.y,
            Axis::Z => &self.z,
        }
    }
}

/// Make slices mutably accessible by `Axis`.
impl IndexMut<Axis> for [f32] {
    fn index_mut(&mut self, axis: Axis) -> &mut f32 {
        &mut self[axis as usize]
    }
}

/// Make `Point3` mutably accessible by `Axis`.
impl IndexMut<Axis> for Point3<f32> {
    fn index_mut(&mut self, axis: Axis) -> &mut f32 {
        match axis {
            Axis::X => &mut self.x,
            Axis::Y => &mut self.y,
            Axis::Z => &mut self.z,
        }
    }
}

/// Make `Vector3` mutably accessible by `Axis`.
impl IndexMut<Axis> for Vector3<f32> {
    fn index_mut(&mut self, axis: Axis) -> &mut f32 {
        match axis {
            Axis::X => &mut self.x,
            Axis::Y => &mut self.y,
            Axis::Z => &mut self.z,
        }
    }
}

#[cfg(test)]
mod test {
    use axis::Axis;

    /// Test whether accessing arrays by index is the same as accessing them by `Axis`.
    quickcheck!{
        fn test_index_by_axis(tpl: (f32, f32, f32)) -> bool {
            let a = [tpl.0, tpl.1, tpl.2];

            a[0] == a[Axis::X] && a[1] == a[Axis::Y] && a[2] == a[Axis::Z]
        }
    }

    /// Test whether arrays can be mutably set, by indexing via `Axis`.
    quickcheck!{
        fn test_set_by_axis(tpl: (f32, f32, f32)) -> bool {
            let mut a = [0.0, 0.0, 0.0];

            a[Axis::X] = tpl.0;
            a[Axis::Y] = tpl.1;
            a[Axis::Z] = tpl.2;

            a[0] == tpl.0 && a[1] == tpl.1 && a[2] == tpl.2
        }
    }
}
