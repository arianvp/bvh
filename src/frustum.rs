use nalgebra::Vector3;

pub struct Frustum {
    pub planes: [Plane; 6],
}

pub struct Plane {
    pub normal: Vector3<f32>,
    pub d: f32,
}
