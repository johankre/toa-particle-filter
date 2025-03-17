use crate::Particles;
use nalgebra::Vector3;

pub struct SwarmElement {
    position: Vector3<f32>,
    particles: Particles,
}

impl SwarmElement {
    pub fn new(position: Vector3<f32>, num_particles: usize) {}
}
