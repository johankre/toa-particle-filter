use nalgebra::Vector3;
use crate::Paricle;

pub struct SwarmElement {
    position: Vector3<f32>,
    particles: Vec<Paricle>,
}
