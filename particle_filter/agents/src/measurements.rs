use crate::swarm_element::SwarmElement;
use rand_distr::Normal;

pub trait Measurements {
    fn ranging(&self, swarm_element: &SwarmElement, raning_noise: &Normal<f32>) -> f32;
}
