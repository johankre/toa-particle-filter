use crate::dynamics_model::DynamicsModel;
use crate::swarm_element::SwarmElement;

pub trait Measurements<M: DynamicsModel> {
    fn ranging(&self, swarm_element: &SwarmElement<M>, std_raning: f64) -> f64;
}
