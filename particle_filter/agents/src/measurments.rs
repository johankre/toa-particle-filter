use crate::swarm_element::SwarmElement;
use rand_distr::Normal;

pub(crate) trait Measurments {
    fn time_of_arival_mesurment(
        &self,
        swarm_element: &SwarmElement,
        raning_noise: &Normal<f32>,
    ) -> f32;
}
