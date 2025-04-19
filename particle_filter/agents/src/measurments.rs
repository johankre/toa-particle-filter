use crate::swarm_element::SwarmElement;

pub(crate) trait Measurments {
    fn time_of_arival_mesurment(
        &self,
        swarm_element: &SwarmElement,
        measurement_std_deviation: f32,
    ) -> Result<f32, rand_distr::NormalError>;
}
