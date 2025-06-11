use crate::measurments::Measurments;
use crate::swarm_element::SwarmElement;

use nalgebra::Vector3;
use rand::rng;
use rand_distr::{Distribution, Normal};

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Anchor {
    pub position: Vector3<f32>,
}

impl Anchor {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self {
            position: Vector3::new(x, y, z),
        }
    }
}

impl Measurments for Anchor {
    fn time_of_arival_mesurment(
        &self,
        swarm_element: &SwarmElement,
        measurement_std_deviation: f32,
    ) -> Result<f32, rand_distr::NormalError> {
        let normal_dist = Normal::new(0.0, measurement_std_deviation)?;
        let diff = self.position - swarm_element.true_position;
        Ok(diff.norm() + normal_dist.sample(&mut rng()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particle_filter::{BoundingBox, ParticleFilter};

    use rayon::prelude::*;

    #[test]
    fn test_new_anchor() {
        let x = 2.0;
        let y = 0.0;
        let z = 1.0;

        let anchor = Anchor::new(x, y, z);

        assert_eq!(anchor.position.x, x);
        assert_eq!(anchor.position.y, y);
        assert_eq!(anchor.position.z, z);
    }

    #[test]
    fn test_anchor_toa() {
        let measurement_std_deviation = 0.1;

        let anchor_x = 2.0;
        let anchor_y = 0.0;
        let anchor_z = 1.0;
        let anchor = Anchor::new(anchor_x, anchor_y, anchor_z);

        let swarm_element_name = String::from("test_1");
        let true_position = Vector3::new(3.3, 2.2, 1.1);
        let velocity = Vector3::new(0.1, 0.1, 0.1);
        let num_particles = 10;
        let enclosure =
            BoundingBox::new(Vector3::new(0.0, 0.0, 0.0), Vector3::new(5.0, 5.0, 5.0)).unwrap();
        let particle_filter = ParticleFilter::new(&enclosure, num_particles);
        let transmission_noise = 0.1;

        let swarm_element = SwarmElement::new(
            swarm_element_name,
            true_position,
            particle_filter,
            velocity,
            transmission_noise,
        );

        let num_samples = 100_000;
        let empirical_sum: f32 = (0..num_samples)
            .into_par_iter()
            .map(|_| {
                anchor
                    .time_of_arival_mesurment(&swarm_element, measurement_std_deviation)
                    .unwrap()
            })
            .sum();

        let empirical_mean = empirical_sum / num_samples as f32;

        let empirical_variance: f32 = (0..num_samples)
            .into_par_iter()
            .map(|_| {
                let x = anchor
                    .time_of_arival_mesurment(&swarm_element, measurement_std_deviation)
                    .unwrap();
                (x - empirical_mean).powi(2)
            })
            .sum::<f32>()
            / num_samples as f32;
        let expected_variance = measurement_std_deviation.powi(2);

        let mean_tolerance = 0.01;
        assert!(
            (empirical_mean - (anchor.position - swarm_element.true_position).norm())
                < mean_tolerance
        );

        let variance_tolerance = 0.01;
        assert!(expected_variance - empirical_variance < variance_tolerance);

        let is_nosiy = 0.0;
        assert!(empirical_variance > is_nosiy);
    }
}
