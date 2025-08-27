use crate::dynamics_model::DynamicsModel;
use crate::swarm_element::SwarmElement;
use crate::Measurements;

use nalgebra::Vector3;
use rand::rng;
use rand_distr::{Distribution, Normal};

#[derive(Debug, Clone, PartialEq)]
pub struct Anchor {
    pub position: Vector3<f64>,
    pub ranging_noise: Normal<f64>,
}

impl Anchor {
    pub fn new(position: Vector3<f64>, sd_ranging_noise: f64) -> Self {
        let ranging_noise = Normal::new(0.0, sd_ranging_noise)
            .expect("SwarmElement: transmition_noise distribution failed");
        Self {
            position,
            ranging_noise,
        }
    }
}

impl Default for Anchor {
    fn default() -> Self {
        let ranging_noise = Normal::new(0.0, 1.0).unwrap();
        Anchor {
            position: Vector3::zeros(),
            ranging_noise,
        }
    }
}

impl<M: DynamicsModel> Measurements<M> for Anchor {
    fn ranging(&self, swarm_element: &SwarmElement<M>, std_raning: f64) -> f64 {
        let noise = Normal::new(0.0, std_raning).unwrap();
        let diff = self.position - swarm_element.dynamics_model.position();
        diff.norm() + noise.sample(&mut rng())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dynamics_model::WhiteNoiseAcceleration,
        particle_filter::{BoundingBox, ParticleFilter},
    };

    use rayon::prelude::*;

    #[test]
    fn test_new_anchor() {
        let position = Vector3::new(2.0, 0.0, 1.0);
        let sd_ranging_noise = 0.1;
        let anchor = Anchor::new(position, sd_ranging_noise);

        assert_eq!(anchor.position.x, position.x);
        assert_eq!(anchor.position.y, position.y);
        assert_eq!(anchor.position.z, position.z);
        assert_eq!(anchor.ranging_noise.std_dev(), sd_ranging_noise);
    }

    #[test]
    fn test_anchor_ranging() {
        let position = Vector3::new(2.0, 0.0, 1.0);
        let sd_ranging_noise = 0.1;
        let anchor = Anchor::new(position, sd_ranging_noise);

        let swarm_element_name = String::from("test_1");
        let position = Vector3::new(3.3, 2.2, 1.1);
        let velocity = Vector3::new(0.1, 0.1, 0.1);
        let mean_a = Vector3::zeros();
        let sigma_a = Vector3::new(0.1, 0.1, 0.1);
        let dynamics_model = WhiteNoiseAcceleration::new(position, velocity, mean_a, sigma_a);

        let enclosure =
            BoundingBox::new(Vector3::new(0.0, 0.0, 0.0), Vector3::new(5.0, 5.0, 5.0)).unwrap();
        let num_particles = 10;
        let ess_tau = 0.5;
        let particle_filter = ParticleFilter::new(&enclosure, num_particles, ess_tau);

        let transmission_noise = 0.1;
        let ranging_noise = 0.1;

        let swarm_element = SwarmElement::new(
            swarm_element_name,
            dynamics_model,
            particle_filter,
            transmission_noise,
            ranging_noise,
        );

        let num_samples = 100_000;
        let empirical_sum: f64 = (0..num_samples)
            .into_par_iter()
            .map(|_| anchor.ranging(&swarm_element, sd_ranging_noise))
            .sum();

        let empirical_mean = empirical_sum / num_samples as f64;

        let empirical_variance: f64 = (0..num_samples)
            .into_par_iter()
            .map(|_| {
                let x = anchor.ranging(&swarm_element, sd_ranging_noise);
                (x - empirical_mean).powi(2)
            })
            .sum::<f64>()
            / num_samples as f64;
        let expected_variance = sd_ranging_noise.powi(2);

        let mean_tolerance = 0.01;
        assert!(
            (empirical_mean - (anchor.position - swarm_element.dynamics_model.position()).norm())
                < mean_tolerance
        );

        let variance_tolerance = 0.01;
        assert!(expected_variance - empirical_variance < variance_tolerance);

        let is_nosiy = 0.0;
        assert!(empirical_variance > is_nosiy);
    }
}
