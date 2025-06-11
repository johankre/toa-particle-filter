use crate::{measurments::Measurments, particle_filter::ParticleFilter};

use nalgebra::Vector3;
use rand::rng;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;

#[derive(Debug, Clone, PartialEq)]
pub struct SwarmElement {
    pub name: String,

    pub true_position: Vector3<f32>,
    pub est_position: Vector3<f32>,
    pub particle_filter: ParticleFilter,
    pub velocity: Vector3<f32>,

    pub transmission_noise: Normal<f32>,
}

impl SwarmElement {
    pub fn new(
        name: String,
        true_position: Vector3<f32>,
        particle_filter: ParticleFilter,
        velocity: Vector3<f32>,
        sd_transmition_noise: f32,
    ) -> Self {
        let transmission_noise = Normal::new(0.0, sd_transmition_noise)
            .expect("SwarmElement: transmition_noise distribution failed");
        Self {
            name,
            true_position,
            est_position: Vector3::zeros(),
            particle_filter,
            velocity,
            transmission_noise,
        }
    }

    pub fn update_est_position(&mut self) {
        self.est_position = self
            .particle_filter
            .particles
            .par_iter()
            .map(|p| p.weight as f32 * p.position)
            .sum();
    }

    pub fn move_position(&mut self) {
        let noise: Vector3<f32> = Vector3::new(
            self.transmission_noise.sample(&mut rng()),
            self.transmission_noise.sample(&mut rng()),
            self.transmission_noise.sample(&mut rng()),
        );
        self.true_position += self.velocity + noise;
    }
}

impl Default for SwarmElement {
    fn default() -> Self {
        let noise = Normal::new(0.0, 1.0).expect("transmission_noise σ must be > 0");

        SwarmElement {
            name: String::new(),
            true_position: Vector3::zeros(),
            est_position: Vector3::zeros(),
            particle_filter: ParticleFilter::default(),
            velocity: Vector3::zeros(),
            transmission_noise: noise,
        }
    }
}

impl Measurments for SwarmElement {
    fn time_of_arival_mesurment(
        &self,
        swarm_element: &SwarmElement,
        raning_noise: &Normal<f32>,
    ) -> f32 {
        let diff = self.true_position - swarm_element.true_position;
        diff.norm() + raning_noise.sample(&mut rng())
    }
}

#[cfg(test)]
mod tests {
    use crate::particle_filter::BoundingBox;

    use super::*;

    #[test]
    fn test_new_swarm_element() {
        let swarm_name = String::from("swarm_element_1");
        let true_position = Vector3::new(0.5, 0.5, 0.5);
        let velocity = Vector3::new(0.1, 0.1, 0.1);
        let num_particles = 10;
        let x_bounds = (0.0, 1.0);
        let y_bounds = (0.0, 1.0);
        let z_bounds = (0.0, 1.0);

        let min = Vector3::new(x_bounds.0, y_bounds.0, z_bounds.0);
        let max = Vector3::new(x_bounds.1, y_bounds.1, z_bounds.1);

        let bounding_box = BoundingBox::new(min, max).unwrap();
        let particle_filter = ParticleFilter::new(&bounding_box, num_particles);

        let transmission_noise = 0.1;

        let swarm_element = SwarmElement::new(
            swarm_name.clone(),
            true_position,
            particle_filter,
            velocity,
            transmission_noise,
        );

        assert_eq!(swarm_element.name, swarm_name);
        assert_eq!(swarm_element.true_position, true_position);
        assert_eq!(swarm_element.est_position, Vector3::zeros());
        assert_eq!(swarm_element.velocity, velocity);
        assert_eq!(swarm_element.particle_filter.particles.len(), 10);
        swarm_element
            .particle_filter
            .particles
            .iter()
            .for_each(|p| assert_eq!(p.weight, 1.0 / (num_particles as f64)));
    }

    #[test]
    fn test_update_est_position() {
        let swarm_name = String::from("swarm_element_1");

        let true_position = Vector3::new(0.5, 0.5, 0.5);
        let velocity = Vector3::new(0.1, 0.1, 0.1);
        let num_particles = 100_000;
        let x_bounds = (0.0, 1.0);
        let y_bounds = (0.0, 2.0);
        let z_bounds = (0.0, 3.0);

        let min = Vector3::new(x_bounds.0, y_bounds.0, z_bounds.0);
        let max = Vector3::new(x_bounds.1, y_bounds.1, z_bounds.1);

        let bounding_box = BoundingBox::new(min, max).unwrap();
        let particle_filter = ParticleFilter::new(&bounding_box, num_particles);

        let transmission_noise = 0.1;

        let mut swarm_element = SwarmElement::new(
            swarm_name,
            true_position,
            particle_filter,
            velocity,
            transmission_noise,
        );
        swarm_element.update_est_position();

        let tolerance = 0.01;

        assert!(
            (swarm_element.est_position.x - (x_bounds.1 - x_bounds.0) / 2.0).abs() <= tolerance
        );
        assert!(
            (swarm_element.est_position.y - (y_bounds.1 - y_bounds.0) / 2.0).abs() <= tolerance
        );
        assert!(
            (swarm_element.est_position.z - (z_bounds.1 - z_bounds.0) / 2.0).abs() <= tolerance
        );
    }

    #[test]
    fn test_swarm_element_toa() {
        let swarm_name_1 = String::from("swarm_element_1");
        let swarm_name_2 = String::from("swarm_element_2");

        let num_particles = 10;
        let enclosure =
            BoundingBox::new(Vector3::new(0.0, 0.0, 0.0), Vector3::new(5.0, 5.0, 5.0)).unwrap();

        let sw1_true_position = Vector3::new(3.3, 2.2, 1.1);
        let velocity_1 = Vector3::new(0.1, 0.1, 0.1);
        let particle_filter = ParticleFilter::new(&enclosure, num_particles);
        let transmission_noise_1 = 0.1;
        let sw1 = SwarmElement::new(
            swarm_name_1,
            sw1_true_position,
            particle_filter,
            velocity_1,
            transmission_noise_1,
        );

        let sw2_true_position = Vector3::new(1.1, 4.2, 2.1);
        let velocity_2 = Vector3::new(0.2, 0.2, 0.2);
        let particle_filter = ParticleFilter::new(&enclosure, num_particles);
        let transmission_noise_2 = 0.1;
        let sw2 = SwarmElement::new(
            swarm_name_2,
            sw2_true_position,
            particle_filter,
            velocity_2,
            transmission_noise_2,
        );

        let measurement_std_deviation = 0.1;
        let ranging_noise = Normal::new(0.0, measurement_std_deviation).unwrap();

        let num_samples = 100_000;
        let empirical_sum: f32 = (0..num_samples)
            .into_par_iter()
            .map(|_| sw1.time_of_arival_mesurment(&sw2, &ranging_noise))
            .sum();

        let empirical_mean = empirical_sum / num_samples as f32;

        let empirical_variance: f32 = (0..num_samples)
            .into_par_iter()
            .map(|_| {
                let x = sw1.time_of_arival_mesurment(&sw2, &ranging_noise);
                (x - empirical_mean).powi(2)
            })
            .sum::<f32>()
            / num_samples as f32;
        let expected_variance = measurement_std_deviation.powi(2);

        let mean_tolerance = 0.01;
        assert!((empirical_mean - (sw1.true_position - sw2.true_position).norm()) < mean_tolerance);

        let variance_tolerance = 0.01;
        assert!(expected_variance - empirical_variance < variance_tolerance);

        let is_nosiy = 0.0;
        assert!(empirical_variance > is_nosiy);
    }
}
