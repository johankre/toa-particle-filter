use crate::{measurments::Measurments, particle_filter::ParticleFilter};

use nalgebra::Vector3;
use rand::rng;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;

pub struct SwarmElement {
    pub true_position: Vector3<f32>,
    est_position: Vector3<f32>,
    particle_filter: ParticleFilter,
}

impl SwarmElement {
    pub fn new(true_position: Vector3<f32>, particle_filter: ParticleFilter) -> Self {
        Self {
            true_position,
            est_position: Vector3::zeros(),
            particle_filter,
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

    pub fn move_position(&mut self, velocity: Vector3<f32>, time_step: f32) {
        self.true_position += velocity * time_step;
    }
}

impl Measurments for SwarmElement {
    fn time_of_arival_mesurment(
        &self,
        swarm_element: &SwarmElement,
        measurement_std_deviation: f32,
    ) -> Result<f32, rand_distr::NormalError> {
        let normal_dist = Normal::new(0.0, measurement_std_deviation)?;
        let diff = self.true_position - swarm_element.true_position;
        Ok(diff.norm() + normal_dist.sample(&mut rng()))
    }
}

#[cfg(test)]
mod tests {
    use crate::particle_filter::BoundingBox;

    use super::*;

    #[test]
    fn test_new_swarm_element() {
        let true_position = Vector3::new(0.5, 0.5, 0.5);
        let num_particles = 10;
        let x_bounds = (0.0, 1.0);
        let y_bounds = (0.0, 1.0);
        let z_bounds = (0.0, 1.0);

        let min = Vector3::new(x_bounds.0, y_bounds.0, z_bounds.0);
        let max = Vector3::new(x_bounds.1, y_bounds.1, z_bounds.1);

        let bounding_box = BoundingBox::new(min, max).unwrap();
        let particle_filter = ParticleFilter::new(&bounding_box, num_particles);

        let swarm_element = SwarmElement::new(true_position, particle_filter);

        assert_eq!(swarm_element.true_position, true_position);
        assert_eq!(swarm_element.est_position, Vector3::zeros());
        assert_eq!(swarm_element.particle_filter.particles.len(), 10);
        swarm_element
            .particle_filter
            .particles
            .iter()
            .for_each(|p| assert_eq!(p.weight, 1.0 / (num_particles as f64)));
    }

    #[test]
    fn test_update_est_position() {
        let true_position = Vector3::new(0.5, 0.5, 0.5);
        let num_particles = 100_000;
        let x_bounds = (0.0, 1.0);
        let y_bounds = (0.0, 2.0);
        let z_bounds = (0.0, 3.0);

        let min = Vector3::new(x_bounds.0, y_bounds.0, z_bounds.0);
        let max = Vector3::new(x_bounds.1, y_bounds.1, z_bounds.1);

        let bounding_box = BoundingBox::new(min, max).unwrap();
        let particle_filter = ParticleFilter::new(&bounding_box, num_particles);

        let mut swarm_element = SwarmElement::new(true_position, particle_filter);
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
    fn test_move_swarm_element() {
        let true_position = Vector3::new(0.5, 0.5, 0.5);
        let velocity = Vector3::new(0.1, 0.1, 0.0);
        let time_step = 1.0;
        let num_particles = 10_000;
        let x_bounds = (0.0, 1.0);
        let y_bounds = (0.0, 2.0);
        let z_bounds = (0.0, 3.0);

        let min = Vector3::new(x_bounds.0, y_bounds.0, z_bounds.0);
        let max = Vector3::new(x_bounds.1, y_bounds.1, z_bounds.1);

        let bounding_box = BoundingBox::new(min, max).unwrap();
        let particle_filter = ParticleFilter::new(&bounding_box, num_particles);

        let mut swarm_element = SwarmElement::new(true_position, particle_filter);

        swarm_element.move_position(velocity, time_step);
        assert_eq!(swarm_element.true_position, Vector3::new(0.6, 0.6, 0.5));
    }

    #[test]
    fn test_swarm_element_toa() {
        let measurement_std_deviation = 0.1;

        let num_particles = 10;
        let enclosure =
            BoundingBox::new(Vector3::new(0.0, 0.0, 0.0), Vector3::new(5.0, 5.0, 5.0)).unwrap();

        let sw1_true_position = Vector3::new(3.3, 2.2, 1.1);
        let particle_filter = ParticleFilter::new(&enclosure, num_particles);
        let sw1 = SwarmElement::new(sw1_true_position, particle_filter);

        let sw2_true_position = Vector3::new(1.1, 4.2, 2.1);
        let particle_filter = ParticleFilter::new(&enclosure, num_particles);
        let sw2 = SwarmElement::new(sw2_true_position, particle_filter);

        let num_samples = 100_000;
        let empirical_sum: f32 = (0..num_samples)
            .into_par_iter()
            .map(|_| {
                sw1.time_of_arival_mesurment(&sw2, measurement_std_deviation)
                    .unwrap()
            })
            .sum();

        let empirical_mean = empirical_sum / num_samples as f32;

        let empirical_variance: f32 = (0..num_samples)
            .into_par_iter()
            .map(|_| {
                let x = sw1
                    .time_of_arival_mesurment(&sw2, measurement_std_deviation)
                    .unwrap();
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
