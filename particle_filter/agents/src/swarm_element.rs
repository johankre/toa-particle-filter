use crate::{dynamics_model::DynamicsModel, particle_filter::ParticleFilter, Measurements};

use nalgebra::Vector3;
use rand::rng;
use rand_distr::{Distribution, Normal};

#[derive(Debug, Clone, PartialEq)]
pub struct PrevPositions {
    pub true_position: Option<Vector3<f64>>,
    pub est_position: Option<Vector3<f64>>,
}

impl Default for PrevPositions {
    fn default() -> Self {
        PrevPositions {
            true_position: None,
            est_position: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SwarmElement<M: DynamicsModel> {
    pub name: String,

    pub dynamics_model: M,
    pub est_position: Vector3<f64>,
    pub particle_filter: ParticleFilter,

    pub transmission_noise: Normal<f64>,
    pub ranging_noise: Normal<f64>,

    pub prev_positions: PrevPositions,
}

impl<M> SwarmElement<M>
where
    M: DynamicsModel,
{
    pub fn new(
        name: String,
        dynamics_model: M,
        particle_filter: ParticleFilter,
        sd_transmition_noise: f64,
        sd_ranging_noise: f64,
    ) -> Self {
        let transmission_noise = Normal::new(0.0, sd_transmition_noise)
            .expect("SwarmElement: transmition_noise distribution failed");
        let ranging_noise = Normal::new(0.0, sd_ranging_noise)
            .expect("SwarmElement: transmition_noise distribution failed");
        let prev_positions = PrevPositions::default();
        Self {
            name,
            dynamics_model,
            est_position: Vector3::zeros(),
            particle_filter,
            transmission_noise,
            ranging_noise,
            prev_positions,
        }
    }

    pub fn update_est_position(&mut self) {
        let new_est = self.particle_filter.posterior_mean();

        match self.prev_positions.est_position {
            None => {
                // first valid estimate: seed both prev & current
                self.prev_positions.est_position = Some(new_est);
                self.est_position = new_est;
            }
            Some(_) => {
                // normal case: shift prev <- current, set current <- new
                self.prev_positions.est_position = Some(self.est_position);
                self.est_position = new_est;
            }
        }
    }

    pub fn estimation_error(&self) -> f64 {
        (self.dynamics_model.position() - self.est_position).norm()
    }

    fn get_ranging_velocity(&self) -> Vector3<f64> {
        let noise: Vector3<f64> = Vector3::new(
            self.transmission_noise.sample(&mut rng()),
            self.transmission_noise.sample(&mut rng()),
            self.transmission_noise.sample(&mut rng()),
        );

        self.dynamics_model.velocity() + noise
    }

    pub fn step(&mut self, dt: f64)
    where
        M: Sync,
    {
        self.particle_filter.predict_with_measured_velocity(
            dt,
            self.get_ranging_velocity(),
            &self.dynamics_model,
        );

        self.prev_positions.true_position = Some(self.dynamics_model.position());
        self.dynamics_model.step(dt, &mut rng());
    }

    pub fn debug_print(&self) {
        println!("=== SwarmElement [{}] ===", self.name);

        let pos = self.dynamics_model.position();
        let vel = self.dynamics_model.velocity();
        println!("Dynamics position: {:?}", pos);
        println!("Dynamics velocity: {:?}", vel);

        println!("PF estimated position: {:?}", self.est_position);
        let pf = &self.particle_filter;
        let n = pf.particles.len();
        println!("Particle filter: {} particles", n);
        let ess = pf.ess();
        println!("Current ESS: {:.2}", ess);

        if n > 0 {
            println!("First 3 particles:");
            for (i, p) in pf.particles.iter().take(3).enumerate() {
                println!("  [{}] pos={:?}, log_w={:.3}", i, p.position, p.log_weight);
            }
        }

        println!("=============================");
    }
}

impl<M> Default for SwarmElement<M>
where
    M: DynamicsModel + Default,
{
    fn default() -> Self {
        let noise = Normal::new(0.0, 1.0).unwrap();

        SwarmElement {
            name: String::new(),
            dynamics_model: M::default(),
            est_position: Vector3::zeros(),
            particle_filter: ParticleFilter::default(),
            transmission_noise: noise,
            ranging_noise: noise,
            prev_positions: PrevPositions::default(),
        }
    }
}

impl<M: DynamicsModel> Measurements<M> for SwarmElement<M> {
    fn ranging(&self, swarm_element: &SwarmElement<M>, std_raning_noise: f64) -> f64 {
        let noise = Normal::new(0.0, std_raning_noise).unwrap();
        let diff =
            (self.dynamics_model.position() - swarm_element.dynamics_model.position()).norm();
        diff + noise.sample(&mut rng())
    }
}

#[cfg(test)]
mod tests {
    use crate::dynamics_model::WhiteNoiseAcceleration;
    use crate::particle_filter::BoundingBox;
    use rayon::iter::IntoParallelIterator;
    use rayon::iter::ParallelIterator;

    use super::*;

    #[test]
    fn test_new_swarm_element() {
        let swarm_name = String::from("swarm_element_1");
        let x_bounds = (0.0, 1.0);
        let y_bounds = (0.0, 1.0);
        let z_bounds = (0.0, 1.0);

        let position = Vector3::new(0.5, 0.5, 0.5);
        let velocity = Vector3::new(0.1, 0.1, 0.1);
        let mean_a = Vector3::zeros();
        let sigma_a = Vector3::new(0.1, 0.1, 0.1);
        let dynamics_model = WhiteNoiseAcceleration::new(position, velocity, mean_a, sigma_a);

        let min = Vector3::new(x_bounds.0, y_bounds.0, z_bounds.0);
        let max = Vector3::new(x_bounds.1, y_bounds.1, z_bounds.1);
        let bounding_box = BoundingBox::new(min, max).unwrap();

        let num_particles = 10;
        let ess_tau = 0.5;
        let particle_filter = ParticleFilter::new(&bounding_box, num_particles, ess_tau);

        let transmission_noise = 0.1;
        let ranging_noise = 0.5;

        let swarm_element = SwarmElement::new(
            swarm_name.clone(),
            dynamics_model,
            particle_filter,
            transmission_noise,
            ranging_noise,
        );

        assert_eq!(swarm_element.name, swarm_name);
        assert_eq!(swarm_element.dynamics_model.position(), position);
        assert_eq!(swarm_element.est_position, Vector3::zeros());
        assert_eq!(swarm_element.prev_positions.est_position, None);
        assert_eq!(swarm_element.prev_positions.true_position, None);
        assert_eq!(swarm_element.dynamics_model.velocity(), velocity);
        assert_eq!(swarm_element.particle_filter.particles.len(), 10);
        swarm_element
            .particle_filter
            .particles
            .iter()
            .for_each(|p| assert_eq!(p.log_weight, -(num_particles as f64).ln()));
    }

    #[test]
    fn test_update_est_position() {
        let swarm_name = String::from("swarm_element_1");

        let position = Vector3::new(0.5, 0.5, 0.5);
        let velocity = Vector3::new(0.1, 0.1, 0.1);
        let mean_a = Vector3::zeros();
        let sigma_a = Vector3::new(0.1, 0.1, 0.1);
        let dynamics_model = WhiteNoiseAcceleration::new(position, velocity, mean_a, sigma_a);

        let x_bounds = (0.0, 1.0);
        let y_bounds = (0.0, 2.0);
        let z_bounds = (0.0, 3.0);

        let min = Vector3::new(x_bounds.0, y_bounds.0, z_bounds.0);
        let max = Vector3::new(x_bounds.1, y_bounds.1, z_bounds.1);
        let bounding_box = BoundingBox::new(min, max).unwrap();
        let num_particles = 100_000;
        let ess_tau = 0.5;
        let particle_filter = ParticleFilter::new(&bounding_box, num_particles, ess_tau);

        let transmission_noise = 0.1;
        let ranging_noise = 0.5;

        let mut swarm_element = SwarmElement::new(
            swarm_name,
            dynamics_model,
            particle_filter,
            transmission_noise,
            ranging_noise,
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

        assert_eq!(
            swarm_element.prev_positions.est_position,
            Some(swarm_element.est_position)
        );
    }

    #[test]
    fn test_swarm_element_ranging() {
        let swarm_name_1 = String::from("swarm_element_1");
        let swarm_name_2 = String::from("swarm_element_2");

        let position_1 = Vector3::new(0.5, 0.5, 0.5);
        let velocity_1 = Vector3::new(0.1, 0.1, 0.1);
        let mean_a = Vector3::zeros();
        let sigma_a = Vector3::new(0.1, 0.1, 0.1);
        let dynamics_model = WhiteNoiseAcceleration::new(position_1, velocity_1, mean_a, sigma_a);

        let enclosure =
            BoundingBox::new(Vector3::new(0.0, 0.0, 0.0), Vector3::new(5.0, 5.0, 5.0)).unwrap();
        let num_particles = 10;
        let ess_tau = 0.5;
        let particle_filter = ParticleFilter::new(&enclosure, num_particles, ess_tau);

        let transmission_noise_1 = 0.1;
        let ranging_noise_1 = 0.5;

        let sw1 = SwarmElement::new(
            swarm_name_1,
            dynamics_model,
            particle_filter,
            transmission_noise_1,
            ranging_noise_1,
        );

        let position_2 = Vector3::new(1.1, 4.2, 2.1);
        let velocity_2 = Vector3::new(0.2, 0.2, 0.2);
        let mean_a = Vector3::zeros();
        let sigma_a = Vector3::new(0.1, 0.1, 0.1);
        let dynamics_model = WhiteNoiseAcceleration::new(position_2, velocity_2, mean_a, sigma_a);

        let particle_filter = ParticleFilter::new(&enclosure, num_particles, ess_tau);

        let transmission_noise_2 = 0.1;
        let ranging_noise_2 = 0.5;

        let sw2 = SwarmElement::new(
            swarm_name_2,
            dynamics_model,
            particle_filter,
            transmission_noise_2,
            ranging_noise_2,
        );

        let measurement_std_deviation: f64 = 0.1;

        let num_samples = 100_000;
        let empirical_sum: f64 = (0..num_samples)
            .into_par_iter()
            .map(|_| sw1.ranging(&sw2, measurement_std_deviation))
            .sum();

        let empirical_mean = empirical_sum / num_samples as f64;

        let empirical_variance: f64 = (0..num_samples)
            .into_par_iter()
            .map(|_| {
                let x = sw1.ranging(&sw2, measurement_std_deviation);
                (x - empirical_mean).powi(2)
            })
            .sum::<f64>()
            / num_samples as f64;
        let expected_variance = measurement_std_deviation.powi(2);

        let mean_tolerance = 0.01;
        assert!(
            (empirical_mean
                - (sw1.dynamics_model.position() - sw2.dynamics_model.position()).norm())
                < mean_tolerance
        );

        let variance_tolerance = 0.01;
        assert!(expected_variance - empirical_variance < variance_tolerance);

        let is_nosiy = 0.0;
        assert!(empirical_variance > is_nosiy);
    }

    #[test]
    fn test_estimation_error_zero_when_estimate_equals_truth() {
        let swarm_name = String::from("test");

        let position = Vector3::new(0.5, 0.5, 0.5);
        let est_position = position.clone();
        let velocity = Vector3::new(0.1, 0.1, 0.1);
        let mean_a = Vector3::zeros();
        let sigma_a = Vector3::new(0.1, 0.1, 0.1);
        let dynamics_model = WhiteNoiseAcceleration::new(position, velocity, mean_a, sigma_a);

        let x_bounds = (0.0, 1.0);
        let y_bounds = (0.0, 2.0);
        let z_bounds = (0.0, 3.0);

        let min = Vector3::new(x_bounds.0, y_bounds.0, z_bounds.0);
        let max = Vector3::new(x_bounds.1, y_bounds.1, z_bounds.1);
        let bounding_box = BoundingBox::new(min, max).unwrap();
        let num_particles = 100_000;
        let ess_tau = 0.5;
        let particle_filter = ParticleFilter::new(&bounding_box, num_particles, ess_tau);

        let transmission_noise = 0.1;
        let ranging_noise = 0.5;

        let mut swarm_element = SwarmElement::new(
            swarm_name,
            dynamics_model,
            particle_filter,
            transmission_noise,
            ranging_noise,
        );

        swarm_element.est_position = est_position;
        let err = swarm_element.estimation_error();
        assert!(err.abs() <= 1e-12, "expected 0, got {err}");
    }

    #[test]
    fn test_estimation_error_is_euclidean_norm_of_difference() {
        let swarm_name = String::from("test");

        let position = Vector3::new(1.0, 2.0, 3.0);
        let velocity = Vector3::new(0.1, 0.1, 0.1);
        let mean_a = Vector3::zeros();
        let sigma_a = Vector3::new(0.1, 0.1, 0.1);
        let dynamics_model = WhiteNoiseAcceleration::new(position, velocity, mean_a, sigma_a);

        let x_bounds = (0.0, 1.0);
        let y_bounds = (0.0, 2.0);
        let z_bounds = (0.0, 3.0);

        let min = Vector3::new(x_bounds.0, y_bounds.0, z_bounds.0);
        let max = Vector3::new(x_bounds.1, y_bounds.1, z_bounds.1);

        let bounding_box = BoundingBox::new(min, max).unwrap();
        let num_particles = 100_000;
        let ess_tau = 0.5;
        let particle_filter = ParticleFilter::new(&bounding_box, num_particles, ess_tau);

        let transmission_noise = 0.1;
        let ranging_noise = 0.5;

        let mut swarm_element = SwarmElement::new(
            swarm_name,
            dynamics_model,
            particle_filter,
            transmission_noise,
            ranging_noise,
        );

        // truth at (1,2,3), estimate at (4,6,3) -> diff = (-3,-4,0), ||diff|| = 5
        let est_position = Vector3::new(4.0, 6.0, 3.0);
        let expected = 5.0;
        swarm_element.est_position = est_position;

        let err = swarm_element.estimation_error();
        assert!(
            (err - expected).abs() <= 1e-12,
            "got {err}, expected {expected}"
        );
    }
}
