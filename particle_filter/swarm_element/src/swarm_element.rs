use crate::Particles;
use nalgebra::Vector3;
use rayon::prelude::*;

pub struct SwarmElement {
    true_position: Vector3<f32>,
    est_position: Vector3<f32>,
    particles: Particles,
}

impl SwarmElement {
    pub fn new(
        true_position: Vector3<f32>,
        num_particles: usize,
        x_bounds: (f32, f32),
        y_bounds: (f32, f32),
        z_bounds: (f32, f32),
    ) -> Self {
        Self {
            true_position,
            est_position: Vector3::zeros(),
            particles: Particles::new(num_particles, x_bounds, y_bounds, z_bounds),
        }
    }

    pub fn update_est_position(&mut self) {
        self.est_position = self
            .particles
            .particles
            .par_iter()
            .map(|p| p.weight as f32 * p.position)
            .sum();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_swarm_element() {
        let true_position = Vector3::new(0.5, 0.5, 0.5);
        let num_particles = 10;
        let x_bounds = (0.0, 1.0);
        let y_bounds = (0.0, 1.0);
        let z_bounds = (0.0, 1.0);

        let swarm_element =
            SwarmElement::new(true_position, num_particles, x_bounds, y_bounds, z_bounds);

        assert_eq!(swarm_element.true_position, true_position);
        assert_eq!(swarm_element.est_position, Vector3::zeros());
        assert_eq!(swarm_element.particles.particles.len(), 10);
        swarm_element
            .particles
            .particles
            .iter()
            .for_each(|p| assert_eq!(p.weight, 1.0 / (num_particles as f64)));
    }

    #[test]
    fn test_update_est_position() {
        let true_position = Vector3::new(0.5, 0.5, 0.5);
        let num_particles = 10_000;
        let x_bounds = (0.0, 1.0);
        let y_bounds = (0.0, 2.0);
        let z_bounds = (0.0, 3.0);

        let mut swarm_element =
            SwarmElement::new(true_position, num_particles, x_bounds, y_bounds, z_bounds);
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
}
