use crate::particle_filter::ParticleFilter;
use nalgebra::Vector3;
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
}
