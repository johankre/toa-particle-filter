use crate::Particles;
use nalgebra::Vector3;

pub struct SwarmElement {
    true_position: Vector3<f32>,
    est_position: Vector3<f32>,
    particles: Particles,
}

impl SwarmElement {
    pub fn new(true_position: Vector3<f32>, num_particles: usize) -> Self {
        Self {
            true_position,
            est_position: Vector3::zeros(),
            particles: Particles::new(num_particles, (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_swarm_element() {
        let true_position = Vector3::new(0.5, 0.5, 0.5);
        let num_particles = 10;

        let swarm_element = SwarmElement::new(true_position, num_particles);

        assert_eq!(swarm_element.true_position, true_position);
        assert_eq!(swarm_element.est_position, Vector3::zeros());
        assert_eq!(swarm_element.particles.particles.len(), 10);
        swarm_element
            .particles
            .particles
            .iter()
            .for_each(|p| assert_eq!(p.weight, 1.0 / (num_particles as f64)));
    }
}
