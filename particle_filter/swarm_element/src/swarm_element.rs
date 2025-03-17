use crate::Particles;
use nalgebra::Vector3;

pub struct SwarmElement {
    position: Vector3<f32>,
    particles: Particles,
}

impl SwarmElement {
    pub fn new(position: Vector3<f32>, num_particles: usize) -> Self {
        Self {
            position,
            particles: Particles::new(num_particles, (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::seq::IndexedRandom;

    use super::*;

    #[test]
    fn test_new_swarm_element() {
        let position = Vector3::new(0.5, 0.5, 0.5);
        let num_particles = 10;

        let swarm_element = SwarmElement::new(position, num_particles);

        assert_eq!(swarm_element.position, position);
        assert_eq!(swarm_element.particles.particles.len(), 10);
        swarm_element
            .particles
            .particles
            .iter()
            .for_each(|p| assert_eq!(p.weight, 1.0 / (num_particles as f64)));
    }
}
