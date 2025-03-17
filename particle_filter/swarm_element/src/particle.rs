use nalgebra::Vector3;
use rand::distr::{Distribution, Uniform};
use rand::{rng, Rng};

pub struct Particle {
    pub position: Vector3<f32>,
    pub weight: f64,
}

pub struct Particles {
    particles: Vec<Particle>,
}

impl Particle {
    pub(super) fn new(x: f32, y: f32, z: f32, weight: f64) -> Self {
        Self {
            position: Vector3::new(x, y, z),
            weight,
        }
    }
}

impl Particles {
    pub(crate) fn new(
        num_particels: usize,
        x_bounds: (f32, f32),
        y_bounds: (f32, f32),
        z_bounds: (f32, f32),
    ) -> Self {
        let mut rng = rand::rng();
        let x_uni = Uniform::new(x_bounds.0, x_bounds.1).unwrap();
        let y_uni = Uniform::new(y_bounds.0, y_bounds.1).unwrap();
        let z_uni = Uniform::new(z_bounds.0, z_bounds.1).unwrap();

        let particles: Vec<Particle> = (0..num_particels)
            .map(|_| {
                Particle::new(
                    rng.sample(x_uni),
                    rng.sample(y_uni),
                    rng.sample(z_uni),
                    1.0 / (num_particels as f64),
                )
            })
            .collect();

        Particles { particles }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_particle_new() {
        let particle = Particle::new(1.0, 2.0, 3.0, 0.1);
        assert_eq!(particle.position, Vector3::new(1.0, 2.0, 3.0));
        assert_eq!(particle.weight, 0.1);
    }
}
