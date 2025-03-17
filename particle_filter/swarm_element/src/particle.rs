use nalgebra::Vector3;
use rand::distr::Uniform;
use rand::Rng;
use rayon::prelude::*;

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
    pub fn normalize_weights(&mut self) {
        let sum: f64 = self.particles.iter().map(|p| p.weight).sum();
        self.particles.par_iter_mut().for_each(|p| p.weight /= sum);
    }
}

#[cfg(test)]
mod tests {
    use rand::seq::IndexedRandom;

    use crate::particle;

    use super::*;

    #[test]
    fn test_particle_new() {
        let particle = Particle::new(1.0, 2.0, 3.0, 0.1);
        assert_eq!(particle.position, Vector3::new(1.0, 2.0, 3.0));
        assert_eq!(particle.weight, 0.1);
    }

    #[test]
    fn test_particles_new() {
        let x_bounds = (0.0, 1.0);
        let y_bounds = (0.0, 1.0);
        let z_bounds = (0.0, 1.0);

        let num_particels = 100;

        let particles = Particles::new(num_particels, x_bounds, y_bounds, z_bounds);

        for particle in particles.particles.iter() {
            assert!(x_bounds.0 <= particle.position.x);
            assert!(x_bounds.1 >= particle.position.x);

            assert!(y_bounds.0 <= particle.position.y);
            assert!(y_bounds.1 >= particle.position.y);

            assert!(z_bounds.0 <= particle.position.z);
            assert!(z_bounds.1 >= particle.position.z);
        }
    }

    #[test]
    fn test_mean_uniform_particle_distribution() {
        let x_bounds = (0.0, 1.0);
        let y_bounds = (0.0, 2.0);
        let z_bounds = (0.0, 3.0);

        let num_particels = 10_000;

        let particles = Particles::new(num_particels, x_bounds, y_bounds, z_bounds);

        let x_mean = (x_bounds.1 - x_bounds.0) / 2.0;
        let y_mean = (y_bounds.1 - y_bounds.0) / 2.0;
        let z_mean = (z_bounds.1 - z_bounds.0) / 2.0;

        let mut x_sum = 0.0;
        let mut y_sum = 0.0;
        let mut z_sum = 0.0;

        particles.particles.iter().for_each(|p| {
            x_sum += p.position.x;
            y_sum += p.position.y;
            z_sum += p.position.z;
        });

        let x_empirical_mean = x_sum / (num_particels as f32);
        let y_empirical_mean = y_sum / (num_particels as f32);
        let z_empirical_mean = z_sum / (num_particels as f32);

        let tolerance = 0.1;

        assert!((x_mean - x_empirical_mean).abs() <= tolerance);
        assert!((y_mean - y_empirical_mean).abs() <= tolerance);
        assert!((z_mean - z_empirical_mean).abs() <= tolerance);
    }

    #[test]
    fn test_normalize_weitghts() {
        let x_bounds = (0.0, 1.0);
        let y_bounds = (0.0, 1.0);
        let z_bounds = (0.0, 1.0);

        let num_particels = 4;

        let mut particles = Particles::new(num_particels, x_bounds, y_bounds, z_bounds);

        for i in 0..particles.particles.len() {
            particles.particles[i].weight = 1.0 + i as f64;
        }

        particles.normalize_weights();

        let particle_weight_sum: f64 = particles.particles.iter().map(|p| p.weight).sum();
        assert!(particle_weight_sum == 1.0);
    }
}
