use nalgebra::Vector3;
use rand::distr::Uniform;
use rand::Rng;
use rayon::prelude::*;

pub trait Enclosure {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vector3<f32>;
}

pub struct BoundingBox {
    pub dist_x: Uniform<f32>,
    pub dist_y: Uniform<f32>,
    pub dist_z: Uniform<f32>,
}

impl BoundingBox {
    pub fn new(min: Vector3<f32>, max: Vector3<f32>) -> Result<Self, rand::distr::uniform::Error> {
        Ok(Self {
            dist_x: Uniform::new(min.x, max.x)?,
            dist_y: Uniform::new(min.y, max.y)?,
            dist_z: Uniform::new(min.z, max.z)?,
        })
    }
}

impl Enclosure for BoundingBox {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vector3<f32> {
        let x = rng.sample(self.dist_x);
        let y = rng.sample(self.dist_y);
        let z = rng.sample(self.dist_z);

        Vector3::new(x, y, z)
    }
}

pub struct Particle {
    pub position: Vector3<f32>,
    pub weight: f64,
}

pub struct ParticleFilter {
    pub particles: Vec<Particle>,
}

impl Particle {
    pub fn new(position: Vector3<f32>, weight: f64) -> Self {
        Self { position, weight }
    }
}

impl ParticleFilter {
    pub fn new<E: Enclosure>(enclosure: &E, num_particles: usize) -> Self {
        let mut rng = rand::rng();
        let particles: Vec<Particle> = (0..num_particles)
            .map(|_| {
                let pos = enclosure.sample(&mut rng);
                Particle::new(pos, 1.0 / (num_particles as f64))
            })
            .collect();

        ParticleFilter { particles }
    }
    pub fn normalize_weights(&mut self) {
        let sum: f64 = self.particles.par_iter().map(|p| p.weight).sum();
        self.particles.par_iter_mut().for_each(|p| p.weight /= sum);
    }

    pub fn update_position(&mut self, velocity: Vector3<f32>, time_step: f32) {
        self.particles
            .par_iter_mut()
            .for_each(|p| p.position += velocity * time_step);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_particle_new() {
        let position = Vector3::new(1.0, 2.0, 3.0);

        let particle = Particle::new(position, 0.1);

        assert_eq!(particle.position, Vector3::new(1.0, 2.0, 3.0));
        assert_eq!(particle.weight, 0.1);
    }

    #[test]
    fn test_particles_new() {
        let x_bounds = (0.0, 1.0);
        let y_bounds = (0.0, 1.0);
        let z_bounds = (0.0, 1.0);

        let min = Vector3::new(x_bounds.0, y_bounds.0, z_bounds.0);
        let max = Vector3::new(x_bounds.1, y_bounds.1, z_bounds.1);

        let bounding_box = BoundingBox::new(min, max).unwrap();

        let num_particles = 100;

        let particle_filter = ParticleFilter::new(&bounding_box, num_particles);

        for particle in particle_filter.particles.iter() {
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
        let x_bounds = (0.0, 10.0);
        let y_bounds = (0.0, 20.0);
        let z_bounds = (0.0, 30.0);

        let min = Vector3::new(x_bounds.0, y_bounds.0, z_bounds.0);
        let max = Vector3::new(x_bounds.1, y_bounds.1, z_bounds.1);

        let bounding_box = BoundingBox::new(min, max).unwrap();

        let num_particles = 100_000;

        let particle_filter = ParticleFilter::new(&bounding_box, num_particles);

        let x_mean = (x_bounds.1 - x_bounds.0) / 2.0;
        let y_mean = (y_bounds.1 - y_bounds.0) / 2.0;
        let z_mean = (z_bounds.1 - z_bounds.0) / 2.0;

        let mut x_sum = 0.0;
        let mut y_sum = 0.0;
        let mut z_sum = 0.0;

        particle_filter.particles.iter().for_each(|p| {
            x_sum += p.position.x;
            y_sum += p.position.y;
            z_sum += p.position.z;
        });

        let x_empirical_mean = x_sum / (num_particles as f32);
        let y_empirical_mean = y_sum / (num_particles as f32);
        let z_empirical_mean = z_sum / (num_particles as f32);

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

        let min = Vector3::new(x_bounds.0, y_bounds.0, z_bounds.0);
        let max = Vector3::new(x_bounds.1, y_bounds.1, z_bounds.1);

        let bounding_box = BoundingBox::new(min, max).unwrap();

        let num_particles = 4;

        let mut particle_filter = ParticleFilter::new(&bounding_box, num_particles);

        for i in 0..particle_filter.particles.len() {
            particle_filter.particles[i].weight = 1.0 + i as f64;
        }

        particle_filter.normalize_weights();

        let particle_weight_sum: f64 = particle_filter.particles.iter().map(|p| p.weight).sum();
        assert!(particle_weight_sum == 1.0);
    }
}
