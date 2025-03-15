use nalgebra::Vector3;

pub struct Paricle {
    pub position: Vector3<f32>,
    pub weight: f64,
}

impl Paricle{
   pub fn new(x: f32, y: f32, z: f32, weight: f64) -> Self {
        Self{ position: Vector3::new(x, y, z), weight }
   } 
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_particle_new() {
        let particle = Paricle::new(1.0, 2.0, 3.0, 0.1);
        assert_eq!(particle.position, Vector3::new(1.0, 2.0, 3.0));
        assert_eq!(particle.weight, 0.1);
    }
}
