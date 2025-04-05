use nalgebra::Vector3;

struct Anchor {
    position: Vector3<f32>,
}

impl Anchor {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self {
            position: Vector3::new(x, y, z),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_anchor() {
        let x = 2.0;
        let y = 0.0;
        let z = 1.0;

        let anchor = Anchor::new(x, y, z);

        assert_eq!(anchor.position.x, x);
        assert_eq!(anchor.position.y, y);
        assert_eq!(anchor.position.z, z);
    }
}
