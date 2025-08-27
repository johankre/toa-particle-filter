use std::f64;

use nalgebra::Vector3;
use rand::Rng;
use rand_distr::{Distribution, Normal};

pub trait DynamicsModel: Send {
    fn step(&mut self, dt: f64, rng: &mut impl Rng);
    fn position(&self) -> Vector3<f64>;
    fn velocity(&self) -> Vector3<f64>;
    fn predict_next_state(
        &self,
        dt: f64,
        position: Vector3<f64>,
        velocity: Vector3<f64>,
        rng: &mut impl Rng,
    ) -> Vector3<f64>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct WhiteNoiseAcceleration {
    pos: Vector3<f64>,
    vel: Vector3<f64>,
    accel_noise: Vector3<Normal<f64>>,
}
impl WhiteNoiseAcceleration {
    pub fn new(
        pos: Vector3<f64>,
        vel: Vector3<f64>,
        mean_a: Vector3<f64>,
        sigma_a: Vector3<f64>,
    ) -> Self {
        let ax =
            Normal::new(mean_a.x, sigma_a.x).expect("WhiteNoiseAcceleration: accel noise x failed");
        let ay =
            Normal::new(mean_a.y, sigma_a.y).expect("WhiteNoiseAcceleration: accel noise y failed");
        let az =
            Normal::new(mean_a.z, sigma_a.z).expect("WhiteNoiseAcceleration: accel noise z failed");

        let accel_noise = Vector3::new(ax, ay, az);
        Self {
            pos,
            vel,
            accel_noise,
        }
    }
}

impl Default for WhiteNoiseAcceleration {
    fn default() -> Self {
        WhiteNoiseAcceleration::new(
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 0.0),
        )
    }
}

impl DynamicsModel for WhiteNoiseAcceleration {
    fn step(&mut self, dt: f64, rng: &mut impl Rng) {
        let a: Vector3<f64> = Vector3::new(
            self.accel_noise.x.sample(rng),
            self.accel_noise.y.sample(rng),
            self.accel_noise.z.sample(rng),
        );

        self.pos += self.vel * dt + 0.5 * a * dt * dt;
        self.vel += a * dt;
    }

    fn position(&self) -> Vector3<f64> {
        self.pos
    }

    fn velocity(&self) -> Vector3<f64> {
        self.vel
    }

    fn predict_next_state(
        &self,
        dt: f64,
        position: Vector3<f64>,
        velocity: Vector3<f64>,
        rng: &mut impl Rng,
    ) -> Vector3<f64> {
        let a: Vector3<f64> = Vector3::new(
            self.accel_noise.x.sample(rng),
            self.accel_noise.y.sample(rng),
            self.accel_noise.z.sample(rng),
        );
        position + velocity * dt + 0.5 * a * dt * dt
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};
    use rand_distr::{Distribution, Normal};

    #[test]
    fn test_new_white_noise_acceleration() {
        let pos: Vector3<f64> = Vector3::new(0.1, 0.2, 0.3);
        let vel: Vector3<f64> = Vector3::new(1.0, 2.0, 3.0);

        let mean_a: Vector3<f64> = Vector3::new(10.0, 20.0, 30.0);
        let sigma_a: Vector3<f64> = Vector3::new(0.01, 0.02, 0.03);

        let white_noise_acceleration = WhiteNoiseAcceleration::new(pos, vel, mean_a, sigma_a);

        assert_eq!(white_noise_acceleration.position(), pos);
        assert_eq!(white_noise_acceleration.velocity(), vel);

        assert_eq!(white_noise_acceleration.accel_noise.x.mean(), mean_a.x);
        assert_eq!(white_noise_acceleration.accel_noise.y.mean(), mean_a.y);
        assert_eq!(white_noise_acceleration.accel_noise.z.mean(), mean_a.z);

        assert_eq!(white_noise_acceleration.accel_noise.x.std_dev(), sigma_a.x);
        assert_eq!(white_noise_acceleration.accel_noise.y.std_dev(), sigma_a.y);
        assert_eq!(white_noise_acceleration.accel_noise.z.std_dev(), sigma_a.z);
    }

    fn close(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn step_updates_pos_and_vel_as_expected() {
        let pos0 = Vector3::new(1.0, 2.0, 3.0);
        let vel0 = Vector3::new(0.5, -0.25, 0.1);
        let mean_a = Vector3::new(0.0, 0.0, 0.0);
        let sigma_a = Vector3::new(1.0, 2.0, 0.5);

        let mut model = WhiteNoiseAcceleration::new(pos0, vel0, mean_a, sigma_a);

        let seed: u64 = 42;
        let mut rng_for_step = StdRng::seed_from_u64(seed);
        let mut rng_for_expected = StdRng::seed_from_u64(seed);

        let dt = 0.1;
        let eps = 1e-12;

        model.step(dt, &mut rng_for_step);

        let got_pos = model.position();
        let got_vel = model.velocity();

        // pre-sample expected accelerations with the same distributions
        let nx = Normal::<f64>::new(mean_a.x, sigma_a.x).unwrap();
        let ny = Normal::<f64>::new(mean_a.y, sigma_a.y).unwrap();
        let nz = Normal::<f64>::new(mean_a.z, sigma_a.z).unwrap();

        let ax = nx.sample(&mut rng_for_expected);
        let ay = ny.sample(&mut rng_for_expected);
        let az = nz.sample(&mut rng_for_expected);
        let a = Vector3::new(ax, ay, az);

        let dt2 = dt * dt;
        let expected_pos = pos0 + vel0 * dt + 0.5 * a * dt2;
        let expected_vel = vel0 + a * dt;

        assert!(close(got_pos.x, expected_pos.x, eps));
        assert!(close(got_pos.y, expected_pos.y, eps));
        assert!(close(got_pos.z, expected_pos.z, eps));

        assert!(close(got_vel.x, expected_vel.x, eps));
        assert!(close(got_vel.y, expected_vel.y, eps));
        assert!(close(got_vel.z, expected_vel.z, eps));
    }

    #[test]
    fn step_with_zero_dt_does_not_change_state() {
        let pos0 = Vector3::new(1.0, 2.0, 3.0);
        let vel0 = Vector3::new(0.5, -0.25, 0.1);
        let mean_a = Vector3::new(0.0, 0.0, 0.0);
        let sigma_a = Vector3::new(1.0, 1.0, 1.0);

        let mut model = WhiteNoiseAcceleration::new(pos0, vel0, mean_a, sigma_a);

        let seed: u64 = 7;
        let mut rng = StdRng::seed_from_u64(seed);

        let step_size: f64 = 0.0;
        model.step(step_size, &mut rng);

        let eps = 1e-12;
        let got_pos = model.position();
        let got_vel = model.velocity();

        assert!((got_pos - pos0).abs().max() <= eps);
        assert!((got_vel - vel0).abs().max() <= eps);
    }

    #[test]
    fn predict_next_state_matches_manual_formula() {
        let mean_a = Vector3::new(0.0, 0.0, 0.0);
        let sigma_a = Vector3::new(1.0, 2.0, 0.5);
        let model = WhiteNoiseAcceleration::new(
            Vector3::zeros(), // model's internal pos (unused by predict_next_state)
            Vector3::zeros(), // model's internal vel (unused by predict_next_state)
            mean_a,
            sigma_a,
        );

        let pos0 = Vector3::new(1.0, 2.0, 3.0);
        let vel0 = Vector3::new(0.3, -0.2, 0.1);
        let dt = 0.1;

        let seed: u64 = 42;
        // one RNG for computing the expected value…
        let mut rng_expected = StdRng::seed_from_u64(seed);
        // …and an identical one passed into the method under test
        let mut rng_call = StdRng::seed_from_u64(seed);

        // Pre-sample the same acceleration the model will draw internally
        let ax = model.accel_noise.x.sample(&mut rng_expected);
        let ay = model.accel_noise.y.sample(&mut rng_expected);
        let az = model.accel_noise.z.sample(&mut rng_expected);
        let a = Vector3::new(ax, ay, az);

        let expected = pos0 + vel0 * dt + 0.5 * a * dt * dt;

        let got = model.predict_next_state(dt, pos0, vel0, &mut rng_call);

        let eps = 1e-12;
        assert!(
            (got - expected).abs().max() <= eps,
            "got {got:?}, expected {expected:?}"
        );
    }

    #[test]
    fn predict_next_state_zero_dt_returns_position() {
        let mean_a = Vector3::new(0.0, 0.0, 0.0);
        let sigma_a = Vector3::new(1.0, 1.0, 1.0);
        let model =
            WhiteNoiseAcceleration::new(Vector3::zeros(), Vector3::zeros(), mean_a, sigma_a);

        let pos0 = Vector3::new(-3.0, 0.5, 8.0);
        let vel0 = Vector3::new(9.0, -2.0, 1.0);
        let mut rng = StdRng::seed_from_u64(7);

        let got = model.predict_next_state(0.0, pos0, vel0, &mut rng);
        let eps = 1e-12;
        assert!((got - pos0).abs().max() <= eps);
    }
}
