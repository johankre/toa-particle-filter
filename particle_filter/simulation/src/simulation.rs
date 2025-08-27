use colorous::INFERNO;
use nalgebra::Vector3;

use agents::{
    Measurements, anchor, dynamics_model::DynamicsModel, particle_filter::Particle, swarm_element,
};
use visualization::visualization::{Command, RerunVisualization};

pub struct Simulation<M: DynamicsModel> {
    pub swarm_elements: Vec<swarm_element::SwarmElement<M>>,
    pub anchors: Vec<anchor::Anchor>,
    visualizer: Option<RerunVisualization>,
}

pub struct SimulationBuilder<M: DynamicsModel> {
    swarm_elements: Option<Vec<swarm_element::SwarmElement<M>>>,
    anchors: Option<Vec<anchor::Anchor>>,

    visualizer: Option<RerunVisualization>,
}

impl<M: DynamicsModel> Simulation<M> {
    pub fn builder() -> SimulationBuilder<M> {
        SimulationBuilder {
            swarm_elements: None,
            anchors: None,
            visualizer: None,
        }
    }

    pub fn run(&mut self, steps: usize, step_size: f64)
    where
        M: Sync,
    {
        for frame in 0..steps {
            for se in &mut self.swarm_elements {
                se.step(step_size);
            }

            for se in &mut self.swarm_elements {
                let var_rx = se.ranging_noise.std_dev().powi(2);
                for anchor in self.anchors.iter() {
                    let var_tx = anchor.ranging_noise.std_dev().powi(2);
                    let combined_std = (var_rx + var_tx).sqrt();
                    let anchor_ranging = anchor.ranging(&se, combined_std);

                    se.particle_filter.update_weights(
                        anchor_ranging,
                        anchor.position,
                        combined_std,
                    );
                    se.particle_filter.normalize_weights();
                }
                se.update_est_position();
                se.particle_filter.resample();
            }

            if self.visualizer.is_some() {
                self.capture_frame(frame);
            }
        }
    }

    fn capture_frame(&mut self, frame: usize) {
        let viz = self.visualizer.as_mut().unwrap();

        // Temp hardcoding, at some point this will be handeld by a config parser
        let particle_size = 1.0;
        let swarm_size = 6.0;
        let anchors_size = 6.0;

        viz.log(Command::SetFrame(frame as i64));

        for swarm in &self.swarm_elements {
            let particle_positions: Vec<[f64; 3]> = swarm
                .particle_filter
                .particles
                .iter()
                .map(|p| {
                    let v: Vector3<f64> = p.position;
                    [v.x, v.y, v.z]
                })
                .collect();

            let particle_colors = Self::color_gradient(&swarm.particle_filter.particles);

            let entity_name = swarm.name.clone() + "/particle_filter";
            viz.log(Command::LogPoints(
                entity_name,
                particle_positions,
                particle_size,
                Some(particle_colors),
            ));

            let pos = swarm.dynamics_model.position();
            let swarm_true_position: Vec<[f64; 3]> = vec![[pos.x, pos.y, pos.z]];

            let entity_name = swarm.name.clone() + "/true_position";
            viz.log(Command::LogPoints(
                entity_name,
                swarm_true_position,
                swarm_size,
                None,
            ));

            let swarm_est_position: Vec<[f64; 3]> = vec![[
                swarm.est_position.x,
                swarm.est_position.y,
                swarm.est_position.z,
            ]];

            let entity_name = swarm.name.clone() + "/est_position";
            viz.log(Command::LogPoints(
                entity_name,
                swarm_est_position,
                swarm_size,
                None,
            ));

            if let Some(prev_est) = swarm.prev_positions.est_position {
                let entity_name = format!("{}/est_position/trajectory", swarm.name);

                let swarm_est_position: [f64; 3] = [
                    swarm.est_position.x,
                    swarm.est_position.y,
                    swarm.est_position.z,
                ];

                let swarm_prev_est_position: [f64; 3] = [prev_est.x, prev_est.y, prev_est.z];

                let est_trajectory = Command::LogTrajectory(
                    entity_name,
                    swarm_prev_est_position,
                    swarm_est_position,
                );
                viz.log(est_trajectory);
            }

            if let Some(prev_true) = swarm.prev_positions.true_position {
                let entity_name = format!("{}/true_position/trajectory", swarm.name);

                let pos = swarm.dynamics_model.position();
                let swarm_true_position: [f64; 3] = [pos.x, pos.y, pos.z];

                let swarm_prev_true_position: [f64; 3] = [prev_true.x, prev_true.y, prev_true.z];

                let true_trajectory = Command::LogTrajectory(
                    entity_name,
                    swarm_prev_true_position,
                    swarm_true_position,
                );
                viz.log(true_trajectory);
            }
        }

        if frame == 0 {
            let anchors_position: Vec<[f64; 3]> = self
                .anchors
                .iter()
                .map(|p| {
                    let v: Vector3<f64> = p.position;
                    [v.x, v.y, v.z]
                })
                .collect();

            let entity_name = String::from("anchors");
            viz.log(Command::LogPoints(
                entity_name,
                anchors_position,
                anchors_size,
                None,
            ));
        }
    }

    fn color_gradient(particles: &Vec<Particle>) -> Vec<[u8; 4]> {
        let n = particles.len() as f64;
        let lw_uniform = -n.ln();
        let lw_max = particles
            .iter()
            .map(|p| p.log_weight)
            .fold(f64::NEG_INFINITY, f64::max);

        particles
            .iter()
            .map(|p| Self::map_weight_to_color(p.log_weight, lw_max, lw_uniform))
            .collect()
    }

    fn map_weight_to_color(log_weight: f64, max_weight: f64, min_weight: f64) -> [u8; 4] {
        let den = (max_weight - min_weight).max(1e-12);
        let mut t = ((log_weight - min_weight) / den).clamp(0.0, 1.0);

        const GAMMA: f64 = 0.90;
        t = t.powf(GAMMA);

        let c = INFERNO.eval_continuous(t);
        [c.r, c.g, c.b, 255]
    }
}

impl<M: DynamicsModel> SimulationBuilder<M> {
    pub fn swarm_elements(mut self, swarm_elements: Vec<swarm_element::SwarmElement<M>>) -> Self {
        self.swarm_elements = Some(swarm_elements);
        self
    }

    pub fn anchors(mut self, anchors: Vec<anchor::Anchor>) -> Self {
        self.anchors = Some(anchors);
        self
    }

    pub fn visualizer(mut self, visualizer: RerunVisualization) -> Self {
        self.visualizer = Some(visualizer);
        self
    }

    pub fn build(self) -> Simulation<M> {
        Simulation {
            swarm_elements: self
                .swarm_elements
                .expect("expected at least one swarm element"),
            anchors: self.anchors.expect("expected at least one anchor"),
            visualizer: self.visualizer,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agents::{
        anchor::Anchor, dynamics_model::WhiteNoiseAcceleration, swarm_element::SwarmElement,
    };

    #[test]
    #[should_panic(expected = "expected at least one swarm element")]
    fn build_without_swarm_elements_panics() {
        Simulation::<WhiteNoiseAcceleration>::builder()
            .anchors(vec![Anchor::default()])
            .build();
    }

    #[test]
    #[should_panic(expected = "expected at least one anchor")]
    fn build_without_anchors_panics() {
        Simulation::<WhiteNoiseAcceleration>::builder()
            .swarm_elements(vec![SwarmElement::default()])
            .build();
    }

    #[test]
    fn build_with_defaults_sets() {
        let swarm_el = SwarmElement::default();
        let anchor = Anchor::default();

        let sim = Simulation::<WhiteNoiseAcceleration>::builder()
            .swarm_elements(vec![swarm_el.clone()])
            .anchors(vec![anchor.clone()])
            .build();

        assert_eq!(sim.swarm_elements.len(), 1);
        assert_eq!(sim.swarm_elements[0], swarm_el);
        assert_eq!(sim.anchors.len(), 1);
        assert_eq!(sim.anchors[0], anchor);
        assert!(sim.visualizer.is_none());
    }
}
