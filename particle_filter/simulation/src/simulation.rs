use colorous::INFERNO;
use nalgebra::Vector3;
use once_cell::sync::Lazy;

use agents::{Measurements, anchor, particle_filter::Particle, swarm_element};
use visualization::visualization::{Command, RerunVisualization};

const W_MIN: f64 = 1e-9;
const W_MAX: f64 = 5e-5;
const EPSILON: f64 = 1e-10;
const GAMMA: f64 = 0.7;

static LOG_MIN: Lazy<f64> = Lazy::new(|| (W_MIN + EPSILON).ln());
static INV_LOG_SPAN: Lazy<f64> = Lazy::new(|| {
    let log_max = (W_MAX + EPSILON).ln();
    let span = (log_max - *LOG_MIN).max(f64::EPSILON);
    1.0 / span
});

pub struct Simulation {
    pub swarm_elements: Vec<swarm_element::SwarmElement>,
    pub anchors: Vec<anchor::Anchor>,
    visualizer: Option<RerunVisualization>,
}

#[derive(Default)]
pub struct SimulationBuilder {
    swarm_elements: Option<Vec<swarm_element::SwarmElement>>,
    anchors: Option<Vec<anchor::Anchor>>,

    visualizer: Option<RerunVisualization>,
}

impl Simulation {
    pub fn builder() -> SimulationBuilder {
        SimulationBuilder::default()
    }

    pub fn run(&mut self, time_steps: usize) {
        let len = self.swarm_elements.len();
        for frame in 0..time_steps {
            for se in &mut self.swarm_elements {
                se.move_position();
            }
            for i in 0..len {
                let (head, tail) = self.swarm_elements.split_at_mut(i);
                let se_i = &mut tail[0];
                for se_j in head.iter() {
                    let var_rx = se_i.ranging_noise.std_dev().powi(2);
                    let var_tx = se_j.ranging_noise.std_dev().powi(2);
                    let combined_std = (var_rx + var_tx).sqrt();

                    let ranging = se_i.ranging(&se_j, combined_std);

                    se_i.particle_filter
                        .update_weights(ranging, se_j.est_position, combined_std);
                }

                let se_i = &mut self.swarm_elements[i];
                let var_rx = se_i.ranging_noise.std_dev().powi(2);
                for anchor in self.anchors.iter() {
                    let var_tx = anchor.ranging_noise.std_dev().powi(2);
                    let combined_std = (var_rx + var_tx).sqrt();
                    let i_ranging_anchor = anchor.ranging(&se_i, combined_std);

                    se_i.particle_filter.update_weights(
                        i_ranging_anchor,
                        anchor.position,
                        combined_std,
                    );
                }
            }

            for se in &mut self.swarm_elements {
                se.particle_filter.normalize_weights();
                se.update_est_position();
            }

            if self.visualizer.is_some() {
                self.capture_frame(frame);
            }

            for se in &mut self.swarm_elements {
                se.particle_filter.resample();
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

            let swarm_true_position: Vec<[f64; 3]> = vec![[
                swarm.true_position.x,
                swarm.true_position.y,
                swarm.true_position.z,
            ]];

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
        }

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

    fn color_gradient(particles: &Vec<Particle>) -> Vec<[u8; 4]> {
        particles
            .iter()
            .map(|&p| Self::map_weight_to_color(p.weight))
            .collect()
    }

    fn map_weight_to_color(w: f64) -> [u8; 4] {
        let wc = (w.clamp(W_MIN, W_MAX) + EPSILON).ln();
        let t_lin = (wc - *LOG_MIN) * *INV_LOG_SPAN;
        let t = t_lin.clamp(0.0, 1.0).powf(GAMMA);
        let c = INFERNO.eval_continuous(t);
        [c.r, c.g, c.b, 255]
    }
}

impl SimulationBuilder {
    pub fn swarm_elements(mut self, swarm_elements: Vec<swarm_element::SwarmElement>) -> Self {
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

    pub fn build(self) -> Simulation {
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
    use agents::{anchor::Anchor, swarm_element::SwarmElement};

    #[test]
    #[should_panic(expected = "expected at least one swarm element")]
    fn build_without_swarm_elements_panics() {
        SimulationBuilder::default()
            .anchors(vec![Anchor::default()])
            .build();
    }

    #[test]
    #[should_panic(expected = "expected at least one anchor")]
    fn build_without_anchors_panics() {
        SimulationBuilder::default()
            .swarm_elements(vec![SwarmElement::default()])
            .build();
    }

    #[test]
    fn build_with_defaults_sets() {
        let swarm_el = SwarmElement::default();
        let anchor = Anchor::default();

        let sim = SimulationBuilder::default()
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
