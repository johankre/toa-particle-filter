use nalgebra::Vector3;

use agents::{anchor, swarm_element};
use visualization::visualization::{Command, RerunVisualization};

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
        for frame in 0..time_steps {
            for se in &mut self.swarm_elements {
                se.move_position();
                se.particle_filter
                    .update_position(se.velocity, se.transmission_noise);

                se.update_est_position();
            }

            if let Some(viz) = &mut self.visualizer {
                // Temp hardcoding, at some point this will be handeld by a config parser
                let particle_size = 2.0;
                let swarm_size = 6.0;
                let anchors_size = 6.0;

                viz.log(Command::SetFrame(frame as i64));

                for swarm in &self.swarm_elements {
                    let particle_positions: Vec<[f32; 3]> = swarm
                        .particle_filter
                        .particles
                        .iter()
                        .map(|p| {
                            let v: Vector3<f32> = p.position;
                            [v.x, v.y, v.z]
                        })
                        .collect();

                    let entity_name = swarm.name.clone() + "/particle_filter";
                    viz.log(Command::LogPoints(
                        entity_name,
                        particle_positions,
                        particle_size,
                    ));

                    let swarm_true_position: Vec<[f32; 3]> = vec![[
                        swarm.true_position.x,
                        swarm.true_position.y,
                        swarm.true_position.z,
                    ]];

                    let entity_name = swarm.name.clone() + "/true_position";
                    viz.log(Command::LogPoints(
                        entity_name,
                        swarm_true_position,
                        swarm_size,
                    ));

                    let swarm_est_position: Vec<[f32; 3]> = vec![[
                        swarm.est_position.x,
                        swarm.est_position.y,
                        swarm.est_position.z,
                    ]];

                    let entity_name = swarm.name.clone() + "/est_position";
                    viz.log(Command::LogPoints(
                        entity_name,
                        swarm_est_position,
                        swarm_size,
                    ));
                }

                let anchors_position: Vec<[f32; 3]> = self
                    .anchors
                    .iter()
                    .map(|p| {
                        let v: Vector3<f32> = p.position;
                        [v.x, v.y, v.z]
                    })
                    .collect();

                let entity_name = String::from("anchors");
                viz.log(Command::LogPoints(
                    entity_name,
                    anchors_position,
                    anchors_size,
                ));
            }
        }
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
