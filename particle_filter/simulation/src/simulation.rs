use agents::{anchor, swarm_element};

pub struct Simulation {
    pub swarm_elements: Vec<swarm_element::SwarmElement>,
    pub anchors: Vec<anchor::Anchor>,

    sd_swarm: f32,
    sd_agent: f32,
}

#[derive(Default)]
pub struct SimulationBuilder {
    swarm_elements: Option<Vec<swarm_element::SwarmElement>>,
    anchors: Option<Vec<anchor::Anchor>>,

    sd_swarm: Option<f32>,
    sd_agent: Option<f32>,
}

impl Simulation {
    pub fn builder() -> SimulationBuilder {
        SimulationBuilder::default()
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

    pub fn sd_swarm(mut self, sd_swarm: f32) -> Self {
        self.sd_swarm = Some(sd_swarm);
        self
    }

    pub fn sd_agent(mut self, sd_agent: f32) -> Self {
        self.sd_agent = Some(sd_agent);
        self
    }

    pub fn build(self) -> Simulation {
        Simulation {
            swarm_elements: self
                .swarm_elements
                .expect("expected at least one swarm element"),
            anchors: self.anchors.expect("expected at least one anchor"),

            sd_swarm: self.sd_swarm.unwrap_or(1.0),
            sd_agent: self.sd_agent.unwrap_or(1.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agents::{swarm_element::SwarmElement, anchor::Anchor};

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
    fn build_with_defaults_sets_1_0() {
        let swarm_el = SwarmElement::default();
        let anchor  = Anchor::default();

        let sim = SimulationBuilder::default()
            .swarm_elements(vec![swarm_el.clone()])
            .anchors(vec![anchor.clone()])
            .build();

        assert_eq!(sim.swarm_elements.len(), 1);
        assert_eq!(sim.swarm_elements[0], swarm_el);
        assert_eq!(sim.anchors.len(), 1);
        assert_eq!(sim.anchors[0], anchor);

        assert!((sim.sd_swarm - 1.0).abs() < f32::EPSILON);
        assert!((sim.sd_agent - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn build_with_custom_sd() {
        let swarm_el = SwarmElement::default();
        let anchor  = Anchor::default();

        let sim = SimulationBuilder::default()
            .swarm_elements(vec![swarm_el])
            .anchors(vec![anchor])
            .sd_swarm(2.5)
            .sd_agent(3.5)
            .build();

        assert!((sim.sd_swarm - 2.5).abs() < f32::EPSILON);
        assert!((sim.sd_agent - 3.5).abs() < f32::EPSILON);
    }
}

