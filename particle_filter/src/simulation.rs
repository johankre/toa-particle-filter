use agents::{anchor, swarm_element};

struct Simulation {
    swarm_elements: Vec<swarm_element::SwarmElement>,
    anchors: Vec<anchor::Anchor>,

    sd_swarm: f32,
    sd_agent: f32,
}

#[derive(Default)]
struct SimulationBuilder {
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
    pub fn swarm_elements(&mut self, swarm_elements: Vec<swarm_element::SwarmElement>) {
        self.swarm_elements = Some(swarm_elements);
    }

    pub fn anchors(&mut self, anchors: Vec<anchor::Anchor>) {
        self.anchors = Some(anchors);
    }

    pub fn sd_swarm(&mut self, sd_swarm: f32) {
        self.sd_swarm = Some(sd_swarm);
    }

    pub fn sd_agent(&mut self, sd_agent: f32) {
        self.sd_agent = Some(sd_agent);
    }

    pub fn build(self) -> Simulation {
        Simulation {
            swarm_elements: self
                .swarm_elements
                .expect("expected at least one swarm element"),
            anchors: self.anchors.expect("expected at least on anchor"),

            sd_swarm: self.sd_swarm.unwrap_or(1.0),
            sd_agent: self.sd_agent.unwrap_or(1.0),
        }
    }
}
