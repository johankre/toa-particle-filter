use agents::{
    anchor::Anchor,
    particle_filter::{ParticleFilter, Sphere},
    swarm_element::SwarmElement,
};
use nalgebra::Vector3;
use simulation::simulation::SimulationBuilder;
use visualization::visualization::RerunVisualization;

fn main() {
    let swarm_name = String::from("swarm_element_1");
    let sd_transmission_noise = 0.01;
    let sd_ranging_noise = 0.4;
    let num_particles = 1_400_000;
    let true_position = Vector3::new(10.0, 10.0, 5.0);
    let velocity = Vector3::new(0.5, 0.5, 0.5);
    let radius = 20.0;
    let origin = Vector3::new(10.0, 10.0, 5.0);

    let sphere = Sphere::new(radius, origin).unwrap();
    let particle_filter = ParticleFilter::new(&sphere, num_particles);

    let swarm_element = SwarmElement::new(
        swarm_name,
        true_position,
        particle_filter,
        velocity,
        sd_transmission_noise,
        sd_ranging_noise,
    );

    let anchor_std = 0.1;
    let anchor1 = Anchor::new(Vector3::new(0.0, 0.0, 0.0), anchor_std);

    let visualizer = RerunVisualization::new(String::from("ToA-Particle-Filter"))
        .expect("Unable to create rerun visualization");

    let mut sim = SimulationBuilder::default()
        .swarm_elements(vec![swarm_element])
        .anchors(vec![anchor1])
        .visualizer(visualizer)
        .build();

    let time_steps = 30;
    sim.run(time_steps);
}
