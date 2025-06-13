use agents::{
    anchor::Anchor,
    particle_filter::{ParticleFilter, Spher},
    swarm_element::SwarmElement,
};
use nalgebra::Vector3;
use simulation::simulation::SimulationBuilder;
use visualization::visualization::RerunVisualization;

fn main() {
    let swarm_name_1 = String::from("swarm_element_1");

    let sd_transmition_noise = 0.02;
    let sd_ragning_noise = 0.5;
    let num_particles = 600_000;
    let true_position_1 = Vector3::new(2.0, 4.0, 1.0);
    let velocity_1 = Vector3::new(0.0, 0.4, 0.4);
    let r = 9.0;
    let origo = Vector3::new(0.5, 5.0, 0.5);
    let spher = Spher::new(r, origo).unwrap();
    let particle_filter_1 = ParticleFilter::new(&spher, num_particles);

    let swarm_element_1 = SwarmElement::new(
        swarm_name_1,
        true_position_1,
        particle_filter_1,
        velocity_1,
        sd_transmition_noise,
        sd_ragning_noise,
    );

    let anchor_std_ranging_1 = 0.2;
    let anchor_position_1 = Vector3::new(0.0, 0.0, 0.0);
    let anchor_1 = Anchor::new(anchor_position_1, anchor_std_ranging_1);

    let anchor_position_2 = Vector3::new(0.0, 10.0, 0.0);
    let anchor_2 = Anchor::new(anchor_position_2, anchor_std_ranging_1);

    let anchor_position_3 = Vector3::new(10.0, 0.0, 0.0);
    let anchor_3 = Anchor::new(anchor_position_3, anchor_std_ranging_1);

    let visualization_name = String::from("ToA-Particle_Filter");
    let visualizer =
        RerunVisualization::new(visualization_name).expect("Unable to create rerun visulization");

    let mut sim = SimulationBuilder::default()
        .swarm_elements(vec![swarm_element_1])
        .anchors(vec![anchor_1, anchor_2, anchor_3])
        .visualizer(visualizer)
        .build();

    let time_steps = 30;
    sim.run(time_steps);
}
