use agents::{
    anchor::Anchor,
    particle_filter::{BoundingBox, ParticleFilter, Spher},
    swarm_element::SwarmElement,
};
use nalgebra::Vector3;
use simulation::simulation::SimulationBuilder;
use visualization::visualization::RerunVisualization;

fn main() {
    let swarm_name_1 = String::from("swarm_element_1");
    let swarm_name_2 = String::from("swarm_element_2");

    let true_position = Vector3::new(0.5, 0.5, 0.5);
    let velocity = Vector3::new(0.1, 0.1, 0.1);
    let sd_transmition_noise = 0.1;
    let num_particles = 100_000;
    let x_bounds = (0.0, 1.0);
    let y_bounds = (0.0, 2.0);
    let z_bounds = (0.0, 3.0);

    let min = Vector3::new(x_bounds.0, y_bounds.0, z_bounds.0);
    let max = Vector3::new(x_bounds.1, y_bounds.1, z_bounds.1);
    let bounding_box = BoundingBox::new(min, max).unwrap();
    let particle_filter_1 = ParticleFilter::new(&bounding_box, num_particles);

    let r = 1.0;
    let origo = Vector3::new(0.5, 0.5, 0.5);
    let spher = Spher::new(r, origo).unwrap();
    let particle_filter_2 = ParticleFilter::new(&spher, num_particles);

    let swarm_element_1 = SwarmElement::new(
        swarm_name_1,
        true_position,
        particle_filter_1,
        velocity,
        sd_transmition_noise,
    );

    let swarm_element_2 = SwarmElement::new(
        swarm_name_2,
        true_position,
        particle_filter_2,
        velocity,
        sd_transmition_noise,
    );

    let anchor = Anchor::default();

    let visualization_name = String::from("ToA-Particle_Filter");
    let visualizer =
        RerunVisualization::new(visualization_name).expect("Unable to create rerun visulization");

    let mut sim = SimulationBuilder::default()
        .swarm_elements(vec![swarm_element_1, swarm_element_2])
        .anchors(vec![anchor.clone()])
        .visualizer(visualizer)
        .build();

    let time_steps = 10;
    sim.run(time_steps);
}
