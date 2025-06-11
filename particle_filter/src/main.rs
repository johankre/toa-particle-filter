use agents::{
    anchor::Anchor,
    particle_filter::{BoundingBox, ParticleFilter},
    swarm_element::SwarmElement,
};
use nalgebra::Vector3;
use simulation::simulation::SimulationBuilder;
use visualization::visualization::RerunVisualization;

fn main() {
    let swarm_name = String::from("swarm_element_1");

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
    let particle_filter = ParticleFilter::new(&bounding_box, num_particles);

    let swarm_element = SwarmElement::new(swarm_name, true_position, particle_filter, velocity, sd_transmition_noise);
    let anchor = Anchor::default();

    let sim = SimulationBuilder::default()
        .swarm_elements(vec![swarm_element])
        .anchors(vec![anchor.clone()])
        .build();

    let visulization_name = String::from("ToA-Particle_Filter");
    let visulizator =
        RerunVisualization::new(visulization_name).expect("Unable to create rerun visulization");

    visulizator
        .capture_frame(&sim)
        .expect("Simulation: unable to capture frame");
}
