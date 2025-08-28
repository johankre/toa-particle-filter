use agents::{
    anchor::Anchor,
    dynamics_model::WhiteNoiseAcceleration,
    particle_filter::{ParticleFilter, Sphere},
    swarm_element::SwarmElement,
};
use nalgebra::Vector3;
use simulation::simulation::Simulation;
use visualization::visualization::RerunVisualization;

fn main() {
    let swarm_name = String::from("1000");

    let position = Vector3::new(100.0, 20.0, 5.0);
    let velocity = Vector3::new(10.0, 0.0, 0.0);
    let mean_a = Vector3::new(0.0, -50.0, 10.0);
    let sigma_a = Vector3::new(0.5, 50.0, 50.0);
    let dynamics_model = WhiteNoiseAcceleration::new(position, velocity, mean_a, sigma_a);

    let radius = 200.0;
    let origin = Vector3::new(10.0, 10.0, 5.0);
    let sphere = Sphere::new(radius, origin).unwrap();
    let num_particles = 5_000;
    let tau = 0.5;
    let particle_filter = ParticleFilter::new(&sphere, num_particles, tau);

    let sd_transmission_noise = 0.1;
    let sd_ranging_noise = 0.8;

    let swarm_element_1 = SwarmElement::new(
        swarm_name,
        dynamics_model,
        particle_filter,
        sd_transmission_noise,
        sd_ranging_noise,
    );

    let anchor_std = 0.4;
    let anchor1 = Anchor::new(Vector3::new(0.0, 0.0, 0.0), anchor_std);
    let anchor2 = Anchor::new(Vector3::new(0.0, 50.0, 0.0), anchor_std);
    let anchor3 = Anchor::new(Vector3::new(50.0, 0.0, 0.0), anchor_std);

    let visualizer = RerunVisualization::new(String::from("ToA-Particle-Filter"))
        .expect("Unable to create rerun visualization");

    let mut sim = Simulation::builder()
        .swarm_elements(vec![swarm_element_1])
        .anchors(vec![anchor1, anchor2, anchor3])
        .visualizer(visualizer)
        .build();

    let time_steps = 1000;
    let step_size = 0.1;
    sim.run(time_steps, step_size);
}
