# TOA Particle Filter
A Rust implementation of a Time-of-Arrival particle filter for localization and positon tracking

## Features
- Simulate TOA measurements from multiple anchors and between swarm elements
- Particle filter for position estimation
- Support for multiple agents (swarm)
- Configurable anchors and swarm elements
- Real-time visualization with Rerun

## Getting Started
### Installation
```console
git clone https://github.com/johankre/ToA-Particle-Filter.git
cd ToA-Particle-Filter.git
cargo build --release
```

### Usage
Run the simulation:
```console
cargo run --release
```

Note: SimulationBuilder requires at least one swarm element and one anchor to build. Providing a visualizer is optional.

### Dependencies

This project uses the [Rerun viewer](https://github.com/rerun-io/rerun) for visualization.  
To enable visualization, you need to have the Rerun viewer installed and available in your environment.

Follow the installation instructions from the [Rerun documentation](https://rerun.io/docs/getting-started/installing-viewer).

##  Example

A minimal example using a single swarm element and two anchors:

```rust
fn main() {
    let swarm_name_1 = String::from("swarm_element_1");

    let position = Vector3::new(10.0, 10.0, 5.0);
    let velocity = Vector3::new(1.0, 0.0, 0.0);
    let mean_a = Vector3::new(0.0, -0.1, 0.1);
    let sigma_a = Vector3::new(0.5, 1.0, 1.0);
    let dynamics_model = WhiteNoiseAcceleration::new(position, velocity, mean_a, sigma_a);

    let radius = 200.0;
    let origin = Vector3::new(10.0, 10.0, 5.0);
    let sphere = Sphere::new(radius, origin).unwrap();
    let num_particles = 10_000;
    let tau = 0.5;
    let particle_filter = ParticleFilter::new(&sphere, num_particles, tau);

    let sd_transmission_noise = 0.1;
    let sd_ranging_noise = 0.8;

    let swarm_element_1 = SwarmElement::new(
        swarm_name_1,
        dynamics_model,
        particle_filter,
        sd_transmission_noise,
        sd_ranging_noise,
    );

    let swarm_name_2 = String::from("swarm_element_2");

    let position = Vector3::new(5.0, 5.0, 5.0);
    let velocity = Vector3::new(-1.0, 0.0, 0.0);
    let mean_a = Vector3::new(0.0, 0.1, -0.1);
    let sigma_a = Vector3::new(0.5, 1.0, 1.0);
    let dynamics_model = WhiteNoiseAcceleration::new(position, velocity, mean_a, sigma_a);

    let radius = 200.0;
    let origin = Vector3::new(5.0, 5.0, 5.0);
    let sphere = Sphere::new(radius, origin).unwrap();
    let num_particles = 10_000;
    let tau = 0.5;
    let particle_filter = ParticleFilter::new(&sphere, num_particles, tau);

    let sd_transmission_noise = 0.1;
    let sd_ranging_noise = 0.8;

    let swarm_element_2 = SwarmElement::new(
        swarm_name_2,
        dynamics_model,
        particle_filter,
        sd_transmission_noise,
        sd_ranging_noise,
    );

    let anchor_std = 0.4;
    let anchor1 = Anchor::new(Vector3::new(0.0, 0.0, 0.0), anchor_std);
    let anchor2 = Anchor::new(Vector3::new(0.0, 20.0, 0.0), anchor_std);
    let anchor3 = Anchor::new(Vector3::new(20.0, 20.0, 0.0), anchor_std);

    let visualizer = RerunVisualization::new(String::from("ToA-Particle-Filter"))
        .expect("Unable to create rerun visualization");

    let mut sim = Simulation::builder()
        .swarm_elements(vec![swarm_element_1, swarm_element_2])
        .anchors(vec![anchor1, anchor2, anchor3])
        .visualizer(visualizer)
        .build();

    let time_steps = 100;
    let step_size = 0.1;
    sim.run(time_steps, step_size);
}
```


https://github.com/user-attachments/assets/bf0f98f1-eb4b-4300-bcae-3f5351299990

