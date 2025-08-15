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

##  Example

A minimal example using a single swarm element and two anchors:

```rust
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
    let anchor2 = Anchor::new(Vector3::new(20.0, 20.0, 0.0), anchor_std);

    let visualizer = RerunVisualization::new(String::from("ToA-Particle-Filter"))
        .expect("Unable to create rerun visualization");

    let mut sim = SimulationBuilder::default()
        .swarm_elements(vec![swarm_element])
        .anchors(vec![anchor1, anchor2])
        .visualizer(visualizer)
        .build();

    let time_steps = 30;
    sim.run(time_steps);
}
```
![TOA-Example](https://github.com/user-attachments/assets/974f15fd-e6be-4ef1-b20d-1dc456505120)
