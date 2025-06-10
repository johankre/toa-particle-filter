use nalgebra::Vector3;
use rerun::{Points3D, RecordingStreamError};
use std::{
    sync::mpsc::{self, SyncSender},
    thread::{self, JoinHandle},
};

use simulation::simulation::Simulation;

enum Command {
    SetFrame(i64),
    LogPoints(String, Vec<[f32; 3]>),
}

pub struct RerunVisualization {
    tx: Option<SyncSender<Command>>,
    handle: Option<JoinHandle<()>>,
    current_frame: i64,
}

impl RerunVisualization {
    pub fn new(visulization_name: String) -> Result<Self, Box<dyn std::error::Error>> {
        let rec = rerun::RecordingStreamBuilder::new(visulization_name).connect_tcp()?;
        let (tx, rx) = mpsc::sync_channel::<Command>(100);

        let handle = thread::spawn(move || {
            for cmd in rx {
                match cmd {
                    Command::SetFrame(frame) => rec.set_time_sequence("frame", frame),
                    Command::LogPoints(path, positions) => rec
                        .log(path, &Points3D::new(positions))
                        .expect("Rerun: unable to log points"),
                }
            }
        });

        Ok(Self {
            tx: Some(tx),
            handle: Some(handle),
            current_frame: 0,
        })
    }

    pub fn capture_frame(mut self, simulation: &Simulation) -> Result<(), RecordingStreamError> {
        self.current_frame += 1;

        self.tx
            .as_ref()
            .unwrap()
            .send(Command::SetFrame(self.current_frame))
            .expect("logging thread has died unexpectedly");

        for swarm in &simulation.swarm_elements {
            let positions: Vec<[f32; 3]> = swarm
                .particle_filter
                .particles
                .iter()
                .map(|p| {
                    let v: Vector3<f32> = p.position;
                    [v.x, v.y, v.z]
                })
                .collect();

            let entity_name = swarm.name.clone() + "/particle_filter";
            self.tx
                .as_ref()
                .unwrap()
                .send(Command::LogPoints(entity_name, positions))
                .expect("logging thread has died unexpectedly");
        }

        Ok(())
    }
}

impl Drop for RerunVisualization {
    fn drop(&mut self) {
        self.tx.take(); // now tx is None, channel is closed

        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}
