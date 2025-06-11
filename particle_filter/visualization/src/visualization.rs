use rerun::Points3D;
use std::{
    sync::mpsc::{self, SyncSender},
    thread::{self, JoinHandle},
};

pub enum Command {
    SetFrame(i64),
    LogPoints(String, Vec<[f32; 3]>, f32),
}

pub struct RerunVisualization {
    tx: Option<SyncSender<Command>>,
    handle: Option<JoinHandle<()>>,
}

impl RerunVisualization {
    pub fn new(visulization_name: String) -> Result<Self, Box<dyn std::error::Error>> {
        let rec = rerun::RecordingStreamBuilder::new(visulization_name).connect_tcp()?;
        let (tx, rx) = mpsc::sync_channel::<Command>(100);

        let handle = thread::spawn(move || {
            for cmd in rx {
                match cmd {
                    Command::SetFrame(frame) => rec.set_time_sequence("frame", frame),
                    Command::LogPoints(path, positions, radius) => {
                        let radii =
                            vec![rerun::components::Radius::new_ui_points(radius); positions.len()];
                        rec.log(path, &Points3D::new(positions).with_radii(radii))
                            .expect("Rerun: unable to log points");
                    }
                }
            }
        });

        Ok(Self {
            tx: Some(tx),
            handle: Some(handle),
        })
    }

    pub fn log(&mut self, command: Command) {
        self.tx
            .as_ref()
            .unwrap()
            .send(command)
            .expect("logging thread has died unexpectedly");
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
