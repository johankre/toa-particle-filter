use rerun::Points3D;
use std::{
    sync::mpsc::{self, SyncSender},
    thread::{self, JoinHandle},
};

pub enum Command {
    SetFrame(i64),
    LogPoints(String, Vec<[f64; 3]>, f64, Option<Vec<[u8; 4]>>),
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
                    Command::LogPoints(path, positions, radius, color) => {
                        let positions_f32: Vec<[f32; 3]> = positions
                            .into_iter()
                            .map(|[x, y, z]| [x as f32, y as f32, z as f32])
                            .collect();

                        let radii = vec![
                            rerun::components::Radius::new_ui_points(radius as f32);
                            positions_f32.len()
                        ];
                        if let Some(col) = color {
                            rec.log(
                                path,
                                &Points3D::new(positions_f32)
                                    .with_radii(radii)
                                    .with_colors(col),
                            )
                            .expect("Rerun: unable to log points");
                        } else {
                            rec.log(path, &Points3D::new(positions_f32).with_radii(radii))
                                .expect("Rerun: unable to log points");
                        }
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
