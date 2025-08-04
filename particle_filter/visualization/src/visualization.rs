use rerun::{Points3D, SpawnOptions};
use rerun::archetypes::Clear;
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
        let rec = rerun::RecordingStreamBuilder::new(visulization_name).spawn_opts(
        &SpawnOptions {
            port: 9876,    
            wait_for_bind: true,
            ..Default::default()
        },
        None,
    )?;

        let (tx, rx) = mpsc::sync_channel::<Command>(100);

        let handle = thread::spawn(move || {
            let mut last_particle_count: std::collections::HashMap<String, (usize, usize)> =
                Default::default();
            for cmd in rx {
                match cmd {
                    Command::SetFrame(frame) => rec.set_time_sequence("frame", frame),
                    Command::LogPoints(path, positions, radius, color) => {
                        let mut ent = path.clone();

                        if path.ends_with("/particle_filter") {
                            let particle_count = positions.len();
                            let (particle_filter_id, last_count) =
                                last_particle_count.get(&ent).unwrap_or(&(0, 0));

                            ent = format!("{}/{:03}", ent, particle_filter_id);
                            if particle_count * 2 < *last_count || *last_count == 0 {
                                rec.log(ent.clone(), &Clear::new(true))
                                    .expect("failed to clear retired generation");
                                let particle_filter_id = particle_filter_id + 1;
                                ent = format!("{}/{:03}", path.clone(), particle_filter_id);

                                last_particle_count
                                    .insert(path.clone(), (particle_filter_id, positions.len()));
                            }
                        }

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
                                ent,
                                &Points3D::new(positions_f32)
                                    .with_radii(radii)
                                    .with_colors(col),
                            )
                            .expect("Rerun: unable to log points");
                        } else {
                            rec.log(ent, &Points3D::new(positions_f32).with_radii(radii))
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
