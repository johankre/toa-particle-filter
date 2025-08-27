use rerun::archetypes::Clear;
use rerun::{Points3D, Scalars, SpawnOptions};
use std::collections::HashMap;
use std::{
    sync::mpsc::{self, SyncSender},
    thread::{self, JoinHandle},
};

pub enum Command {
    SetFrame(i64),
    LogPoints(String, Vec<[f64; 3]>, f64, Option<Vec<[u8; 4]>>),
    LogTrajectory(String, [f64; 3], [f64; 3]),
    LogScalarPlot(String, f64),
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

        let (tx, rx) = mpsc::sync_channel::<Command>(1000);

        let handle = thread::spawn(move || {
            let mut last_particle_count: HashMap<String, (usize, usize)> = Default::default();
            let mut traj_points: HashMap<String, Vec<[f32; 3]>> = HashMap::new();
            const MAX_TRAJECTORY_HISTORY: usize = 1000;

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
                    Command::LogTrajectory(path, start_point, end_point) => {
                        let s = [
                            start_point[0] as f32,
                            start_point[1] as f32,
                            start_point[2] as f32,
                        ];
                        let e = [
                            end_point[0] as f32,
                            end_point[1] as f32,
                            end_point[2] as f32,
                        ];

                        let buf = traj_points.entry(path.to_owned()).or_default();

                        if buf.is_empty() {
                            buf.push(s);
                            if s != e {
                                buf.push(e);
                            }
                        } else {
                            // continuity check: new segment must start where the last one ended
                            let last = *buf.last().unwrap();
                            if last != s {
                                println!(
                                    "trajectory discontinuity for '{}': segment start {:?} != previous end {:?}",
                                    path, s, last
                                );
                            }
                            // append END if we have moved from last element in traj_points
                            if last != e {
                                buf.push(e);
                            }
                        }

                        // bounded history length (keeps the last MAX_TRAJECTORY_HISTORY connected)
                        if buf.len() > MAX_TRAJECTORY_HISTORY {
                            let drop_n = buf.len() - MAX_TRAJECTORY_HISTORY;
                            buf.drain(0..drop_n);
                        }
                        let strip = rerun::LineStrips3D::new([buf.clone()])
                            .with_radii([rerun::components::Radius::new_ui_points(0.5)])
                            .with_colors([[200, 100, 100, 255]]);

                        rec.log(path, &strip)
                            .expect("Rerun: unable to log trajectory");
                    }
                    Command::LogScalarPlot(path, value) => {
                        rec.log(path, &Scalars::single(value))
                            .expect("Rerun: unable to log trajectory");
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
        self.tx.take();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}
