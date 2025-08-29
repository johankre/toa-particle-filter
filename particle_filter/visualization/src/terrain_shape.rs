pub fn load_contours_centered(
    path: &str,
) -> Result<Vec<Vec<[f32; 3]>>, Box<dyn std::error::Error>> {
    let lines: Vec<shapefile::PolylineZ> = shapefile::read_shapes_as(path)?;
    let contours = lines
        .into_iter()
        .map(|pl| {
            pl.parts()
                .iter()
                .flat_map(|part| part.iter().map(|p| [p.x as f32, p.y as f32, p.z as f32]))
                .collect()
        })
        .collect();
    Ok(to_origin_and_scale(contours))
}

pub fn to_origin_and_scale(mut geoms: Vec<Vec<[f32; 3]>>) -> Vec<Vec<[f32; 3]>> {
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;

    for feature in &geoms {
        for [x, y, _z] in feature {
            if *x < min_x {
                min_x = *x;
            }
            if *y < min_y {
                min_y = *y;
            }
        }
    }

    for feature in &mut geoms {
        for p in feature {
            p[0] = p[0] - min_x;
            p[1] = p[1] - min_y;
        }
    }

    geoms
}
