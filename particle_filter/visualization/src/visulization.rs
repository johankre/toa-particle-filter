pub struct RerunVisulization {
    current_frame: usize,
}

impl RerunVisulization {
    pub fn new() -> Self {
        Self { current_frame: 0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_rerun_visulization() {
        let visulization = RerunVisulization::new();
        assert_eq!(visulization.current_frame, 0);
    }
}
