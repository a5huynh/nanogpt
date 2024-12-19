use std::path::PathBuf;

use mnist::*;

fn read_mnist_data() -> Mnist {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("../../data/images/mnist");

    MnistBuilder::new()
        .base_path(path.as_os_str().to_str().unwrap())
        .finalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mnist_loader() {
        read_mnist_data();
    }
}
