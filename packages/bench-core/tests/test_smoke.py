import bench_core


def test_package_importable() -> None:
    assert bench_core.__version__ == "0.1.0"
