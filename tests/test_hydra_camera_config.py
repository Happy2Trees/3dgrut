from omegaconf import OmegaConf


class _FakeColmapDataset:
    def __init__(
        self,
        path,
        device="cuda",
        split="train",
        downsample_factor=1,
        test_split_interval=8,
        ray_jitter=None,
        camera_override_model: str | None = None,
    ):
        self.path = path
        self.split = split
        self.downsample_factor = downsample_factor
        self.test_split_interval = test_split_interval
        self.camera_override_model = camera_override_model

    # Minimal protocol
    def __len__(self):
        return 0
    def __getitem__(self, idx):
        raise IndexError


def test_hydra_render_camera_section():
    conf = OmegaConf.load("configs/render/3dgut.yaml")
    assert "camera" in conf
    assert "model" in conf.camera
    assert conf.camera.model in ("dataset", "equirectangular")

    # Test overriding to equirectangular
    override = OmegaConf.create({"camera": {"model": "equirectangular"}})
    merged = OmegaConf.merge(conf, override)
    assert merged.camera.model == "equirectangular"


def test_dataset_injects_camera_override(monkeypatch):
    # Load base app config for 3DGUT + colmap
    app = OmegaConf.load("configs/apps/colmap_3dgut.yaml")
    # Override camera model to ERP
    app.render.camera.model = "equirectangular"

    import threedgrut.datasets as dsm

    # Monkeypatch ColmapDataset used in factory
    monkeypatch.setattr(dsm, "ColmapDataset", _FakeColmapDataset)

    train, val = dsm.make("colmap", app, ray_jitter=None)
    assert getattr(train, "camera_override_model", None) == "equirectangular"
    assert getattr(val, "camera_override_model", None) == "equirectangular"
