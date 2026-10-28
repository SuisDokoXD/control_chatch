from pathlib import Path
import os


PROJECT_DIR = Path(__file__).resolve().parents[1]
YOLO_SETTINGS = PROJECT_DIR / "reports" / "ultralytics"
YOLO_SETTINGS.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_SETTINGS))

from ultralytics import YOLO
DATA_CONFIG = PROJECT_DIR / "dataset" / "yolo_cube" / "data.yaml"
MODEL_PATH = PROJECT_DIR / "权重" / "yolo11n.pt"


def main() -> None:
    if not DATA_CONFIG.exists():
        raise FileNotFoundError(
            f"Missing {DATA_CONFIG}. Prepare the dataset configuration first."
        )
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing pretrained model: {MODEL_PATH}")

    import yaml
    config = yaml.safe_load(DATA_CONFIG.read_text(encoding="utf-8"))
    dataset_root = (DATA_CONFIG.parent / config.get("path", ".")).resolve()
    # Ultralytics resolves relative dataset roots against its global settings.
    # Resolve from this YAML's directory for this run; keep source YAML portable.
    config["path"] = str(dataset_root)
    runtime_config = PROJECT_DIR / "reports" / "training_data_runtime.yaml"
    runtime_config.parent.mkdir(parents=True, exist_ok=True)
    runtime_config.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    model = YOLO(str(MODEL_PATH))
    model.train(
        data=str(runtime_config),
        model=str(MODEL_PATH),
        project=str(PROJECT_DIR / "runs"),
        name="train_cpu",
        device="cpu",
        imgsz=640,
        batch=4,
        workers=0,
        epochs=100,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
