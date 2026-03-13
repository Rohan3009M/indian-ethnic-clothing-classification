import sys
from pathlib import Path
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_EXE = sys.executable

MODELS = [
    ("resnet50", True),
    ("mobilenet_v2", False),
    ("efficientnet_b0", True),
    ("densenet121", False),
]


def main():
    for model_name, freeze_backbone in MODELS:
        print("\n" + "=" * 70)
        print(f"Evaluating: {model_name} | freeze_backbone={freeze_backbone}")

        cmd = [
            PYTHON_EXE,
            str(PROJECT_ROOT / "scripts" / "evaluate.py"),
            "--model_name",
            model_name,
        ]

        if freeze_backbone:
            cmd.append("--freeze_backbone")

        result = subprocess.run(cmd, check=True)
        print(f"Finished evaluating {model_name} with exit code {result.returncode}")


if __name__ == "__main__":
    main()