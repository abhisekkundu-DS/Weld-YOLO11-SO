"""
Weld-YOLO11-SO Custom Modules Package
Exports custom components and dynamic registration helper for Ultralytics YOLO framework.
"""

from .wavelet import WaveletBlock
from .weldsimam import WeldSimAM
from .dysample import DySample
from .ahsfpn import AHSFPN
from .ennwd_loss import EnNWDLoss, bbox_ennwd

__all__ = ["WaveletBlock", "WeldSimAM", "DySample", "AHSFPN", "EnNWDLoss", "bbox_ennwd", "register_custom_modules"]


def register_custom_modules():
    """
    Registers custom Weld-YOLO11-SO modules with the installed Ultralytics package
    so that YOLO("models/weld_yolo11.yaml") can dynamically parse and build the model.
    """
    try:
        import ultralytics.nn.tasks as tasks
        import ultralytics.nn.modules as modules
    except ModuleNotFoundError:
        import sys
        import subprocess
        print("Ultralytics package not detected. Installing dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "PyWavelets", "opencv-python"])
        import ultralytics.nn.tasks as tasks
        import ultralytics.nn.modules as modules

    custom_classes = {
        "WaveletBlock": WaveletBlock,
        "WeldSimAM": WeldSimAM,
        "DySample": DySample,
        "AHSFPN": AHSFPN,
    }

    # 1. Expose custom modules in ultralytics.nn.tasks and ultralytics.nn.modules
    for name, cls in custom_classes.items():
        setattr(tasks, name, cls)
        setattr(modules, name, cls)

    # 2. Update __all__ in ultralytics.nn.modules
    if hasattr(modules, "__all__"):
        for name in custom_classes:
            if name not in modules.__all__:
                if isinstance(modules.__all__, tuple):
                    modules.__all__ = list(modules.__all__)
                modules.__all__.append(name)

    # 3. Patch installed ultralytics/nn/tasks.py so parse_model treats custom modules as base_modules
    import pathlib
    import importlib
    import ultralytics
    ultra_path = pathlib.Path(ultralytics.__file__).parent
    tasks_path = ultra_path / "nn" / "tasks.py"
    
    if tasks_path.exists():
        content_tasks = tasks_path.read_text(encoding="utf-8")
        if "WaveletBlock" not in content_tasks:
            import_stmt = (
                "\n# Custom Weld-YOLO11-SO Modules\n"
                "from models.modules.wavelet import WaveletBlock\n"
                "from models.modules.weldsimam import WeldSimAM\n"
                "from models.modules.dysample import DySample\n"
                "from models.modules.ahsfpn import AHSFPN\n"
            )
            if "from __future__ import annotations" in content_tasks:
                content_tasks = content_tasks.replace(
                    "from __future__ import annotations",
                    "from __future__ import annotations\n" + import_stmt
                )
            else:
                content_tasks = import_stmt + content_tasks

            content_tasks = content_tasks.replace(
                "base_modules = frozenset(\n        {",
                "base_modules = frozenset(\n        {\n            WaveletBlock,\n            WeldSimAM,\n            DySample,\n            AHSFPN,"
            )
            tasks_path.write_text(content_tasks, encoding="utf-8")
            import importlib
            importlib.reload(tasks)

    print("Successfully registered and patched Weld-YOLO11-SO custom modules with Ultralytics framework.")
