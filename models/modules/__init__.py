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

    print("Successfully registered Weld-YOLO11-SO custom modules with Ultralytics framework.")
