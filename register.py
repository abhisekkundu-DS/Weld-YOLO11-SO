"""
Registration Script to connect Weld-YOLO11-SO custom modules to installed Ultralytics package.
"""

import pathlib
import ultralytics

ultra_path = pathlib.Path(ultralytics.__file__).parent
print('Ultralytics path:', ultra_path)

mod_init_path = ultra_path / 'nn' / 'modules' / '__init__.py'
tasks_path = ultra_path / 'nn' / 'tasks.py'

# Clean previous registration if any
content_mod = mod_init_path.read_text(encoding='utf-8')
if 'from models.modules.wavelet import WaveletBlock' in content_mod:
    content_mod = content_mod.replace('\n# Custom Weld-YOLO11-SO Modules\nfrom models.modules.wavelet import WaveletBlock\nfrom models.modules.weldsimam import WeldSimAM\nfrom models.modules.dysample import DySample\nfrom models.modules.ahsfpn import AHSFPN\n', '')
    content_mod = content_mod.replace('    "WaveletBlock",\n    "WeldSimAM",\n    "DySample",\n    "AHSFPN",\n', '')

import_stmt = (
    "\n# Custom Weld-YOLO11-SO Modules\n"
    "from models.modules.wavelet import WaveletBlock\n"
    "from models.modules.weldsimam import WeldSimAM\n"
    "from models.modules.dysample import DySample\n"
    "from models.modules.ahsfpn import AHSFPN\n"
)
content_mod += import_stmt
custom_all = '    "WaveletBlock",\n    "WeldSimAM",\n    "DySample",\n    "AHSFPN",\n'
content_mod = content_mod.replace('__all__ = (', '__all__ = (\n' + custom_all)
mod_init_path.write_text(content_mod, encoding='utf-8')
print('Updated ultralytics/nn/modules/__init__.py')

# Clean tasks.py
content_tasks = tasks_path.read_text(encoding='utf-8')
if 'from models.modules.wavelet import WaveletBlock' in content_tasks:
    content_tasks = content_tasks.replace('\n# Custom Weld-YOLO11-SO Modules\nfrom models.modules.wavelet import WaveletBlock\nfrom models.modules.weldsimam import WeldSimAM\nfrom models.modules.dysample import DySample\nfrom models.modules.ahsfpn import AHSFPN\n', '')
    content_tasks = content_tasks.replace('            WaveletBlock,\n            WeldSimAM,\n            DySample,\n            AHSFPN,\n', '')

# Insert after __future__ import or docstring
if 'from __future__ import annotations' in content_tasks:
    content_tasks = content_tasks.replace(
        'from __future__ import annotations',
        'from __future__ import annotations\n' + import_stmt
    )
else:
    content_tasks = import_stmt + content_tasks

content_tasks = content_tasks.replace(
    'base_modules = frozenset(\n        {',
    'base_modules = frozenset(\n        {\n            WaveletBlock,\n            WeldSimAM,\n            DySample,\n            AHSFPN,'
)
tasks_path.write_text(content_tasks, encoding='utf-8')
print('Updated ultralytics/nn/tasks.py')

print('\nTesting imports...')
from ultralytics.nn.modules import WaveletBlock, WeldSimAM, DySample, AHSFPN
print('SUCCESS! Successfully imported WaveletBlock, WeldSimAM, DySample, AHSFPN from ultralytics.nn.modules!')
