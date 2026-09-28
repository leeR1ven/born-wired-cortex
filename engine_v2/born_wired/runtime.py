"""One rebuild path for the live MuJoCo window's brain and sensor adapters.

The window rebuilds its brain when the user presses H or edits live_config.json.
A rebuild instantiates controllers out of the package modules, so every module a
controller or a sensor adapter is built from has to be reloaded in the same pass.
The list was once maintained by hand and went stale: embodied.py was reloaded
while the running process still held the older auditory_neurons, and the window
stopped mid-run on a missing key instead of at the rebuild.

RELOAD_ORDER is a topological order of the module-level born_wired imports: a
module is reloaded only after every born_wired module it reads at import time,
because reloading re-runs those import statements. unlisted_modules() compares
the list against the installed package, so a newly added module cannot be left
out silently.
"""

import ast
import importlib
import pkgutil
from pathlib import Path

import born_wired


# Reloaded, in dependency order, before a new controller is built.
RELOAD_ORDER = ('torch_execution', 'synapses', 'regulation', 'adaptive', 'topology', 'sparse', 'encoding',
                'dynamics', 'distributed', 'innate', 'feature_routed',
                'auditory_neurons', 'reflex_controller', 'embodied',
                'reflex_senses', 'stereo_senses', 'binaural_senses')

# Deliberately fixed for the life of the process:
#   go2_body  owns the live MjData the viewer is attached to and is created once
#             at startup; reloading it would pair the running body with a second
#             class object. Body or physics edits need a restart.
#   runtime   this reloader itself; reloading it mid-pass would rebind the names
#             the pass is still using.
NOT_RELOADED = ('go2_body', 'runtime')


def package_modules():
    """Names of the importable modules of this package, excluding __init__."""
    return {info.name for info in pkgutil.iter_modules(born_wired.__path__)}


def unlisted_modules():
    """Package modules that are neither reloaded nor documented as fixed."""
    return sorted(package_modules() - set(RELOAD_ORDER) - set(NOT_RELOADED))


def _module_level_imports(tree):
    """Names imported at module level with a relative born_wired import."""
    found, pending = set(), list(tree.body)
    while pending:
        node = pending.pop()
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
            found.add(node.module.split('.')[0])
        elif isinstance(node, ast.Try):
            pending.extend(node.body)
    return found


def module_dependencies(name):
    """born_wired submodules that name imports at module level."""
    source = (Path(born_wired.__file__).parent / f'{name}.py').read_text(encoding='utf-8')
    return _module_level_imports(ast.parse(source))


def reload_runtime_modules():
    """Reload every runtime module in dependency order; return name -> module.

    The completeness and ordering checks run before anything is reloaded, so a
    rejected list leaves the loaded package untouched.
    """
    unlisted = unlisted_modules()
    if unlisted:
        raise RuntimeError(f'born_wired modules missing from RELOAD_ORDER: {unlisted}')
    order = list(RELOAD_ORDER)
    for name in order:
        for dependency in module_dependencies(name):
            if dependency in order and order.index(dependency) > order.index(name):
                raise RuntimeError(f'{name} is reloaded before its import {dependency}')
    modules = [importlib.import_module(f'born_wired.{name}') for name in order]
    for module in modules:
        importlib.reload(module)
    return {module.__name__.rsplit('.', 1)[-1]: module for module in modules}
