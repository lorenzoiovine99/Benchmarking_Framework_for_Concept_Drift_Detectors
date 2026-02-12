from pathlib import Path
import importlib.util

def _ensure_jvm_with_moa_and_custom_jar():
    import jpype  # IMPORTANT: first line to avoid scope issues

    if jpype.isJVMStarted():
        raise RuntimeError(
            "JVM already started. You must import/use slidingheatmap_capymoa BEFORE importing capymoa "
            "in the same Python process."
        )

    custom_jar = Path(__file__).resolve().parent / "jars" / "slidingheatmap-moa-1.0.0.jar"
    if not custom_jar.exists():
        raise FileNotFoundError(f"Missing jar: {custom_jar}")

    spec = importlib.util.find_spec("capymoa")
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError("capymoa not found. Install capymoa first.")
    capy_dir = Path(spec.origin).resolve().parent
    moa_jar = capy_dir / "jar" / "moa.jar"
    if not moa_jar.exists():
        raise FileNotFoundError(f"Missing moa.jar: {moa_jar}")

    jpype.startJVM(classpath=[str(custom_jar), str(moa_jar)])

    import jpype.imports  # noqa: F401

_ensure_jvm_with_moa_and_custom_jar()

from capymoa.base import MOAClassifier
from jpype import JClass

class SlidingHeatmapClassifier(MOAClassifier):
    """Wrapper CapyMOA per il custom SlidingHeatmapClassifier Java."""

    def __init__(self, schema=None, CLI=None, random_seed=1,
                 n_bins=10, buffer_window=10000):
        self.moa_learner = JClass("moa.classifiers.custom.SlidingHeatmapClassifier")()
        if schema is None:
            self.schema = None
            self.CLI = CLI
            self.random_seed = random_seed
        else:
            super().__init__(schema=schema, CLI=CLI,
                             random_seed=random_seed,
                             moa_learner=self.moa_learner)

        # 🔹 salva i parametri come attributi Python
        self.n_bins = n_bins
        self.buffer_window = buffer_window

        if self.CLI is None:
            self.moa_learner.getOptions().setViaCLIString(
                f"-nBins {n_bins} -bufferWindow {buffer_window}"
            )
            self.moa_learner.prepareForUse()
            self.moa_learner.resetLearning()

    @property
    def random_count(self):
        return self.moa_learner.getRandomCount()

    def __str__(self):
        return "SlidingHeatmapClassifier"
