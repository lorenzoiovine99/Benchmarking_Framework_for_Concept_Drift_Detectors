from pathlib import Path
import jpype

def _add_custom_jar_to_classpath():
    jar_path = Path(__file__).resolve().parent / "jars" / "slidingheatmap-moa-1.0.0.jar"
    if not jar_path.exists():
        raise FileNotFoundError(f"Missing jar: {jar_path}")

    # IMPORTANT: must happen before the JVM starts
    if not jpype.isJVMStarted():
        jpype.addClassPath(str(jar_path))
    else:
        raise RuntimeError(
            "JVM already started before adding the custom jar. "
            "Import slidingheatmap_capymoa before importing capymoa."
        )

_add_custom_jar_to_classpath()

from capymoa.base import MOAClassifier
from jpype import JClass

class SlidingHeatmapClassifier(MOAClassifier):
    """Wrapper CapyMOA per il custom SlidingHeatmapClassifier Java."""

    def __init__(self, schema=None, CLI=None, random_seed=1,
                 n_bins=10, buffer_window=10000):
        self.moa_learner = JClass("moa.classifiers.custom.SlidingHeatmapClassifier")()
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
