# 1) IMPORTA PRIMA IL TUO PACKAGE (aggiunge il jar al classpath)
from slidingheatmap_capymoa import SlidingHeatmapClassifier

# 2) Abilita gli import Java e avvia la JVM tramite CapyMOA
import jpype.imports
import capymoa.env

# 3) SOLO ORA puoi importare evaluation (che fa `from java.util ...`)
from capymoa.datasets import Electricity
from capymoa.evaluation import prequential_evaluation


def main():
    stream = Electricity()

    learner = SlidingHeatmapClassifier(
        schema=stream.get_schema(),
        n_bins=10,
        buffer_window=2000
    )

    res = prequential_evaluation(
        stream=stream,
        learner=learner,
        window_size=500
    )

    print("Cumulative accuracy:", res["cumulative"].accuracy())
    print("Random fallbacks:", learner.random_count)


if __name__ == "__main__":
    main()
