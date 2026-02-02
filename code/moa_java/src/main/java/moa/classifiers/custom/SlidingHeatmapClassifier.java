package moa.classifiers.custom;

import moa.classifiers.AbstractClassifier;
import moa.core.Measurement;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import com.github.javacliparser.Option;
import com.github.javacliparser.IntOption;
import java.util.*;
import java.io.Serializable;

/**
 * SlidingHeatmapClassifier - semplice learner non adattivo basato su binning e majority vote.
 * Implementazione base per benchmark concept drift detector.
 * Include normalizzazione automatica online delle feature.
 */
public class SlidingHeatmapClassifier extends AbstractClassifier implements Serializable {

    private static final long serialVersionUID = 1L;

    // === Opzioni CLI ===
    public IntOption nBinsOption = new IntOption("nBins", 'n', "Number of bins.", 10, 1, Integer.MAX_VALUE);
    public IntOption bufferWindowOption = new IntOption("bufferWindow", 'b', "Size of buffer window.", 10000, 1, Integer.MAX_VALUE);

    // === Variabili interne ===
    private int dims = 0;
    private int nBins;
    private int bufferWindow;

    private Deque<Pair> buffer;
    private Map<String, Counter<Integer>> counts;
    private Random rng;

    private int randomCount = 0;

    // === Normalizzazione ===
    private double[] minVals;
    private double[] maxVals;
    private boolean initialized = false;
    private static final double EPS = 1e-9;

    // === Classi interne ===
    private static class Pair {
        String key;
        int label;
        Pair(String k, int l) { key = k; label = l; }
    }

    private static class Counter<T> extends HashMap<T, Integer> {
        void inc(T key) { put(key, getOrDefault(key, 0) + 1); }
        void dec(T key) {
            int val = getOrDefault(key, 0) - 1;
            if (val <= 0) remove(key);
            else put(key, val);
        }
    }

    @Override
    public String getPurposeString() {
        return "SlidingHeatmapClassifier: learner basato su binning e majority vote con normalizzazione online.";
    }

    @Override
    public void setModelContext(InstancesHeader context) {
        super.setModelContext(context);
        this.dims = context.numAttributes() - 1;
    }

    @Override
    public void resetLearningImpl() {
        this.nBins = nBinsOption.getValue();
        this.bufferWindow = bufferWindowOption.getValue();

        this.buffer = new ArrayDeque<>(bufferWindow);
        this.counts = new HashMap<>();
        this.rng = new Random(1);

        this.minVals = null;
        this.maxVals = null;
        this.initialized = false;
    }

    // === Normalizzazione online ===
    private double[] normalize(double[] x) {
        if (!initialized) {
            minVals = Arrays.copyOf(x, dims);
            maxVals = Arrays.copyOf(x, dims);
            initialized = true;
        }
        for (int d = 0; d < dims; d++) {
            if (x[d] < minVals[d]) minVals[d] = x[d];
            if (x[d] > maxVals[d]) maxVals[d] = x[d];
        }
        double[] norm = new double[dims];
        for (int d = 0; d < dims; d++) {
            double range = maxVals[d] - minVals[d];
            if (range < EPS) {
                norm[d] = 0.5; // se range ~0, metti a centro [0,1]
            } else {
                norm[d] = (x[d] - minVals[d]) / range;
                if (norm[d] < 0.0) norm[d] = 0.0;
                if (norm[d] > 1.0) norm[d] = 1.0;
            }
        }
        return norm;
    }

    private String pointToKey(double[] x) {
        int[] idx = new int[dims];
        for (int d = 0; d < dims; d++) {
            int bin = (int) (x[d] * nBins);
            if (bin >= nBins) bin = nBins - 1;
            if (bin < 0) bin = 0;
            idx[d] = bin;
        }
        return Arrays.toString(idx);
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        double[] x = inst.toDoubleArray();
        double[] normX = normalize(x);
        String key = pointToKey(normX);
        int label = (int) inst.classValue();

        buffer.add(new Pair(key, label));
        counts.computeIfAbsent(key, k -> new Counter<>()).inc(label);

        if (buffer.size() > bufferWindow) {
            Pair old = buffer.removeFirst();
            Counter<Integer> ctr = counts.get(old.key);
            ctr.dec(old.label);
            if (ctr.isEmpty()) counts.remove(old.key);
        }
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        double[] votes = new double[getModelContext().classAttribute().numValues()];
        double[] x = inst.toDoubleArray();
        double[] normX = normalize(x);
        String key = pointToKey(normX);
        Counter<Integer> ctr = counts.get(key);

        if (ctr != null && !ctr.isEmpty()) {
            for (Map.Entry<Integer, Integer> e : ctr.entrySet()) {
                votes[e.getKey()] = e.getValue();
            }
            return votes;
        }

        // fallback sui vicini ±1 in alcune dimensioni
        int[] keyArr = Arrays.stream(key.replaceAll("[\\[\\]\\s]", "").split(","))
                             .mapToInt(Integer::parseInt).toArray();

        Map<Integer, Integer> agg = new HashMap<>();
        for (int j = 0; j < dims; j++) {
            for (int step : new int[]{-1, 1}) {
                int[] nbr = Arrays.copyOf(keyArr, dims);
                nbr[j] += step;
                if (nbr[j] < 0 || nbr[j] >= nBins) continue;
                String nbrKey = Arrays.toString(nbr);
                Counter<Integer> nbrCtr = counts.get(nbrKey);
                if (nbrCtr != null) {
                    for (Map.Entry<Integer, Integer> e : nbrCtr.entrySet()) {
                        agg.put(e.getKey(), agg.getOrDefault(e.getKey(), 0) + e.getValue());
                    }
                }
            }
        }

        if (!agg.isEmpty()) {
            int bestLabel = Collections.max(agg.entrySet(), Map.Entry.comparingByValue()).getKey();
            votes[bestLabel] = 1.0;
            return votes;
        }

        // fallback random
        randomCount++;
        votes[rng.nextInt(votes.length)] = 1.0;
        return votes;
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return null;
    }

    public int getRandomCount() {
        return randomCount;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
        out.append("SlidingHeatmapClassifier (custom learner con normalizzazione online)");
    }

    @Override
    public boolean isRandomizable() {
        return true;
    }
}