package edu.vit.tsedemo;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.Collections;

/**
 * On-device target speech extraction with ONNX Runtime (no Android dependencies, so it can be
 * tested on a desktop JVM as well).
 *
 * The model is split into two ONNX graphs (see scripts/edge_benchmark.py):
 *   embedder  : enrollment (1, Te) -> speaker embedding (1, 256)   run once per target speaker
 *   extractor : mixture (1, T) + embedding (1, 256) -> estimate (1, T)
 * Audio is 8 kHz mono float in [-1, 1].
 */
public final class TseEngine implements AutoCloseable {
    public static final int SR = 8000;

    private final OrtEnvironment env = OrtEnvironment.getEnvironment();
    private final OrtSession embedder;
    private final OrtSession extractor;
    public final long loadMillis;
    public final int threads;

    public TseEngine(byte[] embedderModel, byte[] extractorModel, int threads) throws OrtException {
        long t0 = System.nanoTime();
        this.threads = threads;
        OrtSession.SessionOptions so = new OrtSession.SessionOptions();
        so.setIntraOpNumThreads(threads);
        so.setInterOpNumThreads(1);
        so.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        embedder = env.createSession(embedderModel, so);
        extractor = env.createSession(extractorModel, so);
        loadMillis = (System.nanoTime() - t0) / 1_000_000;
    }

    /** Speaker embedding (256 floats) of an enrollment recording. */
    public float[] embed(float[] enrollment) throws OrtException {
        try (OnnxTensor x = tensor(enrollment);
             OrtSession.Result r = embedder.run(Collections.singletonMap("enrollment", x))) {
            return ((float[][]) r.get(0).getValue())[0];
        }
    }

    /** Target estimate for a mixture (same length; arbitrary output level, see {@link #matchLevel}). */
    public float[] extract(float[] mixture, float[] embedding) throws OrtException {
        try (OnnxTensor x = tensor(mixture); OnnxTensor e = tensor(embedding)) {
            java.util.HashMap<String, OnnxTensor> in = new java.util.HashMap<>();  // Map.of needs Android 11+
            in.put("mixture", x);
            in.put("embedding", e);
            try (OrtSession.Result r = extractor.run(in)) {
                return ((float[][]) r.get(0).getValue())[0];
            }
        }
    }

    private OnnxTensor tensor(float[] a) throws OrtException {
        return OnnxTensor.createTensor(env, FloatBuffer.wrap(a), new long[]{1, a.length});
    }

    @Override
    public void close() throws OrtException {
        embedder.close();
        extractor.close();
    }

    // ------------------------------------------------------------------ streaming

    /** Result of a sliding-window streaming run. */
    public static final class StreamResult {
        public float[] output;
        public double[] hopMillis;   // model compute time of each hop
        public int calls;
    }

    /**
     * Sliding-window streaming, as in src/tse/serving/streaming.py (ChunkedStream): every hop the
     * last {@code windowS} seconds of input are processed, each window's estimate is rescaled by its
     * least-squares gain onto the input window, and the hop ending {@code lookaheadS} before the
     * newest sample is emitted, with a 20 ms crossfade at the seams. Returns the output aligned
     * with the input. Algorithmic latency = hop + look-ahead.
     */
    public StreamResult stream(float[] x, float[] embedding, double windowS, double hopS, double lookaheadS)
            throws OrtException {
        int hop = (int) Math.round(hopS * SR), la = (int) Math.round(lookaheadS * SR);
        int win = Math.max((int) Math.round(windowS * SR), hop + la);
        int xf = Math.min((int) Math.round(0.02 * SR), la);
        int minContext = (int) (0.5 * SR);
        int n = x.length;
        int total = n + la + hop;                          // flush with silence
        float[] in = Arrays.copyOf(x, total + hop);
        float[] out = new float[total + hop];
        float[] tail = new float[0];
        java.util.ArrayList<Double> times = new java.util.ArrayList<>();
        int produced = 0;
        for (int end = hop; end <= total; end += hop) {    // `end` input samples have arrived
            int start = Math.max(0, end - win);
            float[] hist = Arrays.copyOfRange(in, start, end);
            float[] seg = new float[hop];
            if (hist.length >= Math.max(minContext, hop + la)) {
                long t0 = System.nanoTime();
                float[] est = extract(hist, embedding);
                times.add((System.nanoTime() - t0) / 1e6);
                double ee = 0, ex = 0;
                for (int i = 0; i < est.length; i++) { ee += est[i] * est[i]; ex += est[i] * hist[i]; }
                double g = ee > 1e-12 ? ex / ee : 0.0;
                int e = hist.length - la;
                for (int i = 0; i < hop; i++) seg[i] = (float) (est[e - hop + i] * g);
                if (xf > 0 && tail.length == xf) {
                    for (int i = 0; i < xf; i++) {
                        float f = xf == 1 ? 1f : (float) i / (xf - 1);
                        seg[i] = tail[i] * (1 - f) + seg[i] * f;
                    }
                }
                tail = new float[xf];
                for (int i = 0; i < xf; i++) tail[i] = (float) (est[e + i] * g);
            } else {
                tail = new float[0];
            }
            System.arraycopy(seg, 0, out, produced, hop);
            produced += hop;
        }
        StreamResult r = new StreamResult();
        r.output = Arrays.copyOfRange(out, la, la + n);
        r.hopMillis = times.stream().mapToDouble(Double::doubleValue).toArray();
        r.calls = times.size();
        return r;
    }

    // ------------------------------------------------------------------ metrics and audio helpers

    /** Scale-invariant SDR in dB (zero-mean), same formula as the training loss. */
    public static double siSdr(float[] est, float[] ref) {
        int n = Math.min(est.length, ref.length);
        double me = 0, mr = 0;
        for (int i = 0; i < n; i++) { me += est[i]; mr += ref[i]; }
        me /= n; mr /= n;
        double dot = 0, rr = 0;
        for (int i = 0; i < n; i++) { double r = ref[i] - mr; dot += (est[i] - me) * r; rr += r * r; }
        double a = dot / (rr + 1e-8), ps = 0, pn = 0;
        for (int i = 0; i < n; i++) {
            double p = a * (ref[i] - mr), e = (est[i] - me) - p;
            ps += p * p; pn += e * e;
        }
        return 10 * Math.log10((ps + 1e-8) / (pn + 1e-8));
    }

    /** Rescale an estimate by its least-squares gain onto the mixture (the target's level in the mixture). */
    public static float[] matchLevel(float[] est, float[] mix) {
        double ee = 0, ex = 0;
        for (int i = 0; i < Math.min(est.length, mix.length); i++) { ee += est[i] * est[i]; ex += est[i] * mix[i]; }
        double g = ee > 1e-12 ? ex / ee : 0;
        float[] y = new float[est.length];
        float peak = 0;
        for (int i = 0; i < y.length; i++) { y[i] = (float) (est[i] * g); peak = Math.max(peak, Math.abs(y[i])); }
        if (peak > 0.99f) for (int i = 0; i < y.length; i++) y[i] *= 0.99f / peak;
        return y;
    }

    /** Parse a 16-bit PCM mono WAV file into floats. */
    public static float[] readWav16(byte[] wav) {
        ByteBuffer b = ByteBuffer.wrap(wav).order(ByteOrder.LITTLE_ENDIAN);
        int pos = 12;
        while (pos + 8 <= wav.length) {
            String id = new String(wav, pos, 4, java.nio.charset.StandardCharsets.US_ASCII);
            int size = b.getInt(pos + 4);
            if (id.equals("data")) {
                int ns = size / 2;
                float[] out = new float[ns];
                for (int i = 0; i < ns; i++) out[i] = b.getShort(pos + 8 + 2 * i) / 32768f;
                return out;
            }
            pos += 8 + size + (size & 1);
        }
        throw new IllegalArgumentException("no data chunk in WAV");
    }

    public static double median(double[] v) {
        double[] s = v.clone();
        Arrays.sort(s);
        int n = s.length;
        return n == 0 ? Double.NaN : (n % 2 == 1 ? s[n / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]));
    }

    public static double percentile(double[] v, double p) {
        double[] s = v.clone();
        Arrays.sort(s);
        return s.length == 0 ? Double.NaN : s[(int) Math.min(s.length - 1, Math.floor(p * (s.length - 1)))];
    }
}
