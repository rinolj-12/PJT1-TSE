package edu.vit.tsedemo;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * The on-device measurements reported in the project report (Section 4.3.5), for one model:
 * model load time, enrollment (embedder) time, time per 3 s streaming window and real-time factor,
 * extraction quality (SI-SDRi) on bundled test mixtures next to the PC result for the same files,
 * and a sliding-window streaming run. Pure Java so it also runs on a desktop JVM.
 */
public final class Benchmark {

    /** One bundled test sample (16-bit WAVs prepared by phone_demo/prepare_phone_assets.py). */
    public static final class Sample {
        public final String id, set;
        public final float[] mix, tgt, enr;
        public final double pcSiSdri;   // ONNX Runtime on the PC, same model and precision, same files

        public Sample(String id, String set, float[] mix, float[] tgt, float[] enr, double pcSiSdri) {
            this.id = id; this.set = set; this.mix = mix; this.tgt = tgt; this.enr = enr; this.pcSiSdri = pcSiSdri;
        }
    }

    public interface Log { void line(String s); }

    public static String run(String modelName, byte[] embedder, byte[] extractor, int threads,
                             List<Sample> samples, Log log) throws Exception {
        StringBuilder json = new StringBuilder();
        Runtime rt = Runtime.getRuntime();
        log.line(String.format(Locale.US, "== %s, %d thread(s)", modelName, threads));
        try (TseEngine eng = new TseEngine(embedder, extractor, threads)) {
            log.line(String.format(Locale.US, "model load: %d ms (%.1f MB)", eng.loadMillis,
                    (embedder.length + extractor.length) / 1048576.0));
            Sample s0 = samples.get(0);
            // enrollment time (embedder on the 6 s enrollment)
            eng.embed(s0.enr);
            double[] te = new double[5];
            float[] emb = null;
            for (int i = 0; i < te.length; i++) {
                long t0 = System.nanoTime();
                emb = eng.embed(s0.enr);
                te[i] = (System.nanoTime() - t0) / 1e6;
            }
            double enrollMs = TseEngine.median(te);
            log.line(String.format(Locale.US, "enrollment (%.1f s): %.1f ms", s0.enr.length / (double) TseEngine.SR, enrollMs));
            // one 3 s streaming window
            float[] win = java.util.Arrays.copyOf(s0.mix, 3 * TseEngine.SR);
            for (int i = 0; i < 2; i++) eng.extract(win, emb);
            double[] tw = new double[10];
            for (int i = 0; i < tw.length; i++) {
                long t0 = System.nanoTime();
                eng.extract(win, emb);
                tw[i] = (System.nanoTime() - t0) / 1e6;
            }
            double winMs = TseEngine.median(tw), winP90 = TseEngine.percentile(tw, 0.9);
            log.line(String.format(Locale.US, "3 s window: median %.1f ms, p90 %.1f ms, RTF %.3f",
                    winMs, winP90, winMs / 3000.0));
            // quality on the bundled test mixtures
            List<String> q = new ArrayList<>();
            double sumPhone = 0, sumPc = 0, maxDev = 0;
            for (Sample s : samples) {
                float[] e = eng.embed(s.enr);
                long t0 = System.nanoTime();
                float[] y = eng.extract(s.mix, e);
                double ms = (System.nanoTime() - t0) / 1e6;
                double sdri = TseEngine.siSdr(y, s.tgt) - TseEngine.siSdr(s.mix, s.tgt);
                sumPhone += sdri;
                sumPc += s.pcSiSdri;
                maxDev = Math.max(maxDev, Math.abs(sdri - s.pcSiSdri));
                log.line(String.format(Locale.US, "  %s (%.1f s): SI-SDRi %.2f dB (PC %.2f) in %.0f ms",
                        s.id, s.mix.length / (double) TseEngine.SR, sdri, s.pcSiSdri, ms));
                q.add(String.format(Locale.US, "{\"id\":\"%s\",\"si_sdri\":%.3f,\"pc\":%.3f,\"ms\":%.1f}", s.id, sdri, s.pcSiSdri, ms));
            }
            int n = samples.size();
            log.line(String.format(Locale.US, "mean SI-SDRi: phone %.2f dB, PC %.2f dB (max |difference| %.3f dB)",
                    sumPhone / n, sumPc / n, maxDev));
            // sliding-window streaming (3 s window) on one 3-speaker sample, two latency settings.
            // Real-time = the model finishes 90% of the hops within the hop duration.
            Sample ss = samples.stream().filter(s -> s.set.equals("3spk")).findFirst().orElse(s0);
            float[] e = eng.embed(ss.enr);
            double offline = TseEngine.siSdr(eng.extract(ss.mix, e), ss.tgt) - TseEngine.siSdr(ss.mix, ss.tgt);
            List<String> st = new ArrayList<>();
            for (double[] cfg : new double[][]{{0.2, 0.2}, {0.5, 0.5}}) {
                TseEngine.StreamResult sr = eng.stream(ss.mix, e, 3.0, cfg[0], cfg[1]);
                double ssdri = TseEngine.siSdr(sr.output, ss.tgt) - TseEngine.siSdr(ss.mix, ss.tgt);
                double hopMed = TseEngine.median(sr.hopMillis), hopMax = TseEngine.percentile(sr.hopMillis, 1.0);
                boolean realtime = TseEngine.percentile(sr.hopMillis, 0.9) < cfg[0] * 1000;
                log.line(String.format(Locale.US, "streaming hop %.0f + look-ahead %.0f ms (latency %.0f ms) on %s: "
                        + "SI-SDRi %.2f dB (offline %.2f), compute per hop median %.1f ms, max %.1f ms -> %s",
                        cfg[0] * 1000, cfg[1] * 1000, (cfg[0] + cfg[1]) * 1000, ss.id, ssdri, offline, hopMed, hopMax,
                        realtime ? "REAL-TIME" : "NOT real-time"));
                st.add(String.format(Locale.US, "{\"hop_ms\":%.0f,\"lookahead_ms\":%.0f,\"si_sdri\":%.3f,"
                        + "\"offline_si_sdri\":%.3f,\"hop_median_ms\":%.2f,\"hop_max_ms\":%.2f,\"realtime\":%s}",
                        cfg[0] * 1000, cfg[1] * 1000, ssdri, offline, hopMed, hopMax, realtime));
            }
            long heap = rt.totalMemory() - rt.freeMemory();
            json.append(String.format(Locale.US,
                    "{\"model\":\"%s\",\"threads\":%d,\"load_ms\":%d,\"model_mb\":%.2f,\"enroll_ms\":%.2f,"
                            + "\"window3s_median_ms\":%.2f,\"window3s_p90_ms\":%.2f,\"rtf\":%.4f,"
                            + "\"mean_si_sdri_phone\":%.3f,\"mean_si_sdri_pc\":%.3f,\"max_abs_diff\":%.3f,"
                            + "\"streaming\":[%s],\"java_heap_mb\":%.1f,\"samples\":[%s]}",
                    modelName, threads, eng.loadMillis, (embedder.length + extractor.length) / 1048576.0, enrollMs,
                    winMs, winP90, winMs / 3000.0, sumPhone / n, sumPc / n, maxDev, String.join(",", st),
                    heap / 1048576.0, String.join(",", q)));
        }
        return json.toString();
    }
}
