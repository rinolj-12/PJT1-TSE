package edu.vit.tsedemo;

import android.Manifest;
import android.app.ActivityManager;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.ColorStateList;
import android.graphics.Typeface;
import android.media.AudioAttributes;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.AudioTrack;
import android.media.MediaRecorder;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;

import com.google.android.material.bottomnavigation.BottomNavigationView;
import com.google.android.material.button.MaterialButton;
import com.google.android.material.button.MaterialButtonToggleGroup;
import com.google.android.material.card.MaterialCardView;
import com.google.android.material.progressindicator.LinearProgressIndicator;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * TSE Edge: target speech extraction running entirely on the phone with ONNX Runtime.
 * The INT8 models and test audio are packaged in the APK; the app has no INTERNET permission.
 *
 * Start with the extra "autotest" (adb shell am start -n edu.vit.tsedemo/.MainActivity --ez autotest true)
 * to run a self-test; results go to logcat (tag TSEEdge).
 */
public class MainActivity extends AppCompatActivity {
    private static final String TAG = "TSEEdge";
    private static final int SR = TseEngine.SR;

    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private final Handler ui = new Handler(Looper.getMainLooper());
    private final Map<String, TseEngine> engines = new HashMap<>();
    private final Map<String, List<Benchmark.Sample>> sampleCache = new HashMap<>();
    private final StringBuilder resultsJson = new StringBuilder();
    private final List<Button> actionButtons = new ArrayList<>();

    private TextView status, log, mineResult, deviceInfo, modelCaption;
    private LinearProgressIndicator progress;
    private LinearLayout benchResults;
    private WaveformView waveMix, waveOut, waveRef;
    private Step step1, step2, step3;
    private String model = "r3";
    private int threads = 4;
    private volatile boolean busy;
    private float[] demoMix, demoTgt, demoOut, myEnroll, myMix, myOut;
    private Runnable pendingAfterPermission;
    private AudioTrack track;
    private WaveformView playingWave;

    /** One card of the "My voice" flow. */
    private static final class Step {
        TextView num, title, caption;
        WaveformView wave;
        MaterialButton action, play;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        boolean night = (getResources().getConfiguration().uiMode
                & android.content.res.Configuration.UI_MODE_NIGHT_MASK) == android.content.res.Configuration.UI_MODE_NIGHT_YES;
        if (!night && Build.VERSION.SDK_INT >= 26) {  // dark system icons on the light bars
            getWindow().getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR
                    | View.SYSTEM_UI_FLAG_LIGHT_NAVIGATION_BAR);
        }
        status = findViewById(R.id.status);
        log = findViewById(R.id.log);
        mineResult = findViewById(R.id.mineResult);
        deviceInfo = findViewById(R.id.deviceInfo);
        modelCaption = findViewById(R.id.modelCaption);
        progress = findViewById(R.id.progress);
        benchResults = findViewById(R.id.benchResults);
        waveMix = findViewById(R.id.waveDemoMix);
        waveOut = findViewById(R.id.waveDemoOut);
        waveRef = findViewById(R.id.waveDemoRef);
        waveMix.setColor(color(R.color.waveMix));
        waveOut.setColor(color(R.color.accent));
        waveRef.setColor(color(R.color.waveRef));

        // header selectors
        ((MaterialButtonToggleGroup) findViewById(R.id.modelGroup)).addOnButtonCheckedListener((g, id, checked) -> {
            if (checked) { model = id == R.id.modelR6 ? "r6" : "r3"; updateCaption(); }
        });
        ((MaterialButtonToggleGroup) findViewById(R.id.threadGroup)).addOnButtonCheckedListener((g, id, checked) -> {
            if (checked) { threads = id == R.id.threads1 ? 1 : id == R.id.threads2 ? 2 : 4; updateCaption(); }
        });
        updateCaption();

        // tabs
        BottomNavigationView nav = findViewById(R.id.nav);
        nav.setOnItemSelectedListener(item -> {
            int id = item.getItemId();
            findViewById(R.id.pageDemo).setVisibility(id == R.id.tab_demo ? View.VISIBLE : View.GONE);
            findViewById(R.id.pageMine).setVisibility(id == R.id.tab_mine ? View.VISIBLE : View.GONE);
            findViewById(R.id.pageBench).setVisibility(id == R.id.tab_bench ? View.VISIBLE : View.GONE);
            return true;
        });

        // demo
        action(R.id.demoRun, this::demoExtract);
        click(R.id.demoPlayMix, () -> play(demoMix, waveMix, "mixture"));
        click(R.id.demoPlayOut, () -> play(demoOut, waveOut, "extracted voice"));
        click(R.id.demoPlayRef, () -> play(demoTgt, waveRef, "reference"));
        tile(R.id.tileQuality, "-", "quality (SI-SDRi)");
        tile(R.id.tileTime, "-", "processing time");
        tile(R.id.tileRtf, "-", "real-time factor");

        // my voice
        step1 = step(R.id.step1, "1", "Record your voice", "6 seconds, you alone", "Record", R.drawable.ic_mic);
        step2 = step(R.id.step2, "2", "Record the conversation", "8 seconds, you + others talking", "Record", R.drawable.ic_people);
        step3 = step(R.id.step3, "3", "Extract your voice", "Only your voice remains", "Extract", R.drawable.ic_sparkle);
        step1.wave.setColor(color(R.color.accent));
        step2.wave.setColor(color(R.color.waveMix));
        step3.wave.setColor(color(R.color.good));
        actionButtons.add(step1.action);
        actionButtons.add(step2.action);
        actionButtons.add(step3.action);
        step1.action.setOnClickListener(v -> withMic(() -> record(6, true)));
        step2.action.setOnClickListener(v -> withMic(() -> record(8, false)));
        step3.action.setOnClickListener(v -> extractMine());
        step1.play.setOnClickListener(v -> play(myEnroll, step1.wave, "your voice"));
        step2.play.setOnClickListener(v -> play(myMix, step2.wave, "conversation"));
        step3.play.setOnClickListener(v -> play(myOut, step3.wave, "your extracted voice"));

        // benchmark
        action(R.id.benchSelected, () -> benchmark(new String[]{model}, new int[]{threads}));
        action(R.id.benchFull, () -> benchmark(new String[]{"r3", "r6"}, new int[]{1, 4}));
        click(R.id.share, this::share);
        click(R.id.toggleLog, () -> {
            boolean show = log.getVisibility() != View.VISIBLE;
            log.setVisibility(show ? View.VISIBLE : View.GONE);
            ((Button) findViewById(R.id.toggleLog)).setText(show ? "Hide log" : "Show log");
        });

        deviceInfo.setText(deviceInfo());
        logLine(deviceInfo());
        setStatus("Ready - everything runs on this phone, even in airplane mode.");

        // show the demo mixture right away
        worker.execute(() -> {
            try {
                Benchmark.Sample s = demoSample("r3");
                demoMix = s.mix;
                demoTgt = s.tgt;
                waveMix.setAudio(s.mix);
                waveRef.setAudio(s.tgt);
            } catch (Exception e) { Log.e(TAG, "demo preload", e); }
        });

        if (getIntent().getBooleanExtra("autotest", false)) autotest();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        stopPlayback();
        worker.execute(() -> {
            for (TseEngine e : engines.values()) {
                try { e.close(); } catch (Exception ignored) { }
            }
        });
        worker.shutdown();
    }

    // ------------------------------------------------------------------ UI plumbing
    private interface Job { void run() throws Exception; }

    private int color(int res) { return ContextCompat.getColor(this, res); }

    private int dp(float v) { return (int) TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v, getResources().getDisplayMetrics()); }

    private void click(int id, Runnable r) { findViewById(id).setOnClickListener(v -> r.run()); }

    private void action(int id, Runnable r) {
        Button b = findViewById(id);
        actionButtons.add(b);
        b.setOnClickListener(v -> r.run());
    }

    private void tile(int id, String value, String label) {
        View t = findViewById(id);
        ((TextView) t.findViewById(R.id.tileValue)).setText(value);
        ((TextView) t.findViewById(R.id.tileLabel)).setText(label);
    }

    private Step step(int id, String n, String title, String caption, String action, int icon) {
        View v = findViewById(id);
        Step s = new Step();
        s.num = v.findViewById(R.id.stepNum);
        s.title = v.findViewById(R.id.stepTitle);
        s.caption = v.findViewById(R.id.stepCaption);
        s.wave = v.findViewById(R.id.stepWave);
        s.action = v.findViewById(R.id.stepAction);
        s.play = v.findViewById(R.id.stepPlay);
        s.num.setText(n);
        s.title.setText(title);
        s.caption.setText(caption);
        s.action.setText(action);
        s.action.setIconResource(icon);
        return s;
    }

    private void stepDone(Step s, boolean done) {
        runOnUiThread(() -> {
            s.num.setBackgroundResource(done ? R.drawable.bg_step_done : R.drawable.bg_step);
            s.num.setTextColor(done ? color(R.color.surface) : color(R.color.accent));
            if (done) s.num.setText("✓");
        });
    }

    private void updateCaption() {
        modelCaption.setText((model.equals("r3") ? "R=3 · 8.6 MB · faster" : "R=6 · 13.6 MB · more accurate")
                + "   |   " + threads + " CPU thread" + (threads > 1 ? "s" : ""));
    }

    private void setStatus(String s) {
        runOnUiThread(() -> status.setText(s));
        Log.i(TAG, "status: " + s);
    }

    private void logLine(String s) {
        Log.i(TAG, s);
        runOnUiThread(() -> log.append(s + "\n"));
    }

    private void setBusy(boolean b) {
        busy = b;
        runOnUiThread(() -> {
            progress.setVisibility(b ? View.VISIBLE : View.INVISIBLE);
            for (Button btn : actionButtons) btn.setEnabled(!b);
            if (b) getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
            else getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        });
    }

    /** Run a job on the worker thread with progress, error dialog and busy-locking. */
    private void runJob(String what, Job job) {
        if (busy) { setStatus("Please wait - still working"); return; }
        setBusy(true);
        setStatus(what + " ...");
        worker.execute(() -> {
            try {
                job.run();
            } catch (Throwable t) {
                StringWriter sw = new StringWriter();
                t.printStackTrace(new PrintWriter(sw));
                String trace = sw.toString();
                Log.e(TAG, "job failed: " + what, t);
                logLine("ERROR in '" + what + "': " + t);
                setStatus("Error: " + t.getMessage());
                runOnUiThread(() -> new AlertDialog.Builder(this)
                        .setTitle("Something went wrong")
                        .setMessage(what + "\n\n" + t + "\n\n" + trace.substring(0, Math.min(1500, trace.length())))
                        .setPositiveButton("OK", null).show());
            } finally {
                setBusy(false);
            }
        });
    }

    // ------------------------------------------------------------------ models and assets
    private byte[] asset(String path) throws Exception {
        try (InputStream in = getAssets().open(path)) {
            ByteArrayOutputStream bo = new ByteArrayOutputStream(1 << 20);
            byte[] buf = new byte[1 << 16];
            int n;
            while ((n = in.read(buf)) > 0) bo.write(buf, 0, n);
            return bo.toByteArray();
        }
    }

    /** Engine for (model, threads), created once and reused (worker thread only). */
    private TseEngine engine(String m, int th) throws Exception {
        String key = m + "/" + th;
        TseEngine e = engines.get(key);
        if (e == null) {
            setStatus("Loading the " + (m.equals("r6") ? "accurate" : "compact") + " model ...");
            e = new TseEngine(asset("models/" + m + "/embedder_int8.onnx"), asset("models/" + m + "/extractor_int8.onnx"), th);
            engines.put(key, e);
            logLine(String.format(Locale.US, "Loaded %s INT8 with %d thread(s) in %d ms", m.toUpperCase(), th, e.loadMillis));
        }
        return e;
    }

    private List<Benchmark.Sample> samples(String m) throws Exception {
        List<Benchmark.Sample> cached = sampleCache.get(m);
        if (cached != null) return cached;
        List<Benchmark.Sample> out = new ArrayList<>();
        String[] lines = new String(asset("samples.tsv"), StandardCharsets.UTF_8).trim().split("\n");
        int col = java.util.Arrays.asList(lines[0].trim().split("\t")).indexOf("pc_" + m + "_int8");
        for (int i = 1; i < lines.length; i++) {
            String[] f = lines[i].trim().split("\t");
            out.add(new Benchmark.Sample(f[0], f[1],
                    TseEngine.readWav16(asset("samples/" + f[0] + "_mix.wav")),
                    TseEngine.readWav16(asset("samples/" + f[0] + "_tgt.wav")),
                    TseEngine.readWav16(asset("samples/" + f[0] + "_enr.wav")),
                    Double.parseDouble(f[col])));
        }
        sampleCache.put(m, out);
        return out;
    }

    private Benchmark.Sample demoSample(String m) throws Exception {
        for (Benchmark.Sample x : samples(m)) if (x.id.equals("3spk_3")) return x;
        return samples(m).get(0);
    }

    private String deviceInfo() {
        ActivityManager.MemoryInfo mi = new ActivityManager.MemoryInfo();
        ((ActivityManager) getSystemService(Context.ACTIVITY_SERVICE)).getMemoryInfo(mi);
        String soc = Build.VERSION.SDK_INT >= 31 ? Build.SOC_MANUFACTURER + " " + Build.SOC_MODEL : Build.HARDWARE;
        return String.format(Locale.US, "%s %s | SoC %s | Android %s (API %d) | %d cores | RAM %.1f GB | ABI %s",
                Build.MANUFACTURER, Build.MODEL, soc, Build.VERSION.RELEASE, Build.VERSION.SDK_INT,
                Runtime.getRuntime().availableProcessors(), mi.totalMem / 1073741824.0, Build.SUPPORTED_ABIS[0]);
    }

    // ------------------------------------------------------------------ demo
    private void demoExtract() {
        final String m = model;
        final int th = threads;
        stopPlayback();
        runJob("Extracting the target voice", () -> {
            Benchmark.Sample s = demoSample(m);
            TseEngine eng = engine(m, th);
            setStatus("Extracting on this phone ...");
            long t0 = System.nanoTime();
            float[] e = eng.embed(s.enr);
            long t1 = System.nanoTime();
            float[] y = eng.extract(s.mix, e);
            long t2 = System.nanoTime();
            double sdri = TseEngine.siSdr(y, s.tgt) - TseEngine.siSdr(s.mix, s.tgt);
            demoMix = s.mix;
            demoTgt = s.tgt;
            demoOut = TseEngine.matchLevel(y, s.mix);
            waveOut.setAudio(demoOut);
            double sec = s.mix.length / (double) SR, ms = (t2 - t1) / 1e6;
            runOnUiThread(() -> {
                tile(R.id.tileQuality, String.format(Locale.US, "+%.1f dB", sdri), "quality gain (SI-SDRi)");
                tile(R.id.tileTime, String.format(Locale.US, "%.2f s", ms / 1000), String.format(Locale.US, "for %.1f s of audio", sec));
                tile(R.id.tileRtf, String.format(Locale.US, "%.0fx", sec * 1000 / ms), "faster than real time");
            });
            logLine(String.format(Locale.US, "DEMO %s, %d threads: enrollment %.0f ms; %.1f s of audio extracted in %.0f ms "
                    + "(real-time factor %.3f); quality SI-SDRi %.2f dB (PC: %.2f dB).", m.toUpperCase(), th,
                    (t1 - t0) / 1e6, sec, ms, ms / (sec * 1000), sdri, s.pcSiSdri));
            setStatus("Done - play the mixture, then the extracted voice.");
        });
    }

    // ------------------------------------------------------------------ my voice
    private void withMic(Runnable r) {
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
            r.run();
        } else {
            pendingAfterPermission = r;
            requestPermissions(new String[]{Manifest.permission.RECORD_AUDIO}, 7);
        }
    }

    @Override
    public void onRequestPermissionsResult(int code, @NonNull String[] perms, @NonNull int[] res) {
        super.onRequestPermissionsResult(code, perms, res);
        if (code == 7 && res.length > 0 && res[0] == PackageManager.PERMISSION_GRANTED && pendingAfterPermission != null) {
            Runnable r = pendingAfterPermission;
            pendingAfterPermission = null;
            r.run();
        } else if (code == 7) {
            setStatus("Microphone permission is needed to record.");
        }
    }

    private void record(int seconds, boolean enrollment) {
        stopPlayback();
        Step s = enrollment ? step1 : step2;
        runJob(enrollment ? "Recording your voice" : "Recording the conversation", () -> {
            float[] x = recordAudio(seconds, s, enrollment ? "Speak now" : "Talk together");
            if (enrollment) myEnroll = x; else myMix = x;
            s.wave.setAudio(x);
            float peak = 0;
            for (float v : x) peak = Math.max(peak, Math.abs(v));
            runOnUiThread(() -> {
                s.action.setText("Record again");
                s.action.setBackgroundTintList(ColorStateList.valueOf(color(R.color.accent)));
                s.action.setTextColor(color(R.color.onAccent));
                s.action.setIconTint(ColorStateList.valueOf(color(R.color.onAccent)));
            });
            stepDone(s, true);
            logLine(String.format(Locale.US, "Recorded %s: %.1f s.", enrollment ? "your voice" : "the conversation", x.length / (double) SR));
            if (peak < 0.01f) setStatus("That recording is almost silent - check the microphone and try again.");
            else setStatus(enrollment ? "Got your voice. Next: record the conversation." : "Got the conversation. Next: extract.");
        });
    }

    /** Record mono audio and return it at 8 kHz (any supported device rate is resampled), peak-normalised. */
    private float[] recordAudio(int seconds, Step s, String prompt) throws Exception {
        int[] rates = {8000, 16000, 44100, 48000};
        int[] sources = {MediaRecorder.AudioSource.VOICE_RECOGNITION, MediaRecorder.AudioSource.MIC};
        AudioRecord rec = null;
        int rate = 0;
        outer:
        for (int src : sources) {
            for (int r : rates) {
                int min = AudioRecord.getMinBufferSize(r, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
                if (min <= 0) continue;
                try {
                    AudioRecord a = new AudioRecord(src, r, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT,
                            Math.max(min * 2, r));
                    if (a.getState() == AudioRecord.STATE_INITIALIZED) { rec = a; rate = r; break outer; }
                    a.release();
                } catch (Exception ignored) { }
            }
        }
        if (rec == null) throw new IllegalStateException("The microphone could not be opened");
        logLine("Recording at " + rate + " Hz");
        int total = rate * seconds;
        short[] buf = new short[total];
        float[] live = new float[total];
        rec.startRecording();
        int got = 0, lastSec = -1;
        final int rateF = rate;
        runOnUiThread(() -> {  // keep the active button bright while recording (taps are ignored while busy)
            s.action.setEnabled(true);
            s.action.setBackgroundTintList(ColorStateList.valueOf(color(R.color.rec)));
            s.action.setTextColor(0xFFFFFFFF);
            s.action.setIconTint(ColorStateList.valueOf(0xFFFFFFFF));
        });
        while (got < total) {
            int n = rec.read(buf, got, Math.min(1024, total - got));
            if (n < 0) throw new IllegalStateException("Recording failed (" + n + ")");
            for (int i = got; i < got + n; i++) live[i] = buf[i] / 32768f;
            got += n;
            s.wave.setLive(live, got, total);
            int left = (int) Math.ceil((total - got) / (double) rateF);
            if (left != lastSec) {
                lastSec = left;
                final String txt = "Recording  " + left + " s";
                runOnUiThread(() -> s.action.setText(txt));
                setStatus(prompt + " ... " + left + " s");
            }
        }
        rec.stop();
        rec.release();
        float[] x = java.util.Arrays.copyOf(live, got);
        if (rate != SR) x = resampleTo8k(x, rate);
        float peak = 1e-6f;
        for (float v : x) peak = Math.max(peak, Math.abs(v));
        if (peak > 0.01f) for (int i = 0; i < x.length; i++) x[i] *= 0.9f / peak;
        return x;
    }

    /** Anti-aliasing low-pass (windowed sinc, cut-off 3.6 kHz) + linear-interpolation resampling to 8 kHz. */
    static float[] resampleTo8k(float[] x, int rate) {
        int taps = 63, mid = taps / 2;
        double fc = 3600.0 / rate, sum = 0;
        double[] h = new double[taps];
        for (int i = 0; i < taps; i++) {
            int k = i - mid;
            double sinc = k == 0 ? 2 * fc : Math.sin(2 * Math.PI * fc * k) / (Math.PI * k);
            h[i] = sinc * (0.42 - 0.5 * Math.cos(2 * Math.PI * i / (taps - 1)) + 0.08 * Math.cos(4 * Math.PI * i / (taps - 1)));
            sum += h[i];
        }
        float[] f = new float[x.length];
        for (int n = 0; n < x.length; n++) {
            double acc = 0;
            for (int i = 0; i < taps; i++) {
                int j = n + i - mid;
                if (j >= 0 && j < x.length) acc += h[i] * x[j];
            }
            f[n] = (float) (acc / sum);
        }
        int outLen = (int) ((long) x.length * SR / rate);
        float[] y = new float[outLen];
        double step = rate / (double) SR;
        for (int n = 0; n < outLen; n++) {
            double p = n * step;
            int i = (int) p;
            double frac = p - i;
            float a = f[Math.min(i, f.length - 1)], b = f[Math.min(i + 1, f.length - 1)];
            y[n] = (float) (a + (b - a) * frac);
        }
        return y;
    }

    private void extractMine() {
        if (myEnroll == null || myMix == null) {
            setStatus("First record your voice (step 1) and the conversation (step 2).");
            return;
        }
        stopPlayback();
        final String m = model;
        final int th = threads;
        runJob("Extracting your voice", () -> {
            TseEngine eng = engine(m, th);
            setStatus("Extracting your voice on this phone ...");
            long t0 = System.nanoTime();
            float[] e = eng.embed(myEnroll);
            long t1 = System.nanoTime();
            float[] y = eng.extract(myMix, e);
            long t2 = System.nanoTime();
            myOut = TseEngine.matchLevel(y, myMix);
            step3.wave.setAudio(myOut);
            stepDone(step3, true);
            double sec = myMix.length / (double) SR, ms = (t2 - t1) / 1e6;
            String r = String.format(Locale.US, "Extracted %.1f s of audio in %.2f s (%.0fx faster than real time) "
                    + "with the %s model.", sec, ms / 1000, sec * 1000 / ms, m.equals("r6") ? "accurate" : "compact");
            runOnUiThread(() -> mineResult.setText(r));
            logLine(String.format(Locale.US, "MINE %s, %d threads: enrollment %.0f ms; %.1f s extracted in %.0f ms "
                    + "(real-time factor %.3f).", m.toUpperCase(), th, (t1 - t0) / 1e6, sec, ms, ms / (sec * 1000)));
            setStatus("Done - tap Play on step 3 to hear only your voice.");
        });
    }

    // ------------------------------------------------------------------ benchmark
    private void benchmark(String[] models, int[] threadList) {
        stopPlayback();
        runJob("Benchmark", () -> {
            String stamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date());
            StringBuilder js = new StringBuilder("{\"device\":\"" + deviceInfo().replace("\"", "'") + "\",\"time\":\"" + stamp
                    + "\",\"runs\":[");
            boolean first = true;
            runOnUiThread(() -> benchResults.removeAllViews());
            for (String m : models) {
                List<Benchmark.Sample> s = samples(m);
                byte[] emb = asset("models/" + m + "/embedder_int8.onnx");
                byte[] ext = asset("models/" + m + "/extractor_int8.onnx");
                for (int th : threadList) {
                    setStatus("Benchmark: " + (m.equals("r6") ? "accurate" : "compact") + " model, " + th + " thread(s) ...");
                    String r = Benchmark.run(m.toUpperCase() + " INT8", emb, ext, th, s, this::logLine);
                    js.append(first ? "" : ",").append(r);
                    first = false;
                    JSONObject o = new JSONObject(r);
                    runOnUiThread(() -> benchResults.addView(resultCard(o)));
                }
            }
            js.append("]}");
            resultsJson.append(js).append("\n");
            File f = new File(getExternalFilesDir(null), "tse_benchmark_" + stamp + ".json");
            try (FileOutputStream fo = new FileOutputStream(f)) { fo.write(js.toString().getBytes(StandardCharsets.UTF_8)); }
            logLine("BENCH_JSON " + js);
            logLine("Saved " + f.getAbsolutePath());
            setStatus("Benchmark finished - tap Share results to send them.");
        });
    }

    /** A clean card with the key numbers of one benchmark run. */
    private View resultCard(JSONObject o) {
        MaterialCardView card = new MaterialCardView(this, null, com.google.android.material.R.attr.materialCardViewOutlinedStyle);
        card.setCardBackgroundColor(color(R.color.surface));
        card.setStrokeColor(color(R.color.hairline));
        card.setRadius(dp(20));
        card.setContentPadding(dp(16), dp(14), dp(16), dp(14));
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT);
        lp.topMargin = dp(8);
        card.setLayoutParams(lp);
        LinearLayout col = new LinearLayout(this);
        col.setOrientation(LinearLayout.VERTICAL);
        card.addView(col);
        try {
            boolean r6 = o.getString("model").startsWith("R6");
            TextView t = new TextView(this);
            t.setText((r6 ? "Accurate (R=6)" : "Compact (R=3)") + "  ·  " + o.getInt("threads") + " thread"
                    + (o.getInt("threads") > 1 ? "s" : ""));
            t.setTextColor(color(R.color.ink));
            t.setTextSize(16);
            t.setTypeface(Typeface.DEFAULT_BOLD);
            col.addView(t);
            row(col, "Time per 3 s window", String.format(Locale.US, "%.0f ms", o.getDouble("window3s_median_ms")));
            row(col, "Real-time factor", String.format(Locale.US, "%.3f  (%.0fx real time)", o.getDouble("rtf"), 1 / o.getDouble("rtf")));
            row(col, "Enrollment", String.format(Locale.US, "%.0f ms", o.getDouble("enroll_ms")));
            row(col, "Quality (9 mixtures)", String.format(Locale.US, "%.2f dB  (PC %.2f)", o.getDouble("mean_si_sdri_phone"),
                    o.getDouble("mean_si_sdri_pc")));
            JSONArray st = o.getJSONArray("streaming");
            for (int i = 0; i < st.length(); i++) {
                JSONObject s = st.getJSONObject(i);
                boolean rt = s.getBoolean("realtime");
                row(col, String.format(Locale.US, "Live, %.0f ms latency", s.getDouble("hop_ms") + s.getDouble("lookahead_ms")),
                        (rt ? "✓ real-time" : "✗ too slow") + String.format(Locale.US, "  %.0f ms/hop", s.getDouble("hop_median_ms")));
            }
        } catch (Exception e) {
            TextView t = new TextView(this);
            t.setText(o.toString());
            col.addView(t);
        }
        return card;
    }

    private void row(LinearLayout col, String label, String value) {
        LinearLayout r = new LinearLayout(this);
        r.setPadding(0, dp(6), 0, 0);
        TextView a = new TextView(this);
        a.setText(label);
        a.setTextColor(color(R.color.ink2));
        a.setTextSize(13);
        r.addView(a, new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1));
        TextView b = new TextView(this);
        b.setText(value);
        b.setTextColor(color(value.startsWith("✗") ? R.color.warn : value.startsWith("✓") ? R.color.good : R.color.ink));
        b.setTextSize(13);
        b.setTypeface(Typeface.DEFAULT_BOLD);
        b.setGravity(Gravity.END);
        r.addView(b);
        col.addView(r);
    }

    private void share() {
        String text = "TSE Edge benchmark\n" + deviceInfo() + "\n\n" + log.getText() + "\n\nJSON:\n" + resultsJson;
        Intent i = new Intent(Intent.ACTION_SEND);
        i.setType("text/plain");
        i.putExtra(Intent.EXTRA_SUBJECT, "TSE Edge results - " + Build.MODEL);
        i.putExtra(Intent.EXTRA_TEXT, text);
        startActivity(Intent.createChooser(i, "Share results"));
    }

    // ------------------------------------------------------------------ playback with a moving playhead
    private final Runnable tick = new Runnable() {
        @Override
        public void run() {
            if (track == null || playingWave == null) return;
            try {
                int frames = track.getBufferSizeInFrames();
                int pos = track.getPlaybackHeadPosition();
                float p = frames > 0 ? Math.min(1f, pos / (float) frames) : 1f;
                playingWave.setProgress(p);
                if (p < 1f && track.getPlayState() == AudioTrack.PLAYSTATE_PLAYING) ui.postDelayed(this, 30);
                else playingWave.setProgress(-1f);
            } catch (Exception e) {
                playingWave.setProgress(-1f);
            }
        }
    };

    private void stopPlayback() {
        ui.removeCallbacks(tick);
        if (playingWave != null) playingWave.setProgress(-1f);
        playingWave = null;
        if (track != null) {
            try { track.stop(); } catch (Exception ignored) { }
            track.release();
            track = null;
        }
    }

    private void play(float[] x, WaveformView wave, String what) {
        stopPlayback();
        if (x == null) { setStatus("Nothing to play yet: " + what); return; }
        float peak = 1e-6f;
        for (float v : x) peak = Math.max(peak, Math.abs(v));
        float g = 0.95f / peak;
        // played at 16 kHz (2x upsampled), which every device supports
        short[] pcm = new short[x.length * 2];
        for (int i = 0; i < x.length; i++) {
            float a = x[i] * g, b = (i + 1 < x.length ? x[i + 1] : x[i]) * g;
            pcm[2 * i] = (short) Math.max(-32767, Math.min(32767, a * 32767));
            pcm[2 * i + 1] = (short) Math.max(-32767, Math.min(32767, 0.5f * (a + b) * 32767));
        }
        try {
            track = new AudioTrack.Builder()
                    .setAudioAttributes(new AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA)
                            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
                    .setAudioFormat(new AudioFormat.Builder().setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                            .setSampleRate(16000).setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build())
                    .setBufferSizeInBytes(pcm.length * 2)
                    .setTransferMode(AudioTrack.MODE_STATIC)
                    .build();
            track.write(pcm, 0, pcm.length);
            track.play();
            playingWave = wave;
            ui.post(tick);
            setStatus(String.format(Locale.US, "Playing %s (%.1f s)", what, x.length / (double) SR));
        } catch (Exception e) {
            setStatus("Playback failed: " + e.getMessage());
        }
    }

    // ------------------------------------------------------------------ automatic self-test (emulator / adb)
    private void autotest() {
        logLine("AUTOTEST start");
        worker.execute(() -> {
            try {
                for (String m : new String[]{"r3", "r6"}) {
                    Benchmark.Sample s = samples(m).get(3);
                    TseEngine eng = engine(m, 4);
                    float[] y = eng.extract(s.mix, eng.embed(s.enr));
                    double sdri = TseEngine.siSdr(y, s.tgt) - TseEngine.siSdr(s.mix, s.tgt);
                    logLine(String.format(Locale.US, "AUTOTEST %s %s SI-SDRi %.3f (PC %.3f) %s", m, s.id, sdri, s.pcSiSdri,
                            Math.abs(sdri - s.pcSiSdri) < 0.05 ? "MATCH" : "MISMATCH"));
                }
                float[] tone = new float[44100];
                for (int i = 0; i < tone.length; i++) tone[i] = (float) Math.sin(2 * Math.PI * 440 * i / 44100.0);
                logLine("AUTOTEST resample 44.1k->8k length " + resampleTo8k(tone, 44100).length + " (expected 8000)");
                logLine("AUTOTEST done OK");
                setStatus("Ready - everything runs on this phone, even in airplane mode.");
            } catch (Throwable t) {
                Log.e(TAG, "AUTOTEST failed", t);
                logLine("AUTOTEST FAILED " + t);
            }
        });
    }
}
