package edu.vit.tsedemo;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;

/**
 * Minimal waveform: one rounded bar per column (peak of that slice), the played part drawn in the
 * full colour and the rest faded, with an optional "live" mode that grows while recording.
 */
public class WaveformView extends View {
    private float[] audio;
    private int filled = -1;          // live mode: only the first `filled` samples exist
    private int capacity;             // live mode: total samples expected
    private float progress = -1f;     // playback position 0..1, -1 = not playing
    private final Paint on = new Paint(Paint.ANTI_ALIAS_FLAG), off = new Paint(Paint.ANTI_ALIAS_FLAG),
            empty = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final RectF r = new RectF();
    private final float density;

    public WaveformView(Context c, AttributeSet a) {
        super(c, a);
        density = getResources().getDisplayMetrics().density;
        setColor(0xFF2A78D6);
        empty.setColor(0x22888888);
    }

    public void setColor(int color) {
        on.setColor(color);
        off.setColor((color & 0x00FFFFFF) | 0x66000000);
        invalidate();
    }

    public void setAudio(float[] x) {
        audio = x;
        filled = -1;
        progress = -1f;
        postInvalidate();
    }

    /** Live recording: show the first `n` of `cap` samples. */
    public void setLive(float[] x, int n, int cap) {
        audio = x;
        filled = n;
        capacity = cap;
        postInvalidate();
    }

    public void setProgress(float p) {
        progress = p;
        postInvalidate();
    }

    public boolean hasAudio() { return audio != null; }

    @Override
    protected void onDraw(Canvas cv) {
        float w = getWidth(), h = getHeight(), mid = h / 2f;
        float bar = 3f * density, gap = 2f * density;
        int cols = Math.max(1, (int) ((w + gap) / (bar + gap)));
        if (audio == null || audio.length == 0) {
            for (int i = 0; i < cols; i++) {
                float x = i * (bar + gap);
                r.set(x, mid - 1.5f * density, x + bar, mid + 1.5f * density);
                cv.drawRoundRect(r, bar, bar, empty);
            }
            return;
        }
        int n = filled >= 0 ? capacity : audio.length;
        int avail = filled >= 0 ? filled : audio.length;
        float peak = 1e-4f;
        for (int i = 0; i < avail; i++) peak = Math.max(peak, Math.abs(audio[i]));
        for (int c = 0; c < cols; c++) {
            int a = (int) ((long) c * n / cols), b = (int) ((long) (c + 1) * n / cols);
            float x = c * (bar + gap);
            float m = 0;
            if (a < avail) {
                for (int i = a; i < Math.min(b, avail); i++) m = Math.max(m, Math.abs(audio[i]));
            }
            float hh = Math.max(1.5f * density, (float) Math.sqrt(m / peak) * (h / 2f - density));
            r.set(x, mid - hh, x + bar, mid + hh);
            Paint p = a >= avail ? empty : (progress < 0 || (float) c / cols <= progress ? on : off);
            cv.drawRoundRect(r, bar, bar, p);
        }
    }
}
