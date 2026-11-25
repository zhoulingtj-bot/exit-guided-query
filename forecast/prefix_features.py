"""Build the frozen 101-column prefix representation in the archived order."""
import numpy as np

integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
SEGMENTS = ((0, 100), (100, 200), (200, 350), (350, 500))


def context(train):
    """Estimate current scale and nominal voltage from the source training fold."""
    scale = float(np.percentile(abs(train[:, :, 1]), 95))
    voltage = np.round(np.median(train[:, :, 0], axis=0), 3)
    return scale, voltage


def projection(voltage, current):
    return voltage @ current / (voltage @ voltage + 1e-9)


def recent_change(current):
    return np.diff(current[-min(8, len(current)):]).mean()


def state(curve, count, scale):
    voltage = curve[:count, 0]
    current = curve[:count, 1] / scale
    index = np.arange(count)
    resample = np.linspace(0, count - 1, 32)
    sampled_voltage = np.interp(resample, index, voltage) / 1.5
    sampled_current = np.interp(resample, index, current)
    stats = [
        current[-1], current.mean(), current.std(), current.min(), current.max(),
        recent_change(current), projection(voltage, current), integrate(current, voltage),
        voltage.min(), voltage.max(), count / 500, len(current) / 500,
    ]
    for start, stop in SEGMENTS:
        end = min(count, stop)
        if end <= start:
            stats.extend([0.0] * 6)
            continue
        segment_voltage = curve[start:end, 0]
        segment_current = curve[start:end, 1] / scale
        stats.extend([
            1.0, segment_current[-1], segment_current.mean(), segment_current.std(),
            recent_change(segment_current) if len(segment_current) > 1 else 0.0,
            projection(segment_voltage, segment_current),
        ])
    return np.r_[sampled_voltage, sampled_current, stats]


def features(observed, composition, count, scale):
    states = np.stack([state(curve, count, scale) for curve in observed])
    return np.column_stack([states, (composition == "Br-rich").astype(float)])
