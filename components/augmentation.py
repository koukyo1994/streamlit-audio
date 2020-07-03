import librosa
import numpy as np
import streamlit as st


class AudioTransform:
    def __init__(self, always_apply=False, p=0.5):
        self.always_apply = always_apply
        self.p = p

    def __call__(self, y: np.ndarray):
        if self.always_apply:
            return self.apply(y)
        else:
            if np.random.rand() < self.p:
                return self.apply(y)
            else:
                return y

    def apply(self, y: np.ndarray):
        raise NotImplementedError


class NoiseInjection(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_noise_level=0.5, sr=32000):
        super().__init__(always_apply, p)

        self.noise_level = (0.0, max_noise_level)
        self.sr = sr

    def apply(self, y: np.ndarray, **params):
        noise_level = np.random.uniform(*self.noise_level)
        noise = np.random.randn(len(y))
        augmented = (y + noise * noise_level).astype(y.dtype)
        return augmented


class PitchShift(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_range=5, sr=32000):
        super().__init__(always_apply, p)
        self.max_range = max_range
        self.sr = sr

    def apply(self, y: np.ndarray, **params):
        n_steps = np.random.randint(-self.max_range, self.max_range)
        augmented = librosa.effects.pitch_shift(y, self.sr, n_steps)
        return augmented


class TimeStretch(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_rate=1, sr=32000):
        super().__init__(always_apply, p)
        self.max_rate = max_rate
        self.sr = sr

    def apply(self, y: np.ndarray, **params):
        rate = np.random.uniform(0, self.max_rate)
        augmented = librosa.effects.time_stretch(y, rate)
        return augmented


def _db2float(db: float, amplitude=True):
    if amplitude:
        return 10**(db / 20)
    else:
        return 10 ** (db / 10)


def volume_down(y: np.ndarray, db: float):
    """
    Low level API for decreasing the volume
    Parameters
    ----------
    y: numpy.ndarray
        stereo / monaural input audio
    db: float
        how much decibel to decrease
    Returns
    -------
    applied: numpy.ndarray
        audio with decreased volume
    """
    applied = y * _db2float(-db)
    return applied


def volume_up(y: np.ndarray, db: float):
    """
    Low level API for increasing the volume
    Parameters
    ----------
    y: numpy.ndarray
        stereo / monaural input audio
    db: float
        how much decibel to increase
    Returns
    -------
    applied: numpy.ndarray
        audio with increased volume
    """
    applied = y * _db2float(db)
    return applied


class RandomVolume(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, limit=10):
        super().__init__(always_apply, p)
        self.limit = limit

    def apply(self, y: np.ndarray, **params):
        db = np.random.uniform(-self.limit, self.limit)
        if db >= 0:
            return volume_up(y, db)
        else:
            return volume_down(y, db)


def augmentations_on_wave(y: np.ndarray, sr: int):
    st.sidebar.markdown("#### Augmentations option")
    options = st.sidebar.multiselect(
        "augmentation to apply",
        options=["noise", "pitch", "stretch", "volume"])
    compose = []
    if "noise" in options:
        st.sidebar.markdown("NoiseInjection")
        max_noise_level = st.sidebar.number_input(
            "max noise level",
            min_value=0.01,
            max_value=0.5,
            value=0.1,
            step=0.01)
        always_apply = st.sidebar.checkbox(
            "always_apply", key="noise_always_apply")
        p = st.sidebar.slider(
            "p",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            key="noise_p")
        compose.append(
            NoiseInjection(
                always_apply=always_apply,
                p=p,
                max_noise_level=max_noise_level,
                sr=sr))

    if "pitch" in options:
        st.sidebar.markdown("PitchShift")
        max_range = st.sidebar.number_input(
            "max range",
            min_value=1,
            max_value=10,
            value=5,
            step=1)
        always_apply = st.sidebar.checkbox(
            "always_apply", key="pitch_always_apply")
        p = st.sidebar.slider(
            "p",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            key="pitch_p")
        compose.append(
            PitchShift(  # type: ignore
                always_apply=always_apply,
                p=p,
                max_range=max_range,
                sr=sr))

    if "stretch" in options:
        st.sidebar.markdown("Stretch")
        max_rate = st.sidebar.number_input(
            "max_rate",
            min_value=0.01,
            max_value=2.5,
            value=1.0,
            step=0.01)
        always_apply = st.sidebar.checkbox(
            "always_apply", key="stretch_always_apply")
        p = st.sidebar.slider(
            "p",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            key="stretch_p")
        compose.append(
            TimeStretch(  # type: ignore
                always_apply=always_apply,
                p=p,
                max_rate=max_rate,
                sr=sr))

    if "volume" in options:
        st.sidebar.markdown("Volume")
        limit = st.sidebar.number_input(
            "db limit",
            min_value=1,
            max_value=20,
            value=3,
            step=1)
        always_apply = st.sidebar.checkbox(
            "always_apply", key="volume_always_apply")
        p = st.sidebar.slider(
            "p",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            key="volume_p")
        compose.append(
            RandomVolume(  # type: ignore
                always_apply=always_apply,
                p=p,
                limit=limit))

    return apply(y, compose)


@st.cache
def apply(y: np.ndarray, compose: list):
    y_processed = y.copy()
    for augmentor in compose:
        y_processed = augmentor(y=y_processed)
    return y_processed
