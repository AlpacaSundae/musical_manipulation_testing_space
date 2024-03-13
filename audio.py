#
#  some audio related functions
#

import librosa
import numpy as np

def split_beats(
        ts: np.ndarray,
        sr: float = 22050,
        ) -> list[np.ndarray]:
    r"""Split up an audio file based on librosa beat detection

    Parameters
    ----------
    ts : np.array
        audio time series

    sr : number > 0 
        sample rate of ts
        (librosa defaults to ``sr=22050`` Hz when unspecified)
    
    Returns
    -------
    ts_list : list of np.array
    
    """
    _, beat_samples = librosa.beat.beat_track(y=ts, sr=sr, units="samples")


    ts_list = []

    # append up to first dtected beat
    ts_list.append(np.array(ts[:beat_samples[0]]))

    # append each beat (accessing up to idx+1 so need to cut range one early)
    for idx in range(0, len(beat_samples) - 1):
        ts_list.append(np.array(ts[beat_samples[idx]:beat_samples[idx+1]]))

    # append final part of song
    ts_list.append(np.array(ts[beat_samples[-1]:]))

    return ts_list

def combine_audio_list(ts_list: list[np.ndarray]):
    """For recombining a list of time series data, we just call np.concatenate"""
    return np.concatenate(ts_list)

if __name__ == "__main__":
    ts, sr = librosa.load("./media/edcast.mp3")
    ts_trim, _ = librosa.effects.trim(ts)
    output = split_beats(ts_trim, sr)

    import random

    random.shuffle(output)
    together = combine_audio_list(output)

    # import soundfile as sf
    # sf.write("out1.wav", ts_trim, sr)
    # sf.write("out2.wav", together, sr)

    print(len(ts_trim))
    print(len(together))
