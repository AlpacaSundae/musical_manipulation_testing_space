import librosa
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import numpy as np
import sys

#
# specify song to flip beats around as first argument
# second argument is a float value which specifies an offset on the beat position
#       use this if detection does not perfectly align with beats
#       multiplies the value with a proportion of the detected samples per beat
#

offset_percent = 0.0

if len(sys.argv) > 2:
    offset_percent = float(sys.argv[2])
else:
    print("INFO: as a second argument, you can specify a float to offset beat by (multiple of a beat length)")

if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    print("ERROR: i need a filename of a song as the first argument")
    exit()

time_series, sample_rate = librosa.load(filename)

tempo, beat_frames = librosa.beat.beat_track(y=time_series, sr=sample_rate)

samples_per_beat = (tempo/60) * sample_rate

print(f"estimated tempo: {tempo:.2f} bpm, @ a sample rate of {sample_rate} Hz we have {samples_per_beat:.0f} samples/beat")

#sd.play(time_series, sample_rate)

beat_samples = librosa.frames_to_samples(beat_frames)

rearranged = []
offset = int(samples_per_beat * offset_percent) 
for ii in range(0, len(beat_samples)-2,2):
    rearranged.append((beat_samples[ii+1]+offset, beat_samples[ii+2]+offset))
    rearranged.append((beat_samples[ii]+offset, beat_samples[ii+1]+offset))

time_series_rearranged = np.array(0.0)
for (start, end) in rearranged:
    if (start < 0
        or end < 0 
        or start > len(time_series)
        or end > len(time_series)):
        continue
    time_series_rearranged = np.append(time_series_rearranged, time_series[start:end])

#sd.play(time_series_rearranged, sample_rate)
sf.write("out.wav", time_series_rearranged, sample_rate)

while True:
    pass