""" Music theory, timbre control?
"""

import os
from collections import deque
import numpy as np
from scipy.io import wavfile
import music21


def chord_test(base, base_name="", file_name="test", file_fmt="flac", srate=48000):

  enc = "pcm24"
  fmt = audiolab.Format(file_fmt, enc)
  chans = 1

  # Remove any file with the same name as the one that will be written
  if os.path.isfile("{}.{}".format(file_name, file_fmt)):
    os.remove("{}.{}".format(file_name, file_fmt))
  # Create Audiolab class handler (audio file interface)
  fout = audiolab.Sndfile(fname, "w", fmt, chans, rate)

  chords = []
  # Number of overtones in each SINGLE note
  note_tones = 16
  # Weighting function for overtones in each note (higher y == less weight to higher tones)
  w_fac = 3
  def spectrum(x):
    return (x + 3) ** (-w_fac)

  melody = [0, 3, 0, 4, 0, 5, 0, 7]
  durs = [0.5] * len(melody)
  vols = [0.6, 0.7, 0.6, 0.75, 0.7, 0.8, 0.75, 1]

  equal_scale = Scale(base, "equal")
  hind_scale = Scale(base, "Hindemith")

  #print("Notes in the equal-temperament scale: ", " Hz,
  #      ".join(["%6.2f"%i for i in equal_scale.notes]), "Hz")
  #print("Notes in the just scale of Hindemith: ", " Hz,
  #      ".join(["%6.2f"%i for i in hind_scale.notes]), "Hz")

  nbases = [hind_scale.notes[i] for i in melody]
  nbases *= 4
  durs *= 4
  vols *= 4

  series_len = 1
  n_series = len(nbases)

  for k in range(n_series): # number of series played consecutively
    nbase = nbases[k]      # change in base note between series
    for j in range(1, series_len + 1): # Number of notes played consecutively
      # Add note
      chords.append(Chord([Note(1, nbase * j * i, vols[k]) for i in range(1, note_tones + 1)]))
      #chords[-1].addnotes(Note(1, nbase * 0.5,1)) # doesn't work, adds to the END
      chords[-1].weight(spectrum)
      chords[-1].to_file(fout.out, durs[k])


if __name__ == '__main__':
  chord_test(110, base_name="A")
