import os
from collections import deque
import numpy as np
import scikits.audiolab as audiolab


class Note:
  """ Single note. """
  def __init__(self, dur=None, freq=None, amp=None):
    self.dur   = dur  # duration
    self.freq  = freq # pitch
    self.amp   = amp  # amplitude/volume


class Chord:
  """ Collection of Note instances. """
  def __init__(self, notes=()):
    self.notes = ()
    self.addnotes(notes)

  def addnotes(self, notes):
    """ Chord references existing Note objects, does not make copies for itself."""
    if isinstance(notes, dict):
      for key in notes:
        self.__addnote(notes[key])
    elif isinstance(notes, (list, tuple)):
      for note in notes:
        self.__addnote(note)
    else:
      self.__addnote(notes)

  def __addnote(self, note):
    """ Reciprocal function with self.addnotes, together they handle single values, lists, nested lists of notes."""
    if isinstance(note, Note):
      self.notes += (note,)
    elif isinstance(note, (list,tuple,dict)):
      self.addnotes(note)
    else:
      print("Object with " + str(type(note)) + " was not added to Chord.")

  def del_notes(self, notes):
    """ TBA: Remove notes from self.notes. By frequency? By label (must add labels)? By index? """
    # Cannot remove directly from tuple. Must reconstruct tuple.
    pass

  def to_file(self, out, dur):
    """ Write all Chord Notes to file, in superposition/summation with constant duration. """
    nc = np.zeros(dur * out.samplerate)
    nbase = np.linspace(0, dur, dur * out.samplerate)
    try:
      # Ensure a Sndfile instance is being used for writing. Is type-asserting halal?
      assert isinstance(out, audiolab.Sndfile)
      for note in self.notes:
        # Superposition each consecutive Note in the chord
        nc += note.amp * np.sin(2 * np.pi * note.freq * nbase)
      out.write_frames(nc)

    except AssertionError:
      print("Chord not written. Method tofile takes Sndfile object.")

  def weight(self, rule, switch=True):
    """ Takes a weighting function or a set of weights,
        and alters the amplitude of each note in the chord.

        switch=True (default): the weighting function is passed the index of each note.

        switch=False: the weighting function is passed the note's frequency.
    """

    # Check if rule quacks like a duck
    if hasattr(rule, "__call__"):
      try:
        for i in range(len(self.notes)):
          if switch:
            # pass index ("order" of note in chord) to weighting function
            result = rule(i)
          else:
            # pass note frequency to weighting function
            result = rule(self.notes[i].freq)

          # *** could move the following material to a method in Note (?)
          # can only be weighted by integer or float
          assert isinstance(result, (int, float))
          # REPLACES current amplitude (what about factor weight?)
          self.notes[i].amp *= result

      except (TypeError, AssertionError):
        print("Functions given to Chord's weight method must accept a single argument and return a single number.")

    elif hasattr(rule, '__iter__') and len(rule) == len(self.notes):
      # Assign a sequence of appropriate length
      # Is this necessary? could assume user is sane, the following code could easily be made not to break for short/long rule sequence
      # Could also allow for a specified subset to be weighted. Easier if labels/dict is implemented for Chord.
      for i in rule:
        self.notes[i].amp *= rule[i]

    else:
      print("Chord could not be weighted.")


class Scale:
    """ 12-tone octave scale (from frequency base to 2*base)
        Intervening frequencies depend on chosen intonation/temperament.
    """
    names = [u"A", u"B\u266D", u"B", u"C", u"D\u266D", u"D", u"E\u266D", u"E", u"F", u"G\u266D", u"G", u"A\u266D"]
    names_alt = [u"A", u"A\u266F", u"C\u266D", u"B\u266F", u"C\u266F", u"D", u"D\u266F", u"F\u266D", u"E\u266F", u"F\u266F", u"G", u"G\u266F"]

    def __init__(self, base, intonation="equal", base_name="", exclusions=()):

        self.base = base
        self.notes = [base] * 13
        self.tune(intonation)
        self.exclude(exclusions)
        self.name(base_name)

    def __str__(self):
        return unicode(self).encode('utf-8')

    def __unicode__(self):
        string = unicode("")
        for i in range(len(self.names)):
            space = u"\t" if (i + 1) % 3 else u"\n"
            string += u"{:<2}/{:6.2f} Hz{}".format(self.names[i], self.notes[i], space)
        return string

    def tune(self, intonation):

        self.intonation = intonation

        if intonation == "equal":
            semitone = 1.05946
            # do not specify the base or the octave, but the notes between
            for i in range(len(self.notes) - 1):
                self.notes[i] = self.base * semitone ** i
            self.notes[-1] = 2 * self.base

        elif intonation == "Hindemith":
            #omitting Gb (only F# included) for now
            ratios = [1., 4.*(4./3)*(1./5), 3.*(3./2)*(1./4), 6.*(1./5), 5.*(1./4),
                        4.*(1./3), 3.*(3./2)*(5./4)*(1./4), 3.*(1./2), 4.*(2./5),
                        5.*(1./3), 4.*(4./3)*(1./3), 5.*(3./4)*(1./2), 2.]
            for i in range(len(self.notes)):
                self.notes[i] *= ratios[i]

    def exclude(self, exclusions):

        self.exclusions = [self.notes[i] for i in exclusions]
        self.notes = [self.notes[i] for i in range(len(self.notes)) if not (i in exclusions)]

    def name(self, base_name):

        self.tonic = unicode(base_name).capitalize()

        if len(self.tonic) > 1:
            self.tonic = self.tonic.replace("b", u"\u266D")
            self.tonic = self.tonic.replace("#", u"\u266F")

        try:
            tonic = Scale.names.index(self.tonic)
            names_ = deque(Scale.names)
            deque.rotate(names_, -tonic)
            self.names = list(names_)
        except ValueError:
            print("The tonic could not be identified from the string ", base_name)


def reveal_formats():
  """ From Audiolab documentation. Print the available audio output formats and encodings."""
  for fmt in audiolab.available_file_formats():
    print("File format {} is supported; available encodings are:".format(fmt))
    for enc in audiolab.available_encodings(fmt):
        print("\t{}".format(enc))
    print('')


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
