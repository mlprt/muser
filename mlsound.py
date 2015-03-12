import scikits.audiolab as alab
import numpy as np
import Nsound as ns
import os

class Soundwrite:
  """ Prepare a file stream (self.out) for writing. 
      (Just a handler class for the Audiolab classes.) """
  def __init__(self, rate, chans=1, fmt="wav", enc="pcm24", name="sound"):
    self.__update(rate, chans, fmt, enc, name)
  
  def __update(self, rate=None, chans=None, fmt=None, enc=None, name=None):
    """ Update any changed variables, and create new self.out instance """
    
    # Only update variables that are specified
    # Allows functions new_fmt, new_name, etc. to all operate through here
    if not(rate==None):  self.rate     = rate
    if not(chans==None): self.chans    = chans
    
    if not(fmt==None and enc==None):
      # New Audiolab Format instance needed if either new file format or encoding
      self.fmt = alab.Format( self.fmt.file_format if (fmt==None) else fmt, 
                              self.fmt.encoding    if (enc==None) else enc )
    
    if not(name==None and fmt==None):
      # New filename needed if either new file format or name 
      # Handles redundant extensions and recovery of name from self.filename
      new_name = os.path.splitext(self.filename if (name==None) else name)
      new_fmt  = self.fmt.file_format if (fmt==None) else fmt
      self.filename = ( new_name[0] + "." + new_fmt)
      if not(new_name[1]==new_fmt or len(new_name[1])==0):
        print("\tFilename was specified with conflicting extension.\n\
        Discarded in favour of separately specified/existing extension.")
    
    # Create new Sndfile instance with modifications
    self.out = alab.Sndfile(self.filename, "w", self.fmt, self.chans, self.rate) 

  def new_fmt(self, fmt, enc=None):
    """Change file format."""
    self.__update(fmt=fmt, enc=enc)
  
  def new_name(self, name):
    """Change file name."""
    self.__update(name=name)
    
class Note:
  """ Single note. """
  def __init__(self, dur=None, freq=None, amp=None):
    self.dur   = dur  # duration
    self.freq  = freq # pitch
    self.amp   = amp  # amplitude/volume
  
    
class Chord:
  """ Collection of Note instances. """
  def __init__(self, notes=()):
    self.notes = ()      # Tuple? Any advantage at all? Numpy array? List?
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
    """ Reciprocal function with self.addnotes, together handling single values, lists, nested lists of notes."""
    if isinstance(note, Note):  
      self.notes += (note,)
    elif isinstance(note, (list,tuple,dict)):    
      self.addnotes(note)
    else:
      print("Object with " + str(type(note)) + " was not added to Chord.")
  
  def delnotes(self, notes):
    """ TBA: Remove notes from self.notes. By frequency? By label (must add labels)? By index? """
    # Cannot remove directly from tuple. Must reconstruct tuple.
    pass
    
  def tofile(self, out, dur):
    """ Write all Chord Notes to file, in superposition/summation with constant duration. """
    nc = np.zeros(dur*out.samplerate)
    nbase = np.linspace(0,dur,dur*out.samplerate)
    try:
      # Ensure a Sndfile instance is being used for writing. Is type-asserting halal?
      assert isinstance(out, alab.Sndfile)
      for note in self.notes:
        # Superposition each consecutive Note in the chord
        nc += note.amp*np.sin(2*np.pi*note.freq*nbase)
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
          self.notes[i].amp = rule(i)
          
      except TypeError, AssertionError:
        print("Functions given to Chord's weight method must accept a single argument and return a single number.")
    
    elif isinstance(rule, (list, tuple, dict)) and len(rule)==len(self.notes):
      # Assign a sequence of appropriate length 
      # Is this necessary? could assume user is sane, the following code could easily be made not to break for short/long rule sequence
      # Could also allow for a specified subset to be weighted. Easier if labels/dict is implemented for Chord.
      for i in rule:
        self.notes[i].amp = rule[i]
    
    else:
      print("Chord could not be weighted. ")

def reveal_formats():
  """ From Audiolab documentation. Print the available audio output formats and encodings."""
  for format in alab.available_file_formats():
    print "File format %s is supported; available encodings are:" % format
    for enc in alab.available_encodings(format):
        print "\t%s" % enc
    print ""

def chord_test(base,fname="test",ffmt="flac",srate=48000):
  
  # Remove any file with the same name as the one that will be written
  if os.path.isfile(fname+"."+ffmt): os.remove(fname+"."+ffmt)
  # Create Audiolab class handler (audio file interface)
  fout = Soundwrite(srate, fmt=ffmt, name=fname)

  chords = []
  # Number of overtones in each SINGLE note
  note_tones = 16   
  # Weighting function for overtones in each note (higher y == less weight to higher tones)
  w_fac = 2.5
  #spectrum = lambda x: (x+1)**(-w_fac) 
  spectrum = lambda x: 10**(-x**2)
  interval = 1.05946 
  nbases = [interval**i for i in [0,3,7,12,7,3]]
  nbases *= 4
  #nbases = [interval**i for i in range(13)]
  #osc = [nbases[0],nbases[-1]]
  #nbases += osc + osc + osc
  
  series_len = 1
  n_series = len(nbases)
  
  for k in range(n_series): # number of series played consecutively
    nbase = base*nbases[k]      # change in base note between series
    for j in range(1,series_len+1): # Number of notes played consecutively
      # Add note
      chords.append(Chord([Note(1,nbase*j*i,1) for i in range(1,note_tones+1)]))
      #chords[-1].addnotes(Note(1,nbase*0.5,1)) # doesn't work, adds to the END
      chords[-1].weight(spectrum)
      chords[-1].tofile(fout.out,0.15)

chord_test(55)
