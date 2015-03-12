import scikits.audiolab as alab
import numpy as np
import Nsound as ns
import os

class Soundwrite:
  """ Prepare a file stream (self.out) for writing. """
  def __init__(self, rate, chans=1, fmt="wav", enc="pcm24", name="sound"):
    self.__update(rate, chans, fmt, enc, name)
  
  def __update(self, rate=None, chans=None, fmt=None, enc=None, name=None):
    """ Update any changed variables, and create new self.out instance """
    if not(rate==None):  self.rate     = rate
    if not(chans==None): self.chans    = chans
    if not(fmt==None and enc==None):     
                         self.fmt      = alab.Format( self.fmt.file_format if fmt==None else fmt, 
                                                      self.fmt.encoding    if enc==None else enc )
    if not(name==None and fmt==None):   
                         self.filename = ( os.path.splitext(self.filename if name==None else name)[0] + "." 
                                                  + (self.fmt.file_format if fmt ==None else fmt)          )
                                        
    self.out = alab.Sndfile(self.filename, "w", self.fmt, self.chans, self.rate) 

  def new_fmt(self, fmt, enc=None):
    """Change file format."""
    self.__update(fmt=fmt, enc=enc)
  
  def new_name(self, name):
    """Change file name."""
    self.__update(name=name)
    
def reveal_formats():
  for format in alab.available_file_formats():
    print "File format %s is supported; available encodings are:" % format
    for enc in alab.available_encodings(format):
        print "\t%s" % enc
    print ""
    
class Note:
  """ Single note. 
  """
  def __init__(self, dur=None, freq=None, amp=None):
    self.dur   = dur  # duration
    self.freq  = freq # pitch
    self.amp   = amp  # amplitude/volume
  
    
class Chord:
  """ Collection of Note instances.
      Currently using tuple self.notes
      Methods for analysis and alteration (???)
  """
  def __init__(self, notes=[]):
    self.notes = ()
    self.addnotes(notes)
  
  def addnotes(self, notes):
    """Currently shares existing Note objects, does not copy them."""
    if isinstance(notes, dict): 
      for key in notes:   
        self.__addnote(notes[key])
    elif isinstance(notes, (list, tuple)):                       
      for note in notes:  
        self.__addnote(note)
    else:                   
      self.__addnote(notes)
        
  def __addnote(self, note):
    if isinstance(note, Note):  
      self.notes += (note,)
    elif isinstance(note, (list,tuple,dict)):    
      self.addnotes(note)
    else:
      print("Object of " + str(type(note)) + " was not added to Chord.")
  
  def delnotes(self, notes):
    pass
    
  def tofile(self, out, dur):
    nc = np.zeros(dur*out.samplerate)
    nbase = np.linspace(0,dur,dur*out.samplerate)
    try:
      assert isinstance(out, alab.Sndfile)
      for note in self.notes:
        nc += note.amp*np.sin(2*np.pi*note.freq*nbase)
      out.write_frames(nc)
    except AssertionError:
      print("Chord not written. Method tofile takes Sndfile object.")
    
  def weight(self, rule, switch=True):
    """ Takes a weighting function or a set of weights,
        and alters the amplitude of each note in the chord. 
        
        switch=True (default): the weighting function is
        passed the index of each note.
        
        switch=False: the weighting function is passed the note's frequency. 
    """
    if hasattr(rule, "__call__"):
      try:
        for i in range(len(self.notes)):
          if switch:
            result = rule(i)
          else:
            result = rule(self.notes[i].freq)
          # can only be weighted by integer or float
          assert isinstance(result, (int, float))
          # REPLACES current amplitude (what about factor weight?)
          self.notes[i].amp = rule(i)
          
      except TypeError, AssertionError:
        print("Functions given to Chord's weight method must accept a single argument and return a single number.")
    
    elif isinstance(rule, (list, tuple, dict)) and len(rule)==len(self.notes):
      for i in rule:
        self.notes[i].amp = rule[i]
    
    else:
      print("Chord could not be weighted. ")
    # could also add another case for dict 2D list/tuple weighting a subset of notes

srate = 48000
fname = "test"
ffmt = "flac"
if os.path.isfile(fname+"."+ffmt): os.remove(fname+"."+ffmt)
fout = Soundwrite(srate, fmt=ffmt, name=fname)

nbase = 220 

a_tones = []
otones = 8

for k in range(1,10000):
  for j in range(1,16):
    a_tones.append(Chord([Note(1,nbase*j*i,1) for i in range(1,otones+1)]))
    a_tones[-1].weight(lambda i: (i+1)**(-3))
    a_tones[-1].tofile(fout.out,0.5/k)
