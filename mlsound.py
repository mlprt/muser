import scikits.audiolab as alab
import numpy as np
import Nsound as ns
import os

class Soundwrite:
  """ Prepare a file stream (self.out) for writing. 
      (Handler class for the Audiolab classes.) """
  def __init__(self, rate=48000, chans=1, fmt="wav", enc="pcm24", name="sound"):
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
    """ Reciprocal function with self.addnotes, together they handle single values, lists, nested lists of notes."""
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
          self.notes[i].amp *= result
          
      except TypeError, AssertionError:
        print("Functions given to Chord's weight method must accept a single argument and return a single number.")
    
    elif isinstance(rule, (list, tuple, dict)) and len(rule)==len(self.notes):
      # Assign a sequence of appropriate length 
      # Is this necessary? could assume user is sane, the following code could easily be made not to break for short/long rule sequence
      # Could also allow for a specified subset to be weighted. Easier if labels/dict is implemented for Chord.
      for i in rule:
        self.notes[i].amp *= rule[i]
    
    else:
      print("Chord could not be weighted. ")

class Scale:
    """ 12-tone octave scale (from frequency base to 2*base)
        Intervening frequencies depend on chosen intonation/temperament.
        
    """
    names = [u"A", u"B\u266D", u"B", u"C", u"D\u266D", u"D", u"E\u266D", u"E", u"F", u"G\u266D", u"G", u"A\u266D"]
    names_alt = [u"A", u"A\u266F", u"C\u266D", u"B\u266F", u"C\u266F", u"D", u"D\u266F", u"F\u266D", u"E\u266F", u"F\u266F", u"G", u"G\u266F"]
    
    def __init__(self, base, intonation="equal", base_name="", exclusions=[]):
        
        self.base = base
        self.notes = [base]*13
        self.tune(intonation)        
        self.exclude(exclusions)
        self.name(base_name)
    
    def __str__(self):
        return unicode(self).encode('utf-8')
    
    def __unicode__(self):
        string = unicode("")
        for i in range(len(self.names)):
            space = u"\n" if not((i+1)%3) else u"\t"
            string += self.names[i].ljust(2, " ") + u"/%6.2f Hz" % self.notes[i] + space  
        return string
    
    def tune(self, intonation):
        
        self.intonation = intonation
        
        if intonation == "equal":
            semitone = 1.05946
            # do not specify the base or the octave, but the notes between
            for i in range(len(self.notes)-1):
                self.notes[i] = self.base*semitone**i
            self.notes[-1] = 2*self.base

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
        if not (base_name == ""):
            
            from collections import deque
            
            self.tonic = unicode(base_name).capitalize() 
            
            if len(self.tonic) > 1:
                self.tonic = self.tonic.replace("b", u"\u266D")
                self.tonic = self.tonic.replace("#", u"\u266F")
            
            try:
                tonic = Scale.names.index(self.tonic)
                names_ = deque(Scale.names)
                deque.rotate(names_,-tonic)
                self.names = list(names_)
                
            except ValueError:
                print("The tonic could not be identified from the string ", base_name)
        
          
def reveal_formats():
  """ From Audiolab documentation. Print the available audio output formats and encodings."""
  for format in alab.available_file_formats():
    print "File format %s is supported; available encodings are:" % format
    for enc in alab.available_encodings(format):
        print "\t%s" % enc
    print ""

def chord_test(base, base_name="",fname="test",ffmt="flac",srate=48000):
  
  # Remove any file with the same name as the one that will be written
  if os.path.isfile(fname+"."+ffmt): os.remove(fname+"."+ffmt)
  # Create Audiolab class handler (audio file interface)
  fout = Soundwrite(srate, fmt=ffmt, name=fname)

  chords = []
  # Number of overtones in each SINGLE note
  note_tones = 16
  # Weighting function for overtones in each note (higher y == less weight to higher tones)
  w_fac = 3
  spectrum = lambda x: (x+3)**(-w_fac)
  
  melody = [0,3,0,4,0,5,0,7]  
  durs = [0.5]*len(melody)
  vols = [0.6,0.7,0.6,0.75,0.7,0.8,0.75,1]
  
  equal_scale = Scale(base, "equal")
  hind_scale = Scale(base, "Hindemith")  
  
  #print "Notes in the equal-temperament scale: ", " Hz, ".join(["%6.2f"%i for i in equal_scale.notes]), "Hz"
  #print "Notes in the just scale of Hindemith: ", " Hz, ".join(["%6.2f"%i for i in hind_scale.notes]), "Hz"
  
  nbases = [hind_scale.notes[i] for i in melody]
  nbases *= 4
  durs *= 4
  vols *= 4
  
  series_len = 1
  n_series = len(nbases)
  
  for k in range(n_series): # number of series played consecutively
    nbase = nbases[k]      # change in base note between series
    for j in range(1,series_len+1): # Number of notes played consecutively
      # Add note
      chords.append(Chord([Note(1,nbase*j*i,vols[k]) for i in range(1,note_tones+1)]))
      #chords[-1].addnotes(Note(1,nbase*0.5,1)) # doesn't work, adds to the END
      chords[-1].weight(spectrum)
      chords[-1].tofile(fout.out,durs[k])

chord_test(110, base_name="A")
