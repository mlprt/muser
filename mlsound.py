import scikits.audiolab as alab
import numpy as np
import Nsound as ns
import os

class Soundwrite:
  """ Prepare a file stream (self.out) for writing. """
  def __init__(self, rate, chans=1, fmt="wav", enc="pcm24", name="sound"):
    self._update(rate, chans, fmt, enc, name)
  
  def _update(self, rate=None, chans=None, fmt=None, enc=None, name=None):
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
    self._update(fmt=fmt, enc=enc)
  
  def new_name(self, name):
    """Change file name."""
    self._update(name=name)
    
def reveal_formats():
  for format in alab.available_file_formats():
    print "File format %s is supported; available encodings are:" % format
    for enc in alab.available_encodings(format):
        print "\t%s" % enc
    print ""
    
class Note:
  """ Single note, with duration, frequency, sampling frequency. 
  """
  def __init__(self, dur=None, freq=None, rate=None):
    if (dur=None and freq=)
    self.dur, self.freq = (dur, freq) # duration and pitch
    self.rate           = rate    # sampling rate, Hz
  
  def clone(self):
    class Clone(self):
      pass
    return Clone
    
class Chord:
  """ Collection of Note instances.
      Methods for analysis and alteration (???)
  """
  def __init__(self):
    pass
  
  def weight(self, ):
    try:
      pass
    except:
      pass

sdur = 2
sfreq = 48000
fund = 220
fname = "test"
ffmt = "flac"

if os.path.isfile(fname+"."+ffmt): os.remove(fname+ffmt)

f = Soundwrite(sfreq, fmt=ffmt, name=fname)

sample_base = np.linspace(0,sdur,sfreq*sdur)

#for j in range(1,6):
  #nc = np.zeros(sfreq*sdur)
  #for i in range(1,16):
    #n = np.sin(2*np.pi*i*fund*sample_base)
    #nc += n/(i**(j)) # subtracting i%2 favors octaves most
    ##f.out.write_frames(n)
  #f.out.write_frames(nc)

#f.out.write_frames(np.sin(2*np.pi*fund*sample_base))
