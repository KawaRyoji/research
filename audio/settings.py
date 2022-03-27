import math
from util.calc import sec2point


class params:
    # ポイント数で指定
    def __init__(self, fs, flen, fshift, fftl, image_width) -> None:
        self.fs = fs
        self.flen = flen
        self.fshift = fshift
        self.fftl = fftl
        self.image_width = image_width

    @classmethod
    def librosa_default(cls, fs, image_width, fftl=2048):
        flen_p = fftl
        fshift_p = flen_p // 4
        
        return cls(fs, flen_p, fshift_p, fftl, image_width)
    
    @classmethod
    def from_sec(cls, fs, flen_sec, fshift_sec, fftl, image_width_sec):
        flen_p = sec2point(fs, flen_sec)
        fshift_p = sec2point(fs, fshift_sec)
        
        image_width_f = math.floor(image_width_sec / fshift_sec)
                
        return cls(fs, flen_p, fshift_p, fftl, image_width_f)
    
    def __str__(self) -> str:
        return "sampling frequency: \t{}\tHz\nflame shift:\t\t{}\tpoint\nflame length: \t\t{}\tpoint\nfft length: \t\t{}\tpoint\nimage_width: \t\t{}\tframes".format(self.fs, self.fshift, self.flen, self.fftl, self.image_width)
