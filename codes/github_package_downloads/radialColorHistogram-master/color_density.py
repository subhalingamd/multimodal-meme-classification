''' Class to process image data into color density features for model training '''

import os
from numpy import array, ravel, log1p
from cv2 import imread, cvtColor, calcHist, COLOR_BGR2RGB, \
    COLOR_BGR2HSV, COLOR_BGR2GRAY
from sklearn.preprocessing import MinMaxScaler
from . import radial_split
from joblib import Parallel, delayed


model_ranges = {'RGB': [0, 256, 0, 256, 0, 256],
                'HSV': [0, 180, 0, 256, 0, 256],
                'GRAY': [0, 256]}
model_conversion = {'RGB': COLOR_BGR2RGB,
                    'HSV': COLOR_BGR2HSV,
                    'GRAY': COLOR_BGR2GRAY}
model_channels = {'RGB': 3, 'HSV': 3, 'GRAY': 1}


class ColorDensity(object):
    """Class that processes images and outputs color density features"""

    def __init__(self,
                 splitter=radial_split.RadialSplitter(),
                 color_model='RGB',
                 n_bins=8,
                 log_transform=False,
                 scaler=False,
                 ):
        """Constructs attributes needed to process images into
            histogram features

            Parameters
            ----------
            splitter: RadialSplitter Object or None,
                From radial_split.py, used to split images into segments,
                meaning the featureset produced by this class will contain
                a set of color density features for each segment.
                Or pass None to not split image.
            color_model: str
                Color model to use, must by 'RGB', 'HSV' or 'GRAY'
            n_bins: int
                Number of color bins per color channel.
                Note the number of columns in the color density featureset
                (per segment) will equal n_bins to the power of the number
                of channels. (e.g. if n_bins=8 and color_model='RGB',
                it will be 8^3 per segment)
            log_transform: bool
                If true a log transformation will be applied to the color
                density features (before applying the scaler)
            scaler: bool or Scaler Object
                Used to apply a scaler to the color density featureset
                (or pass True to use MinMaxScaler() or False to use no scaler)
                When the fit_transform method is called the scaler will
                be fit and applied, when the the
                transform method is called the scaler just be applied.
        """
        # Check validity of arguments
        assert isinstance(n_bins, int), "'n_bins' must be int"
        assert isinstance(
            log_transform, bool), "'log_transform' must be boolean"
        assert color_model in ['RGB', 'HSV', 'GRAY'], \
            "'color_model must be 'RGB', 'HSV' or 'GRAY'"
        if type(scaler) != bool:  #  check scaler input is valid
            assert hasattr(scaler, 'fit_transform') and hasattr(
                scaler, 'transform') and callable(getattr(
                    scaler, 'fit_transform')) and callable(getattr(
                        scaler, 'transform')), \
                "'scaler' must have 'fit_transform' and 'transform'"

        self.splitter = splitter
        self.log_transform = log_transform
        self.color_model = color_model.upper()
        self.n_bins = n_bins
        self.data = []
        self.image_paths = []
        self.nsegs = splitter.nsegs if splitter is not None else 1

        if type(scaler) == bool:
            self.scaler = MinMaxScaler() if scaler else None
        else:
            self.scaler = scaler

        self.mranges = model_ranges[self.color_model]
        self.conversion = model_conversion[self.color_model]
        self.nchannel = model_channels[self.color_model]
        self.channels = list(range(self.nchannel))

    def fit_transform(self, image_paths, Y=None):
        """Processes images and fits the scaler"""
        return self.process(image_paths, fit=True)

    def transform(self, image_paths, Y=None):
        """Processes images using the previously fit scaler"""
        return self.process(image_paths, fit=False)

    
    def process(self, image_paths, fit=True):
        """Reads in images and outputs color density features

        Parameters
        ---------
        image_paths: iterable of str
            Iterable where each entry is a str that is
            a path to an image file
        fit: bool
            if True we fit the scaler, otherwise do not fit,
            just use scaler to transform
        """

        self.image_paths = image_paths  # save image paths
        self.read_data()  # read in those images

        if self.scaler is not None:
            if fit:  # fit and apply scaler
                self.data = self.scaler.fit_transform(self.data)
            else:  # apply scaler
                self.data = self.scaler.transform(self.data)

        return self.data

    def read_data(self):
        """Reads in images and calculates color histograms"""

        # first check that all images exist on disk
        for path in self.image_paths:
            if os.path.isfile(path) == False:
                raise ValueError('this file does not exist: ' + path)

        # data = [read_image(path) for path in self.image_paths]

        data = Parallel(n_jobs=os.cpu_count(), backend='loky')(
            delayed(self.read_image)(p) for p in self.image_paths)

        self.data = array(data)  # convert list to array

    def read_image(self, path):
        ''' Reads image file from a give file path'''

        image = imread(path)  # read image
        image = cvtColor(image, self.conversion)  # convert
        # if a radial splitter was defined, apply it here
        if self.splitter is None:
            segs = [image]
        else:
            segs = self.splitter.split(image)
        return ravel([self.color_histogram(s) for s in segs])

    def color_histogram(self, image):
        """Calculates the color histogram from an image (or pixel subset)"""

        if self.nchannel == 1:  # only 1 dimension if grayscale
            image = [image]
        elif image.ndim == 2:  # only 2 dimension if this is a pixel subest
            image = [image[:, c] for c in self.channels]
        else:  # 3 dimension otherwise
            image = [image[:, :, c] for c in self.channels]

        hist = calcHist(images=image,
                        channels=self.channels,
                        histSize=[self.n_bins] * self.nchannel,
                        mask=None,
                        ranges=self.mranges).flatten()

        # apply log transformation here if chosen
        if self.log_transform:
            hist = log1p(hist)

        return hist / hist.sum()  # normalise to be density
