''' Class used to split an image into radial segments '''

from collections import Iterable
from cv2 import GaussianBlur, fillConvexPoly, bitwise_and
import numpy as np


class RadialSplitter(object):
    """Class that can split the pixels of an image into radial segments"""

    def __init__(self,
                 nrings=1,
                 nqslices=2,
                 blur=False,
                 blur_size=(9, 9)
                 ):
        """Constructs the attributes of the splitter, which can then be used to
            split an image in X radial segment using the split method
            where X = rings * qslices * 4

            Parameters
            ----------
            nrings: int
                Number of rings to split an image into before slicing
            nqslices: int
                Number of slices to split each quarter of the image into
            blur: bool
                Boolean, if True apply a blur to the image before splitting
            blur_size: tuple
                Tuple of two int, size of blur to apply (if blur=True)
        """
        for (v, s) in [(nrings, 'nrings'), (nqslices, 'nqslices')]:
            assert isinstance(v, int), "'" + s + "' must be int"
        assert isinstance(blur, bool), "'blur' must be boolean"
        assert isinstance(
            blur_size, Iterable), "'blur_size' must be a list-like iterable"
        assert len(blur_size) == 2, "'blur_size' must be of length 2"
        assert (type(blur_size[0]) == int) and (
            type(blur_size[1]) == int), "'blur_size' elements must be int"

        self.nrings = nrings
        self.nqslices = nqslices
        self.blur = blur
        self.blur_size = blur_size
        self.nsegs = 4 * nrings * nqslices
        self.image = None
        self.vertices = None
        self.segments = []

    def split(self, image):
        """Given an image in numpy array format, returns the pixels of the
            image split into X segments (X = rings * qslices * 4) """

        # save the image, optionally blurring first
        self.image = GaussianBlur(
            image, self.blur_size, 0) if self.blur else image

        self.getVertices()  # get the vertices of each segment

        self.segments = []
        for r in range(self.nrings):  # for each ring
            for i in range(4 * self.nqslices):  # for each segment in this ring
                # get pixels for this segment
                self.segments.append(self.getSegmentPixels(i, r))

        return self.segments

    def getVertices(self):
        """Given an image of a certain size, and the desired number of rings
            and slices, returns the vertices of all segments,
            arranged into rings"""

        xy = np.array(self.image.shape[:2])  # height & width of image
        center = np.array([xy / 2])  # center of image
        # replicate for each segment + 1
        center_coords = list(center) * (1 + 4 * self.nqslices)
        # proportion of the quarter that each qslice takes up
        qinterval = 1 / self.nqslices
        # list to define qslice boundaries
        qfractions = np.arange(0, 1 + qinterval, qinterval)
        # used to split rings into equal areas, the outer ring
        sizeDenominators = self.getSizeDenominators()

        vertices = []
        for denom in sizeDenominators:
            ring_xy = xy / denom  # height and width of this ring
            #  upper left co-ordinate of this ring
            ring_upperleft = (xy - ring_xy) / 2
            # get keypoints on this ring needed for segment vertices
            vertices.append(self.getRingKeypoints(
                ring_xy, ring_upperleft, qfractions))
        vertices.append(center_coords)  # add center co-oords
        # convert to array of integers
        vertices = np.array(vertices).astype(int)
        self.vertices = vertices

    def getSizeDenominators(self):
        """Given a number of rings, returns the list of denominators with which
            to divide the height & width of the image by in order to return
            rings of equal area. Note that the area of ring i will exclude the
            area of any smaller rings contained within it."""

        countDown = np.arange(self.nrings, 0, -1)  # sequence [nrings -> 1]
        # denominators that could be used to define equal areas
        relAreaDenominator = self.nrings / countDown
        # square root to get how much we need to adjust the width & height by
        return np.sqrt(relAreaDenominator)

    def getRingKeypoints(self, xy, upperleft, fractions):
        """Returns 4 * nqlices different keypoints for this ring (the first
            keypoint repeated meaning the list returned is of
            length 1 + 4 * nqslices)

        Parameters
            ----------
            xy: numpy array
                Vector of length two, the height & width of the ring
            upperleft: numpy array
                Vector of length two, the upper left co-ordinate of the ring
            fractions: numpy array
                Vector of length equal to nqslices, the fraction of height
                & width for each keypoint needed to define every segment
        """

        n = len(fractions) - 2  # required to avoid duplicating keypoints
        # get list of X and Y positions
        x_list, y_list = np.array([upperleft + xy * f for f in fractions]).T
        # upper (nqlsices + 1 positions needed for upper only)
        upper = [[x, y_list[0]] for x in x_list]
        # right
        right = [[x_list[-1], y] for y in y_list[1:]]
        # lower
        lower = [[x, y_list[-1]] for x in x_list[n::-1]]
        # left
        left = [[x_list[0], y] for y in y_list[n::-1]]
        # return in a single list
        return np.concatenate([upper, right, lower, left])

    def getSegmentPixels(self, i, r):
        """Returns the pixles of the i'th segment of the r'th ring"""

        # get vertices from relevant rings
        outer, inner = self.vertices[r], self.vertices[r + 1]

        # get the four vertices of this segment
        vertices = np.int32([outer[i], outer[i + 1],
                             inner[i + 1], inner[i]])[:, :: -1]

        # remove edge pixels to avoid overlap with other segments
        vertices[vertices == vertices.min(axis=0)] += 1
        vertices[vertices == vertices.max(axis=0)] -= 1

        # create mask and fill the segment area
        mask = np.array(np.zeros(self.image.shape), dtype=np.uint8)
        fillConvexPoly(mask, vertices, (255, 255, 255))

        # mask the original image
        masked_image = bitwise_and(self.image, mask)

        # return the pixels that aren't masked
        return masked_image[mask.sum(axis=2) > 0]
