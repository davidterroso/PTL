import numpy as np
import os.path as path
import random as rnd
import nutsml.datautil as ut
import nutsml.imageutil as ni
from skimage.morphology import disk, rectangle, closing, opening, binary_closing, convex_hull_image
from skimage.filters.rank import entropy
import matplotlib.pyplot as plt
from nutsflow import Nut, NutFunction, nut_processor, nut_function
from nutsflow.common import as_set
import skimage
import skimage.transform
from skimage.color import rgb2gray
import skimage.io as sio
import os.path
import scipy
from hyper_parameters import *
from PIL import Image
import matplotlib.patches

SHAPE_MULT = {1024: 2., 496: 1., 650: 0.004 / 0.0035, 885: 0.004 / 0.0026}


def calculate_oct_roi_mask(img, tresh=1e-10):
    """
        Calculate the interesting region MASK of the image using entropy
        :param img: 
        :param tresh: entropy cutoff threshold
        :return: mask of the interesting region (MASK = 1 interesting)
        """
    if img.ndim == 2:
        im_slice = skimage.img_as_float(img.astype(np.float32) / 128. - 1.)
    elif img.ndim == 3:
        im_slice = skimage.img_as_float(img[:, :, 1].astype(np.float32) / 128. - 1.)
    assert img.ndim in {2, 3}
    im_slice_ = entropy(im_slice, disk(11))
    im_slice_ = im_slice_ / (np.max(im_slice_) + 1e-16)
    im_slice_ = np.asarray(im_slice_ > tresh, dtype=np.int8)
    selem = disk(35)
    im_slice_ = binary_closing(im_slice_, selem=selem)
    im_slice_ = convex_hull_image(im_slice_)

    plt.imshow(im_slice, cmap='gray')
    plt.imshow(im_slice_, cmap='jet', alpha=0.5)
    plt.pause(.1)

    return im_slice_


def calculate_oct_y_range(img, tresh=1e-10):
    """
    Calculate the interesting y region of the image using entropy
    :param img: 
    :param tresh: entropy cutoff threshold
    :return: min and maximum y index containing the interesting region
    """

    if img.ndim == 2:
        im_slice = skimage.img_as_float(img.astype(np.float32) / 128. - 1.)
    elif img.ndim == 3:
        im_slice = skimage.img_as_float(img[:, :, 1].astype(np.float32) / 128. - 1.)
    assert img.ndim in {2, 3}
    im_slice_ = entropy(im_slice, disk(11))
    p_ = np.mean(im_slice_, axis=1)
    p_ = p_ / (np.max(p_) + 1e-16)
    p_ = np.asarray(p_ > tresh, dtype=np.int)
    inx = np.where(p_ == 1)

    # im_slice_ = im_slice_ / (np.max(im_slice_) + 1e-16)
    # im_slice_ = np.asarray(im_slice_ > tresh, dtype=np.int)
    #
    #
    # plt.subplot(1,3,1)
    # plt.imshow(im_slice)
    # plt.subplot(1,3,2)
    # plt.imshow(im_slice_)
    # plt.subplot(1,3,3)
    # selem = disk(35)
    # im_slice_ = binary_closing(im_slice_,selem=selem)
    # plt.imshow(im_slice_)
    # plt.pause(.1)

    # y = list()
    # for i in range(0, im_slice_.shape[1]):
    #     if np.any(im_slice_[:,i]==1):
    #         yy = np.where(im_slice_[:,i]==1)[0][0]
    #     else:
    #         yy = 0
    #     y.append(yy)
    #
    # y = scipy.signal.medfilt(y, kernel_size=81)
    # p_ = np.zeros(p_.shape, dtype=np.float32)
    # if len(inx[0]) > 0:
    #     p_[inx[0][0]:inx[0][-1]] = 512.
    # plt.imshow(img)
    # print img.shape
    # plt.plot(p_, range(0, len(p_)), color='red')
    # plt.plot(range(0, len(y)), y, color='blue')
    # plt.pause(1)
    # plt.clf()

    if len(inx[0]) > 0:
        return np.min(inx[0]), np.max(inx[0])
    else:
        return 0, img.shape[0]


def sample_oct_patch_centers(roimask, pshape, npos, pos=1, neg=0):
    PYINX = 0
    PXINX = 1
    h, w = roimask.shape

    x = range(int(pshape[PXINX] / 2), w - int(pshape[PXINX] / 2), 10)
    x_samples = np.random.choice(x, npos * 2, replace=False)
    np.random.shuffle(x_samples)
    c_hold = list()
    for x in x_samples:
        nz = np.nonzero(roimask[:, x] == pos)[0]
        if len(nz) > 0:
            y = int(float(np.min(nz)) + (
                (float(np.max(nz)) - float(np.min(nz))) / 2.) + np.random.uniform()) + np.random.randint(-10, 10)
            if (y - pshape[PYINX] / 2) < 1:
                y = int(pshape[PYINX] / 2) + 1
            elif (y + pshape[PYINX] / 2) >= h:
                y = h - int(pshape[PYINX] / 2) - 1

            c_hold.append([y, x, 1])
        if len(c_hold) >= npos:
            break

    return c_hold


def sample_patches_entropy_mask(img, mask=None, roimask=None, pshape=(224, 224), npos=10, nneg=1, pos=255, neg=0,
                                patch_border=12):
    """
    Generate patches from the interesting region of the OCT slice
    :param img: oct image slice
    :param mask: oct segmentation GT
    :param roimask: oct ROI
    :param pshape: patch shape
    :param npos: Number of patches to sample from interesting region
    :param nneg: Number of patches to sample from non interesting region
    :param pos: Mask value indicating positives
    :param neg: Mask value indicating negative
    :param patch_border: boder to ignore when creating IRF,SRF,PED labels for patches (ignore border pixels for predicting labels)
    :return: 
    """

    # y_min, y_max = calculate_oct_y_range(img)
    # roi_mask = np.zeros(mask.shape, dtype=np.int8)
    # # print y_min, y_max
    # roi_mask[y_min:y_max, :] = pos
    # # TODO : Why 32?
    # roi_mask[y_min:y_min + 32, :] = 0
    # roi_mask[y_max - 32:y_max, :] = 0
    # # plt.imshow(img)
    # # print np.max(roi_mask), np.min(roi_mask)

    roimask = roimask.astype(np.int8)
    # it = ni.sample_patch_centers(roimask, pshape=pshape, npos=npos, nneg=nneg, pos=pos, neg=neg)
    it = sample_oct_patch_centers(roimask, pshape=pshape, npos=npos, pos=pos, neg=neg)
    # TODO : check code
    it1 = ni.sample_patch_centers(roimask, pshape=pshape, npos=int(float(npos)*.2), nneg=0, pos=pos, neg=neg)
    for r, c, l in it1:
        it.append([r, c, l])
    # h,w = roimask.shape
    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot(111, aspect='equal')
    for r, c, label in it:
        img_patch = ni.extract_patch(img, pshape, r, c)
        mask_patch = ni.extract_patch(mask, pshape, r, c)
        label_IRF = np.int8(np.any(mask_patch[patch_border:-patch_border, patch_border:-patch_border] == IRF_CODE))
        label_SRF = np.int8(np.any(mask_patch[patch_border:-patch_border, patch_border:-patch_border] == SRF_CODE))
        label_PED = np.int8(np.any(mask_patch[patch_border:-patch_border, patch_border:-patch_border] == PED_CODE))
        yield img_patch, mask_patch, label_IRF, label_SRF, label_PED

        # ax1.imshow(img, cmap='gray')
        # ax1.imshow(roimask, cmap='jet', alpha=0.5)
        # ax1.plot(c, r, 'ro')
        # # print h,r, pshape[0], h-(r+int(pshape[0]/2))
        # ax1.add_patch(matplotlib.patches.Rectangle(xy=(c-int(pshape[1]/2), (r-int(pshape[0]/2))), width=pshape[1], height=pshape[0], fill=False, edgecolor="red"))

    # plt.pause(1)
    # plt.close()


def sample_patches_retouch_mask(img, mask=None, pshape=(224, 224), npos=10, nneg=1, pos=255, neg=0, patch_border=12):
    """
    Generate patches from the interesting region of the OCT slice
    :param img: oct image slice
    :param mask: oct segmentation GT
    :param pshape: patch shape
    :param npos: Number of patches to sample from interesting region
    :param nneg: Number of patches to sample from non interesting region
    :param pos: Mask value indicating positives
    :param neg: Mask value indicating negative
    :param patch_border: boder to ignore when creating IRF,SRF,PED labels for patches (ignore border pixels for predicting labels)
    :return: 
    """
    assert mask.dtype == np.uint8
    roi_mask = np.logical_not(mask == 0)

    it = ni.sample_patch_centers(roi_mask, pshape=pshape, npos=npos, nneg=nneg, pos=pos, neg=neg)
    for r, c, label in it:
        img_patch = ni.extract_patch(img, pshape, r, c)
        mask_patch = ni.extract_patch(mask, pshape, r, c)
        label_IRF = np.int8(np.any(mask_patch[patch_border:-patch_border, patch_border:-patch_border] == IRF_CODE))
        label_SRF = np.int8(np.any(mask_patch[patch_border:-patch_border, patch_border:-patch_border] == SRF_CODE))
        label_PED = np.int8(np.any(mask_patch[patch_border:-patch_border, patch_border:-patch_border] == PED_CODE))
        yield img_patch, mask_patch, label_IRF, label_SRF, label_PED
        # plt.plot(c, r, 'ro')

        # plt.pause(.1)
        # plt.clf()


@nut_processor
def ImagePatchesByMaskRetouch(iterable, imagecol, maskcol, IRFcol, SRFcol, PEDcol, roicol, pshape, npos,
                              nneg, pos=255, neg=0, patch_border=12, use_entropy=False):
    """
    :param iterable: iterable: Samples with images
    :param imagecol: Index of sample column that contain image
    :param maskcol: Index of sample column that contain mask
    :param IRFcol: Index of sample column that contain IRF label
    :param SRFcol: Index of sample column that contain SRF label
    :param PEDcol: Index of sample column that contain PED label
    :param pshape: Shape of patch
    :param npos: Number of patches to sample from interesting region
    :param nneg: Number of patches to sample from outside interesting region
    :param pos: Mask value indicating positives
    :param neg: Mask value indicating negativr
    :param patch_border: boder to ignore when creating IRF,SRF,PED labels for patches (ignore border pixels for predicting labels)
    :return: Iterator over samples where images and masks are replaced by image and mask patches
        and labels are replaced by labels [0,1] for patches
    """

    for sample in iterable:
        image, mask, roim = sample[imagecol], sample[maskcol], sample[roicol]
        img_height = image.shape[0]

        if image.shape[:2] != mask.shape:
            raise ValueError('Image and mask size don''t match!')

        if use_entropy:
            # TODO : Further test downscaling strategy
            if img_height > 700:
                # print 'Cirrus image'
                it = sample_patches_entropy_mask(image, mask, roimask=roim, pshape=(pshape[0] * 2, pshape[1]),
                                                 npos=npos, nneg=nneg, pos=pos,
                                                 neg=neg,
                                                 patch_border=patch_border)
            else:
                # print 'Spectralisis image'
                it = sample_patches_entropy_mask(image, mask, roimask=roim, pshape=pshape, npos=npos, nneg=nneg,
                                                 pos=pos, neg=neg,
                                                 patch_border=patch_border)
        else:
            if img_height > 700:
                # print 'Cirrus image'
                it = sample_patches_retouch_mask(image, mask, pshape=(pshape[0] * 2, pshape[1]), npos=npos, nneg=nneg,
                                                 pos=pos,
                                                 neg=neg,
                                                 patch_border=patch_border)
            else:
                # print 'Spectralisis image'
                it = sample_patches_retouch_mask(image, mask, pshape=pshape, npos=npos, nneg=nneg, pos=pos, neg=neg,
                                                 patch_border=patch_border)

        for img_patch, mask_patch, label_IRF, label_SRF, label_PED in it:
            outsample = list(sample)[:]
            if img_height > 700:
                # TODO : check if averaging is better than ignoring image rows
                # outsample[imagecol] = (
                # 0.5 * img_patch[0::2, :].astype(np.float32) + 0.5 * img_patch[1::2, :].astype(np.float32)).astype(
                #     np.int32)
                outsample[imagecol] = img_patch[0::2, :]
                temp = np.zeros((2, pshape[0], pshape[1]), dtype=np.int8)
                temp[0, :] = mask_patch[0::2, :]
                temp[1, :] = mask_patch[1::2, :]
                outsample[maskcol] = np.max(temp, axis=0)
            else:
                outsample[imagecol] = img_patch
                outsample[maskcol] = mask_patch

            outsample[IRFcol] = label_IRF
            outsample[SRFcol] = label_SRF
            outsample[PEDcol] = label_PED
            # TODO : does it need to yield roim

            yield tuple(outsample)


@nut_processor
def ImagePatchesByMaskRetouch_resampled(iterable, imagecol, maskcol, IRFcol, SRFcol, PEDcol, roicol, pshape, npos,
                                        nneg, pos=255, neg=0, patch_border=12, use_entropy=False):
    """
    :param iterable: iterable: Samples with images
    :param imagecol: Index of sample column that contain image
    :param maskcol: Index of sample column that contain mask
    :param IRFcol: Index of sample column that contain IRF label
    :param SRFcol: Index of sample column that contain SRF label
    :param PEDcol: Index of sample column that contain PED label
    :param pshape: Shape of patch
    :param npos: Number of patches to sample from interesting region
    :param nneg: Number of patches to sample from outside interesting region
    :param pos: Mask value indicating positives
    :param neg: Mask value indicating negativr
    :param patch_border: boder to ignore when creating IRF,SRF,PED labels for patches (ignore border pixels for predicting labels)
    :return: Iterator over samples where images and masks are replaced by image and mask patches
        and labels are replaced by labels [0,1] for patches
    """

    for sample in iterable:
        image, mask, roim = sample[imagecol], sample[maskcol], sample[roicol]
        img_height = image.shape[0]

        if image.shape[:2] != mask.shape:
            raise ValueError('Image and mask size don''t match!')

        assert img_height in {1024, 496, 650, 885}
        npshape = (int(pshape[0] * SHAPE_MULT[img_height]), pshape[1])
        if use_entropy:
            # TODO : Further test downscaling strategy
            it = sample_patches_entropy_mask(image, mask, roimask=roim, pshape=npshape, npos=npos, nneg=nneg, pos=pos,
                                             neg=neg, patch_border=patch_border)

        else:
            it = sample_patches_retouch_mask(image, mask, pshape=npshape, npos=npos, nneg=nneg,
                                             pos=pos, neg=neg, patch_border=patch_border)

        for img_patch, mask_patch, label_IRF, label_SRF, label_PED in it:
            outsample = list(sample)[:]

            if img_height == 496:
                outsample[imagecol] = img_patch
                outsample[maskcol] = mask_patch
            else:
                outsample[imagecol] = skimage.transform.resize(img_patch.astype(np.uint8), pshape, order=0, preserve_range=True).astype('uint8')
                # img_patch = Image.fromarray(np.uint8(img_patch))
                # img_patch_ = img_patch.resize(pshape, resample=Image.NEAREST)
                # outsample[imagecol] = np.asarray(img_patch_)

                # mask_patch = Image.fromarray(np.uint8(mask_patch), mode='L')
                # mask_patch_ = mask_patch.resize(pshape, resample=Image.NEAREST)
                # outsample[maskcol] = np.asarray(mask_patch_)
                outsample[maskcol] = skimage.transform.resize(mask_patch.astype(np.uint8), pshape, order=0, preserve_range=True).astype('uint8')

            outsample[IRFcol] = np.int8(np.any(outsample[maskcol][patch_border:-patch_border, patch_border:-patch_border] == IRF_CODE))
            outsample[SRFcol] = np.int8(np.any(outsample[maskcol][patch_border:-patch_border, patch_border:-patch_border] == SRF_CODE))
            outsample[PEDcol] = np.int8(np.any(outsample[maskcol][patch_border:-patch_border, patch_border:-patch_border] == PED_CODE))
            # TODO : does it need to yield roim

            yield tuple(outsample)


@nut_processor
def ImagePatchesForTest_resampled(iterable, imagecol, maskcol, IRFcol, SRFcol, PEDcol, roicol, pshape, npos,
                                        nneg, pos=255, neg=0, patch_border=12, use_entropy=False):
    """
    :param iterable: iterable: Samples with images
    :param imagecol: Index of sample column that contain image
    :param maskcol: Index of sample column that contain mask
    :param IRFcol: Index of sample column that contain IRF label
    :param SRFcol: Index of sample column that contain SRF label
    :param PEDcol: Index of sample column that contain PED label
    :param pshape: Shape of patch
    :param npos: Number of patches to sample from interesting region
    :param nneg: Number of patches to sample from outside interesting region
    :param pos: Mask value indicating positives
    :param neg: Mask value indicating negativr
    :param patch_border: boder to ignore when creating IRF,SRF,PED labels for patches (ignore border pixels for predicting labels)
    :return: Iterator over samples where images and masks are replaced by image and mask patches
        and labels are replaced by labels [0,1] for patches
    """

    for sample in iterable:
        image, mask, roim = sample[imagecol], sample[maskcol], sample[roicol]
        img_height = image.shape[0]

        if image.shape[:2] != mask.shape:
            raise ValueError('Image and mask size don''t match!')

        assert img_height in {1024, 496, 650, 885}
        npshape = (int(pshape[0] * SHAPE_MULT[img_height]), pshape[1])
        if use_entropy:
            # TODO : Further test downscaling strategy
            roimask = roim.astype(np.int8)
            nx_y = np.nonzero(np.sum(roimask, axis=1))[0]
            # print nx_y
            if len(nx_y) > 0:
                miny, maxy = float(np.min(nx_y)), float(np.max(nx_y))
                print (miny, maxy)
                y = int(miny + (maxy-miny)/2.)
                if not y > int(float(npshape[0])/2.):
                    y = int(float(npshape[0])/2.)
            else:
                y = int(float(npshape[0])/2)
            print (y)
            img_patch = image[int(y - npshape[0] // 2):int(y + npshape[0] // 2), :, :]
            mask_patch = mask[int(y - npshape[0] // 2):int(y + npshape[0] // 2), :]
            roi_patch = roim[int(y - npshape[0] // 2):int(y + npshape[0] // 2), :]
            # it = sample_patches_entropy_mask(image, mask, roimask=roim, pshape=npshape, npos=npos, nneg=nneg, pos=pos,
            #                                  neg=neg, patch_border=patch_border)

        else:
            print ("Error")
            # it = sample_patches_retouch_mask(image, mask, pshape=npshape, npos=npos, nneg=nneg,
            #                                  pos=pos, neg=neg, patch_border=patch_border)

        # for img_patch, mask_patch, label_IRF, label_SRF, label_PED in it:
        outsample = list(sample)[:]

        if img_height == 496:
            outsample[imagecol] = img_patch
            outsample[maskcol] = mask_patch
            outsample[roicol] = roi_patch
        else:
            outsample[imagecol] = skimage.transform.resize(img_patch.astype(np.uint8), pshape, order=0, preserve_range=True).astype('uint8')
            # img_patch = Image.fromarray(np.uint8(img_patch))
            # img_patch_ = img_patch.resize(pshape, resample=Image.NEAREST)
            # outsample[imagecol] = np.asarray(img_patch_)

            # mask_patch = Image.fromarray(np.uint8(mask_patch), mode='L')
            # mask_patch_ = mask_patch.resize(pshape, resample=Image.NEAREST)
            # outsample[maskcol] = np.asarray(mask_patch_)
            outsample[maskcol] = skimage.transform.resize(mask_patch.astype(np.uint8), pshape, order=0, preserve_range=True).astype('uint8')
            outsample[roicol] = skimage.transform.resize(roi_patch.astype(np.uint8), pshape, order=0,
                                                          preserve_range=True).astype('uint8')

        outsample[IRFcol] = np.int8(np.any(outsample[maskcol][patch_border:-patch_border, patch_border:-patch_border] == IRF_CODE))
        outsample[SRFcol] = np.int8(np.any(outsample[maskcol][patch_border:-patch_border, patch_border:-patch_border] == SRF_CODE))
        outsample[PEDcol] = np.int8(np.any(outsample[maskcol][patch_border:-patch_border, patch_border:-patch_border] == PED_CODE))
        # TODO : does it need to yield roim

        yield tuple(outsample)


def load_oct_image(filepath, as_grey=False, dtype='uint8', no_alpha=True):
    """
    Load three consecative oct slices given the filepath.

    Supported formats: tiff

    >>> img = load_oct_image('/home/truwan/DATA/retouch/pre_processed/oct_imgs/Spectralis_fe02f982b78218ab05c755c01c7876b1_035.tiff')
    >>> print img.shape
    (496, 512, 3)

    :param string filepath: Filepath to image file or numpy array.
    :param bool as_grey:
    :return: numpy array with shapes
             (h, w, 3) for oct slices of grayscale or monochrome,
             pixel values are in range [0,255] for dtype = uint8
    :rtype: numpy ndarray
    """
    # TODO : think about z axis of the spectralasis images, they have different spacing than other two
    if filepath.endswith('.npy'):  # image as numpy array
        print ("reading numpy OCT not yet implemented...")
        # arr = np.load(filepath).astype(dtype)
        # arr = rgb2gray(arr) if as_grey else arr
    else:
        # img_num=0 due to
        # https://github.com/scikit-image/scikit-image/issues/2406
        arr1 = np.expand_dims(sio.imread(filepath, as_gray=as_grey, img_num=0).astype(dtype), axis=-1)
        slice_num = int(filepath[:-5].split('_')[-1])
        if slice_num == 0:
            arr0 = arr1
        else:
            s0_filepath = filepath[:-8] + str(slice_num - 1).zfill(3) + '.tiff'
            arr0 = np.expand_dims(sio.imread(s0_filepath, as_gray=as_grey, img_num=0).astype(dtype), axis=-1)
        s2_filepath = filepath[:-8] + str(slice_num + 1).zfill(3) + '.tiff'
        if os.path.isfile(s2_filepath):
            arr2 = np.expand_dims(sio.imread(s2_filepath, as_gray=as_grey, img_num=0).astype(dtype), axis=-1)
        else:
            arr2 = arr1

        arr = np.concatenate([arr0, arr1, arr2], axis=-1)

    if arr.ndim == 3 and arr.shape[2] == 4 and no_alpha:
        arr = arr[..., :3]  # cut off alpha channel
    return arr


@nut_function
def ReadOCT(sample, columns, pathfunc=None, as_grey=False, dtype='uint8'):
    """
    Load OCT images for samples.

    Loads 3 consecative oct slices tif format.
    Images are returned as numpy arrays of shape (h, w, 3) for
    gray scale images.
    See custom_nuts.load_oct_image for details.

    :param tuple|list sample: ('nut_color', 1)
    :param None|int|tuple columns: Indices of columns in sample to be replaced
                              by image (based on image id in that column)
                              If None then a flat samples is assumed and
                              a tuple with the image is returned.
    :param string|function|None pathfunc: Filepath with wildcard '*',
      which is replaced by the imageid provided in the sample, e.g.
      'tests/data/img_formats/*.jpg' for sample ('nut_grayscale', 2)
      will become 'tests/data/img_formats/nut_grayscale.jpg'
      or
      Function to compute path to image file from sample, e.g.
      lambda sample: 'tests/data/img_formats/{1}.jpg'.format(*sample)
      or
      None, in this case the image id is take as filepath.
    :param bool as_grey: If true, load as grayscale image.
    :param dtype dtype: Numpy data type of the image.
    :return: Sample with image ids replaced by image (=ndarray)
            of shape (h, w, c) or (h, w)
    :rtype: tuple
    """

    def load(fileid):
        """Load image for given fileid"""
        if isinstance(pathfunc, str):
            filepath = pathfunc.replace('*', fileid)
        elif hasattr(pathfunc, '__call__'):
            filepath = pathfunc(sample)
        else:
            filepath = fileid
        return load_oct_image(filepath, as_grey=as_grey, dtype=dtype)

    if columns is None:
        return (load(sample),)  # image as tuple with one element

    colset = as_set(columns)
    elems = enumerate(sample)
    return tuple(load(e) if i in colset else e for i, e in elems)
