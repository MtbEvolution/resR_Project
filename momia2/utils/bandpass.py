import numpy as np
from scipy import fftpack
import warnings


def fft(img, subtract_mean=True):
    """
    fast Fourier transform module
    :param img: input image
    :param subtract_mean:
    :return: FFT transformed image
    """
    warnings.filterwarnings("ignore")
    if subtract_mean:
        img = img - np.mean(img)
    return fftpack.fftshift(fftpack.fft2(img))


def fft_reconstruction(fft_img, bandpass_filters):
    """
    reconstruct image after FFT band pass filtering.
    :param fft_img: input FFT transformed image
    :param filters: low/high frequency bandpass filters
    :return: bandpass filtered, restored phase contrast image
    """

    warnings.filterwarnings("ignore")
    if len(bandpass_filters) > 0:
        for f in bandpass_filters:
            try:
                fft_img *= f
            except:
                raise ValueError("Illegal input filter found, shape doesn't match?")
    output = fftpack.ifft2(fftpack.ifftshift(fft_img)).real
    output -= output.min()
    return output


def make_centered_mesh(pixel_microns=0.065, 
                       img_width=2048, 
                       img_height=2048):

    """
    modified to make generic, 20210821
    :param pixel_microns: pixel unit length
    :param img_width: width of image by pixel
    :param img_height: height of image by pixel
    :param bandpass_widths: series of 1/f where f is the lower bound of high frequency signal
    :return: bandpass filters

    """
    
    # create mesh
    u_max = round(1 / pixel_microns, 3) / 2
    v_max = round(1 / pixel_microns, 3) / 2
    u_axis_vec = np.linspace(-u_max / 2, u_max / 2, img_width)
    v_axis_vec = np.linspace(-v_max / 2, v_max / 2, img_height)
    u_mat, v_mat = np.meshgrid(u_axis_vec, v_axis_vec)
    centered_mesh = np.sqrt(u_mat ** 2 + v_mat ** 2)
    return centered_mesh


def bandpass_filter(centered_mesh,bandpass_width,highpass=True):
    """
    generic discrete bandpass filter
    """
    shape = centered_mesh.shape
    if bandpass_width==0:
        return np.ones((shape[1], shape[0]))
    elif highpass:
        return np.e ** (-(centered_mesh * bandpass_width) ** 2)
    else:
        return 1 - np.e ** (-(centered_mesh * bandpass_width) ** 2)
    
    
def dual_bandpass(img,
                  pixel_microns=0.065,
                  min_structure_scale=0.2,
                  max_structure_scale=20):
    """
    dual-bandpass image processing
    """
    if min_structure_scale>=max_structure_scale:
        raise ValueError('Minimal structure size (highpass) should be smaller \nthan maximal structure size (lowpass)')
    
    fft_img = fft(img)
    img_height, img_width = img.shape
    freq_domain = make_centered_mesh(pixel_microns,
                                     img_width=img_width,
                                     img_height=img_height)
    highpass_filter = bandpass_filter(freq_domain,min_structure_scale,True)
    lowpass_filter = bandpass_filter(freq_domain,max_structure_scale,False)
    
    # dual-bandpass filter
    filtered = fft_reconstruction(fft_img,(highpass_filter,lowpass_filter))
    return filtered.astype(img.dtype)
                                