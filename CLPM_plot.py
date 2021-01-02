#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 13:52:41 2020

@author: marco and riccardo
"""

import math
import numpy as np
import torch

from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
cividis = cm.get_cmap('cividis', 12)

from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
import os

def fade_node_colors(dataset, Z, bending = 1):
    """
    Defines the colors of the nodes as values between 0 and 1 at each changepoint -- this will be used in the animation
    @param      dataset         the observed data
    @param      Z               the estimated latent positions at each of the changepoints
    @param      bending         the values are amplified or reduced according to this positive value: >1 will increase variability, <1 will decrease variability
    """
    colors = np.zeros((dataset.n_nodes, dataset.n_changepoints))
    for t in range(dataset.n_changepoints):
        for i in range(dataset.n_nodes):
            for j in range(dataset.n_nodes):
                if i != j:
                    colors[i,t] += (  1 + (Z[i,0,t]-Z[j,0,t])**2 + (Z[i,1,t]-Z[j,1,t])**2  ) / (dataset.n_nodes-1)
    colors = colors**bending
    colors /= colors.min()
    colors -= 1
    colors /= colors.max()# now we have values between 0 and 1
    colors *= -1
    colors += 1# now we have 1 minus the previous values
    return colors

def fade_node_sizes(dataset, bending = 1):
    """
    Defines the sizes of the nodes as values between 0 and 1 at each changepoint -- this will be used in the animation
    @param      dataset         the observed data
    @param      bending         the values are amplified or reduced according to this positive value: >1 will increase variability, <1 will decrease variability
    """
    sizes = np.zeros((dataset.n_nodes, dataset.n_changepoints))
    for edge in range(dataset.n_entries):
        i = dataset.interactions[edge,0].item()
        j = dataset.interactions[edge,1].item()
        kappa = (dataset.timestamps[edge] // dataset.segment_length).int().item()
        sizes[i,kappa] += 2
        sizes[j,kappa] += 2
        if kappa >= 1: 
            sizes[i,kappa-1] += 1
            sizes[j,kappa-1] += 1
        if kappa <= dataset.n_changepoints - 2: 
            sizes[i,kappa+1] += 1
            sizes[j,kappa+1] += 1
    sizes = sizes**bending
    sizes /= sizes.min()
    sizes -= 1
    sizes /= sizes.max()
    return sizes

def create_snaps(Z, changepoints, frames_btw, node_colors, node_sizes, dpi = 100):
    """
    Creates a sequence of images that will compose the video.
    @param      Z               the latent positions from the output of the fitting algorithm
    @param      changepoints    the changepoints that are being used
    @param      frames_btw      the number of additional frames that are inserted inbetween any two changepoints, to make the transitions smoother
    @param      node_colors     how the nodes should be colored at each changepoint
    @param      node_sizes      the size of the nodes at each changepoint
    @param      dpi             resolution of the exported pdf images
    @return                     a list of strings indicating the path to the images that must be collated in the video
    """
    n_nodes = Z.shape[0]
    n_dim = Z.shape[1]
    n_cps = Z.shape[2]
    n_frames = frames_btw * (n_cps-1) + n_cps
    cps_large = np.zeros(n_frames)
    cps_large[n_frames-1] = changepoints[n_cps-1]
    pos = np.zeros((n_nodes, n_dim, n_frames))
    pos[:,:,n_frames-1] = Z[:,:,n_cps-1]
    colors_large = np.zeros((n_nodes, n_frames))
    colors_large[:,n_frames-1] = node_colors[:,n_cps-1]
    sizes_large = np.zeros((n_nodes, n_frames))
    sizes_large[:,n_frames-1] = node_sizes[:,n_cps-1]
    for frame in range(n_frames-1):
        cp0 = frame // (frames_btw+1)
        cp1 = (frame // (frames_btw+1)) + 1
        delta = (frame % (frames_btw+1)) / (frames_btw+1)
        cps_large[frame] = (1-delta)*changepoints[cp0] + delta*changepoints[cp1]
        for i in range(n_nodes):
            pos[i,0,frame] = (1-delta)*Z[i,0,cp0] + delta*Z[i,0,cp1]
            pos[i,1,frame] = (1-delta)*Z[i,1,cp0] + delta*Z[i,1,cp1]
            colors_large[i,frame] = (1-delta)*node_colors[i,cp0] + delta*node_colors[i,cp1]
            sizes_large[i,frame] = (1-delta)*node_sizes[i,cp0] + delta*node_sizes[i,cp1]
    pos_limit = np.abs(pos).max()
    for frame in range(n_frames):
        plt.figure()
        plt.title("Latent Positions at time " + str(round(cps_large[frame],2)), loc = "left")
        plt.xlim((-pos_limit,pos_limit))
        plt.ylim((-pos_limit,pos_limit))
        for idi in range(n_nodes):
            if frame >= 2: plt.plot([pos[idi,0,frame-2], pos[idi,0,frame-1]], [pos[idi,1,frame-2], pos[idi,1,frame-1]], 'k-', alpha = 0.4, color = cividis(colors_large[idi,frame]))
            if frame >= 1: plt.plot([pos[idi,0,frame-1], pos[idi,0,frame-0]], [pos[idi,1,frame-1], pos[idi,1,frame-0]], 'k-', alpha = 0.8, color = cividis(colors_large[idi,frame]))
            #if frame < n_frames-1: plt.plot([pos[idi,0,frame+0], pos[idi,0,frame+1]], [pos[idi,1,frame+0], pos[idi,1,frame+1]], 'k-', alpha = 0.3)
            #if frame < n_frames-2: plt.plot([pos[idi,0,frame+1], pos[idi,0,frame+2]], [pos[idi,1,frame+1], pos[idi,1,frame+2]], 'k-', alpha = 0.1)
            plt.plot(pos[idi,0,frame], pos[idi,1,frame], 'ro', markersize = 1 + sizes_large[idi,frame] * 8, alpha = 1, color = cividis(colors_large[idi,frame]))
        plt.savefig('results/snaps/snap_'+str(frame)+'.png', dpi = dpi)
        plt.close()
    images = []
    for i in range(n_frames):
        images.append('results/snaps/snap_'+str(i)+'.png')
    return images

def make_video(outvid, images, outimg = None, fps = 2, size = (600,450), is_color = True, format = "mp4v"):
    """
    Create a video from a list of images.
    @param      outvid      output video
    @param      images      list of images to use in the video
    @param      fps         frame per second
    @param      size        size of each frame
    @param      is_color    color
    @param      format      see http://www.fourcc.org/codecs.php
    @return                 see http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
    The function relies on http://opencv-python-tutroals.readthedocs.org/en/latest/.
    By default, the video will have the size of the first image.
    It will resize every image to this size before adding them to the video.
    """
    fourcc = VideoWriter_fourcc(*format)
    vid = None
    for image in images:
        if not os.path.exists(image):
            raise FileNotFoundError(image)
        img = imread(image)
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        vid.write(img)
    vid.release()
    return vid

def clpm_animation(outvid, Z, changepoints, frames_btw, node_colors, node_sizes, dpi, period, size = (1200,900), is_color = True, format = "mp4v"):
    """
    Combines the functions create_snaps and make_video to produce the animation for a fitted clpm
    @param      outvid          output video
    @param      Z               the latent positions from the output of the fitting algorithm
    @param      changepoints    the changepoints that are being used
    @param      frames_btw      the number of additional frames that are inserted inbetween any two changepoints, to make the transitions smoother
    @param      node_colors     how the nodes should be colored at each changepoint
    @param      node_sizes      the size of the nodes at each changepoint
    @param      dpi             resolution of the exported pdf images
    @param      period          time it takes on video to progress from one changepoint to the next
    @param      size            see make_video
    @param      is_color        see make_video
    @param      format          see make_video
    @return                     see make_video
    """
    images = create_snaps(Z, changepoints, frames_btw, node_colors, node_sizes, dpi)
    fps = (frames_btw+1) / period
    return make_video(outvid, images, None, fps, size, is_color, format)

