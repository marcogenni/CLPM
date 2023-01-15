#!/usr/bin/env python3

import torch
import numpy as np
import os
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize


def integrand_torch(t, a, b, c):
    """
    Function that appears in the integral of the CLPM projection model. The values below follow the notation from the paper.
    :param t: integral variable
    :param a: S_{ij}^{(hh)}
    :param b: S_{ij}^{(gh)} + S_{ij}^{(hg)}
    :param c: S_{ij}^{(gg)}
    :return: function value.
    """
    return torch.exp(a*t**2 + b*t*(1-t) + c*(1-t)**2)

def integrate_simpsons (a, b, c, s):
    """
    Composite Simpson's rule 1/3 to approximate the integral that turns up in the CLPM projection model.
    :param a: same as in integrand_torch
    :param b: same as in integrand_torch
    :param c: same as in integrand_torch
    :param s: number of points to use for integral approximations
    :return: approximate value of integral
    """
    step_size = 1/s
    res = integrand_torch(0, a, b, c) + integrand_torch(1, a, b, c)
    for k in range(1, s, 1):
        if k % 2 != 0: res += 4 * integrand_torch(step_size*k, a, b, c)
        if k % 2 == 0: res += 2 * integrand_torch(step_size*k, a, b, c)
    return res / 3 / s
    
def make_video(outvid, images, fps=2, size=(600, 450), is_color=True, format="mp4v"):
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

def get_sub_graph(edgelist,
                  n_hubs=2,
                  type_of='friendship',
                  n_sub_nodes=100
                  ):
    """

    Parameters
    ----------
    edgelist : The interactions edgelist
    n_hubs : An integer equal to the number of hubs whose friends will be considered for the plot. The default is 2.
    type_of : A string, either 'friendship' or 'degree', designating the way the sub-graph will be extracted. The default is 'friendship'.
    n_sub_nodes : An integer indicating the number of nodes of the extracted sub-graph (ignored if type = 'friendship'). The default is 100.
    Returns
    -------
    sub_nodes: The array containing the subnodes forming the sub-graph.
    edgelist : The edgelist corresponding to the sub-graph.

    """
    n_nodes = np.max(edgelist['receiver']) + 1
    Adj = np.zeros(shape=(n_nodes, n_nodes))
    for idx in range(len(edgelist)):
        sender = edgelist.iloc[idx, 1]
        receiver = edgelist.iloc[idx, 2]
        Adj[sender, receiver] += 1
        Adj[receiver, sender] += 1

    deg = np.sum(Adj, axis=1)
    if type_of == 'friendship':
        # Three most active nodes
        hubs = np.argsort(-deg)[0:n_hubs]
        sAdj1 = Adj[hubs, :]
        tmp = np.sum(sAdj1, 0)
        sub_nodes = np.where(tmp != 0)
        pos_1 = np.isin(edgelist['sender'], sub_nodes)
        pos_2 = np.isin(edgelist['receiver'], sub_nodes)
        pos_3 = pos_1 & pos_2
        pos_final = np.where(pos_3 == True)[0]
        edgelist = edgelist.iloc[pos_final, :]
        return sub_nodes[0], edgelist
    elif type_of == 'degree':
        sub_nodes = np.argsort(-deg)[0:n_sub_nodes]
        sub_nodes = np.sort(sub_nodes)
        pos_1 = np.isin(edgelist['sender'], sub_nodes)
        pos_2 = np.isin(edgelist['receiver'], sub_nodes)
        pos_3 = pos_1 & pos_2
        pos_final = np.where(pos_3 == True)[0]
        edgelist = edgelist.iloc[pos_final, :]
        return sub_nodes, edgelist
    else:
        print('Error: unknown type_of!', sys.stderr)

def edgelist_conversion(edgelist_, sub_nodes, n_nodes):
    """
    This function takes the new edgelist output by get_sub_graph and renames both senders and receivers in  such a way to have a continuous list of nodes

    Parameters
    ----------
    edgelist_ : The new edgelist related to the subgraph extracted by get_sub_graph.
    sub_nodes : The subset of the original nodes' set forming the sub-graph.
    n_nodes : The number of nodes in the **orginal** graph.

    Returns
    -------
    The new edgelist with renamed senders/receivers and a conversion table for nodes.
    """
    new_n_nodes = len(sub_nodes)
    conversion = np.repeat(-1, n_nodes)
    conversion[sub_nodes] = np.arange(0, new_n_nodes)
    edgelist_['sender'] = conversion[edgelist_['sender'].values]
    edgelist_['receiver'] = conversion[edgelist_['receiver'].values]
    return edgelist_, conversion

