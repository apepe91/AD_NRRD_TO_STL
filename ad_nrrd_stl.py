import numpy as np
import torch
import pyvista as pv
import scipy
import os
import mcubes
from tqdm import tqdm
from skimage.measure import marching_cubes
import pyrender, trimesh, pyvista
import open3d as o3d
import time
import open3d as o3d

import nrrd
import torchio as tio


import pyacvd


def covert_h5():
    pyvista.global_theme.background = 'white'
    resample = tio.Resample(0.45)
    for i in range(25,41):
        image = tio.ScalarImage("truelumen"+str(i)+".seg.nrrd")
        image = resample(image).data.numpy().astype(int)[0]
        max = image.max() * 0.2
        image = np.where(image>=max,1,0)

        imageF = tio.ScalarImage("falselumen"+str(i)+".seg.nrrd")
        imageF = resample(imageF).data.numpy().astype(int)[0]


        t = [27,26,13,12,37,10,8,9,3,31]
        if t.count(i) > 0:
            imageT = tio.ScalarImage("thrombus" + str(i) + ".seg.nrrd")
            #imageT = scipy.ndimage.morphology.binary_dilation(np.where(imageT, 1, 0), iterations=1).astype(int)
            imageT = resample(imageT).data.numpy().astype(int)[0]
            imageF = imageT + imageF

        imageF = np.where(imageF, 1, 0)

        imageDil = scipy.ndimage.morphology.binary_dilation(np.where(image,1,0),iterations=1).astype(int)
        imageF = imageF - imageDil
        imageF = np.where(imageF > 0, 1, 0)
        image = (image + imageF).astype(bool)

        mcs_vert, mcs_tri = mcubes.marching_cubes(image, 0)
        mcs_mesh = o3d.geometry.TriangleMesh()
        mcs_mesh.vertices = o3d.utility.Vector3dVector(mcs_vert)
        mcs_mesh.triangles = o3d.utility.Vector3iVector(mcs_tri)

        otmesh = trimesh.Trimesh(np.asarray(mcs_mesh.vertices), faces=np.asarray(mcs_mesh.triangles))
        pmesh = pv.wrap(otmesh)
        clus = pyacvd.Clustering(pmesh)
        clus.subdivide(3)
        clus.cluster(12000)
        remesh = clus.create_mesh()
        remesh.save("mesh" + str(i) + ".stl")




if __name__ == '__main__':
    covert_h5()