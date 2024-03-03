import numpy as np
from sklearn.neighbors import NearestNeighbors

def fit_transform(source_pts, target_pts):
    dims = source_pts.shape[1]

    # recentre
    centroid_source = np.mean(source_pts, axis=0)
    centroid_target = np.mean(target_pts, axis=0)
    centered_source = source_pts - centroid_source
    centered_target = target_pts - centroid_target

    # calculate rotation
    matrix_H = np.dot(centered_source.T, centered_target)
    U, S, Vt = np.linalg.svd(matrix_H)
    rotation_matrix = np.dot(Vt.T, U.T)

    # avoid reflection
    if np.linalg.det(rotation_matrix) < 0:
       Vt[dims-1,:] *= -1
       rotation_matrix = np.dot(Vt.T, U.T)

    # calculate translation
    translation_vector = centroid_target.T - np.dot(rotation_matrix,centroid_source.T)

    transform_matrix = np.identity(dims+1)
    transform_matrix[:dims, :dims] = rotation_matrix
    transform_matrix[:dims, dims] = translation_vector

    return transform_matrix, rotation_matrix, translation_vector


def find_nearest_neighbor(src_points, dest_points):
    neighbor_finder = NearestNeighbors(n_neighbors=1)
    neighbor_finder.fit(dest_points)
    distances, indices = neighbor_finder.kneighbors(src_points, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(source, target, init_pose=None, max_iterations=20, tolerance=0.001):
    dims = source.shape[1]

    # homogeneous coordinates
    src_homogeneous = np.ones((dims+1,source.shape[0]))
    tgt_homogeneous = np.ones((dims+1,target.shape[0]))
    src_homogeneous[:dims,:] = np.copy(source.T)
    tgt_homogeneous[:dims,:] = np.copy(target.T)

    # apply initial pose
    if init_pose is not None:
        src_homogeneous = np.dot(init_pose, src_homogeneous)

    prev_error = 0

    for iter_count in range(max_iterations):
        # find the nearest neighbors
        distances, indices = find_nearest_neighbor(src_homogeneous[:dims,:].T, tgt_homogeneous[:dims,:].T)

        # calculate the transformation
        transform_matrix,_,_ = fit_transform(src_homogeneous[:dims,:].T, tgt_homogeneous[:dims,indices].T)

        # apply the transformation
        src_homogeneous = np.dot(transform_matrix, src_homogeneous)

        # calculate the error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # get the final transformation
    final_transform,_,_ = fit_transform(source, src_homogeneous[:dims,:].T)

    return final_transform, distances, iter_count
