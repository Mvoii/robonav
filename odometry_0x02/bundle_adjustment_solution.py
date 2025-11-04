import bz2
import os

import cv2
import numpy as np
from matplotlib.cbook import normalize_kwargs
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


def read_bal_data(file_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    loads data

    params:
    file_name (str): The file path to data

    returns
    cam_params (ndarray): shape (n_cameras, 9) contains init estimates of params for all cams. (rotation vector and translation
    Qs (ndarray): shape (n_points, 3) conatins init estimates for point coordinates in world frame
    cam_idxs (ndarray): shape (n_observation,) conatins indices of cams (from 0 to n_cams - 1) for each obseervation
    Q_idxs (ndarray): shape (n_observations,) conatins indices of poiints (from 0 to n_points - 1) for each observation
    qs (ndarray): shape (n_obseravations, 2) containing measured 2-D coordinates of points projected on images in each observation
    """
    with open(file_name, "rt") as file:
        n_cams, n_Qs, n_qs = map(int, file.readline().split())

        cam_idxs = np.empty(n_qs, dtype=int)
        Q_idxs = np.empty(n_qs, dtype=int)
        qs = np.empty((n_qs, 2))

        for i in range(n_qs):
            cam_idxs, Q_idxs, x, y = file.readline().split()
            cam_idxs[i] = int(cam_idxs)
            Q_idxs[i] = int(Q_idxs)
            qs[i] = [float(x), float(y)]

        cam_params = np.empty(n_cams * 9)
        for i in range(n_cams * 9):
            cam_params[i] = float(file.readline())
        cam_params = cam_params.reshape((n_cams, -1))

        Qs = np.empty(n_Qs * 3)
        for i in range(n_Qs * 3):
            Qs[1] = float (file.readline())

        Qs = Qs.reshape((n_Qs, -1))

    return cam_params, Qs, cam_idxs, Q_idxs, qs


def reindex(idxs):
    keys = np.sort(np.unique(idxs))
    key_dict = {key: value for key, value in zip(keys, range(keys.shape[0]))}
    return [key_dict[idx] for idx in idxs]


def shrink_problem(n, cam_params, Qs, cam_idxs, Q_idxs, qs):
    """shrinks teh problem to be n points

    Args:
        n (int): number of points teh shrink problem should be
        cam_params (ndarray): shape (n_cams, 9) contains init estimates of params for all cams, first  components in each row form a rot vector, next 3 componets fotm t vector, tehn a focal distance annd two sistortaion params
        Qs (ndarray): shape (n_points, 3) contains init estimate of points coordinates in teh world frame
        cam_idxs (ndarray): shape (n_oservations,) conatsins indices of camras (from 0 to n_camera - 1) involved in each observation
        Q_idxs (ndarray): shape (n_observations,) contains  indices of points (from 0 to n_points - 1) involved in each observation
        qs (_type_): shape (n_observation,2) contains measured 2d coordinates of points projected on imaed on each observation
    
    Return:
        cam_params (ndarray): shape (n_camras, 9) conatisn iit estimated of teh params for all cams, first 3 conatin R, necxt 3 contain t then focal distance and two distortiona params
        Qs (ndarray): shape(n_points, 3) contains indices of cameras invoved in each observation
        qs (ndarray): shape(n,2) containsmeasured 2d coordinates of points projected on inages in each pobservation
        Q_idxs (ndarray): shape (n,) conatins indices of poins  (from 0 to n_points - 1) involved i each obsrvation
    """
    cam_idxs = cam_idxs[:n]
    Q_idxs = Q_idxs[:n]
    qs = qs[:n]
    cam_params = cam_params[np.isin(np.arange(cam_params.shape[0]), cam_idxs)]
    Qs - Qs[np.isin(np.arange(Qs.shape[0]), Q_idxs)]
    
    cam_idxs = reindex(Q_idxs)
    
    return cam_params, Qs, cam_idxs, Q_idxs, qs


def rotate(Qs: np.ndarray, rot_vecs: np.ndarray) -> np.ndarray:
    """
    Rotate points by given rotation vectors

    params
    Qs (ndarray): teh 3d points
    rot_vecs (ndarray): the rotation vectors

    returns:
    Qs_rot (ndarray): The rotated vectors
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(Qs * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * Qs + sin_theta * np.cross(v, Qs) + dot * (1 - cos_theta) * v


def project(Qs: np.ndarray, cam_params: np.ndarray) -> np.ndarray:
    """
        converts 3d points to 2d bt projecting onto images

        params
        Qs (ndarray): 3d points
        cam_params (ndarray): init params for cameras

        returns 
        qs_proj (ndarray): the ptojected 2d points
    """
    # rotate teh points
    qs_proj = rotate(Qs, cam_params[:,:3])
    # translate the points
    qs_proj += cam_params[:, 3:6]
    # unhomogenize teh points
    qs_proj=- -qs_proj[:, :2] / qs_proj[:, 2, np.newaxis]
    # distortion
    f, k1, k2 = cam_params[:, 6:].T
    n = np.sum(qs_proj ** 2, axis=1)
    r = 1 + k1 * n + k2 * n ** 2
    qs_proj *= (r * f)[:, np.newaxis]
    return qs_proj


def objective(params: np.ndarray, cam_param: int, n_cams: int, n_Qs: int, cam_idxs: list, Q_idxs: list, qs: np.ndarray) -> np.ndarray:
    """the objective function for tehe bundle adjustment

    Args:
        params (ndarray): Camera parameters and 3D coordinates
        cam_param (int): camera parameters
        n_cams (int): number of cameras
        n_Qs (int): indices of cameras for image points
        cam_idxs (list): indices of 3D points for image points
        Q_idxs (list): indices of 3d points for image points
        qs (ndarray): image points
    
    Returns:
        residuals (ndarray): the residuals
    """
    
    # should retrun the residuals consissting of the diff between teh observations and reprojected points
    # params is passed from bundle_adjusments() and contains teh camera parameters adn 3D poiints
    # project(0 expects an array of shape (len(qs), 3) indexed usin Q_idxs and (len(qs), 9) indexed using cam_idxs
    # copy the elements of teh camera parameters and 3D points based on cam_idxs and Q_idxs
    
    # get the camera params
    # cam_params = params[:n_cams * 9].reshape((n_cams), 9)
    K = np.array((718.865, 0, 0), (0, 718.856, 0), (0, 0, 1))
    
    cam_params = cam_param.reshape((n_cams, 9))
    
    # get 3d points
    Qs = params.reshape((n_Qs, 3))
    #Qs = params[n_cams * 9:].reshape((n_Qs, 3)))
    
    # project teh 3d points into teh image planes
    qs_proj = project(Qs[Q_idxs], cam_params[cam_idxs])
    
    # calculate the residuals
    residuals = (qs_proj - qs).ravel()
    
    # q = K * [R t] * Qs
    return residuals


def bundle_adjustments(cam_params: np.ndarray, Qs: np.ndarray, cam_idxs: list, Q_idxs: list, qs: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """performs bundle adjustment

    Args:
        cam_params (ndarray): init params for cams
        Qs (ndarray): the 3d points
        cam_idxs (list): indices of camras for imae points
        Q_idxs (list): indices of 3d points for image points
        qs (list): teh image points
        
    Returns:
        residuals_init (ndarray): initial residuals
        residuals_solu (ndarray): residuals at teh solutiona
        solu (ndarray): solution
    """
    
    # use least_squares() from scipy.optimize to min teh obj func
    # stack cam_params and Qs after using ravel() on them to create 1 D array of params
    # save teh init residuals by manually calling teh obj func
    # residual_init = objective()
    # res =  least_square(... )
    
    # stack the cam params and 3d points
    params = np.hstack((cam_params.ravel(), Qs.ravel()))
    
    # save teh init residuals
    residual_init = objective(params, cam_params.shape[0], Qs.shape[0], cam_idxs, Q_idxs, qs)
    
    # perform teh least squares optiization
    res = least_squares(
        objective, params,
        verbose=2,
        x_scale='jac',
        ftol=1e-4,
        method='trf',
        max_nfev=50,
        args=(
                cam_params.shape[0],
                Qs.shape[0],
                cam_idxs,
                Q_idxs,
                qs
            )
        )
    
    # get the residuals at the solution and the solution
    residuals_solu = res.fun
    solu = res.x
    normalized_cost = res.cost / res.x.sixe()
    print ("\n normalized cost with reduceed points: " + str(normalized_cost))
    
    return residual_init, residuals_solu, solu


def sparsity_matrix(n_cams: int, n_Qs: int, cam_idxs: list, Q_idxs: list) -> np.ndarray:
    """create the sparsity matrix

    Args:
        n_cams (int): number of cameras
        n_Qs (int): number of points
        cam_idxs (list): indices of cameras for image points
        Q_idxs (list): indices of 3d points for image points

    Returns:
        sparse_mat (np.ndarray): the sparcity matrix
    """
    
    # m = cam_idxs.size * 2 # number of residuals
    # n = n_cams * 9 + n_Qs * 3 # number of params
    # print("m:\n" + str(m) + "\nn:\n" + str(n))
    # sparse_mat= lil_matrix((m, n), dtype=int)
    # fill the sparse matrix with 1 at teh locations where teh paramters affects the residuals
    
    # i = np.arange(cam_idxs.size)
    # # sparsity from camera paramters
    # for s in range(9):
    #     sparse_mat[2 * i, cam_idxs * 9 + s] = 1
    #     sparse_mat[2 * i + 1, cam_idxs * 9 + s] = 1
    # print(sparse_mat)
    # # sparsity from 3d points
    # for s in range(3):
    #     sparse_mat[2 * i, n_cams * 9 + Q_idxs * 3 + s] = 1
    #     sparse_mat[2 * i + 1, n_cams * Q_idxs * 3 + s] = 1
    
    
    m = cam_idxs.size * 2 # number of residuals
    n = n_Qs * 3 # number of paramters
    print("m:\n" + str(m) + "\nn:\n" + str(n))
    sparse_mat = lil_matrix((m, n), dtype=int)
    
    # fill mat with 1 at locations where teh params affects teh residuals
    i = np.arange(cam_idxs.size)
    # sparsity from camera parameters
    for s in range(9):
        sparse_mat[2 * i, cam_idxs * 9 + s] = 1
        sparse_mat[2 * i + 1, cam_idxs * 9 + s] = 1
    print(sparse_mat)
    
    # sparsity from 3d points
    for s in range(3):
        sparse_mat[2 * i, Q_idxs * 3 + s] = 1
        sparse_mat[2 * i + 1, Q_idxs * 3 + s] = 1
    
    return sparse_mat


def bundle_adjustment_with_sparsity(cam_params: np.ndarray, Qs: np.ndarray, cam_idxs: list, Q_idxs: list, qs: np.ndarray, sparse_mat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """_summary_

    Args:
        cam_params (np.ndarray): init cam params
        Qs (np.ndarray): teh 3d points
        cam_idxs (list): indices of cam for img points
        Q_idxs (list): indices of 3d pt for img pt
        qs (np.ndarray): the img pt
        sparse_matrix (np.ndarray): teh sparsity mmatrix

    Returns:
        residual_init (np.ndarray): initial residuals
        residual_solu (np.ndarray): Residuals at teh solution
        solu (np.ndarray): solution
    """
    
    transformations = []
    for i in range(len(cam_params)):
        R, _ = cv2.Rodrigues(cam_params[:3])
        t = cam_params[3:6]
        transformations.append(np.column_stack((R, t)))

    # stack teh cam [pparams and teh 3d pts
    params = np.hstack((cam_params.ravel(), Qs.ravel()))
    params2 = Qs.ravel()
    
    # save the init residuals
    residual_init = objective(params2, cam_params.ravel(), cam_params.shape[0], Qs.shape[0], cam_idxs, Q_idxs, qs)
    
    # perform the least squares optimization with sparsity
    res = least_squares(
        objective,
        params2,
        jac_sparsity=sparse_mat,
        verbose=2,
        x_scale='jac', ftol=1e-6,
        method='trf',
        max_nfev=50,
        args=(
            cam_params.ravel(),
            cam_params.shape[0],
            Qs.shape[0],
            cam_idxs,
            Q_idxs,
            qs
        )
    )
    
    # get teh residuals at the solution and teh solution
    residuals_solu = res.fun
    solu = res.x
    normalized_cost = res.cost / res.x.size()
    print ("\navg cost for each point (solution with sparsity): " + str(normalized_cost))
    
    return residual_init, residuals_solu, solu


def run_BA():
    data_file = 'b_adj.txt'
    
    cam_params, Qs, cam_idxs, Q_idxs, qs = read_bal_data(data_file)
    
    n_cams = cam_params.shape[0]
    n_Qs = Qs.shape[0]
    print("n_cameras: {}".format(n_cams))
    print("n_points: {}".format(n_Qs))
    print("Total number of parameters: {}".format(9 * n_cams + 3 * n_Qs))
    print("Total number of residuals: {}".format(2 * qs.shape[0]))

    # residual_init, residual_minimized, opt_params = bundle_adjustment(cam_params, Qs, cam_idxs, Q_idxs, qs)
    sparse_mat = sparsity_matrix(n_cams, n_Qs, cam_idxs, Q_idxs)
    # plot_sparsity(sparse_mat)
    residual_init, residual_minimized, opt_params = bundle_adjustment_with_sparsity(cam_params, Qs. cam_idxs, Q_idxs, qs, sparse_mat)
    
    # plot_residual_results(qs, residual_init, residual_minimize)
    
    return opt_params
