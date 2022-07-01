import time

import numpy as np
from numpy import newaxis as na
from scipy.special import gamma


# from scipy.interpolate import interp1d

# import jax.numpy as np
# from jax import grad

thrust = [
    1.19187945,
    1.17284634,
    1.09860817,
    1.02889592,
    0.97373036,
    0.92826162,
    0.89210543,
    0.86100905,
    0.835423,
    0.81237673,
    0.79225789,
    0.77584769,
    0.7629228,
    0.76156073,
    0.76261984,
    0.76169723,
    0.75232027,
    0.74026851,
    0.72987175,
    0.70701647,
    0.54054532,
    0.45509459,
    0.39343381,
    0.34250785,
    0.30487242,
    0.27164979,
    0.24361964,
    0.21973831,
    0.19918151,
    0.18131868,
    0.16537679,
    0.15103727,
    0.13998636,
    0.1289037,
    0.11970413,
    0.11087113,
    0.10339901,
    0.09617888,
    0.09009926,
    0.08395078,
    0.0791188,
    0.07448356,
    0.07050731,
    0.06684119,
    0.06345518,
    0.06032267,
    0.05741999,
    0.05472609,
]
wind_speed = [
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
    4.5,
    5.0,
    5.5,
    6.0,
    6.5,
    7.0,
    7.5,
    8.0,
    8.5,
    9.0,
    9.5,
    10.0,
    10.5,
    11.0,
    11.5,
    12.0,
    12.5,
    13.0,
    13.5,
    14.0,
    14.5,
    15.0,
    15.5,
    16.0,
    16.5,
    17.0,
    17.5,
    18.0,
    18.5,
    19.0,
    19.5,
    20.0,
    20.5,
    21.0,
    21.5,
    22.0,
    22.5,
    23.0,
    23.5,
    24.0,
    24.5,
    25.0,
    25.5,
]


def cosd(angle):
    return np.cos(np.radians(angle))


def sind(angle):
    return np.sin(np.radians(angle))


def tand(angle):
    return np.tan(np.radians(angle))


def rotate_fields(mesh_x, mesh_y, mesh_z, wd, x_coord, y_coord, z_coord):
    # Find center of rotation
    x_center_of_rotation = np.mean(np.array([np.min(mesh_x), np.max(mesh_x)]))
    y_center_of_rotation = np.mean(np.array([np.min(mesh_y), np.max(mesh_y)]))

    # Convert from compass rose angle to cartesian angle
    angle = ((wd - 270) % 360 + 360) % 360

    # Rotate grid points
    x_offset = mesh_x - x_center_of_rotation
    y_offset = mesh_y - y_center_of_rotation
    mesh_x_rotated = (
        x_offset * cosd(angle) - y_offset * sind(angle) + x_center_of_rotation
    )
    mesh_y_rotated = (
        x_offset * sind(angle) + y_offset * cosd(angle) + y_center_of_rotation
    )

    x_coord_offset = (x_coord - x_center_of_rotation)[:, na, na]
    y_coord_offset = (y_coord - y_center_of_rotation)[:, na, na]

    x_coord_rotated = (
        x_coord_offset * cosd(angle)
        - y_coord_offset * sind(angle)
        + x_center_of_rotation
    )
    y_coord_rotated = (
        x_coord_offset * sind(angle)
        + y_coord_offset * cosd(angle)
        + y_center_of_rotation
    )

    inds_sorted = x_coord_rotated.argsort(axis=3)

    x_coord_rotated_sorted = np.take_along_axis(x_coord_rotated, inds_sorted, axis=3)
    y_coord_rotated_sorted = np.take_along_axis(y_coord_rotated, inds_sorted, axis=3)
    z_coord_rotated_sorted = np.take_along_axis(
        z_coord[na, na, na, :, na, na] * np.ones((np.shape(x_coord_rotated))),
        inds_sorted,
        axis=3,
    )

    mesh_x_rotated_sorted = np.take_along_axis(mesh_x_rotated, inds_sorted, axis=3)
    mesh_y_rotated_sorted = np.take_along_axis(mesh_y_rotated, inds_sorted, axis=3)
    mesh_z_rotated_sorted = np.take_along_axis(
        mesh_z * np.ones((np.shape(mesh_x_rotated))), inds_sorted, axis=3
    )

    inds_unsorted = x_coord_rotated_sorted.argsort(axis=3)

    return (
        mesh_x_rotated_sorted,
        mesh_y_rotated_sorted,
        mesh_z_rotated_sorted,
        x_coord_rotated_sorted,
        y_coord_rotated_sorted,
        z_coord_rotated_sorted,
        inds_sorted,
        inds_unsorted,
    )


def TKE_to_TI(turbulence_kinetic_energy, turb_avg_vels):
    total_turbulence_intensity = (
        np.sqrt((2 / 3) * turbulence_kinetic_energy)
    ) / turb_avg_vels
    return total_turbulence_intensity


def yaw_added_turbulence_mixing(
    turb_avg_vels,
    turbine_ti,
    flow_field_v,
    flow_field_w,
    turb_v,
    turb_w,
):
    # calculate fluctuations
    v_prime = flow_field_v + turb_v
    w_prime = flow_field_w + turb_w

    # get u_prime from current turbulence intensity
    # u_prime = turbine.u_prime()
    TKE = ((turb_avg_vels * turbine_ti) ** 2) / (2 / 3)
    u_prime = np.sqrt(2 * TKE)

    # compute the new TKE
    TKE = (1 / 2) * (
        u_prime ** 2
        + np.mean(v_prime, axis=(4, 5))[:, :, :, :, na, na] ** 2
        + np.mean(w_prime, axis=(4, 5))[:, :, :, :, na, na] ** 2
    )

    # convert TKE back to TI
    TI_total = TKE_to_TI(TKE, turb_avg_vels)

    # convert to turbulence due to mixing
    TI_mixing = np.array(TI_total) - turbine_ti

    return TI_mixing


def calc_VW(
    x_coord_rotated,
    y_coord_rotated,
    wind_shear,
    specified_wind_height,
    turb_avg_vels,
    turbine_Ct,
    turbine_aI,
    turbine_TSR,
    turbine_yaw,
    turbine_hub_height,
    turbine_diameter,
    flow_field_u_initial,
    x_locations,
    y_locations,
    z_locations,
):
    eps_gain = 0.2
    # turbine parameters
    D = turbine_diameter
    HH = turbine_hub_height
    yaw = turbine_yaw
    Ct = turbine_Ct
    TSR = turbine_TSR
    aI = turbine_aI

    # flow parameters
    Uinf = np.mean(flow_field_u_initial, axis=(3, 4, 5))[:, :, :, na, na, na]

    scale = 1.0
    vel_top = (Uinf * ((HH + D / 2) / specified_wind_height) ** wind_shear) / Uinf
    vel_bottom = (Uinf * ((HH - D / 2) / specified_wind_height) ** wind_shear) / Uinf
    Gamma_top = scale * (np.pi / 8) * D * vel_top * Uinf * Ct * sind(yaw) * cosd(yaw)
    Gamma_bottom = (
        -scale * (np.pi / 8) * D * vel_bottom * Uinf * Ct * sind(yaw) * cosd(yaw)
    )
    Gamma_wake_rotation = 0.25 * 2 * np.pi * D * (aI - aI ** 2) * turb_avg_vels / TSR

    # compute the spanwise and vertical velocities induced by yaw
    eps = eps_gain * D  # Use set value

    # decay the vortices as they move downstream - using mixing length
    lmda = D / 8
    kappa = 0.41
    lm = kappa * z_locations / (1 + kappa * z_locations / lmda)
    z = np.linspace(z_locations.min(), z_locations.max(), flow_field_u_initial.shape[5])
    # dudz_initial = np.gradient(flow_field_u_initial, z, axis=5)
    # print('0: ', dudz_initial)
    dudz_initial = np.gradient(flow_field_u_initial, z[1] - z[0], axis=5)
    # print('1: ', dudz_initial)
    # lkj
    nu = lm ** 2 * np.abs(dudz_initial[:, :, :, 0, :, :][:, :, :, na, :, :])

    den1 = 4 * nu * (x_locations - x_coord_rotated) / Uinf + eps ** 2

    # top vortex
    yLocs = y_locations + 0.01 - (y_coord_rotated)
    zT = z_locations + 0.01 - (HH + D / 2)
    rT = yLocs ** 2 + zT ** 2
    val1 = (2 * np.pi * rT) * (1 - np.exp(-rT / (eps ** 2))) * eps ** 2 / den1
    V1 = (zT * Gamma_top) / val1

    W1 = (-yLocs * Gamma_top) / val1

    # bottom vortex
    zB = z_locations + 0.01 - (HH - D / 2)
    rB = yLocs ** 2 + zB ** 2
    val2 = (2 * np.pi * rB) * (1 - np.exp(-rB / (eps ** 2))) * eps ** 2 / den1
    V2 = (zB * Gamma_bottom) / val2

    W2 = (-yLocs * Gamma_bottom) / val2

    # top vortex - ground
    # yLocs = y_locations + 0.01 - (y_coord_rotated)
    zLocs = z_locations + 0.01 + (HH + D / 2)
    val3 = (
        (
            ((2 * np.pi * (yLocs ** 2 + zLocs ** 2)))
            * (1 - np.exp(-(yLocs ** 2 + zLocs ** 2) / (eps ** 2)))
        )
        * eps ** 2
        / den1
    )
    V3 = (zLocs * -Gamma_top) / val3

    W3 = (-yLocs * -Gamma_top) / val3

    # bottom vortex - ground
    # yLocs = y_locations + 0.01 - (y_coord_rotated)
    zLocs = z_locations + 0.01 + (HH - D / 2)
    val4 = (
        (
            ((2 * np.pi * (yLocs ** 2 + zLocs ** 2)))
            * (1 - np.exp(-(yLocs ** 2 + zLocs ** 2) / (eps ** 2)))
        )
        * eps ** 2
        / den1
    )
    V4 = (zLocs * -Gamma_bottom) / val4

    W4 = (-yLocs * -Gamma_bottom) / val4

    # wake rotation vortex
    zC = z_locations + 0.01 - (HH)
    rC = yLocs ** 2 + zC ** 2
    val5 = (2 * np.pi * rC) * (1 - np.exp(-rC / (eps ** 2))) * eps ** 2 / den1
    V5 = (zC * Gamma_wake_rotation) / val5

    W5 = (-yLocs * Gamma_wake_rotation) / val5

    # wake rotation vortex - ground effect
    # yLocs = y_locations + 0.01 - y_coord_rotated
    zLocs = z_locations + 0.01 + HH
    val6 = (
        (
            ((2 * np.pi * (yLocs ** 2 + zLocs ** 2)))
            * (1 - np.exp(-(yLocs ** 2 + zLocs ** 2) / (eps ** 2)))
        )
        * eps ** 2
        / den1
    )
    V6 = (zLocs * -Gamma_wake_rotation) / val6

    W6 = (-yLocs * -Gamma_wake_rotation) / val6

    # total spanwise velocity
    V = V1 + V2 + V3 + V4 + V5 + V6
    W = W1 + W2 + W3 + W4 + W5 + W6

    mask1 = np.array(x_locations >= x_coord_rotated - 1)

    V = V * mask1
    W = W * mask1
    W = W * np.array(W >= 0)

    return V, W


def mask_upstream_wake(mesh_y_rotated, x_coord_rotated, y_coord_rotated, turbine_yaw):
    yR = mesh_y_rotated - y_coord_rotated
    xR = yR * tand(turbine_yaw) + x_coord_rotated
    return xR, yR


def initial_velocity_deficits(U_local, turbine_Ct):
    uR = U_local * turbine_Ct / (2.0 * (1 - np.sqrt(1 - turbine_Ct)))
    u0 = U_local * np.sqrt(1 - turbine_Ct)
    return uR, u0


def initial_wake_expansion(turbine_yaw, turbine_diameter, U_local, veer, uR, u0):
    yaw = -1 * turbine_yaw
    sigma_z0 = turbine_diameter * 0.5 * np.sqrt(uR / (U_local + u0))
    sigma_y0 = sigma_z0 * cosd(yaw) * cosd(veer)
    return sigma_y0, sigma_z0


# def wake_expansion(mesh_x_rotated, mesh_y_rotated, x_coord_rotated, y_coord_rotated, U_local, veer, turbine_yaw, turbine_Ct, turbine_ti, turbine_diameter, ka, kb, alpha, beta):
#     xR, _ = mask_upstream_wake(
#         mesh_y_rotated, x_coord_rotated, y_coord_rotated, turbine_yaw
#     )
#     uR, u0 = initial_velocity_deficits(U_local, turbine_Ct)
#     sigma_y0, _ = initial_wake_expansion(
#         turbine_yaw, turbine_diameter, U_local, veer, uR, u0
#     )

#     # quantity that determines when the far wake starts
#     x0 = (
#         turbine_diameter
#         * (cosd(turbine_yaw) * (1 + np.sqrt(1 - turbine_Ct)))
#         / (np.sqrt(2) * (4 * alpha * turbine_ti + 2 * beta * (1 - np.sqrt(1 - turbine_Ct))))
#         + x_coord_rotated
#     )

#     # dx = np.zeros_like(mesh_x_rotated)
#     dmesh_x_rot_xR = np.subtract(mesh_x_rotated, xR)
#     dmesh_x_rotated_x0 = np.subtract(mesh_x_rotated, x0)
#     dx0_xR = np.subtract(x0, xR)
#     # dmesh_y_rot_y_coord_rot = np.subtract(mesh_y_rotated, y_coord_rotated)
#     # dmesh_y_delta = np.subtract(dmesh_y_rot_y_coord_rot, deflection_field)
#     # dmesh_z_HH = np.subtract(mesh_z, turbine_hub_height)

#     # velocity deficit in the near wake
#     sigma_y = (((x0 - xR) - (mesh_x_rotated - xR)) / (x0 - xR)) * 0.501 * turbine_diameter * np.sqrt(
#         turbine_Ct / 2.0
#     ) + ((mesh_x_rotated - xR) / (x0 - xR)) * sigma_y0

#     # sigma_z = (((x0 - xR) - (mesh_x_rotated - xR)) / (x0 - xR)) * 0.501 * D * np.sqrt(
#     #     Ct / 2.0
#     # ) + ((mesh_x_rotated - xR) / (x0 - xR)) * sigma_z0
#     # sigma_y = (
#     #     sigma_y * np.array(mesh_x_rotated >= xR)
#     #     + np.ones_like(sigma_y) * np.array(mesh_x_rotated < xR) * 0.5 * D
#     # )
#     # sigma_z = (
#     #     sigma_z * np.array(mesh_x_rotated >= xR)
#     #     + np.ones_like(sigma_z) * np.array(mesh_x_rotated < xR) * 0.5 * D
#     # )

#     mask1 = np.array(mesh_x_rotated >= xR)
#     mask2 = np.array(mesh_x_rotated < xR)
#     # mask3 = np.array(mesh_x_rotated <= x0)
#     mask4 = np.array(mesh_x_rotated >= x0)
#     # mask5 = np.array(mesh_x_rotated < x0)

#     # sigma_base = (((dx0_xR) - (dmesh_x_rot_xR)) / (dx0_xR)) * 0.501 * turbine_diameter * np.sqrt(
#     #     turbine_Ct / 2.0
#     # ) + ((dmesh_x_rot_xR) / (dx0_xR))
#     # sigma_y = sigma_base * sigma_y0
#     # sigma_z = sigma_base * sigma_z0
#     sigma_y = (
#         sigma_y * mask1
#         + np.ones_like(sigma_y) * mask2 * 0.5 * turbine_diameter
#     )
#     # sigma_y = np.where(mask1, sigma_y, 0.5 * turbine_diameter)
#     # sigma_z = (
#     #     sigma_z * mask1
#     #     + np.ones_like(sigma_z) * mask2 * 0.5 * D
#     # )
#     # sigma_z = np.where(mask1, sigma_z, 0.5 * turbine_diameter)

#     # a = cosd(veer) ** 2 / (2 * sigma_y ** 2) + sind(veer) ** 2 / (2 * sigma_z ** 2)
#     # b = -sind(2 * veer) / (4 * sigma_y ** 2) + sind(2 * veer) / (4 * sigma_z ** 2)
#     # c = sind(veer) ** 2 / (2 * sigma_y ** 2) + cosd(veer) ** 2 / (2 * sigma_z ** 2)
#     # r = (
#     #     a * (dmesh_y_delta) ** 2
#     #     - 2 * b * (dmesh_y_delta) * ((dmesh_z_HH))
#     #     + c * ((dmesh_z_HH)) ** 2
#     # )
#     # C = 1 - np.sqrt(
#     #     np.clip(1 - (Ct * cosd(yaw) / (8.0 * sigma_y * sigma_z / D ** 2)), 0.0, 1.0)
#     # )

#     # velDef = gaussian_function(U_local, C, r, 1, np.sqrt(0.5))
#     # velDef = velDef * np.array(mask1)
#     # velDef = velDef * np.array(mask3)

#     # wake expansion in the lateral (y) and the vertical (z)
#     ky = ka * turbine_ti + kb  # wake expansion parameters
#     # kz = ka * turbine_ti + kb  # wake expansion parameters
#     sigma_y1 = ky * (dmesh_x_rotated_x0) + sigma_y0
#     # sigma_z1 = kz * (dmesh_x_rotated_x0) + sigma_z0
#     sigma_y1 = sigma_y1 * np.array(mask4) #+ sigma_y0 * np.array(mask5)
#     # sigma_z1 = sigma_z1 * np.array(mask4) #+ sigma_z0 * np.array(mask5)

#     print(sigma_y)
#     print(sigma_y1)
#     lkj

#     sigma_y = sigma_y + sigma_y1
#     # sigma_z = sigma_z + sigma_z1

#     # sigma_n = sigma_y

#     return sigma_y


def wake_expansion(
    mesh_x_rotated,
    x_coord_rotated,
    turbine_Ct,
    turbine_ti,
    turbine_diameter,
    a_s,
    b_s,
    c_s1,
    c_s2,
):
    # Calculate Beta (Eq 10, pp 5 of ref. [1] and table 4 of ref. [2] in docstring)
    beta = 0.5 * (1.0 + np.sqrt(1.0 - turbine_Ct)) / np.sqrt(1.0 - turbine_Ct)
    k = a_s * turbine_ti + b_s
    eps = (c_s1 * turbine_Ct + c_s2) * np.sqrt(beta)

    # Calculate sigma_tilde (Eq 9, pp 5 of ref. [1] and table 4 of ref. [2] in docstring)
    x_tilde = np.abs(mesh_x_rotated - x_coord_rotated) / turbine_diameter
    sigma_y = k * x_tilde + eps

    return sigma_y[na, :, :, :, :, :, :]


def calculate_effective_yaw_angle(
    mesh_y_rotated,
    mesh_z,
    y_coord_rotated,
    turb_avg_vels,
    turbine_Ct,
    turbine_aI,
    turbine_TSR,
    turbine_yaw,
    turbine_hub_height,
    turbine_diameter,
    specified_wind_height,
    wind_shear,
    flow_field_v,
    flow_field_u_initial,
):
    eps_gain = 0.2
    Ct = turbine_Ct
    D = turbine_diameter
    HH = turbine_hub_height
    aI = turbine_aI
    TSR = turbine_TSR
    V = flow_field_v
    Uinf = np.mean(flow_field_u_initial, axis=(3, 4, 5))[:, :, :, na, na, na]
    eps = eps_gain * D  # Use set value

    shape = np.shape(mesh_z)
    yLocs = np.reshape(
        mesh_y_rotated + 0.01 - y_coord_rotated,
        (shape[0], shape[1], shape[2], shape[3], 1, shape[4] * shape[5]),
    )

    # location of top vortex
    zT = np.reshape(
        mesh_z + 0.01 - (HH + D / 2),
        (shape[0], shape[1], shape[2], shape[3], 1, shape[4] * shape[5]),
    )
    rT = yLocs ** 2 + zT ** 2

    # location of bottom vortex
    zB = np.reshape(
        mesh_z + 0.01 - (HH - D / 2),
        (shape[0], shape[1], shape[2], shape[3], 1, shape[4] * shape[5]),
    )
    rB = yLocs ** 2 + zB ** 2

    # wake rotation vortex
    zC = np.reshape(
        mesh_z + 0.01 - (HH),
        (shape[0], shape[1], shape[2], shape[3], 1, shape[4] * shape[5]),
    )
    rC = yLocs ** 2 + zC ** 2

    # find wake deflection from CRV
    min_yaw = -45.0
    max_yaw = 45.0
    test_yaw = np.linspace(min_yaw, max_yaw, 91)
    avg_V = np.mean(V, axis=(4, 5))[:, :, :, :, na, na]

    # what yaw angle would have produced that same average spanwise velocity
    vel_top = ((HH + D / 2) / specified_wind_height) ** wind_shear
    vel_bottom = ((HH - D / 2) / specified_wind_height) ** wind_shear
    Gamma_top = (np.pi / 8) * D * vel_top * Uinf * Ct * sind(test_yaw) * cosd(test_yaw)
    Gamma_bottom = (
        -(np.pi / 8) * D * vel_bottom * Uinf * Ct * sind(test_yaw) * cosd(test_yaw)
    )
    Gamma_wake_rotation = 0.25 * 2 * np.pi * D * (aI - aI ** 2) * turb_avg_vels / TSR

    Veff = (
        np.divide(
            np.einsum("...i,...j->...ij", Gamma_top, zT),
            (2 * np.pi * rT[:, :, :, :, :, na, :]),
        )
        * (1 - np.exp(-rT[:, :, :, :, :, na, :] / (eps ** 2)))
        + np.einsum("...i,...j->...ij", Gamma_bottom, zB)
        / (2 * np.pi * rB[:, :, :, :, :, na, :])
        * (1 - np.exp(-rB[:, :, :, :, :, na, :] / (eps ** 2)))
        + np.einsum("...i,...j->...ij", Gamma_wake_rotation, zC)
        / (2 * np.pi * rC[:, :, :, :, :, na, :])
        * (1 - np.exp(-rC[:, :, :, :, :, na, :] / (eps ** 2)))
    )

    tmp = avg_V - np.mean(Veff, axis=6)

    # return indices of sorted residuals to find effective yaw angle
    order = np.argsort(np.abs(tmp), axis=5)
    idx_1 = np.take_along_axis(order, np.array([[[[[[0]]]]]]), axis=5)
    idx_2 = np.take_along_axis(order, np.array([[[[[[1]]]]]]), axis=5)

    # check edge case, if true, assign max yaw value
    mask1 = np.array(idx_1 > idx_2)
    mask2 = np.array(idx_1 <= idx_2)

    idx_right_1 = idx_1 + 1  # adjacent point
    idx_left_1 = idx_2 - 1  # adjacent point
    mR_1 = abs(
        np.take_along_axis(tmp, idx_right_1, axis=5)
        - abs(np.take_along_axis(tmp, idx_1, axis=5))
    )  # slope
    mL_1 = abs(np.take_along_axis(tmp, idx_2, axis=5)) - abs(
        np.take_along_axis(tmp, idx_left_1, axis=5)
    )  # slope
    bR_1 = abs(np.take_along_axis(tmp, idx_1, axis=5)) - mR_1 * idx_1  # intercept
    bL_1 = abs(np.take_along_axis(tmp, idx_2, axis=5)) - mL_1 * idx_2  # intercept

    idx_right_2 = idx_2 + 1  # adjacent point
    idx_left_2 = idx_1 - 1  # adjacent point
    mR_2 = abs(
        np.take_along_axis(tmp, idx_right_2, axis=5)
        - abs(np.take_along_axis(tmp, idx_2, axis=5))
    )  # slope
    mL_2 = abs(np.take_along_axis(tmp, idx_1, axis=5)) - abs(
        np.take_along_axis(tmp, idx_left_2, axis=5)
    )  # slope
    bR_2 = abs(np.take_along_axis(tmp, idx_2, axis=5)) - mR_2 * idx_2  # intercept
    bL_2 = abs(np.take_along_axis(tmp, idx_1, axis=5)) - mL_2 * idx_1  # intercept

    mR = mR_1 * mask1 + mR_2 * mask2
    mL = mL_1 * mask1 + mL_2 * mask2
    bR = bR_1 * mask1 + bR_2 * mask2
    bL = bL_1 * mask1 + bL_2 * mask2

    ival = (bR - bL) / (mL - mR)
    # convert the indice into degrees
    yaw_effective = ival - max_yaw

    return yaw_effective + turbine_yaw


def gaussian_function(U, C, r, n, sigma):
    return U * C * np.exp(-1 * r ** n / (2 * sigma ** 2))


def gauss_cumulative_vel_model(
    veer,
    flow_field_u_initial,
    turb_avg_vels,
    turbine_ti,
    turbine_Ct,
    turbine_yaw,
    turbine_hub_height,
    turbine_diameter,
    mesh_x_rotated,
    mesh_y_rotated,
    mesh_z,
    x_coord_rotated,
    y_coord_rotated,
    z_coord_rotated,
    deflection_field,
    ii,
    turb_u_wake,
    Ctmp,
    sigma_i,
):
    # gch_gain = 2.0

    a_s = 0.179367259
    b_s = 0.0118889215
    c_s1 = 0.0563691592
    c_s2 = 0.13290157

    a_f = 3.11
    b_f = -0.68
    c_f = 2.41

    alpha_mod = 1.0

    sigma_n = wake_expansion(
        mesh_x_rotated,
        x_coord_rotated[:, :, :, ii, :, :][:, :, :, na, :, :],
        turbine_Ct[:, :, :, ii, na, na][:, :, :, na, :, :],
        turbine_ti[:, :, :, ii, :, :][:, :, :, na, :, :],
        turbine_diameter,
        a_s,
        b_s,
        c_s1,
        c_s2,
    )

    # sum_lbda = np.zeros_like(flow_field_u_initial)

    if ii >= 2:
        S = sigma_n ** 2 + sigma_i[0 : ii - 1, :, :, :, :, :, :] ** 2
        Y = (
            y_coord_rotated[:, :, :, ii, :, :][:, :, :, na, :, :]
            - y_coord_rotated
            - deflection_field
        )[na, :, :, :, :, :, :] ** 2 / (2 * S)
        Z = (z_coord_rotated[:, :, :, ii, :, :][:, :, :, na, :, :] - z_coord_rotated)[
            na, :, :, :, :, :, :
        ] ** 2 / (2 * S)
        lbda = (
            alpha_mod
            * sigma_i[0 : ii - 1, :, :, :, :, :, :] ** 2
            / S
            * np.exp(-Y)
            * np.exp(-Z)
        )

        sum_lbda = np.sum(
            lbda * (Ctmp[0 : ii - 1, :, :, :, :, :, :] / flow_field_u_initial), axis=0
        )

    else:
        sum_lbda = 0.0

    sigma_i[ii] = sigma_n

    # blondel
    # super gaussian
    # b_f = self.b_f1 * np.exp(self.b_f2 * TI) + self.b_f3
    x_tilde = (
        np.abs(mesh_x_rotated - x_coord_rotated[:, :, :, ii, :, :][:, :, :, na, :, :])
        / turbine_diameter
    )
    r_tilde = (
        np.sqrt(
            (
                mesh_y_rotated
                - y_coord_rotated[:, :, :, ii, :, :][:, :, :, na, :, :]
                - deflection_field
            )
            ** 2
            + (mesh_z - z_coord_rotated[:, :, :, ii, :, :][:, :, :, na, :, :]) ** 2,
        )
        / turbine_diameter
    )
    # a_f = 3.11
    # b_f = -0.68
    # c_f = 2.41
    n = a_f * np.exp(b_f * x_tilde) + c_f

    # print('n: ', n)

    a1 = 2 ** (2 / n - 1)
    a2 = 2 ** (4 / n - 2)

    # # Calculate Beta (Eq 10, pp 5 of ref. [1] and table 4 of ref. [2] in docstring)
    # beta = 0.5 * (1.0 + np.sqrt(1.0 - turbine.Ct)) / np.sqrt(1.0 - turbine.Ct)
    # k = self.a_s * TI + self.b_s
    # eps = (self.c_s1 * turbine.Ct + self.c_s2) * np.sqrt(beta)
    #
    # # Calculate sigma_tilde (Eq 9, pp 5 of ref. [1] and table 4 of ref. [2] in docstring)
    # sigma_n = k * x_tilde + eps

    C = a1 - np.sqrt(
        a2
        - (
            (n * turbine_Ct[:, :, :, ii, na, na][:, :, :, na, :, :])
            * cosd(turbine_yaw[:, :, :, ii, :, :][:, :, :, na, :, :])
            / (
                16.0
                * gamma(2 / n)
                * np.sign(sigma_n[0, :, :, :, :, :, :])
                * (np.abs(sigma_n[0, :, :, :, :, :, :]) ** (4 / n))
                * (1 - sum_lbda) ** 2
            )
        )
    )

    C = C * (1 - sum_lbda)

    Ctmp[ii] = C

    # f = np.exp(
    #     -(
    #         (
    #             mesh_y_rotated
    #             - y_coord_rotated[:, :, :, ii, :, :][:, :, :, na, :, :]
    #             - deflection_field
    #         )
    #         ** 2
    #     )
    #     / (2 * sigma_n[0, :, :, :, :, :, :] ** 2)
    # ) * np.exp(
    #     -((mesh_z - z_coord_rotated[:, :, :, ii, :, :][:, :, :, na, :, :]) ** 2)
    #     / (2 * sigma_n[0, :, :, :, :, :, :] ** 2)
    # )

    yR = mesh_y_rotated - y_coord_rotated[:, :, :, ii, :, :][:, :, :, na, :, :]
    xR = (
        yR * tand(turbine_yaw[:, :, :, ii, :, :][:, :, :, na, :, :])
        + x_coord_rotated[:, :, :, ii, :, :][:, :, :, na, :, :]
    )

    # add turbines together
    velDef = C * np.exp((-1 * r_tilde ** n) / (2 * sigma_n[0, :, :, :, :, :, :] ** 2))

    # velDef[mesh_x_rotated < xR] = 0
    velDef = velDef * np.array(mesh_x_rotated >= xR)

    # mirrored wake
    r_tilde2 = (
        np.sqrt(
            (
                mesh_y_rotated
                - y_coord_rotated[:, :, :, ii, :, :][:, :, :, na, :, :]
                - deflection_field
            )
            ** 2
            + (mesh_z + z_coord_rotated[:, :, :, ii, :, :][:, :, :, na, :, :]) ** 2,
        )
        / turbine_diameter
    )
    velDef2 = C * np.exp((-1 * r_tilde2 ** n) / (2 * sigma_n[0, :, :, :, :, :, :] ** 2))
    # velDef2[mesh_x_rotated < xR] = 0
    velDef2 = velDef2 * np.array(mesh_x_rotated >= xR)

    turb_u_wake = turb_u_wake + turb_avg_vels[:, :, :, ii][
        :, :, :, na, na, na
    ] * np.sqrt(velDef ** 2 + velDef2 ** 2)

    return (
        turb_u_wake,
        np.zeros(np.shape(turb_u_wake)),
        np.zeros(np.shape(turb_u_wake)),
        Ctmp,
        sigma_i,
    )


def gauss_defl_model(
    mesh_x_rotated,
    mesh_y_rotated,
    x_coord_rotated,
    y_coord_rotated,
    flow_field_u_initial,
    wind_veer,
    turbine_ti,
    turbine_Ct,
    yaw,
    turbine_tilt,
    turbine_diameter,
):
    # free-stream velocity (m/s)
    wind_speed = flow_field_u_initial
    veer = wind_veer

    # added turbulence model
    TI = turbine_ti

    ka = 0.38  # wake expansion parameter
    kb = 0.004  # wake expansion parameter
    alpha = 0.58  # near wake parameter
    beta = 0.077  # near wake parameter
    ad = 0.0  # natural lateral deflection parameter
    bd = 0.0  # natural lateral deflection parameter
    dm = 1.0

    # turbine parameters
    D = turbine_diameter
    tilt = turbine_tilt
    Ct = turbine_Ct

    # U_local = flow_field.wind_map.grid_wind_speed
    # just a placeholder for now, should be initialized with the flow_field
    U_local = flow_field_u_initial

    # initial velocity deficits
    uR = (
        U_local
        * Ct
        * cosd(tilt)
        * cosd(yaw)
        / (2.0 * (1 - np.sqrt(1 - (Ct * cosd(tilt) * cosd(yaw)))))
    )
    u0 = U_local * np.sqrt(1 - Ct)

    # length of near wake
    x0 = (
        D
        * (cosd(yaw) * (1 + np.sqrt(1 - Ct * cosd(yaw))))
        / (np.sqrt(2) * (4 * alpha * TI + 2 * beta * (1 - np.sqrt(1 - Ct))))
        + x_coord_rotated
    )

    # wake expansion parameters
    ky = ka * TI + kb
    kz = ka * TI + kb

    C0 = 1 - u0 / wind_speed
    M0 = C0 * (2 - C0)
    E0 = C0 ** 2 - 3 * np.exp(1.0 / 12.0) * C0 + 3 * np.exp(1.0 / 3.0)

    # initial Gaussian wake expansion
    sigma_z0 = D * 0.5 * np.sqrt(uR / (U_local + u0))
    sigma_y0 = sigma_z0 * cosd(yaw) * cosd(veer)

    yR = mesh_y_rotated - y_coord_rotated
    xR = yR * tand(yaw) + x_coord_rotated

    # yaw parameters (skew angle and distance from centerline)
    theta_c0 = (
        dm * (0.3 * np.radians(yaw) / cosd(yaw)) * (1 - np.sqrt(1 - Ct * cosd(yaw)))
    )  # skew angle in radians
    delta0 = np.tan(theta_c0) * (x0 - x_coord_rotated)  # initial wake deflection;
    # NOTE: use np.tan here since theta_c0 is radians

    dmesh_x_x0 = mesh_x_rotated - x0
    dmesh_x_rot_xcoord_rot = mesh_x_rotated - x_coord_rotated

    mask1 = np.array(mesh_x_rotated >= xR)
    mask2 = np.array(mesh_x_rotated <= x0)
    mask3 = np.array(mesh_x_rotated >= x0)
    mask4 = np.array(mesh_x_rotated < x0)

    # deflection in the near wake
    delta_near_wake = ((mesh_x_rotated - xR) / (x0 - xR)) * delta0 + (
        ad + bd * (dmesh_x_rot_xcoord_rot)
    )
    delta_near_wake = delta_near_wake * mask1 * mask2
    # delta_near_wake = delta_near_wake * np.array(mesh_x_rotated <= x0)

    # deflection in the far wake
    sigma_y = ky * (dmesh_x_x0) + sigma_y0
    sigma_z = kz * (dmesh_x_x0) + sigma_z0
    sigma_y = sigma_y * mask3 + sigma_y0 * mask4
    sigma_z = sigma_z * mask3 + sigma_z0 * mask4

    sigma_yz = sigma_y * sigma_z
    sigma_y0z0 = sigma_y0 * sigma_z0
    sigma_yz_sqrt = np.sqrt(sigma_yz / (sigma_y0z0))

    ln_deltaNum = (1.6 + np.sqrt(M0)) * (1.6 * sigma_yz_sqrt - np.sqrt(M0))
    ln_deltaDen = (1.6 - np.sqrt(M0)) * (1.6 * sigma_yz_sqrt + np.sqrt(M0))

    delta_far_wake = (
        delta0
        + (theta_c0 * E0 / 5.2)
        * np.sqrt(sigma_y0z0 / (ky * kz * M0))
        * np.log(ln_deltaNum / ln_deltaDen)
        + (ad + bd * (dmesh_x_rot_xcoord_rot))
    )

    delta_far_wake = delta_far_wake * np.array(mesh_x_rotated > x0)
    deflection = delta_near_wake + delta_far_wake

    return deflection


def crespo_hernandez(
    ambient_TI, x_coord_downstream, x_coord_upstream, rotor_diameter, aI
):
    ti_initial = 0.01
    ti_constant = 0.5
    ti_ai = 0.8
    ti_downstream = -0.32
#     ti_initial = 0.01
#     ti_constant = 0.9
#     ti_ai = 0.83
#     ti_downstream = -0.25
    
    # replace zeros and negatives with 1 to prevent nans/infs
    # keep downstream components; set upstream to 1.0
    delta_x = np.array(x_coord_downstream - x_coord_upstream)
    upstream_mask = np.array(delta_x <= 0.1)
    downstream_mask = np.array(delta_x > -0.1)      
    delta_x = delta_x * downstream_mask + np.ones_like(delta_x) * np.array(upstream_mask)

    # turbulence intensity calculation based on Crespo et. al.
    ti_calculation = (
        ti_constant
        * aI ** ti_ai
        * ambient_TI ** ti_initial
        * (delta_x / rotor_diameter) ** ti_downstream
    )
    
    #mask the 1 values from above w/ zeros
    return ti_calculation * np.array(downstream_mask)


def turbine_avg_velocity(turb_inflow_vel):
    return np.cbrt(np.mean(turb_inflow_vel ** 3, axis=(4, 5)))


# fCtInterp = interp1d(wind_speed, thrust, fill_value="extrapolate")


def Ct(turb_avg_vels, wind_speed, thrust):
    # Ct_vals = fCtInterp(turb_avg_vels)
    Ct_vals = np.interp(turb_avg_vels, np.array(wind_speed), np.array(thrust))
    Ct_vals = Ct_vals * np.array(Ct_vals < 1.0) + 0.9999 * np.ones(
        np.shape(Ct_vals)
    ) * np.array(Ct_vals >= 1.0)
    Ct_vals = Ct_vals * np.array(Ct_vals > 0.0) + 0.0001 * np.ones(
        np.shape(Ct_vals)
    ) * np.array(Ct_vals <= 0.0)
    return Ct_vals


def aI(turb_Ct, yaw_angles):
    return 0.5 / cosd(yaw_angles) * (1 - np.sqrt(1 - turb_Ct * cosd(yaw_angles)))


# ///// #
# SETUP #
# ///// #

# Turbine parameters
turbine_diameter = 126.0
turbine_radius = turbine_diameter / 2.0
turbine_hub_height = 90.0

x_spc = 5 * 126.0
# x_coord = np.array([0.0, x_spc, 2 * x_spc])
# y_coord = np.array([0.0, 0.0, 0.0])
# z_coord = np.array([90.0] * len(x_coord))

nturbs = 20
x_coord = np.array([xx * x_spc for xx in range(nturbs)])
y_coord = np.array([0.0] * len(x_coord))
z_coord = np.array([90.0] * len(x_coord))

y_ngrid = 5
z_ngrid = 5
rloc = 0.5

dtype = np.float64

# Wind parameters
ws = np.array([25.0])
# ws = np.arange(3.0, 25.1, 1.0)
wd = np.array([270.0])
# wd = np.arange(0.0, 360.0, 5.0)
# i  j  k  l  m
# wd ws x  y  z

specified_wind_height = 90.0
wind_shear = 0.12
wind_veer = 0.0


# ///////////////// #
# ONLY ROTOR POINTS #
# ///////////////// #


def update_grid(x_grid_i, y_grid_i, wind_direction_i, x1, x2):
    xoffset = x_grid_i - x1[:, na, na]
    yoffset = y_grid_i - x2[:, na, na]

    wind_cos = cosd(-wind_direction_i)
    wind_sin = sind(-wind_direction_i)

    x_grid_i = xoffset * wind_cos - yoffset * wind_sin + x1[:, na, na]
    y_grid_i = yoffset * wind_cos + xoffset * wind_sin + x2[:, na, na]
    return x_grid_i, y_grid_i


def calculate_area_overlap(wake_velocities, freestream_velocities, y_ngrid, z_ngrid):
    """
    compute wake overlap based on the number of points that are not freestream velocity, i.e. affected by the wake
    """
    count = np.sum(freestream_velocities - wake_velocities <= 0.05, axis=(4, 5))
    return (y_ngrid * z_ngrid - count) / (y_ngrid * z_ngrid)


def initialize_flow_field(
    x_coord,
    y_coord,
    z_coord,
    y_ngrid,
    z_ngrid,
    wd,
    ws,
    specified_wind_height,
    wind_shear,
):
    # Flow field bounds
    x_grid = np.zeros((len(x_coord), y_ngrid, z_ngrid))
    y_grid = np.zeros((len(x_coord), y_ngrid, z_ngrid))
    z_grid = np.zeros((len(x_coord), y_ngrid, z_ngrid))

    angle = ((wd - 270) % 360 + 360) % 360

    x1, x2, x3 = x_coord, y_coord, z_coord

    pt = rloc * turbine_radius

    # TODO: would it be simpler to create rotor points inherently rotated to be
    # perpendicular to the wind
    yt = np.linspace(
        x2 - pt,
        x2 + pt,
        y_ngrid,
    )
    zt = np.linspace(
        x3 - pt,
        x3 + pt,
        z_ngrid,
    )

    x_grid = np.ones((len(x_coord), y_ngrid, z_ngrid)) * x_coord[:, na, na]
    y_grid = np.ones((len(x_coord), y_ngrid, z_ngrid)) * yt.T[:, :, na]
    z_grid = np.ones((len(x_coord), y_ngrid, z_ngrid)) * zt.T[:, na, :]

    # yaw turbines to be perpendicular to the wind direction
    # TODO: update update_grid to be called something better
    x_grid, y_grid = update_grid(x_grid, y_grid, angle[na, :, na, na, na, na], x1, x2)

    mesh_x = x_grid
    mesh_y = y_grid
    mesh_z = z_grid

    flow_field_u_initial = (
        ws[na, na, :, na, na, na] * (mesh_z / specified_wind_height) ** wind_shear
    ) * np.ones((1, len(wd), 1, 1, 1, 1))
    flow_field_v_initial = np.zeros(np.shape(flow_field_u_initial))
    flow_field_w_initial = np.zeros(np.shape(flow_field_u_initial))

    # rotate turbine locations/fields to be perpendicular to wind direction
    (
        mesh_x_rotated,
        mesh_y_rotated,
        mesh_z_rotated,
        x_coord_rotated,
        y_coord_rotated,
        z_coord_rotated,
        inds_sorted,
        inds_unsorted,
    ) = rotate_fields(
        mesh_x, mesh_y, mesh_z, wd[na, :, na, na, na, na], x_coord, y_coord, z_coord
    )

    return (
        flow_field_u_initial,
        flow_field_v_initial,
        flow_field_w_initial,
        mesh_x_rotated,
        mesh_y_rotated,
        mesh_z_rotated,
        x_coord_rotated,
        y_coord_rotated,
        z_coord_rotated,
        inds_sorted,
        inds_unsorted,
    )


# ///////////////// #
# GAUSS WAKE MODEL #
# ///////////////// #

# VECTORIZED CALLS
# Initialize field values
def main():
    (
        flow_field_u_initial,
        flow_field_v_initial,
        flow_field_w_initial,
        mesh_x_rotated,
        mesh_y_rotated,
        mesh_z,
        x_coord_rotated,
        y_coord_rotated,
        z_coord_rotated,
        inds_sorted,
        inds_unsorted,
    ) = initialize_flow_field(
        x_coord,
        y_coord,
        z_coord,
        y_ngrid,
        z_ngrid,
        wd,
        ws,
        specified_wind_height,
        wind_shear,
    )

    # Initialize other field values
    u_wake = np.zeros(np.shape(flow_field_u_initial))
    flow_field_u = flow_field_u_initial - u_wake
    flow_field_v = flow_field_v_initial
    flow_field_w = flow_field_w_initial
    turb_inflow_field = np.ones(np.shape(flow_field_u_initial)) * flow_field_u_initial
    turb_u_wake = np.zeros_like(flow_field_u_initial)

    # Initialize turbine values
    TI = 0.06
    turb_TIs = np.ones_like(x_coord_rotated) * TI
    ambient_TIs = np.ones_like(x_coord_rotated) * TI
    yaw_angle = np.ones_like(x_coord_rotated) * 0.0
    turbine_tilt = np.ones_like(x_coord_rotated) * 0.0
    turbine_TSR = np.ones_like(x_coord_rotated) * 8.0

    tic = time.perf_counter()

    # Loop over turbines to solve wakes
    Ctmp = np.zeros((len(x_coord), 1, len(wd), len(ws), len(x_coord), y_ngrid, z_ngrid))
    sigma_i = np.zeros((len(x_coord), 1, len(wd), len(ws), len(x_coord), y_ngrid, z_ngrid))
    for i in range(len(x_coord)):
        mask1 = np.array(
            mesh_x_rotated != x_coord_rotated[:, :, :, i, :, :][:, :, :, na, :, :]
        )
        mask2 = np.array(
            mesh_x_rotated == x_coord_rotated[:, :, :, i, :, :][:, :, :, na, :, :]
        )
        turb_inflow_field = (
            turb_inflow_field * mask1 + (flow_field_u_initial - turb_u_wake) * mask2
        )

        turb_avg_vels = turbine_avg_velocity(turb_inflow_field)
        turb_Cts = Ct(turb_avg_vels, wind_speed, thrust)
        turb_aIs = aI(turb_Cts, np.squeeze(yaw_angle, axis=(4, 5)))

        # Secondary steering calculation
        # yaw = -1 * calculate_effective_yaw_angle(
        #     mesh_y_rotated[:, :, :, i, :, :][:, :, :, na, :, :],
        #     mesh_z[:, :, :, i, :, :][:, :, :, na, :, :],
        #     y_coord_rotated[:, :, :, i, :, :][:, :, :, na, :, :],
        #     turb_avg_vels[:, :, :, i][:, :, :, na, na, na],
        #     turb_Cts[:, :, :, i][:, :, :, na, na, na],
        #     turb_aIs[:, :, :, i][:, :, :, na, na, na],
        #     turbine_TSR[:, :, :, i, :, :][:, :, :, na, :, :],
        #     yaw_angle,
        #     turbine_hub_height,
        #     turbine_diameter,
        #     specified_wind_height,
        #     wind_shear,
        #     flow_field_v[:, :, :, i, :, :][:, :, :, na, :, :],
        #     flow_field_u_initial,
        # )

        # Wake deflection calculation
        deflection_field = gauss_defl_model(
            mesh_x_rotated,
            mesh_y_rotated,
            x_coord_rotated[:, :, :, i, :, :][:, :, :, na, :, :],
            y_coord_rotated[:, :, :, i, :, :][:, :, :, na, :, :],
            flow_field_u_initial,
            wind_veer,
            turb_TIs[:, :, :, i, :, :][:, :, :, na, :, :],
            turb_Cts[:, :, :, i][:, :, :, na, na, na],
            yaw_angle[:, :, :, i, :, :][:, :, :, na, :, :],
            turbine_tilt,
            turbine_diameter,
        )

        # Determine V and W wind components
        # turb_v_wake, turb_w_wake = calc_VW(
        #     x_coord_rotated[:, :, :, i, :, :][:, :, :, na, :, :],
        #     y_coord_rotated[:, :, :, i, :, :][:, :, :, na, :, :],
        #     wind_shear,
        #     specified_wind_height,
        #     turb_avg_vels[:, :, :, i][:, :, :, na, na, na],
        #     turb_Cts[:, :, :, i][:, :, :, na, na, na],
        #     turb_aIs[:, :, :, i][:, :, :, na, na, na],
        #     turbine_TSR[:, :, :, i, :, :][:, :, :, na, :, :],
        #     yaw_angle,
        #     turbine_hub_height,
        #     turbine_diameter,
        #     flow_field_u_initial,
        #     mesh_x_rotated,
        #     mesh_y_rotated,
        #     mesh_z,
        # )

        # Yaw-added wake recovery (YAR) calculation
        # TI_mixing = yaw_added_turbulence_mixing(
        #     turb_avg_vels[:, :, :, i][:, :, :, na, na, na],
        #     turb_TIs[:, :, :, i, :, :][:, :, :, na, :, :],
        #     flow_field_v[:, :, :, i, :, :][:, :, :, na, :, :],
        #     flow_field_w[:, :, :, i, :, :][:, :, :, na, :, :],
        #     turb_v_wake[:, :, :, i, :, :][:, :, :, na, :, :],
        #     turb_w_wake[:, :, :, i, :, :][:, :, :, na, :, :],
        # )

        # # Modify turbine TIs based on YAR
        # gch_gain = 2
        # mask1 = np.array(
        #     x_coord_rotated == x_coord_rotated[:, :, :, i, :, :][:, :, :, na, :, :]
        # )
        # turb_TIs = turb_TIs + gch_gain * TI_mixing * mask1

        # Calculate wake deficits
        veer = 0.0

        turb_u_wake, turb_v_wake, turb_w_wake, Ctmp, sigma_i = gauss_cumulative_vel_model(
            veer,
            flow_field_u_initial,
            turb_avg_vels,
            turb_TIs,
            turb_Cts,
            yaw_angle,
            turbine_hub_height,
            turbine_diameter,
            mesh_x_rotated,
            mesh_y_rotated,
            mesh_z,
            x_coord_rotated,
            y_coord_rotated,
            z_coord_rotated,
            deflection_field,
            i,
            turb_u_wake,
            Ctmp,
            sigma_i,
        )

        # turb_u_wake = np.zeros_like(flow_field_u_initial)
        turb_v_wake = np.zeros_like(turb_u_wake)
        turb_w_wake = np.zeros_like(turb_u_wake)

        # Perform wake/field combinations
        # u_wake = np.sqrt((u_wake ** 2) + (np.array(turb_u_wake) ** 2))
        # flow_field_u = flow_field_u_initial - u_wake
        flow_field_v = flow_field_v + turb_v_wake
        flow_field_w = flow_field_w + turb_w_wake

        # Calculate wake overlap for wake-added turbulence (WAT)
        turb_wake_field = flow_field_u_initial - turb_u_wake
        area_overlap = calculate_area_overlap(
            turb_wake_field, flow_field_u_initial, y_ngrid, z_ngrid
        )

        # Calculate WAT for turbines
        WAT_TIs = crespo_hernandez(
            ambient_TIs,
            x_coord_rotated,
            x_coord_rotated[:, :, :, i, :, :][:, :, :, na, :, :],
            turbine_diameter,
            turb_aIs[:, :, :, i][:, :, :, na, na, na],
        )

        # Modify WAT by wake area overlap
        # TODO: will need to make the rotor_diameter part of this mask work for
        # turbines of different types
        downstream_influence_length = 15 * turbine_diameter
        ti_added = (
            area_overlap[:, :, :, :, na, na]
            * np.nan_to_num(WAT_TIs, posinf=0.0)
            * (
                np.array(
                    x_coord_rotated > x_coord_rotated[:, :, :, i, :, :][:, :, :, na, :, :]
                )
            )
            * (
                np.array(
                    np.abs(
                        y_coord_rotated[:, :, :, i, :, :][:, :, :, na, :, :]
                        - y_coord_rotated
                    )
                    < 2 * turbine_diameter
                )
            )
            * (
                np.array(
                    x_coord_rotated
                    <= downstream_influence_length
                    + x_coord_rotated[:, :, :, i, :, :][:, :, :, na, :, :]
                )
            )
        )

        # Combine turbine TIs with WAT
        turb_TIs = np.maximum(
            np.sqrt(ti_added ** 2 + ambient_TIs ** 2),
            turb_TIs,
        )

    flow_field_u = flow_field_u_initial - turb_u_wake

    toc = time.perf_counter()
    print(f"Computed wake deficits in {toc - tic:0.4f} seconds")
    print("avg flow_field u value: ", np.mean(flow_field_u))
    
if __name__ == "__main__":
    main()


# //////////////// #
# COMPUTE GRADIENT #
# //////////////// #


# def calculate_wake(
#     x_coord_rotated,
#     y_coord_rotated,
#     mesh_x_rotated,
#     mesh_y_rotated,
#     mesh_z,
#     flow_field_u_initial,
#     flow_field_u,
#     flow_field_v,
#     flow_field_w,
#     turb_inflow_field,
#     u_wake,
#     turbine_TSR,
#     wind_speed,
#     thrust,
#     turb_TIs,
#     yaw_angle,
#     turbine_hub_height,
#     turbine_diameter,
#     specified_wind_height,
#     wind_shear,
# ):
#     for i in range(len(x_coord)):
#         turb_inflow_field = turb_inflow_field * np.array(
#             mesh_x_rotated != x_coord_rotated[:, :, :, i, :, :][:, :, :, na, :, :]
#         ) + (flow_field_u_initial - u_wake) * np.array(
#             mesh_x_rotated == x_coord_rotated[:, :, :, i, :, :][:, :, :, na, :, :]
#         )

#         turb_avg_vels = turbine_avg_velocity(turb_inflow_field)
#         turb_Cts = Ct(turb_avg_vels, wind_speed, thrust)
#         turb_aIs = aI(turb_Cts, np.squeeze(yaw_angle, axis=(4,5)))

#         # Secondary steering calculation
#         yaw = -1 * calculate_effective_yaw_angle(
#             mesh_y_rotated[:, :, :, i, :, :][:, :, :, na, :, :],
#             mesh_z[:, :, :, i, :, :][:, :, :, na, :, :],
#             y_coord_rotated[:, :, :, i, :, :][:, :, :, na, :, :],
#             turb_avg_vels[:, :, :, i][:, :, :, na, na, na],
#             turb_Cts[:, :, :, i][:, :, :, na, na, na],
#             turb_aIs[:, :, :, i][:, :, :, na, na, na],
#             turbine_TSR[:, :, :, i, :, :][:, :, :, na, :, :],
#             yaw_angle,
#             turbine_hub_height,
#             turbine_diameter,
#             specified_wind_height,
#             wind_shear,
#             flow_field_v[:, :, :, i, :, :][:, :, :, na, :, :],
#             flow_field_u_initial,
#         )

#         # Wake deflection calculation
#         deflection_field = gauss_defl_model(
#             mesh_x_rotated,
#             mesh_y_rotated,
#             x_coord_rotated[:, :, :, i, :, :][:, :, :, na, :, :],
#             y_coord_rotated[:, :, :, i, :, :][:, :, :, na, :, :],
#             flow_field_u_initial,
#             wind_veer,
#             turb_TIs[:, :, :, i, :, :][:, :, :, na, :, :],
#             turb_Cts[:, :, :, i][:, :, :, na, na, na],
#             yaw[:, :, :, i, :, :][:, :, :, na, :, :],
#             turbine_tilt,
#             turbine_diameter,
#         )

#         # Determine V and W wind components
#         turb_v_wake, turb_w_wake = calc_VW(
#             x_coord_rotated[:, :, :, i, :, :][:, :, :, na, :, :],
#             y_coord_rotated[:, :, :, i, :, :][:, :, :, na, :, :],
#             wind_shear,
#             specified_wind_height,
#             turb_avg_vels[:, :, :, i][:, :, :, na, na, na],
#             turb_Cts[:, :, :, i][:, :, :, na, na, na],
#             turb_aIs[:, :, :, i][:, :, :, na, na, na],
#             turbine_TSR[:, :, :, i, :, :][:, :, :, na, :, :],
#             yaw_angle,
#             turbine_hub_height,
#             turbine_diameter,
#             flow_field_u_initial,
#             mesh_x_rotated,
#             mesh_y_rotated,
#             mesh_z,
#         )

#         # Yaw-added wake recovery (YAR) calculation
#         TI_mixing = yaw_added_turbulence_mixing(
#             turb_avg_vels[:, :, :, i][:, :, :, na, na, na],
#             turb_TIs[:, :, :, i, :, :][:, :, :, na, :, :],
#             flow_field_v[:, :, :, i, :, :][:, :, :, na, :, :],
#             flow_field_w[:, :, :, i, :, :][:, :, :, na, :, :],
#             turb_v_wake[:, :, :, i, :, :][:, :, :, na, :, :],
#             turb_w_wake[:, :, :, i, :, :][:, :, :, na, :, :],
#         )

#         # Modify turbine TIs based on YAR
#         gch_gain = 2
#         turb_TIs = turb_TIs + gch_gain * TI_mixing * (
#             np.array(
#                 x_coord_rotated == x_coord_rotated[:, :, :, i, :, :][:, :, :, na, :, :]
#             )
#         )

#         # Calculate wake deficits
#         veer = 0.0
#         turb_u_wake = gauss_vel_model(
#             veer,
#             flow_field_u_initial,
#             turb_TIs[:, :, :, i, :, :][:, :, :, na, :, :],
#             turb_Cts[:, :, :, i][:, :, :, na, na, na],
#             yaw_angle,
#             turbine_hub_height,
#             turbine_diameter,
#             mesh_x_rotated,
#             mesh_y_rotated,
#             mesh_z,
#             x_coord_rotated[:, :, :, i, :, :][:, :, :, na, :, :],
#             y_coord_rotated[:, :, :, i, :, :][:, :, :, na, :, :],
#             deflection_field,
#         )

#         # Perform wake/field combinations
#         u_wake = np.sqrt((u_wake ** 2) + (np.array(turb_u_wake) ** 2))
#         flow_field_u = flow_field_u_initial - u_wake
#         flow_field_v = flow_field_v + turb_v_wake
#         flow_field_w = flow_field_w + turb_w_wake

#         # Calculate wake overlap for wake-added turbulence (WAT)
#         turb_wake_field = flow_field_u_initial - turb_u_wake
#         area_overlap = calculate_area_overlap(
#             turb_wake_field, flow_field_u_initial, y_ngrid, z_ngrid
#         )

#         # Calculate WAT for turbines
#         WAT_TIs = crespo_hernandez(
#             ambient_TIs,
#             x_coord_rotated,
#             x_coord_rotated[:, :, :, i, :, :][:, :, :, na, :, :],
#             turbine_diameter,
#             turb_aIs[:, :, :, i][:, :, :, na, na, na],
#         )

#         # Modify WAT by wake area overlap
#         # TODO: will need to make the rotor_diameter part of this mask work for
#         # turbines of different types
#         downstream_influence_length = 15 * turbine_diameter
#         ti_added = (
#             area_overlap[:, :, :, :, na, na]
#             * np.nan_to_num(WAT_TIs, posinf=0.0)
#             * (
#                 np.array(
#                     x_coord_rotated > x_coord_rotated[:, :, :, i, :, :][:, :, :, na, :, :]
#                 )
#             )
#             * (
#                 np.array(
#                     np.abs(
#                         y_coord_rotated[:, :, :, i, :, :][:, :, :, na, :, :]
#                         - y_coord_rotated
#                     )
#                     < 2 * turbine_diameter
#                 )
#             )
#             * (
#                 np.array(
#                     x_coord_rotated
#                     <= downstream_influence_length
#                     + x_coord_rotated[:, :, :, i, :, :][:, :, :, na, :, :]
#                 )
#             )
#         )

#         # Combine turbine TIs with WAT
#         turb_TIs = np.maximum(np.sqrt(ti_added ** 2 + ambient_TIs ** 2), turb_TIs,)

#     return np.mean(flow_field_u)


# tic = time.perf_counter()
# calc_wake_grad = grad(calculate_wake)
# toc = time.perf_counter()

# print(f"Created gradient in {toc - tic:0.4f} seconds")

# tic = time.perf_counter()
# calc_wake_grad(
#     x_coord_rotated,
#     y_coord_rotated,
#     mesh_x_rotated,
#     mesh_y_rotated,
#     mesh_z,
#     flow_field_u_initial,
#     flow_field_u,
#     flow_field_v,
#     flow_field_w,
#     turb_inflow_field,
#     u_wake,
#     turbine_TSR,
#     wind_speed,
#     thrust,
#     turb_TIs,
#     yaw_angle,
#     turbine_hub_height,
#     turbine_diameter,
#     specified_wind_height,
#     wind_shear,
# )
# # print('gradient: ', calc_wake_grad)
# toc = time.perf_counter()
# print(f"Computed gradient in {toc - tic:0.4f} seconds")

# tic = time.perf_counter()
# calc_wake_grad(
#     x_coord_rotated,
#     y_coord_rotated,
#     mesh_x_rotated,
#     mesh_y_rotated,
#     mesh_z,
#     flow_field_u_initial,
#     flow_field_u,
#     flow_field_v,
#     flow_field_w,
#     turb_inflow_field,
#     u_wake,
#     turbine_TSR,
#     wind_speed,
#     thrust,
#     turb_TIs,
#     yaw_angle,
#     turbine_hub_height,
#     turbine_diameter,
#     specified_wind_height,
#     wind_shear,
# )
# # print('gradient: ', calc_wake_grad)
# toc = time.perf_counter()
# print(f"Computed gradient in {toc - tic:0.4f} seconds")

# tic = time.perf_counter()
# calc_wake_grad(
#     x_coord_rotated,
#     y_coord_rotated,
#     mesh_x_rotated,
#     mesh_y_rotated,
#     mesh_z,
#     flow_field_u_initial,
#     flow_field_u,
#     flow_field_v,
#     flow_field_w,
#     turb_inflow_field,
#     u_wake,
#     turbine_TSR,
#     wind_speed,
#     thrust,
#     turb_TIs,
#     yaw_angle,
#     turbine_hub_height,
#     turbine_diameter,
#     specified_wind_height,
#     wind_shear,
# )
# # print('gradient: ', calc_wake_grad)
# toc = time.perf_counter()
# print(f"Computed gradient in {toc - tic:0.4f} seconds")

# tic = time.perf_counter()
# calc_wake_grad(
#     x_coord_rotated,
#     y_coord_rotated,
#     mesh_x_rotated,
#     mesh_y_rotated,
#     mesh_z,
#     flow_field_u_initial,
#     flow_field_u,
#     flow_field_v,
#     flow_field_w,
#     turb_inflow_field,
#     u_wake,
#     turbine_TSR,
#     wind_speed,
#     thrust,
#     turb_TIs,
#     yaw_angle,
#     turbine_hub_height,
#     turbine_diameter,
#     specified_wind_height,
#     wind_shear,
# )
# # print('gradient: ', calc_wake_grad)
# toc = time.perf_counter()
# print(f"Computed gradient in {toc - tic:0.4f} seconds")

# tic = time.perf_counter()
# calc_wake_grad(
#     x_coord_rotated,
#     y_coord_rotated,
#     mesh_x_rotated,
#     mesh_y_rotated,
#     mesh_z,
#     flow_field_u_initial,
#     flow_field_u,
#     flow_field_v,
#     flow_field_w,
#     turb_inflow_field,
#     u_wake,
#     turbine_TSR,
#     wind_speed,
#     thrust,
#     turb_TIs,
#     yaw_angle,
#     turbine_hub_height,
#     turbine_diameter,
#     specified_wind_height,
#     wind_shear,
# )
# # print('gradient: ', calc_wake_grad)
# toc = time.perf_counter()
# print(f"Computed gradient in {toc - tic:0.4f} seconds")
