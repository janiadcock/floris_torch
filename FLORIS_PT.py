import yaml
import os
import torch
import numpy as np

class Loader(yaml.SafeLoader):

    def __init__(self, stream):

        self._root = os.path.split(stream.name)[0]

        super().__init__(stream)

    def include(self, node):

        filename = os.path.join(self._root, self.construct_scalar(node))

        with open(filename, 'r') as f:
            return yaml.load(f, self.__class__)
        
def cosd(angle):
    return torch.cos(torch.deg2rad(angle))

def sind(angle):
    return torch.sin(torch.deg2rad(angle))

def tand(angle):
    return torch.tan(torch.deg2rad(angle))

def interp(x, y, x_new):
    x_new_indices = torch.searchsorted(x, x_new)
    x_new_indices = x_new_indices.clip(1,len(x)-1)
    lo = x_new_indices - 1
    hi = x_new_indices
    x_lo = x[lo]
    x_hi = x[hi]
    y_lo = y[lo]
    y_hi = y[hi]
    slope = (y_hi - y_lo) / (x_hi - x_lo)
    return slope*(x_new - x_lo) + y_lo

def gaussian_function(U, C, r, n, sigma):
    return C * torch.exp(-1 * r ** n / (2 * sigma ** 2))

class FLORIS_PT():
    turbine_grid_points: int
    turbine_type: str
    flow_field: dict
    wake: dict
    turbine: dict
    
    # initialize from input file path and turbine file path
    def __init__(self, input_file_path):
        with open(input_file_path) as input_file:
            input_dict = yaml.load(input_file, Loader)
            
        self.turbine_grid_points = input_dict["turbine_grid_points"]
        self.turbine_type = input_dict["turbine_type"][0]
        self.flow_field = input_dict["flow_field"]
        self.wake = input_dict["wake"]
        
        turbine_file_path = './' + \
            self.turbine_type+'.yaml'
        with open(turbine_file_path) as input_file:
            self.turbine = yaml.load(input_file, Loader)
            
    # generate mesh on each turbine's rotor disk
    # rotate mesh based on wd to align with 270 degrees
    def get_turbine_mesh(self, wd, x_coord_full, y_coord_full, z_coord_full, active_turbs=None):
        # apply dropout
        if active_turbs is not None:
            x_coord = x_coord_full[active_turbs]
            y_coord = y_coord_full[active_turbs]
            z_coord = z_coord_full[active_turbs]
        
        y_ngrid = self.turbine_grid_points
        z_ngrid = self.turbine_grid_points # could change to allow diff. value than y_ngrid
        x_grid = torch.zeros((len(x_coord), y_ngrid, z_ngrid))
        y_grid = torch.zeros((len(x_coord), y_ngrid, z_ngrid))
        z_grid = torch.zeros((len(x_coord), y_ngrid, z_ngrid))

        angle = ((wd - 270) % 360 + 360) % 360

        x1, x2, x3 = x_coord, y_coord, z_coord

        rloc = 0.5
        pt = rloc * self.turbine['rotor_diameter'] / 2.

        # linspace for an array; torch.linspace only takes scalar
        steps = torch.arange(y_ngrid).unsqueeze(-1) / (y_ngrid - 1)
        yt = (x2 - pt) + steps*2.*pt

        steps = torch.arange(z_ngrid).unsqueeze(-1) / (z_ngrid - 1)
        zt = (x3 - pt) + steps*2.*pt

        x_grid = torch.ones((len(x_coord), y_ngrid, z_ngrid)) * x_coord[:, None, None]
        y_grid = torch.ones((len(x_coord), y_ngrid, z_ngrid)) * yt.T[:, :, None]
        z_grid = torch.ones((len(x_coord), y_ngrid, z_ngrid)) * zt.T[:, None, :]

        # yaw turbines to be perpendicular to the wind direction
        wind_direction_i = angle[None, :, None, None, None, None]
        xoffset = x_grid - x1[:, None, None]
        yoffset = y_grid - x2[:, None, None]
        wind_cos = cosd(-wind_direction_i)
        wind_sin = sind(-wind_direction_i)
        x_grid = xoffset * wind_cos - yoffset * wind_sin + x1[:, None, None]
        y_grid = yoffset * wind_cos + xoffset * wind_sin + x2[:, None, None]

        mesh_x = x_grid
        mesh_y = y_grid
        mesh_z = z_grid

        # rotate turbine locations/fields to be perpendicular to wind direction
        x_center_of_rotation = torch.mean(torch.stack([torch.min(mesh_x), torch.max(mesh_x)]))
        y_center_of_rotation = torch.mean(torch.stack([torch.min(mesh_y), torch.max(mesh_y)]))
        angle = ((wd[None, :, None, None, None, None] - 270) % 360 + 360) % 360
        x_offset = mesh_x - x_center_of_rotation
        y_offset = mesh_y - y_center_of_rotation
        mesh_x_rotated = (x_offset * cosd(angle) - y_offset * sind(angle) 
                          + x_center_of_rotation)
        mesh_y_rotated = (x_offset * sind(angle) + y_offset * cosd(angle) 
                          + y_center_of_rotation)
        x_coord_offset = (x_coord - x_center_of_rotation)[:, None, None]
        y_coord_offset = (y_coord - y_center_of_rotation)[:, None, None]
        x_coord_rotated = (x_coord_offset * cosd(angle)
            - y_coord_offset * sind(angle)
            + x_center_of_rotation)
        y_coord_rotated = (x_coord_offset * sind(angle)
            + y_coord_offset * cosd(angle)
            + y_center_of_rotation)
        inds_sorted = x_coord_rotated.argsort(axis=3)

        x_coord_rotated = torch.gather(x_coord_rotated, 3, inds_sorted)
        y_coord_rotated = torch.gather(y_coord_rotated, 3, inds_sorted)

        mesh_x_rotated = torch.take_along_dim(mesh_x_rotated, inds_sorted, 3)
        mesh_y_rotated = torch.take_along_dim(mesh_y_rotated, inds_sorted, 3)

        return x_coord_rotated, y_coord_rotated, mesh_x_rotated, \
            mesh_y_rotated, mesh_z, inds_sorted, x_coord
    
    def get_field_rotor(self, ws, wd, clipped_u, x_coord, x_coord_rotated, y_coord_rotated, 
        mesh_x_rotated, mesh_y_rotated, mesh_z, inds_sorted):

        # name constants
        wind_shear = self.flow_field['wind_shear']
        wind_veer = self.flow_field['wind_veer']
        TI = self.flow_field['turbulence_intensity']
        specified_wind_height = self.flow_field['specified_wind_height']
        turbine_hub_height = self.turbine['hub_height']
        rotor_diameter = self.turbine['rotor_diameter']
        wind_speed = torch.tensor(self.turbine['power_thrust_table']['wind_speed'])
        thrust = torch.tensor(self.turbine['power_thrust_table']['thrust'])
        y_ngrid = self.turbine_grid_points
        z_ngrid = self.turbine_grid_points # could change to allow diff. value than y_ngrid

        ## initialize flow field ##
        flow_field_u_initial = (ws[None, None, :, None, None, None] 
                            * (mesh_z / specified_wind_height) ** wind_shear) \
                            * torch.ones((1, len(wd), 1, 1, 1, 1))

        ## initialize other field values ##
        u_wake = torch.zeros(flow_field_u_initial.shape)
        flow_field_u = flow_field_u_initial - u_wake
        turb_inflow_field = torch.ones(flow_field_u_initial.shape) * flow_field_u_initial

        ## Initialize turbine values ##
        turb_TIs = torch.ones_like(x_coord_rotated) * TI
        ambient_TIs = torch.ones_like(x_coord_rotated) * TI
        yaw_angle = clipped_u.reshape(x_coord_rotated.shape)
        turbine_tilt = torch.ones_like(x_coord_rotated) * 0.0

        # Loop over turbines to solve wakes
        for i in range(len(x_coord)):
            turb_inflow_field = turb_inflow_field \
            * (mesh_x_rotated != x_coord_rotated[:, :, :, i, :, :][:, :, :, None, :, :]) \
            + (flow_field_u_initial - u_wake) \
            * (mesh_x_rotated == x_coord_rotated[:, :, :, i, :, :][:, :, :, None, :, :])

            turb_avg_vels = torch.pow(torch.mean(turb_inflow_field ** 3, dim=(4,5)), 1./3.)
            
            Ct = interp(wind_speed, thrust, turb_avg_vels)
            yaw_angle_flat = torch.squeeze(torch.squeeze(yaw_angle, dim=4), dim=4)
            Ct *= cosd(yaw_angle_flat) # effective thrust
            Ct = Ct * (Ct < 1.0) + 0.9999 * torch.ones_like(Ct) * (Ct >= 1.0)
            turb_Cts = Ct * (Ct > 0.0) + 0.0001 * torch.ones_like(Ct) * (Ct <= 0.0)

            turb_aIs = 0.5 / cosd(yaw_angle_flat) * (1 - torch.sqrt(1 - turb_Cts * cosd(yaw_angle_flat) + 1e-16))

            ## Wake deflection calculation ##
            yaw = yaw_angle # if no secondary steering    
            x_coord_rotated_i = x_coord_rotated[:, :, :, i, :, :][:, :, :, None, :, :]
            y_coord_rotated_i = y_coord_rotated[:, :, :, i, :, :][:, :, :, None, :, :]
            turbine_ti_i = turb_TIs[:, :, :, i, :, :][:, :, :, None, :, :]
            turbine_Ct_i = turb_Cts[:, :, :, i][:, :, :, None, None, None]
            yaw_i = yaw[:, :, :, i, :, :][:, :, :, None, :, :]
            neg_yaw_i = -1. * yaw_i # wake deflection has opposite sign convention

            ka = 0.38  # wake expansion parameter
            kb = 0.004  # wake expansion parameter
            alpha = 0.58  # near wake parameter
            beta = 0.077  # near wake parameter

            ad = 0.0  # natural lateral defleturbine_Ct_iion parameter
            bd = 0.0  # natural lateral defleturbine_Ct_iion parameter
            dm = 1.0

            U_local = flow_field_u_initial

            # initial velocity deficits
            uR = (U_local * turbine_Ct_i * cosd(turbine_tilt) * cosd(neg_yaw_i)
                  / (2.0 * (1 - torch.sqrt(1 - (turbine_Ct_i * cosd(turbine_tilt) * cosd(neg_yaw_i)) + 1e-16))))
            u0 = U_local * torch.sqrt(1 - turbine_Ct_i + 1e-16)

            # length of near wake
            x0 = (rotor_diameter
                  * (cosd(neg_yaw_i) * (1 + torch.sqrt(1 - turbine_Ct_i * cosd(neg_yaw_i) + 1e-16)))
                  / (np.sqrt(2) * (4 * alpha * turbine_ti_i + 2 * beta * (1 - torch.sqrt(1 - turbine_Ct_i + 1e-16))))
                  + x_coord_rotated_i)

            # wake expansion parameters
            ky = ka * turbine_ti_i + kb
            kz = ka * turbine_ti_i + kb

            C0 = 1 - u0 / flow_field_u_initial
            M0 = C0 * (2 - C0)
            E0 = C0 ** 2 - 3 * np.exp(1.0 / 12.0) * C0 + 3 * np.exp(1.0 / 3.0)

            # initial Gaussian wake expansion
            sigma_z0 = rotor_diameter * 0.5 * torch.sqrt(uR / (U_local + u0) + 1e-16)
            sigma_y0 = sigma_z0 * cosd(neg_yaw_i) * cosd(torch.tensor(wind_veer))

            yR = mesh_y_rotated - y_coord_rotated_i
            xR = x_coord_rotated_i

            # yaw_i parameter (skew angle)
            theta_c0 = (dm * (0.3 * torch.deg2rad(neg_yaw_i) / cosd(neg_yaw_i)) 
                        * (1 - torch.sqrt(1 - turbine_Ct_i * cosd(neg_yaw_i) + 1e-16))) # skew angle in radians

            # yaw_i param (distance from centerline=initial wake deflection)
            # NOTE: use tan here since theta_c0 is radians
            delta0 = torch.tan(theta_c0) * (x0 - x_coord_rotated_i) 

            # deflection in the near wake
            delta_near_wake = ((mesh_x_rotated - xR) / (x0 - xR)) * delta0 \
                + (ad + bd * (mesh_x_rotated - x_coord_rotated_i))
            delta_near_wake = delta_near_wake * (mesh_x_rotated >= xR)
            delta_near_wake = delta_near_wake * (mesh_x_rotated <= x0)

            # deflection in the far wake
            sigma_y = ky * (mesh_x_rotated - x0) + sigma_y0
            sigma_z = kz * (mesh_x_rotated - x0) + sigma_z0
            sigma_y = sigma_y * (mesh_x_rotated >= x0) + sigma_y0 * (mesh_x_rotated < x0)
            sigma_z = sigma_z * (mesh_x_rotated >= x0) + sigma_z0 * (mesh_x_rotated < x0)

            ln_deltaNum = (1.6 + torch.sqrt(M0 + 1e-16)) \
                * (1.6 * torch.sqrt(sigma_y * sigma_z / (sigma_y0 * sigma_z0) + 1e-16) - torch.sqrt(M0 + 1e-16))
            ln_deltaDen = (1.6 - torch.sqrt(M0)) \
                * (1.6 * torch.sqrt(sigma_y * sigma_z / (sigma_y0 * sigma_z0) + 1e-16) + torch.sqrt(M0 + 1e-16))
            delta_far_wake = (delta0 + (theta_c0 * E0 / 5.2)
                              * torch.sqrt(sigma_y0 * sigma_z0 / (ky * kz * M0) + 1e-16)
                              * torch.log(ln_deltaNum / ln_deltaDen)
                              + (ad + bd * (mesh_x_rotated - x_coord_rotated_i)))
            delta_far_wake = delta_far_wake * (mesh_x_rotated > x0)
            deflection_field = delta_near_wake + delta_far_wake

            ## Calculate wake deficits ##
            # wake deflection
            delta = deflection_field

            # mask upstream wake
            xR = x_coord_rotated_i

            # mask upstream wake
            #   initial velocity deficits
            uR = U_local * turbine_Ct_i / (2.0 * (1. - torch.sqrt(1. - turbine_Ct_i + 1e-16)))
            u0 = U_local * torch.sqrt(1. - turbine_Ct_i + 1e-16)
            #   initial wake expansion
            sigma_z0 = rotor_diameter * 0.5 * torch.sqrt(uR / (U_local + u0) + 1e-16)
            sigma_y0 = sigma_z0 * cosd(neg_yaw_i) * cosd(torch.tensor(wind_veer))

            # quantity that determines when the far wake starts
            x0 = (rotor_diameter * (cosd(neg_yaw_i) * (1. + torch.sqrt(1. - turbine_Ct_i + 1e-16)))
                  / (np.sqrt(2.) * (4. * alpha * turbine_ti_i + 2. * beta 
                                       * (1. - torch.sqrt(1. - turbine_Ct_i + 1e-16))))
                  + x_coord_rotated_i)

            # velocity deficit in the near wake
            turb_u_wake = torch.zeros_like(U_local)
            near_wake_mask = (mesh_x_rotated > xR + 0.1) * (mesh_x_rotated < x0)
            if torch.sum(near_wake_mask):
                sigma_y = (((x0 - xR) - (mesh_x_rotated - xR)) / (x0 - xR)) * 0.501 * rotor_diameter \
                    * torch.sqrt(turbine_Ct_i / 2. + 1e-16) + ((mesh_x_rotated - xR) / (x0 - xR)) * sigma_y0
                sigma_z = (((x0 - xR) - (mesh_x_rotated - xR)) / (x0 - xR)) * 0.501 * rotor_diameter \
                    * torch.sqrt(turbine_Ct_i / 2. + 1e-16) + ((mesh_x_rotated - xR) / (x0 - xR)) * sigma_z0

                sigma_y = (sigma_y * (mesh_x_rotated >= xR) + torch.ones_like(sigma_y) 
                           * (mesh_x_rotated < xR) * 0.5 * rotor_diameter)
                sigma_z = (sigma_z * (mesh_x_rotated >= xR) + torch.ones_like(sigma_z) 
                           * (mesh_x_rotated < xR) * 0.5 * rotor_diameter)

                wind_veer_tensor = torch.tensor(wind_veer)
                a = cosd(wind_veer_tensor) ** 2 / (2 * sigma_y ** 2) + sind(wind_veer_tensor) ** 2 / (2 * sigma_z ** 2)
                b = -sind(2 * wind_veer_tensor) / (4 * sigma_y ** 2) + sind(2 * wind_veer_tensor) / (4 * sigma_z ** 2)
                c = sind(wind_veer_tensor) ** 2 / (2 * sigma_y ** 2) + cosd(wind_veer_tensor) ** 2 / (2 * sigma_z ** 2)
                r = (a * ((mesh_y_rotated - y_coord_rotated_i) - delta) ** 2
                     - 2 * b * ((mesh_y_rotated - y_coord_rotated_i) - delta) * ((mesh_z - turbine_hub_height))
                     + c * ((mesh_z - turbine_hub_height)) ** 2)

                C = 1 - torch.sqrt(torch.clip(1 - (turbine_Ct_i * cosd(neg_yaw_i) 
                    / (8.0 * sigma_y * sigma_z / rotor_diameter ** 2)), 0.0, 1.0) + 1e-16)

                turb_u_wake = gaussian_function(U_local, C, r, 1, np.sqrt(0.5)) * near_wake_mask

            far_wake_mask = (mesh_x_rotated >= x0)
            if torch.sum(far_wake_mask):
                # wake expansion in the lateral (y) and the vertical (z)
                sigma_y = ky * (mesh_x_rotated - x0) + sigma_y0
                sigma_z = kz * (mesh_x_rotated - x0) + sigma_z0
                sigma_y = sigma_y * (mesh_x_rotated >= x0) + sigma_y0 * (mesh_x_rotated < x0)
                sigma_z = sigma_z * (mesh_x_rotated >= x0) + sigma_z0 * (mesh_x_rotated < x0)

                # velocity deficit outside the near wake
                wind_veer_tensor = torch.tensor(wind_veer)
                a = cosd(wind_veer_tensor) ** 2 / (2 * sigma_y ** 2) + sind(wind_veer_tensor) ** 2 / (2 * sigma_z ** 2)
                b = -sind(2 * wind_veer_tensor) / (4 * sigma_y ** 2) + sind(2 * wind_veer_tensor) / (4 * sigma_z ** 2)
                c = sind(wind_veer_tensor) ** 2 / (2 * sigma_y ** 2) + cosd(wind_veer_tensor) ** 2 / (2 * sigma_z ** 2)
                r = (a * (mesh_y_rotated - y_coord_rotated_i - delta) ** 2
                     - 2 * b * (mesh_y_rotated - y_coord_rotated_i - delta) * (mesh_z - turbine_hub_height)
                     + c * (mesh_z - turbine_hub_height) ** 2)
                C = 1 - torch.sqrt(torch.clip(1 - (turbine_Ct_i * cosd(neg_yaw_i) 
                    / (8.0 * sigma_y * sigma_z / rotor_diameter ** 2)), 0.0, 1.0) + 1e-16)

                # compute velocities in the far wake
                turb_u_wake1 = gaussian_function(U_local, C, r, 1, np.sqrt(0.5)) * far_wake_mask
                turb_u_wake += turb_u_wake1

            ## Perform wake/field combinations ##
            u_wake = torch.sqrt((u_wake ** 2) + ((turb_u_wake * flow_field_u_initial) ** 2) + 1e-16)
            flow_field_u = flow_field_u_initial - u_wake

            ## Calculate wake overlap for wake-added turbulence (WAT) ##
            area_overlap = torch.sum(turb_u_wake * flow_field_u_initial > 0.05, axis=(4, 5)) / (y_ngrid * z_ngrid)

            ## Calculate WAT for turbines ##
            # turbulence intensity calculation based on Crespo et. al.
            turbine_aI_i = turb_aIs[:, :, :, i][:, :, :, None, None, None]
            ti_initial = 0.1
            ti_constant = 0.5
            ti_ai = 0.8
            ti_downstream = -0.32

            # replace zeros and negatives with 1 to prevent nans/infs
            # keep downstream components; set upstream to 1.0
            delta_x = mesh_x_rotated - x_coord_rotated_i
            upstream_mask = (delta_x <= 0.1)
            downstream_mask = (delta_x > -0.1)
            delta_x = delta_x * downstream_mask + torch.ones_like(delta_x) * upstream_mask
            ti_calculation = (ti_constant * turbine_aI_i ** ti_ai * ambient_TIs ** ti_initial \
                              * (delta_x / rotor_diameter) ** ti_downstream)

            # mask the 1 values from above w/ zeros
            WAT_TIs = ti_calculation * downstream_mask

            ## Modify WAT by wake area overlap ##
            # TODO: will need to make the rotor_diameter part of this mask work for
            # turbines of different types
            downstream_influence_length = 15 * rotor_diameter
            ti_added = (area_overlap[:, :, :, :, None, None] * torch.nan_to_num(WAT_TIs, posinf=0.0) 
                * (x_coord_rotated > x_coord_rotated[:, :, :, i, :, :][:, :, :, None, :, :])
                * (torch.abs(y_coord_rotated[:, :, :, i, :, :][:, :, :, None, :, :] - y_coord_rotated) 
                   < 2 * rotor_diameter)
                * (x_coord_rotated <= downstream_influence_length
                   + x_coord_rotated[:, :, :, i, :, :][:, :, :, None, :, :]))

            ## Combine turbine TIs with WAT
            turb_TIs = torch.maximum(torch.sqrt(ti_added ** 2 + ambient_TIs ** 2 + 1e-16), turb_TIs,)    

        return flow_field_u, yaw_angle
    
    # calculate power produced by each turbine in farm
    def get_power(self, flow_field_u, x_coord_rotated, yaw_angle):
        ## power calculation (based on main floris branch) ##
        # omitted fCp_interp b/c interp power from wind_speed to wind_speed so does nothing...
        
        rotor_radius = self.turbine['rotor_diameter']/2.
        power = torch.tensor(self.turbine['power_thrust_table']['power'])
        wind_speed = torch.tensor(self.turbine['power_thrust_table']['wind_speed'])
        generator_efficiency = self.turbine['generator_efficiency']
        pT = self.turbine['pT']
        pP = self.turbine['pP']
        air_density = self.flow_field['air_density']
        
        rotor_area = torch.pi * rotor_radius ** 2.0
        inner_power = 0.5*rotor_area*power*generator_efficiency*wind_speed**3

        # omiting some lines here b/c assuming all turbines have same type
        # ix_filter not implemented

        # Compute the yaw effective velocity
        pPs = torch.ones_like(x_coord_rotated) * pP
        pW = pPs / 3.0  # Convert from pP to w
        axis = tuple([4 + i for i in range(flow_field_u.ndim - 4)])
        average_velocity = torch.pow(torch.mean(flow_field_u ** 3, axis=axis), 1./3.).reshape(yaw_angle.shape)
        yaw_effective_velocity = ((air_density/1.225)**(1/3)) * average_velocity * cosd(yaw_angle) ** pW

        # Power produced by a turbine adjusted for yaw and tilt. Value given in kW
        p = 1.225 * interp(wind_speed, inner_power, yaw_effective_velocity) / 1000.0

        # negative sign on power b/c good -> want to minimize
        return p