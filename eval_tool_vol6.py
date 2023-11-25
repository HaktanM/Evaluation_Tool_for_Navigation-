import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import cv2
import os
from scipy.spatial.transform import Rotation as orientation

def perform_assocation(ref_time_arr, target_time_arr, target_data):
    ref_size = ref_time_arr.shape[0]
    target_alignment_statu = np.zeros(ref_size,dtype=bool)
    target_aligned_data = np.zeros((ref_size,target_data.shape[1]),dtype=float)

    max_difference = 0.02 * 1e9
    for ref_time_index in range(ref_size):
        curr_ref_time = ref_time_arr[ref_time_index]
        difference_array = np.absolute(target_time_arr-curr_ref_time)
        best_target_idx = difference_array.argmin()
        best_diff = difference_array[best_target_idx]
        if best_diff <= max_difference:
            target_alignment_statu[ref_time_index] = True
            target_aligned_data[ref_time_index,:] = target_data[best_target_idx,:]
        else:
            target_alignment_statu[ref_time_index] = False
    return target_alignment_statu, target_aligned_data



class _eval:
    def __init__(self,dir_gt):
        if dir_gt[-3:] == "txt":
            data = np.loadtxt(dir_gt, dtype=np.float64, comments='#', delimiter=" ")
        elif dir_gt[-3:] == "csv":
            data = np.loadtxt(dir_gt, dtype=np.float64, comments='#', delimiter=",")
        else:
            print("ERROR: eval is not initialized properly. Only txt and csv extensions are allowed when uploading data.")

        self.time = data[:,0]
        self.gt = data[:,1:11]
        self.dataLength = self.time.shape[0]
        
        self.estimations = {}
        self.__assocData = {}
        self.__assocStatus = {}
        self.__assocAll = np.ones(self.dataLength,dtype=bool)    
        self.num_of_estimated_var = 100   
        
        self.show = True
        self.plotVar = False
        self.save = True

        self.haslabels = False
        self.labels = {}
        
        self.__hasCov = True
        self.__association_performed = False
        self.__estimations_are_cropped = False
        self.__count = 0
        
        
    def addEstimation(self, dir_es):
        if dir_es[-3:] == "txt":
            data = np.loadtxt(dir_es, dtype=np.float64, comments='#')
        elif dir_es[-3:] == "csv":
            data = np.loadtxt(dir_es, dtype=np.float64, comments='#', delimiter=",")
        else:
            print("ERROR: Estimation file (which is: " + dir_es + " ) cannot be uploaded. Only txt and csv extensions are allowed when uploading data.")

        self.estimations.update({self.__count: data})
 
        if data.shape[1] <= 20:            
            self.__hasCov = False
            print(f"The estimation has no covariance. Hence, the parameter (__hasCov) set to be False. (Estimation : {dir_es})")

        self.num_of_estimated_var = np.minimum(self.num_of_estimated_var,data.shape[1]-1)
        
        self.__count += 1
        self.__association_performed = False
        self.__estimations_are_cropped = False
      
        
    def performAssociation(self):
        for i in range(self.__count):
            gt_time = self.time
            es_time = self.estimations[i][:,0]
            es_data = self.estimations[i][:,1:]
            assocStatu, assocData = perform_assocation(gt_time, es_time, es_data)
            self.__assocData.update({i:assocData})
            self.__assocStatus.update({i:assocStatu})
            self.__assocAll = np.logical_and(self.__assocAll,assocStatu)
        self.__association_performed = True
        
    
    def crop_estimations(self):
        if not self.__association_performed:
            self.performAssociation()
            
        self.__assocData_cropped = {}        
        
        time = self.time
        gt = self.gt
        cropped_time = np.zeros(0)
        cropped_gt = np.zeros(0)
        assoc_counter = 0
        for idx,associtaion_statu in enumerate(self.__assocAll):
            if associtaion_statu:
                cropped_time = np.append(cropped_time,time[idx])
                cropped_gt = np.append(cropped_gt,gt[idx,:])
                assoc_counter +=1
                
        cropped_gt = cropped_gt.reshape(assoc_counter,gt.shape[1])
        self.__cropped_gt = cropped_gt
        self.__cropped_time = cropped_time

        
        croped_estimations = np.zeros((assoc_counter,self.num_of_estimated_var,self.__count))
        """
        First index stands for the time where the association is happened.
        Second index is the variables belonging to a single estimation such as pose, velocity, covariances
        Third term is the estimation index
        """
        
        
        """
        The loop checks where the association is common for all estimations.
        Only the associated data is saved in croped_estimations. The rest is cropped.
        """  
        assoc_idx = 0
        for time_idx,associtaion_statu in enumerate(self.__assocAll):
            if associtaion_statu:
                for est_idx in range(self.__count):
                    estimation = self.__assocData[est_idx][:,0:self.num_of_estimated_var]
                    croped_estimations[assoc_idx,:,est_idx] = estimation[time_idx,:]
                assoc_idx += 1
              
                
        for est_idx in range(self.__count):
            self.__assocData_cropped.update({est_idx:croped_estimations[:,:,est_idx]})
            
        self.__estimations_are_cropped = True
        
    def get_position_from_estimation(self,est_idx):
        return self.__assocData_cropped[est_idx][:,0:3]
    
    def get_orientation_from_estimation(self,est_idx):
        return self.__assocData_cropped[est_idx][:,3:7]
    
    def get_orientation_covariance_from_estimation(self,est_idx):
        return self.__assocData_cropped[est_idx][:,7:13]
    
    def get_position_covariance_from_estimation(self,est_idx):
        return self.__assocData_cropped[est_idx][:,13:19]
    
    def get_linear_velocity_from_estimation(self,est_idx):
        return self.__assocData_cropped[est_idx][:,19:22]
    
    def get_angular_velocity_from_estimation(self,est_idx):
        return self.__assocData_cropped[est_idx][:,22:25]

    def get_gyr_bias_from_estimation(self,est_idx):
        return self.__assocData_cropped[est_idx][:,25:28]
    
    def get_acc_bias_from_estimation(self,est_idx):
        return self.__assocData_cropped[est_idx][:,28:31]
    
    def get_linear_velocity_covariance_from_estimation(self,est_idx):
        return self.__assocData_cropped[est_idx][:,31:34]
    
    def get_angular_velocity_covariance_from_estimation(self,est_idx):
        return self.__assocData_cropped[est_idx][:,34:37]
    
    def get_gyr_bias_covariance_from_estimation(self,est_idx):
        return self.__assocData_cropped[est_idx][:,37:40]
    
    def get_acc_bias_covariance_from_estimation(self,est_idx):
        return self.__assocData_cropped[est_idx][:,40:43]
    
    def getAlignedTraj(self, saving_folder):
        if not self.__association_performed:
            self.performAssociation()
        
        if not self.__estimations_are_cropped:
            self.crop_estimations()

        if not os.path.exists(saving_folder):
            os.makedirs(saving_folder)


        for est_idx in range(self.__count):
            # Save data in evo format
            saving_path = os.path.join(saving_folder, str(est_idx) + '.csv')
            with open(saving_path, 'w') as file:
                pass

            cropped_estimation_position = self.get_position_from_estimation(est_idx)
            cropped_estimation_orientation = self.get_orientation_from_estimation(est_idx)
            time = self.__cropped_time
            for time_idx in range(time.shape[0]):
                timestamp = time[time_idx]*1e9
                current_loc = cropped_estimation_position[time_idx,:]
                current_quat =  cropped_estimation_orientation[time_idx,:]
                with open(saving_path, 'a') as file:
                    file.write(f"{int(timestamp)} {current_loc[0]} {current_loc[1]} {current_loc[2]} {current_quat[0]} {current_quat[1]} {current_quat[2]} {current_quat[3]}\n")
    
            
    
    def create_xyz_plots(self,file_name):
        if not self.__association_performed:
            self.performAssociation()
        
        if not self.__estimations_are_cropped:
            self.crop_estimations()

        gt_position= self.__cropped_gt[:,0:3]
        time = (self.__cropped_time - self.__cropped_time[0]) * 1e-9

        fig, ax = plt.subplots(3,figsize=(6,6))
        self.last = -5
        ax[0].plot(time, gt_position[:,0],color="red",linewidth=1.5,label="Ground Truth")
        ax[1].plot(time, gt_position[:,1],color="red",linewidth=1.5)
        ax[2].plot(time, gt_position[:,2],color="red",linewidth=1.5)

        aligned_estimation_mean = np.zeros((time.shape[0],3), dtype=np.float64)
        for est_idx in range(self.__count):
            cropped_estimations_position = self.get_position_from_estimation(est_idx)
            aligned_estimation_mean += cropped_estimations_position/self.__count
            ax[0].plot(time, cropped_estimations_position[:,0],"--",linewidth=1.5)
            ax[1].plot(time, cropped_estimations_position[:,1],"--",linewidth=1.5)
            ax[2].plot(time, cropped_estimations_position[:,2],"--",linewidth=1.5)

        # ax[0].plot(time, cropped_estimations_position[:,0],"--", color="blue",linewidth=1.5,label="Estimations")
        # ax[1].plot(time, cropped_estimations_position[:,1],"--", color="blue",linewidth=1.5,label="Estimations")
        # ax[2].plot(time, cropped_estimations_position[:,2],"--", color="blue",linewidth=1.5,label="Estimations")
        
        # ax[0].plot(time, aligned_estimation_mean[:,0],linewidth=2.5, color="black",label="Mean")
        # ax[1].plot(time, aligned_estimation_mean[:,1],linewidth=2.5, color="black",label="Mean")
        # ax[2].plot(time, aligned_estimation_mean[:,2],linewidth=2.5, color="black",label="Mean")

        ax[0].legend()
        ax[2].set_xlabel("time (sec)")
        ax[0].set_ylabel("x")
        ax[2].set_ylabel("z")
        ax[1].set_ylabel("location (meter)")
        ax[0].set_title("x,y,z Axes versus time")

        ax[0].grid()
        ax[1].grid()
        ax[2].grid()

        if self.save: 
            fig.savefig(file_name)

        fig.canvas.draw()
        image_from_plot1 = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot1 = image_from_plot1.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        image_from_plot1 = cv2.cvtColor(image_from_plot1, cv2.COLOR_RGB2BGR)

        if self.show:
            cv2.imshow("x y z", image_from_plot1)
            cv2.waitKey() 
            
    def create_traj_plots(self,file_name):
        if not self.__association_performed:
            self.performAssociation()
        
        if not self.__estimations_are_cropped:
            self.crop_estimations()

        gt_position = self.__cropped_gt[:,0:3]

        fig, ax = plt.subplots(1,figsize=(6,6))
        ax.plot(gt_position[:,0],gt_position[:,1],color="red",linewidth=1.5,label="Ground Truth")

        cropped_estimation_mean = np.zeros((gt_position.shape[0],3), dtype=np.float64)
        for est_idx in range(self.__count):
            cropped_estimation = self.get_position_from_estimation(est_idx)
            cropped_estimation_mean += cropped_estimation/self.__count
            ax.plot(cropped_estimation[:,0],cropped_estimation[:,1],"--",linewidth=1.5)
    
        #ax.plot(cropped_estimation[:,0],cropped_estimation[:,1],"--",color="blue",linewidth=1.5,label="Estimations")
        ax.plot(cropped_estimation_mean[:,0], cropped_estimation_mean[:,1],linewidth=2.5, color="black",label="Estimations Mean")

        ax.legend()
        ax.set_xlabel("meter")
        ax.set_ylabel("meter")
        ax.set_title("x-y Trajectory")
        ax.set_aspect('equal', adjustable='datalim')
        # ax.set_xlim([0,40000])
        # ax.set_ylim([0,40000])
        ax.grid()
        
        if self.save: 
            fig.savefig(file_name)

        fig.canvas.draw()
        image_from_plot1 = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot1 = image_from_plot1.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        image_from_plot1 = cv2.cvtColor(image_from_plot1, cv2.COLOR_RGB2BGR)

        if self.show:
            cv2.imshow("Trajectory", image_from_plot1)
            cv2.waitKey()
            
    def create_error_plots(self,file_name):
        if not self.__association_performed:
            self.performAssociation()
        
        if not self.__estimations_are_cropped:
            self.crop_estimations()
        
        
        gt_position= self.__cropped_gt[:,0:3]
        time = (self.__cropped_time - self.__cropped_time[0]) * 1e-9

        fig, ax = plt.subplots(3,figsize=(6,6))
        error_mean = np.zeros((time.shape[0],3), dtype=np.float64)
        for est_idx in range(self.__count):
            error = self.get_position_from_estimation(est_idx) - gt_position
            error_mean += error/self.__count
            if self.haslabels:
                ax[0].plot(time, error[:,0],"--",linewidth=1.5,label = self.labels[est_idx])
                ax[1].plot(time, error[:,1],"--",linewidth=1.5)
                ax[2].plot(time, error[:,2],"--",linewidth=1.5)
            else: 
                ax[0].plot(time, error[:,0],"--",linewidth=1.5)
                ax[1].plot(time, error[:,1],"--",linewidth=1.5)
                ax[2].plot(time, error[:,2],"--",linewidth=1.5)

        # ax[0].plot(time, error[:,0],"--",linewidth=1.5,color="blue",label="Individual Errors")
        # ax[1].plot(time, error[:,1],"--",linewidth=1.5,color="blue",label="Individual Errors")
        # ax[2].plot(time, error[:,2],"--",linewidth=1.5,color="blue",label="Individual Errors")
        
        
        if self.plotVar and self.__hasCov:
            var_mean = np.zeros((time.shape[0],3), dtype=np.float32)

            for est_idx in range(self.__count):
                transCov = self.get_position_covariance_from_estimation(est_idx)
                for time_idx in range(time.shape[0]):
                    cov = transCov[time_idx,:]
                    C = np.array([cov[0],cov[1],cov[2],   cov[1], cov[3], cov[4],   cov[2], cov[4], cov[5]], dtype=np.float64)
                    C = C.reshape(3,3)

                    sigma_x = np.sqrt(C[0,0])
                    
                    sigma_y = np.sqrt(C[1,1])
                    sigma_z = np.sqrt(C[2,2])
                    
                    var_mean[time_idx,0] += sigma_x / self.__count
                    var_mean[time_idx,1] += sigma_y / self.__count
                    var_mean[time_idx,2] += sigma_z / self.__count
                    
            ax[0].plot(time, var_mean[:,0],"--",linewidth=1.5,color="goldenrod",label = "± sigma")
            ax[0].plot(time, -var_mean[:,0],"--",linewidth=1.5,color="goldenrod")

            ax[1].plot(time, var_mean[:,1],"--",linewidth=1.5,color="goldenrod")
            ax[1].plot(time, -var_mean[:,1],"--",linewidth=1.5,color="goldenrod")

            ax[2].plot(time, var_mean[:,2],"--",linewidth=1.5,color="goldenrod")
            ax[2].plot(time, -var_mean[:,2],"--",linewidth=1.5,color="goldenrod")
        
        
        ax[0].legend()
        ax[2].set_xlabel("time (sec)")
        ax[0].set_ylabel("x")
        ax[1].set_ylabel("y")
        ax[2].set_ylabel("z")
        ax[0].set_title("Position Errors versus Time")

        ax[0].grid()
        ax[1].grid()
        ax[2].grid()
        
        # ax[0].set_ylim(-50,50)
        # ax[1].set_ylim(-25,25)
        # ax[2].set_ylim(-50,50)

        if self.save:
            fig.savefig(file_name)

        fig.canvas.draw()
        image_from_plot1 = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot1 = image_from_plot1.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        image_from_plot1 = cv2.cvtColor(image_from_plot1, cv2.COLOR_RGB2BGR)
        
        if self.show:
            cv2.imshow("Position Errors", image_from_plot1)
            cv2.waitKey()  
            
    def create_velocity_error(self,file_name="velocity_errors.png"):
        if not self.__association_performed:
            self.performAssociation()
        
        if not self.__estimations_are_cropped:
            self.crop_estimations()
            
        time = (self.__cropped_time - self.__cropped_time[0]) * 1e-9
        gt_velocity = self.__cropped_gt[:,7:10]
        
            
        fig, ax = plt.subplots(3,figsize=(6,6))
        vel_var_mean = np.zeros((time.shape[0],3))
        for est_idx in range(self.__count):
            error = self.get_linear_velocity_from_estimation(est_idx) - gt_velocity
            variance = np.sqrt(self.get_linear_velocity_covariance_from_estimation(est_idx))
            if self.haslabels:
                ax[0].plot(time, error[:,0],"--",linewidth=1.5,label = self.labels[est_idx])
                ax[1].plot(time, error[:,1],"--",linewidth=1.5)
                ax[2].plot(time, error[:,2],"--",linewidth=1.5)
            else:
                ax[0].plot(time, error[:,0],"--",linewidth=1.5)
                ax[1].plot(time, error[:,1],"--",linewidth=1.5)
                ax[2].plot(time, error[:,2],"--",linewidth=1.5)
            vel_var_mean += (variance/self.__count) 
            
        if self.plotVar and self.__hasCov:             
            ax[0].plot(time, vel_var_mean[:,0],"--",linewidth=1.5,color="goldenrod",label = "± sigma")
            ax[0].plot(time, -vel_var_mean[:,0],"--",linewidth=1.5,color="goldenrod")

            ax[1].plot(time, vel_var_mean[:,1],"--",linewidth=1.5,color="goldenrod")
            ax[1].plot(time, -vel_var_mean[:,1],"--",linewidth=1.5,color="goldenrod")

            ax[2].plot(time, vel_var_mean[:,2],"--",linewidth=1.5,color="goldenrod")
            ax[2].plot(time, -vel_var_mean[:,2],"--",linewidth=1.5,color="goldenrod")
                
        ax[0].legend()
        ax[2].set_xlabel("time (sec)")
        ax[0].set_ylabel("x")
        ax[1].set_ylabel("y")
        ax[2].set_ylabel("z")
        ax[0].set_title("Velocity Error versus Time (meter/sec)")

        ax[0].grid()
        ax[1].grid()
        ax[2].grid()
        
        # ax[0].set_ylim(-2,2)
        # ax[1].set_ylim(-1,1)
        # ax[2].set_ylim(-2,2)

        if self.save:
            fig.savefig(file_name)

        fig.canvas.draw()
        image_from_plot1 = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot1 = image_from_plot1.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        image_from_plot1 = cv2.cvtColor(image_from_plot1, cv2.COLOR_RGB2BGR)
        
        if self.show:
            cv2.imshow("x y z", image_from_plot1)
            cv2.waitKey()  
            
    def create_attitude_error(self, file_name="attitude_errors.png"):
        if not self.__association_performed:
            self.performAssociation()
        
        if not self.__estimations_are_cropped:
            self.crop_estimations()
            
        time = (self.__cropped_time - self.__cropped_time[0]) * 1e-9        
        
        ori_gt_quat_xyzw = self.__cropped_gt[:,3:7]
        ori_gt_rotation_matrix = orientation.from_quat(ori_gt_quat_xyzw).as_matrix()
        
        euler_errors = np.zeros((time.shape[0],3,self.__count))
        """
        First index stands for the time
        Second index stands for the estimation axis, i.e. roll, pitch, yaw
        Third axis stands for the estimation id
        """
        
        euler_variances = np.zeros((time.shape[0],3,self.__count))
        """
        First index stands for the time
        Second index stands for the estimation axis, i.e. roll, pitch, yaw
        Third axis stands for the estimation id
        """
        

        for est_idx in range(self.__count):
            ori_es_quat_xyzw = self.get_orientation_from_estimation(est_idx)   
            ori_es_rotation_matrix = orientation.from_quat(ori_es_quat_xyzw).as_matrix()    
            
            if self.plotVar and self.__hasCov:
                ori_covariance = self.get_orientation_covariance_from_estimation(est_idx)
                for time_idx in range(time.shape[0]):
                    rot_es = ori_es_rotation_matrix[time_idx,:,:]
                    rot_gt = ori_gt_rotation_matrix[time_idx,:,:]
                    rot_err = rot_gt @ rot_es.transpose()
                    euler_errors[time_idx,:,est_idx] = orientation.from_matrix(rot_err).as_euler("xyz", degrees=True)
                    
                    ori_covariance_matrix = np.array([ori_covariance[time_idx,0],ori_covariance[time_idx,1],ori_covariance[time_idx,2],  ori_covariance[time_idx,1],ori_covariance[time_idx,3],ori_covariance[time_idx,4],   ori_covariance[time_idx,2],ori_covariance[time_idx,4],ori_covariance[time_idx,5]],dtype=np.float64).reshape(3,3)
                    ori_variance_matrix_diagonal = np.sqrt( np.array([ori_covariance_matrix[0,0],ori_covariance_matrix[1,1],ori_covariance_matrix[2,2]]) )
                    euler_variances[time_idx,:,est_idx] = np.array([ori_variance_matrix_diagonal[0],ori_variance_matrix_diagonal[1],ori_variance_matrix_diagonal[2]]) * (180/np.pi)
            else:    
                for time_idx in range(time.shape[0]):
                    rot_es = ori_es_rotation_matrix[time_idx,:,:]
                    rot_gt = ori_gt_rotation_matrix[time_idx,:,:]
                    rot_err = rot_gt @ rot_es.transpose()
                    euler_errors[time_idx,:,est_idx] = orientation.from_matrix(rot_err).as_euler("xyz", degrees=True)
                    
        fig, ax = plt.subplots(3,figsize=(6,6))
        if self.plotVar and self.__hasCov:
            ori_var_mean = np.zeros((time.shape[0],3))
            for est_idx in range(self.__count):
                error = euler_errors[:,:,est_idx]
                variance = euler_variances[:,:,est_idx]
                ax[0].plot(time, error[:,0],"--",linewidth=1.5)
                ax[1].plot(time, error[:,1],"--",linewidth=1.5)
                ax[2].plot(time, error[:,2],"--",linewidth=1.5)
                ori_var_mean += (variance/self.__count)
                
            ax[0].plot(time, ori_var_mean[:,0],"--",linewidth=1.5,color="goldenrod",label = "± sigma")
            ax[0].plot(time, -ori_var_mean[:,0],"--",linewidth=1.5,color="goldenrod")

            ax[1].plot(time, ori_var_mean[:,1],"--",linewidth=1.5,color="goldenrod")
            ax[1].plot(time, -ori_var_mean[:,1],"--",linewidth=1.5,color="goldenrod")

            ax[2].plot(time, ori_var_mean[:,2],"--",linewidth=1.5,color="goldenrod")
            ax[2].plot(time, -ori_var_mean[:,2],"--",linewidth=1.5,color="goldenrod")
        else:
            for est_idx in range(self.__count):
                error = euler_errors[:,:,est_idx]
                ax[0].plot(time, error[:,0],"--",linewidth=1.5)
                ax[1].plot(time, error[:,1],"--",linewidth=1.5)
                ax[2].plot(time, error[:,2],"--",linewidth=1.5)
                
        ax[0].legend()
        ax[2].set_xlabel("time (sec)")
        ax[0].set_ylabel("x")
        ax[1].set_ylabel("y")
        ax[2].set_ylabel("z")
        ax[0].set_title("Attitude Error - Euler (xyz) - Degree")

        ax[0].grid()
        ax[1].grid()
        ax[2].grid()

        if self.save:
            fig.savefig(file_name)

        fig.canvas.draw()
        image_from_plot1 = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot1 = image_from_plot1.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        image_from_plot1 = cv2.cvtColor(image_from_plot1, cv2.COLOR_RGB2BGR)
        
        if self.show:
            cv2.imshow("x y z", image_from_plot1)
            cv2.waitKey() 

    def ECEF_to_NED(self,C_b_e,r_eb_e, v_eb_e):
        # Parameters
        R_0 = 6378137  # WGS84 Equatorial radius in meters
        e = 0.0818191908425  # WGS84 eccentricity

        # Convert position using Borkowski closed-form exact solution
        # From (2.113)
        lambda_b = np.arctan2(r_eb_e[1], r_eb_e[0])

        # From (C.29) and (C.30)
        k1 = np.sqrt(1 - e**2) * np.abs(r_eb_e[2])
        k2 = e**2 * R_0
        beta = np.sqrt(r_eb_e[0]**2 + r_eb_e[1]**2)
        E = (k1 - k2) / beta
        F = (k1 + k2) / beta

        # From (C.31)
        P = 4/3 * (E*F + 1)

        # From (C.32)
        Q = 2 * (E**2 - F**2)

        # From (C.33)
        D = P**3 + Q**2

        # From (C.34)
        V = (np.sqrt(D) - Q)**(1/3) - (np.sqrt(D) + Q)**(1/3)

        # From (C.35)
        G = 0.5 * (np.sqrt(E**2 + V) + E)

        # From (C.36)
        T = np.sqrt(G**2 + (F - V * G) / (2 * G - E)) - G

        # From (C.37)
        L_b = np.sign(r_eb_e[2]) * np.arctan((1 - T**2) / (2 * T * np.sqrt(1 - e**2)))

        # From (C.38)
        h_b = (beta - R_0 * T) * np.cos(L_b) + (r_eb_e[2] - np.sign(r_eb_e[2]) * R_0 * np.sqrt(1 - e**2)) * np.sin(L_b)

        # Calculate ECEF to NED coordinate transformation matrix using (2.150)
        cos_lat = np.cos(L_b)
        sin_lat = np.sin(L_b)
        cos_long = np.cos(lambda_b)
        sin_long = np.sin(lambda_b)
        C_e_n = np.array([
            [-sin_lat * cos_long, -sin_lat * sin_long,  cos_lat],
            [-sin_long,            cos_long,           0],
            [-cos_lat * cos_long, -cos_lat * sin_long, -sin_lat]
        ])

        # Transform velocity using (2.73)
        v_eb_n = np.dot(C_e_n, v_eb_e)

        # Transform attitude using (2.15)
        C_b_n = np.dot(C_e_n, C_b_e)

        lat_lon_alt = np.array([L_b,lambda_b, h_b])

        return C_b_n,lat_lon_alt, v_eb_n
    
    def Calculate_errors_NED(self,est_C_b_n,est_lat_lon_alt, est_v_eb_n,
                            true_C_b_n, true_lat_lon_alt, true_v_eb_n):
        
        # print(f"est_lat_lon_alt: \n {est_lat_lon_alt}")
        # print(f"true_lat_lon_alt: \n {true_lat_lon_alt}")
        est_L_b = est_lat_lon_alt[0]
        est_lambda_b = est_lat_lon_alt[1]
        est_h_b = est_lat_lon_alt[2]

        true_L_b = true_lat_lon_alt[0]
        true_lambda_b = true_lat_lon_alt[1]
        true_h_b = true_lat_lon_alt[2]

        # print(f"est_L_b: \n {est_L_b}")
        # print(f"true_L_b: \n {true_L_b}")

        # Position error calculation, using (2.119)
        R_N, R_E = self.Radii_of_curvature(true_L_b)
        delta_r_eb_n = np.zeros(3)
        delta_r_eb_n[0] = (est_L_b - true_L_b) * (R_N + true_h_b)
        delta_r_eb_n[1] = (est_lambda_b - true_lambda_b) * (R_E + true_h_b) * np.cos(true_L_b)
        delta_r_eb_n[2] = -(est_h_b - true_h_b)

        # Velocity error calculation
        delta_v_eb_n = est_v_eb_n - true_v_eb_n

        # Attitude error calculation, using (5.109) and (5.111)
        delta_C_b_n = est_C_b_n @ true_C_b_n.T
        delta_eul_nb_n = orientation.from_matrix(delta_C_b_n).as_euler("xyz", degrees=True)

        return delta_eul_nb_n,delta_r_eb_n, delta_v_eb_n
    
    def Calculate_errors_ECEF(self,est_C_b_e,  est_p_eb_e,  est_v_eb_e,
                                   true_C_b_e, true_p_eb_e, true_v_eb_e):

        # Position error calculation
        delta_r_eb_e = est_p_eb_e - true_p_eb_e

        # Velocity error calculation
        delta_v_eb_e = est_v_eb_e - true_v_eb_e

        # Attitude error calculation, using (5.109) and (5.111)
        delta_C_b_e = est_C_b_e @ true_C_b_e.T
        delta_eul_nb_e = orientation.from_matrix(delta_C_b_e).as_euler("xyz", degrees=True)

        return delta_eul_nb_e,delta_r_eb_e, delta_v_eb_e
    
    def Radii_of_curvature(self,L):
        # Parameters
        R_0 = 6378137  # WGS84 Equatorial radius in meters
        e = 0.0818191908425  # WGS84 eccentricity

        # Calculate meridian radius of curvature using (2.105)
        temp = 1 - (e * np.sin(L))**2
        R_N = R_0 * (1 - e**2) / temp**1.5

        # Calculate transverse radius of curvature using (2.105)
        R_E = R_0 / np.sqrt(temp)

        return R_N, R_E
    
    def plot_errors_in_NED(self,file_name):
        if not self.__association_performed:
            self.performAssociation()
        
        if not self.__estimations_are_cropped:
            self.crop_estimations()
            
        time = (self.__cropped_time - self.__cropped_time[0]) * 1e-9        
        
        
        ori_gt_quat_xyzw = self.__cropped_gt[:,3:7]
        ori_gt_rotation_matrix = orientation.from_quat(ori_gt_quat_xyzw).as_matrix()

        gt_position= self.__cropped_gt[:,0:3]
        gt_velocity = self.__cropped_gt[:,7:10]

        """
        In the error matrices, first index stands for the time
        Second index stands for the estimation axis, i.e. roll, pitch, yaw
        Third axis stands for the estimation id
        """        
        euler_errors = np.zeros((time.shape[0],3,self.__count))
        position_errors = np.zeros((time.shape[0],3,self.__count))
        velocity_errors = np.zeros((time.shape[0],3,self.__count))


        for est_idx in range(self.__count):
            ori_es_quat_xyzw = self.get_orientation_from_estimation(est_idx)   
            ori_es_rotation_matrix = orientation.from_quat(ori_es_quat_xyzw).as_matrix()  
            est_p_eb_e_all = self.get_position_from_estimation(est_idx)
            est_v_eb_e_all = self.get_linear_velocity_from_estimation(est_idx)  

            for time_idx in range(time.shape[0]):
                est_C_b_e = ori_es_rotation_matrix[time_idx,:,:].reshape(3,3)
                est_p_eb_e = est_p_eb_e_all[time_idx,:].reshape(-1)
                est_v_eb_e = est_v_eb_e_all[time_idx,:].reshape(-1)
                est_C_b_n,est_lat_lon_alt, est_v_eb_n = self.ECEF_to_NED(est_C_b_e,est_p_eb_e,est_v_eb_e)
                # print(f"est_p_eb_e:\n{est_p_eb_e}")
                # print(f"est_lat_lon_alt:\n{est_lat_lon_alt}")

                gt_C_b_e = ori_gt_rotation_matrix[time_idx,:,:].reshape(3,3)
                gt_p_eb_e = gt_position[time_idx,:].reshape(-1)
                gt_v_eb_e = gt_velocity[time_idx,:].reshape(-1)
                gt_C_b_n, gt_lat_lon_alt, gt_v_eb_n = self.ECEF_to_NED(gt_C_b_e, gt_p_eb_e, gt_v_eb_e)
                # print(f"gt_p_eb_e:\n{gt_p_eb_e}")
                # print(f"gt_lat_lon_alt:\n{gt_lat_lon_alt}")

                delta_eul_nb_n,delta_p_eb_n, delta_v_eb_n = self.Calculate_errors_NED(est_C_b_n,est_lat_lon_alt,est_v_eb_n,gt_C_b_n,gt_lat_lon_alt,gt_v_eb_n)
                euler_errors[time_idx,:,est_idx] = delta_eul_nb_n.reshape(1,3)
                position_errors[time_idx,:,est_idx] = delta_p_eb_n.reshape(1,3)
                velocity_errors[time_idx,:,est_idx] = delta_v_eb_n.reshape(1,3)

        font_size = 15
        fig, ax = plt.subplots(3,3,figsize=(9,9))
        fig.suptitle(r'Errors in NED Frame',fontsize = font_size)
        fig.supxlabel(r'time $(sec)$',fontsize = font_size)
        #fig.tight_layout()
        fig.subplots_adjust(top=0.92)
        
        for est_idx in range(self.__count):
            ori_error = euler_errors[:,:,est_idx]
            ax[0,0].plot(time, ori_error[:,0],"--",linewidth=1.5)
            ax[0,0].set_ylabel(r'Ori Err $(deg)$',fontsize = font_size)
            ax[0,0].set_title(r'North',fontsize = font_size)
            ax[0,1].plot(time, ori_error[:,1],"--",linewidth=1.5)
            ax[0,1].set_title(r'Easth',fontsize = font_size)
            ax[0,2].plot(time, ori_error[:,2],"--",linewidth=1.5)
            ax[0,2].set_title(r'Down',fontsize = font_size)

            pos_error = position_errors[:,:,est_idx]
            ax[1,0].plot(time, pos_error[:,0],"--",linewidth=1.5)
            ax[1,0].set_ylabel(r'Pos Err $(met)$',fontsize = font_size)
            ax[1,1].plot(time, pos_error[:,1],"--",linewidth=1.5)
            ax[1,2].plot(time, pos_error[:,2],"--",linewidth=1.5)

            vel_error = velocity_errors[:,:,est_idx]
            ax[2,0].plot(time, vel_error[:,0],"--",linewidth=1.5)
            ax[2,0].set_ylabel(r'Vel Err $(m/s)$',fontsize = font_size)
            ax[2,1].plot(time, vel_error[:,1],"--",linewidth=1.5)
            ax[2,2].plot(time, vel_error[:,2],"--",linewidth=1.5)
                
        # ax[0,0].legend()
        # ax[0,2].set_xlabel("time (sec)")
        # ax[0,0].set_ylabel("x")
        # ax[0,1].set_ylabel("y")
        # ax[0,2].set_ylabel("z")
        # ax[0,0].set_title("Attitude Error - Euler (xyz) - Degree")

        for h_i in range(3):
            for v_i in range(3):
                ax[v_i,h_i].grid()
        
        if self.save:
            fig.savefig(file_name)

        fig.canvas.draw()
        image_from_plot1 = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot1 = image_from_plot1.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        image_from_plot1 = cv2.cvtColor(image_from_plot1, cv2.COLOR_RGB2BGR)
        
        if self.show:
            cv2.imshow("x y z", image_from_plot1)
            cv2.waitKey() 



    def plot_errors_in_ECEF(self,file_name):
        if not self.__association_performed:
            self.performAssociation()
        
        if not self.__estimations_are_cropped:
            self.crop_estimations()
            
        time = (self.__cropped_time - self.__cropped_time[0]) * 1e-9        
        
        
        ori_gt_quat_xyzw = self.__cropped_gt[:,3:7]
        ori_gt_rotation_matrix = orientation.from_quat(ori_gt_quat_xyzw).as_matrix()

        gt_position= self.__cropped_gt[:,0:3]
        gt_velocity = self.__cropped_gt[:,7:10]

        """
        In the error matrices, first index stands for the time
        Second index stands for the estimation axis, i.e. roll, pitch, yaw
        Third axis stands for the estimation id
        """        
        euler_errors = np.zeros((time.shape[0],3,self.__count))
        position_errors = np.zeros((time.shape[0],3,self.__count))
        velocity_errors = np.zeros((time.shape[0],3,self.__count))


        for est_idx in range(self.__count):
            ori_es_quat_xyzw = self.get_orientation_from_estimation(est_idx)   
            ori_es_rotation_matrix = orientation.from_quat(ori_es_quat_xyzw).as_matrix()  
            est_p_eb_e_all = self.get_position_from_estimation(est_idx)
            est_v_eb_e_all = self.get_linear_velocity_from_estimation(est_idx)  

            for time_idx in range(time.shape[0]):
                est_C_b_e = ori_es_rotation_matrix[time_idx,:,:].reshape(3,3)
                est_p_eb_e = est_p_eb_e_all[time_idx,:].reshape(-1)
                est_v_eb_e = est_v_eb_e_all[time_idx,:].reshape(-1)

                gt_C_b_e = ori_gt_rotation_matrix[time_idx,:,:].reshape(3,3)
                gt_p_eb_e = gt_position[time_idx,:].reshape(-1)
                gt_v_eb_e = gt_velocity[time_idx,:].reshape(-1)

                delta_eul_nb_n,delta_p_eb_n, delta_v_eb_n = self.Calculate_errors_ECEF(est_C_b_e,est_p_eb_e,est_v_eb_e,gt_C_b_e,gt_p_eb_e,gt_v_eb_e)
                euler_errors[time_idx,:,est_idx] = delta_eul_nb_n.reshape(1,3)
                position_errors[time_idx,:,est_idx] = delta_p_eb_n.reshape(1,3)
                velocity_errors[time_idx,:,est_idx] = delta_v_eb_n.reshape(1,3)

        font_size = 15     
        fig, ax = plt.subplots(3,3,figsize=(9,9))
        fig.suptitle(r'Errors in ECEF Frame',fontsize = font_size)
        fig.supxlabel(r'time $(sec)$',fontsize = font_size)
        #fig.tight_layout()
        fig.subplots_adjust(top=0.92)
        
        for est_idx in range(self.__count):
            ori_error = euler_errors[:,:,est_idx]
            ax[0,0].plot(time, ori_error[:,0],"--",linewidth=1.5)
            ax[0,0].set_ylabel(r'Ori Err $(deg)$',fontsize = font_size)
            ax[0,1].plot(time, ori_error[:,1],"--",linewidth=1.5)
            ax[0,2].plot(time, ori_error[:,2],"--",linewidth=1.5)

            pos_error = position_errors[:,:,est_idx]
            ax[1,0].plot(time, pos_error[:,0],"--",linewidth=1.5)
            ax[1,0].set_ylabel(r'Pos Err $(met)$',fontsize = font_size)
            ax[1,1].plot(time, pos_error[:,1],"--",linewidth=1.5)
            ax[1,2].plot(time, pos_error[:,2],"--",linewidth=1.5)

            vel_error = velocity_errors[:,:,est_idx]
            ax[2,0].plot(time, vel_error[:,0],"--",linewidth=1.5)
            ax[2,0].set_ylabel(r'Vel Err $(m/s)$',fontsize = font_size)
            ax[2,1].plot(time, vel_error[:,1],"--",linewidth=1.5)
            ax[2,2].plot(time, vel_error[:,2],"--",linewidth=1.5)
                
        # ax[0,0].legend()
        # ax[0,2].set_xlabel("time (sec)")
        # ax[0,0].set_ylabel("x")
        # ax[0,1].set_ylabel("y")
        # ax[0,2].set_ylabel("z")
        # ax[0,0].set_title("Attitude Error - Euler (xyz) - Degree")

        for h_i in range(3):
            for v_i in range(3):
                ax[v_i,h_i].grid()
        
        if self.save:
            fig.savefig(file_name)

        fig.canvas.draw()
        image_from_plot1 = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot1 = image_from_plot1.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        image_from_plot1 = cv2.cvtColor(image_from_plot1, cv2.COLOR_RGB2BGR)
        
        if self.show:
            cv2.imshow("x y z", image_from_plot1)
            cv2.waitKey() 
        
            
if __name__ == "__main__":
    dataset = "Selcuk"
    gt_dir = "/home/hakito/Python_Scripts/SelcukVerisi/Selcuk/gt.csv"
    estimations = os.path.join("ExampleDataset/Runs",dataset)

    eval = _eval(gt_dir)
    eval.show = False
    eval.save = True
    eval.plotVar = False
    eval.haslabels = True

    estimation_path = os.path.join("/home/hakito/workspace/evaluation/Selcuk/long_ins","imu_state.txt")
    eval.addEstimation(estimation_path)
    eval.labels.update({0:"ins"})

    estimation_path = os.path.join("/home/hakito/workspace/evaluation/Selcuk/long_vins","imu_state.txt")
    eval.addEstimation(estimation_path)
    eval.labels.update({1:"vins"})

    # estimation_path = os.path.join("/home/hakito/workspace/evaluation/Selcuk/long_vins_w_t","imu_state.txt")
    # eval.addEstimation(estimation_path)
    # eval.labels.update({2:"vins-optimize-t"})

    # estimation_path = os.path.join("/home/hakito/workspace/evaluation/Selcuk/long_vins_w_t","imu_state.txt")
    # eval.addEstimation(estimation_path)
    # eval.labels.update({3:"vins_s100"})


    # estimation_path = os.path.join("/home/hakito/workspace/evaluation/Selcuk/ins2","imu_state.txt")
    # eval.addEstimation(estimation_path)
    # eval.labels.update({0:"ins"})
    # estimation_path = os.path.join("/home/hakito/workspace/evaluation/Selcuk/vins2","imu_state.txt")
    # eval.addEstimation(estimation_path)
    # eval.labels.update({1:"Paul"})
    # estimation_path = os.path.join("/home/hakito/workspace/evaluation/Selcuk/vins3","imu_state.txt")
    # eval.addEstimation(estimation_path)
    # eval.labels.update({2:"i-Frame"})
    # estimation_path = os.path.join("/home/hakito/workspace/evaluation/Selcuk/vins5","imu_state.txt")
    # eval.addEstimation(estimation_path)
    # eval.labels.update({3:"e-Frame"})
    
    # estimation_path = os.path.join("/home/hakito/workspace/evaluation/Selcuk/sigma_100","imu_state.txt")
    # eval.addEstimation(estimation_path)
    # eval.labels.update({1:"100"})
    # estimation_path = os.path.join("/home/hakito/workspace/evaluation/Selcuk/vins6","imu_state.txt")
    # eval.addEstimation(estimation_path)
    # eval.labels.update({2:"256"})
    # estimation_path = os.path.join("/home/hakito/workspace/evaluation/Selcuk/sigma_400","imu_state.txt")
    # eval.addEstimation(estimation_path)
    # eval.labels.update({3:"400"})
    # estimation_path = os.path.join("/home/hakito/workspace/evaluation/Selcuk/sigma_900","imu_state.txt")
    # eval.addEstimation(estimation_path)
    # eval.labels.update({1:"not stabized"})
    # estimation_path = os.path.join("/home/hakito/workspace/evaluation/Selcuk/sigma_1600","imu_state.txt")
    # eval.addEstimation(estimation_path)
    # eval.labels.update({5:"1600"})

    # estimation_path = os.path.join("/home/hakito/workspace/evaluation/Selcuk/stabilize","imu_state.txt")
    # eval.addEstimation(estimation_path)
    # eval.labels.update({2:"stabized 0.1"})

    # estimation_path = os.path.join("/home/hakito/workspace/evaluation/Selcuk/stabilize_0.99","imu_state.txt")
    # eval.addEstimation(estimation_path)
    # eval.labels.update({3:"stabized 0.99"})

    saving_path = "deneme/"
    eval.getAlignedTraj(saving_path+dataset+"/")
    
    # eval.create_xyz_plots(saving_path+dataset + "_xyz.png")
    # eval.create_error_plots(saving_path+dataset + "_error.png")
    # eval.create_traj_plots(saving_path+dataset + "_traj.png")
    # eval.create_velocity_error(saving_path+dataset + "_vel.png")
    # eval.create_attitude_error(saving_path+dataset + "_ori.png")

    eval.plot_errors_in_NED(saving_path+dataset + "_ned.png")
    eval.plot_errors_in_ECEF(saving_path+dataset + "_ecef.png")
