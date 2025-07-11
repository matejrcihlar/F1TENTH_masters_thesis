import numpy as np
import csv
import scipy.interpolate as sp_int
import matplotlib.pyplot as plt


class RewardNode():
    def __init__(self, reward_function='sum'):
        
        self.start_x = None
        self.start_y = None
        self.start_wp_idx = None
        self.prev_action = np.array([0.0, 0.0])
        self.reward_function = reward_function
        
        self.logger = {
            "speed": [],
            "raceline_error": [],
            "distance": [],
            "overtake": [],
            "smoothness_penalty": [],
            "heading_error" : [],
        }

        self.colors = {
            "speed": 'red',
            "raceline_error": 'blue',
            "distance": 'green',
            "overtake": 'orange',
            "smoothness_penalty": 'purple',
            "heading_error" : 'cyan',
        }

        self.K1 = 2.5  #speed
        self.K2 = 2.5  #raceline
        self.K3 = 4.5  #crash
        self.K4 = 0.0  #distance
        self.K5 = 0.9  #overtake
        self.K6 = 0.0  #laptime
        self.K7 = 2.5  #smoothness
        self.K8 = 2.0  #heading

        self.waypoints = []
        '''with open('/home/mrc/sim_ws/src/f1tenth_lab6_template/waypoints.csv', mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                self.waypoints.append([float(i) for i in row])

        self.original_waypoints = self.waypoints
        self.waypoints = np.transpose(self.waypoints)
        
        tck, _ = sp_int.splprep(self.waypoints, s=0)
        u_fine = np.linspace(0, 1, 100)
        interpolated = sp_int.splev(u_fine, tck)
        self.waypoints = list(zip(interpolated[0], interpolated[1]))  # List of (x, y) tuples'''

        waypoints = np.loadtxt('/home/mrc/sim_ws/src/ppo_racing/ppo_racing/Spielberg_waypoints.csv', delimiter=',')

        # Keep only the first two columns
        waypoints = waypoints[:, :2]
        waypoints = waypoints[::5,:]
        self.waypoints = [tuple(row) for row in waypoints]
        #print(self.waypoints)

    def get_reward(self, obs, action, ego=0):
        #ego = obs['ego_idx']
        opp = 1 - ego  # assuming 2 agents
        current_x = obs['poses_x'][ego] 
        current_y = obs['poses_y'][ego]
        current_yaw = obs['poses_theta'][ego]
        opp_x = obs['poses_x'][opp]
        opp_y = obs['poses_x'][opp]
        current_speed = obs['linear_vels_x'][ego] / 3.0
        #if current_speed * 3.0 < 0.5:
        #    current_speed = -2
        raceline_error = self.raceline_error(current_x, current_y) 
        laptime = self.laptime(obs['lap_times'][ego])
        crash = obs['collisions'][ego]
        #if np.linalg.norm(np.array([current_x, current_y]),np.array([opp_x, opp_y])) < 0.3:
        #    crash += 0.5

        #overtake = self.overtake(current_x, current_y, opp_x, opp_y) #-1/0/1
        overtake = self.overtake_2(current_x, current_y, opp_x, opp_y) # linear waypoint diff
        distance_traveled = self.distance_traveled(current_x, current_y)
        smoothness_penalty = self.smoothness_penalty(action)
        heading_error = self.heading_error(current_x, current_y, current_yaw)

        if self.reward_function == 'sum':
            reward = self.K1 * (current_speed-1.0) - self.K2 * raceline_error - self.K3 * crash + self.K4 * distance_traveled \
            - self.K7 * smoothness_penalty + self.K5 * (overtake-1)/2 - self.K8 * np.abs(heading_error/np.pi)#+ self.K6 * laptime
        elif self.reward_function == 'multiplication':
            reward = ((self.K1 * current_speed * (np.cos(heading_error)**self.K8)) / (self.K2 * raceline_error * (smoothness_penalty**self.K7)+1)) \
                + self.K3 * crash + self.K4 * distance_traveled + self.K5 * overtake
        elif self.reward_function == 'combined':
            reward = 1+self.K1 * current_speed * np.cos(np.clip(self.K8 * heading_error,-np.pi,np.pi)) - self.K2 * raceline_error * np.abs(overtake) + self.K5 * overtake - self.K3 * crash - self.K7 * smoothness_penalty 
        #print("Reward:", current_speed, raceline_error, reward)

        self.start_x = current_x
        self.start_y = current_y
        
        self.logger["speed"].append(current_speed)
        self.logger["raceline_error"].append(-raceline_error)
        self.logger["distance"].append(distance_traveled)
        self.logger["overtake"].append(overtake)
        self.logger["smoothness_penalty"].append(-smoothness_penalty)
        self.logger["heading_error"].append(-heading_error/np.pi)

        return reward

    def raceline_error(self, x, y):
        
    
        index = self.waypoint_index(x, y)
        wp1 = self.waypoints[(index+1+len(self.waypoints))%len(self.waypoints)]
        wp2 = self.waypoints[index]
        distance = self.point_to_line_distance(wp1, wp2, x, y)
        distance_clipped = np.clip(distance, 0.0, 2.0) / 2.0

        
                    
        return distance_clipped

    def point_to_line_distance(self, p1, p2, x, y):
          """
          Calculate the distance from point p3 to the line defined by points p1 and p2.
          
          Parameters:
               p1, p2, p3: Tuples or lists representing points in 2D (x, y)
          
          Returns:
               float: Distance from point p3 to the line through p1 and p2
          """
          # Convert points to numpy arrays
          p1 = np.array(p1)
          p2 = np.array(p2)
          p3 = np.array([x, y])
          
          # Compute the vector from p1 to p2 and from p1 to p3
          line_vec = p2 - p1
          point_vec = p3 - p1

          # Compute the area of the parallelogram (cross product magnitude in 2D)
          area = np.abs(np.cross(line_vec, point_vec))
          
          # Compute the length of the line
          line_length = np.linalg.norm(line_vec)
          
          # Distance is area divided by base length
          distance = area / line_length if line_length != 0 else np.linalg.norm(point_vec)
          
          return distance

    def laptime(self, laptime):
         laptime_reward = 20.0 / laptime
              
         return laptime_reward
    
    def overtake(self, ego_x, ego_y, opp_x, opp_y):
          ego_idx = self.waypoint_index(ego_x, ego_y)
          opp_idx = self.waypoint_index(opp_x, opp_y)
         
          if ego_idx == opp_idx:
               return 0  # same position

          # Forward distance from ego to opponent
          forward_dist = (opp_idx - ego_idx + len(self.waypoints)) % len(self.waypoints)

          # If forward distance is less than half the loop, opponent is ahead
          if forward_dist < len(self.waypoints) / 2:
               return -1  # opponent ahead
          else:
               return 1   # ego ahead

    def overtake_2(self, ego_x, ego_y, opp_x, opp_y, max_gap=3):
        ego_idx = self.waypoint_index(ego_x, ego_y)
        opp_idx = self.waypoint_index(opp_x, opp_y)

        forward_dist = (opp_idx - ego_idx)%len(self.waypoints)

        if forward_dist < len(self.waypoints)/2:
            dist = -forward_dist
        else:
            dist = (ego_idx - opp_idx)%len(self.waypoints)
        
        #print(dist, np.clip(dist, -max_gap, max_gap)/max_gap)

        return np.clip(dist, -max_gap, max_gap)/max_gap
        

    def waypoint_index(self, x, y):
         min_dist = float('inf')
         best_wp_index = 0

         for wp_i in range(len(self.waypoints)):
            dx = self.waypoints[wp_i][0] - x
            dy = self.waypoints[wp_i][1] - y
            distance = np.hypot(dx, dy)
            if distance < min_dist:
                min_dist = distance
                best_wp_index = wp_i

         return best_wp_index

    def remember_start(self, wp_i):
         self.start_wp_idx = wp_i
         self.prev_action = np.array([0.0, 0.0])
         self.logger = {
            "speed": [],
            "raceline_error": [],
            "distance": [],
            "overtake": [],
            "smoothness_penalty": [],
            "heading_error" : [],
         } 

         return
    
    def distance_traveled(self, x, y):
         
         curr_way_idx = self.waypoint_index( x, y)
         dist = (curr_way_idx - self.start_wp_idx + len(self.waypoints))%len(self.waypoints)
         if dist < len(self.waypoints) * (9/10):
               return dist / 100.0 # opponent ahead
         else:
               return 0   # ego ahead
         
    def smoothness_penalty(self, action):
         #delta_action = np.linalg.norm(action - self.prev_action)
         
         delta_action = np.abs(action[0]-self.prev_action[0])
         
         self.prev_action = action

         return delta_action / 0.68
         
    def heading_error(self, x, y, yaw, lookahead=1):
        curr_way_idx = self.waypoint_index( x, y)
        wp_current = self.waypoints[curr_way_idx]
        wp_ahead = self.waypoints[(curr_way_idx + lookahead) % len(self.waypoints)]
        
        # Raceline heading (from current wp to ahead wp)
        dx = wp_ahead[0] - wp_current[0]
        dy = wp_ahead[1] - wp_current[1]
        raceline_heading = np.arctan2(dy, dx)

        # Heading error: difference between car yaw and raceline heading
        heading_error = raceline_heading - yaw

        # Normalize to [-pi, pi]
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi

        return np.clip(heading_error, -np.pi, np.pi) 


    def plot_logs(self, figure_file, episode):
        x = [i+1 for i in range(len(self.logger["speed"]))]
        plt.figure()
        for key, values in self.logger.items():
            plt.plot(values, label=f'{key}', color=self.colors[key])
        plt.title(f'Reward of episode: {episode}')
        plt.xlabel('Timesteps')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid()
        plt.savefig(figure_file)
        plt.close()
            
class OpponentWaypointLogger:
    def __init__(self, min_dist=0.5, lap_threshold=1.0, min_lap_points=10):
        self.min_dist = min_dist          # Min dist to log a new point
        self.lap_threshold = lap_threshold  # Max dist to starting point to count as lap complete
        self.min_lap_points = min_lap_points  # Prevent false early completions

        self.opponent_waypoints = []
        self.replace_index = 0
        self.lap_complete = False
        self.start_point = None

    def reset(self):
        self.opponent_waypoints = []
        self.replace_index = 0
        self.lap_complete = False
        self.start_point = None

    def add_position(self, x, y):
        point = np.array([x, y])

        if not self.opponent_waypoints:
            self.opponent_waypoints.append(point)
            self.start_point = point
            return

        # Check distance from previous point
        if not self.lap_complete:
            last_point = self.opponent_waypoints[-1]
        else:
            last_index = (self.replace_index - 1) % len(self.opponent_waypoints)
            last_point = self.opponent_waypoints[last_index]

        dist = np.linalg.norm(point - last_point)

        if dist >= self.min_dist:
            if not self.lap_complete:
                self.opponent_waypoints.append(point)

                # ✅ Check for lap completion
                if (
                    len(self.opponent_waypoints) >= self.min_lap_points and
                    np.linalg.norm(point - self.start_point) <= self.lap_threshold
                ):
                    self.lap_complete = True
                    self.replace_index = 0
            else:
                # ✅ Circular replacement mode
                self.opponent_waypoints[self.replace_index] = point
                self.replace_index = (self.replace_index + 1) % len(self.opponent_waypoints)

    def get_waypoints(self):
        return np.array(self.opponent_waypoints)

    def lap_finished(self):
        return self.lap_complete