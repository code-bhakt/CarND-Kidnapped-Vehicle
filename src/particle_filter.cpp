  /**
   * particle_filter.cpp
   *
   * Created on: Dec 12, 2016
   * Author: Tiffany Huang
   */

  #include "particle_filter.h"

  #include <math.h>
  #include <algorithm>
  #include <iostream>
  #include <iterator>
  #include <numeric>
  #include <random>
  #include <string>
  #include <vector>
  #include <cmath>

  #include "helper_functions.h"

  using std::string;
  using std::vector;
  using std::normal_distribution;
  using std::default_random_engine;

  void ParticleFilter::init(double x, double y, double theta, double std[]) {
    /**
     * TODO: Set the number of particles. Initialize all particles to 
     *   first position (based on estimates of x, y, theta and their uncertainties
     *   from GPS) and all weights to 1. 
     * TODO: Add random Gaussian noise to each particle.
     * NOTE: Consult particle_filter.h for more information about this method 
     *   (and others in this file).
     */
    num_particles = 1000;  // TODO: Set the number of particles
    normal_distribution<double> norm_x(x, std[0]);
    normal_distribution<double> norm_y(y, std[1]);
    normal_distribution<double> norm_theta(theta, std[2]);
    Particle part;
    default_random_engine gen;
    for (int i=0; i< num_particles; i++){
      part.id = i;
      part.x = norm_x(gen);
      part.y = norm_y(gen);
      part.theta = norm_theta(gen);
      part.weight = 1.0;
      particles.push_back(part);
      weights.push_back(part.weight);
    }
    is_initialized = true;
  }

  void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                  double velocity, double yaw_rate) {
    /**
     * TODO: Add measurements to each particle and add random Gaussian noise.
     * NOTE: When adding noise you may find std::normal_distribution 
     *   and std::default_random_engine useful.
     *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
     *  http://www.cplusplus.com/reference/random/default_random_engine/
     */
    normal_distribution<double> norm_x(0.0, std_pos[0]);
    normal_distribution<double> norm_y(0.0, std_pos[1]);
    normal_distribution<double> norm_theta(0.0, std_pos[2]);
    default_random_engine gen;
    
    for(int i=0; i<num_particles; i++){
      if(fabs(yaw_rate) > 0.0001){
        double final_yaw = particles[i].theta + yaw_rate * delta_t;
        particles[i].x += velocity/yaw_rate * (sin(final_yaw) - sin(particles[i].theta)) + norm_x(gen);
        particles[i].y += velocity/yaw_rate * (cos(particles[i].theta) - cos(final_yaw)) + norm_y(gen);
        particles[i].theta = final_yaw + norm_theta(gen);
      }
      else{
        particles[i].x += velocity*delta_t*(cos(particles[i].theta)) + norm_x(gen);
        particles[i].y += velocity*delta_t*(sin(particles[i].theta)) + norm_y(gen);  
      }
    }
  }

  void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                       vector<LandmarkObs>& observations) {
    /**
     * TODO: Find the predicted measurement that is closest to each 
     *   observed measurement and assign the observed measurement to this 
     *   particular landmark.
     * NOTE: this method will NOT be called by the grading code. But you will 
     *   probably find it useful to implement this method and use it as a helper 
     *   during the updateWeights phase.
     */
    for(unsigned int i=0; i<observations.size(); i++){
      double min_dist, distance;
      min_dist = dist(observations[i].x, observations[i].y, predicted[0].x, predicted[0].y);
      for(unsigned int j=1; j<predicted.size(); j++){
        double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
        if(distance < min_dist){
          min_dist=distance;
          observations[i].id = j;
        }
      }
    }
  }

  double multiv_prob(double gauss_norm, double sig_x, double sig_y, double x_obs, double y_obs, double mu_x, double mu_y){
        
    // calculate exponent
    double exponent;
    exponent = (pow(x_obs -mu_x, 2) / (2 * pow(sig_x, 2))) + (pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2)));

    // calculate weight using normalization term and exponent
    double weight;
    weight = gauss_norm * exp(-exponent);

    return weight;
  }


  void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                     const vector<LandmarkObs> &observations, 
                                     const Map &map_landmarks) {
    /**
     * TODO: Update the weights of each particle using a mult-variate Gaussian 
     *   distribution. You can read more about this distribution here: 
     *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
     * NOTE: The observations are given in the VEHICLE'S coordinate system. 
     *   Your particles are located according to the MAP'S coordinate system. 
     *   You will need to transform between the two systems. Keep in mind that
     *   this transformation requires both rotation AND translation (but no scaling).
     *   The following is a good resource for the theory:
     *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
     *   and the following is a good resource for the actual equation to implement
     *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
     */

    double sig_x = std_landmark[0];
    double sig_y = std_landmark[1];
    // calculate normalization term
    double gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);

    double sum_weights = 0.0;
    for(int i = 0; i<num_particles; i++){
      std::vector<LandmarkObs> trans_obs;
      for(unsigned int j=0; j<observations.size(); j++){
        LandmarkObs trans;
        trans.x = particles[i].x + (cos(particles[i].theta) * observations[j].x) - (sin(particles[i].theta) * observations[j].y);
        trans.y = particles[i].y + (sin(particles[i].theta) * observations[j].x) + (cos(particles[i].theta) * observations[j].y);
        trans.id = 0;
        trans_obs.push_back(trans);
      }
      std::vector<LandmarkObs> predicted;
      LandmarkObs pred;
      for(unsigned int l = 0; l<map_landmarks.landmark_list.size(); l++){
        double distance = dist(particles[i].x, particles[i].y, map_landmarks.landmark_list[l].x_f, map_landmarks.landmark_list[l].y_f);
        if(fabs(distance) < sensor_range){
          pred.x = map_landmarks.landmark_list[l].x_f;
          pred.y = map_landmarks.landmark_list[l].y_f;
          pred.id = map_landmarks.landmark_list[l].id_i;
          predicted.push_back(pred);
        }
      }
      dataAssociation(predicted, trans_obs);

      double multiv_gaussian_prob;
      double particle_weight = 1;

      for(unsigned int k=0; k<trans_obs.size(); k++){
        int p_id = trans_obs[k].id;
        double x_obs = trans_obs[k].x;
        double y_obs = trans_obs[k].y;
        double mu_x = predicted[p_id].x;
        double mu_y = predicted[p_id].y;
        multiv_gaussian_prob = multiv_prob(gauss_norm, sig_x, sig_y, x_obs, y_obs, mu_x, mu_y);
        particle_weight *= multiv_gaussian_prob;
      }
      particles[i].weight = particle_weight;
      sum_weights += particle_weight;
    }
    for(int i = 0; i<num_particles; i++){
      particles[i].weight /= sum_weights;
      weights[i] = particles[i].weight;
    }
  }

  void ParticleFilter::resample() {
    /*
     * TODO: Resample particles with replacement with probability proportional 
     *   to their weight. 
     * NOTE: You may find std::discrete_distribution helpful here.
     *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
     */
    default_random_engine gen;
    std::uniform_int_distribution<int> rand_int(0, num_particles-1);
    std::uniform_real_distribution<double> random_double(0.0, 1.0);
    std::vector<Particle> resampled_particles;
    double beta = 0.0;
    double max_weight = *std::max_element(weights.begin(), weights.end());
    int index = rand_int(gen);

    for(int i = 0; i<num_particles; i++){
      beta += random_double(gen) * 2.0 *max_weight;
      while(beta > weights[index]){
        beta -= weights[index];
        index = (index + 1) % num_particles;
      } 
      resampled_particles.push_back(particles[index]);
    }
    particles = resampled_particles;
  }

  void ParticleFilter::SetAssociations(Particle& particle, 
                                       const vector<int>& associations, 
                                       const vector<double>& sense_x, 
                                       const vector<double>& sense_y) {
    // particle: the particle to which assign each listed association, 
    //   and association's (x,y) world coordinates mapping
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
  }

  string ParticleFilter::getAssociations(Particle best) {
    vector<int> v = best.associations;
    std::stringstream ss;
    copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
  }

  string ParticleFilter::getSenseCoord(Particle best, string coord) {
    vector<double> v;

    if (coord == "X") {
      v = best.sense_x;
    } else {
      v = best.sense_y;
    }

    std::stringstream ss;
    copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
  }
