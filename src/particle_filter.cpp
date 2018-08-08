/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	/*
	 * 1. Set Number of particles
	 */
	num_particles = 50;

	/*
	 * 2. Create normal distribution for x, y and theta.
	 */
	default_random_engine gen;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	/*
	 * 3. Create the particles
	 */
	for (int i = 0; i < num_particles; ++i)
	{
		struct Particle p;
		
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 0.0;

		particles.push_back(p);
		weights.push_back(1.0);
	}

	/*
	 * 4. Mark the particle filter as initilaized
	 */
	is_initialized = 1;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;
	normal_distribution<double> dist_x(0.0, std_pos[0]);
	normal_distribution<double> dist_y(0.0, std_pos[1]);
	normal_distribution<double> dist_theta(0.0, std_pos[2]);
	struct Particle p;

	/*
	 * Equations for updating x, y and yaw angle
	 *
	 * Case 1: fabs(yaw_rate) < 0.0001
	 * x_f = x_0 + v*cos(theta_0)*delta_t
	 * y_f = y_0 + v*sin(theta_0)*delta_t
	 * theta_f = theta_0
	 *
	 * Case 2: fabs(yaw_rate) >= 0.0001
	 * x_f = x_0 + v/yaw_rate[ sin(theta_0 + yaw_rate*delta_t) - sin(theta_0) ]
	 * y_f = y_0 + v/yaw_rate[ cos(theta_0) - cos(theta_0 + yaw_rate*delta_t) ]
	 * theta_f = theta_0 + yaw_rate*delta_t;
	 */

	/*
	 * For each particle calculate predictiona and add Gaussian noise to the measurements
	 */
	double c1 = (fabs(yaw_rate) >= 0.0001) ? velocity / yaw_rate : 1.0;
	double c2 = yaw_rate * delta_t;

	for (int i = 0; i < num_particles; ++i)
	{
		/* 1. Calculate prediction */
		p = particles[i];
		if (fabs(yaw_rate) < 0.0001)
		{
			p.x = p.x + velocity * cos(p.theta) * delta_t;
			p.y = p.y + velocity * sin(p.theta) * delta_t;
			p.theta = p.theta;
		}
		else
		{
			p.x = p.x + c1 * (sin(p.theta + c2) - sin(p.theta));
			p.y = p.y + c1 * (cos(p.theta) - cos(p.theta + c2));
			p.theta = p.theta + c2;
		}

		/* 2. Add Gaussian noise */
		p.x += dist_x(gen);
		p.y += dist_y(gen);
		p.theta += dist_theta(gen);

		particles[i] = p;
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	/*
	 * The "observations" vector contains the observed measurements in the map coordinates system
	 */
	for (unsigned int i = 0; i < observations.size(); i++)
	{
		double d;
		double d_min = 3.4e+38;
		LandmarkObs o_mc;
		int cli = -1; /* closest landmark index */

		o_mc = observations[i];

		/*
		 * The "predicted" vector contains the "in-range" landmarks in the map coordinates system
		 */
		for (unsigned int j = 0; j < predicted.size(); j++)
		{
			LandmarkObs l;
			l = predicted[j];

			d = dist(o_mc.x, o_mc.y, l.x, l.y);
			if (d < d_min)
			{
				d_min = d;
				cli = j;
			}
		}

		observations[i].id = cli >= 0 ? predicted[cli].id : -1;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	struct Particle *p;
	struct LandmarkObs o;							/* an observation in car coordinates system */
	struct LandmarkObs o_mc;						/* an observation in map coordinates system */
	std::vector<LandmarkObs> in_range_landmarks;	/* landmarks withing the particle sensor range */
	std::vector<LandmarkObs> observations_mc;		/* observations transformed to map coordinates */
	unsigned j;
	/*
	 * Pre-calculate some constant values used in the Multivariate-Gaussian probability calculations below
	 */
	double gauss_norm = 1.0 / (2.0 * M_PI*std_landmark[0] * std_landmark[1]);
	double c = sensor_range / sqrt(2);
	double out_of_range_exponent = 0.5*(pow(c / std_landmark[0], 2) + pow(c / std_landmark[1], 2));
	double out_of_range_prob = gauss_norm * exp(-out_of_range_exponent);
	double sigma_x2 = pow(std_landmark[0], 2);
	double sigma_y2 = pow(std_landmark[1], 2);
	double weights_sum = 0.0;
	
	/* 
	 * Loop through all particles 
	 */
	for (int i = 0; i < num_particles; ++i)
	{
		p = &particles[i];
		p->weight = 1.0;
		p->associations.clear();
		p->sense_x.clear();
		p->sense_y.clear();

		/*
		 * 1. Find the landmarks within the sensor range
		 */
		in_range_landmarks.clear();
		for (j = 0; j < map_landmarks.landmark_list.size(); j++)
		{
			struct Map::single_landmark_s l;
			struct LandmarkObs rl;
			double d;

			l = map_landmarks.landmark_list[j];
			d = dist(p->x, p->y, l.x_f, l.y_f);
			if (d <= sensor_range)
			{
				rl.id = l.id_i;
				rl.x = l.x_f;
				rl.y = l.y_f;
				in_range_landmarks.push_back(rl);
			}
		}

		/*
		 * 2. Transform an observations from car coordinates (observations) to 
		 *    map coordinates (observations_mc) assuming the car is in the selected particke (p[i]) position:
		 *
		 *  (xm,ym) - map observation coordinates
		 *  (xc,yc) - car observation coordinates
		 *  (xp,yp) - car coordinates == particle coordinates
		 *
		 *  | xm |   | cos(theta)  -sin(theta)	xp |   | xc |
		 *  | ym | = | sin(theta)  cos(theta)   yp | x | yc |
		 *  | 1  |   | 0			0            1 |   |  1 |
		 *
		 *  xm = xp + ( cos(theta) * xc ) - ( sin(theta) * yc )
		 *  ym = yp + ( sin(theta) * xc ) + ( cos(theta) * yc )
		 */
		observations_mc.clear();
		for (j = 0; j < observations.size(); j++)
		{
			o = observations[j];

			o_mc.x = p->x + (cos(p->theta) * o.x) - (sin(p->theta) * o.y);
			o_mc.y = p->y + (sin(p->theta) * o.x) + (cos(p->theta) * o.y);

			observations_mc.push_back(o_mc);
		}

		/*
		 * 3. Associate the predicted measurement with the corresponding "in-range" landmark 
		 */
		dataAssociation(in_range_landmarks, observations_mc);

		/*
		* 4. Calculate Multivariate-Gaussian probability density and the particle weight 
		*/
		for (j = 0; j < observations_mc.size(); j++)
		{
			struct Map::single_landmark_s l;
			double exponent;
			double prob;
			double diff_x;
			double diff_y;

			/*
			* 4.1 Calculate probability for each observation
			*/
			o_mc = observations_mc[j];

			if (o_mc.id >= 0)
			{
				p->associations.push_back(o_mc.id);
				p->sense_x.push_back(o_mc.x);
				p->sense_y.push_back(o_mc.y);

				/* NOTE: The landmark id starts from 1 */
				l = map_landmarks.landmark_list[o_mc.id-1];
				diff_x = o_mc.x - l.x_f;
				diff_y = o_mc.y - l.y_f;
				exponent = (pow(diff_x,2) / (2.0*sigma_x2)) + (pow(diff_y,2) / (2.0*sigma_y2));
				prob = gauss_norm * exp(-exponent);
			}
			else
			{
				prob = out_of_range_prob;
			}

			/*
			 * 4.2 Calculate particle weight
			 */
			p->weight *= prob;
		}

		weights[i] = p->weight;
		weights_sum += p->weight;
	}

	/* 
	 * Step 5: Normalize the weights 
	 */
	for (unsigned int i = 0; i < particles.size(); i++) 
	{
		particles[i].weight /= weights_sum;
		weights[i] = particles[i].weight;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	
	/* Generate random index value*/
	default_random_engine gen;
	uniform_int_distribution<int> random_index(0, num_particles - 1);
	int index = random_index(gen);

	/*  From Python example: Lesson13, section 20 
	*   p3 = []
	*	index = int(random.random() * N)
	*	beta = 0.0
	*	mw = max(w)
	*	for i in range(N) :
	*		beta += random.random() * 2.0 * mw
	*		while beta > w[index]:
	*         beta -= w[index]
	*    	   index = (index + 1) % N
	* 	    p3.append(p[index])
	*	p = p3
	*/
	std::vector<Particle> p3;
	double beta = 0.0;
	double max_weight = *max_element(weights.begin(), weights.end());
	
	for (int i = 0; i < num_particles; i++)
	{
		/* Generate random weight value */
		uniform_real_distribution<double> random_weight(0.0, 2.0*max_weight);
		beta += random_weight(gen);

		while (beta > weights[index])
		{
			beta -= weights[index];
			index = (index + 1) % num_particles;
		}
		p3.push_back(particles[index]);
	}

	for (int i = 0; i < num_particles; i++)
	{
		particles[i] = p3[i];
	}
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
