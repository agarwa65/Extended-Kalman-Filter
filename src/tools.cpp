#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
    * Calculate the RMSE here.
  */

	VectorXd rmse(4);
	rmse << 0,0,0,0;

	if(estimations.size() != ground_truth.size() || estimations.size() == 0){
		// std::cout <<  "Invalid estimation or ground_truth data" << '\n';
    	return rmse;
	}

	//accumulate squared residuals
	for(unsigned int i=0; i < estimations.size(); ++i){

		VectorXd residual = estimations[i] - ground_truth[i];

		//coefficient-wise multiplication
		residual = residual.array()*residual.array();
		rmse += residual;
	}

	//calculate the mean
	rmse = rmse/estimations.size();

	//calculate the squared root
	rmse = rmse.array().sqrt();

	//return the result
	return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
    * Calculate a Jacobian here.
  */

  // Declare Hj:
  MatrixXd Hj(3,4);

  // recover state vector terms
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  // Calculate intermediate terms to avoid repeated calculation
  float c1 = px*px + py*py;
  // Check for division by zero:
  if (c1 < 0.00001){
    // std::cout << "CalculateJacobian () - Error - Division by Zero" << std::endl;
    // return Hj;
    // Add a small normalizing factor to c1
    c1 += 0.00001;
  }
  float c2 = sqrt(c1);
  float c3 = c1 * c2;

  //compute the Jacobian Matrix
  Hj << (px/c2), (py/c2), 0, 0,
        -(py/c1), (px/c1), 0, 0,
        py*(vx*py-vy*py)/c3, px*(px*vy-vx*py)/c3, px/c2, py/c2;

  return Hj;
}
