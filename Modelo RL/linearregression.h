#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H

#include <iostream>
#include <eigen3/Eigen/Dense>
#include <list>
#include <vector>
#include <fstream>

class LinearRegression
{
public:
    float F_OLS_Costo(Eigen::MatrixXd X,Eigen::MatrixXd y, Eigen::MatrixXd thetas);
    std::tuple<Eigen::VectorXd, std::vector<float>> GradientDescent(Eigen::MatrixXd X,
                                                                     Eigen::MatrixXd y,
                                                                     Eigen::VectorXd thetas,
                                                                     float alpha,
                                                                     int num_iter);
    float R2_Score(Eigen::MatrixXd y, Eigen::MatrixXd y_hat);
};

#endif // LINEARREGRESSION_H
