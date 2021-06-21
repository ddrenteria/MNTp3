#include <algorithm>
//#include <chrono>
#include <pybind11/pybind11.h>
#include <iostream>
#include <exception>
#include "linear_regression.h"

using namespace std;
namespace py=pybind11;

LinearRegression::LinearRegression() {
}

void LinearRegression::fit(Matrix X, Matrix y) {
    Matrix A = X.transpose() * X; 
    Matrix b = X.transpose() * y;
    _cm_solution = A.colPivHouseholderQr().solve(b);
    cout << _cm_solution << endl;
}


Matrix LinearRegression::predict(Matrix X) {
    return X * _cm_solution;
}

Matrix LinearRegression::coef() {
    return _cm_solution;
}
