#pragma once

#include "types.h"

class LinearRegression {
public:
    LinearRegression();

    void fit(Matrix X, Matrix y);

    Matrix predict(Matrix X);

    Matrix coef();
private:
    Matrix _cm_solution;
};
