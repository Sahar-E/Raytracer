//
// Created by Sahar on 08/06/2022.
//

#pragma once
#include <iostream>

class Vec3 {
public:
    Vec3() : _x{0, 0, 0}{}

    Vec3(double x0, double x1, double x2) : _x{x0, x1, x2} {}

    [[nodiscard]] double x0() const { return _x[0]; }

    [[nodiscard]] double x1() const { return _x[1]; }

    [[nodiscard]] double x2() const { return _x[2]; }


private:
    double _x[3];
};

