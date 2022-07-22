//
// Created by Sahar on 21/07/2022.
//

#pragma once


class MouseInput {

public:

    [[nodiscard]] double getXPos() const;
    void setXPos(double xPos);
    [[nodiscard]] double getYPos() const;
    void setYPos(double yPos);

private:
    double _xPos;
    double _yPos;
};
