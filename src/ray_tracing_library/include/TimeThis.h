//
// Created by Sahar on 30/06/2022.
//

#pragma once
#include <chrono>
#include <iostream>

class TimeThis {
public:
    TimeThis() : _startTime(std::chrono::steady_clock::now()) {
    }

    virtual ~TimeThis() {
        auto endTime = std::chrono::steady_clock::now();
        long duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - _startTime).count();
        std::cout << "Duration (ms): " << duration << "\n";
    }

    std::chrono::time_point<std::chrono::steady_clock> _startTime;
};