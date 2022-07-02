//
// Created by Sahar on 30/06/2022.
//

#pragma once
#include <chrono>
#include <iostream>
#include <utility>


/**
 * Just put this class as an instance in a scope, and with its destruction, a print of the duration it lived will be
 * printed.
 */
class TimeThis {
public:
    TimeThis() : _startTime(std::chrono::steady_clock::now()) {}

    explicit TimeThis(std::string  title) : _startTime(std::chrono::steady_clock::now()), _title(std::move(title)){}

    virtual ~TimeThis() {
        auto endTime = std::chrono::steady_clock::now();
        long duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - _startTime).count();
        if (_title.length() == 0) {
            std::cout << "Duration (ms): " << duration << "\n";
        } else {
            std::cout << "Duration of " << _title << " is (ms): " << duration << "\n";
        }
    }

    std::chrono::time_point<std::chrono::steady_clock> _startTime;
    std::string _title;
};