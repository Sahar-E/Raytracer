//
// Created by Sahar on 24/07/2022.
//

#pragma once

#define BIT(x) ((1) << (x))
#define BIND_EVENT_FN(fn) [this](auto&&... args) -> decltype(auto) { return this->fn(std::forward<decltype(args)>(args)...); }