//
// Created by Sahar on 15/07/2022.
//

#pragma once

#include <string>
#include <unordered_map>
#include "glm/glm.hpp"

class Shader {
public:
    explicit Shader(const std::string &filepath);
    virtual ~Shader();

    Shader(const Shader &other) = delete;
    Shader &operator=(const Shader &other) = delete;

    void bind() const;
    void unbind() const;

    void setUniform1i(const std::string &name, int value);
    void setUniform4f(const std::string &name, float v0, float v1, float v2, float v3);

    void setUniformMat4f(const std::string &name, const glm::mat4 &matrix);

private:
    unsigned int _rendererId;
    std::string _filepath;
    std::unordered_map<std::string, int> _uniformLocationCache;

    int getUniformLocation(const std::string &name);
    unsigned int compileShader(unsigned int type, const std::string &source);
    unsigned int createShader(const std::string& vertexShader, const std::string& fragmentShader);
    std::tuple<std::string, std::string> parseShader(const std::string &filepath);
};
