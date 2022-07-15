//
// Created by Sahar on 15/07/2022.
//

#pragma once

#include <string>
#include <unordered_map>

class Shader {
public:
    explicit Shader(const std::string &filepath);
    virtual ~Shader();

    Shader(const Shader &other) = delete;
    Shader &operator=(const Shader &other) = delete;

    void bind() const;
    void unbind() const;

    void setUniform(const std::string &name, float v0, float v1, float v2, float v3);

private:
    unsigned int _rendererId;
    std::string _filepath;
    std::unordered_map<std::string, int> _uniformLocationCache;

    unsigned int getUniformLocation(const std::string &name);
    unsigned int compileShader(unsigned int type, const std::string &source);
    unsigned int createShader(const std::string& vertexShader, const std::string& fragmentShader);
    std::tuple<std::string, std::string> parseShader(const std::string &filepath);
};
