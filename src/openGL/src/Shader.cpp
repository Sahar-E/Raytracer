//
// Created by Sahar on 15/07/2022.
//

#include "Shader.h"
#include "glew-2.1.0/include/GL/glew.h"
#include "commonOpenGL.h"

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <memory>


Shader::Shader(const std::string &filepath) : _filepath(filepath), _rendererId(0) {
    auto [vertexShader, fragmentShader] = parseShader(filepath);
    _rendererId = createShader(vertexShader, fragmentShader);

}

Shader::~Shader() {
    checkGLErrors(glDeleteProgram(_rendererId));
}

void Shader::bind() const {
    checkGLErrors(glUseProgram(_rendererId));
}

void Shader::unbind() const {
    checkGLErrors(glUseProgram(0));
}

void Shader::setUniform4f(const std::string &name, float v0, float v1, float v2, float v3) {
    checkGLErrors(glUniform4f(getUniformLocation(name), v0, v1, v2, v3));
}

void Shader::setUniformMat4f(const std::string &name, const glm::mat4 &matrix) {
    checkGLErrors(glUniformMatrix4fv(getUniformLocation(name), 1, GL_FALSE, &matrix[0][0]));
}

void Shader::setUniform1i(const std::string &name, int value) {
    checkGLErrors(glUniform1i(getUniformLocation(name), value));
}

int Shader::getUniformLocation(const std::string &name) {
    if (_uniformLocationCache.find(name) != _uniformLocationCache.end()) {
        return _uniformLocationCache[name];
    }
    checkGLErrors(int location = glGetUniformLocation(_rendererId, name.c_str()));
    if (location == -1) {
        std::cerr << "Failed to find uniform " << name << "\n";
    }
    _uniformLocationCache[name] = location;
    return location;
}


unsigned int Shader::compileShader(unsigned int type, const std::string &source) {
    checkGLErrors(unsigned int id = glCreateShader(type));
    const char *src = source.c_str();
    checkGLErrors(glShaderSource(id, 1, &src, nullptr));
    checkGLErrors(glCompileShader(id));

    int result;
    checkGLErrors(glGetShaderiv(id, GL_COMPILE_STATUS, &result));
    if (result == GL_FALSE) {
        int length;
        checkGLErrors(glGetShaderiv(id, GL_INFO_LOG_LENGTH, &length));
        auto log = std::make_unique<char[]>(length + 1);
        checkGLErrors(glGetShaderInfoLog(id,length, &length, log.get()));
        auto whatFailed = (type == GL_VERTEX_SHADER) ? "shader!\n" : "fragment!\n";
        std::cerr << "Failed to compile" << whatFailed;
        std::cerr << log.get() << std::endl;
        checkGLErrors(glDeleteShader(id));
        return 0;
    }

    return id;
}

unsigned int Shader::createShader(const std::string& vertexShader, const std::string& fragmentShader){
    checkGLErrors(unsigned int program = glCreateProgram());
    unsigned int vs =  compileShader(GL_VERTEX_SHADER, vertexShader);
    unsigned int fs =  compileShader(GL_FRAGMENT_SHADER, fragmentShader);

    checkGLErrors(glAttachShader(program, vs));
    checkGLErrors(glAttachShader(program, fs));
    checkGLErrors(glLinkProgram(program));
    checkGLErrors(glValidateProgram(program));

    checkGLErrors(glDeleteShader(vs));
    checkGLErrors(glDeleteShader(fs));

    return program;
}

std::tuple<std::string, std::string> Shader::parseShader(const std::string &filepath) {
    enum class ShaderType {
        NONE = -1, VERTEX = 0,FRAGMENT = 1
    };
    std::ifstream stream(filepath);
    std::string line;
    std::stringstream stringstream[2];
    ShaderType type = ShaderType::NONE;
    while (std::getline(stream, line)) {
        if (line.find("#shader") != std::string::npos) {
            if (line.find("vertex") != std::string::npos) {
                type = ShaderType::VERTEX;
            } else if (line.find("fragment") != std::string::npos) {
                type = ShaderType::FRAGMENT;
            }
        } else {
            stringstream[static_cast<int>(type)] << line << '\n';
        }
    }
    return {stringstream[static_cast<int>(ShaderType::VERTEX)].str(),
            stringstream[static_cast<int>(ShaderType::FRAGMENT)].str()};
}