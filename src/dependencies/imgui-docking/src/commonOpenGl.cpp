//
// Created by Sahar on 13/07/2022.
//


#include <glew-2.1.0/include/GL/glew.h>
#include <iostream>


void GLClearError() {
    while (glGetError() != GL_NO_ERROR);
}

void GLCheckError(const char *const func, const char *const file, const int line) { // Don't change signature.
    while (GLenum error = glGetError()) {
        std::cerr << "[OpenGL Error] (" << error << ")" <<
                  " at " << file << ":" << line << " '" << func << "' \n";
    }
}
