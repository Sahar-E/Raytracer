//
// Created by Sahar on 13/07/2022.
//

#pragma once


#define checkGLErrors(x) GLClearError();\
        x;\
        GLCheckError(#x, __FILE__, __LINE__ )


void GLClearError();

void GLCheckError(char const *const func, const char *const file, int const line);  // Don't change signature.
