//
// Created by Sahar on 14/07/2022.
//

#pragma once


#include "VertexArray.h"
#include "IndexBuffer.h"
#include "Shader.h"

/**
 * Responsible for drawing into shaders.
 */
class VertexDrawer {
public:

    /**
     * Clear. Called once per main loop iteration.
     */
    static void clear();

    /**
     * Draw the vertex array data according to the index buffer into the shader.
     * @param va          The vertex array buffer data.
     * @param ib          The index buffer.
     * @param shader      The shader to draw the into.
     */
    static void draw(const VertexArray &va, const IndexBuffer &ib, const Shader &shader);
};
