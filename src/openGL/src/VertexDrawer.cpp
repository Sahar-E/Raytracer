//
// Created by Sahar on 14/07/2022.
//

#include "VertexDrawer.h"
#include "commonOpenGL.h"

void VertexDrawer::draw(const VertexArray &va, const IndexBuffer &ib, const Shader &shader) {
    shader.bind();
    ib.bind();
    va.bind();

    checkGLErrors(glDrawElements(GL_TRIANGLES, ib.getCount(), GL_UNSIGNED_INT, nullptr));

}

void VertexDrawer::clear() {
    checkGLErrors(glClear(GL_COLOR_BUFFER_BIT));
}
