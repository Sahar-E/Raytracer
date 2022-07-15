//
// Created by Sahar on 14/07/2022.
//

#include "GUIRenderer.h"
#include "commonOpenGL.h"

void GUIRenderer::draw(const VertexArray &va, const IndexBuffer &ib, const Shader &shader) const{
    shader.bind();
    ib.bind();
    va.bind();

    checkGLErrors(glDrawElements(GL_TRIANGLES, ib.getCount(), GL_UNSIGNED_INT, nullptr));

}

void GUIRenderer::clear() const {
    checkGLErrors(glClear(GL_COLOR_BUFFER_BIT));
}
