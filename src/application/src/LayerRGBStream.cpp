//
// Created by Sahar on 22/07/2022.
//

#include <iostream>
#include <utility>

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "LayerRGBStream.h"
#include "commonOpenGL.h"


LayerRGBStream::LayerRGBStream(std::shared_ptr<unsigned char[]> rgbBuffer,
                               int rgbImageWidth,
                               int rgbImageHeight) : _rgbBuffer(std::move(rgbBuffer)),
                                                     _rgb_image_width(rgbImageWidth),
                                                     _rgb_image_height(rgbImageHeight),
                                                     _proj(glm::ortho(-1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f)),
                                                     _view(glm::translate(glm::mat4(1.0f), glm::vec3(-0, 0, 0))),
                                                     _model(glm::scale(glm::mat4(1.0f), glm::vec3(1.0f, -1.0f, 1.0f))) {
}

void LayerRGBStream::onUpdate() {
    _texture->updateTexture();

    glm::mat4 mvp = _proj * _view * _model;
    _shader->bind();
    _shader->setUniformMat4f("u_mvpMatrix", mvp);
    VertexDrawer::draw(*_va, *_ib, *_shader);
}

void LayerRGBStream::onAttach() {
    float positions[] = {
            -0.9999f, -0.9999f, 0.0f, 0.0f,
            0.9999f, -0.9999f, 1.0f, 0.0f,
            0.9999f, 0.9999f, 1.0f, 1.0f,
            -0.9999f, 0.9999f, 0.0f, 1.0f
    };
    unsigned int indices[] = {
            0, 1, 2,
            2, 3, 0
    };


    _va = std::make_shared<VertexArray>();
    _vb = std::make_shared<VertexBuffer>(positions, sizeof(float) * 4 * 4);

    _layout = std::make_shared<VertexBufferLayout>();
    _layout->push<float>(2);
    _layout->push<float>(2);
    _va->addBuffer(*_vb, *_layout);

    _ib = std::make_shared<IndexBuffer>(indices, 6);

    _shader = std::make_shared<Shader>("resources/shaders/Basic.shader");
    _shader->bind();
    _texture = std::make_shared<LiveTexture>(_rgbBuffer, _rgb_image_width, _rgb_image_height, GL_RGB);
    _texture->bind();
    _shader->setUniform1i("u_texture", 0);

    _va->unbind();
    _vb->unbind();
    _ib->unbind();
    _shader->unbind();
}

void LayerRGBStream::onDetach() {
}

