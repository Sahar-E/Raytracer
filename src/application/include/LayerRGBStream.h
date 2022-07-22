//
// Created by Sahar on 22/07/2022.
//

#pragma once


#include "glew-2.1.0/include/GL/glew.h"
#include "GLFW/glfw3.h"
#include "Layer.h"
#include <memory>

#include "VertexArray.h"
#include "IndexBuffer.h"
#include "Shader.h"
#include "LiveTexture.h"
#include "InputHandler.h"
#include "VertexDrawer.h"


class LayerRGBStream : public Layer {
public:
    LayerRGBStream(std::shared_ptr<unsigned char[]> rgbBuffer, int rgbImageWidth, int rgbImageHeight);

    void onUpdate() override;
    void onAttach() override;
    void onDetach() override;

private:

private:
    const std::shared_ptr<unsigned char[]> _rgbBuffer;
    int _rgb_image_width;
    int _rgb_image_height;

    std::shared_ptr<VertexArray> _va;
    std::shared_ptr<VertexBuffer> _vb;
    std::shared_ptr<VertexBufferLayout> _layout;
    std::shared_ptr<IndexBuffer> _ib;
    std::shared_ptr<Shader> _shader;
    std::shared_ptr<LiveTexture> _texture;

    glm::mat4 _proj;
    glm::mat4 _view;
    glm::mat4 _model;

    std::shared_ptr<InputHandler> _inputHandler;
};
