//
// Created by Sahar on 14/07/2022.
//

#pragma once


#include "VertexArray.h"
#include "IndexBuffer.h"
#include "Shader.h"

class GUIRenderer {
public:

    void clear() const;
    void draw(const VertexArray &va, const IndexBuffer &ib, const Shader &shader) const;



private:

};
