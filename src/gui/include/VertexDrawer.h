//
// Created by Sahar on 14/07/2022.
//

#pragma once


#include "VertexArray.h"
#include "IndexBuffer.h"
#include "Shader.h"

class VertexDrawer {
public:

    static void clear() ;
    static void draw(const VertexArray &va, const IndexBuffer &ib, const Shader &shader) ;



private:

};
