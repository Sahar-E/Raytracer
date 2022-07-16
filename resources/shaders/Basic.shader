#shader vertex
#version 330 core

layout(location = 0) in vec4 position;
layout(location = 1) in vec2 texCoords;

out vec2 v_texCoords;

uniform mat4 u_mvpMatrix;

void main() {
   gl_Position = u_mvpMatrix * position;
   v_texCoords = texCoords;
};


#shader fragment
#version 330 core

layout(location = 0) out vec4 color;

in vec2 v_texCoords;

// uniform vec4 u_color;
uniform sampler2D u_texture;

void main() {
   vec4 texColor = texture(u_texture, v_texCoords);
   color = texColor;
};
