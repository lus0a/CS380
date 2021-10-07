#version 330 core

//take the VBOs
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;


out vec2 TexCoord;

void main()
{
    
    TexCoord = aTexCoord;
    gl_Position = vec4(aPos, 1.0);


} 


