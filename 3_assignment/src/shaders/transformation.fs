#version 330 core


//in vec3 Color;
out vec4 FragColor;
in vec3 Color; // the input variable from the vertex shader (same name and same type)  
//in vec3 Normal;

void main()
{
  //FragColor =Color;
  FragColor = vec4(Color, 1.0);
} 