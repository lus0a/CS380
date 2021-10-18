#version 330 core

uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D gAlbedoSpec;
uniform sampler2D gDepth;

in vec2 TexCoord;

uniform vec3 viewPos; 
uniform vec3 lightPos;
uniform vec3 lightColor;
uniform vec3 objectColor;
out vec4 FragColor;
int Width = 1024;
int Height = 1024; 

float dx = 1.0 / float(Width);
float dy = 1.0 / float(Height);


float luma(vec3 color)
{
  return dot(vec3(1.0, 1.0, 1.0),color); //vec3(0.2126, 0.7152, 0.0722)
}
void main()
{


 float s00 = luma(texture( gDepth, TexCoord + vec2(-dx,dy) ).rgb);
 float s10 = luma(texture( gDepth, TexCoord + vec2(-dx,0.0) ).rgb);
 float s20 = luma(texture( gDepth, TexCoord + vec2(-dx,-dy) ).rgb);
 float s01 = luma(texture( gDepth, TexCoord + vec2(0.0,dy) ).rgb);
 float s21 = luma(texture( gDepth, TexCoord + vec2(0.0,-dy) ).rgb);
 float s02 = luma(texture( gDepth, TexCoord + vec2(dx, dy) ).rgb);
 float s12 = luma(texture( gDepth, TexCoord + vec2(dx, 0.0) ).rgb);
 float s22 = luma(texture( gDepth, TexCoord + vec2(dx, -dy) ).rgb);


  // float sx = s00 + 2 * s10 + s20 - (s02 + 2 * s12 + s22);
  // float sy = s00 + 2 * s01 + s02 - (s20 + 2 * s21 + s22);

  // float g = sx * sx + sy * sy;
  float g = s00 + s10 + s20 + s00 + s01 + s02 - (s02 +  s12 + s22 + s20 + s21 + s22);

    if( g > 0.005f )
        FragColor = vec4(1.0,1.0,1.0,1.0);
    else
        FragColor = vec4(0.0,0.0,0.0,1.0);

//  vec3 ngDepth = texture(gDepth, TexCoord).rgb;
//  FragColor = vec4(ngDepth,1.0);

} 