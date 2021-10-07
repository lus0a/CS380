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

uniform int kernelSize;
uniform int myInt;

void main()
{

vec3 sum = vec3(0.0f);
//sum += texture( gAlbedoSpec, TexCoord + vec2(0, 0) ).rgb;
for (int i=(-1*myInt); i <= myInt; i++)
    for (int j=(-1*myInt); j <= myInt; j++)
    {
        sum += texture( gAlbedoSpec, TexCoord + vec2(dx*float(j),dy*float(i)) ).rgb; 
    }

FragColor = vec4(vec3(sum/float(4 * myInt * myInt)),1.0f) ;

}