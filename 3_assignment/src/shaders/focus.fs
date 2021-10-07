#version 330 core

uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D gAlbedoSpec;
uniform sampler2D gDepth;

//layout (location = 3) in vec3 gdepth;

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


// uniform int kernelSize;
// uniform int myInt;
uniform float Plane;

void main()
{

vec3 sum = vec3(0.0f);
//sum += texture( gAlbedoSpec, TexCoord + vec2(0, 0) ).rgb;
//float Distance = abs(Plane - gl_FragCoord.z);
float depth = dot(vec3(1.0, 1.0, 1.0),texture( gDepth, TexCoord ).rgb)/3.0;
float Distance = abs(Plane - depth);
int myInt = int(Distance*15);

if (myInt == 0)
  FragColor = vec4(vec3(texture( gAlbedoSpec, TexCoord ).rgb),1.0f);
else
{
for (int i=(-1*myInt); i <= myInt; i++)
    for (int j=(-1*myInt); j <= myInt; j++)
    {
        sum += texture( gAlbedoSpec, TexCoord + vec2(dx*float(j),dy*float(i)) ).rgb; 
    }

FragColor = vec4(vec3(sum/float(4 * myInt * myInt)),1.0f);
}

}