#version 460

layout (location = 0) in vec3 VertexPosition;
layout (location = 1) in vec3 VertexNormal;

out vec3 Position;
out vec3 Normal;

uniform mat4 ModelViewMatrix;
uniform mat3 NormalMatrix;
uniform mat4 ProjectionMatrix;
uniform mat4 MVP;

void main()
{
    Normal = normalize( NormalMatrix * VertexNormal);
    Position = (ModelViewMatrix * vec4(VertexPosition,1.0)).xyz;
    gl_Position = MVP * vec4(VertexPosition,1.0);
}

//struct LightInfo {
//  vec4 Position; // Light position in eye coords.
//  vec3 La;       // Ambient light intensity
//  vec3 Ld;       // Diffuse light intensity
//  vec3 Ls;       // Specular light intensity
//};
//uniform LightInfo Light;
//
//struct MaterialInfo {
//  vec3 Ka;            // Ambient reflectivity
//  vec3 Kd;            // Diffuse reflectivity
//  vec3 Ks;            // Specular reflectivity
//  float Shininess;    // Specular shininess factor
//};
//uniform MaterialInfo Material;
//
//
//layout (location = 0) in vec3 aPos;
//layout (location = 1) in vec3 aNormal;
//
//
//uniform mat4 model;
//uniform mat4 view;
//uniform mat4 projection;
//
//
//out vec3 Normal;
//out vec3 FragPos;
//
//
//void main()
//{
//	FragPos = vec3(model * vec4(aPos, 1.0));
//	mat4 mv = view * model;
//	mat3 NormalMatrix = mat3(vec3(mv[0]), vec3(mv[1]),vec3(mv[2]));
//  vec3 Normal = normalize( NormalMatrix * aNormal);
//
//	gl_Position = projection * view * model * vec4(aPos,1.0);
//}
