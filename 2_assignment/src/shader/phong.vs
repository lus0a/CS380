#version 400

layout (location = 0) in vec3 VertexPosition;
layout (location = 1) in vec3 VertexNormal;

out vec3 tnorm;
out vec4 eyeCoords;

struct LightInfo {
  vec4 Position; // Light position in eye coords.
  vec3 La;       // Ambient light intensity
  vec3 Ld;       // Diffuse light intensity
  vec3 Ls;       // Specular light intensity
};
uniform LightInfo Light;

struct MaterialInfo {
  vec3 Ka;            // Ambient reflectivity
  vec3 Kd;            // Diffuse reflectivity
  vec3 Ks;            // Specular reflectivity
  float Shininess;    // Specular shininess factor
};
uniform MaterialInfo Material;



uniform mat4 ModelViewMatrix;
uniform mat3 NormalMatrix;
uniform mat4 ProjectionMatrix;
uniform mat4 MVP;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
 		mat4 mv = view * model;
		mat4 ModelViewMatrix = mv;
		mat3 NormalMatrix = mat3(vec3(mv[0]), vec3(mv[1]),vec3(mv[2]));
		mat4 MVP = projection * mv;
		mat4 ProjectionMatrix = projection;

    tnorm = normalize( NormalMatrix * VertexNormal);
    eyeCoords = ModelViewMatrix * vec4(VertexPosition,1.0);
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
