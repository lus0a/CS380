#version 460

layout (location = 0) in vec3 VertexPosition;
layout (location = 1) in vec3 VertexNormal;

out vec3 Position;
out vec3 Normal;

//uniform mat4 ModelViewMatrix;
//uniform mat3 NormalMatrix;
//uniform mat4 ProjectionMatrix;
//uniform mat4 MVP;

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
    Normal = normalize( NormalMatrix * VertexNormal);
    Position = vec3( ModelViewMatrix * vec4(VertexPosition,1.0) );
	gl_Position = MVP * vec4(VertexPosition,1.0);
}
