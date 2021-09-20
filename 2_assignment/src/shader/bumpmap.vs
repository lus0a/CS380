#version 460

uniform vec3 LightPosition;

uniform mat4 MVMatrix;
uniform mat4 MVPMatrix;
uniform mat3 NormalMatrix;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

layout (location = 0) in vec3 MCVertex;
layout (location = 1) in vec3 MCNormal;
layout (location = 2) in vec2 TexCoord0;
layout (location = 3) in vec4 MCTangent;

out vec3 LightDir;
out vec3 EyeDir;
out vec2 TexCoord; 

void main()
{
	mat4 mv = view * model;
	mat4 MVMatrix = mv;
	mat3 NormalMatrix = mat3(vec3(mv[0]), vec3(mv[1]),vec3(mv[2]));
	mat4 MVPMatrix = projection * mv;

	gl_Position = MVPMatrix * vec4(MCVertex,1.0);
	EyeDir = vec3(MVMatrix *vec4(MCVertex,1.0));
	TexCoord = TexCoord0.st;
	vec3 n = normalize(NormalMatrix * MCNormal);
	vec3 t = normalize(NormalMatrix * vec3(MCTangent));
	vec3 b = cross(n, t);
	vec3 v;
	v.x = dot(LightPosition, t);
	v.y = dot(LightPosition, b);
	v.z = dot(LightPosition, n);
	LightDir = normalize(v);
	//LightDir = LightPosition;
	v.x = dot(EyeDir,t);
	v.y = dot(EyeDir,b);
	v.z = dot(EyeDir,n);
	EyeDir = normalize(v);
}


