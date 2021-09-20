#version 460

uniform vec3 LightPosition;

uniform mat4 ModelViewMatrix;
uniform mat4 MVP;
uniform mat3 NormalMatrix;

in vec4 MCVertex;
in vec3 MCNormal;
in vec3 MCTangent;
in vec2 TexCoord0;

out vec3 LightDir;
out vec3 EyeDir;
out vec2 TexCoord; 

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
	mat4 MVMatrix = view*model;
	mat4 MVPMatrix = projection*view*model;

	gl_Position = MVPMatrix * MCVertex;
	EyeDir = vec3(MVMatrix * MCVertex);
	TexCoord = TexCoord0.st;
	vec3 n = normalize(NormalMatrix * MCNormal);
	vec3 t = normalize(NormalMatrix * MCTangent);
	vec3 b = cross(n, t);

	vec3 v;
	v.x = dot(LightPosition, t);
	v.y = dot(LightPosition, b);
	v.z = dot(LightPosition, n);
	LightDir = normalize(v);

	v.x = dot(EyeDir,t);
	v.y = dot(EyeDir,b);
	v.z = dot(EyeDir,n);
	EyeDir = normalize(v);
}


