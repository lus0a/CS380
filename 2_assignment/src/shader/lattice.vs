#version 460

uniform vec3 LightPosition;
uniform vec3 LightColor;
uniform vec3 EyePosition;
uniform vec3 Specular;
uniform vec3 Ambient;
uniform float Kd;

uniform mat4 MVMatrix;
uniform mat4 MVPMatrix;
uniform mat3 NormalMatrix;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

layout (location = 0) in vec3 MCVertex;
layout (location = 1) in vec3 MCNormal;
layout (location = 2) in vec2 TexCoord0;

out vec3 DiffuseColor;
out vec3 SpecularColor;
out vec2 TexCoord; 

void main()
{
	mat4 mv = view * model;
	mat4 MVMatrix = mv;
	mat3 NormalMatrix = mat3(vec3(mv[0]), vec3(mv[1]),vec3(mv[2]));
	mat4 MVPMatrix = projection * mv;

	vec3 ecPosition = vec3(MVMatrix * vec4(MCVertex,1.0));
	vec3 tnorm = normalize(NormalMatrix * MCNormal);
	vec3 lightVec = normalize(LightPosition - ecPosition);
	vec3 viewVec = normalize(EyePosition - ecPosition);
	vec3 hvec = normalize(viewVec + lightVec);

	float spec = clamp(dot(hvec, tnorm), 0.0, 1.0);
	spec = pow(spec, 16.0);

	DiffuseColor = LightColor * vec3(Kd * dot(lightVec, tnorm));
	DiffuseColor = clamp(Ambient + DiffuseColor, 0.0, 1.0);
	SpecularColor = clamp(LightColor * Specular * spec, 0.0, 1.0);
	TexCoord = TexCoord0;
	gl_Position = MVPMatrix * vec4(MCVertex,1.0);
}
