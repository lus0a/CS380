#version 460

in vec3 DiffuseColor;
in vec3 SpecularColor;
in vec2 TexCoord;

out vec4 FragColor;

uniform vec2 Scale;
uniform vec2 Threshold;
uniform vec3 SurfaceColor;

void main()
{
	float ss = fract(TexCoord.s * Scale.s);
	float tt = fract(TexCoord.t * Scale.t);
	
	if ( (ss>Threshold.s) && ( tt>Threshold.t)) discard;
	vec3 finalColor = SurfaceColor * DiffuseColor + SpecularColor;
	FragColor = vec4(finalColor, 1.0);
}
