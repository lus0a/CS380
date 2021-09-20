#version 460

in vec3 Position;
in vec3 Normal;

layout (location = 0) out vec4 FragColor;

uniform struct LightInfo {
  vec4 Position; // Light position in eye coords.
  vec3 La;       // Ambient light intensity
  vec3 Ld;       // Diffuse light intensity
  vec3 Ls;       // Specular light intensity
} Light;

uniform struct MaterialInfo {
  vec3 Ka;            // Ambient reflectivity
  vec3 Kd;            // Diffuse reflectivity
  vec3 Ks;            // Specular reflectivity
  float Shininess;    // Specular shininess factor
} Material;

uniform mat4 ModelViewMatrix;
uniform mat3 NormalMatrix;
uniform mat4 ProjectionMatrix;
uniform mat4 MVP;

vec3 phongModel( vec3 position, vec3 normal )
{
    vec3 n = normalize( NormalMatrix * normal);
    vec4 camCoords = ModelViewMatrix * vec4(position,1.0);
    vec3 ambient = Light.La * Material.Ka;
    vec3 s = normalize(vec3(Light.Position - camCoords));
    float sDotN = max( dot(s,n), 0.0 );
    vec3 diffuse = Light.Ld * Material.Kd * sDotN;
    vec3 spec = vec3(0.0);
    if( sDotN > 0.0 ){
      vec3 v = normalize(-camCoords.xyz);
      vec3 r = reflect( -s, n );
      spec = Light.Ls * Material.Ks * pow( max( dot(r,v), 0.0 ), Material.Shininess );
    }

    vec3 LightIntensity = ambient + diffuse + spec;
    return LightIntensity;
}

void main() {
    FragColor = vec4(phongModel(Position, normalize(Normal)), 1);
}




