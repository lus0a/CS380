#version 330 core
layout (location = 0) out vec3 gPosition;
layout (location = 1) out vec3 gNormal;
layout (location = 2) out vec4 gAlbedoSpec;
layout (location = 3) out vec3 gDepth;

in vec2 TexCoord;
in vec3 FragPos;
in vec3 Normal;
in vec3 Color;

//uniform sampler2D texture_diffuse1;
//uniform sampler2D texture_specular1;
uniform vec3 lightPos; 
uniform vec3 viewPos; 
uniform vec3 lightColor;

out vec4 FragColor;

 float near=0.1f; 
 float far = 500.0f; 
  
float LinearizeDepth(float depth) 
{
    float z = depth * 2.0 - 1.0; // back to NDC 
    return (2.0 * near * far ) / (far + near - z * (far - near));	
}

void main()
{
   

  // store the fragment position vector in the first gbuffer texture
    gPosition = FragPos;
    // // also store the per-fragment normals into the gbuffer
    gNormal = normalize(Normal);
    // // and the diffuse per-fragment color
    gAlbedoSpec.rgb = Color;
    //linear depth 
    float depth =LinearizeDepth(gl_FragCoord.z) / far; // divide by far for demonstration
    gDepth = vec3(depth);
   //FragColor = vec4(vec3(depth), 1.0);


  //FragColor = vec4(Color, 1.0);
} 