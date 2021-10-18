// CS 380 - GPGPU Programming, KAUST
//
// Programming Assignment #3

// includes
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <sstream>
#include <assert.h>
#include <math.h>
#include <vector>

#include "glad/glad.h" 
#include "GLFW/glfw3.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/glm.hpp>


// framework includes
#include "vbocube.h"
#include "vbomesh.h"
#include "vboquad.h"
#include "glslprogram.h"

// window size
const unsigned int gWindowWidth = 1024;
const unsigned int gWindowHeight = 1024;

// a simple cube
VBOCube* m_pCube;

//Quad
Quad* quad;

// The Balloon Mesh
std::vector<VBOMesh*> m_pMeshList;
std::vector<glm::mat4> m_modelList;
std::vector<glm::vec3> m_colors;

// Plane Settings
float zFar = 500.0f;
float zNear = 0.1f;
float fov = 190.0f;
float z = 0.5f;

unsigned int gBuffer;
unsigned int gPosition, gNormal, gAlbedoSpec, gDepth;

int shaderType = 1;

int kernelSize =3;
int myInt = std::floor(kernelSize/2);

// TODO: define glsl programms
GLSLProgram program;
GLSLProgram geometryPassProgram;
GLSLProgram secondPassProgram;
GLSLProgram edgeD;
GLSLProgram blur;
GLSLProgram focus;

// glfw error callback
void glfwErrorCallback(int error, const char* description)
{
	fprintf(stderr, "Error: %s\n", description);
}

/* float gauss(float x, float sigma2)
{
	
	double coeff = 1.0 / (glm::two_pi<double>() * sigma2);
	double expon = -(x * x) / (2.0 * sigma2);
	return (float)(coeff * exp(expon));
} */

void key_callback( GLFWwindow* window, int key, int scancode, int action, int mods )
{   
	if ( key == GLFW_KEY_0 && action == GLFW_PRESS )
		shaderType = 0;
	else if ( key == GLFW_KEY_1 && action == GLFW_PRESS )
		shaderType = 1;
	else if ( key == GLFW_KEY_2 && action == GLFW_PRESS )
		shaderType = 2;
	else if ( key == GLFW_KEY_3 && action == GLFW_PRESS )
	{
		shaderType = 3;
	}
	else if ( key == GLFW_KEY_4 && action == GLFW_PRESS )
	{
		shaderType = 4;
	}
	else if ( key == GLFW_KEY_UP && action == GLFW_PRESS )
	{
		kernelSize += 2;
		myInt += 1;
	}
	else if ( key == GLFW_KEY_DOWN && action == GLFW_PRESS )
	{
		kernelSize -= 2;
		myInt -= 1;
	}
	else if ( key == GLFW_KEY_W && action == GLFW_PRESS )
	{
		if (z <= 0.9)
			z += 0.1;
		else
			z = 1.0;
	}
	else if ( key == GLFW_KEY_S && action == GLFW_PRESS )
	{
		if (z >= 0.1)
			z -= 0.1;
		else
			z = 0.0;
	}	
	
}  

// OpenGL error debugging callback
void APIENTRY glDebugOutput(GLenum source,
	GLenum type,
	GLuint id,
	GLenum severity,
	GLsizei length,
	const GLchar* message,
	const void* userParam)
{
	// ignore non-significant error/warning codes
	if (id == 131169 || id == 131185 || id == 131218 || id == 131204) return;

	std::cout << "---------------" << std::endl;
	std::cout << "Debug message (" << id << "): " << message << std::endl;

	switch (source)
	{
	case GL_DEBUG_SOURCE_API:             std::cout << "Source: API"; break;
	case GL_DEBUG_SOURCE_WINDOW_SYSTEM:   std::cout << "Source: Window System"; break;
	case GL_DEBUG_SOURCE_SHADER_COMPILER: std::cout << "Source: Shader Compiler"; break;
	case GL_DEBUG_SOURCE_THIRD_PARTY:     std::cout << "Source: Third Party"; break;
	case GL_DEBUG_SOURCE_APPLICATION:     std::cout << "Source: Application"; break;
	case GL_DEBUG_SOURCE_OTHER:           std::cout << "Source: Other"; break;
	} std::cout << std::endl;

	switch (type)
	{
	case GL_DEBUG_TYPE_ERROR:               std::cout << "Type: Error"; break;
	case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: std::cout << "Type: Deprecated Behaviour"; break;
	case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:  std::cout << "Type: Undefined Behaviour"; break;
	case GL_DEBUG_TYPE_PORTABILITY:         std::cout << "Type: Portability"; break;
	case GL_DEBUG_TYPE_PERFORMANCE:         std::cout << "Type: Performance"; break;
	case GL_DEBUG_TYPE_MARKER:              std::cout << "Type: Marker"; break;
	case GL_DEBUG_TYPE_PUSH_GROUP:          std::cout << "Type: Push Group"; break;
	case GL_DEBUG_TYPE_POP_GROUP:           std::cout << "Type: Pop Group"; break;
	case GL_DEBUG_TYPE_OTHER:               std::cout << "Type: Other"; break;
	} std::cout << std::endl;

	switch (severity)
	{
	case GL_DEBUG_SEVERITY_HIGH:         std::cout << "Severity: high"; break;
	case GL_DEBUG_SEVERITY_MEDIUM:       std::cout << "Severity: medium"; break;
	case GL_DEBUG_SEVERITY_LOW:          std::cout << "Severity: low"; break;
	case GL_DEBUG_SEVERITY_NOTIFICATION: std::cout << "Severity: notification"; break;
	} std::cout << std::endl;
	std::cout << std::endl;
}


// init application 
// - load application specific data 
// - set application specific parameters
// - initialize stuff

glm::mat4 view;
glm::mat4 model;
glm::mat4 projection;
float angle = 45.0f; //projection
float rotateAng = glm::radians(90.0f);
glm::vec3 rotatAxis = glm::vec3(1.0f, 1.0f, 1.0f);
glm::vec3 scvar = glm::vec3(0.2f);

//camera variables 
glm::vec3 cameraPos = glm::vec3(0.0f, 0.0f, 3.0f);
glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);

//shaders variables 
glm::vec3 lightPos = glm::vec3(0.f, 0.f, 20.f);
glm::vec3 viewPos;
glm::vec3 viewCenter;
glm::vec3 upVec;
glm::vec3 lightColor = glm::vec3(1.0f, 1.0f, 1.0f);
glm::vec3 objectColor;

glm::vec3 Diffuse;
glm::vec3 Specular;


bool initApplication(int argc, char** argv)
{

	std::string version((const char*)glGetString(GL_VERSION));
	std::stringstream stream(version);
	unsigned major, minor;
	char dot;

	stream >> major >> dot >> minor;

	assert(dot == '.');
	if (major > 3 || (major == 2 && minor >= 0)) {
		std::cout << "OpenGL Version " << major << "." << minor << std::endl;
	}
	else {
		std::cout << "The minimum required OpenGL version is not supported on this machine. Supported is only " << major << "." << minor << std::endl;
		return false;
	}

	// default initialization
	glClearColor(0.1, 0.0, 0.0, 1.0);
	glDisable(GL_DEPTH_TEST);

	// viewport
	glViewport(0, 0, gWindowWidth, gWindowHeight);

	// projection
	//glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	return true;
}

// one time initialization to setup the 3D scene
void setupScene()
{
	// TODO: Set up a camera
	/* copy your solution from last assignment
	*/
	viewPos = glm::vec3(0.0f, 0.0f, 3.0f);//use it for both transformation and camera
	viewCenter = glm::vec3(0.0f, 0.0f, -1.0f);
	upVec = glm::vec3(0.0f, 1.0f, 0.0f);
	view = glm::lookAt(viewPos, // camera position: wheres the camera
		viewCenter, // center: what am I looking at?
		upVec); //up vector: is camera up or down 

	model = glm::mat4(1);
	projection = glm::perspective(glm::radians(fov), (float)gWindowWidth / (float)gWindowHeight, zNear, zFar);

	// TODO: Set up a light
	// define at least one light and use later as uniform(s)
	angle = 0.3f;
	vec4 lightPosition = vec4(10.0f * cos(angle), 10.0f, 10.0f * sin(angle), 1.0f);

	// TODO: Set up glsl programs (you need a vertex shaders and a fragment shaders for each pass --> 2 vertex and 2 fragment shaders) 
	/* Use GLSLProgram class or your own solution*/
	program.compileShader("./src/shaders/transformation.vs");
	program.compileShader("./src/shaders/transformation.fs");
	program.link();

	try {
		geometryPassProgram.compileShader("./src/shaders/GeometryPass.vs");
		secondPassProgram.compileShader("./src/shaders/SecondPass.vs");
		edgeD.compileShader("./src/shaders/edgeD.vs");
		blur.compileShader("./src/shaders/blur.vs");
		focus.compileShader("./src/shaders/focus.vs");
	}
	catch (GLSLProgramException e) {
		std::cout << "loading vertex shader failed." << std::endl;
		std::cout << e.what() << std::endl;

	}
	try {
		geometryPassProgram.compileShader("./src/shaders/GeometryPass.fs");
		secondPassProgram.compileShader("./src/shaders/SecondPass.fs");
		edgeD.compileShader("./src/shaders/edgeD.fs");
		blur.compileShader("./src/shaders/blur.fs");
		focus.compileShader("./src/shaders/focus.fs");
	}
	catch (GLSLProgramException e) {
		std::cout << "loading fragment shader failed." << std::endl;
		std::cout << e.what() << std::endl;
	}
	geometryPassProgram.link();
	secondPassProgram.link();
	edgeD.link();
	blur.link();
	focus.link();

	//quad
	quad = new Quad();

	// init objects in the scene
	glm::mat4 modelMatrix;

	std::string filename = "../data/Balloon.obj";
	m_pMeshList.push_back(new VBOMesh(filename.c_str(), false, true, true));
	modelMatrix = glm::translate(glm::mat4(1.0), glm::vec3(-11, 4, 20 - zFar));
	m_modelList.push_back(modelMatrix);
	m_colors.push_back(vec3(0.815, 0.211, 0.270));

	m_pMeshList.push_back(new VBOMesh(filename.c_str(), false, true, true));
	modelMatrix = glm::translate(glm::mat4(1.0), glm::vec3(1, -2, 150 - zFar));
	m_modelList.push_back(modelMatrix);
	m_colors.push_back(vec3(0.972, 0.662, 0.611));

	m_pMeshList.push_back(new VBOMesh(filename.c_str(), false, true, true));
	modelMatrix = glm::translate(glm::mat4(1.0), glm::vec3(-3.5, -5, 150 - zFar));
	m_modelList.push_back(modelMatrix);
	m_colors.push_back(vec3(0.890, 0.266, 0.294));

	m_pMeshList.push_back(new VBOMesh(filename.c_str(), false, true, true));
	modelMatrix = glm::translate(glm::mat4(1.0), glm::vec3(-8, -6, 150 - zFar));
	m_modelList.push_back(modelMatrix);
	m_colors.push_back(vec3(0.662, 0.552, 0.717));

	m_pMeshList.push_back(new VBOMesh(filename.c_str(), false, true, true));
	modelMatrix = glm::translate(glm::mat4(1.0), glm::vec3(8, 5, 150 - zFar));
	m_modelList.push_back(modelMatrix);
	m_colors.push_back(vec3(0.580, 0.788, 0.313));

	m_pMeshList.push_back(new VBOMesh(filename.c_str(), false, true, true));
	modelMatrix = glm::translate(glm::mat4(1.0), glm::vec3(8, -2, 200 - zFar));
	m_modelList.push_back(modelMatrix);
	m_colors.push_back(vec3(0.101, 0.725, 0.890));

	m_pMeshList.push_back(new VBOMesh(filename.c_str(), false, true, true));
	modelMatrix = glm::translate(glm::mat4(1.0), glm::vec3(3, -6, 190 - zFar));
	m_modelList.push_back(modelMatrix);
	m_colors.push_back(vec3(0.992, 0.780, 0.290));

	m_pMeshList.push_back(new VBOMesh(filename.c_str(), false, true, true));
	modelMatrix = glm::translate(glm::mat4(1.0), glm::vec3(-6, -3, 190 - zFar));
	m_modelList.push_back(modelMatrix);
	m_colors.push_back(vec3(0.803, 0.568, 0.305));

	m_pMeshList.push_back(new VBOMesh(filename.c_str(), false, true, true));
	modelMatrix = glm::translate(glm::mat4(1.0), glm::vec3(-0, -4, 180 - zFar));
	m_modelList.push_back(modelMatrix);
	m_colors.push_back(vec3(0.737, 0.843, 0.345));

	m_pMeshList.push_back(new VBOMesh(filename.c_str(), false, true, true));
	modelMatrix = glm::translate(glm::mat4(1.0), glm::vec3(6, -1, 250 - zFar));
	m_modelList.push_back(modelMatrix);
	m_colors.push_back(vec3(0.509, 0, 0.254));

	m_pMeshList.push_back(new VBOMesh(filename.c_str(), false, true, true));
	modelMatrix = glm::translate(glm::mat4(1.0), glm::vec3(-2, -2, 230 - zFar));
	m_modelList.push_back(modelMatrix);
	m_colors.push_back(vec3(1, 0.803, 0.372));

	m_pMeshList.push_back(new VBOMesh(filename.c_str(), false, true, true));
	modelMatrix = glm::translate(glm::mat4(1.0), glm::vec3(-3, -8, 280 - zFar));
	m_modelList.push_back(modelMatrix);
	m_colors.push_back(vec3(0.380, 0.298, 0.619));

	m_pMeshList.push_back(new VBOMesh(filename.c_str(), false, true, true));
	modelMatrix = glm::translate(glm::mat4(1.0), glm::vec3(3, -3, 300 - zFar));
	m_modelList.push_back(modelMatrix);
	m_colors.push_back(vec3(0.376, 0.576, 0.796));

	m_pMeshList.push_back(new VBOMesh(filename.c_str(), false, true, true));
	modelMatrix = glm::translate(glm::mat4(1.0), glm::vec3(-0.5, -7, 300 - zFar));
	m_modelList.push_back(modelMatrix);
	m_colors.push_back(vec3(0.462, 0.294, 0.219));

	m_pMeshList.push_back(new VBOMesh(filename.c_str(), false, true, true));
	modelMatrix = glm::translate(glm::mat4(1.0), glm::vec3(-3, -1, 320 - zFar));
	m_modelList.push_back(modelMatrix);
	m_colors.push_back(vec3(0.862, 0.380, 0.345));

	m_pMeshList.push_back(new VBOMesh(filename.c_str(), false, true, true));
	modelMatrix = glm::translate(glm::mat4(1.0), glm::vec3(-4, -8, 350 - zFar));
	m_modelList.push_back(modelMatrix);
	m_colors.push_back(vec3(0.086, 0.850, 0.827));

	m_pMeshList.push_back(new VBOMesh(filename.c_str(), false, true, true));
	modelMatrix = glm::translate(glm::mat4(1.0), glm::vec3(1, -8, 370 - zFar));
	m_modelList.push_back(modelMatrix);
	m_colors.push_back(vec3(0.898, 0.415, 0.101));

	m_pMeshList.push_back(new VBOMesh(filename.c_str(), false, true, true));
	modelMatrix = glm::translate(glm::mat4(1.0), glm::vec3(0, -2, 390 - zFar));
	m_modelList.push_back(modelMatrix);
	m_colors.push_back(vec3(0.482, 0.694, 0.733));

	m_pMeshList.push_back(new VBOMesh(filename.c_str(), false, true, true));
	modelMatrix = glm::translate(glm::mat4(1.0), glm::vec3(-3, -1, 400 - zFar));
	m_modelList.push_back(modelMatrix);
	m_colors.push_back(vec3(0.843, 0.701, 0.415));

	m_pMeshList.push_back(new VBOMesh(filename.c_str(), false, true, true));
	modelMatrix = glm::translate(glm::mat4(1.0), glm::vec3(2, -6, 420 - zFar));
	m_modelList.push_back(modelMatrix);
	m_colors.push_back(vec3(0.972, 0.635, 0.254));

	m_pMeshList.push_back(new VBOMesh(filename.c_str(), false, true, true));
	modelMatrix = glm::translate(glm::mat4(1.0), glm::vec3(-1, -7.5, 460 - zFar));
	m_modelList.push_back(modelMatrix);
	m_colors.push_back(vec3(1, 0.803, 0.372));

}


// render a frame (two-pass rendering)
void renderFrame() {

	// TODO bind your framebuffer
	// glBindFramebuffer(GL_FRAMEBUFFER, ...); 
	// render off-screen 

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);
	
	glDepthFunc(GL_LESS);

	// TODO: The first render-pass starts here
	// TODO: use your program
	if (shaderType==0)
	{
		program.use();
		program.setUniform("projection", projection);
    	program.setUniform("view", view);
	}	
	else
		geometryPassProgram.use();


	// render geometry
	// TODO set uniforms (like projection matrix, view matrix, other parameters)
	//Transformation uniforms
	//program->setUniform("lightPos", lightPos);
	//program->setUniform("lightColor", lightColor);

	// geometryPassProgram.setUniform(...);
	// 1. geometry pass: render scene's geometry/color data into gbuffer
		// -----------------------------------------------------------------
	glBindFramebuffer(GL_FRAMEBUFFER, gBuffer);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	geometryPassProgram.setUniform("view", view);
	geometryPassProgram.setUniform("projection", projection);
	glm::mat4 model;
	glm::vec3 color;
	for (int i = 0; i < m_modelList.size(); i++) {
		model = m_modelList.at(i);
		color = m_colors.at(i);
		// TODO set uniforms that are different per object (i.e., color, model matrix)
		// geometryPassProgram.setUniform(...);

		if (shaderType==0)
		{
			program.setUniform("model", model);
     		program.setUniform("objectColor", color);
		}
		else
		{
			geometryPassProgram.setUniform("model", model);
			geometryPassProgram.setUniform("objectColor", color);
		}

		m_pMeshList.at(i)->render();
		
	}


	// second pass starts here
	// TODO: unbind frame buffer
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	//// render to screen 
	//// TODO: clear screen
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// TODO: use your program for the second pass

	//choose an option for the shader
	if (shaderType==1)
		secondPassProgram.use();
	else if (shaderType==2)
		edgeD.use();
	else if (shaderType==3)
		blur.use();
	else if (shaderType==4)
		focus.use();

	// TODO: set uniforms for second pass
	// secondPassProgram.setUniform(...);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, gPosition);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, gNormal);
	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, gAlbedoSpec);
	glActiveTexture(GL_TEXTURE3);
	glBindTexture(GL_TEXTURE_2D, gDepth);
	// send light relevant uniforms


	if (shaderType==1)
	{
		secondPassProgram.setUniform("gPosition", 0);
		secondPassProgram.setUniform("gNormal", 1);
		secondPassProgram.setUniform("gAlbedoSpec", 2);
		secondPassProgram.setUniform("gDepth", 3);
		secondPassProgram.setUniform("lightPos", lightPos);
		secondPassProgram.setUniform("lightColor", lightColor);
		secondPassProgram.setUniform("viewPos", viewPos);
		secondPassProgram.setUniform("near", zNear);
		secondPassProgram.setUniform("far", zFar);
		secondPassProgram.setUniform("kernelSize", kernelSize);
		secondPassProgram.setUniform("myInt", myInt);
	}
	else if (shaderType==2)
	{
		edgeD.setUniform("gPosition", 0);
		edgeD.setUniform("gNormal", 1);
		edgeD.setUniform("gAlbedoSpec", 2);
		edgeD.setUniform("gDepth", 3);
		edgeD.setUniform("lightPos", lightPos);
		edgeD.setUniform("lightColor", lightColor);
		edgeD.setUniform("viewPos", viewPos);
		edgeD.setUniform("near", zNear);
		edgeD.setUniform("far", zFar);
		edgeD.setUniform("kernelSize", kernelSize);
		edgeD.setUniform("myInt", myInt);
	}	
	else if (shaderType==3)
	{
		blur.setUniform("gPosition", 0);
		blur.setUniform("gNormal", 1);
		blur.setUniform("gAlbedoSpec", 2);
		blur.setUniform("gDepth", 3);
		blur.setUniform("lightPos", lightPos);
		blur.setUniform("lightColor", lightColor);
		blur.setUniform("viewPos", viewPos);
		blur.setUniform("near", zNear);
		blur.setUniform("far", zFar);
		blur.setUniform("kernelSize", kernelSize);
		blur.setUniform("myInt", myInt);
	}
	else if (shaderType==4)
	{
		focus.setUniform("gPosition", 0);
		focus.setUniform("gNormal", 1);
		focus.setUniform("gAlbedoSpec", 2);
		focus.setUniform("gDepth", 3);
		focus.setUniform("lightPos", lightPos);
		focus.setUniform("lightColor", lightColor);
		focus.setUniform("viewPos", viewPos);
		focus.setUniform("near", zNear);
		focus.setUniform("far", zFar);
		focus.setUniform("Plane", z);
	}
	
	//secondPassProgram->setUniform("objectColor", color);

	// TODO: set active texture units and bind textures
	// TODO: render a screen-filling quad
	// Hint: You can use the class vboquad for that
	quad->render();

}


// TODO: Set up off-screen framebuffers 
void generateFrameBuffer() {
	// TODO: generate and bind frame buffer (must be same size as the window, all following buffers must also be of the same size)
	// TODO: generate buffer for positions and use for framebuffer
	// TODO: generate buffer for normals and use for framebuffer
	// TODO: generate buffer for depth and use for framebuffer
	// TODO: generate other buffers if you need 

	glGenFramebuffers(1, &gBuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, gBuffer);

	// position color buffer
	glGenTextures(1, &gPosition);
	glBindTexture(GL_TEXTURE_2D, gPosition);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, gWindowWidth, gWindowHeight, 0, GL_RGBA, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gPosition, 0);
	// normal color buffer
	glGenTextures(1, &gNormal);
	glBindTexture(GL_TEXTURE_2D, gNormal);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, gWindowWidth, gWindowHeight, 0, GL_RGBA, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, gNormal, 0);
	// color + specular color buffer
	glGenTextures(1, &gAlbedoSpec);
	glBindTexture(GL_TEXTURE_2D, gAlbedoSpec);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, gWindowWidth, gWindowHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, gAlbedoSpec, 0);
	//depth
	glGenTextures(1, &gDepth);
	glBindTexture(GL_TEXTURE_2D, gDepth);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, gWindowWidth, gWindowHeight, 0, GL_RGBA, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_TEXTURE_2D, gDepth, 0);

	// tell OpenGL which color attachments we'll use (of this framebuffer) for rendering 
	unsigned int attachments[4] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3 };
	glDrawBuffers(4, attachments);
	// create and attach depth buffer (renderbuffer)
	unsigned int rboDepth;
	glGenRenderbuffers(1, &rboDepth);
	glBindRenderbuffer(GL_RENDERBUFFER, rboDepth);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, gWindowWidth, gWindowHeight);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rboDepth);
	// finally check if framebuffer is complete
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		std::cout << "Framebuffer not complete!" << std::endl;

	// TODO: tell OpenGL which color attachments we'll use (of this framebuffer) for rendering 
	// unsigned int attachments[4] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3 };
	// glDrawBuffers(4, attachments);

	// TODO: finally check if framebuffer is complete
	//if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
	//{ std::cout << "Framebuffer not complete!" << std::endl; }

	// unbind framebuffer
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}


// entry point
int main(int argc, char** argv)
{

	// set glfw error callback
	glfwSetErrorCallback(glfwErrorCallback);

	// init glfw
	if (!glfwInit()) {
		exit(EXIT_FAILURE);
	}

	// init glfw window 
	GLFWwindow* window;
	window = glfwCreateWindow(gWindowWidth, gWindowHeight, "CS380 - OpenGL and GLSL Shaders", nullptr, nullptr);
	if (!window)
	{
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	glfwSetKeyCallback(window, key_callback);

	// make context current (once is sufficient)
	glfwMakeContextCurrent(window);

	// get the frame buffer size
	int width, height;
	glfwGetFramebufferSize(window, &width, &height);

	// init the OpenGL API (we need to do this once before any calls to the OpenGL API)
	gladLoadGL();

	// init our application
	if (!initApplication(argc, argv)) {
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	// setting up our 3D scene
	setupScene();

	// setting up frame buffers
	generateFrameBuffer();


	// start traversing the main loop
	// loop until the user closes the window 
	while (!glfwWindowShouldClose(window))
	{
		// render one frame  
		renderFrame();

		// swap front and back buffers 
		glfwSwapBuffers(window);

		// poll and process input events (keyboard, mouse, window, ...)
		glfwPollEvents();
	}

	glfwTerminate();
	return EXIT_SUCCESS;
}