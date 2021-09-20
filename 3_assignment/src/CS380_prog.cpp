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
VBOCube *m_pCube;

// The Balloon Mesh
std::vector<VBOMesh*> m_pMeshList;
std::vector<glm::mat4> m_modelList;
std::vector<glm::vec3> m_colors;

// Plane Settings
float zFar = 500.0f;
float zNear = 0.1f;
float fov = 190.0f;

// TODO: define glsl programms
//GLSLProgram geometryPassProgram;
//GLSLProgram secondPassProgram;


// glfw error callback
void glfwErrorCallback(int error, const char* description)
{
	fprintf(stderr, "Error: %s\n", description);
}

// OpenGL error debugging callback
void APIENTRY glDebugOutput(GLenum source,
	GLenum type,
	GLuint id,
	GLenum severity,
	GLsizei length,
	const GLchar *message,
	const void *userParam)
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
bool initApplication(int argc, char **argv)
{
	
	std::string version((const char *)glGetString(GL_VERSION));
	std::stringstream stream(version);
	unsigned major, minor;
	char dot;

	stream >> major >> dot >> minor;
	
	assert(dot == '.');
	if (major > 3 || (major == 2 && minor >= 0)) {
		std::cout << "OpenGL Version " << major << "." << minor << std::endl;
	} else {
		std::cout << "The minimum required OpenGL version is not supported on this machine. Supported is only " << major << "." << minor << std::endl;
		return false;
	}

	// default initialization
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glDisable(GL_DEPTH_TEST);

	// viewport
	glViewport(0, 0, gWindowWidth, gWindowHeight);

	// projection
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
		
	return true;
}

// one time initialization to setup the 3D scene
void setupScene()
{
	// TODO: Set up a camera
	/* copy your solution from last assignment	
	*/
	 
	// TODO: Set up a light
	/* define at least one light and use later as uniform(s)
	vec4 lightPosition = vec4(...);
	*/

	// TODO: Set up glsl programs (you need a vertex shaders and a fragment shaders for each pass --> 2 vertex and 2 fragment shaders) 
	/* Use GLSLProgram class or your own solution
	
	
	try {
		geometryPassProgram.compileShader("./src/shaders/GeometryPass.vert");
		secondPassProgram.compileShader("./src/shaders/SecondPass.vert");
	} catch (GLSLProgramException e) {
		std::cout << "loading vertex shader failed." << std::endl;
		std::cout << e.what() << std::endl;

	} 
	try {
		geometryPassProgram.compileShader("./src/shaders/GeometryPass.frag");
		secondPassProgram.compileShader("./src/shaders/SecondPass.frag");
	} catch (GLSLProgramException e) {
		std::cout << "loading fragment shader failed." << std::endl;
		std::cout << e.what() << std::endl;
	}
	geometryPassProgram.link();
	secondPassProgram.link();*/


	
	// init objects in the scene
	glm::mat4 modelMatrix;

	std::string filename = "../data/Balloon.obj";
	m_pMeshList.push_back(new VBOMesh(filename.c_str(), false, true, true));
	modelMatrix = glm::translate(glm::mat4(1.0), glm::vec3(-11, 4, 20 - zFar));
	m_modelList.push_back(modelMatrix);
	m_colors.push_back(vec3(0.815, 0.211, 0.270));

	m_pMeshList.push_back(new VBOMesh(filename.c_str(), false, true, true));
	modelMatrix = glm::translate(glm::mat4(1.0), glm::vec3(1, -2, 150 - zFar ));
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
	modelMatrix = glm::translate(glm::mat4(1.0), glm::vec3(-3, -1, 400  - zFar));
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
void renderFrame(){
	
	// TODO bind your framebuffer
	// glBindFramebuffer(GL_FRAMEBUFFER, ...); 
	// render off-screen 

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	
	// TODO: The first render-pass starts here
	// TODO: use your program
	// geometryPassProgram.use();

	// render geometry
	// TODO set uniforms (like projection matrix, view matrix, other parameters)
	// geometryPassProgram.setUniform(...);
	glm::mat4 model;
	glm::vec3 color;
	for (int i = 0; i < m_modelList.size(); i++) {
		model = m_modelList.at(i);
		color = m_colors.at(i);
		// TODO set uniforms that are different per object (i.e., color, model matrix)
		// geometryPassProgram.setUniform(...);
		m_pMeshList.at(i)->render();
	}

	// second pass starts here
	// TODO: unbind frame buffer
	// glBindFramebuffer(GL_FRAMEBUFFER, 0); 
	// render to screen 
	// TODO: clear screen
	// glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// TODO: use your program for the second pass
	// secondPassProgram.use();
	// TODO: set uniforms for second pass
	// secondPassProgram.setUniform(...);

	// TODO: set active texture units and bind textures

	// TODO: render a screen-filling quad
	// Hint: You can use the class vboquad for that

}


// TODO: Set up off-screen framebuffers 
void generateFrameBuffer() {
	// TODO: generate and bind frame buffer (must be same size as the window, all following buffers must also be of the same size)
	// TODO: generate buffer for positions and use for framebuffer
	// TODO: generate buffer for normals and use for framebuffer
	// TODO: generate buffer for depth and use for framebuffer
	// TODO: generate other buffers if you need 
	
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


