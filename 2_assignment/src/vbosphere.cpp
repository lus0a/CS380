#include "vbosphere.h"

#define PI 3.141592653589793
#define TWOPI 6.2831853071795862
#define TWOPI_F 6.2831853f
#define TO_RADIANS(x) (x * 0.017453292519943295)
#define TO_DEGREES(x) (x * 57.29577951308232)


#include "glad/glad.h" 

// include glfw library: http://www.glfw.org/
#include <GLFW/glfw3.h>

//#include "glutils.h"

#include <cstdio>
#include <cmath>

VBOSphere::VBOSphere(float Radius, int samplea, int sampleb)
{
    int nVert = 4*samplea*sampleb;
    double alpha = TWOPI/samplea;
    double beta = TWOPI/sampleb;
    float *v = new float[3*nVert];
    float *n = new float[3*nVert];
    float *tex = new float[2*nVert];
    faces = samplea*sampleb;
    unsigned int *el = new unsigned int[6*faces];
    for (int a = 0; a<samplea; a++)
    {
        for (int i = 0; i<sampleb; i++)
        {
            v[12*sampleb*a+12*i] = Radius*cos(i*beta)*cos(a*alpha);
            v[12*sampleb*a+12*i+1] = Radius*sin(i*beta)*cos(a*alpha);
            v[12*sampleb*a+12*i+2] = Radius*sin(a*alpha);

            v[12*sampleb*a+12*i+3] = Radius*cos((i+1)*beta)*cos(a*alpha);
            v[12*sampleb*a+12*i+4] = Radius*sin((i+1)*beta)*cos(a*alpha);
            v[12*sampleb*a+12*i+5] = Radius*sin(a*alpha);

            v[12*sampleb*a+12*i+6] = Radius*cos((i+1)*beta)*cos((a+1)*alpha);
            v[12*sampleb*a+12*i+7] = Radius*sin((i+1)*beta)*cos((a+1)*alpha);
            v[12*sampleb*a+12*i+8] = Radius*sin((a+1)*alpha);

            v[12*sampleb*a+12*i+9] = Radius*cos(i*beta)*cos((a+1)*alpha);
            v[12*sampleb*a+12*i+10] = Radius*sin(i*beta)*cos((a+1)*alpha);
            v[12*sampleb*a+12*i+11] = Radius*sin((a+1)*alpha);


            n[12*sampleb*a+12*i] = cos(i*beta)*cos(a*alpha);
            n[12*sampleb*a+12*i+1] = sin(i*beta)*cos(a*alpha);
            n[12*sampleb*a+12*i+2] = sin(a*alpha);

            n[12*sampleb*a+12*i+3] = cos((i+1)*beta)*cos(a*alpha);
            n[12*sampleb*a+12*i+4] = sin((i+1)*beta)*cos(a*alpha);
            n[12*sampleb*a+12*i+5] = sin(a*alpha);

            n[12*sampleb*a+12*i+6] = cos((i+1)*beta)*cos((a+1)*alpha);
            n[12*sampleb*a+12*i+7] = sin((i+1)*beta)*cos((a+1)*alpha);
            n[12*sampleb*a+12*i+8] = sin((a+1)*alpha);


            n[12*sampleb*a+12*i+9] = cos(i*beta)*cos((a+1)*alpha);
            n[12*sampleb*a+12*i+10] = sin(i*beta)*cos((a+1)*alpha);
            n[12*sampleb*a+12*i+11] = sin((a+1)*alpha);

            tex[8*sampleb*a+8*i] = 0.0f;
            tex[8*sampleb*a+8*i+1] = 0.0f;
            tex[8*sampleb*a+8*i+2] = 0.0f;
            tex[8*sampleb*a+8*i+3] = 1.0f;
            tex[8*sampleb*a+8*i+4] = 1.0f;
            tex[8*sampleb*a+8*i+5] = 1.0f;
            tex[8*sampleb*a+8*i+6] = 1.0f;
            tex[8*sampleb*a+8*i+7] = 0.0f;

            el[6*sampleb*a+6*i] = 4*sampleb*a+4*i;
            el[6*sampleb*a+6*i+1] = 4*sampleb*a+4*i+1;
            el[6*sampleb*a+6*i+2] = 4*sampleb*a+4*i+2;
            el[6*sampleb*a+6*i+3] = 4*sampleb*a+4*i;
            el[6*sampleb*a+6*i+4] = 4*sampleb*a+4*i+2;
            el[6*sampleb*a+6*i+5] = 4*sampleb*a+4*i+3;
        }
    }

    glGenVertexArrays( 1, &vaoHandle );
    glBindVertexArray(vaoHandle);

    unsigned int handle[4];
    glGenBuffers(4, handle);

    glBindBuffer(GL_ARRAY_BUFFER, handle[0]);
    glBufferData(GL_ARRAY_BUFFER, nVert * 3 * sizeof(float), v, GL_STATIC_DRAW);
    glVertexAttribPointer( (GLuint)0, 3, GL_FLOAT, GL_FALSE, 0, ((GLubyte *)NULL + (0)) );
    glEnableVertexAttribArray(0);  // Vertex position

    glBindBuffer(GL_ARRAY_BUFFER, handle[1]);
    glBufferData(GL_ARRAY_BUFFER, nVert * 3 * sizeof(float), n, GL_STATIC_DRAW);
    glVertexAttribPointer( (GLuint)1, 3, GL_FLOAT, GL_FALSE, 0, ((GLubyte *)NULL + (0)) );
    glEnableVertexAttribArray(1);  // Vertex normal

    glBindBuffer(GL_ARRAY_BUFFER, handle[2]);
    glBufferData(GL_ARRAY_BUFFER, nVert * 2 * sizeof(float), tex, GL_STATIC_DRAW);
    glVertexAttribPointer( (GLuint)2, 2, GL_FLOAT, GL_FALSE, 0, ((GLubyte *)NULL + (0)) );
    glEnableVertexAttribArray(2);  // texture coords

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, handle[3]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 6*faces * sizeof(GLuint), el, GL_STATIC_DRAW);

    glBindVertexArray(0);
}

void VBOSphere::render() {
    glBindVertexArray(vaoHandle);
    glDrawElements(GL_TRIANGLES, 6*faces, GL_UNSIGNED_INT, ((GLubyte *)NULL + (0)));
}
