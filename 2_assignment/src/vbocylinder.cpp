#include "vbocylinder.h"

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

VBODisc::VBOCylinder(float outerRadius, float innerRadius, float H, int sampleb):
{
    int nVert = 4*sampleb;
    double beta = TWOPI/sampleb;
    float T2 = Thickness/2;
    float *v = new float[4*3*nVert];
    float *n = new float[4*3*nVert];
    float *tex = new float[4*2*nVert];
    faces = sampleb;
    unsigned int *el = new unsigned int[4*6*face];
    
    //top
    for (int i = 0; i<sampleb; i++)
    {
        v[12*i] = innerRadius*cos(i*beta);
        v[12*i+1] = innerRadius*sin(i*beta);
        v[12*i+2] = T2;

        v[12*i+3] = outerRadius*cos(i*beta);
        v[12*i+4] = outerRadius*sin(i*beta);
        v[12*i+5] = T2;

        v[12*i+6] = outerRadius*cos((i+1)*beta);
        v[12*i+7] = outerRadius*sin((i+1)*beta);
        v[12*i+8] = T2;

        v[12*i+9] = innerRadius*cos((i+1)*beta);
        v[12*i+10] = innerRadius*sin((i+1)*beta);
        v[12*i+11] = T2;

        n[12*i] = 0.0f;
        n[12*i+1] = 0.0f;
        n[12*i+2] = 1.0f;

        n[12*i+3] = 0.0f;
        n[12*i+4] = 0.0f;
        n[12*i+5] = 1.0f;

        n[12*i+6] = 0.0f;
        n[12*i+7] = 0.0f;
        n[12*i+8] = 1.0f;

        n[12*i+9] = 0.0f;
        n[12*i+10] = 0.0f;
        n[12*i+11] = 1.0f;

        tex[8*i] = 0.0f;
        tex[8*i+1] = 0.0f;
        tex[8*i+2] = 0.0f;
        tex[8*i+3] = 1.0f;
        tex[8*i+4] = 1.0f;
        tex[8*i+5] = 1.0f;
        tex[8*i+6] = 1.0f;
        tex[8*i+7] = 0.0f;

        el[6*i] = 4*i;
        el[6*i+1] = 4*i+1;
        el[6*i+2] = 4*i+2;
        el[6*i+3] = 4*i;
        el[6*i+4] = 4*i+2;
        el[6*i+5] = 4*i+3;
    }
    
    //bot
    for (int i = 0; i<sampleb; i++)
    {
        v[12*sampleb+12*i] = innerRadius*cos(i*beta);
        v[12*sampleb+12*i+1] = innerRadius*sin(i*beta);
        v[12*sampleb+12*i+2] = -T2;

        v[12*sampleb+12*i+3] = outerRadius*cos(i*beta);
        v[12*sampleb+12*i+4] = outerRadius*sin(i*beta);
        v[12*sampleb+12*i+5] = -T2;

        v[12*sampleb+12*i+6] = outerRadius*cos((i+1)*beta);
        v[12*sampleb+12*i+7] = outerRadius*sin((i+1)*beta);
        v[12*sampleb+12*i+8] = -T2;

        v[12*sampleb+12*i+9] = innerRadius*cos((i+1)*beta);
        v[12*sampleb+12*i+10] = innerRadius*sin((i+1)*beta);
        v[12*sampleb+12*i+11] = -T2;

        n[12*sampleb+12*i] = 0.0f;
        n[12*sampleb+12*i+1] = 0.0f;
        n[12*sampleb+12*i+2] = -1.0f;

        n[12*sampleb+12*i+3] = 0.0f;
        n[12*sampleb+12*i+4] = 0.0f;
        n[12*sampleb+12*i+5] = -1.0f;

        n[12*sampleb+12*i+6] = 0.0f;
        n[12*sampleb+12*i+7] = 0.0f;
        n[12*sampleb+12*i+8] = -1.0f;

        n[12*sampleb+12*i+9] = 0.0f;
        n[12*sampleb+12*i+10] = 0.0f;
        n[12*sampleb+12*i+11] = -1.0f;

        tex[8*sampleb+8*i] = 0.0f;
        tex[8*sampleb+8*i+1] = 0.0f;
        tex[8*sampleb+8*i+2] = 0.0f;
        tex[8*sampleb+8*i+3] = 1.0f;
        tex[8*sampleb+8*i+4] = 1.0f;
        tex[8*sampleb+8*i+5] = 1.0f;
        tex[8*sampleb+8*i+6] = 1.0f;
        tex[8*sampleb+8*i+7] = 0.0f;

        el[6*sampleb+6*i] = 4*sampleb+4*i;
        el[6*sampleb+6*i+1] = 4*sampleb+4*i+1;
        el[6*sampleb+6*i+2] = 4*sampleb+4*i+2;
        el[6*sampleb+6*i+3] = 4*sampleb+4*i;
        el[6*sampleb+6*i+4] = 4*sampleb+4*i+2;
        el[6*sampleb+6*i+5] = 4*sampleb+4*i+3;
    }
    
    
    //outer
    for (int i = 0; i<sampleb; i++)
    {
        v[2*12*sampleb+12*i] = outerRadius*cos(i*beta);
        v[2*12*sampleb+12*i+1] = outerRadius*sin(i*beta);
        v[2*12*sampleb+12*i+2] = T2;

        v[2*12*sampleb+12*i+3] = outerRadius*cos(i*beta);
        v[2*12*sampleb+12*i+4] = outerRadius*sin(i*beta);
        v[2*12*sampleb+12*i+5] = -T2;

        v[2*12*sampleb+12*i+6] = outerRadius*cos((i+1)*beta);
        v[2*12*sampleb+12*i+7] = outerRadius*sin((i+1)*beta);
        v[2*12*sampleb+12*i+8] = -T2;

        v[2*12*sampleb+12*i+9] = outerRadius*cos((i+1)*beta);
        v[2*12*sampleb+12*i+10] = outerRadius*sin((i+1)*beta);
        v[2*12*sampleb+12*i+11] = T2;

        n[2*12*sampleb+12*i] = cos(i*beta);
        n[2*12*sampleb+12*i+1] = sin(i*beta);
        n[2*12*sampleb+12*i+2] = 0.0f;

        n[2*12*sampleb+12*i+3] = cos(i*beta);
        n[2*12*sampleb+12*i+4] = sin(i*beta);
        n[2*12*sampleb+12*i+5] = 0.0f;

        n[2*12*sampleb+12*i+6] = cos((i+1)*beta);
        n[2*12*sampleb+12*i+7] = sin((i+1)*beta);
        n[2*12*sampleb+12*i+8] = 0.0f;

        n[2*12*sampleb+12*i+9] = cos((i+1)*beta);
        n[2*12*sampleb+12*i+10] = sin((i+1)*beta);
        n[2*12*sampleb+12*i+11] = 0.0f;

        tex[2*8*sampleb+8*i] = 0.0f;
        tex[2*8*sampleb+8*i+1] = 0.0f;
        tex[2*8*sampleb+8*i+2] = 0.0f;
        tex[2*8*sampleb+8*i+3] = 1.0f;
        tex[2*8*sampleb+8*i+4] = 1.0f;
        tex[2*8*sampleb+8*i+5] = 1.0f;
        tex[2*8*sampleb+8*i+6] = 1.0f;
        tex[2*8*sampleb+8*i+7] = 0.0f;

        el[2*6*sampleb+6*i] = 2*4*sampleb+4*i;
        el[2*6*sampleb+6*i+1] = 2*4*sampleb+4*i+1;
        el[2*6*sampleb+6*i+2] = 2*4*sampleb+4*i+2;
        el[2*6*sampleb+6*i+3] = 2*4*sampleb+4*i;
        el[2*6*sampleb+6*i+4] = 2*4*sampleb+4*i+2;
        el[2*6*sampleb+6*i+5] = 2*4*sampleb+4*i+3;
    }
    
    //inner
    for (int i = 0; i<sampleb; i++)
    {
        v[3*12*sampleb+12*i] = innerRadius*cos(i*beta);
        v[3*12*sampleb+12*i+1] = innerRadius*sin(i*beta);
        v[3*12*sampleb+12*i+2] = T2;

        v[3*12*sampleb+12*i+3] = innerRadius*cos(i*beta);
        v[3*12*sampleb+12*i+4] = innerRadius*sin(i*beta);
        v[3*12*sampleb+12*i+5] = -T2;

        v[3*12*sampleb+12*i+6] = innerRadius*cos((i+1)*beta);
        v[3*12*sampleb+12*i+7] = innerRadius*sin((i+1)*beta);
        v[3*12*sampleb+12*i+8] = -T2;

        v[3*12*sampleb+12*i+9] = innerRadius*cos((i+1)*beta);
        v[3*12*sampleb+12*i+10] = innerRadius*sin((i+1)*beta);
        v[3*12*sampleb+12*i+11] = T2;

        n[3*12*sampleb+12*i] = -cos(i*beta);
        n[3*12*sampleb+12*i+1] = -sin(i*beta);
        n[3*12*sampleb+12*i+2] = 0.0f;

        n[3*12*sampleb+12*i+3] = -cos(i*beta);
        n[3*12*sampleb+12*i+4] = -sin(i*beta);
        n[3*12*sampleb+12*i+5] = 0.0f;

        n[3*12*sampleb+12*i+6] = -cos((i+1)*beta);
        n[3*12*sampleb+12*i+7] = -sin((i+1)*beta);
        n[3*12*sampleb+12*i+8] = 0.0f;

        n[3*12*sampleb+12*i+9] = -cos((i+1)*beta);
        n[3*12*sampleb+12*i+10] = -sin((i+1)*beta);
        n[3*12*sampleb+12*i+11] = 0.0f;

        tex[3*8*sampleb+8*i] = 0.0f;
        tex[3*8*sampleb+8*i+1] = 0.0f;
        tex[3*8*sampleb+8*i+2] = 0.0f;
        tex[3*8*sampleb+8*i+3] = 1.0f;
        tex[3*8*sampleb+8*i+4] = 1.0f;
        tex[3*8*sampleb+8*i+5] = 1.0f;
        tex[3*8*sampleb+8*i+6] = 1.0f;
        tex[3*8*sampleb+8*i+7] = 0.0f;

        el[3*6*sampleb+6*i] = 3*4*sampleb+4*i;
        el[3*6*sampleb+6*i+1] = 3*4*sampleb+4*i+1;
        el[3*6*sampleb+6*i+2] = 3*4*sampleb+4*i+2;
        el[3*6*sampleb+6*i+3] = 3*4*sampleb+4*i;
        el[3*6*sampleb+6*i+4] = 3*4*sampleb+4*i+2;
        el[3*6*sampleb+6*i+5] = 3*4*sampleb+4*i+3;
    }

    // GLuint el[] = {
    //     0,1,2,0,2,3,
    //     4,5,6,4,6,7,
    //     8,9,10,8,10,11,
    //     12,13,14,12,14,15,
    //     16,17,18,16,18,19,
    //     20,21,22,20,22,23
    // };

    glGenVertexArrays( 1, &vaoHandle );
    glBindVertexArray(vaoHandle);

    unsigned int handle[4];
    glGenBuffers(4, handle);

    glBindBuffer(GL_ARRAY_BUFFER, handle[0]);
    glBufferData(GL_ARRAY_BUFFER, 4*nVert * 3 * sizeof(float), v, GL_STATIC_DRAW);
    glVertexAttribPointer( (GLuint)0, 3, GL_FLOAT, GL_FALSE, 0, ((GLubyte *)NULL + (0)) );
    glEnableVertexAttribArray(0);  // Vertex position

    glBindBuffer(GL_ARRAY_BUFFER, handle[1]);
    glBufferData(GL_ARRAY_BUFFER, 4*nVert * 3 * sizeof(float), n, GL_STATIC_DRAW);
    glVertexAttribPointer( (GLuint)1, 3, GL_FLOAT, GL_FALSE, 0, ((GLubyte *)NULL + (0)) );
    glEnableVertexAttribArray(1);  // Vertex normal

    glBindBuffer(GL_ARRAY_BUFFER, handle[2]);
    glBufferData(GL_ARRAY_BUFFER, 4*nVert * 2 * sizeof(float), tex, GL_STATIC_DRAW);
    glVertexAttribPointer( (GLuint)2, 2, GL_FLOAT, GL_FALSE, 0, ((GLubyte *)NULL + (0)) );
    glEnableVertexAttribArray(2);  // texture coords

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, handle[3]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 4*6*faces * sizeof(GLuint), el, GL_STATIC_DRAW);

    glBindVertexArray(0);
}

void VBOCylinder::render() {
    glBindVertexArray(vaoHandle);
    glDrawElements(GL_TRIANGLES, 4*6*faces, GL_UNSIGNED_INT, ((GLubyte *)NULL + (0)));
}
