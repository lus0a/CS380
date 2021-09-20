#ifndef VBOCYLINDER_H
#define VBOCYLINDER_H

class VBOCylinder
{

private:
    unsigned int vaoHandle;
    int faces;

public:
    VBOCylinder(float outRadius, float innerRadius, float H, int sampleb);

    void render();
};

#endif // VBOCYLINDER_H
