#ifndef VBOSPHERE_H
#define VBOSPHERE_H

class VBOSphere
{

private:
    unsigned int vaoHandle;
    int faces;

public:
    VBOSphere(float , int , int );

    void render();
};

#endif // VBOSPHERE_H
