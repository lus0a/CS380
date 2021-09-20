#ifndef VBODISC_H
#define VBODISC_H

class VBODisc
{

private:
    unsigned int vaoHandle;
    int faces;

public:
    VBODisc(float Radius, float Thickness, int sampleb);

    void render();
};

#endif // VBODISC_H
