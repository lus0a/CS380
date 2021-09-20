#ifndef VBODISC_H
#define VBODISC_H

class VBODisc
{

private:
    unsigned int vaoHandle;
    int faces;

public:
    VBODisc(float , float , int );

    void render();
};

#endif // VBODISC_H
