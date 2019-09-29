# include "stdio.h"

int main()
{
    signed int a;
    unsigned int b;
    signed int c;
    unsigned int d;

    a = -1;
    b = -1;
    c = +1;
    d = +1;

    printf("signed got %d \n", a);
    printf("unsigned got %d\n", b);
    printf("signed got %d\n", c);
    printf("unsigned got %d", d);

    return 0;

}