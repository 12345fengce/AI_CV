# include <stdio.h>

int main()
{
    int a;
    char b;
    float c;
    double d;

    a = 3;
    b = 'L';
    c = 3.14;
    d = 3.141592653;

    printf("I'll finish it in %d days!\n", a);
    printf("my signal is %c!\n", b);
    printf("pi is %.2f!\n", c);
    printf("pi is %.9f, more precision!\n", d);

    return 0;
}